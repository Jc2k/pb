use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[cfg(target_os = "macos")]
use std::ffi::c_int;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::experts::{
    AggregateExpertTensor, EXPERT_SCALE_BIAS_DTYPE_BF16, EXPERT_SCALE_BIAS_DTYPE_F32,
    ExpertSourceTensor, expert_scale_bias_dtype_size,
};
use super::math::{q4_dequantize_rows_with_group_size, quantize_q4, softmax_in_place};
#[cfg(test)]
use super::math::{q4_fma_matvec_with_group_size, rms_norm_with_weight_in_place};
use super::metal::{
    MetalBatchProjectionInput, MetalGlmMlaAbsorbedAttentionInput, MetalGlmMlaFusedAttentionInput,
    MetalGlmMlaFusedAttentionOutput, MetalGlmMlaPostAttentionInput, MetalObjcId as ObjcId,
    MetalPostAttentionPrep,
};
use super::model_family::{
    QwenModelConfig, QwenMoeFamily, QwenMoeLayerKind, QwenNormWeightSemantics,
};
use super::runtime::MetalExecutionFacade;
use super::safetensors::{SafetensorShard, parse_safetensors_header};
use super::scheduler::{ScheduledRouterScoreProjectionCommand, ScheduledRoutingCommand};
use super::state::{
    FlashMoeRoutingOutputSource, FlashMoeRoutingOutputState, LinearAttentionLayout,
};
use super::text::{QwenTokenizer, TokenSampler, rerank_resident_lm_head_candidates};
#[cfg(test)]
use super::types::FlashMoeLayerKind;
use super::types::{
    ExpertQuantization, GROUP_SIZE, LINEAR_KEY_DIM, LINEAR_TOTAL_KEY, LINEAR_TOTAL_VALUE,
    LINEAR_VALUE_DIM,
};

#[cfg(test)]
fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}
use anyhow::{Context, Result, bail};

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: c_int = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: c_int = 111;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemv(
        order: c_int,
        trans_a: c_int,
        m: c_int,
        n: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        x: *const f32,
        inc_x: c_int,
        beta: f32,
        y: *mut f32,
        inc_y: c_int,
    );
}

pub(crate) const TENSOR_ALIGNMENT: u64 = 4096;
#[cfg(test)]
const DENSE_PROJECTION_TILE_BYTES: usize = 64 * 1024 * 1024;
const DENSE_DECODED_TILE_CACHE_BYTES: usize = 512 * 1024 * 1024;
const DENSE_Q4_FULL_DECODE_MAX_BYTES: usize = 256 * 1024 * 1024;

pub(super) const DENSE_Q4_MLX_FORMAT: &str = "dense-q4-affine-mlx-v1";
pub(super) const DENSE_Q4_COLIBRI_FORMAT: &str = "dense-q4-affine-colibri-import-v1";
pub(super) const DENSE_Q4_MXFP4_FORMAT: &str = "dense-q4-affine-mxfp4-import-v1";

pub(super) fn skip_flashmoe_runtime_tensor(canonical_tensor: &str) -> bool {
    canonical_tensor.starts_with("mtp.")
}

pub(super) fn is_q4_aux_tensor_name(canonical_tensor: &str) -> bool {
    canonical_tensor.ends_with(".scales")
        || canonical_tensor.ends_with(".biases")
        || canonical_tensor.ends_with(".weight.qs")
}

pub(super) fn q4_weight_name_for_aux(tensor: &str) -> String {
    tensor
        .strip_suffix(".scales")
        .or_else(|| tensor.strip_suffix(".biases"))
        .or_else(|| tensor.strip_suffix(".qs"))
        .map(|base| format!("{base}.weight"))
        .map(|name| name.replace(".weight.weight", ".weight"))
        .unwrap_or_else(|| tensor.to_string())
}

pub(super) fn q4_aux_tensor_name(weight: &str, suffix: &str) -> String {
    weight
        .strip_suffix(".weight")
        .map(|base| format!("{base}.{suffix}"))
        .unwrap_or_else(|| format!("{weight}.{suffix}"))
}

pub(super) fn logical_shape_for_mlx_q4(shape: &[usize]) -> Result<Vec<usize>> {
    let Some((last, prefix)) = shape.split_last() else {
        bail!("native dense q4 tensor has empty shape");
    };
    let cols = last
        .checked_mul(8)
        .context("native dense q4 logical column count overflow")?;
    let mut logical = prefix.to_vec();
    logical.push(cols);
    Ok(logical)
}

pub(super) fn dense_native_q4_sources(
    snapshot_dir: &Path,
    weight_map: &BTreeMap<String, String>,
    shard_cache: &mut BTreeMap<String, SafetensorShard>,
    tensor: &str,
    glm_shape: Option<&[usize]>,
) -> Result<Option<DenseQ4SourceRefs>> {
    let canonical_tensor = canonical_hf_tensor_name(tensor);
    if !canonical_tensor.ends_with(".weight") {
        return Ok(None);
    }
    let Some(weight_shard) = weight_map.get(tensor) else {
        return Ok(None);
    };
    if !shard_cache.contains_key(weight_shard) {
        let path = snapshot_dir.join(weight_shard);
        shard_cache.insert(weight_shard.clone(), parse_safetensors_header(&path)?);
    }
    let (weight_dtype, weight_shape, weight_offsets) = {
        let weight_info = shard_cache
            .get(weight_shard)
            .and_then(|shard| shard.tensors.get(tensor))
            .with_context(|| format!("tensor {tensor} missing from safetensors header"))?;
        (
            weight_info.dtype.clone(),
            weight_info.shape.clone(),
            weight_info.data_offsets,
        )
    };

    if weight_dtype.eq_ignore_ascii_case("U8")
        && let Some(logical_shape) = glm_shape
    {
        let scales_name = format!("{tensor}.qs");
        let Some(scales_shard) = weight_map.get(&scales_name) else {
            return Ok(None);
        };
        if !shard_cache.contains_key(scales_shard) {
            let path = snapshot_dir.join(scales_shard);
            shard_cache.insert(scales_shard.clone(), parse_safetensors_header(&path)?);
        }
        let scales_info = shard_cache
            .get(scales_shard)
            .and_then(|shard| shard.tensors.get(&scales_name))
            .with_context(|| format!("tensor {scales_name} missing from safetensors header"))?;
        if !scales_info
            .dtype
            .eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_F32)
        {
            bail!(
                "Colibri tensor {tensor} expects F32 .qs scales, found {}",
                scales_info.dtype
            );
        }
        let [rows, cols] = logical_shape else {
            bail!("Colibri tensor {tensor} logical shape must be a matrix, got {logical_shape:?}");
        };
        let weight_bytes = weight_offsets[1].saturating_sub(weight_offsets[0]) as usize;
        let q4_bytes = rows
            .checked_mul(cols.div_ceil(2))
            .context("Colibri q4 tensor byte count overflow")?;
        let int8_bytes = rows
            .checked_mul(*cols)
            .context("Colibri int8 tensor byte count overflow")?;
        let bits = if weight_bytes == q4_bytes {
            4
        } else if weight_bytes == int8_bytes {
            8
        } else {
            bail!(
                "Colibri tensor {tensor} has {weight_bytes} packed bytes for logical shape {logical_shape:?}; expected {q4_bytes} (int4) or {int8_bytes} (int8)"
            );
        };
        let scale_bytes =
            scales_info.data_offsets[1].saturating_sub(scales_info.data_offsets[0]) as usize;
        if !scale_bytes.is_multiple_of(4 * rows) {
            bail!(
                "Colibri tensor {tensor} has {scale_bytes} scale bytes, which is not F32 row-group data for {rows} rows"
            );
        }
        let source_groups_per_row = scale_bytes / 4 / rows;
        if source_groups_per_row == 0 || source_groups_per_row > *cols {
            bail!(
                "Colibri tensor {tensor} has invalid source groups per row {source_groups_per_row} for {cols} columns"
            );
        }
        let source_group_size = cols.div_ceil(source_groups_per_row);
        if cols.div_ceil(source_group_size) != source_groups_per_row {
            bail!(
                "Colibri tensor {tensor} scale count cannot resolve a contiguous source group size"
            );
        }
        return Ok(Some(DenseQ4SourceRefs {
            scales_shard: scales_shard.clone(),
            scales_offsets: scales_info.data_offsets,
            biases_shard: scales_shard.clone(),
            biases_offsets: scales_info.data_offsets,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            source_format: if bits == 4 {
                DenseQ4SourceFormat::ColibriInt4
            } else {
                DenseQ4SourceFormat::ColibriInt8
            },
            source_group_size: Some(source_group_size),
        }));
    }

    if !weight_dtype.eq_ignore_ascii_case("U32") {
        return Ok(None);
    }

    let scales_name = q4_aux_tensor_name(tensor, "scales");
    let Some(scales_shard) = weight_map.get(&scales_name) else {
        return Ok(None);
    };
    if !shard_cache.contains_key(scales_shard) {
        let path = snapshot_dir.join(scales_shard);
        shard_cache.insert(scales_shard.clone(), parse_safetensors_header(&path)?);
    }
    let (scales_dtype, scales_offsets) = {
        let scales_info = shard_cache
            .get(scales_shard)
            .and_then(|shard| shard.tensors.get(&scales_name))
            .with_context(|| format!("tensor {scales_name} missing from safetensors header"))?;
        (scales_info.dtype.clone(), scales_info.data_offsets)
    };
    let logical_shape = logical_shape_for_mlx_q4(&weight_shape)?;

    if scales_dtype.eq_ignore_ascii_case("U8") {
        let (cols, prefix) = logical_shape
            .split_last()
            .with_context(|| format!("MLX MXFP4 tensor {tensor} has an empty logical shape"))?;
        let rows = prefix.iter().try_fold(1usize, |rows, dimension| {
            rows.checked_mul(*dimension)
                .context("MLX MXFP4 logical row count overflow")
        })?;
        let source_group_size = 32;
        let expected_weight_bytes = rows
            .checked_mul(cols.div_ceil(2))
            .context("MLX MXFP4 packed byte count overflow")?;
        let expected_scale_bytes = rows
            .checked_mul(cols.div_ceil(source_group_size))
            .context("MLX MXFP4 scale byte count overflow")?;
        let weight_bytes = usize::try_from(weight_offsets[1] - weight_offsets[0])
            .context("MLX MXFP4 packed byte count exceeds usize")?;
        let scale_bytes = usize::try_from(scales_offsets[1] - scales_offsets[0])
            .context("MLX MXFP4 scale byte count exceeds usize")?;
        if weight_bytes != expected_weight_bytes || scale_bytes != expected_scale_bytes {
            bail!(
                "MLX MXFP4 tensor {tensor} layout mismatch: weight/scales bytes {weight_bytes}/{scale_bytes}, expected {expected_weight_bytes}/{expected_scale_bytes} for {logical_shape:?}"
            );
        }
        return Ok(Some(DenseQ4SourceRefs {
            scales_shard: scales_shard.clone(),
            scales_offsets,
            biases_shard: scales_shard.clone(),
            biases_offsets: scales_offsets,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            source_format: DenseQ4SourceFormat::MlxMxfp4,
            source_group_size: Some(source_group_size),
        }));
    }

    let biases_name = q4_aux_tensor_name(tensor, "biases");
    let Some(biases_shard) = weight_map.get(&biases_name) else {
        return Ok(None);
    };
    for shard in [biases_shard] {
        if !shard_cache.contains_key(shard) {
            let path = snapshot_dir.join(shard);
            shard_cache.insert(shard.clone(), parse_safetensors_header(&path)?);
        }
    }
    let biases_info = shard_cache
        .get(biases_shard)
        .and_then(|shard| shard.tensors.get(&biases_name))
        .with_context(|| format!("tensor {biases_name} missing from safetensors header"))?;
    if !scales_dtype.eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16)
        || !biases_info
            .dtype
            .eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16)
    {
        bail!(
            "native dense q4 tensor {tensor} expects BF16 scales/biases, found {}/{}",
            scales_dtype,
            biases_info.dtype
        );
    }
    let layout = dense_q4_layout_with_scale_bias_dtype(
        &logical_shape,
        GROUP_SIZE,
        EXPERT_SCALE_BIAS_DTYPE_BF16,
    )?;
    let weight_bytes = weight_offsets[1].saturating_sub(weight_offsets[0]);
    let scales_bytes = scales_offsets[1].saturating_sub(scales_offsets[0]);
    let biases_bytes = biases_info.data_offsets[1].saturating_sub(biases_info.data_offsets[0]);
    if weight_bytes != layout.packed_bytes as u64
        || scales_bytes != layout.scales_bytes as u64
        || biases_bytes != layout.scales_bytes as u64
    {
        bail!(
            "native dense q4 tensor {tensor} layout mismatch: weight/scales/biases bytes {weight_bytes}/{scales_bytes}/{biases_bytes}, expected {}/{}/{}",
            layout.packed_bytes,
            layout.scales_bytes,
            layout.scales_bytes
        );
    }
    Ok(Some(DenseQ4SourceRefs {
        scales_shard: scales_shard.clone(),
        scales_offsets,
        biases_shard: biases_shard.clone(),
        biases_offsets: biases_info.data_offsets,
        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        source_format: DenseQ4SourceFormat::MlxAffine,
        source_group_size: None,
    }))
}

pub(super) fn dense_tensor_quantization(
    canonical_tensor: &str,
    tensor_dtype: &str,
    native_q4: &Option<DenseQ4SourceRefs>,
) -> TensorQuantization {
    let _ = (canonical_tensor, tensor_dtype);
    if native_q4
        .as_ref()
        .is_some_and(|source| source.source_format == DenseQ4SourceFormat::ColibriInt8)
    {
        // Colibri keeps the large embedding and LM-head matrices at int8 by
        // default. Preserve that source precision in a runtime layout the
        // existing resident dense kernels can consume instead of requantizing
        // the values to q4.
        TensorQuantization::None
    } else if let Some(native_q4) = native_q4 {
        TensorQuantization::Q4 {
            group_size: GROUP_SIZE,
            format: match native_q4.source_format {
                DenseQ4SourceFormat::MlxAffine => DENSE_Q4_MLX_FORMAT,
                DenseQ4SourceFormat::ColibriInt4 | DenseQ4SourceFormat::ColibriInt8 => {
                    DENSE_Q4_COLIBRI_FORMAT
                }
                DenseQ4SourceFormat::MlxMxfp4 => DENSE_Q4_MXFP4_FORMAT,
            }
            .to_string(),
            scale_bias_dtype: native_q4.scale_bias_dtype.clone(),
        }
    } else {
        TensorQuantization::None
    }
}

pub(super) fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

pub(super) fn write_dense_tensor_store(
    snapshot_dir: &Path,
    destination: &Path,
    dense_tensors: &[DenseTensorRef],
) -> Result<()> {
    let mut out = fs::File::create(destination).with_context(|| {
        format!(
            "failed to create dense tensor store {}",
            destination.display()
        )
    })?;
    let mut current = 0u64;
    let mut shard_cache = BTreeMap::<String, (memmap2::Mmap, SafetensorShard)>::new();
    for tensor in dense_tensors {
        if !shard_cache.contains_key(&tensor.shard) {
            let path = snapshot_dir.join(&tensor.shard);
            let file = fs::File::open(&path)
                .with_context(|| format!("failed to open shard {}", path.display()))?;
            let mmap = unsafe {
                memmap2::MmapOptions::new()
                    .map(&file)
                    .with_context(|| format!("failed to memory-map {}", path.display()))?
            };
            shard_cache.insert(
                tensor.shard.clone(),
                (mmap, parse_safetensors_header(&path)?),
            );
        }
        if current < tensor.runtime_offset {
            write_padding(&mut out, tensor.runtime_offset - current)?;
            current = tensor.runtime_offset;
        }
        if let Some(q4_sources) = &tensor.q4_sources {
            for shard in [&q4_sources.scales_shard, &q4_sources.biases_shard] {
                if !shard_cache.contains_key(shard) {
                    let path = snapshot_dir.join(shard);
                    let file = fs::File::open(&path)
                        .with_context(|| format!("failed to open shard {}", path.display()))?;
                    let mmap = unsafe {
                        memmap2::MmapOptions::new()
                            .map(&file)
                            .with_context(|| format!("failed to memory-map {}", path.display()))?
                    };
                    shard_cache.insert(shard.clone(), (mmap, parse_safetensors_header(&path)?));
                }
            }
        }
        let (bytes, shard) = shard_cache.get(&tensor.shard).expect("inserted above");
        let start = shard.data_start + tensor.source_offsets[0];
        let end = shard.data_start + tensor.source_offsets[1];
        let raw = &bytes[start as usize..end as usize];
        match &tensor.quantization {
            TensorQuantization::None => {
                if let Some(q4_sources) = &tensor.q4_sources
                    && q4_sources.source_format == DenseQ4SourceFormat::ColibriInt8
                {
                    let (scale_bytes, scale_shard) = shard_cache
                        .get(&q4_sources.scales_shard)
                        .expect("inserted above");
                    let scale_start = scale_shard.data_start + q4_sources.scales_offsets[0];
                    let scale_end = scale_shard.data_start + q4_sources.scales_offsets[1];
                    write_colibri_int8_bf16_tensor(
                        &mut out,
                        &tensor.tensor,
                        raw,
                        &scale_bytes[scale_start as usize..scale_end as usize],
                        q4_sources.source_group_size.with_context(|| {
                            format!(
                                "Colibri int8 tensor {} is missing its source group size",
                                tensor.tensor
                            )
                        })?,
                        &tensor.shape,
                    )?;
                } else {
                    out.write_all(raw).with_context(|| {
                        format!("failed to write dense tensor {}", tensor.tensor)
                    })?;
                }
            }
            TensorQuantization::Q4 {
                group_size,
                scale_bias_dtype,
                ..
            } => {
                let layout = dense_q4_layout_with_scale_bias_dtype(
                    &tensor.shape,
                    *group_size,
                    scale_bias_dtype,
                )?;
                if let Some(q4_sources) = &tensor.q4_sources {
                    if matches!(
                        q4_sources.source_format,
                        DenseQ4SourceFormat::ColibriInt4 | DenseQ4SourceFormat::ColibriInt8
                    ) {
                        let bits = if q4_sources.source_format == DenseQ4SourceFormat::ColibriInt4 {
                            4
                        } else {
                            8
                        };
                        let (scale_bytes, scale_shard) = shard_cache
                            .get(&q4_sources.scales_shard)
                            .expect("inserted above");
                        let scale_start = scale_shard.data_start + q4_sources.scales_offsets[0];
                        let scale_end = scale_shard.data_start + q4_sources.scales_offsets[1];
                        let source_scales = &scale_bytes[scale_start as usize..scale_end as usize];
                        write_colibri_q4_affine_tensor(
                            &mut out,
                            &tensor.tensor,
                            raw,
                            source_scales,
                            bits,
                            q4_sources.source_group_size.with_context(|| {
                                format!(
                                    "Colibri tensor {} is missing its source group size",
                                    tensor.tensor
                                )
                            })?,
                            layout,
                        )?;
                        current = current.saturating_add(tensor.byte_len);
                        continue;
                    }
                    if q4_sources.source_format == DenseQ4SourceFormat::MlxMxfp4 {
                        let (scale_bytes, scale_shard) = shard_cache
                            .get(&q4_sources.scales_shard)
                            .expect("inserted above");
                        let scale_start = scale_shard.data_start + q4_sources.scales_offsets[0];
                        let scale_end = scale_shard.data_start + q4_sources.scales_offsets[1];
                        write_mlx_mxfp4_affine_tensor(
                            &mut out,
                            &tensor.tensor,
                            raw,
                            &scale_bytes[scale_start as usize..scale_end as usize],
                            q4_sources.source_group_size.with_context(|| {
                                format!(
                                    "MLX MXFP4 tensor {} is missing its source group size",
                                    tensor.tensor
                                )
                            })?,
                            layout,
                        )?;
                        current = current.saturating_add(tensor.byte_len);
                        continue;
                    }
                    if raw.len() != layout.packed_bytes {
                        bail!(
                            "native dense q4 packed byte length mismatch for {}: raw={} expected={}",
                            tensor.tensor,
                            raw.len(),
                            layout.packed_bytes
                        );
                    }
                    out.write_all(raw).with_context(|| {
                        format!(
                            "failed to write native dense q4 packed values for {}",
                            tensor.tensor
                        )
                    })?;
                    let (scale_bytes, scale_shard) = shard_cache
                        .get(&q4_sources.scales_shard)
                        .expect("inserted above");
                    let scale_start = scale_shard.data_start + q4_sources.scales_offsets[0];
                    let scale_end = scale_shard.data_start + q4_sources.scales_offsets[1];
                    let scales = &scale_bytes[scale_start as usize..scale_end as usize];
                    if scales.len() != layout.scales_bytes {
                        bail!(
                            "native dense q4 scale byte length mismatch for {}: raw={} expected={}",
                            tensor.tensor,
                            scales.len(),
                            layout.scales_bytes
                        );
                    }
                    out.write_all(scales).with_context(|| {
                        format!(
                            "failed to write native dense q4 scales for {}",
                            tensor.tensor
                        )
                    })?;
                    let (bias_bytes, bias_shard) = shard_cache
                        .get(&q4_sources.biases_shard)
                        .expect("inserted above");
                    let bias_start = bias_shard.data_start + q4_sources.biases_offsets[0];
                    let bias_end = bias_shard.data_start + q4_sources.biases_offsets[1];
                    let biases = &bias_bytes[bias_start as usize..bias_end as usize];
                    if biases.len() != layout.scales_bytes {
                        bail!(
                            "native dense q4 bias byte length mismatch for {}: raw={} expected={}",
                            tensor.tensor,
                            biases.len(),
                            layout.scales_bytes
                        );
                    }
                    out.write_all(biases).with_context(|| {
                        format!(
                            "failed to write native dense q4 biases for {}",
                            tensor.tensor
                        )
                    })?;
                    current = current.saturating_add(tensor.byte_len);
                    continue;
                }
                if !scale_bias_dtype.eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_F32) {
                    bail!(
                        "post-hoc dense q4 quantization for {} only supports F32 scale/bias output, requested {}",
                        tensor.tensor,
                        scale_bias_dtype
                    );
                }
                let values = decode_dense_tensor_f32(&tensor.dtype, raw).with_context(|| {
                    format!(
                        "failed to decode dense tensor {} as {} before q4 quantization",
                        tensor.tensor, tensor.dtype
                    )
                })?;
                let packed =
                    quantize_q4(&values, &tensor.shape, *group_size).with_context(|| {
                        format!(
                            "failed to quantize dense tensor {} into q4 groups",
                            tensor.tensor
                        )
                    })?;
                if packed.values.len() != layout.packed_bytes
                    || packed.scales.len() != layout.scales_bytes / std::mem::size_of::<f32>()
                    || packed.biases.len() != layout.scales_bytes / std::mem::size_of::<f32>()
                    || tensor.byte_len as usize != layout.total_bytes
                {
                    bail!(
                        "dense q4 layout mismatch for {}: packed={} scales={} biases={} manifest_bytes={} computed_bytes={}",
                        tensor.tensor,
                        packed.values.len(),
                        packed.scales.len(),
                        packed.biases.len(),
                        tensor.byte_len,
                        layout.total_bytes
                    );
                }
                out.write_all(&packed.values).with_context(|| {
                    format!(
                        "failed to write dense q4 packed values for {}",
                        tensor.tensor
                    )
                })?;
                for scale in &packed.scales {
                    out.write_all(&scale.to_le_bytes())?;
                }
                for bias in &packed.biases {
                    out.write_all(&bias.to_le_bytes())?;
                }
            }
        }
        current = current.saturating_add(tensor.byte_len);
    }
    Ok(())
}

fn encode_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

fn decode_f32_le(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        bail!("F32 byte length {} is not divisible by four", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

fn write_colibri_int8_bf16_tensor(
    out: &mut impl Write,
    tensor_name: &str,
    source_weights: &[u8],
    source_scale_bytes: &[u8],
    source_group_size: usize,
    shape: &[usize],
) -> Result<()> {
    let [rows, cols] = shape else {
        bail!("Colibri int8 tensor {tensor_name} must be a matrix, got {shape:?}");
    };
    let expected_weights = rows
        .checked_mul(*cols)
        .context("Colibri int8 tensor element count overflow")?;
    if source_weights.len() != expected_weights {
        bail!(
            "Colibri int8 tensor {tensor_name} has {} source bytes, expected {expected_weights}",
            source_weights.len()
        );
    }
    let source_scales = decode_f32_le(source_scale_bytes)?;
    let groups_per_row = cols.div_ceil(source_group_size);
    let expected_scales = rows
        .checked_mul(groups_per_row)
        .context("Colibri int8 scale count overflow")?;
    if source_scales.len() != expected_scales {
        bail!(
            "Colibri int8 tensor {tensor_name} has {} scales, expected {expected_scales}",
            source_scales.len()
        );
    }
    for row in 0..*rows {
        for col in 0..*cols {
            let quantized = source_weights[row * *cols + col] as i8 as f32;
            let scale = source_scales[row * groups_per_row + col / source_group_size];
            out.write_all(&encode_bf16_bits(quantized * scale).to_le_bytes())?;
        }
    }
    Ok(())
}

pub(super) fn write_colibri_q4_affine_tensor(
    out: &mut impl Write,
    tensor_name: &str,
    source_weights: &[u8],
    source_scale_bytes: &[u8],
    bits: u8,
    source_group_size: usize,
    layout: DenseQ4Layout,
) -> Result<()> {
    if layout.scale_bias_bytes != 2 {
        bail!("Colibri import for {tensor_name} requires BF16 runtime scale/bias storage");
    }
    if bits != 4 && bits != 8 {
        bail!("Colibri tensor {tensor_name} has unsupported quantization width {bits}");
    }
    let source_scales = decode_f32_le(source_scale_bytes)?;
    let source_groups_per_row = layout.cols.div_ceil(source_group_size);
    let expected_source_scales = layout
        .rows
        .checked_mul(source_groups_per_row)
        .context("Colibri source scale count overflow")?;
    if source_scales.len() != expected_source_scales {
        bail!(
            "Colibri tensor {tensor_name} has {} source scales, expected {expected_source_scales}",
            source_scales.len()
        );
    }
    let source_row_bytes = match bits {
        4 => layout.cols.div_ceil(2),
        8 => layout.cols,
        _ => unreachable!(),
    };
    if source_weights.len() != layout.rows * source_row_bytes {
        bail!(
            "Colibri tensor {tensor_name} has {} weight bytes, expected {}",
            source_weights.len(),
            layout.rows * source_row_bytes
        );
    }

    let can_preserve_q4 = bits == 4
        && (0..layout.groups_per_row).all(|group| {
            let start = group * layout.group_size;
            let end = ((group + 1) * layout.group_size).min(layout.cols);
            start / source_group_size == end.saturating_sub(1) / source_group_size
        });
    let mut runtime_scales = Vec::with_capacity(layout.rows * layout.groups_per_row);
    let mut runtime_biases = Vec::with_capacity(layout.rows * layout.groups_per_row);

    if can_preserve_q4 {
        out.write_all(source_weights)
            .with_context(|| format!("failed to write Colibri packed q4 tensor {tensor_name}"))?;
        for row in 0..layout.rows {
            for group in 0..layout.groups_per_row {
                let source_group = (group * layout.group_size) / source_group_size;
                let scale = source_scales[row * source_groups_per_row + source_group];
                runtime_scales.push(encode_bf16_bits(scale));
                runtime_biases.push(encode_bf16_bits(-8.0 * scale));
            }
        }
    } else {
        let mut packed_row = vec![0u8; layout.row_packed_bytes];
        let mut values = vec![0.0f32; layout.cols];
        for row in 0..layout.rows {
            packed_row.fill(0);
            let source_row = &source_weights[row * source_row_bytes..(row + 1) * source_row_bytes];
            for (col, value) in values.iter_mut().enumerate() {
                let quantized = if bits == 8 {
                    source_row[col] as i8 as i32
                } else {
                    let byte = source_row[col / 2];
                    let nibble = if col.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    nibble as i32 - 8
                };
                let source_group = col / source_group_size;
                *value =
                    quantized as f32 * source_scales[row * source_groups_per_row + source_group];
            }
            for group in 0..layout.groups_per_row {
                let start = group * layout.group_size;
                let end = ((group + 1) * layout.group_size).min(layout.cols);
                let max_abs = values[start..end]
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0f32, f32::max);
                let scale = (max_abs / 7.0).max(1e-8);
                runtime_scales.push(encode_bf16_bits(scale));
                runtime_biases.push(encode_bf16_bits(-8.0 * scale));
                for (offset, value) in values[start..end].iter().enumerate() {
                    let col = start + offset;
                    let quantized = (value / scale).round().clamp(-8.0, 7.0) as i32 + 8;
                    if col.is_multiple_of(2) {
                        packed_row[col / 2] |= quantized as u8;
                    } else {
                        packed_row[col / 2] |= (quantized as u8) << 4;
                    }
                }
            }
            out.write_all(&packed_row).with_context(|| {
                format!("failed to write converted Colibri q4 row for {tensor_name}")
            })?;
        }
    }

    if runtime_scales.len() * 2 != layout.scales_bytes
        || runtime_biases.len() * 2 != layout.scales_bytes
    {
        bail!(
            "Colibri tensor {tensor_name} produced invalid affine scale/bias lengths {}/{}; expected {} bytes each",
            runtime_scales.len() * 2,
            runtime_biases.len() * 2,
            layout.scales_bytes
        );
    }
    for scale in runtime_scales {
        out.write_all(&scale.to_le_bytes())?;
    }
    for bias in runtime_biases {
        out.write_all(&bias.to_le_bytes())?;
    }
    Ok(())
}

fn decode_mlx_mxfp4_e2m1(nibble: u8) -> f32 {
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = MAGNITUDES[(nibble & 0x07) as usize];
    if nibble & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn decode_mlx_mxfp4_e8m0(bits: u8) -> Result<f32> {
    let ieee = if bits == 0 {
        0x0040_0000
    } else {
        u32::from(bits) << 23
    };
    let value = f32::from_bits(ieee);
    if !value.is_finite() {
        bail!("MLX MXFP4 E8M0 scale byte 0x{bits:02x} is not finite");
    }
    Ok(value)
}

pub(super) fn write_mlx_mxfp4_affine_tensor(
    out: &mut impl Write,
    tensor_name: &str,
    source_weights: &[u8],
    source_scale_bytes: &[u8],
    source_group_size: usize,
    layout: DenseQ4Layout,
) -> Result<()> {
    if layout.scale_bias_bytes != 2 {
        bail!("MLX MXFP4 import for {tensor_name} requires BF16 runtime scale/bias storage");
    }
    if source_group_size != 32 {
        bail!(
            "MLX MXFP4 tensor {tensor_name} requires source group size 32, found {source_group_size}"
        );
    }
    let source_row_bytes = layout.cols.div_ceil(2);
    let source_groups_per_row = layout.cols.div_ceil(source_group_size);
    let expected_weights = layout
        .rows
        .checked_mul(source_row_bytes)
        .context("MLX MXFP4 source weight byte count overflow")?;
    let expected_scales = layout
        .rows
        .checked_mul(source_groups_per_row)
        .context("MLX MXFP4 source scale byte count overflow")?;
    if source_weights.len() != expected_weights || source_scale_bytes.len() != expected_scales {
        bail!(
            "MLX MXFP4 tensor {tensor_name} has weight/scales bytes {}/{}, expected {expected_weights}/{expected_scales}",
            source_weights.len(),
            source_scale_bytes.len()
        );
    }

    let mut runtime_scales = Vec::with_capacity(layout.rows * layout.groups_per_row);
    let mut runtime_biases = Vec::with_capacity(layout.rows * layout.groups_per_row);
    let mut values = vec![0.0f32; layout.cols];
    for row in 0..layout.rows {
        let source_row = &source_weights[row * source_row_bytes..(row + 1) * source_row_bytes];
        let source_scales =
            &source_scale_bytes[row * source_groups_per_row..(row + 1) * source_groups_per_row];
        for (col, value) in values.iter_mut().enumerate() {
            let byte = source_row[col / 2];
            let nibble = if col.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            };
            let scale = decode_mlx_mxfp4_e8m0(source_scales[col / source_group_size])?;
            *value = decode_mlx_mxfp4_e2m1(nibble) * scale;
        }
        let quantized = quantize_q4(&values, &[1, layout.cols], layout.group_size)
            .with_context(|| format!("failed to requantize MLX MXFP4 tensor {tensor_name}"))?;
        if quantized.values.len() != layout.row_packed_bytes
            || quantized.scales.len() != layout.groups_per_row
            || quantized.biases.len() != layout.groups_per_row
        {
            bail!("MLX MXFP4 tensor {tensor_name} produced an invalid runtime q4 row layout");
        }
        out.write_all(&quantized.values).with_context(|| {
            format!("failed to write converted MLX MXFP4 q4 row for {tensor_name}")
        })?;
        runtime_scales.extend(quantized.scales.into_iter().map(encode_bf16_bits));
        runtime_biases.extend(quantized.biases.into_iter().map(encode_bf16_bits));
    }

    if runtime_scales.len() * 2 != layout.scales_bytes
        || runtime_biases.len() * 2 != layout.scales_bytes
    {
        bail!("MLX MXFP4 tensor {tensor_name} produced invalid affine scale/bias lengths");
    }
    for scale in runtime_scales {
        out.write_all(&scale.to_le_bytes())?;
    }
    for bias in runtime_biases {
        out.write_all(&bias.to_le_bytes())?;
    }
    Ok(())
}

pub(super) fn write_padding(out: &mut fs::File, mut bytes: u64) -> Result<()> {
    const ZEROES: [u8; 4096] = [0; 4096];
    while bytes > 0 {
        let n = usize::try_from(bytes.min(ZEROES.len() as u64)).unwrap_or(ZEROES.len());
        out.write_all(&ZEROES[..n])
            .context("failed to write tensor alignment padding")?;
        bytes -= n as u64;
    }
    Ok(())
}

pub(super) fn dense_f32_matvec_rows(
    weights: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Option<Vec<f32>>> {
    let expected = rows
        .checked_mul(cols)
        .context("dense f32 matvec row-major weight size overflow")?;
    if weights.len() < expected || input.len() < cols {
        return Ok(None);
    }

    #[cfg(target_os = "macos")]
    {
        let Ok(m) = c_int::try_from(rows) else {
            return Ok(None);
        };
        let Ok(n) = c_int::try_from(cols) else {
            return Ok(None);
        };
        let mut out = vec![0.0f32; rows];
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                m,
                n,
                1.0,
                weights.as_ptr(),
                n,
                input.as_ptr(),
                1,
                0.0,
                out.as_mut_ptr(),
                1,
            );
        }
        return Ok(Some(out));
    }

    #[cfg(not(target_os = "macos"))]
    {
        let mut out = vec![0.0f32; rows];
        for row in 0..rows {
            let start = row
                .checked_mul(cols)
                .context("dense f32 matvec row offset overflow")?;
            let weights = &weights[start..start + cols];
            out[row] = weights
                .iter()
                .zip(input.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
        }
        Ok(Some(out))
    }
}

pub(super) fn validate_required_tensor_manifest(
    config: &QwenModelConfig,
    registry: &TensorRegistry,
) -> Result<()> {
    require_tensor_shape(
        registry,
        "model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
    )?;
    require_tensor_shape(registry, "model.norm.weight", &[config.hidden_size])?;
    // lm_head.weight is optional: when absent (or when tie_word_embeddings is true) the model
    // uses tied embeddings and reuses model.embed_tokens.weight (already validated above) for
    // the output projection.
    if registry.tensor("lm_head.weight").is_some() {
        require_tensor_shape(
            registry,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
    }
    for layer in 0..config.num_hidden_layers {
        match infer_attention_layer_type(registry, layer)? {
            AttentionLayerType::Full => {
                let layout = infer_full_attention_layout(config, registry, layer)?;
                for projection in ["q_norm", "k_norm"] {
                    require_tensor_shape(
                        registry,
                        &layer_norm_tensor_name(layer, &format!("self_attn.{projection}")),
                        &[layout.head_dim],
                    )?;
                }
            }
            AttentionLayerType::Mla => {
                let _ = infer_mla_attention_layout(config, registry, layer)?;
            }
            AttentionLayerType::Linear => {
                let _ = infer_linear_attention_layout(config, registry, layer)?;
            }
        }
        require_tensor_shape(
            registry,
            &layer_norm_tensor_name(layer, "input_layernorm"),
            &[config.hidden_size],
        )?;
        require_tensor_shape(
            registry,
            &layer_norm_tensor_name(layer, "post_attention_layernorm"),
            &[config.hidden_size],
        )?;
        if config.is_dense_mlp_layer(layer) {
            let intermediate = config.intermediate_size.with_context(|| {
                format!("GLM dense MLP layer {layer} is missing intermediate_size")
            })?;
            for (projection, shape) in [
                ("gate_proj", [intermediate, config.hidden_size]),
                ("up_proj", [intermediate, config.hidden_size]),
                ("down_proj", [config.hidden_size, intermediate]),
            ] {
                require_tensor_shape(
                    registry,
                    &format!("model.layers.{layer}.mlp.{projection}.weight"),
                    &shape,
                )?;
            }
            continue;
        }
        require_tensor_shape(
            registry,
            &router_tensor_name(layer),
            &[config.experts(), config.hidden_size],
        )?;
        if config.glm.is_some() {
            require_tensor_shape(
                registry,
                &format!("model.layers.{layer}.mlp.gate.e_score_correction_bias"),
                &[config.experts()],
            )?;
        }
        let shared_experts = config.shared_experts();
        if shared_experts > 0 {
            let shared_inter = config.shared_expert_intermediate_size();
            if shared_inter == 0 {
                bail!(
                    "Qwen config declares {shared_experts} shared expert(s) but no shared expert intermediate size"
                );
            }
            let total_shared_inter = shared_experts
                .checked_mul(shared_inter)
                .context("shared expert intermediate size overflow")?;
            require_tensor_shape(
                registry,
                &shared_expert_tensor_name(layer, "gate_proj"),
                &[total_shared_inter, config.hidden_size],
            )?;
            require_tensor_shape(
                registry,
                &shared_expert_tensor_name(layer, "up_proj"),
                &[total_shared_inter, config.hidden_size],
            )?;
            require_tensor_shape(
                registry,
                &shared_expert_tensor_name(layer, "down_proj"),
                &[config.hidden_size, total_shared_inter],
            )?;
            if config.glm.is_none() {
                require_tensor_shape(
                    registry,
                    &shared_expert_gate_tensor_name(layer),
                    &[shared_experts, config.hidden_size],
                )?;
            }
        }
        // Per-expert tensor presence is intentionally not validated here.
        //
        // Reasons:
        // 1. Expert MLP correctness (gate/up/down projection shapes) is enforced per-expert at
        //    pack time by `validate_expert_tensor_group`.
        // 2. At runtime the packed expert files are managed by `ExpertSlotStore`; the registry
        //    records their original source metadata but is not used for expert inference.
        // 3. Real Qwen3 revision checkpoints may differ in expert naming (e.g. shared experts)
        //    or use a naming scheme that doesn't match the exact pattern assumed here.
        //    A rigid per-name loop would cause false rejections for such models.
    }
    Ok(())
}

pub(super) fn validate_qwen_q4_graph_bindings(
    family: QwenMoeFamily,
    config: &QwenModelConfig,
    runtime: &DenseTransformerRuntime,
    registry: &TensorRegistry,
    store_len: u64,
) -> Result<()> {
    for layer in 0..config.num_hidden_layers {
        if runtime.is_mla_attention_layer(layer) {
            let layout = runtime.mla_attention_layout(layer)?;
            for (projection, output_width, input_width) in [
                ("q_a_proj", layout.q_lora_rank, runtime.width),
                ("q_b_proj", layout.q_width, layout.q_lora_rank),
                ("kv_a_proj_with_mqa", layout.kv_a_width, runtime.width),
                ("o_proj", runtime.width, layout.attention_output_width),
            ] {
                require_resident_q4_graph_projection(
                    family,
                    registry,
                    store_len,
                    "MLA projection",
                    &attention_tensor_name(layer, projection),
                    output_width,
                    input_width,
                )?;
            }
            match layout.kv_projection {
                MlaKvProjectionLayout::FusedKvB => {
                    require_resident_q4_graph_projection(
                        family,
                        registry,
                        store_len,
                        "MLA KV-B projection",
                        &attention_tensor_name(layer, "kv_b_proj"),
                        layout.kv_b_width,
                        layout.kv_lora_rank,
                    )?;
                }
                MlaKvProjectionLayout::AbsorbedMultiLinear => {
                    require_resident_q4_multilinear_projection(
                        family,
                        registry,
                        store_len,
                        "MLA absorbed query projection",
                        &attention_tensor_name(layer, "embed_q"),
                        layout.num_heads,
                        layout.kv_lora_rank,
                        layout.qk_nope_head_dim,
                    )?;
                    require_resident_q4_multilinear_projection(
                        family,
                        registry,
                        store_len,
                        "MLA absorbed output projection",
                        &attention_tensor_name(layer, "unembed_out"),
                        layout.num_heads,
                        layout.v_head_dim,
                        layout.kv_lora_rank,
                    )?;
                }
            }
        } else if !runtime.is_linear_attention_layer(layer) {
            let layout = runtime.full_attention_layout(layer)?;
            let requests = full_attention_input_projection_requests(
                layer,
                layout.q_projection_width,
                layout.kv_width,
            )?;
            for request in requests.requests() {
                require_resident_q4_graph_projection(
                    family,
                    registry,
                    store_len,
                    "CMD1 full-attention projection",
                    request.tensor_name,
                    request.output_width,
                    runtime.width,
                )?;
            }
            require_resident_q4_graph_projection(
                family,
                registry,
                store_len,
                "CMD2 full-attention output projection",
                &attention_tensor_name(layer, "o_proj"),
                runtime.width,
                layout.num_q_heads * layout.head_dim,
            )?;
        }

        if config.is_dense_mlp_layer(layer) {
            let intermediate = config.intermediate_size.with_context(|| {
                format!("GLM dense MLP layer {layer} is missing intermediate_size")
            })?;
            for (projection, output_width, input_width) in [
                ("gate_proj", intermediate, runtime.width),
                ("up_proj", intermediate, runtime.width),
                ("down_proj", runtime.width, intermediate),
            ] {
                require_resident_q4_graph_projection(
                    family,
                    registry,
                    store_len,
                    "dense lead-in MLP projection",
                    &format!("model.layers.{layer}.mlp.{projection}.weight"),
                    output_width,
                    input_width,
                )?;
            }
        } else {
            if config.glm.is_some() {
                require_resident_graph_projection(
                    family,
                    registry,
                    store_len,
                    "CMD2 GLM router projection",
                    &router_tensor_name(layer),
                    config.experts(),
                    runtime.width,
                )?;
            } else {
                require_resident_q4_graph_projection(
                    family,
                    registry,
                    store_len,
                    "CMD2 router projection",
                    &router_tensor_name(layer),
                    config.experts(),
                    runtime.width,
                )?;
            }
        }
    }

    let lm_head_name = if registry.tensor("lm_head.weight").is_some() {
        "lm_head.weight"
    } else {
        "model.embed_tokens.weight"
    };
    if config.glm.is_some() {
        require_resident_graph_projection(
            family,
            registry,
            store_len,
            "GLM LM-head sampling projection",
            lm_head_name,
            config.vocab_size,
            runtime.width,
        )?;
    } else {
        require_resident_q4_graph_projection(
            family,
            registry,
            store_len,
            "LM-head sampling projection",
            lm_head_name,
            config.vocab_size,
            runtime.width,
        )?;
    }
    Ok(())
}

pub(super) fn require_resident_graph_projection(
    family: QwenMoeFamily,
    registry: &TensorRegistry,
    store_len: u64,
    stage: &str,
    tensor_name: &str,
    output_width: usize,
    input_width: usize,
) -> Result<()> {
    let entry = registry.require(tensor_name)?;
    ResidentMmapMatvecProjection::from_entry(
        tensor_name,
        entry,
        store_len,
        output_width,
        input_width,
    )
    .with_context(|| {
        format!(
            "FlashMoe unsupported resolved {family:?} {stage}: tensor {tensor_name} cannot bind resident shape {output_width}x{input_width}"
        )
    })?;
    Ok(())
}

pub(super) fn require_resident_q4_graph_projection(
    family: QwenMoeFamily,
    registry: &TensorRegistry,
    store_len: u64,
    stage: &str,
    tensor_name: &str,
    output_width: usize,
    input_width: usize,
) -> Result<()> {
    let entry = registry.require(tensor_name)?;
    DenseQ4MmapMatvecProjection::from_entry(
        tensor_name,
        entry,
        store_len,
        output_width,
        input_width,
    )?
    .with_context(|| {
        format!(
            "FlashMoe unsupported resolved {family:?} Q4 {stage}: tensor {tensor_name} cannot bind the resident projection for shape {output_width}x{input_width}"
        )
    })?;
    Ok(())
}

pub(super) fn require_resident_q4_multilinear_projection(
    family: QwenMoeFamily,
    registry: &TensorRegistry,
    store_len: u64,
    stage: &str,
    tensor_name: &str,
    heads: usize,
    output_width_per_head: usize,
    input_width: usize,
) -> Result<()> {
    let entry = registry.require(tensor_name)?;
    DenseQ4MmapMatvecProjection::from_multilinear_entry(
        tensor_name,
        entry,
        store_len,
        heads,
        output_width_per_head,
        input_width,
    )?
    .with_context(|| {
        format!(
            "FlashMoe unsupported resolved {family:?} Q4 {stage}: tensor {tensor_name} cannot bind the resident multilinear projection for shape {heads}x{output_width_per_head}x{input_width}"
        )
    })?;
    Ok(())
}

#[derive(Debug, Clone)]
pub(super) struct DenseTransformerRuntime {
    pub(super) width: usize,
    #[cfg(test)]
    pub(super) head_dim: usize,
    #[cfg(test)]
    pub(super) num_q_heads: usize,
    #[cfg(test)]
    pub(super) kv_heads: usize,
    pub(super) full_attention: Vec<Option<FullAttentionLayout>>,
    pub(super) mla_attention: Vec<Option<MlaAttentionLayout>>,
    pub(super) linear_attention: Vec<Option<LinearAttentionLayout>>,
}

impl DenseTransformerRuntime {
    pub(super) fn new(config: &QwenModelConfig) -> Self {
        #[cfg(test)]
        let head_dim = config.full_attention_head_dim();
        #[cfg(test)]
        let kv_heads = config.kv_heads();
        Self {
            width: config.hidden_size,
            #[cfg(test)]
            head_dim,
            #[cfg(test)]
            num_q_heads: config.num_attention_heads,
            #[cfg(test)]
            kv_heads,
            full_attention: vec![None; config.num_hidden_layers],
            mla_attention: vec![None; config.num_hidden_layers],
            linear_attention: vec![None; config.num_hidden_layers],
        }
    }

    pub(super) fn from_registry(
        config: &QwenModelConfig,
        registry: &TensorRegistry,
    ) -> Result<Self> {
        let mut runtime = Self::new(config);
        for layer in 0..config.num_hidden_layers {
            match infer_attention_layer_type(registry, layer)? {
                AttentionLayerType::Full => {
                    runtime.full_attention[layer] =
                        Some(infer_full_attention_layout(config, registry, layer)?);
                }
                AttentionLayerType::Mla => {
                    runtime.mla_attention[layer] =
                        Some(infer_mla_attention_layout(config, registry, layer)?);
                }
                AttentionLayerType::Linear => {
                    runtime.linear_attention[layer] =
                        Some(infer_linear_attention_layout(config, registry, layer)?);
                }
            }
        }

        Ok(runtime)
    }

    pub(super) fn full_attention_layout(&self, layer: usize) -> Result<FullAttentionLayout> {
        self.full_attention
            .get(layer)
            .copied()
            .flatten()
            .with_context(|| format!("missing full-attention runtime layout for layer {layer}"))
    }

    pub(super) fn linear_attention_layout(&self, layer: usize) -> Result<LinearAttentionLayout> {
        self.linear_attention
            .get(layer)
            .copied()
            .flatten()
            .with_context(|| format!("missing linear-attention runtime layout for layer {layer}"))
    }

    pub(super) fn mla_attention_layout(&self, layer: usize) -> Result<MlaAttentionLayout> {
        self.mla_attention
            .get(layer)
            .copied()
            .flatten()
            .with_context(|| format!("missing MLA runtime layout for layer {layer}"))
    }

    pub(super) fn is_mla_attention_layer(&self, layer: usize) -> bool {
        self.mla_attention
            .get(layer)
            .and_then(|layout| *layout)
            .is_some()
    }

    pub(super) fn is_linear_attention_layer(&self, layer: usize) -> bool {
        self.linear_attention
            .get(layer)
            .and_then(|layout| *layout)
            .is_some()
    }

    pub(super) fn resolved_attention_layers(&self) -> Result<Vec<QwenMoeLayerKind>> {
        self.full_attention
            .iter()
            .zip(&self.mla_attention)
            .zip(&self.linear_attention)
            .enumerate()
            .map(|(layer, ((full, mla), linear))| match (
                full.is_some(),
                mla.is_some(),
                linear.is_some(),
            ) {
                (true, false, false) | (false, true, false) => {
                    Ok(QwenMoeLayerKind::FullAttention)
                }
                (false, false, true) => Ok(QwenMoeLayerKind::LinearAttention),
                _ => bail!(
                    "FlashMoe dense runtime layer {layer} must resolve exactly one attention implementation"
                ),
            })
            .collect()
    }

    #[cfg(test)]
    pub(super) fn layer_kind(&self, layer: usize) -> FlashMoeLayerKind {
        if self.is_linear_attention_layer(layer) {
            FlashMoeLayerKind::LinearAttention
        } else if self.is_mla_attention_layer(layer)
            || self
                .full_attention
                .get(layer)
                .and_then(|layout| *layout)
                .is_some()
        {
            FlashMoeLayerKind::FullAttention
        } else {
            FlashMoeLayerKind::Unknown
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttentionLayerType {
    Full,
    Mla,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MlaKvProjectionLayout {
    FusedKvB,
    AbsorbedMultiLinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MlaAttentionLayout {
    pub(super) q_lora_rank: usize,
    pub(super) kv_lora_rank: usize,
    pub(super) qk_nope_head_dim: usize,
    pub(super) qk_rope_head_dim: usize,
    pub(super) qk_head_dim: usize,
    pub(super) v_head_dim: usize,
    pub(super) num_heads: usize,
    pub(super) q_width: usize,
    pub(super) kv_a_width: usize,
    pub(super) kv_b_width: usize,
    pub(super) attention_output_width: usize,
    pub(super) kv_projection: MlaKvProjectionLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(super) struct GlmMlaPostAttentionRequest<'a> {
    pub(super) residual: MetalBatchProjectionInput<'a>,
    pub(super) post_norm_weight: &'a [f32],
    pub(super) router_correction_bias: Option<&'a [f32]>,
    pub(super) experts: usize,
    pub(super) active_experts: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FullAttentionQLayout {
    Standard,
    Gated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RotaryPairing {
    #[allow(dead_code)]
    Adjacent,
    SplitHalf,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FullAttentionLayout {
    pub(super) q_layout: FullAttentionQLayout,
    pub(super) q_projection_width: usize,
    pub(super) q_width: usize,
    pub(super) kv_width: usize,
    pub(super) head_dim: usize,
    pub(super) rotary_dim: usize,
    pub(super) num_q_heads: usize,
    pub(super) kv_heads: usize,
    pub(super) rotary_pairing: RotaryPairing,
}

pub(super) fn infer_linear_attention_layout(
    config: &QwenModelConfig,
    registry: &TensorRegistry,
    layer: usize,
) -> Result<LinearAttentionLayout> {
    let qkv_name = linear_attention_tensor_name(layer, "in_proj_qkv");
    let z_name = linear_attention_tensor_name(layer, "in_proj_z");
    let b_name = linear_attention_tensor_name(layer, "in_proj_b");
    let a_name = linear_attention_tensor_name(layer, "in_proj_a");
    let conv_name = linear_attention_tensor_name(layer, "conv1d");
    let a_log_name = linear_attention_scalar_tensor_name(layer, "A_log");
    let dt_bias_name = linear_attention_scalar_tensor_name(layer, "dt_bias");
    let norm_name = linear_attention_tensor_name(layer, "norm");
    let out_proj_name = linear_attention_tensor_name(layer, "out_proj");

    let (qkv_rows, qkv_cols) = require_2d_tensor_shape(registry, &qkv_name)?;
    let (z_rows, z_cols) = require_2d_tensor_shape(registry, &z_name)?;
    let (b_rows, b_cols) = require_2d_tensor_shape(registry, &b_name)?;
    let (a_rows, a_cols) = require_2d_tensor_shape(registry, &a_name)?;
    let (out_rows, out_cols) = require_2d_tensor_shape(registry, &out_proj_name)?;
    let a_log_len = require_1d_tensor_shape(registry, &a_log_name)?;
    let dt_bias_len = require_1d_tensor_shape(registry, &dt_bias_name)?;
    let value_dim = require_1d_tensor_shape(registry, &norm_name)?;
    let (conv_channels, conv_kernel_size) = require_conv1d_tensor_shape(registry, &conv_name)?;
    if conv_kernel_size < 2 {
        bail!(
            "linear-attention layer {layer} conv1d kernel size {conv_kernel_size} must be at least 2"
        );
    }

    if qkv_cols != config.hidden_size
        || z_cols != config.hidden_size
        || b_cols != config.hidden_size
        || a_cols != config.hidden_size
    {
        bail!(
            "linear-attention layer {layer} projection input widths are qkv={qkv_cols}, z={z_cols}, b={b_cols}, a={a_cols}; expected hidden_size {}",
            config.hidden_size
        );
    }
    if out_rows != config.hidden_size {
        bail!(
            "linear-attention layer {layer} out_proj output rows {out_rows}; expected hidden_size {}",
            config.hidden_size
        );
    }
    if z_rows != out_cols {
        bail!(
            "linear-attention layer {layer} z width {z_rows} does not match out_proj input width {out_cols}"
        );
    }
    if value_dim == 0 || z_rows % value_dim != 0 {
        bail!(
            "linear-attention layer {layer} value width {z_rows} is not divisible by norm/value dim {value_dim}"
        );
    }
    let num_value_heads = z_rows / value_dim;
    if b_rows != num_value_heads
        || a_rows != num_value_heads
        || a_log_len != num_value_heads
        || dt_bias_len != num_value_heads
    {
        bail!(
            "linear-attention layer {layer} value head counts disagree: inferred={num_value_heads}, b={b_rows}, a={a_rows}, A_log={a_log_len}, dt_bias={dt_bias_len}"
        );
    }
    if qkv_rows < z_rows {
        bail!(
            "linear-attention layer {layer} qkv width {qkv_rows} is smaller than value width {z_rows}"
        );
    }
    let paired_key_width = qkv_rows - z_rows;
    if paired_key_width % 2 != 0 {
        bail!(
            "linear-attention layer {layer} qkv non-value width {paired_key_width} cannot split evenly into Q and K"
        );
    }
    let total_key_width = paired_key_width / 2;
    let key_dim = infer_linear_attention_key_dim(config, total_key_width, z_rows, value_dim)
        .with_context(|| {
            format!(
                "linear-attention layer {layer} cannot infer key dimension from key width {total_key_width}, value width {z_rows}, value dim {value_dim}"
            )
        })?;
    let num_key_heads = total_key_width / key_dim;
    if num_key_heads == 0 {
        bail!("linear-attention layer {layer} inferred zero key heads");
    }
    if num_value_heads % num_key_heads != 0 {
        bail!(
            "linear-attention layer {layer} value heads {num_value_heads} must be divisible by key heads {num_key_heads}"
        );
    }
    if conv_channels != qkv_rows {
        bail!(
            "linear-attention layer {layer} conv1d channels {conv_channels} do not match qkv width {qkv_rows}"
        );
    }

    Ok(LinearAttentionLayout {
        num_value_heads,
        num_key_heads,
        key_dim,
        value_dim,
        total_key_width,
        total_value_width: z_rows,
        conv_dim: qkv_rows,
        conv_kernel_size,
    })
}

pub(super) fn infer_linear_attention_key_dim(
    config: &QwenModelConfig,
    total_key_width: usize,
    total_value_width: usize,
    value_dim: usize,
) -> Result<usize> {
    let config_head_dim = config.hidden_size / config.num_attention_heads.max(1);
    if config_head_dim > 0 && total_key_width.is_multiple_of(config_head_dim) {
        return Ok(config_head_dim);
    }
    if is_known_qwen35_linear_attention_shape(total_key_width, total_value_width, value_dim) {
        return Ok(LINEAR_KEY_DIM);
    }
    if total_key_width.is_multiple_of(value_dim) {
        return Ok(value_dim);
    }
    bail!(
        "key width is not divisible by config head_dim {config_head_dim} or manifest value_dim {value_dim}"
    )
}

fn is_known_qwen35_linear_attention_shape(
    total_key_width: usize,
    total_value_width: usize,
    value_dim: usize,
) -> bool {
    total_key_width == LINEAR_TOTAL_KEY
        && total_value_width == LINEAR_TOTAL_VALUE
        && value_dim == LINEAR_VALUE_DIM
}

pub(super) fn infer_full_attention_layout(
    config: &QwenModelConfig,
    registry: &TensorRegistry,
    layer: usize,
) -> Result<FullAttentionLayout> {
    let q_name = attention_tensor_name(layer, "q_proj");
    let k_name = attention_tensor_name(layer, "k_proj");
    let v_name = attention_tensor_name(layer, "v_proj");
    let o_name = attention_tensor_name(layer, "o_proj");

    let (q_rows, q_cols) = require_2d_tensor_shape(registry, &q_name)?;
    let (k_rows, k_cols) = require_2d_tensor_shape(registry, &k_name)?;
    let (v_rows, v_cols) = require_2d_tensor_shape(registry, &v_name)?;
    let (o_rows, o_cols) = require_2d_tensor_shape(registry, &o_name)?;

    let num_q_heads = config.num_attention_heads;
    let kv_heads = config.kv_heads();

    if q_cols != config.hidden_size || k_cols != config.hidden_size || v_cols != config.hidden_size
    {
        bail!(
            "full-attention layer {layer} projection input widths are q={q_cols}, k={k_cols}, v={v_cols}; expected hidden_size {}",
            config.hidden_size
        );
    }
    if o_rows != config.hidden_size {
        bail!(
            "full-attention layer {layer} o_proj output rows {o_rows}; expected hidden_size {}",
            config.hidden_size
        );
    }
    if k_rows != v_rows {
        bail!(
            "full-attention layer {layer} k_proj rows {k_rows} do not match v_proj rows {v_rows}"
        );
    }
    if k_rows == 0 || k_rows % kv_heads != 0 {
        bail!(
            "full-attention layer {layer} k/v width {k_rows} is not divisible by kv_heads {kv_heads}"
        );
    }

    let head_dim = k_rows / kv_heads;
    let q_width = num_q_heads
        .checked_mul(head_dim)
        .context("full-attention q_width overflow")?;
    let gated_q_width = q_width
        .checked_mul(2)
        .context("gated full-attention q_width overflow")?;

    if q_rows == q_width && o_cols == q_width {
        return Ok(FullAttentionLayout {
            q_layout: FullAttentionQLayout::Standard,
            q_projection_width: q_width,
            q_width,
            kv_width: k_rows,
            head_dim,
            rotary_dim: rotary_dim_for(config, head_dim, FullAttentionQLayout::Standard),
            num_q_heads,
            kv_heads,
            rotary_pairing: RotaryPairing::SplitHalf,
        });
    }

    if q_rows == gated_q_width && o_cols == q_width {
        return Ok(FullAttentionLayout {
            q_layout: FullAttentionQLayout::Gated,
            q_projection_width: gated_q_width,
            q_width,
            kv_width: k_rows,
            head_dim,
            rotary_dim: rotary_dim_for(config, head_dim, FullAttentionQLayout::Gated),
            num_q_heads,
            kv_heads,
            rotary_pairing: RotaryPairing::SplitHalf,
        });
    }

    bail!(
        "unsupported full-attention layer {layer} layout: q_proj rows={q_rows}, k/v rows={k_rows}, o_proj shape=[{o_rows},{o_cols}], num_q_heads={num_q_heads}, kv_heads={kv_heads}, inferred head_dim={head_dim}; expected standard q_rows={q_width}, o_cols={q_width}, or gated q_rows={gated_q_width}, o_cols={q_width}"
    )
}

pub(super) fn infer_mla_attention_layout(
    config: &QwenModelConfig,
    registry: &TensorRegistry,
    layer: usize,
) -> Result<MlaAttentionLayout> {
    let glm = config
        .glm
        .as_ref()
        .with_context(|| format!("MLA tensors at layer {layer} require a GLM config"))?;
    let qk_head_dim = glm
        .qk_nope_head_dim
        .checked_add(glm.qk_rope_head_dim)
        .context("MLA q/k head width overflow")?;
    let q_width = config
        .num_attention_heads
        .checked_mul(qk_head_dim)
        .context("MLA query width overflow")?;
    let kv_a_width = glm
        .kv_lora_rank
        .checked_add(glm.qk_rope_head_dim)
        .context("MLA compressed KV width overflow")?;
    let kv_b_head_width = glm
        .qk_nope_head_dim
        .checked_add(glm.v_head_dim)
        .context("MLA KV-B head width overflow")?;
    let kv_b_width = config
        .num_attention_heads
        .checked_mul(kv_b_head_width)
        .context("MLA KV-B width overflow")?;
    let attention_output_width = config
        .num_attention_heads
        .checked_mul(glm.v_head_dim)
        .context("MLA attention output width overflow")?;

    let expected = [
        ("q_a_proj", glm.q_lora_rank, config.hidden_size),
        ("q_b_proj", q_width, glm.q_lora_rank),
        ("kv_a_proj_with_mqa", kv_a_width, config.hidden_size),
        ("o_proj", config.hidden_size, attention_output_width),
    ];
    for (projection, expected_rows, expected_cols) in expected {
        let tensor_name = attention_tensor_name(layer, projection);
        let (rows, cols) = require_2d_tensor_shape(registry, &tensor_name)?;
        if rows != expected_rows || cols != expected_cols {
            bail!(
                "MLA layer {layer} projection {tensor_name} has shape [{rows},{cols}], expected [{expected_rows},{expected_cols}]"
            );
        }
    }
    let kv_b_name = attention_tensor_name(layer, "kv_b_proj");
    let embed_q_name = attention_tensor_name(layer, "embed_q");
    let unembed_out_name = attention_tensor_name(layer, "unembed_out");
    let kv_projection = if registry.tensor(&kv_b_name).is_some() {
        require_tensor_shape(registry, &kv_b_name, &[kv_b_width, glm.kv_lora_rank])?;
        MlaKvProjectionLayout::FusedKvB
    } else if registry.tensor(&embed_q_name).is_some()
        && registry.tensor(&unembed_out_name).is_some()
    {
        require_tensor_shape(
            registry,
            &embed_q_name,
            &[
                config.num_attention_heads,
                glm.kv_lora_rank,
                glm.qk_nope_head_dim,
            ],
        )?;
        require_tensor_shape(
            registry,
            &unembed_out_name,
            &[config.num_attention_heads, glm.v_head_dim, glm.kv_lora_rank],
        )?;
        MlaKvProjectionLayout::AbsorbedMultiLinear
    } else {
        bail!(
            "MLA layer {layer} is missing {kv_b_name} or the absorbed pair {embed_q_name} and {unembed_out_name}"
        );
    };
    require_tensor_shape(
        registry,
        &layer_norm_tensor_name(layer, "self_attn.q_a_layernorm"),
        &[glm.q_lora_rank],
    )?;
    require_tensor_shape(
        registry,
        &layer_norm_tensor_name(layer, "self_attn.kv_a_layernorm"),
        &[glm.kv_lora_rank],
    )?;

    Ok(MlaAttentionLayout {
        q_lora_rank: glm.q_lora_rank,
        kv_lora_rank: glm.kv_lora_rank,
        qk_nope_head_dim: glm.qk_nope_head_dim,
        qk_rope_head_dim: glm.qk_rope_head_dim,
        qk_head_dim,
        v_head_dim: glm.v_head_dim,
        num_heads: config.num_attention_heads,
        q_width,
        kv_a_width,
        kv_b_width,
        attention_output_width,
        kv_projection,
    })
}

pub(super) fn rotary_dim_for(
    config: &QwenModelConfig,
    head_dim: usize,
    q_layout: FullAttentionQLayout,
) -> usize {
    let factor = config.partial_rotary_factor.unwrap_or_else(|| {
        if q_layout == FullAttentionQLayout::Gated {
            0.25
        } else {
            1.0
        }
    });

    let mut rotary_dim = ((head_dim as f64) * factor).round() as usize;
    rotary_dim = rotary_dim.clamp(2, head_dim);
    rotary_dim - (rotary_dim % 2)
}

pub(super) fn require_2d_tensor_shape(
    registry: &TensorRegistry,
    canonical_name: &str,
) -> Result<(usize, usize)> {
    let tensor = registry.require(canonical_name)?;
    ensure_runtime_tensor_storage_supported(canonical_name, tensor)?;
    match tensor.shape.as_slice() {
        [rows, cols] if *rows > 0 && *cols > 0 => Ok((*rows, *cols)),
        shape => bail!(
            "Flash-MoE tensor {canonical_name} has shape {:?}; expected non-empty 2-D matrix",
            shape
        ),
    }
}

pub(super) fn require_tensor_shape(
    registry: &TensorRegistry,
    canonical_name: &str,
    expected_shape: &[usize],
) -> Result<()> {
    let tensor = registry.require(canonical_name)?;
    ensure_runtime_tensor_storage_supported(canonical_name, tensor)?;
    if tensor.shape.as_slice() != expected_shape {
        bail!(
            "Flash-MoE tensor {canonical_name} has shape {:?}; expected {:?}",
            tensor.shape,
            expected_shape
        );
    }
    Ok(())
}

pub(super) fn require_1d_tensor_shape(
    registry: &TensorRegistry,
    canonical_name: &str,
) -> Result<usize> {
    let tensor = registry.require(canonical_name)?;
    ensure_runtime_tensor_storage_supported(canonical_name, tensor)?;
    match tensor.shape.as_slice() {
        [width] if *width > 0 => Ok(*width),
        shape => bail!(
            "Flash-MoE tensor {canonical_name} has shape {:?}; expected non-empty 1-D vector",
            shape
        ),
    }
}

pub(super) fn require_conv1d_tensor_shape(
    registry: &TensorRegistry,
    canonical_name: &str,
) -> Result<(usize, usize)> {
    let tensor = registry.require(canonical_name)?;
    ensure_runtime_tensor_storage_supported(canonical_name, tensor)?;
    match tensor.shape.as_slice() {
        [channels, kernel_size] if *channels > 0 && *kernel_size > 0 => {
            Ok((*channels, *kernel_size))
        }
        [channels, 1, kernel_size] if *channels > 0 && *kernel_size > 0 => {
            Ok((*channels, *kernel_size))
        }
        [channels, kernel_size, 1] if *channels > 0 && *kernel_size > 0 => {
            Ok((*channels, *kernel_size))
        }
        shape => bail!(
            "Flash-MoE tensor {canonical_name} has shape {:?}; expected non-empty [channels, kernel], [channels, 1, kernel], or [channels, kernel, 1]",
            shape
        ),
    }
}

pub(super) fn ensure_runtime_tensor_storage_supported(
    canonical_name: &str,
    tensor: &RuntimeTensorEntry,
) -> Result<()> {
    match &tensor.quantization {
        TensorQuantization::None => {
            if dtype_size(&tensor.dtype).is_none() {
                bail!(
                    "Flash-MoE tensor {canonical_name} has unsupported dtype {}",
                    tensor.dtype
                );
            }
        }
        TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } => {
            if !(tensor.dtype.eq_ignore_ascii_case("U32")
                || tensor.dtype.eq_ignore_ascii_case("U8"))
            {
                bail!(
                    "Flash-MoE q4 tensor {canonical_name} has unsupported packed dtype {}",
                    tensor.dtype
                );
            }
            dense_q4_layout_with_scale_bias_dtype(&tensor.shape, *group_size, scale_bias_dtype)
                .with_context(|| {
                    format!("Flash-MoE q4 tensor {canonical_name} has unsupported runtime layout")
                })?;
        }
    }
    Ok(())
}

pub(super) fn infer_attention_layer_type(
    registry: &TensorRegistry,
    layer: usize,
) -> Result<AttentionLayerType> {
    let linear_prefix = format!("model.layers.{layer}.linear_attn.");
    let has_linear = registry.has_tensor_with_prefix(&linear_prefix);
    let has_mla = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"]
        .iter()
        .any(|projection| {
            registry
                .tensor(&attention_tensor_name(layer, projection))
                .is_some()
        });
    let has_full = ["q_proj", "k_proj", "v_proj"].iter().any(|projection| {
        registry
            .tensor(&attention_tensor_name(layer, projection))
            .is_some()
    });

    match (has_linear, has_mla, has_full) {
        (true, false, true) => bail!(
            "Flash-MoE tensor manifest has both linear-attention tensors ({linear_prefix}*) and full-attention self_attn projection tensors for layer {layer}"
        ),
        (true, true, _) | (_, true, true) => bail!(
            "Flash-MoE tensor manifest resolves more than one attention implementation for layer {layer}"
        ),
        (true, false, false) => Ok(AttentionLayerType::Linear),
        (false, true, false) => Ok(AttentionLayerType::Mla),
        (false, false, true) => Ok(AttentionLayerType::Full),
        (false, false, false) => bail!(
            "Flash-MoE tensor manifest is missing attention tensors for layer {layer}; expected linear attention, MLA, or self_attn.{{q,k,v,o}}_proj tensors"
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentDenseLayout {
    Q4,
    Bf16,
    F16,
    F32,
}

impl ResidentDenseLayout {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Q4 => "resident Q4",
            Self::Bf16 => "resident BF16",
            Self::F16 => "resident F16",
            Self::F32 => "resident F32",
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FlashMoeManifest {
    pub model: String,
    pub cache_version: String,
    pub dense_shards: Vec<String>,
    pub expert_tensors: Vec<ExpertTensorRef>,
    pub dense_tensors: Vec<DenseTensorRef>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpertTensorRef {
    pub tensor: String,
    pub shard: String,
    pub layer: Option<usize>,
    pub expert: Option<usize>,
    pub dtype: Option<String>,
    pub shape: Vec<usize>,
    pub source_offsets: Option<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q4_sources: Option<DenseQ4SourceRefs>,
}

impl AggregateExpertTensor for ExpertTensorRef {
    fn aggregate_tensor_name(&self) -> &str {
        &self.tensor
    }

    fn aggregate_tensor_shape(&self) -> &[usize] {
        &self.shape
    }

    fn aggregate_tensor_has_native_q4(&self) -> bool {
        self.q4_sources.is_some()
    }
}

impl ExpertSourceTensor for ExpertTensorRef {
    fn expert_source_offsets(&self) -> Option<[u64; 2]> {
        self.source_offsets
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DenseTensorRef {
    pub tensor: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub source_offsets: [u64; 2],
    pub runtime_offset: u64,
    pub byte_len: u64,
    #[serde(default)]
    pub quantization: TensorQuantization,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q4_sources: Option<DenseQ4SourceRefs>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ4SourceFormat {
    MlxAffine,
    ColibriInt4,
    ColibriInt8,
    MlxMxfp4,
}

impl Default for DenseQ4SourceFormat {
    fn default() -> Self {
        Self::MlxAffine
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DenseQ4SourceRefs {
    pub scales_shard: String,
    pub scales_offsets: [u64; 2],
    pub biases_shard: String,
    pub biases_offsets: [u64; 2],
    pub scale_bias_dtype: String,
    #[serde(default)]
    pub source_format: DenseQ4SourceFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_group_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TensorQuantization {
    None,
    Q4 {
        group_size: usize,
        format: String,
        #[serde(default = "default_dense_q4_scale_bias_dtype")]
        scale_bias_dtype: String,
    },
}

impl Default for TensorQuantization {
    fn default() -> Self {
        Self::None
    }
}

fn default_dense_q4_scale_bias_dtype() -> String {
    "F32".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeTensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub alignment: u64,
    pub quantization: TensorQuantization,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorRegistry {
    tensors: BTreeMap<String, RuntimeTensorEntry>,
}

impl TensorRegistry {
    pub fn load(manifest_path: &Path) -> Result<Self> {
        let manifest: FlashMoeManifest =
            serde_json::from_slice(&fs::read(manifest_path).with_context(|| {
                format!(
                    "failed to read Flash-MoE tensor manifest {}",
                    manifest_path.display()
                )
            })?)
            .with_context(|| {
                format!(
                    "failed to parse Flash-MoE tensor manifest {}",
                    manifest_path.display()
                )
            })?;
        Ok(Self::from_manifest(&manifest))
    }

    pub(crate) fn from_manifest(manifest: &FlashMoeManifest) -> Self {
        let mut tensors = BTreeMap::new();
        for tensor in &manifest.dense_tensors {
            insert_tensor_entry_with_aliases(
                &mut tensors,
                &tensor.tensor,
                RuntimeTensorEntry {
                    name: tensor.tensor.clone(),
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    byte_offset: tensor.runtime_offset,
                    byte_len: tensor.byte_len,
                    alignment: TENSOR_ALIGNMENT,
                    quantization: tensor.quantization.clone(),
                },
            );
        }
        for tensor in &manifest.expert_tensors {
            if let Some([start, end]) = tensor.source_offsets {
                insert_tensor_entry_with_aliases(
                    &mut tensors,
                    &tensor.tensor,
                    RuntimeTensorEntry {
                        name: tensor.tensor.clone(),
                        dtype: tensor
                            .dtype
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string()),
                        shape: tensor.shape.clone(),
                        byte_offset: start,
                        byte_len: end.saturating_sub(start),
                        alignment: TENSOR_ALIGNMENT,
                        quantization: TensorQuantization::Q4 {
                            group_size: GROUP_SIZE,
                            format: ExpertQuantization::FourBitProduction.as_str().to_string(),
                            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                        },
                    },
                );
            }
        }
        Self { tensors }
    }

    pub fn tensor(&self, canonical_name: &str) -> Option<&RuntimeTensorEntry> {
        self.tensors.get(canonical_name)
    }

    pub(crate) fn has_tensor_with_prefix(&self, prefix: &str) -> bool {
        self.tensors.keys().any(|name| name.starts_with(prefix))
    }

    pub fn require(&self, canonical_name: &str) -> Result<&RuntimeTensorEntry> {
        self.tensor(canonical_name)
            .with_context(|| format!("Flash-MoE tensor registry is missing {canonical_name}"))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub(crate) fn resolve_resident_dense_layout(&self) -> Result<ResidentDenseLayout> {
        let matrix_tensors = self
            .tensors
            .values()
            .filter(|tensor| tensor.shape.len() >= 2)
            .filter(|tensor| !is_routed_expert_tensor_name(&tensor.name));
        let mut found_matrix = false;
        let mut found_q4 = false;
        let mut unquantized_layout: Option<ResidentDenseLayout> = None;

        for tensor in matrix_tensors {
            found_matrix = true;
            match tensor.quantization {
                TensorQuantization::Q4 { .. } => found_q4 = true,
                TensorQuantization::None => {
                    let layout =
                        resident_dense_layout_for_dtype(&tensor.dtype).with_context(|| {
                            format!(
                                "FlashMoe dense tensor {} has unsupported resident dtype {}",
                                tensor.name, tensor.dtype
                            )
                        })?;
                    if let Some(existing) = unquantized_layout
                        && existing != layout
                    {
                        bail!(
                            "FlashMoe dense manifest mixes resident matrix layouts {} and {}",
                            existing.as_str(),
                            layout.as_str()
                        );
                    }
                    unquantized_layout = Some(layout);
                }
            }
        }

        if !found_matrix {
            bail!("FlashMoe dense manifest contains no matrix tensors");
        }
        if found_q4 {
            return Ok(ResidentDenseLayout::Q4);
        }
        unquantized_layout
            .context("FlashMoe dense manifest has no resolvable resident matrix layout")
    }
}

fn resident_dense_layout_for_dtype(dtype: &str) -> Option<ResidentDenseLayout> {
    match dtype.to_ascii_uppercase().as_str() {
        "BF16" | "BFLOAT16" => Some(ResidentDenseLayout::Bf16),
        "F16" | "FLOAT16" | "FP16" => Some(ResidentDenseLayout::F16),
        "F32" | "FLOAT32" | "FP32" => Some(ResidentDenseLayout::F32),
        _ => None,
    }
}

fn is_routed_expert_tensor_name(name: &str) -> bool {
    name.contains(".mlp.experts.") || name.contains(".switch_mlp.")
}

fn insert_tensor_entry_with_aliases(
    tensors: &mut BTreeMap<String, RuntimeTensorEntry>,
    name: &str,
    entry: RuntimeTensorEntry,
) {
    tensors
        .entry(name.to_string())
        .or_insert_with(|| entry.clone());
    let canonical_name = canonical_hf_tensor_name(name);
    if canonical_name != name {
        tensors.entry(canonical_name).or_insert(entry);
    }
}

pub(crate) fn canonical_hf_tensor_name(name: &str) -> String {
    let canonical = if let Some(rest) = name.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else if let Some(rest) = name.strip_prefix("language_model.") {
        rest.to_string()
    } else if let Some(rest) = name.strip_prefix("model.visual.") {
        format!("visual.{rest}")
    } else if let Some(rest) = name.strip_prefix("vision_tower.") {
        format!("visual.{rest}")
    } else {
        name.to_string()
    };
    if canonical.starts_with("visual.") {
        canonical
            .replace(".mlp.linear_fc1.", ".mlp.fc1.")
            .replace(".mlp.linear_fc2.", ".mlp.fc2.")
    } else {
        canonical.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseQ4Layout {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) group_size: usize,
    pub(crate) row_packed_bytes: usize,
    pub(crate) groups_per_row: usize,
    pub(crate) packed_bytes: usize,
    pub(crate) scales_bytes: usize,
    pub(crate) scale_bias_bytes: usize,
    pub(crate) total_bytes: usize,
}

#[cfg(test)]
pub(crate) fn dense_q4_layout(shape: &[usize], group_size: usize) -> Result<DenseQ4Layout> {
    dense_q4_layout_with_scale_bias_dtype(shape, group_size, EXPERT_SCALE_BIAS_DTYPE_F32)
}

pub(crate) fn dense_q4_layout_with_scale_bias_dtype(
    shape: &[usize],
    group_size: usize,
    scale_bias_dtype: &str,
) -> Result<DenseQ4Layout> {
    if group_size == 0 {
        bail!("dense q4 group_size must be positive");
    }
    let cols = shape.last().copied().unwrap_or(0);
    if shape.len() < 2 || cols == 0 {
        bail!(
            "dense q4 tensor shape {:?} is not a non-empty matrix",
            shape
        );
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .context("dense q4 tensor row count overflow")
        })?;
    let row_packed_bytes = cols.div_ceil(2);
    let groups_per_row = cols.div_ceil(group_size);
    let packed_bytes = rows
        .checked_mul(row_packed_bytes)
        .context("dense q4 packed byte length overflow")?;
    let groups = rows
        .checked_mul(groups_per_row)
        .context("dense q4 group count overflow")?;
    let scale_bias_bytes = expert_scale_bias_dtype_size(scale_bias_dtype)
        .with_context(|| format!("unsupported dense q4 scale/bias dtype {scale_bias_dtype}"))?;
    let scales_bytes = groups
        .checked_mul(scale_bias_bytes)
        .context("dense q4 scale byte length overflow")?;
    let total_bytes = packed_bytes
        .checked_add(scales_bytes)
        .and_then(|value| value.checked_add(scales_bytes))
        .context("dense q4 total byte length overflow")?;
    Ok(DenseQ4Layout {
        rows,
        cols,
        group_size,
        row_packed_bytes,
        groups_per_row,
        packed_bytes,
        scales_bytes,
        scale_bias_bytes,
        total_bytes,
    })
}

pub(crate) fn validate_dense_matvec_shape(
    entry: &RuntimeTensorEntry,
    canonical_name: &str,
    expected_rows: usize,
    input_len: usize,
) -> Result<(usize, usize)> {
    let expected_shape = [expected_rows, input_len];
    match entry.shape.as_slice() {
        [rows, cols] if *rows == expected_rows && *cols == input_len => Ok((*rows, *cols)),
        _ => bail!(
            "Flash-MoE dense tensor {canonical_name} shape mismatch: expected shape {:?}, actual shape {:?}, input length {input_len}",
            expected_shape,
            entry.shape
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RouterScoreProjectionBinding {
    ResidentDense(DenseMmapMatvecProjection),
    ResidentQ4(DenseQ4MmapMatvecProjection),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RouterScoreProjectionDescriptor {
    pub(crate) layer: usize,
    pub(crate) tensor_name: String,
    pub(crate) experts: usize,
    pub(crate) hidden_width: usize,
    pub(crate) binding: RouterScoreProjectionBinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterScoreProjectionExecutionKind {
    ResidentDense,
    ResidentQ4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RouterScoreProjectionExecution<'a> {
    pub(crate) layer: usize,
    pub(crate) tensor_name: &'a str,
    pub(crate) experts: usize,
    pub(crate) hidden_width: usize,
    pub(crate) kind: RouterScoreProjectionExecutionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterScoreProjectionScoreSource {
    ResidentDenseFullTensor,
    DeclaredRows,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RouterScoreProjectionScorePlan<'a> {
    pub(crate) tensor_name: &'a str,
    pub(crate) experts: usize,
    pub(crate) hidden_width: usize,
    pub(crate) source: RouterScoreProjectionScoreSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum RouterScoreProjectionTopKSource {
    ResidentDense(DenseMmapMatvecProjection),
    ResidentQ4(DenseQ4MmapMatvecProjection),
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct RouterScoreProjectionTopKPlan {
    pub(crate) layer: usize,
    pub(crate) tensor_name: String,
    pub(crate) experts: usize,
    pub(crate) hidden_width: usize,
    pub(crate) active_experts: usize,
    pub(crate) source: RouterScoreProjectionTopKSource,
}

impl<'a> RouterScoreProjectionExecution<'a> {
    pub(crate) fn score_plan(
        self,
        hidden_len: usize,
    ) -> Result<RouterScoreProjectionScorePlan<'a>> {
        if self.hidden_width != hidden_len {
            bail!(
                "Flash-MoE router score projection hidden length {} does not match declared width {}",
                hidden_len,
                self.hidden_width
            );
        }
        let source = match self.kind {
            RouterScoreProjectionExecutionKind::ResidentDense => {
                RouterScoreProjectionScoreSource::ResidentDenseFullTensor
            }
            RouterScoreProjectionExecutionKind::ResidentQ4 => {
                RouterScoreProjectionScoreSource::DeclaredRows
            }
        };
        Ok(RouterScoreProjectionScorePlan {
            tensor_name: self.tensor_name,
            experts: self.experts,
            hidden_width: self.hidden_width,
            source,
        })
    }
}

impl RouterScoreProjectionDescriptor {
    pub(crate) fn from_entry(
        layer: usize,
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        experts: usize,
        hidden_width: usize,
    ) -> Result<Self> {
        match &entry.quantization {
            TensorQuantization::None => {
                let Some(element_size) = dense_dtype_size(&entry.dtype) else {
                    bail!(
                        "Flash-MoE router tensor {} has unsupported dtype {}",
                        tensor_name,
                        entry.dtype
                    );
                };
                let projection = DenseMmapMatvecProjection::from_entry(
                    tensor_name,
                    entry,
                    store_len,
                    experts,
                    hidden_width,
                    element_size,
                )?;
                Ok(Self {
                    layer,
                    tensor_name: tensor_name.to_string(),
                    experts,
                    hidden_width,
                    binding: RouterScoreProjectionBinding::ResidentDense(projection),
                })
            }
            TensorQuantization::Q4 { .. } => {
                let Some(projection) = DenseQ4MmapMatvecProjection::from_entry(
                    tensor_name,
                    entry,
                    store_len,
                    experts,
                    hidden_width,
                )?
                else {
                    bail!(
                        "Flash-MoE router tensor {tensor_name} cannot resolve a resident Q4 projection descriptor for shape [{experts}, {hidden_width}]"
                    );
                };
                Ok(Self {
                    layer,
                    tensor_name: tensor_name.to_string(),
                    experts,
                    hidden_width,
                    binding: RouterScoreProjectionBinding::ResidentQ4(projection),
                })
            }
        }
    }

    pub(crate) fn execution(
        &self,
        layer: usize,
        experts: usize,
        hidden_width: usize,
    ) -> Result<RouterScoreProjectionExecution<'_>> {
        if self.layer != layer {
            bail!(
                "Flash-MoE router score projection execution layer {} does not match scheduled layer {}",
                self.layer,
                layer
            );
        }
        if self.experts != experts {
            bail!(
                "Flash-MoE router score projection execution experts {} does not match scheduled experts {}",
                self.experts,
                experts
            );
        }
        if self.hidden_width != hidden_width {
            bail!(
                "Flash-MoE router score projection execution hidden width {} does not match scheduled hidden width {}",
                self.hidden_width,
                hidden_width
            );
        }
        let kind = match self.binding {
            RouterScoreProjectionBinding::ResidentDense(_) => {
                RouterScoreProjectionExecutionKind::ResidentDense
            }
            RouterScoreProjectionBinding::ResidentQ4(_) => {
                RouterScoreProjectionExecutionKind::ResidentQ4
            }
        };
        Ok(RouterScoreProjectionExecution {
            layer: self.layer,
            tensor_name: &self.tensor_name,
            experts: self.experts,
            hidden_width: self.hidden_width,
            kind,
        })
    }

    #[cfg(test)]
    pub(crate) fn topk_plan(
        &self,
        hidden_len: usize,
        active_experts: usize,
    ) -> Result<RouterScoreProjectionTopKPlan> {
        if self.hidden_width != hidden_len {
            bail!(
                "Flash-MoE router score projection topK hidden length {} does not match declared width {}",
                hidden_len,
                self.hidden_width
            );
        }
        if active_experts == 0 || active_experts > self.experts {
            bail!(
                "Flash-MoE router score projection topK active experts {} is outside declared expert range 1..={}",
                active_experts,
                self.experts
            );
        }
        let source = match &self.binding {
            RouterScoreProjectionBinding::ResidentDense(projection) => {
                RouterScoreProjectionTopKSource::ResidentDense(projection.clone())
            }
            RouterScoreProjectionBinding::ResidentQ4(projection) => {
                RouterScoreProjectionTopKSource::ResidentQ4(projection.clone())
            }
        };
        Ok(RouterScoreProjectionTopKPlan {
            layer: self.layer,
            tensor_name: self.tensor_name.clone(),
            experts: self.experts,
            hidden_width: self.hidden_width,
            active_experts,
            source,
        })
    }
}

pub(crate) fn build_router_score_projection_descriptor<'a, F>(
    layer: usize,
    experts: usize,
    hidden_width: usize,
    store_len: u64,
    mut lookup: F,
) -> Result<Option<RouterScoreProjectionDescriptor>>
where
    F: FnMut(&str) -> Option<&'a RuntimeTensorEntry>,
{
    let tensor_name = router_tensor_name(layer);
    let Some(entry) = lookup(&tensor_name) else {
        return Ok(None);
    };
    RouterScoreProjectionDescriptor::from_entry(
        layer,
        &tensor_name,
        entry,
        store_len,
        experts,
        hidden_width,
    )
    .map(Some)
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RouterScoreBatch {
    state: FlashMoeRoutingOutputState,
    pub(crate) projection: Option<RouterScoreProjectionDescriptor>,
    pub(crate) scores: Vec<f32>,
}

impl RouterScoreBatch {
    pub(crate) fn new(
        state: FlashMoeRoutingOutputState,
        projection: Option<RouterScoreProjectionDescriptor>,
        scores: Vec<f32>,
    ) -> Result<Self> {
        if !state.is_declared_graph_state() {
            bail!("FlashMoe router score batch is not declared graph state");
        }
        if state.source() != FlashMoeRoutingOutputSource::CpuRouterScores {
            bail!(
                "FlashMoe router score batch source {:?} is not CPU router scores",
                state.source()
            );
        }
        if scores.len() != state.experts() {
            bail!(
                "FlashMoe router score batch has {} scores for {} declared experts",
                scores.len(),
                state.experts()
            );
        }
        if let Some(projection) = projection.as_ref() {
            if projection.layer != state.layer() {
                bail!(
                    "FlashMoe router score batch layer {} does not match projection layer {}",
                    state.layer(),
                    projection.layer
                );
            }
            if projection.experts != state.experts() {
                bail!(
                    "FlashMoe router score batch expert count {} does not match projection experts {}",
                    state.experts(),
                    projection.experts
                );
            }
        }
        Ok(Self {
            state,
            projection,
            scores,
        })
    }

    pub(crate) fn state(&self) -> FlashMoeRoutingOutputState {
        self.state
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Cmd2ResidentPostAttentionPrepProjections {
    pub(crate) layer: usize,
    pub(crate) out_proj: ResidentMmapMatvecProjection,
    pub(crate) router: ResidentMmapMatvecProjection,
    pub(crate) experts: usize,
    pub(crate) residual_width: usize,
    pub(crate) attention_width: usize,
    pub(crate) active_experts: usize,
}

impl Cmd2ResidentPostAttentionPrepProjections {
    pub(crate) fn new(
        layer: usize,
        out_proj: ResidentMmapMatvecProjection,
        router: ResidentMmapMatvecProjection,
        experts: usize,
        residual_width: usize,
        attention_width: usize,
        active_experts: usize,
    ) -> Result<Self> {
        if residual_width == 0 || attention_width == 0 || experts == 0 {
            bail!(
                "FlashMoe CMD2 resident post-attention prep requires non-zero experts, residual width, and attention width"
            );
        }
        if out_proj.output_width() != residual_width || out_proj.cols() != attention_width {
            bail!(
                "FlashMoe CMD2 resident post-attention output projection shape is invalid: output_width={} cols={} expected output_width={} cols={}",
                out_proj.output_width(),
                out_proj.cols(),
                residual_width,
                attention_width
            );
        }
        if router.output_width() != experts || router.cols() != residual_width {
            bail!(
                "FlashMoe CMD2 resident post-attention router projection shape is invalid: output_width={} cols={} expected output_width={} cols={}",
                router.output_width(),
                router.cols(),
                experts,
                residual_width
            );
        }
        Ok(Self {
            layer,
            out_proj,
            router,
            experts,
            residual_width,
            attention_width,
            active_experts,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cmd2ResidentPostAttentionPrepPlan {
    pub(crate) layer: usize,
    pub(crate) width: usize,
    pub(crate) attention_width: usize,
    pub(crate) experts: usize,
    pub(crate) active_count: usize,
}

impl Cmd2ResidentPostAttentionPrepProjections {
    pub(crate) fn resident_plan(
        &self,
        attention_width: usize,
        residual_width: usize,
        post_norm_weight_len: usize,
    ) -> Result<Cmd2ResidentPostAttentionPrepPlan> {
        if self.active_experts == 0 {
            bail!(
                "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: active expert count is zero"
            );
        }
        if attention_width == 0 || residual_width == 0 {
            bail!(
                "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: attention width {attention_width} and residual width {residual_width} must be non-zero"
            );
        }
        if residual_width != post_norm_weight_len {
            bail!(
                "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: post-attention norm weight length {post_norm_weight_len} does not match residual width {residual_width}"
            );
        }
        if self.out_proj.output_width() != residual_width
            || self.out_proj.rows() != residual_width
            || self.out_proj.cols() != attention_width
            || self.router.cols() != residual_width
            || self.router.output_width() != self.router.rows()
        {
            bail!(
                "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: projection shapes out={}x{} rows={} router={}x{} rows={} do not match attention width {} and residual width {}",
                self.out_proj.output_width(),
                self.out_proj.cols(),
                self.out_proj.rows(),
                self.router.output_width(),
                self.router.cols(),
                self.router.rows(),
                attention_width,
                residual_width
            );
        }
        Ok(Cmd2ResidentPostAttentionPrepPlan {
            layer: self.layer,
            width: residual_width,
            attention_width,
            experts: self.router.rows(),
            active_count: self.active_experts.min(self.router.rows()).max(1),
        })
    }
}

#[cfg(test)]
fn build_cmd2_resident_post_attention_prep_projections<F>(
    layer: usize,
    experts: usize,
    out_proj_name: &str,
    attention_width: usize,
    residual_width: usize,
    active_experts: usize,
    mut projection: F,
) -> Result<Option<Cmd2ResidentPostAttentionPrepProjections>>
where
    F: FnMut(&str, usize, usize) -> Result<Option<ResidentMmapMatvecProjection>>,
{
    if experts == 0 || attention_width == 0 || residual_width == 0 {
        return Ok(None);
    }
    let Some(out_proj) = projection(out_proj_name, residual_width, attention_width)? else {
        return Ok(None);
    };
    let router_name = router_tensor_name(layer);
    let Some(router) = projection(&router_name, experts, residual_width)? else {
        return Ok(None);
    };
    Cmd2ResidentPostAttentionPrepProjections::new(
        layer,
        out_proj,
        router,
        experts,
        residual_width,
        attention_width,
        active_experts,
    )
    .map(Some)
}

pub(crate) fn build_required_cmd2_resident_post_attention_prep_projections<F>(
    layer: usize,
    experts: usize,
    out_proj_name: &str,
    attention_width: usize,
    residual_width: usize,
    active_experts: usize,
    mut projection: F,
) -> Result<Cmd2ResidentPostAttentionPrepProjections>
where
    F: FnMut(&str, usize, usize) -> Result<Option<ResidentMmapMatvecProjection>>,
{
    if experts == 0 || attention_width == 0 || residual_width == 0 {
        bail!(
            "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: experts {experts}, attention width {attention_width}, and residual width {residual_width} must be non-zero"
        );
    }
    let out_proj = projection(out_proj_name, residual_width, attention_width)?.ok_or_else(|| {
        anyhow::anyhow!(
            "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: missing output projection {out_proj_name}"
        )
    })?;
    let router_name = router_tensor_name(layer);
    let router = projection(&router_name, experts, residual_width)?.ok_or_else(|| {
        anyhow::anyhow!(
            "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: missing router projection {router_name}"
        )
    })?;
    Cmd2ResidentPostAttentionPrepProjections::new(
        layer,
        out_proj,
        router,
        experts,
        residual_width,
        attention_width,
        active_experts,
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuVisibleNextNormWeights<'a> {
    pub(crate) tensor_name: &'a str,
    pub(crate) width: usize,
    values: &'a [f32],
}

impl<'a> CpuVisibleNextNormWeights<'a> {
    pub(crate) fn new(tensor_name: &'a str, values: &'a [f32], width: usize) -> Result<Self> {
        if tensor_name.is_empty() {
            bail!("FlashMoe scheduled next-norm weights require a tensor name");
        }
        if width == 0 {
            bail!("FlashMoe scheduled next-norm weights require non-zero width");
        }
        if values.len() < width {
            bail!(
                "FlashMoe scheduled next-norm weight tensor {tensor_name} length {} is smaller than width {width}",
                values.len()
            );
        }
        Ok(Self {
            tensor_name,
            width,
            values,
        })
    }

    pub(crate) fn values(self) -> &'a [f32] {
        &self.values[..self.width]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScheduledNextNormWeights<'a> {
    None,
    CpuVisible(CpuVisibleNextNormWeights<'a>),
}

impl<'a> ScheduledNextNormWeights<'a> {
    pub(crate) fn none() -> Self {
        Self::None
    }

    pub(crate) fn cpu_visible(
        tensor_name: &'a str,
        values: &'a [f32],
        width: usize,
    ) -> Result<Self> {
        Ok(Self::CpuVisible(CpuVisibleNextNormWeights::new(
            tensor_name,
            values,
            width,
        )?))
    }

    pub(crate) fn is_none(self) -> bool {
        matches!(self, Self::None)
    }

    pub(crate) fn is_cpu_visible(self) -> bool {
        matches!(self, Self::CpuVisible(_))
    }

    pub(crate) fn width(self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::CpuVisible(weights) => Some(weights.width),
        }
    }

    pub(crate) fn values(self) -> Option<&'a [f32]> {
        match self {
            Self::None => None,
            Self::CpuVisible(weights) => Some(weights.values()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedScheduledNextNormWeights {
    tensor_name: Option<String>,
    values: Option<Vec<f32>>,
    width: usize,
}

impl PreparedScheduledNextNormWeights {
    pub(crate) fn none() -> Self {
        Self {
            tensor_name: None,
            values: None,
            width: 0,
        }
    }

    pub(crate) fn cpu_visible(tensor_name: String, values: Vec<f32>, width: usize) -> Result<Self> {
        CpuVisibleNextNormWeights::new(&tensor_name, &values, width)?;
        Ok(Self {
            tensor_name: Some(tensor_name),
            values: Some(values),
            width,
        })
    }

    pub(crate) fn scheduled(&self) -> Result<ScheduledNextNormWeights<'_>> {
        match (self.tensor_name.as_deref(), self.values.as_deref()) {
            (Some(tensor_name), Some(values)) => {
                ScheduledNextNormWeights::cpu_visible(tensor_name, values, self.width)
            }
            (None, None) => Ok(ScheduledNextNormWeights::none()),
            _ => bail!("FlashMoe scheduled next-norm weights have incomplete descriptor state"),
        }
    }
}

pub(crate) fn layer_norm_tensor_name(layer: usize, name: &str) -> String {
    format!("model.layers.{layer}.{name}.weight")
}

pub(crate) fn router_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.gate.weight")
}

pub(crate) fn shared_expert_tensor_name(layer: usize, projection: &str) -> String {
    format!("model.layers.{layer}.mlp.shared_expert.{projection}.weight")
}

pub(crate) fn shared_expert_gate_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.shared_expert_gate.weight")
}

pub(crate) fn attention_tensor_name(layer: usize, projection: &str) -> String {
    format!("model.layers.{layer}.self_attn.{projection}.weight")
}

pub(crate) fn linear_attention_tensor_name(layer: usize, projection: &str) -> String {
    format!("model.layers.{layer}.linear_attn.{projection}.weight")
}

pub(crate) fn linear_attention_scalar_tensor_name(layer: usize, name: &str) -> String {
    format!("model.layers.{layer}.linear_attn.{name}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseProjectionRequest<'a> {
    pub(crate) tensor_name: &'a str,
    pub(crate) output_width: usize,
}

impl<'a> DenseProjectionRequest<'a> {
    fn validate(tensor_name: &str, output_width: usize) -> Result<()> {
        if tensor_name.is_empty() {
            bail!("FlashMoe dense projection request requires a tensor name");
        }
        if output_width == 0 {
            bail!("FlashMoe dense projection request requires non-zero output width");
        }
        Ok(())
    }

    pub(crate) fn new(tensor_name: &'a str, output_width: usize) -> Result<Self> {
        Self::validate(tensor_name, output_width)?;
        Ok(Self {
            tensor_name,
            output_width,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseProjectionRequestGroup<const N: usize> {
    tensor_names: [String; N],
    output_widths: [usize; N],
}

impl<const N: usize> DenseProjectionRequestGroup<N> {
    fn new(tensor_names: [String; N], output_widths: [usize; N]) -> Result<Self> {
        for (tensor_name, output_width) in tensor_names.iter().zip(output_widths.iter().copied()) {
            DenseProjectionRequest::validate(tensor_name, output_width)?;
        }
        Ok(Self {
            tensor_names,
            output_widths,
        })
    }

    pub(crate) fn requests(&self) -> [DenseProjectionRequest<'_>; N] {
        std::array::from_fn(|idx| DenseProjectionRequest {
            tensor_name: &self.tensor_names[idx],
            output_width: self.output_widths[idx],
        })
    }

    #[cfg(test)]
    pub(crate) fn tensor_name(&self, idx: usize) -> &str {
        &self.tensor_names[idx]
    }
}

pub(crate) fn full_attention_input_projection_requests(
    layer: usize,
    q_projection_width: usize,
    kv_width: usize,
) -> Result<DenseProjectionRequestGroup<3>> {
    DenseProjectionRequestGroup::new(
        [
            attention_tensor_name(layer, "q_proj"),
            attention_tensor_name(layer, "k_proj"),
            attention_tensor_name(layer, "v_proj"),
        ],
        [q_projection_width, kv_width, kv_width],
    )
}

pub(crate) fn linear_attention_input_projection_requests(
    layer: usize,
    conv_dim: usize,
    total_value_width: usize,
    num_value_heads: usize,
) -> Result<DenseProjectionRequestGroup<4>> {
    DenseProjectionRequestGroup::new(
        [
            linear_attention_tensor_name(layer, "in_proj_qkv"),
            linear_attention_tensor_name(layer, "in_proj_z"),
            linear_attention_tensor_name(layer, "in_proj_b"),
            linear_attention_tensor_name(layer, "in_proj_a"),
        ],
        [
            conv_dim,
            total_value_width,
            num_value_heads,
            num_value_heads,
        ],
    )
}

pub(crate) fn prepare_scheduled_next_norm_weights<F>(
    layer: usize,
    total_layers: usize,
    width: usize,
    needs_next_layer_norm: bool,
    mut lookup: F,
) -> Result<PreparedScheduledNextNormWeights>
where
    F: FnMut(&str, usize) -> Result<Option<Vec<f32>>>,
{
    if !needs_next_layer_norm || layer + 1 >= total_layers {
        return Ok(PreparedScheduledNextNormWeights::none());
    }

    let name = layer_norm_tensor_name(layer + 1, "input_layernorm");
    let values = lookup(&name, width)?.with_context(|| {
        format!(
            "FlashMoe unsupported scheduled CMD3 path: missing next-layer norm weight {name} for layer {layer}"
        )
    })?;
    PreparedScheduledNextNormWeights::cpu_visible(name, values, width)
}

pub(crate) fn qwen_norm_uses_offset(
    semantics: QwenNormWeightSemantics,
    canonical_name: &str,
) -> bool {
    semantics == QwenNormWeightSemantics::Offset
        && (canonical_name == "model.norm.weight"
            || canonical_name.contains(".input_layernorm.weight")
            || canonical_name.contains(".post_attention_layernorm.weight")
            || canonical_name.contains(".self_attn.q_norm.weight")
            || canonical_name.contains(".self_attn.k_norm.weight"))
}

pub(crate) fn apply_qwen_norm_weight_semantics(
    semantics: QwenNormWeightSemantics,
    canonical_name: &str,
    weight: &mut [f32],
) {
    if qwen_norm_uses_offset(semantics, canonical_name) {
        for value in weight {
            *value += 1.0;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResidentStaticDtype {
    Bf16,
    F16,
    F32,
}

impl ResidentStaticDtype {
    fn from_declared(dtype: &str) -> Option<Self> {
        match dtype.to_ascii_uppercase().as_str() {
            "BF16" | "BFLOAT16" => Some(Self::Bf16),
            "F16" | "FLOAT16" | "FP16" => Some(Self::F16),
            "F32" | "FLOAT32" | "FP32" => Some(Self::F32),
            _ => None,
        }
    }

    pub(crate) const fn as_str(&self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }

    const fn element_size(&self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResidentStaticTensorRef {
    pub(crate) tensor_name: String,
    pub(crate) byte_offset: u64,
    pub(crate) dtype: ResidentStaticDtype,
    pub(crate) values: usize,
    pub(crate) element_size: usize,
}

impl ResidentStaticTensorRef {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        expected_values: usize,
        allowed_dtypes: &[ResidentStaticDtype],
    ) -> Result<Option<Self>> {
        if entry.quantization != TensorQuantization::None {
            return Ok(None);
        }
        let Some(dtype) = ResidentStaticDtype::from_declared(&entry.dtype) else {
            return Ok(None);
        };
        if !allowed_dtypes.contains(&dtype) {
            return Ok(None);
        }
        let element_size = dtype.element_size();
        let expected_bytes = expected_values
            .checked_mul(element_size)
            .context("resident static tensor byte length overflow")?;
        if entry.byte_len as usize != expected_bytes {
            return Ok(None);
        }
        if entry
            .byte_offset
            .checked_add(entry.byte_len)
            .map_or(true, |end| end > store_len)
        {
            return Ok(None);
        }
        if entry.byte_offset % element_size as u64 != 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            tensor_name: tensor_name.to_string(),
            byte_offset: entry.byte_offset,
            dtype,
            values: expected_values,
            element_size,
        }))
    }
}

fn dense_dtype_size(dtype: &str) -> Option<usize> {
    match dtype.to_ascii_uppercase().as_str() {
        "BF16" | "BFLOAT16" | "F16" | "FLOAT16" | "FP16" => Some(2),
        "F32" | "FLOAT32" | "FP32" => Some(4),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseMmapMatvecProjection {
    pub(crate) tensor_name: String,
    pub(crate) byte_offset: u64,
    pub(crate) dtype: String,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) output_width: usize,
}

impl DenseMmapMatvecProjection {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        output_width: usize,
        input_len: usize,
        element_size: usize,
    ) -> Result<Self> {
        let (rows, cols) =
            validate_dense_matvec_shape(entry, tensor_name, output_width, input_len)?;
        let row_bytes = cols
            .checked_mul(element_size)
            .context("dense tensor resident row byte length overflow")?;
        let byte_len = rows
            .checked_mul(row_bytes)
            .context("dense tensor resident byte length overflow")?;
        if entry
            .byte_offset
            .checked_add(byte_len as u64)
            .map_or(true, |end| end > store_len)
        {
            bail!(
                "Flash-MoE dense tensor {} byte range {}..{} exceeds dense store length {}",
                tensor_name,
                entry.byte_offset,
                entry.byte_offset.saturating_add(byte_len as u64),
                store_len
            );
        }
        Ok(Self {
            tensor_name: tensor_name.to_string(),
            byte_offset: entry.byte_offset,
            dtype: entry.dtype.clone(),
            rows,
            cols,
            output_width,
        })
    }

    #[cfg(test)]
    pub(crate) fn stride(&self) -> usize {
        self.cols
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseQ4MmapMatvecProjection {
    pub(crate) tensor_name: String,
    pub(crate) packed_byte_offset: u64,
    pub(crate) scales_byte_offset: u64,
    pub(crate) biases_byte_offset: u64,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) output_width: usize,
    pub(crate) row_packed_bytes: usize,
    pub(crate) groups_per_row: usize,
    pub(crate) group_size: usize,
    pub(crate) scale_bias_dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DenseQ4ProjectionKey {
    pub(crate) name: String,
    pub(crate) output_width: usize,
    pub(crate) input_len: usize,
}

impl DenseQ4ProjectionKey {
    pub(crate) fn new(name: &str, output_width: usize, input_len: usize) -> Self {
        Self {
            name: name.to_string(),
            output_width,
            input_len,
        }
    }
}

impl DenseQ4MmapMatvecProjection {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        output_width: usize,
        input_len: usize,
    ) -> Result<Option<Self>> {
        let TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } = &entry.quantization
        else {
            return Ok(None);
        };
        let (rows, cols) =
            validate_dense_matvec_shape(entry, tensor_name, output_width, input_len)?;
        Self::from_validated_entry(
            tensor_name,
            entry,
            store_len,
            rows,
            cols,
            output_width,
            *group_size,
            scale_bias_dtype,
        )
    }

    pub(crate) fn from_multilinear_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        heads: usize,
        output_width_per_head: usize,
        input_len: usize,
    ) -> Result<Option<Self>> {
        let TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } = &entry.quantization
        else {
            return Ok(None);
        };
        let expected_shape = [heads, output_width_per_head, input_len];
        if entry.shape.as_slice() != expected_shape {
            bail!(
                "Flash-MoE dense tensor {tensor_name} shape mismatch: expected multilinear shape {:?}, actual shape {:?}",
                expected_shape,
                entry.shape
            );
        }
        let rows = heads
            .checked_mul(output_width_per_head)
            .context("dense Q4 multilinear row count overflow")?;
        Self::from_validated_entry(
            tensor_name,
            entry,
            store_len,
            rows,
            input_len,
            rows,
            *group_size,
            scale_bias_dtype,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_validated_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        rows: usize,
        cols: usize,
        output_width: usize,
        group_size: usize,
        scale_bias_dtype: &str,
    ) -> Result<Option<Self>> {
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&entry.shape, group_size, scale_bias_dtype)?;
        if entry.byte_len as usize != layout.total_bytes
            || rows != layout.rows
            || cols != layout.cols
        {
            return Ok(None);
        }
        if entry
            .byte_offset
            .checked_add(entry.byte_len)
            .map_or(true, |end| end > store_len)
        {
            return Ok(None);
        }
        let packed_byte_offset = entry.byte_offset;
        let scales_byte_offset = entry
            .byte_offset
            .checked_add(layout.packed_bytes as u64)
            .context("dense q4 projection scales offset overflow")?;
        let biases_byte_offset = scales_byte_offset
            .checked_add(layout.scales_bytes as u64)
            .context("dense q4 projection biases offset overflow")?;
        Ok(Some(Self {
            tensor_name: tensor_name.to_string(),
            packed_byte_offset,
            scales_byte_offset,
            biases_byte_offset,
            rows,
            cols,
            output_width,
            row_packed_bytes: layout.row_packed_bytes,
            groups_per_row: layout.groups_per_row,
            group_size,
            scale_bias_dtype: scale_bias_dtype.to_string(),
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResidentMmapMatvecProjection {
    Q4(DenseQ4MmapMatvecProjection),
    Dense(DenseMmapMatvecProjection),
}

impl From<DenseQ4MmapMatvecProjection> for ResidentMmapMatvecProjection {
    fn from(projection: DenseQ4MmapMatvecProjection) -> Self {
        Self::Q4(projection)
    }
}

impl ResidentMmapMatvecProjection {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        output_width: usize,
        input_len: usize,
    ) -> Result<Self> {
        match &entry.quantization {
            TensorQuantization::Q4 { .. } => {
                let projection = DenseQ4MmapMatvecProjection::from_entry(
                    tensor_name,
                    entry,
                    store_len,
                    output_width,
                    input_len,
                )?
                .with_context(|| {
                    format!(
                        "resident Q4 projection {tensor_name} does not match its declared shape or byte range"
                    )
                })?;
                Ok(Self::Q4(projection))
            }
            TensorQuantization::None => {
                let element_size = dense_dtype_size(&entry.dtype).with_context(|| {
                    format!(
                        "resident dense projection {tensor_name} has unsupported dtype {}",
                        entry.dtype
                    )
                })?;
                Ok(Self::Dense(DenseMmapMatvecProjection::from_entry(
                    tensor_name,
                    entry,
                    store_len,
                    output_width,
                    input_len,
                    element_size,
                )?))
            }
        }
    }

    pub(crate) fn tensor_name(&self) -> &str {
        match self {
            Self::Q4(projection) => &projection.tensor_name,
            Self::Dense(projection) => &projection.tensor_name,
        }
    }

    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Q4(projection) => projection.rows,
            Self::Dense(projection) => projection.rows,
        }
    }

    pub(crate) fn cols(&self) -> usize {
        match self {
            Self::Q4(projection) => projection.cols,
            Self::Dense(projection) => projection.cols,
        }
    }

    pub(crate) fn output_width(&self) -> usize {
        match self {
            Self::Q4(projection) => projection.output_width,
            Self::Dense(projection) => projection.output_width,
        }
    }

    pub(crate) fn q4(&self) -> Option<&DenseQ4MmapMatvecProjection> {
        match self {
            Self::Q4(projection) => Some(projection),
            Self::Dense(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LinearAttentionStaticBindings {
    pub(crate) conv_weight: ResidentStaticTensorRef,
    pub(crate) a_log: ResidentStaticTensorRef,
    pub(crate) dt_bias: ResidentStaticTensorRef,
    pub(crate) norm_weight: ResidentStaticTensorRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LinearAttentionResidentBindings {
    pub(crate) layer: usize,
    pub(crate) input_projections: [ResidentMmapMatvecProjection; 4],
    pub(crate) static_tensors: LinearAttentionStaticBindings,
    pub(crate) out_proj: ResidentMmapMatvecProjection,
    pub(crate) router: ResidentMmapMatvecProjection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LinearAttentionWeightTable {
    layers: Vec<Option<LinearAttentionResidentBindings>>,
}

impl LinearAttentionWeightTable {
    pub(crate) fn require(&self, layer: usize) -> Result<&LinearAttentionResidentBindings> {
        self.layers
            .get(layer)
            .and_then(Option::as_ref)
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled linear-attention weight path: layer {layer} has no resolved resident bindings"
                )
            })
    }

    #[cfg(test)]
    pub(crate) fn layer(&self, layer: usize) -> Option<&LinearAttentionResidentBindings> {
        self.layers.get(layer).and_then(Option::as_ref)
    }
}

pub(crate) fn build_dense_q4_mmap_projection<'a, F>(
    tensor_name: &str,
    output_width: usize,
    input_len: usize,
    store_len: u64,
    mut lookup: F,
) -> Result<Option<DenseQ4MmapMatvecProjection>>
where
    F: FnMut(&str) -> Option<&'a RuntimeTensorEntry>,
{
    let Some(entry) = lookup(tensor_name) else {
        return Ok(None);
    };
    DenseQ4MmapMatvecProjection::from_entry(tensor_name, entry, store_len, output_width, input_len)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SharedExpertPhaseShape {
    pub(crate) width: usize,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) total_intermediate: usize,
}

impl SharedExpertPhaseShape {
    pub(crate) fn new(width: usize, shared_experts: usize, intermediate: usize) -> Result<Self> {
        if width == 0 || shared_experts == 0 || intermediate == 0 {
            bail!(
                "shared expert graph shape requires non-zero width, shared expert count, and intermediate width"
            );
        }
        let total_intermediate = shared_experts
            .checked_mul(intermediate)
            .context("shared expert intermediate width overflow")?;
        Ok(Self {
            width,
            shared_experts,
            intermediate,
            total_intermediate,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SharedExpertPhaseWeights {
    pub(crate) gate: Arc<Vec<f32>>,
    pub(crate) up: Arc<Vec<f32>>,
    pub(crate) down: Arc<Vec<f32>>,
    pub(crate) router: Arc<Vec<f32>>,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) width: usize,
}

impl SharedExpertPhaseWeights {
    #[cfg(test)]
    pub(crate) fn new(
        gate: Arc<Vec<f32>>,
        up: Arc<Vec<f32>>,
        down: Arc<Vec<f32>>,
        router: Arc<Vec<f32>>,
        shared_experts: usize,
        intermediate: usize,
        width: usize,
    ) -> Result<Self> {
        let weights = Self {
            gate,
            up,
            down,
            router,
            shared_experts,
            intermediate,
            width,
        };
        weights.validated_shape()?;
        Ok(weights)
    }

    pub(crate) fn validated_shape(&self) -> Result<SharedExpertPhaseShape> {
        let shape =
            SharedExpertPhaseShape::new(self.width, self.shared_experts, self.intermediate)?;
        let dense_len = shape
            .total_intermediate
            .checked_mul(shape.width)
            .context("shared expert dense projection width overflow")?;
        let router_len = shape
            .shared_experts
            .checked_mul(shape.width)
            .context("shared expert router projection width overflow")?;
        if self.gate.len() != dense_len
            || self.up.len() != dense_len
            || self.down.len() != dense_len
            || self.router.len() != router_len
        {
            bail!(
                "FlashMoe scheduled shared dense expert shape is invalid: width={} shared_experts={} intermediate={} gate={} up={} down={} router={}",
                self.width,
                self.shared_experts,
                self.intermediate,
                self.gate.len(),
                self.up.len(),
                self.down.len(),
                self.router.len()
            );
        }
        Ok(shape)
    }
}

#[cfg(test)]
pub(crate) fn build_shared_expert_phase_weights<F>(
    layer: usize,
    width: usize,
    shared_experts: usize,
    intermediate: usize,
    mut lookup: F,
) -> Result<Option<SharedExpertPhaseWeights>>
where
    F: FnMut(&str) -> Result<Option<Arc<Vec<f32>>>>,
{
    if shared_experts == 0 || intermediate == 0 {
        return Ok(None);
    }
    SharedExpertPhaseShape::new(width, shared_experts, intermediate)?;

    let gate_name = shared_expert_tensor_name(layer, "gate_proj");
    let up_name = shared_expert_tensor_name(layer, "up_proj");
    let down_name = shared_expert_tensor_name(layer, "down_proj");
    let shared_gate_name = shared_expert_gate_tensor_name(layer);
    let gate = lookup(&gate_name)?
        .with_context(|| format!("missing configured shared expert tensor {gate_name}"))?;
    let up = lookup(&up_name)?
        .with_context(|| format!("missing configured shared expert tensor {up_name}"))?;
    let down = lookup(&down_name)?
        .with_context(|| format!("missing configured shared expert tensor {down_name}"))?;
    let router = lookup(&shared_gate_name)?.with_context(|| {
        format!("missing configured shared expert gate tensor {shared_gate_name}")
    })?;

    SharedExpertPhaseWeights::new(gate, up, down, router, shared_experts, intermediate, width)
        .map(Some)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SharedExpertPhaseResidentProjections {
    pub(crate) gate: ResidentMmapMatvecProjection,
    pub(crate) up: ResidentMmapMatvecProjection,
    pub(crate) down: ResidentMmapMatvecProjection,
    pub(crate) router: Option<ResidentMmapMatvecProjection>,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) width: usize,
}

impl SharedExpertPhaseResidentProjections {
    pub(crate) fn validated_shape(&self) -> Result<SharedExpertPhaseShape> {
        let shape =
            SharedExpertPhaseShape::new(self.width, self.shared_experts, self.intermediate)?;
        if self.gate.cols() != shape.width
            || self.up.cols() != shape.width
            || self
                .router
                .as_ref()
                .is_some_and(|router| router.cols() != shape.width)
            || self.down.cols() != shape.total_intermediate
            || self.gate.output_width() != shape.total_intermediate
            || self.up.output_width() != shape.total_intermediate
            || self.down.output_width() != shape.width
            || self
                .router
                .as_ref()
                .is_some_and(|router| router.output_width() != shape.shared_experts)
        {
            bail!(
                "FlashMoe scheduled resident shared-expert shape is invalid: width={} shared_experts={} intermediate={} gate=({},{}) up=({},{}) down=({},{}) router=({},{})",
                self.width,
                self.shared_experts,
                self.intermediate,
                self.gate.output_width(),
                self.gate.cols(),
                self.up.output_width(),
                self.up.cols(),
                self.down.output_width(),
                self.down.cols(),
                self.router
                    .as_ref()
                    .map_or(0, |router| router.output_width()),
                self.router.as_ref().map_or(0, |router| router.cols())
            );
        }
        Ok(shape)
    }
}

#[cfg(test)]
pub(crate) fn build_shared_expert_resident_phase_projections<F, P>(
    layer: usize,
    width: usize,
    shared_experts: usize,
    intermediate: usize,
    mut projection: F,
) -> Result<Option<SharedExpertPhaseResidentProjections>>
where
    F: FnMut(&str, usize, usize) -> Result<Option<P>>,
    P: Into<ResidentMmapMatvecProjection>,
{
    if width == 0 || shared_experts == 0 || intermediate == 0 {
        return Ok(None);
    }
    let shape = SharedExpertPhaseShape::new(width, shared_experts, intermediate)?;

    let gate_name = shared_expert_tensor_name(layer, "gate_proj");
    let up_name = shared_expert_tensor_name(layer, "up_proj");
    let down_name = shared_expert_tensor_name(layer, "down_proj");
    let router_name = shared_expert_gate_tensor_name(layer);
    let Some(gate) = projection(&gate_name, shape.total_intermediate, shape.width)? else {
        return Ok(None);
    };
    let Some(up) = projection(&up_name, shape.total_intermediate, shape.width)? else {
        return Ok(None);
    };
    let Some(down) = projection(&down_name, shape.width, shape.total_intermediate)? else {
        return Ok(None);
    };
    let Some(router) = projection(&router_name, shape.shared_experts, shape.width)? else {
        return Ok(None);
    };

    let shared = SharedExpertPhaseResidentProjections {
        gate: gate.into(),
        up: up.into(),
        down: down.into(),
        router: Some(router.into()),
        shared_experts,
        intermediate,
        width,
    };
    shared.validated_shape()?;
    Ok(Some(shared))
}

#[cfg(test)]
pub(crate) fn build_required_shared_expert_resident_phase_projections<F, P>(
    layer: usize,
    width: usize,
    shared_experts: usize,
    intermediate: usize,
    projection: F,
) -> Result<Option<SharedExpertPhaseResidentProjections>>
where
    F: FnMut(&str, usize, usize) -> Result<Option<P>>,
    P: Into<ResidentMmapMatvecProjection>,
{
    build_required_shared_expert_resident_phase_projections_with_router(
        layer,
        width,
        shared_experts,
        intermediate,
        true,
        projection,
    )
}

pub(crate) fn build_required_shared_expert_resident_phase_projections_with_router<F, P>(
    layer: usize,
    width: usize,
    shared_experts: usize,
    intermediate: usize,
    requires_router: bool,
    mut projection: F,
) -> Result<Option<SharedExpertPhaseResidentProjections>>
where
    F: FnMut(&str, usize, usize) -> Result<Option<P>>,
    P: Into<ResidentMmapMatvecProjection>,
{
    if shared_experts == 0 {
        return Ok(None);
    }
    let shape = SharedExpertPhaseShape::new(width, shared_experts, intermediate)?;

    let gate_name = shared_expert_tensor_name(layer, "gate_proj");
    let up_name = shared_expert_tensor_name(layer, "up_proj");
    let down_name = shared_expert_tensor_name(layer, "down_proj");
    let router_name = shared_expert_gate_tensor_name(layer);
    let gate = projection(&gate_name, shape.total_intermediate, shape.width)?.ok_or_else(|| {
        anyhow::anyhow!(
            "FlashMoe unsupported scheduled CMD3 shared-expert path: missing resident shared gate projection {gate_name}"
        )
    })?;
    let up = projection(&up_name, shape.total_intermediate, shape.width)?.ok_or_else(|| {
        anyhow::anyhow!(
            "FlashMoe unsupported scheduled CMD3 shared-expert path: missing resident shared up projection {up_name}"
        )
    })?;
    let down = projection(&down_name, shape.width, shape.total_intermediate)?.ok_or_else(|| {
        anyhow::anyhow!(
            "FlashMoe unsupported scheduled CMD3 shared-expert path: missing resident shared down projection {down_name}"
        )
    })?;
    let router = if requires_router {
        Some(
            projection(&router_name, shape.shared_experts, shape.width)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "FlashMoe unsupported scheduled CMD3 shared-expert path: missing resident shared router projection {router_name}"
                )
            })?,
        )
    } else {
        None
    };

    let shared = SharedExpertPhaseResidentProjections {
        gate: gate.into(),
        up: up.into(),
        down: down.into(),
        router: router.map(Into::into),
        shared_experts,
        intermediate,
        width,
    };
    shared.validated_shape()?;
    Ok(Some(shared))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SharedExpertLayerWeights {
    None,
    Resident(SharedExpertPhaseResidentProjections),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SharedExpertWeightTable {
    layers: Vec<SharedExpertLayerWeights>,
}

impl SharedExpertWeightTable {
    pub(crate) fn layer(&self, layer: usize) -> Result<&SharedExpertLayerWeights> {
        self.layers.get(layer).with_context(|| {
            format!("FlashMoe scheduled shared-expert weight table has no entry for layer {layer}")
        })
    }
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct SharedExpertPhaseCache {
    dense: Mutex<BTreeMap<usize, Arc<SharedExpertPhaseWeights>>>,
}

#[cfg(test)]
impl SharedExpertPhaseCache {
    pub(crate) fn dense<F>(
        &self,
        layer: usize,
        width: usize,
        shared_experts: usize,
        intermediate: usize,
        lookup: F,
    ) -> Result<Option<Arc<SharedExpertPhaseWeights>>>
    where
        F: FnMut(&str) -> Result<Option<Arc<Vec<f32>>>>,
    {
        if shared_experts == 0 || intermediate == 0 {
            return Ok(None);
        }
        if let Some(shared) = self.cached_dense(layer, width)? {
            return Ok(Some(shared));
        }

        let Some(shared) =
            build_shared_expert_phase_weights(layer, width, shared_experts, intermediate, lookup)?
        else {
            return Ok(None);
        };
        let shared = Arc::new(shared);
        let mut cache = self.dense.lock().expect("shared expert cache poisoned");
        if let Some(existing) = cache.get(&layer).cloned() {
            validate_cached_shared_width("shared expert tensors", layer, existing.width, width)?;
            Ok(Some(existing))
        } else {
            cache.insert(layer, shared.clone());
            Ok(Some(shared))
        }
    }

    fn cached_dense(
        &self,
        layer: usize,
        width: usize,
    ) -> Result<Option<Arc<SharedExpertPhaseWeights>>> {
        let cache = self.dense.lock().expect("shared expert cache poisoned");
        let Some(shared) = cache.get(&layer).cloned() else {
            return Ok(None);
        };
        validate_cached_shared_width("shared expert tensors", layer, shared.width, width)?;
        Ok(Some(shared))
    }
}

#[cfg(test)]
fn validate_cached_shared_width(
    label: &str,
    layer: usize,
    cached_width: usize,
    requested_width: usize,
) -> Result<()> {
    if cached_width != requested_width {
        bail!(
            "cached {label} for layer {layer} have width {cached_width}, requested {requested_width}"
        );
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn dense_projection_tile_rows(cols: usize, rows: usize) -> usize {
    let bytes_per_row = cols.saturating_mul(std::mem::size_of::<f32>()).max(1);
    (DENSE_PROJECTION_TILE_BYTES / bytes_per_row)
        .max(1)
        .min(rows.max(1))
}

#[cfg(test)]
fn dense_mmap_dtype_supported(dtype: &str) -> bool {
    matches!(
        dtype.to_ascii_uppercase().as_str(),
        "F32" | "FLOAT32" | "FP32" | "BF16" | "BFLOAT16"
    )
}

#[cfg(test)]
pub(super) fn dense_projection_tile_rows_for_metal(
    dtype: &str,
    cols: usize,
    rows: usize,
    resident_mmap_available: bool,
) -> usize {
    if resident_mmap_available && dense_mmap_dtype_supported(dtype) {
        rows.max(1)
    } else {
        dense_projection_tile_rows(cols, rows)
    }
}

fn validate_lm_head_matvec_shape(
    entry: &RuntimeTensorEntry,
    canonical_name: &str,
    vocab_size: usize,
    input_len: usize,
) -> Result<(usize, usize)> {
    let expected_shape = [vocab_size, input_len];
    match entry.shape.as_slice() {
        [rows, cols] if *rows >= vocab_size && *cols == input_len => Ok((*rows, *cols)),
        _ => bail!(
            "Flash-MoE dense tensor {canonical_name} shape mismatch: expected at least {:?}, actual shape {:?}, input length {input_len}",
            expected_shape,
            entry.shape
        ),
    }
}

#[derive(Debug, Clone)]
pub struct DenseStore {
    #[cfg(test)]
    manifest_path: PathBuf,
    pub(super) len: u64,
    pub(super) mmap: Arc<memmap2::Mmap>,
    registry: TensorRegistry,
    pub(super) resident: Arc<std::sync::Mutex<DenseTensorCache>>,
    pub(super) norm_weights: Arc<std::sync::Mutex<BTreeMap<DenseNormWeightKey, Arc<Vec<f32>>>>>,
    q4_mmap_projections:
        Arc<std::sync::Mutex<BTreeMap<DenseQ4ProjectionKey, Arc<DenseQ4MmapMatvecProjection>>>>,
    pub(super) decoded_tiles: Arc<std::sync::Mutex<DenseTensorTileCache>>,
    #[cfg(test)]
    #[allow(dead_code)]
    raw_tiles: Arc<std::sync::Mutex<DenseRawTensorTileCache>>,
    #[cfg(test)]
    decoded_full_tensors: Arc<std::sync::atomic::AtomicUsize>,
    #[cfg(test)]
    pub(super) decoded_tensor_tiles: Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Debug, Default)]
pub(super) struct DenseTensorCache {
    tensors: BTreeMap<String, Arc<Vec<f32>>>,
    pub(super) bytes: usize,
    max_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DenseTensorTileKey {
    name: String,
    start_row: usize,
    row_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct DenseNormWeightKey {
    pub(super) name: String,
    pub(super) width: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DenseTileReadTiming {
    pub(super) total: Duration,
    pub(super) read_range: Duration,
    pub(super) decode: Duration,
    pub(super) cache_insert: Duration,
    pub(super) cache_evict: Duration,
    pub(super) cache_hits: u64,
    pub(super) cache_misses: u64,
    pub(super) cache_inserts: u64,
    pub(super) cache_evictions: u64,
    pub(super) bytes_read: u64,
    pub(super) decoded_bytes: u64,
}

impl DenseTileReadTiming {
    fn add(&mut self, other: Self) {
        self.total += other.total;
        self.read_range += other.read_range;
        self.decode += other.decode;
        self.cache_insert += other.cache_insert;
        self.cache_evict += other.cache_evict;
        self.cache_hits = self.cache_hits.saturating_add(other.cache_hits);
        self.cache_misses = self.cache_misses.saturating_add(other.cache_misses);
        self.cache_inserts = self.cache_inserts.saturating_add(other.cache_inserts);
        self.cache_evictions = self.cache_evictions.saturating_add(other.cache_evictions);
        self.bytes_read = self.bytes_read.saturating_add(other.bytes_read);
        self.decoded_bytes = self.decoded_bytes.saturating_add(other.decoded_bytes);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct DenseTensorTileCacheInsertStats {
    inserts: u64,
    evictions: u64,
    insert_time: Duration,
    evict_time: Duration,
}

#[derive(Debug, Default)]
pub(super) struct DenseTensorTileCache {
    tiles: BTreeMap<DenseTensorTileKey, Arc<Vec<f32>>>,
    pub(super) bytes: usize,
    max_bytes: usize,
}

#[derive(Debug, Default)]
#[cfg(test)]
#[allow(dead_code)]
struct DenseRawTensorTileCache {
    tiles: BTreeMap<DenseTensorTileKey, Arc<Vec<u8>>>,
    bytes: usize,
    max_bytes: usize,
}

impl DenseTensorTileCache {
    fn with_budget(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            ..Self::default()
        }
    }

    fn get(&self, key: &DenseTensorTileKey) -> Option<Arc<Vec<f32>>> {
        self.tiles.get(key).cloned()
    }

    fn insert(
        &mut self,
        key: DenseTensorTileKey,
        tile: Arc<Vec<f32>>,
    ) -> DenseTensorTileCacheInsertStats {
        let bytes = tile.len() * std::mem::size_of::<f32>();
        if bytes == 0 || bytes > self.max_bytes {
            return DenseTensorTileCacheInsertStats::default();
        }
        if let Some(previous) = self.tiles.remove(&key) {
            self.bytes = self
                .bytes
                .saturating_sub(previous.len() * std::mem::size_of::<f32>());
        }
        let evict_started = Instant::now();
        let mut evictions = 0u64;
        while self.bytes.saturating_add(bytes) > self.max_bytes && !self.tiles.is_empty() {
            let Some(victim) = self.tiles.keys().next().cloned() else {
                break;
            };
            if let Some(previous) = self.tiles.remove(&victim) {
                self.bytes = self
                    .bytes
                    .saturating_sub(previous.len() * std::mem::size_of::<f32>());
                evictions = evictions.saturating_add(1);
            }
        }
        let evict_time = if evictions > 0 {
            evict_started.elapsed()
        } else {
            Duration::ZERO
        };

        let insert_started = Instant::now();
        self.tiles.insert(key, tile);
        self.bytes = self.bytes.saturating_add(bytes);
        DenseTensorTileCacheInsertStats {
            inserts: 1,
            evictions,
            insert_time: insert_started.elapsed(),
            evict_time,
        }
    }
}

#[cfg(test)]
#[allow(dead_code)]
impl DenseRawTensorTileCache {
    fn with_budget(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            ..Self::default()
        }
    }

    fn get(&self, key: &DenseTensorTileKey) -> Option<Arc<Vec<u8>>> {
        self.tiles.get(key).cloned()
    }

    fn insert(
        &mut self,
        key: DenseTensorTileKey,
        tile: Arc<Vec<u8>>,
    ) -> DenseTensorTileCacheInsertStats {
        let bytes = tile.len();
        if bytes == 0 || bytes > self.max_bytes {
            return DenseTensorTileCacheInsertStats::default();
        }
        if let Some(previous) = self.tiles.remove(&key) {
            self.bytes = self.bytes.saturating_sub(previous.len());
        }
        let evict_started = Instant::now();
        let mut evictions = 0u64;
        while self.bytes.saturating_add(bytes) > self.max_bytes && !self.tiles.is_empty() {
            let Some(victim) = self.tiles.keys().next().cloned() else {
                break;
            };
            if let Some(previous) = self.tiles.remove(&victim) {
                self.bytes = self.bytes.saturating_sub(previous.len());
                evictions = evictions.saturating_add(1);
            }
        }
        let evict_time = if evictions > 0 {
            evict_started.elapsed()
        } else {
            Duration::ZERO
        };

        let insert_started = Instant::now();
        self.tiles.insert(key, tile);
        self.bytes = self.bytes.saturating_add(bytes);
        DenseTensorTileCacheInsertStats {
            inserts: 1,
            evictions,
            insert_time: insert_started.elapsed(),
            evict_time,
        }
    }
}

impl DenseTensorCache {
    fn with_budget(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            ..Self::default()
        }
    }

    fn get(&self, name: &str) -> Option<Arc<Vec<f32>>> {
        self.tensors.get(name).cloned()
    }

    fn insert(&mut self, name: String, tensor: Arc<Vec<f32>>) {
        let bytes = tensor.len() * std::mem::size_of::<f32>();
        if bytes > self.max_bytes {
            return;
        }
        while self.bytes.saturating_add(bytes) > self.max_bytes && !self.tensors.is_empty() {
            let Some(victim) = self.tensors.keys().next().cloned() else {
                break;
            };
            if let Some(previous) = self.tensors.remove(&victim) {
                self.bytes = self
                    .bytes
                    .saturating_sub(previous.len() * std::mem::size_of::<f32>());
            }
        }
        if let Some(previous) = self.tensors.insert(name, tensor) {
            self.bytes = self
                .bytes
                .saturating_sub(previous.len() * std::mem::size_of::<f32>());
        }
        self.bytes = self.bytes.saturating_add(bytes);
    }
}

impl DenseStore {
    pub fn open(path: PathBuf, manifest_path: PathBuf) -> Result<Self> {
        let file = fs::File::open(&path)
            .with_context(|| format!("failed to open dense store {}", path.display()))?;
        let len = file
            .metadata()
            .with_context(|| format!("failed to stat dense store {}", path.display()))?
            .len();
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .with_context(|| format!("failed to memory-map dense store {}", path.display()))?
        };
        let registry = TensorRegistry::load(&manifest_path)?;
        Ok(Self {
            #[cfg(test)]
            manifest_path,
            len,
            mmap: Arc::new(mmap),
            registry,
            resident: Arc::new(std::sync::Mutex::new(DenseTensorCache::with_budget(
                512 * 1024 * 1024,
            ))),
            norm_weights: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            q4_mmap_projections: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            decoded_tiles: Arc::new(std::sync::Mutex::new(DenseTensorTileCache::with_budget(
                DENSE_DECODED_TILE_CACHE_BYTES,
            ))),
            #[cfg(test)]
            raw_tiles: Arc::new(std::sync::Mutex::new(DenseRawTensorTileCache::with_budget(
                DENSE_DECODED_TILE_CACHE_BYTES,
            ))),
            #[cfg(test)]
            decoded_full_tensors: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            #[cfg(test)]
            decoded_tensor_tiles: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    pub fn registry(&self) -> &TensorRegistry {
        &self.registry
    }

    #[cfg(test)]
    pub(super) fn q4_mmap_projection_cache_len(&self) -> usize {
        self.q4_mmap_projections
            .lock()
            .expect("dense q4 projection cache poisoned")
            .len()
    }

    pub(super) fn seed(&self, position: usize, previous: u32) -> Result<u64> {
        Ok(self
            .read_u64(position as u64)?
            .wrapping_add(u64::from(previous)))
    }

    pub(super) fn embedding(&self, token: u32, width: usize) -> Result<Vec<f32>> {
        if let Some(row) =
            self.read_tensor_row_f32("model.embed_tokens.weight", token as usize, width)?
        {
            return Ok(row);
        }
        bail!(
            "Flash-MoE dense tensor registry cannot provide model.embed_tokens.weight row for token {token}; refusing synthetic embeddings"
        )
    }

    #[cfg(test)]
    pub(super) fn project(
        &self,
        layer: usize,
        name: &str,
        input: &[f32],
        width: usize,
    ) -> Result<Vec<f32>> {
        let tensor_name = attention_tensor_name(layer, name);
        if let Some(projected) = self.matvec_tensor_prefix(&tensor_name, input, width)? {
            return Ok(projected);
        }
        let salt = self.tensor_seed(&tensor_name, stable_hash(name) ^ ((layer as u64) << 32));
        let mut out = vec![0.0f32; width];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (col, value) in input.iter().enumerate() {
                let bits = self.read_u64(salt ^ ((row as u64) << 20) ^ col as u64)?;
                let weight = ((bits >> 40) as f32 / ((1u64 << 24) as f32)) * 2.0 - 1.0;
                acc = value.mul_add(weight, acc);
            }
            *slot = acc / (input.len().max(1) as f32).sqrt();
        }
        Ok(out)
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(super) fn project_with_metal(
        &self,
        metal: Option<&MetalExecutionFacade>,
        layer: usize,
        name: &str,
        input: &[f32],
        width: usize,
    ) -> Result<Vec<f32>> {
        let tensor_name = attention_tensor_name(layer, name);
        if let Some(entry) = self.registry.tensor(&tensor_name) {
            let (rows, cols) =
                validate_dense_matvec_shape(entry, &tensor_name, width, input.len())?;
            if let Some(metal) = metal {
                return self.metal_matvec_tiled(metal, &tensor_name, input, rows, cols, width);
            }
            if let TensorQuantization::Q4 { .. } = entry.quantization {
                return self.q4_matvec_tiled(&tensor_name, input, rows, cols, width);
            }
        }
        self.project(layer, name, input, width)
    }
    pub(super) fn project_resident_tensors_from_cpu_input(
        &self,
        metal: &MetalExecutionFacade,
        specs: &[DenseProjectionRequest<'_>],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        if specs.is_empty() {
            bail!("FlashMoe scheduled resident projection batch has no projections");
        }
        metal.require_resident_dense_weights()?;
        let mut projections = Vec::with_capacity(specs.len());
        for spec in specs {
            let projection = self
                .resident_mmap_projection(spec.tensor_name, spec.output_width, input.len())?
                .with_context(|| {
                    format!(
                        "FlashMoe unsupported scheduled resident projection batch: missing projection {}",
                        spec.tensor_name
                    )
                })?;
            projections.push(projection);
        }
        let (outputs, _, _) = metal.resident_mmap_matvec_batch(&projections, input)?;
        Ok(outputs)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_mla_input_projections_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        layer: usize,
        layout: MlaAttentionLayout,
        input: MetalBatchProjectionInput<'_>,
        q_norm_weight: &[f32],
        kv_norm_weight: &[f32],
        norm_epsilon: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let q_a_name = attention_tensor_name(layer, "q_a_proj");
        let kv_a_name = attention_tensor_name(layer, "kv_a_proj_with_mqa");
        let q_b_name = attention_tensor_name(layer, "q_b_proj");
        let q_a = self
            .resident_mmap_projection(&q_a_name, layout.q_lora_rank, input.len())?
            .with_context(|| format!("missing resident GLM MLA projection {q_a_name}"))?;
        let kv_a = self
            .resident_mmap_projection(&kv_a_name, layout.kv_a_width, input.len())?
            .with_context(|| format!("missing resident GLM MLA projection {kv_a_name}"))?;
        let q_b = self
            .resident_mmap_projection(&q_b_name, layout.q_width, layout.q_lora_rank)?
            .with_context(|| format!("missing resident GLM MLA projection {q_b_name}"))?;
        metal.resident_glm_mla_input_projection_chain(
            &q_a,
            &kv_a,
            &q_b,
            input,
            q_norm_weight,
            kv_norm_weight,
            layout.kv_lora_rank,
            norm_epsilon,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_mla_fused_attention_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        layer: usize,
        layout: MlaAttentionLayout,
        input: MetalBatchProjectionInput<'_>,
        q_norm_weight: &[f32],
        kv_norm_weight: &[f32],
        norm_epsilon: f32,
        previous_records: &[(&[f32], &[f32])],
        rope_cos: &[f32],
        rope_sin: &[f32],
        post_attention: Option<GlmMlaPostAttentionRequest<'_>>,
    ) -> Result<MetalGlmMlaFusedAttentionOutput> {
        if layout.kv_projection != MlaKvProjectionLayout::AbsorbedMultiLinear {
            bail!(
                "MLA layer {layer} fused Metal execution requires pre-absorbed embed_q/unembed_out weights"
            );
        }
        let q_a_name = attention_tensor_name(layer, "q_a_proj");
        let kv_a_name = attention_tensor_name(layer, "kv_a_proj_with_mqa");
        let q_b_name = attention_tensor_name(layer, "q_b_proj");
        let embed_q_name = attention_tensor_name(layer, "embed_q");
        let unembed_out_name = attention_tensor_name(layer, "unembed_out");
        let q_a = self
            .resident_mmap_projection(&q_a_name, layout.q_lora_rank, input.len())?
            .with_context(|| format!("missing resident GLM MLA projection {q_a_name}"))?;
        let kv_a = self
            .resident_mmap_projection(&kv_a_name, layout.kv_a_width, input.len())?
            .with_context(|| format!("missing resident GLM MLA projection {kv_a_name}"))?;
        let q_b = self
            .resident_mmap_projection(&q_b_name, layout.q_width, layout.q_lora_rank)?
            .with_context(|| format!("missing resident GLM MLA projection {q_b_name}"))?;
        let embed_q = self
            .dense_q4_mmap_multilinear_projection(
                &embed_q_name,
                layout.num_heads,
                layout.kv_lora_rank,
                layout.qk_nope_head_dim,
            )?
            .with_context(|| {
                format!(
                    "GLM MLA fused Metal execution requires resident Q4 projection {embed_q_name}"
                )
            })?;
        let unembed_out = self
            .dense_q4_mmap_multilinear_projection(
                &unembed_out_name,
                layout.num_heads,
                layout.v_head_dim,
                layout.kv_lora_rank,
            )?
            .with_context(|| {
                format!(
                    "GLM MLA fused Metal execution requires resident Q4 projection {unembed_out_name}"
                )
            })?;
        let post_projections = post_attention
            .map(|post| {
                let out_proj_name = attention_tensor_name(layer, "o_proj");
                build_required_cmd2_resident_post_attention_prep_projections(
                    layer,
                    post.experts,
                    &out_proj_name,
                    layout.attention_output_width,
                    post.residual.len(),
                    post.active_experts,
                    |tensor_name, output_width, input_len| {
                        self.resident_mmap_projection(tensor_name, output_width, input_len)
                    },
                )
            })
            .transpose()?;

        let mut previous_record_latents = Vec::with_capacity(
            previous_records
                .len()
                .checked_mul(layout.kv_lora_rank)
                .context("MLA fused previous latent size overflow")?,
        );
        let mut previous_record_rotary = Vec::with_capacity(
            previous_records
                .len()
                .checked_mul(layout.qk_rope_head_dim)
                .context("MLA fused previous rotary size overflow")?,
        );
        for (latent, rotary) in previous_records {
            if latent.len() != layout.kv_lora_rank || rotary.len() != layout.qk_rope_head_dim {
                bail!(
                    "MLA layer {layer} previous cache record has latent/rotary widths {}/{}, expected {}/{}",
                    latent.len(),
                    rotary.len(),
                    layout.kv_lora_rank,
                    layout.qk_rope_head_dim,
                );
            }
            previous_record_latents.extend_from_slice(latent);
            previous_record_rotary.extend_from_slice(rotary);
        }
        let scale = (layout.qk_head_dim as f32).sqrt().recip();
        metal.resident_glm_mla_fused_attention(
            &q_a,
            &kv_a,
            &q_b,
            &embed_q,
            &unembed_out,
            MetalGlmMlaFusedAttentionInput {
                input,
                heads: layout.num_heads,
                latent_rank: layout.kv_lora_rank,
                nope_dim: layout.qk_nope_head_dim,
                rope_dim: layout.qk_rope_head_dim,
                previous_record_latents: &previous_record_latents,
                previous_record_rotary: &previous_record_rotary,
                rope_cos,
                rope_sin,
                scale,
                post_attention: post_attention.zip(post_projections.as_ref()).map(
                    |(post, projections)| MetalGlmMlaPostAttentionInput {
                        projections,
                        residual: post.residual,
                        post_norm_weight: post.post_norm_weight,
                        router_correction_bias: post.router_correction_bias,
                    },
                ),
            },
            q_norm_weight,
            kv_norm_weight,
            norm_epsilon,
        )
    }

    fn required_resident_static_tensor(
        &self,
        layer: usize,
        tensor_name: &str,
        expected_values: usize,
        allowed_dtypes: &[ResidentStaticDtype],
    ) -> Result<ResidentStaticTensorRef> {
        let entry = self.registry.require(tensor_name).with_context(|| {
            format!(
                "FlashMoe unsupported scheduled linear-attention static-weight path at layer {layer}: missing tensor {tensor_name}"
            )
        })?;
        ResidentStaticTensorRef::from_entry(
            tensor_name,
            entry,
            self.len,
            expected_values,
            allowed_dtypes,
        )?
        .with_context(|| {
            format!(
                "FlashMoe unsupported scheduled linear-attention static-weight path at layer {layer}: tensor {tensor_name} does not resolve {} values as {}",
                expected_values,
                allowed_dtypes
                    .iter()
                    .map(ResidentStaticDtype::as_str)
                    .collect::<Vec<_>>()
                    .join("/")
            )
        })
    }

    fn required_linear_attention_resident_bindings(
        &self,
        layer: usize,
        layout: LinearAttentionLayout,
        hidden_width: usize,
        experts: usize,
    ) -> Result<LinearAttentionResidentBindings> {
        let input_requests = linear_attention_input_projection_requests(
            layer,
            layout.conv_dim,
            layout.total_value_width,
            layout.num_value_heads,
        )?;
        let mut input_projections = Vec::with_capacity(4);
        for spec in input_requests.requests() {
            input_projections.push(
                self.resident_mmap_projection(spec.tensor_name, spec.output_width, hidden_width)?
                    .with_context(|| {
                        format!(
                            "FlashMoe unsupported scheduled linear-attention CMD1 path at layer {layer}: missing resident projection {}",
                            spec.tensor_name
                        )
                    })?,
            );
        }
        let input_projections = input_projections.try_into().map_err(|values: Vec<_>| {
            anyhow::anyhow!(
                "FlashMoe unsupported scheduled linear-attention CMD1 path at layer {layer}: expected 4 resident projections, resolved {}",
                values.len()
            )
        })?;

        let conv_name = linear_attention_tensor_name(layer, "conv1d");
        let a_log_name = linear_attention_scalar_tensor_name(layer, "A_log");
        let dt_bias_name = linear_attention_scalar_tensor_name(layer, "dt_bias");
        let norm_name = linear_attention_tensor_name(layer, "norm");
        let static_tensors = LinearAttentionStaticBindings {
            conv_weight: self.required_resident_static_tensor(
                layer,
                &conv_name,
                layout.conv_dim * layout.conv_kernel_size,
                &[
                    ResidentStaticDtype::Bf16,
                    ResidentStaticDtype::F16,
                    ResidentStaticDtype::F32,
                ],
            )?,
            a_log: self.required_resident_static_tensor(
                layer,
                &a_log_name,
                layout.num_value_heads,
                &[ResidentStaticDtype::F32],
            )?,
            dt_bias: self.required_resident_static_tensor(
                layer,
                &dt_bias_name,
                layout.num_value_heads,
                &[
                    ResidentStaticDtype::Bf16,
                    ResidentStaticDtype::F16,
                    ResidentStaticDtype::F32,
                ],
            )?,
            norm_weight: self.required_resident_static_tensor(
                layer,
                &norm_name,
                layout.value_dim,
                &[
                    ResidentStaticDtype::Bf16,
                    ResidentStaticDtype::F16,
                    ResidentStaticDtype::F32,
                ],
            )?,
        };
        let out_proj_name = linear_attention_tensor_name(layer, "out_proj");
        let out_proj = self
            .resident_mmap_projection(&out_proj_name, hidden_width, layout.total_value_width)?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled linear-attention CMD2 path at layer {layer}: missing resident output projection {out_proj_name}"
                )
            })?;
        let router_name = router_tensor_name(layer);
        let router = self
            .resident_mmap_projection(&router_name, experts, hidden_width)?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled linear-attention CMD2 path at layer {layer}: missing resident router projection {router_name}"
                )
            })?;
        Ok(LinearAttentionResidentBindings {
            layer,
            input_projections,
            static_tensors,
            out_proj,
            router,
        })
    }

    pub(super) fn resolve_linear_attention_weight_table(
        &self,
        layouts: &[Option<LinearAttentionLayout>],
        hidden_width: usize,
        experts: usize,
    ) -> Result<LinearAttentionWeightTable> {
        let layers = layouts
            .iter()
            .copied()
            .enumerate()
            .map(|(layer, layout)| {
                layout
                    .map(|layout| {
                        self.required_linear_attention_resident_bindings(
                            layer,
                            layout,
                            hidden_width,
                            experts,
                        )
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(LinearAttentionWeightTable { layers })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn linear_attention_post_attention_prep_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        input: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        active_experts: usize,
    ) -> Result<MetalPostAttentionPrep> {
        let layer = bindings.layer;
        let residual_len = residual.len();
        metal.require_resident_dense_weights()?;
        if residual_len != post_norm_weight.len() {
            bail!(
                "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD1/CMD2 path at layer {layer}: residual/norm widths {residual_len}/{} do not match",
                post_norm_weight.len()
            );
        }
        metal.linear_attention_post_attention_prep(
            layout,
            bindings,
            input,
            residual,
            post_norm_weight,
            active_experts,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn project_resident_tensors_from_metal_input(
        &self,
        metal: &MetalExecutionFacade,
        specs: &[DenseProjectionRequest<'_>],
        input_buffer: ObjcId,
        input_len: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if specs.is_empty() {
            bail!("FlashMoe scheduled resident projection batch has no projections");
        }
        metal.require_resident_dense_weights()?;
        let mut projections = Vec::with_capacity(specs.len());
        for spec in specs {
            let projection = self
                .resident_mmap_projection(spec.tensor_name, spec.output_width, input_len)?
                .with_context(|| {
                    format!(
                        "FlashMoe unsupported scheduled resident projection batch: missing projection {}",
                        spec.tensor_name
                    )
                })?;
            projections.push(projection);
        }
        let (outputs, _, _) = metal.resident_mmap_matvec_batch_with_input_buffer(
            &projections,
            input_buffer,
            input_len,
        )?;
        Ok(outputs)
    }

    /// Project using a fully-qualified canonical tensor name (e.g. for shared
    /// experts or any non-attention projection).  Falls back to a zero-vector
    /// when the tensor is absent (tensor not present in this checkpoint means
    /// the feature is disabled for this model variant).
    #[cfg(test)]
    pub(super) fn project_dense_tensor_with_metal(
        &self,
        metal: Option<&MetalExecutionFacade>,
        tensor_name: &str,
        input: &[f32],
        output_width: usize,
    ) -> Result<Option<Vec<f32>>> {
        let entry = match self.registry.tensor(tensor_name) {
            Some(e) => e,
            None => return Ok(None),
        };
        let (rows, cols) =
            validate_dense_matvec_shape(entry, tensor_name, output_width, input.len())?;
        if let Some(metal) = metal {
            return self
                .metal_matvec_tiled(metal, tensor_name, input, rows, cols, output_width)
                .map(Some);
        }
        if let TensorQuantization::Q4 { .. } = entry.quantization {
            return self
                .q4_matvec_tiled(tensor_name, input, rows, cols, output_width)
                .map(Some);
        }
        if let Some(projected) = self.matvec_tensor_prefix(tensor_name, input, output_width)? {
            return Ok(Some(projected));
        }
        Ok(None)
    }

    #[cfg(test)]
    pub(super) fn rms_norm(&self, canonical_name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let mut out = input.to_vec();
        let weight = self.norm_weight(canonical_name, input.len())?;
        rms_norm_with_weight_in_place(&mut out, weight.as_deref());
        Ok(out)
    }

    pub(super) fn norm_weight(
        &self,
        canonical_name: &str,
        width: usize,
    ) -> Result<Option<Vec<f32>>> {
        let key = DenseNormWeightKey {
            name: canonical_name.to_string(),
            width,
        };
        if let Some(weight) = self
            .norm_weights
            .lock()
            .expect("dense norm weight cache poisoned")
            .get(&key)
            .cloned()
        {
            return Ok(Some((*weight).clone()));
        }
        let Some(weight) = self.read_tensor_row_f32(canonical_name, 0, width)? else {
            return Ok(None);
        };
        self.norm_weights
            .lock()
            .expect("dense norm weight cache poisoned")
            .insert(key, Arc::new(weight.clone()));
        Ok(Some(weight))
    }

    pub(super) fn declared_router_projection(
        &self,
        tensor_name: &str,
        expert: usize,
        hidden: &[f32],
    ) -> Result<f32> {
        if let Some(row) = self.read_tensor_row_f32(tensor_name, expert, hidden.len())? {
            let acc = row
                .iter()
                .zip(hidden)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
            return Ok(acc);
        }
        bail!(
            "FlashMoe declared router projection {tensor_name} cannot provide row {expert}; refusing synthetic router fallback"
        )
    }

    pub(super) fn router_command_with_metal(
        &self,
        metal: Option<&MetalExecutionFacade>,
        command: ScheduledRouterScoreProjectionCommand,
        hidden: &[f32],
    ) -> Result<ScheduledRoutingCommand> {
        let execution = command.projection_execution()?;
        let score_plan = execution.score_plan(hidden.len())?;
        let experts = score_plan.experts;
        let _ = metal;
        if score_plan.source == RouterScoreProjectionScoreSource::ResidentDenseFullTensor
            && let Some(scores) = self.router_scores_with_accelerate(score_plan, hidden)?
        {
            return command.into_routing_command(scores);
        }
        let tensor_name = score_plan.tensor_name.to_string();
        let mut router_scores = vec![0.0f32; experts];
        for (expert, score) in router_scores.iter_mut().enumerate() {
            *score = self.declared_router_projection(&tensor_name, expert, hidden)?;
        }
        command.into_routing_command(router_scores)
    }

    pub(super) fn router_score_projection_descriptor(
        &self,
        layer: usize,
        experts: usize,
        hidden_width: usize,
    ) -> Result<Option<RouterScoreProjectionDescriptor>> {
        build_router_score_projection_descriptor(
            layer,
            experts,
            hidden_width,
            self.len,
            |tensor_name| self.registry.tensor(tensor_name),
        )
    }

    pub(super) fn dense_q4_mmap_projection(
        &self,
        tensor_name: &str,
        output_width: usize,
        input_len: usize,
    ) -> Result<Option<DenseQ4MmapMatvecProjection>> {
        let key = DenseQ4ProjectionKey::new(tensor_name, output_width, input_len);
        if let Some(projection) = self
            .q4_mmap_projections
            .lock()
            .expect("dense q4 projection cache poisoned")
            .get(&key)
            .cloned()
        {
            return Ok(Some((*projection).clone()));
        }
        let Some(projection) = build_dense_q4_mmap_projection(
            tensor_name,
            output_width,
            input_len,
            self.len,
            |name| self.registry.tensor(name),
        )?
        else {
            return Ok(None);
        };
        let projection = Arc::new(projection);
        let mut cache = self
            .q4_mmap_projections
            .lock()
            .expect("dense q4 projection cache poisoned");
        if let Some(existing) = cache.get(&key).cloned() {
            Ok(Some((*existing).clone()))
        } else {
            cache.insert(key, projection.clone());
            Ok(Some((*projection).clone()))
        }
    }

    fn dense_q4_mmap_multilinear_projection(
        &self,
        tensor_name: &str,
        heads: usize,
        output_width_per_head: usize,
        input_len: usize,
    ) -> Result<Option<DenseQ4MmapMatvecProjection>> {
        let output_width = heads
            .checked_mul(output_width_per_head)
            .context("dense Q4 multilinear output width overflow")?;
        let key = DenseQ4ProjectionKey::new(tensor_name, output_width, input_len);
        if let Some(projection) = self
            .q4_mmap_projections
            .lock()
            .expect("dense q4 projection cache poisoned")
            .get(&key)
            .cloned()
        {
            return Ok(Some((*projection).clone()));
        }
        let Some(entry) = self.registry.tensor(tensor_name) else {
            return Ok(None);
        };
        let Some(projection) = DenseQ4MmapMatvecProjection::from_multilinear_entry(
            tensor_name,
            entry,
            self.len,
            heads,
            output_width_per_head,
            input_len,
        )?
        else {
            return Ok(None);
        };
        let projection = Arc::new(projection);
        let mut cache = self
            .q4_mmap_projections
            .lock()
            .expect("dense q4 projection cache poisoned");
        if let Some(existing) = cache.get(&key).cloned() {
            Ok(Some((*existing).clone()))
        } else {
            cache.insert(key, projection.clone());
            Ok(Some((*projection).clone()))
        }
    }

    pub(super) fn resident_mmap_projection(
        &self,
        tensor_name: &str,
        output_width: usize,
        input_len: usize,
    ) -> Result<Option<ResidentMmapMatvecProjection>> {
        let Some(entry) = self.registry.tensor(tensor_name) else {
            return Ok(None);
        };
        if matches!(&entry.quantization, TensorQuantization::Q4 { .. }) {
            return self
                .dense_q4_mmap_projection(tensor_name, output_width, input_len)
                .map(|projection| projection.map(ResidentMmapMatvecProjection::Q4));
        }
        ResidentMmapMatvecProjection::from_entry(
            tensor_name,
            entry,
            self.len,
            output_width,
            input_len,
        )
        .map(Some)
    }

    fn q4_affine_scalar(&self, byte_offset: u64, dtype: &str, index: usize) -> Result<f32> {
        let element_size = expert_scale_bias_dtype_size(dtype)
            .with_context(|| format!("unsupported Q4 scale/bias dtype {dtype}"))?;
        let start = usize::try_from(byte_offset)
            .context("Q4 scale/bias offset exceeds usize")?
            .checked_add(
                index
                    .checked_mul(element_size)
                    .context("Q4 scalar offset overflow")?,
            )
            .context("Q4 scalar offset overflow")?;
        let end = start
            .checked_add(element_size)
            .context("Q4 scalar range overflow")?;
        let bytes = self
            .mmap
            .get(start..end)
            .with_context(|| format!("Q4 scalar range {start}..{end} exceeds dense mmap"))?;
        if dtype.eq_ignore_ascii_case("F32")
            || dtype.eq_ignore_ascii_case("FLOAT32")
            || dtype.eq_ignore_ascii_case("FP32")
        {
            Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
        } else if dtype.eq_ignore_ascii_case("BF16") || dtype.eq_ignore_ascii_case("BFLOAT16") {
            let bits = u16::from_le_bytes(bytes.try_into().unwrap()) as u32;
            Ok(f32::from_bits(bits << 16))
        } else if dtype.eq_ignore_ascii_case("F16")
            || dtype.eq_ignore_ascii_case("FLOAT16")
            || dtype.eq_ignore_ascii_case("FP16")
        {
            Ok(f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
        } else {
            bail!("unsupported Q4 scale/bias dtype {dtype}")
        }
    }

    fn q4_row_add_scaled(
        &self,
        projection: &DenseQ4MmapMatvecProjection,
        row: usize,
        coefficient: f32,
        output: &mut [f32],
    ) -> Result<()> {
        if row >= projection.rows || output.len() != projection.cols {
            bail!(
                "Q4 row accumulation for {} has row {row}/{} and output width {}/{}",
                projection.tensor_name,
                projection.rows,
                output.len(),
                projection.cols
            );
        }
        let packed_start = usize::try_from(projection.packed_byte_offset)
            .context("Q4 packed offset exceeds usize")?
            .checked_add(
                row.checked_mul(projection.row_packed_bytes)
                    .context("Q4 packed row offset overflow")?,
            )
            .context("Q4 packed row offset overflow")?;
        let packed = self
            .mmap
            .get(packed_start..packed_start + projection.row_packed_bytes)
            .with_context(|| format!("Q4 packed row {row} exceeds dense mmap"))?;
        for group in 0..projection.groups_per_row {
            let scalar_index = row * projection.groups_per_row + group;
            let scale = self.q4_affine_scalar(
                projection.scales_byte_offset,
                &projection.scale_bias_dtype,
                scalar_index,
            )?;
            let bias = self.q4_affine_scalar(
                projection.biases_byte_offset,
                &projection.scale_bias_dtype,
                scalar_index,
            )?;
            let start = group * projection.group_size;
            let end = (start + projection.group_size).min(projection.cols);
            for col in start..end {
                let byte = packed[col / 2];
                let q = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                output[col] += coefficient * q.mul_add(scale, bias);
            }
        }
        Ok(())
    }

    fn q4_row_dot(
        &self,
        projection: &DenseQ4MmapMatvecProjection,
        row: usize,
        input: &[f32],
    ) -> Result<f32> {
        if row >= projection.rows || input.len() != projection.cols {
            bail!(
                "Q4 row dot for {} has row {row}/{} and input width {}/{}",
                projection.tensor_name,
                projection.rows,
                input.len(),
                projection.cols
            );
        }
        let packed_start = usize::try_from(projection.packed_byte_offset)
            .context("Q4 packed offset exceeds usize")?
            .checked_add(
                row.checked_mul(projection.row_packed_bytes)
                    .context("Q4 packed row offset overflow")?,
            )
            .context("Q4 packed row offset overflow")?;
        let packed = self
            .mmap
            .get(packed_start..packed_start + projection.row_packed_bytes)
            .with_context(|| format!("Q4 packed row {row} exceeds dense mmap"))?;
        let mut sum = 0.0;
        for group in 0..projection.groups_per_row {
            let scalar_index = row * projection.groups_per_row + group;
            let scale = self.q4_affine_scalar(
                projection.scales_byte_offset,
                &projection.scale_bias_dtype,
                scalar_index,
            )?;
            let bias = self.q4_affine_scalar(
                projection.biases_byte_offset,
                &projection.scale_bias_dtype,
                scalar_index,
            )?;
            let start = group * projection.group_size;
            let end = (start + projection.group_size).min(projection.cols);
            for col in start..end {
                let byte = packed[col / 2];
                let q = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                sum += input[col] * q.mul_add(scale, bias);
            }
        }
        Ok(sum)
    }

    pub(super) fn mla_absorbed_attention(
        &self,
        layer: usize,
        layout: MlaAttentionLayout,
        query: &[f32],
        records: &[(&[f32], &[f32])],
    ) -> Result<Vec<f32>> {
        if query.len() != layout.q_width || records.is_empty() {
            bail!(
                "MLA layer {layer} requires query width {} and a non-empty KV cache, got {} and {} records",
                layout.q_width,
                query.len(),
                records.len()
            );
        }
        for (latent, rotary) in records {
            if latent.len() != layout.kv_lora_rank || rotary.len() != layout.qk_rope_head_dim {
                bail!(
                    "MLA layer {layer} cache record has latent/rotary widths {}/{}, expected {}/{}",
                    latent.len(),
                    rotary.len(),
                    layout.kv_lora_rank,
                    layout.qk_rope_head_dim
                );
            }
        }
        enum AbsorbedWeights {
            Fused(DenseQ4MmapMatvecProjection),
            MultiLinear {
                embed_q: DenseQ4MmapMatvecProjection,
                unembed_out: DenseQ4MmapMatvecProjection,
            },
        }
        let weights = match layout.kv_projection {
            MlaKvProjectionLayout::FusedKvB => {
                let tensor_name = attention_tensor_name(layer, "kv_b_proj");
                let projection = self
                    .dense_q4_mmap_projection(
                        &tensor_name,
                        layout.kv_b_width,
                        layout.kv_lora_rank,
                    )?
                    .with_context(|| {
                        format!(
                            "GLM MLA weight absorption requires resident Q4 projection {tensor_name}"
                        )
                    })?;
                AbsorbedWeights::Fused(projection)
            }
            MlaKvProjectionLayout::AbsorbedMultiLinear => {
                let embed_q_name = attention_tensor_name(layer, "embed_q");
                let embed_q = self
                    .dense_q4_mmap_multilinear_projection(
                        &embed_q_name,
                        layout.num_heads,
                        layout.kv_lora_rank,
                        layout.qk_nope_head_dim,
                    )?
                    .with_context(|| {
                        format!(
                            "GLM MLA weight absorption requires resident Q4 projection {embed_q_name}"
                        )
                    })?;
                let unembed_out_name = attention_tensor_name(layer, "unembed_out");
                let unembed_out = self
                    .dense_q4_mmap_multilinear_projection(
                        &unembed_out_name,
                        layout.num_heads,
                        layout.v_head_dim,
                        layout.kv_lora_rank,
                    )?
                    .with_context(|| {
                        format!(
                            "GLM MLA weight absorption requires resident Q4 projection {unembed_out_name}"
                        )
                    })?;
                AbsorbedWeights::MultiLinear {
                    embed_q,
                    unembed_out,
                }
            }
        };
        let mut output = vec![0.0; layout.attention_output_width];
        let scale = (layout.qk_head_dim as f32).sqrt().recip();
        let kv_b_head_width = layout.qk_nope_head_dim + layout.v_head_dim;
        for head in 0..layout.num_heads {
            let query_head = &query[head * layout.qk_head_dim..(head + 1) * layout.qk_head_dim];
            let (query_nope, query_rope) = query_head.split_at(layout.qk_nope_head_dim);
            let absorbed_query = match &weights {
                AbsorbedWeights::Fused(projection) => {
                    let row_base = head * kv_b_head_width;
                    let mut absorbed_query = vec![0.0; layout.kv_lora_rank];
                    for (dimension, coefficient) in query_nope.iter().copied().enumerate() {
                        self.q4_row_add_scaled(
                            projection,
                            row_base + dimension,
                            coefficient,
                            &mut absorbed_query,
                        )?;
                    }
                    absorbed_query
                }
                AbsorbedWeights::MultiLinear { embed_q, .. } => (0..layout.kv_lora_rank)
                    .map(|dimension| {
                        self.q4_row_dot(embed_q, head * layout.kv_lora_rank + dimension, query_nope)
                    })
                    .collect::<Result<Vec<_>>>()?,
            };
            let mut scores = records
                .iter()
                .map(|(latent, rotary)| {
                    let latent_score = absorbed_query
                        .iter()
                        .zip(*latent)
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    let rotary_score = query_rope
                        .iter()
                        .zip(*rotary)
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    (latent_score + rotary_score) * scale
                })
                .collect::<Vec<_>>();
            softmax_in_place(&mut scores);
            let mut context = vec![0.0; layout.kv_lora_rank];
            for (weight, (latent, _)) in scores.iter().zip(records) {
                for (slot, value) in context.iter_mut().zip(*latent) {
                    *slot += *weight * value;
                }
            }
            let head_output = &mut output[head * layout.v_head_dim..(head + 1) * layout.v_head_dim];
            for (dimension, slot) in head_output.iter_mut().enumerate() {
                *slot = match &weights {
                    AbsorbedWeights::Fused(projection) => self.q4_row_dot(
                        projection,
                        head * kv_b_head_width + layout.qk_nope_head_dim + dimension,
                        &context,
                    )?,
                    AbsorbedWeights::MultiLinear { unembed_out, .. } => self.q4_row_dot(
                        unembed_out,
                        head * layout.v_head_dim + dimension,
                        &context,
                    )?,
                };
            }
        }
        Ok(output)
    }

    pub(super) fn mla_absorbed_attention_metal(
        &self,
        metal: &MetalExecutionFacade,
        layer: usize,
        layout: MlaAttentionLayout,
        query: &[f32],
        records: &[(&[f32], &[f32])],
    ) -> Result<Vec<f32>> {
        if query.len() != layout.q_width || records.is_empty() {
            bail!(
                "MLA layer {layer} requires query width {} and a non-empty KV cache, got {} and {} records",
                layout.q_width,
                query.len(),
                records.len()
            );
        }
        for (latent, rotary) in records {
            if latent.len() != layout.kv_lora_rank || rotary.len() != layout.qk_rope_head_dim {
                bail!(
                    "MLA layer {layer} cache record has latent/rotary widths {}/{}, expected {}/{}",
                    latent.len(),
                    rotary.len(),
                    layout.kv_lora_rank,
                    layout.qk_rope_head_dim
                );
            }
        }
        if layout.kv_projection != MlaKvProjectionLayout::AbsorbedMultiLinear {
            bail!(
                "MLA layer {layer} Metal multilinear execution requires pre-absorbed embed_q/unembed_out weights"
            );
        }
        let embed_q_name = attention_tensor_name(layer, "embed_q");
        let embed_q = self
            .dense_q4_mmap_multilinear_projection(
                &embed_q_name,
                layout.num_heads,
                layout.kv_lora_rank,
                layout.qk_nope_head_dim,
            )?
            .with_context(|| {
                format!(
                    "GLM MLA Metal weight absorption requires resident Q4 projection {embed_q_name}"
                )
            })?;
        let unembed_out_name = attention_tensor_name(layer, "unembed_out");
        let unembed_out = self
            .dense_q4_mmap_multilinear_projection(
                &unembed_out_name,
                layout.num_heads,
                layout.v_head_dim,
                layout.kv_lora_rank,
            )?
            .with_context(|| {
                format!(
                    "GLM MLA Metal weight absorption requires resident Q4 projection {unembed_out_name}"
                )
            })?;

        let mut query_nope = Vec::with_capacity(
            layout
                .num_heads
                .checked_mul(layout.qk_nope_head_dim)
                .context("MLA no-PE query size overflow")?,
        );
        let mut query_rope = Vec::with_capacity(
            layout
                .num_heads
                .checked_mul(layout.qk_rope_head_dim)
                .context("MLA rotary query size overflow")?,
        );
        for head in 0..layout.num_heads {
            let start = head * layout.qk_head_dim;
            query_nope.extend_from_slice(&query[start..start + layout.qk_nope_head_dim]);
            query_rope.extend_from_slice(
                &query[start + layout.qk_nope_head_dim..start + layout.qk_head_dim],
            );
        }
        let mut record_latents = Vec::with_capacity(
            records
                .len()
                .checked_mul(layout.kv_lora_rank)
                .context("MLA latent record size overflow")?,
        );
        let mut record_rotary = Vec::with_capacity(
            records
                .len()
                .checked_mul(layout.qk_rope_head_dim)
                .context("MLA rotary record size overflow")?,
        );
        for (latent, rotary) in records {
            record_latents.extend_from_slice(latent);
            record_rotary.extend_from_slice(rotary);
        }
        let scale = (layout.qk_head_dim as f32).sqrt().recip();
        metal.resident_glm_mla_absorbed_attention(
            &embed_q,
            &unembed_out,
            MetalGlmMlaAbsorbedAttentionInput {
                heads: layout.num_heads,
                latent_rank: layout.kv_lora_rank,
                query_nope: &query_nope,
                query_rope: &query_rope,
                record_latents: &record_latents,
                record_rotary: &record_rotary,
                sequence: records.len(),
                rope_dim: layout.qk_rope_head_dim,
                scale,
            },
        )
    }

    #[cfg(test)]
    pub(super) fn resolve_shared_expert_weight_table(
        &self,
        layer_count: usize,
        width: usize,
        shared_experts: usize,
        intermediate: usize,
    ) -> Result<SharedExpertWeightTable> {
        self.resolve_shared_expert_weight_table_from(
            layer_count,
            width,
            shared_experts,
            intermediate,
            0,
            true,
        )
    }

    pub(super) fn resolve_shared_expert_weight_table_from(
        &self,
        layer_count: usize,
        width: usize,
        shared_experts: usize,
        intermediate: usize,
        first_sparse_layer: usize,
        requires_router: bool,
    ) -> Result<SharedExpertWeightTable> {
        let layers = (0..layer_count)
            .map(|layer| {
                if layer < first_sparse_layer {
                    return Ok(SharedExpertLayerWeights::None);
                }
                build_required_shared_expert_resident_phase_projections_with_router(
                    layer,
                    width,
                    shared_experts,
                    intermediate,
                    requires_router,
                    |tensor_name, output_width, input_len| {
                        self.resident_mmap_projection(tensor_name, output_width, input_len)
                    },
                )
                .map(|weights| match weights {
                    Some(weights) => SharedExpertLayerWeights::Resident(weights),
                    None => SharedExpertLayerWeights::None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SharedExpertWeightTable { layers })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn post_attention_prep_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        layer: usize,
        experts: usize,
        out_proj_name: &str,
        attention_output: &[f32],
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        active_experts: usize,
        router_correction_bias: Option<&[f32]>,
    ) -> Result<MetalPostAttentionPrep> {
        metal.require_resident_dense_weights()?;
        let residual_len = residual.len();
        let projections = build_required_cmd2_resident_post_attention_prep_projections(
            layer,
            experts,
            out_proj_name,
            attention_output.len(),
            residual_len,
            active_experts,
            |tensor_name, output_width, input_len| {
                self.resident_mmap_projection(tensor_name, output_width, input_len)
            },
        )?;
        metal.resident_post_attention_prep_topk(
            &projections,
            attention_output,
            residual,
            post_norm_weight,
            router_correction_bias,
        )
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(super) fn router_topk_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        layer: usize,
        experts: usize,
        hidden: &[f32],
        active_experts: usize,
    ) -> Result<Option<Vec<(usize, f32)>>> {
        if active_experts == 0 || !metal.has_resident_dense_weights() {
            return Ok(None);
        }
        let Some(descriptor) = build_router_score_projection_descriptor(
            layer,
            experts,
            hidden.len(),
            self.len,
            |tensor_name| self.registry.tensor(tensor_name),
        )?
        else {
            return Ok(None);
        };
        let plan = descriptor.topk_plan(hidden.len(), active_experts)?;
        metal.router_score_top_candidates(&plan, hidden)
    }

    pub(super) fn router_scores_with_accelerate(
        &self,
        score_plan: RouterScoreProjectionScorePlan<'_>,
        hidden: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        if score_plan.source != RouterScoreProjectionScoreSource::ResidentDenseFullTensor {
            return Ok(None);
        }
        if score_plan.hidden_width != hidden.len() {
            return Ok(None);
        }
        let weights =
            self.read_tensor_rows_f32_cached(score_plan.tensor_name, 0, score_plan.experts)?;
        dense_f32_matvec_rows(
            weights.as_slice(),
            hidden,
            score_plan.experts,
            score_plan.hidden_width,
        )
    }

    pub(super) fn lm_head_logits_with_metal(
        &self,
        metal: Option<&MetalExecutionFacade>,
        _state: u64,
        hidden: &[f32],
        tokenizer: &QwenTokenizer,
    ) -> Result<Vec<f32>> {
        let lm_head_name = self.lm_head_tensor_name()?;
        if let Some(metal) = metal
            && let Some(entry) = self.registry.tensor(lm_head_name)
        {
            let (rows, cols) = validate_lm_head_matvec_shape(
                entry,
                lm_head_name,
                tokenizer.vocab_size(),
                hidden.len(),
            )?;
            let mut logits = vec![f32::NEG_INFINITY; tokenizer.vocab_size()];
            let projected =
                self.metal_matvec_tiled(metal, lm_head_name, hidden, rows, cols, rows)?;
            for (token, value) in projected
                .into_iter()
                .take(tokenizer.vocab_size())
                .enumerate()
            {
                logits[token] = value;
            }
            return Ok(logits);
        }

        self.lm_head_logits(lm_head_name, hidden, tokenizer)
    }

    pub(super) fn lm_head_top_candidates_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        hidden: &[f32],
        tokenizer: &QwenTokenizer,
        sampler: &TokenSampler,
        prompt: &[u32],
        generated: &[u32],
    ) -> Result<Vec<(usize, f32)>> {
        metal.require_resident_dense_weights()?;
        let lm_head_name = self.lm_head_tensor_name()?;
        let entry = self.registry.require(lm_head_name)?;
        let (rows, cols) = validate_lm_head_matvec_shape(
            entry,
            lm_head_name,
            tokenizer.vocab_size(),
            hidden.len(),
        )?;

        let vocab_rows = tokenizer.vocab_size();
        let top_k = sampler.top_k.min(vocab_rows).max(1);
        let repeated = sampler.repeated_tokens(prompt, generated);
        let repeated_vocab_tokens = repeated.iter().filter(|token| **token < vocab_rows).count();
        let raw_candidate_count = top_k
            .saturating_add(repeated_vocab_tokens)
            .min(vocab_rows)
            .max(1);
        let projection = self
            .resident_mmap_projection(lm_head_name, rows, cols)?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported resolved LM-head path: missing resident projection {lm_head_name}"
                )
            })?;
        let raw_candidates =
            metal.resident_top_candidates(&projection, hidden, vocab_rows, raw_candidate_count)?;
        Ok(rerank_resident_lm_head_candidates(
            &raw_candidates,
            top_k,
            sampler.repeat_penalty,
            &repeated,
        ))
    }

    pub(super) fn lm_head_logits(
        &self,
        lm_head_name: &str,
        hidden: &[f32],
        tokenizer: &QwenTokenizer,
    ) -> Result<Vec<f32>> {
        let entry = self.registry.require(lm_head_name)?;
        validate_lm_head_matvec_shape(entry, lm_head_name, tokenizer.vocab_size(), hidden.len())?;
        let mut logits = vec![f32::NEG_INFINITY; tokenizer.vocab_size()];
        for idx in 0..tokenizer.vocab_size() {
            let Some(row) = self.read_tensor_row_f32(lm_head_name, idx, hidden.len())? else {
                bail!(
                    "Flash-MoE LM head tensor {lm_head_name} cannot provide row for token {idx}; refusing synthetic logits"
                );
            };
            logits[idx] = row
                .iter()
                .zip(hidden)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
        }
        Ok(logits)
    }

    pub(super) fn lm_head_tensor_name(&self) -> Result<&'static str> {
        if self.registry.tensor("lm_head.weight").is_some() {
            Ok("lm_head.weight")
        } else if self.registry.tensor("model.embed_tokens.weight").is_some() {
            Ok("model.embed_tokens.weight")
        } else {
            bail!(
                "Flash-MoE dense tensor registry is missing lm_head.weight and tied model.embed_tokens.weight"
            )
        }
    }

    pub(super) fn matvec_tensor_prefix(
        &self,
        canonical_name: &str,
        input: &[f32],
        width: usize,
    ) -> Result<Option<Vec<f32>>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            return Ok(None);
        };
        if entry.quantization != TensorQuantization::None {
            bail!("dense q4 tensor {canonical_name} cannot be read as a full f32 tensor");
        }
        let (rows, cols) = validate_dense_matvec_shape(entry, canonical_name, width, input.len())?;
        if let Some(tensor) = self.dense_tensor_f32(canonical_name)? {
            let expected_len = rows
                .checked_mul(cols)
                .context("dense resident tensor value count overflow")?;
            if tensor.len() != expected_len {
                bail!(
                    "Flash-MoE dense tensor {canonical_name} has {} decoded values; expected {expected_len} for shape {:?} and input length {}",
                    tensor.len(),
                    entry.shape,
                    input.len()
                );
            }
            let mut out = vec![0.0f32; width];
            for (row, slot) in out.iter_mut().take(rows).enumerate() {
                let start = row
                    .checked_mul(cols)
                    .context("dense resident tensor row offset overflow")?;
                let end = start
                    .checked_add(cols)
                    .context("dense resident tensor row length overflow")?;
                let weights = &tensor[start..end];
                let acc = weights
                    .iter()
                    .zip(input.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>();
                *slot = acc;
            }
            return Ok(Some(out));
        }
        let mut out = vec![0.0f32; width];
        for (row, slot) in out.iter_mut().take(rows).enumerate() {
            let weights = self.read_tensor_row_f32(canonical_name, row, cols)?;
            let Some(weights) = weights else {
                return Ok(None);
            };
            let acc = weights
                .iter()
                .zip(input.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
            *slot = acc;
        }
        Ok(Some(out))
    }

    pub(super) fn metal_matvec_tiled(
        &self,
        metal: &MetalExecutionFacade,
        canonical_name: &str,
        input: &[f32],
        rows: usize,
        cols: usize,
        output_width: usize,
    ) -> Result<Vec<f32>> {
        let entry = self.registry.tensor(canonical_name).with_context(|| {
            format!("Flash-MoE dense tensor registry is missing {canonical_name}")
        })?;
        validate_dense_matvec_shape(entry, canonical_name, output_width, input.len())?;
        if rows != output_width || cols != input.len() {
            bail!(
                "FlashMoe scheduled Q4 projection {canonical_name} dimensions do not match output/input widths"
            );
        }
        let projection = self
            .dense_q4_mmap_projection(canonical_name, output_width, input.len())?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled Q4 projection: missing resident descriptor for {canonical_name}"
                )
            })?;
        let projection = ResidentMmapMatvecProjection::Q4(projection);
        let (mut outputs, _, _) =
            metal.resident_mmap_matvec_batch(std::slice::from_ref(&projection), input)?;
        outputs
            .pop()
            .with_context(|| format!("Metal Q4 projection {canonical_name} returned no output"))
    }

    #[cfg(test)]
    pub(super) fn q4_matvec_tiled(
        &self,
        canonical_name: &str,
        input: &[f32],
        rows: usize,
        cols: usize,
        output_width: usize,
    ) -> Result<Vec<f32>> {
        let entry = self.registry.tensor(canonical_name).with_context(|| {
            format!("Flash-MoE dense tensor registry is missing {canonical_name}")
        })?;
        let TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } = &entry.quantization
        else {
            bail!("Flash-MoE dense tensor {canonical_name} is not q4-quantized");
        };
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&entry.shape, *group_size, scale_bias_dtype)?;
        if rows != layout.rows || cols != layout.cols {
            bail!(
                "Flash-MoE dense q4 tensor {canonical_name} matvec dimensions mismatch: layout rows={}, cols={}, requested rows={rows}, cols={cols}",
                layout.rows,
                layout.cols
            );
        }
        let mut output = vec![0.0f32; output_width];
        let tile_rows = dense_projection_tile_rows(cols, rows);
        for start in (0..rows).step_by(tile_rows) {
            let end = (start + tile_rows).min(rows);
            let rows = end - start;
            let (packed, scales, biases, _) =
                self.read_dense_q4_rows(entry, start, rows, *group_size)?;
            let projected = q4_fma_matvec_with_group_size(
                &packed,
                input,
                &scales,
                &biases,
                rows,
                cols,
                *group_size,
            )?;
            output[start..end].copy_from_slice(&projected);
        }
        Ok(output)
    }

    pub(super) fn dense_tensor_f32(&self, canonical_name: &str) -> Result<Option<Arc<Vec<f32>>>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            return Ok(None);
        };
        if let Some(tensor) = self
            .resident
            .lock()
            .expect("dense tensor cache poisoned")
            .get(canonical_name)
        {
            return Ok(Some(tensor));
        }
        if let TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } = &entry.quantization
        {
            let layout =
                dense_q4_layout_with_scale_bias_dtype(&entry.shape, *group_size, scale_bias_dtype)?;
            let decoded_bytes = layout
                .rows
                .checked_mul(layout.cols)
                .and_then(|items| items.checked_mul(std::mem::size_of::<f32>()))
                .context("dense q4 full tensor decoded byte length overflow")?;
            if decoded_bytes > DENSE_Q4_FULL_DECODE_MAX_BYTES {
                bail!(
                    "dense q4 tensor {canonical_name} would decode to {decoded_bytes} bytes, over full decode limit {DENSE_Q4_FULL_DECODE_MAX_BYTES}"
                );
            }
            let (packed, scales, biases, _) =
                self.read_dense_q4_rows(entry, 0, layout.rows, *group_size)?;
            let tensor = Arc::new(q4_dequantize_rows_with_group_size(
                &packed,
                &scales,
                &biases,
                layout.rows,
                layout.cols,
                *group_size,
            )?);
            #[cfg(test)]
            self.decoded_full_tensors
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.resident
                .lock()
                .expect("dense tensor cache poisoned")
                .insert(canonical_name.to_string(), tensor.clone());
            return Ok(Some(tensor));
        }
        let bytes = self.read_range(entry.byte_offset, entry.byte_len as usize)?;
        let tensor = Arc::new(decode_dense_tensor_f32(&entry.dtype, &bytes)?);
        #[cfg(test)]
        self.decoded_full_tensors
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.resident
            .lock()
            .expect("dense tensor cache poisoned")
            .insert(canonical_name.to_string(), tensor.clone());
        Ok(Some(tensor))
    }

    pub(super) fn read_tensor_rows_f32_cached(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<Arc<Vec<f32>>> {
        let (tile, _) =
            self.read_tensor_rows_f32_cached_profiled(canonical_name, start_row, row_count)?;
        Ok(tile)
    }

    pub(super) fn read_tensor_rows_f32_cached_profiled(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<(Arc<Vec<f32>>, DenseTileReadTiming)> {
        let started = Instant::now();
        let key = DenseTensorTileKey {
            name: canonical_name.to_string(),
            start_row,
            row_count,
        };
        if let Some(tile) = self
            .decoded_tiles
            .lock()
            .expect("dense decoded tile cache poisoned")
            .get(&key)
        {
            let mut timing = DenseTileReadTiming {
                cache_hits: 1,
                ..DenseTileReadTiming::default()
            };
            timing.total = started.elapsed();
            return Ok((tile, timing));
        }
        let mut timing = DenseTileReadTiming {
            cache_misses: 1,
            ..DenseTileReadTiming::default()
        };
        let (decoded, uncached_timing) =
            self.read_tensor_rows_f32_profiled(canonical_name, start_row, row_count)?;
        timing.add(uncached_timing);
        let tile = Arc::new(decoded);
        let stats = self
            .decoded_tiles
            .lock()
            .expect("dense decoded tile cache poisoned")
            .insert(key, tile.clone());
        timing.cache_inserts = timing.cache_inserts.saturating_add(stats.inserts);
        timing.cache_evictions = timing.cache_evictions.saturating_add(stats.evictions);
        timing.cache_insert += stats.insert_time;
        timing.cache_evict += stats.evict_time;
        timing.total = started.elapsed();
        Ok((tile, timing))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(super) fn read_tensor_rows_raw_cached_profiled(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<(Arc<Vec<u8>>, String, DenseTileReadTiming)> {
        let started = Instant::now();
        let key = DenseTensorTileKey {
            name: canonical_name.to_string(),
            start_row,
            row_count,
        };
        if let Some(tile) = self
            .raw_tiles
            .lock()
            .expect("dense raw tile cache poisoned")
            .get(&key)
        {
            let mut timing = DenseTileReadTiming {
                cache_hits: 1,
                ..DenseTileReadTiming::default()
            };
            timing.total = started.elapsed();
            let dtype = self
                .registry
                .tensor(canonical_name)
                .map(|entry| entry.dtype.clone())
                .with_context(|| {
                    format!("Flash-MoE dense tensor registry is missing {canonical_name}")
                })?;
            return Ok((tile, dtype, timing));
        }

        let mut timing = DenseTileReadTiming {
            cache_misses: 1,
            ..DenseTileReadTiming::default()
        };
        let (bytes, dtype, uncached_timing) =
            self.read_tensor_rows_raw_profiled(canonical_name, start_row, row_count)?;
        timing.add(uncached_timing);
        let tile = Arc::new(bytes);
        let stats = self
            .raw_tiles
            .lock()
            .expect("dense raw tile cache poisoned")
            .insert(key, tile.clone());
        timing.cache_inserts = timing.cache_inserts.saturating_add(stats.inserts);
        timing.cache_evictions = timing.cache_evictions.saturating_add(stats.evictions);
        timing.cache_insert += stats.insert_time;
        timing.cache_evict += stats.evict_time;
        timing.total = started.elapsed();
        Ok((tile, dtype, timing))
    }

    pub(super) fn read_dense_q4_rows(
        &self,
        entry: &RuntimeTensorEntry,
        start_row: usize,
        row_count: usize,
        group_size: usize,
    ) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>, DenseTileReadTiming)> {
        let started = Instant::now();
        let mut timing = DenseTileReadTiming::default();
        let TensorQuantization::Q4 {
            scale_bias_dtype, ..
        } = &entry.quantization
        else {
            bail!("dense tensor {} is not q4-quantized", entry.name);
        };
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&entry.shape, group_size, scale_bias_dtype)?;
        if entry.byte_len as usize != layout.total_bytes {
            bail!(
                "dense q4 tensor {} byte length {} does not match computed layout {}",
                entry.name,
                entry.byte_len,
                layout.total_bytes
            );
        }
        let end_row = start_row
            .checked_add(row_count)
            .context("dense q4 tile row range overflow")?;
        if end_row > layout.rows {
            bail!(
                "dense q4 tensor {} rows {}..{} exceed row count {}",
                entry.name,
                start_row,
                end_row,
                layout.rows
            );
        }
        if row_count == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new(), timing));
        }
        let packed_offset = start_row
            .checked_mul(layout.row_packed_bytes)
            .context("dense q4 packed tile offset overflow")?;
        let packed_len = row_count
            .checked_mul(layout.row_packed_bytes)
            .context("dense q4 packed tile length overflow")?;
        let groups_offset = start_row
            .checked_mul(layout.groups_per_row)
            .and_then(|groups| groups.checked_mul(layout.scale_bias_bytes))
            .context("dense q4 groups tile offset overflow")?;
        let groups_len = row_count
            .checked_mul(layout.groups_per_row)
            .and_then(|groups| groups.checked_mul(layout.scale_bias_bytes))
            .context("dense q4 groups tile byte length overflow")?;

        let (packed, read_packed) =
            self.read_range_profiled(entry.byte_offset + packed_offset as u64, packed_len)?;
        let (scale_bytes, read_scales) = self.read_range_profiled(
            entry.byte_offset + layout.packed_bytes as u64 + groups_offset as u64,
            groups_len,
        )?;
        let (bias_bytes, read_biases) = self.read_range_profiled(
            entry.byte_offset
                + layout.packed_bytes as u64
                + layout.scales_bytes as u64
                + groups_offset as u64,
            groups_len,
        )?;
        timing.read_range += read_packed + read_scales + read_biases;
        timing.bytes_read = timing
            .bytes_read
            .saturating_add((packed_len + groups_len + groups_len) as u64);
        let decode_started = Instant::now();
        let scales = decode_dense_tensor_f32(scale_bias_dtype, &scale_bytes)?;
        let biases = decode_dense_tensor_f32(scale_bias_dtype, &bias_bytes)?;
        timing.decode += decode_started.elapsed();
        timing.decoded_bytes = timing
            .decoded_bytes
            .saturating_add(((scales.len() + biases.len()) * std::mem::size_of::<f32>()) as u64);
        timing.total = started.elapsed();
        Ok((packed, scales, biases, timing))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(super) fn read_tensor_rows_raw_profiled(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<(Vec<u8>, String, DenseTileReadTiming)> {
        let started = Instant::now();
        let mut timing = DenseTileReadTiming::default();
        let Some(entry) = self.registry.tensor(canonical_name) else {
            bail!("Flash-MoE dense tensor registry is missing {canonical_name}");
        };
        if entry.quantization != TensorQuantization::None {
            bail!("dense q4 tensor {canonical_name} cannot be read as raw dense rows");
        }
        let Some(element_size) = dtype_size(&entry.dtype) else {
            bail!(
                "Flash-MoE dense tensor {} has unsupported dtype {}",
                entry.name,
                entry.dtype
            );
        };
        let cols = entry.shape.last().copied().unwrap_or(0);
        if entry.shape.is_empty() || cols == 0 || row_count == 0 {
            return Ok((Vec::new(), entry.dtype.clone(), timing));
        }
        let rows = entry
            .shape
            .iter()
            .take(entry.shape.len() - 1)
            .product::<usize>()
            .max(1);
        let end_row = start_row
            .checked_add(row_count)
            .context("dense tensor raw tile row range overflow")?;
        if end_row > rows {
            bail!(
                "Flash-MoE dense tensor {} raw tile rows {}..{} exceed row count {}",
                entry.name,
                start_row,
                end_row,
                rows
            );
        }
        let row_bytes = cols
            .checked_mul(element_size)
            .context("dense tensor raw tile row byte length overflow")?;
        let byte_offset = start_row
            .checked_mul(row_bytes)
            .context("dense tensor raw tile byte offset overflow")?;
        let byte_len = row_count
            .checked_mul(row_bytes)
            .context("dense tensor raw tile byte length overflow")?;
        let (bytes, read_range) =
            self.read_range_profiled(entry.byte_offset + byte_offset as u64, byte_len)?;
        timing.read_range += read_range;
        timing.bytes_read = timing.bytes_read.saturating_add(byte_len as u64);
        timing.total = started.elapsed();
        Ok((bytes, entry.dtype.clone(), timing))
    }

    #[cfg(test)]
    pub(super) fn read_tensor_rows_f32(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        let (tensor, _) =
            self.read_tensor_rows_f32_profiled(canonical_name, start_row, row_count)?;
        Ok(tensor)
    }

    pub(super) fn read_tensor_rows_f32_profiled(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<(Vec<f32>, DenseTileReadTiming)> {
        let started = Instant::now();
        let mut timing = DenseTileReadTiming::default();
        let Some(entry) = self.registry.tensor(canonical_name) else {
            bail!("Flash-MoE dense tensor registry is missing {canonical_name}");
        };
        if entry.quantization != TensorQuantization::None {
            bail!("dense q4 tensor {canonical_name} cannot be decoded as f32 rows");
        }
        let Some(element_size) = dtype_size(&entry.dtype) else {
            bail!(
                "Flash-MoE dense tensor {} has unsupported dtype {}",
                entry.name,
                entry.dtype
            );
        };
        let cols = entry.shape.last().copied().unwrap_or(0);
        if entry.shape.is_empty() || cols == 0 || row_count == 0 {
            return Ok((Vec::new(), timing));
        }
        let rows = entry
            .shape
            .iter()
            .take(entry.shape.len() - 1)
            .product::<usize>()
            .max(1);
        let end_row = start_row
            .checked_add(row_count)
            .context("dense tensor tile row range overflow")?;
        if end_row > rows {
            bail!(
                "Flash-MoE dense tensor {} tile rows {}..{} exceed row count {}",
                entry.name,
                start_row,
                end_row,
                rows
            );
        }
        let row_bytes = cols
            .checked_mul(element_size)
            .context("dense tensor tile row byte length overflow")?;
        let byte_offset = start_row
            .checked_mul(row_bytes)
            .context("dense tensor tile byte offset overflow")?;
        let byte_len = row_count
            .checked_mul(row_bytes)
            .context("dense tensor tile byte length overflow")?;
        let (bytes, read_range) =
            self.read_range_profiled(entry.byte_offset + byte_offset as u64, byte_len)?;
        timing.read_range += read_range;
        timing.bytes_read = timing.bytes_read.saturating_add(byte_len as u64);
        #[cfg(test)]
        self.decoded_tensor_tiles
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let decode_started = Instant::now();
        let tensor = decode_dense_tensor_f32(&entry.dtype, &bytes)?;
        timing.decode += decode_started.elapsed();
        timing.decoded_bytes = timing
            .decoded_bytes
            .saturating_add((tensor.len() * std::mem::size_of::<f32>()) as u64);
        timing.total = started.elapsed();
        Ok((tensor, timing))
    }

    pub(super) fn read_tensor_row_f32(
        &self,
        canonical_name: &str,
        row: usize,
        requested_cols: usize,
    ) -> Result<Option<Vec<f32>>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            return Ok(None);
        };
        if let TensorQuantization::Q4 { group_size, .. } = entry.quantization {
            let cols = entry.shape.last().copied().unwrap_or(0);
            if entry.shape.is_empty() || requested_cols == 0 || cols == 0 {
                return Ok(None);
            }
            let rows = entry
                .shape
                .iter()
                .take(entry.shape.len() - 1)
                .product::<usize>()
                .max(1);
            if row >= rows {
                return Ok(None);
            }
            let (packed, scales, biases, _) = self.read_dense_q4_rows(entry, row, 1, group_size)?;
            let mut decoded =
                q4_dequantize_rows_with_group_size(&packed, &scales, &biases, 1, cols, group_size)?;
            decoded.truncate(requested_cols.min(cols));
            return Ok(Some(decoded));
        }
        let Some(element_size) = dtype_size(&entry.dtype) else {
            bail!(
                "Flash-MoE dense tensor {} has unsupported dtype {}",
                entry.name,
                entry.dtype
            );
        };
        if entry.shape.is_empty() || requested_cols == 0 {
            return Ok(None);
        }
        let cols = entry.shape.last().copied().unwrap_or(0);
        if cols == 0 {
            return Ok(None);
        }
        let rows = entry
            .shape
            .iter()
            .take(entry.shape.len() - 1)
            .product::<usize>()
            .max(1);
        if row >= rows {
            return Ok(None);
        }
        let used_cols = requested_cols.min(cols);
        let row_offset = row
            .checked_mul(cols)
            .and_then(|items| items.checked_mul(element_size))
            .context("dense tensor row offset overflow")? as u64;
        let byte_len = used_cols
            .checked_mul(element_size)
            .context("dense tensor row byte length overflow")?;
        let bytes = self.read_range(entry.byte_offset + row_offset, byte_len)?;
        Ok(Some(decode_dense_tensor_f32(&entry.dtype, &bytes)?))
    }

    pub(super) fn read_range(&self, offset: u64, byte_len: usize) -> Result<Vec<u8>> {
        let (bytes, _) = self.read_range_profiled(offset, byte_len)?;
        Ok(bytes)
    }

    pub(super) fn read_range_profiled(
        &self,
        offset: u64,
        byte_len: usize,
    ) -> Result<(Vec<u8>, Duration)> {
        if offset.saturating_add(byte_len as u64) > self.len {
            bail!(
                "dense tensor read {}..{} exceeds store length {}",
                offset,
                offset.saturating_add(byte_len as u64),
                self.len
            );
        }
        let started = Instant::now();
        let bytes = self.mmap[offset as usize..offset as usize + byte_len].to_vec();
        Ok((bytes, started.elapsed()))
    }

    #[cfg(test)]
    pub(super) fn tensor_seed(&self, canonical_name: &str, fallback: u64) -> u64 {
        if let Some(tensor) = self.registry.tensor(canonical_name) {
            stable_hash(&tensor.name)
                ^ stable_hash(&tensor.dtype)
                ^ tensor.byte_offset
                ^ tensor.byte_len.rotate_left(7)
                ^ ((tensor.shape.iter().copied().product::<usize>() as u64) << 11)
        } else {
            tracing::trace!(
                tensor = canonical_name,
                manifest = %self.manifest_path.display(),
                "Flash-MoE tensor registry missing canonical tensor; using deterministic fallback seed"
            );
            fallback
        }
    }

    pub(super) fn read_u64(&self, offset_hint: u64) -> Result<u64> {
        if self.len == 0 {
            return Ok(offset_hint.rotate_left(13) ^ 0x9e37_79b9_7f4a_7c15);
        }
        let offset = offset_hint % self.len;
        let mut out = [0u8; 8];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.mmap[((offset as usize) + i) % self.mmap.len()];
        }
        Ok(u64::from_le_bytes(out) ^ offset_hint.rotate_left(7))
    }

    /// Read a full 1-D or 2-D F32/BF16 tensor into a `Vec<f32>`.
    ///
    /// Returns `Ok(None)` when the tensor name is absent from the manifest.
    pub(super) fn read_full_tensor_f32(&self, canonical_name: &str) -> Result<Option<Vec<f32>>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            return Ok(None);
        };
        if entry.quantization != TensorQuantization::None {
            bail!("dense q4 tensor {canonical_name} cannot be read as a full f32 tensor");
        }
        let Some(_element_size) = dtype_size(&entry.dtype) else {
            bail!(
                "Flash-MoE dense tensor {} has unsupported dtype {}",
                entry.name,
                entry.dtype
            );
        };
        let byte_len = entry.byte_len as usize;
        let bytes = self.read_range(entry.byte_offset, byte_len)?;
        Ok(Some(decode_dense_tensor_f32(&entry.dtype, &bytes)?))
    }

    #[cfg(test)]
    pub(super) fn read_full_tensor_f32_cached(
        &self,
        canonical_name: &str,
    ) -> Result<Option<Arc<Vec<f32>>>> {
        self.dense_tensor_f32(canonical_name)
    }

    #[cfg(test)]
    pub(super) fn decoded_full_tensor_count(&self) -> usize {
        self.decoded_full_tensors
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(super) fn decoded_tensor_tile_count(&self) -> usize {
        self.decoded_tensor_tiles
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

pub(super) fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype.to_ascii_uppercase().as_str() {
        "F32" | "FLOAT32" | "FP32" => Some(4),
        "BF16" | "BFLOAT16" | "F16" | "FLOAT16" | "FP16" => Some(2),
        "U8" | "I8" => Some(1),
        _ => None,
    }
}

pub(super) fn decode_dense_tensor_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype.to_ascii_uppercase().as_str() {
        "F32" | "FLOAT32" | "FP32" => {
            if !bytes.len().is_multiple_of(4) {
                bail!(
                    "F32 tensor byte length {} is not divisible by 4",
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        "BF16" | "BFLOAT16" => {
            if !bytes.len().is_multiple_of(2) {
                bail!(
                    "BF16 tensor byte length {} is not divisible by 2",
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let hi = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                    f32::from_bits(hi << 16)
                })
                .collect())
        }
        "F16" | "FLOAT16" | "FP16" => {
            if !bytes.len().is_multiple_of(2) {
                bail!(
                    "F16 tensor byte length {} is not divisible by 2",
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect())
        }
        "U8" => Ok(bytes.iter().map(|value| *value as f32).collect()),
        "I8" => Ok(bytes.iter().map(|value| (*value as i8) as f32).collect()),
        other => bail!("unsupported dense tensor dtype {other}"),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = (bits & 0x03ff) as u32;
    let value = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut frac = frac;
                let mut exp = -14i32;
                while (frac & 0x0400) == 0 {
                    frac <<= 1;
                    exp -= 1;
                }
                frac &= 0x03ff;
                sign | (((exp + 127) as u32) << 23) | (frac << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13),
    };
    f32::from_bits(value)
}

#[cfg(test)]
#[path = "weights_parity_tests.rs"]
mod parity_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colibri_int4_import_preserves_nibbles_and_builds_affine_bias() {
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&[1, 64], 64, EXPERT_SCALE_BIAS_DTYPE_BF16)
                .unwrap();
        let packed = (0..32).map(|value| value as u8).collect::<Vec<_>>();
        let mut output = Vec::new();

        write_colibri_q4_affine_tensor(
            &mut output,
            "tiny.weight",
            &packed,
            &2.0f32.to_le_bytes(),
            4,
            64,
            layout,
        )
        .unwrap();

        assert_eq!(&output[..packed.len()], packed.as_slice());
        let scale = u16::from_le_bytes(output[32..34].try_into().unwrap()) as u32;
        let bias = u16::from_le_bytes(output[34..36].try_into().unwrap()) as u32;
        assert_eq!(f32::from_bits(scale << 16), 2.0);
        assert_eq!(f32::from_bits(bias << 16), -16.0);
    }

    #[test]
    fn colibri_int8_import_preserves_source_precision_as_bf16() {
        let mut output = Vec::new();
        write_colibri_int8_bf16_tensor(
            &mut output,
            "lm_head.weight",
            &[(-2i8) as u8, 3],
            &0.5f32.to_le_bytes(),
            2,
            &[1, 2],
        )
        .unwrap();

        let first = u16::from_le_bytes(output[0..2].try_into().unwrap()) as u32;
        let second = u16::from_le_bytes(output[2..4].try_into().unwrap()) as u32;
        assert_eq!(f32::from_bits(first << 16), -1.0);
        assert_eq!(f32::from_bits(second << 16), 1.5);
    }

    #[test]
    fn mlx_mxfp4_import_decodes_e2m1_and_e8m0_before_runtime_q4() {
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&[1, 64], 64, EXPERT_SCALE_BIAS_DTYPE_BF16)
                .unwrap();
        let mut packed = vec![0x91; 16]; // +0.5, -0.5 at E8M0 scale 1.
        packed.extend(vec![0xe6; 16]); // +4, -4 at E8M0 scale 2.
        let mut output = Vec::new();

        write_mlx_mxfp4_affine_tensor(&mut output, "tiny.weight", &packed, &[127, 128], 32, layout)
            .unwrap();

        assert_eq!(output.len(), layout.total_bytes);
        let scale_bits = u16::from_le_bytes(output[32..34].try_into().unwrap()) as u32;
        let bias_bits = u16::from_le_bytes(output[34..36].try_into().unwrap()) as u32;
        let decoded = q4_dequantize_rows_with_group_size(
            &output[..32],
            &[f32::from_bits(scale_bits << 16)],
            &[f32::from_bits(bias_bits << 16)],
            1,
            64,
            64,
        )
        .unwrap();
        for (actual, expected) in decoded[..32].iter().zip([0.5f32, -0.5].into_iter().cycle()) {
            assert!((actual - expected).abs() < 0.6, "{actual} != {expected}");
        }
        for (actual, expected) in decoded[32..].iter().zip([8.0f32, -8.0].into_iter().cycle()) {
            assert!((actual - expected).abs() < 0.6, "{actual} != {expected}");
        }
    }

    #[test]
    fn mla_weight_absorption_uses_compressed_latent_without_expanding_kv() {
        let temp = tempfile::tempdir().unwrap();
        let dense_path = temp.path().join("dense.bin");
        let manifest_path = temp.path().join("manifest.json");
        let tensor_name = attention_tensor_name(0, "kv_b_proj");
        let layout = dense_q4_layout_with_scale_bias_dtype(&[2, 2], 2, "F32").unwrap();
        let mut bytes = vec![0x01, 0x10]; // Wk=[1,0], Wv=[0,1]
        for value in [1.0f32, 1.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0.0f32, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(bytes.len(), layout.total_bytes);
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: "tiny-glm".to_string(),
                cache_version: "test".to_string(),
                dense_shards: vec!["tiny.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![DenseTensorRef {
                    tensor: tensor_name,
                    shard: "tiny.safetensors".to_string(),
                    dtype: "U32".to_string(),
                    shape: vec![2, 2],
                    source_offsets: [0, 2],
                    runtime_offset: 0,
                    byte_len: layout.total_bytes as u64,
                    quantization: TensorQuantization::Q4 {
                        group_size: 2,
                        format: "test".to_string(),
                        scale_bias_dtype: "F32".to_string(),
                    },
                    q4_sources: None,
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let mla = MlaAttentionLayout {
            q_lora_rank: 2,
            kv_lora_rank: 2,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 2,
            qk_head_dim: 3,
            v_head_dim: 1,
            num_heads: 1,
            q_width: 3,
            kv_a_width: 4,
            kv_b_width: 2,
            attention_output_width: 1,
            kv_projection: MlaKvProjectionLayout::FusedKvB,
        };
        let latent = [1.0, 3.0];
        let rotary = [0.0, 0.0];
        let output = store
            .mla_absorbed_attention(0, mla, &[2.0, 0.0, 0.0], &[(&latent, &rotary)])
            .unwrap();

        assert_eq!(output, vec![3.0]);
    }

    #[test]
    fn mla_weight_absorption_accepts_mlx_preabsorbed_multilinear_weights() {
        let temp = tempfile::tempdir().unwrap();
        let dense_path = temp.path().join("dense.bin");
        let manifest_path = temp.path().join("manifest.json");
        let embed_name = attention_tensor_name(0, "embed_q");
        let unembed_name = attention_tensor_name(0, "unembed_out");
        let embed_layout = dense_q4_layout_with_scale_bias_dtype(&[1, 2, 1], 1, "F32").unwrap();
        let unembed_layout = dense_q4_layout_with_scale_bias_dtype(&[1, 1, 2], 2, "F32").unwrap();
        let mut bytes = vec![0x01, 0x00]; // embed_q maps [q] to [q, 0].
        for value in [1.0f32, 1.0, 0.0, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(bytes.len(), embed_layout.total_bytes);
        let unembed_offset = bytes.len() as u64;
        bytes.push(0x10); // unembed_out maps [x, y] to y.
        for value in [1.0f32, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(
            bytes.len(),
            embed_layout.total_bytes + unembed_layout.total_bytes
        );
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: "tiny-glm-mlx".to_string(),
                cache_version: "test".to_string(),
                dense_shards: vec!["tiny.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![
                    DenseTensorRef {
                        tensor: embed_name,
                        shard: "tiny.safetensors".to_string(),
                        dtype: "U32".to_string(),
                        shape: vec![1, 2, 1],
                        source_offsets: [0, 2],
                        runtime_offset: 0,
                        byte_len: embed_layout.total_bytes as u64,
                        quantization: TensorQuantization::Q4 {
                            group_size: 1,
                            format: "test".to_string(),
                            scale_bias_dtype: "F32".to_string(),
                        },
                        q4_sources: None,
                    },
                    DenseTensorRef {
                        tensor: unembed_name,
                        shard: "tiny.safetensors".to_string(),
                        dtype: "U32".to_string(),
                        shape: vec![1, 1, 2],
                        source_offsets: [0, 1],
                        runtime_offset: unembed_offset,
                        byte_len: unembed_layout.total_bytes as u64,
                        quantization: TensorQuantization::Q4 {
                            group_size: 2,
                            format: "test".to_string(),
                            scale_bias_dtype: "F32".to_string(),
                        },
                        q4_sources: None,
                    },
                ],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let mla = MlaAttentionLayout {
            q_lora_rank: 2,
            kv_lora_rank: 2,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 2,
            qk_head_dim: 3,
            v_head_dim: 1,
            num_heads: 1,
            q_width: 3,
            kv_a_width: 4,
            kv_b_width: 2,
            attention_output_width: 1,
            kv_projection: MlaKvProjectionLayout::AbsorbedMultiLinear,
        };
        let latent = [1.0, 3.0];
        let rotary = [0.0, 0.0];
        let output = store
            .mla_absorbed_attention(0, mla, &[2.0, 0.0, 0.0], &[(&latent, &rotary)])
            .unwrap();

        assert_eq!(output, vec![3.0]);
    }

    #[test]
    fn qwen_family_tensor_names_are_canonicalized_for_runtime() {
        assert_eq!(
            canonical_hf_tensor_name("model.language_model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            canonical_hf_tensor_name("language_model.model.layers.3.self_attn.q_proj.weight"),
            "model.layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            canonical_hf_tensor_name("language_model.lm_head.weight"),
            "lm_head.weight"
        );
        assert_eq!(
            canonical_hf_tensor_name("model.visual.patch_embed.proj.weight"),
            "visual.patch_embed.proj.weight"
        );
        assert_eq!(
            canonical_hf_tensor_name("vision_tower.blocks.7.mlp.linear_fc1.weight"),
            "visual.blocks.7.mlp.fc1.weight"
        );
        assert_eq!(
            canonical_hf_tensor_name("vision_tower.merger.linear_fc2.weight"),
            "visual.merger.linear_fc2.weight"
        );
        assert_eq!(canonical_hf_tensor_name("lm_head.weight"), "lm_head.weight");
    }

    fn layout_config() -> QwenModelConfig {
        QwenModelConfig {
            model_type: Some("qwen3_moe".to_string()),
            architectures: Some(vec!["Qwen3MoeForCausalLM".to_string()]),
            num_hidden_layers: 1,
            hidden_size: 8,
            num_attention_heads: 2,
            head_dim: Some(4),
            num_key_value_heads: Some(1),
            vocab_size: 32,
            rope_theta: Some(1_000_000.0),
            partial_rotary_factor: None,
            torch_dtype: Some("float32".to_string()),
            num_experts: Some(4),
            num_experts_per_tok: Some(2),
            norm_topk_prob: Some(true),
            moe_intermediate_size: Some(16),
            intermediate_size: None,
            max_position_embeddings: Some(1024),
            mrope_section: None,
            tie_word_embeddings: Some(true),
            num_shared_experts: None,
            shared_expert_intermediate_size: None,
            vision_config: None,
            glm: None,
        }
    }

    fn layout_tensor(name: &str, shape: &[usize]) -> RuntimeTensorEntry {
        RuntimeTensorEntry {
            name: name.to_string(),
            dtype: "F32".to_string(),
            shape: shape.to_vec(),
            byte_offset: 0,
            byte_len: shape.iter().product::<usize>() as u64 * 4,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        }
    }

    #[test]
    fn dense_runtime_layout_resolves_full_attention_from_manifest_shapes() {
        let registry = TensorRegistry {
            tensors: BTreeMap::from([
                (
                    attention_tensor_name(0, "q_proj"),
                    layout_tensor(&attention_tensor_name(0, "q_proj"), &[8, 8]),
                ),
                (
                    attention_tensor_name(0, "k_proj"),
                    layout_tensor(&attention_tensor_name(0, "k_proj"), &[4, 8]),
                ),
                (
                    attention_tensor_name(0, "v_proj"),
                    layout_tensor(&attention_tensor_name(0, "v_proj"), &[4, 8]),
                ),
                (
                    attention_tensor_name(0, "o_proj"),
                    layout_tensor(&attention_tensor_name(0, "o_proj"), &[8, 8]),
                ),
            ]),
        };

        let runtime = DenseTransformerRuntime::from_registry(&layout_config(), &registry).unwrap();
        let layout = runtime.full_attention_layout(0).unwrap();
        assert_eq!(layout.q_layout, FullAttentionQLayout::Standard);
        assert_eq!(layout.q_width, 8);
        assert_eq!(layout.kv_width, 4);
        assert_eq!(layout.head_dim, 4);
        assert_eq!(layout.rotary_dim, 4);
    }

    #[test]
    fn dense_runtime_layout_rejects_mixed_attention_implementations() {
        let mut tensors = BTreeMap::from([(
            attention_tensor_name(0, "q_proj"),
            layout_tensor(&attention_tensor_name(0, "q_proj"), &[8, 8]),
        )]);
        tensors.insert(
            linear_attention_tensor_name(0, "in_proj_qkv"),
            layout_tensor(&linear_attention_tensor_name(0, "in_proj_qkv"), &[8, 8]),
        );
        let error =
            DenseTransformerRuntime::from_registry(&layout_config(), &TensorRegistry { tensors })
                .unwrap_err();

        assert!(error.to_string().contains("both linear-attention tensors"));
    }

    #[test]
    fn required_manifest_validation_resolves_complete_full_attention_layer() {
        let entries = [
            ("model.embed_tokens.weight".to_string(), vec![32, 8]),
            ("model.norm.weight".to_string(), vec![8]),
            (attention_tensor_name(0, "q_proj"), vec![8, 8]),
            (attention_tensor_name(0, "k_proj"), vec![4, 8]),
            (attention_tensor_name(0, "v_proj"), vec![4, 8]),
            (attention_tensor_name(0, "o_proj"), vec![8, 8]),
            (layer_norm_tensor_name(0, "self_attn.q_norm"), vec![4]),
            (layer_norm_tensor_name(0, "self_attn.k_norm"), vec![4]),
            (layer_norm_tensor_name(0, "input_layernorm"), vec![8]),
            (
                layer_norm_tensor_name(0, "post_attention_layernorm"),
                vec![8],
            ),
            (router_tensor_name(0), vec![4, 8]),
        ];
        let registry = TensorRegistry {
            tensors: entries
                .into_iter()
                .map(|(name, shape)| {
                    let tensor = layout_tensor(&name, &shape);
                    (name, tensor)
                })
                .collect(),
        };

        validate_required_tensor_manifest(&layout_config(), &registry).unwrap();
    }

    #[test]
    fn dense_cache_conversion_resolves_native_mlx_q4_layout() {
        assert_eq!(logical_shape_for_mlx_q4(&[3, 4]).unwrap(), vec![3, 32]);
        let native = DenseQ4SourceRefs {
            scales_shard: "scales.safetensors".to_string(),
            scales_offsets: [0, 8],
            biases_shard: "biases.safetensors".to_string(),
            biases_offsets: [0, 8],
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            source_format: DenseQ4SourceFormat::MlxAffine,
            source_group_size: None,
        };

        assert_eq!(
            dense_tensor_quantization(
                "model.layers.0.self_attn.q_proj.weight",
                "U32",
                &Some(native)
            ),
            TensorQuantization::Q4 {
                group_size: GROUP_SIZE,
                format: DENSE_Q4_MLX_FORMAT.to_string(),
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            }
        );
    }

    fn runtime_matrix(
        name: &str,
        dtype: &str,
        quantization: TensorQuantization,
    ) -> RuntimeTensorEntry {
        RuntimeTensorEntry {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape: vec![4, 8],
            byte_offset: 0,
            byte_len: 128,
            alignment: TENSOR_ALIGNMENT,
            quantization,
        }
    }

    #[test]
    fn tensor_quantization_defaults_to_unquantized_dense() {
        assert_eq!(TensorQuantization::default(), TensorQuantization::None);
    }

    #[test]
    fn tensor_quantization_q4_defaults_scale_bias_dtype_for_legacy_manifests() {
        let quantization: TensorQuantization =
            serde_json::from_str(r#"{"Q4":{"group_size":16,"format":"dense-q4"}}"#).unwrap();

        assert_eq!(
            quantization,
            TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "F32".to_string(),
            }
        );
    }

    #[test]
    fn tensor_registry_resolves_one_concrete_dense_layout() {
        for (dtype, expected) in [
            ("BF16", ResidentDenseLayout::Bf16),
            ("F16", ResidentDenseLayout::F16),
            ("F32", ResidentDenseLayout::F32),
        ] {
            let registry = TensorRegistry {
                tensors: BTreeMap::from([(
                    "model.layers.0.self_attn.q_proj.weight".to_string(),
                    runtime_matrix(
                        "model.layers.0.self_attn.q_proj.weight",
                        dtype,
                        TensorQuantization::None,
                    ),
                )]),
            };

            assert_eq!(registry.resolve_resident_dense_layout().unwrap(), expected);
        }
    }

    #[test]
    fn tensor_registry_resolves_q4_with_unquantized_auxiliary_matrices() {
        let registry = TensorRegistry {
            tensors: BTreeMap::from([
                (
                    "model.layers.0.self_attn.q_proj.weight".to_string(),
                    runtime_matrix(
                        "model.layers.0.self_attn.q_proj.weight",
                        "U32",
                        TensorQuantization::Q4 {
                            group_size: 64,
                            format: "mlx-q4".to_string(),
                            scale_bias_dtype: "BF16".to_string(),
                        },
                    ),
                ),
                (
                    "model.embed_tokens.weight".to_string(),
                    runtime_matrix(
                        "model.embed_tokens.weight",
                        "BF16",
                        TensorQuantization::None,
                    ),
                ),
            ]),
        };

        assert_eq!(
            registry.resolve_resident_dense_layout().unwrap(),
            ResidentDenseLayout::Q4
        );
    }

    #[test]
    fn tensor_registry_ignores_routed_expert_storage_when_resolving_dense_layout() {
        let registry = TensorRegistry {
            tensors: BTreeMap::from([
                (
                    "model.layers.0.self_attn.q_proj.weight".to_string(),
                    runtime_matrix(
                        "model.layers.0.self_attn.q_proj.weight",
                        "BF16",
                        TensorQuantization::None,
                    ),
                ),
                (
                    "model.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
                    runtime_matrix(
                        "model.layers.0.mlp.experts.0.gate_proj.weight",
                        "U32",
                        TensorQuantization::Q4 {
                            group_size: 64,
                            format: "expert-q4".to_string(),
                            scale_bias_dtype: "F32".to_string(),
                        },
                    ),
                ),
            ]),
        };

        assert_eq!(
            registry.resolve_resident_dense_layout().unwrap(),
            ResidentDenseLayout::Bf16
        );
    }

    #[test]
    fn tensor_registry_rejects_mixed_unquantized_matrix_layouts() {
        let registry = TensorRegistry {
            tensors: BTreeMap::from([
                (
                    "model.layers.0.self_attn.q_proj.weight".to_string(),
                    runtime_matrix(
                        "model.layers.0.self_attn.q_proj.weight",
                        "BF16",
                        TensorQuantization::None,
                    ),
                ),
                (
                    "lm_head.weight".to_string(),
                    runtime_matrix("lm_head.weight", "F32", TensorQuantization::None),
                ),
            ]),
        };

        let err = registry.resolve_resident_dense_layout().unwrap_err();
        assert!(
            err.to_string().contains("mixes resident matrix layouts"),
            "{err:#}"
        );
    }

    #[test]
    fn dense_tensor_ref_preserves_runtime_binding_offsets() {
        let tensor = DenseTensorRef {
            tensor: "model.embed_tokens.weight".to_string(),
            shard: "model-00001.safetensors".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![8, 4],
            source_offsets: [128, 192],
            runtime_offset: 4096,
            byte_len: 64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        };

        assert_eq!(tensor.runtime_offset, 4096);
        assert_eq!(tensor.byte_len, 64);
        assert_eq!(tensor.quantization, TensorQuantization::None);
    }

    #[test]
    fn tensor_registry_builds_dense_aliases_from_manifest() {
        let manifest = FlashMoeManifest {
            model: "fixture".to_string(),
            cache_version: "test".to_string(),
            dense_shards: Vec::new(),
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: "model.language_model.layers.7.self_attn.q_proj.weight".to_string(),
                shard: "model.safetensors".to_string(),
                dtype: "BF16".to_string(),
                shape: vec![4, 8],
                source_offsets: [0, 64],
                runtime_offset: 4096,
                byte_len: 64,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };

        let registry = TensorRegistry::from_manifest(&manifest);
        let alias = registry
            .tensor("model.layers.7.self_attn.q_proj.weight")
            .unwrap();

        assert_eq!(
            alias.name,
            "model.language_model.layers.7.self_attn.q_proj.weight"
        );
        assert_eq!(alias.byte_offset, 4096);
        assert_eq!(alias.alignment, TENSOR_ALIGNMENT);
        assert!(registry.has_tensor_with_prefix("model.layers.7"));
    }

    #[test]
    fn tensor_registry_keeps_expert_manifest_refs_as_import_compatibility() {
        let manifest = FlashMoeManifest {
            model: "fixture".to_string(),
            cache_version: "test".to_string(),
            dense_shards: Vec::new(),
            dense_tensors: Vec::new(),
            expert_tensors: vec![ExpertTensorRef {
                tensor: "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
                shard: "model.safetensors".to_string(),
                layer: Some(0),
                expert: Some(1),
                dtype: Some("F32".to_string()),
                shape: vec![2, 4],
                source_offsets: Some([128, 256]),
                q4_sources: None,
            }],
        };

        let registry = TensorRegistry::from_manifest(&manifest);
        let tensor = registry
            .tensor("model.layers.0.mlp.experts.1.gate_proj.weight")
            .unwrap();

        assert_eq!(tensor.byte_offset, 128);
        assert_eq!(tensor.byte_len, 128);
        assert_eq!(
            tensor.quantization,
            TensorQuantization::Q4 {
                group_size: GROUP_SIZE,
                format: ExpertQuantization::FourBitProduction.as_str().to_string(),
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
            }
        );
    }

    #[test]
    fn dense_mmap_projection_stride_uses_runtime_cols() {
        let projection = DenseMmapMatvecProjection {
            tensor_name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            byte_offset: 4096,
            dtype: "BF16".to_string(),
            rows: 16,
            cols: 32,
            output_width: 64,
        };

        assert_eq!(projection.stride(), 32);
    }

    #[test]
    fn dense_mmap_projection_descriptor_resolves_entry_bounds() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![4, 8],
            byte_offset: 64,
            byte_len: 64,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let projection = DenseMmapMatvecProjection::from_entry(
            "model.layers.0.self_attn.q_proj.weight",
            &entry,
            256,
            4,
            8,
            2,
        )
        .unwrap();

        assert_eq!(projection.byte_offset, 64);
        assert_eq!(projection.rows, 4);
        assert_eq!(projection.cols, 8);
        assert_eq!(projection.output_width, 4);
    }

    #[test]
    fn resident_projection_binding_resolves_bf16_f16_and_f32_without_layout_probe() {
        for (dtype, element_size) in [("BF16", 2), ("F16", 2), ("F32", 4)] {
            let entry = RuntimeTensorEntry {
                name: format!("model.layers.0.self_attn.{dtype}_proj.weight"),
                dtype: dtype.to_string(),
                shape: vec![3, 4],
                byte_offset: 64,
                byte_len: (12 * element_size) as u64,
                alignment: TENSOR_ALIGNMENT,
                quantization: TensorQuantization::None,
            };

            let projection =
                ResidentMmapMatvecProjection::from_entry(&entry.name, &entry, 256, 3, 4).unwrap();
            assert_eq!(projection.tensor_name(), entry.name);
            assert_eq!(projection.rows(), 3);
            assert_eq!(projection.cols(), 4);
            assert_eq!(projection.output_width(), 3);
            assert!(matches!(projection, ResidentMmapMatvecProjection::Dense(_)));
        }

        let unsupported = RuntimeTensorEntry {
            name: "model.layers.0.self_attn.i8_proj.weight".to_string(),
            dtype: "I8".to_string(),
            shape: vec![3, 4],
            byte_offset: 64,
            byte_len: 12,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };
        let error =
            ResidentMmapMatvecProjection::from_entry(&unsupported.name, &unsupported, 256, 3, 4)
                .unwrap_err();
        assert!(
            error.to_string().contains("unsupported dtype I8"),
            "{error:#}"
        );
    }

    #[test]
    fn router_score_projection_descriptor_resolves_dense_binding() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 4],
            byte_offset: 64,
            byte_len: 32,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let descriptor =
            RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 128, 2, 4).unwrap();

        assert_eq!(descriptor.layer, 3);
        assert_eq!(descriptor.experts, 2);
        assert_eq!(descriptor.hidden_width, 4);
        match descriptor.binding {
            RouterScoreProjectionBinding::ResidentDense(projection) => {
                assert_eq!(projection.tensor_name, entry.name);
                assert_eq!(projection.byte_offset, 64);
                assert_eq!(projection.output_width, 2);
            }
            RouterScoreProjectionBinding::ResidentQ4(_) => panic!("expected dense binding"),
        }
    }

    #[test]
    fn router_score_projection_topk_plan_declares_dense_binding() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 4],
            byte_offset: 64,
            byte_len: 32,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };
        let descriptor =
            RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 128, 2, 4).unwrap();

        let plan = descriptor.topk_plan(4, 1).unwrap();
        assert_eq!(plan.layer, 3);
        assert_eq!(plan.tensor_name, entry.name);
        assert_eq!(plan.experts, 2);
        assert_eq!(plan.hidden_width, 4);
        assert_eq!(plan.active_experts, 1);
        match plan.source {
            RouterScoreProjectionTopKSource::ResidentDense(projection) => {
                assert_eq!(projection.byte_offset, 64);
                assert_eq!(projection.rows, 2);
                assert_eq!(projection.cols, 4);
            }
            RouterScoreProjectionTopKSource::ResidentQ4(_) => panic!("expected dense topK plan"),
        }

        let hidden_err = descriptor.topk_plan(3, 1).unwrap_err();
        assert!(
            hidden_err
                .to_string()
                .contains("topK hidden length 3 does not match declared width 4"),
            "{hidden_err:#}"
        );
        let active_err = descriptor.topk_plan(4, 0).unwrap_err();
        assert!(
            active_err
                .to_string()
                .contains("active experts 0 is outside declared expert range 1..=2"),
            "{active_err:#}"
        );
    }

    #[test]
    fn router_score_projection_descriptor_resolves_q4_binding() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            byte_offset: 128,
            byte_len: 12,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "BF16".to_string(),
            },
        };

        let descriptor =
            RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 256, 2, 4).unwrap();

        match descriptor.binding {
            RouterScoreProjectionBinding::ResidentQ4(projection) => {
                assert_eq!(projection.packed_byte_offset, 128);
                assert_eq!(projection.output_width, 2);
                assert_eq!(projection.cols, 4);
            }
            RouterScoreProjectionBinding::ResidentDense(_) => panic!("expected q4 binding"),
        }
    }

    #[test]
    fn router_score_projection_topk_plan_declares_q4_binding() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            byte_offset: 128,
            byte_len: 12,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "BF16".to_string(),
            },
        };
        let descriptor =
            RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 256, 2, 4).unwrap();

        let plan = descriptor.topk_plan(4, 2).unwrap();
        assert_eq!(plan.active_experts, 2);
        match plan.source {
            RouterScoreProjectionTopKSource::ResidentQ4(projection) => {
                assert_eq!(projection.packed_byte_offset, 128);
                assert_eq!(projection.output_width, 2);
                assert_eq!(projection.cols, 4);
            }
            RouterScoreProjectionTopKSource::ResidentDense(_) => panic!("expected q4 topK plan"),
        }
    }

    #[test]
    fn router_score_projection_builder_uses_canonical_layer_tensor_name() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 4],
            byte_offset: 64,
            byte_len: 32,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };
        let mut seen_name = None;

        let descriptor = build_router_score_projection_descriptor(3, 2, 4, 128, |name| {
            seen_name = Some(name.to_string());
            (name == entry.name).then_some(&entry)
        })
        .unwrap()
        .unwrap();

        assert_eq!(seen_name.unwrap(), "model.layers.3.mlp.gate.weight");
        assert_eq!(descriptor.tensor_name, entry.name);
        assert_eq!(descriptor.experts, 2);
        assert_eq!(descriptor.hidden_width, 4);
    }

    #[test]
    fn router_score_projection_builder_returns_none_for_missing_router() {
        let descriptor = build_router_score_projection_descriptor(3, 2, 4, 128, |_| None).unwrap();

        assert!(descriptor.is_none());
    }

    #[test]
    fn router_score_projection_builder_rejects_wrong_shape_without_fallback() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![3, 4],
            byte_offset: 0,
            byte_len: 48,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let err = build_router_score_projection_descriptor(3, 2, 4, 64, |name| {
            (name == entry.name).then_some(&entry)
        })
        .unwrap_err();

        assert!(err.to_string().contains("shape mismatch"), "{err:#}");
    }

    #[test]
    fn router_score_projection_execution_declares_binding_without_fallback() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 4],
            byte_offset: 64,
            byte_len: 32,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };
        let descriptor =
            RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 128, 2, 4).unwrap();

        let execution = descriptor.execution(3, 2, 4).unwrap();
        assert_eq!(execution.layer, 3);
        assert_eq!(execution.tensor_name, entry.name);
        assert_eq!(execution.experts, 2);
        assert_eq!(execution.hidden_width, 4);
        assert_eq!(
            execution.kind,
            RouterScoreProjectionExecutionKind::ResidentDense
        );
        let score_plan = execution.score_plan(4).unwrap();
        assert_eq!(score_plan.tensor_name, entry.name);
        assert_eq!(score_plan.experts, 2);
        assert_eq!(score_plan.hidden_width, 4);
        assert_eq!(
            score_plan.source,
            RouterScoreProjectionScoreSource::ResidentDenseFullTensor
        );

        let hidden_err = execution.score_plan(3).unwrap_err();
        assert!(
            hidden_err
                .to_string()
                .contains("hidden length 3 does not match declared width 4"),
            "{hidden_err:#}"
        );

        let err = descriptor.execution(3, 3, 4).unwrap_err();
        assert!(
            err.to_string()
                .contains("experts 2 does not match scheduled experts 3"),
            "{err:#}"
        );
    }

    #[test]
    fn router_score_projection_score_plan_declares_q4_row_execution() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 32],
            byte_offset: 64,
            byte_len: 48,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "BF16".to_string(),
            },
        };
        let descriptor =
            RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 512, 2, 32)
                .unwrap();

        let execution = descriptor.execution(3, 2, 32).unwrap();
        assert_eq!(
            execution.kind,
            RouterScoreProjectionExecutionKind::ResidentQ4
        );
        assert_eq!(
            execution.score_plan(32).unwrap().source,
            RouterScoreProjectionScoreSource::DeclaredRows
        );
    }

    #[test]
    fn router_score_projection_descriptor_rejects_wrong_shape_without_fallback() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![3, 4],
            byte_offset: 0,
            byte_len: 48,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let err = RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 64, 2, 4)
            .unwrap_err();

        assert!(err.to_string().contains("shape mismatch"), "{err:#}");
    }

    #[test]
    fn cmd2_resident_post_attention_prep_projection_bundle_resolves_bindings() {
        let projections = build_cmd2_resident_post_attention_prep_projections(
            7,
            16,
            "model.layers.7.self_attn.o_proj.weight",
            24,
            32,
            4,
            |name, output_width, input_len| {
                Ok(Some(ResidentMmapMatvecProjection::Q4(
                    DenseQ4MmapMatvecProjection {
                        tensor_name: name.to_string(),
                        packed_byte_offset: 128,
                        scales_byte_offset: 256,
                        biases_byte_offset: 512,
                        rows: output_width,
                        cols: input_len,
                        output_width,
                        row_packed_bytes: input_len.div_ceil(2),
                        groups_per_row: input_len.div_ceil(16),
                        group_size: 16,
                        scale_bias_dtype: "BF16".to_string(),
                    },
                )))
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(projections.layer, 7);
        assert_eq!(projections.experts, 16);
        assert_eq!(projections.residual_width, 32);
        assert_eq!(projections.attention_width, 24);
        assert_eq!(projections.active_experts, 4);
        assert_eq!(
            projections.out_proj.tensor_name(),
            "model.layers.7.self_attn.o_proj.weight"
        );
        assert_eq!(
            projections.router.tensor_name(),
            "model.layers.7.mlp.gate.weight"
        );
        assert_eq!(projections.out_proj.output_width(), 32);
        assert_eq!(projections.out_proj.cols(), 24);
        assert_eq!(projections.router.output_width(), 16);
        assert_eq!(projections.router.cols(), 32);
    }

    #[test]
    fn cmd2_resident_post_attention_prep_plan_declares_executable_shape() {
        let projections = build_cmd2_resident_post_attention_prep_projections(
            7,
            16,
            "model.layers.7.self_attn.o_proj.weight",
            24,
            32,
            20,
            |name, output_width, input_len| {
                Ok(Some(ResidentMmapMatvecProjection::Q4(
                    DenseQ4MmapMatvecProjection {
                        tensor_name: name.to_string(),
                        packed_byte_offset: 128,
                        scales_byte_offset: 256,
                        biases_byte_offset: 512,
                        rows: output_width,
                        cols: input_len,
                        output_width,
                        row_packed_bytes: input_len.div_ceil(2),
                        groups_per_row: input_len.div_ceil(16),
                        group_size: 16,
                        scale_bias_dtype: "BF16".to_string(),
                    },
                )))
            },
        )
        .unwrap()
        .unwrap();

        let plan = projections.resident_plan(24, 32, 32).unwrap();

        assert_eq!(
            plan,
            Cmd2ResidentPostAttentionPrepPlan {
                layer: 7,
                width: 32,
                attention_width: 24,
                experts: 16,
                active_count: 16,
            }
        );
    }

    #[test]
    fn cmd2_resident_post_attention_prep_plan_rejects_undeclared_inputs() {
        let projections = build_cmd2_resident_post_attention_prep_projections(
            7,
            16,
            "model.layers.7.self_attn.o_proj.weight",
            24,
            32,
            4,
            |name, output_width, input_len| {
                Ok(Some(ResidentMmapMatvecProjection::Q4(
                    DenseQ4MmapMatvecProjection {
                        tensor_name: name.to_string(),
                        packed_byte_offset: 128,
                        scales_byte_offset: 256,
                        biases_byte_offset: 512,
                        rows: output_width,
                        cols: input_len,
                        output_width,
                        row_packed_bytes: input_len.div_ceil(2),
                        groups_per_row: input_len.div_ceil(16),
                        group_size: 16,
                        scale_bias_dtype: "BF16".to_string(),
                    },
                )))
            },
        )
        .unwrap()
        .unwrap();

        let norm_err = projections.resident_plan(24, 32, 31).unwrap_err();
        assert!(
            norm_err
                .to_string()
                .contains("norm weight length 31 does not match residual width 32"),
            "{norm_err:#}"
        );
    }

    #[test]
    fn cmd2_resident_post_attention_prep_plan_errors_on_undeclared_shape() {
        let projections = Cmd2ResidentPostAttentionPrepProjections {
            layer: 7,
            out_proj: ResidentMmapMatvecProjection::Q4(DenseQ4MmapMatvecProjection {
                tensor_name: "model.layers.7.self_attn.o_proj.weight".to_string(),
                packed_byte_offset: 128,
                scales_byte_offset: 256,
                biases_byte_offset: 512,
                rows: 32,
                cols: 25,
                output_width: 32,
                row_packed_bytes: 13,
                groups_per_row: 2,
                group_size: 16,
                scale_bias_dtype: "BF16".to_string(),
            }),
            router: ResidentMmapMatvecProjection::Q4(DenseQ4MmapMatvecProjection {
                tensor_name: "model.layers.7.mlp.gate.weight".to_string(),
                packed_byte_offset: 128,
                scales_byte_offset: 256,
                biases_byte_offset: 512,
                rows: 16,
                cols: 32,
                output_width: 16,
                row_packed_bytes: 16,
                groups_per_row: 2,
                group_size: 16,
                scale_bias_dtype: "BF16".to_string(),
            }),
            experts: 16,
            residual_width: 32,
            attention_width: 24,
            active_experts: 4,
        };

        let err = projections.resident_plan(24, 32, 32).unwrap_err();
        assert!(
            err.to_string()
                .contains("projection shapes out=32x25 rows=32 router=16x32 rows=16"),
            "{err:#}"
        );
    }

    #[test]
    fn cmd2_resident_post_attention_prep_projection_bundle_skips_missing_bindings() {
        let missing_out = build_cmd2_resident_post_attention_prep_projections(
            7,
            16,
            "model.layers.7.self_attn.o_proj.weight",
            24,
            32,
            4,
            |name, output_width, input_len| {
                if name.ends_with("o_proj.weight") {
                    return Ok(None);
                }
                Ok(Some(ResidentMmapMatvecProjection::Q4(
                    DenseQ4MmapMatvecProjection {
                        tensor_name: name.to_string(),
                        packed_byte_offset: 128,
                        scales_byte_offset: 256,
                        biases_byte_offset: 512,
                        rows: output_width,
                        cols: input_len,
                        output_width,
                        row_packed_bytes: input_len.div_ceil(2),
                        groups_per_row: input_len.div_ceil(16),
                        group_size: 16,
                        scale_bias_dtype: "BF16".to_string(),
                    },
                )))
            },
        )
        .unwrap();
        assert!(missing_out.is_none());

        let disabled = build_cmd2_resident_post_attention_prep_projections(
            7,
            0,
            "out",
            24,
            32,
            4,
            |_, _, _| panic!("disabled CMD2 prep must not request projections"),
        )
        .unwrap();
        assert!(disabled.is_none());
    }

    #[test]
    fn required_cmd2_resident_post_attention_prep_projection_errors_on_missing_bindings() {
        let missing_out = build_required_cmd2_resident_post_attention_prep_projections(
            7,
            16,
            "model.layers.7.self_attn.o_proj.weight",
            24,
            32,
            4,
            |name, output_width, input_len| {
                if name.ends_with("o_proj.weight") {
                    return Ok(None);
                }
                Ok(Some(ResidentMmapMatvecProjection::Q4(
                    DenseQ4MmapMatvecProjection {
                        tensor_name: name.to_string(),
                        packed_byte_offset: 128,
                        scales_byte_offset: 256,
                        biases_byte_offset: 512,
                        rows: output_width,
                        cols: input_len,
                        output_width,
                        row_packed_bytes: input_len.div_ceil(2),
                        groups_per_row: input_len.div_ceil(16),
                        group_size: 16,
                        scale_bias_dtype: "BF16".to_string(),
                    },
                )))
            },
        )
        .unwrap_err();
        assert!(
            missing_out
                .to_string()
                .contains("missing output projection"),
            "{missing_out:#}"
        );

        let missing_router = build_required_cmd2_resident_post_attention_prep_projections(
            7,
            16,
            "model.layers.7.self_attn.o_proj.weight",
            24,
            32,
            4,
            |name, output_width, input_len| {
                if name.ends_with("mlp.gate.weight") {
                    return Ok(None);
                }
                Ok(Some(ResidentMmapMatvecProjection::Q4(
                    DenseQ4MmapMatvecProjection {
                        tensor_name: name.to_string(),
                        packed_byte_offset: 128,
                        scales_byte_offset: 256,
                        biases_byte_offset: 512,
                        rows: output_width,
                        cols: input_len,
                        output_width,
                        row_packed_bytes: input_len.div_ceil(2),
                        groups_per_row: input_len.div_ceil(16),
                        group_size: 16,
                        scale_bias_dtype: "BF16".to_string(),
                    },
                )))
            },
        )
        .unwrap_err();
        assert!(
            missing_router
                .to_string()
                .contains("missing router projection model.layers.7.mlp.gate.weight"),
            "{missing_router:#}"
        );
    }

    #[test]
    fn cmd2_resident_post_attention_prep_projection_bundle_rejects_mismatched_shape() {
        let err = build_cmd2_resident_post_attention_prep_projections(
            7,
            16,
            "model.layers.7.self_attn.o_proj.weight",
            24,
            32,
            4,
            |name, output_width, input_len| {
                Ok(Some(ResidentMmapMatvecProjection::Q4(
                    DenseQ4MmapMatvecProjection {
                        tensor_name: name.to_string(),
                        packed_byte_offset: 128,
                        scales_byte_offset: 256,
                        biases_byte_offset: 512,
                        rows: output_width,
                        cols: input_len + usize::from(name.ends_with("o_proj.weight")),
                        output_width,
                        row_packed_bytes: input_len.div_ceil(2),
                        groups_per_row: input_len.div_ceil(16),
                        group_size: 16,
                        scale_bias_dtype: "BF16".to_string(),
                    },
                )))
            },
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("CMD2 resident post-attention output projection shape is invalid")
        );
    }

    #[test]
    fn router_score_batch_keeps_projection_with_scores() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 4],
            byte_offset: 64,
            byte_len: 32,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };
        let projection =
            RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 128, 2, 4).unwrap();

        let state = FlashMoeRoutingOutputState::cpu_router_scores(3, 2, 1);
        let batch = RouterScoreBatch::new(state, Some(projection), vec![1.0, -2.0]).unwrap();

        assert_eq!(batch.state(), state);
        assert_eq!(batch.scores, vec![1.0, -2.0]);
        assert_eq!(batch.projection.as_ref().unwrap().layer, 3);
        assert_eq!(batch.projection.as_ref().unwrap().experts, 2);
    }

    #[test]
    fn router_score_batch_rejects_scores_outside_declared_state() {
        let err = RouterScoreBatch::new(
            FlashMoeRoutingOutputState::cpu_router_scores(3, 2, 1),
            None,
            vec![1.0],
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("1 scores for 2 declared experts"),
            "{err:#}"
        );
    }

    #[test]
    fn resident_static_tensor_descriptor_resolves_offsets_and_dtype() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.self_attn.conv1d.weight".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![8],
            byte_offset: 16,
            byte_len: 16,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let resident = ResidentStaticTensorRef::from_entry(
            &entry.name,
            &entry,
            64,
            8,
            &[ResidentStaticDtype::Bf16],
        )
        .unwrap()
        .unwrap();

        assert_eq!(resident.tensor_name, entry.name);
        assert_eq!(resident.byte_offset, 16);
        assert_eq!(resident.dtype, ResidentStaticDtype::Bf16);
        assert_eq!(resident.values, 8);
        assert_eq!(resident.element_size, 2);
    }

    #[test]
    fn resident_static_tensor_descriptor_rejects_wrong_layout_without_fallback() {
        let mut entry = RuntimeTensorEntry {
            name: "model.layers.0.self_attn.A_log".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4],
            byte_offset: 4,
            byte_len: 16,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        assert!(
            ResidentStaticTensorRef::from_entry(
                &entry.name,
                &entry,
                32,
                4,
                &[ResidentStaticDtype::F32],
            )
            .unwrap()
            .is_some()
        );

        entry.byte_len = 12;
        assert!(
            ResidentStaticTensorRef::from_entry(
                &entry.name,
                &entry,
                32,
                4,
                &[ResidentStaticDtype::F32],
            )
            .unwrap()
            .is_none()
        );

        entry.byte_len = 16;
        entry.quantization = TensorQuantization::Q4 {
            group_size: GROUP_SIZE,
            format: "mlx".to_string(),
            scale_bias_dtype: "F32".to_string(),
        };
        assert!(
            ResidentStaticTensorRef::from_entry(
                &entry.name,
                &entry,
                32,
                4,
                &[ResidentStaticDtype::F32],
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn linear_attention_weight_table_resolves_all_resident_dense_layouts_at_load() {
        fn append_tensor(
            bytes: &mut Vec<u8>,
            tensors: &mut Vec<DenseTensorRef>,
            name: String,
            dtype: &str,
            shape: Vec<usize>,
        ) {
            while !(bytes.len() as u64).is_multiple_of(TENSOR_ALIGNMENT) {
                bytes.push(0);
            }
            let runtime_offset = bytes.len() as u64;
            let element_size = dense_dtype_size(dtype).unwrap();
            let byte_len = shape.iter().product::<usize>() * element_size;
            bytes.resize(bytes.len() + byte_len, 0);
            tensors.push(DenseTensorRef {
                tensor: name,
                shard: "fixture.safetensors".to_string(),
                dtype: dtype.to_string(),
                shape,
                source_offsets: [0, byte_len as u64],
                runtime_offset,
                byte_len: byte_len as u64,
                quantization: TensorQuantization::None,
                q4_sources: None,
            });
        }

        fn fixture(dtype: &str, omit_norm: bool) -> (tempfile::TempDir, DenseStore) {
            let hidden = 4;
            let experts = 3;
            let layout = LinearAttentionLayout {
                num_value_heads: 2,
                num_key_heads: 1,
                key_dim: 2,
                value_dim: 2,
                total_key_width: 2,
                total_value_width: 4,
                conv_dim: 8,
                conv_kernel_size: 2,
            };
            let mut bytes = Vec::new();
            let mut tensors = Vec::new();
            for request in linear_attention_input_projection_requests(
                0,
                layout.conv_dim,
                layout.total_value_width,
                layout.num_value_heads,
            )
            .unwrap()
            .requests()
            {
                append_tensor(
                    &mut bytes,
                    &mut tensors,
                    request.tensor_name.to_string(),
                    dtype,
                    vec![request.output_width, hidden],
                );
            }
            append_tensor(
                &mut bytes,
                &mut tensors,
                linear_attention_tensor_name(0, "conv1d"),
                dtype,
                vec![layout.conv_dim, layout.conv_kernel_size],
            );
            append_tensor(
                &mut bytes,
                &mut tensors,
                linear_attention_scalar_tensor_name(0, "A_log"),
                "F32",
                vec![layout.num_value_heads],
            );
            append_tensor(
                &mut bytes,
                &mut tensors,
                linear_attention_scalar_tensor_name(0, "dt_bias"),
                dtype,
                vec![layout.num_value_heads],
            );
            if !omit_norm {
                append_tensor(
                    &mut bytes,
                    &mut tensors,
                    linear_attention_tensor_name(0, "norm"),
                    dtype,
                    vec![layout.value_dim],
                );
            }
            append_tensor(
                &mut bytes,
                &mut tensors,
                linear_attention_tensor_name(0, "out_proj"),
                dtype,
                vec![hidden, layout.total_value_width],
            );
            append_tensor(
                &mut bytes,
                &mut tensors,
                router_tensor_name(0),
                dtype,
                vec![experts, hidden],
            );

            let temp = tempfile::tempdir().unwrap();
            let dense_path = temp.path().join("non-expert.bin");
            let manifest_path = temp.path().join("manifest.json");
            fs::write(&dense_path, bytes).unwrap();
            fs::write(
                &manifest_path,
                serde_json::to_vec(&FlashMoeManifest {
                    model: "fixture".to_string(),
                    cache_version: "test".to_string(),
                    dense_shards: vec!["fixture.safetensors".to_string()],
                    expert_tensors: Vec::new(),
                    dense_tensors: tensors,
                })
                .unwrap(),
            )
            .unwrap();
            let store = DenseStore::open(dense_path, manifest_path).unwrap();
            (temp, store)
        }

        let layout = LinearAttentionLayout {
            num_value_heads: 2,
            num_key_heads: 1,
            key_dim: 2,
            value_dim: 2,
            total_key_width: 2,
            total_value_width: 4,
            conv_dim: 8,
            conv_kernel_size: 2,
        };
        for (dtype, expected_static_dtype) in [
            ("BF16", ResidentStaticDtype::Bf16),
            ("F16", ResidentStaticDtype::F16),
            ("F32", ResidentStaticDtype::F32),
        ] {
            let (_temp, store) = fixture(dtype, false);
            let table = store
                .resolve_linear_attention_weight_table(&[Some(layout)], 4, 3)
                .unwrap();
            let bindings = table.layer(0).unwrap();
            assert_eq!(bindings.layer, 0);
            assert_eq!(bindings.input_projections.len(), 4);
            assert_eq!(
                bindings.static_tensors.conv_weight.dtype,
                expected_static_dtype
            );
            assert_eq!(bindings.static_tensors.dt_bias.dtype, expected_static_dtype);
            assert_eq!(
                bindings.static_tensors.norm_weight.dtype,
                expected_static_dtype
            );
            assert_eq!(
                bindings.static_tensors.a_log.dtype,
                ResidentStaticDtype::F32
            );
            assert_eq!(bindings.out_proj.rows(), 4);
            assert_eq!(bindings.router.rows(), 3);
        }

        let (_temp, store) = fixture("BF16", true);
        let error = store
            .resolve_linear_attention_weight_table(&[Some(layout)], 4, 3)
            .unwrap_err();
        assert!(
            error.to_string().contains("static-weight path"),
            "{error:#}"
        );
        assert!(error.to_string().contains("linear_attn.norm"), "{error:#}");
    }

    #[test]
    fn shared_expert_weight_table_resolves_all_resident_dense_layouts_at_load() {
        fn append_tensor(
            bytes: &mut Vec<u8>,
            tensors: &mut Vec<DenseTensorRef>,
            name: String,
            dtype: &str,
            shape: Vec<usize>,
        ) {
            while !(bytes.len() as u64).is_multiple_of(TENSOR_ALIGNMENT) {
                bytes.push(0);
            }
            let runtime_offset = bytes.len() as u64;
            let byte_len = shape.iter().product::<usize>() * dense_dtype_size(dtype).unwrap();
            bytes.resize(bytes.len() + byte_len, 0);
            tensors.push(DenseTensorRef {
                tensor: name,
                shard: "fixture.safetensors".to_string(),
                dtype: dtype.to_string(),
                shape,
                source_offsets: [0, byte_len as u64],
                runtime_offset,
                byte_len: byte_len as u64,
                quantization: TensorQuantization::None,
                q4_sources: None,
            });
        }

        fn fixture(dtype: &str, omit_down_layer: Option<usize>) -> (tempfile::TempDir, DenseStore) {
            let width = 4;
            let shared_experts = 2;
            let intermediate = 3;
            let total_intermediate = shared_experts * intermediate;
            let mut bytes = Vec::new();
            let mut tensors = Vec::new();
            for layer in 0..2 {
                for projection in ["gate_proj", "up_proj"] {
                    append_tensor(
                        &mut bytes,
                        &mut tensors,
                        shared_expert_tensor_name(layer, projection),
                        dtype,
                        vec![total_intermediate, width],
                    );
                }
                if omit_down_layer != Some(layer) {
                    append_tensor(
                        &mut bytes,
                        &mut tensors,
                        shared_expert_tensor_name(layer, "down_proj"),
                        dtype,
                        vec![width, total_intermediate],
                    );
                }
                append_tensor(
                    &mut bytes,
                    &mut tensors,
                    shared_expert_gate_tensor_name(layer),
                    dtype,
                    vec![shared_experts, width],
                );
            }

            let temp = tempfile::tempdir().unwrap();
            let dense_path = temp.path().join("non-expert.bin");
            let manifest_path = temp.path().join("manifest.json");
            fs::write(&dense_path, bytes).unwrap();
            fs::write(
                &manifest_path,
                serde_json::to_vec(&FlashMoeManifest {
                    model: "fixture".to_string(),
                    cache_version: "test".to_string(),
                    dense_shards: vec!["fixture.safetensors".to_string()],
                    expert_tensors: Vec::new(),
                    dense_tensors: tensors,
                })
                .unwrap(),
            )
            .unwrap();
            let store = DenseStore::open(dense_path, manifest_path).unwrap();
            (temp, store)
        }

        for dtype in ["BF16", "F16", "F32"] {
            let (_temp, store) = fixture(dtype, None);
            let table = store
                .resolve_shared_expert_weight_table(2, 4, 2, 3)
                .unwrap();
            for layer in 0..2 {
                let SharedExpertLayerWeights::Resident(shared) = table.layer(layer).unwrap() else {
                    panic!("configured shared experts must resolve resident bindings");
                };
                for projection in [&shared.gate, &shared.up, &shared.down] {
                    let ResidentMmapMatvecProjection::Dense(projection) = projection else {
                        panic!("{dtype} fixture resolved a Q4 projection");
                    };
                    assert_eq!(projection.dtype, dtype);
                }
                let ResidentMmapMatvecProjection::Dense(router) = shared.router.as_ref().unwrap()
                else {
                    panic!("{dtype} fixture resolved a Q4 router projection");
                };
                assert_eq!(router.dtype, dtype);
                assert_eq!(
                    shared.validated_shape().unwrap(),
                    SharedExpertPhaseShape::new(4, 2, 3).unwrap()
                );
            }

            let disabled = store
                .resolve_shared_expert_weight_table(2, 4, 0, 0)
                .unwrap();
            assert!(matches!(
                disabled.layer(0).unwrap(),
                SharedExpertLayerWeights::None
            ));
        }

        let (_temp, store) = fixture("BF16", Some(1));
        let error = store
            .resolve_shared_expert_weight_table(2, 4, 2, 3)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("missing resident shared down projection"),
            "{error:#}"
        );
        assert!(
            error
                .to_string()
                .contains("model.layers.1.mlp.shared_expert.down_proj.weight"),
            "{error:#}"
        );
    }

    #[test]
    fn dense_q4_projection_descriptor_carries_one_binding_shape() {
        let projection = DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            packed_byte_offset: 128,
            scales_byte_offset: 256,
            biases_byte_offset: 512,
            rows: 16,
            cols: 32,
            output_width: 16,
            row_packed_bytes: 16,
            groups_per_row: 2,
            group_size: 16,
            scale_bias_dtype: "BF16".to_string(),
        };

        assert_eq!(projection.row_packed_bytes, projection.cols.div_ceil(2));
        assert_eq!(projection.groups_per_row, 2);
        assert_eq!(projection.output_width, projection.rows);
    }

    #[test]
    fn dense_q4_projection_key_names_cached_binding_shape() {
        let key = DenseQ4ProjectionKey::new("model.layers.0.mlp.gate_proj.weight", 16, 32);

        assert_eq!(key.name, "model.layers.0.mlp.gate_proj.weight");
        assert_eq!(key.output_width, 16);
        assert_eq!(key.input_len, 32);
    }

    #[test]
    fn dense_q4_projection_builder_uses_lookup_callback() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            byte_offset: 128,
            byte_len: 12,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "BF16".to_string(),
            },
        };
        let mut seen_name = None;

        let projection = build_dense_q4_mmap_projection(
            "model.layers.0.mlp.gate_proj.weight",
            2,
            4,
            256,
            |name| {
                seen_name = Some(name.to_string());
                (name == entry.name).then_some(&entry)
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(seen_name.unwrap(), entry.name);
        assert_eq!(projection.packed_byte_offset, 128);
        assert_eq!(projection.output_width, 2);
        assert_eq!(projection.cols, 4);

        let missing =
            build_dense_q4_mmap_projection("missing.weight", 2, 4, 256, |_| None).unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn shared_expert_dense_descriptor_groups_projection_weights() {
        let shared = SharedExpertPhaseWeights::new(
            Arc::new(vec![1.0, 2.0]),
            Arc::new(vec![3.0, 4.0]),
            Arc::new(vec![5.0, 6.0]),
            Arc::new(vec![7.0]),
            1,
            2,
            1,
        )
        .unwrap();

        assert_eq!(shared.shared_experts, 1);
        assert_eq!(shared.intermediate, 2);
        assert_eq!(shared.width, 1);
        assert_eq!(shared.gate.as_slice(), &[1.0, 2.0]);
        assert_eq!(shared.router.as_slice(), &[7.0]);
        assert_eq!(
            shared.validated_shape().unwrap(),
            SharedExpertPhaseShape::new(1, 1, 2).unwrap()
        );
    }

    #[test]
    fn shared_expert_weight_builder_loads_named_dense_tensors() {
        let mut tensors = BTreeMap::<String, Arc<Vec<f32>>>::new();
        tensors.insert(
            shared_expert_tensor_name(3, "gate_proj"),
            Arc::new(vec![1.0, 2.0, 3.0, 4.0]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "up_proj"),
            Arc::new(vec![5.0, 6.0, 7.0, 8.0]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "down_proj"),
            Arc::new(vec![9.0, 10.0, 11.0, 12.0]),
        );
        tensors.insert(
            shared_expert_gate_tensor_name(3),
            Arc::new(vec![13.0, 14.0]),
        );

        let shared =
            build_shared_expert_phase_weights(3, 2, 1, 2, |name| Ok(tensors.get(name).cloned()))
                .unwrap()
                .unwrap();

        assert_eq!(shared.width, 2);
        assert_eq!(shared.shared_experts, 1);
        assert_eq!(shared.intermediate, 2);
        assert_eq!(shared.gate.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shared.router.as_slice(), &[13.0, 14.0]);
    }

    #[test]
    fn shared_expert_weight_builder_skips_disabled_shared_experts() {
        let none = build_shared_expert_phase_weights(3, 2, 0, 2, |_| {
            panic!("disabled shared experts must not request tensors")
        })
        .unwrap();
        assert!(none.is_none());

        let none = build_shared_expert_phase_weights(3, 2, 1, 0, |_| {
            panic!("zero intermediate shared experts must not request tensors")
        })
        .unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn shared_expert_phase_cache_reuses_weight_owned_dense_phase() {
        let cache = SharedExpertPhaseCache::default();
        let mut tensors = BTreeMap::<String, Arc<Vec<f32>>>::new();
        tensors.insert(
            shared_expert_tensor_name(3, "gate_proj"),
            Arc::new(vec![1.0, 2.0, 3.0, 4.0]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "up_proj"),
            Arc::new(vec![5.0, 6.0, 7.0, 8.0]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "down_proj"),
            Arc::new(vec![9.0, 10.0, 11.0, 12.0]),
        );
        tensors.insert(
            shared_expert_gate_tensor_name(3),
            Arc::new(vec![13.0, 14.0]),
        );
        let mut lookup_count = 0usize;

        let first = cache
            .dense(3, 2, 1, 2, |name| {
                lookup_count += 1;
                Ok(tensors.get(name).cloned())
            })
            .unwrap()
            .unwrap();
        let second = cache
            .dense(3, 2, 1, 2, |_| {
                panic!("cached shared expert phase must not reload tensors")
            })
            .unwrap()
            .unwrap();

        assert_eq!(lookup_count, 4);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn shared_expert_phase_cache_skips_disabled_shared_experts() {
        let cache = SharedExpertPhaseCache::default();

        let none = cache
            .dense(3, 2, 0, 2, |_| {
                panic!("disabled shared experts must not request tensors")
            })
            .unwrap();

        assert!(none.is_none());
    }

    #[test]
    fn shared_expert_phase_cache_rejects_width_mismatch_without_reload() {
        let cache = SharedExpertPhaseCache::default();
        let mut tensors = BTreeMap::<String, Arc<Vec<f32>>>::new();
        tensors.insert(
            shared_expert_tensor_name(3, "gate_proj"),
            Arc::new(vec![1.0, 2.0, 3.0, 4.0]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "up_proj"),
            Arc::new(vec![5.0, 6.0, 7.0, 8.0]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "down_proj"),
            Arc::new(vec![9.0, 10.0, 11.0, 12.0]),
        );
        tensors.insert(
            shared_expert_gate_tensor_name(3),
            Arc::new(vec![13.0, 14.0]),
        );

        cache
            .dense(3, 2, 1, 2, |name| Ok(tensors.get(name).cloned()))
            .unwrap()
            .unwrap();
        let err = cache
            .dense(3, 4, 1, 2, |_| {
                panic!("mismatched cached shared expert phase must fail before reload")
            })
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("cached shared expert tensors for layer 3 have width 2, requested 4")
        );
    }

    #[test]
    fn shared_expert_weight_builder_rejects_missing_and_mismatched_tensors() {
        let missing = build_shared_expert_phase_weights(3, 2, 1, 2, |_| Ok(None)).unwrap_err();
        assert!(missing.to_string().contains(
            "missing configured shared expert tensor model.layers.3.mlp.shared_expert.gate_proj.weight"
        ));

        let mut tensors = BTreeMap::<String, Arc<Vec<f32>>>::new();
        tensors.insert(
            shared_expert_tensor_name(3, "gate_proj"),
            Arc::new(vec![1.0]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "up_proj"),
            Arc::new(vec![2.0; 4]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "down_proj"),
            Arc::new(vec![3.0; 4]),
        );
        tensors.insert(shared_expert_gate_tensor_name(3), Arc::new(vec![4.0; 2]));

        let mismatch =
            build_shared_expert_phase_weights(3, 2, 1, 2, |name| Ok(tensors.get(name).cloned()))
                .unwrap_err();
        assert!(mismatch.to_string().contains("shape is invalid"));
    }

    #[test]
    fn shared_expert_resident_descriptor_groups_projection_bindings() {
        let gate = DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.0.mlp.shared_expert.gate_proj.weight".to_string(),
            packed_byte_offset: 128,
            scales_byte_offset: 256,
            biases_byte_offset: 512,
            rows: 16,
            cols: 32,
            output_width: 16,
            row_packed_bytes: 16,
            groups_per_row: 2,
            group_size: 16,
            scale_bias_dtype: "BF16".to_string(),
        };
        let up = DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.0.mlp.shared_expert.up_proj.weight".to_string(),
            ..gate.clone()
        };
        let down = DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.0.mlp.shared_expert.down_proj.weight".to_string(),
            rows: 32,
            cols: 16,
            output_width: 32,
            row_packed_bytes: 8,
            groups_per_row: 1,
            ..gate.clone()
        };
        let router = DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.0.mlp.shared_expert_gate.weight".to_string(),
            rows: 1,
            output_width: 1,
            ..gate.clone()
        };
        let shared = SharedExpertPhaseResidentProjections {
            gate: gate.into(),
            up: up.into(),
            down: down.into(),
            router: Some(router.into()),
            shared_experts: 1,
            intermediate: 16,
            width: 32,
        };

        assert_eq!(shared.gate.q4().unwrap().packed_byte_offset, 128);
        assert_eq!(shared.down.output_width(), 32);
        assert_eq!(shared.router.as_ref().unwrap().cols(), 32);
        assert_eq!(shared.shared_experts, 1);
        assert_eq!(shared.intermediate, 16);
        assert_eq!(shared.width, 32);
        assert_eq!(
            shared.validated_shape().unwrap(),
            SharedExpertPhaseShape::new(32, 1, 16).unwrap()
        );
    }

    #[test]
    fn shared_expert_resident_builder_resolves_named_projection_bindings() {
        let shared = build_shared_expert_resident_phase_projections(
            4,
            32,
            2,
            16,
            |name, output_width, input_len| {
                Ok(Some(DenseQ4MmapMatvecProjection {
                    tensor_name: name.to_string(),
                    packed_byte_offset: 128,
                    scales_byte_offset: 256,
                    biases_byte_offset: 512,
                    rows: output_width,
                    cols: input_len,
                    output_width,
                    row_packed_bytes: input_len.div_ceil(2),
                    groups_per_row: input_len.div_ceil(16),
                    group_size: 16,
                    scale_bias_dtype: "BF16".to_string(),
                }))
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            shared.gate.tensor_name(),
            "model.layers.4.mlp.shared_expert.gate_proj.weight"
        );
        assert_eq!(shared.gate.output_width(), 32);
        assert_eq!(shared.gate.cols(), 32);
        assert_eq!(shared.down.output_width(), 32);
        assert_eq!(shared.down.cols(), 32);
        assert_eq!(shared.router.as_ref().unwrap().output_width(), 2);
        assert_eq!(shared.router.as_ref().unwrap().cols(), 32);
        assert_eq!(
            shared.validated_shape().unwrap(),
            SharedExpertPhaseShape::new(32, 2, 16).unwrap()
        );
    }

    #[test]
    fn shared_expert_resident_builder_skips_disabled_or_partial_bindings() {
        let disabled = build_shared_expert_resident_phase_projections(
            4,
            32,
            0,
            16,
            |_, _, _| -> Result<Option<DenseQ4MmapMatvecProjection>> {
                panic!("disabled shared experts must not request projections")
            },
        )
        .unwrap();
        assert!(disabled.is_none());

        let partial = build_shared_expert_resident_phase_projections(
            4,
            32,
            2,
            16,
            |name, output_width, input_len| {
                if name.ends_with("up_proj.weight") {
                    return Ok(None);
                }
                Ok(Some(DenseQ4MmapMatvecProjection {
                    tensor_name: name.to_string(),
                    packed_byte_offset: 128,
                    scales_byte_offset: 256,
                    biases_byte_offset: 512,
                    rows: output_width,
                    cols: input_len,
                    output_width,
                    row_packed_bytes: input_len.div_ceil(2),
                    groups_per_row: input_len.div_ceil(16),
                    group_size: 16,
                    scale_bias_dtype: "BF16".to_string(),
                }))
            },
        )
        .unwrap();
        assert!(partial.is_none());
    }

    #[test]
    fn required_shared_expert_resident_builder_errors_on_missing_configured_binding() {
        let disabled = build_required_shared_expert_resident_phase_projections(
            4,
            32,
            0,
            16,
            |_, _, _| -> Result<Option<DenseQ4MmapMatvecProjection>> {
                panic!("disabled shared experts must not request projections")
            },
        )
        .unwrap();
        assert!(disabled.is_none());

        let invalid = build_required_shared_expert_resident_phase_projections(
            4,
            32,
            2,
            0,
            |_, _, _| -> Result<Option<DenseQ4MmapMatvecProjection>> {
                panic!("invalid shared-expert shape must fail before requesting projections")
            },
        )
        .unwrap_err();
        assert!(
            invalid.to_string().contains("requires non-zero width"),
            "{invalid:#}"
        );

        let missing = build_required_shared_expert_resident_phase_projections(
            4,
            32,
            2,
            16,
            |name, output_width, input_len| {
                if name.ends_with("down_proj.weight") {
                    return Ok(None);
                }
                Ok(Some(DenseQ4MmapMatvecProjection {
                    tensor_name: name.to_string(),
                    packed_byte_offset: 128,
                    scales_byte_offset: 256,
                    biases_byte_offset: 512,
                    rows: output_width,
                    cols: input_len,
                    output_width,
                    row_packed_bytes: input_len.div_ceil(2),
                    groups_per_row: input_len.div_ceil(16),
                    group_size: 16,
                    scale_bias_dtype: "BF16".to_string(),
                }))
            },
        )
        .unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("missing resident shared down projection"),
            "{missing:#}"
        );
    }

    #[test]
    fn shared_expert_resident_builder_rejects_mismatched_projection_shape() {
        let err = build_shared_expert_resident_phase_projections(
            4,
            32,
            2,
            16,
            |name, output_width, input_len| {
                Ok(Some(DenseQ4MmapMatvecProjection {
                    tensor_name: name.to_string(),
                    packed_byte_offset: 128,
                    scales_byte_offset: 256,
                    biases_byte_offset: 512,
                    rows: output_width,
                    cols: input_len + usize::from(name.ends_with("gate_proj.weight")),
                    output_width,
                    row_packed_bytes: input_len.div_ceil(2),
                    groups_per_row: input_len.div_ceil(16),
                    group_size: 16,
                    scale_bias_dtype: "BF16".to_string(),
                }))
            },
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("resident shared-expert shape is invalid")
        );
    }

    #[test]
    fn shared_expert_descriptors_reject_mismatched_graph_shape() {
        let shared = SharedExpertPhaseWeights {
            gate: Arc::new(vec![1.0, 2.0]),
            up: Arc::new(vec![3.0, 4.0]),
            down: Arc::new(vec![5.0, 6.0]),
            router: Arc::new(vec![7.0]),
            shared_experts: 1,
            intermediate: 2,
            width: 2,
        };

        let err = shared.validated_shape().unwrap_err();
        assert!(err.to_string().contains("shape is invalid"));
    }

    #[test]
    fn scheduled_next_norm_weights_declare_cpu_visible_width() {
        let values = [1.0, 0.5, 0.25, 0.125];
        let weights = ScheduledNextNormWeights::cpu_visible(
            "model.layers.1.input_layernorm.weight",
            &values,
            3,
        )
        .unwrap();

        assert!(weights.is_cpu_visible());
        assert_eq!(weights.width(), Some(3));
        assert_eq!(weights.values().unwrap(), &[1.0, 0.5, 0.25]);
        assert!(ScheduledNextNormWeights::none().is_none());

        let empty_name_err = ScheduledNextNormWeights::cpu_visible("", &values, 3).unwrap_err();
        assert!(empty_name_err.to_string().contains("require a tensor name"));

        let short_err = ScheduledNextNormWeights::cpu_visible(
            "model.layers.1.input_layernorm.weight",
            &values[..2],
            3,
        )
        .unwrap_err();
        assert!(short_err.to_string().contains("smaller than width 3"));
    }

    #[test]
    fn prepared_next_norm_weights_resolve_declared_cmd3_descriptor() {
        let prepared = PreparedScheduledNextNormWeights::cpu_visible(
            "model.layers.1.input_layernorm.weight".to_string(),
            vec![1.0, 0.5, 0.25, 0.125],
            3,
        )
        .unwrap();
        let scheduled = prepared.scheduled().unwrap();

        assert!(scheduled.is_cpu_visible());
        assert_eq!(scheduled.width(), Some(3));
        assert_eq!(scheduled.values().unwrap(), &[1.0, 0.5, 0.25]);
    }

    #[test]
    fn prepare_next_norm_weights_declares_only_non_terminal_cmd3_layers() {
        let prepared = prepare_scheduled_next_norm_weights(0, 2, 4, true, |name, width| {
            assert_eq!(name, "model.layers.1.input_layernorm.weight");
            assert_eq!(width, 4);
            Ok(Some(vec![1.0, 1.1, 1.2, 1.3]))
        })
        .unwrap();
        assert!(prepared.scheduled().unwrap().is_cpu_visible());

        let terminal = prepare_scheduled_next_norm_weights(1, 2, 4, true, |_, _| {
            panic!("terminal layer must not request next-layer norm weights")
        })
        .unwrap();
        assert!(terminal.scheduled().unwrap().is_none());

        let disabled = prepare_scheduled_next_norm_weights(0, 2, 4, false, |_, _| {
            panic!("disabled next-layer norm must not request weights")
        })
        .unwrap();
        assert!(disabled.scheduled().unwrap().is_none());
    }

    #[test]
    fn prepare_next_norm_weights_reports_missing_scheduled_cmd3_weight() {
        let err = prepare_scheduled_next_norm_weights(2, 4, 8, true, |_, _| Ok(None)).unwrap_err();

        assert!(err.to_string().contains(
            "missing next-layer norm weight model.layers.3.input_layernorm.weight for layer 2"
        ));
    }

    #[test]
    fn qwen_moe_weight_tensor_names_are_canonical_hf_paths() {
        assert_eq!(
            layer_norm_tensor_name(7, "post_attention_layernorm"),
            "model.layers.7.post_attention_layernorm.weight"
        );
        assert_eq!(
            attention_tensor_name(7, "q_proj"),
            "model.layers.7.self_attn.q_proj.weight"
        );
        assert_eq!(
            attention_tensor_name(7, "o_proj"),
            "model.layers.7.self_attn.o_proj.weight"
        );
        assert_eq!(router_tensor_name(7), "model.layers.7.mlp.gate.weight");
        assert_eq!(
            shared_expert_tensor_name(7, "gate_proj"),
            "model.layers.7.mlp.shared_expert.gate_proj.weight"
        );
        assert_eq!(
            shared_expert_gate_tensor_name(7),
            "model.layers.7.mlp.shared_expert_gate.weight"
        );
    }

    #[test]
    fn linear_attention_weight_tensor_names_are_canonical_hf_paths() {
        assert_eq!(
            linear_attention_tensor_name(7, "in_proj_qkv"),
            "model.layers.7.linear_attn.in_proj_qkv.weight"
        );
        assert_eq!(
            linear_attention_tensor_name(7, "out_proj"),
            "model.layers.7.linear_attn.out_proj.weight"
        );
        assert_eq!(
            linear_attention_scalar_tensor_name(7, "A_log"),
            "model.layers.7.linear_attn.A_log"
        );
    }

    #[test]
    fn dense_projection_request_requires_named_nonzero_output() {
        let request =
            DenseProjectionRequest::new("model.layers.7.linear_attn.in_proj_qkv.weight", 128)
                .unwrap();

        assert_eq!(
            request.tensor_name,
            "model.layers.7.linear_attn.in_proj_qkv.weight"
        );
        assert_eq!(request.output_width, 128);

        let missing_name = DenseProjectionRequest::new("", 128).unwrap_err();
        assert!(missing_name.to_string().contains("requires a tensor name"));

        let zero_width =
            DenseProjectionRequest::new("model.layers.7.linear_attn.in_proj_qkv.weight", 0)
                .unwrap_err();
        assert!(zero_width.to_string().contains("non-zero output width"));
    }

    #[test]
    fn full_attention_projection_requests_use_canonical_self_attention_names() {
        let requests = full_attention_input_projection_requests(3, 24, 8).unwrap();
        let specs = requests.requests();

        assert_eq!(
            specs
                .iter()
                .map(|spec| (spec.tensor_name, spec.output_width))
                .collect::<Vec<_>>(),
            vec![
                ("model.layers.3.self_attn.q_proj.weight", 24),
                ("model.layers.3.self_attn.k_proj.weight", 8),
                ("model.layers.3.self_attn.v_proj.weight", 8),
            ]
        );
        assert_eq!(
            requests.tensor_name(0),
            "model.layers.3.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn linear_attention_projection_requests_use_canonical_gated_delta_names() {
        let requests = linear_attention_input_projection_requests(5, 16, 32, 4).unwrap();
        let specs = requests.requests();

        assert_eq!(
            specs
                .iter()
                .map(|spec| (spec.tensor_name, spec.output_width))
                .collect::<Vec<_>>(),
            vec![
                ("model.layers.5.linear_attn.in_proj_qkv.weight", 16),
                ("model.layers.5.linear_attn.in_proj_z.weight", 32),
                ("model.layers.5.linear_attn.in_proj_b.weight", 4),
                ("model.layers.5.linear_attn.in_proj_a.weight", 4),
            ]
        );
        assert_eq!(
            requests.tensor_name(3),
            "model.layers.5.linear_attn.in_proj_a.weight"
        );
    }

    #[test]
    fn projection_request_groups_reject_zero_width_without_fallback() {
        let full = full_attention_input_projection_requests(3, 0, 8).unwrap_err();
        assert!(full.to_string().contains("non-zero output width"));

        let linear = linear_attention_input_projection_requests(5, 16, 32, 0).unwrap_err();
        assert!(linear.to_string().contains("non-zero output width"));
    }

    #[test]
    fn qwen_norm_offset_policy_matches_declared_semantics_and_reference_names() {
        for name in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.3.self_attn.q_norm.weight",
            "model.layers.3.self_attn.k_norm.weight",
        ] {
            assert!(
                qwen_norm_uses_offset(QwenNormWeightSemantics::Offset, name),
                "{name} should use Qwen3Next 1+weight RMSNorm semantics"
            );
        }

        for name in [
            "model.layers.0.linear_attn.norm.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
        ] {
            assert!(
                !qwen_norm_uses_offset(QwenNormWeightSemantics::Offset, name),
                "{name} is not a plain Qwen3NextRMSNorm weight"
            );
        }

        assert!(!qwen_norm_uses_offset(
            QwenNormWeightSemantics::Multiplicative,
            "model.norm.weight"
        ));
    }

    #[test]
    fn qwen_norm_semantics_are_resolved_without_value_probing() {
        let mut offset = vec![0.6679, 0.7187, 0.7265, 0.7031];
        apply_qwen_norm_weight_semantics(
            QwenNormWeightSemantics::Offset,
            "model.layers.0.input_layernorm.weight",
            &mut offset,
        );
        for (actual, expected) in offset.iter().zip([1.6679, 1.7187, 1.7265, 1.7031]) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let mut disabled = vec![0.6679, 0.7187, 0.7265, 0.7031];
        apply_qwen_norm_weight_semantics(
            QwenNormWeightSemantics::Multiplicative,
            "model.norm.weight",
            &mut disabled,
        );
        assert_eq!(disabled, vec![0.6679, 0.7187, 0.7265, 0.7031]);
    }

    #[test]
    fn dense_q4_layout_accounts_for_scale_bias_dtype() {
        let layout = dense_q4_layout_with_scale_bias_dtype(&[2, 4], 16, "BF16").unwrap();

        assert_eq!(layout.rows, 2);
        assert_eq!(layout.cols, 4);
        assert_eq!(layout.row_packed_bytes, 2);
        assert_eq!(layout.groups_per_row, 1);
        assert_eq!(layout.packed_bytes, 4);
        assert_eq!(layout.scales_bytes, 4);
        assert_eq!(layout.scale_bias_bytes, 2);
        assert_eq!(layout.total_bytes, 12);
    }

    #[test]
    fn dense_q4_projection_descriptor_resolves_offsets_from_entry() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            byte_offset: 128,
            byte_len: 12,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "BF16".to_string(),
            },
        };

        let projection = DenseQ4MmapMatvecProjection::from_entry(
            "model.layers.0.mlp.gate_proj.weight",
            &entry,
            256,
            2,
            4,
        )
        .unwrap()
        .unwrap();

        assert_eq!(projection.packed_byte_offset, 128);
        assert_eq!(projection.scales_byte_offset, 132);
        assert_eq!(projection.biases_byte_offset, 136);
        assert_eq!(projection.rows, 2);
        assert_eq!(projection.cols, 4);
        assert_eq!(projection.scale_bias_dtype, "BF16");
    }

    #[test]
    fn dense_q4_projection_descriptor_rejects_missing_capacity() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            byte_offset: 128,
            byte_len: 12,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "BF16".to_string(),
            },
        };

        let projection = DenseQ4MmapMatvecProjection::from_entry(
            "model.layers.0.mlp.gate_proj.weight",
            &entry,
            139,
            2,
            4,
        )
        .unwrap();

        assert_eq!(projection, None);
    }
}
