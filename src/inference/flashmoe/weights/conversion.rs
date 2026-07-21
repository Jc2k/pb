use super::*;

pub(crate) const TENSOR_ALIGNMENT: u64 = 4096;
#[cfg(test)]
pub(in crate::inference::flashmoe) const DENSE_PROJECTION_TILE_BYTES: usize = 64 * 1024 * 1024;
pub(in crate::inference::flashmoe) const DENSE_DECODED_TILE_CACHE_BYTES: usize = 512 * 1024 * 1024;
pub(in crate::inference::flashmoe) const DENSE_Q4_FULL_DECODE_MAX_BYTES: usize = 256 * 1024 * 1024;

pub(in crate::inference::flashmoe) const DENSE_Q4_MLX_FORMAT: &str = "dense-q4-affine-mlx-v1";
pub(in crate::inference::flashmoe) const DENSE_Q4_COLIBRI_FORMAT: &str =
    "dense-q4-affine-colibri-import-v1";
pub(in crate::inference::flashmoe) const DENSE_Q4_MXFP4_FORMAT: &str =
    "dense-q4-affine-mxfp4-import-v1";

pub(in crate::inference::flashmoe) fn skip_flashmoe_runtime_tensor(canonical_tensor: &str) -> bool {
    canonical_tensor.starts_with("mtp.")
}

pub(in crate::inference::flashmoe) fn is_q4_aux_tensor_name(canonical_tensor: &str) -> bool {
    canonical_tensor.ends_with(".scales")
        || canonical_tensor.ends_with(".biases")
        || canonical_tensor.ends_with(".weight.qs")
}

pub(in crate::inference::flashmoe) fn q4_weight_name_for_aux(tensor: &str) -> String {
    tensor
        .strip_suffix(".scales")
        .or_else(|| tensor.strip_suffix(".biases"))
        .or_else(|| tensor.strip_suffix(".qs"))
        .map(|base| format!("{base}.weight"))
        .map(|name| name.replace(".weight.weight", ".weight"))
        .unwrap_or_else(|| tensor.to_string())
}

pub(in crate::inference::flashmoe) fn q4_aux_tensor_name(weight: &str, suffix: &str) -> String {
    weight
        .strip_suffix(".weight")
        .map(|base| format!("{base}.{suffix}"))
        .unwrap_or_else(|| format!("{weight}.{suffix}"))
}

pub(in crate::inference::flashmoe) fn logical_shape_for_mlx_q4(
    shape: &[usize],
) -> Result<Vec<usize>> {
    logical_shape_for_mlx_packed(shape, 8)
}

pub(in crate::inference::flashmoe) fn logical_shape_for_mlx_source(
    shape: &[usize],
    source: &DenseQ4SourceRefs,
) -> Result<Vec<usize>> {
    logical_shape_for_mlx_packed(
        shape,
        if source.source_format == DenseQ4SourceFormat::MlxAffine8 {
            4
        } else {
            8
        },
    )
}

fn logical_shape_for_mlx_packed(shape: &[usize], values_per_u32: usize) -> Result<Vec<usize>> {
    let Some((last, prefix)) = shape.split_last() else {
        bail!("native dense q4 tensor has empty shape");
    };
    let cols = last
        .checked_mul(values_per_u32)
        .context("native MLX logical column count overflow")?;
    let mut logical = prefix.to_vec();
    logical.push(cols);
    Ok(logical)
}

pub(in crate::inference::flashmoe) fn dense_native_q4_sources(
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
            source_row_order: None,
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
            source_row_order: None,
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
    let weight_bytes = weight_offsets[1].saturating_sub(weight_offsets[0]);
    let scales_bytes = scales_offsets[1].saturating_sub(scales_offsets[0]);
    let biases_bytes = biases_info.data_offsets[1].saturating_sub(biases_info.data_offsets[0]);
    let source_rows = weight_shape
        .iter()
        .take(weight_shape.len().saturating_sub(1))
        .try_fold(1usize, |rows, dimension| {
            rows.checked_mul(*dimension)
                .context("native MLX affine row count overflow")
        })?;
    let scale_values = usize::try_from(scales_bytes)
        .context("native MLX affine scale byte count exceeds usize")?
        .checked_div(2)
        .context("native MLX affine scale byte count division failed")?;
    if source_rows == 0 || !scale_values.is_multiple_of(source_rows) {
        bail!("native MLX affine tensor {tensor} has {scale_values} scales for {source_rows} rows");
    }
    let q4_shape = logical_shape_for_mlx_packed(&weight_shape, 8)?;
    let q8_shape = logical_shape_for_mlx_packed(&weight_shape, 4)?;
    let q4_cols = *q4_shape
        .last()
        .context("native MLX affine Q4 tensor has empty logical shape")?;
    let q8_cols = *q8_shape
        .last()
        .context("native MLX affine int8 tensor has empty logical shape")?;
    let q4_scale_values = source_rows
        .checked_mul(q4_cols.div_ceil(GROUP_SIZE))
        .context("native MLX affine Q4 scale count overflow")?;
    let q8_scale_values = source_rows
        .checked_mul(q8_cols.div_ceil(GROUP_SIZE))
        .context("native MLX affine int8 scale count overflow")?;
    let weight_bytes = usize::try_from(weight_bytes)
        .context("native MLX affine weight byte count exceeds usize")?;
    let q4_weight_bytes = q4_shape.iter().try_fold(1usize, |values, dimension| {
        values
            .checked_mul(*dimension)
            .context("native MLX affine Q4 logical value count overflow")
    })? / 2;
    let q8_weight_bytes = q8_shape.iter().try_fold(1usize, |values, dimension| {
        values
            .checked_mul(*dimension)
            .context("native MLX affine int8 logical value count overflow")
    })?;
    let source_format = if scale_values == q4_scale_values && weight_bytes == q4_weight_bytes {
        DenseQ4SourceFormat::MlxAffine
    } else if scale_values == q8_scale_values && weight_bytes == q8_weight_bytes {
        DenseQ4SourceFormat::MlxAffine8
    } else {
        bail!(
            "native MLX affine tensor {tensor} has {weight_bytes} weight bytes and {scale_values} scales; expected Q4 {q4_weight_bytes}/{q4_scale_values} or int8 {q8_weight_bytes}/{q8_scale_values}"
        );
    };
    let expected_scale_bytes = scale_values
        .checked_mul(2)
        .context("native MLX affine scale byte count overflow")?
        as u64;
    if scales_bytes != expected_scale_bytes || biases_bytes != expected_scale_bytes {
        bail!(
            "native MLX affine tensor {tensor} scale/bias bytes {scales_bytes}/{biases_bytes}, expected {expected_scale_bytes}"
        );
    }
    Ok(Some(DenseQ4SourceRefs {
        scales_shard: scales_shard.clone(),
        scales_offsets,
        biases_shard: biases_shard.clone(),
        biases_offsets: biases_info.data_offsets,
        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        source_format,
        source_group_size: None,
        source_row_order: None,
    }))
}

pub(in crate::inference::flashmoe) fn dense_tensor_quantization(
    canonical_tensor: &str,
    tensor_dtype: &str,
    native_q4: &Option<DenseQ4SourceRefs>,
) -> TensorQuantization {
    let _ = (canonical_tensor, tensor_dtype);
    if native_q4.as_ref().is_some_and(|source| {
        matches!(
            source.source_format,
            DenseQ4SourceFormat::ColibriInt8 | DenseQ4SourceFormat::MlxAffine8
        )
    }) {
        // Colibri keeps the large embedding and LM-head matrices at int8 by
        // default. Preserve that source precision in a runtime layout the
        // existing resident dense kernels can consume instead of requantizing
        // the values to q4.
        TensorQuantization::None
    } else if let Some(native_q4) = native_q4 {
        TensorQuantization::Q4 {
            group_size: GROUP_SIZE,
            format: match native_q4.source_format {
                DenseQ4SourceFormat::MlxAffine | DenseQ4SourceFormat::MlxAffine8 => {
                    DENSE_Q4_MLX_FORMAT
                }
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

pub(in crate::inference::flashmoe) fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

pub(in crate::inference::flashmoe) fn write_dense_tensor_store(
    snapshot_dir: &Path,
    destination: &Path,
    dense_tensors: &[DenseTensorRef],
    config: Option<&QwenModelConfig>,
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
                if config.is_some_and(QwenModelConfig::is_qwen3_next)
                    && tensor.tensor.ends_with(".linear_attn.A_log")
                    && tensor.dtype.eq_ignore_ascii_case("F32")
                {
                    write_bf16_as_f32_tensor(
                        &mut out,
                        &tensor.tensor,
                        raw,
                        tensor.byte_len as usize,
                    )?;
                } else if let Some(q4_sources) = &tensor.q4_sources
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
                } else if let Some(q4_sources) = &tensor.q4_sources
                    && q4_sources.source_format == DenseQ4SourceFormat::MlxAffine8
                {
                    let (scale_bytes, scale_shard) = shard_cache
                        .get(&q4_sources.scales_shard)
                        .expect("inserted above");
                    let scale_start = scale_shard.data_start + q4_sources.scales_offsets[0];
                    let scale_end = scale_shard.data_start + q4_sources.scales_offsets[1];
                    let (bias_bytes, bias_shard) = shard_cache
                        .get(&q4_sources.biases_shard)
                        .expect("inserted above");
                    let bias_start = bias_shard.data_start + q4_sources.biases_offsets[0];
                    let bias_end = bias_shard.data_start + q4_sources.biases_offsets[1];
                    write_mlx_affine8_bf16_tensor(
                        &mut out,
                        &tensor.tensor,
                        raw,
                        &scale_bytes[scale_start as usize..scale_end as usize],
                        &bias_bytes[bias_start as usize..bias_end as usize],
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
                    if let Some(source_row_order) = q4_sources.source_row_order.as_deref() {
                        if q4_sources.source_format != DenseQ4SourceFormat::MlxAffine {
                            bail!(
                                "row-reordered dense projection {} requires MLX affine Q4 source, found {:?}",
                                tensor.tensor,
                                q4_sources.source_format
                            );
                        }
                        if source_row_order.len() != layout.rows
                            || tensor.byte_len as usize != layout.total_bytes
                        {
                            bail!(
                                "row-reordered dense projection {} has {} target rows and {} manifest bytes; expected {} rows and {} bytes",
                                tensor.tensor,
                                source_row_order.len(),
                                tensor.byte_len,
                                layout.rows,
                                layout.total_bytes
                            );
                        }
                        let (scale_bytes, scale_shard) = shard_cache
                            .get(&q4_sources.scales_shard)
                            .expect("inserted above");
                        let scale_start = scale_shard.data_start + q4_sources.scales_offsets[0];
                        let scale_end = scale_shard.data_start + q4_sources.scales_offsets[1];
                        let scales = &scale_bytes[scale_start as usize..scale_end as usize];
                        let (bias_bytes, bias_shard) = shard_cache
                            .get(&q4_sources.biases_shard)
                            .expect("inserted above");
                        let bias_start = bias_shard.data_start + q4_sources.biases_offsets[0];
                        let bias_end = bias_shard.data_start + q4_sources.biases_offsets[1];
                        let biases = &bias_bytes[bias_start as usize..bias_end as usize];
                        let scalar_bytes = expert_scale_bias_dtype_size(scale_bias_dtype)
                            .with_context(|| {
                                format!(
                                    "row-reordered dense projection {} has unsupported scale/bias dtype {}",
                                    tensor.tensor, scale_bias_dtype
                                )
                            })?;
                        let scale_row_bytes = layout
                            .groups_per_row
                            .checked_mul(scalar_bytes)
                            .context("row-reordered dense scale row byte count overflow")?;
                        write_rows_in_order(
                            &mut out,
                            raw,
                            layout.row_packed_bytes,
                            source_row_order,
                            &tensor.tensor,
                            "packed weights",
                        )?;
                        write_rows_in_order(
                            &mut out,
                            scales,
                            scale_row_bytes,
                            source_row_order,
                            &tensor.tensor,
                            "scales",
                        )?;
                        write_rows_in_order(
                            &mut out,
                            biases,
                            scale_row_bytes,
                            source_row_order,
                            &tensor.tensor,
                            "biases",
                        )?;
                        current = current.saturating_add(tensor.byte_len);
                        continue;
                    }
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
            TensorQuantization::Gguf { .. } => {
                out.write_all(raw).with_context(|| {
                    format!("failed to write native GGUF tensor {}", tensor.tensor)
                })?;
            }
        }
        current = current.saturating_add(tensor.byte_len);
    }
    Ok(())
}

pub(in crate::inference::flashmoe) fn write_rows_in_order(
    out: &mut impl Write,
    source: &[u8],
    row_bytes: usize,
    row_order: &[usize],
    tensor_name: &str,
    component: &str,
) -> Result<()> {
    if row_bytes == 0 || !source.len().is_multiple_of(row_bytes) {
        bail!(
            "row-reordered dense projection {tensor_name} has invalid {component} byte length {} for {row_bytes}-byte rows",
            source.len()
        );
    }
    let source_rows = source.len() / row_bytes;
    for &source_row in row_order {
        if source_row >= source_rows {
            bail!(
                "row-reordered dense projection {tensor_name} requests {component} row {source_row}, but source has {source_rows} rows"
            );
        }
        let start = source_row
            .checked_mul(row_bytes)
            .context("row-reordered dense source byte offset overflow")?;
        out.write_all(&source[start..start + row_bytes])
            .with_context(|| {
                format!("failed to write row-reordered {component} for {tensor_name}")
            })?;
    }
    Ok(())
}

pub(in crate::inference::flashmoe) fn encode_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

pub(in crate::inference::flashmoe) fn decode_bf16_le(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        bail!("BF16 byte length {} is not divisible by two", bytes.len());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|bytes| {
            let bits = u16::from_le_bytes(bytes.try_into().expect("two-byte chunk"));
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect())
}

pub(in crate::inference::flashmoe) fn write_bf16_as_f32_tensor(
    out: &mut impl Write,
    tensor_name: &str,
    source: &[u8],
    expected_runtime_bytes: usize,
) -> Result<()> {
    if source.len().checked_mul(2) != Some(expected_runtime_bytes) {
        bail!(
            "BF16-to-F32 tensor {tensor_name} has {} source bytes, expected {} for a {expected_runtime_bytes}-byte runtime tensor",
            source.len(),
            expected_runtime_bytes / 2
        );
    }
    for value in decode_bf16_le(source)? {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

pub(in crate::inference::flashmoe) fn write_mlx_affine8_bf16_tensor(
    out: &mut impl Write,
    tensor_name: &str,
    source_weights: &[u8],
    source_scale_bytes: &[u8],
    source_bias_bytes: &[u8],
    shape: &[usize],
) -> Result<()> {
    let [rows, cols] = shape else {
        bail!("MLX affine int8 tensor {tensor_name} must be a matrix, got {shape:?}");
    };
    let expected_weights = rows
        .checked_mul(*cols)
        .context("MLX affine int8 tensor element count overflow")?;
    if source_weights.len() != expected_weights {
        bail!(
            "MLX affine int8 tensor {tensor_name} has {} source bytes, expected {expected_weights}",
            source_weights.len()
        );
    }
    let source_scales = decode_bf16_le(source_scale_bytes)?;
    let source_biases = decode_bf16_le(source_bias_bytes)?;
    let groups_per_row = cols.div_ceil(GROUP_SIZE);
    let expected_groups = rows
        .checked_mul(groups_per_row)
        .context("MLX affine int8 scale/bias count overflow")?;
    if source_scales.len() != expected_groups || source_biases.len() != expected_groups {
        bail!(
            "MLX affine int8 tensor {tensor_name} has {}/{} scales/biases, expected {expected_groups} each",
            source_scales.len(),
            source_biases.len()
        );
    }
    for row in 0..*rows {
        for col in 0..*cols {
            let group = row * groups_per_row + col / GROUP_SIZE;
            let value = source_weights[row * *cols + col] as f32 * source_scales[group]
                + source_biases[group];
            out.write_all(&encode_bf16_bits(value).to_le_bytes())?;
        }
    }
    Ok(())
}

pub(in crate::inference::flashmoe) fn decode_f32_le(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        bail!("F32 byte length {} is not divisible by four", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

pub(in crate::inference::flashmoe) fn write_colibri_int8_bf16_tensor(
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

pub(in crate::inference::flashmoe) fn write_colibri_q4_affine_tensor(
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

pub(in crate::inference::flashmoe) fn write_mlx_mxfp4_affine_tensor(
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

pub(in crate::inference::flashmoe) fn write_padding(
    out: &mut fs::File,
    mut bytes: u64,
) -> Result<()> {
    const ZEROES: [u8; 4096] = [0; 4096];
    while bytes > 0 {
        let n = usize::try_from(bytes.min(ZEROES.len() as u64)).unwrap_or(ZEROES.len());
        out.write_all(&ZEROES[..n])
            .context("failed to write tensor alignment padding")?;
        bytes -= n as u64;
    }
    Ok(())
}
