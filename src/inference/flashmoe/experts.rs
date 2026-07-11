use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
#[cfg(not(unix))]
use std::io::Read;
use std::io::{Seek, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(test)]
use super::math::q4_fma_matvec_with_group_size;
use super::math::quantize_q4;
use super::model_family::{
    QwenModelConfig, QwenMoeExpertComponentKind, QwenMoeExpertComponentLayout, QwenMoeModelLayout,
    QwenMoeQ4ExpertLayout,
};
use super::safetensors::{SafetensorShard, parse_safetensors_header};
#[cfg(test)]
use super::types::HIDDEN_DIM;
use super::types::{ACTIVE_EXPERTS_PER_TOKEN, ExpertQuantization, GROUP_SIZE};
use super::weights::{ExpertTensorRef, decode_dense_tensor_f32};

pub type ReusableExpertBytePool = Arc<Mutex<Vec<Vec<u8>>>>;

const FIXED_Q4_EXPERT_BUFFER_POOL_LIMIT: usize = ACTIVE_EXPERTS_PER_TOKEN * 4;
pub(crate) const PBQ4_EXPERT_MAGIC: &[u8] = b"PBQ4EXPERT ";
pub(crate) const PBQ4_EXPERT_LAYER_FORMAT_V1: &str = "PBQ4EXPERT_LAYER_V1";
pub(crate) const PBQ4_EXPERT_LAYER_FORMAT_V2: &str = "PBQ4EXPERT_LAYER_V2";
pub(crate) const FIXED_Q4_EXPERT_LAYER_FORMAT_V1: &str = "FIXED_Q4_EXPERT_LAYER_V1";
pub(crate) const FIXED_DENSE_EXPERT_LAYER_FORMAT_V1: &str = "FIXED_DENSE_EXPERT_LAYER_V1";
const EXPERT_COMPONENT_ALIGNMENT: usize = 4096;
pub(crate) const EXPERT_SCALE_BIAS_DTYPE_F32: &str = "F32";
pub(crate) const EXPERT_SCALE_BIAS_DTYPE_BF16: &str = "BF16";
pub(crate) const EXPERT_PACK_SCALE_BIAS_DTYPE: &str = EXPERT_SCALE_BIAS_DTYPE_BF16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertReadPath {
    PositionedRead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertIoPolicy {
    pub expert_read_path: ExpertReadPath,
    pub application_expert_cache: bool,
    pub lz4_expert_compression: bool,
    pub speculative_routing: bool,
    pub broad_ssd_gpu_overlap: bool,
}

// Expert scheduler policy guardrails:
// - read packed experts with positioned reads, not mmap;
// - do not add an application-level expert LRU/cache;
// - do not add LZ4 expert compression;
// - do not speculate future expert routes;
// - avoid broad SSD/GPU overlap beyond the existing narrow deferred expert phase.
//
// These choices follow Flash-MoE's "Trust the OS" result: the OS page cache plus
// parallel pread won over custom expert caches, mmap expert files, LZ4, prefetch
// hints, speculative routing, dispatch_io, and aggressive SSD/GPU overlap.
// See https://github.com/danveloper/flash-moe, especially the README "Trust the
// OS" notes and docs/optimization-experiments-q4.md.
pub const FLASHMOE_EXPERT_IO_POLICY: ExpertIoPolicy = ExpertIoPolicy {
    expert_read_path: ExpertReadPath::PositionedRead,
    application_expert_cache: false,
    lz4_expert_compression: false,
    speculative_routing: false,
    broad_ssd_gpu_overlap: false,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ExpertLayerPackMetadata {
    pub(crate) format: String,
    pub(crate) layer: usize,
    pub(crate) expert_size: u64,
    pub(crate) experts: usize,
    pub(crate) packs: Vec<ExpertPackMetadata>,
}

impl ExpertLayerPackMetadata {
    pub(crate) fn new(
        layer: usize,
        expert_size: u64,
        experts: usize,
        packs: Vec<ExpertPackMetadata>,
    ) -> Self {
        Self {
            format: PBQ4_EXPERT_LAYER_FORMAT_V2.to_string(),
            layer,
            expert_size,
            experts,
            packs,
        }
    }

    pub(crate) fn new_fixed_q4(
        layer: usize,
        expert_size: u64,
        experts: usize,
        packs: Vec<ExpertPackMetadata>,
    ) -> Self {
        Self {
            format: FIXED_Q4_EXPERT_LAYER_FORMAT_V1.to_string(),
            layer,
            expert_size,
            experts,
            packs,
        }
    }

    pub(crate) fn new_fixed_dense(
        layer: usize,
        expert_size: u64,
        experts: usize,
        packs: Vec<ExpertPackMetadata>,
    ) -> Self {
        Self {
            format: FIXED_DENSE_EXPERT_LAYER_FORMAT_V1.to_string(),
            layer,
            expert_size,
            experts,
            packs,
        }
    }

    pub(crate) fn pack_for(&self, expert: usize) -> Option<&ExpertPackMetadata> {
        self.packs.iter().find(|metadata| metadata.expert == expert)
    }

    pub(crate) fn validate(&self, path: &Path, layer: usize) -> Result<()> {
        if self.format != PBQ4_EXPERT_LAYER_FORMAT_V1
            && self.format != PBQ4_EXPERT_LAYER_FORMAT_V2
            && self.format != FIXED_Q4_EXPERT_LAYER_FORMAT_V1
            && self.format != FIXED_DENSE_EXPERT_LAYER_FORMAT_V1
        {
            bail!(
                "expert metadata {} has unsupported format {}",
                path.display(),
                self.format
            );
        }
        if self.layer != layer {
            bail!(
                "expert metadata {} describes layer {}, expected layer {layer}",
                path.display(),
                self.layer
            );
        }
        if self.expert_size == 0 {
            bail!("expert metadata {} has zero expert_size", path.display());
        }
        if self.experts == 0 {
            bail!("expert metadata {} has zero experts", path.display());
        }
        let mut seen = BTreeSet::new();
        for pack in &self.packs {
            validate_expert_pack_metadata(path, pack, layer, pack.expert)?;
            if pack.expert >= self.experts {
                bail!(
                    "expert metadata {} describes expert {} outside 0..{}",
                    path.display(),
                    pack.expert,
                    self.experts
                );
            }
            if !seen.insert(pack.expert) {
                bail!(
                    "expert metadata {} has duplicate expert {}",
                    path.display(),
                    pack.expert
                );
            }
            if pack.packed_bytes > self.expert_size {
                bail!(
                    "expert metadata {} expert {} length {} exceeds slot size {}",
                    path.display(),
                    pack.expert,
                    pack.packed_bytes,
                    self.expert_size
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ExpertPackMetadata {
    pub(crate) layer: usize,
    pub(crate) expert: usize,
    #[serde(default)]
    pub(crate) packed_bytes: u64,
    pub(crate) records: Vec<ExpertPackRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ExpertPackRecord {
    pub(crate) tensor: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) source_offsets: [u64; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_hash: Option<String>,
    pub(crate) record_offset: u64,
    pub(crate) packed_bytes: u64,
    pub(crate) groups: usize,
    pub(crate) group_size: usize,
    #[serde(default = "default_expert_scale_bias_dtype")]
    pub(crate) scale_bias_dtype: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedExpertTensor {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) source_offsets: [u64; 2],
    pub(crate) source_hash: Option<String>,
    pub(crate) group_size: usize,
    pub(crate) scale_bias_dtype: String,
    pub(crate) packed: Vec<u8>,
    pub(crate) scales: Vec<f32>,
    pub(crate) biases: Vec<f32>,
    pub(crate) scale_bytes: Vec<u8>,
    pub(crate) bias_bytes: Vec<u8>,
}

impl PackedExpertTensor {
    pub(crate) fn source_offsets(&self) -> [u64; 2] {
        self.source_offsets
    }

    #[cfg(test)]
    pub(crate) fn matvec_payload(
        &self,
        hidden: &[f32],
        width: usize,
    ) -> Option<Q4MatvecPayload<'_>> {
        if hidden.is_empty() || width == 0 || self.packed.is_empty() || self.group_size == 0 {
            return None;
        }
        let shape_cols = self.shape.last().copied().unwrap_or(hidden.len());
        let cols = shape_cols.min(hidden.len()).max(1);
        let shape_rows = self.shape.first().copied().unwrap_or(width);
        let rows = shape_rows.min(width).max(1);
        let groups_per_row = cols.div_ceil(self.group_size).max(1);
        let needed_groups = rows.checked_mul(groups_per_row)?;
        if self.scales.len() < needed_groups || self.biases.len() < needed_groups {
            return None;
        }
        let needed_packed = rows.checked_mul(cols.div_ceil(2))?;
        if self.packed.len() < needed_packed {
            return None;
        }
        let scale_bias_groups = needed_groups;
        let scale_bytes_len = if self
            .scale_bias_dtype
            .eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16)
        {
            needed_groups.checked_mul(2)?
        } else {
            needed_groups.checked_mul(4)?
        };
        let scale_bytes = if self.scale_bytes.len() >= scale_bytes_len {
            &self.scale_bytes[..scale_bytes_len]
        } else {
            &[]
        };
        let bias_bytes = if self.bias_bytes.len() >= scale_bytes_len {
            &self.bias_bytes[..scale_bytes_len]
        } else {
            &[]
        };
        Some(Q4MatvecPayload {
            rows,
            cols,
            group_size: self.group_size,
            packed: &self.packed[..needed_packed],
            #[cfg(test)]
            scales: &self.scales[..needed_groups],
            #[cfg(test)]
            biases: &self.biases[..needed_groups],
            scale_bias_groups,
            scale_bias_dtype: &self.scale_bias_dtype,
            scale_bytes,
            bias_bytes,
            source: None,
        })
    }
}

pub(crate) fn parse_pbq4_expert_pack(
    bytes: &[u8],
    metadata: Option<&ExpertPackMetadata>,
) -> Result<Vec<PackedExpertTensor>> {
    if let Some(metadata) = metadata {
        return parse_pbq4_expert_pack_with_metadata(bytes, metadata);
    }
    parse_pbq4_expert_pack_generic(bytes, None)
}

fn parse_pbq4_expert_pack_with_metadata(
    bytes: &[u8],
    metadata: &ExpertPackMetadata,
) -> Result<Vec<PackedExpertTensor>> {
    if !bytes.starts_with(PBQ4_EXPERT_MAGIC) {
        bail!("expert pack is missing PBQ4EXPERT header");
    }
    let mut cursor = PBQ4_EXPERT_MAGIC.len();
    let mut records = Vec::with_capacity(metadata.records.len());
    for meta in &metadata.records {
        if cursor as u64 != meta.record_offset {
            bail!(
                "expert tensor {} metadata offset mismatch: file cursor {}, metadata has {}",
                meta.tensor,
                cursor,
                meta.record_offset
            );
        }
        let name_len = read_u32_le(bytes, &mut cursor)? as usize;
        let name_end = cursor
            .checked_add(name_len)
            .context("expert tensor name length overflow")?;
        if name_end > bytes.len() {
            bail!("expert tensor name extends past end of pack");
        }
        let name_bytes = &bytes[cursor..name_end];
        if name_bytes != meta.tensor.as_bytes() {
            let file_name =
                std::str::from_utf8(name_bytes).unwrap_or("<invalid utf-8 expert tensor name>");
            bail!(
                "expert tensor metadata name mismatch: file has {file_name}, metadata has {}",
                meta.tensor
            );
        }
        cursor = name_end;
        let packed_len = read_u64_le(bytes, &mut cursor)?;
        if meta.packed_bytes != packed_len {
            bail!(
                "expert tensor {} packed length mismatch: file has {packed_len}, metadata has {}",
                meta.tensor,
                meta.packed_bytes
            );
        }
        let group_count = usize::try_from(read_u64_le(bytes, &mut cursor)?)
            .context("expert group count does not fit usize")?;
        if meta.groups != group_count {
            bail!(
                "expert tensor {} group count mismatch: file has {group_count}, metadata has {}",
                meta.tensor,
                meta.groups
            );
        }
        let scale_start = cursor;
        let scales =
            read_expert_scale_bias_vec_le(bytes, &mut cursor, group_count, &meta.scale_bias_dtype)
                .with_context(|| {
                    format!(
                        "failed to parse q4 scales for expert tensor {}",
                        meta.tensor
                    )
                })?;
        let scale_bytes = bytes[scale_start..cursor].to_vec();
        let bias_start = cursor;
        let biases =
            read_expert_scale_bias_vec_le(bytes, &mut cursor, group_count, &meta.scale_bias_dtype)
                .with_context(|| {
                    format!(
                        "failed to parse q4 biases for expert tensor {}",
                        meta.tensor
                    )
                })?;
        let bias_bytes = bytes[bias_start..cursor].to_vec();
        let packed_len =
            usize::try_from(packed_len).context("expert packed length does not fit usize")?;
        let packed_end = cursor
            .checked_add(packed_len)
            .context("expert packed value range overflow")?;
        if packed_end > bytes.len() {
            bail!(
                "expert packed values for tensor {} extend past end of pack",
                meta.tensor
            );
        }
        let packed = bytes[cursor..packed_end].to_vec();
        cursor = packed_end;
        records.push(PackedExpertTensor {
            name: meta.tensor.clone(),
            dtype: meta.dtype.clone(),
            shape: meta.shape.clone(),
            source_offsets: meta.source_offsets,
            source_hash: meta.source_hash.clone(),
            group_size: meta.group_size,
            scale_bias_dtype: meta.scale_bias_dtype.clone(),
            packed,
            scales,
            biases,
            scale_bytes,
            bias_bytes,
        });
    }
    if cursor != bytes.len() {
        bail!(
            "expert pack has {} trailing bytes after metadata records",
            bytes.len() - cursor
        );
    }
    Ok(records)
}

pub(crate) fn parse_pbq4_expert_pack_generic(
    bytes: &[u8],
    metadata: Option<&ExpertPackMetadata>,
) -> Result<Vec<PackedExpertTensor>> {
    if !bytes.starts_with(PBQ4_EXPERT_MAGIC) {
        bail!("expert pack is missing PBQ4EXPERT header");
    }
    let mut cursor = PBQ4_EXPERT_MAGIC.len();
    let mut records = Vec::new();
    while cursor < bytes.len() {
        let record_start = cursor as u64;
        let name_len = read_u32_le(bytes, &mut cursor)? as usize;
        let name_end = cursor
            .checked_add(name_len)
            .context("expert tensor name length overflow")?;
        if name_end > bytes.len() {
            bail!("expert tensor name extends past end of pack");
        }
        let name = std::str::from_utf8(&bytes[cursor..name_end])
            .context("expert tensor name is not valid UTF-8")?
            .to_string();
        cursor = name_end;
        let packed_len = usize::try_from(read_u64_le(bytes, &mut cursor)?)
            .context("expert packed length does not fit usize")?;
        let group_count = usize::try_from(read_u64_le(bytes, &mut cursor)?)
            .context("expert group count does not fit usize")?;

        let meta = metadata.and_then(|metadata| {
            metadata
                .records
                .iter()
                .find(|record| record.tensor == name && record.record_offset == record_start)
                .or_else(|| metadata.records.iter().find(|record| record.tensor == name))
        });
        let scale_bias_dtype = meta
            .map(|record| record.scale_bias_dtype.as_str())
            .unwrap_or(EXPERT_SCALE_BIAS_DTYPE_F32);
        let scale_start = cursor;
        let scales =
            read_expert_scale_bias_vec_le(bytes, &mut cursor, group_count, scale_bias_dtype)
                .with_context(|| format!("failed to parse q4 scales for expert tensor {name}"))?;
        let scale_bytes = bytes[scale_start..cursor].to_vec();
        let bias_start = cursor;
        let biases =
            read_expert_scale_bias_vec_le(bytes, &mut cursor, group_count, scale_bias_dtype)
                .with_context(|| format!("failed to parse q4 biases for expert tensor {name}"))?;
        let bias_bytes = bytes[bias_start..cursor].to_vec();
        let packed_end = cursor
            .checked_add(packed_len)
            .context("expert packed value range overflow")?;
        if packed_end > bytes.len() {
            bail!("expert packed values for tensor {name} extend past end of pack");
        }
        let packed = bytes[cursor..packed_end].to_vec();
        cursor = packed_end;

        if let Some(meta) = meta {
            if meta.packed_bytes != packed_len as u64 {
                bail!(
                    "expert tensor {name} packed length mismatch: file has {packed_len}, metadata has {}",
                    meta.packed_bytes
                );
            }
            if meta.groups != group_count {
                bail!(
                    "expert tensor {name} group count mismatch: file has {group_count}, metadata has {}",
                    meta.groups
                );
            }
        }
        records.push(PackedExpertTensor {
            name,
            dtype: meta
                .map(|record| record.dtype.clone())
                .unwrap_or_else(|| "q4".to_string()),
            shape: meta.map(|record| record.shape.clone()).unwrap_or_default(),
            source_offsets: meta.map(|record| record.source_offsets).unwrap_or([0, 0]),
            source_hash: meta.and_then(|record| record.source_hash.clone()),
            group_size: meta.map(|record| record.group_size).unwrap_or(GROUP_SIZE),
            scale_bias_dtype: scale_bias_dtype.to_string(),
            packed,
            scales,
            biases,
            scale_bytes,
            bias_bytes,
        });
    }
    Ok(records)
}

fn read_expert_scale_bias_vec_le(
    bytes: &[u8],
    cursor: &mut usize,
    len: usize,
    dtype: &str,
) -> Result<Vec<f32>> {
    match dtype.to_ascii_uppercase().as_str() {
        EXPERT_SCALE_BIAS_DTYPE_F32 | "FLOAT32" | "FP32" => read_f32_vec_le(bytes, cursor, len),
        EXPERT_SCALE_BIAS_DTYPE_BF16 | "BFLOAT16" => read_bf16_vec_le(bytes, cursor, len),
        other => bail!("unsupported q4 scale/bias dtype {other}"),
    }
}

#[cfg(test)]
fn decode_expert_scale_bias_bytes(bytes: &[u8], len: usize, dtype: &str) -> Result<Vec<f32>> {
    let mut cursor = 0;
    let values = read_expert_scale_bias_vec_le(bytes, &mut cursor, len, dtype)?;
    if cursor != bytes.len() {
        bail!(
            "expert scale/bias payload has {} trailing bytes",
            bytes.len() - cursor
        );
    }
    Ok(values)
}

#[cfg(test)]
fn fixed_q4_expert_records(
    view: &FixedQ4ExpertSlotView<'_>,
    spec: FixedQ4ExpertSlotSpec,
) -> Result<Vec<PackedExpertTensor>> {
    let descriptor = view.descriptor();
    Ok(vec![
        fixed_q4_expert_record(
            view,
            spec,
            QwenMoeExpertComponentKind::GateWeight,
            QwenMoeExpertComponentKind::GateScale,
            QwenMoeExpertComponentKind::GateBias,
            "gate_proj",
            vec![spec.intermediate_size, spec.hidden_size],
            descriptor.layer,
            descriptor.expert,
        )?,
        fixed_q4_expert_record(
            view,
            spec,
            QwenMoeExpertComponentKind::UpWeight,
            QwenMoeExpertComponentKind::UpScale,
            QwenMoeExpertComponentKind::UpBias,
            "up_proj",
            vec![spec.intermediate_size, spec.hidden_size],
            descriptor.layer,
            descriptor.expert,
        )?,
        fixed_q4_expert_record(
            view,
            spec,
            QwenMoeExpertComponentKind::DownWeight,
            QwenMoeExpertComponentKind::DownScale,
            QwenMoeExpertComponentKind::DownBias,
            "down_proj",
            vec![spec.hidden_size, spec.intermediate_size],
            descriptor.layer,
            descriptor.expert,
        )?,
    ])
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn fixed_q4_expert_record(
    view: &FixedQ4ExpertSlotView<'_>,
    spec: FixedQ4ExpertSlotSpec,
    packed_kind: QwenMoeExpertComponentKind,
    scale_kind: QwenMoeExpertComponentKind,
    bias_kind: QwenMoeExpertComponentKind,
    projection: &str,
    shape: Vec<usize>,
    layer: usize,
    expert: usize,
) -> Result<PackedExpertTensor> {
    let scale_bytes = view.component(scale_kind).to_vec();
    let bias_bytes = view.component(bias_kind).to_vec();
    let scale_value_bytes = scale_bytes.len() / 2 * 2;
    let bias_value_bytes = bias_bytes.len() / 2 * 2;
    let scales = decode_expert_scale_bias_bytes(
        &scale_bytes[..scale_value_bytes],
        scale_bytes.len() / 2,
        EXPERT_SCALE_BIAS_DTYPE_BF16,
    )
    .with_context(|| format!("failed to decode fixed Q4 {projection} scales"))?;
    let biases = decode_expert_scale_bias_bytes(
        &bias_bytes[..bias_value_bytes],
        bias_bytes.len() / 2,
        EXPERT_SCALE_BIAS_DTYPE_BF16,
    )
    .with_context(|| format!("failed to decode fixed Q4 {projection} biases"))?;
    Ok(PackedExpertTensor {
        name: format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"),
        dtype: "q4".to_string(),
        shape,
        source_offsets: [0, 0],
        source_hash: None,
        group_size: spec.layout.group_size,
        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        packed: view.component(packed_kind).to_vec(),
        scales,
        biases,
        scale_bytes,
        bias_bytes,
    })
}

fn read_u32_le(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let end = cursor.checked_add(4).context("u32 cursor overflow")?;
    if end > bytes.len() {
        bail!("unexpected end of expert pack while reading u32");
    }
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[*cursor..end]);
    *cursor = end;
    Ok(u32::from_le_bytes(raw))
}

fn read_u64_le(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let end = cursor.checked_add(8).context("u64 cursor overflow")?;
    if end > bytes.len() {
        bail!("unexpected end of expert pack while reading u64");
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[*cursor..end]);
    *cursor = end;
    Ok(u64::from_le_bytes(raw))
}

fn read_f32_vec_le(bytes: &[u8], cursor: &mut usize, len: usize) -> Result<Vec<f32>> {
    let byte_len = len.checked_mul(4).context("f32 vector length overflow")?;
    let end = cursor
        .checked_add(byte_len)
        .context("f32 vector cursor overflow")?;
    if end > bytes.len() {
        bail!("unexpected end of expert pack while reading f32 vector");
    }
    #[cfg(target_endian = "little")]
    {
        let mut values = vec![0.0f32; len];
        if byte_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes[*cursor..end].as_ptr(),
                    values.as_mut_ptr().cast::<u8>(),
                    byte_len,
                );
            }
        }
        *cursor = end;
        return Ok(values);
    }

    #[cfg(not(target_endian = "little"))]
    let values = bytes[*cursor..end]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    #[cfg(not(target_endian = "little"))]
    {
        *cursor = end;
        Ok(values)
    }
}

fn read_bf16_vec_le(bytes: &[u8], cursor: &mut usize, len: usize) -> Result<Vec<f32>> {
    let byte_len = len.checked_mul(2).context("bf16 vector length overflow")?;
    let end = cursor
        .checked_add(byte_len)
        .context("bf16 vector cursor overflow")?;
    if end > bytes.len() {
        bail!("unexpected end of expert pack while reading bf16 vector");
    }
    let values = bytes[*cursor..end]
        .chunks_exact(2)
        .map(|chunk| {
            let hi = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
            f32::from_bits(hi << 16)
        })
        .collect();
    *cursor = end;
    Ok(values)
}

pub(crate) fn default_expert_scale_bias_dtype() -> String {
    EXPERT_SCALE_BIAS_DTYPE_F32.to_string()
}

pub(crate) trait ExpertPackWireRecord {
    fn tensor_name(&self) -> &str;
    fn packed_bytes(&self) -> u64;
    fn scale_bias_groups(&self) -> usize;
    fn scale_bias_dtype(&self) -> &str;
}

#[derive(Debug, Clone)]
pub(crate) struct ExpectedExpertPackRecord {
    pub(crate) tensor: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) source_offsets: [u64; 2],
    pub(crate) source_hash: String,
    pub(crate) packed_bytes: u64,
    pub(crate) groups: usize,
    pub(crate) group_size: usize,
    pub(crate) scale_bias_dtype: String,
}

impl ExpertPackWireRecord for ExpectedExpertPackRecord {
    fn tensor_name(&self) -> &str {
        &self.tensor
    }

    fn packed_bytes(&self) -> u64 {
        self.packed_bytes
    }

    fn scale_bias_groups(&self) -> usize {
        self.groups
    }

    fn scale_bias_dtype(&self) -> &str {
        &self.scale_bias_dtype
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExpectedExpertPack {
    pub(crate) expert: usize,
    pub(crate) packed_bytes: u64,
    pub(crate) records: Vec<ExpectedExpertPackRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertLayerStorageFormat {
    Pbq4Import,
    FixedQ4(FixedQ4ExpertSlotSpec),
    FixedDense(FixedDenseExpertSlotSpec),
}

impl ExpertLayerStorageFormat {
    fn slot_size(self, expected: &[ExpectedExpertPack]) -> u64 {
        match self {
            Self::Pbq4Import => expected
                .iter()
                .map(|pack| pack.packed_bytes)
                .max()
                .unwrap_or(0)
                .max(1),
            Self::FixedQ4(spec) => spec.layout.expert_bytes as u64,
            Self::FixedDense(spec) => spec.expert_bytes as u64,
        }
    }

    fn metadata_matches(self, metadata: &ExpertLayerPackMetadata) -> bool {
        match self {
            Self::Pbq4Import => matches!(
                metadata.format.as_str(),
                PBQ4_EXPERT_LAYER_FORMAT_V1 | PBQ4_EXPERT_LAYER_FORMAT_V2
            ),
            Self::FixedQ4(_) => metadata.format == FIXED_Q4_EXPERT_LAYER_FORMAT_V1,
            Self::FixedDense(_) => metadata.format == FIXED_DENSE_EXPERT_LAYER_FORMAT_V1,
        }
    }

    fn layer_metadata(
        self,
        layer: usize,
        slot_size: u64,
        experts: usize,
        packs: Vec<ExpertPackMetadata>,
    ) -> ExpertLayerPackMetadata {
        match self {
            Self::Pbq4Import => ExpertLayerPackMetadata::new(layer, slot_size, experts, packs),
            Self::FixedQ4(_) => {
                ExpertLayerPackMetadata::new_fixed_q4(layer, slot_size, experts, packs)
            }
            Self::FixedDense(_) => {
                ExpertLayerPackMetadata::new_fixed_dense(layer, slot_size, experts, packs)
            }
        }
    }
}

pub(crate) fn expected_expert_pack_record_from_source(
    tensor: String,
    dtype: String,
    shape: Vec<usize>,
    source_offsets: [u64; 2],
    source_hash: String,
) -> Result<ExpectedExpertPackRecord> {
    let (packed_bytes, groups) = q4_record_layout_for_shape(&shape)?;
    Ok(ExpectedExpertPackRecord {
        tensor,
        dtype,
        shape,
        source_offsets,
        source_hash,
        packed_bytes,
        groups,
        group_size: GROUP_SIZE,
        scale_bias_dtype: EXPERT_PACK_SCALE_BIAS_DTYPE.to_string(),
    })
}

pub(crate) fn expected_native_q4_expert_record_from_input(
    input: NativeQ4ExpertRecordInput,
) -> Result<ExpectedExpertPackRecord> {
    Ok(ExpectedExpertPackRecord {
        tensor: input.tensor,
        dtype: input.dtype,
        shape: input.shape,
        source_offsets: input.source_offsets,
        source_hash: input
            .source_hash
            .context("native q4 expert record is missing source hash")?,
        packed_bytes: input.packed.len() as u64,
        groups: input.groups,
        group_size: GROUP_SIZE,
        scale_bias_dtype: input.scale_bias_dtype,
    })
}

pub(crate) fn expected_expert_pack_from_records(
    expert: usize,
    records: Vec<ExpectedExpertPackRecord>,
) -> Result<ExpectedExpertPack> {
    let packed_bytes = pbq4_expert_pack_wire_size(&records)?;
    Ok(ExpectedExpertPack {
        expert,
        packed_bytes,
        records,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AggregateExpertTensorKind {
    GateUp,
    Gate,
    Up,
    Down,
}

pub(crate) fn aggregate_expert_tensor_kind(name: &str) -> Option<AggregateExpertTensorKind> {
    if name.ends_with(".mlp.experts.gate_up_proj")
        || name.ends_with(".mlp.experts.gate_up_proj.weight")
    {
        Some(AggregateExpertTensorKind::GateUp)
    } else if name.ends_with(".mlp.switch_mlp.gate_proj")
        || name.ends_with(".mlp.switch_mlp.gate_proj.weight")
    {
        Some(AggregateExpertTensorKind::Gate)
    } else if name.ends_with(".mlp.switch_mlp.up_proj")
        || name.ends_with(".mlp.switch_mlp.up_proj.weight")
    {
        Some(AggregateExpertTensorKind::Up)
    } else if name.ends_with(".mlp.experts.down_proj")
        || name.ends_with(".mlp.experts.down_proj.weight")
        || name.ends_with(".mlp.switch_mlp.down_proj")
        || name.ends_with(".mlp.switch_mlp.down_proj.weight")
    {
        Some(AggregateExpertTensorKind::Down)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AggregateExpertLayout {
    pub(crate) experts: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) gate_up_expert_values: usize,
    pub(crate) single_projection_values: usize,
    pub(crate) down_expert_values: usize,
}

impl AggregateExpertLayout {
    pub(crate) fn new(experts: usize, hidden: usize, intermediate: usize) -> Result<Self> {
        let gate_up_expert_values = intermediate
            .checked_mul(2)
            .and_then(|rows| rows.checked_mul(hidden))
            .context("aggregate gate_up expert element count overflow")?;
        let single_projection_values = intermediate
            .checked_mul(hidden)
            .context("aggregate gate/up projection element count overflow")?;
        let down_expert_values = hidden
            .checked_mul(intermediate)
            .context("aggregate down projection element count overflow")?;
        Ok(Self {
            experts,
            hidden,
            intermediate,
            gate_up_expert_values,
            single_projection_values,
            down_expert_values,
        })
    }
}

pub(crate) trait AggregateExpertTensor {
    fn aggregate_tensor_name(&self) -> &str;
    fn aggregate_tensor_shape(&self) -> &[usize];
    fn aggregate_tensor_has_native_q4(&self) -> bool;
}

pub(crate) trait ExpertSourceTensor: AggregateExpertTensor {
    fn expert_source_offsets(&self) -> Option<[u64; 2]>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AggregateExpertTensors<'a, T> {
    pub(crate) gate: AggregateExpertSlice<'a, T>,
    pub(crate) up: AggregateExpertSlice<'a, T>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AggregateExpertSlice<'a, T> {
    pub(crate) tensor: &'a T,
    pub(crate) expert_stride_values: usize,
    pub(crate) expert_offset_values: usize,
}

impl<T> AggregateExpertSlice<'_, T> {
    pub(crate) fn start(&self, expert: usize) -> Result<usize> {
        expert
            .checked_mul(self.expert_stride_values)
            .and_then(|base| base.checked_add(self.expert_offset_values))
            .context("aggregate expert slice offset overflow")
    }
}

pub(crate) fn aggregate_expert_tensors<'a, T: AggregateExpertTensor>(
    tensors: &[&'a T],
    layer: usize,
    layout: AggregateExpertLayout,
) -> Result<AggregateExpertTensors<'a, T>> {
    let gate_up = optional_aggregate_expert_tensor(tensors, AggregateExpertTensorKind::GateUp);
    let gate = optional_aggregate_expert_tensor(tensors, AggregateExpertTensorKind::Gate);
    let up = optional_aggregate_expert_tensor(tensors, AggregateExpertTensorKind::Up);
    match (gate_up, gate, up) {
        (Some(gate_up), None, None) => {
            validate_aggregate_expert_tensor_shape(
                gate_up,
                &[layout.experts, layout.intermediate * 2, layout.hidden],
                "gate_up_proj",
            )?;
            Ok(AggregateExpertTensors {
                gate: AggregateExpertSlice {
                    tensor: gate_up,
                    expert_stride_values: layout.gate_up_expert_values,
                    expert_offset_values: 0,
                },
                up: AggregateExpertSlice {
                    tensor: gate_up,
                    expert_stride_values: layout.gate_up_expert_values,
                    expert_offset_values: layout.single_projection_values,
                },
            })
        }
        (None, Some(gate), Some(up)) => {
            validate_aggregate_expert_tensor_shape(
                gate,
                &[layout.experts, layout.intermediate, layout.hidden],
                "switch_mlp.gate_proj",
            )?;
            validate_aggregate_expert_tensor_shape(
                up,
                &[layout.experts, layout.intermediate, layout.hidden],
                "switch_mlp.up_proj",
            )?;
            Ok(AggregateExpertTensors {
                gate: AggregateExpertSlice {
                    tensor: gate,
                    expert_stride_values: layout.single_projection_values,
                    expert_offset_values: 0,
                },
                up: AggregateExpertSlice {
                    tensor: up,
                    expert_stride_values: layout.single_projection_values,
                    expert_offset_values: 0,
                },
            })
        }
        _ => bail!(
            "aggregate expert layer {layer} must contain either combined gate_up_proj or separate switch_mlp gate_proj/up_proj tensors"
        ),
    }
}

pub(crate) fn single_aggregate_expert_tensor<'a, T: AggregateExpertTensor>(
    tensors: &[&'a T],
    kind: AggregateExpertTensorKind,
    layer: usize,
) -> Result<&'a T> {
    let matches = tensors_matching_aggregate_expert_kind(tensors, kind);
    match matches.as_slice() {
        [tensor] => Ok(*tensor),
        [] => bail!("aggregate expert layer {layer} is missing {kind:?} tensor"),
        _ => bail!("aggregate expert layer {layer} has duplicate {kind:?} tensors"),
    }
}

pub(crate) fn aggregate_native_q4_enabled<T: AggregateExpertTensor>(
    aggregate_tensors: &AggregateExpertTensors<'_, T>,
    down: &T,
) -> Result<bool> {
    let native_count = [
        aggregate_tensors
            .gate
            .tensor
            .aggregate_tensor_has_native_q4(),
        aggregate_tensors.up.tensor.aggregate_tensor_has_native_q4(),
        down.aggregate_tensor_has_native_q4(),
    ]
    .into_iter()
    .filter(|native| *native)
    .count();
    match native_count {
        0 => Ok(false),
        3 => Ok(true),
        _ => bail!("aggregate expert tensors must be all native MLX Q4 or all decoded tensors"),
    }
}

pub(crate) fn fixed_native_q4_aggregate_layout<T: AggregateExpertTensor>(
    aggregate_tensors: &AggregateExpertTensors<'_, T>,
    down: &T,
    layout: AggregateExpertLayout,
) -> Result<Option<QwenMoeQ4ExpertLayout>> {
    if !aggregate_native_q4_enabled(aggregate_tensors, down)? {
        return Ok(None);
    }
    if !layout.hidden.is_multiple_of(GROUP_SIZE) || !layout.intermediate.is_multiple_of(GROUP_SIZE)
    {
        return Ok(None);
    }
    QwenMoeQ4ExpertLayout::fixed_bf16(layout.hidden, layout.intermediate, GROUP_SIZE).map(Some)
}

pub(crate) fn validate_aggregate_expert_tensor_shape<T: AggregateExpertTensor>(
    tensor: &T,
    expected: &[usize; 3],
    label: &str,
) -> Result<()> {
    if tensor.aggregate_tensor_shape() != expected {
        bail!(
            "aggregate expert tensor {} has shape {:?}; expected {:?} for {label}",
            tensor.aggregate_tensor_name(),
            tensor.aggregate_tensor_shape(),
            expected
        );
    }
    Ok(())
}

pub(crate) fn q4_record_layout_for_shape(shape: &[usize]) -> Result<(u64, usize)> {
    let cols = shape.last().copied().unwrap_or(0);
    if cols == 0 {
        bail!("cannot compute q4 layout for zero-column tensor");
    }
    let rows = if shape.len() > 1 {
        shape[..shape.len() - 1].iter().product::<usize>().max(1)
    } else {
        1
    };
    let packed_bytes = rows
        .checked_mul(cols.div_ceil(2))
        .context("q4 packed byte count overflow")?;
    let groups = rows
        .checked_mul(cols.div_ceil(GROUP_SIZE))
        .context("q4 group count overflow")?;
    Ok((packed_bytes as u64, groups))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectExpertTensorShape {
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
}

impl DirectExpertTensorShape {
    pub(crate) fn new(hidden: usize, intermediate: usize) -> Result<Self> {
        if hidden == 0 || intermediate == 0 {
            bail!(
                "direct expert tensor shape must have non-zero hidden and intermediate dimensions"
            );
        }
        Ok(Self {
            hidden,
            intermediate,
        })
    }
}

pub(crate) fn validate_direct_expert_tensor_group<T: AggregateExpertTensor>(
    layer: usize,
    expert: usize,
    tensors: &[&T],
    shape: Option<DirectExpertTensorShape>,
) -> Result<()> {
    let mut seen = BTreeMap::<&'static str, &T>::new();
    for suffix in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
        let matches: Vec<&T> = tensors
            .iter()
            .copied()
            .filter(|tensor| tensor.aggregate_tensor_name().ends_with(suffix))
            .collect();
        match matches.as_slice() {
            [tensor] => {
                seen.insert(suffix, *tensor);
            }
            [] => {
                bail!(
                    "Flash-MoE expert layer {layer} expert {expert} is missing required tensor {suffix}"
                );
            }
            _ => {
                bail!(
                    "Flash-MoE expert layer {layer} expert {expert} has duplicate tensors ending in {suffix}"
                );
            }
        }
    }

    if let Some(shape) = shape {
        validate_direct_expert_matrix_shape(
            seen["gate_proj.weight"],
            &[shape.intermediate, shape.hidden],
            "gate_proj.weight",
        )?;
        validate_direct_expert_matrix_shape(
            seen["up_proj.weight"],
            &[shape.intermediate, shape.hidden],
            "up_proj.weight",
        )?;
        validate_direct_expert_matrix_shape(
            seen["down_proj.weight"],
            &[shape.hidden, shape.intermediate],
            "down_proj.weight",
        )?;
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeQ4SliceByteRanges {
    pub(crate) packed_offset: usize,
    pub(crate) packed_bytes: usize,
    pub(crate) scale_bias_offset: usize,
    pub(crate) scale_bias_bytes: usize,
    pub(crate) groups: usize,
}

pub(crate) fn native_q4_slice_byte_ranges<T: AggregateExpertTensor>(
    source: &T,
    shape: &[usize],
    scale_bias_dtype: &str,
    element_offset: usize,
    element_count: usize,
) -> Result<NativeQ4SliceByteRanges> {
    let source_shape = source.aggregate_tensor_shape();
    if source_shape.len() < 2 {
        bail!(
            "native q4 expert tensor {} has invalid logical shape {:?}",
            source.aggregate_tensor_name(),
            source_shape
        );
    }
    let cols = source_shape.last().copied().unwrap_or(0);
    if cols == 0 || element_offset % cols != 0 || element_count % cols != 0 {
        bail!(
            "native q4 expert tensor {} slice {element_offset}..{} is not aligned to {cols} columns",
            source.aggregate_tensor_name(),
            element_offset.saturating_add(element_count)
        );
    }
    let slice_rows = element_count / cols;
    let expected_rows =
        shape[..shape.len().saturating_sub(1)]
            .iter()
            .try_fold(1usize, |acc, dim| {
                acc.checked_mul(*dim)
                    .context("native q4 expert slice row count overflow")
            })?;
    if shape.last().copied() != Some(cols) || expected_rows != slice_rows {
        bail!(
            "native q4 expert tensor {} slice shape {:?} does not match {slice_rows} rows x {cols} cols",
            source.aggregate_tensor_name(),
            shape
        );
    }

    let scale_bias_width = expert_scale_bias_dtype_size(scale_bias_dtype)?;
    let row_packed_bytes = cols.div_ceil(2);
    let groups_per_row = cols.div_ceil(GROUP_SIZE);
    let row_start = element_offset / cols;
    let packed_offset = row_start
        .checked_mul(row_packed_bytes)
        .context("native q4 expert packed byte offset overflow")?;
    let scale_bias_offset = row_start
        .checked_mul(groups_per_row)
        .and_then(|groups| groups.checked_mul(scale_bias_width))
        .context("native q4 expert scale/bias byte offset overflow")?;
    let packed_bytes = slice_rows
        .checked_mul(row_packed_bytes)
        .context("native q4 expert packed byte count overflow")?;
    let groups = slice_rows
        .checked_mul(groups_per_row)
        .context("native q4 expert group count overflow")?;
    let scale_bias_bytes = groups
        .checked_mul(scale_bias_width)
        .context("native q4 expert scale/bias byte count overflow")?;
    Ok(NativeQ4SliceByteRanges {
        packed_offset,
        packed_bytes,
        scale_bias_offset,
        scale_bias_bytes,
        groups,
    })
}

pub(crate) fn expert_tensor_byte_range<T: ExpertSourceTensor>(
    tensor: &T,
    dtype: &str,
    element_offset: usize,
    element_count: usize,
) -> Result<[u64; 2]> {
    let [tensor_start, tensor_end] = tensor.expert_source_offsets().with_context(|| {
        format!(
            "expert tensor {} is missing source offsets",
            tensor.aggregate_tensor_name()
        )
    })?;
    let element_size = expert_tensor_dtype_size(dtype).with_context(|| {
        format!(
            "expert tensor {} has unsupported dtype {dtype}",
            tensor.aggregate_tensor_name()
        )
    })?;
    let byte_start = tensor_start
        .checked_add(
            (element_offset
                .checked_mul(element_size)
                .context("expert tensor byte offset overflow")?) as u64,
        )
        .context("expert tensor source offset overflow")?;
    let byte_len = element_count
        .checked_mul(element_size)
        .context("expert tensor byte length overflow")?;
    let byte_end = byte_start
        .checked_add(byte_len as u64)
        .context("expert tensor byte range overflow")?;
    if byte_end > tensor_end {
        bail!(
            "expert tensor {} range {}..{} exceeds source offsets {:?}",
            tensor.aggregate_tensor_name(),
            byte_start,
            byte_end,
            [tensor_start, tensor_end]
        );
    }
    Ok([byte_start, byte_end])
}

fn expert_tensor_dtype_size(dtype: &str) -> Option<usize> {
    match dtype.to_ascii_uppercase().as_str() {
        "F32" | "FLOAT32" | "FP32" => Some(4),
        "BF16" | "BFLOAT16" | "F16" | "FLOAT16" | "FP16" => Some(2),
        "U8" | "I8" => Some(1),
        _ => None,
    }
}

fn validate_direct_expert_matrix_shape<T: AggregateExpertTensor>(
    tensor: &T,
    expected: &[usize; 2],
    suffix: &str,
) -> Result<()> {
    if tensor.aggregate_tensor_shape() != expected {
        bail!(
            "Flash-MoE expert tensor {} has shape {:?}; expected {:?} for {suffix}",
            tensor.aggregate_tensor_name(),
            tensor.aggregate_tensor_shape(),
            expected
        );
    }
    Ok(())
}

fn optional_aggregate_expert_tensor<'a, T: AggregateExpertTensor>(
    tensors: &[&'a T],
    kind: AggregateExpertTensorKind,
) -> Option<&'a T> {
    let matches = tensors_matching_aggregate_expert_kind(tensors, kind);
    match matches.as_slice() {
        [tensor] => Some(*tensor),
        _ => None,
    }
}

fn tensors_matching_aggregate_expert_kind<'a, T: AggregateExpertTensor>(
    tensors: &[&'a T],
    kind: AggregateExpertTensorKind,
) -> Vec<&'a T> {
    tensors
        .iter()
        .copied()
        .filter(|tensor| aggregate_expert_tensor_kind(tensor.aggregate_tensor_name()) == Some(kind))
        .collect()
}

pub(crate) fn pbq4_expert_pack_wire_size<R: ExpertPackWireRecord>(records: &[R]) -> Result<u64> {
    let mut size = PBQ4_EXPERT_MAGIC.len() as u64;
    for record in records {
        let scale_bias_bytes = expert_scale_bias_dtype_size(record.scale_bias_dtype())
            .with_context(|| {
                format!(
                    "cannot compute expert pack wire size for q4 scale/bias dtype {}",
                    record.scale_bias_dtype()
                )
            })?;
        let groups = record.scale_bias_groups() as u64;
        let record_size = 4u64
            .checked_add(record.tensor_name().len() as u64)
            .and_then(|size| size.checked_add(8))
            .and_then(|size| size.checked_add(8))
            .and_then(|size| size.checked_add(groups.checked_mul(scale_bias_bytes as u64)?))
            .and_then(|size| size.checked_add(groups.checked_mul(scale_bias_bytes as u64)?))
            .and_then(|size| size.checked_add(record.packed_bytes()))
            .context("expert pack record wire size overflow")?;
        size = size
            .checked_add(record_size)
            .context("expert pack wire size overflow")?;
    }
    Ok(size)
}

pub(crate) fn expert_scale_bias_dtype_size(dtype: &str) -> Result<usize> {
    match dtype.to_ascii_uppercase().as_str() {
        EXPERT_SCALE_BIAS_DTYPE_F32 | "FLOAT32" | "FP32" => Ok(4),
        EXPERT_SCALE_BIAS_DTYPE_BF16 | "BFLOAT16" => Ok(2),
        other => bail!("unsupported q4 scale/bias dtype {other}"),
    }
}

pub(crate) struct ExpertRecordInput {
    pub(crate) tensor: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) source_offsets: [u64; 2],
    pub(crate) source_hash: Option<String>,
    pub(crate) values: Vec<f32>,
}

pub(crate) struct FixedDenseExpertRecordInput {
    pub(crate) tensor: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) source_offsets: [u64; 2],
    pub(crate) source_hash: Option<String>,
    pub(crate) bytes: Vec<u8>,
}

pub(crate) struct NativeQ4ExpertRecordInput {
    pub(crate) tensor: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) source_offsets: [u64; 2],
    pub(crate) source_hash: Option<String>,
    pub(crate) packed: Vec<u8>,
    pub(crate) scale_bytes: Vec<u8>,
    pub(crate) bias_bytes: Vec<u8>,
    pub(crate) groups: usize,
    pub(crate) scale_bias_dtype: String,
}

pub(crate) fn build_expert_pack(
    layer: usize,
    expert: usize,
    inputs: Vec<ExpertRecordInput>,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    let mut out = std::io::Cursor::new(Vec::new());
    let mut records = Vec::with_capacity(inputs.len());
    out.write_all(PBQ4_EXPERT_MAGIC)
        .context("failed to write packed expert header")?;
    for input in inputs {
        write_quantized_expert_record(&mut out, &mut records, input)?;
    }
    let packed = out.into_inner();
    let metadata = ExpertPackMetadata {
        layer,
        expert,
        packed_bytes: packed.len() as u64,
        records,
    };
    Ok((packed, metadata))
}

pub(crate) fn build_fixed_dense_expert_pack(
    layer: usize,
    expert: usize,
    spec: FixedDenseExpertSlotSpec,
    inputs: Vec<FixedDenseExpertRecordInput>,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    if inputs.len() != 3 {
        bail!(
            "fixed {} expert pack requires gate, up, and down inputs",
            spec.dtype.as_str()
        );
    }
    let mut out = vec![0u8; spec.expert_bytes];
    let mut records = Vec::with_capacity(3);
    for projection in [
        ExpertMlpProjection::Gate,
        ExpertMlpProjection::Up,
        ExpertMlpProjection::Down,
    ] {
        let suffix = match projection {
            ExpertMlpProjection::Gate => "gate_proj.weight",
            ExpertMlpProjection::Up => "up_proj.weight",
            ExpertMlpProjection::Down => "down_proj.weight",
        };
        let index = inputs
            .iter()
            .position(|input| input.tensor.ends_with(suffix))
            .with_context(|| {
                format!(
                    "fixed {} expert pack layer {layer} expert {expert} is missing {suffix}",
                    spec.dtype.as_str()
                )
            })?;
        let input = &inputs[index];
        let component = spec.projection(projection);
        if input.shape != [component.rows, component.cols] {
            bail!(
                "fixed {} expert tensor {} has shape {:?}, expected [{}, {}]",
                spec.dtype.as_str(),
                input.tensor,
                input.shape,
                component.rows,
                component.cols
            );
        }
        let dtype_matches = match spec.dtype {
            DenseExpertDtype::Bf16 => {
                matches!(
                    input.dtype.to_ascii_uppercase().as_str(),
                    "BF16" | "BFLOAT16"
                )
            }
            DenseExpertDtype::F16 => matches!(
                input.dtype.to_ascii_uppercase().as_str(),
                "F16" | "FLOAT16" | "FP16"
            ),
        };
        if !dtype_matches || input.bytes.len() != component.bytes {
            bail!(
                "fixed {} expert tensor {} has dtype {} and {} bytes, expected {} bytes",
                spec.dtype.as_str(),
                input.tensor,
                input.dtype,
                input.bytes.len(),
                component.bytes
            );
        }
        out[component.offset..component.offset + component.bytes].copy_from_slice(&input.bytes);
        records.push(ExpertPackRecord {
            tensor: input.tensor.clone(),
            dtype: input.dtype.clone(),
            shape: input.shape.clone(),
            source_offsets: input.source_offsets,
            source_hash: input.source_hash.clone(),
            record_offset: component.offset as u64,
            packed_bytes: component.bytes as u64,
            groups: 0,
            group_size: 0,
            scale_bias_dtype: spec.dtype.as_str().to_string(),
        });
    }
    Ok((
        out,
        ExpertPackMetadata {
            layer,
            expert,
            packed_bytes: spec.expert_bytes as u64,
            records,
        },
    ))
}

pub(crate) fn build_native_q4_expert_pack(
    layer: usize,
    expert: usize,
    inputs: Vec<NativeQ4ExpertRecordInput>,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    let mut out = std::io::Cursor::new(Vec::new());
    let mut records = Vec::with_capacity(inputs.len());
    out.write_all(PBQ4_EXPERT_MAGIC)
        .context("failed to write packed expert header")?;
    for input in inputs {
        write_native_q4_expert_record(&mut out, &mut records, input)?;
    }
    let packed = out.into_inner();
    let metadata = ExpertPackMetadata {
        layer,
        expert,
        packed_bytes: packed.len() as u64,
        records,
    };
    Ok((packed, metadata))
}

pub(crate) fn build_fixed_native_q4_expert_pack(
    layer: usize,
    expert: usize,
    fixed: QwenMoeQ4ExpertLayout,
    inputs: Vec<NativeQ4ExpertRecordInput>,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    if inputs.len() != 3 {
        bail!("fixed native q4 expert pack requires gate, up, and down inputs");
    }
    let [gate, up, down]: [NativeQ4ExpertRecordInput; 3] =
        inputs
            .try_into()
            .map_err(|inputs: Vec<NativeQ4ExpertRecordInput>| {
                anyhow::anyhow!(
                    "fixed native q4 expert pack requires 3 inputs, got {}",
                    inputs.len()
                )
            })?;
    let mut out = vec![0u8; fixed.expert_bytes];
    let mut records = Vec::with_capacity(3);
    write_fixed_native_q4_component(
        &mut out,
        &mut records,
        fixed,
        gate,
        QwenMoeExpertComponentKind::GateWeight,
        QwenMoeExpertComponentKind::GateScale,
        QwenMoeExpertComponentKind::GateBias,
    )?;
    write_fixed_native_q4_component(
        &mut out,
        &mut records,
        fixed,
        up,
        QwenMoeExpertComponentKind::UpWeight,
        QwenMoeExpertComponentKind::UpScale,
        QwenMoeExpertComponentKind::UpBias,
    )?;
    write_fixed_native_q4_component(
        &mut out,
        &mut records,
        fixed,
        down,
        QwenMoeExpertComponentKind::DownWeight,
        QwenMoeExpertComponentKind::DownScale,
        QwenMoeExpertComponentKind::DownBias,
    )?;
    let slot = ExpertSlotView::new(layer, expert, 0, fixed.expert_bytes, &out)?;
    FixedQ4ExpertSlotView::new(slot, fixed)?;
    let metadata = ExpertPackMetadata {
        layer,
        expert,
        packed_bytes: out.len() as u64,
        records,
    };
    Ok((out, metadata))
}

fn write_fixed_native_q4_component(
    out: &mut [u8],
    records: &mut Vec<ExpertPackRecord>,
    layout: QwenMoeQ4ExpertLayout,
    input: NativeQ4ExpertRecordInput,
    weight_kind: QwenMoeExpertComponentKind,
    scale_kind: QwenMoeExpertComponentKind,
    bias_kind: QwenMoeExpertComponentKind,
) -> Result<()> {
    let scale_bias_bytes = expert_scale_bias_dtype_size(&input.scale_bias_dtype)?;
    let scale_bias_len = input
        .groups
        .checked_mul(scale_bias_bytes)
        .context("fixed native q4 expert scale/bias byte length overflow")?;
    if input.scale_bytes.len() != scale_bias_len || input.bias_bytes.len() != scale_bias_len {
        bail!(
            "native q4 expert tensor {} scale/bias bytes {}/{} do not match {} groups of {} bytes",
            input.tensor,
            input.scale_bytes.len(),
            input.bias_bytes.len(),
            input.groups,
            scale_bias_bytes
        );
    }
    let weight = layout.component(weight_kind);
    let scale = layout.component(scale_kind);
    let bias = layout.component(bias_kind);
    if input.packed.len() != weight.bytes
        || input.scale_bytes.len() != scale.bytes
        || input.bias_bytes.len() != bias.bytes
    {
        bail!(
            "native q4 expert tensor {} byte lengths packed/scales/biases {}/{}/{} do not match fixed layout {}/{}/{}",
            input.tensor,
            input.packed.len(),
            input.scale_bytes.len(),
            input.bias_bytes.len(),
            weight.bytes,
            scale.bytes,
            bias.bytes
        );
    }
    out[weight.offset..weight.offset + weight.bytes].copy_from_slice(&input.packed);
    out[scale.offset..scale.offset + scale.bytes].copy_from_slice(&input.scale_bytes);
    out[bias.offset..bias.offset + bias.bytes].copy_from_slice(&input.bias_bytes);
    records.push(ExpertPackRecord {
        tensor: input.tensor,
        dtype: input.dtype,
        shape: input.shape,
        source_offsets: input.source_offsets,
        source_hash: input.source_hash,
        record_offset: weight.offset as u64,
        packed_bytes: input.packed.len() as u64,
        groups: input.groups,
        group_size: layout.group_size,
        scale_bias_dtype: input.scale_bias_dtype,
    });
    Ok(())
}

fn write_quantized_expert_record<W: Write + Seek>(
    out: &mut W,
    records: &mut Vec<ExpertPackRecord>,
    input: ExpertRecordInput,
) -> Result<()> {
    let packed = quantize_q4(&input.values, &input.shape, GROUP_SIZE).with_context(|| {
        format!(
            "failed to quantize decoded expert tensor {} into q4 groups",
            input.tensor
        )
    })?;
    let record_offset = out
        .stream_position()
        .context("failed to get expert record offset")?;
    out.write_all(&(input.tensor.len() as u32).to_le_bytes())?;
    out.write_all(input.tensor.as_bytes())?;
    out.write_all(&(packed.values.len() as u64).to_le_bytes())?;
    out.write_all(&(packed.scales.len() as u64).to_le_bytes())?;
    write_expert_scale_bias_vec_le(out, &packed.scales, EXPERT_PACK_SCALE_BIAS_DTYPE)?;
    write_expert_scale_bias_vec_le(out, &packed.biases, EXPERT_PACK_SCALE_BIAS_DTYPE)?;
    out.write_all(&packed.values)?;
    records.push(ExpertPackRecord {
        tensor: input.tensor,
        dtype: input.dtype,
        shape: input.shape,
        source_offsets: input.source_offsets,
        source_hash: input.source_hash,
        record_offset,
        packed_bytes: packed.values.len() as u64,
        groups: packed.scales.len(),
        group_size: GROUP_SIZE,
        scale_bias_dtype: EXPERT_PACK_SCALE_BIAS_DTYPE.to_string(),
    });
    Ok(())
}

fn write_native_q4_expert_record<W: Write + Seek>(
    out: &mut W,
    records: &mut Vec<ExpertPackRecord>,
    input: NativeQ4ExpertRecordInput,
) -> Result<()> {
    let scale_bias_bytes = expert_scale_bias_dtype_size(&input.scale_bias_dtype)?;
    let scale_bias_len = input
        .groups
        .checked_mul(scale_bias_bytes)
        .context("native q4 expert scale/bias byte length overflow")?;
    if input.scale_bytes.len() != scale_bias_len || input.bias_bytes.len() != scale_bias_len {
        bail!(
            "native q4 expert tensor {} scale/bias bytes {}/{} do not match {} groups of {} bytes",
            input.tensor,
            input.scale_bytes.len(),
            input.bias_bytes.len(),
            input.groups,
            scale_bias_bytes
        );
    }

    let record_offset = out
        .stream_position()
        .context("failed to get expert record offset")?;
    out.write_all(&(input.tensor.len() as u32).to_le_bytes())?;
    out.write_all(input.tensor.as_bytes())?;
    out.write_all(&(input.packed.len() as u64).to_le_bytes())?;
    out.write_all(&(input.groups as u64).to_le_bytes())?;
    out.write_all(&input.scale_bytes)?;
    out.write_all(&input.bias_bytes)?;
    out.write_all(&input.packed)?;
    records.push(ExpertPackRecord {
        tensor: input.tensor,
        dtype: input.dtype,
        shape: input.shape,
        source_offsets: input.source_offsets,
        source_hash: input.source_hash,
        record_offset,
        packed_bytes: input.packed.len() as u64,
        groups: input.groups,
        group_size: GROUP_SIZE,
        scale_bias_dtype: input.scale_bias_dtype,
    });
    Ok(())
}

fn write_expert_scale_bias_vec_le<W: Write>(
    out: &mut W,
    values: &[f32],
    dtype: &str,
) -> Result<()> {
    match dtype.to_ascii_uppercase().as_str() {
        EXPERT_SCALE_BIAS_DTYPE_F32 | "FLOAT32" | "FP32" => {
            for value in values {
                out.write_all(&value.to_le_bytes())?;
            }
        }
        EXPERT_SCALE_BIAS_DTYPE_BF16 | "BFLOAT16" => {
            for value in values {
                out.write_all(&f32_to_bf16_bits(*value).to_le_bytes())?;
            }
        }
        other => bail!("unsupported q4 scale/bias dtype {other}"),
    }
    Ok(())
}

fn expert_pack_records_match_expected(
    actual: &[ExpertPackRecord],
    expected: &[ExpectedExpertPackRecord],
) -> bool {
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(actual, expected)| {
            actual.tensor == expected.tensor
                && actual.dtype == expected.dtype
                && actual.shape == expected.shape
                && actual.source_offsets == expected.source_offsets
                && actual.source_hash.as_deref() == Some(expected.source_hash.as_str())
                && actual.packed_bytes == expected.packed_bytes
                && actual.groups == expected.groups
                && actual.group_size == expected.group_size
                && actual.scale_bias_dtype == expected.scale_bias_dtype
        })
}

pub(crate) fn rewrite_expert_layer_pack(
    root: &Path,
    layer: usize,
    experts: usize,
    storage_format: ExpertLayerStorageFormat,
    expected: &[ExpectedExpertPack],
    mut build: impl FnMut(usize) -> Result<(Vec<u8>, ExpertPackMetadata)>,
) -> Result<usize> {
    if expected.is_empty() {
        return Ok(0);
    }
    let slot_size = storage_format.slot_size(expected);
    let old_metadata = read_expert_layer_pack_metadata(root, layer)?;
    let layer_path = expert_layer_path(root, layer);
    let all_reusable = old_metadata.as_ref().is_some_and(|metadata| {
        expected.iter().all(|pack| {
            expert_layer_slot_is_reusable(&layer_path, metadata, storage_format, pack)
                .unwrap_or(false)
        })
    });
    if all_reusable {
        return Ok(expected.len());
    }

    let temp_path = temp_pack_path(&layer_path);
    let out = fs::File::create(&temp_path).with_context(|| {
        format!(
            "failed to create temporary packed expert layer {}",
            temp_path.display()
        )
    })?;
    let layer_size = (experts as u64)
        .checked_mul(slot_size)
        .context("expert layer file size overflow")?;
    out.set_len(layer_size)
        .with_context(|| format!("failed to preallocate {}", temp_path.display()))?;

    let mut packs = Vec::with_capacity(expected.len());
    let mut reused = 0usize;
    for pack in expected {
        let offset = expert_slot_offset(pack.expert, slot_size)?;
        let (packed, metadata) = if let Some(old_metadata) = old_metadata.as_ref()
            && expert_layer_slot_is_reusable(&layer_path, old_metadata, storage_format, pack)?
        {
            reused += 1;
            let metadata = old_metadata
                .pack_for(pack.expert)
                .cloned()
                .context("reusable expert metadata disappeared")?;
            let bytes = read_layer_slot_bytes(&layer_path, old_metadata, pack.expert, &metadata)?;
            (bytes, metadata)
        } else {
            build(pack.expert)?
        };
        if packed.len() as u64 > slot_size {
            bail!(
                "packed expert layer {layer} expert {} is {} bytes, exceeds slot size {slot_size}",
                pack.expert,
                packed.len()
            );
        }
        write_all_at_positioned(&out, &packed, offset).with_context(|| {
            format!(
                "failed to write layer {layer} expert {} into {}",
                pack.expert,
                temp_path.display()
            )
        })?;
        packs.push(ExpertPackMetadata {
            packed_bytes: packed.len() as u64,
            ..metadata
        });
    }

    finish_expert_pack_atomically(out, &temp_path, &layer_path)?;
    let metadata = storage_format.layer_metadata(layer, slot_size, experts, packs);
    write_expert_metadata_atomically(root, layer, &metadata)?;
    Ok(reused)
}

pub(crate) fn expert_layer_slot_is_reusable(
    layer_path: &Path,
    layer_metadata: &ExpertLayerPackMetadata,
    storage_format: ExpertLayerStorageFormat,
    expected: &ExpectedExpertPack,
) -> Result<bool> {
    if !storage_format.metadata_matches(layer_metadata)
        || layer_metadata.expert_size != storage_format.slot_size(std::slice::from_ref(expected))
            && !matches!(storage_format, ExpertLayerStorageFormat::Pbq4Import)
    {
        return Ok(false);
    }
    let Some(metadata) = layer_metadata.pack_for(expected.expert) else {
        return Ok(false);
    };
    if metadata.packed_bytes != expected.packed_bytes
        || !expert_pack_records_match_expected(&metadata.records, &expected.records)
    {
        return Ok(false);
    }
    if metadata
        .records
        .iter()
        .any(|record| record.source_hash.is_none())
    {
        return Ok(false);
    }
    if metadata.packed_bytes > layer_metadata.expert_size {
        return Ok(false);
    }
    let file = match fs::File::open(layer_path) {
        Ok(file) => file,
        Err(_) => return Ok(false),
    };
    let file_len = file.metadata()?.len();
    let end = expert_slot_end(
        expected.expert,
        layer_metadata.expert_size,
        metadata.packed_bytes,
    )?;
    if file_len < end {
        return Ok(false);
    }
    if matches!(storage_format, ExpertLayerStorageFormat::Pbq4Import) {
        let offset = expert_slot_offset(expected.expert, layer_metadata.expert_size)?;
        let mut magic = vec![0u8; PBQ4_EXPERT_MAGIC.len()];
        if read_exact_at_positioned(&file, &mut magic, offset).is_err() {
            return Ok(false);
        }
        return Ok(magic == PBQ4_EXPERT_MAGIC);
    }
    Ok(metadata.packed_bytes == layer_metadata.expert_size)
}

pub(crate) fn read_layer_slot_bytes(
    layer_path: &Path,
    layer_metadata: &ExpertLayerPackMetadata,
    expert: usize,
    metadata: &ExpertPackMetadata,
) -> Result<Vec<u8>> {
    let file = fs::File::open(layer_path)
        .with_context(|| format!("failed to open expert layer {}", layer_path.display()))?;
    let offset = expert_slot_offset(expert, layer_metadata.expert_size)?;
    let len =
        usize::try_from(metadata.packed_bytes).context("expert pack length does not fit usize")?;
    let mut bytes = vec![0u8; len];
    read_exact_at_positioned(&file, &mut bytes, offset)?;
    Ok(bytes)
}

pub(crate) fn write_all_at_positioned(file: &fs::File, buf: &[u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        let mut written = 0usize;
        while written < buf.len() {
            let n = file.write_at(
                &buf[written..],
                offset
                    .checked_add(written as u64)
                    .context("positioned write offset overflow")?,
            )?;
            if n == 0 {
                bail!("short positioned write");
            }
            written += n;
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let mut file = file.try_clone().context("failed to clone file for write")?;
        file.seek(std::io::SeekFrom::Start(offset))?;
        file.write_all(buf)?;
        Ok(())
    }
}

pub(crate) fn temp_pack_path(path: &Path) -> PathBuf {
    let suffix = format!("tmp-{}-{:?}", std::process::id(), thread::current().id());
    let extension = match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => format!("{ext}.{suffix}"),
        None => suffix,
    };
    path.with_extension(extension)
}

pub(crate) fn finish_expert_pack_atomically(
    mut out: fs::File,
    temp_path: &Path,
    final_path: &Path,
) -> Result<()> {
    out.flush()
        .with_context(|| format!("failed to flush {}", temp_path.display()))?;
    out.sync_all()
        .with_context(|| format!("failed to sync {}", temp_path.display()))?;
    drop(out);
    fs::rename(temp_path, final_path).with_context(|| {
        format!(
            "failed to atomically move {} to {}",
            temp_path.display(),
            final_path.display()
        )
    })?;
    Ok(())
}

pub(crate) fn write_expert_metadata_atomically(
    root: &Path,
    layer: usize,
    metadata: &ExpertLayerPackMetadata,
) -> Result<()> {
    let path = expert_layer_metadata_path(root, layer);
    let temp_path = temp_pack_path(&path);
    let bytes = serde_json::to_vec_pretty(metadata).context("failed to encode expert metadata")?;
    {
        let mut out = fs::File::create(&temp_path).with_context(|| {
            format!(
                "failed to create temporary expert metadata {}",
                temp_path.display()
            )
        })?;
        out.write_all(&bytes)
            .with_context(|| format!("failed to write {}", temp_path.display()))?;
        out.flush()
            .with_context(|| format!("failed to flush {}", temp_path.display()))?;
        out.sync_all()
            .with_context(|| format!("failed to sync {}", temp_path.display()))?;
    }
    fs::rename(&temp_path, &path).with_context(|| {
        format!(
            "failed to atomically move {} to {}",
            temp_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

pub(crate) fn cleanup_stale_expert_temp_files(experts_dir: &Path) -> Result<usize> {
    if !experts_dir.is_dir() {
        return Ok(0);
    }
    let mut deleted = 0usize;
    for entry in fs::read_dir(experts_dir)
        .with_context(|| format!("failed to read {}", experts_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type()?.is_file() {
            continue;
        }
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if file_name.contains(".tmp-") {
            fs::remove_file(&path)
                .with_context(|| format!("failed to delete stale temp file {}", path.display()))?;
            deleted += 1;
        }
    }
    Ok(deleted)
}

pub(crate) fn rewrite_pbq4_layer_to_fixed_q4(
    root: &Path,
    layer: usize,
    experts: usize,
    spec: FixedQ4ExpertSlotSpec,
) -> Result<bool> {
    let Some(old_metadata) = read_expert_layer_pack_metadata(root, layer)? else {
        return Ok(false);
    };
    if old_metadata.format == FIXED_Q4_EXPERT_LAYER_FORMAT_V1
        && old_metadata.expert_size == spec.layout.expert_bytes as u64
    {
        return Ok(false);
    }
    if old_metadata.experts < experts || old_metadata.expert_size == 0 {
        return Ok(false);
    }
    let layer_path = expert_layer_path(root, layer);
    let old_file = match fs::File::open(&layer_path) {
        Ok(file) => file,
        Err(_) => return Ok(false),
    };
    if !pbq4_layer_looks_fixed_q4_compatible(&old_file, &old_metadata, spec)? {
        return Ok(false);
    }

    let temp_path = temp_pack_path(&layer_path);
    let out = fs::File::create(&temp_path).with_context(|| {
        format!(
            "failed to create temporary fixed Q4 expert layer {}",
            temp_path.display()
        )
    })?;
    let fixed_slot_size = spec.layout.expert_bytes as u64;
    out.set_len(
        (experts as u64)
            .checked_mul(fixed_slot_size)
            .context("fixed Q4 expert layer file size overflow")?,
    )
    .with_context(|| format!("failed to preallocate {}", temp_path.display()))?;

    let mut packs = Vec::with_capacity(experts);
    for expert in 0..experts {
        let metadata = old_metadata
            .pack_for(expert)
            .cloned()
            .with_context(|| format!("expert layer {layer} has no metadata for expert {expert}"))?;
        let bytes = read_layer_slot_bytes(&layer_path, &old_metadata, expert, &metadata)?;
        if !bytes.starts_with(PBQ4_EXPERT_MAGIC) {
            bail!("expert layer {layer} expert {expert} is already not PBQ4");
        }
        let records = parse_pbq4_expert_pack(&bytes, Some(&metadata))?;
        let (fixed_bytes, fixed_metadata) =
            fixed_q4_pack_from_pbq4_records(layer, expert, spec, &records)?;
        write_all_at_positioned(
            &out,
            &fixed_bytes,
            expert_slot_offset(expert, fixed_slot_size)?,
        )
        .with_context(|| {
            format!(
                "failed to write fixed Q4 layer {layer} expert {expert} into {}",
                temp_path.display()
            )
        })?;
        packs.push(fixed_metadata);
    }

    finish_expert_pack_atomically(out, &temp_path, &layer_path)?;
    let metadata = ExpertLayerPackMetadata::new_fixed_q4(layer, fixed_slot_size, experts, packs);
    write_expert_metadata_atomically(root, layer, &metadata)?;
    Ok(true)
}

fn pbq4_layer_looks_fixed_q4_compatible(
    file: &fs::File,
    metadata: &ExpertLayerPackMetadata,
    spec: FixedQ4ExpertSlotSpec,
) -> Result<bool> {
    let Some(first) = metadata.packs.first() else {
        return Ok(false);
    };
    if first.packed_bytes > metadata.expert_size {
        return Ok(false);
    }
    let offset = expert_slot_offset(first.expert, metadata.expert_size)?;
    let mut bytes = vec![
        0u8;
        usize::try_from(first.packed_bytes)
            .context("expert pack length does not fit usize")?
    ];
    read_exact_at_positioned(file, &mut bytes, offset)?;
    if !bytes.starts_with(PBQ4_EXPERT_MAGIC) {
        return Ok(false);
    }
    let records = parse_pbq4_expert_pack(&bytes, Some(first))?;
    Ok(fixed_q4_pack_from_pbq4_records(metadata.layer, first.expert, spec, &records).is_ok())
}

#[cfg(test)]
pub(crate) fn expert_pack_is_complete(root: &Path, layer: usize, expert: usize) -> bool {
    let path = expert_layer_path(root, layer);
    let Ok(file) = fs::File::open(&path) else {
        return false;
    };
    let Ok(Some(layer_metadata)) = read_expert_layer_pack_metadata(root, layer) else {
        return false;
    };
    let Some(_metadata) = layer_metadata.pack_for(expert) else {
        return false;
    };
    let Ok(offset) = expert_slot_offset(expert, layer_metadata.expert_size) else {
        return false;
    };
    let mut magic = vec![0u8; PBQ4_EXPERT_MAGIC.len()];
    read_exact_at_positioned(&file, &mut magic, offset).is_ok() && magic == PBQ4_EXPERT_MAGIC
}

pub(crate) fn first_missing_expert_pack_for_shape(
    experts_dir: &Path,
    layers: usize,
    experts: usize,
) -> Result<Option<PathBuf>> {
    for layer in 0..layers {
        let path = expert_layer_path(experts_dir, layer);
        if !path.is_file() {
            return Ok(Some(path));
        }
        let metadata_path = expert_layer_metadata_path(experts_dir, layer);
        let Some(metadata) = read_expert_layer_pack_metadata(experts_dir, layer)? else {
            return Ok(Some(metadata_path));
        };
        if metadata.experts < experts {
            return Ok(Some(metadata_path));
        }
        let file_len = fs::metadata(&path)
            .with_context(|| format!("failed to stat {}", path.display()))?
            .len();
        let required = (experts as u64)
            .checked_mul(metadata.expert_size)
            .context("expert layer size overflow")?;
        if file_len < required {
            return Ok(Some(path));
        }
        for expert in 0..experts {
            if metadata.pack_for(expert).is_none() {
                return Ok(Some(metadata_path));
            }
        }
    }
    Ok(None)
}

pub(crate) fn validate_expert_pack_metadata(
    path: &Path,
    metadata: &ExpertPackMetadata,
    layer: usize,
    expert: usize,
) -> Result<()> {
    if metadata.layer != layer || metadata.expert != expert {
        bail!(
            "expert metadata {} describes layer {} expert {}, expected layer {layer} expert {expert}",
            path.display(),
            metadata.layer,
            metadata.expert
        );
    }
    Ok(())
}

pub(crate) fn expert_layer_path(root: &Path, layer: usize) -> PathBuf {
    root.join(format!("layer_{layer:02}.bin"))
}

pub(crate) fn expert_layer_metadata_path(root: &Path, layer: usize) -> PathBuf {
    root.join(format!("layer_{layer:02}.json"))
}

#[cfg(test)]
pub(crate) fn read_expert_pack_metadata(
    root: &Path,
    layer: usize,
    expert: usize,
) -> Result<Option<ExpertPackMetadata>> {
    let Some(metadata) = read_expert_layer_pack_metadata(root, layer)? else {
        return Ok(None);
    };
    Ok(metadata.pack_for(expert).cloned())
}

pub(crate) fn read_expert_layer_pack_metadata(
    root: &Path,
    layer: usize,
) -> Result<Option<ExpertLayerPackMetadata>> {
    let path = expert_layer_metadata_path(root, layer);
    if !path.is_file() {
        return Ok(None);
    }
    let metadata: ExpertLayerPackMetadata = serde_json::from_slice(
        &fs::read(&path)
            .with_context(|| format!("failed to read expert metadata {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse expert metadata {}", path.display()))?;
    metadata.validate(&path, layer)?;
    Ok(Some(metadata))
}

#[derive(Debug)]
pub(crate) struct ExpertLayerReader {
    path: PathBuf,
    file: fs::File,
    metadata: ExpertLayerPackMetadata,
    slot_spec: ExpertSlotSpec,
    buffer_pool: ReusableExpertBytePool,
}

#[derive(Debug, Clone)]
pub(crate) struct ExpertSlotStore {
    root: PathBuf,
    slot_spec: ExpertSlotSpec,
    buffer_pool: ReusableExpertBytePool,
    layers: Arc<Mutex<BTreeMap<usize, Arc<ExpertLayerReader>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertStorageLayout {
    FixedQ4,
    FixedBf16,
    FixedF16,
}

impl ExpertStorageLayout {
    fn quantization(self) -> ExpertQuantization {
        match self {
            Self::FixedQ4 => ExpertQuantization::FourBitProduction,
            Self::FixedBf16 => ExpertQuantization::Bf16,
            Self::FixedF16 => ExpertQuantization::F16,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::FixedQ4 => "fixed-Q4",
            Self::FixedBf16 => "fixed-BF16",
            Self::FixedF16 => "fixed-F16",
        }
    }
}

fn validate_requested_expert_storage(
    root: &Path,
    resolved_layout: ExpertStorageLayout,
    requested_quantization: ExpertQuantization,
) -> Result<()> {
    if resolved_layout.quantization() != requested_quantization {
        bail!(
            "FlashMoe unsupported expert storage policy: requested {} but cache metadata in {} resolves {} expert slots",
            requested_quantization.as_str(),
            root.display(),
            resolved_layout.as_str()
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpertStoreExecutionDescriptor {
    pub(crate) layout: ExpertStorageLayout,
    pub(crate) slot_spec: ExpertSlotSpec,
    pub(crate) layers: usize,
    pub(crate) experts_per_layer: usize,
}

#[derive(Debug)]
pub(crate) struct ResolvedExpertSlotStore {
    pub(crate) store: ExpertSlotStore,
    pub(crate) descriptor: ExpertStoreExecutionDescriptor,
    pub(crate) upgraded_pbq4_layers: usize,
}

#[derive(Default)]
pub(crate) struct ExpertReadWorkerPool {
    workers: Vec<thread::JoinHandle<()>>,
    senders: Vec<mpsc::Sender<ExpertReadJob>>,
    next_worker: usize,
}

impl std::fmt::Debug for ExpertReadWorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertReadWorkerPool")
            .field("workers", &self.workers.len())
            .field("next_worker", &self.next_worker)
            .finish()
    }
}

impl ExpertReadWorkerPool {
    pub(crate) fn submit_read(
        &mut self,
        id: u64,
        expert: usize,
        reader: Arc<ExpertLayerReader>,
        plan: ExpertReadPlan,
        warm: bool,
        issued_at: Instant,
    ) -> Result<mpsc::Receiver<ExpertRawReadResponse>> {
        if self.senders.is_empty() {
            self.ensure_workers(1);
        }
        let (tx, rx) = mpsc::channel();
        let worker = self.next_worker % self.senders.len();
        self.next_worker = self.next_worker.wrapping_add(1);
        self.senders[worker]
            .send(ExpertReadJob {
                id,
                expert,
                reader,
                plan,
                warm,
                issued_at,
                tx,
            })
            .context("failed to submit expert read to I/O worker")?;
        Ok(rx)
    }

    pub(crate) fn ensure_workers(&mut self, workers: usize) {
        while self.workers.len() < workers {
            let (tx, rx) = mpsc::channel::<ExpertReadJob>();
            let handle = thread::spawn(move || {
                let mut scratch = ReusableExpertBuffer::default();
                while let Ok(job) = rx.recv() {
                    let started_at = Instant::now();
                    let queue_latency = started_at.saturating_duration_since(job.issued_at);
                    let bytes_read = job.plan.packed_len as u64;
                    let result = job
                        .reader
                        .read_prepared_into(job.expert, job.plan, &mut scratch);
                    let (result, read_latency, read_path) = match result {
                        Ok(raw) => {
                            let read_latency = raw.read_latency;
                            let read_path = raw.read_path;
                            (Ok(raw), read_latency, read_path)
                        }
                        Err(error) => (
                            Err(error),
                            started_at.elapsed(),
                            FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
                        ),
                    };
                    let _ = job.tx.send(ExpertRawReadResponse {
                        id: job.id,
                        queue_latency,
                        read_path,
                        read_latency,
                        bytes_read,
                        warm: job.warm,
                        result,
                    });
                }
            });
            self.senders.push(tx);
            self.workers.push(handle);
        }
    }

    #[cfg(test)]
    pub(crate) fn worker_count(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for ExpertReadWorkerPool {
    fn drop(&mut self) {
        self.senders.clear();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

struct ExpertReadJob {
    id: u64,
    expert: usize,
    reader: Arc<ExpertLayerReader>,
    plan: ExpertReadPlan,
    warm: bool,
    issued_at: Instant,
    tx: mpsc::Sender<ExpertRawReadResponse>,
}

#[derive(Debug)]
pub(crate) struct ExpertRawReadResponse {
    pub(crate) id: u64,
    pub(crate) queue_latency: Duration,
    pub(crate) read_path: ExpertReadPath,
    pub(crate) read_latency: Duration,
    pub(crate) bytes_read: u64,
    pub(crate) warm: bool,
    pub(crate) result: Result<ExpertRawRead>,
}

fn resolve_fixed_dense_metadata_dtype(
    metadata: &ExpertLayerPackMetadata,
) -> Result<DenseExpertDtype> {
    let mut resolved: Option<DenseExpertDtype> = None;
    for record in metadata.packs.iter().flat_map(|pack| &pack.records) {
        let dtype = DenseExpertDtype::from_metadata_dtype(&record.dtype).with_context(|| {
            format!(
                "FlashMoe unsupported fixed dense expert dtype {} in layer {} tensor {}",
                record.dtype, metadata.layer, record.tensor
            )
        })?;
        if let Some(existing) = resolved
            && existing != dtype
        {
            bail!(
                "FlashMoe unsupported fixed dense expert storage at layer {}: metadata mixes {} and {} tensors",
                metadata.layer,
                existing.as_str(),
                dtype.as_str()
            );
        }
        resolved = Some(dtype);
    }
    resolved.with_context(|| {
        format!(
            "FlashMoe unsupported fixed dense expert storage at layer {}: metadata declares no BF16/F16 tensor records",
            metadata.layer
        )
    })
}

impl ExpertSlotStore {
    #[cfg(test)]
    pub(crate) fn open(root: PathBuf) -> Result<Self> {
        Self::open_with_fixed_q4(root, FixedQ4ExpertSlotSpec::qwen35_a17b()?)
    }

    pub(crate) fn resolve_from_metadata(
        root: PathBuf,
        layout: &QwenMoeModelLayout,
        requested_quantization: ExpertQuantization,
    ) -> Result<ResolvedExpertSlotStore> {
        if !root.is_dir() {
            bail!("expert store {} does not exist", root.display());
        }
        let metadata = read_expert_layer_pack_metadata(&root, 0)?.with_context(|| {
            format!(
                "FlashMoe unsupported expert storage: layer 0 metadata is missing from {}",
                root.display()
            )
        })?;
        let resolved_layout = match metadata.format.as_str() {
            FIXED_Q4_EXPERT_LAYER_FORMAT_V1
            | PBQ4_EXPERT_LAYER_FORMAT_V1
            | PBQ4_EXPERT_LAYER_FORMAT_V2 => ExpertStorageLayout::FixedQ4,
            FIXED_DENSE_EXPERT_LAYER_FORMAT_V1 => {
                match resolve_fixed_dense_metadata_dtype(&metadata)? {
                    DenseExpertDtype::Bf16 => ExpertStorageLayout::FixedBf16,
                    DenseExpertDtype::F16 => ExpertStorageLayout::FixedF16,
                }
            }
            format => {
                bail!("FlashMoe unsupported expert storage format {format} in layer 0 metadata")
            }
        };
        validate_requested_expert_storage(&root, resolved_layout, requested_quantization)?;
        let (slot_spec, upgraded_pbq4_layers) = match metadata.format.as_str() {
            FIXED_Q4_EXPERT_LAYER_FORMAT_V1 => (
                ExpertSlotSpec::from_model_layout(layout, resolved_layout)?,
                0,
            ),
            FIXED_DENSE_EXPERT_LAYER_FORMAT_V1 => (
                ExpertSlotSpec::from_model_layout(layout, resolved_layout)?,
                0,
            ),
            PBQ4_EXPERT_LAYER_FORMAT_V1 | PBQ4_EXPERT_LAYER_FORMAT_V2 => {
                let slot_spec =
                    ExpertSlotSpec::from_model_layout(layout, ExpertStorageLayout::FixedQ4)?;
                let spec = slot_spec
                    .fixed_q4()
                    .expect("fixed-Q4 storage resolves Q4 spec");
                let mut upgraded = 0usize;
                for layer in 0..layout.layers {
                    if rewrite_pbq4_layer_to_fixed_q4(&root, layer, layout.experts_per_layer, spec)
                        .with_context(|| {
                            format!("failed to upgrade layer {layer} expert cache to fixed Q4")
                        })?
                    {
                        upgraded += 1;
                    }
                }
                (slot_spec, upgraded)
            }
            _ => unreachable!("expert storage format was validated before slot resolution"),
        };
        let store = Self::open_with_slot_spec(root, slot_spec)?;
        let descriptor =
            store.resolve_execution_descriptor(layout.layers, layout.experts_per_layer)?;
        Ok(ResolvedExpertSlotStore {
            store,
            descriptor,
            upgraded_pbq4_layers,
        })
    }

    #[cfg(test)]
    pub(crate) fn open_with_fixed_q4(
        root: PathBuf,
        fixed_q4: FixedQ4ExpertSlotSpec,
    ) -> Result<Self> {
        Self::open_with_slot_spec(root, ExpertSlotSpec::FixedQ4(fixed_q4))
    }

    #[cfg(test)]
    pub(crate) fn open_with_fixed_dense(
        root: PathBuf,
        fixed_dense: FixedDenseExpertSlotSpec,
    ) -> Result<Self> {
        Self::open_with_slot_spec(root, ExpertSlotSpec::FixedDense(fixed_dense))
    }

    pub(crate) fn open_with_slot_spec(root: PathBuf, slot_spec: ExpertSlotSpec) -> Result<Self> {
        if !root.is_dir() {
            bail!("expert store {} does not exist", root.display());
        }
        Ok(Self {
            root,
            slot_spec,
            buffer_pool: Arc::new(Mutex::new(Vec::new())),
            layers: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    pub(crate) fn resolve_execution_descriptor(
        &self,
        layers: usize,
        experts_per_layer: usize,
    ) -> Result<ExpertStoreExecutionDescriptor> {
        if layers == 0 || experts_per_layer == 0 {
            bail!(
                "FlashMoe expert storage resolution requires non-zero layers and experts, layers={layers}, experts_per_layer={experts_per_layer}"
            );
        }
        let slot_bytes = self.slot_spec.expert_bytes();
        let storage_layout = self.slot_spec.storage_layout();
        let expected_format = self.slot_spec.metadata_format();
        let expected_layer_bytes = (slot_bytes as u64)
            .checked_mul(experts_per_layer as u64)
            .context("fixed expert layer byte length overflow")?;
        for layer in 0..layers {
            let metadata = read_expert_layer_pack_metadata(&self.root, layer)?.with_context(|| {
                format!(
                    "FlashMoe unsupported {storage_layout:?} expert storage: layer {layer} metadata is missing"
                )
            })?;
            if metadata.format != expected_format {
                bail!(
                    "FlashMoe unsupported expert storage at layer {layer}: format {} is import compatibility only; expected {}",
                    metadata.format,
                    expected_format
                );
            }
            if let Some(expected) = self.slot_spec.fixed_dense() {
                let actual = resolve_fixed_dense_metadata_dtype(&metadata)?;
                if actual != expected.dtype {
                    bail!(
                        "FlashMoe unsupported fixed dense expert storage at layer {layer}: metadata declares {}, resolved graph requires {}",
                        actual.as_str(),
                        expected.dtype.as_str()
                    );
                }
            }
            if metadata.expert_size != slot_bytes as u64 {
                bail!(
                    "FlashMoe {storage_layout:?} expert storage layer {layer} has slot size {}, expected {}",
                    metadata.expert_size,
                    slot_bytes
                );
            }
            if metadata.experts != experts_per_layer || metadata.packs.len() != experts_per_layer {
                bail!(
                    "FlashMoe {storage_layout:?} expert storage layer {layer} declares {} experts and {} records, expected {experts_per_layer}",
                    metadata.experts,
                    metadata.packs.len()
                );
            }
            for expert in 0..experts_per_layer {
                let pack = metadata.pack_for(expert).with_context(|| {
                    format!(
                        "FlashMoe {storage_layout:?} expert storage layer {layer} is missing expert {expert}"
                    )
                })?;
                if pack.packed_bytes != metadata.expert_size {
                    bail!(
                        "FlashMoe {storage_layout:?} expert storage layer {layer} expert {expert} has {} bytes, expected whole-slot payload {}",
                        pack.packed_bytes,
                        metadata.expert_size
                    );
                }
            }
            let path = expert_layer_path(&self.root, layer);
            let actual_layer_bytes = fs::metadata(&path)
                .with_context(|| format!("failed to stat expert layer {}", path.display()))?
                .len();
            if actual_layer_bytes != expected_layer_bytes {
                bail!(
                    "FlashMoe {storage_layout:?} expert storage layer {layer} has file size {actual_layer_bytes}, expected {expected_layer_bytes}"
                );
            }
        }
        Ok(ExpertStoreExecutionDescriptor {
            layout: storage_layout,
            slot_spec: self.slot_spec,
            layers,
            experts_per_layer,
        })
    }

    #[cfg(test)]
    pub(crate) fn read_many_raw(
        &self,
        layer: usize,
        experts: &[usize],
    ) -> Result<Vec<ExpertRawRead>> {
        let reader = self.layer_reader(layer)?;
        let mut scratch = ReusableExpertBuffer::default();
        let mut out = Vec::with_capacity(experts.len());
        for &expert in experts {
            let plan = reader.prepare_read(expert)?;
            out.push(reader.read_prepared_into(expert, plan, &mut scratch)?);
        }
        Ok(out)
    }

    pub(crate) fn layer_reader(&self, layer: usize) -> Result<Arc<ExpertLayerReader>> {
        if let Some(reader) = self
            .layers
            .lock()
            .expect("expert layer cache poisoned")
            .get(&layer)
            .cloned()
        {
            return Ok(reader);
        }

        let reader = Arc::new(ExpertLayerReader::open(
            &self.root,
            layer,
            self.slot_spec,
            Arc::clone(&self.buffer_pool),
        )?);
        let mut layers = self.layers.lock().expect("expert layer cache poisoned");
        Ok(layers.entry(layer).or_insert_with(|| reader).clone())
    }
}

impl ExpertLayerReader {
    pub(crate) fn open(
        root: &Path,
        layer: usize,
        slot_spec: ExpertSlotSpec,
        buffer_pool: ReusableExpertBytePool,
    ) -> Result<Self> {
        let path = expert_layer_path(root, layer);
        let metadata = read_expert_layer_pack_metadata(root, layer)?.with_context(|| {
            format!(
                "failed to read expert layer metadata {}",
                expert_layer_metadata_path(root, layer).display()
            )
        })?;
        let file = fs::File::open(&path)
            .with_context(|| format!("failed to open expert layer {}", path.display()))?;
        Ok(Self {
            path,
            file,
            metadata,
            slot_spec,
            buffer_pool,
        })
    }

    pub(crate) fn prepare_read(&self, expert: usize) -> Result<ExpertReadPlan> {
        let metadata = self.metadata.pack_for(expert).cloned().with_context(|| {
            format!(
                "expert layer {} has no metadata for expert {expert}",
                self.metadata.layer
            )
        })?;
        if metadata.packed_bytes > self.metadata.expert_size {
            bail!(
                "expert layer {} expert {expert} metadata length {} exceeds slot size {}",
                self.metadata.layer,
                metadata.packed_bytes,
                self.metadata.expert_size
            );
        }
        let offset = expert_slot_offset(expert, self.metadata.expert_size)?;
        let packed_len = usize::try_from(metadata.packed_bytes)
            .context("expert pack length does not fit usize")?;
        let slot_capacity = usize::try_from(self.metadata.expert_size)
            .context("expert layer slot size does not fit usize")?;
        Ok(ExpertReadPlan {
            #[cfg(test)]
            metadata,
            offset,
            packed_len,
            slot_capacity,
        })
    }

    pub(crate) fn read_prepared_into(
        &self,
        expert: usize,
        plan: ExpertReadPlan,
        scratch: &mut ReusableExpertBuffer,
    ) -> Result<ExpertRawRead> {
        if scratch.capacity() < plan.slot_capacity
            && let Some(bytes) = take_reusable_expert_bytes(&self.buffer_pool, plan.slot_capacity)
        {
            let previous = scratch.adopt_buffer(bytes);
            recycle_reusable_expert_bytes(
                &self.buffer_pool,
                previous,
                self.slot_spec.expert_bytes(),
            );
        }
        let payload = scratch.prepare_payload(plan.slot_capacity, plan.packed_len)?;
        let read_started = Instant::now();
        read_exact_at_positioned(&self.file, payload, plan.offset).with_context(|| {
            format!(
                "failed to read expert {expert} from {}",
                self.path.display()
            )
        })?;
        let read_latency = read_started.elapsed();
        let slot =
            scratch.slot_view(self.metadata.layer, expert, plan.offset, plan.slot_capacity)?;
        let descriptor = slot.descriptor();
        let payload = if slot.payload().starts_with(PBQ4_EXPERT_MAGIC) {
            ExpertRawPayload::Pbq4(scratch.take_payload())
        } else {
            match self.slot_spec {
                ExpertSlotSpec::FixedQ4(spec) => {
                    FixedQ4ExpertSlotView::new(slot, spec.layout).with_context(|| {
                        format!(
                            "expert {} is neither a PBQ4 pack nor a fixed Q4 slot matching the model layout",
                            self.path.display()
                        )
                    })?;
                    ExpertRawPayload::FixedQ4(FixedQ4ExpertPayload::from_whole_slot(
                        spec,
                        scratch.take_payload(),
                        Some(Arc::clone(&self.buffer_pool)),
                    )?)
                }
                ExpertSlotSpec::FixedDense(spec) => {
                    ExpertRawPayload::FixedDense(FixedDenseExpertPayload::from_whole_slot(
                        spec,
                        scratch.take_payload(),
                        Some(Arc::clone(&self.buffer_pool)),
                    )?)
                }
            }
        };
        Ok(ExpertRawRead {
            layer: self.metadata.layer,
            expert,
            slot: descriptor,
            #[cfg(test)]
            metadata: plan.metadata,
            payload,
            read_latency,
            read_path: FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ExpertReadPlan {
    #[cfg(test)]
    pub(crate) metadata: ExpertPackMetadata,
    pub(crate) offset: u64,
    pub(crate) packed_len: usize,
    pub(crate) slot_capacity: usize,
}

#[derive(Debug)]
pub(crate) struct ExpertRawRead {
    pub(crate) layer: usize,
    pub(crate) expert: usize,
    pub(crate) slot: ExpertSlotDescriptor,
    #[cfg(test)]
    pub(crate) metadata: ExpertPackMetadata,
    pub(crate) payload: ExpertRawPayload,
    pub(crate) read_latency: Duration,
    pub(crate) read_path: ExpertReadPath,
}

#[derive(Debug)]
pub(crate) enum ExpertRawPayload {
    Pbq4(Vec<u8>),
    FixedQ4(FixedQ4ExpertPayload),
    FixedDense(FixedDenseExpertPayload),
}

pub(crate) fn expert_slot_offset(expert: usize, expert_size: u64) -> Result<u64> {
    (expert as u64)
        .checked_mul(expert_size)
        .context("expert slot offset overflow")
}

pub(crate) fn expert_slot_end(expert: usize, expert_size: u64, packed_bytes: u64) -> Result<u64> {
    expert_slot_offset(expert, expert_size)?
        .checked_add(packed_bytes)
        .context("expert slot end overflow")
}

pub(crate) fn read_exact_at_positioned(file: &fs::File, buf: &mut [u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        let mut read_total = 0usize;
        while read_total < buf.len() {
            let n = file.read_at(
                &mut buf[read_total..],
                offset
                    .checked_add(read_total as u64)
                    .context("positioned read offset overflow")?,
            )?;
            if n == 0 {
                bail!("short positioned read");
            }
            read_total += n;
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let mut file = file.try_clone().context("failed to clone file for read")?;
        file.seek(std::io::SeekFrom::Start(offset))?;
        file.read_exact(buf)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FixedQ4ExpertSlotSpec {
    pub(crate) layout: QwenMoeQ4ExpertLayout,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
}

impl FixedQ4ExpertSlotSpec {
    pub(crate) fn new(
        layout: QwenMoeQ4ExpertLayout,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        layout.validate()?;
        if hidden_size == 0 || intermediate_size == 0 {
            bail!(
                "fixed Q4 expert slot spec requires non-zero dimensions, hidden_size={hidden_size}, intermediate_size={intermediate_size}"
            );
        }
        Ok(Self {
            layout,
            hidden_size,
            intermediate_size,
        })
    }

    #[cfg(test)]
    pub(crate) fn qwen35_a17b() -> Result<Self> {
        Self::new(QwenMoeQ4ExpertLayout::qwen35_a17b(), HIDDEN_DIM, 1024)
    }

    pub(crate) fn from_model_layout(layout: &QwenMoeModelLayout) -> Result<Self> {
        let q4_layout = QwenMoeQ4ExpertLayout::fixed_bf16(
            layout.hidden_size,
            layout.moe_intermediate_size,
            GROUP_SIZE,
        )
        .with_context(|| {
            format!(
                "FlashMoe unsupported {:?} fixed-Q4 expert storage dimensions",
                layout.family
            )
        })?;
        Self::new(q4_layout, layout.hidden_size, layout.moe_intermediate_size)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseExpertDtype {
    Bf16,
    F16,
}

impl DenseExpertDtype {
    fn from_metadata_dtype(dtype: &str) -> Option<Self> {
        match dtype.to_ascii_uppercase().as_str() {
            "BF16" | "BFLOAT16" => Some(Self::Bf16),
            "F16" | "FLOAT16" | "FP16" => Some(Self::F16),
            _ => None,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F16 => "F16",
        }
    }

    pub(crate) const fn element_size(self) -> usize {
        2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseExpertProjectionSpec {
    pub(crate) offset: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FixedDenseExpertSlotSpec {
    pub(crate) dtype: DenseExpertDtype,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) gate: DenseExpertProjectionSpec,
    pub(crate) up: DenseExpertProjectionSpec,
    pub(crate) down: DenseExpertProjectionSpec,
    pub(crate) expert_bytes: usize,
}

impl FixedDenseExpertSlotSpec {
    pub(crate) fn new(
        dtype: DenseExpertDtype,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        if hidden_size == 0 || intermediate_size == 0 {
            bail!(
                "fixed {} expert slot spec requires non-zero dimensions, hidden_size={hidden_size}, intermediate_size={intermediate_size}",
                dtype.as_str()
            );
        }
        let projection_bytes = |rows: usize, cols: usize| {
            rows.checked_mul(cols)
                .and_then(|values| values.checked_mul(dtype.element_size()))
                .context("fixed dense expert projection byte length overflow")
        };
        let aligned = |value: usize| {
            value
                .checked_add(EXPERT_COMPONENT_ALIGNMENT - 1)
                .map(|value| value / EXPERT_COMPONENT_ALIGNMENT * EXPERT_COMPONENT_ALIGNMENT)
                .context("fixed dense expert component alignment overflow")
        };
        let gate_bytes = projection_bytes(intermediate_size, hidden_size)?;
        let down_bytes = projection_bytes(hidden_size, intermediate_size)?;
        let gate = DenseExpertProjectionSpec {
            offset: 0,
            rows: intermediate_size,
            cols: hidden_size,
            bytes: gate_bytes,
        };
        let up = DenseExpertProjectionSpec {
            offset: aligned(
                gate.offset
                    .checked_add(gate.bytes)
                    .context("fixed dense expert gate component end overflow")?,
            )?,
            rows: intermediate_size,
            cols: hidden_size,
            bytes: gate_bytes,
        };
        let down = DenseExpertProjectionSpec {
            offset: aligned(
                up.offset
                    .checked_add(up.bytes)
                    .context("fixed dense expert up component end overflow")?,
            )?,
            rows: hidden_size,
            cols: intermediate_size,
            bytes: down_bytes,
        };
        let expert_bytes = aligned(
            down.offset
                .checked_add(down.bytes)
                .context("fixed dense expert down component end overflow")?,
        )?;
        Ok(Self {
            dtype,
            hidden_size,
            intermediate_size,
            gate,
            up,
            down,
            expert_bytes,
        })
    }

    pub(crate) fn from_model_layout(
        layout: &QwenMoeModelLayout,
        dtype: DenseExpertDtype,
    ) -> Result<Self> {
        Self::new(dtype, layout.hidden_size, layout.moe_intermediate_size).with_context(|| {
            format!(
                "FlashMoe unsupported {:?} fixed-{} expert storage dimensions",
                layout.family,
                dtype.as_str()
            )
        })
    }

    pub(crate) const fn projection(
        self,
        projection: ExpertMlpProjection,
    ) -> DenseExpertProjectionSpec {
        match projection {
            ExpertMlpProjection::Gate => self.gate,
            ExpertMlpProjection::Up => self.up,
            ExpertMlpProjection::Down => self.down,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertSlotSpec {
    FixedQ4(FixedQ4ExpertSlotSpec),
    FixedDense(FixedDenseExpertSlotSpec),
}

impl ExpertSlotSpec {
    pub(crate) fn from_model_layout(
        layout: &QwenMoeModelLayout,
        storage: ExpertStorageLayout,
    ) -> Result<Self> {
        match storage {
            ExpertStorageLayout::FixedQ4 => {
                FixedQ4ExpertSlotSpec::from_model_layout(layout).map(Self::FixedQ4)
            }
            ExpertStorageLayout::FixedBf16 => {
                FixedDenseExpertSlotSpec::from_model_layout(layout, DenseExpertDtype::Bf16)
                    .map(Self::FixedDense)
            }
            ExpertStorageLayout::FixedF16 => {
                FixedDenseExpertSlotSpec::from_model_layout(layout, DenseExpertDtype::F16)
                    .map(Self::FixedDense)
            }
        }
    }

    pub(crate) const fn storage_layout(self) -> ExpertStorageLayout {
        match self {
            Self::FixedQ4(_) => ExpertStorageLayout::FixedQ4,
            Self::FixedDense(spec) => match spec.dtype {
                DenseExpertDtype::Bf16 => ExpertStorageLayout::FixedBf16,
                DenseExpertDtype::F16 => ExpertStorageLayout::FixedF16,
            },
        }
    }

    pub(crate) const fn expert_bytes(self) -> usize {
        match self {
            Self::FixedQ4(spec) => spec.layout.expert_bytes,
            Self::FixedDense(spec) => spec.expert_bytes,
        }
    }

    pub(crate) const fn metadata_format(self) -> &'static str {
        match self {
            Self::FixedQ4(_) => FIXED_Q4_EXPERT_LAYER_FORMAT_V1,
            Self::FixedDense(_) => FIXED_DENSE_EXPERT_LAYER_FORMAT_V1,
        }
    }

    pub(crate) const fn fixed_q4(self) -> Option<FixedQ4ExpertSlotSpec> {
        match self {
            Self::FixedQ4(spec) => Some(spec),
            Self::FixedDense(_) => None,
        }
    }

    pub(crate) const fn fixed_dense(self) -> Option<FixedDenseExpertSlotSpec> {
        match self {
            Self::FixedDense(spec) => Some(spec),
            Self::FixedQ4(_) => None,
        }
    }
}

#[derive(Debug)]
pub(crate) struct FixedQ4ExpertPayload {
    pub(crate) spec: FixedQ4ExpertSlotSpec,
    pub(crate) bytes: Vec<u8>,
    pub(crate) recycle_pool: Option<ReusableExpertBytePool>,
}

impl Clone for FixedQ4ExpertPayload {
    fn clone(&self) -> Self {
        Self {
            spec: self.spec,
            bytes: self.bytes.clone(),
            recycle_pool: None,
        }
    }
}

impl PartialEq for FixedQ4ExpertPayload {
    fn eq(&self, other: &Self) -> bool {
        self.spec == other.spec && self.bytes == other.bytes
    }
}

impl Drop for FixedQ4ExpertPayload {
    fn drop(&mut self) {
        if let Some(pool) = &self.recycle_pool {
            recycle_reusable_expert_bytes(
                pool,
                std::mem::take(&mut self.bytes),
                self.spec.layout.expert_bytes,
            );
        }
    }
}

impl FixedQ4ExpertPayload {
    pub(crate) fn from_whole_slot(
        spec: FixedQ4ExpertSlotSpec,
        bytes: Vec<u8>,
        recycle_pool: Option<ReusableExpertBytePool>,
    ) -> Result<Self> {
        if bytes.len() < spec.layout.expert_bytes {
            bail!(
                "fixed Q4 expert whole-slot payload length {} is shorter than layout size {}",
                bytes.len(),
                spec.layout.expert_bytes
            );
        }
        Ok(Self {
            spec,
            bytes,
            recycle_pool,
        })
    }

    #[cfg(test)]
    pub(crate) fn payload_prefix(&self, max_len: usize) -> &[u8] {
        &self.bytes[..self.bytes.len().min(max_len)]
    }

    pub(crate) fn component(&self, kind: QwenMoeExpertComponentKind) -> &[u8] {
        let component = self.spec.layout.component(kind);
        &self.bytes[component.offset..component.offset + component.bytes]
    }

    fn component_source(
        &self,
        weight_kind: QwenMoeExpertComponentKind,
        scale_kind: QwenMoeExpertComponentKind,
        bias_kind: QwenMoeExpertComponentKind,
    ) -> Q4MatvecSource<'_> {
        Q4MatvecSource {
            bytes: &self.bytes,
            packed_offset: self.spec.layout.component(weight_kind).offset,
            scale_offset: self.spec.layout.component(scale_kind).offset,
            bias_offset: self.spec.layout.component(bias_kind).offset,
        }
    }

    #[cfg(test)]
    fn decoded_scales_biases(
        &self,
        scale_kind: QwenMoeExpertComponentKind,
        bias_kind: QwenMoeExpertComponentKind,
        needed_groups: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let scales = decode_fixed_q4_bf16_component_bytes(self.component(scale_kind))
            .with_context(|| format!("failed to decode fixed Q4 {scale_kind:?} scales"))?;
        let biases = decode_fixed_q4_bf16_component_bytes(self.component(bias_kind))
            .with_context(|| format!("failed to decode fixed Q4 {bias_kind:?} biases"))?;
        if scales.len() < needed_groups || biases.len() < needed_groups {
            bail!(
                "fixed Q4 expert scale/bias payload is shorter than projection requires: scales={}, biases={}, required={needed_groups}",
                scales.len(),
                biases.len()
            );
        }
        Ok((scales, biases))
    }

    #[cfg(test)]
    pub(crate) fn project_cpu(
        &self,
        projection: ExpertMlpProjection,
        input: &[f32],
        output_width: usize,
    ) -> Result<Vec<f32>> {
        let payload = self
            .matvec_payload(projection, input.len(), output_width)
            .context("fixed Q4 projection metadata is incompatible with input/output shape")?;
        let (scale_kind, bias_kind) = projection.scale_bias_kinds();
        let (owned_scales, owned_biases);
        let (scales, biases) = if payload.scales.len() >= payload.scale_bias_groups
            && payload.biases.len() >= payload.scale_bias_groups
        {
            (payload.scales, payload.biases)
        } else {
            (owned_scales, owned_biases) =
                self.decoded_scales_biases(scale_kind, bias_kind, payload.scale_bias_groups)?;
            (
                &owned_scales[..payload.scale_bias_groups],
                &owned_biases[..payload.scale_bias_groups],
            )
        };
        q4_fma_matvec_with_group_size(
            payload.packed,
            &input[..payload.cols],
            scales,
            biases,
            payload.rows,
            payload.cols,
            payload.group_size,
        )
    }

    pub(crate) fn matvec_payload(
        &self,
        projection: ExpertMlpProjection,
        input_len: usize,
        output_width: usize,
    ) -> Option<Q4MatvecPayload<'_>> {
        if input_len == 0 || output_width == 0 {
            return None;
        }
        let (rows, cols) = match projection {
            ExpertMlpProjection::Gate | ExpertMlpProjection::Up => (
                self.spec.intermediate_size.min(output_width).max(1),
                self.spec.hidden_size.min(input_len).max(1),
            ),
            ExpertMlpProjection::Down => (
                self.spec.hidden_size.min(output_width).max(1),
                self.spec.intermediate_size.min(input_len).max(1),
            ),
        };
        let groups_per_row = cols.div_ceil(self.spec.layout.group_size).max(1);
        let needed_groups = rows.checked_mul(groups_per_row)?;
        let needed_packed = rows.checked_mul(cols.div_ceil(2))?;
        let (packed, scale_bytes, bias_bytes, source) = match projection {
            ExpertMlpProjection::Gate => (
                self.component(QwenMoeExpertComponentKind::GateWeight),
                self.component(QwenMoeExpertComponentKind::GateScale),
                self.component(QwenMoeExpertComponentKind::GateBias),
                self.component_source(
                    QwenMoeExpertComponentKind::GateWeight,
                    QwenMoeExpertComponentKind::GateScale,
                    QwenMoeExpertComponentKind::GateBias,
                ),
            ),
            ExpertMlpProjection::Up => (
                self.component(QwenMoeExpertComponentKind::UpWeight),
                self.component(QwenMoeExpertComponentKind::UpScale),
                self.component(QwenMoeExpertComponentKind::UpBias),
                self.component_source(
                    QwenMoeExpertComponentKind::UpWeight,
                    QwenMoeExpertComponentKind::UpScale,
                    QwenMoeExpertComponentKind::UpBias,
                ),
            ),
            ExpertMlpProjection::Down => (
                self.component(QwenMoeExpertComponentKind::DownWeight),
                self.component(QwenMoeExpertComponentKind::DownScale),
                self.component(QwenMoeExpertComponentKind::DownBias),
                self.component_source(
                    QwenMoeExpertComponentKind::DownWeight,
                    QwenMoeExpertComponentKind::DownScale,
                    QwenMoeExpertComponentKind::DownBias,
                ),
            ),
        };
        if packed.len() < needed_packed
            || scale_bytes.len() < needed_groups * 2
            || bias_bytes.len() < needed_groups * 2
        {
            return None;
        }
        Some(Q4MatvecPayload {
            rows,
            cols,
            group_size: self.spec.layout.group_size,
            packed: &packed[..needed_packed],
            #[cfg(test)]
            scales: &[],
            #[cfg(test)]
            biases: &[],
            scale_bias_groups: needed_groups,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16,
            scale_bytes: &scale_bytes[..needed_groups * 2],
            bias_bytes: &bias_bytes[..needed_groups * 2],
            source: Some(source),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseMatvecSource<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) byte_offset: usize,
}

impl DenseMatvecSource<'_> {
    pub(crate) fn same_buffer(self, other: Self) -> bool {
        self.bytes.as_ptr() == other.bytes.as_ptr() && self.bytes.len() == other.bytes.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseMatvecPayload<'a> {
    pub(crate) dtype: DenseExpertDtype,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) source: DenseMatvecSource<'a>,
}

#[derive(Debug)]
pub(crate) struct FixedDenseExpertPayload {
    pub(crate) spec: FixedDenseExpertSlotSpec,
    pub(crate) bytes: Vec<u8>,
    pub(crate) recycle_pool: Option<ReusableExpertBytePool>,
}

impl Clone for FixedDenseExpertPayload {
    fn clone(&self) -> Self {
        Self {
            spec: self.spec,
            bytes: self.bytes.clone(),
            recycle_pool: None,
        }
    }
}

impl PartialEq for FixedDenseExpertPayload {
    fn eq(&self, other: &Self) -> bool {
        self.spec == other.spec && self.bytes == other.bytes
    }
}

impl Eq for FixedDenseExpertPayload {}

impl Drop for FixedDenseExpertPayload {
    fn drop(&mut self) {
        if let Some(pool) = &self.recycle_pool {
            recycle_reusable_expert_bytes(
                pool,
                std::mem::take(&mut self.bytes),
                self.spec.expert_bytes,
            );
        }
    }
}

impl FixedDenseExpertPayload {
    pub(crate) fn from_whole_slot(
        spec: FixedDenseExpertSlotSpec,
        bytes: Vec<u8>,
        recycle_pool: Option<ReusableExpertBytePool>,
    ) -> Result<Self> {
        if bytes.len() < spec.expert_bytes {
            bail!(
                "fixed {} expert whole-slot payload length {} is shorter than layout size {}",
                spec.dtype.as_str(),
                bytes.len(),
                spec.expert_bytes
            );
        }
        Ok(Self {
            spec,
            bytes,
            recycle_pool,
        })
    }

    pub(crate) fn matvec_payload(
        &self,
        projection: ExpertMlpProjection,
        input_len: usize,
        output_width: usize,
    ) -> Result<DenseMatvecPayload<'_>> {
        let component = self.spec.projection(projection);
        if input_len != component.cols || output_width != component.rows {
            bail!(
                "fixed {} expert {projection:?} projection requires input/output {}/{}, got {input_len}/{output_width}",
                self.spec.dtype.as_str(),
                component.cols,
                component.rows
            );
        }
        let end = component
            .offset
            .checked_add(component.bytes)
            .context("fixed dense expert component end overflow")?;
        if end > self.bytes.len() {
            bail!(
                "fixed {} expert {projection:?} component range {}..{} exceeds whole-slot payload {}",
                self.spec.dtype.as_str(),
                component.offset,
                end,
                self.bytes.len()
            );
        }
        Ok(DenseMatvecPayload {
            dtype: self.spec.dtype,
            rows: component.rows,
            cols: component.cols,
            source: DenseMatvecSource {
                bytes: &self.bytes,
                byte_offset: component.offset,
            },
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertMlpProjection {
    Gate,
    Up,
    Down,
}

impl ExpertMlpProjection {
    #[cfg(test)]
    fn scale_bias_kinds(self) -> (QwenMoeExpertComponentKind, QwenMoeExpertComponentKind) {
        match self {
            ExpertMlpProjection::Gate => (
                QwenMoeExpertComponentKind::GateScale,
                QwenMoeExpertComponentKind::GateBias,
            ),
            ExpertMlpProjection::Up => (
                QwenMoeExpertComponentKind::UpScale,
                QwenMoeExpertComponentKind::UpBias,
            ),
            ExpertMlpProjection::Down => (
                QwenMoeExpertComponentKind::DownScale,
                QwenMoeExpertComponentKind::DownBias,
            ),
        }
    }
}

#[cfg(test)]
pub(crate) fn fixed_q4_payload_from_pbq4_records(
    layer: usize,
    expert: usize,
    spec: FixedQ4ExpertSlotSpec,
    records: &[PackedExpertTensor],
    recycle_pool: Option<ReusableExpertBytePool>,
) -> Result<FixedQ4ExpertPayload> {
    let (bytes, _) = fixed_q4_pack_from_pbq4_records(layer, expert, spec, records)?;
    FixedQ4ExpertPayload::from_whole_slot(spec, bytes, recycle_pool)
}

pub(crate) fn fixed_q4_pack_from_pbq4_records(
    layer: usize,
    expert: usize,
    spec: FixedQ4ExpertSlotSpec,
    records: &[PackedExpertTensor],
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    spec.layout.validate()?;
    let gate = pbq4_record_by_suffix(records, "gate_proj.weight")?;
    let up = pbq4_record_by_suffix(records, "up_proj.weight")?;
    let down = pbq4_record_by_suffix(records, "down_proj.weight")?;

    let mut bytes = vec![0u8; spec.layout.expert_bytes];
    let mut metadata_records = Vec::with_capacity(3);
    copy_pbq4_record_to_fixed_q4_component(
        &mut bytes,
        &mut metadata_records,
        spec,
        gate,
        &[spec.intermediate_size, spec.hidden_size],
        QwenMoeExpertComponentKind::GateWeight,
        QwenMoeExpertComponentKind::GateScale,
        QwenMoeExpertComponentKind::GateBias,
    )?;
    copy_pbq4_record_to_fixed_q4_component(
        &mut bytes,
        &mut metadata_records,
        spec,
        up,
        &[spec.intermediate_size, spec.hidden_size],
        QwenMoeExpertComponentKind::UpWeight,
        QwenMoeExpertComponentKind::UpScale,
        QwenMoeExpertComponentKind::UpBias,
    )?;
    copy_pbq4_record_to_fixed_q4_component(
        &mut bytes,
        &mut metadata_records,
        spec,
        down,
        &[spec.hidden_size, spec.intermediate_size],
        QwenMoeExpertComponentKind::DownWeight,
        QwenMoeExpertComponentKind::DownScale,
        QwenMoeExpertComponentKind::DownBias,
    )?;

    let slot = ExpertSlotView::new(layer, expert, 0, spec.layout.expert_bytes, &bytes)?;
    FixedQ4ExpertSlotView::new(slot, spec.layout)?;
    Ok((
        bytes,
        ExpertPackMetadata {
            layer,
            expert,
            packed_bytes: spec.layout.expert_bytes as u64,
            records: metadata_records,
        },
    ))
}

fn pbq4_record_by_suffix<'a>(
    records: &'a [PackedExpertTensor],
    suffix: &str,
) -> Result<&'a PackedExpertTensor> {
    let matches: Vec<&PackedExpertTensor> = records
        .iter()
        .filter(|record| record.name.ends_with(suffix))
        .collect();
    match matches.as_slice() {
        [record] => Ok(*record),
        [] => bail!("PBQ4 expert pack is missing {suffix}"),
        _ => bail!("PBQ4 expert pack has duplicate {suffix} records"),
    }
}

fn copy_pbq4_record_to_fixed_q4_component(
    out: &mut [u8],
    metadata_records: &mut Vec<ExpertPackRecord>,
    spec: FixedQ4ExpertSlotSpec,
    record: &PackedExpertTensor,
    expected_shape: &[usize],
    weight_kind: QwenMoeExpertComponentKind,
    scale_kind: QwenMoeExpertComponentKind,
    bias_kind: QwenMoeExpertComponentKind,
) -> Result<()> {
    if record.shape != expected_shape {
        bail!(
            "PBQ4 expert tensor {} has shape {:?}; expected {:?}",
            record.name,
            record.shape,
            expected_shape
        );
    }
    if record.group_size != spec.layout.group_size {
        bail!(
            "PBQ4 expert tensor {} has group size {}; expected {}",
            record.name,
            record.group_size,
            spec.layout.group_size
        );
    }

    let weight = spec.layout.component(weight_kind);
    let scale = spec.layout.component(scale_kind);
    let bias = spec.layout.component(bias_kind);
    if record.packed.len() != weight.bytes {
        bail!(
            "PBQ4 expert tensor {} packed bytes {}; expected {}",
            record.name,
            record.packed.len(),
            weight.bytes
        );
    }
    out[weight.offset..weight.offset + weight.bytes].copy_from_slice(&record.packed);

    let scale_bytes = fixed_q4_bf16_scale_bias_bytes(record, true, scale.bytes)?;
    let bias_bytes = fixed_q4_bf16_scale_bias_bytes(record, false, bias.bytes)?;
    out[scale.offset..scale.offset + scale.bytes].copy_from_slice(&scale_bytes);
    out[bias.offset..bias.offset + bias.bytes].copy_from_slice(&bias_bytes);
    metadata_records.push(ExpertPackRecord {
        tensor: record.name.clone(),
        dtype: record.dtype.clone(),
        shape: record.shape.clone(),
        source_offsets: record.source_offsets(),
        source_hash: record.source_hash.clone(),
        record_offset: weight.offset as u64,
        packed_bytes: weight.bytes as u64,
        groups: record.scales.len(),
        group_size: spec.layout.group_size,
        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
    });
    Ok(())
}

fn fixed_q4_bf16_scale_bias_bytes(
    record: &PackedExpertTensor,
    scales: bool,
    expected_bytes: usize,
) -> Result<Vec<u8>> {
    if !expected_bytes.is_multiple_of(2) {
        bail!(
            "fixed Q4 component for {} has odd scale/bias byte length {expected_bytes}",
            record.name
        );
    }
    let values = if scales {
        &record.scales
    } else {
        &record.biases
    };
    let raw = if scales {
        &record.scale_bytes
    } else {
        &record.bias_bytes
    };
    let groups = expected_bytes / 2;
    if values.len() != groups {
        bail!(
            "PBQ4 expert tensor {} scale/bias groups {}; expected {groups}",
            record.name,
            values.len()
        );
    }
    if record
        .scale_bias_dtype
        .eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16)
        || record.scale_bias_dtype.eq_ignore_ascii_case("BFLOAT16")
    {
        if raw.len() != expected_bytes {
            bail!(
                "PBQ4 expert tensor {} bf16 scale/bias bytes {}; expected {expected_bytes}",
                record.name,
                raw.len()
            );
        }
        return Ok(raw.clone());
    }
    let mut out = Vec::with_capacity(expected_bytes);
    for value in values {
        out.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
    }
    Ok(out)
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

#[derive(Debug, Clone)]
pub(crate) struct Q4MatvecPayload<'a> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) group_size: usize,
    pub(crate) packed: &'a [u8],
    #[cfg(test)]
    pub(crate) scales: &'a [f32],
    #[cfg(test)]
    pub(crate) biases: &'a [f32],
    pub(crate) scale_bias_groups: usize,
    pub(crate) scale_bias_dtype: &'a str,
    pub(crate) scale_bytes: &'a [u8],
    pub(crate) bias_bytes: &'a [u8],
    pub(crate) source: Option<Q4MatvecSource<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Q4MatvecSource<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) packed_offset: usize,
    pub(crate) scale_offset: usize,
    pub(crate) bias_offset: usize,
}

impl<'a> Q4MatvecSource<'a> {
    pub(crate) fn same_buffer(self, other: Self) -> bool {
        self.bytes.as_ptr() == other.bytes.as_ptr() && self.bytes.len() == other.bytes.len()
    }

    pub(crate) fn covers(self, payload: &Q4MatvecPayload<'_>) -> bool {
        self.packed_offset
            .checked_add(payload.packed.len())
            .is_some_and(|end| end <= self.bytes.len())
            && self
                .scale_offset
                .checked_add(payload.scale_bytes.len())
                .is_some_and(|end| end <= self.bytes.len())
            && self
                .bias_offset
                .checked_add(payload.bias_bytes.len())
                .is_some_and(|end| end <= self.bytes.len())
    }

    pub(crate) fn offsets_are_metal_aligned(self) -> bool {
        self.packed_offset % 4 == 0 && self.scale_offset % 4 == 0 && self.bias_offset % 4 == 0
    }
}

#[cfg(test)]
pub(crate) fn decode_fixed_q4_bf16_component_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    let chunks = bytes.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        bail!(
            "fixed Q4 bf16 component has odd byte length {}",
            bytes.len()
        );
    }
    Ok(chunks
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect())
}

pub fn take_reusable_expert_bytes(
    pool: &ReusableExpertBytePool,
    min_capacity: usize,
) -> Option<Vec<u8>> {
    let mut pool = pool.lock().expect("fixed Q4 expert byte pool poisoned");
    let index = pool
        .iter()
        .position(|bytes| bytes.capacity() >= min_capacity)?;
    Some(pool.swap_remove(index))
}

pub fn recycle_reusable_expert_bytes(
    pool: &ReusableExpertBytePool,
    mut bytes: Vec<u8>,
    min_capacity: usize,
) {
    if bytes.capacity() < min_capacity {
        return;
    }
    bytes.clear();
    let mut pool = pool.lock().expect("fixed Q4 expert byte pool poisoned");
    if pool.len() < FIXED_Q4_EXPERT_BUFFER_POOL_LIMIT {
        pool.push(bytes);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlotDescriptor {
    pub layer: usize,
    pub expert: usize,
    pub slot_offset: u64,
    pub slot_capacity: usize,
    pub payload_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlotView<'a> {
    descriptor: ExpertSlotDescriptor,
    payload: &'a [u8],
}

impl<'a> ExpertSlotView<'a> {
    pub fn new(
        layer: usize,
        expert: usize,
        slot_offset: u64,
        slot_capacity: usize,
        payload: &'a [u8],
    ) -> Result<Self> {
        if payload.len() > slot_capacity {
            bail!(
                "expert slot payload length {} exceeds slot capacity {}",
                payload.len(),
                slot_capacity
            );
        }
        Ok(Self {
            descriptor: ExpertSlotDescriptor {
                layer,
                expert,
                slot_offset,
                slot_capacity,
                payload_len: payload.len(),
            },
            payload,
        })
    }

    pub fn descriptor(&self) -> ExpertSlotDescriptor {
        self.descriptor
    }

    pub fn payload(&self) -> &'a [u8] {
        self.payload
    }

    pub fn payload_prefix(&self, max_len: usize) -> &'a [u8] {
        &self.payload[..self.payload.len().min(max_len)]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedQ4ExpertSlotView<'a> {
    slot: ExpertSlotView<'a>,
    layout: QwenMoeQ4ExpertLayout,
}

impl<'a> FixedQ4ExpertSlotView<'a> {
    pub fn new(slot: ExpertSlotView<'a>, layout: QwenMoeQ4ExpertLayout) -> Result<Self> {
        layout.validate()?;
        let payload_len = slot.payload().len();
        if payload_len < layout.expert_bytes {
            bail!(
                "fixed Q4 expert slot payload length {payload_len} is shorter than layout size {}",
                layout.expert_bytes
            );
        }
        Ok(Self { slot, layout })
    }

    pub fn descriptor(&self) -> ExpertSlotDescriptor {
        self.slot.descriptor()
    }

    pub fn layout(&self) -> QwenMoeQ4ExpertLayout {
        self.layout
    }

    pub fn payload(&self) -> &'a [u8] {
        self.slot.payload()
    }

    pub fn component(&self, kind: QwenMoeExpertComponentKind) -> &'a [u8] {
        let component = self.layout.component(kind);
        self.component_bytes(component)
    }

    fn component_bytes(&self, component: QwenMoeExpertComponentLayout) -> &'a [u8] {
        let start = component.offset;
        let end = start + component.bytes;
        &self.slot.payload()[start..end]
    }
}

#[derive(Debug, Default)]
pub struct ReusableExpertBuffer {
    bytes: Vec<u8>,
}

impl ReusableExpertBuffer {
    pub fn prepare_payload(
        &mut self,
        slot_capacity: usize,
        payload_len: usize,
    ) -> Result<&mut [u8]> {
        if payload_len > slot_capacity {
            bail!("expert payload length {payload_len} exceeds slot capacity {slot_capacity}");
        }
        if self.bytes.capacity() < slot_capacity {
            self.bytes
                .try_reserve_exact(slot_capacity - self.bytes.capacity())
                .context("failed to reserve reusable expert buffer")?;
        }
        self.bytes.resize(payload_len, 0);
        Ok(&mut self.bytes)
    }

    pub fn slot_view(
        &self,
        layer: usize,
        expert: usize,
        slot_offset: u64,
        slot_capacity: usize,
    ) -> Result<ExpertSlotView<'_>> {
        ExpertSlotView::new(layer, expert, slot_offset, slot_capacity, &self.bytes)
    }

    pub fn take_payload(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.bytes)
    }

    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    pub fn adopt_buffer(&mut self, mut bytes: Vec<u8>) -> Vec<u8> {
        bytes.clear();
        std::mem::replace(&mut self.bytes, bytes)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ExpertPackingPolicy<'a> {
    model: &'a str,
    experts_dir: &'a Path,
    quantization: ExpertQuantization,
}

impl<'a> ExpertPackingPolicy<'a> {
    pub(super) fn new(
        model: &'a str,
        experts_dir: &'a Path,
        quantization: ExpertQuantization,
    ) -> Self {
        Self {
            model,
            experts_dir,
            quantization,
        }
    }
}

pub(super) fn pack_expert_tensors(
    snapshot_dir: &Path,
    policy: ExpertPackingPolicy<'_>,
    expert_tensors: &[ExpertTensorRef],
    config: Option<&QwenModelConfig>,
) -> Result<()> {
    let mut by_layer: BTreeMap<usize, BTreeMap<usize, Vec<&ExpertTensorRef>>> = BTreeMap::new();
    let mut aggregate_by_layer: BTreeMap<usize, Vec<&ExpertTensorRef>> = BTreeMap::new();
    for tensor in expert_tensors {
        if let (Some(layer), Some(expert)) = (tensor.layer, tensor.expert) {
            by_layer
                .entry(layer)
                .or_default()
                .entry(expert)
                .or_default()
                .push(tensor);
        } else if let Some(layer) = tensor.layer
            && aggregate_expert_tensor_kind(&tensor.tensor).is_some()
        {
            aggregate_by_layer.entry(layer).or_default().push(tensor);
        }
    }

    let deleted_temps = cleanup_stale_expert_temp_files(policy.experts_dir)?;
    if deleted_temps > 0 {
        eprintln!(
            "deleted {deleted_temps} stale temporary expert pack file(s) from {}",
            policy.experts_dir.display()
        );
    }

    let aggregate_layers = aggregate_by_layer.len();
    if aggregate_layers > 0 {
        eprintln!("packing aggregate experts across {aggregate_layers} layer(s)");
    }
    let mut shard_cache = BTreeMap::<String, (memmap2::Mmap, SafetensorShard)>::new();
    for (layer_index, (layer, tensors)) in aggregate_by_layer.into_iter().enumerate() {
        pack_aggregate_expert_layer(
            snapshot_dir,
            policy,
            layer,
            layer_index + 1,
            aggregate_layers,
            &tensors,
            config,
        )?;
    }
    for (layer, experts) in by_layer {
        pack_direct_expert_layer(
            snapshot_dir,
            policy,
            layer,
            experts,
            config,
            &mut shard_cache,
        )?;
    }
    Ok(())
}

pub(super) fn fixed_dense_expert_slot_spec_for_pack(
    policy: ExpertPackingPolicy<'_>,
    config: Option<&QwenModelConfig>,
) -> Result<Option<FixedDenseExpertSlotSpec>> {
    let dtype = match policy.quantization {
        ExpertQuantization::FourBitProduction => return Ok(None),
        ExpertQuantization::Bf16 => DenseExpertDtype::Bf16,
        ExpertQuantization::F16 => DenseExpertDtype::F16,
    };
    let config = config.context("Qwen config is required for fixed dense expert packing")?;
    let layout = QwenMoeModelLayout::from_config(policy.model, config)?;
    FixedDenseExpertSlotSpec::from_model_layout(&layout, dtype).map(Some)
}

pub(super) fn pack_direct_expert_layer(
    snapshot_dir: &Path,
    policy: ExpertPackingPolicy<'_>,
    layer: usize,
    experts: BTreeMap<usize, Vec<&ExpertTensorRef>>,
    config: Option<&QwenModelConfig>,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
) -> Result<()> {
    let fixed_dense = fixed_dense_expert_slot_spec_for_pack(policy, config)?;
    let mut expected = Vec::with_capacity(experts.len());
    for (expert, tensors) in &experts {
        validate_expert_tensor_group(layer, *expert, tensors, config)?;
        expected.push(match fixed_dense {
            Some(spec) => expected_fixed_dense_expert_pack(
                snapshot_dir,
                shard_cache,
                layer,
                *expert,
                tensors,
                spec,
            )?,
            None => expected_expert_pack(snapshot_dir, shard_cache, *expert, tensors)?,
        });
    }
    let expert_count = layer_expert_count(config, &experts);
    rewrite_expert_layer_pack(
        policy.experts_dir,
        layer,
        expert_count,
        match fixed_dense {
            Some(spec) => ExpertLayerStorageFormat::FixedDense(spec),
            None => ExpertLayerStorageFormat::Pbq4Import,
        },
        &expected,
        |expert| {
            let tensors = experts
                .get(&expert)
                .with_context(|| format!("missing expert {expert} tensors for layer {layer}"))?;
            build_direct_expert_pack(
                snapshot_dir,
                shard_cache,
                layer,
                expert,
                tensors,
                fixed_dense,
            )
        },
    )?;
    Ok(())
}

pub(super) fn layer_expert_count(
    config: Option<&QwenModelConfig>,
    experts: &BTreeMap<usize, Vec<&ExpertTensorRef>>,
) -> usize {
    let declared = config.map(|config| config.experts()).unwrap_or(0);
    let observed = experts
        .keys()
        .next_back()
        .map(|expert| expert + 1)
        .unwrap_or(0);
    declared.max(observed).max(1)
}

pub(super) fn build_direct_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    tensors: &[&ExpertTensorRef],
    fixed_dense: Option<FixedDenseExpertSlotSpec>,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    if let Some(spec) = fixed_dense {
        let inputs = tensors
            .iter()
            .map(|tensor| {
                fixed_dense_expert_record_input(
                    snapshot_dir,
                    shard_cache,
                    tensor,
                    tensor.tensor.clone(),
                    tensor.shape.clone(),
                    0,
                    tensor.shape.iter().product(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        return build_fixed_dense_expert_pack(layer, expert, spec, inputs);
    }
    let mut inputs = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let dtype = tensor.dtype.as_deref().unwrap_or("unknown");
        let (values, source_offsets, source_hash) = decode_expert_tensor_range(
            snapshot_dir,
            shard_cache,
            tensor,
            0,
            tensor.shape.iter().product(),
        )?;
        inputs.push(ExpertRecordInput {
            tensor: tensor.tensor.clone(),
            dtype: dtype.to_string(),
            shape: tensor.shape.clone(),
            source_offsets,
            source_hash: Some(source_hash),
            values,
        });
    }
    build_expert_pack(layer, expert, inputs)
}

pub(super) fn pack_aggregate_expert_layer(
    snapshot_dir: &Path,
    policy: ExpertPackingPolicy<'_>,
    layer: usize,
    layer_index: usize,
    layer_total: usize,
    tensors: &[&ExpertTensorRef],
    config: Option<&QwenModelConfig>,
) -> Result<()> {
    let config = config.context("Qwen config is required to split aggregate expert tensors")?;
    let intermediate = config
        .moe_intermediate_size
        .or(config.intermediate_size)
        .context("Qwen config is missing moe_intermediate_size/intermediate_size for aggregate expert packing")?;
    let layout = AggregateExpertLayout::new(config.experts(), config.hidden_size, intermediate)?;

    let aggregate_tensors = aggregate_expert_tensors(tensors, layer, layout)?;
    let down = single_aggregate_expert_tensor(tensors, AggregateExpertTensorKind::Down, layer)?;
    validate_aggregate_expert_tensor_shape(
        down,
        &[layout.experts, layout.hidden, layout.intermediate],
        "down_proj",
    )?;

    eprintln!(
        "packing aggregate experts for layer {layer} ({layer_index}/{layer_total}): {} experts",
        layout.experts
    );
    let fixed_dense = fixed_dense_expert_slot_spec_for_pack(policy, Some(config))?;
    let fixed_native_q4 = fixed_native_q4_aggregate_layout(&aggregate_tensors, down, layout)?;
    let mut shard_cache = BTreeMap::<String, (memmap2::Mmap, SafetensorShard)>::new();
    let mut expected = Vec::with_capacity(layout.experts);
    for expert in 0..layout.experts {
        let records = match fixed_dense {
            Some(spec) => expected_fixed_dense_aggregate_expert_records(
                snapshot_dir,
                &mut shard_cache,
                layer,
                expert,
                &aggregate_tensors,
                down,
                layout,
                spec,
            )?,
            None => expected_aggregate_expert_records(
                snapshot_dir,
                &mut shard_cache,
                layer,
                expert,
                &aggregate_tensors,
                down,
                layout,
            )?,
        };
        let packed_bytes = match (fixed_dense, fixed_native_q4) {
            (Some(spec), _) => spec.expert_bytes as u64,
            (None, Some(fixed)) => fixed.expert_bytes as u64,
            (None, None) => pbq4_expert_pack_wire_size(&records)?,
        };
        expected.push(ExpectedExpertPack {
            expert,
            packed_bytes,
            records,
        });
    }
    let skipped =
        rewrite_expert_layer_pack(
            policy.experts_dir,
            layer,
            layout.experts,
            match (fixed_dense, fixed_native_q4) {
                (Some(spec), _) => ExpertLayerStorageFormat::FixedDense(spec),
                (None, Some(fixed)) => ExpertLayerStorageFormat::FixedQ4(
                    FixedQ4ExpertSlotSpec::new(fixed, layout.hidden, layout.intermediate)?,
                ),
                (None, None) => ExpertLayerStorageFormat::Pbq4Import,
            },
            &expected,
            |expert| {
                if let Some(spec) = fixed_dense {
                    build_fixed_dense_aggregate_expert_pack(
                        snapshot_dir,
                        &mut shard_cache,
                        layer,
                        expert,
                        &aggregate_tensors,
                        down,
                        layout,
                        spec,
                    )
                } else {
                    build_aggregate_expert_pack(
                        snapshot_dir,
                        &mut shard_cache,
                        layer,
                        expert,
                        &aggregate_tensors,
                        down,
                        layout,
                    )
                }
            },
        )?;
    eprintln!(
        "prepared aggregate experts for layer {layer} ({layer_index}/{layer_total}): {}/{} ({skipped} reused)",
        layout.experts, layout.experts,
    );
    Ok(())
}

pub(super) fn build_aggregate_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    if aggregate_native_q4_enabled(aggregate_tensors, down)? {
        return build_native_q4_aggregate_expert_pack(
            snapshot_dir,
            shard_cache,
            layer,
            expert,
            aggregate_tensors,
            down,
            layout,
        );
    }

    let mut inputs = Vec::with_capacity(3);
    let (gate_values, gate_offsets, gate_hash) = decode_expert_tensor_range(
        snapshot_dir,
        shard_cache,
        aggregate_tensors.gate.tensor,
        aggregate_tensors.gate.start(expert)?,
        layout.single_projection_values,
    )?;
    inputs.push(ExpertRecordInput {
        tensor: format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
        dtype: aggregate_tensors
            .gate
            .tensor
            .dtype
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        shape: vec![layout.intermediate, layout.hidden],
        source_offsets: gate_offsets,
        source_hash: Some(gate_hash),
        values: gate_values,
    });

    let (up_values, up_offsets, up_hash) = decode_expert_tensor_range(
        snapshot_dir,
        shard_cache,
        aggregate_tensors.up.tensor,
        aggregate_tensors.up.start(expert)?,
        layout.single_projection_values,
    )?;
    inputs.push(ExpertRecordInput {
        tensor: format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
        dtype: aggregate_tensors
            .up
            .tensor
            .dtype
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        shape: vec![layout.intermediate, layout.hidden],
        source_offsets: up_offsets,
        source_hash: Some(up_hash),
        values: up_values,
    });

    let down_base = expert
        .checked_mul(layout.down_expert_values)
        .context("aggregate down expert offset overflow")?;
    let (down_values, down_offsets, down_hash) = decode_expert_tensor_range(
        snapshot_dir,
        shard_cache,
        down,
        down_base,
        layout.down_expert_values,
    )?;
    inputs.push(ExpertRecordInput {
        tensor: format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
        dtype: down.dtype.clone().unwrap_or_else(|| "unknown".to_string()),
        shape: vec![layout.hidden, layout.intermediate],
        source_offsets: down_offsets,
        source_hash: Some(down_hash),
        values: down_values,
    });
    build_expert_pack(layer, expert, inputs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_fixed_dense_aggregate_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
    spec: FixedDenseExpertSlotSpec,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    let inputs = vec![
        fixed_dense_expert_record_input(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.gate.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.gate.start(expert)?,
            layout.single_projection_values,
        )?,
        fixed_dense_expert_record_input(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.up.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.up.start(expert)?,
            layout.single_projection_values,
        )?,
        fixed_dense_expert_record_input(
            snapshot_dir,
            shard_cache,
            down,
            format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
            vec![layout.hidden, layout.intermediate],
            expert
                .checked_mul(layout.down_expert_values)
                .context("aggregate down expert offset overflow")?,
            layout.down_expert_values,
        )?,
    ];
    build_fixed_dense_expert_pack(layer, expert, spec, inputs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn expected_fixed_dense_aggregate_expert_records(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
    spec: FixedDenseExpertSlotSpec,
) -> Result<Vec<ExpectedExpertPackRecord>> {
    let sources = [
        (
            aggregate_tensors.gate.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.gate.start(expert)?,
            layout.single_projection_values,
            ExpertMlpProjection::Gate,
        ),
        (
            aggregate_tensors.up.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.up.start(expert)?,
            layout.single_projection_values,
            ExpertMlpProjection::Up,
        ),
        (
            down,
            format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
            vec![layout.hidden, layout.intermediate],
            expert
                .checked_mul(layout.down_expert_values)
                .context("aggregate down expert offset overflow")?,
            layout.down_expert_values,
            ExpertMlpProjection::Down,
        ),
    ];
    sources
        .into_iter()
        .map(|(source, tensor, shape, offset, count, projection)| {
            let (source_offsets, source_hash) =
                expert_tensor_source_fingerprint(snapshot_dir, shard_cache, source, offset, count)?;
            let component = spec.projection(projection);
            Ok(ExpectedExpertPackRecord {
                tensor,
                dtype: source
                    .dtype
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                shape,
                source_offsets,
                source_hash,
                packed_bytes: component.bytes as u64,
                groups: 0,
                group_size: 0,
                scale_bias_dtype: spec.dtype.as_str().to_string(),
            })
        })
        .collect()
}

pub(super) fn expected_aggregate_expert_records(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
) -> Result<Vec<ExpectedExpertPackRecord>> {
    if aggregate_native_q4_enabled(aggregate_tensors, down)? {
        return expected_native_q4_aggregate_expert_records(
            snapshot_dir,
            shard_cache,
            layer,
            expert,
            aggregate_tensors,
            down,
            layout,
        );
    }

    let gate = expected_expert_pack_record(
        snapshot_dir,
        shard_cache,
        aggregate_tensors.gate.tensor,
        format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
        vec![layout.intermediate, layout.hidden],
        aggregate_tensors.gate.start(expert)?,
        layout.single_projection_values,
    )?;
    let up = expected_expert_pack_record(
        snapshot_dir,
        shard_cache,
        aggregate_tensors.up.tensor,
        format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
        vec![layout.intermediate, layout.hidden],
        aggregate_tensors.up.start(expert)?,
        layout.single_projection_values,
    )?;
    let down_base = expert
        .checked_mul(layout.down_expert_values)
        .context("aggregate down expert offset overflow")?;
    let down = expected_expert_pack_record(
        snapshot_dir,
        shard_cache,
        down,
        format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
        vec![layout.hidden, layout.intermediate],
        down_base,
        layout.down_expert_values,
    )?;
    Ok(vec![gate, up, down])
}

pub(super) fn build_native_q4_aggregate_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    let inputs = vec![
        native_q4_expert_record_input(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.gate.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.gate.start(expert)?,
            layout.single_projection_values,
        )?,
        native_q4_expert_record_input(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.up.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.up.start(expert)?,
            layout.single_projection_values,
        )?,
        native_q4_expert_record_input(
            snapshot_dir,
            shard_cache,
            down,
            format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
            vec![layout.hidden, layout.intermediate],
            expert
                .checked_mul(layout.down_expert_values)
                .context("aggregate down expert offset overflow")?,
            layout.down_expert_values,
        )?,
    ];
    if let Some(fixed) = fixed_native_q4_aggregate_layout(aggregate_tensors, down, layout)? {
        return build_fixed_native_q4_expert_pack(layer, expert, fixed, inputs);
    }
    build_native_q4_expert_pack(layer, expert, inputs)
}

pub(super) fn expected_native_q4_aggregate_expert_records(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
) -> Result<Vec<ExpectedExpertPackRecord>> {
    Ok(vec![
        expected_native_q4_expert_record(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.gate.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.gate.start(expert)?,
            layout.single_projection_values,
        )?,
        expected_native_q4_expert_record(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.up.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.up.start(expert)?,
            layout.single_projection_values,
        )?,
        expected_native_q4_expert_record(
            snapshot_dir,
            shard_cache,
            down,
            format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
            vec![layout.hidden, layout.intermediate],
            expert
                .checked_mul(layout.down_expert_values)
                .context("aggregate down expert offset overflow")?,
            layout.down_expert_values,
        )?,
    ])
}

pub(super) fn expected_native_q4_expert_record(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    source: &ExpertTensorRef,
    tensor: String,
    shape: Vec<usize>,
    element_offset: usize,
    element_count: usize,
) -> Result<ExpectedExpertPackRecord> {
    let input = native_q4_expert_record_input(
        snapshot_dir,
        shard_cache,
        source,
        tensor,
        shape,
        element_offset,
        element_count,
    )?;
    expected_native_q4_expert_record_from_input(input)
}

pub(super) fn expected_expert_pack_record(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    source: &ExpertTensorRef,
    tensor: String,
    shape: Vec<usize>,
    element_offset: usize,
    element_count: usize,
) -> Result<ExpectedExpertPackRecord> {
    let (source_offsets, source_hash) = expert_tensor_source_fingerprint(
        snapshot_dir,
        shard_cache,
        source,
        element_offset,
        element_count,
    )?;
    expected_expert_pack_record_from_source(
        tensor,
        source
            .dtype
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        shape,
        source_offsets,
        source_hash,
    )
}

pub(super) fn expected_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    expert: usize,
    tensors: &[&ExpertTensorRef],
) -> Result<ExpectedExpertPack> {
    let mut records = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let shape = tensor.shape.clone();
        let element_count = shape.iter().product();
        records.push(expected_expert_pack_record(
            snapshot_dir,
            shard_cache,
            tensor,
            tensor.tensor.clone(),
            shape,
            0,
            element_count,
        )?);
    }
    expected_expert_pack_from_records(expert, records)
}

pub(super) fn expected_fixed_dense_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    tensors: &[&ExpertTensorRef],
    spec: FixedDenseExpertSlotSpec,
) -> Result<ExpectedExpertPack> {
    let mut records = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let projection = if tensor.tensor.ends_with("gate_proj.weight") {
            ExpertMlpProjection::Gate
        } else if tensor.tensor.ends_with("up_proj.weight") {
            ExpertMlpProjection::Up
        } else if tensor.tensor.ends_with("down_proj.weight") {
            ExpertMlpProjection::Down
        } else {
            bail!(
                "fixed {} expert pack layer {layer} expert {expert} has unknown tensor {}",
                spec.dtype.as_str(),
                tensor.tensor
            );
        };
        let component = spec.projection(projection);
        if tensor.shape != [component.rows, component.cols] {
            bail!(
                "fixed {} expert tensor {} has shape {:?}, expected [{}, {}]",
                spec.dtype.as_str(),
                tensor.tensor,
                tensor.shape,
                component.rows,
                component.cols
            );
        }
        let (source_offsets, source_hash) = expert_tensor_source_fingerprint(
            snapshot_dir,
            shard_cache,
            tensor,
            0,
            tensor.shape.iter().product(),
        )?;
        records.push(ExpectedExpertPackRecord {
            tensor: tensor.tensor.clone(),
            dtype: tensor
                .dtype
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            shape: tensor.shape.clone(),
            source_offsets,
            source_hash,
            packed_bytes: component.bytes as u64,
            groups: 0,
            group_size: 0,
            scale_bias_dtype: spec.dtype.as_str().to_string(),
        });
    }
    Ok(ExpectedExpertPack {
        expert,
        packed_bytes: spec.expert_bytes as u64,
        records,
    })
}

pub(super) fn decode_expert_tensor_range(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    tensor: &ExpertTensorRef,
    element_offset: usize,
    element_count: usize,
) -> Result<(Vec<f32>, [u64; 2], String)> {
    with_expert_tensor_raw_range(
        snapshot_dir,
        shard_cache,
        tensor,
        element_offset,
        element_count,
        |raw, source_offsets, dtype| {
            let values = decode_dense_tensor_f32(dtype, raw).with_context(|| {
                format!(
                    "failed to decode expert tensor {} as {dtype} before q4 quantization",
                    tensor.tensor
                )
            })?;
            Ok((values, source_offsets, sha256_hex(raw)))
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fixed_dense_expert_record_input(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    source: &ExpertTensorRef,
    tensor: String,
    shape: Vec<usize>,
    element_offset: usize,
    element_count: usize,
) -> Result<FixedDenseExpertRecordInput> {
    with_expert_tensor_raw_range(
        snapshot_dir,
        shard_cache,
        source,
        element_offset,
        element_count,
        |raw, source_offsets, dtype| {
            Ok(FixedDenseExpertRecordInput {
                tensor,
                dtype: dtype.to_string(),
                shape,
                source_offsets,
                source_hash: Some(sha256_hex(raw)),
                bytes: raw.to_vec(),
            })
        },
    )
}

pub(super) fn expert_tensor_source_fingerprint(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    tensor: &ExpertTensorRef,
    element_offset: usize,
    element_count: usize,
) -> Result<([u64; 2], String)> {
    with_expert_tensor_raw_range(
        snapshot_dir,
        shard_cache,
        tensor,
        element_offset,
        element_count,
        |raw, source_offsets, _| Ok((source_offsets, sha256_hex(raw))),
    )
}

pub(super) fn with_expert_tensor_raw_range<R>(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    tensor: &ExpertTensorRef,
    element_offset: usize,
    element_count: usize,
    read: impl FnOnce(&[u8], [u64; 2], &str) -> Result<R>,
) -> Result<R> {
    if !shard_cache.contains_key(&tensor.shard) {
        let shard_path = snapshot_dir.join(&tensor.shard);
        let file = fs::File::open(&shard_path)
            .with_context(|| format!("failed to open shard {}", shard_path.display()))?;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .with_context(|| format!("failed to memory-map {}", shard_path.display()))?
        };
        shard_cache.insert(
            tensor.shard.clone(),
            (mmap, parse_safetensors_header(&shard_path)?),
        );
    }
    let (bytes, shard) = shard_cache.get(&tensor.shard).expect("inserted above");
    let dtype = tensor.dtype.as_deref().unwrap_or("unknown");
    let [byte_start, byte_end] =
        expert_tensor_byte_range(tensor, dtype, element_offset, element_count)?;
    let abs_start = shard.data_start + byte_start;
    let abs_end = shard.data_start + byte_end;
    let raw = &bytes[abs_start as usize..abs_end as usize];
    read(raw, [byte_start, byte_end], dtype)
}

pub(super) fn native_q4_expert_record_input(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    source: &ExpertTensorRef,
    tensor: String,
    shape: Vec<usize>,
    element_offset: usize,
    element_count: usize,
) -> Result<NativeQ4ExpertRecordInput> {
    let q4_sources = source
        .q4_sources
        .as_ref()
        .with_context(|| format!("expert tensor {} is not native MLX Q4", source.tensor))?;
    let slice = native_q4_slice_byte_ranges(
        source,
        shape.as_slice(),
        &q4_sources.scale_bias_dtype,
        element_offset,
        element_count,
    )?;
    let source_offsets = source
        .source_offsets
        .with_context(|| format!("expert tensor {} is missing source offsets", source.tensor))?;
    let (packed, packed_offsets) = read_safetensor_source_byte_range(
        snapshot_dir,
        shard_cache,
        &source.shard,
        source_offsets,
        slice.packed_offset,
        slice.packed_bytes,
    )?;
    let (scale_bytes, _) = read_safetensor_source_byte_range(
        snapshot_dir,
        shard_cache,
        &q4_sources.scales_shard,
        q4_sources.scales_offsets,
        slice.scale_bias_offset,
        slice.scale_bias_bytes,
    )?;
    let (bias_bytes, _) = read_safetensor_source_byte_range(
        snapshot_dir,
        shard_cache,
        &q4_sources.biases_shard,
        q4_sources.biases_offsets,
        slice.scale_bias_offset,
        slice.scale_bias_bytes,
    )?;
    let source_hash = sha256_hex_parts(&[&packed, &scale_bytes, &bias_bytes]);
    Ok(NativeQ4ExpertRecordInput {
        tensor,
        dtype: source
            .dtype
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        shape,
        source_offsets: packed_offsets,
        source_hash: Some(source_hash),
        packed,
        scale_bytes,
        bias_bytes,
        groups: slice.groups,
        scale_bias_dtype: q4_sources.scale_bias_dtype.clone(),
    })
}

pub(super) fn read_safetensor_source_byte_range(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    shard_name: &str,
    data_offsets: [u64; 2],
    relative_offset: usize,
    byte_len: usize,
) -> Result<(Vec<u8>, [u64; 2])> {
    if !shard_cache.contains_key(shard_name) {
        let shard_path = snapshot_dir.join(shard_name);
        let file = fs::File::open(&shard_path)
            .with_context(|| format!("failed to open shard {}", shard_path.display()))?;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .with_context(|| format!("failed to memory-map {}", shard_path.display()))?
        };
        shard_cache.insert(
            shard_name.to_string(),
            (mmap, parse_safetensors_header(&shard_path)?),
        );
    }
    let (bytes, shard) = shard_cache.get(shard_name).expect("inserted above");
    let byte_start = data_offsets[0]
        .checked_add(relative_offset as u64)
        .context("safetensor source byte offset overflow")?;
    let byte_end = byte_start
        .checked_add(byte_len as u64)
        .context("safetensor source byte range overflow")?;
    if byte_end > data_offsets[1] {
        bail!(
            "safetensor source range {byte_start}..{byte_end} exceeds offsets {:?} in {shard_name}",
            data_offsets
        );
    }
    let abs_start = shard
        .data_start
        .checked_add(byte_start)
        .context("safetensor absolute byte offset overflow")?;
    let abs_end = shard
        .data_start
        .checked_add(byte_end)
        .context("safetensor absolute byte range overflow")?;
    Ok((
        bytes[abs_start as usize..abs_end as usize].to_vec(),
        [byte_start, byte_end],
    ))
}

pub(super) fn validate_expert_tensor_group(
    layer: usize,
    expert: usize,
    tensors: &[&ExpertTensorRef],
    config: Option<&QwenModelConfig>,
) -> Result<()> {
    let shape = if let Some(config) = config {
        let intermediate = config
            .moe_intermediate_size
            .or(config.intermediate_size)
            .context("Qwen config is missing moe_intermediate_size/intermediate_size for expert validation")?;
        Some(DirectExpertTensorShape::new(
            config.hidden_size,
            intermediate,
        )?)
    } else {
        None
    };
    validate_direct_expert_tensor_group(layer, expert, tensors, shape)
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

pub(super) fn sha256_hex_parts(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

#[cfg(test)]
pub(super) fn read_pbq4_expert_records(
    root: &Path,
    layer: usize,
    expert: usize,
) -> Result<Vec<PackedExpertTensor>> {
    let store = ExpertSlotStore::open(root.to_path_buf())?;
    let raw = store
        .read_many_raw(layer, &[expert])?
        .pop()
        .with_context(|| format!("expert layer {layer} returned no expert {expert}"))?;
    let ExpertRawPayload::Pbq4(bytes) = raw.payload else {
        bail!(
            "expert layer {layer} expert {expert} is fixed-slot execution storage, not PBQ4 import data"
        );
    };
    parse_pbq4_expert_pack(&bytes, Some(&raw.metadata))
}

#[cfg(test)]
pub(super) fn packed_expert_record_suffix<'a>(
    records: &'a [PackedExpertTensor],
    suffix: &str,
) -> Option<&'a PackedExpertTensor> {
    records.iter().find(|record| record.name.ends_with(suffix))
}

#[cfg(test)]
pub(super) fn project_packed_expert_record(
    record: &PackedExpertTensor,
    input: &[f32],
    output_width: usize,
) -> Result<Vec<f32>> {
    let payload = record
        .matvec_payload(input, output_width)
        .with_context(|| format!("PBQ4 record {} has no compatible Q4 payload", record.name))?;
    q4_fma_matvec_with_group_size(
        payload.packed,
        &input[..payload.cols],
        payload.scales,
        payload.biases,
        payload.rows,
        payload.cols,
        payload.group_size,
    )
    .with_context(|| format!("failed to project PBQ4 import record {}", record.name))
}

#[cfg(test)]
mod tests {
    use super::super::model_family::QwenModelConfig;
    use super::*;

    fn packing_layout_config() -> QwenModelConfig {
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
            torch_dtype: Some("bfloat16".to_string()),
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
        }
    }

    #[test]
    fn expert_packing_policy_resolves_storage_from_declared_variant() {
        let model = "hf://Qwen/Qwen3-30B-A3B";
        let experts_dir = Path::new("unused-experts");
        let config = packing_layout_config();

        let q4 =
            ExpertPackingPolicy::new(model, experts_dir, ExpertQuantization::FourBitProduction);
        assert_eq!(
            fixed_dense_expert_slot_spec_for_pack(q4, None).unwrap(),
            None
        );

        for (quantization, expected_dtype) in [
            (ExpertQuantization::Bf16, DenseExpertDtype::Bf16),
            (ExpertQuantization::F16, DenseExpertDtype::F16),
        ] {
            let policy = ExpertPackingPolicy::new(model, experts_dir, quantization);
            let spec = fixed_dense_expert_slot_spec_for_pack(policy, Some(&config))
                .unwrap()
                .unwrap();
            assert_eq!(spec.dtype, expected_dtype);
            assert_eq!(spec.hidden_size, config.hidden_size);
            assert_eq!(
                spec.intermediate_size,
                config.moe_intermediate_size.unwrap()
            );
        }

        let missing_config = ExpertPackingPolicy::new(model, experts_dir, ExpertQuantization::Bf16);
        assert_eq!(
            fixed_dense_expert_slot_spec_for_pack(missing_config, None)
                .unwrap_err()
                .to_string(),
            "Qwen config is required for fixed dense expert packing"
        );
    }

    #[derive(Debug)]
    struct TestAggregateTensor {
        name: &'static str,
        shape: Vec<usize>,
        native_q4: bool,
    }

    impl AggregateExpertTensor for TestAggregateTensor {
        fn aggregate_tensor_name(&self) -> &str {
            self.name
        }

        fn aggregate_tensor_shape(&self) -> &[usize] {
            &self.shape
        }

        fn aggregate_tensor_has_native_q4(&self) -> bool {
            self.native_q4
        }
    }

    impl ExpertSourceTensor for TestAggregateTensor {
        fn expert_source_offsets(&self) -> Option<[u64; 2]> {
            Some([100, 200])
        }
    }

    #[test]
    fn aggregate_expert_tensor_kind_classifies_qwen_and_mlx_names() {
        assert_eq!(
            aggregate_expert_tensor_kind("model.layers.0.mlp.experts.gate_up_proj"),
            Some(AggregateExpertTensorKind::GateUp)
        );
        assert_eq!(
            aggregate_expert_tensor_kind("model.layers.0.mlp.switch_mlp.gate_proj.weight"),
            Some(AggregateExpertTensorKind::Gate)
        );
        assert_eq!(
            aggregate_expert_tensor_kind("model.layers.0.mlp.switch_mlp.up_proj.weight"),
            Some(AggregateExpertTensorKind::Up)
        );
        assert_eq!(
            aggregate_expert_tensor_kind("model.layers.0.mlp.experts.down_proj.weight"),
            Some(AggregateExpertTensorKind::Down)
        );
        assert_eq!(
            aggregate_expert_tensor_kind("model.layers.0.self_attn.q_proj"),
            None
        );
    }

    #[test]
    fn aggregate_expert_layout_computes_split_ranges_from_model_shape() {
        let layout = AggregateExpertLayout::new(64, 3584, 1024).unwrap();

        assert_eq!(layout.experts, 64);
        assert_eq!(layout.hidden, 3584);
        assert_eq!(layout.intermediate, 1024);
        assert_eq!(layout.single_projection_values, 1024 * 3584);
        assert_eq!(layout.gate_up_expert_values, 2 * 1024 * 3584);
        assert_eq!(layout.down_expert_values, 3584 * 1024);
    }

    #[test]
    fn aggregate_expert_tensors_plan_combined_gate_up_slices() {
        let gate_up = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.gate_up_proj.weight",
            shape: vec![2, 4, 3],
            native_q4: false,
        };
        let down = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.down_proj.weight",
            shape: vec![2, 3, 2],
            native_q4: false,
        };
        let layout = AggregateExpertLayout::new(2, 3, 2).unwrap();
        let tensors = aggregate_expert_tensors(&[&gate_up, &down], 0, layout).unwrap();

        assert_eq!(tensors.gate.start(1).unwrap(), 12);
        assert_eq!(tensors.up.start(1).unwrap(), 18);
        assert_eq!(
            single_aggregate_expert_tensor(&[&gate_up, &down], AggregateExpertTensorKind::Down, 0)
                .unwrap()
                .name,
            down.name
        );
    }

    #[test]
    fn aggregate_expert_tensors_plan_split_switch_mlp_slices() {
        let gate = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.gate_proj.weight",
            shape: vec![2, 2, 3],
            native_q4: true,
        };
        let up = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.up_proj.weight",
            shape: vec![2, 2, 3],
            native_q4: true,
        };
        let down = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.down_proj.weight",
            shape: vec![2, 3, 2],
            native_q4: true,
        };
        let layout = AggregateExpertLayout::new(2, 3, 2).unwrap();
        let tensors = aggregate_expert_tensors(&[&gate, &up, &down], 1, layout).unwrap();

        assert_eq!(tensors.gate.start(1).unwrap(), 6);
        assert_eq!(tensors.up.start(1).unwrap(), 6);
        assert!(aggregate_native_q4_enabled(&tensors, &down).unwrap());
    }

    #[test]
    fn native_q4_layout_resolves_from_qwen_moe_dimensions() {
        let gate = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.gate_proj.weight",
            shape: vec![512, 1536, 4096],
            native_q4: true,
        };
        let up = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.up_proj.weight",
            shape: vec![512, 1536, 4096],
            native_q4: true,
        };
        let down = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.down_proj.weight",
            shape: vec![512, 4096, 1536],
            native_q4: true,
        };
        let aggregate = AggregateExpertLayout::new(512, 4096, 1536).unwrap();
        let tensors = aggregate_expert_tensors(&[&gate, &up, &down], 1, aggregate).unwrap();
        let fixed = fixed_native_q4_aggregate_layout(&tensors, &down, aggregate)
            .unwrap()
            .unwrap();

        assert_eq!(fixed.group_size, GROUP_SIZE);
        assert_eq!(fixed.expert_bytes, 10_616_832);
        assert_ne!(fixed, QwenMoeQ4ExpertLayout::qwen35_a17b());
    }

    #[test]
    fn native_q4_layout_keeps_non_runtime_fixture_as_import_format() {
        let gate = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.gate_proj.weight",
            shape: vec![2, 2, 3],
            native_q4: true,
        };
        let up = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.up_proj.weight",
            shape: vec![2, 2, 3],
            native_q4: true,
        };
        let down = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.down_proj.weight",
            shape: vec![2, 3, 2],
            native_q4: true,
        };
        let aggregate = AggregateExpertLayout::new(2, 3, 2).unwrap();
        let tensors = aggregate_expert_tensors(&[&gate, &up, &down], 1, aggregate).unwrap();

        assert!(
            fixed_native_q4_aggregate_layout(&tensors, &down, aggregate)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn aggregate_native_q4_requires_consistent_sources() {
        let gate = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.gate_proj.weight",
            shape: vec![2, 2, 3],
            native_q4: true,
        };
        let up = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.up_proj.weight",
            shape: vec![2, 2, 3],
            native_q4: false,
        };
        let down = TestAggregateTensor {
            name: "model.layers.1.mlp.switch_mlp.down_proj.weight",
            shape: vec![2, 3, 2],
            native_q4: true,
        };
        let layout = AggregateExpertLayout::new(2, 3, 2).unwrap();
        let tensors = aggregate_expert_tensors(&[&gate, &up, &down], 1, layout).unwrap();
        let err = aggregate_native_q4_enabled(&tensors, &down).unwrap_err();

        assert!(
            err.to_string().contains(
                "aggregate expert tensors must be all native MLX Q4 or all decoded tensors"
            )
        );
    }

    #[test]
    fn q4_record_layout_counts_packed_bytes_and_groups_from_shape() {
        assert_eq!(q4_record_layout_for_shape(&[3, 5]).unwrap(), (9, 3));
        assert_eq!(q4_record_layout_for_shape(&[2, 3, 33]).unwrap(), (102, 6));
    }

    #[test]
    fn q4_record_layout_rejects_zero_column_shape() {
        let err = q4_record_layout_for_shape(&[3, 0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot compute q4 layout for zero-column tensor")
        );
    }

    #[test]
    fn direct_expert_tensor_group_accepts_gate_up_down_shapes() {
        let gate = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.7.gate_proj.weight",
            shape: vec![2, 3],
            native_q4: false,
        };
        let up = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.7.up_proj.weight",
            shape: vec![2, 3],
            native_q4: false,
        };
        let down = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.7.down_proj.weight",
            shape: vec![3, 2],
            native_q4: false,
        };

        validate_direct_expert_tensor_group(
            0,
            7,
            &[&gate, &up, &down],
            Some(DirectExpertTensorShape::new(3, 2).unwrap()),
        )
        .unwrap();
    }

    #[test]
    fn direct_expert_tensor_group_rejects_missing_or_duplicate_projection() {
        let gate = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.7.gate_proj.weight",
            shape: vec![2, 3],
            native_q4: false,
        };
        let duplicate_gate = TestAggregateTensor {
            name: "alias.model.layers.0.mlp.experts.7.gate_proj.weight",
            shape: vec![2, 3],
            native_q4: false,
        };
        let up = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.7.up_proj.weight",
            shape: vec![2, 3],
            native_q4: false,
        };
        let down = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.7.down_proj.weight",
            shape: vec![3, 2],
            native_q4: false,
        };

        let missing = validate_direct_expert_tensor_group(0, 7, &[&gate, &up], None).unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("is missing required tensor down_proj.weight")
        );

        let duplicate =
            validate_direct_expert_tensor_group(0, 7, &[&gate, &duplicate_gate, &up, &down], None)
                .unwrap_err();
        assert!(
            duplicate
                .to_string()
                .contains("has duplicate tensors ending in gate_proj.weight")
        );
    }

    #[test]
    fn direct_expert_tensor_group_rejects_shape_mismatch() {
        let gate = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.7.gate_proj.weight",
            shape: vec![2, 3],
            native_q4: false,
        };
        let up = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.7.up_proj.weight",
            shape: vec![2, 4],
            native_q4: false,
        };
        let down = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.7.down_proj.weight",
            shape: vec![3, 2],
            native_q4: false,
        };

        let err = validate_direct_expert_tensor_group(
            0,
            7,
            &[&gate, &up, &down],
            Some(DirectExpertTensorShape::new(3, 2).unwrap()),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("expected [2, 3] for up_proj.weight")
        );
    }

    #[test]
    fn native_q4_slice_byte_ranges_plan_row_aligned_slices() {
        let source = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.gate_up_proj.weight",
            shape: vec![4, 8],
            native_q4: true,
        };
        let ranges = native_q4_slice_byte_ranges(&source, &[2, 8], "BF16", 8, 16).unwrap();

        assert_eq!(
            ranges,
            NativeQ4SliceByteRanges {
                packed_offset: 4,
                packed_bytes: 8,
                scale_bias_offset: 2,
                scale_bias_bytes: 4,
                groups: 2,
            }
        );
    }

    #[test]
    fn native_q4_slice_byte_ranges_reject_unaligned_or_mismatched_slices() {
        let source = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.gate_up_proj.weight",
            shape: vec![4, 8],
            native_q4: true,
        };

        let unaligned = native_q4_slice_byte_ranges(&source, &[1, 8], "BF16", 1, 8).unwrap_err();
        assert!(
            unaligned
                .to_string()
                .contains("is not aligned to 8 columns")
        );

        let mismatch = native_q4_slice_byte_ranges(&source, &[3, 8], "BF16", 8, 16).unwrap_err();
        assert!(
            mismatch
                .to_string()
                .contains("slice shape [3, 8] does not match 2 rows x 8 cols")
        );
    }

    #[test]
    fn expert_tensor_byte_range_uses_source_offsets_and_dtype_width() {
        let tensor = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.7.gate_proj.weight",
            shape: vec![2, 3],
            native_q4: false,
        };

        assert_eq!(
            expert_tensor_byte_range(&tensor, "BF16", 3, 4).unwrap(),
            [106, 114]
        );
        assert_eq!(
            expert_tensor_byte_range(&tensor, "U8", 3, 4).unwrap(),
            [103, 107]
        );
    }

    #[test]
    fn expert_tensor_byte_range_rejects_unsupported_or_out_of_bounds_ranges() {
        let tensor = TestAggregateTensor {
            name: "model.layers.0.mlp.experts.7.gate_proj.weight",
            shape: vec![2, 3],
            native_q4: false,
        };

        let dtype = expert_tensor_byte_range(&tensor, "U32", 0, 1).unwrap_err();
        assert!(dtype.to_string().contains("has unsupported dtype U32"));

        let bounds = expert_tensor_byte_range(&tensor, "F32", 20, 10).unwrap_err();
        assert!(bounds.to_string().contains("exceeds source offsets"));
    }

    #[test]
    fn expected_expert_record_from_source_uses_q4_layout_accounting() {
        let record = expected_expert_pack_record_from_source(
            "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
            "BF16".to_string(),
            vec![3, 5],
            [12, 42],
            "hash".to_string(),
        )
        .unwrap();

        assert_eq!(record.packed_bytes, 9);
        assert_eq!(record.groups, 3);
        assert_eq!(record.group_size, GROUP_SIZE);
        assert_eq!(record.scale_bias_dtype, EXPERT_PACK_SCALE_BIAS_DTYPE);
    }

    #[test]
    fn expected_native_q4_record_requires_source_hash() {
        let input = NativeQ4ExpertRecordInput {
            tensor: "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
            dtype: "U32".to_string(),
            shape: vec![3, 5],
            source_offsets: [12, 42],
            source_hash: None,
            packed: vec![0; 9],
            scale_bytes: vec![0; 6],
            bias_bytes: vec![0; 6],
            groups: 3,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        };

        let err = expected_native_q4_expert_record_from_input(input).unwrap_err();
        assert!(
            err.to_string()
                .contains("native q4 expert record is missing source hash")
        );
    }

    #[test]
    fn expected_expert_pack_from_records_accounts_for_wire_size() {
        let record = expected_expert_pack_record_from_source(
            "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
            "BF16".to_string(),
            vec![3, 5],
            [12, 42],
            "hash".to_string(),
        )
        .unwrap();
        let expected_size = pbq4_expert_pack_wire_size(&[record.clone()]).unwrap();
        let pack = expected_expert_pack_from_records(1, vec![record]).unwrap();

        assert_eq!(pack.expert, 1);
        assert_eq!(pack.packed_bytes, expected_size);
        assert_eq!(pack.records.len(), 1);
    }

    fn tiny_fixed_q4_layout() -> QwenMoeQ4ExpertLayout {
        use QwenMoeExpertComponentKind::*;
        QwenMoeQ4ExpertLayout {
            expert_bytes: 45,
            group_size: 2,
            components: [
                QwenMoeExpertComponentLayout {
                    kind: GateWeight,
                    offset: 0,
                    bytes: 8,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateScale,
                    offset: 8,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateBias,
                    offset: 12,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpWeight,
                    offset: 16,
                    bytes: 8,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpScale,
                    offset: 24,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpBias,
                    offset: 28,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownWeight,
                    offset: 32,
                    bytes: 8,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownScale,
                    offset: 40,
                    bytes: 3,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownBias,
                    offset: 43,
                    bytes: 2,
                },
            ],
        }
    }

    fn tiny_pbq4_expert_pack() -> (Vec<u8>, ExpertPackMetadata) {
        let tensor = "model.layers.0.mlp.experts.1.gate_proj.weight".to_string();
        let mut pack = PBQ4_EXPERT_MAGIC.to_vec();
        let record_offset = pack.len() as u64;
        pack.extend_from_slice(&(tensor.len() as u32).to_le_bytes());
        pack.extend_from_slice(tensor.as_bytes());
        pack.extend_from_slice(&2u64.to_le_bytes());
        pack.extend_from_slice(&2u64.to_le_bytes());
        for value in [1.0f32, 1.5] {
            pack.extend_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
        }
        for value in [0.0f32, 2.0] {
            pack.extend_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
        }
        pack.extend_from_slice(&[0xf0, 0x0f]);
        let packed_bytes = pack.len() as u64;
        (
            pack,
            ExpertPackMetadata {
                layer: 0,
                expert: 1,
                packed_bytes,
                records: vec![ExpertPackRecord {
                    tensor,
                    dtype: "F32".to_string(),
                    shape: vec![2, 2],
                    source_offsets: [16, 32],
                    source_hash: Some("fixture".to_string()),
                    record_offset,
                    packed_bytes: 2,
                    groups: 2,
                    group_size: GROUP_SIZE,
                    scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                }],
            },
        )
    }

    #[derive(Debug, Clone)]
    struct TestExpertPackWireRecord {
        tensor: String,
        packed_bytes: u64,
        groups: usize,
        scale_bias_dtype: String,
    }

    impl ExpertPackWireRecord for TestExpertPackWireRecord {
        fn tensor_name(&self) -> &str {
            &self.tensor
        }

        fn packed_bytes(&self) -> u64 {
            self.packed_bytes
        }

        fn scale_bias_groups(&self) -> usize {
            self.groups
        }

        fn scale_bias_dtype(&self) -> &str {
            &self.scale_bias_dtype
        }
    }

    #[test]
    fn pbq4_expert_wire_size_accounts_for_bf16_scale_bias_metadata() {
        let records: Vec<TestExpertPackWireRecord> = [
            ("gate_proj.weight", 2_097_152, 65_536),
            ("up_proj.weight", 2_097_152, 65_536),
            ("down_proj.weight", 2_097_152, 65_536),
        ]
        .into_iter()
        .map(|(tensor, packed_bytes, groups)| TestExpertPackWireRecord {
            tensor: tensor.to_string(),
            packed_bytes,
            groups,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
        })
        .collect();
        let f32_size = pbq4_expert_pack_wire_size(&records).unwrap();
        let mut bf16_records = records.clone();
        for record in &mut bf16_records {
            record.scale_bias_dtype = EXPERT_SCALE_BIAS_DTYPE_BF16.to_string();
        }
        let bf16_size = pbq4_expert_pack_wire_size(&bf16_records).unwrap();

        assert_eq!(f32_size - bf16_size, 786_432);
        assert!(bf16_size < f32_size);
    }

    #[test]
    fn fixed_q4_expert_slot_records_are_derived_from_layout_offsets() {
        let layout = tiny_fixed_q4_layout();
        let mut payload: Vec<u8> = (0..45).collect();
        payload[8..12].copy_from_slice(
            &[
                f32_to_bf16_bits(0.5).to_le_bytes(),
                f32_to_bf16_bits(1.5).to_le_bytes(),
            ]
            .concat(),
        );
        payload[12..16].copy_from_slice(
            &[
                f32_to_bf16_bits(-1.0).to_le_bytes(),
                f32_to_bf16_bits(2.0).to_le_bytes(),
            ]
            .concat(),
        );
        payload[24..28].copy_from_slice(
            &[
                f32_to_bf16_bits(3.0).to_le_bytes(),
                f32_to_bf16_bits(4.0).to_le_bytes(),
            ]
            .concat(),
        );
        payload[28..32].copy_from_slice(
            &[
                f32_to_bf16_bits(5.0).to_le_bytes(),
                f32_to_bf16_bits(6.0).to_le_bytes(),
            ]
            .concat(),
        );

        let slot = ExpertSlotView::new(2, 9, 128, 45, &payload).unwrap();
        let view = FixedQ4ExpertSlotView::new(slot, layout).unwrap();
        let records = fixed_q4_expert_records(
            &view,
            FixedQ4ExpertSlotSpec {
                layout,
                hidden_size: 2,
                intermediate_size: 2,
            },
        )
        .unwrap();

        assert_eq!(records.len(), 3);
        assert_eq!(
            records[0].name,
            "model.layers.2.mlp.experts.9.gate_proj.weight"
        );
        assert_eq!(records[0].shape, vec![2, 2]);
        assert_eq!(records[0].packed, payload[0..8]);
        assert_eq!(records[0].scales, vec![0.5, 1.5]);
        assert_eq!(records[0].biases, vec![-1.0, 2.0]);
        assert_eq!(records[1].packed, payload[16..24]);
        assert_eq!(records[1].scales, vec![3.0, 4.0]);
        assert_eq!(records[1].biases, vec![5.0, 6.0]);
        assert_eq!(records[2].shape, vec![2, 2]);
        assert_eq!(records[2].scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
    }

    #[test]
    fn pbq4_metadata_parser_matches_generic_parser() {
        let (pack, metadata) = tiny_pbq4_expert_pack();
        let generic = parse_pbq4_expert_pack_generic(&pack, Some(&metadata)).unwrap();
        let metadata_fast = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap();

        assert_eq!(metadata_fast, generic);
        assert_eq!(metadata_fast[0].packed, vec![0xf0, 0x0f]);
        assert_eq!(metadata_fast[0].scales, vec![1.0, 1.5]);
        assert_eq!(metadata_fast[0].biases, vec![0.0, 2.0]);
        assert_eq!(metadata_fast[0].source_offsets(), [16, 32]);
    }

    #[test]
    fn pbq4_metadata_parser_rejects_record_offset_drift() {
        let (pack, mut metadata) = tiny_pbq4_expert_pack();
        metadata.records[0].record_offset += 1;

        let err = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap_err();

        assert!(
            err.to_string().contains("metadata offset mismatch"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn read_f32_vec_le_bulk_path_decodes_values_and_advances_cursor() {
        let mut bytes = vec![0xaa, 0xbb, 0xcc];
        for value in [1.0f32, -2.5, f32::from_bits(0x7fc0_1234)] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[0xdd, 0xee]);

        let mut cursor = 3;
        let values = read_f32_vec_le(&bytes, &mut cursor, 3).unwrap();
        assert_eq!(cursor, 15);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], -2.5);
        assert_eq!(values[2].to_bits(), 0x7fc0_1234);
    }

    #[test]
    fn read_bf16_vec_le_decodes_values_and_advances_cursor() {
        let mut bytes = vec![0xaa];
        for value in [0.5f32, -2.0, 1.25] {
            bytes.extend_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
        }
        bytes.push(0xbb);

        let mut cursor = 1;
        let values = read_bf16_vec_le(&bytes, &mut cursor, 3).unwrap();
        assert_eq!(cursor, 7);
        assert_eq!(values, vec![0.5, -2.0, 1.25]);
    }

    #[test]
    fn reusable_expert_buffer_keeps_capacity_across_smaller_reads() {
        let mut buffer = ReusableExpertBuffer::default();
        buffer.prepare_payload(128, 96).unwrap().fill(7);
        let initial_capacity = buffer.capacity();

        buffer.prepare_payload(128, 8).unwrap().fill(3);
        let slot = buffer.slot_view(2, 5, 1024, 128).unwrap();

        assert_eq!(buffer.capacity(), initial_capacity);
        assert_eq!(
            slot.descriptor(),
            ExpertSlotDescriptor {
                layer: 2,
                expert: 5,
                slot_offset: 1024,
                slot_capacity: 128,
                payload_len: 8,
            }
        );
        assert_eq!(slot.payload(), &[3; 8]);
    }

    #[test]
    fn reusable_expert_buffer_can_move_a_whole_slot_payload_without_copying() {
        let mut buffer = ReusableExpertBuffer::default();
        buffer.prepare_payload(128, 96).unwrap().fill(9);
        let initial_capacity = buffer.capacity();

        let payload = buffer.take_payload();

        assert_eq!(payload, vec![9; 96]);
        assert_eq!(payload.capacity(), initial_capacity);
        assert_eq!(buffer.capacity(), 0);
    }

    #[test]
    fn fixed_q4_expert_slot_view_slices_components_from_one_payload() {
        let payload: Vec<u8> = (0..45).collect();
        let slot = ExpertSlotView::new(4, 7, 4096, 45, &payload).unwrap();
        let view = FixedQ4ExpertSlotView::new(slot, tiny_fixed_q4_layout()).unwrap();

        assert_eq!(view.descriptor(), slot.descriptor());
        assert_eq!(view.payload(), payload.as_slice());
        assert_eq!(
            view.component(QwenMoeExpertComponentKind::GateWeight),
            &payload[0..8]
        );
        assert_eq!(
            view.component(QwenMoeExpertComponentKind::UpScale),
            &payload[24..28]
        );
        assert_eq!(
            view.component(QwenMoeExpertComponentKind::DownBias),
            &payload[43..45]
        );
    }

    #[test]
    fn fixed_q4_expert_slot_view_rejects_short_payloads() {
        let payload = [0u8; 44];
        let slot = ExpertSlotView::new(0, 0, 0, 45, &payload).unwrap();
        let err = FixedQ4ExpertSlotView::new(slot, tiny_fixed_q4_layout()).unwrap_err();

        assert!(
            err.to_string().contains("shorter than layout size 45"),
            "{err:#}"
        );
    }

    #[test]
    fn fixed_q4_payload_requires_whole_slot_bytes() {
        let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
        let payload = FixedQ4ExpertPayload::from_whole_slot(spec, (0..45).collect(), None).unwrap();

        assert_eq!(
            payload.component(QwenMoeExpertComponentKind::GateWeight),
            &[0, 1, 2, 3, 4, 5, 6, 7]
        );

        let err = FixedQ4ExpertPayload::from_whole_slot(spec, vec![0; 44], None).unwrap_err();
        assert!(
            err.to_string().contains("whole-slot payload length 44"),
            "{err:#}"
        );
    }

    #[test]
    fn expert_store_resolves_validated_fixed_q4_execution_storage() {
        let temp = tempfile::tempdir().unwrap();
        let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
        let experts = 2;
        let bytes = vec![0u8; spec.layout.expert_bytes * experts];
        fs::write(expert_layer_path(temp.path(), 0), bytes).unwrap();
        let metadata = ExpertLayerPackMetadata::new_fixed_q4(
            0,
            spec.layout.expert_bytes as u64,
            experts,
            (0..experts)
                .map(|expert| tiny_pack_metadata(0, expert, spec.layout.expert_bytes as u64))
                .collect(),
        );
        write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();
        let store = ExpertSlotStore::open_with_fixed_q4(temp.path().to_path_buf(), spec).unwrap();

        let descriptor = store.resolve_execution_descriptor(1, experts).unwrap();

        assert_eq!(descriptor.layout, ExpertStorageLayout::FixedQ4);
        assert_eq!(descriptor.slot_spec, ExpertSlotSpec::FixedQ4(spec));
        assert_eq!(descriptor.layers, 1);
        assert_eq!(descriptor.experts_per_layer, experts);
    }

    #[test]
    fn expert_store_resolves_q4_layout_from_fixed_slot_metadata() {
        let layout = tiny_qwen_moe_layout();
        let temp = tempfile::tempdir().unwrap();
        let spec = FixedQ4ExpertSlotSpec::from_model_layout(&layout).unwrap();
        fs::write(
            expert_layer_path(temp.path(), 0),
            vec![0u8; spec.layout.expert_bytes * layout.experts_per_layer],
        )
        .unwrap();
        let metadata = ExpertLayerPackMetadata::new_fixed_q4(
            0,
            spec.layout.expert_bytes as u64,
            layout.experts_per_layer,
            (0..layout.experts_per_layer)
                .map(|expert| tiny_pack_metadata(0, expert, spec.layout.expert_bytes as u64))
                .collect(),
        );
        write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();

        let resolved = ExpertSlotStore::resolve_from_metadata(
            temp.path().to_path_buf(),
            &layout,
            ExpertQuantization::FourBitProduction,
        )
        .unwrap();

        assert_eq!(resolved.descriptor.slot_spec, ExpertSlotSpec::FixedQ4(spec));
        assert_eq!(resolved.upgraded_pbq4_layers, 0);

        let error = ExpertSlotStore::resolve_from_metadata(
            temp.path().to_path_buf(),
            &layout,
            ExpertQuantization::Bf16,
        )
        .unwrap_err();
        assert!(error.to_string().contains("requested BF16 expert weights"));
        assert!(error.to_string().contains("resolves fixed-Q4 expert slots"));
    }

    #[test]
    fn requested_expert_storage_rejects_every_mismatched_resolved_layout() {
        for resolved in [
            ExpertStorageLayout::FixedQ4,
            ExpertStorageLayout::FixedBf16,
            ExpertStorageLayout::FixedF16,
        ] {
            for requested in [
                ExpertQuantization::FourBitProduction,
                ExpertQuantization::Bf16,
                ExpertQuantization::F16,
            ] {
                let result = validate_requested_expert_storage(
                    Path::new("/cache/packed_experts"),
                    resolved,
                    requested,
                );
                if resolved.quantization() == requested {
                    result.unwrap();
                } else {
                    let error = result.unwrap_err().to_string();
                    assert!(
                        error.contains("unsupported expert storage policy"),
                        "{error}"
                    );
                    assert!(error.contains(requested.as_str()), "{error}");
                    assert!(error.contains(resolved.as_str()), "{error}");
                }
            }
        }
    }

    #[test]
    fn fixed_dense_expert_slots_resolve_offsets_reads_and_typed_payloads() {
        for (dtype, expected_layout) in [
            (DenseExpertDtype::Bf16, ExpertStorageLayout::FixedBf16),
            (DenseExpertDtype::F16, ExpertStorageLayout::FixedF16),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let spec = FixedDenseExpertSlotSpec::new(dtype, 2, 3).unwrap();
            assert_eq!(spec.gate.offset, 0);
            assert_eq!(spec.gate.rows, 3);
            assert_eq!(spec.gate.cols, 2);
            assert_eq!(spec.up.offset, EXPERT_COMPONENT_ALIGNMENT);
            assert_eq!(spec.down.offset, EXPERT_COMPONENT_ALIGNMENT * 2);
            assert_eq!(spec.expert_bytes, EXPERT_COMPONENT_ALIGNMENT * 3);

            let experts = 2;
            let mut bytes = vec![0u8; spec.expert_bytes * experts];
            bytes[spec.gate.offset] = 11;
            bytes[spec.up.offset] = 22;
            bytes[spec.down.offset] = 33;
            fs::write(expert_layer_path(temp.path(), 0), bytes).unwrap();
            let metadata = ExpertLayerPackMetadata::new_fixed_dense(
                0,
                spec.expert_bytes as u64,
                experts,
                (0..experts)
                    .map(|expert| fixed_dense_test_pack(0, expert, spec, [dtype.as_str(); 3]))
                    .collect(),
            );
            write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();
            let store =
                ExpertSlotStore::open_with_fixed_dense(temp.path().to_path_buf(), spec).unwrap();

            let descriptor = store.resolve_execution_descriptor(1, experts).unwrap();
            assert_eq!(descriptor.layout, expected_layout);
            assert_eq!(descriptor.slot_spec.fixed_dense(), Some(spec));

            let mut reads = store.read_many_raw(0, &[0]).unwrap();
            let raw = reads.pop().unwrap();
            assert_eq!(raw.slot.slot_capacity, spec.expert_bytes);
            let ExpertRawPayload::FixedDense(payload) = raw.payload else {
                panic!("fixed dense slot did not resolve a fixed dense payload");
            };
            let gate = payload
                .matvec_payload(ExpertMlpProjection::Gate, 2, 3)
                .unwrap();
            let up = payload
                .matvec_payload(ExpertMlpProjection::Up, 2, 3)
                .unwrap();
            let down = payload
                .matvec_payload(ExpertMlpProjection::Down, 3, 2)
                .unwrap();
            assert_eq!(gate.dtype, dtype);
            assert_eq!(gate.source.byte_offset, spec.gate.offset);
            assert_eq!(up.source.byte_offset, spec.up.offset);
            assert_eq!(down.source.byte_offset, spec.down.offset);
            assert_eq!(gate.source.bytes[gate.source.byte_offset], 11);
            assert_eq!(up.source.bytes[up.source.byte_offset], 22);
            assert_eq!(down.source.bytes[down.source.byte_offset], 33);
        }
    }

    #[test]
    fn expert_store_resolves_typed_layout_from_fixed_slot_metadata() {
        let layout = tiny_qwen_moe_layout();
        for dtype in [DenseExpertDtype::Bf16, DenseExpertDtype::F16] {
            let temp = tempfile::tempdir().unwrap();
            let spec = FixedDenseExpertSlotSpec::from_model_layout(&layout, dtype).unwrap();
            fs::write(
                expert_layer_path(temp.path(), 0),
                vec![0u8; spec.expert_bytes * layout.experts_per_layer],
            )
            .unwrap();
            let metadata = ExpertLayerPackMetadata::new_fixed_dense(
                0,
                spec.expert_bytes as u64,
                layout.experts_per_layer,
                (0..layout.experts_per_layer)
                    .map(|expert| fixed_dense_test_pack(0, expert, spec, [dtype.as_str(); 3]))
                    .collect(),
            );
            write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();

            let quantization = match dtype {
                DenseExpertDtype::Bf16 => ExpertQuantization::Bf16,
                DenseExpertDtype::F16 => ExpertQuantization::F16,
            };
            let resolved = ExpertSlotStore::resolve_from_metadata(
                temp.path().to_path_buf(),
                &layout,
                quantization,
            )
            .unwrap();

            assert_eq!(resolved.descriptor.slot_spec.fixed_dense(), Some(spec));
            assert_eq!(resolved.upgraded_pbq4_layers, 0);

            let mismatched_quantization = match dtype {
                DenseExpertDtype::Bf16 => ExpertQuantization::F16,
                DenseExpertDtype::F16 => ExpertQuantization::Bf16,
            };
            let error = ExpertSlotStore::resolve_from_metadata(
                temp.path().to_path_buf(),
                &layout,
                mismatched_quantization,
            )
            .unwrap_err();
            assert!(
                error.to_string().contains(mismatched_quantization.as_str()),
                "{error:#}"
            );
            assert!(
                error
                    .to_string()
                    .contains(resolved.descriptor.layout.as_str()),
                "{error:#}"
            );
        }
    }

    #[test]
    fn expert_store_rejects_mixed_fixed_dense_metadata_before_scheduling() {
        let layout = tiny_qwen_moe_layout();
        let temp = tempfile::tempdir().unwrap();
        let spec =
            FixedDenseExpertSlotSpec::from_model_layout(&layout, DenseExpertDtype::Bf16).unwrap();
        fs::write(
            expert_layer_path(temp.path(), 0),
            vec![0u8; spec.expert_bytes * layout.experts_per_layer],
        )
        .unwrap();
        let metadata = ExpertLayerPackMetadata::new_fixed_dense(
            0,
            spec.expert_bytes as u64,
            layout.experts_per_layer,
            (0..layout.experts_per_layer)
                .map(|expert| fixed_dense_test_pack(0, expert, spec, ["BF16", "F16", "BF16"]))
                .collect(),
        );
        write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();

        let error = ExpertSlotStore::resolve_from_metadata(
            temp.path().to_path_buf(),
            &layout,
            ExpertQuantization::Bf16,
        )
        .unwrap_err();

        assert!(
            error.to_string().contains("mixes BF16 and F16"),
            "{error:#}"
        );
    }

    #[test]
    fn expert_store_rejects_partial_fixed_q4_execution_storage() {
        let temp = tempfile::tempdir().unwrap();
        let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
        fs::write(
            expert_layer_path(temp.path(), 0),
            vec![0u8; spec.layout.expert_bytes],
        )
        .unwrap();
        let metadata = ExpertLayerPackMetadata::new_fixed_q4(
            0,
            spec.layout.expert_bytes as u64,
            2,
            vec![tiny_pack_metadata(0, 0, spec.layout.expert_bytes as u64)],
        );
        write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();
        let store = ExpertSlotStore::open_with_fixed_q4(temp.path().to_path_buf(), spec).unwrap();

        let err = store.resolve_execution_descriptor(1, 2).unwrap_err();

        assert!(err.to_string().contains("1 records, expected 2"), "{err:#}");
    }

    #[test]
    fn fixed_q4_payload_resolves_typed_matvec_offsets_from_whole_slot() {
        let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
        let payload = FixedQ4ExpertPayload::from_whole_slot(spec, (0..45).collect(), None).unwrap();

        let gate = payload
            .matvec_payload(ExpertMlpProjection::Gate, 2, 2)
            .unwrap();
        assert_eq!(gate.rows, 2);
        assert_eq!(gate.cols, 2);
        assert_eq!(gate.group_size, 2);
        assert_eq!(gate.scale_bias_groups, 2);
        assert_eq!(gate.scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
        assert_eq!(gate.packed, &[0, 1]);
        assert_eq!(gate.scale_bytes, &[8, 9, 10, 11]);
        assert_eq!(gate.bias_bytes, &[12, 13, 14, 15]);

        let source = gate.source.unwrap();
        assert_eq!(source.bytes, payload.payload_prefix(45));
        assert_eq!(source.packed_offset, 0);
        assert_eq!(source.scale_offset, 8);
        assert_eq!(source.bias_offset, 12);
        assert!(source.covers(&gate));
        assert!(source.offsets_are_metal_aligned());

        let up = payload
            .matvec_payload(ExpertMlpProjection::Up, 2, 2)
            .unwrap();
        let up_source = up.source.unwrap();
        assert!(source.same_buffer(up_source));
        assert_eq!(up_source.packed_offset, 16);
        assert_eq!(up_source.scale_offset, 24);
        assert_eq!(up_source.bias_offset, 28);
    }

    #[test]
    fn fixed_q4_payload_rejects_partial_projection_bytes() {
        let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
        let payload = FixedQ4ExpertPayload::from_whole_slot(spec, (0..45).collect(), None).unwrap();

        assert!(
            payload
                .matvec_payload(ExpertMlpProjection::Down, 2, 2)
                .is_none()
        );
    }

    fn tiny_pack_metadata(layer: usize, expert: usize, packed_bytes: u64) -> ExpertPackMetadata {
        ExpertPackMetadata {
            layer,
            expert,
            packed_bytes,
            records: Vec::new(),
        }
    }

    fn tiny_qwen_moe_layout() -> QwenMoeModelLayout {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"num_hidden_layers":1,"hidden_size":64,"num_attention_heads":8,"num_key_value_heads":1,"vocab_size":128,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":64,"norm_topk_prob":true}"#,
        )
        .unwrap();
        QwenMoeModelLayout::from_config("hf://Qwen/tiny-moe", &config).unwrap()
    }

    fn fixed_dense_test_pack(
        layer: usize,
        expert: usize,
        spec: FixedDenseExpertSlotSpec,
        dtypes: [&str; 3],
    ) -> ExpertPackMetadata {
        let records = [
            ("gate_proj.weight", spec.gate, dtypes[0]),
            ("up_proj.weight", spec.up, dtypes[1]),
            ("down_proj.weight", spec.down, dtypes[2]),
        ]
        .into_iter()
        .map(|(suffix, projection, dtype)| ExpertPackRecord {
            tensor: format!("model.layers.{layer}.mlp.experts.{expert}.{suffix}"),
            dtype: dtype.to_string(),
            shape: vec![projection.rows, projection.cols],
            source_offsets: [0, projection.bytes as u64],
            source_hash: Some("fixture".to_string()),
            record_offset: projection.offset as u64,
            packed_bytes: projection.bytes as u64,
            groups: 0,
            group_size: 0,
            scale_bias_dtype: String::new(),
        })
        .collect();
        ExpertPackMetadata {
            layer,
            expert,
            packed_bytes: spec.expert_bytes as u64,
            records,
        }
    }

    #[test]
    fn expert_layer_metadata_validates_supported_formats_and_slots() {
        let metadata = ExpertLayerPackMetadata::new_fixed_q4(
            3,
            128,
            2,
            vec![tiny_pack_metadata(3, 0, 128), tiny_pack_metadata(3, 1, 64)],
        );
        metadata.validate(Path::new("layer_03.json"), 3).unwrap();

        let duplicate = ExpertLayerPackMetadata::new_fixed_q4(
            3,
            128,
            2,
            vec![tiny_pack_metadata(3, 0, 128), tiny_pack_metadata(3, 0, 64)],
        );
        let err = duplicate
            .validate(Path::new("layer_03.json"), 3)
            .unwrap_err();
        assert!(err.to_string().contains("duplicate expert 0"), "{err:#}");

        let oversized =
            ExpertLayerPackMetadata::new_fixed_q4(3, 128, 2, vec![tiny_pack_metadata(3, 1, 129)]);
        let err = oversized
            .validate(Path::new("layer_03.json"), 3)
            .unwrap_err();
        assert!(
            err.to_string().contains("length 129 exceeds slot size 128"),
            "{err:#}"
        );
    }

    #[test]
    fn expert_metadata_reader_uses_expert_owned_paths_and_validation() {
        let tmp = tempfile::tempdir().unwrap();
        let metadata = ExpertLayerPackMetadata::new(5, 256, 4, vec![tiny_pack_metadata(5, 2, 200)]);
        std::fs::write(
            expert_layer_metadata_path(tmp.path(), 5),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();

        let read_layer = read_expert_layer_pack_metadata(tmp.path(), 5)
            .unwrap()
            .unwrap();
        let read_pack = read_expert_pack_metadata(tmp.path(), 5, 2)
            .unwrap()
            .unwrap();

        assert_eq!(read_layer.format, PBQ4_EXPERT_LAYER_FORMAT_V2);
        assert_eq!(read_layer.pack_for(2), Some(&read_pack));
        assert_eq!(read_pack.packed_bytes, 200);
        assert_eq!(
            expert_layer_path(tmp.path(), 5),
            tmp.path().join("layer_05.bin")
        );
    }

    #[test]
    fn expert_layer_reader_reads_fixed_q4_whole_slots() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = tiny_fixed_q4_layout();
        let spec = FixedQ4ExpertSlotSpec::new(layout, 2, 2).unwrap();
        let payload: Vec<u8> = (0..45).collect();
        std::fs::write(expert_layer_path(tmp.path(), 0), &payload).unwrap();
        let metadata = ExpertLayerPackMetadata::new_fixed_q4(
            0,
            layout.expert_bytes as u64,
            1,
            vec![tiny_pack_metadata(0, 0, layout.expert_bytes as u64)],
        );
        std::fs::write(
            expert_layer_metadata_path(tmp.path(), 0),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();

        let reader = ExpertLayerReader::open(
            tmp.path(),
            0,
            ExpertSlotSpec::FixedQ4(spec),
            Arc::new(Mutex::new(Vec::new())),
        )
        .unwrap();
        let plan = reader.prepare_read(0).unwrap();
        let mut scratch = ReusableExpertBuffer::default();
        let raw = reader.read_prepared_into(0, plan, &mut scratch).unwrap();

        assert_eq!(raw.slot.layer, 0);
        assert_eq!(raw.slot.expert, 0);
        assert_eq!(raw.read_path, ExpertReadPath::PositionedRead);
        match raw.payload {
            ExpertRawPayload::FixedQ4(fixed) => {
                assert_eq!(
                    fixed.component(QwenMoeExpertComponentKind::GateWeight),
                    &[0, 1, 2, 3, 4, 5, 6, 7]
                );
            }
            ExpertRawPayload::Pbq4(_) => panic!("fixed slot classified as PBQ4"),
            ExpertRawPayload::FixedDense(_) => panic!("Q4 slot classified as fixed dense"),
        }
    }

    #[test]
    fn expert_layer_reader_keeps_pbq4_as_import_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = tiny_fixed_q4_layout();
        let spec = FixedQ4ExpertSlotSpec::new(layout, 2, 2).unwrap();
        let mut payload = PBQ4_EXPERT_MAGIC.to_vec();
        payload.extend_from_slice(&[1, 2, 3, 4]);
        std::fs::write(expert_layer_path(tmp.path(), 0), &payload).unwrap();
        let metadata = ExpertLayerPackMetadata::new(
            0,
            payload.len() as u64,
            1,
            vec![tiny_pack_metadata(0, 0, payload.len() as u64)],
        );
        std::fs::write(
            expert_layer_metadata_path(tmp.path(), 0),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();

        let reader = ExpertLayerReader::open(
            tmp.path(),
            0,
            ExpertSlotSpec::FixedQ4(spec),
            Arc::new(Mutex::new(Vec::new())),
        )
        .unwrap();
        let plan = reader.prepare_read(0).unwrap();
        let mut scratch = ReusableExpertBuffer::default();
        let raw = reader.read_prepared_into(0, plan, &mut scratch).unwrap();

        match raw.payload {
            ExpertRawPayload::Pbq4(bytes) => assert_eq!(bytes, payload),
            ExpertRawPayload::FixedQ4(_) => panic!("PBQ4 slot classified as fixed Q4"),
            ExpertRawPayload::FixedDense(_) => panic!("PBQ4 slot classified as fixed dense"),
        }
    }

    #[test]
    fn expert_read_worker_pool_returns_raw_whole_slot_reads() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = tiny_fixed_q4_layout();
        let spec = FixedQ4ExpertSlotSpec::new(layout, 2, 2).unwrap();
        let payload: Vec<u8> = (0..45).collect();
        std::fs::write(expert_layer_path(tmp.path(), 0), &payload).unwrap();
        let metadata = ExpertLayerPackMetadata::new_fixed_q4(
            0,
            layout.expert_bytes as u64,
            1,
            vec![tiny_pack_metadata(0, 0, layout.expert_bytes as u64)],
        );
        std::fs::write(
            expert_layer_metadata_path(tmp.path(), 0),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        let reader = Arc::new(
            ExpertLayerReader::open(
                tmp.path(),
                0,
                ExpertSlotSpec::FixedQ4(spec),
                Arc::new(Mutex::new(Vec::new())),
            )
            .unwrap(),
        );
        let plan = reader.prepare_read(0).unwrap();
        let mut pool = ExpertReadWorkerPool::default();

        let rx = pool
            .submit_read(11, 0, reader, plan, false, Instant::now())
            .unwrap();
        let response = rx.recv().unwrap();
        let raw = response.result.unwrap();

        assert_eq!(pool.worker_count(), 1);
        assert_eq!(response.id, 11);
        assert_eq!(response.read_path, ExpertReadPath::PositionedRead);
        assert_eq!(response.bytes_read, layout.expert_bytes as u64);
        assert!(!response.warm);
        assert_eq!(raw.slot.layer, 0);
        assert_eq!(raw.slot.expert, 0);
        match raw.payload {
            ExpertRawPayload::FixedQ4(fixed) => {
                assert_eq!(
                    fixed.component(QwenMoeExpertComponentKind::DownBias),
                    &[43, 44]
                );
            }
            ExpertRawPayload::Pbq4(_) => panic!("fixed slot classified as PBQ4"),
            ExpertRawPayload::FixedDense(_) => panic!("Q4 slot classified as fixed dense"),
        }
    }

    #[test]
    fn expert_slot_store_owns_layer_reader_cache_and_raw_reads() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = tiny_fixed_q4_layout();
        let spec = FixedQ4ExpertSlotSpec::new(layout, 2, 2).unwrap();
        let payload: Vec<u8> = (0..45).collect();
        std::fs::write(expert_layer_path(tmp.path(), 0), &payload).unwrap();
        let metadata = ExpertLayerPackMetadata::new_fixed_q4(
            0,
            layout.expert_bytes as u64,
            1,
            vec![tiny_pack_metadata(0, 0, layout.expert_bytes as u64)],
        );
        std::fs::write(
            expert_layer_metadata_path(tmp.path(), 0),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();

        let store = ExpertSlotStore::open_with_fixed_q4(tmp.path().to_path_buf(), spec).unwrap();
        let first_reader = store.layer_reader(0).unwrap();
        let second_reader = store.layer_reader(0).unwrap();
        let reads = store.read_many_raw(0, &[0]).unwrap();

        assert!(Arc::ptr_eq(&first_reader, &second_reader));
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0].slot.payload_len, layout.expert_bytes);
        assert!(matches!(reads[0].payload, ExpertRawPayload::FixedQ4(_)));
    }

    #[test]
    fn fixed_q4_slot_spec_validates_layout_and_dimensions() {
        let layout = tiny_fixed_q4_layout();
        let spec = FixedQ4ExpertSlotSpec::new(layout, 8, 4).unwrap();

        assert_eq!(spec.layout, layout);
        assert_eq!(spec.hidden_size, 8);
        assert_eq!(spec.intermediate_size, 4);
    }

    #[test]
    fn fixed_q4_slot_spec_rejects_invalid_layout_offsets() {
        let mut layout = tiny_fixed_q4_layout();
        layout.components[1].offset += 1;

        let err = FixedQ4ExpertSlotSpec::new(layout, 8, 4).unwrap_err();

        assert!(
            err.to_string()
                .contains("GateScale starts at 9, expected 8"),
            "{err:#}"
        );
    }

    #[test]
    fn fixed_q4_slot_spec_rejects_zero_dimensions() {
        let err = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 0, 4).unwrap_err();

        assert!(
            err.to_string().contains("requires non-zero dimensions"),
            "{err:#}"
        );
    }

    #[test]
    fn expert_slot_rejects_payloads_larger_than_the_slot() {
        let err = ExpertSlotView::new(0, 0, 0, 2, &[1, 2, 3]).unwrap_err();

        assert!(
            err.to_string()
                .contains("payload length 3 exceeds slot capacity 2"),
            "{err:#}"
        );
    }

    #[test]
    fn expert_io_policy_keeps_upstream_positioned_read_guardrails() {
        assert_eq!(
            FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
            ExpertReadPath::PositionedRead
        );
        assert!(!FLASHMOE_EXPERT_IO_POLICY.application_expert_cache);
        assert!(!FLASHMOE_EXPERT_IO_POLICY.lz4_expert_compression);
        assert!(!FLASHMOE_EXPERT_IO_POLICY.speculative_routing);
        assert!(!FLASHMOE_EXPERT_IO_POLICY.broad_ssd_gpu_overlap);
    }

    #[test]
    fn fixed_q4_payload_drop_recycles_whole_slot_bytes() {
        let spec = FixedQ4ExpertSlotSpec {
            layout: QwenMoeQ4ExpertLayout::qwen35_a17b(),
            hidden_size: 3584,
            intermediate_size: 1024,
        };
        let pool = Arc::new(Mutex::new(Vec::new()));
        let mut bytes = Vec::with_capacity(spec.layout.expert_bytes);
        bytes.resize(spec.layout.expert_bytes, 0);

        {
            let _payload = FixedQ4ExpertPayload {
                spec,
                bytes,
                recycle_pool: Some(Arc::clone(&pool)),
            };
        }

        assert_eq!(pool.lock().unwrap().len(), 1);
        let returned = take_reusable_expert_bytes(&pool, spec.layout.expert_bytes).unwrap();
        assert!(returned.capacity() >= spec.layout.expert_bytes);
        let mut scratch = ReusableExpertBuffer::default();
        let previous = scratch.adopt_buffer(returned);
        assert_eq!(previous.capacity(), 0);
        assert!(scratch.capacity() >= spec.layout.expert_bytes);
    }

    #[test]
    fn cleanup_stale_expert_temp_files_preserves_committed_layers() {
        let temp = tempfile::tempdir().unwrap();
        let experts_dir = temp.path();
        let final_bin = experts_dir.join("layer_00.bin");
        let temp_bin = experts_dir.join("layer_00.bin.tmp-123-ThreadId(1)");
        let temp_json = experts_dir.join("layer_00.json.tmp-123-ThreadId(1)");

        fs::write(&final_bin, b"PBQ4EXPERT ").unwrap();
        fs::write(&temp_bin, b"partial").unwrap();
        fs::write(&temp_json, b"partial").unwrap();

        let deleted = cleanup_stale_expert_temp_files(experts_dir).unwrap();

        assert_eq!(deleted, 2);
        assert!(final_bin.is_file());
        assert!(!temp_bin.exists());
        assert!(!temp_json.exists());
    }

    #[test]
    fn reusable_expert_byte_pool_reuses_capacity_qualified_buffers() {
        let pool: ReusableExpertBytePool = Arc::new(Mutex::new(Vec::new()));
        recycle_reusable_expert_bytes(&pool, Vec::with_capacity(64), 64);

        let returned = take_reusable_expert_bytes(&pool, 32).unwrap();

        assert!(returned.capacity() >= 64);
        assert!(pool.lock().unwrap().is_empty());
    }
}
