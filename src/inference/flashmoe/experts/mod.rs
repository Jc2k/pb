use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
#[cfg(not(unix))]
use std::io::Read;
use std::io::{Seek, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::OnceLock;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

pub(crate) use super::artifact::{
    AggregateExpertTensor, EXPERT_PACK_SCALE_BIAS_DTYPE, EXPERT_SCALE_BIAS_DTYPE_BF16,
    EXPERT_SCALE_BIAS_DTYPE_F32, EXPERT_SCALE_DTYPE_E8M0, ExpertSourceTensor,
    expert_scale_bias_dtype_size,
};
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
use super::weights::{
    DenseQ4SourceFormat, ExpertTensorRef, decode_dense_tensor_f32,
    dense_q4_layout_with_scale_bias_dtype, write_colibri_q4_affine_tensor,
    write_mlx_mxfp4_affine_tensor,
};

pub(crate) type ReusableExpertBytePool = Arc<Mutex<Vec<ReusableExpertBytes>>>;

const FIXED_Q4_EXPERT_BUFFER_POOL_LIMIT: usize = ACTIVE_EXPERTS_PER_TOKEN * 4;
pub(crate) const PBQ4_EXPERT_MAGIC: &[u8] = b"PBQ4EXPERT ";
pub(crate) const PBQ4_EXPERT_LAYER_FORMAT_V1: &str = "PBQ4EXPERT_LAYER_V1";
pub(crate) const PBQ4_EXPERT_LAYER_FORMAT_V2: &str = "PBQ4EXPERT_LAYER_V2";
pub(crate) const FIXED_Q4_EXPERT_LAYER_FORMAT_V1: &str = "FIXED_Q4_EXPERT_LAYER_V1";
pub(crate) const FIXED_MXFP4_EXPERT_LAYER_FORMAT_V1: &str = "FIXED_MXFP4_EXPERT_LAYER_V1";
pub(crate) const FIXED_DENSE_EXPERT_LAYER_FORMAT_V1: &str = "FIXED_DENSE_EXPERT_LAYER_V1";
pub(crate) const FIXED_DEEPSEEK_GGUF_EXPERT_LAYER_FORMAT_V1: &str =
    "FIXED_DEEPSEEK_GGUF_EXPERT_LAYER_V1";
const EXPERT_COMPONENT_ALIGNMENT: usize = 4096;
pub(crate) struct ReusableExpertBytes {
    // Attachments must drop before backing because they can contain non-owning views of it.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    attachment: OnceLock<Box<dyn Any + Send + Sync>>,
    backing: ReusableExpertBytesBacking,
    len: usize,
}

enum ReusableExpertBytesBacking {
    Heap(Vec<u8>),
    PageAligned(memmap2::MmapMut),
}

impl ReusableExpertBytes {
    fn page_aligned(capacity: usize) -> Result<Self> {
        let mmap = memmap2::MmapMut::map_anon(capacity)
            .context("failed to allocate page-aligned reusable expert buffer")?;
        Ok(Self {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            attachment: OnceLock::new(),
            backing: ReusableExpertBytesBacking::PageAligned(mmap),
            len: 0,
        })
    }

    fn resident_file_slot(file: &fs::File, offset: u64, len: usize) -> Result<Self> {
        if len == 0 {
            bail!("resident expert slot cannot be empty");
        }
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(offset)
                .len(len)
                .map_copy(file)
        }
        .with_context(|| {
            format!("failed to map resident expert slot at offset {offset} with length {len}")
        })?;
        mmap.advise(memmap2::Advice::WillNeed)
            .context("failed to advise resident expert slot pages")?;
        let mut checksum = 0u8;
        for byte in mmap.iter().step_by(4096) {
            checksum ^= *byte;
        }
        checksum ^= mmap[len - 1];
        std::hint::black_box(checksum);
        Ok(Self {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            attachment: OnceLock::new(),
            backing: ReusableExpertBytesBacking::PageAligned(mmap),
            len,
        })
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        match &self.backing {
            ReusableExpertBytesBacking::Heap(bytes) => &bytes[..self.len],
            ReusableExpertBytesBacking::PageAligned(bytes) => &bytes[..self.len],
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match &mut self.backing {
            ReusableExpertBytesBacking::Heap(bytes) => &mut bytes[..self.len],
            ReusableExpertBytesBacking::PageAligned(bytes) => &mut bytes[..self.len],
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        match &self.backing {
            ReusableExpertBytesBacking::Heap(bytes) => bytes.capacity(),
            ReusableExpertBytesBacking::PageAligned(bytes) => bytes.len(),
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn attachment<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.attachment.get()?.downcast_ref()
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn install_attachment<T: Any + Send + Sync>(&self, attachment: T) -> Option<&T> {
        if self.attachment.get().is_none() {
            let _ = self.attachment.set(Box::new(attachment));
        } else {
            drop(attachment);
        }
        self.attachment()
    }

    fn clear(&mut self) {
        if let ReusableExpertBytesBacking::Heap(bytes) = &mut self.backing {
            bytes.clear();
        }
        self.len = 0;
    }

    fn resize_zeroed(&mut self, len: usize) -> Result<()> {
        if len > self.capacity() {
            bail!(
                "reusable expert buffer length {len} exceeds capacity {}",
                self.capacity()
            );
        }
        if let ReusableExpertBytesBacking::Heap(bytes) = &mut self.backing {
            bytes.resize(len, 0);
        }
        self.len = len;
        Ok(())
    }

    fn into_vec(mut self) -> Vec<u8> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(attachment) = self.attachment.take() {
            drop(attachment);
        }
        let backing = std::mem::replace(
            &mut self.backing,
            ReusableExpertBytesBacking::Heap(Vec::new()),
        );
        match backing {
            ReusableExpertBytesBacking::Heap(mut bytes) => {
                bytes.truncate(self.len);
                bytes
            }
            ReusableExpertBytesBacking::PageAligned(bytes) => bytes[..self.len].to_vec(),
        }
    }
}

impl Default for ReusableExpertBytes {
    fn default() -> Self {
        Self::from(Vec::new())
    }
}

impl From<Vec<u8>> for ReusableExpertBytes {
    fn from(bytes: Vec<u8>) -> Self {
        let len = bytes.len();
        Self {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            attachment: OnceLock::new(),
            backing: ReusableExpertBytesBacking::Heap(bytes),
            len,
        }
    }
}

impl Clone for ReusableExpertBytes {
    fn clone(&self) -> Self {
        Self::from(self.as_slice().to_vec())
    }
}

impl std::fmt::Debug for ReusableExpertBytes {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReusableExpertBytes")
            .field("len", &self.len)
            .field("capacity", &self.capacity())
            .field(
                "page_aligned",
                &matches!(self.backing, ReusableExpertBytesBacking::PageAligned(_)),
            )
            .finish()
    }
}

impl PartialEq for ReusableExpertBytes {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<Vec<u8>> for ReusableExpertBytes {
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for ReusableExpertBytes {}

impl std::ops::Deref for ReusableExpertBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertReadPath {
    PositionedRead,
    ResidentMapped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpertIoPolicy {
    pub(crate) expert_read_path: ExpertReadPath,
    pub(crate) application_expert_cache: bool,
    pub(crate) lz4_expert_compression: bool,
    pub(crate) speculative_routing: bool,
    pub(crate) broad_ssd_gpu_overlap: bool,
    pub(crate) layer_ahead_request_staging: bool,
}

// Expert scheduler policy guardrails:
// - read packed experts with positioned reads, not mmap;
// - do not add an application-level expert LRU/cache;
// - do not add LZ4 expert compression;
// - do not speculate future expert routes;
// - avoid broad or speculative SSD/GPU overlap;
// - permit the graph-resolved one-layer-ahead positioned read directly into
//   request-scoped staging for saturated DeepSeek batch geometry.
//
// These choices follow Flash-MoE's "Trust the OS" result: the OS page cache plus
// parallel pread won over custom expert caches, mmap expert files, LZ4, prefetch
// hints, speculative routing, dispatch_io, and aggressive SSD/GPU overlap.
// See https://github.com/danveloper/flash-moe, especially the README "Trust the
// OS" notes and docs/optimization-experiments-q4.md.
pub(crate) const FLASHMOE_EXPERT_IO_POLICY: ExpertIoPolicy = ExpertIoPolicy {
    expert_read_path: ExpertReadPath::PositionedRead,
    application_expert_cache: false,
    lz4_expert_compression: false,
    speculative_routing: false,
    broad_ssd_gpu_overlap: false,
    layer_ahead_request_staging: true,
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

    pub(crate) fn new_fixed_mxfp4(
        layer: usize,
        expert_size: u64,
        experts: usize,
        packs: Vec<ExpertPackMetadata>,
    ) -> Self {
        Self {
            format: FIXED_MXFP4_EXPERT_LAYER_FORMAT_V1.to_string(),
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

    pub(crate) fn new_fixed_deepseek_gguf(
        layer: usize,
        expert_size: u64,
        experts: usize,
        packs: Vec<ExpertPackMetadata>,
    ) -> Self {
        Self {
            format: FIXED_DEEPSEEK_GGUF_EXPERT_LAYER_FORMAT_V1.to_string(),
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
            && self.format != FIXED_MXFP4_EXPERT_LAYER_FORMAT_V1
            && self.format != FIXED_DENSE_EXPERT_LAYER_FORMAT_V1
            && self.format != FIXED_DEEPSEEK_GGUF_EXPERT_LAYER_FORMAT_V1
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
pub(crate) struct PackedExpertTensor {
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
            Self::FixedQ4(spec) => metadata.format == spec.metadata_format(),
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
            Self::FixedQ4(spec) => match spec.encoding {
                FixedQ4ExpertEncoding::AffineBf16 => {
                    ExpertLayerPackMetadata::new_fixed_q4(layer, slot_size, experts, packs)
                }
                FixedQ4ExpertEncoding::MlxMxfp4 => {
                    ExpertLayerPackMetadata::new_fixed_mxfp4(layer, slot_size, experts, packs)
                }
            },
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

#[cfg(test)]
pub(crate) fn fixed_native_q4_aggregate_layout<T: AggregateExpertTensor>(
    aggregate_tensors: &AggregateExpertTensors<'_, T>,
    down: &T,
    layout: AggregateExpertLayout,
) -> Result<Option<QwenMoeQ4ExpertLayout>> {
    Ok(fixed_native_q4_aggregate_spec(aggregate_tensors, down, layout)?.map(|spec| spec.layout))
}

pub(crate) fn fixed_native_q4_aggregate_spec<T: AggregateExpertTensor>(
    aggregate_tensors: &AggregateExpertTensors<'_, T>,
    down: &T,
    layout: AggregateExpertLayout,
) -> Result<Option<FixedQ4ExpertSlotSpec>> {
    if !aggregate_native_q4_enabled(aggregate_tensors, down)? {
        return Ok(None);
    }
    let mxfp4_count = [
        aggregate_tensors.gate.tensor.aggregate_tensor_is_mxfp4(),
        aggregate_tensors.up.tensor.aggregate_tensor_is_mxfp4(),
        down.aggregate_tensor_is_mxfp4(),
    ]
    .into_iter()
    .filter(|mxfp4| *mxfp4)
    .count();
    if mxfp4_count != 0 && mxfp4_count != 3 {
        bail!("aggregate expert tensors must be all MLX MXFP4 or all affine Q4");
    }
    if mxfp4_count == 3 {
        let fixed = QwenMoeQ4ExpertLayout::fixed_mxfp4(layout.hidden, layout.intermediate, 32)?;
        return FixedQ4ExpertSlotSpec::new_mxfp4(fixed, layout.hidden, layout.intermediate)
            .map(Some);
    }
    if !layout.hidden.is_multiple_of(GROUP_SIZE) || !layout.intermediate.is_multiple_of(GROUP_SIZE)
    {
        return Ok(None);
    }
    let fixed = QwenMoeQ4ExpertLayout::fixed_bf16(layout.hidden, layout.intermediate, GROUP_SIZE)?;
    FixedQ4ExpertSlotSpec::new(fixed, layout.hidden, layout.intermediate).map(Some)
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
    spec: FixedQ4ExpertSlotSpec,
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
    let fixed = spec.layout;
    let mut out = vec![0u8; fixed.expert_bytes];
    let mut records = Vec::with_capacity(3);
    write_fixed_native_q4_component(
        &mut out,
        &mut records,
        spec,
        gate,
        QwenMoeExpertComponentKind::GateWeight,
        QwenMoeExpertComponentKind::GateScale,
        QwenMoeExpertComponentKind::GateBias,
    )?;
    write_fixed_native_q4_component(
        &mut out,
        &mut records,
        spec,
        up,
        QwenMoeExpertComponentKind::UpWeight,
        QwenMoeExpertComponentKind::UpScale,
        QwenMoeExpertComponentKind::UpBias,
    )?;
    write_fixed_native_q4_component(
        &mut out,
        &mut records,
        spec,
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
    spec: FixedQ4ExpertSlotSpec,
    input: NativeQ4ExpertRecordInput,
    weight_kind: QwenMoeExpertComponentKind,
    scale_kind: QwenMoeExpertComponentKind,
    bias_kind: QwenMoeExpertComponentKind,
) -> Result<()> {
    let layout = spec.layout;
    match spec.encoding {
        FixedQ4ExpertEncoding::AffineBf16 => {
            let scale_bias_bytes = expert_scale_bias_dtype_size(&input.scale_bias_dtype)?;
            let scale_bias_len = input
                .groups
                .checked_mul(scale_bias_bytes)
                .context("fixed native q4 expert scale/bias byte length overflow")?;
            if input.scale_bytes.len() != scale_bias_len || input.bias_bytes.len() != scale_bias_len
            {
                bail!(
                    "native q4 expert tensor {} scale/bias bytes {}/{} do not match {} groups of {} bytes",
                    input.tensor,
                    input.scale_bytes.len(),
                    input.bias_bytes.len(),
                    input.groups,
                    scale_bias_bytes
                );
            }
        }
        FixedQ4ExpertEncoding::MlxMxfp4 => {
            if !input
                .scale_bias_dtype
                .eq_ignore_ascii_case(EXPERT_SCALE_DTYPE_E8M0)
                || input.scale_bytes.len() != input.groups
                || !input.bias_bytes.is_empty()
            {
                bail!(
                    "native MXFP4 expert tensor {} requires one E8M0 byte per group and no bias bytes; found dtype={} scales={} biases={} groups={}",
                    input.tensor,
                    input.scale_bias_dtype,
                    input.scale_bytes.len(),
                    input.bias_bytes.len(),
                    input.groups
                );
            }
        }
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

pub(crate) fn first_missing_expert_pack_for_shape_from(
    experts_dir: &Path,
    layers: usize,
    experts: usize,
    first_expert_layer: usize,
) -> Result<Option<PathBuf>> {
    if first_expert_layer >= layers {
        bail!("first expert layer {first_expert_layer} must be within {layers} model layers");
    }
    for layer in first_expert_layer..layers {
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

mod store;
pub(crate) use store::*;

mod slots;
pub(crate) use slots::*;

pub(crate) fn take_reusable_expert_bytes(
    pool: &ReusableExpertBytePool,
    min_capacity: usize,
) -> Option<ReusableExpertBytes> {
    let mut pool = pool.lock().expect("fixed Q4 expert byte pool poisoned");
    let index = pool
        .iter()
        .position(|bytes| bytes.capacity() >= min_capacity)?;
    Some(pool.swap_remove(index))
}

pub(crate) fn recycle_reusable_expert_bytes(
    pool: &ReusableExpertBytePool,
    bytes: impl Into<ReusableExpertBytes>,
    min_capacity: usize,
) {
    let mut bytes = bytes.into();
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
pub(crate) struct ExpertSlotDescriptor {
    pub(crate) layer: usize,
    pub(crate) expert: usize,
    pub(crate) slot_offset: u64,
    pub(crate) slot_capacity: usize,
    pub(crate) payload_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpertSlotView<'a> {
    descriptor: ExpertSlotDescriptor,
    payload: &'a [u8],
}

impl<'a> ExpertSlotView<'a> {
    pub(crate) fn new(
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

    pub(crate) fn descriptor(&self) -> ExpertSlotDescriptor {
        self.descriptor
    }

    pub(crate) fn payload(&self) -> &'a [u8] {
        self.payload
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FixedQ4ExpertSlotView<'a> {
    slot: ExpertSlotView<'a>,
    layout: QwenMoeQ4ExpertLayout,
}

#[allow(dead_code)]
impl<'a> FixedQ4ExpertSlotView<'a> {
    pub(crate) fn new(slot: ExpertSlotView<'a>, layout: QwenMoeQ4ExpertLayout) -> Result<Self> {
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

    pub(crate) fn descriptor(&self) -> ExpertSlotDescriptor {
        self.slot.descriptor()
    }

    pub(crate) fn layout(&self) -> QwenMoeQ4ExpertLayout {
        self.layout
    }

    pub(crate) fn payload(&self) -> &'a [u8] {
        self.slot.payload()
    }

    pub(crate) fn component(&self, kind: QwenMoeExpertComponentKind) -> &'a [u8] {
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
pub(crate) struct ReusableExpertBuffer {
    bytes: ReusableExpertBytes,
}

impl ReusableExpertBuffer {
    pub(crate) fn prepare_payload(
        &mut self,
        slot_capacity: usize,
        payload_len: usize,
    ) -> Result<&mut [u8]> {
        if payload_len > slot_capacity {
            bail!("expert payload length {payload_len} exceeds slot capacity {slot_capacity}");
        }
        if self.bytes.capacity() < slot_capacity {
            self.bytes = if payload_len == slot_capacity && slot_capacity > 0 {
                ReusableExpertBytes::page_aligned(slot_capacity)?
            } else {
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(slot_capacity)
                    .context("failed to reserve reusable expert buffer")?;
                ReusableExpertBytes::from(bytes)
            };
        }
        self.bytes.resize_zeroed(payload_len)?;
        Ok(self.bytes.as_mut_slice())
    }

    #[cfg(test)]
    pub(crate) fn slot_view(
        &self,
        layer: usize,
        expert: usize,
        slot_offset: u64,
        slot_capacity: usize,
    ) -> Result<ExpertSlotView<'_>> {
        ExpertSlotView::new(
            layer,
            expert,
            slot_offset,
            slot_capacity,
            self.bytes.as_slice(),
        )
    }

    pub(crate) fn take_payload(&mut self) -> ReusableExpertBytes {
        std::mem::take(&mut self.bytes)
    }

    pub(crate) fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    pub(crate) fn adopt_buffer(&mut self, mut bytes: ReusableExpertBytes) -> ReusableExpertBytes {
        bytes.clear();
        std::mem::replace(&mut self.bytes, bytes)
    }
}

mod packing;
pub(in crate::inference::flashmoe) use packing::*;

#[cfg(test)]
#[path = "parity_tests.rs"]
mod parity_tests;

#[cfg(test)]
#[path = "../tests/experts.rs"]
mod tests;
