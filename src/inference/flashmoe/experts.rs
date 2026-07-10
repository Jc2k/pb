use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
#[cfg(not(unix))]
use std::io::{Read, Seek};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use super::math::q4_fma_matvec_with_group_size;
use super::model_family::{
    QwenMoeExpertComponentKind, QwenMoeExpertComponentLayout, QwenMoeModelLayout,
    QwenMoeQ4ExpertLayout,
};
#[cfg(test)]
use super::types::HIDDEN_DIM;
use super::types::{ACTIVE_EXPERTS_PER_TOKEN, GROUP_SIZE};

pub type ReusableExpertBytePool = Arc<Mutex<Vec<Vec<u8>>>>;

const FIXED_Q4_EXPERT_BUFFER_POOL_LIMIT: usize = ACTIVE_EXPERTS_PER_TOKEN * 4;
pub(crate) const PBQ4_EXPERT_MAGIC: &[u8] = b"PBQ4EXPERT ";
pub(crate) const PBQ4_EXPERT_LAYER_FORMAT_V1: &str = "PBQ4EXPERT_LAYER_V1";
pub(crate) const PBQ4_EXPERT_LAYER_FORMAT_V2: &str = "PBQ4EXPERT_LAYER_V2";
pub(crate) const FIXED_Q4_EXPERT_LAYER_FORMAT_V1: &str = "FIXED_Q4_EXPERT_LAYER_V1";
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

    pub(crate) fn pack_for(&self, expert: usize) -> Option<&ExpertPackMetadata> {
        self.packs.iter().find(|metadata| metadata.expert == expert)
    }

    pub(crate) fn validate(&self, path: &Path, layer: usize) -> Result<()> {
        if self.format != PBQ4_EXPERT_LAYER_FORMAT_V1
            && self.format != PBQ4_EXPERT_LAYER_FORMAT_V2
            && self.format != FIXED_Q4_EXPERT_LAYER_FORMAT_V1
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
            scales: &self.scales[..needed_groups],
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

pub(crate) fn decode_expert_scale_bias_bytes(
    bytes: &[u8],
    len: usize,
    dtype: &str,
) -> Result<Vec<f32>> {
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
    fixed_q4: FixedQ4ExpertSlotSpec,
    fixed_q4_buffer_pool: ReusableExpertBytePool,
}

#[derive(Debug, Clone)]
pub(crate) struct ExpertSlotStore {
    root: PathBuf,
    fixed_q4: FixedQ4ExpertSlotSpec,
    fixed_q4_buffer_pool: ReusableExpertBytePool,
    layers: Arc<Mutex<BTreeMap<usize, Arc<ExpertLayerReader>>>>,
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

impl ExpertSlotStore {
    #[cfg(test)]
    pub(crate) fn open(root: PathBuf) -> Result<Self> {
        Self::open_with_fixed_q4(root, FixedQ4ExpertSlotSpec::qwen35_a17b()?)
    }

    pub(crate) fn open_with_model_layout(
        root: PathBuf,
        layout: &QwenMoeModelLayout,
    ) -> Result<Self> {
        Self::open_with_fixed_q4(root, FixedQ4ExpertSlotSpec::from_model_layout(layout)?)
    }

    pub(crate) fn open_with_fixed_q4(
        root: PathBuf,
        fixed_q4: FixedQ4ExpertSlotSpec,
    ) -> Result<Self> {
        if !root.is_dir() {
            bail!("expert store {} does not exist", root.display());
        }
        Ok(Self {
            root,
            fixed_q4,
            fixed_q4_buffer_pool: Arc::new(Mutex::new(Vec::new())),
            layers: Arc::new(Mutex::new(BTreeMap::new())),
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
            self.fixed_q4,
            Arc::clone(&self.fixed_q4_buffer_pool),
        )?);
        let mut layers = self.layers.lock().expect("expert layer cache poisoned");
        Ok(layers.entry(layer).or_insert_with(|| reader).clone())
    }
}

impl ExpertLayerReader {
    pub(crate) fn open(
        root: &Path,
        layer: usize,
        fixed_q4: FixedQ4ExpertSlotSpec,
        fixed_q4_buffer_pool: ReusableExpertBytePool,
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
            fixed_q4,
            fixed_q4_buffer_pool,
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
            && let Some(bytes) =
                take_reusable_expert_bytes(&self.fixed_q4_buffer_pool, plan.slot_capacity)
        {
            let previous = scratch.adopt_buffer(bytes);
            recycle_reusable_expert_bytes(
                &self.fixed_q4_buffer_pool,
                previous,
                self.fixed_q4.layout.expert_bytes,
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
            FixedQ4ExpertSlotView::new(slot, self.fixed_q4.layout).with_context(|| {
                format!(
                    "expert {} is neither a PBQ4 pack nor a fixed Q4 slot matching the model layout",
                    self.path.display()
                )
            })?;
            ExpertRawPayload::FixedQ4(FixedQ4ExpertPayload::from_whole_slot(
                self.fixed_q4,
                scratch.take_payload(),
                Some(Arc::clone(&self.fixed_q4_buffer_pool)),
            )?)
        };
        Ok(ExpertRawRead {
            layer: self.metadata.layer,
            expert,
            slot: descriptor,
            metadata: plan.metadata,
            fixed_q4: self.fixed_q4,
            recycle_pool: Some(Arc::clone(&self.fixed_q4_buffer_pool)),
            payload,
            read_latency,
            read_path: FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ExpertReadPlan {
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
    pub(crate) metadata: ExpertPackMetadata,
    pub(crate) fixed_q4: FixedQ4ExpertSlotSpec,
    pub(crate) recycle_pool: Option<ReusableExpertBytePool>,
    pub(crate) payload: ExpertRawPayload,
    pub(crate) read_latency: Duration,
    pub(crate) read_path: ExpertReadPath,
}

#[derive(Debug)]
pub(crate) enum ExpertRawPayload {
    Pbq4(Vec<u8>),
    FixedQ4(FixedQ4ExpertPayload),
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
        Self::new(
            layout.q4_expert_layout,
            layout.hidden_size,
            layout.moe_intermediate_size,
        )
    }
}

#[derive(Debug)]
pub(crate) struct FixedQ4ExpertPayload {
    pub(crate) spec: FixedQ4ExpertSlotSpec,
    pub(crate) bytes: Vec<u8>,
    pub(crate) decoded: Option<FixedQ4ExpertPayloadDecoded>,
    pub(crate) recycle_pool: Option<ReusableExpertBytePool>,
}

impl Clone for FixedQ4ExpertPayload {
    fn clone(&self) -> Self {
        Self {
            spec: self.spec,
            bytes: self.bytes.clone(),
            decoded: self.decoded.clone(),
            recycle_pool: None,
        }
    }
}

impl PartialEq for FixedQ4ExpertPayload {
    fn eq(&self, other: &Self) -> bool {
        self.spec == other.spec && self.bytes == other.bytes && self.decoded == other.decoded
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
            decoded: None,
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

    pub(crate) fn project_cpu(
        &self,
        projection: FixedQ4ExpertProjection,
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
        projection: FixedQ4ExpertProjection,
        input_len: usize,
        output_width: usize,
    ) -> Option<Q4MatvecPayload<'_>> {
        if input_len == 0 || output_width == 0 {
            return None;
        }
        let (rows, cols) = match projection {
            FixedQ4ExpertProjection::Gate | FixedQ4ExpertProjection::Up => (
                self.spec.intermediate_size.min(output_width).max(1),
                self.spec.hidden_size.min(input_len).max(1),
            ),
            FixedQ4ExpertProjection::Down => (
                self.spec.hidden_size.min(output_width).max(1),
                self.spec.intermediate_size.min(input_len).max(1),
            ),
        };
        let groups_per_row = cols.div_ceil(self.spec.layout.group_size).max(1);
        let needed_groups = rows.checked_mul(groups_per_row)?;
        let needed_packed = rows.checked_mul(cols.div_ceil(2))?;
        let decoded = self.decoded.as_ref();
        let (packed, scale_bytes, bias_bytes, scales, biases, source) = match projection {
            FixedQ4ExpertProjection::Gate => (
                self.component(QwenMoeExpertComponentKind::GateWeight),
                self.component(QwenMoeExpertComponentKind::GateScale),
                self.component(QwenMoeExpertComponentKind::GateBias),
                decoded
                    .map(|decoded| decoded.gate_scales.as_slice())
                    .unwrap_or(&[]),
                decoded
                    .map(|decoded| decoded.gate_biases.as_slice())
                    .unwrap_or(&[]),
                self.component_source(
                    QwenMoeExpertComponentKind::GateWeight,
                    QwenMoeExpertComponentKind::GateScale,
                    QwenMoeExpertComponentKind::GateBias,
                ),
            ),
            FixedQ4ExpertProjection::Up => (
                self.component(QwenMoeExpertComponentKind::UpWeight),
                self.component(QwenMoeExpertComponentKind::UpScale),
                self.component(QwenMoeExpertComponentKind::UpBias),
                decoded
                    .map(|decoded| decoded.up_scales.as_slice())
                    .unwrap_or(&[]),
                decoded
                    .map(|decoded| decoded.up_biases.as_slice())
                    .unwrap_or(&[]),
                self.component_source(
                    QwenMoeExpertComponentKind::UpWeight,
                    QwenMoeExpertComponentKind::UpScale,
                    QwenMoeExpertComponentKind::UpBias,
                ),
            ),
            FixedQ4ExpertProjection::Down => (
                self.component(QwenMoeExpertComponentKind::DownWeight),
                self.component(QwenMoeExpertComponentKind::DownScale),
                self.component(QwenMoeExpertComponentKind::DownBias),
                decoded
                    .map(|decoded| decoded.down_scales.as_slice())
                    .unwrap_or(&[]),
                decoded
                    .map(|decoded| decoded.down_biases.as_slice())
                    .unwrap_or(&[]),
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
            || (!scales.is_empty() && scales.len() < needed_groups)
            || (!biases.is_empty() && biases.len() < needed_groups)
        {
            return None;
        }
        Some(Q4MatvecPayload {
            rows,
            cols,
            group_size: self.spec.layout.group_size,
            packed: &packed[..needed_packed],
            scales: if scales.is_empty() {
                &[]
            } else {
                &scales[..needed_groups]
            },
            biases: if biases.is_empty() {
                &[]
            } else {
                &biases[..needed_groups]
            },
            scale_bias_groups: needed_groups,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16,
            scale_bytes: &scale_bytes[..needed_groups * 2],
            bias_bytes: &bias_bytes[..needed_groups * 2],
            source: Some(source),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FixedQ4ExpertProjection {
    Gate,
    Up,
    Down,
}

impl FixedQ4ExpertProjection {
    fn scale_bias_kinds(self) -> (QwenMoeExpertComponentKind, QwenMoeExpertComponentKind) {
        match self {
            FixedQ4ExpertProjection::Gate => (
                QwenMoeExpertComponentKind::GateScale,
                QwenMoeExpertComponentKind::GateBias,
            ),
            FixedQ4ExpertProjection::Up => (
                QwenMoeExpertComponentKind::UpScale,
                QwenMoeExpertComponentKind::UpBias,
            ),
            FixedQ4ExpertProjection::Down => (
                QwenMoeExpertComponentKind::DownScale,
                QwenMoeExpertComponentKind::DownBias,
            ),
        }
    }
}

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
    pub(crate) scales: &'a [f32],
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FixedQ4ExpertPayloadDecoded {
    pub(crate) gate_scales: Vec<f32>,
    pub(crate) gate_biases: Vec<f32>,
    pub(crate) up_scales: Vec<f32>,
    pub(crate) up_biases: Vec<f32>,
    pub(crate) down_scales: Vec<f32>,
    pub(crate) down_biases: Vec<f32>,
}

impl FixedQ4ExpertPayloadDecoded {
    #[cfg(test)]
    pub(crate) fn from_slot(
        view: &FixedQ4ExpertSlotView<'_>,
        spec: FixedQ4ExpertSlotSpec,
    ) -> Result<Self> {
        Ok(Self {
            gate_scales: decode_fixed_q4_bf16_component(
                view,
                QwenMoeExpertComponentKind::GateScale,
            )?,
            gate_biases: decode_fixed_q4_bf16_component(
                view,
                QwenMoeExpertComponentKind::GateBias,
            )?,
            up_scales: decode_fixed_q4_bf16_component(view, QwenMoeExpertComponentKind::UpScale)?,
            up_biases: decode_fixed_q4_bf16_component(view, QwenMoeExpertComponentKind::UpBias)?,
            down_scales: decode_fixed_q4_bf16_component(
                view,
                QwenMoeExpertComponentKind::DownScale,
            )?,
            down_biases: decode_fixed_q4_bf16_component(
                view,
                QwenMoeExpertComponentKind::DownBias,
            )?,
        })
        .and_then(|decoded| {
            let groups_per_gate_row = spec.hidden_size.div_ceil(spec.layout.group_size);
            let groups_per_down_row = spec.intermediate_size.div_ceil(spec.layout.group_size);
            let gate_groups = spec.intermediate_size * groups_per_gate_row;
            let down_groups = spec.hidden_size * groups_per_down_row;
            if decoded.gate_scales.len() < gate_groups
                || decoded.gate_biases.len() < gate_groups
                || decoded.up_scales.len() < gate_groups
                || decoded.up_biases.len() < gate_groups
                || decoded.down_scales.len() < down_groups
                || decoded.down_biases.len() < down_groups
            {
                bail!("fixed Q4 expert scale/bias payload is shorter than model layout requires");
            }
            Ok(decoded)
        })
    }
}

#[cfg(test)]
fn decode_fixed_q4_bf16_component(
    view: &FixedQ4ExpertSlotView<'_>,
    kind: QwenMoeExpertComponentKind,
) -> Result<Vec<f32>> {
    decode_fixed_q4_bf16_component_bytes(view.component(kind))
}

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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn fixed_q4_payload_resolves_typed_matvec_offsets_from_whole_slot() {
        let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
        let payload = FixedQ4ExpertPayload::from_whole_slot(spec, (0..45).collect(), None).unwrap();

        let gate = payload
            .matvec_payload(FixedQ4ExpertProjection::Gate, 2, 2)
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
            .matvec_payload(FixedQ4ExpertProjection::Up, 2, 2)
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
                .matvec_payload(FixedQ4ExpertProjection::Down, 2, 2)
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

        let reader =
            ExpertLayerReader::open(tmp.path(), 0, spec, Arc::new(Mutex::new(Vec::new()))).unwrap();
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

        let reader =
            ExpertLayerReader::open(tmp.path(), 0, spec, Arc::new(Mutex::new(Vec::new()))).unwrap();
        let plan = reader.prepare_read(0).unwrap();
        let mut scratch = ReusableExpertBuffer::default();
        let raw = reader.read_prepared_into(0, plan, &mut scratch).unwrap();

        match raw.payload {
            ExpertRawPayload::Pbq4(bytes) => assert_eq!(bytes, payload),
            ExpertRawPayload::FixedQ4(_) => panic!("PBQ4 slot classified as fixed Q4"),
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
            ExpertLayerReader::open(tmp.path(), 0, spec, Arc::new(Mutex::new(Vec::new()))).unwrap(),
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
    fn reusable_expert_byte_pool_reuses_capacity_qualified_buffers() {
        let pool: ReusableExpertBytePool = Arc::new(Mutex::new(Vec::new()));
        recycle_reusable_expert_bytes(&pool, Vec::with_capacity(64), 64);

        let returned = take_reusable_expert_bytes(&pool, 32).unwrap();

        assert!(returned.capacity() >= 64);
        assert!(pool.lock().unwrap().is_empty());
    }
}
