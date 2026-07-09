use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
#[cfg(not(unix))]
use std::io::{Read, Seek};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::model_family::{
    QwenMoeExpertComponentKind, QwenMoeExpertComponentLayout, QwenMoeModelLayout,
    QwenMoeQ4ExpertLayout,
};
use super::types::{ACTIVE_EXPERTS_PER_TOKEN, HIDDEN_DIM};

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

impl ExpertSlotStore {
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
