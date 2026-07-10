#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::ffi::c_void;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::Arc;
use std::time::Duration;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::scheduler::{
    ScheduledCmd3MetalPostAttentionInput, ScheduledExpertPhaseMlpPayload,
    ScheduledQ4ExpertPhaseMlpPayload, ScheduledRoutingCandidateSource, ScheduledRoutingCommand,
    ScheduledSharedExpertPhaseRef,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::state::{FlashMoeCmd3OutputState, FlashMoePostAttentionPrepState};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::weights::{SharedExpertPhaseQ4Projections, SharedExpertPhaseWeights};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) type MetalObjcId = *mut c_void;

const DEFAULT_METAL_ATTENTION_CPU_MAX_TOKENS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalAttentionBackend {
    Cpu,
    Gpu,
}

#[derive(Debug, Default)]
pub(crate) struct MetalAttentionPolicy;

impl MetalAttentionPolicy {
    pub(crate) fn backend(&self, tokens: usize) -> MetalAttentionBackend {
        if tokens > DEFAULT_METAL_ATTENTION_CPU_MAX_TOKENS {
            MetalAttentionBackend::Gpu
        } else {
            MetalAttentionBackend::Cpu
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalLinearAttentionStaticOffsets {
    pub(crate) conv_weight_byte_offset: u64,
    pub(crate) a_log_byte_offset: u64,
    pub(crate) dt_bias_byte_offset: u64,
    pub(crate) norm_weight_byte_offset: u64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalLinearAttentionStaticOffsets {
    pub(crate) fn new(
        conv_weight_byte_offset: u64,
        a_log_byte_offset: u64,
        dt_bias_byte_offset: u64,
        norm_weight_byte_offset: u64,
    ) -> Self {
        Self {
            conv_weight_byte_offset,
            a_log_byte_offset,
            dt_bias_byte_offset,
            norm_weight_byte_offset,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) enum MetalBatchProjectionInput<'a> {
    Cpu(&'a [f32]),
    Buffer { buffer: MetalObjcId, len: usize },
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalBatchProjectionInput<'_> {
    pub(crate) fn len(self) -> usize {
        match self {
            Self::Cpu(input) => input.len(),
            Self::Buffer { len, .. } => len,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalProjectionBatch {
    pub(crate) output_buffer: MetalObjcId,
    pub(crate) output_offsets: Vec<usize>,
    pub(crate) output_widths: Vec<usize>,
    pub(crate) total_rows: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalProjectionBatch {
    pub(crate) fn new(
        output_buffer: MetalObjcId,
        output_offsets: Vec<usize>,
        output_widths: Vec<usize>,
        total_rows: usize,
    ) -> Self {
        Self {
            output_buffer,
            output_offsets,
            output_widths,
            total_rows,
        }
    }

    pub(crate) fn empty() -> Self {
        Self::new(std::ptr::null_mut(), Vec::new(), Vec::new(), 0)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalAttentionValues {
    pub(crate) buffer: MetalObjcId,
    pub(crate) len: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalAttentionValues {
    pub(crate) fn new(buffer: MetalObjcId, len: usize) -> Self {
        Self { buffer, len }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalQ4SourceBufferKey {
    ptr: usize,
    len: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
struct MetalQ4SourceBufferEntry {
    key: MetalQ4SourceBufferKey,
    buffer: MetalObjcId,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Default)]
pub(crate) struct MetalQ4SourceBufferCache {
    entries: Vec<MetalQ4SourceBufferEntry>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalQ4SourceBufferCache {
    fn key_for(bytes: &[u8]) -> MetalQ4SourceBufferKey {
        MetalQ4SourceBufferKey {
            ptr: bytes.as_ptr() as usize,
            len: bytes.len(),
        }
    }

    pub(crate) fn get(&self, bytes: &[u8]) -> Option<MetalObjcId> {
        let key = Self::key_for(bytes);
        self.entries
            .iter()
            .find_map(|entry| (entry.key == key).then_some(entry.buffer))
    }

    pub(crate) fn insert(&mut self, bytes: &[u8], buffer: MetalObjcId) {
        let key = Self::key_for(bytes);
        if self.entries.iter().any(|entry| entry.key == key) {
            return;
        }
        self.entries.push(MetalQ4SourceBufferEntry { key, buffer });
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalKvCacheInner {
    pub(crate) keys: MetalObjcId,
    pub(crate) values: MetalObjcId,
    layers: Vec<MetalKvLayer>,
    pub(crate) max_context: usize,
    total_items: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalKvCacheInner {
    pub(crate) fn new(
        keys: MetalObjcId,
        values: MetalObjcId,
        widths: &[usize],
        max_context: usize,
    ) -> anyhow::Result<Self> {
        let mut offset = 0usize;
        let mut layers = Vec::with_capacity(widths.len());
        for width in widths.iter().copied() {
            layers.push(MetalKvLayer { offset, width });
            offset = offset
                .checked_add(width.saturating_mul(max_context))
                .ok_or_else(|| anyhow::anyhow!("Metal KV layer offset overflow"))?;
        }
        Ok(Self {
            keys,
            values,
            layers,
            max_context,
            total_items: offset,
        })
    }

    pub(crate) fn layer(&self, layer: usize) -> anyhow::Result<MetalKvLayer> {
        let layer = self
            .layers
            .get(layer)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Metal KV cache has no layer {layer}"))?;
        if layer.width == 0 {
            anyhow::bail!("Metal KV cache layer is not a full-attention layer");
        }
        if layer
            .offset
            .saturating_add(layer.width.saturating_mul(self.max_context))
            > self.total_items
        {
            anyhow::bail!("Metal KV cache layer range exceeds allocation");
        }
        Ok(layer)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalKvLayer {
    pub(crate) offset: usize,
    pub(crate) width: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalLinearAttentionStateCache {
    pub(crate) layers: Vec<Option<MetalLinearAttentionLayerState>>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalLinearAttentionStateCache {
    pub(crate) fn new(layers: Vec<Option<MetalLinearAttentionLayerState>>) -> Self {
        Self { layers }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalLinearAttentionLayerState {
    pub(crate) conv_state: MetalObjcId,
    pub(crate) ssm_state: MetalObjcId,
    pub(crate) conv_output: MetalObjcId,
    pub(crate) delta_output: MetalObjcId,
    pub(crate) g_decay: MetalObjcId,
    pub(crate) beta_gate: MetalObjcId,
    pub(crate) conv_state_len: usize,
    pub(crate) ssm_state_len: usize,
    pub(crate) conv_dim: usize,
    pub(crate) total_value_width: usize,
    pub(crate) num_value_heads: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalLinearAttentionLayerState {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        conv_state: MetalObjcId,
        ssm_state: MetalObjcId,
        conv_output: MetalObjcId,
        delta_output: MetalObjcId,
        g_decay: MetalObjcId,
        beta_gate: MetalObjcId,
        conv_state_len: usize,
        ssm_state_len: usize,
        conv_dim: usize,
        total_value_width: usize,
        num_value_heads: usize,
    ) -> Self {
        Self {
            conv_state,
            ssm_state,
            conv_output,
            delta_output,
            g_decay,
            beta_gate,
            conv_state_len,
            ssm_state_len,
            conv_dim,
            total_value_width,
            num_value_heads,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) const METAL_REUSABLE_BUFFER_POOL_LIMIT: usize = 64;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalReusableBuffer {
    pub(crate) id: MetalObjcId,
    pub(crate) len: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalReusableBuffer {
    pub(crate) fn new(id: MetalObjcId, len: usize) -> Self {
        Self { id, len }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalLmHeadBuffer {
    pub(crate) weights: MetalObjcId,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) bytes: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalLmHeadBuffer {
    pub(crate) fn new(weights: MetalObjcId, rows: usize, cols: usize, bytes: usize) -> Self {
        Self {
            weights,
            rows,
            cols,
            bytes,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalLmHeadBufferCache {
    buffers: BTreeMap<String, MetalLmHeadBuffer>,
    bytes: usize,
    max_bytes: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalLmHeadBufferCache {
    pub(crate) fn with_budget(max_bytes: usize) -> Self {
        Self {
            buffers: BTreeMap::new(),
            bytes: 0,
            max_bytes,
        }
    }

    pub(crate) fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub(crate) fn get(&self, key: &str, rows: usize, cols: usize) -> Option<MetalLmHeadBuffer> {
        let buffer = self.buffers.get(key).copied()?;
        (buffer.rows == rows && buffer.cols == cols).then_some(buffer)
    }

    pub(crate) fn insert(
        &mut self,
        key: String,
        buffer: MetalLmHeadBuffer,
        mut release_evicted: impl FnMut(MetalLmHeadBuffer),
    ) -> Option<MetalLmHeadBuffer> {
        if buffer.bytes > self.max_bytes {
            return Some(buffer);
        }
        while self.bytes.saturating_add(buffer.bytes) > self.max_bytes && !self.buffers.is_empty() {
            let Some(victim) = self.buffers.keys().next().cloned() else {
                break;
            };
            if let Some(previous) = self.buffers.remove(&victim) {
                self.bytes = self.bytes.saturating_sub(previous.bytes);
                release_evicted(previous);
            }
        }
        let previous = self.buffers.insert(key, buffer);
        if let Some(previous_buffer) = previous.as_ref() {
            self.bytes = self.bytes.saturating_sub(previous_buffer.bytes);
        }
        self.bytes = self.bytes.saturating_add(buffer.bytes);
        previous
    }

    pub(crate) fn release_all(&mut self, mut release: impl FnMut(MetalLmHeadBuffer)) {
        for (_, buffer) in std::mem::take(&mut self.buffers) {
            release(buffer);
        }
        self.bytes = 0;
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalDenseWeights {
    pub(crate) buffer: MetalObjcId,
    _mmap: Arc<memmap2::Mmap>,
    pub(crate) len: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalDenseWeights {
    pub(crate) fn new(buffer: MetalObjcId, mmap: Arc<memmap2::Mmap>, len: usize) -> Self {
        Self {
            buffer,
            _mmap: mmap,
            len,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalSharedExpertBuffers {
    pub(crate) gate: MetalObjcId,
    pub(crate) up: MetalObjcId,
    pub(crate) down: MetalObjcId,
    pub(crate) router: MetalObjcId,
    pub(crate) width: usize,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) total_intermediate: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalSharedExpertBuffers {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        gate: MetalObjcId,
        up: MetalObjcId,
        down: MetalObjcId,
        router: MetalObjcId,
        width: usize,
        shared_experts: usize,
        intermediate: usize,
        total_intermediate: usize,
    ) -> Self {
        Self {
            gate,
            up,
            down,
            router,
            width,
            shared_experts,
            intermediate,
            total_intermediate,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalDispatchSize {
    pub(crate) width: u64,
    pub(crate) height: u64,
    pub(crate) depth: u64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalDispatchSize {
    pub(crate) const fn new(width: u64, height: u64, depth: u64) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalDispatchMode {
    Threads,
    Threadgroups,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalDispatchPlan {
    pub(crate) mode: MetalDispatchMode,
    pub(crate) grid: MetalDispatchSize,
    pub(crate) threadgroup: MetalDispatchSize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalDispatchPlan {
    pub(crate) fn threads(threads: u64) -> Self {
        Self {
            mode: MetalDispatchMode::Threads,
            grid: MetalDispatchSize::new(threads, 1, 1),
            threadgroup: MetalDispatchSize::new(threads.clamp(1, 64), 1, 1),
        }
    }

    pub(crate) fn q4_threadgroups(rows: u64) -> Self {
        const Q4_ROWS_PER_THREADGROUP: u64 = 8;
        Self {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(rows.div_ceil(Q4_ROWS_PER_THREADGROUP).max(1), 1, 1),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    }

    pub(crate) fn q4_mmap_threadgroups(rows: u64) -> Self {
        const Q4_MMAP_ROWS_PER_THREADGROUP: u64 = 16;
        Self {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(rows.div_ceil(Q4_MMAP_ROWS_PER_THREADGROUP).max(1), 1, 1),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    }

    pub(crate) fn single_threadgroup(threads: u64) -> Self {
        Self {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(1, 1, 1),
            threadgroup: MetalDispatchSize::new(threads.clamp(1, 256), 1, 1),
        }
    }
}

const DEFAULT_FLASHMOE_METAL_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_FLASHMOE_METAL_COMMAND_POLL_INTERVAL: Duration = Duration::from_millis(2);

pub(crate) mod kernels {
    pub(crate) const Q4_FMA_MATVEC: &str = "q4_fma_matvec";
    pub(crate) const Q4_FMA_MATVEC_BF16_SCALE_BIAS: &str = "q4_fma_matvec_bf16_scale_bias";
    pub(crate) const Q4_SWIGLU_FUSED: &str = "q4_swiglu_fused";
    pub(crate) const Q4_SWIGLU_FUSED_BF16_SCALE_BIAS: &str = "q4_swiglu_fused_bf16_scale_bias";
    pub(crate) const Q4_MMAP_FMA_MATVEC: &str = "q4_mmap_fma_matvec";
    pub(crate) const Q4_MMAP_FMA_MATVEC_BF16_SCALE_BIAS: &str =
        "q4_mmap_fma_matvec_bf16_scale_bias";
    pub(crate) const Q4_MMAP_FMA_MATVEC_BATCH: &str = "q4_mmap_fma_matvec_batch";
    pub(crate) const Q4_MMAP_FMA_MATVEC_BATCH_BF16_SCALE_BIAS: &str =
        "q4_mmap_fma_matvec_batch_bf16_scale_bias";
    pub(crate) const ROUTE_TOP4: &str = "route_top4";
    pub(crate) const DENSE_MATVEC: &str = "dense_matvec";
    pub(crate) const DENSE_MATVEC_BF16: &str = "dense_matvec_bf16";
    pub(crate) const DENSE_MMAP_MATVEC_F32: &str = "dense_mmap_matvec_f32";
    pub(crate) const DENSE_MMAP_MATVEC_BF16: &str = "dense_mmap_matvec_bf16";
    pub(crate) const DENSE_MMAP_MATVEC_BF16_SIMD: &str = "dense_mmap_matvec_bf16_simd";
    pub(crate) const RMS_NORM: &str = "rms_norm";
    pub(crate) const RMS_NORM_REDUCED: &str = "rms_norm_reduced";
    pub(crate) const RESIDUAL_ADD_RMS_NORM: &str = "residual_add_rms_norm";
    pub(crate) const ROPE_APPLY: &str = "rope_apply";
    pub(crate) const ROPE_SPLIT_HALF_APPLY: &str = "rope_split_half_apply";
    pub(crate) const ATTENTION_SCORES: &str = "attention_scores";
    pub(crate) const KV_CACHE_WRITE: &str = "kv_cache_write";
    pub(crate) const KV_CACHE_READ_ATTENTION: &str = "kv_cache_read_attention";
    pub(crate) const EXPERT_MLP_FUSED: &str = "expert_mlp_fused";
    pub(crate) const SILU_PRODUCT: &str = "silu_product";
    pub(crate) const SHARED_EXPERT_ACTIVATION: &str = "shared_expert_activation";
    pub(crate) const COMBINE_EXPERT_PHASE: &str = "combine_expert_phase";
    pub(crate) const FILL_ZERO: &str = "fill_zero";
    pub(crate) const LM_HEAD_LOGITS: &str = "lm_head_logits";
    pub(crate) const TOPK_VOCAB: &str = "topk_vocab";
    pub(crate) const GQA_ATTENTION_SCORES: &str = "gqa_attention_scores";
    pub(crate) const GQA_KV_READ_ATTENTION: &str = "gqa_kv_read_attention";
    pub(crate) const LINEAR_CONV1D_STEP: &str = "linear_conv1d_step";
    pub(crate) const LINEAR_RMS_NORM_QK: &str = "linear_rms_norm_qk";
    pub(crate) const LINEAR_COMPUTE_DECAY_BETA: &str = "linear_compute_decay_beta";
    pub(crate) const LINEAR_GATED_DELTA_STEP: &str = "linear_gated_delta_step";
    pub(crate) const LINEAR_GATED_RMS_NORM: &str = "linear_gated_rms_norm";
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalRuntimeCapabilities {
    kernels: BTreeSet<&'static str>,
}

impl MetalRuntimeCapabilities {
    pub(crate) fn from_pipeline_names(names: MetalPipelineNameSet) -> Self {
        Self {
            kernels: names.kernel_names().into_iter().collect(),
        }
    }

    pub(crate) fn supports(&self, kernel: &str) -> bool {
        self.kernels.contains(kernel)
    }

    #[cfg(test)]
    pub(crate) fn empty_for_test() -> Self {
        Self {
            kernels: BTreeSet::new(),
        }
    }

    pub(crate) fn require_all(&self, kernels: &[&'static str]) -> anyhow::Result<()> {
        let missing = kernels
            .iter()
            .copied()
            .filter(|kernel| !self.supports(kernel))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            anyhow::bail!("missing Metal kernels: {}", missing.join(", "));
        }
        Ok(())
    }
}

#[cfg(test)]
const REQUIRED_FORWARD_KERNELS: &[&str] = &[
    kernels::Q4_FMA_MATVEC,
    kernels::Q4_FMA_MATVEC_BF16_SCALE_BIAS,
    kernels::Q4_SWIGLU_FUSED,
    kernels::Q4_SWIGLU_FUSED_BF16_SCALE_BIAS,
    kernels::Q4_MMAP_FMA_MATVEC,
    kernels::Q4_MMAP_FMA_MATVEC_BF16_SCALE_BIAS,
    kernels::Q4_MMAP_FMA_MATVEC_BATCH,
    kernels::Q4_MMAP_FMA_MATVEC_BATCH_BF16_SCALE_BIAS,
    kernels::ROUTE_TOP4,
    kernels::DENSE_MATVEC,
    kernels::DENSE_MATVEC_BF16,
    kernels::DENSE_MMAP_MATVEC_F32,
    kernels::DENSE_MMAP_MATVEC_BF16,
    kernels::DENSE_MMAP_MATVEC_BF16_SIMD,
    kernels::RMS_NORM,
    kernels::RMS_NORM_REDUCED,
    kernels::RESIDUAL_ADD_RMS_NORM,
    kernels::ROPE_APPLY,
    kernels::ROPE_SPLIT_HALF_APPLY,
    kernels::ATTENTION_SCORES,
    kernels::KV_CACHE_WRITE,
    kernels::KV_CACHE_READ_ATTENTION,
    kernels::EXPERT_MLP_FUSED,
    kernels::SILU_PRODUCT,
    kernels::SHARED_EXPERT_ACTIVATION,
    kernels::COMBINE_EXPERT_PHASE,
    kernels::FILL_ZERO,
    kernels::LM_HEAD_LOGITS,
    kernels::TOPK_VOCAB,
    kernels::GQA_ATTENTION_SCORES,
    kernels::GQA_KV_READ_ATTENTION,
    kernels::LINEAR_CONV1D_STEP,
    kernels::LINEAR_RMS_NORM_QK,
    kernels::LINEAR_COMPUTE_DECAY_BETA,
    kernels::LINEAR_GATED_DELTA_STEP,
    kernels::LINEAR_GATED_RMS_NORM,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalPipelineNameSet {
    pub(crate) q4: &'static str,
    pub(crate) q4_bf16_scale_bias: &'static str,
    pub(crate) q4_swiglu: &'static str,
    pub(crate) q4_swiglu_bf16_scale_bias: &'static str,
    pub(crate) q4_mmap: &'static str,
    pub(crate) q4_mmap_bf16_scale_bias: &'static str,
    pub(crate) q4_mmap_batch: &'static str,
    pub(crate) q4_mmap_batch_bf16_scale_bias: &'static str,
    pub(crate) route_top4: Option<&'static str>,
    pub(crate) dense_matvec: &'static str,
    pub(crate) dense_matvec_bf16: &'static str,
    pub(crate) dense_mmap_matvec: &'static str,
    pub(crate) dense_mmap_matvec_bf16: &'static str,
    pub(crate) dense_mmap_matvec_bf16_simd: &'static str,
    pub(crate) rms_norm: &'static str,
    pub(crate) rms_norm_reduced: &'static str,
    pub(crate) residual_rms_norm: &'static str,
    pub(crate) rope: &'static str,
    pub(crate) rope_split_half: &'static str,
    pub(crate) attention: &'static str,
    pub(crate) kv_write: &'static str,
    pub(crate) kv_read_attention: &'static str,
    pub(crate) expert_mlp: &'static str,
    pub(crate) silu_product: &'static str,
    pub(crate) shared_expert_activation: &'static str,
    pub(crate) combine_expert_phase: &'static str,
    pub(crate) fill_zero: &'static str,
    pub(crate) lm_head: &'static str,
    pub(crate) topk_vocab: &'static str,
    pub(crate) gqa_scores: &'static str,
    pub(crate) gqa_read: &'static str,
    pub(crate) linear_conv1d: &'static str,
    pub(crate) linear_rms_norm_qk: &'static str,
    pub(crate) linear_decay_beta: &'static str,
    pub(crate) linear_delta_step: &'static str,
    pub(crate) linear_gated_rms_norm: &'static str,
}

impl MetalPipelineNameSet {
    pub(crate) fn new(route_top4_enabled: bool) -> Self {
        Self {
            q4: kernels::Q4_FMA_MATVEC,
            q4_bf16_scale_bias: kernels::Q4_FMA_MATVEC_BF16_SCALE_BIAS,
            q4_swiglu: kernels::Q4_SWIGLU_FUSED,
            q4_swiglu_bf16_scale_bias: kernels::Q4_SWIGLU_FUSED_BF16_SCALE_BIAS,
            q4_mmap: kernels::Q4_MMAP_FMA_MATVEC,
            q4_mmap_bf16_scale_bias: kernels::Q4_MMAP_FMA_MATVEC_BF16_SCALE_BIAS,
            q4_mmap_batch: kernels::Q4_MMAP_FMA_MATVEC_BATCH,
            q4_mmap_batch_bf16_scale_bias: kernels::Q4_MMAP_FMA_MATVEC_BATCH_BF16_SCALE_BIAS,
            route_top4: route_top4_enabled.then_some(kernels::ROUTE_TOP4),
            dense_matvec: kernels::DENSE_MATVEC,
            dense_matvec_bf16: kernels::DENSE_MATVEC_BF16,
            dense_mmap_matvec: kernels::DENSE_MMAP_MATVEC_F32,
            dense_mmap_matvec_bf16: kernels::DENSE_MMAP_MATVEC_BF16,
            dense_mmap_matvec_bf16_simd: kernels::DENSE_MMAP_MATVEC_BF16_SIMD,
            rms_norm: kernels::RMS_NORM,
            rms_norm_reduced: kernels::RMS_NORM_REDUCED,
            residual_rms_norm: kernels::RESIDUAL_ADD_RMS_NORM,
            rope: kernels::ROPE_APPLY,
            rope_split_half: kernels::ROPE_SPLIT_HALF_APPLY,
            attention: kernels::ATTENTION_SCORES,
            kv_write: kernels::KV_CACHE_WRITE,
            kv_read_attention: kernels::KV_CACHE_READ_ATTENTION,
            expert_mlp: kernels::EXPERT_MLP_FUSED,
            silu_product: kernels::SILU_PRODUCT,
            shared_expert_activation: kernels::SHARED_EXPERT_ACTIVATION,
            combine_expert_phase: kernels::COMBINE_EXPERT_PHASE,
            fill_zero: kernels::FILL_ZERO,
            lm_head: kernels::LM_HEAD_LOGITS,
            topk_vocab: kernels::TOPK_VOCAB,
            gqa_scores: kernels::GQA_ATTENTION_SCORES,
            gqa_read: kernels::GQA_KV_READ_ATTENTION,
            linear_conv1d: kernels::LINEAR_CONV1D_STEP,
            linear_rms_norm_qk: kernels::LINEAR_RMS_NORM_QK,
            linear_decay_beta: kernels::LINEAR_COMPUTE_DECAY_BETA,
            linear_delta_step: kernels::LINEAR_GATED_DELTA_STEP,
            linear_gated_rms_norm: kernels::LINEAR_GATED_RMS_NORM,
        }
    }

    pub(crate) fn kernel_names(self) -> Vec<&'static str> {
        let mut kernels = vec![
            self.q4,
            self.q4_bf16_scale_bias,
            self.q4_swiglu,
            self.q4_swiglu_bf16_scale_bias,
            self.q4_mmap,
            self.q4_mmap_bf16_scale_bias,
            self.q4_mmap_batch,
            self.q4_mmap_batch_bf16_scale_bias,
            self.dense_matvec,
            self.dense_matvec_bf16,
            self.dense_mmap_matvec,
            self.dense_mmap_matvec_bf16,
            self.dense_mmap_matvec_bf16_simd,
            self.rms_norm,
            self.rms_norm_reduced,
            self.residual_rms_norm,
            self.rope,
            self.rope_split_half,
            self.attention,
            self.kv_write,
            self.kv_read_attention,
            self.expert_mlp,
            self.silu_product,
            self.shared_expert_activation,
            self.combine_expert_phase,
            self.fill_zero,
            self.lm_head,
            self.topk_vocab,
            self.gqa_scores,
            self.gqa_read,
            self.linear_conv1d,
            self.linear_rms_norm_qk,
            self.linear_decay_beta,
            self.linear_delta_step,
            self.linear_gated_rms_norm,
        ];
        if let Some(route_top4) = self.route_top4 {
            kernels.push(route_top4);
        }
        kernels
    }
}

#[derive(Debug)]
pub(crate) struct MetalPipelineSet<T> {
    pub(crate) q4_pipeline: T,
    pub(crate) q4_bf16_scale_bias_pipeline: T,
    pub(crate) q4_swiglu_pipeline: T,
    pub(crate) q4_swiglu_bf16_scale_bias_pipeline: T,
    pub(crate) q4_mmap_pipeline: T,
    pub(crate) q4_mmap_bf16_scale_bias_pipeline: T,
    pub(crate) q4_mmap_batch_pipeline: T,
    pub(crate) q4_mmap_batch_bf16_scale_bias_pipeline: T,
    pub(crate) route_pipeline: Option<T>,
    pub(crate) dense_matvec_pipeline: T,
    pub(crate) dense_matvec_bf16_pipeline: T,
    pub(crate) dense_mmap_matvec_pipeline: T,
    pub(crate) dense_mmap_matvec_bf16_pipeline: T,
    pub(crate) dense_mmap_matvec_bf16_simd_pipeline: T,
    pub(crate) rms_norm_pipeline: T,
    pub(crate) rms_norm_reduced_pipeline: T,
    pub(crate) residual_rms_norm_pipeline: T,
    pub(crate) rope_pipeline: T,
    pub(crate) rope_split_half_pipeline: T,
    pub(crate) attention_pipeline: T,
    pub(crate) kv_write_pipeline: T,
    pub(crate) kv_read_attention_pipeline: T,
    pub(crate) expert_mlp_pipeline: T,
    pub(crate) silu_product_pipeline: T,
    pub(crate) shared_expert_activation_pipeline: T,
    pub(crate) combine_expert_phase_pipeline: T,
    pub(crate) fill_zero_pipeline: T,
    pub(crate) lm_head_pipeline: T,
    pub(crate) topk_vocab_pipeline: T,
    pub(crate) gqa_scores_pipeline: T,
    pub(crate) gqa_read_pipeline: T,
    pub(crate) linear_conv1d_pipeline: T,
    pub(crate) linear_rms_norm_qk_pipeline: T,
    pub(crate) linear_decay_beta_pipeline: T,
    pub(crate) linear_delta_step_pipeline: T,
    pub(crate) linear_gated_rms_norm_pipeline: T,
}

impl<T: Copy> MetalPipelineSet<T> {
    pub(crate) fn release_with(&self, mut release: impl FnMut(T)) {
        release(self.q4_pipeline);
        release(self.q4_bf16_scale_bias_pipeline);
        release(self.q4_swiglu_pipeline);
        release(self.q4_swiglu_bf16_scale_bias_pipeline);
        release(self.q4_mmap_pipeline);
        release(self.q4_mmap_bf16_scale_bias_pipeline);
        release(self.q4_mmap_batch_pipeline);
        release(self.q4_mmap_batch_bf16_scale_bias_pipeline);
        if let Some(route_pipeline) = self.route_pipeline {
            release(route_pipeline);
        }
        release(self.dense_matvec_pipeline);
        release(self.dense_matvec_bf16_pipeline);
        release(self.dense_mmap_matvec_pipeline);
        release(self.dense_mmap_matvec_bf16_pipeline);
        release(self.dense_mmap_matvec_bf16_simd_pipeline);
        release(self.rms_norm_pipeline);
        release(self.rms_norm_reduced_pipeline);
        release(self.residual_rms_norm_pipeline);
        release(self.rope_pipeline);
        release(self.rope_split_half_pipeline);
        release(self.attention_pipeline);
        release(self.kv_write_pipeline);
        release(self.kv_read_attention_pipeline);
        release(self.expert_mlp_pipeline);
        release(self.silu_product_pipeline);
        release(self.shared_expert_activation_pipeline);
        release(self.combine_expert_phase_pipeline);
        release(self.fill_zero_pipeline);
        release(self.lm_head_pipeline);
        release(self.topk_vocab_pipeline);
        release(self.gqa_scores_pipeline);
        release(self.gqa_read_pipeline);
        release(self.linear_conv1d_pipeline);
        release(self.linear_rms_norm_qk_pipeline);
        release(self.linear_decay_beta_pipeline);
        release(self.linear_delta_step_pipeline);
        release(self.linear_gated_rms_norm_pipeline);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalPostAttentionPrep {
    pub(crate) residual_buffer: MetalObjcId,
    pub(crate) normed_buffer: MetalObjcId,
    pub(crate) input: ScheduledCmd3MetalPostAttentionInput,
    pub(crate) state: FlashMoePostAttentionPrepState,
    pub(crate) width: usize,
    pub(crate) active: Vec<(usize, f32)>,
    routing_command: Option<ScheduledRoutingCommand>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalPostAttentionPrep {
    pub(crate) fn new(
        layer: usize,
        width: usize,
        expert_count: usize,
        active: Vec<(usize, f32)>,
        residual_buffer: MetalObjcId,
        normed_buffer: MetalObjcId,
    ) -> anyhow::Result<Self> {
        let state = FlashMoePostAttentionPrepState::new(layer, width, expert_count, active.len());
        if !state.is_declared_graph_state() {
            anyhow::bail!(
                "FlashMoe unsupported Metal post-attention input for layer {layer}: prep state is not declared graph state"
            );
        }
        let input = ScheduledCmd3MetalPostAttentionInput::new(state, active.len())?;
        Ok(Self {
            residual_buffer,
            normed_buffer,
            input,
            state,
            width,
            active,
            routing_command: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn routing_command(&self) -> Option<&ScheduledRoutingCommand> {
        self.routing_command.as_ref()
    }

    pub(crate) fn attach_routing_command(
        &mut self,
        command: ScheduledRoutingCommand,
    ) -> anyhow::Result<ScheduledRoutingCommand> {
        let routing = self.state.routing();
        if command.layer != routing.layer() || command.routing.layer != routing.layer() {
            anyhow::bail!(
                "FlashMoe CMD2 Metal post-attention prep routing layer {} does not match command layer {}",
                routing.layer(),
                command.layer
            );
        }
        if command.routing.experts != routing.experts() {
            anyhow::bail!(
                "FlashMoe CMD2 Metal post-attention prep expert count {} does not match command experts {}",
                routing.experts(),
                command.routing.experts
            );
        }
        if command.active_experts != routing.active_experts()
            || command.routing.active_experts != routing.active_experts()
            || command.routes.len() != self.active.len()
        {
            anyhow::bail!(
                "FlashMoe CMD2 Metal post-attention prep active route count {} does not match command active_experts={} routes={}",
                self.active.len(),
                command.active_experts,
                command.routes.len()
            );
        }
        if command.source != ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK {
            anyhow::bail!(
                "FlashMoe CMD2 Metal post-attention prep requires fused-prep CPU topK routing, got {:?}",
                command.source
            );
        }
        if command.routes != self.active {
            anyhow::bail!(
                "FlashMoe CMD2 Metal post-attention prep routes do not match the scheduler routing command"
            );
        }
        self.routing_command = Some(command.clone());
        Ok(command)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalCmd3DeferredOutput {
    pub(crate) hidden_buffer: MetalObjcId,
    pub(crate) next_normed_buffer: Option<MetalObjcId>,
    pub(crate) output_state: FlashMoeCmd3OutputState,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3DeferredOutput {
    pub(crate) fn new(
        hidden_buffer: MetalObjcId,
        next_normed_buffer: Option<MetalObjcId>,
        output_state: FlashMoeCmd3OutputState,
    ) -> anyhow::Result<Self> {
        if hidden_buffer.is_null() {
            anyhow::bail!("FlashMoe CMD3 deferred output requires a non-null hidden buffer");
        }
        if !output_state.is_declared_graph_state() {
            anyhow::bail!("FlashMoe CMD3 deferred output state is not declared graph state");
        }
        if next_normed_buffer.is_some() != output_state.has_next_normed() {
            anyhow::bail!(
                "FlashMoe CMD3 deferred output next-norm buffer presence does not match declared output state"
            );
        }
        Ok(Self {
            hidden_buffer,
            next_normed_buffer,
            output_state,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3PhasePlan {
    pub(crate) position: usize,
    pub(crate) layer: usize,
    pub(crate) expert_count: usize,
    pub(crate) width: usize,
    pub(crate) output_state: FlashMoeCmd3OutputState,
    pub(crate) has_next_norm: bool,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3InputBuffers {
    pub(crate) normed: MetalObjcId,
    pub(crate) residual: MetalObjcId,
    pub(crate) phase: MetalCmd3PhasePlan,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3InputBuffers {
    pub(crate) fn new(
        phase: MetalCmd3PhasePlan,
        normed: MetalObjcId,
        residual: MetalObjcId,
    ) -> anyhow::Result<Self> {
        if normed.is_null() {
            anyhow::bail!("FlashMoe Metal CMD3 input requires a non-null normed buffer");
        }
        if residual.is_null() {
            anyhow::bail!("FlashMoe Metal CMD3 input requires a non-null residual buffer");
        }
        Ok(Self {
            normed,
            residual,
            phase,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3PhasePlan {
    pub(crate) fn new(
        position: usize,
        layer: usize,
        expert_count: usize,
        width: usize,
        weights_len: usize,
        payloads_len: usize,
        output_state: FlashMoeCmd3OutputState,
        has_next_norm: bool,
    ) -> anyhow::Result<Self> {
        if width == 0 {
            anyhow::bail!("FlashMoe Metal CMD3 phase requires non-zero width");
        }
        if expert_count == 0 {
            anyhow::bail!("FlashMoe Metal CMD3 phase requires at least one active expert");
        }
        if width > u32::MAX as usize {
            anyhow::bail!(
                "FlashMoe Metal CMD3 phase width {} does not fit Metal u32 constants",
                width
            );
        }
        if expert_count > u32::MAX as usize {
            anyhow::bail!(
                "FlashMoe Metal CMD3 phase expert count {} does not fit Metal u32 constants",
                expert_count
            );
        }
        if weights_len != expert_count || payloads_len != expert_count {
            anyhow::bail!(
                "FlashMoe Metal CMD3 phase expert count {} does not match weights={} payloads={}",
                expert_count,
                weights_len,
                payloads_len
            );
        }
        if !output_state.is_declared_graph_state() {
            anyhow::bail!("FlashMoe Metal CMD3 phase output state is not declared graph state");
        }
        if output_state.width() != width {
            anyhow::bail!(
                "FlashMoe Metal CMD3 phase output width {} does not match command width {}",
                output_state.width(),
                width
            );
        }
        if output_state.has_next_normed() != has_next_norm {
            anyhow::bail!(
                "FlashMoe Metal CMD3 phase next-norm output declaration does not match next-norm weights"
            );
        }
        Ok(Self {
            position,
            layer,
            expert_count,
            width,
            output_state,
            has_next_norm,
        })
    }

    pub(crate) fn width_u32(self) -> u32 {
        self.width as u32
    }

    pub(crate) fn expert_outputs_bytes(self) -> anyhow::Result<usize> {
        let items = self.expert_count.checked_mul(self.width).ok_or_else(|| {
            anyhow::anyhow!("FlashMoe Metal CMD3 expert output item count overflow")
        })?;
        Self::f32_bytes("expert output", items)
    }

    pub(crate) fn shared_output_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("shared expert output", self.width)
    }

    pub(crate) fn hidden_output_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("hidden output", self.width)
    }

    pub(crate) fn next_normed_output_bytes(self) -> anyhow::Result<Option<usize>> {
        if self.has_next_norm {
            Self::f32_bytes("next-normed output", self.width).map(Some)
        } else {
            Ok(None)
        }
    }

    pub(crate) fn expert_output_offset(self, index: usize) -> anyhow::Result<u64> {
        if index >= self.expert_count {
            anyhow::bail!(
                "FlashMoe Metal CMD3 expert output index {} exceeds active expert count {}",
                index,
                self.expert_count
            );
        }
        let items = index.checked_mul(self.width).ok_or_else(|| {
            anyhow::anyhow!("FlashMoe Metal CMD3 expert output offset item count overflow")
        })?;
        let bytes = Self::f32_bytes("expert output offset", items)?;
        Ok(bytes as u64)
    }

    fn f32_bytes(label: &str, items: usize) -> anyhow::Result<usize> {
        items
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("FlashMoe Metal CMD3 {label} byte size overflow"))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3CombinePlan {
    pub(crate) width: usize,
    pub(crate) active_count: usize,
    pub(crate) dispatch_threads: u64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3CombineBufferLayout {
    pub(crate) width_u32: u32,
    pub(crate) active_count_u32: u32,
    pub(crate) routing_weights_bytes: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3CombineBuffers {
    pub(crate) routing_weights: MetalObjcId,
    pub(crate) width: MetalObjcId,
    pub(crate) active_count: MetalObjcId,
    pub(crate) layout: MetalCmd3CombineBufferLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3CombineStageBuffers {
    pub(crate) residual: MetalObjcId,
    pub(crate) shared_output: MetalObjcId,
    pub(crate) expert_outputs: MetalObjcId,
    pub(crate) routing_weights: MetalObjcId,
    pub(crate) hidden: MetalObjcId,
    pub(crate) width: MetalObjcId,
    pub(crate) active_count: MetalObjcId,
    pub(crate) plan: MetalCmd3CombinePlan,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3CombineStageBuffers {
    pub(crate) fn new(
        plan: MetalCmd3CombinePlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        combine: MetalCmd3CombineBuffers,
    ) -> anyhow::Result<Self> {
        let layout = plan.buffer_layout()?;
        if combine.layout != layout {
            anyhow::bail!("FlashMoe Metal CMD3 combine stage constants do not match plan");
        }
        if outputs.layout.width_u32 != layout.width_u32
            || outputs.layout.active_count_u32 != layout.active_count_u32
        {
            anyhow::bail!("FlashMoe Metal CMD3 combine stage outputs do not match plan layout");
        }
        Ok(Self {
            residual: inputs.residual,
            shared_output: outputs.shared_output,
            expert_outputs: outputs.expert_outputs,
            routing_weights: combine.routing_weights,
            hidden: outputs.hidden,
            width: combine.width,
            active_count: combine.active_count,
            plan,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3CombineBuffers {
    pub(crate) fn new(
        plan: MetalCmd3CombinePlan,
        routing_weights: MetalObjcId,
        width: MetalObjcId,
        active_count: MetalObjcId,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            routing_weights,
            width,
            active_count,
            layout: plan.buffer_layout()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3CombinePlan {
    pub(crate) fn new(phase: MetalCmd3PhasePlan) -> Self {
        Self {
            width: phase.width,
            active_count: phase.expert_count,
            dispatch_threads: phase.width as u64,
        }
    }

    pub(crate) fn active_count_u32(self) -> u32 {
        self.active_count as u32
    }

    pub(crate) fn routing_weights_bytes(self) -> anyhow::Result<usize> {
        self.active_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                anyhow::anyhow!("FlashMoe Metal CMD3 combine routing weights byte size overflow")
            })
    }

    pub(crate) fn buffer_layout(self) -> anyhow::Result<MetalCmd3CombineBufferLayout> {
        Ok(MetalCmd3CombineBufferLayout {
            width_u32: self.width as u32,
            active_count_u32: self.active_count_u32(),
            routing_weights_bytes: self.routing_weights_bytes()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3NextNormPlan {
    pub(crate) width: usize,
    pub(crate) dispatch_threads: u64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3NextNormBufferLayout {
    pub(crate) width_u32: u32,
    pub(crate) weight_bytes: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3NextNormBuffers {
    pub(crate) hidden: MetalObjcId,
    pub(crate) weight: MetalObjcId,
    pub(crate) next_normed: MetalObjcId,
    pub(crate) width: MetalObjcId,
    pub(crate) layout: MetalCmd3NextNormBufferLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3NextNormBuffers {
    pub(crate) fn new(
        plan: MetalCmd3NextNormPlan,
        hidden: MetalObjcId,
        weight: MetalObjcId,
        next_normed: MetalObjcId,
        width: MetalObjcId,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            hidden,
            weight,
            next_normed,
            width,
            layout: plan.buffer_layout()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3NextNormPlan {
    const RMS_NORM_REDUCED_THREADS: u64 = 256;

    pub(crate) fn new(
        phase: MetalCmd3PhasePlan,
        weight_len: Option<usize>,
    ) -> anyhow::Result<Option<Self>> {
        match (phase.has_next_norm, weight_len) {
            (false, None) => Ok(None),
            (false, Some(_)) => anyhow::bail!(
                "FlashMoe Metal CMD3 next-norm weights were provided for a no-next-norm phase"
            ),
            (true, None) => anyhow::bail!(
                "FlashMoe Metal CMD3 next-norm output is declared but no next-norm weights were provided"
            ),
            (true, Some(weight_len)) => {
                if weight_len < phase.width {
                    anyhow::bail!(
                        "FlashMoe Metal CMD3 next-norm weight length {} is smaller than width {} for layer {}",
                        weight_len,
                        phase.width,
                        phase.layer
                    );
                }
                Ok(Some(Self {
                    width: phase.width,
                    dispatch_threads: Self::RMS_NORM_REDUCED_THREADS,
                }))
            }
        }
    }

    pub(crate) fn weight_bytes(self) -> anyhow::Result<usize> {
        self.width
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                anyhow::anyhow!("FlashMoe Metal CMD3 next-norm weight byte size overflow")
            })
    }

    pub(crate) fn buffer_layout(self) -> anyhow::Result<MetalCmd3NextNormBufferLayout> {
        Ok(MetalCmd3NextNormBufferLayout {
            width_u32: self.width as u32,
            weight_bytes: self.weight_bytes()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalCmd3SharedPhaseSource {
    None,
    Dense,
    ResidentQ4,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3SharedPhasePlan {
    pub(crate) source: MetalCmd3SharedPhaseSource,
    pub(crate) width: usize,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) total_intermediate: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3SharedBufferLayout {
    pub(crate) total_intermediate_u32: u32,
    pub(crate) intermediate_u32: u32,
    pub(crate) projection_output_bytes: usize,
    pub(crate) router_output_bytes: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3SharedWorkBuffers {
    pub(crate) gate_out: MetalObjcId,
    pub(crate) up_out: MetalObjcId,
    pub(crate) router_out: MetalObjcId,
    pub(crate) activated: MetalObjcId,
    pub(crate) total_intermediate: MetalObjcId,
    pub(crate) intermediate: MetalObjcId,
    pub(crate) layout: MetalCmd3SharedBufferLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3SharedStageBuffers {
    pub(crate) source: MetalCmd3SharedPhaseSource,
    pub(crate) normed: MetalObjcId,
    pub(crate) width: MetalObjcId,
    pub(crate) shared_output: MetalObjcId,
    pub(crate) work: Option<MetalCmd3SharedWorkBuffers>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3SharedStageBuffers {
    pub(crate) fn projected(
        plan: MetalCmd3SharedPhasePlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        combine: MetalCmd3CombineBuffers,
        work: MetalCmd3SharedWorkBuffers,
    ) -> anyhow::Result<Self> {
        if plan.source == MetalCmd3SharedPhaseSource::None {
            anyhow::bail!(
                "FlashMoe Metal CMD3 projected shared stage requires a declared shared expert source"
            );
        }
        if work.layout != plan.buffer_layout()? {
            anyhow::bail!("FlashMoe Metal CMD3 shared stage work layout does not match plan");
        }
        Ok(Self {
            source: plan.source,
            normed: inputs.normed,
            width: combine.width,
            shared_output: outputs.shared_output,
            work: Some(work),
        })
    }

    pub(crate) fn fill_zero(
        plan: MetalCmd3SharedPhasePlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        combine: MetalCmd3CombineBuffers,
    ) -> anyhow::Result<Self> {
        if plan.source != MetalCmd3SharedPhaseSource::None {
            anyhow::bail!(
                "FlashMoe Metal CMD3 fill-zero shared stage requires no shared expert source"
            );
        }
        Ok(Self {
            source: plan.source,
            normed: inputs.normed,
            width: combine.width,
            shared_output: outputs.shared_output,
            work: None,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3SharedWorkBuffers {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        plan: MetalCmd3SharedPhasePlan,
        gate_out: MetalObjcId,
        up_out: MetalObjcId,
        router_out: MetalObjcId,
        activated: MetalObjcId,
        total_intermediate: MetalObjcId,
        intermediate: MetalObjcId,
    ) -> anyhow::Result<Self> {
        if plan.source == MetalCmd3SharedPhaseSource::None {
            anyhow::bail!(
                "FlashMoe Metal CMD3 shared work buffers require a declared shared expert source"
            );
        }
        Ok(Self {
            gate_out,
            up_out,
            router_out,
            activated,
            total_intermediate,
            intermediate,
            layout: plan.buffer_layout()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3SharedPhasePlan {
    pub(crate) const fn none(width: usize) -> Self {
        Self {
            source: MetalCmd3SharedPhaseSource::None,
            width,
            shared_experts: 0,
            intermediate: 0,
            total_intermediate: 0,
        }
    }

    pub(crate) fn dense(width: usize, shared: &SharedExpertPhaseWeights) -> anyhow::Result<Self> {
        let shape = shared.validated_shape()?;
        Self::from_shape(MetalCmd3SharedPhaseSource::Dense, width, shape)
    }

    pub(crate) fn resident_q4(
        width: usize,
        shared: &SharedExpertPhaseQ4Projections,
    ) -> anyhow::Result<Self> {
        let shape = shared.validated_shape()?;
        Self::from_shape(MetalCmd3SharedPhaseSource::ResidentQ4, width, shape)
    }

    pub(crate) fn total_intermediate_u32(self) -> anyhow::Result<u32> {
        Self::usize_to_u32("total intermediate width", self.total_intermediate)
    }

    pub(crate) fn intermediate_u32(self) -> anyhow::Result<u32> {
        Self::usize_to_u32("per-shared-expert intermediate width", self.intermediate)
    }

    pub(crate) fn projection_output_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("projection output", self.total_intermediate)
    }

    pub(crate) fn router_output_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("router output", self.shared_experts)
    }

    pub(crate) fn projection_rows(self) -> usize {
        self.total_intermediate
    }

    pub(crate) fn router_rows(self) -> usize {
        self.shared_experts
    }

    pub(crate) fn activation_dispatch_threads(self) -> u64 {
        self.total_intermediate as u64
    }

    pub(crate) fn fill_zero_width(self) -> usize {
        self.width
    }

    pub(crate) fn buffer_layout(self) -> anyhow::Result<MetalCmd3SharedBufferLayout> {
        Ok(MetalCmd3SharedBufferLayout {
            total_intermediate_u32: self.total_intermediate_u32()?,
            intermediate_u32: self.intermediate_u32()?,
            projection_output_bytes: self.projection_output_bytes()?,
            router_output_bytes: self.router_output_bytes()?,
        })
    }

    fn from_shape(
        source: MetalCmd3SharedPhaseSource,
        width: usize,
        shape: super::weights::SharedExpertPhaseShape,
    ) -> anyhow::Result<Self> {
        if shape.width != width {
            anyhow::bail!(
                "FlashMoe Metal CMD3 shared expert width {} does not match phase width {}",
                shape.width,
                width
            );
        }
        Self::usize_to_u32("total intermediate width", shape.total_intermediate)?;
        Self::usize_to_u32("per-shared-expert intermediate width", shape.intermediate)?;
        Ok(Self {
            source,
            width,
            shared_experts: shape.shared_experts,
            intermediate: shape.intermediate,
            total_intermediate: shape.total_intermediate,
        })
    }

    fn usize_to_u32(label: &str, value: usize) -> anyhow::Result<u32> {
        u32::try_from(value).map_err(|_| {
            anyhow::anyhow!(
                "FlashMoe Metal CMD3 shared expert {label} {value} does not fit Metal u32 constants"
            )
        })
    }

    fn f32_bytes(label: &str, items: usize) -> anyhow::Result<usize> {
        items
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                anyhow::anyhow!("FlashMoe Metal CMD3 shared expert {label} byte size overflow")
            })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3ActiveExpertPlan {
    pub(crate) index: usize,
    pub(crate) intermediate: usize,
    pub(crate) output_offset: u64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3ActiveExpertBufferLayout {
    pub(crate) intermediate_u32: u32,
    pub(crate) activation_bytes: usize,
    pub(crate) projection_output_bytes: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3ActiveExpertWorkBuffers {
    pub(crate) activated: MetalObjcId,
    pub(crate) gate_out: Option<MetalObjcId>,
    pub(crate) up_out: Option<MetalObjcId>,
    pub(crate) intermediate: Option<MetalObjcId>,
    pub(crate) layout: MetalCmd3ActiveExpertBufferLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3ActiveExpertStageBuffers {
    pub(crate) normed: MetalObjcId,
    pub(crate) activated: MetalObjcId,
    pub(crate) expert_outputs: MetalObjcId,
    pub(crate) output_offset: u64,
    pub(crate) plan: MetalCmd3ActiveExpertPlan,
    pub(crate) work: MetalCmd3ActiveExpertWorkBuffers,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3ActiveExpertStageBuffers {
    pub(crate) fn new(
        plan: MetalCmd3ActiveExpertPlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        work: MetalCmd3ActiveExpertWorkBuffers,
    ) -> anyhow::Result<Self> {
        if work.layout != plan.buffer_layout()? {
            anyhow::bail!("FlashMoe Metal CMD3 active expert work layout does not match plan");
        }
        Ok(Self {
            normed: inputs.normed,
            activated: work.activated,
            expert_outputs: outputs.expert_outputs,
            output_offset: plan.output_offset,
            plan,
            work,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3ActiveExpertWorkBuffers {
    pub(crate) fn fused(
        plan: MetalCmd3ActiveExpertPlan,
        activated: MetalObjcId,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            activated,
            gate_out: None,
            up_out: None,
            intermediate: None,
            layout: plan.buffer_layout()?,
        })
    }

    pub(crate) fn unfused(
        plan: MetalCmd3ActiveExpertPlan,
        activated: MetalObjcId,
        gate_out: MetalObjcId,
        up_out: MetalObjcId,
        intermediate: MetalObjcId,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            activated,
            gate_out: Some(gate_out),
            up_out: Some(up_out),
            intermediate: Some(intermediate),
            layout: plan.buffer_layout()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3ActiveExpertPlan {
    pub(crate) fn new(
        phase: MetalCmd3PhasePlan,
        index: usize,
        payload: &ScheduledQ4ExpertPhaseMlpPayload<'_>,
    ) -> anyhow::Result<Self> {
        if payload.gate.rows == 0 {
            anyhow::bail!("FlashMoe Metal CMD3 active expert requires non-zero intermediate width");
        }
        if payload.gate.rows != payload.up.rows || payload.down.cols != payload.gate.rows {
            anyhow::bail!(
                "FlashMoe Metal CMD3 active expert payload has mismatched intermediate widths: gate={} up={} down_cols={}",
                payload.gate.rows,
                payload.up.rows,
                payload.down.cols
            );
        }
        if payload.gate.cols != phase.width
            || payload.up.cols != phase.width
            || payload.down.rows != phase.width
        {
            anyhow::bail!(
                "FlashMoe Metal CMD3 active expert payload width does not match phase width {}: gate={} up={} down_rows={}",
                phase.width,
                payload.gate.cols,
                payload.up.cols,
                payload.down.rows
            );
        }
        Self::usize_to_u32("intermediate width", payload.gate.rows)?;
        Ok(Self {
            index,
            intermediate: payload.gate.rows,
            output_offset: phase.expert_output_offset(index)?,
        })
    }

    pub(crate) fn intermediate_u32(self) -> anyhow::Result<u32> {
        Self::usize_to_u32("intermediate width", self.intermediate)
    }

    pub(crate) fn activation_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("activation", self.intermediate)
    }

    pub(crate) fn projection_output_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("projection output", self.intermediate)
    }

    pub(crate) fn dispatch_threads(self) -> u64 {
        self.intermediate as u64
    }

    pub(crate) fn buffer_layout(self) -> anyhow::Result<MetalCmd3ActiveExpertBufferLayout> {
        Ok(MetalCmd3ActiveExpertBufferLayout {
            intermediate_u32: self.intermediate_u32()?,
            activation_bytes: self.activation_bytes()?,
            projection_output_bytes: self.projection_output_bytes()?,
        })
    }

    fn usize_to_u32(label: &str, value: usize) -> anyhow::Result<u32> {
        u32::try_from(value).map_err(|_| {
            anyhow::anyhow!(
                "FlashMoe Metal CMD3 active expert {label} {value} does not fit Metal u32 constants"
            )
        })
    }

    fn f32_bytes(label: &str, items: usize) -> anyhow::Result<usize> {
        items
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                anyhow::anyhow!("FlashMoe Metal CMD3 active expert {label} byte size overflow")
            })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalCmd3ExecutionPlan {
    pub(crate) phase: MetalCmd3PhasePlan,
    pub(crate) next_norm: Option<MetalCmd3NextNormPlan>,
    pub(crate) shared: MetalCmd3SharedPhasePlan,
    pub(crate) active_experts: Vec<MetalCmd3ActiveExpertPlan>,
    pub(crate) combine: MetalCmd3CombinePlan,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3BufferLayout {
    pub(crate) width_u32: u32,
    pub(crate) active_count_u32: u32,
    pub(crate) expert_outputs_bytes: usize,
    pub(crate) shared_output_bytes: usize,
    pub(crate) hidden_output_bytes: usize,
    pub(crate) next_normed_output_bytes: Option<usize>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3OutputBuffers {
    pub(crate) expert_outputs: MetalObjcId,
    pub(crate) shared_output: MetalObjcId,
    pub(crate) hidden: MetalObjcId,
    pub(crate) next_normed: Option<MetalObjcId>,
    pub(crate) layout: MetalCmd3BufferLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3OutputBuffers {
    pub(crate) fn new(
        plan: &MetalCmd3ExecutionPlan,
        expert_outputs: MetalObjcId,
        shared_output: MetalObjcId,
        hidden: MetalObjcId,
        next_normed: Option<MetalObjcId>,
    ) -> anyhow::Result<Self> {
        let layout = plan.buffer_layout()?;
        if next_normed.is_some() != layout.next_normed_output_bytes.is_some() {
            anyhow::bail!(
                "FlashMoe Metal CMD3 output buffers next-normed presence does not match declared output state"
            );
        }
        Ok(Self {
            expert_outputs,
            shared_output,
            hidden,
            next_normed,
            layout,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3ExecutionPlan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        position: usize,
        layer: usize,
        expert_count: usize,
        width: usize,
        weights_len: usize,
        output_state: FlashMoeCmd3OutputState,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight_len: Option<usize>,
        payloads: &[ScheduledExpertPhaseMlpPayload<'_>],
    ) -> anyhow::Result<Self> {
        let phase = MetalCmd3PhasePlan::new(
            position,
            layer,
            expert_count,
            width,
            weights_len,
            payloads.len(),
            output_state,
            next_norm_weight_len.is_some(),
        )?;
        let next_norm = MetalCmd3NextNormPlan::new(phase, next_norm_weight_len)?;
        let shared = match shared {
            ScheduledSharedExpertPhaseRef::None => MetalCmd3SharedPhasePlan::none(width),
            ScheduledSharedExpertPhaseRef::Dense(shared) => {
                MetalCmd3SharedPhasePlan::dense(width, shared)?
            }
            ScheduledSharedExpertPhaseRef::Q4(shared) => {
                MetalCmd3SharedPhasePlan::resident_q4(width, shared)?
            }
        };
        let active_experts = payloads
            .iter()
            .enumerate()
            .map(|(idx, payload)| MetalCmd3ActiveExpertPlan::new(phase, idx, payload.q4()))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let combine = MetalCmd3CombinePlan::new(phase);
        Ok(Self {
            phase,
            next_norm,
            shared,
            active_experts,
            combine,
        })
    }

    pub(crate) fn buffer_layout(&self) -> anyhow::Result<MetalCmd3BufferLayout> {
        Ok(MetalCmd3BufferLayout {
            width_u32: self.phase.width_u32(),
            active_count_u32: self.combine.active_count_u32(),
            expert_outputs_bytes: self.phase.expert_outputs_bytes()?,
            shared_output_bytes: self.phase.shared_output_bytes()?,
            hidden_output_bytes: self.phase.hidden_output_bytes()?,
            next_normed_output_bytes: self.phase.next_normed_output_bytes()?,
        })
    }

    pub(crate) fn command_context(&self, expert_ids: impl ToString) -> MetalCommandContext {
        MetalCommandContext::new("deferred_expert_phase_from_buffers")
            .with("position", self.phase.position)
            .with("layer", self.phase.layer)
            .with("active_experts", self.phase.expert_count)
            .with("experts", expert_ids)
            .with("width", self.phase.width)
            .with(
                "shared",
                !matches!(self.shared.source, MetalCmd3SharedPhaseSource::None),
            )
            .with("next_norm", self.next_norm.is_some())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalPhaseBuffer {
    pub(crate) id: MetalObjcId,
    pub(crate) recycle: bool,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalPhaseBuffer {
    pub(crate) fn recyclable(id: MetalObjcId) -> Self {
        Self { id, recycle: true }
    }

    pub(crate) fn borrowed(id: MetalObjcId) -> Self {
        Self { id, recycle: false }
    }
}

pub const METAL_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4_fma_matvec(
    device const uchar* packed [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& groups_per_row [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = scales[scale_row + group];
            float bias = biases[scale_row + group];

            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];

            acc += fma(float((word >>  0) & 0x0f), scale * x0, bias * x0);
            acc += fma(float((word >>  4) & 0x0f), scale * x1, bias * x1);
            acc += fma(float((word >>  8) & 0x0f), scale * x2, bias * x2);
            acc += fma(float((word >> 12) & 0x0f), scale * x3, bias * x3);
            acc += fma(float((word >> 16) & 0x0f), scale * x4, bias * x4);
            acc += fma(float((word >> 20) & 0x0f), scale * x5, bias * x5);
            acc += fma(float((word >> 24) & 0x0f), scale * x6, bias * x6);
            acc += fma(float((word >> 28) & 0x0f), scale * x7, bias * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float scale0 = scales[scale_row + group0];
            float bias0 = biases[scale_row + group0];
            acc += fma(float(byte & 0x0f), scale0 * x0, bias0 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float scale1 = scales[scale_row + group1];
                float bias1 = biases[scale_row + group1];
                acc += fma(float(byte >> 4), scale1 * x1, bias1 * x1);
            }
        }
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        output[row] = sum;
    }
}

inline float bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16u);
}

kernel void q4_fma_matvec_bf16_scale_bias(
    device const uchar* packed [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const ushort* scales [[buffer(2)]],
    device const ushort* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& groups_per_row [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = bf16_to_float(scales[scale_row + group]);
            float bias = bf16_to_float(biases[scale_row + group]);

            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];

            acc += fma(float((word >>  0) & 0x0f), scale * x0, bias * x0);
            acc += fma(float((word >>  4) & 0x0f), scale * x1, bias * x1);
            acc += fma(float((word >>  8) & 0x0f), scale * x2, bias * x2);
            acc += fma(float((word >> 12) & 0x0f), scale * x3, bias * x3);
            acc += fma(float((word >> 16) & 0x0f), scale * x4, bias * x4);
            acc += fma(float((word >> 20) & 0x0f), scale * x5, bias * x5);
            acc += fma(float((word >> 24) & 0x0f), scale * x6, bias * x6);
            acc += fma(float((word >> 28) & 0x0f), scale * x7, bias * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float scale0 = bf16_to_float(scales[scale_row + group0]);
            float bias0 = bf16_to_float(biases[scale_row + group0]);
            acc += fma(float(byte & 0x0f), scale0 * x0, bias0 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float scale1 = bf16_to_float(scales[scale_row + group1]);
                float bias1 = bf16_to_float(biases[scale_row + group1]);
                acc += fma(float(byte >> 4), scale1 * x1, bias1 * x1);
            }
        }
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        output[row] = sum;
    }
}

kernel void q4_swiglu_fused(
    device const uchar* gate_packed [[buffer(0)]],
    device const uchar* up_packed [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device const float* gate_scales [[buffer(3)]],
    device const float* gate_biases [[buffer(4)]],
    device const float* up_scales [[buffer(5)]],
    device const float* up_biases [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& groups_per_row [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* gate_words = reinterpret_cast<device const uint*>(gate_packed);
        device const uint* up_words = reinterpret_cast<device const uint*>(up_packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint gate_word = gate_words[word_row + packed_word];
            uint up_word = up_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float gate_scale = gate_scales[scale_row + group];
            float gate_bias = gate_biases[scale_row + group];
            float up_scale = up_scales[scale_row + group];
            float up_bias = up_biases[scale_row + group];

            for (uint i = 0; i < 8; i++) {
                uint shift = i * 4;
                float x = use_input_cache ? input_cache[col0 + i] : input[col0 + i];
                gate_acc += fma(float((gate_word >> shift) & 0x0f), gate_scale * x, gate_bias * x);
                up_acc += fma(float((up_word >> shift) & 0x0f), up_scale * x, up_bias * x);
            }
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar gate_byte = gate_packed[packed_row + packed_col];
            uchar up_byte = up_packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float gate_scale0 = gate_scales[scale_row + group0];
            float gate_bias0 = gate_biases[scale_row + group0];
            float up_scale0 = up_scales[scale_row + group0];
            float up_bias0 = up_biases[scale_row + group0];
            gate_acc += fma(float(gate_byte & 0x0f), gate_scale0 * x0, gate_bias0 * x0);
            up_acc += fma(float(up_byte & 0x0f), up_scale0 * x0, up_bias0 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float gate_scale1 = gate_scales[scale_row + group1];
                float gate_bias1 = gate_biases[scale_row + group1];
                float up_scale1 = up_scales[scale_row + group1];
                float up_bias1 = up_biases[scale_row + group1];
                gate_acc += fma(float(gate_byte >> 4), gate_scale1 * x1, gate_bias1 * x1);
                up_acc += fma(float(up_byte >> 4), up_scale1 * x1, up_bias1 * x1);
            }
        }
    }
    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0) {
        output[row] = (gate_sum / (1.0f + exp(-gate_sum))) * up_sum;
    }
}

kernel void q4_swiglu_fused_bf16_scale_bias(
    device const uchar* gate_packed [[buffer(0)]],
    device const uchar* up_packed [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device const ushort* gate_scales [[buffer(3)]],
    device const ushort* gate_biases [[buffer(4)]],
    device const ushort* up_scales [[buffer(5)]],
    device const ushort* up_biases [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& groups_per_row [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* gate_words = reinterpret_cast<device const uint*>(gate_packed);
        device const uint* up_words = reinterpret_cast<device const uint*>(up_packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint gate_word = gate_words[word_row + packed_word];
            uint up_word = up_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float gate_scale = bf16_to_float(gate_scales[scale_row + group]);
            float gate_bias = bf16_to_float(gate_biases[scale_row + group]);
            float up_scale = bf16_to_float(up_scales[scale_row + group]);
            float up_bias = bf16_to_float(up_biases[scale_row + group]);

            for (uint i = 0; i < 8; i++) {
                uint shift = i * 4;
                float x = use_input_cache ? input_cache[col0 + i] : input[col0 + i];
                gate_acc += fma(float((gate_word >> shift) & 0x0f), gate_scale * x, gate_bias * x);
                up_acc += fma(float((up_word >> shift) & 0x0f), up_scale * x, up_bias * x);
            }
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar gate_byte = gate_packed[packed_row + packed_col];
            uchar up_byte = up_packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float gate_scale0 = bf16_to_float(gate_scales[scale_row + group0]);
            float gate_bias0 = bf16_to_float(gate_biases[scale_row + group0]);
            float up_scale0 = bf16_to_float(up_scales[scale_row + group0]);
            float up_bias0 = bf16_to_float(up_biases[scale_row + group0]);
            gate_acc += fma(float(gate_byte & 0x0f), gate_scale0 * x0, gate_bias0 * x0);
            up_acc += fma(float(up_byte & 0x0f), up_scale0 * x0, up_bias0 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float gate_scale1 = bf16_to_float(gate_scales[scale_row + group1]);
                float gate_bias1 = bf16_to_float(gate_biases[scale_row + group1]);
                float up_scale1 = bf16_to_float(up_scales[scale_row + group1]);
                float up_bias1 = bf16_to_float(up_biases[scale_row + group1]);
                gate_acc += fma(float(gate_byte >> 4), gate_scale1 * x1, gate_bias1 * x1);
                up_acc += fma(float(up_byte >> 4), up_scale1 * x1, up_bias1 * x1);
            }
        }
    }
    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0) {
        output[row] = (gate_sum / (1.0f + exp(-gate_sum))) * up_sum;
    }
}

kernel void q4_mmap_fma_matvec(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& packed_byte_offset [[buffer(3)]],
    constant ulong& scales_byte_offset [[buffer(4)]],
    constant ulong& biases_byte_offset [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& groups_per_row [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const float* scales = reinterpret_cast<device const float*>(weight_bytes + scales_byte_offset);
    device const float* biases = reinterpret_cast<device const float*>(weight_bytes + biases_byte_offset);
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint row0 = tile * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[4096];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 >= rows) {
        return;
    }

    bool row1_valid = row1 < rows;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint packed_row0 = row0 * packed_stride;
    uint packed_row1 = row1 * packed_stride;
    uint scale_row0 = row0 * groups_per_row;
    uint scale_row1 = row1 * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0) && ((packed_byte_offset & 3ul) == 0ul);
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row0 = row0 * packed_words_per_row;
        uint word_row1 = row1 * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word0 = packed_words[word_row0 + packed_word];
            uint word1 = row1_valid ? packed_words[word_row1 + packed_word] : 0u;
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale0 = scales[scale_row0 + group];
            float bias0 = biases[scale_row0 + group];
            float scale1 = row1_valid ? scales[scale_row1 + group] : 0.0f;
            float bias1 = row1_valid ? biases[scale_row1 + group] : 0.0f;

            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];

            acc0 += fma(float((word0 >>  0) & 0x0f), scale0 * x0, bias0 * x0);
            acc0 += fma(float((word0 >>  4) & 0x0f), scale0 * x1, bias0 * x1);
            acc0 += fma(float((word0 >>  8) & 0x0f), scale0 * x2, bias0 * x2);
            acc0 += fma(float((word0 >> 12) & 0x0f), scale0 * x3, bias0 * x3);
            acc0 += fma(float((word0 >> 16) & 0x0f), scale0 * x4, bias0 * x4);
            acc0 += fma(float((word0 >> 20) & 0x0f), scale0 * x5, bias0 * x5);
            acc0 += fma(float((word0 >> 24) & 0x0f), scale0 * x6, bias0 * x6);
            acc0 += fma(float((word0 >> 28) & 0x0f), scale0 * x7, bias0 * x7);

            acc1 += fma(float((word1 >>  0) & 0x0f), scale1 * x0, bias1 * x0);
            acc1 += fma(float((word1 >>  4) & 0x0f), scale1 * x1, bias1 * x1);
            acc1 += fma(float((word1 >>  8) & 0x0f), scale1 * x2, bias1 * x2);
            acc1 += fma(float((word1 >> 12) & 0x0f), scale1 * x3, bias1 * x3);
            acc1 += fma(float((word1 >> 16) & 0x0f), scale1 * x4, bias1 * x4);
            acc1 += fma(float((word1 >> 20) & 0x0f), scale1 * x5, bias1 * x5);
            acc1 += fma(float((word1 >> 24) & 0x0f), scale1 * x6, bias1 * x6);
            acc1 += fma(float((word1 >> 28) & 0x0f), scale1 * x7, bias1 * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte0 = packed[packed_row0 + packed_col];
            uchar byte1 = row1_valid ? packed[packed_row1 + packed_col] : uchar(0);
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float scale00 = scales[scale_row0 + group0];
            float bias00 = biases[scale_row0 + group0];
            float scale10 = row1_valid ? scales[scale_row1 + group0] : 0.0f;
            float bias10 = row1_valid ? biases[scale_row1 + group0] : 0.0f;
            acc0 += fma(float(byte0 & 0x0f), scale00 * x0, bias00 * x0);
            acc1 += fma(float(byte1 & 0x0f), scale10 * x0, bias10 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float scale01 = scales[scale_row0 + group1];
                float bias01 = biases[scale_row0 + group1];
                float scale11 = row1_valid ? scales[scale_row1 + group1] : 0.0f;
                float bias11 = row1_valid ? biases[scale_row1 + group1] : 0.0f;
                acc0 += fma(float(byte0 >> 4), scale01 * x1, bias01 * x1);
                acc1 += fma(float(byte1 >> 4), scale11 * x1, bias11 * x1);
            }
        }
    }
    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        output[row0] = sum0;
        if (row1_valid) {
            output[row1] = sum1;
        }
    }
}

kernel void q4_mmap_fma_matvec_bf16_scale_bias(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& packed_byte_offset [[buffer(3)]],
    constant ulong& scales_byte_offset [[buffer(4)]],
    constant ulong& biases_byte_offset [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& groups_per_row [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const ushort* scales = reinterpret_cast<device const ushort*>(weight_bytes + scales_byte_offset);
    device const ushort* biases = reinterpret_cast<device const ushort*>(weight_bytes + biases_byte_offset);
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint row0 = tile * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[4096];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 >= rows) {
        return;
    }

    bool row1_valid = row1 < rows;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint packed_row0 = row0 * packed_stride;
    uint packed_row1 = row1 * packed_stride;
    uint scale_row0 = row0 * groups_per_row;
    uint scale_row1 = row1 * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0) && ((packed_byte_offset & 3ul) == 0ul);
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row0 = row0 * packed_words_per_row;
        uint word_row1 = row1 * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word0 = packed_words[word_row0 + packed_word];
            uint word1 = row1_valid ? packed_words[word_row1 + packed_word] : 0u;
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale0 = bf16_to_float(scales[scale_row0 + group]);
            float bias0 = bf16_to_float(biases[scale_row0 + group]);
            float scale1 = row1_valid ? bf16_to_float(scales[scale_row1 + group]) : 0.0f;
            float bias1 = row1_valid ? bf16_to_float(biases[scale_row1 + group]) : 0.0f;

            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];

            acc0 += fma(float((word0 >>  0) & 0x0f), scale0 * x0, bias0 * x0);
            acc0 += fma(float((word0 >>  4) & 0x0f), scale0 * x1, bias0 * x1);
            acc0 += fma(float((word0 >>  8) & 0x0f), scale0 * x2, bias0 * x2);
            acc0 += fma(float((word0 >> 12) & 0x0f), scale0 * x3, bias0 * x3);
            acc0 += fma(float((word0 >> 16) & 0x0f), scale0 * x4, bias0 * x4);
            acc0 += fma(float((word0 >> 20) & 0x0f), scale0 * x5, bias0 * x5);
            acc0 += fma(float((word0 >> 24) & 0x0f), scale0 * x6, bias0 * x6);
            acc0 += fma(float((word0 >> 28) & 0x0f), scale0 * x7, bias0 * x7);

            acc1 += fma(float((word1 >>  0) & 0x0f), scale1 * x0, bias1 * x0);
            acc1 += fma(float((word1 >>  4) & 0x0f), scale1 * x1, bias1 * x1);
            acc1 += fma(float((word1 >>  8) & 0x0f), scale1 * x2, bias1 * x2);
            acc1 += fma(float((word1 >> 12) & 0x0f), scale1 * x3, bias1 * x3);
            acc1 += fma(float((word1 >> 16) & 0x0f), scale1 * x4, bias1 * x4);
            acc1 += fma(float((word1 >> 20) & 0x0f), scale1 * x5, bias1 * x5);
            acc1 += fma(float((word1 >> 24) & 0x0f), scale1 * x6, bias1 * x6);
            acc1 += fma(float((word1 >> 28) & 0x0f), scale1 * x7, bias1 * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte0 = packed[packed_row0 + packed_col];
            uchar byte1 = row1_valid ? packed[packed_row1 + packed_col] : uchar(0);
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float scale00 = bf16_to_float(scales[scale_row0 + group0]);
            float bias00 = bf16_to_float(biases[scale_row0 + group0]);
            float scale10 = row1_valid ? bf16_to_float(scales[scale_row1 + group0]) : 0.0f;
            float bias10 = row1_valid ? bf16_to_float(biases[scale_row1 + group0]) : 0.0f;
            acc0 += fma(float(byte0 & 0x0f), scale00 * x0, bias00 * x0);
            acc1 += fma(float(byte1 & 0x0f), scale10 * x0, bias10 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float scale01 = bf16_to_float(scales[scale_row0 + group1]);
                float bias01 = bf16_to_float(biases[scale_row0 + group1]);
                float scale11 = row1_valid ? bf16_to_float(scales[scale_row1 + group1]) : 0.0f;
                float bias11 = row1_valid ? bf16_to_float(biases[scale_row1 + group1]) : 0.0f;
                acc0 += fma(float(byte0 >> 4), scale01 * x1, bias01 * x1);
                acc1 += fma(float(byte1 >> 4), scale11 * x1, bias11 * x1);
            }
        }
    }
    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        output[row0] = sum0;
        if (row1_valid) {
            output[row1] = sum1;
        }
    }
}

inline uint q4_batch_projection_for_row(
    uint row,
    device const uint* row_offsets,
    device const uint* rows,
    uint projection_count) {
    for (uint idx = 0; idx < projection_count; idx++) {
        uint start = row_offsets[idx];
        uint end = start + rows[idx];
        if (row >= start && row < end) {
            return idx;
        }
    }
    return projection_count;
}

inline float q4_mmap_fma_row_f32(
    device const uchar* weight_bytes,
    device const float* input,
    threadgroup float* input_cache,
    ulong packed_byte_offset,
    ulong scales_byte_offset,
    ulong biases_byte_offset,
    uint row,
    uint cols,
    uint groups_per_row,
    uint group_size,
    uint simd_lane) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const float* scales = reinterpret_cast<device const float*>(weight_bytes + scales_byte_offset);
    device const float* biases = reinterpret_cast<device const float*>(weight_bytes + biases_byte_offset);
    uint packed_stride = (cols + 1) / 2;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0) && ((packed_byte_offset & 3ul) == 0ul);
    float acc = 0.0f;
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = scales[scale_row + group];
            float bias = biases[scale_row + group];
            float x0 = input_cache[col0 + 0];
            float x1 = input_cache[col0 + 1];
            float x2 = input_cache[col0 + 2];
            float x3 = input_cache[col0 + 3];
            float x4 = input_cache[col0 + 4];
            float x5 = input_cache[col0 + 5];
            float x6 = input_cache[col0 + 6];
            float x7 = input_cache[col0 + 7];
            acc += fma(float((word >>  0) & 0x0f), scale * x0, bias * x0);
            acc += fma(float((word >>  4) & 0x0f), scale * x1, bias * x1);
            acc += fma(float((word >>  8) & 0x0f), scale * x2, bias * x2);
            acc += fma(float((word >> 12) & 0x0f), scale * x3, bias * x3);
            acc += fma(float((word >> 16) & 0x0f), scale * x4, bias * x4);
            acc += fma(float((word >> 20) & 0x0f), scale * x5, bias * x5);
            acc += fma(float((word >> 24) & 0x0f), scale * x6, bias * x6);
            acc += fma(float((word >> 28) & 0x0f), scale * x7, bias * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = input_cache[col0];
            uint group0 = col0 / group_size;
            float scale0 = scales[scale_row + group0];
            float bias0 = biases[scale_row + group0];
            acc += fma(float(byte & 0x0f), scale0 * x0, bias0 * x0);
            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = input_cache[col1];
                uint group1 = col1 / group_size;
                float scale1 = scales[scale_row + group1];
                float bias1 = biases[scale_row + group1];
                acc += fma(float(byte >> 4), scale1 * x1, bias1 * x1);
            }
        }
    }
    return simd_sum(acc);
}

inline float q4_mmap_fma_row_bf16(
    device const uchar* weight_bytes,
    device const float* input,
    threadgroup float* input_cache,
    ulong packed_byte_offset,
    ulong scales_byte_offset,
    ulong biases_byte_offset,
    uint row,
    uint cols,
    uint groups_per_row,
    uint group_size,
    uint simd_lane) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const ushort* scales = reinterpret_cast<device const ushort*>(weight_bytes + scales_byte_offset);
    device const ushort* biases = reinterpret_cast<device const ushort*>(weight_bytes + biases_byte_offset);
    uint packed_stride = (cols + 1) / 2;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0) && ((packed_byte_offset & 3ul) == 0ul);
    float acc = 0.0f;
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = bf16_to_float(scales[scale_row + group]);
            float bias = bf16_to_float(biases[scale_row + group]);
            float x0 = input_cache[col0 + 0];
            float x1 = input_cache[col0 + 1];
            float x2 = input_cache[col0 + 2];
            float x3 = input_cache[col0 + 3];
            float x4 = input_cache[col0 + 4];
            float x5 = input_cache[col0 + 5];
            float x6 = input_cache[col0 + 6];
            float x7 = input_cache[col0 + 7];
            acc += fma(float((word >>  0) & 0x0f), scale * x0, bias * x0);
            acc += fma(float((word >>  4) & 0x0f), scale * x1, bias * x1);
            acc += fma(float((word >>  8) & 0x0f), scale * x2, bias * x2);
            acc += fma(float((word >> 12) & 0x0f), scale * x3, bias * x3);
            acc += fma(float((word >> 16) & 0x0f), scale * x4, bias * x4);
            acc += fma(float((word >> 20) & 0x0f), scale * x5, bias * x5);
            acc += fma(float((word >> 24) & 0x0f), scale * x6, bias * x6);
            acc += fma(float((word >> 28) & 0x0f), scale * x7, bias * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = input_cache[col0];
            uint group0 = col0 / group_size;
            float scale0 = bf16_to_float(scales[scale_row + group0]);
            float bias0 = bf16_to_float(biases[scale_row + group0]);
            acc += fma(float(byte & 0x0f), scale0 * x0, bias0 * x0);
            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = input_cache[col1];
                uint group1 = col1 / group_size;
                float scale1 = bf16_to_float(scales[scale_row + group1]);
                float bias1 = bf16_to_float(biases[scale_row + group1]);
                acc += fma(float(byte >> 4), scale1 * x1, bias1 * x1);
            }
        }
    }
    return simd_sum(acc);
}

kernel void q4_mmap_fma_matvec_batch(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const ulong* packed_byte_offsets [[buffer(3)]],
    device const ulong* scales_byte_offsets [[buffer(4)]],
    device const ulong* biases_byte_offsets [[buffer(5)]],
    device const uint* row_offsets [[buffer(6)]],
    device const uint* rows [[buffer(7)]],
    device const uint* groups_per_rows [[buffer(8)]],
    constant uint& projection_count [[buffer(9)]],
    constant uint& cols [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint row0 = tile * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    threadgroup float input_cache[4096];
    for (uint col = lid; col < cols && col < input_cache_len; col += 256) {
        input_cache[col] = input[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint projection0 = q4_batch_projection_for_row(row0, row_offsets, rows, projection_count);
    if (projection0 < projection_count) {
        uint local_row = row0 - row_offsets[projection0];
        float sum = q4_mmap_fma_row_f32(
            weight_bytes, input, input_cache,
            packed_byte_offsets[projection0],
            scales_byte_offsets[projection0],
            biases_byte_offsets[projection0],
            local_row, cols, groups_per_rows[projection0], group_size, simd_lane);
        if (simd_lane == 0) {
            output[row0] = sum;
        }
    }
    uint projection1 = q4_batch_projection_for_row(row1, row_offsets, rows, projection_count);
    if (projection1 < projection_count) {
        uint local_row = row1 - row_offsets[projection1];
        float sum = q4_mmap_fma_row_f32(
            weight_bytes, input, input_cache,
            packed_byte_offsets[projection1],
            scales_byte_offsets[projection1],
            biases_byte_offsets[projection1],
            local_row, cols, groups_per_rows[projection1], group_size, simd_lane);
        if (simd_lane == 0) {
            output[row1] = sum;
        }
    }
}

kernel void q4_mmap_fma_matvec_batch_bf16_scale_bias(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const ulong* packed_byte_offsets [[buffer(3)]],
    device const ulong* scales_byte_offsets [[buffer(4)]],
    device const ulong* biases_byte_offsets [[buffer(5)]],
    device const uint* row_offsets [[buffer(6)]],
    device const uint* rows [[buffer(7)]],
    device const uint* groups_per_rows [[buffer(8)]],
    constant uint& projection_count [[buffer(9)]],
    constant uint& cols [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint row0 = tile * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    threadgroup float input_cache[4096];
    for (uint col = lid; col < cols && col < input_cache_len; col += 256) {
        input_cache[col] = input[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint projection0 = q4_batch_projection_for_row(row0, row_offsets, rows, projection_count);
    if (projection0 < projection_count) {
        uint local_row = row0 - row_offsets[projection0];
        float sum = q4_mmap_fma_row_bf16(
            weight_bytes, input, input_cache,
            packed_byte_offsets[projection0],
            scales_byte_offsets[projection0],
            biases_byte_offsets[projection0],
            local_row, cols, groups_per_rows[projection0], group_size, simd_lane);
        if (simd_lane == 0) {
            output[row0] = sum;
        }
    }
    uint projection1 = q4_batch_projection_for_row(row1, row_offsets, rows, projection_count);
    if (projection1 < projection_count) {
        uint local_row = row1 - row_offsets[projection1];
        float sum = q4_mmap_fma_row_bf16(
            weight_bytes, input, input_cache,
            packed_byte_offsets[projection1],
            scales_byte_offsets[projection1],
            biases_byte_offsets[projection1],
            local_row, cols, groups_per_rows[projection1], group_size, simd_lane);
        if (simd_lane == 0) {
            output[row1] = sum;
        }
    }
}

kernel void route_top4(
    device const float* scores [[buffer(0)]],
    device uint4* indices [[buffer(1)]],
    device float4* weights [[buffer(2)]],
    constant uint& experts [[buffer(3)]],
    uint token [[thread_position_in_grid]]) {
    float4 best = float4(-INFINITY);
    uint4 best_i = uint4(0);
    for (uint i = 0; i < experts; ++i) {
        float score = scores[token * experts + i];
        if (score > best.x) { best.w = best.z; best_i.w = best_i.z; best.z = best.y; best_i.z = best_i.y; best.y = best.x; best_i.y = best_i.x; best.x = score; best_i.x = i; }
        else if (score > best.y) { best.w = best.z; best_i.w = best_i.z; best.z = best.y; best_i.z = best_i.y; best.y = score; best_i.y = i; }
        else if (score > best.z) { best.w = best.z; best_i.w = best_i.z; best.z = score; best_i.z = i; }
        else if (score > best.w) { best.w = score; best_i.w = i; }
    }
    weights[token] = best;
    indices[token] = best_i;
}

kernel void dense_matvec(
    device const float* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(weights[row * cols + col], input[col], acc);
    }
    output[row] = acc;
}

kernel void dense_matvec_bf16(
    device const ushort* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    float acc = 0.0f;
    uint row_offset = row * cols;
    for (uint col = 0; col < cols; ++col) {
        uint bits = uint(weights[row_offset + col]) << 16u;
        float weight = as_type<float>(bits);
        acc = fma(weight, input[col], acc);
    }
    output[row] = acc;
}

kernel void dense_mmap_matvec_f32(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& stride [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) { return; }
    device const float* weights = reinterpret_cast<device const float*>(weight_bytes + byte_offset);
    float acc = 0.0f;
    uint row_offset = row * stride;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(weights[row_offset + col], input[col], acc);
    }
    output[row] = acc;
}

kernel void dense_mmap_matvec_bf16(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& stride [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) { return; }
    device const ushort* weights = reinterpret_cast<device const ushort*>(weight_bytes + byte_offset);
    float acc = 0.0f;
    uint row_offset = row * stride;
    for (uint col = 0; col < cols; ++col) {
        uint bits = uint(weights[row_offset + col]) << 16u;
        acc = fma(as_type<float>(bits), input[col], acc);
    }
    output[row] = acc;
}

kernel void dense_mmap_matvec_bf16_simd(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& stride [[buffer(6)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 4096;
    uint row = tile * rows_per_threadgroup + simd_group;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[4096];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) { return; }

    device const ushort* weights = reinterpret_cast<device const ushort*>(weight_bytes + byte_offset);
    uint row_offset = row * stride;
    float acc = 0.0f;
    for (uint col = simd_lane; col < cols; col += 32) {
        uint bits = uint(weights[row_offset + col]) << 16u;
        float x = use_input_cache ? input_cache[col] : input[col];
        acc = fma(as_type<float>(bits), x, acc);
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        output[row] = sum;
    }
}

kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    float sum = 0.0f;
    for (uint i = 0; i < width; ++i) {
        sum = fma(input[i], input[i], sum);
    }
    float scale = rsqrt(sum / float(max(width, 1u)) + 1.0e-6f);
    output[idx] = input[idx] * scale * weight[idx];
}

kernel void rms_norm_reduced(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]]) {
    const uint threads = 256;
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint i = lid; i < width; i += threads) {
        sum = fma(input[i], input[i], sum);
    }
    partial[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(partial[0] / float(max(width, 1u)) + 1.0e-6f);
    for (uint i = lid; i < width; i += threads) {
        output[i] = input[i] * scale * weight[i];
    }
}

kernel void residual_add_rms_norm(
    device const float* projected [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    device float* normed [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]]) {
    const uint threads = 256;
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint i = lid; i < width; i += threads) {
        float value = projected[i] + residual[i];
        sum = fma(value, value, sum);
    }
    partial[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(partial[0] / float(max(width, 1u)) + 1.0e-6f);
    for (uint i = lid; i < width; i += threads) {
        float value = projected[i] + residual[i];
        hidden[i] = value;
        normed[i] = value * scale * weight[i];
    }
}

kernel void rope_apply(
    device float* values [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
    uint pair = idx * 2u;
    uint lane = pair % head_dim;
    float inv_freq = pow(theta, -float(lane) / float(max(head_dim, 1u)));
    float angle = float(position) * inv_freq;
    float s = sin(angle);
    float c = cos(angle);
    float x = values[pair];
    float y = values[pair + 1u];
    values[pair] = x * c - y * s;
    values[pair + 1u] = x * s + y * c;
}

kernel void rope_split_half_apply(
    device float* values [[buffer(0)]],
    constant uint& temporal_position [[buffer(1)]],
    constant uint& height_position [[buffer(2)]],
    constant uint& width_position [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& rotary_dim [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    constant uint& use_mrope [[buffer(7)]],
    device const uint* mrope_section [[buffer(8)]],
    uint idx [[thread_position_in_grid]]) {
    uint safe_head_dim = max(head_dim, 1u);
    uint safe_rotary = min(rotary_dim, safe_head_dim);
    safe_rotary -= safe_rotary % 2u;
    uint rotary_half = max(safe_rotary / 2u, 1u);
    uint head = idx / rotary_half;
    uint i = idx % rotary_half;
    if (i >= rotary_half) { return; }

    uint position = temporal_position;
    if (use_mrope != 0u) {
        uint height = mrope_section[1];
        uint width = mrope_section[2];
        if ((i % 3u) == 1u && i < height * 3u) {
            position = height_position;
        } else if ((i % 3u) == 2u && i < width * 3u) {
            position = width_position;
        }
    }

    float inv_freq = pow(max(theta, 1.0f), -float(2u * i) / float(max(safe_rotary, 1u)));
    float angle = float(position) * inv_freq;
    float s = sin(angle);
    float c = cos(angle);
    uint base = head * safe_head_dim;
    uint lo = base + i;
    uint hi = base + i + rotary_half;
    float x0 = values[lo];
    float x1 = values[hi];
    values[lo] = x0 * c - x1 * s;
    values[hi] = x0 * s + x1 * c;
}

kernel void attention_scores(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint token [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < width; ++i) {
        acc = fma(query[i], keys[token * width + i], acc);
    }
    scores[token] = acc * rsqrt(float(max(head_dim, 1u)));
}

kernel void kv_cache_write(
    device const float* key [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* keys [[buffer(2)]],
    device float* values [[buffer(3)]],
    constant ulong& offset [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    keys[offset + idx] = key[idx];
    values[offset + idx] = value[idx];
}

kernel void kv_cache_read_attention(
    device const float* weights [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    float acc = 0.0f;
    for (uint token = 0; token < tokens; ++token) {
        acc = fma(weights[token], values[token * width + idx], acc);
    }
    output[idx] = acc;
}

kernel void expert_mlp_fused(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* down [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& intermediate [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < intermediate; ++i) {
        float g = gate[i] / (1.0f + exp(-gate[i]));
        acc = fma(down[row * intermediate + i], g * up[i], acc);
    }
    output[row] = acc * rsqrt(float(max(intermediate, 1u)));
}

kernel void silu_product(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    float g = gate[idx];
    output[idx] = (g / (1.0f + exp(-g))) * up[idx];
}

kernel void shared_expert_activation(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* router [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& intermediate [[buffer(4)]],
    constant uint& total_intermediate [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= total_intermediate) { return; }
    uint shared_idx = intermediate == 0 ? 0 : idx / intermediate;
    float route = router[shared_idx];
    float route_weight = 1.0f / (1.0f + exp(-route));
    float g = gate[idx];
    output[idx] = (g / (1.0f + exp(-g))) * up[idx] * route_weight;
}

kernel void combine_expert_phase(
    device const float* residual [[buffer(0)]],
    device const float* shared [[buffer(1)]],
    device const float* expert_outputs [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device float* hidden [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& active_experts [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    float acc = residual[idx] + shared[idx];
    for (uint expert = 0; expert < active_experts; ++expert) {
        acc = fma(expert_outputs[expert * width + idx], weights[expert], acc);
    }
    hidden[idx] = acc;
}

kernel void fill_zero(
    device float* output [[buffer(0)]],
    constant uint& width [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    output[idx] = 0.0f;
}

kernel void lm_head_logits(
    device const float* lm_head [[buffer(0)]],
    device const float* hidden [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint& hidden_width [[buffer(3)]],
    constant uint& vocab [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint hidden_cache_len = 4096;
    uint token = tile * rows_per_threadgroup + simd_group;
    bool use_hidden_cache = hidden_width <= hidden_cache_len;
    threadgroup float hidden_cache[4096];
    if (use_hidden_cache) {
        for (uint i = lid; i < hidden_width; i += 256) {
            hidden_cache[i] = hidden[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (token >= vocab) {
        return;
    }

    float acc = 0.0f;
    for (uint i = simd_lane; i < hidden_width; i += 32) {
        float h = use_hidden_cache ? hidden_cache[i] : hidden[i];
        acc = fma(lm_head[token * hidden_width + i], h, acc);
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        logits[token] = sum;
    }
}

kernel void topk_vocab(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* values [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    uint slot [[thread_position_in_grid]]) {
    if (slot != 0) { return; }
    uint limit = min(top_k, vocab);
    for (uint out = 0; out < limit; ++out) {
        float best = -INFINITY;
        uint best_i = 0;
        bool found = false;
        for (uint i = 0; i < vocab; ++i) {
            float raw_value = logits[i];
            float value = isfinite(raw_value) ? raw_value : -INFINITY;
            bool already_used = false;
            for (uint prev = 0; prev < out; ++prev) {
                already_used = already_used || (indices[prev] == i);
            }
            if (!already_used && (!found || value > best)) {
                best = value;
                best_i = i;
                found = true;
            }
        }
        indices[out] = best_i;
        values[out] = best;
    }
}

// Multi-head GQA attention scores.
// One thread per (q_head, token) pair: tid = q_head * tokens + token.
// query   : [num_q_heads * head_dim]
// keys    : [tokens * kv_width]   (layer-offset slice supplied by the caller)
// scores  : [num_q_heads * tokens]  (output)
kernel void gqa_attention_scores(
    device const float* query       [[buffer(0)]],
    device const float* keys        [[buffer(1)]],
    device float*       scores      [[buffer(2)]],
    constant uint& head_dim         [[buffer(3)]],
    constant uint& groups_per_kv    [[buffer(4)]],
    constant uint& tokens           [[buffer(5)]],
    constant uint& kv_width         [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    uint q_head  = tid / max(tokens, 1u);
    uint token   = tid % max(tokens, 1u);
    uint kv_head = q_head / max(groups_per_kv, 1u);
    float acc = 0.0f;
    uint q_base = q_head  * head_dim;
    uint k_base = token   * kv_width + kv_head * head_dim;
    for (uint d = 0; d < head_dim; ++d) {
        acc = fma(query[q_base + d], keys[k_base + d], acc);
    }
    scores[q_head * max(tokens, 1u) + token] = acc * rsqrt(float(max(head_dim, 1u)));
}

// Multi-head GQA weighted value aggregation.
// One thread per output element idx = q_head * head_dim + d.
// scores  : [num_q_heads * tokens]  (softmax-normalised per Q-head, supplied by caller)
// values  : [tokens * kv_width]     (layer-offset slice supplied by the caller)
// output  : [num_q_heads * head_dim]
kernel void gqa_kv_read_attention(
    device const float* scores      [[buffer(0)]],
    device const float* values      [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant uint& head_dim         [[buffer(3)]],
    constant uint& groups_per_kv    [[buffer(4)]],
    constant uint& tokens           [[buffer(5)]],
    constant uint& kv_width         [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
    uint q_head  = idx / max(head_dim, 1u);
    uint d       = idx % max(head_dim, 1u);
    uint kv_head = q_head / max(groups_per_kv, 1u);
    float acc = 0.0f;
    for (uint token = 0; token < tokens; ++token) {
        float w = scores[q_head * max(tokens, 1u) + token];
        float v = values[token * kv_width + kv_head * head_dim + d];
        acc = fma(w, v, acc);
    }
    output[idx] = acc;
}

kernel void linear_conv1d_step(
    device float* conv_state [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const ushort* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& conv_dim [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= conv_dim || kernel_size == 0) {
        return;
    }
    float acc = 0.0f;
    uint w_base = idx * kernel_size;
    for (uint k = 0; k + 1 < kernel_size; ++k) {
        acc = fma(conv_state[k * conv_dim + idx], bf16_to_float(weights[w_base + k]), acc);
    }
    float inp = input[idx];
    acc = fma(inp, bf16_to_float(weights[w_base + kernel_size - 1]), acc);
    output[idx] = acc / (1.0f + exp(-acc));
    for (uint k = 0; k + 2 < kernel_size; ++k) {
        conv_state[k * conv_dim + idx] = conv_state[(k + 1) * conv_dim + idx];
    }
    if (kernel_size > 1) {
        conv_state[(kernel_size - 2) * conv_dim + idx] = inp;
    }
}

kernel void linear_rms_norm_qk(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant uint& key_dim [[buffer(2)]],
    constant float& inv_scale [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    uint base = head * key_dim;
    threadgroup float partial[256];
    float qval = (tid < key_dim) ? q[base + tid] : 0.0f;
    partial[tid] = qval * qval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < key_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(key_dim, 1u)) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < key_dim) {
        q[base + tid] = qval * partial[0] * inv_scale * inv_scale;
    }

    float kval = (tid < key_dim) ? k[base + tid] : 0.0f;
    partial[tid] = kval * kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < key_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(key_dim, 1u)) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < key_dim) {
        k[base + tid] = kval * partial[0] * inv_scale;
    }
}

kernel void linear_compute_decay_beta(
    device const float* alpha [[buffer(0)]],
    device const float* beta [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const ushort* dt_bias [[buffer(3)]],
    device float* g_decay [[buffer(4)]],
    device float* beta_gate [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= heads) {
        return;
    }
    float softplus_value = log(1.0f + exp(alpha[idx] + bf16_to_float(dt_bias[idx])));
    g_decay[idx] = exp(-exp(a_log[idx]) * softplus_value);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta[idx]));
}

kernel void linear_gated_delta_step(
    device float* state [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device const float* g_decay [[buffer(4)]],
    device const float* beta_gate [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& key_dim [[buffer(7)]],
    constant uint& value_dim [[buffer(8)]],
    constant uint& k_heads_per_v [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]) {
    if (vi >= value_dim || key_dim > 256) {
        return;
    }
    uint key_head = head / max(k_heads_per_v, 1u);
    uint state_base = head * value_dim * key_dim + vi * key_dim;
    uint key_base = key_head * key_dim;
    uint value_base = head * value_dim;
    float decay = g_decay[head];
    float beta = beta_gate[head];
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < key_dim; ++ki) {
        float s = state[state_base + ki] * decay;
        state[state_base + ki] = s;
        kv_mem = fma(s, k[key_base + ki], kv_mem);
    }
    float delta = (v[value_base + vi] - kv_mem) * beta;
    for (uint ki = 0; ki < key_dim; ++ki) {
        state[state_base + ki] = fma(k[key_base + ki], delta, state[state_base + ki]);
    }
    float out_value = 0.0f;
    for (uint ki = 0; ki < key_dim; ++ki) {
        out_value = fma(state[state_base + ki], q[key_base + ki], out_value);
    }
    output[value_base + vi] = out_value;
}

kernel void linear_gated_rms_norm(
    device const float* values [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const ushort* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& value_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    uint base = head * value_dim;
    threadgroup float partial[256];
    float value = (tid < value_dim) ? values[base + tid] : 0.0f;
    partial[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < value_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(value_dim, 1u)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < value_dim) {
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));
        output[base + tid] = value * partial[0] * gate * bf16_to_float(weight[tid]);
    }
}
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalCommandContext {
    operation: String,
    details: Vec<(String, String)>,
}

impl MetalCommandContext {
    pub(crate) fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            details: Vec::new(),
        }
    }

    pub(crate) fn with(mut self, key: impl Into<String>, value: impl ToString) -> Self {
        self.details.push((key.into(), value.to_string()));
        self
    }

    pub(crate) fn label(&self) -> String {
        let mut label = format!("Flash-MoE {}", self.operation);
        for (key, value) in &self.details {
            label.push(' ');
            label.push_str(key);
            label.push('=');
            label.push_str(value);
        }
        label
    }

    pub(crate) fn detail_summary(&self) -> String {
        if self.details.is_empty() {
            "none".to_string()
        } else {
            self.details
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalCommandStatus {
    NotEnqueued,
    Enqueued,
    Committed,
    Scheduled,
    Completed,
    Error,
    Unknown(usize),
}

impl MetalCommandStatus {
    pub(crate) fn from_raw(raw: usize) -> Self {
        match raw {
            0 => Self::NotEnqueued,
            1 => Self::Enqueued,
            2 => Self::Committed,
            3 => Self::Scheduled,
            4 => Self::Completed,
            5 => Self::Error,
            value => Self::Unknown(value),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::NotEnqueued => "not_enqueued",
            Self::Enqueued => "enqueued",
            Self::Committed => "committed",
            Self::Scheduled => "scheduled",
            Self::Completed => "completed",
            Self::Error => "error",
            Self::Unknown(_) => "unknown",
        }
    }

    pub(crate) fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Error)
    }
}

impl std::fmt::Display for MetalCommandStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(raw) => write!(f, "unknown({raw})"),
            status => f.write_str(status.as_str()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalCommandFailureKind {
    Timeout,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalCommandBufferFailure {
    kind: MetalCommandFailureKind,
    message: String,
}

impl MetalCommandBufferFailure {
    pub(crate) fn timeout(
        context: &MetalCommandContext,
        elapsed: Duration,
        status: MetalCommandStatus,
        metal_error: Option<String>,
    ) -> Self {
        Self {
            kind: MetalCommandFailureKind::Timeout,
            message: format_metal_command_failure(
                MetalCommandFailureKind::Timeout,
                context,
                elapsed,
                status,
                metal_error.as_deref(),
            ),
        }
    }

    pub(crate) fn failed(
        context: &MetalCommandContext,
        elapsed: Duration,
        status: MetalCommandStatus,
        metal_error: Option<String>,
    ) -> Self {
        Self {
            kind: MetalCommandFailureKind::Failed,
            message: format_metal_command_failure(
                MetalCommandFailureKind::Failed,
                context,
                elapsed,
                status,
                metal_error.as_deref(),
            ),
        }
    }

    pub(crate) fn should_release_buffers(&self) -> bool {
        true
    }
}

impl std::fmt::Display for MetalCommandBufferFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MetalCommandBufferFailure {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCommandWaitPolicy {
    pub(crate) timeout: Duration,
    pub(crate) poll_interval: Duration,
}

impl Default for MetalCommandWaitPolicy {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_FLASHMOE_METAL_COMMAND_TIMEOUT,
            poll_interval: DEFAULT_FLASHMOE_METAL_COMMAND_POLL_INTERVAL,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MetalCommandWaitResult {
    Pending,
    Finished(std::result::Result<(), MetalCommandBufferFailure>),
}

pub(crate) fn resolve_metal_command_wait(
    context: &MetalCommandContext,
    elapsed: Duration,
    status: MetalCommandStatus,
    metal_error: Option<String>,
    timed_out: bool,
) -> MetalCommandWaitResult {
    if status.is_terminal() {
        return match status {
            MetalCommandStatus::Completed if metal_error.is_none() => {
                MetalCommandWaitResult::Finished(Ok(()))
            }
            _ => MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure::failed(
                context,
                elapsed,
                status,
                metal_error,
            ))),
        };
    }
    if timed_out {
        return MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure::timeout(
            context,
            elapsed,
            status,
            metal_error,
        )));
    }
    MetalCommandWaitResult::Pending
}

pub(crate) fn metal_command_failure_requires_release(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<MetalCommandBufferFailure>()
        .is_some_and(MetalCommandBufferFailure::should_release_buffers)
}

pub(crate) fn format_metal_command_failure(
    kind: MetalCommandFailureKind,
    context: &MetalCommandContext,
    elapsed: Duration,
    status: MetalCommandStatus,
    metal_error: Option<&str>,
) -> String {
    let action = match kind {
        MetalCommandFailureKind::Timeout => "timed out",
        MetalCommandFailureKind::Failed => "failed",
    };
    let error = metal_error
        .filter(|error| !error.trim().is_empty())
        .unwrap_or("none reported");
    format!(
        "Flash-MoE Metal command buffer {action}: label=\"{}\", elapsed={}ms, status={}, metal_error=\"{}\", details={}",
        context.label(),
        elapsed.as_millis(),
        status,
        error,
        context.detail_summary()
    )
}

#[cfg(test)]
mod tests {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use super::super::capabilities::{
        FlashMoeGraphStage, FlashMoeStageCapability, FlashMoeStageImplementation,
        FlashMoeStagePlacement,
    };
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use super::super::experts::Q4MatvecPayload;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use super::super::scheduler::ScheduledRoutingTopK;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use super::super::weights::DenseQ4MmapMatvecProjection;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use super::super::weights::SharedExpertPhaseShape;
    use super::*;

    #[test]
    fn command_context_label_includes_actionable_details() {
        let context = MetalCommandContext::new("deferred_expert_phase")
            .with("position", 17)
            .with("layer", 3)
            .with("experts", "1,7,9,11")
            .with("width", 4096);

        assert_eq!(
            context.label(),
            "Flash-MoE deferred_expert_phase position=17 layer=3 experts=1,7,9,11 width=4096"
        );
        assert_eq!(
            context.detail_summary(),
            "position=17, layer=3, experts=1,7,9,11, width=4096"
        );
    }

    #[test]
    fn command_status_names_known_and_unknown_values() {
        assert_eq!(MetalCommandStatus::from_raw(0).to_string(), "not_enqueued");
        assert_eq!(MetalCommandStatus::from_raw(3).to_string(), "scheduled");
        assert_eq!(MetalCommandStatus::from_raw(4).to_string(), "completed");
        assert_eq!(MetalCommandStatus::from_raw(5).to_string(), "error");
        assert_eq!(MetalCommandStatus::from_raw(99).to_string(), "unknown(99)");
        assert!(MetalCommandStatus::Completed.is_terminal());
        assert!(MetalCommandStatus::Error.is_terminal());
        assert!(!MetalCommandStatus::Scheduled.is_terminal());
    }

    #[test]
    fn command_failure_diagnostic_is_actionable() {
        let context = MetalCommandContext::new("gqa_attention_scores")
            .with("layer", 12)
            .with("position", 128)
            .with("tokens", 129)
            .with("q_heads", 32)
            .with("kv_heads", 8);

        let message = format_metal_command_failure(
            MetalCommandFailureKind::Timeout,
            &context,
            Duration::from_millis(1234),
            MetalCommandStatus::Scheduled,
            Some("GPU timeout"),
        );

        assert!(message.contains("timed out"));
        assert!(message.contains("label=\"Flash-MoE gqa_attention_scores"));
        assert!(message.contains("elapsed=1234ms"));
        assert!(message.contains("status=scheduled"));
        assert!(message.contains("metal_error=\"GPU timeout\""));
        assert!(message.contains("layer=12"));
        assert!(message.contains("position=128"));
        assert!(message.contains("tokens=129"));
    }

    #[test]
    fn command_failure_marks_buffers_for_release() {
        let context = MetalCommandContext::new("lm_head_topk").with("rows", 42);
        let error = MetalCommandBufferFailure::failed(
            &context,
            Duration::from_millis(7),
            MetalCommandStatus::Error,
            None,
        );
        let anyhow_error = anyhow::Error::from(error.clone());

        assert!(error.should_release_buffers());
        assert!(metal_command_failure_requires_release(&anyhow_error));
        assert!(error.to_string().contains("none reported"));
    }

    #[test]
    fn command_wait_policy_uses_upstream_shaped_timeout_defaults() {
        let policy = MetalCommandWaitPolicy::default();
        assert_eq!(policy.timeout, Duration::from_secs(120));
        assert_eq!(policy.poll_interval, Duration::from_millis(2));
    }

    #[test]
    fn command_wait_resolution_handles_completed_failed_timeout_and_pending() {
        let context = MetalCommandContext::new("cmd3");

        assert_eq!(
            resolve_metal_command_wait(
                &context,
                Duration::from_millis(4),
                MetalCommandStatus::Completed,
                None,
                false,
            ),
            MetalCommandWaitResult::Finished(Ok(()))
        );

        let failed = resolve_metal_command_wait(
            &context,
            Duration::from_millis(5),
            MetalCommandStatus::Error,
            Some("encoder failed".to_string()),
            false,
        );
        assert!(matches!(
            failed,
            MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure {
                kind: MetalCommandFailureKind::Failed,
                ..
            }))
        ));

        let timed_out = resolve_metal_command_wait(
            &context,
            Duration::from_secs(120),
            MetalCommandStatus::Scheduled,
            Some("still running".to_string()),
            true,
        );
        assert!(matches!(
            timed_out,
            MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure {
                kind: MetalCommandFailureKind::Timeout,
                ..
            }))
        ));

        assert_eq!(
            resolve_metal_command_wait(
                &context,
                Duration::from_millis(1),
                MetalCommandStatus::Scheduled,
                None,
                false,
            ),
            MetalCommandWaitResult::Pending
        );
    }

    #[test]
    fn shader_source_defines_full_forward_kernel_set() {
        for kernel in REQUIRED_FORWARD_KERNELS {
            assert!(
                METAL_SHADERS.contains(&format!("kernel void {kernel}")),
                "missing Metal kernel {kernel}"
            );
        }
        assert!(METAL_SHADERS.contains("threadgroup float input_cache"));
        assert!(METAL_SHADERS.contains("simd_sum(acc)"));
        assert!(METAL_SHADERS.contains("thread_index_in_simdgroup"));
        assert!(METAL_SHADERS.contains("constant uint& group_size"));
        assert!(METAL_SHADERS.contains("constant uint& top_k"));
        assert!(METAL_SHADERS.contains("fma(float(byte & 0x0f), scale0 * x0, bias0 * x0)"));
        assert!(
            !METAL_SHADERS.contains("uint half"),
            "`half` is a Metal scalar type and cannot be reused as a variable name"
        );
    }

    #[test]
    fn pipeline_name_set_declares_optional_route_top4() {
        let without_route = MetalPipelineNameSet::new(false);
        assert_eq!(without_route.route_top4, None);
        assert!(!without_route.kernel_names().contains(&kernels::ROUTE_TOP4));

        let with_route = MetalPipelineNameSet::new(true);
        assert_eq!(with_route.route_top4, Some(kernels::ROUTE_TOP4));
        assert!(with_route.kernel_names().contains(&kernels::ROUTE_TOP4));
    }

    #[test]
    fn pipeline_name_set_matches_declared_forward_kernel_surface() {
        let mut compiled = MetalPipelineNameSet::new(true).kernel_names();
        compiled.sort_unstable();
        compiled.dedup();

        let mut required = REQUIRED_FORWARD_KERNELS.to_vec();
        required.sort_unstable();
        required.dedup();

        assert_eq!(compiled, required);
    }

    #[test]
    fn pipeline_set_release_order_includes_optional_route_pipeline() {
        let without_route = test_pipeline_set(None);
        let mut released = Vec::new();
        without_route.release_with(|pipeline| released.push(pipeline));
        assert_eq!(released.first(), Some(&1));
        assert!(!released.contains(&9));
        assert_eq!(released.last(), Some(&36));

        let with_route = test_pipeline_set(Some(9));
        let mut released = Vec::new();
        with_route.release_with(|pipeline| released.push(pipeline));
        assert_eq!(released.first(), Some(&1));
        assert!(released.contains(&9));
        assert_eq!(released.last(), Some(&36));
    }

    fn test_pipeline_set(route_pipeline: Option<i32>) -> MetalPipelineSet<i32> {
        MetalPipelineSet {
            q4_pipeline: 1,
            q4_bf16_scale_bias_pipeline: 2,
            q4_swiglu_pipeline: 3,
            q4_swiglu_bf16_scale_bias_pipeline: 4,
            q4_mmap_pipeline: 5,
            q4_mmap_bf16_scale_bias_pipeline: 6,
            q4_mmap_batch_pipeline: 7,
            q4_mmap_batch_bf16_scale_bias_pipeline: 8,
            route_pipeline,
            dense_matvec_pipeline: 10,
            dense_matvec_bf16_pipeline: 11,
            dense_mmap_matvec_pipeline: 12,
            dense_mmap_matvec_bf16_pipeline: 13,
            dense_mmap_matvec_bf16_simd_pipeline: 14,
            rms_norm_pipeline: 15,
            rms_norm_reduced_pipeline: 16,
            residual_rms_norm_pipeline: 17,
            rope_pipeline: 18,
            rope_split_half_pipeline: 19,
            attention_pipeline: 20,
            kv_write_pipeline: 21,
            kv_read_attention_pipeline: 22,
            expert_mlp_pipeline: 23,
            silu_product_pipeline: 24,
            shared_expert_activation_pipeline: 25,
            combine_expert_phase_pipeline: 26,
            fill_zero_pipeline: 27,
            lm_head_pipeline: 28,
            topk_vocab_pipeline: 29,
            gqa_scores_pipeline: 30,
            gqa_read_pipeline: 31,
            linear_conv1d_pipeline: 32,
            linear_rms_norm_qk_pipeline: 33,
            linear_decay_beta_pipeline: 34,
            linear_delta_step_pipeline: 35,
            linear_gated_rms_norm_pipeline: 36,
        }
    }

    #[test]
    fn attention_policy_declares_cpu_then_gpu_context_boundary() {
        let policy = MetalAttentionPolicy;

        assert_eq!(policy.backend(0), MetalAttentionBackend::Cpu);
        assert_eq!(
            policy.backend(DEFAULT_METAL_ATTENTION_CPU_MAX_TOKENS),
            MetalAttentionBackend::Cpu
        );
        assert_eq!(
            policy.backend(DEFAULT_METAL_ATTENTION_CPU_MAX_TOKENS + 1),
            MetalAttentionBackend::Gpu
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn linear_attention_static_offsets_preserve_resident_weight_bindings() {
        let offsets = MetalLinearAttentionStaticOffsets::new(16, 32, 48, 64);

        assert_eq!(offsets.conv_weight_byte_offset, 16);
        assert_eq!(offsets.a_log_byte_offset, 32);
        assert_eq!(offsets.dt_bias_byte_offset, 48);
        assert_eq!(offsets.norm_weight_byte_offset, 64);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn batch_projection_input_reports_declared_input_len() {
        let values = [1.0, 2.0, 3.0];
        assert_eq!(MetalBatchProjectionInput::Cpu(&values).len(), values.len());

        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        assert_eq!(
            MetalBatchProjectionInput::Buffer { buffer: id, len: 7 }.len(),
            7
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn projection_batch_declares_metal_output_layout() {
        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let batch = MetalProjectionBatch::new(id, vec![0, 4, 12], vec![4, 8, 2], 14);

        assert_eq!(batch.output_buffer, id);
        assert_eq!(batch.output_offsets, vec![0, 4, 12]);
        assert_eq!(batch.output_widths, vec![4, 8, 2]);
        assert_eq!(batch.total_rows, 14);

        let empty = MetalProjectionBatch::empty();
        assert!(empty.output_buffer.is_null());
        assert!(empty.output_offsets.is_empty());
        assert!(empty.output_widths.is_empty());
        assert_eq!(empty.total_rows, 0);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn attention_values_declares_metal_buffer_and_len() {
        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let values = MetalAttentionValues::new(id, 64);

        assert_eq!(values.buffer, id);
        assert_eq!(values.len, 64);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn post_attention_prep_builds_declared_cmd3_metal_input() {
        let mut prep = MetalPostAttentionPrep::new(
            3,
            8,
            16,
            vec![(2, 0.75), (5, 0.25)],
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .unwrap();

        assert_eq!(prep.width, 8);
        assert_eq!(prep.active, vec![(2, 0.75), (5, 0.25)]);
        assert!(prep.routing_command().is_none());
        assert_eq!(prep.input.state(), prep.state);
        assert!(prep.state.is_declared_graph_state());

        let command = test_fused_prep_routing_command(3, 16, &prep.active);
        let attached = prep.attach_routing_command(command.clone()).unwrap();
        assert_eq!(attached, command);
        assert_eq!(prep.routing_command(), Some(&command));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn post_attention_prep_rejects_mismatched_routing_command() {
        let mut prep = MetalPostAttentionPrep::new(
            3,
            8,
            16,
            vec![(2, 0.75), (5, 0.25)],
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .unwrap();
        let command = test_fused_prep_routing_command(4, 16, &prep.active);

        let err = prep.attach_routing_command(command).unwrap_err();
        assert!(err.to_string().contains("routing layer 3"), "{err:#}");
        assert!(prep.routing_command().is_none());
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn post_attention_prep_rejects_undeclared_cmd3_metal_input() {
        let err = MetalPostAttentionPrep::new(
            3,
            0,
            16,
            vec![(2, 1.0)],
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("Metal post-attention input for layer 3")
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_deferred_output_declares_gpu_resident_buffers() {
        let hidden = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let next_normed = hidden;
        let output = MetalCmd3DeferredOutput::new(
            hidden,
            Some(next_normed),
            FlashMoeCmd3OutputState::gpu_resident(16, true),
        )
        .unwrap();

        assert_eq!(output.hidden_buffer, hidden);
        assert_eq!(output.next_normed_buffer, Some(next_normed));
        assert_eq!(output.output_state.width(), 16);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_deferred_output_rejects_undeclared_buffer_state() {
        let hidden = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();

        let missing_next = MetalCmd3DeferredOutput::new(
            hidden,
            None,
            FlashMoeCmd3OutputState::gpu_resident(16, true),
        )
        .unwrap_err();
        assert!(
            missing_next
                .to_string()
                .contains("next-norm buffer presence"),
            "{missing_next:#}"
        );

        let null_hidden = MetalCmd3DeferredOutput::new(
            std::ptr::null_mut(),
            None,
            FlashMoeCmd3OutputState::gpu_resident(16, false),
        )
        .unwrap_err();
        assert!(
            null_hidden.to_string().contains("non-null hidden buffer"),
            "{null_hidden:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_phase_plan_declares_supported_command_shape() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(16, true);

        let plan = MetalCmd3PhasePlan::new(9, 3, 4, 16, 4, 4, output_state, true).unwrap();

        assert_eq!(plan.position, 9);
        assert_eq!(plan.layer, 3);
        assert_eq!(plan.expert_count, 4);
        assert_eq!(plan.width, 16);
        assert_eq!(plan.output_state, output_state);
        assert!(plan.has_next_norm);
        assert_eq!(plan.width_u32(), 16);
        assert_eq!(plan.expert_outputs_bytes().unwrap(), 4 * 16 * 4);
        assert_eq!(plan.shared_output_bytes().unwrap(), 16 * 4);
        assert_eq!(plan.hidden_output_bytes().unwrap(), 16 * 4);
        assert_eq!(plan.next_normed_output_bytes().unwrap(), Some(16 * 4));
        assert_eq!(plan.expert_output_offset(0).unwrap(), 0);
        assert_eq!(plan.expert_output_offset(3).unwrap(), 3 * 16 * 4);

        let combine = MetalCmd3CombinePlan::new(plan);
        assert_eq!(combine.width, 16);
        assert_eq!(combine.active_count, 4);
        assert_eq!(combine.active_count_u32(), 4);
        assert_eq!(combine.dispatch_threads, 16);
        assert_eq!(combine.routing_weights_bytes().unwrap(), 4 * 4);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_phase_plan_rejects_unsupported_command_shape() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(16, true);

        let count_err = MetalCmd3PhasePlan::new(9, 3, 4, 16, 3, 4, output_state, true).unwrap_err();
        assert!(
            count_err.to_string().contains("expert count 4"),
            "{count_err:#}"
        );

        let output_err =
            MetalCmd3PhasePlan::new(9, 3, 4, 16, 4, 4, output_state, false).unwrap_err();
        assert!(
            output_err
                .to_string()
                .contains("next-norm output declaration"),
            "{output_err:#}"
        );

        let wide_state = FlashMoeCmd3OutputState::gpu_resident(u32::MAX as usize + 1, false);
        let width_err =
            MetalCmd3PhasePlan::new(9, 3, 4, u32::MAX as usize + 1, 4, 4, wide_state, false)
                .unwrap_err();
        assert!(
            width_err.to_string().contains("does not fit Metal u32"),
            "{width_err:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_input_buffers_carry_declared_phase_inputs() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();

        let inputs = MetalCmd3InputBuffers::new(
            phase,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
        )
        .unwrap();

        assert_eq!(inputs.normed, 0x1000usize as MetalObjcId);
        assert_eq!(inputs.residual, 0x2000usize as MetalObjcId);
        assert_eq!(inputs.phase, phase);

        let missing_normed =
            MetalCmd3InputBuffers::new(phase, std::ptr::null_mut(), inputs.residual).unwrap_err();
        assert!(
            missing_normed.to_string().contains("non-null normed"),
            "{missing_normed:#}"
        );

        let missing_residual =
            MetalCmd3InputBuffers::new(phase, inputs.normed, std::ptr::null_mut()).unwrap_err();
        assert!(
            missing_residual.to_string().contains("non-null residual"),
            "{missing_residual:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_combine_buffers_carry_declared_bindings() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
        let plan = MetalCmd3CombinePlan::new(phase);

        let buffers = MetalCmd3CombineBuffers::new(
            plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
        )
        .unwrap();

        assert_eq!(buffers.routing_weights, 0x1000usize as MetalObjcId);
        assert_eq!(buffers.width, 0x2000usize as MetalObjcId);
        assert_eq!(buffers.active_count, 0x3000usize as MetalObjcId);
        assert_eq!(buffers.layout.width_u32, 4);
        assert_eq!(buffers.layout.active_count_u32, 2);
        assert_eq!(buffers.layout.routing_weights_bytes, 2 * 4);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_combine_stage_buffers_match_declared_layout() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let payloads = vec![
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
        ];
        let execution = MetalCmd3ExecutionPlan::new(
            9,
            3,
            2,
            4,
            2,
            output_state,
            ScheduledSharedExpertPhaseRef::None,
            None,
            &payloads,
        )
        .unwrap();
        let inputs = MetalCmd3InputBuffers::new(
            execution.phase,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
        )
        .unwrap();
        let outputs = MetalCmd3OutputBuffers::new(
            &execution,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
            0x5000usize as MetalObjcId,
            None,
        )
        .unwrap();
        let combine = MetalCmd3CombineBuffers::new(
            execution.combine,
            0x6000usize as MetalObjcId,
            0x7000usize as MetalObjcId,
            0x8000usize as MetalObjcId,
        )
        .unwrap();

        let stage = MetalCmd3CombineStageBuffers::new(execution.combine, inputs, &outputs, combine)
            .unwrap();

        assert_eq!(stage.residual, inputs.residual);
        assert_eq!(stage.shared_output, outputs.shared_output);
        assert_eq!(stage.expert_outputs, outputs.expert_outputs);
        assert_eq!(stage.routing_weights, combine.routing_weights);
        assert_eq!(stage.hidden, outputs.hidden);
        assert_eq!(stage.width, combine.width);
        assert_eq!(stage.active_count, combine.active_count);
        assert_eq!(stage.plan, execution.combine);

        let stale_plan = MetalCmd3CombinePlan {
            width: execution.combine.width,
            active_count: execution.combine.active_count + 1,
            dispatch_threads: execution.combine.dispatch_threads,
        };
        let stale =
            MetalCmd3CombineStageBuffers::new(stale_plan, inputs, &outputs, combine).unwrap_err();
        assert!(
            stale.to_string().contains("constants do not match plan"),
            "{stale:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_next_norm_buffers_carry_declared_bindings() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, true);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, true).unwrap();
        let plan = MetalCmd3NextNormPlan::new(phase, Some(4)).unwrap().unwrap();

        let buffers = MetalCmd3NextNormBuffers::new(
            plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
        )
        .unwrap();

        assert_eq!(buffers.hidden, 0x1000usize as MetalObjcId);
        assert_eq!(buffers.weight, 0x2000usize as MetalObjcId);
        assert_eq!(buffers.next_normed, 0x3000usize as MetalObjcId);
        assert_eq!(buffers.width, 0x4000usize as MetalObjcId);
        assert_eq!(buffers.layout.width_u32, 4);
        assert_eq!(buffers.layout.weight_bytes, 4 * 4);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_shared_stage_buffers_require_declared_source() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let shared = SharedExpertPhaseQ4Projections {
            gate: test_q4_projection("gate", 6, 4),
            up: test_q4_projection("up", 6, 4),
            down: test_q4_projection("down", 4, 6),
            router: test_q4_projection("router", 2, 4),
            shared_experts: 2,
            intermediate: 3,
            width: 4,
        };
        let payloads = vec![
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
        ];
        let plan = MetalCmd3ExecutionPlan::new(
            9,
            3,
            2,
            4,
            2,
            output_state,
            ScheduledSharedExpertPhaseRef::Q4(&shared),
            None,
            &payloads,
        )
        .unwrap();
        let inputs = MetalCmd3InputBuffers::new(
            plan.phase,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
        )
        .unwrap();
        let outputs = MetalCmd3OutputBuffers::new(
            &plan,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
            0x5000usize as MetalObjcId,
            None,
        )
        .unwrap();
        let combine = MetalCmd3CombineBuffers::new(
            plan.combine,
            0x6000usize as MetalObjcId,
            0x7000usize as MetalObjcId,
            0x8000usize as MetalObjcId,
        )
        .unwrap();
        let work = MetalCmd3SharedWorkBuffers::new(
            plan.shared,
            0x9000usize as MetalObjcId,
            0xa000usize as MetalObjcId,
            0xb000usize as MetalObjcId,
            0xc000usize as MetalObjcId,
            0xd000usize as MetalObjcId,
            0xe000usize as MetalObjcId,
        )
        .unwrap();

        let projected =
            MetalCmd3SharedStageBuffers::projected(plan.shared, inputs, &outputs, combine, work)
                .unwrap();

        assert_eq!(projected.source, MetalCmd3SharedPhaseSource::ResidentQ4);
        assert_eq!(projected.normed, inputs.normed);
        assert_eq!(projected.width, combine.width);
        assert_eq!(projected.shared_output, outputs.shared_output);
        assert_eq!(projected.work, Some(work));

        let no_shared = MetalCmd3SharedPhasePlan::none(4);
        let fill_zero =
            MetalCmd3SharedStageBuffers::fill_zero(no_shared, inputs, &outputs, combine).unwrap();
        assert_eq!(fill_zero.source, MetalCmd3SharedPhaseSource::None);
        assert_eq!(fill_zero.work, None);
        assert_eq!(fill_zero.shared_output, outputs.shared_output);

        let projected_none =
            MetalCmd3SharedStageBuffers::projected(no_shared, inputs, &outputs, combine, work)
                .unwrap_err();
        assert!(
            projected_none
                .to_string()
                .contains("declared shared expert source"),
            "{projected_none:#}"
        );

        let fill_projected =
            MetalCmd3SharedStageBuffers::fill_zero(plan.shared, inputs, &outputs, combine)
                .unwrap_err();
        assert!(
            fill_projected
                .to_string()
                .contains("no shared expert source"),
            "{fill_projected:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_execution_plan_declares_full_command_topology() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, true);
        let shared = SharedExpertPhaseQ4Projections {
            gate: test_q4_projection("gate", 6, 4),
            up: test_q4_projection("up", 6, 4),
            down: test_q4_projection("down", 4, 6),
            router: test_q4_projection("router", 2, 4),
            shared_experts: 2,
            intermediate: 3,
            width: 4,
        };
        let payloads = vec![
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
        ];

        let plan = MetalCmd3ExecutionPlan::new(
            9,
            3,
            2,
            4,
            2,
            output_state,
            ScheduledSharedExpertPhaseRef::Q4(&shared),
            Some(4),
            &payloads,
        )
        .unwrap();

        assert_eq!(plan.phase.position, 9);
        assert_eq!(plan.phase.layer, 3);
        assert_eq!(plan.phase.output_state, output_state);
        assert_eq!(plan.shared.source, MetalCmd3SharedPhaseSource::ResidentQ4);
        assert_eq!(plan.shared.total_intermediate, 6);
        assert_eq!(plan.active_experts.len(), 2);
        assert_eq!(plan.active_experts[0].intermediate, 5);
        assert_eq!(plan.active_experts[0].output_offset, 0);
        assert_eq!(plan.active_experts[1].intermediate, 7);
        assert_eq!(plan.active_experts[1].output_offset, 4 * 4);
        assert_eq!(plan.combine.active_count, 2);
        assert_eq!(plan.next_norm.unwrap().width, 4);

        let layout = plan.buffer_layout().unwrap();
        assert_eq!(layout.width_u32, 4);
        assert_eq!(layout.active_count_u32, 2);
        assert_eq!(layout.expert_outputs_bytes, 2 * 4 * 4);
        assert_eq!(layout.shared_output_bytes, 4 * 4);
        assert_eq!(layout.hidden_output_bytes, 4 * 4);
        assert_eq!(layout.next_normed_output_bytes, Some(4 * 4));

        let context = plan.command_context("1,7");
        assert_eq!(
            context.label(),
            "Flash-MoE deferred_expert_phase_from_buffers position=9 layer=3 active_experts=2 experts=1,7 width=4 shared=true next_norm=true"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_output_buffers_match_declared_output_state() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, true);
        let payloads = vec![ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(
            5, 4,
        ))];
        let plan = MetalCmd3ExecutionPlan::new(
            9,
            3,
            1,
            4,
            1,
            output_state,
            ScheduledSharedExpertPhaseRef::None,
            Some(4),
            &payloads,
        )
        .unwrap();

        let buffers = MetalCmd3OutputBuffers::new(
            &plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            Some(0x4000usize as MetalObjcId),
        )
        .unwrap();

        assert_eq!(buffers.layout.width_u32, 4);
        assert_eq!(buffers.layout.active_count_u32, 1);
        assert_eq!(buffers.layout.expert_outputs_bytes, 4 * 4);
        assert_eq!(buffers.layout.shared_output_bytes, 4 * 4);
        assert_eq!(buffers.layout.hidden_output_bytes, 4 * 4);
        assert_eq!(buffers.layout.next_normed_output_bytes, Some(4 * 4));
        assert_eq!(buffers.hidden, 0x3000usize as MetalObjcId);

        let missing_next = MetalCmd3OutputBuffers::new(
            &plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            None,
        )
        .unwrap_err();
        assert!(
            missing_next
                .to_string()
                .contains("does not match declared output state"),
            "{missing_next:#}"
        );

        let no_next_plan = MetalCmd3ExecutionPlan::new(
            9,
            3,
            1,
            4,
            1,
            FlashMoeCmd3OutputState::gpu_resident(4, false),
            ScheduledSharedExpertPhaseRef::None,
            None,
            &payloads,
        )
        .unwrap();
        let unexpected_next = MetalCmd3OutputBuffers::new(
            &no_next_plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            Some(0x4000usize as MetalObjcId),
        )
        .unwrap_err();
        assert!(
            unexpected_next
                .to_string()
                .contains("does not match declared output state"),
            "{unexpected_next:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_active_expert_work_buffers_carry_declared_layout() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
        let payload = test_q4_expert_payload(6, 4);
        let plan = MetalCmd3ActiveExpertPlan::new(phase, 1, &payload).unwrap();

        let fused =
            MetalCmd3ActiveExpertWorkBuffers::fused(plan, 0x1000usize as MetalObjcId).unwrap();

        assert_eq!(fused.activated, 0x1000usize as MetalObjcId);
        assert_eq!(fused.gate_out, None);
        assert_eq!(fused.up_out, None);
        assert_eq!(fused.intermediate, None);
        assert_eq!(fused.layout.intermediate_u32, 6);
        assert_eq!(fused.layout.activation_bytes, 6 * 4);
        assert_eq!(fused.layout.projection_output_bytes, 6 * 4);

        let unfused = MetalCmd3ActiveExpertWorkBuffers::unfused(
            plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
        )
        .unwrap();

        assert_eq!(unfused.activated, 0x1000usize as MetalObjcId);
        assert_eq!(unfused.gate_out, Some(0x2000usize as MetalObjcId));
        assert_eq!(unfused.up_out, Some(0x3000usize as MetalObjcId));
        assert_eq!(unfused.intermediate, Some(0x4000usize as MetalObjcId));
        assert_eq!(unfused.layout, fused.layout);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_active_expert_stage_buffers_match_declared_layout() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let payloads = vec![
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
        ];
        let execution = MetalCmd3ExecutionPlan::new(
            9,
            3,
            2,
            4,
            2,
            output_state,
            ScheduledSharedExpertPhaseRef::None,
            None,
            &payloads,
        )
        .unwrap();
        let inputs = MetalCmd3InputBuffers::new(
            execution.phase,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
        )
        .unwrap();
        let outputs = MetalCmd3OutputBuffers::new(
            &execution,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
            0x5000usize as MetalObjcId,
            None,
        )
        .unwrap();
        let active_plan = execution.active_experts[1];
        let work = MetalCmd3ActiveExpertWorkBuffers::fused(active_plan, 0x6000usize as MetalObjcId)
            .unwrap();

        let stage =
            MetalCmd3ActiveExpertStageBuffers::new(active_plan, inputs, &outputs, work).unwrap();

        assert_eq!(stage.normed, inputs.normed);
        assert_eq!(stage.activated, work.activated);
        assert_eq!(stage.expert_outputs, outputs.expert_outputs);
        assert_eq!(stage.output_offset, active_plan.output_offset);
        assert_eq!(stage.plan, active_plan);
        assert_eq!(stage.work, work);

        let stale_plan = MetalCmd3ActiveExpertPlan {
            intermediate: active_plan.intermediate + 1,
            ..active_plan
        };
        let stale =
            MetalCmd3ActiveExpertStageBuffers::new(stale_plan, inputs, &outputs, work).unwrap_err();
        assert!(
            stale
                .to_string()
                .contains("work layout does not match plan"),
            "{stale:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_execution_plan_rejects_mismatched_subplans() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let payloads = vec![ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(
            5, 6,
        ))];

        let payload_err = MetalCmd3ExecutionPlan::new(
            9,
            3,
            1,
            4,
            1,
            output_state,
            ScheduledSharedExpertPhaseRef::None,
            None,
            &payloads,
        )
        .unwrap_err();
        assert!(
            payload_err
                .to_string()
                .contains("does not match phase width 4"),
            "{payload_err:#}"
        );

        let shared = SharedExpertPhaseWeights::new(
            Arc::new(vec![1.0; 24]),
            Arc::new(vec![2.0; 24]),
            Arc::new(vec![3.0; 24]),
            Arc::new(vec![4.0; 8]),
            2,
            3,
            4,
        )
        .unwrap();
        let payloads = vec![ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(
            5, 8,
        ))];
        let shared_err = MetalCmd3ExecutionPlan::new(
            9,
            3,
            1,
            8,
            1,
            FlashMoeCmd3OutputState::gpu_resident(8, false),
            ScheduledSharedExpertPhaseRef::Dense(&shared),
            None,
            &payloads,
        )
        .unwrap_err();
        assert!(
            shared_err.to_string().contains("shared expert width 4"),
            "{shared_err:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_next_norm_plan_declares_weight_slice_and_dispatch() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(16, true);
        let phase = MetalCmd3PhasePlan::new(9, 3, 4, 16, 4, 4, output_state, true).unwrap();

        let plan = MetalCmd3NextNormPlan::new(phase, Some(32))
            .unwrap()
            .unwrap();

        assert_eq!(plan.width, 16);
        assert_eq!(plan.dispatch_threads, 256);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_next_norm_plan_rejects_undeclared_or_short_weights() {
        let with_next = MetalCmd3PhasePlan::new(
            9,
            3,
            4,
            16,
            4,
            4,
            FlashMoeCmd3OutputState::gpu_resident(16, true),
            true,
        )
        .unwrap();
        let short = MetalCmd3NextNormPlan::new(with_next, Some(15)).unwrap_err();
        assert!(
            short.to_string().contains("smaller than width 16"),
            "{short:#}"
        );

        let missing = MetalCmd3NextNormPlan::new(with_next, None).unwrap_err();
        assert!(
            missing.to_string().contains("no next-norm weights"),
            "{missing:#}"
        );

        let without_next = MetalCmd3PhasePlan::new(
            9,
            3,
            4,
            16,
            4,
            4,
            FlashMoeCmd3OutputState::gpu_resident(16, false),
            false,
        )
        .unwrap();
        let unexpected = MetalCmd3NextNormPlan::new(without_next, Some(16)).unwrap_err();
        assert!(
            unexpected
                .to_string()
                .contains("provided for a no-next-norm phase"),
            "{unexpected:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_active_expert_plan_declares_payload_layout() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
        let payload = test_q4_expert_payload(6, 4);

        let plan = MetalCmd3ActiveExpertPlan::new(phase, 1, &payload).unwrap();

        assert_eq!(plan.index, 1);
        assert_eq!(plan.intermediate, 6);
        assert_eq!(plan.intermediate_u32().unwrap(), 6);
        assert_eq!(plan.activation_bytes().unwrap(), 6 * 4);
        assert_eq!(plan.projection_output_bytes().unwrap(), 6 * 4);
        assert_eq!(plan.output_offset, 4 * 4);
        assert_eq!(plan.dispatch_threads(), 6);
        assert_eq!(
            plan.buffer_layout().unwrap(),
            MetalCmd3ActiveExpertBufferLayout {
                intermediate_u32: 6,
                activation_bytes: 6 * 4,
                projection_output_bytes: 6 * 4,
            }
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_active_expert_plan_rejects_mismatched_payload() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
        let payload = test_q4_expert_payload(6, 5);

        let err = MetalCmd3ActiveExpertPlan::new(phase, 0, &payload).unwrap_err();

        assert!(
            err.to_string().contains("does not match phase width 4"),
            "{err:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_shared_phase_plan_declares_dense_shape() {
        let shared = SharedExpertPhaseWeights::new(
            Arc::new(vec![1.0; 24]),
            Arc::new(vec![2.0; 24]),
            Arc::new(vec![3.0; 24]),
            Arc::new(vec![4.0; 8]),
            2,
            3,
            4,
        )
        .unwrap();

        let plan = MetalCmd3SharedPhasePlan::dense(4, &shared).unwrap();

        assert_eq!(plan.source, MetalCmd3SharedPhaseSource::Dense);
        assert_eq!(plan.width, 4);
        assert_eq!(plan.shared_experts, 2);
        assert_eq!(plan.intermediate, 3);
        assert_eq!(plan.total_intermediate, 6);
        assert_eq!(plan.total_intermediate_u32().unwrap(), 6);
        assert_eq!(plan.intermediate_u32().unwrap(), 3);
        assert_eq!(plan.projection_output_bytes().unwrap(), 6 * 4);
        assert_eq!(plan.router_output_bytes().unwrap(), 2 * 4);
        assert_eq!(plan.projection_rows(), 6);
        assert_eq!(plan.router_rows(), 2);
        assert_eq!(plan.activation_dispatch_threads(), 6);
        assert_eq!(
            plan.buffer_layout().unwrap(),
            MetalCmd3SharedBufferLayout {
                total_intermediate_u32: 6,
                intermediate_u32: 3,
                projection_output_bytes: 6 * 4,
                router_output_bytes: 2 * 4,
            }
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_shared_phase_plan_declares_resident_q4_shape() {
        let shared = SharedExpertPhaseQ4Projections {
            gate: test_q4_projection("gate", 6, 4),
            up: test_q4_projection("up", 6, 4),
            down: test_q4_projection("down", 4, 6),
            router: test_q4_projection("router", 2, 4),
            shared_experts: 2,
            intermediate: 3,
            width: 4,
        };

        let plan = MetalCmd3SharedPhasePlan::resident_q4(4, &shared).unwrap();

        assert_eq!(plan.source, MetalCmd3SharedPhaseSource::ResidentQ4);
        assert_eq!(plan.width, 4);
        assert_eq!(plan.shared_experts, 2);
        assert_eq!(plan.intermediate, 3);
        assert_eq!(plan.total_intermediate, 6);
        assert_eq!(plan.total_intermediate_u32().unwrap(), 6);
        assert_eq!(plan.intermediate_u32().unwrap(), 3);
        assert_eq!(plan.projection_output_bytes().unwrap(), 6 * 4);
        assert_eq!(plan.router_output_bytes().unwrap(), 2 * 4);
        assert_eq!(plan.projection_rows(), 6);
        assert_eq!(plan.router_rows(), 2);
        assert_eq!(plan.activation_dispatch_threads(), 6);
        assert_eq!(
            plan.buffer_layout().unwrap(),
            MetalCmd3SharedBufferLayout {
                total_intermediate_u32: 6,
                intermediate_u32: 3,
                projection_output_bytes: 6 * 4,
                router_output_bytes: 2 * 4,
            }
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_shared_work_buffers_carry_declared_layout() {
        let shared = SharedExpertPhaseQ4Projections {
            gate: test_q4_projection("gate", 6, 4),
            up: test_q4_projection("up", 6, 4),
            down: test_q4_projection("down", 4, 6),
            router: test_q4_projection("router", 2, 4),
            shared_experts: 2,
            intermediate: 3,
            width: 4,
        };
        let plan = MetalCmd3SharedPhasePlan::resident_q4(4, &shared).unwrap();

        let buffers = MetalCmd3SharedWorkBuffers::new(
            plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
            0x5000usize as MetalObjcId,
            0x6000usize as MetalObjcId,
        )
        .unwrap();

        assert_eq!(buffers.layout.total_intermediate_u32, 6);
        assert_eq!(buffers.layout.intermediate_u32, 3);
        assert_eq!(buffers.layout.projection_output_bytes, 6 * 4);
        assert_eq!(buffers.layout.router_output_bytes, 2 * 4);
        assert_eq!(buffers.gate_out, 0x1000usize as MetalObjcId);

        let err = MetalCmd3SharedWorkBuffers::new(
            MetalCmd3SharedPhasePlan::none(4),
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
            0x5000usize as MetalObjcId,
            0x6000usize as MetalObjcId,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("declared shared expert source"),
            "{err:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_shared_phase_plan_rejects_width_mismatch() {
        let shared = SharedExpertPhaseWeights::new(
            Arc::new(vec![1.0; 24]),
            Arc::new(vec![2.0; 24]),
            Arc::new(vec![3.0; 24]),
            Arc::new(vec![4.0; 8]),
            2,
            3,
            4,
        )
        .unwrap();

        let err = MetalCmd3SharedPhasePlan::dense(8, &shared).unwrap_err();

        assert!(err.to_string().contains("shared expert width 4"), "{err:#}");

        let huge_shape = SharedExpertPhaseShape::new(1, u32::MAX as usize + 1, 1).unwrap();
        let huge_err =
            MetalCmd3SharedPhasePlan::from_shape(MetalCmd3SharedPhaseSource::Dense, 1, huge_shape)
                .unwrap_err();
        assert!(
            huge_err.to_string().contains("does not fit Metal u32"),
            "{huge_err:#}"
        );

        let none = MetalCmd3SharedPhasePlan::none(4);
        assert_eq!(none.fill_zero_width(), 4);
        assert_eq!(none.activation_dispatch_threads(), 0);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_q4_expert_payload(
        intermediate: usize,
        width: usize,
    ) -> ScheduledQ4ExpertPhaseMlpPayload<'static> {
        let gate = test_q4_matvec_payload(intermediate, width);
        let up = test_q4_matvec_payload(intermediate, width);
        let down = test_q4_matvec_payload(width, intermediate);
        ScheduledQ4ExpertPhaseMlpPayload::new(3, 1, width, gate, up, down).unwrap()
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_q4_matvec_payload(rows: usize, cols: usize) -> Q4MatvecPayload<'static> {
        Q4MatvecPayload {
            rows,
            cols,
            group_size: 16,
            packed: &[],
            scales: &[],
            biases: &[],
            scale_bias_groups: 0,
            scale_bias_dtype: "F32",
            scale_bytes: &[],
            bias_bytes: &[],
            source: None,
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_q4_projection(
        tensor_name: &str,
        output_width: usize,
        input_width: usize,
    ) -> DenseQ4MmapMatvecProjection {
        DenseQ4MmapMatvecProjection {
            tensor_name: tensor_name.to_string(),
            packed_byte_offset: 0,
            scales_byte_offset: 64,
            biases_byte_offset: 96,
            rows: output_width,
            cols: input_width,
            output_width,
            row_packed_bytes: 16,
            groups_per_row: 1,
            group_size: 16,
            scale_bias_dtype: "F32".to_string(),
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_fused_prep_routing_command(
        layer: usize,
        experts: usize,
        routes: &[(usize, f32)],
    ) -> ScheduledRoutingCommand {
        let stage = FlashMoeStageCapability::new(
            FlashMoeGraphStage::RoutingSoftmaxTopK,
            FlashMoeStagePlacement::CpuDeclared,
            FlashMoeStageImplementation::CpuSoftmaxTopK,
        );
        let routing = ScheduledRoutingTopK {
            stage,
            layer,
            experts,
            active_experts: routes.len(),
            source: ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
        };
        ScheduledRoutingCommand {
            routing,
            layer,
            active_experts: routes.len(),
            source: ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
            routes: routes.to_vec(),
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_kv_cache_declares_layer_offsets_and_rejects_missing_layers() {
        let keys = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let values = keys;
        let cache = MetalKvCacheInner::new(keys, values, &[4, 0, 8], 3).unwrap();

        assert_eq!(cache.keys, keys);
        assert_eq!(cache.values, values);
        assert_eq!(cache.max_context, 3);

        let first = cache.layer(0).unwrap();
        assert_eq!(first.offset, 0);
        assert_eq!(first.width, 4);

        let third = cache.layer(2).unwrap();
        assert_eq!(third.offset, 12);
        assert_eq!(third.width, 8);

        assert!(
            cache
                .layer(1)
                .unwrap_err()
                .to_string()
                .contains("not a full-attention layer")
        );
        assert!(
            cache
                .layer(3)
                .unwrap_err()
                .to_string()
                .contains("has no layer 3")
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_linear_attention_state_cache_preserves_gpu_buffer_roles() {
        let base = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let state = MetalLinearAttentionLayerState::new(
            base, base, base, base, base, base, 12, 20, 4, 8, 2,
        );
        let cache = MetalLinearAttentionStateCache::new(vec![None, Some(state)]);
        let layer = cache.layers[1].as_ref().unwrap();

        assert_eq!(layer.conv_state, base);
        assert_eq!(layer.ssm_state, base);
        assert_eq!(layer.conv_output, base);
        assert_eq!(layer.delta_output, base);
        assert_eq!(layer.g_decay, base);
        assert_eq!(layer.beta_gate, base);
        assert_eq!(layer.conv_state_len, 12);
        assert_eq!(layer.ssm_state_len, 20);
        assert_eq!(layer.conv_dim, 4);
        assert_eq!(layer.total_value_width, 8);
        assert_eq!(layer.num_value_heads, 2);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_dispatch_plans_preserve_command_geometry() {
        assert_eq!(
            MetalDispatchPlan::threads(96),
            MetalDispatchPlan {
                mode: MetalDispatchMode::Threads,
                grid: MetalDispatchSize::new(96, 1, 1),
                threadgroup: MetalDispatchSize::new(64, 1, 1),
            }
        );
        assert_eq!(
            MetalDispatchPlan::q4_threadgroups(17),
            MetalDispatchPlan {
                mode: MetalDispatchMode::Threadgroups,
                grid: MetalDispatchSize::new(3, 1, 1),
                threadgroup: MetalDispatchSize::new(256, 1, 1),
            }
        );
        assert_eq!(
            MetalDispatchPlan::q4_mmap_threadgroups(17),
            MetalDispatchPlan {
                mode: MetalDispatchMode::Threadgroups,
                grid: MetalDispatchSize::new(2, 1, 1),
                threadgroup: MetalDispatchSize::new(256, 1, 1),
            }
        );
        assert_eq!(
            MetalDispatchPlan::single_threadgroup(512),
            MetalDispatchPlan {
                mode: MetalDispatchMode::Threadgroups,
                grid: MetalDispatchSize::new(1, 1, 1),
                threadgroup: MetalDispatchSize::new(256, 1, 1),
            }
        );
        assert_eq!(MetalDispatchPlan::q4_threadgroups(0).grid.width, 1);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_shared_expert_buffers_preserve_projection_bindings_and_shape() {
        let gate = 0x1000usize as MetalObjcId;
        let up = 0x2000usize as MetalObjcId;
        let down = 0x3000usize as MetalObjcId;
        let router = 0x4000usize as MetalObjcId;
        let buffers = MetalSharedExpertBuffers::new(gate, up, down, router, 8, 2, 4, 8);

        assert_eq!(buffers.gate, gate);
        assert_eq!(buffers.up, up);
        assert_eq!(buffers.down, down);
        assert_eq!(buffers.router, router);
        assert_eq!(buffers.width, 8);
        assert_eq!(buffers.shared_experts, 2);
        assert_eq!(buffers.intermediate, 4);
        assert_eq!(buffers.total_intermediate, 8);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_dense_weights_hold_buffer_len_and_mmap_owner() {
        let mmap = Arc::new(
            memmap2::MmapMut::map_anon(16)
                .unwrap()
                .make_read_only()
                .unwrap(),
        );
        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let dense = MetalDenseWeights::new(id, Arc::clone(&mmap), 16);

        assert_eq!(dense.buffer, id);
        assert_eq!(dense.len, 16);
        assert_eq!(Arc::strong_count(&mmap), 2);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_lm_head_buffer_cache_owns_budget_and_eviction_policy() {
        let first_id = 0x1000usize as MetalObjcId;
        let second_id = 0x2000usize as MetalObjcId;
        let oversized_id = 0x3000usize as MetalObjcId;
        let mut cache = MetalLmHeadBufferCache::with_budget(10);
        let mut released = Vec::new();

        assert_eq!(cache.max_bytes(), 10);
        assert_eq!(
            cache.insert(
                "first".to_string(),
                MetalLmHeadBuffer::new(first_id, 2, 3, 6),
                |buffer| released.push(buffer.weights),
            ),
            None
        );
        assert_eq!(cache.get("first", 2, 3).unwrap().weights, first_id);
        assert!(cache.get("first", 3, 2).is_none());

        assert_eq!(
            cache.insert(
                "second".to_string(),
                MetalLmHeadBuffer::new(second_id, 2, 3, 6),
                |buffer| released.push(buffer.weights),
            ),
            None
        );
        assert_eq!(released, vec![first_id]);
        assert!(cache.get("first", 2, 3).is_none());
        assert_eq!(cache.get("second", 2, 3).unwrap().weights, second_id);

        let oversized = MetalLmHeadBuffer::new(oversized_id, 4, 4, 16);
        assert_eq!(
            cache.insert("oversized".to_string(), oversized, |buffer| {
                released.push(buffer.weights)
            }),
            Some(oversized)
        );

        cache.release_all(|buffer| released.push(buffer.weights));
        assert_eq!(released, vec![first_id, second_id]);
        assert!(cache.get("second", 2, 3).is_none());
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_reusable_buffer_records_pool_entry_shape() {
        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let buffer = MetalReusableBuffer::new(id, 4096);

        assert_eq!(buffer.id, id);
        assert_eq!(buffer.len, 4096);
        assert_eq!(METAL_REUSABLE_BUFFER_POOL_LIMIT, 64);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_q4_source_buffer_cache_reuses_same_fixed_payload_key() {
        let first = [1u8; 16];
        let second = [2u8; 16];
        let buffer = 0x1000usize as MetalObjcId;
        let mut cache = MetalQ4SourceBufferCache::default();

        cache.insert(&first, buffer);

        assert_eq!(cache.get(&first), Some(buffer));
        assert_eq!(cache.get(&second), None);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn phase_buffer_declares_recyclable_or_borrowed_lifecycle() {
        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();

        let recyclable = MetalPhaseBuffer::recyclable(id);
        assert_eq!(recyclable.id, id);
        assert!(recyclable.recycle);

        let borrowed = MetalPhaseBuffer::borrowed(id);
        assert_eq!(borrowed.id, id);
        assert!(!borrowed.recycle);
    }
}
