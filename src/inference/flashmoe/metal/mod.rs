use std::collections::BTreeSet;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::collections::{BTreeMap, HashMap};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::ffi::{CStr, CString, c_char, c_void};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::ptr;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::{Arc, Mutex};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::thread;
use std::time::{Duration, Instant};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod deepseek_execution;
mod diagnostics;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod ffi;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod resources;

pub(crate) use diagnostics::*;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) use ffi::*;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use resources::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) use deepseek_execution::DeepSeekV4SessionSnapshot;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[derive(Debug)]
pub(super) struct DeepSeekV4SessionSnapshot;

#[derive(Debug, Clone)]
pub(super) struct MetalExecutionFacade {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) inner: Arc<MetalExecutionContext>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::deepseek_metal::DEEPSEEK_V4_METAL_SHADERS;
use super::deepseek_metal::DEEPSEEK_V4_REQUIRED_METAL_KERNELS;
use super::state::{
    FlashMoeExpertPhaseOutput, FlashMoeGpuBufferDescriptor, FlashMoeGpuMatrixDescriptor,
    FlashMoeStateBufferRole,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::types::FlashMoeMetalResourceSnapshot;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::experts::{
    EXPERT_SCALE_BIAS_DTYPE_BF16, EXPERT_SCALE_BIAS_DTYPE_F32, ReusableExpertBytes,
    expert_scale_bias_dtype_size,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::math::{routing_sigmoid_noaux_top_k, routing_softmax_top_k};
#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
use super::scheduler::ScheduledQ4ExpertPhaseMlpPayload;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::scheduler::{
    ScheduledCmd3ExpertPayload, ScheduledCmd3MetalPostAttentionInput, ScheduledCmd3OutputState,
    ScheduledDenseExpertPhaseMlpPayload, ScheduledExpertPhaseMlpPayload, ScheduledExpertSlot,
    ScheduledLayerMajorExperts, ScheduledRoutingCandidateSource, ScheduledRoutingCommand,
    ScheduledSharedExpertPhaseRef,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::state::{
    FlashMoeCmd3OutputState, FlashMoeLinearAttentionCacheState,
    FlashMoeLinearAttentionLayerSnapshot, FlashMoeLinearAttentionSessionSnapshot,
    FlashMoePostAttentionPrepState, LinearAttentionLayout,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::weights::{
    Cmd2ResidentPostAttentionPrepProjections, DenseMmapMatvecProjection,
    DenseQ4MmapMatvecProjection, FullAttentionLayout, FullAttentionQLayout,
    LinearAttentionResidentBindings, ResidentMmapMatvecProjection, ResidentStaticDtype,
    RotaryPairing, SharedExpertPhaseResidentProjections, SharedExpertPhaseWeights,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use anyhow::{Context as _, Result, bail};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) type MetalObjcId = *mut c_void;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalStateBuffer {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    buffer: MetalObjcId,
    state: FlashMoeGpuBufferDescriptor,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalMatrixBuffer {
    buffer: MetalObjcId,
    state: FlashMoeGpuMatrixDescriptor,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalMatrixBuffer {
    fn new(buffer: MetalObjcId, state: FlashMoeGpuMatrixDescriptor) -> anyhow::Result<Self> {
        if buffer.is_null() {
            anyhow::bail!("FlashMoe Metal matrix buffer requires a non-null buffer");
        }
        if !state.is_declared_graph_state() {
            anyhow::bail!("FlashMoe Metal matrix buffer requires declared GPU matrix state");
        }
        Ok(Self { buffer, state })
    }

    pub(crate) fn buffer(self) -> MetalObjcId {
        self.buffer
    }

    pub(crate) fn state(self) -> FlashMoeGpuMatrixDescriptor {
        self.state
    }

    pub(crate) fn rows(self) -> usize {
        self.state.rows()
    }

    pub(crate) fn cols(self) -> usize {
        self.state.cols()
    }

    pub(crate) fn values(self) -> usize {
        self.state.values()
    }
}

impl MetalStateBuffer {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn new(buffer: MetalObjcId, state: FlashMoeGpuBufferDescriptor) -> anyhow::Result<Self> {
        if buffer.is_null() {
            anyhow::bail!("FlashMoe Metal state buffer requires a non-null buffer");
        }
        if !state.is_declared_graph_state()
            || state.placement() != super::state::FlashMoeStatePlacement::GpuResident
        {
            anyhow::bail!("FlashMoe Metal state buffer requires declared GpuResident state");
        }
        Ok(Self { buffer, state })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn buffer(self) -> MetalObjcId {
        self.buffer
    }

    pub(crate) fn len(self) -> usize {
        self.state.len()
    }

    pub(crate) fn state(self) -> FlashMoeGpuBufferDescriptor {
        self.state
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) type MetalSelector = *mut c_void;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MetalMatvecTiming {
    pub(crate) buffer_upload: Duration,
    pub(crate) dispatch: Duration,
    pub(crate) readback: Duration,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) enum MetalBatchProjectionInput<'a> {
    Cpu(&'a [f32]),
    Buffer { buffer: MetalObjcId, len: usize },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalGlmMlaAbsorbedAttentionInput<'a> {
    pub(crate) heads: usize,
    pub(crate) latent_rank: usize,
    pub(crate) query_nope: &'a [f32],
    pub(crate) query_rope: &'a [f32],
    pub(crate) record_latents: &'a [f32],
    pub(crate) record_rotary: &'a [f32],
    pub(crate) sequence: usize,
    pub(crate) rope_dim: usize,
    pub(crate) scale: f32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalGlmMlaFusedAttentionInput<'a> {
    pub(crate) input: MetalBatchProjectionInput<'a>,
    pub(crate) heads: usize,
    pub(crate) latent_rank: usize,
    pub(crate) nope_dim: usize,
    pub(crate) rope_dim: usize,
    pub(crate) previous_record_latents: &'a [f32],
    pub(crate) previous_record_rotary: &'a [f32],
    pub(crate) rope_cos: &'a [f32],
    pub(crate) rope_sin: &'a [f32],
    pub(crate) scale: f32,
    pub(crate) post_attention: Option<MetalGlmMlaPostAttentionInput<'a>>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalGlmMlaPostAttentionInput<'a> {
    pub(crate) projections: &'a Cmd2ResidentPostAttentionPrepProjections,
    pub(crate) residual: MetalBatchProjectionInput<'a>,
    pub(crate) post_norm_weight: &'a [f32],
    pub(crate) router_correction_bias: Option<&'a [f32]>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) enum MetalGlmMlaFusedAttentionTerminal {
    Attention(Vec<f32>),
    PostAttention(MetalPostAttentionPrep),
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalGlmMlaFusedAttentionOutput {
    pub(crate) terminal: MetalGlmMlaFusedAttentionTerminal,
    pub(crate) latent: Vec<f32>,
    pub(crate) rotary: Vec<f32>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalExpertSourceBufferKey {
    ptr: usize,
    len: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
struct MetalExpertSourceBufferEntry {
    key: MetalExpertSourceBufferKey,
    buffer: MetalObjcId,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
struct MetalPersistentExpertBuffer {
    device: usize,
    buffer: usize,
    resources: Arc<MetalResourceLedger>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalPersistentExpertBuffer {
    fn new(
        device: MetalObjcId,
        buffer: MetalObjcId,
        len: usize,
        resources: Arc<MetalResourceLedger>,
    ) -> Self {
        resources.register_buffer(buffer, len, MetalTrackedBufferClass::ResidentExpertWrapper);
        Self {
            device: device as usize,
            buffer: buffer as usize,
            resources,
        }
    }

    fn buffer_for_device(&self, device: MetalObjcId) -> Option<MetalObjcId> {
        (self.device == device as usize).then_some(self.buffer as MetalObjcId)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalPersistentExpertBuffer {
    fn drop(&mut self) {
        unsafe {
            self.resources.release_buffer(self.buffer as MetalObjcId);
            release(self.buffer as MetalObjcId);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Default)]
pub(crate) struct MetalExpertSourceBufferCache {
    entries: Vec<MetalExpertSourceBufferEntry>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalExpertSourceBufferCache {
    fn key_for(bytes: &[u8]) -> MetalExpertSourceBufferKey {
        MetalExpertSourceBufferKey {
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
        self.entries
            .push(MetalExpertSourceBufferEntry { key, buffer });
    }
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

    pub(crate) fn q4_mmap_matrix_threadgroups(rows: u64, input_rows: u64) -> Self {
        const Q4_MMAP_ROWS_PER_THREADGROUP: u64 = 16;
        Self {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(
                rows.div_ceil(Q4_MMAP_ROWS_PER_THREADGROUP).max(1),
                input_rows.max(1),
                1,
            ),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    }

    pub(crate) fn q4_mmap_matrix_bf16_threadgroups(
        rows: u64,
        input_rows: u64,
        input_rows_per_threadgroup: u64,
    ) -> Self {
        const Q4_MMAP_ROWS_PER_THREADGROUP: u64 = 16;
        let input_rows_per_threadgroup = input_rows_per_threadgroup.clamp(1, 2);
        Self {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(
                rows.div_ceil(Q4_MMAP_ROWS_PER_THREADGROUP).max(1),
                input_rows.div_ceil(input_rows_per_threadgroup).max(1),
                1,
            ),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    }

    pub(crate) fn qwen_attention_threadgroups(query_rows: u64, query_heads: u64) -> Self {
        Self {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(query_rows.max(1), query_heads.max(1), 1),
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
const DEFAULT_FLASHMOE_METAL_COMMAND_POLL_INTERVAL: Duration = Duration::from_micros(100);

pub(crate) mod kernels {
    pub(crate) const DEEPSEEK_IQ2_XXS_PAIR_SWIGLU: &str =
        "kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32";
    pub(crate) const DEEPSEEK_Q2_K_SUM6: &str = "kernel_mul_mv_slots6_q2_K_sum6_f32";
    pub(crate) const Q4_FMA_MATVEC: &str = "q4_fma_matvec";
    pub(crate) const Q4_FMA_MATVEC_BF16_SCALE_BIAS: &str = "q4_fma_matvec_bf16_scale_bias";
    pub(crate) const MXFP4_FMA_MATVEC_E8M0: &str = "mxfp4_fma_matvec_e8m0";
    pub(crate) const Q4_SWIGLU_FUSED: &str = "q4_swiglu_fused";
    pub(crate) const Q4_SWIGLU_FUSED_BF16_SCALE_BIAS: &str = "q4_swiglu_fused_bf16_scale_bias";
    pub(crate) const Q4_MMAP_FMA_MATVEC: &str = "q4_mmap_fma_matvec";
    pub(crate) const Q4_MMAP_FMA_MATVEC_BF16_SCALE_BIAS: &str =
        "q4_mmap_fma_matvec_bf16_scale_bias";
    pub(crate) const Q4_MMAP_FMA_MATVEC_BATCH: &str = "q4_mmap_fma_matvec_batch";
    pub(crate) const Q4_MMAP_FMA_MATVEC_BATCH_BF16_SCALE_BIAS: &str =
        "q4_mmap_fma_matvec_batch_bf16_scale_bias";
    pub(crate) const Q4_MMAP_FMA_MULTILINEAR_BF16_SCALE_BIAS: &str =
        "q4_mmap_fma_multilinear_bf16_scale_bias";
    pub(crate) const GLM_MLA_PREPARE_QUERY_KV: &str = "glm_mla_prepare_query_kv";
    pub(crate) const GLM_MLA_ABSORBED_SCORES: &str = "glm_mla_absorbed_scores";
    pub(crate) const GLM_MLA_SOFTMAX: &str = "glm_mla_softmax";
    pub(crate) const GLM_MLA_CONTEXT: &str = "glm_mla_context";
    pub(crate) const DENSE_MMAP_FMA_MATVEC_BF16: &str = "dense_mmap_fma_matvec_bf16";
    pub(crate) const DENSE_MMAP_FMA_MATVEC_F16: &str = "dense_mmap_fma_matvec_f16";
    pub(crate) const DENSE_MMAP_FMA_MATVEC_F32: &str = "dense_mmap_fma_matvec_f32";
    pub(crate) const DENSE_MMAP_FMA_MATRIX_BF16: &str = "dense_mmap_fma_matrix_bf16";
    pub(crate) const DENSE_MMAP_FMA_MATRIX_F16: &str = "dense_mmap_fma_matrix_f16";
    pub(crate) const DENSE_MMAP_FMA_MATRIX_F32: &str = "dense_mmap_fma_matrix_f32";
    pub(crate) const RMS_NORM_REDUCED: &str = "rms_norm_reduced";
    pub(crate) const RESIDUAL_ADD_RMS_NORM: &str = "residual_add_rms_norm";
    pub(crate) const ATTENTION_SCORES: &str = "attention_scores";
    pub(crate) const QWEN_PREPARE_QKV_ROWS: &str = "qwen_prepare_qkv_rows";
    pub(crate) const QWEN_CAUSAL_ATTENTION_ROWS: &str = "qwen_causal_attention_rows";
    pub(crate) const QWEN_APPLY_ATTENTION_GATE: &str = "qwen_apply_attention_gate";
    pub(crate) const QWEN_FINAL_RMS_NORM_ROW: &str = "qwen_final_rms_norm_row";
    pub(crate) const EXPERT_MLP_FUSED: &str = "expert_mlp_fused";
    pub(crate) const SILU_PRODUCT: &str = "silu_product";
    pub(crate) const SHARED_EXPERT_ACTIVATION: &str = "shared_expert_activation";
    pub(crate) const COMBINE_EXPERT_PHASE: &str = "combine_expert_phase";
    pub(crate) const QWEN_LAYER_MAJOR_GATHER: &str = "qwen_layer_major_gather";
    pub(crate) const QWEN_LAYER_MAJOR_COMBINE: &str = "qwen_layer_major_combine";
    pub(crate) const FILL_ZERO: &str = "fill_zero";
    pub(crate) const TOPK_VOCAB: &str = "topk_vocab";
    pub(crate) const LINEAR_CONV1D_STEP_BF16: &str = "linear_conv1d_step_bf16";
    pub(crate) const LINEAR_CONV1D_STEP_F16: &str = "linear_conv1d_step_f16";
    pub(crate) const LINEAR_CONV1D_STEP_F32: &str = "linear_conv1d_step_f32";
    pub(crate) const LINEAR_RMS_NORM_QK: &str = "linear_rms_norm_qk";
    pub(crate) const LINEAR_COMPUTE_DECAY_BETA_BF16: &str = "linear_compute_decay_beta_bf16";
    pub(crate) const LINEAR_COMPUTE_DECAY_BETA_F16: &str = "linear_compute_decay_beta_f16";
    pub(crate) const LINEAR_COMPUTE_DECAY_BETA_F32: &str = "linear_compute_decay_beta_f32";
    pub(crate) const LINEAR_GATED_DELTA_STEP: &str = "linear_gated_delta_step";
    pub(crate) const LINEAR_GATED_RMS_NORM_BF16: &str = "linear_gated_rms_norm_bf16";
    pub(crate) const LINEAR_GATED_RMS_NORM_F16: &str = "linear_gated_rms_norm_f16";
    pub(crate) const LINEAR_GATED_RMS_NORM_F32: &str = "linear_gated_rms_norm_f32";
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

    pub(crate) fn with_additional(mut self, kernels: &'static [&'static str]) -> Self {
        self.kernels.extend(kernels.iter().copied());
        self
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
    kernels::MXFP4_FMA_MATVEC_E8M0,
    kernels::Q4_SWIGLU_FUSED,
    kernels::Q4_SWIGLU_FUSED_BF16_SCALE_BIAS,
    kernels::Q4_MMAP_FMA_MATVEC,
    kernels::Q4_MMAP_FMA_MATVEC_BF16_SCALE_BIAS,
    kernels::Q4_MMAP_FMA_MATVEC_BATCH,
    kernels::Q4_MMAP_FMA_MATVEC_BATCH_BF16_SCALE_BIAS,
    kernels::Q4_MMAP_FMA_MULTILINEAR_BF16_SCALE_BIAS,
    kernels::GLM_MLA_PREPARE_QUERY_KV,
    kernels::GLM_MLA_ABSORBED_SCORES,
    kernels::GLM_MLA_SOFTMAX,
    kernels::GLM_MLA_CONTEXT,
    kernels::DENSE_MMAP_FMA_MATVEC_BF16,
    kernels::DENSE_MMAP_FMA_MATVEC_F16,
    kernels::DENSE_MMAP_FMA_MATVEC_F32,
    kernels::DENSE_MMAP_FMA_MATRIX_BF16,
    kernels::DENSE_MMAP_FMA_MATRIX_F16,
    kernels::DENSE_MMAP_FMA_MATRIX_F32,
    kernels::RMS_NORM_REDUCED,
    kernels::RESIDUAL_ADD_RMS_NORM,
    kernels::ATTENTION_SCORES,
    kernels::QWEN_PREPARE_QKV_ROWS,
    kernels::QWEN_CAUSAL_ATTENTION_ROWS,
    kernels::QWEN_APPLY_ATTENTION_GATE,
    kernels::QWEN_FINAL_RMS_NORM_ROW,
    kernels::EXPERT_MLP_FUSED,
    kernels::SILU_PRODUCT,
    kernels::SHARED_EXPERT_ACTIVATION,
    kernels::COMBINE_EXPERT_PHASE,
    kernels::QWEN_LAYER_MAJOR_GATHER,
    kernels::QWEN_LAYER_MAJOR_COMBINE,
    kernels::FILL_ZERO,
    kernels::TOPK_VOCAB,
    kernels::LINEAR_CONV1D_STEP_BF16,
    kernels::LINEAR_CONV1D_STEP_F16,
    kernels::LINEAR_CONV1D_STEP_F32,
    kernels::LINEAR_RMS_NORM_QK,
    kernels::LINEAR_COMPUTE_DECAY_BETA_BF16,
    kernels::LINEAR_COMPUTE_DECAY_BETA_F16,
    kernels::LINEAR_COMPUTE_DECAY_BETA_F32,
    kernels::LINEAR_GATED_DELTA_STEP,
    kernels::LINEAR_GATED_RMS_NORM_BF16,
    kernels::LINEAR_GATED_RMS_NORM_F16,
    kernels::LINEAR_GATED_RMS_NORM_F32,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalPipelineNameSet {
    pub(crate) q4: &'static str,
    pub(crate) q4_bf16_scale_bias: &'static str,
    pub(crate) mxfp4_e8m0: &'static str,
    pub(crate) q4_swiglu: &'static str,
    pub(crate) q4_swiglu_bf16_scale_bias: &'static str,
    pub(crate) q4_mmap: &'static str,
    pub(crate) q4_mmap_bf16_scale_bias: &'static str,
    pub(crate) q4_mmap_batch: &'static str,
    pub(crate) q4_mmap_batch_bf16_scale_bias: &'static str,
    pub(crate) q4_mmap_multilinear_bf16_scale_bias: &'static str,
    pub(crate) glm_mla_prepare_query_kv: &'static str,
    pub(crate) glm_mla_absorbed_scores: &'static str,
    pub(crate) glm_mla_softmax: &'static str,
    pub(crate) glm_mla_context: &'static str,
    pub(crate) dense_mmap_bf16: &'static str,
    pub(crate) dense_mmap_f16: &'static str,
    pub(crate) dense_mmap_f32: &'static str,
    pub(crate) dense_matrix_bf16: &'static str,
    pub(crate) dense_matrix_f16: &'static str,
    pub(crate) dense_matrix_f32: &'static str,
    pub(crate) rms_norm_reduced: &'static str,
    pub(crate) residual_rms_norm: &'static str,
    pub(crate) attention: &'static str,
    pub(crate) qwen_prepare_qkv_rows: &'static str,
    pub(crate) qwen_causal_attention_rows: &'static str,
    pub(crate) qwen_apply_attention_gate: &'static str,
    pub(crate) qwen_final_rms_norm_row: &'static str,
    pub(crate) expert_mlp: &'static str,
    pub(crate) silu_product: &'static str,
    pub(crate) shared_expert_activation: &'static str,
    pub(crate) combine_expert_phase: &'static str,
    pub(crate) qwen_layer_major_gather: &'static str,
    pub(crate) qwen_layer_major_combine: &'static str,
    pub(crate) fill_zero: &'static str,
    pub(crate) topk_vocab: &'static str,
    pub(crate) linear_conv1d_bf16: &'static str,
    pub(crate) linear_conv1d_f16: &'static str,
    pub(crate) linear_conv1d_f32: &'static str,
    pub(crate) linear_rms_norm_qk: &'static str,
    pub(crate) linear_decay_beta_bf16: &'static str,
    pub(crate) linear_decay_beta_f16: &'static str,
    pub(crate) linear_decay_beta_f32: &'static str,
    pub(crate) linear_delta_step: &'static str,
    pub(crate) linear_gated_rms_norm_bf16: &'static str,
    pub(crate) linear_gated_rms_norm_f16: &'static str,
    pub(crate) linear_gated_rms_norm_f32: &'static str,
}

impl MetalPipelineNameSet {
    pub(crate) fn new() -> Self {
        Self {
            q4: kernels::Q4_FMA_MATVEC,
            q4_bf16_scale_bias: kernels::Q4_FMA_MATVEC_BF16_SCALE_BIAS,
            mxfp4_e8m0: kernels::MXFP4_FMA_MATVEC_E8M0,
            q4_swiglu: kernels::Q4_SWIGLU_FUSED,
            q4_swiglu_bf16_scale_bias: kernels::Q4_SWIGLU_FUSED_BF16_SCALE_BIAS,
            q4_mmap: kernels::Q4_MMAP_FMA_MATVEC,
            q4_mmap_bf16_scale_bias: kernels::Q4_MMAP_FMA_MATVEC_BF16_SCALE_BIAS,
            q4_mmap_batch: kernels::Q4_MMAP_FMA_MATVEC_BATCH,
            q4_mmap_batch_bf16_scale_bias: kernels::Q4_MMAP_FMA_MATVEC_BATCH_BF16_SCALE_BIAS,
            q4_mmap_multilinear_bf16_scale_bias: kernels::Q4_MMAP_FMA_MULTILINEAR_BF16_SCALE_BIAS,
            glm_mla_prepare_query_kv: kernels::GLM_MLA_PREPARE_QUERY_KV,
            glm_mla_absorbed_scores: kernels::GLM_MLA_ABSORBED_SCORES,
            glm_mla_softmax: kernels::GLM_MLA_SOFTMAX,
            glm_mla_context: kernels::GLM_MLA_CONTEXT,
            dense_mmap_bf16: kernels::DENSE_MMAP_FMA_MATVEC_BF16,
            dense_mmap_f16: kernels::DENSE_MMAP_FMA_MATVEC_F16,
            dense_mmap_f32: kernels::DENSE_MMAP_FMA_MATVEC_F32,
            dense_matrix_bf16: kernels::DENSE_MMAP_FMA_MATRIX_BF16,
            dense_matrix_f16: kernels::DENSE_MMAP_FMA_MATRIX_F16,
            dense_matrix_f32: kernels::DENSE_MMAP_FMA_MATRIX_F32,
            rms_norm_reduced: kernels::RMS_NORM_REDUCED,
            residual_rms_norm: kernels::RESIDUAL_ADD_RMS_NORM,
            attention: kernels::ATTENTION_SCORES,
            qwen_prepare_qkv_rows: kernels::QWEN_PREPARE_QKV_ROWS,
            qwen_causal_attention_rows: kernels::QWEN_CAUSAL_ATTENTION_ROWS,
            qwen_apply_attention_gate: kernels::QWEN_APPLY_ATTENTION_GATE,
            qwen_final_rms_norm_row: kernels::QWEN_FINAL_RMS_NORM_ROW,
            expert_mlp: kernels::EXPERT_MLP_FUSED,
            silu_product: kernels::SILU_PRODUCT,
            shared_expert_activation: kernels::SHARED_EXPERT_ACTIVATION,
            combine_expert_phase: kernels::COMBINE_EXPERT_PHASE,
            qwen_layer_major_gather: kernels::QWEN_LAYER_MAJOR_GATHER,
            qwen_layer_major_combine: kernels::QWEN_LAYER_MAJOR_COMBINE,
            fill_zero: kernels::FILL_ZERO,
            topk_vocab: kernels::TOPK_VOCAB,
            linear_conv1d_bf16: kernels::LINEAR_CONV1D_STEP_BF16,
            linear_conv1d_f16: kernels::LINEAR_CONV1D_STEP_F16,
            linear_conv1d_f32: kernels::LINEAR_CONV1D_STEP_F32,
            linear_rms_norm_qk: kernels::LINEAR_RMS_NORM_QK,
            linear_decay_beta_bf16: kernels::LINEAR_COMPUTE_DECAY_BETA_BF16,
            linear_decay_beta_f16: kernels::LINEAR_COMPUTE_DECAY_BETA_F16,
            linear_decay_beta_f32: kernels::LINEAR_COMPUTE_DECAY_BETA_F32,
            linear_delta_step: kernels::LINEAR_GATED_DELTA_STEP,
            linear_gated_rms_norm_bf16: kernels::LINEAR_GATED_RMS_NORM_BF16,
            linear_gated_rms_norm_f16: kernels::LINEAR_GATED_RMS_NORM_F16,
            linear_gated_rms_norm_f32: kernels::LINEAR_GATED_RMS_NORM_F32,
        }
    }

    pub(crate) fn kernel_names(self) -> Vec<&'static str> {
        vec![
            self.q4,
            self.q4_bf16_scale_bias,
            self.mxfp4_e8m0,
            self.q4_swiglu,
            self.q4_swiglu_bf16_scale_bias,
            self.q4_mmap,
            self.q4_mmap_bf16_scale_bias,
            self.q4_mmap_batch,
            self.q4_mmap_batch_bf16_scale_bias,
            self.q4_mmap_multilinear_bf16_scale_bias,
            self.glm_mla_prepare_query_kv,
            self.glm_mla_absorbed_scores,
            self.glm_mla_softmax,
            self.glm_mla_context,
            self.dense_mmap_bf16,
            self.dense_mmap_f16,
            self.dense_mmap_f32,
            self.dense_matrix_bf16,
            self.dense_matrix_f16,
            self.dense_matrix_f32,
            self.rms_norm_reduced,
            self.residual_rms_norm,
            self.attention,
            self.qwen_prepare_qkv_rows,
            self.qwen_causal_attention_rows,
            self.qwen_apply_attention_gate,
            self.qwen_final_rms_norm_row,
            self.expert_mlp,
            self.silu_product,
            self.shared_expert_activation,
            self.combine_expert_phase,
            self.qwen_layer_major_gather,
            self.qwen_layer_major_combine,
            self.fill_zero,
            self.topk_vocab,
            self.linear_conv1d_bf16,
            self.linear_conv1d_f16,
            self.linear_conv1d_f32,
            self.linear_rms_norm_qk,
            self.linear_decay_beta_bf16,
            self.linear_decay_beta_f16,
            self.linear_decay_beta_f32,
            self.linear_delta_step,
            self.linear_gated_rms_norm_bf16,
            self.linear_gated_rms_norm_f16,
            self.linear_gated_rms_norm_f32,
        ]
    }
}

#[derive(Debug)]
pub(crate) struct MetalPipelineSet<T> {
    pub(crate) q4_pipeline: T,
    pub(crate) q4_bf16_scale_bias_pipeline: T,
    pub(crate) mxfp4_e8m0_pipeline: T,
    pub(crate) q4_swiglu_pipeline: T,
    pub(crate) q4_swiglu_bf16_scale_bias_pipeline: T,
    pub(crate) q4_mmap_pipeline: T,
    pub(crate) q4_mmap_bf16_scale_bias_pipeline: T,
    pub(crate) q4_mmap_batch_pipeline: T,
    pub(crate) q4_mmap_batch_bf16_scale_bias_pipeline: T,
    pub(crate) q4_mmap_multilinear_bf16_scale_bias_pipeline: T,
    pub(crate) glm_mla_prepare_query_kv_pipeline: T,
    pub(crate) glm_mla_absorbed_scores_pipeline: T,
    pub(crate) glm_mla_softmax_pipeline: T,
    pub(crate) glm_mla_context_pipeline: T,
    pub(crate) dense_mmap_bf16_pipeline: T,
    pub(crate) dense_mmap_f16_pipeline: T,
    pub(crate) dense_mmap_f32_pipeline: T,
    pub(crate) dense_matrix_bf16_pipeline: T,
    pub(crate) dense_matrix_f16_pipeline: T,
    pub(crate) dense_matrix_f32_pipeline: T,
    pub(crate) rms_norm_reduced_pipeline: T,
    pub(crate) residual_rms_norm_pipeline: T,
    pub(crate) attention_pipeline: T,
    pub(crate) qwen_prepare_qkv_rows_pipeline: T,
    pub(crate) qwen_causal_attention_rows_pipeline: T,
    pub(crate) qwen_apply_attention_gate_pipeline: T,
    pub(crate) qwen_final_rms_norm_row_pipeline: T,
    pub(crate) expert_mlp_pipeline: T,
    pub(crate) silu_product_pipeline: T,
    pub(crate) shared_expert_activation_pipeline: T,
    pub(crate) combine_expert_phase_pipeline: T,
    pub(crate) qwen_layer_major_gather_pipeline: T,
    pub(crate) qwen_layer_major_combine_pipeline: T,
    pub(crate) fill_zero_pipeline: T,
    pub(crate) topk_vocab_pipeline: T,
    pub(crate) linear_conv1d_bf16_pipeline: T,
    pub(crate) linear_conv1d_f16_pipeline: T,
    pub(crate) linear_conv1d_f32_pipeline: T,
    pub(crate) linear_rms_norm_qk_pipeline: T,
    pub(crate) linear_decay_beta_bf16_pipeline: T,
    pub(crate) linear_decay_beta_f16_pipeline: T,
    pub(crate) linear_decay_beta_f32_pipeline: T,
    pub(crate) linear_delta_step_pipeline: T,
    pub(crate) linear_gated_rms_norm_bf16_pipeline: T,
    pub(crate) linear_gated_rms_norm_f16_pipeline: T,
    pub(crate) linear_gated_rms_norm_f32_pipeline: T,
}

impl<T: Copy> MetalPipelineSet<T> {
    pub(crate) fn release_with(&self, mut release: impl FnMut(T)) {
        release(self.q4_pipeline);
        release(self.q4_bf16_scale_bias_pipeline);
        release(self.mxfp4_e8m0_pipeline);
        release(self.q4_swiglu_pipeline);
        release(self.q4_swiglu_bf16_scale_bias_pipeline);
        release(self.q4_mmap_pipeline);
        release(self.q4_mmap_bf16_scale_bias_pipeline);
        release(self.q4_mmap_batch_pipeline);
        release(self.q4_mmap_batch_bf16_scale_bias_pipeline);
        release(self.q4_mmap_multilinear_bf16_scale_bias_pipeline);
        release(self.glm_mla_prepare_query_kv_pipeline);
        release(self.glm_mla_absorbed_scores_pipeline);
        release(self.glm_mla_softmax_pipeline);
        release(self.glm_mla_context_pipeline);
        release(self.dense_mmap_bf16_pipeline);
        release(self.dense_mmap_f16_pipeline);
        release(self.dense_mmap_f32_pipeline);
        release(self.dense_matrix_bf16_pipeline);
        release(self.dense_matrix_f16_pipeline);
        release(self.dense_matrix_f32_pipeline);
        release(self.rms_norm_reduced_pipeline);
        release(self.residual_rms_norm_pipeline);
        release(self.attention_pipeline);
        release(self.qwen_prepare_qkv_rows_pipeline);
        release(self.qwen_causal_attention_rows_pipeline);
        release(self.qwen_apply_attention_gate_pipeline);
        release(self.qwen_final_rms_norm_row_pipeline);
        release(self.expert_mlp_pipeline);
        release(self.silu_product_pipeline);
        release(self.shared_expert_activation_pipeline);
        release(self.combine_expert_phase_pipeline);
        release(self.qwen_layer_major_gather_pipeline);
        release(self.qwen_layer_major_combine_pipeline);
        release(self.fill_zero_pipeline);
        release(self.topk_vocab_pipeline);
        release(self.linear_conv1d_bf16_pipeline);
        release(self.linear_conv1d_f16_pipeline);
        release(self.linear_conv1d_f32_pipeline);
        release(self.linear_rms_norm_qk_pipeline);
        release(self.linear_decay_beta_bf16_pipeline);
        release(self.linear_decay_beta_f16_pipeline);
        release(self.linear_decay_beta_f32_pipeline);
        release(self.linear_delta_step_pipeline);
        release(self.linear_gated_rms_norm_bf16_pipeline);
        release(self.linear_gated_rms_norm_f16_pipeline);
        release(self.linear_gated_rms_norm_f32_pipeline);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
struct OwnedMetalObject(MetalObjcId);

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl OwnedMetalObject {
    fn new(id: MetalObjcId) -> anyhow::Result<Self> {
        if id.is_null() {
            anyhow::bail!("failed to create required Flash-MoE Metal object");
        }
        Ok(Self(id))
    }

    fn id(&self) -> MetalObjcId {
        self.0
    }

    fn into_raw(mut self) -> MetalObjcId {
        let id = self.0;
        self.0 = ptr::null_mut();
        id
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for OwnedMetalObject {
    fn drop(&mut self) {
        unsafe { release(self.0) }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
struct MetalCommandLease {
    resources: Arc<MetalResourceLedger>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCommandLease {
    fn new(resources: Arc<MetalResourceLedger>) -> Self {
        resources.command_started();
        Self { resources }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalCommandLease {
    fn drop(&mut self) {
        self.resources.command_finished();
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
struct MetalCommandEncoding {
    command_buffer: MetalObjcId,
    encoder: MetalObjcId,
    ended: bool,
    command_lease: Option<MetalCommandLease>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCommandEncoding {
    unsafe fn new(
        command_queue: MetalObjcId,
        resources: Arc<MetalResourceLedger>,
        command_buffer_error: &'static str,
        encoder_error: &'static str,
    ) -> anyhow::Result<Self> {
        unsafe {
            // Every encoded resource is held explicitly until completion (or transferred to a
            // deferred submission), so the command buffer does not need to retain it again.
            let command_buffer = retain_autoreleased_return_value(msg_send_id0(
                command_queue,
                sel("commandBufferWithUnretainedReferences"),
            ));
            if command_buffer.is_null() {
                anyhow::bail!(command_buffer_error);
            }
            let encoder = retain_autoreleased_return_value(msg_send_id0(
                command_buffer,
                sel("computeCommandEncoder"),
            ));
            if encoder.is_null() {
                release(command_buffer);
                anyhow::bail!(encoder_error);
            }
            Ok(Self {
                command_buffer,
                encoder,
                ended: false,
                command_lease: Some(MetalCommandLease::new(resources)),
            })
        }
    }

    fn command_buffer(&self) -> MetalObjcId {
        self.command_buffer
    }

    fn encoder(&self) -> MetalObjcId {
        self.encoder
    }

    unsafe fn end_encoding(&mut self) {
        unsafe {
            if !self.ended {
                msg_send_void0(self.encoder, sel("endEncoding"));
                self.ended = true;
            }
        }
    }

    unsafe fn into_command_buffer(mut self) -> (MetalObjcId, MetalCommandLease) {
        unsafe {
            self.end_encoding();
            release(self.encoder);
            self.encoder = ptr::null_mut();
            let command_buffer = self.command_buffer;
            self.command_buffer = ptr::null_mut();
            let command_lease = self
                .command_lease
                .take()
                .expect("Metal command encoding is missing its resource lease");
            (command_buffer, command_lease)
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalCommandEncoding {
    fn drop(&mut self) {
        unsafe {
            if !self.encoder.is_null() {
                self.end_encoding();
                release(self.encoder);
            }
            release(self.command_buffer);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalRuntime {
    pub(crate) device: MetalObjcId,
    pub(crate) command_queue: MetalObjcId,
    pub(crate) pipelines: MetalPipelineSet<MetalObjcId>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalRuntime {
    pub(crate) fn compile(
        shader_source: &str,
        names: MetalPipelineNameSet,
    ) -> anyhow::Result<Self> {
        unsafe {
            let device = OwnedMetalObject::new(metal_default_device()).map_err(|_| {
                anyhow::anyhow!(
                    "Metal is required for Flash-MoE on ARM macOS, but no default Metal device is available"
                )
            })?;
            let source = OwnedMetalObject::new(ns_string(shader_source))?;
            let mut compile_error = ptr::null_mut();
            let library_id = msg_send_id2_id_error(
                device.id(),
                sel("newLibraryWithSource:options:error:"),
                source.id(),
                ptr::null_mut(),
                &mut compile_error,
            );
            if library_id.is_null() {
                let error = ns_error_localized_description(compile_error)
                    .unwrap_or_else(|| "unknown Metal compiler error".to_string());
                anyhow::bail!("failed to compile Flash-MoE Metal shader library: {error}");
            }
            let library = OwnedMetalObject::new(library_id)?;
            let options_alloc = msg_send_id0(class("MTLCompileOptions"), sel("alloc"));
            let precise_options = OwnedMetalObject::new(msg_send_id0(options_alloc, sel("init")))?;
            msg_send_void1_bool(precise_options.id(), sel("setFastMathEnabled:"), false);
            let mut precise_error = ptr::null_mut();
            let precise_library_id = msg_send_id2_id_error(
                device.id(),
                sel("newLibraryWithSource:options:error:"),
                source.id(),
                precise_options.id(),
                &mut precise_error,
            );
            if precise_library_id.is_null() {
                let error = ns_error_localized_description(precise_error)
                    .unwrap_or_else(|| "unknown precise Metal compiler error".to_string());
                anyhow::bail!("failed to compile precise Qwen Metal shader library: {error}");
            }
            let precise_library = OwnedMetalObject::new(precise_library_id)?;
            let precise_qwen_prepare = OwnedMetalObject::new(compile_pipeline(
                device.id(),
                precise_library.id(),
                names.qwen_prepare_qkv_rows,
            )?)?;
            let precise_qwen_attention_gate = OwnedMetalObject::new(compile_pipeline(
                device.id(),
                precise_library.id(),
                names.qwen_apply_attention_gate,
            )?)?;
            let precise_qwen_final_norm = OwnedMetalObject::new(compile_pipeline(
                device.id(),
                precise_library.id(),
                names.qwen_final_rms_norm_row,
            )?)?;
            let mut compiled = BTreeMap::new();
            for name in names.kernel_names() {
                compiled.insert(
                    name,
                    OwnedMetalObject::new(compile_pipeline(device.id(), library.id(), name)?)?,
                );
            }
            let command_queue =
                OwnedMetalObject::new(msg_send_id0(device.id(), sel("newCommandQueue"))).map_err(
                    |_| anyhow::anyhow!("failed to create Flash-MoE Metal command queue"),
                )?;
            compiled
                .remove(names.qwen_prepare_qkv_rows)
                .expect("compiled fast Qwen QKV preparation pipeline disappeared");
            compiled
                .remove(names.qwen_apply_attention_gate)
                .expect("compiled fast Qwen attention-gate pipeline disappeared");
            compiled
                .remove(names.qwen_final_rms_norm_row)
                .expect("compiled fast Qwen final-norm pipeline disappeared");
            let precise_qwen_prepare = precise_qwen_prepare.into_raw();
            let precise_qwen_attention_gate = precise_qwen_attention_gate.into_raw();
            let precise_qwen_final_norm = precise_qwen_final_norm.into_raw();
            let mut take_pipeline = |name: &'static str| -> MetalObjcId {
                compiled
                    .remove(name)
                    .expect("compiled Metal pipeline name disappeared")
                    .into_raw()
            };
            let pipelines = MetalPipelineSet {
                q4_pipeline: take_pipeline(names.q4),
                q4_bf16_scale_bias_pipeline: take_pipeline(names.q4_bf16_scale_bias),
                mxfp4_e8m0_pipeline: take_pipeline(names.mxfp4_e8m0),
                q4_swiglu_pipeline: take_pipeline(names.q4_swiglu),
                q4_swiglu_bf16_scale_bias_pipeline: take_pipeline(names.q4_swiglu_bf16_scale_bias),
                q4_mmap_pipeline: take_pipeline(names.q4_mmap),
                q4_mmap_bf16_scale_bias_pipeline: take_pipeline(names.q4_mmap_bf16_scale_bias),
                q4_mmap_batch_pipeline: take_pipeline(names.q4_mmap_batch),
                q4_mmap_batch_bf16_scale_bias_pipeline: take_pipeline(
                    names.q4_mmap_batch_bf16_scale_bias,
                ),
                q4_mmap_multilinear_bf16_scale_bias_pipeline: take_pipeline(
                    names.q4_mmap_multilinear_bf16_scale_bias,
                ),
                glm_mla_prepare_query_kv_pipeline: take_pipeline(names.glm_mla_prepare_query_kv),
                glm_mla_absorbed_scores_pipeline: take_pipeline(names.glm_mla_absorbed_scores),
                glm_mla_softmax_pipeline: take_pipeline(names.glm_mla_softmax),
                glm_mla_context_pipeline: take_pipeline(names.glm_mla_context),
                dense_mmap_bf16_pipeline: take_pipeline(names.dense_mmap_bf16),
                dense_mmap_f16_pipeline: take_pipeline(names.dense_mmap_f16),
                dense_mmap_f32_pipeline: take_pipeline(names.dense_mmap_f32),
                dense_matrix_bf16_pipeline: take_pipeline(names.dense_matrix_bf16),
                dense_matrix_f16_pipeline: take_pipeline(names.dense_matrix_f16),
                dense_matrix_f32_pipeline: take_pipeline(names.dense_matrix_f32),
                rms_norm_reduced_pipeline: take_pipeline(names.rms_norm_reduced),
                residual_rms_norm_pipeline: take_pipeline(names.residual_rms_norm),
                attention_pipeline: take_pipeline(names.attention),
                qwen_prepare_qkv_rows_pipeline: precise_qwen_prepare,
                qwen_causal_attention_rows_pipeline: take_pipeline(
                    names.qwen_causal_attention_rows,
                ),
                qwen_apply_attention_gate_pipeline: precise_qwen_attention_gate,
                qwen_final_rms_norm_row_pipeline: precise_qwen_final_norm,
                expert_mlp_pipeline: take_pipeline(names.expert_mlp),
                silu_product_pipeline: take_pipeline(names.silu_product),
                shared_expert_activation_pipeline: take_pipeline(names.shared_expert_activation),
                combine_expert_phase_pipeline: take_pipeline(names.combine_expert_phase),
                qwen_layer_major_gather_pipeline: take_pipeline(names.qwen_layer_major_gather),
                qwen_layer_major_combine_pipeline: take_pipeline(names.qwen_layer_major_combine),
                fill_zero_pipeline: take_pipeline(names.fill_zero),
                topk_vocab_pipeline: take_pipeline(names.topk_vocab),
                linear_conv1d_bf16_pipeline: take_pipeline(names.linear_conv1d_bf16),
                linear_conv1d_f16_pipeline: take_pipeline(names.linear_conv1d_f16),
                linear_conv1d_f32_pipeline: take_pipeline(names.linear_conv1d_f32),
                linear_rms_norm_qk_pipeline: take_pipeline(names.linear_rms_norm_qk),
                linear_decay_beta_bf16_pipeline: take_pipeline(names.linear_decay_beta_bf16),
                linear_decay_beta_f16_pipeline: take_pipeline(names.linear_decay_beta_f16),
                linear_decay_beta_f32_pipeline: take_pipeline(names.linear_decay_beta_f32),
                linear_delta_step_pipeline: take_pipeline(names.linear_delta_step),
                linear_gated_rms_norm_bf16_pipeline: take_pipeline(
                    names.linear_gated_rms_norm_bf16,
                ),
                linear_gated_rms_norm_f16_pipeline: take_pipeline(names.linear_gated_rms_norm_f16),
                linear_gated_rms_norm_f32_pipeline: take_pipeline(names.linear_gated_rms_norm_f32),
            };
            debug_assert!(compiled.is_empty());
            Ok(Self {
                device: device.into_raw(),
                command_queue: command_queue.into_raw(),
                pipelines,
            })
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalRuntime {
    fn drop(&mut self) {
        unsafe {
            self.pipelines.release_with(|pipeline| release(pipeline));
            release(self.command_queue);
            release(self.device);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct DeepSeekMetalPipelineSet {
    pipelines: BTreeMap<&'static str, MetalObjcId>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
// SAFETY: the map is immutable after compilation and contains retained Metal
// pipeline-state handles. Metal permits pipeline states to be shared while
// separate command buffers are encoded on different threads.
unsafe impl Send for DeepSeekMetalPipelineSet {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
// SAFETY: see the `Send` rationale above; concurrent access only reads the
// immutable pipeline map and retained pipeline-state objects.
unsafe impl Sync for DeepSeekMetalPipelineSet {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl DeepSeekMetalPipelineSet {
    unsafe fn compile(device: MetalObjcId) -> anyhow::Result<Self> {
        unsafe {
            let source = OwnedMetalObject::new(ns_string(DEEPSEEK_V4_METAL_SHADERS))?;
            let mut compile_error = ptr::null_mut();
            let library_id = msg_send_id2_id_error(
                device,
                sel("newLibraryWithSource:options:error:"),
                source.id(),
                ptr::null_mut(),
                &mut compile_error,
            );
            if library_id.is_null() {
                let error = ns_error_localized_description(compile_error)
                    .unwrap_or_else(|| "unknown Metal compiler error".to_string());
                anyhow::bail!(
                    "failed to compile required DeepSeek V4 Flash Metal shader library: {error}"
                );
            }
            let library = OwnedMetalObject::new(library_id)?;
            let mut pipelines = BTreeMap::new();
            for &name in DEEPSEEK_V4_REQUIRED_METAL_KERNELS {
                let constants: &[(u64, u64, &[u8])] = match name {
                    "kernel_mul_mv_q8_0_f32"
                    | "kernel_mul_mv_f16_f32"
                    | "kernel_dsv4_shared_gate_up_swiglu_q8_0"
                    | "kernel_dsv4_shared_down_hc_expand4_q8_0"
                    | "kernel_dsv4_q8_hc_expand4_q8_0" => &[(600, 37, &4i16.to_ne_bytes())],
                    "kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32"
                    | "kernel_mul_mv_slots6_q2_K_sum6_f32" => &[(600, 37, &2i16.to_ne_bytes())],
                    "kernel_sum_rows_f32_f32" => &[(1400, 37, &10i16.to_ne_bytes())],
                    "kernel_mul_mm_q8_0_f32" | "kernel_mul_mm_f16_f32" => {
                        &[(700, 53, &[0]), (701, 53, &[1])]
                    }
                    "kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16" | "kernel_mul_mm_id_q2_K_f16" => {
                        &[(700, 53, &[0])]
                    }
                    "kernel_flash_attn_ext_pad" => {
                        &[(100, 53, &[1]), (125, 29, &64i32.to_ne_bytes())]
                    }
                    "kernel_flash_attn_ext_blk" => &[
                        (224, 29, &8i32.to_ne_bytes()),
                        (225, 29, &64i32.to_ne_bytes()),
                    ],
                    "kernel_flash_attn_ext_f16_dk512_dv512" => &[
                        (300, 53, &[1]),
                        (301, 53, &[1]),
                        (302, 53, &[0]),
                        (303, 53, &[0]),
                        (304, 53, &[1]),
                        (310, 53, &[1]),
                        (320, 29, &512i32.to_ne_bytes()),
                        (321, 29, &512i32.to_ne_bytes()),
                        (322, 29, &8i32.to_ne_bytes()),
                    ],
                    _ => &[],
                };
                let pipeline = OwnedMetalObject::new(if constants.is_empty() {
                    compile_pipeline(device, library.id(), name)?
                } else {
                    compile_pipeline_with_constants(device, library.id(), name, constants)?
                })?;
                pipelines.insert(name, pipeline.into_raw());
            }
            let no_pad_constants: &[(u64, u64, &[u8])] = &[
                (300, 53, &[1]),
                (301, 53, &[1]),
                (302, 53, &[0]),
                (303, 53, &[0]),
                (304, 53, &[0]),
                (310, 53, &[1]),
                (320, 29, &512i32.to_ne_bytes()),
                (321, 29, &512i32.to_ne_bytes()),
                (322, 29, &8i32.to_ne_bytes()),
            ];
            let no_pad = OwnedMetalObject::new(compile_pipeline_with_constants(
                device,
                library.id(),
                "kernel_flash_attn_ext_f16_dk512_dv512",
                no_pad_constants,
            )?)?;
            pipelines.insert(
                "kernel_flash_attn_ext_f16_dk512_dv512_nopad",
                no_pad.into_raw(),
            );
            Ok(Self { pipelines })
        }
    }

    pub(crate) fn require(&self, name: &'static str) -> anyhow::Result<MetalObjcId> {
        self.pipelines.get(name).copied().with_context(|| {
            format!("DeepSeek V4 Flash Metal graph is missing compiled kernel {name}")
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for DeepSeekMetalPipelineSet {
    fn drop(&mut self) {
        unsafe {
            for pipeline in self.pipelines.values().copied() {
                release(pipeline);
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalExecutionContext {
    runtime: MetalRuntime,
    deepseek_pipelines: Option<DeepSeekMetalPipelineSet>,
    dense_weights: Option<MetalDenseWeights>,
    linear_attention_state: Mutex<MetalLinearAttentionStateCache>,
    deepseek_state: Mutex<Option<deepseek_execution::DeepSeekV4MetalState>>,
    buffers: Arc<MetalBufferPool>,
    resources: Arc<MetalResourceLedger>,
    norm_epsilon: f32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
// SAFETY: Metal devices, command queues, pipeline states, and retained resource
// handles may be used from multiple threads. Mutable host-side state is behind
// mutexes, while buffer ownership and command completion are explicitly
// serialized by the execution context and buffer pool.
unsafe impl Send for MetalExecutionContext {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
// SAFETY: see the `Send` rationale above. Shared access cannot reach mutable
// host-side state without locking its owning mutex.
unsafe impl Sync for MetalExecutionContext {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalExecutionContext {
    fn drop(&mut self) {
        unsafe {
            if let Some(dense_weights) = self.dense_weights.take() {
                release(dense_weights.buffer);
            }
            if let Ok(linear_state) = self.linear_attention_state.get_mut() {
                release_linear_attention_state(linear_state);
            }
            if let Ok(deepseek_state) = self.deepseek_state.get_mut()
                && let Some(mut state) = deepseek_state.take()
            {
                state.release();
            }
            self.resources.record_resident_resources(0, 0);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalExecutionContext {
    #[allow(dead_code)]
    pub(crate) fn compile(
        dense_mmap: Arc<memmap2::Mmap>,
        dense_len: u64,
        linear_layouts: &[Option<LinearAttentionLayout>],
        norm_epsilon: f32,
    ) -> anyhow::Result<Self> {
        Self::compile_resolved(dense_mmap, dense_len, linear_layouts, norm_epsilon, false)
    }

    pub(crate) fn compile_resolved(
        dense_mmap: Arc<memmap2::Mmap>,
        dense_len: u64,
        linear_layouts: &[Option<LinearAttentionLayout>],
        norm_epsilon: f32,
        deepseek_v4_flash: bool,
    ) -> anyhow::Result<Self> {
        let runtime = MetalRuntime::compile(METAL_SHADERS, MetalPipelineNameSet::new())?;
        let deepseek_pipelines = deepseek_v4_flash
            .then(|| unsafe { DeepSeekMetalPipelineSet::compile(runtime.device) })
            .transpose()?;
        let resources = Arc::new(unsafe { MetalResourceLedger::from_device(runtime.device) });
        let dense_weights = wrap_dense_mmap_as_metal_buffer(runtime.device, dense_mmap, dense_len)?;
        let linear_attention_state =
            allocate_linear_attention_state(runtime.device, linear_layouts)?;
        resources.record_resident_resources(
            dense_weights.as_ref().map_or(0, |weights| weights.len),
            linear_attention_state_bytes(&linear_attention_state),
        );
        unsafe {
            resources.sample_device(runtime.device, false);
        }
        Ok(Self {
            runtime,
            deepseek_pipelines,
            dense_weights,
            linear_attention_state: Mutex::new(linear_attention_state),
            deepseek_state: Mutex::new(None),
            buffers: Arc::new(MetalBufferPool::new(Arc::clone(&resources))),
            resources,
            norm_epsilon,
        })
    }

    pub(crate) fn runtime_capabilities(&self) -> MetalRuntimeCapabilities {
        let capabilities =
            MetalRuntimeCapabilities::from_pipeline_names(MetalPipelineNameSet::new());
        if self.deepseek_pipelines.is_some() {
            capabilities.with_additional(DEEPSEEK_V4_REQUIRED_METAL_KERNELS)
        } else {
            capabilities
        }
    }

    pub(crate) fn deepseek_pipelines(&self) -> anyhow::Result<&DeepSeekMetalPipelineSet> {
        self.deepseek_pipelines
            .as_ref()
            .context("DeepSeek V4 Flash Metal pipelines were not selected by the load-time graph")
    }

    pub(crate) fn runtime(&self) -> &MetalRuntime {
        &self.runtime
    }

    pub(crate) fn dense_weights(&self) -> Option<&MetalDenseWeights> {
        self.dense_weights.as_ref()
    }

    pub(crate) fn buffers(&self) -> &Arc<MetalBufferPool> {
        &self.buffers
    }

    pub(crate) fn norm_epsilon(&self) -> f32 {
        self.norm_epsilon
    }

    pub(crate) fn set_working_set_limit_bytes(&self, limit: usize) -> anyhow::Result<()> {
        self.resources.set_working_set_limit_bytes(limit)
    }

    pub(crate) fn resource_snapshot(&self) -> FlashMoeMetalResourceSnapshot {
        unsafe {
            self.resources.sample_device(self.runtime.device, false);
        }
        self.resources.snapshot()
    }

    pub(crate) fn prepare_resident_expert_backing(
        &self,
        bytes: &ReusableExpertBytes,
    ) -> anyhow::Result<()> {
        let buffer = unsafe {
            persistent_expert_source_buffer(
                self.runtime.device,
                bytes.as_slice(),
                bytes,
                self.buffers.as_ref(),
            )?
        };
        if buffer.is_none() {
            anyhow::bail!(
                "resident expert backing is not page-aligned for a persistent no-copy Metal buffer: bytes={}",
                bytes.len()
            );
        }
        Ok(())
    }

    pub(crate) fn finish_token_boundary(&self, position: usize) -> anyhow::Result<()> {
        let current = unsafe { self.resources.sample_device(self.runtime.device, true) };
        let mut snapshot = self.resources.snapshot();
        if snapshot.active_general_buffers != 0
            || snapshot.transient_expert_buffers != 0
            || snapshot.in_flight_commands != 0
        {
            anyhow::bail!(
                "FlashMoe Metal resource ownership imbalance at token boundary: position={position} active_general_buffers={} active_general_bytes={} pooled_buffers={} pooled_bytes={} transient_expert_buffers={} transient_expert_bytes={} in_flight_commands={} buffer_allocations={} buffer_reuses={} buffer_recycles={} buffer_releases={} phase_cleanup_calls={} phase_cleanup_buffers={}",
                snapshot.active_general_buffers,
                snapshot.active_general_bytes,
                snapshot.pooled_buffers,
                snapshot.pooled_bytes,
                snapshot.transient_expert_buffers,
                snapshot.transient_expert_bytes,
                snapshot.in_flight_commands,
                snapshot.buffer_allocations,
                snapshot.buffer_reuses,
                snapshot.buffer_recycles,
                snapshot.buffer_releases,
                snapshot.phase_cleanup_calls,
                snapshot.phase_cleanup_buffers,
            );
        }
        if current <= snapshot.working_set_limit_bytes {
            return Ok(());
        }

        let (released_pooled_buffers, released_pooled_bytes) = self.buffers.release_idle_buffers();
        let after_drain = unsafe { self.resources.sample_device(self.runtime.device, false) };
        snapshot = self.resources.snapshot();
        if after_drain > snapshot.working_set_limit_bytes {
            self.resources.record_resource_limit_abort();
            anyhow::bail!(
                "FlashMoe Metal resource limit exceeded at token boundary: position={position} current_allocated_bytes={after_drain} working_set_limit_bytes={} recommended_working_set_bytes={} driver_high_water_bytes={} released_pooled_buffers={released_pooled_buffers} released_pooled_bytes={released_pooled_bytes} ledger_live_bytes={}",
                snapshot.working_set_limit_bytes,
                snapshot.recommended_working_set_bytes,
                snapshot.driver_high_water_bytes,
                snapshot.ledger_live_bytes,
            );
        }
        self.resources.record_pressure_recovery();
        tracing::warn!(
            position,
            current_allocated_bytes = current,
            current_after_drain_bytes = after_drain,
            released_pooled_buffers,
            released_pooled_bytes,
            "FlashMoe Metal token boundary recovered from working-set pressure"
        );
        Ok(())
    }

    pub(crate) fn has_resident_dense_weights(&self) -> bool {
        self.dense_weights.is_some()
    }

    pub(crate) fn reset_linear_attention_state(&self) -> anyhow::Result<()> {
        let state = self.linear_attention_state.lock().map_err(|_| {
            anyhow::anyhow!("FlashMoe Metal linear-attention state lock is poisoned during reset")
        })?;
        unsafe {
            for layer in state.layers.iter().flatten() {
                zero_buffer(layer.conv_state, layer.conv_state_len);
                zero_buffer(layer.ssm_state, layer.ssm_state_len);
                zero_buffer(layer.conv_output, layer.conv_dim);
                zero_buffer(layer.delta_output, layer.total_value_width);
                zero_buffer(layer.g_decay, layer.num_value_heads);
                zero_buffer(layer.beta_gate, layer.num_value_heads);
            }
        }
        Ok(())
    }

    pub(crate) fn capture_linear_attention_session_state(
        &self,
    ) -> anyhow::Result<FlashMoeLinearAttentionSessionSnapshot> {
        let state = self.linear_attention_state.lock().map_err(|_| {
            anyhow::anyhow!(
                "FlashMoe Metal linear-attention state lock is poisoned during session capture"
            )
        })?;
        let readback_values = state
            .layers
            .iter()
            .flatten()
            .map(|layer| layer.conv_state_len.saturating_add(layer.ssm_state_len))
            .fold(0usize, usize::saturating_add);
        let snapshot = capture_linear_attention_session_snapshot(&state)?;
        self.buffers
            .resources
            .record_host_readback(readback_values.saturating_mul(std::mem::size_of::<f32>()));
        Ok(snapshot)
    }

    pub(crate) fn restore_linear_attention_session_state(
        &self,
        snapshot: &FlashMoeLinearAttentionSessionSnapshot,
    ) -> anyhow::Result<()> {
        let state = self.linear_attention_state.lock().map_err(|_| {
            anyhow::anyhow!(
                "FlashMoe Metal linear-attention state lock is poisoned during session restore"
            )
        })?;
        restore_linear_attention_session_snapshot(&state, snapshot)
    }

    pub(crate) fn resident_top_candidates(
        &self,
        projection: &ResidentMmapMatvecProjection,
        input: &[f32],
        output_rows: usize,
        top_k: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let dense_weights = self.dense_weights.as_ref().context(
            "FlashMoe unsupported resident topK path: resident dense Metal weights are unavailable",
        )?;
        MetalResidentTopKBuilder::new(
            self.runtime.device,
            self.runtime.command_queue,
            &self.runtime.pipelines,
            dense_weights,
            &self.buffers,
        )
        .execute(projection, input, output_rows, top_k)
    }

    pub(crate) fn resident_post_attention_prep_topk(
        &self,
        projections: &Cmd2ResidentPostAttentionPrepProjections,
        attention_output: &[f32],
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        router_correction_bias: Option<&[f32]>,
    ) -> anyhow::Result<MetalPostAttentionPrep> {
        let dense_weights = self.dense_weights.as_ref().context(
            "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: resident dense Metal weights are unavailable",
        )?;
        MetalResidentPostAttentionPrepBuilder::new(
            self.runtime.device,
            self.runtime.command_queue,
            &self.runtime.pipelines,
            dense_weights,
            &self.buffers,
            self.norm_epsilon,
        )
        .execute(
            projections,
            attention_output,
            residual,
            post_norm_weight,
            router_correction_bias,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_attention_post_attention_prep(
        &self,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        input: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        top_k: usize,
    ) -> anyhow::Result<MetalPostAttentionPrep> {
        MetalFusedLinearAttentionBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.linear_attention_state,
            &self.buffers,
        )
        .execute(layout, bindings, input, residual, post_norm_weight, top_k)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen_linear_attention_graph(
        &self,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        rows: usize,
        width: usize,
        input: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
    ) -> anyhow::Result<MetalLayerMajorPostAttention> {
        MetalFusedLinearAttentionBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.linear_attention_state,
            &self.buffers,
        )
        .execute_layer_major_graph(
            layout,
            bindings,
            rows,
            width,
            input,
            residual,
            post_norm_weight,
            self.norm_epsilon,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen_post_attention_matrix(
        &self,
        out_proj: &ResidentMmapMatvecProjection,
        router: &ResidentMmapMatvecProjection,
        rows: usize,
        attention_width: usize,
        width: usize,
        attention: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
    ) -> anyhow::Result<MetalLayerMajorPostAttention> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_post_attention_matrix(
            out_proj,
            router,
            rows,
            attention_width,
            width,
            attention,
            residual,
            post_norm_weight,
            self.norm_epsilon,
        )?
        .context("Qwen layer-major post-attention matrix requires resident dense weights")
    }

    pub(crate) fn qwen_layer_major_experts(
        &self,
        scheduled: &ScheduledLayerMajorExperts,
        post_attention: &MetalLayerMajorPostAttention,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
    ) -> anyhow::Result<MetalQwenPrefillLayerOutput> {
        let dense_weights = self
            .dense_weights
            .as_ref()
            .context("Qwen layer-major expert graph requires resident dense Metal weights")?;
        MetalScheduledCmd3Builder::new(
            &self.runtime,
            dense_weights,
            Arc::clone(&self.buffers),
            self.norm_epsilon,
        )
        .execute_layer_major(scheduled, post_attention, shared, next_norm_weight)
    }

    pub(crate) fn qwen_final_norm_last_row(
        &self,
        state: &MetalQwenPrefillLayerOutput,
        weight: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        let rows = state.hidden().rows();
        let width = state.hidden().cols();
        if rows == 0 || width == 0 || weight.len() != width || width > u32::MAX as usize {
            anyhow::bail!(
                "Qwen final-row norm has incompatible geometry rows={rows} width={width} weight={}",
                weight.len()
            );
        }
        let row_offset = (rows - 1)
            .checked_mul(width)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("Qwen final-row norm input offset overflow")?;
        let row_offset = u64::try_from(row_offset)
            .context("Qwen final-row norm input offset does not fit Metal")?;
        unsafe {
            let mut owned = Vec::with_capacity(2);
            let weight_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(weight),
                &mut owned,
            )?;
            let output_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                width * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let mut encoding = match MetalCommandEncoding::new(
                self.runtime.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Qwen final-row norm command buffer",
                "failed to create Qwen final-row norm encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release(&owned, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.runtime.pipelines.qwen_final_rms_norm_row_pipeline,
            );
            set_buffer_with_offset(encoder, state.hidden().buffer(), row_offset, 0);
            set_buffer(encoder, weight_buffer, 1);
            set_buffer(encoder, output_buffer, 2);
            let width_u32 = width as u32;
            set_bytes(encoder, u32_as_bytes(&width_u32), 3);
            set_bytes(
                encoder,
                f32_as_bytes(std::slice::from_ref(&self.norm_epsilon)),
                4,
            );
            msg_send_void2_size(
                encoder,
                sel("dispatchThreads:threadsPerThreadgroup:"),
                MetalDispatchSize::new(1, 1, 1),
                MetalDispatchSize::new(1, 1, 1),
            );
            encoding.end_encoding();
            let context = MetalCommandContext::new("qwen_final_norm_last_row")
                .with("rows", rows)
                .with("width", width);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.buffers
                    .recycle_or_release(&owned, error.should_release_buffers());
                return Err(error.into());
            }
            let output = self.buffers.read_f32_buffer(output_buffer, width);
            drop(encoding);
            self.buffers.recycle_or_release(&owned, false);
            Ok(output)
        }
    }

    #[cfg(test)]
    pub(crate) fn qwen_rms_norm_rows(
        &self,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        width: usize,
    ) -> anyhow::Result<Vec<f32>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_rms_norm_rows(input, weight, rows, width, self.norm_epsilon)
    }

    pub(crate) fn resident_projection_batch(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input: &[f32],
    ) -> anyhow::Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute(projections, input)
    }

    #[cfg(test)]
    pub(crate) fn resident_projection_matrix_batch(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input_rows: usize,
        input_cols: usize,
        input: &[f32],
    ) -> anyhow::Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_matrix(projections, input_rows, input_cols, input)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen_causal_attention_rows(
        &self,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        query_rows: usize,
        prefix_rows: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<Vec<f32>> {
        Ok(self
            .qwen_causal_attention_rows_owned(
                queries,
                keys,
                values,
                None,
                query_rows,
                prefix_rows,
                query_heads,
                kv_heads,
                head_dim,
            )?
            .materialize())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen_causal_attention_rows_owned(
        &self,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        query_gates: Option<&[f32]>,
        query_rows: usize,
        prefix_rows: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<MetalQwenAttentionRows> {
        let query_width = query_heads
            .checked_mul(head_dim)
            .context("Qwen attention query width overflow")?;
        let kv_width = kv_heads
            .checked_mul(head_dim)
            .context("Qwen attention KV width overflow")?;
        let key_rows = prefix_rows
            .checked_add(query_rows)
            .context("Qwen attention key row count overflow")?;
        if query_rows == 0
            || query_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || head_dim > 256
            || !query_heads.is_multiple_of(kv_heads)
            || queries.len() != query_rows.saturating_mul(query_width)
            || keys.len() != key_rows.saturating_mul(kv_width)
            || values.len() != key_rows.saturating_mul(kv_width)
            || query_gates.is_some_and(|gates| gates.len() != queries.len())
        {
            anyhow::bail!(
                "Qwen causal attention matrix has incompatible geometry: queries={} keys={} values={} rows={} prefix={} q_heads={} kv_heads={} head_dim={}",
                queries.len(),
                keys.len(),
                values.len(),
                query_rows,
                prefix_rows,
                query_heads,
                kv_heads,
                head_dim
            );
        }
        let query_rows_u32 = u32::try_from(query_rows)?;
        let prefix_rows_u32 = u32::try_from(prefix_rows)?;
        let query_heads_u32 = u32::try_from(query_heads)?;
        let kv_heads_u32 = u32::try_from(kv_heads)?;
        let head_dim_u32 = u32::try_from(head_dim)?;
        let query_values_u32 = u32::try_from(queries.len())?;
        unsafe {
            let mut buffers = Vec::with_capacity(5);
            let allocated = (|| -> anyhow::Result<_> {
                let query_buffer = self.buffers.tracked_buffer_with_bytes(
                    self.runtime.device,
                    f32_as_bytes(queries),
                    &mut buffers,
                )?;
                let key_buffer = self.buffers.tracked_buffer_with_bytes(
                    self.runtime.device,
                    f32_as_bytes(keys),
                    &mut buffers,
                )?;
                let value_buffer = self.buffers.tracked_buffer_with_bytes(
                    self.runtime.device,
                    f32_as_bytes(values),
                    &mut buffers,
                )?;
                let output_buffer = self.buffers.tracked_buffer_with_len(
                    self.runtime.device,
                    queries.len() * std::mem::size_of::<f32>(),
                    &mut buffers,
                )?;
                let gate_buffer = if let Some(gates) = query_gates {
                    self.buffers.tracked_buffer_with_bytes(
                        self.runtime.device,
                        f32_as_bytes(gates),
                        &mut buffers,
                    )?
                } else {
                    query_buffer
                };
                Ok((
                    query_buffer,
                    key_buffer,
                    value_buffer,
                    output_buffer,
                    gate_buffer,
                ))
            })();
            let (query_buffer, key_buffer, value_buffer, output_buffer, gate_buffer) =
                match allocated {
                    Ok(allocated) => allocated,
                    Err(error) => {
                        self.buffers.recycle_or_release_phase(
                            buffers
                                .into_iter()
                                .map(MetalPhaseBuffer::recyclable)
                                .collect(),
                            true,
                        );
                        return Err(error);
                    }
                };
            let mut encoding = match MetalCommandEncoding::new(
                self.runtime.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Qwen causal-attention row command buffer",
                "failed to create Qwen causal-attention row encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release_phase(
                        buffers
                            .into_iter()
                            .map(MetalPhaseBuffer::recyclable)
                            .collect(),
                        true,
                    );
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.runtime.pipelines.qwen_causal_attention_rows_pipeline,
            );
            set_buffer(encoder, query_buffer, 0);
            set_buffer(encoder, key_buffer, 1);
            set_buffer(encoder, value_buffer, 2);
            set_buffer(encoder, output_buffer, 3);
            set_buffer(encoder, gate_buffer, 4);
            set_bytes(encoder, u32_as_bytes(&query_rows_u32), 5);
            set_bytes(encoder, u32_as_bytes(&prefix_rows_u32), 6);
            set_bytes(encoder, u32_as_bytes(&query_heads_u32), 7);
            set_bytes(encoder, u32_as_bytes(&kv_heads_u32), 8);
            set_bytes(encoder, u32_as_bytes(&head_dim_u32), 9);
            let gated = u32::from(query_gates.is_some());
            set_bytes(encoder, u32_as_bytes(&gated), 10);
            dispatch_metal_plan(
                encoder,
                MetalDispatchPlan::qwen_attention_threadgroups(
                    query_rows as u64,
                    query_heads as u64,
                ),
            );
            if query_gates.is_some() {
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.runtime.pipelines.qwen_apply_attention_gate_pipeline,
                );
                set_buffer(encoder, output_buffer, 0);
                set_buffer(encoder, gate_buffer, 1);
                set_bytes(encoder, u32_as_bytes(&query_values_u32), 2);
                dispatch_threads(encoder, queries.len() as u64);
            }
            encoding.end_encoding();
            let context = MetalCommandContext::new("qwen_causal_attention_rows")
                .with("query_rows", query_rows)
                .with("prefix_rows", prefix_rows)
                .with("query_heads", query_heads)
                .with("kv_heads", kv_heads)
                .with("head_dim", head_dim);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.buffers.recycle_or_release_phase(
                    buffers
                        .into_iter()
                        .map(MetalPhaseBuffer::recyclable)
                        .collect(),
                    error.should_release_buffers(),
                );
                return Err(error.into());
            }
            let output = MetalQwenAttentionRows::new(
                Arc::clone(&self.buffers),
                output_buffer,
                query_rows,
                query_width,
            );
            drop(encoding);
            match output {
                Ok(output) => {
                    self.buffers.recycle_or_release_phase(
                        buffers
                            .into_iter()
                            .filter(|buffer| *buffer != output_buffer)
                            .map(MetalPhaseBuffer::recyclable)
                            .collect(),
                        false,
                    );
                    Ok(output)
                }
                Err(error) => {
                    self.buffers.recycle_or_release_phase(
                        buffers
                            .into_iter()
                            .map(MetalPhaseBuffer::recyclable)
                            .collect(),
                        false,
                    );
                    Err(error)
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen_full_attention_graph(
        &self,
        projections: &[ResidentMmapMatvecProjection; 3],
        input: MetalBatchProjectionInput<'_>,
        rows: usize,
        prefix_rows: usize,
        layout: FullAttentionLayout,
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
        rope_sin: &[f32],
        rope_cos: &[f32],
        prefix_keys: &[f32],
        prefix_values: &[f32],
    ) -> anyhow::Result<MetalQwenFullAttentionOutput> {
        let input_cols = input
            .len()
            .checked_div(rows.max(1))
            .context("Qwen full-attention graph input width division failed")?;
        let key_rows = prefix_rows
            .checked_add(rows)
            .context("Qwen full-attention graph key row overflow")?;
        let prefix_kv_values = prefix_rows
            .checked_mul(layout.kv_width)
            .context("Qwen full-attention prefix KV size overflow")?;
        let current_kv_values = rows
            .checked_mul(layout.kv_width)
            .context("Qwen full-attention current KV size overflow")?;
        let all_kv_values = key_rows
            .checked_mul(layout.kv_width)
            .context("Qwen full-attention complete KV size overflow")?;
        let query_values = rows
            .checked_mul(layout.q_width)
            .context("Qwen full-attention query size overflow")?;
        let rotary_half = layout.rotary_dim / 2;
        let rotation_values = rows
            .checked_mul(rotary_half)
            .context("Qwen full-attention rotation size overflow")?;
        if rows == 0
            || input_cols == 0
            || input.len() != rows.saturating_mul(input_cols)
            || layout.rotary_pairing != RotaryPairing::SplitHalf
            || layout.rotary_dim == 0
            || !layout.rotary_dim.is_multiple_of(2)
            || layout.rotary_dim > layout.head_dim
            || layout.head_dim == 0
            || layout.head_dim > 256
            || layout.num_q_heads == 0
            || layout.kv_heads == 0
            || !layout.num_q_heads.is_multiple_of(layout.kv_heads)
            || q_norm_weight.len() != layout.head_dim
            || k_norm_weight.len() != layout.head_dim
            || rope_sin.len() != rotation_values
            || rope_cos.len() != rotation_values
            || prefix_keys.len() != prefix_kv_values
            || prefix_values.len() != prefix_kv_values
            || projections[0].rows() != layout.q_projection_width
            || projections[1].rows() != layout.kv_width
            || projections[2].rows() != layout.kv_width
        {
            anyhow::bail!(
                "Qwen full-attention graph has incompatible geometry rows={rows} prefix={prefix_rows} input={}x{input_cols} q={} kv={} heads={}/{} head_dim={} rotary={} rotations={} prefix_kv={}/{}",
                input.len(),
                layout.q_projection_width,
                layout.kv_width,
                layout.num_q_heads,
                layout.kv_heads,
                layout.head_dim,
                layout.rotary_dim,
                rope_sin.len(),
                prefix_keys.len(),
                prefix_values.len()
            );
        }
        let dense_weights = self
            .dense_weights
            .as_ref()
            .context("Qwen full-attention graph requires resident dense Metal weights")?;
        let mut projection_offsets = Vec::with_capacity(3);
        let mut projection_width = 0usize;
        for projection in projections {
            validate_resident_projection(projection, input_cols, dense_weights.len)?;
            projection_offsets.push(projection_width);
            projection_width = projection_width
                .checked_add(projection.rows())
                .context("Qwen full-attention packed projection width overflow")?;
        }
        let q4_projections = projections
            .iter()
            .map(ResidentMmapMatvecProjection::q4)
            .collect::<Option<Vec<_>>>()
            .context("Qwen full-attention graph requires affine-Q4 Q/K/V projections")?;
        let projected_values = rows
            .checked_mul(projection_width)
            .context("Qwen full-attention packed projection size overflow")?;
        let mut all_keys = Vec::with_capacity(all_kv_values);
        all_keys.extend_from_slice(prefix_keys);
        all_keys.resize(all_kv_values, 0.0);
        let mut all_values = Vec::with_capacity(all_kv_values);
        all_values.extend_from_slice(prefix_values);
        all_values.resize(all_kv_values, 0.0);

        unsafe {
            let mut owned = Vec::with_capacity(24);
            let input_buffer = match input {
                MetalBatchProjectionInput::Cpu(values) => self.buffers.tracked_buffer_with_bytes(
                    self.runtime.device,
                    f32_as_bytes(values),
                    &mut owned,
                )?,
                MetalBatchProjectionInput::Buffer { buffer, .. } => buffer,
            };
            let projected_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                projected_values * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let q_norm_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(q_norm_weight),
                &mut owned,
            )?;
            let k_norm_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(k_norm_weight),
                &mut owned,
            )?;
            let sin_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(rope_sin),
                &mut owned,
            )?;
            let cos_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(rope_cos),
                &mut owned,
            )?;
            let query_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                query_values * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let gated = matches!(layout.q_layout, FullAttentionQLayout::Gated);
            let gate_buffer = if gated {
                self.buffers.tracked_buffer_with_len(
                    self.runtime.device,
                    query_values * std::mem::size_of::<f32>(),
                    &mut owned,
                )?
            } else {
                query_buffer
            };
            let key_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(&all_keys),
                &mut owned,
            )?;
            let value_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(&all_values),
                &mut owned,
            )?;
            let attention_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                query_values * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let projection_builder = MetalResidentProjectionBatchBuilder::new(
                &self.runtime,
                self.dense_weights.as_ref(),
                &self.buffers,
            );
            let mut encoding = match MetalCommandEncoding::new(
                self.runtime.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Qwen full-attention graph command buffer",
                "failed to create Qwen full-attention graph encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release(&owned, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            let encode_result = (|| -> Result<()> {
                if !projection_builder.try_encode_q4_mmap_projection_batch(
                    encoder,
                    &q4_projections,
                    input_buffer,
                    rows,
                    projected_buffer,
                    &projection_offsets,
                    projection_width,
                    &mut owned,
                )? {
                    bail!("Qwen full-attention Q/K/V projections did not resolve a matrix command");
                }
                let projection_width_u32 = u32::try_from(projection_width)?;
                let q_offset_u32 = u32::try_from(projection_offsets[0])?;
                let k_offset_u32 = u32::try_from(projection_offsets[1])?;
                let v_offset_u32 = u32::try_from(projection_offsets[2])?;
                let query_heads_u32 = u32::try_from(layout.num_q_heads)?;
                let kv_heads_u32 = u32::try_from(layout.kv_heads)?;
                let head_dim_u32 = u32::try_from(layout.head_dim)?;
                let rotary_half_u32 = u32::try_from(rotary_half)?;
                let prefix_rows_u32 = u32::try_from(prefix_rows)?;
                let gated_u32 = u32::from(gated);
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.runtime.pipelines.qwen_prepare_qkv_rows_pipeline,
                );
                for (index, buffer) in [
                    projected_buffer,
                    q_norm_buffer,
                    k_norm_buffer,
                    sin_buffer,
                    cos_buffer,
                    query_buffer,
                    gate_buffer,
                    key_buffer,
                    value_buffer,
                ]
                .into_iter()
                .enumerate()
                {
                    set_buffer(encoder, buffer, index as u64);
                }
                for (index, value) in [
                    projection_width_u32,
                    q_offset_u32,
                    k_offset_u32,
                    v_offset_u32,
                    query_heads_u32,
                    kv_heads_u32,
                    head_dim_u32,
                    rotary_half_u32,
                    prefix_rows_u32,
                    gated_u32,
                ]
                .iter()
                .enumerate()
                {
                    set_bytes(encoder, u32_as_bytes(value), (9 + index) as u64);
                }
                msg_send_void2_size(
                    encoder,
                    sel("dispatchThreads:threadsPerThreadgroup:"),
                    MetalDispatchSize::new(
                        layout.num_q_heads.max(layout.kv_heads) as u64,
                        rows as u64,
                        1,
                    ),
                    MetalDispatchSize::new(1, 1, 1),
                );

                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.runtime.pipelines.qwen_causal_attention_rows_pipeline,
                );
                set_buffer(encoder, query_buffer, 0);
                set_buffer(encoder, key_buffer, 1);
                set_buffer(encoder, value_buffer, 2);
                set_buffer(encoder, attention_buffer, 3);
                set_buffer(encoder, gate_buffer, 4);
                let rows_u32 = u32::try_from(rows)?;
                for (index, value) in [
                    rows_u32,
                    prefix_rows_u32,
                    query_heads_u32,
                    kv_heads_u32,
                    head_dim_u32,
                    gated_u32,
                ]
                .iter()
                .enumerate()
                {
                    set_bytes(encoder, u32_as_bytes(value), (5 + index) as u64);
                }
                dispatch_metal_plan(
                    encoder,
                    MetalDispatchPlan::qwen_attention_threadgroups(
                        rows as u64,
                        layout.num_q_heads as u64,
                    ),
                );
                if gated {
                    let query_values_u32 = u32::try_from(query_values)?;
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.qwen_apply_attention_gate_pipeline,
                    );
                    set_buffer(encoder, attention_buffer, 0);
                    set_buffer(encoder, gate_buffer, 1);
                    set_bytes(encoder, u32_as_bytes(&query_values_u32), 2);
                    dispatch_threads(encoder, query_values as u64);
                }
                Ok(())
            })();
            if let Err(error) = encode_result {
                drop(encoding);
                self.buffers.recycle_or_release(&owned, true);
                return Err(error);
            }
            encoding.end_encoding();
            let context = MetalCommandContext::new("qwen_full_attention_graph")
                .with("rows", rows)
                .with("prefix_rows", prefix_rows)
                .with("query_heads", layout.num_q_heads)
                .with("kv_heads", layout.kv_heads)
                .with("head_dim", layout.head_dim);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.buffers
                    .recycle_or_release(&owned, error.should_release_buffers());
                return Err(error.into());
            }
            let current_offset = prefix_rows * layout.kv_width;
            let current_keys =
                self.buffers
                    .read_f32_buffer_offset(key_buffer, current_offset, current_kv_values);
            let current_values = self.buffers.read_f32_buffer_offset(
                value_buffer,
                current_offset,
                current_kv_values,
            );
            let attention = MetalQwenAttentionRows::new(
                Arc::clone(&self.buffers),
                attention_buffer,
                rows,
                layout.q_width,
            );
            drop(encoding);
            match attention {
                Ok(attention) => {
                    for buffer in owned {
                        if buffer != attention_buffer {
                            self.buffers.recycle(buffer);
                        }
                    }
                    MetalQwenFullAttentionOutput::new(attention, current_keys, current_values)
                }
                Err(error) => {
                    self.buffers.recycle_or_release(&owned, false);
                    Err(error)
                }
            }
        }
    }

    pub(crate) fn resident_projection_batch_with_input_buffer(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input_buffer: MetalObjcId,
        input_len: usize,
    ) -> anyhow::Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_with_input_buffer(projections, input_buffer, input_len)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resident_glm_mla_input_projection_chain(
        &self,
        q_a: &ResidentMmapMatvecProjection,
        kv_a: &ResidentMmapMatvecProjection,
        q_b: &ResidentMmapMatvecProjection,
        input: MetalBatchProjectionInput<'_>,
        q_norm_weight: &[f32],
        kv_norm_weight: &[f32],
        kv_lora_rank: usize,
        norm_epsilon: f32,
    ) -> anyhow::Result<Option<(Vec<f32>, Vec<f32>)>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_glm_mla_input_projection_chain(
            q_a,
            kv_a,
            q_b,
            input,
            q_norm_weight,
            kv_norm_weight,
            kv_lora_rank,
            norm_epsilon,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resident_glm_mla_fused_attention(
        &self,
        q_a: &ResidentMmapMatvecProjection,
        kv_a: &ResidentMmapMatvecProjection,
        q_b: &ResidentMmapMatvecProjection,
        embed_q: &DenseQ4MmapMatvecProjection,
        unembed_out: &DenseQ4MmapMatvecProjection,
        input: MetalGlmMlaFusedAttentionInput<'_>,
        q_norm_weight: &[f32],
        kv_norm_weight: &[f32],
        norm_epsilon: f32,
    ) -> anyhow::Result<Option<MetalGlmMlaFusedAttentionOutput>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_glm_mla_fused_attention(
            q_a,
            kv_a,
            q_b,
            embed_q,
            unembed_out,
            input,
            q_norm_weight,
            kv_norm_weight,
            norm_epsilon,
        )
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn resident_q4_multilinear(
        &self,
        projection: &DenseQ4MmapMatvecProjection,
        heads: usize,
        rows_per_head: usize,
        inputs: &[f32],
    ) -> anyhow::Result<Option<Vec<f32>>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_q4_multilinear(projection, heads, rows_per_head, inputs)
    }

    pub(crate) fn resident_glm_mla_absorbed_attention(
        &self,
        embed_q: &DenseQ4MmapMatvecProjection,
        unembed_out: &DenseQ4MmapMatvecProjection,
        input: MetalGlmMlaAbsorbedAttentionInput<'_>,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_glm_mla_absorbed_attention(embed_q, unembed_out, input)
    }

    #[cfg(test)]
    pub(crate) fn read_and_recycle_f32(&self, buffer: MetalObjcId, len: usize) -> Vec<f32> {
        unsafe {
            let values = self.buffers.read_f32_buffer(buffer, len);
            self.buffers.recycle(buffer);
            values
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn validate_linear_attention_session_snapshot(
    resident: &MetalLinearAttentionStateCache,
    snapshot: &FlashMoeLinearAttentionSessionSnapshot,
) -> anyhow::Result<()> {
    if snapshot.len() != resident.layers.len() {
        anyhow::bail!(
            "FlashMoe Metal recurrent session snapshot has {} layers, expected {}",
            snapshot.len(),
            resident.layers.len()
        );
    }
    for (layer, resident) in resident.layers.iter().enumerate() {
        match (resident, snapshot.layer(layer)) {
            (None, None) => {}
            (Some(resident), Some(snapshot)) => {
                let declared = snapshot.state();
                if declared.layer() != layer
                    || declared.conv_state_len() != resident.conv_state_len
                    || declared.ssm_state_len() != resident.ssm_state_len
                    || declared.conv_output_len() != resident.conv_dim
                    || declared.output_len() != resident.total_value_width
                {
                    anyhow::bail!(
                        "FlashMoe Metal recurrent session snapshot for layer {layer} does not match the resolved resident state"
                    );
                }
            }
            (Some(_), None) => {
                anyhow::bail!(
                    "FlashMoe Metal recurrent session snapshot is missing resolved linear-attention layer {layer}"
                );
            }
            (None, Some(_)) => {
                anyhow::bail!(
                    "FlashMoe Metal recurrent session snapshot unexpectedly contains full-attention layer {layer}"
                );
            }
        }
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn capture_linear_attention_session_snapshot(
    resident: &MetalLinearAttentionStateCache,
) -> anyhow::Result<FlashMoeLinearAttentionSessionSnapshot> {
    let layers = resident
        .layers
        .iter()
        .enumerate()
        .map(|(layer, state)| {
            state
                .as_ref()
                .map(|state| unsafe {
                    FlashMoeLinearAttentionLayerSnapshot::new(
                        layer,
                        read_f32_buffer(state.conv_state, state.conv_state_len),
                        read_f32_buffer(state.ssm_state, state.ssm_state_len),
                        state.conv_dim,
                        state.total_value_width,
                    )
                })
                .transpose()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    FlashMoeLinearAttentionSessionSnapshot::new(layers)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn restore_linear_attention_session_snapshot(
    resident: &MetalLinearAttentionStateCache,
    snapshot: &FlashMoeLinearAttentionSessionSnapshot,
) -> anyhow::Result<()> {
    validate_linear_attention_session_snapshot(resident, snapshot)?;
    for (layer, resident) in resident.layers.iter().enumerate() {
        if let (Some(resident), Some(snapshot)) = (resident, snapshot.layer(layer)) {
            unsafe {
                write_f32_buffer(resident.conv_state, snapshot.conv_state());
                write_f32_buffer(resident.ssm_state, snapshot.ssm_state());
                zero_buffer(resident.conv_output, resident.conv_dim);
                zero_buffer(resident.delta_output, resident.total_value_width);
                zero_buffer(resident.g_decay, resident.num_value_heads);
                zero_buffer(resident.beta_gate, resident.num_value_heads);
            }
        }
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalResidentTopKBuilder<'a> {
    device: MetalObjcId,
    command_queue: MetalObjcId,
    pipelines: &'a MetalPipelineSet<MetalObjcId>,
    topk_pipeline: MetalObjcId,
    dense_weights: &'a MetalDenseWeights,
    buffers: &'a MetalBufferPool,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<'a> MetalResidentTopKBuilder<'a> {
    pub(crate) fn new(
        device: MetalObjcId,
        command_queue: MetalObjcId,
        pipelines: &'a MetalPipelineSet<MetalObjcId>,
        dense_weights: &'a MetalDenseWeights,
        buffers: &'a MetalBufferPool,
    ) -> Self {
        Self {
            device,
            command_queue,
            pipelines,
            topk_pipeline: pipelines.topk_vocab_pipeline,
            dense_weights,
            buffers,
        }
    }

    pub(crate) fn execute(
        &self,
        projection: &ResidentMmapMatvecProjection,
        input: &[f32],
        output_rows: usize,
        top_k: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        if output_rows == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        if output_rows > projection.rows() {
            anyhow::bail!(
                "Metal resident topK output width {} exceeds tensor rows {} for {}",
                output_rows,
                projection.rows(),
                projection.tensor_name()
            );
        }
        validate_resident_projection(projection, input.len(), self.dense_weights.len)?;

        let top_k = top_k.min(output_rows).max(1);
        unsafe { self.encode_and_read(projection, input, output_rows, top_k) }
    }

    unsafe fn transient_buffer(
        &self,
        len: usize,
        owned: &mut Vec<MetalObjcId>,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            match self.buffers.buffer_with_len(self.device, len) {
                Ok(buffer) => {
                    owned.push(buffer);
                    Ok(buffer)
                }
                Err(error) => {
                    self.buffers.recycle_or_release(owned, true);
                    owned.clear();
                    Err(error)
                }
            }
        }
    }

    unsafe fn encode_and_read(
        &self,
        projection: &ResidentMmapMatvecProjection,
        input: &[f32],
        rows: usize,
        top_k: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        unsafe {
            let mut transient = Vec::with_capacity(4);
            let input_buffer =
                self.transient_buffer(std::mem::size_of_val(input), &mut transient)?;
            let input_contents = msg_send_ptr0(input_buffer, sel("contents"));
            ptr::copy_nonoverlapping(
                input.as_ptr().cast::<u8>(),
                input_contents.cast::<u8>(),
                std::mem::size_of_val(input),
            );
            let logits_buffer =
                self.transient_buffer(rows * std::mem::size_of::<f32>(), &mut transient)?;
            let indices_buffer =
                self.transient_buffer(top_k * std::mem::size_of::<u32>(), &mut transient)?;
            let values_buffer =
                self.transient_buffer(top_k * std::mem::size_of::<f32>(), &mut transient)?;

            let constants = (|| -> anyhow::Result<_> {
                Ok((
                    u32::try_from(rows).context("resident topK rows exceed u32")?,
                    u32::try_from(top_k).context("resident topK count exceeds u32")?,
                ))
            })();
            let (rows_u32, top_k_u32) = match constants {
                Ok(constants) => constants,
                Err(error) => {
                    self.buffers.recycle_or_release(&transient, true);
                    return Err(error);
                }
            };

            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE resident topK command buffer",
                "failed to create Flash-MoE resident topK compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release(&transient, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();

            if let Err(error) = encode_resident_projection_rows(
                self.pipelines,
                encoder,
                self.dense_weights,
                projection,
                rows,
                input_buffer,
                logits_buffer,
                0,
            ) {
                drop(encoding);
                self.buffers.recycle_or_release(&transient, true);
                return Err(error);
            }

            msg_send_void1_id(encoder, sel("setComputePipelineState:"), self.topk_pipeline);
            set_buffer(encoder, logits_buffer, 0);
            set_buffer(encoder, indices_buffer, 1);
            set_buffer(encoder, values_buffer, 2);
            set_bytes(encoder, u32_as_bytes(&rows_u32), 3);
            set_bytes(encoder, u32_as_bytes(&top_k_u32), 4);
            dispatch_threads(encoder, 1);
            encoding.end_encoding();

            let context = MetalCommandContext::new("resident_topk")
                .with("tensor", projection.tensor_name())
                .with("rows", rows)
                .with("physical_rows", projection.rows())
                .with("cols", projection.cols())
                .with("top_k", top_k);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.buffers
                    .recycle_or_release(&transient, error.should_release_buffers());
                return Err(error.into());
            }

            let indices_ptr = msg_send_ptr0(indices_buffer, sel("contents")).cast::<u32>();
            let values_ptr = msg_send_ptr0(values_buffer, sel("contents")).cast::<f32>();
            let candidates = (0..top_k)
                .map(|index| (*indices_ptr.add(index) as usize, *values_ptr.add(index)))
                .collect();

            drop(encoding);
            self.buffers.recycle_or_release(&transient, false);
            Ok(candidates)
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalResidentPostAttentionPrepBuilder<'a> {
    device: MetalObjcId,
    command_queue: MetalObjcId,
    pipelines: &'a MetalPipelineSet<MetalObjcId>,
    residual_rms_norm_pipeline: MetalObjcId,
    dense_weights: &'a MetalDenseWeights,
    buffers: &'a MetalBufferPool,
    norm_epsilon: f32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<'a> MetalResidentPostAttentionPrepBuilder<'a> {
    pub(crate) fn new(
        device: MetalObjcId,
        command_queue: MetalObjcId,
        pipelines: &'a MetalPipelineSet<MetalObjcId>,
        dense_weights: &'a MetalDenseWeights,
        buffers: &'a MetalBufferPool,
        norm_epsilon: f32,
    ) -> Self {
        Self {
            device,
            command_queue,
            pipelines,
            residual_rms_norm_pipeline: pipelines.residual_rms_norm_pipeline,
            dense_weights,
            buffers,
            norm_epsilon,
        }
    }

    pub(crate) fn execute(
        &self,
        projections: &Cmd2ResidentPostAttentionPrepProjections,
        attention_output: &[f32],
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        router_correction_bias: Option<&[f32]>,
    ) -> anyhow::Result<MetalPostAttentionPrep> {
        let plan = projections.resident_plan(
            attention_output.len(),
            residual.len(),
            post_norm_weight.len(),
        )?;
        validate_resident_projection(
            &projections.out_proj,
            attention_output.len(),
            self.dense_weights.len,
        )?;
        validate_resident_projection(&projections.router, residual.len(), self.dense_weights.len)?;
        let width_u32 =
            u32::try_from(plan.width).context("FlashMoe Metal CMD2 residual width exceeds u32")?;
        unsafe {
            let mut owned = Vec::with_capacity(7);
            let attention_buffer = self.buffers.tracked_buffer_with_bytes(
                self.device,
                f32_as_bytes(attention_output),
                &mut owned,
            )?;
            let (residual_input_buffer, owned_residual_input) = match residual {
                MetalBatchProjectionInput::Cpu(residual) => (
                    self.buffers.tracked_buffer_with_bytes(
                        self.device,
                        f32_as_bytes(residual),
                        &mut owned,
                    )?,
                    true,
                ),
                MetalBatchProjectionInput::Buffer { buffer, .. } => (buffer, false),
            };
            let norm_weight_buffer = self.buffers.tracked_buffer_with_bytes(
                self.device,
                f32_as_bytes(post_norm_weight),
                &mut owned,
            )?;
            let projected_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                plan.width * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let residual_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                plan.width * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let normed_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                plan.width * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let router_logits_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                projections.router.rows() * std::mem::size_of::<f32>(),
                &mut owned,
            )?;

            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE CMD2 post-attention command buffer",
                "failed to create Flash-MoE CMD2 post-attention compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release(&owned, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();

            let encode_result = (|| -> Result<()> {
                encode_resident_projection(
                    self.pipelines,
                    encoder,
                    self.dense_weights,
                    &projections.out_proj,
                    attention_buffer,
                    projected_buffer,
                    0,
                )?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.residual_rms_norm_pipeline,
                );
                set_buffer(encoder, projected_buffer, 0);
                set_buffer(encoder, residual_input_buffer, 1);
                set_buffer(encoder, norm_weight_buffer, 2);
                set_buffer(encoder, residual_buffer, 3);
                set_buffer(encoder, normed_buffer, 4);
                set_bytes(encoder, u32_as_bytes(&width_u32), 5);
                set_bytes(
                    encoder,
                    f32_as_bytes(std::slice::from_ref(&self.norm_epsilon)),
                    6,
                );
                dispatch_single_threadgroup(encoder, 256);
                encode_resident_projection(
                    self.pipelines,
                    encoder,
                    self.dense_weights,
                    &projections.router,
                    normed_buffer,
                    router_logits_buffer,
                    0,
                )
            })();
            if let Err(error) = encode_result {
                drop(encoding);
                self.buffers.recycle_or_release(&owned, true);
                return Err(error);
            }
            encoding.end_encoding();

            let context = MetalCommandContext::new("cmd2_resident_post_attention")
                .with("layer", plan.layer)
                .with("width", plan.width)
                .with("attention_width", plan.attention_width)
                .with("experts", plan.experts)
                .with("top_k", plan.active_count)
                .with("routing_topk", "cpu")
                .with("out_proj", projections.out_proj.tensor_name())
                .with("router", projections.router.tensor_name());
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.buffers
                    .recycle_or_release(&owned, error.should_release_buffers());
                return Err(error.into());
            }

            let router_logits_ptr =
                msg_send_ptr0(router_logits_buffer, sel("contents")).cast::<f32>();
            let router_scores =
                std::slice::from_raw_parts(router_logits_ptr, projections.router.rows()).to_vec();
            let active = if let Some(correction_bias) = router_correction_bias {
                routing_sigmoid_noaux_top_k(&router_scores, correction_bias, plan.active_count)?
            } else {
                routing_softmax_top_k(&router_scores, plan.active_count)
            };
            let output = MetalPostAttentionPrep::new(
                plan.layer,
                plan.width,
                plan.experts,
                active,
                residual_buffer,
                normed_buffer,
            );

            drop(encoding);
            if output.is_err() {
                self.buffers.recycle_or_release(&owned, false);
                return output;
            }
            let transient = owned
                .into_iter()
                .filter(|buffer| *buffer != residual_buffer && *buffer != normed_buffer)
                .collect::<Vec<_>>();
            debug_assert_eq!(transient.len(), if owned_residual_input { 5 } else { 4 });
            self.buffers.recycle_or_release(&transient, false);
            output
        }
    }
}

#[derive(Debug)]
#[must_use = "scheduled Metal CMD3 submissions must be waited or explicitly finished"]
pub(crate) struct MetalScheduledCmd3Submission {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    buffers: Arc<MetalBufferPool>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    command_buffer: Option<MetalObjcId>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    _command_lease: MetalCommandLease,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    phase_buffers: Option<Vec<MetalPhaseBuffer>>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    _expert_slots: Arc<[Arc<ScheduledExpertSlot>]>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    output: MetalCmd3DeferredOutput,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    context: MetalCommandContext,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    scheduled_output: ScheduledCmd3OutputState,
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    unsupported: std::convert::Infallible,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalScheduledCmd3Submission {
    pub(crate) fn next_normed_input(&self) -> anyhow::Result<Option<MetalStateBuffer>> {
        self.output
            .next_normed_buffer
            .zip(self.output.output_state.next_normed())
            .map(|(buffer, state)| MetalStateBuffer::new(buffer, state))
            .transpose()
    }

    pub(crate) fn hidden_input(&self) -> anyhow::Result<MetalStateBuffer> {
        MetalStateBuffer::new(self.output.hidden_buffer, self.output.output_state.hidden())
    }

    pub(crate) fn finish_without_readback(mut self) -> anyhow::Result<()> {
        objc2::rc::autoreleasepool(|_| unsafe {
            let command_buffer = self
                .command_buffer
                .take()
                .context("FlashMoe Metal CMD3 submission has already been finished")?;
            let wait = wait_for_metal_command_buffer(command_buffer, &self.context);
            release(command_buffer);
            let phase_buffers = self.phase_buffers.take().unwrap_or_default();
            match wait {
                Ok(()) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, false);
                    Ok(())
                }
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    Err(error.into())
                }
            }
        })
    }

    pub(crate) fn wait(mut self) -> anyhow::Result<FlashMoeExpertPhaseOutput> {
        objc2::rc::autoreleasepool(|_| unsafe {
            let command_buffer = self
                .command_buffer
                .take()
                .context("FlashMoe Metal CMD3 submission has already been waited")?;
            if let Err(error) = wait_for_metal_command_buffer(command_buffer, &self.context) {
                release(command_buffer);
                self.buffers
                    .recycle_or_release_phase(self.phase_buffers.take().unwrap_or_default(), true);
                return Err(error.into());
            }
            let hidden = self.buffers.read_f32_buffer(
                self.output.hidden_buffer,
                self.output.output_state.hidden().len(),
            );
            let next_normed = self
                .output
                .next_normed_buffer
                .zip(self.output.output_state.next_normed())
                .map(|(buffer, state)| self.buffers.read_f32_buffer(buffer, state.len()));
            release(command_buffer);
            self.buffers
                .recycle_or_release_phase(self.phase_buffers.take().unwrap_or_default(), false);
            self.scheduled_output
                .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(hidden, next_normed))
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalScheduledCmd3Submission {
    fn drop(&mut self) {
        let Some(command_buffer) = self.command_buffer.take() else {
            return;
        };
        objc2::rc::autoreleasepool(|_| unsafe {
            let wait = wait_for_metal_command_buffer(command_buffer, &self.context);
            release(command_buffer);
            self.buffers.recycle_or_release_phase(
                self.phase_buffers.take().unwrap_or_default(),
                wait.is_err(),
            );
            if let Err(error) = wait {
                tracing::warn!(
                    target: "flashmoe::resources",
                    error = %error,
                    "FlashMoe cleaned up an unfinished Metal CMD3 submission after a command failure"
                );
            } else {
                tracing::warn!(
                    target: "flashmoe::resources",
                    "FlashMoe cleaned up a Metal CMD3 submission that was dropped without an explicit wait"
                );
            }
        });
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl MetalScheduledCmd3Submission {
    pub(crate) fn next_normed_input(&self) -> anyhow::Result<Option<MetalStateBuffer>> {
        match self.unsupported {}
    }

    pub(crate) fn hidden_input(&self) -> anyhow::Result<MetalStateBuffer> {
        match self.unsupported {}
    }

    pub(crate) fn finish_without_readback(self) -> anyhow::Result<()> {
        match self.unsupported {}
    }

    pub(crate) fn wait(self) -> anyhow::Result<FlashMoeExpertPhaseOutput> {
        match self.unsupported {}
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalScheduledCmd3Builder<'a> {
    runtime: &'a MetalRuntime,
    dense_weights: &'a MetalDenseWeights,
    buffers: Arc<MetalBufferPool>,
    norm_epsilon: f32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Clone, Copy)]
struct MetalLayerMajorOneRowReference {
    expert_outputs: MetalObjcId,
    shared_output: MetalObjcId,
    shared_router: MetalObjcId,
    shared_router_width: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
type MetalLayerMajorGroupPlan = (Vec<u32>, Vec<u32>, Vec<(usize, usize)>);

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn qwen_layer_major_group_plan(
    route_slots: &[usize],
    unique_experts: usize,
    active_experts: usize,
) -> anyhow::Result<MetalLayerMajorGroupPlan> {
    if route_slots.is_empty() || unique_experts == 0 || active_experts == 0 {
        bail!("Qwen layer-major group plan requires non-empty routes and experts");
    }
    let mut routes_by_slot = vec![Vec::new(); unique_experts];
    for (route, slot) in route_slots.iter().copied().enumerate() {
        routes_by_slot
            .get_mut(slot)
            .with_context(|| {
                format!("Qwen layer-major route {route} references missing expert slot {slot}")
            })?
            .push(route);
    }
    if routes_by_slot.iter().any(Vec::is_empty) {
        bail!("Qwen layer-major unique expert union contains an unused slot");
    }

    let mut grouped_source_rows = Vec::with_capacity(route_slots.len());
    let mut grouped_output_indices = vec![0u32; route_slots.len()];
    let mut groups = Vec::with_capacity(routes_by_slot.len());
    for routes in &routes_by_slot {
        let start = grouped_source_rows.len();
        for route in routes {
            let row = route / active_experts;
            grouped_source_rows
                .push(u32::try_from(row).context("Qwen layer-major source row exceeds u32")?);
            grouped_output_indices[*route] = u32::try_from(grouped_source_rows.len() - 1)
                .context("Qwen layer-major grouped route index exceeds u32")?;
        }
        groups.push((start, routes.len()));
    }
    Ok((grouped_source_rows, grouped_output_indices, groups))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<'a> MetalScheduledCmd3Builder<'a> {
    pub(crate) fn new(
        runtime: &'a MetalRuntime,
        dense_weights: &'a MetalDenseWeights,
        buffers: Arc<MetalBufferPool>,
        norm_epsilon: f32,
    ) -> Self {
        Self {
            runtime,
            dense_weights,
            buffers,
            norm_epsilon,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn submit(
        &self,
        position: usize,
        layer: usize,
        experts: Arc<[Arc<ScheduledExpertSlot>]>,
        routing_weights: &[f32],
        input: MetalPostAttentionPrep,
        scheduled_output: ScheduledCmd3OutputState,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
        payloads: &[ScheduledExpertPhaseMlpPayload<'_>],
    ) -> anyhow::Result<MetalScheduledCmd3Submission> {
        if scheduled_output.layer != layer {
            self.recycle_input(input);
            anyhow::bail!(
                "FlashMoe Metal CMD3 output layer {} does not match scheduled layer {layer}",
                scheduled_output.layer
            );
        }
        if input.input.state() != input.state {
            self.recycle_input(input);
            anyhow::bail!(
                "FlashMoe Metal CMD3 typed input state does not match post-attention state for layer {layer}"
            );
        }
        if input.input.state().routing().layer() != layer || input.state.routing().layer() != layer
        {
            self.recycle_input(input);
            anyhow::bail!("FlashMoe Metal CMD3 input layer does not match scheduled layer {layer}");
        }
        let width = input.input.width();
        let output_state = scheduled_output.state();
        let command_plan = match MetalCmd3ExecutionPlan::new(
            position,
            layer,
            experts.len(),
            width,
            routing_weights.len(),
            output_state,
            shared,
            next_norm_weight.map(|weight| weight.len()),
            payloads,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                self.recycle_input(input);
                return Err(error.context(
                    "FlashMoe unsupported Metal CMD3 phase: invalid scheduled command plan",
                ));
            }
        };
        if let Err(error) = Self::require_shared_implementation(command_plan.shared.source) {
            self.recycle_input(input);
            return Err(error);
        }
        let input_buffers = match MetalCmd3InputBuffers::new(
            command_plan.phase,
            input.normed_buffer,
            input.residual_buffer,
        ) {
            Ok(input_buffers) => input_buffers,
            Err(error) => {
                self.recycle_input(input);
                return Err(error.context(
                    "FlashMoe unsupported Metal CMD3 phase: invalid scheduled input buffers",
                ));
            }
        };
        objc2::rc::autoreleasepool(|_| unsafe {
            self.encode_and_submit(
                command_plan,
                input_buffers,
                experts,
                routing_weights,
                scheduled_output,
                shared,
                next_norm_weight,
                payloads,
            )
        })
    }

    pub(crate) fn execute_layer_major(
        &self,
        scheduled: &ScheduledLayerMajorExperts,
        post_attention: &MetalLayerMajorPostAttention,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
    ) -> anyhow::Result<MetalQwenPrefillLayerOutput> {
        let rows = scheduled.rows();
        let active_experts = scheduled.active_experts();
        let normed = post_attention.normed();
        let residual = post_attention.residual();
        let width = residual.cols();
        let route_count = rows
            .checked_mul(active_experts)
            .context("Qwen layer-major route count overflow")?;
        if rows == 0
            || active_experts == 0
            || width == 0
            || residual.rows() != rows
            || residual.state().role() != FlashMoeStateBufferRole::Residual
            || normed.rows() != rows
            || normed.cols() != width
            || normed.state().role() != FlashMoeStateBufferRole::Normed
            || scheduled.route_slots().len() != route_count
            || scheduled.weights().len() != route_count
            || scheduled.experts().is_empty()
            || next_norm_weight.is_some_and(|weight| weight.len() != width)
        {
            bail!(
                "Qwen layer-major expert command has incompatible geometry layer={} rows={rows} width={width} active={active_experts} normed={} residual={} routes={} weights={} unique={}",
                scheduled.layer(),
                normed.values(),
                residual.values(),
                scheduled.route_slots().len(),
                scheduled.weights().len(),
                scheduled.experts().len()
            );
        }

        let (grouped_source_rows, grouped_output_indices, groups) = qwen_layer_major_group_plan(
            scheduled.route_slots(),
            scheduled.experts().len(),
            active_experts,
        )?;
        debug_assert_eq!(grouped_source_rows.len(), route_count);

        objc2::rc::autoreleasepool(|_| unsafe {
            self.encode_and_execute_layer_major(
                scheduled,
                width,
                normed,
                residual,
                &grouped_source_rows,
                &grouped_output_indices,
                &groups,
                shared,
                next_norm_weight,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_one_row_scalar_reference(
        &self,
        encoder: MetalObjcId,
        scheduled: &ScheduledLayerMajorExperts,
        width: usize,
        intermediate: usize,
        payloads: &[ScheduledExpertPhaseMlpPayload<'_>],
        normed_buffer: MetalObjcId,
        _residual_buffer: MetalObjcId,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        phase_buffers: &mut Vec<MetalPhaseBuffer>,
        source_buffers: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<MetalLayerMajorOneRowReference> {
        unsafe {
            let active_experts = scheduled.active_experts();
            let gate =
                self.phase_buffer(intermediate * std::mem::size_of::<f32>(), phase_buffers)?;
            let up = self.phase_buffer(intermediate * std::mem::size_of::<f32>(), phase_buffers)?;
            let activated =
                self.phase_buffer(intermediate * std::mem::size_of::<f32>(), phase_buffers)?;
            let expert_outputs = self.phase_buffer(
                active_experts * width * std::mem::size_of::<f32>(),
                phase_buffers,
            )?;
            for route in 0..active_experts {
                let slot = scheduled.route_slots()[route];
                let payload = payloads
                    .get(slot)
                    .and_then(ScheduledExpertPhaseMlpPayload::q4_checked)
                    .with_context(|| {
                        format!("one-row Qwen parity route {route} has no affine-Q4 payload")
                    })?;
                self.encode_q4_matvec(
                    encoder,
                    &payload.gate,
                    payload.gate_source(),
                    normed_buffer,
                    gate,
                    0,
                    phase_buffers,
                    source_buffers,
                )?;
                self.encode_q4_matvec(
                    encoder,
                    &payload.up,
                    payload.up_source(),
                    normed_buffer,
                    up,
                    0,
                    phase_buffers,
                    source_buffers,
                )?;
                let intermediate_u32 = u32::try_from(intermediate)
                    .context("one-row Qwen parity intermediate exceeds u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.runtime.pipelines.silu_product_pipeline,
                );
                set_buffer(encoder, gate, 0);
                set_buffer(encoder, up, 1);
                set_buffer(encoder, activated, 2);
                set_bytes(encoder, u32_as_bytes(&intermediate_u32), 3);
                dispatch_threads(encoder, intermediate as u64);
                self.encode_q4_matvec(
                    encoder,
                    &payload.down,
                    payload.down_source(),
                    activated,
                    expert_outputs,
                    (route * width * std::mem::size_of::<f32>()) as u64,
                    phase_buffers,
                    source_buffers,
                )?;
            }

            let (shared_output, shared_router, shared_router_width) = match shared {
                ScheduledSharedExpertPhaseRef::Resident(shared) => {
                    let shape = shared.validated_shape()?;
                    let gate = self.phase_buffer(
                        shape.total_intermediate * std::mem::size_of::<f32>(),
                        phase_buffers,
                    )?;
                    let up = self.phase_buffer(
                        shape.total_intermediate * std::mem::size_of::<f32>(),
                        phase_buffers,
                    )?;
                    let activated = self.phase_buffer(
                        shape.total_intermediate * std::mem::size_of::<f32>(),
                        phase_buffers,
                    )?;
                    let output =
                        self.phase_buffer(width * std::mem::size_of::<f32>(), phase_buffers)?;
                    let router_width = shared.router.as_ref().map_or(0, |router| router.rows());
                    let router = self.phase_buffer(
                        router_width.max(1) * std::mem::size_of::<f32>(),
                        phase_buffers,
                    )?;
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &shared.gate,
                        normed_buffer,
                        gate,
                        0,
                    )?;
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &shared.up,
                        normed_buffer,
                        up,
                        0,
                    )?;
                    if let Some(projection) = shared.router.as_ref() {
                        encode_resident_projection(
                            &self.runtime.pipelines,
                            encoder,
                            self.dense_weights,
                            projection,
                            normed_buffer,
                            router,
                            0,
                        )?;
                    }
                    let total_u32 = u32::try_from(shape.total_intermediate)
                        .context("one-row shared intermediate exceeds u32")?;
                    let intermediate_u32 = u32::try_from(shape.intermediate)
                        .context("one-row shared expert width exceeds u32")?;
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.shared_expert_activation_pipeline,
                    );
                    set_buffer(encoder, gate, 0);
                    set_buffer(encoder, up, 1);
                    set_buffer(encoder, router, 2);
                    set_buffer(encoder, activated, 3);
                    set_bytes(encoder, u32_as_bytes(&intermediate_u32), 4);
                    set_bytes(encoder, u32_as_bytes(&total_u32), 5);
                    dispatch_threads(encoder, shape.total_intermediate as u64);
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &shared.down,
                        activated,
                        output,
                        0,
                    )?;
                    (output, router, router_width)
                }
                ScheduledSharedExpertPhaseRef::None => {
                    let output =
                        self.phase_buffer(width * std::mem::size_of::<f32>(), phase_buffers)?;
                    let width_u32 =
                        u32::try_from(width).context("one-row shared fill width exceeds u32")?;
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.fill_zero_pipeline,
                    );
                    set_buffer(encoder, output, 0);
                    set_bytes(encoder, u32_as_bytes(&width_u32), 1);
                    dispatch_threads(encoder, width as u64);
                    let router = self.phase_buffer_with_bytes(
                        f32_as_bytes(std::slice::from_ref(&80.0f32)),
                        phase_buffers,
                    )?;
                    (output, router, 0)
                }
                ScheduledSharedExpertPhaseRef::Dense(_) => {
                    bail!("one-row Qwen parity requires resident shared projections")
                }
            };
            Ok(MetalLayerMajorOneRowReference {
                expert_outputs,
                shared_output,
                shared_router,
                shared_router_width,
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_and_execute_layer_major(
        &self,
        scheduled: &ScheduledLayerMajorExperts,
        width: usize,
        normed: MetalMatrixBuffer,
        residual: MetalMatrixBuffer,
        grouped_source_rows: &[u32],
        grouped_output_indices: &[u32],
        groups: &[(usize, usize)],
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
    ) -> anyhow::Result<MetalQwenPrefillLayerOutput> {
        unsafe {
            let rows = scheduled.rows();
            let active_experts = scheduled.active_experts();
            let route_count = rows * active_experts;
            let payloads = scheduled
                .experts()
                .iter()
                .map(|expert| expert.scheduled_cmd3_expert_phase_payload(width))
                .collect::<Result<Vec<_>>>()?;
            let intermediate = payloads
                .first()
                .and_then(ScheduledExpertPhaseMlpPayload::q4_checked)
                .map(|payload| payload.gate.rows)
                .context("Qwen layer-major expert graph requires fixed affine-Q4 payloads")?;
            for payload in &payloads {
                let q4 = payload
                    .q4_checked()
                    .context("Qwen layer-major expert graph requires fixed affine-Q4 payloads")?;
                if q4.gate.rows != intermediate
                    || !q4
                        .gate
                        .scale_bias_dtype
                        .eq_ignore_ascii_case(super::experts::EXPERT_SCALE_BIAS_DTYPE_BF16)
                {
                    bail!("Qwen layer-major expert graph requires uniform BF16 affine-Q4 experts");
                }
            }

            let mut phase_buffers = Vec::new();
            let setup = (|| -> anyhow::Result<_> {
                let gathered_buffer = self.phase_buffer(
                    route_count * width * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let grouped_source_rows_buffer = self.phase_buffer_with_bytes(
                    u32_as_bytes_slice(grouped_source_rows),
                    &mut phase_buffers,
                )?;
                let weights_buffer = self.phase_buffer_with_bytes(
                    f32_as_bytes(scheduled.weights()),
                    &mut phase_buffers,
                )?;
                let grouped_indices_buffer = self.phase_buffer_with_bytes(
                    u32_as_bytes_slice(grouped_output_indices),
                    &mut phase_buffers,
                )?;
                let route_intermediate_values = route_count
                    .checked_mul(intermediate)
                    .context("Qwen layer-major expert activation size overflow")?;
                let grouped_output_values = route_count
                    .checked_mul(width)
                    .context("Qwen layer-major expert output size overflow")?;
                let gate_buffer = self.phase_buffer(
                    route_intermediate_values * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let up_buffer = self.phase_buffer(
                    route_intermediate_values * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let activated_buffer = self.phase_buffer(
                    route_intermediate_values * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let grouped_output_buffer = self.phase_buffer(
                    grouped_output_values * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let shared_output_buffer = self.phase_buffer(
                    rows * width * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let hidden_buffer = self.phase_buffer(
                    rows * width * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let next_norm = next_norm_weight
                    .map(|weight| -> anyhow::Result<_> {
                        let weight_buffer =
                            self.phase_buffer_with_bytes(f32_as_bytes(weight), &mut phase_buffers)?;
                        let output_buffer = self.phase_buffer(
                            rows * width * std::mem::size_of::<f32>(),
                            &mut phase_buffers,
                        )?;
                        Ok((weight_buffer, output_buffer))
                    })
                    .transpose()?;
                Ok((
                    gathered_buffer,
                    grouped_source_rows_buffer,
                    weights_buffer,
                    grouped_indices_buffer,
                    gate_buffer,
                    up_buffer,
                    activated_buffer,
                    grouped_output_buffer,
                    shared_output_buffer,
                    hidden_buffer,
                    next_norm,
                ))
            })();
            let (
                gathered_buffer,
                grouped_source_rows_buffer,
                weights_buffer,
                grouped_indices_buffer,
                gate_buffer,
                up_buffer,
                activated_buffer,
                grouped_output_buffer,
                shared_output_buffer,
                hidden_buffer,
                next_norm,
            ) = match setup {
                Ok(values) => values,
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };

            let mut encoding = match MetalCommandEncoding::new(
                self.runtime.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Qwen layer-major expert command buffer",
                "failed to create Qwen layer-major expert encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            let encode_result = (||
             -> anyhow::Result<(
                MetalObjcId,
                MetalObjcId,
                MetalObjcId,
                MetalObjcId,
                usize,
                Option<MetalObjcId>,
                Option<MetalLayerMajorOneRowReference>,
            )> {
                let gathered_values = route_count
                    .checked_mul(width)
                    .context("Qwen layer-major gathered matrix size overflow")?;
                let gathered_values_u32 = u32::try_from(gathered_values)
                    .context("Qwen layer-major gathered matrix exceeds u32")?;
                let width_u32 =
                    u32::try_from(width).context("Qwen layer-major gather width exceeds u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.runtime.pipelines.qwen_layer_major_gather_pipeline,
                );
                set_buffer(encoder, normed.buffer(), 0);
                set_buffer(encoder, grouped_source_rows_buffer, 1);
                set_buffer(encoder, gathered_buffer, 2);
                set_bytes(encoder, u32_as_bytes(&width_u32), 3);
                set_bytes(encoder, u32_as_bytes(&gathered_values_u32), 4);
                dispatch_threads(encoder, gathered_values as u64);

                let mut source_buffers = MetalExpertSourceBufferCache::default();
                for (((payload, group), expert), slot_index) in payloads
                    .iter()
                    .zip(groups)
                    .zip(scheduled.experts().iter())
                    .zip(0usize..)
                {
                    let q4 = payload
                        .q4_checked()
                        .context("Qwen layer-major expert payload changed after validation")?;
                    let (start, count) = *group;
                    self.encode_layer_major_q4_matrix(
                        encoder,
                        &q4.gate,
                        q4.gate_source(),
                        gathered_buffer,
                        start * width,
                        gate_buffer,
                        start * intermediate,
                        count,
                        &mut phase_buffers,
                        &mut source_buffers,
                    )?;
                    self.encode_layer_major_q4_matrix(
                        encoder,
                        &q4.up,
                        q4.up_source(),
                        gathered_buffer,
                        start * width,
                        up_buffer,
                        start * intermediate,
                        count,
                        &mut phase_buffers,
                        &mut source_buffers,
                    )?;
                    debug_assert_eq!(expert.expert(), scheduled.experts()[slot_index].expert());
                }
                let activated_values = route_count * intermediate;
                let activated_u32 = u32::try_from(activated_values)
                    .context("Qwen layer-major activation width exceeds u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.runtime.pipelines.silu_product_pipeline,
                );
                set_buffer(encoder, gate_buffer, 0);
                set_buffer(encoder, up_buffer, 1);
                set_buffer(encoder, activated_buffer, 2);
                set_bytes(encoder, u32_as_bytes(&activated_u32), 3);
                dispatch_threads(encoder, activated_values as u64);
                for (payload, group) in payloads.iter().zip(groups) {
                    let q4 = payload
                        .q4_checked()
                        .context("Qwen layer-major expert payload changed after validation")?;
                    let (start, count) = *group;
                    self.encode_layer_major_q4_matrix(
                        encoder,
                        &q4.down,
                        q4.down_source(),
                        activated_buffer,
                        start * intermediate,
                        grouped_output_buffer,
                        start * width,
                        count,
                        &mut phase_buffers,
                        &mut source_buffers,
                    )?;
                }

                let (shared_router, shared_router_width) = match shared {
                    ScheduledSharedExpertPhaseRef::Resident(shared) => {
                        let shape = shared.validated_shape()?;
                        let shared_gate = self.phase_buffer(
                            rows * shape.total_intermediate * std::mem::size_of::<f32>(),
                            &mut phase_buffers,
                        )?;
                        let shared_up = self.phase_buffer(
                            rows * shape.total_intermediate * std::mem::size_of::<f32>(),
                            &mut phase_buffers,
                        )?;
                        let shared_activated = self.phase_buffer(
                            rows * shape.total_intermediate * std::mem::size_of::<f32>(),
                            &mut phase_buffers,
                        )?;
                        let shared_router_width = shared.router.as_ref().map_or(0, |p| p.rows());
                        let shared_router = self.phase_buffer(
                            rows * shared_router_width.max(1) * std::mem::size_of::<f32>(),
                            &mut phase_buffers,
                        )?;
                        self.encode_layer_major_resident_matrix(
                            encoder,
                            &shared.gate,
                            normed.buffer(),
                            shared_gate,
                            rows,
                        )?;
                        self.encode_layer_major_resident_matrix(
                            encoder,
                            &shared.up,
                            normed.buffer(),
                            shared_up,
                            rows,
                        )?;
                        if let Some(router) = shared.router.as_ref() {
                            self.encode_layer_major_resident_matrix(
                                encoder,
                                router,
                                normed.buffer(),
                                shared_router,
                                rows,
                            )?;
                        }
                        let shared_values = rows * shape.total_intermediate;
                        let shared_values_u32 = u32::try_from(shared_values)
                            .context("Qwen shared activation width exceeds u32")?;
                        msg_send_void1_id(
                            encoder,
                            sel("setComputePipelineState:"),
                            self.runtime.pipelines.silu_product_pipeline,
                        );
                        set_buffer(encoder, shared_gate, 0);
                        set_buffer(encoder, shared_up, 1);
                        set_buffer(encoder, shared_activated, 2);
                        set_bytes(encoder, u32_as_bytes(&shared_values_u32), 3);
                        dispatch_threads(encoder, shared_values as u64);
                        self.encode_layer_major_resident_matrix(
                            encoder,
                            &shared.down,
                            shared_activated,
                            shared_output_buffer,
                            rows,
                        )?;
                        (shared_router, shared_router_width)
                    }
                    ScheduledSharedExpertPhaseRef::None => {
                        let zero_width = u32::try_from(rows * width)
                            .context("Qwen shared zero-fill width exceeds u32")?;
                        msg_send_void1_id(
                            encoder,
                            sel("setComputePipelineState:"),
                            self.runtime.pipelines.fill_zero_pipeline,
                        );
                        set_buffer(encoder, shared_output_buffer, 0);
                        set_bytes(encoder, u32_as_bytes(&zero_width), 1);
                        dispatch_threads(encoder, (rows * width) as u64);
                        (shared_output_buffer, 0)
                    }
                    ScheduledSharedExpertPhaseRef::Dense(_) => {
                        bail!("Qwen layer-major shared expert requires resident projections")
                    }
                };

                let one_row_reference = (rows == 1)
                    .then(|| {
                        self.encode_one_row_scalar_reference(
                            encoder,
                            scheduled,
                            width,
                            intermediate,
                            &payloads,
                            normed.buffer(),
                            residual.buffer(),
                            shared,
                            &mut phase_buffers,
                            &mut source_buffers,
                        )
                    })
                    .transpose()?;

                let rows_u32 = u32::try_from(rows).context("Qwen batch rows exceed u32")?;
                let width_u32 = u32::try_from(width).context("Qwen batch width exceeds u32")?;
                let active_u32 = u32::try_from(active_experts)
                    .context("Qwen active expert count exceeds u32")?;
                let shared_router_width_u32 = u32::try_from(shared_router_width)
                    .context("Qwen shared router width exceeds u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.runtime.pipelines.qwen_layer_major_combine_pipeline,
                );
                set_buffer(encoder, residual.buffer(), 0);
                set_buffer(encoder, shared_output_buffer, 1);
                set_buffer(encoder, grouped_output_buffer, 2);
                set_buffer(encoder, weights_buffer, 3);
                set_buffer(encoder, grouped_indices_buffer, 4);
                set_buffer(encoder, shared_router, 5);
                set_buffer(encoder, hidden_buffer, 6);
                set_bytes(encoder, u32_as_bytes(&rows_u32), 7);
                set_bytes(encoder, u32_as_bytes(&width_u32), 8);
                set_bytes(encoder, u32_as_bytes(&active_u32), 9);
                set_bytes(encoder, u32_as_bytes(&shared_router_width_u32), 10);
                dispatch_threads(encoder, (rows * width) as u64);
                let next_normed_buffer = if let Some((weight_buffer, output_buffer)) = next_norm {
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.rms_norm_reduced_pipeline,
                    );
                    set_buffer(encoder, hidden_buffer, 0);
                    set_buffer(encoder, weight_buffer, 1);
                    set_buffer(encoder, output_buffer, 2);
                    set_bytes(encoder, u32_as_bytes(&width_u32), 3);
                    set_bytes(
                        encoder,
                        f32_as_bytes(std::slice::from_ref(&self.norm_epsilon)),
                        4,
                    );
                    msg_send_void2_size(
                        encoder,
                        sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                        MetalDispatchSize::new(1, rows as u64, 1),
                        MetalDispatchSize::new(256, 1, 1),
                    );
                    Some(output_buffer)
                } else {
                    None
                };
                Ok((
                    hidden_buffer,
                    grouped_output_buffer,
                    shared_output_buffer,
                    shared_router,
                    shared_router_width,
                    next_normed_buffer,
                    one_row_reference,
                ))
            })();
            let (
                hidden_buffer,
                grouped_output_buffer,
                shared_output_buffer,
                shared_router,
                shared_router_width,
                next_normed_buffer,
                one_row_reference,
            ) = match encode_result {
                Ok(result) => result,
                Err(error) => {
                    drop(encoding);
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };
            encoding.end_encoding();
            let context = MetalCommandContext::new("qwen_layer_major_experts")
                .with("layer", scheduled.layer())
                .with("rows", rows)
                .with("active_experts", active_experts)
                .with("unique_experts", scheduled.experts().len());
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.buffers
                    .recycle_or_release_phase(phase_buffers, error.should_release_buffers());
                return Err(error.into());
            }
            if let Some(reference) = one_row_reference {
                let grouped = self
                    .buffers
                    .read_f32_buffer(grouped_output_buffer, route_count * width);
                let scalar = self
                    .buffers
                    .read_f32_buffer(reference.expert_outputs, route_count * width);
                for route in 0..route_count {
                    let grouped_route = grouped_output_indices[route] as usize;
                    for col in 0..width {
                        let actual = grouped[grouped_route * width + col];
                        let expected = scalar[route * width + col];
                        if actual.to_bits() != expected.to_bits() {
                            drop(encoding);
                            self.buffers.recycle_or_release_phase(phase_buffers, false);
                            bail!(
                                "Qwen one-row layer-major expert parity failed layer={} route={route} col={col}: matrix={actual} scalar={expected} delta={}",
                                scheduled.layer(),
                                (actual - expected).abs()
                            );
                        }
                    }
                }
                let shared_actual = self.buffers.read_f32_buffer(shared_output_buffer, width);
                let shared_expected = self.buffers.read_f32_buffer(reference.shared_output, width);
                for (col, (actual, expected)) in
                    shared_actual.iter().zip(shared_expected.iter()).enumerate()
                {
                    if actual.to_bits() != expected.to_bits() {
                        drop(encoding);
                        self.buffers.recycle_or_release_phase(phase_buffers, false);
                        bail!(
                            "Qwen one-row layer-major shared-expert parity failed layer={} col={col}: matrix={actual} scalar={expected} delta={}",
                            scheduled.layer(),
                            (actual - expected).abs()
                        );
                    }
                }
                if shared_router_width != reference.shared_router_width {
                    drop(encoding);
                    self.buffers.recycle_or_release_phase(phase_buffers, false);
                    bail!("Qwen one-row shared-router widths differ");
                }
                if shared_router_width > 0 {
                    let router_actual = self
                        .buffers
                        .read_f32_buffer(shared_router, shared_router_width);
                    let router_expected = self
                        .buffers
                        .read_f32_buffer(reference.shared_router, shared_router_width);
                    for (index, (actual, expected)) in
                        router_actual.iter().zip(router_expected.iter()).enumerate()
                    {
                        if actual.to_bits() != expected.to_bits() {
                            drop(encoding);
                            self.buffers.recycle_or_release_phase(phase_buffers, false);
                            bail!(
                                "Qwen one-row shared-router parity failed layer={} index={index}: matrix={actual} scalar={expected} delta={}",
                                scheduled.layer(),
                                (actual - expected).abs()
                            );
                        }
                    }
                }
            }
            drop(encoding);
            let output = MetalQwenPrefillLayerOutput::new(
                Arc::clone(&self.buffers),
                scheduled.layer(),
                hidden_buffer,
                next_normed_buffer,
                rows,
                width,
            );
            match output {
                Ok(output) => {
                    let transient = phase_buffers
                        .into_iter()
                        .filter(|phase| {
                            phase.id != hidden_buffer && next_normed_buffer != Some(phase.id)
                        })
                        .collect();
                    self.buffers.recycle_or_release_phase(transient, false);
                    Ok(output)
                }
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, false);
                    Err(error)
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_layer_major_q4_matrix(
        &self,
        encoder: MetalObjcId,
        payload: &super::experts::Q4MatvecPayload<'_>,
        source: super::experts::Q4MatvecSource<'_>,
        input: MetalObjcId,
        input_value_offset: usize,
        output: MetalObjcId,
        output_value_offset: usize,
        input_rows: usize,
        buffers: &mut Vec<MetalPhaseBuffer>,
        source_buffers: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<()> {
        unsafe {
            if input_rows == 0
                || !payload
                    .scale_bias_dtype
                    .eq_ignore_ascii_case(super::experts::EXPERT_SCALE_BIAS_DTYPE_BF16)
            {
                bail!("Qwen layer-major Q4 matrix requires non-empty BF16 affine rows");
            }
            let buffer = self.expert_source_buffer(
                source.bytes,
                source.reusable_bytes,
                buffers,
                source_buffers,
            )?;
            self.encode_layer_major_q4_matrix_from_buffer(
                encoder,
                buffer,
                source.packed_offset as u64,
                source.scale_offset as u64,
                source.bias_offset as u64,
                payload.rows,
                payload.cols,
                payload.group_size,
                input,
                input_value_offset,
                output,
                output_value_offset,
                input_rows,
            )
        }
    }

    unsafe fn encode_layer_major_resident_matrix(
        &self,
        encoder: MetalObjcId,
        projection: &ResidentMmapMatvecProjection,
        input: MetalObjcId,
        output: MetalObjcId,
        input_rows: usize,
    ) -> anyhow::Result<()> {
        unsafe {
            match projection {
                ResidentMmapMatvecProjection::Q4(projection) => {
                    if !projection
                        .scale_bias_dtype
                        .eq_ignore_ascii_case(super::experts::EXPERT_SCALE_BIAS_DTYPE_BF16)
                    {
                        bail!("Qwen layer-major resident matrix requires BF16 scale/bias weights");
                    }
                    self.encode_layer_major_q4_matrix_from_buffer(
                        encoder,
                        self.dense_weights.buffer,
                        projection.packed_byte_offset,
                        projection.scales_byte_offset,
                        projection.biases_byte_offset,
                        projection.rows,
                        projection.cols,
                        projection.group_size,
                        input,
                        0,
                        output,
                        0,
                        input_rows,
                    )
                }
                ResidentMmapMatvecProjection::Dense(projection) => encode_dense_resident_matrix(
                    &self.runtime.pipelines,
                    encoder,
                    self.dense_weights,
                    projection,
                    input,
                    output,
                    input_rows,
                ),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_layer_major_q4_matrix_from_buffer(
        &self,
        encoder: MetalObjcId,
        weights: MetalObjcId,
        packed_offset: u64,
        scale_offset: u64,
        bias_offset: u64,
        rows: usize,
        cols: usize,
        group_size: usize,
        input: MetalObjcId,
        input_value_offset: usize,
        output: MetalObjcId,
        output_value_offset: usize,
        input_rows: usize,
    ) -> anyhow::Result<()> {
        unsafe {
            let rows_u32 = u32::try_from(rows).context("Qwen matrix rows exceed u32")?;
            let cols_u32 = u32::try_from(cols).context("Qwen matrix cols exceed u32")?;
            let groups_u32 = u32::try_from(cols.div_ceil(group_size.max(1)))
                .context("Qwen matrix groups exceed u32")?;
            let group_size_u32 =
                u32::try_from(group_size).context("Qwen matrix group size exceeds u32")?;
            let input_rows_u32 =
                u32::try_from(input_rows).context("Qwen matrix input rows exceed u32")?;
            let input_rows_per_threadgroup = if cols <= 2_048 && input_rows > 1 {
                2u32
            } else {
                1u32
            };
            let projection_count = 1u32;
            let row_offset = 0u32;
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.runtime
                    .pipelines
                    .q4_mmap_batch_bf16_scale_bias_pipeline,
            );
            set_buffer(encoder, weights, 0);
            set_buffer_with_offset(
                encoder,
                input,
                (input_value_offset * std::mem::size_of::<f32>()) as u64,
                1,
            );
            set_buffer_with_offset(
                encoder,
                output,
                (output_value_offset * std::mem::size_of::<f32>()) as u64,
                2,
            );
            set_bytes(encoder, u64_as_bytes(&packed_offset), 3);
            set_bytes(encoder, u64_as_bytes(&scale_offset), 4);
            set_bytes(encoder, u64_as_bytes(&bias_offset), 5);
            set_bytes(encoder, u32_as_bytes(&row_offset), 6);
            set_bytes(encoder, u32_as_bytes(&rows_u32), 7);
            set_bytes(encoder, u32_as_bytes(&groups_u32), 8);
            set_bytes(encoder, u32_as_bytes(&projection_count), 9);
            set_bytes(encoder, u32_as_bytes(&cols_u32), 10);
            set_bytes(encoder, u32_as_bytes(&group_size_u32), 11);
            set_bytes(encoder, u32_as_bytes(&input_rows_u32), 12);
            set_bytes(encoder, u32_as_bytes(&input_rows_per_threadgroup), 13);
            dispatch_q4_mmap_matrix_bf16_threadgroups(
                encoder,
                rows as u64,
                input_rows as u64,
                input_rows_per_threadgroup as u64,
            );
            Ok(())
        }
    }

    fn recycle_input(&self, input: MetalPostAttentionPrep) {
        self.buffers
            .recycle_or_release(&[input.normed_buffer, input.residual_buffer], false);
    }

    fn require_shared_implementation(source: MetalCmd3SharedPhaseSource) -> anyhow::Result<()> {
        match source {
            MetalCmd3SharedPhaseSource::Resident | MetalCmd3SharedPhaseSource::None => Ok(()),
            MetalCmd3SharedPhaseSource::Dense => anyhow::bail!(
                "FlashMoe unsupported Metal CMD3 implementation: dense CPU shared-expert weights are not a declared implementation"
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_and_submit(
        &self,
        command_plan: MetalCmd3ExecutionPlan,
        input_buffers: MetalCmd3InputBuffers,
        experts: Arc<[Arc<ScheduledExpertSlot>]>,
        routing_weights: &[f32],
        scheduled_output: ScheduledCmd3OutputState,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
        payloads: &[ScheduledExpertPhaseMlpPayload<'_>],
    ) -> anyhow::Result<MetalScheduledCmd3Submission> {
        unsafe {
            let mut phase_buffers = vec![
                MetalPhaseBuffer::recyclable(input_buffers.normed),
                MetalPhaseBuffer::recyclable(input_buffers.residual),
            ];
            let setup = (|| -> anyhow::Result<_> {
                let output_buffers = self.output_buffers(&command_plan, &mut phase_buffers)?;
                let combine_buffers = self.combine_buffers(
                    command_plan.combine,
                    routing_weights,
                    &mut phase_buffers,
                )?;
                Ok((output_buffers, combine_buffers))
            })();
            let (output_buffers, combine_buffers) = match setup {
                Ok(setup) => setup,
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };

            let mut encoding = match MetalCommandEncoding::new(
                self.runtime.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE scheduled CMD3 command buffer",
                "failed to create Flash-MoE scheduled CMD3 compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();

            let encode_result = (|| -> anyhow::Result<()> {
                let mut source_buffers = MetalExpertSourceBufferCache::default();
                let shared_router = self.encode_shared(
                    encoder,
                    &command_plan,
                    input_buffers,
                    &output_buffers,
                    combine_buffers,
                    shared,
                    &mut phase_buffers,
                )?;
                for (active_plan, payload) in command_plan.active_experts.iter().zip(payloads) {
                    let active_work = self.active_work_buffers(*active_plan, &mut phase_buffers)?;
                    let active_stage = MetalCmd3ActiveExpertStageBuffers::new(
                        *active_plan,
                        input_buffers,
                        &output_buffers,
                        active_work,
                    )?;
                    match payload {
                        ScheduledExpertPhaseMlpPayload::Q4(payload) => {
                            let gate_out = active_stage
                                .work
                                .gate_out
                                .context("Metal Q4 expert stage has no gate projection buffer")?;
                            let up_out = active_stage
                                .work
                                .up_out
                                .context("Metal Q4 expert stage has no up projection buffer")?;
                            self.encode_q4_matvec(
                                encoder,
                                &payload.gate,
                                payload.gate_source(),
                                active_stage.normed,
                                gate_out,
                                0,
                                &mut phase_buffers,
                                &mut source_buffers,
                            )?;
                            self.encode_q4_matvec(
                                encoder,
                                &payload.up,
                                payload.up_source(),
                                active_stage.normed,
                                up_out,
                                0,
                                &mut phase_buffers,
                                &mut source_buffers,
                            )?;
                            let intermediate = active_stage.plan.intermediate_u32()?;
                            msg_send_void1_id(
                                encoder,
                                sel("setComputePipelineState:"),
                                self.runtime.pipelines.silu_product_pipeline,
                            );
                            set_buffer(encoder, gate_out, 0);
                            set_buffer(encoder, up_out, 1);
                            set_buffer(encoder, active_stage.activated, 2);
                            set_bytes(encoder, u32_as_bytes(&intermediate), 3);
                            dispatch_threads(encoder, active_stage.plan.intermediate as u64);
                            self.encode_q4_matvec(
                                encoder,
                                &payload.down,
                                payload.down_source(),
                                active_stage.activated,
                                active_stage.expert_outputs,
                                active_stage.output_offset,
                                &mut phase_buffers,
                                &mut source_buffers,
                            )?;
                        }
                        ScheduledExpertPhaseMlpPayload::Dense(payload) => {
                            self.encode_dense_swiglu(
                                encoder,
                                payload,
                                active_stage,
                                &mut phase_buffers,
                                &mut source_buffers,
                            )?;
                        }
                        ScheduledExpertPhaseMlpPayload::DeepSeekGguf(_) => {
                            bail!(
                                "DeepSeek GGUF expert payload reached the Qwen/GLM CMD3 builder instead of its load-resolved fused Metal builder"
                            );
                        }
                    }
                }
                self.encode_combine(
                    encoder,
                    command_plan.combine,
                    command_plan.shared,
                    input_buffers,
                    &output_buffers,
                    combine_buffers,
                    shared_router,
                    &mut phase_buffers,
                )?;
                if let (Some(weight), Some(next_normed), Some(next_plan)) = (
                    next_norm_weight,
                    output_buffers.next_normed,
                    command_plan.next_norm,
                ) {
                    let next_buffers = self.next_norm_buffers(
                        next_plan,
                        weight,
                        output_buffers.hidden,
                        next_normed,
                        combine_buffers.width,
                        &mut phase_buffers,
                    )?;
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.rms_norm_reduced_pipeline,
                    );
                    set_buffer(encoder, next_buffers.hidden, 0);
                    set_buffer(encoder, next_buffers.weight, 1);
                    set_buffer(encoder, next_buffers.next_normed, 2);
                    set_buffer(encoder, next_buffers.width, 3);
                    set_bytes(
                        encoder,
                        f32_as_bytes(std::slice::from_ref(&self.norm_epsilon)),
                        4,
                    );
                    dispatch_single_threadgroup(encoder, next_plan.dispatch_threads);
                }
                Ok(())
            })();
            if let Err(error) = encode_result {
                drop(encoding);
                self.buffers.recycle_or_release_phase(phase_buffers, true);
                return Err(error);
            }
            encoding.end_encoding();

            let output = match MetalCmd3DeferredOutput::new(
                output_buffers.hidden,
                output_buffers.next_normed,
                command_plan.phase.output_state,
            ) {
                Ok(output) => output,
                Err(error) => {
                    drop(encoding);
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };
            let expert_ids = experts
                .iter()
                .map(|slot| slot.expert().to_string())
                .collect::<Vec<_>>()
                .join(",");
            let context = command_plan.command_context(expert_ids);
            let (command_buffer, command_lease) = encoding.into_command_buffer();
            commit_metal_command_buffer(command_buffer, &context);
            Ok(MetalScheduledCmd3Submission {
                buffers: Arc::clone(&self.buffers),
                command_buffer: Some(command_buffer),
                _command_lease: command_lease,
                phase_buffers: Some(phase_buffers),
                _expert_slots: experts,
                output,
                context,
                scheduled_output,
            })
        }
    }

    unsafe fn output_buffers(
        &self,
        plan: &MetalCmd3ExecutionPlan,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalCmd3OutputBuffers> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let expert_outputs = self.phase_buffer(layout.expert_outputs_bytes, buffers)?;
            let shared_output = self.phase_buffer(layout.shared_output_bytes, buffers)?;
            let hidden = self.phase_buffer(layout.hidden_output_bytes, buffers)?;
            let next_normed = layout
                .next_normed_output_bytes
                .map(|bytes| self.phase_buffer(bytes, buffers))
                .transpose()?;
            MetalCmd3OutputBuffers::new(plan, expert_outputs, shared_output, hidden, next_normed)
        }
    }

    unsafe fn combine_buffers(
        &self,
        plan: MetalCmd3CombinePlan,
        routing_weights: &[f32],
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalCmd3CombineBuffers> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let weight_bytes = f32_as_bytes(routing_weights);
            if weight_bytes.len() != layout.routing_weights_bytes {
                anyhow::bail!(
                    "FlashMoe Metal CMD3 routing weights byte length {} does not match {}",
                    weight_bytes.len(),
                    layout.routing_weights_bytes
                );
            }
            let routing_weights = self.phase_buffer_with_bytes(weight_bytes, buffers)?;
            let width = self.phase_buffer_with_bytes(u32_as_bytes(&layout.width_u32), buffers)?;
            let active_count =
                self.phase_buffer_with_bytes(u32_as_bytes(&layout.active_count_u32), buffers)?;
            MetalCmd3CombineBuffers::new(plan, routing_weights, width, active_count)
        }
    }

    unsafe fn shared_work_buffers(
        &self,
        plan: MetalCmd3SharedPhasePlan,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalCmd3SharedWorkBuffers> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let gate = self.phase_buffer(layout.projection_output_bytes, buffers)?;
            let up = self.phase_buffer(layout.projection_output_bytes, buffers)?;
            let router = self.phase_buffer(layout.router_output_bytes, buffers)?;
            let activated = self.phase_buffer(layout.projection_output_bytes, buffers)?;
            let total = self
                .phase_buffer_with_bytes(u32_as_bytes(&layout.total_intermediate_u32), buffers)?;
            let intermediate =
                self.phase_buffer_with_bytes(u32_as_bytes(&layout.intermediate_u32), buffers)?;
            MetalCmd3SharedWorkBuffers::new(plan, gate, up, router, activated, total, intermediate)
        }
    }

    unsafe fn active_work_buffers(
        &self,
        plan: MetalCmd3ActiveExpertPlan,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalCmd3ActiveExpertWorkBuffers> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let gate_out = layout
                .projection_output_bytes
                .map(|bytes| self.phase_buffer(bytes, buffers))
                .transpose()?;
            let up_out = layout
                .projection_output_bytes
                .map(|bytes| self.phase_buffer(bytes, buffers))
                .transpose()?;
            let activated = self.phase_buffer(layout.activation_bytes, buffers)?;
            MetalCmd3ActiveExpertWorkBuffers::new(plan, gate_out, up_out, activated)
        }
    }

    unsafe fn next_norm_buffers(
        &self,
        plan: MetalCmd3NextNormPlan,
        weight: &[f32],
        hidden: MetalObjcId,
        next_normed: MetalObjcId,
        width: MetalObjcId,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalCmd3NextNormBuffers> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let bytes = f32_as_bytes(&weight[..plan.width]);
            if bytes.len() != layout.weight_bytes {
                anyhow::bail!(
                    "FlashMoe Metal CMD3 next-norm weight bytes {} do not match {}",
                    bytes.len(),
                    layout.weight_bytes
                );
            }
            let weight = self.phase_buffer_with_bytes(bytes, buffers)?;
            MetalCmd3NextNormBuffers::new(plan, hidden, weight, next_normed, width)
        }
    }

    unsafe fn encode_shared(
        &self,
        encoder: MetalObjcId,
        command: &MetalCmd3ExecutionPlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        combine: MetalCmd3CombineBuffers,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            match command.shared.source {
                MetalCmd3SharedPhaseSource::Resident => {
                    let weights = shared.resident().context(
                        "FlashMoe Metal CMD3 resident shared stage has no resolved projections",
                    )?;
                    let work = self.shared_work_buffers(command.shared, buffers)?;
                    let stage = MetalCmd3SharedStageBuffers::projected(
                        command.shared,
                        inputs,
                        outputs,
                        combine,
                        work,
                    )?;
                    let work = stage
                        .work
                        .context("missing Metal CMD3 shared work buffers")?;
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &weights.gate,
                        stage.normed,
                        work.gate_out,
                        0,
                    )?;
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &weights.up,
                        stage.normed,
                        work.up_out,
                        0,
                    )?;
                    let shared_router = if let Some(router) = weights.router.as_ref() {
                        encode_resident_projection(
                            &self.runtime.pipelines,
                            encoder,
                            self.dense_weights,
                            router,
                            stage.normed,
                            work.router_out,
                            0,
                        )?;
                        work.router_out
                    } else {
                        self.phase_buffer_with_bytes(
                            f32_as_bytes(std::slice::from_ref(&80.0f32)),
                            buffers,
                        )?
                    };
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.shared_expert_activation_pipeline,
                    );
                    set_buffer(encoder, work.gate_out, 0);
                    set_buffer(encoder, work.up_out, 1);
                    set_buffer(encoder, shared_router, 2);
                    set_buffer(encoder, work.activated, 3);
                    set_buffer(encoder, work.intermediate, 4);
                    set_buffer(encoder, work.total_intermediate, 5);
                    dispatch_threads(encoder, command.shared.activation_dispatch_threads());
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &weights.down,
                        work.activated,
                        stage.shared_output,
                        0,
                    )?;
                    Ok(shared_router)
                }
                MetalCmd3SharedPhaseSource::None => {
                    let fill_width = u32::try_from(command.shared.fill_zero_width())
                        .context("Metal CMD3 shared fill width exceeds u32")?;
                    let stage = MetalCmd3SharedStageBuffers::fill_zero(
                        command.shared,
                        inputs,
                        outputs,
                        combine,
                    )?;
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.fill_zero_pipeline,
                    );
                    set_buffer(encoder, stage.shared_output, 0);
                    set_bytes(encoder, u32_as_bytes(&fill_width), 1);
                    dispatch_threads(encoder, command.shared.fill_zero_width() as u64);
                    Ok(stage.shared_output)
                }
                MetalCmd3SharedPhaseSource::Dense => unreachable!("rejected before encoding"),
            }
        }
    }

    unsafe fn encode_combine(
        &self,
        encoder: MetalObjcId,
        plan: MetalCmd3CombinePlan,
        shared_plan: MetalCmd3SharedPhasePlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        combine: MetalCmd3CombineBuffers,
        shared_router: MetalObjcId,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<()> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let stage = MetalCmd3CombineStageBuffers::new(plan, inputs, outputs, combine)?;
            let grouped_indices = (0..plan.active_count)
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("Metal CMD3 active expert index exceeds u32")?;
            let grouped_indices =
                self.phase_buffer_with_bytes(u32_as_bytes_slice(&grouped_indices), buffers)?;
            let rows = 1u32;
            let shared_router_width = if shared_plan.source == MetalCmd3SharedPhaseSource::Resident
            {
                u32::try_from(shared_plan.shared_experts)
                    .context("Metal CMD3 shared router width exceeds u32")?
            } else {
                0
            };
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.runtime.pipelines.qwen_layer_major_combine_pipeline,
            );
            set_buffer(encoder, stage.residual, 0);
            set_buffer(encoder, stage.shared_output, 1);
            set_buffer(encoder, stage.expert_outputs, 2);
            set_buffer(encoder, stage.routing_weights, 3);
            set_buffer(encoder, grouped_indices, 4);
            set_buffer(encoder, shared_router, 5);
            set_buffer(encoder, stage.hidden, 6);
            set_bytes(encoder, u32_as_bytes(&rows), 7);
            set_bytes(encoder, u32_as_bytes(&layout.width_u32), 8);
            set_bytes(encoder, u32_as_bytes(&layout.active_count_u32), 9);
            set_bytes(encoder, u32_as_bytes(&shared_router_width), 10);
            dispatch_threads(encoder, stage.plan.dispatch_threads);
            Ok(())
        }
    }

    unsafe fn encode_dense_swiglu(
        &self,
        encoder: MetalObjcId,
        payload: &ScheduledDenseExpertPhaseMlpPayload<'_>,
        stage: MetalCmd3ActiveExpertStageBuffers,
        buffers: &mut Vec<MetalPhaseBuffer>,
        source_buffers: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<()> {
        unsafe {
            let gate_out = stage
                .work
                .gate_out
                .context("Metal dense expert stage has no gate projection buffer")?;
            let up_out = stage
                .work
                .up_out
                .context("Metal dense expert stage has no up projection buffer")?;
            self.encode_dense_expert_matvec(
                encoder,
                &payload.gate,
                stage.normed,
                gate_out,
                0,
                buffers,
                source_buffers,
            )?;
            self.encode_dense_expert_matvec(
                encoder,
                &payload.up,
                stage.normed,
                up_out,
                0,
                buffers,
                source_buffers,
            )?;
            let intermediate = stage.plan.intermediate_u32()?;
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.runtime.pipelines.silu_product_pipeline,
            );
            set_buffer(encoder, gate_out, 0);
            set_buffer(encoder, up_out, 1);
            set_buffer(encoder, stage.activated, 2);
            set_bytes(encoder, u32_as_bytes(&intermediate), 3);
            dispatch_threads(encoder, stage.plan.intermediate as u64);
            self.encode_dense_expert_matvec(
                encoder,
                &payload.down,
                stage.activated,
                stage.expert_outputs,
                stage.output_offset,
                buffers,
                source_buffers,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_dense_expert_matvec(
        &self,
        encoder: MetalObjcId,
        payload: &super::experts::DenseMatvecPayload<'_>,
        input: MetalObjcId,
        output: MetalObjcId,
        output_offset: u64,
        buffers: &mut Vec<MetalPhaseBuffer>,
        source_buffers: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<()> {
        unsafe {
            let source = payload.source;
            let buffer = self.expert_source_buffer(
                source.bytes,
                source.reusable_bytes,
                buffers,
                source_buffers,
            )?;
            let rows =
                u32::try_from(payload.rows).context("Metal CMD3 dense expert rows exceed u32")?;
            let cols =
                u32::try_from(payload.cols).context("Metal CMD3 dense expert cols exceed u32")?;
            let byte_offset = u64::try_from(source.byte_offset)
                .context("Metal CMD3 dense expert byte offset exceeds u64")?;
            let pipeline = match payload.dtype {
                super::experts::DenseExpertDtype::Bf16 => {
                    self.runtime.pipelines.dense_mmap_bf16_pipeline
                }
                super::experts::DenseExpertDtype::F16 => {
                    self.runtime.pipelines.dense_mmap_f16_pipeline
                }
            };
            msg_send_void1_id(encoder, sel("setComputePipelineState:"), pipeline);
            set_buffer(encoder, buffer, 0);
            set_buffer(encoder, input, 1);
            set_buffer_with_offset(encoder, output, output_offset, 2);
            set_bytes(encoder, u64_as_bytes(&byte_offset), 3);
            set_bytes(encoder, u32_as_bytes(&rows), 4);
            set_bytes(encoder, u32_as_bytes(&cols), 5);
            dispatch_threads(encoder, payload.rows as u64);
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_q4_matvec(
        &self,
        encoder: MetalObjcId,
        payload: &super::experts::Q4MatvecPayload<'_>,
        source: super::experts::Q4MatvecSource<'_>,
        input: MetalObjcId,
        output: MetalObjcId,
        output_offset: u64,
        buffers: &mut Vec<MetalPhaseBuffer>,
        source_buffers: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<()> {
        unsafe {
            let buffer = self.expert_source_buffer(
                source.bytes,
                source.reusable_bytes,
                buffers,
                source_buffers,
            )?;
            let rows = u32::try_from(payload.rows).context("Metal CMD3 expert rows exceed u32")?;
            let cols = u32::try_from(payload.cols).context("Metal CMD3 expert cols exceed u32")?;
            let groups = u32::try_from(payload.cols.div_ceil(payload.group_size).max(1))
                .context("Metal CMD3 expert groups exceed u32")?;
            let group_size = u32::try_from(payload.group_size)
                .context("Metal CMD3 expert group size exceeds u32")?;
            let pipeline = if payload
                .scale_bias_dtype
                .eq_ignore_ascii_case(super::experts::EXPERT_SCALE_DTYPE_E8M0)
            {
                self.runtime.pipelines.mxfp4_e8m0_pipeline
            } else {
                self.runtime.pipelines.q4_bf16_scale_bias_pipeline
            };
            msg_send_void1_id(encoder, sel("setComputePipelineState:"), pipeline);
            set_buffer_with_offset(encoder, buffer, source.packed_offset as u64, 0);
            set_buffer(encoder, input, 1);
            set_buffer_with_offset(encoder, buffer, source.scale_offset as u64, 2);
            set_buffer_with_offset(encoder, buffer, source.bias_offset as u64, 3);
            set_buffer_with_offset(encoder, output, output_offset, 4);
            set_bytes(encoder, u32_as_bytes(&rows), 5);
            set_bytes(encoder, u32_as_bytes(&cols), 6);
            set_bytes(encoder, u32_as_bytes(&groups), 7);
            set_bytes(encoder, u32_as_bytes(&group_size), 8);
            dispatch_q4_threadgroups(encoder, payload.rows as u64);
            Ok(())
        }
    }

    unsafe fn expert_source_buffer(
        &self,
        bytes: &[u8],
        reusable_bytes: Option<&ReusableExpertBytes>,
        buffers: &mut Vec<MetalPhaseBuffer>,
        cache: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<MetalObjcId> {
        if let Some(buffer) = cache.get(bytes) {
            return Ok(buffer);
        }
        if let Some(reusable_bytes) = reusable_bytes
            && let Some(buffer) = unsafe {
                persistent_expert_source_buffer(
                    self.runtime.device,
                    bytes,
                    reusable_bytes,
                    self.buffers.as_ref(),
                )?
            }
        {
            cache.insert(bytes, buffer);
            return Ok(buffer);
        }
        let phase = unsafe { self.expert_source_phase_buffer(bytes)? };
        let buffer = phase.id;
        buffers.push(phase);
        cache.insert(bytes, buffer);
        Ok(buffer)
    }

    unsafe fn expert_source_phase_buffer(&self, bytes: &[u8]) -> anyhow::Result<MetalPhaseBuffer> {
        unsafe {
            if let Some(buffer) = wrap_expert_slot_as_metal_buffer(self.runtime.device, bytes) {
                return Ok(MetalPhaseBuffer::borrowed_expert(buffer));
            }
            let buffer = self
                .buffers
                .transient_expert_buffer_with_bytes(self.runtime.device, bytes)?;
            Ok(MetalPhaseBuffer::transient_expert(buffer))
        }
    }

    unsafe fn phase_buffer(
        &self,
        bytes: usize,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            let buffer = self.buffers.buffer_with_len(self.runtime.device, bytes)?;
            buffers.push(MetalPhaseBuffer::recyclable(buffer));
            Ok(buffer)
        }
    }

    unsafe fn phase_buffer_with_bytes(
        &self,
        bytes: &[u8],
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            let buffer = self.buffers.buffer_with_bytes(self.runtime.device, bytes)?;
            buffers.push(MetalPhaseBuffer::recyclable(buffer));
            Ok(buffer)
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod projection;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) use projection::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod linear_attention;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) use linear_attention::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod qwen_execution;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) use qwen_execution::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod qwen_cmd3;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) use qwen_cmd3::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalPhaseBufferClass {
    General,
    BorrowedExpert,
    TransientExpert,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalPhaseBuffer {
    pub(crate) id: MetalObjcId,
    pub(crate) class: MetalPhaseBufferClass,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalPhaseBuffer {
    pub(crate) fn recyclable(id: MetalObjcId) -> Self {
        Self {
            id,
            class: MetalPhaseBufferClass::General,
        }
    }

    pub(crate) fn transient_expert(id: MetalObjcId) -> Self {
        Self {
            id,
            class: MetalPhaseBufferClass::TransientExpert,
        }
    }

    pub(crate) fn borrowed_expert(id: MetalObjcId) -> Self {
        Self {
            id,
            class: MetalPhaseBufferClass::BorrowedExpert,
        }
    }
}

pub const METAL_SHADERS: &str = include_str!("qwen/shaders.metal");

#[cfg(test)]
#[path = "../tests/metal.rs"]
mod tests;
