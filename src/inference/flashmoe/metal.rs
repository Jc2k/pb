#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::ffi::c_void;
use std::time::Duration;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::scheduler::{ScheduledCmd3MetalPostAttentionInput, ScheduledRoutingCommand};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::state::FlashMoePostAttentionPrepState;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) type MetalObjcId = *mut c_void;

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

    #[cfg(test)]
    fn compiled_kernels(self) -> Vec<&'static str> {
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
    pub(crate) routing_command: Option<ScheduledRoutingCommand>,
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
        assert!(
            !without_route
                .compiled_kernels()
                .contains(&kernels::ROUTE_TOP4)
        );

        let with_route = MetalPipelineNameSet::new(true);
        assert_eq!(with_route.route_top4, Some(kernels::ROUTE_TOP4));
        assert!(with_route.compiled_kernels().contains(&kernels::ROUTE_TOP4));
    }

    #[test]
    fn pipeline_name_set_matches_declared_forward_kernel_surface() {
        let mut compiled = MetalPipelineNameSet::new(true).compiled_kernels();
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

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn post_attention_prep_builds_declared_cmd3_metal_input() {
        let prep = MetalPostAttentionPrep::new(
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
        assert!(prep.routing_command.is_none());
        assert_eq!(prep.input.state(), prep.state);
        assert!(prep.state.is_declared_graph_state());
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
