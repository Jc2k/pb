use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, de};

use super::types::{
    ACTIVE_EXPERTS_PER_TOKEN, DEFAULT_MROPE_SECTION, FOUR_BIT_EXPERT_SIZE, FULL_ATTN_INTERVAL,
    FlashMoeLayerKind, GLM52_MODEL_MARKER, GROUP_SIZE, LEGACY_QWEN_CODER_MARKER, NUM_EXPERTS,
    QWEN3_ACTIVE_PARAMS_MARKER, QWEN3_VL_MODEL_MARKER, QWEN35_MODEL_MARKER,
};
#[cfg(test)]
use super::types::{GLM52_COLIBRI_MODEL, GLM52_MODEL, GLM52_MXFP4_MODEL};
use super::vision::Qwen3VLVisionConfig;

pub const QWEN35_Q4_EXPERT_PACKED_WEIGHT_BYTES: usize = 2_097_152;
pub const QWEN35_Q4_EXPERT_SCALE_BYTES: usize = 131_072;
pub const QWEN35_Q4_EXPERT_BIAS_BYTES: usize = 131_072;

pub fn is_qwen35_or_legacy_alias(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase();
    normalized.contains(QWEN35_MODEL_MARKER) || normalized.contains(LEGACY_QWEN_CODER_MARKER)
}

pub fn is_glm52(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase();
    normalized.contains(GLM52_MODEL_MARKER)
        || normalized.contains("glm_5.2")
        || normalized.contains("glm5.2")
}

pub fn is_qwen3_vl(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase();
    normalized.contains(QWEN3_VL_MODEL_MARKER)
        && (normalized.contains("moe") || contains_active_parameter_marker(&normalized))
}

pub fn is_qwen3_moe(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase();
    normalized.contains("qwen3")
        && (normalized.contains("moe") || contains_active_parameter_marker(&normalized))
}

fn contains_active_parameter_marker(model: &str) -> bool {
    let mut remainder = model;
    while let Some(start) = remainder.find(QWEN3_ACTIVE_PARAMS_MARKER) {
        let rest = &remainder[start + QWEN3_ACTIVE_PARAMS_MARKER.len()..];
        let digits = rest.chars().take_while(|ch| ch.is_ascii_digit()).count();
        if digits > 0 && rest[digits..].starts_with('b') {
            return true;
        }
        remainder = rest;
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeExecutionArchitecture {
    UnifiedFlashMoe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeFamily {
    Qwen35A17B,
    Qwen3Moe,
    Qwen3VlMoe,
    Glm52,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeLayerKind {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenNormWeightSemantics {
    Multiplicative,
    Offset,
}

impl From<QwenMoeLayerKind> for FlashMoeLayerKind {
    fn from(value: QwenMoeLayerKind) -> Self {
        match value {
            QwenMoeLayerKind::FullAttention => Self::FullAttention,
            QwenMoeLayerKind::LinearAttention => Self::LinearAttention,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeRoutingPlacement {
    CpuSoftmaxTopK,
    CpuSigmoidNoAuxTopK,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeRoutingWeightNormalization {
    RenormalizeSelected,
    PreserveFullSoftmax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeExpertReadStrategy {
    ParallelPositionedReads,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeExpertCachePolicy {
    OsPageCacheOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeExpertBufferOwnership {
    SchedulerReusableWholeExpertSlots,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeCommandTopology {
    UpstreamCmd1Cmd2Cmd3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenMoeExecutionPolicy {
    pub architecture: QwenMoeExecutionArchitecture,
    pub routing: QwenMoeRoutingPlacement,
    pub expert_reads: QwenMoeExpertReadStrategy,
    pub expert_cache: QwenMoeExpertCachePolicy,
    pub expert_buffer_ownership: QwenMoeExpertBufferOwnership,
    pub command_topology: QwenMoeCommandTopology,
}

impl QwenMoeExecutionPolicy {
    pub const UPSTREAM_PARITY: Self = Self {
        architecture: QwenMoeExecutionArchitecture::UnifiedFlashMoe,
        routing: QwenMoeRoutingPlacement::CpuSoftmaxTopK,
        expert_reads: QwenMoeExpertReadStrategy::ParallelPositionedReads,
        expert_cache: QwenMoeExpertCachePolicy::OsPageCacheOnly,
        expert_buffer_ownership: QwenMoeExpertBufferOwnership::SchedulerReusableWholeExpertSlots,
        command_topology: QwenMoeCommandTopology::UpstreamCmd1Cmd2Cmd3,
    };

    pub const GLM52_PARITY: Self = Self {
        architecture: QwenMoeExecutionArchitecture::UnifiedFlashMoe,
        routing: QwenMoeRoutingPlacement::CpuSigmoidNoAuxTopK,
        expert_reads: QwenMoeExpertReadStrategy::ParallelPositionedReads,
        expert_cache: QwenMoeExpertCachePolicy::OsPageCacheOnly,
        expert_buffer_ownership: QwenMoeExpertBufferOwnership::SchedulerReusableWholeExpertSlots,
        command_topology: QwenMoeCommandTopology::UpstreamCmd1Cmd2Cmd3,
    };
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlmMoeConfig {
    pub first_k_dense_replace: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
    pub index_topk: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QwenModelConfig {
    pub model_type: Option<String>,
    pub architectures: Option<Vec<String>>,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    pub rope_theta: Option<f64>,
    pub partial_rotary_factor: Option<f64>,
    pub torch_dtype: Option<String>,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub norm_topk_prob: Option<bool>,
    pub moe_intermediate_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub mrope_section: Option<[usize; 3]>,
    pub tie_word_embeddings: Option<bool>,
    pub num_shared_experts: Option<usize>,
    pub shared_expert_intermediate_size: Option<usize>,
    pub vision_config: Option<Qwen3VLVisionConfig>,
    pub glm: Option<GlmMoeConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct RawQwenModelConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    num_hidden_layers: Option<usize>,
    hidden_size: Option<usize>,
    num_attention_heads: Option<usize>,
    head_dim: Option<usize>,
    num_key_value_heads: Option<usize>,
    vocab_size: Option<usize>,
    rope_theta: Option<f64>,
    partial_rotary_factor: Option<f64>,
    torch_dtype: Option<String>,
    dtype: Option<String>,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    norm_topk_prob: Option<bool>,
    moe_intermediate_size: Option<usize>,
    intermediate_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
    num_shared_experts: Option<usize>,
    shared_expert_intermediate_size: Option<usize>,
    vision_config: Option<Qwen3VLVisionConfig>,
    text_config: Option<RawQwenTextConfig>,
    rope_parameters: Option<RawQwenRopeParameters>,
    rope_scaling: Option<RawQwenRopeParameters>,
    n_routed_experts: Option<usize>,
    n_shared_experts: Option<usize>,
    first_k_dense_replace: Option<usize>,
    q_lora_rank: Option<usize>,
    kv_lora_rank: Option<usize>,
    qk_nope_head_dim: Option<usize>,
    qk_rope_head_dim: Option<usize>,
    v_head_dim: Option<usize>,
    n_group: Option<usize>,
    topk_group: Option<usize>,
    routed_scaling_factor: Option<f32>,
    rms_norm_eps: Option<f32>,
    index_topk: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct RawQwenTextConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    num_hidden_layers: Option<usize>,
    hidden_size: Option<usize>,
    num_attention_heads: Option<usize>,
    head_dim: Option<usize>,
    num_key_value_heads: Option<usize>,
    vocab_size: Option<usize>,
    rope_theta: Option<f64>,
    partial_rotary_factor: Option<f64>,
    torch_dtype: Option<String>,
    dtype: Option<String>,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    norm_topk_prob: Option<bool>,
    moe_intermediate_size: Option<usize>,
    intermediate_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
    num_shared_experts: Option<usize>,
    shared_expert_intermediate_size: Option<usize>,
    rope_parameters: Option<RawQwenRopeParameters>,
    rope_scaling: Option<RawQwenRopeParameters>,
}

#[derive(Debug, Default, Deserialize)]
struct RawQwenRopeParameters {
    rope_theta: Option<f64>,
    partial_rotary_factor: Option<f64>,
    mrope_section: Option<[usize; 3]>,
}

impl<'de> Deserialize<'de> for QwenModelConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let raw = RawQwenModelConfig::deserialize(deserializer)?;
        let text = raw.text_config.unwrap_or_default();
        let required = |field: &'static str,
                        top: Option<usize>,
                        nested: Option<usize>|
         -> std::result::Result<usize, D::Error> {
            top.or(nested)
                .ok_or_else(|| de::Error::missing_field(field))
        };

        Ok(Self {
            model_type: raw.model_type.or(text.model_type),
            architectures: raw.architectures.or(text.architectures),
            num_hidden_layers: required(
                "num_hidden_layers",
                raw.num_hidden_layers,
                text.num_hidden_layers,
            )?,
            hidden_size: required("hidden_size", raw.hidden_size, text.hidden_size)?,
            num_attention_heads: required(
                "num_attention_heads",
                raw.num_attention_heads,
                text.num_attention_heads,
            )?,
            head_dim: raw.head_dim.or(text.head_dim),
            num_key_value_heads: raw.num_key_value_heads.or(text.num_key_value_heads),
            vocab_size: required("vocab_size", raw.vocab_size, text.vocab_size)?,
            rope_theta: raw
                .rope_theta
                .or_else(|| {
                    raw.rope_parameters
                        .as_ref()
                        .and_then(|params| params.rope_theta)
                })
                .or_else(|| {
                    raw.rope_scaling
                        .as_ref()
                        .and_then(|params| params.rope_theta)
                })
                .or(text.rope_theta)
                .or_else(|| {
                    text.rope_parameters
                        .as_ref()
                        .and_then(|params| params.rope_theta)
                })
                .or_else(|| {
                    text.rope_scaling
                        .as_ref()
                        .and_then(|params| params.rope_theta)
                }),
            partial_rotary_factor: raw
                .partial_rotary_factor
                .or_else(|| {
                    raw.rope_parameters
                        .as_ref()
                        .and_then(|params| params.partial_rotary_factor)
                })
                .or_else(|| {
                    raw.rope_scaling
                        .as_ref()
                        .and_then(|params| params.partial_rotary_factor)
                })
                .or(text.partial_rotary_factor)
                .or_else(|| {
                    text.rope_parameters
                        .as_ref()
                        .and_then(|params| params.partial_rotary_factor)
                })
                .or_else(|| {
                    text.rope_scaling
                        .as_ref()
                        .and_then(|params| params.partial_rotary_factor)
                }),
            torch_dtype: raw
                .torch_dtype
                .or(raw.dtype)
                .or(text.torch_dtype)
                .or(text.dtype),
            num_experts: raw
                .num_experts
                .or(raw.n_routed_experts)
                .or(text.num_experts),
            num_experts_per_tok: raw.num_experts_per_tok.or(text.num_experts_per_tok),
            norm_topk_prob: raw.norm_topk_prob.or(text.norm_topk_prob),
            moe_intermediate_size: raw.moe_intermediate_size.or(text.moe_intermediate_size),
            intermediate_size: raw.intermediate_size.or(text.intermediate_size),
            max_position_embeddings: raw.max_position_embeddings.or(text.max_position_embeddings),
            mrope_section: raw
                .rope_parameters
                .as_ref()
                .and_then(|params| params.mrope_section)
                .or_else(|| {
                    raw.rope_scaling
                        .as_ref()
                        .and_then(|params| params.mrope_section)
                })
                .or_else(|| {
                    text.rope_parameters
                        .as_ref()
                        .and_then(|params| params.mrope_section)
                })
                .or_else(|| {
                    text.rope_scaling
                        .as_ref()
                        .and_then(|params| params.mrope_section)
                }),
            tie_word_embeddings: raw.tie_word_embeddings.or(text.tie_word_embeddings),
            num_shared_experts: raw
                .num_shared_experts
                .or(raw.n_shared_experts)
                .or(text.num_shared_experts),
            shared_expert_intermediate_size: raw
                .shared_expert_intermediate_size
                .or(text.shared_expert_intermediate_size),
            vision_config: raw.vision_config,
            glm: raw.q_lora_rank.map(|q_lora_rank| GlmMoeConfig {
                first_k_dense_replace: raw.first_k_dense_replace.unwrap_or(0),
                q_lora_rank,
                kv_lora_rank: raw.kv_lora_rank.unwrap_or(0),
                qk_nope_head_dim: raw.qk_nope_head_dim.unwrap_or(0),
                qk_rope_head_dim: raw.qk_rope_head_dim.unwrap_or(0),
                v_head_dim: raw.v_head_dim.unwrap_or(0),
                n_group: raw.n_group.unwrap_or(0),
                topk_group: raw.topk_group.unwrap_or(0),
                routed_scaling_factor: raw.routed_scaling_factor.unwrap_or(1.0),
                rms_norm_eps: raw.rms_norm_eps.unwrap_or(1e-5),
                index_topk: raw.index_topk.unwrap_or(0),
            }),
        })
    }
}

impl QwenModelConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read model config {}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse model config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let model_type = self
            .model_type
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !(model_type.contains("qwen")
            || model_type.contains("glm")
            || self.architectures.as_ref().is_some_and(|items| {
                items.iter().any(|item| {
                    let item = item.to_ascii_lowercase();
                    item.contains("qwen") || item.contains("glm")
                })
            }))
        {
            bail!(
                "Flash-MoE only supports declared Qwen/GLM MoE configs, found model_type={:?} architectures={:?}",
                self.model_type,
                self.architectures
            );
        }
        if self.num_hidden_layers == 0
            || self.hidden_size == 0
            || self.num_attention_heads == 0
            || self.vocab_size == 0
        {
            bail!("Qwen config contains zero-valued required dimensions");
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            bail!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                self.hidden_size,
                self.num_attention_heads
            );
        }
        if let Some(head_dim) = self.head_dim
            && head_dim == 0
        {
            bail!("head_dim must be positive when present");
        }
        if let Some(kv_heads) = self.num_key_value_heads
            && (kv_heads == 0 || !self.num_attention_heads.is_multiple_of(kv_heads))
        {
            bail!(
                "num_key_value_heads {kv_heads} must divide num_attention_heads {}",
                self.num_attention_heads
            );
        }
        let experts = self.experts();
        let active = self.config_active_experts();
        if experts == 0 || active == 0 || active > experts {
            bail!(
                "invalid MoE routing config: num_experts={experts}, num_experts_per_tok={active}"
            );
        }
        if let Some(theta) = self.rope_theta
            && (!theta.is_finite() || theta <= 0.0)
        {
            bail!("rope_theta must be positive and finite, got {theta}");
        }
        if let Some(factor) = self.partial_rotary_factor
            && (!factor.is_finite() || factor <= 0.0 || factor > 1.0)
        {
            bail!("partial_rotary_factor must be in (0, 1], got {factor}");
        }
        if let Some(section) = self.mrope_section
            && section.contains(&0)
        {
            bail!("mrope_section entries must be positive, got {section:?}");
        }
        if let Some(vision) = &self.vision_config {
            if vision.depth == 0
                || vision.embed_dim == 0
                || vision.num_heads == 0
                || vision.patch_size == 0
                || vision.merge_size == 0
                || vision.temporal_patch_size == 0
                || vision.in_chans == 0
            {
                bail!("Qwen3-VL vision_config contains zero-valued required dimensions");
            }
            if vision.embed_dim % vision.num_heads != 0 {
                bail!(
                    "vision hidden_size {} is not divisible by num_heads {}",
                    vision.embed_dim,
                    vision.num_heads
                );
            }
            if let Some(idx) = vision
                .deepstack_visual_indexes
                .iter()
                .copied()
                .find(|idx| *idx >= vision.depth)
            {
                bail!(
                    "vision deepstack_visual_indexes contains {idx}, but depth is {}",
                    vision.depth
                );
            }
        }
        if let Some(glm) = &self.glm {
            if glm.first_k_dense_replace > self.num_hidden_layers
                || glm.q_lora_rank == 0
                || glm.kv_lora_rank == 0
                || glm.qk_nope_head_dim == 0
                || glm.qk_rope_head_dim == 0
                || !glm.qk_rope_head_dim.is_multiple_of(2)
                || glm.v_head_dim == 0
                || glm.n_group != 1
                || glm.topk_group != 1
            {
                bail!(
                    "invalid GLM MLA/MoE config: first_dense={}, q_lora={}, kv_lora={}, qk_nope={}, qk_rope={}, v_head={}, n_group={}, topk_group={}",
                    glm.first_k_dense_replace,
                    glm.q_lora_rank,
                    glm.kv_lora_rank,
                    glm.qk_nope_head_dim,
                    glm.qk_rope_head_dim,
                    glm.v_head_dim,
                    glm.n_group,
                    glm.topk_group
                );
            }
            if !glm.routed_scaling_factor.is_finite() || glm.routed_scaling_factor <= 0.0 {
                bail!(
                    "GLM routed_scaling_factor must be positive and finite, got {}",
                    glm.routed_scaling_factor
                );
            }
            if !glm.rms_norm_eps.is_finite() || glm.rms_norm_eps <= 0.0 {
                bail!(
                    "GLM rms_norm_eps must be positive and finite, got {}",
                    glm.rms_norm_eps
                );
            }
        }
        if let Some(dtype) = &self.torch_dtype {
            let dtype = dtype.to_ascii_lowercase();
            if !matches!(
                dtype.as_str(),
                "bfloat16" | "float16" | "float32" | "bf16" | "fp16" | "fp32"
            ) {
                bail!("unsupported Qwen dtype {dtype}; expected bf16/fp16/fp32 compatible weights");
            }
        }
        Ok(())
    }

    pub(crate) fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub(crate) fn derived_attention_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads.max(1)
    }

    pub(crate) fn full_attention_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or_else(|| self.derived_attention_head_dim())
    }

    pub(crate) fn experts(&self) -> usize {
        self.num_experts.unwrap_or(NUM_EXPERTS)
    }

    pub(crate) fn config_active_experts(&self) -> usize {
        self.num_experts_per_tok.unwrap_or(ACTIVE_EXPERTS_PER_TOKEN)
    }

    pub(crate) fn shared_experts(&self) -> usize {
        self.num_shared_experts
            .unwrap_or_else(|| usize::from(self.shared_expert_intermediate_size.unwrap_or(0) > 0))
    }

    pub(crate) fn norm_weight_semantics(&self) -> QwenNormWeightSemantics {
        if self
            .model_type
            .as_deref()
            .is_some_and(|model_type| model_type.contains("qwen3_next"))
            || self.architectures.as_ref().is_some_and(|architectures| {
                architectures
                    .iter()
                    .any(|architecture| architecture.contains("Qwen3Next"))
            })
        {
            QwenNormWeightSemantics::Offset
        } else {
            QwenNormWeightSemantics::Multiplicative
        }
    }

    pub(super) fn linear_attention_qkv_projection_requires_reorder(&self) -> bool {
        let is_qwen35 = self
            .model_type
            .as_deref()
            .is_some_and(|model_type| model_type.contains("qwen3_5"))
            || self.architectures.as_ref().is_some_and(|architectures| {
                architectures
                    .iter()
                    .any(|architecture| architecture.contains("Qwen3_5"))
            });
        !is_qwen35
    }

    pub(crate) fn text_mrope_section(&self) -> Option<[usize; 3]> {
        self.mrope_section
            .or_else(|| self.vision_config.as_ref().map(|_| DEFAULT_MROPE_SECTION))
    }

    pub(crate) fn shared_expert_intermediate_size(&self) -> usize {
        self.shared_expert_intermediate_size
            .or(self.moe_intermediate_size)
            .or(self.intermediate_size)
            .unwrap_or(0)
    }

    pub(crate) fn first_sparse_layer(&self) -> usize {
        self.glm.as_ref().map_or(0, |glm| glm.first_k_dense_replace)
    }

    pub(crate) fn is_dense_mlp_layer(&self, layer: usize) -> bool {
        layer < self.first_sparse_layer()
    }

    pub(crate) fn rms_norm_epsilon(&self) -> f32 {
        self.glm.as_ref().map_or(1e-6, |glm| glm.rms_norm_eps)
    }

    pub(crate) fn glm_mla_norm_epsilon(&self) -> Option<f32> {
        self.glm.as_ref().map(|_| 1e-6)
    }

    /// Derives GLM's logical matrix shape from the architecture. Source
    /// adapters use this when a packed checkpoint flattens or compresses the
    /// safetensors shape.
    pub(crate) fn glm_logical_tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        let glm = self.glm.as_ref()?;
        if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
            return Some(vec![self.vocab_size, self.hidden_size]);
        }
        let parts = name.split('.').collect::<Vec<_>>();
        let layer = parts
            .windows(2)
            .find(|part| part[0] == "layers")?
            .get(1)?
            .parse::<usize>()
            .ok()?;
        if layer >= self.num_hidden_layers {
            return None;
        }
        if name.ends_with("mlp.switch_mlp.gate_proj.weight")
            || name.ends_with("mlp.switch_mlp.up_proj.weight")
        {
            return Some(vec![
                self.experts(),
                self.moe_intermediate_size?,
                self.hidden_size,
            ]);
        }
        if name.ends_with("mlp.switch_mlp.down_proj.weight") {
            return Some(vec![
                self.experts(),
                self.hidden_size,
                self.moe_intermediate_size?,
            ]);
        }
        let qk_head = glm.qk_nope_head_dim.checked_add(glm.qk_rope_head_dim)?;
        let attention_rows = |per_head: usize| self.num_attention_heads.checked_mul(per_head);
        let shape = if name.ends_with("self_attn.q_a_proj.weight") {
            [glm.q_lora_rank, self.hidden_size]
        } else if name.ends_with("self_attn.q_b_proj.weight") {
            [attention_rows(qk_head)?, glm.q_lora_rank]
        } else if name.ends_with("self_attn.kv_a_proj_with_mqa.weight") {
            [
                glm.kv_lora_rank.checked_add(glm.qk_rope_head_dim)?,
                self.hidden_size,
            ]
        } else if name.ends_with("self_attn.kv_b_proj.weight") {
            [
                attention_rows(glm.qk_nope_head_dim.checked_add(glm.v_head_dim)?)?,
                glm.kv_lora_rank,
            ]
        } else if name.ends_with("self_attn.embed_q.weight") {
            return Some(vec![
                self.num_attention_heads,
                glm.kv_lora_rank,
                glm.qk_nope_head_dim,
            ]);
        } else if name.ends_with("self_attn.unembed_out.weight") {
            return Some(vec![
                self.num_attention_heads,
                glm.v_head_dim,
                glm.kv_lora_rank,
            ]);
        } else if name.ends_with("self_attn.o_proj.weight") {
            [self.hidden_size, attention_rows(glm.v_head_dim)?]
        } else if name.ends_with("mlp.gate_proj.weight") || name.ends_with("mlp.up_proj.weight") {
            [self.intermediate_size?, self.hidden_size]
        } else if name.ends_with("mlp.down_proj.weight") {
            [self.hidden_size, self.intermediate_size?]
        } else if (name.ends_with("mlp.shared_experts.gate_proj.weight")
            || name.ends_with("mlp.shared_expert.gate_proj.weight"))
            || (name.ends_with("mlp.shared_experts.up_proj.weight")
                || name.ends_with("mlp.shared_expert.up_proj.weight"))
        {
            [
                self.moe_intermediate_size?
                    .checked_mul(self.shared_experts())?,
                self.hidden_size,
            ]
        } else if name.ends_with("mlp.shared_experts.down_proj.weight")
            || name.ends_with("mlp.shared_expert.down_proj.weight")
        {
            [
                self.hidden_size,
                self.moe_intermediate_size?
                    .checked_mul(self.shared_experts())?,
            ]
        } else if name.contains(".mlp.experts.")
            && (name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight"))
        {
            [self.moe_intermediate_size?, self.hidden_size]
        } else if name.contains(".mlp.experts.") && name.ends_with("down_proj.weight") {
            [self.hidden_size, self.moe_intermediate_size?]
        } else {
            return None;
        };
        Some(shape.to_vec())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeExpertComponentKind {
    GateWeight,
    GateScale,
    GateBias,
    UpWeight,
    UpScale,
    UpBias,
    DownWeight,
    DownScale,
    DownBias,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenMoeExpertComponentLayout {
    pub kind: QwenMoeExpertComponentKind,
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenMoeQ4ExpertLayout {
    pub expert_bytes: usize,
    pub group_size: usize,
    pub components: [QwenMoeExpertComponentLayout; 9],
}

impl QwenMoeQ4ExpertLayout {
    pub fn fixed_bf16(
        hidden_size: usize,
        intermediate_size: usize,
        group_size: usize,
    ) -> Result<Self> {
        if hidden_size == 0 || intermediate_size == 0 || group_size == 0 {
            bail!(
                "fixed-Q4 expert layout requires non-zero hidden, intermediate, and group dimensions"
            );
        }
        if !hidden_size.is_multiple_of(group_size) || !intermediate_size.is_multiple_of(group_size)
        {
            bail!(
                "fixed-Q4 expert dimensions hidden={hidden_size} intermediate={intermediate_size} must be divisible by group_size={group_size}"
            );
        }

        fn projection_bytes(rows: usize, cols: usize, group_size: usize) -> Result<(usize, usize)> {
            let values = rows
                .checked_mul(cols)
                .context("fixed-Q4 projection element count overflow")?;
            if !values.is_multiple_of(2) {
                bail!("fixed-Q4 projection element count {values} must be even");
            }
            let packed = values / 2;
            let scale_bias = rows
                .checked_mul(cols / group_size)
                .and_then(|groups| groups.checked_mul(2))
                .context("fixed-Q4 BF16 scale/bias byte count overflow")?;
            Ok((packed, scale_bias))
        }

        let (gate_weight_bytes, gate_scale_bias_bytes) =
            projection_bytes(intermediate_size, hidden_size, group_size)?;
        let (down_weight_bytes, down_scale_bias_bytes) =
            projection_bytes(hidden_size, intermediate_size, group_size)?;
        let gate_scale_offset = gate_weight_bytes;
        let gate_bias_offset = gate_scale_offset
            .checked_add(gate_scale_bias_bytes)
            .context("fixed-Q4 gate bias offset overflow")?;
        let up_weight_offset = gate_bias_offset
            .checked_add(gate_scale_bias_bytes)
            .context("fixed-Q4 up weight offset overflow")?;
        let up_scale_offset = up_weight_offset
            .checked_add(gate_weight_bytes)
            .context("fixed-Q4 up scale offset overflow")?;
        let up_bias_offset = up_scale_offset
            .checked_add(gate_scale_bias_bytes)
            .context("fixed-Q4 up bias offset overflow")?;
        let down_weight_offset = up_bias_offset
            .checked_add(gate_scale_bias_bytes)
            .context("fixed-Q4 down weight offset overflow")?;
        let down_scale_offset = down_weight_offset
            .checked_add(down_weight_bytes)
            .context("fixed-Q4 down scale offset overflow")?;
        let down_bias_offset = down_scale_offset
            .checked_add(down_scale_bias_bytes)
            .context("fixed-Q4 down bias offset overflow")?;
        let expert_bytes = down_bias_offset
            .checked_add(down_scale_bias_bytes)
            .context("fixed-Q4 expert byte count overflow")?;
        let layout = Self {
            expert_bytes,
            group_size,
            components: [
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::GateWeight,
                    offset: 0,
                    bytes: gate_weight_bytes,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::GateScale,
                    offset: gate_scale_offset,
                    bytes: gate_scale_bias_bytes,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::GateBias,
                    offset: gate_bias_offset,
                    bytes: gate_scale_bias_bytes,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::UpWeight,
                    offset: up_weight_offset,
                    bytes: gate_weight_bytes,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::UpScale,
                    offset: up_scale_offset,
                    bytes: gate_scale_bias_bytes,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::UpBias,
                    offset: up_bias_offset,
                    bytes: gate_scale_bias_bytes,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::DownWeight,
                    offset: down_weight_offset,
                    bytes: down_weight_bytes,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::DownScale,
                    offset: down_scale_offset,
                    bytes: down_scale_bias_bytes,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::DownBias,
                    offset: down_bias_offset,
                    bytes: down_scale_bias_bytes,
                },
            ],
        };
        layout.validate()?;
        Ok(layout)
    }

    pub const fn qwen35_a17b() -> Self {
        Self {
            expert_bytes: FOUR_BIT_EXPERT_SIZE as usize,
            group_size: GROUP_SIZE,
            components: [
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::GateWeight,
                    offset: 0,
                    bytes: QWEN35_Q4_EXPERT_PACKED_WEIGHT_BYTES,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::GateScale,
                    offset: 2_097_152,
                    bytes: QWEN35_Q4_EXPERT_SCALE_BYTES,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::GateBias,
                    offset: 2_228_224,
                    bytes: QWEN35_Q4_EXPERT_BIAS_BYTES,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::UpWeight,
                    offset: 2_359_296,
                    bytes: QWEN35_Q4_EXPERT_PACKED_WEIGHT_BYTES,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::UpScale,
                    offset: 4_456_448,
                    bytes: QWEN35_Q4_EXPERT_SCALE_BYTES,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::UpBias,
                    offset: 4_587_520,
                    bytes: QWEN35_Q4_EXPERT_BIAS_BYTES,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::DownWeight,
                    offset: 4_718_592,
                    bytes: QWEN35_Q4_EXPERT_PACKED_WEIGHT_BYTES,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::DownScale,
                    offset: 6_815_744,
                    bytes: QWEN35_Q4_EXPERT_SCALE_BYTES,
                },
                QwenMoeExpertComponentLayout {
                    kind: QwenMoeExpertComponentKind::DownBias,
                    offset: 6_946_816,
                    bytes: QWEN35_Q4_EXPERT_BIAS_BYTES,
                },
            ],
        }
    }

    pub fn component(&self, kind: QwenMoeExpertComponentKind) -> QwenMoeExpertComponentLayout {
        self.components
            .iter()
            .copied()
            .find(|component| component.kind == kind)
            .expect("Qwen MoE expert layout has every component")
    }

    pub fn validate(&self) -> Result<()> {
        let mut expected_offset = 0;
        for component in self.components {
            if component.offset != expected_offset {
                bail!(
                    "Qwen MoE expert component {:?} starts at {}, expected {}",
                    component.kind,
                    component.offset,
                    expected_offset
                );
            }
            expected_offset += component.bytes;
        }
        if expected_offset != self.expert_bytes {
            bail!(
                "Qwen MoE expert layout totals {expected_offset} bytes, expected {}",
                self.expert_bytes
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QwenMoeModelLayout {
    pub family: QwenMoeFamily,
    pub execution: QwenMoeExecutionPolicy,
    pub layers: usize,
    pub first_sparse_layer: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub experts_per_layer: usize,
    pub configured_active_experts: usize,
    pub scheduled_active_experts: usize,
    pub routed_expert_scale: f32,
    pub routing_weight_normalization: Option<QwenMoeRoutingWeightNormalization>,
    pub moe_intermediate_size: usize,
    pub shared_experts: usize,
    pub shared_expert_intermediate_size: usize,
    pub rope_theta: Option<f64>,
    pub partial_rotary_factor: Option<f64>,
    pub mrope_section: Option<[usize; 3]>,
    pub has_vision: bool,
}

impl QwenMoeModelLayout {
    pub fn from_config(model: &str, config: &QwenModelConfig) -> Result<Self> {
        config.validate()?;
        let family = QwenMoeFamily::from_model_and_config(model, config)?;
        let head_dim = config.full_attention_head_dim();
        let configured_active_experts = config.config_active_experts();
        let scheduled_active_experts = match family {
            QwenMoeFamily::Qwen35A17B if config.experts() >= ACTIVE_EXPERTS_PER_TOKEN => {
                ACTIVE_EXPERTS_PER_TOKEN
            }
            QwenMoeFamily::Qwen35A17B => configured_active_experts,
            QwenMoeFamily::Qwen3Moe | QwenMoeFamily::Qwen3VlMoe | QwenMoeFamily::Glm52 => {
                configured_active_experts
            }
        };
        let routed_expert_scale = config
            .glm
            .as_ref()
            .map_or(1.0, |glm| glm.routed_scaling_factor);
        let routing_weight_normalization = match family {
            QwenMoeFamily::Qwen35A17B => {
                Some(QwenMoeRoutingWeightNormalization::RenormalizeSelected)
            }
            QwenMoeFamily::Qwen3Moe | QwenMoeFamily::Qwen3VlMoe => {
                config.norm_topk_prob.map(|normalize| {
                    if normalize {
                        QwenMoeRoutingWeightNormalization::RenormalizeSelected
                    } else {
                        QwenMoeRoutingWeightNormalization::PreserveFullSoftmax
                    }
                })
            }
            QwenMoeFamily::Glm52 => Some(QwenMoeRoutingWeightNormalization::RenormalizeSelected),
        };
        let layout = Self {
            family,
            execution: if family == QwenMoeFamily::Glm52 {
                QwenMoeExecutionPolicy::GLM52_PARITY
            } else {
                QwenMoeExecutionPolicy::UPSTREAM_PARITY
            },
            layers: config.num_hidden_layers,
            first_sparse_layer: config.first_sparse_layer(),
            hidden_size: config.hidden_size,
            attention_heads: config.num_attention_heads,
            kv_heads: config.kv_heads(),
            head_dim,
            vocab_size: config.vocab_size,
            experts_per_layer: config.experts(),
            configured_active_experts,
            scheduled_active_experts,
            routed_expert_scale,
            routing_weight_normalization,
            moe_intermediate_size: config
                .moe_intermediate_size
                .or(config.intermediate_size)
                .unwrap_or(0),
            shared_experts: config.shared_experts(),
            shared_expert_intermediate_size: config.shared_expert_intermediate_size(),
            rope_theta: config.rope_theta,
            partial_rotary_factor: config.partial_rotary_factor,
            mrope_section: config.text_mrope_section(),
            has_vision: config.vision_config.is_some(),
        };
        layout.validate()?;
        Ok(layout)
    }

    pub fn layer_kind(&self, layer: usize) -> QwenMoeLayerKind {
        match self.family {
            QwenMoeFamily::Qwen35A17B => {
                if (layer + 1).is_multiple_of(FULL_ATTN_INTERVAL) {
                    QwenMoeLayerKind::FullAttention
                } else {
                    QwenMoeLayerKind::LinearAttention
                }
            }
            QwenMoeFamily::Qwen3Moe | QwenMoeFamily::Qwen3VlMoe => QwenMoeLayerKind::FullAttention,
            QwenMoeFamily::Glm52 => QwenMoeLayerKind::FullAttention,
        }
    }

    pub fn with_scheduled_active_experts(mut self, active_experts: usize) -> Result<Self> {
        self.scheduled_active_experts = active_experts;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.layers == 0
            || self.hidden_size == 0
            || self.attention_heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.vocab_size == 0
        {
            bail!("Qwen MoE layout contains zero-valued required dimensions");
        }
        if self.first_sparse_layer >= self.layers {
            bail!(
                "MoE layout first sparse layer {} must be within {} model layers",
                self.first_sparse_layer,
                self.layers
            );
        }
        if self.experts_per_layer == 0
            || self.configured_active_experts == 0
            || self.scheduled_active_experts == 0
            || self.scheduled_active_experts > self.experts_per_layer
            || !self.routed_expert_scale.is_finite()
            || self.routed_expert_scale <= 0.0
        {
            bail!(
                "invalid Qwen MoE expert schedule: experts={}, configured_k={}, scheduled_k={}, routed_scale={}",
                self.experts_per_layer,
                self.configured_active_experts,
                self.scheduled_active_experts,
                self.routed_expert_scale
            );
        }
        Ok(())
    }
}

impl QwenMoeFamily {
    pub fn from_model_and_config(model: &str, config: &QwenModelConfig) -> Result<Self> {
        if is_glm52(model) || config_is_glm52(config) {
            return Ok(Self::Glm52);
        }
        if is_qwen35_or_legacy_alias(model) || config_is_qwen35(config) {
            return Ok(Self::Qwen35A17B);
        }
        if is_qwen3_vl(model) || config.vision_config.is_some() || config_is_qwen_vl(config) {
            return Ok(Self::Qwen3VlMoe);
        }
        if is_qwen3_moe(model) || config_is_qwen_moe(config) {
            return Ok(Self::Qwen3Moe);
        }
        bail!(
            "FlashMoe only supports declared Qwen/GLM MoE models, found model={model:?} model_type={:?} architectures={:?}",
            config.model_type,
            config.architectures
        );
    }
}

fn config_is_glm52(config: &QwenModelConfig) -> bool {
    config.glm.is_some()
        && (config_field_contains(config.model_type.as_deref(), "glm_moe_dsa")
            || config_architecture_contains(config, "GlmMoeDsa"))
}

fn config_is_qwen35(config: &QwenModelConfig) -> bool {
    config_field_contains(config.model_type.as_deref(), "qwen3_5")
        || config_architecture_contains(config, "Qwen3_5")
        || config_architecture_contains(config, "Qwen3Next")
}

fn config_is_qwen_vl(config: &QwenModelConfig) -> bool {
    config_field_contains(config.model_type.as_deref(), "qwen3_vl")
        || config_architecture_contains(config, "Qwen3VLMoe")
}

fn config_is_qwen_moe(config: &QwenModelConfig) -> bool {
    config_field_contains(config.model_type.as_deref(), "qwen")
        && (config_field_contains(config.model_type.as_deref(), "moe")
            || config.architectures.as_ref().is_some_and(|architectures| {
                architectures.iter().any(|architecture| {
                    architecture.to_ascii_lowercase().contains("moe")
                        || architecture.to_ascii_lowercase().contains("qwen")
                })
            }))
}

fn config_field_contains(field: Option<&str>, needle: &str) -> bool {
    field.is_some_and(|value| {
        value
            .to_ascii_lowercase()
            .contains(&needle.to_ascii_lowercase())
    })
}

fn config_architecture_contains(config: &QwenModelConfig, needle: &str) -> bool {
    let needle = needle.to_ascii_lowercase();
    config.architectures.as_ref().is_some_and(|architectures| {
        architectures
            .iter()
            .any(|architecture| architecture.to_ascii_lowercase().contains(&needle))
    })
}

pub fn default_qwen_vl_mrope_section(config: &QwenModelConfig) -> Option<[usize; 3]> {
    config
        .mrope_section
        .or_else(|| config.vision_config.as_ref().map(|_| DEFAULT_MROPE_SECTION))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::flashmoe::types::{
        HIDDEN_DIM, NUM_EXPERTS, NUM_LAYERS, QWEN3_VL_MODEL, QWEN35_MODEL,
    };

    fn config(json: &[u8]) -> QwenModelConfig {
        serde_json::from_slice(json).unwrap()
    }

    fn qwen35_config() -> QwenModelConfig {
        config(
            br#"{
  "model_type": "qwen3_5_moe",
  "architectures": ["Qwen3_5MoeForCausalLM"],
  "num_hidden_layers": 60,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "head_dim": 256,
  "num_key_value_heads": 2,
  "vocab_size": 248320,
  "rope_theta": 10000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "moe_intermediate_size": 1024,
  "num_shared_experts": 1,
  "shared_expert_intermediate_size": 1024
}"#,
        )
    }

    #[test]
    fn qwen_model_config_resolves_nested_text_and_rope_metadata() {
        let config = config(
            br#"{
  "model_type": "qwen3_vl_moe",
  "text_config": {
    "architectures": ["Qwen3MoeForCausalLM"],
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "vocab_size": 128,
    "torch_dtype": "bfloat16",
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "norm_topk_prob": true,
    "moe_intermediate_size": 64,
    "rope_parameters": {
      "rope_theta": 1000000.0,
      "mrope_section": [24, 20, 20]
    }
  },
  "vision_config": {
    "depth": 1,
    "hidden_size": 64,
    "num_heads": 4,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "out_hidden_size": 64
  }
}"#,
        );

        config.validate().unwrap();
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.kv_heads(), 2);
        assert_eq!(config.rope_theta, Some(1_000_000.0));
        assert_eq!(config.text_mrope_section(), Some([24, 20, 20]));
        assert!(config.vision_config.is_some());
    }

    #[test]
    fn qwen_model_config_rejects_unsupported_dtype_at_family_boundary() {
        let config = config(
            br#"{
  "model_type": "qwen3_moe",
  "num_hidden_layers": 1,
  "hidden_size": 64,
  "num_attention_heads": 8,
  "vocab_size": 128,
  "torch_dtype": "int8",
  "num_experts": 2,
  "num_experts_per_tok": 1
}"#,
        );

        let error = config.validate().unwrap_err();
        assert!(error.to_string().contains("unsupported Qwen dtype int8"));
    }

    #[test]
    fn qwen_model_config_exposes_validated_runtime_dimensions() {
        let config = config(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        );

        config.validate().unwrap();
        assert_eq!(config.kv_heads(), 8);
        assert_eq!(config.experts(), 512);
        assert_eq!(config.config_active_experts(), 10);
    }

    #[test]
    fn norm_weight_semantics_are_resolved_from_the_concrete_model_config() {
        assert_eq!(
            qwen35_config().norm_weight_semantics(),
            QwenNormWeightSemantics::Multiplicative
        );
        let qwen3_next = config(
            br#"{"model_type":"qwen3_next_moe","architectures":["Qwen3NextForCausalLM"],"num_hidden_layers":1,"hidden_size":64,"num_attention_heads":8,"vocab_size":128,"num_experts":2,"num_experts_per_tok":1}"#,
        );
        assert_eq!(
            qwen3_next.norm_weight_semantics(),
            QwenNormWeightSemantics::Offset
        );
    }

    #[test]
    fn qwen_family_name_detection_is_owned_by_model_family() {
        assert!(is_qwen35_or_legacy_alias("hf://Qwen/Qwen3.5-397B-A17B"));
        assert!(is_qwen3_moe("hf://Qwen/Qwen3-30B-A3B"));
        assert!(is_qwen3_vl(QWEN3_VL_MODEL));
        assert!(is_qwen3_vl("hf://Qwen/Qwen3-VL-30B-A3B-Instruct"));
        assert!(is_qwen3_vl("hf://Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"));
        assert!(is_qwen3_vl("hf://Qwen/Qwen3-VL-MoE-Instruct"));
        assert!(!is_qwen3_vl("hf://Qwen/Qwen3-VL-8B-Instruct"));
        assert!(!is_qwen3_moe("qwen3-dense-8b"));
    }

    #[test]
    fn qwen35_layout_matches_upstream_flash_moe_profile() {
        let layout = QwenMoeModelLayout::from_config(QWEN35_MODEL, &qwen35_config()).unwrap();

        assert_eq!(layout.family, QwenMoeFamily::Qwen35A17B);
        assert_eq!(
            layout.execution,
            QwenMoeExecutionPolicy {
                architecture: QwenMoeExecutionArchitecture::UnifiedFlashMoe,
                routing: QwenMoeRoutingPlacement::CpuSoftmaxTopK,
                expert_reads: QwenMoeExpertReadStrategy::ParallelPositionedReads,
                expert_cache: QwenMoeExpertCachePolicy::OsPageCacheOnly,
                expert_buffer_ownership:
                    QwenMoeExpertBufferOwnership::SchedulerReusableWholeExpertSlots,
                command_topology: QwenMoeCommandTopology::UpstreamCmd1Cmd2Cmd3,
            }
        );
        assert_eq!(layout.layers, 60);
        assert_eq!(layout.hidden_size, 4096);
        assert_eq!(layout.attention_heads, 32);
        assert_eq!(layout.head_dim, 256);
        assert_eq!(layout.kv_heads, 2);
        assert_eq!(layout.configured_active_experts, 10);
        assert_eq!(layout.scheduled_active_experts, 4);
        assert_eq!(layout.routed_expert_scale, 1.0);
        assert_eq!(
            layout.routing_weight_normalization,
            Some(QwenMoeRoutingWeightNormalization::RenormalizeSelected)
        );
        assert_eq!(layout.experts_per_layer, 512);
        assert_eq!(layout.layers, NUM_LAYERS);
        assert_eq!(layout.hidden_size, HIDDEN_DIM);
        assert_eq!(layout.experts_per_layer, NUM_EXPERTS);
        assert_eq!(layout.moe_intermediate_size, 1024);
        assert_eq!(layout.shared_experts, 1);
        assert_eq!(layout.shared_expert_intermediate_size, 1024);
        assert!(!layout.has_vision);
    }

    #[test]
    fn qwen35_layout_allows_tiny_synthetic_fixtures() {
        let config = config(
            br#"{
  "model_type": "qwen3_5_moe",
  "architectures": ["Qwen3_5MoeForCausalLM"],
  "num_hidden_layers": 1,
  "hidden_size": 8,
  "num_attention_heads": 2,
  "num_key_value_heads": 1,
  "vocab_size": 32,
  "rope_theta": 10000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 1,
  "num_experts_per_tok": 1,
  "moe_intermediate_size": 4
}"#,
        );
        let layout = QwenMoeModelLayout::from_config(QWEN35_MODEL, &config).unwrap();

        assert_eq!(layout.family, QwenMoeFamily::Qwen35A17B);
        assert_eq!(layout.scheduled_active_experts, 1);
        assert_eq!(layout.experts_per_layer, 1);
    }

    #[test]
    fn qwen35_attention_schedule_is_linear_except_every_fourth_layer() {
        let layout = QwenMoeModelLayout::from_config(QWEN35_MODEL, &qwen35_config()).unwrap();

        assert_eq!(layout.layer_kind(0), QwenMoeLayerKind::LinearAttention);
        assert_eq!(layout.layer_kind(1), QwenMoeLayerKind::LinearAttention);
        assert_eq!(layout.layer_kind(2), QwenMoeLayerKind::LinearAttention);
        assert_eq!(layout.layer_kind(3), QwenMoeLayerKind::FullAttention);
        assert_eq!(layout.layer_kind(7), QwenMoeLayerKind::FullAttention);
    }

    #[test]
    fn qwen35_q4_expert_offsets_match_upstream_repack_layout() {
        let expert_layout = QwenMoeQ4ExpertLayout::qwen35_a17b();
        assert_eq!(
            QwenMoeQ4ExpertLayout::fixed_bf16(4096, 1024, GROUP_SIZE).unwrap(),
            expert_layout
        );

        assert_eq!(expert_layout.expert_bytes, 7_077_888);
        assert_eq!(
            expert_layout.component(QwenMoeExpertComponentKind::GateWeight),
            QwenMoeExpertComponentLayout {
                kind: QwenMoeExpertComponentKind::GateWeight,
                offset: 0,
                bytes: 2_097_152,
            }
        );
        assert_eq!(
            expert_layout.component(QwenMoeExpertComponentKind::UpWeight),
            QwenMoeExpertComponentLayout {
                kind: QwenMoeExpertComponentKind::UpWeight,
                offset: 2_359_296,
                bytes: 2_097_152,
            }
        );
        assert_eq!(
            expert_layout.component(QwenMoeExpertComponentKind::DownBias),
            QwenMoeExpertComponentLayout {
                kind: QwenMoeExpertComponentKind::DownBias,
                offset: 6_946_816,
                bytes: 131_072,
            }
        );
        expert_layout.validate().unwrap();
    }

    #[test]
    fn qwen3_moe_layout_uses_same_execution_policy_with_model_config_k() {
        let config = config(
            br#"{
  "model_type": "qwen3_moe",
  "architectures": ["Qwen3MoeForCausalLM"],
  "num_hidden_layers": 48,
  "hidden_size": 2048,
  "num_attention_heads": 32,
  "head_dim": 128,
  "num_key_value_heads": 4,
  "vocab_size": 151936,
  "rope_theta": 1000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "norm_topk_prob": true,
  "moe_intermediate_size": 768
}"#,
        );
        let layout = QwenMoeModelLayout::from_config("hf://Qwen/Qwen3-30B-A3B", &config).unwrap();

        assert_eq!(layout.family, QwenMoeFamily::Qwen3Moe);
        assert_eq!(layout.execution, QwenMoeExecutionPolicy::UPSTREAM_PARITY);
        assert_eq!(layout.configured_active_experts, 8);
        assert_eq!(layout.scheduled_active_experts, 8);
        assert_eq!(layout.routed_expert_scale, 1.0);
        assert_eq!(
            layout.routing_weight_normalization,
            Some(QwenMoeRoutingWeightNormalization::RenormalizeSelected)
        );
        assert_eq!(layout.layer_kind(0), QwenMoeLayerKind::FullAttention);
        let expert_layout = QwenMoeQ4ExpertLayout::fixed_bf16(
            layout.hidden_size,
            layout.moe_intermediate_size,
            GROUP_SIZE,
        )
        .unwrap();
        assert_ne!(expert_layout, QwenMoeQ4ExpertLayout::qwen35_a17b());
        assert_eq!(expert_layout.expert_bytes, 2_654_208);
    }

    #[test]
    fn qwen_vl_layout_keeps_vision_as_metadata_for_same_moe_architecture() {
        let config = config(
            br#"{
  "model_type": "qwen3_vl_moe",
  "architectures": ["Qwen3VLMoeForConditionalGeneration"],
  "num_hidden_layers": 2,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 248320,
  "rope_theta": 1000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 512,
  "num_experts_per_tok": 3,
  "norm_topk_prob": true,
  "moe_intermediate_size": 1536,
  "vision_config": {
    "depth": 1,
    "hidden_size": 64,
    "num_heads": 4,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "out_hidden_size": 4096
  }
}"#,
        );
        let layout = QwenMoeModelLayout::from_config(QWEN3_VL_MODEL, &config).unwrap();

        assert_eq!(layout.family, QwenMoeFamily::Qwen3VlMoe);
        assert_eq!(layout.execution, QwenMoeExecutionPolicy::UPSTREAM_PARITY);
        assert_eq!(layout.scheduled_active_experts, 3);
        assert_eq!(layout.routed_expert_scale, 1.0);
        assert!(layout.has_vision);
        assert_eq!(layout.mrope_section, Some(DEFAULT_MROPE_SECTION));
    }

    #[test]
    fn glm52_config_resolves_mla_dense_lead_in_and_sigmoid_routing() {
        assert!(is_glm52(GLM52_MXFP4_MODEL));
        assert!(is_glm52(GLM52_COLIBRI_MODEL));
        let config = config(
            br#"{
  "model_type": "glm_moe_dsa",
  "architectures": ["GlmMoeDsaForCausalLM"],
  "num_hidden_layers": 78,
  "hidden_size": 6144,
  "num_attention_heads": 64,
  "head_dim": 192,
  "vocab_size": 154880,
  "rope_parameters": {"rope_theta": 8000000.0},
  "torch_dtype": "bfloat16",
  "n_routed_experts": 256,
  "num_experts_per_tok": 8,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "moe_intermediate_size": 2048,
  "intermediate_size": 12288,
  "first_k_dense_replace": 3,
  "q_lora_rank": 2048,
  "kv_lora_rank": 512,
  "qk_nope_head_dim": 192,
  "qk_rope_head_dim": 64,
  "v_head_dim": 256,
  "n_group": 1,
  "topk_group": 1,
  "routed_scaling_factor": 2.5,
  "rms_norm_eps": 0.00001,
  "index_topk": 2048
}"#,
        );
        let layout = QwenMoeModelLayout::from_config(GLM52_MODEL, &config).unwrap();

        assert_eq!(layout.family, QwenMoeFamily::Glm52);
        assert_eq!(layout.execution, QwenMoeExecutionPolicy::GLM52_PARITY);
        assert_eq!(layout.first_sparse_layer, 3);
        assert!(config.is_dense_mlp_layer(2));
        assert!(!config.is_dense_mlp_layer(3));
        assert_eq!(layout.experts_per_layer, 256);
        assert_eq!(layout.scheduled_active_experts, 8);
        assert_eq!(layout.routed_expert_scale, 2.5);
        assert_eq!(config.rms_norm_epsilon(), 1e-5);
        assert_eq!(config.glm_mla_norm_epsilon(), Some(1e-6));
        assert_eq!(
            config.glm_logical_tensor_shape("model.layers.3.self_attn.kv_b_proj.weight"),
            Some(vec![64 * (192 + 256), 512])
        );
        assert_eq!(
            config.glm_logical_tensor_shape("model.layers.3.self_attn.embed_q.weight"),
            Some(vec![64, 512, 192])
        );
        assert_eq!(
            config.glm_logical_tensor_shape("model.layers.3.self_attn.unembed_out.weight"),
            Some(vec![64, 256, 512])
        );
        assert_eq!(
            config.glm_logical_tensor_shape("model.layers.3.mlp.experts.7.down_proj.weight"),
            Some(vec![6144, 2048])
        );
        assert_eq!(
            config.glm_logical_tensor_shape("model.layers.3.mlp.switch_mlp.gate_proj.weight"),
            Some(vec![256, 2048, 6144])
        );
        assert_eq!(
            config.glm_logical_tensor_shape("model.layers.3.mlp.switch_mlp.up_proj.weight"),
            Some(vec![256, 2048, 6144])
        );
        assert_eq!(
            config.glm_logical_tensor_shape("model.layers.3.mlp.switch_mlp.down_proj.weight"),
            Some(vec![256, 6144, 2048])
        );
    }
}
