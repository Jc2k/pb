use anyhow::{Result, bail};

use super::legacy::{QwenModelConfig, is_qwen3_moe, is_qwen3_vl, is_qwen35_or_legacy_alias};
use super::types::{
    ACTIVE_EXPERTS_PER_TOKEN, DEFAULT_MROPE_SECTION, FOUR_BIT_EXPERT_SIZE, FULL_ATTN_INTERVAL,
    FlashMoeLayerKind, GROUP_SIZE,
};

pub const QWEN35_Q4_EXPERT_PACKED_WEIGHT_BYTES: usize = 2_097_152;
pub const QWEN35_Q4_EXPERT_SCALE_BYTES: usize = 131_072;
pub const QWEN35_Q4_EXPERT_BIAS_BYTES: usize = 131_072;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeExecutionArchitecture {
    UnifiedFlashMoe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeFamily {
    Qwen35A17B,
    Qwen3Moe,
    Qwen3VlMoe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeLayerKind {
    FullAttention,
    LinearAttention,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenMoeQ4ExpertLayout {
    pub expert_bytes: usize,
    pub group_size: usize,
    pub components: [QwenMoeExpertComponentLayout; 9],
}

impl QwenMoeQ4ExpertLayout {
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
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub experts_per_layer: usize,
    pub configured_active_experts: usize,
    pub scheduled_active_experts: usize,
    pub moe_intermediate_size: usize,
    pub shared_experts: usize,
    pub shared_expert_intermediate_size: usize,
    pub rope_theta: Option<f64>,
    pub partial_rotary_factor: Option<f64>,
    pub mrope_section: Option<[usize; 3]>,
    pub has_vision: bool,
    pub q4_expert_layout: QwenMoeQ4ExpertLayout,
}

impl QwenMoeModelLayout {
    pub fn from_config(model: &str, config: &QwenModelConfig) -> Result<Self> {
        config.validate()?;
        let family = QwenMoeFamily::from_model_and_config(model, config)?;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let configured_active_experts = config.config_active_experts();
        let scheduled_active_experts = match family {
            QwenMoeFamily::Qwen35A17B if config.experts() >= ACTIVE_EXPERTS_PER_TOKEN => {
                ACTIVE_EXPERTS_PER_TOKEN
            }
            QwenMoeFamily::Qwen35A17B => configured_active_experts,
            QwenMoeFamily::Qwen3Moe | QwenMoeFamily::Qwen3VlMoe => configured_active_experts,
        };
        let layout = Self {
            family,
            execution: QwenMoeExecutionPolicy::UPSTREAM_PARITY,
            layers: config.num_hidden_layers,
            hidden_size: config.hidden_size,
            attention_heads: config.num_attention_heads,
            kv_heads: config.kv_heads(),
            head_dim,
            vocab_size: config.vocab_size,
            experts_per_layer: config.experts(),
            configured_active_experts,
            scheduled_active_experts,
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
            q4_expert_layout: QwenMoeQ4ExpertLayout::qwen35_a17b(),
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
        }
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
        if self.hidden_size != self.attention_heads * self.head_dim {
            bail!(
                "Qwen MoE hidden size {} does not match attention_heads {} * head_dim {}",
                self.hidden_size,
                self.attention_heads,
                self.head_dim
            );
        }
        if self.experts_per_layer == 0
            || self.configured_active_experts == 0
            || self.scheduled_active_experts == 0
            || self.scheduled_active_experts > self.experts_per_layer
        {
            bail!(
                "invalid Qwen MoE expert schedule: experts={}, configured_k={}, scheduled_k={}",
                self.experts_per_layer,
                self.configured_active_experts,
                self.scheduled_active_experts
            );
        }
        self.q4_expert_layout.validate()?;
        Ok(())
    }
}

impl QwenMoeFamily {
    pub fn from_model_and_config(model: &str, config: &QwenModelConfig) -> Result<Self> {
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
            "FlashMoe only supports Qwen-family MoE models, found model={model:?} model_type={:?} architectures={:?}",
            config.model_type,
            config.architectures
        );
    }
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
  "num_key_value_heads": 2,
  "vocab_size": 248320,
  "rope_theta": 10000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "moe_intermediate_size": 1536
}"#,
        )
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
        assert_eq!(layout.kv_heads, 2);
        assert_eq!(layout.configured_active_experts, 10);
        assert_eq!(layout.scheduled_active_experts, 4);
        assert_eq!(layout.experts_per_layer, 512);
        assert_eq!(layout.layers, NUM_LAYERS);
        assert_eq!(layout.hidden_size, HIDDEN_DIM);
        assert_eq!(layout.experts_per_layer, NUM_EXPERTS);
        assert_eq!(layout.moe_intermediate_size, 1536);
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
  "num_hidden_layers": 2,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 151936,
  "rope_theta": 1000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 512,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 1536
}"#,
        );
        let layout = QwenMoeModelLayout::from_config("hf://Qwen/Qwen3-30B-A3B", &config).unwrap();

        assert_eq!(layout.family, QwenMoeFamily::Qwen3Moe);
        assert_eq!(layout.execution, QwenMoeExecutionPolicy::UPSTREAM_PARITY);
        assert_eq!(layout.configured_active_experts, 2);
        assert_eq!(layout.scheduled_active_experts, 2);
        assert_eq!(layout.layer_kind(0), QwenMoeLayerKind::FullAttention);
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
        assert!(layout.has_vision);
        assert_eq!(layout.mrope_section, Some(DEFAULT_MROPE_SECTION));
    }
}
