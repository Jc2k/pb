use std::error::Error;
use std::fmt;

use super::model_family::{
    QwenMoeCommandTopology, QwenMoeExecutionArchitecture, QwenMoeExpertBufferOwnership,
    QwenMoeExpertCachePolicy, QwenMoeExpertReadStrategy, QwenMoeFamily, QwenMoeModelLayout,
    QwenMoeRoutingPlacement,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FlashMoeGraphStage {
    TokenPositionInputPreparation,
    DeferredPreviousCmd3,
    Cmd1AttentionProjections,
    AttentionMath,
    Cmd2PostAttentionAndRoutingProjection,
    RoutingSoftmaxTopK,
    ActiveExpertReads,
    Cmd3ExpertAndSharedCombine,
    LmHeadAndSampling,
}

impl FlashMoeGraphStage {
    pub const ALL: [Self; 9] = [
        Self::TokenPositionInputPreparation,
        Self::DeferredPreviousCmd3,
        Self::Cmd1AttentionProjections,
        Self::AttentionMath,
        Self::Cmd2PostAttentionAndRoutingProjection,
        Self::RoutingSoftmaxTopK,
        Self::ActiveExpertReads,
        Self::Cmd3ExpertAndSharedCombine,
        Self::LmHeadAndSampling,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TokenPositionInputPreparation => "token/position input preparation",
            Self::DeferredPreviousCmd3 => "deferred previous-layer CMD3 completion",
            Self::Cmd1AttentionProjections => "CMD1 attention projections",
            Self::AttentionMath => "attention math",
            Self::Cmd2PostAttentionAndRoutingProjection => {
                "CMD2 post-attention and routing projection"
            }
            Self::RoutingSoftmaxTopK => "routing softmax/topK",
            Self::ActiveExpertReads => "active expert reads",
            Self::Cmd3ExpertAndSharedCombine => "CMD3 expert/shared combine",
            Self::LmHeadAndSampling => "LM-head and sampling",
        }
    }
}

impl fmt::Display for FlashMoeGraphStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashMoeStagePlacement {
    InputAdapter,
    Metal,
    CpuDeclared,
    SchedulerIo,
    Sampler,
}

impl FlashMoeStagePlacement {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InputAdapter => "input adapter",
            Self::Metal => "Metal",
            Self::CpuDeclared => "declared CPU",
            Self::SchedulerIo => "scheduler I/O",
            Self::Sampler => "sampler",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashMoeDenseLayoutCapability {
    ResidentQ4,
    ResidentBf16,
    ResidentF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashMoeExpertLayoutCapability {
    FixedQ4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashMoeStageCapability {
    pub stage: FlashMoeGraphStage,
    pub placement: FlashMoeStagePlacement,
    pub implementation: &'static str,
}

impl FlashMoeStageCapability {
    pub const fn new(
        stage: FlashMoeGraphStage,
        placement: FlashMoeStagePlacement,
        implementation: &'static str,
    ) -> Self {
        Self {
            stage,
            placement,
            implementation,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeCapabilityPlan {
    pub family: QwenMoeFamily,
    pub dense_layouts: Vec<FlashMoeDenseLayoutCapability>,
    pub expert_layout: FlashMoeExpertLayoutCapability,
    pub routing: QwenMoeRoutingPlacement,
    pub stages: Vec<FlashMoeStageCapability>,
}

impl FlashMoeCapabilityPlan {
    pub fn for_model_layout(
        layout: &QwenMoeModelLayout,
    ) -> Result<Self, FlashMoeUnsupportedCapability> {
        validate_upstream_execution_policy(layout)?;
        match layout.family {
            QwenMoeFamily::Qwen35A17B => Ok(Self::qwen35_q4(layout)),
            QwenMoeFamily::Qwen3Moe => Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::TokenPositionInputPreparation,
                "Qwen3 MoE graph-stage capabilities have not been wired into the unified scheduler",
            )),
            QwenMoeFamily::Qwen3VlMoe => Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::TokenPositionInputPreparation,
                "Qwen3-VL vision adapter has not been wired into the unified scheduler",
            )),
        }
    }

    fn qwen35_q4(layout: &QwenMoeModelLayout) -> Self {
        let stages = vec![
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::TokenPositionInputPreparation,
                FlashMoeStagePlacement::InputAdapter,
                "Qwen tokenizer token/position input",
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::DeferredPreviousCmd3,
                FlashMoeStagePlacement::Metal,
                "deferred upstream CMD3 handoff",
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::Cmd1AttentionProjections,
                FlashMoeStagePlacement::Metal,
                "resident dense projection descriptors",
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::AttentionMath,
                FlashMoeStagePlacement::CpuDeclared,
                "Qwen3.5 upstream-parity CPU attention math",
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection,
                FlashMoeStagePlacement::Metal,
                "resident CMD2 post-attention/router/shared gate-up",
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::RoutingSoftmaxTopK,
                FlashMoeStagePlacement::CpuDeclared,
                "Qwen3.5 upstream-parity CPU softmax/topK",
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::ActiveExpertReads,
                FlashMoeStagePlacement::SchedulerIo,
                "parallel positioned reads into whole-expert slots",
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::Cmd3ExpertAndSharedCombine,
                FlashMoeStagePlacement::Metal,
                "fixed-Q4 active experts and shared combine",
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::LmHeadAndSampling,
                FlashMoeStagePlacement::Sampler,
                "resident LM-head projection and sampler",
            ),
        ];
        Self {
            family: layout.family,
            dense_layouts: vec![
                FlashMoeDenseLayoutCapability::ResidentQ4,
                FlashMoeDenseLayoutCapability::ResidentBf16,
                FlashMoeDenseLayoutCapability::ResidentF32,
            ],
            expert_layout: FlashMoeExpertLayoutCapability::FixedQ4,
            routing: layout.execution.routing,
            stages,
        }
    }

    pub fn validate_complete(&self) -> Result<(), FlashMoeUnsupportedCapability> {
        for stage in FlashMoeGraphStage::ALL {
            if self.stage(stage).is_none() {
                return Err(FlashMoeUnsupportedCapability::new(
                    self.family,
                    stage,
                    "graph stage has no declared implementation",
                ));
            }
        }
        Ok(())
    }

    pub fn stage(&self, stage: FlashMoeGraphStage) -> Option<&FlashMoeStageCapability> {
        self.stages
            .iter()
            .find(|capability| capability.stage == stage)
    }
}

fn validate_upstream_execution_policy(
    layout: &QwenMoeModelLayout,
) -> Result<(), FlashMoeUnsupportedCapability> {
    let execution = layout.execution;
    if execution.architecture != QwenMoeExecutionArchitecture::UnifiedFlashMoe {
        return Err(FlashMoeUnsupportedCapability::new(
            layout.family,
            FlashMoeGraphStage::TokenPositionInputPreparation,
            "only the unified FlashMoe execution architecture is supported",
        ));
    }
    if execution.routing != QwenMoeRoutingPlacement::CpuSoftmaxTopK {
        return Err(FlashMoeUnsupportedCapability::new(
            layout.family,
            FlashMoeGraphStage::RoutingSoftmaxTopK,
            "routing placement is not implemented by the unified scheduler",
        ));
    }
    if execution.expert_reads != QwenMoeExpertReadStrategy::ParallelPositionedReads
        || execution.expert_cache != QwenMoeExpertCachePolicy::OsPageCacheOnly
        || execution.expert_buffer_ownership
            != QwenMoeExpertBufferOwnership::SchedulerReusableWholeExpertSlots
    {
        return Err(FlashMoeUnsupportedCapability::new(
            layout.family,
            FlashMoeGraphStage::ActiveExpertReads,
            "expert reads must use scheduler-owned whole slots backed by positioned reads",
        ));
    }
    if execution.command_topology != QwenMoeCommandTopology::UpstreamCmd1Cmd2Cmd3 {
        return Err(FlashMoeUnsupportedCapability::new(
            layout.family,
            FlashMoeGraphStage::Cmd1AttentionProjections,
            "only the upstream CMD1/CMD2/CMD3 command topology is supported",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeUnsupportedCapability {
    pub family: QwenMoeFamily,
    pub stage: FlashMoeGraphStage,
    pub reason: &'static str,
}

impl FlashMoeUnsupportedCapability {
    pub const fn new(
        family: QwenMoeFamily,
        stage: FlashMoeGraphStage,
        reason: &'static str,
    ) -> Self {
        Self {
            family,
            stage,
            reason,
        }
    }
}

impl fmt::Display for FlashMoeUnsupportedCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FlashMoe unsupported {:?} path: {} is not implemented: {}",
            self.family, self.stage, self.reason
        )
    }
}

impl Error for FlashMoeUnsupportedCapability {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::flashmoe::{QWEN3_VL_MODEL, QWEN35_MODEL, QwenModelConfig};

    fn config(json: &[u8]) -> QwenModelConfig {
        serde_json::from_slice(json).unwrap()
    }

    fn qwen35_layout() -> QwenMoeModelLayout {
        let config = config(
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
  "moe_intermediate_size": 1024,
  "num_shared_experts": 1,
  "shared_expert_intermediate_size": 1024
}"#,
        );
        QwenMoeModelLayout::from_config(QWEN35_MODEL, &config).unwrap()
    }

    fn qwen3_vl_layout() -> QwenMoeModelLayout {
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
        QwenMoeModelLayout::from_config(QWEN3_VL_MODEL, &config).unwrap()
    }

    #[test]
    fn qwen35_q4_capability_plan_resolves_the_full_graph() {
        let plan = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();

        plan.validate_complete().unwrap();
        assert_eq!(plan.family, QwenMoeFamily::Qwen35A17B);
        assert_eq!(plan.routing, QwenMoeRoutingPlacement::CpuSoftmaxTopK);
        assert_eq!(plan.expert_layout, FlashMoeExpertLayoutCapability::FixedQ4);
        assert_eq!(plan.stages.len(), FlashMoeGraphStage::ALL.len());
        assert_eq!(
            plan.stage(FlashMoeGraphStage::ActiveExpertReads)
                .unwrap()
                .placement,
            FlashMoeStagePlacement::SchedulerIo
        );
        assert_eq!(
            plan.stage(FlashMoeGraphStage::RoutingSoftmaxTopK)
                .unwrap()
                .placement,
            FlashMoeStagePlacement::CpuDeclared
        );
    }

    #[test]
    fn qwen_vl_capability_plan_fails_with_a_named_missing_stage() {
        let err = FlashMoeCapabilityPlan::for_model_layout(&qwen3_vl_layout()).unwrap_err();

        assert_eq!(err.family, QwenMoeFamily::Qwen3VlMoe);
        assert_eq!(err.stage, FlashMoeGraphStage::TokenPositionInputPreparation);
        assert!(err.to_string().contains("Qwen3-VL vision adapter"), "{err}");
    }

    #[test]
    fn incomplete_capability_plan_reports_the_first_missing_stage() {
        let plan = FlashMoeCapabilityPlan {
            family: QwenMoeFamily::Qwen35A17B,
            dense_layouts: vec![FlashMoeDenseLayoutCapability::ResidentQ4],
            expert_layout: FlashMoeExpertLayoutCapability::FixedQ4,
            routing: QwenMoeRoutingPlacement::CpuSoftmaxTopK,
            stages: vec![FlashMoeStageCapability::new(
                FlashMoeGraphStage::TokenPositionInputPreparation,
                FlashMoeStagePlacement::InputAdapter,
                QWEN35_MODEL,
            )],
        };

        let err = plan.validate_complete().unwrap_err();

        assert_eq!(err.family, QwenMoeFamily::Qwen35A17B);
        assert_eq!(err.stage, FlashMoeGraphStage::DeferredPreviousCmd3);
        assert!(
            err.to_string()
                .contains("graph stage has no declared implementation"),
            "{err}"
        );
    }
}
