use std::error::Error;
use std::fmt;

use super::experts::{ExpertStorageLayout, ExpertStoreExecutionDescriptor, FixedQ4ExpertSlotSpec};
#[cfg(test)]
use super::metal::MetalPipelineNameSet;
use super::metal::{MetalRuntimeCapabilities, kernels};
use super::model_family::{
    QwenMoeCommandTopology, QwenMoeExecutionArchitecture, QwenMoeExpertBufferOwnership,
    QwenMoeExpertCachePolicy, QwenMoeExpertReadStrategy, QwenMoeFamily, QwenMoeLayerKind,
    QwenMoeModelLayout, QwenMoeRoutingPlacement, QwenMoeRoutingWeightNormalization,
};
use super::weights::ResidentDenseLayout;

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
pub enum FlashMoeStageImplementation {
    QwenTextInput,
    DeferredMetalCmd3,
    MetalResidentQ4AttentionProjections,
    QwenFullAttentionCpuKv,
    MetalResidentQ4PostAttention,
    CpuSoftmaxTopK,
    ParallelPositionedFixedQ4Reads,
    MetalFixedQ4ExpertSharedCombine,
    MetalResidentQ4LmHeadSampler,
}

impl FlashMoeStageImplementation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QwenTextInput => "Qwen text token/position adapter",
            Self::DeferredMetalCmd3 => "deferred Metal CMD3 handoff",
            Self::MetalResidentQ4AttentionProjections => "Metal resident-Q4 attention projections",
            Self::QwenFullAttentionCpuKv => "Qwen full-attention CPU KV implementation",
            Self::MetalResidentQ4PostAttention => {
                "Metal resident-Q4 post-attention and router projection"
            }
            Self::CpuSoftmaxTopK => "Qwen-family CPU softmax/topK",
            Self::ParallelPositionedFixedQ4Reads => {
                "parallel positioned reads into fixed-Q4 whole-expert slots"
            }
            Self::MetalFixedQ4ExpertSharedCombine => {
                "Metal fixed-Q4 active experts and declared shared/no-shared combine"
            }
            Self::MetalResidentQ4LmHeadSampler => "Metal resident-Q4 LM-head and sampler",
        }
    }
}

impl fmt::Display for FlashMoeStageImplementation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashMoeStageCapability {
    pub stage: FlashMoeGraphStage,
    pub placement: FlashMoeStagePlacement,
    pub implementation: FlashMoeStageImplementation,
}

impl FlashMoeStageCapability {
    pub const fn new(
        stage: FlashMoeGraphStage,
        placement: FlashMoeStagePlacement,
        implementation: FlashMoeStageImplementation,
    ) -> Self {
        Self {
            stage,
            placement,
            implementation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashMoeStatePolicy {
    DeferredGpuNextLayer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlashMoeDeviceCapability {
    pub(crate) metal: MetalRuntimeCapabilities,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlashMoeCapabilityPlan {
    pub family: QwenMoeFamily,
    pub(crate) dense_layout: ResidentDenseLayout,
    pub(crate) expert_storage: ExpertStoreExecutionDescriptor,
    pub(crate) device: FlashMoeDeviceCapability,
    pub routing: QwenMoeRoutingPlacement,
    pub experts_per_layer: usize,
    pub active_experts: usize,
    pub routing_weight_normalization: QwenMoeRoutingWeightNormalization,
    pub routed_expert_scale: f32,
    pub state_policy: FlashMoeStatePolicy,
    pub stages: Vec<FlashMoeStageCapability>,
}

impl FlashMoeCapabilityPlan {
    pub(crate) fn resolve(
        layout: &QwenMoeModelLayout,
        dense_layout: ResidentDenseLayout,
        expert_storage: ExpertStoreExecutionDescriptor,
        metal: Option<MetalRuntimeCapabilities>,
    ) -> Result<Self, FlashMoeUnsupportedCapability> {
        validate_upstream_execution_policy(layout)?;
        match layout.family {
            QwenMoeFamily::Qwen35A17B | QwenMoeFamily::Qwen3Moe => {
                Self::resolve_qwen_text_q4(layout, dense_layout, expert_storage, metal)
            }
            QwenMoeFamily::Qwen3VlMoe => Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::TokenPositionInputPreparation,
                "Qwen3-VL vision adapter has not been wired into the unified scheduler",
            )),
        }
    }

    fn resolve_qwen_text_q4(
        layout: &QwenMoeModelLayout,
        dense_layout: ResidentDenseLayout,
        expert_storage: ExpertStoreExecutionDescriptor,
        metal: Option<MetalRuntimeCapabilities>,
    ) -> Result<Self, FlashMoeUnsupportedCapability> {
        if dense_layout != ResidentDenseLayout::Q4 {
            return Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::Cmd1AttentionProjections,
                format!(
                    "the unified Qwen-family Q4 graph requires resident Q4 dense projections, loaded {}",
                    dense_layout.as_str()
                ),
            ));
        }
        if expert_storage.layout != ExpertStorageLayout::FixedQ4 {
            return Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::ActiveExpertReads,
                "the unified Qwen-family Q4 graph requires fixed-Q4 whole-expert storage",
            ));
        }
        let expected_fixed_q4 =
            FixedQ4ExpertSlotSpec::from_model_layout(layout).map_err(|error| {
                FlashMoeUnsupportedCapability::new(
                    layout.family,
                    FlashMoeGraphStage::ActiveExpertReads,
                    format!("fixed-Q4 expert layout cannot be resolved: {error}"),
                )
            })?;
        if expert_storage.fixed_q4 != expected_fixed_q4
            || expert_storage.layers != layout.layers
            || expert_storage.experts_per_layer != layout.experts_per_layer
        {
            return Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::ActiveExpertReads,
                "fixed-Q4 expert storage does not match the resolved Qwen-family model layout",
            ));
        }
        let metal = metal.ok_or_else(|| {
            FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::DeferredPreviousCmd3,
                "the resolved Qwen-family Q4 graph requires a compiled Metal executor",
            )
        })?;
        let routing_weight_normalization = require_selected_route_renormalization(layout)?;

        require_stage_kernels(
            layout.family,
            &metal,
            FlashMoeGraphStage::Cmd1AttentionProjections,
            &[
                kernels::Q4_MMAP_FMA_MATVEC,
                kernels::Q4_MMAP_FMA_MATVEC_BF16_SCALE_BIAS,
                kernels::Q4_MMAP_FMA_MATVEC_BATCH,
                kernels::Q4_MMAP_FMA_MATVEC_BATCH_BF16_SCALE_BIAS,
            ],
        )?;
        if (0..layout.layers)
            .any(|layer| layout.layer_kind(layer) == QwenMoeLayerKind::LinearAttention)
        {
            require_stage_kernels(
                layout.family,
                &metal,
                FlashMoeGraphStage::Cmd1AttentionProjections,
                &[
                    kernels::LINEAR_CONV1D_STEP,
                    kernels::LINEAR_RMS_NORM_QK,
                    kernels::LINEAR_COMPUTE_DECAY_BETA,
                    kernels::LINEAR_GATED_DELTA_STEP,
                    kernels::LINEAR_GATED_RMS_NORM,
                ],
            )?;
        }
        require_stage_kernels(
            layout.family,
            &metal,
            FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection,
            &[kernels::RESIDUAL_ADD_RMS_NORM],
        )?;
        require_stage_kernels(
            layout.family,
            &metal,
            FlashMoeGraphStage::Cmd3ExpertAndSharedCombine,
            &[
                kernels::Q4_FMA_MATVEC_BF16_SCALE_BIAS,
                kernels::Q4_SWIGLU_FUSED_BF16_SCALE_BIAS,
                kernels::COMBINE_EXPERT_PHASE,
                kernels::RMS_NORM_REDUCED,
                kernels::FILL_ZERO,
            ],
        )?;
        if layout.shared_experts > 0 {
            require_stage_kernels(
                layout.family,
                &metal,
                FlashMoeGraphStage::Cmd3ExpertAndSharedCombine,
                &[kernels::SHARED_EXPERT_ACTIVATION],
            )?;
        }
        require_stage_kernels(
            layout.family,
            &metal,
            FlashMoeGraphStage::LmHeadAndSampling,
            &[kernels::LM_HEAD_LOGITS, kernels::TOPK_VOCAB],
        )?;

        let stages = vec![
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::TokenPositionInputPreparation,
                FlashMoeStagePlacement::InputAdapter,
                FlashMoeStageImplementation::QwenTextInput,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::DeferredPreviousCmd3,
                FlashMoeStagePlacement::Metal,
                FlashMoeStageImplementation::DeferredMetalCmd3,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::Cmd1AttentionProjections,
                FlashMoeStagePlacement::Metal,
                FlashMoeStageImplementation::MetalResidentQ4AttentionProjections,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::AttentionMath,
                FlashMoeStagePlacement::CpuDeclared,
                FlashMoeStageImplementation::QwenFullAttentionCpuKv,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection,
                FlashMoeStagePlacement::Metal,
                FlashMoeStageImplementation::MetalResidentQ4PostAttention,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::RoutingSoftmaxTopK,
                FlashMoeStagePlacement::CpuDeclared,
                FlashMoeStageImplementation::CpuSoftmaxTopK,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::ActiveExpertReads,
                FlashMoeStagePlacement::SchedulerIo,
                FlashMoeStageImplementation::ParallelPositionedFixedQ4Reads,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::Cmd3ExpertAndSharedCombine,
                FlashMoeStagePlacement::Metal,
                FlashMoeStageImplementation::MetalFixedQ4ExpertSharedCombine,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::LmHeadAndSampling,
                FlashMoeStagePlacement::Sampler,
                FlashMoeStageImplementation::MetalResidentQ4LmHeadSampler,
            ),
        ];
        let plan = Self {
            family: layout.family,
            dense_layout,
            expert_storage,
            device: FlashMoeDeviceCapability { metal },
            routing: layout.execution.routing,
            experts_per_layer: layout.experts_per_layer,
            active_experts: layout.scheduled_active_experts,
            routing_weight_normalization,
            routed_expert_scale: layout.routed_expert_scale,
            state_policy: FlashMoeStatePolicy::DeferredGpuNextLayer,
            stages,
        };
        plan.validate_complete()?;
        Ok(plan)
    }

    #[cfg(test)]
    pub fn for_model_layout(
        layout: &QwenMoeModelLayout,
    ) -> Result<Self, FlashMoeUnsupportedCapability> {
        validate_upstream_execution_policy(layout)?;
        Self::resolve(
            layout,
            ResidentDenseLayout::Q4,
            test_expert_storage(layout)?,
            Some(MetalRuntimeCapabilities::from_pipeline_names(
                MetalPipelineNameSet::new(),
            )),
        )
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

#[cfg(test)]
fn test_expert_storage(
    layout: &QwenMoeModelLayout,
) -> Result<ExpertStoreExecutionDescriptor, FlashMoeUnsupportedCapability> {
    let fixed_q4 = FixedQ4ExpertSlotSpec::from_model_layout(layout).map_err(|error| {
        FlashMoeUnsupportedCapability::new(
            layout.family,
            FlashMoeGraphStage::ActiveExpertReads,
            format!("model family has no fixed-Q4 test layout: {error}"),
        )
    })?;
    Ok(ExpertStoreExecutionDescriptor {
        layout: ExpertStorageLayout::FixedQ4,
        fixed_q4,
        layers: layout.layers,
        experts_per_layer: layout.experts_per_layer,
    })
}

fn require_stage_kernels(
    family: QwenMoeFamily,
    metal: &MetalRuntimeCapabilities,
    stage: FlashMoeGraphStage,
    kernels: &[&'static str],
) -> Result<(), FlashMoeUnsupportedCapability> {
    metal.require_all(kernels).map_err(|error| {
        FlashMoeUnsupportedCapability::new(
            family,
            stage,
            format!("compiled Metal kernel surface is incomplete: {error}"),
        )
    })
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

fn require_selected_route_renormalization(
    layout: &QwenMoeModelLayout,
) -> Result<QwenMoeRoutingWeightNormalization, FlashMoeUnsupportedCapability> {
    match layout.routing_weight_normalization {
        Some(QwenMoeRoutingWeightNormalization::RenormalizeSelected) => {
            Ok(QwenMoeRoutingWeightNormalization::RenormalizeSelected)
        }
        Some(QwenMoeRoutingWeightNormalization::PreserveFullSoftmax) => {
            Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::RoutingSoftmaxTopK,
                "the resolved scheduler implements selected-route renormalization, but norm_topk_prob=false requires preserving probabilities from the full expert softmax",
            ))
        }
        None => Err(FlashMoeUnsupportedCapability::new(
            layout.family,
            FlashMoeGraphStage::RoutingSoftmaxTopK,
            "the model config does not declare norm_topk_prob, so routing-weight normalization cannot be resolved",
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeUnsupportedCapability {
    pub family: QwenMoeFamily,
    pub stage: FlashMoeGraphStage,
    pub reason: String,
}

impl FlashMoeUnsupportedCapability {
    pub fn new(
        family: QwenMoeFamily,
        stage: FlashMoeGraphStage,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            family,
            stage,
            reason: reason.into(),
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

    fn qwen3_moe_layout(norm_topk_prob: Option<bool>) -> QwenMoeModelLayout {
        let mut value = serde_json::json!({
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
            "moe_intermediate_size": 768
        });
        if let Some(normalize) = norm_topk_prob {
            value["norm_topk_prob"] = serde_json::Value::Bool(normalize);
        }
        let config: QwenModelConfig = serde_json::from_value(value).unwrap();
        QwenMoeModelLayout::from_config("hf://Qwen/Qwen3-30B-A3B", &config).unwrap()
    }

    fn fixed_q4_experts(layout: &QwenMoeModelLayout) -> ExpertStoreExecutionDescriptor {
        test_expert_storage(layout).unwrap()
    }

    fn full_metal() -> MetalRuntimeCapabilities {
        MetalRuntimeCapabilities::from_pipeline_names(MetalPipelineNameSet::new())
    }

    fn metal_without_shared_activation() -> MetalRuntimeCapabilities {
        let mut names = MetalPipelineNameSet::new();
        names.shared_expert_activation = kernels::FILL_ZERO;
        MetalRuntimeCapabilities::from_pipeline_names(names)
    }

    fn metal_without_linear_conv1d() -> MetalRuntimeCapabilities {
        let mut names = MetalPipelineNameSet::new();
        names.linear_conv1d = kernels::FILL_ZERO;
        MetalRuntimeCapabilities::from_pipeline_names(names)
    }

    #[test]
    fn qwen35_q4_capability_plan_resolves_concrete_storage_and_device() {
        let layout = qwen35_layout();
        let plan = FlashMoeCapabilityPlan::resolve(
            &layout,
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            Some(full_metal()),
        )
        .unwrap();

        plan.validate_complete().unwrap();
        assert_eq!(plan.family, QwenMoeFamily::Qwen35A17B);
        assert_eq!(plan.dense_layout, ResidentDenseLayout::Q4);
        assert_eq!(plan.expert_storage.layout, ExpertStorageLayout::FixedQ4);
        assert_eq!(plan.routing, QwenMoeRoutingPlacement::CpuSoftmaxTopK);
        assert_eq!(plan.experts_per_layer, 512);
        assert_eq!(plan.active_experts, 4);
        assert_eq!(plan.state_policy, FlashMoeStatePolicy::DeferredGpuNextLayer);
        assert_eq!(plan.stages.len(), FlashMoeGraphStage::ALL.len());
        assert_eq!(
            plan.stage(FlashMoeGraphStage::Cmd3ExpertAndSharedCombine)
                .unwrap()
                .implementation,
            FlashMoeStageImplementation::MetalFixedQ4ExpertSharedCombine
        );
    }

    #[test]
    fn qwen35_q4_capability_plan_rejects_missing_metal_executor() {
        let layout = qwen35_layout();
        let err = FlashMoeCapabilityPlan::resolve(
            &layout,
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            None,
        )
        .unwrap_err();

        assert_eq!(err.stage, FlashMoeGraphStage::DeferredPreviousCmd3);
        assert!(
            err.to_string()
                .contains("requires a compiled Metal executor"),
            "{err}"
        );
    }

    #[test]
    fn qwen35_q4_capability_plan_rejects_non_q4_dense_storage() {
        let layout = qwen35_layout();
        let err = FlashMoeCapabilityPlan::resolve(
            &layout,
            ResidentDenseLayout::Bf16,
            fixed_q4_experts(&layout),
            Some(full_metal()),
        )
        .unwrap_err();

        assert_eq!(err.stage, FlashMoeGraphStage::Cmd1AttentionProjections);
        assert!(err.to_string().contains("loaded resident BF16"), "{err}");
    }

    #[test]
    fn qwen35_q4_capability_plan_rejects_incomplete_kernel_surface() {
        let layout = qwen35_layout();
        let err = FlashMoeCapabilityPlan::resolve(
            &layout,
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            Some(MetalRuntimeCapabilities::empty_for_test()),
        )
        .unwrap_err();

        assert_eq!(err.stage, FlashMoeGraphStage::Cmd1AttentionProjections);
        assert!(err.to_string().contains("missing Metal kernels"), "{err}");
    }

    #[test]
    fn qwen_vl_capability_plan_fails_with_a_named_missing_stage() {
        let layout = qwen3_vl_layout();
        let qwen35 = qwen35_layout();
        let err = FlashMoeCapabilityPlan::resolve(
            &layout,
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&qwen35),
            Some(full_metal()),
        )
        .unwrap_err();

        assert_eq!(err.family, QwenMoeFamily::Qwen3VlMoe);
        assert_eq!(err.stage, FlashMoeGraphStage::TokenPositionInputPreparation);
        assert!(err.to_string().contains("Qwen3-VL vision adapter"), "{err}");
    }

    #[test]
    fn qwen_moe_capability_requires_explicit_selected_route_normalization() {
        let missing = qwen3_moe_layout(None);
        let missing_error = FlashMoeCapabilityPlan::for_model_layout(&missing).unwrap_err();
        assert_eq!(missing_error.stage, FlashMoeGraphStage::RoutingSoftmaxTopK);
        assert!(
            missing_error
                .reason
                .contains("does not declare norm_topk_prob")
        );

        let preserve = qwen3_moe_layout(Some(false));
        let preserve_error = FlashMoeCapabilityPlan::for_model_layout(&preserve).unwrap_err();
        assert_eq!(preserve_error.stage, FlashMoeGraphStage::RoutingSoftmaxTopK);
        assert!(preserve_error.reason.contains("norm_topk_prob=false"));
    }

    #[test]
    fn qwen_moe_q4_capability_resolves_the_unified_graph() {
        let layout = qwen3_moe_layout(Some(true));
        let plan = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();

        plan.validate_complete().unwrap();
        assert_eq!(plan.family, QwenMoeFamily::Qwen3Moe);
        assert_eq!(plan.dense_layout, ResidentDenseLayout::Q4);
        assert_eq!(plan.expert_storage.layout, ExpertStorageLayout::FixedQ4);
        assert_eq!(plan.expert_storage.fixed_q4.layout.expert_bytes, 2_654_208);
        assert_eq!(plan.experts_per_layer, 128);
        assert_eq!(plan.active_experts, 8);
        assert_eq!(
            plan.routing_weight_normalization,
            QwenMoeRoutingWeightNormalization::RenormalizeSelected
        );
        assert_eq!(plan.routed_expert_scale, 1.0);
        assert_eq!(plan.stages.len(), FlashMoeGraphStage::ALL.len());
        assert_eq!(
            plan.stage(FlashMoeGraphStage::AttentionMath)
                .unwrap()
                .implementation,
            FlashMoeStageImplementation::QwenFullAttentionCpuKv
        );
    }

    #[test]
    fn qwen_text_q4_capability_carries_resolved_active_expert_policy() {
        let layout = qwen3_moe_layout(Some(true))
            .with_scheduled_active_experts(6)
            .unwrap();
        let plan = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();

        assert_eq!(plan.experts_per_layer, 128);
        assert_eq!(plan.active_experts, 6);
        let graph =
            crate::inference::flashmoe::scheduler::FlashMoeScheduledGraph::from_capabilities(&plan)
                .unwrap();
        assert_eq!(graph.experts_per_layer(), 128);
        assert_eq!(graph.active_experts(), 6);

        let error = qwen3_moe_layout(Some(true))
            .with_scheduled_active_experts(129)
            .unwrap_err();
        assert!(error.to_string().contains("scheduled_k=129"), "{error:#}");
    }

    #[test]
    fn qwen_text_q4_kernel_requirements_follow_resolved_layer_metadata() {
        let qwen3 = qwen3_moe_layout(Some(true));
        FlashMoeCapabilityPlan::resolve(
            &qwen3,
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&qwen3),
            Some(metal_without_shared_activation()),
        )
        .expect("Qwen3 without shared experts must not require the shared activation kernel");

        let qwen35 = qwen35_layout();
        let shared_error = FlashMoeCapabilityPlan::resolve(
            &qwen35,
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&qwen35),
            Some(metal_without_shared_activation()),
        )
        .unwrap_err();
        assert_eq!(
            shared_error.stage,
            FlashMoeGraphStage::Cmd3ExpertAndSharedCombine
        );
        assert!(
            shared_error
                .reason
                .contains(kernels::SHARED_EXPERT_ACTIVATION)
        );

        let linear_error = FlashMoeCapabilityPlan::resolve(
            &qwen35,
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&qwen35),
            Some(metal_without_linear_conv1d()),
        )
        .unwrap_err();
        assert_eq!(
            linear_error.stage,
            FlashMoeGraphStage::Cmd1AttentionProjections
        );
        assert!(linear_error.reason.contains(kernels::LINEAR_CONV1D_STEP));
    }

    #[test]
    fn incomplete_capability_plan_reports_the_first_missing_stage() {
        let layout = qwen35_layout();
        let plan = FlashMoeCapabilityPlan {
            family: QwenMoeFamily::Qwen35A17B,
            dense_layout: ResidentDenseLayout::Q4,
            expert_storage: fixed_q4_experts(&layout),
            device: FlashMoeDeviceCapability {
                metal: full_metal(),
            },
            routing: QwenMoeRoutingPlacement::CpuSoftmaxTopK,
            experts_per_layer: 512,
            active_experts: 4,
            routing_weight_normalization: QwenMoeRoutingWeightNormalization::RenormalizeSelected,
            routed_expert_scale: 0.9,
            state_policy: FlashMoeStatePolicy::DeferredGpuNextLayer,
            stages: vec![FlashMoeStageCapability::new(
                FlashMoeGraphStage::TokenPositionInputPreparation,
                FlashMoeStagePlacement::InputAdapter,
                FlashMoeStageImplementation::QwenTextInput,
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
