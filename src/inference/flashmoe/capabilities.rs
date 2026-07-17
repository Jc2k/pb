use std::error::Error;
use std::fmt;

#[cfg(test)]
use super::experts::FixedQ4ExpertSlotSpec;
use super::experts::{ExpertSlotSpec, ExpertStorageLayout, ExpertStoreExecutionDescriptor};
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
    QwenVlTypedInput,
    DeferredMetalCmd3,
    MetalResidentAttentionProjections,
    MetalResidentQ4AttentionProjections,
    QwenFullAttentionCpuKv,
    GlmMlaCpuWeightAbsorption,
    MetalResidentPostAttention,
    CpuSoftmaxTopK,
    CpuSigmoidNoAuxTopK,
    ParallelPositionedWholeExpertReads,
    MetalTypedExpertResidentSharedCombine,
    MetalResidentLmHeadSampler,
}

impl FlashMoeStageImplementation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QwenTextInput => "Qwen text token/position adapter",
            Self::QwenVlTypedInput => "Qwen-VL typed token/position/embedding/DeepStack adapter",
            Self::DeferredMetalCmd3 => "deferred Metal CMD3 handoff",
            Self::MetalResidentAttentionProjections => {
                "Metal resident Q4/BF16/F16/F32 attention projections"
            }
            Self::MetalResidentQ4AttentionProjections => "Metal resident-Q4 attention projections",
            Self::QwenFullAttentionCpuKv => "Qwen full-attention CPU KV implementation",
            Self::GlmMlaCpuWeightAbsorption => "GLM compressed-KV MLA with CPU weight absorption",
            Self::MetalResidentPostAttention => {
                "Metal resident Q4/BF16/F16/F32 post-attention and router projection"
            }
            Self::CpuSoftmaxTopK => "Qwen-family CPU softmax/topK",
            Self::CpuSigmoidNoAuxTopK => "GLM CPU sigmoid/noaux topK",
            Self::ParallelPositionedWholeExpertReads => {
                "parallel positioned reads into typed fixed Q4/BF16/F16 whole-expert slots"
            }
            Self::MetalTypedExpertResidentSharedCombine => {
                "Metal typed Q4/BF16/F16 active experts and resident Q4/BF16/F16/F32 shared/no-shared combine"
            }
            Self::MetalResidentLmHeadSampler => {
                "Metal resident Q4/BF16/F16/F32 LM-head and sampler"
            }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashMoeInputAdapterCapability {
    QwenText,
    QwenVl {
        text_hidden_size: usize,
        vision_embed_dim: usize,
        vision_depth: usize,
        deepstack_layers: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlashMoeDeviceCapability {
    pub(crate) metal: MetalRuntimeCapabilities,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlashMoeCapabilityPlan {
    pub family: QwenMoeFamily,
    pub(crate) input_adapter: FlashMoeInputAdapterCapability,
    pub(crate) dense_layout: ResidentDenseLayout,
    pub(crate) expert_storage: ExpertStoreExecutionDescriptor,
    pub(crate) device: FlashMoeDeviceCapability,
    pub routing: QwenMoeRoutingPlacement,
    pub experts_per_layer: usize,
    pub active_experts: usize,
    pub routing_weight_normalization: QwenMoeRoutingWeightNormalization,
    pub routed_expert_scale: f32,
    pub state_policy: FlashMoeStatePolicy,
    pub(crate) attention_layers: Box<[QwenMoeLayerKind]>,
    pub stages: Vec<FlashMoeStageCapability>,
}

impl FlashMoeCapabilityPlan {
    pub(crate) fn resolve(
        layout: &QwenMoeModelLayout,
        input_adapter: FlashMoeInputAdapterCapability,
        dense_layout: ResidentDenseLayout,
        expert_storage: ExpertStoreExecutionDescriptor,
        manifest_attention_layers: &[QwenMoeLayerKind],
        metal: Option<MetalRuntimeCapabilities>,
    ) -> Result<Self, FlashMoeUnsupportedCapability> {
        validate_upstream_execution_policy(layout)?;
        let input_implementation = resolve_input_adapter(layout, input_adapter)?;
        let attention_layers = resolve_attention_layers(layout, manifest_attention_layers)?;
        Self::resolve_qwen_graph(
            layout,
            input_adapter,
            input_implementation,
            dense_layout,
            expert_storage,
            attention_layers,
            metal,
        )
    }

    fn resolve_qwen_graph(
        layout: &QwenMoeModelLayout,
        input_adapter: FlashMoeInputAdapterCapability,
        input_implementation: FlashMoeStageImplementation,
        dense_layout: ResidentDenseLayout,
        expert_storage: ExpertStoreExecutionDescriptor,
        attention_layers: Box<[QwenMoeLayerKind]>,
        metal: Option<MetalRuntimeCapabilities>,
    ) -> Result<Self, FlashMoeUnsupportedCapability> {
        let expected_slot_spec = ExpertSlotSpec::from_model_layout(layout, expert_storage.layout)
            .map_err(|error| {
            FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::ActiveExpertReads,
                format!(
                    "{:?} expert layout cannot be resolved: {error}",
                    expert_storage.layout
                ),
            )
        })?;
        if expert_storage.slot_spec != expected_slot_spec
            || expert_storage.layers != layout.layers
            || expert_storage.first_expert_layer != layout.first_sparse_layer
            || expert_storage.experts_per_layer != layout.experts_per_layer
        {
            return Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::ActiveExpertReads,
                format!(
                    "{:?} expert storage does not match the resolved Qwen-family model layout",
                    expert_storage.layout
                ),
            ));
        }
        let metal = metal.ok_or_else(|| {
            FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::DeferredPreviousCmd3,
                "the resolved Qwen-family graph requires a compiled Metal executor",
            )
        })?;
        let routing_weight_normalization = require_selected_route_renormalization(layout)?;

        let has_linear_attention = attention_layers.contains(&QwenMoeLayerKind::LinearAttention);
        match dense_layout {
            ResidentDenseLayout::Q4 => require_stage_kernels(
                layout.family,
                &metal,
                FlashMoeGraphStage::Cmd1AttentionProjections,
                &[
                    kernels::Q4_MMAP_FMA_MATVEC,
                    kernels::Q4_MMAP_FMA_MATVEC_BF16_SCALE_BIAS,
                    kernels::Q4_MMAP_FMA_MATVEC_BATCH,
                    kernels::Q4_MMAP_FMA_MATVEC_BATCH_BF16_SCALE_BIAS,
                ],
            )?,
            ResidentDenseLayout::Bf16 => require_stage_kernels(
                layout.family,
                &metal,
                FlashMoeGraphStage::Cmd1AttentionProjections,
                &[kernels::DENSE_MMAP_FMA_MATVEC_BF16],
            )?,
            ResidentDenseLayout::F16 => require_stage_kernels(
                layout.family,
                &metal,
                FlashMoeGraphStage::Cmd1AttentionProjections,
                &[kernels::DENSE_MMAP_FMA_MATVEC_F16],
            )?,
            ResidentDenseLayout::F32 => require_stage_kernels(
                layout.family,
                &metal,
                FlashMoeGraphStage::Cmd1AttentionProjections,
                &[kernels::DENSE_MMAP_FMA_MATVEC_F32],
            )?,
        }
        if has_linear_attention {
            let (conv_kernel, decay_kernel, gated_norm_kernel) = match dense_layout {
                ResidentDenseLayout::Q4 | ResidentDenseLayout::Bf16 => (
                    kernels::LINEAR_CONV1D_STEP_BF16,
                    kernels::LINEAR_COMPUTE_DECAY_BETA_BF16,
                    kernels::LINEAR_GATED_RMS_NORM_BF16,
                ),
                ResidentDenseLayout::F16 => (
                    kernels::LINEAR_CONV1D_STEP_F16,
                    kernels::LINEAR_COMPUTE_DECAY_BETA_F16,
                    kernels::LINEAR_GATED_RMS_NORM_F16,
                ),
                ResidentDenseLayout::F32 => (
                    kernels::LINEAR_CONV1D_STEP_F32,
                    kernels::LINEAR_COMPUTE_DECAY_BETA_F32,
                    kernels::LINEAR_GATED_RMS_NORM_F32,
                ),
            };
            require_stage_kernels(
                layout.family,
                &metal,
                FlashMoeGraphStage::Cmd1AttentionProjections,
                &[
                    conv_kernel,
                    kernels::LINEAR_RMS_NORM_QK,
                    decay_kernel,
                    kernels::LINEAR_GATED_DELTA_STEP,
                    gated_norm_kernel,
                ],
            )?;
        }
        let cmd2_projection_kernel = match dense_layout {
            ResidentDenseLayout::Q4 => kernels::Q4_MMAP_FMA_MATVEC,
            ResidentDenseLayout::Bf16 => kernels::DENSE_MMAP_FMA_MATVEC_BF16,
            ResidentDenseLayout::F16 => kernels::DENSE_MMAP_FMA_MATVEC_F16,
            ResidentDenseLayout::F32 => kernels::DENSE_MMAP_FMA_MATVEC_F32,
        };
        require_stage_kernels(
            layout.family,
            &metal,
            FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection,
            &[cmd2_projection_kernel, kernels::RESIDUAL_ADD_RMS_NORM],
        )?;
        if layout.family == QwenMoeFamily::Glm52 {
            // Colibri keeps the router in F32 even though the resident
            // attention/shared projections are Q4.
            require_stage_kernels(
                layout.family,
                &metal,
                FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection,
                &[kernels::DENSE_MMAP_FMA_MATVEC_F32],
            )?;
        }
        let active_expert_kernels: &[&str] = match expert_storage.layout {
            ExpertStorageLayout::FixedQ4 => &[
                kernels::Q4_FMA_MATVEC_BF16_SCALE_BIAS,
                kernels::SILU_PRODUCT,
            ],
            ExpertStorageLayout::FixedBf16 => {
                &[kernels::DENSE_MMAP_FMA_MATVEC_BF16, kernels::SILU_PRODUCT]
            }
            ExpertStorageLayout::FixedF16 => {
                &[kernels::DENSE_MMAP_FMA_MATVEC_F16, kernels::SILU_PRODUCT]
            }
        };
        require_stage_kernels(
            layout.family,
            &metal,
            FlashMoeGraphStage::Cmd3ExpertAndSharedCombine,
            active_expert_kernels,
        )?;
        require_stage_kernels(
            layout.family,
            &metal,
            FlashMoeGraphStage::Cmd3ExpertAndSharedCombine,
            &[
                kernels::COMBINE_EXPERT_PHASE,
                kernels::RMS_NORM_REDUCED,
                kernels::FILL_ZERO,
            ],
        )?;
        if layout.shared_experts > 1 {
            return Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::Cmd3ExpertAndSharedCombine,
                "post-down shared-expert gating currently supports exactly one shared expert",
            ));
        }
        if layout.shared_experts == 1 {
            require_stage_kernels(
                layout.family,
                &metal,
                FlashMoeGraphStage::Cmd3ExpertAndSharedCombine,
                &[cmd2_projection_kernel, kernels::SHARED_EXPERT_ACTIVATION],
            )?;
        }
        let lm_head_projection_kernel = match dense_layout {
            ResidentDenseLayout::Q4 => kernels::Q4_MMAP_FMA_MATVEC,
            ResidentDenseLayout::Bf16 => kernels::DENSE_MMAP_FMA_MATVEC_BF16,
            ResidentDenseLayout::F16 => kernels::DENSE_MMAP_FMA_MATVEC_F16,
            ResidentDenseLayout::F32 => kernels::DENSE_MMAP_FMA_MATVEC_F32,
        };
        require_stage_kernels(
            layout.family,
            &metal,
            FlashMoeGraphStage::LmHeadAndSampling,
            &[lm_head_projection_kernel, kernels::TOPK_VOCAB],
        )?;
        if layout.family == QwenMoeFamily::Glm52 {
            // The published Colibri snapshot stores embedding/LM-head source
            // weights at int8; pull preserves them as resident BF16.
            require_stage_kernels(
                layout.family,
                &metal,
                FlashMoeGraphStage::LmHeadAndSampling,
                &[kernels::DENSE_MMAP_FMA_MATVEC_BF16],
            )?;
        }

        let stages = vec![
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::TokenPositionInputPreparation,
                FlashMoeStagePlacement::InputAdapter,
                input_implementation,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::DeferredPreviousCmd3,
                FlashMoeStagePlacement::Metal,
                FlashMoeStageImplementation::DeferredMetalCmd3,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::Cmd1AttentionProjections,
                FlashMoeStagePlacement::Metal,
                FlashMoeStageImplementation::MetalResidentAttentionProjections,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::AttentionMath,
                FlashMoeStagePlacement::CpuDeclared,
                if layout.family == QwenMoeFamily::Glm52 {
                    FlashMoeStageImplementation::GlmMlaCpuWeightAbsorption
                } else {
                    FlashMoeStageImplementation::QwenFullAttentionCpuKv
                },
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection,
                FlashMoeStagePlacement::Metal,
                FlashMoeStageImplementation::MetalResidentPostAttention,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::RoutingSoftmaxTopK,
                FlashMoeStagePlacement::CpuDeclared,
                match layout.execution.routing {
                    QwenMoeRoutingPlacement::CpuSoftmaxTopK => {
                        FlashMoeStageImplementation::CpuSoftmaxTopK
                    }
                    QwenMoeRoutingPlacement::CpuSigmoidNoAuxTopK => {
                        FlashMoeStageImplementation::CpuSigmoidNoAuxTopK
                    }
                },
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::ActiveExpertReads,
                FlashMoeStagePlacement::SchedulerIo,
                FlashMoeStageImplementation::ParallelPositionedWholeExpertReads,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::Cmd3ExpertAndSharedCombine,
                FlashMoeStagePlacement::Metal,
                FlashMoeStageImplementation::MetalTypedExpertResidentSharedCombine,
            ),
            FlashMoeStageCapability::new(
                FlashMoeGraphStage::LmHeadAndSampling,
                FlashMoeStagePlacement::Sampler,
                FlashMoeStageImplementation::MetalResidentLmHeadSampler,
            ),
        ];
        let plan = Self {
            family: layout.family,
            input_adapter,
            dense_layout,
            expert_storage,
            device: FlashMoeDeviceCapability { metal },
            routing: layout.execution.routing,
            experts_per_layer: layout.experts_per_layer,
            active_experts: layout.scheduled_active_experts,
            routing_weight_normalization,
            routed_expert_scale: layout.routed_expert_scale,
            state_policy: FlashMoeStatePolicy::DeferredGpuNextLayer,
            attention_layers,
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
        let attention_layers = (0..layout.layers)
            .map(|layer| layout.layer_kind(layer))
            .collect::<Vec<_>>();
        Self::resolve(
            layout,
            FlashMoeInputAdapterCapability::QwenText,
            ResidentDenseLayout::Q4,
            test_expert_storage(layout)?,
            &attention_layers,
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

fn resolve_attention_layers(
    layout: &QwenMoeModelLayout,
    manifest_attention_layers: &[QwenMoeLayerKind],
) -> Result<Box<[QwenMoeLayerKind]>, FlashMoeUnsupportedCapability> {
    if manifest_attention_layers.len() != layout.layers {
        return Err(FlashMoeUnsupportedCapability::new(
            layout.family,
            FlashMoeGraphStage::Cmd1AttentionProjections,
            format!(
                "the tensor manifest resolves {} attention layers but the model declares {}",
                manifest_attention_layers.len(),
                layout.layers
            ),
        ));
    }
    for (layer, actual) in manifest_attention_layers.iter().copied().enumerate() {
        let expected = layout.layer_kind(layer);
        if actual != expected {
            return Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::Cmd1AttentionProjections,
                format!(
                    "layer {layer} tensor layout resolves {actual:?} but the model family requires {expected:?}"
                ),
            ));
        }
    }
    Ok(manifest_attention_layers.to_vec().into_boxed_slice())
}

fn resolve_input_adapter(
    layout: &QwenMoeModelLayout,
    input_adapter: FlashMoeInputAdapterCapability,
) -> Result<FlashMoeStageImplementation, FlashMoeUnsupportedCapability> {
    match (layout.family, input_adapter) {
        (
            QwenMoeFamily::Qwen35A17B | QwenMoeFamily::Qwen3Moe | QwenMoeFamily::Glm52,
            FlashMoeInputAdapterCapability::QwenText,
        ) => Ok(FlashMoeStageImplementation::QwenTextInput),
        (
            QwenMoeFamily::Qwen3VlMoe,
            FlashMoeInputAdapterCapability::QwenVl {
                text_hidden_size,
                vision_embed_dim,
                vision_depth,
                deepstack_layers,
            },
        ) if layout.has_vision
            && text_hidden_size == layout.hidden_size
            && vision_embed_dim > 0
            && vision_depth > 0
            && deepstack_layers <= vision_depth =>
        {
            Ok(FlashMoeStageImplementation::QwenVlTypedInput)
        }
        (QwenMoeFamily::Qwen3VlMoe, FlashMoeInputAdapterCapability::QwenText) => {
            Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::TokenPositionInputPreparation,
                "the resolved resources do not include a Qwen-VL vision encoder and typed input adapter",
            ))
        }
        (
            QwenMoeFamily::Qwen3VlMoe,
            FlashMoeInputAdapterCapability::QwenVl {
                text_hidden_size,
                vision_embed_dim,
                vision_depth,
                deepstack_layers,
            },
        ) => Err(FlashMoeUnsupportedCapability::new(
            layout.family,
            FlashMoeGraphStage::TokenPositionInputPreparation,
            format!(
                "Qwen-VL adapter metadata does not match the model: has_vision={}, text_hidden_size={text_hidden_size} expected={}, vision_embed_dim={vision_embed_dim}, vision_depth={vision_depth}, deepstack_layers={deepstack_layers}",
                layout.has_vision, layout.hidden_size
            ),
        )),
        (_, FlashMoeInputAdapterCapability::QwenVl { .. }) => {
            Err(FlashMoeUnsupportedCapability::new(
                layout.family,
                FlashMoeGraphStage::TokenPositionInputPreparation,
                "a Qwen-VL input adapter cannot be bound to a text-only Qwen graph",
            ))
        }
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
        slot_spec: ExpertSlotSpec::FixedQ4(fixed_q4),
        layers: layout.layers,
        first_expert_layer: layout.first_sparse_layer,
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
    if !matches!(
        execution.routing,
        QwenMoeRoutingPlacement::CpuSoftmaxTopK | QwenMoeRoutingPlacement::CpuSigmoidNoAuxTopK
    ) {
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
    use crate::inference::flashmoe::experts::{DenseExpertDtype, FixedDenseExpertSlotSpec};
    use crate::inference::flashmoe::{GLM52_MODEL, QWEN3_VL_MODEL, QWEN35_MODEL, QwenModelConfig};

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

    fn glm52_layout() -> QwenMoeModelLayout {
        let config = config(
            br#"{
  "model_type": "glm_moe_dsa",
  "architectures": ["GlmMoeDsaForCausalLM"],
  "num_hidden_layers": 4,
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
        QwenMoeModelLayout::from_config(GLM52_MODEL, &config).unwrap()
    }

    fn attention_layers(layout: &QwenMoeModelLayout) -> Vec<QwenMoeLayerKind> {
        (0..layout.layers)
            .map(|layer| layout.layer_kind(layer))
            .collect()
    }

    fn fixed_q4_experts(layout: &QwenMoeModelLayout) -> ExpertStoreExecutionDescriptor {
        test_expert_storage(layout).unwrap()
    }

    fn fixed_dense_experts(
        layout: &QwenMoeModelLayout,
        dtype: DenseExpertDtype,
    ) -> ExpertStoreExecutionDescriptor {
        let slot = FixedDenseExpertSlotSpec::from_model_layout(layout, dtype).unwrap();
        ExpertStoreExecutionDescriptor {
            layout: match dtype {
                DenseExpertDtype::Bf16 => ExpertStorageLayout::FixedBf16,
                DenseExpertDtype::F16 => ExpertStorageLayout::FixedF16,
            },
            slot_spec: ExpertSlotSpec::FixedDense(slot),
            layers: layout.layers,
            first_expert_layer: layout.first_sparse_layer,
            experts_per_layer: layout.experts_per_layer,
        }
    }

    fn full_metal() -> MetalRuntimeCapabilities {
        MetalRuntimeCapabilities::from_pipeline_names(MetalPipelineNameSet::new())
    }

    fn text_adapter() -> FlashMoeInputAdapterCapability {
        FlashMoeInputAdapterCapability::QwenText
    }

    fn qwen_vl_adapter() -> FlashMoeInputAdapterCapability {
        FlashMoeInputAdapterCapability::QwenVl {
            text_hidden_size: 4096,
            vision_embed_dim: 64,
            vision_depth: 1,
            deepstack_layers: 0,
        }
    }

    fn metal_without_shared_activation() -> MetalRuntimeCapabilities {
        let mut names = MetalPipelineNameSet::new();
        names.shared_expert_activation = kernels::FILL_ZERO;
        MetalRuntimeCapabilities::from_pipeline_names(names)
    }

    fn metal_without_linear_conv1d() -> MetalRuntimeCapabilities {
        let mut names = MetalPipelineNameSet::new();
        names.linear_conv1d_bf16 = kernels::FILL_ZERO;
        MetalRuntimeCapabilities::from_pipeline_names(names)
    }

    fn metal_without_dense_bf16() -> MetalRuntimeCapabilities {
        let mut names = MetalPipelineNameSet::new();
        names.dense_mmap_bf16 = kernels::FILL_ZERO;
        MetalRuntimeCapabilities::from_pipeline_names(names)
    }

    fn metal_without_dense_f32() -> MetalRuntimeCapabilities {
        let mut names = MetalPipelineNameSet::new();
        names.dense_mmap_f32 = kernels::FILL_ZERO;
        MetalRuntimeCapabilities::from_pipeline_names(names)
    }

    fn metal_without_residual_rms_norm() -> MetalRuntimeCapabilities {
        let mut names = MetalPipelineNameSet::new();
        names.residual_rms_norm = kernels::FILL_ZERO;
        MetalRuntimeCapabilities::from_pipeline_names(names)
    }

    fn metal_without_topk_vocab() -> MetalRuntimeCapabilities {
        let mut names = MetalPipelineNameSet::new();
        names.topk_vocab = kernels::FILL_ZERO;
        MetalRuntimeCapabilities::from_pipeline_names(names)
    }

    #[test]
    fn qwen35_q4_capability_plan_resolves_concrete_storage_and_device() {
        let mut layout = qwen35_layout();
        // Some Qwen3.5 checkpoints retain incidental vision metadata even
        // though family resolution selects the text-only Qwen3.5 graph.
        layout.has_vision = true;
        let plan = FlashMoeCapabilityPlan::resolve(
            &layout,
            text_adapter(),
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            &attention_layers(&layout),
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
        assert_eq!(plan.attention_layers[0], QwenMoeLayerKind::LinearAttention);
        assert_eq!(plan.attention_layers[3], QwenMoeLayerKind::FullAttention);
        assert_eq!(plan.stages.len(), FlashMoeGraphStage::ALL.len());
        assert_eq!(
            plan.stage(FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection)
                .unwrap()
                .implementation,
            FlashMoeStageImplementation::MetalResidentPostAttention
        );
        assert_eq!(
            plan.stage(FlashMoeGraphStage::Cmd3ExpertAndSharedCombine)
                .unwrap()
                .implementation,
            FlashMoeStageImplementation::MetalTypedExpertResidentSharedCombine
        );
    }

    #[test]
    fn capability_resolution_rejects_unimplemented_multi_shared_post_down_gate() {
        let mut layout = qwen35_layout();
        layout.shared_experts = 2;

        let error = FlashMoeCapabilityPlan::resolve(
            &layout,
            text_adapter(),
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            &attention_layers(&layout),
            Some(full_metal()),
        )
        .unwrap_err();

        assert_eq!(error.stage, FlashMoeGraphStage::Cmd3ExpertAndSharedCombine);
        assert!(error.reason.contains("exactly one shared expert"));
    }

    #[test]
    fn capability_resolution_rejects_manifest_attention_schedule_mismatches() {
        let layout = qwen3_moe_layout(Some(true));
        let mut wrong_kind = attention_layers(&layout);
        wrong_kind[0] = QwenMoeLayerKind::LinearAttention;
        let kind_error = FlashMoeCapabilityPlan::resolve(
            &layout,
            text_adapter(),
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            &wrong_kind,
            Some(full_metal()),
        )
        .unwrap_err();
        assert_eq!(
            kind_error.stage,
            FlashMoeGraphStage::Cmd1AttentionProjections
        );
        assert!(
            kind_error.reason.contains(
                "layer 0 tensor layout resolves LinearAttention but the model family requires FullAttention"
            ),
            "{kind_error}"
        );

        let missing_layer = &attention_layers(&layout)[..layout.layers - 1];
        let count_error = FlashMoeCapabilityPlan::resolve(
            &layout,
            text_adapter(),
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            missing_layer,
            Some(full_metal()),
        )
        .unwrap_err();
        assert_eq!(
            count_error.stage,
            FlashMoeGraphStage::Cmd1AttentionProjections
        );
        assert!(
            count_error
                .reason
                .contains("resolves 47 attention layers but the model declares 48"),
            "{count_error}"
        );
    }

    #[test]
    fn qwen35_q4_capability_plan_rejects_missing_metal_executor() {
        let layout = qwen35_layout();
        let err = FlashMoeCapabilityPlan::resolve(
            &layout,
            text_adapter(),
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            &attention_layers(&layout),
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
    fn qwen35_non_q4_dense_resolves_complete_unified_graph() {
        let layout = qwen35_layout();
        for dense_layout in [
            ResidentDenseLayout::Bf16,
            ResidentDenseLayout::F16,
            ResidentDenseLayout::F32,
        ] {
            let plan = FlashMoeCapabilityPlan::resolve(
                &layout,
                text_adapter(),
                dense_layout,
                fixed_q4_experts(&layout),
                &attention_layers(&layout),
                Some(full_metal()),
            )
            .unwrap();

            plan.validate_complete().unwrap();
            assert_eq!(plan.dense_layout, dense_layout);
            assert_eq!(plan.stages.len(), FlashMoeGraphStage::ALL.len());
            assert_eq!(
                plan.stage(FlashMoeGraphStage::Cmd3ExpertAndSharedCombine)
                    .unwrap()
                    .implementation,
                FlashMoeStageImplementation::MetalTypedExpertResidentSharedCombine
            );
        }
    }

    #[test]
    fn qwen35_typed_dense_expert_slots_resolve_complete_unified_graph() {
        let layout = qwen35_layout();
        for expert_dtype in [DenseExpertDtype::Bf16, DenseExpertDtype::F16] {
            let plan = FlashMoeCapabilityPlan::resolve(
                &layout,
                text_adapter(),
                ResidentDenseLayout::Q4,
                fixed_dense_experts(&layout, expert_dtype),
                &attention_layers(&layout),
                Some(full_metal()),
            )
            .unwrap();
            plan.validate_complete().unwrap();
            assert_eq!(plan.stages.len(), FlashMoeGraphStage::ALL.len());
            assert_eq!(
                plan.stage(FlashMoeGraphStage::ActiveExpertReads)
                    .unwrap()
                    .implementation,
                FlashMoeStageImplementation::ParallelPositionedWholeExpertReads
            );
            assert_eq!(
                plan.stage(FlashMoeGraphStage::Cmd3ExpertAndSharedCombine)
                    .unwrap()
                    .implementation,
                FlashMoeStageImplementation::MetalTypedExpertResidentSharedCombine
            );
        }

        let missing_kernel = FlashMoeCapabilityPlan::resolve(
            &layout,
            text_adapter(),
            ResidentDenseLayout::Q4,
            fixed_dense_experts(&layout, DenseExpertDtype::Bf16),
            &attention_layers(&layout),
            Some(metal_without_dense_bf16()),
        )
        .unwrap_err();
        assert_eq!(
            missing_kernel.stage,
            FlashMoeGraphStage::Cmd3ExpertAndSharedCombine
        );
        assert!(
            missing_kernel
                .reason
                .contains(kernels::DENSE_MMAP_FMA_MATVEC_BF16)
        );
    }

    #[test]
    fn qwen_full_attention_non_q4_dense_resolves_complete_unified_graph() {
        let layout = qwen3_moe_layout(Some(true));
        for dense_layout in [
            ResidentDenseLayout::Bf16,
            ResidentDenseLayout::F16,
            ResidentDenseLayout::F32,
        ] {
            let plan = FlashMoeCapabilityPlan::resolve(
                &layout,
                text_adapter(),
                dense_layout,
                fixed_q4_experts(&layout),
                &attention_layers(&layout),
                Some(full_metal()),
            )
            .unwrap();
            plan.validate_complete().unwrap();
            assert_eq!(plan.dense_layout, dense_layout);
            assert_eq!(
                plan.stage(FlashMoeGraphStage::LmHeadAndSampling)
                    .unwrap()
                    .implementation,
                FlashMoeStageImplementation::MetalResidentLmHeadSampler
            );
        }

        let kernel_error = FlashMoeCapabilityPlan::resolve(
            &layout,
            text_adapter(),
            ResidentDenseLayout::Bf16,
            fixed_q4_experts(&layout),
            &attention_layers(&layout),
            Some(metal_without_dense_bf16()),
        )
        .unwrap_err();
        assert_eq!(
            kernel_error.stage,
            FlashMoeGraphStage::Cmd1AttentionProjections
        );
        assert!(
            kernel_error
                .reason
                .contains(kernels::DENSE_MMAP_FMA_MATVEC_BF16)
        );

        let cmd2_kernel_error = FlashMoeCapabilityPlan::resolve(
            &layout,
            text_adapter(),
            ResidentDenseLayout::Bf16,
            fixed_q4_experts(&layout),
            &attention_layers(&layout),
            Some(metal_without_residual_rms_norm()),
        )
        .unwrap_err();
        assert_eq!(
            cmd2_kernel_error.stage,
            FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection
        );
        assert!(
            cmd2_kernel_error
                .reason
                .contains(kernels::RESIDUAL_ADD_RMS_NORM),
            "{cmd2_kernel_error}"
        );

        let lm_head_kernel_error = FlashMoeCapabilityPlan::resolve(
            &layout,
            text_adapter(),
            ResidentDenseLayout::Bf16,
            fixed_q4_experts(&layout),
            &attention_layers(&layout),
            Some(metal_without_topk_vocab()),
        )
        .unwrap_err();
        assert_eq!(
            lm_head_kernel_error.stage,
            FlashMoeGraphStage::LmHeadAndSampling
        );
        assert!(
            lm_head_kernel_error.reason.contains(kernels::TOPK_VOCAB),
            "{lm_head_kernel_error}"
        );

        let vl_layout = qwen3_vl_layout();
        let vl_plan = FlashMoeCapabilityPlan::resolve(
            &vl_layout,
            qwen_vl_adapter(),
            ResidentDenseLayout::Bf16,
            fixed_q4_experts(&vl_layout),
            &attention_layers(&vl_layout),
            Some(full_metal()),
        )
        .unwrap();
        vl_plan.validate_complete().unwrap();
        assert_eq!(vl_plan.dense_layout, ResidentDenseLayout::Bf16);
        assert_eq!(
            vl_plan
                .stage(FlashMoeGraphStage::LmHeadAndSampling)
                .unwrap()
                .implementation,
            FlashMoeStageImplementation::MetalResidentLmHeadSampler
        );
    }

    #[test]
    fn qwen_text_and_vl_typed_dense_experts_resolve_the_same_graph() {
        for (layout, input_adapter) in [
            (qwen3_moe_layout(Some(true)), text_adapter()),
            (qwen3_vl_layout(), qwen_vl_adapter()),
        ] {
            for dense_layout in [
                ResidentDenseLayout::Bf16,
                ResidentDenseLayout::F16,
                ResidentDenseLayout::F32,
            ] {
                for expert_dtype in [DenseExpertDtype::Bf16, DenseExpertDtype::F16] {
                    let plan = FlashMoeCapabilityPlan::resolve(
                        &layout,
                        input_adapter,
                        dense_layout,
                        fixed_dense_experts(&layout, expert_dtype),
                        &attention_layers(&layout),
                        Some(full_metal()),
                    )
                    .unwrap();

                    plan.validate_complete().unwrap();
                    assert_eq!(plan.family, layout.family);
                    assert_eq!(plan.dense_layout, dense_layout);
                    assert_eq!(plan.stages.len(), FlashMoeGraphStage::ALL.len());
                    assert!(
                        plan.attention_layers
                            .iter()
                            .all(|kind| *kind == QwenMoeLayerKind::FullAttention)
                    );
                    assert_eq!(
                        plan.stage(FlashMoeGraphStage::ActiveExpertReads)
                            .unwrap()
                            .implementation,
                        FlashMoeStageImplementation::ParallelPositionedWholeExpertReads
                    );
                    assert_eq!(
                        plan.stage(FlashMoeGraphStage::Cmd3ExpertAndSharedCombine)
                            .unwrap()
                            .implementation,
                        FlashMoeStageImplementation::MetalTypedExpertResidentSharedCombine
                    );
                }
            }
        }
    }

    #[test]
    fn qwen35_q4_capability_plan_rejects_incomplete_kernel_surface() {
        let layout = qwen35_layout();
        let err = FlashMoeCapabilityPlan::resolve(
            &layout,
            text_adapter(),
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            &attention_layers(&layout),
            Some(MetalRuntimeCapabilities::empty_for_test()),
        )
        .unwrap_err();

        assert_eq!(err.stage, FlashMoeGraphStage::Cmd1AttentionProjections);
        assert!(err.to_string().contains("missing Metal kernels"), "{err}");
    }

    #[test]
    fn qwen_vl_capability_plan_requires_and_resolves_the_typed_adapter() {
        let layout = qwen3_vl_layout();
        let err = FlashMoeCapabilityPlan::resolve(
            &layout,
            text_adapter(),
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            &attention_layers(&layout),
            Some(full_metal()),
        )
        .unwrap_err();

        assert_eq!(err.family, QwenMoeFamily::Qwen3VlMoe);
        assert_eq!(err.stage, FlashMoeGraphStage::TokenPositionInputPreparation);
        assert!(err.to_string().contains("Qwen-VL vision encoder"), "{err}");

        let metadata_error = FlashMoeCapabilityPlan::resolve(
            &layout,
            FlashMoeInputAdapterCapability::QwenVl {
                text_hidden_size: 2048,
                vision_embed_dim: 64,
                vision_depth: 1,
                deepstack_layers: 0,
            },
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            &attention_layers(&layout),
            Some(full_metal()),
        )
        .unwrap_err();
        assert_eq!(
            metadata_error.stage,
            FlashMoeGraphStage::TokenPositionInputPreparation
        );
        assert!(
            metadata_error.reason.contains("text_hidden_size=2048"),
            "{metadata_error}"
        );

        let plan = FlashMoeCapabilityPlan::resolve(
            &layout,
            qwen_vl_adapter(),
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&layout),
            &attention_layers(&layout),
            Some(full_metal()),
        )
        .unwrap();
        assert_eq!(plan.family, QwenMoeFamily::Qwen3VlMoe);
        assert_eq!(plan.input_adapter, qwen_vl_adapter());
        assert_eq!(
            plan.stage(FlashMoeGraphStage::TokenPositionInputPreparation)
                .unwrap()
                .implementation,
            FlashMoeStageImplementation::QwenVlTypedInput
        );
        plan.validate_complete().unwrap();
        let graph =
            crate::inference::flashmoe::scheduler::FlashMoeScheduledGraph::from_capabilities(&plan)
                .unwrap();
        assert_eq!(graph.family(), QwenMoeFamily::Qwen3VlMoe);
        assert_eq!(graph.active_experts(), 3);
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
        assert_eq!(
            plan.expert_storage
                .slot_spec
                .fixed_q4()
                .unwrap()
                .layout
                .expert_bytes,
            2_654_208
        );
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
    fn glm52_capability_binds_mla_sigmoid_routing_and_sparse_expert_boundary() {
        let layout = glm52_layout();
        let plan = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();

        assert_eq!(plan.family, QwenMoeFamily::Glm52);
        assert_eq!(plan.expert_storage.first_expert_layer, 3);
        assert_eq!(plan.routed_expert_scale, 2.5);
        assert_eq!(
            plan.stage(FlashMoeGraphStage::AttentionMath)
                .unwrap()
                .implementation,
            FlashMoeStageImplementation::GlmMlaCpuWeightAbsorption
        );
        assert_eq!(
            plan.stage(FlashMoeGraphStage::RoutingSoftmaxTopK)
                .unwrap()
                .implementation,
            FlashMoeStageImplementation::CpuSigmoidNoAuxTopK
        );
        let graph =
            crate::inference::flashmoe::scheduler::FlashMoeScheduledGraph::from_capabilities(&plan)
                .unwrap();
        assert_eq!(
            graph.build_attention_math(3, 0).unwrap().implementation,
            crate::inference::flashmoe::scheduler::ScheduledAttentionMathImplementation::CpuGlmMlaWeightAbsorption
        );

        for (metal, expected_stage) in [
            (
                metal_without_dense_f32(),
                FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection,
            ),
            (
                metal_without_dense_bf16(),
                FlashMoeGraphStage::LmHeadAndSampling,
            ),
        ] {
            let error = FlashMoeCapabilityPlan::resolve(
                &layout,
                text_adapter(),
                ResidentDenseLayout::Q4,
                fixed_q4_experts(&layout),
                &attention_layers(&layout),
                Some(metal),
            )
            .unwrap_err();
            assert_eq!(error.stage, expected_stage);
        }
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
            text_adapter(),
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&qwen3),
            &attention_layers(&qwen3),
            Some(metal_without_shared_activation()),
        )
        .expect("Qwen3 without shared experts must not require the shared activation kernel");

        let qwen35 = qwen35_layout();
        let shared_error = FlashMoeCapabilityPlan::resolve(
            &qwen35,
            text_adapter(),
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&qwen35),
            &attention_layers(&qwen35),
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
            text_adapter(),
            ResidentDenseLayout::Q4,
            fixed_q4_experts(&qwen35),
            &attention_layers(&qwen35),
            Some(metal_without_linear_conv1d()),
        )
        .unwrap_err();
        assert_eq!(
            linear_error.stage,
            FlashMoeGraphStage::Cmd1AttentionProjections
        );
        assert!(
            linear_error
                .reason
                .contains(kernels::LINEAR_CONV1D_STEP_BF16)
        );
    }

    #[test]
    fn incomplete_capability_plan_reports_the_first_missing_stage() {
        let layout = qwen35_layout();
        let plan = FlashMoeCapabilityPlan {
            family: QwenMoeFamily::Qwen35A17B,
            input_adapter: text_adapter(),
            dense_layout: ResidentDenseLayout::Q4,
            expert_storage: fixed_q4_experts(&layout),
            device: FlashMoeDeviceCapability {
                metal: full_metal(),
            },
            routing: QwenMoeRoutingPlacement::CpuSoftmaxTopK,
            experts_per_layer: 512,
            active_experts: 4,
            routing_weight_normalization: QwenMoeRoutingWeightNormalization::RenormalizeSelected,
            routed_expert_scale: 1.0,
            state_policy: FlashMoeStatePolicy::DeferredGpuNextLayer,
            attention_layers: attention_layers(&layout).into_boxed_slice(),
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
