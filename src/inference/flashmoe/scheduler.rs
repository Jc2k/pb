use super::capabilities::{
    FlashMoeCapabilityPlan, FlashMoeGraphStage, FlashMoeStageCapability,
    FlashMoeStageImplementation, FlashMoeStagePlacement, FlashMoeUnsupportedCapability,
};
use super::experts::{
    DeepSeekGgufExpertSlotSpec, DenseMatvecPayload, EXPERT_SCALE_BIAS_DTYPE_BF16,
    ExpertMlpProjection, ExpertRawPayload, ExpertRawRead, ExpertRawReadResponse, ExpertReadPath,
    ExpertReadWorkerPool, ExpertSlotDescriptor, ExpertSlotStore, ExpertStorageLayout,
    FLASHMOE_EXPERT_IO_POLICY, Q4MatvecPayload, Q4MatvecSource, ReusableExpertBytes,
};
use super::math::routing_softmax_top_k;
#[cfg(test)]
use super::math::{routing_top_k, softmax_in_place, top_k};
use super::model_family::{QwenMoeFamily, QwenMoeLayerKind, QwenMoeRoutingWeightNormalization};
use super::state::{
    FlashMoeCmd1InputState, FlashMoeCmd2InputState, FlashMoeCmd3InputState,
    FlashMoeCmd3OutputState, FlashMoeExpertPhaseOutput, FlashMoeFullAttentionKvState,
    FlashMoeGpuBufferDescriptor, FlashMoeMlaKvState, FlashMoePostAttentionPrepState,
    FlashMoeRoutingOutputSource, FlashMoeRoutingOutputState, FlashMoeStateBufferRole,
    FlashMoeStatePlacement,
};
use super::weights::{
    RouterScoreBatch, RouterScoreProjectionDescriptor, RouterScoreProjectionExecution,
    ScheduledNextNormWeights, SharedExpertPhaseResidentProjections, SharedExpertPhaseShape,
    SharedExpertPhaseWeights,
};
use anyhow::{Context, Result, bail};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub struct FlashMoeScheduledGraph {
    family: QwenMoeFamily,
    experts_per_layer: usize,
    active_experts: usize,
    expert_storage: ExpertStorageLayout,
    routing_weight_normalization: QwenMoeRoutingWeightNormalization,
    routed_expert_scale: f32,
    attention_layers: Box<[ScheduledLayerAttentionImplementation]>,
    stages: Vec<FlashMoeStageCapability>,
}

impl FlashMoeScheduledGraph {
    pub fn from_capabilities(
        capabilities: &FlashMoeCapabilityPlan,
    ) -> Result<Self, FlashMoeUnsupportedCapability> {
        capabilities.validate_complete()?;
        let stages = FlashMoeGraphStage::ALL
            .iter()
            .map(|stage| {
                capabilities.stage(*stage).copied().ok_or_else(|| {
                    FlashMoeUnsupportedCapability::new(
                        capabilities.family,
                        *stage,
                        "graph stage has no declared implementation",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            family: capabilities.family,
            experts_per_layer: capabilities.experts_per_layer,
            active_experts: capabilities.active_experts,
            expert_storage: capabilities.expert_storage.layout,
            routing_weight_normalization: capabilities.routing_weight_normalization,
            routed_expert_scale: capabilities.routed_expert_scale,
            attention_layers: capabilities
                .attention_layers
                .iter()
                .copied()
                .map(ScheduledLayerAttentionImplementation::from)
                .collect(),
            stages,
        })
    }

    pub fn family(&self) -> QwenMoeFamily {
        self.family
    }

    pub fn experts_per_layer(&self) -> usize {
        self.experts_per_layer
    }

    pub fn active_experts(&self) -> usize {
        self.active_experts
    }

    pub fn routing_weight_normalization(&self) -> QwenMoeRoutingWeightNormalization {
        self.routing_weight_normalization
    }

    pub fn routed_expert_scale(&self) -> f32 {
        self.routed_expert_scale
    }

    pub fn stages(&self) -> &[FlashMoeStageCapability] {
        &self.stages
    }

    fn attention_implementation(
        &self,
        layer: usize,
    ) -> Option<ScheduledLayerAttentionImplementation> {
        self.attention_layers.get(layer).copied()
    }

    pub fn stage(&self, stage: FlashMoeGraphStage) -> &FlashMoeStageCapability {
        self.stages
            .iter()
            .find(|capability| capability.stage == stage)
            .expect("scheduled graph contains every FlashMoe stage")
    }

    pub fn active_expert_reads(&self) -> &FlashMoeStageCapability {
        self.stage(FlashMoeGraphStage::ActiveExpertReads)
    }

    pub fn cmd_sequence(&self) -> [&FlashMoeStageCapability; 3] {
        [
            self.stage(FlashMoeGraphStage::DeferredPreviousCmd3),
            self.stage(FlashMoeGraphStage::Cmd1AttentionProjections),
            self.stage(FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection),
        ]
    }

    pub fn declares_scheduler_owned_expert_reads(&self) -> bool {
        self.active_expert_reads().placement == FlashMoeStagePlacement::SchedulerIo
    }

    pub fn build_cmd2_post_attention(
        &self,
        layer: usize,
        active_experts: usize,
        attention: ScheduledCmd2AttentionSource,
        residual: ScheduledCmd2ResidualSource,
    ) -> Result<ScheduledCmd2PostAttention, FlashMoeUnsupportedCapability> {
        let stage = *self.stage(FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection);
        if stage.placement != FlashMoeStagePlacement::Metal {
            return Err(FlashMoeUnsupportedCapability::new(
                self.family,
                stage.stage,
                "CMD2 post-attention stage must be implemented as a declared Metal command",
            ));
        }
        Ok(ScheduledCmd2PostAttention {
            stage,
            layer,
            active_experts,
            attention,
            residual,
        })
    }

    pub fn build_cmd2_submission<TInputs>(
        &self,
        cmd2: ScheduledCmd2PostAttention,
        inputs: TInputs,
    ) -> Result<ScheduledCmd2Submission<TInputs>>
    where
        TInputs: ScheduledCmd2Inputs,
    {
        let expected_stage = *self.stage(FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection);
        if cmd2.stage != expected_stage {
            bail!(
                "FlashMoe scheduled CMD2 descriptor stage {:?} does not match scheduled graph CMD2 stage {:?}",
                cmd2.stage,
                expected_stage
            );
        }
        ScheduledCmd2Submission::new(cmd2, inputs)
    }

    pub fn build_cmd2_command(
        &self,
        layer: usize,
        active_experts: usize,
        inputs: ScheduledCmd2PhaseInputs,
    ) -> Result<ScheduledCmd2Command<ScheduledCmd2PhaseInputs>> {
        let cmd2 = self.build_cmd2_post_attention(
            layer,
            active_experts,
            inputs.scheduled_cmd2_attention_source(),
            inputs.scheduled_cmd2_residual_source(),
        )?;
        Ok(self
            .build_cmd2_submission(cmd2, inputs)?
            .into_cmd2_command())
    }

    pub fn build_attention_math(
        &self,
        layer: usize,
        position: usize,
    ) -> Result<ScheduledAttentionMath, FlashMoeUnsupportedCapability> {
        let stage = *self.stage(FlashMoeGraphStage::AttentionMath);
        let implementation = ScheduledAttentionMathImplementation::resolve(self.family, stage)?;
        Ok(ScheduledAttentionMath {
            stage,
            layer,
            position,
            implementation,
        })
    }

    pub fn build_routing_topk(
        &self,
        layer: usize,
        experts: usize,
        active_experts: usize,
        source: ScheduledRoutingCandidateSource,
    ) -> Result<ScheduledRoutingTopK, FlashMoeUnsupportedCapability> {
        let stage = *self.stage(FlashMoeGraphStage::RoutingSoftmaxTopK);
        if stage.placement != FlashMoeStagePlacement::CpuDeclared {
            return Err(FlashMoeUnsupportedCapability::new(
                self.family,
                stage.stage,
                "routing softmax/topK stage must be implemented as a declared CPU routing command",
            ));
        }
        Ok(ScheduledRoutingTopK {
            stage,
            layer,
            experts,
            active_experts,
            source,
        })
    }

    pub(crate) fn build_router_score_projection(
        &self,
        layer: usize,
        experts: usize,
        active_experts: usize,
        projection: Option<RouterScoreProjectionDescriptor>,
        hidden_width: usize,
    ) -> Result<ScheduledRouterScoreProjectionCommand, FlashMoeUnsupportedCapability> {
        let routing = self.build_routing_topk(
            layer,
            experts,
            active_experts,
            ScheduledRoutingCandidateSource::CpuRouterScores,
        )?;
        routing
            .build_score_projection_command(projection, hidden_width)
            .map_err(|_| {
                FlashMoeUnsupportedCapability::new(
                    self.family,
                    routing.stage.stage,
                    "invalid scheduled router score projection",
                )
            })
    }

    pub fn build_cmd1_attention_projections(
        &self,
        layer: usize,
        input: ScheduledCmd1InputSource,
    ) -> Result<ScheduledCmd1AttentionProjections, FlashMoeUnsupportedCapability> {
        let stage = *self.stage(FlashMoeGraphStage::Cmd1AttentionProjections);
        if stage.placement != FlashMoeStagePlacement::Metal {
            return Err(FlashMoeUnsupportedCapability::new(
                self.family,
                stage.stage,
                "CMD1 attention projection stage must be implemented as a declared Metal command",
            ));
        }
        Ok(ScheduledCmd1AttentionProjections {
            stage,
            layer,
            input,
        })
    }

    pub fn build_cmd1_submission<TInput>(
        &self,
        cmd1: ScheduledCmd1AttentionProjections,
        input: TInput,
    ) -> Result<ScheduledCmd1Submission<TInput>>
    where
        TInput: ScheduledCmd1Input,
    {
        let expected_stage = *self.stage(FlashMoeGraphStage::Cmd1AttentionProjections);
        if cmd1.stage != expected_stage {
            bail!(
                "FlashMoe scheduled CMD1 descriptor stage {:?} does not match scheduled graph CMD1 stage {:?}",
                cmd1.stage,
                expected_stage
            );
        }
        ScheduledCmd1Submission::new(cmd1, input)
    }

    pub fn build_cmd3_expert_phase(
        &self,
        layer: usize,
        expert_count: usize,
        input: ScheduledCmd3InputSource,
        shared: ScheduledSharedExpertSource,
        next_norm: ScheduledNextNormSource,
    ) -> Result<ScheduledCmd3ExpertPhase, FlashMoeUnsupportedCapability> {
        let stage = *self.stage(FlashMoeGraphStage::Cmd3ExpertAndSharedCombine);
        if stage.placement != FlashMoeStagePlacement::Metal {
            return Err(FlashMoeUnsupportedCapability::new(
                self.family,
                stage.stage,
                "CMD3 expert/shared combine must be implemented as a declared Metal command",
            ));
        }
        if stage.implementation
            == FlashMoeStageImplementation::MetalTypedExpertResidentSharedCombine
            && shared == ScheduledSharedExpertSource::DenseCpuWeights
        {
            return Err(FlashMoeUnsupportedCapability::new(
                self.family,
                stage.stage,
                "the typed active-expert CMD3 implementation requires resident shared projections or the declared no-shared source; dense CPU shared weights are not a declared graph-stage implementation",
            ));
        }
        Ok(ScheduledCmd3ExpertPhase {
            stage,
            layer,
            expert_count,
            expert_storage: self.expert_storage,
            input,
            shared,
            shared_descriptor: None,
            next_norm,
        })
    }

    pub fn build_cmd3_expert_phase_with_shared_descriptor(
        &self,
        layer: usize,
        expert_count: usize,
        input: ScheduledCmd3InputSource,
        shared: ScheduledSharedExpertDescriptor,
        next_norm: ScheduledNextNormSource,
    ) -> Result<ScheduledCmd3ExpertPhase, FlashMoeUnsupportedCapability> {
        let mut cmd3 =
            self.build_cmd3_expert_phase(layer, expert_count, input, shared.source, next_norm)?;
        cmd3.shared_descriptor = Some(shared);
        Ok(cmd3)
    }

    pub fn build_cmd3_expert_phase_from_descriptors(
        &self,
        layer: usize,
        expert_count: usize,
        input: ScheduledCmd3InputSource,
        shared: ScheduledSharedExpertDescriptor,
        next_norm_weights: ScheduledNextNormWeights<'_>,
    ) -> Result<ScheduledCmd3ExpertPhase, FlashMoeUnsupportedCapability> {
        self.build_cmd3_expert_phase_with_shared_descriptor(
            layer,
            expert_count,
            input,
            shared,
            ScheduledNextNormSource::from_weights(next_norm_weights),
        )
    }

    pub fn build_cmd3_submission<'a, TExpert, TInput, TShared>(
        &self,
        position: usize,
        cmd3: ScheduledCmd3ExpertPhase,
        scheduled: &'a ScheduledExpertSet<TExpert>,
        input: TInput,
        shared: TShared,
        next_norm_weights: ScheduledNextNormWeights<'a>,
    ) -> Result<ScheduledCmd3Submission<'a, TExpert, TInput, TShared>>
    where
        TExpert: ScheduledCmd3Expert,
        TInput: ScheduledCmd3Input,
        TShared: ScheduledSharedExpert,
    {
        let expected_stage = *self.stage(FlashMoeGraphStage::Cmd3ExpertAndSharedCombine);
        if cmd3.stage != expected_stage {
            bail!(
                "FlashMoe scheduled CMD3 descriptor stage {:?} does not match scheduled graph CMD3 stage {:?}",
                cmd3.stage,
                expected_stage
            );
        }
        ScheduledCmd3Submission::new(position, cmd3, scheduled, input, shared, next_norm_weights)
    }

    pub fn build_cmd3_command_from_descriptors<'a, TExpert, TInput, TShared>(
        &self,
        position: usize,
        scheduled: &'a ScheduledExpertSet<TExpert>,
        input: TInput,
        shared: TShared,
        next_norm_weights: ScheduledNextNormWeights<'a>,
    ) -> Result<ScheduledCmd3Command<'a, TExpert, TInput, TShared>>
    where
        TExpert: ScheduledCmd3Expert + ScheduledCmd3ExpertPayload,
        TInput: ScheduledCmd3Input,
        TShared: ScheduledSharedExpert,
    {
        let cmd3 = self.build_cmd3_expert_phase_from_descriptors(
            scheduled.layer,
            scheduled.len(),
            input.scheduled_cmd3_input_source(),
            shared.scheduled_shared_expert_descriptor()?,
            next_norm_weights,
        )?;
        self.build_cmd3_submission(position, cmd3, scheduled, input, shared, next_norm_weights)?
            .into_cmd3_command()
    }
}

#[derive(Debug)]
pub(crate) struct FlashMoeExecutionScheduler {
    graph: FlashMoeScheduledGraph,
    expert_reads: ScheduledExpertReadCoordinator,
}

impl FlashMoeExecutionScheduler {
    pub(crate) fn new(
        graph: FlashMoeScheduledGraph,
        expert_store: ExpertSlotStore,
    ) -> Result<Self> {
        if graph.attention_layers.is_empty() {
            bail!(
                "FlashMoe execution scheduler requires a resolved attention implementation for every layer"
            );
        }
        let expert_reads = ScheduledExpertReadCoordinator::new_with_routing_policy(
            expert_store,
            graph.routing_weight_normalization(),
            graph.routed_expert_scale(),
        );
        Ok(Self {
            graph,
            expert_reads,
        })
    }

    pub(crate) fn begin_layer(
        &self,
        position: usize,
        layer: usize,
        layers: usize,
        active_experts: usize,
        previous: ScheduledPreviousCmd3Handoff,
        allow_deferred_output: bool,
    ) -> Result<ScheduledLayerCmd1> {
        if layers == 0 || layer >= layers {
            bail!("FlashMoe scheduled layer {layer} is outside resolved layer count {layers}");
        }
        if layers != self.graph.attention_layers.len() {
            bail!(
                "FlashMoe scheduled layer count {layers} does not match resolved attention implementation count {}",
                self.graph.attention_layers.len()
            );
        }
        if active_experts == 0 {
            bail!("FlashMoe scheduled layer {layer} requires at least one active expert");
        }
        previous.validate(layer)?;
        Ok(ScheduledLayerCmd1 {
            identity: ScheduledLayerIdentity {
                position,
                layer,
                active_experts,
                attention: self
                    .graph
                    .attention_implementation(layer)
                    .expect("validated scheduled layer index"),
                output_handoff: if allow_deferred_output && layer + 1 < layers {
                    ScheduledCmd3OutputHandoff::DeferredToNextLayer
                } else {
                    ScheduledCmd3OutputHandoff::CompleteHere
                },
            },
            previous,
        })
    }

    pub(crate) fn begin_resolved_layer(
        &self,
        position: usize,
        layer: usize,
        layers: usize,
        previous: ScheduledPreviousCmd3Handoff,
        allow_deferred_output: bool,
    ) -> Result<ScheduledLayerCmd1> {
        self.begin_layer(
            position,
            layer,
            layers,
            self.graph.active_experts(),
            previous,
            allow_deferred_output,
        )
    }

    pub(crate) fn experts_per_layer(&self) -> usize {
        self.graph.experts_per_layer()
    }

    pub(crate) fn active_experts(&self) -> usize {
        self.graph.active_experts()
    }

    pub(crate) fn resolve_cmd1(
        &self,
        layer: usize,
        input: ScheduledCmd1InputSource,
        input_state: FlashMoeCmd1InputState,
    ) -> Result<ScheduledCmd1ResolvedCommand<ScheduledCmd1InputSource>> {
        self.graph
            .build_cmd1_submission(
                self.graph.build_cmd1_attention_projections(layer, input)?,
                input,
            )?
            .into_cmd1_command()
            .into_resolved_command(input_state)
    }

    pub(crate) fn resolve_attention_math(
        &self,
        layer: usize,
        position: usize,
    ) -> Result<ScheduledAttentionMath, FlashMoeUnsupportedCapability> {
        self.graph.build_attention_math(layer, position)
    }

    pub(crate) fn resolve_cmd2(
        &self,
        layer: usize,
        active_experts: usize,
        inputs: ScheduledCmd2PhaseInputs,
    ) -> Result<ScheduledCmd2Command<ScheduledCmd2PhaseInputs>> {
        self.graph.build_cmd2_command(layer, active_experts, inputs)
    }

    pub(crate) fn routing_from_post_attention_prep(
        &self,
        cmd2: &ScheduledCmd2Command<ScheduledCmd2PhaseInputs>,
        state: FlashMoePostAttentionPrepState,
        routes: &[(usize, f32)],
    ) -> Result<ScheduledRoutingCommand> {
        cmd2.command_from_post_attention_prep_routes(&self.graph, state, routes)
    }

    /// DeepSeek's router is already fully selected by its typed graph (token
    /// hash for layers 0..2, biased sqrt-softplus top-k afterwards).  This
    /// enters the same scheduler-owned positioned-read lifecycle used by every
    /// other FlashMoe family and returns reusable whole-expert slots.
    pub(crate) fn read_preselected_experts(
        &mut self,
        layer: usize,
        routes: &[(usize, f32)],
    ) -> Result<ScheduledExpertSet<Arc<ScheduledExpertSlot>>> {
        let routing = self
            .graph
            .build_routing_topk(
                layer,
                self.graph.experts_per_layer(),
                self.graph.active_experts(),
                ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
            )?
            .command_from_preselected(routes)?;
        let pending = self.expert_reads.issue_routing_command(&routing)?;
        self.expert_reads.finish_routes(pending)
    }

    pub(crate) fn resolve_router_score_projection(
        &self,
        layer: usize,
        projection: Option<RouterScoreProjectionDescriptor>,
        hidden_width: usize,
    ) -> Result<ScheduledRouterScoreProjectionCommand, FlashMoeUnsupportedCapability> {
        self.graph.build_router_score_projection(
            layer,
            self.graph.experts_per_layer(),
            self.graph.active_experts(),
            projection,
            hidden_width,
        )
    }

    fn issue_cmd3(&mut self, routing: &ScheduledRoutingCommand) -> Result<PendingScheduledCmd3> {
        let before = self.expert_reads.snapshot();
        let issue_started = Instant::now();
        let pending = self.expert_reads.issue_routing_command(routing)?;
        Ok(PendingScheduledCmd3 {
            before,
            pending,
            issue_elapsed: issue_started.elapsed(),
        })
    }

    fn finish_cmd3<TInput, TShared, TSubmission>(
        &mut self,
        pending: PendingScheduledCmd3,
        position: usize,
        input: TInput,
        shared: TShared,
        next_norm_weights: ScheduledNextNormWeights<'_>,
        submit: impl FnOnce(
            ScheduledCmd3Command<'_, Arc<ScheduledExpertSlot>, TInput, TShared>,
        ) -> Result<TSubmission>,
    ) -> Result<ScheduledCmd3Execution<TSubmission>>
    where
        TInput: ScheduledCmd3Input,
        TShared: ScheduledSharedExpert,
    {
        let finish_started = Instant::now();
        let scheduled = self.expert_reads.finish_routes(pending.pending)?;
        let expert_io_elapsed = pending.issue_elapsed + finish_started.elapsed();
        let expert_delta = self
            .expert_reads
            .snapshot()
            .saturating_delta(pending.before);
        let expert_mixes = scheduled
            .experts
            .iter()
            .zip(scheduled.weights.iter().copied())
            .map(|(expert, weight)| (expert.mix_hash(), weight))
            .collect();
        let submit_started = Instant::now();
        let command = self.graph.build_cmd3_command_from_descriptors(
            position,
            &scheduled,
            input,
            shared,
            next_norm_weights,
        )?;
        let submission = submit(command)?;
        let submit_elapsed = submit_started.elapsed();
        Ok(ScheduledCmd3Execution {
            submission,
            expert_delta,
            expert_mixes,
            expert_io_elapsed,
            submit_elapsed,
        })
    }

    pub(crate) fn snapshot(&self) -> ExpertSchedulerSnapshot {
        self.expert_reads.snapshot()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScheduledPreviousCmd3Handoff {
    InitialToken {
        hidden_len: usize,
    },
    CpuVisible {
        previous_layer: usize,
        hidden_len: usize,
    },
    DeferredGpu {
        previous_layer: usize,
        hidden: FlashMoeGpuBufferDescriptor,
        next_normed: FlashMoeGpuBufferDescriptor,
    },
}

impl ScheduledPreviousCmd3Handoff {
    pub(crate) const fn initial(hidden_len: usize) -> Self {
        Self::InitialToken { hidden_len }
    }

    pub(crate) const fn cpu_visible(previous_layer: usize, hidden_len: usize) -> Self {
        Self::CpuVisible {
            previous_layer,
            hidden_len,
        }
    }

    pub(crate) const fn deferred_gpu(
        previous_layer: usize,
        hidden: FlashMoeGpuBufferDescriptor,
        next_normed: FlashMoeGpuBufferDescriptor,
    ) -> Self {
        Self::DeferredGpu {
            previous_layer,
            hidden,
            next_normed,
        }
    }

    fn validate(self, layer: usize) -> Result<()> {
        match self {
            Self::InitialToken { hidden_len } => {
                if layer != 0 {
                    bail!(
                        "FlashMoe scheduled layer {layer} cannot start from an initial-token handoff"
                    );
                }
                if hidden_len == 0 {
                    bail!("FlashMoe scheduled initial-token handoff has empty hidden state");
                }
            }
            Self::CpuVisible {
                previous_layer,
                hidden_len,
            } => {
                if previous_layer.checked_add(1) != Some(layer) {
                    bail!(
                        "FlashMoe scheduled CPU handoff from layer {previous_layer} does not feed layer {layer}"
                    );
                }
                if hidden_len == 0 {
                    bail!("FlashMoe scheduled CPU handoff has empty hidden state");
                }
            }
            Self::DeferredGpu {
                previous_layer,
                hidden,
                next_normed,
            } => {
                if previous_layer.checked_add(1) != Some(layer) {
                    bail!(
                        "FlashMoe scheduled deferred GPU handoff from layer {previous_layer} does not feed layer {layer}"
                    );
                }
                if !hidden.is_declared_graph_state()
                    || hidden.role() != FlashMoeStateBufferRole::Hidden
                    || !next_normed.is_declared_graph_state()
                    || next_normed.role() != FlashMoeStateBufferRole::NextLayerNormed
                    || hidden.len() != next_normed.len()
                {
                    bail!(
                        "FlashMoe scheduled deferred GPU handoff requires equal non-empty Hidden and NextLayerNormed buffers"
                    );
                }
            }
        }
        Ok(())
    }

    fn cmd1_source(self) -> ScheduledCmd1InputSource {
        match self {
            Self::InitialToken { .. } | Self::CpuVisible { .. } => {
                ScheduledCmd1InputSource::CpuNormedHidden
            }
            Self::DeferredGpu { .. } => ScheduledCmd1InputSource::DeferredMetalNextNormed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScheduledLayerIdentity {
    position: usize,
    layer: usize,
    active_experts: usize,
    attention: ScheduledLayerAttentionImplementation,
    output_handoff: ScheduledCmd3OutputHandoff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScheduledLayerAttentionImplementation {
    FullAttentionCpuKv,
    FusedLinearAttentionMetal,
}

impl From<QwenMoeLayerKind> for ScheduledLayerAttentionImplementation {
    fn from(value: QwenMoeLayerKind) -> Self {
        match value {
            QwenMoeLayerKind::FullAttention => Self::FullAttentionCpuKv,
            QwenMoeLayerKind::LinearAttention => Self::FusedLinearAttentionMetal,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ScheduledLayerCmd1 {
    identity: ScheduledLayerIdentity,
    previous: ScheduledPreviousCmd3Handoff,
}

impl ScheduledLayerCmd1 {
    pub(crate) fn attention_implementation(&self) -> ScheduledLayerAttentionImplementation {
        self.identity.attention
    }

    pub(crate) fn resolve(
        self,
        scheduler: &FlashMoeExecutionScheduler,
        input: ScheduledCmd1InputSource,
        input_state: FlashMoeCmd1InputState,
    ) -> Result<(
        ScheduledCmd1ResolvedCommand<ScheduledCmd1InputSource>,
        ScheduledLayerCmd2,
    )> {
        let expected = self.previous.cmd1_source();
        if input != expected {
            bail!(
                "FlashMoe scheduled layer {} previous CMD3 handoff requires CMD1 input {:?}, got {:?}",
                self.identity.layer,
                expected,
                input
            );
        }
        let cmd1 = scheduler.resolve_cmd1(self.identity.layer, input, input_state)?;
        Ok((
            cmd1,
            ScheduledLayerCmd2 {
                identity: self.identity,
            },
        ))
    }
}

#[derive(Debug)]
pub(crate) struct ScheduledLayerCmd2 {
    identity: ScheduledLayerIdentity,
}

impl ScheduledLayerCmd2 {
    pub(crate) fn resolve(
        self,
        scheduler: &FlashMoeExecutionScheduler,
        inputs: ScheduledCmd2PhaseInputs,
    ) -> Result<(
        ScheduledCmd2Command<ScheduledCmd2PhaseInputs>,
        ScheduledLayerRouting,
    )> {
        let cmd2 =
            scheduler.resolve_cmd2(self.identity.layer, self.identity.active_experts, inputs)?;
        Ok((
            cmd2,
            ScheduledLayerRouting {
                identity: self.identity,
            },
        ))
    }
}

#[derive(Debug)]
pub(crate) struct ScheduledLayerRouting {
    identity: ScheduledLayerIdentity,
}

impl ScheduledLayerRouting {
    pub(crate) fn resolve(self, routing: &ScheduledRoutingCommand) -> Result<ScheduledLayerRouted> {
        routing.validate_for_active_expert_issue()?;
        if routing.layer != self.identity.layer
            || routing.active_experts != self.identity.active_experts
        {
            bail!(
                "FlashMoe scheduled routing layer {} K={} does not match layer transaction {} K={}",
                routing.layer,
                routing.active_experts,
                self.identity.layer,
                self.identity.active_experts
            );
        }
        Ok(ScheduledLayerRouted {
            identity: self.identity,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ScheduledLayerRouted {
    identity: ScheduledLayerIdentity,
}

impl ScheduledLayerRouted {
    pub(crate) fn issue_cmd3(
        self,
        scheduler: &mut FlashMoeExecutionScheduler,
        routing: &ScheduledRoutingCommand,
    ) -> Result<ScheduledLayerPendingCmd3> {
        let pending = scheduler.issue_cmd3(routing)?;
        Ok(ScheduledLayerPendingCmd3 {
            identity: self.identity,
            pending,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ScheduledLayerPendingCmd3 {
    identity: ScheduledLayerIdentity,
    pending: PendingScheduledCmd3,
}

impl ScheduledLayerPendingCmd3 {
    pub(crate) fn finish<TInput, TShared, TSubmission>(
        self,
        scheduler: &mut FlashMoeExecutionScheduler,
        input: TInput,
        shared: TShared,
        next_norm_weights: ScheduledNextNormWeights<'_>,
        submit: impl FnOnce(
            ScheduledCmd3Command<'_, Arc<ScheduledExpertSlot>, TInput, TShared>,
        ) -> Result<TSubmission>,
    ) -> Result<ScheduledLayerExecution<TSubmission>>
    where
        TInput: ScheduledCmd3Input,
        TShared: ScheduledSharedExpert,
    {
        let cmd3 = scheduler.finish_cmd3(
            self.pending,
            self.identity.position,
            input,
            shared,
            next_norm_weights,
            submit,
        )?;
        Ok(ScheduledLayerExecution {
            cmd3,
            output_handoff: self.identity.output_handoff,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScheduledCmd3OutputHandoff {
    CompleteHere,
    DeferredToNextLayer,
}

#[derive(Debug)]
pub(crate) struct ScheduledLayerExecution<TSubmission> {
    pub(crate) cmd3: ScheduledCmd3Execution<TSubmission>,
    pub(crate) output_handoff: ScheduledCmd3OutputHandoff,
}

#[derive(Debug)]
pub(crate) struct PendingScheduledCmd3 {
    before: ExpertSchedulerSnapshot,
    pending: PendingScheduledExpertSet<ExpertRawReadResponse>,
    issue_elapsed: Duration,
}

#[derive(Debug)]
pub(crate) struct ScheduledCmd3Execution<TSubmission> {
    pub(crate) submission: TSubmission,
    pub(crate) expert_delta: ExpertSchedulerSnapshot,
    pub(crate) expert_mixes: Vec<(u64, f32)>,
    pub(crate) expert_io_elapsed: Duration,
    pub(crate) submit_elapsed: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledCmd1InputSource {
    CpuNormedHidden,
    DeferredMetalNextNormed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledCmd1AttentionProjections {
    pub stage: FlashMoeStageCapability,
    pub layer: usize,
    pub input: ScheduledCmd1InputSource,
}

pub trait ScheduledCmd1Input {
    fn scheduled_cmd1_input_source(&self) -> ScheduledCmd1InputSource;
}

impl ScheduledCmd1Input for ScheduledCmd1InputSource {
    fn scheduled_cmd1_input_source(&self) -> ScheduledCmd1InputSource {
        *self
    }
}

#[derive(Debug)]
pub struct ScheduledCmd1Submission<TInput> {
    pub cmd1: ScheduledCmd1AttentionProjections,
    pub input: TInput,
}

impl<TInput> ScheduledCmd1Submission<TInput>
where
    TInput: ScheduledCmd1Input,
{
    pub fn new(cmd1: ScheduledCmd1AttentionProjections, input: TInput) -> Result<Self> {
        if cmd1.input != input.scheduled_cmd1_input_source() {
            bail!(
                "FlashMoe scheduled CMD1 input {:?} does not match submitted input {:?}",
                cmd1.input,
                input.scheduled_cmd1_input_source()
            );
        }
        Ok(Self { cmd1, input })
    }

    pub(crate) fn into_cmd1_command(self) -> ScheduledCmd1Command<TInput> {
        ScheduledCmd1Command {
            cmd1: self.cmd1,
            layer: self.cmd1.layer,
            input: self.input,
        }
    }
}

#[derive(Debug)]
pub struct ScheduledCmd1Command<TInput> {
    pub cmd1: ScheduledCmd1AttentionProjections,
    pub layer: usize,
    pub input: TInput,
}

impl<TInput> ScheduledCmd1Command<TInput>
where
    TInput: ScheduledCmd1Input,
{
    fn validate_input_state(&self, state: FlashMoeCmd1InputState) -> Result<()> {
        if !state.is_declared_graph_state() {
            bail!("FlashMoe scheduled CMD1 input is not declared graph state");
        }
        if state.layer() != self.layer {
            bail!(
                "FlashMoe scheduled CMD1 layer {} does not match input state layer {}",
                self.layer,
                state.layer()
            );
        }
        match self.input.scheduled_cmd1_input_source() {
            ScheduledCmd1InputSource::CpuNormedHidden => {
                if state.role() != FlashMoeStateBufferRole::Normed
                    || state.placement() != FlashMoeStatePlacement::CpuVisible
                {
                    bail!(
                        "FlashMoe scheduled CMD1 CPU input requires CpuVisible Normed state, got {:?} {:?}",
                        state.placement(),
                        state.role()
                    );
                }
            }
            ScheduledCmd1InputSource::DeferredMetalNextNormed => {
                if state.role() != FlashMoeStateBufferRole::NextLayerNormed
                    || state.placement() != FlashMoeStatePlacement::GpuResident
                {
                    bail!(
                        "FlashMoe scheduled CMD1 deferred input requires GpuResident NextLayerNormed state, got {:?} {:?}",
                        state.placement(),
                        state.role()
                    );
                }
            }
        }
        Ok(())
    }

    pub(crate) fn into_resolved_command(
        self,
        input_state: FlashMoeCmd1InputState,
    ) -> Result<ScheduledCmd1ResolvedCommand<TInput>> {
        self.validate_input_state(input_state)?;
        Ok(ScheduledCmd1ResolvedCommand {
            cmd1: self.cmd1,
            layer: self.layer,
            input: self.input,
            input_state,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ScheduledCmd1ResolvedCommand<TInput> {
    pub(crate) cmd1: ScheduledCmd1AttentionProjections,
    pub(crate) layer: usize,
    pub(crate) input: TInput,
    pub(crate) input_state: FlashMoeCmd1InputState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledAttentionMathImplementation {
    CpuKvCache,
    CpuGlmMlaWeightAbsorption,
    MetalQ4GlmMlaAbsorbedAttention,
}

impl ScheduledAttentionMathImplementation {
    fn resolve(
        family: QwenMoeFamily,
        stage: FlashMoeStageCapability,
    ) -> Result<Self, FlashMoeUnsupportedCapability> {
        match (stage.placement, stage.implementation) {
            (
                FlashMoeStagePlacement::CpuDeclared,
                FlashMoeStageImplementation::QwenFullAttentionCpuKv,
            ) => Ok(Self::CpuKvCache),
            (
                FlashMoeStagePlacement::CpuDeclared,
                FlashMoeStageImplementation::GlmMlaCpuWeightAbsorption,
            ) => Ok(Self::CpuGlmMlaWeightAbsorption),
            (
                FlashMoeStagePlacement::Metal,
                FlashMoeStageImplementation::GlmMlaMetalQ4AbsorbedAttention,
            ) => Ok(Self::MetalQ4GlmMlaAbsorbedAttention),
            _ => Err(FlashMoeUnsupportedCapability::new(
                family,
                stage.stage,
                format!(
                    "attention implementation {} at {:?} has no scheduled executor",
                    stage.implementation, stage.placement
                ),
            )),
        }
    }

    fn kv_placement(self) -> FlashMoeStatePlacement {
        match self {
            Self::CpuKvCache
            | Self::CpuGlmMlaWeightAbsorption
            | Self::MetalQ4GlmMlaAbsorbedAttention => FlashMoeStatePlacement::CpuVisible,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledAttentionMath {
    pub stage: FlashMoeStageCapability,
    pub layer: usize,
    pub position: usize,
    pub implementation: ScheduledAttentionMathImplementation,
}

impl ScheduledAttentionMath {
    pub(crate) fn resolve_kv_state(
        self,
        state: FlashMoeFullAttentionKvState,
    ) -> Result<ScheduledAttentionMathOutput> {
        if self.implementation != ScheduledAttentionMathImplementation::CpuKvCache {
            bail!(
                "FlashMoe scheduled attention implementation {:?} requires compressed MLA KV state",
                self.implementation
            );
        }
        if !state.is_declared_graph_state() {
            bail!("FlashMoe scheduled attention KV state is not declared graph state");
        }
        if state.layer() != self.layer {
            bail!(
                "FlashMoe scheduled attention layer {} does not match KV state layer {}",
                self.layer,
                state.layer()
            );
        }
        if state.position() != self.position {
            bail!(
                "FlashMoe scheduled attention position {} does not match KV state position {}",
                self.position,
                state.position()
            );
        }
        if state.placement() != self.implementation.kv_placement() {
            bail!(
                "FlashMoe scheduled attention implementation {:?} requires {:?} KV state, got {:?}",
                self.implementation,
                self.implementation.kv_placement(),
                state.placement()
            );
        }
        Ok(ScheduledAttentionMathOutput {
            attention: self,
            state,
        })
    }

    pub(crate) fn resolve_mla_kv_state(
        self,
        state: FlashMoeMlaKvState,
    ) -> Result<ScheduledMlaAttentionMathOutput> {
        if !matches!(
            self.implementation,
            ScheduledAttentionMathImplementation::CpuGlmMlaWeightAbsorption
                | ScheduledAttentionMathImplementation::MetalQ4GlmMlaAbsorbedAttention
        ) {
            bail!(
                "FlashMoe scheduled attention implementation {:?} does not accept compressed MLA KV state",
                self.implementation
            );
        }
        if !state.is_declared_graph_state() {
            bail!("FlashMoe scheduled MLA KV state is not declared graph state");
        }
        if state.layer() != self.layer {
            bail!(
                "FlashMoe scheduled attention layer {} does not match MLA KV state layer {}",
                self.layer,
                state.layer()
            );
        }
        if state.position() != self.position {
            bail!(
                "FlashMoe scheduled attention position {} does not match MLA KV state position {}",
                self.position,
                state.position()
            );
        }
        if state.placement() != self.implementation.kv_placement() {
            bail!(
                "FlashMoe scheduled attention implementation {:?} requires {:?} MLA KV state, got {:?}",
                self.implementation,
                self.implementation.kv_placement(),
                state.placement()
            );
        }
        Ok(ScheduledMlaAttentionMathOutput {
            attention: self,
            state,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledAttentionMathOutput {
    pub attention: ScheduledAttentionMath,
    state: FlashMoeFullAttentionKvState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledMlaAttentionMathOutput {
    pub attention: ScheduledAttentionMath,
    state: FlashMoeMlaKvState,
}

impl ScheduledMlaAttentionMathOutput {
    pub(crate) fn implementation(self) -> ScheduledAttentionMathImplementation {
        self.attention.implementation
    }

    #[cfg(test)]
    fn state(&self) -> FlashMoeMlaKvState {
        self.state
    }
}

impl ScheduledAttentionMathOutput {
    pub(crate) fn implementation(self) -> ScheduledAttentionMathImplementation {
        self.attention.implementation
    }

    #[cfg(test)]
    fn state(&self) -> FlashMoeFullAttentionKvState {
        self.state
    }

    pub(crate) fn validate_execution_state(
        self,
        layer: usize,
        position: usize,
        kv_width: usize,
    ) -> Result<Self> {
        if self.state.layer() != layer || self.state.position() != position {
            bail!(
                "FlashMoe scheduled attention KV state layer {} position {} does not match execution layer {layer} position {position}",
                self.state.layer(),
                self.state.position()
            );
        }
        if self.state.width() != kv_width {
            bail!(
                "FlashMoe scheduled attention KV width {} does not match execution width {}",
                self.state.width(),
                kv_width
            );
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledCmd2AttentionSource {
    CpuAttentionValues,
    MetalAttentionValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledCmd2ResidualSource {
    CpuHidden,
    MetalBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledCmd2AttentionInput {
    CpuValues { len: usize },
    MetalValues { len: usize },
}

impl ScheduledCmd2AttentionInput {
    pub const fn cpu_values(len: usize) -> Self {
        Self::CpuValues { len }
    }

    pub const fn metal_values(len: usize) -> Self {
        Self::MetalValues { len }
    }

    const fn source(self) -> ScheduledCmd2AttentionSource {
        match self {
            Self::CpuValues { .. } => ScheduledCmd2AttentionSource::CpuAttentionValues,
            Self::MetalValues { .. } => ScheduledCmd2AttentionSource::MetalAttentionValues,
        }
    }

    const fn len(self) -> usize {
        match self {
            Self::CpuValues { len } | Self::MetalValues { len } => len,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledCmd2ResidualInput {
    CpuHidden { len: usize },
    MetalBuffer { len: usize },
}

impl ScheduledCmd2ResidualInput {
    pub const fn cpu_hidden(len: usize) -> Self {
        Self::CpuHidden { len }
    }

    pub const fn metal_buffer(len: usize) -> Self {
        Self::MetalBuffer { len }
    }

    const fn source(self) -> ScheduledCmd2ResidualSource {
        match self {
            Self::CpuHidden { .. } => ScheduledCmd2ResidualSource::CpuHidden,
            Self::MetalBuffer { .. } => ScheduledCmd2ResidualSource::MetalBuffer,
        }
    }

    const fn len(self) -> usize {
        match self {
            Self::CpuHidden { len } | Self::MetalBuffer { len } => len,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledCmd2PostAttention {
    pub stage: FlashMoeStageCapability,
    pub layer: usize,
    pub active_experts: usize,
    pub attention: ScheduledCmd2AttentionSource,
    pub residual: ScheduledCmd2ResidualSource,
}

pub trait ScheduledCmd2Inputs {
    fn scheduled_cmd2_attention_source(&self) -> ScheduledCmd2AttentionSource;
    fn scheduled_cmd2_residual_source(&self) -> ScheduledCmd2ResidualSource;
    fn scheduled_cmd2_input_state(&self, layer: usize) -> FlashMoeCmd2InputState;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledCmd2PhaseInputs {
    attention: ScheduledCmd2AttentionSource,
    residual: ScheduledCmd2ResidualSource,
    attention_len: usize,
    residual_len: usize,
}

impl ScheduledCmd2PhaseInputs {
    pub const fn new(
        attention: ScheduledCmd2AttentionSource,
        residual: ScheduledCmd2ResidualSource,
        attention_len: usize,
        residual_len: usize,
    ) -> Self {
        Self {
            attention,
            residual,
            attention_len,
            residual_len,
        }
    }

    pub const fn from_inputs(
        attention: ScheduledCmd2AttentionInput,
        residual: ScheduledCmd2ResidualInput,
    ) -> Self {
        Self::new(
            attention.source(),
            residual.source(),
            attention.len(),
            residual.len(),
        )
    }
}

impl ScheduledCmd2Inputs for ScheduledCmd2PhaseInputs {
    fn scheduled_cmd2_attention_source(&self) -> ScheduledCmd2AttentionSource {
        self.attention
    }

    fn scheduled_cmd2_residual_source(&self) -> ScheduledCmd2ResidualSource {
        self.residual
    }

    fn scheduled_cmd2_input_state(&self, layer: usize) -> FlashMoeCmd2InputState {
        let attention_placement = match self.attention {
            ScheduledCmd2AttentionSource::CpuAttentionValues => FlashMoeStatePlacement::CpuVisible,
            ScheduledCmd2AttentionSource::MetalAttentionValues => {
                FlashMoeStatePlacement::GpuResident
            }
        };
        let residual_placement = match self.residual {
            ScheduledCmd2ResidualSource::CpuHidden => FlashMoeStatePlacement::CpuVisible,
            ScheduledCmd2ResidualSource::MetalBuffer => FlashMoeStatePlacement::GpuResident,
        };
        FlashMoeCmd2InputState::new(
            layer,
            self.attention_len,
            attention_placement,
            self.residual_len,
            residual_placement,
        )
    }
}

#[derive(Debug)]
pub struct ScheduledCmd2Submission<TInputs> {
    pub cmd2: ScheduledCmd2PostAttention,
    pub inputs: TInputs,
    input_state: FlashMoeCmd2InputState,
}

impl<TInputs> ScheduledCmd2Submission<TInputs>
where
    TInputs: ScheduledCmd2Inputs,
{
    pub fn new(cmd2: ScheduledCmd2PostAttention, inputs: TInputs) -> Result<Self> {
        if cmd2.attention != inputs.scheduled_cmd2_attention_source() {
            bail!(
                "FlashMoe scheduled CMD2 attention source {:?} does not match submitted source {:?}",
                cmd2.attention,
                inputs.scheduled_cmd2_attention_source()
            );
        }
        if cmd2.residual != inputs.scheduled_cmd2_residual_source() {
            bail!(
                "FlashMoe scheduled CMD2 residual source {:?} does not match submitted source {:?}",
                cmd2.residual,
                inputs.scheduled_cmd2_residual_source()
            );
        }
        let input_state = inputs.scheduled_cmd2_input_state(cmd2.layer);
        if input_state.layer() != cmd2.layer {
            bail!(
                "FlashMoe scheduled CMD2 input state layer {} does not match descriptor layer {}",
                input_state.layer(),
                cmd2.layer
            );
        }
        if !input_state.is_declared_graph_state() {
            bail!(
                "FlashMoe scheduled CMD2 input is not declared graph state: attention={:?} residual={:?}",
                input_state.attention(),
                input_state.residual()
            );
        }
        let expected_attention_placement = match cmd2.attention {
            ScheduledCmd2AttentionSource::CpuAttentionValues => FlashMoeStatePlacement::CpuVisible,
            ScheduledCmd2AttentionSource::MetalAttentionValues => {
                FlashMoeStatePlacement::GpuResident
            }
        };
        if input_state.attention().placement() != expected_attention_placement {
            bail!(
                "FlashMoe scheduled CMD2 attention placement {:?} does not match source {:?}",
                input_state.attention().placement(),
                cmd2.attention
            );
        }
        let expected_residual_placement = match cmd2.residual {
            ScheduledCmd2ResidualSource::CpuHidden => FlashMoeStatePlacement::CpuVisible,
            ScheduledCmd2ResidualSource::MetalBuffer => FlashMoeStatePlacement::GpuResident,
        };
        if input_state.residual().placement() != expected_residual_placement {
            bail!(
                "FlashMoe scheduled CMD2 residual placement {:?} does not match source {:?}",
                input_state.residual().placement(),
                cmd2.residual
            );
        }
        Ok(Self {
            cmd2,
            inputs,
            input_state,
        })
    }

    pub(crate) fn into_cmd2_command(self) -> ScheduledCmd2Command<TInputs> {
        ScheduledCmd2Command {
            cmd2: self.cmd2,
            layer: self.cmd2.layer,
            active_experts: self.cmd2.active_experts,
            inputs: self.inputs,
            input_state: self.input_state,
        }
    }
}

#[derive(Debug)]
pub struct ScheduledCmd2Command<TInputs> {
    pub cmd2: ScheduledCmd2PostAttention,
    pub layer: usize,
    pub active_experts: usize,
    pub inputs: TInputs,
    input_state: FlashMoeCmd2InputState,
}

impl<TInputs> ScheduledCmd2Command<TInputs> {
    pub(crate) fn input_state(&self) -> FlashMoeCmd2InputState {
        self.input_state
    }

    pub(crate) fn resolve_post_attention_prep(
        &self,
        state: FlashMoePostAttentionPrepState,
    ) -> Result<ScheduledCmd2PostAttentionPrepOutput> {
        if !state.is_declared_graph_state() {
            bail!("FlashMoe scheduled CMD2 post-attention prep is not declared graph state");
        }
        if state.routing().layer() != self.layer {
            bail!(
                "FlashMoe scheduled CMD2 layer {} does not match post-attention prep layer {}",
                self.layer,
                state.routing().layer()
            );
        }
        if state.active_experts() != self.active_experts {
            bail!(
                "FlashMoe scheduled CMD2 active expert count {} does not match post-attention prep active expert count {}",
                self.active_experts,
                state.active_experts()
            );
        }
        if state.residual().len() != state.normed().len() || state.width() == 0 {
            bail!(
                "FlashMoe scheduled CMD2 post-attention prep has invalid residual/normed width {}",
                state.width()
            );
        }
        if state.width() != self.input_state.residual().len() {
            bail!(
                "FlashMoe scheduled CMD2 post-attention prep width {} does not match residual input width {}",
                state.width(),
                self.input_state.residual().len()
            );
        }
        Ok(ScheduledCmd2PostAttentionPrepOutput {
            cmd2: self.cmd2,
            layer: self.layer,
            active_experts: self.active_experts,
            input_state: self.input_state,
            state,
        })
    }

    pub(crate) fn command_from_post_attention_prep_routes(
        &self,
        graph: &FlashMoeScheduledGraph,
        state: FlashMoePostAttentionPrepState,
        routes: &[(usize, f32)],
    ) -> Result<ScheduledRoutingCommand> {
        self.resolve_post_attention_prep(state)?
            .command_from_preselected_routes(graph, routes)
    }

    #[cfg(any(test, not(all(target_os = "macos", target_arch = "aarch64"))))]
    pub(crate) fn reject_missing_post_attention_prep(&self, reason: &str) -> Result<()> {
        bail!(
            "FlashMoe unsupported scheduled CMD2 path: layer {} declares {} implementation '{}' but no Metal post-attention prep was submitted: {}",
            self.layer,
            self.cmd2.stage.stage,
            self.cmd2.stage.implementation,
            reason
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledCmd2PostAttentionPrepOutput {
    pub cmd2: ScheduledCmd2PostAttention,
    pub layer: usize,
    pub active_experts: usize,
    pub input_state: FlashMoeCmd2InputState,
    state: FlashMoePostAttentionPrepState,
}

impl ScheduledCmd2PostAttentionPrepOutput {
    pub(crate) fn routing(self) -> FlashMoeRoutingOutputState {
        self.state.routing()
    }

    #[cfg(test)]
    pub(crate) fn width(self) -> usize {
        self.state.width()
    }

    pub(crate) fn command_from_preselected_routes(
        self,
        graph: &FlashMoeScheduledGraph,
        routes: &[(usize, f32)],
    ) -> Result<ScheduledRoutingCommand> {
        let routing_state = self.routing();
        let routing = graph.build_routing_topk(
            self.layer,
            routing_state.experts(),
            self.active_experts,
            ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
        )?;
        let routing_output = routing.validate_output_state(routing_state)?;
        routing.command_from_preselected_output(routing_output, routes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledRoutingCandidateSource {
    CpuRouterScores,
    MetalRouterScoresReadback,
    FusedMetalPostAttentionPrepCpuTopK,
}

impl From<FlashMoeRoutingOutputSource> for ScheduledRoutingCandidateSource {
    fn from(source: FlashMoeRoutingOutputSource) -> Self {
        match source {
            FlashMoeRoutingOutputSource::CpuRouterScores => Self::CpuRouterScores,
            FlashMoeRoutingOutputSource::FusedMetalPostAttentionPrepCpuTopK => {
                Self::FusedMetalPostAttentionPrepCpuTopK
            }
        }
    }
}

pub(crate) trait ScheduledRoutingScores {
    fn scheduled_routing_score_layer(&self) -> usize;
    fn scheduled_routing_score_source(&self) -> ScheduledRoutingCandidateSource;
    fn scheduled_routing_scores(&self) -> &[f32];
    fn scheduled_routing_projection(&self) -> Option<&RouterScoreProjectionDescriptor> {
        None
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg(test)]
pub(crate) struct ScheduledRoutingScoreView<'a> {
    layer: usize,
    source: ScheduledRoutingCandidateSource,
    scores: &'a [f32],
    projection: Option<&'a RouterScoreProjectionDescriptor>,
}

#[cfg(test)]
impl<'a> ScheduledRoutingScoreView<'a> {
    pub(crate) const fn new(
        layer: usize,
        source: ScheduledRoutingCandidateSource,
        scores: &'a [f32],
    ) -> Self {
        Self {
            layer,
            source,
            scores,
            projection: None,
        }
    }

    pub(crate) const fn from_router_projection(
        source: ScheduledRoutingCandidateSource,
        projection: &'a RouterScoreProjectionDescriptor,
        scores: &'a [f32],
    ) -> Self {
        Self {
            layer: projection.layer,
            source,
            scores,
            projection: Some(projection),
        }
    }
}

#[cfg(test)]
impl ScheduledRoutingScores for ScheduledRoutingScoreView<'_> {
    fn scheduled_routing_score_layer(&self) -> usize {
        self.layer
    }

    fn scheduled_routing_score_source(&self) -> ScheduledRoutingCandidateSource {
        self.source
    }

    fn scheduled_routing_scores(&self) -> &[f32] {
        self.scores
    }

    fn scheduled_routing_projection(&self) -> Option<&RouterScoreProjectionDescriptor> {
        self.projection
    }
}

impl ScheduledRoutingScores for RouterScoreBatch {
    fn scheduled_routing_score_layer(&self) -> usize {
        self.state().layer()
    }

    fn scheduled_routing_score_source(&self) -> ScheduledRoutingCandidateSource {
        ScheduledRoutingCandidateSource::from(self.state().source())
    }

    fn scheduled_routing_scores(&self) -> &[f32] {
        &self.scores
    }

    fn scheduled_routing_projection(&self) -> Option<&RouterScoreProjectionDescriptor> {
        self.projection.as_ref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledRoutingTopK {
    pub stage: FlashMoeStageCapability,
    pub layer: usize,
    pub experts: usize,
    pub active_experts: usize,
    pub source: ScheduledRoutingCandidateSource,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScheduledRouterScoreProjectionCommand {
    pub(crate) routing: ScheduledRoutingTopK,
    pub(crate) state: FlashMoeRoutingOutputState,
    pub(crate) projection: Option<RouterScoreProjectionDescriptor>,
    pub(crate) hidden_width: usize,
}

impl ScheduledRouterScoreProjectionCommand {
    pub(crate) fn new(
        routing: ScheduledRoutingTopK,
        projection: Option<RouterScoreProjectionDescriptor>,
        hidden_width: usize,
    ) -> Result<Self> {
        if routing.source != ScheduledRoutingCandidateSource::CpuRouterScores {
            bail!(
                "FlashMoe scheduled router score projection requires CPU router-score routing, got {:?}",
                routing.source
            );
        }
        if hidden_width == 0 {
            bail!("FlashMoe scheduled router score projection requires non-zero hidden width");
        }
        routing.validate_bounds()?;
        if let Some(projection) = projection.as_ref() {
            if projection.layer != routing.layer {
                bail!(
                    "FlashMoe scheduled router score projection layer {} does not match routing layer {}",
                    projection.layer,
                    routing.layer
                );
            }
            if projection.experts != routing.experts {
                bail!(
                    "FlashMoe scheduled router score projection experts {} does not match routing experts {}",
                    projection.experts,
                    routing.experts
                );
            }
            if projection.hidden_width != hidden_width {
                bail!(
                    "FlashMoe scheduled router score projection hidden width {} does not match submitted hidden width {}",
                    projection.hidden_width,
                    hidden_width
                );
            }
        }
        let state = FlashMoeRoutingOutputState::cpu_router_scores(
            routing.layer,
            routing.experts,
            routing.active_experts,
        );
        if !state.is_declared_graph_state() {
            bail!("FlashMoe scheduled router score projection did not declare graph state");
        }
        Ok(Self {
            routing,
            state,
            projection,
            hidden_width,
        })
    }

    pub(crate) fn into_score_batch(self, scores: Vec<f32>) -> Result<RouterScoreBatch> {
        RouterScoreBatch::new(self.state, self.projection, scores)
    }

    pub(crate) fn projection_execution(&self) -> Result<RouterScoreProjectionExecution<'_>> {
        let Some(projection) = self.projection.as_ref() else {
            bail!(
                "FlashMoe scheduled router score projection for layer {} has no declared resident projection implementation for stage {:?}",
                self.routing.layer,
                FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection
            );
        };
        projection.execution(self.routing.layer, self.routing.experts, self.hidden_width)
    }

    pub(crate) fn into_routing_command(self, scores: Vec<f32>) -> Result<ScheduledRoutingCommand> {
        let routing = self.routing;
        let batch = self.into_score_batch(scores)?;
        let output = routing.validate_output_state(batch.state())?;
        routing.select_command_from_output_scores(output, &batch)
    }
}

impl ScheduledRoutingTopK {
    pub(crate) fn build_score_projection_command(
        self,
        projection: Option<RouterScoreProjectionDescriptor>,
        hidden_width: usize,
    ) -> Result<ScheduledRouterScoreProjectionCommand> {
        ScheduledRouterScoreProjectionCommand::new(self, projection, hidden_width)
    }

    fn command_from_routes(&self, routes: Vec<(usize, f32)>) -> ScheduledRoutingCommand {
        ScheduledRoutingCommand {
            routing: *self,
            layer: self.layer,
            active_experts: self.active_experts,
            source: self.source,
            routes,
        }
    }

    fn validate_bounds(&self) -> Result<usize> {
        if self.experts == 0 {
            bail!("FlashMoe scheduled routing requires at least one expert");
        }
        if self.active_experts == 0 {
            bail!("FlashMoe scheduled routing active expert count must be non-zero");
        }
        if self.active_experts > self.experts {
            bail!(
                "FlashMoe scheduled routing active expert count {} exceeds expert count {}",
                self.active_experts,
                self.experts
            );
        }
        Ok(self.active_experts)
    }

    pub(crate) fn validate_output_state(
        &self,
        state: FlashMoeRoutingOutputState,
    ) -> Result<ScheduledRoutingOutputState> {
        if !state.is_declared_graph_state() {
            bail!("FlashMoe scheduled routing output is not declared graph state");
        }
        if self.layer != state.layer() {
            bail!(
                "FlashMoe scheduled routing layer {} does not match submitted routing output layer {}",
                self.layer,
                state.layer()
            );
        }
        if self.experts != state.experts() {
            bail!(
                "FlashMoe scheduled routing expert count {} does not match submitted routing output experts {}",
                self.experts,
                state.experts()
            );
        }
        if self.active_experts != state.active_experts() {
            bail!(
                "FlashMoe scheduled routing active expert count {} does not match submitted routing output active expert count {}",
                self.active_experts,
                state.active_experts()
            );
        }
        let source = ScheduledRoutingCandidateSource::from(state.source());
        if self.source != source {
            bail!(
                "FlashMoe scheduled routing source {:?} does not match submitted routing output source {:?}",
                self.source,
                source
            );
        }
        Ok(ScheduledRoutingOutputState {
            routing: *self,
            state,
        })
    }

    pub(crate) fn select_from_scores<TScores>(&self, scores: &TScores) -> Result<Vec<(usize, f32)>>
    where
        TScores: ScheduledRoutingScores,
    {
        if self.source == ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK {
            bail!(
                "FlashMoe scheduled routing source {:?} must submit preselected CPU topK candidates",
                self.source
            );
        }
        if self.layer != scores.scheduled_routing_score_layer() {
            bail!(
                "FlashMoe scheduled routing layer {} does not match submitted score layer {}",
                self.layer,
                scores.scheduled_routing_score_layer()
            );
        }
        if self.source != scores.scheduled_routing_score_source() {
            bail!(
                "FlashMoe scheduled routing source {:?} does not match submitted score source {:?}",
                self.source,
                scores.scheduled_routing_score_source()
            );
        }
        if let Some(projection) = scores.scheduled_routing_projection() {
            if projection.layer != self.layer {
                bail!(
                    "FlashMoe scheduled routing layer {} does not match submitted router projection layer {}",
                    self.layer,
                    projection.layer
                );
            }
            if projection.experts != self.experts {
                bail!(
                    "FlashMoe scheduled routing expert count {} does not match submitted router projection experts {}",
                    self.experts,
                    projection.experts
                );
            }
        }
        let scores = scores.scheduled_routing_scores();
        let active_experts = self.validate_bounds()?;
        if scores.len() != self.experts {
            bail!(
                "FlashMoe scheduled routing received {} router scores for {} experts",
                scores.len(),
                self.experts
            );
        }
        for (expert, score) in scores.iter().copied().enumerate() {
            if !score.is_finite() {
                bail!(
                    "FlashMoe scheduled routing score for expert {} is not finite: {}",
                    expert,
                    score
                );
            }
        }
        Ok(routing_softmax_top_k(scores, active_experts))
    }

    pub(crate) fn select_command_from_scores<TScores>(
        &self,
        scores: &TScores,
    ) -> Result<ScheduledRoutingCommand>
    where
        TScores: ScheduledRoutingScores,
    {
        let routes = self.select_from_scores(scores)?;
        Ok(self.command_from_routes(routes))
    }

    pub(crate) fn select_command_from_output_scores<TScores>(
        &self,
        output: ScheduledRoutingOutputState,
        scores: &TScores,
    ) -> Result<ScheduledRoutingCommand>
    where
        TScores: ScheduledRoutingScores,
    {
        if output.routing != *self {
            bail!(
                "FlashMoe scheduled routing output for layer {} does not match routing command layer {}",
                output.routing.layer,
                self.layer
            );
        }
        self.select_command_from_scores(scores)
    }

    pub fn validate_preselected(&self, routes: &[(usize, f32)]) -> Result<Vec<(usize, f32)>> {
        if self.source != ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK {
            bail!(
                "FlashMoe scheduled routing source {:?} must submit router scores, not preselected candidates",
                self.source
            );
        }
        let active_experts = self.validate_bounds()?;
        if routes.len() != active_experts {
            bail!(
                "FlashMoe scheduled routing received {} preselected experts; expected {}",
                routes.len(),
                active_experts
            );
        }
        let mut seen = BTreeSet::new();
        for (expert, score) in routes.iter().copied() {
            if expert >= self.experts {
                bail!(
                    "FlashMoe scheduled routing selected expert {} outside expert count {}",
                    expert,
                    self.experts
                );
            }
            if !score.is_finite() {
                bail!(
                    "FlashMoe scheduled routing selected expert {} has non-finite score {}",
                    expert,
                    score
                );
            }
            if !seen.insert(expert) {
                bail!("FlashMoe scheduled routing selected expert {expert} more than once");
            }
        }
        Ok(routes.to_vec())
    }

    pub(crate) fn command_from_preselected(
        &self,
        routes: &[(usize, f32)],
    ) -> Result<ScheduledRoutingCommand> {
        let routes = self.validate_preselected(routes)?;
        Ok(self.command_from_routes(routes))
    }

    pub(crate) fn command_from_preselected_output(
        &self,
        output: ScheduledRoutingOutputState,
        routes: &[(usize, f32)],
    ) -> Result<ScheduledRoutingCommand> {
        if output.routing != *self {
            bail!(
                "FlashMoe scheduled routing output for layer {} does not match routing command layer {}",
                output.routing.layer,
                self.layer
            );
        }
        self.command_from_preselected(routes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledRoutingOutputState {
    pub(crate) routing: ScheduledRoutingTopK,
    state: FlashMoeRoutingOutputState,
}

impl ScheduledRoutingOutputState {
    #[cfg(test)]
    pub(crate) fn state(self) -> FlashMoeRoutingOutputState {
        self.state
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledRoutingCommand {
    pub routing: ScheduledRoutingTopK,
    pub layer: usize,
    pub active_experts: usize,
    pub source: ScheduledRoutingCandidateSource,
    pub routes: Vec<(usize, f32)>,
}

impl ScheduledRoutingCommand {
    pub(crate) fn validate_for_active_expert_issue(&self) -> Result<()> {
        if self.layer != self.routing.layer {
            bail!(
                "FlashMoe scheduled routing command layer {} does not match routing descriptor layer {}",
                self.layer,
                self.routing.layer
            );
        }
        if self.active_experts != self.routing.active_experts {
            bail!(
                "FlashMoe scheduled routing command active expert count {} does not match routing descriptor active expert count {}",
                self.active_experts,
                self.routing.active_experts
            );
        }
        if self.source != self.routing.source {
            bail!(
                "FlashMoe scheduled routing command source {:?} does not match routing descriptor source {:?}",
                self.source,
                self.routing.source
            );
        }
        self.routing.validate_bounds()?;
        if self.routes.len() != self.active_experts {
            bail!(
                "FlashMoe scheduled routing command carries {} routes for active expert count {}",
                self.routes.len(),
                self.active_experts
            );
        }
        let mut seen = BTreeSet::new();
        for (expert, score) in self.routes.iter().copied() {
            if expert >= self.routing.experts {
                bail!(
                    "FlashMoe scheduled routing command selected expert {} outside expert count {}",
                    expert,
                    self.routing.experts
                );
            }
            if !seen.insert(expert) {
                bail!("FlashMoe scheduled routing command selected expert {expert} more than once");
            }
            if !score.is_finite() {
                bail!(
                    "FlashMoe scheduled routing command score for expert {} is not finite: {}",
                    expert,
                    score
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledCmd3InputSource {
    MetalPostAttentionPrep,
    CpuNormedResidualUpload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledSharedExpertSource {
    None,
    DenseCpuWeights,
    ResidentProjections,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledNextNormSource {
    None,
    CpuVisibleWeights,
}

impl ScheduledNextNormSource {
    pub(crate) fn from_weights(weights: ScheduledNextNormWeights<'_>) -> Self {
        if weights.is_cpu_visible() {
            Self::CpuVisibleWeights
        } else {
            Self::None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledSharedExpertDescriptor {
    pub source: ScheduledSharedExpertSource,
    pub shape: Option<ScheduledSharedExpertShape>,
}

impl ScheduledSharedExpertDescriptor {
    pub fn new(
        source: ScheduledSharedExpertSource,
        shape: Option<ScheduledSharedExpertShape>,
    ) -> Result<Self> {
        match (source, shape) {
            (ScheduledSharedExpertSource::None, Some(_)) => bail!(
                "FlashMoe scheduled shared expert descriptor cannot declare a shape for source None"
            ),
            (ScheduledSharedExpertSource::DenseCpuWeights, None)
            | (ScheduledSharedExpertSource::ResidentProjections, None) => bail!(
                "FlashMoe scheduled shared expert descriptor source {:?} requires a declared shape",
                source
            ),
            _ => Ok(Self { source, shape }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledSharedExpertShape {
    pub width: usize,
    pub shared_experts: usize,
    pub intermediate: usize,
    pub total_intermediate: usize,
}

impl ScheduledSharedExpertShape {
    pub fn new(width: usize, shared_experts: usize, intermediate: usize) -> Result<Self> {
        let total_intermediate = shared_experts
            .checked_mul(intermediate)
            .ok_or_else(|| anyhow::anyhow!("shared expert intermediate width overflow"))?;
        Ok(Self {
            width,
            shared_experts,
            intermediate,
            total_intermediate,
        })
    }

    pub fn is_declared_graph_shape(self) -> bool {
        self.width > 0
            && self.shared_experts > 0
            && self.intermediate > 0
            && self.total_intermediate == self.shared_experts.saturating_mul(self.intermediate)
    }
}

impl From<SharedExpertPhaseShape> for ScheduledSharedExpertShape {
    fn from(shape: SharedExpertPhaseShape) -> Self {
        Self {
            width: shape.width,
            shared_experts: shape.shared_experts,
            intermediate: shape.intermediate,
            total_intermediate: shape.total_intermediate,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledCmd3ExpertPhase {
    pub stage: FlashMoeStageCapability,
    pub layer: usize,
    pub expert_count: usize,
    pub(crate) expert_storage: ExpertStorageLayout,
    pub input: ScheduledCmd3InputSource,
    pub shared: ScheduledSharedExpertSource,
    pub shared_descriptor: Option<ScheduledSharedExpertDescriptor>,
    pub next_norm: ScheduledNextNormSource,
}

pub trait ScheduledCmd3Input {
    fn scheduled_cmd3_input_source(&self) -> ScheduledCmd3InputSource;
    fn scheduled_cmd3_input_state(&self, layer: usize) -> FlashMoeCmd3InputState;
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScheduledCmd3CpuInput<'a> {
    #[allow(dead_code)]
    pub(crate) normed: &'a [f32],
    #[allow(dead_code)]
    pub(crate) residual: &'a [f32],
    state: FlashMoeCmd3InputState,
}

#[cfg(test)]
impl<'a> ScheduledCmd3CpuInput<'a> {
    #[cfg(test)]
    pub(crate) fn new(layer: usize, normed: &'a [f32], residual: &'a [f32]) -> Result<Self> {
        let state =
            FlashMoeCmd3InputState::cpu_normed_residual(layer, normed.len(), residual.len());
        if !state.is_declared_graph_state() {
            bail!(
                "FlashMoe unsupported scheduled CMD3 CPU input for layer {layer}: normed_len={} residual_len={} is not a declared graph state",
                normed.len(),
                residual.len()
            );
        }
        Ok(Self {
            normed,
            residual,
            state,
        })
    }

    pub(crate) fn width(self) -> usize {
        self.state.width()
    }

    #[allow(dead_code)]
    pub(crate) fn state(self) -> FlashMoeCmd3InputState {
        self.state
    }
}

#[cfg(test)]
impl ScheduledCmd3Input for ScheduledCmd3CpuInput<'_> {
    fn scheduled_cmd3_input_source(&self) -> ScheduledCmd3InputSource {
        ScheduledCmd3InputSource::CpuNormedResidualUpload
    }

    fn scheduled_cmd3_input_state(&self, _layer: usize) -> FlashMoeCmd3InputState {
        self.state
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScheduledCmd3MetalPostAttentionInput {
    state: FlashMoePostAttentionPrepState,
    input_state: FlashMoeCmd3InputState,
}

impl ScheduledCmd3MetalPostAttentionInput {
    pub(crate) fn new(state: FlashMoePostAttentionPrepState, active_routes: usize) -> Result<Self> {
        let layer = state.routing().layer();
        if !state.is_declared_graph_state() {
            bail!(
                "FlashMoe unsupported scheduled CMD3 Metal post-attention input for layer {layer}: prep state is not a declared graph state"
            );
        }
        if state.active_experts() != active_routes {
            bail!(
                "FlashMoe unsupported scheduled CMD3 Metal post-attention input for layer {layer}: state declares {} active experts but prep carries {active_routes} routes",
                state.active_experts()
            );
        }
        let input_state = FlashMoeCmd3InputState::metal_post_attention_prep(layer, state);
        if !input_state.is_declared_graph_state() {
            bail!(
                "FlashMoe unsupported scheduled CMD3 Metal post-attention input for layer {layer}: input state is not a declared graph state"
            );
        }
        Ok(Self { state, input_state })
    }

    pub(crate) fn width(self) -> usize {
        self.state.width()
    }

    pub(crate) fn state(self) -> FlashMoePostAttentionPrepState {
        self.state
    }
}

impl ScheduledCmd3Input for ScheduledCmd3MetalPostAttentionInput {
    fn scheduled_cmd3_input_source(&self) -> ScheduledCmd3InputSource {
        ScheduledCmd3InputSource::MetalPostAttentionPrep
    }

    fn scheduled_cmd3_input_state(&self, _layer: usize) -> FlashMoeCmd3InputState {
        self.input_state
    }
}

pub trait ScheduledCmd3Expert {
    fn scheduled_expert_layer(&self) -> usize;
    fn scheduled_expert_id(&self) -> usize;
    fn scheduled_expert_slot_descriptor(&self) -> ExpertSlotDescriptor;
}

impl<T> ScheduledCmd3Expert for Arc<T>
where
    T: ScheduledCmd3Expert,
{
    fn scheduled_expert_layer(&self) -> usize {
        self.as_ref().scheduled_expert_layer()
    }

    fn scheduled_expert_id(&self) -> usize {
        self.as_ref().scheduled_expert_id()
    }

    fn scheduled_expert_slot_descriptor(&self) -> ExpertSlotDescriptor {
        self.as_ref().scheduled_expert_slot_descriptor()
    }
}

#[derive(Debug, Clone)]
pub enum ScheduledExpertPhaseMlpPayload<'a> {
    Q4(ScheduledQ4ExpertPhaseMlpPayload<'a>),
    Dense(ScheduledDenseExpertPhaseMlpPayload<'a>),
    DeepSeekGguf(ScheduledDeepSeekGgufExpertPhaseMlpPayload<'a>),
}

impl<'a> ScheduledExpertPhaseMlpPayload<'a> {
    #[cfg(test)]
    pub(crate) fn q4(&self) -> &ScheduledQ4ExpertPhaseMlpPayload<'a> {
        match self {
            Self::Q4(payload) => payload,
            Self::Dense(_) | Self::DeepSeekGguf(_) => {
                panic!("scheduled expert payload is not Q4")
            }
        }
    }

    pub(crate) fn storage_layout(&self) -> ExpertStorageLayout {
        match self {
            Self::Q4(payload) => {
                if payload
                    .gate
                    .scale_bias_dtype
                    .eq_ignore_ascii_case(super::experts::EXPERT_SCALE_DTYPE_E8M0)
                {
                    ExpertStorageLayout::FixedMxfp4
                } else {
                    ExpertStorageLayout::FixedQ4
                }
            }
            Self::Dense(payload) => match payload.gate.dtype {
                super::experts::DenseExpertDtype::Bf16 => ExpertStorageLayout::FixedBf16,
                super::experts::DenseExpertDtype::F16 => ExpertStorageLayout::FixedF16,
            },
            Self::DeepSeekGguf(_) => ExpertStorageLayout::FixedDeepSeekGguf,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScheduledDeepSeekGgufExpertPhaseMlpPayload<'a> {
    pub(crate) layer: usize,
    pub(crate) expert: usize,
    pub(crate) spec: DeepSeekGgufExpertSlotSpec,
    pub(crate) bytes: &'a ReusableExpertBytes,
}

impl<'a> ScheduledDeepSeekGgufExpertPhaseMlpPayload<'a> {
    fn new(
        layer: usize,
        expert: usize,
        spec: DeepSeekGgufExpertSlotSpec,
        bytes: &'a ReusableExpertBytes,
        width: usize,
    ) -> Result<Self> {
        if width != spec.hidden_size || bytes.len() != spec.expert_bytes {
            bail!(
                "FlashMoe unsupported DeepSeek active expert CMD3 slot layer {layer} expert {expert}: width/bytes {width}/{} do not match resolved {}/{}",
                bytes.len(),
                spec.hidden_size,
                spec.expert_bytes
            );
        }
        Ok(Self {
            layer,
            expert,
            spec,
            bytes,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ScheduledDenseExpertPhaseMlpPayload<'a> {
    pub(crate) gate: DenseMatvecPayload<'a>,
    pub(crate) up: DenseMatvecPayload<'a>,
    pub(crate) down: DenseMatvecPayload<'a>,
}

impl<'a> ScheduledDenseExpertPhaseMlpPayload<'a> {
    pub(crate) fn new(
        layer: usize,
        expert: usize,
        width: usize,
        gate: DenseMatvecPayload<'a>,
        up: DenseMatvecPayload<'a>,
        down: DenseMatvecPayload<'a>,
    ) -> Result<Self> {
        if gate.dtype != up.dtype || gate.dtype != down.dtype {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed dense expert layer {layer} expert {expert} has mismatched projection dtypes"
            );
        }
        if gate.rows == 0
            || gate.rows != up.rows
            || down.cols != gate.rows
            || gate.cols != width
            || up.cols != width
            || down.rows != width
        {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed {} expert layer {} expert {} has payload shape gate={}x{}, up={}x{}, down={}x{} for width {width}",
                gate.dtype.as_str(),
                layer,
                expert,
                gate.rows,
                gate.cols,
                up.rows,
                up.cols,
                down.rows,
                down.cols
            );
        }
        let source = gate.source;
        if !source.same_buffer(up.source) || !source.same_buffer(down.source) {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed {} expert layer {layer} expert {expert} projections do not share one scheduler-owned whole-expert slot",
                gate.dtype.as_str()
            );
        }
        Ok(Self { gate, up, down })
    }
}

#[derive(Debug, Clone)]
pub struct ScheduledQ4ExpertPhaseMlpPayload<'a> {
    pub(crate) gate: Q4MatvecPayload<'a>,
    pub(crate) up: Q4MatvecPayload<'a>,
    pub(crate) down: Q4MatvecPayload<'a>,
    gate_source: Q4MatvecSource<'a>,
    up_source: Q4MatvecSource<'a>,
    down_source: Q4MatvecSource<'a>,
}

impl<'a> ScheduledQ4ExpertPhaseMlpPayload<'a> {
    pub(crate) fn new(
        layer: usize,
        expert: usize,
        width: usize,
        gate: Q4MatvecPayload<'a>,
        up: Q4MatvecPayload<'a>,
        down: Q4MatvecPayload<'a>,
    ) -> Result<Self> {
        if gate.rows != up.rows {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed Q4 expert layer {} expert {} has mismatched gate/up rows {} vs {}",
                layer,
                expert,
                gate.rows,
                up.rows
            );
        }
        if down.cols != gate.rows {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed Q4 expert layer {} expert {} has down cols {} for gate rows {}",
                layer,
                expert,
                down.cols,
                gate.rows
            );
        }
        if gate.cols != width || up.cols != width || down.rows != width {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed Q4 expert layer {} expert {} has payload shape gate={}x{}, up={}x{}, down={}x{} for width {width}",
                layer,
                expert,
                gate.rows,
                gate.cols,
                up.rows,
                up.cols,
                down.rows,
                down.cols
            );
        }
        if gate.group_size != up.group_size || gate.group_size != down.group_size {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed Q4 expert layer {layer} expert {expert} has mismatched group sizes gate={} up={} down={}",
                gate.group_size,
                up.group_size,
                down.group_size
            );
        }
        let gate_source = Self::required_fixed_source(layer, expert, "gate", &gate)?;
        let up_source = Self::required_fixed_source(layer, expert, "up", &up)?;
        let down_source = Self::required_fixed_source(layer, expert, "down", &down)?;
        if !gate_source.same_buffer(up_source) || !gate_source.same_buffer(down_source) {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed Q4 expert layer {layer} expert {expert} projections do not share one scheduler-owned whole-expert slot"
            );
        }
        Ok(Self {
            gate,
            up,
            down,
            gate_source,
            up_source,
            down_source,
        })
    }

    fn required_fixed_source(
        layer: usize,
        expert: usize,
        projection: &str,
        payload: &Q4MatvecPayload<'a>,
    ) -> Result<Q4MatvecSource<'a>> {
        let affine_bf16 = payload
            .scale_bias_dtype
            .eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16);
        let mxfp4 = payload
            .scale_bias_dtype
            .eq_ignore_ascii_case(super::experts::EXPERT_SCALE_DTYPE_E8M0);
        if !affine_bf16 && !mxfp4 {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed Q4 expert layer {layer} expert {expert} {projection} projection uses {} scale/bias values; the resolved implementation requires BF16 affine or E8M0 MXFP4",
                payload.scale_bias_dtype
            );
        }
        let expected_scale_bytes = payload
            .scale_bias_groups
            .checked_mul(if affine_bf16 {
                std::mem::size_of::<u16>()
            } else {
                1
            })
            .context("fixed Q4 expert scale byte size overflow")?;
        let expected_bias_bytes = if affine_bf16 { expected_scale_bytes } else { 0 };
        if payload.scale_bytes.len() != expected_scale_bytes
            || payload.bias_bytes.len() != expected_bias_bytes
        {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed Q4 expert layer {layer} expert {expert} {projection} projection has scale/bias byte lengths {}/{}; expected {expected_scale_bytes}/{expected_bias_bytes}",
                payload.scale_bytes.len(),
                payload.bias_bytes.len()
            );
        }
        let source = payload.source.with_context(|| {
            format!(
                "FlashMoe unsupported active expert CMD3 path: fixed Q4 expert layer {layer} expert {expert} {projection} projection is not backed by a scheduler-owned whole-expert slot"
            )
        })?;
        if !source.covers(payload) || !source.offsets_are_metal_aligned() {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed Q4 expert layer {layer} expert {expert} {projection} projection offsets are outside or misaligned for its scheduler-owned whole-expert slot"
            );
        }
        Ok(source)
    }

    pub(crate) fn gate_source(&self) -> Q4MatvecSource<'a> {
        self.gate_source
    }

    pub(crate) fn up_source(&self) -> Q4MatvecSource<'a> {
        self.up_source
    }

    pub(crate) fn down_source(&self) -> Q4MatvecSource<'a> {
        self.down_source
    }
}

pub trait ScheduledCmd3ExpertPayload {
    fn scheduled_cmd3_expert_phase_payload(
        &self,
        width: usize,
    ) -> Result<ScheduledExpertPhaseMlpPayload<'_>>;
}

impl<T> ScheduledCmd3ExpertPayload for Arc<T>
where
    T: ScheduledCmd3ExpertPayload,
{
    fn scheduled_cmd3_expert_phase_payload(
        &self,
        width: usize,
    ) -> Result<ScheduledExpertPhaseMlpPayload<'_>> {
        self.as_ref().scheduled_cmd3_expert_phase_payload(width)
    }
}

pub trait ScheduledSharedExpert {
    fn scheduled_shared_expert_descriptor(&self) -> Result<ScheduledSharedExpertDescriptor>;
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) enum ScheduledSharedExpertPhaseRef<'a> {
    None,
    Dense(&'a SharedExpertPhaseWeights),
    Resident(&'a SharedExpertPhaseResidentProjections),
}

impl<'a> ScheduledSharedExpertPhaseRef<'a> {
    #[allow(dead_code)]
    pub(crate) fn from_options(
        dense: Option<&'a SharedExpertPhaseWeights>,
        resident: Option<&'a SharedExpertPhaseResidentProjections>,
    ) -> Self {
        if let Some(resident) = resident {
            Self::Resident(resident)
        } else if let Some(dense) = dense {
            Self::Dense(dense)
        } else {
            Self::None
        }
    }

    #[cfg(test)]
    pub(crate) fn dense(self) -> Option<&'a SharedExpertPhaseWeights> {
        match self {
            Self::Dense(shared) => Some(shared),
            Self::None | Self::Resident(_) => None,
        }
    }

    pub(crate) fn resident(self) -> Option<&'a SharedExpertPhaseResidentProjections> {
        match self {
            Self::Resident(shared) => Some(shared),
            Self::None | Self::Dense(_) => None,
        }
    }
}

impl ScheduledSharedExpert for ScheduledSharedExpertPhaseRef<'_> {
    fn scheduled_shared_expert_descriptor(&self) -> Result<ScheduledSharedExpertDescriptor> {
        match self {
            Self::None => {
                ScheduledSharedExpertDescriptor::new(ScheduledSharedExpertSource::None, None)
            }
            Self::Dense(shared) => ScheduledSharedExpertDescriptor::new(
                ScheduledSharedExpertSource::DenseCpuWeights,
                Some(ScheduledSharedExpertShape::from(shared.validated_shape()?)),
            ),
            Self::Resident(shared) => ScheduledSharedExpertDescriptor::new(
                ScheduledSharedExpertSource::ResidentProjections,
                Some(ScheduledSharedExpertShape::from(shared.validated_shape()?)),
            ),
        }
    }
}

#[derive(Debug)]
pub struct ScheduledCmd3Submission<'a, TExpert, TInput, TShared> {
    pub cmd3: ScheduledCmd3ExpertPhase,
    pub position: usize,
    pub scheduled: &'a ScheduledExpertSet<TExpert>,
    pub input: TInput,
    input_state: FlashMoeCmd3InputState,
    pub shared: TShared,
    pub next_norm_weights: ScheduledNextNormWeights<'a>,
}

impl<'a, TExpert, TInput, TShared> ScheduledCmd3Submission<'a, TExpert, TInput, TShared>
where
    TExpert: ScheduledCmd3Expert,
    TInput: ScheduledCmd3Input,
    TShared: ScheduledSharedExpert,
{
    pub fn new(
        position: usize,
        cmd3: ScheduledCmd3ExpertPhase,
        scheduled: &'a ScheduledExpertSet<TExpert>,
        input: TInput,
        shared: TShared,
        next_norm_weights: ScheduledNextNormWeights<'a>,
    ) -> Result<Self> {
        if cmd3.layer != scheduled.layer || cmd3.expert_count != scheduled.len() {
            bail!(
                "FlashMoe scheduled CMD3 descriptor layer {} experts {} does not match scheduled expert set layer {} experts {}",
                cmd3.layer,
                cmd3.expert_count,
                scheduled.layer,
                scheduled.len()
            );
        }
        for (route, expert) in scheduled.routes.iter().zip(scheduled.experts.iter()) {
            let expert_layer = expert.scheduled_expert_layer();
            let expert_id = expert.scheduled_expert_id();
            if expert_layer != scheduled.layer || expert_id != route.expert {
                bail!(
                    "FlashMoe scheduled CMD3 expert layer {} expert {} does not match routed layer {} expert {}",
                    expert_layer,
                    expert_id,
                    scheduled.layer,
                    route.expert
                );
            }
            let descriptor = expert.scheduled_expert_slot_descriptor();
            if descriptor.layer != scheduled.layer || descriptor.expert != route.expert {
                bail!(
                    "FlashMoe scheduled CMD3 expert slot descriptor layer {} expert {} does not match routed layer {} expert {}",
                    descriptor.layer,
                    descriptor.expert,
                    scheduled.layer,
                    route.expert
                );
            }
            if descriptor.slot_capacity == 0 || descriptor.payload_len != descriptor.slot_capacity {
                bail!(
                    "FlashMoe scheduled CMD3 expert slot layer {} expert {} must be a whole-expert slot, payload_len={} slot_capacity={}",
                    descriptor.layer,
                    descriptor.expert,
                    descriptor.payload_len,
                    descriptor.slot_capacity
                );
            }
        }
        if cmd3.input != input.scheduled_cmd3_input_source() {
            bail!(
                "FlashMoe scheduled CMD3 input {:?} does not match phase input {:?}",
                cmd3.input,
                input.scheduled_cmd3_input_source()
            );
        }
        let shared_descriptor = shared.scheduled_shared_expert_descriptor()?;
        if cmd3.shared != shared_descriptor.source {
            bail!(
                "FlashMoe scheduled CMD3 shared source {:?} does not match phase shared source {:?}",
                cmd3.shared,
                shared_descriptor.source
            );
        }
        if let Some(expected_shared) = cmd3.shared_descriptor
            && expected_shared != shared_descriptor
        {
            bail!(
                "FlashMoe scheduled CMD3 shared descriptor {:?} does not match phase shared descriptor {:?}",
                expected_shared,
                shared_descriptor
            );
        }
        let input_state = input.scheduled_cmd3_input_state(cmd3.layer);
        if input_state.layer() != cmd3.layer {
            bail!(
                "FlashMoe scheduled CMD3 input state layer {} does not match descriptor layer {}",
                input_state.layer(),
                cmd3.layer
            );
        }
        if !input_state.is_declared_graph_state() {
            bail!(
                "FlashMoe scheduled CMD3 input is not declared graph state: placement={:?} residual={:?} normed={:?}",
                input_state.placement(),
                input_state.residual(),
                input_state.normed()
            );
        }
        let expected_placement = match cmd3.input {
            ScheduledCmd3InputSource::CpuNormedResidualUpload => FlashMoeStatePlacement::CpuVisible,
            ScheduledCmd3InputSource::MetalPostAttentionPrep => FlashMoeStatePlacement::GpuResident,
        };
        if input_state.placement() != expected_placement {
            bail!(
                "FlashMoe scheduled CMD3 input placement {:?} does not match source {:?}",
                input_state.placement(),
                cmd3.input
            );
        }
        let input_width = input_state.width();
        if input_width == 0 {
            bail!("FlashMoe scheduled CMD3 input width must be non-zero");
        }
        if let Some(shape) = shared_descriptor.shape {
            if !shape.is_declared_graph_shape() {
                bail!(
                    "FlashMoe scheduled CMD3 shared expert shape is not declared graph shape: width={} shared_experts={} intermediate={} total_intermediate={}",
                    shape.width,
                    shape.shared_experts,
                    shape.intermediate,
                    shape.total_intermediate
                );
            }
            if shape.width != input_width {
                bail!(
                    "FlashMoe scheduled CMD3 shared expert width {} does not match input width {}",
                    shape.width,
                    input_width
                );
            }
        }
        if cmd3.next_norm == ScheduledNextNormSource::CpuVisibleWeights
            && !next_norm_weights.is_cpu_visible()
        {
            bail!("FlashMoe scheduled CMD3 requires next-norm weights but none were provided");
        }
        if let Some(width) = next_norm_weights.width()
            && width != input_width
        {
            bail!(
                "FlashMoe scheduled CMD3 next-norm weight width {} does not match input width {}",
                width,
                input_width
            );
        }
        if cmd3.next_norm == ScheduledNextNormSource::None && !next_norm_weights.is_none() {
            bail!("FlashMoe scheduled CMD3 received next-norm weights for a no-next-norm stage");
        }
        Ok(Self {
            cmd3,
            position,
            scheduled,
            input,
            input_state,
            shared,
            next_norm_weights,
        })
    }
}

#[derive(Debug)]
pub struct ScheduledCmd3Command<'a, TExpert, TInput, TShared> {
    pub cmd3: ScheduledCmd3ExpertPhase,
    pub position: usize,
    pub layer: usize,
    pub experts: Arc<[TExpert]>,
    pub weights: &'a [f32],
    pub input: TInput,
    pub input_state: FlashMoeCmd3InputState,
    pub shared: TShared,
    pub next_norm_weights: ScheduledNextNormWeights<'a>,
    pub payloads: Vec<ScheduledExpertPhaseMlpPayload<'a>>,
}

impl<TExpert, TInput, TShared> ScheduledCmd3Command<'_, TExpert, TInput, TShared>
where
    TInput: ScheduledCmd3Input,
{
    pub(crate) fn resolve_output_state(&self) -> Result<ScheduledCmd3OutputState> {
        let input_state = self.input_state;
        if !input_state.is_declared_graph_state() {
            bail!("FlashMoe scheduled CMD3 output cannot resolve from undeclared input state");
        }
        let width = input_state.width();
        let state = FlashMoeCmd3OutputState::gpu_resident(
            width,
            self.cmd3.next_norm == ScheduledNextNormSource::CpuVisibleWeights,
        );
        if !state.is_declared_graph_state() {
            bail!("FlashMoe scheduled CMD3 output is not declared graph state");
        }
        if state.width() != width {
            bail!(
                "FlashMoe scheduled CMD3 output width {} does not match input width {}",
                state.width(),
                width
            );
        }
        if state.has_next_normed()
            != (self.cmd3.next_norm == ScheduledNextNormSource::CpuVisibleWeights)
        {
            bail!(
                "FlashMoe scheduled CMD3 output next-norm state does not match descriptor {:?}",
                self.cmd3.next_norm
            );
        }
        Ok(ScheduledCmd3OutputState {
            cmd3: self.cmd3,
            layer: self.layer,
            input_state,
            state,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledCmd3OutputState {
    pub cmd3: ScheduledCmd3ExpertPhase,
    pub layer: usize,
    pub input_state: FlashMoeCmd3InputState,
    state: FlashMoeCmd3OutputState,
}

impl ScheduledCmd3OutputState {
    pub(crate) fn state(self) -> FlashMoeCmd3OutputState {
        self.state
    }

    pub(crate) fn validate_expert_phase_output(
        self,
        output: FlashMoeExpertPhaseOutput,
    ) -> Result<FlashMoeExpertPhaseOutput> {
        let state = self.state();
        if output.hidden_len() != state.hidden().len() {
            bail!(
                "FlashMoe scheduled CMD3 output hidden length {} does not match declared hidden length {} for layer {}",
                output.hidden_len(),
                state.hidden().len(),
                self.layer
            );
        }
        match (output.next_normed_len(), state.next_normed()) {
            (Some(actual), Some(expected)) if actual != expected.len() => {
                bail!(
                    "FlashMoe scheduled CMD3 output next-normed length {} does not match declared next-normed length {} for layer {}",
                    actual,
                    expected.len(),
                    self.layer
                );
            }
            (Some(_), None) => {
                bail!(
                    "FlashMoe scheduled CMD3 output produced next-normed state for layer {}, but the graph did not declare one",
                    self.layer
                );
            }
            (None, Some(_)) => {
                bail!(
                    "FlashMoe scheduled CMD3 output did not produce declared next-normed state for layer {}",
                    self.layer
                );
            }
            _ => {}
        }
        Ok(output.with_declared_cmd3_output(state))
    }
}

impl<'a, TExpert, TInput, TShared> ScheduledCmd3Submission<'a, TExpert, TInput, TShared>
where
    TExpert: ScheduledCmd3Expert + ScheduledCmd3ExpertPayload,
    TInput: ScheduledCmd3Input,
    TShared: ScheduledSharedExpert,
{
    pub(crate) fn into_cmd3_command(
        self,
    ) -> Result<ScheduledCmd3Command<'a, TExpert, TInput, TShared>> {
        let input_width = self.input_state.width();
        let payloads = self.scheduled.cmd3_expert_phase_payloads(input_width)?;
        if let Some((index, payload)) = payloads
            .iter()
            .enumerate()
            .find(|(_, payload)| payload.storage_layout() != self.cmd3.expert_storage)
        {
            bail!(
                "FlashMoe scheduled CMD3 expert payload {index} resolves {:?} storage, but the graph requires {:?}",
                payload.storage_layout(),
                self.cmd3.expert_storage
            );
        }
        Ok(ScheduledCmd3Command {
            cmd3: self.cmd3,
            position: self.position,
            layer: self.scheduled.layer,
            experts: self.scheduled.experts.clone(),
            weights: &self.scheduled.weights,
            input: self.input,
            input_state: self.input_state,
            shared: self.shared,
            next_norm_weights: self.next_norm_weights,
            payloads,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertRoute {
    pub expert: usize,
    pub score: f32,
}

impl ExpertRoute {
    pub fn from_pair((expert, score): (usize, f32)) -> Self {
        Self { expert, score }
    }

    pub fn from_scores(routes: &[(usize, f32)]) -> Result<Vec<Self>> {
        routes
            .iter()
            .copied()
            .map(|route| {
                let route = Self::from_pair(route);
                route.validate()?;
                Ok(route)
            })
            .collect()
    }

    fn validate(&self) -> Result<()> {
        if !self.score.is_finite() {
            bail!(
                "expert route score for expert {} is not finite: {}",
                self.expert,
                self.score
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledExpertRoutes {
    pub layer: usize,
    pub routes: Vec<ExpertRoute>,
    pub weights: Vec<f32>,
}

impl ScheduledExpertRoutes {
    pub fn from_routes(
        layer: usize,
        routes: Vec<ExpertRoute>,
        routed_expert_scale: f32,
    ) -> Result<Self> {
        Self::from_routes_with_policy(
            layer,
            routes,
            QwenMoeRoutingWeightNormalization::RenormalizeSelected,
            routed_expert_scale,
        )
    }

    pub fn from_routes_with_policy(
        layer: usize,
        routes: Vec<ExpertRoute>,
        normalization: QwenMoeRoutingWeightNormalization,
        routed_expert_scale: f32,
    ) -> Result<Self> {
        if !(routed_expert_scale.is_finite() && routed_expert_scale > 0.0) {
            bail!("routed expert scale must be positive and finite");
        }
        for route in &routes {
            route.validate()?;
        }
        let mut weights: Vec<f32> = routes.iter().map(|route| route.score).collect();
        match normalization {
            QwenMoeRoutingWeightNormalization::RenormalizeSelected => {
                let sum = weights.iter().sum::<f32>();
                if !(sum.is_finite() && sum > 0.0) {
                    bail!("selected expert probabilities must have a positive finite sum");
                }
                let inverse_sum = sum.recip();
                for weight in &mut weights {
                    if *weight < 0.0 {
                        bail!("selected expert probabilities must be non-negative");
                    }
                    *weight *= inverse_sum;
                }
            }
            QwenMoeRoutingWeightNormalization::DeepSeekRenormalizeSelectedWithFloor => {
                const DEEPSEEK_SELECTED_SUM_FLOOR: f32 = 6.103515625e-5;
                let sum = weights.iter().sum::<f32>();
                if !(sum.is_finite() && sum >= 0.0) {
                    bail!(
                        "DeepSeek selected expert probabilities must have a finite non-negative sum"
                    );
                }
                let inverse_sum = sum.max(DEEPSEEK_SELECTED_SUM_FLOOR).recip();
                for weight in &mut weights {
                    if *weight < 0.0 {
                        bail!("DeepSeek selected expert probabilities must be non-negative");
                    }
                    *weight *= inverse_sum;
                }
            }
            QwenMoeRoutingWeightNormalization::PreserveFullSoftmax => bail!(
                "FlashMoe unsupported routing weights: preserving probabilities from the full expert softmax requires a declared scheduler implementation"
            ),
        }
        for weight in &mut weights {
            *weight *= routed_expert_scale;
        }
        Ok(Self {
            layer,
            routes,
            weights,
        })
    }

    pub fn from_scores(
        layer: usize,
        routes: &[(usize, f32)],
        routed_expert_scale: f32,
    ) -> Result<Self> {
        Self::from_routes(
            layer,
            ExpertRoute::from_scores(routes)?,
            routed_expert_scale,
        )
    }

    #[cfg(test)]
    pub(crate) fn from_routing_command(
        command: &ScheduledRoutingCommand,
        routed_expert_scale: f32,
    ) -> Result<Self> {
        command.validate_for_active_expert_issue()?;
        Self::from_scores(command.layer, &command.routes, routed_expert_scale)
    }

    pub(crate) fn from_routing_command_with_policy(
        command: &ScheduledRoutingCommand,
        normalization: QwenMoeRoutingWeightNormalization,
        routed_expert_scale: f32,
    ) -> Result<Self> {
        command.validate_for_active_expert_issue()?;
        Self::from_routes_with_policy(
            command.layer,
            ExpertRoute::from_scores(&command.routes)?,
            normalization,
            routed_expert_scale,
        )
    }

    #[cfg(test)]
    pub(crate) fn expert_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.routes.iter().map(|route| route.expert)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledExpertBatch<T> {
    pub layer: usize,
    pub routes: Vec<ExpertRoute>,
    pub weights: Vec<f32>,
    pub experts: Arc<[T]>,
}

impl<T> ScheduledExpertBatch<T> {
    pub fn from_parts(
        routes: ScheduledExpertRoutes,
        experts: Vec<T>,
    ) -> Result<ScheduledExpertBatch<T>> {
        if experts.len() != routes.routes.len() {
            bail!(
                "scheduled expert batch has {} experts for {} routes on layer {}",
                experts.len(),
                routes.routes.len(),
                routes.layer
            );
        }
        if routes.weights.len() != routes.routes.len() {
            bail!(
                "scheduled expert batch has {} weights for {} routes on layer {}",
                routes.weights.len(),
                routes.routes.len(),
                routes.layer
            );
        }
        Ok(Self {
            layer: routes.layer,
            routes: routes.routes,
            weights: routes.weights,
            experts: Arc::from(experts),
        })
    }

    pub fn len(&self) -> usize {
        self.experts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }
}

impl<T> ScheduledExpertBatch<T>
where
    T: ScheduledCmd3ExpertPayload,
{
    pub(crate) fn cmd3_expert_phase_payloads(
        &self,
        width: usize,
    ) -> Result<Vec<ScheduledExpertPhaseMlpPayload<'_>>> {
        self.experts
            .iter()
            .map(|expert| expert.scheduled_cmd3_expert_phase_payload(width))
            .collect()
    }
}

pub type ScheduledExpertSet<T> = ScheduledExpertBatch<T>;

pub(crate) struct PendingScheduledRead<T> {
    id: u64,
    rx: mpsc::Receiver<T>,
}

impl<T> fmt::Debug for PendingScheduledRead<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingScheduledRead")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<T> PendingScheduledRead<T> {
    pub(crate) fn new(id: u64, rx: mpsc::Receiver<T>) -> Self {
        Self { id, rx }
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn recv(self) -> Result<T, mpsc::RecvError> {
        self.rx.recv()
    }
}

pub(crate) struct PendingScheduledExpertSet<T> {
    routes: ScheduledExpertRoutes,
    reads: Vec<PendingScheduledRead<T>>,
}

impl<T> fmt::Debug for PendingScheduledExpertSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingScheduledExpertSet")
            .field("layer", &self.routes.layer)
            .field("routes", &self.routes)
            .field("read_count", &self.reads.len())
            .finish()
    }
}

impl<T> PendingScheduledExpertSet<T> {
    pub(crate) fn new(routes: ScheduledExpertRoutes, reads: Vec<PendingScheduledRead<T>>) -> Self {
        Self { routes, reads }
    }

    pub(crate) fn into_parts(self) -> (ScheduledExpertRoutes, Vec<PendingScheduledRead<T>>) {
        (self.routes, self.reads)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ScheduledExpertReadSet {
    routes: ScheduledExpertRoutes,
    issues: Vec<ScheduledExpertReadIssue>,
}

impl ScheduledExpertReadSet {
    pub(crate) fn layer(&self) -> usize {
        self.routes.layer
    }

    pub(crate) fn len(&self) -> usize {
        self.issues.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.issues.is_empty()
    }

    pub(crate) fn issues(&self) -> &[ScheduledExpertReadIssue] {
        &self.issues
    }

    pub(crate) fn into_routes(self) -> ScheduledExpertRoutes {
        self.routes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ExpertReadKey {
    pub(crate) layer: usize,
    pub(crate) expert: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledExpertReadIssue {
    pub(crate) id: u64,
    pub(crate) key: ExpertReadKey,
    pub(crate) warm: bool,
    pub(crate) issued_at: Instant,
}

#[derive(Debug)]
pub(crate) struct ScheduledExpertReadResponse<T> {
    pub(crate) id: u64,
    pub(crate) queue_latency: Duration,
    pub(crate) read_path: ExpertReadPath,
    pub(crate) read_latency: Duration,
    pub(crate) bytes_read: u64,
    pub(crate) warm: bool,
    pub(crate) result: Result<T>,
}

#[derive(Debug)]
pub(crate) struct ScheduledExpertSlot {
    raw: ExpertRawRead,
}

impl ScheduledExpertSlot {
    fn from_raw(raw: ExpertRawRead) -> Self {
        Self { raw }
    }

    pub(crate) fn layer(&self) -> usize {
        self.raw.layer
    }

    pub(crate) fn expert(&self) -> usize {
        self.raw.expert
    }

    pub(crate) fn descriptor(&self) -> ExpertSlotDescriptor {
        self.raw.slot
    }

    pub(crate) fn mix_hash(&self) -> u64 {
        let mut hash = ((self.layer() as u64) << 32) ^ self.expert() as u64;
        let prefix = match &self.raw.payload {
            ExpertRawPayload::Pbq4(bytes) => bytes.as_slice(),
            ExpertRawPayload::FixedQ4(fixed_q4) => fixed_q4.bytes.as_slice(),
            ExpertRawPayload::FixedDense(fixed_dense) => fixed_dense.bytes.as_slice(),
            ExpertRawPayload::FixedDeepSeekGguf(deepseek) => deepseek.bytes.as_slice(),
        };
        for byte in prefix.iter().take(4096) {
            hash = hash.rotate_left(5) ^ u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash
    }
}

impl ScheduledCmd3Expert for ScheduledExpertSlot {
    fn scheduled_expert_layer(&self) -> usize {
        self.layer()
    }

    fn scheduled_expert_id(&self) -> usize {
        self.expert()
    }

    fn scheduled_expert_slot_descriptor(&self) -> ExpertSlotDescriptor {
        self.descriptor()
    }
}

impl ScheduledCmd3ExpertPayload for ScheduledExpertSlot {
    fn scheduled_cmd3_expert_phase_payload(
        &self,
        width: usize,
    ) -> Result<ScheduledExpertPhaseMlpPayload<'_>> {
        match &self.raw.payload {
            ExpertRawPayload::FixedQ4(fixed_q4) => {
                let gate = fixed_q4.matvec_payload(
                    ExpertMlpProjection::Gate,
                    width,
                    fixed_q4.spec.intermediate_size,
                );
                let up = fixed_q4.matvec_payload(
                    ExpertMlpProjection::Up,
                    width,
                    fixed_q4.spec.intermediate_size,
                );
                let Some((gate, up)) = gate.zip(up) else {
                    bail!(
                        "FlashMoe unsupported active expert CMD3 path: scheduler-owned fixed-Q4 slot layer {} expert {} does not provide gate/up payloads for width {width}",
                        self.layer(),
                        self.expert()
                    );
                };
                let Some(down) =
                    fixed_q4.matvec_payload(ExpertMlpProjection::Down, gate.rows, width)
                else {
                    bail!(
                        "FlashMoe unsupported active expert CMD3 path: scheduler-owned fixed-Q4 slot layer {} expert {} does not provide down payload for width {width}",
                        self.layer(),
                        self.expert()
                    );
                };
                Ok(ScheduledExpertPhaseMlpPayload::Q4(
                    ScheduledQ4ExpertPhaseMlpPayload::new(
                        self.layer(),
                        self.expert(),
                        width,
                        gate,
                        up,
                        down,
                    )?,
                ))
            }
            ExpertRawPayload::FixedDense(fixed_dense) => {
                let intermediate = fixed_dense.spec.intermediate_size;
                let gate =
                    fixed_dense.matvec_payload(ExpertMlpProjection::Gate, width, intermediate)?;
                let up =
                    fixed_dense.matvec_payload(ExpertMlpProjection::Up, width, intermediate)?;
                let down =
                    fixed_dense.matvec_payload(ExpertMlpProjection::Down, intermediate, width)?;
                Ok(ScheduledExpertPhaseMlpPayload::Dense(
                    ScheduledDenseExpertPhaseMlpPayload::new(
                        self.layer(),
                        self.expert(),
                        width,
                        gate,
                        up,
                        down,
                    )?,
                ))
            }
            ExpertRawPayload::FixedDeepSeekGguf(deepseek) => {
                Ok(ScheduledExpertPhaseMlpPayload::DeepSeekGguf(
                    ScheduledDeepSeekGgufExpertPhaseMlpPayload::new(
                        self.layer(),
                        self.expert(),
                        deepseek.spec,
                        &deepseek.bytes,
                        width,
                    )?,
                ))
            }
            ExpertRawPayload::Pbq4(_) => {
                bail!(
                    "FlashMoe unsupported active expert CMD3 path: scheduler-owned layer {} expert {} slot contains PBQ4/component import data instead of a resolved whole-expert payload",
                    self.layer(),
                    self.expert()
                )
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct ActiveExpertReadScheduler {
    metrics: ExpertSchedulerMetrics,
    seen_reads: BTreeSet<ExpertReadKey>,
    next_read_id: u64,
    routing_weight_normalization: QwenMoeRoutingWeightNormalization,
    routed_expert_scale: f32,
}

impl ActiveExpertReadScheduler {
    #[cfg(test)]
    pub(crate) fn new(routed_expert_scale: f32) -> Self {
        Self::new_with_routing_policy(
            QwenMoeRoutingWeightNormalization::RenormalizeSelected,
            routed_expert_scale,
        )
    }

    pub(crate) fn new_with_routing_policy(
        routing_weight_normalization: QwenMoeRoutingWeightNormalization,
        routed_expert_scale: f32,
    ) -> Self {
        assert_eq!(
            FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
            ExpertReadPath::PositionedRead,
            "expert files must be read with positioned reads"
        );
        assert!(
            routed_expert_scale.is_finite() && routed_expert_scale > 0.0,
            "routed expert scale must be positive and finite"
        );
        assert!(
            !FLASHMOE_EXPERT_IO_POLICY.application_expert_cache,
            "do not add an application-level expert cache; trust the OS page cache"
        );
        assert!(
            !FLASHMOE_EXPERT_IO_POLICY.lz4_expert_compression,
            "do not add LZ4 expert compression"
        );
        assert!(
            !FLASHMOE_EXPERT_IO_POLICY.speculative_routing,
            "do not add speculative expert routing"
        );
        assert!(
            !FLASHMOE_EXPERT_IO_POLICY.broad_ssd_gpu_overlap,
            "do not broadly overlap SSD expert reads with GPU compute"
        );
        Self {
            metrics: ExpertSchedulerMetrics::default(),
            seen_reads: BTreeSet::new(),
            next_read_id: 0,
            routing_weight_normalization,
            routed_expert_scale,
        }
    }

    pub(crate) fn issue_read(&mut self, layer: usize, expert: usize) -> ScheduledExpertReadIssue {
        let key = ExpertReadKey { layer, expert };
        let warm = !self.seen_reads.insert(key);
        self.metrics.record_issued_read();
        let id = self.next_read_id;
        self.next_read_id = self.next_read_id.wrapping_add(1);
        ScheduledExpertReadIssue {
            id,
            key,
            warm,
            issued_at: Instant::now(),
        }
    }

    pub(crate) fn finish_read<T>(
        &mut self,
        pending_id: u64,
        response: ScheduledExpertReadResponse<T>,
    ) -> Result<T> {
        if response.id != pending_id {
            self.metrics.record_read_failure();
            bail!(
                "expert I/O worker returned response {} for pending read {}",
                response.id,
                pending_id
            );
        }
        self.metrics.record_queue_latency(response.queue_latency);
        match response.read_path {
            ExpertReadPath::PositionedRead => {
                self.metrics.record_positioned_read();
            }
        }
        self.metrics.record_read_latency(response.read_latency);
        self.metrics.record_bytes_read(response.bytes_read);
        if response.warm {
            self.metrics
                .record_warm_read(response.read_latency, response.bytes_read);
        }
        response.result.inspect_err(|_| {
            self.metrics.record_read_failure();
        })
    }

    pub(crate) fn finish_slot_read(
        &mut self,
        pending_id: u64,
        response: ExpertRawReadResponse,
    ) -> Result<ScheduledExpertSlot> {
        self.finish_read(
            pending_id,
            ScheduledExpertReadResponse {
                id: response.id,
                queue_latency: response.queue_latency,
                read_path: response.read_path,
                read_latency: response.read_latency,
                bytes_read: response.bytes_read,
                warm: response.warm,
                result: response.result.map(ScheduledExpertSlot::from_raw),
            },
        )
    }

    pub(crate) fn scheduled_routes_from_command(
        &self,
        command: &ScheduledRoutingCommand,
    ) -> Result<ScheduledExpertRoutes> {
        ScheduledExpertRoutes::from_routing_command_with_policy(
            command,
            self.routing_weight_normalization,
            self.routed_expert_scale,
        )
    }

    pub(crate) fn issue_routed_reads(
        &mut self,
        command: &ScheduledRoutingCommand,
    ) -> Result<ScheduledExpertReadSet> {
        let routes = self.scheduled_routes_from_command(command)?;
        let issues = routes
            .routes
            .iter()
            .map(|route| self.issue_read(routes.layer, route.expert))
            .collect();
        Ok(ScheduledExpertReadSet { routes, issues })
    }

    pub(crate) fn finish_routes<T>(
        &mut self,
        scheduled_routes: ScheduledExpertRoutes,
        experts: Vec<T>,
        mut identify: impl FnMut(&T) -> (usize, usize),
    ) -> Result<ScheduledExpertSet<T>> {
        if experts.len() != scheduled_routes.routes.len() {
            bail!(
                "expert scheduler returned {} experts for {} routed entries on layer {}",
                experts.len(),
                scheduled_routes.routes.len(),
                scheduled_routes.layer
            );
        }
        for (route, expert) in scheduled_routes.routes.iter().zip(experts.iter()) {
            let (expert_layer, expert_id) = identify(expert);
            if expert_layer != scheduled_routes.layer || expert_id != route.expert {
                bail!(
                    "expert scheduler returned layer {} expert {} for routed layer {} expert {}",
                    expert_layer,
                    expert_id,
                    scheduled_routes.layer,
                    route.expert
                );
            }
        }
        ScheduledExpertSet::from_parts(scheduled_routes, experts)
    }

    pub(crate) fn snapshot(&self) -> ExpertSchedulerSnapshot {
        self.metrics.snapshot()
    }
}

#[derive(Debug)]
pub(crate) struct ScheduledExpertReadCoordinator {
    store: ExpertSlotStore,
    pool: ExpertReadWorkerPool,
    core: ActiveExpertReadScheduler,
}

impl ScheduledExpertReadCoordinator {
    #[cfg(test)]
    pub(crate) fn new(store: ExpertSlotStore) -> Self {
        Self::new_with_routed_expert_scale(store, 1.0)
    }

    #[cfg(test)]
    pub(crate) fn new_with_routed_expert_scale(
        store: ExpertSlotStore,
        routed_expert_scale: f32,
    ) -> Self {
        Self::new_with_routing_policy(
            store,
            QwenMoeRoutingWeightNormalization::RenormalizeSelected,
            routed_expert_scale,
        )
    }

    pub(crate) fn new_with_routing_policy(
        store: ExpertSlotStore,
        routing_weight_normalization: QwenMoeRoutingWeightNormalization,
        routed_expert_scale: f32,
    ) -> Self {
        Self {
            store,
            pool: ExpertReadWorkerPool::default(),
            core: ActiveExpertReadScheduler::new_with_routing_policy(
                routing_weight_normalization,
                routed_expert_scale,
            ),
        }
    }

    #[cfg(test)]
    pub(crate) fn issue(
        &mut self,
        layer: usize,
        experts: &[usize],
    ) -> Result<Vec<PendingScheduledRead<ExpertRawReadResponse>>> {
        if experts.is_empty() {
            return Ok(Vec::new());
        }
        self.pool.ensure_workers(experts.len().max(1));
        let reader = self.store.layer_reader(layer)?;
        let mut pending = Vec::with_capacity(experts.len());
        for expert in experts {
            let plan = reader.prepare_read(*expert)?;
            let issue = self.core.issue_read(layer, *expert);
            let rx = self.pool.submit_read(
                issue.id,
                issue.key.expert,
                Arc::clone(&reader),
                plan,
                issue.warm,
                issue.issued_at,
            )?;
            pending.push(PendingScheduledRead::new(issue.id, rx));
        }
        Ok(pending)
    }

    pub(crate) fn issue_routing_command(
        &mut self,
        command: &ScheduledRoutingCommand,
    ) -> Result<PendingScheduledExpertSet<ExpertRawReadResponse>> {
        let issued = self.core.issue_routed_reads(command)?;
        let reads = self.submit_issued_reads(&issued)?;
        let routes = issued.into_routes();
        Ok(PendingScheduledExpertSet::new(routes, reads))
    }

    fn submit_issued_reads(
        &mut self,
        issued: &ScheduledExpertReadSet,
    ) -> Result<Vec<PendingScheduledRead<ExpertRawReadResponse>>> {
        if issued.is_empty() {
            return Ok(Vec::new());
        }
        self.pool.ensure_workers(issued.len().max(1));
        let reader = self.store.layer_reader(issued.layer())?;
        let mut pending = Vec::with_capacity(issued.len());
        // Submit positioned reads directly into reusable whole-expert slots; the OS page cache
        // remains the cache policy for this stage.
        for issue in issued.issues() {
            let plan = reader.prepare_read(issue.key.expert)?;
            let rx = self.pool.submit_read(
                issue.id,
                issue.key.expert,
                Arc::clone(&reader),
                plan,
                issue.warm,
                issue.issued_at,
            )?;
            pending.push(PendingScheduledRead::new(issue.id, rx));
        }
        Ok(pending)
    }

    pub(crate) fn finish(
        &mut self,
        pending: Vec<PendingScheduledRead<ExpertRawReadResponse>>,
    ) -> Result<Vec<Arc<ScheduledExpertSlot>>> {
        let mut out = Vec::with_capacity(pending.len());
        for pending in pending {
            let pending_id = pending.id();
            let response = pending
                .recv()
                .context("expert I/O worker dropped response channel")?;
            let slot = self.core.finish_slot_read(pending_id, response)?;
            out.push(Arc::new(slot));
        }
        Ok(out)
    }

    pub(crate) fn finish_routes(
        &mut self,
        pending: PendingScheduledExpertSet<ExpertRawReadResponse>,
    ) -> Result<ScheduledExpertSet<Arc<ScheduledExpertSlot>>> {
        let (routes, reads) = pending.into_parts();
        let experts = self.finish(reads)?;
        self.core
            .finish_routes(routes, experts, |expert| (expert.layer(), expert.expert()))
    }

    pub(crate) fn snapshot(&self) -> ExpertSchedulerSnapshot {
        self.core.snapshot()
    }

    #[cfg(test)]
    pub(crate) fn worker_count(&self) -> usize {
        self.pool.worker_count()
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ExpertSchedulerMetrics {
    issued_reads: u64,
    positioned_reads: u64,
    read_failures: u64,
    total_queue_latency: Duration,
    max_queue_latency: Duration,
    total_read_latency: Duration,
    max_read_latency: Duration,
    bytes_read: u64,
    warm_reads: u64,
    total_warm_read_latency: Duration,
    max_warm_read_latency: Duration,
    warm_bytes_read: u64,
}

impl ExpertSchedulerMetrics {
    pub(crate) fn record_issued_read(&mut self) {
        self.issued_reads = self.issued_reads.saturating_add(1);
    }

    pub(crate) fn record_positioned_read(&mut self) {
        self.positioned_reads = self.positioned_reads.saturating_add(1);
    }

    pub(crate) fn record_read_failure(&mut self) {
        self.read_failures = self.read_failures.saturating_add(1);
    }

    pub(crate) fn record_queue_latency(&mut self, latency: Duration) {
        self.total_queue_latency += latency;
        self.max_queue_latency = self.max_queue_latency.max(latency);
    }

    pub(crate) fn record_read_latency(&mut self, latency: Duration) {
        self.total_read_latency += latency;
        self.max_read_latency = self.max_read_latency.max(latency);
    }

    pub(crate) fn record_bytes_read(&mut self, bytes: u64) {
        self.bytes_read = self.bytes_read.saturating_add(bytes);
    }

    pub(crate) fn record_warm_read(&mut self, latency: Duration, bytes: u64) {
        self.warm_reads = self.warm_reads.saturating_add(1);
        self.total_warm_read_latency += latency;
        self.max_warm_read_latency = self.max_warm_read_latency.max(latency);
        self.warm_bytes_read = self.warm_bytes_read.saturating_add(bytes);
    }

    pub(crate) fn snapshot(&self) -> ExpertSchedulerSnapshot {
        ExpertSchedulerSnapshot {
            issued_reads: self.issued_reads,
            positioned_reads: self.positioned_reads,
            read_failures: self.read_failures,
            total_queue_latency: self.total_queue_latency,
            max_queue_latency: self.max_queue_latency,
            total_read_latency: self.total_read_latency,
            max_read_latency: self.max_read_latency,
            bytes_read: self.bytes_read,
            warm_reads: self.warm_reads,
            total_warm_read_latency: self.total_warm_read_latency,
            max_warm_read_latency: self.max_warm_read_latency,
            warm_bytes_read: self.warm_bytes_read,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertSchedulerSnapshot {
    pub issued_reads: u64,
    pub positioned_reads: u64,
    pub read_failures: u64,
    pub total_queue_latency: Duration,
    pub max_queue_latency: Duration,
    pub total_read_latency: Duration,
    pub max_read_latency: Duration,
    pub bytes_read: u64,
    pub warm_reads: u64,
    pub total_warm_read_latency: Duration,
    pub max_warm_read_latency: Duration,
    pub warm_bytes_read: u64,
}

impl ExpertSchedulerSnapshot {
    pub fn saturating_delta(self, before: Self) -> Self {
        Self {
            issued_reads: self.issued_reads.saturating_sub(before.issued_reads),
            positioned_reads: self
                .positioned_reads
                .saturating_sub(before.positioned_reads),
            read_failures: self.read_failures.saturating_sub(before.read_failures),
            total_queue_latency: self
                .total_queue_latency
                .saturating_sub(before.total_queue_latency),
            max_queue_latency: self.max_queue_latency,
            total_read_latency: self
                .total_read_latency
                .saturating_sub(before.total_read_latency),
            max_read_latency: self.max_read_latency,
            bytes_read: self.bytes_read.saturating_sub(before.bytes_read),
            warm_reads: self.warm_reads.saturating_sub(before.warm_reads),
            total_warm_read_latency: self
                .total_warm_read_latency
                .saturating_sub(before.total_warm_read_latency),
            max_warm_read_latency: self.max_warm_read_latency,
            warm_bytes_read: self.warm_bytes_read.saturating_sub(before.warm_bytes_read),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::flashmoe::experts::{
        DenseExpertDtype, EXPERT_SCALE_BIAS_DTYPE_F32, ExpertLayerPackMetadata, ExpertPackMetadata,
        ExpertPackRecord, ExpertRawPayload, ExpertSlotSpec, ExpertStoreExecutionDescriptor,
        FixedDenseExpertPayload, FixedDenseExpertSlotSpec, FixedQ4ExpertPayload,
        FixedQ4ExpertSlotSpec, PBQ4_EXPERT_MAGIC, expert_layer_path,
        write_expert_metadata_atomically,
    };
    use crate::inference::flashmoe::math::causal_attention;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use crate::inference::flashmoe::metal::{
        MetalExecutionContext, MetalPostAttentionPrep, MetalScheduledCmd3Builder,
    };
    use crate::inference::flashmoe::model_family::{
        QwenMoeExpertComponentKind, QwenMoeExpertComponentLayout, QwenMoeQ4ExpertLayout,
    };
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use crate::inference::flashmoe::runtime::ExpertPhaseInput;
    use crate::inference::flashmoe::state::{
        FlashMoeExpertPhaseApplication, FlashMoeGpuBufferDescriptor, FlashMoeTokenState,
    };
    use crate::inference::flashmoe::weights::{
        DenseMmapMatvecProjection, DenseQ4MmapMatvecProjection, ResidentMmapMatvecProjection,
        RouterScoreProjectionBinding, RouterScoreProjectionDescriptor,
        SharedExpertPhaseResidentProjections, SharedExpertPhaseWeights,
    };
    use crate::inference::flashmoe::{
        GROUP_SIZE, QWEN35_MODEL, QwenModelConfig, QwenMoeModelLayout,
    };
    use std::{fs, path::Path};

    fn qwen35_layout() -> QwenMoeModelLayout {
        let config: QwenModelConfig = serde_json::from_slice(
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
        )
        .unwrap();
        QwenMoeModelLayout::from_config(QWEN35_MODEL, &config).unwrap()
    }

    fn qwen3_moe_layout() -> QwenMoeModelLayout {
        let config: QwenModelConfig = serde_json::from_slice(
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
  "moe_intermediate_size": 768,
  "norm_topk_prob": true
}"#,
        )
        .unwrap();
        QwenMoeModelLayout::from_config("hf://Qwen/Qwen3-30B-A3B", &config).unwrap()
    }

    fn pbq4_import_store(experts: &[usize]) -> (tempfile::TempDir, ExpertSlotStore) {
        let temp = tempfile::tempdir().unwrap();
        let mut packs = Vec::new();
        for expert in experts {
            let tensor = format!("model.layers.0.mlp.experts.{expert}.down_proj.weight");
            let mut bytes = PBQ4_EXPERT_MAGIC.to_vec();
            let record_offset = bytes.len() as u64;
            bytes.extend_from_slice(&(tensor.len() as u32).to_le_bytes());
            bytes.extend_from_slice(tensor.as_bytes());
            bytes.extend_from_slice(&2u64.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&0.5f32.to_le_bytes());
            bytes.extend_from_slice(&1.0f32.to_le_bytes());
            bytes.extend_from_slice(&[0x21, 0x43]);
            let metadata = ExpertPackMetadata {
                layer: 0,
                expert: *expert,
                packed_bytes: bytes.len() as u64,
                records: vec![ExpertPackRecord {
                    tensor,
                    dtype: "F32".to_string(),
                    shape: vec![1, 4],
                    source_offsets: [0, 4],
                    source_hash: Some(format!("fixture-{expert}")),
                    record_offset,
                    packed_bytes: 2,
                    groups: 1,
                    group_size: GROUP_SIZE,
                    scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                }],
            };
            packs.push((*expert, bytes, metadata));
        }
        let slot_size = packs
            .iter()
            .map(|(_, bytes, _)| bytes.len())
            .max()
            .unwrap_or(1);
        let expert_count = experts.iter().copied().max().unwrap_or(0) + 1;
        let mut layer = vec![0; slot_size * expert_count];
        let mut metadata = Vec::new();
        for (expert, bytes, pack) in packs {
            let offset = expert * slot_size;
            layer[offset..offset + bytes.len()].copy_from_slice(&bytes);
            metadata.push(pack);
        }
        fs::write(expert_layer_path(temp.path(), 0), layer).unwrap();
        write_expert_metadata_atomically(
            temp.path(),
            0,
            &ExpertLayerPackMetadata::new(0, slot_size as u64, expert_count, metadata),
        )
        .unwrap();
        let store = ExpertSlotStore::open(temp.path().to_path_buf()).unwrap();
        (temp, store)
    }

    fn tiny_fixed_q4_layout() -> QwenMoeQ4ExpertLayout {
        use QwenMoeExpertComponentKind::*;
        QwenMoeQ4ExpertLayout {
            expert_bytes: 48,
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
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownBias,
                    offset: 44,
                    bytes: 4,
                },
            ],
        }
    }

    fn raw_pbq4_read(layer: usize, expert: usize, payload: Vec<u8>) -> ExpertRawRead {
        ExpertRawRead {
            layer,
            expert,
            slot: ExpertSlotDescriptor {
                layer,
                expert,
                slot_offset: 1024,
                slot_capacity: payload.len(),
                payload_len: payload.len(),
            },
            metadata: ExpertPackMetadata {
                layer,
                expert,
                packed_bytes: payload.len() as u64,
                records: Vec::new(),
            },
            payload: ExpertRawPayload::Pbq4(payload),
            read_latency: Duration::from_millis(7),
            read_path: ExpertReadPath::PositionedRead,
        }
    }

    fn raw_fixed_q4_read(layer: usize, expert: usize) -> ExpertRawRead {
        let fixed_q4 = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
        let payload = FixedQ4ExpertPayload::from_whole_slot(
            fixed_q4,
            vec![0; fixed_q4.layout.expert_bytes],
            None,
        )
        .unwrap();
        ExpertRawRead {
            layer,
            expert,
            slot: ExpertSlotDescriptor {
                layer,
                expert,
                slot_offset: 512,
                slot_capacity: fixed_q4.layout.expert_bytes,
                payload_len: fixed_q4.layout.expert_bytes,
            },
            metadata: ExpertPackMetadata {
                layer,
                expert,
                packed_bytes: fixed_q4.layout.expert_bytes as u64,
                records: Vec::new(),
            },
            payload: ExpertRawPayload::FixedQ4(payload),
            read_latency: Duration::from_millis(3),
            read_path: ExpertReadPath::PositionedRead,
        }
    }

    fn raw_fixed_dense_read(layer: usize, expert: usize, dtype: DenseExpertDtype) -> ExpertRawRead {
        let spec = FixedDenseExpertSlotSpec::new(dtype, 2, 2).unwrap();
        let payload =
            FixedDenseExpertPayload::from_whole_slot(spec, vec![0; spec.expert_bytes], None)
                .unwrap();
        ExpertRawRead {
            layer,
            expert,
            slot: ExpertSlotDescriptor {
                layer,
                expert,
                slot_offset: 512,
                slot_capacity: spec.expert_bytes,
                payload_len: spec.expert_bytes,
            },
            metadata: ExpertPackMetadata {
                layer,
                expert,
                packed_bytes: spec.expert_bytes as u64,
                records: Vec::new(),
            },
            payload: ExpertRawPayload::FixedDense(payload),
            read_latency: Duration::from_millis(3),
            read_path: ExpertReadPath::PositionedRead,
        }
    }

    fn identity_fixed_q4_slot_bytes() -> Vec<u8> {
        let layout = tiny_fixed_q4_layout();
        let mut bytes = vec![0u8; layout.expert_bytes];
        let one_bf16 = 0x3f80u16.to_le_bytes();
        for (weight_offset, scale_offset) in [(0, 8), (16, 24), (32, 40)] {
            // Row-major 2x2 identity, low nibble first. Remaining component
            // bytes are fixed-slot padding and must stay addressable.
            bytes[weight_offset] = 0x01;
            bytes[weight_offset + 1] = 0x10;
            bytes[scale_offset..scale_offset + 2].copy_from_slice(&one_bf16);
            bytes[scale_offset + 2..scale_offset + 4].copy_from_slice(&one_bf16);
        }
        bytes
    }

    fn write_identity_fixed_q4_layer(root: &std::path::Path, layer: usize, experts: usize) {
        let slot = identity_fixed_q4_slot_bytes();
        let bytes = slot.repeat(experts);
        std::fs::write(expert_layer_path(root, layer), bytes).unwrap();
        let metadata = ExpertLayerPackMetadata::new_fixed_q4(
            layer,
            slot.len() as u64,
            experts,
            (0..experts)
                .map(|expert| ExpertPackMetadata {
                    layer,
                    expert,
                    packed_bytes: slot.len() as u64,
                    records: Vec::new(),
                })
                .collect(),
        );
        write_expert_metadata_atomically(root, layer, &metadata).unwrap();
    }

    fn reference_bf16(bytes: &[u8], group: usize) -> f32 {
        let offset = group * 2;
        f32::from_bits((u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as u32) << 16)
    }

    fn reference_q4_matvec(payload: &Q4MatvecPayload<'_>, input: &[f32]) -> Vec<f32> {
        let row_bytes = payload.cols.div_ceil(2);
        let groups_per_row = payload.cols.div_ceil(payload.group_size);
        (0..payload.rows)
            .map(|row| {
                (0..payload.cols)
                    .map(|col| {
                        let byte = payload.packed[row * row_bytes + col / 2];
                        let quantized = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                        let group = row * groups_per_row + col / payload.group_size;
                        let scale = reference_bf16(payload.scale_bytes, group);
                        let bias = reference_bf16(payload.bias_bytes, group);
                        (quantized as f32 * scale + bias) * input[col]
                    })
                    .sum()
            })
            .collect()
    }

    fn reference_q4_swiglu(
        payload: &ScheduledQ4ExpertPhaseMlpPayload<'_>,
        input: &[f32],
    ) -> Vec<f32> {
        let gate = reference_q4_matvec(&payload.gate, input);
        let up = reference_q4_matvec(&payload.up, input);
        let intermediate: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect();
        reference_q4_matvec(&payload.down, &intermediate)
    }

    fn reference_rms_norm(values: &[f32], weights: &[f32]) -> Vec<f32> {
        let mean_square =
            values.iter().map(|value| value * value).sum::<f32>() / values.len().max(1) as f32;
        let scale = (mean_square + 1e-6).sqrt().recip();
        values
            .iter()
            .zip(weights.iter())
            .map(|(value, weight)| value * scale * weight)
            .collect()
    }

    #[derive(Debug, Clone, Copy)]
    struct DummyCmd3Input {
        source: ScheduledCmd3InputSource,
        width: usize,
    }

    #[derive(Debug, Clone, Copy)]
    struct DummyCmd3InputState {
        source: ScheduledCmd3InputSource,
        state: FlashMoeCmd3InputState,
    }

    impl ScheduledCmd3Input for DummyCmd3Input {
        fn scheduled_cmd3_input_source(&self) -> ScheduledCmd3InputSource {
            self.source
        }

        fn scheduled_cmd3_input_state(&self, layer: usize) -> FlashMoeCmd3InputState {
            match self.source {
                ScheduledCmd3InputSource::CpuNormedResidualUpload => {
                    FlashMoeCmd3InputState::cpu_normed_residual(layer, self.width, self.width)
                }
                ScheduledCmd3InputSource::MetalPostAttentionPrep => {
                    FlashMoeCmd3InputState::metal_post_attention_prep(
                        layer,
                        FlashMoePostAttentionPrepState::new(layer, self.width, 16, 4),
                    )
                }
            }
        }
    }

    impl ScheduledCmd3Input for DummyCmd3InputState {
        fn scheduled_cmd3_input_source(&self) -> ScheduledCmd3InputSource {
            self.source
        }

        fn scheduled_cmd3_input_state(&self, _layer: usize) -> FlashMoeCmd3InputState {
            self.state
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct DummySharedExpert {
        source: ScheduledSharedExpertSource,
        shape: Option<ScheduledSharedExpertShape>,
    }

    impl ScheduledSharedExpert for DummySharedExpert {
        fn scheduled_shared_expert_descriptor(&self) -> Result<ScheduledSharedExpertDescriptor> {
            ScheduledSharedExpertDescriptor::new(self.source, self.shape)
        }
    }

    fn dummy_cmd3_input(source: ScheduledCmd3InputSource) -> DummyCmd3Input {
        DummyCmd3Input { source, width: 8 }
    }

    fn dummy_cmd3_input_with_width(
        source: ScheduledCmd3InputSource,
        width: usize,
    ) -> DummyCmd3Input {
        DummyCmd3Input { source, width }
    }

    fn dummy_shared_expert(source: ScheduledSharedExpertSource) -> DummySharedExpert {
        let shape = match source {
            ScheduledSharedExpertSource::None => None,
            ScheduledSharedExpertSource::DenseCpuWeights
            | ScheduledSharedExpertSource::ResidentProjections => {
                Some(ScheduledSharedExpertShape::new(8, 2, 2).unwrap())
            }
        };
        DummySharedExpert { source, shape }
    }

    fn dummy_shared_expert_with_shape(
        source: ScheduledSharedExpertSource,
        shape: Option<ScheduledSharedExpertShape>,
    ) -> DummySharedExpert {
        DummySharedExpert { source, shape }
    }

    fn test_execution_scheduler() -> (tempfile::TempDir, FlashMoeExecutionScheduler) {
        test_execution_scheduler_with_attention(vec![
            ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
            ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
            ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
            ScheduledLayerAttentionImplementation::FullAttentionCpuKv,
            ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
        ])
    }

    fn test_execution_scheduler_with_attention(
        attention_layers: Vec<ScheduledLayerAttentionImplementation>,
    ) -> (tempfile::TempDir, FlashMoeExecutionScheduler) {
        let temp = tempfile::tempdir().unwrap();
        let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
        let store = ExpertSlotStore::open_with_fixed_q4(temp.path().to_path_buf(), spec).unwrap();
        let mut layout = qwen35_layout();
        layout.layers = attention_layers.len();
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
        let resolved_attention = capabilities
            .attention_layers
            .iter()
            .copied()
            .map(ScheduledLayerAttentionImplementation::from)
            .collect::<Vec<_>>();
        assert_eq!(attention_layers, resolved_attention);
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        (temp, FlashMoeExecutionScheduler::new(graph, store).unwrap())
    }

    fn test_dense_execution_scheduler_with_attention(
        dtype: DenseExpertDtype,
        attention_layers: Vec<ScheduledLayerAttentionImplementation>,
    ) -> (tempfile::TempDir, FlashMoeExecutionScheduler) {
        let temp = tempfile::tempdir().unwrap();
        let tiny_spec = FixedDenseExpertSlotSpec::new(dtype, 2, 2).unwrap();
        let store =
            ExpertSlotStore::open_with_fixed_dense(temp.path().to_path_buf(), tiny_spec).unwrap();
        let mut layout = qwen35_layout();
        layout.layers = attention_layers.len();
        let graph_spec = FixedDenseExpertSlotSpec::from_model_layout(&layout, dtype).unwrap();
        let mut capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
        capabilities.expert_storage = ExpertStoreExecutionDescriptor {
            layout: match dtype {
                DenseExpertDtype::Bf16 => ExpertStorageLayout::FixedBf16,
                DenseExpertDtype::F16 => ExpertStorageLayout::FixedF16,
            },
            slot_spec: ExpertSlotSpec::FixedDense(graph_spec),
            layers: layout.layers,
            first_expert_layer: layout.first_sparse_layer,
            experts_per_layer: layout.experts_per_layer,
        };
        let resolved_attention = capabilities
            .attention_layers
            .iter()
            .copied()
            .map(ScheduledLayerAttentionImplementation::from)
            .collect::<Vec<_>>();
        assert_eq!(attention_layers, resolved_attention);
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        (temp, FlashMoeExecutionScheduler::new(graph, store).unwrap())
    }

    fn write_identity_fixed_dense_layer(
        root: &Path,
        layer: usize,
        experts: usize,
        dtype: DenseExpertDtype,
    ) {
        let spec = FixedDenseExpertSlotSpec::new(dtype, 2, 2).unwrap();
        let one = match dtype {
            DenseExpertDtype::Bf16 => 0x3f80u16.to_le_bytes(),
            DenseExpertDtype::F16 => 0x3c00u16.to_le_bytes(),
        };
        let mut bytes = vec![0u8; spec.expert_bytes * experts];
        for expert in 0..experts {
            let slot = expert * spec.expert_bytes;
            for projection in [spec.gate, spec.up, spec.down] {
                let start = slot + projection.offset;
                bytes[start..start + 2].copy_from_slice(&one);
                bytes[start + 6..start + 8].copy_from_slice(&one);
            }
        }
        fs::write(expert_layer_path(root, layer), bytes).unwrap();
        let metadata = ExpertLayerPackMetadata::new_fixed_dense(
            layer,
            spec.expert_bytes as u64,
            experts,
            (0..experts)
                .map(|expert| ExpertPackMetadata {
                    layer,
                    expert,
                    packed_bytes: spec.expert_bytes as u64,
                    records: Vec::new(),
                })
                .collect(),
        );
        write_expert_metadata_atomically(root, layer, &metadata).unwrap();
    }

    fn test_qwen3_execution_scheduler() -> (tempfile::TempDir, FlashMoeExecutionScheduler) {
        let temp = tempfile::tempdir().unwrap();
        let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
        let store = ExpertSlotStore::open_with_fixed_q4(temp.path().to_path_buf(), spec).unwrap();
        let mut layout = qwen3_moe_layout();
        layout.layers = 2;
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        (temp, FlashMoeExecutionScheduler::new(graph, store).unwrap())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_identity_k4_layer(
        scheduler: &mut FlashMoeExecutionScheduler,
        position: usize,
        layer: usize,
        layers: usize,
        previous: ScheduledPreviousCmd3Handoff,
        cmd1_input: ScheduledCmd1InputSource,
        cmd1_state: FlashMoeCmd1InputState,
        residual: &[f32],
        normed: &[f32],
        shared_output: &[f32],
        next_norm_weights: Option<&[f32]>,
    ) -> ScheduledLayerExecution<FlashMoeExpertPhaseOutput> {
        let active_experts = 4;
        let experts = 9;
        let width = residual.len();
        let scheduled = scheduler
            .begin_layer(position, layer, layers, active_experts, previous, true)
            .unwrap();
        let (_, scheduled) = scheduled
            .resolve(scheduler, cmd1_input, cmd1_state)
            .unwrap();
        let (cmd2, scheduled) = scheduled
            .resolve(
                scheduler,
                ScheduledCmd2PhaseInputs::from_inputs(
                    ScheduledCmd2AttentionInput::metal_values(width),
                    ScheduledCmd2ResidualInput::metal_buffer(width),
                ),
            )
            .unwrap();
        let router_scores = [0.1, 2.0, -1.0, 3.0, 0.5, 2.5, -0.2, 1.5, 4.0];
        let active = top_k(&router_scores, active_experts);
        let prep_state = FlashMoePostAttentionPrepState::new(layer, width, experts, active_experts);
        let routing = scheduler
            .routing_from_post_attention_prep(&cmd2, prep_state, &active)
            .unwrap();
        let pending = scheduled
            .resolve(&routing)
            .unwrap()
            .issue_cmd3(scheduler, &routing)
            .unwrap();
        let input = DummyCmd3InputState {
            source: ScheduledCmd3InputSource::MetalPostAttentionPrep,
            state: FlashMoeCmd3InputState::metal_post_attention_prep(layer, prep_state),
        };
        let shared = dummy_shared_expert_with_shape(
            ScheduledSharedExpertSource::ResidentProjections,
            Some(ScheduledSharedExpertShape::new(width, 1, width).unwrap()),
        );
        let next_norm = match next_norm_weights {
            Some(weights) => ScheduledNextNormWeights::cpu_visible(
                "model.layers.next.input_layernorm.weight",
                weights,
                width,
            )
            .unwrap(),
            None => ScheduledNextNormWeights::none(),
        };
        pending
            .finish(scheduler, input, shared, next_norm, |command| {
                let mut expert_output = vec![0.0f32; width];
                for (payload, weight) in command.payloads.iter().zip(command.weights.iter()) {
                    let output = reference_q4_swiglu(payload.q4(), normed);
                    for (combined, value) in expert_output.iter_mut().zip(output.iter()) {
                        *combined += value * weight;
                    }
                }
                let hidden: Vec<f32> = residual
                    .iter()
                    .zip(expert_output.iter())
                    .zip(shared_output.iter())
                    .map(|((residual, expert), shared)| residual + expert + shared)
                    .collect();
                let next_normed = command
                    .next_norm_weights
                    .values()
                    .map(|weights| reference_rms_norm(&hidden, weights));
                command
                    .resolve_output_state()?
                    .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(
                        hidden,
                        next_normed,
                    ))
            })
            .unwrap()
    }

    #[test]
    fn execution_scheduler_resolves_cmd1_cmd2_and_routing_with_one_graph_owner() {
        let (_temp, scheduler) = test_execution_scheduler();
        let layer = scheduler
            .begin_layer(
                17,
                3,
                5,
                2,
                ScheduledPreviousCmd3Handoff::cpu_visible(2, 8),
                true,
            )
            .unwrap();
        let (cmd1, layer) = layer
            .resolve(
                &scheduler,
                ScheduledCmd1InputSource::CpuNormedHidden,
                FlashMoeCmd1InputState::cpu_normed(3, 8),
            )
            .unwrap();
        assert_eq!(cmd1.layer, 3);
        assert_eq!(cmd1.input_state.layer(), 3);

        let (cmd2, layer) = layer
            .resolve(
                &scheduler,
                ScheduledCmd2PhaseInputs::from_inputs(
                    ScheduledCmd2AttentionInput::metal_values(8),
                    ScheduledCmd2ResidualInput::metal_buffer(8),
                ),
            )
            .unwrap();
        let state = FlashMoePostAttentionPrepState::new(3, 8, 5, 2);
        let routing = scheduler
            .routing_from_post_attention_prep(&cmd2, state, &[(4, 3.0), (1, 2.0)])
            .unwrap();
        let routed = layer.resolve(&routing).unwrap();

        assert_eq!(routing.layer, 3);
        assert_eq!(routing.active_experts, 2);
        assert_eq!(routing.routes, vec![(4, 3.0), (1, 2.0)]);
        assert_eq!(
            routed.identity.output_handoff,
            ScheduledCmd3OutputHandoff::DeferredToNextLayer
        );

        let complete_here = scheduler
            .begin_layer(
                17,
                3,
                5,
                2,
                ScheduledPreviousCmd3Handoff::cpu_visible(2, 8),
                false,
            )
            .unwrap();
        assert_eq!(
            complete_here.identity.output_handoff,
            ScheduledCmd3OutputHandoff::CompleteHere
        );
    }

    #[test]
    fn execution_scheduler_rejects_mismatched_previous_cmd3_handoffs() {
        let (_temp, scheduler) = test_execution_scheduler();
        let err = scheduler
            .begin_layer(
                17,
                3,
                5,
                2,
                ScheduledPreviousCmd3Handoff::cpu_visible(1, 8),
                true,
            )
            .unwrap_err();
        assert!(err.to_string().contains("does not feed layer 3"), "{err:#}");

        let layer = scheduler
            .begin_layer(
                17,
                3,
                5,
                2,
                ScheduledPreviousCmd3Handoff::deferred_gpu(
                    2,
                    FlashMoeGpuBufferDescriptor::hidden(8),
                    FlashMoeGpuBufferDescriptor::next_layer_normed(8),
                ),
                true,
            )
            .unwrap();
        let err = layer
            .resolve(
                &scheduler,
                ScheduledCmd1InputSource::CpuNormedHidden,
                FlashMoeCmd1InputState::cpu_normed(3, 8),
            )
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("requires CMD1 input DeferredMetalNextNormed"),
            "{err:#}"
        );
    }

    #[test]
    fn execution_scheduler_finishes_whole_slot_reads_and_submits_cmd3_transaction() {
        let (_temp, mut scheduler) = test_execution_scheduler();
        let layer = scheduler
            .begin_layer(
                19,
                3,
                5,
                1,
                ScheduledPreviousCmd3Handoff::cpu_visible(2, 2),
                true,
            )
            .unwrap();
        let (_, layer) = layer
            .resolve(
                &scheduler,
                ScheduledCmd1InputSource::CpuNormedHidden,
                FlashMoeCmd1InputState::cpu_normed(3, 2),
            )
            .unwrap();
        let (cmd2, layer) = layer
            .resolve(
                &scheduler,
                ScheduledCmd2PhaseInputs::from_inputs(
                    ScheduledCmd2AttentionInput::metal_values(2),
                    ScheduledCmd2ResidualInput::metal_buffer(2),
                ),
            )
            .unwrap();
        let routing = scheduler
            .routing_from_post_attention_prep(
                &cmd2,
                FlashMoePostAttentionPrepState::new(3, 2, 9, 1),
                &[(8, 1.0)],
            )
            .unwrap();
        let routed = layer.resolve(&routing).unwrap();
        let routes = ScheduledExpertRoutes::from_routing_command(&routing, 1.0).unwrap();
        let (tx, rx) = mpsc::channel();
        let pending =
            PendingScheduledExpertSet::new(routes, vec![PendingScheduledRead::new(77, rx)]);
        tx.send(ExpertRawReadResponse {
            id: 77,
            queue_latency: Duration::from_millis(1),
            read_path: ExpertReadPath::PositionedRead,
            read_latency: Duration::from_millis(2),
            bytes_read: tiny_fixed_q4_layout().expert_bytes as u64,
            warm: false,
            result: Ok(raw_fixed_q4_read(3, 8)),
        })
        .unwrap();
        let transaction = PendingScheduledCmd3 {
            before: scheduler.snapshot(),
            pending,
            issue_elapsed: Duration::from_millis(1),
        };
        let shared = dummy_shared_expert_with_shape(
            ScheduledSharedExpertSource::ResidentProjections,
            Some(ScheduledSharedExpertShape::new(2, 1, 2).unwrap()),
        );

        let pending_layer = ScheduledLayerPendingCmd3 {
            identity: routed.identity,
            pending: transaction,
        };
        let execution = pending_layer
            .finish(
                &mut scheduler,
                dummy_cmd3_input_with_width(ScheduledCmd3InputSource::CpuNormedResidualUpload, 2),
                shared,
                ScheduledNextNormWeights::none(),
                |command| {
                    assert_eq!(command.layer, 3);
                    assert_eq!(command.experts.len(), 1);
                    Ok(command.layer)
                },
            )
            .unwrap();
        let cmd3 = execution.cmd3;

        assert_eq!(
            execution.output_handoff,
            ScheduledCmd3OutputHandoff::DeferredToNextLayer
        );
        assert_eq!(cmd3.submission, 3);
        assert_eq!(cmd3.expert_delta.positioned_reads, 1);
        assert_eq!(
            cmd3.expert_delta.bytes_read,
            tiny_fixed_q4_layout().expert_bytes as u64
        );
        assert_eq!(cmd3.expert_mixes.len(), 1);
        assert_eq!(cmd3.expert_mixes[0].1, 1.0);
        assert!(cmd3.expert_io_elapsed >= Duration::from_millis(1));
    }

    #[test]
    fn qwen35_q4_layer_parity_fixture_follows_resolved_k4_transaction() {
        // Golden values are independently derived from the Qwen3.5/Qwen3Next
        // equations: scaled dot-product attention, residual RMSNorm, router
        // topK then selected-score softmax, Q4 SwiGLU, shared addition, RMSNorm.
        let position = 7;
        let layer = 3;
        let experts = 9;
        let active_experts = 4;
        let width = 2;
        let query = [1.0, 0.0];
        let key_0 = [1.0, 0.0];
        let value_0 = [2.0, 1.0];
        let key_1 = [0.0, 1.0];
        let value_1 = [-1.0, 3.0];
        let attention = causal_attention(
            &query,
            &[(&key_0, &value_0), (&key_1, &value_1)],
            1,
            1,
            width,
        );
        for (actual, expected) in attention.iter().zip([1.0092846, 1.6604769]) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
        let residual_input = [0.5, -1.0];
        let residual = [
            residual_input[0] + attention[0],
            residual_input[1] + attention[1],
        ];
        let normed = reference_rms_norm(&residual, &[1.0, 1.0]);
        for (actual, expected) in normed.iter().zip([1.2955897, 0.5669620]) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }

        let router_scores = [0.1, 2.0, -1.0, 3.0, 0.5, 2.5, -0.2, 1.5, 4.0];
        let mut router_probabilities = router_scores.to_vec();
        softmax_in_place(&mut router_probabilities);
        let active = routing_top_k(&router_probabilities, active_experts);
        assert_eq!(
            active.iter().map(|(expert, _)| *expert).collect::<Vec<_>>(),
            vec![5, 1, 8, 3]
        );

        let (temp, mut scheduler) = test_execution_scheduler();
        write_identity_fixed_q4_layer(temp.path(), layer, experts);
        let attention_math = scheduler
            .graph
            .build_attention_math(layer, position)
            .unwrap()
            .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(
                position, layer, width, width,
            ))
            .unwrap();
        assert_eq!(
            attention_math.implementation(),
            ScheduledAttentionMathImplementation::CpuKvCache
        );

        let scheduled = scheduler
            .begin_layer(
                position,
                layer,
                5,
                active_experts,
                ScheduledPreviousCmd3Handoff::deferred_gpu(
                    layer - 1,
                    FlashMoeGpuBufferDescriptor::hidden(width),
                    FlashMoeGpuBufferDescriptor::next_layer_normed(width),
                ),
                true,
            )
            .unwrap();
        let (_, scheduled) = scheduled
            .resolve(
                &scheduler,
                ScheduledCmd1InputSource::DeferredMetalNextNormed,
                FlashMoeCmd1InputState::gpu_next_layer_normed(
                    layer,
                    FlashMoeGpuBufferDescriptor::next_layer_normed(width),
                ),
            )
            .unwrap();
        let (cmd2, scheduled) = scheduled
            .resolve(
                &scheduler,
                ScheduledCmd2PhaseInputs::from_inputs(
                    ScheduledCmd2AttentionInput::metal_values(width),
                    ScheduledCmd2ResidualInput::metal_buffer(width),
                ),
            )
            .unwrap();
        let prep_state = FlashMoePostAttentionPrepState::new(layer, width, experts, active_experts);
        let routing = scheduler
            .routing_from_post_attention_prep(&cmd2, prep_state, &active)
            .unwrap();
        let routed = scheduled.resolve(&routing).unwrap();
        let pending = routed.issue_cmd3(&mut scheduler, &routing).unwrap();
        let next_norm_weights = [1.0, 0.5];
        let shared_output = [0.25, -0.5];
        let input = DummyCmd3InputState {
            source: ScheduledCmd3InputSource::MetalPostAttentionPrep,
            state: FlashMoeCmd3InputState::metal_post_attention_prep(layer, prep_state),
        };
        let shared = dummy_shared_expert_with_shape(
            ScheduledSharedExpertSource::ResidentProjections,
            Some(ScheduledSharedExpertShape::new(width, 1, width).unwrap()),
        );
        let execution = pending
            .finish(
                &mut scheduler,
                input,
                shared,
                ScheduledNextNormWeights::cpu_visible(
                    "model.layers.4.input_layernorm.weight",
                    &next_norm_weights,
                    width,
                )
                .unwrap(),
                |command| {
                    assert_eq!(
                        command
                            .experts
                            .iter()
                            .map(|expert| expert.expert())
                            .collect::<Vec<_>>(),
                        vec![5, 1, 8, 3]
                    );
                    for (actual, expected) in command
                        .weights
                        .iter()
                        .zip([0.12925005, 0.07839412, 0.57925856, 0.2130973])
                    {
                        assert!((actual - expected).abs() <= 1e-6);
                    }
                    let mut expert_output = vec![0.0f32; width];
                    for (payload, weight) in command.payloads.iter().zip(command.weights.iter()) {
                        let output = reference_q4_swiglu(payload.q4(), &normed);
                        for (combined, value) in expert_output.iter_mut().zip(output.iter()) {
                            *combined += value * weight;
                        }
                    }
                    let hidden: Vec<f32> = residual
                        .iter()
                        .zip(expert_output.iter())
                        .zip(shared_output.iter())
                        .map(|((residual, expert), shared)| residual + expert + shared)
                        .collect();
                    for (actual, expected) in hidden.iter().zip([3.0771027, 0.36557937]) {
                        assert!(
                            (actual - expected).abs() <= 1e-5,
                            "{actual} != {expected}; hidden={hidden:?}"
                        );
                    }
                    let next_normed =
                        reference_rms_norm(&hidden, command.next_norm_weights.values().unwrap());
                    command
                        .resolve_output_state()?
                        .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(
                            hidden,
                            Some(next_normed),
                        ))
                },
            )
            .unwrap();

        assert_eq!(
            execution.output_handoff,
            ScheduledCmd3OutputHandoff::DeferredToNextLayer
        );
        assert_eq!(execution.cmd3.expert_delta.positioned_reads, 4);
        assert_eq!(execution.cmd3.expert_delta.bytes_read, 4 * 48);
        let mut token_state = FlashMoeTokenState::new(vec![0.0; width], 0);
        token_state
            .apply_declared_expert_phase(
                execution.cmd3.submission,
                FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
            )
            .unwrap();
        for (actual, expected) in token_state.hidden().iter().zip([3.0771027, 0.36557937]) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
        let next_normed = token_state.take_next_layer_normed_as_normed().unwrap();
        for (actual, expected) in next_normed.iter().zip([1.404337, 0.08342209]) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn qwen3_q4_layer_parity_follows_resolved_k8_no_shared_transaction() {
        // Golden values follow Qwen3 MoE's selected-route normalization with
        // scale 1.0, full attention, fixed-Q4 SwiGLU, and no shared expert.
        let position = 7;
        let layer = 0;
        let experts = 9;
        let active_experts = 8;
        let width = 2;
        let query = [1.0, 0.0];
        let key_0 = [1.0, 0.0];
        let value_0 = [2.0, 1.0];
        let key_1 = [0.0, 1.0];
        let value_1 = [-1.0, 3.0];
        let attention = causal_attention(
            &query,
            &[(&key_0, &value_0), (&key_1, &value_1)],
            1,
            1,
            width,
        );
        let residual = [0.5 + attention[0], -1.0 + attention[1]];
        let normed = reference_rms_norm(&residual, &[1.0, 1.0]);
        let router_scores = [0.1, 2.0, -1.0, 3.0, 0.5, 2.5, -0.2, 1.5, 4.0];
        let mut router_probabilities = router_scores.to_vec();
        softmax_in_place(&mut router_probabilities);
        let active = routing_top_k(&router_probabilities, active_experts);
        assert_eq!(
            active.iter().map(|(expert, _)| *expert).collect::<Vec<_>>(),
            vec![0, 1, 8, 3, 4, 5, 6, 7]
        );

        let (temp, mut scheduler) = test_qwen3_execution_scheduler();
        assert_eq!(scheduler.graph.family(), QwenMoeFamily::Qwen3Moe);
        assert_eq!(scheduler.experts_per_layer(), 128);
        assert_eq!(scheduler.active_experts(), 8);
        assert_eq!(scheduler.graph.routed_expert_scale(), 1.0);
        write_identity_fixed_q4_layer(temp.path(), layer, experts);

        let attention_math = scheduler
            .resolve_attention_math(layer, position)
            .unwrap()
            .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(
                position, layer, width, width,
            ))
            .unwrap();
        assert_eq!(
            attention_math.implementation(),
            ScheduledAttentionMathImplementation::CpuKvCache
        );

        let scheduled = scheduler
            .begin_resolved_layer(
                position,
                layer,
                2,
                ScheduledPreviousCmd3Handoff::initial(width),
                true,
            )
            .unwrap();
        let (_, scheduled) = scheduled
            .resolve(
                &scheduler,
                ScheduledCmd1InputSource::CpuNormedHidden,
                FlashMoeCmd1InputState::cpu_normed(layer, width),
            )
            .unwrap();
        let (cmd2, scheduled) = scheduled
            .resolve(
                &scheduler,
                ScheduledCmd2PhaseInputs::from_inputs(
                    ScheduledCmd2AttentionInput::metal_values(width),
                    ScheduledCmd2ResidualInput::metal_buffer(width),
                ),
            )
            .unwrap();
        assert_eq!(cmd2.active_experts, active_experts);
        let prep_state = FlashMoePostAttentionPrepState::new(layer, width, experts, active_experts);
        let routing = scheduler
            .routing_from_post_attention_prep(&cmd2, prep_state, &active)
            .unwrap();
        let routed = scheduled.resolve(&routing).unwrap();
        let pending = routed.issue_cmd3(&mut scheduler, &routing).unwrap();
        let next_norm_weights = [1.0, 0.5];
        let execution = pending
            .finish(
                &mut scheduler,
                DummyCmd3InputState {
                    source: ScheduledCmd3InputSource::MetalPostAttentionPrep,
                    state: FlashMoeCmd3InputState::metal_post_attention_prep(layer, prep_state),
                },
                dummy_shared_expert(ScheduledSharedExpertSource::None),
                ScheduledNextNormWeights::cpu_visible(
                    "model.layers.1.input_layernorm.weight",
                    &next_norm_weights,
                    width,
                )
                .unwrap(),
                |command| {
                    assert_eq!(command.cmd3.shared, ScheduledSharedExpertSource::None);
                    assert_eq!(
                        command
                            .experts
                            .iter()
                            .map(|expert| expert.expert())
                            .collect::<Vec<_>>(),
                        vec![0, 1, 8, 3, 4, 5, 6, 7]
                    );
                    for (actual, expected) in command.weights.iter().zip([
                        0.010802226,
                        0.072222546,
                        0.5336564,
                        0.19632123,
                        0.016115028,
                        0.11907485,
                        0.008002486,
                        0.04380519,
                    ]) {
                        assert!((actual - expected).abs() <= 1e-6);
                    }
                    let mut expert_output = vec![0.0f32; width];
                    for (payload, weight) in command.payloads.iter().zip(command.weights.iter()) {
                        let output = reference_q4_swiglu(payload.q4(), &normed);
                        for (combined, value) in expert_output.iter_mut().zip(output.iter()) {
                            *combined += value * weight;
                        }
                    }
                    let hidden: Vec<f32> = residual
                        .iter()
                        .zip(expert_output.iter())
                        .map(|(residual, expert)| residual + expert)
                        .collect();
                    for (actual, expected) in hidden.iter().zip([2.8271027, 0.86557925]) {
                        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
                    }
                    let next_normed =
                        reference_rms_norm(&hidden, command.next_norm_weights.values().unwrap());
                    command
                        .resolve_output_state()?
                        .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(
                            hidden,
                            Some(next_normed),
                        ))
                },
            )
            .unwrap();

        assert_eq!(execution.cmd3.expert_delta.positioned_reads, 8);
        assert_eq!(execution.cmd3.expert_delta.bytes_read, 8 * 48);
        let mut token_state = FlashMoeTokenState::new(vec![0.0; width], 0);
        token_state
            .apply_declared_expert_phase(
                execution.cmd3.submission,
                FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
            )
            .unwrap();
        for (actual, expected) in token_state.hidden().iter().zip([2.8271027, 0.86557925]) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
        let next_normed = token_state.take_next_layer_normed_as_normed().unwrap();
        for (actual, expected) in next_normed.iter().zip([1.3522521, 0.20701078]) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn qwen35_q4_multi_linear_layer_parity_preserves_deferred_state_and_logits() {
        let (temp, mut scheduler) = test_execution_scheduler_with_attention(vec![
            ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
            ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
        ]);
        write_identity_fixed_q4_layer(temp.path(), 0, 9);
        write_identity_fixed_q4_layer(temp.path(), 1, 9);
        assert_eq!(
            scheduler.graph.attention_layers.as_ref(),
            [
                ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
                ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
            ]
        );

        let shared_output = [0.25, -0.5];
        let residual_0 = [0.5, -1.0];
        let normed_0 = reference_rms_norm(&residual_0, &[1.0, 1.0]);
        let next_norm_weights = [1.0, 0.5];
        let layer_0 = run_identity_k4_layer(
            &mut scheduler,
            11,
            0,
            2,
            ScheduledPreviousCmd3Handoff::initial(2),
            ScheduledCmd1InputSource::CpuNormedHidden,
            FlashMoeCmd1InputState::cpu_normed(0, 2),
            &residual_0,
            &normed_0,
            &shared_output,
            Some(&next_norm_weights),
        );
        assert_eq!(
            layer_0.output_handoff,
            ScheduledCmd3OutputHandoff::DeferredToNextLayer
        );
        let mut token_state = FlashMoeTokenState::new(vec![0.0; 2], 0);
        for (mix_hash, weight) in &layer_0.cmd3.expert_mixes {
            token_state.mix_active_expert(*mix_hash, *weight);
        }
        token_state
            .apply_declared_expert_phase(
                layer_0.cmd3.submission,
                FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
            )
            .unwrap();
        let hidden_0 = token_state.hidden().to_vec();
        for (actual, expected) in hidden_0.iter().zip([1.011218, -1.1477928]) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "{actual} != {expected}; hidden={hidden_0:?}"
            );
        }
        let next_normed_0 = token_state
            .take_next_layer_normed_as_normed()
            .unwrap()
            .into_values();
        for (actual, expected) in next_normed_0.iter().zip([0.9348729, -0.5305683]) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
        let recurrent_after_linear = token_state.recurrent_value();
        assert_ne!(recurrent_after_linear, 0);

        let attention_1 = [0.6036627f32, 0.7642249];
        for (actual, expected) in attention_1.iter().zip([0.6036627, 0.7642249]) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
        let residual_1 = [hidden_0[0] + attention_1[0], hidden_0[1] + attention_1[1]];
        let normed_1 = reference_rms_norm(&residual_1, &[1.0, 1.0]);
        let layer_1 = run_identity_k4_layer(
            &mut scheduler,
            11,
            1,
            2,
            ScheduledPreviousCmd3Handoff::deferred_gpu(
                0,
                FlashMoeGpuBufferDescriptor::hidden(2),
                FlashMoeGpuBufferDescriptor::next_layer_normed(2),
            ),
            ScheduledCmd1InputSource::DeferredMetalNextNormed,
            FlashMoeCmd1InputState::gpu_next_layer_normed(
                1,
                FlashMoeGpuBufferDescriptor::next_layer_normed(2),
            ),
            &residual_1,
            &normed_1,
            &shared_output,
            None,
        );
        assert_eq!(
            layer_1.output_handoff,
            ScheduledCmd3OutputHandoff::CompleteHere
        );
        for (mix_hash, weight) in &layer_1.cmd3.expert_mixes {
            token_state.mix_active_expert(*mix_hash, *weight);
        }
        token_state
            .apply_declared_expert_phase(
                layer_1.cmd3.submission,
                FlashMoeExpertPhaseApplication::HiddenOnly,
            )
            .unwrap();
        assert_ne!(token_state.recurrent_value(), recurrent_after_linear);
        let hidden_1 = token_state.hidden().to_vec();
        for (actual, expected) in hidden_1.iter().zip([3.3762858, -0.8388142]) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "{actual} != {expected}; hidden={hidden_1:?}"
            );
        }
        assert!(token_state.take_next_layer_normed_as_normed().is_none());

        let logits = [
            hidden_1[0],
            hidden_1[1],
            -hidden_1[0] + 0.5 * hidden_1[1],
            0.25 * hidden_1[0] - 0.75 * hidden_1[1],
        ];
        for (actual, expected) in logits
            .iter()
            .zip([3.3762858, -0.8388142, -3.795693, 1.4731821])
        {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
        let candidates = top_k(&logits, 2);
        assert_eq!(candidates[0].0, 0);
        assert_eq!(candidates[1].0, 3);
        assert!((candidates[0].1 - 3.3762858).abs() <= 1e-5);
        assert!((candidates[1].1 - 1.4731821).abs() <= 1e-5);
        let metrics = scheduler.snapshot();
        assert_eq!(metrics.positioned_reads, 8);
        assert_eq!(metrics.bytes_read, 8 * 48);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn qwen35_resident_shared_cmd3_metal_output_matches_layer_reference() {
        #[derive(Debug, Clone, Copy)]
        enum ActiveLayout {
            Q4,
            Dense(DenseExpertDtype),
        }

        #[derive(Debug, Clone, Copy)]
        enum SharedLayout {
            Q4,
            Dense(&'static str),
        }

        let position = 7;
        let layer = 0;
        let width = 2;
        let experts = 9;
        let active_experts = 4;
        let residual = [1.5092846, 0.6604769];
        let normed = [1.2955897, 0.5669620];
        let router_scores = [0.1, 2.0, -1.0, 3.0, 0.5, 2.5, -0.2, 1.5, 4.0];
        let active = top_k(&router_scores, active_experts);
        for (active_layout, shared_layout) in [
            (ActiveLayout::Q4, SharedLayout::Q4),
            (ActiveLayout::Q4, SharedLayout::Dense("BF16")),
            (ActiveLayout::Q4, SharedLayout::Dense("F16")),
            (ActiveLayout::Q4, SharedLayout::Dense("F32")),
            (
                ActiveLayout::Dense(DenseExpertDtype::Bf16),
                SharedLayout::Q4,
            ),
            (ActiveLayout::Dense(DenseExpertDtype::F16), SharedLayout::Q4),
        ] {
            let attention = vec![ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal];
            let (temp, mut scheduler) = match active_layout {
                ActiveLayout::Q4 => test_execution_scheduler_with_attention(attention),
                ActiveLayout::Dense(dtype) => {
                    test_dense_execution_scheduler_with_attention(dtype, attention)
                }
            };
            match active_layout {
                ActiveLayout::Q4 => write_identity_fixed_q4_layer(temp.path(), layer, experts),
                ActiveLayout::Dense(dtype) => {
                    write_identity_fixed_dense_layer(temp.path(), layer, experts, dtype)
                }
            }

            let scheduled = scheduler
                .begin_layer(
                    position,
                    layer,
                    1,
                    active_experts,
                    ScheduledPreviousCmd3Handoff::initial(width),
                    true,
                )
                .unwrap();
            let (_, scheduled) = scheduled
                .resolve(
                    &scheduler,
                    ScheduledCmd1InputSource::CpuNormedHidden,
                    FlashMoeCmd1InputState::cpu_normed(layer, width),
                )
                .unwrap();
            let (cmd2, scheduled) = scheduled
                .resolve(
                    &scheduler,
                    ScheduledCmd2PhaseInputs::from_inputs(
                        ScheduledCmd2AttentionInput::metal_values(width),
                        ScheduledCmd2ResidualInput::metal_buffer(width),
                    ),
                )
                .unwrap();
            let prep_state =
                FlashMoePostAttentionPrepState::new(layer, width, experts, active_experts);
            let routing = scheduler
                .routing_from_post_attention_prep(&cmd2, prep_state, &active)
                .unwrap();
            let pending = scheduled
                .resolve(&routing)
                .unwrap()
                .issue_cmd3(&mut scheduler, &routing)
                .unwrap();

            let dense_path = temp.path().join("cmd3-parity-dense.bin");
            let mut dense_bytes = vec![0u8; 16 * 1024];
            let shared = match shared_layout {
                SharedLayout::Q4 => {
                    let one_bf16 = 0x3f80u16.to_le_bytes();
                    let mut write_identity_q4 = |packed: usize, scales: usize, rows: usize| {
                        dense_bytes[packed] = 0x01;
                        if rows > 1 {
                            dense_bytes[packed + 1] = 0x10;
                        }
                        for row in 0..rows {
                            dense_bytes[scales + row * 2..scales + row * 2 + 2]
                                .copy_from_slice(&one_bf16);
                        }
                    };
                    write_identity_q4(0, 16, 2);
                    write_identity_q4(64, 80, 2);
                    write_identity_q4(128, 144, 2);
                    dense_bytes[208..210].copy_from_slice(&one_bf16);
                    let projection = |tensor_name: &str,
                                      packed_byte_offset,
                                      scales_byte_offset,
                                      biases_byte_offset,
                                      rows| {
                        DenseQ4MmapMatvecProjection {
                            tensor_name: tensor_name.to_string(),
                            packed_byte_offset,
                            scales_byte_offset,
                            biases_byte_offset,
                            rows,
                            cols: width,
                            output_width: rows,
                            row_packed_bytes: 1,
                            groups_per_row: 1,
                            group_size: 2,
                            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                        }
                        .into()
                    };
                    SharedExpertPhaseResidentProjections {
                        gate: projection("shared.gate", 0, 16, 32, 2),
                        up: projection("shared.up", 64, 80, 96, 2),
                        down: projection("shared.down", 128, 144, 160, 2),
                        router: Some(projection("shared.router", 192, 208, 224, 1)),
                        shared_experts: 1,
                        intermediate: 2,
                        width,
                    }
                }
                SharedLayout::Dense(dtype) => {
                    let scalar_bytes = |value: f32| match dtype {
                        "BF16" => ((value.to_bits() >> 16) as u16).to_le_bytes().to_vec(),
                        "F16" => {
                            let bits = if value == 0.0 { 0u16 } else { 0x3c00u16 };
                            bits.to_le_bytes().to_vec()
                        }
                        "F32" => value.to_le_bytes().to_vec(),
                        _ => unreachable!(),
                    };
                    for offset in [0usize, 64, 128] {
                        let values = [1.0f32, 0.0, 0.0, 1.0];
                        let mut cursor = offset;
                        for value in values {
                            let bytes = scalar_bytes(value);
                            dense_bytes[cursor..cursor + bytes.len()].copy_from_slice(&bytes);
                            cursor += bytes.len();
                        }
                    }
                    let projection = |tensor_name: &str, byte_offset, rows| {
                        ResidentMmapMatvecProjection::Dense(DenseMmapMatvecProjection {
                            tensor_name: tensor_name.to_string(),
                            byte_offset,
                            dtype: dtype.to_string(),
                            rows,
                            cols: width,
                            output_width: rows,
                        })
                    };
                    SharedExpertPhaseResidentProjections {
                        gate: projection("shared.gate", 0, 2),
                        up: projection("shared.up", 64, 2),
                        down: projection("shared.down", 128, 2),
                        router: Some(projection("shared.router", 192, 1)),
                        shared_experts: 1,
                        intermediate: 2,
                        width,
                    }
                }
            };
            std::fs::write(&dense_path, dense_bytes).unwrap();
            let dense_file = std::fs::File::open(&dense_path).unwrap();
            let dense_mmap =
                Arc::new(unsafe { memmap2::MmapOptions::new().map(&dense_file).unwrap() });
            let metal = MetalExecutionContext::compile(
                Arc::clone(&dense_mmap),
                dense_mmap.len() as u64,
                &[None],
                1e-6,
            )
            .unwrap();
            let f32_bytes = |values: &[f32]| {
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            };
            let normed_buffer = unsafe {
                metal
                    .buffers()
                    .buffer_with_bytes(metal.runtime().device, &f32_bytes(&normed))
                    .unwrap()
            };
            let residual_buffer = unsafe {
                metal
                    .buffers()
                    .buffer_with_bytes(metal.runtime().device, &f32_bytes(&residual))
                    .unwrap()
            };
            let mut prep = MetalPostAttentionPrep::new(
                layer,
                width,
                experts,
                active.clone(),
                residual_buffer,
                normed_buffer,
            )
            .unwrap();
            prep.attach_routing_command(routing.clone()).unwrap();
            let dense_weights = metal.dense_weights().unwrap();
            let execution = pending
                .finish(
                    &mut scheduler,
                    ExpertPhaseInput::MetalPostAttention(prep),
                    ScheduledSharedExpertPhaseRef::Resident(&shared),
                    ScheduledNextNormWeights::none(),
                    |command| {
                        let output = command.resolve_output_state()?;
                        let ScheduledCmd3Command {
                            position,
                            layer,
                            experts,
                            weights,
                            input,
                            shared,
                            next_norm_weights,
                            payloads,
                            ..
                        } = command;
                        let ExpertPhaseInput::MetalPostAttention(input) = input;
                        MetalScheduledCmd3Builder::new(
                            metal.runtime(),
                            dense_weights,
                            Arc::clone(metal.buffers()),
                            metal.norm_epsilon(),
                        )
                        .submit(
                            position,
                            layer,
                            experts,
                            weights,
                            input,
                            output,
                            shared,
                            next_norm_weights.values(),
                            &payloads,
                        )
                    },
                )
                .unwrap();
            let output = execution.cmd3.submission.wait().unwrap();
            let (hidden, next_normed) = output.into_hidden_and_next_normed();
            assert!(next_normed.is_none());
            for (actual, expected) in hidden.iter().zip([3.3542297, 0.9476203]) {
                assert!(
                    (actual - expected).abs() <= 1e-4,
                    "{actual} != {expected} for active={active_layout:?} shared={shared_layout:?}"
                );
            }
        }
    }

    fn dummy_shared_dense_phase() -> SharedExpertPhaseWeights {
        SharedExpertPhaseWeights {
            gate: Arc::new(vec![1.0, 2.0]),
            up: Arc::new(vec![3.0, 4.0]),
            down: Arc::new(vec![5.0, 6.0]),
            router: Arc::new(vec![7.0]),
            shared_experts: 1,
            intermediate: 2,
            width: 1,
        }
    }

    fn dummy_q4_projection(
        name: &str,
        output_width: usize,
        cols: usize,
    ) -> DenseQ4MmapMatvecProjection {
        DenseQ4MmapMatvecProjection {
            tensor_name: name.to_string(),
            packed_byte_offset: 128,
            scales_byte_offset: 256,
            biases_byte_offset: 512,
            rows: output_width,
            cols,
            output_width,
            row_packed_bytes: cols.div_ceil(2),
            groups_per_row: cols.div_ceil(16),
            group_size: 16,
            scale_bias_dtype: "BF16".to_string(),
        }
    }

    fn dummy_router_projection(
        layer: usize,
        experts: usize,
        hidden_width: usize,
    ) -> RouterScoreProjectionDescriptor {
        let tensor_name = format!("model.layers.{layer}.mlp.gate.weight");
        RouterScoreProjectionDescriptor {
            layer,
            tensor_name: tensor_name.clone(),
            experts,
            hidden_width,
            binding: RouterScoreProjectionBinding::ResidentDense(DenseMmapMatvecProjection {
                tensor_name,
                byte_offset: 4096,
                dtype: "F32".to_string(),
                rows: experts,
                cols: hidden_width,
                output_width: experts,
            }),
        }
    }

    fn dummy_shared_q4_phase() -> SharedExpertPhaseResidentProjections {
        SharedExpertPhaseResidentProjections {
            gate: dummy_q4_projection("shared.gate", 16, 32).into(),
            up: dummy_q4_projection("shared.up", 16, 32).into(),
            down: dummy_q4_projection("shared.down", 32, 16).into(),
            router: Some(dummy_q4_projection("shared.router", 1, 32).into()),
            shared_experts: 1,
            intermediate: 16,
            width: 32,
        }
    }

    #[test]
    fn shared_expert_phase_ref_resolves_scheduler_source() {
        let dense = dummy_shared_dense_phase();
        let q4 = dummy_shared_q4_phase();

        let none = ScheduledSharedExpertPhaseRef::from_options(None, None);
        let none_descriptor = none.scheduled_shared_expert_descriptor().unwrap();
        assert_eq!(none_descriptor.source, ScheduledSharedExpertSource::None);
        assert_eq!(none_descriptor.shape, None);
        assert!(none.dense().is_none());
        assert!(none.resident().is_none());

        let dense_ref = ScheduledSharedExpertPhaseRef::from_options(Some(&dense), None);
        let dense_descriptor = dense_ref.scheduled_shared_expert_descriptor().unwrap();
        assert_eq!(
            dense_descriptor.source,
            ScheduledSharedExpertSource::DenseCpuWeights
        );
        assert!(dense_ref.dense().is_some());
        assert!(dense_ref.resident().is_none());
        assert_eq!(
            dense_descriptor.shape,
            Some(ScheduledSharedExpertShape::new(1, 1, 2).unwrap())
        );

        let q4_ref = ScheduledSharedExpertPhaseRef::from_options(Some(&dense), Some(&q4));
        let q4_descriptor = q4_ref.scheduled_shared_expert_descriptor().unwrap();
        assert_eq!(
            q4_descriptor.source,
            ScheduledSharedExpertSource::ResidentProjections
        );
        assert!(q4_ref.dense().is_none());
        assert!(q4_ref.resident().is_some());
        assert_eq!(
            q4_descriptor.shape,
            Some(ScheduledSharedExpertShape::new(32, 1, 16).unwrap())
        );
    }

    #[derive(Debug, Clone)]
    struct DummyCmd3Expert {
        layer: usize,
        expert: usize,
        descriptor: ExpertSlotDescriptor,
    }

    impl DummyCmd3Expert {
        fn whole_slot(layer: usize, expert: usize) -> Self {
            Self {
                layer,
                expert,
                descriptor: ExpertSlotDescriptor {
                    layer,
                    expert,
                    slot_offset: (expert as u64) * 128,
                    slot_capacity: 128,
                    payload_len: 128,
                },
            }
        }

        fn with_descriptor(mut self, descriptor: ExpertSlotDescriptor) -> Self {
            self.descriptor = descriptor;
            self
        }
    }

    impl ScheduledCmd3Expert for DummyCmd3Expert {
        fn scheduled_expert_layer(&self) -> usize {
            self.layer
        }

        fn scheduled_expert_id(&self) -> usize {
            self.expert
        }

        fn scheduled_expert_slot_descriptor(&self) -> ExpertSlotDescriptor {
            self.descriptor
        }
    }

    static DUMMY_Q4_SLOT: [u8; 512] = [0; 512];

    fn dummy_q4_payload(rows: usize, cols: usize) -> Q4MatvecPayload<'static> {
        let packed_bytes = rows * cols.div_ceil(2);
        let scale_bias_bytes = rows * cols.div_ceil(8) * 2;
        Q4MatvecPayload {
            rows,
            cols,
            group_size: 8,
            packed: &DUMMY_Q4_SLOT[..packed_bytes],
            scales: &[],
            biases: &[],
            scale_bias_groups: rows * cols.div_ceil(8),
            scale_bias_dtype: "BF16",
            scale_bytes: &DUMMY_Q4_SLOT[128..128 + scale_bias_bytes],
            bias_bytes: &DUMMY_Q4_SLOT[256..256 + scale_bias_bytes],
            source: Some(Q4MatvecSource {
                bytes: &DUMMY_Q4_SLOT,
                packed_offset: 0,
                scale_offset: 128,
                bias_offset: 256,
                reusable_bytes: None,
            }),
        }
    }

    impl ScheduledCmd3ExpertPayload for DummyCmd3Expert {
        fn scheduled_cmd3_expert_phase_payload(
            &self,
            width: usize,
        ) -> Result<ScheduledExpertPhaseMlpPayload<'_>> {
            Ok(ScheduledExpertPhaseMlpPayload::Q4(
                ScheduledQ4ExpertPhaseMlpPayload::new(
                    self.layer,
                    self.expert,
                    width,
                    dummy_q4_payload(4, width),
                    dummy_q4_payload(4, width),
                    dummy_q4_payload(width, 4),
                )?,
            ))
        }
    }

    fn dummy_scheduled_experts(
        layer: usize,
        experts: usize,
    ) -> ScheduledExpertSet<DummyCmd3Expert> {
        let routes = (0..experts)
            .map(|expert| ExpertRoute {
                expert,
                score: expert as f32,
            })
            .collect::<Vec<_>>();
        let scheduled_routes = ScheduledExpertRoutes::from_routes(layer, routes, 1.0).unwrap();
        ScheduledExpertSet::from_parts(
            scheduled_routes,
            (0..experts)
                .map(|expert| DummyCmd3Expert::whole_slot(layer, expert))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn scheduled_graph_preserves_the_declared_stage_order() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

        let stages: Vec<_> = graph.stages().iter().map(|stage| stage.stage).collect();

        assert_eq!(stages, FlashMoeGraphStage::ALL);
        assert!(graph.declares_scheduler_owned_expert_reads());
    }

    #[test]
    fn scheduled_graph_exposes_the_upstream_command_sequence() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd_sequence: Vec<_> = graph
            .cmd_sequence()
            .iter()
            .map(|stage| stage.stage)
            .collect();

        assert_eq!(
            cmd_sequence,
            vec![
                FlashMoeGraphStage::DeferredPreviousCmd3,
                FlashMoeGraphStage::Cmd1AttentionProjections,
                FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection,
            ]
        );
        assert_eq!(
            graph
                .stage(FlashMoeGraphStage::RoutingSoftmaxTopK)
                .placement,
            FlashMoeStagePlacement::CpuDeclared
        );
    }

    #[test]
    fn scheduled_graph_builds_explicit_cmd1_cmd2_and_cmd3_descriptors() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

        let cmd1 = graph
            .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::DeferredMetalNextNormed)
            .unwrap();
        let cmd2 = graph
            .build_cmd2_post_attention(
                14,
                4,
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
            )
            .unwrap();
        let routing = graph
            .build_routing_topk(
                14,
                512,
                4,
                ScheduledRoutingCandidateSource::MetalRouterScoresReadback,
            )
            .unwrap();
        let cmd3 = graph
            .build_cmd3_expert_phase(
                14,
                4,
                ScheduledCmd3InputSource::MetalPostAttentionPrep,
                ScheduledSharedExpertSource::ResidentProjections,
                ScheduledNextNormSource::CpuVisibleWeights,
            )
            .unwrap();

        assert_eq!(
            cmd1.stage.stage,
            FlashMoeGraphStage::Cmd1AttentionProjections
        );
        assert_eq!(cmd1.stage.placement, FlashMoeStagePlacement::Metal);
        assert_eq!(cmd1.layer, 14);
        assert_eq!(
            cmd1.input,
            ScheduledCmd1InputSource::DeferredMetalNextNormed
        );
        assert_eq!(
            cmd2.stage.stage,
            FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection
        );
        assert_eq!(cmd2.stage.placement, FlashMoeStagePlacement::Metal);
        assert_eq!(cmd2.layer, 14);
        assert_eq!(cmd2.active_experts, 4);
        assert_eq!(
            cmd2.attention,
            ScheduledCmd2AttentionSource::MetalAttentionValues
        );
        assert_eq!(cmd2.residual, ScheduledCmd2ResidualSource::MetalBuffer);
        assert_eq!(routing.stage.stage, FlashMoeGraphStage::RoutingSoftmaxTopK);
        assert_eq!(routing.stage.placement, FlashMoeStagePlacement::CpuDeclared);
        assert_eq!(routing.layer, 14);
        assert_eq!(routing.experts, 512);
        assert_eq!(routing.active_experts, 4);
        assert_eq!(
            routing.source,
            ScheduledRoutingCandidateSource::MetalRouterScoresReadback
        );
        assert_eq!(
            cmd3.stage.stage,
            FlashMoeGraphStage::Cmd3ExpertAndSharedCombine
        );
        assert_eq!(cmd3.stage.placement, FlashMoeStagePlacement::Metal);
        assert_eq!(cmd3.layer, 14);
        assert_eq!(cmd3.expert_count, 4);
        assert_eq!(cmd3.input, ScheduledCmd3InputSource::MetalPostAttentionPrep);
        assert_eq!(
            cmd3.shared,
            ScheduledSharedExpertSource::ResidentProjections
        );
        assert_eq!(cmd3.next_norm, ScheduledNextNormSource::CpuVisibleWeights);
    }

    #[test]
    fn scheduled_attention_math_resolves_declared_cpu_kv_state() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let attention = graph.build_attention_math(14, 9).unwrap();

        let output = attention
            .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(9, 14, 128, 128))
            .unwrap();

        assert_eq!(
            output.implementation(),
            ScheduledAttentionMathImplementation::CpuKvCache
        );
        assert_eq!(output.state().position(), 9);
        assert_eq!(output.state().layer(), 14);
        assert!(output.validate_execution_state(14, 9, 128).is_ok());
    }

    #[test]
    fn scheduled_glm_attention_resolves_distinct_compressed_mla_state() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let mut graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let attention = graph
            .stages
            .iter_mut()
            .find(|stage| stage.stage == FlashMoeGraphStage::AttentionMath)
            .unwrap();
        attention.implementation = FlashMoeStageImplementation::GlmMlaCpuWeightAbsorption;
        let attention = graph.build_attention_math(14, 9).unwrap();

        let output = attention
            .resolve_mla_kv_state(FlashMoeMlaKvState::cpu_visible(9, 14, 512, 64))
            .unwrap();

        assert_eq!(
            output.implementation(),
            ScheduledAttentionMathImplementation::CpuGlmMlaWeightAbsorption
        );
        assert_eq!(output.state().latent_len(), 512);
        assert_eq!(output.state().rotary_len(), 64);
        assert!(
            attention
                .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(9, 14, 512, 512))
                .unwrap_err()
                .to_string()
                .contains("requires compressed MLA KV state")
        );
    }

    #[test]
    fn scheduled_attention_math_rejects_stage_without_executor() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let mut graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let attention = graph
            .stages
            .iter_mut()
            .find(|stage| stage.stage == FlashMoeGraphStage::AttentionMath)
            .unwrap();
        attention.placement = FlashMoeStagePlacement::Metal;
        attention.implementation = FlashMoeStageImplementation::MetalResidentQ4AttentionProjections;

        let err = graph.build_attention_math(14, 9).unwrap_err();

        assert_eq!(err.stage, FlashMoeGraphStage::AttentionMath);
        assert!(
            err.to_string().contains("has no scheduled executor"),
            "{err}"
        );
    }

    #[test]
    fn scheduled_attention_math_rejects_mismatched_kv_state_without_fallback() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let attention = graph.build_attention_math(14, 9).unwrap();

        let placement_err = attention
            .resolve_kv_state(FlashMoeFullAttentionKvState::gpu_resident(9, 14, 128, 128))
            .unwrap_err();
        assert!(
            placement_err
                .to_string()
                .contains("requires CpuVisible KV state"),
            "{placement_err:#}"
        );

        let layer_err = attention
            .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(9, 15, 128, 128))
            .unwrap_err();
        assert!(
            layer_err
                .to_string()
                .contains("does not match KV state layer"),
            "{layer_err:#}"
        );

        let position_err = attention
            .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(8, 14, 128, 128))
            .unwrap_err();
        assert!(
            position_err
                .to_string()
                .contains("does not match KV state position"),
            "{position_err:#}"
        );

        let width_err = attention
            .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(9, 14, 128, 127))
            .unwrap_err();
        assert!(
            width_err
                .to_string()
                .contains("KV state is not declared graph state"),
            "{width_err:#}"
        );

        let output = attention
            .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(9, 14, 128, 128))
            .unwrap();
        let execution_err = output.validate_execution_state(14, 9, 256).unwrap_err();
        assert!(
            execution_err
                .to_string()
                .contains("does not match execution width"),
            "{execution_err:#}"
        );
    }

    #[test]
    fn scheduled_cmd1_submission_builds_resolved_command() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd1 = graph
            .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::DeferredMetalNextNormed)
            .unwrap();

        let command =
            ScheduledCmd1Submission::new(cmd1, ScheduledCmd1InputSource::DeferredMetalNextNormed)
                .unwrap()
                .into_cmd1_command();

        assert_eq!(command.layer, 14);
        assert_eq!(command.cmd1.layer, 14);
        assert_eq!(
            command.input.scheduled_cmd1_input_source(),
            ScheduledCmd1InputSource::DeferredMetalNextNormed
        );

        let resolved = command
            .into_resolved_command(FlashMoeCmd1InputState::gpu_next_layer_normed(
                14,
                FlashMoeGpuBufferDescriptor::next_layer_normed(4096),
            ))
            .unwrap();
        assert_eq!(resolved.layer, 14);
        assert_eq!(resolved.cmd1.layer, 14);
        assert_eq!(
            resolved.input.scheduled_cmd1_input_source(),
            ScheduledCmd1InputSource::DeferredMetalNextNormed
        );
        assert_eq!(resolved.input_state.layer(), 14);
        assert_eq!(resolved.input_state.len(), 4096);
        assert_eq!(
            resolved.input_state.placement(),
            FlashMoeStatePlacement::GpuResident
        );
    }

    #[test]
    fn scheduled_graph_builds_cmd1_submission_and_rejects_stale_stage() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd1 = graph
            .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::CpuNormedHidden)
            .unwrap();

        graph
            .build_cmd1_submission(cmd1, ScheduledCmd1InputSource::CpuNormedHidden)
            .unwrap();

        let mut stale_graph = graph.clone();
        stale_graph
            .stages
            .iter_mut()
            .find(|stage| stage.stage == FlashMoeGraphStage::Cmd1AttentionProjections)
            .unwrap()
            .implementation = FlashMoeStageImplementation::DeferredMetalCmd3;

        let err = stale_graph
            .build_cmd1_submission(cmd1, ScheduledCmd1InputSource::CpuNormedHidden)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("does not match scheduled graph CMD1 stage"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_cmd1_resolves_declared_input_state() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cpu_cmd1 = graph
            .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::CpuNormedHidden)
            .unwrap();
        let cpu_command =
            ScheduledCmd1Submission::new(cpu_cmd1, ScheduledCmd1InputSource::CpuNormedHidden)
                .unwrap()
                .into_cmd1_command();

        let cpu_resolved = cpu_command
            .into_resolved_command(FlashMoeCmd1InputState::cpu_normed(14, 4096))
            .unwrap();
        assert_eq!(cpu_resolved.layer, 14);
        assert_eq!(cpu_resolved.input_state.len(), 4096);
        assert_eq!(
            cpu_resolved.input.scheduled_cmd1_input_source(),
            ScheduledCmd1InputSource::CpuNormedHidden
        );

        let gpu_cmd1 = graph
            .build_cmd1_attention_projections(15, ScheduledCmd1InputSource::DeferredMetalNextNormed)
            .unwrap();
        let gpu_command = ScheduledCmd1Submission::new(
            gpu_cmd1,
            ScheduledCmd1InputSource::DeferredMetalNextNormed,
        )
        .unwrap()
        .into_cmd1_command();

        let gpu_resolved = gpu_command
            .into_resolved_command(FlashMoeCmd1InputState::gpu_next_layer_normed(
                15,
                FlashMoeGpuBufferDescriptor::next_layer_normed(4096),
            ))
            .unwrap();
        assert_eq!(gpu_resolved.layer, 15);
        assert_eq!(
            gpu_resolved.input_state.placement(),
            FlashMoeStatePlacement::GpuResident
        );
    }

    #[test]
    fn scheduled_cmd1_rejects_mismatched_input_state_without_fallback() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cpu_cmd1 = graph
            .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::CpuNormedHidden)
            .unwrap();
        let cpu_command = || {
            ScheduledCmd1Submission::new(cpu_cmd1, ScheduledCmd1InputSource::CpuNormedHidden)
                .unwrap()
                .into_cmd1_command()
        };

        let layer_err = cpu_command()
            .into_resolved_command(FlashMoeCmd1InputState::cpu_normed(15, 4096))
            .unwrap_err();
        assert!(
            layer_err
                .to_string()
                .contains("does not match input state layer"),
            "{layer_err:#}"
        );

        let source_err = cpu_command()
            .into_resolved_command(FlashMoeCmd1InputState::gpu_next_layer_normed(
                14,
                FlashMoeGpuBufferDescriptor::next_layer_normed(4096),
            ))
            .unwrap_err();
        assert!(
            source_err
                .to_string()
                .contains("CPU input requires CpuVisible Normed state"),
            "{source_err:#}"
        );

        let empty_err = cpu_command()
            .into_resolved_command(FlashMoeCmd1InputState::cpu_normed(14, 0))
            .unwrap_err();
        assert!(
            empty_err
                .to_string()
                .contains("CMD1 input is not declared graph state"),
            "{empty_err:#}"
        );
    }

    #[test]
    fn scheduled_cmd1_submission_rejects_mismatched_input_without_fallback() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd1 = graph
            .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::CpuNormedHidden)
            .unwrap();

        let err =
            ScheduledCmd1Submission::new(cmd1, ScheduledCmd1InputSource::DeferredMetalNextNormed)
                .unwrap_err();

        assert!(err.to_string().contains("does not match submitted input"));
    }

    #[test]
    fn scheduled_routing_selects_cpu_topk_from_declared_scores() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(3, 5, 3, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap();

        let logits = [0.0, 2.0, 2.0, -1.0, 1.0];
        let selected = routing
            .select_from_scores(&ScheduledRoutingScoreView::new(
                3,
                ScheduledRoutingCandidateSource::CpuRouterScores,
                &logits,
            ))
            .unwrap();

        let mut probabilities = logits.to_vec();
        softmax_in_place(&mut probabilities);
        assert_eq!(selected, routing_top_k(&probabilities, 3));

        let layer_err = routing
            .select_from_scores(&ScheduledRoutingScoreView::new(
                4,
                ScheduledRoutingCandidateSource::CpuRouterScores,
                &[0.0, 2.0, 2.0, -1.0, 1.0],
            ))
            .unwrap_err();
        assert!(
            layer_err
                .to_string()
                .contains("does not match submitted score layer"),
            "{layer_err:#}"
        );

        let source_err = routing
            .select_from_scores(&ScheduledRoutingScoreView::new(
                3,
                ScheduledRoutingCandidateSource::MetalRouterScoresReadback,
                &[0.0, 2.0, 2.0, -1.0, 1.0],
            ))
            .unwrap_err();
        assert!(
            source_err
                .to_string()
                .contains("does not match submitted score source"),
            "{source_err:#}"
        );

        let projection = dummy_router_projection(3, 5, 4096);
        let projected_selected = routing
            .select_from_scores(&ScheduledRoutingScoreView::from_router_projection(
                ScheduledRoutingCandidateSource::CpuRouterScores,
                &projection,
                &[0.0, 2.0, 2.0, -1.0, 1.0],
            ))
            .unwrap();
        assert_eq!(projected_selected, selected);

        let wrong_experts = dummy_router_projection(3, 4, 4096);
        let projection_err = routing
            .select_from_scores(&ScheduledRoutingScoreView::from_router_projection(
                ScheduledRoutingCandidateSource::CpuRouterScores,
                &wrong_experts,
                &[0.0, 2.0, 2.0, -1.0, 1.0],
            ))
            .unwrap_err();
        assert!(
            projection_err
                .to_string()
                .contains("does not match submitted router projection experts"),
            "{projection_err:#}"
        );
    }

    #[test]
    fn scheduled_router_score_projection_command_declares_score_state() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(3, 5, 2, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap();
        let projection = dummy_router_projection(3, 5, 8);

        let command = routing
            .build_score_projection_command(Some(projection.clone()), 8)
            .unwrap();

        assert_eq!(command.routing, routing);
        assert_eq!(
            command.state,
            FlashMoeRoutingOutputState::cpu_router_scores(3, 5, 2)
        );
        assert_eq!(command.projection, Some(projection));
        assert_eq!(command.hidden_width, 8);
    }

    #[test]
    fn scheduled_router_score_projection_command_finalizes_declared_batch() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(3, 5, 2, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap();
        let projection = dummy_router_projection(3, 5, 8);

        let batch = routing
            .build_score_projection_command(Some(projection.clone()), 8)
            .unwrap()
            .into_score_batch(vec![0.0, 1.0, 2.0, 3.0, 4.0])
            .unwrap();
        assert_eq!(
            batch.state(),
            FlashMoeRoutingOutputState::cpu_router_scores(3, 5, 2)
        );
        assert_eq!(batch.projection, Some(projection.clone()));
        assert_eq!(batch.scores, vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let err = routing
            .build_score_projection_command(Some(projection), 8)
            .unwrap()
            .into_score_batch(vec![0.0, 1.0])
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("has 2 scores for 5 declared experts"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_router_score_projection_command_selects_routing_command() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(3, 5, 2, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap();
        let projection = dummy_router_projection(3, 5, 8);

        let command = routing
            .build_score_projection_command(Some(projection.clone()), 8)
            .unwrap();
        let execution = command.projection_execution().unwrap();
        assert_eq!(execution.layer, 3);
        assert_eq!(execution.experts, 5);
        assert_eq!(execution.hidden_width, 8);
        assert_eq!(execution.tensor_name, projection.tensor_name);

        let routed = command
            .into_routing_command(vec![0.5, 9.0, -1.0, 4.0, 3.0])
            .unwrap();
        assert_eq!(routed.layer, 3);
        assert_eq!(routed.active_experts, 2);
        assert_eq!(
            routed.source,
            ScheduledRoutingCandidateSource::CpuRouterScores
        );
        let mut probabilities = vec![0.5, 9.0, -1.0, 4.0, 3.0];
        softmax_in_place(&mut probabilities);
        assert_eq!(routed.routes, routing_top_k(&probabilities, 2));

        let err = routing
            .build_score_projection_command(Some(projection), 8)
            .unwrap()
            .into_routing_command(vec![0.5, f32::NAN, -1.0, 4.0, 3.0])
            .unwrap_err();
        assert!(
            err.to_string().contains("score for expert 1 is not finite"),
            "{err:#}"
        );

        let missing_projection_err = routing
            .build_score_projection_command(None, 8)
            .unwrap()
            .projection_execution()
            .unwrap_err();
        assert!(
            missing_projection_err
                .to_string()
                .contains("has no declared resident projection implementation"),
            "{missing_projection_err:#}"
        );
    }

    #[test]
    fn scheduled_graph_builds_router_score_projection_command() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let projection = dummy_router_projection(3, 5, 8);

        let command = graph
            .build_router_score_projection(3, 5, 2, Some(projection.clone()), 8)
            .unwrap();

        assert_eq!(command.routing.layer, 3);
        assert_eq!(command.routing.experts, 5);
        assert_eq!(command.routing.active_experts, 2);
        assert_eq!(
            command.routing.source,
            ScheduledRoutingCandidateSource::CpuRouterScores
        );
        assert_eq!(
            command.state,
            FlashMoeRoutingOutputState::cpu_router_scores(3, 5, 2)
        );
        assert_eq!(command.projection, Some(projection));

        let err = graph
            .build_router_score_projection(3, 5, 2, Some(dummy_router_projection(3, 5, 4)), 8)
            .unwrap_err();
        assert_eq!(err.stage, FlashMoeGraphStage::RoutingSoftmaxTopK);
        assert!(
            err.to_string()
                .contains("invalid scheduled router score projection"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_router_score_projection_command_rejects_mismatched_projection() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(3, 5, 2, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap();

        let width_err = routing
            .build_score_projection_command(Some(dummy_router_projection(3, 5, 4)), 8)
            .unwrap_err();
        assert!(
            width_err
                .to_string()
                .contains("hidden width 4 does not match submitted hidden width 8"),
            "{width_err:#}"
        );

        let preselected = graph
            .build_routing_topk(
                3,
                5,
                2,
                ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
            )
            .unwrap();
        let source_err = preselected
            .build_score_projection_command(None, 8)
            .unwrap_err();
        assert!(
            source_err
                .to_string()
                .contains("requires CPU router-score routing"),
            "{source_err:#}"
        );
    }

    #[test]
    fn scheduled_routing_builds_command_from_declared_scores() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(3, 5, 3, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap();
        let output = routing
            .validate_output_state(FlashMoeRoutingOutputState::cpu_router_scores(3, 5, 3))
            .unwrap();

        let command = routing
            .select_command_from_output_scores(
                output,
                &ScheduledRoutingScoreView::new(
                    3,
                    ScheduledRoutingCandidateSource::CpuRouterScores,
                    &[0.1, 0.9, 0.2, 1.5, -0.2],
                ),
            )
            .unwrap();

        assert_eq!(command.layer, 3);
        assert_eq!(command.active_experts, 3);
        assert_eq!(
            command.source,
            ScheduledRoutingCandidateSource::CpuRouterScores
        );
        assert_eq!(command.routing.layer, 3);
        assert_eq!(command.routes.len(), 3);
        assert_eq!(command.routes[0].0, 3);
    }

    #[test]
    fn scheduled_routing_command_validates_active_expert_issue_shape() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(3, 5, 3, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap();
        let output = routing
            .validate_output_state(FlashMoeRoutingOutputState::cpu_router_scores(3, 5, 3))
            .unwrap();
        let command = routing
            .select_command_from_output_scores(
                output,
                &ScheduledRoutingScoreView::new(
                    3,
                    ScheduledRoutingCandidateSource::CpuRouterScores,
                    &[0.1, 0.9, 0.2, 1.5, -0.2],
                ),
            )
            .unwrap();

        command.validate_for_active_expert_issue().unwrap();

        let mut wrong_count = command.clone();
        wrong_count.active_experts = 2;
        let count_err = wrong_count.validate_for_active_expert_issue().unwrap_err();
        assert!(
            count_err
                .to_string()
                .contains("does not match routing descriptor active expert count"),
            "{count_err:#}"
        );

        let mut repeated = command;
        repeated.routes[1].0 = repeated.routes[0].0;
        let repeated_err = repeated.validate_for_active_expert_issue().unwrap_err();
        assert!(
            repeated_err
                .to_string()
                .contains("selected expert 3 more than once"),
            "{repeated_err:#}"
        );
    }

    #[test]
    fn scheduled_routing_validates_preselected_fused_prep_candidates() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(
                3,
                8,
                4,
                ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
            )
            .unwrap();

        let selected = routing
            .validate_preselected(&[(7, 3.0), (1, 2.0), (3, 1.0), (5, 0.0)])
            .unwrap();
        assert_eq!(selected, vec![(7, 3.0), (1, 2.0), (3, 1.0), (5, 0.0)]);

        let duplicate_err = routing
            .validate_preselected(&[(7, 3.0), (1, 2.0), (7, 1.0), (5, 0.0)])
            .unwrap_err();
        assert!(
            duplicate_err
                .to_string()
                .contains("selected expert 7 more than once"),
            "{duplicate_err:#}"
        );

        let source_err = routing
            .select_from_scores(&ScheduledRoutingScoreView::new(
                3,
                ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            ))
            .unwrap_err();
        assert!(
            source_err
                .to_string()
                .contains("must submit preselected CPU topK candidates"),
            "{source_err:#}"
        );
    }

    #[test]
    fn scheduled_routing_validates_declared_cmd2_output_state() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(
                3,
                8,
                4,
                ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
            )
            .unwrap();

        let output = routing
            .validate_output_state(
                FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(3, 8, 4),
            )
            .unwrap();
        assert_eq!(output.routing, routing);
        assert_eq!(
            output.state().source(),
            FlashMoeRoutingOutputSource::FusedMetalPostAttentionPrepCpuTopK
        );

        let layer_err = routing
            .validate_output_state(
                FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(4, 8, 4),
            )
            .unwrap_err();
        assert!(
            layer_err
                .to_string()
                .contains("does not match submitted routing output layer"),
            "{layer_err:#}"
        );

        let source_err = routing
            .validate_output_state(FlashMoeRoutingOutputState::cpu_router_scores(3, 8, 4))
            .unwrap_err();
        assert!(
            source_err
                .to_string()
                .contains("does not match submitted routing output source"),
            "{source_err:#}"
        );
    }

    #[test]
    fn scheduled_routing_builds_command_from_preselected_candidates() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(
                4,
                8,
                2,
                ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
            )
            .unwrap();
        let output = routing
            .validate_output_state(
                FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(4, 8, 2),
            )
            .unwrap();

        let command = routing
            .command_from_preselected_output(output, &[(7, 0.75), (1, 0.25)])
            .unwrap();

        assert_eq!(command.layer, 4);
        assert_eq!(command.active_experts, 2);
        assert_eq!(
            command.source,
            ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK
        );
        assert_eq!(command.routes, vec![(7, 0.75), (1, 0.25)]);
    }

    #[test]
    fn scheduled_routing_rejects_wrong_stage_placement_or_bounds() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let mut graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .stages
            .iter_mut()
            .find(|stage| stage.stage == FlashMoeGraphStage::RoutingSoftmaxTopK)
            .unwrap();
        routing.placement = FlashMoeStagePlacement::Metal;

        let err = graph
            .build_routing_topk(0, 8, 4, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap_err();
        assert_eq!(err.family, graph.family());
        assert_eq!(err.stage, FlashMoeGraphStage::RoutingSoftmaxTopK);
        assert!(
            err.to_string()
                .contains("routing softmax/topK stage must be implemented"),
            "{err:#}"
        );

        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(0, 2, 4, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap();
        let bounds_err = routing
            .select_from_scores(&ScheduledRoutingScoreView::new(
                0,
                ScheduledRoutingCandidateSource::CpuRouterScores,
                &[1.0, 2.0],
            ))
            .unwrap_err();
        assert!(
            bounds_err
                .to_string()
                .contains("active expert count 4 exceeds expert count 2"),
            "{bounds_err:#}"
        );
    }

    #[test]
    fn scheduled_graph_rejects_non_metal_cmd1_builder() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let mut graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd1 = graph
            .stages
            .iter_mut()
            .find(|stage| stage.stage == FlashMoeGraphStage::Cmd1AttentionProjections)
            .unwrap();
        cmd1.placement = FlashMoeStagePlacement::CpuDeclared;

        let err = graph
            .build_cmd1_attention_projections(0, ScheduledCmd1InputSource::CpuNormedHidden)
            .unwrap_err();

        assert_eq!(err.family, graph.family());
        assert_eq!(err.stage, FlashMoeGraphStage::Cmd1AttentionProjections);
        assert!(
            err.to_string()
                .contains("CMD1 attention projection stage must be implemented"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_cmd2_submission_validates_attention_and_residual_sources() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd2 = graph
            .build_cmd2_post_attention(
                11,
                4,
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
            )
            .unwrap();

        let submission = ScheduledCmd2Submission::new(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
                4096,
                4096,
            ),
        )
        .unwrap();

        assert_eq!(submission.cmd2.layer, 11);
        assert_eq!(submission.cmd2.active_experts, 4);
    }

    #[test]
    fn scheduled_cmd2_submission_builds_resolved_command() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd2 = graph
            .build_cmd2_post_attention(
                11,
                4,
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
            )
            .unwrap();

        let command = ScheduledCmd2Submission::new(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
                4096,
                4096,
            ),
        )
        .unwrap()
        .into_cmd2_command();

        assert_eq!(command.layer, 11);
        assert_eq!(command.active_experts, 4);
        assert_eq!(
            command.inputs.scheduled_cmd2_attention_source(),
            ScheduledCmd2AttentionSource::MetalAttentionValues
        );
        assert_eq!(
            command.inputs.scheduled_cmd2_residual_source(),
            ScheduledCmd2ResidualSource::MetalBuffer
        );
        let input_state = command.input_state();
        assert_eq!(input_state.attention().len(), 4096);
        assert_eq!(
            input_state.attention().placement(),
            FlashMoeStatePlacement::GpuResident
        );
        assert_eq!(input_state.residual().len(), 4096);
        assert!(input_state.is_declared_graph_state());
    }

    #[test]
    fn scheduled_cmd2_input_descriptors_build_declared_command() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

        let command = graph
            .build_cmd2_command(
                11,
                4,
                ScheduledCmd2PhaseInputs::from_inputs(
                    ScheduledCmd2AttentionInput::metal_values(4096),
                    ScheduledCmd2ResidualInput::cpu_hidden(4096),
                ),
            )
            .unwrap();

        assert_eq!(
            command.cmd2.attention,
            ScheduledCmd2AttentionSource::MetalAttentionValues
        );
        assert_eq!(
            command.cmd2.residual,
            ScheduledCmd2ResidualSource::CpuHidden
        );
        assert_eq!(
            command.input_state().attention().placement(),
            FlashMoeStatePlacement::GpuResident
        );
        assert_eq!(
            command.input_state().residual().placement(),
            FlashMoeStatePlacement::CpuVisible
        );
        assert_eq!(command.input_state().attention().len(), 4096);
        assert_eq!(command.input_state().residual().len(), 4096);
    }

    #[test]
    fn scheduled_cmd2_input_descriptors_reject_empty_graph_state_without_fallback() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

        let err = graph
            .build_cmd2_command(
                11,
                4,
                ScheduledCmd2PhaseInputs::from_inputs(
                    ScheduledCmd2AttentionInput::cpu_values(0),
                    ScheduledCmd2ResidualInput::cpu_hidden(4096),
                ),
            )
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("scheduled CMD2 input is not declared graph state"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_graph_builds_cmd2_submission_and_rejects_stale_stage() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd2 = graph
            .build_cmd2_post_attention(
                11,
                4,
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
            )
            .unwrap();

        graph
            .build_cmd2_submission(
                cmd2,
                ScheduledCmd2PhaseInputs::new(
                    ScheduledCmd2AttentionSource::MetalAttentionValues,
                    ScheduledCmd2ResidualSource::MetalBuffer,
                    4096,
                    4096,
                ),
            )
            .unwrap();

        let mut stale_graph = graph.clone();
        stale_graph
            .stages
            .iter_mut()
            .find(|stage| stage.stage == FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection)
            .unwrap()
            .implementation = FlashMoeStageImplementation::QwenTextInput;

        let err = stale_graph
            .build_cmd2_submission(
                cmd2,
                ScheduledCmd2PhaseInputs::new(
                    ScheduledCmd2AttentionSource::MetalAttentionValues,
                    ScheduledCmd2ResidualSource::MetalBuffer,
                    4096,
                    4096,
                ),
            )
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("does not match scheduled graph CMD2 stage"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_cmd2_resolves_declared_post_attention_prep_output() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd2 = graph
            .build_cmd2_post_attention(
                11,
                4,
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
            )
            .unwrap();
        let command = ScheduledCmd2Submission::new(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
                4096,
                4096,
            ),
        )
        .unwrap()
        .into_cmd2_command();

        let output = command
            .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(11, 4096, 512, 4))
            .unwrap();

        assert_eq!(output.layer, 11);
        assert_eq!(output.active_experts, 4);
        assert_eq!(output.width(), 4096);
        assert_eq!(output.input_state, command.input_state());
        assert_eq!(output.routing().layer(), 11);
        assert_eq!(output.routing().experts(), 512);
    }

    #[test]
    fn scheduled_cmd2_output_builds_preselected_routing_command() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd2 = graph
            .build_cmd2_post_attention(
                11,
                2,
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
            )
            .unwrap();
        let command = ScheduledCmd2Submission::new(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
                4096,
                4096,
            ),
        )
        .unwrap()
        .into_cmd2_command();
        let output = command
            .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(11, 4096, 512, 2))
            .unwrap();

        let routing = output
            .command_from_preselected_routes(&graph, &[(7, 0.75), (3, 0.25)])
            .unwrap();

        assert_eq!(routing.layer, 11);
        assert_eq!(routing.active_experts, 2);
        assert_eq!(
            routing.source,
            ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK
        );
        assert_eq!(routing.routes, vec![(7, 0.75), (3, 0.25)]);
    }

    #[test]
    fn scheduled_cmd2_command_builds_preselected_routing_command() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd2 = graph
            .build_cmd2_post_attention(
                11,
                2,
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
            )
            .unwrap();
        let command = ScheduledCmd2Submission::new(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
                4096,
                4096,
            ),
        )
        .unwrap()
        .into_cmd2_command();

        let routing = command
            .command_from_post_attention_prep_routes(
                &graph,
                FlashMoePostAttentionPrepState::new(11, 4096, 512, 2),
                &[(7, 0.75), (3, 0.25)],
            )
            .unwrap();

        assert_eq!(routing.layer, 11);
        assert_eq!(routing.active_experts, 2);
        assert_eq!(
            routing.source,
            ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK
        );
        assert_eq!(routing.routes, vec![(7, 0.75), (3, 0.25)]);

        let err = command
            .command_from_post_attention_prep_routes(
                &graph,
                FlashMoePostAttentionPrepState::new(11, 4096, 512, 2),
                &[(7, 0.75)],
            )
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("received 1 preselected experts; expected 2"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_cmd2_rejects_missing_post_attention_prep_without_cpu_fallback() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let command = graph
            .build_cmd2_command(
                11,
                4,
                ScheduledCmd2PhaseInputs::from_inputs(
                    ScheduledCmd2AttentionInput::metal_values(4096),
                    ScheduledCmd2ResidualInput::metal_buffer(4096),
                ),
            )
            .unwrap();

        let err = command
            .reject_missing_post_attention_prep("test missing prep")
            .unwrap_err();

        assert!(
            err.to_string().contains(
                "FlashMoe unsupported scheduled CMD2 path: layer 11 declares CMD2 post-attention and routing projection implementation"
            ),
            "{err:#}"
        );
        assert!(err.to_string().contains("test missing prep"), "{err:#}");
    }

    #[test]
    fn scheduled_cmd2_output_rejects_mismatched_preselected_routes() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd2 = graph
            .build_cmd2_post_attention(
                11,
                2,
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
            )
            .unwrap();
        let command = ScheduledCmd2Submission::new(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
                4096,
                4096,
            ),
        )
        .unwrap()
        .into_cmd2_command();
        let output = command
            .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(11, 4096, 512, 2))
            .unwrap();

        let err = output
            .command_from_preselected_routes(&graph, &[(7, 0.75)])
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("received 1 preselected experts; expected 2"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_cmd2_rejects_mismatched_post_attention_prep_output() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd2 = graph
            .build_cmd2_post_attention(
                11,
                4,
                ScheduledCmd2AttentionSource::CpuAttentionValues,
                ScheduledCmd2ResidualSource::CpuHidden,
            )
            .unwrap();
        let command = ScheduledCmd2Submission::new(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::CpuAttentionValues,
                ScheduledCmd2ResidualSource::CpuHidden,
                4096,
                4096,
            ),
        )
        .unwrap()
        .into_cmd2_command();

        let layer_err = command
            .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(12, 4096, 512, 4))
            .unwrap_err();
        assert!(
            layer_err
                .to_string()
                .contains("does not match post-attention prep layer"),
            "{layer_err:#}"
        );

        let active_err = command
            .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(11, 4096, 512, 3))
            .unwrap_err();
        assert!(
            active_err
                .to_string()
                .contains("does not match post-attention prep active expert count"),
            "{active_err:#}"
        );

        let width_err = command
            .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(11, 2048, 512, 4))
            .unwrap_err();
        assert!(
            width_err
                .to_string()
                .contains("does not match residual input width"),
            "{width_err:#}"
        );
    }

    #[test]
    fn scheduled_cmd2_submission_rejects_mismatched_sources_without_fallback() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd2 = graph
            .build_cmd2_post_attention(
                11,
                4,
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
            )
            .unwrap();

        let attention_err = ScheduledCmd2Submission::new(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::CpuAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
                4096,
                4096,
            ),
        )
        .unwrap_err();
        assert!(
            attention_err
                .to_string()
                .contains("does not match submitted source")
        );

        let residual_err = ScheduledCmd2Submission::new(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::CpuHidden,
                4096,
                4096,
            ),
        )
        .unwrap_err();
        assert!(
            residual_err
                .to_string()
                .contains("does not match submitted source")
        );

        let invalid_state_err = ScheduledCmd2Submission::new(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
                0,
                4096,
            ),
        )
        .unwrap_err();
        assert!(
            invalid_state_err
                .to_string()
                .contains("input is not declared graph state")
        );
    }

    #[test]
    fn scheduled_cmd3_cpu_input_declares_whole_phase_or_errors() {
        let normed = [1.0f32, 2.0, 3.0];
        let residual = [4.0f32, 5.0, 6.0];
        let input = ScheduledCmd3CpuInput::new(9, &normed, &residual).unwrap();
        assert_eq!(input.width(), 3);
        assert_eq!(
            input.scheduled_cmd3_input_source(),
            ScheduledCmd3InputSource::CpuNormedResidualUpload
        );
        assert_eq!(
            input.scheduled_cmd3_input_state(9),
            FlashMoeCmd3InputState::cpu_normed_residual(9, 3, 3)
        );

        let mismatched = ScheduledCmd3CpuInput::new(9, &normed, &residual[..2]).unwrap_err();
        assert!(
            mismatched
                .to_string()
                .contains("is not a declared graph state"),
            "{mismatched:#}"
        );

        let empty = ScheduledCmd3CpuInput::new(9, &[], &[]).unwrap_err();
        assert!(
            empty.to_string().contains("is not a declared graph state"),
            "{empty:#}"
        );
    }

    #[test]
    fn scheduled_cmd3_metal_post_attention_input_declares_prep_state_or_errors() {
        let state = FlashMoePostAttentionPrepState::new(4, 8, 16, 2);
        let input = ScheduledCmd3MetalPostAttentionInput::new(state, 2).unwrap();
        assert_eq!(input.width(), 8);
        assert_eq!(input.state(), state);
        assert_eq!(
            input.scheduled_cmd3_input_source(),
            ScheduledCmd3InputSource::MetalPostAttentionPrep
        );
        assert_eq!(
            input.scheduled_cmd3_input_state(4),
            FlashMoeCmd3InputState::metal_post_attention_prep(4, state)
        );

        let route_err = ScheduledCmd3MetalPostAttentionInput::new(state, 1).unwrap_err();
        assert!(
            route_err
                .to_string()
                .contains("state declares 2 active experts but prep carries 1 routes"),
            "{route_err:#}"
        );

        let empty = FlashMoePostAttentionPrepState::new(4, 0, 16, 2);
        let empty_err = ScheduledCmd3MetalPostAttentionInput::new(empty, 2).unwrap_err();
        assert!(
            empty_err
                .to_string()
                .contains("prep state is not a declared graph state"),
            "{empty_err:#}"
        );
    }

    #[test]
    fn scheduled_cmd3_submission_validates_batch_and_sources() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                2,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::ResidentProjections,
                ScheduledNextNormSource::None,
            )
            .unwrap();

        let submission = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::none(),
        )
        .unwrap();

        assert_eq!(submission.position, 19);
        assert_eq!(submission.cmd3.layer, 7);
        assert_eq!(submission.scheduled.len(), 2);
        assert_eq!(submission.input_state.width(), 8);
        assert_eq!(
            submission.input_state.placement(),
            FlashMoeStatePlacement::CpuVisible
        );
    }

    #[test]
    fn scheduled_expert_batch_resolves_cmd3_payloads_from_scheduled_experts() {
        let scheduled = dummy_scheduled_experts(7, 2);

        let payloads = scheduled.cmd3_expert_phase_payloads(8).unwrap();

        assert_eq!(payloads.len(), 2);
        let payload = payloads[0].q4();
        assert_eq!(payload.gate.rows, 4);
        assert_eq!(payload.gate.cols, 8);
        assert_eq!(payload.up.rows, 4);
        assert_eq!(payload.down.rows, 8);
        assert_eq!(payload.down.cols, 4);
    }

    #[test]
    fn scheduled_cmd3_submission_builds_resolved_command_payloads() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let next_norm = [1.0; 8];
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                2,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::ResidentProjections,
                ScheduledNextNormSource::CpuVisibleWeights,
            )
            .unwrap();
        let submission = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &next_norm,
                8,
            )
            .unwrap(),
        )
        .unwrap();

        let command = submission.into_cmd3_command().unwrap();

        assert_eq!(command.position, 19);
        assert_eq!(command.layer, 7);
        assert_eq!(command.cmd3.layer, 7);
        assert_eq!(command.experts.len(), 2);
        assert_eq!(command.weights.len(), 2);
        assert_eq!(command.payloads.len(), 2);
        assert_eq!(
            command.input.scheduled_cmd3_input_source(),
            ScheduledCmd3InputSource::CpuNormedResidualUpload
        );
        assert_eq!(command.input_state.layer(), 7);
        assert_eq!(command.input_state.width(), 8);
        assert_eq!(
            command.input_state.placement(),
            FlashMoeStatePlacement::CpuVisible
        );
        assert_eq!(command.next_norm_weights.values().unwrap().len(), 8);
        assert_eq!(command.payloads[0].q4().gate.cols, 8);

        let output = command.resolve_output_state().unwrap();
        assert_eq!(output.cmd3, command.cmd3);
        assert_eq!(output.layer, 7);
        assert_eq!(output.input_state, command.input_state);
        let output_state = output.state();
        assert_eq!(output_state.width(), 8);
        assert!(output_state.has_next_normed());
        assert_eq!(output_state.hidden().len(), 8);
        assert_eq!(output_state.next_normed().unwrap().len(), 8);
    }

    #[test]
    fn scheduled_graph_builds_cmd3_command_from_typed_descriptors() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let next_norm = [1.0; 8];

        let command = graph
            .build_cmd3_command_from_descriptors(
                19,
                &scheduled,
                dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
                dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
                ScheduledNextNormWeights::cpu_visible(
                    "model.layers.8.input_layernorm.weight",
                    &next_norm,
                    8,
                )
                .unwrap(),
            )
            .unwrap();

        assert_eq!(command.position, 19);
        assert_eq!(command.layer, 7);
        assert_eq!(command.cmd3.layer, 7);
        assert_eq!(command.cmd3.expert_count, 2);
        assert_eq!(
            command.cmd3.input,
            ScheduledCmd3InputSource::CpuNormedResidualUpload
        );
        assert_eq!(
            command.cmd3.shared,
            ScheduledSharedExpertSource::ResidentProjections
        );
        assert_eq!(
            command.cmd3.next_norm,
            ScheduledNextNormSource::CpuVisibleWeights
        );
        assert_eq!(command.input_state.width(), 8);
        assert_eq!(command.payloads.len(), 2);
    }

    #[test]
    fn scheduled_graph_cmd3_command_rejects_mismatched_typed_descriptor() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);

        let err = graph
            .build_cmd3_command_from_descriptors(
                19,
                &scheduled,
                dummy_cmd3_input_with_width(ScheduledCmd3InputSource::CpuNormedResidualUpload, 4),
                dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
                ScheduledNextNormWeights::none(),
            )
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("shared expert width 8 does not match input width 4"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_cmd3_output_state_tracks_absent_next_norm() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                2,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::ResidentProjections,
                ScheduledNextNormSource::None,
            )
            .unwrap();
        let command = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::none(),
        )
        .unwrap()
        .into_cmd3_command()
        .unwrap();

        let output = command.resolve_output_state().unwrap();
        assert_eq!(output.cmd3, command.cmd3);
        assert_eq!(output.layer, 7);
        assert_eq!(output.input_state, command.input_state);
        let output_state = output.state();

        assert_eq!(output_state.width(), 8);
        assert!(!output_state.has_next_normed());
        assert!(output_state.next_normed().is_none());
    }

    #[test]
    fn scheduled_cmd3_output_accepts_declared_phase_output() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                2,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::ResidentProjections,
                ScheduledNextNormSource::CpuVisibleWeights,
            )
            .unwrap();
        let command = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0; 8],
                8,
            )
            .unwrap(),
        )
        .unwrap()
        .into_cmd3_command()
        .unwrap();
        let output_state = command.resolve_output_state().unwrap();

        let output = FlashMoeExpertPhaseOutput::new(vec![0.0; 8], Some(vec![1.0; 8]));

        let output = output_state.validate_expert_phase_output(output).unwrap();
        assert_eq!(output.declared_cmd3_output(), Some(output_state.state()));
    }

    #[test]
    fn scheduled_cmd3_output_rejects_mismatched_phase_output_without_fallback() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                2,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::ResidentProjections,
                ScheduledNextNormSource::CpuVisibleWeights,
            )
            .unwrap();
        let command = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0; 8],
                8,
            )
            .unwrap(),
        )
        .unwrap()
        .into_cmd3_command()
        .unwrap();
        let output_state = command.resolve_output_state().unwrap();

        let hidden_err = output_state
            .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(
                vec![0.0; 7],
                Some(vec![1.0; 8]),
            ))
            .unwrap_err();
        assert!(
            hidden_err
                .to_string()
                .contains("hidden length 7 does not match declared hidden length 8"),
            "{hidden_err:#}"
        );

        let missing_next_norm_err = output_state
            .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(vec![0.0; 8], None))
            .unwrap_err();
        assert!(
            missing_next_norm_err
                .to_string()
                .contains("did not produce declared next-normed state"),
            "{missing_next_norm_err:#}"
        );
    }

    #[test]
    fn scheduled_q4_cmd3_payload_rejects_mismatched_shapes_without_fallback() {
        let err = ScheduledQ4ExpertPhaseMlpPayload::new(
            7,
            3,
            8,
            dummy_q4_payload(4, 8),
            dummy_q4_payload(5, 8),
            dummy_q4_payload(8, 4),
        )
        .unwrap_err();

        assert!(err.to_string().contains("mismatched gate/up rows"));
    }

    #[test]
    fn scheduled_q4_cmd3_payload_requires_fixed_whole_slot_source() {
        let mut gate = dummy_q4_payload(4, 8);
        gate.source = None;
        let err = ScheduledQ4ExpertPhaseMlpPayload::new(
            7,
            3,
            8,
            gate,
            dummy_q4_payload(4, 8),
            dummy_q4_payload(8, 4),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("not backed by a scheduler-owned whole-expert slot"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_q4_cmd3_payload_rejects_unresolved_scale_layout() {
        let mut gate = dummy_q4_payload(4, 8);
        gate.scale_bias_dtype = "F32";
        let err = ScheduledQ4ExpertPhaseMlpPayload::new(
            7,
            3,
            8,
            gate,
            dummy_q4_payload(4, 8),
            dummy_q4_payload(8, 4),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("resolved implementation requires BF16"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_cmd3_submission_rejects_mismatched_sources_without_fallback() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                2,
                ScheduledCmd3InputSource::MetalPostAttentionPrep,
                ScheduledSharedExpertSource::ResidentProjections,
                ScheduledNextNormSource::CpuVisibleWeights,
            )
            .unwrap();

        let input_err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0],
                1,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(input_err.to_string().contains("does not match phase input"));

        let shared_err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
            dummy_shared_expert(ScheduledSharedExpertSource::DenseCpuWeights),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0],
                1,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(
            shared_err
                .to_string()
                .contains("does not match phase shared source")
        );

        let missing_shared_shape_err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
            dummy_shared_expert_with_shape(ScheduledSharedExpertSource::ResidentProjections, None),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0; 8],
                8,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(
            missing_shared_shape_err
                .to_string()
                .contains("requires a declared shape")
        );

        let next_norm_err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::none(),
        )
        .unwrap_err();
        assert!(
            next_norm_err
                .to_string()
                .contains("requires next-norm weights")
        );

        let shared_width_err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
            dummy_shared_expert_with_shape(
                ScheduledSharedExpertSource::ResidentProjections,
                Some(ScheduledSharedExpertShape::new(4, 2, 2).unwrap()),
            ),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0; 8],
                8,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(
            shared_width_err
                .to_string()
                .contains("does not match input width")
        );

        let shared_shape_err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
            dummy_shared_expert_with_shape(
                ScheduledSharedExpertSource::ResidentProjections,
                Some(ScheduledSharedExpertShape {
                    width: 8,
                    shared_experts: 2,
                    intermediate: 2,
                    total_intermediate: 5,
                }),
            ),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0; 8],
                8,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(
            shared_shape_err
                .to_string()
                .contains("not declared graph shape")
        );

        let next_norm_width_err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input_with_width(ScheduledCmd3InputSource::MetalPostAttentionPrep, 8),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0; 4],
                4,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(
            next_norm_width_err
                .to_string()
                .contains("width 4 does not match input width 8")
        );

        let invalid_input_state_err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            DummyCmd3InputState {
                source: ScheduledCmd3InputSource::MetalPostAttentionPrep,
                state: FlashMoeCmd3InputState::cpu_normed_residual(7, 8, 4),
            },
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0; 8],
                8,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(
            invalid_input_state_err
                .to_string()
                .contains("input is not declared graph state")
        );
    }

    #[test]
    fn scheduled_cmd3_descriptor_carries_shared_expert_shape() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let shared_descriptor = ScheduledSharedExpertDescriptor::new(
            ScheduledSharedExpertSource::ResidentProjections,
            Some(ScheduledSharedExpertShape::new(8, 2, 2).unwrap()),
        )
        .unwrap();
        let cmd3 = graph
            .build_cmd3_expert_phase_with_shared_descriptor(
                7,
                2,
                ScheduledCmd3InputSource::MetalPostAttentionPrep,
                shared_descriptor,
                ScheduledNextNormSource::CpuVisibleWeights,
            )
            .unwrap();

        assert_eq!(
            cmd3.shared,
            ScheduledSharedExpertSource::ResidentProjections
        );
        assert_eq!(cmd3.shared_descriptor, Some(shared_descriptor));
        ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0; 8],
                8,
            )
            .unwrap(),
        )
        .unwrap();

        let mismatch = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
            dummy_shared_expert_with_shape(
                ScheduledSharedExpertSource::ResidentProjections,
                Some(ScheduledSharedExpertShape::new(8, 1, 4).unwrap()),
            ),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0; 8],
                8,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(
            mismatch.to_string().contains("shared descriptor"),
            "{mismatch:#}"
        );
    }

    #[test]
    fn scheduled_cmd3_builder_derives_next_norm_source_from_weights() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let shared_descriptor = ScheduledSharedExpertDescriptor::new(
            ScheduledSharedExpertSource::ResidentProjections,
            Some(ScheduledSharedExpertShape::new(8, 1, 2).unwrap()),
        )
        .unwrap();

        let no_next_norm = graph
            .build_cmd3_expert_phase_from_descriptors(
                7,
                2,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                shared_descriptor,
                ScheduledNextNormWeights::none(),
            )
            .unwrap();
        assert_eq!(no_next_norm.next_norm, ScheduledNextNormSource::None);

        let cpu_next_norm = graph
            .build_cmd3_expert_phase_from_descriptors(
                7,
                2,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                shared_descriptor,
                ScheduledNextNormWeights::cpu_visible(
                    "model.layers.8.input_layernorm.weight",
                    &[1.0; 8],
                    8,
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            cpu_next_norm.next_norm,
            ScheduledNextNormSource::CpuVisibleWeights
        );
    }

    #[test]
    fn scheduled_graph_builds_cmd3_submission_and_rejects_stale_stage() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                2,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::ResidentProjections,
                ScheduledNextNormSource::None,
            )
            .unwrap();

        graph
            .build_cmd3_submission(
                19,
                cmd3,
                &scheduled,
                dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
                dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
                ScheduledNextNormWeights::none(),
            )
            .unwrap();

        let mut stale_graph = graph.clone();
        stale_graph
            .stages
            .iter_mut()
            .find(|stage| stage.stage == FlashMoeGraphStage::Cmd3ExpertAndSharedCombine)
            .unwrap()
            .implementation = FlashMoeStageImplementation::QwenTextInput;

        let err = stale_graph
            .build_cmd3_submission(
                19,
                cmd3,
                &scheduled,
                dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
                dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
                ScheduledNextNormWeights::none(),
            )
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("does not match scheduled graph CMD3 stage"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_cmd3_submission_rejects_mismatched_or_partial_expert_slots() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                1,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::ResidentProjections,
                ScheduledNextNormSource::None,
            )
            .unwrap();
        let routes = ScheduledExpertRoutes::from_routes(
            7,
            vec![ExpertRoute {
                expert: 0,
                score: 1.0,
            }],
            1.0,
        )
        .unwrap();
        let wrong_expert =
            ScheduledExpertSet::from_parts(routes.clone(), vec![DummyCmd3Expert::whole_slot(7, 9)])
                .unwrap();

        let err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &wrong_expert,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::none(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not match routed layer"));

        let partial_slot =
            DummyCmd3Expert::whole_slot(7, 0).with_descriptor(ExpertSlotDescriptor {
                layer: 7,
                expert: 0,
                slot_offset: 0,
                slot_capacity: 128,
                payload_len: 64,
            });
        let partial_expert = ScheduledExpertSet::from_parts(routes, vec![partial_slot]).unwrap();
        let err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &partial_expert,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::none(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("must be a whole-expert slot"));
    }

    #[test]
    fn fixed_q4_graph_rejects_dense_shared_weights_for_each_text_family() {
        for layout in [qwen35_layout(), qwen3_moe_layout()] {
            let capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
            let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

            let err = graph
                .build_cmd3_expert_phase(
                    7,
                    2,
                    ScheduledCmd3InputSource::CpuNormedResidualUpload,
                    ScheduledSharedExpertSource::DenseCpuWeights,
                    ScheduledNextNormSource::None,
                )
                .unwrap_err();

            assert_eq!(err.family, graph.family());
            assert_eq!(err.stage, FlashMoeGraphStage::Cmd3ExpertAndSharedCombine);
            assert!(
                err.to_string()
                    .contains("requires resident shared projections"),
                "{err:#}"
            );
            assert!(
                err.to_string()
                    .contains("not a declared graph-stage implementation"),
                "{err:#}"
            );
        }
    }

    #[test]
    fn scheduled_graph_rejects_non_metal_cmd3_builder() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let mut graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd3 = graph
            .stages
            .iter_mut()
            .find(|stage| stage.stage == FlashMoeGraphStage::Cmd3ExpertAndSharedCombine)
            .unwrap();
        cmd3.placement = FlashMoeStagePlacement::CpuDeclared;

        let err = graph
            .build_cmd3_expert_phase(
                0,
                4,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::DenseCpuWeights,
                ScheduledNextNormSource::None,
            )
            .unwrap_err();

        assert_eq!(err.family, graph.family());
        assert_eq!(err.stage, FlashMoeGraphStage::Cmd3ExpertAndSharedCombine);
        assert!(
            err.to_string()
                .contains("CMD3 expert/shared combine must be implemented"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_expert_routes_renormalize_and_scale_full_softmax_probabilities() {
        let scheduled =
            ScheduledExpertRoutes::from_scores(12, &[(7, 0.2), (3, 0.6), (9, 0.1)], 0.25).unwrap();
        let expected = [0.25 * 2.0 / 9.0, 0.25 * 6.0 / 9.0, 0.25 * 1.0 / 9.0];

        assert_eq!(scheduled.layer, 12);
        assert_eq!(
            scheduled.routes,
            vec![
                ExpertRoute {
                    expert: 7,
                    score: 0.2,
                },
                ExpertRoute {
                    expert: 3,
                    score: 0.6,
                },
                ExpertRoute {
                    expert: 9,
                    score: 0.1,
                },
            ]
        );
        assert_eq!(scheduled.weights.len(), expected.len());
        for (actual, expected) in scheduled.weights.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn deepseek_routes_apply_the_fixed_selected_sum_floor_without_affecting_standard_routes() {
        let routes = ExpertRoute::from_scores(&[(7, 1.0e-6), (3, 2.0e-6)]).unwrap();
        let deepseek = ScheduledExpertRoutes::from_routes_with_policy(
            12,
            routes.clone(),
            QwenMoeRoutingWeightNormalization::DeepSeekRenormalizeSelectedWithFloor,
            1.5,
        )
        .unwrap();
        let standard = ScheduledExpertRoutes::from_routes_with_policy(
            12,
            routes,
            QwenMoeRoutingWeightNormalization::RenormalizeSelected,
            1.5,
        )
        .unwrap();

        assert!((deepseek.weights[0] - 1.5e-6 / 6.103515625e-5).abs() < 1.0e-7);
        assert!((deepseek.weights[1] - 3.0e-6 / 6.103515625e-5).abs() < 1.0e-7);
        assert!((standard.weights[0] - 0.5).abs() < 1.0e-6);
        assert!((standard.weights[1] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn scheduled_expert_routes_reject_unimplemented_full_softmax_weights() {
        let error = ScheduledExpertRoutes::from_routes_with_policy(
            12,
            ExpertRoute::from_scores(&[(7, 1.0), (3, 2.0)]).unwrap(),
            QwenMoeRoutingWeightNormalization::PreserveFullSoftmax,
            1.0,
        )
        .unwrap_err();

        assert!(error.to_string().contains("full expert softmax"));
    }

    #[test]
    fn scheduled_expert_routes_resolve_from_routing_command() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(12, 10, 3, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap();
        let command = routing.command_from_routes(vec![(7, 0.2), (3, 0.6), (9, 0.1)]);

        let scheduled = ScheduledExpertRoutes::from_routing_command(&command, 0.25).unwrap();

        assert_eq!(scheduled.layer, 12);
        assert_eq!(scheduled.expert_ids().collect::<Vec<_>>(), vec![7, 3, 9]);
        let expected = [0.25 * 2.0 / 9.0, 0.25 * 6.0 / 9.0, 0.25 * 1.0 / 9.0];
        for (actual, expected) in scheduled.weights.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn active_expert_scheduler_issues_routed_read_set_from_command() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(12, 10, 2, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap();
        let command = routing.command_from_routes(vec![(7, 0.25), (3, 0.75)]);
        let mut scheduler = ActiveExpertReadScheduler::new(0.25);

        let issued = scheduler.issue_routed_reads(&command).unwrap();
        assert_eq!(issued.layer(), 12);
        assert_eq!(issued.len(), 2);
        assert_eq!(issued.issues()[0].id, 0);
        assert_eq!(issued.issues()[0].key.expert, 7);
        assert_eq!(issued.issues()[1].id, 1);
        assert_eq!(issued.issues()[1].key.expert, 3);
        assert!(!issued.issues()[0].warm);
        assert!(!issued.issues()[1].warm);
        let routes = issued.into_routes();
        let expected = [0.25 * 0.25, 0.25 * 0.75];
        for (actual, expected) in routes.weights.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        let repeated = scheduler.issue_routed_reads(&command).unwrap();
        assert_eq!(repeated.issues()[0].id, 2);
        assert_eq!(repeated.issues()[1].id, 3);
        assert!(repeated.issues()[0].warm);
        assert!(repeated.issues()[1].warm);
        assert_eq!(scheduler.snapshot().issued_reads, 4);
    }

    #[test]
    fn scheduled_read_coordinator_streams_routed_slots_in_order() {
        let (_temp, store) = pbq4_import_store(&[1, 3, 7]);
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let routing = graph
            .build_routing_topk(
                0,
                graph.experts_per_layer(),
                3,
                ScheduledRoutingCandidateSource::CpuRouterScores,
            )
            .unwrap();
        let command = routing.command_from_routes(vec![(7, 0.6), (1, 0.3), (3, 0.1)]);
        let mut coordinator =
            ScheduledExpertReadCoordinator::new_with_routed_expert_scale(store, 0.9);

        let pending = coordinator.issue_routing_command(&command).unwrap();
        let scheduled = coordinator.finish_routes(pending).unwrap();

        assert_eq!(coordinator.worker_count(), 3);
        assert_eq!(
            scheduled
                .experts
                .iter()
                .map(|expert| expert.expert())
                .collect::<Vec<_>>(),
            vec![7, 1, 3]
        );
        let expected = [0.54, 0.27, 0.09];
        for (actual, expected) in scheduled.weights.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6);
        }
        let first = coordinator.snapshot();
        assert_eq!(first.issued_reads, 3);
        assert_eq!(first.positioned_reads, 3);
        assert_eq!(first.read_failures, 0);
        assert_eq!(first.warm_reads, 0);

        let pending = coordinator.issue(0, &[3]).unwrap();
        let repeated = coordinator.finish(pending).unwrap();
        assert_eq!(repeated[0].expert(), 3);
        let second = coordinator.snapshot();
        assert_eq!(second.issued_reads, 4);
        assert_eq!(second.positioned_reads, 4);
        assert_eq!(second.warm_reads, 1);
        assert!(second.warm_bytes_read > 0);
    }

    #[test]
    fn scheduled_read_coordinator_records_positioned_read_failure() {
        let (temp, store) = pbq4_import_store(&[2]);
        fs::OpenOptions::new()
            .write(true)
            .open(expert_layer_path(temp.path(), 0))
            .unwrap()
            .set_len(0)
            .unwrap();
        let mut coordinator = ScheduledExpertReadCoordinator::new(store);

        let pending = coordinator.issue(0, &[2]).unwrap();
        let error = coordinator.finish(pending).unwrap_err();

        assert!(
            error.to_string().contains("failed to read expert 2"),
            "{error:#}"
        );
        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.issued_reads, 1);
        assert_eq!(snapshot.positioned_reads, 1);
        assert_eq!(snapshot.read_failures, 1);
    }

    #[test]
    fn scheduled_expert_routes_reject_non_finite_scores() {
        let err = ScheduledExpertRoutes::from_scores(0, &[(2, f32::NAN)], 1.0).unwrap_err();

        assert!(
            err.to_string()
                .contains("expert route score for expert 2 is not finite"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_expert_batch_validates_route_weight_and_expert_counts() {
        let routes = ScheduledExpertRoutes::from_scores(3, &[(8, 0.6), (4, 0.4)], 1.0).unwrap();
        let batch = ScheduledExpertBatch::from_parts(routes, vec!["expert-8", "expert-4"]).unwrap();

        assert_eq!(batch.layer, 3);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert_eq!(batch.experts.as_ref(), ["expert-8", "expert-4"]);

        let routes = ScheduledExpertRoutes::from_scores(3, &[(8, 0.6), (4, 0.4)], 1.0).unwrap();
        let err = ScheduledExpertBatch::from_parts(routes, vec!["expert-8"]).unwrap_err();
        assert!(
            err.to_string()
                .contains("scheduled expert batch has 1 experts for 2 routes"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_expert_slot_resolves_cmd3_payload_without_legacy_adapter() {
        let slot = ScheduledExpertSlot::from_raw(raw_fixed_q4_read(3, 8));

        assert_eq!(slot.scheduled_expert_layer(), 3);
        assert_eq!(slot.scheduled_expert_id(), 8);
        assert_eq!(
            slot.scheduled_expert_slot_descriptor(),
            ExpertSlotDescriptor {
                layer: 3,
                expert: 8,
                slot_offset: 512,
                slot_capacity: tiny_fixed_q4_layout().expert_bytes,
                payload_len: tiny_fixed_q4_layout().expert_bytes,
            }
        );

        let payload = slot.scheduled_cmd3_expert_phase_payload(2).unwrap();
        let q4 = payload.q4();
        assert_eq!(q4.gate.rows, 2);
        assert_eq!(q4.gate.cols, 2);
        assert_eq!(q4.up.rows, 2);
        assert_eq!(q4.up.cols, 2);
        assert_eq!(q4.down.rows, 2);
        assert_eq!(q4.down.cols, 2);
        let source = q4
            .gate
            .source
            .expect("fixed slot should expose source offsets");
        assert_eq!(source.packed_offset, 0);
        assert_eq!(source.scale_offset, 8);
        assert_eq!(source.bias_offset, 12);
    }

    #[test]
    fn scheduled_expert_slot_resolves_typed_dense_payload_from_same_lease() {
        for dtype in [DenseExpertDtype::Bf16, DenseExpertDtype::F16] {
            let spec = FixedDenseExpertSlotSpec::new(dtype, 2, 2).unwrap();
            let slot = ScheduledExpertSlot::from_raw(raw_fixed_dense_read(3, 8, dtype));
            let payload = slot.scheduled_cmd3_expert_phase_payload(2).unwrap();
            let ScheduledExpertPhaseMlpPayload::Dense(dense) = payload else {
                panic!("fixed dense slot resolved a Q4 payload");
            };
            assert_eq!(dense.gate.dtype, dtype);
            assert_eq!(dense.gate.rows, 2);
            assert_eq!(dense.gate.cols, 2);
            assert_eq!(
                dense.up.source.bytes.as_ptr(),
                dense.gate.source.bytes.as_ptr()
            );
            assert_eq!(
                dense.down.source.bytes.as_ptr(),
                dense.gate.source.bytes.as_ptr()
            );
            assert_eq!(dense.gate.source.byte_offset, 0);
            assert_eq!(dense.up.source.byte_offset, spec.up.offset);
            assert_eq!(dense.down.source.byte_offset, spec.down.offset);
        }
    }

    #[test]
    fn scheduled_expert_slot_rejects_pbq4_component_payload_for_cmd3() {
        let slot = ScheduledExpertSlot::from_raw(raw_pbq4_read(3, 8, vec![1, 2, 3]));

        let err = slot.scheduled_cmd3_expert_phase_payload(2).unwrap_err();

        assert!(
            err.to_string().contains("PBQ4/component import data"),
            "{err:#}"
        );
    }

    #[test]
    fn pending_scheduled_expert_set_owns_read_receivers_and_routes() {
        let (tx, rx) = mpsc::channel();
        let read = PendingScheduledRead::new(77, rx);
        assert_eq!(read.id(), 77);
        let scheduled_routes = ScheduledExpertRoutes::from_routes(
            5,
            vec![ExpertRoute {
                expert: 9,
                score: 1.25,
            }],
            1.0,
        )
        .unwrap();
        let pending = PendingScheduledExpertSet::new(scheduled_routes, vec![read]);

        tx.send("expert-9").unwrap();
        let (routes, reads) = pending.into_parts();

        assert_eq!(routes.layer, 5);
        assert_eq!(
            routes.routes,
            vec![ExpertRoute {
                expert: 9,
                score: 1.25
            }]
        );
        assert_eq!(reads.len(), 1);
        assert_eq!(
            reads.into_iter().next().unwrap().recv().unwrap(),
            "expert-9"
        );
    }

    #[test]
    fn active_expert_scheduler_issues_ids_and_marks_repeated_reads_warm() {
        let mut scheduler = ActiveExpertReadScheduler::new(1.0);

        let cold = scheduler.issue_read(4, 7);
        let warm = scheduler.issue_read(4, 7);

        assert_eq!(cold.id, 0);
        assert_eq!(
            cold.key,
            ExpertReadKey {
                layer: 4,
                expert: 7
            }
        );
        assert!(!cold.warm);
        assert_eq!(warm.id, 1);
        assert_eq!(warm.key, cold.key);
        assert!(warm.warm);
        assert_eq!(scheduler.snapshot().issued_reads, 2);
    }

    #[test]
    fn active_expert_scheduler_finishes_responses_and_records_failures() {
        let mut scheduler = ActiveExpertReadScheduler::new(0.5);
        let first = scheduler.issue_read(2, 9);
        let value = scheduler
            .finish_read(
                first.id,
                ScheduledExpertReadResponse {
                    id: first.id,
                    queue_latency: Duration::from_millis(2),
                    read_path: ExpertReadPath::PositionedRead,
                    read_latency: Duration::from_millis(5),
                    bytes_read: 128,
                    warm: first.warm,
                    result: Ok("expert-9"),
                },
            )
            .unwrap();

        assert_eq!(value, "expert-9");
        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.positioned_reads, 1);
        assert_eq!(snapshot.bytes_read, 128);
        assert_eq!(snapshot.read_failures, 0);

        let second = scheduler.issue_read(2, 10);
        let err = scheduler
            .finish_read(
                second.id,
                ScheduledExpertReadResponse {
                    id: second.id + 1,
                    queue_latency: Duration::ZERO,
                    read_path: ExpertReadPath::PositionedRead,
                    read_latency: Duration::ZERO,
                    bytes_read: 0,
                    warm: false,
                    result: Ok("wrong-id"),
                },
            )
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("returned response 2 for pending read 1"),
            "{err:#}"
        );
        assert_eq!(scheduler.snapshot().read_failures, 1);
    }

    #[test]
    fn active_expert_scheduler_finishes_raw_reads_as_scheduled_slots() {
        let mut scheduler = ActiveExpertReadScheduler::new(1.0);
        let issue = scheduler.issue_read(3, 8);

        let slot = scheduler
            .finish_slot_read(
                issue.id,
                ExpertRawReadResponse {
                    id: issue.id,
                    queue_latency: Duration::from_millis(1),
                    read_path: ExpertReadPath::PositionedRead,
                    read_latency: Duration::from_millis(7),
                    bytes_read: 3,
                    warm: issue.warm,
                    result: Ok(raw_pbq4_read(3, 8, vec![1, 2, 3])),
                },
            )
            .unwrap();

        assert_eq!(slot.layer(), 3);
        assert_eq!(slot.expert(), 8);
        assert_eq!(
            slot.descriptor(),
            ExpertSlotDescriptor {
                layer: 3,
                expert: 8,
                slot_offset: 1024,
                slot_capacity: 3,
                payload_len: 3,
            }
        );
        assert!(
            slot.scheduled_cmd3_expert_phase_payload(2)
                .unwrap_err()
                .to_string()
                .contains("PBQ4/component import data instead of a resolved whole-expert payload")
        );
        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.issued_reads, 1);
        assert_eq!(snapshot.positioned_reads, 1);
        assert_eq!(snapshot.bytes_read, 3);
        assert_eq!(snapshot.read_failures, 0);
    }

    #[test]
    fn expert_scheduler_metrics_snapshot_reports_saturating_delta() {
        let mut metrics = ExpertSchedulerMetrics::default();
        metrics.record_issued_read();
        metrics.record_positioned_read();
        metrics.record_queue_latency(Duration::from_millis(7));
        metrics.record_read_latency(Duration::from_millis(11));
        metrics.record_bytes_read(128);
        let before = metrics.snapshot();

        metrics.record_issued_read();
        metrics.record_read_failure();
        metrics.record_queue_latency(Duration::from_millis(3));
        metrics.record_read_latency(Duration::from_millis(5));
        metrics.record_bytes_read(32);
        metrics.record_warm_read(Duration::from_millis(5), 32);

        let delta = metrics.snapshot().saturating_delta(before);
        assert_eq!(delta.issued_reads, 1);
        assert_eq!(delta.positioned_reads, 0);
        assert_eq!(delta.read_failures, 1);
        assert_eq!(delta.total_queue_latency, Duration::from_millis(3));
        assert_eq!(delta.total_read_latency, Duration::from_millis(5));
        assert_eq!(delta.bytes_read, 32);
        assert_eq!(delta.warm_reads, 1);
        assert_eq!(delta.warm_bytes_read, 32);
    }
}
