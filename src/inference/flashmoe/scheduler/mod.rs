use super::capabilities::{
    FlashMoeCapabilityPlan, FlashMoeGraphStage, FlashMoeStageCapability,
    FlashMoeStageImplementation, FlashMoeStagePlacement, FlashMoeUnsupportedCapability,
};
use super::experts::{
    DeepSeekGgufExpertSlotSpec, DenseMatvecPayload, DirectExpertReadSummary,
    EXPERT_SCALE_BIAS_DTYPE_BF16, ExpertLayerPrepareSummary, ExpertMlpProjection, ExpertRawPayload,
    ExpertRawRead, ExpertRawReadResponse, ExpertReadPath, ExpertReadWorkerPool,
    ExpertSlotDescriptor, ExpertSlotStore, ExpertStorageLayout, FLASHMOE_EXPERT_IO_POLICY,
    PendingExpertLayerPrepare, Q4MatvecPayload, Q4MatvecSource, ReusableExpertBytes,
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
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

const BATCH_EXPERT_READ_WORKERS: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeScheduledGraph {
    family: QwenMoeFamily,
    layers: usize,
    first_expert_layer: usize,
    experts_per_layer: usize,
    active_experts: usize,
    expert_storage: ExpertStorageLayout,
    routing_weight_normalization: QwenMoeRoutingWeightNormalization,
    routed_expert_scale: f32,
    attention_layers: Box<[ScheduledLayerAttentionImplementation]>,
    stages: Vec<FlashMoeStageCapability>,
}

impl FlashMoeScheduledGraph {
    pub(crate) fn from_capabilities(
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
            layers: capabilities.expert_storage.layers,
            first_expert_layer: capabilities.expert_storage.first_expert_layer,
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

    #[cfg(test)]
    pub(crate) fn family(&self) -> QwenMoeFamily {
        self.family
    }

    pub(crate) fn experts_per_layer(&self) -> usize {
        self.experts_per_layer
    }

    pub(crate) fn layers(&self) -> usize {
        self.layers
    }

    pub(crate) fn first_expert_layer(&self) -> usize {
        self.first_expert_layer
    }

    pub(crate) fn active_experts(&self) -> usize {
        self.active_experts
    }

    pub(crate) fn routing_weight_normalization(&self) -> QwenMoeRoutingWeightNormalization {
        self.routing_weight_normalization
    }

    pub(crate) fn routed_expert_scale(&self) -> f32 {
        self.routed_expert_scale
    }

    #[cfg(test)]
    pub(crate) fn stages(&self) -> &[FlashMoeStageCapability] {
        &self.stages
    }

    fn attention_implementation(
        &self,
        layer: usize,
    ) -> Option<ScheduledLayerAttentionImplementation> {
        self.attention_layers.get(layer).copied()
    }

    pub(crate) fn stage(&self, stage: FlashMoeGraphStage) -> &FlashMoeStageCapability {
        self.stages
            .iter()
            .find(|capability| capability.stage == stage)
            .expect("scheduled graph contains every FlashMoe stage")
    }

    pub(crate) fn active_expert_reads(&self) -> &FlashMoeStageCapability {
        self.stage(FlashMoeGraphStage::ActiveExpertReads)
    }

    #[cfg(test)]
    pub(crate) fn cmd_sequence(&self) -> [&FlashMoeStageCapability; 3] {
        [
            self.stage(FlashMoeGraphStage::DeferredPreviousCmd3),
            self.stage(FlashMoeGraphStage::Cmd1AttentionProjections),
            self.stage(FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection),
        ]
    }

    #[cfg(test)]
    pub(crate) fn declares_scheduler_owned_expert_reads(&self) -> bool {
        matches!(
            self.active_expert_reads().placement,
            FlashMoeStagePlacement::SchedulerIo | FlashMoeStagePlacement::SchedulerMemory
        )
    }

    pub(crate) fn build_cmd2_post_attention(
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

    pub(crate) fn build_cmd2_submission<TInputs>(
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

    pub(crate) fn build_cmd2_command(
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

    pub(crate) fn build_attention_math(
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

    pub(crate) fn build_routing_topk(
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

    pub(crate) fn build_cmd1_attention_projections(
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

    pub(crate) fn build_cmd1_submission<TInput>(
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

    pub(crate) fn build_cmd3_expert_phase(
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

    pub(crate) fn build_cmd3_expert_phase_with_shared_descriptor(
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

    pub(crate) fn build_cmd3_expert_phase_from_descriptors(
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

    pub(crate) fn build_cmd3_submission<'a, TExpert, TInput, TShared>(
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

    pub(crate) fn build_cmd3_command_from_descriptors<'a, TExpert, TInput, TShared>(
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
    expert_access: ScheduledExpertAccessCoordinator,
}

#[derive(Debug)]
pub(crate) struct PendingScheduledExpertLayerPrepare {
    layer: usize,
    pending: PendingExpertLayerPrepare,
}

/// Scheduler-resolved expert working set for one layer-major token matrix.
///
/// Routes and weights remain in token-major/top-k order so the Metal combine
/// stage can preserve the scalar accumulation order. `experts` is the sorted
/// unique union acquired through the graph's resident or positioned-read
/// implementation exactly once for this layer.
#[derive(Debug)]
pub(crate) struct ScheduledLayerMajorExperts {
    layer: usize,
    rows: usize,
    active_experts: usize,
    route_slots: Vec<usize>,
    weights: Vec<f32>,
    experts: Arc<[Arc<ScheduledExpertSlot>]>,
}

impl ScheduledLayerMajorExperts {
    pub(crate) fn layer(&self) -> usize {
        self.layer
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn active_experts(&self) -> usize {
        self.active_experts
    }

    pub(crate) fn route_slots(&self) -> &[usize] {
        &self.route_slots
    }

    pub(crate) fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub(crate) fn experts(&self) -> &Arc<[Arc<ScheduledExpertSlot>]> {
        &self.experts
    }

    pub(crate) fn route_mix_hashes(&self) -> impl Iterator<Item = u64> + '_ {
        self.route_slots
            .iter()
            .map(|slot| self.experts[*slot].mix_hash())
    }
}

impl FlashMoeExecutionScheduler {
    #[cfg(test)]
    pub(crate) fn new(
        graph: FlashMoeScheduledGraph,
        expert_store: ExpertSlotStore,
    ) -> Result<Self> {
        Self::new_with_resident_binding(graph, expert_store, |_| {
            bail!("resident expert graph construction requires a Metal backing binder")
        })
    }

    pub(crate) fn new_with_resident_binding(
        graph: FlashMoeScheduledGraph,
        expert_store: ExpertSlotStore,
        mut bind_resident: impl FnMut(&ReusableExpertBytes) -> Result<()>,
    ) -> Result<Self> {
        if graph.attention_layers.is_empty() {
            bail!(
                "FlashMoe execution scheduler requires a resolved attention implementation for every layer"
            );
        }
        let expert_access = match graph.active_expert_reads().implementation {
            FlashMoeStageImplementation::ParallelPositionedWholeExpertReads => {
                ScheduledExpertAccessCoordinator::Streamed(
                    ScheduledExpertReadCoordinator::new_with_routing_policy(
                        expert_store,
                        graph.routing_weight_normalization(),
                        graph.routed_expert_scale(),
                    ),
                )
            }
            FlashMoeStageImplementation::ResidentMappedWholeExpertSlots => {
                ScheduledExpertAccessCoordinator::Resident(ScheduledResidentExpertTable::new(
                    &graph,
                    expert_store,
                    &mut bind_resident,
                )?)
            }
            implementation => {
                bail!(
                    "FlashMoe active-expert stage resolved unsupported scheduler implementation {implementation:?}"
                )
            }
        };
        Ok(Self {
            graph,
            expert_access,
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
        let pending = self.expert_access.issue_routing_command(&routing)?;
        self.expert_access.finish_routes(pending)
    }

    /// Resolve all token routes for a layer-major prefill matrix and acquire
    /// their sorted unique expert union through the already-selected expert
    /// access implementation. Resident graphs clone mapped slots without
    /// issuing reads; streamed graphs issue one scheduler-owned parallel pread
    /// for each unique expert.
    pub(crate) fn resolve_layer_major_experts(
        &mut self,
        layer: usize,
        row_routes: &[Vec<(usize, f32)>],
    ) -> Result<ScheduledLayerMajorExperts> {
        if row_routes.is_empty() {
            bail!("FlashMoe layer-major routing requires at least one token row");
        }
        let active_experts = self.graph.active_experts();
        let mut normalized_rows = Vec::with_capacity(row_routes.len());
        let mut unique_ids = BTreeSet::new();
        for routes in row_routes {
            let command = self
                .graph
                .build_routing_topk(
                    layer,
                    self.graph.experts_per_layer(),
                    active_experts,
                    ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
                )?
                .command_from_preselected(routes)?;
            let normalized = self.expert_access.normalize_routes(&command)?;
            if normalized.routes.len() != active_experts
                || normalized.weights.len() != active_experts
            {
                bail!(
                    "FlashMoe layer-major row resolved {} routes and {} weights, expected K={active_experts} at layer {layer}",
                    normalized.routes.len(),
                    normalized.weights.len()
                );
            }
            unique_ids.extend(normalized.routes.iter().map(|route| route.expert));
            normalized_rows.push(normalized);
        }
        let unique_ids = unique_ids.into_iter().collect::<Vec<_>>();
        let experts = self.expert_access.acquire_unique(layer, &unique_ids)?;
        if experts.len() != unique_ids.len() {
            bail!(
                "FlashMoe layer-major expert access returned {} slots for {} unique experts at layer {layer}",
                experts.len(),
                unique_ids.len()
            );
        }
        let slot_by_id = unique_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, expert)| (expert, slot))
            .collect::<BTreeMap<_, _>>();
        let route_count = row_routes
            .len()
            .checked_mul(active_experts)
            .context("FlashMoe layer-major route count overflow")?;
        let mut route_slots = Vec::with_capacity(route_count);
        let mut weights = Vec::with_capacity(route_count);
        for normalized in normalized_rows {
            for (route, weight) in normalized.routes.iter().zip(normalized.weights) {
                route_slots.push(*slot_by_id.get(&route.expert).with_context(|| {
                    format!(
                        "FlashMoe layer-major unique expert union omitted layer {layer} expert {}",
                        route.expert
                    )
                })?);
                weights.push(weight);
            }
        }
        Ok(ScheduledLayerMajorExperts {
            layer,
            rows: row_routes.len(),
            active_experts,
            route_slots,
            weights,
            experts: Arc::from(experts),
        })
    }

    /// Read the unique expert working set for one layer-major prefill batch.
    ///
    /// The caller supplies sorted unique ids so each whole expert is streamed
    /// exactly once into request-scoped batch storage. The reads enter the
    /// scheduler's positioned-I/O metrics but intentionally bypass reusable
    /// decode slots for graph-owned staging; no application expert cache is
    /// introduced.
    fn validate_unique_expert_ids(&self, layer: usize, experts: &[usize]) -> Result<()> {
        if layer >= self.graph.attention_layers.len() {
            bail!(
                "FlashMoe batch expert layer {layer} is outside resolved layer count {}",
                self.graph.attention_layers.len()
            );
        }
        if experts.is_empty() {
            bail!("FlashMoe batch expert read requires at least one expert");
        }
        let mut previous = None;
        for &expert in experts {
            if expert >= self.graph.experts_per_layer() {
                bail!(
                    "FlashMoe batch expert {expert} is outside resolved expert count {} for layer {layer}",
                    self.graph.experts_per_layer()
                );
            }
            if previous.is_some_and(|previous| expert <= previous) {
                bail!(
                    "FlashMoe batch expert ids must be sorted and unique, got {previous:?} then {expert}"
                );
            }
            previous = Some(expert);
        }
        Ok(())
    }

    /// Stream a calculated sorted-unique batch working set directly into one
    /// graph-declared request-scoped whole-layer staging buffer.
    pub(crate) fn read_unique_experts_into(
        &mut self,
        layer: usize,
        experts: &[usize],
        destination: &mut [u8],
        slot_stride: usize,
    ) -> Result<DirectExpertReadSummary> {
        self.validate_unique_expert_ids(layer, experts)?;
        self.expert_access.read_experts_into(
            layer,
            experts,
            destination,
            slot_stride,
            BATCH_EXPERT_READ_WORKERS,
        )
    }

    /// Start a fixed whole-layer positioned stream directly into a
    /// graph-owned request staging allocation. The caller must not access or
    /// release the destination until `finish_expert_layer_prepare` returns.
    pub(crate) unsafe fn issue_expert_layer_prepare_into(
        &mut self,
        layer: usize,
        destination: &mut [u8],
    ) -> Result<PendingScheduledExpertLayerPrepare> {
        if layer >= self.graph.attention_layers.len() {
            bail!(
                "FlashMoe expert layer staging {layer} is outside resolved layer count {}",
                self.graph.attention_layers.len()
            );
        }
        Ok(PendingScheduledExpertLayerPrepare {
            layer,
            pending: unsafe {
                self.expert_access.issue_layer_prepare_into(
                    layer,
                    destination,
                    BATCH_EXPERT_READ_WORKERS,
                )?
            },
        })
    }

    pub(crate) fn finish_expert_layer_prepare(
        &mut self,
        pending: PendingScheduledExpertLayerPrepare,
    ) -> Result<ExpertLayerPrepareSummary> {
        if pending.pending.layer() != pending.layer {
            bail!(
                "FlashMoe expert layer preparation handle changed from layer {} to {}",
                pending.layer,
                pending.pending.layer()
            );
        }
        let summary = self.expert_access.finish_layer_prepare(pending.pending)?;
        if summary.layer != pending.layer {
            bail!(
                "FlashMoe expert layer preparation completed layer {}, expected {}",
                summary.layer,
                pending.layer
            );
        }
        Ok(summary)
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
        let before = self.expert_access.snapshot();
        let issue_started = Instant::now();
        let pending = self.expert_access.issue_routing_command(routing)?;
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
        let scheduled = self.expert_access.finish_routes(pending.pending)?;
        let expert_io_elapsed = pending.issue_elapsed + finish_started.elapsed();
        let expert_delta = self
            .expert_access
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

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> ExpertSchedulerSnapshot {
        self.expert_access.snapshot()
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
    pending: PendingScheduledExpertAccess,
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

mod protocol;
pub(crate) use protocol::*;

mod expert_access;
pub(crate) use expert_access::*;

#[cfg(test)]
#[path = "../tests/scheduler.rs"]
mod tests;
