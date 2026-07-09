use super::capabilities::{
    FlashMoeCapabilityPlan, FlashMoeGraphStage, FlashMoeStageCapability, FlashMoeStagePlacement,
    FlashMoeUnsupportedCapability,
};
use super::experts::{
    ExpertRawPayload, ExpertRawRead, ExpertRawReadResponse, ExpertReadPath, ExpertSlotDescriptor,
    FLASHMOE_EXPERT_IO_POLICY, FixedQ4ExpertProjection, Q4MatvecPayload,
};
use super::math::{softmax_in_place, top_k};
use super::model_family::QwenMoeFamily;
use super::state::{
    FlashMoeCmd1InputState, FlashMoeCmd2InputState, FlashMoeCmd3InputState,
    FlashMoeCmd3OutputState, FlashMoeExpertPhaseOutput, FlashMoeFullAttentionKvState,
    FlashMoePostAttentionPrepState, FlashMoeRoutingOutputSource, FlashMoeRoutingOutputState,
    FlashMoeStateBufferRole, FlashMoeStatePlacement,
};
use super::weights::{
    RouterScoreBatch, RouterScoreProjectionDescriptor, ScheduledNextNormWeights,
    SharedExpertPhaseQ4Projections, SharedExpertPhaseShape, SharedExpertPhaseWeights,
};
use anyhow::{Result, bail};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeScheduledGraph {
    family: QwenMoeFamily,
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
            stages,
        })
    }

    pub fn family(&self) -> QwenMoeFamily {
        self.family
    }

    pub fn stages(&self) -> &[FlashMoeStageCapability] {
        &self.stages
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

    pub fn build_attention_math(
        &self,
        layer: usize,
        position: usize,
        implementation: ScheduledAttentionMathImplementation,
    ) -> Result<ScheduledAttentionMath, FlashMoeUnsupportedCapability> {
        let stage = *self.stage(FlashMoeGraphStage::AttentionMath);
        let expected_placement = implementation.stage_placement();
        if stage.placement != expected_placement {
            return Err(FlashMoeUnsupportedCapability::new(
                self.family,
                stage.stage,
                implementation.unsupported_reason(),
            ));
        }
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

    pub(crate) fn command_from_cmd2_preselected_routes(
        &self,
        output: ScheduledCmd2PostAttentionPrepOutput,
        routes: &[(usize, f32)],
    ) -> Result<ScheduledRoutingCommand> {
        let routing_state = output.routing();
        let routing = self.build_routing_topk(
            output.layer,
            routing_state.experts(),
            output.active_experts,
            ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
        )?;
        let routing_output = routing.validate_output_state(routing_state)?;
        routing.command_from_preselected_output(routing_output, routes)
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
        Ok(ScheduledCmd3ExpertPhase {
            stage,
            layer,
            expert_count,
            input,
            shared,
            next_norm,
        })
    }
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
    MetalKvCache,
}

impl ScheduledAttentionMathImplementation {
    fn stage_placement(self) -> FlashMoeStagePlacement {
        match self {
            Self::CpuKvCache => FlashMoeStagePlacement::CpuDeclared,
            Self::MetalKvCache => FlashMoeStagePlacement::Metal,
        }
    }

    fn kv_placement(self) -> FlashMoeStatePlacement {
        match self {
            Self::CpuKvCache => FlashMoeStatePlacement::CpuVisible,
            Self::MetalKvCache => FlashMoeStatePlacement::GpuResident,
        }
    }

    fn unsupported_reason(self) -> &'static str {
        match self {
            Self::CpuKvCache => {
                "CPU full-attention math is not declared for this graph-stage capability"
            }
            Self::MetalKvCache => {
                "Metal full-attention KV/cache math is not declared for this graph-stage capability"
            }
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledAttentionMathOutput {
    pub attention: ScheduledAttentionMath,
    state: FlashMoeFullAttentionKvState,
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
    pub(crate) fn state(self) -> FlashMoePostAttentionPrepState {
        self.state
    }

    pub(crate) fn routing(self) -> FlashMoeRoutingOutputState {
        self.state.routing()
    }

    pub(crate) fn width(self) -> usize {
        self.state.width()
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
        Ok(top_k(scores, active_experts))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledCmd3InputSource {
    MetalPostAttentionPrep,
    CpuNormedResidualUpload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledSharedExpertSource {
    None,
    DenseCpuWeights,
    ResidentQ4Projections,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledNextNormSource {
    None,
    CpuVisibleWeights,
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
            | (ScheduledSharedExpertSource::ResidentQ4Projections, None) => bail!(
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
    pub input: ScheduledCmd3InputSource,
    pub shared: ScheduledSharedExpertSource,
    pub next_norm: ScheduledNextNormSource,
}

pub trait ScheduledCmd3Input {
    fn scheduled_cmd3_input_source(&self) -> ScheduledCmd3InputSource;
    fn scheduled_cmd3_input_state(&self, layer: usize) -> FlashMoeCmd3InputState;
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
}

impl<'a> ScheduledExpertPhaseMlpPayload<'a> {
    pub(crate) fn q4(&self) -> &ScheduledQ4ExpertPhaseMlpPayload<'a> {
        match self {
            Self::Q4(payload) => payload,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScheduledQ4ExpertPhaseMlpPayload<'a> {
    pub(crate) gate: Q4MatvecPayload<'a>,
    pub(crate) up: Q4MatvecPayload<'a>,
    pub(crate) down: Q4MatvecPayload<'a>,
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
        Ok(Self { gate, up, down })
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
pub(crate) enum ScheduledSharedExpertPhaseRef<'a> {
    None,
    Dense(&'a SharedExpertPhaseWeights),
    Q4(&'a SharedExpertPhaseQ4Projections),
}

impl<'a> ScheduledSharedExpertPhaseRef<'a> {
    pub(crate) fn from_options(
        dense: Option<&'a SharedExpertPhaseWeights>,
        q4: Option<&'a SharedExpertPhaseQ4Projections>,
    ) -> Self {
        if let Some(q4) = q4 {
            Self::Q4(q4)
        } else if let Some(dense) = dense {
            Self::Dense(dense)
        } else {
            Self::None
        }
    }

    pub(crate) fn dense(self) -> Option<&'a SharedExpertPhaseWeights> {
        match self {
            Self::Dense(shared) => Some(shared),
            Self::None | Self::Q4(_) => None,
        }
    }

    pub(crate) fn q4(self) -> Option<&'a SharedExpertPhaseQ4Projections> {
        match self {
            Self::Q4(shared) => Some(shared),
            Self::None | Self::Dense(_) => None,
        }
    }

    pub(crate) fn is_some(self) -> bool {
        !matches!(self, Self::None)
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
            Self::Q4(shared) => ScheduledSharedExpertDescriptor::new(
                ScheduledSharedExpertSource::ResidentQ4Projections,
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
        if !(routed_expert_scale.is_finite() && routed_expert_scale > 0.0) {
            bail!("routed expert scale must be positive and finite");
        }
        for route in &routes {
            route.validate()?;
        }
        let mut weights: Vec<f32> = routes.iter().map(|route| route.score).collect();
        softmax_in_place(&mut weights);
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
    layer: usize,
    routes: Vec<ExpertRoute>,
    reads: Vec<PendingScheduledRead<T>>,
}

impl<T> fmt::Debug for PendingScheduledExpertSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingScheduledExpertSet")
            .field("layer", &self.layer)
            .field("routes", &self.routes)
            .field("read_count", &self.reads.len())
            .finish()
    }
}

impl<T> PendingScheduledExpertSet<T> {
    pub(crate) fn new(
        layer: usize,
        routes: Vec<ExpertRoute>,
        reads: Vec<PendingScheduledRead<T>>,
    ) -> Self {
        Self {
            layer,
            routes,
            reads,
        }
    }

    pub(crate) fn into_parts(self) -> (usize, Vec<ExpertRoute>, Vec<PendingScheduledRead<T>>) {
        (self.layer, self.routes, self.reads)
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
        let ExpertRawPayload::FixedQ4(fixed_q4) = &self.raw.payload else {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: scheduler-owned layer {} expert {} slot is not a fixed-Q4 whole-expert payload; PBQ4/component records are import compatibility only",
                self.layer(),
                self.expert()
            );
        };
        let gate = fixed_q4.matvec_payload(
            FixedQ4ExpertProjection::Gate,
            width,
            fixed_q4.spec.intermediate_size,
        );
        let up = fixed_q4.matvec_payload(
            FixedQ4ExpertProjection::Up,
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
        let Some(down) = fixed_q4.matvec_payload(FixedQ4ExpertProjection::Down, gate.rows, width)
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
}

#[derive(Debug)]
pub(crate) struct ActiveExpertReadScheduler {
    metrics: ExpertSchedulerMetrics,
    seen_reads: BTreeSet<ExpertReadKey>,
    next_read_id: u64,
    routed_expert_scale: f32,
}

impl ActiveExpertReadScheduler {
    pub(crate) fn new(routed_expert_scale: f32) -> Self {
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

    pub(crate) fn finish_routes<T>(
        &mut self,
        layer: usize,
        routes: Vec<ExpertRoute>,
        experts: Vec<T>,
        mut identify: impl FnMut(&T) -> (usize, usize),
    ) -> Result<ScheduledExpertSet<T>> {
        let scheduled_routes =
            ScheduledExpertRoutes::from_routes(layer, routes, self.routed_expert_scale)?;
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
        ExpertPackMetadata, ExpertRawPayload, FixedQ4ExpertPayload, FixedQ4ExpertSlotSpec,
    };
    use crate::inference::flashmoe::model_family::{
        QwenMoeExpertComponentKind, QwenMoeExpertComponentLayout, QwenMoeQ4ExpertLayout,
    };
    use crate::inference::flashmoe::state::FlashMoeGpuBufferDescriptor;
    use crate::inference::flashmoe::weights::{
        DenseMmapMatvecProjection, DenseQ4MmapMatvecProjection, RouterScoreProjectionBinding,
        RouterScoreProjectionDescriptor, SharedExpertPhaseQ4Projections, SharedExpertPhaseWeights,
    };
    use crate::inference::flashmoe::{QWEN35_MODEL, QwenModelConfig, QwenMoeModelLayout};

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

    fn tiny_fixed_q4_layout() -> QwenMoeQ4ExpertLayout {
        use QwenMoeExpertComponentKind::*;
        QwenMoeQ4ExpertLayout {
            expert_bytes: 30,
            group_size: 2,
            components: [
                QwenMoeExpertComponentLayout {
                    kind: GateWeight,
                    offset: 0,
                    bytes: 2,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateScale,
                    offset: 2,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateBias,
                    offset: 6,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpWeight,
                    offset: 10,
                    bytes: 2,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpScale,
                    offset: 12,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpBias,
                    offset: 16,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownWeight,
                    offset: 20,
                    bytes: 2,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownScale,
                    offset: 22,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownBias,
                    offset: 26,
                    bytes: 4,
                },
            ],
        }
    }

    fn raw_pbq4_read(layer: usize, expert: usize, payload: Vec<u8>) -> ExpertRawRead {
        let layout = qwen35_layout();
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
            fixed_q4: FixedQ4ExpertSlotSpec::new(
                layout.q4_expert_layout,
                layout.hidden_size,
                layout.moe_intermediate_size,
            )
            .unwrap(),
            recycle_pool: None,
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
            fixed_q4,
            recycle_pool: None,
            payload: ExpertRawPayload::FixedQ4(payload),
            read_latency: Duration::from_millis(3),
            read_path: ExpertReadPath::PositionedRead,
        }
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
            | ScheduledSharedExpertSource::ResidentQ4Projections => {
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

    fn dummy_shared_q4_phase() -> SharedExpertPhaseQ4Projections {
        SharedExpertPhaseQ4Projections {
            gate: dummy_q4_projection("shared.gate", 16, 32),
            up: dummy_q4_projection("shared.up", 16, 32),
            down: dummy_q4_projection("shared.down", 32, 16),
            router: dummy_q4_projection("shared.router", 1, 32),
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
        assert!(!none.is_some());
        assert!(none.dense().is_none());
        assert!(none.q4().is_none());

        let dense_ref = ScheduledSharedExpertPhaseRef::from_options(Some(&dense), None);
        let dense_descriptor = dense_ref.scheduled_shared_expert_descriptor().unwrap();
        assert_eq!(
            dense_descriptor.source,
            ScheduledSharedExpertSource::DenseCpuWeights
        );
        assert!(dense_ref.is_some());
        assert!(dense_ref.dense().is_some());
        assert!(dense_ref.q4().is_none());
        assert_eq!(
            dense_descriptor.shape,
            Some(ScheduledSharedExpertShape::new(1, 1, 2).unwrap())
        );

        let q4_ref = ScheduledSharedExpertPhaseRef::from_options(Some(&dense), Some(&q4));
        let q4_descriptor = q4_ref.scheduled_shared_expert_descriptor().unwrap();
        assert_eq!(
            q4_descriptor.source,
            ScheduledSharedExpertSource::ResidentQ4Projections
        );
        assert!(q4_ref.dense().is_none());
        assert!(q4_ref.q4().is_some());
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

    static DUMMY_Q4_PACKED: [u8; 64] = [0; 64];
    static DUMMY_Q4_SCALES: [f32; 64] = [1.0; 64];
    static DUMMY_Q4_BIASES: [f32; 64] = [0.0; 64];
    static DUMMY_Q4_SCALE_BYTES: [u8; 128] = [0; 128];
    static DUMMY_Q4_BIAS_BYTES: [u8; 128] = [0; 128];

    fn dummy_q4_payload(rows: usize, cols: usize) -> Q4MatvecPayload<'static> {
        Q4MatvecPayload {
            rows,
            cols,
            group_size: 8,
            packed: &DUMMY_Q4_PACKED[..rows * cols.div_ceil(2)],
            scales: &DUMMY_Q4_SCALES[..rows * cols.div_ceil(8)],
            biases: &DUMMY_Q4_BIASES[..rows * cols.div_ceil(8)],
            scale_bias_groups: rows * cols.div_ceil(8),
            scale_bias_dtype: "BF16",
            scale_bytes: &DUMMY_Q4_SCALE_BYTES[..rows * cols.div_ceil(8) * 2],
            bias_bytes: &DUMMY_Q4_BIAS_BYTES[..rows * cols.div_ceil(8) * 2],
            source: None,
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
                ScheduledSharedExpertSource::ResidentQ4Projections,
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
            ScheduledSharedExpertSource::ResidentQ4Projections
        );
        assert_eq!(cmd3.next_norm, ScheduledNextNormSource::CpuVisibleWeights);
    }

    #[test]
    fn scheduled_attention_math_resolves_declared_cpu_kv_state() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let attention = graph
            .build_attention_math(14, 9, ScheduledAttentionMathImplementation::CpuKvCache)
            .unwrap();

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
    fn scheduled_attention_math_rejects_undeclared_metal_kv_without_fallback() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

        let err = graph
            .build_attention_math(14, 9, ScheduledAttentionMathImplementation::MetalKvCache)
            .unwrap_err();

        assert_eq!(err.stage, FlashMoeGraphStage::AttentionMath);
        assert!(
            err.to_string()
                .contains("Metal full-attention KV/cache math is not declared"),
            "{err}"
        );
    }

    #[test]
    fn scheduled_attention_math_rejects_mismatched_kv_state_without_fallback() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let attention = graph
            .build_attention_math(14, 9, ScheduledAttentionMathImplementation::CpuKvCache)
            .unwrap();

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

        let selected = routing
            .select_from_scores(&ScheduledRoutingScoreView::new(
                3,
                ScheduledRoutingCandidateSource::CpuRouterScores,
                &[0.0, 2.0, 2.0, -1.0, 1.0],
            ))
            .unwrap();

        assert_eq!(selected, vec![(1, 2.0), (2, 2.0), (4, 1.0)]);

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

        let routing = graph
            .command_from_cmd2_preselected_routes(output, &[(7, 0.75), (3, 0.25)])
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

        let err = graph
            .command_from_cmd2_preselected_routes(output, &[(7, 0.75)])
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
    fn scheduled_cmd3_submission_validates_batch_and_sources() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                2,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::DenseCpuWeights,
                ScheduledNextNormSource::None,
            )
            .unwrap();

        let submission = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::DenseCpuWeights),
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
                ScheduledSharedExpertSource::DenseCpuWeights,
                ScheduledNextNormSource::CpuVisibleWeights,
            )
            .unwrap();
        let submission = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::DenseCpuWeights),
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
    fn scheduled_cmd3_output_state_tracks_absent_next_norm() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                2,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::DenseCpuWeights,
                ScheduledNextNormSource::None,
            )
            .unwrap();
        let command = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::DenseCpuWeights),
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
                ScheduledSharedExpertSource::DenseCpuWeights,
                ScheduledNextNormSource::CpuVisibleWeights,
            )
            .unwrap();
        let command = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::DenseCpuWeights),
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
                ScheduledSharedExpertSource::DenseCpuWeights,
                ScheduledNextNormSource::CpuVisibleWeights,
            )
            .unwrap();
        let command = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::DenseCpuWeights),
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
    fn scheduled_cmd3_submission_rejects_mismatched_sources_without_fallback() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let scheduled = dummy_scheduled_experts(7, 2);
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                2,
                ScheduledCmd3InputSource::MetalPostAttentionPrep,
                ScheduledSharedExpertSource::ResidentQ4Projections,
                ScheduledNextNormSource::CpuVisibleWeights,
            )
            .unwrap();

        let input_err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentQ4Projections),
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
            dummy_shared_expert_with_shape(
                ScheduledSharedExpertSource::ResidentQ4Projections,
                None,
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
            missing_shared_shape_err
                .to_string()
                .contains("requires a declared shape")
        );

        let next_norm_err = ScheduledCmd3Submission::new(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentQ4Projections),
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
                ScheduledSharedExpertSource::ResidentQ4Projections,
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
                ScheduledSharedExpertSource::ResidentQ4Projections,
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
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentQ4Projections),
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
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentQ4Projections),
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
    fn scheduled_cmd3_submission_rejects_mismatched_or_partial_expert_slots() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd3 = graph
            .build_cmd3_expert_phase(
                7,
                1,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::DenseCpuWeights,
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
            dummy_shared_expert(ScheduledSharedExpertSource::DenseCpuWeights),
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
            dummy_shared_expert(ScheduledSharedExpertSource::DenseCpuWeights),
            ScheduledNextNormWeights::none(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("must be a whole-expert slot"));
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
    fn scheduled_expert_routes_normalize_and_scale_scores() {
        let scheduled =
            ScheduledExpertRoutes::from_scores(12, &[(7, 1.0), (3, 2.0), (9, -1.0)], 0.25).unwrap();
        let mut expected = vec![1.0, 2.0, -1.0];
        softmax_in_place(&mut expected);
        for weight in &mut expected {
            *weight *= 0.25;
        }

        assert_eq!(scheduled.layer, 12);
        assert_eq!(
            scheduled.routes,
            vec![
                ExpertRoute {
                    expert: 7,
                    score: 1.0,
                },
                ExpertRoute {
                    expert: 3,
                    score: 2.0,
                },
                ExpertRoute {
                    expert: 9,
                    score: -1.0,
                },
            ]
        );
        assert_eq!(scheduled.weights.len(), expected.len());
        for (actual, expected) in scheduled.weights.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
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
        let routes = ScheduledExpertRoutes::from_scores(3, &[(8, 2.0), (4, 1.0)], 1.0).unwrap();
        let batch = ScheduledExpertBatch::from_parts(routes, vec!["expert-8", "expert-4"]).unwrap();

        assert_eq!(batch.layer, 3);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert_eq!(batch.experts.as_ref(), ["expert-8", "expert-4"]);

        let routes = ScheduledExpertRoutes::from_scores(3, &[(8, 2.0), (4, 1.0)], 1.0).unwrap();
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
        assert_eq!(source.scale_offset, 2);
        assert_eq!(source.bias_offset, 6);
    }

    #[test]
    fn scheduled_expert_slot_rejects_pbq4_component_payload_for_cmd3() {
        let slot = ScheduledExpertSlot::from_raw(raw_pbq4_read(3, 8, vec![1, 2, 3]));

        let err = slot.scheduled_cmd3_expert_phase_payload(2).unwrap_err();

        assert!(
            err.to_string()
                .contains("PBQ4/component records are import compatibility only"),
            "{err:#}"
        );
    }

    #[test]
    fn pending_scheduled_expert_set_owns_read_receivers_and_routes() {
        let (tx, rx) = mpsc::channel();
        let read = PendingScheduledRead::new(77, rx);
        assert_eq!(read.id(), 77);
        let pending = PendingScheduledExpertSet::new(
            5,
            vec![ExpertRoute {
                expert: 9,
                score: 1.25,
            }],
            vec![read],
        );

        tx.send("expert-9").unwrap();
        let (layer, routes, reads) = pending.into_parts();

        assert_eq!(layer, 5);
        assert_eq!(
            routes,
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
                .contains("PBQ4/component records are import compatibility only")
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
