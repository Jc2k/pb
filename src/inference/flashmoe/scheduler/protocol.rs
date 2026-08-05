#![cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScheduledCmd1InputSource {
    CpuNormedHidden,
    DeferredMetalNextNormed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledCmd1AttentionProjections {
    pub(crate) stage: FlashMoeStageCapability,
    pub(crate) layer: usize,
    pub(crate) input: ScheduledCmd1InputSource,
}

pub(crate) trait ScheduledCmd1Input {
    fn scheduled_cmd1_input_source(&self) -> ScheduledCmd1InputSource;
}

impl ScheduledCmd1Input for ScheduledCmd1InputSource {
    fn scheduled_cmd1_input_source(&self) -> ScheduledCmd1InputSource {
        *self
    }
}

#[derive(Debug)]
pub(crate) struct ScheduledCmd1Submission<TInput> {
    pub(crate) cmd1: ScheduledCmd1AttentionProjections,
    pub(crate) input: TInput,
}

impl<TInput> ScheduledCmd1Submission<TInput>
where
    TInput: ScheduledCmd1Input,
{
    pub(crate) fn new(cmd1: ScheduledCmd1AttentionProjections, input: TInput) -> Result<Self> {
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
pub(crate) struct ScheduledCmd1Command<TInput> {
    pub(crate) cmd1: ScheduledCmd1AttentionProjections,
    pub(crate) layer: usize,
    pub(crate) input: TInput,
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
pub(crate) enum ScheduledAttentionMathImplementation {
    CpuKvCache,
    CpuGlmMlaWeightAbsorption,
    MetalQ4GlmMlaAbsorbedAttention,
}

impl ScheduledAttentionMathImplementation {
    pub(crate) fn resolve(
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
pub(crate) struct ScheduledAttentionMath {
    pub(crate) stage: FlashMoeStageCapability,
    pub(crate) layer: usize,
    pub(crate) position: usize,
    pub(crate) implementation: ScheduledAttentionMathImplementation,
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
pub(crate) struct ScheduledAttentionMathOutput {
    pub(crate) attention: ScheduledAttentionMath,
    state: FlashMoeFullAttentionKvState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledMlaAttentionMathOutput {
    pub(crate) attention: ScheduledAttentionMath,
    state: FlashMoeMlaKvState,
}

impl ScheduledMlaAttentionMathOutput {
    pub(crate) fn implementation(self) -> ScheduledAttentionMathImplementation {
        self.attention.implementation
    }

    #[cfg(test)]
    pub(crate) fn state(&self) -> FlashMoeMlaKvState {
        self.state
    }
}

impl ScheduledAttentionMathOutput {
    pub(crate) fn implementation(self) -> ScheduledAttentionMathImplementation {
        self.attention.implementation
    }

    #[cfg(test)]
    pub(crate) fn state(&self) -> FlashMoeFullAttentionKvState {
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
pub(crate) enum ScheduledCmd2AttentionSource {
    CpuAttentionValues,
    MetalAttentionValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScheduledCmd2ResidualSource {
    CpuHidden,
    MetalBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScheduledCmd2AttentionInput {
    CpuValues { len: usize },
    MetalValues { len: usize },
}

impl ScheduledCmd2AttentionInput {
    pub(crate) const fn cpu_values(len: usize) -> Self {
        Self::CpuValues { len }
    }

    pub(crate) const fn metal_values(len: usize) -> Self {
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
pub(crate) enum ScheduledCmd2ResidualInput {
    CpuHidden { len: usize },
    MetalBuffer { len: usize },
}

impl ScheduledCmd2ResidualInput {
    pub(crate) const fn cpu_hidden(len: usize) -> Self {
        Self::CpuHidden { len }
    }

    pub(crate) const fn metal_buffer(len: usize) -> Self {
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
pub(crate) struct ScheduledCmd2PostAttention {
    pub(crate) stage: FlashMoeStageCapability,
    pub(crate) layer: usize,
    pub(crate) active_experts: usize,
    pub(crate) attention: ScheduledCmd2AttentionSource,
    pub(crate) residual: ScheduledCmd2ResidualSource,
}

pub(crate) trait ScheduledCmd2Inputs {
    fn scheduled_cmd2_attention_source(&self) -> ScheduledCmd2AttentionSource;
    fn scheduled_cmd2_residual_source(&self) -> ScheduledCmd2ResidualSource;
    fn scheduled_cmd2_input_state(&self, layer: usize) -> FlashMoeCmd2InputState;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledCmd2PhaseInputs {
    attention: ScheduledCmd2AttentionSource,
    residual: ScheduledCmd2ResidualSource,
    attention_len: usize,
    residual_len: usize,
}

impl ScheduledCmd2PhaseInputs {
    pub(crate) const fn new(
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

    pub(crate) const fn from_inputs(
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
pub(crate) struct ScheduledCmd2Submission<TInputs> {
    pub(crate) cmd2: ScheduledCmd2PostAttention,
    pub(crate) inputs: TInputs,
    input_state: FlashMoeCmd2InputState,
}

impl<TInputs> ScheduledCmd2Submission<TInputs>
where
    TInputs: ScheduledCmd2Inputs,
{
    pub(crate) fn new(cmd2: ScheduledCmd2PostAttention, inputs: TInputs) -> Result<Self> {
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
pub(crate) struct ScheduledCmd2Command<TInputs> {
    pub(crate) cmd2: ScheduledCmd2PostAttention,
    pub(crate) layer: usize,
    pub(crate) active_experts: usize,
    #[allow(dead_code)]
    pub(crate) inputs: TInputs,
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
pub(crate) struct ScheduledCmd2PostAttentionPrepOutput {
    pub(crate) cmd2: ScheduledCmd2PostAttention,
    pub(crate) layer: usize,
    pub(crate) active_experts: usize,
    pub(crate) input_state: FlashMoeCmd2InputState,
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
#[allow(dead_code)]
pub(crate) enum ScheduledRoutingCandidateSource {
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
pub(crate) struct ScheduledRoutingTopK {
    pub(crate) stage: FlashMoeStageCapability,
    pub(crate) layer: usize,
    pub(crate) experts: usize,
    pub(crate) active_experts: usize,
    pub(crate) source: ScheduledRoutingCandidateSource,
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

    pub(crate) fn command_from_routes(
        &self,
        routes: impl Into<InlineRoutePairs>,
    ) -> ScheduledRoutingCommand {
        ScheduledRoutingCommand {
            routing: *self,
            layer: self.layer,
            active_experts: self.active_experts,
            source: self.source,
            routes: routes.into(),
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

    pub(crate) fn select_from_scores<TScores>(&self, scores: &TScores) -> Result<InlineRoutePairs>
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

    pub(crate) fn validate_preselected(&self, routes: &[(usize, f32)]) -> Result<InlineRoutePairs> {
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
        let mut validated = InlineRoutePairs::with_capacity(routes.len());
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
            if validated.iter().any(|(selected, _)| *selected == expert) {
                bail!("FlashMoe scheduled routing selected expert {expert} more than once");
            }
            validated.push((expert, score));
        }
        Ok(validated)
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
pub(crate) struct ScheduledRoutingCommand {
    pub(crate) routing: ScheduledRoutingTopK,
    pub(crate) layer: usize,
    pub(crate) active_experts: usize,
    pub(crate) source: ScheduledRoutingCandidateSource,
    pub(crate) routes: InlineRoutePairs,
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
        for (index, (expert, score)) in self.routes.iter().copied().enumerate() {
            if expert >= self.routing.experts {
                bail!(
                    "FlashMoe scheduled routing command selected expert {} outside expert count {}",
                    expert,
                    self.routing.experts
                );
            }
            if self.routes[..index]
                .iter()
                .any(|(selected, _)| *selected == expert)
            {
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
#[allow(dead_code)]
pub(crate) enum ScheduledCmd3InputSource {
    MetalPostAttentionPrep,
    CpuNormedResidualUpload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScheduledSharedExpertSource {
    None,
    DenseCpuWeights,
    ResidentProjections,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScheduledNextNormSource {
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
pub(crate) struct ScheduledSharedExpertDescriptor {
    pub(crate) source: ScheduledSharedExpertSource,
    pub(crate) shape: Option<ScheduledSharedExpertShape>,
}

impl ScheduledSharedExpertDescriptor {
    pub(crate) fn new(
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
pub(crate) struct ScheduledSharedExpertShape {
    pub(crate) width: usize,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) total_intermediate: usize,
}

impl ScheduledSharedExpertShape {
    #[cfg(test)]
    pub(crate) fn new(width: usize, shared_experts: usize, intermediate: usize) -> Result<Self> {
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

    pub(crate) fn is_declared_graph_shape(self) -> bool {
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
pub(crate) struct ScheduledCmd3ExpertPhase {
    pub(crate) stage: FlashMoeStageCapability,
    pub(crate) layer: usize,
    pub(crate) expert_count: usize,
    pub(crate) expert_storage: ExpertStorageLayout,
    pub(crate) input: ScheduledCmd3InputSource,
    pub(crate) shared: ScheduledSharedExpertSource,
    pub(crate) shared_descriptor: Option<ScheduledSharedExpertDescriptor>,
    pub(crate) next_norm: ScheduledNextNormSource,
}

pub(crate) trait ScheduledCmd3Input {
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

pub(crate) trait ScheduledCmd3Expert {
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
pub(crate) enum ScheduledExpertPhaseMlpPayload<'a> {
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

    pub(crate) fn q4_checked(&self) -> Option<&ScheduledQ4ExpertPhaseMlpPayload<'a>> {
        match self {
            Self::Q4(payload) => Some(payload),
            Self::Dense(_) | Self::DeepSeekGguf(_) => None,
        }
    }

    pub(crate) fn storage_layout(&self) -> ExpertStorageLayout {
        match self {
            Self::Q4(payload) => {
                if payload
                    .gate
                    .scale_bias_dtype
                    .eq_ignore_ascii_case(super::super::experts::EXPERT_SCALE_DTYPE_E8M0)
                {
                    ExpertStorageLayout::FixedMxfp4
                } else {
                    ExpertStorageLayout::FixedQ4
                }
            }
            Self::Dense(payload) => match payload.gate.dtype {
                super::super::experts::DenseExpertDtype::Bf16 => ExpertStorageLayout::FixedBf16,
                super::super::experts::DenseExpertDtype::F16 => ExpertStorageLayout::FixedF16,
            },
            Self::DeepSeekGguf(_) => ExpertStorageLayout::FixedDeepSeekGguf,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScheduledDeepSeekGgufExpertPhaseMlpPayload<'a> {
    pub(crate) layer: usize,
    pub(crate) expert: usize,
    pub(crate) spec: DeepSeekGgufExpertSlotSpec,
    pub(crate) bytes: &'a ReusableExpertBytes,
}

impl<'a> ScheduledDeepSeekGgufExpertPhaseMlpPayload<'a> {
    pub(crate) fn new(
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
pub(crate) struct ScheduledDenseExpertPhaseMlpPayload<'a> {
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
pub(crate) struct ScheduledQ4ExpertPhaseMlpPayload<'a> {
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
            .eq_ignore_ascii_case(super::super::experts::EXPERT_SCALE_DTYPE_E8M0);
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

pub(crate) trait ScheduledCmd3ExpertPayload {
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

pub(crate) trait ScheduledSharedExpert {
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
pub(crate) struct ScheduledCmd3Submission<'a, TExpert, TInput, TShared> {
    pub(crate) cmd3: ScheduledCmd3ExpertPhase,
    pub(crate) position: usize,
    pub(crate) scheduled: &'a ScheduledExpertSet<TExpert>,
    pub(crate) input: TInput,
    pub(crate) input_state: FlashMoeCmd3InputState,
    pub(crate) shared: TShared,
    pub(crate) next_norm_weights: ScheduledNextNormWeights<'a>,
}

impl<'a, TExpert, TInput, TShared> ScheduledCmd3Submission<'a, TExpert, TInput, TShared>
where
    TExpert: ScheduledCmd3Expert,
    TInput: ScheduledCmd3Input,
    TShared: ScheduledSharedExpert,
{
    pub(crate) fn new(
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
        for (route, expert) in scheduled.routes().iter().zip(scheduled.experts.iter()) {
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
pub(crate) struct ScheduledCmd3Command<'a, TExpert, TInput, TShared> {
    pub(crate) cmd3: ScheduledCmd3ExpertPhase,
    pub(crate) position: usize,
    pub(crate) layer: usize,
    pub(crate) experts: Arc<[TExpert]>,
    pub(crate) weights: &'a [f32],
    pub(crate) input: TInput,
    pub(crate) input_state: FlashMoeCmd3InputState,
    pub(crate) shared: TShared,
    pub(crate) next_norm_weights: ScheduledNextNormWeights<'a>,
    pub(crate) payloads: Vec<ScheduledExpertPhaseMlpPayload<'a>>,
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
    pub(crate) cmd3: ScheduledCmd3ExpertPhase,
    pub(crate) layer: usize,
    pub(crate) input_state: FlashMoeCmd3InputState,
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
            weights: self.scheduled.weights(),
            input: self.input,
            input_state: self.input_state,
            shared: self.shared,
            next_norm_weights: self.next_norm_weights,
            payloads,
        })
    }
}
