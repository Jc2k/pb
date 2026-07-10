use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use anyhow::{Result, bail};

#[derive(Debug, Clone)]
pub(crate) struct FlashMoeSessionState<K> {
    pub(crate) tokens: Vec<u32>,
    pub(crate) kv_cache: K,
    pub(crate) last_hidden: Vec<f32>,
}

impl<K> FlashMoeSessionState<K> {
    pub(crate) fn new(tokens: Vec<u32>, kv_cache: K, last_hidden: Vec<f32>) -> Self {
        Self {
            tokens,
            kv_cache,
            last_hidden,
        }
    }
}

pub(crate) fn common_token_prefix_len(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

pub(crate) fn reusable_session_prefix_len(
    cached_tokens: &[u32],
    prompt_tokens: &[u32],
) -> Option<usize> {
    let prefix_len = common_token_prefix_len(cached_tokens, prompt_tokens);
    (prefix_len == cached_tokens.len()).then_some(prefix_len)
}

pub(crate) fn take_reusable_session_cache_entry<K>(
    session_cache: &mut BTreeMap<String, FlashMoeSessionState<K>>,
    session_id: &str,
    prompt_tokens: &[u32],
) -> Option<(usize, FlashMoeSessionState<K>)> {
    let prefix_len = session_cache
        .get(session_id)
        .and_then(|state| reusable_session_prefix_len(&state.tokens, prompt_tokens))?;
    session_cache
        .remove(session_id)
        .map(|state| (prefix_len, state))
}

pub(crate) fn stable_session_cache_tokens(prompt_tokens: &[u32]) -> Vec<u32> {
    // Assistant generations can be parsed into structured tool calls and then
    // re-rendered canonically on the next turn. Cache only rendered prompt
    // tokens whose exact bytes are already part of the transcript contract.
    prompt_tokens.to_vec()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashMoeStatePlacement {
    CpuVisible,
    GpuResident,
}

impl FlashMoeStatePlacement {
    pub(crate) const GRAPH_PLACEMENTS: [Self; 2] = [Self::CpuVisible, Self::GpuResident];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashMoeStateBufferRole {
    AttentionValues,
    Hidden,
    Residual,
    Normed,
    NextLayerNormed,
    RouterScores,
    RoutingTopK,
    Kv,
    Recurrent,
}

impl FlashMoeStateBufferRole {
    pub(crate) const GENERATION_ROLES: [Self; 9] = [
        Self::AttentionValues,
        Self::Hidden,
        Self::Residual,
        Self::Normed,
        Self::NextLayerNormed,
        Self::RouterScores,
        Self::RoutingTopK,
        Self::Kv,
        Self::Recurrent,
    ];
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeCpuBuffer {
    role: FlashMoeStateBufferRole,
    values: Vec<f32>,
}

impl FlashMoeCpuBuffer {
    fn new(role: FlashMoeStateBufferRole, values: Vec<f32>) -> Self {
        Self { role, values }
    }

    pub(crate) fn hidden(values: Vec<f32>) -> Self {
        Self::new(FlashMoeStateBufferRole::Hidden, values)
    }

    #[cfg(test)]
    pub(crate) fn residual(values: Vec<f32>) -> Self {
        Self::new(FlashMoeStateBufferRole::Residual, values)
    }

    pub(crate) fn normed(values: Vec<f32>) -> Self {
        Self::new(FlashMoeStateBufferRole::Normed, values)
    }

    pub(crate) fn next_layer_normed(values: Vec<f32>) -> Self {
        Self::new(FlashMoeStateBufferRole::NextLayerNormed, values)
    }

    pub(crate) fn into_role(mut self, role: FlashMoeStateBufferRole) -> Self {
        self.role = role;
        self
    }

    pub(crate) fn role(&self) -> FlashMoeStateBufferRole {
        self.role
    }

    pub(crate) fn placement(&self) -> FlashMoeStatePlacement {
        FlashMoeStatePlacement::CpuVisible
    }

    pub(crate) fn is_declared_graph_state(&self) -> bool {
        FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }

    pub(crate) fn replace_values(&mut self, values: Vec<f32>) {
        self.values = values;
    }

    #[cfg(test)]
    pub(crate) fn clone_values(&self) -> Vec<f32> {
        self.values.clone()
    }

    pub(crate) fn into_values(self) -> Vec<f32> {
        self.values
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeGpuBufferDescriptor {
    role: FlashMoeStateBufferRole,
    len: usize,
}

impl FlashMoeGpuBufferDescriptor {
    pub(crate) fn new(role: FlashMoeStateBufferRole, len: usize) -> Self {
        Self { role, len }
    }

    pub(crate) fn hidden(len: usize) -> Self {
        Self::new(FlashMoeStateBufferRole::Hidden, len)
    }

    pub(crate) fn residual(len: usize) -> Self {
        Self::new(FlashMoeStateBufferRole::Residual, len)
    }

    pub(crate) fn normed(len: usize) -> Self {
        Self::new(FlashMoeStateBufferRole::Normed, len)
    }

    pub(crate) fn next_layer_normed(len: usize) -> Self {
        Self::new(FlashMoeStateBufferRole::NextLayerNormed, len)
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        self.role
    }

    pub(crate) fn len(self) -> usize {
        self.len
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        FlashMoeStatePlacement::GpuResident
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
            && self.len() > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashMoeStateBufferDescriptor {
    role: FlashMoeStateBufferRole,
    len: usize,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeStateBufferDescriptor {
    pub(crate) fn new(
        role: FlashMoeStateBufferRole,
        len: usize,
        placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            role,
            len,
            placement,
        }
    }

    #[cfg(test)]
    pub(crate) fn cpu(role: FlashMoeStateBufferRole, len: usize) -> Self {
        Self::new(role, len, FlashMoeStatePlacement::CpuVisible)
    }

    pub(crate) fn gpu(descriptor: FlashMoeGpuBufferDescriptor) -> Self {
        Self::new(
            descriptor.role(),
            descriptor.len(),
            FlashMoeStatePlacement::GpuResident,
        )
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        self.role
    }

    pub(crate) fn len(self) -> usize {
        self.len
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
            && self.len() > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashMoeCmd2InputState {
    layer: usize,
    attention: FlashMoeStateBufferDescriptor,
    residual: FlashMoeStateBufferDescriptor,
}

impl FlashMoeCmd2InputState {
    pub(crate) fn new(
        layer: usize,
        attention_len: usize,
        attention_placement: FlashMoeStatePlacement,
        residual_len: usize,
        residual_placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            layer,
            attention: FlashMoeStateBufferDescriptor::new(
                FlashMoeStateBufferRole::AttentionValues,
                attention_len,
                attention_placement,
            ),
            residual: FlashMoeStateBufferDescriptor::new(
                FlashMoeStateBufferRole::Residual,
                residual_len,
                residual_placement,
            ),
        }
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn attention(self) -> FlashMoeStateBufferDescriptor {
        self.attention
    }

    pub(crate) fn residual(self) -> FlashMoeStateBufferDescriptor {
        self.residual
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.attention.is_declared_graph_state()
            && self.residual.is_declared_graph_state()
            && self.attention.role() == FlashMoeStateBufferRole::AttentionValues
            && self.residual.role() == FlashMoeStateBufferRole::Residual
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashMoeRoutingOutputSource {
    CpuRouterScores,
    FusedMetalPostAttentionPrepCpuTopK,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeRoutingOutputState {
    layer: usize,
    experts: usize,
    active_experts: usize,
    source: FlashMoeRoutingOutputSource,
    role: FlashMoeStateBufferRole,
    len: usize,
}

impl FlashMoeRoutingOutputState {
    pub(crate) fn cpu_router_scores(layer: usize, experts: usize, active_experts: usize) -> Self {
        Self::new(
            layer,
            experts,
            active_experts,
            FlashMoeRoutingOutputSource::CpuRouterScores,
            FlashMoeStateBufferRole::RouterScores,
            experts,
        )
    }

    pub(crate) fn fused_metal_post_attention_cpu_topk(
        layer: usize,
        experts: usize,
        active_experts: usize,
    ) -> Self {
        Self::new(
            layer,
            experts,
            active_experts,
            FlashMoeRoutingOutputSource::FusedMetalPostAttentionPrepCpuTopK,
            FlashMoeStateBufferRole::RoutingTopK,
            active_experts,
        )
    }

    fn new(
        layer: usize,
        experts: usize,
        active_experts: usize,
        source: FlashMoeRoutingOutputSource,
        role: FlashMoeStateBufferRole,
        len: usize,
    ) -> Self {
        Self {
            layer,
            experts,
            active_experts,
            source,
            role,
            len,
        }
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn experts(self) -> usize {
        self.experts
    }

    pub(crate) fn active_experts(self) -> usize {
        self.active_experts
    }

    pub(crate) fn source(self) -> FlashMoeRoutingOutputSource {
        self.source
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        self.role
    }

    pub(crate) fn len(self) -> usize {
        self.len
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        FlashMoeStatePlacement::CpuVisible
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        let expected_role = match self.source {
            FlashMoeRoutingOutputSource::CpuRouterScores => FlashMoeStateBufferRole::RouterScores,
            FlashMoeRoutingOutputSource::FusedMetalPostAttentionPrepCpuTopK => {
                FlashMoeStateBufferRole::RoutingTopK
            }
        };
        self.experts > 0
            && self.active_experts > 0
            && self.active_experts <= self.experts
            && self.role() == expected_role
            && self.len()
                == match expected_role {
                    FlashMoeStateBufferRole::RouterScores => self.experts,
                    FlashMoeStateBufferRole::RoutingTopK => self.active_experts,
                    _ => 0,
                }
            && FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoePostAttentionPrepState {
    residual: FlashMoeGpuBufferDescriptor,
    normed: FlashMoeGpuBufferDescriptor,
    routing: FlashMoeRoutingOutputState,
}

impl FlashMoePostAttentionPrepState {
    pub(crate) fn new(layer: usize, width: usize, experts: usize, active_experts: usize) -> Self {
        Self {
            residual: FlashMoeGpuBufferDescriptor::residual(width),
            normed: FlashMoeGpuBufferDescriptor::normed(width),
            routing: FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(
                layer,
                experts,
                active_experts,
            ),
        }
    }

    pub(crate) fn width(self) -> usize {
        self.residual.len()
    }

    pub(crate) fn residual(self) -> FlashMoeGpuBufferDescriptor {
        self.residual
    }

    pub(crate) fn normed(self) -> FlashMoeGpuBufferDescriptor {
        self.normed
    }

    pub(crate) fn active_experts(self) -> usize {
        self.routing.active_experts()
    }

    pub(crate) fn routing(self) -> FlashMoeRoutingOutputState {
        self.routing
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.residual.is_declared_graph_state()
            && self.normed.is_declared_graph_state()
            && self.residual.role() == FlashMoeStateBufferRole::Residual
            && self.normed.role() == FlashMoeStateBufferRole::Normed
            && self.residual.len() == self.normed.len()
            && self.routing.is_declared_graph_state()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashMoeCmd3InputState {
    layer: usize,
    residual: FlashMoeStateBufferDescriptor,
    normed: FlashMoeStateBufferDescriptor,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeCmd3InputState {
    #[cfg(test)]
    pub(crate) fn cpu_normed_residual(
        layer: usize,
        normed_len: usize,
        residual_len: usize,
    ) -> Self {
        Self {
            layer,
            residual: FlashMoeStateBufferDescriptor::cpu(
                FlashMoeStateBufferRole::Residual,
                residual_len,
            ),
            normed: FlashMoeStateBufferDescriptor::cpu(FlashMoeStateBufferRole::Normed, normed_len),
            placement: FlashMoeStatePlacement::CpuVisible,
        }
    }

    pub(crate) fn metal_post_attention_prep(
        layer: usize,
        prep: FlashMoePostAttentionPrepState,
    ) -> Self {
        Self {
            layer,
            residual: FlashMoeStateBufferDescriptor::gpu(prep.residual()),
            normed: FlashMoeStateBufferDescriptor::gpu(prep.normed()),
            placement: FlashMoeStatePlacement::GpuResident,
        }
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn width(self) -> usize {
        self.residual.len()
    }

    pub(crate) fn residual(self) -> FlashMoeStateBufferDescriptor {
        self.residual
    }

    pub(crate) fn normed(self) -> FlashMoeStateBufferDescriptor {
        self.normed
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.residual.is_declared_graph_state()
            && self.normed.is_declared_graph_state()
            && self.residual.role() == FlashMoeStateBufferRole::Residual
            && self.normed.role() == FlashMoeStateBufferRole::Normed
            && self.residual.len() == self.normed.len()
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeCmd3OutputState {
    hidden: FlashMoeGpuBufferDescriptor,
    next_normed: Option<FlashMoeGpuBufferDescriptor>,
}

impl FlashMoeCmd3OutputState {
    pub(crate) fn gpu_resident(width: usize, next_normed: bool) -> Self {
        Self {
            hidden: FlashMoeGpuBufferDescriptor::hidden(width),
            next_normed: next_normed.then(|| FlashMoeGpuBufferDescriptor::next_layer_normed(width)),
        }
    }

    pub(crate) fn width(self) -> usize {
        self.hidden.len()
    }

    pub(crate) fn hidden(self) -> FlashMoeGpuBufferDescriptor {
        self.hidden
    }

    pub(crate) fn next_normed(self) -> Option<FlashMoeGpuBufferDescriptor> {
        self.next_normed
    }

    pub(crate) fn has_next_normed(self) -> bool {
        self.next_normed.is_some()
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.hidden.is_declared_graph_state()
            && self.hidden.role() == FlashMoeStateBufferRole::Hidden
            && self.next_normed.map_or(true, |next_normed| {
                next_normed.is_declared_graph_state()
                    && next_normed.role() == FlashMoeStateBufferRole::NextLayerNormed
                    && next_normed.len() == self.hidden.len()
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeCmd1InputState {
    layer: usize,
    role: FlashMoeStateBufferRole,
    len: usize,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeCmd1InputState {
    pub(crate) fn new(
        layer: usize,
        role: FlashMoeStateBufferRole,
        len: usize,
        placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            layer,
            role,
            len,
            placement,
        }
    }

    pub(crate) fn cpu_normed(layer: usize, len: usize) -> Self {
        Self::new(
            layer,
            FlashMoeStateBufferRole::Normed,
            len,
            FlashMoeStatePlacement::CpuVisible,
        )
    }

    pub(crate) fn gpu_next_layer_normed(
        layer: usize,
        descriptor: FlashMoeGpuBufferDescriptor,
    ) -> Self {
        Self::new(
            layer,
            descriptor.role(),
            descriptor.len(),
            descriptor.placement(),
        )
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        self.role
    }

    pub(crate) fn len(self) -> usize {
        self.len
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.len() > 0
            && matches!(
                (self.role(), self.placement()),
                (
                    FlashMoeStateBufferRole::Normed,
                    FlashMoeStatePlacement::CpuVisible
                ) | (
                    FlashMoeStateBufferRole::NextLayerNormed,
                    FlashMoeStatePlacement::GpuResident
                )
            )
            && FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

impl Deref for FlashMoeCpuBuffer {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl DerefMut for FlashMoeCpuBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeRecurrentState {
    value: u64,
}

impl FlashMoeRecurrentState {
    pub(crate) fn new(value: u64) -> Self {
        Self { value }
    }

    pub(crate) fn mix_active_expert(&mut self, expert_hash: u64, weight: f32) {
        self.value = self
            .value
            .wrapping_add(expert_hash.wrapping_mul((weight.to_bits() as u64).max(1)));
    }

    pub(crate) fn value(self) -> u64 {
        self.value
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeTokenState {
    hidden: FlashMoeCpuBuffer,
    next_layer_normed: Option<FlashMoeCpuBuffer>,
    recurrent: FlashMoeRecurrentState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashMoeExpertPhaseApplication {
    HiddenAndNextNormed,
    HiddenOnly,
}

impl FlashMoeTokenState {
    pub(crate) fn new(hidden_values: Vec<f32>, recurrent_seed: u64) -> Self {
        Self {
            hidden: FlashMoeCpuBuffer::hidden(hidden_values),
            next_layer_normed: None,
            recurrent: FlashMoeRecurrentState::new(recurrent_seed),
        }
    }

    pub(crate) fn hidden(&self) -> &FlashMoeCpuBuffer {
        &self.hidden
    }

    pub(crate) fn hidden_mut(&mut self) -> &mut FlashMoeCpuBuffer {
        &mut self.hidden
    }

    pub(crate) fn replace_hidden(&mut self, values: Vec<f32>) {
        self.hidden.replace_values(values);
    }

    #[cfg(test)]
    pub(crate) fn residual_snapshot(&self) -> FlashMoeCpuBuffer {
        FlashMoeCpuBuffer::residual(self.hidden.clone_values())
    }

    pub(crate) fn set_next_layer_normed(&mut self, values: Option<Vec<f32>>) {
        self.next_layer_normed = values.map(FlashMoeCpuBuffer::next_layer_normed);
    }

    pub(crate) fn clear_next_layer_normed(&mut self) {
        self.next_layer_normed = None;
    }

    pub(crate) fn apply_expert_phase_output(&mut self, output: FlashMoeExpertPhaseOutput) {
        let (hidden, next_normed) = output.into_hidden_and_next_normed();
        self.replace_hidden(hidden);
        self.set_next_layer_normed(next_normed);
    }

    pub(crate) fn apply_expert_phase_hidden_only(&mut self, output: FlashMoeExpertPhaseOutput) {
        let (hidden, _) = output.into_hidden_and_next_normed();
        self.replace_hidden(hidden);
        self.clear_next_layer_normed();
    }

    pub(crate) fn apply_declared_expert_phase(
        &mut self,
        output: FlashMoeExpertPhaseOutput,
        application: FlashMoeExpertPhaseApplication,
    ) -> Result<()> {
        if output.declared_cmd3_output().is_none() {
            bail!("FlashMoe token state refused undeclared expert phase output");
        }
        match application {
            FlashMoeExpertPhaseApplication::HiddenAndNextNormed => {
                self.apply_expert_phase_output(output)
            }
            FlashMoeExpertPhaseApplication::HiddenOnly => {
                self.apply_expert_phase_hidden_only(output)
            }
        }
        Ok(())
    }

    pub(crate) fn take_next_layer_normed_as_normed(&mut self) -> Option<FlashMoeCpuBuffer> {
        self.next_layer_normed
            .take()
            .map(|buffer| buffer.into_role(FlashMoeStateBufferRole::Normed))
    }

    pub(crate) fn mix_active_expert(&mut self, expert_hash: u64, weight: f32) {
        self.recurrent.mix_active_expert(expert_hash, weight);
    }

    pub(crate) fn recurrent_value(&self) -> u64 {
        self.recurrent.value()
    }

    pub(crate) fn layer_state_record(
        &self,
        position: usize,
        layer: usize,
    ) -> FlashMoeLayerStateRecord {
        FlashMoeLayerStateRecord {
            position,
            layer,
            recurrent_value: self.recurrent_value(),
        }
    }

    pub(crate) fn into_hidden_values(self) -> Vec<f32> {
        self.hidden.into_values()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeLayerStateRecord {
    position: usize,
    layer: usize,
    recurrent_value: u64,
}

impl FlashMoeLayerStateRecord {
    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn recurrent_value(self) -> u64 {
        self.recurrent_value
    }

    pub(crate) fn state(self, placement: FlashMoeStatePlacement) -> FlashMoeRecurrentLayerState {
        match placement {
            FlashMoeStatePlacement::CpuVisible => FlashMoeRecurrentLayerState::cpu_visible(
                self.position(),
                self.layer(),
                self.recurrent_value(),
            ),
            FlashMoeStatePlacement::GpuResident => FlashMoeRecurrentLayerState::new(
                self.position(),
                self.layer(),
                self.recurrent_value(),
                placement,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeRecurrentLayerState {
    position: usize,
    layer: usize,
    value: u64,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeRecurrentLayerState {
    pub(crate) fn new(
        position: usize,
        layer: usize,
        value: u64,
        placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            position,
            layer,
            value,
            placement,
        }
    }

    pub(crate) fn cpu_visible(position: usize, layer: usize, value: u64) -> Self {
        Self::new(position, layer, value, FlashMoeStatePlacement::CpuVisible)
    }

    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn value(self) -> u64 {
        self.value
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        FlashMoeStateBufferRole::Recurrent
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeLinearAttentionCacheState {
    layer: usize,
    conv_state_len: usize,
    ssm_state_len: usize,
    conv_output_len: usize,
    output_len: usize,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeLinearAttentionCacheState {
    pub(crate) fn new(
        layer: usize,
        conv_state_len: usize,
        ssm_state_len: usize,
        conv_output_len: usize,
        output_len: usize,
        placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            layer,
            conv_state_len,
            ssm_state_len,
            conv_output_len,
            output_len,
            placement,
        }
    }

    pub(crate) fn cpu_visible(
        layer: usize,
        conv_state_len: usize,
        ssm_state_len: usize,
        conv_output_len: usize,
        output_len: usize,
    ) -> Self {
        Self::new(
            layer,
            conv_state_len,
            ssm_state_len,
            conv_output_len,
            output_len,
            FlashMoeStatePlacement::CpuVisible,
        )
    }

    pub(crate) fn gpu_resident(
        layer: usize,
        conv_state_len: usize,
        ssm_state_len: usize,
        conv_output_len: usize,
        output_len: usize,
    ) -> Self {
        Self::new(
            layer,
            conv_state_len,
            ssm_state_len,
            conv_output_len,
            output_len,
            FlashMoeStatePlacement::GpuResident,
        )
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn conv_state_len(self) -> usize {
        self.conv_state_len
    }

    pub(crate) fn ssm_state_len(self) -> usize {
        self.ssm_state_len
    }

    pub(crate) fn conv_output_len(self) -> usize {
        self.conv_output_len
    }

    pub(crate) fn output_len(self) -> usize {
        self.output_len
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        FlashMoeStateBufferRole::Recurrent
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.conv_state_len() > 0
            && self.ssm_state_len() > 0
            && self.conv_output_len() > 0
            && self.output_len() > 0
            && FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeLinearAttentionCacheShape {
    pub(crate) conv_state_len: usize,
    pub(crate) ssm_state_len: usize,
    pub(crate) conv_output_len: usize,
    pub(crate) output_len: usize,
    pub(crate) value_scratch_len: usize,
}

impl FlashMoeLinearAttentionCacheShape {
    pub(crate) fn new(
        conv_state_len: usize,
        ssm_state_len: usize,
        conv_output_len: usize,
        output_len: usize,
        value_scratch_len: usize,
    ) -> Self {
        Self {
            conv_state_len,
            ssm_state_len,
            conv_output_len,
            output_len,
            value_scratch_len,
        }
    }

    pub(crate) fn state(
        self,
        layer: usize,
        placement: FlashMoeStatePlacement,
    ) -> FlashMoeLinearAttentionCacheState {
        FlashMoeLinearAttentionCacheState::new(
            layer,
            self.conv_state_len,
            self.ssm_state_len,
            self.conv_output_len,
            self.output_len,
            placement,
        )
    }

    pub(crate) fn is_declared_graph_shape(self) -> bool {
        self.conv_state_len > 0
            && self.ssm_state_len > 0
            && self.conv_output_len > 0
            && self.output_len > 0
            && self.value_scratch_len > 0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FlashMoeLinearAttentionStateData {
    pub(crate) conv_state: Vec<f32>,
    pub(crate) ssm_state: Vec<f32>,
    pub(crate) conv_out: Vec<f32>,
    pub(crate) out_values: Vec<f32>,
    pub(crate) kv_mem: Vec<f32>,
    pub(crate) delta: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct FlashMoeLinearAttentionState {
    inner: Arc<FlashMoeLinearAttentionStateData>,
}

impl FlashMoeLinearAttentionState {
    pub(crate) fn new(shape: FlashMoeLinearAttentionCacheShape) -> Self {
        debug_assert!(shape.is_declared_graph_shape());
        Self {
            inner: Arc::new(FlashMoeLinearAttentionStateData {
                conv_state: vec![0.0; shape.conv_state_len],
                ssm_state: vec![0.0; shape.ssm_state_len],
                conv_out: vec![0.0; shape.conv_output_len],
                out_values: vec![0.0; shape.output_len],
                kv_mem: vec![0.0; shape.value_scratch_len],
                delta: vec![0.0; shape.value_scratch_len],
            }),
        }
    }

    pub(crate) fn expected_state(
        layer: usize,
        shape: FlashMoeLinearAttentionCacheShape,
        placement: FlashMoeStatePlacement,
    ) -> FlashMoeLinearAttentionCacheState {
        match placement {
            FlashMoeStatePlacement::CpuVisible => FlashMoeLinearAttentionCacheState::cpu_visible(
                layer,
                shape.conv_state_len,
                shape.ssm_state_len,
                shape.conv_output_len,
                shape.output_len,
            ),
            FlashMoeStatePlacement::GpuResident => FlashMoeLinearAttentionCacheState::gpu_resident(
                layer,
                shape.conv_state_len,
                shape.ssm_state_len,
                shape.conv_output_len,
                shape.output_len,
            ),
        }
    }

    pub(crate) fn state(
        &self,
        layer: usize,
        placement: FlashMoeStatePlacement,
    ) -> FlashMoeLinearAttentionCacheState {
        FlashMoeLinearAttentionCacheShape::new(
            self.conv_state.len(),
            self.ssm_state.len(),
            self.conv_out.len(),
            self.out_values.len(),
            self.kv_mem.len(),
        )
        .state(layer, placement)
    }

    pub(crate) fn matches_shape(&self, shape: FlashMoeLinearAttentionCacheShape) -> bool {
        self.conv_state.len() == shape.conv_state_len
            && self.ssm_state.len() == shape.ssm_state_len
            && self.conv_out.len() == shape.conv_output_len
            && self.out_values.len() == shape.output_len
            && self.kv_mem.len() == shape.value_scratch_len
            && self.delta.len() == shape.value_scratch_len
    }

    #[cfg(test)]
    pub(crate) fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Deref for FlashMoeLinearAttentionState {
    type Target = FlashMoeLinearAttentionStateData;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for FlashMoeLinearAttentionState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.inner)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeExpertPhaseOutput {
    hidden: Vec<f32>,
    next_normed: Option<Vec<f32>>,
    declared_cmd3_output: Option<FlashMoeCmd3OutputState>,
}

impl FlashMoeExpertPhaseOutput {
    pub(crate) fn new(hidden: Vec<f32>, next_normed: Option<Vec<f32>>) -> Self {
        Self {
            hidden,
            next_normed,
            declared_cmd3_output: None,
        }
    }

    pub(crate) fn hidden_len(&self) -> usize {
        self.hidden.len()
    }

    pub(crate) fn next_normed_len(&self) -> Option<usize> {
        self.next_normed.as_ref().map(Vec::len)
    }

    pub(crate) fn with_declared_cmd3_output(mut self, state: FlashMoeCmd3OutputState) -> Self {
        self.declared_cmd3_output = Some(state);
        self
    }

    pub(crate) fn declared_cmd3_output(&self) -> Option<FlashMoeCmd3OutputState> {
        self.declared_cmd3_output
    }

    pub(crate) fn into_hidden_and_next_normed(self) -> (Vec<f32>, Option<Vec<f32>>) {
        (self.hidden, self.next_normed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeGeneratedTokenRecord {
    position: usize,
    token: u32,
}

impl FlashMoeGeneratedTokenRecord {
    pub(crate) fn new(position: usize, token: u32) -> Self {
        Self { position, token }
    }

    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn token(self) -> u32 {
        self.token
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoePromptTokenRecord {
    position: usize,
    token: u32,
}

impl FlashMoePromptTokenRecord {
    pub(crate) fn new(position: usize, token: u32) -> Self {
        Self { position, token }
    }

    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn token(self) -> u32 {
        self.token
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeFullAttentionKvRecord {
    position: usize,
    layer: usize,
    key: Vec<f32>,
    value: Vec<f32>,
}

impl FlashMoeFullAttentionKvRecord {
    pub(crate) fn new(position: usize, layer: usize, key: Vec<f32>, value: Vec<f32>) -> Self {
        Self {
            position,
            layer,
            key,
            value,
        }
    }

    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn layer(&self) -> usize {
        self.layer
    }

    #[cfg(test)]
    pub(crate) fn key(&self) -> &[f32] {
        &self.key
    }

    #[cfg(test)]
    pub(crate) fn value(&self) -> &[f32] {
        &self.value
    }

    pub(crate) fn state(&self, placement: FlashMoeStatePlacement) -> FlashMoeFullAttentionKvState {
        match placement {
            FlashMoeStatePlacement::CpuVisible => FlashMoeFullAttentionKvState::cpu_visible(
                self.position,
                self.layer,
                self.key.len(),
                self.value.len(),
            ),
            FlashMoeStatePlacement::GpuResident => FlashMoeFullAttentionKvState::gpu_resident(
                self.position,
                self.layer,
                self.key.len(),
                self.value.len(),
            ),
        }
    }

    pub(crate) fn into_key_value(self) -> (Vec<f32>, Vec<f32>) {
        (self.key, self.value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeFullAttentionKvState {
    position: usize,
    layer: usize,
    key_len: usize,
    value_len: usize,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeFullAttentionKvState {
    pub(crate) fn new(
        position: usize,
        layer: usize,
        key_len: usize,
        value_len: usize,
        placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            position,
            layer,
            key_len,
            value_len,
            placement,
        }
    }

    pub(crate) fn cpu_visible(
        position: usize,
        layer: usize,
        key_len: usize,
        value_len: usize,
    ) -> Self {
        Self::new(
            position,
            layer,
            key_len,
            value_len,
            FlashMoeStatePlacement::CpuVisible,
        )
    }

    pub(crate) fn gpu_resident(
        position: usize,
        layer: usize,
        key_len: usize,
        value_len: usize,
    ) -> Self {
        Self::new(
            position,
            layer,
            key_len,
            value_len,
            FlashMoeStatePlacement::GpuResident,
        )
    }

    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn width(self) -> usize {
        self.key_len
    }

    pub(crate) fn key_len(self) -> usize {
        self.key_len
    }

    pub(crate) fn value_len(self) -> usize {
        self.value_len
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        FlashMoeStateBufferRole::Kv
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.width() > 0
            && self.key_len() == self.value_len()
            && FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reusable_session_prefix_requires_complete_cached_prefix() {
        assert_eq!(
            reusable_session_prefix_len(&[1, 2, 3], &[1, 2, 3, 4]),
            Some(3)
        );
        assert_eq!(reusable_session_prefix_len(&[1, 2, 3], &[1, 2, 9]), None);
        assert_eq!(reusable_session_prefix_len(&[1, 2, 3], &[1, 2]), None);
    }

    #[test]
    fn reusable_session_entry_moves_state_only_for_matching_prefix() {
        let mut sessions = BTreeMap::new();
        sessions.insert(
            "chat".to_string(),
            FlashMoeSessionState::new(vec![10, 20], "kv-state", vec![1.0, 2.0]),
        );

        assert!(take_reusable_session_cache_entry(&mut sessions, "chat", &[10, 99, 30]).is_none());
        assert!(sessions.contains_key("chat"));

        let (prefix_len, state) =
            take_reusable_session_cache_entry(&mut sessions, "chat", &[10, 20, 30]).unwrap();
        assert_eq!(prefix_len, 2);
        assert_eq!(state.kv_cache, "kv-state");
        assert_eq!(state.last_hidden, vec![1.0, 2.0]);
        assert!(sessions.is_empty());
    }

    #[test]
    fn stable_session_cache_tokens_keep_prompt_only() {
        assert_eq!(stable_session_cache_tokens(&[4, 5, 6]), vec![4, 5, 6]);
    }

    #[test]
    fn cpu_hidden_buffer_declares_role_and_cpu_visibility() {
        let mut hidden = FlashMoeCpuBuffer::hidden(vec![1.0, 2.0]);
        assert_eq!(hidden.role(), FlashMoeStateBufferRole::Hidden);
        assert_eq!(hidden.placement(), FlashMoeStatePlacement::CpuVisible);
        assert_eq!(&hidden[..], &[1.0, 2.0]);

        hidden[1] = 3.0;
        assert_eq!(hidden.clone_values(), vec![1.0, 3.0]);
        assert_eq!(hidden.into_values(), vec![1.0, 3.0]);
    }

    #[test]
    fn cpu_buffers_cover_normed_residual_and_next_layer_transitions() {
        let residual = FlashMoeCpuBuffer::residual(vec![0.5, -1.0]);
        assert_eq!(residual.role(), FlashMoeStateBufferRole::Residual);
        assert!(residual.is_declared_graph_state());

        let next_normed = FlashMoeCpuBuffer::next_layer_normed(vec![2.0, 4.0]);
        assert_eq!(next_normed.role(), FlashMoeStateBufferRole::NextLayerNormed);
        let normed = next_normed.into_role(FlashMoeStateBufferRole::Normed);
        assert_eq!(normed.role(), FlashMoeStateBufferRole::Normed);
        assert_eq!(&normed[..], &[2.0, 4.0]);
    }

    #[test]
    fn gpu_buffer_descriptor_declares_role_length_and_residency() {
        let hidden = FlashMoeGpuBufferDescriptor::hidden(4096);
        assert_eq!(hidden.role(), FlashMoeStateBufferRole::Hidden);
        assert_eq!(hidden.len(), 4096);
        assert_eq!(hidden.placement(), FlashMoeStatePlacement::GpuResident);
        assert!(hidden.is_declared_graph_state());

        let next_normed = FlashMoeGpuBufferDescriptor::next_layer_normed(4096);
        assert_eq!(next_normed.role(), FlashMoeStateBufferRole::NextLayerNormed);
        assert_eq!(next_normed.placement(), FlashMoeStatePlacement::GpuResident);
    }

    #[test]
    fn gpu_buffer_descriptor_rejects_zero_length_graph_state() {
        let hidden = FlashMoeGpuBufferDescriptor::hidden(0);

        assert_eq!(hidden.len(), 0);
        assert!(!hidden.is_declared_graph_state());
    }

    #[test]
    fn post_attention_prep_state_names_gpu_residual_normed_and_routes() {
        let state = FlashMoePostAttentionPrepState::new(12, 4096, 128, 4);

        assert_eq!(state.width(), 4096);
        assert_eq!(state.active_experts(), 4);
        assert_eq!(state.routing().layer(), 12);
        assert_eq!(state.routing().experts(), 128);
        assert_eq!(
            state.routing().source(),
            FlashMoeRoutingOutputSource::FusedMetalPostAttentionPrepCpuTopK
        );
        assert_eq!(state.residual().role(), FlashMoeStateBufferRole::Residual);
        assert_eq!(state.normed().role(), FlashMoeStateBufferRole::Normed);
        assert_eq!(
            state.residual().placement(),
            FlashMoeStatePlacement::GpuResident
        );
        assert!(state.is_declared_graph_state());
    }

    #[test]
    fn post_attention_prep_state_rejects_empty_width_or_routes() {
        assert!(!FlashMoePostAttentionPrepState::new(12, 0, 128, 4).is_declared_graph_state());
        assert!(!FlashMoePostAttentionPrepState::new(12, 4096, 128, 0).is_declared_graph_state());
        assert!(!FlashMoePostAttentionPrepState::new(12, 4096, 2, 4).is_declared_graph_state());
    }

    #[test]
    fn routing_output_state_declares_cpu_visible_scores_and_topk() {
        let scores = FlashMoeRoutingOutputState::cpu_router_scores(2, 8, 4);
        assert_eq!(scores.role(), FlashMoeStateBufferRole::RouterScores);
        assert_eq!(scores.len(), 8);
        assert_eq!(scores.placement(), FlashMoeStatePlacement::CpuVisible);
        assert!(scores.is_declared_graph_state());

        let topk = FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(2, 8, 4);
        assert_eq!(topk.role(), FlashMoeStateBufferRole::RoutingTopK);
        assert_eq!(topk.len(), 4);
        assert_eq!(topk.active_experts(), 4);
        assert!(topk.is_declared_graph_state());
    }

    #[test]
    fn routing_output_state_rejects_empty_or_oversized_active_routes() {
        assert!(!FlashMoeRoutingOutputState::cpu_router_scores(2, 0, 0).is_declared_graph_state());
        assert!(
            !FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(2, 2, 4)
                .is_declared_graph_state()
        );
    }

    #[test]
    fn cmd2_input_state_declares_attention_and_residual_placements() {
        let cpu = FlashMoeCmd2InputState::new(
            3,
            2048,
            FlashMoeStatePlacement::CpuVisible,
            4096,
            FlashMoeStatePlacement::CpuVisible,
        );
        assert_eq!(cpu.layer(), 3);
        assert_eq!(
            cpu.attention().role(),
            FlashMoeStateBufferRole::AttentionValues
        );
        assert_eq!(cpu.residual().role(), FlashMoeStateBufferRole::Residual);
        assert_eq!(cpu.attention().len(), 2048);
        assert_eq!(cpu.residual().len(), 4096);
        assert_eq!(
            cpu.attention().placement(),
            FlashMoeStatePlacement::CpuVisible
        );
        assert!(cpu.is_declared_graph_state());

        let gpu = FlashMoeCmd2InputState::new(
            3,
            2048,
            FlashMoeStatePlacement::GpuResident,
            4096,
            FlashMoeStatePlacement::GpuResident,
        );
        assert_eq!(
            gpu.attention().placement(),
            FlashMoeStatePlacement::GpuResident
        );
        assert_eq!(
            gpu.residual().placement(),
            FlashMoeStatePlacement::GpuResident
        );
        assert!(gpu.is_declared_graph_state());
    }

    #[test]
    fn cmd2_input_state_rejects_empty_attention_or_residual() {
        assert!(
            !FlashMoeCmd2InputState::new(
                3,
                0,
                FlashMoeStatePlacement::CpuVisible,
                4096,
                FlashMoeStatePlacement::CpuVisible,
            )
            .is_declared_graph_state()
        );
        assert!(
            !FlashMoeCmd2InputState::new(
                3,
                2048,
                FlashMoeStatePlacement::CpuVisible,
                0,
                FlashMoeStatePlacement::CpuVisible,
            )
            .is_declared_graph_state()
        );
    }

    #[test]
    fn cmd3_output_state_declares_gpu_hidden_and_optional_next_normed() {
        let output = FlashMoeCmd3OutputState::gpu_resident(4096, true);

        assert_eq!(output.width(), 4096);
        assert_eq!(output.hidden().role(), FlashMoeStateBufferRole::Hidden);
        assert_eq!(
            output.hidden().placement(),
            FlashMoeStatePlacement::GpuResident
        );
        let next_normed = output.next_normed().unwrap();
        assert_eq!(next_normed.role(), FlashMoeStateBufferRole::NextLayerNormed);
        assert_eq!(next_normed.len(), 4096);
        assert!(output.has_next_normed());
        assert!(output.is_declared_graph_state());

        let hidden_only = FlashMoeCmd3OutputState::gpu_resident(4096, false);
        assert!(!hidden_only.has_next_normed());
        assert!(hidden_only.next_normed().is_none());
        assert!(hidden_only.is_declared_graph_state());
    }

    #[test]
    fn cmd3_output_state_rejects_zero_width() {
        let output = FlashMoeCmd3OutputState::gpu_resident(0, true);

        assert_eq!(output.width(), 0);
        assert!(!output.is_declared_graph_state());
    }

    #[test]
    fn cmd3_input_state_declares_cpu_or_gpu_normed_residual_pair() {
        let cpu = FlashMoeCmd3InputState::cpu_normed_residual(5, 4096, 4096);
        assert_eq!(cpu.layer(), 5);
        assert_eq!(cpu.width(), 4096);
        assert_eq!(cpu.residual().role(), FlashMoeStateBufferRole::Residual);
        assert_eq!(cpu.normed().role(), FlashMoeStateBufferRole::Normed);
        assert_eq!(cpu.placement(), FlashMoeStatePlacement::CpuVisible);
        assert!(cpu.is_declared_graph_state());

        let prep = FlashMoePostAttentionPrepState::new(5, 4096, 128, 4);
        let gpu = FlashMoeCmd3InputState::metal_post_attention_prep(5, prep);
        assert_eq!(gpu.width(), 4096);
        assert_eq!(
            gpu.residual().placement(),
            FlashMoeStatePlacement::GpuResident
        );
        assert_eq!(
            gpu.normed().placement(),
            FlashMoeStatePlacement::GpuResident
        );
        assert_eq!(gpu.placement(), FlashMoeStatePlacement::GpuResident);
        assert!(gpu.is_declared_graph_state());
    }

    #[test]
    fn cmd3_input_state_rejects_zero_or_mismatched_buffers() {
        assert!(!FlashMoeCmd3InputState::cpu_normed_residual(5, 0, 0).is_declared_graph_state());
        assert!(
            !FlashMoeCmd3InputState::cpu_normed_residual(5, 4096, 2048).is_declared_graph_state()
        );
        let prep = FlashMoePostAttentionPrepState::new(5, 0, 128, 4);
        assert!(
            !FlashMoeCmd3InputState::metal_post_attention_prep(5, prep).is_declared_graph_state()
        );
    }

    #[test]
    fn cmd1_input_state_declares_cpu_normed_or_gpu_next_normed() {
        let cpu = FlashMoeCmd1InputState::cpu_normed(12, 4096);
        assert_eq!(cpu.layer(), 12);
        assert_eq!(cpu.role(), FlashMoeStateBufferRole::Normed);
        assert_eq!(cpu.len(), 4096);
        assert_eq!(cpu.placement(), FlashMoeStatePlacement::CpuVisible);
        assert!(cpu.is_declared_graph_state());

        let gpu = FlashMoeCmd1InputState::gpu_next_layer_normed(
            12,
            FlashMoeGpuBufferDescriptor::next_layer_normed(4096),
        );
        assert_eq!(gpu.layer(), 12);
        assert_eq!(gpu.role(), FlashMoeStateBufferRole::NextLayerNormed);
        assert_eq!(gpu.len(), 4096);
        assert_eq!(gpu.placement(), FlashMoeStatePlacement::GpuResident);
        assert!(gpu.is_declared_graph_state());

        assert!(!FlashMoeCmd1InputState::cpu_normed(12, 0).is_declared_graph_state());
        assert!(
            !FlashMoeCmd1InputState::gpu_next_layer_normed(
                12,
                FlashMoeGpuBufferDescriptor::hidden(4096),
            )
            .is_declared_graph_state()
        );
    }

    #[test]
    fn recurrent_state_mixes_active_experts_with_wrapping_math() {
        let mut state = FlashMoeRecurrentState::new(u64::MAX - 4);
        state.mix_active_expert(3, 1.0);
        let expected = (u64::MAX - 4).wrapping_add(3_u64.wrapping_mul(1.0f32.to_bits() as u64));
        assert_eq!(state.value(), expected);
    }

    #[test]
    fn recurrent_state_uses_nonzero_multiplier_for_zero_weight_bits() {
        let mut state = FlashMoeRecurrentState::new(10);
        state.mix_active_expert(7, 0.0);
        assert_eq!(state.value(), 17);
    }

    #[test]
    fn token_state_owns_hidden_next_normed_and_recurrent_values() {
        let mut state = FlashMoeTokenState::new(vec![1.0, 2.0], 10);
        assert_eq!(state.hidden().role(), FlashMoeStateBufferRole::Hidden);
        assert_eq!(&state.hidden()[..], &[1.0, 2.0]);

        let residual = state.residual_snapshot();
        assert_eq!(residual.role(), FlashMoeStateBufferRole::Residual);
        assert_eq!(&residual[..], &[1.0, 2.0]);

        state.set_next_layer_normed(Some(vec![3.0, 4.0]));
        let normed = state.take_next_layer_normed_as_normed().unwrap();
        assert_eq!(normed.role(), FlashMoeStateBufferRole::Normed);
        assert_eq!(&normed[..], &[3.0, 4.0]);
        assert!(state.take_next_layer_normed_as_normed().is_none());

        state.apply_expert_phase_output(FlashMoeExpertPhaseOutput::new(
            vec![8.0, 9.0],
            Some(vec![10.0, 11.0]),
        ));
        assert_eq!(&state.hidden()[..], &[8.0, 9.0]);
        let normed = state.take_next_layer_normed_as_normed().unwrap();
        assert_eq!(normed.role(), FlashMoeStateBufferRole::Normed);
        assert_eq!(&normed[..], &[10.0, 11.0]);

        state.apply_expert_phase_hidden_only(FlashMoeExpertPhaseOutput::new(
            vec![12.0],
            Some(vec![13.0]),
        ));
        assert_eq!(&state.hidden()[..], &[12.0]);
        assert!(state.take_next_layer_normed_as_normed().is_none());

        state.mix_active_expert(7, 0.0);
        assert_eq!(state.recurrent_value(), 17);
        assert_eq!(
            state.layer_state_record(5, 2),
            FlashMoeLayerStateRecord {
                position: 5,
                layer: 2,
                recurrent_value: 17
            }
        );
        state.replace_hidden(vec![5.0]);
        assert_eq!(state.into_hidden_values(), vec![5.0]);
    }

    #[test]
    fn token_state_requires_declared_expert_output_for_scheduled_application() {
        let mut state = FlashMoeTokenState::new(vec![1.0, 2.0], 10);
        let raw_err = state
            .apply_declared_expert_phase(
                FlashMoeExpertPhaseOutput::new(vec![3.0, 4.0], Some(vec![5.0, 6.0])),
                FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
            )
            .unwrap_err();
        assert!(
            raw_err
                .to_string()
                .contains("refused undeclared expert phase output"),
            "{raw_err:#}"
        );

        let declared = FlashMoeExpertPhaseOutput::new(vec![3.0, 4.0], Some(vec![5.0, 6.0]))
            .with_declared_cmd3_output(FlashMoeCmd3OutputState::gpu_resident(2, true));
        state
            .apply_declared_expert_phase(
                declared,
                FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
            )
            .unwrap();
        assert_eq!(&state.hidden()[..], &[3.0, 4.0]);
        let normed = state.take_next_layer_normed_as_normed().unwrap();
        assert_eq!(&normed[..], &[5.0, 6.0]);

        let declared_hidden_only =
            FlashMoeExpertPhaseOutput::new(vec![7.0, 8.0], Some(vec![9.0, 10.0]))
                .with_declared_cmd3_output(FlashMoeCmd3OutputState::gpu_resident(2, true));
        state
            .apply_declared_expert_phase(
                declared_hidden_only,
                FlashMoeExpertPhaseApplication::HiddenOnly,
            )
            .unwrap();
        assert_eq!(&state.hidden()[..], &[7.0, 8.0]);
        assert!(state.take_next_layer_normed_as_normed().is_none());
    }

    #[test]
    fn recurrent_layer_state_declares_cpu_visible_layer_transition() {
        let record = FlashMoeLayerStateRecord {
            position: 5,
            layer: 2,
            recurrent_value: 17,
        };
        let state = record.state(FlashMoeStatePlacement::CpuVisible);

        assert_eq!(state, FlashMoeRecurrentLayerState::cpu_visible(5, 2, 17));
        assert_eq!(state.position(), 5);
        assert_eq!(state.layer(), 2);
        assert_eq!(state.value(), 17);
        assert_eq!(state.role(), FlashMoeStateBufferRole::Recurrent);
        assert_eq!(state.placement(), FlashMoeStatePlacement::CpuVisible);
        assert!(state.is_declared_graph_state());
    }

    #[test]
    fn linear_attention_cache_state_declares_lengths_and_placement() {
        let cpu = FlashMoeLinearAttentionCacheState::cpu_visible(3, 8, 16, 4, 6);
        assert_eq!(cpu.layer(), 3);
        assert_eq!(cpu.conv_state_len(), 8);
        assert_eq!(cpu.ssm_state_len(), 16);
        assert_eq!(cpu.conv_output_len(), 4);
        assert_eq!(cpu.output_len(), 6);
        assert_eq!(cpu.role(), FlashMoeStateBufferRole::Recurrent);
        assert_eq!(cpu.placement(), FlashMoeStatePlacement::CpuVisible);
        assert!(cpu.is_declared_graph_state());

        let gpu = FlashMoeLinearAttentionCacheState::gpu_resident(3, 8, 16, 4, 6);
        assert_eq!(gpu.placement(), FlashMoeStatePlacement::GpuResident);
        assert!(gpu.is_declared_graph_state());

        assert!(
            !FlashMoeLinearAttentionCacheState::cpu_visible(3, 0, 16, 4, 6)
                .is_declared_graph_state()
        );
        assert!(
            !FlashMoeLinearAttentionCacheState::gpu_resident(3, 8, 0, 4, 6)
                .is_declared_graph_state()
        );
        assert!(
            !FlashMoeLinearAttentionCacheState::cpu_visible(3, 8, 16, 0, 6)
                .is_declared_graph_state()
        );
        assert!(
            !FlashMoeLinearAttentionCacheState::gpu_resident(3, 8, 16, 4, 0)
                .is_declared_graph_state()
        );
    }

    #[test]
    fn linear_attention_state_owns_recurrent_buffers_for_declared_shape() {
        let shape = FlashMoeLinearAttentionCacheShape::new(8, 16, 4, 6, 2);
        assert!(shape.is_declared_graph_shape());

        let mut state = FlashMoeLinearAttentionState::new(shape);
        assert_eq!(state.conv_state.len(), 8);
        assert_eq!(state.ssm_state.len(), 16);
        assert_eq!(state.conv_out.len(), 4);
        assert_eq!(state.out_values.len(), 6);
        assert_eq!(state.kv_mem.len(), 2);
        assert_eq!(state.delta.len(), 2);
        assert!(state.matches_shape(shape));
        assert!(!state.matches_shape(FlashMoeLinearAttentionCacheShape::new(8, 16, 4, 6, 3)));

        state.conv_state[0] = 1.5;
        assert_eq!(state.conv_state[0], 1.5);
        assert_eq!(
            FlashMoeLinearAttentionState::expected_state(
                3,
                shape,
                FlashMoeStatePlacement::CpuVisible
            ),
            FlashMoeLinearAttentionCacheState::cpu_visible(3, 8, 16, 4, 6)
        );
        assert_eq!(
            state.state(3, FlashMoeStatePlacement::GpuResident),
            FlashMoeLinearAttentionCacheState::gpu_resident(3, 8, 16, 4, 6)
        );
    }

    #[test]
    fn expert_phase_output_owns_hidden_and_optional_next_normed_transition() {
        let output = FlashMoeExpertPhaseOutput::new(vec![1.0, 2.0], Some(vec![0.5, 1.5]));

        let (hidden, next_normed) = output.into_hidden_and_next_normed();

        assert_eq!(hidden, vec![1.0, 2.0]);
        assert_eq!(next_normed, Some(vec![0.5, 1.5]));
    }

    #[test]
    fn generated_token_record_names_position_and_token() {
        let record = FlashMoeGeneratedTokenRecord::new(12, 99);
        assert_eq!(record.position(), 12);
        assert_eq!(record.token(), 99);
    }

    #[test]
    fn prompt_token_record_names_position_and_token() {
        let record = FlashMoePromptTokenRecord::new(3, 42);
        assert_eq!(record.position(), 3);
        assert_eq!(record.token(), 42);
    }

    #[test]
    fn full_attention_kv_record_owns_position_layer_and_values() {
        let record = FlashMoeFullAttentionKvRecord::new(5, 2, vec![1.0, 1.5], vec![2.0, 2.5]);
        assert_eq!(record.position(), 5);
        assert_eq!(record.layer(), 2);
        assert_eq!(record.key(), &[1.0, 1.5]);
        assert_eq!(record.value(), &[2.0, 2.5]);
        assert_eq!(
            record.state(FlashMoeStatePlacement::CpuVisible),
            FlashMoeFullAttentionKvState::cpu_visible(5, 2, 2, 2)
        );
        let (key, value) = record.into_key_value();
        assert_eq!(key, vec![1.0, 1.5]);
        assert_eq!(value, vec![2.0, 2.5]);
    }

    #[test]
    fn full_attention_kv_state_declares_placement_width_and_role() {
        let cpu = FlashMoeFullAttentionKvState::cpu_visible(7, 3, 4, 4);
        assert_eq!(cpu.position(), 7);
        assert_eq!(cpu.layer(), 3);
        assert_eq!(cpu.width(), 4);
        assert_eq!(cpu.key_len(), 4);
        assert_eq!(cpu.value_len(), 4);
        assert_eq!(cpu.role(), FlashMoeStateBufferRole::Kv);
        assert_eq!(cpu.placement(), FlashMoeStatePlacement::CpuVisible);
        assert!(cpu.is_declared_graph_state());

        let gpu = FlashMoeFullAttentionKvState::gpu_resident(7, 3, 4, 4);
        assert_eq!(gpu.placement(), FlashMoeStatePlacement::GpuResident);
        assert!(gpu.is_declared_graph_state());

        assert!(!FlashMoeFullAttentionKvState::cpu_visible(7, 3, 0, 0).is_declared_graph_state());
        assert!(!FlashMoeFullAttentionKvState::gpu_resident(7, 3, 4, 5).is_declared_graph_state());
    }
}
