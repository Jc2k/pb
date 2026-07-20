use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::Instant;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use super::math::causal_attention;
use super::session_cache::FlashMoeDiskCache;
use super::types::PromptCacheSource;
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

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

#[cfg(test)]
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

    pub(crate) fn from_recurrent_value(hidden_values: Vec<f32>, recurrent_value: u64) -> Self {
        Self {
            hidden: FlashMoeCpuBuffer::hidden(hidden_values),
            next_layer_normed: None,
            recurrent: FlashMoeRecurrentState::new(recurrent_value),
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

    #[cfg(test)]
    pub(crate) fn into_hidden_values(self) -> Vec<f32> {
        self.hidden.into_values()
    }

    pub(crate) fn into_hidden_and_recurrent(self) -> (Vec<f32>, u64) {
        (self.hidden.into_values(), self.recurrent.value())
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

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeLinearAttentionCacheShape {
    pub(crate) conv_state_len: usize,
    pub(crate) ssm_state_len: usize,
    pub(crate) conv_output_len: usize,
    pub(crate) output_len: usize,
    pub(crate) value_scratch_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LinearAttentionLayout {
    pub(crate) num_value_heads: usize,
    pub(crate) num_key_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) value_dim: usize,
    pub(crate) total_key_width: usize,
    pub(crate) total_value_width: usize,
    pub(crate) conv_dim: usize,
    pub(crate) conv_kernel_size: usize,
}

impl LinearAttentionLayout {
    pub(crate) fn conv_state_len(self) -> usize {
        self.conv_kernel_size.saturating_sub(1) * self.conv_dim
    }

    pub(crate) fn ssm_state_len(self) -> usize {
        self.num_value_heads * self.value_dim * self.key_dim
    }

    pub(crate) fn value_heads_per_key_head(self) -> usize {
        (self.num_value_heads / self.num_key_heads).max(1)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeLinearAttentionLayerSnapshot {
    state: FlashMoeLinearAttentionCacheState,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

impl FlashMoeLinearAttentionLayerSnapshot {
    pub(crate) fn new(
        layer: usize,
        conv_state: Vec<f32>,
        ssm_state: Vec<f32>,
        conv_output_len: usize,
        output_len: usize,
    ) -> Result<Self> {
        let state = FlashMoeLinearAttentionCacheState::cpu_visible(
            layer,
            conv_state.len(),
            ssm_state.len(),
            conv_output_len,
            output_len,
        );
        if !state.is_declared_graph_state() {
            bail!(
                "FlashMoe linear-attention session snapshot for layer {layer} is not declared CPU-visible graph state"
            );
        }
        Ok(Self {
            state,
            conv_state,
            ssm_state,
        })
    }

    pub(crate) fn state(&self) -> FlashMoeLinearAttentionCacheState {
        self.state
    }

    pub(crate) fn conv_state(&self) -> &[f32] {
        &self.conv_state
    }

    pub(crate) fn ssm_state(&self) -> &[f32] {
        &self.ssm_state
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeLinearAttentionSessionSnapshot {
    layers: Box<[Option<FlashMoeLinearAttentionLayerSnapshot>]>,
}

impl FlashMoeLinearAttentionSessionSnapshot {
    pub(crate) fn new(layers: Vec<Option<FlashMoeLinearAttentionLayerSnapshot>>) -> Result<Self> {
        if layers.is_empty() {
            bail!("FlashMoe linear-attention session snapshot requires resolved model layers");
        }
        for (layer, snapshot) in layers.iter().enumerate() {
            if let Some(snapshot) = snapshot
                && (snapshot.state().layer() != layer
                    || snapshot.state().placement() != FlashMoeStatePlacement::CpuVisible
                    || !snapshot.state().is_declared_graph_state())
            {
                bail!(
                    "FlashMoe linear-attention session snapshot layer {layer} does not match its declared CPU-visible state"
                );
            }
        }
        Ok(Self {
            layers: layers.into_boxed_slice(),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn layer(&self, layer: usize) -> Option<&FlashMoeLinearAttentionLayerSnapshot> {
        self.layers.get(layer).and_then(Option::as_ref)
    }

    pub(crate) fn state_sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"pb.flashmoe.linear-attention-state.v1\0");
        digest.update((self.layers.len() as u64).to_le_bytes());
        for (layer, snapshot) in self.layers.iter().enumerate() {
            digest.update((layer as u64).to_le_bytes());
            match snapshot {
                Some(snapshot) => {
                    digest.update([1]);
                    let state = snapshot.state();
                    digest.update((state.layer() as u64).to_le_bytes());
                    digest.update((state.conv_state_len() as u64).to_le_bytes());
                    digest.update((state.ssm_state_len() as u64).to_le_bytes());
                    update_f32_digest(&mut digest, snapshot.conv_state());
                    update_f32_digest(&mut digest, snapshot.ssm_state());
                }
                None => digest.update([0]),
            }
        }
        format!("{:x}", digest.finalize())
    }

    pub(crate) fn layer_state_sha256(&self) -> Vec<Option<String>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer, snapshot)| {
                snapshot.as_ref().map(|snapshot| {
                    let mut digest = Sha256::new();
                    digest.update(b"pb.flashmoe.linear-attention-layer.v1\0");
                    digest.update((layer as u64).to_le_bytes());
                    update_f32_digest(&mut digest, snapshot.conv_state());
                    update_f32_digest(&mut digest, snapshot.ssm_state());
                    format!("{:x}", digest.finalize())
                })
            })
            .collect()
    }
}

fn update_f32_digest(digest: &mut Sha256, values: &[f32]) {
    digest.update((values.len() as u64).to_le_bytes());
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
}

#[cfg(test)]
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

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct FlashMoeLinearAttentionStateData {
    pub(crate) conv_state: Vec<f32>,
    pub(crate) ssm_state: Vec<f32>,
    pub(crate) conv_out: Vec<f32>,
    pub(crate) out_values: Vec<f32>,
    pub(crate) kv_mem: Vec<f32>,
    pub(crate) delta: Vec<f32>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct FlashMoeLinearAttentionState {
    inner: Arc<FlashMoeLinearAttentionStateData>,
}

#[cfg(test)]
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
}

#[cfg(test)]
impl Deref for FlashMoeLinearAttentionState {
    type Target = FlashMoeLinearAttentionStateData;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(test)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeMlaKvState {
    position: usize,
    layer: usize,
    latent_len: usize,
    rotary_len: usize,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeMlaKvState {
    pub(crate) fn cpu_visible(
        position: usize,
        layer: usize,
        latent_len: usize,
        rotary_len: usize,
    ) -> Self {
        Self {
            position,
            layer,
            latent_len,
            rotary_len,
            placement: FlashMoeStatePlacement::CpuVisible,
        }
    }

    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn latent_len(self) -> usize {
        self.latent_len
    }

    pub(crate) fn rotary_len(self) -> usize {
        self.rotary_len
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        FlashMoeStateBufferRole::Kv
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.latent_len() > 0
            && self.rotary_len() > 0
            && FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

pub(super) type KvEntry = (Arc<[f32]>, Arc<[f32]>);
pub(super) type MlaKvEntry = (Arc<[f32]>, Arc<[f32]>);

#[derive(Debug, Clone)]
pub(super) struct KvCache {
    pub(super) layers: usize,
    pub(super) capacity: usize,
    prompt_tokens: Vec<(usize, u32)>,
    generated_tokens: Vec<(usize, u32)>,
    layer_states: Vec<(usize, usize, u64)>,
    pub(super) kv: Vec<Vec<Option<KvEntry>>>,
    pub(super) mla_kv: Vec<Vec<Option<MlaKvEntry>>>,
}

impl KvCache {
    pub(crate) fn new(layers: usize, capacity: usize) -> Self {
        Self {
            layers,
            capacity,
            prompt_tokens: Vec::new(),
            generated_tokens: Vec::new(),
            layer_states: Vec::new(),
            kv: vec![vec![None; capacity]; layers],
            mla_kv: vec![vec![None; capacity]; layers],
        }
    }

    pub(crate) fn shallow_snapshot(&self) -> Self {
        self.clone()
    }

    pub(crate) fn record_prompt_token(&mut self, position: usize, token: u32) -> Result<()> {
        self.ensure_position(position)?;
        self.prompt_tokens.push((position, token));
        Ok(())
    }

    pub(crate) fn record_prompt_token_record(
        &mut self,
        record: FlashMoePromptTokenRecord,
    ) -> Result<()> {
        self.record_prompt_token(record.position(), record.token())
    }

    pub(crate) fn resize_capacity(&mut self, capacity: usize) {
        if capacity <= self.capacity {
            return;
        }
        for layer in &mut self.kv {
            layer.resize_with(capacity, || None);
        }
        for layer in &mut self.mla_kv {
            layer.resize_with(capacity, || None);
        }
        self.capacity = capacity;
    }

    pub(crate) fn record_generated_token(&mut self, position: usize, token: u32) -> Result<()> {
        self.ensure_position(position)?;
        self.generated_tokens.push((position, token));
        Ok(())
    }

    pub(super) fn record_generated_token_record(
        &mut self,
        record: FlashMoeGeneratedTokenRecord,
    ) -> Result<()> {
        self.record_generated_token(record.position(), record.token())
    }

    pub(crate) fn record_layer_state(
        &mut self,
        position: usize,
        layer: usize,
        state: u64,
    ) -> Result<()> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        self.layer_states.push((position, layer, state));
        Ok(())
    }

    pub(super) fn record_layer_state_record(
        &mut self,
        record: FlashMoeLayerStateRecord,
    ) -> Result<()> {
        self.record_recurrent_layer_state(record.state(FlashMoeStatePlacement::CpuVisible))
    }

    pub(crate) fn record_recurrent_layer_state(
        &mut self,
        state: FlashMoeRecurrentLayerState,
    ) -> Result<()> {
        if !state.is_declared_graph_state() {
            bail!("FlashMoe recurrent layer state is not declared graph state");
        }
        if state.placement() != FlashMoeStatePlacement::CpuVisible {
            bail!(
                "FlashMoe recurrent layer state recording requires CpuVisible placement, got {:?}",
                state.placement()
            );
        }
        self.record_layer_state(state.position(), state.layer(), state.value())
    }

    pub(crate) fn record_kv(
        &mut self,
        position: usize,
        layer: usize,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<()> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        self.kv[layer][position] = Some((Arc::from(key), Arc::from(value)));
        Ok(())
    }

    pub(super) fn record_kv_record(&mut self, record: FlashMoeFullAttentionKvRecord) -> Result<()> {
        let position = record.position();
        let layer = record.layer();
        let (key, value) = record.into_key_value();
        self.record_kv(position, layer, key, value)
    }

    pub(super) fn record_mla_kv(
        &mut self,
        position: usize,
        layer: usize,
        latent: Vec<f32>,
        rotary_key: Vec<f32>,
    ) -> Result<()> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!(
                "MLA KV cache layer {layer} exceeds layer count {}",
                self.layers
            );
        }
        self.mla_kv[layer][position] = Some((Arc::from(latent), Arc::from(rotary_key)));
        Ok(())
    }

    pub(super) fn mla_records(
        &self,
        position: usize,
        layer: usize,
    ) -> Result<Vec<(&[f32], &[f32])>> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!(
                "MLA KV cache layer {layer} exceeds layer count {}",
                self.layers
            );
        }
        Ok(self.mla_kv[layer]
            .iter()
            .take(position + 1)
            .filter_map(|entry| {
                entry
                    .as_ref()
                    .map(|(latent, rotary)| (&latent[..], &rotary[..]))
            })
            .collect())
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    pub(super) fn causal_attention(
        &self,
        position: usize,
        layer: usize,
        query: &[f32],
        num_q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        let keys_values: Vec<(&[f32], &[f32])> = self.kv[layer]
            .iter()
            .take(position + 1)
            .filter_map(|entry| entry.as_ref().map(|(key, value)| (&key[..], &value[..])))
            .collect();
        Ok(causal_attention(
            query,
            &keys_values,
            num_q_heads,
            kv_heads,
            head_dim,
        ))
    }

    #[allow(dead_code)]
    pub(crate) fn keys_values(
        &self,
        position: usize,
        layer: usize,
    ) -> Result<Vec<(&[f32], &[f32])>> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        Ok(self.kv[layer]
            .iter()
            .take(position + 1)
            .filter_map(|entry| entry.as_ref().map(|(key, value)| (&key[..], &value[..])))
            .collect())
    }

    pub(crate) fn ensure_position(&self, position: usize) -> Result<()> {
        if position >= self.capacity {
            bail!(
                "KV cache position {position} exceeds capacity {}",
                self.capacity
            );
        }
        Ok(())
    }

    pub(crate) fn prefill_state_sha256(&self) -> (String, String) {
        let mut kv_digest = Sha256::new();
        kv_digest.update(b"pb.flashmoe.full-attention-kv.v1\0");
        kv_digest.update((self.layers as u64).to_le_bytes());
        for (layer, entries) in self.kv.iter().enumerate() {
            for (position, entry) in entries.iter().enumerate() {
                let Some((key, value)) = entry else {
                    continue;
                };
                kv_digest.update((layer as u64).to_le_bytes());
                kv_digest.update((position as u64).to_le_bytes());
                update_f32_digest(&mut kv_digest, key);
                update_f32_digest(&mut kv_digest, value);
            }
        }

        // Token-major and layer-major execution record the same states in a
        // different traversal order. Canonicalize by graph coordinates so the
        // digest measures state, not scheduling order.
        let mut layer_states = self.layer_states.clone();
        layer_states.sort_unstable_by_key(|(position, layer, _)| (*position, *layer));
        let mut trace_digest = Sha256::new();
        trace_digest.update(b"pb.flashmoe.router-recurrent-trace.v1\0");
        trace_digest.update((layer_states.len() as u64).to_le_bytes());
        for (position, layer, value) in layer_states {
            trace_digest.update((position as u64).to_le_bytes());
            trace_digest.update((layer as u64).to_le_bytes());
            trace_digest.update(value.to_le_bytes());
        }

        (
            format!("{:x}", kv_digest.finalize()),
            format!("{:x}", trace_digest.finalize()),
        )
    }

    pub(crate) fn prefill_layer_state_sha256(&self) -> (Vec<Option<String>>, Vec<Option<String>>) {
        let kv = self
            .kv
            .iter()
            .enumerate()
            .map(|(layer, entries)| {
                let present = entries.iter().any(Option::is_some);
                present.then(|| {
                    let mut digest = Sha256::new();
                    digest.update(b"pb.flashmoe.full-attention-kv-layer.v1\0");
                    digest.update((layer as u64).to_le_bytes());
                    for (position, entry) in entries.iter().enumerate() {
                        let Some((key, value)) = entry else {
                            continue;
                        };
                        digest.update((position as u64).to_le_bytes());
                        update_f32_digest(&mut digest, key);
                        update_f32_digest(&mut digest, value);
                    }
                    format!("{:x}", digest.finalize())
                })
            })
            .collect();
        let trace = (0..self.layers)
            .map(|layer| {
                let mut states = self
                    .layer_states
                    .iter()
                    .filter(|(_, state_layer, _)| *state_layer == layer)
                    .copied()
                    .collect::<Vec<_>>();
                if states.is_empty() {
                    return None;
                }
                states.sort_unstable_by_key(|(position, _, _)| *position);
                let mut digest = Sha256::new();
                digest.update(b"pb.flashmoe.router-recurrent-layer.v1\0");
                digest.update((layer as u64).to_le_bytes());
                for (position, _, value) in states {
                    digest.update((position as u64).to_le_bytes());
                    digest.update(value.to_le_bytes());
                }
                Some(format!("{:x}", digest.finalize()))
            })
            .collect();
        (kv, trace)
    }
}

#[derive(Debug, Default)]
pub(super) struct FlashMoeSessionCache {
    entries: BTreeMap<String, Vec<FlashMoeCachedSessionState>>,
    session_order: VecDeque<String>,
    prefixes: BTreeMap<String, FlashMoeCachedSessionState>,
    prefix_order: VecDeque<String>,
    dirty_sessions: BTreeSet<String>,
    dirty_prefixes: BTreeSet<String>,
    disk: Option<FlashMoeDiskCache>,
}

#[derive(Debug, Clone)]
pub(super) struct FlashMoeCachedSessionState {
    pub(super) cpu: FlashMoeSessionState<KvCache>,
    pub(super) recurrent: FlashMoeLinearAttentionSessionSnapshot,
}

impl FlashMoeSessionCache {
    pub(crate) fn new(disk: Option<FlashMoeDiskCache>) -> Self {
        Self {
            disk,
            ..Self::default()
        }
    }

    #[cfg(test)]
    pub(crate) fn begin_generation(
        &mut self,
        session_id: Option<&str>,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        layers: usize,
    ) -> FlashMoeGenerationState {
        self.begin_generation_with_base(session_id, prompt_tokens, 0, max_tokens, layers)
    }

    pub(crate) fn begin_generation_with_base(
        &mut self,
        session_id: Option<&str>,
        prompt_tokens: Vec<u32>,
        base_prefix_len: usize,
        max_tokens: usize,
        layers: usize,
    ) -> FlashMoeGenerationState {
        let capacity = prompt_tokens.len() + max_tokens;
        // A harness workflow keeps one logical session id while moving between
        // fresh stage prompts. Once the new prompt diverges, the cached state
        // cannot contribute to this generation and must not remain resident
        // beside the replacement KV cache for the duration of a long prefill.
        if let Some(id) = session_id {
            self.session_order.retain(|existing| existing != id);
        }
        let mut cached = session_id
            .and_then(|id| self.entries.remove(id))
            .and_then(|states| {
                states
                    .into_iter()
                    .filter_map(|state| {
                        reusable_session_prefix_len(&state.cpu.tokens, &prompt_tokens)
                            .map(|prefix_len| (prefix_len, state))
                    })
                    .max_by_key(|(prefix_len, _)| *prefix_len)
            });
        let mut cache_source = if cached.is_some() {
            PromptCacheSource::MemorySession
        } else {
            PromptCacheSource::None
        };
        let base_prefix_len = base_prefix_len.min(prompt_tokens.len());
        let base_key = (base_prefix_len > 0)
            .then(|| FlashMoeDiskCache::token_key(&prompt_tokens[..base_prefix_len]));
        if let Some(key) = base_key.as_ref()
            && let Some(state) = self.prefixes.get(key).cloned()
            && reusable_session_prefix_len(&state.cpu.tokens, &prompt_tokens).is_some_and(
                |prefix_len| {
                    cached
                        .as_ref()
                        .is_none_or(|(cached_len, _)| prefix_len > *cached_len)
                },
            )
        {
            self.touch_prefix(key);
            cached = Some((state.cpu.tokens.len(), state));
            cache_source = PromptCacheSource::MemoryPrefix;
        }
        let restore_started = Instant::now();
        let mut used_disk = false;
        if cached.is_none()
            && let (Some(id), Some(disk)) = (session_id, self.disk.as_ref())
        {
            match disk.load_session(id) {
                Ok(states) => {
                    if let Some(found) = states
                        .into_iter()
                        .filter_map(|state| {
                            reusable_session_prefix_len(&state.cpu.tokens, &prompt_tokens)
                                .map(|prefix_len| (prefix_len, state))
                        })
                        .max_by_key(|(prefix_len, _)| *prefix_len)
                    {
                        cached = Some(found);
                        cache_source = PromptCacheSource::DiskSession;
                        used_disk = true;
                    }
                }
                Err(error) => tracing::warn!(
                    session = id,
                    error = %format!("{error:#}"),
                    "ignored unreadable FlashMoe session cache"
                ),
            }
        }
        if cached.is_none()
            && let (Some(key), Some(disk)) = (base_key.as_ref(), self.disk.as_ref())
        {
            match disk.load_prefix(&prompt_tokens[..base_prefix_len]) {
                Ok(Some(state)) => {
                    self.prefixes.insert(key.clone(), state.clone());
                    self.touch_prefix(key);
                    cached = Some((base_prefix_len, state));
                    cache_source = PromptCacheSource::DiskPrefix;
                    used_disk = true;
                }
                Ok(None) => {}
                Err(error) => tracing::warn!(
                    prefix = key,
                    error = %format!("{error:#}"),
                    "ignored unreadable FlashMoe prefix cache"
                ),
            }
        }
        let restore_ms = used_disk
            .then(|| u64::try_from(restore_started.elapsed().as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        let (kv_cache, prefill_start, cached_last_hidden, cached_recurrent) =
            if let Some((prefix_len, state)) = cached {
                let FlashMoeSessionState {
                    tokens: _,
                    mut kv_cache,
                    last_hidden,
                } = state.cpu;
                kv_cache.resize_capacity(capacity);
                let cached_last_hidden = (prefix_len == prompt_tokens.len()).then_some(last_hidden);
                (
                    kv_cache,
                    prefix_len,
                    cached_last_hidden,
                    Some(state.recurrent),
                )
            } else {
                (KvCache::new(layers, capacity), 0, None, None)
            };

        FlashMoeGenerationState {
            session_id: session_id.map(str::to_owned),
            prompt_tokens,
            kv_cache,
            prefill_start,
            cached_last_hidden,
            prompt_cache: None,
            cached_recurrent,
            prompt_recurrent: None,
            generated_cache: None,
            generated_recurrent: None,
            cache_source,
            cache_restore_ms: restore_ms,
            base_prefix_len,
            base_cache: None,
            base_recurrent: None,
            generated: Vec::new(),
            max_tokens,
            stopped: false,
            stopped_by_terminal_tool_call: false,
            stopped_by_constraint_payload_limit: false,
        }
    }

    pub(crate) fn begin_external_prefix_generation(
        prompt_tokens: Vec<u32>,
        prefill_start: usize,
        cached_last_hidden: Option<Vec<f32>>,
        max_tokens: usize,
        layers: usize,
        cache_source: PromptCacheSource,
        cache_restore_ms: u64,
    ) -> Result<FlashMoeGenerationState> {
        if prefill_start > prompt_tokens.len() {
            bail!(
                "external FlashMoe prefix {prefill_start} exceeds prompt length {}",
                prompt_tokens.len()
            );
        }
        let mut kv_cache = KvCache::new(layers, prompt_tokens.len() + max_tokens);
        for (position, token) in prompt_tokens
            .iter()
            .copied()
            .enumerate()
            .take(prefill_start)
        {
            kv_cache.record_prompt_token_record(FlashMoePromptTokenRecord::new(position, token))?;
        }
        Ok(FlashMoeGenerationState {
            session_id: None,
            prompt_tokens,
            kv_cache,
            prefill_start,
            cached_last_hidden,
            prompt_cache: None,
            cached_recurrent: None,
            prompt_recurrent: None,
            generated_cache: None,
            generated_recurrent: None,
            cache_source,
            cache_restore_ms,
            base_prefix_len: 0,
            base_cache: None,
            base_recurrent: None,
            generated: Vec::new(),
            max_tokens,
            stopped: false,
            stopped_by_terminal_tool_call: false,
            stopped_by_constraint_payload_limit: false,
        })
    }

    pub(crate) fn commit_generation(
        &mut self,
        generation: &mut FlashMoeGenerationState,
    ) -> Result<()> {
        let Some(session_id) = generation.session_id.as_ref() else {
            return Ok(());
        };
        let cpu = generation
            .prompt_cache
            .take()
            .context("session cache prompt snapshot is missing")?;
        let recurrent = generation
            .prompt_recurrent
            .take()
            .context("session cache recurrent snapshot is missing")?;
        let mut checkpoints = vec![FlashMoeCachedSessionState { cpu, recurrent }];
        if let (Some(cpu), Some(recurrent)) = (
            generation.generated_cache.take(),
            generation.generated_recurrent.take(),
        ) {
            checkpoints.push(FlashMoeCachedSessionState { cpu, recurrent });
        }
        self.entries.insert(session_id.clone(), checkpoints);
        self.touch_session(session_id);
        self.dirty_sessions.insert(session_id.clone());
        self.evict_excess_sessions(memory_session_limit());
        if let (Some(cpu), Some(recurrent)) = (
            generation.base_cache.take(),
            generation.base_recurrent.take(),
        ) {
            let state = FlashMoeCachedSessionState { cpu, recurrent };
            let key = FlashMoeDiskCache::token_key(&state.cpu.tokens);
            self.prefixes.insert(key.clone(), state);
            self.touch_prefix(&key);
            self.dirty_prefixes.insert(key);
            while self.prefixes.len() > 4 {
                let Some(oldest) = self.prefix_order.pop_front() else {
                    break;
                };
                self.prefixes.remove(&oldest);
                self.dirty_prefixes.remove(&oldest);
            }
        }
        Ok(())
    }

    fn touch_prefix(&mut self, key: &str) {
        self.prefix_order.retain(|existing| existing != key);
        self.prefix_order.push_back(key.to_string());
    }

    fn touch_session(&mut self, session_id: &str) {
        self.session_order.retain(|existing| existing != session_id);
        self.session_order.push_back(session_id.to_string());
    }

    fn evict_excess_sessions(&mut self, limit: usize) {
        while self.entries.len() > limit {
            let Some(oldest) = self.session_order.pop_front() else {
                break;
            };
            if self.dirty_sessions.contains(&oldest)
                && let (Some(disk), Some(safe_prompt)) = (
                    self.disk.as_ref(),
                    self.entries.get(&oldest).and_then(|states| states.first()),
                )
                && let Err(error) = disk.persist_session(&oldest, std::slice::from_ref(safe_prompt))
            {
                tracing::warn!(
                    session = oldest,
                    error = %format!("{error:#}"),
                    "could not persist evicted FlashMoe session cache"
                );
            }
            self.entries.remove(&oldest);
            self.dirty_sessions.remove(&oldest);
        }
    }

    pub(crate) fn persist_session(&mut self, session_id: &str) -> Result<()> {
        let Some(disk) = self.disk.as_ref() else {
            return Ok(());
        };
        let prefix_keys = self.dirty_prefixes.iter().cloned().collect::<Vec<_>>();
        for key in &prefix_keys {
            if let Some(state) = self.prefixes.get(key) {
                disk.persist_prefix(state)?;
            }
        }
        if self.dirty_sessions.contains(session_id)
            && let Some(states) = self.entries.get(session_id)
            && let Some(safe_prompt) = states.first()
        {
            // The generated head is a speculative in-memory accelerator. Persist
            // only the canonical prompt boundary so restart durability does not
            // double checkpoint writes or depend on output re-tokenization.
            disk.persist_session(session_id, std::slice::from_ref(safe_prompt))?;
        }
        for key in prefix_keys {
            self.dirty_prefixes.remove(&key);
        }
        self.dirty_sessions.remove(session_id);
        Ok(())
    }
}

fn memory_session_limit() -> usize {
    const DEFAULT_MEMORY_SESSIONS: usize = 2;
    std::env::var("PB_FLASHMOE_MEMORY_SESSIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MEMORY_SESSIONS)
}

#[derive(Debug)]
pub(super) struct FlashMoeGenerationState {
    session_id: Option<String>,
    prompt_tokens: Vec<u32>,
    kv_cache: KvCache,
    prefill_start: usize,
    cached_last_hidden: Option<Vec<f32>>,
    prompt_cache: Option<FlashMoeSessionState<KvCache>>,
    cached_recurrent: Option<FlashMoeLinearAttentionSessionSnapshot>,
    prompt_recurrent: Option<FlashMoeLinearAttentionSessionSnapshot>,
    generated_cache: Option<FlashMoeSessionState<KvCache>>,
    generated_recurrent: Option<FlashMoeLinearAttentionSessionSnapshot>,
    cache_source: PromptCacheSource,
    cache_restore_ms: u64,
    base_prefix_len: usize,
    base_cache: Option<FlashMoeSessionState<KvCache>>,
    base_recurrent: Option<FlashMoeLinearAttentionSessionSnapshot>,
    generated: Vec<u32>,
    max_tokens: usize,
    stopped: bool,
    stopped_by_terminal_tool_call: bool,
    stopped_by_constraint_payload_limit: bool,
}

impl FlashMoeGenerationState {
    pub(crate) fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    pub(crate) fn checkpoint_tokens(&self, evaluated_generated_tokens: usize) -> Vec<u32> {
        let mut tokens = stable_session_cache_tokens(&self.prompt_tokens);
        tokens.extend_from_slice(
            &self.generated[..evaluated_generated_tokens.min(self.generated.len())],
        );
        tokens
    }

    pub(crate) fn prompt_tokens_through(&self, prefix_len: usize) -> Vec<u32> {
        self.prompt_tokens[..prefix_len.min(self.prompt_tokens.len())].to_vec()
    }

    pub(crate) fn prefill_start(&self) -> usize {
        self.prefill_start
    }

    pub(crate) fn cache_source(&self) -> PromptCacheSource {
        self.cache_source
    }

    pub(crate) fn cache_restore_ms(&self) -> u64 {
        self.cache_restore_ms
    }

    pub(crate) fn base_prefix_len(&self) -> usize {
        self.base_prefix_len
    }

    pub(crate) fn take_cached_last_hidden(&mut self) -> Option<Vec<f32>> {
        self.cached_last_hidden.take()
    }

    pub(crate) fn take_cached_recurrent(
        &mut self,
    ) -> Option<FlashMoeLinearAttentionSessionSnapshot> {
        self.cached_recurrent.take()
    }

    pub(crate) fn prefill_inputs(&mut self) -> (&[u32], usize, &mut KvCache) {
        (&self.prompt_tokens, self.prefill_start, &mut self.kv_cache)
    }

    pub(crate) fn prefill_state_sha256(&self) -> (String, String) {
        self.kv_cache.prefill_state_sha256()
    }

    pub(crate) fn prefill_layer_state_sha256(&self) -> (Vec<Option<String>>, Vec<Option<String>>) {
        self.kv_cache.prefill_layer_state_sha256()
    }

    pub(crate) fn requires_prompt_snapshot(&self) -> bool {
        self.session_id.is_some()
    }

    pub(crate) fn capture_prompt_cache(
        &mut self,
        last_hidden: Vec<f32>,
        recurrent: FlashMoeLinearAttentionSessionSnapshot,
    ) {
        if self.session_id.is_none() {
            return;
        }
        self.prompt_cache = Some(FlashMoeSessionState::new(
            stable_session_cache_tokens(&self.prompt_tokens),
            self.kv_cache.shallow_snapshot(),
            last_hidden,
        ));
        self.prompt_recurrent = Some(recurrent);
    }

    pub(crate) fn capture_base_cache(
        &mut self,
        last_hidden: Vec<f32>,
        recurrent: FlashMoeLinearAttentionSessionSnapshot,
    ) {
        if self.base_prefix_len == 0 || self.base_prefix_len > self.prompt_tokens.len() {
            return;
        }
        self.base_cache = Some(FlashMoeSessionState::new(
            self.prompt_tokens[..self.base_prefix_len].to_vec(),
            self.kv_cache.shallow_snapshot(),
            last_hidden,
        ));
        self.base_recurrent = Some(recurrent);
    }

    pub(crate) fn capture_generated_cache(
        &mut self,
        evaluated_generated_tokens: usize,
        last_hidden: Vec<f32>,
        recurrent: FlashMoeLinearAttentionSessionSnapshot,
    ) {
        if self.session_id.is_none() || evaluated_generated_tokens == 0 {
            return;
        }
        let mut tokens = stable_session_cache_tokens(&self.prompt_tokens);
        tokens.extend_from_slice(
            &self.generated[..evaluated_generated_tokens.min(self.generated.len())],
        );
        self.generated_cache = Some(FlashMoeSessionState::new(
            tokens,
            self.kv_cache.shallow_snapshot(),
            last_hidden,
        ));
        self.generated_recurrent = Some(recurrent);
    }

    pub(crate) fn should_sample_first(&self) -> bool {
        self.max_tokens > 0
    }

    pub(crate) fn sample_inputs(&self) -> (&[u32], &[u32]) {
        (&self.prompt_tokens, &self.generated)
    }

    pub(crate) fn record_sampled_token(
        &mut self,
        token: u32,
        is_eos: bool,
        terminal_tool_call: bool,
    ) {
        if is_eos {
            self.stopped = true;
            self.stopped_by_terminal_tool_call = false;
            self.stopped_by_constraint_payload_limit = false;
        } else {
            self.generated.push(token);
            self.stopped = terminal_tool_call;
            self.stopped_by_terminal_tool_call = terminal_tool_call;
            self.stopped_by_constraint_payload_limit = false;
        }
    }

    pub(crate) fn stopped_by_terminal_tool_call(&self) -> bool {
        self.stopped_by_terminal_tool_call
    }

    pub(crate) fn stop_at_constraint_payload_limit(&mut self) {
        self.stopped = true;
        self.stopped_by_terminal_tool_call = false;
        self.stopped_by_constraint_payload_limit = true;
    }

    pub(crate) fn stopped_by_constraint_payload_limit(&self) -> bool {
        self.stopped_by_constraint_payload_limit
    }

    pub(crate) fn should_decode(&self) -> bool {
        !self.stopped && self.generated.len() < self.max_tokens
    }

    pub(crate) fn decode_inputs(&mut self) -> Result<(&[u32], &[u32], &mut KvCache, usize)> {
        let position = self
            .prompt_tokens
            .len()
            .checked_add(self.generated.len())
            .and_then(|position| position.checked_sub(1))
            .context("FlashMoe decode requires a prompt or generated token")?;
        Ok((
            &self.prompt_tokens,
            &self.generated,
            &mut self.kv_cache,
            position,
        ))
    }

    pub(crate) fn generated_len(&self) -> usize {
        self.generated.len()
    }

    pub(crate) fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub(crate) fn into_generated(self) -> Vec<u32> {
        self.generated
    }
}

#[cfg(test)]
#[path = "state_parity_tests.rs"]
mod parity_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn recurrent_session_snapshot() -> FlashMoeLinearAttentionSessionSnapshot {
        FlashMoeLinearAttentionSessionSnapshot::new(vec![Some(
            FlashMoeLinearAttentionLayerSnapshot::new(0, vec![1.0, 2.0], vec![3.0, 4.0, 5.0], 2, 2)
                .unwrap(),
        )])
        .unwrap()
    }

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
    fn kv_cache_snapshot_shares_existing_entries_and_grows_independently() {
        let mut cache = KvCache::new(1, 2);
        cache
            .record_kv(1, 0, vec![1.0, 1.5], vec![2.0, 2.5])
            .unwrap();

        let mut snapshot = cache.shallow_snapshot();
        let (cache_key, cache_value) = cache.kv[0][1].as_ref().unwrap();
        let (snapshot_key, snapshot_value) = snapshot.kv[0][1].as_ref().unwrap();
        assert!(Arc::ptr_eq(cache_key, snapshot_key));
        assert!(Arc::ptr_eq(cache_value, snapshot_value));

        snapshot.resize_capacity(3);
        snapshot
            .record_kv(2, 0, vec![3.0, 3.5], vec![4.0, 4.5])
            .unwrap();
        assert_eq!(cache.capacity, 2);
        assert_eq!(snapshot.capacity, 3);
        assert_eq!(snapshot.keys_values(2, 0).unwrap().len(), 2);
    }

    #[test]
    fn kv_cache_rejects_gpu_recurrent_state_without_fallback() {
        let mut cache = KvCache::new(2, 2);
        cache
            .record_recurrent_layer_state(FlashMoeRecurrentLayerState::cpu_visible(1, 0, 99))
            .unwrap();
        assert_eq!(cache.layer_states, vec![(1, 0, 99)]);

        let err = cache
            .record_recurrent_layer_state(FlashMoeRecurrentLayerState::new(
                1,
                0,
                99,
                FlashMoeStatePlacement::GpuResident,
            ))
            .unwrap_err();
        assert!(
            err.to_string().contains("requires CpuVisible placement"),
            "{err:#}"
        );
    }

    #[test]
    fn prefill_state_digest_canonicalizes_token_and_layer_major_record_order() {
        let mut token_major = KvCache::new(2, 2);
        let mut layer_major = KvCache::new(2, 2);
        for cache in [&mut token_major, &mut layer_major] {
            cache.record_kv(0, 1, vec![1.0], vec![2.0]).unwrap();
            cache.record_kv(1, 1, vec![3.0], vec![4.0]).unwrap();
        }
        token_major.record_layer_state(0, 0, 10).unwrap();
        token_major.record_layer_state(0, 1, 11).unwrap();
        token_major.record_layer_state(1, 0, 12).unwrap();
        token_major.record_layer_state(1, 1, 13).unwrap();
        layer_major.record_layer_state(0, 0, 10).unwrap();
        layer_major.record_layer_state(1, 0, 12).unwrap();
        layer_major.record_layer_state(0, 1, 11).unwrap();
        layer_major.record_layer_state(1, 1, 13).unwrap();

        assert_eq!(
            token_major.prefill_state_sha256(),
            layer_major.prefill_state_sha256()
        );
        layer_major.layer_states[3].2 ^= 1;
        assert_ne!(
            token_major.prefill_state_sha256(),
            layer_major.prefill_state_sha256()
        );
    }

    #[test]
    fn linear_attention_state_digest_includes_exact_float_bits() {
        let first = recurrent_session_snapshot();
        let changed = FlashMoeLinearAttentionSessionSnapshot::new(vec![Some(
            FlashMoeLinearAttentionLayerSnapshot::new(
                0,
                vec![1.0, f32::from_bits(2.0f32.to_bits() + 1)],
                vec![3.0, 4.0, 5.0],
                2,
                2,
            )
            .unwrap(),
        )])
        .unwrap();
        assert_ne!(first.state_sha256(), changed.state_sha256());
    }

    #[test]
    fn generation_lifecycle_commits_and_reuses_state_owned_prompt_snapshot() {
        let mut sessions = FlashMoeSessionCache::default();
        let mut generation = sessions.begin_generation(Some("chat"), vec![10, 20], 2, 1);
        assert_eq!(generation.prefill_start(), 0);
        {
            let (prompt, start, kv_cache) = generation.prefill_inputs();
            assert_eq!(prompt, &[10, 20]);
            assert_eq!(start, 0);
            kv_cache
                .record_kv(0, 0, vec![1.0, 1.5], vec![2.0, 2.5])
                .unwrap();
            kv_cache
                .record_kv(1, 0, vec![3.0, 3.5], vec![4.0, 4.5])
                .unwrap();
        }
        generation.capture_prompt_cache(vec![9.0, 9.5], recurrent_session_snapshot());
        generation.record_sampled_token(30, false, false);
        assert_eq!(generation.generated, vec![30]);
        sessions.commit_generation(&mut generation).unwrap();

        let mut reused = sessions.begin_generation(Some("chat"), vec![10, 20], 1, 1);
        assert_eq!(reused.prefill_start(), 2);
        assert_eq!(reused.take_cached_last_hidden(), Some(vec![9.0, 9.5]));
        assert_eq!(
            reused.take_cached_recurrent(),
            Some(recurrent_session_snapshot())
        );
        assert_eq!(reused.kv_cache.keys_values(1, 0).unwrap().len(), 2);
        assert!(reused.generated.is_empty());
    }

    #[test]
    fn generation_lifecycle_prefers_the_exact_generated_head_checkpoint() {
        let mut sessions = FlashMoeSessionCache::default();
        let mut generation = sessions.begin_generation(Some("chat"), vec![10, 20], 3, 1);
        generation.capture_prompt_cache(vec![2.0], recurrent_session_snapshot());
        generation.record_sampled_token(30, false, false);
        generation
            .kv_cache
            .record_kv(2, 0, vec![3.0], vec![4.0])
            .unwrap();
        generation.capture_generated_cache(1, vec![3.0], recurrent_session_snapshot());
        sessions.commit_generation(&mut generation).unwrap();

        let mut reused = sessions.begin_generation(Some("chat"), vec![10, 20, 30, 40], 1, 1);
        assert_eq!(reused.prefill_start(), 3);
        assert_eq!(reused.cache_source(), PromptCacheSource::MemorySession);
        assert_eq!(reused.take_cached_last_hidden(), None);
        assert_eq!(reused.kv_cache.keys_values(2, 0).unwrap().len(), 1);
    }

    #[test]
    fn stable_base_checkpoint_is_shared_across_logical_sessions() {
        let mut sessions = FlashMoeSessionCache::default();
        let mut first =
            sessions.begin_generation_with_base(Some("first"), vec![10, 20, 30], 2, 1, 1);
        first
            .kv_cache
            .record_kv(0, 0, vec![1.0], vec![2.0])
            .unwrap();
        first
            .kv_cache
            .record_kv(1, 0, vec![3.0], vec![4.0])
            .unwrap();
        first.capture_base_cache(vec![5.0], recurrent_session_snapshot());
        first.capture_prompt_cache(vec![6.0], recurrent_session_snapshot());
        sessions.commit_generation(&mut first).unwrap();

        let mut second =
            sessions.begin_generation_with_base(Some("second"), vec![10, 20, 99], 2, 1, 1);
        assert_eq!(second.prefill_start(), 2);
        assert_eq!(second.cache_source(), PromptCacheSource::MemoryPrefix);
        assert_eq!(second.take_cached_last_hidden(), None);
        assert_eq!(second.kv_cache.keys_values(1, 0).unwrap().len(), 2);
    }

    #[test]
    fn memory_session_cache_evicts_the_least_recently_used_conversation() {
        let mut sessions = FlashMoeSessionCache::default();
        for (session_id, token) in [("first", 10), ("second", 20), ("third", 30)] {
            let mut generation = sessions.begin_generation(Some(session_id), vec![token], 1, 1);
            generation.capture_prompt_cache(vec![token as f32], recurrent_session_snapshot());
            sessions.commit_generation(&mut generation).unwrap();
        }

        sessions.evict_excess_sessions(2);
        assert!(!sessions.entries.contains_key("first"));
        assert!(sessions.entries.contains_key("second"));
        assert!(sessions.entries.contains_key("third"));
        assert_eq!(
            sessions.session_order,
            VecDeque::from(["second".into(), "third".into()])
        );
    }

    #[test]
    fn generation_lifecycle_evicts_a_nonmatching_session_before_fresh_prefill() {
        let mut sessions = FlashMoeSessionCache::default();
        let mut generation = sessions.begin_generation(Some("chat"), vec![10, 20], 1, 1);
        generation.capture_prompt_cache(vec![9.0, 9.5], recurrent_session_snapshot());
        sessions.commit_generation(&mut generation).unwrap();
        assert_eq!(sessions.entries.len(), 1);

        let fresh = sessions.begin_generation(Some("chat"), vec![30, 40], 1, 1);

        assert_eq!(fresh.prefill_start(), 0);
        assert!(fresh.cached_last_hidden.is_none());
        assert!(fresh.cached_recurrent.is_none());
        assert!(sessions.entries.is_empty());
    }

    #[test]
    fn generation_lifecycle_owns_decode_position_and_stop_state() {
        let mut sessions = FlashMoeSessionCache::default();
        let mut generation = sessions.begin_generation(None, vec![10, 20], 2, 1);
        assert!(generation.should_sample_first());
        generation.record_sampled_token(30, false, false);
        assert!(generation.should_decode());
        let (prompt, generated, _, position) = generation.decode_inputs().unwrap();
        assert_eq!(prompt, &[10, 20]);
        assert_eq!(generated, &[30]);
        assert_eq!(position, 2);

        generation.record_sampled_token(0, true, false);
        assert!(!generation.should_decode());
        assert_eq!(generation.into_generated(), vec![30]);
    }

    #[test]
    fn generation_lifecycle_keeps_the_token_that_closes_a_terminal_tool_call() {
        let mut sessions = FlashMoeSessionCache::default();
        let mut generation = sessions.begin_generation(None, vec![10, 20], 4, 1);

        generation.record_sampled_token(30, false, true);

        assert!(!generation.should_decode());
        assert!(generation.stopped_by_terminal_tool_call());
        assert_eq!(generation.into_generated(), vec![30]);
    }

    #[test]
    fn generation_lifecycle_stops_before_a_constraint_payload_limit_sentinel() {
        let mut sessions = FlashMoeSessionCache::default();
        let mut generation = sessions.begin_generation(None, vec![10, 20], 4, 1);
        generation.record_sampled_token(30, false, false);

        generation.stop_at_constraint_payload_limit();

        assert!(!generation.should_decode());
        assert!(generation.stopped_by_constraint_payload_limit());
        assert_eq!(generation.into_generated(), vec![30]);
    }

    #[test]
    fn recurrent_session_snapshot_requires_declared_layer_shape_and_order() {
        let snapshot = recurrent_session_snapshot();
        let layer = snapshot.layer(0).unwrap();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(layer.state().layer(), 0);
        assert_eq!(
            layer.state().placement(),
            FlashMoeStatePlacement::CpuVisible
        );
        assert_eq!(layer.conv_state(), &[1.0, 2.0]);
        assert_eq!(layer.ssm_state(), &[3.0, 4.0, 5.0]);

        let empty =
            FlashMoeLinearAttentionLayerSnapshot::new(0, Vec::new(), vec![1.0], 1, 1).unwrap_err();
        assert!(
            empty
                .to_string()
                .contains("not declared CPU-visible graph state"),
            "{empty:#}"
        );

        let misplaced = FlashMoeLinearAttentionSessionSnapshot::new(vec![
            None,
            Some(FlashMoeLinearAttentionLayerSnapshot::new(0, vec![1.0], vec![2.0], 1, 1).unwrap()),
        ])
        .unwrap_err();
        assert!(
            misplaced.to_string().contains("layer 1 does not match"),
            "{misplaced:#}"
        );
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

    #[test]
    fn mla_kv_state_declares_distinct_latent_and_rotary_widths() {
        let state = FlashMoeMlaKvState::cpu_visible(7, 3, 512, 64);
        assert_eq!(state.position(), 7);
        assert_eq!(state.layer(), 3);
        assert_eq!(state.latent_len(), 512);
        assert_eq!(state.rotary_len(), 64);
        assert_eq!(state.role(), FlashMoeStateBufferRole::Kv);
        assert_eq!(state.placement(), FlashMoeStatePlacement::CpuVisible);
        assert!(state.is_declared_graph_state());
        assert!(!FlashMoeMlaKvState::cpu_visible(7, 3, 0, 64).is_declared_graph_state());
        assert!(!FlashMoeMlaKvState::cpu_visible(7, 3, 512, 0).is_declared_graph_state());
    }
}
