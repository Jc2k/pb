use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};

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
    pub(crate) const GENERATION_ROLES: [Self; 8] = [
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
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeExpertPhaseOutput {
    hidden: Vec<f32>,
    next_normed: Option<Vec<f32>>,
}

impl FlashMoeExpertPhaseOutput {
    pub(crate) fn new(hidden: Vec<f32>, next_normed: Option<Vec<f32>>) -> Self {
        Self {
            hidden,
            next_normed,
        }
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

    pub(crate) fn key(&self) -> &[f32] {
        &self.key
    }

    pub(crate) fn value(&self) -> &[f32] {
        &self.value
    }

    pub(crate) fn into_key_value(self) -> (Vec<f32>, Vec<f32>) {
        (self.key, self.value)
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
        let (key, value) = record.into_key_value();
        assert_eq!(key, vec![1.0, 1.5]);
        assert_eq!(value, vec![2.0, 2.5]);
    }
}
