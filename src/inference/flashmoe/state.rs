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
    Kv,
    Recurrent,
}

impl FlashMoeStateBufferRole {
    pub(crate) const GENERATION_ROLES: [Self; 6] = [
        Self::Hidden,
        Self::Residual,
        Self::Normed,
        Self::NextLayerNormed,
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
}
