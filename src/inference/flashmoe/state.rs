use std::collections::BTreeMap;

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
}
