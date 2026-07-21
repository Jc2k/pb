#[cfg(test)]
use std::collections::{BTreeMap, VecDeque};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use super::math::causal_attention;
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

mod buffers;
pub use buffers::*;

mod recurrent;
pub(crate) use recurrent::*;

mod kv_cache;
pub(in crate::inference::flashmoe) use kv_cache::*;

mod generation;
pub(in crate::inference::flashmoe) use generation::*;

#[cfg(test)]
use super::session_cache::FlashMoeSessionCache;

#[cfg(test)]
#[path = "parity_tests.rs"]
mod parity_tests;

#[cfg(test)]
#[path = "../tests/state.rs"]
mod tests;
