use std::collections::{BTreeMap, VecDeque};

use anyhow::{Result, bail};

use super::state::reusable_session_prefix_len;

const DEEPSEEK_MEMORY_SESSION_LIMIT: usize = 2;
const DEEPSEEK_CHECKPOINTS_PER_SESSION: usize = 2;

pub(super) fn scoped_structured_session_id(session_id: &str, stable_prompt_sha256: &str) -> String {
    format!("{session_id}:root:{stable_prompt_sha256}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeepSeekV4CheckpointKind {
    StablePrompt,
    Prompt,
    Generated,
}

#[derive(Debug)]
pub(super) struct DeepSeekV4SessionCheckpoint<S> {
    kind: DeepSeekV4CheckpointKind,
    tokens: Vec<u32>,
    last_hidden: Vec<f32>,
    state: S,
}

impl<S> DeepSeekV4SessionCheckpoint<S> {
    pub(super) fn new(
        kind: DeepSeekV4CheckpointKind,
        tokens: Vec<u32>,
        last_hidden: Vec<f32>,
        state: S,
    ) -> Self {
        Self {
            kind,
            tokens,
            last_hidden,
            state,
        }
    }

    pub(super) fn kind(&self) -> DeepSeekV4CheckpointKind {
        self.kind
    }

    pub(super) fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub(super) fn last_hidden(&self) -> &[f32] {
        &self.last_hidden
    }

    pub(super) fn state(&self) -> &S {
        &self.state
    }
}

#[derive(Debug)]
pub(super) struct DeepSeekV4SessionStore<S> {
    entries: BTreeMap<String, Vec<DeepSeekV4SessionCheckpoint<S>>>,
    order: VecDeque<String>,
}

impl<S> Default for DeepSeekV4SessionStore<S> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
            order: VecDeque::new(),
        }
    }
}

impl<S> DeepSeekV4SessionStore<S> {
    pub(super) fn reusable_checkpoint(
        &self,
        session_id: &str,
        prompt_tokens: &[u32],
    ) -> Result<Option<(usize, &DeepSeekV4SessionCheckpoint<S>)>> {
        let Some(checkpoints) = self.entries.get(session_id) else {
            return Ok(None);
        };
        let checkpoint = checkpoints
            .iter()
            .filter_map(|checkpoint| {
                reusable_session_prefix_len(checkpoint.tokens(), prompt_tokens)
                    .map(|prefix| (prefix, checkpoint))
            })
            .max_by_key(|(prefix, _)| *prefix);
        if checkpoint.is_none() {
            let frontiers = checkpoints
                .iter()
                .map(|checkpoint| checkpoint.tokens().len().to_string())
                .collect::<Vec<_>>()
                .join(",");
            bail!(
                "DeepSeek V4 session prefix mismatch for session '{session_id}': prompt tokens={} cached frontiers=[{frontiers}]",
                prompt_tokens.len()
            );
        }
        Ok(checkpoint)
    }

    pub(super) fn replace_stable_prompt(
        &mut self,
        session_id: &str,
        checkpoint: DeepSeekV4SessionCheckpoint<S>,
    ) {
        debug_assert_eq!(checkpoint.kind(), DeepSeekV4CheckpointKind::StablePrompt);
        let checkpoints = self.entries.entry(session_id.to_string()).or_default();
        checkpoints.retain(|existing| existing.kind() != DeepSeekV4CheckpointKind::StablePrompt);
        checkpoints.push(checkpoint);
        Self::bound_checkpoints(checkpoints);
        self.touch(session_id);
        self.evict_excess_sessions();
    }

    pub(super) fn push_checkpoint(
        &mut self,
        session_id: &str,
        checkpoint: DeepSeekV4SessionCheckpoint<S>,
    ) {
        let checkpoints = self.entries.entry(session_id.to_string()).or_default();
        checkpoints.retain(|existing| {
            existing.kind() != checkpoint.kind() || existing.tokens() != checkpoint.tokens()
        });
        checkpoints.push(checkpoint);
        Self::bound_checkpoints(checkpoints);
        self.touch(session_id);
        self.evict_excess_sessions();
    }

    fn bound_checkpoints(checkpoints: &mut Vec<DeepSeekV4SessionCheckpoint<S>>) {
        checkpoints.sort_by_key(|checkpoint| checkpoint.tokens().len());
        while checkpoints.len() > DEEPSEEK_CHECKPOINTS_PER_SESSION {
            let removable = checkpoints
                .iter()
                .position(|checkpoint| checkpoint.kind() != DeepSeekV4CheckpointKind::StablePrompt)
                .unwrap_or(0);
            checkpoints.remove(removable);
        }
    }

    fn touch(&mut self, session_id: &str) {
        self.order.retain(|existing| existing != session_id);
        self.order.push_back(session_id.to_string());
    }

    fn evict_excess_sessions(&mut self) {
        while self.entries.len() > DEEPSEEK_MEMORY_SESSION_LIMIT {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(
        kind: DeepSeekV4CheckpointKind,
        tokens: &[u32],
        marker: u8,
    ) -> DeepSeekV4SessionCheckpoint<u8> {
        DeepSeekV4SessionCheckpoint::new(kind, tokens.to_vec(), vec![marker as f32], marker)
    }

    #[test]
    fn exact_prefix_selects_the_longest_prompt_or_generated_checkpoint() {
        let mut sessions = DeepSeekV4SessionStore::default();
        sessions.replace_stable_prompt(
            "agent",
            checkpoint(DeepSeekV4CheckpointKind::StablePrompt, &[1, 2], 1),
        );
        sessions.push_checkpoint(
            "agent",
            checkpoint(DeepSeekV4CheckpointKind::Generated, &[1, 2, 3, 4], 2),
        );

        let (prefix, selected) = sessions
            .reusable_checkpoint("agent", &[1, 2, 3, 4, 5])
            .unwrap()
            .unwrap();

        assert_eq!(prefix, 4);
        assert_eq!(*selected.state(), 2);
        assert_eq!(selected.last_hidden(), &[2.0]);
    }

    #[test]
    fn same_session_mismatch_is_a_named_error_without_eviction() {
        let mut sessions = DeepSeekV4SessionStore::default();
        sessions.replace_stable_prompt(
            "agent",
            checkpoint(DeepSeekV4CheckpointKind::StablePrompt, &[1, 2], 1),
        );

        let error = sessions
            .reusable_checkpoint("agent", &[1, 9, 3])
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("DeepSeek V4 session prefix mismatch")
        );
        assert!(
            sessions
                .reusable_checkpoint("agent", &[1, 2, 3])
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn structured_session_ids_are_scoped_to_the_exact_stable_prompt() {
        assert_eq!(
            scoped_structured_session_id("agent:workflow:Planning", "abc123"),
            "agent:workflow:Planning:root:abc123"
        );
        assert_ne!(
            scoped_structured_session_id("agent:workflow:Planning", "schema-a"),
            scoped_structured_session_id("agent:workflow:Planning", "schema-b")
        );
    }

    #[test]
    fn lru_keeps_two_sessions_and_two_checkpoints_each() {
        let mut sessions = DeepSeekV4SessionStore::default();
        sessions.replace_stable_prompt(
            "a",
            checkpoint(DeepSeekV4CheckpointKind::StablePrompt, &[1], 1),
        );
        sessions.replace_stable_prompt(
            "b",
            checkpoint(DeepSeekV4CheckpointKind::StablePrompt, &[2], 2),
        );
        sessions.push_checkpoint(
            "a",
            checkpoint(DeepSeekV4CheckpointKind::Prompt, &[1, 3], 3),
        );
        sessions.push_checkpoint(
            "a",
            checkpoint(DeepSeekV4CheckpointKind::Generated, &[1, 3, 4], 4),
        );
        sessions.replace_stable_prompt(
            "c",
            checkpoint(DeepSeekV4CheckpointKind::StablePrompt, &[5], 5),
        );

        assert!(
            sessions
                .reusable_checkpoint("b", &[2, 9])
                .unwrap()
                .is_none()
        );
        assert!(
            sessions
                .reusable_checkpoint("c", &[5, 9])
                .unwrap()
                .is_some()
        );
        let (prefix, selected) = sessions
            .reusable_checkpoint("a", &[1, 3, 4, 6])
            .unwrap()
            .unwrap();
        assert_eq!(prefix, 3);
        assert_eq!(*selected.state(), 4);
    }

    #[test]
    fn stable_prompt_survives_prompt_and_generated_checkpoint_pressure() {
        let mut sessions = DeepSeekV4SessionStore::default();
        sessions.replace_stable_prompt(
            "agent",
            checkpoint(DeepSeekV4CheckpointKind::StablePrompt, &[1, 2], 1),
        );
        sessions.push_checkpoint(
            "agent",
            checkpoint(DeepSeekV4CheckpointKind::Prompt, &[1, 2, 3], 2),
        );
        sessions.push_checkpoint(
            "agent",
            checkpoint(DeepSeekV4CheckpointKind::Generated, &[1, 2, 3, 4], 3),
        );

        let (prefix, selected) = sessions
            .reusable_checkpoint("agent", &[1, 2, 9])
            .unwrap()
            .unwrap();
        assert_eq!(prefix, 2);
        assert_eq!(selected.kind(), DeepSeekV4CheckpointKind::StablePrompt);
    }
}
