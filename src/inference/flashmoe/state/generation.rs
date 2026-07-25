use super::*;

#[derive(Debug)]
pub(in crate::inference::flashmoe) struct FlashMoeGenerationState {
    pub(in crate::inference::flashmoe) session_id: Option<String>,
    pub(in crate::inference::flashmoe) prompt_tokens: Vec<u32>,
    pub(in crate::inference::flashmoe) kv_cache: KvCache,
    pub(in crate::inference::flashmoe) prefill_start: usize,
    pub(in crate::inference::flashmoe) cached_last_hidden: Option<Vec<f32>>,
    pub(in crate::inference::flashmoe) prompt_cache: Option<FlashMoeSessionState<KvCache>>,
    pub(in crate::inference::flashmoe) cached_recurrent:
        Option<FlashMoeLinearAttentionSessionSnapshot>,
    pub(in crate::inference::flashmoe) prompt_recurrent:
        Option<FlashMoeLinearAttentionSessionSnapshot>,
    pub(in crate::inference::flashmoe) generated_cache: Option<FlashMoeSessionState<KvCache>>,
    pub(in crate::inference::flashmoe) generated_recurrent:
        Option<FlashMoeLinearAttentionSessionSnapshot>,
    pub(in crate::inference::flashmoe) cache_source: PromptCacheSource,
    pub(in crate::inference::flashmoe) cache_restore_ms: u64,
    pub(in crate::inference::flashmoe) cache_miss_reason:
        Option<crate::inference::PromptCacheMissReason>,
    pub(in crate::inference::flashmoe) base_prefix_len: usize,
    pub(in crate::inference::flashmoe) base_cache: Option<FlashMoeSessionState<KvCache>>,
    pub(in crate::inference::flashmoe) base_recurrent:
        Option<FlashMoeLinearAttentionSessionSnapshot>,
    pub(in crate::inference::flashmoe) generated: Vec<u32>,
    pub(in crate::inference::flashmoe) max_tokens: usize,
    pub(in crate::inference::flashmoe) stopped: bool,
    pub(in crate::inference::flashmoe) stopped_by_terminal_tool_call: bool,
    pub(in crate::inference::flashmoe) stopped_by_constraint_payload_limit: bool,
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

    pub(crate) fn cache_miss_reason(&self) -> Option<crate::inference::PromptCacheMissReason> {
        self.cache_miss_reason
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

    pub(crate) fn stop_at_json_value(&mut self) {
        self.stopped = true;
        self.stopped_by_terminal_tool_call = false;
        self.stopped_by_constraint_payload_limit = false;
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
