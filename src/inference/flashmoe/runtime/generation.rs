use super::*;

impl FlashMoeEngine {
    pub fn set_metal_working_set_limit_bytes(&mut self, limit: usize) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return self.metal.inner.set_working_set_limit_bytes(limit);
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = limit;
            bail!("FlashMoe Metal resource policy requires Apple Silicon Metal")
        }
    }

    pub fn metal_resource_snapshot(&self) -> Option<FlashMoeMetalResourceSnapshot> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Some(self.metal.inner.resource_snapshot())
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            None
        }
    }

    pub fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationOutput> {
        let request = StructuredGenerationRequest::from_prompt(request);
        self.generate_structured(&request)
    }

    pub fn generate_raw(&mut self, request: &GenerationRequest) -> Result<GenerationOutput> {
        let mut request = StructuredGenerationRequest::from_prompt(request);
        request.raw_prompt = true;
        request.add_generation_prompt = false;
        self.generate_structured(&request)
    }

    pub fn generate_structured(
        &mut self,
        request: &StructuredGenerationRequest,
    ) -> Result<GenerationOutput> {
        Ok(self.generate_structured_inner(request, None, false)?.output)
    }

    pub(crate) fn supports_session_snapshots(&self) -> bool {
        true
    }

    pub(crate) fn supports_thinking(&self) -> bool {
        self.model_layout.family.supports_thinking()
    }

    pub(crate) fn requires_exact_session_prefix(&self) -> bool {
        self.executor.is_deepseek_v4()
    }

    /// Render and tokenize the exact prompt used by structured generation.
    pub fn measure_structured_prompt(
        &self,
        request: &StructuredGenerationRequest,
    ) -> Result<usize> {
        Ok(self.structured_prompt_tokens(request)?.1.len())
    }

    /// Resolve a raw-text prefix whose standalone tokenization is exactly the
    /// requested leading slice of the full prompt. This avoids treating an
    /// arbitrary byte prefix as a reusable session boundary when a tokenizer
    /// merge crosses that boundary.
    pub fn exact_raw_prompt_prefix(
        &self,
        full_prompt: &str,
        prefix_tokens: usize,
    ) -> Result<(String, usize)> {
        let full_tokens = self.tokenizer.encode(full_prompt)?;
        if prefix_tokens == 0 || prefix_tokens >= full_tokens.len() {
            bail!(
                "raw prompt parity prefix tokens {prefix_tokens} must be between 1 and {}",
                full_tokens.len().saturating_sub(1)
            );
        }
        let decoded = self.tokenizer.decode(&full_tokens[..prefix_tokens])?;
        if full_prompt.starts_with(&decoded)
            && self.tokenizer.encode(&decoded)? == full_tokens[..prefix_tokens]
        {
            return Ok((decoded, prefix_tokens));
        }
        for end in full_prompt
            .char_indices()
            .skip(1)
            .map(|(index, _)| index)
            .chain(std::iter::once(full_prompt.len()))
        {
            let candidate = &full_prompt[..end];
            let candidate_tokens = self.tokenizer.encode(candidate)?;
            if candidate_tokens.len() == prefix_tokens && full_tokens.starts_with(&candidate_tokens)
            {
                return Ok((candidate.to_string(), prefix_tokens));
            }
        }
        bail!(
            "raw prompt has no exact reusable text boundary at token {prefix_tokens}; choose a different --prefill-parity-prefix-tokens value"
        )
    }

    fn structured_prompt_tokens(
        &self,
        request: &StructuredGenerationRequest,
    ) -> Result<(String, Vec<u32>)> {
        if !self.executor.is_deepseek_v4() {
            // Compile during preflight as well as generation so unsupported schema features
            // fail before any model work or durable invocation accounting begins.
            let _ = NativeToolConstraint::compile_with_terminal_tools(
                request.tool_constraint_mode,
                &request.tools,
                &request.terminal_tool_names,
            )?;
        }
        if !request.raw_prompt {
            return self.tokenizer.render_and_encode_chat_prompt(
                &request.messages,
                &request.tools,
                request.add_generation_prompt,
                request.enable_thinking && self.supports_thinking(),
            );
        }
        let prompt = self.render_structured_prompt(request)?;
        let prompt_tokens = self.tokenizer.encode(&prompt)?;
        Ok((prompt, prompt_tokens))
    }

    pub(crate) fn rendered_structured_prompt_identity(
        &self,
        request: &StructuredGenerationRequest,
    ) -> Result<(String, usize)> {
        let prompt = self.render_structured_prompt(request)?;
        let bytes = prompt.as_bytes();
        Ok((format!("{:x}", Sha256::digest(bytes)), bytes.len()))
    }

    fn render_structured_prompt(&self, request: &StructuredGenerationRequest) -> Result<String> {
        if request.raw_prompt {
            if !request.tools.is_empty() {
                bail!("raw Flash-MoE generation does not support tools");
            }
            return match request.messages.as_slice() {
                [
                    ChatMessage {
                        content: ChatMessageContent::Text(prompt),
                        ..
                    },
                ] => Ok(prompt.clone()),
                _ => bail!("raw Flash-MoE generation requires exactly one text prompt"),
            };
        }
        self.tokenizer
            .apply_chat_template_to_messages_with_thinking(
                &request.messages,
                &request.tools,
                request.add_generation_prompt,
                request.enable_thinking && self.supports_thinking(),
            )
    }

    fn stable_base_prefix_len(
        &self,
        request: &StructuredGenerationRequest,
        prompt_tokens: &[u32],
    ) -> Result<usize> {
        if request.raw_prompt
            || !matches!(
                request.messages.first().map(|message| &message.role),
                Some(ChatRole::System)
            )
        {
            return Ok(0);
        }
        let mut base = request.clone();
        base.messages.truncate(1);
        base.add_generation_prompt = false;
        base.max_tokens = 0;
        let (_, rendered_base) = self.structured_prompt_tokens(&base)?;
        Ok(common_token_prefix_len(prompt_tokens, &rendered_base))
    }

    fn deepseek_stable_prompt_prefix_len(
        &self,
        request: &StructuredGenerationRequest,
        prompt_tokens: &[u32],
    ) -> Result<usize> {
        if request.raw_prompt || !request.add_generation_prompt {
            return Ok(prompt_tokens.len());
        }
        if request.messages.is_empty() {
            return Ok(0);
        }
        let mut stable = request.clone();
        stable.messages.truncate(1);
        stable.add_generation_prompt = false;
        stable.max_tokens = 0;
        let (_, stable_tokens) = self.structured_prompt_tokens(&stable)?;
        Ok(common_token_prefix_len(prompt_tokens, &stable_tokens))
    }

    pub fn generate_in_session(
        &mut self,
        session_id: &str,
        request: &GenerationRequest,
    ) -> Result<GenerationOutput> {
        if session_id.is_empty() {
            return self.generate(request);
        }
        let request = StructuredGenerationRequest::from_prompt(request);
        self.generate_structured_in_session(session_id, &request)
    }

    pub fn generate_structured_in_session(
        &mut self,
        session_id: &str,
        request: &StructuredGenerationRequest,
    ) -> Result<GenerationOutput> {
        if session_id.is_empty() {
            return self.generate_structured(request);
        }
        Ok(self
            .generate_structured_inner_with_session(request, Some(session_id), None, false)?
            .output)
    }

    pub fn generate_timed(&mut self, request: &GenerationRequest) -> Result<TimedGenerationOutput> {
        let request = StructuredGenerationRequest::from_prompt(request);
        self.generate_structured_timed(&request)
    }

    pub fn generate_timed_with_progress<F>(
        &mut self,
        request: &GenerationRequest,
        mut progress: F,
    ) -> Result<TimedGenerationOutput>
    where
        F: FnMut(String),
    {
        let request = StructuredGenerationRequest::from_prompt(request);
        self.generate_structured_timed_with_progress(&request, &mut progress)
    }

    pub fn generate_raw_timed_with_progress<F>(
        &mut self,
        request: &GenerationRequest,
        mut progress: F,
    ) -> Result<TimedGenerationOutput>
    where
        F: FnMut(String),
    {
        let mut request = StructuredGenerationRequest::from_prompt(request);
        request.raw_prompt = true;
        request.add_generation_prompt = false;
        self.generate_structured_timed_with_progress(&request, &mut progress)
    }

    pub fn generate_structured_timed(
        &mut self,
        request: &StructuredGenerationRequest,
    ) -> Result<TimedGenerationOutput> {
        let mut timing = self.new_generation_timing();
        self.generate_structured_inner_with_session(request, None, Some(&mut timing), true)
    }

    pub fn generate_structured_timed_with_progress(
        &mut self,
        request: &StructuredGenerationRequest,
        progress: &mut dyn FnMut(String),
    ) -> Result<TimedGenerationOutput> {
        let mut timing = self.new_generation_timing();
        let progress = Some(Rc::new(RefCell::new(progress)));
        self.generate_structured_inner_with_session_progress(
            request,
            None,
            Some(&mut timing),
            progress,
            true,
        )
    }

    pub fn generate_structured_summary_timed(
        &mut self,
        request: &StructuredGenerationRequest,
    ) -> Result<TimedGenerationOutput> {
        let mut timing = self.new_generation_timing();
        self.generate_structured_inner_with_session(request, None, Some(&mut timing), false)
    }

    pub fn generate_structured_summary_timed_in_session(
        &mut self,
        session_id: &str,
        request: &StructuredGenerationRequest,
    ) -> Result<TimedGenerationOutput> {
        let mut timing = self.new_generation_timing();
        self.generate_structured_inner_with_session(
            request,
            (!session_id.is_empty()).then_some(session_id),
            Some(&mut timing),
            false,
        )
    }

    pub fn generate_structured_summary_timed_with_progress(
        &mut self,
        request: &StructuredGenerationRequest,
        progress: &mut dyn FnMut(String),
    ) -> Result<TimedGenerationOutput> {
        let mut timing = self.new_generation_timing();
        let progress = Some(Rc::new(RefCell::new(progress)));
        self.generate_structured_inner_with_session_progress(
            request,
            None,
            Some(&mut timing),
            progress,
            false,
        )
    }

    pub fn generate_structured_summary_timed_in_session_with_progress(
        &mut self,
        session_id: &str,
        request: &StructuredGenerationRequest,
        progress: &mut dyn FnMut(String),
    ) -> Result<TimedGenerationOutput> {
        let mut timing = self.new_generation_timing();
        let progress = Some(Rc::new(RefCell::new(progress)));
        self.generate_structured_inner_with_session_progress(
            request,
            (!session_id.is_empty()).then_some(session_id),
            Some(&mut timing),
            progress,
            false,
        )
    }

    fn new_generation_timing(&self) -> FlashMoeGenerationTiming {
        FlashMoeGenerationTiming {
            model: self.plan.model.clone(),
            dimensions: self.model_dimensions(),
            prefill_or_ttft_tokens: 0,
            prefill_or_ttft_wall: Duration::ZERO,
            decode_tokens: 0,
            decode_wall: Duration::ZERO,
            tokens: Vec::new(),
            total_wall: Duration::ZERO,
        }
    }

    fn generate_structured_inner(
        &mut self,
        request: &StructuredGenerationRequest,
        timing: Option<&mut FlashMoeGenerationTiming>,
        detailed_timing: bool,
    ) -> Result<TimedGenerationOutput> {
        self.generate_structured_inner_with_session_progress(
            request,
            None,
            timing,
            None,
            detailed_timing,
        )
    }

    fn generate_structured_inner_with_session(
        &mut self,
        request: &StructuredGenerationRequest,
        session_id: Option<&str>,
        timing: Option<&mut FlashMoeGenerationTiming>,
        detailed_timing: bool,
    ) -> Result<TimedGenerationOutput> {
        self.generate_structured_inner_with_session_progress(
            request,
            session_id,
            timing,
            None,
            detailed_timing,
        )
    }

    fn generate_structured_inner_with_session_progress(
        &mut self,
        request: &StructuredGenerationRequest,
        session_id: Option<&str>,
        mut timing: Option<&mut FlashMoeGenerationTiming>,
        progress: GenerationProgress<'_>,
        detailed_timing: bool,
    ) -> Result<TimedGenerationOutput> {
        let generation_started = Instant::now();
        let render_started = Instant::now();
        let prompt = self.render_structured_prompt(request)?;
        let render_elapsed = render_started.elapsed();
        let encode_started = Instant::now();
        let prompt_tokens = self.tokenizer.encode(&prompt)?;
        let encode_elapsed = encode_started.elapsed();
        let deepseek_v4 = self.executor.is_deepseek_v4();
        let mut tool_constraint = if deepseek_v4 {
            None
        } else {
            NativeToolConstraint::compile_with_terminal_tools(
                request.tool_constraint_mode,
                &request.tools,
                &request.terminal_tool_names,
            )?
        };
        let deepseek_stable_prefix_len = if deepseek_v4 {
            self.deepseek_stable_prompt_prefix_len(request, &prompt_tokens)?
        } else {
            0
        };
        let base_prefix_len = if deepseek_v4 {
            0
        } else {
            self.stable_base_prefix_len(request, &prompt_tokens)?
        };
        let max_tokens = request.max_tokens.max(0) as usize;
        validate_context_capacity(prompt_tokens.len(), max_tokens, request.context_size)?;
        if let Some(graph) = self.executor.deepseek_v4_graph() {
            let capacity = prompt_tokens
                .len()
                .checked_add(max_tokens)
                .context("DeepSeek V4 request context capacity overflow")?
                .max(1);
            self.metal.prepare_deepseek_v4_state(graph, capacity)?;
        }
        let deepseek_restore_started = Instant::now();
        let deepseek_reuse = if deepseek_v4 {
            session_id
                .map(|session_id| {
                    self.deepseek_sessions
                        .reusable_checkpoint(session_id, &prompt_tokens)
                        .and_then(|checkpoint| {
                            checkpoint
                                .map(|(prefix, checkpoint)| {
                                    self.metal
                                        .restore_deepseek_v4_session_state(checkpoint.state())?;
                                    Ok((prefix, checkpoint.last_hidden().to_vec()))
                                })
                                .transpose()
                        })
                })
                .transpose()?
                .flatten()
        } else {
            None
        };
        if let Some(glm) = self.config.glm.as_ref()
            && glm.index_topk > 0
        {
            let required_tokens = prompt_tokens
                .len()
                .checked_add(max_tokens)
                .context("GLM context token count overflow")?;
            if required_tokens > glm.index_topk {
                bail!(
                    "GLM-5.2 full-causal MLA baseline is validated through index_topk={} tokens, but this request needs {required_tokens}; DSA selection is not implemented",
                    glm.index_topk
                );
            }
        }
        let generation_span = trace_span!(
            target: "flashmoe::perf",
            "generation",
            model = %self.plan.model,
            prompt_tokens = prompt_tokens.len(),
            max_tokens = request.max_tokens.max(0),
            raw_prompt = request.raw_prompt
        );
        let _generation_span = generation_span.enter();
        report_generation_progress(&progress, || {
            format!(
                "rendered prompt chars={} tokens={} render_ms={} encode_ms={}",
                prompt.len(),
                prompt_tokens.len(),
                render_elapsed.as_millis(),
                encode_elapsed.as_millis()
            )
        });
        debug!(
            target: "flashmoe::lifecycle",
            "flashmoe: rendered prompt chars={} tokens={} render_ms={} encode_ms={} tools={} session={}",
            prompt.len(),
            prompt_tokens.len(),
            render_elapsed.as_millis(),
            encode_elapsed.as_millis(),
            request.tools.len(),
            session_id.unwrap_or("<none>")
        );
        let mut generation = if deepseek_v4 {
            let (prefill_start, cached_last_hidden, cache_source, restore_ms) =
                if let Some((prefix, hidden)) = deepseek_reuse {
                    (
                        prefix,
                        (prefix == prompt_tokens.len()).then_some(hidden),
                        PromptCacheSource::MemorySession,
                        u64::try_from(deepseek_restore_started.elapsed().as_millis())
                            .unwrap_or(u64::MAX),
                    )
                } else {
                    (0, None, PromptCacheSource::None, 0)
                };
            FlashMoeSessionCache::begin_external_prefix_generation(
                prompt_tokens,
                prefill_start,
                cached_last_hidden,
                max_tokens,
                self.config.num_hidden_layers,
                cache_source,
                restore_ms,
            )?
        } else {
            self.session_cache.begin_generation_with_base(
                session_id,
                prompt_tokens,
                base_prefix_len,
                max_tokens,
                self.config.num_hidden_layers,
            )
        };
        let prefill_start = generation.prefill_start();
        let prompt_len = generation.prompt_len();
        let prompt_cache_source = generation.cache_source();
        let prompt_cache_restore_ms = generation.cache_restore_ms();
        if prefill_start > 0 {
            debug!(
                target: "flashmoe::lifecycle",
                "flashmoe: reusing session cache prefix_tokens={} prompt_tokens={}",
                prefill_start, prompt_len
            );
        }
        if !deepseek_v4 {
            if prefill_start == 0 {
                self.metal.reset_linear_attention_state()?;
            } else {
                let recurrent = generation
                    .take_cached_recurrent()
                    .context("session cache entry is missing the Metal recurrent-state snapshot")?;
                self.metal
                    .restore_linear_attention_session_state(&recurrent)?;
            }
        }
        let prefill_resources_before = self.metal_resource_snapshot();
        let prefill_or_ttft_started = Instant::now();
        let mut deepseek_stable_checkpoint = None;
        let prefill_hidden = if prefill_start == prompt_len {
            debug!(
                target: "flashmoe::lifecycle",
                "flashmoe: prompt prefill fully cached tokens={}",
                prompt_len
            );
            generation
                .take_cached_last_hidden()
                .context("session cache entry is missing the final hidden state")?
        } else {
            let prefill_started = Instant::now();
            report_generation_progress(&progress, || {
                format!(
                    "prefill begin start_token={} remaining_tokens={}",
                    prefill_start,
                    prompt_len.saturating_sub(prefill_start)
                )
            });
            debug!(
                target: "flashmoe::lifecycle",
                "flashmoe: prefill begin start_token={} remaining_tokens={}",
                prefill_start,
                prompt_len.saturating_sub(prefill_start)
            );
            let mut cursor = prefill_start;
            let mut hidden = None;
            let base_prefix_len = generation.base_prefix_len();
            if cursor < base_prefix_len {
                hidden = Some({
                    let (prompt_tokens, _, kv_cache) = generation.prefill_inputs();
                    let detailed = if detailed_timing {
                        timing.as_deref_mut()
                    } else {
                        None
                    };
                    self.prefill_range(
                        prompt_tokens,
                        cursor,
                        base_prefix_len,
                        kv_cache,
                        request.prefill_mode,
                        request.prefill_chunk_tokens,
                        request.prefill_state_summary,
                        detailed,
                        progress.clone(),
                    )?
                });
                let recurrent = self.metal.capture_linear_attention_session_state()?;
                generation.capture_base_cache(
                    hidden
                        .as_ref()
                        .expect("base prefill produced hidden")
                        .clone(),
                    recurrent,
                );
                cursor = base_prefix_len;
            }
            if deepseek_v4
                && session_id.is_some()
                && cursor < deepseek_stable_prefix_len
                && deepseek_stable_prefix_len < prompt_len
            {
                hidden = Some({
                    let (prompt_tokens, _, kv_cache) = generation.prefill_inputs();
                    let detailed = if detailed_timing {
                        timing.as_deref_mut()
                    } else {
                        None
                    };
                    self.prefill_range(
                        prompt_tokens,
                        cursor,
                        deepseek_stable_prefix_len,
                        kv_cache,
                        request.prefill_mode,
                        request.prefill_chunk_tokens,
                        request.prefill_state_summary,
                        detailed,
                        progress.clone(),
                    )?
                });
                deepseek_stable_checkpoint = Some(DeepSeekV4SessionCheckpoint::new(
                    DeepSeekV4CheckpointKind::StablePrompt,
                    generation.prompt_tokens_through(deepseek_stable_prefix_len),
                    hidden
                        .as_ref()
                        .expect("stable DeepSeek prefill produced hidden")
                        .clone(),
                    self.metal.capture_deepseek_v4_session_state()?,
                ));
                cursor = deepseek_stable_prefix_len;
            }
            if cursor < prompt_len {
                hidden = Some({
                    let (prompt_tokens, _, kv_cache) = generation.prefill_inputs();
                    let detailed = if detailed_timing {
                        timing.as_deref_mut()
                    } else {
                        None
                    };
                    self.prefill_range(
                        prompt_tokens,
                        cursor,
                        prompt_len,
                        kv_cache,
                        request.prefill_mode,
                        request.prefill_chunk_tokens,
                        request.prefill_state_summary,
                        detailed,
                        progress.clone(),
                    )?
                });
            }
            let hidden = hidden.context("FlashMoe prefill produced no final hidden state")?;
            report_generation_progress(&progress, || {
                format!(
                    "prefill complete tokens={} elapsed_ms={}",
                    prompt_len.saturating_sub(prefill_start),
                    prefill_started.elapsed().as_millis()
                )
            });
            debug!(
                target: "flashmoe::lifecycle",
                "flashmoe: prefill complete tokens={} elapsed_ms={}",
                prompt_len.saturating_sub(prefill_start),
                prefill_started.elapsed().as_millis()
            );
            hidden
        };
        if deepseek_v4 {
            if let Some(session_id) = session_id {
                if let Some(checkpoint) = deepseek_stable_checkpoint {
                    self.deepseek_sessions
                        .replace_stable_prompt(session_id, checkpoint);
                } else if deepseek_stable_prefix_len == prompt_len {
                    let checkpoint = DeepSeekV4SessionCheckpoint::new(
                        DeepSeekV4CheckpointKind::StablePrompt,
                        generation.checkpoint_tokens(0),
                        prefill_hidden.clone(),
                        self.metal.capture_deepseek_v4_session_state()?,
                    );
                    self.deepseek_sessions
                        .replace_stable_prompt(session_id, checkpoint);
                }
                if deepseek_stable_prefix_len != prompt_len {
                    let checkpoint = DeepSeekV4SessionCheckpoint::new(
                        DeepSeekV4CheckpointKind::Prompt,
                        generation.checkpoint_tokens(0),
                        prefill_hidden.clone(),
                        self.metal.capture_deepseek_v4_session_state()?,
                    );
                    self.deepseek_sessions
                        .push_checkpoint(session_id, checkpoint);
                }
            }
        } else if generation.requires_prompt_snapshot() {
            let recurrent = self.metal.capture_linear_attention_session_state()?;
            generation.capture_prompt_cache(prefill_hidden.clone(), recurrent);
        }

        let prefill_state = if request.prefill_state_summary {
            if deepseek_v4 {
                bail!(
                    "prefill state summaries currently qualify Qwen linear/full-attention graphs only"
                );
            }
            let (full_attention_kv_sha256, router_recurrent_trace_sha256) =
                generation.prefill_state_sha256();
            let (full_attention_kv_layer_sha256, router_recurrent_layer_sha256) =
                generation.prefill_layer_state_sha256();
            let recurrent = self.metal.capture_linear_attention_session_state()?;
            Some(NativePrefillStateStats {
                final_hidden_sha256: f32_values_sha256(
                    b"pb.flashmoe.final-prefill-hidden.v1\0",
                    &prefill_hidden,
                ),
                full_attention_kv_sha256,
                router_recurrent_trace_sha256,
                linear_attention_state_sha256: recurrent.state_sha256(),
                full_attention_kv_layer_sha256,
                router_recurrent_layer_sha256,
                linear_attention_layer_sha256: recurrent.layer_state_sha256(),
            })
        } else {
            None
        };

        let prefill_resources_after = self.metal_resource_snapshot();
        let prefill_resources = native_prefill_resource_delta(
            prefill_resources_before.as_ref(),
            prefill_resources_after.as_ref(),
        );
        let prefill_wall = prefill_or_ttft_started.elapsed();
        let mut sampler = TokenSampler::new(request.temperature, request.top_k, request.seed);
        if tool_constraint.is_some() {
            sampler.widen_candidates(128);
        }
        if generation.should_sample_first() {
            let sample_started = Instant::now();
            report_generation_progress(&progress, || "first-token sampling begin".to_string());
            debug!(target: "flashmoe::lifecycle", "flashmoe: first-token sampling begin");
            let token = {
                let (prompt_tokens, generated) = generation.sample_inputs();
                self.sample_from_hidden(
                    &mut sampler,
                    &prefill_hidden,
                    prompt_tokens,
                    generated,
                    request.trace_candidates,
                    &progress,
                    tool_constraint.as_mut(),
                )?
            };
            report_generation_progress(&progress, || {
                format!(
                    "first-token sampling complete token={} elapsed_ms={}",
                    token,
                    sample_started.elapsed().as_millis()
                )
            });
            debug!(
                target: "flashmoe::lifecycle",
                "flashmoe: first-token sampling complete token={} elapsed_ms={}",
                token,
                sample_started.elapsed().as_millis()
            );
            if detailed_timing
                && let Some(timing) = timing.as_deref_mut()
                && let Some(last) = timing.tokens.last_mut()
            {
                let elapsed = sample_started.elapsed();
                last.buckets.sampling += elapsed;
                last.buckets.total_wall += elapsed;
                last.sampled_token = Some(token);
            }
            let payload_limit_stop = tool_constraint
                .as_mut()
                .and_then(|constraint| constraint.take_payload_limit_stop())
                .is_some();
            if payload_limit_stop {
                generation.stop_at_constraint_payload_limit();
            } else {
                let terminal_tool_call = if let Some(constraint) = tool_constraint.as_ref() {
                    let (_, generated) = generation.sample_inputs();
                    constraint.should_stop_after_token(&self.tokenizer, generated, token)?
                } else {
                    false
                };
                generation.record_sampled_token(
                    token,
                    self.tokenizer.is_eos(token),
                    terminal_tool_call,
                );
            }
        }
        let prefill_or_ttft_wall = prefill_or_ttft_started.elapsed();
        let decode_phase_started = Instant::now();
        let mut decode_tokens = 0usize;
        let mut evaluated_generated_tokens = 0usize;
        let mut generated_head_hidden = None;
        let report_decode_progress = progress.is_some()
            || tracing::enabled!(target: "flashmoe::perf", tracing::Level::TRACE);
        while generation.should_decode() {
            let generated_len = generation.generated_len();
            let max_tokens = generation.max_tokens();
            let position = generation.decode_inputs()?.3;
            report_generation_progress(&progress, || {
                format!(
                    "decode begin generated={}/{} position={}",
                    generated_len, max_tokens, position
                )
            });
            trace!(
                target: "flashmoe::perf",
                "flashmoe: decode begin generated={}/{} position={}",
                generated_len, max_tokens, position
            );
            let decode_started = OptionalInstant::now(report_decode_progress);
            let sampled = {
                let (prompt_tokens, generated, kv_cache, position) = generation.decode_inputs()?;
                let detailed = if detailed_timing {
                    timing.as_deref_mut()
                } else {
                    None
                };
                self.sample_next_token(
                    &mut sampler,
                    prompt_tokens,
                    generated,
                    kv_cache,
                    position,
                    MropePosition::text(position),
                    detailed,
                    request.trace_candidates,
                    progress.clone(),
                    tool_constraint.as_mut(),
                )?
            };
            let token = sampled.token;
            evaluated_generated_tokens = generated_len;
            generated_head_hidden = Some(sampled.hidden);
            decode_tokens = decode_tokens.saturating_add(1);
            report_generation_progress(&progress, || {
                format!(
                    "decode complete generated={}/{} token={} elapsed_ms={}",
                    generated_len + 1,
                    max_tokens,
                    token,
                    decode_started.elapsed().as_millis()
                )
            });
            trace!(
                target: "flashmoe::perf",
                "flashmoe: decode complete generated={}/{} token={} elapsed_ms={}",
                generated_len + 1,
                max_tokens,
                token,
                decode_started.elapsed().as_millis()
            );
            let payload_limit_stop = tool_constraint
                .as_mut()
                .and_then(|constraint| constraint.take_payload_limit_stop())
                .is_some();
            if payload_limit_stop {
                generation.stop_at_constraint_payload_limit();
            } else {
                let terminal_tool_call = if let Some(constraint) = tool_constraint.as_ref() {
                    let (_, generated) = generation.sample_inputs();
                    constraint.should_stop_after_token(&self.tokenizer, generated, token)?
                } else {
                    false
                };
                generation.record_sampled_token(
                    token,
                    self.tokenizer.is_eos(token),
                    terminal_tool_call,
                );
            }
        }
        let decode_wall = decode_phase_started.elapsed();

        if let Some(last_hidden) = generated_head_hidden {
            if deepseek_v4 && request.raw_prompt {
                if let Some(session_id) = session_id {
                    let checkpoint = DeepSeekV4SessionCheckpoint::new(
                        DeepSeekV4CheckpointKind::Generated,
                        generation.checkpoint_tokens(evaluated_generated_tokens),
                        last_hidden,
                        self.metal.capture_deepseek_v4_session_state()?,
                    );
                    self.deepseek_sessions
                        .push_checkpoint(session_id, checkpoint);
                }
            } else if generation.requires_prompt_snapshot() {
                let recurrent = self.metal.capture_linear_attention_session_state()?;
                generation.capture_generated_cache(
                    evaluated_generated_tokens,
                    last_hidden,
                    recurrent,
                );
            }
        }

        if !deepseek_v4 {
            self.session_cache.commit_generation(&mut generation)?;
        }

        let stopped_by_terminal_tool_call = generation.stopped_by_terminal_tool_call();
        let stopped_by_constraint_payload_limit = generation.stopped_by_constraint_payload_limit();
        let generated = generation.into_generated();
        let decoded = self.tokenizer.decode(&generated)?;
        let finish_reason = if stopped_by_constraint_payload_limit {
            GenerationFinishReason::MaxTokens
        } else {
            generation_finish_reason(generated.len(), max_tokens)
        };
        let parseable_decoded = if stopped_by_terminal_tool_call {
            close_unclosed_qwen_terminal_tool_call(&decoded)
        } else {
            Cow::Borrowed(decoded.as_str())
        };
        let (content, tool_calls) = self.parse_native_tool_output(
            &parseable_decoded,
            finish_reason == GenerationFinishReason::MaxTokens,
        )?;
        let tool_constraints =
            tool_constraint
                .as_ref()
                .map(|constraint| NativeToolConstraintStats {
                    mode: constraint.mode(),
                    schema_sha256: constraint.schema_sha256().to_string(),
                    rejected_candidates: constraint.rejected_candidates(),
                    terminal_state: constraint.terminal_state(&decoded).to_string(),
                });
        let performance = NativeGenerationStats {
            fresh_prefill_tokens: prompt_len.saturating_sub(prefill_start),
            cached_tokens: prefill_start,
            prefill_wall_ms: duration_millis(prefill_wall),
            prefill_tokens_per_second: tokens_per_second(
                prompt_len.saturating_sub(prefill_start),
                prefill_wall,
            ),
            prefill_metal_commands: prefill_resources.metal_commands,
            prefill_host_upload_bytes: prefill_resources.host_upload_bytes,
            prefill_host_readback_bytes: prefill_resources.host_readback_bytes,
            decode_tokens,
            decode_wall_ms: duration_millis(decode_wall),
            decode_tokens_per_second: tokens_per_second(decode_tokens, decode_wall),
            model_family: format!("{:?}", self.model_layout.family),
            active_experts_per_token: nonzero_usize(self.model_layout.scheduled_active_experts),
            expert_strategy: match self.expert_access {
                FlashMoeExpertAccessCapability::ParallelPositionedWholeExpertReads => {
                    "streamed_parallel_pread"
                }
                FlashMoeExpertAccessCapability::ResidentMappedWholeExpertSlots => {
                    "resident_complete_corpus"
                }
            }
            .to_string(),
            prefill_command_kind: if prefill_start == prompt_len {
                "cache_only"
            } else if deepseek_v4
                && deepseek_v4_uses_batch_prefill(prompt_len.saturating_sub(prefill_start))
            {
                "deepseek_layer_major_batch"
            } else if self.qwen_prefill_graph.supports_layer_major()
                && (request.prefill_mode == NativePrefillMode::LayerMajor
                    || (request.prefill_mode == NativePrefillMode::Auto
                        && qwen_prefill_chunk_tokens(
                            self.qwen_prefill_graph,
                            prompt_len.saturating_sub(prefill_start),
                            self.metal_resource_snapshot().as_ref(),
                        )
                        .is_some()))
            {
                "qwen_layer_major_matrix"
            } else {
                "scalar_token"
            }
            .to_string(),
            thinking_enabled: request.enable_thinking && self.supports_thinking(),
            prefill_state,
        };
        let output = GenerationOutput {
            content,
            tool_calls,
            finish_reason,
            prompt_tokens: prompt_len,
            generated_tokens: generated.len(),
            prompt_cache: PromptCacheStats {
                source: prompt_cache_source,
                cached_tokens: prefill_start,
                prefilled_tokens: prompt_len.saturating_sub(prefill_start),
                restore_ms: prompt_cache_restore_ms,
            },
            tool_constraints,
            performance,
        };
        let total_wall = generation_started.elapsed();
        info!(
            "flashmoe: generation complete generated_tokens={} total_ms={}",
            generated.len(),
            total_wall.as_millis()
        );
        if let Some(timing) = timing {
            timing.prefill_or_ttft_tokens = prompt_len.saturating_sub(prefill_start);
            timing.prefill_or_ttft_wall = prefill_or_ttft_wall;
            timing.decode_tokens = decode_tokens;
            timing.decode_wall = decode_wall;
            timing.total_wall = total_wall;
            return Ok(TimedGenerationOutput {
                output,
                timing: timing.clone(),
            });
        }
        let mut timing = self.new_generation_timing();
        timing.prefill_or_ttft_tokens = prompt_len.saturating_sub(prefill_start);
        timing.prefill_or_ttft_wall = prefill_or_ttft_wall;
        timing.decode_tokens = decode_tokens;
        timing.decode_wall = decode_wall;
        timing.total_wall = total_wall;
        Ok(TimedGenerationOutput { output, timing })
    }

    fn prefill_range(
        &mut self,
        prompt_tokens: &[u32],
        start_position: usize,
        end_position: usize,
        kv_cache: &mut KvCache,
        prefill_mode: NativePrefillMode,
        prefill_chunk_tokens: Option<usize>,
        record_prefill_state: bool,
        mut timing: Option<&mut FlashMoeGenerationTiming>,
        progress: GenerationProgress<'_>,
    ) -> Result<Vec<f32>> {
        if start_position > end_position || end_position > prompt_tokens.len() {
            bail!(
                "prefill range {start_position}..{end_position} exceeds prompt length {}",
                prompt_tokens.len()
            );
        }
        let batch_tokens = end_position.saturating_sub(start_position);
        if let Some(graph) = self.executor.deepseek_v4_graph().cloned()
            && deepseek_v4_uses_batch_prefill(batch_tokens)
        {
            if end_position == 0 {
                bail!("cannot generate from an empty DeepSeek V4 prompt");
            }
            let started = Instant::now();
            report_generation_progress(&progress, || {
                format!("prefill batch begin start={start_position} tokens={batch_tokens}")
            });
            for (position, token) in prompt_tokens[start_position..end_position]
                .iter()
                .copied()
                .enumerate()
            {
                kv_cache.record_prompt_token_record(FlashMoePromptTokenRecord::new(
                    start_position + position,
                    token,
                ))?;
            }
            let hidden = self.metal.deepseek_v4_prefill(
                &graph,
                &mut self.scheduler,
                &prompt_tokens[start_position..end_position],
                start_position,
            )?;
            let elapsed = started.elapsed();
            report_generation_progress(&progress, || {
                format!(
                    "prefill batch complete processed={batch_tokens} remaining=0 position={} elapsed_ms={}",
                    end_position - 1,
                    elapsed.as_millis()
                )
            });
            if let Some(timing) = timing.as_deref_mut() {
                let mut token_timing = FlashMoeTokenTiming::new(
                    end_position - 1,
                    end_position - 1,
                    FlashMoeTokenPhase::Prefill,
                    prompt_tokens[end_position - 1],
                );
                token_timing.buckets.total_wall = elapsed;
                timing.tokens.push(token_timing);
            }
            return Ok(hidden);
        }
        if prefill_mode == NativePrefillMode::LayerMajor
            && batch_tokens > 0
            && !self.qwen_prefill_graph.supports_layer_major()
        {
            bail!(
                "explicit Qwen layer-major prefill is unavailable for prepared graph {}; it requires Qwen3-Coder-Next with resident affine-Q4 dense weights and fixed affine-Q4 expert slots",
                self.qwen_prefill_graph.as_str()
            );
        }
        let qwen_chunk_tokens = if let Some(chunk_tokens) = prefill_chunk_tokens {
            if prefill_mode != NativePrefillMode::LayerMajor {
                bail!("explicit Qwen prefill chunks require layer-major prefill mode");
            }
            if chunk_tokens == 0 || self.model_layout.family == QwenMoeFamily::DeepSeekV4Flash {
                bail!("explicit Qwen prefill chunks require positive non-DeepSeek geometry");
            }
            (batch_tokens > 0).then_some(chunk_tokens.min(batch_tokens))
        } else {
            match prefill_mode {
                NativePrefillMode::Auto => qwen_prefill_chunk_tokens(
                    self.qwen_prefill_graph,
                    batch_tokens,
                    self.metal_resource_snapshot().as_ref(),
                ),
                NativePrefillMode::LayerMajor if batch_tokens > 0 => qwen_prefill_chunk_tokens(
                    self.qwen_prefill_graph,
                    batch_tokens.max(QWEN_BATCH_PREFILL_MIN_TOKENS),
                    self.metal_resource_snapshot().as_ref(),
                )
                .map(|chunk_tokens| chunk_tokens.min(batch_tokens)),
                NativePrefillMode::LayerMajor | NativePrefillMode::Scalar => None,
            }
        };
        if prefill_mode == NativePrefillMode::LayerMajor
            && batch_tokens > 0
            && qwen_chunk_tokens.is_none()
        {
            bail!(
                "explicit Qwen layer-major prefill cannot reserve the minimum {QWEN_BATCH_PREFILL_MIN_TOKENS}-row graph within the prepared Metal working-set and session-reserve limits"
            );
        }
        if let Some(chunk_tokens) = qwen_chunk_tokens {
            return self.prefill_qwen_chunks(
                prompt_tokens,
                start_position,
                end_position,
                chunk_tokens,
                kv_cache,
                record_prefill_state,
                timing,
                progress,
            );
        }
        let mut last_hidden = None;
        let report_prefill_progress = progress.is_some()
            || tracing::enabled!(target: "flashmoe::lifecycle", tracing::Level::DEBUG);
        let progress_started = OptionalInstant::now(report_prefill_progress);
        let mut last_progress = OptionalInstant::now(report_prefill_progress);
        for (position, token) in prompt_tokens
            .iter()
            .copied()
            .enumerate()
            .take(end_position)
            .skip(start_position)
        {
            kv_cache.record_prompt_token_record(FlashMoePromptTokenRecord::new(position, token))?;
            let mut token_timing = timing.as_ref().map(|_| {
                FlashMoeTokenTiming::new(position, position, FlashMoeTokenPhase::Prefill, token)
            });
            report_generation_progress(&progress, || {
                format!(
                    "prefill token begin processed={} remaining={} position={}",
                    position.saturating_sub(start_position) + 1,
                    end_position.saturating_sub(position + 1),
                    position
                )
            });
            // Populate the causal KV cache with the prompt tokens so decode can
            // attend to the full rendered prompt rather than only the latest
            // generated token.
            last_hidden = Some(self.forward_token_input(
                FlashMoeTokenInput::text(token, position),
                kv_cache,
                position,
                false,
                token_timing.as_mut(),
                progress.clone(),
            )?);
            if let Some(token_timing) = token_timing
                && let Some(timing) = timing.as_deref_mut()
            {
                timing.tokens.push(token_timing);
            }
            let processed = position.saturating_sub(start_position) + 1;
            let remaining = end_position.saturating_sub(position + 1);
            let should_report = report_prefill_progress
                && (processed == 1
                    || remaining == 0
                    || processed % 16 == 0
                    || last_progress.elapsed() >= Duration::from_secs(10));
            if should_report {
                report_generation_progress(&progress, || {
                    format!(
                        "prefill progress processed={} remaining={} position={} elapsed_ms={}",
                        processed,
                        remaining,
                        position,
                        progress_started.elapsed().as_millis()
                    )
                });
                debug!(
                    target: "flashmoe::lifecycle",
                    "flashmoe: prefill progress processed={} remaining={} position={} elapsed_ms={}",
                    processed,
                    remaining,
                    position,
                    progress_started.elapsed().as_millis()
                );
                last_progress = OptionalInstant::now(report_prefill_progress);
            }
        }
        last_hidden.context("cannot generate from an empty prompt")
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_qwen_chunks(
        &mut self,
        prompt_tokens: &[u32],
        start_position: usize,
        end_position: usize,
        chunk_tokens: usize,
        kv_cache: &mut KvCache,
        record_prefill_state: bool,
        mut timing: Option<&mut FlashMoeGenerationTiming>,
        progress: GenerationProgress<'_>,
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let mut last_hidden = None;
            let mut chunk_start = start_position;
            while chunk_start < end_position {
                let chunk_end = chunk_start.saturating_add(chunk_tokens).min(end_position);
                let started = Instant::now();
                report_generation_progress(&progress, || {
                    format!(
                        "qwen prefill chunk begin start={chunk_start} tokens={}",
                        chunk_end.saturating_sub(chunk_start)
                    )
                });
                let hidden = autoreleasepool(|_| -> Result<Vec<f32>> {
                    let mut rows = Vec::with_capacity(chunk_end - chunk_start);
                    let mut row_timings = Vec::with_capacity(chunk_end - chunk_start);
                    for position in chunk_start..chunk_end {
                        let token = prompt_tokens[position];
                        kv_cache.record_prompt_token_record(FlashMoePromptTokenRecord::new(
                            position, token,
                        ))?;
                        row_timings.push(timing.as_ref().map(|_| {
                            FlashMoeTokenTiming::new(
                                position,
                                position,
                                FlashMoeTokenPhase::Prefill,
                                token,
                            )
                        }));
                        rows.push(QwenTokenExecutionOutput {
                            hidden: self.dense.embedding(token, self.runtime.width)?,
                            recurrent_value: self.dense.seed(position, token)?
                                ^ (self.plan.model.len() as u64),
                        });
                    }
                    let mut device_state = None;
                    for layer in 0..self.config.num_hidden_layers {
                        let next = self.forward_qwen_layer_major_matrix(
                            layer,
                            &mut rows,
                            device_state.as_ref(),
                            chunk_start,
                            kv_cache,
                            record_prefill_state,
                            &mut row_timings,
                        )?;
                        device_state = Some(next);
                    }
                    let final_state = device_state
                        .context("Qwen layer-major prefill graph produced no layer output")?;
                    let final_norm_weight = self
                        .model_norm_weight("model.norm.weight", self.runtime.width)?
                        .context("missing Qwen final norm weight")?;
                    let norm_started = OptionalInstant::now(timing.is_some());
                    let hidden = self
                        .metal
                        .qwen_final_norm_last_row(&final_state, &final_norm_weight)?;
                    if hidden.len() != self.runtime.width {
                        bail!("Qwen layer-major final hidden row has incompatible geometry");
                    }
                    let norm_elapsed = norm_started.elapsed();
                    let final_timing_index = row_timings.len().saturating_sub(1);
                    for (index, token_timing) in row_timings.iter_mut().enumerate() {
                        if let Some(token_timing) = token_timing {
                            token_timing.buckets.total_wall = token_timing
                                .layers
                                .iter()
                                .map(|layer| layer.buckets.total_wall)
                                .sum::<Duration>();
                            if index == final_timing_index {
                                token_timing.buckets.combine_norm += norm_elapsed;
                                token_timing.buckets.total_wall += norm_elapsed;
                            }
                        }
                    }
                    if let Some(timing) = timing.as_deref_mut() {
                        timing.tokens.extend(row_timings.into_iter().flatten());
                    }
                    Ok(hidden)
                })?;
                self.metal.inner.finish_token_boundary(chunk_end - 1)?;
                last_hidden = Some(hidden);
                report_generation_progress(&progress, || {
                    format!(
                        "qwen prefill chunk complete processed={} remaining={} elapsed_ms={}",
                        chunk_end.saturating_sub(start_position),
                        end_position.saturating_sub(chunk_end),
                        started.elapsed().as_millis()
                    )
                });
                chunk_start = chunk_end;
            }
            return last_hidden.context("cannot generate from an empty Qwen prompt");
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (
                prompt_tokens,
                start_position,
                end_position,
                chunk_tokens,
                kv_cache,
                record_prefill_state,
                timing,
                progress,
            );
            bail!("Qwen chunked prefill requires Apple Silicon Metal")
        }
    }

    fn prefill_with_vision(
        &mut self,
        inputs: &QwenVlRuntimeInputs,
        kv_cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        let mut cursor = inputs.token_inputs()?;
        let mut last_hidden = None;
        while let Some((position, input)) = cursor.next_input()? {
            let token = input.token();
            kv_cache.record_prompt_token_record(FlashMoePromptTokenRecord::new(position, token))?;
            last_hidden =
                Some(self.forward_token_input(input, kv_cache, position, false, None, None)?);
        }
        last_hidden.context("cannot generate from empty Qwen-VL runtime inputs")
    }

    /// Generate text from ordered text and image content using the Qwen3-VL vision encoder.
    ///
    /// Returns an error when the engine was not loaded from a Qwen3-VL plan
    /// (i.e. `plan.vision_weights` is `None`).
    pub fn generate_multimodal(
        &mut self,
        request: &MultimodalGenerationRequest,
    ) -> Result<GenerationOutput> {
        let image_count = request
            .content
            .iter()
            .filter(|part| matches!(part, MultimodalContent::Image { .. }))
            .count();
        if image_count == 0 {
            bail!("generate_multimodal requires at least one image block");
        }

        let vision_config = self
            .config
            .vision_config
            .as_ref()
            .context("generate_multimodal requires a Qwen3-VL plan with a vision_config")?;
        let preprocessor = ImagePreprocessor::from_vision_config(vision_config);
        let (parts, visual_encodings) = {
            let encoder = self.input_adapter_executor.vision_encoder()?;
            let mut parts = Vec::with_capacity(request.content.len());
            let mut visual_encodings = Vec::with_capacity(image_count);
            for part in &request.content {
                match part {
                    MultimodalContent::Text { text } => {
                        parts.push(ChatContentPart::Text { text: text.clone() });
                    }
                    MultimodalContent::Image { image_path } => {
                        let visual = encoder.encode(&preprocessor, image_path)?;
                        let num_visual_tokens = visual.embeddings.len();
                        parts.push(ChatContentPart::Image {
                            image: Some(image_path.display().to_string()),
                            placeholder_tokens: Some(num_visual_tokens),
                        });
                        visual_encodings.push(visual);
                    }
                }
            }
            (parts, visual_encodings)
        };

        self.generate_with_encoded_visual_prompt(
            ChatMessageContent::Parts(parts),
            visual_encodings,
            request.max_tokens,
            request.temperature,
            request.top_k,
            request.seed,
        )
    }

    /// Generate text from an image + text prompt using the Qwen3-VL vision encoder.
    ///
    /// Compatibility wrapper around the structured multimodal path.
    pub fn generate_with_image(
        &mut self,
        request: &VisionGenerationRequest,
    ) -> Result<GenerationOutput> {
        if request.prompt.contains("<|image_pad|>") {
            let vision_config = self
                .config
                .vision_config
                .as_ref()
                .context("generate_with_image requires a Qwen3-VL plan with a vision_config")?;
            let preprocessor = ImagePreprocessor::from_vision_config(vision_config);
            let visual = {
                let encoder = self.input_adapter_executor.vision_encoder()?;
                encoder.encode(&preprocessor, &request.image_path)?
            };
            return self.generate_with_encoded_visual_prompt(
                ChatMessageContent::Text(request.prompt.clone()),
                vec![visual],
                request.max_tokens,
                request.temperature,
                request.top_k,
                request.seed,
            );
        }

        self.generate_multimodal(&MultimodalGenerationRequest {
            content: vec![
                MultimodalContent::Image {
                    image_path: request.image_path.clone(),
                },
                MultimodalContent::Text {
                    text: request.prompt.clone(),
                },
            ],
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_k: request.top_k,
            seed: request.seed,
        })
    }

    fn generate_with_encoded_visual_prompt(
        &mut self,
        content: ChatMessageContent,
        visual_encodings: Vec<VisionEncoding>,
        max_tokens: i32,
        temperature: f32,
        top_k: i32,
        seed: u32,
    ) -> Result<GenerationOutput> {
        // Qwen3-VL chat template: <|vision_start|> + N×<|image_pad|> + <|vision_end|>
        let vision_start = self.tokenizer.token_id("<|vision_start|>");
        let vision_end = self.tokenizer.token_id("<|vision_end|>");
        let image_pad = self.tokenizer.token_id("<|image_pad|>");
        let (vs_tok, ve_tok, pad_tok) = match (vision_start, vision_end, image_pad) {
            (Some(vs), Some(ve), Some(pad)) => (vs, ve, pad),
            _ => bail!(
                "Qwen3-VL tokenizer is missing required vision special tokens \
                 (<|vision_start|>, <|vision_end|>, <|image_pad|>); \
                 ensure the tokenizer.json is from a VL checkpoint"
            ),
        };

        let chat_text = self.tokenizer.apply_chat_template_to_messages(
            &[ChatMessage {
                role: ChatRole::User,
                content,
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            }],
            &[],
            true,
        )?;
        let runtime_inputs = QwenVlRuntimeInputs::build(
            self.tokenizer.encode(&chat_text)?,
            vs_tok,
            ve_tok,
            pad_tok,
            visual_encodings,
        )?;

        let mut kv_cache = KvCache::new(
            self.config.num_hidden_layers,
            runtime_inputs.prompt_tokens().len() + max_tokens.max(0) as usize,
        );
        let prefill_hidden = self.prefill_with_vision(&runtime_inputs, &mut kv_cache)?;

        let mut sampler = TokenSampler::new(temperature, top_k, seed);
        let mut generated = Vec::new();
        let max_tokens = max_tokens.max(0) as usize;
        let mut stopped = false;
        if max_tokens > 0 {
            let token = self.sample_from_hidden(
                &mut sampler,
                &prefill_hidden,
                runtime_inputs.prompt_tokens(),
                &generated,
                false,
                &None,
                None,
            )?;
            if !self.tokenizer.is_eos(token) {
                generated.push(token);
            } else {
                stopped = true;
            }
        }
        while !stopped && generated.len() < max_tokens {
            let position = runtime_inputs.prompt_tokens().len() + generated.len() - 1;
            let sampled = self.sample_next_token(
                &mut sampler,
                runtime_inputs.prompt_tokens(),
                &generated,
                &mut kv_cache,
                position,
                MropePosition::text(runtime_inputs.next_mrope_position() + generated.len() - 1),
                None,
                false,
                None,
                None,
            )?;
            let token = sampled.token;
            if self.tokenizer.is_eos(token) {
                break;
            }
            generated.push(token);
        }

        let decoded = self.tokenizer.decode(&generated)?;
        let finish_reason = generation_finish_reason(generated.len(), max_tokens);
        let (content, tool_calls) = self.parse_native_tool_output(
            &decoded,
            finish_reason == GenerationFinishReason::MaxTokens,
        )?;
        Ok(GenerationOutput {
            content,
            tool_calls,
            finish_reason,
            prompt_tokens: runtime_inputs.prompt_tokens().len(),
            generated_tokens: generated.len(),
            prompt_cache: PromptCacheStats {
                source: PromptCacheSource::None,
                cached_tokens: 0,
                prefilled_tokens: runtime_inputs.prompt_tokens().len(),
                restore_ms: 0,
            },
            tool_constraints: None,
            performance: NativeGenerationStats {
                fresh_prefill_tokens: runtime_inputs.prompt_tokens().len(),
                cached_tokens: 0,
                model_family: format!("{:?}", self.model_layout.family),
                active_experts_per_token: nonzero_usize(self.model_layout.scheduled_active_experts),
                expert_strategy: match self.expert_access {
                    FlashMoeExpertAccessCapability::ParallelPositionedWholeExpertReads => {
                        "streamed_parallel_pread"
                    }
                    FlashMoeExpertAccessCapability::ResidentMappedWholeExpertSlots => {
                        "resident_complete_corpus"
                    }
                }
                .to_string(),
                prefill_command_kind: "scalar_multimodal".to_string(),
                thinking_enabled: false,
                ..NativeGenerationStats::default()
            },
        })
    }

    pub(super) fn model_norm_weight(
        &self,
        canonical_name: &str,
        width: usize,
    ) -> Result<Option<Vec<f32>>> {
        let Some(mut weight) = self.dense.norm_weight(canonical_name, width)? else {
            return Ok(None);
        };
        apply_qwen_norm_weight_semantics(
            self.config.norm_weight_semantics(),
            canonical_name,
            &mut weight,
        );
        Ok(Some(weight))
    }

    fn parse_native_tool_output(
        &self,
        content: &str,
        allow_incomplete: bool,
    ) -> Result<(String, Vec<ChatToolCall>)> {
        if self.executor.is_deepseek_v4() {
            parse_deepseek_tool_call_output_with_incomplete(content, allow_incomplete)
        } else {
            parse_qwen_tool_call_output_with_incomplete(content, allow_incomplete)
        }
    }

    pub(super) fn rms_norm_with_model_weight(
        &self,
        canonical_name: &str,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let weight = self.model_norm_weight(canonical_name, input.len())?;
        let mut out = input.to_vec();
        rms_norm_with_weight_and_epsilon_in_place(
            &mut out,
            weight.as_deref(),
            self.config.rms_norm_epsilon(),
        );
        Ok(out)
    }

    fn sample_next_token(
        &mut self,
        sampler: &mut TokenSampler,
        prompt_tokens: &[u32],
        generated: &[u32],
        kv_cache: &mut KvCache,
        position: usize,
        rope_position: MropePosition,
        timing: Option<&mut FlashMoeGenerationTiming>,
        trace_candidates: bool,
        progress: GenerationProgress<'_>,
        tool_constraint: Option<&mut NativeToolConstraint>,
    ) -> Result<SampledDecode> {
        let previous = generated
            .last()
            .copied()
            .or_else(|| prompt_tokens.last().copied())
            .unwrap_or_else(|| self.tokenizer.eos_token_id());
        let mut token_timing = timing.as_ref().map(|_| {
            FlashMoeTokenTiming::new(
                prompt_tokens.len() + generated.len(),
                position,
                FlashMoeTokenPhase::Decode,
                previous,
            )
        });
        let hidden = self.forward_token_input(
            FlashMoeTokenInput::resident(previous, rope_position),
            kv_cache,
            position,
            true,
            token_timing.as_mut(),
            progress.clone(),
        )?;
        let sample_started = OptionalInstant::now(token_timing.is_some());
        let token = self.sample_from_hidden(
            sampler,
            &hidden,
            prompt_tokens,
            generated,
            trace_candidates,
            &progress,
            tool_constraint,
        )?;
        let elapsed = sample_started.elapsed();
        if let Some(mut token_timing) = token_timing {
            token_timing.buckets.sampling += elapsed;
            token_timing.buckets.total_wall += elapsed;
            token_timing.sampled_token = Some(token);
            if let Some(timing) = timing {
                timing.tokens.push(token_timing);
            }
        }
        Ok(SampledDecode { token, hidden })
    }

    fn sample_from_hidden(
        &self,
        sampler: &mut TokenSampler,
        hidden: &[f32],
        prompt_tokens: &[u32],
        generated: &[u32],
        trace_candidates: bool,
        progress: &GenerationProgress<'_>,
        tool_constraint: Option<&mut NativeToolConstraint>,
    ) -> Result<u32> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return autoreleasepool(|_| {
                self.sample_from_hidden_in_autoreleasepool(
                    sampler,
                    hidden,
                    prompt_tokens,
                    generated,
                    trace_candidates,
                    progress,
                    tool_constraint,
                )
            });
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        self.sample_from_hidden_in_autoreleasepool(
            sampler,
            hidden,
            prompt_tokens,
            generated,
            trace_candidates,
            progress,
            tool_constraint,
        )
    }

    fn sample_from_hidden_in_autoreleasepool(
        &self,
        sampler: &mut TokenSampler,
        hidden: &[f32],
        prompt_tokens: &[u32],
        generated: &[u32],
        trace_candidates: bool,
        progress: &GenerationProgress<'_>,
        mut tool_constraint: Option<&mut NativeToolConstraint>,
    ) -> Result<u32> {
        if let Some(constraint) = tool_constraint.as_deref_mut()
            && let Some(token) = constraint.forced_next_token(&self.tokenizer, generated)?
        {
            return Ok(token);
        }
        if let Some(graph) = self.executor.deepseek_v4_graph() {
            let logits = self.metal.deepseek_v4_logits(graph, hidden)?;
            let candidates = sampler.top_candidates(&logits, prompt_tokens, generated);
            trace_sampling_candidates(
                progress,
                &self.tokenizer,
                prompt_tokens.len(),
                generated,
                &candidates,
                trace_candidates.then_some((hidden, logits.as_slice())),
            );
            return sampler.sample_candidates(candidates);
        }
        if trace_candidates || tool_constraint.is_some() {
            let logits = self.dense.lm_head_logits_with_metal(
                Some(&self.metal),
                hidden,
                self.tokenizer.vocab_size(),
            )?;
            let mut candidates = sampler.top_candidates(&logits, prompt_tokens, generated);
            if let Some(constraint) = tool_constraint.as_deref_mut() {
                loop {
                    let filtered = constraint.filter_candidates(
                        &self.tokenizer,
                        generated,
                        candidates,
                        sampler.top_k,
                    )?;
                    if !filtered.is_empty() {
                        candidates = filtered;
                        break;
                    }
                    if sampler.candidate_limit() >= logits.len() {
                        bail!(
                            "native tool constraint rejected every vocabulary candidate at generated token {}",
                            generated.len()
                        );
                    }
                    sampler.widen_candidates(
                        sampler
                            .candidate_limit()
                            .saturating_mul(4)
                            .min(logits.len()),
                    );
                    candidates = sampler.top_candidates(&logits, prompt_tokens, generated);
                }
            }
            sampler.truncate_for_sampling(&mut candidates);
            trace_sampling_candidates(
                progress,
                &self.tokenizer,
                prompt_tokens.len(),
                generated,
                &candidates,
                Some((hidden, &logits)),
            );
            return sampler.sample_candidates(candidates);
        }
        let vocab_size = self.tokenizer.vocab_size();
        let top_k = sampler.candidate_limit().min(vocab_size).max(1);
        let repeated = sampler.repeated_tokens(prompt_tokens, generated);
        let repeated_vocab_tokens = repeated.iter().filter(|token| **token < vocab_size).count();
        let raw_candidate_count = top_k
            .saturating_add(repeated_vocab_tokens)
            .min(vocab_size)
            .max(1);
        let raw_candidates = self.dense.lm_head_raw_top_candidates_with_metal(
            &self.metal,
            hidden,
            vocab_size,
            raw_candidate_count,
        )?;
        let candidates = rerank_resident_lm_head_candidates(
            &raw_candidates,
            top_k,
            sampler.repeat_penalty,
            &repeated,
        );
        trace_sampling_candidates(
            progress,
            &self.tokenizer,
            prompt_tokens.len(),
            generated,
            &candidates,
            None,
        );
        sampler.sample_candidates(candidates)
    }

    fn model_dimensions(&self) -> FlashMoeModelDimensions {
        FlashMoeModelDimensions {
            layers: self.model_layout.layers,
            hidden_size: self.model_layout.hidden_size,
            attention_heads: self.model_layout.attention_heads,
            kv_heads: self.model_layout.kv_heads,
            vocab_size: self.model_layout.vocab_size,
            experts_per_layer: Some(self.model_layout.experts_per_layer),
            active_experts_per_token: Some(self.routing_policy.active_experts),
            moe_intermediate_size: nonzero_usize(self.model_layout.moe_intermediate_size),
            shared_experts: nonzero_usize(self.model_layout.shared_experts),
        }
    }

    pub(super) fn layer_dimensions(&self, layer: usize) -> FlashMoeLayerDimensions {
        let full_layout = self
            .runtime
            .full_attention
            .get(layer)
            .and_then(|layout| *layout);
        let linear_layout = self
            .runtime
            .linear_attention
            .get(layer)
            .and_then(|layout| *layout);
        FlashMoeLayerDimensions {
            hidden_size: self.model_layout.hidden_size,
            q_width: full_layout
                .map(|layout| layout.q_width)
                .or_else(|| linear_layout.map(|layout| layout.total_key_width)),
            kv_width: full_layout
                .map(|layout| layout.kv_width)
                .or_else(|| linear_layout.map(|layout| layout.total_value_width)),
            head_dim: full_layout
                .map(|layout| layout.head_dim)
                .or_else(|| linear_layout.map(|layout| layout.key_dim)),
            experts_per_layer: Some(self.model_layout.experts_per_layer),
            active_experts_per_token: Some(self.routing_policy.active_experts),
            shared_experts: nonzero_usize(self.model_layout.shared_experts),
        }
    }
}
