//! llama.cpp inference backend.
//!
//! Wraps the `llama_cpp_2` crate and exposes a [`LlamaCppBackend`] struct that
//! handles text and multimodal (vision) generation through llama.cpp.  This
//! module is the sibling of `flashmoe` inside `crate::inference`.

use std::fs;
use std::num::NonZeroU32;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Instant, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use encoding_rs::UTF_8;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker,
};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::energy::{self, EnergyEstimate};
use crate::inference::PromptCacheMissReason;
use crate::inference::chat_template::{ChatTemplateOptions, TokenizerChatTemplate};

const BATCH_SIZE: usize = 512;
const MIN_GENERATION_CONTEXT_TOKENS: usize = 1;
pub(crate) const LLAMA_SESSION_CACHE_VERSION: &str = "llamacpp-session-v1";

/// Parameters for a single generation call.
#[derive(Debug, Clone)]
pub struct LlamaCppRequest {
    pub prompt: String,
    pub ctx_size: u32,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub gpu_layers: u32,
    pub max_tokens: i32,
    pub top_k: i32,
    pub temperature: f32,
    pub seed: u32,
}

/// Parameters for a structured chat generation call.
#[derive(Debug, Clone)]
pub struct LlamaCppChatRequest {
    pub messages: Value,
    pub tools: Value,
    /// Controller-owned stable-root identity for managed invocations. Generic chat omits it.
    pub stage_root: Option<crate::inference::StageRootDescriptor>,
    /// Optional JSON schema enforced token-by-token during generation.
    pub json_schema: Option<Value>,
    pub ctx_size: u32,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub gpu_layers: u32,
    pub max_tokens: i32,
    pub top_k: i32,
    pub temperature: f32,
    pub seed: u32,
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    EndOfGeneration,
    MaxTokens,
}

/// Output of a generation call.
#[derive(Debug, Clone)]
pub struct Output {
    pub content: String,
    pub finish_reason: FinishReason,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub prefilled_prompt_tokens: usize,
    pub prompt_cache_source: Option<String>,
    pub prompt_cache_restore_ms: u64,
    pub prompt_cache_miss_reason: Option<PromptCacheMissReason>,
    pub prompt_cache_lookup_detail: Option<crate::inference::PromptCacheLookupDetail>,
    pub prompt_root: Option<crate::inference::BackendPromptRoot>,
    pub duration_ms: u64,
    pub energy: Option<EnergyEstimate>,
}

/// A loaded llama.cpp backend + model, ready for inference.
pub struct LlamaCppBackend {
    backend: LlamaBackend,
    model: LlamaModel,
    chat_template: Option<TokenizerChatTemplate>,
    /// Path to the primary model GGUF file.
    pub model_path: PathBuf,
    session_cache: crate::config::ResolvedSessionCacheConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LlamaSessionSettings {
    ctx_size: u32,
    threads: Option<i32>,
    threads_batch: Option<i32>,
}

struct CachedLlamaContext<'a> {
    context: LlamaContext<'a>,
    evaluated_tokens: Vec<LlamaToken>,
    settings: LlamaSessionSettings,
    restored_from_disk: bool,
    restore_ms: u64,
}

/// One logical chat session with a live llama.cpp context and a crash-safe disk snapshot.
///
/// Exact token-prefix comparison is the invalidation authority. A changed system prompt,
/// tool schema, chat template, or compacted transcript therefore falls back to the longest
/// safe prefix instead of reusing stale attention state.
pub struct LlamaCppChatSession<'a> {
    backend: &'a LlamaCppBackend,
    session_id: String,
    cached: Option<CachedLlamaContext<'a>>,
}

fn suppress_logs() {
    static SUPPRESSED: OnceLock<()> = OnceLock::new();
    SUPPRESSED.get_or_init(|| {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
    });
}

/// Load a llama.cpp backend from the given GGUF model file path.
///
/// The caller is responsible for resolving the file path (e.g. via
/// [`crate::agent_core::find_model_in_cache_in`]).
pub fn load_from_file(path: &Path, gpu_layers: u32) -> Result<LlamaCppBackend> {
    let session_cache = crate::config::UserConfig::load()?.effective_llamacpp_session_cache();
    load_from_file_with_cache(path, gpu_layers, session_cache)
}

fn load_from_file_with_cache(
    path: &Path,
    gpu_layers: u32,
    session_cache: crate::config::ResolvedSessionCacheConfig,
) -> Result<LlamaCppBackend> {
    suppress_logs();
    let mut backend = LlamaBackend::init().context("failed to initialize llama backend")?;
    backend.void_logs();
    let model_params = model_params(gpu_layers)?;
    let loaded_model = LlamaModel::load_from_file(&backend, path, &model_params)
        .with_context(|| format!("failed to load model {}", path.display()))?;
    let chat_template = load_sidecar_chat_template(path)?;
    Ok(LlamaCppBackend {
        backend,
        model: loaded_model,
        chat_template,
        model_path: path.to_owned(),
        session_cache,
    })
}

/// Load a text backend and prove that it can create the requested context.
///
/// Some llama.cpp/Metal combinations can load offloaded weights but fail before the first token
/// when the context graph is created. Retrying only K/Q/V on the CPU is not sufficient in that
/// case because the model itself is still attached to the failing accelerator. Probe while the
/// accelerated model can still be dropped cleanly, then reload the model CPU-only as a bounded
/// correctness fallback.
pub fn load_text_from_file(
    path: &Path,
    gpu_layers: u32,
    ctx_size: u32,
    threads: Option<i32>,
    threads_batch: Option<i32>,
) -> Result<(LlamaCppBackend, Option<String>)> {
    let session_cache = crate::config::UserConfig::load()?.effective_llamacpp_session_cache();
    let settings = LlamaSessionSettings {
        ctx_size,
        threads,
        threads_batch,
    };
    let accelerated = load_from_file_with_cache(path, gpu_layers, session_cache.clone())?;
    let accelerated_probe = accelerated.new_text_context(settings).map(drop);
    match accelerated_probe {
        Ok(()) => Ok((accelerated, None)),
        Err(accelerated_error) if gpu_layers > 0 => {
            drop(accelerated);
            let cpu = load_from_file_with_cache(path, 0, session_cache).with_context(|| {
                format!(
                    "failed to reload llama.cpp model CPU-only after accelerated context setup failed: {accelerated_error:#}"
                )
            })?;
            cpu.new_text_context(settings).map(drop).with_context(|| {
                format!(
                    "failed to create CPU-only llama.cpp context after accelerated context setup failed: {accelerated_error:#}"
                )
            })?;
            Ok((
                cpu,
                Some(format!(
                    "Accelerated llama.cpp context setup failed; reloaded the model CPU-only for this session. Original error: {accelerated_error:#}"
                )),
            ))
        }
        Err(error) => Err(error).context("failed to validate CPU-only llama.cpp context setup"),
    }
}

fn model_params(gpu_layers: u32) -> Result<LlamaModelParams> {
    let mut model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
    if gpu_layers == 0 {
        model_params = model_params
            .with_devices(&[])
            .context("failed to configure CPU-only llama.cpp model devices")?;
    }
    Ok(model_params)
}

impl LlamaCppBackend {
    /// Run text generation for the given request.
    pub fn generate(&self, request: &LlamaCppRequest) -> Result<Output> {
        let (prompt, add_bos) = self.render_prompt(&request.prompt)?;
        self.generate_rendered(
            &prompt,
            add_bos,
            request.ctx_size,
            request.threads,
            request.threads_batch,
            request.max_tokens,
            request.top_k,
            request.temperature,
            request.seed,
            None,
        )
    }

    /// Run text generation for a structured chat request.
    pub fn generate_chat(&self, request: &LlamaCppChatRequest) -> Result<Output> {
        let (prompt, add_bos) = self.render_chat_prompt(&request.messages, &request.tools)?;
        self.generate_rendered(
            &prompt,
            add_bos,
            request.ctx_size,
            request.threads,
            request.threads_batch,
            request.max_tokens,
            request.top_k,
            request.temperature,
            request.seed,
            request.json_schema.as_ref(),
        )
    }

    /// Start a reusable structured-chat session.
    pub fn start_chat_session(&self, session_id: impl Into<String>) -> LlamaCppChatSession<'_> {
        LlamaCppChatSession {
            backend: self,
            session_id: session_id.into(),
            cached: None,
        }
    }

    /// Render and tokenize the exact structured chat prompt used by [`Self::generate_chat`].
    pub fn measure_chat_prompt(&self, request: &LlamaCppChatRequest) -> Result<usize> {
        let (prompt, add_bos) = self.render_chat_prompt(&request.messages, &request.tools)?;
        Ok(self
            .model
            .str_to_token(&prompt, add_bos)
            .context("failed to tokenize chat prompt for preflight")?
            .len())
    }

    fn generate_rendered(
        &self,
        prompt: &str,
        add_bos: AddBos,
        ctx_size: u32,
        threads: Option<i32>,
        threads_batch: Option<i32>,
        max_tokens: i32,
        top_k: i32,
        temperature: f32,
        seed: u32,
        json_schema: Option<&Value>,
    ) -> Result<Output> {
        let energy_start = energy::sample();
        let started = std::time::Instant::now();
        let mut sampler = self.sampler(top_k, temperature, seed, json_schema)?;
        let settings = LlamaSessionSettings {
            ctx_size,
            threads,
            threads_batch,
        };
        let mut ctx = self.new_text_context(settings)?;

        let tokens = self
            .model
            .str_to_token(prompt, add_bos)
            .context("failed to tokenize prompt")?;

        ensure_prompt_fits_context(tokens.len(), max_tokens, ctx.n_ctx())?;

        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        for range in prompt_batch_ranges(tokens.len(), BATCH_SIZE) {
            batch.clear();
            let is_final_batch = range.end == tokens.len();
            for token_index in range.clone() {
                let is_last_prompt_token = is_final_batch && token_index + 1 == tokens.len();
                batch
                    .add(tokens[token_index], token_index as i32, &[0], is_last_prompt_token)
                    .with_context(|| {
                        format!(
                            "failed to add prompt token {token_index} to batch (batch capacity: {BATCH_SIZE}, prompt tokens: {})",
                            tokens.len()
                        )
                    })?;
            }
            ctx.decode(&mut batch).with_context(|| {
                format!(
                    "failed to decode prompt batch {}..{}",
                    range.start, range.end
                )
            })?;
        }

        let mut decoder = UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur =
            i32::try_from(tokens.len()).context("prompt token count exceeds i32::MAX")?;
        let mut generated_tokens: usize = 0;
        let mut finish_reason = FinishReason::MaxTokens;

        while generated_tokens < usize::try_from(max_tokens).unwrap_or(0) {
            let token = sample_and_accept_token(&mut sampler, &ctx, batch.n_tokens() - 1);
            if self.model.is_eog_token(token) {
                finish_reason = FinishReason::EndOfGeneration;
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .context("failed to decode output token")?;
            output.push_str(&piece);
            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .context("failed to queue generated token")?;
            ctx.decode(&mut batch)
                .context("failed to decode generated token")?;
            n_cur += 1;
            generated_tokens += 1;
            if json_schema.is_some()
                && let Some(end) = complete_json_value_end(&output)
            {
                output.truncate(end);
                finish_reason = FinishReason::EndOfGeneration;
                break;
            }
        }

        let energy =
            energy_start.and_then(|sample| sample.estimate_since(energy::sample(), started));
        Ok(Output {
            content: output,
            finish_reason,
            prompt_tokens: tokens.len(),
            generated_tokens,
            cached_prompt_tokens: 0,
            prefilled_prompt_tokens: tokens.len(),
            prompt_cache_source: None,
            prompt_cache_restore_ms: 0,
            prompt_cache_miss_reason: Some(PromptCacheMissReason::CacheDisabled),
            prompt_cache_lookup_detail: None,
            prompt_root: None,
            duration_ms: duration_millis(started),
            energy,
        })
    }

    fn new_text_context(&self, settings: LlamaSessionSettings) -> Result<LlamaContext<'_>> {
        let n_ctx = NonZeroU32::new(settings.ctx_size).context("ctx-size must be > 0")?;
        let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
        if let Some(threads) = settings.threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = settings.threads_batch.or(settings.threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }

        match self.model.new_context(&self.backend, ctx_params.clone()) {
            Ok(ctx) => Ok(ctx),
            Err(accelerated_error) => self
                .model
                .new_context(&self.backend, ctx_params.with_offload_kqv(false))
                .with_context(|| {
                    format!(
                        "failed to create llama context, including CPU K/Q/V fallback after accelerated context error: {accelerated_error}"
                    )
                }),
        }
    }

    fn sampler(
        &self,
        top_k: i32,
        temperature: f32,
        seed: u32,
        json_schema: Option<&Value>,
    ) -> Result<LlamaSampler> {
        let mut samplers = Vec::with_capacity(if json_schema.is_some() { 4 } else { 3 });
        if let Some(schema) = json_schema {
            let schema = serde_json::to_string(schema)
                .context("failed to serialize constrained-output JSON schema")?;
            samplers.push(
                LlamaSampler::llguidance(&self.model, "json_schema", &schema)
                    .context("failed to initialize constrained-output JSON guidance")?,
            );
        }
        samplers.extend([
            LlamaSampler::top_k(top_k),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(seed),
        ]);
        Ok(LlamaSampler::chain_simple(samplers))
    }

    /// Run vision (multimodal) generation for the given request and image path.
    pub fn generate_vision(&self, request: &LlamaCppRequest, image_path: &Path) -> Result<Output> {
        let mmproj_path = find_multimodal_projector(&self.model_path)?;
        let energy_start = energy::sample();
        let started = std::time::Instant::now();
        let n_ctx = NonZeroU32::new(request.ctx_size).context("ctx-size must be > 0")?;
        let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
        if let Some(threads) = request.threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = request.threads_batch.or(request.threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }
        let mut ctx = match self.model.new_context(&self.backend, ctx_params.clone()) {
            Ok(ctx) => ctx,
            Err(accelerated_error) => self
                .model
                .new_context(&self.backend, ctx_params.with_offload_kqv(false))
                .with_context(|| {
                    format!(
                        "failed to create llama context for vision tool, including CPU K/Q/V fallback after accelerated context error: {accelerated_error}"
                    )
                })?,
        };

        let mtmd_params = MtmdContextParams {
            use_gpu: request.gpu_layers > 0,
            print_timings: false,
            n_threads: request.threads_batch.or(request.threads).unwrap_or(0),
            media_marker: std::ffi::CString::new(mtmd_default_marker())?,
        };
        let mtmd =
            MtmdContext::init_from_file(&mmproj_path.to_string_lossy(), &self.model, &mtmd_params)
                .with_context(|| {
                    format!(
                        "failed to initialize multimodal projector {}",
                        mmproj_path.display()
                    )
                })?;
        if !mtmd.support_vision() {
            bail!(
                "multimodal projector {} does not report vision support",
                mmproj_path.display()
            );
        }
        let bitmap = MtmdBitmap::from_file(&mtmd, &image_path.to_string_lossy())
            .with_context(|| format!("failed to load vision image {}", image_path.display()))?;
        let prompt_with_image = format!("{}\n\nImage: {}", request.prompt, mtmd_default_marker());
        let chunks = mtmd
            .tokenize(
                MtmdInputText {
                    text: prompt_with_image,
                    add_special: true,
                    parse_special: true,
                },
                &[&bitmap],
            )
            .context("failed to tokenize multimodal vision prompt")?;

        ensure_prompt_fits_context(
            chunks.total_positions() as usize,
            request.max_tokens,
            ctx.n_ctx(),
        )?;
        let mut n_cur = chunks
            .eval_chunks(&mtmd, &ctx, 0, 0, BATCH_SIZE as i32, true)
            .context("failed to evaluate multimodal vision prompt")?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(request.top_k),
            LlamaSampler::temp(request.temperature),
            LlamaSampler::dist(request.seed),
        ]);
        let mut decoder = UTF_8.new_decoder();
        let mut output = String::new();
        let mut generated_tokens: usize = 0;
        let mut finish_reason = FinishReason::MaxTokens;
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        let mut sample_index = -1;

        while generated_tokens < usize::try_from(request.max_tokens).unwrap_or(0) {
            let token = sample_and_accept_token(&mut sampler, &ctx, sample_index);
            if self.model.is_eog_token(token) {
                finish_reason = FinishReason::EndOfGeneration;
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .context("failed to decode vision output token")?;
            output.push_str(&piece);
            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .context("failed to queue generated vision token")?;
            ctx.decode(&mut batch)
                .context("failed to decode generated vision token")?;
            n_cur += 1;
            generated_tokens += 1;
            sample_index = batch.n_tokens() - 1;
        }

        let energy =
            energy_start.and_then(|sample| sample.estimate_since(energy::sample(), started));
        Ok(Output {
            content: output,
            finish_reason,
            prompt_tokens: chunks.total_positions() as usize,
            generated_tokens,
            cached_prompt_tokens: 0,
            prefilled_prompt_tokens: chunks.total_positions() as usize,
            prompt_cache_source: None,
            prompt_cache_restore_ms: 0,
            prompt_cache_miss_reason: Some(PromptCacheMissReason::RuntimeUnsupported),
            prompt_cache_lookup_detail: None,
            prompt_root: None,
            duration_ms: duration_millis(started),
            energy,
        })
    }
}

impl LlamaCppChatSession<'_> {
    /// Generate the next response while reusing the longest exact token prefix from this session.
    pub fn generate_chat(&mut self, request: &LlamaCppChatRequest) -> Result<Output> {
        let energy_start = energy::sample();
        let started = std::time::Instant::now();
        let (prompt, add_bos) = self
            .backend
            .render_chat_prompt(&request.messages, &request.tools)?;
        let tokens = self
            .backend
            .model
            .str_to_token(&prompt, add_bos)
            .context("failed to tokenize chat prompt")?;
        let prompt_root = self.backend.stable_chat_prompt_root(request, &tokens)?;
        let settings = LlamaSessionSettings {
            ctx_size: request.ctx_size,
            threads: request.threads,
            threads_batch: request.threads_batch,
        };
        // Compile the constraint before changing a reusable KV cache. A schema/grammar error must
        // leave the session prefix untouched so a bounded retry cannot double-prefill the context.
        let mut sampler = self.backend.sampler(
            request.top_k,
            request.temperature,
            request.seed,
            request.json_schema.as_ref(),
        )?;

        let needs_context = self
            .cached
            .as_ref()
            .is_none_or(|cached| cached.settings != settings);
        let mut prompt_cache_miss_reason = None;
        if needs_context {
            let mut context = self.backend.new_text_context(settings)?;
            let restore_started = Instant::now();
            let (evaluated_tokens, miss_reason) =
                self.load_persisted_state(&mut context, settings, &tokens);
            prompt_cache_miss_reason = miss_reason;
            let restored_from_disk = !evaluated_tokens.is_empty();
            self.cached = Some(CachedLlamaContext {
                context,
                evaluated_tokens,
                settings,
                restored_from_disk,
                restore_ms: restored_from_disk
                    .then(|| duration_millis(restore_started))
                    .unwrap_or(0),
            });
        }

        let cached = self
            .cached
            .as_mut()
            .context("llama session context is missing")?;
        ensure_prompt_fits_context(tokens.len(), request.max_tokens, cached.context.n_ctx())?;

        let previous_cached_tokens = cached.evaluated_tokens.len();
        let mut prefill_start = common_token_prefix_len(&cached.evaluated_tokens, &tokens);
        let mut context_reset = false;
        if prefill_start < cached.evaluated_tokens.len() {
            // A shorter prompt needs its final token evaluated again because llama.cpp's logits
            // buffer still belongs to the longer cached sequence.
            if prefill_start == tokens.len() {
                prefill_start = prefill_start.saturating_sub(1);
            }
            let truncated = if prefill_start == 0 {
                cached.context.clear_kv_cache();
                true
            } else {
                cached
                    .context
                    .clear_kv_cache_seq(Some(0), Some(prefill_start as u32), None)
                    .unwrap_or(false)
            };
            if !truncated {
                cached.context = self.backend.new_text_context(settings)?;
                prefill_start = 0;
                context_reset = true;
            }
        }

        tracing::debug!(
            session_id = %self.session_id,
            prompt_tokens = tokens.len(),
            reused_prefix_tokens = prefill_start,
            "prepared llama.cpp session prefix"
        );
        let prompt_cache_source = (prefill_start > 0).then(|| {
            if cached.restored_from_disk {
                "disk_session".to_string()
            } else {
                "memory_session".to_string()
            }
        });
        let prompt_cache_restore_ms = if prompt_cache_source.as_deref() == Some("disk_session") {
            cached.restore_ms
        } else {
            0
        };
        prompt_cache_miss_reason = llama_prompt_cache_miss_reason(
            prefill_start,
            previous_cached_tokens,
            context_reset,
            prompt_cache_miss_reason,
        );
        let prompt_cache_lookup_detail = match prompt_cache_miss_reason {
            Some(PromptCacheMissReason::ColdSession) => {
                Some(crate::inference::PromptCacheLookupDetail::SessionCheckpointMissing)
            }
            Some(PromptCacheMissReason::PromptDiverged) => {
                Some(crate::inference::PromptCacheLookupDetail::SessionCheckpointDiverged)
            }
            _ => None,
        };
        cached.restored_from_disk = false;
        cached.restore_ms = 0;

        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        for range in prompt_batch_ranges_from(prefill_start, tokens.len(), BATCH_SIZE) {
            batch.clear();
            let is_final_batch = range.end == tokens.len();
            for token_index in range.clone() {
                let is_last_prompt_token = is_final_batch && token_index + 1 == tokens.len();
                batch
                    .add(tokens[token_index], token_index as i32, &[0], is_last_prompt_token)
                    .with_context(|| {
                        format!(
                            "failed to add prompt token {token_index} to batch (batch capacity: {BATCH_SIZE}, prompt tokens: {})",
                            tokens.len()
                        )
                    })?;
            }
            cached.context.decode(&mut batch).with_context(|| {
                format!(
                    "failed to decode prompt batch {}..{}",
                    range.start, range.end
                )
            })?;
        }

        let mut evaluated_tokens = tokens;
        let mut decoder = UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur =
            i32::try_from(evaluated_tokens.len()).context("prompt token count exceeds i32::MAX")?;
        let mut generated_tokens: usize = 0;
        let mut finish_reason = FinishReason::MaxTokens;
        let mut sample_index = if batch.n_tokens() == 0 {
            -1
        } else {
            batch.n_tokens() - 1
        };

        while generated_tokens < usize::try_from(request.max_tokens).unwrap_or(0) {
            let token = sample_and_accept_token(&mut sampler, &cached.context, sample_index);
            if self.backend.model.is_eog_token(token) {
                finish_reason = FinishReason::EndOfGeneration;
                break;
            }
            let piece = self
                .backend
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .context("failed to decode output token")?;
            output.push_str(&piece);
            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .context("failed to queue generated token")?;
            cached
                .context
                .decode(&mut batch)
                .context("failed to decode generated token")?;
            evaluated_tokens.push(token);
            n_cur += 1;
            generated_tokens += 1;
            sample_index = batch.n_tokens() - 1;
            if request.json_schema.is_some()
                && let Some(end) = complete_json_value_end(&output)
            {
                output.truncate(end);
                finish_reason = FinishReason::EndOfGeneration;
                break;
            }
        }

        let prompt_token_count = evaluated_tokens.len().saturating_sub(generated_tokens);
        cached.evaluated_tokens = evaluated_tokens;
        if let Err(error) = self.persist_state(settings) {
            tracing::warn!(
                session_id = %self.session_id,
                error = %error,
                "failed to persist llama.cpp session cache; generation remains valid"
            );
        }

        let energy =
            energy_start.and_then(|sample| sample.estimate_since(energy::sample(), started));
        Ok(Output {
            content: output,
            finish_reason,
            prompt_tokens: prompt_token_count,
            generated_tokens,
            cached_prompt_tokens: prefill_start,
            prefilled_prompt_tokens: prompt_token_count.saturating_sub(prefill_start),
            prompt_cache_source,
            prompt_cache_restore_ms,
            prompt_cache_miss_reason,
            prompt_cache_lookup_detail,
            prompt_root,
            duration_ms: duration_millis(started),
            energy,
        })
    }

    fn load_persisted_state(
        &self,
        context: &mut LlamaContext<'_>,
        settings: LlamaSessionSettings,
        prompt_tokens: &[LlamaToken],
    ) -> (Vec<LlamaToken>, Option<PromptCacheMissReason>) {
        let Some(path) = self.cache_path(settings) else {
            return (Vec::new(), Some(PromptCacheMissReason::CacheDisabled));
        };
        if !path.is_file() {
            return (Vec::new(), Some(PromptCacheMissReason::ColdSession));
        }
        match context.state_load_file(&path, settings.ctx_size as usize) {
            Ok(tokens) => {
                let prefix_len = common_token_prefix_len(&tokens, prompt_tokens);
                tracing::debug!(
                    session_id = %self.session_id,
                    cache = %path.display(),
                    cached_tokens = tokens.len(),
                    reusable_prefix_tokens = prefix_len,
                    "loaded llama.cpp session cache"
                );
                let miss_reason =
                    (prefix_len == 0).then_some(PromptCacheMissReason::PromptDiverged);
                (tokens, miss_reason)
            }
            Err(error) => {
                context.clear_kv_cache();
                tracing::warn!(
                    session_id = %self.session_id,
                    cache = %path.display(),
                    error = %error,
                    "ignored incompatible llama.cpp session cache"
                );
                (Vec::new(), Some(PromptCacheMissReason::CacheUnreadable))
            }
        }
    }

    fn persist_state(&self, settings: LlamaSessionSettings) -> Result<()> {
        let Some(path) = self.cache_path(settings) else {
            return Ok(());
        };
        let cached = self
            .cached
            .as_ref()
            .context("llama session context is missing")?;
        let parent = path
            .parent()
            .context("llama session cache path has no parent")?;
        fs::create_dir_all(parent).with_context(|| {
            format!("failed to create llama session cache {}", parent.display())
        })?;
        secure_cache_directory(parent)?;
        let temporary = tempfile::Builder::new()
            .prefix(".llamacpp-state-")
            .tempfile_in(parent)
            .with_context(|| format!("failed to create temporary cache in {}", parent.display()))?;
        cached
            .context
            .state_save_file(temporary.path(), &cached.evaluated_tokens)
            .with_context(|| {
                format!(
                    "failed to save llama session state to {}",
                    temporary.path().display()
                )
            })?;
        let max_bytes = self.backend.session_cache.max_bytes;
        let state_bytes = temporary.as_file().metadata()?.len();
        if state_bytes > max_bytes {
            tracing::warn!(
                session_id = %self.session_id,
                state_bytes,
                max_bytes,
                "llama.cpp session state exceeds the configured disk-cache budget"
            );
            return Ok(());
        }
        let temporary = temporary.into_temp_path();
        replace_cache_file(&temporary, &path)?;
        prune_session_cache(parent, &path, max_bytes)?;
        Ok(())
    }

    fn cache_path(&self, settings: LlamaSessionSettings) -> Option<PathBuf> {
        if self.session_id.trim().is_empty() || !self.backend.session_cache.enabled {
            return None;
        }
        let root = self
            .backend
            .session_cache
            .root
            .as_ref()?
            .join(LLAMA_SESSION_CACHE_VERSION);
        Some(llama_session_cache_path(
            &root,
            &self.backend.model_path,
            &self.session_id,
            settings.ctx_size,
        ))
    }
}

impl LlamaCppBackend {
    pub(crate) fn rendered_chat_prompt_identity(
        &self,
        request: &LlamaCppChatRequest,
    ) -> Result<(String, usize)> {
        let (prompt, _) = self.render_chat_prompt(&request.messages, &request.tools)?;
        let bytes = prompt.as_bytes();
        Ok((format!("{:x}", Sha256::digest(bytes)), bytes.len()))
    }

    fn stable_chat_prompt_root(
        &self,
        request: &LlamaCppChatRequest,
        prompt_tokens: &[LlamaToken],
    ) -> Result<Option<crate::inference::BackendPromptRoot>> {
        let Some(first) = request
            .messages
            .as_array()
            .and_then(|messages| messages.first())
        else {
            return Ok(None);
        };
        if first.get("role").and_then(Value::as_str) != Some("system") {
            return Ok(None);
        }
        let messages = Value::Array(vec![first.clone()]);
        let (rendered, add_bos) =
            self.render_chat_prompt_with_generation(&messages, &request.tools, false)?;
        let root_tokens = self
            .model
            .str_to_token(&rendered, add_bos)
            .context("failed to tokenize stable llama.cpp chat root")?;
        let root_len = common_token_prefix_len(prompt_tokens, &root_tokens);
        if root_len == 0 {
            return Ok(None);
        }
        Ok(Some(crate::inference::BackendPromptRoot {
            descriptor_version: crate::inference::PROMPT_ROOT_DESCRIPTOR_VERSION,
            backend: "llamacpp".to_string(),
            cache_format_version: LLAMA_SESSION_CACHE_VERSION.to_string(),
            model_namespace_sha256: llama_model_namespace_sha256(
                &self.model_path,
                request.ctx_size,
            ),
            rendered_token_sha256: crate::inference::rendered_token_sha256(
                prompt_tokens[..root_len]
                    .iter()
                    .map(|token| u32::from_le_bytes(token.0.to_le_bytes())),
            ),
            tokens: root_len,
            stage: request.stage_root.clone(),
        }))
    }

    fn render_prompt(&self, prompt: &str) -> Result<(String, AddBos)> {
        if prompt.contains("<|im_start|>") {
            return Ok((prompt.to_string(), AddBos::Never));
        }
        let Some(chat_template) = &self.chat_template else {
            return Ok((prompt.to_string(), AddBos::Always));
        };
        let rendered = chat_template.render(
            serde_json::json!([{"role": "user", "content": prompt}]),
            serde_json::json!([]),
            ChatTemplateOptions::default(),
        )?;
        Ok((rendered, AddBos::Never))
    }

    fn render_chat_prompt(&self, messages: &Value, tools: &Value) -> Result<(String, AddBos)> {
        self.render_chat_prompt_with_generation(messages, tools, true)
    }

    fn render_chat_prompt_with_generation(
        &self,
        messages: &Value,
        tools: &Value,
        add_generation_prompt: bool,
    ) -> Result<(String, AddBos)> {
        let Some(chat_template) = &self.chat_template else {
            return Ok((
                render_plain_chat_prompt(messages, add_generation_prompt),
                AddBos::Always,
            ));
        };
        let rendered = chat_template.render(
            messages,
            tools,
            ChatTemplateOptions {
                add_generation_prompt,
                ..ChatTemplateOptions::default()
            },
        )?;
        Ok((rendered, AddBos::Never))
    }
}

fn render_plain_chat_prompt(messages: &Value, add_generation_prompt: bool) -> String {
    let mut prompt = String::new();
    prompt.push_str("<conversation>\n");
    if let Some(messages) = messages.as_array() {
        for message in messages {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user");
            let content = message
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default();
            prompt.push('[');
            prompt.push_str(role);
            prompt.push_str("]\n");
            prompt.push_str(content);
            if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array)
                && !tool_calls.is_empty()
            {
                prompt.push_str("\nTool calls:\n");
                for call in tool_calls {
                    let name = call
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    let arguments = call.get("arguments").unwrap_or(&Value::Null);
                    prompt.push_str(&format!("tool={name} args={arguments}\n"));
                }
            }
            if role == "tool" {
                if let Some(name) = message.get("name").and_then(Value::as_str) {
                    prompt.push_str(&format!("\nTool name: {name}"));
                }
                if let Some(tool_call_id) = message.get("tool_call_id").and_then(Value::as_str) {
                    prompt.push_str(&format!("\nTool call id: {tool_call_id}"));
                }
            }
            prompt.push_str("\n\n");
        }
    }
    if add_generation_prompt {
        prompt.push_str("[assistant]\n");
    }
    prompt
}

fn load_sidecar_chat_template(model_path: &Path) -> Result<Option<TokenizerChatTemplate>> {
    let Some(model_dir) = model_path.parent() else {
        return Ok(None);
    };
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&config_path)
        .with_context(|| format!("failed to read tokenizer config {}", config_path.display()))?;
    TokenizerChatTemplate::from_tokenizer_config_bytes(Some(&bytes))
}

fn find_multimodal_projector(model_path: &Path) -> Result<PathBuf> {
    let model_dir = model_path
        .parent()
        .context("model path has no parent directory")?;
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .with_context(|| format!("failed to read model directory {}", model_dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path != model_path)
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| {
                    let name = name.to_ascii_lowercase();
                    name.contains("mmproj") && name.ends_with(".gguf")
                })
                .unwrap_or(false)
        })
        .collect();
    candidates.sort();
    candidates.into_iter().next().with_context(|| {
        format!(
            "no multimodal projector (mmproj*.gguf) found next to model {}; pull a Qwen vision GGUF with its mmproj file",
            model_path.display()
        )
    })
}

fn sample_and_accept_token(
    sampler: &mut LlamaSampler,
    context: &LlamaContext<'_>,
    logits_index: i32,
) -> LlamaToken {
    // LlamaSampler::sample delegates to llama_sampler_sample, which both selects and accepts the
    // token. Calling accept again corrupts stateful samplers such as grammars and penalties.
    sampler.sample(context, logits_index)
}

fn prompt_batch_ranges(token_count: usize, batch_size: usize) -> Vec<Range<usize>> {
    prompt_batch_ranges_from(0, token_count, batch_size)
}

fn prompt_batch_ranges_from(
    start_token: usize,
    token_count: usize,
    batch_size: usize,
) -> Vec<Range<usize>> {
    assert!(batch_size > 0, "batch_size must be greater than zero");
    (start_token..token_count)
        .step_by(batch_size)
        .map(|start| start..std::cmp::min(start + batch_size, token_count))
        .collect()
}

fn common_token_prefix_len(left: &[LlamaToken], right: &[LlamaToken]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn llama_prompt_cache_miss_reason(
    reused_tokens: usize,
    previous_cached_tokens: usize,
    context_reset: bool,
    load_miss_reason: Option<PromptCacheMissReason>,
) -> Option<PromptCacheMissReason> {
    if reused_tokens > 0 {
        return None;
    }
    load_miss_reason.or_else(|| {
        Some(if context_reset {
            PromptCacheMissReason::ContextReset
        } else if previous_cached_tokens > 0 {
            PromptCacheMissReason::PromptDiverged
        } else {
            PromptCacheMissReason::ColdSession
        })
    })
}

fn llama_session_cache_path(
    root: &Path,
    model_path: &Path,
    session_id: &str,
    ctx_size: u32,
) -> PathBuf {
    let mut digest = llama_model_namespace_hasher(model_path, ctx_size);
    digest.update(session_id.as_bytes());
    root.join(format!("{:x}.state", digest.finalize()))
}

fn llama_model_namespace_sha256(model_path: &Path, ctx_size: u32) -> String {
    format!(
        "{:x}",
        llama_model_namespace_hasher(model_path, ctx_size).finalize()
    )
}

fn llama_model_namespace_hasher(model_path: &Path, ctx_size: u32) -> Sha256 {
    let canonical_model = model_path
        .canonicalize()
        .unwrap_or_else(|_| model_path.to_path_buf());
    let metadata = model_path.metadata().ok();
    let size = metadata.as_ref().map_or(0, fs::Metadata::len);
    let modified_nanos = metadata
        .and_then(|metadata| metadata.modified().ok())
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos());
    let mut digest = Sha256::new();
    digest.update(LLAMA_SESSION_CACHE_VERSION.as_bytes());
    digest.update([0]);
    digest.update(canonical_model.to_string_lossy().as_bytes());
    digest.update([0]);
    digest.update(size.to_le_bytes());
    digest.update(modified_nanos.to_le_bytes());
    digest.update(ctx_size.to_le_bytes());
    digest
}

fn replace_cache_file(source: &Path, destination: &Path) -> Result<()> {
    #[cfg(target_os = "windows")]
    if destination.exists() {
        fs::remove_file(destination).with_context(|| {
            format!(
                "failed to replace llama session cache {}",
                destination.display()
            )
        })?;
    }
    fs::rename(source, destination).with_context(|| {
        format!(
            "failed to atomically replace llama session cache {}",
            destination.display()
        )
    })
}

fn secure_cache_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("failed to secure llama session cache {}", path.display()))?;
    }
    Ok(())
}

fn prune_session_cache(root: &Path, current: &Path, max_bytes: u64) -> Result<()> {
    let mut states = fs::read_dir(root)
        .with_context(|| format!("failed to inspect llama session cache {}", root.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "state")
        })
        .map(|path| {
            let metadata = path.metadata();
            let modified = metadata
                .as_ref()
                .ok()
                .and_then(|metadata| metadata.modified().ok())
                .unwrap_or(UNIX_EPOCH);
            let bytes = metadata.map(|metadata| metadata.len()).unwrap_or(0);
            (modified, bytes, path)
        })
        .collect::<Vec<_>>();
    let mut total = states.iter().map(|(_, bytes, _)| *bytes).sum::<u64>();
    states.sort_by_key(|(modified, _, _)| *modified);
    for (_, bytes, path) in states {
        if total <= max_bytes {
            break;
        }
        if path == current {
            continue;
        }
        fs::remove_file(&path)
            .with_context(|| format!("failed to prune llama session cache {}", path.display()))?;
        total = total.saturating_sub(bytes);
    }
    Ok(())
}

fn complete_json_value_end(output: &str) -> Option<usize> {
    let mut values = serde_json::Deserializer::from_str(output).into_iter::<Value>();
    values.next()?.ok()?;
    Some(values.byte_offset())
}

fn ensure_prompt_fits_context(prompt_tokens: usize, max_tokens: i32, n_ctx: u32) -> Result<()> {
    let n_ctx = usize::try_from(n_ctx).context("context size does not fit usize")?;
    let requested_generation_tokens = usize::try_from(max_tokens.max(0))
        .context("requested generation token count does not fit usize")?;
    let reserved_generation_tokens = requested_generation_tokens.max(MIN_GENERATION_CONTEXT_TOKENS);
    if prompt_tokens + reserved_generation_tokens > n_ctx {
        bail!(
            "prompt is too long for the configured context: {prompt_tokens} prompt tokens + {reserved_generation_tokens} reserved generation tokens exceeds ctx-size {n_ctx}. Increase --ctx-size or reduce the task/history size."
        );
    }
    Ok(())
}

fn duration_millis(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_only_model_params_exclude_accelerator_devices() {
        let params = model_params(0).unwrap();

        assert!(params.devices().is_empty());
    }

    #[test]
    fn prompt_batch_ranges_splits_prompts_larger_than_batch_capacity() {
        assert_eq!(
            prompt_batch_ranges(1_025, 512),
            vec![0..512, 512..1_024, 1_024..1_025]
        );
        assert_eq!(prompt_batch_ranges_from(700, 1_025, 512), vec![700..1_025]);
    }

    #[test]
    fn session_prefix_reuse_stops_at_the_first_changed_token() {
        let cached = [1, 2, 3, 4].map(LlamaToken::new);
        let extended = [1, 2, 3, 4, 5].map(LlamaToken::new);
        let changed = [1, 2, 9, 4].map(LlamaToken::new);

        assert_eq!(common_token_prefix_len(&cached, &extended), 4);
        assert_eq!(common_token_prefix_len(&cached, &changed), 2);
    }

    #[test]
    fn prompt_cache_miss_reason_preserves_backend_causes_and_clears_on_reuse() {
        assert_eq!(
            llama_prompt_cache_miss_reason(0, 0, false, None),
            Some(PromptCacheMissReason::ColdSession)
        );
        assert_eq!(
            llama_prompt_cache_miss_reason(0, 10, false, None),
            Some(PromptCacheMissReason::PromptDiverged)
        );
        assert_eq!(
            llama_prompt_cache_miss_reason(
                0,
                0,
                false,
                Some(PromptCacheMissReason::CacheUnreadable)
            ),
            Some(PromptCacheMissReason::CacheUnreadable)
        );
        assert_eq!(
            llama_prompt_cache_miss_reason(
                4,
                10,
                false,
                Some(PromptCacheMissReason::PromptDiverged)
            ),
            None
        );
    }

    #[test]
    fn session_cache_key_invalidates_model_context_and_session_changes() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model = tmp.path().join("model.gguf");
        fs::write(&model, b"model-v1").unwrap();
        let root = tmp.path().join("cache");

        let original = llama_session_cache_path(&root, &model, "session-a", 8_192);
        let other_session = llama_session_cache_path(&root, &model, "session-b", 8_192);
        let other_context = llama_session_cache_path(&root, &model, "session-a", 16_384);
        fs::write(&model, b"model-v2-with-a-new-size").unwrap();
        let changed_model = llama_session_cache_path(&root, &model, "session-a", 8_192);

        assert_ne!(original, other_session);
        assert_ne!(original, other_context);
        assert_ne!(original, changed_model);
        assert!(
            !original
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("session-a")
        );
    }

    #[test]
    fn session_cache_pruning_is_bounded_and_preserves_current_state() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut paths = Vec::new();
        for index in 0..6 {
            let path = tmp.path().join(format!("{index}.state"));
            fs::write(&path, index.to_string()).unwrap();
            paths.push(path);
        }
        let current = paths[0].clone();

        prune_session_cache(tmp.path(), &current, 4).unwrap();

        let remaining_bytes = fs::read_dir(tmp.path())
            .unwrap()
            .map(|entry| entry.unwrap().metadata().unwrap().len())
            .sum::<u64>();
        assert!(remaining_bytes <= 4);
        assert!(current.is_file());
    }

    #[test]
    fn ensure_prompt_fits_context_allows_generation_room() {
        ensure_prompt_fits_context(8_000, 128, 8_192).unwrap();
    }

    #[test]
    fn ensure_prompt_fits_context_rejects_overflow_with_actionable_message() {
        let err = ensure_prompt_fits_context(8_100, 128, 8_192)
            .unwrap_err()
            .to_string();

        assert!(err.contains("prompt is too long"), "error was: {err}");
        assert!(err.contains("--ctx-size"), "error was: {err}");
    }

    #[test]
    fn structured_generation_detects_the_first_complete_json_value() {
        assert_eq!(complete_json_value_end("{\"tasks\":[\"one\"]}"), Some(17));
        assert_eq!(
            complete_json_value_end("{\"tasks\":[\"one\"]}\nignored"),
            Some(17)
        );
        assert_eq!(complete_json_value_end("{\"tasks\":[\"one\"]"), None);
    }

    #[test]
    fn vision_projector_is_resolved_from_model_cache() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model = tmp.path().join("qwen-vision.gguf");
        let projector = tmp.path().join("mmproj-qwen-vision.gguf");
        std::fs::write(&model, b"GGUF model").unwrap();
        std::fs::write(&projector, b"GGUF projector").unwrap();

        assert_eq!(find_multimodal_projector(&model).unwrap(), projector);
    }
}
