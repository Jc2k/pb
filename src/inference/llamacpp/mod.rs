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

use anyhow::{Context, Result, bail};
use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker,
};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use serde_json::Value;

use crate::energy::{self, EnergyEstimate};
use crate::inference::chat_template::{ChatTemplateOptions, TokenizerChatTemplate};

const BATCH_SIZE: usize = 512;
const MIN_GENERATION_CONTEXT_TOKENS: usize = 1;

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
    })
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
        )
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
    ) -> Result<Output> {
        let energy_start = energy::sample();
        let started = std::time::Instant::now();
        let n_ctx = NonZeroU32::new(ctx_size).context("ctx-size must be > 0")?;
        let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
        if let Some(threads) = threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = threads_batch.or(threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }

        let mut ctx = match self.model.new_context(&self.backend, ctx_params.clone()) {
            Ok(ctx) => ctx,
            Err(accelerated_error) => self
                .model
                .new_context(&self.backend, ctx_params.with_offload_kqv(false))
                .with_context(|| {
                    format!(
                        "failed to create llama context, including CPU K/Q/V fallback after accelerated context error: {accelerated_error}"
                    )
                })?,
        };

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

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(top_k),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(seed),
        ]);

        let mut decoder = UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur =
            i32::try_from(tokens.len()).context("prompt token count exceeds i32::MAX")?;
        let mut generated_tokens: usize = 0;
        let mut finish_reason = FinishReason::MaxTokens;

        while generated_tokens < usize::try_from(max_tokens).unwrap_or(0) {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
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
        }

        let energy =
            energy_start.and_then(|sample| sample.estimate_since(energy::sample(), started));
        Ok(Output {
            content: output,
            finish_reason,
            prompt_tokens: tokens.len(),
            generated_tokens,
            duration_ms: duration_millis(started),
            energy,
        })
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
            let token = sampler.sample(&ctx, sample_index);
            sampler.accept(token);
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
            duration_ms: duration_millis(started),
            energy,
        })
    }
}

impl LlamaCppBackend {
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
        let Some(chat_template) = &self.chat_template else {
            return Ok((render_plain_chat_prompt(messages), AddBos::Always));
        };
        let rendered = chat_template.render(
            messages,
            tools,
            ChatTemplateOptions {
                add_generation_prompt: true,
                ..ChatTemplateOptions::default()
            },
        )?;
        Ok((rendered, AddBos::Never))
    }
}

fn render_plain_chat_prompt(messages: &Value) -> String {
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
    prompt.push_str("[assistant]\n");
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

fn prompt_batch_ranges(token_count: usize, batch_size: usize) -> Vec<Range<usize>> {
    assert!(batch_size > 0, "batch_size must be greater than zero");
    (0..token_count)
        .step_by(batch_size)
        .map(|start| start..std::cmp::min(start + batch_size, token_count))
        .collect()
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
    fn vision_projector_is_resolved_from_model_cache() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model = tmp.path().join("qwen-vision.gguf");
        let projector = tmp.path().join("mmproj-qwen-vision.gguf");
        std::fs::write(&model, b"GGUF model").unwrap();
        std::fs::write(&projector, b"GGUF projector").unwrap();

        assert_eq!(find_multimodal_projector(&model).unwrap(), projector);
    }
}
