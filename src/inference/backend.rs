use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::energy::{self, EnergyEstimate};
use crate::inference::flashmoe::{ChatMessage, ChatTool, ChatToolCall};

#[derive(Debug, Clone, Default)]
pub struct BackendGenerationOptions {
    pub ctx_size: Option<u32>,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub gpu_layers: Option<u32>,
    pub max_tokens: i32,
    pub top_k: i32,
    pub temperature: f32,
    pub seed: u32,
}

#[derive(Debug, Clone)]
pub struct TextInferenceRequest {
    pub prompt: String,
    pub raw_prompt: bool,
    pub options: BackendGenerationOptions,
}

#[derive(Debug, Clone)]
pub struct ChatInferenceRequest {
    pub session_id: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ChatTool>,
    pub add_generation_prompt: bool,
    pub options: BackendGenerationOptions,
}

#[derive(Debug, Clone)]
pub struct VisionInferenceRequest {
    pub prompt: String,
    pub image_path: PathBuf,
    pub options: BackendGenerationOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFinishReason {
    EndOfGeneration,
    MaxTokens,
}

#[derive(Debug, Clone)]
pub struct BackendOutput {
    pub content: String,
    pub tool_calls: Vec<ChatToolCall>,
    pub finish_reason: BackendFinishReason,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub duration_ms: u64,
    pub energy: Option<EnergyEstimate>,
}

pub trait InferenceBackend {
    fn generate_text(&mut self, request: &TextInferenceRequest) -> Result<BackendOutput>;
    fn generate_chat(&mut self, request: &ChatInferenceRequest) -> Result<BackendOutput>;
    fn generate_vision(&mut self, request: &VisionInferenceRequest) -> Result<BackendOutput>;
}

fn llama_ctx_size(options: &BackendGenerationOptions) -> Result<u32> {
    options
        .ctx_size
        .context("llama.cpp inference requires ctx_size")
}

fn llama_gpu_layers(options: &BackendGenerationOptions) -> u32 {
    options.gpu_layers.unwrap_or(0)
}

fn duration_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

impl InferenceBackend for crate::inference::llamacpp::LlamaCppBackend {
    fn generate_text(&mut self, request: &TextInferenceRequest) -> Result<BackendOutput> {
        let output = self.generate(&crate::inference::llamacpp::LlamaCppRequest {
            prompt: request.prompt.clone(),
            ctx_size: llama_ctx_size(&request.options)?,
            threads: request.options.threads,
            threads_batch: request.options.threads_batch,
            gpu_layers: llama_gpu_layers(&request.options),
            max_tokens: request.options.max_tokens,
            top_k: request.options.top_k,
            temperature: request.options.temperature,
            seed: request.options.seed,
        })?;
        Ok(output.into())
    }

    fn generate_chat(&mut self, request: &ChatInferenceRequest) -> Result<BackendOutput> {
        let output = crate::inference::llamacpp::LlamaCppBackend::generate_chat(
            self,
            &crate::inference::llamacpp::LlamaCppChatRequest {
                messages: serde_json::to_value(&request.messages)?,
                tools: serde_json::to_value(&request.tools)?,
                ctx_size: llama_ctx_size(&request.options)?,
                threads: request.options.threads,
                threads_batch: request.options.threads_batch,
                gpu_layers: llama_gpu_layers(&request.options),
                max_tokens: request.options.max_tokens,
                top_k: request.options.top_k,
                temperature: request.options.temperature,
                seed: request.options.seed,
            },
        )?;
        Ok(output.into())
    }

    fn generate_vision(&mut self, request: &VisionInferenceRequest) -> Result<BackendOutput> {
        let output = crate::inference::llamacpp::LlamaCppBackend::generate_vision(
            self,
            &crate::inference::llamacpp::LlamaCppRequest {
                prompt: request.prompt.clone(),
                ctx_size: llama_ctx_size(&request.options)?,
                threads: request.options.threads,
                threads_batch: request.options.threads_batch,
                gpu_layers: llama_gpu_layers(&request.options),
                max_tokens: request.options.max_tokens,
                top_k: request.options.top_k,
                temperature: request.options.temperature,
                seed: request.options.seed,
            },
            &request.image_path,
        )?;
        Ok(output.into())
    }
}

impl From<crate::inference::llamacpp::Output> for BackendOutput {
    fn from(output: crate::inference::llamacpp::Output) -> Self {
        Self {
            content: output.content,
            tool_calls: Vec::new(),
            finish_reason: match output.finish_reason {
                crate::inference::llamacpp::FinishReason::EndOfGeneration => {
                    BackendFinishReason::EndOfGeneration
                }
                crate::inference::llamacpp::FinishReason::MaxTokens => {
                    BackendFinishReason::MaxTokens
                }
            },
            prompt_tokens: output.prompt_tokens,
            generated_tokens: output.generated_tokens,
            duration_ms: output.duration_ms,
            energy: output.energy,
        }
    }
}

impl InferenceBackend for crate::inference::flashmoe::FlashMoeEngine {
    fn generate_text(&mut self, request: &TextInferenceRequest) -> Result<BackendOutput> {
        let energy_start = energy::sample();
        let started = Instant::now();
        let generation_request = crate::inference::flashmoe::GenerationRequest {
            prompt: request.prompt.clone(),
            max_tokens: request.options.max_tokens,
            temperature: request.options.temperature,
            top_k: request.options.top_k,
            seed: request.options.seed,
        };
        let output = if request.raw_prompt {
            self.generate_raw(&generation_request)?
        } else {
            self.generate(&generation_request)?
        };
        let energy =
            energy_start.and_then(|sample| sample.estimate_since(energy::sample(), started));
        Ok(flashmoe_output_to_backend(
            output,
            BackendFinishReason::EndOfGeneration,
            0,
            duration_millis(started),
            energy,
        ))
    }

    fn generate_chat(&mut self, request: &ChatInferenceRequest) -> Result<BackendOutput> {
        let energy_start = energy::sample();
        let started = Instant::now();
        let structured = crate::inference::flashmoe::StructuredGenerationRequest {
            messages: request.messages.clone(),
            tools: request.tools.clone(),
            add_generation_prompt: request.add_generation_prompt,
            raw_prompt: false,
            max_tokens: request.options.max_tokens,
            temperature: request.options.temperature,
            top_k: request.options.top_k,
            seed: request.options.seed,
        };
        let output = if let Some(session_id) = request.session_id.as_deref() {
            self.generate_structured_in_session(session_id, &structured)?
        } else {
            self.generate_structured(&structured)?
        };
        let energy =
            energy_start.and_then(|sample| sample.estimate_since(energy::sample(), started));
        Ok(flashmoe_output_to_backend(
            output,
            BackendFinishReason::EndOfGeneration,
            request
                .messages
                .iter()
                .map(|message| format!("{:?}", message.content).len())
                .sum(),
            duration_millis(started),
            energy,
        ))
    }

    fn generate_vision(&mut self, request: &VisionInferenceRequest) -> Result<BackendOutput> {
        let energy_start = energy::sample();
        let started = Instant::now();
        let output =
            self.generate_with_image(&crate::inference::flashmoe::VisionGenerationRequest {
                prompt: request.prompt.clone(),
                image_path: request.image_path.clone(),
                max_tokens: request.options.max_tokens,
                temperature: request.options.temperature,
                top_k: request.options.top_k,
                seed: request.options.seed,
            })?;
        let energy =
            energy_start.and_then(|sample| sample.estimate_since(energy::sample(), started));
        Ok(flashmoe_output_to_backend(
            output,
            BackendFinishReason::EndOfGeneration,
            0,
            duration_millis(started),
            energy,
        ))
    }
}

fn flashmoe_output_to_backend(
    output: crate::inference::flashmoe::GenerationOutput,
    finish_reason: BackendFinishReason,
    prompt_tokens: usize,
    duration_ms: u64,
    energy: Option<EnergyEstimate>,
) -> BackendOutput {
    BackendOutput {
        content: output.content,
        tool_calls: output.tool_calls,
        finish_reason,
        prompt_tokens,
        generated_tokens: output.generated_tokens,
        duration_ms,
        energy,
    }
}
