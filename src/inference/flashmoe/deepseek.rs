//! DeepSeek V4 Flash source adapter for the existing FlashMoe runtime.
//!
//! The adapter accepts one pinned GGUF profile, resolves every semantic tensor
//! before publication, and preserves GGUF blocks in FlashMoe's canonical
//! resident and page-aligned expert stores. Runtime code consumes only those
//! stores and a load-time graph manifest; it never makes fallback decisions by
//! inspecting the source model.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::experts::{
    ExpertLayerPackMetadata, ExpertPackMetadata, ExpertPackRecord, expert_layer_path,
    finish_expert_pack_atomically, temp_pack_path, write_expert_metadata_atomically,
};
use super::gguf::{GgufFile, GgufMetadataType, GgufTensorInfo, GgufTensorType, GgufValue};
use super::metal::METAL_SHADERS;
use super::model_family::QwenModelConfig;
use super::planning::{FlashMoePlan, FlashMoeRoutingPolicy, plan_unchecked_with_cache_version};
use super::weights::{
    DenseTensorRef, ExpertTensorRef, FlashMoeManifest, RuntimeTensorEntry, TENSOR_ALIGNMENT,
    TensorQuantization, TensorRegistry,
};

pub const DEEPSEEK_V4_FLASH_REPOSITORY: &str = "hf://antirez/deepseek-v4-gguf";
pub const DEEPSEEK_V4_FLASH_FILENAME: &str =
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
pub const DEEPSEEK_V4_FLASH_MODEL: &str = "hf://antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
pub const DEEPSEEK_V4_FLASH_CACHE_VERSION: &str = "flashmoe-deepseek-v4-flash-v1";

const DEEPSEEK_TOKENIZER_FORMAT: &str = "PB_DEEPSEEK_V4_JOYAI_BPE_V1";
const DEEPSEEK_EXPERT_SCALE_BIAS_DTYPE: &str = "GGUF_NATIVE";
const EXPERT_COMPONENT_ALIGNMENT: u64 = 4096;
const COPY_BUFFER_BYTES: usize = 8 * 1024 * 1024;

pub fn is_deepseek_v4_flash(model: &str) -> bool {
    let normalized = model.trim_end_matches('/').to_ascii_lowercase();
    normalized == DEEPSEEK_V4_FLASH_REPOSITORY.to_ascii_lowercase()
        || normalized == DEEPSEEK_V4_FLASH_MODEL.to_ascii_lowercase()
}

pub(crate) fn canonical_deepseek_v4_flash_model(model: &str) -> Option<&'static str> {
    is_deepseek_v4_flash(model).then_some(DEEPSEEK_V4_FLASH_MODEL)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeepSeekV4Config {
    pub architecture: String,
    pub block_count: usize,
    pub embedding_length: usize,
    pub vocab_size: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub attention_key_length: usize,
    pub attention_value_length: usize,
    pub rope_dimension_count: usize,
    pub q_lora_rank: usize,
    pub output_lora_rank: usize,
    pub output_group_count: usize,
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub expert_feed_forward_length: usize,
    pub expert_shared_count: usize,
    pub hash_layer_count: usize,
    pub sliding_window: usize,
    pub indexer_head_count: usize,
    pub indexer_key_length: usize,
    pub indexer_top_k: usize,
    pub hyper_connection_count: usize,
    pub hyper_connection_sinkhorn_iterations: usize,
    pub compress_ratios: Vec<usize>,
    pub swiglu_clamp_exp: Vec<f32>,
    pub rope_original_context_length: usize,
    pub rope_freq_base: f32,
    pub rope_scaling_factor: f32,
    pub rope_yarn_beta_fast: f32,
    pub rope_yarn_beta_slow: f32,
    pub compress_rope_freq_base: f32,
    pub expert_weights_scale: f32,
    pub attention_layer_norm_rms_epsilon: f32,
    pub hyper_connection_epsilon: f32,
    pub expert_weights_norm: bool,
}

impl DeepSeekV4Config {
    pub(crate) fn from_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read DeepSeek config {}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse DeepSeek config {}", path.display()))?;
        config.validate_cached_flash_profile()?;
        Ok(config)
    }

    pub(crate) fn expected_flash_profile() -> Self {
        Self {
            architecture: "deepseek4".to_string(),
            block_count: 43,
            embedding_length: 4096,
            vocab_size: 129_280,
            attention_head_count: 64,
            attention_head_count_kv: 1,
            attention_key_length: 512,
            attention_value_length: 512,
            rope_dimension_count: 64,
            q_lora_rank: 1024,
            output_lora_rank: 1024,
            output_group_count: 8,
            expert_count: 256,
            expert_used_count: 6,
            expert_feed_forward_length: 2048,
            expert_shared_count: 1,
            hash_layer_count: 3,
            sliding_window: 128,
            indexer_head_count: 64,
            indexer_key_length: 128,
            indexer_top_k: 512,
            hyper_connection_count: 4,
            hyper_connection_sinkhorn_iterations: 20,
            compress_ratios: (0..43)
                .map(|layer| {
                    if layer < 2 {
                        0
                    } else if layer % 2 == 0 {
                        4
                    } else {
                        128
                    }
                })
                .collect(),
            swiglu_clamp_exp: vec![10.0; 43],
            rope_original_context_length: 65_536,
            rope_freq_base: 10_000.0,
            rope_scaling_factor: 16.0,
            rope_yarn_beta_fast: 32.0,
            rope_yarn_beta_slow: 1.0,
            compress_rope_freq_base: 160_000.0,
            expert_weights_scale: 1.5,
            attention_layer_norm_rms_epsilon: 1.0e-6,
            hyper_connection_epsilon: 1.0e-6,
            expert_weights_norm: true,
        }
    }

    pub(crate) fn validate_cached_flash_profile(&self) -> Result<()> {
        let expected = Self::expected_flash_profile();
        if self != &expected {
            bail!(
                "DeepSeek V4 Flash cached config does not match the pinned 43-layer IQ2_XXS/Q2_K execution profile"
            );
        }
        Ok(())
    }

    pub(crate) fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = required_string(gguf, "general.architecture")?;
        if architecture != "deepseek4" {
            bail!(
                "DeepSeek V4 Flash GGUF declares architecture {architecture:?}, expected \"deepseek4\""
            );
        }
        let mut config = Self {
            architecture,
            block_count: required_usize(gguf, "deepseek4.block_count")?,
            embedding_length: required_usize(gguf, "deepseek4.embedding_length")?,
            vocab_size: required_usize(gguf, "deepseek4.vocab_size")?,
            attention_head_count: required_usize(gguf, "deepseek4.attention.head_count")?,
            attention_head_count_kv: required_usize(gguf, "deepseek4.attention.head_count_kv")?,
            attention_key_length: required_usize(gguf, "deepseek4.attention.key_length")?,
            attention_value_length: required_usize(gguf, "deepseek4.attention.value_length")?,
            rope_dimension_count: required_usize(gguf, "deepseek4.rope.dimension_count")?,
            q_lora_rank: required_usize(gguf, "deepseek4.attention.q_lora_rank")?,
            output_lora_rank: required_usize(gguf, "deepseek4.attention.output_lora_rank")?,
            output_group_count: required_usize(gguf, "deepseek4.attention.output_group_count")?,
            expert_count: required_usize(gguf, "deepseek4.expert_count")?,
            expert_used_count: required_usize(gguf, "deepseek4.expert_used_count")?,
            expert_feed_forward_length: required_usize(
                gguf,
                "deepseek4.expert_feed_forward_length",
            )?,
            expert_shared_count: required_usize(gguf, "deepseek4.expert_shared_count")?,
            hash_layer_count: required_usize(gguf, "deepseek4.hash_layer_count")?,
            sliding_window: required_usize(gguf, "deepseek4.attention.sliding_window")?,
            indexer_head_count: required_usize(gguf, "deepseek4.attention.indexer.head_count")?,
            indexer_key_length: required_usize(gguf, "deepseek4.attention.indexer.key_length")?,
            indexer_top_k: required_usize(gguf, "deepseek4.attention.indexer.top_k")?,
            hyper_connection_count: required_usize(gguf, "deepseek4.hyper_connection.count")?,
            hyper_connection_sinkhorn_iterations: required_usize(
                gguf,
                "deepseek4.hyper_connection.sinkhorn_iterations",
            )?,
            compress_ratios: required_usize_array(gguf, "deepseek4.attention.compress_ratios")?,
            swiglu_clamp_exp: required_f32_array(gguf, "deepseek4.swiglu_clamp_exp")?,
            rope_original_context_length: optional_usize(
                gguf,
                "deepseek4.rope.scaling.original_context_length",
            )?
            .unwrap_or(65_536),
            rope_freq_base: required_f32(gguf, "deepseek4.rope.freq_base")?,
            rope_scaling_factor: optional_f32(gguf, "deepseek4.rope.scaling.factor")?
                .unwrap_or(16.0),
            rope_yarn_beta_fast: optional_f32(gguf, "deepseek4.rope.scaling.yarn_beta_fast")?
                .unwrap_or(32.0),
            rope_yarn_beta_slow: optional_f32(gguf, "deepseek4.rope.scaling.yarn_beta_slow")?
                .unwrap_or(1.0),
            compress_rope_freq_base: required_f32(
                gguf,
                "deepseek4.attention.compress_rope_freq_base",
            )?,
            expert_weights_scale: required_f32(gguf, "deepseek4.expert_weights_scale")?,
            attention_layer_norm_rms_epsilon: required_f32(
                gguf,
                "deepseek4.attention.layer_norm_rms_epsilon",
            )?,
            hyper_connection_epsilon: required_f32(gguf, "deepseek4.hyper_connection.epsilon")?,
            expert_weights_norm: gguf
                .required_metadata("deepseek4.expert_weights_norm")?
                .as_bool()
                .context("GGUF metadata deepseek4.expert_weights_norm must be bool")?,
        };
        config.validate_flash_profile(gguf)?;
        // Published Flash GGUFs may append one MTP entry to per-layer metadata.
        // The resolved base graph has exactly block_count layers, so normalize
        // the already-validated prefix before serializing the runtime config.
        config.compress_ratios.truncate(config.block_count);
        config.swiglu_clamp_exp.truncate(config.block_count);
        config.validate_cached_flash_profile()?;
        Ok(config)
    }

    fn validate_flash_profile(&self, gguf: &GgufFile) -> Result<()> {
        let expected = [
            ("block_count", self.block_count, 43),
            ("embedding_length", self.embedding_length, 4096),
            ("vocab_size", self.vocab_size, 129_280),
            ("attention.head_count", self.attention_head_count, 64),
            ("attention.head_count_kv", self.attention_head_count_kv, 1),
            ("attention.key_length", self.attention_key_length, 512),
            ("attention.value_length", self.attention_value_length, 512),
            ("rope.dimension_count", self.rope_dimension_count, 64),
            ("attention.q_lora_rank", self.q_lora_rank, 1024),
            ("attention.output_lora_rank", self.output_lora_rank, 1024),
            ("attention.output_group_count", self.output_group_count, 8),
            ("expert_count", self.expert_count, 256),
            ("expert_used_count", self.expert_used_count, 6),
            (
                "expert_feed_forward_length",
                self.expert_feed_forward_length,
                2048,
            ),
            ("expert_shared_count", self.expert_shared_count, 1),
            ("hash_layer_count", self.hash_layer_count, 3),
            ("attention.sliding_window", self.sliding_window, 128),
            ("attention.indexer.head_count", self.indexer_head_count, 64),
            ("attention.indexer.key_length", self.indexer_key_length, 128),
            ("attention.indexer.top_k", self.indexer_top_k, 512),
            ("hyper_connection.count", self.hyper_connection_count, 4),
            (
                "hyper_connection.sinkhorn_iterations",
                self.hyper_connection_sinkhorn_iterations,
                20,
            ),
            (
                "rope.scaling.original_context_length",
                self.rope_original_context_length,
                65_536,
            ),
        ];
        for (name, actual, wanted) in expected {
            if actual != wanted {
                bail!("unsupported DeepSeek V4 Flash {name}: got {actual}, expected {wanted}");
            }
        }
        for key in [
            "deepseek4.expert_group_count",
            "deepseek4.expert_group_used_count",
        ] {
            if optional_usize(gguf, key)?.unwrap_or(0) != 0 {
                bail!("DeepSeek V4 Flash grouped routing is unsupported: {key} must be zero");
            }
        }

        let expected_ratios: Vec<usize> = (0..43)
            .map(|layer| {
                if layer < 2 {
                    0
                } else if layer % 2 == 0 {
                    4
                } else {
                    128
                }
            })
            .collect();
        if self.compress_ratios.len() < self.block_count
            || self.compress_ratios[..self.block_count] != expected_ratios
        {
            bail!(
                "DeepSeek V4 Flash attention.compress_ratios does not contain the fixed 43-layer Flash graph prefix: got {:?}, expected prefix {:?}",
                self.compress_ratios,
                expected_ratios
            );
        }
        if self.swiglu_clamp_exp.len() < self.block_count
            || self
                .swiglu_clamp_exp
                .iter()
                .take(self.block_count)
                .any(|value| !float_matches(*value, 10.0))
        {
            bail!("DeepSeek V4 Flash swiglu_clamp_exp must contain a 43-layer prefix equal to 10");
        }
        for (name, actual, wanted) in [
            ("rope.freq_base", self.rope_freq_base, 10_000.0),
            ("rope.scaling.factor", self.rope_scaling_factor, 16.0),
            (
                "rope.scaling.yarn_beta_fast",
                self.rope_yarn_beta_fast,
                32.0,
            ),
            ("rope.scaling.yarn_beta_slow", self.rope_yarn_beta_slow, 1.0),
            (
                "attention.compress_rope_freq_base",
                self.compress_rope_freq_base,
                160_000.0,
            ),
            ("expert_weights_scale", self.expert_weights_scale, 1.5),
            (
                "attention.layer_norm_rms_epsilon",
                self.attention_layer_norm_rms_epsilon,
                1.0e-6,
            ),
            (
                "hyper_connection.epsilon",
                self.hyper_connection_epsilon,
                1.0e-6,
            ),
        ] {
            if !float_matches(actual, wanted) {
                bail!("unsupported DeepSeek V4 Flash {name}: got {actual}, expected {wanted}");
            }
        }
        if !self.expert_weights_norm {
            bail!("DeepSeek V4 Flash requires expert_weights_norm=true");
        }
        Ok(())
    }

    /// Adapts the family-neutral dimensions consumed by the shared FlashMoe
    /// scheduler. DeepSeek-only graph/state semantics stay in the typed
    /// DeepSeek execution descriptor and are never inferred from this view.
    pub(crate) fn shared_runtime_config(&self) -> QwenModelConfig {
        QwenModelConfig {
            model_type: Some("deepseek_v4_flash".to_string()),
            architectures: Some(vec!["DeepSeekV4FlashForCausalLM".to_string()]),
            num_hidden_layers: self.block_count,
            hidden_size: self.embedding_length,
            num_attention_heads: self.attention_head_count,
            head_dim: Some(self.attention_key_length),
            num_key_value_heads: Some(self.attention_head_count_kv),
            vocab_size: self.vocab_size,
            rope_theta: Some(self.rope_freq_base as f64),
            partial_rotary_factor: Some(
                self.rope_dimension_count as f64 / self.attention_key_length as f64,
            ),
            torch_dtype: Some("float16".to_string()),
            num_experts: Some(self.expert_count),
            num_experts_per_tok: Some(self.expert_used_count),
            norm_topk_prob: Some(self.expert_weights_norm),
            moe_intermediate_size: Some(self.expert_feed_forward_length),
            intermediate_size: Some(self.expert_feed_forward_length),
            max_position_embeddings: Some(
                self.rope_original_context_length
                    .saturating_mul(self.rope_scaling_factor as usize),
            ),
            full_attention_interval: None,
            linear_attention: None,
            mrope_section: None,
            tie_word_embeddings: Some(false),
            num_shared_experts: Some(self.expert_shared_count),
            shared_expert_intermediate_size: Some(self.expert_feed_forward_length),
            vision_config: None,
            glm: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct DeepSeekV4TokenizerCache {
    format: String,
    model: String,
    pre_tokenizer: String,
    tokens: Vec<String>,
    merges: Vec<String>,
    special_tokens: BTreeMap<String, u32>,
}

/// Native tokenizer bound from the DeepSeek cache artifact.  This remains an
/// input adapter of the existing FlashMoe engine; it is not a second inference
/// runtime.  The implementation mirrors the pinned `joyai-llm` GPT-2
/// byte-level BPE profile used by the reference model.
#[derive(Debug, Clone)]
pub(crate) struct DeepSeekV4Tokenizer {
    tokens: Vec<String>,
    token_to_id: BTreeMap<String, u32>,
    merge_rank: BTreeMap<String, usize>,
    special_tokens: BTreeMap<String, u32>,
    eos_id: u32,
}

impl DeepSeekV4Tokenizer {
    pub(crate) fn from_cache_bytes(bytes: &[u8]) -> Result<Self> {
        let cache: DeepSeekV4TokenizerCache = serde_json::from_slice(bytes)
            .context("DeepSeek V4 Flash tokenizer cache JSON is invalid")?;
        if cache.format != DEEPSEEK_TOKENIZER_FORMAT
            || cache.model != "gpt2"
            || cache.pre_tokenizer != "joyai-llm"
        {
            bail!(
                "DeepSeek V4 Flash tokenizer cache does not declare the pinned joyai-llm GPT-2 byte-BPE profile"
            );
        }
        if cache.tokens.len() != DeepSeekV4Config::expected_flash_profile().vocab_size {
            bail!(
                "DeepSeek V4 Flash tokenizer cache contains {} tokens, expected {}",
                cache.tokens.len(),
                DeepSeekV4Config::expected_flash_profile().vocab_size
            );
        }
        let mut token_to_id = BTreeMap::new();
        for (index, token) in cache.tokens.iter().enumerate() {
            let id = u32::try_from(index).context("DeepSeek tokenizer token id exceeds u32")?;
            if token_to_id.insert(token.clone(), id).is_some() {
                bail!("DeepSeek V4 Flash tokenizer contains duplicate token {token:?}");
            }
        }
        let merge_rank = cache
            .merges
            .iter()
            .enumerate()
            .map(|(rank, merge)| (merge.clone(), rank))
            .collect::<BTreeMap<_, _>>();
        for (token, id) in &cache.special_tokens {
            if cache.tokens.get(*id as usize) != Some(token) {
                bail!(
                    "DeepSeek V4 Flash tokenizer special token {token:?} points at invalid id {id}"
                );
            }
        }
        for required in [
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<think>",
            "</think>",
            "｜DSML｜",
        ] {
            if !cache.special_tokens.contains_key(required) {
                bail!("DeepSeek V4 Flash tokenizer cache is missing special token {required}");
            }
        }
        let eos_id = cache.special_tokens["<｜end▁of▁sentence｜>"];
        Ok(Self {
            tokens: cache.tokens,
            token_to_id,
            merge_rank,
            special_tokens: cache.special_tokens,
            eos_id,
        })
    }

    pub(crate) fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let bytes = text.as_bytes();
        let mut span_start = 0usize;
        let mut position = 0usize;
        while position < bytes.len() {
            let matched = self.special_tokens.iter().find_map(|(special, id)| {
                bytes[position..]
                    .starts_with(special.as_bytes())
                    .then_some((special.len(), *id))
            });
            if let Some((len, id)) = matched {
                self.encode_text_bytes(&bytes[span_start..position], &mut out)?;
                out.push(id);
                position += len;
                span_start = position;
            } else {
                position += 1;
            }
        }
        self.encode_text_bytes(&bytes[span_start..], &mut out)?;
        Ok(out)
    }

    pub(crate) fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for id in token_ids {
            let token = self.tokens.get(*id as usize).with_context(|| {
                format!("DeepSeek V4 Flash tokenizer token id {id} is out of range")
            })?;
            if token.contains('｜') {
                bytes.extend_from_slice(token.as_bytes());
                continue;
            }
            for ch in token.chars() {
                if let Some(byte) = gpt2_codepoint_to_byte(ch as u32) {
                    bytes.push(byte);
                }
            }
        }
        String::from_utf8(bytes).context("DeepSeek V4 Flash tokenizer produced invalid UTF-8")
    }

    pub(crate) fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub(crate) fn eos_id(&self) -> u32 {
        self.eos_id
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn constraint_token_bytes(&self) -> Result<Vec<Vec<u8>>> {
        let special_ids = self
            .special_tokens
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        self.tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let id = u32::try_from(index).context("DeepSeek tokenizer token id exceeds u32")?;
                if special_ids.contains(&id) {
                    let mut bytes = Vec::with_capacity(token.len() + 1);
                    bytes.push(llguidance::toktrie::TokTrie::SPECIAL_TOKEN_MARKER);
                    bytes.extend_from_slice(token.as_bytes());
                    return Ok(bytes);
                }
                if token.contains('｜') {
                    return Ok(token.as_bytes().to_vec());
                }
                token
                    .chars()
                    .map(|character| {
                        gpt2_codepoint_to_byte(character as u32).with_context(|| {
                            format!(
                                "DeepSeek tokenizer token {id} contains unmapped GPT-2 codepoint U+{:04X}",
                                character as u32
                            )
                        })
                    })
                    .collect()
            })
            .collect()
    }

    fn encode_text_bytes(&self, text: &[u8], out: &mut Vec<u32>) -> Result<()> {
        let mut position = 0usize;
        while position < text.len() {
            let start = position;
            let byte = text[position];
            if byte.is_ascii_digit() {
                let mut digits = 0;
                while position < text.len() && text[position].is_ascii_digit() && digits < 3 {
                    position += 1;
                    digits += 1;
                }
            } else if joyai_cjk_at(text, position) {
                loop {
                    position = next_utf8_char(text, position);
                    if position >= text.len() || !joyai_cjk_at(text, position) {
                        break;
                    }
                }
            } else if joyai_ascii_punct_symbol(byte)
                && position + 1 < text.len()
                && text[position + 1].is_ascii_alphabetic()
            {
                position += 1;
                while position < text.len() && text[position].is_ascii_alphabetic() {
                    position += 1;
                }
            } else if joyai_letter_like_at(text, position) {
                position = joyai_consume_letters(text, position);
            } else if !ascii_newline(byte)
                && !joyai_ascii_punct_symbol(byte)
                && position + 1 < text.len()
                && joyai_letter_like_at(text, position + 1)
            {
                position += 1;
                position = joyai_consume_letters(text, position);
            } else if byte == b' '
                && position + 1 < text.len()
                && joyai_ascii_punct_symbol(text[position + 1])
            {
                position += 1;
                while position < text.len() && joyai_ascii_punct_symbol(text[position]) {
                    position += 1;
                }
                while position < text.len() && ascii_newline(text[position]) {
                    position += 1;
                }
            } else if joyai_ascii_punct_symbol(byte) {
                while position < text.len() && joyai_ascii_punct_symbol(text[position]) {
                    position += 1;
                }
                while position < text.len() && ascii_newline(text[position]) {
                    position += 1;
                }
            } else if byte.is_ascii_whitespace() {
                let mut cursor = position;
                let mut last_newline_end = None;
                while cursor < text.len() && text[cursor].is_ascii_whitespace() {
                    let current = text[cursor];
                    cursor += 1;
                    if ascii_newline(current) {
                        last_newline_end = Some(cursor);
                    }
                }
                position = if let Some(last_newline_end) = last_newline_end {
                    last_newline_end
                } else if cursor < text.len()
                    && cursor > position + 1
                    && (joyai_letter_like_at(text, cursor)
                        || joyai_ascii_punct_symbol(text[cursor]))
                {
                    cursor - 1
                } else {
                    cursor
                };
            } else {
                position = next_utf8_char(text, position);
            }
            if position == start {
                position = next_utf8_char(text, position);
            }
            self.emit_bpe_piece(&text[start..position], out)?;
        }
        Ok(())
    }

    fn emit_bpe_piece(&self, raw: &[u8], out: &mut Vec<u32>) -> Result<()> {
        let mut symbols = raw
            .iter()
            .map(|byte| {
                char::from_u32(gpt2_byte_to_codepoint(*byte))
                    .expect("GPT-2 byte alphabet always maps to a Unicode scalar")
                    .to_string()
            })
            .collect::<Vec<_>>();
        loop {
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(index, pair)| {
                    let key = format!("{} {}", pair[0], pair[1]);
                    self.merge_rank.get(&key).copied().map(|rank| (rank, index))
                })
                .min_by_key(|(rank, _)| *rank);
            let Some((_, index)) = best else {
                break;
            };
            let right = symbols.remove(index + 1);
            symbols[index].push_str(&right);
        }
        for symbol in symbols {
            if let Some(id) = self.token_to_id.get(&symbol) {
                out.push(*id);
                continue;
            }
            // Matches the reference's defensive byte-symbol fallback.  A
            // valid pinned vocabulary should always take the direct path.
            for ch in symbol.chars() {
                let single = ch.to_string();
                let id = self.token_to_id.get(&single).with_context(|| {
                    format!("DeepSeek tokenizer cannot encode BPE symbol {single:?}")
                })?;
                out.push(*id);
            }
        }
        Ok(())
    }
}

fn gpt2_byte_to_codepoint(byte: u8) -> u32 {
    if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
        return u32::from(byte);
    }
    let mut mapped = 256u32;
    for candidate in 0u16..=255 {
        let candidate = candidate as u8;
        if (33..=126).contains(&candidate) || (161..=172).contains(&candidate) || candidate >= 174 {
            continue;
        }
        if candidate == byte {
            return mapped;
        }
        mapped += 1;
    }
    u32::from(byte)
}

fn gpt2_codepoint_to_byte(codepoint: u32) -> Option<u8> {
    if (33..=126).contains(&codepoint)
        || (161..=172).contains(&codepoint)
        || (174..=255).contains(&codepoint)
    {
        return Some(codepoint as u8);
    }
    let mut mapped = 256u32;
    for candidate in 0u16..=255 {
        let candidate = candidate as u8;
        if (33..=126).contains(&candidate) || (161..=172).contains(&candidate) || candidate >= 174 {
            continue;
        }
        if mapped == codepoint {
            return Some(candidate);
        }
        mapped += 1;
    }
    None
}

fn next_utf8_char(text: &[u8], position: usize) -> usize {
    let width = match text.get(position).copied().unwrap_or_default() {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => 1,
    };
    position.saturating_add(width).min(text.len())
}

fn utf8_codepoint_at(text: &[u8], position: usize) -> Option<u32> {
    let end = next_utf8_char(text, position);
    std::str::from_utf8(text.get(position..end)?)
        .ok()?
        .chars()
        .next()
        .map(|ch| ch as u32)
}

fn joyai_cjk_at(text: &[u8], position: usize) -> bool {
    matches!(
        utf8_codepoint_at(text, position),
        Some(0x4e00..=0x9fa5 | 0x3040..=0x309f | 0x30a0..=0x30ff)
    )
}

fn joyai_letter_like_at(text: &[u8], position: usize) -> bool {
    text.get(position)
        .is_some_and(|byte| !byte.is_ascii() || byte.is_ascii_alphabetic())
}

fn joyai_consume_letters(text: &[u8], mut position: usize) -> usize {
    while position < text.len() && joyai_letter_like_at(text, position) {
        position = next_utf8_char(text, position);
    }
    position
}

fn joyai_ascii_punct_symbol(byte: u8) -> bool {
    byte.is_ascii_punctuation()
}

fn ascii_newline(byte: u8) -> bool {
    matches!(byte, b'\n' | b'\r')
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeepSeekResidentDtype {
    F16,
    F32,
    I32,
    Q8_0,
}

impl DeepSeekResidentDtype {
    const fn gguf(self) -> GgufTensorType {
        match self {
            Self::F16 => GgufTensorType::F16,
            Self::F32 => GgufTensorType::F32,
            Self::I32 => GgufTensorType::I32,
            Self::Q8_0 => GgufTensorType::Q8_0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeepSeekResidentTensor {
    pub(crate) name: String,
    pub(crate) dtype: DeepSeekResidentDtype,
    pub(crate) shape: Vec<usize>,
    pub(crate) byte_offset: u64,
    pub(crate) byte_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeepSeekResidentRange {
    pub(crate) byte_offset: u64,
    pub(crate) byte_len: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct DeepSeekV4CompressorGraph {
    pub(crate) ratio: usize,
    pub(crate) ape: DeepSeekResidentTensor,
    pub(crate) kv: DeepSeekResidentTensor,
    pub(crate) gate: DeepSeekResidentTensor,
    pub(crate) norm: DeepSeekResidentTensor,
}

#[derive(Debug, Clone)]
pub(crate) struct DeepSeekV4IndexerGraph {
    pub(crate) q_b: DeepSeekResidentTensor,
    pub(crate) projection: DeepSeekResidentTensor,
    pub(crate) compressor_ape: DeepSeekResidentTensor,
    pub(crate) compressor_kv: DeepSeekResidentTensor,
    pub(crate) compressor_gate: DeepSeekResidentTensor,
    pub(crate) compressor_norm: DeepSeekResidentTensor,
}

#[derive(Debug, Clone)]
pub(crate) struct DeepSeekV4LayerGraph {
    pub(crate) hc_attn_fn: DeepSeekResidentTensor,
    pub(crate) hc_attn_scale: DeepSeekResidentTensor,
    pub(crate) hc_attn_base: DeepSeekResidentTensor,
    pub(crate) attn_norm: DeepSeekResidentTensor,
    pub(crate) attn_q_a: DeepSeekResidentTensor,
    pub(crate) attn_q_a_norm: DeepSeekResidentTensor,
    pub(crate) attn_q_b: DeepSeekResidentTensor,
    pub(crate) attn_kv: DeepSeekResidentTensor,
    pub(crate) attn_kv_a_norm: DeepSeekResidentTensor,
    pub(crate) attn_sinks: DeepSeekResidentTensor,
    pub(crate) attn_output_a: DeepSeekResidentTensor,
    pub(crate) attn_output_b: DeepSeekResidentTensor,
    pub(crate) hc_ffn_fn: DeepSeekResidentTensor,
    pub(crate) hc_ffn_scale: DeepSeekResidentTensor,
    pub(crate) hc_ffn_base: DeepSeekResidentTensor,
    pub(crate) ffn_norm: DeepSeekResidentTensor,
    pub(crate) router: DeepSeekResidentTensor,
    pub(crate) router_bias: Option<DeepSeekResidentTensor>,
    pub(crate) shared_gate: DeepSeekResidentTensor,
    pub(crate) shared_up: DeepSeekResidentTensor,
    pub(crate) shared_down: DeepSeekResidentTensor,
    pub(crate) compressor: Option<DeepSeekV4CompressorGraph>,
    pub(crate) indexer: Option<DeepSeekV4IndexerGraph>,
    pub(crate) token_hash_routes: Option<DeepSeekResidentTensor>,
}

/// Fully resolved, family-typed graph descriptor.  Every tensor, compression
/// mode, routing mode, and kernel-relevant shape is fixed once during load.
/// Token execution never probes the registry or selects an alternative graph.
#[derive(Debug, Clone)]
pub(crate) struct DeepSeekV4ExecutionGraph {
    pub(crate) config: DeepSeekV4Config,
    /// Batch geometry at or above this loaded expert count uses the fixed
    /// one-layer-ahead positioned-read preparation schedule.
    pub(crate) prefill_layer_prepare_min_tokens: usize,
    /// Exact resident tensor ranges prepared one layer ahead alongside the
    /// scheduler-owned routed-expert stream. These ranges are resolved once
    /// from the bound graph and never rediscovered by token execution.
    pub(crate) prefill_resident_layer_ranges: Vec<Box<[DeepSeekResidentRange]>>,
    pub(crate) embedding: DeepSeekResidentTensor,
    pub(crate) output_hc_base: DeepSeekResidentTensor,
    pub(crate) output_hc_fn: DeepSeekResidentTensor,
    pub(crate) output_hc_scale: DeepSeekResidentTensor,
    pub(crate) output_norm: DeepSeekResidentTensor,
    pub(crate) output: DeepSeekResidentTensor,
    pub(crate) layers: Vec<DeepSeekV4LayerGraph>,
}

impl DeepSeekV4ExecutionGraph {
    pub(crate) fn from_registry(
        config: DeepSeekV4Config,
        registry: &TensorRegistry,
        store_len: u64,
    ) -> Result<Self> {
        config.validate_cached_flash_profile()?;
        let tensor = |name: &str, dtype, shape: &[usize]| {
            bind_deepseek_tensor(registry, store_len, name, dtype, shape)
        };
        let embedding = tensor(
            "token_embd.weight",
            DeepSeekResidentDtype::F16,
            &[4096, 129_280],
        )?;
        let output_hc_base = tensor("output_hc_base.weight", DeepSeekResidentDtype::F32, &[4])?;
        let output_hc_fn = tensor(
            "output_hc_fn.weight",
            DeepSeekResidentDtype::F16,
            &[16_384, 4],
        )?;
        let output_hc_scale = tensor("output_hc_scale.weight", DeepSeekResidentDtype::F32, &[1])?;
        let output_norm = tensor("output_norm.weight", DeepSeekResidentDtype::F32, &[4096])?;
        let output = tensor(
            "output.weight",
            DeepSeekResidentDtype::Q8_0,
            &[4096, 129_280],
        )?;
        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}");
            let ratio = config.compress_ratios[layer];
            let compressor = if ratio == 0 {
                None
            } else {
                let width = if ratio == 4 { 1024 } else { 512 };
                Some(DeepSeekV4CompressorGraph {
                    ratio,
                    ape: tensor(
                        &name("attn_compressor_ape.weight"),
                        DeepSeekResidentDtype::F16,
                        &[width, ratio],
                    )?,
                    kv: tensor(
                        &name("attn_compressor_kv.weight"),
                        DeepSeekResidentDtype::F16,
                        &[4096, width],
                    )?,
                    gate: tensor(
                        &name("attn_compressor_gate.weight"),
                        DeepSeekResidentDtype::F16,
                        &[4096, width],
                    )?,
                    norm: tensor(
                        &name("attn_compressor_norm.weight"),
                        DeepSeekResidentDtype::F32,
                        &[512],
                    )?,
                })
            };
            let indexer = if ratio == 4 {
                let q_b_entry = registry.require(&name("indexer.attn_q_b.weight"))?;
                let q_b_dtype = match q_b_entry.dtype.as_str() {
                    "F16" => DeepSeekResidentDtype::F16,
                    "Q8_0" => DeepSeekResidentDtype::Q8_0,
                    other => bail!(
                        "DeepSeek indexer q_b layer {layer} has unsupported resident dtype {other}"
                    ),
                };
                Some(DeepSeekV4IndexerGraph {
                    q_b: bind_deepseek_entry(store_len, q_b_entry, q_b_dtype, &[1024, 8192])?,
                    projection: tensor(
                        &name("indexer.proj.weight"),
                        DeepSeekResidentDtype::F16,
                        &[4096, 64],
                    )?,
                    compressor_ape: tensor(
                        &name("indexer_compressor_ape.weight"),
                        DeepSeekResidentDtype::F16,
                        &[256, 4],
                    )?,
                    compressor_kv: tensor(
                        &name("indexer_compressor_kv.weight"),
                        DeepSeekResidentDtype::F16,
                        &[4096, 256],
                    )?,
                    compressor_gate: tensor(
                        &name("indexer_compressor_gate.weight"),
                        DeepSeekResidentDtype::F16,
                        &[4096, 256],
                    )?,
                    compressor_norm: tensor(
                        &name("indexer_compressor_norm.weight"),
                        DeepSeekResidentDtype::F32,
                        &[128],
                    )?,
                })
            } else {
                None
            };
            let token_hash_routes = if layer < config.hash_layer_count {
                Some(tensor(
                    &name("ffn_gate_tid2eid.weight"),
                    DeepSeekResidentDtype::I32,
                    &[6, 129_280],
                )?)
            } else {
                None
            };
            let router_bias = registry
                .tensor(&name("exp_probs_b.bias"))
                .map(|entry| {
                    bind_deepseek_entry(store_len, entry, DeepSeekResidentDtype::F32, &[256])
                })
                .transpose()?;
            layers.push(DeepSeekV4LayerGraph {
                hc_attn_fn: tensor(
                    &name("hc_attn_fn.weight"),
                    DeepSeekResidentDtype::F16,
                    &[16_384, 24],
                )?,
                hc_attn_scale: tensor(
                    &name("hc_attn_scale.weight"),
                    DeepSeekResidentDtype::F32,
                    &[3],
                )?,
                hc_attn_base: tensor(
                    &name("hc_attn_base.weight"),
                    DeepSeekResidentDtype::F32,
                    &[24],
                )?,
                attn_norm: tensor(
                    &name("attn_norm.weight"),
                    DeepSeekResidentDtype::F32,
                    &[4096],
                )?,
                attn_q_a: tensor(
                    &name("attn_q_a.weight"),
                    DeepSeekResidentDtype::Q8_0,
                    &[4096, 1024],
                )?,
                attn_q_a_norm: tensor(
                    &name("attn_q_a_norm.weight"),
                    DeepSeekResidentDtype::F32,
                    &[1024],
                )?,
                attn_q_b: tensor(
                    &name("attn_q_b.weight"),
                    DeepSeekResidentDtype::Q8_0,
                    &[1024, 32_768],
                )?,
                attn_kv: tensor(
                    &name("attn_kv.weight"),
                    DeepSeekResidentDtype::Q8_0,
                    &[4096, 512],
                )?,
                attn_kv_a_norm: tensor(
                    &name("attn_kv_a_norm.weight"),
                    DeepSeekResidentDtype::F32,
                    &[512],
                )?,
                attn_sinks: tensor(
                    &name("attn_sinks.weight"),
                    DeepSeekResidentDtype::F32,
                    &[64],
                )?,
                attn_output_a: tensor(
                    &name("attn_output_a.weight"),
                    DeepSeekResidentDtype::Q8_0,
                    &[4096, 8192],
                )?,
                attn_output_b: tensor(
                    &name("attn_output_b.weight"),
                    DeepSeekResidentDtype::Q8_0,
                    &[8192, 4096],
                )?,
                hc_ffn_fn: tensor(
                    &name("hc_ffn_fn.weight"),
                    DeepSeekResidentDtype::F16,
                    &[16_384, 24],
                )?,
                hc_ffn_scale: tensor(
                    &name("hc_ffn_scale.weight"),
                    DeepSeekResidentDtype::F32,
                    &[3],
                )?,
                hc_ffn_base: tensor(
                    &name("hc_ffn_base.weight"),
                    DeepSeekResidentDtype::F32,
                    &[24],
                )?,
                ffn_norm: tensor(
                    &name("ffn_norm.weight"),
                    DeepSeekResidentDtype::F32,
                    &[4096],
                )?,
                router: tensor(
                    &name("ffn_gate_inp.weight"),
                    DeepSeekResidentDtype::F16,
                    &[4096, 256],
                )?,
                router_bias,
                shared_gate: tensor(
                    &name("ffn_gate_shexp.weight"),
                    DeepSeekResidentDtype::Q8_0,
                    &[4096, 2048],
                )?,
                shared_up: tensor(
                    &name("ffn_up_shexp.weight"),
                    DeepSeekResidentDtype::Q8_0,
                    &[4096, 2048],
                )?,
                shared_down: tensor(
                    &name("ffn_down_shexp.weight"),
                    DeepSeekResidentDtype::Q8_0,
                    &[2048, 4096],
                )?,
                compressor,
                indexer,
                token_hash_routes,
            });
        }
        let prefill_layer_prepare_min_tokens = config.expert_count;
        let prefill_resident_layer_ranges = layers
            .iter()
            .map(deepseek_layer_resident_ranges)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config,
            prefill_layer_prepare_min_tokens,
            prefill_resident_layer_ranges,
            embedding,
            output_hc_base,
            output_hc_fn,
            output_hc_scale,
            output_norm,
            output,
            layers,
        })
    }
}

fn deepseek_layer_resident_ranges(
    layer: &DeepSeekV4LayerGraph,
) -> Result<Box<[DeepSeekResidentRange]>> {
    let mut tensors = vec![
        &layer.hc_attn_fn,
        &layer.hc_attn_scale,
        &layer.hc_attn_base,
        &layer.attn_norm,
        &layer.attn_q_a,
        &layer.attn_q_a_norm,
        &layer.attn_q_b,
        &layer.attn_kv,
        &layer.attn_kv_a_norm,
        &layer.attn_sinks,
        &layer.attn_output_a,
        &layer.attn_output_b,
        &layer.hc_ffn_fn,
        &layer.hc_ffn_scale,
        &layer.hc_ffn_base,
        &layer.ffn_norm,
        &layer.router,
        &layer.shared_gate,
        &layer.shared_up,
        &layer.shared_down,
    ];
    tensors.extend(layer.router_bias.iter());
    tensors.extend(layer.token_hash_routes.iter());
    if let Some(compressor) = &layer.compressor {
        tensors.extend([
            &compressor.ape,
            &compressor.kv,
            &compressor.gate,
            &compressor.norm,
        ]);
    }
    if let Some(indexer) = &layer.indexer {
        tensors.extend([
            &indexer.q_b,
            &indexer.projection,
            &indexer.compressor_ape,
            &indexer.compressor_kv,
            &indexer.compressor_gate,
            &indexer.compressor_norm,
        ]);
    }
    coalesce_deepseek_resident_ranges(tensors.into_iter().map(|tensor| DeepSeekResidentRange {
        byte_offset: tensor.byte_offset,
        byte_len: tensor.byte_len,
    }))
}

fn coalesce_deepseek_resident_ranges(
    ranges: impl IntoIterator<Item = DeepSeekResidentRange>,
) -> Result<Box<[DeepSeekResidentRange]>> {
    let mut ranges = ranges.into_iter().collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|range| range.byte_offset);
    let mut coalesced = Vec::<DeepSeekResidentRange>::with_capacity(ranges.len());
    for range in ranges {
        if range.byte_len == 0 {
            bail!("DeepSeek resident preparation range cannot be empty");
        }
        let end = range
            .byte_offset
            .checked_add(range.byte_len)
            .context("DeepSeek resident preparation range overflow")?;
        if let Some(previous) = coalesced.last_mut() {
            let previous_end = previous
                .byte_offset
                .checked_add(previous.byte_len)
                .context("DeepSeek coalesced resident preparation range overflow")?;
            if range.byte_offset < previous_end {
                bail!(
                    "DeepSeek resident preparation ranges overlap at {}..{} and {}..{}",
                    previous.byte_offset,
                    previous_end,
                    range.byte_offset,
                    end
                );
            }
            if range.byte_offset == previous_end {
                previous.byte_len = end
                    .checked_sub(previous.byte_offset)
                    .context("DeepSeek resident preparation merge underflow")?;
                continue;
            }
        }
        coalesced.push(range);
    }
    if coalesced.is_empty() {
        bail!("DeepSeek layer has no resident ranges to prepare");
    }
    Ok(coalesced.into_boxed_slice())
}

fn bind_deepseek_tensor(
    registry: &TensorRegistry,
    store_len: u64,
    name: &str,
    dtype: DeepSeekResidentDtype,
    shape: &[usize],
) -> Result<DeepSeekResidentTensor> {
    bind_deepseek_entry(store_len, registry.require(name)?, dtype, shape)
}

fn bind_deepseek_entry(
    store_len: u64,
    entry: &RuntimeTensorEntry,
    dtype: DeepSeekResidentDtype,
    shape: &[usize],
) -> Result<DeepSeekResidentTensor> {
    let expected = dtype.gguf();
    if entry.dtype != expected.name || entry.shape != shape {
        bail!(
            "DeepSeek resident tensor {} has dtype/shape {}/{:?}, expected {}/{shape:?}",
            entry.name,
            entry.dtype,
            entry.shape,
            expected.name
        );
    }
    let elements = shape.iter().try_fold(1u64, |product, dimension| {
        product
            .checked_mul(*dimension as u64)
            .context("DeepSeek tensor element count overflow")
    })?;
    let expected_bytes = elements
        .div_ceil(expected.block_elements)
        .checked_mul(expected.block_bytes)
        .context("DeepSeek tensor byte count overflow")?;
    if entry.byte_len != expected_bytes
        || entry.byte_offset % TENSOR_ALIGNMENT != 0
        || entry
            .byte_offset
            .checked_add(entry.byte_len)
            .is_none_or(|end| end > store_len)
    {
        bail!(
            "DeepSeek resident tensor {} has invalid store range offset={} bytes={} expected_bytes={} store_bytes={store_len}",
            entry.name,
            entry.byte_offset,
            entry.byte_len,
            expected_bytes
        );
    }
    match (&entry.quantization, dtype) {
        (
            TensorQuantization::None,
            DeepSeekResidentDtype::F16 | DeepSeekResidentDtype::F32 | DeepSeekResidentDtype::I32,
        ) => {}
        (
            TensorQuantization::Gguf {
                tensor_type,
                block_elements,
                block_bytes,
            },
            DeepSeekResidentDtype::Q8_0,
        ) if *tensor_type == expected.id
            && *block_elements == expected.block_elements
            && *block_bytes == expected.block_bytes => {}
        _ => bail!(
            "DeepSeek resident tensor {} has incompatible quantization descriptor {:?}",
            entry.name,
            entry.quantization
        ),
    }
    Ok(DeepSeekResidentTensor {
        name: entry.name.clone(),
        dtype,
        shape: entry.shape.clone(),
        byte_offset: entry.byte_offset,
        byte_len: entry.byte_len,
    })
}

pub(crate) fn deepseek_v4_router_probabilities(logits: &[f32]) -> Result<Vec<f32>> {
    if logits.len() != 256 {
        bail!(
            "DeepSeek V4 router received {} logits, expected 256",
            logits.len()
        );
    }
    logits
        .iter()
        .copied()
        .enumerate()
        .map(|(expert, logit)| {
            if !logit.is_finite() {
                bail!("DeepSeek V4 router logit for expert {expert} is not finite");
            }
            let softplus = if logit > 20.0 {
                logit
            } else if logit < -20.0 {
                logit.exp()
            } else {
                logit.exp().ln_1p()
            };
            Ok(softplus.sqrt())
        })
        .collect()
}

/// Selects the exact six DeepSeek routes.  Returned scores are deliberately
/// unbiased and unnormalised; the shared scheduler performs the declared
/// selected-route renormalisation and 1.5 scale exactly once while issuing the
/// positioned expert reads.
pub(crate) fn deepseek_v4_select_routes(
    probabilities: &[f32],
    correction_bias: Option<&[f32]>,
    hash_selected: Option<&[i32]>,
) -> Result<Vec<(usize, f32)>> {
    if probabilities.len() != 256 {
        bail!(
            "DeepSeek V4 route selection received {} probabilities, expected 256",
            probabilities.len()
        );
    }
    if probabilities
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        bail!("DeepSeek V4 route probabilities must be finite and non-negative");
    }
    let selected = if let Some(selected) = hash_selected {
        if selected.len() != 6 {
            bail!(
                "DeepSeek V4 hash routing received {} expert ids, expected 6",
                selected.len()
            );
        }
        let mut seen = std::collections::BTreeSet::new();
        selected
            .iter()
            .copied()
            .map(|expert| {
                let expert = usize::try_from(expert)
                    .context("DeepSeek V4 hash routing contains a negative expert id")?;
                if expert >= 256 || !seen.insert(expert) {
                    bail!("DeepSeek V4 hash routing expert {expert} is out of range or duplicated");
                }
                Ok(expert)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let bias = correction_bias.unwrap_or(&[]);
        if !bias.is_empty() && bias.len() != 256 {
            bail!(
                "DeepSeek V4 correction bias has {} values, expected 256",
                bias.len()
            );
        }
        if bias.iter().any(|value| !value.is_finite()) {
            bail!("DeepSeek V4 correction bias must be finite");
        }
        let mut candidates = (0..256usize).collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            let left_score = probabilities[*left] + bias.get(*left).copied().unwrap_or(0.0);
            let right_score = probabilities[*right] + bias.get(*right).copied().unwrap_or(0.0);
            right_score
                .total_cmp(&left_score)
                .then_with(|| left.cmp(right))
        });
        candidates.truncate(6);
        candidates
    };
    Ok(selected
        .into_iter()
        .map(|expert| (expert, probabilities[expert]))
        .collect())
}

struct ValidatedDeepSeekV4Source<'a> {
    config: DeepSeekV4Config,
    resident: Vec<&'a GgufTensorInfo>,
    experts: Vec<DeepSeekExpertLayerSource<'a>>,
    tokenizer: DeepSeekV4TokenizerCache,
}

struct DeepSeekExpertLayerSource<'a> {
    layer: usize,
    gate: &'a GgufTensorInfo,
    up: &'a GgufTensorInfo,
    down: &'a GgufTensorInfo,
}

pub fn build_deepseek_v4_flash_cache_from_gguf(
    model: &str,
    gguf_path: &Path,
    models_root: &Path,
) -> Result<FlashMoePlan> {
    if !is_deepseek_v4_flash(model) {
        bail!("{model} is not the pinned DeepSeek V4 Flash GGUF profile");
    }
    if gguf_path.file_name().and_then(|name| name.to_str()) != Some(DEEPSEEK_V4_FLASH_FILENAME) {
        bail!(
            "DeepSeek V4 Flash requires source file {DEEPSEEK_V4_FLASH_FILENAME}, got {}",
            gguf_path.display()
        );
    }

    // Source validation is intentionally complete before any cache file is
    // created or replaced.
    let gguf = GgufFile::open(gguf_path)?;
    let source = validate_source(&gguf)?;
    let plan = plan_unchecked_with_cache_version(
        DEEPSEEK_V4_FLASH_MODEL,
        models_root,
        FlashMoeRoutingPolicy::default(),
        DEEPSEEK_V4_FLASH_CACHE_VERSION,
    );
    fs::create_dir_all(&plan.runtime_dir)
        .with_context(|| format!("failed to create {}", plan.runtime_dir.display()))?;
    fs::create_dir_all(&plan.experts_dir)
        .with_context(|| format!("failed to create {}", plan.experts_dir.display()))?;
    fs::create_dir_all(&plan.model_cache_dir)
        .with_context(|| format!("failed to create {}", plan.model_cache_dir.display()))?;

    let manifest = publish_resident_store(&gguf, &source, &plan)?;
    publish_expert_store(&gguf, &source, &plan)?;
    atomic_write_json(&plan.model_config, &source.config)?;
    atomic_write_json(&plan.tensor_manifest, &manifest)?;
    atomic_write_json(&plan.tokenizer, &source.tokenizer)?;
    atomic_write_json(
        &plan.tokenizer_config,
        &serde_json::json!({
            "format": DEEPSEEK_TOKENIZER_FORMAT,
            "chat_template": "deepseek_v4_flash",
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>"
        }),
    )?;
    atomic_write(
        &plan.chat_template,
        "<｜begin▁of▁sentence｜><｜User｜>{{ prompt }}<｜Assistant｜>".as_bytes(),
    )?;
    atomic_write(
        &plan.runtime_dir.join("kernels.metal"),
        METAL_SHADERS.as_bytes(),
    )?;
    atomic_write(
        &plan.runtime_dir.join("README.txt"),
        format!(
            "FlashMoe DeepSeek V4 Flash cache: fixed load-time graph, {} layers, {} experts/layer, K={}, resident GGUF blocks in {}, scheduler-owned page-aligned expert slots in {}\n",
            source.config.block_count,
            source.config.expert_count,
            source.config.expert_used_count,
            plan.non_expert_weights.display(),
            plan.experts_dir.display(),
        )
        .as_bytes(),
    )?;
    Ok(plan)
}

fn validate_source(gguf: &GgufFile) -> Result<ValidatedDeepSeekV4Source<'_>> {
    let config = DeepSeekV4Config::from_gguf(gguf)?;
    let mut resident = Vec::new();
    expect_resident(
        gguf,
        &mut resident,
        "token_embd.weight",
        GgufTensorType::F16,
        &[4096, 129_280],
    )?;
    expect_resident(
        gguf,
        &mut resident,
        "output_hc_base.weight",
        GgufTensorType::F32,
        &[4],
    )?;
    expect_resident(
        gguf,
        &mut resident,
        "output_hc_fn.weight",
        GgufTensorType::F16,
        &[16_384, 4],
    )?;
    expect_resident(
        gguf,
        &mut resident,
        "output_hc_scale.weight",
        GgufTensorType::F32,
        &[1],
    )?;
    expect_resident(
        gguf,
        &mut resident,
        "output_norm.weight",
        GgufTensorType::F32,
        &[4096],
    )?;
    expect_resident(
        gguf,
        &mut resident,
        "output.weight",
        GgufTensorType::Q8_0,
        &[4096, 129_280],
    )?;

    let mut experts = Vec::with_capacity(config.block_count);
    for layer in 0..config.block_count {
        let ratio = config.compress_ratios[layer];
        for (suffix, tensor_type, dimensions) in [
            ("hc_attn_fn.weight", GgufTensorType::F16, vec![16_384, 24]),
            ("hc_attn_scale.weight", GgufTensorType::F32, vec![3]),
            ("hc_attn_base.weight", GgufTensorType::F32, vec![24]),
            ("attn_norm.weight", GgufTensorType::F32, vec![4096]),
            ("attn_q_a.weight", GgufTensorType::Q8_0, vec![4096, 1024]),
            ("attn_q_a_norm.weight", GgufTensorType::F32, vec![1024]),
            ("attn_q_b.weight", GgufTensorType::Q8_0, vec![1024, 32_768]),
            ("attn_kv.weight", GgufTensorType::Q8_0, vec![4096, 512]),
            ("attn_kv_a_norm.weight", GgufTensorType::F32, vec![512]),
            ("attn_sinks.weight", GgufTensorType::F32, vec![64]),
            (
                "attn_output_a.weight",
                GgufTensorType::Q8_0,
                vec![4096, 8192],
            ),
            (
                "attn_output_b.weight",
                GgufTensorType::Q8_0,
                vec![8192, 4096],
            ),
            ("hc_ffn_fn.weight", GgufTensorType::F16, vec![16_384, 24]),
            ("hc_ffn_scale.weight", GgufTensorType::F32, vec![3]),
            ("hc_ffn_base.weight", GgufTensorType::F32, vec![24]),
            ("ffn_norm.weight", GgufTensorType::F32, vec![4096]),
            ("ffn_gate_inp.weight", GgufTensorType::F16, vec![4096, 256]),
            (
                "ffn_gate_shexp.weight",
                GgufTensorType::Q8_0,
                vec![4096, 2048],
            ),
            (
                "ffn_up_shexp.weight",
                GgufTensorType::Q8_0,
                vec![4096, 2048],
            ),
            (
                "ffn_down_shexp.weight",
                GgufTensorType::Q8_0,
                vec![2048, 4096],
            ),
        ] {
            let name = format!("blk.{layer}.{suffix}");
            expect_resident(gguf, &mut resident, &name, tensor_type, &dimensions)?;
        }
        if let Some(bias) = gguf.tensors.get(&format!("blk.{layer}.exp_probs_b.bias")) {
            expect_tensor_info(bias, GgufTensorType::F32, &[256])?;
            resident.push(bias);
        }
        if ratio != 0 {
            let comp_width = if ratio == 4 { 1024 } else { 512 };
            for (suffix, tensor_type, dimensions) in [
                (
                    "attn_compressor_ape.weight",
                    GgufTensorType::F16,
                    vec![comp_width, ratio as u64],
                ),
                (
                    "attn_compressor_kv.weight",
                    GgufTensorType::F16,
                    vec![4096, comp_width],
                ),
                (
                    "attn_compressor_gate.weight",
                    GgufTensorType::F16,
                    vec![4096, comp_width],
                ),
                (
                    "attn_compressor_norm.weight",
                    GgufTensorType::F32,
                    vec![512],
                ),
            ] {
                let name = format!("blk.{layer}.{suffix}");
                expect_resident(gguf, &mut resident, &name, tensor_type, &dimensions)?;
            }
        }
        if ratio == 4 {
            let indexer_q =
                gguf.required_tensor(&format!("blk.{layer}.indexer.attn_q_b.weight"))?;
            if indexer_q.tensor_type != GgufTensorType::F16
                && indexer_q.tensor_type != GgufTensorType::Q8_0
            {
                bail!(
                    "GGUF tensor {} has type {}, expected F16 or Q8_0",
                    indexer_q.name,
                    indexer_q.tensor_type.name
                );
            }
            expect_dimensions(indexer_q, &[1024, 8192])?;
            resident.push(indexer_q);
            for (suffix, tensor_type, dimensions) in [
                ("indexer.proj.weight", GgufTensorType::F16, vec![4096, 64]),
                (
                    "indexer_compressor_ape.weight",
                    GgufTensorType::F16,
                    vec![256, 4],
                ),
                (
                    "indexer_compressor_kv.weight",
                    GgufTensorType::F16,
                    vec![4096, 256],
                ),
                (
                    "indexer_compressor_gate.weight",
                    GgufTensorType::F16,
                    vec![4096, 256],
                ),
                (
                    "indexer_compressor_norm.weight",
                    GgufTensorType::F32,
                    vec![128],
                ),
            ] {
                let name = format!("blk.{layer}.{suffix}");
                expect_resident(gguf, &mut resident, &name, tensor_type, &dimensions)?;
            }
        }
        if layer < config.hash_layer_count {
            expect_resident(
                gguf,
                &mut resident,
                &format!("blk.{layer}.ffn_gate_tid2eid.weight"),
                GgufTensorType::I32,
                &[6, 129_280],
            )?;
        }

        let gate = gguf.required_tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))?;
        let up = gguf.required_tensor(&format!("blk.{layer}.ffn_up_exps.weight"))?;
        let down = gguf.required_tensor(&format!("blk.{layer}.ffn_down_exps.weight"))?;
        expect_tensor_info(gate, GgufTensorType::IQ2_XXS, &[4096, 2048, 256])?;
        expect_tensor_info(up, GgufTensorType::IQ2_XXS, &[4096, 2048, 256])?;
        expect_tensor_info(down, GgufTensorType::Q2_K, &[2048, 4096, 256])?;
        experts.push(DeepSeekExpertLayerSource {
            layer,
            gate,
            up,
            down,
        });
    }

    let tokenizer = validate_tokenizer(gguf, config.vocab_size)?;
    Ok(ValidatedDeepSeekV4Source {
        config,
        resident,
        experts,
        tokenizer,
    })
}

fn validate_tokenizer(gguf: &GgufFile, vocab_size: usize) -> Result<DeepSeekV4TokenizerCache> {
    let model = required_string(gguf, "tokenizer.ggml.model")?;
    if model != "gpt2" {
        bail!("DeepSeek V4 Flash tokenizer.ggml.model must be \"gpt2\", got {model:?}");
    }
    let pre_tokenizer = required_string(gguf, "tokenizer.ggml.pre")?;
    if pre_tokenizer != "joyai-llm" {
        bail!("DeepSeek V4 Flash tokenizer.ggml.pre must be \"joyai-llm\", got {pre_tokenizer:?}");
    }
    let tokens = required_string_array(gguf, "tokenizer.ggml.tokens")?;
    if tokens.len() != vocab_size {
        bail!(
            "DeepSeek V4 Flash tokenizer contains {} tokens, expected {vocab_size}",
            tokens.len()
        );
    }
    let merges = required_string_array(gguf, "tokenizer.ggml.merges")?;
    let mut special_tokens = BTreeMap::new();
    for token in [
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
        "｜DSML｜",
    ] {
        let id = tokens
            .iter()
            .position(|candidate| candidate == token)
            .with_context(|| {
                format!("DeepSeek V4 Flash tokenizer is missing special token {token}")
            })?;
        special_tokens.insert(
            token.to_string(),
            u32::try_from(id).context("DeepSeek tokenizer token id does not fit u32")?,
        );
    }
    Ok(DeepSeekV4TokenizerCache {
        format: DEEPSEEK_TOKENIZER_FORMAT.to_string(),
        model,
        pre_tokenizer,
        tokens,
        merges,
        special_tokens,
    })
}

fn publish_resident_store(
    gguf: &GgufFile,
    source: &ValidatedDeepSeekV4Source<'_>,
    plan: &FlashMoePlan,
) -> Result<FlashMoeManifest> {
    let temp_path = temp_pack_path(&plan.non_expert_weights);
    let mut input = File::open(&gguf.path)
        .with_context(|| format!("failed to reopen GGUF source {}", gguf.path.display()))?;
    let mut output = File::create(&temp_path)
        .with_context(|| format!("failed to create {}", temp_path.display()))?;
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    let mut runtime_offset = 0u64;
    let shard = gguf
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(DEEPSEEK_V4_FLASH_FILENAME)
        .to_string();
    let mut dense_tensors = Vec::with_capacity(source.resident.len());
    for tensor in &source.resident {
        runtime_offset = align_up(runtime_offset, TENSOR_ALIGNMENT)?;
        copy_range(
            &mut input,
            &mut output,
            tensor.absolute_offset,
            runtime_offset,
            tensor.byte_len,
            &mut buffer,
        )?;
        dense_tensors.push(DenseTensorRef {
            tensor: tensor.name.clone(),
            shard: shard.clone(),
            dtype: tensor.tensor_type.name.to_string(),
            shape: dimensions_to_usize(&tensor.dimensions)?,
            source_offsets: [
                tensor.absolute_offset,
                tensor.absolute_offset + tensor.byte_len,
            ],
            runtime_offset,
            byte_len: tensor.byte_len,
            quantization: gguf_quantization(tensor),
            q4_sources: None,
        });
        runtime_offset = runtime_offset
            .checked_add(tensor.byte_len)
            .context("DeepSeek resident store offset overflow")?;
    }
    output
        .set_len(runtime_offset)
        .with_context(|| format!("failed to size {}", temp_path.display()))?;
    finish_expert_pack_atomically(output, &temp_path, &plan.non_expert_weights)?;

    let expert_tensors = source
        .experts
        .iter()
        .flat_map(|layer| [layer.gate, layer.up, layer.down])
        .map(|tensor| ExpertTensorRef {
            tensor: tensor.name.clone(),
            shard: shard.clone(),
            layer: tensor_layer(&tensor.name),
            expert: None,
            dtype: Some(tensor.tensor_type.name.to_string()),
            shape: dimensions_to_usize(&tensor.dimensions).unwrap_or_default(),
            source_offsets: Some([
                tensor.absolute_offset,
                tensor.absolute_offset + tensor.byte_len,
            ]),
            q4_sources: None,
        })
        .collect();
    Ok(FlashMoeManifest {
        model: DEEPSEEK_V4_FLASH_MODEL.to_string(),
        cache_version: DEEPSEEK_V4_FLASH_CACHE_VERSION.to_string(),
        dense_shards: vec![shard],
        expert_tensors,
        dense_tensors,
    })
}

fn publish_expert_store(
    gguf: &GgufFile,
    source: &ValidatedDeepSeekV4Source<'_>,
    plan: &FlashMoePlan,
) -> Result<()> {
    let mut input = File::open(&gguf.path)
        .with_context(|| format!("failed to reopen GGUF source {}", gguf.path.display()))?;
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    for layer in &source.experts {
        let gate_bytes = per_expert_bytes(layer.gate, source.config.expert_count)?;
        let up_bytes = per_expert_bytes(layer.up, source.config.expert_count)?;
        let down_bytes = per_expert_bytes(layer.down, source.config.expert_count)?;
        let gate_offset = 0u64;
        let up_offset = align_up(gate_bytes, EXPERT_COMPONENT_ALIGNMENT)?;
        let down_offset = align_up(
            up_offset
                .checked_add(up_bytes)
                .context("DeepSeek expert up component end overflow")?,
            EXPERT_COMPONENT_ALIGNMENT,
        )?;
        let slot_bytes = align_up(
            down_offset
                .checked_add(down_bytes)
                .context("DeepSeek expert down component end overflow")?,
            EXPERT_COMPONENT_ALIGNMENT,
        )?;

        let final_path = expert_layer_path(&plan.experts_dir, layer.layer);
        let temp_path = temp_pack_path(&final_path);
        let mut output = File::create(&temp_path)
            .with_context(|| format!("failed to create {}", temp_path.display()))?;
        output.set_len(
            slot_bytes
                .checked_mul(source.config.expert_count as u64)
                .context("DeepSeek expert layer byte length overflow")?,
        )?;
        let mut packs = Vec::with_capacity(source.config.expert_count);
        for expert in 0..source.config.expert_count {
            let slot_offset = slot_bytes
                .checked_mul(expert as u64)
                .context("DeepSeek expert slot offset overflow")?;
            let components = [
                (layer.gate, gate_bytes, gate_offset),
                (layer.up, up_bytes, up_offset),
                (layer.down, down_bytes, down_offset),
            ];
            let mut records = Vec::with_capacity(3);
            for (tensor, component_bytes, component_offset) in components {
                let source_offset = tensor
                    .absolute_offset
                    .checked_add(
                        component_bytes
                            .checked_mul(expert as u64)
                            .context("DeepSeek expert source component offset overflow")?,
                    )
                    .context("DeepSeek expert source offset overflow")?;
                let runtime_component_offset = slot_offset
                    .checked_add(component_offset)
                    .context("DeepSeek expert destination offset overflow")?;
                copy_range(
                    &mut input,
                    &mut output,
                    source_offset,
                    runtime_component_offset,
                    component_bytes,
                    &mut buffer,
                )?;
                records.push(ExpertPackRecord {
                    tensor: tensor.name.clone(),
                    dtype: tensor.tensor_type.name.to_string(),
                    shape: dimensions_to_usize(&tensor.dimensions[..2])?,
                    source_offsets: [source_offset, source_offset + component_bytes],
                    source_hash: None,
                    record_offset: component_offset,
                    packed_bytes: component_bytes,
                    groups: 0,
                    group_size: tensor.tensor_type.block_elements as usize,
                    scale_bias_dtype: DEEPSEEK_EXPERT_SCALE_BIAS_DTYPE.to_string(),
                });
            }
            packs.push(ExpertPackMetadata {
                layer: layer.layer,
                expert,
                packed_bytes: slot_bytes,
                records,
            });
        }
        finish_expert_pack_atomically(output, &temp_path, &final_path)?;
        let metadata = ExpertLayerPackMetadata::new_fixed_deepseek_gguf(
            layer.layer,
            slot_bytes,
            source.config.expert_count,
            packs,
        );
        write_expert_metadata_atomically(&plan.experts_dir, layer.layer, &metadata)?;
    }
    Ok(())
}

fn expect_resident<'a>(
    gguf: &'a GgufFile,
    resident: &mut Vec<&'a GgufTensorInfo>,
    name: &str,
    tensor_type: GgufTensorType,
    dimensions: &[u64],
) -> Result<()> {
    let tensor = gguf.required_tensor(name)?;
    expect_tensor_info(tensor, tensor_type, dimensions)?;
    resident.push(tensor);
    Ok(())
}

fn expect_tensor_info(
    tensor: &GgufTensorInfo,
    tensor_type: GgufTensorType,
    dimensions: &[u64],
) -> Result<()> {
    if tensor.tensor_type != tensor_type {
        bail!(
            "GGUF tensor {} has type {}, expected {}",
            tensor.name,
            tensor.tensor_type.name,
            tensor_type.name
        );
    }
    expect_dimensions(tensor, dimensions)
}

fn expect_dimensions(tensor: &GgufTensorInfo, dimensions: &[u64]) -> Result<()> {
    if tensor.dimensions != dimensions {
        bail!(
            "GGUF tensor {} has dimensions {:?}, expected {:?}",
            tensor.name,
            tensor.dimensions,
            dimensions
        );
    }
    Ok(())
}

fn required_usize(gguf: &GgufFile, key: &str) -> Result<usize> {
    let value = gguf
        .required_metadata(key)?
        .as_u64_compat()
        .with_context(|| format!("GGUF metadata {key} must be uint32 or uint64"))?;
    usize::try_from(value).with_context(|| format!("GGUF metadata {key} does not fit usize"))
}

fn optional_usize(gguf: &GgufFile, key: &str) -> Result<Option<usize>> {
    gguf.metadata
        .get(key)
        .map(|value| {
            value
                .as_u64_compat()
                .with_context(|| format!("GGUF metadata {key} must be uint32 or uint64"))
                .and_then(|value| {
                    usize::try_from(value)
                        .with_context(|| format!("GGUF metadata {key} does not fit usize"))
                })
        })
        .transpose()
}

fn required_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    optional_f32(gguf, key)?
        .with_context(|| format!("required GGUF metadata key is missing: {key}"))
}

fn optional_f32(gguf: &GgufFile, key: &str) -> Result<Option<f32>> {
    gguf.metadata
        .get(key)
        .map(|value| {
            value
                .as_f64_compat()
                .map(|value| value as f32)
                .with_context(|| format!("GGUF metadata {key} must be numeric"))
        })
        .transpose()
}

fn required_string(gguf: &GgufFile, key: &str) -> Result<String> {
    gguf.required_metadata(key)?
        .as_str()
        .map(str::to_string)
        .with_context(|| format!("GGUF metadata {key} must be string"))
}

fn required_usize_array(gguf: &GgufFile, key: &str) -> Result<Vec<usize>> {
    let (element_type, values) = gguf
        .required_metadata(key)?
        .as_array()
        .with_context(|| format!("GGUF metadata {key} must be an array"))?;
    if element_type != GgufMetadataType::Uint32 && element_type != GgufMetadataType::Int32 {
        bail!("GGUF metadata {key} must be an int32 or uint32 array");
    }
    values
        .iter()
        .map(|value| match value {
            GgufValue::Uint32(value) => Ok(*value as usize),
            GgufValue::Int32(value) if *value >= 0 => Ok(*value as usize),
            _ => bail!("GGUF metadata {key} contains an invalid integer"),
        })
        .collect()
}

fn required_f32_array(gguf: &GgufFile, key: &str) -> Result<Vec<f32>> {
    let (element_type, values) = gguf
        .required_metadata(key)?
        .as_array()
        .with_context(|| format!("GGUF metadata {key} must be an array"))?;
    if element_type != GgufMetadataType::Float32 && element_type != GgufMetadataType::Float64 {
        bail!("GGUF metadata {key} must be a float32 or float64 array");
    }
    values
        .iter()
        .map(|value| match value {
            GgufValue::Float32(value) => Ok(*value),
            GgufValue::Float64(value) => Ok(*value as f32),
            _ => bail!("GGUF metadata {key} contains a non-float value"),
        })
        .collect()
}

fn required_string_array(gguf: &GgufFile, key: &str) -> Result<Vec<String>> {
    let (element_type, values) = gguf
        .required_metadata(key)?
        .as_array()
        .with_context(|| format!("GGUF metadata {key} must be an array"))?;
    if element_type != GgufMetadataType::String {
        bail!("GGUF metadata {key} must be a string array");
    }
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .with_context(|| format!("GGUF metadata {key} contains a non-string value"))
        })
        .collect()
}

fn float_matches(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-6
}

fn gguf_quantization(tensor: &GgufTensorInfo) -> TensorQuantization {
    if tensor.tensor_type == GgufTensorType::F32
        || tensor.tensor_type == GgufTensorType::F16
        || tensor.tensor_type == GgufTensorType::I32
    {
        TensorQuantization::None
    } else {
        TensorQuantization::Gguf {
            tensor_type: tensor.tensor_type.id,
            block_elements: tensor.tensor_type.block_elements,
            block_bytes: tensor.tensor_type.block_bytes,
        }
    }
}

fn dimensions_to_usize(dimensions: &[u64]) -> Result<Vec<usize>> {
    dimensions
        .iter()
        .map(|dimension| usize::try_from(*dimension).context("GGUF dimension does not fit usize"))
        .collect()
}

fn per_expert_bytes(tensor: &GgufTensorInfo, experts: usize) -> Result<u64> {
    if tensor.byte_len % experts as u64 != 0 {
        bail!(
            "GGUF routed tensor {} byte length {} is not divisible by {experts} experts",
            tensor.name,
            tensor.byte_len
        );
    }
    Ok(tensor.byte_len / experts as u64)
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - remainder)
        .context("DeepSeek cache alignment overflow")
}

fn copy_range(
    input: &mut File,
    output: &mut File,
    input_offset: u64,
    output_offset: u64,
    len: u64,
    buffer: &mut [u8],
) -> Result<()> {
    input.seek(SeekFrom::Start(input_offset))?;
    output.seek(SeekFrom::Start(output_offset))?;
    let mut remaining = len;
    while remaining != 0 {
        let wanted = usize::try_from(remaining.min(buffer.len() as u64))
            .context("DeepSeek copy chunk does not fit usize")?;
        input
            .read_exact(&mut buffer[..wanted])
            .with_context(|| format!("failed reading GGUF payload at offset {input_offset}"))?;
        output
            .write_all(&buffer[..wanted])
            .with_context(|| format!("failed writing FlashMoe cache at offset {output_offset}"))?;
        remaining -= wanted as u64;
    }
    Ok(())
}

fn atomic_write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("failed to encode DeepSeek cache JSON")?;
    atomic_write(path, &bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp_path = temp_pack_path(path);
    let mut output = File::create(&temp_path)
        .with_context(|| format!("failed to create {}", temp_path.display()))?;
    output
        .write_all(bytes)
        .with_context(|| format!("failed to write {}", temp_path.display()))?;
    finish_expert_pack_atomically(output, &temp_path, path)
}

fn tensor_layer(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    rest.split('.').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joyai_constraint_bytes_preserve_byte_bpe_and_mark_special_tokens() {
        let tokenizer = DeepSeekV4Tokenizer {
            tokens: vec![
                "a".to_string(),
                "Ġ".to_string(),
                "<｜end｜>".to_string(),
                String::new(),
            ],
            token_to_id: BTreeMap::from([
                ("a".to_string(), 0),
                ("Ġ".to_string(), 1),
                ("<｜end｜>".to_string(), 2),
            ]),
            merge_rank: BTreeMap::new(),
            special_tokens: BTreeMap::from([
                ("<｜end｜>".to_string(), 2),
                ("reserved".to_string(), 3),
            ]),
            eos_id: 2,
        };

        let bytes = tokenizer.constraint_token_bytes().unwrap();

        assert_eq!(bytes[0], b"a");
        assert_eq!(bytes[1], b" ");
        assert_eq!(
            bytes[2],
            [
                &[llguidance::toktrie::TokTrie::SPECIAL_TOKEN_MARKER],
                "<｜end｜>".as_bytes(),
            ]
            .concat()
        );
        assert_eq!(
            bytes[3],
            vec![llguidance::toktrie::TokTrie::SPECIAL_TOKEN_MARKER]
        );
    }

    fn value_u32(value: usize) -> GgufValue {
        GgufValue::Uint32(value as u32)
    }

    fn valid_metadata() -> BTreeMap<String, GgufValue> {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek4".to_string()),
        );
        for (key, value) in [
            ("deepseek4.block_count", 43),
            ("deepseek4.embedding_length", 4096),
            ("deepseek4.vocab_size", 129_280),
            ("deepseek4.attention.head_count", 64),
            ("deepseek4.attention.head_count_kv", 1),
            ("deepseek4.attention.key_length", 512),
            ("deepseek4.attention.value_length", 512),
            ("deepseek4.rope.dimension_count", 64),
            ("deepseek4.attention.q_lora_rank", 1024),
            ("deepseek4.attention.output_lora_rank", 1024),
            ("deepseek4.attention.output_group_count", 8),
            ("deepseek4.expert_count", 256),
            ("deepseek4.expert_used_count", 6),
            ("deepseek4.expert_feed_forward_length", 2048),
            ("deepseek4.expert_shared_count", 1),
            ("deepseek4.hash_layer_count", 3),
            ("deepseek4.attention.sliding_window", 128),
            ("deepseek4.attention.indexer.head_count", 64),
            ("deepseek4.attention.indexer.key_length", 128),
            ("deepseek4.attention.indexer.top_k", 512),
            ("deepseek4.hyper_connection.count", 4),
            ("deepseek4.hyper_connection.sinkhorn_iterations", 20),
        ] {
            metadata.insert(key.to_string(), value_u32(value));
        }
        let ratios = (0..43)
            .map(|layer| {
                GgufValue::Uint32(if layer < 2 {
                    0
                } else if layer % 2 == 0 {
                    4
                } else {
                    128
                })
            })
            .collect();
        metadata.insert(
            "deepseek4.attention.compress_ratios".to_string(),
            GgufValue::Array {
                element_type: GgufMetadataType::Uint32,
                values: ratios,
            },
        );
        metadata.insert(
            "deepseek4.swiglu_clamp_exp".to_string(),
            GgufValue::Array {
                element_type: GgufMetadataType::Float32,
                values: vec![GgufValue::Float32(10.0); 43],
            },
        );
        for (key, value) in [
            ("deepseek4.rope.freq_base", 10_000.0),
            ("deepseek4.rope.scaling.factor", 16.0),
            ("deepseek4.rope.scaling.yarn_beta_fast", 32.0),
            ("deepseek4.rope.scaling.yarn_beta_slow", 1.0),
            ("deepseek4.attention.compress_rope_freq_base", 160_000.0),
            ("deepseek4.expert_weights_scale", 1.5),
            ("deepseek4.attention.layer_norm_rms_epsilon", 1.0e-6),
            ("deepseek4.hyper_connection.epsilon", 1.0e-6),
        ] {
            metadata.insert(key.to_string(), GgufValue::Float32(value));
        }
        metadata.insert(
            "deepseek4.rope.scaling.original_context_length".to_string(),
            GgufValue::Uint64(65_536),
        );
        metadata.insert(
            "deepseek4.expert_weights_norm".to_string(),
            GgufValue::Bool(true),
        );
        metadata
    }

    fn gguf_with_metadata(metadata: BTreeMap<String, GgufValue>) -> GgufFile {
        GgufFile {
            path: "fixture.gguf".into(),
            version: 3,
            alignment: 32,
            tensor_data_offset: 0,
            file_len: 0,
            metadata,
            tensors: BTreeMap::new(),
        }
    }

    #[test]
    fn validates_the_exact_deepseek_v4_flash_profile() {
        let gguf = gguf_with_metadata(valid_metadata());
        let config = DeepSeekV4Config::from_gguf(&gguf).unwrap();
        assert_eq!(config.block_count, 43);
        assert_eq!(config.expert_used_count, 6);
        assert_eq!(config.compress_ratios[2], 4);
        assert_eq!(config.compress_ratios[3], 128);
    }

    #[test]
    fn normalizes_trailing_mtp_metadata_after_validating_the_base_graph() {
        let mut metadata = valid_metadata();
        let GgufValue::Array { values, .. } = metadata
            .get_mut("deepseek4.attention.compress_ratios")
            .unwrap()
        else {
            unreachable!()
        };
        values.push(GgufValue::Uint32(0));
        let GgufValue::Array { values, .. } =
            metadata.get_mut("deepseek4.swiglu_clamp_exp").unwrap()
        else {
            unreachable!()
        };
        values.push(GgufValue::Float32(0.0));

        let config = DeepSeekV4Config::from_gguf(&gguf_with_metadata(metadata)).unwrap();

        assert_eq!(config.compress_ratios.len(), 43);
        assert_eq!(config.swiglu_clamp_exp.len(), 43);
        config.validate_cached_flash_profile().unwrap();
    }

    #[test]
    fn rejects_shape_and_schedule_drift_before_graph_binding() {
        let mut shape = valid_metadata();
        shape.insert("deepseek4.expert_used_count".to_string(), value_u32(2));
        assert!(
            DeepSeekV4Config::from_gguf(&gguf_with_metadata(shape))
                .unwrap_err()
                .to_string()
                .contains("expert_used_count")
        );

        let mut schedule = valid_metadata();
        let GgufValue::Array { values, .. } = schedule
            .get_mut("deepseek4.attention.compress_ratios")
            .unwrap()
        else {
            unreachable!()
        };
        values[2] = GgufValue::Uint32(128);
        assert!(
            DeepSeekV4Config::from_gguf(&gguf_with_metadata(schedule))
                .unwrap_err()
                .to_string()
                .contains("compress_ratios")
        );

        let mut short_schedule = valid_metadata();
        let GgufValue::Array { values, .. } = short_schedule
            .get_mut("deepseek4.attention.compress_ratios")
            .unwrap()
        else {
            unreachable!()
        };
        values.pop();
        assert!(
            DeepSeekV4Config::from_gguf(&gguf_with_metadata(short_schedule))
                .unwrap_err()
                .to_string()
                .contains("compress_ratios")
        );
    }

    #[test]
    fn model_detection_is_exact_and_does_not_capture_other_ggufs() {
        assert!(is_deepseek_v4_flash(DEEPSEEK_V4_FLASH_REPOSITORY));
        assert!(is_deepseek_v4_flash(DEEPSEEK_V4_FLASH_MODEL));
        assert!(!is_deepseek_v4_flash(
            "hf://antirez/deepseek-v4-gguf/DeepSeek-V4-Pro-Q4_K_M.gguf"
        ));
        assert!(!is_deepseek_v4_flash("hf://other/deepseek-v4-flash"));
    }

    #[test]
    fn resident_prepare_ranges_are_sorted_exact_and_only_merge_adjacency() {
        let ranges = coalesce_deepseek_resident_ranges([
            DeepSeekResidentRange {
                byte_offset: 128,
                byte_len: 32,
            },
            DeepSeekResidentRange {
                byte_offset: 0,
                byte_len: 64,
            },
            DeepSeekResidentRange {
                byte_offset: 64,
                byte_len: 32,
            },
        ])
        .unwrap();

        assert_eq!(
            &*ranges,
            &[
                DeepSeekResidentRange {
                    byte_offset: 0,
                    byte_len: 96,
                },
                DeepSeekResidentRange {
                    byte_offset: 128,
                    byte_len: 32,
                },
            ]
        );

        assert!(
            coalesce_deepseek_resident_ranges([
                DeepSeekResidentRange {
                    byte_offset: 0,
                    byte_len: 65,
                },
                DeepSeekResidentRange {
                    byte_offset: 64,
                    byte_len: 1,
                },
            ])
            .unwrap_err()
            .to_string()
            .contains("overlap")
        );
    }
}
