//! DeepSeek V4 Flash source adapter for the existing FlashMoe runtime.
//!
//! The adapter accepts one pinned GGUF profile, resolves every semantic tensor
//! before publication, and preserves GGUF blocks in FlashMoe's canonical
//! resident and page-aligned expert stores. Runtime code consumes only those
//! stores and a load-time graph manifest; it never makes fallback decisions by
//! inspecting the source model.

use std::collections::BTreeMap;
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
use super::planning::{FlashMoePlan, FlashMoeRoutingPolicy, plan_unchecked_with_cache_version};
use super::weights::{
    DenseTensorRef, ExpertTensorRef, FlashMoeManifest, TENSOR_ALIGNMENT, TensorQuantization,
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
    pub(crate) fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = required_string(gguf, "general.architecture")?;
        if architecture != "deepseek4" {
            bail!(
                "DeepSeek V4 Flash GGUF declares architecture {architecture:?}, expected \"deepseek4\""
            );
        }
        let config = Self {
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
        if self.compress_ratios != expected_ratios {
            bail!(
                "DeepSeek V4 Flash attention.compress_ratios does not match the fixed 43-layer Flash graph"
            );
        }
        if self.swiglu_clamp_exp.len() != 43
            || self
                .swiglu_clamp_exp
                .iter()
                .any(|value| !float_matches(*value, 10.0))
        {
            bail!("DeepSeek V4 Flash swiglu_clamp_exp must contain 43 values equal to 10");
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
}
