//! Flash-MoE inspired inference backend for Qwen3.5-397B-A17B on Apple Silicon.
//!
//! The upstream flash-moe design is very different from llama.cpp: non-expert
//! tensors are mmap'd, routed expert tensors stay on SSD, and each token reads
//! only the active MoE experts with parallel `pread` before dispatching fused
//! Metal kernels.  This module captures that runtime contract in pb instead of
//! pretending a GGUF file is required for Qwen3.5.

use std::collections::VecDeque;
use std::ffi::OsString;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::ffi::{CString, c_char, c_void};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::ptr;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Seek, Write};

pub const QWEN35_MODEL: &str = "hf://Qwen/Qwen3.5-397B-A17B";
pub const QWEN35_MODEL_MARKER: &str = "qwen3.5-397b-a17b";
pub const LEGACY_QWEN_CODER_MARKER: &str = "qwen3-coder-next";
pub const CACHE_VERSION: &str = "flashmoe-v1";
pub const NUM_LAYERS: usize = 60;
pub const NUM_EXPERTS: usize = 512;
pub const ACTIVE_EXPERTS_PER_TOKEN: usize = 4;
pub const HIDDEN_DIM: usize = 4096;
pub const GROUP_SIZE: usize = 64;
const DENSE_PROJECTION_TILE_BYTES: usize = 64 * 1024 * 1024;
pub const FOUR_BIT_EXPERT_SIZE: u64 = 7_077_888;
pub const EXPECTED_EXPERT_BYTES: u64 =
    FOUR_BIT_EXPERT_SIZE * NUM_LAYERS as u64 * NUM_EXPERTS as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelection {
    FlashMoePreferred,
    LlamaCpp,
}

#[derive(Debug, Clone)]
pub struct FlashMoePlan {
    pub model: String,
    pub model_cache_dir: PathBuf,
    pub runtime_dir: PathBuf,
    pub non_expert_weights: PathBuf,
    pub tensor_manifest: PathBuf,
    pub model_config: PathBuf,
    pub tokenizer: PathBuf,
    pub experts_dir: PathBuf,
    pub uses_metal: bool,
    pub streams_experts_from_nand: bool,
    pub quantization: ExpertQuantization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertQuantization {
    FourBitProduction,
}

impl ExpertQuantization {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FourBitProduction => "4-bit expert weights",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStatus {
    pub ready: bool,
    pub missing: Vec<PathBuf>,
    pub expert_files: usize,
    pub expert_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub seed: u32,
}

#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub content: String,
    pub generated_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FlashMoeManifest {
    pub model: String,
    pub cache_version: String,
    pub dense_shards: Vec<String>,
    pub expert_tensors: Vec<ExpertTensorRef>,
    pub dense_tensors: Vec<DenseTensorRef>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpertTensorRef {
    pub tensor: String,
    pub shard: String,
    pub layer: Option<usize>,
    pub expert: Option<usize>,
    pub dtype: Option<String>,
    pub shape: Vec<usize>,
    pub source_offsets: Option<[u64; 2]>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DenseTensorRef {
    pub tensor: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub source_offsets: [u64; 2],
    pub runtime_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone)]
struct SafetensorShard {
    data_start: u64,
    tensors: BTreeMap<String, SafetensorTensorInfo>,
}

#[derive(Debug, Clone, Deserialize)]
struct SafetensorTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct QwenModelConfig {
    pub model_type: Option<String>,
    pub architectures: Option<Vec<String>>,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    pub rope_theta: Option<f64>,
    pub torch_dtype: Option<String>,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
}

impl QwenModelConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read model config {}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse model config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let model_type = self
            .model_type
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !(model_type.contains("qwen")
            || self.architectures.as_ref().is_some_and(|items| {
                items
                    .iter()
                    .any(|item| item.to_ascii_lowercase().contains("qwen"))
            }))
        {
            bail!(
                "Flash-MoE only supports Qwen-family configs, found model_type={:?} architectures={:?}",
                self.model_type,
                self.architectures
            );
        }
        if self.num_hidden_layers == 0
            || self.hidden_size == 0
            || self.num_attention_heads == 0
            || self.vocab_size == 0
        {
            bail!("Qwen config contains zero-valued required dimensions");
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            bail!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                self.hidden_size,
                self.num_attention_heads
            );
        }
        if let Some(kv_heads) = self.num_key_value_heads {
            if kv_heads == 0 || self.num_attention_heads % kv_heads != 0 {
                bail!(
                    "num_key_value_heads {kv_heads} must divide num_attention_heads {}",
                    self.num_attention_heads
                );
            }
        }
        let experts = self.num_experts.unwrap_or(NUM_EXPERTS);
        let active = self.num_experts_per_tok.unwrap_or(ACTIVE_EXPERTS_PER_TOKEN);
        if experts == 0 || active == 0 || active > experts {
            bail!(
                "invalid MoE routing config: num_experts={experts}, num_experts_per_tok={active}"
            );
        }
        if let Some(theta) = self.rope_theta {
            if !theta.is_finite() || theta <= 0.0 {
                bail!("rope_theta must be positive and finite, got {theta}");
            }
        }
        if let Some(dtype) = &self.torch_dtype {
            let dtype = dtype.to_ascii_lowercase();
            if !matches!(
                dtype.as_str(),
                "bfloat16" | "float16" | "float32" | "bf16" | "fp16" | "fp32"
            ) {
                bail!("unsupported Qwen dtype {dtype}; expected bf16/fp16/fp32 compatible weights");
            }
        }
        Ok(())
    }

    fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    fn experts(&self) -> usize {
        self.num_experts.unwrap_or(NUM_EXPERTS)
    }

    fn active_experts(&self) -> usize {
        self.num_experts_per_tok.unwrap_or(ACTIVE_EXPERTS_PER_TOKEN)
    }
}

pub fn select_backend(model: &str) -> BackendSelection {
    if supports_flashmoe(model) {
        BackendSelection::FlashMoePreferred
    } else {
        BackendSelection::LlamaCpp
    }
}

pub fn supports_flashmoe(model: &str) -> bool {
    is_arm_macos() && is_qwen35_or_legacy_alias(model)
}

pub fn is_arm_macos() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

pub fn is_qwen35_or_legacy_alias(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase();
    normalized.contains(QWEN35_MODEL_MARKER) || normalized.contains(LEGACY_QWEN_CODER_MARKER)
}

pub fn canonical_model(model: &str) -> String {
    if model
        .to_ascii_lowercase()
        .contains(LEGACY_QWEN_CODER_MARKER)
    {
        QWEN35_MODEL.to_string()
    } else {
        model.to_string()
    }
}

pub fn plan(model: &str, models_root: &Path) -> Option<FlashMoePlan> {
    supports_flashmoe(model).then(|| plan_unchecked(model, models_root))
}

pub fn plan_unchecked(model: &str, models_root: &Path) -> FlashMoePlan {
    let model = canonical_model(model);
    let model_cache_dir = models_root.join(crate::cache_dir_name(&model));
    let runtime_dir = model_cache_dir.join(CACHE_VERSION);
    FlashMoePlan {
        model,
        model_cache_dir,
        non_expert_weights: runtime_dir.join("model_weights.bin"),
        tensor_manifest: runtime_dir.join("model_weights.json"),
        model_config: runtime_dir.join("config.json"),
        tokenizer: runtime_dir.join("tokenizer.bin"),
        experts_dir: runtime_dir.join("packed_experts"),
        runtime_dir,
        uses_metal: true,
        streams_experts_from_nand: true,
        quantization: ExpertQuantization::FourBitProduction,
    }
}

impl FlashMoePlan {
    pub fn cache_status(&self) -> Result<CacheStatus> {
        let required = [
            self.non_expert_weights.clone(),
            self.tensor_manifest.clone(),
            self.model_config.clone(),
            self.tokenizer.clone(),
        ];
        let mut missing: Vec<PathBuf> = required
            .into_iter()
            .filter(|path| !path.is_file())
            .collect();
        if !self.experts_dir.is_dir() {
            missing.push(self.experts_dir.clone());
        }

        let (expert_files, expert_bytes) = expert_store_size(&self.experts_dir)?;
        let expected_expert_files =
            expected_packed_expert_files(&self.tensor_manifest).unwrap_or(NUM_LAYERS * NUM_EXPERTS);
        let ready = missing.is_empty() && expert_files >= expected_expert_files && expert_bytes > 0;

        Ok(CacheStatus {
            ready,
            missing,
            expert_files,
            expert_bytes,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "Flash-MoE {} for {}: {} layers, {} experts/layer, K={}, hidden={}, cache={}, expert store={} (~{} GiB)",
            CACHE_VERSION,
            self.model,
            NUM_LAYERS,
            NUM_EXPERTS,
            ACTIVE_EXPERTS_PER_TOKEN,
            HIDDEN_DIM,
            self.runtime_dir.display(),
            self.experts_dir.display(),
            EXPECTED_EXPERT_BYTES / (1024 * 1024 * 1024)
        )
    }
}

pub fn load(plan: &FlashMoePlan) -> Result<FlashMoeEngine> {
    let status = plan.cache_status()?;
    if !status.ready {
        bail!(
            "Flash-MoE cache is not ready for {}. Missing: {}. Found {} expert files totaling {} bytes. Run `pb pull {}` on ARM macOS to download and prepare the Qwen3.5 cache.",
            plan.model,
            format_missing(&status.missing),
            status.expert_files,
            status.expert_bytes,
            plan.model
        );
    }
    let config = QwenModelConfig::from_file(&plan.model_config)?;
    let dense = DenseStore::open(
        plan.non_expert_weights.clone(),
        plan.tensor_manifest.clone(),
    )?;
    validate_required_tensor_manifest(&config, dense.registry())?;
    Ok(FlashMoeEngine {
        plan: plan.clone(),
        experts: ExpertStore::open(plan.experts_dir.clone())?,
        scheduler: ExpertScheduler::new(ExpertStore::open(plan.experts_dir.clone())?),
        dense,
        tokenizer: QwenTokenizer::from_file(&plan.tokenizer)?,
        metal: MetalExecutor::new(plan, &config)?,
        config,
    })
}

#[derive(Debug, Clone)]
pub struct FlashMoeEngine {
    plan: FlashMoePlan,
    experts: ExpertStore,
    scheduler: ExpertScheduler,
    dense: DenseStore,
    tokenizer: QwenTokenizer,
    metal: Option<MetalExecutor>,
    config: QwenModelConfig,
}

#[derive(Debug, Clone)]
struct MetalExecutor {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    inner: Arc<MetalExecutorInner>,
}

impl MetalExecutor {
    fn new(plan: &FlashMoePlan, config: &QwenModelConfig) -> Result<Option<Self>> {
        if !plan.uses_metal {
            return Ok(None);
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return Ok(Some(Self {
                inner: Arc::new(MetalExecutorInner::new(plan, config)?),
            }));
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = config;
            Ok(None)
        }
    }

    fn project_q4_expert(
        &self,
        expert: &ExpertWeights,
        hidden: &[f32],
        width: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return self.inner.project_q4_expert(expert, hidden, width);
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            expert.project(hidden, width)
        }
    }

    fn route_topk(&self, scores: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return self.inner.route_topk(scores, k);
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            Ok(top_k(scores, k))
        }
    }

    fn rms_norm(&self, input: &[f32], weight: Option<&[f32]>) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return self.inner.rms_norm(input, weight);
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let mut out = input.to_vec();
            rms_norm_with_weight_in_place(&mut out, weight);
            Ok(out)
        }
    }

    fn dense_matvec(
        &self,
        weights: &[f32],
        input: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return self.inner.dense_matvec(weights, input, rows, cols);
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            Ok(cpu_dense_matvec(weights, input, rows, cols))
        }
    }

    fn apply_rope(
        &self,
        values: &[f32],
        position: usize,
        head_dim: usize,
        theta: f64,
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return self.inner.apply_rope(values, position, head_dim, theta);
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let mut out = values.to_vec();
            apply_rotary(&mut out, position, head_dim, theta);
            Ok(out)
        }
    }

    #[allow(dead_code)]
    fn causal_attention(
        &self,
        query: &[f32],
        keys_values: &[(&[f32], &[f32])],
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return self.inner.causal_attention(query, keys_values, head_dim);
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            Ok(causal_attention(query, keys_values, head_dim))
        }
    }

    fn record_kv(&self, position: usize, layer: usize, key: &[f32], value: &[f32]) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return self.inner.record_kv(position, layer, key, value);
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (position, layer, key, value);
            Ok(())
        }
    }

    fn causal_attention_cached(
        &self,
        position: usize,
        layer: usize,
        query: &[f32],
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return self
                .inner
                .causal_attention_cached(position, layer, query, head_dim);
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (position, layer, head_dim);
            Ok(vec![0.0; query.len()])
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
struct MetalExecutorInner {
    device: ObjcId,
    command_queue: ObjcId,
    q4_pipeline: ObjcId,
    route_pipeline: ObjcId,
    dense_matvec_pipeline: ObjcId,
    rms_norm_pipeline: ObjcId,
    rope_pipeline: ObjcId,
    attention_pipeline: ObjcId,
    kv_write_pipeline: ObjcId,
    kv_read_attention_pipeline: ObjcId,
    expert_mlp_pipeline: ObjcId,
    lm_head_pipeline: ObjcId,
    topk_vocab_pipeline: ObjcId,
    kv_cache: std::sync::Mutex<Option<MetalKvCacheInner>>,
    reusable: std::sync::Mutex<Vec<ObjcId>>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
struct MetalKvCacheInner {
    keys: ObjcId,
    values: ObjcId,
    layers: usize,
    max_context: usize,
    width: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe impl Send for MetalExecutorInner {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe impl Sync for MetalExecutorInner {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalExecutorInner {
    fn drop(&mut self) {
        unsafe {
            release(self.q4_pipeline);
            release(self.route_pipeline);
            release(self.dense_matvec_pipeline);
            release(self.rms_norm_pipeline);
            release(self.rope_pipeline);
            release(self.attention_pipeline);
            release(self.kv_write_pipeline);
            release(self.kv_read_attention_pipeline);
            release(self.expert_mlp_pipeline);
            release(self.lm_head_pipeline);
            release(self.topk_vocab_pipeline);
            if let Ok(kv_cache) = self.kv_cache.get_mut() {
                if let Some(kv_cache) = kv_cache.take() {
                    release(kv_cache.keys);
                    release(kv_cache.values);
                }
            }
            release(self.command_queue);
            release(self.device);
            if let Ok(buffers) = self.reusable.get_mut() {
                for buffer in buffers.drain(..) {
                    release(buffer);
                }
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalExecutorInner {
    fn new(plan: &FlashMoePlan, config: &QwenModelConfig) -> Result<Self> {
        unsafe {
            let device = MTLCreateSystemDefaultDevice();
            if device.is_null() {
                bail!(
                    "Metal is required for Flash-MoE on ARM macOS, but no default Metal device is available"
                );
            }
            let source = ns_string(METAL_SHADERS);
            let library = msg_send_id4(
                device,
                sel("newLibraryWithSource:options:error:"),
                source,
                ptr::null_mut(),
                ptr::null_mut(),
            );
            release(source);
            if library.is_null() {
                release(device);
                bail!("failed to compile Flash-MoE Metal shader library");
            }

            let q4_pipeline = compile_pipeline(device, library, "q4_fma_matvec")?;
            let route_pipeline = compile_pipeline(device, library, "route_top4")?;
            let dense_matvec_pipeline = compile_pipeline(device, library, "dense_matvec")?;
            let rms_norm_pipeline = compile_pipeline(device, library, "rms_norm")?;
            let rope_pipeline = compile_pipeline(device, library, "rope_apply")?;
            let attention_pipeline = compile_pipeline(device, library, "attention_scores")?;
            let kv_write_pipeline = compile_pipeline(device, library, "kv_cache_write")?;
            let kv_read_attention_pipeline =
                compile_pipeline(device, library, "kv_cache_read_attention")?;
            let expert_mlp_pipeline = compile_pipeline(device, library, "expert_mlp_fused")?;
            let lm_head_pipeline = compile_pipeline(device, library, "lm_head_logits")?;
            let topk_vocab_pipeline = compile_pipeline(device, library, "topk_vocab")?;
            release(library);

            let command_queue = msg_send_id0(device, sel("newCommandQueue"));
            if command_queue.is_null() {
                release(q4_pipeline);
                release(route_pipeline);
                release(dense_matvec_pipeline);
                release(rms_norm_pipeline);
                release(rope_pipeline);
                release(attention_pipeline);
                release(kv_write_pipeline);
                release(kv_read_attention_pipeline);
                release(expert_mlp_pipeline);
                release(lm_head_pipeline);
                release(topk_vocab_pipeline);
                release(device);
                bail!("failed to create Flash-MoE Metal command queue");
            }

            let runtime = DenseTransformerRuntime::new(config);
            let max_context = metal_kv_max_context(
                config,
                runtime.width,
                system_memory_bytes().unwrap_or(64 * 1024 * 1024 * 1024),
            );
            let kv_cache = allocate_metal_kv_cache(
                device,
                config.num_hidden_layers,
                max_context,
                runtime.width,
            )?;

            tracing::info!(
                model = %plan.model,
                layers = config.num_hidden_layers,
                max_context,
                kv_cache_mib = (metal_kv_cache_bytes(config.num_hidden_layers, max_context, runtime.width) / (1024 * 1024)),
                experts = config.experts(),
                "Flash-MoE Metal executor initialized"
            );

            Ok(Self {
                device,
                command_queue,
                q4_pipeline,
                route_pipeline,
                dense_matvec_pipeline,
                rms_norm_pipeline,
                rope_pipeline,
                attention_pipeline,
                kv_write_pipeline,
                kv_read_attention_pipeline,
                expert_mlp_pipeline,
                lm_head_pipeline,
                topk_vocab_pipeline,
                kv_cache: std::sync::Mutex::new(Some(kv_cache)),
                reusable: std::sync::Mutex::new(Vec::new()),
            })
        }
    }

    fn project_q4_expert(
        &self,
        expert: &ExpertWeights,
        hidden: &[f32],
        width: usize,
    ) -> Result<Vec<f32>> {
        if hidden.is_empty() || width == 0 {
            return Ok(vec![0.0; width]);
        }
        expert.mlp_with_projector(hidden, width, |tensor, input, output_width| {
            let Some(payload) = tensor.matvec_payload(
                input,
                output_width.max(tensor.shape.first().copied().unwrap_or(output_width)),
            ) else {
                return Ok(None);
            };
            let output = self.dispatch_q4_matvec(
                &payload.packed,
                &input[..payload.cols],
                &payload.scales,
                &payload.biases,
                payload.rows,
                payload.cols,
            )?;
            Ok(Some(output))
        })
    }

    fn dispatch_q4_matvec(
        &self,
        packed: &[u8],
        input: &[f32],
        scales: &[f32],
        biases: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        unsafe {
            let output_bytes = rows
                .checked_mul(std::mem::size_of::<f32>())
                .context("Metal output buffer size overflow")?;
            let packed_buffer = self.buffer_with_bytes(packed)?;
            let input_buffer = self.buffer_with_bytes(f32_as_bytes(input))?;
            let scale_buffer = self.buffer_with_bytes(f32_as_bytes(scales))?;
            let bias_buffer = self.buffer_with_bytes(f32_as_bytes(biases))?;
            let output_buffer = self.buffer_with_len(output_bytes)?;
            let cols_u32 = cols as u32;
            let groups = cols.div_ceil(GROUP_SIZE).max(1) as u32;
            let cols_buffer = self.buffer_with_bytes(u32_as_bytes(&cols_u32))?;
            let groups_buffer = self.buffer_with_bytes(u32_as_bytes(&groups))?;

            let command_buffer = msg_send_id0(self.command_queue, sel("commandBuffer"));
            if command_buffer.is_null() {
                bail!("failed to create Flash-MoE Metal command buffer");
            }
            let encoder = msg_send_id0(command_buffer, sel("computeCommandEncoder"));
            if encoder.is_null() {
                release(command_buffer);
                bail!("failed to create Flash-MoE Metal compute encoder");
            }

            msg_send_void1_id(encoder, sel("setComputePipelineState:"), self.q4_pipeline);
            set_buffer(encoder, packed_buffer, 0);
            set_buffer(encoder, input_buffer, 1);
            set_buffer(encoder, scale_buffer, 2);
            set_buffer(encoder, bias_buffer, 3);
            set_buffer(encoder, output_buffer, 4);
            set_buffer(encoder, cols_buffer, 5);
            set_buffer(encoder, groups_buffer, 6);
            dispatch_threads(encoder, rows as u64);
            msg_send_void0(encoder, sel("endEncoding"));
            msg_send_void0(command_buffer, sel("commit"));
            msg_send_void0(command_buffer, sel("waitUntilCompleted"));

            let contents = msg_send_ptr0(output_buffer, sel("contents"));
            let mut output = vec![0.0f32; rows];
            ptr::copy_nonoverlapping(contents.cast::<f32>(), output.as_mut_ptr(), rows);

            release(encoder);
            release(command_buffer);
            self.recycle(packed_buffer);
            self.recycle(input_buffer);
            self.recycle(scale_buffer);
            self.recycle(bias_buffer);
            self.recycle(output_buffer);
            self.recycle(cols_buffer);
            self.recycle(groups_buffer);
            Ok(output)
        }
    }

    fn route_topk(&self, scores: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if scores.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        if k > ACTIVE_EXPERTS_PER_TOKEN {
            return Ok(top_k(scores, k));
        }
        unsafe {
            let scores_buffer = self.buffer_with_bytes(f32_as_bytes(scores))?;
            let indices_buffer = self.buffer_with_len(4 * std::mem::size_of::<u32>())?;
            let weights_buffer = self.buffer_with_len(4 * std::mem::size_of::<f32>())?;
            let experts = scores.len() as u32;
            let experts_buffer = self.buffer_with_bytes(u32_as_bytes(&experts))?;

            let command_buffer = msg_send_id0(self.command_queue, sel("commandBuffer"));
            if command_buffer.is_null() {
                bail!("failed to create Flash-MoE routing command buffer");
            }
            let encoder = msg_send_id0(command_buffer, sel("computeCommandEncoder"));
            if encoder.is_null() {
                release(command_buffer);
                bail!("failed to create Flash-MoE routing compute encoder");
            }

            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.route_pipeline,
            );
            set_buffer(encoder, scores_buffer, 0);
            set_buffer(encoder, indices_buffer, 1);
            set_buffer(encoder, weights_buffer, 2);
            set_buffer(encoder, experts_buffer, 3);
            dispatch_threads(encoder, 1);
            msg_send_void0(encoder, sel("endEncoding"));
            msg_send_void0(command_buffer, sel("commit"));
            msg_send_void0(command_buffer, sel("waitUntilCompleted"));

            let indices_ptr = msg_send_ptr0(indices_buffer, sel("contents")).cast::<u32>();
            let weights_ptr = msg_send_ptr0(weights_buffer, sel("contents")).cast::<f32>();
            let mut routed = Vec::with_capacity(k);
            for idx in 0..k.min(ACTIVE_EXPERTS_PER_TOKEN) {
                routed.push((*indices_ptr.add(idx) as usize, *weights_ptr.add(idx)));
            }

            release(encoder);
            release(command_buffer);
            self.recycle(scores_buffer);
            self.recycle(indices_buffer);
            self.recycle(weights_buffer);
            self.recycle(experts_buffer);
            Ok(routed)
        }
    }

    fn rms_norm(&self, input: &[f32], weight: Option<&[f32]>) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let weights;
        let weight = if let Some(weight) = weight {
            weight
        } else {
            weights = vec![1.0f32; input.len()];
            &weights
        };
        unsafe {
            let input_buffer = self.buffer_with_bytes(f32_as_bytes(input))?;
            let weight_buffer = self.buffer_with_bytes(f32_as_bytes(weight))?;
            let output_buffer = self.buffer_with_len(std::mem::size_of_val(input))?;
            let width = input.len() as u32;
            let width_buffer = self.buffer_with_bytes(u32_as_bytes(&width))?;
            self.dispatch_unary(
                self.rms_norm_pipeline,
                &[input_buffer, weight_buffer, output_buffer, width_buffer],
                input.len() as u64,
            )?;
            let output = read_f32_buffer(output_buffer, input.len());
            self.recycle(input_buffer);
            self.recycle(weight_buffer);
            self.recycle(output_buffer);
            self.recycle(width_buffer);
            Ok(output)
        }
    }

    fn dense_matvec(
        &self,
        weights: &[f32],
        input: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        if rows == 0 || cols == 0 {
            return Ok(Vec::new());
        }
        unsafe {
            let weights_buffer = self.buffer_with_bytes(f32_as_bytes(weights))?;
            let input_buffer = self.buffer_with_bytes(f32_as_bytes(input))?;
            let output_buffer = self.buffer_with_len(rows * std::mem::size_of::<f32>())?;
            let cols_u32 = cols as u32;
            let cols_buffer = self.buffer_with_bytes(u32_as_bytes(&cols_u32))?;
            self.dispatch_unary(
                self.dense_matvec_pipeline,
                &[weights_buffer, input_buffer, output_buffer, cols_buffer],
                rows as u64,
            )?;
            let output = read_f32_buffer(output_buffer, rows);
            self.recycle(weights_buffer);
            self.recycle(input_buffer);
            self.recycle(output_buffer);
            self.recycle(cols_buffer);
            Ok(output)
        }
    }

    fn apply_rope(
        &self,
        values: &[f32],
        position: usize,
        head_dim: usize,
        theta: f64,
    ) -> Result<Vec<f32>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }
        unsafe {
            let values_buffer = self.buffer_with_bytes(f32_as_bytes(values))?;
            let position = position as u32;
            let head_dim = head_dim as u32;
            let theta = theta as f32;
            let position_buffer = self.buffer_with_bytes(u32_as_bytes(&position))?;
            let head_dim_buffer = self.buffer_with_bytes(u32_as_bytes(&head_dim))?;
            let theta_buffer = self.buffer_with_bytes(f32_as_bytes(&[theta]))?;
            self.dispatch_unary(
                self.rope_pipeline,
                &[
                    values_buffer,
                    position_buffer,
                    head_dim_buffer,
                    theta_buffer,
                ],
                (values.len() / 2) as u64,
            )?;
            let output = read_f32_buffer(values_buffer, values.len());
            self.recycle(values_buffer);
            self.recycle(position_buffer);
            self.recycle(head_dim_buffer);
            self.recycle(theta_buffer);
            Ok(output)
        }
    }

    fn causal_attention(
        &self,
        query: &[f32],
        keys_values: &[(&[f32], &[f32])],
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        if query.is_empty() || keys_values.is_empty() {
            return Ok(vec![0.0; query.len()]);
        }
        let width = query.len();
        let mut keys = Vec::with_capacity(keys_values.len() * width);
        let mut values = Vec::with_capacity(keys_values.len() * width);
        for (key, value) in keys_values {
            keys.extend(key.iter().copied().take(width));
            values.extend(value.iter().copied().take(width));
        }
        unsafe {
            let query_buffer = self.buffer_with_bytes(f32_as_bytes(query))?;
            let keys_buffer = self.buffer_with_bytes(f32_as_bytes(&keys))?;
            let scores_buffer =
                self.buffer_with_len(keys_values.len() * std::mem::size_of::<f32>())?;
            let width_u32 = width as u32;
            let head_dim_u32 = head_dim as u32;
            let width_buffer = self.buffer_with_bytes(u32_as_bytes(&width_u32))?;
            let head_dim_buffer = self.buffer_with_bytes(u32_as_bytes(&head_dim_u32))?;
            self.dispatch_unary(
                self.attention_pipeline,
                &[
                    query_buffer,
                    keys_buffer,
                    scores_buffer,
                    width_buffer,
                    head_dim_buffer,
                ],
                keys_values.len() as u64,
            )?;
            let mut scores = read_f32_buffer(scores_buffer, keys_values.len());
            softmax_in_place(&mut scores);

            let scores_buffer_2 = self.buffer_with_bytes(f32_as_bytes(&scores))?;
            let values_buffer = self.buffer_with_bytes(f32_as_bytes(&values))?;
            let output_buffer = self.buffer_with_len(width * std::mem::size_of::<f32>())?;
            let tokens_u32 = keys_values.len() as u32;
            let tokens_buffer = self.buffer_with_bytes(u32_as_bytes(&tokens_u32))?;
            self.dispatch_unary(
                self.kv_read_attention_pipeline,
                &[
                    scores_buffer_2,
                    values_buffer,
                    output_buffer,
                    width_buffer,
                    tokens_buffer,
                ],
                width as u64,
            )?;
            let output = read_f32_buffer(output_buffer, width);
            self.recycle(query_buffer);
            self.recycle(keys_buffer);
            self.recycle(scores_buffer);
            self.recycle(width_buffer);
            self.recycle(head_dim_buffer);
            self.recycle(scores_buffer_2);
            self.recycle(values_buffer);
            self.recycle(output_buffer);
            self.recycle(tokens_buffer);
            Ok(output)
        }
    }

    fn record_kv(&self, position: usize, layer: usize, key: &[f32], value: &[f32]) -> Result<()> {
        let kv_cache = self.kv_cache.lock().expect("metal kv cache poisoned");
        let kv_cache = kv_cache
            .as_ref()
            .context("Flash-MoE Metal KV cache is not allocated")?;
        if layer >= kv_cache.layers || position >= kv_cache.max_context {
            bail!(
                "Metal KV write layer {layer} position {position} exceeds cache {} layers x {} tokens",
                kv_cache.layers,
                kv_cache.max_context
            );
        }
        if key.len() < kv_cache.width || value.len() < kv_cache.width {
            bail!(
                "Metal KV write width mismatch: key {}, value {}, cache width {}",
                key.len(),
                value.len(),
                kv_cache.width
            );
        }
        unsafe {
            let key_buffer = self.buffer_with_bytes(f32_as_bytes(&key[..kv_cache.width]))?;
            let value_buffer = self.buffer_with_bytes(f32_as_bytes(&value[..kv_cache.width]))?;
            let offset = (layer * kv_cache.max_context + position)
                .checked_mul(kv_cache.width)
                .context("Metal KV cache offset overflow")? as u32;
            let width = kv_cache.width as u32;
            let offset_buffer = self.buffer_with_bytes(u32_as_bytes(&offset))?;
            let width_buffer = self.buffer_with_bytes(u32_as_bytes(&width))?;
            self.dispatch_unary(
                self.kv_write_pipeline,
                &[
                    key_buffer,
                    value_buffer,
                    kv_cache.keys,
                    kv_cache.values,
                    offset_buffer,
                    width_buffer,
                ],
                kv_cache.width as u64,
            )?;
            self.recycle(key_buffer);
            self.recycle(value_buffer);
            self.recycle(offset_buffer);
            self.recycle(width_buffer);
        }
        Ok(())
    }

    fn causal_attention_cached(
        &self,
        position: usize,
        layer: usize,
        query: &[f32],
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let kv_cache = self.kv_cache.lock().expect("metal kv cache poisoned");
        let kv_cache = kv_cache
            .as_ref()
            .context("Flash-MoE Metal KV cache is not allocated")?;
        if layer >= kv_cache.layers || position >= kv_cache.max_context {
            bail!(
                "Metal KV read layer {layer} position {position} exceeds cache {} layers x {} tokens",
                kv_cache.layers,
                kv_cache.max_context
            );
        }
        if query.len() < kv_cache.width {
            bail!(
                "Metal KV attention query width {} is smaller than cache width {}",
                query.len(),
                kv_cache.width
            );
        }
        let tokens = position + 1;
        let layer_offset_items = layer
            .checked_mul(kv_cache.max_context)
            .and_then(|items| items.checked_mul(kv_cache.width))
            .context("Metal KV layer offset overflow")?;
        let layer_offset_bytes = (layer_offset_items * std::mem::size_of::<f32>()) as u64;
        unsafe {
            let query_buffer = self.buffer_with_bytes(f32_as_bytes(&query[..kv_cache.width]))?;
            let scores_buffer = self.buffer_with_len(tokens * std::mem::size_of::<f32>())?;
            let width_u32 = kv_cache.width as u32;
            let head_dim_u32 = head_dim as u32;
            let width_buffer = self.buffer_with_bytes(u32_as_bytes(&width_u32))?;
            let head_dim_buffer = self.buffer_with_bytes(u32_as_bytes(&head_dim_u32))?;
            let command_buffer = msg_send_id0(self.command_queue, sel("commandBuffer"));
            if command_buffer.is_null() {
                bail!("failed to create Flash-MoE Metal command buffer");
            }
            let encoder = msg_send_id0(command_buffer, sel("computeCommandEncoder"));
            if encoder.is_null() {
                release(command_buffer);
                bail!("failed to create Flash-MoE Metal compute encoder");
            }
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.attention_pipeline,
            );
            set_buffer(encoder, query_buffer, 0);
            set_buffer_with_offset(encoder, kv_cache.keys, layer_offset_bytes, 1);
            set_buffer(encoder, scores_buffer, 2);
            set_buffer(encoder, width_buffer, 3);
            set_buffer(encoder, head_dim_buffer, 4);
            dispatch_threads(encoder, tokens as u64);
            msg_send_void0(encoder, sel("endEncoding"));
            msg_send_void0(command_buffer, sel("commit"));
            msg_send_void0(command_buffer, sel("waitUntilCompleted"));
            release(encoder);
            release(command_buffer);

            let mut scores = read_f32_buffer(scores_buffer, tokens);
            softmax_in_place(&mut scores);
            let scores_buffer_2 = self.buffer_with_bytes(f32_as_bytes(&scores))?;
            let output_buffer =
                self.buffer_with_len(kv_cache.width * std::mem::size_of::<f32>())?;
            let tokens_u32 = tokens as u32;
            let tokens_buffer = self.buffer_with_bytes(u32_as_bytes(&tokens_u32))?;
            let command_buffer = msg_send_id0(self.command_queue, sel("commandBuffer"));
            if command_buffer.is_null() {
                bail!("failed to create Flash-MoE Metal command buffer");
            }
            let encoder = msg_send_id0(command_buffer, sel("computeCommandEncoder"));
            if encoder.is_null() {
                release(command_buffer);
                bail!("failed to create Flash-MoE Metal compute encoder");
            }
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.kv_read_attention_pipeline,
            );
            set_buffer(encoder, scores_buffer_2, 0);
            set_buffer_with_offset(encoder, kv_cache.values, layer_offset_bytes, 1);
            set_buffer(encoder, output_buffer, 2);
            set_buffer(encoder, width_buffer, 3);
            set_buffer(encoder, tokens_buffer, 4);
            dispatch_threads(encoder, kv_cache.width as u64);
            msg_send_void0(encoder, sel("endEncoding"));
            msg_send_void0(command_buffer, sel("commit"));
            msg_send_void0(command_buffer, sel("waitUntilCompleted"));
            let output = read_f32_buffer(output_buffer, kv_cache.width);
            release(encoder);
            release(command_buffer);
            self.recycle(query_buffer);
            self.recycle(scores_buffer);
            self.recycle(width_buffer);
            self.recycle(head_dim_buffer);
            self.recycle(scores_buffer_2);
            self.recycle(output_buffer);
            self.recycle(tokens_buffer);
            Ok(output)
        }
    }

    unsafe fn dispatch_unary(
        &self,
        pipeline: ObjcId,
        buffers: &[ObjcId],
        threads: u64,
    ) -> Result<()> {
        let command_buffer = msg_send_id0(self.command_queue, sel("commandBuffer"));
        if command_buffer.is_null() {
            bail!("failed to create Flash-MoE Metal command buffer");
        }
        let encoder = msg_send_id0(command_buffer, sel("computeCommandEncoder"));
        if encoder.is_null() {
            release(command_buffer);
            bail!("failed to create Flash-MoE Metal compute encoder");
        }
        msg_send_void1_id(encoder, sel("setComputePipelineState:"), pipeline);
        for (idx, buffer) in buffers.iter().copied().enumerate() {
            set_buffer(encoder, buffer, idx as u64);
        }
        dispatch_threads(encoder, threads);
        msg_send_void0(encoder, sel("endEncoding"));
        msg_send_void0(command_buffer, sel("commit"));
        msg_send_void0(command_buffer, sel("waitUntilCompleted"));
        release(encoder);
        release(command_buffer);
        Ok(())
    }

    unsafe fn buffer_with_bytes(&self, bytes: &[u8]) -> Result<ObjcId> {
        let buffer = msg_send_id3_ptr_usize_u64(
            self.device,
            sel("newBufferWithBytes:length:options:"),
            bytes.as_ptr().cast(),
            bytes.len(),
            0,
        );
        if buffer.is_null() {
            bail!("failed to allocate Flash-MoE Metal upload buffer");
        }
        Ok(buffer)
    }

    unsafe fn buffer_with_len(&self, len: usize) -> Result<ObjcId> {
        let buffer =
            msg_send_id2_usize_u64(self.device, sel("newBufferWithLength:options:"), len, 0);
        if buffer.is_null() {
            bail!("failed to allocate Flash-MoE Metal output buffer");
        }
        Ok(buffer)
    }

    unsafe fn recycle(&self, buffer: ObjcId) {
        // Keep a tiny reuse pool so repeated decode steps do not immediately
        // churn all buffers under memory pressure. Drop older buffers quickly
        // because expert data is streamed and can be very large.
        let mut reusable = self.reusable.lock().expect("metal buffer pool poisoned");
        if reusable.len() < 8 {
            reusable.push(buffer);
        } else {
            release(buffer);
        }
    }
}

impl FlashMoeEngine {
    pub fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationOutput> {
        let prompt = self.tokenizer.apply_chat_template(&request.prompt);
        let prompt_tokens = self.tokenizer.encode(&prompt)?;
        let mut kv_cache = KvCache::new(
            self.config.num_hidden_layers,
            prompt_tokens.len() + request.max_tokens.max(0) as usize,
        );
        self.prefill(&prompt_tokens, &mut kv_cache)?;

        let mut sampler = TokenSampler::new(request.temperature, request.top_k, request.seed);
        let mut generated = Vec::new();
        for position in
            prompt_tokens.len()..prompt_tokens.len() + request.max_tokens.max(0) as usize
        {
            let logits =
                self.forward_next_token(&prompt_tokens, &generated, &mut kv_cache, position)?;
            let token = sampler.sample(&logits, &prompt_tokens, &generated)?;
            if self.tokenizer.is_eos(token) {
                break;
            }
            generated.push(token);
        }

        Ok(GenerationOutput {
            content: self.tokenizer.decode(&generated)?,
            generated_tokens: generated.len(),
        })
    }

    fn prefill(&mut self, prompt_tokens: &[u32], kv_cache: &mut KvCache) -> Result<()> {
        for (position, token) in prompt_tokens.iter().copied().enumerate() {
            kv_cache.record_prompt_token(position, token)?;
            // Populate the causal KV cache with the prompt tokens so decode can
            // attend to the full rendered prompt rather than only the latest
            // generated token.
            let _ = self.forward_token(token, kv_cache, position, false)?;
        }
        Ok(())
    }

    fn forward_next_token(
        &mut self,
        prompt_tokens: &[u32],
        generated: &[u32],
        kv_cache: &mut KvCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let previous = generated
            .last()
            .copied()
            .or_else(|| prompt_tokens.last().copied())
            .unwrap_or_else(|| self.tokenizer.eos_token_id());
        self.forward_token(previous, kv_cache, position, true)
    }

    fn forward_token(
        &mut self,
        previous: u32,
        kv_cache: &mut KvCache,
        position: usize,
        record_generated: bool,
    ) -> Result<Vec<f32>> {
        let runtime = DenseTransformerRuntime::new(&self.config);
        let mut hidden = self.dense.embedding(previous, runtime.width)?;
        let mut state = self.dense.seed(position, previous)? ^ (self.plan.model.len() as u64);

        for layer in 0..self.config.num_hidden_layers {
            let attention_residual = hidden.clone();
            let input_norm_name = layer_norm_tensor_name(layer, "input_layernorm");
            let input_norm_weight = self.dense.norm_weight(&input_norm_name, hidden.len())?;
            let mut normed = if let Some(metal) = &self.metal {
                metal.rms_norm(&hidden, input_norm_weight.as_deref())?
            } else {
                self.dense.rms_norm(input_norm_name.as_str(), &hidden)?
            };
            let mut q = self.dense.project_with_metal(
                self.metal.as_ref(),
                layer,
                "q_proj",
                &normed,
                runtime.width,
            )?;
            let mut k = self.dense.project_with_metal(
                self.metal.as_ref(),
                layer,
                "k_proj",
                &normed,
                runtime.width,
            )?;
            let v = self.dense.project_with_metal(
                self.metal.as_ref(),
                layer,
                "v_proj",
                &normed,
                runtime.width,
            )?;
            if let Some(metal) = &self.metal {
                q = metal.apply_rope(
                    &q,
                    position,
                    runtime.head_dim,
                    self.config.rope_theta.unwrap_or(1_000_000.0),
                )?;
                k = metal.apply_rope(
                    &k,
                    position,
                    runtime.head_dim,
                    self.config.rope_theta.unwrap_or(1_000_000.0),
                )?;
            } else {
                apply_rotary(
                    &mut q,
                    position,
                    runtime.head_dim,
                    self.config.rope_theta.unwrap_or(1_000_000.0),
                );
                apply_rotary(
                    &mut k,
                    position,
                    runtime.head_dim,
                    self.config.rope_theta.unwrap_or(1_000_000.0),
                );
            }
            let attended = if let Some(metal) = &self.metal {
                metal.record_kv(position, layer, &k, &v)?;
                metal.causal_attention_cached(position, layer, &q, runtime.head_dim)?
            } else {
                kv_cache.record_kv(position, layer, k, v)?;
                kv_cache.causal_attention(position, layer, &q, runtime.head_dim)?
            };
            let projected = self.dense.project_with_metal(
                self.metal.as_ref(),
                layer,
                "o_proj",
                &attended,
                runtime.width,
            )?;
            hidden = attention_residual;
            add_in_place(&mut hidden, &projected);

            let mlp_residual = hidden.clone();
            let post_norm_name = layer_norm_tensor_name(layer, "post_attention_layernorm");
            let post_norm_weight = self.dense.norm_weight(&post_norm_name, hidden.len())?;
            normed = if let Some(metal) = &self.metal {
                metal.rms_norm(&hidden, post_norm_weight.as_deref())?
            } else {
                self.dense.rms_norm(post_norm_name.as_str(), &hidden)?
            };
            let router_scores = self.dense.router_scores_with_metal(
                self.metal.as_ref(),
                layer,
                self.config.experts(),
                &normed,
            )?;
            let active = if let Some(metal) = &self.metal {
                metal.route_topk(&router_scores, self.config.active_experts())?
            } else {
                top_k(&router_scores, self.config.active_experts())
            };
            let active_ids: Vec<usize> = active.iter().map(|(expert, _)| *expert).collect();
            let pending_experts = self.scheduler.issue(layer, &active_ids)?;
            let mut weights: Vec<f32> = active.iter().map(|(_, score)| *score).collect();
            if self.metal.is_none() {
                softmax_in_place(&mut weights);
            }
            let experts = self.scheduler.finish(pending_experts)?;
            let mut moe = vec![0.0f32; runtime.width];
            for (expert, weight) in experts.iter().zip(weights) {
                state = state.wrapping_add(
                    expert
                        .mix_hash()
                        .wrapping_mul((weight.to_bits() as u64).max(1)),
                );
                let contribution = if let Some(metal) = &self.metal {
                    metal.project_q4_expert(expert, &normed, runtime.width)?
                } else {
                    expert.mlp(&normed, runtime.width)?
                };
                add_scaled_in_place(&mut moe, &contribution, weight);
            }
            hidden = mlp_residual;
            add_in_place(&mut hidden, &moe);
            kv_cache.record_layer_state(position, layer, state)?;
        }

        let final_norm_weight = self.dense.norm_weight("model.norm.weight", hidden.len())?;
        hidden = if let Some(metal) = &self.metal {
            metal.rms_norm(&hidden, final_norm_weight.as_deref())?
        } else {
            self.dense.rms_norm("model.norm.weight", &hidden)?
        };
        if record_generated {
            kv_cache.record_generated_token(position, previous)?;
        }
        Ok(self.dense.lm_head_logits_with_metal(
            self.metal.as_ref(),
            state,
            &hidden,
            &self.tokenizer,
        )?)
    }

    pub fn read_active_experts(
        &self,
        layer: usize,
        experts: &[usize],
    ) -> Result<Vec<ExpertWeights>> {
        self.experts.read_many(layer, experts)
    }

    pub fn expert_scheduler_metrics(&self) -> ExpertSchedulerSnapshot {
        self.scheduler.snapshot()
    }
}

#[derive(Debug, Clone, Copy)]
struct DenseTransformerRuntime {
    width: usize,
    head_dim: usize,
}

impl DenseTransformerRuntime {
    fn new(config: &QwenModelConfig) -> Self {
        let full_head_dim = config.hidden_size / config.num_attention_heads.max(1);
        Self {
            // The runtime width is the model hidden size. Truncating this value
            // makes every projection, residual, expert output, and LM-head row
            // numerically incompatible with the checkpoint even if the tensor
            // manifest validates successfully.
            width: config.hidden_size,
            head_dim: full_head_dim,
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn rms_norm_in_place(values: &mut [f32]) {
    rms_norm_with_weight_in_place(values, None)
}

fn rms_norm_with_weight_in_place(values: &mut [f32], weight: Option<&[f32]>) {
    let mean_square =
        values.iter().map(|value| value * value).sum::<f32>() / values.len().max(1) as f32;
    let scale = (mean_square + 1e-6).sqrt().recip();
    for (idx, value) in values.iter_mut().enumerate() {
        *value *= scale;
        if let Some(weight) = weight {
            if let Some(weight) = weight.get(idx) {
                *value *= *weight;
            }
        }
    }
}

fn apply_rotary(values: &mut [f32], position: usize, head_dim: usize, theta: f64) {
    let theta = theta.max(1.0) as f32;
    let head_dim = head_dim.max(2);
    for head in values.chunks_mut(head_dim) {
        let rotary_dims = head.len() - (head.len() % 2);
        for pair_idx in (0..rotary_dims).step_by(2) {
            let inv_freq = theta.powf(-(pair_idx as f32) / head_dim as f32);
            let angle = (position as f32) * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x = head[pair_idx];
            let y = head[pair_idx + 1];
            head[pair_idx] = x * cos - y * sin;
            head[pair_idx + 1] = x * sin + y * cos;
        }
    }
}

#[allow(dead_code)]
fn metal_kv_cache_bytes(layers: usize, max_context: usize, width: usize) -> usize {
    layers
        .saturating_mul(max_context)
        .saturating_mul(width)
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<f32>())
}

#[allow(dead_code)]
fn metal_kv_max_context(config: &QwenModelConfig, width: usize, total_ram_bytes: usize) -> usize {
    let requested = config.max_position_embeddings.unwrap_or(32_768).max(1);
    let bytes_per_token = config
        .num_hidden_layers
        .saturating_mul(width)
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1);
    let budget = (total_ram_bytes / 4).max(256 * 1024 * 1024);
    requested.min((budget / bytes_per_token).max(1))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn allocate_metal_kv_cache(
    device: ObjcId,
    layers: usize,
    max_context: usize,
    width: usize,
) -> Result<MetalKvCacheInner> {
    let bytes = metal_kv_cache_bytes(layers, max_context, width);
    unsafe {
        let keys = msg_send_id2_usize_u64(device, sel("newBufferWithLength:options:"), bytes, 0);
        if keys.is_null() {
            bail!("failed to allocate Flash-MoE Metal KV key buffer ({bytes} bytes)");
        }
        let values = msg_send_id2_usize_u64(device, sel("newBufferWithLength:options:"), bytes, 0);
        if values.is_null() {
            release(keys);
            bail!("failed to allocate Flash-MoE Metal KV value buffer ({bytes} bytes)");
        }
        Ok(MetalKvCacheInner {
            keys,
            values,
            layers,
            max_context,
            width,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn system_memory_bytes() -> Option<usize> {
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        (pages > 0 && page_size > 0).then(|| pages as usize * page_size as usize)
    }
}

fn causal_attention(q: &[f32], keys_values: &[(&[f32], &[f32])], head_dim: usize) -> Vec<f32> {
    if keys_values.is_empty() {
        return vec![0.0; q.len()];
    }
    let scale = (head_dim.max(1) as f32).sqrt().recip();
    let mut scores: Vec<f32> = keys_values
        .iter()
        .map(|(k, _)| q.iter().zip(*k).map(|(a, b)| a * b).sum::<f32>() * scale)
        .collect();
    softmax_in_place(&mut scores);
    let mut out = vec![0.0f32; q.len()];
    for (weight, (_, value)) in scores.into_iter().zip(keys_values) {
        add_scaled_in_place(&mut out, value, weight);
    }
    out
}

fn cpu_dense_matvec(weights: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let used_cols = cols.min(input.len());
    let mut out = vec![0.0f32; rows];
    for (row, slot) in out.iter_mut().enumerate() {
        let start = row.saturating_mul(cols);
        let end = start.saturating_add(used_cols).min(weights.len());
        let acc = weights
            .get(start..end)
            .unwrap_or(&[])
            .iter()
            .zip(input.iter().take(used_cols))
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        *slot = acc / (used_cols.max(1) as f32).sqrt();
    }
    out
}

fn add_in_place(target: &mut [f32], update: &[f32]) {
    for (target, update) in target.iter_mut().zip(update) {
        *target += *update;
    }
}

fn add_scaled_in_place(target: &mut [f32], update: &[f32], scale: f32) {
    for (target, update) in target.iter_mut().zip(update) {
        *target += *update * scale;
    }
}

#[derive(Debug, Clone)]
struct QwenTokenizer {
    id_to_token: Vec<String>,
    token_to_id: BTreeMap<String, u32>,
    merge_ranks: BTreeMap<(String, String), usize>,
    unk_token: Option<String>,
    model_kind: TokenizerModelKind,
    special_tokens: Vec<(String, u32)>,
    chat_template: Option<String>,
    eos_token: u32,
    im_start: Option<u32>,
    im_end: Option<u32>,
    vocab_size: usize,
    candidate_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenizerModelKind {
    Bpe,
    WordLevel,
    Other,
}

impl QwenTokenizer {
    fn from_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read tokenizer {}", path.display()))?;
        let config_path = path.with_file_name("tokenizer_config.json");
        let config_bytes = if config_path.is_file() {
            Some(fs::read(&config_path).with_context(|| {
                format!("failed to read tokenizer config {}", config_path.display())
            })?)
        } else {
            None
        };
        Self::from_json_bytes_with_config(&bytes, config_bytes.as_deref())
            .with_context(|| format!("failed to parse tokenizer {}", path.display()))
    }

    #[cfg(test)]
    fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_json_bytes_with_config(bytes, None)
    }

    fn from_json_bytes_with_config(bytes: &[u8], config_bytes: Option<&[u8]>) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_slice(bytes).context("tokenizer JSON is invalid")?;
        let chat_template = parse_tokenizer_chat_template(config_bytes)?;
        let model_kind = match value
            .pointer("/model/type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "bpe" => TokenizerModelKind::Bpe,
            "wordlevel" => TokenizerModelKind::WordLevel,
            _ => TokenizerModelKind::Other,
        };
        let mut token_to_id = BTreeMap::new();
        if let Some(vocab) = value
            .pointer("/model/vocab")
            .and_then(serde_json::Value::as_object)
        {
            for (token, id) in vocab {
                if let Some(id) = id.as_u64().and_then(|id| u32::try_from(id).ok()) {
                    token_to_id.insert(token.clone(), id);
                }
            }
        }
        if token_to_id.is_empty() {
            bail!("Qwen tokenizer JSON does not contain model.vocab");
        }
        let mut special_tokens = Vec::new();
        if let Some(added) = value
            .get("added_tokens")
            .and_then(serde_json::Value::as_array)
        {
            for token in added {
                let content = token.get("content").and_then(serde_json::Value::as_str);
                let id = token
                    .get("id")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|id| u32::try_from(id).ok());
                if let (Some(content), Some(id)) = (content, id) {
                    token_to_id.insert(content.to_string(), id);
                    if token
                        .get("special")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
                    {
                        special_tokens.push((content.to_string(), id));
                    }
                }
            }
        }
        special_tokens.sort_by(|(left, _), (right, _)| {
            right.len().cmp(&left.len()).then_with(|| left.cmp(right))
        });
        let merge_ranks = parse_tokenizer_merges(&value)?;
        let unk_token = value
            .pointer("/model/unk_token")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let eos_token = ["<|im_end|>", "<|endoftext|>", "</s>"]
            .iter()
            .find_map(|token| token_to_id.get(*token).copied())
            .with_context(
                || "Qwen tokenizer is missing an EOS token (<|im_end|>, <|endoftext|>, or </s>)",
            )?;
        let im_start = token_to_id.get("<|im_start|>").copied();
        let im_end = token_to_id.get("<|im_end|>").copied();
        let max_id = token_to_id.values().copied().max().unwrap_or(eos_token) as usize;
        let vocab_size = max_id + 1;
        let mut id_to_token = vec![String::new(); vocab_size];
        for (token, id) in &token_to_id {
            if let Some(slot) = id_to_token.get_mut(*id as usize) {
                *slot = token.clone();
            }
        }
        let mut candidate_ids: Vec<u32> = token_to_id
            .values()
            .copied()
            .filter(|id| (*id as usize) < vocab_size)
            .collect();
        candidate_ids.sort_unstable();
        candidate_ids.dedup();
        if candidate_ids.is_empty() {
            bail!("Qwen tokenizer vocabulary is empty");
        }
        Ok(Self {
            id_to_token,
            token_to_id,
            merge_ranks,
            unk_token,
            model_kind,
            special_tokens,
            chat_template,
            eos_token,
            im_start,
            im_end,
            vocab_size,
            candidate_ids,
        })
    }

    fn apply_chat_template(&self, prompt: &str) -> String {
        if let Some(template) = &self.chat_template {
            return render_qwen_chat_template(template, prompt);
        }
        if self.im_start.is_some() && self.im_end.is_some() && !prompt.contains("<|im_start|>") {
            format!(
                "<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"
            )
        } else {
            prompt.to_string()
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(match self.model_kind {
            TokenizerModelKind::Bpe => self.encode_byte_level_bpe(text),
            TokenizerModelKind::WordLevel | TokenizerModelKind::Other => {
                self.encode_wordlevel_compatible(text)
            }
        })
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut out = String::new();
        let mut byte_buffer = Vec::new();
        for token in tokens
            .iter()
            .copied()
            .take_while(|token| !self.is_eos(*token))
        {
            if let Some(piece) = self
                .id_to_token
                .get(token as usize)
                .filter(|piece| !piece.is_empty())
            {
                if !piece.starts_with("<|") {
                    if let Some(bytes) = byte_level_piece_to_bytes(piece) {
                        byte_buffer.extend(bytes);
                    } else {
                        if !byte_buffer.is_empty() {
                            out.push_str(&String::from_utf8_lossy(&byte_buffer));
                            byte_buffer.clear();
                        }
                        out.push_str(&decode_token_piece(piece));
                    }
                }
            }
        }
        if !byte_buffer.is_empty() {
            out.push_str(&String::from_utf8_lossy(&byte_buffer));
        }
        Ok(out)
    }

    fn is_eos(&self, token: u32) -> bool {
        token == self.eos_token
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn candidate_token_ids(&self) -> &[u32] {
        &self.candidate_ids
    }

    fn encode_wordlevel_compatible(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut cursor = 0usize;
        while cursor < text.len() {
            if let Some((special, id)) = self.special_tokens.iter().find_map(|(special, id)| {
                text[cursor..]
                    .starts_with(special)
                    .then_some((special.as_str(), *id))
            }) {
                out.push(id);
                cursor += special.len();
                continue;
            }

            let ch = text[cursor..].chars().next().expect("cursor inside text");
            let piece = ch.to_string();
            if ch.is_whitespace() {
                if let Some(id) = self.token_to_id.get(&piece).copied() {
                    out.push(id);
                }
                cursor += ch.len_utf8();
                continue;
            }

            let start = cursor;
            cursor += ch.len_utf8();
            while cursor < text.len() {
                let next = text[cursor..].chars().next().expect("cursor inside text");
                if next.is_whitespace()
                    || self
                        .special_tokens
                        .iter()
                        .any(|(special, _)| text[cursor..].starts_with(special))
                {
                    break;
                }
                cursor += next.len_utf8();
            }
            out.extend(self.encode_piece(&text[start..cursor]));
        }
        out
    }

    fn encode_piece(&self, piece: &str) -> Vec<u32> {
        if let Some(id) = self.token_to_id.get(piece).copied() {
            return vec![id];
        }
        let mut parts: Vec<String> = piece.chars().map(|ch| ch.to_string()).collect();
        while parts.len() > 1 {
            let Some((idx, _)) = parts
                .windows(2)
                .enumerate()
                .filter_map(|(idx, pair)| {
                    self.merge_ranks
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .copied()
                        .map(|rank| (idx, rank))
                })
                .min_by_key(|(_, rank)| *rank)
            else {
                break;
            };
            let merged = format!("{}{}", parts[idx], parts[idx + 1]);
            parts.splice(idx..=idx + 1, [merged]);
        }
        let unk = self
            .unk_token
            .as_ref()
            .and_then(|token| self.token_to_id.get(token).copied());
        parts
            .into_iter()
            .filter_map(|part| self.token_to_id.get(&part).copied().or(unk))
            .collect()
    }

    fn encode_byte_level_bpe(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut cursor = 0usize;
        while cursor < text.len() {
            if let Some((special, id)) = self.special_tokens.iter().find_map(|(special, id)| {
                text[cursor..]
                    .starts_with(special)
                    .then_some((special.as_str(), *id))
            }) {
                out.push(id);
                cursor += special.len();
                continue;
            }

            let start = cursor;
            cursor += text[cursor..]
                .chars()
                .next()
                .expect("cursor inside text")
                .len_utf8();
            while cursor < text.len()
                && !self
                    .special_tokens
                    .iter()
                    .any(|(special, _)| text[cursor..].starts_with(special))
            {
                cursor += text[cursor..]
                    .chars()
                    .next()
                    .expect("cursor inside text")
                    .len_utf8();
            }
            out.extend(self.encode_byte_level_piece(&text[start..cursor]));
        }
        out
    }

    fn encode_byte_level_piece(&self, piece: &str) -> Vec<u32> {
        if let Some(id) = self.token_to_id.get(piece).copied() {
            return vec![id];
        }
        let byte_piece: String = piece.bytes().map(byte_to_unicode).collect();
        if let Some(id) = self.token_to_id.get(&byte_piece).copied() {
            return vec![id];
        }
        let mut parts: Vec<String> = byte_piece.chars().map(|ch| ch.to_string()).collect();
        while parts.len() > 1 {
            let Some((idx, _)) = parts
                .windows(2)
                .enumerate()
                .filter_map(|(idx, pair)| {
                    self.merge_ranks
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .copied()
                        .map(|rank| (idx, rank))
                })
                .min_by_key(|(_, rank)| *rank)
            else {
                break;
            };
            let merged = format!("{}{}", parts[idx], parts[idx + 1]);
            parts.splice(idx..=idx + 1, [merged]);
        }
        let unk = self
            .unk_token
            .as_ref()
            .and_then(|token| self.token_to_id.get(token).copied());
        parts
            .into_iter()
            .filter_map(|part| self.token_to_id.get(&part).copied().or(unk))
            .collect()
    }
}

fn parse_tokenizer_merges(value: &serde_json::Value) -> Result<BTreeMap<(String, String), usize>> {
    let mut out = BTreeMap::new();
    let Some(merges) = value
        .pointer("/model/merges")
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(out);
    };
    for (rank, merge) in merges.iter().enumerate() {
        let pair = if let Some(text) = merge.as_str() {
            let mut parts = text.split_whitespace();
            match (parts.next(), parts.next(), parts.next()) {
                (Some(left), Some(right), None) => Some((left.to_string(), right.to_string())),
                _ => None,
            }
        } else if let Some(parts) = merge.as_array() {
            match (
                parts.first().and_then(serde_json::Value::as_str),
                parts.get(1).and_then(serde_json::Value::as_str),
            ) {
                (Some(left), Some(right)) => Some((left.to_string(), right.to_string())),
                _ => None,
            }
        } else {
            None
        };
        if let Some(pair) = pair {
            out.insert(pair, rank);
        }
    }
    Ok(out)
}

fn parse_tokenizer_chat_template(config_bytes: Option<&[u8]>) -> Result<Option<String>> {
    let Some(bytes) = config_bytes else {
        return Ok(None);
    };
    let value: serde_json::Value =
        serde_json::from_slice(bytes).context("tokenizer_config.json is invalid")?;
    Ok(value
        .get("chat_template")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string))
}

fn render_qwen_chat_template(template: &str, prompt: &str) -> String {
    if prompt.contains("<|im_start|>") {
        return prompt.to_string();
    }
    if template.contains("{% for message in messages %}") && template.contains("{% endfor %}") {
        return render_single_message_jinja_chat_template(template, "user", prompt, true);
    }
    template
        .replace("{{ prompt }}", prompt)
        .replace("{{prompt}}", prompt)
}

fn render_single_message_jinja_chat_template(
    template: &str,
    role: &str,
    content: &str,
    add_generation_prompt: bool,
) -> String {
    let mut rendered = String::new();
    let mut rest = template;
    while let Some(start) = rest.find("{% for message in messages %}") {
        rendered.push_str(&rest[..start]);
        let block_start = start + "{% for message in messages %}".len();
        let Some(relative_end) = rest[block_start..].find("{% endfor %}") else {
            rendered.push_str(&rest[start..]);
            return render_generation_prompt_blocks(&rendered, add_generation_prompt);
        };
        let block_end = block_start + relative_end;
        rendered.push_str(&render_message_template_block(
            &rest[block_start..block_end],
            role,
            content,
        ));
        rest = &rest[block_end + "{% endfor %}".len()..];
    }
    rendered.push_str(rest);
    render_generation_prompt_blocks(&rendered, add_generation_prompt)
}

fn render_message_template_block(block: &str, role: &str, content: &str) -> String {
    block
        .replace("{{ message['role'] }}", role)
        .replace("{{message['role']}}", role)
        .replace("{{ message[\"role\"] }}", role)
        .replace("{{message[\"role\"]}}", role)
        .replace("{{ message.role }}", role)
        .replace("{{message.role}}", role)
        .replace("{{ message['content'] }}", content)
        .replace("{{message['content']}}", content)
        .replace("{{ message[\"content\"] }}", content)
        .replace("{{message[\"content\"]}}", content)
        .replace("{{ message.content }}", content)
        .replace("{{message.content}}", content)
}

fn render_generation_prompt_blocks(template: &str, add_generation_prompt: bool) -> String {
    let mut rendered = String::new();
    let mut rest = template;
    while let Some(start) = rest.find("{% if add_generation_prompt %}") {
        rendered.push_str(&rest[..start]);
        let block_start = start + "{% if add_generation_prompt %}".len();
        let Some(relative_end) = rest[block_start..].find("{% endif %}") else {
            rendered.push_str(&rest[start..]);
            return rendered;
        };
        let block_end = block_start + relative_end;
        if add_generation_prompt {
            rendered.push_str(&rest[block_start..block_end]);
        }
        rest = &rest[block_end + "{% endif %}".len()..];
    }
    rendered.push_str(rest);
    rendered
}

fn decode_token_piece(piece: &str) -> String {
    piece.replace('Ġ', " ").replace('▁', " ")
}

fn byte_to_unicode(byte: u8) -> char {
    match byte {
        b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF => char::from(byte),
        _ => {
            let mut next = 256u32;
            for candidate in 0u16..=u8::MAX as u16 {
                let candidate = candidate as u8;
                if matches!(candidate, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF) {
                    continue;
                }
                if candidate == byte {
                    return char::from_u32(next).expect("valid GPT-2 byte-level unicode scalar");
                }
                next += 1;
            }
            unreachable!("all bytes are covered by GPT-2 byte-level unicode mapping")
        }
    }
}

fn unicode_to_byte(ch: char) -> Option<u8> {
    let codepoint = ch as u32;
    if matches!(codepoint, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF) {
        return u8::try_from(codepoint).ok();
    }
    let mut next = 256u32;
    for candidate in 0u16..=u8::MAX as u16 {
        let candidate = candidate as u8;
        if matches!(candidate, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF) {
            continue;
        }
        if codepoint == next {
            return Some(candidate);
        }
        next += 1;
    }
    None
}

fn byte_level_piece_to_bytes(piece: &str) -> Option<Vec<u8>> {
    piece.chars().map(unicode_to_byte).collect()
}

#[derive(Debug, Clone)]
struct KvCache {
    layers: usize,
    capacity: usize,
    prompt_tokens: Vec<(usize, u32)>,
    generated_tokens: Vec<(usize, u32)>,
    layer_states: Vec<(usize, usize, u64)>,
    kv: Vec<Vec<Option<(Vec<f32>, Vec<f32>)>>>,
}

impl KvCache {
    fn new(layers: usize, capacity: usize) -> Self {
        Self {
            layers,
            capacity,
            prompt_tokens: Vec::new(),
            generated_tokens: Vec::new(),
            layer_states: Vec::new(),
            kv: vec![vec![None; capacity]; layers],
        }
    }

    fn record_prompt_token(&mut self, position: usize, token: u32) -> Result<()> {
        self.ensure_position(position)?;
        self.prompt_tokens.push((position, token));
        Ok(())
    }

    fn record_generated_token(&mut self, position: usize, token: u32) -> Result<()> {
        self.ensure_position(position)?;
        self.generated_tokens.push((position, token));
        Ok(())
    }

    fn record_layer_state(&mut self, position: usize, layer: usize, state: u64) -> Result<()> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        self.layer_states.push((position, layer, state));
        Ok(())
    }

    fn record_kv(
        &mut self,
        position: usize,
        layer: usize,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<()> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        self.kv[layer][position] = Some((key, value));
        Ok(())
    }

    fn causal_attention(
        &self,
        position: usize,
        layer: usize,
        query: &[f32],
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        let keys_values: Vec<(&[f32], &[f32])> = self.kv[layer]
            .iter()
            .take(position + 1)
            .filter_map(|entry| {
                entry
                    .as_ref()
                    .map(|(key, value)| (key.as_slice(), value.as_slice()))
            })
            .collect();
        Ok(causal_attention(query, &keys_values, head_dim))
    }

    #[allow(dead_code)]
    fn keys_values(&self, position: usize, layer: usize) -> Result<Vec<(&[f32], &[f32])>> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        Ok(self.kv[layer]
            .iter()
            .take(position + 1)
            .filter_map(|entry| {
                entry
                    .as_ref()
                    .map(|(key, value)| (key.as_slice(), value.as_slice()))
            })
            .collect())
    }

    fn ensure_position(&self, position: usize) -> Result<()> {
        if position >= self.capacity {
            bail!(
                "KV cache position {position} exceeds capacity {}",
                self.capacity
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct TokenSampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    state: u64,
}

impl TokenSampler {
    fn new(temperature: f32, top_k: i32, seed: u32) -> Self {
        let deterministic = temperature <= 0.0 || top_k <= 1;
        Self {
            temperature,
            top_k: usize::try_from(top_k.max(1)).unwrap_or(1),
            top_p: if deterministic { 1.0 } else { 0.95 },
            repeat_penalty: if deterministic { 1.0 } else { 1.05 },
            state: u64::from(seed).max(1),
        }
    }

    fn sample(&mut self, logits: &[f32], prompt: &[u32], generated: &[u32]) -> Result<u32> {
        if logits.is_empty() {
            bail!("cannot sample from empty logits");
        }
        let processed = self.process_logits(logits, prompt, generated);
        let mut candidates = top_k(&processed, self.top_k.min(processed.len()).max(1));
        if candidates.is_empty() {
            bail!("no logits candidates available");
        }
        if self.temperature <= 0.0 || candidates.len() == 1 {
            return u32::try_from(candidates[0].0).context("sampled token id does not fit u32");
        }
        let inv_temp = 1.0 / self.temperature.max(1e-6);
        let mut probabilities: Vec<f32> = candidates
            .iter()
            .map(|(_, logit)| *logit * inv_temp)
            .collect();
        softmax_in_place(&mut probabilities);
        self.apply_top_p(&mut candidates, &mut probabilities);
        let draw = self.next_f32();
        let mut cumulative = 0.0f32;
        let mut fallback = candidates[0].0;
        for ((token, _), weight) in candidates.into_iter().zip(probabilities) {
            fallback = token;
            cumulative += weight;
            if draw <= cumulative {
                return u32::try_from(token).context("sampled token id does not fit u32");
            }
        }
        u32::try_from(fallback).context("sampled token id does not fit u32")
    }

    fn process_logits(&self, logits: &[f32], prompt: &[u32], generated: &[u32]) -> Vec<f32> {
        let mut processed: Vec<f32> = logits
            .iter()
            .map(|logit| {
                if logit.is_finite() {
                    *logit
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        if self.repeat_penalty > 1.0 {
            let window = generated.len().saturating_sub(256);
            for token in prompt.iter().chain(generated[window..].iter()).copied() {
                let idx = token as usize;
                if let Some(logit) = processed.get_mut(idx) {
                    if *logit > 0.0 {
                        *logit /= self.repeat_penalty;
                    } else {
                        *logit *= self.repeat_penalty;
                    }
                }
            }
        }
        processed
    }

    fn apply_top_p(&self, candidates: &mut Vec<(usize, f32)>, probabilities: &mut Vec<f32>) {
        if self.top_p >= 1.0 || candidates.len() <= 1 {
            return;
        }
        let mut cumulative = 0.0f32;
        let mut keep = candidates.len();
        for (idx, probability) in probabilities.iter().enumerate() {
            cumulative += *probability;
            if cumulative >= self.top_p {
                keep = idx + 1;
                break;
            }
        }
        keep = keep.max(1);
        candidates.truncate(keep);
        probabilities.truncate(keep);
        let total = probabilities.iter().sum::<f32>();
        if total.is_finite() && total > 0.0 {
            for probability in probabilities {
                *probability /= total;
            }
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TensorQuantization {
    None,
    Q4 { group_size: usize, format: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeTensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub alignment: u64,
    pub quantization: TensorQuantization,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorRegistry {
    tensors: BTreeMap<String, RuntimeTensorEntry>,
}

impl TensorRegistry {
    pub fn load(manifest_path: &Path) -> Result<Self> {
        let manifest: FlashMoeManifest =
            serde_json::from_slice(&fs::read(manifest_path).with_context(|| {
                format!(
                    "failed to read Flash-MoE tensor manifest {}",
                    manifest_path.display()
                )
            })?)
            .with_context(|| {
                format!(
                    "failed to parse Flash-MoE tensor manifest {}",
                    manifest_path.display()
                )
            })?;
        Ok(Self::from_manifest(&manifest))
    }

    fn from_manifest(manifest: &FlashMoeManifest) -> Self {
        let mut tensors = BTreeMap::new();
        for tensor in &manifest.dense_tensors {
            tensors.insert(
                tensor.tensor.clone(),
                RuntimeTensorEntry {
                    name: tensor.tensor.clone(),
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    byte_offset: tensor.runtime_offset,
                    byte_len: tensor.byte_len,
                    alignment: TENSOR_ALIGNMENT,
                    quantization: TensorQuantization::None,
                },
            );
        }
        for tensor in &manifest.expert_tensors {
            if let Some([start, end]) = tensor.source_offsets {
                tensors.insert(
                    tensor.tensor.clone(),
                    RuntimeTensorEntry {
                        name: tensor.tensor.clone(),
                        dtype: tensor
                            .dtype
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string()),
                        shape: tensor.shape.clone(),
                        byte_offset: start,
                        byte_len: end.saturating_sub(start),
                        alignment: TENSOR_ALIGNMENT,
                        quantization: TensorQuantization::Q4 {
                            group_size: GROUP_SIZE,
                            format: ExpertQuantization::FourBitProduction.as_str().to_string(),
                        },
                    },
                );
            }
        }
        Self { tensors }
    }

    pub fn tensor(&self, canonical_name: &str) -> Option<&RuntimeTensorEntry> {
        self.tensors.get(canonical_name)
    }

    pub fn require(&self, canonical_name: &str) -> Result<&RuntimeTensorEntry> {
        self.tensor(canonical_name)
            .with_context(|| format!("Flash-MoE tensor registry is missing {canonical_name}"))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

fn validate_required_tensor_manifest(
    config: &QwenModelConfig,
    registry: &TensorRegistry,
) -> Result<()> {
    let head_dim = config.hidden_size / config.num_attention_heads.max(1);
    let kv_width = config.kv_heads() * head_dim;
    let intermediate = config
        .moe_intermediate_size
        .or(config.intermediate_size)
        .context("Qwen config is missing moe_intermediate_size/intermediate_size")?;
    require_tensor_shape(
        registry,
        "model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
    )?;
    require_tensor_shape(registry, "model.norm.weight", &[config.hidden_size])?;
    if registry.tensor("lm_head.weight").is_none()
        && registry.tensor("model.embed_tokens.weight").is_none()
    {
        bail!(
            "Flash-MoE dense tensor registry is missing lm_head.weight and tied model.embed_tokens.weight"
        );
    }
    if registry.tensor("lm_head.weight").is_some() {
        require_tensor_shape(
            registry,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
    }
    for layer in 0..config.num_hidden_layers {
        require_tensor_shape(
            registry,
            &attention_tensor_name(layer, "q_proj"),
            &[config.hidden_size, config.hidden_size],
        )?;
        require_tensor_shape(
            registry,
            &attention_tensor_name(layer, "k_proj"),
            &[kv_width, config.hidden_size],
        )?;
        require_tensor_shape(
            registry,
            &attention_tensor_name(layer, "v_proj"),
            &[kv_width, config.hidden_size],
        )?;
        require_tensor_shape(
            registry,
            &attention_tensor_name(layer, "o_proj"),
            &[config.hidden_size, config.hidden_size],
        )?;
        require_tensor_shape(
            registry,
            &layer_norm_tensor_name(layer, "input_layernorm"),
            &[config.hidden_size],
        )?;
        require_tensor_shape(
            registry,
            &layer_norm_tensor_name(layer, "post_attention_layernorm"),
            &[config.hidden_size],
        )?;
        require_tensor_shape(
            registry,
            &router_tensor_name(layer),
            &[config.experts(), config.hidden_size],
        )?;
        for expert in 0..config.experts() {
            let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
            require_tensor_shape(
                registry,
                &format!("{prefix}.gate_proj.weight"),
                &[intermediate, config.hidden_size],
            )?;
            require_tensor_shape(
                registry,
                &format!("{prefix}.up_proj.weight"),
                &[intermediate, config.hidden_size],
            )?;
            require_tensor_shape(
                registry,
                &format!("{prefix}.down_proj.weight"),
                &[config.hidden_size, intermediate],
            )?;
        }
    }
    Ok(())
}

fn require_tensor_shape(
    registry: &TensorRegistry,
    canonical_name: &str,
    expected_shape: &[usize],
) -> Result<()> {
    let tensor = registry.require(canonical_name)?;
    if dtype_size(&tensor.dtype).is_none() {
        bail!(
            "Flash-MoE tensor {canonical_name} has unsupported dtype {}",
            tensor.dtype
        );
    }
    if tensor.shape.as_slice() != expected_shape {
        bail!(
            "Flash-MoE tensor {canonical_name} has shape {:?}; expected {:?}",
            tensor.shape,
            expected_shape
        );
    }
    Ok(())
}

fn ensure_synthetic_runtime_allowed(tensor_name: &str) -> Result<()> {
    if cfg!(test) {
        Ok(())
    } else {
        bail!(
            "Flash-MoE tensor {tensor_name} is unavailable; synthetic runtime fallback is disabled outside tests"
        )
    }
}

fn attention_tensor_name(layer: usize, projection: &str) -> String {
    format!("model.layers.{layer}.self_attn.{projection}.weight")
}

fn router_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.gate.weight")
}

fn layer_norm_tensor_name(layer: usize, name: &str) -> String {
    format!("model.layers.{layer}.{name}.weight")
}

fn dense_projection_tile_rows(cols: usize, rows: usize) -> usize {
    let bytes_per_row = cols.saturating_mul(std::mem::size_of::<f32>()).max(1);
    (DENSE_PROJECTION_TILE_BYTES / bytes_per_row)
        .max(1)
        .min(rows.max(1))
}

#[derive(Debug, Clone)]
pub struct DenseStore {
    manifest_path: PathBuf,
    len: u64,
    mmap: Arc<memmap2::Mmap>,
    registry: TensorRegistry,
    resident: Arc<std::sync::Mutex<DenseTensorCache>>,
}

#[derive(Debug, Default)]
struct DenseTensorCache {
    tensors: BTreeMap<String, Arc<Vec<f32>>>,
    bytes: usize,
    max_bytes: usize,
}

impl DenseTensorCache {
    fn with_budget(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            ..Self::default()
        }
    }

    fn get(&self, name: &str) -> Option<Arc<Vec<f32>>> {
        self.tensors.get(name).cloned()
    }

    fn insert(&mut self, name: String, tensor: Arc<Vec<f32>>) {
        let bytes = tensor.len() * std::mem::size_of::<f32>();
        if bytes > self.max_bytes {
            return;
        }
        while self.bytes.saturating_add(bytes) > self.max_bytes && !self.tensors.is_empty() {
            let Some(victim) = self.tensors.keys().next().cloned() else {
                break;
            };
            if let Some(previous) = self.tensors.remove(&victim) {
                self.bytes = self
                    .bytes
                    .saturating_sub(previous.len() * std::mem::size_of::<f32>());
            }
        }
        if let Some(previous) = self.tensors.insert(name, tensor) {
            self.bytes = self
                .bytes
                .saturating_sub(previous.len() * std::mem::size_of::<f32>());
        }
        self.bytes = self.bytes.saturating_add(bytes);
    }
}

impl DenseStore {
    pub fn open(path: PathBuf, manifest_path: PathBuf) -> Result<Self> {
        let file = fs::File::open(&path)
            .with_context(|| format!("failed to open dense store {}", path.display()))?;
        let len = file
            .metadata()
            .with_context(|| format!("failed to stat dense store {}", path.display()))?
            .len();
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .with_context(|| format!("failed to memory-map dense store {}", path.display()))?
        };
        let registry = TensorRegistry::load(&manifest_path)?;
        Ok(Self {
            manifest_path,
            len,
            mmap: Arc::new(mmap),
            registry,
            resident: Arc::new(std::sync::Mutex::new(DenseTensorCache::with_budget(
                512 * 1024 * 1024,
            ))),
        })
    }

    pub fn registry(&self) -> &TensorRegistry {
        &self.registry
    }

    fn seed(&self, position: usize, previous: u32) -> Result<u64> {
        Ok(self
            .read_u64(position as u64)?
            .wrapping_add(u64::from(previous)))
    }

    fn embedding(&self, token: u32, width: usize) -> Result<Vec<f32>> {
        if let Some(row) =
            self.read_tensor_row_f32("model.embed_tokens.weight", token as usize, width)?
        {
            return Ok(row);
        }
        bail!(
            "Flash-MoE dense tensor registry cannot provide model.embed_tokens.weight row for token {token}; refusing synthetic embeddings"
        )
    }

    fn project(&self, layer: usize, name: &str, input: &[f32], width: usize) -> Result<Vec<f32>> {
        let tensor_name = attention_tensor_name(layer, name);
        if let Some(projected) = self.matvec_tensor_prefix(&tensor_name, input, width)? {
            return Ok(projected);
        }
        ensure_synthetic_runtime_allowed(&tensor_name)?;
        let salt = self.tensor_seed(&tensor_name, stable_hash(name) ^ ((layer as u64) << 32));
        let mut out = vec![0.0f32; width];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (col, value) in input.iter().enumerate() {
                let bits = self.read_u64(salt ^ ((row as u64) << 20) ^ col as u64)?;
                let weight = ((bits >> 40) as f32 / ((1u64 << 24) as f32)) * 2.0 - 1.0;
                acc = value.mul_add(weight, acc);
            }
            *slot = acc / (input.len().max(1) as f32).sqrt();
        }
        Ok(out)
    }

    fn project_with_metal(
        &self,
        metal: Option<&MetalExecutor>,
        layer: usize,
        name: &str,
        input: &[f32],
        width: usize,
    ) -> Result<Vec<f32>> {
        let tensor_name = attention_tensor_name(layer, name);
        if let Some(metal) = metal {
            if let Some(entry) = self.registry.tensor(&tensor_name) {
                let cols = entry.shape.last().copied().unwrap_or(0);
                let rows = entry.shape.first().copied().unwrap_or(width).min(width);
                let used_cols = cols.min(input.len());
                if rows > 0 && used_cols > 0 && used_cols == cols {
                    return self.metal_matvec_tiled(metal, &tensor_name, input, rows, cols, width);
                }
            }
        }
        self.project(layer, name, input, width)
    }

    fn rms_norm(&self, canonical_name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let mut out = input.to_vec();
        let weight = self.norm_weight(canonical_name, input.len())?;
        if weight.is_none() {
            ensure_synthetic_runtime_allowed(canonical_name)?;
        }
        rms_norm_with_weight_in_place(&mut out, weight.as_deref());
        Ok(out)
    }

    fn norm_weight(&self, canonical_name: &str, width: usize) -> Result<Option<Vec<f32>>> {
        self.read_tensor_row_f32(canonical_name, 0, width)
    }

    fn router_projection(&self, layer: usize, expert: usize, hidden: &[f32]) -> Result<f32> {
        let tensor_name = router_tensor_name(layer);
        if let Some(row) = self.read_tensor_row_f32(&tensor_name, expert, hidden.len())? {
            let acc = row
                .iter()
                .zip(hidden)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
            return Ok(acc / (hidden.len().max(1) as f32).sqrt());
        }
        ensure_synthetic_runtime_allowed(&tensor_name)?;
        let salt = self.tensor_seed(&tensor_name, ((layer as u64) << 32) ^ expert as u64);
        let mut acc = 0.0f32;
        for (idx, value) in hidden.iter().enumerate() {
            let bits = self.read_u64(salt ^ idx as u64)?;
            let weight = ((bits >> 40) as f32 / ((1u64 << 24) as f32)) * 2.0 - 1.0;
            acc = value.mul_add(weight, acc);
        }
        Ok(acc / (hidden.len().max(1) as f32).sqrt())
    }

    fn router_scores_with_metal(
        &self,
        metal: Option<&MetalExecutor>,
        layer: usize,
        experts: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let tensor_name = router_tensor_name(layer);
        if let Some(metal) = metal {
            if let Some(entry) = self.registry.tensor(&tensor_name) {
                let cols = entry.shape.last().copied().unwrap_or(0);
                let rows = entry.shape.first().copied().unwrap_or(experts).min(experts);
                if rows > 0 && cols > 0 && cols <= hidden.len() {
                    let mut scores =
                        self.metal_matvec_tiled(metal, &tensor_name, hidden, rows, cols, experts)?;
                    for score in scores.iter_mut().take(rows) {
                        *score /= (cols.max(1) as f32).sqrt();
                    }
                    return Ok(scores);
                }
            }
        }

        let mut router_scores = vec![0.0f32; experts];
        for (expert, score) in router_scores.iter_mut().enumerate() {
            *score = self.router_projection(layer, expert, hidden)?;
        }
        Ok(router_scores)
    }

    fn lm_head_logits_with_metal(
        &self,
        metal: Option<&MetalExecutor>,
        _state: u64,
        hidden: &[f32],
        tokenizer: &QwenTokenizer,
    ) -> Result<Vec<f32>> {
        let lm_head_name = self.lm_head_tensor_name()?;
        if let Some(metal) = metal {
            if let Some(entry) = self.registry.tensor(lm_head_name) {
                let cols = entry.shape.last().copied().unwrap_or(0);
                let rows = entry
                    .shape
                    .first()
                    .copied()
                    .unwrap_or(tokenizer.vocab_size())
                    .min(tokenizer.vocab_size());
                if rows > 0 && cols > 0 && cols <= hidden.len() {
                    let mut logits = vec![f32::NEG_INFINITY; tokenizer.vocab_size()];
                    let tile_rows = dense_projection_tile_rows(cols, rows);
                    for start in (0..rows).step_by(tile_rows) {
                        let end = (start + tile_rows).min(rows);
                        let tensor = self.read_tensor_rows_f32(lm_head_name, start, end - start)?;
                        let mut projected =
                            metal.dense_matvec(&tensor, hidden, end - start, cols)?;
                        for value in &mut projected {
                            *value /= (cols.max(1) as f32).sqrt();
                        }
                        for (offset, value) in projected.into_iter().enumerate() {
                            logits[start + offset] = value;
                        }
                    }
                    return Ok(logits);
                }
            }
        }

        self.lm_head_logits(lm_head_name, hidden, tokenizer)
    }

    fn lm_head_logits(
        &self,
        lm_head_name: &str,
        hidden: &[f32],
        tokenizer: &QwenTokenizer,
    ) -> Result<Vec<f32>> {
        let mut logits = vec![f32::NEG_INFINITY; tokenizer.vocab_size()];
        for token in tokenizer.candidate_token_ids().iter().copied() {
            let idx = token as usize;
            if idx >= logits.len() {
                continue;
            }
            let Some(row) = self.read_tensor_row_f32(lm_head_name, idx, hidden.len())? else {
                bail!(
                    "Flash-MoE LM head tensor {lm_head_name} cannot provide row for token {idx}; refusing synthetic logits"
                );
            };
            logits[idx] = row
                .iter()
                .zip(hidden)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                / (hidden.len().max(1) as f32).sqrt();
        }
        Ok(logits)
    }

    fn lm_head_tensor_name(&self) -> Result<&'static str> {
        if self.registry.tensor("lm_head.weight").is_some() {
            Ok("lm_head.weight")
        } else if self.registry.tensor("model.embed_tokens.weight").is_some() {
            Ok("model.embed_tokens.weight")
        } else {
            bail!(
                "Flash-MoE dense tensor registry is missing lm_head.weight and tied model.embed_tokens.weight"
            )
        }
    }

    fn matvec_tensor_prefix(
        &self,
        canonical_name: &str,
        input: &[f32],
        width: usize,
    ) -> Result<Option<Vec<f32>>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            return Ok(None);
        };
        let cols = entry.shape.last().copied().unwrap_or(0);
        if cols == 0 {
            return Ok(None);
        }
        let rows = entry.shape.first().copied().unwrap_or(width).min(width);
        let used_cols = input.len().min(cols);
        if let Some(tensor) = self.dense_tensor_f32(canonical_name)? {
            let mut out = vec![0.0f32; width];
            for (row, slot) in out.iter_mut().take(rows).enumerate() {
                let start = row
                    .checked_mul(cols)
                    .context("dense resident tensor row offset overflow")?;
                let end = start
                    .checked_add(used_cols)
                    .context("dense resident tensor row length overflow")?;
                let Some(weights) = tensor.get(start..end) else {
                    return Ok(None);
                };
                let acc = weights
                    .iter()
                    .zip(input.iter().take(used_cols))
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>();
                *slot = acc / (used_cols.max(1) as f32).sqrt();
            }
            return Ok(Some(out));
        }
        let mut out = vec![0.0f32; width];
        for (row, slot) in out.iter_mut().take(rows).enumerate() {
            let weights = self.read_tensor_row_f32(canonical_name, row, used_cols)?;
            let Some(weights) = weights else {
                return Ok(None);
            };
            let acc = weights
                .iter()
                .zip(input.iter().take(used_cols))
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
            *slot = acc / (used_cols.max(1) as f32).sqrt();
        }
        Ok(Some(out))
    }

    fn metal_matvec_tiled(
        &self,
        metal: &MetalExecutor,
        canonical_name: &str,
        input: &[f32],
        rows: usize,
        cols: usize,
        output_width: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; output_width];
        let tile_rows = dense_projection_tile_rows(cols, rows);
        for start in (0..rows).step_by(tile_rows) {
            let end = (start + tile_rows).min(rows);
            let tensor = self.read_tensor_rows_f32(canonical_name, start, end - start)?;
            let projected = metal.dense_matvec(&tensor, input, end - start, cols)?;
            for (offset, value) in projected.into_iter().enumerate() {
                output[start + offset] = value;
            }
        }
        Ok(output)
    }

    fn dense_tensor_f32(&self, canonical_name: &str) -> Result<Option<Arc<Vec<f32>>>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            return Ok(None);
        };
        if let Some(tensor) = self
            .resident
            .lock()
            .expect("dense tensor cache poisoned")
            .get(canonical_name)
        {
            return Ok(Some(tensor));
        }
        let bytes = self.read_range(entry.byte_offset, entry.byte_len as usize)?;
        let tensor = Arc::new(decode_dense_tensor_f32(&entry.dtype, &bytes)?);
        self.resident
            .lock()
            .expect("dense tensor cache poisoned")
            .insert(canonical_name.to_string(), tensor.clone());
        Ok(Some(tensor))
    }

    fn read_tensor_rows_f32(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            bail!("Flash-MoE dense tensor registry is missing {canonical_name}");
        };
        let Some(element_size) = dtype_size(&entry.dtype) else {
            bail!(
                "Flash-MoE dense tensor {} has unsupported dtype {}",
                entry.name,
                entry.dtype
            );
        };
        let cols = entry.shape.last().copied().unwrap_or(0);
        if entry.shape.is_empty() || cols == 0 || row_count == 0 {
            return Ok(Vec::new());
        }
        let rows = entry
            .shape
            .iter()
            .take(entry.shape.len() - 1)
            .product::<usize>()
            .max(1);
        let end_row = start_row
            .checked_add(row_count)
            .context("dense tensor tile row range overflow")?;
        if end_row > rows {
            bail!(
                "Flash-MoE dense tensor {} tile rows {}..{} exceed row count {}",
                entry.name,
                start_row,
                end_row,
                rows
            );
        }
        let row_bytes = cols
            .checked_mul(element_size)
            .context("dense tensor tile row byte length overflow")?;
        let byte_offset = start_row
            .checked_mul(row_bytes)
            .context("dense tensor tile byte offset overflow")?;
        let byte_len = row_count
            .checked_mul(row_bytes)
            .context("dense tensor tile byte length overflow")?;
        let bytes = self.read_range(entry.byte_offset + byte_offset as u64, byte_len)?;
        decode_dense_tensor_f32(&entry.dtype, &bytes)
    }

    fn read_tensor_row_f32(
        &self,
        canonical_name: &str,
        row: usize,
        requested_cols: usize,
    ) -> Result<Option<Vec<f32>>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            return Ok(None);
        };
        let Some(element_size) = dtype_size(&entry.dtype) else {
            bail!(
                "Flash-MoE dense tensor {} has unsupported dtype {}",
                entry.name,
                entry.dtype
            );
        };
        if entry.shape.is_empty() || requested_cols == 0 {
            return Ok(None);
        }
        let cols = entry.shape.last().copied().unwrap_or(0);
        if cols == 0 {
            return Ok(None);
        }
        let rows = entry
            .shape
            .iter()
            .take(entry.shape.len() - 1)
            .product::<usize>()
            .max(1);
        if row >= rows {
            return Ok(None);
        }
        let used_cols = requested_cols.min(cols);
        let row_offset = row
            .checked_mul(cols)
            .and_then(|items| items.checked_mul(element_size))
            .context("dense tensor row offset overflow")? as u64;
        let byte_len = used_cols
            .checked_mul(element_size)
            .context("dense tensor row byte length overflow")?;
        let bytes = self.read_range(entry.byte_offset + row_offset, byte_len)?;
        Ok(Some(decode_dense_tensor_f32(&entry.dtype, &bytes)?))
    }

    fn read_range(&self, offset: u64, byte_len: usize) -> Result<Vec<u8>> {
        if offset.saturating_add(byte_len as u64) > self.len {
            bail!(
                "dense tensor read {}..{} exceeds store length {}",
                offset,
                offset.saturating_add(byte_len as u64),
                self.len
            );
        }
        Ok(self.mmap[offset as usize..offset as usize + byte_len].to_vec())
    }

    fn tensor_seed(&self, canonical_name: &str, fallback: u64) -> u64 {
        if let Some(tensor) = self.registry.tensor(canonical_name) {
            stable_hash(&tensor.name)
                ^ stable_hash(&tensor.dtype)
                ^ tensor.byte_offset
                ^ tensor.byte_len.rotate_left(7)
                ^ ((tensor.shape.iter().copied().product::<usize>() as u64) << 11)
        } else {
            tracing::trace!(
                tensor = canonical_name,
                manifest = %self.manifest_path.display(),
                "Flash-MoE tensor registry missing canonical tensor; using deterministic fallback seed"
            );
            fallback
        }
    }

    fn read_u64(&self, offset_hint: u64) -> Result<u64> {
        if self.len == 0 {
            return Ok(offset_hint.rotate_left(13) ^ 0x9e37_79b9_7f4a_7c15);
        }
        let offset = offset_hint % self.len;
        let mut out = [0u8; 8];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.mmap[((offset as usize) + i) % self.mmap.len()];
        }
        Ok(u64::from_le_bytes(out) ^ offset_hint.rotate_left(7))
    }
}

fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype.to_ascii_uppercase().as_str() {
        "F32" | "FLOAT32" | "FP32" => Some(4),
        "BF16" | "BFLOAT16" | "F16" | "FLOAT16" | "FP16" => Some(2),
        "U8" | "I8" => Some(1),
        _ => None,
    }
}

fn decode_dense_tensor_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype.to_ascii_uppercase().as_str() {
        "F32" | "FLOAT32" | "FP32" => {
            if bytes.len() % 4 != 0 {
                bail!(
                    "F32 tensor byte length {} is not divisible by 4",
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        "BF16" | "BFLOAT16" => {
            if bytes.len() % 2 != 0 {
                bail!(
                    "BF16 tensor byte length {} is not divisible by 2",
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let hi = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                    f32::from_bits(hi << 16)
                })
                .collect())
        }
        "F16" | "FLOAT16" | "FP16" => {
            if bytes.len() % 2 != 0 {
                bail!(
                    "F16 tensor byte length {} is not divisible by 2",
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect())
        }
        "U8" => Ok(bytes.iter().map(|value| *value as f32).collect()),
        "I8" => Ok(bytes.iter().map(|value| (*value as i8) as f32).collect()),
        other => bail!("unsupported dense tensor dtype {other}"),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = (bits & 0x03ff) as u32;
    let value = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut frac = frac;
                let mut exp = -14i32;
                while (frac & 0x0400) == 0 {
                    frac <<= 1;
                    exp -= 1;
                }
                frac &= 0x03ff;
                sign | (((exp + 127) as u32) << 23) | (frac << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13),
    };
    f32::from_bits(value)
}

#[derive(Debug, Clone)]
pub struct ExpertStore {
    root: PathBuf,
}

impl ExpertStore {
    pub fn open(root: PathBuf) -> Result<Self> {
        if !root.is_dir() {
            bail!("expert store {} does not exist", root.display());
        }
        Ok(Self { root })
    }

    pub fn read_many(&self, layer: usize, experts: &[usize]) -> Result<Vec<ExpertWeights>> {
        if layer >= NUM_LAYERS {
            bail!("layer {layer} is outside 0..{NUM_LAYERS}");
        }
        if experts.len() > ACTIVE_EXPERTS_PER_TOKEN {
            bail!(
                "requested {} experts, max active experts is {}",
                experts.len(),
                ACTIVE_EXPERTS_PER_TOKEN
            );
        }
        let root = Arc::new(self.root.clone());
        let mut handles = Vec::with_capacity(experts.len());
        for &expert in experts {
            if expert >= NUM_EXPERTS {
                bail!("expert {expert} is outside 0..{NUM_EXPERTS}");
            }
            let root = Arc::clone(&root);
            handles.push(thread::spawn(move || read_one_expert(&root, layer, expert)));
        }
        let mut out = Vec::with_capacity(handles.len());
        for handle in handles {
            out.push(
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("expert read thread panicked"))??,
            );
        }
        Ok(out)
    }
}

const DEFAULT_EXPERT_CACHE_BYTES: usize = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ExpertKey {
    layer: usize,
    expert: usize,
}

#[derive(Debug)]
struct PendingExpertRead {
    key: ExpertKey,
    cached: Option<Arc<ExpertWeights>>,
    handle: Option<thread::JoinHandle<Result<ExpertWeights>>>,
    issued_at: Instant,
}

#[derive(Debug, Clone, Default)]
struct ExpertSchedulerMetrics {
    issued_reads: u64,
    cache_hits: u64,
    cache_misses: u64,
    read_failures: u64,
    total_read_latency: Duration,
    max_read_latency: Duration,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertSchedulerSnapshot {
    pub issued_reads: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub read_failures: u64,
    pub cached_bytes: usize,
    pub max_cached_bytes: usize,
    pub total_read_latency: Duration,
    pub max_read_latency: Duration,
}

#[derive(Debug, Clone)]
struct ExpertScheduler {
    store: ExpertStore,
    cache: BTreeMap<ExpertKey, Arc<ExpertWeights>>,
    lru: VecDeque<ExpertKey>,
    cached_bytes: usize,
    max_cached_bytes: usize,
    metrics: ExpertSchedulerMetrics,
}

impl ExpertScheduler {
    fn new(store: ExpertStore) -> Self {
        Self {
            store,
            cache: BTreeMap::new(),
            lru: VecDeque::new(),
            cached_bytes: 0,
            max_cached_bytes: DEFAULT_EXPERT_CACHE_BYTES,
            metrics: ExpertSchedulerMetrics::default(),
        }
    }

    fn issue(&mut self, layer: usize, experts: &[usize]) -> Result<Vec<PendingExpertRead>> {
        let mut pending = Vec::with_capacity(experts.len());
        for expert in experts {
            let key = ExpertKey {
                layer,
                expert: *expert,
            };
            let issued_at = Instant::now();
            if let Some(cached) = self.cache.get(&key).cloned() {
                self.metrics.cache_hits = self.metrics.cache_hits.saturating_add(1);
                self.touch(key);
                pending.push(PendingExpertRead {
                    key,
                    cached: Some(cached),
                    handle: None,
                    issued_at,
                });
                continue;
            }

            self.metrics.cache_misses = self.metrics.cache_misses.saturating_add(1);
            self.metrics.issued_reads = self.metrics.issued_reads.saturating_add(1);
            let root = self.store.root.clone();
            pending.push(PendingExpertRead {
                key,
                cached: None,
                handle: Some(thread::spawn(move || {
                    read_one_expert(&root, key.layer, key.expert)
                })),
                issued_at,
            });
        }
        Ok(pending)
    }

    fn finish(&mut self, pending: Vec<PendingExpertRead>) -> Result<Vec<Arc<ExpertWeights>>> {
        let mut out = Vec::with_capacity(pending.len());
        for pending in pending {
            if let Some(cached) = pending.cached {
                out.push(cached);
                continue;
            }

            let handle = pending
                .handle
                .context("pending expert read missing thread handle")?;
            let started = pending.issued_at;
            let expert = match handle.join() {
                Ok(result) => result,
                Err(_) => {
                    self.metrics.read_failures = self.metrics.read_failures.saturating_add(1);
                    bail!("expert read thread panicked");
                }
            }?;
            let latency = started.elapsed();
            self.metrics.total_read_latency += latency;
            self.metrics.max_read_latency = self.metrics.max_read_latency.max(latency);
            let expert = Arc::new(expert);
            self.insert_cache(pending.key, expert.clone());
            out.push(expert);
        }
        Ok(out)
    }

    fn insert_cache(&mut self, key: ExpertKey, expert: Arc<ExpertWeights>) {
        let size = expert.packed.len();
        if let Some(previous) = self.cache.insert(key, expert) {
            self.cached_bytes = self.cached_bytes.saturating_sub(previous.packed.len());
        }
        self.cached_bytes = self.cached_bytes.saturating_add(size);
        self.touch(key);
        self.evict_to_budget();
    }

    fn touch(&mut self, key: ExpertKey) {
        self.lru.retain(|existing| *existing != key);
        self.lru.push_back(key);
    }

    fn evict_to_budget(&mut self) {
        while self.cached_bytes > self.max_cached_bytes {
            let Some(victim) = self.lru.pop_front() else {
                break;
            };
            if let Some(expert) = self.cache.remove(&victim) {
                self.cached_bytes = self.cached_bytes.saturating_sub(expert.packed.len());
            }
        }
    }

    fn snapshot(&self) -> ExpertSchedulerSnapshot {
        ExpertSchedulerSnapshot {
            issued_reads: self.metrics.issued_reads,
            cache_hits: self.metrics.cache_hits,
            cache_misses: self.metrics.cache_misses,
            read_failures: self.metrics.read_failures,
            cached_bytes: self.cached_bytes,
            max_cached_bytes: self.max_cached_bytes,
            total_read_latency: self.metrics.total_read_latency,
            max_read_latency: self.metrics.max_read_latency,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertWeights {
    pub layer: usize,
    pub expert: usize,
    pub packed: Vec<u8>,
    pub records: Vec<PackedExpertTensor>,
}

impl ExpertWeights {
    pub fn q4_fma_matvec(
        &self,
        input: &[f32],
        scales: &[f32],
        biases: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        q4_fma_matvec(&self.packed, input, scales, biases, rows, cols)
    }

    fn project(&self, hidden: &[f32], width: usize) -> Result<Vec<f32>> {
        if !self.records.is_empty() {
            let mut out = vec![0.0f32; width];
            let mut used = 0usize;
            for tensor in &self.records {
                let Some(payload) = tensor.matvec_payload(hidden, width) else {
                    continue;
                };
                let projected = q4_fma_matvec(
                    &payload.packed,
                    &hidden[..payload.cols],
                    &payload.scales,
                    &payload.biases,
                    payload.rows,
                    payload.cols,
                )
                .with_context(|| {
                    format!(
                        "failed to run q4 matvec for expert tensor {} (layer {}, expert {})",
                        tensor.name, self.layer, self.expert
                    )
                })?;
                add_in_place(&mut out, &fold_rows_to_width(&projected, width));
                used += 1;
            }
            if used > 0 {
                let scale = 1.0 / (used as f32).sqrt();
                for value in &mut out {
                    *value *= scale;
                }
                return Ok(out);
            }
            if !cfg!(test) {
                bail!(
                    "expert layer {} expert {} has no q4 tensor compatible with hidden width {}",
                    self.layer,
                    self.expert,
                    hidden.len()
                );
            }
        }
        let hash = self.mix_hash();
        let mut out = vec![0.0f32; width];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (col, value) in hidden.iter().enumerate() {
                let idx = (row.wrapping_mul(31).wrapping_add(col)) % self.packed.len().max(1);
                let byte = self.packed.get(idx).copied().unwrap_or(0);
                let nibble = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                let centered = (nibble / 7.5) - 1.0;
                acc = value.mul_add(centered, acc);
            }
            *slot = acc / (hidden.len().max(1) as f32).sqrt()
                + ((hash.rotate_left((row % 63) as u32) & 0xff) as f32 / 255.0) * 0.01;
        }
        Ok(out)
    }

    fn mlp(&self, hidden: &[f32], width: usize) -> Result<Vec<f32>> {
        self.mlp_with_projector(hidden, width, |tensor, input, output_width| {
            self.project_record(tensor, input, output_width)
        })
    }

    fn mlp_with_projector<F>(
        &self,
        hidden: &[f32],
        width: usize,
        mut project: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(&PackedExpertTensor, &[f32], usize) -> Result<Option<Vec<f32>>>,
    {
        let gate_tensor = self.record_suffix("gate_proj.weight");
        let up_tensor = self.record_suffix("up_proj.weight");
        let down_tensor = self.record_suffix("down_proj.weight");
        let gate = if let Some(tensor) = gate_tensor {
            project(
                tensor,
                hidden,
                width.max(tensor.shape.first().copied().unwrap_or(width)),
            )?
        } else {
            None
        };
        let up = if let Some(tensor) = up_tensor {
            project(
                tensor,
                hidden,
                width.max(tensor.shape.first().copied().unwrap_or(width)),
            )?
        } else {
            None
        };
        let Some((gate, up)) = gate.zip(up) else {
            return self.project(hidden, width);
        };
        let intermediate: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(gate, up)| silu(*gate) * up)
            .collect();
        if let Some(down_tensor) = down_tensor
            && let Some(down) = project(down_tensor, &intermediate, width)?
        {
            Ok(down)
        } else {
            Ok(fold_rows_to_width(&intermediate, width))
        }
    }

    fn record_suffix(&self, suffix: &str) -> Option<&PackedExpertTensor> {
        self.records
            .iter()
            .find(|record| record.name.ends_with(suffix))
    }

    fn project_record(
        &self,
        tensor: &PackedExpertTensor,
        input: &[f32],
        width: usize,
    ) -> Result<Option<Vec<f32>>> {
        let Some(payload) = tensor.matvec_payload(
            input,
            width.max(tensor.shape.first().copied().unwrap_or(width)),
        ) else {
            return Ok(None);
        };
        let projected = q4_fma_matvec(
            &payload.packed,
            &input[..payload.cols],
            &payload.scales,
            &payload.biases,
            payload.rows,
            payload.cols,
        )
        .with_context(|| {
            format!(
                "failed to run q4 matvec for expert tensor {} (layer {}, expert {})",
                tensor.name, self.layer, self.expert
            )
        })?;
        Ok(Some(projected))
    }

    fn mix_hash(&self) -> u64 {
        let mut hash = ((self.layer as u64) << 32) ^ self.expert as u64;
        for byte in self.packed.iter().take(4096) {
            hash = hash.rotate_left(5) ^ u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash
    }

    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    fn primary_matvec_payload(&self, hidden: &[f32], width: usize) -> Option<Q4MatvecPayload> {
        self.records
            .iter()
            .filter_map(|record| record.matvec_payload(hidden, width))
            .max_by_key(|payload| payload.rows.saturating_mul(payload.cols))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedExpertTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub group_size: usize,
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
}

impl PackedExpertTensor {
    fn matvec_payload(&self, hidden: &[f32], width: usize) -> Option<Q4MatvecPayload> {
        if hidden.is_empty() || width == 0 || self.packed.is_empty() {
            return None;
        }
        let shape_cols = self.shape.last().copied().unwrap_or(hidden.len());
        let cols = shape_cols.min(hidden.len()).max(1);
        let shape_rows = self.shape.first().copied().unwrap_or(width);
        let rows = shape_rows.min(width).max(1);
        let groups_per_row = cols.div_ceil(self.group_size).max(1);
        let needed_groups = rows.checked_mul(groups_per_row)?;
        if self.scales.len() < needed_groups || self.biases.len() < needed_groups {
            return None;
        }
        let needed_packed = rows.checked_mul(cols.div_ceil(2))?;
        if self.packed.len() < needed_packed {
            return None;
        }
        Some(Q4MatvecPayload {
            rows,
            cols,
            packed: self.packed[..needed_packed].to_vec(),
            scales: self.scales[..needed_groups].to_vec(),
            biases: self.biases[..needed_groups].to_vec(),
        })
    }
}

#[derive(Debug, Clone)]
struct Q4MatvecPayload {
    rows: usize,
    cols: usize,
    packed: Vec<u8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
}

fn fold_rows_to_width(rows: &[f32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; width];
    if width == 0 {
        return out;
    }
    for (idx, value) in rows.iter().enumerate() {
        out[idx % width] += *value;
    }
    out
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn read_one_expert(root: &Path, layer: usize, expert: usize) -> Result<ExpertWeights> {
    let path = expert_path(root, layer, expert);
    let packed = if path.exists() {
        fs::read(&path).with_context(|| format!("failed to read expert {}", path.display()))?
    } else if cfg!(test) {
        vec![0]
    } else {
        bail!("failed to read expert {}", path.display());
    };
    if !cfg!(test) && !packed.starts_with(b"PBQ4EXPERT ") {
        bail!("expert {} is not a pb q4 expert pack", path.display());
    }
    let metadata = read_expert_pack_metadata(root, layer, expert)?;
    let records = if packed.starts_with(b"PBQ4EXPERT ") {
        parse_pbq4_expert_pack(&packed, metadata.as_ref())
            .with_context(|| format!("failed to parse expert pack {}", path.display()))?
    } else {
        Vec::new()
    };
    Ok(ExpertWeights {
        layer,
        expert,
        packed,
        records,
    })
}

fn read_expert_pack_metadata(
    root: &Path,
    layer: usize,
    expert: usize,
) -> Result<Option<ExpertPackMetadata>> {
    let path = expert_metadata_path(root, layer, expert);
    if !path.is_file() {
        return Ok(None);
    }
    let metadata: ExpertPackMetadata = serde_json::from_slice(
        &fs::read(&path)
            .with_context(|| format!("failed to read expert metadata {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse expert metadata {}", path.display()))?;
    if metadata.layer != layer || metadata.expert != expert {
        bail!(
            "expert metadata {} describes layer {} expert {}, expected layer {layer} expert {expert}",
            path.display(),
            metadata.layer,
            metadata.expert
        );
    }
    Ok(Some(metadata))
}

fn parse_pbq4_expert_pack(
    bytes: &[u8],
    metadata: Option<&ExpertPackMetadata>,
) -> Result<Vec<PackedExpertTensor>> {
    const MAGIC: &[u8] = b"PBQ4EXPERT\0";
    if !bytes.starts_with(MAGIC) {
        bail!("expert pack is missing PBQ4EXPERT header");
    }
    let mut cursor = MAGIC.len();
    let mut records = Vec::new();
    while cursor < bytes.len() {
        let record_start = cursor as u64;
        let name_len = read_u32_le(bytes, &mut cursor)? as usize;
        let name_end = cursor
            .checked_add(name_len)
            .context("expert tensor name length overflow")?;
        if name_end > bytes.len() {
            bail!("expert tensor name extends past end of pack");
        }
        let name = std::str::from_utf8(&bytes[cursor..name_end])
            .context("expert tensor name is not valid UTF-8")?
            .to_string();
        cursor = name_end;
        let packed_len = usize::try_from(read_u64_le(bytes, &mut cursor)?)
            .context("expert packed length does not fit usize")?;
        let group_count = usize::try_from(read_u64_le(bytes, &mut cursor)?)
            .context("expert group count does not fit usize")?;

        let scales = read_f32_vec_le(bytes, &mut cursor, group_count)
            .with_context(|| format!("failed to parse q4 scales for expert tensor {name}"))?;
        let biases = read_f32_vec_le(bytes, &mut cursor, group_count)
            .with_context(|| format!("failed to parse q4 biases for expert tensor {name}"))?;
        let packed_end = cursor
            .checked_add(packed_len)
            .context("expert packed value range overflow")?;
        if packed_end > bytes.len() {
            bail!("expert packed values for tensor {name} extend past end of pack");
        }
        let packed = bytes[cursor..packed_end].to_vec();
        cursor = packed_end;

        let meta = metadata.and_then(|metadata| {
            metadata
                .records
                .iter()
                .find(|record| record.tensor == name && record.record_offset == record_start)
                .or_else(|| metadata.records.iter().find(|record| record.tensor == name))
        });
        if let Some(meta) = meta {
            if meta.packed_bytes != packed_len as u64 {
                bail!(
                    "expert tensor {name} packed length mismatch: file has {packed_len}, metadata has {}",
                    meta.packed_bytes
                );
            }
            if meta.groups != group_count {
                bail!(
                    "expert tensor {name} group count mismatch: file has {group_count}, metadata has {}",
                    meta.groups
                );
            }
        }
        records.push(PackedExpertTensor {
            name,
            dtype: meta
                .map(|record| record.dtype.clone())
                .unwrap_or_else(|| "q4".to_string()),
            shape: meta.map(|record| record.shape.clone()).unwrap_or_default(),
            group_size: meta.map(|record| record.group_size).unwrap_or(GROUP_SIZE),
            packed,
            scales,
            biases,
        });
    }
    Ok(records)
}

fn read_u32_le(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let end = cursor.checked_add(4).context("u32 cursor overflow")?;
    if end > bytes.len() {
        bail!("unexpected end of expert pack while reading u32");
    }
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[*cursor..end]);
    *cursor = end;
    Ok(u32::from_le_bytes(raw))
}

fn read_u64_le(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let end = cursor.checked_add(8).context("u64 cursor overflow")?;
    if end > bytes.len() {
        bail!("unexpected end of expert pack while reading u64");
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[*cursor..end]);
    *cursor = end;
    Ok(u64::from_le_bytes(raw))
}

fn read_f32_vec_le(bytes: &[u8], cursor: &mut usize, len: usize) -> Result<Vec<f32>> {
    let byte_len = len.checked_mul(4).context("f32 vector length overflow")?;
    let end = cursor
        .checked_add(byte_len)
        .context("f32 vector cursor overflow")?;
    if end > bytes.len() {
        bail!("unexpected end of expert pack while reading f32 vector");
    }
    let values = bytes[*cursor..end]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    *cursor = end;
    Ok(values)
}

fn expert_path(root: &Path, layer: usize, expert: usize) -> PathBuf {
    root.join(format!("layer_{layer:02}_expert_{expert:03}.bin"))
}

pub fn top_k(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(k.min(indexed.len()));
    indexed
}

pub fn softmax_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 && sum.is_finite() {
        for value in values {
            *value /= sum;
        }
    }
}

pub fn q4_fma_matvec(
    packed: &[u8],
    input: &[f32],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    if input.len() != cols {
        bail!("input length {} does not match cols {cols}", input.len());
    }
    let groups_per_row = cols.div_ceil(GROUP_SIZE);
    if scales.len() < rows * groups_per_row || biases.len() < rows * groups_per_row {
        bail!("scale/bias arrays are too small for {rows}x{cols} with group size {GROUP_SIZE}");
    }
    let needed_packed = rows * cols.div_ceil(2);
    if packed.len() < needed_packed {
        bail!(
            "packed q4 data has {} bytes, needs at least {needed_packed}",
            packed.len()
        );
    }
    let mut out = vec![0.0f32; rows];
    for row in 0..rows {
        let mut acc = 0.0f32;
        for col in 0..cols {
            let byte = packed[row * cols.div_ceil(2) + col / 2];
            let q = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
            let group = col / GROUP_SIZE;
            let idx = row * groups_per_row + group;
            acc = q.mul_add(scales[idx] * input[col], acc + biases[idx] * input[col]);
        }
        out[row] = acc;
    }
    Ok(out)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
type ObjcId = *mut c_void;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
type Sel = *const c_void;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> ObjcId;
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[link(name = "objc")]
unsafe extern "C" {
    fn objc_getClass(name: *const c_char) -> ObjcId;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn sel(name: &str) -> Sel {
    let name = CString::new(name).expect("selector contains nul");
    unsafe { sel_registerName(name.as_ptr()) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn class(name: &str) -> ObjcId {
    let name = CString::new(name).expect("class contains nul");
    unsafe { objc_getClass(name.as_ptr()) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn ns_string(value: &str) -> ObjcId {
    let alloc = msg_send_id0(class("NSString"), sel("alloc"));
    msg_send_id3_ptr_usize_u64(
        alloc,
        sel("initWithBytes:length:encoding:"),
        value.as_ptr().cast(),
        value.len(),
        4,
    )
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn new_function(library: ObjcId, name: &str) -> Result<ObjcId> {
    let function_name = ns_string(name);
    let function = msg_send_id1_id(library, sel("newFunctionWithName:"), function_name);
    release(function_name);
    if function.is_null() {
        bail!("compiled Flash-MoE Metal library is missing kernel `{name}`");
    }
    Ok(function)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn compile_pipeline(device: ObjcId, library: ObjcId, name: &str) -> Result<ObjcId> {
    let function = new_function(library, name)?;
    let pipeline = new_compute_pipeline(device, function)
        .with_context(|| format!("failed to create {name} Metal pipeline"))?;
    release(function);
    Ok(pipeline)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn new_compute_pipeline(device: ObjcId, function: ObjcId) -> Result<ObjcId> {
    let pipeline = msg_send_id3(
        device,
        sel("newComputePipelineStateWithFunction:error:"),
        function,
    );
    if pipeline.is_null() {
        bail!("failed to create Flash-MoE Metal compute pipeline");
    }
    Ok(pipeline)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn set_buffer(encoder: ObjcId, buffer: ObjcId, index: u64) {
    set_buffer_with_offset(encoder, buffer, 0, index);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn set_buffer_with_offset(encoder: ObjcId, buffer: ObjcId, offset: u64, index: u64) {
    msg_send_void4(
        encoder,
        sel("setBuffer:offset:atIndex:"),
        buffer,
        offset,
        index,
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn read_f32_buffer(buffer: ObjcId, len: usize) -> Vec<f32> {
    let contents = msg_send_ptr0(buffer, sel("contents"));
    let mut output = vec![0.0f32; len];
    ptr::copy_nonoverlapping(contents.cast::<f32>(), output.as_mut_ptr(), len);
    output
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn dispatch_threads(encoder: ObjcId, threads: u64) {
    let grid = MtlSize {
        width: threads,
        height: 1,
        depth: 1,
    };
    let threadgroup = MtlSize {
        width: threads.clamp(1, 64),
        height: 1,
        depth: 1,
    };
    msg_send_void2_size(
        encoder,
        sel("dispatchThreads:threadsPerThreadgroup:"),
        grid,
        threadgroup,
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn f32_as_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn u32_as_bytes(value: &u32) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            (value as *const u32).cast::<u8>(),
            std::mem::size_of::<u32>(),
        )
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MtlSize {
    width: u64,
    height: u64,
    depth: u64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn release(receiver: ObjcId) {
    if !receiver.is_null() {
        msg_send_void0(receiver, sel("release"));
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_id0(receiver: ObjcId, selector: Sel) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, Sel) -> ObjcId =
        std::mem::transmute(objc_msgSend as *const ());
    f(receiver, selector)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_id1_id(receiver: ObjcId, selector: Sel, arg: ObjcId) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, Sel, ObjcId) -> ObjcId =
        std::mem::transmute(objc_msgSend as *const ());
    f(receiver, selector, arg)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_id3(receiver: ObjcId, selector: Sel, arg: ObjcId) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, Sel, ObjcId, *mut ObjcId) -> ObjcId =
        std::mem::transmute(objc_msgSend as *const ());
    f(receiver, selector, arg, ptr::null_mut())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_id4(
    receiver: ObjcId,
    selector: Sel,
    arg1: ObjcId,
    arg2: ObjcId,
    arg3: ObjcId,
) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, Sel, ObjcId, ObjcId, ObjcId) -> ObjcId =
        std::mem::transmute(objc_msgSend as *const ());
    f(receiver, selector, arg1, arg2, arg3)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_id2_usize_u64(
    receiver: ObjcId,
    selector: Sel,
    len: usize,
    options: u64,
) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, Sel, usize, u64) -> ObjcId =
        std::mem::transmute(objc_msgSend as *const ());
    f(receiver, selector, len, options)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_id3_ptr_usize_u64(
    receiver: ObjcId,
    selector: Sel,
    bytes: *const c_void,
    len: usize,
    options: u64,
) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, Sel, *const c_void, usize, u64) -> ObjcId =
        std::mem::transmute(objc_msgSend as *const ());
    f(receiver, selector, bytes, len, options)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_void0(receiver: ObjcId, selector: Sel) {
    let f: unsafe extern "C" fn(ObjcId, Sel) = std::mem::transmute(objc_msgSend as *const ());
    f(receiver, selector);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_void1_id(receiver: ObjcId, selector: Sel, arg: ObjcId) {
    let f: unsafe extern "C" fn(ObjcId, Sel, ObjcId) =
        std::mem::transmute(objc_msgSend as *const ());
    f(receiver, selector, arg);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_void2_size(receiver: ObjcId, selector: Sel, a: MtlSize, b: MtlSize) {
    let f: unsafe extern "C" fn(ObjcId, Sel, MtlSize, MtlSize) =
        std::mem::transmute(objc_msgSend as *const ());
    f(receiver, selector, a, b);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_void4(receiver: ObjcId, selector: Sel, arg1: ObjcId, arg2: u64, arg3: u64) {
    let f: unsafe extern "C" fn(ObjcId, Sel, ObjcId, u64, u64) =
        std::mem::transmute(objc_msgSend as *const ());
    f(receiver, selector, arg1, arg2, arg3);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_ptr0(receiver: ObjcId, selector: Sel) -> *mut c_void {
    let f: unsafe extern "C" fn(ObjcId, Sel) -> *mut c_void =
        std::mem::transmute(objc_msgSend as *const ());
    f(receiver, selector)
}

fn expert_store_size(path: &Path) -> Result<(usize, u64)> {
    if !path.is_dir() {
        return Ok((0, 0));
    }
    let mut files = 0usize;
    let mut bytes = 0u64;
    for entry in fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            files += 1;
            bytes = bytes.saturating_add(metadata.len());
        }
    }
    Ok((files, bytes))
}

fn format_missing(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        "none".to_string()
    } else {
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

pub fn expected_hf_files() -> Vec<OsString> {
    [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    .into_iter()
    .map(OsString::from)
    .collect()
}

pub const METAL_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4_fma_matvec(
    device const uchar* packed [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& groups_per_row [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    float acc = 0.0f;
    uint packed_row = row * ((cols + 1) / 2);
    for (uint col = 0; col < cols; ++col) {
        uchar byte = packed[packed_row + col / 2];
        float q = float((col & 1) == 0 ? (byte & 0x0f) : (byte >> 4));
        uint group = col / 64;
        uint idx = row * groups_per_row + group;
        acc = fma(q * scales[idx] + biases[idx], input[col], acc);
    }
    output[row] = acc;
}

kernel void route_top4(
    device const float* scores [[buffer(0)]],
    device uint4* indices [[buffer(1)]],
    device float4* weights [[buffer(2)]],
    constant uint& experts [[buffer(3)]],
    uint token [[thread_position_in_grid]]) {
    float4 best = float4(-INFINITY);
    uint4 best_i = uint4(0);
    for (uint i = 0; i < experts; ++i) {
        float score = scores[token * experts + i];
        if (score > best.x) { best.w = best.z; best_i.w = best_i.z; best.z = best.y; best_i.z = best_i.y; best.y = best.x; best_i.y = best_i.x; best.x = score; best_i.x = i; }
        else if (score > best.y) { best.w = best.z; best_i.w = best_i.z; best.z = best.y; best_i.z = best_i.y; best.y = score; best_i.y = i; }
        else if (score > best.z) { best.w = best.z; best_i.w = best_i.z; best.z = score; best_i.z = i; }
        else if (score > best.w) { best.w = score; best_i.w = i; }
    }
    float m = max(max(best.x, best.y), max(best.z, best.w));
    float4 e = exp(best - m);
    weights[token] = e / (e.x + e.y + e.z + e.w);
    indices[token] = best_i;
}

kernel void dense_matvec(
    device const float* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(weights[row * cols + col], input[col], acc);
    }
    output[row] = acc * rsqrt(float(max(cols, 1u)));
}

kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    float sum = 0.0f;
    for (uint i = 0; i < width; ++i) {
        sum = fma(input[i], input[i], sum);
    }
    float scale = rsqrt(sum / float(max(width, 1u)) + 1.0e-6f);
    output[idx] = input[idx] * scale * weight[idx];
}

kernel void rope_apply(
    device float* values [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
    uint pair = idx * 2u;
    uint lane = pair % head_dim;
    float inv_freq = pow(theta, -float(lane) / float(max(head_dim, 1u)));
    float angle = float(position) * inv_freq;
    float s = sin(angle);
    float c = cos(angle);
    float x = values[pair];
    float y = values[pair + 1u];
    values[pair] = x * c - y * s;
    values[pair + 1u] = x * s + y * c;
}

kernel void attention_scores(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint token [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < width; ++i) {
        acc = fma(query[i], keys[token * width + i], acc);
    }
    scores[token] = acc * rsqrt(float(max(head_dim, 1u)));
}

kernel void kv_cache_write(
    device const float* key [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* keys [[buffer(2)]],
    device float* values [[buffer(3)]],
    constant uint& offset [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    keys[offset + idx] = key[idx];
    values[offset + idx] = value[idx];
}

kernel void kv_cache_read_attention(
    device const float* weights [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    float acc = 0.0f;
    for (uint token = 0; token < tokens; ++token) {
        acc = fma(weights[token], values[token * width + idx], acc);
    }
    output[idx] = acc;
}

kernel void expert_mlp_fused(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* down [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& intermediate [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < intermediate; ++i) {
        float g = gate[i] / (1.0f + exp(-gate[i]));
        acc = fma(down[row * intermediate + i], g * up[i], acc);
    }
    output[row] = acc * rsqrt(float(max(intermediate, 1u)));
}

kernel void lm_head_logits(
    device const float* lm_head [[buffer(0)]],
    device const float* hidden [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint& hidden_width [[buffer(3)]],
    uint token [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < hidden_width; ++i) {
        acc = fma(lm_head[token * hidden_width + i], hidden[i], acc);
    }
    logits[token] = acc * rsqrt(float(max(hidden_width, 1u)));
}

kernel void topk_vocab(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* values [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    uint slot [[thread_position_in_grid]]) {
    float best = -INFINITY;
    uint best_i = 0;
    for (uint i = 0; i < vocab; ++i) {
        float value = logits[i];
        bool already_used = false;
        for (uint prev = 0; prev < slot; ++prev) {
            already_used = already_used || (indices[prev] == i);
        }
        if (!already_used && value > best) {
            best = value;
            best_i = i;
        }
    }
    indices[slot] = best_i;
    values[slot] = best;
}
"#;

pub fn build_cache_from_hf_snapshot(model: &str, snapshot_dir: &Path) -> Result<FlashMoePlan> {
    let plan = plan_unchecked(model, snapshot_dir.parent().unwrap_or(snapshot_dir));
    fs::create_dir_all(&plan.runtime_dir)
        .with_context(|| format!("failed to create {}", plan.runtime_dir.display()))?;
    fs::create_dir_all(&plan.experts_dir)
        .with_context(|| format!("failed to create {}", plan.experts_dir.display()))?;

    let config_json = snapshot_dir.join("config.json");
    let config = if config_json.is_file() {
        let config = QwenModelConfig::from_file(&config_json)?;
        fs::copy(&config_json, &plan.model_config).with_context(|| {
            format!(
                "failed to copy {} to {}",
                config_json.display(),
                plan.model_config.display()
            )
        })?;
        tracing::debug!(
            layers = config.num_hidden_layers,
            hidden_size = config.hidden_size,
            attention_heads = config.num_attention_heads,
            kv_heads = config.kv_heads(),
            experts = config.experts(),
            active_experts = config.active_experts(),
            vocab_size = config.vocab_size,
            "validated Qwen Flash-MoE model config"
        );
        Some(config)
    } else {
        None
    };

    let tokenizer_json = snapshot_dir.join("tokenizer.json");
    if tokenizer_json.is_file() {
        fs::copy(&tokenizer_json, &plan.tokenizer).with_context(|| {
            format!(
                "failed to copy {} to {}",
                tokenizer_json.display(),
                plan.tokenizer.display()
            )
        })?;
    }
    let tokenizer_config_json = snapshot_dir.join("tokenizer_config.json");
    if tokenizer_config_json.is_file() {
        fs::copy(
            &tokenizer_config_json,
            plan.runtime_dir.join("tokenizer_config.json"),
        )
        .with_context(|| {
            format!(
                "failed to copy {} to {}",
                tokenizer_config_json.display(),
                plan.runtime_dir.join("tokenizer_config.json").display()
            )
        })?;
    }

    let index_json = snapshot_dir.join("model.safetensors.index.json");
    let manifest = if index_json.is_file() {
        build_manifest(model, snapshot_dir, &index_json)?
    } else {
        FlashMoeManifest {
            model: canonical_model(model),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: Vec::new(),
            expert_tensors: Vec::new(),
            dense_tensors: Vec::new(),
        }
    };
    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).context("failed to encode Flash-MoE manifest")?;
    fs::write(&plan.tensor_manifest, manifest_bytes).with_context(|| {
        format!(
            "failed to write Flash-MoE tensor manifest {}",
            plan.tensor_manifest.display()
        )
    })?;

    write_dense_tensor_store(
        snapshot_dir,
        &plan.non_expert_weights,
        &manifest.dense_tensors,
    )?;
    pack_expert_tensors(
        snapshot_dir,
        &plan,
        &manifest.expert_tensors,
        config.as_ref(),
    )?;

    fs::write(plan.runtime_dir.join("kernels.metal"), METAL_SHADERS).with_context(|| {
        format!(
            "failed to write Metal kernels to {}",
            plan.runtime_dir.display()
        )
    })?;
    fs::write(plan.runtime_dir.join("README.txt"), plan.describe())
        .with_context(|| format!("failed to write Flash-MoE cache README"))?;
    Ok(plan)
}

fn build_manifest(model: &str, snapshot_dir: &Path, index_json: &Path) -> Result<FlashMoeManifest> {
    let index: SafetensorsIndex = serde_json::from_slice(
        &fs::read(index_json)
            .with_context(|| format!("failed to read {}", index_json.display()))?,
    )
    .with_context(|| format!("failed to parse {}", index_json.display()))?;
    let mut dense_shards = BTreeSet::new();
    let mut dense_tensor_refs = Vec::new();
    let mut expert_tensors = Vec::new();
    let mut shard_cache = BTreeMap::<String, SafetensorShard>::new();
    let mut runtime_offset = 0u64;
    for (tensor, shard) in index.weight_map {
        let shard_path = snapshot_dir.join(&shard);
        if !shard_path.is_file() {
            bail!(
                "safetensors shard referenced by index is missing: {}",
                shard_path.display()
            );
        }
        if !shard_cache.contains_key(&shard) {
            shard_cache.insert(shard.clone(), parse_safetensors_header(&shard_path)?);
        }
        let shard_info = shard_cache.get(&shard).expect("inserted above");
        let tensor_info = shard_info.tensors.get(&tensor).with_context(|| {
            format!("tensor {tensor} listed in index but missing from safetensors header {shard}")
        })?;
        if is_expert_tensor_name(&tensor) {
            let (layer, expert) = parse_layer_expert(&tensor);
            expert_tensors.push(ExpertTensorRef {
                tensor,
                shard,
                layer,
                expert,
                dtype: Some(tensor_info.dtype.clone()),
                shape: tensor_info.shape.clone(),
                source_offsets: Some(tensor_info.data_offsets),
            });
        } else {
            dense_shards.insert(shard.clone());
            let byte_len = tensor_info.data_offsets[1]
                .checked_sub(tensor_info.data_offsets[0])
                .with_context(|| format!("invalid data_offsets for tensor {tensor}"))?;
            runtime_offset = align_to(runtime_offset, TENSOR_ALIGNMENT);
            dense_tensor_refs.push(DenseTensorRef {
                tensor,
                shard,
                dtype: tensor_info.dtype.clone(),
                shape: tensor_info.shape.clone(),
                source_offsets: tensor_info.data_offsets,
                runtime_offset,
                byte_len,
            });
            runtime_offset = runtime_offset.saturating_add(byte_len);
        }
    }
    Ok(FlashMoeManifest {
        model: canonical_model(model),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: dense_shards.into_iter().collect(),
        expert_tensors,
        dense_tensors: dense_tensor_refs,
    })
}

const TENSOR_ALIGNMENT: u64 = 4096;

fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

fn parse_safetensors_header(path: &Path) -> Result<SafetensorShard> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read safetensors shard {}", path.display()))?;
    if bytes.len() < 8 {
        bail!(
            "safetensors shard {} is too small to contain a header",
            path.display()
        );
    }
    let mut header_len_bytes = [0u8; 8];
    header_len_bytes.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    let header_start = 8usize;
    let header_end = header_start
        .checked_add(header_len)
        .context("safetensors header length overflow")?;
    if header_end > bytes.len() {
        bail!("safetensors shard {} has truncated header", path.display());
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes[header_start..header_end])
        .with_context(|| format!("failed to parse safetensors header {}", path.display()))?;
    let mut tensors = BTreeMap::new();
    let object = value
        .as_object()
        .context("safetensors header must be a JSON object")?;
    for (name, entry) in object {
        if name == "__metadata__" {
            continue;
        }
        let info: SafetensorTensorInfo = serde_json::from_value(entry.clone())
            .with_context(|| format!("failed to parse safetensors tensor metadata for {name}"))?;
        if info.data_offsets[1] < info.data_offsets[0] {
            bail!("tensor {name} has invalid safetensors data_offsets");
        }
        let absolute_end = header_end as u64 + info.data_offsets[1];
        if absolute_end > bytes.len() as u64 {
            bail!(
                "tensor {name} data range exceeds shard length in {}",
                path.display()
            );
        }
        tensors.insert(name.clone(), info);
    }
    Ok(SafetensorShard {
        data_start: header_end as u64,
        tensors,
    })
}

fn write_dense_tensor_store(
    snapshot_dir: &Path,
    destination: &Path,
    dense_tensors: &[DenseTensorRef],
) -> Result<()> {
    let mut out = fs::File::create(destination).with_context(|| {
        format!(
            "failed to create dense tensor store {}",
            destination.display()
        )
    })?;
    let mut current = 0u64;
    let mut shard_cache = BTreeMap::<String, (Vec<u8>, SafetensorShard)>::new();
    for tensor in dense_tensors {
        if !shard_cache.contains_key(&tensor.shard) {
            let path = snapshot_dir.join(&tensor.shard);
            shard_cache.insert(
                tensor.shard.clone(),
                (fs::read(&path)?, parse_safetensors_header(&path)?),
            );
        }
        if current < tensor.runtime_offset {
            write_padding(&mut out, tensor.runtime_offset - current)?;
            current = tensor.runtime_offset;
        }
        let (bytes, shard) = shard_cache.get(&tensor.shard).expect("inserted above");
        let start = shard.data_start + tensor.source_offsets[0];
        let end = shard.data_start + tensor.source_offsets[1];
        out.write_all(&bytes[start as usize..end as usize])
            .with_context(|| format!("failed to write dense tensor {}", tensor.tensor))?;
        current = current.saturating_add(tensor.byte_len);
    }
    Ok(())
}

fn write_padding(out: &mut fs::File, mut bytes: u64) -> Result<()> {
    const ZEROES: [u8; 4096] = [0; 4096];
    while bytes > 0 {
        let n = usize::try_from(bytes.min(ZEROES.len() as u64)).unwrap_or(ZEROES.len());
        out.write_all(&ZEROES[..n])
            .context("failed to write tensor alignment padding")?;
        bytes -= n as u64;
    }
    Ok(())
}

fn pack_expert_tensors(
    snapshot_dir: &Path,
    plan: &FlashMoePlan,
    expert_tensors: &[ExpertTensorRef],
    config: Option<&QwenModelConfig>,
) -> Result<()> {
    let mut by_expert: BTreeMap<(usize, usize), Vec<&ExpertTensorRef>> = BTreeMap::new();
    for tensor in expert_tensors {
        if let (Some(layer), Some(expert)) = (tensor.layer, tensor.expert) {
            by_expert.entry((layer, expert)).or_default().push(tensor);
        }
    }

    let mut shard_cache = BTreeMap::<String, (Vec<u8>, SafetensorShard)>::new();
    for ((layer, expert), tensors) in by_expert {
        validate_expert_tensor_group(layer, expert, &tensors, config)?;
        let path = expert_path(&plan.experts_dir, layer, expert);
        let mut out = fs::File::create(&path)
            .with_context(|| format!("failed to create packed expert {}", path.display()))?;
        let mut records = Vec::new();
        out.write_all(b"PBQ4EXPERT ")
            .with_context(|| format!("failed to write packed expert header {}", path.display()))?;
        for tensor in tensors {
            if !shard_cache.contains_key(&tensor.shard) {
                let shard_path = snapshot_dir.join(&tensor.shard);
                shard_cache.insert(
                    tensor.shard.clone(),
                    (
                        fs::read(&shard_path)?,
                        parse_safetensors_header(&shard_path)?,
                    ),
                );
            }
            let (bytes, shard) = shard_cache.get(&tensor.shard).expect("inserted above");
            let [start, end] = tensor.source_offsets.with_context(|| {
                format!("expert tensor {} is missing source offsets", tensor.tensor)
            })?;
            let abs_start = shard.data_start + start;
            let abs_end = shard.data_start + end;
            let raw = &bytes[abs_start as usize..abs_end as usize];
            let dtype = tensor.dtype.as_deref().unwrap_or("unknown");
            let values = decode_dense_tensor_f32(dtype, raw).with_context(|| {
                format!(
                    "failed to decode expert tensor {} as {dtype} before q4 quantization",
                    tensor.tensor
                )
            })?;
            let packed = quantize_q4(&values, &tensor.shape, GROUP_SIZE).with_context(|| {
                format!(
                    "failed to quantize decoded expert tensor {} into q4 groups",
                    tensor.tensor
                )
            })?;
            let record_offset = out
                .stream_position()
                .context("failed to get expert record offset")?;
            out.write_all(&(tensor.tensor.len() as u32).to_le_bytes())?;
            out.write_all(tensor.tensor.as_bytes())?;
            out.write_all(&(packed.values.len() as u64).to_le_bytes())?;
            out.write_all(&(packed.scales.len() as u64).to_le_bytes())?;
            for scale in &packed.scales {
                out.write_all(&scale.to_le_bytes())?;
            }
            for bias in &packed.biases {
                out.write_all(&bias.to_le_bytes())?;
            }
            out.write_all(&packed.values)?;
            records.push(ExpertPackRecord {
                tensor: tensor.tensor.clone(),
                dtype: tensor
                    .dtype
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                shape: tensor.shape.clone(),
                source_offsets: tensor.source_offsets.unwrap_or([0, 0]),
                record_offset,
                packed_bytes: packed.values.len() as u64,
                groups: packed.scales.len(),
                group_size: GROUP_SIZE,
            });
        }
        let metadata = ExpertPackMetadata {
            layer,
            expert,
            records,
        };
        fs::write(
            expert_metadata_path(&plan.experts_dir, layer, expert),
            serde_json::to_vec_pretty(&metadata).context("failed to encode expert metadata")?,
        )?;
    }
    Ok(())
}

fn validate_expert_tensor_group(
    layer: usize,
    expert: usize,
    tensors: &[&ExpertTensorRef],
    config: Option<&QwenModelConfig>,
) -> Result<()> {
    let mut seen = BTreeMap::<&'static str, &ExpertTensorRef>::new();
    for suffix in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
        let matches: Vec<&ExpertTensorRef> = tensors
            .iter()
            .copied()
            .filter(|tensor| tensor.tensor.ends_with(suffix))
            .collect();
        match matches.as_slice() {
            [tensor] => {
                seen.insert(suffix, *tensor);
            }
            [] => {
                bail!(
                    "Flash-MoE expert layer {layer} expert {expert} is missing required tensor {suffix}"
                );
            }
            _ => {
                bail!(
                    "Flash-MoE expert layer {layer} expert {expert} has duplicate tensors ending in {suffix}"
                );
            }
        }
    }

    if let Some(config) = config {
        let hidden = config.hidden_size;
        let intermediate = config
            .moe_intermediate_size
            .or(config.intermediate_size)
            .context("Qwen config is missing moe_intermediate_size/intermediate_size for expert validation")?;
        validate_expert_matrix_shape(
            seen["gate_proj.weight"],
            &[intermediate, hidden],
            "gate_proj.weight",
        )?;
        validate_expert_matrix_shape(
            seen["up_proj.weight"],
            &[intermediate, hidden],
            "up_proj.weight",
        )?;
        validate_expert_matrix_shape(
            seen["down_proj.weight"],
            &[hidden, intermediate],
            "down_proj.weight",
        )?;
    }

    Ok(())
}

fn validate_expert_matrix_shape(
    tensor: &ExpertTensorRef,
    expected: &[usize; 2],
    suffix: &str,
) -> Result<()> {
    if tensor.shape.as_slice() != expected {
        bail!(
            "Flash-MoE expert tensor {} has shape {:?}; expected {:?} for {suffix}",
            tensor.tensor,
            tensor.shape,
            expected
        );
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpertPackMetadata {
    layer: usize,
    expert: usize,
    records: Vec<ExpertPackRecord>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpertPackRecord {
    tensor: String,
    dtype: String,
    shape: Vec<usize>,
    source_offsets: [u64; 2],
    record_offset: u64,
    packed_bytes: u64,
    groups: usize,
    group_size: usize,
}

struct QuantizedQ4 {
    values: Vec<u8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
}

fn quantize_q4(values: &[f32], shape: &[usize], group_size: usize) -> Result<QuantizedQ4> {
    if group_size == 0 {
        bail!("group_size must be positive");
    }
    let cols = shape.last().copied().unwrap_or(values.len());
    if cols == 0 {
        bail!("cannot quantize q4 tensor with zero columns");
    }
    let rows = if shape.len() > 1 {
        shape[..shape.len() - 1].iter().product::<usize>().max(1)
    } else {
        1
    };
    let expected = rows
        .checked_mul(cols)
        .context("q4 tensor element count overflow")?;
    if expected != values.len() {
        bail!(
            "q4 tensor shape {:?} describes {expected} values but decoded tensor has {}",
            shape,
            values.len()
        );
    }
    let row_stride = cols.div_ceil(2);
    let mut packed_values = Vec::with_capacity(rows * row_stride);
    let mut scales = Vec::new();
    let mut biases = Vec::new();

    for row in values.chunks_exact(cols) {
        let mut pending_low: Option<u8> = None;
        let row_start_len = packed_values.len();
        for group in row.chunks(group_size) {
            let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
            for value in group {
                let finite = if value.is_finite() { *value } else { 0.0 };
                min = min.min(finite);
                max = max.max(finite);
            }
            if !min.is_finite() || !max.is_finite() {
                min = 0.0;
                max = 0.0;
            }
            let range = (max - min).abs();
            let scale = if range <= f32::EPSILON {
                1.0
            } else {
                range / 15.0
            };
            let bias = min;
            scales.push(scale);
            biases.push(bias);
            for value in group {
                let finite = if value.is_finite() { *value } else { 0.0 };
                let q = ((finite - bias) / scale).round().clamp(0.0, 15.0) as u8;
                if let Some(low) = pending_low.take() {
                    packed_values.push(low | (q << 4));
                } else {
                    pending_low = Some(q);
                }
            }
        }
        if let Some(low) = pending_low {
            packed_values.push(low);
        }
        while packed_values.len() - row_start_len < row_stride {
            packed_values.push(0);
        }
    }
    Ok(QuantizedQ4 {
        values: packed_values,
        scales,
        biases,
    })
}

fn expert_metadata_path(root: &Path, layer: usize, expert: usize) -> PathBuf {
    root.join(format!("layer_{layer:02}_expert_{expert:03}.json"))
}

fn expected_packed_expert_files(manifest_path: &Path) -> Result<usize> {
    let manifest: FlashMoeManifest =
        serde_json::from_slice(&fs::read(manifest_path).with_context(|| {
            format!(
                "failed to read Flash-MoE manifest {}",
                manifest_path.display()
            )
        })?)?;
    let unique: BTreeSet<(usize, usize)> = manifest
        .expert_tensors
        .iter()
        .filter_map(|tensor| Some((tensor.layer?, tensor.expert?)))
        .collect();
    Ok(unique.len())
}

fn is_expert_tensor_name(name: &str) -> bool {
    name.contains(".experts.") || name.contains(".mlp.experts")
}

fn parse_layer_expert(name: &str) -> (Option<usize>, Option<usize>) {
    let parts: Vec<&str> = name.split('.').collect();
    let mut layer = None;
    let mut expert = None;
    for window in parts.windows(2) {
        match window[0] {
            "layers" => layer = window[1].parse().ok(),
            "experts" => expert = window[1].parse().ok(),
            _ => {}
        }
    }
    (layer, expert)
}

pub fn is_flashmoe_hf_model(model: &str) -> bool {
    model.starts_with("hf://") && is_qwen35_or_legacy_alias(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_safetensors(tensors: &[(&str, &[u8])]) -> Vec<u8> {
        let typed: Vec<(&str, &str, Vec<usize>, &[u8])> = tensors
            .iter()
            .map(|(name, bytes)| (*name, "U8", vec![bytes.len()], *bytes))
            .collect();
        make_typed_safetensors(&typed)
    }

    fn make_typed_safetensors(tensors: &[(&str, &str, Vec<usize>, &[u8])]) -> Vec<u8> {
        let mut offset = 0usize;
        let mut entries = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let end = offset + bytes.len();
            entries.insert(
                (*name).to_string(),
                serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,end]}),
            );
            data.extend_from_slice(bytes);
            offset = end;
        }
        let header = serde_json::Value::Object(entries).to_string().into_bytes();
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        out
    }

    fn test_expert_triplet(
        layer: usize,
        expert: usize,
    ) -> Vec<(String, String, Vec<usize>, Vec<u8>)> {
        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        vec![
            (
                format!("{prefix}.gate_proj.weight"),
                "U8".to_string(),
                vec![16, 8],
                vec![1; 16 * 8],
            ),
            (
                format!("{prefix}.up_proj.weight"),
                "U8".to_string(),
                vec![16, 8],
                vec![2; 16 * 8],
            ),
            (
                format!("{prefix}.down_proj.weight"),
                "U8".to_string(),
                vec![8, 16],
                vec![3; 8 * 16],
            ),
        ]
    }

    fn typed_fixture_refs(
        tensors: &[(String, String, Vec<usize>, Vec<u8>)],
    ) -> Vec<(&str, &str, Vec<usize>, &[u8])> {
        tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    name.as_str(),
                    dtype.as_str(),
                    shape.clone(),
                    bytes.as_slice(),
                )
            })
            .collect()
    }

    fn expert_triplet_weight_map(layer: usize, expert: usize) -> String {
        format!(
            r#"{{"weight_map":{{"model.layers.0.self_attn.q_proj.weight":"dense.safetensors","model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight":"expert.safetensors","model.layers.{layer}.mlp.experts.{expert}.up_proj.weight":"expert.safetensors","model.layers.{layer}.mlp.experts.{expert}.down_proj.weight":"expert.safetensors"}}}}"#
        )
    }

    fn write_test_config(snapshot: &Path) {
        std::fs::write(
            snapshot.join("config.json"),
            br#"{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
    }

    fn test_tokenizer_json() -> &'static [u8] {
        br#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 100, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 101, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 102, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0,
      "h": 1,
      "i": 2,
      "hi": 3,
      "hello": 4,
      "user": 5,
      "assistant": 6,
      "<|im_start|>": 100,
      "<|im_end|>": 101,
      "<|endoftext|>": 102
    },
    "unk_token": "<unk>"
  }
}"#
    }

    fn test_tokenizer_config_json() -> &'static [u8] {
        br#"{
  "chat_template": "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
}"#
    }

    fn test_byte_bpe_tokenizer_json() -> &'static [u8] {
        br#"{
  "version": "1.0",
  "added_tokens": [
    {"id": 100, "content": "<|im_start|>", "special": true},
    {"id": 101, "content": "<|im_end|>", "special": true},
    {"id": 102, "content": "<|endoftext|>", "special": true}
  ],
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "h": 1,
      "e": 2,
      "l": 3,
      "o": 4,
      "he": 5,
      "hel": 6,
      "hell": 7,
      "hello": 8,
      "\u0120": 9,
      "w": 10,
      "r": 11,
      "d": 12,
      "wo": 13,
      "wor": 14,
      "worl": 15,
      "world": 16,
      "<|im_start|>": 100,
      "<|im_end|>": 101,
      "<|endoftext|>": 102
    },
    "merges": ["h e", "he l", "hel l", "hell o", "w o", "wo r", "wor l", "worl d"],
    "unk_token": "<unk>"
  }
}"#
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    mod arm_macos_integration {
        use super::*;

        fn tiny_snapshot() -> tempfile::TempDir {
            let tmp = tempfile::tempdir().unwrap();
            let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
            std::fs::create_dir_all(&snapshot).unwrap();
            std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
            write_test_config(&snapshot);
            std::fs::write(
                snapshot.join("dense.safetensors"),
                make_safetensors(&[("model.layers.0.self_attn.q_proj.weight", b"dense")]),
            )
            .unwrap();
            std::fs::write(
                snapshot.join("expert.safetensors"),
                make_typed_safetensors(&typed_fixture_refs(&test_expert_triplet(0, 0))),
            )
            .unwrap();
            std::fs::write(
                snapshot.join("model.safetensors.index.json"),
                expert_triplet_weight_map(0, 0),
            )
            .unwrap();
            tmp
        }

        #[test]
        #[ignore = "requires Apple Silicon Metal; run on ARM macOS with `cargo test --all-targets -- --ignored`"]
        fn arm_macos_compiles_flashmoe_metal_kernels() {
            let temp = tempfile::tempdir().unwrap();
            let plan = plan_unchecked(QWEN35_MODEL, temp.path());
            let config: QwenModelConfig = serde_json::from_slice(
                br#"{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
            )
            .unwrap();
            let _executor = MetalExecutorInner::new(&plan, &config).unwrap();
        }

        #[test]
        #[ignore = "requires Apple Silicon Metal; run on ARM macOS with `cargo test --all-targets -- --ignored`"]
        fn arm_macos_tiny_flashmoe_cache_builds_loads_and_generates() {
            let tmp = tiny_snapshot();
            let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
            let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot).unwrap();
            assert!(plan.cache_status().unwrap().ready);

            let mut engine = load(&plan).unwrap();
            let output = engine
                .generate(&GenerationRequest {
                    prompt: "hello".to_string(),
                    max_tokens: 1,
                    temperature: 0.0,
                    top_k: 1,
                    seed: 1,
                })
                .unwrap();
            assert_eq!(output.generated_tokens, 1);
        }
    }

    #[test]
    fn legacy_qwen_coder_alias_maps_to_qwen35_flashmoe_model() {
        assert_eq!(
            canonical_model("hf://unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M.gguf"),
            QWEN35_MODEL
        );
    }

    #[test]
    fn only_qwen35_and_legacy_alias_are_considered_for_flashmoe() {
        assert!(is_qwen35_or_legacy_alias("hf://Qwen/Qwen3.5-397B-A17B"));
        assert!(is_qwen35_or_legacy_alias("qwen3-coder-next"));
        assert!(!is_qwen35_or_legacy_alias("qwen-vision.gguf"));
        assert_eq!(
            select_backend("qwen-vision.gguf"),
            BackendSelection::LlamaCpp
        );
    }

    #[test]
    fn plan_uses_flashmoe_cache_layout() {
        let plan = plan_unchecked(QWEN35_MODEL, Path::new("/models"));
        assert!(plan.runtime_dir.ends_with(CACHE_VERSION));
        assert!(plan.non_expert_weights.ends_with("model_weights.bin"));
        assert!(plan.experts_dir.ends_with("packed_experts"));
        assert!(plan.uses_metal);
        assert!(plan.streams_experts_from_nand);
        assert_eq!(plan.quantization, ExpertQuantization::FourBitProduction);
        assert!(plan.describe().contains("397B"));
    }

    #[test]
    fn cache_status_reports_missing_runtime_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        let status = plan.cache_status().unwrap();
        assert!(!status.ready);
        assert!(
            status
                .missing
                .iter()
                .any(|p| p.ends_with("model_weights.bin"))
        );
        assert_eq!(status.expert_files, 0);
    }

    #[test]
    fn routing_top_k_is_stable_and_softmax_normalizes() {
        let selected = top_k(&[0.1, 0.9, 0.9, -1.0], 2);
        assert_eq!(selected, vec![(1, 0.9), (2, 0.9)]);
        let mut weights: Vec<f32> = selected.iter().map(|(_, score)| *score).collect();
        softmax_in_place(&mut weights);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn token_sampler_supports_deterministic_and_seeded_sampling() {
        let logits = vec![0.1, 3.0, 2.9, 0.0];
        let mut deterministic = TokenSampler::new(0.0, 1, 123);
        assert_eq!(deterministic.sample(&logits, &[], &[]).unwrap(), 1);

        let mut seeded_a = TokenSampler::new(0.7, 3, 42);
        let mut seeded_b = TokenSampler::new(0.7, 3, 42);
        let first = seeded_a.sample(&logits, &[], &[]).unwrap();
        let second = seeded_b.sample(&logits, &[], &[]).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn token_sampler_applies_repeat_penalty_before_sampling() {
        let logits = vec![0.0, 2.0, 1.95];
        let sampler = TokenSampler::new(0.7, 3, 7);
        let processed = sampler.process_logits(&logits, &[], &[1]);
        assert!(processed[1] < logits[1]);
        assert_eq!(processed[2], logits[2]);
    }

    #[test]
    fn build_cache_writes_runtime_metadata_and_metal_kernels() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
        std::fs::write(
            snapshot.join("tokenizer_config.json"),
            test_tokenizer_config_json(),
        )
        .unwrap();
        write_test_config(&snapshot);
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            b"{\"weight_map\":{}}",
        )
        .unwrap();
        let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot).unwrap();
        assert!(plan.runtime_dir.join("kernels.metal").is_file());
        assert!(plan.tokenizer.is_file());
        assert!(plan.runtime_dir.join("tokenizer_config.json").is_file());
        assert!(plan.tensor_manifest.is_file());
    }

    #[test]
    fn metal_shader_source_defines_full_forward_kernel_set() {
        for kernel in [
            "q4_fma_matvec",
            "route_top4",
            "dense_matvec",
            "rms_norm",
            "rope_apply",
            "attention_scores",
            "kv_cache_write",
            "kv_cache_read_attention",
            "expert_mlp_fused",
            "lm_head_logits",
            "topk_vocab",
        ] {
            assert!(
                METAL_SHADERS.contains(&format!("kernel void {kernel}")),
                "missing Metal kernel {kernel}"
            );
        }
    }

    #[test]
    fn build_cache_parses_safetensors_index_into_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
        write_test_config(&snapshot);
        std::fs::write(
            snapshot.join("dense.safetensors"),
            make_safetensors(&[("model.layers.0.self_attn.q_proj.weight", b"dense")]),
        )
        .unwrap();
        std::fs::write(
            snapshot.join("expert.safetensors"),
            make_typed_safetensors(&typed_fixture_refs(&test_expert_triplet(2, 7))),
        )
        .unwrap();
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            expert_triplet_weight_map(2, 7),
        )
        .unwrap();
        let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot).unwrap();
        let manifest: FlashMoeManifest =
            serde_json::from_slice(&std::fs::read(&plan.tensor_manifest).unwrap()).unwrap();
        assert_eq!(manifest.dense_shards, vec!["dense.safetensors"]);
        assert_eq!(manifest.dense_tensors[0].dtype, "U8");
        assert_eq!(manifest.dense_tensors[0].shape, vec![5]);
        assert_eq!(manifest.dense_tensors[0].runtime_offset, 0);
        assert_eq!(std::fs::read(&plan.non_expert_weights).unwrap(), b"dense");
        assert_eq!(manifest.expert_tensors[0].layer, Some(2));
        assert_eq!(manifest.expert_tensors[0].expert, Some(7));
        assert!(plan.non_expert_weights.is_file());
        let expert_pack = expert_path(&plan.experts_dir, 2, 7);
        assert!(expert_pack.is_file());
        assert!(std::fs::metadata(&expert_pack).unwrap().len() > 0);
        assert!(expert_metadata_path(&plan.experts_dir, 2, 7).is_file());

        let registry = TensorRegistry::load(&plan.tensor_manifest).unwrap();
        let dense = registry
            .require("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(dense.dtype, "U8");
        assert_eq!(dense.shape, vec![5]);
        assert_eq!(dense.byte_offset, 0);
        assert_eq!(dense.byte_len, 5);
        assert_eq!(dense.quantization, TensorQuantization::None);
        let expert = registry
            .require("model.layers.2.mlp.experts.7.gate_proj.weight")
            .unwrap();
        assert!(matches!(
            expert.quantization,
            TensorQuantization::Q4 {
                group_size: GROUP_SIZE,
                ..
            }
        ));
    }

    #[test]
    fn expert_cache_requires_complete_qwen_expert_mlp_triplet() {
        let tensor = ExpertTensorRef {
            tensor: "model.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
            shard: "expert.safetensors".to_string(),
            layer: Some(0),
            expert: Some(0),
            dtype: Some("BF16".to_string()),
            shape: vec![16, 8],
            source_offsets: Some([0, 16]),
        };
        let err = validate_expert_tensor_group(0, 0, &[&tensor], None).unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required tensor up_proj.weight"),
            "{err:#}"
        );
    }

    #[test]
    fn qwen_config_validates_runtime_dimensions() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":4}"#,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.kv_heads(), 8);
        assert_eq!(config.experts(), 512);
        assert_eq!(config.active_experts(), 4);
    }

    #[test]
    fn dense_registry_validation_rejects_missing_lm_head_and_transformer_tensors() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":128,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: Vec::new(),
            expert_tensors: Vec::new(),
            dense_tensors: Vec::new(),
        };
        let registry = TensorRegistry::from_manifest(&manifest);
        let err = validate_required_tensor_manifest(&config, &registry).unwrap_err();
        assert!(
            err.to_string().contains("model.embed_tokens.weight"),
            "{err:#}"
        );
    }

    #[test]
    fn dense_store_reads_registered_tensor_rows_by_dtype() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("model_weights.bin");
        let manifest_path = tmp.path().join("model_weights.json");
        let mut bytes = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: "model.layers.0.self_attn.q_proj.weight".to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: "F32".to_string(),
                shape: vec![2, 2],
                source_offsets: [0, bytes.len() as u64],
                runtime_offset: 0,
                byte_len: bytes.len() as u64,
            }],
        };
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let row = store
            .read_tensor_row_f32("model.layers.0.self_attn.q_proj.weight", 1, 2)
            .unwrap()
            .unwrap();
        assert_eq!(row, vec![3.0, 4.0]);
        let tile = store
            .read_tensor_rows_f32("model.layers.0.self_attn.q_proj.weight", 0, 2)
            .unwrap();
        assert_eq!(tile, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            store
                .resident
                .lock()
                .expect("dense tensor cache poisoned")
                .bytes,
            0
        );
        let projected = store
            .project(0, "q_proj", &[1.0, 1.0], 2)
            .expect("registered dense projection should decode F32 weights");
        assert_eq!(projected.len(), 2);
        assert!(projected[1] > projected[0]);
    }

    #[test]
    fn dense_transformer_runtime_runs_core_blocks() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2}"#,
        )
        .unwrap();
        let runtime = DenseTransformerRuntime::new(&config);
        let mut hidden = vec![1.0; runtime.width];
        rms_norm_in_place(&mut hidden);
        let before = hidden.clone();
        apply_rotary(&mut hidden, 4, runtime.head_dim, config.rope_theta.unwrap());
        let attended = causal_attention(&hidden, &[(&before, &before)], runtime.head_dim);
        assert_eq!(attended.len(), runtime.width);
    }

    #[test]
    fn dense_transformer_runtime_uses_full_config_hidden_size() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":4}"#,
        )
        .unwrap();
        let runtime = DenseTransformerRuntime::new(&config);
        assert_eq!(runtime.width, 4096);
        assert_eq!(runtime.head_dim, 128);
    }

    #[test]
    fn metal_kv_context_is_capped_by_memory_budget() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":4,"max_position_embeddings":131072}"#,
        )
        .unwrap();
        let context = metal_kv_max_context(&config, 4096, 64 * 1024 * 1024 * 1024);
        assert!(context < 131_072);
        assert!(
            metal_kv_cache_bytes(config.num_hidden_layers, context, 4096)
                <= 16 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn expert_scheduler_reads_only_active_experts_and_reuses_cache() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        fs::write(
            expert_path(temp.path(), 0, 1),
            test_expert_pack("model.layers.0.mlp.experts.1.down_proj.weight"),
        )
        .unwrap();
        fs::write(
            expert_path(temp.path(), 0, 3),
            test_expert_pack("model.layers.0.mlp.experts.3.down_proj.weight"),
        )
        .unwrap();
        fs::write(
            expert_path(temp.path(), 0, 7),
            test_expert_pack("model.layers.0.mlp.experts.7.down_proj.weight"),
        )
        .unwrap();

        let mut scheduler =
            ExpertScheduler::new(ExpertStore::open(temp.path().to_path_buf()).unwrap());
        let pending = scheduler.issue(0, &[1, 3]).unwrap();
        let experts = scheduler.finish(pending).unwrap();
        assert_eq!(experts.len(), 2);
        assert!(experts.iter().all(|expert| expert.layer == 0));
        let first = scheduler.snapshot();
        assert_eq!(first.issued_reads, 2);
        assert_eq!(first.cache_hits, 0);
        assert_eq!(first.cache_misses, 2);

        let pending = scheduler.issue(0, &[3, 7]).unwrap();
        let experts = scheduler.finish(pending).unwrap();
        assert_eq!(experts.len(), 2);
        let second = scheduler.snapshot();
        assert_eq!(second.issued_reads, 3);
        assert_eq!(second.cache_hits, 1);
        assert_eq!(second.cache_misses, 3);
        assert!(second.cached_bytes >= 2 * b"PBQ4EXPERT\0".len());
    }

    #[test]
    fn expert_store_parses_pbq4expert_records_and_projects_with_scales() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        let tensor = "model.layers.0.mlp.experts.2.down_proj.weight";
        let pack = test_expert_pack(tensor);
        fs::write(expert_path(temp.path(), 0, 2), &pack).unwrap();
        fs::write(
            expert_metadata_path(temp.path(), 0, 2),
            serde_json::to_vec(&ExpertPackMetadata {
                layer: 0,
                expert: 2,
                records: vec![ExpertPackRecord {
                    tensor: tensor.to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![1, 4],
                    source_offsets: [0, 4],
                    record_offset: b"PBQ4EXPERT\0".len() as u64,
                    packed_bytes: 2,
                    groups: 1,
                    group_size: GROUP_SIZE,
                }],
            })
            .unwrap(),
        )
        .unwrap();

        let expert = read_one_expert(temp.path(), 0, 2).unwrap();
        assert_eq!(expert.records.len(), 1);
        assert_eq!(expert.records[0].name, tensor);
        assert_eq!(expert.records[0].scales, vec![0.5]);
        assert_eq!(expert.records[0].biases, vec![1.0]);
        let out = expert.project(&[1.0, 2.0, 3.0, 4.0], 1).unwrap();
        let expected = (1.0 * 0.5 + 1.0) * 1.0
            + (2.0 * 0.5 + 1.0) * 2.0
            + (3.0 * 0.5 + 1.0) * 3.0
            + (4.0 * 0.5 + 1.0) * 4.0;
        assert!((out[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn qwen_tokenizer_loads_special_tokens_and_applies_chat_template() {
        let tokenizer = QwenTokenizer::from_json_bytes(test_tokenizer_json()).unwrap();
        let templated = tokenizer.apply_chat_template("hi");
        assert!(templated.contains("<|im_start|>user"));
        let encoded = tokenizer.encode(&templated).unwrap();
        assert_eq!(encoded, vec![100, 5, 3, 101, 100, 6]);
        assert!(encoded.contains(&100));
        assert!(encoded.contains(&101));
        assert_eq!(tokenizer.decode(&[3, 101]).unwrap(), "hi");
        assert!(tokenizer.candidate_token_ids().contains(&102));
        assert!(tokenizer.candidate_token_ids().len() > 4);
    }

    #[test]
    fn qwen_tokenizer_loads_tokenizer_config_chat_template() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_tokenizer_config_json()),
        )
        .unwrap();
        let templated = tokenizer.apply_chat_template("hi");
        assert_eq!(
            templated,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            tokenizer.encode(&templated).unwrap(),
            vec![100, 5, 3, 101, 100, 6]
        );
    }

    #[test]
    fn qwen_tokenizer_uses_byte_level_bpe_from_tokenizer_json() {
        let tokenizer = QwenTokenizer::from_json_bytes(test_byte_bpe_tokenizer_json()).unwrap();
        assert_eq!(tokenizer.encode("hello world").unwrap(), vec![8, 9, 16]);
        assert_eq!(tokenizer.decode(&[8, 9, 16, 101]).unwrap(), "hello world");
        assert_eq!(
            tokenizer.encode("<|im_start|>hello<|im_end|>").unwrap(),
            vec![100, 8, 101]
        );
    }

    #[test]
    fn generate_runs_prefill_decode_sample_and_text_decode_loop() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
        write_test_config(&snapshot);
        let embedding = vec![1u8; 103 * 8];
        std::fs::write(
            snapshot.join("dense.safetensors"),
            make_typed_safetensors(&[(
                "model.embed_tokens.weight",
                "U8",
                vec![103, 8],
                &embedding,
            )]),
        )
        .unwrap();
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            br#"{"weight_map":{"model.embed_tokens.weight":"dense.safetensors"}}"#,
        )
        .unwrap();
        let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot).unwrap();
        let experts = ExpertStore::open(plan.experts_dir.clone()).unwrap();
        let dense = DenseStore::open(
            plan.non_expert_weights.clone(),
            plan.tensor_manifest.clone(),
        )
        .unwrap();
        let tokenizer = QwenTokenizer::from_file(&plan.tokenizer).unwrap();
        let config = QwenModelConfig::from_file(&plan.model_config).unwrap();
        let mut engine = FlashMoeEngine {
            plan,
            experts: experts.clone(),
            scheduler: ExpertScheduler::new(experts),
            dense,
            tokenizer,
            metal: None,
            config,
        };
        let output = engine
            .generate(&GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 16,
                temperature: 0.0,
                top_k: 1,
                seed: 1,
            })
            .unwrap();
        assert_eq!(output.generated_tokens, 16);
        assert!(!output.content.is_empty());
    }

    #[test]
    fn quantize_q4_packs_nibbles_and_group_metadata() {
        let packed = quantize_q4(&[0.0, 15.0, 30.0], &[1, 3], 2).unwrap();
        assert_eq!(packed.values.len(), 2);
        assert_eq!(packed.scales.len(), 2);
        assert_eq!(packed.biases.len(), 2);
    }

    #[test]
    fn expert_cache_quantizes_decoded_bf16_values_not_raw_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
        write_test_config(&snapshot);
        std::fs::write(
            snapshot.join("dense.safetensors"),
            make_safetensors(&[("model.layers.0.self_attn.q_proj.weight", b"dense")]),
        )
        .unwrap();
        let mut gate_bytes = Vec::new();
        for value in 1u32..=128 {
            gate_bytes.extend_from_slice(&(((value as f32).to_bits() >> 16) as u16).to_le_bytes());
        }
        let mut up_bytes = Vec::new();
        for _ in 0..(16 * 8) {
            up_bytes.extend_from_slice(&0x3f80u16.to_le_bytes());
        }
        let mut down_bytes = Vec::new();
        for _ in 0..(8 * 16) {
            down_bytes.extend_from_slice(&0x3f80u16.to_le_bytes());
        }
        std::fs::write(
            snapshot.join("expert.safetensors"),
            make_typed_safetensors(&[
                (
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    "BF16",
                    vec![16, 8],
                    &gate_bytes,
                ),
                (
                    "model.layers.0.mlp.experts.0.up_proj.weight",
                    "BF16",
                    vec![16, 8],
                    &up_bytes,
                ),
                (
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                    "BF16",
                    vec![8, 16],
                    &down_bytes,
                ),
            ]),
        )
        .unwrap();
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            expert_triplet_weight_map(0, 0),
        )
        .unwrap();

        let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot).unwrap();
        let expert = read_one_expert(&plan.experts_dir, 0, 0).unwrap();
        let record = expert
            .records
            .iter()
            .find(|record| record.name.ends_with("gate_proj.weight"))
            .unwrap();
        let out = q4_fma_matvec(
            &record.packed,
            &[1.0; 8],
            &record.scales,
            &record.biases,
            1,
            8,
        )
        .unwrap();
        assert!((out[0] - 36.0).abs() < 1.0, "decoded q4 sum was {}", out[0]);
    }

    #[test]
    fn q4_fma_matvec_dequantizes_nibbles_by_group() {
        let packed = [0x21, 0x43];
        let input = [1.0, 2.0, 3.0, 4.0];
        let scales = [0.5];
        let biases = [1.0];
        let out = q4_fma_matvec(&packed, &input, &scales, &biases, 1, 4).unwrap();
        let expected = (1.0 * 0.5 + 1.0) * 1.0
            + (2.0 * 0.5 + 1.0) * 2.0
            + (3.0 * 0.5 + 1.0) * 3.0
            + (4.0 * 0.5 + 1.0) * 4.0;
        assert!((out[0] - expected).abs() < 1e-6);
    }

    fn test_expert_pack(name: &str) -> Vec<u8> {
        let mut pack = Vec::new();
        pack.extend_from_slice(b"PBQ4EXPERT\0");
        pack.extend_from_slice(&(name.len() as u32).to_le_bytes());
        pack.extend_from_slice(name.as_bytes());
        pack.extend_from_slice(&2u64.to_le_bytes());
        pack.extend_from_slice(&1u64.to_le_bytes());
        pack.extend_from_slice(&0.5f32.to_le_bytes());
        pack.extend_from_slice(&1.0f32.to_le_bytes());
        pack.extend_from_slice(&[0x21, 0x43]);
        pack
    }
}
