use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const QWEN35_MODEL: &str = "hf://mlx-community/Qwen3.5-397B-A17B-4bit";
pub const QWEN35_BF16_MODEL: &str = "hf://Qwen/Qwen3.5-397B-A17B";
pub const QWEN35_MODEL_MARKER: &str = "qwen3.5-397b-a17b";
/// Production native-Q4 source for Qwen3-Coder-Next on Apple Silicon.
pub const QWEN3_CODER_NEXT_MODEL: &str = "hf://mlx-community/Qwen3-Coder-Next-4bit";
pub const QWEN3_CODER_NEXT_MODEL_MARKER: &str = "qwen3-coder-next";
pub const QWEN3_NEXT_CACHE_VERSION: &str = "flashmoe-v2-qwen3-next-mlxq4";
/// Preferred GLM-5.2 source checkpoint. The runtime cache is source-format
/// independent; Colibri remains available through `GLM52_COLIBRI_MODEL`.
pub const GLM52_MXFP4_MODEL: &str = "hf://mlx-community/GLM-5.2-mxfp4";
pub const GLM52_MODEL: &str = GLM52_MXFP4_MODEL;
pub const GLM52_COLIBRI_MODEL: &str = "hf://jlnsrk/GLM-5.2-colibri-int4";
pub const GLM52_MODEL_MARKER: &str = "glm-5.2";
pub const GLM52_CACHE_VERSION: &str = "flashmoe-v5-glm52-mxfp4";
/// Lowercase substring used to identify Qwen3 MoE checkpoints with active
/// parameter counts in their HF repository names, e.g. Qwen3-30B-A3B.
pub const QWEN3_ACTIVE_PARAMS_MARKER: &str = "-a";
/// Hugging Face model URI for the production Q4 Qwen3-VL multimodal MoE model.
pub const QWEN3_VL_MODEL: &str = "hf://mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit";
/// Lowercase family marker used with MoE/active-parameter detection.
pub const QWEN3_VL_MODEL_MARKER: &str = "qwen3-vl";
pub const CACHE_VERSION: &str = "flashmoe-v2-mlxq4";
pub const BF16_CACHE_VERSION: &str = "flashmoe-v2-bf16";
pub const F16_CACHE_VERSION: &str = "flashmoe-v2-f16";
pub const NUM_LAYERS: usize = 60;
pub const NUM_EXPERTS: usize = 512;
pub const ACTIVE_EXPERTS_PER_TOKEN: usize = 4;
pub const HIDDEN_DIM: usize = 4096;
pub const GROUP_SIZE: usize = 64;
pub const FULL_ATTN_INTERVAL: usize = 4;
pub const LINEAR_NUM_V_HEADS: usize = 64;
pub const LINEAR_NUM_K_HEADS: usize = 16;
pub const LINEAR_KEY_DIM: usize = 128;
pub const LINEAR_VALUE_DIM: usize = 128;
pub const LINEAR_TOTAL_KEY: usize = LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM;
pub const LINEAR_TOTAL_VALUE: usize = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM;
pub const LINEAR_CONV_DIM: usize = LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE;
pub const CONV_KERNEL_SIZE: usize = 4;
pub const FOUR_BIT_EXPERT_SIZE: u64 = 7_077_888;
pub const EXPECTED_EXPERT_BYTES: u64 =
    FOUR_BIT_EXPERT_SIZE * NUM_LAYERS as u64 * NUM_EXPERTS as u64;

// ── Vision constants (Qwen3-VL image preprocessor) ───────────────────────────

/// Pixels per spatial patch edge (14 px for Qwen3-VL ViT).
pub const VIT_PATCH_SIZE: usize = 14;
/// Spatial patches merged into one visual language-model token (2×2 = 4).
pub const VIT_MERGE_SIZE: usize = 2;
/// Pixel stride per merged visual token: `VIT_PATCH_SIZE * VIT_MERGE_SIZE`.
pub const VIT_SPATIAL_MERGE_SIZE: usize = VIT_PATCH_SIZE * VIT_MERGE_SIZE; // 28
/// ImageNet pixel mean for ViT normalisation (RGB order).
pub const VIT_IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
/// ImageNet pixel std for ViT normalisation (RGB order).
pub const VIT_IMAGE_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];
/// Upper pixel budget for an input image (~1 280 merged visual tokens).
pub const VIT_MAX_PIXELS: usize = 1280 * VIT_SPATIAL_MERGE_SIZE * VIT_SPATIAL_MERGE_SIZE;
/// Lower pixel budget for an input image (at least 4 merged visual tokens).
pub const VIT_MIN_PIXELS: usize = 4 * VIT_SPATIAL_MERGE_SIZE * VIT_SPATIAL_MERGE_SIZE;
/// Default Qwen3-VL text M-RoPE frequency allocation: temporal, height, width.
pub const DEFAULT_MROPE_SECTION: [usize; 3] = [24, 20, 20];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelection {
    FlashMoePreferred,
    LlamaCpp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertQuantization {
    FourBitProduction,
    Bf16,
    F16,
}

impl ExpertQuantization {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FourBitProduction => "4-bit expert weights",
            Self::Bf16 => "BF16 expert weights",
            Self::F16 => "F16 expert weights",
        }
    }

    pub const fn cache_version(self) -> &'static str {
        match self {
            Self::FourBitProduction => CACHE_VERSION,
            Self::Bf16 => BF16_CACHE_VERSION,
            Self::F16 => F16_CACHE_VERSION,
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

/// Ordered content for multimodal (Qwen3-VL) generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultimodalContent {
    Text { text: String },
    Image { image_path: PathBuf },
}

/// A structured multimodal request for Qwen3-VL inference.
#[derive(Debug, Clone)]
pub struct MultimodalGenerationRequest {
    pub content: Vec<MultimodalContent>,
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub seed: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    pub(crate) fn as_qwen_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

impl Default for ChatMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for ChatMessageContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for ChatMessageContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text {
        text: String,
    },
    Image {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        placeholder_tokens: Option<usize>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(default)]
    pub content: ChatMessageContent,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: ChatMessageContent::Text(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeToolConstraintMode {
    #[default]
    Auto,
    ToolsAllowed,
    ToolRequired,
}

impl NativeToolConstraintMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ToolsAllowed => "tools_allowed",
            Self::ToolRequired => "tool_required",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativePrefillMode {
    #[default]
    Auto,
    Scalar,
    /// Explicit qualification surface for exercising the layer-major graph
    /// below its production prompt-geometry threshold.
    LayerMajor,
}

#[derive(Debug, Clone)]
pub struct StructuredGenerationRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ChatTool>,
    /// Controller-owned stable-root identity for managed invocations. Raw inference omits it.
    pub stage_root: Option<crate::inference::StageRootDescriptor>,
    /// Optional strict JSON artifact schema. This is mutually exclusive with
    /// native tool generation and is enforced token by token.
    pub json_schema: Option<Value>,
    /// Immutable controller-authorized bytes for generation-time mutation validation. Sampling
    /// code must never consult the live workspace.
    pub mutation_snapshot: Option<pb_control_collar::mutation::WorkspaceSnapshot>,
    pub add_generation_prompt: bool,
    /// Whether the tokenizer chat template should permit emitted reasoning.
    /// Structured harness recovery turns disable this so their bounded budget
    /// is available for the required native tool call.
    pub enable_thinking: bool,
    pub raw_prompt: bool,
    pub trace_candidates: bool,
    pub tool_constraint_mode: NativeToolConstraintMode,
    /// Exposed workflow submissions whose first complete constrained call
    /// closes generation. Ordinary tool calls remain batchable.
    pub terminal_tool_names: Vec<String>,
    /// Explicit harness qualification control. Production requests use `Auto`;
    /// `Scalar` preserves the exact token-major reference for A/B parity;
    /// `LayerMajor` is an explicit harness qualification override.
    pub prefill_mode: NativePrefillMode,
    /// Emit exact prefill-state fingerprints in native harness summaries.
    /// This is an opt-in qualification aid because capturing Metal recurrent
    /// state requires a large device-to-host readback.
    pub prefill_state_summary: bool,
    /// Explicit harness qualification chunk boundary for layer-major Qwen
    /// prefill. Production requests leave this unset and use the resource-
    /// resolved graph geometry.
    pub prefill_chunk_tokens: Option<usize>,
    /// Maximum combined prompt and generated-token capacity for this request.
    /// `None` retains the model/runtime default used by direct FlashMoe tools.
    pub context_size: Option<usize>,
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub seed: u32,
}

impl StructuredGenerationRequest {
    pub fn from_prompt(request: &GenerationRequest) -> Self {
        Self {
            messages: vec![ChatMessage::text(ChatRole::User, request.prompt.clone())],
            tools: Vec::new(),
            stage_root: None,
            json_schema: None,
            mutation_snapshot: None,
            add_generation_prompt: true,
            enable_thinking: true,
            raw_prompt: false,
            trace_candidates: false,
            tool_constraint_mode: NativeToolConstraintMode::Auto,
            terminal_tool_names: Vec::new(),
            prefill_mode: NativePrefillMode::Auto,
            prefill_state_summary: false,
            prefill_chunk_tokens: None,
            context_size: None,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_k: request.top_k,
            seed: request.seed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationFinishReason {
    EndOfGeneration,
    MaxTokens,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheSource {
    #[default]
    None,
    MemorySession,
    MemoryPrefix,
    DiskSession,
    DiskPrefix,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCacheStats {
    pub source: PromptCacheSource,
    pub cached_tokens: usize,
    pub prefilled_tokens: usize,
    pub restore_ms: u64,
    pub miss_reason: Option<crate::inference::PromptCacheMissReason>,
    pub lookup_detail: Option<crate::inference::PromptCacheLookupDetail>,
    pub root: Option<crate::inference::BackendPromptRoot>,
}

#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub content: String,
    pub tool_calls: Vec<ChatToolCall>,
    pub finish_reason: GenerationFinishReason,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_cache: PromptCacheStats,
    pub tool_constraints: Option<NativeToolConstraintStats>,
    pub json_constraints: Option<NativeJsonConstraintStats>,
    pub performance: NativeGenerationStats,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeToolConstraintStats {
    pub mode: NativeToolConstraintMode,
    pub dialect: String,
    pub schema_sha256: String,
    pub rejected_candidates: usize,
    pub mutation_rejections: std::collections::BTreeMap<String, usize>,
    pub snapshot_files: usize,
    pub snapshot_bytes: usize,
    pub terminal_state: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeJsonConstraintStats {
    pub schema_sha256: String,
    pub terminal_state: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NativeGenerationStats {
    pub fresh_prefill_tokens: usize,
    pub cached_tokens: usize,
    pub prefill_wall_ms: u64,
    pub prefill_tokens_per_second: f64,
    #[serde(default)]
    pub prefill_metal_commands: usize,
    #[serde(default)]
    pub prefill_host_upload_bytes: usize,
    #[serde(default)]
    pub prefill_host_readback_bytes: usize,
    pub decode_tokens: usize,
    pub decode_wall_ms: u64,
    pub decode_tokens_per_second: f64,
    pub model_family: String,
    pub active_experts_per_token: Option<usize>,
    pub expert_strategy: String,
    pub prefill_command_kind: String,
    #[serde(default)]
    pub prefill_command_reason: String,
    pub thinking_enabled: bool,
    pub refill: NativeRefillStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_state: Option<NativePrefillStateStats>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeRefillStats {
    pub cache_lookup_wall_ms: u64,
    #[serde(default)]
    pub disk_read_decode_wall_ms: u64,
    #[serde(default)]
    pub cpu_state_validation_allocation_wall_ms: u64,
    pub state_hydration_wall_ms: u64,
    pub fresh_suffix_prefill_wall_ms: u64,
    pub snapshot_capture_wall_ms: u64,
    #[serde(default)]
    pub persistence_queue_wall_ms: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCachePersistenceStats {
    pub queued_checkpoints: usize,
    pub completed_checkpoints: usize,
    #[serde(default)]
    pub failed_checkpoints: usize,
    pub wall_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativePrefillStateStats {
    pub final_hidden_sha256: String,
    pub full_attention_kv_sha256: String,
    pub router_recurrent_trace_sha256: String,
    pub linear_attention_state_sha256: String,
    pub full_attention_kv_layer_sha256: Vec<Option<String>>,
    pub router_recurrent_layer_sha256: Vec<Option<String>>,
    pub linear_attention_layer_sha256: Vec<Option<String>>,
}

#[derive(Debug, Clone)]
pub struct TimedGenerationOutput {
    pub output: GenerationOutput,
    pub timing: FlashMoeGenerationTiming,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct FlashMoeMetalResourceSnapshot {
    pub recommended_working_set_bytes: usize,
    pub working_set_limit_bytes: usize,
    pub current_allocated_bytes: usize,
    pub driver_high_water_bytes: usize,
    pub ledger_live_bytes: usize,
    pub ledger_high_water_bytes: usize,
    pub resident_dense_bytes: usize,
    pub recurrent_state_bytes: usize,
    pub resident_expert_wrapper_buffers: usize,
    pub resident_expert_wrapper_bytes: usize,
    pub active_general_buffers: usize,
    pub active_general_bytes: usize,
    pub pooled_buffers: usize,
    pub pooled_bytes: usize,
    pub transient_expert_buffers: usize,
    pub transient_expert_bytes: usize,
    pub in_flight_commands: usize,
    pub command_high_water: usize,
    #[serde(default)]
    pub command_submissions: usize,
    #[serde(default)]
    pub host_upload_bytes: usize,
    #[serde(default)]
    pub host_readback_bytes: usize,
    pub token_boundaries: usize,
    pub pressure_recoveries: usize,
    pub resource_limit_aborts: usize,
    pub buffer_allocations: usize,
    pub buffer_reuses: usize,
    pub buffer_recycles: usize,
    pub buffer_releases: usize,
    pub phase_cleanup_calls: usize,
    pub phase_cleanup_buffers: usize,
}

#[derive(Debug, Clone)]
pub struct FlashMoeGenerationTiming {
    pub model: String,
    pub dimensions: FlashMoeModelDimensions,
    pub prefill_or_ttft_tokens: usize,
    pub prefill_or_ttft_wall: Duration,
    pub decode_tokens: usize,
    pub decode_wall: Duration,
    pub tokens: Vec<FlashMoeTokenTiming>,
    pub total_wall: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeModelDimensions {
    pub layers: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub vocab_size: usize,
    pub experts_per_layer: Option<usize>,
    pub active_experts_per_token: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub shared_experts: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct FlashMoeTokenTiming {
    pub token_index: usize,
    pub position: usize,
    pub phase: FlashMoeTokenPhase,
    pub input_token: u32,
    pub sampled_token: Option<u32>,
    pub layers: Vec<FlashMoeLayerTiming>,
    pub buckets: FlashMoeTimingBuckets,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashMoeTokenPhase {
    Prefill,
    Decode,
}

impl FlashMoeTokenPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlashMoeLayerTiming {
    pub layer: usize,
    pub layer_kind: FlashMoeLayerKind,
    pub active_experts: usize,
    pub dimensions: FlashMoeLayerDimensions,
    pub buckets: FlashMoeTimingBuckets,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashMoeLayerKind {
    FullAttention,
    LinearAttention,
    Unknown,
}

impl FlashMoeLayerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullAttention => "full_attention",
            Self::LinearAttention => "linear_attention",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeLayerDimensions {
    pub hidden_size: usize,
    pub q_width: Option<usize>,
    pub kv_width: Option<usize>,
    pub head_dim: Option<usize>,
    pub experts_per_layer: Option<usize>,
    pub active_experts_per_token: Option<usize>,
    pub shared_experts: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FlashMoeTimingBuckets {
    pub attention_projection: Duration,
    pub attention_input_projection: Duration,
    pub attention_kernel: Duration,
    pub attention_output_projection: Duration,
    pub attention_misc: Duration,
    pub routing: Duration,
    pub deferred_wait: Duration,
    pub expert_io: Duration,
    pub expert_queue: Duration,
    pub expert_read: Duration,
    pub expert_compute: Duration,
    pub combine_norm: Duration,
    pub sampling: Duration,
    pub total_wall: Duration,
    pub expert_bytes_read: u64,
    pub expert_warm_reads: u64,
    pub expert_warm_read: Duration,
    pub expert_warm_bytes_read: u64,
}

impl FlashMoeTimingBuckets {
    pub(crate) fn add(&mut self, other: Self) {
        self.attention_projection += other.attention_projection;
        self.attention_input_projection += other.attention_input_projection;
        self.attention_kernel += other.attention_kernel;
        self.attention_output_projection += other.attention_output_projection;
        self.attention_misc += other.attention_misc;
        self.routing += other.routing;
        self.deferred_wait += other.deferred_wait;
        self.expert_io += other.expert_io;
        self.expert_queue += other.expert_queue;
        self.expert_read += other.expert_read;
        self.expert_compute += other.expert_compute;
        self.combine_norm += other.combine_norm;
        self.sampling += other.sampling;
        self.expert_bytes_read = self
            .expert_bytes_read
            .saturating_add(other.expert_bytes_read);
        self.expert_warm_reads = self
            .expert_warm_reads
            .saturating_add(other.expert_warm_reads);
        self.expert_warm_read += other.expert_warm_read;
        self.expert_warm_bytes_read = self
            .expert_warm_bytes_read
            .saturating_add(other.expert_warm_bytes_read);
    }
}

impl FlashMoeTokenTiming {
    pub(crate) fn new(
        token_index: usize,
        position: usize,
        phase: FlashMoeTokenPhase,
        input_token: u32,
    ) -> Self {
        Self {
            token_index,
            position,
            phase,
            input_token,
            sampled_token: None,
            layers: Vec::new(),
            buckets: FlashMoeTimingBuckets::default(),
        }
    }
}

/// A generation request that includes an image for multimodal (Qwen3-VL) inference.
#[derive(Debug, Clone)]
pub struct VisionGenerationRequest {
    /// Text prompt (will be wrapped in the model's chat template).
    pub prompt: String,
    /// Path to the image to encode.
    pub image_path: PathBuf,
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub seed: u32,
}
