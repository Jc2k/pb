use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use objc2::rc::autoreleasepool;
use sha2::{Digest, Sha256};
use tracing::{debug, info, trace, trace_span};

use super::capabilities::{
    FlashMoeAttentionMathCapability, FlashMoeCapabilityPlan, FlashMoeExpertAccessCapability,
    QwenPrefillGraphCapability,
};
use super::constraints::{NativeToolConstraint, RuntimeToolConstraint};
use super::deepseek::{DeepSeekV4Config, DeepSeekV4ExecutionGraph, is_deepseek_v4_flash};
use super::deepseek_session::{
    DeepSeekV4CheckpointKind, DeepSeekV4SessionCheckpoint, DeepSeekV4SessionStore,
};
use super::experts::ExpertSlotStore;
use super::generation_progress::{GenerationProgress, report_generation_progress};
use super::json_constraints::JsonConstraintSession;
use super::math::*;
use super::metal::*;
use super::model_family::{QwenModelConfig, QwenMoeFamily, QwenMoeModelLayout};
use super::planning::{FlashMoePlan, ResolvedRoutingPolicy};
use super::scheduler::*;
use super::session_cache::{FlashMoeDiskCache, FlashMoeSessionCache};
use super::state::*;
use super::text::*;
use super::types::*;
use super::vision::{
    FlashMoeInputAdapterExecutor, FlashMoeTokenInput, ImagePreprocessor, MropePosition,
    QwenVlRuntimeInputs, VisionEncoding,
};
use super::weights::*;

// Short DeepSeek prompts retain the validated token graph's accumulation order;
// layer-major prefill starts once it can fill one 32-row matrix tile. This is a
// prompt-geometry calculation, never an error fallback.
const DEEPSEEK_V4_BATCH_PREFILL_MIN_TOKENS: usize = 32;
const QWEN_BATCH_PREFILL_MIN_TOKENS: usize = 32;
const QWEN_BATCH_PREFILL_MAX_CHUNK_TOKENS: usize = 8_192;
const QWEN_LAYER_MAJOR_ESTIMATED_BYTES_PER_TOKEN: usize = 384 * 1024;
const QWEN_LAYER_MAJOR_SAFETY_RESERVE_BYTES: usize = 512 * 1024 * 1024;
const QWEN_LAYER_MAJOR_SESSION_RESERVE_DIVISOR: usize = 20;
const RESIDENT_EXPERT_MINIMUM_RESERVE_BYTES: usize = 1024 * 1024 * 1024;
const RESIDENT_EXPERT_RESERVE_DIVISOR: usize = 10;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct QwenTokenExecutionOutput {
    hidden: Vec<f32>,
    recurrent_value: u64,
}

fn deepseek_v4_uses_batch_prefill(tokens: usize) -> bool {
    tokens >= DEEPSEEK_V4_BATCH_PREFILL_MIN_TOKENS
}

fn qwen_prefill_chunk_tokens(
    graph: QwenPrefillGraphCapability,
    tokens: usize,
    resources: Option<&FlashMoeMetalResourceSnapshot>,
) -> Option<usize> {
    if !graph.supports_layer_major() || tokens < QWEN_BATCH_PREFILL_MIN_TOKENS {
        return None;
    }
    let resource_bound = resources.map_or(QWEN_BATCH_PREFILL_MAX_CHUNK_TOKENS, |snapshot| {
        let available = snapshot
            .working_set_limit_bytes
            .saturating_sub(snapshot.current_allocated_bytes)
            .saturating_sub(QWEN_LAYER_MAJOR_SAFETY_RESERVE_BYTES);
        let resident_basis = snapshot
            .current_allocated_bytes
            .max(snapshot.ledger_live_bytes);
        let session_reserve = resident_basis / QWEN_LAYER_MAJOR_SESSION_RESERVE_DIVISOR;
        available
            .min(session_reserve)
            .checked_div(QWEN_LAYER_MAJOR_ESTIMATED_BYTES_PER_TOKEN)
            .unwrap_or(0)
            .min(QWEN_BATCH_PREFILL_MAX_CHUNK_TOKENS)
    });
    if resource_bound < QWEN_BATCH_PREFILL_MIN_TOKENS {
        return None;
    }
    Some(
        tokens
            .min(resource_bound)
            .max(QWEN_BATCH_PREFILL_MIN_TOKENS),
    )
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct NativePrefillResourceDelta {
    metal_commands: usize,
    host_upload_bytes: usize,
    host_readback_bytes: usize,
}

fn native_prefill_resource_delta(
    before: Option<&FlashMoeMetalResourceSnapshot>,
    after: Option<&FlashMoeMetalResourceSnapshot>,
) -> NativePrefillResourceDelta {
    let (Some(before), Some(after)) = (before, after) else {
        return NativePrefillResourceDelta::default();
    };
    NativePrefillResourceDelta {
        metal_commands: after
            .command_submissions
            .saturating_sub(before.command_submissions),
        host_upload_bytes: after
            .host_upload_bytes
            .saturating_sub(before.host_upload_bytes),
        host_readback_bytes: after
            .host_readback_bytes
            .saturating_sub(before.host_readback_bytes),
    }
}

fn resolve_expert_access(
    family: QwenMoeFamily,
    expert_storage: super::experts::ExpertStoreExecutionDescriptor,
    resources: Option<&FlashMoeMetalResourceSnapshot>,
) -> Result<FlashMoeExpertAccessCapability> {
    if family == QwenMoeFamily::DeepSeekV4Flash {
        return Ok(FlashMoeExpertAccessCapability::ParallelPositionedWholeExpertReads);
    }
    let Some(resources) = resources else {
        return Ok(FlashMoeExpertAccessCapability::ParallelPositionedWholeExpertReads);
    };
    let expert_bytes = expert_storage.total_expert_bytes()?;
    let reserve = (resources.working_set_limit_bytes / RESIDENT_EXPERT_RESERVE_DIVISOR)
        .max(RESIDENT_EXPERT_MINIMUM_RESERVE_BYTES);
    let resident_capacity = resources
        .working_set_limit_bytes
        .saturating_sub(resources.current_allocated_bytes)
        .saturating_sub(reserve);
    if expert_bytes <= resident_capacity {
        Ok(FlashMoeExpertAccessCapability::ResidentMappedWholeExpertSlots)
    } else {
        Ok(FlashMoeExpertAccessCapability::ParallelPositionedWholeExpertReads)
    }
}

#[cfg(test)]
mod deepseek_prefill_tests {
    use super::*;

    #[test]
    fn batch_prefill_selection_is_a_fixed_prompt_geometry_calculation() {
        assert!(!deepseek_v4_uses_batch_prefill(
            DEEPSEEK_V4_BATCH_PREFILL_MIN_TOKENS - 1
        ));
        assert!(deepseek_v4_uses_batch_prefill(
            DEEPSEEK_V4_BATCH_PREFILL_MIN_TOKENS
        ));
    }

    #[test]
    fn qwen_batch_chunk_selection_is_geometry_and_resource_resolved() {
        assert_eq!(
            qwen_prefill_chunk_tokens(QwenPrefillGraphCapability::LayerMajorAffineQ4, 31, None),
            None
        );
        assert_eq!(
            qwen_prefill_chunk_tokens(QwenPrefillGraphCapability::LayerMajorAffineQ4, 32, None),
            Some(32)
        );
        assert_eq!(
            qwen_prefill_chunk_tokens(QwenPrefillGraphCapability::LayerMajorAffineQ4, 4_354, None,),
            Some(4_354)
        );
        let constrained = FlashMoeMetalResourceSnapshot {
            working_set_limit_bytes: 48 * 1024 * 1024 * 1024,
            current_allocated_bytes: 47 * 1024 * 1024 * 1024,
            ledger_live_bytes: 47 * 1024 * 1024 * 1024,
            ..FlashMoeMetalResourceSnapshot::default()
        };
        assert_eq!(
            qwen_prefill_chunk_tokens(
                QwenPrefillGraphCapability::LayerMajorAffineQ4,
                4_354,
                Some(&constrained),
            ),
            Some(1_365)
        );
        assert_eq!(
            qwen_prefill_chunk_tokens(QwenPrefillGraphCapability::ScalarToken, 4_354, None),
            None
        );
        let exhausted = FlashMoeMetalResourceSnapshot {
            working_set_limit_bytes: 48 * 1024 * 1024 * 1024,
            current_allocated_bytes: 48 * 1024 * 1024 * 1024,
            ledger_live_bytes: 48 * 1024 * 1024 * 1024,
            ..FlashMoeMetalResourceSnapshot::default()
        };
        assert_eq!(
            qwen_prefill_chunk_tokens(
                QwenPrefillGraphCapability::LayerMajorAffineQ4,
                4_354,
                Some(&exhausted),
            ),
            None
        );
    }

    #[test]
    fn prefill_resource_delta_uses_monotonic_metal_counters() {
        let before = FlashMoeMetalResourceSnapshot {
            command_submissions: 11,
            host_upload_bytes: 1_000,
            host_readback_bytes: 700,
            ..FlashMoeMetalResourceSnapshot::default()
        };
        let after = FlashMoeMetalResourceSnapshot {
            command_submissions: 59,
            host_upload_bytes: 5_096,
            host_readback_bytes: 2_748,
            ..FlashMoeMetalResourceSnapshot::default()
        };
        assert_eq!(
            native_prefill_resource_delta(Some(&before), Some(&after)),
            NativePrefillResourceDelta {
                metal_commands: 48,
                host_upload_bytes: 4_096,
                host_readback_bytes: 2_048,
            }
        );
        assert_eq!(
            native_prefill_resource_delta(None, Some(&after)),
            NativePrefillResourceDelta::default()
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::metal::MetalObjcId as ObjcId;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::scheduler::ScheduledSharedExpertPhaseRef as SharedExpertPhaseRef;

#[derive(Debug)]
pub(super) enum ExpertPhaseInput {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    MetalPostAttention(MetalPostAttentionPrep),
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
enum MlaAttentionOutput {
    Values(Vec<f32>),
    MetalPostAttention(MetalPostAttentionPrep),
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl ScheduledCmd3Input for ExpertPhaseInput {
    fn scheduled_cmd3_input_source(&self) -> ScheduledCmd3InputSource {
        match self {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Self::MetalPostAttention(_) => ScheduledCmd3InputSource::MetalPostAttentionPrep,
        }
    }

    fn scheduled_cmd3_input_state(&self, layer: usize) -> FlashMoeCmd3InputState {
        match self {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Self::MetalPostAttention(prep) => prep.input.scheduled_cmd3_input_state(layer),
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
type ScheduledExpertCommand<'a> =
    ScheduledCmd3Command<'a, Arc<ScheduledExpertSlot>, ExpertPhaseInput, SharedExpertPhaseRef<'a>>;

#[derive(Debug, Clone, Copy)]
struct OptionalInstant(Option<Instant>);

impl OptionalInstant {
    fn now(enabled: bool) -> Self {
        Self(enabled.then(Instant::now))
    }

    fn elapsed(self) -> Duration {
        self.0.map_or(Duration::ZERO, |started| started.elapsed())
    }
}

impl FlashMoeTimingBuckets {
    pub(super) fn add_expert_scheduler_delta(&mut self, delta: ExpertSchedulerSnapshot) {
        self.expert_queue += delta.total_queue_latency;
        self.expert_read += delta.total_read_latency;
        self.expert_bytes_read = self.expert_bytes_read.saturating_add(delta.bytes_read);
        self.expert_warm_reads = self.expert_warm_reads.saturating_add(delta.warm_reads);
        self.expert_warm_read += delta.total_warm_read_latency;
        self.expert_warm_bytes_read = self
            .expert_warm_bytes_read
            .saturating_add(delta.warm_bytes_read);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn trace_cmd2_state(position: usize, layer: usize, prep: &MetalPostAttentionPrep) {
    if !tracing::enabled!(target: "flashmoe_cmd2_state", tracing::Level::TRACE) {
        return;
    }
    let residual = unsafe { read_f32_buffer(prep.residual_buffer, prep.width) };
    let normed = unsafe { read_f32_buffer(prep.normed_buffer, prep.width) };
    let vector_rms = |values: &[f32]| {
        (values.iter().map(|value| value * value).sum::<f32>() / values.len().max(1) as f32).sqrt()
    };
    trace!(
        target: "flashmoe_cmd2_state",
        position,
        layer,
        residual_rms = vector_rms(&residual),
        normed_rms = vector_rms(&normed),
        routes = ?prep.active,
        "flashmoe resolved CMD2 state"
    );
}

struct SampledDecode {
    token: u32,
    hidden: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) enum ResolvedModelExecutor {
    Qwen,
    DeepSeekV4(Arc<DeepSeekV4ExecutionGraph>),
}

impl ResolvedModelExecutor {
    pub(super) fn is_deepseek_v4(&self) -> bool {
        matches!(self, Self::DeepSeekV4(_))
    }

    pub(super) fn deepseek_v4_graph(&self) -> Option<&Arc<DeepSeekV4ExecutionGraph>> {
        match self {
            Self::Qwen => None,
            Self::DeepSeekV4(graph) => Some(graph),
        }
    }
}

#[derive(Debug)]
pub struct FlashMoeEngine {
    pub(super) plan: FlashMoePlan,
    pub(super) scheduler: FlashMoeExecutionScheduler,
    pub(super) dense: DenseStore,
    pub(super) tokenizer: QwenTokenizer,
    pub(super) metal: MetalExecutionFacade,
    pub(super) config: QwenModelConfig,
    pub(super) model_layout: QwenMoeModelLayout,
    pub(super) routing_policy: ResolvedRoutingPolicy,
    pub(super) expert_access: FlashMoeExpertAccessCapability,
    pub(super) qwen_prefill_graph: QwenPrefillGraphCapability,
    pub(super) runtime: DenseTransformerRuntime,
    pub(super) executor: ResolvedModelExecutor,
    pub(super) linear_attention_weights: LinearAttentionWeightTable,
    pub(super) shared_expert_weights: SharedExpertWeightTable,
    pub(super) input_adapter_executor: FlashMoeInputAdapterExecutor,
    pub(super) session_cache: FlashMoeSessionCache,
    pub(super) deepseek_sessions: DeepSeekV4SessionStore<DeepSeekV4SessionSnapshot>,
}

mod loader;
pub use loader::*;

fn nonzero_usize(value: usize) -> Option<usize> {
    (value > 0).then_some(value)
}

fn validate_context_capacity(
    prompt_tokens: usize,
    max_tokens: usize,
    context_size: Option<usize>,
) -> Result<()> {
    let Some(context_size) = context_size else {
        return Ok(());
    };
    if context_size == 0 {
        bail!("FlashMoe context size must be at least one token");
    }
    let required_tokens = prompt_tokens
        .checked_add(max_tokens)
        .context("FlashMoe context token count overflow")?;
    if required_tokens > context_size {
        bail!(
            "FlashMoe context limit exceeded before KV allocation: prompt_tokens={prompt_tokens} max_tokens={max_tokens} required_tokens={required_tokens} ctx_size={context_size}"
        );
    }
    Ok(())
}

fn generation_finish_reason(generated_tokens: usize, max_tokens: usize) -> GenerationFinishReason {
    if generated_tokens < max_tokens {
        GenerationFinishReason::EndOfGeneration
    } else {
        GenerationFinishReason::MaxTokens
    }
}

fn close_unclosed_qwen_terminal_tool_call(content: &str) -> Cow<'_, str> {
    let last_open = content.rfind("<tool_call>");
    let last_close = content.rfind("</tool_call>");
    if last_open.is_some() && last_open > last_close {
        Cow::Owned(format!("{content}</tool_call>"))
    } else {
        Cow::Borrowed(content)
    }
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn tokens_per_second(tokens: usize, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if tokens == 0 || seconds <= f64::EPSILON {
        0.0
    } else {
        tokens as f64 / seconds
    }
}

fn f32_values_sha256(domain: &[u8], values: &[f32]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update((values.len() as u64).to_le_bytes());
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use crate::inference::flashmoe::deepseek::DEEPSEEK_V4_FLASH_MODEL;
    use crate::inference::flashmoe::experts::{
        DenseExpertDtype, ExpertSlotSpec, ExpertStorageLayout, ExpertStoreExecutionDescriptor,
        FixedDenseExpertSlotSpec,
    };
    use crate::inference::flashmoe::planning::plan_unchecked;
    use crate::inference::flashmoe::types::QWEN35_MODEL;

    #[test]
    fn runtime_load_rejects_missing_cache_before_executor_construction() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, root.path());
        let mut phases = Vec::new();

        let error = load_with_progress(&plan, |phase, _| phases.push(phase)).unwrap_err();

        assert_eq!(phases, ["cache_status"]);
        assert!(
            error.to_string().contains("cache is not ready"),
            "{error:#}"
        );
        assert!(!error.to_string().contains("Metal executor"), "{error:#}");
    }

    #[test]
    fn missing_deepseek_cache_diagnostic_is_family_neutral() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(DEEPSEEK_V4_FLASH_MODEL, root.path());

        let error = load(&plan).unwrap_err();
        let diagnostic = error.to_string();

        assert!(
            diagnostic.contains("prepare the FlashMoe cache"),
            "{error:#}"
        );
        assert!(!diagnostic.contains("Qwen3.5 cache"), "{error:#}");
    }

    #[test]
    fn context_capacity_rejects_oversized_requests_before_runtime_allocation() {
        validate_context_capacity(3_900, 256, Some(4_096)).unwrap_err();
        validate_context_capacity(3_840, 256, Some(4_096)).unwrap();
        validate_context_capacity(usize::MAX, 1, Some(usize::MAX)).unwrap_err();
        validate_context_capacity(usize::MAX, 1, None).unwrap();
    }

    #[test]
    fn generation_finish_reason_distinguishes_eos_from_the_token_cap() {
        assert_eq!(
            generation_finish_reason(19, 24),
            GenerationFinishReason::EndOfGeneration
        );
        assert_eq!(
            generation_finish_reason(24, 24),
            GenerationFinishReason::MaxTokens
        );
        assert_eq!(
            generation_finish_reason(0, 0),
            GenerationFinishReason::MaxTokens
        );
    }

    #[test]
    fn terminal_qwen_tool_body_is_closed_only_for_structured_parsing() {
        let body = "<tool_call>{\"name\":\"submit_plan\",\"arguments\":{}}";
        assert_eq!(
            close_unclosed_qwen_terminal_tool_call(body),
            format!("{body}</tool_call>")
        );
        let closed = format!("{body}</tool_call>");
        assert!(matches!(
            close_unclosed_qwen_terminal_tool_call(&closed),
            Cow::Borrowed(_)
        ));
    }

    fn small_expert_storage() -> ExpertStoreExecutionDescriptor {
        let spec = FixedDenseExpertSlotSpec::new(DenseExpertDtype::Bf16, 2, 2).unwrap();
        ExpertStoreExecutionDescriptor {
            layout: ExpertStorageLayout::FixedBf16,
            slot_spec: ExpertSlotSpec::FixedDense(spec),
            layers: 2,
            first_expert_layer: 0,
            experts_per_layer: 4,
        }
    }

    #[test]
    fn expert_access_resolution_selects_only_complete_fitting_resident_corpora() {
        let fitting = FlashMoeMetalResourceSnapshot {
            working_set_limit_bytes: 8 * 1024 * 1024 * 1024,
            current_allocated_bytes: 2 * 1024 * 1024 * 1024,
            ..FlashMoeMetalResourceSnapshot::default()
        };
        assert_eq!(
            resolve_expert_access(
                QwenMoeFamily::Qwen3Moe,
                small_expert_storage(),
                Some(&fitting),
            )
            .unwrap(),
            FlashMoeExpertAccessCapability::ResidentMappedWholeExpertSlots
        );

        let pressured = FlashMoeMetalResourceSnapshot {
            working_set_limit_bytes: 8 * 1024 * 1024 * 1024,
            current_allocated_bytes: 7 * 1024 * 1024 * 1024,
            ..FlashMoeMetalResourceSnapshot::default()
        };
        assert_eq!(
            resolve_expert_access(
                QwenMoeFamily::Qwen3Moe,
                small_expert_storage(),
                Some(&pressured),
            )
            .unwrap(),
            FlashMoeExpertAccessCapability::ParallelPositionedWholeExpertReads
        );
        assert_eq!(
            resolve_expert_access(
                QwenMoeFamily::DeepSeekV4Flash,
                small_expert_storage(),
                Some(&fitting),
            )
            .unwrap(),
            FlashMoeExpertAccessCapability::ParallelPositionedWholeExpertReads
        );
    }

    #[test]
    #[ignore = "requires the pinned local DeepSeek V4 cache and an Apple Silicon Metal device"]
    fn deepseek_session_snapshot_restores_a_b_and_nonzero_batch_suffix_exactly() {
        let plan = plan_unchecked(DEEPSEEK_V4_FLASH_MODEL, &crate::default_models_dir());
        let mut engine = load(&plan).unwrap();
        let full_prompt = format!(
            "Audit this dependency-free browser game and identify the most important invariant.\n\n{}\nThe most important invariant is",
            "function tick(state) { state.score += state.alive ? 1 : 0; return state; }\n"
                .repeat(18)
        );
        let full_tokens = engine.tokenizer.encode(&full_prompt).unwrap();
        let (prefix_prompt, prefix_tokens) = full_prompt
            .char_indices()
            .map(|(index, _)| &full_prompt[..index])
            .filter_map(|prefix| {
                let tokens = engine.tokenizer.encode(prefix).ok()?;
                (tokens.len() >= 32
                    && full_tokens.len().saturating_sub(tokens.len()) >= 32
                    && full_tokens.starts_with(&tokens))
                .then_some((prefix.to_string(), tokens.len()))
            })
            .next()
            .expect("fixture must expose an exact prefix with a batched suffix");
        let raw = |prompt: String| {
            let mut request = StructuredGenerationRequest::from_prompt(&GenerationRequest {
                prompt,
                max_tokens: 1,
                temperature: 0.0,
                top_k: 1,
                seed: 7,
            });
            request.raw_prompt = true;
            request.add_generation_prompt = false;
            request
        };

        engine
            .generate_structured_in_session("session-a", &raw(prefix_prompt))
            .unwrap();
        engine
            .generate_structured_in_session("session-b", &raw("2+2=".to_string()))
            .unwrap();

        let reused = engine
            .generate_structured_in_session("session-a", &raw(full_prompt.clone()))
            .unwrap();
        assert_eq!(reused.prompt_cache.source, PromptCacheSource::MemorySession);
        assert_eq!(reused.prompt_cache.cached_tokens, prefix_tokens);
        assert_eq!(
            reused.prompt_cache.prefilled_tokens,
            full_tokens.len() - prefix_tokens
        );

        let fresh = engine
            .generate_structured_in_session("session-c", &raw(full_prompt.clone()))
            .unwrap();
        assert_eq!(fresh.prompt_cache.source, PromptCacheSource::None);
        assert_eq!(reused.content, fresh.content);
        assert_eq!(reused.tool_calls, fresh.tool_calls);

        let mismatch = engine
            .generate_structured_in_session("session-a", &raw("unrelated".to_string()))
            .unwrap_err();
        assert!(
            mismatch
                .to_string()
                .contains("DeepSeek V4 session prefix mismatch")
        );
        let restored = engine
            .generate_structured_in_session("session-a", &raw(full_prompt))
            .unwrap();
        assert_eq!(restored.prompt_cache.cached_tokens, full_tokens.len());
        assert_eq!(restored.content, fresh.content);
    }
}

impl MetalExecutionFacade {
    pub(super) fn new(
        plan: &FlashMoePlan,
        config: &QwenModelConfig,
        runtime: &DenseTransformerRuntime,
        dense: &DenseStore,
    ) -> Result<Self> {
        if !plan.uses_metal {
            bail!(
                "FlashMoe unsupported required Metal execution: the resolved graph cannot be loaded with Metal disabled"
            );
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let inner = MetalExecutionContext::compile_resolved(
                dense.mmap.clone(),
                dense.len,
                &runtime.linear_attention,
                config.rms_norm_epsilon(),
                super::deepseek::is_deepseek_v4_flash(&plan.model),
            )?;
            tracing::info!(
                model = %plan.model,
                layers = config.num_hidden_layers,
                experts = config.experts(),
                dense_resident = inner.has_resident_dense_weights(),
                "Flash-MoE Metal executor initialized"
            );
            Ok(Self {
                inner: Arc::new(inner),
            })
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (config, runtime, dense);
            bail!(
                "FlashMoe unsupported required Metal execution: the resolved graph requires Apple Silicon Metal"
            )
        }
    }

    pub(super) fn runtime_capabilities(&self) -> MetalRuntimeCapabilities {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.runtime_capabilities()
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            MetalRuntimeCapabilities::from_pipeline_names(MetalPipelineNameSet::new())
        }
    }

    pub(super) fn resource_snapshot(&self) -> Option<FlashMoeMetalResourceSnapshot> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Some(self.inner.resource_snapshot())
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            None
        }
    }

    pub(super) fn set_working_set_limit_bytes(&self, limit: usize) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.set_working_set_limit_bytes(limit)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = limit;
            bail!("FlashMoe Metal resource policy requires Apple Silicon Metal")
        }
    }

    pub(super) fn prepare_resident_expert_backing(
        &self,
        bytes: &super::experts::ReusableExpertBytes,
    ) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.prepare_resident_expert_backing(bytes)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = bytes;
            bail!("resident expert mappings require Apple Silicon Metal")
        }
    }

    pub(super) fn reset_linear_attention_state(&self) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.reset_linear_attention_state()
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            bail!(
                "FlashMoe unsupported recurrent-state reset: the resolved graph requires Apple Silicon Metal"
            )
        }
    }

    pub(super) fn prepare_deepseek_v4_state(
        &self,
        graph: &DeepSeekV4ExecutionGraph,
        capacity: usize,
    ) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.prepare_deepseek_v4_state(graph, capacity)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (graph, capacity);
            bail!("DeepSeek V4 Flash requires Apple Silicon Metal")
        }
    }

    pub(super) fn capture_deepseek_v4_session_state(
        &self,
    ) -> Result<super::metal::DeepSeekV4SessionSnapshot> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.capture_deepseek_v4_session_state()
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            bail!("DeepSeek V4 Flash requires Apple Silicon Metal")
        }
    }

    pub(super) fn restore_deepseek_v4_session_state(
        &self,
        snapshot: &super::metal::DeepSeekV4SessionSnapshot,
    ) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.restore_deepseek_v4_session_state(snapshot)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = snapshot;
            bail!("DeepSeek V4 Flash requires Apple Silicon Metal")
        }
    }

    pub(super) fn deepseek_v4_forward_token(
        &self,
        graph: &DeepSeekV4ExecutionGraph,
        scheduler: &mut FlashMoeExecutionScheduler,
        token: u32,
        position: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner
                .deepseek_v4_forward_token(graph, scheduler, token, position)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (graph, scheduler, token, position);
            bail!("DeepSeek V4 Flash requires Apple Silicon Metal")
        }
    }

    pub(super) fn deepseek_v4_prefill(
        &self,
        graph: &DeepSeekV4ExecutionGraph,
        scheduler: &mut FlashMoeExecutionScheduler,
        tokens: &[u32],
        pos0: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let hidden = autoreleasepool(|_| {
                self.inner
                    .deepseek_v4_prefill(graph, scheduler, tokens, pos0)
            })?;
            self.inner
                .finish_token_boundary(pos0 + tokens.len().saturating_sub(1))?;
            Ok(hidden)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (graph, scheduler, tokens, pos0);
            bail!("DeepSeek V4 Flash requires Apple Silicon Metal")
        }
    }

    pub(super) fn deepseek_v4_logits(
        &self,
        graph: &DeepSeekV4ExecutionGraph,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.deepseek_v4_logits(graph, hidden)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (graph, hidden);
            bail!("DeepSeek V4 Flash requires Apple Silicon Metal")
        }
    }

    pub(super) fn capture_linear_attention_session_state(
        &self,
    ) -> Result<FlashMoeLinearAttentionSessionSnapshot> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.capture_linear_attention_session_state()
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            bail!(
                "FlashMoe unsupported recurrent session capture: the resolved graph requires Apple Silicon Metal"
            )
        }
    }

    pub(super) fn restore_linear_attention_session_state(
        &self,
        snapshot: &FlashMoeLinearAttentionSessionSnapshot,
    ) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.restore_linear_attention_session_state(snapshot)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = snapshot;
            bail!(
                "FlashMoe unsupported recurrent session restore: the resolved graph requires Apple Silicon Metal"
            )
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    fn submit_scheduled_cmd3(
        &self,
        position: usize,
        layer: usize,
        experts: Arc<[Arc<ScheduledExpertSlot>]>,
        routing_weights: &[f32],
        input: MetalPostAttentionPrep,
        output: ScheduledCmd3OutputState,
        shared: SharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
        payloads: &[ScheduledExpertPhaseMlpPayload<'_>],
    ) -> Result<MetalScheduledCmd3Submission> {
        let dense_weights = self.inner.dense_weights().context(
            "FlashMoe unsupported scheduled CMD3 implementation: resident dense Metal weights are unavailable",
        )?;
        MetalScheduledCmd3Builder::new(
            self.inner.runtime(),
            dense_weights,
            Arc::clone(self.inner.buffers()),
            self.inner.norm_epsilon(),
        )
        .submit(
            position,
            layer,
            experts,
            routing_weights,
            input,
            output,
            shared,
            next_norm_weight,
            payloads,
        )
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn submit_scheduled_expert_command(
        &self,
        command: ScheduledExpertCommand<'_>,
    ) -> Result<MetalScheduledCmd3Submission> {
        let output = command.resolve_output_state()?;
        debug_assert_eq!(output.layer, command.layer);
        debug_assert_eq!(output.cmd3, command.cmd3);
        debug_assert_eq!(output.input_state, command.input_state);
        let ScheduledCmd3Command {
            position,
            layer,
            experts,
            weights,
            input,
            shared,
            next_norm_weights,
            payloads,
            ..
        } = command;
        let next_norm_weight = next_norm_weights.values();
        match input {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            ExpertPhaseInput::MetalPostAttention(prep) => {
                debug_assert_eq!(prep.input.state(), prep.state);
                debug_assert_eq!(prep.input.width(), prep.width);
                let pending = self.submit_scheduled_cmd3(
                    position,
                    layer,
                    experts,
                    weights,
                    prep,
                    output,
                    shared,
                    next_norm_weight,
                    &payloads,
                )?;
                Ok(pending)
            }
        }
    }

    pub(super) fn require_resident_dense_weights(&self) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if !self.inner.has_resident_dense_weights() {
                bail!(
                    "FlashMoe unsupported required Metal execution: resident dense weights are unavailable"
                );
            }
            Ok(())
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            bail!(
                "FlashMoe unsupported required Metal execution: Apple Silicon Metal is unavailable"
            )
        }
    }

    #[cfg(test)]
    pub(super) fn has_resident_dense_weights(&self) -> bool {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.has_resident_dense_weights()
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            false
        }
    }

    pub(super) fn resident_top_candidates(
        &self,
        projection: &ResidentMmapMatvecProjection,
        input: &[f32],
        output_rows: usize,
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner
                .resident_top_candidates(projection, input, output_rows, top_k)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (projection, input, output_rows, top_k);
            bail!("FlashMoe unsupported resident topK path: Apple Silicon Metal is required")
        }
    }

    pub(super) fn resident_top_candidates_masked(
        &self,
        projection: &ResidentMmapMatvecProjection,
        input: &[f32],
        output_rows: usize,
        top_k: usize,
        allowed_tokens: &[u32],
    ) -> Result<Vec<(usize, f32)>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.resident_top_candidates_masked(
                projection,
                input,
                output_rows,
                top_k,
                allowed_tokens,
            )
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (projection, input, output_rows, top_k, allowed_tokens);
            bail!("FlashMoe unsupported masked resident topK path: Apple Silicon Metal is required")
        }
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(super) fn router_score_top_candidates(
        &self,
        plan: &RouterScoreProjectionTopKPlan,
        hidden: &[f32],
    ) -> Result<Option<Vec<(usize, f32)>>> {
        if hidden.len() != plan.hidden_width {
            bail!(
                "FlashMoe router topK hidden length {} does not match declared width {} for layer {}",
                hidden.len(),
                plan.hidden_width,
                plan.layer
            );
        }
        let projection = match &plan.source {
            RouterScoreProjectionTopKSource::ResidentDense(projection) => {
                ResidentMmapMatvecProjection::Dense(projection.clone())
            }
            RouterScoreProjectionTopKSource::ResidentQ4(projection) => {
                ResidentMmapMatvecProjection::Q4(projection.clone())
            }
        };
        self.resident_top_candidates(&projection, hidden, plan.experts, plan.active_experts)
            .map(Some)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn resident_post_attention_prep_topk(
        &self,
        projections: &Cmd2ResidentPostAttentionPrepProjections,
        attention_output: &[f32],
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        router_correction_bias: Option<&[f32]>,
    ) -> Result<MetalPostAttentionPrep> {
        self.inner.resident_post_attention_prep_topk(
            projections,
            attention_output,
            residual,
            post_norm_weight,
            router_correction_bias,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn linear_attention_post_attention_prep(
        &self,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        input: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        top_k: usize,
    ) -> Result<MetalPostAttentionPrep> {
        self.inner.linear_attention_post_attention_prep(
            layout,
            bindings,
            input,
            residual,
            post_norm_weight,
            top_k,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen_linear_attention_graph(
        &self,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        rows: usize,
        width: usize,
        input: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
    ) -> Result<MetalLayerMajorPostAttention> {
        self.inner.qwen_linear_attention_graph(
            layout,
            bindings,
            rows,
            width,
            input,
            residual,
            post_norm_weight,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen_post_attention_matrix(
        &self,
        out_proj: &ResidentMmapMatvecProjection,
        router: &ResidentMmapMatvecProjection,
        rows: usize,
        attention_width: usize,
        width: usize,
        attention: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
    ) -> Result<MetalLayerMajorPostAttention> {
        self.inner.qwen_post_attention_matrix(
            out_proj,
            router,
            rows,
            attention_width,
            width,
            attention,
            residual,
            post_norm_weight,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn qwen_layer_major_experts(
        &self,
        scheduled: &ScheduledLayerMajorExperts,
        post_attention: &MetalLayerMajorPostAttention,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
    ) -> Result<MetalQwenPrefillLayerOutput> {
        self.inner
            .qwen_layer_major_experts(scheduled, post_attention, shared, next_norm_weight)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn qwen_final_norm_last_row(
        &self,
        state: &MetalQwenPrefillLayerOutput,
        weight: &[f32],
    ) -> Result<Vec<f32>> {
        self.inner.qwen_final_norm_last_row(state, weight)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[cfg(test)]
    pub(super) fn qwen_rms_norm_rows(
        &self,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        self.inner.qwen_rms_norm_rows(input, weight, rows, width)
    }

    pub(super) fn resident_mmap_matvec_batch(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input: &[f32],
    ) -> Result<(Vec<Vec<f32>>, MetalMatvecTiming, usize)> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner
                .resident_projection_batch(projections, input)?
                .context("FlashMoe required resident Metal projection batch did not resolve")
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (projections, input);
            bail!(
                "FlashMoe unsupported required resident projection batch: Apple Silicon Metal is unavailable"
            )
        }
    }

    #[cfg(test)]
    pub(super) fn resident_mmap_projection_matrix(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input_rows: usize,
        input_cols: usize,
        input: &[f32],
    ) -> Result<(Vec<Vec<f32>>, MetalMatvecTiming, usize)> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner
                .resident_projection_matrix_batch(projections, input_rows, input_cols, input)?
                .context("FlashMoe required resident Metal projection matrix did not resolve")
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (projections, input_rows, input_cols, input);
            bail!(
                "FlashMoe unsupported required resident projection matrix: Apple Silicon Metal is unavailable"
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen_causal_attention_rows(
        &self,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        query_rows: usize,
        prefix_rows: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.qwen_causal_attention_rows(
                queries,
                keys,
                values,
                query_rows,
                prefix_rows,
                query_heads,
                kv_heads,
                head_dim,
            )
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (
                queries,
                keys,
                values,
                query_rows,
                prefix_rows,
                query_heads,
                kv_heads,
                head_dim,
            );
            bail!("Qwen causal-attention rows require Apple Silicon Metal")
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen_causal_attention_rows_owned(
        &self,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        query_gates: Option<&[f32]>,
        query_rows: usize,
        prefix_rows: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<MetalQwenAttentionRows> {
        self.inner.qwen_causal_attention_rows_owned(
            queries,
            keys,
            values,
            query_gates,
            query_rows,
            prefix_rows,
            query_heads,
            kv_heads,
            head_dim,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen_full_attention_graph(
        &self,
        projections: &[ResidentMmapMatvecProjection; 3],
        input: MetalBatchProjectionInput<'_>,
        rows: usize,
        prefix_rows: usize,
        layout: FullAttentionLayout,
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
        rope_sin: &[f32],
        rope_cos: &[f32],
        prefix_keys: &[f32],
        prefix_values: &[f32],
    ) -> Result<MetalQwenFullAttentionOutput> {
        self.inner.qwen_full_attention_graph(
            projections,
            input,
            rows,
            prefix_rows,
            layout,
            q_norm_weight,
            k_norm_weight,
            rope_sin,
            rope_cos,
            prefix_keys,
            prefix_values,
        )
    }

    pub(super) fn resident_glm_mla_absorbed_attention(
        &self,
        embed_q: &DenseQ4MmapMatvecProjection,
        unembed_out: &DenseQ4MmapMatvecProjection,
        input: MetalGlmMlaAbsorbedAttentionInput<'_>,
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner
                .resident_glm_mla_absorbed_attention(embed_q, unembed_out, input)?
                .context("FlashMoe required resident GLM MLA absorbed attention did not resolve")
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (embed_q, unembed_out, input);
            bail!(
                "FlashMoe unsupported resident GLM MLA absorbed attention: Apple Silicon Metal is unavailable"
            )
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resident_glm_mla_input_projection_chain(
        &self,
        q_a: &ResidentMmapMatvecProjection,
        kv_a: &ResidentMmapMatvecProjection,
        q_b: &ResidentMmapMatvecProjection,
        input: MetalBatchProjectionInput<'_>,
        q_norm_weight: &[f32],
        kv_norm_weight: &[f32],
        kv_lora_rank: usize,
        norm_epsilon: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.inner
            .resident_glm_mla_input_projection_chain(
                q_a,
                kv_a,
                q_b,
                input,
                q_norm_weight,
                kv_norm_weight,
                kv_lora_rank,
                norm_epsilon,
            )?
            .context("FlashMoe required GLM MLA input projection chain did not resolve")
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resident_glm_mla_fused_attention(
        &self,
        q_a: &ResidentMmapMatvecProjection,
        kv_a: &ResidentMmapMatvecProjection,
        q_b: &ResidentMmapMatvecProjection,
        embed_q: &DenseQ4MmapMatvecProjection,
        unembed_out: &DenseQ4MmapMatvecProjection,
        input: MetalGlmMlaFusedAttentionInput<'_>,
        q_norm_weight: &[f32],
        kv_norm_weight: &[f32],
        norm_epsilon: f32,
    ) -> Result<MetalGlmMlaFusedAttentionOutput> {
        self.inner
            .resident_glm_mla_fused_attention(
                q_a,
                kv_a,
                q_b,
                embed_q,
                unembed_out,
                input,
                q_norm_weight,
                kv_norm_weight,
                norm_epsilon,
            )?
            .context("FlashMoe required fused GLM MLA attention did not resolve")
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn resident_mmap_matvec_batch_with_input_buffer(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input_buffer: ObjcId,
        input_len: usize,
    ) -> Result<(Vec<Vec<f32>>, MetalMatvecTiming, usize)> {
        self.inner
            .resident_projection_batch_with_input_buffer(projections, input_buffer, input_len)?
            .context("FlashMoe required resident Metal projection batch did not resolve")
    }
}

impl FlashMoeEngine {
    pub fn persist_session_cache(
        &mut self,
        session_id: &str,
    ) -> Result<super::types::PromptCachePersistenceStats> {
        if session_id.trim().is_empty() {
            return Ok(super::types::PromptCachePersistenceStats::default());
        }
        if self.executor.is_deepseek_v4() {
            return Ok(super::types::PromptCachePersistenceStats::default());
        }
        self.session_cache.persist_session(session_id)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn forward_token_input(
        &mut self,
        input: FlashMoeTokenInput<'_>,
        kv_cache: &mut KvCache,
        position: usize,
        record_generated: bool,
        timing: Option<&mut FlashMoeTokenTiming>,
        progress: GenerationProgress<'_>,
    ) -> Result<Vec<f32>> {
        // Raw `commandBuffer`/encoder Objective-C messages return autoreleased
        // objects. A CLI process has no ambient AppKit autorelease pool, so a
        // long prefill or decode must drain those transients at the token
        // boundary while retained model/state buffers remain alive.
        let hidden = autoreleasepool(|_| {
            self.forward_token_input_in_autoreleasepool(
                input,
                kv_cache,
                position,
                record_generated,
                timing,
                progress,
            )
        })?;
        self.metal.inner.finish_token_boundary(position)?;
        Ok(hidden.hidden)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn forward_token_input_in_autoreleasepool(
        &mut self,
        input: FlashMoeTokenInput<'_>,
        kv_cache: &mut KvCache,
        position: usize,
        record_generated: bool,
        mut timing: Option<&mut FlashMoeTokenTiming>,
        progress: GenerationProgress<'_>,
    ) -> Result<QwenTokenExecutionOutput> {
        let runtime = &self.runtime;
        let record_detailed_timing = timing.is_some();
        let token_started = OptionalInstant::now(record_detailed_timing);
        let previous = input.token();
        let rope_position = input.rope_position();
        let token_span = trace_span!(
            target: "flashmoe::perf",
            "token",
            token_position = position,
            input_token = previous,
            record_generated
        );
        let _token_span = token_span.enter();
        if let Some(graph) = self.executor.deepseek_v4_graph().cloned() {
            if input.precomputed_embedding(runtime.width)?.is_some() {
                bail!(
                    "DeepSeek V4 Flash does not declare a precomputed/vision embedding input graph"
                );
            }
            let hidden = self.metal.deepseek_v4_forward_token(
                &graph,
                &mut self.scheduler,
                previous,
                position,
            )?;
            if record_generated {
                kv_cache.record_generated_token_record(FlashMoeGeneratedTokenRecord::new(
                    position, previous,
                ))?;
            }
            if let Some(timing) = timing {
                timing.buckets.total_wall = token_started.elapsed();
            }
            return Ok(QwenTokenExecutionOutput {
                hidden,
                recurrent_value: 0,
            });
        }
        let hidden_values = match input.precomputed_embedding(runtime.width)? {
            Some(values) => values.to_vec(),
            None => self.dense.embedding(previous, runtime.width)?,
        };
        let mut token_state = FlashMoeTokenState::new(
            hidden_values,
            self.dense.seed(position, previous)? ^ (self.plan.model.len() as u64),
        );
        let prepared_full_attention: Option<&[f32]> = None;
        debug_assert!(token_state.hidden().is_declared_graph_state());
        let mut deferred_expert_phase: Option<MetalScheduledCmd3Submission> = None;

        for layer in 0..self.config.num_hidden_layers {
            let layer_addition = input.layer_addition(layer, runtime.width)?;
            let allow_deferred_output = layer_addition.is_none();
            let report_layer_progress = progress.is_some();
            if report_layer_progress {
                report_generation_progress(&progress, || {
                    format!(
                        "forward layer begin position={} layer={}/{}",
                        position,
                        layer + 1,
                        self.config.num_hidden_layers
                    )
                });
            }
            if self.config.is_dense_mlp_layer(layer) {
                if deferred_expert_phase.is_some() {
                    bail!(
                        "GLM dense lead-in layer {layer} received a deferred sparse expert phase"
                    );
                }
                let layer_started = OptionalInstant::now(record_detailed_timing);
                let mut layer_timing = FlashMoeLayerTiming {
                    layer,
                    layer_kind: FlashMoeLayerKind::FullAttention,
                    active_experts: 0,
                    dimensions: self.layer_dimensions(layer),
                    buckets: FlashMoeTimingBuckets::default(),
                };
                self.forward_glm_dense_layer(
                    layer,
                    &mut token_state,
                    kv_cache,
                    position,
                    rope_position,
                    runtime,
                    record_detailed_timing.then_some(&mut layer_timing.buckets),
                )?;
                if let Some(addition) = layer_addition {
                    add_in_place(token_state.hidden_mut(), addition);
                }
                token_state.clear_next_layer_normed();
                kv_cache
                    .record_layer_state_record(token_state.layer_state_record(position, layer))?;
                layer_timing.buckets.total_wall = layer_started.elapsed();
                if report_layer_progress {
                    report_generation_progress(&progress, || {
                        format!(
                            "forward dense layer complete position={} layer={}/{} total_ms={}",
                            position,
                            layer + 1,
                            self.config.num_hidden_layers,
                            layer_started.elapsed().as_millis()
                        )
                    });
                }
                if let Some(timing) = timing.as_deref_mut() {
                    timing.buckets.add(layer_timing.buckets);
                    timing.layers.push(layer_timing);
                }
                continue;
            }
            let mut pending_for_layer = deferred_expert_phase.take();
            let (deferred_attention_input, deferred_residual_input) = {
                let normed_candidate = pending_for_layer
                    .as_ref()
                    .map(MetalScheduledCmd3Submission::next_normed_input)
                    .transpose()?
                    .flatten();
                if normed_candidate.is_some() {
                    let residual_candidate = Some(
                        pending_for_layer
                            .as_ref()
                            .context("missing deferred Metal CMD3 submission")?
                            .hidden_input()?,
                    );
                    (normed_candidate, residual_candidate)
                } else {
                    (None, None)
                }
            };

            if pending_for_layer.is_some() && deferred_attention_input.is_none() {
                let pending = pending_for_layer
                    .take()
                    .context("missing deferred expert phase")?;
                let wait_span = trace_span!(
                    target: "flashmoe::perf",
                    "expert_wait",
                    token_position = position,
                    completed_layer = layer.saturating_sub(1)
                );
                let _wait_span = wait_span.enter();
                let wait_started = OptionalInstant::now(record_detailed_timing);
                let output = pending.wait()?;
                let wait_elapsed = wait_started.elapsed();
                trace!(
                    target: "flashmoe::perf",
                    token_position = position,
                    completed_layer = layer.saturating_sub(1),
                    "flashmoe deferred expert wait complete"
                );
                if let Some(timing) = timing.as_deref_mut() {
                    timing.buckets.deferred_wait += wait_elapsed;
                    if let Some(previous_layer) = timing.layers.last_mut() {
                        previous_layer.buckets.deferred_wait += wait_elapsed;
                        previous_layer.buckets.total_wall += wait_elapsed;
                    }
                }
                token_state.apply_declared_expert_phase(
                    output,
                    FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
                )?;
            }
            let previous_handoff = if let Some(next_normed) = deferred_attention_input {
                let hidden = deferred_residual_input.with_context(|| {
                    format!(
                        "FlashMoe scheduled layer {layer} has deferred next-normed input without deferred hidden state"
                    )
                })?;
                ScheduledPreviousCmd3Handoff::deferred_gpu(
                    layer.saturating_sub(1),
                    hidden.state(),
                    next_normed.state(),
                )
            } else if layer == 0 {
                ScheduledPreviousCmd3Handoff::initial(token_state.hidden().len())
            } else {
                ScheduledPreviousCmd3Handoff::cpu_visible(layer - 1, token_state.hidden().len())
            };
            let layer_schedule = self.scheduler.begin_resolved_layer(
                position,
                layer,
                self.config.num_hidden_layers,
                previous_handoff,
                allow_deferred_output,
            )?;
            let attention_implementation = layer_schedule.attention_implementation();
            let layer_span = trace_span!(
                target: "flashmoe::perf",
                "layer",
                token_position = position,
                layer,
                layer_kind = ?attention_implementation
            );
            let _layer_span = layer_span.enter();
            let layer_started = OptionalInstant::now(record_detailed_timing);
            let mut layer_timing = FlashMoeLayerTiming {
                layer,
                layer_kind: match attention_implementation {
                    ScheduledLayerAttentionImplementation::FullAttentionCpuKv => {
                        FlashMoeLayerKind::FullAttention
                    }
                    ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal => {
                        FlashMoeLayerKind::LinearAttention
                    }
                },
                active_experts: 0,
                dimensions: self.layer_dimensions(layer),
                buckets: FlashMoeTimingBuckets::default(),
            };
            let combine_started = OptionalInstant::now(record_detailed_timing);
            let input_norm_name = layer_norm_tensor_name(layer, "input_layernorm");
            let mut normed =
                if deferred_attention_input.is_some() {
                    FlashMoeCpuBuffer::normed(Vec::new())
                } else if let Some(normed) = token_state.take_next_layer_normed_as_normed() {
                    normed
                } else {
                    FlashMoeCpuBuffer::normed(self.rms_norm_with_model_weight(
                        input_norm_name.as_str(),
                        token_state.hidden(),
                    )?)
                };
            layer_timing.buckets.combine_norm += combine_started.elapsed();
            let attention_started = OptionalInstant::now(record_detailed_timing);
            let cmd1_input = if deferred_attention_input.is_some() {
                ScheduledCmd1InputSource::DeferredMetalNextNormed
            } else {
                ScheduledCmd1InputSource::CpuNormedHidden
            };
            let cmd1_input_state = if let Some(input) = deferred_attention_input {
                FlashMoeCmd1InputState::gpu_next_layer_normed(layer, input.state())
            } else {
                FlashMoeCmd1InputState::cpu_normed(layer, normed.len())
            };
            let (scheduled_cmd1, layer_schedule) =
                layer_schedule.resolve(&self.scheduler, cmd1_input, cmd1_input_state)?;
            debug_assert_eq!(scheduled_cmd1.layer, layer);
            debug_assert_eq!(scheduled_cmd1.cmd1.layer, layer);
            debug_assert_eq!(scheduled_cmd1.input, cmd1_input);
            debug_assert_eq!(scheduled_cmd1.input_state.layer(), layer);
            debug_assert!(scheduled_cmd1.input_state.is_declared_graph_state());
            let mut post_attention_values_for_prep = None;
            let post_norm_name = layer_norm_tensor_name(layer, "post_attention_layernorm");
            let mut early_metal_post_attention_prep: Option<MetalPostAttentionPrep> = None;
            let projected = match attention_implementation {
                ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal => {
                    let residual_input = deferred_residual_input
                        .map(|input| MetalBatchProjectionInput::Buffer {
                            buffer: input.buffer(),
                            len: input.len(),
                        })
                        .unwrap_or(MetalBatchProjectionInput::Cpu(token_state.hidden()));
                    let post_norm_weight = self
                        .model_norm_weight(post_norm_name.as_str(), runtime.width)?
                        .with_context(|| {
                            format!(
                                "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD2 path: missing norm tensor {post_norm_name}"
                            )
                        })?;
                    let prep = self.linear_attention_post_attention_prep_with_metal(
                        layer,
                        &normed,
                        deferred_attention_input,
                        residual_input,
                        &post_norm_weight,
                        runtime,
                        record_detailed_timing.then_some(&mut layer_timing.buckets),
                    )?;
                    if deferred_residual_input.is_some()
                        && let Some(pending) = pending_for_layer.take()
                    {
                        pending.finish_without_readback()?;
                    }
                    early_metal_post_attention_prep = Some(prep);
                    Vec::new()
                }
                ScheduledLayerAttentionImplementation::FullAttentionCpuKv => {
                    if runtime.is_mla_attention_layer(layer) {
                        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                        let mla_output = {
                            let residual = deferred_residual_input
                                .map(|input| MetalBatchProjectionInput::Buffer {
                                    buffer: input.buffer(),
                                    len: input.len(),
                                })
                                .unwrap_or(MetalBatchProjectionInput::Cpu(token_state.hidden()));
                            let post_norm_weight = self
                                .model_norm_weight(post_norm_name.as_str(), runtime.width)?
                                .with_context(|| {
                                    format!(
                                        "FlashMoe unsupported fused GLM MLA CMD2 path: missing norm tensor {post_norm_name}"
                                    )
                                })?;
                            let router_correction_bias = self.router_correction_bias(layer)?;
                            self.mla_attention_output_values(
                                layer,
                                &normed,
                                deferred_attention_input,
                                kv_cache,
                                position,
                                rope_position,
                                runtime,
                                Some(GlmMlaPostAttentionRequest {
                                    residual,
                                    post_norm_weight: &post_norm_weight,
                                    router_correction_bias: router_correction_bias.as_deref(),
                                    experts: self.scheduler.experts_per_layer(),
                                    active_experts: self.scheduler.active_experts(),
                                }),
                                record_detailed_timing.then_some(&mut layer_timing.buckets),
                            )?
                        };
                        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                        let mla_output = self.mla_attention_output_values(
                            layer,
                            &normed,
                            deferred_attention_input,
                            kv_cache,
                            position,
                            rope_position,
                            runtime,
                            None,
                            record_detailed_timing.then_some(&mut layer_timing.buckets),
                        )?;
                        match mla_output {
                            MlaAttentionOutput::Values(values) => {
                                post_attention_values_for_prep =
                                    Some((attention_tensor_name(layer, "o_proj"), values));
                            }
                            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                            MlaAttentionOutput::MetalPostAttention(prep) => {
                                if deferred_residual_input.is_some()
                                    && let Some(pending) = pending_for_layer.take()
                                {
                                    pending.finish_without_readback()?;
                                }
                                early_metal_post_attention_prep = Some(prep);
                            }
                        }
                    } else {
                        let values = if let Some(prepared) = prepared_full_attention {
                            let expected = runtime.full_attention_layout(layer)?.q_width;
                            if prepared.len() != expected {
                                bail!(
                                    "Qwen layer-major full-attention row at layer {layer} has {} values, expected {expected}",
                                    prepared.len()
                                );
                            }
                            prepared.to_vec()
                        } else {
                            self.full_attention_output_values(
                                layer,
                                &normed,
                                deferred_attention_input,
                                kv_cache,
                                position,
                                rope_position,
                                runtime,
                                record_detailed_timing.then_some(&mut layer_timing.buckets),
                            )?
                        };
                        post_attention_values_for_prep =
                            Some((attention_tensor_name(layer, "o_proj"), values));
                    }
                    Vec::new()
                }
            };
            trace_layer_values(position, layer, "attention", &projected);
            layer_timing.buckets.attention_projection += attention_started.elapsed();
            let can_defer_residual_wait_for_post_prep =
                deferred_residual_input.is_some() && post_attention_values_for_prep.is_some();
            if deferred_attention_input.is_some()
                && !can_defer_residual_wait_for_post_prep
                && let Some(pending) = pending_for_layer.take()
            {
                let wait_span = trace_span!(
                    target: "flashmoe::perf",
                    "expert_wait",
                    token_position = position,
                    completed_layer = layer.saturating_sub(1),
                    after_input_projection = true
                );
                let _wait_span = wait_span.enter();
                let wait_started = OptionalInstant::now(record_detailed_timing);
                let output = pending.wait()?;
                let wait_elapsed = wait_started.elapsed();
                trace!(
                    target: "flashmoe::perf",
                    token_position = position,
                    completed_layer = layer.saturating_sub(1),
                    "flashmoe deferred expert wait complete after input projection"
                );
                if let Some(timing) = timing.as_deref_mut() {
                    timing.buckets.deferred_wait += wait_elapsed;
                    if let Some(previous_layer) = timing.layers.last_mut() {
                        previous_layer.buckets.deferred_wait += wait_elapsed;
                        previous_layer.buckets.total_wall += wait_elapsed;
                    }
                }
                token_state.apply_declared_expert_phase(
                    output,
                    FlashMoeExpertPhaseApplication::HiddenOnly,
                )?;
            }
            let mut metal_post_attention_prep: Option<MetalPostAttentionPrep> =
                early_metal_post_attention_prep;
            let cmd2_attention_len = metal_post_attention_prep
                .as_ref()
                .map(|prep| prep.width)
                .or_else(|| {
                    post_attention_values_for_prep
                        .as_ref()
                        .map(|(_, values)| values.len())
                })
                .unwrap_or(projected.len());
            let cmd2_residual_len = deferred_residual_input
                .map(|input| input.len())
                .unwrap_or_else(|| token_state.hidden().len());
            let cmd2_attention_input = if metal_post_attention_prep.is_some() {
                ScheduledCmd2AttentionInput::metal_values(cmd2_attention_len)
            } else {
                ScheduledCmd2AttentionInput::cpu_values(cmd2_attention_len)
            };
            let cmd2_residual_input = if deferred_residual_input.is_some() {
                ScheduledCmd2ResidualInput::metal_buffer(cmd2_residual_len)
            } else {
                ScheduledCmd2ResidualInput::cpu_hidden(cmd2_residual_len)
            };
            let (scheduled_cmd2, layer_schedule) = layer_schedule.resolve(
                &self.scheduler,
                ScheduledCmd2PhaseInputs::from_inputs(cmd2_attention_input, cmd2_residual_input),
            )?;
            debug_assert_eq!(scheduled_cmd2.input_state().layer(), layer);
            debug_assert_eq!(
                scheduled_cmd2.input_state().attention().len(),
                cmd2_attention_len
            );
            debug_assert_eq!(
                scheduled_cmd2.input_state().residual().len(),
                cmd2_residual_len
            );
            let combine_started = OptionalInstant::now(record_detailed_timing);
            let mut precomputed_active: Option<ScheduledRoutingCommand> = None;
            if let Some(prep) = metal_post_attention_prep.as_mut() {
                let routing_command = self.scheduler.routing_from_post_attention_prep(
                    &scheduled_cmd2,
                    prep.state,
                    &prep.active,
                )?;
                debug_assert_eq!(
                    routing_command.source,
                    ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK
                );
                precomputed_active = Some(prep.attach_routing_command(routing_command)?);
            }
            let active = if let Some(routing_command) = precomputed_active {
                layer_timing.buckets.combine_norm += combine_started.elapsed();
                routing_command
            } else if let Some((out_proj_name, attention_values)) =
                post_attention_values_for_prep.take()
            {
                let residual_input = deferred_residual_input
                    .map(|input| MetalBatchProjectionInput::Buffer {
                        buffer: input.buffer(),
                        len: input.len(),
                    })
                    .unwrap_or(MetalBatchProjectionInput::Cpu(token_state.hidden()));
                let metal = &self.metal;
                let post_norm_weight = self
                    .model_norm_weight(post_norm_name.as_str(), runtime.width)?
                    .with_context(|| {
                        format!(
                            "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: missing norm tensor {post_norm_name}"
                        )
                        })?;
                let router_correction_bias = self.router_correction_bias(layer)?;
                let mut prep = self.dense.post_attention_prep_with_metal(
                    metal,
                    layer,
                    self.scheduler.experts_per_layer(),
                    &out_proj_name,
                    &attention_values,
                    residual_input,
                    &post_norm_weight,
                    scheduled_cmd2.active_experts,
                    router_correction_bias.as_deref(),
                )?;
                if deferred_residual_input.is_some()
                    && let Some(pending) = pending_for_layer.take()
                {
                    pending.finish_without_readback()?;
                }
                let routing_command = self.scheduler.routing_from_post_attention_prep(
                    &scheduled_cmd2,
                    prep.state,
                    &prep.active,
                )?;
                debug_assert_eq!(
                    routing_command.source,
                    ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK
                );
                let routing_command = prep.attach_routing_command(routing_command)?;
                metal_post_attention_prep = Some(prep);
                layer_timing.buckets.combine_norm += combine_started.elapsed();
                routing_command
            } else {
                add_in_place(token_state.hidden_mut(), &projected);
                normed = FlashMoeCpuBuffer::normed(
                    self.rms_norm_with_model_weight(post_norm_name.as_str(), token_state.hidden())?,
                );
                layer_timing.buckets.combine_norm += combine_started.elapsed();
                let routing_started = OptionalInstant::now(record_detailed_timing);
                let active = self.route_layer(layer, &normed, scheduled_cmd2.active_experts)?;
                layer_timing.buckets.routing += routing_started.elapsed();
                active
            };
            layer_timing.active_experts = active.routes.len();
            if let Some(prep) = metal_post_attention_prep.as_ref() {
                trace_cmd2_state(position, layer, prep);
            }
            let layer_schedule = layer_schedule.resolve(&active)?;
            let pending_cmd3 = layer_schedule.issue_cmd3(&mut self.scheduler, &active)?;
            // While expert reads are still pending, prepare the always-active
            // shared-expert branch for the deferred expert command buffer.
            let shared_compute_started = OptionalInstant::now(record_detailed_timing);
            let shared_phase = match self.shared_expert_weights.layer(layer)? {
                SharedExpertLayerWeights::Resident(shared) => {
                    SharedExpertPhaseRef::Resident(shared)
                }
                SharedExpertLayerWeights::None => SharedExpertPhaseRef::None,
            };
            layer_timing.buckets.expert_compute += shared_compute_started.elapsed();
            let cmd3_prepare_started = OptionalInstant::now(record_detailed_timing);
            let prepared_next_norm_weights = prepare_scheduled_next_norm_weights(
                layer,
                self.config.num_hidden_layers,
                runtime.width,
                allow_deferred_output,
                |name, width| self.model_norm_weight(name, width),
            )?;
            let next_norm_weights = prepared_next_norm_weights.scheduled()?;
            layer_timing.buckets.expert_compute += cmd3_prepare_started.elapsed();
            let prep = metal_post_attention_prep.take().with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled Qwen3.5 CMD3 path at layer {layer}: resolved Metal CMD2 did not produce post-attention state; CPU normed/residual upload is not a declared implementation"
                )
            })?;
            let expert_delta = {
                let metal = &self.metal;
                let layer_execution = pending_cmd3.finish(
                    &mut self.scheduler,
                    ExpertPhaseInput::MetalPostAttention(prep),
                    shared_phase,
                    next_norm_weights,
                    |command| metal.submit_scheduled_expert_command(command),
                )?;
                let cmd3_execution = layer_execution.cmd3;
                let expert_delta = cmd3_execution.expert_delta;
                layer_timing.buckets.expert_io += cmd3_execution.expert_io_elapsed;
                layer_timing
                    .buckets
                    .add_expert_scheduler_delta(expert_delta);
                for (mix_hash, weight) in cmd3_execution.expert_mixes {
                    token_state.mix_active_expert(mix_hash, weight);
                }
                layer_timing.buckets.expert_compute += cmd3_execution.submit_elapsed;
                let pending = cmd3_execution.submission;
                match layer_execution.output_handoff {
                    ScheduledCmd3OutputHandoff::DeferredToNextLayer => {
                        deferred_expert_phase = Some(pending);
                    }
                    ScheduledCmd3OutputHandoff::CompleteHere => {
                        let output = pending.wait()?;
                        token_state.apply_declared_expert_phase(
                            output,
                            if layer_addition.is_some() {
                                FlashMoeExpertPhaseApplication::HiddenOnly
                            } else {
                                FlashMoeExpertPhaseApplication::HiddenAndNextNormed
                            },
                        )?;
                    }
                }
                expert_delta
            };
            if let Some(addition) = layer_addition {
                if deferred_expert_phase.is_some() {
                    bail!(
                        "FlashMoe scheduled layer {layer} deferred CMD3 despite a declared layer addition"
                    );
                }
                add_in_place(token_state.hidden_mut(), addition);
                token_state.clear_next_layer_normed();
            }
            trace_layer_values(position, layer, "moe", token_state.hidden());
            let combine_started = OptionalInstant::now(record_detailed_timing);
            if deferred_expert_phase.is_some() {
                kv_cache
                    .record_layer_state_record(token_state.layer_state_record(position, layer))?;
                layer_timing.buckets.combine_norm += combine_started.elapsed();
                layer_timing.buckets.total_wall = layer_started.elapsed();
                trace!(
                    target: "flashmoe::perf",
                    token_position = position,
                    layer,
                    layer_kind = layer_timing.layer_kind.as_str(),
                    active_experts = layer_timing.active_experts,
                    expert_deferred = true,
                    bytes_read = expert_delta.bytes_read,
                    "flashmoe layer complete"
                );
                if report_layer_progress {
                    report_generation_progress(&progress, || {
                        format!(
                            "forward layer complete position={} layer={}/{} attention_ms={} routing_ms={} expert_io_ms={} expert_read_ms={} expert_compute_ms={} combine_norm_ms={} total_ms={}",
                            position,
                            layer + 1,
                            self.config.num_hidden_layers,
                            layer_timing.buckets.attention_projection.as_millis(),
                            layer_timing.buckets.routing.as_millis(),
                            layer_timing.buckets.expert_io.as_millis(),
                            layer_timing.buckets.expert_read.as_millis(),
                            layer_timing.buckets.expert_compute.as_millis(),
                            layer_timing.buckets.combine_norm.as_millis(),
                            layer_started.elapsed().as_millis()
                        )
                    });
                }
                if let Some(timing) = timing.as_deref_mut() {
                    timing.buckets.add(layer_timing.buckets);
                    timing.layers.push(layer_timing);
                }
                continue;
            }
            kv_cache.record_layer_state_record(token_state.layer_state_record(position, layer))?;
            layer_timing.buckets.combine_norm += combine_started.elapsed();
            layer_timing.buckets.total_wall = layer_started.elapsed();
            trace!(
                target: "flashmoe::perf",
                token_position = position,
                layer,
                layer_kind = layer_timing.layer_kind.as_str(),
                active_experts = layer_timing.active_experts,
                expert_deferred = false,
                bytes_read = expert_delta.bytes_read,
                "flashmoe layer complete"
            );
            if report_layer_progress {
                report_generation_progress(&progress, || {
                    format!(
                        "forward layer complete position={} layer={}/{} attention_ms={} routing_ms={} expert_io_ms={} expert_read_ms={} expert_compute_ms={} combine_norm_ms={} total_ms={}",
                        position,
                        layer + 1,
                        self.config.num_hidden_layers,
                        layer_timing.buckets.attention_projection.as_millis(),
                        layer_timing.buckets.routing.as_millis(),
                        layer_timing.buckets.expert_io.as_millis(),
                        layer_timing.buckets.expert_read.as_millis(),
                        layer_timing.buckets.expert_compute.as_millis(),
                        layer_timing.buckets.combine_norm.as_millis(),
                        layer_started.elapsed().as_millis()
                    )
                });
            }
            if let Some(timing) = timing.as_deref_mut() {
                timing.buckets.add(layer_timing.buckets);
                timing.layers.push(layer_timing);
            }
        }
        if let Some(pending) = deferred_expert_phase.take() {
            let wait_span = trace_span!(
                target: "flashmoe::perf",
                "expert_wait",
                token_position = position,
                completed_layer = self.config.num_hidden_layers.saturating_sub(1),
                final_wait = true
            );
            let _wait_span = wait_span.enter();
            let wait_started = OptionalInstant::now(record_detailed_timing);
            let output = pending.wait()?;
            let wait_elapsed = wait_started.elapsed();
            trace!(
                target: "flashmoe::perf",
                token_position = position,
                completed_layer = self.config.num_hidden_layers.saturating_sub(1),
                "flashmoe deferred expert wait complete"
            );
            if let Some(timing) = timing.as_deref_mut() {
                timing.buckets.deferred_wait += wait_elapsed;
                if let Some(previous_layer) = timing.layers.last_mut() {
                    previous_layer.buckets.deferred_wait += wait_elapsed;
                    previous_layer.buckets.total_wall += wait_elapsed;
                }
            }
            token_state.apply_declared_expert_phase(
                output,
                FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
            )?;
        }
        token_state.clear_next_layer_normed();

        let combine_started = OptionalInstant::now(record_detailed_timing);
        token_state.replace_hidden(
            self.rms_norm_with_model_weight("model.norm.weight", token_state.hidden())?,
        );
        if record_generated {
            kv_cache.record_generated_token_record(FlashMoeGeneratedTokenRecord::new(
                position, previous,
            ))?;
        }
        if let Some(timing) = timing {
            timing.buckets.combine_norm += combine_started.elapsed();
            timing.buckets.total_wall = token_started.elapsed();
        }
        let (hidden, recurrent_value) = token_state.into_hidden_and_recurrent();
        Ok(QwenTokenExecutionOutput {
            hidden,
            recurrent_value,
        })
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    pub(super) fn forward_token_input(
        &mut self,
        input: FlashMoeTokenInput<'_>,
        kv_cache: &mut KvCache,
        position: usize,
        record_generated: bool,
        timing: Option<&mut FlashMoeTokenTiming>,
        progress: GenerationProgress<'_>,
    ) -> Result<Vec<f32>> {
        let _ = (
            input,
            kv_cache,
            position,
            record_generated,
            timing,
            progress,
        );
        bail!(
            "FlashMoe unsupported scheduled token execution: the resolved graph requires Apple Silicon Metal"
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn full_attention_output_values(
        &self,
        layer: usize,
        normed: &[f32],
        deferred_input: Option<MetalStateBuffer>,
        kv_cache: &mut KvCache,
        position: usize,
        rope_position: MropePosition,
        runtime: &DenseTransformerRuntime,
        mut attention_buckets: Option<&mut FlashMoeTimingBuckets>,
    ) -> Result<Vec<f32>> {
        let layout = runtime.full_attention_layout(layer)?;
        let subphase_started = OptionalInstant::now(attention_buckets.is_some());
        let input_requests = full_attention_input_projection_requests(
            layer,
            layout.q_projection_width,
            layout.kv_width,
        )?;
        let input_specs = input_requests.requests();
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let mut projections = if let Some(input) = deferred_input {
            self.dense.project_resident_tensors_from_metal_input(
                &self.metal,
                &input_specs,
                input.buffer(),
                input.len(),
            )?
        } else {
            self.dense
                .project_resident_tensors_from_cpu_input(&self.metal, &input_specs, normed)?
        };
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let mut projections: Vec<Vec<f32>> = {
            bail!(
                "FlashMoe unsupported scheduled Qwen3.5 CMD1 path at layer {layer}: the resolved implementation requires Apple Silicon Metal"
            )
        };
        let v = projections
            .pop()
            .context("missing batched self_attn.v_proj result")?;
        let mut k = projections
            .pop()
            .context("missing batched self_attn.k_proj result")?;
        let q_projected = projections
            .pop()
            .context("missing batched self_attn.q_proj result")?;
        if let Some(buckets) = attention_buckets.as_deref_mut() {
            buckets.attention_input_projection += subphase_started.elapsed();
        }

        let subphase_started = OptionalInstant::now(attention_buckets.is_some());
        let (mut q, q_gate) = split_q_projection(q_projected, layout)?;

        let q_norm_name = layer_norm_tensor_name(layer, "self_attn.q_norm");
        let k_norm_name = layer_norm_tensor_name(layer, "self_attn.k_norm");

        let q_norm_w = self
            .model_norm_weight(&q_norm_name, layout.head_dim)?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported Qwen full-attention CMD1 at layer {layer}: required Q norm tensor {q_norm_name} is unavailable"
                )
            })?;
        let k_norm_w = self
            .model_norm_weight(&k_norm_name, layout.head_dim)?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported Qwen full-attention CMD1 at layer {layer}: required K norm tensor {k_norm_name} is unavailable"
                )
            })?;
        let theta = self.config.rope_theta.unwrap_or_else(|| {
            if layout.q_layout == FullAttentionQLayout::Gated {
                10_000_000.0
            } else {
                1_000_000.0
            }
        });
        apply_full_attention_qk_norm_and_rotary(
            &mut q,
            &mut k,
            layout,
            rope_position,
            theta,
            self.config.text_mrope_section(),
            Some(&q_norm_w),
            Some(&k_norm_w),
        )?;
        if let Some(buckets) = attention_buckets.as_deref_mut() {
            buckets.attention_misc += subphase_started.elapsed();
        }

        let subphase_started = OptionalInstant::now(attention_buckets.is_some());
        let kv_record = FlashMoeFullAttentionKvRecord::new(position, layer, k, v);
        let attention_output =
            self.resolve_full_attention_kv_state(position, layer, layout, &kv_record)?;
        kv_cache.record_kv_record(kv_record)?;
        let attention_output =
            attention_output.validate_execution_state(layer, position, layout.kv_width)?;
        if attention_output.implementation() != ScheduledAttentionMathImplementation::CpuKvCache {
            bail!(
                "Qwen full attention at layer {layer} resolved an incompatible scheduled implementation"
            );
        }
        let records = kv_cache.keys_values(position, layer)?;
        if records.len() != position + 1 {
            bail!(
                "Qwen scalar attention layer {layer} expected {} KV rows through position {position}, found {}",
                position + 1,
                records.len()
            );
        }
        let mut keys = Vec::with_capacity(records.len() * layout.kv_width);
        let mut values = Vec::with_capacity(records.len() * layout.kv_width);
        for (key, value) in records {
            keys.extend_from_slice(key);
            values.extend_from_slice(value);
        }
        // Scalar qualification and the layer-major graph deliberately share
        // this query-independent causal-attention implementation. Each query
        // observes the same causal KV prefix, regardless of traversal order.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let mut attended = self.metal.qwen_causal_attention_rows(
            &q,
            &keys,
            &values,
            1,
            position,
            layout.num_q_heads,
            layout.kv_heads,
            layout.head_dim,
        )?;
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let mut attended = kv_cache.causal_attention(
            position,
            layer,
            &q,
            layout.num_q_heads,
            layout.kv_heads,
            layout.head_dim,
        )?;

        if let Some(q_gate) = q_gate {
            for (value, gate) in attended.iter_mut().zip(q_gate.iter()) {
                *value *= qwen_attention_sigmoid(*gate);
            }
        }
        if let Some(buckets) = attention_buckets {
            buckets.attention_kernel += subphase_started.elapsed();
        }
        Ok(attended)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn full_attention_output_matrix_values(
        &self,
        layer: usize,
        rows: &[QwenTokenExecutionOutput],
        normed: MetalBatchProjectionInput<'_>,
        start_position: usize,
        kv_cache: &mut KvCache,
    ) -> Result<MetalQwenAttentionRows> {
        if rows.is_empty() {
            bail!("Qwen layer-major full attention requires at least one row");
        }
        let runtime = &self.runtime;
        let layout = runtime.full_attention_layout(layer)?;
        let expected_normed = rows
            .len()
            .checked_mul(runtime.width)
            .context("Qwen layer-major normalized input size overflow")?;
        if normed.len() != expected_normed {
            bail!(
                "Qwen layer-major normalized input has {} values, expected {expected_normed} at layer {layer}",
                normed.len()
            );
        }

        let input_requests = full_attention_input_projection_requests(
            layer,
            layout.q_projection_width,
            layout.kv_width,
        )?;
        let input_specs = input_requests.requests();
        let mut projections = Vec::with_capacity(3);
        for spec in input_specs {
            projections.push(
                self.dense
                    .resident_mmap_projection(spec.tensor_name, spec.output_width, runtime.width)?
                    .with_context(|| {
                        format!(
                            "missing Qwen full-attention graph projection {}",
                            spec.tensor_name
                        )
                    })?,
            );
        }
        let projections: [ResidentMmapMatvecProjection; 3] =
            projections.try_into().map_err(|projections: Vec<_>| {
                anyhow::anyhow!(
                    "Qwen full-attention graph resolved {} projections, expected three",
                    projections.len()
                )
            })?;

        let q_norm_name = layer_norm_tensor_name(layer, "self_attn.q_norm");
        let k_norm_name = layer_norm_tensor_name(layer, "self_attn.k_norm");
        let q_norm_w = self
            .model_norm_weight(&q_norm_name, layout.head_dim)?
            .with_context(|| format!("missing Qwen layer-major Q norm {q_norm_name}"))?;
        let k_norm_w = self
            .model_norm_weight(&k_norm_name, layout.head_dim)?
            .with_context(|| format!("missing Qwen layer-major K norm {k_norm_name}"))?;
        let theta = self.config.rope_theta.unwrap_or_else(|| {
            if layout.q_layout == FullAttentionQLayout::Gated {
                10_000_000.0
            } else {
                1_000_000.0
            }
        });
        let rotary_half = layout.rotary_dim / 2;
        let mut rope_sin = Vec::with_capacity(rows.len() * rotary_half);
        let mut rope_cos = Vec::with_capacity(rows.len() * rotary_half);
        for row_index in 0..rows.len() {
            let position = start_position + row_index;
            let rotations = split_half_rope_rotations(
                FlashMoeTokenInput::text(0, position).rope_position(),
                layout.head_dim,
                layout.rotary_dim,
                theta,
                self.config.text_mrope_section(),
            );
            if rotations.len() != rotary_half {
                bail!(
                    "Qwen full-attention layer {layer} produced {} rotations, expected {rotary_half}",
                    rotations.len()
                );
            }
            for (sin, cos) in rotations {
                rope_sin.push(sin);
                rope_cos.push(cos);
            }
        }
        let records = kv_cache.keys_values(start_position.saturating_sub(1), layer)?;
        if records.len() != start_position {
            bail!(
                "Qwen full-attention layer {layer} expected {start_position} prefix KV rows, found {}",
                records.len()
            );
        }
        let mut prefix_keys = Vec::with_capacity(records.len() * layout.kv_width);
        let mut prefix_values = Vec::with_capacity(records.len() * layout.kv_width);
        for (key, value) in records {
            if key.len() != layout.kv_width || value.len() != layout.kv_width {
                bail!(
                    "Qwen full-attention layer {layer} encountered prefix KV widths {}/{}, expected {}",
                    key.len(),
                    value.len(),
                    layout.kv_width
                );
            }
            prefix_keys.extend_from_slice(key);
            prefix_values.extend_from_slice(value);
        }
        let output = self.metal.qwen_full_attention_graph(
            &projections,
            normed,
            rows.len(),
            start_position,
            layout,
            &q_norm_w,
            &k_norm_w,
            &rope_sin,
            &rope_cos,
            &prefix_keys,
            &prefix_values,
        )?;
        let expected_current = rows
            .len()
            .checked_mul(layout.kv_width)
            .context("Qwen full-attention current KV size overflow")?;
        if output.current_keys().len() != expected_current
            || output.current_values().len() != expected_current
        {
            bail!("Qwen full-attention graph returned incompatible current KV geometry");
        }
        for row_index in 0..rows.len() {
            let position = start_position + row_index;
            let start = row_index * layout.kv_width;
            let end = start + layout.kv_width;
            let kv_record = FlashMoeFullAttentionKvRecord::new(
                position,
                layer,
                output.current_keys()[start..end].to_vec(),
                output.current_values()[start..end].to_vec(),
            );
            self.resolve_full_attention_kv_state(position, layer, layout, &kv_record)?;
            kv_cache.record_kv_record(kv_record)?;
        }
        Ok(output.into_attention())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn post_attention_output_matrix_values(
        &self,
        layer: usize,
        rows: usize,
        attention_width: usize,
        out_proj_name: &str,
        attention: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
    ) -> Result<MetalLayerMajorPostAttention> {
        let post_norm_name = layer_norm_tensor_name(layer, "post_attention_layernorm");
        let post_norm_weight = self
            .model_norm_weight(&post_norm_name, self.runtime.width)?
            .with_context(|| format!("missing Qwen layer-major norm {post_norm_name}"))?;
        let projections = self.dense.layer_major_post_attention_projections(
            layer,
            self.scheduler.experts_per_layer(),
            out_proj_name,
            attention_width,
            self.runtime.width,
            self.scheduler.active_experts(),
        )?;
        self.metal.qwen_post_attention_matrix(
            &projections.out_proj,
            &projections.router,
            rows,
            attention_width,
            self.runtime.width,
            attention,
            residual,
            &post_norm_weight,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn forward_qwen_layer_major_matrix(
        &mut self,
        layer: usize,
        rows: &mut [QwenTokenExecutionOutput],
        previous: Option<&MetalQwenPrefillLayerOutput>,
        start_position: usize,
        kv_cache: &mut KvCache,
        record_recurrent_trace: bool,
        row_timings: &mut [Option<FlashMoeTokenTiming>],
    ) -> Result<MetalQwenPrefillLayerOutput> {
        if rows.is_empty() || rows.len() != row_timings.len() {
            bail!("Qwen layer-major layer requires aligned non-empty rows and timings");
        }
        let layer_started = Instant::now();
        let width = self.runtime.width;
        let hidden_values = rows
            .len()
            .checked_mul(width)
            .context("Qwen layer-major hidden matrix size overflow")?;
        if let Some(previous) = previous {
            if previous.layer() + 1 != layer
                || previous.hidden().rows() != rows.len()
                || previous.hidden().cols() != width
                || previous
                    .next_normed()
                    .is_none_or(|normed| normed.rows() != rows.len() || normed.cols() != width)
            {
                bail!(
                    "Qwen layer-major device input does not match layer {layer} geometry {}x{width}",
                    rows.len()
                );
            }
        } else {
            for row in rows.iter() {
                if row.hidden.len() != width {
                    bail!(
                        "Qwen layer-major row width {} does not match {width} at layer {layer}",
                        row.hidden.len()
                    );
                }
            }
        }

        let post_norm_name = layer_norm_tensor_name(layer, "post_attention_layernorm");
        let post_norm_weight = self
            .model_norm_weight(&post_norm_name, width)?
            .with_context(|| format!("missing Qwen layer-major norm {post_norm_name}"))?;
        let (post, layer_kind) = if self.runtime.is_linear_attention_layer(layer) {
            let layout = self.runtime.linear_attention_layout(layer)?;
            let bindings = self.linear_attention_weights.require(layer)?;
            if let Some(previous) = previous {
                let normed = previous
                    .next_normed()
                    .context("Qwen layer-major device input is missing its prepared norm")?;
                (
                    self.metal.qwen_linear_attention_graph(
                        layout,
                        bindings,
                        rows.len(),
                        width,
                        MetalBatchProjectionInput::Buffer {
                            buffer: normed.buffer(),
                            len: normed.values(),
                        },
                        MetalBatchProjectionInput::Buffer {
                            buffer: previous.hidden().buffer(),
                            len: previous.hidden().values(),
                        },
                        &post_norm_weight,
                    )?,
                    FlashMoeLayerKind::LinearAttention,
                )
            } else {
                let mut residual = Vec::with_capacity(hidden_values);
                let mut normed = Vec::with_capacity(hidden_values);
                let input_norm_name = layer_norm_tensor_name(layer, "input_layernorm");
                for row in rows.iter() {
                    residual.extend_from_slice(&row.hidden);
                    normed.extend(self.rms_norm_with_model_weight(&input_norm_name, &row.hidden)?);
                }
                (
                    self.metal.qwen_linear_attention_graph(
                        layout,
                        bindings,
                        rows.len(),
                        width,
                        MetalBatchProjectionInput::Cpu(&normed),
                        MetalBatchProjectionInput::Cpu(&residual),
                        &post_norm_weight,
                    )?,
                    FlashMoeLayerKind::LinearAttention,
                )
            }
        } else {
            let mut residual_cpu = Vec::new();
            let mut normed_cpu = Vec::new();
            let (residual, normed) = if let Some(previous) = previous {
                let next_normed = previous
                    .next_normed()
                    .context("Qwen full-attention device input is missing its prepared norm")?;
                (
                    MetalBatchProjectionInput::Buffer {
                        buffer: previous.hidden().buffer(),
                        len: previous.hidden().values(),
                    },
                    MetalBatchProjectionInput::Buffer {
                        buffer: next_normed.buffer(),
                        len: next_normed.values(),
                    },
                )
            } else {
                residual_cpu.reserve(hidden_values);
                normed_cpu.reserve(hidden_values);
                let input_norm_name = layer_norm_tensor_name(layer, "input_layernorm");
                for row in rows.iter() {
                    residual_cpu.extend_from_slice(&row.hidden);
                    normed_cpu
                        .extend(self.rms_norm_with_model_weight(&input_norm_name, &row.hidden)?);
                }
                (
                    MetalBatchProjectionInput::Cpu(&residual_cpu),
                    MetalBatchProjectionInput::Cpu(&normed_cpu),
                )
            };
            let layout = self.runtime.full_attention_layout(layer)?;
            let attention = self.full_attention_output_matrix_values(
                layer,
                rows,
                normed,
                start_position,
                kv_cache,
            )?;
            (
                self.post_attention_output_matrix_values(
                    layer,
                    rows.len(),
                    layout.q_width,
                    &attention_tensor_name(layer, "o_proj"),
                    MetalBatchProjectionInput::Buffer {
                        buffer: attention.values().buffer(),
                        len: attention.values().values(),
                    },
                    residual,
                )?,
                FlashMoeLayerKind::FullAttention,
            )
        };
        let experts = self.scheduler.experts_per_layer();
        let active = self.scheduler.active_experts();
        if post.router_scores().len() != rows.len() * experts {
            bail!(
                "Qwen layer-major router matrix has {} values, expected {}x{experts}",
                post.router_scores().len(),
                rows.len()
            );
        }
        let row_routes = post
            .router_scores()
            .chunks_exact(experts)
            .map(|scores| routing_softmax_top_k(scores, active))
            .collect::<Vec<_>>();
        let scheduled = self
            .scheduler
            .resolve_layer_major_experts(layer, &row_routes)?;
        let shared_phase = match self.shared_expert_weights.layer(layer)? {
            SharedExpertLayerWeights::Resident(shared) => {
                ScheduledSharedExpertPhaseRef::Resident(shared)
            }
            SharedExpertLayerWeights::None => ScheduledSharedExpertPhaseRef::None,
        };
        let next_norm_weight = if layer + 1 < self.config.num_hidden_layers {
            let next_norm_name = layer_norm_tensor_name(layer + 1, "input_layernorm");
            Some(
                self.model_norm_weight(&next_norm_name, width)?
                    .with_context(|| format!("missing Qwen layer-major norm {next_norm_name}"))?,
            )
        } else {
            None
        };
        let layer_output = self.metal.qwen_layer_major_experts(
            &scheduled,
            &post,
            shared_phase,
            next_norm_weight.as_deref(),
        )?;
        if layer_output.layer() != layer
            || layer_output.hidden().rows() != rows.len()
            || layer_output.hidden().cols() != width
            || layer_output.next_normed().is_some() != next_norm_weight.is_some()
        {
            bail!("Qwen layer-major expert output does not match the requested layer geometry");
        }
        let elapsed = layer_started.elapsed();
        let per_row_elapsed = elapsed / u32::try_from(rows.len()).unwrap_or(u32::MAX);
        if record_recurrent_trace {
            let mix_hashes = scheduled.route_mix_hashes().collect::<Vec<_>>();
            for row_index in 0..rows.len() {
                let mut recurrent = FlashMoeRecurrentState::new(rows[row_index].recurrent_value);
                let route_start = row_index * active;
                for route in route_start..route_start + active {
                    recurrent.mix_active_expert(mix_hashes[route], scheduled.weights()[route]);
                }
                rows[row_index].recurrent_value = recurrent.value();
            }
            kv_cache.record_layer_state_values(
                start_position,
                layer,
                rows.iter().map(|row| row.recurrent_value),
            )?;
        }
        for row_index in 0..rows.len() {
            if let Some(timing) = row_timings[row_index].as_mut() {
                let mut layer_timing = FlashMoeLayerTiming {
                    layer,
                    layer_kind,
                    active_experts: active,
                    dimensions: self.layer_dimensions(layer),
                    buckets: FlashMoeTimingBuckets::default(),
                };
                layer_timing.buckets.total_wall = per_row_elapsed;
                timing.buckets.total_wall += per_row_elapsed;
                timing.layers.push(layer_timing);
            }
        }
        Ok(layer_output)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn mla_attention_output_values(
        &self,
        layer: usize,
        normed: &[f32],
        deferred_input: Option<MetalStateBuffer>,
        kv_cache: &mut KvCache,
        position: usize,
        rope_position: MropePosition,
        runtime: &DenseTransformerRuntime,
        post_attention: Option<GlmMlaPostAttentionRequest<'_>>,
        mut attention_buckets: Option<&mut FlashMoeTimingBuckets>,
    ) -> Result<MlaAttentionOutput> {
        let layout = runtime.mla_attention_layout(layer)?;
        let subphase_started = OptionalInstant::now(attention_buckets.is_some());
        let q_norm_name = layer_norm_tensor_name(layer, "self_attn.q_a_layernorm");
        let q_norm = self
            .model_norm_weight(&q_norm_name, layout.q_lora_rank)?
            .with_context(|| format!("missing GLM MLA norm tensor {q_norm_name}"))?;
        let kv_norm_name = layer_norm_tensor_name(layer, "self_attn.kv_a_layernorm");
        let kv_norm = self
            .model_norm_weight(&kv_norm_name, layout.kv_lora_rank)?
            .with_context(|| format!("missing GLM MLA norm tensor {kv_norm_name}"))?;
        let norm_epsilon = self
            .config
            .glm_mla_norm_epsilon()
            .context("GLM MLA execution requires its projection-norm epsilon")?;
        let scheduled_attention = self.scheduler.resolve_attention_math(layer, position)?;
        let scheduled_output =
            scheduled_attention.resolve_mla_kv_state(FlashMoeMlaKvState::cpu_visible(
                position,
                layer,
                layout.kv_lora_rank,
                layout.qk_rope_head_dim,
            ))?;
        let scheduled_implementation = scheduled_output.implementation();
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if scheduled_implementation
            == ScheduledAttentionMathImplementation::MetalQ4GlmMlaAbsorbedAttention
        {
            let records = kv_cache.mla_records(position, layer)?;
            if records.len() != position {
                bail!(
                    "GLM MLA layer {layer} position {position} expected {position} previous cache records, found {}",
                    records.len()
                );
            }
            let theta = self.config.rope_theta.unwrap_or(10_000.0);
            let rope_half = layout.qk_rope_head_dim / 2;
            let mut rope_cos = Vec::with_capacity(rope_half);
            let mut rope_sin = Vec::with_capacity(rope_half);
            for pair in 0..rope_half {
                let frequency = theta.powf(-((2 * pair) as f64) / layout.qk_rope_head_dim as f64);
                let angle = rope_position.temporal as f64 * frequency;
                let (sin, cos) = angle.sin_cos();
                rope_cos.push(cos as f32);
                rope_sin.push(sin as f32);
            }
            let input = deferred_input
                .map(|input| MetalBatchProjectionInput::Buffer {
                    buffer: input.buffer(),
                    len: input.len(),
                })
                .unwrap_or(MetalBatchProjectionInput::Cpu(normed));
            let fused = self.dense.glm_mla_fused_attention_with_metal(
                &self.metal,
                layer,
                layout,
                input,
                &q_norm,
                &kv_norm,
                norm_epsilon,
                &records,
                &rope_cos,
                &rope_sin,
                post_attention,
            )?;
            kv_cache.record_mla_kv(position, layer, fused.latent, fused.rotary)?;
            if let Some(buckets) = attention_buckets {
                buckets.attention_kernel += subphase_started.elapsed();
            }
            return Ok(match fused.terminal {
                MetalGlmMlaFusedAttentionTerminal::Attention(attention) => {
                    MlaAttentionOutput::Values(attention)
                }
                MetalGlmMlaFusedAttentionTerminal::PostAttention(prep) => {
                    MlaAttentionOutput::MetalPostAttention(prep)
                }
            });
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let (mut query, mut compressed) = {
            let input = deferred_input
                .map(|input| MetalBatchProjectionInput::Buffer {
                    buffer: input.buffer(),
                    len: input.len(),
                })
                .unwrap_or(MetalBatchProjectionInput::Cpu(normed));
            self.dense.glm_mla_input_projections_with_metal(
                &self.metal,
                layer,
                layout,
                input,
                &q_norm,
                &kv_norm,
                norm_epsilon,
            )?
        };
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let (mut query, mut compressed): (Vec<f32>, Vec<f32>) = {
            let _ = (deferred_input, normed, q_norm, kv_norm, norm_epsilon);
            bail!("GLM MLA execution requires Apple Silicon Metal projections")
        };
        if let Some(buckets) = attention_buckets.as_deref_mut() {
            buckets.attention_input_projection += subphase_started.elapsed();
        }

        let subphase_started = OptionalInstant::now(attention_buckets.is_some());
        let theta = self.config.rope_theta.unwrap_or(10_000.0);
        for head in 0..layout.num_heads {
            let start = head * layout.qk_head_dim + layout.qk_nope_head_dim;
            apply_rotary_interleaved_to_split_half(
                &mut query[start..start + layout.qk_rope_head_dim],
                rope_position.temporal,
                layout.qk_rope_head_dim,
                theta,
            )?;
        }
        let mut rotary_key = compressed.split_off(layout.kv_lora_rank);
        apply_rotary_interleaved_to_split_half(
            &mut rotary_key,
            rope_position.temporal,
            layout.qk_rope_head_dim,
            theta,
        )?;
        kv_cache.record_mla_kv(position, layer, compressed, rotary_key)?;
        if !matches!(
            scheduled_implementation,
            ScheduledAttentionMathImplementation::CpuGlmMlaWeightAbsorption
                | ScheduledAttentionMathImplementation::MetalQ4GlmMlaAbsorbedAttention
        ) {
            bail!(
                "GLM MLA layer {layer} resolved unexpected scheduled attention implementation {:?}",
                scheduled_implementation
            );
        }
        if let Some(buckets) = attention_buckets.as_deref_mut() {
            buckets.attention_misc += subphase_started.elapsed();
        }

        let subphase_started = OptionalInstant::now(attention_buckets.is_some());
        let records = kv_cache.mla_records(position, layer)?;
        let output = match scheduled_implementation {
            ScheduledAttentionMathImplementation::CpuGlmMlaWeightAbsorption => self
                .dense
                .mla_absorbed_attention(layer, layout, &query, &records)?,
            ScheduledAttentionMathImplementation::MetalQ4GlmMlaAbsorbedAttention => self
                .dense
                .mla_absorbed_attention_metal(&self.metal, layer, layout, &query, &records)?,
            ScheduledAttentionMathImplementation::CpuKvCache => {
                bail!("GLM MLA layer {layer} resolved the full-attention CPU KV implementation")
            }
        };
        if let Some(buckets) = attention_buckets {
            buckets.attention_kernel += subphase_started.elapsed();
        }
        Ok(MlaAttentionOutput::Values(output))
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn forward_glm_dense_layer(
        &self,
        layer: usize,
        token_state: &mut FlashMoeTokenState,
        kv_cache: &mut KvCache,
        position: usize,
        rope_position: MropePosition,
        runtime: &DenseTransformerRuntime,
        mut buckets: Option<&mut FlashMoeTimingBuckets>,
    ) -> Result<()> {
        let norm_started = OptionalInstant::now(buckets.is_some());
        let input_norm_name = layer_norm_tensor_name(layer, "input_layernorm");
        let normed = self.rms_norm_with_model_weight(&input_norm_name, token_state.hidden())?;
        if let Some(buckets) = buckets.as_deref_mut() {
            buckets.combine_norm += norm_started.elapsed();
        }
        let attention = self.mla_attention_output_values(
            layer,
            &normed,
            None,
            kv_cache,
            position,
            rope_position,
            runtime,
            None,
            buckets.as_deref_mut(),
        )?;
        let MlaAttentionOutput::Values(attention) = attention else {
            bail!("dense GLM MLA layer {layer} unexpectedly produced sparse post-attention state")
        };
        let projection_started = OptionalInstant::now(buckets.is_some());
        let o_name = attention_tensor_name(layer, "o_proj");
        let projected = self
            .dense
            .project_resident_tensors_from_cpu_input(
                &self.metal,
                &[DenseProjectionRequest::new(&o_name, runtime.width)?],
                &attention,
            )?
            .pop()
            .context("missing GLM MLA output projection")?;
        add_in_place(token_state.hidden_mut(), &projected);
        if let Some(buckets) = buckets.as_deref_mut() {
            buckets.attention_projection += projection_started.elapsed();
        }

        let mlp_started = OptionalInstant::now(buckets.is_some());
        let post_norm_name = layer_norm_tensor_name(layer, "post_attention_layernorm");
        let post_normed = self.rms_norm_with_model_weight(&post_norm_name, token_state.hidden())?;
        let intermediate = self
            .config
            .intermediate_size
            .with_context(|| format!("GLM dense MLP layer {layer} is missing intermediate_size"))?;
        let gate_name = format!("model.layers.{layer}.mlp.gate_proj.weight");
        let up_name = format!("model.layers.{layer}.mlp.up_proj.weight");
        let mut gate_up = self.dense.project_resident_tensors_from_cpu_input(
            &self.metal,
            &[
                DenseProjectionRequest::new(&gate_name, intermediate)?,
                DenseProjectionRequest::new(&up_name, intermediate)?,
            ],
            &post_normed,
        )?;
        let up = gate_up
            .pop()
            .context("missing GLM dense MLP up projection")?;
        let mut activated = gate_up
            .pop()
            .context("missing GLM dense MLP gate projection")?;
        for (gate, up) in activated.iter_mut().zip(up) {
            *gate = *gate * sigmoid(*gate) * up;
        }
        let down_name = format!("model.layers.{layer}.mlp.down_proj.weight");
        let down = self
            .dense
            .project_resident_tensors_from_cpu_input(
                &self.metal,
                &[DenseProjectionRequest::new(&down_name, runtime.width)?],
                &activated,
            )?
            .pop()
            .context("missing GLM dense MLP down projection")?;
        add_in_place(token_state.hidden_mut(), &down);
        if let Some(buckets) = buckets {
            buckets.expert_compute += mlp_started.elapsed();
        }
        Ok(())
    }

    fn resolve_full_attention_kv_state(
        &self,
        position: usize,
        layer: usize,
        _layout: FullAttentionLayout,
        kv_record: &FlashMoeFullAttentionKvRecord,
    ) -> Result<ScheduledAttentionMathOutput> {
        let scheduled_attention = self.scheduler.resolve_attention_math(layer, position)?;
        scheduled_attention.resolve_kv_state(kv_record.state(FlashMoeStatePlacement::CpuVisible))
    }

    fn route_layer(
        &self,
        layer: usize,
        normed: &[f32],
        active_experts: usize,
    ) -> Result<ScheduledRoutingCommand> {
        let projection = self.dense.router_score_projection_descriptor(
            layer,
            self.scheduler.experts_per_layer(),
            normed.len(),
        )?;
        let router_score_command =
            self.scheduler
                .resolve_router_score_projection(layer, projection, normed.len())?;
        if active_experts != self.scheduler.active_experts() {
            bail!(
                "FlashMoe scheduled routing for layer {layer} carries K={active_experts}, but the resolved graph requires K={}",
                self.scheduler.active_experts()
            );
        }
        let execution = router_score_command.projection_execution()?;
        let score_plan = execution.score_plan(normed.len())?;
        let scores = self.dense.router_scores(score_plan, normed)?;
        router_score_command.into_routing_command(scores)
    }

    fn router_correction_bias(&self, layer: usize) -> Result<Option<Vec<f32>>> {
        if self.config.glm.is_none() {
            return Ok(None);
        }
        let tensor_name = format!("model.layers.{layer}.mlp.gate.e_score_correction_bias");
        let bias = self
            .dense
            .read_full_tensor_f32(&tensor_name)?
            .with_context(|| format!("missing GLM router correction bias {tensor_name}"))?;
        if bias.len() != self.scheduler.experts_per_layer() {
            bail!(
                "GLM router correction bias {tensor_name} has {} values, expected {}",
                bias.len(),
                self.scheduler.experts_per_layer()
            );
        }
        Ok(Some(bias))
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn linear_attention_post_attention_prep_with_metal(
        &self,
        layer: usize,
        normed: &[f32],
        deferred_input: Option<MetalStateBuffer>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        runtime: &DenseTransformerRuntime,
        mut attention_buckets: Option<&mut FlashMoeTimingBuckets>,
    ) -> Result<MetalPostAttentionPrep> {
        let metal = &self.metal;
        let layout = runtime.linear_attention_layout(layer)?;
        let bindings = self.linear_attention_weights.require(layer)?;
        let residual_len = residual.len();
        if residual_len != runtime.width || residual_len != post_norm_weight.len() {
            bail!(
                "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD2 path at layer {layer}: residual width {residual_len}, runtime width {}, and norm width {} do not match",
                runtime.width,
                post_norm_weight.len()
            );
        }
        let projection_input = if let Some(input) = deferred_input {
            MetalBatchProjectionInput::Buffer {
                buffer: input.buffer(),
                len: input.len(),
            }
        } else if !normed.is_empty() {
            MetalBatchProjectionInput::Cpu(normed)
        } else {
            bail!(
                "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD1 path at layer {layer}: neither deferred Metal nor CPU normed input is available"
            );
        };
        let started = OptionalInstant::now(attention_buckets.is_some());
        let prep = self.dense.linear_attention_post_attention_prep_with_metal(
            metal,
            layout,
            bindings,
            projection_input,
            residual,
            post_norm_weight,
            self.scheduler.active_experts(),
        )?;
        if let Some(buckets) = attention_buckets.as_deref_mut() {
            buckets.attention_input_projection += started.elapsed();
        }
        Ok(prep)
    }
}
mod generation;
