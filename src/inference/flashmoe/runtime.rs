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
use super::constraints::NativeToolConstraint;
use super::deepseek::{DeepSeekV4Config, DeepSeekV4ExecutionGraph, is_deepseek_v4_flash};
use super::deepseek_session::{
    DeepSeekV4CheckpointKind, DeepSeekV4SessionCheckpoint, DeepSeekV4SessionStore,
};
use super::experts::ExpertSlotStore;
use super::math::*;
use super::metal::*;
use super::model_family::{QwenModelConfig, QwenMoeFamily, QwenMoeModelLayout};
use super::planning::{FlashMoePlan, ResolvedRoutingPolicy};
use super::scheduler::*;
use super::session_cache::FlashMoeDiskCache;
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
use super::scheduler::ScheduledSharedExpertPhaseRef as SharedExpertPhaseRef;

#[derive(Debug, Clone)]
pub(super) struct MetalExecutionFacade {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) inner: Arc<MetalExecutionContext>,
}

#[derive(Debug)]
pub(super) enum ExpertPhaseInput {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    MetalPostAttention(MetalPostAttentionPrep),
}

#[derive(Debug)]
enum MlaAttentionOutput {
    Values(Vec<f32>),
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
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

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[derive(Debug, Clone, Copy)]
struct GlmMlaPostAttentionRequest<'a>(std::marker::PhantomData<&'a ()>);

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

pub(super) type GenerationProgress<'a> = Option<Rc<RefCell<&'a mut dyn FnMut(String)>>>;

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
    pub(super) deepseek_graph: Option<Arc<DeepSeekV4ExecutionGraph>>,
    pub(super) linear_attention_weights: LinearAttentionWeightTable,
    pub(super) shared_expert_weights: SharedExpertWeightTable,
    pub(super) input_adapter_executor: FlashMoeInputAdapterExecutor,
    pub(super) session_cache: FlashMoeSessionCache,
    pub(super) deepseek_sessions: DeepSeekV4SessionStore<DeepSeekV4SessionSnapshot>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlashMoeLoadOptions {
    pub metal_working_set_limit_bytes: Option<usize>,
}

pub fn load(plan: &FlashMoePlan) -> Result<FlashMoeEngine> {
    load_with_options(plan, FlashMoeLoadOptions::default())
}

pub fn load_with_progress<F>(plan: &FlashMoePlan, progress: F) -> Result<FlashMoeEngine>
where
    F: FnMut(&'static str, Duration),
{
    load_with_options_and_progress(plan, FlashMoeLoadOptions::default(), progress)
}

pub fn load_with_options(
    plan: &FlashMoePlan,
    options: FlashMoeLoadOptions,
) -> Result<FlashMoeEngine> {
    load_with_options_and_progress(plan, options, |_, _| {})
}

pub fn load_with_options_and_progress<F>(
    plan: &FlashMoePlan,
    options: FlashMoeLoadOptions,
    mut progress: F,
) -> Result<FlashMoeEngine>
where
    F: FnMut(&'static str, Duration),
{
    let mut phase_started = Instant::now();
    let status = plan.cache_status()?;
    progress("cache_status", phase_started.elapsed());
    if !status.ready {
        bail!(
            "Flash-MoE cache is not ready for {}. Missing: {}. Found {} expert files totaling {} bytes. Run `pb pull {}` on ARM macOS to download and prepare the FlashMoe cache.",
            plan.model,
            format_missing(&status.missing),
            status.expert_files,
            status.expert_bytes,
            plan.model
        );
    }
    phase_started = Instant::now();
    let deepseek_config = if is_deepseek_v4_flash(&plan.model) {
        Some(DeepSeekV4Config::from_file(&plan.model_config)?)
    } else {
        None
    };
    let config = match &deepseek_config {
        Some(config) => config.shared_runtime_config(),
        None => QwenModelConfig::from_file(&plan.model_config)?,
    };
    progress("config", phase_started.elapsed());
    phase_started = Instant::now();
    let routing_policy = plan.routing_policy.resolve(&plan.model, &config)?;
    progress("routing_policy", phase_started.elapsed());
    phase_started = Instant::now();
    let model_layout = QwenMoeModelLayout::from_config(&plan.model, &config)?
        .with_scheduled_active_experts(routing_policy.active_experts)?;
    progress("model_layout", phase_started.elapsed());
    phase_started = Instant::now();
    let resolved_experts = ExpertSlotStore::resolve_from_metadata(
        plan.experts_dir.clone(),
        &model_layout,
        plan.quantization,
    )?;
    if resolved_experts.upgraded_pbq4_layers > 0 {
        tracing::info!(
            model = %plan.model,
            layers = resolved_experts.upgraded_pbq4_layers,
            "upgraded PBQ4 expert cache layers to fixed Q4 slots"
        );
    }
    progress("expert_cache_format", phase_started.elapsed());
    phase_started = Instant::now();
    let dense = DenseStore::open(
        plan.non_expert_weights.clone(),
        plan.tensor_manifest.clone(),
    )?;
    progress("dense_store", phase_started.elapsed());
    phase_started = Instant::now();
    let deepseek_graph = deepseek_config
        .map(|config| DeepSeekV4ExecutionGraph::from_registry(config, dense.registry(), dense.len))
        .transpose()?
        .map(Arc::new);
    if deepseek_graph.is_none() {
        validate_required_tensor_manifest(&config, dense.registry())?;
    }
    progress("manifest_validation", phase_started.elapsed());
    phase_started = Instant::now();
    let runtime = if deepseek_graph.is_some() {
        DenseTransformerRuntime::new(&config)
    } else {
        DenseTransformerRuntime::from_registry(&config, dense.registry())?
    };
    let attention_layers = if deepseek_graph.is_some() {
        vec![super::model_family::QwenMoeLayerKind::FullAttention; config.num_hidden_layers]
    } else {
        runtime.resolved_attention_layers()?
    };
    progress("runtime_layout", phase_started.elapsed());
    phase_started = Instant::now();
    let linear_attention_weights = if deepseek_graph.is_some() {
        LinearAttentionWeightTable::empty(config.num_hidden_layers)
    } else {
        dense.resolve_linear_attention_weight_table(
            &runtime.linear_attention,
            config.hidden_size,
            model_layout.experts_per_layer,
        )?
    };
    progress("linear_attention_weights", phase_started.elapsed());
    phase_started = Instant::now();
    let shared_expert_weights = if deepseek_graph.is_some() {
        SharedExpertWeightTable::none(config.num_hidden_layers)
    } else {
        dense.resolve_shared_expert_weight_table_from(
            config.num_hidden_layers,
            config.hidden_size,
            config.shared_experts(),
            config.shared_expert_intermediate_size(),
            config.first_sparse_layer(),
            config.glm.is_none(),
        )?
    };
    progress("shared_expert_weights", phase_started.elapsed());
    phase_started = Instant::now();
    let dense_layout = if deepseek_graph.is_some() {
        // DeepSeek's typed graph validates its exact F16/F32/I32/Q8 GGUF
        // mixture above. This legacy field is not consulted by its execution
        // implementation.
        ResidentDenseLayout::F16
    } else {
        dense.registry().resolve_resident_dense_layout()?
    };
    if matches!(
        model_layout.family,
        QwenMoeFamily::Qwen35A17B
            | QwenMoeFamily::Qwen3NextMoe
            | QwenMoeFamily::Qwen3Moe
            | QwenMoeFamily::Qwen3VlMoe
            | QwenMoeFamily::Glm52
    ) && dense_layout == ResidentDenseLayout::Q4
    {
        validate_qwen_q4_graph_bindings(
            model_layout.family,
            &config,
            &runtime,
            dense.registry(),
            dense.len,
        )?;
    }
    progress("dense_graph_bindings", phase_started.elapsed());
    phase_started = Instant::now();
    let input_adapter_executor =
        FlashMoeInputAdapterExecutor::from_plan(model_layout.family, plan, &config)?;
    let input_adapter = input_adapter_executor.capability()?;
    progress("vision_encoder", phase_started.elapsed());
    phase_started = Instant::now();
    let metal = MetalExecutionFacade::new(plan, &config, &runtime, &dense)?;
    if let Some(limit) = options.metal_working_set_limit_bytes {
        metal.set_working_set_limit_bytes(limit)?;
    }
    progress("metal_executor", phase_started.elapsed());
    phase_started = Instant::now();
    let experts = resolved_experts.store;
    let expert_storage = resolved_experts.descriptor;
    progress("expert_store", phase_started.elapsed());
    phase_started = Instant::now();
    let attention_math = if model_layout.family == QwenMoeFamily::DeepSeekV4Flash {
        FlashMoeAttentionMathCapability::DeepSeekV4HyperconnectionCompressedAttentionMetal
    } else if model_layout.family == QwenMoeFamily::Glm52
        && runtime.mla_attention.iter().all(|layout| {
            matches!(
                layout.map(|layout| layout.kv_projection),
                Some(MlaKvProjectionLayout::AbsorbedMultiLinear)
            )
        })
    {
        FlashMoeAttentionMathCapability::GlmMlaMetalQ4AbsorbedAttention
    } else if model_layout.family == QwenMoeFamily::Glm52 {
        FlashMoeAttentionMathCapability::GlmMlaCpuWeightAbsorption
    } else {
        FlashMoeAttentionMathCapability::QwenFullAttentionCpuKv
    };
    let resource_snapshot = metal.resource_snapshot();
    let expert_access = resolve_expert_access(
        model_layout.family,
        expert_storage,
        resource_snapshot.as_ref(),
    )?;
    tracing::info!(
        model = %plan.model,
        implementation = ?expert_access,
        expert_bytes = expert_storage.total_expert_bytes()?,
        working_set_limit_bytes = resource_snapshot
            .as_ref()
            .map_or(0, |snapshot| snapshot.working_set_limit_bytes),
        current_allocated_bytes = resource_snapshot
            .as_ref()
            .map_or(0, |snapshot| snapshot.current_allocated_bytes),
        "resolved FlashMoe expert access implementation"
    );
    let capability_plan = FlashMoeCapabilityPlan::resolve_with_attention_math_and_expert_access(
        &model_layout,
        input_adapter,
        dense_layout,
        expert_storage,
        &attention_layers,
        attention_math,
        expert_access,
        Some(metal.runtime_capabilities()),
    )?;
    let qwen_prefill_graph = capability_plan.qwen_prefill_graph;
    tracing::info!(
        model = %plan.model,
        implementation = qwen_prefill_graph.as_str(),
        dense_layout = dense_layout.as_str(),
        expert_layout = ?expert_storage.layout,
        "prepared FlashMoe Qwen prefill graph"
    );
    let scheduled_graph = FlashMoeScheduledGraph::from_capabilities(&capability_plan)?;
    let scheduler =
        FlashMoeExecutionScheduler::new_with_resident_binding(scheduled_graph, experts, |bytes| {
            metal.prepare_resident_expert_backing(bytes)
        })?;
    progress("capability_graph", phase_started.elapsed());
    phase_started = Instant::now();
    let tokenizer = QwenTokenizer::from_files(
        &plan.tokenizer,
        &plan.tokenizer_config,
        Some(&plan.chat_template),
    )?;
    progress("tokenizer", phase_started.elapsed());
    let session_cache =
        FlashMoeSessionCache::new(FlashMoeDiskCache::from_plan(plan, config.num_hidden_layers));
    Ok(FlashMoeEngine {
        plan: plan.clone(),
        scheduler,
        dense,
        tokenizer,
        metal,
        input_adapter_executor,
        config,
        model_layout,
        routing_policy,
        expert_access,
        qwen_prefill_graph,
        runtime,
        deepseek_graph,
        linear_attention_weights,
        shared_expert_weights,
        session_cache,
        deepseek_sessions: DeepSeekV4SessionStore::default(),
    })
}

fn format_missing(paths: &[std::path::PathBuf]) -> String {
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
    pub(super) fn qwen_linear_attention_matrix(
        &self,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        rows: usize,
        qkv: &[f32],
        z: &[f32],
        beta: &[f32],
        alpha: &[f32],
    ) -> Result<Vec<f32>> {
        self.inner
            .qwen_linear_attention_matrix(layout, bindings, rows, qkv, z, beta, alpha)
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
        attention: &[f32],
        residual: &[f32],
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
    pub fn persist_session_cache(&mut self, session_id: &str) -> Result<()> {
        if session_id.trim().is_empty() {
            return Ok(());
        }
        if self.deepseek_graph.is_some() {
            return Ok(());
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
        if let Some(graph) = self.deepseek_graph.clone() {
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
                *value *= sigmoid(*gate);
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
        normed: &[f32],
        start_position: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        if rows.is_empty() {
            bail!("Qwen layer-major full attention requires at least one row");
        }
        let runtime = &self.runtime;
        let layout = runtime.full_attention_layout(layer)?;
        for row in rows {
            if row.hidden.len() != runtime.width {
                bail!(
                    "Qwen layer-major hidden row has {} values, expected {} at layer {layer}",
                    row.hidden.len(),
                    runtime.width
                );
            }
        }
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
        let mut projections = self.dense.project_resident_tensors_from_cpu_matrix(
            &self.metal,
            &input_specs,
            rows.len(),
            runtime.width,
            normed,
        )?;
        let values = projections
            .pop()
            .context("missing layer-major self_attn.v_proj matrix")?;
        let keys = projections
            .pop()
            .context("missing layer-major self_attn.k_proj matrix")?;
        let queries = projections
            .pop()
            .context("missing layer-major self_attn.q_proj matrix")?;
        let expected_queries = rows
            .len()
            .checked_mul(layout.q_projection_width)
            .context("layer-major query matrix size overflow")?;
        let expected_kv = rows
            .len()
            .checked_mul(layout.kv_width)
            .context("layer-major KV matrix size overflow")?;
        if queries.len() != expected_queries
            || keys.len() != expected_kv
            || values.len() != expected_kv
        {
            bail!(
                "Qwen layer-major projection geometry mismatch at layer {layer}: q={} expected={expected_queries}, k={} v={} expected_kv={expected_kv}",
                queries.len(),
                keys.len(),
                values.len()
            );
        }

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
        let mut normalized_queries = Vec::with_capacity(rows.len() * layout.q_width);
        let mut query_gates = Vec::with_capacity(rows.len());
        for row_index in 0..rows.len() {
            let position = start_position + row_index;
            let q_start = row_index * layout.q_projection_width;
            let kv_start = row_index * layout.kv_width;
            let (mut q, q_gate) = split_q_projection(
                queries[q_start..q_start + layout.q_projection_width].to_vec(),
                layout,
            )?;
            let mut k = keys[kv_start..kv_start + layout.kv_width].to_vec();
            let v = values[kv_start..kv_start + layout.kv_width].to_vec();
            let rope_position = FlashMoeTokenInput::text(0, position).rope_position();
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
            let kv_record = FlashMoeFullAttentionKvRecord::new(position, layer, k, v);
            self.resolve_full_attention_kv_state(position, layer, layout, &kv_record)?;
            kv_cache.record_kv_record(kv_record)?;
            normalized_queries.extend(q);
            query_gates.push(q_gate);
        }
        let end_position = start_position + rows.len() - 1;
        let records = kv_cache.keys_values(end_position, layer)?;
        if records.len() != start_position + rows.len() {
            bail!(
                "Qwen layer-major attention layer {layer} expected {} KV rows through position {end_position}, found {}",
                start_position + rows.len(),
                records.len()
            );
        }
        let mut all_keys = Vec::with_capacity(records.len() * layout.kv_width);
        let mut all_values = Vec::with_capacity(records.len() * layout.kv_width);
        for (key, value) in records {
            if key.len() != layout.kv_width || value.len() != layout.kv_width {
                bail!(
                    "Qwen layer-major attention layer {layer} encountered KV widths {}/{}, expected {}",
                    key.len(),
                    value.len(),
                    layout.kv_width
                );
            }
            all_keys.extend_from_slice(key);
            all_values.extend_from_slice(value);
        }
        let mut output = self.metal.qwen_causal_attention_rows(
            &normalized_queries,
            &all_keys,
            &all_values,
            rows.len(),
            start_position,
            layout.num_q_heads,
            layout.kv_heads,
            layout.head_dim,
        )?;
        for (row_index, q_gate) in query_gates.into_iter().enumerate() {
            if let Some(q_gate) = q_gate {
                let start = row_index * layout.q_width;
                for (value, gate) in output[start..start + layout.q_width]
                    .iter_mut()
                    .zip(q_gate.iter())
                {
                    *value *= sigmoid(*gate);
                }
            }
        }
        Ok(output)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn linear_attention_output_matrix_values(
        &self,
        layer: usize,
        rows: usize,
        normed: &[f32],
    ) -> Result<Vec<f32>> {
        let layout = self.runtime.linear_attention_layout(layer)?;
        let bindings = self.linear_attention_weights.require(layer)?;
        let (mut projections, _, _) = self.metal.resident_mmap_projection_matrix(
            &bindings.input_projections,
            rows,
            self.runtime.width,
            normed,
        )?;
        if rows == 1 {
            let (scalar_projections, _, _) = self
                .metal
                .resident_mmap_matvec_batch(&bindings.input_projections, normed)?;
            if scalar_projections.len() != projections.len() {
                bail!(
                    "Qwen one-row linear-attention input projection counts differ at layer {layer}: matrix={} scalar={}",
                    projections.len(),
                    scalar_projections.len()
                );
            }
            for ((projection, actual), expected) in bindings
                .input_projections
                .iter()
                .zip(projections.iter())
                .zip(scalar_projections.iter())
            {
                if actual.len() != expected.len() {
                    bail!(
                        "Qwen one-row linear-attention input projection widths differ at layer {layer} for {}: matrix={} scalar={}",
                        projection.tensor_name(),
                        actual.len(),
                        expected.len()
                    );
                }
                if let Some((index, (actual, expected))) = actual
                    .iter()
                    .zip(expected.iter())
                    .enumerate()
                    .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
                {
                    bail!(
                        "Qwen one-row linear-attention input projection parity failed at layer {layer} for {} index={index}: matrix={actual} scalar={expected} delta={}",
                        projection.tensor_name(),
                        (actual - expected).abs()
                    );
                }
            }
        }
        let alpha = projections
            .pop()
            .context("missing layer-major linear-attention alpha matrix")?;
        let beta = projections
            .pop()
            .context("missing layer-major linear-attention beta matrix")?;
        let z = projections
            .pop()
            .context("missing layer-major linear-attention Z matrix")?;
        let qkv = projections
            .pop()
            .context("missing layer-major linear-attention QKV matrix")?;
        self.metal
            .qwen_linear_attention_matrix(layout, bindings, rows, &qkv, &z, &beta, &alpha)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn post_attention_output_matrix_values(
        &self,
        layer: usize,
        rows: usize,
        attention_width: usize,
        out_proj_name: &str,
        attention: &[f32],
        residual: &[f32],
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
        prepared_input_norm: Option<&[f32]>,
        start_position: usize,
        kv_cache: &mut KvCache,
        row_timings: &mut [Option<FlashMoeTokenTiming>],
    ) -> Result<Option<Vec<f32>>> {
        if rows.is_empty() || rows.len() != row_timings.len() {
            bail!("Qwen layer-major layer requires aligned non-empty rows and timings");
        }
        let layer_started = Instant::now();
        let width = self.runtime.width;
        let hidden_values = rows
            .len()
            .checked_mul(width)
            .context("Qwen layer-major hidden matrix size overflow")?;
        let mut residual = Vec::with_capacity(hidden_values);
        let mut normed = Vec::with_capacity(hidden_values);
        for row in rows.iter() {
            if row.hidden.len() != width {
                bail!(
                    "Qwen layer-major row width {} does not match {width} at layer {layer}",
                    row.hidden.len()
                );
            }
            residual.extend_from_slice(&row.hidden);
        }
        if let Some(prepared) = prepared_input_norm {
            if prepared.len() != hidden_values {
                bail!(
                    "Qwen layer-major prepared input norm has {} values, expected {hidden_values} at layer {layer}",
                    prepared.len()
                );
            }
            normed.extend_from_slice(prepared);
        } else {
            let input_norm_name = layer_norm_tensor_name(layer, "input_layernorm");
            for row in rows.iter() {
                normed.extend(self.rms_norm_with_model_weight(&input_norm_name, &row.hidden)?);
            }
        }
        let (attention, attention_width, out_proj_name, layer_kind) =
            if self.runtime.is_linear_attention_layer(layer) {
                let layout = self.runtime.linear_attention_layout(layer)?;
                (
                    self.linear_attention_output_matrix_values(layer, rows.len(), &normed)?,
                    layout.total_value_width,
                    linear_attention_tensor_name(layer, "out_proj"),
                    FlashMoeLayerKind::LinearAttention,
                )
            } else {
                let layout = self.runtime.full_attention_layout(layer)?;
                (
                    self.full_attention_output_matrix_values(
                        layer,
                        rows,
                        &normed,
                        start_position,
                        kv_cache,
                    )?,
                    layout.q_width,
                    attention_tensor_name(layer, "o_proj"),
                    FlashMoeLayerKind::FullAttention,
                )
            };
        let post = self.post_attention_output_matrix_values(
            layer,
            rows.len(),
            attention_width,
            &out_proj_name,
            &attention,
            &residual,
        )?;
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
        let (hidden, next_normed) = layer_output.materialize();
        if hidden.len() != hidden_values {
            bail!(
                "Qwen layer-major expert output has {} values, expected {hidden_values}",
                hidden.len()
            );
        }
        if next_normed
            .as_ref()
            .is_some_and(|values| values.len() != hidden_values)
        {
            bail!("Qwen layer-major next norm does not match hidden matrix geometry");
        }
        let mix_hashes = scheduled.route_mix_hashes().collect::<Vec<_>>();
        let elapsed = layer_started.elapsed();
        let per_row_elapsed = elapsed / u32::try_from(rows.len()).unwrap_or(u32::MAX);
        for row_index in 0..rows.len() {
            let mut state = FlashMoeTokenState::from_recurrent_value(
                hidden[row_index * width..(row_index + 1) * width].to_vec(),
                rows[row_index].recurrent_value,
            );
            let route_start = row_index * active;
            for route in route_start..route_start + active {
                state.mix_active_expert(mix_hashes[route], scheduled.weights()[route]);
            }
            let layer_state = state.layer_state_record(start_position + row_index, layer);
            let (row_hidden, recurrent_value) = state.into_hidden_and_recurrent();
            rows[row_index].hidden = row_hidden;
            rows[row_index].recurrent_value = recurrent_value;
            kv_cache.record_layer_state_record(layer_state)?;
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
        Ok(next_normed)
    }

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
        self.dense
            .router_command_with_metal(Some(&self.metal), router_score_command, normed)
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
        self.deepseek_graph.is_some()
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
        if self.deepseek_graph.is_none() {
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
        let deepseek_v4 = self.deepseek_graph.is_some();
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
        if let Some(graph) = self.deepseek_graph.as_ref() {
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
        if let Some(graph) = self.deepseek_graph.clone()
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
                    let mut prepared_input_norm = None;
                    for layer in 0..self.config.num_hidden_layers {
                        prepared_input_norm = self.forward_qwen_layer_major_matrix(
                            layer,
                            &mut rows,
                            prepared_input_norm.as_deref(),
                            chunk_start,
                            kv_cache,
                            &mut row_timings,
                        )?;
                    }
                    for (row, token_timing) in rows.iter_mut().zip(row_timings.iter_mut()) {
                        let norm_started = OptionalInstant::now(token_timing.is_some());
                        row.hidden =
                            self.rms_norm_with_model_weight("model.norm.weight", &row.hidden)?;
                        if let Some(token_timing) = token_timing {
                            token_timing.buckets.combine_norm += norm_started.elapsed();
                            token_timing.buckets.total_wall = token_timing
                                .layers
                                .iter()
                                .map(|layer| layer.buckets.total_wall)
                                .sum::<Duration>()
                                + norm_started.elapsed();
                        }
                    }
                    if let Some(timing) = timing.as_deref_mut() {
                        timing.tokens.extend(row_timings.into_iter().flatten());
                    }
                    rows.pop()
                        .map(|row| row.hidden)
                        .context("Qwen layer-major prefill chunk produced no hidden state")
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
        if self.deepseek_graph.is_some() {
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
        if let Some(graph) = self.deepseek_graph.as_ref() {
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
                0,
                hidden,
                &self.tokenizer,
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
        let candidates = self.dense.lm_head_top_candidates_with_metal(
            &self.metal,
            hidden,
            &self.tokenizer,
            sampler,
            prompt_tokens,
            generated,
        )?;
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

    pub fn expert_scheduler_metrics(&self) -> ExpertSchedulerSnapshot {
        self.scheduler.snapshot()
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
