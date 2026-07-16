use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use objc2::rc::autoreleasepool;
use tracing::{debug, info, trace, trace_span};

use super::capabilities::FlashMoeCapabilityPlan;
use super::experts::ExpertSlotStore;
use super::math::*;
use super::metal::*;
use super::model_family::{QwenModelConfig, QwenMoeFamily, QwenMoeModelLayout};
use super::planning::{FlashMoePlan, ResolvedRoutingPolicy};
use super::scheduler::*;
use super::state::*;
use super::text::*;
use super::types::*;
use super::vision::{
    FlashMoeInputAdapterExecutor, FlashMoeTokenInput, ImagePreprocessor, MropePosition,
    QwenVlRuntimeInputs, VisionEncoding,
};
use super::weights::*;

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
    pub(super) runtime: DenseTransformerRuntime,
    pub(super) linear_attention_weights: LinearAttentionWeightTable,
    pub(super) shared_expert_weights: SharedExpertWeightTable,
    pub(super) input_adapter_executor: FlashMoeInputAdapterExecutor,
    pub(super) session_cache: FlashMoeSessionCache,
}

pub fn load(plan: &FlashMoePlan) -> Result<FlashMoeEngine> {
    load_with_progress(plan, |_, _| {})
}

pub fn load_with_progress<F>(plan: &FlashMoePlan, mut progress: F) -> Result<FlashMoeEngine>
where
    F: FnMut(&'static str, Duration),
{
    let mut phase_started = Instant::now();
    let status = plan.cache_status()?;
    progress("cache_status", phase_started.elapsed());
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
    phase_started = Instant::now();
    let config = QwenModelConfig::from_file(&plan.model_config)?;
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
    validate_required_tensor_manifest(&config, dense.registry())?;
    progress("manifest_validation", phase_started.elapsed());
    phase_started = Instant::now();
    let runtime = DenseTransformerRuntime::from_registry(&config, dense.registry())?;
    let attention_layers = runtime.resolved_attention_layers()?;
    progress("runtime_layout", phase_started.elapsed());
    phase_started = Instant::now();
    let linear_attention_weights = dense.resolve_linear_attention_weight_table(
        &runtime.linear_attention,
        config.hidden_size,
        model_layout.experts_per_layer,
    )?;
    progress("linear_attention_weights", phase_started.elapsed());
    phase_started = Instant::now();
    let shared_expert_weights = dense.resolve_shared_expert_weight_table(
        config.num_hidden_layers,
        config.hidden_size,
        config.shared_experts(),
        config.shared_expert_intermediate_size(),
    )?;
    progress("shared_expert_weights", phase_started.elapsed());
    phase_started = Instant::now();
    let dense_layout = dense.registry().resolve_resident_dense_layout()?;
    if matches!(
        model_layout.family,
        QwenMoeFamily::Qwen35A17B | QwenMoeFamily::Qwen3Moe | QwenMoeFamily::Qwen3VlMoe
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
    progress("metal_executor", phase_started.elapsed());
    phase_started = Instant::now();
    let experts = resolved_experts.store;
    let expert_storage = resolved_experts.descriptor;
    progress("expert_store", phase_started.elapsed());
    phase_started = Instant::now();
    let capability_plan = FlashMoeCapabilityPlan::resolve(
        &model_layout,
        input_adapter,
        dense_layout,
        expert_storage,
        &attention_layers,
        Some(metal.runtime_capabilities()),
    )?;
    let scheduled_graph = FlashMoeScheduledGraph::from_capabilities(&capability_plan)?;
    let scheduler = FlashMoeExecutionScheduler::new(scheduled_graph, experts)?;
    progress("capability_graph", phase_started.elapsed());
    phase_started = Instant::now();
    let tokenizer = QwenTokenizer::from_files(&plan.tokenizer, &plan.tokenizer_config)?;
    progress("tokenizer", phase_started.elapsed());
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
        runtime,
        linear_attention_weights,
        shared_expert_weights,
        session_cache: FlashMoeSessionCache::default(),
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

#[cfg(test)]
mod ownership_tests {
    use super::*;
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
            let inner = MetalExecutionContext::compile(
                dense.mmap.clone(),
                dense.len,
                &runtime.linear_attention,
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
        MetalRuntimeCapabilities::from_pipeline_names(MetalPipelineNameSet::new())
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
    ) -> Result<MetalPostAttentionPrep> {
        self.inner.resident_post_attention_prep_topk(
            projections,
            attention_output,
            residual,
            post_norm_weight,
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
        Ok(hidden)
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
    ) -> Result<Vec<f32>> {
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
        let hidden_values = match input.precomputed_embedding(runtime.width)? {
            Some(values) => values.to_vec(),
            None => self.dense.embedding(previous, runtime.width)?,
        };
        let mut token_state = FlashMoeTokenState::new(
            hidden_values,
            self.dense.seed(position, previous)? ^ (self.plan.model.len() as u64),
        );
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
                    let values = self.full_attention_output_values(
                        layer,
                        &normed,
                        deferred_attention_input,
                        kv_cache,
                        position,
                        rope_position,
                        runtime,
                        record_detailed_timing.then_some(&mut layer_timing.buckets),
                    )?;
                    post_attention_values_for_prep =
                        Some((attention_tensor_name(layer, "o_proj"), values));
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
                let mut prep = self.dense.post_attention_prep_with_metal(
                    metal,
                    layer,
                    self.scheduler.experts_per_layer(),
                    &out_proj_name,
                    &attention_values,
                    residual_input,
                    &post_norm_weight,
                    scheduled_cmd2.active_experts,
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
        Ok(token_state.into_hidden_values())
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
        let mut attended =
            self.full_attention_cached(kv_cache, position, layer, &q, layout, attention_output)?;

        if let Some(q_gate) = q_gate {
            for (value, gate) in attended.iter_mut().zip(q_gate.iter()) {
                *value *= sigmoid(*gate);
            }
        }
        if let Some(buckets) = attention_buckets.as_deref_mut() {
            buckets.attention_kernel += subphase_started.elapsed();
        }
        Ok(attended)
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

    fn full_attention_cached(
        &self,
        kv_cache: &KvCache,
        position: usize,
        layer: usize,
        q: &[f32],
        layout: FullAttentionLayout,
        attention_output: ScheduledAttentionMathOutput,
    ) -> Result<Vec<f32>> {
        let attention_output =
            attention_output.validate_execution_state(layer, position, layout.kv_width)?;

        match attention_output.implementation() {
            ScheduledAttentionMathImplementation::CpuKvCache => kv_cache.causal_attention(
                position,
                layer,
                q,
                layout.num_q_heads,
                layout.kv_heads,
                layout.head_dim,
            ),
        }
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
        if self
            .config
            .linear_attention_qkv_projection_requires_reorder()
        {
            bail!(
                "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD1 path at layer {layer}: the resolved implementation does not support QKV projection reorder"
            );
        }
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

    /// Render and tokenize the exact prompt used by structured generation.
    pub fn measure_structured_prompt(
        &self,
        request: &StructuredGenerationRequest,
    ) -> Result<usize> {
        Ok(self.structured_prompt_tokens(request)?.1.len())
    }

    fn structured_prompt_tokens(
        &self,
        request: &StructuredGenerationRequest,
    ) -> Result<(String, Vec<u32>)> {
        if !request.raw_prompt {
            return self.tokenizer.render_and_encode_chat_prompt(
                &request.messages,
                &request.tools,
                request.add_generation_prompt,
                request.enable_thinking,
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
                request.enable_thinking,
            )
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
        let max_tokens = request.max_tokens.max(0) as usize;
        validate_context_capacity(prompt_tokens.len(), max_tokens, request.context_size)?;
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
        let mut generation = self.session_cache.begin_generation(
            session_id,
            prompt_tokens,
            max_tokens,
            self.config.num_hidden_layers,
        );
        let prefill_start = generation.prefill_start();
        let prompt_len = generation.prompt_len();
        if prefill_start > 0 {
            debug!(
                target: "flashmoe::lifecycle",
                "flashmoe: reusing session cache prefix_tokens={} prompt_tokens={}",
                prefill_start, prompt_len
            );
        }
        if prefill_start == 0 {
            self.metal.reset_linear_attention_state()?;
        } else {
            let recurrent = generation
                .take_cached_recurrent()
                .context("session cache entry is missing the Metal recurrent-state snapshot")?;
            self.metal
                .restore_linear_attention_session_state(&recurrent)?;
        }
        let prefill_or_ttft_started = Instant::now();
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
            let hidden = {
                let (prompt_tokens, prefill_start, kv_cache) = generation.prefill_inputs();
                let detailed = if detailed_timing {
                    timing.as_deref_mut()
                } else {
                    None
                };
                self.prefill_from(
                    prompt_tokens,
                    prefill_start,
                    kv_cache,
                    detailed,
                    progress.clone(),
                )?
            };
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
        if generation.requires_prompt_snapshot() {
            let recurrent = self.metal.capture_linear_attention_session_state()?;
            generation.capture_prompt_cache(prefill_hidden.clone(), recurrent);
        }

        let mut sampler = TokenSampler::new(request.temperature, request.top_k, request.seed);
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
            generation.record_sampled_token(token, self.tokenizer.is_eos(token));
        }
        let prefill_or_ttft_wall = prefill_or_ttft_started.elapsed();
        let decode_phase_started = Instant::now();
        let mut decode_tokens = 0usize;
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
                )?
            };
            let token = sampled.token;
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
            generation.record_sampled_token(token, self.tokenizer.is_eos(token));
        }
        let decode_wall = decode_phase_started.elapsed();

        self.session_cache.commit_generation(&mut generation)?;

        let generated = generation.into_generated();
        let decoded = self.tokenizer.decode(&generated)?;
        let finish_reason = generation_finish_reason(generated.len(), max_tokens);
        let (content, tool_calls) = parse_qwen_tool_call_output_with_incomplete(
            &decoded,
            finish_reason == GenerationFinishReason::MaxTokens,
        )?;
        let output = GenerationOutput {
            content,
            tool_calls,
            finish_reason,
            prompt_tokens: prompt_len,
            generated_tokens: generated.len(),
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

    fn prefill_from(
        &mut self,
        prompt_tokens: &[u32],
        start_position: usize,
        kv_cache: &mut KvCache,
        mut timing: Option<&mut FlashMoeGenerationTiming>,
        progress: GenerationProgress<'_>,
    ) -> Result<Vec<f32>> {
        if start_position > prompt_tokens.len() {
            bail!(
                "prefill start position {start_position} exceeds prompt length {}",
                prompt_tokens.len()
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
                    prompt_tokens.len().saturating_sub(position + 1),
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
            let remaining = prompt_tokens.len().saturating_sub(position + 1);
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
            )?;
            let token = sampled.token;
            if self.tokenizer.is_eos(token) {
                break;
            }
            generated.push(token);
        }

        let decoded = self.tokenizer.decode(&generated)?;
        let finish_reason = generation_finish_reason(generated.len(), max_tokens);
        let (content, tool_calls) = parse_qwen_tool_call_output_with_incomplete(
            &decoded,
            finish_reason == GenerationFinishReason::MaxTokens,
        )?;
        Ok(GenerationOutput {
            content,
            tool_calls,
            finish_reason,
            prompt_tokens: runtime_inputs.prompt_tokens().len(),
            generated_tokens: generated.len(),
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

    pub(super) fn rms_norm_with_model_weight(
        &self,
        canonical_name: &str,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let weight = self.model_norm_weight(canonical_name, input.len())?;
        let mut out = input.to_vec();
        rms_norm_with_weight_in_place(&mut out, weight.as_deref());
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
        Ok(SampledDecode { token })
    }

    fn sample_from_hidden(
        &self,
        sampler: &mut TokenSampler,
        hidden: &[f32],
        prompt_tokens: &[u32],
        generated: &[u32],
        trace_candidates: bool,
        progress: &GenerationProgress<'_>,
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
    ) -> Result<u32> {
        if trace_candidates {
            let logits = self.dense.lm_head_logits_with_metal(
                Some(&self.metal),
                0,
                hidden,
                &self.tokenizer,
            )?;
            let candidates = sampler.top_candidates(&logits, prompt_tokens, generated);
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
