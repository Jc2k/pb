use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use tracing::info;

use super::legacy::*;
use super::metal::*;
use super::scheduler::*;
use super::state::*;
use super::types::*;
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

#[derive(Debug)]
pub(super) enum DeferredExpertPhase {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    Ready(FlashMoeExpertPhaseOutput),
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    ScheduledMetal(MetalScheduledCmd3Submission),
}

impl DeferredExpertPhase {
    pub(super) fn wait(self) -> Result<FlashMoeExpertPhaseOutput> {
        match self {
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            Self::Ready(output) => Ok(output),
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Self::ScheduledMetal(output) => output.wait(),
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn next_normed_metal_input(&self) -> Option<DeferredMetalInput> {
        match self {
            Self::ScheduledMetal(output) => output
                .next_normed_buffer()
                .map(|(buffer, len)| DeferredMetalInput::next_layer_normed(buffer, len)),
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn hidden_metal_input(&self) -> Option<DeferredMetalInput> {
        match self {
            Self::ScheduledMetal(output) => {
                let (buffer, len) = output.hidden_buffer();
                Some(DeferredMetalInput::hidden(buffer, len))
            }
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn finish_without_readback(self) -> Result<()> {
        match self {
            Self::ScheduledMetal(output) => output.finish_without_readback(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DeferredMetalInput {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) buffer: ObjcId,
    pub(super) state: FlashMoeGpuBufferDescriptor,
}

impl DeferredMetalInput {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn hidden(buffer: ObjcId, len: usize) -> Self {
        Self {
            buffer,
            state: FlashMoeGpuBufferDescriptor::hidden(len),
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn next_layer_normed(buffer: ObjcId, len: usize) -> Self {
        Self {
            buffer,
            state: FlashMoeGpuBufferDescriptor::next_layer_normed(len),
        }
    }

    pub(super) fn len(self) -> usize {
        self.state.len()
    }

    pub(super) fn state(self) -> FlashMoeGpuBufferDescriptor {
        self.state
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
    ) -> Result<DeferredExpertPhase> {
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
                Ok(DeferredExpertPhase::ScheduledMetal(pending))
            }
        }
    }

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

    pub(super) fn resident_q4_top_candidates(
        &self,
        projection: &DenseQ4MmapMatvecProjection,
        input: &[f32],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner
                .resident_q4_top_candidates(projection, input, top_k)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (projection, input, top_k);
            bail!("FlashMoe unsupported resident Q4 topK path: Apple Silicon Metal is required")
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
        match &plan.source {
            RouterScoreProjectionTopKSource::ResidentDense(projection) => bail!(
                "FlashMoe unsupported router-score layout for layer {}: resolved Metal graph requires resident Q4, got {}",
                plan.layer,
                projection.dtype
            ),
            RouterScoreProjectionTopKSource::ResidentQ4(projection) => self
                .resident_q4_top_candidates(projection, hidden, plan.active_experts)
                .map(Some),
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn q4_post_attention_prep_topk(
        &self,
        projections: &Cmd2Q4PostAttentionPrepProjections,
        attention_output: &[f32],
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
    ) -> Result<MetalPostAttentionPrep> {
        self.inner.q4_post_attention_prep_topk(
            projections,
            attention_output,
            residual,
            post_norm_weight,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn linear_attention_q4_post_attention_prep(
        &self,
        layer: usize,
        layout: LinearAttentionLayout,
        input_projections: &[DenseQ4MmapMatvecProjection],
        input: MetalBatchProjectionInput<'_>,
        static_offsets: MetalLinearAttentionStaticOffsets,
        out_proj: &DenseQ4MmapMatvecProjection,
        router: &DenseQ4MmapMatvecProjection,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        top_k: usize,
    ) -> Result<MetalPostAttentionPrep> {
        self.inner.linear_attention_q4_post_attention_prep(
            layer,
            layout,
            input_projections,
            input,
            static_offsets,
            out_proj,
            router,
            residual,
            post_norm_weight,
            top_k,
        )
    }

    pub(super) fn q4_mmap_matvec_batch(
        &self,
        projections: &[DenseQ4MmapMatvecProjection],
        input: &[f32],
    ) -> Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.inner.q4_projection_batch(projections, input)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (projections, input);
            Ok(None)
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn q4_mmap_matvec_batch_with_input_buffer(
        &self,
        projections: &[DenseQ4MmapMatvecProjection],
        input_buffer: ObjcId,
        input_len: usize,
    ) -> Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        self.inner
            .q4_projection_batch_with_input_buffer(projections, input_buffer, input_len)
    }
}

impl FlashMoeEngine {
    pub(super) fn forward_hidden(
        &mut self,
        previous: u32,
        kv_cache: &mut KvCache,
        position: usize,
        rope_position: MropePosition,
        record_generated: bool,
        mut timing: Option<&mut FlashMoeTokenTiming>,
        progress: GenerationProgress<'_>,
    ) -> Result<Vec<f32>> {
        let runtime = &self.runtime;
        let token_started = Instant::now();
        let hidden_values = self.dense.embedding(previous, runtime.width)?;
        let mut token_state = FlashMoeTokenState::new(
            hidden_values,
            self.dense.seed(position, previous)? ^ (self.plan.model.len() as u64),
        );
        debug_assert!(token_state.hidden().is_declared_graph_state());
        let mut deferred_expert_phase: Option<DeferredExpertPhase> = None;

        for layer in 0..self.config.num_hidden_layers {
            let report_layer_progress = progress.is_some()
                || layer == 0
                || layer + 1 == self.config.num_hidden_layers
                || layer % 10 == 0;
            if report_layer_progress {
                report_generation_progress(
                    &progress,
                    format!(
                        "forward layer begin position={} layer={}/{}",
                        position,
                        layer + 1,
                        self.config.num_hidden_layers
                    ),
                );
            }
            let mut pending_for_layer = deferred_expert_phase.take();
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let (deferred_attention_input, deferred_residual_input) = {
                let normed_candidate = pending_for_layer
                    .as_ref()
                    .and_then(DeferredExpertPhase::next_normed_metal_input);
                if normed_candidate.is_some() {
                    let residual_candidate = pending_for_layer
                        .as_ref()
                        .and_then(DeferredExpertPhase::hidden_metal_input);
                    (normed_candidate, residual_candidate)
                } else {
                    (None, None)
                }
            };
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let deferred_attention_input: Option<DeferredMetalInput> = None;

            if pending_for_layer.is_some() && deferred_attention_input.is_none() {
                let pending = pending_for_layer
                    .take()
                    .context("missing deferred expert phase")?;
                let wait_started = Instant::now();
                let output = pending.wait()?;
                let wait_elapsed = wait_started.elapsed();
                info!(
                    token_position = position,
                    completed_layer = layer.saturating_sub(1),
                    wait_ms = wait_elapsed.as_millis(),
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
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
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
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let previous_handoff = if layer == 0 {
                ScheduledPreviousCmd3Handoff::initial(token_state.hidden().len())
            } else {
                ScheduledPreviousCmd3Handoff::cpu_visible(layer - 1, token_state.hidden().len())
            };
            let layer_schedule = self.scheduler.begin_layer(
                position,
                layer,
                self.config.num_hidden_layers,
                self.routing_policy.active_experts,
                previous_handoff,
                true,
            )?;
            let attention_implementation = layer_schedule.attention_implementation();
            let layer_started = Instant::now();
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
            let combine_started = Instant::now();
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
            let attention_started = Instant::now();
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
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let mut early_metal_post_attention_prep: Option<MetalPostAttentionPrep> = None;
            let projected = match attention_implementation {
                ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal => {
                    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                    {
                        let residual_input = deferred_residual_input
                            .map(|input| MetalBatchProjectionInput::Buffer {
                                buffer: input.buffer,
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
                            Some(&mut layer_timing.buckets),
                        )?;
                        if deferred_residual_input.is_some()
                            && let Some(pending) = pending_for_layer.take()
                        {
                            pending.finish_without_readback()?;
                        }
                        early_metal_post_attention_prep = Some(prep);
                        Vec::new()
                    }
                    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                    {
                        bail!(
                            "FlashMoe unsupported scheduled linear-attention implementation at layer {layer}: Apple Silicon Metal is required"
                        )
                    }
                }
                ScheduledLayerAttentionImplementation::FullAttentionCpuKv => {
                    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                    {
                        let values = self.full_attention_output_values(
                            layer,
                            &normed,
                            deferred_attention_input,
                            kv_cache,
                            position,
                            rope_position,
                            runtime,
                            Some(&mut layer_timing.buckets),
                        )?;
                        post_attention_values_for_prep =
                            Some((attention_tensor_name(layer, "o_proj"), values));
                        Vec::new()
                    }
                    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                    {
                        bail!(
                            "FlashMoe unsupported scheduled full-attention CMD1/CMD2 implementation at layer {layer}: Apple Silicon Metal is required"
                        )
                    }
                }
            };
            trace_layer_values(position, layer, "attention", &projected);
            layer_timing.buckets.attention_projection += attention_started.elapsed();
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let can_defer_residual_wait_for_post_prep =
                deferred_residual_input.is_some() && post_attention_values_for_prep.is_some();
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let can_defer_residual_wait_for_post_prep = false;
            if deferred_attention_input.is_some()
                && let Some(pending) = pending_for_layer.take()
                && !can_defer_residual_wait_for_post_prep
            {
                let wait_started = Instant::now();
                let output = pending.wait()?;
                let wait_elapsed = wait_started.elapsed();
                info!(
                    token_position = position,
                    completed_layer = layer.saturating_sub(1),
                    wait_ms = wait_elapsed.as_millis(),
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
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
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
            let combine_started = Instant::now();
            let mut precomputed_active: Option<ScheduledRoutingCommand> = None;
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
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
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    let residual_input = deferred_residual_input
                        .map(|input| MetalBatchProjectionInput::Buffer {
                            buffer: input.buffer,
                            len: input.len(),
                        })
                        .unwrap_or(MetalBatchProjectionInput::Cpu(token_state.hidden()));
                    let metal = &self.metal;
                    let post_norm_weight = self
                        .model_norm_weight(post_norm_name.as_str(), runtime.width)?
                        .with_context(|| {
                            format!(
                                "FlashMoe unsupported scheduled CMD2 Q4 post-attention prep path: missing norm tensor {post_norm_name}"
                            )
                        })?;
                    let mut prep = self.dense.post_attention_q4_prep_with_metal(
                        metal,
                        layer,
                        self.config.experts(),
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
                }
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                {
                    let _ = (out_proj_name, attention_values);
                    scheduled_cmd2.reject_missing_post_attention_prep(
                        "the resolved CMD2 Q4 post-attention prep implementation requires Apple Silicon Metal",
                    )?;
                    unreachable!("reject_missing_post_attention_prep always returns an error")
                }
            } else {
                add_in_place(token_state.hidden_mut(), &projected);
                normed = FlashMoeCpuBuffer::normed(
                    self.rms_norm_with_model_weight(post_norm_name.as_str(), token_state.hidden())?,
                );
                layer_timing.buckets.combine_norm += combine_started.elapsed();
                let routing_started = Instant::now();
                let active = self.route_layer(layer, &normed, scheduled_cmd2.active_experts)?;
                layer_timing.buckets.routing += routing_started.elapsed();
                active
            };
            layer_timing.active_experts = active.routes.len();
            let layer_schedule = layer_schedule.resolve(&active)?;
            let pending_cmd3 = layer_schedule.issue_cmd3(&mut self.scheduler, &active)?;
            // While expert reads are still pending, prepare the always-active
            // shared-expert branch for the deferred expert command buffer.
            let shared_compute_started = Instant::now();
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let shared_q4_phase =
                self.required_shared_expert_q4_phase_projections(layer, runtime.width)?;
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let shared_phase = SharedExpertPhaseRef::Q4(shared_q4_phase.as_ref());
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let shared_phase = SharedExpertPhaseRef::None;
            layer_timing.buckets.expert_compute += shared_compute_started.elapsed();
            let cmd3_prepare_started = Instant::now();
            let prepared_next_norm_weights = prepare_scheduled_next_norm_weights(
                layer,
                self.config.num_hidden_layers,
                runtime.width,
                true,
                |name, width| self.model_norm_weight(name, width),
            )?;
            let next_norm_weights = prepared_next_norm_weights.scheduled()?;
            layer_timing.buckets.expert_compute += cmd3_prepare_started.elapsed();
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let prep = metal_post_attention_prep.take().with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled Qwen3.5 CMD3 path at layer {layer}: resolved Metal CMD2 did not produce post-attention state; CPU normed/residual upload is not a declared implementation"
                )
            })?;
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            bail!(
                "FlashMoe unsupported scheduled Qwen3.5 CMD3 path at layer {layer}: the resolved implementation requires Apple Silicon Metal"
            );
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
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
                            FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
                        )?;
                    }
                }
                expert_delta
            };
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let expert_delta = ExpertSchedulerSnapshot::default();
            trace_layer_values(position, layer, "moe", token_state.hidden());
            let combine_started = Instant::now();
            if deferred_expert_phase.is_some() {
                kv_cache
                    .record_layer_state_record(token_state.layer_state_record(position, layer))?;
                layer_timing.buckets.combine_norm += combine_started.elapsed();
                layer_timing.buckets.total_wall = layer_started.elapsed();
                info!(
                    token_position = position,
                    layer,
                    layer_kind = layer_timing.layer_kind.as_str(),
                    active_experts = layer_timing.active_experts,
                    expert_deferred = true,
                    attention_ms = layer_timing.buckets.attention_projection.as_millis(),
                    routing_ms = layer_timing.buckets.routing.as_millis(),
                    expert_io_ms = layer_timing.buckets.expert_io.as_millis(),
                    expert_read_ms = layer_timing.buckets.expert_read.as_millis(),
                    expert_compute_ms = layer_timing.buckets.expert_compute.as_millis(),
                    total_ms = layer_timing.buckets.total_wall.as_millis(),
                    bytes_read = expert_delta.bytes_read,
                    "flashmoe layer complete"
                );
                if report_layer_progress {
                    report_generation_progress(
                        &progress,
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
                        ),
                    );
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
            info!(
                token_position = position,
                layer,
                layer_kind = layer_timing.layer_kind.as_str(),
                active_experts = layer_timing.active_experts,
                expert_deferred = false,
                attention_ms = layer_timing.buckets.attention_projection.as_millis(),
                routing_ms = layer_timing.buckets.routing.as_millis(),
                expert_io_ms = layer_timing.buckets.expert_io.as_millis(),
                expert_read_ms = layer_timing.buckets.expert_read.as_millis(),
                expert_compute_ms = layer_timing.buckets.expert_compute.as_millis(),
                total_ms = layer_timing.buckets.total_wall.as_millis(),
                bytes_read = expert_delta.bytes_read,
                "flashmoe layer complete"
            );
            if report_layer_progress {
                report_generation_progress(
                    &progress,
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
                    ),
                );
            }
            if let Some(timing) = timing.as_deref_mut() {
                timing.buckets.add(layer_timing.buckets);
                timing.layers.push(layer_timing);
            }
        }
        if let Some(pending) = deferred_expert_phase.take() {
            let wait_started = Instant::now();
            let output = pending.wait()?;
            let wait_elapsed = wait_started.elapsed();
            info!(
                token_position = position,
                completed_layer = self.config.num_hidden_layers.saturating_sub(1),
                wait_ms = wait_elapsed.as_millis(),
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

        let combine_started = Instant::now();
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

    fn full_attention_output_values(
        &self,
        layer: usize,
        normed: &[f32],
        deferred_input: Option<DeferredMetalInput>,
        kv_cache: &mut KvCache,
        position: usize,
        rope_position: MropePosition,
        runtime: &DenseTransformerRuntime,
        mut attention_buckets: Option<&mut FlashMoeTimingBuckets>,
    ) -> Result<Vec<f32>> {
        let layout = runtime.full_attention_layout(layer)?;
        let subphase_started = Instant::now();
        let input_requests = full_attention_input_projection_requests(
            layer,
            layout.q_projection_width,
            layout.kv_width,
        )?;
        let input_specs = input_requests.requests();
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let mut projections = if let Some(input) = deferred_input {
            self.dense.project_q4_tensors_from_metal_input(
                &self.metal,
                &input_specs,
                input.buffer,
                input.len(),
            )?
        } else {
            self.dense
                .project_q4_tensors_from_cpu_input(&self.metal, &input_specs, normed)?
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

        let subphase_started = Instant::now();
        let (mut q, q_gate) = split_q_projection(q_projected, layout)?;

        // Some Qwen full-attention variants apply Q/K RMSNorm before RoPE.
        let q_norm_name = layer_norm_tensor_name(layer, "self_attn.q_norm");
        let k_norm_name = layer_norm_tensor_name(layer, "self_attn.k_norm");

        let q_norm_w = self.model_norm_weight(&q_norm_name, layout.head_dim)?;
        let k_norm_w = self.model_norm_weight(&k_norm_name, layout.head_dim)?;
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
            q_norm_w.as_deref(),
            k_norm_w.as_deref(),
        )?;
        if let Some(buckets) = attention_buckets.as_deref_mut() {
            buckets.attention_misc += subphase_started.elapsed();
        }

        let subphase_started = Instant::now();
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
            self.config.experts(),
            normed.len(),
        )?;
        let router_score_command = self.scheduler.resolve_router_score_projection(
            layer,
            self.config.experts(),
            active_experts,
            projection,
            normed.len(),
        )?;
        self.dense
            .router_command_with_metal(Some(&self.metal), router_score_command, normed)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn linear_attention_post_attention_prep_with_metal(
        &self,
        layer: usize,
        normed: &[f32],
        deferred_input: Option<DeferredMetalInput>,
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
        let static_offsets = self
            .dense
            .linear_attention_static_offsets_for_metal(layer, layout)?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled Qwen3.5 linear-attention state path at layer {layer}: resident static tensor offsets are unavailable"
                )
            })?;
        let residual_len = residual.len();
        if residual_len != runtime.width || residual_len != post_norm_weight.len() {
            bail!(
                "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD2 path at layer {layer}: residual width {residual_len}, runtime width {}, and norm width {} do not match",
                runtime.width,
                post_norm_weight.len()
            );
        }
        let out_proj_name = linear_attention_tensor_name(layer, "out_proj");
        let input_requests = linear_attention_input_projection_requests(
            layer,
            layout.conv_dim,
            layout.total_value_width,
            layout.num_value_heads,
        )?;
        let input_specs = input_requests.requests();
        let projection_input = if let Some(input) = deferred_input {
            MetalBatchProjectionInput::Buffer {
                buffer: input.buffer,
                len: input.len(),
            }
        } else if !normed.is_empty() {
            MetalBatchProjectionInput::Cpu(normed)
        } else {
            bail!(
                "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD1 path at layer {layer}: neither deferred Metal nor CPU normed input is available"
            );
        };
        let started = Instant::now();
        let prep = self
            .dense
            .linear_attention_q4_post_attention_prep_with_metal(
                metal,
                layer,
                layout,
                &input_specs,
                projection_input,
                static_offsets,
                self.config.experts(),
                &out_proj_name,
                residual,
                post_norm_weight,
                self.routing_policy.active_experts,
            )?;
        if let Some(buckets) = attention_buckets.as_deref_mut() {
            buckets.attention_input_projection += started.elapsed();
        }
        Ok(prep)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn required_shared_expert_q4_phase_projections(
        &self,
        layer: usize,
        width: usize,
    ) -> Result<Arc<SharedExpertPhaseQ4Projections>> {
        self.shared_expert_phases
            .q4(
            layer,
            width,
            self.config.shared_experts(),
            self.config.shared_expert_intermediate_size(),
            |layer, width, shared_experts, intermediate| {
                self.dense.required_shared_expert_q4_phase_projections(
                    layer,
                    width,
                    shared_experts,
                    intermediate,
                )
            },
        )?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled Qwen3.5 CMD3 shared-expert path at layer {layer}: resident Q4 shared projections are unavailable"
                )
            })
    }
}
