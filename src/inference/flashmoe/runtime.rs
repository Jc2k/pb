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

use super::scheduler::ScheduledSharedExpertPhaseRef as SharedExpertPhaseRef;

impl FlashMoeEngine {
    pub(super) fn forward_hidden(
        &mut self,
        previous: u32,
        embedding_override: Option<Vec<f32>>,
        kv_cache: &mut KvCache,
        position: usize,
        rope_position: MropePosition,
        deepstack: Option<DeepstackTokenContext<'_>>,
        record_generated: bool,
        expert_execution: ExpertExecution,
        mut timing: Option<&mut FlashMoeTokenTiming>,
        progress: GenerationProgress<'_>,
    ) -> Result<Vec<f32>> {
        let runtime = &self.runtime;
        let token_started = Instant::now();
        let hidden_values = if let Some(mut emb) = embedding_override {
            if emb.len() != runtime.width {
                tracing::warn!(
                    got = emb.len(),
                    expected = runtime.width,
                    "vision embedding dimension mismatch; zero-padding to runtime width"
                );
                emb.resize(runtime.width, 0.0);
            }
            emb
        } else {
            self.dense.embedding(previous, runtime.width)?
        };
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
            let (deferred_attention_input, deferred_residual_input) = if deepstack.is_none() {
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
            } else {
                (None, None)
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
            let layer_started = Instant::now();
            let mut layer_timing = FlashMoeLayerTiming {
                layer,
                layer_kind: self.layer_kind(layer),
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
            let scheduled_cmd1 = self
                .scheduled_graph
                .build_cmd1_attention_projections(layer, cmd1_input)?;
            let scheduled_cmd1 = self
                .scheduled_graph
                .build_cmd1_submission(scheduled_cmd1, cmd1_input)?
                .into_cmd1_command();
            debug_assert_eq!(scheduled_cmd1.layer, layer);
            debug_assert_eq!(scheduled_cmd1.input, cmd1_input);
            let cmd1_input_state = if let Some(input) = deferred_attention_input {
                FlashMoeCmd1InputState::gpu_next_layer_normed(layer, input.state())
            } else {
                FlashMoeCmd1InputState::cpu_normed(layer, normed.len())
            };
            let scheduled_cmd1 = scheduled_cmd1.into_resolved_command(cmd1_input_state)?;
            debug_assert_eq!(scheduled_cmd1.layer, layer);
            debug_assert_eq!(scheduled_cmd1.cmd1.layer, layer);
            debug_assert_eq!(scheduled_cmd1.input, cmd1_input);
            debug_assert_eq!(scheduled_cmd1.input_state.layer(), layer);
            debug_assert!(scheduled_cmd1.input_state.is_declared_graph_state());
            let mut post_attention_values_for_prep = None;
            let post_norm_name = layer_norm_tensor_name(layer, "post_attention_layernorm");
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let mut early_metal_post_attention_prep: Option<MetalPostAttentionPrep> = None;
            let projected = if self.runtime.is_linear_attention_layer(layer) {
                if deepstack.is_some() {
                    bail!(
                        "FlashMoe unsupported linear-attention input adapter at layer {layer}: Qwen-VL deepstack is not resolved by the scheduled graph"
                    );
                }
                if expert_execution == ExpertExecution::Skip {
                    bail!(
                        "FlashMoe unsupported linear-attention execution at layer {layer}: skipping expert stages is not a declared graph implementation"
                    );
                }
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
            } else {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    if deepstack.is_none() && expert_execution != ExpertExecution::Skip {
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
                    } else {
                        self.full_attention_projected(
                            layer,
                            &normed,
                            deferred_attention_input,
                            kv_cache,
                            position,
                            rope_position,
                            runtime,
                            Some(&mut layer_timing.buckets),
                        )?
                    }
                }
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                {
                    self.full_attention_projected(
                        layer,
                        &normed,
                        deferred_attention_input,
                        kv_cache,
                        position,
                        rope_position,
                        runtime,
                        Some(&mut layer_timing.buckets),
                    )?
                }
            };
            trace_layer_values(position, layer, "attention", &projected);
            layer_timing.buckets.attention_projection += attention_started.elapsed();
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let can_defer_residual_wait_for_post_prep = deferred_residual_input.is_some()
                && expert_execution != ExpertExecution::Skip
                && post_attention_values_for_prep.is_some();
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
            let scheduled_cmd2 = self.scheduled_graph.build_cmd2_command(
                layer,
                self.routing_policy.active_experts,
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
                let routing_command = scheduled_cmd2.command_from_post_attention_prep_routes(
                    &self.scheduled_graph,
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
                    let routing_command = scheduled_cmd2.command_from_post_attention_prep_routes(
                        &self.scheduled_graph,
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
            if expert_execution == ExpertExecution::Skip && deepstack.is_none() {
                kv_cache
                    .record_layer_state_record(token_state.layer_state_record(position, layer))?;
                layer_timing.buckets.total_wall = layer_started.elapsed();
                if let Some(timing) = timing.as_deref_mut() {
                    timing.buckets.add(layer_timing.buckets);
                    timing.layers.push(layer_timing);
                }
                continue;
            }
            let expert_metrics_before = self.scheduler.snapshot();
            let expert_io_started = Instant::now();
            let pending_experts = self.scheduler.issue_routing_command(&active)?;
            layer_timing.buckets.expert_io += expert_io_started.elapsed();
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
            let expert_io_started = Instant::now();
            let scheduled_experts = self.scheduler.finish_routes(pending_experts)?;
            layer_timing.buckets.expert_io += expert_io_started.elapsed();
            let expert_metrics_after = self.scheduler.snapshot();
            let expert_delta = expert_metrics_after.saturating_delta(expert_metrics_before);
            layer_timing
                .buckets
                .add_expert_scheduler_delta(expert_delta);
            let expert_compute_started = Instant::now();
            debug_assert_eq!(scheduled_experts.layer, layer);
            debug_assert_eq!(scheduled_experts.len(), scheduled_experts.routes.len());
            debug_assert_eq!(scheduled_experts.len(), scheduled_experts.weights.len());
            debug_assert_eq!(scheduled_experts.is_empty(), active.routes.is_empty());
            for (expert, weight) in scheduled_experts
                .experts
                .iter()
                .zip(scheduled_experts.weights.iter().copied())
            {
                token_state.mix_active_expert(expert.mix_hash(), weight);
            }
            let prepared_next_norm_weights = prepare_scheduled_next_norm_weights(
                layer,
                self.config.num_hidden_layers,
                runtime.width,
                deepstack.is_none(),
                |name, width| self.model_norm_weight(name, width),
            )?;
            let next_norm_weights = prepared_next_norm_weights.scheduled()?;
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
            {
                let command = self.scheduled_graph.build_cmd3_command_from_descriptors(
                    position,
                    &scheduled_experts,
                    ExpertPhaseInput::MetalPostAttention(prep),
                    shared_phase,
                    next_norm_weights,
                )?;
                let pending = self.metal.submit_scheduled_expert_command(command)?;
                if deepstack.is_none() && layer + 1 < self.config.num_hidden_layers {
                    deferred_expert_phase = Some(pending);
                } else {
                    let output = pending.wait()?;
                    token_state.apply_declared_expert_phase(
                        output,
                        FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
                    )?;
                }
            }
            trace_layer_values(position, layer, "moe", token_state.hidden());
            layer_timing.buckets.expert_compute += expert_compute_started.elapsed();
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
            if let Some(context) = deepstack
                && let Some(features_for_layer) = context.features.get(layer)
            {
                let feature = features_for_layer
                    .get(context.visual_index)
                    .with_context(|| {
                        format!(
                            "deepstack layer {layer} has no feature for visual token {}",
                            context.visual_index
                        )
                    })?;
                if feature.len() != token_state.hidden().len() {
                    bail!(
                        "deepstack feature for layer {layer} has len {}; expected {}",
                        feature.len(),
                        token_state.hidden().len()
                    );
                }
                add_in_place(token_state.hidden_mut(), feature);
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
        let scheduled_attention = self.scheduled_graph.build_attention_math(layer, position)?;
        scheduled_attention.resolve_kv_state(kv_record.state(FlashMoeStatePlacement::CpuVisible))
    }

    fn full_attention_projected(
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
        let attended = self.full_attention_output_values(
            layer,
            normed,
            deferred_input,
            kv_cache,
            position,
            rope_position,
            runtime,
            attention_buckets.as_deref_mut(),
        )?;
        let subphase_started = Instant::now();
        let projected = self.dense.project_with_metal(
            Some(&self.metal),
            layer,
            "o_proj",
            &attended,
            runtime.width,
        )?;
        if let Some(buckets) = attention_buckets.as_deref_mut() {
            buckets.attention_output_projection += subphase_started.elapsed();
        }
        Ok(projected)
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
        let router_score_command = self.scheduled_graph.build_router_score_projection(
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
