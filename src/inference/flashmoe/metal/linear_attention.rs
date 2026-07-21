use super::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalFusedLinearAttentionBuilder<'a> {
    runtime: &'a MetalRuntime,
    dense_weights: Option<&'a MetalDenseWeights>,
    linear_attention_state: &'a Mutex<MetalLinearAttentionStateCache>,
    buffers: &'a Arc<MetalBufferPool>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<'a> MetalFusedLinearAttentionBuilder<'a> {
    pub(crate) fn new(
        runtime: &'a MetalRuntime,
        dense_weights: Option<&'a MetalDenseWeights>,
        linear_attention_state: &'a Mutex<MetalLinearAttentionStateCache>,
        buffers: &'a Arc<MetalBufferPool>,
    ) -> Self {
        Self {
            runtime,
            dense_weights,
            linear_attention_state,
            buffers,
        }
    }

    unsafe fn buffer_with_bytes(&self, bytes: &[u8]) -> anyhow::Result<MetalObjcId> {
        unsafe { self.buffers.buffer_with_bytes(self.runtime.device, bytes) }
    }

    unsafe fn buffer_with_len(&self, len: usize) -> anyhow::Result<MetalObjcId> {
        unsafe { self.buffers.buffer_with_len(self.runtime.device, len) }
    }

    unsafe fn recycle(&self, buffer: MetalObjcId) {
        unsafe { self.buffers.recycle(buffer) }
    }

    fn recycle_or_release_buffers(&self, buffers: &[MetalObjcId], release_only: bool) {
        self.buffers.recycle_or_release(buffers, release_only);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl std::ops::Deref for MetalFusedLinearAttentionBuilder<'_> {
    type Target = MetalRuntime;

    fn deref(&self) -> &Self::Target {
        self.runtime
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn select_static_dtype_pipeline(
    dtype: &ResidentStaticDtype,
    bf16: MetalObjcId,
    f16: MetalObjcId,
    f32: MetalObjcId,
) -> MetalObjcId {
    match dtype {
        ResidentStaticDtype::Bf16 => bf16,
        ResidentStaticDtype::F16 => f16,
        ResidentStaticDtype::F32 => f32,
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalFusedLinearAttentionBuilder<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_layer_major_graph(
        &self,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        rows: usize,
        width: usize,
        input: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        norm_epsilon: f32,
    ) -> Result<MetalLayerMajorPostAttention> {
        let layer = bindings.layer;
        let projections = &bindings.input_projections;
        let out_proj = &bindings.out_proj;
        let router = &bindings.router;
        let static_tensors = &bindings.static_tensors;
        let hidden_values = rows
            .checked_mul(width)
            .context("Qwen layer-major hidden matrix size overflow")?;
        if rows == 0
            || width == 0
            || input.len() != hidden_values
            || residual.len() != hidden_values
            || post_norm_weight.len() != width
            || !norm_epsilon.is_finite()
            || norm_epsilon <= 0.0
            || projections[0].output_width() != layout.conv_dim
            || projections[1].output_width() != layout.total_value_width
            || projections[2].output_width() != layout.num_value_heads
            || projections[3].output_width() != layout.num_value_heads
            || out_proj.rows() != width
            || out_proj.cols() != layout.total_value_width
            || router.cols() != width
            || router.output_width() != router.rows()
            || static_tensors.conv_weight.values != layout.conv_dim * layout.conv_kernel_size
            || static_tensors.a_log.values != layout.num_value_heads
            || static_tensors.dt_bias.values != layout.num_value_heads
            || static_tensors.norm_weight.values != layout.value_dim
            || static_tensors.a_log.dtype != ResidentStaticDtype::F32
            || layout.key_dim == 0
            || layout.key_dim > 256
            || layout.value_dim == 0
            || layout.value_dim > 256
            || layout.num_key_heads == 0
            || layout.num_value_heads == 0
        {
            bail!(
                "Qwen layer-major linear-attention graph has incompatible geometry at layer {layer}: rows={rows} width={width} input={} residual={} norm={} out={}x{} router={}x{}",
                input.len(),
                residual.len(),
                post_norm_weight.len(),
                out_proj.rows(),
                out_proj.cols(),
                router.rows(),
                router.cols()
            );
        }
        let dense_weights = self.dense_weights.as_ref().context(
            "Qwen layer-major linear-attention graph requires resident dense Metal weights",
        )?;
        let mut projection_offsets = Vec::with_capacity(projections.len());
        let mut total_projection_rows = 0usize;
        for projection in projections {
            validate_resident_projection(projection, width, dense_weights.len)?;
            projection_offsets.push(total_projection_rows);
            total_projection_rows = total_projection_rows
                .checked_add(projection.rows())
                .context("Qwen linear-attention projection width overflow")?;
        }
        validate_resident_projection(out_proj, layout.total_value_width, dense_weights.len)?;
        validate_resident_projection(router, width, dense_weights.len)?;
        let q4_projections = projections
            .iter()
            .map(ResidentMmapMatvecProjection::q4)
            .collect::<Option<Vec<_>>>()
            .context(
                "Qwen layer-major linear-attention graph requires affine-Q4 input projections",
            )?;
        let projection_values = rows
            .checked_mul(total_projection_rows)
            .context("Qwen linear-attention packed projection size overflow")?;
        let attention_values = rows
            .checked_mul(layout.total_value_width)
            .context("Qwen linear-attention output matrix size overflow")?;
        let router_values = rows
            .checked_mul(router.rows())
            .context("Qwen linear-attention router matrix size overflow")?;

        let mut state_guard = self
            .linear_attention_state
            .lock()
            .expect("metal linear attention state poisoned");
        let state = state_guard
            .layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .with_context(|| {
                format!("Qwen layer-major linear-attention graph has no resolved state for layer {layer}")
            })?;
        if state.conv_dim != layout.conv_dim
            || state.total_value_width != layout.total_value_width
            || state.num_value_heads != layout.num_value_heads
            || state.conv_state_len != layout.conv_state_len()
            || state.ssm_state_len != layout.ssm_state_len()
        {
            bail!("Qwen layer-major linear-attention graph state does not match layer {layer}");
        }

        unsafe {
            let mut owned = Vec::with_capacity(20);
            let input_buffer =
                match input {
                    MetalBatchProjectionInput::Cpu(values) => self
                        .buffers
                        .tracked_buffer_with_bytes(self.device, f32_as_bytes(values), &mut owned)?,
                    MetalBatchProjectionInput::Buffer { buffer, .. } => buffer,
                };
            let residual_buffer =
                match residual {
                    MetalBatchProjectionInput::Cpu(values) => self
                        .buffers
                        .tracked_buffer_with_bytes(self.device, f32_as_bytes(values), &mut owned)?,
                    MetalBatchProjectionInput::Buffer { buffer, .. } => buffer,
                };
            let projection_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                projection_values * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let attention_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                attention_values * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let norm_weight_buffer = self.buffers.tracked_buffer_with_bytes(
                self.device,
                f32_as_bytes(post_norm_weight),
                &mut owned,
            )?;
            let projected_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                hidden_values * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let hidden_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                hidden_values * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let normed_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                hidden_values * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let router_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                router_values * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let projection_builder = MetalResidentProjectionBatchBuilder::new(
                self.runtime,
                self.dense_weights,
                self.buffers,
            );
            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Qwen layer-major linear-attention graph command buffer",
                "failed to create Qwen layer-major linear-attention graph encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.recycle_or_release_buffers(&owned, true);
                    drop(state_guard);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            let encode_result = (|| -> Result<()> {
                if !projection_builder.try_encode_q4_mmap_projection_batch(
                    encoder,
                    &q4_projections,
                    input_buffer,
                    rows,
                    projection_buffer,
                    &projection_offsets,
                    total_projection_rows,
                    &mut owned,
                )? {
                    bail!(
                        "Qwen layer-major linear-attention input projections did not resolve a matrix command"
                    );
                }

                let conv_dim_u32 = u32::try_from(layout.conv_dim)
                    .context("linear-attention conv width exceeds u32")?;
                let kernel_size_u32 = u32::try_from(layout.conv_kernel_size)
                    .context("linear-attention convolution width exceeds u32")?;
                let key_dim_u32 = u32::try_from(layout.key_dim)
                    .context("linear-attention key width exceeds u32")?;
                let value_dim_u32 = u32::try_from(layout.value_dim)
                    .context("linear-attention value width exceeds u32")?;
                let heads_u32 = u32::try_from(layout.num_value_heads)
                    .context("linear-attention head count exceeds u32")?;
                let heads_per_key_u32 = u32::try_from(layout.value_heads_per_key_head())
                    .context("linear-attention head ratio exceeds u32")?;
                let inv_scale = 1.0f32 / (layout.key_dim as f32).sqrt();
                let eps = 1e-6f32;
                for row in 0..rows {
                    let projection_row = row * total_projection_rows;
                    let qkv_offset = ((projection_row + projection_offsets[0])
                        * std::mem::size_of::<f32>()) as u64;
                    let z_offset = ((projection_row + projection_offsets[1])
                        * std::mem::size_of::<f32>()) as u64;
                    let beta_offset = ((projection_row + projection_offsets[2])
                        * std::mem::size_of::<f32>()) as u64;
                    let alpha_offset = ((projection_row + projection_offsets[3])
                        * std::mem::size_of::<f32>()) as u64;
                    let output_offset =
                        (row * layout.total_value_width * std::mem::size_of::<f32>()) as u64;

                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        select_static_dtype_pipeline(
                            &static_tensors.conv_weight.dtype,
                            self.pipelines.linear_conv1d_bf16_pipeline,
                            self.pipelines.linear_conv1d_f16_pipeline,
                            self.pipelines.linear_conv1d_f32_pipeline,
                        ),
                    );
                    set_buffer(encoder, state.conv_state, 0);
                    set_buffer_with_offset(encoder, projection_buffer, qkv_offset, 1);
                    set_buffer_with_offset(
                        encoder,
                        dense_weights.buffer(),
                        static_tensors.conv_weight.byte_offset,
                        2,
                    );
                    set_buffer(encoder, state.conv_output, 3);
                    set_bytes(encoder, u32_as_bytes(&conv_dim_u32), 4);
                    set_bytes(encoder, u32_as_bytes(&kernel_size_u32), 5);
                    dispatch_threads(encoder, layout.conv_dim as u64);

                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.pipelines.linear_rms_norm_qk_pipeline,
                    );
                    set_buffer(encoder, state.conv_output, 0);
                    set_buffer_with_offset(
                        encoder,
                        state.conv_output,
                        (layout.total_key_width * std::mem::size_of::<f32>()) as u64,
                        1,
                    );
                    set_bytes(encoder, u32_as_bytes(&key_dim_u32), 2);
                    set_bytes(encoder, f32_as_bytes(std::slice::from_ref(&inv_scale)), 3);
                    msg_send_void2_size(
                        encoder,
                        sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                        MetalDispatchSize::new(layout.num_key_heads as u64, 1, 1),
                        MetalDispatchSize::new(layout.key_dim as u64, 1, 1),
                    );

                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        select_static_dtype_pipeline(
                            &static_tensors.dt_bias.dtype,
                            self.pipelines.linear_decay_beta_bf16_pipeline,
                            self.pipelines.linear_decay_beta_f16_pipeline,
                            self.pipelines.linear_decay_beta_f32_pipeline,
                        ),
                    );
                    set_buffer_with_offset(encoder, projection_buffer, alpha_offset, 0);
                    set_buffer_with_offset(encoder, projection_buffer, beta_offset, 1);
                    set_buffer_with_offset(
                        encoder,
                        dense_weights.buffer(),
                        static_tensors.a_log.byte_offset,
                        2,
                    );
                    set_buffer_with_offset(
                        encoder,
                        dense_weights.buffer(),
                        static_tensors.dt_bias.byte_offset,
                        3,
                    );
                    set_buffer(encoder, state.g_decay, 4);
                    set_buffer(encoder, state.beta_gate, 5);
                    set_bytes(encoder, u32_as_bytes(&heads_u32), 6);
                    dispatch_threads(encoder, layout.num_value_heads as u64);

                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.pipelines.linear_delta_step_pipeline,
                    );
                    set_buffer(encoder, state.ssm_state, 0);
                    set_buffer(encoder, state.conv_output, 1);
                    set_buffer_with_offset(
                        encoder,
                        state.conv_output,
                        (layout.total_key_width * std::mem::size_of::<f32>()) as u64,
                        2,
                    );
                    set_buffer_with_offset(
                        encoder,
                        state.conv_output,
                        (2 * layout.total_key_width * std::mem::size_of::<f32>()) as u64,
                        3,
                    );
                    set_buffer(encoder, state.g_decay, 4);
                    set_buffer(encoder, state.beta_gate, 5);
                    set_buffer(encoder, state.delta_output, 6);
                    set_bytes(encoder, u32_as_bytes(&key_dim_u32), 7);
                    set_bytes(encoder, u32_as_bytes(&value_dim_u32), 8);
                    set_bytes(encoder, u32_as_bytes(&heads_per_key_u32), 9);
                    msg_send_void2_size(
                        encoder,
                        sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                        MetalDispatchSize::new(layout.num_value_heads as u64, 1, 1),
                        MetalDispatchSize::new(layout.value_dim as u64, 1, 1),
                    );

                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        select_static_dtype_pipeline(
                            &static_tensors.norm_weight.dtype,
                            self.pipelines.linear_gated_rms_norm_bf16_pipeline,
                            self.pipelines.linear_gated_rms_norm_f16_pipeline,
                            self.pipelines.linear_gated_rms_norm_f32_pipeline,
                        ),
                    );
                    set_buffer(encoder, state.delta_output, 0);
                    set_buffer_with_offset(encoder, projection_buffer, z_offset, 1);
                    set_buffer_with_offset(
                        encoder,
                        dense_weights.buffer(),
                        static_tensors.norm_weight.byte_offset,
                        2,
                    );
                    set_buffer_with_offset(encoder, attention_buffer, output_offset, 3);
                    set_bytes(encoder, u32_as_bytes(&value_dim_u32), 4);
                    set_bytes(encoder, f32_as_bytes(std::slice::from_ref(&eps)), 5);
                    msg_send_void2_size(
                        encoder,
                        sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                        MetalDispatchSize::new(layout.num_value_heads as u64, 1, 1),
                        MetalDispatchSize::new(layout.value_dim as u64, 1, 1),
                    );
                }

                match out_proj {
                    ResidentMmapMatvecProjection::Q4(out_q4) => {
                        if !projection_builder.try_encode_q4_mmap_projection_batch(
                            encoder,
                            &[out_q4],
                            attention_buffer,
                            rows,
                            projected_buffer,
                            &[0],
                            width,
                            &mut owned,
                        )? {
                            bail!(
                                "Qwen layer-major linear-attention output projection did not resolve a matrix command"
                            );
                        }
                    }
                    ResidentMmapMatvecProjection::Dense(out_dense) => encode_dense_resident_matrix(
                        &self.pipelines,
                        encoder,
                        dense_weights,
                        out_dense,
                        attention_buffer,
                        projected_buffer,
                        rows,
                    )?,
                }
                let width_u32 = u32::try_from(width)
                    .context("Qwen layer-major linear-attention hidden width exceeds u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.residual_rms_norm_pipeline,
                );
                set_buffer(encoder, projected_buffer, 0);
                set_buffer(encoder, residual_buffer, 1);
                set_buffer(encoder, norm_weight_buffer, 2);
                set_buffer(encoder, hidden_buffer, 3);
                set_buffer(encoder, normed_buffer, 4);
                set_bytes(encoder, u32_as_bytes(&width_u32), 5);
                set_bytes(
                    encoder,
                    f32_as_bytes(std::slice::from_ref(&norm_epsilon)),
                    6,
                );
                msg_send_void2_size(
                    encoder,
                    sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                    MetalDispatchSize::new(1, rows as u64, 1),
                    MetalDispatchSize::new(256, 1, 1),
                );
                match router {
                    ResidentMmapMatvecProjection::Q4(router_q4) => {
                        if !projection_builder.try_encode_q4_mmap_projection_batch(
                            encoder,
                            &[router_q4],
                            normed_buffer,
                            rows,
                            router_buffer,
                            &[0],
                            router.rows(),
                            &mut owned,
                        )? {
                            bail!(
                                "Qwen layer-major linear-attention router did not resolve a matrix command"
                            );
                        }
                    }
                    ResidentMmapMatvecProjection::Dense(router_dense) => {
                        encode_dense_resident_matrix(
                            &self.pipelines,
                            encoder,
                            dense_weights,
                            router_dense,
                            normed_buffer,
                            router_buffer,
                            rows,
                        )?
                    }
                }
                Ok(())
            })();
            if let Err(error) = encode_result {
                drop(encoding);
                self.recycle_or_release_buffers(&owned, true);
                drop(state_guard);
                return Err(error);
            }
            encoding.end_encoding();
            let context = MetalCommandContext::new("qwen_layer_major_linear_attention_graph")
                .with("layer", layer)
                .with("rows", rows)
                .with("width", width)
                .with("experts", router.rows());
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.recycle_or_release_buffers(&owned, error.should_release_buffers());
                drop(state_guard);
                return Err(error.into());
            }
            let router_scores = self.buffers.read_f32_buffer(router_buffer, router_values);
            let output = MetalLayerMajorPostAttention::new(
                Arc::clone(self.buffers),
                hidden_buffer,
                normed_buffer,
                rows,
                width,
                router_scores,
            );
            drop(encoding);
            drop(state_guard);
            match output {
                Ok(output) => {
                    for buffer in owned {
                        if buffer != hidden_buffer && buffer != normed_buffer {
                            self.recycle(buffer);
                        }
                    }
                    Ok(output)
                }
                Err(error) => {
                    self.recycle_or_release_buffers(&owned, false);
                    Err(error)
                }
            }
        }
    }

    pub(crate) fn execute(
        &self,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        input: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        top_k: usize,
    ) -> Result<MetalPostAttentionPrep> {
        let layer = bindings.layer;
        let projections = &bindings.input_projections;
        let out_proj = &bindings.out_proj;
        let router = &bindings.router;
        let static_tensors = &bindings.static_tensors;
        let residual_len = residual.len();
        if top_k == 0
            || residual_len == 0
            || residual_len != post_norm_weight.len()
            || projections[0].output_width() != layout.conv_dim
            || projections[1].output_width() != layout.total_value_width
            || projections[2].output_width() != layout.num_value_heads
            || projections[3].output_width() != layout.num_value_heads
            || out_proj.output_width() != residual_len
            || out_proj.rows() != residual_len
            || out_proj.cols() != layout.total_value_width
            || router.cols() != residual_len
            || router.output_width() != router.rows()
            || static_tensors.conv_weight.values != layout.conv_dim * layout.conv_kernel_size
            || static_tensors.a_log.values != layout.num_value_heads
            || static_tensors.dt_bias.values != layout.num_value_heads
            || static_tensors.norm_weight.values != layout.value_dim
            || static_tensors.a_log.dtype != ResidentStaticDtype::F32
            || layout.key_dim == 0
            || layout.value_dim == 0
            || layout.key_dim > 256
            || layout.value_dim > 256
            || layout.num_key_heads == 0
            || layout.num_value_heads == 0
        {
            bail!(
                "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD1/CMD2 path at layer {layer}: incompatible dimensions or routing policy (input projections {}, input width {}, residual width {residual_len}, norm width {}, topK {top_k})",
                projections.len(),
                input.len(),
                post_norm_weight.len()
            );
        }
        let dense_weights = self.dense_weights.as_ref().context(
            "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD1/CMD2 path: resident dense Metal weights are unavailable",
        )?;
        let input_len = input.len();
        let mut total_rows = 0usize;
        let mut output_offsets = Vec::with_capacity(projections.len());
        for projection in projections {
            validate_resident_projection(projection, input_len, dense_weights.len).with_context(
                || {
                    format!(
                        "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD1 path at layer {layer}: projection {} is incompatible",
                        projection.tensor_name()
                    )
                },
            )?;
            output_offsets.push(total_rows);
            total_rows = total_rows
                .checked_add(projection.rows())
                .context("linear-attention projection output row overflow")?;
        }
        validate_resident_projection(out_proj, layout.total_value_width, dense_weights.len)
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD2 path at layer {layer}: output projection {} is incompatible",
                    out_proj.tensor_name()
                )
            })?;
        validate_resident_projection(router, residual_len, dense_weights.len).with_context(|| {
            format!(
                "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD2 path at layer {layer}: router projection {} is incompatible",
                router.tensor_name()
            )
        })?;

        let output_byte_offsets = output_offsets
            .iter()
            .map(|offset| {
                offset
                    .checked_mul(std::mem::size_of::<f32>())
                    .map(|offset| offset as u64)
                    .context("linear-attention projection byte offset overflow")
            })
            .collect::<Result<Vec<_>>>()?;
        let qkv_offset = output_byte_offsets[0];
        let z_offset = output_byte_offsets[1];
        let beta_offset = output_byte_offsets[2];
        let alpha_offset = output_byte_offsets[3];

        let mut state_guard = self
            .linear_attention_state
            .lock()
            .expect("metal linear attention state poisoned");
        let state = state_guard
            .layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled Qwen3.5 linear-attention state path: layer {layer} has no resolved Metal recurrent state"
                )
            })?;
        if state.conv_dim != layout.conv_dim
            || state.total_value_width != layout.total_value_width
            || state.num_value_heads != layout.num_value_heads
            || state.conv_state_len != layout.conv_state_len()
            || state.ssm_state_len != layout.ssm_state_len()
        {
            bail!(
                "FlashMoe unsupported scheduled Qwen3.5 linear-attention state path: layer {layer} recurrent state does not match the resolved layout"
            );
        }

        unsafe {
            let (input_buffer, owned_input_buffer) = match input {
                MetalBatchProjectionInput::Cpu(input) => {
                    let buffer = self.buffer_with_bytes(f32_as_bytes(input))?;
                    (buffer, Some(buffer))
                }
                MetalBatchProjectionInput::Buffer { buffer, .. } => (buffer, None),
            };
            let projection_buffer =
                self.buffer_with_len(total_rows * std::mem::size_of::<f32>())?;
            let attention_output_buffer =
                self.buffer_with_len(layout.total_value_width * std::mem::size_of::<f32>())?;
            let (residual_input_buffer, owned_residual_input_buffer) = match residual {
                MetalBatchProjectionInput::Cpu(residual) => {
                    let buffer = self.buffer_with_bytes(f32_as_bytes(residual))?;
                    (buffer, Some(buffer))
                }
                MetalBatchProjectionInput::Buffer { buffer, .. } => (buffer, None),
            };
            let norm_weight_buffer = self.buffer_with_bytes(f32_as_bytes(post_norm_weight))?;
            let projected_buffer =
                self.buffer_with_len(residual_len * std::mem::size_of::<f32>())?;
            let residual_buffer =
                self.buffer_with_len(residual_len * std::mem::size_of::<f32>())?;
            let normed_buffer = self.buffer_with_len(residual_len * std::mem::size_of::<f32>())?;
            let router_logits_buffer =
                self.buffer_with_len(router.rows() * std::mem::size_of::<f32>())?;
            let mut owned_buffers = vec![
                projection_buffer,
                attention_output_buffer,
                norm_weight_buffer,
                projected_buffer,
                residual_buffer,
                normed_buffer,
                router_logits_buffer,
            ];
            if let Some(buffer) = owned_residual_input_buffer {
                owned_buffers.push(buffer);
            }
            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE fused linear-attention command buffer",
                "failed to create Flash-MoE fused linear-attention compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    if let Some(buffer) = owned_input_buffer {
                        self.recycle(buffer);
                    }
                    self.recycle_or_release_buffers(&owned_buffers, true);
                    drop(state_guard);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();

            for (idx, projection) in projections.iter().enumerate() {
                if let Err(error) = encode_resident_projection(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    projection,
                    input_buffer,
                    projection_buffer,
                    output_byte_offsets[idx],
                ) {
                    drop(encoding);
                    if let Some(buffer) = owned_input_buffer {
                        self.recycle_or_release_buffers(&[buffer], true);
                    }
                    self.recycle_or_release_buffers(&owned_buffers, true);
                    drop(state_guard);
                    return Err(error);
                }
            }

            let conv_dim_u32 = layout.conv_dim as u32;
            let kernel_size_u32 = layout.conv_kernel_size as u32;
            let key_dim_u32 = layout.key_dim as u32;
            let value_dim_u32 = layout.value_dim as u32;
            let heads_u32 = layout.num_value_heads as u32;
            let heads_per_key_u32 = layout.value_heads_per_key_head() as u32;
            let inv_scale = 1.0f32 / (layout.key_dim as f32).sqrt();
            let eps = 1e-6f32;

            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                select_static_dtype_pipeline(
                    &static_tensors.conv_weight.dtype,
                    self.pipelines.linear_conv1d_bf16_pipeline,
                    self.pipelines.linear_conv1d_f16_pipeline,
                    self.pipelines.linear_conv1d_f32_pipeline,
                ),
            );
            set_buffer(encoder, state.conv_state, 0);
            set_buffer_with_offset(encoder, projection_buffer, qkv_offset, 1);
            set_buffer_with_offset(
                encoder,
                dense_weights.buffer(),
                static_tensors.conv_weight.byte_offset,
                2,
            );
            set_buffer(encoder, state.conv_output, 3);
            set_bytes(encoder, u32_as_bytes(&conv_dim_u32), 4);
            set_bytes(encoder, u32_as_bytes(&kernel_size_u32), 5);
            dispatch_threads(encoder, layout.conv_dim as u64);

            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.pipelines.linear_rms_norm_qk_pipeline,
            );
            set_buffer(encoder, state.conv_output, 0);
            set_buffer_with_offset(
                encoder,
                state.conv_output,
                (layout.total_key_width * std::mem::size_of::<f32>()) as u64,
                1,
            );
            set_bytes(encoder, u32_as_bytes(&key_dim_u32), 2);
            set_bytes(encoder, f32_as_bytes(std::slice::from_ref(&inv_scale)), 3);
            msg_send_void2_size(
                encoder,
                sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                MetalDispatchSize {
                    width: layout.num_key_heads as u64,
                    height: 1,
                    depth: 1,
                },
                MetalDispatchSize {
                    width: layout.key_dim as u64,
                    height: 1,
                    depth: 1,
                },
            );

            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                select_static_dtype_pipeline(
                    &static_tensors.dt_bias.dtype,
                    self.pipelines.linear_decay_beta_bf16_pipeline,
                    self.pipelines.linear_decay_beta_f16_pipeline,
                    self.pipelines.linear_decay_beta_f32_pipeline,
                ),
            );
            set_buffer_with_offset(encoder, projection_buffer, alpha_offset, 0);
            set_buffer_with_offset(encoder, projection_buffer, beta_offset, 1);
            set_buffer_with_offset(
                encoder,
                dense_weights.buffer(),
                static_tensors.a_log.byte_offset,
                2,
            );
            set_buffer_with_offset(
                encoder,
                dense_weights.buffer(),
                static_tensors.dt_bias.byte_offset,
                3,
            );
            set_buffer(encoder, state.g_decay, 4);
            set_buffer(encoder, state.beta_gate, 5);
            set_bytes(encoder, u32_as_bytes(&heads_u32), 6);
            dispatch_threads(encoder, layout.num_value_heads as u64);

            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.pipelines.linear_delta_step_pipeline,
            );
            set_buffer(encoder, state.ssm_state, 0);
            set_buffer(encoder, state.conv_output, 1);
            set_buffer_with_offset(
                encoder,
                state.conv_output,
                (layout.total_key_width * std::mem::size_of::<f32>()) as u64,
                2,
            );
            set_buffer_with_offset(
                encoder,
                state.conv_output,
                (2 * layout.total_key_width * std::mem::size_of::<f32>()) as u64,
                3,
            );
            set_buffer(encoder, state.g_decay, 4);
            set_buffer(encoder, state.beta_gate, 5);
            set_buffer(encoder, state.delta_output, 6);
            set_bytes(encoder, u32_as_bytes(&key_dim_u32), 7);
            set_bytes(encoder, u32_as_bytes(&value_dim_u32), 8);
            set_bytes(encoder, u32_as_bytes(&heads_per_key_u32), 9);
            msg_send_void2_size(
                encoder,
                sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                MetalDispatchSize {
                    width: layout.num_value_heads as u64,
                    height: 1,
                    depth: 1,
                },
                MetalDispatchSize {
                    width: layout.value_dim as u64,
                    height: 1,
                    depth: 1,
                },
            );

            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                select_static_dtype_pipeline(
                    &static_tensors.norm_weight.dtype,
                    self.pipelines.linear_gated_rms_norm_bf16_pipeline,
                    self.pipelines.linear_gated_rms_norm_f16_pipeline,
                    self.pipelines.linear_gated_rms_norm_f32_pipeline,
                ),
            );
            set_buffer(encoder, state.delta_output, 0);
            set_buffer_with_offset(encoder, projection_buffer, z_offset, 1);
            set_buffer_with_offset(
                encoder,
                dense_weights.buffer(),
                static_tensors.norm_weight.byte_offset,
                2,
            );
            set_buffer(encoder, attention_output_buffer, 3);
            set_bytes(encoder, u32_as_bytes(&value_dim_u32), 4);
            set_bytes(encoder, f32_as_bytes(std::slice::from_ref(&eps)), 5);
            msg_send_void2_size(
                encoder,
                sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                MetalDispatchSize {
                    width: layout.num_value_heads as u64,
                    height: 1,
                    depth: 1,
                },
                MetalDispatchSize {
                    width: layout.value_dim as u64,
                    height: 1,
                    depth: 1,
                },
            );
            let post_projection_result = (|| -> Result<()> {
                encode_resident_projection(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    out_proj,
                    attention_output_buffer,
                    projected_buffer,
                    0,
                )?;
                let width_u32 = u32::try_from(residual_len)
                    .context("linear-attention residual width exceeds u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.residual_rms_norm_pipeline,
                );
                set_buffer(encoder, projected_buffer, 0);
                set_buffer(encoder, residual_input_buffer, 1);
                set_buffer(encoder, norm_weight_buffer, 2);
                set_buffer(encoder, residual_buffer, 3);
                set_buffer(encoder, normed_buffer, 4);
                set_bytes(encoder, u32_as_bytes(&width_u32), 5);
                set_bytes(encoder, f32_as_bytes(std::slice::from_ref(&eps)), 6);
                dispatch_single_threadgroup(encoder, 256);
                encode_resident_projection(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    router,
                    normed_buffer,
                    router_logits_buffer,
                    0,
                )
            })();
            if let Err(error) = post_projection_result {
                drop(encoding);
                if let Some(buffer) = owned_input_buffer {
                    self.recycle_or_release_buffers(&[buffer], true);
                }
                self.recycle_or_release_buffers(&owned_buffers, true);
                drop(state_guard);
                return Err(error);
            }

            encoding.end_encoding();

            let active_count = top_k.min(router.rows()).max(1);
            let context = MetalCommandContext::new("linear_attention_fused_post")
                .with("layer", layer)
                .with("projections", projections.len())
                .with("rows", total_rows)
                .with("input_len", input_len)
                .with("experts", router.rows())
                .with("top_k", active_count);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                if let Some(buffer) = owned_input_buffer {
                    self.recycle_or_release_buffers(&[buffer], error.should_release_buffers());
                }
                self.recycle_or_release_buffers(&owned_buffers, error.should_release_buffers());
                drop(state_guard);
                return Err(error.into());
            }

            let router_logits_ptr =
                msg_send_ptr0(router_logits_buffer, sel("contents")).cast::<f32>();
            let router_scores =
                std::slice::from_raw_parts(router_logits_ptr, router.rows()).to_vec();
            let active = routing_softmax_top_k(&router_scores, active_count);

            drop(encoding);
            if let Some(buffer) = owned_input_buffer {
                self.recycle(buffer);
            }
            if let Some(buffer) = owned_residual_input_buffer {
                self.recycle(buffer);
            }
            for buffer in [
                projection_buffer,
                attention_output_buffer,
                norm_weight_buffer,
                projected_buffer,
                router_logits_buffer,
            ] {
                self.recycle(buffer);
            }
            drop(state_guard);
            Ok(MetalPostAttentionPrep::new(
                layer,
                residual_len,
                router.rows(),
                active,
                residual_buffer,
                normed_buffer,
            )?)
        }
    }
}
