use super::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalResidentProjectionBatchBuilder<'a> {
    runtime: &'a MetalRuntime,
    dense_weights: Option<&'a MetalDenseWeights>,
    buffers: &'a Arc<MetalBufferPool>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalLayerMajorPostAttention {
    buffers: Arc<MetalBufferPool>,
    residual: MetalMatrixBuffer,
    normed: MetalMatrixBuffer,
    router_scores: Vec<f32>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalQwenAttentionRows {
    buffers: Arc<MetalBufferPool>,
    values: MetalMatrixBuffer,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalQwenFullAttentionOutput {
    attention: MetalQwenAttentionRows,
    current_keys: Vec<f32>,
    current_values: Vec<f32>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalQwenFullAttentionOutput {
    pub(crate) fn new(
        attention: MetalQwenAttentionRows,
        current_keys: Vec<f32>,
        current_values: Vec<f32>,
    ) -> Result<Self> {
        if current_keys.is_empty() || current_keys.len() != current_values.len() {
            bail!("Qwen full-attention graph returned incompatible current KV matrices");
        }
        Ok(Self {
            attention,
            current_keys,
            current_values,
        })
    }

    pub(crate) fn current_keys(&self) -> &[f32] {
        &self.current_keys
    }

    pub(crate) fn current_values(&self) -> &[f32] {
        &self.current_values
    }

    pub(crate) fn into_attention(self) -> MetalQwenAttentionRows {
        self.attention
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalQwenAttentionRows {
    pub(crate) fn new(
        buffers: Arc<MetalBufferPool>,
        buffer: MetalObjcId,
        rows: usize,
        width: usize,
    ) -> Result<Self> {
        Ok(Self {
            buffers,
            values: MetalMatrixBuffer::new(
                buffer,
                FlashMoeGpuMatrixDescriptor::attention_values(rows, width)?,
            )?,
        })
    }

    pub(crate) fn values(&self) -> MetalMatrixBuffer {
        self.values
    }

    pub(crate) fn materialize(&self) -> Vec<f32> {
        unsafe {
            self.buffers
                .read_f32_buffer(self.values.buffer(), self.values.values())
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalQwenAttentionRows {
    fn drop(&mut self) {
        self.buffers
            .recycle_or_release(&[self.values.buffer()], false);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalLayerMajorPostAttention {
    pub(crate) fn new(
        buffers: Arc<MetalBufferPool>,
        residual_buffer: MetalObjcId,
        normed_buffer: MetalObjcId,
        rows: usize,
        width: usize,
        router_scores: Vec<f32>,
    ) -> Result<Self> {
        if residual_buffer == normed_buffer {
            bail!("Qwen layer-major post-attention matrices cannot alias");
        }
        let residual = MetalMatrixBuffer::new(
            residual_buffer,
            FlashMoeGpuMatrixDescriptor::residual(rows, width)?,
        )?;
        let normed = MetalMatrixBuffer::new(
            normed_buffer,
            FlashMoeGpuMatrixDescriptor::normed(rows, width)?,
        )?;
        Ok(Self {
            buffers,
            residual,
            normed,
            router_scores,
        })
    }

    pub(crate) fn residual(&self) -> MetalMatrixBuffer {
        self.residual
    }

    pub(crate) fn normed(&self) -> MetalMatrixBuffer {
        self.normed
    }

    pub(crate) fn router_scores(&self) -> &[f32] {
        &self.router_scores
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalLayerMajorPostAttention {
    fn drop(&mut self) {
        self.buffers
            .recycle_or_release(&[self.residual.buffer(), self.normed.buffer()], false);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalQwenPrefillLayerOutput {
    buffers: Arc<MetalBufferPool>,
    layer: usize,
    hidden: MetalMatrixBuffer,
    next_normed: Option<MetalMatrixBuffer>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalQwenPrefillLayerOutput {
    pub(crate) fn new(
        buffers: Arc<MetalBufferPool>,
        layer: usize,
        hidden_buffer: MetalObjcId,
        next_normed_buffer: Option<MetalObjcId>,
        rows: usize,
        width: usize,
    ) -> Result<Self> {
        if next_normed_buffer == Some(hidden_buffer) {
            bail!("Qwen prefill hidden and next-norm matrices cannot alias");
        }
        let hidden = MetalMatrixBuffer::new(
            hidden_buffer,
            FlashMoeGpuMatrixDescriptor::hidden(rows, width)?,
        )?;
        let next_normed = next_normed_buffer
            .map(|buffer| {
                MetalMatrixBuffer::new(
                    buffer,
                    FlashMoeGpuMatrixDescriptor::next_layer_normed(rows, width)?,
                )
            })
            .transpose()?;
        Ok(Self {
            buffers,
            layer,
            hidden,
            next_normed,
        })
    }

    pub(crate) fn layer(&self) -> usize {
        self.layer
    }

    pub(crate) fn hidden(&self) -> MetalMatrixBuffer {
        self.hidden
    }

    pub(crate) fn next_normed(&self) -> Option<MetalMatrixBuffer> {
        self.next_normed
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalQwenPrefillLayerOutput {
    fn drop(&mut self) {
        let mut buffers = vec![self.hidden.buffer()];
        if let Some(next_normed) = self.next_normed {
            buffers.push(next_normed.buffer());
        }
        self.buffers.recycle_or_release(&buffers, false);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn encode_q4_mmap_multilinear(
    pipelines: &MetalPipelineSet<MetalObjcId>,
    encoder: MetalObjcId,
    dense_weights: &MetalDenseWeights,
    projection: &DenseQ4MmapMatvecProjection,
    input_buffer: MetalObjcId,
    output_buffer: MetalObjcId,
    rows_per_head: usize,
) -> Result<()> {
    unsafe {
        let rows_u32 = u32::try_from(projection.rows).context("multilinear rows do not fit u32")?;
        let cols_u32 = u32::try_from(projection.cols).context("multilinear cols do not fit u32")?;
        let groups_u32 = u32::try_from(projection.groups_per_row)
            .context("multilinear groups do not fit u32")?;
        let group_size_u32 = u32::try_from(projection.group_size)
            .context("multilinear group size does not fit u32")?;
        let rows_per_head_u32 =
            u32::try_from(rows_per_head).context("multilinear rows per head do not fit u32")?;

        msg_send_void1_id(
            encoder,
            sel("setComputePipelineState:"),
            pipelines.q4_mmap_multilinear_bf16_scale_bias_pipeline,
        );
        set_buffer(encoder, dense_weights.buffer(), 0);
        set_buffer(encoder, input_buffer, 1);
        set_buffer(encoder, output_buffer, 2);
        set_bytes(encoder, u64_as_bytes(&projection.packed_byte_offset), 3);
        set_bytes(encoder, u64_as_bytes(&projection.scales_byte_offset), 4);
        set_bytes(encoder, u64_as_bytes(&projection.biases_byte_offset), 5);
        set_bytes(encoder, u32_as_bytes(&rows_u32), 6);
        set_bytes(encoder, u32_as_bytes(&cols_u32), 7);
        set_bytes(encoder, u32_as_bytes(&groups_u32), 8);
        set_bytes(encoder, u32_as_bytes(&group_size_u32), 9);
        set_bytes(encoder, u32_as_bytes(&rows_per_head_u32), 10);
        dispatch_q4_mmap_threadgroups(encoder, projection.rows as u64);
        Ok(())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<'a> MetalResidentProjectionBatchBuilder<'a> {
    pub(crate) fn new(
        runtime: &'a MetalRuntime,
        dense_weights: Option<&'a MetalDenseWeights>,
        buffers: &'a Arc<MetalBufferPool>,
    ) -> Self {
        Self {
            runtime,
            dense_weights,
            buffers,
        }
    }

    unsafe fn buffer_with_bytes(&self, bytes: &[u8]) -> Result<MetalObjcId> {
        unsafe { self.buffers.buffer_with_bytes(self.runtime.device, bytes) }
    }

    unsafe fn buffer_with_len(&self, len: usize) -> Result<MetalObjcId> {
        unsafe { self.buffers.buffer_with_len(self.runtime.device, len) }
    }

    unsafe fn recycle(&self, buffer: MetalObjcId) {
        unsafe { self.buffers.recycle(buffer) }
    }

    fn recycle_or_release_buffers(&self, buffers: &[MetalObjcId], release_only: bool) {
        self.buffers.recycle_or_release(buffers, release_only);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_glm_mla_input_projection_chain(
        &self,
        q_a: &ResidentMmapMatvecProjection,
        kv_a: &ResidentMmapMatvecProjection,
        q_b: &ResidentMmapMatvecProjection,
        input: MetalBatchProjectionInput<'_>,
        q_norm_weight: &[f32],
        kv_norm_weight: &[f32],
        kv_lora_rank: usize,
        norm_epsilon: f32,
    ) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let Some(dense_weights) = &self.dense_weights else {
            return Ok(None);
        };
        let input_len = input.len();
        validate_resident_projection(q_a, input_len, dense_weights.len)?;
        validate_resident_projection(kv_a, input_len, dense_weights.len)?;
        validate_resident_projection(q_b, q_a.output_width(), dense_weights.len)?;
        if q_a.rows() != q_a.output_width()
            || kv_a.rows() != kv_a.output_width()
            || q_b.rows() != q_b.output_width()
            || q_norm_weight.len() != q_a.output_width()
            || kv_lora_rank == 0
            || kv_lora_rank > kv_a.output_width()
            || kv_norm_weight.len() != kv_lora_rank
            || !norm_epsilon.is_finite()
            || norm_epsilon <= 0.0
        {
            bail!(
                "GLM MLA fused input projection chain has incompatible shapes q_a={}x{} output={} kv_a={}x{} output={} q_b={}x{} output={} q_norm={} kv_norm={} kv_lora_rank={} epsilon={norm_epsilon}",
                q_a.rows(),
                q_a.cols(),
                q_a.output_width(),
                kv_a.rows(),
                kv_a.cols(),
                kv_a.output_width(),
                q_b.rows(),
                q_b.cols(),
                q_b.output_width(),
                q_norm_weight.len(),
                kv_norm_weight.len(),
                kv_lora_rank,
            );
        }

        unsafe {
            let mut buffers = Vec::with_capacity(6);
            let input_buffer = match input {
                MetalBatchProjectionInput::Cpu(values) => self.buffers.tracked_buffer_with_bytes(
                    self.runtime.device,
                    f32_as_bytes(values),
                    &mut buffers,
                )?,
                MetalBatchProjectionInput::Buffer { buffer, .. } => buffer,
            };
            let q_a_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                q_a.rows() * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let kv_a_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                kv_a.rows() * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let query_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                q_b.rows() * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let q_norm_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(q_norm_weight),
                &mut buffers,
            )?;
            let kv_norm_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(kv_norm_weight),
                &mut buffers,
            )?;

            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE GLM MLA input projection command buffer",
                "failed to create Flash-MoE GLM MLA input projection compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            let encode_result = (|| -> Result<()> {
                encode_resident_projection(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    q_a,
                    input_buffer,
                    q_a_buffer,
                    0,
                )?;
                encode_resident_projection(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    kv_a,
                    input_buffer,
                    kv_a_buffer,
                    0,
                )?;

                let q_width = u32::try_from(q_a.output_width())
                    .context("GLM MLA q_a width does not fit u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.rms_norm_reduced_pipeline,
                );
                set_buffer(encoder, q_a_buffer, 0);
                set_buffer(encoder, q_norm_buffer, 1);
                set_buffer(encoder, q_a_buffer, 2);
                set_bytes(encoder, u32_as_bytes(&q_width), 3);
                set_bytes(
                    encoder,
                    f32_as_bytes(std::slice::from_ref(&norm_epsilon)),
                    4,
                );
                dispatch_single_threadgroup(encoder, 256);

                let kv_width =
                    u32::try_from(kv_lora_rank).context("GLM MLA KV LoRA rank does not fit u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.rms_norm_reduced_pipeline,
                );
                set_buffer(encoder, kv_a_buffer, 0);
                set_buffer(encoder, kv_norm_buffer, 1);
                set_buffer(encoder, kv_a_buffer, 2);
                set_bytes(encoder, u32_as_bytes(&kv_width), 3);
                set_bytes(
                    encoder,
                    f32_as_bytes(std::slice::from_ref(&norm_epsilon)),
                    4,
                );
                dispatch_single_threadgroup(encoder, 256);

                encode_resident_projection(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    q_b,
                    q_a_buffer,
                    query_buffer,
                    0,
                )?;
                Ok(())
            })();
            if let Err(error) = encode_result {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, true);
                return Err(error);
            }
            encoding.end_encoding();

            let context = MetalCommandContext::new("glm_mla_input_projection_chain")
                .with("input_len", input_len)
                .with("q_a", q_a.tensor_name())
                .with("kv_a", kv_a.tensor_name())
                .with("q_b", q_b.tensor_name())
                .with("q_width", q_b.output_width())
                .with("kv_width", kv_a.output_width());
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, error.should_release_buffers());
                return Err(error.into());
            }
            let query = self
                .buffers
                .read_f32_buffer(query_buffer, q_b.output_width());
            let compressed = self
                .buffers
                .read_f32_buffer(kv_a_buffer, kv_a.output_width());
            drop(encoding);
            for buffer in buffers {
                self.recycle(buffer);
            }
            Ok(Some((query, compressed)))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_glm_mla_fused_attention(
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
    ) -> Result<Option<MetalGlmMlaFusedAttentionOutput>> {
        let Some(dense_weights) = &self.dense_weights else {
            return Ok(None);
        };
        let input_len = input.input.len();
        validate_resident_projection(q_a, input_len, dense_weights.len)?;
        validate_resident_projection(kv_a, input_len, dense_weights.len)?;
        validate_resident_projection(q_b, q_a.output_width(), dense_weights.len)?;
        validate_resident_projection(embed_q, input.nope_dim, dense_weights.len)?;
        validate_resident_projection(unembed_out, input.latent_rank, dense_weights.len)?;

        let query_width = input
            .heads
            .checked_mul(
                input
                    .nope_dim
                    .checked_add(input.rope_dim)
                    .context("GLM MLA fused query head width overflow")?,
            )
            .context("GLM MLA fused query width overflow")?;
        let query_nope_width = input
            .heads
            .checked_mul(input.nope_dim)
            .context("GLM MLA fused no-PE query width overflow")?;
        let query_rope_width = input
            .heads
            .checked_mul(input.rope_dim)
            .context("GLM MLA fused rotary-query width overflow")?;
        let previous_records = input
            .previous_record_latents
            .len()
            .checked_div(input.latent_rank.max(1))
            .context("GLM MLA fused previous-record count division failed")?;
        let sequence = previous_records
            .checked_add(1)
            .context("GLM MLA fused sequence overflow")?;
        let record_latent_width = sequence
            .checked_mul(input.latent_rank)
            .context("GLM MLA fused latent-record width overflow")?;
        let record_rotary_width = sequence
            .checked_mul(input.rope_dim)
            .context("GLM MLA fused rotary-record width overflow")?;
        let absorbed_width = input
            .heads
            .checked_mul(input.latent_rank)
            .context("GLM MLA fused absorbed-query width overflow")?;
        let score_count = input
            .heads
            .checked_mul(sequence)
            .context("GLM MLA fused score count overflow")?;
        let prepare_count = query_nope_width
            .checked_add(query_rope_width)
            .and_then(|values| values.checked_add(input.latent_rank))
            .and_then(|values| values.checked_add(input.rope_dim))
            .context("GLM MLA fused preparation width overflow")?;
        let output_rows_per_head = unembed_out
            .rows
            .checked_div(input.heads.max(1))
            .context("GLM MLA fused output rows-per-head division failed")?;
        let post_plan = input
            .post_attention
            .map(|post| {
                post.projections.resident_plan(
                    unembed_out.rows,
                    post.residual.len(),
                    post.post_norm_weight.len(),
                )
            })
            .transpose()?;
        if let (Some(post), Some(plan)) = (input.post_attention, post_plan) {
            validate_resident_projection(
                &post.projections.out_proj,
                unembed_out.rows,
                dense_weights.len,
            )?;
            validate_resident_projection(&post.projections.router, plan.width, dense_weights.len)?;
            if post
                .router_correction_bias
                .is_some_and(|bias| bias.len() != plan.experts)
            {
                bail!(
                    "GLM MLA fused post-attention router correction bias length does not match {} experts",
                    plan.experts
                );
            }
        }
        let scale_bias_is_bf16 = |projection: &DenseQ4MmapMatvecProjection| {
            projection
                .scale_bias_dtype
                .eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16)
        };
        if input.heads == 0
            || input.latent_rank == 0
            || input.nope_dim == 0
            || input.rope_dim == 0
            || !input.rope_dim.is_multiple_of(2)
            || !input.latent_rank.is_multiple_of(16)
            || !input.scale.is_finite()
            || !norm_epsilon.is_finite()
            || norm_epsilon <= 0.0
            || q_a.rows() != q_a.output_width()
            || kv_a.rows() != input.latent_rank + input.rope_dim
            || kv_a.output_width() != input.latent_rank + input.rope_dim
            || q_b.rows() != query_width
            || q_b.output_width() != query_width
            || q_norm_weight.len() != q_a.output_width()
            || kv_norm_weight.len() != input.latent_rank
            || embed_q.rows != absorbed_width
            || embed_q.output_width != absorbed_width
            || embed_q.cols != input.nope_dim
            || !scale_bias_is_bf16(embed_q)
            || unembed_out.rows == 0
            || unembed_out.rows % input.heads != 0
            || unembed_out.output_width != unembed_out.rows
            || unembed_out.cols != input.latent_rank
            || !output_rows_per_head.is_multiple_of(16)
            || !scale_bias_is_bf16(unembed_out)
            || input.previous_record_latents.len()
                != previous_records.saturating_mul(input.latent_rank)
            || input.previous_record_rotary.len() != previous_records.saturating_mul(input.rope_dim)
            || input.rope_cos.len() != input.rope_dim / 2
            || input.rope_sin.len() != input.rope_dim / 2
        {
            bail!(
                "GLM MLA fused Metal attention has incompatible shapes q_a={}x{} output={} kv_a={}x{} output={} q_b={}x{} output={} embed={}x{} output={} unembed={}x{} output={} heads={} latent={} nope={} rope={} previous_latents={} previous_rotary={} cos={} sin={} q_norm={} kv_norm={} scale={} epsilon={norm_epsilon}",
                q_a.rows(),
                q_a.cols(),
                q_a.output_width(),
                kv_a.rows(),
                kv_a.cols(),
                kv_a.output_width(),
                q_b.rows(),
                q_b.cols(),
                q_b.output_width(),
                embed_q.rows,
                embed_q.cols,
                embed_q.output_width,
                unembed_out.rows,
                unembed_out.cols,
                unembed_out.output_width,
                input.heads,
                input.latent_rank,
                input.nope_dim,
                input.rope_dim,
                input.previous_record_latents.len(),
                input.previous_record_rotary.len(),
                input.rope_cos.len(),
                input.rope_sin.len(),
                q_norm_weight.len(),
                kv_norm_weight.len(),
                input.scale,
            );
        }

        let mut record_latents = Vec::with_capacity(record_latent_width);
        record_latents.extend_from_slice(input.previous_record_latents);
        record_latents.resize(record_latent_width, 0.0);
        let mut record_rotary = Vec::with_capacity(record_rotary_width);
        record_rotary.extend_from_slice(input.previous_record_rotary);
        record_rotary.resize(record_rotary_width, 0.0);

        unsafe {
            let mut buffers = Vec::with_capacity(24);
            let input_buffer = match input.input {
                MetalBatchProjectionInput::Cpu(values) => self.buffers.tracked_buffer_with_bytes(
                    self.runtime.device,
                    f32_as_bytes(values),
                    &mut buffers,
                )?,
                MetalBatchProjectionInput::Buffer { buffer, .. } => buffer,
            };
            let q_a_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                q_a.rows() * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let kv_a_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                kv_a.rows() * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let query_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                query_width * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let q_norm_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(q_norm_weight),
                &mut buffers,
            )?;
            let kv_norm_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(kv_norm_weight),
                &mut buffers,
            )?;
            let query_nope_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                query_nope_width * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let query_rope_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                query_rope_width * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let record_latents_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(&record_latents),
                &mut buffers,
            )?;
            let record_rotary_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(&record_rotary),
                &mut buffers,
            )?;
            let rope_cos_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(input.rope_cos),
                &mut buffers,
            )?;
            let rope_sin_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(input.rope_sin),
                &mut buffers,
            )?;
            let absorbed_queries_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                absorbed_width * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let scores_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                score_count * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let contexts_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                absorbed_width * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let output_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                unembed_out.rows * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let post_buffers = if let (Some(post), Some(plan)) = (input.post_attention, post_plan) {
                let residual_input_buffer = match post.residual {
                    MetalBatchProjectionInput::Cpu(values) => {
                        self.buffers.tracked_buffer_with_bytes(
                            self.runtime.device,
                            f32_as_bytes(values),
                            &mut buffers,
                        )?
                    }
                    MetalBatchProjectionInput::Buffer { buffer, .. } => buffer,
                };
                let norm_weight_buffer = self.buffers.tracked_buffer_with_bytes(
                    self.runtime.device,
                    f32_as_bytes(post.post_norm_weight),
                    &mut buffers,
                )?;
                let projected_buffer = self.buffers.tracked_buffer_with_len(
                    self.runtime.device,
                    plan.width * std::mem::size_of::<f32>(),
                    &mut buffers,
                )?;
                let residual_buffer = self.buffers.tracked_buffer_with_len(
                    self.runtime.device,
                    plan.width * std::mem::size_of::<f32>(),
                    &mut buffers,
                )?;
                let normed_buffer = self.buffers.tracked_buffer_with_len(
                    self.runtime.device,
                    plan.width * std::mem::size_of::<f32>(),
                    &mut buffers,
                )?;
                let router_logits_buffer = self.buffers.tracked_buffer_with_len(
                    self.runtime.device,
                    plan.experts * std::mem::size_of::<f32>(),
                    &mut buffers,
                )?;
                Some((
                    residual_input_buffer,
                    norm_weight_buffer,
                    projected_buffer,
                    residual_buffer,
                    normed_buffer,
                    router_logits_buffer,
                ))
            } else {
                None
            };

            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE fused GLM MLA command buffer",
                "failed to create Flash-MoE fused GLM MLA compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            let encode_result = (|| -> Result<()> {
                encode_resident_projection(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    q_a,
                    input_buffer,
                    q_a_buffer,
                    0,
                )?;
                encode_resident_projection(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    kv_a,
                    input_buffer,
                    kv_a_buffer,
                    0,
                )?;

                let q_rank = u32::try_from(q_a.output_width())
                    .context("GLM MLA fused Q LoRA rank does not fit u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.rms_norm_reduced_pipeline,
                );
                set_buffer(encoder, q_a_buffer, 0);
                set_buffer(encoder, q_norm_buffer, 1);
                set_buffer(encoder, q_a_buffer, 2);
                set_bytes(encoder, u32_as_bytes(&q_rank), 3);
                set_bytes(
                    encoder,
                    f32_as_bytes(std::slice::from_ref(&norm_epsilon)),
                    4,
                );
                dispatch_single_threadgroup(encoder, 256);

                let latent_rank = u32::try_from(input.latent_rank)
                    .context("GLM MLA fused latent rank does not fit u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.rms_norm_reduced_pipeline,
                );
                set_buffer(encoder, kv_a_buffer, 0);
                set_buffer(encoder, kv_norm_buffer, 1);
                set_buffer(encoder, kv_a_buffer, 2);
                set_bytes(encoder, u32_as_bytes(&latent_rank), 3);
                set_bytes(
                    encoder,
                    f32_as_bytes(std::slice::from_ref(&norm_epsilon)),
                    4,
                );
                dispatch_single_threadgroup(encoder, 256);

                encode_resident_projection(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    q_b,
                    q_a_buffer,
                    query_buffer,
                    0,
                )?;

                let heads =
                    u32::try_from(input.heads).context("GLM MLA fused heads do not fit u32")?;
                let nope_dim = u32::try_from(input.nope_dim)
                    .context("GLM MLA fused no-PE dim does not fit u32")?;
                let rope_dim = u32::try_from(input.rope_dim)
                    .context("GLM MLA fused RoPE dim does not fit u32")?;
                let sequence =
                    u32::try_from(sequence).context("GLM MLA fused sequence does not fit u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.glm_mla_prepare_query_kv_pipeline,
                );
                set_buffer(encoder, query_buffer, 0);
                set_buffer(encoder, kv_a_buffer, 1);
                set_buffer(encoder, query_nope_buffer, 2);
                set_buffer(encoder, query_rope_buffer, 3);
                set_buffer(encoder, record_latents_buffer, 4);
                set_buffer(encoder, record_rotary_buffer, 5);
                set_buffer(encoder, rope_cos_buffer, 6);
                set_buffer(encoder, rope_sin_buffer, 7);
                set_bytes(encoder, u32_as_bytes(&heads), 8);
                set_bytes(encoder, u32_as_bytes(&nope_dim), 9);
                set_bytes(encoder, u32_as_bytes(&rope_dim), 10);
                set_bytes(encoder, u32_as_bytes(&latent_rank), 11);
                set_bytes(encoder, u32_as_bytes(&sequence), 12);
                dispatch_threads(encoder, prepare_count as u64);

                encode_q4_mmap_multilinear(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    embed_q,
                    query_nope_buffer,
                    absorbed_queries_buffer,
                    input.latent_rank,
                )?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.glm_mla_absorbed_scores_pipeline,
                );
                set_buffer(encoder, absorbed_queries_buffer, 0);
                set_buffer(encoder, query_rope_buffer, 1);
                set_buffer(encoder, record_latents_buffer, 2);
                set_buffer(encoder, record_rotary_buffer, 3);
                set_buffer(encoder, scores_buffer, 4);
                set_bytes(encoder, u32_as_bytes(&heads), 5);
                set_bytes(encoder, u32_as_bytes(&latent_rank), 6);
                set_bytes(encoder, u32_as_bytes(&rope_dim), 7);
                set_bytes(encoder, u32_as_bytes(&sequence), 8);
                set_bytes(encoder, f32_as_bytes(std::slice::from_ref(&input.scale)), 9);
                dispatch_threads(encoder, score_count as u64);

                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.glm_mla_softmax_pipeline,
                );
                set_buffer(encoder, scores_buffer, 0);
                set_bytes(encoder, u32_as_bytes(&heads), 1);
                set_bytes(encoder, u32_as_bytes(&sequence), 2);
                dispatch_threads(encoder, input.heads as u64);

                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.glm_mla_context_pipeline,
                );
                set_buffer(encoder, scores_buffer, 0);
                set_buffer(encoder, record_latents_buffer, 1);
                set_buffer(encoder, contexts_buffer, 2);
                set_bytes(encoder, u32_as_bytes(&heads), 3);
                set_bytes(encoder, u32_as_bytes(&latent_rank), 4);
                set_bytes(encoder, u32_as_bytes(&sequence), 5);
                dispatch_threads(encoder, absorbed_width as u64);

                encode_q4_mmap_multilinear(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    unembed_out,
                    contexts_buffer,
                    output_buffer,
                    output_rows_per_head,
                )?;
                if let (Some(post), Some(plan), Some(post_buffers)) =
                    (input.post_attention, post_plan, post_buffers)
                {
                    let (
                        residual_input_buffer,
                        norm_weight_buffer,
                        projected_buffer,
                        residual_buffer,
                        normed_buffer,
                        router_logits_buffer,
                    ) = post_buffers;
                    encode_resident_projection(
                        &self.pipelines,
                        encoder,
                        dense_weights,
                        &post.projections.out_proj,
                        output_buffer,
                        projected_buffer,
                        0,
                    )?;
                    let width = u32::try_from(plan.width)
                        .context("GLM MLA fused post-attention width does not fit u32")?;
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
                    set_bytes(encoder, u32_as_bytes(&width), 5);
                    set_bytes(
                        encoder,
                        f32_as_bytes(std::slice::from_ref(&norm_epsilon)),
                        6,
                    );
                    dispatch_single_threadgroup(encoder, 256);
                    encode_resident_projection(
                        &self.pipelines,
                        encoder,
                        dense_weights,
                        &post.projections.router,
                        normed_buffer,
                        router_logits_buffer,
                        0,
                    )?;
                }
                Ok(())
            })();
            if let Err(error) = encode_result {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, true);
                return Err(error);
            }
            encoding.end_encoding();

            let context = MetalCommandContext::new("glm_mla_fused_attention")
                .with("input_len", input_len)
                .with("heads", input.heads)
                .with("latent_rank", input.latent_rank)
                .with("rope_dim", input.rope_dim)
                .with("sequence", sequence);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, error.should_release_buffers());
                return Err(error.into());
            }
            let latent = self.buffers.read_f32_buffer(kv_a_buffer, input.latent_rank);
            let rotary = self.buffers.read_f32_buffer_offset(
                record_rotary_buffer,
                previous_records * input.rope_dim,
                input.rope_dim,
            );
            let terminal = (|| -> Result<MetalGlmMlaFusedAttentionTerminal> {
                if let (Some(post), Some(plan), Some(post_buffers)) =
                    (input.post_attention, post_plan, post_buffers)
                {
                    let (_, _, _, residual_buffer, normed_buffer, router_logits_buffer) =
                        post_buffers;
                    let router_scores = self
                        .buffers
                        .read_f32_buffer(router_logits_buffer, plan.experts);
                    let active = if let Some(correction_bias) = post.router_correction_bias {
                        routing_sigmoid_noaux_top_k(
                            &router_scores,
                            correction_bias,
                            plan.active_count,
                        )?
                    } else {
                        routing_softmax_top_k(&router_scores, plan.active_count)
                    };
                    Ok(MetalGlmMlaFusedAttentionTerminal::PostAttention(
                        MetalPostAttentionPrep::new(
                            plan.layer,
                            plan.width,
                            plan.experts,
                            active,
                            residual_buffer,
                            normed_buffer,
                        )?,
                    ))
                } else {
                    Ok(MetalGlmMlaFusedAttentionTerminal::Attention(
                        self.buffers
                            .read_f32_buffer(output_buffer, unembed_out.rows),
                    ))
                }
            })();
            drop(encoding);
            let terminal = match terminal {
                Ok(terminal) => terminal,
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, false);
                    return Err(error);
                }
            };
            let retained = match &terminal {
                MetalGlmMlaFusedAttentionTerminal::Attention(_) => None,
                MetalGlmMlaFusedAttentionTerminal::PostAttention(prep) => {
                    Some((prep.residual_buffer, prep.normed_buffer))
                }
            };
            for buffer in buffers {
                if retained.is_none_or(|retained| buffer != retained.0 && buffer != retained.1) {
                    self.recycle(buffer);
                }
            }
            Ok(Some(MetalGlmMlaFusedAttentionOutput {
                terminal,
                latent,
                rotary,
            }))
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn execute_q4_multilinear(
        &self,
        projection: &DenseQ4MmapMatvecProjection,
        heads: usize,
        rows_per_head: usize,
        inputs: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        let Some(dense_weights) = &self.dense_weights else {
            return Ok(None);
        };
        let rows = heads
            .checked_mul(rows_per_head)
            .context("resident Q4 multilinear row count overflow")?;
        let input_values = heads
            .checked_mul(projection.cols)
            .context("resident Q4 multilinear input length overflow")?;
        if heads == 0
            || rows_per_head == 0
            || !rows_per_head.is_multiple_of(16)
            || projection.rows != rows
            || projection.output_width != rows
            || inputs.len() != input_values
            || !projection
                .scale_bias_dtype
                .eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16)
        {
            bail!(
                "resident Q4 multilinear projection {} has incompatible shape rows={} cols={} output_width={} heads={heads} rows_per_head={rows_per_head} input_len={} scale_bias_dtype={}",
                projection.tensor_name,
                projection.rows,
                projection.cols,
                projection.output_width,
                inputs.len(),
                projection.scale_bias_dtype,
            );
        }
        validate_resident_projection(projection, projection.cols, dense_weights.len)?;

        unsafe {
            let input_buffer = self.buffer_with_bytes(f32_as_bytes(inputs))?;
            let output_buffer = self.buffer_with_len(rows * std::mem::size_of::<f32>())?;
            let buffers = [input_buffer, output_buffer];
            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE resident Q4 multilinear command buffer",
                "failed to create Flash-MoE resident Q4 multilinear compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            encode_q4_mmap_multilinear(
                &self.pipelines,
                encoder,
                dense_weights,
                projection,
                input_buffer,
                output_buffer,
                rows_per_head,
            )?;
            encoding.end_encoding();

            let context = MetalCommandContext::new("dense_q4_mmap_multilinear")
                .with("tensor", projection.tensor_name.as_str())
                .with("heads", heads)
                .with("rows_per_head", rows_per_head)
                .with("cols", projection.cols);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, error.should_release_buffers());
                return Err(error.into());
            }
            let output = self.buffers.read_f32_buffer(output_buffer, rows);
            drop(encoding);
            self.recycle(input_buffer);
            self.recycle(output_buffer);
            Ok(Some(output))
        }
    }

    pub(crate) fn execute_glm_mla_absorbed_attention(
        &self,
        embed_q: &DenseQ4MmapMatvecProjection,
        unembed_out: &DenseQ4MmapMatvecProjection,
        input: MetalGlmMlaAbsorbedAttentionInput<'_>,
    ) -> Result<Option<Vec<f32>>> {
        let Some(dense_weights) = &self.dense_weights else {
            return Ok(None);
        };
        let absorbed_width = input
            .heads
            .checked_mul(input.latent_rank)
            .context("GLM MLA absorbed-query width overflow")?;
        let query_nope_width = input
            .heads
            .checked_mul(embed_q.cols)
            .context("GLM MLA no-PE query width overflow")?;
        let query_rope_width = input
            .heads
            .checked_mul(input.rope_dim)
            .context("GLM MLA rotary-query width overflow")?;
        let score_count = input
            .heads
            .checked_mul(input.sequence)
            .context("GLM MLA score count overflow")?;
        let record_latent_width = input
            .sequence
            .checked_mul(input.latent_rank)
            .context("GLM MLA latent-record width overflow")?;
        let record_rotary_width = input
            .sequence
            .checked_mul(input.rope_dim)
            .context("GLM MLA rotary-record width overflow")?;
        let output_rows_per_head = unembed_out
            .rows
            .checked_div(input.heads.max(1))
            .context("GLM MLA output rows-per-head division failed")?;
        let scale_bias_is_bf16 = |projection: &DenseQ4MmapMatvecProjection| {
            projection
                .scale_bias_dtype
                .eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16)
        };
        if input.heads == 0
            || input.latent_rank == 0
            || input.rope_dim == 0
            || input.sequence == 0
            || !input.scale.is_finite()
            || embed_q.rows != absorbed_width
            || embed_q.output_width != absorbed_width
            || embed_q.cols == 0
            || !input.latent_rank.is_multiple_of(16)
            || !scale_bias_is_bf16(embed_q)
            || unembed_out.rows == 0
            || unembed_out.rows % input.heads != 0
            || unembed_out.output_width != unembed_out.rows
            || unembed_out.cols != input.latent_rank
            || !output_rows_per_head.is_multiple_of(16)
            || !scale_bias_is_bf16(unembed_out)
            || input.query_nope.len() != query_nope_width
            || input.query_rope.len() != query_rope_width
            || input.record_latents.len() != record_latent_width
            || input.record_rotary.len() != record_rotary_width
        {
            bail!(
                "GLM MLA Metal absorbed attention has incompatible shapes embed={}x{} output={} unembed={}x{} output={} heads={} latent_rank={} rope_dim={} sequence={} query_nope={} query_rope={} record_latents={} record_rotary={} scale={}",
                embed_q.rows,
                embed_q.cols,
                embed_q.output_width,
                unembed_out.rows,
                unembed_out.cols,
                unembed_out.output_width,
                input.heads,
                input.latent_rank,
                input.rope_dim,
                input.sequence,
                input.query_nope.len(),
                input.query_rope.len(),
                input.record_latents.len(),
                input.record_rotary.len(),
                input.scale,
            );
        }
        validate_resident_projection(embed_q, embed_q.cols, dense_weights.len)?;
        validate_resident_projection(unembed_out, input.latent_rank, dense_weights.len)?;

        unsafe {
            let mut buffers = Vec::with_capacity(8);
            let query_nope_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(input.query_nope),
                &mut buffers,
            )?;
            let query_rope_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(input.query_rope),
                &mut buffers,
            )?;
            let record_latents_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(input.record_latents),
                &mut buffers,
            )?;
            let record_rotary_buffer = self.buffers.tracked_buffer_with_bytes(
                self.runtime.device,
                f32_as_bytes(input.record_rotary),
                &mut buffers,
            )?;
            let absorbed_queries_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                absorbed_width * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let scores_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                score_count * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let contexts_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                absorbed_width * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let output_buffer = self.buffers.tracked_buffer_with_len(
                self.runtime.device,
                unembed_out.rows * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE GLM MLA absorbed-attention command buffer",
                "failed to create Flash-MoE GLM MLA absorbed-attention compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            let encode_result = (|| -> Result<()> {
                encode_q4_mmap_multilinear(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    embed_q,
                    query_nope_buffer,
                    absorbed_queries_buffer,
                    input.latent_rank,
                )?;

                let heads = u32::try_from(input.heads).context("GLM MLA heads do not fit u32")?;
                let latent_rank = u32::try_from(input.latent_rank)
                    .context("GLM MLA latent rank does not fit u32")?;
                let rope_dim =
                    u32::try_from(input.rope_dim).context("GLM MLA rope dim does not fit u32")?;
                let sequence =
                    u32::try_from(input.sequence).context("GLM MLA sequence does not fit u32")?;

                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.glm_mla_absorbed_scores_pipeline,
                );
                set_buffer(encoder, absorbed_queries_buffer, 0);
                set_buffer(encoder, query_rope_buffer, 1);
                set_buffer(encoder, record_latents_buffer, 2);
                set_buffer(encoder, record_rotary_buffer, 3);
                set_buffer(encoder, scores_buffer, 4);
                set_bytes(encoder, u32_as_bytes(&heads), 5);
                set_bytes(encoder, u32_as_bytes(&latent_rank), 6);
                set_bytes(encoder, u32_as_bytes(&rope_dim), 7);
                set_bytes(encoder, u32_as_bytes(&sequence), 8);
                set_bytes(encoder, f32_as_bytes(std::slice::from_ref(&input.scale)), 9);
                dispatch_threads(encoder, score_count as u64);

                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.glm_mla_softmax_pipeline,
                );
                set_buffer(encoder, scores_buffer, 0);
                set_bytes(encoder, u32_as_bytes(&heads), 1);
                set_bytes(encoder, u32_as_bytes(&sequence), 2);
                dispatch_threads(encoder, input.heads as u64);

                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.glm_mla_context_pipeline,
                );
                set_buffer(encoder, scores_buffer, 0);
                set_buffer(encoder, record_latents_buffer, 1);
                set_buffer(encoder, contexts_buffer, 2);
                set_bytes(encoder, u32_as_bytes(&heads), 3);
                set_bytes(encoder, u32_as_bytes(&latent_rank), 4);
                set_bytes(encoder, u32_as_bytes(&sequence), 5);
                dispatch_threads(encoder, absorbed_width as u64);

                encode_q4_mmap_multilinear(
                    &self.pipelines,
                    encoder,
                    dense_weights,
                    unembed_out,
                    contexts_buffer,
                    output_buffer,
                    output_rows_per_head,
                )?;
                Ok(())
            })();
            if let Err(error) = encode_result {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, true);
                return Err(error);
            }
            encoding.end_encoding();

            let context = MetalCommandContext::new("glm_mla_absorbed_attention")
                .with("heads", input.heads)
                .with("latent_rank", input.latent_rank)
                .with("rope_dim", input.rope_dim)
                .with("sequence", input.sequence);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, error.should_release_buffers());
                return Err(error.into());
            }
            let output = self
                .buffers
                .read_f32_buffer(output_buffer, unembed_out.rows);
            drop(encoding);
            for buffer in buffers {
                self.recycle(buffer);
            }
            Ok(Some(output))
        }
    }

    pub(crate) unsafe fn try_encode_q4_mmap_projection_batch(
        &self,
        encoder: MetalObjcId,
        projections: &[&DenseQ4MmapMatvecProjection],
        input_buffer: MetalObjcId,
        input_rows: usize,
        output_buffer: MetalObjcId,
        output_offsets: &[usize],
        total_rows: usize,
        buffers: &mut Vec<MetalObjcId>,
    ) -> Result<bool> {
        unsafe {
            if projections.is_empty()
                || input_rows == 0
                || output_offsets.len() != projections.len()
            {
                return Ok(false);
            }
            let Some(dense_weights) = &self.dense_weights else {
                return Ok(false);
            };
            let first = &projections[0];
            if first.cols == 0 || first.cols > 4096 || first.group_size == 0 {
                return Ok(false);
            }
            let scale_bias_dtype = first.scale_bias_dtype.as_str();
            if !scale_bias_dtype.eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_F32)
                && !scale_bias_dtype.eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16)
            {
                return Ok(false);
            }
            if projections.iter().any(|projection| {
                projection.cols != first.cols
                    || projection.group_size != first.group_size
                    || projection.row_packed_bytes != first.row_packed_bytes
                    || !projection
                        .scale_bias_dtype
                        .eq_ignore_ascii_case(scale_bias_dtype)
            }) {
                return Ok(false);
            }

            let packed_offsets: Vec<u64> = projections
                .iter()
                .map(|projection| projection.packed_byte_offset)
                .collect();
            let scale_offsets: Vec<u64> = projections
                .iter()
                .map(|projection| projection.scales_byte_offset)
                .collect();
            let bias_offsets: Vec<u64> = projections
                .iter()
                .map(|projection| projection.biases_byte_offset)
                .collect();
            let row_offsets: Vec<u32> = output_offsets
                .iter()
                .map(|offset| u32::try_from(*offset))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("dense q4 mmap batch row offset does not fit u32")?;
            let rows: Vec<u32> = projections
                .iter()
                .map(|projection| u32::try_from(projection.rows))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("dense q4 mmap batch row count does not fit u32")?;
            let groups_per_rows: Vec<u32> = projections
                .iter()
                .map(|projection| u32::try_from(projection.groups_per_row))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("dense q4 mmap batch group count does not fit u32")?;
            let projection_count = u32::try_from(projections.len())
                .context("dense q4 mmap batch projection count does not fit u32")?;
            let cols = first.cols as u32;
            let group_size = first.group_size as u32;
            let input_rows_u32 = u32::try_from(input_rows)
                .context("dense q4 mmap batch input row count does not fit u32")?;
            let input_rows_per_threadgroup = if first.cols <= 2_048 && input_rows > 1 {
                2u32
            } else {
                1u32
            };

            let packed_offsets_buffer =
                self.buffer_with_bytes(u64_as_bytes_slice(&packed_offsets))?;
            let scale_offsets_buffer =
                self.buffer_with_bytes(u64_as_bytes_slice(&scale_offsets))?;
            let bias_offsets_buffer = self.buffer_with_bytes(u64_as_bytes_slice(&bias_offsets))?;
            let row_offsets_buffer = self.buffer_with_bytes(u32_as_bytes_slice(&row_offsets))?;
            let rows_buffer = self.buffer_with_bytes(u32_as_bytes_slice(&rows))?;
            let groups_buffer = self.buffer_with_bytes(u32_as_bytes_slice(&groups_per_rows))?;
            buffers.extend_from_slice(&[
                packed_offsets_buffer,
                scale_offsets_buffer,
                bias_offsets_buffer,
                row_offsets_buffer,
                rows_buffer,
                groups_buffer,
            ]);

            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                if scale_bias_dtype.eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16) {
                    self.pipelines.q4_mmap_batch_bf16_scale_bias_pipeline
                } else {
                    self.pipelines.q4_mmap_batch_pipeline
                },
            );
            set_buffer(encoder, dense_weights.buffer(), 0);
            set_buffer(encoder, input_buffer, 1);
            set_buffer(encoder, output_buffer, 2);
            set_buffer(encoder, packed_offsets_buffer, 3);
            set_buffer(encoder, scale_offsets_buffer, 4);
            set_buffer(encoder, bias_offsets_buffer, 5);
            set_buffer(encoder, row_offsets_buffer, 6);
            set_buffer(encoder, rows_buffer, 7);
            set_buffer(encoder, groups_buffer, 8);
            set_bytes(encoder, u32_as_bytes(&projection_count), 9);
            set_bytes(encoder, u32_as_bytes(&cols), 10);
            set_bytes(encoder, u32_as_bytes(&group_size), 11);
            if scale_bias_dtype.eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16) {
                set_bytes(encoder, u32_as_bytes(&input_rows_u32), 12);
                set_bytes(encoder, u32_as_bytes(&input_rows_per_threadgroup), 13);
                dispatch_q4_mmap_matrix_bf16_threadgroups(
                    encoder,
                    total_rows as u64,
                    input_rows as u64,
                    input_rows_per_threadgroup as u64,
                );
            } else {
                dispatch_q4_mmap_matrix_threadgroups(encoder, total_rows as u64, input_rows as u64);
            }
            Ok(true)
        }
    }

    pub(crate) fn execute(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input: &[f32],
    ) -> Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        if projections.is_empty() {
            return Ok(Some((Vec::new(), MetalMatvecTiming::default(), 0)));
        }
        let Some(dense_weights) = &self.dense_weights else {
            return Ok(None);
        };

        let mut total_rows = 0usize;
        let mut output_offsets = Vec::with_capacity(projections.len());
        for projection in projections {
            validate_resident_projection(projection, input.len(), dense_weights.len)?;
            let output_offset = total_rows;
            total_rows = total_rows
                .checked_add(projection.rows())
                .context("resident mmap batch output row count overflow")?;
            output_offsets.push(output_offset);
        }

        unsafe {
            let mut timing = MetalMatvecTiming::default();
            let upload_started = Instant::now();
            let input_buffer = self.buffer_with_bytes(f32_as_bytes(input))?;
            let output_buffer = self.buffer_with_len(total_rows * std::mem::size_of::<f32>())?;
            let mut buffers = vec![input_buffer, output_buffer];
            timing.buffer_upload += upload_started.elapsed();

            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE dense q4 mmap batch Metal command buffer",
                "failed to create Flash-MoE dense q4 mmap batch Metal compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();

            let q4_projections = projections
                .iter()
                .map(ResidentMmapMatvecProjection::q4)
                .collect::<Option<Vec<_>>>();
            let encode_result = (|| -> Result<usize> {
                if let Some(q4_projections) = q4_projections
                    && self.try_encode_q4_mmap_projection_batch(
                        encoder,
                        &q4_projections,
                        input_buffer,
                        1,
                        output_buffer,
                        &output_offsets,
                        total_rows,
                        &mut buffers,
                    )?
                {
                    return Ok(1);
                }
                for (idx, projection) in projections.iter().enumerate() {
                    let output_offset = output_offsets[idx]
                        .checked_mul(std::mem::size_of::<f32>())
                        .context("dense q4 mmap batch output byte offset overflow")?
                        as u64;
                    encode_resident_projection(
                        &self.pipelines,
                        encoder,
                        dense_weights,
                        projection,
                        input_buffer,
                        output_buffer,
                        output_offset,
                    )?;
                }
                Ok(projections.len())
            })();
            let dispatch_count = match encode_result {
                Ok(dispatch_count) => dispatch_count,
                Err(error) => {
                    drop(encoding);
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            encoding.end_encoding();

            let dispatch_started = Instant::now();
            let names = projections
                .iter()
                .map(ResidentMmapMatvecProjection::tensor_name)
                .collect::<Vec<_>>()
                .join(",");
            let context = MetalCommandContext::new("dense_q4_mmap_matvec_batch")
                .with("projections", projections.len())
                .with("dispatches", dispatch_count)
                .with("rows", total_rows)
                .with("input_len", input.len())
                .with("tensors", names);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, error.should_release_buffers());
                return Err(error.into());
            }
            timing.dispatch += dispatch_started.elapsed();

            let readback_started = Instant::now();
            let packed_output = self.buffers.read_f32_buffer(output_buffer, total_rows);
            timing.readback += readback_started.elapsed();

            let mut outputs = Vec::with_capacity(projections.len());
            for (projection, output_offset) in projections.iter().zip(output_offsets.iter()) {
                let start = *output_offset;
                let end = start + projection.rows();
                let mut output = vec![0.0f32; projection.output_width()];
                output[..projection.rows()].copy_from_slice(&packed_output[start..end]);
                outputs.push(output);
            }

            drop(encoding);
            for buffer in buffers {
                self.recycle(buffer);
            }
            Ok(Some((outputs, timing, dispatch_count)))
        }
    }

    #[cfg(test)]
    pub(crate) fn execute_matrix(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input_rows: usize,
        input_cols: usize,
        input: &[f32],
    ) -> Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        self.execute_matrix_input(
            projections,
            input_rows,
            input_cols,
            MetalBatchProjectionInput::Cpu(input),
        )
    }

    #[cfg(test)]
    fn execute_matrix_input(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input_rows: usize,
        input_cols: usize,
        input: MetalBatchProjectionInput<'_>,
    ) -> Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        if projections.is_empty() {
            return Ok(Some((Vec::new(), MetalMatvecTiming::default(), 0)));
        }
        if input_rows == 0 || input_cols == 0 {
            bail!(
                "FlashMoe resident projection matrix requires non-zero geometry, got {input_rows}x{input_cols}"
            );
        }
        let expected_input = input_rows
            .checked_mul(input_cols)
            .context("resident projection matrix input size overflow")?;
        if input.len() != expected_input {
            bail!(
                "FlashMoe resident projection matrix has {} values, expected {input_rows}x{input_cols}={expected_input}",
                input.len()
            );
        }
        let Some(dense_weights) = &self.dense_weights else {
            return Ok(None);
        };

        let mut total_rows = 0usize;
        let mut output_offsets = Vec::with_capacity(projections.len());
        for projection in projections {
            validate_resident_projection(projection, input_cols, dense_weights.len)?;
            let output_offset = total_rows;
            total_rows = total_rows
                .checked_add(projection.rows())
                .context("resident mmap matrix output row count overflow")?;
            output_offsets.push(output_offset);
        }
        let total_output_values = input_rows
            .checked_mul(total_rows)
            .context("resident mmap matrix output size overflow")?;
        let q4_projections = projections
            .iter()
            .map(ResidentMmapMatvecProjection::q4)
            .collect::<Option<Vec<_>>>()
            .context(
                "FlashMoe layer-major resident projection matrix requires affine-Q4 projections",
            )?;

        unsafe {
            let mut timing = MetalMatvecTiming::default();
            let upload_started = Instant::now();
            let mut buffers = Vec::with_capacity(10);
            let input_buffer = match input {
                MetalBatchProjectionInput::Cpu(values) => self.buffers.tracked_buffer_with_bytes(
                    self.device,
                    f32_as_bytes(values),
                    &mut buffers,
                )?,
                MetalBatchProjectionInput::Buffer { buffer, .. } => buffer,
            };
            let output_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                total_output_values * std::mem::size_of::<f32>(),
                &mut buffers,
            )?;
            timing.buffer_upload += upload_started.elapsed();

            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE dense q4 mmap matrix Metal command buffer",
                "failed to create Flash-MoE dense q4 mmap matrix Metal compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            let encoded = match self.try_encode_q4_mmap_projection_batch(
                encoder,
                &q4_projections,
                input_buffer,
                input_rows,
                output_buffer,
                &output_offsets,
                total_rows,
                &mut buffers,
            ) {
                Ok(encoded) => encoded,
                Err(error) => {
                    drop(encoding);
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            if !encoded {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, true);
                bail!(
                    "FlashMoe layer-major resident projection matrix did not resolve a compatible affine-Q4 command"
                );
            }
            encoding.end_encoding();

            let dispatch_started = Instant::now();
            let names = projections
                .iter()
                .map(ResidentMmapMatvecProjection::tensor_name)
                .collect::<Vec<_>>()
                .join(",");
            let context = MetalCommandContext::new("dense_q4_mmap_projection_matrix")
                .with("projections", projections.len())
                .with("input_rows", input_rows)
                .with("input_cols", input_cols)
                .with("output_rows", total_rows)
                .with("tensors", names);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, error.should_release_buffers());
                return Err(error.into());
            }
            timing.dispatch += dispatch_started.elapsed();

            let readback_started = Instant::now();
            let packed_output = self
                .buffers
                .read_f32_buffer(output_buffer, total_output_values);
            timing.readback += readback_started.elapsed();
            let mut outputs = Vec::with_capacity(projections.len());
            for (projection, output_offset) in projections.iter().zip(output_offsets.iter()) {
                let output_width = projection.output_width();
                let mut output = vec![0.0f32; input_rows * output_width];
                for input_row in 0..input_rows {
                    let source_start = input_row * total_rows + *output_offset;
                    let source_end = source_start + projection.rows();
                    let target_start = input_row * output_width;
                    let target_end = target_start + projection.rows();
                    output[target_start..target_end]
                        .copy_from_slice(&packed_output[source_start..source_end]);
                }
                outputs.push(output);
            }

            drop(encoding);
            for buffer in buffers {
                self.recycle(buffer);
            }
            Ok(Some((outputs, timing, 1)))
        }
    }

    #[cfg(test)]
    pub(crate) fn execute_rms_norm_rows(
        &self,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>> {
        let expected_values = rows
            .checked_mul(width)
            .context("Qwen RMS-normalization matrix size overflow")?;
        if rows == 0
            || width == 0
            || width > u32::MAX as usize
            || input.len() != expected_values
            || weight.len() != width
        {
            bail!(
                "Qwen RMS-normalization matrix has incompatible geometry: rows={rows} width={width} input={} weight={}",
                input.len(),
                weight.len()
            );
        }
        unsafe {
            let input_buffer = self.buffer_with_bytes(f32_as_bytes(input))?;
            let weight_buffer = self.buffer_with_bytes(f32_as_bytes(weight))?;
            let output_buffer =
                self.buffer_with_len(expected_values * std::mem::size_of::<f32>())?;
            let buffers = vec![input_buffer, weight_buffer, output_buffer];
            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Qwen RMS-normalization matrix command buffer",
                "failed to create Qwen RMS-normalization matrix encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            let width_u32 = width as u32;
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.pipelines.rms_norm_reduced_pipeline,
            );
            set_buffer(encoder, input_buffer, 0);
            set_buffer(encoder, weight_buffer, 1);
            set_buffer(encoder, output_buffer, 2);
            set_bytes(encoder, u32_as_bytes(&width_u32), 3);
            set_bytes(encoder, f32_as_bytes(std::slice::from_ref(&epsilon)), 4);
            msg_send_void2_size(
                encoder,
                sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                MetalDispatchSize::new(1, rows as u64, 1),
                MetalDispatchSize::new(256, 1, 1),
            );
            encoding.end_encoding();
            let context = MetalCommandContext::new("qwen_rms_norm_rows")
                .with("rows", rows)
                .with("width", width);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, error.should_release_buffers());
                return Err(error.into());
            }
            let output = self.buffers.read_f32_buffer(output_buffer, expected_values);
            drop(encoding);
            for buffer in buffers {
                self.recycle(buffer);
            }
            Ok(output)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_post_attention_matrix(
        &self,
        out_proj: &ResidentMmapMatvecProjection,
        router: &ResidentMmapMatvecProjection,
        rows: usize,
        attention_width: usize,
        width: usize,
        attention: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        norm_epsilon: f32,
    ) -> Result<Option<MetalLayerMajorPostAttention>> {
        let Some(dense_weights) = &self.dense_weights else {
            return Ok(None);
        };
        let attention_values = rows
            .checked_mul(attention_width)
            .context("layer-major attention matrix size overflow")?;
        let hidden_values = rows
            .checked_mul(width)
            .context("layer-major hidden matrix size overflow")?;
        if rows == 0
            || attention_width == 0
            || width == 0
            || attention.len() != attention_values
            || residual.len() != hidden_values
            || post_norm_weight.len() != width
            || !norm_epsilon.is_finite()
            || norm_epsilon <= 0.0
        {
            bail!(
                "Qwen layer-major post-attention matrix has incompatible geometry rows={rows} attention_width={attention_width} width={width} attention={} residual={} norm={} epsilon={norm_epsilon}",
                attention.len(),
                residual.len(),
                post_norm_weight.len()
            );
        }
        validate_resident_projection(out_proj, attention_width, dense_weights.len)?;
        validate_resident_projection(router, width, dense_weights.len)?;
        if out_proj.output_width() != width || router.output_width() != router.rows() {
            bail!(
                "Qwen layer-major post-attention projection shapes are incompatible: out={}x{} router={}x{} width={width}",
                out_proj.rows(),
                out_proj.cols(),
                router.rows(),
                router.cols()
            );
        }
        unsafe {
            let (attention_buffer, owned_attention_input) = match attention {
                MetalBatchProjectionInput::Cpu(values) => {
                    (self.buffer_with_bytes(f32_as_bytes(values))?, true)
                }
                MetalBatchProjectionInput::Buffer { buffer, .. } => (buffer, false),
            };
            let (residual_input_buffer, owned_residual_input) = match residual {
                MetalBatchProjectionInput::Cpu(values) => {
                    (self.buffer_with_bytes(f32_as_bytes(values))?, true)
                }
                MetalBatchProjectionInput::Buffer { buffer, .. } => (buffer, false),
            };
            let norm_weight_buffer = self.buffer_with_bytes(f32_as_bytes(post_norm_weight))?;
            let projected_buffer =
                self.buffer_with_len(hidden_values * std::mem::size_of::<f32>())?;
            let hidden_buffer = self.buffer_with_len(hidden_values * std::mem::size_of::<f32>())?;
            let normed_buffer = self.buffer_with_len(hidden_values * std::mem::size_of::<f32>())?;
            let router_values = rows
                .checked_mul(router.rows())
                .context("layer-major router matrix size overflow")?;
            let router_buffer = self.buffer_with_len(router_values * std::mem::size_of::<f32>())?;
            let mut buffers = vec![
                norm_weight_buffer,
                projected_buffer,
                hidden_buffer,
                normed_buffer,
                router_buffer,
            ];
            if owned_attention_input {
                buffers.push(attention_buffer);
            }
            if owned_residual_input {
                buffers.push(residual_input_buffer);
            }
            let scalar_reference = if rows == 1 {
                let projected = self.buffer_with_len(hidden_values * std::mem::size_of::<f32>())?;
                let hidden = self.buffer_with_len(hidden_values * std::mem::size_of::<f32>())?;
                let normed = self.buffer_with_len(hidden_values * std::mem::size_of::<f32>())?;
                let router = self.buffer_with_len(router_values * std::mem::size_of::<f32>())?;
                buffers.extend_from_slice(&[projected, hidden, normed, router]);
                Some((projected, hidden, normed, router))
            } else {
                None
            };
            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Qwen layer-major post-attention command buffer",
                "failed to create Qwen layer-major post-attention encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            let encode_result = (|| -> Result<()> {
                match out_proj {
                    ResidentMmapMatvecProjection::Q4(out_q4) => {
                        if !self.try_encode_q4_mmap_projection_batch(
                            encoder,
                            &[out_q4],
                            attention_buffer,
                            rows,
                            projected_buffer,
                            &[0],
                            width,
                            &mut buffers,
                        )? {
                            bail!(
                                "Qwen layer-major output projection did not resolve a matrix command"
                            );
                        }
                    }
                    ResidentMmapMatvecProjection::Dense(out_dense) => {
                        encode_dense_resident_matrix(
                            &self.pipelines,
                            encoder,
                            dense_weights,
                            out_dense,
                            attention_buffer,
                            projected_buffer,
                            rows,
                        )?;
                    }
                }
                let width_u32 =
                    u32::try_from(width).context("Qwen layer-major hidden width exceeds u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.pipelines.residual_rms_norm_pipeline,
                );
                set_buffer(encoder, projected_buffer, 0);
                set_buffer(encoder, residual_input_buffer, 1);
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
                        if !self.try_encode_q4_mmap_projection_batch(
                            encoder,
                            &[router_q4],
                            normed_buffer,
                            rows,
                            router_buffer,
                            &[0],
                            router.rows(),
                            &mut buffers,
                        )? {
                            bail!(
                                "Qwen layer-major router projection did not resolve a matrix command"
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
                        )?;
                    }
                }
                if let Some((scalar_projected, scalar_hidden, scalar_normed, scalar_router)) =
                    scalar_reference
                {
                    encode_resident_projection(
                        &self.pipelines,
                        encoder,
                        dense_weights,
                        out_proj,
                        attention_buffer,
                        scalar_projected,
                        0,
                    )?;
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.pipelines.residual_rms_norm_pipeline,
                    );
                    set_buffer(encoder, scalar_projected, 0);
                    set_buffer(encoder, residual_input_buffer, 1);
                    set_buffer(encoder, norm_weight_buffer, 2);
                    set_buffer(encoder, scalar_hidden, 3);
                    set_buffer(encoder, scalar_normed, 4);
                    set_bytes(encoder, u32_as_bytes(&width_u32), 5);
                    set_bytes(
                        encoder,
                        f32_as_bytes(std::slice::from_ref(&norm_epsilon)),
                        6,
                    );
                    dispatch_single_threadgroup(encoder, 256);
                    encode_resident_projection(
                        &self.pipelines,
                        encoder,
                        dense_weights,
                        router,
                        scalar_normed,
                        scalar_router,
                        0,
                    )?;
                }
                Ok(())
            })();
            if let Err(error) = encode_result {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, true);
                return Err(error);
            }
            encoding.end_encoding();
            let context = MetalCommandContext::new("qwen_layer_major_post_attention")
                .with("rows", rows)
                .with("attention_width", attention_width)
                .with("width", width)
                .with("experts", router.rows());
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, error.should_release_buffers());
                return Err(error.into());
            }
            if let Some((scalar_projected, scalar_hidden, scalar_normed, scalar_router)) =
                scalar_reference
            {
                for (phase, actual_buffer, expected_buffer, values) in [
                    (
                        "output_projection",
                        projected_buffer,
                        scalar_projected,
                        hidden_values,
                    ),
                    ("residual", hidden_buffer, scalar_hidden, hidden_values),
                    ("post_norm", normed_buffer, scalar_normed, hidden_values),
                    ("router", router_buffer, scalar_router, router_values),
                ] {
                    let actual = self.buffers.read_f32_buffer(actual_buffer, values);
                    let expected = self.buffers.read_f32_buffer(expected_buffer, values);
                    if let Some((index, (actual, expected))) = actual
                        .iter()
                        .zip(expected.iter())
                        .enumerate()
                        .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
                    {
                        drop(encoding);
                        self.recycle_or_release_buffers(&buffers, false);
                        bail!(
                            "Qwen one-row post-attention parity failed phase={phase} index={index}: matrix={actual} scalar={expected} delta={}",
                            (actual - expected).abs()
                        );
                    }
                }
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
            match output {
                Ok(output) => {
                    for buffer in buffers {
                        if buffer != hidden_buffer && buffer != normed_buffer {
                            self.recycle(buffer);
                        }
                    }
                    Ok(Some(output))
                }
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, false);
                    Err(error)
                }
            }
        }
    }

    pub(crate) fn execute_with_input_buffer(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input_buffer: MetalObjcId,
        input_len: usize,
    ) -> Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        if projections.is_empty() {
            return Ok(Some((Vec::new(), MetalMatvecTiming::default(), 0)));
        }
        let Some(dense_weights) = &self.dense_weights else {
            return Ok(None);
        };

        let mut total_rows = 0usize;
        let mut output_offsets = Vec::with_capacity(projections.len());
        for projection in projections {
            validate_resident_projection(projection, input_len, dense_weights.len)?;
            let output_offset = total_rows;
            total_rows = total_rows
                .checked_add(projection.rows())
                .context("resident mmap batch output row count overflow")?;
            output_offsets.push(output_offset);
        }

        unsafe {
            let mut timing = MetalMatvecTiming::default();
            let upload_started = Instant::now();
            let output_buffer = self.buffer_with_len(total_rows * std::mem::size_of::<f32>())?;
            let mut buffers = vec![output_buffer];
            timing.buffer_upload += upload_started.elapsed();

            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE dense q4 mmap batch Metal command buffer",
                "failed to create Flash-MoE dense q4 mmap batch Metal compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();

            let q4_projections = projections
                .iter()
                .map(ResidentMmapMatvecProjection::q4)
                .collect::<Option<Vec<_>>>();
            let encode_result = (|| -> Result<usize> {
                if let Some(q4_projections) = q4_projections
                    && self.try_encode_q4_mmap_projection_batch(
                        encoder,
                        &q4_projections,
                        input_buffer,
                        1,
                        output_buffer,
                        &output_offsets,
                        total_rows,
                        &mut buffers,
                    )?
                {
                    return Ok(1);
                }
                for (idx, projection) in projections.iter().enumerate() {
                    let output_offset = output_offsets[idx]
                        .checked_mul(std::mem::size_of::<f32>())
                        .context("dense q4 mmap batch output byte offset overflow")?
                        as u64;
                    encode_resident_projection(
                        &self.pipelines,
                        encoder,
                        dense_weights,
                        projection,
                        input_buffer,
                        output_buffer,
                        output_offset,
                    )?;
                }
                Ok(projections.len())
            })();
            let dispatch_count = match encode_result {
                Ok(dispatch_count) => dispatch_count,
                Err(error) => {
                    drop(encoding);
                    self.recycle_or_release_buffers(&buffers, true);
                    return Err(error);
                }
            };
            encoding.end_encoding();

            let dispatch_started = Instant::now();
            let names = projections
                .iter()
                .map(ResidentMmapMatvecProjection::tensor_name)
                .collect::<Vec<_>>()
                .join(",");
            let context = MetalCommandContext::new("dense_q4_mmap_matvec_batch_deferred_input")
                .with("projections", projections.len())
                .with("dispatches", dispatch_count)
                .with("rows", total_rows)
                .with("input_len", input_len)
                .with("tensors", names);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, error.should_release_buffers());
                return Err(error.into());
            }
            timing.dispatch += dispatch_started.elapsed();

            let readback_started = Instant::now();
            let packed_output = self.buffers.read_f32_buffer(output_buffer, total_rows);
            timing.readback += readback_started.elapsed();

            let mut outputs = Vec::with_capacity(projections.len());
            for (projection, output_offset) in projections.iter().zip(output_offsets.iter()) {
                let start = *output_offset;
                let end = start + projection.rows();
                let mut output = vec![0.0f32; projection.output_width()];
                output[..projection.rows()].copy_from_slice(&packed_output[start..end]);
                outputs.push(output);
            }

            drop(encoding);
            for buffer in buffers {
                self.recycle(buffer);
            }
            Ok(Some((outputs, timing, dispatch_count)))
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl std::ops::Deref for MetalResidentProjectionBatchBuilder<'_> {
    type Target = MetalRuntime;

    fn deref(&self) -> &Self::Target {
        self.runtime
    }
}
