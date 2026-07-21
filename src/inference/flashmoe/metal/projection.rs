use super::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) trait ResidentProjectionValidation {
    fn tensor_name(&self) -> &str;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn output_width(&self) -> usize;
    fn validate_storage(&self, dense_len: usize) -> Result<()>;
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn validate_resident_projection<TProjection>(
    projection: &TProjection,
    input_len: usize,
    dense_len: usize,
) -> Result<()>
where
    TProjection: ResidentProjectionValidation + ?Sized,
{
    if projection.rows() == 0 || projection.cols() == 0 {
        bail!(
            "resident projection {} has a zero-sized shape",
            projection.tensor_name()
        );
    }
    if projection.cols() != input_len {
        bail!(
            "resident projection {} input len {input_len} does not match cols {}",
            projection.tensor_name(),
            projection.cols()
        );
    }
    if projection.output_width() != projection.rows() {
        bail!(
            "resident projection {} output width {} does not match rows {}",
            projection.tensor_name(),
            projection.output_width(),
            projection.rows()
        );
    }
    projection.validate_storage(dense_len)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl ResidentProjectionValidation for DenseQ4MmapMatvecProjection {
    fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn output_width(&self) -> usize {
        self.output_width
    }

    fn validate_storage(&self, dense_len: usize) -> Result<()> {
        if self.row_packed_bytes != self.cols.div_ceil(2) {
            bail!(
                "resident Q4 projection {} row packed bytes {} do not match cols {}",
                self.tensor_name,
                self.row_packed_bytes,
                self.cols
            );
        }
        let scale_bias_bytes = expert_scale_bias_dtype_size(&self.scale_bias_dtype)?;
        if self.scales_byte_offset % scale_bias_bytes as u64 != 0
            || self.biases_byte_offset % scale_bias_bytes as u64 != 0
        {
            bail!(
                "resident Q4 projection {} has unaligned scale/bias offsets",
                self.tensor_name
            );
        }
        let packed_len = self
            .rows
            .checked_mul(self.row_packed_bytes)
            .context("resident Q4 projection packed byte length overflow")?;
        let group_bytes = self
            .rows
            .checked_mul(self.groups_per_row)
            .and_then(|groups| groups.checked_mul(scale_bias_bytes))
            .context("resident Q4 projection group byte length overflow")?;
        for (offset, len, label) in [
            (self.packed_byte_offset, packed_len, "packed"),
            (self.scales_byte_offset, group_bytes, "scales"),
            (self.biases_byte_offset, group_bytes, "biases"),
        ] {
            let offset = usize::try_from(offset).with_context(|| {
                format!("resident Q4 projection {label} offset does not fit usize")
            })?;
            if offset.checked_add(len).map_or(true, |end| end > dense_len) {
                bail!(
                    "resident Q4 projection {label} range for {} exceeds resident dense weights",
                    self.tensor_name
                );
            }
        }
        Ok(())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl ResidentProjectionValidation for DenseMmapMatvecProjection {
    fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn output_width(&self) -> usize {
        self.output_width
    }

    fn validate_storage(&self, dense_len: usize) -> Result<()> {
        let element_size = self.dtype.element_size();
        if self.byte_offset % element_size as u64 != 0 {
            bail!(
                "resident dense projection {} offset {} is unaligned for dtype {}",
                self.tensor_name,
                self.byte_offset,
                self.dtype.as_str()
            );
        }
        let byte_len = self
            .rows
            .checked_mul(self.cols)
            .and_then(|values| values.checked_mul(element_size))
            .context("resident dense projection byte length overflow")?;
        let offset = usize::try_from(self.byte_offset)
            .context("resident dense projection offset does not fit usize")?;
        if offset
            .checked_add(byte_len)
            .map_or(true, |end| end > dense_len)
        {
            bail!(
                "resident dense projection {} range exceeds resident dense weights",
                self.tensor_name
            );
        }
        Ok(())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl ResidentProjectionValidation for ResidentMmapMatvecProjection {
    fn tensor_name(&self) -> &str {
        self.tensor_name()
    }

    fn rows(&self) -> usize {
        self.rows()
    }

    fn cols(&self) -> usize {
        self.cols()
    }

    fn output_width(&self) -> usize {
        self.output_width()
    }

    fn validate_storage(&self, dense_len: usize) -> Result<()> {
        match self {
            Self::Q4(projection) => projection.validate_storage(dense_len),
            Self::Dense(projection) => projection.validate_storage(dense_len),
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn encode_resident_projection(
    pipelines: &MetalPipelineSet<MetalObjcId>,
    encoder: MetalObjcId,
    dense_weights: &MetalDenseWeights,
    projection: &ResidentMmapMatvecProjection,
    input_buffer: MetalObjcId,
    output_buffer: MetalObjcId,
    output_offset: u64,
) -> Result<()> {
    unsafe {
        encode_resident_projection_rows(
            pipelines,
            encoder,
            dense_weights,
            projection,
            projection.rows(),
            input_buffer,
            output_buffer,
            output_offset,
        )
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn encode_resident_projection_rows(
    pipelines: &MetalPipelineSet<MetalObjcId>,
    encoder: MetalObjcId,
    dense_weights: &MetalDenseWeights,
    projection: &ResidentMmapMatvecProjection,
    output_rows: usize,
    input_buffer: MetalObjcId,
    output_buffer: MetalObjcId,
    output_offset: u64,
) -> Result<()> {
    if output_rows == 0 || output_rows > projection.rows() {
        bail!(
            "resident projection {} requested {} output rows from {} physical rows",
            projection.tensor_name(),
            output_rows,
            projection.rows()
        );
    }
    unsafe {
        set_buffer(encoder, dense_weights.buffer(), 0);
        set_buffer(encoder, input_buffer, 1);
        set_buffer_with_offset(encoder, output_buffer, output_offset, 2);
        match projection {
            ResidentMmapMatvecProjection::Q4(projection) => {
                let rows = u32::try_from(output_rows).context("resident Q4 rows do not fit u32")?;
                let cols =
                    u32::try_from(projection.cols).context("resident Q4 cols do not fit u32")?;
                let groups = u32::try_from(projection.groups_per_row)
                    .context("resident Q4 groups do not fit u32")?;
                let group_size = u32::try_from(projection.group_size)
                    .context("resident Q4 group size does not fit u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    if projection
                        .scale_bias_dtype
                        .eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16)
                    {
                        pipelines.q4_mmap_bf16_scale_bias_pipeline
                    } else {
                        pipelines.q4_mmap_pipeline
                    },
                );
                set_bytes(encoder, u64_as_bytes(&projection.packed_byte_offset), 3);
                set_bytes(encoder, u64_as_bytes(&projection.scales_byte_offset), 4);
                set_bytes(encoder, u64_as_bytes(&projection.biases_byte_offset), 5);
                set_bytes(encoder, u32_as_bytes(&rows), 6);
                set_bytes(encoder, u32_as_bytes(&cols), 7);
                set_bytes(encoder, u32_as_bytes(&groups), 8);
                set_bytes(encoder, u32_as_bytes(&group_size), 9);
                dispatch_q4_mmap_threadgroups(encoder, output_rows as u64);
            }
            ResidentMmapMatvecProjection::Dense(projection) => {
                let pipeline = match projection.dtype {
                    ResidentStaticDtype::Bf16 => pipelines.dense_mmap_bf16_pipeline,
                    ResidentStaticDtype::F16 => pipelines.dense_mmap_f16_pipeline,
                    ResidentStaticDtype::F32 => pipelines.dense_mmap_f32_pipeline,
                };
                let rows =
                    u32::try_from(output_rows).context("resident dense rows do not fit u32")?;
                let cols =
                    u32::try_from(projection.cols).context("resident dense cols do not fit u32")?;
                msg_send_void1_id(encoder, sel("setComputePipelineState:"), pipeline);
                set_bytes(encoder, u64_as_bytes(&projection.byte_offset), 3);
                set_bytes(encoder, u32_as_bytes(&rows), 4);
                set_bytes(encoder, u32_as_bytes(&cols), 5);
                dispatch_threads(encoder, output_rows as u64);
            }
        }
        Ok(())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn encode_dense_resident_matrix(
    pipelines: &MetalPipelineSet<MetalObjcId>,
    encoder: MetalObjcId,
    dense_weights: &MetalDenseWeights,
    projection: &DenseMmapMatvecProjection,
    input_buffer: MetalObjcId,
    output_buffer: MetalObjcId,
    input_rows: usize,
) -> Result<()> {
    unsafe {
        let pipeline = match projection.dtype {
            ResidentStaticDtype::Bf16 => pipelines.dense_matrix_bf16_pipeline,
            ResidentStaticDtype::F16 => pipelines.dense_matrix_f16_pipeline,
            ResidentStaticDtype::F32 => pipelines.dense_matrix_f32_pipeline,
        };
        let rows =
            u32::try_from(projection.rows).context("resident dense matrix rows do not fit u32")?;
        let cols =
            u32::try_from(projection.cols).context("resident dense matrix cols do not fit u32")?;
        msg_send_void1_id(encoder, sel("setComputePipelineState:"), pipeline);
        set_buffer(encoder, dense_weights.buffer(), 0);
        set_buffer(encoder, input_buffer, 1);
        set_buffer(encoder, output_buffer, 2);
        set_bytes(encoder, u64_as_bytes(&projection.byte_offset), 3);
        set_bytes(encoder, u32_as_bytes(&rows), 4);
        set_bytes(encoder, u32_as_bytes(&cols), 5);
        msg_send_void2_size(
            encoder,
            sel("dispatchThreads:threadsPerThreadgroup:"),
            MetalDispatchSize::new(projection.rows as u64, input_rows as u64, 1),
            MetalDispatchSize::new(projection.rows.min(256) as u64, 1, 1),
        );
        Ok(())
    }
}
