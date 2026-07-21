use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FixedQ4ExpertEncoding {
    AffineBf16,
    MlxMxfp4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FixedQ4ExpertSlotSpec {
    pub(crate) layout: QwenMoeQ4ExpertLayout,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) encoding: FixedQ4ExpertEncoding,
}

impl FixedQ4ExpertSlotSpec {
    pub(crate) fn new(
        layout: QwenMoeQ4ExpertLayout,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        layout.validate()?;
        if hidden_size == 0 || intermediate_size == 0 {
            bail!(
                "fixed Q4 expert slot spec requires non-zero dimensions, hidden_size={hidden_size}, intermediate_size={intermediate_size}"
            );
        }
        Ok(Self {
            layout,
            hidden_size,
            intermediate_size,
            encoding: FixedQ4ExpertEncoding::AffineBf16,
        })
    }

    pub(crate) fn new_mxfp4(
        layout: QwenMoeQ4ExpertLayout,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        let mut spec = Self::new(layout, hidden_size, intermediate_size)?;
        spec.encoding = FixedQ4ExpertEncoding::MlxMxfp4;
        Ok(spec)
    }

    #[cfg(test)]
    pub(crate) fn qwen35_a17b() -> Result<Self> {
        Self::new(QwenMoeQ4ExpertLayout::qwen35_a17b(), HIDDEN_DIM, 1024)
    }

    pub(crate) fn from_model_layout(layout: &QwenMoeModelLayout) -> Result<Self> {
        let q4_layout = QwenMoeQ4ExpertLayout::fixed_bf16(
            layout.hidden_size,
            layout.moe_intermediate_size,
            GROUP_SIZE,
        )
        .with_context(|| {
            format!(
                "FlashMoe unsupported {:?} fixed-Q4 expert storage dimensions",
                layout.family
            )
        })?;
        Self::new(q4_layout, layout.hidden_size, layout.moe_intermediate_size)
    }

    pub(crate) fn mxfp4_from_model_layout(layout: &QwenMoeModelLayout) -> Result<Self> {
        let q4_layout = QwenMoeQ4ExpertLayout::fixed_mxfp4(
            layout.hidden_size,
            layout.moe_intermediate_size,
            32,
        )
        .with_context(|| {
            format!(
                "FlashMoe unsupported {:?} fixed-MXFP4 expert storage dimensions",
                layout.family
            )
        })?;
        Self::new_mxfp4(q4_layout, layout.hidden_size, layout.moe_intermediate_size)
    }

    pub(crate) const fn metadata_format(self) -> &'static str {
        match self.encoding {
            FixedQ4ExpertEncoding::AffineBf16 => FIXED_Q4_EXPERT_LAYER_FORMAT_V1,
            FixedQ4ExpertEncoding::MlxMxfp4 => FIXED_MXFP4_EXPERT_LAYER_FORMAT_V1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseExpertDtype {
    Bf16,
    F16,
}

impl DenseExpertDtype {
    pub(crate) fn from_metadata_dtype(dtype: &str) -> Option<Self> {
        match dtype.to_ascii_uppercase().as_str() {
            "BF16" | "BFLOAT16" => Some(Self::Bf16),
            "F16" | "FLOAT16" | "FP16" => Some(Self::F16),
            _ => None,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F16 => "F16",
        }
    }

    pub(crate) const fn element_size(self) -> usize {
        2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseExpertProjectionSpec {
    pub(crate) offset: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FixedDenseExpertSlotSpec {
    pub(crate) dtype: DenseExpertDtype,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) gate: DenseExpertProjectionSpec,
    pub(crate) up: DenseExpertProjectionSpec,
    pub(crate) down: DenseExpertProjectionSpec,
    pub(crate) expert_bytes: usize,
}

impl FixedDenseExpertSlotSpec {
    pub(crate) fn new(
        dtype: DenseExpertDtype,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        if hidden_size == 0 || intermediate_size == 0 {
            bail!(
                "fixed {} expert slot spec requires non-zero dimensions, hidden_size={hidden_size}, intermediate_size={intermediate_size}",
                dtype.as_str()
            );
        }
        let projection_bytes = |rows: usize, cols: usize| {
            rows.checked_mul(cols)
                .and_then(|values| values.checked_mul(dtype.element_size()))
                .context("fixed dense expert projection byte length overflow")
        };
        let aligned = |value: usize| {
            value
                .checked_add(EXPERT_COMPONENT_ALIGNMENT - 1)
                .map(|value| value / EXPERT_COMPONENT_ALIGNMENT * EXPERT_COMPONENT_ALIGNMENT)
                .context("fixed dense expert component alignment overflow")
        };
        let gate_bytes = projection_bytes(intermediate_size, hidden_size)?;
        let down_bytes = projection_bytes(hidden_size, intermediate_size)?;
        let gate = DenseExpertProjectionSpec {
            offset: 0,
            rows: intermediate_size,
            cols: hidden_size,
            bytes: gate_bytes,
        };
        let up = DenseExpertProjectionSpec {
            offset: aligned(
                gate.offset
                    .checked_add(gate.bytes)
                    .context("fixed dense expert gate component end overflow")?,
            )?,
            rows: intermediate_size,
            cols: hidden_size,
            bytes: gate_bytes,
        };
        let down = DenseExpertProjectionSpec {
            offset: aligned(
                up.offset
                    .checked_add(up.bytes)
                    .context("fixed dense expert up component end overflow")?,
            )?,
            rows: hidden_size,
            cols: intermediate_size,
            bytes: down_bytes,
        };
        let expert_bytes = aligned(
            down.offset
                .checked_add(down.bytes)
                .context("fixed dense expert down component end overflow")?,
        )?;
        Ok(Self {
            dtype,
            hidden_size,
            intermediate_size,
            gate,
            up,
            down,
            expert_bytes,
        })
    }

    pub(crate) fn from_model_layout(
        layout: &QwenMoeModelLayout,
        dtype: DenseExpertDtype,
    ) -> Result<Self> {
        Self::new(dtype, layout.hidden_size, layout.moe_intermediate_size).with_context(|| {
            format!(
                "FlashMoe unsupported {:?} fixed-{} expert storage dimensions",
                layout.family,
                dtype.as_str()
            )
        })
    }

    pub(crate) const fn projection(
        self,
        projection: ExpertMlpProjection,
    ) -> DenseExpertProjectionSpec {
        match projection {
            ExpertMlpProjection::Gate => self.gate,
            ExpertMlpProjection::Up => self.up,
            ExpertMlpProjection::Down => self.down,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeepSeekGgufExpertDtype {
    Iq2Xxs,
    Q2K,
}

impl DeepSeekGgufExpertDtype {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Q2K => "Q2_K",
        }
    }

    const fn block_elements(self) -> usize {
        256
    }

    const fn block_bytes(self) -> usize {
        match self {
            Self::Iq2Xxs => 66,
            Self::Q2K => 84,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeepSeekGgufExpertProjectionSpec {
    pub(crate) offset: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) bytes: usize,
    pub(crate) dtype: DeepSeekGgufExpertDtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeepSeekGgufExpertSlotSpec {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) gate: DeepSeekGgufExpertProjectionSpec,
    pub(crate) up: DeepSeekGgufExpertProjectionSpec,
    pub(crate) down: DeepSeekGgufExpertProjectionSpec,
    pub(crate) expert_bytes: usize,
}

impl DeepSeekGgufExpertSlotSpec {
    fn aligned(value: usize) -> Result<usize> {
        value
            .checked_add(EXPERT_COMPONENT_ALIGNMENT - 1)
            .map(|value| value / EXPERT_COMPONENT_ALIGNMENT * EXPERT_COMPONENT_ALIGNMENT)
            .context("DeepSeek GGUF expert component alignment overflow")
    }

    fn projection_bytes(rows: usize, cols: usize, dtype: DeepSeekGgufExpertDtype) -> Result<usize> {
        let values = rows
            .checked_mul(cols)
            .context("DeepSeek GGUF expert projection element count overflow")?;
        if values % dtype.block_elements() != 0 {
            bail!(
                "DeepSeek GGUF {} expert projection {}x{} is not block aligned to {} elements",
                dtype.as_str(),
                rows,
                cols,
                dtype.block_elements()
            );
        }
        values
            .checked_div(dtype.block_elements())
            .and_then(|blocks| blocks.checked_mul(dtype.block_bytes()))
            .context("DeepSeek GGUF expert projection byte length overflow")
    }

    pub(crate) fn new(hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        if hidden_size == 0 || intermediate_size == 0 {
            bail!(
                "DeepSeek GGUF expert slot spec requires non-zero dimensions, hidden_size={hidden_size}, intermediate_size={intermediate_size}"
            );
        }
        let gate_bytes = Self::projection_bytes(
            intermediate_size,
            hidden_size,
            DeepSeekGgufExpertDtype::Iq2Xxs,
        )?;
        let down_bytes =
            Self::projection_bytes(hidden_size, intermediate_size, DeepSeekGgufExpertDtype::Q2K)?;
        let gate = DeepSeekGgufExpertProjectionSpec {
            offset: 0,
            rows: intermediate_size,
            cols: hidden_size,
            bytes: gate_bytes,
            dtype: DeepSeekGgufExpertDtype::Iq2Xxs,
        };
        let up = DeepSeekGgufExpertProjectionSpec {
            offset: Self::aligned(gate.bytes)?,
            ..gate
        };
        let down = DeepSeekGgufExpertProjectionSpec {
            offset: Self::aligned(
                up.offset
                    .checked_add(up.bytes)
                    .context("DeepSeek GGUF expert up component end overflow")?,
            )?,
            rows: hidden_size,
            cols: intermediate_size,
            bytes: down_bytes,
            dtype: DeepSeekGgufExpertDtype::Q2K,
        };
        let expert_bytes = Self::aligned(
            down.offset
                .checked_add(down.bytes)
                .context("DeepSeek GGUF expert down component end overflow")?,
        )?;
        Ok(Self {
            hidden_size,
            intermediate_size,
            gate,
            up,
            down,
            expert_bytes,
        })
    }

    pub(crate) fn from_model_layout(layout: &QwenMoeModelLayout) -> Result<Self> {
        Self::new(layout.hidden_size, layout.moe_intermediate_size).with_context(|| {
            format!(
                "FlashMoe unsupported {:?} fixed DeepSeek GGUF expert storage dimensions",
                layout.family
            )
        })
    }

    pub(crate) const fn projection(
        self,
        projection: ExpertMlpProjection,
    ) -> DeepSeekGgufExpertProjectionSpec {
        match projection {
            ExpertMlpProjection::Gate => self.gate,
            ExpertMlpProjection::Up => self.up,
            ExpertMlpProjection::Down => self.down,
        }
    }

    pub(crate) fn validate_metadata(self, metadata: &ExpertLayerPackMetadata) -> Result<()> {
        if metadata.format != FIXED_DEEPSEEK_GGUF_EXPERT_LAYER_FORMAT_V1 {
            bail!(
                "DeepSeek GGUF expert layer {} declares format {}, expected {}",
                metadata.layer,
                metadata.format,
                FIXED_DEEPSEEK_GGUF_EXPERT_LAYER_FORMAT_V1
            );
        }
        if metadata.expert_size != self.expert_bytes as u64 {
            bail!(
                "DeepSeek GGUF expert layer {} has slot size {}, expected {}",
                metadata.layer,
                metadata.expert_size,
                self.expert_bytes
            );
        }
        for pack in &metadata.packs {
            if pack.packed_bytes != self.expert_bytes as u64 || pack.records.len() != 3 {
                bail!(
                    "DeepSeek GGUF expert layer {} expert {} must contain one whole slot and exactly gate/up/down records",
                    metadata.layer,
                    pack.expert
                );
            }
            for (suffix, expected) in [
                ("ffn_gate_exps.weight", self.gate),
                ("ffn_up_exps.weight", self.up),
                ("ffn_down_exps.weight", self.down),
            ] {
                let record = pack
                    .records
                    .iter()
                    .find(|record| record.tensor.ends_with(suffix))
                    .with_context(|| {
                        format!(
                            "DeepSeek GGUF expert layer {} expert {} is missing {suffix}",
                            metadata.layer, pack.expert
                        )
                    })?;
                if !record.dtype.eq_ignore_ascii_case(expected.dtype.as_str())
                    || record.shape != [expected.cols, expected.rows]
                    || record.record_offset != expected.offset as u64
                    || record.packed_bytes != expected.bytes as u64
                    || record.group_size != expected.dtype.block_elements()
                    || !record.scale_bias_dtype.eq_ignore_ascii_case("GGUF_NATIVE")
                {
                    bail!(
                        "DeepSeek GGUF expert layer {} expert {} tensor {} does not match resolved {} {}x{} block layout",
                        metadata.layer,
                        pack.expert,
                        record.tensor,
                        expected.dtype.as_str(),
                        expected.rows,
                        expected.cols
                    );
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertSlotSpec {
    FixedQ4(FixedQ4ExpertSlotSpec),
    FixedDense(FixedDenseExpertSlotSpec),
    FixedDeepSeekGguf(DeepSeekGgufExpertSlotSpec),
}

impl ExpertSlotSpec {
    pub(crate) fn from_model_layout(
        layout: &QwenMoeModelLayout,
        storage: ExpertStorageLayout,
    ) -> Result<Self> {
        match storage {
            ExpertStorageLayout::FixedQ4 => {
                FixedQ4ExpertSlotSpec::from_model_layout(layout).map(Self::FixedQ4)
            }
            ExpertStorageLayout::FixedMxfp4 => {
                FixedQ4ExpertSlotSpec::mxfp4_from_model_layout(layout).map(Self::FixedQ4)
            }
            ExpertStorageLayout::FixedBf16 => {
                FixedDenseExpertSlotSpec::from_model_layout(layout, DenseExpertDtype::Bf16)
                    .map(Self::FixedDense)
            }
            ExpertStorageLayout::FixedF16 => {
                FixedDenseExpertSlotSpec::from_model_layout(layout, DenseExpertDtype::F16)
                    .map(Self::FixedDense)
            }
            ExpertStorageLayout::FixedDeepSeekGguf => {
                DeepSeekGgufExpertSlotSpec::from_model_layout(layout).map(Self::FixedDeepSeekGguf)
            }
        }
    }

    pub(crate) const fn storage_layout(self) -> ExpertStorageLayout {
        match self {
            Self::FixedQ4(spec) => match spec.encoding {
                FixedQ4ExpertEncoding::AffineBf16 => ExpertStorageLayout::FixedQ4,
                FixedQ4ExpertEncoding::MlxMxfp4 => ExpertStorageLayout::FixedMxfp4,
            },
            Self::FixedDense(spec) => match spec.dtype {
                DenseExpertDtype::Bf16 => ExpertStorageLayout::FixedBf16,
                DenseExpertDtype::F16 => ExpertStorageLayout::FixedF16,
            },
            Self::FixedDeepSeekGguf(_) => ExpertStorageLayout::FixedDeepSeekGguf,
        }
    }

    pub(crate) const fn expert_bytes(self) -> usize {
        match self {
            Self::FixedQ4(spec) => spec.layout.expert_bytes,
            Self::FixedDense(spec) => spec.expert_bytes,
            Self::FixedDeepSeekGguf(spec) => spec.expert_bytes,
        }
    }

    pub(crate) const fn metadata_format(self) -> &'static str {
        match self {
            Self::FixedQ4(spec) => spec.metadata_format(),
            Self::FixedDense(_) => FIXED_DENSE_EXPERT_LAYER_FORMAT_V1,
            Self::FixedDeepSeekGguf(_) => FIXED_DEEPSEEK_GGUF_EXPERT_LAYER_FORMAT_V1,
        }
    }

    pub(crate) const fn fixed_q4(self) -> Option<FixedQ4ExpertSlotSpec> {
        match self {
            Self::FixedQ4(spec) => Some(spec),
            Self::FixedDense(_) | Self::FixedDeepSeekGguf(_) => None,
        }
    }

    pub(crate) const fn fixed_dense(self) -> Option<FixedDenseExpertSlotSpec> {
        match self {
            Self::FixedDense(spec) => Some(spec),
            Self::FixedQ4(_) | Self::FixedDeepSeekGguf(_) => None,
        }
    }

    pub(crate) const fn fixed_deepseek_gguf(self) -> Option<DeepSeekGgufExpertSlotSpec> {
        match self {
            Self::FixedDeepSeekGguf(spec) => Some(spec),
            Self::FixedQ4(_) | Self::FixedDense(_) => None,
        }
    }
}

#[derive(Debug)]
pub(crate) struct FixedQ4ExpertPayload {
    pub(crate) spec: FixedQ4ExpertSlotSpec,
    pub(crate) bytes: ReusableExpertBytes,
    pub(crate) recycle_pool: Option<ReusableExpertBytePool>,
}

impl Clone for FixedQ4ExpertPayload {
    fn clone(&self) -> Self {
        Self {
            spec: self.spec,
            bytes: self.bytes.clone(),
            recycle_pool: None,
        }
    }
}

impl PartialEq for FixedQ4ExpertPayload {
    fn eq(&self, other: &Self) -> bool {
        self.spec == other.spec && self.bytes == other.bytes
    }
}

impl Drop for FixedQ4ExpertPayload {
    fn drop(&mut self) {
        if let Some(pool) = &self.recycle_pool {
            recycle_reusable_expert_bytes(
                pool,
                std::mem::take(&mut self.bytes),
                self.spec.layout.expert_bytes,
            );
        }
    }
}

impl FixedQ4ExpertPayload {
    #[cfg(test)]
    pub(crate) fn from_whole_slot(
        spec: FixedQ4ExpertSlotSpec,
        bytes: Vec<u8>,
        recycle_pool: Option<ReusableExpertBytePool>,
    ) -> Result<Self> {
        Self::from_reusable_whole_slot(spec, bytes.into(), recycle_pool)
    }

    pub(crate) fn from_reusable_whole_slot(
        spec: FixedQ4ExpertSlotSpec,
        bytes: ReusableExpertBytes,
        recycle_pool: Option<ReusableExpertBytePool>,
    ) -> Result<Self> {
        if bytes.len() < spec.layout.expert_bytes {
            bail!(
                "fixed Q4 expert whole-slot payload length {} is shorter than layout size {}",
                bytes.len(),
                spec.layout.expert_bytes
            );
        }
        Ok(Self {
            spec,
            bytes,
            recycle_pool,
        })
    }

    #[cfg(test)]
    pub(crate) fn payload_prefix(&self, max_len: usize) -> &[u8] {
        &self.bytes[..self.bytes.len().min(max_len)]
    }

    pub(crate) fn component(&self, kind: QwenMoeExpertComponentKind) -> &[u8] {
        let component = self.spec.layout.component(kind);
        &self.bytes[component.offset..component.offset + component.bytes]
    }

    fn component_source(
        &self,
        weight_kind: QwenMoeExpertComponentKind,
        scale_kind: QwenMoeExpertComponentKind,
        bias_kind: QwenMoeExpertComponentKind,
    ) -> Q4MatvecSource<'_> {
        Q4MatvecSource {
            bytes: &self.bytes,
            packed_offset: self.spec.layout.component(weight_kind).offset,
            scale_offset: self.spec.layout.component(scale_kind).offset,
            bias_offset: self.spec.layout.component(bias_kind).offset,
            reusable_bytes: Some(&self.bytes),
        }
    }

    #[cfg(test)]
    fn decoded_scales_biases(
        &self,
        scale_kind: QwenMoeExpertComponentKind,
        bias_kind: QwenMoeExpertComponentKind,
        needed_groups: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let scales = decode_fixed_q4_bf16_component_bytes(self.component(scale_kind))
            .with_context(|| format!("failed to decode fixed Q4 {scale_kind:?} scales"))?;
        let biases = decode_fixed_q4_bf16_component_bytes(self.component(bias_kind))
            .with_context(|| format!("failed to decode fixed Q4 {bias_kind:?} biases"))?;
        if scales.len() < needed_groups || biases.len() < needed_groups {
            bail!(
                "fixed Q4 expert scale/bias payload is shorter than projection requires: scales={}, biases={}, required={needed_groups}",
                scales.len(),
                biases.len()
            );
        }
        Ok((scales, biases))
    }

    #[cfg(test)]
    pub(crate) fn project_cpu(
        &self,
        projection: ExpertMlpProjection,
        input: &[f32],
        output_width: usize,
    ) -> Result<Vec<f32>> {
        let payload = self
            .matvec_payload(projection, input.len(), output_width)
            .context("fixed Q4 projection metadata is incompatible with input/output shape")?;
        if self.spec.encoding == FixedQ4ExpertEncoding::MlxMxfp4 {
            return mxfp4_fma_matvec_with_group_size(
                payload.packed,
                &input[..payload.cols],
                payload.scale_bytes,
                payload.rows,
                payload.cols,
                payload.group_size,
            );
        }
        let (scale_kind, bias_kind) = projection.scale_bias_kinds();
        let (owned_scales, owned_biases);
        let (scales, biases) = if payload.scales.len() >= payload.scale_bias_groups
            && payload.biases.len() >= payload.scale_bias_groups
        {
            (payload.scales, payload.biases)
        } else {
            (owned_scales, owned_biases) =
                self.decoded_scales_biases(scale_kind, bias_kind, payload.scale_bias_groups)?;
            (
                &owned_scales[..payload.scale_bias_groups],
                &owned_biases[..payload.scale_bias_groups],
            )
        };
        q4_fma_matvec_with_group_size(
            payload.packed,
            &input[..payload.cols],
            scales,
            biases,
            payload.rows,
            payload.cols,
            payload.group_size,
        )
    }

    pub(crate) fn matvec_payload(
        &self,
        projection: ExpertMlpProjection,
        input_len: usize,
        output_width: usize,
    ) -> Option<Q4MatvecPayload<'_>> {
        if input_len == 0 || output_width == 0 {
            return None;
        }
        let (rows, cols) = match projection {
            ExpertMlpProjection::Gate | ExpertMlpProjection::Up => (
                self.spec.intermediate_size.min(output_width).max(1),
                self.spec.hidden_size.min(input_len).max(1),
            ),
            ExpertMlpProjection::Down => (
                self.spec.hidden_size.min(output_width).max(1),
                self.spec.intermediate_size.min(input_len).max(1),
            ),
        };
        let groups_per_row = cols.div_ceil(self.spec.layout.group_size).max(1);
        let needed_groups = rows.checked_mul(groups_per_row)?;
        let needed_packed = rows.checked_mul(cols.div_ceil(2))?;
        let (packed, scale_bytes, bias_bytes, source) = match projection {
            ExpertMlpProjection::Gate => (
                self.component(QwenMoeExpertComponentKind::GateWeight),
                self.component(QwenMoeExpertComponentKind::GateScale),
                self.component(QwenMoeExpertComponentKind::GateBias),
                self.component_source(
                    QwenMoeExpertComponentKind::GateWeight,
                    QwenMoeExpertComponentKind::GateScale,
                    QwenMoeExpertComponentKind::GateBias,
                ),
            ),
            ExpertMlpProjection::Up => (
                self.component(QwenMoeExpertComponentKind::UpWeight),
                self.component(QwenMoeExpertComponentKind::UpScale),
                self.component(QwenMoeExpertComponentKind::UpBias),
                self.component_source(
                    QwenMoeExpertComponentKind::UpWeight,
                    QwenMoeExpertComponentKind::UpScale,
                    QwenMoeExpertComponentKind::UpBias,
                ),
            ),
            ExpertMlpProjection::Down => (
                self.component(QwenMoeExpertComponentKind::DownWeight),
                self.component(QwenMoeExpertComponentKind::DownScale),
                self.component(QwenMoeExpertComponentKind::DownBias),
                self.component_source(
                    QwenMoeExpertComponentKind::DownWeight,
                    QwenMoeExpertComponentKind::DownScale,
                    QwenMoeExpertComponentKind::DownBias,
                ),
            ),
        };
        let (scale_bytes_per_group, bias_bytes_per_group, scale_bias_dtype) =
            match self.spec.encoding {
                FixedQ4ExpertEncoding::AffineBf16 => (2, 2, EXPERT_SCALE_BIAS_DTYPE_BF16),
                FixedQ4ExpertEncoding::MlxMxfp4 => (1, 0, EXPERT_SCALE_DTYPE_E8M0),
            };
        let needed_scale_bytes = needed_groups.checked_mul(scale_bytes_per_group)?;
        let needed_bias_bytes = needed_groups.checked_mul(bias_bytes_per_group)?;
        if packed.len() < needed_packed
            || scale_bytes.len() < needed_scale_bytes
            || bias_bytes.len() < needed_bias_bytes
        {
            return None;
        }
        Some(Q4MatvecPayload {
            rows,
            cols,
            group_size: self.spec.layout.group_size,
            packed: &packed[..needed_packed],
            #[cfg(test)]
            scales: &[],
            #[cfg(test)]
            biases: &[],
            scale_bias_groups: needed_groups,
            scale_bias_dtype,
            scale_bytes: &scale_bytes[..needed_scale_bytes],
            bias_bytes: &bias_bytes[..needed_bias_bytes],
            source: Some(source),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseMatvecSource<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) byte_offset: usize,
    pub(crate) reusable_bytes: Option<&'a ReusableExpertBytes>,
}

impl DenseMatvecSource<'_> {
    pub(crate) fn same_buffer(self, other: Self) -> bool {
        self.bytes.as_ptr() == other.bytes.as_ptr() && self.bytes.len() == other.bytes.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseMatvecPayload<'a> {
    pub(crate) dtype: DenseExpertDtype,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) source: DenseMatvecSource<'a>,
}

#[derive(Debug)]
pub(crate) struct FixedDenseExpertPayload {
    pub(crate) spec: FixedDenseExpertSlotSpec,
    pub(crate) bytes: ReusableExpertBytes,
    pub(crate) recycle_pool: Option<ReusableExpertBytePool>,
}

impl Clone for FixedDenseExpertPayload {
    fn clone(&self) -> Self {
        Self {
            spec: self.spec,
            bytes: self.bytes.clone(),
            recycle_pool: None,
        }
    }
}

impl PartialEq for FixedDenseExpertPayload {
    fn eq(&self, other: &Self) -> bool {
        self.spec == other.spec && self.bytes == other.bytes
    }
}

impl Eq for FixedDenseExpertPayload {}

impl Drop for FixedDenseExpertPayload {
    fn drop(&mut self) {
        if let Some(pool) = &self.recycle_pool {
            recycle_reusable_expert_bytes(
                pool,
                std::mem::take(&mut self.bytes),
                self.spec.expert_bytes,
            );
        }
    }
}

impl FixedDenseExpertPayload {
    #[cfg(test)]
    pub(crate) fn from_whole_slot(
        spec: FixedDenseExpertSlotSpec,
        bytes: Vec<u8>,
        recycle_pool: Option<ReusableExpertBytePool>,
    ) -> Result<Self> {
        Self::from_reusable_whole_slot(spec, bytes.into(), recycle_pool)
    }

    pub(crate) fn from_reusable_whole_slot(
        spec: FixedDenseExpertSlotSpec,
        bytes: ReusableExpertBytes,
        recycle_pool: Option<ReusableExpertBytePool>,
    ) -> Result<Self> {
        if bytes.len() < spec.expert_bytes {
            bail!(
                "fixed {} expert whole-slot payload length {} is shorter than layout size {}",
                spec.dtype.as_str(),
                bytes.len(),
                spec.expert_bytes
            );
        }
        Ok(Self {
            spec,
            bytes,
            recycle_pool,
        })
    }

    pub(crate) fn matvec_payload(
        &self,
        projection: ExpertMlpProjection,
        input_len: usize,
        output_width: usize,
    ) -> Result<DenseMatvecPayload<'_>> {
        let component = self.spec.projection(projection);
        if input_len != component.cols || output_width != component.rows {
            bail!(
                "fixed {} expert {projection:?} projection requires input/output {}/{}, got {input_len}/{output_width}",
                self.spec.dtype.as_str(),
                component.cols,
                component.rows
            );
        }
        let end = component
            .offset
            .checked_add(component.bytes)
            .context("fixed dense expert component end overflow")?;
        if end > self.bytes.len() {
            bail!(
                "fixed {} expert {projection:?} component range {}..{} exceeds whole-slot payload {}",
                self.spec.dtype.as_str(),
                component.offset,
                end,
                self.bytes.len()
            );
        }
        Ok(DenseMatvecPayload {
            dtype: self.spec.dtype,
            rows: component.rows,
            cols: component.cols,
            source: DenseMatvecSource {
                bytes: &self.bytes,
                byte_offset: component.offset,
                reusable_bytes: Some(&self.bytes),
            },
        })
    }
}

#[derive(Debug)]
pub(crate) struct DeepSeekGgufExpertPayload {
    pub(crate) spec: DeepSeekGgufExpertSlotSpec,
    pub(crate) bytes: ReusableExpertBytes,
    pub(crate) recycle_pool: Option<ReusableExpertBytePool>,
}

impl Clone for DeepSeekGgufExpertPayload {
    fn clone(&self) -> Self {
        Self {
            spec: self.spec,
            bytes: self.bytes.clone(),
            recycle_pool: None,
        }
    }
}

impl PartialEq for DeepSeekGgufExpertPayload {
    fn eq(&self, other: &Self) -> bool {
        self.spec == other.spec && self.bytes == other.bytes
    }
}

impl Eq for DeepSeekGgufExpertPayload {}

impl Drop for DeepSeekGgufExpertPayload {
    fn drop(&mut self) {
        if let Some(pool) = &self.recycle_pool {
            recycle_reusable_expert_bytes(
                pool,
                std::mem::take(&mut self.bytes),
                self.spec.expert_bytes,
            );
        }
    }
}

impl DeepSeekGgufExpertPayload {
    pub(crate) fn from_reusable_whole_slot(
        spec: DeepSeekGgufExpertSlotSpec,
        bytes: ReusableExpertBytes,
        recycle_pool: Option<ReusableExpertBytePool>,
    ) -> Result<Self> {
        if bytes.len() != spec.expert_bytes {
            bail!(
                "DeepSeek GGUF expert whole-slot payload length {} does not match resolved layout size {}",
                bytes.len(),
                spec.expert_bytes
            );
        }
        Ok(Self {
            spec,
            bytes,
            recycle_pool,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertMlpProjection {
    Gate,
    Up,
    Down,
}

impl ExpertMlpProjection {
    #[cfg(test)]
    fn scale_bias_kinds(self) -> (QwenMoeExpertComponentKind, QwenMoeExpertComponentKind) {
        match self {
            ExpertMlpProjection::Gate => (
                QwenMoeExpertComponentKind::GateScale,
                QwenMoeExpertComponentKind::GateBias,
            ),
            ExpertMlpProjection::Up => (
                QwenMoeExpertComponentKind::UpScale,
                QwenMoeExpertComponentKind::UpBias,
            ),
            ExpertMlpProjection::Down => (
                QwenMoeExpertComponentKind::DownScale,
                QwenMoeExpertComponentKind::DownBias,
            ),
        }
    }
}

#[cfg(test)]
pub(crate) fn fixed_q4_payload_from_pbq4_records(
    layer: usize,
    expert: usize,
    spec: FixedQ4ExpertSlotSpec,
    records: &[PackedExpertTensor],
    recycle_pool: Option<ReusableExpertBytePool>,
) -> Result<FixedQ4ExpertPayload> {
    let (bytes, _) = fixed_q4_pack_from_pbq4_records(layer, expert, spec, records)?;
    FixedQ4ExpertPayload::from_whole_slot(spec, bytes, recycle_pool)
}

pub(crate) fn fixed_q4_pack_from_pbq4_records(
    layer: usize,
    expert: usize,
    spec: FixedQ4ExpertSlotSpec,
    records: &[PackedExpertTensor],
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    spec.layout.validate()?;
    let gate = pbq4_record_by_suffix(records, "gate_proj.weight")?;
    let up = pbq4_record_by_suffix(records, "up_proj.weight")?;
    let down = pbq4_record_by_suffix(records, "down_proj.weight")?;

    let mut bytes = vec![0u8; spec.layout.expert_bytes];
    let mut metadata_records = Vec::with_capacity(3);
    copy_pbq4_record_to_fixed_q4_component(
        &mut bytes,
        &mut metadata_records,
        spec,
        gate,
        &[spec.intermediate_size, spec.hidden_size],
        QwenMoeExpertComponentKind::GateWeight,
        QwenMoeExpertComponentKind::GateScale,
        QwenMoeExpertComponentKind::GateBias,
    )?;
    copy_pbq4_record_to_fixed_q4_component(
        &mut bytes,
        &mut metadata_records,
        spec,
        up,
        &[spec.intermediate_size, spec.hidden_size],
        QwenMoeExpertComponentKind::UpWeight,
        QwenMoeExpertComponentKind::UpScale,
        QwenMoeExpertComponentKind::UpBias,
    )?;
    copy_pbq4_record_to_fixed_q4_component(
        &mut bytes,
        &mut metadata_records,
        spec,
        down,
        &[spec.hidden_size, spec.intermediate_size],
        QwenMoeExpertComponentKind::DownWeight,
        QwenMoeExpertComponentKind::DownScale,
        QwenMoeExpertComponentKind::DownBias,
    )?;

    let slot = ExpertSlotView::new(layer, expert, 0, spec.layout.expert_bytes, &bytes)?;
    FixedQ4ExpertSlotView::new(slot, spec.layout)?;
    Ok((
        bytes,
        ExpertPackMetadata {
            layer,
            expert,
            packed_bytes: spec.layout.expert_bytes as u64,
            records: metadata_records,
        },
    ))
}

fn pbq4_record_by_suffix<'a>(
    records: &'a [PackedExpertTensor],
    suffix: &str,
) -> Result<&'a PackedExpertTensor> {
    let matches: Vec<&PackedExpertTensor> = records
        .iter()
        .filter(|record| record.name.ends_with(suffix))
        .collect();
    match matches.as_slice() {
        [record] => Ok(*record),
        [] => bail!("PBQ4 expert pack is missing {suffix}"),
        _ => bail!("PBQ4 expert pack has duplicate {suffix} records"),
    }
}

fn copy_pbq4_record_to_fixed_q4_component(
    out: &mut [u8],
    metadata_records: &mut Vec<ExpertPackRecord>,
    spec: FixedQ4ExpertSlotSpec,
    record: &PackedExpertTensor,
    expected_shape: &[usize],
    weight_kind: QwenMoeExpertComponentKind,
    scale_kind: QwenMoeExpertComponentKind,
    bias_kind: QwenMoeExpertComponentKind,
) -> Result<()> {
    if record.shape != expected_shape {
        bail!(
            "PBQ4 expert tensor {} has shape {:?}; expected {:?}",
            record.name,
            record.shape,
            expected_shape
        );
    }
    if record.group_size != spec.layout.group_size {
        bail!(
            "PBQ4 expert tensor {} has group size {}; expected {}",
            record.name,
            record.group_size,
            spec.layout.group_size
        );
    }

    let weight = spec.layout.component(weight_kind);
    let scale = spec.layout.component(scale_kind);
    let bias = spec.layout.component(bias_kind);
    if record.packed.len() != weight.bytes {
        bail!(
            "PBQ4 expert tensor {} packed bytes {}; expected {}",
            record.name,
            record.packed.len(),
            weight.bytes
        );
    }
    out[weight.offset..weight.offset + weight.bytes].copy_from_slice(&record.packed);

    let scale_bytes = fixed_q4_bf16_scale_bias_bytes(record, true, scale.bytes)?;
    let bias_bytes = fixed_q4_bf16_scale_bias_bytes(record, false, bias.bytes)?;
    out[scale.offset..scale.offset + scale.bytes].copy_from_slice(&scale_bytes);
    out[bias.offset..bias.offset + bias.bytes].copy_from_slice(&bias_bytes);
    metadata_records.push(ExpertPackRecord {
        tensor: record.name.clone(),
        dtype: record.dtype.clone(),
        shape: record.shape.clone(),
        source_offsets: record.source_offsets(),
        source_hash: record.source_hash.clone(),
        record_offset: weight.offset as u64,
        packed_bytes: weight.bytes as u64,
        groups: record.scales.len(),
        group_size: spec.layout.group_size,
        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
    });
    Ok(())
}

fn fixed_q4_bf16_scale_bias_bytes(
    record: &PackedExpertTensor,
    scales: bool,
    expected_bytes: usize,
) -> Result<Vec<u8>> {
    if !expected_bytes.is_multiple_of(2) {
        bail!(
            "fixed Q4 component for {} has odd scale/bias byte length {expected_bytes}",
            record.name
        );
    }
    let values = if scales {
        &record.scales
    } else {
        &record.biases
    };
    let raw = if scales {
        &record.scale_bytes
    } else {
        &record.bias_bytes
    };
    let groups = expected_bytes / 2;
    if values.len() != groups {
        bail!(
            "PBQ4 expert tensor {} scale/bias groups {}; expected {groups}",
            record.name,
            values.len()
        );
    }
    if record
        .scale_bias_dtype
        .eq_ignore_ascii_case(EXPERT_SCALE_BIAS_DTYPE_BF16)
        || record.scale_bias_dtype.eq_ignore_ascii_case("BFLOAT16")
    {
        if raw.len() != expected_bytes {
            bail!(
                "PBQ4 expert tensor {} bf16 scale/bias bytes {}; expected {expected_bytes}",
                record.name,
                raw.len()
            );
        }
        return Ok(raw.clone());
    }
    let mut out = Vec::with_capacity(expected_bytes);
    for value in values {
        out.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
    }
    Ok(out)
}

pub(crate) fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

#[derive(Debug, Clone)]
pub(crate) struct Q4MatvecPayload<'a> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) group_size: usize,
    pub(crate) packed: &'a [u8],
    #[cfg(test)]
    pub(crate) scales: &'a [f32],
    #[cfg(test)]
    pub(crate) biases: &'a [f32],
    pub(crate) scale_bias_groups: usize,
    pub(crate) scale_bias_dtype: &'a str,
    pub(crate) scale_bytes: &'a [u8],
    pub(crate) bias_bytes: &'a [u8],
    pub(crate) source: Option<Q4MatvecSource<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Q4MatvecSource<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) packed_offset: usize,
    pub(crate) scale_offset: usize,
    pub(crate) bias_offset: usize,
    pub(crate) reusable_bytes: Option<&'a ReusableExpertBytes>,
}

impl<'a> Q4MatvecSource<'a> {
    pub(crate) fn same_buffer(self, other: Self) -> bool {
        self.bytes.as_ptr() == other.bytes.as_ptr() && self.bytes.len() == other.bytes.len()
    }

    pub(crate) fn covers(self, payload: &Q4MatvecPayload<'_>) -> bool {
        self.packed_offset
            .checked_add(payload.packed.len())
            .is_some_and(|end| end <= self.bytes.len())
            && self
                .scale_offset
                .checked_add(payload.scale_bytes.len())
                .is_some_and(|end| end <= self.bytes.len())
            && self
                .bias_offset
                .checked_add(payload.bias_bytes.len())
                .is_some_and(|end| end <= self.bytes.len())
    }

    pub(crate) fn offsets_are_metal_aligned(self) -> bool {
        self.packed_offset % 4 == 0 && self.scale_offset % 4 == 0 && self.bias_offset % 4 == 0
    }
}

#[cfg(test)]
pub(crate) fn decode_fixed_q4_bf16_component_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    let chunks = bytes.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        bail!(
            "fixed Q4 bf16 component has odd byte length {}",
            bytes.len()
        );
    }
    Ok(chunks
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect())
}

#[cfg(test)]
fn mxfp4_fma_matvec_with_group_size(
    packed: &[u8],
    input: &[f32],
    scales: &[u8],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Result<Vec<f32>> {
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let row_bytes = cols.div_ceil(2);
    let groups_per_row = cols.div_ceil(group_size);
    if input.len() < cols
        || packed.len() < rows.saturating_mul(row_bytes)
        || scales.len() < rows.saturating_mul(groups_per_row)
    {
        bail!("MXFP4 matvec payload is shorter than its declared shape");
    }
    let mut output = vec![0.0f32; rows];
    for (row, value) in output.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for col in 0..cols {
            let byte = packed[row * row_bytes + col / 2];
            let nibble = if col.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            };
            let magnitude = MAGNITUDES[(nibble & 0x07) as usize];
            let weight = if nibble & 0x08 == 0 {
                magnitude
            } else {
                -magnitude
            };
            let scale_bits = scales[row * groups_per_row + col / group_size];
            let scale = if scale_bits == 0 {
                f32::from_bits(0x0040_0000)
            } else {
                f32::from_bits(u32::from(scale_bits) << 23)
            };
            if !scale.is_finite() {
                bail!("MXFP4 E8M0 scale byte 0x{scale_bits:02x} is not finite");
            }
            sum += weight * scale * input[col];
        }
        *value = sum;
    }
    Ok(output)
}
