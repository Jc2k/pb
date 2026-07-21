use super::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalPostAttentionPrep {
    pub(crate) residual_buffer: MetalObjcId,
    pub(crate) normed_buffer: MetalObjcId,
    pub(crate) input: ScheduledCmd3MetalPostAttentionInput,
    pub(crate) state: FlashMoePostAttentionPrepState,
    pub(crate) width: usize,
    pub(crate) active: InlineRoutePairs,
    routing_command: Option<ScheduledRoutingCommand>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalPostAttentionPrep {
    pub(crate) fn new(
        layer: usize,
        width: usize,
        expert_count: usize,
        active: impl Into<InlineRoutePairs>,
        residual_buffer: MetalObjcId,
        normed_buffer: MetalObjcId,
    ) -> anyhow::Result<Self> {
        let active = active.into();
        let state = FlashMoePostAttentionPrepState::new(layer, width, expert_count, active.len());
        if !state.is_declared_graph_state() {
            anyhow::bail!(
                "FlashMoe unsupported Metal post-attention input for layer {layer}: prep state is not declared graph state"
            );
        }
        let input = ScheduledCmd3MetalPostAttentionInput::new(state, active.len())?;
        Ok(Self {
            residual_buffer,
            normed_buffer,
            input,
            state,
            width,
            active,
            routing_command: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn routing_command(&self) -> Option<&ScheduledRoutingCommand> {
        self.routing_command.as_ref()
    }

    pub(crate) fn attach_routing_command(
        &mut self,
        command: ScheduledRoutingCommand,
    ) -> anyhow::Result<ScheduledRoutingCommand> {
        let routing = self.state.routing();
        if command.layer != routing.layer() || command.routing.layer != routing.layer() {
            anyhow::bail!(
                "FlashMoe CMD2 Metal post-attention prep routing layer {} does not match command layer {}",
                routing.layer(),
                command.layer
            );
        }
        if command.routing.experts != routing.experts() {
            anyhow::bail!(
                "FlashMoe CMD2 Metal post-attention prep expert count {} does not match command experts {}",
                routing.experts(),
                command.routing.experts
            );
        }
        if command.active_experts != routing.active_experts()
            || command.routing.active_experts != routing.active_experts()
            || command.routes.len() != self.active.len()
        {
            anyhow::bail!(
                "FlashMoe CMD2 Metal post-attention prep active route count {} does not match command active_experts={} routes={}",
                self.active.len(),
                command.active_experts,
                command.routes.len()
            );
        }
        if command.source != ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK {
            anyhow::bail!(
                "FlashMoe CMD2 Metal post-attention prep requires fused-prep CPU topK routing, got {:?}",
                command.source
            );
        }
        if command.routes != self.active {
            anyhow::bail!(
                "FlashMoe CMD2 Metal post-attention prep routes do not match the scheduler routing command"
            );
        }
        self.routing_command = Some(command.clone());
        Ok(command)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalCmd3DeferredOutput {
    pub(crate) hidden_buffer: MetalObjcId,
    pub(crate) next_normed_buffer: Option<MetalObjcId>,
    pub(crate) output_state: FlashMoeCmd3OutputState,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3DeferredOutput {
    pub(crate) fn new(
        hidden_buffer: MetalObjcId,
        next_normed_buffer: Option<MetalObjcId>,
        output_state: FlashMoeCmd3OutputState,
    ) -> anyhow::Result<Self> {
        if hidden_buffer.is_null() {
            anyhow::bail!("FlashMoe CMD3 deferred output requires a non-null hidden buffer");
        }
        if !output_state.is_declared_graph_state() {
            anyhow::bail!("FlashMoe CMD3 deferred output state is not declared graph state");
        }
        if next_normed_buffer.is_some() != output_state.has_next_normed() {
            anyhow::bail!(
                "FlashMoe CMD3 deferred output next-norm buffer presence does not match declared output state"
            );
        }
        Ok(Self {
            hidden_buffer,
            next_normed_buffer,
            output_state,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3PhasePlan {
    pub(crate) position: usize,
    pub(crate) layer: usize,
    pub(crate) expert_count: usize,
    pub(crate) width: usize,
    pub(crate) output_state: FlashMoeCmd3OutputState,
    pub(crate) has_next_norm: bool,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3InputBuffers {
    pub(crate) normed: MetalObjcId,
    pub(crate) residual: MetalObjcId,
    pub(crate) phase: MetalCmd3PhasePlan,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3InputBuffers {
    pub(crate) fn new(
        phase: MetalCmd3PhasePlan,
        normed: MetalObjcId,
        residual: MetalObjcId,
    ) -> anyhow::Result<Self> {
        if normed.is_null() {
            anyhow::bail!("FlashMoe Metal CMD3 input requires a non-null normed buffer");
        }
        if residual.is_null() {
            anyhow::bail!("FlashMoe Metal CMD3 input requires a non-null residual buffer");
        }
        Ok(Self {
            normed,
            residual,
            phase,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3PhasePlan {
    pub(crate) fn new(
        position: usize,
        layer: usize,
        expert_count: usize,
        width: usize,
        weights_len: usize,
        payloads_len: usize,
        output_state: FlashMoeCmd3OutputState,
        has_next_norm: bool,
    ) -> anyhow::Result<Self> {
        if width == 0 {
            anyhow::bail!("FlashMoe Metal CMD3 phase requires non-zero width");
        }
        if expert_count == 0 {
            anyhow::bail!("FlashMoe Metal CMD3 phase requires at least one active expert");
        }
        if width > u32::MAX as usize {
            anyhow::bail!(
                "FlashMoe Metal CMD3 phase width {} does not fit Metal u32 constants",
                width
            );
        }
        if expert_count > u32::MAX as usize {
            anyhow::bail!(
                "FlashMoe Metal CMD3 phase expert count {} does not fit Metal u32 constants",
                expert_count
            );
        }
        if weights_len != expert_count || payloads_len != expert_count {
            anyhow::bail!(
                "FlashMoe Metal CMD3 phase expert count {} does not match weights={} payloads={}",
                expert_count,
                weights_len,
                payloads_len
            );
        }
        if !output_state.is_declared_graph_state() {
            anyhow::bail!("FlashMoe Metal CMD3 phase output state is not declared graph state");
        }
        if output_state.width() != width {
            anyhow::bail!(
                "FlashMoe Metal CMD3 phase output width {} does not match command width {}",
                output_state.width(),
                width
            );
        }
        if output_state.has_next_normed() != has_next_norm {
            anyhow::bail!(
                "FlashMoe Metal CMD3 phase next-norm output declaration does not match next-norm weights"
            );
        }
        Ok(Self {
            position,
            layer,
            expert_count,
            width,
            output_state,
            has_next_norm,
        })
    }

    pub(crate) fn width_u32(self) -> u32 {
        self.width as u32
    }

    pub(crate) fn expert_outputs_bytes(self) -> anyhow::Result<usize> {
        let items = self.expert_count.checked_mul(self.width).ok_or_else(|| {
            anyhow::anyhow!("FlashMoe Metal CMD3 expert output item count overflow")
        })?;
        Self::f32_bytes("expert output", items)
    }

    pub(crate) fn shared_output_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("shared expert output", self.width)
    }

    pub(crate) fn hidden_output_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("hidden output", self.width)
    }

    pub(crate) fn next_normed_output_bytes(self) -> anyhow::Result<Option<usize>> {
        if self.has_next_norm {
            Self::f32_bytes("next-normed output", self.width).map(Some)
        } else {
            Ok(None)
        }
    }

    pub(crate) fn expert_output_offset(self, index: usize) -> anyhow::Result<u64> {
        if index >= self.expert_count {
            anyhow::bail!(
                "FlashMoe Metal CMD3 expert output index {} exceeds active expert count {}",
                index,
                self.expert_count
            );
        }
        let items = index.checked_mul(self.width).ok_or_else(|| {
            anyhow::anyhow!("FlashMoe Metal CMD3 expert output offset item count overflow")
        })?;
        let bytes = Self::f32_bytes("expert output offset", items)?;
        Ok(bytes as u64)
    }

    fn f32_bytes(label: &str, items: usize) -> anyhow::Result<usize> {
        items
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("FlashMoe Metal CMD3 {label} byte size overflow"))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3CombinePlan {
    pub(crate) width: usize,
    pub(crate) active_count: usize,
    pub(crate) dispatch_threads: u64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3CombineBufferLayout {
    pub(crate) width_u32: u32,
    pub(crate) active_count_u32: u32,
    pub(crate) routing_weights_bytes: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3CombineBuffers {
    pub(crate) routing_weights: MetalObjcId,
    pub(crate) width: MetalObjcId,
    pub(crate) active_count: MetalObjcId,
    pub(crate) layout: MetalCmd3CombineBufferLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3CombineStageBuffers {
    pub(crate) residual: MetalObjcId,
    pub(crate) shared_output: MetalObjcId,
    pub(crate) expert_outputs: MetalObjcId,
    pub(crate) routing_weights: MetalObjcId,
    pub(crate) hidden: MetalObjcId,
    pub(crate) width: MetalObjcId,
    pub(crate) active_count: MetalObjcId,
    pub(crate) plan: MetalCmd3CombinePlan,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3CombineStageBuffers {
    pub(crate) fn new(
        plan: MetalCmd3CombinePlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        combine: MetalCmd3CombineBuffers,
    ) -> anyhow::Result<Self> {
        let layout = plan.buffer_layout()?;
        if combine.layout != layout {
            anyhow::bail!("FlashMoe Metal CMD3 combine stage constants do not match plan");
        }
        if outputs.layout.width_u32 != layout.width_u32
            || outputs.layout.active_count_u32 != layout.active_count_u32
        {
            anyhow::bail!("FlashMoe Metal CMD3 combine stage outputs do not match plan layout");
        }
        Ok(Self {
            residual: inputs.residual,
            shared_output: outputs.shared_output,
            expert_outputs: outputs.expert_outputs,
            routing_weights: combine.routing_weights,
            hidden: outputs.hidden,
            width: combine.width,
            active_count: combine.active_count,
            plan,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3CombineBuffers {
    pub(crate) fn new(
        plan: MetalCmd3CombinePlan,
        routing_weights: MetalObjcId,
        width: MetalObjcId,
        active_count: MetalObjcId,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            routing_weights,
            width,
            active_count,
            layout: plan.buffer_layout()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3CombinePlan {
    pub(crate) fn new(phase: MetalCmd3PhasePlan) -> Self {
        Self {
            width: phase.width,
            active_count: phase.expert_count,
            dispatch_threads: phase.width as u64,
        }
    }

    pub(crate) fn active_count_u32(self) -> u32 {
        self.active_count as u32
    }

    pub(crate) fn routing_weights_bytes(self) -> anyhow::Result<usize> {
        self.active_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                anyhow::anyhow!("FlashMoe Metal CMD3 combine routing weights byte size overflow")
            })
    }

    pub(crate) fn buffer_layout(self) -> anyhow::Result<MetalCmd3CombineBufferLayout> {
        Ok(MetalCmd3CombineBufferLayout {
            width_u32: self.width as u32,
            active_count_u32: self.active_count_u32(),
            routing_weights_bytes: self.routing_weights_bytes()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3NextNormPlan {
    pub(crate) width: usize,
    pub(crate) dispatch_threads: u64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3NextNormBufferLayout {
    pub(crate) width_u32: u32,
    pub(crate) weight_bytes: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3NextNormBuffers {
    pub(crate) hidden: MetalObjcId,
    pub(crate) weight: MetalObjcId,
    pub(crate) next_normed: MetalObjcId,
    pub(crate) width: MetalObjcId,
    pub(crate) layout: MetalCmd3NextNormBufferLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3NextNormBuffers {
    pub(crate) fn new(
        plan: MetalCmd3NextNormPlan,
        hidden: MetalObjcId,
        weight: MetalObjcId,
        next_normed: MetalObjcId,
        width: MetalObjcId,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            hidden,
            weight,
            next_normed,
            width,
            layout: plan.buffer_layout()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3NextNormPlan {
    const RMS_NORM_REDUCED_THREADS: u64 = 256;

    pub(crate) fn new(
        phase: MetalCmd3PhasePlan,
        weight_len: Option<usize>,
    ) -> anyhow::Result<Option<Self>> {
        match (phase.has_next_norm, weight_len) {
            (false, None) => Ok(None),
            (false, Some(_)) => anyhow::bail!(
                "FlashMoe Metal CMD3 next-norm weights were provided for a no-next-norm phase"
            ),
            (true, None) => anyhow::bail!(
                "FlashMoe Metal CMD3 next-norm output is declared but no next-norm weights were provided"
            ),
            (true, Some(weight_len)) => {
                if weight_len < phase.width {
                    anyhow::bail!(
                        "FlashMoe Metal CMD3 next-norm weight length {} is smaller than width {} for layer {}",
                        weight_len,
                        phase.width,
                        phase.layer
                    );
                }
                Ok(Some(Self {
                    width: phase.width,
                    dispatch_threads: Self::RMS_NORM_REDUCED_THREADS,
                }))
            }
        }
    }

    pub(crate) fn weight_bytes(self) -> anyhow::Result<usize> {
        self.width
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                anyhow::anyhow!("FlashMoe Metal CMD3 next-norm weight byte size overflow")
            })
    }

    pub(crate) fn buffer_layout(self) -> anyhow::Result<MetalCmd3NextNormBufferLayout> {
        Ok(MetalCmd3NextNormBufferLayout {
            width_u32: self.width as u32,
            weight_bytes: self.weight_bytes()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalCmd3SharedPhaseSource {
    None,
    Dense,
    Resident,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3SharedPhasePlan {
    pub(crate) source: MetalCmd3SharedPhaseSource,
    pub(crate) width: usize,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) total_intermediate: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3SharedBufferLayout {
    pub(crate) total_intermediate_u32: u32,
    pub(crate) intermediate_u32: u32,
    pub(crate) projection_output_bytes: usize,
    pub(crate) router_output_bytes: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3SharedWorkBuffers {
    pub(crate) gate_out: MetalObjcId,
    pub(crate) up_out: MetalObjcId,
    pub(crate) router_out: MetalObjcId,
    pub(crate) activated: MetalObjcId,
    pub(crate) total_intermediate: MetalObjcId,
    pub(crate) intermediate: MetalObjcId,
    pub(crate) layout: MetalCmd3SharedBufferLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3SharedStageBuffers {
    pub(crate) source: MetalCmd3SharedPhaseSource,
    pub(crate) normed: MetalObjcId,
    pub(crate) width: MetalObjcId,
    pub(crate) shared_output: MetalObjcId,
    pub(crate) work: Option<MetalCmd3SharedWorkBuffers>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3SharedStageBuffers {
    pub(crate) fn projected(
        plan: MetalCmd3SharedPhasePlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        combine: MetalCmd3CombineBuffers,
        work: MetalCmd3SharedWorkBuffers,
    ) -> anyhow::Result<Self> {
        if plan.source == MetalCmd3SharedPhaseSource::None {
            anyhow::bail!(
                "FlashMoe Metal CMD3 projected shared stage requires a declared shared expert source"
            );
        }
        if work.layout != plan.buffer_layout()? {
            anyhow::bail!("FlashMoe Metal CMD3 shared stage work layout does not match plan");
        }
        Ok(Self {
            source: plan.source,
            normed: inputs.normed,
            width: combine.width,
            shared_output: outputs.shared_output,
            work: Some(work),
        })
    }

    pub(crate) fn fill_zero(
        plan: MetalCmd3SharedPhasePlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        combine: MetalCmd3CombineBuffers,
    ) -> anyhow::Result<Self> {
        if plan.source != MetalCmd3SharedPhaseSource::None {
            anyhow::bail!(
                "FlashMoe Metal CMD3 fill-zero shared stage requires no shared expert source"
            );
        }
        Ok(Self {
            source: plan.source,
            normed: inputs.normed,
            width: combine.width,
            shared_output: outputs.shared_output,
            work: None,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3SharedWorkBuffers {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        plan: MetalCmd3SharedPhasePlan,
        gate_out: MetalObjcId,
        up_out: MetalObjcId,
        router_out: MetalObjcId,
        activated: MetalObjcId,
        total_intermediate: MetalObjcId,
        intermediate: MetalObjcId,
    ) -> anyhow::Result<Self> {
        if plan.source == MetalCmd3SharedPhaseSource::None {
            anyhow::bail!(
                "FlashMoe Metal CMD3 shared work buffers require a declared shared expert source"
            );
        }
        Ok(Self {
            gate_out,
            up_out,
            router_out,
            activated,
            total_intermediate,
            intermediate,
            layout: plan.buffer_layout()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3SharedPhasePlan {
    pub(crate) const fn none(width: usize) -> Self {
        Self {
            source: MetalCmd3SharedPhaseSource::None,
            width,
            shared_experts: 0,
            intermediate: 0,
            total_intermediate: 0,
        }
    }

    pub(crate) fn dense(width: usize, shared: &SharedExpertPhaseWeights) -> anyhow::Result<Self> {
        let shape = shared.validated_shape()?;
        Self::from_shape(MetalCmd3SharedPhaseSource::Dense, width, shape)
    }

    pub(crate) fn resident(
        width: usize,
        shared: &SharedExpertPhaseResidentProjections,
    ) -> anyhow::Result<Self> {
        let shape = shared.validated_shape()?;
        Self::from_shape(MetalCmd3SharedPhaseSource::Resident, width, shape)
    }

    pub(crate) fn total_intermediate_u32(self) -> anyhow::Result<u32> {
        Self::usize_to_u32("total intermediate width", self.total_intermediate)
    }

    pub(crate) fn intermediate_u32(self) -> anyhow::Result<u32> {
        Self::usize_to_u32("per-shared-expert intermediate width", self.intermediate)
    }

    pub(crate) fn projection_output_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("projection output", self.total_intermediate)
    }

    pub(crate) fn router_output_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("router output", self.shared_experts)
    }

    #[cfg(test)]
    pub(crate) fn projection_rows(self) -> usize {
        self.total_intermediate
    }

    #[cfg(test)]
    pub(crate) fn router_rows(self) -> usize {
        self.shared_experts
    }

    pub(crate) fn activation_dispatch_threads(self) -> u64 {
        self.total_intermediate as u64
    }

    pub(crate) fn fill_zero_width(self) -> usize {
        self.width
    }

    pub(crate) fn buffer_layout(self) -> anyhow::Result<MetalCmd3SharedBufferLayout> {
        Ok(MetalCmd3SharedBufferLayout {
            total_intermediate_u32: self.total_intermediate_u32()?,
            intermediate_u32: self.intermediate_u32()?,
            projection_output_bytes: self.projection_output_bytes()?,
            router_output_bytes: self.router_output_bytes()?,
        })
    }

    pub(crate) fn from_shape(
        source: MetalCmd3SharedPhaseSource,
        width: usize,
        shape: super::super::weights::SharedExpertPhaseShape,
    ) -> anyhow::Result<Self> {
        if shape.width != width {
            anyhow::bail!(
                "FlashMoe Metal CMD3 shared expert width {} does not match phase width {}",
                shape.width,
                width
            );
        }
        Self::usize_to_u32("total intermediate width", shape.total_intermediate)?;
        Self::usize_to_u32("per-shared-expert intermediate width", shape.intermediate)?;
        Ok(Self {
            source,
            width,
            shared_experts: shape.shared_experts,
            intermediate: shape.intermediate,
            total_intermediate: shape.total_intermediate,
        })
    }

    fn usize_to_u32(label: &str, value: usize) -> anyhow::Result<u32> {
        u32::try_from(value).map_err(|_| {
            anyhow::anyhow!(
                "FlashMoe Metal CMD3 shared expert {label} {value} does not fit Metal u32 constants"
            )
        })
    }

    fn f32_bytes(label: &str, items: usize) -> anyhow::Result<usize> {
        items
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                anyhow::anyhow!("FlashMoe Metal CMD3 shared expert {label} byte size overflow")
            })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3ActiveExpertPlan {
    pub(crate) index: usize,
    pub(crate) source: MetalCmd3ActiveExpertSource,
    pub(crate) intermediate: usize,
    pub(crate) output_offset: u64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalCmd3ActiveExpertSource {
    Q4,
    Dense,
    DeepSeekGguf,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3ActiveExpertBufferLayout {
    pub(crate) intermediate_u32: u32,
    pub(crate) activation_bytes: usize,
    pub(crate) projection_output_bytes: Option<usize>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3ActiveExpertWorkBuffers {
    pub(crate) gate_out: Option<MetalObjcId>,
    pub(crate) up_out: Option<MetalObjcId>,
    pub(crate) activated: MetalObjcId,
    pub(crate) layout: MetalCmd3ActiveExpertBufferLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3ActiveExpertStageBuffers {
    pub(crate) normed: MetalObjcId,
    pub(crate) activated: MetalObjcId,
    pub(crate) expert_outputs: MetalObjcId,
    pub(crate) output_offset: u64,
    pub(crate) plan: MetalCmd3ActiveExpertPlan,
    pub(crate) work: MetalCmd3ActiveExpertWorkBuffers,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3ActiveExpertStageBuffers {
    pub(crate) fn new(
        plan: MetalCmd3ActiveExpertPlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        work: MetalCmd3ActiveExpertWorkBuffers,
    ) -> anyhow::Result<Self> {
        if work.layout != plan.buffer_layout()? {
            anyhow::bail!("FlashMoe Metal CMD3 active expert work layout does not match plan");
        }
        Ok(Self {
            normed: inputs.normed,
            activated: work.activated,
            expert_outputs: outputs.expert_outputs,
            output_offset: plan.output_offset,
            plan,
            work,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3ActiveExpertWorkBuffers {
    pub(crate) fn new(
        plan: MetalCmd3ActiveExpertPlan,
        gate_out: Option<MetalObjcId>,
        up_out: Option<MetalObjcId>,
        activated: MetalObjcId,
    ) -> anyhow::Result<Self> {
        let requires_projection_outputs = true;
        if gate_out.is_some() != requires_projection_outputs
            || up_out.is_some() != requires_projection_outputs
        {
            anyhow::bail!(
                "FlashMoe Metal CMD3 active expert work buffers do not match the declared payload source"
            );
        }
        Ok(Self {
            gate_out,
            up_out,
            activated,
            layout: plan.buffer_layout()?,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3ActiveExpertPlan {
    pub(crate) fn new(
        phase: MetalCmd3PhasePlan,
        index: usize,
        payload: &ScheduledExpertPhaseMlpPayload<'_>,
    ) -> anyhow::Result<Self> {
        let (source, gate_rows, gate_cols, up_rows, up_cols, down_rows, down_cols) = match payload {
            ScheduledExpertPhaseMlpPayload::Q4(payload) => (
                MetalCmd3ActiveExpertSource::Q4,
                payload.gate.rows,
                payload.gate.cols,
                payload.up.rows,
                payload.up.cols,
                payload.down.rows,
                payload.down.cols,
            ),
            ScheduledExpertPhaseMlpPayload::Dense(payload) => (
                MetalCmd3ActiveExpertSource::Dense,
                payload.gate.rows,
                payload.gate.cols,
                payload.up.rows,
                payload.up.cols,
                payload.down.rows,
                payload.down.cols,
            ),
            ScheduledExpertPhaseMlpPayload::DeepSeekGguf(payload) => (
                MetalCmd3ActiveExpertSource::DeepSeekGguf,
                payload.spec.gate.rows,
                payload.spec.gate.cols,
                payload.spec.up.rows,
                payload.spec.up.cols,
                payload.spec.down.rows,
                payload.spec.down.cols,
            ),
        };
        if gate_rows == 0 {
            anyhow::bail!("FlashMoe Metal CMD3 active expert requires non-zero intermediate width");
        }
        if gate_rows != up_rows || down_cols != gate_rows {
            anyhow::bail!(
                "FlashMoe Metal CMD3 active expert payload has mismatched intermediate widths: gate={gate_rows} up={up_rows} down_cols={down_cols}"
            );
        }
        if gate_cols != phase.width || up_cols != phase.width || down_rows != phase.width {
            anyhow::bail!(
                "FlashMoe Metal CMD3 active expert payload width does not match phase width {}: gate={} up={} down_rows={}",
                phase.width,
                gate_cols,
                up_cols,
                down_rows
            );
        }
        Self::usize_to_u32("intermediate width", gate_rows)?;
        Ok(Self {
            index,
            source,
            intermediate: gate_rows,
            output_offset: phase.expert_output_offset(index)?,
        })
    }

    pub(crate) fn intermediate_u32(self) -> anyhow::Result<u32> {
        Self::usize_to_u32("intermediate width", self.intermediate)
    }

    pub(crate) fn activation_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("activation", self.intermediate)
    }

    pub(crate) fn projection_output_bytes(self) -> anyhow::Result<usize> {
        Self::f32_bytes("projection output", self.intermediate)
    }

    pub(crate) fn buffer_layout(self) -> anyhow::Result<MetalCmd3ActiveExpertBufferLayout> {
        Ok(MetalCmd3ActiveExpertBufferLayout {
            intermediate_u32: self.intermediate_u32()?,
            activation_bytes: self.activation_bytes()?,
            projection_output_bytes: Some(self.projection_output_bytes()?),
        })
    }

    fn usize_to_u32(label: &str, value: usize) -> anyhow::Result<u32> {
        u32::try_from(value).map_err(|_| {
            anyhow::anyhow!(
                "FlashMoe Metal CMD3 active expert {label} {value} does not fit Metal u32 constants"
            )
        })
    }

    fn f32_bytes(label: &str, items: usize) -> anyhow::Result<usize> {
        items
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                anyhow::anyhow!("FlashMoe Metal CMD3 active expert {label} byte size overflow")
            })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalCmd3ExecutionPlan {
    pub(crate) phase: MetalCmd3PhasePlan,
    pub(crate) next_norm: Option<MetalCmd3NextNormPlan>,
    pub(crate) shared: MetalCmd3SharedPhasePlan,
    pub(crate) active_experts: Vec<MetalCmd3ActiveExpertPlan>,
    pub(crate) combine: MetalCmd3CombinePlan,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3BufferLayout {
    pub(crate) width_u32: u32,
    pub(crate) active_count_u32: u32,
    pub(crate) expert_outputs_bytes: usize,
    pub(crate) shared_output_bytes: usize,
    pub(crate) hidden_output_bytes: usize,
    pub(crate) next_normed_output_bytes: Option<usize>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCmd3OutputBuffers {
    pub(crate) expert_outputs: MetalObjcId,
    pub(crate) shared_output: MetalObjcId,
    pub(crate) hidden: MetalObjcId,
    pub(crate) next_normed: Option<MetalObjcId>,
    pub(crate) layout: MetalCmd3BufferLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3OutputBuffers {
    pub(crate) fn new(
        plan: &MetalCmd3ExecutionPlan,
        expert_outputs: MetalObjcId,
        shared_output: MetalObjcId,
        hidden: MetalObjcId,
        next_normed: Option<MetalObjcId>,
    ) -> anyhow::Result<Self> {
        let layout = plan.buffer_layout()?;
        if next_normed.is_some() != layout.next_normed_output_bytes.is_some() {
            anyhow::bail!(
                "FlashMoe Metal CMD3 output buffers next-normed presence does not match declared output state"
            );
        }
        Ok(Self {
            expert_outputs,
            shared_output,
            hidden,
            next_normed,
            layout,
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCmd3ExecutionPlan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        position: usize,
        layer: usize,
        expert_count: usize,
        width: usize,
        weights_len: usize,
        output_state: FlashMoeCmd3OutputState,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight_len: Option<usize>,
        payloads: &[ScheduledExpertPhaseMlpPayload<'_>],
    ) -> anyhow::Result<Self> {
        let phase = MetalCmd3PhasePlan::new(
            position,
            layer,
            expert_count,
            width,
            weights_len,
            payloads.len(),
            output_state,
            next_norm_weight_len.is_some(),
        )?;
        let next_norm = MetalCmd3NextNormPlan::new(phase, next_norm_weight_len)?;
        let shared = match shared {
            ScheduledSharedExpertPhaseRef::None => MetalCmd3SharedPhasePlan::none(width),
            ScheduledSharedExpertPhaseRef::Dense(shared) => {
                MetalCmd3SharedPhasePlan::dense(width, shared)?
            }
            ScheduledSharedExpertPhaseRef::Resident(shared) => {
                MetalCmd3SharedPhasePlan::resident(width, shared)?
            }
        };
        let active_experts = payloads
            .iter()
            .enumerate()
            .map(|(idx, payload)| MetalCmd3ActiveExpertPlan::new(phase, idx, payload))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let combine = MetalCmd3CombinePlan::new(phase);
        Ok(Self {
            phase,
            next_norm,
            shared,
            active_experts,
            combine,
        })
    }

    pub(crate) fn buffer_layout(&self) -> anyhow::Result<MetalCmd3BufferLayout> {
        Ok(MetalCmd3BufferLayout {
            width_u32: self.phase.width_u32(),
            active_count_u32: self.combine.active_count_u32(),
            expert_outputs_bytes: self.phase.expert_outputs_bytes()?,
            shared_output_bytes: self.phase.shared_output_bytes()?,
            hidden_output_bytes: self.phase.hidden_output_bytes()?,
            next_normed_output_bytes: self.phase.next_normed_output_bytes()?,
        })
    }

    pub(crate) fn command_context(&self, expert_ids: impl ToString) -> MetalCommandContext {
        MetalCommandContext::new("deferred_expert_phase_from_buffers")
            .with("position", self.phase.position)
            .with("layer", self.phase.layer)
            .with("active_experts", self.phase.expert_count)
            .with("experts", expert_ids)
            .with("width", self.phase.width)
            .with(
                "shared",
                !matches!(self.shared.source, MetalCmd3SharedPhaseSource::None),
            )
            .with("next_norm", self.next_norm.is_some())
    }
}
