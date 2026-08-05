#![cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashMoeStatePlacement {
    CpuVisible,
    GpuResident,
}

impl FlashMoeStatePlacement {
    pub(crate) const GRAPH_PLACEMENTS: [Self; 2] = [Self::CpuVisible, Self::GpuResident];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashMoeStateBufferRole {
    AttentionValues,
    Hidden,
    Residual,
    Normed,
    NextLayerNormed,
    RouterScores,
    RoutingTopK,
    Kv,
    Recurrent,
}

impl FlashMoeStateBufferRole {
    pub(crate) const GENERATION_ROLES: [Self; 9] = [
        Self::AttentionValues,
        Self::Hidden,
        Self::Residual,
        Self::Normed,
        Self::NextLayerNormed,
        Self::RouterScores,
        Self::RoutingTopK,
        Self::Kv,
        Self::Recurrent,
    ];
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeCpuBuffer {
    role: FlashMoeStateBufferRole,
    values: Vec<f32>,
}

impl FlashMoeCpuBuffer {
    fn new(role: FlashMoeStateBufferRole, values: Vec<f32>) -> Self {
        Self { role, values }
    }

    pub(crate) fn hidden(values: Vec<f32>) -> Self {
        Self::new(FlashMoeStateBufferRole::Hidden, values)
    }

    #[cfg(test)]
    pub(crate) fn residual(values: Vec<f32>) -> Self {
        Self::new(FlashMoeStateBufferRole::Residual, values)
    }

    pub(crate) fn normed(values: Vec<f32>) -> Self {
        Self::new(FlashMoeStateBufferRole::Normed, values)
    }

    pub(crate) fn next_layer_normed(values: Vec<f32>) -> Self {
        Self::new(FlashMoeStateBufferRole::NextLayerNormed, values)
    }

    pub(crate) fn into_role(mut self, role: FlashMoeStateBufferRole) -> Self {
        self.role = role;
        self
    }

    pub(crate) fn role(&self) -> FlashMoeStateBufferRole {
        self.role
    }

    pub(crate) fn placement(&self) -> FlashMoeStatePlacement {
        FlashMoeStatePlacement::CpuVisible
    }

    pub(crate) fn is_declared_graph_state(&self) -> bool {
        FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }

    pub(crate) fn replace_values(&mut self, values: Vec<f32>) {
        self.values = values;
    }

    #[cfg(test)]
    pub(crate) fn clone_values(&self) -> Vec<f32> {
        self.values.clone()
    }

    pub(crate) fn into_values(self) -> Vec<f32> {
        self.values
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeGpuBufferDescriptor {
    role: FlashMoeStateBufferRole,
    len: usize,
}

impl FlashMoeGpuBufferDescriptor {
    pub(crate) fn new(role: FlashMoeStateBufferRole, len: usize) -> Self {
        Self { role, len }
    }

    pub(crate) fn hidden(len: usize) -> Self {
        Self::new(FlashMoeStateBufferRole::Hidden, len)
    }

    pub(crate) fn residual(len: usize) -> Self {
        Self::new(FlashMoeStateBufferRole::Residual, len)
    }

    pub(crate) fn normed(len: usize) -> Self {
        Self::new(FlashMoeStateBufferRole::Normed, len)
    }

    pub(crate) fn next_layer_normed(len: usize) -> Self {
        Self::new(FlashMoeStateBufferRole::NextLayerNormed, len)
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        self.role
    }

    pub(crate) fn len(self) -> usize {
        self.len
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        FlashMoeStatePlacement::GpuResident
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
            && self.len() > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeGpuMatrixDescriptor {
    role: FlashMoeStateBufferRole,
    rows: usize,
    cols: usize,
    values: usize,
}

impl FlashMoeGpuMatrixDescriptor {
    pub(crate) fn new(role: FlashMoeStateBufferRole, rows: usize, cols: usize) -> Result<Self> {
        if rows == 0 || cols == 0 {
            bail!("FlashMoe GPU matrix requires non-zero geometry, got {rows}x{cols} for {role:?}");
        }
        let values = rows.checked_mul(cols).with_context(|| {
            format!("FlashMoe GPU matrix geometry overflows usize: {rows}x{cols} for {role:?}")
        })?;
        Ok(Self {
            role,
            rows,
            cols,
            values,
        })
    }

    pub(crate) fn hidden(rows: usize, width: usize) -> Result<Self> {
        Self::new(FlashMoeStateBufferRole::Hidden, rows, width)
    }

    pub(crate) fn attention_values(rows: usize, width: usize) -> Result<Self> {
        Self::new(FlashMoeStateBufferRole::AttentionValues, rows, width)
    }

    pub(crate) fn residual(rows: usize, width: usize) -> Result<Self> {
        Self::new(FlashMoeStateBufferRole::Residual, rows, width)
    }

    pub(crate) fn normed(rows: usize, width: usize) -> Result<Self> {
        Self::new(FlashMoeStateBufferRole::Normed, rows, width)
    }

    pub(crate) fn next_layer_normed(rows: usize, width: usize) -> Result<Self> {
        Self::new(FlashMoeStateBufferRole::NextLayerNormed, rows, width)
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        self.role
    }

    pub(crate) fn rows(self) -> usize {
        self.rows
    }

    pub(crate) fn cols(self) -> usize {
        self.cols
    }

    pub(crate) fn values(self) -> usize {
        self.values
    }

    #[cfg(test)]
    pub(crate) fn bytes(self) -> Result<usize> {
        self.values
            .checked_mul(std::mem::size_of::<f32>())
            .context("FlashMoe GPU matrix byte size overflows usize")
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        FlashMoeStatePlacement::GpuResident
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && self.placement() == FlashMoeStatePlacement::GpuResident
            && self.rows() > 0
            && self.cols() > 0
            && self.values() == self.rows().saturating_mul(self.cols())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashMoeStateBufferDescriptor {
    role: FlashMoeStateBufferRole,
    len: usize,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeStateBufferDescriptor {
    pub(crate) fn new(
        role: FlashMoeStateBufferRole,
        len: usize,
        placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            role,
            len,
            placement,
        }
    }

    #[cfg(test)]
    pub(crate) fn cpu(role: FlashMoeStateBufferRole, len: usize) -> Self {
        Self::new(role, len, FlashMoeStatePlacement::CpuVisible)
    }

    pub(crate) fn gpu(descriptor: FlashMoeGpuBufferDescriptor) -> Self {
        Self::new(
            descriptor.role(),
            descriptor.len(),
            FlashMoeStatePlacement::GpuResident,
        )
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        self.role
    }

    pub(crate) fn len(self) -> usize {
        self.len
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
            && self.len() > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashMoeCmd2InputState {
    layer: usize,
    attention: FlashMoeStateBufferDescriptor,
    residual: FlashMoeStateBufferDescriptor,
}

impl FlashMoeCmd2InputState {
    pub(crate) fn new(
        layer: usize,
        attention_len: usize,
        attention_placement: FlashMoeStatePlacement,
        residual_len: usize,
        residual_placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            layer,
            attention: FlashMoeStateBufferDescriptor::new(
                FlashMoeStateBufferRole::AttentionValues,
                attention_len,
                attention_placement,
            ),
            residual: FlashMoeStateBufferDescriptor::new(
                FlashMoeStateBufferRole::Residual,
                residual_len,
                residual_placement,
            ),
        }
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn attention(self) -> FlashMoeStateBufferDescriptor {
        self.attention
    }

    pub(crate) fn residual(self) -> FlashMoeStateBufferDescriptor {
        self.residual
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.attention.is_declared_graph_state()
            && self.residual.is_declared_graph_state()
            && self.attention.role() == FlashMoeStateBufferRole::AttentionValues
            && self.residual.role() == FlashMoeStateBufferRole::Residual
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashMoeRoutingOutputSource {
    CpuRouterScores,
    FusedMetalPostAttentionPrepCpuTopK,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeRoutingOutputState {
    layer: usize,
    experts: usize,
    active_experts: usize,
    source: FlashMoeRoutingOutputSource,
    role: FlashMoeStateBufferRole,
    len: usize,
}

impl FlashMoeRoutingOutputState {
    pub(crate) fn cpu_router_scores(layer: usize, experts: usize, active_experts: usize) -> Self {
        Self::new(
            layer,
            experts,
            active_experts,
            FlashMoeRoutingOutputSource::CpuRouterScores,
            FlashMoeStateBufferRole::RouterScores,
            experts,
        )
    }

    pub(crate) fn fused_metal_post_attention_cpu_topk(
        layer: usize,
        experts: usize,
        active_experts: usize,
    ) -> Self {
        Self::new(
            layer,
            experts,
            active_experts,
            FlashMoeRoutingOutputSource::FusedMetalPostAttentionPrepCpuTopK,
            FlashMoeStateBufferRole::RoutingTopK,
            active_experts,
        )
    }

    fn new(
        layer: usize,
        experts: usize,
        active_experts: usize,
        source: FlashMoeRoutingOutputSource,
        role: FlashMoeStateBufferRole,
        len: usize,
    ) -> Self {
        Self {
            layer,
            experts,
            active_experts,
            source,
            role,
            len,
        }
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn experts(self) -> usize {
        self.experts
    }

    pub(crate) fn active_experts(self) -> usize {
        self.active_experts
    }

    pub(crate) fn source(self) -> FlashMoeRoutingOutputSource {
        self.source
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        self.role
    }

    pub(crate) fn len(self) -> usize {
        self.len
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        FlashMoeStatePlacement::CpuVisible
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        let expected_role = match self.source {
            FlashMoeRoutingOutputSource::CpuRouterScores => FlashMoeStateBufferRole::RouterScores,
            FlashMoeRoutingOutputSource::FusedMetalPostAttentionPrepCpuTopK => {
                FlashMoeStateBufferRole::RoutingTopK
            }
        };
        self.experts > 0
            && self.active_experts > 0
            && self.active_experts <= self.experts
            && self.role() == expected_role
            && self.len()
                == match expected_role {
                    FlashMoeStateBufferRole::RouterScores => self.experts,
                    FlashMoeStateBufferRole::RoutingTopK => self.active_experts,
                    _ => 0,
                }
            && FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoePostAttentionPrepState {
    residual: FlashMoeGpuBufferDescriptor,
    normed: FlashMoeGpuBufferDescriptor,
    routing: FlashMoeRoutingOutputState,
}

impl FlashMoePostAttentionPrepState {
    pub(crate) fn new(layer: usize, width: usize, experts: usize, active_experts: usize) -> Self {
        Self {
            residual: FlashMoeGpuBufferDescriptor::residual(width),
            normed: FlashMoeGpuBufferDescriptor::normed(width),
            routing: FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(
                layer,
                experts,
                active_experts,
            ),
        }
    }

    pub(crate) fn width(self) -> usize {
        self.residual.len()
    }

    pub(crate) fn residual(self) -> FlashMoeGpuBufferDescriptor {
        self.residual
    }

    pub(crate) fn normed(self) -> FlashMoeGpuBufferDescriptor {
        self.normed
    }

    pub(crate) fn active_experts(self) -> usize {
        self.routing.active_experts()
    }

    pub(crate) fn routing(self) -> FlashMoeRoutingOutputState {
        self.routing
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.residual.is_declared_graph_state()
            && self.normed.is_declared_graph_state()
            && self.residual.role() == FlashMoeStateBufferRole::Residual
            && self.normed.role() == FlashMoeStateBufferRole::Normed
            && self.residual.len() == self.normed.len()
            && self.routing.is_declared_graph_state()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashMoeCmd3InputState {
    layer: usize,
    residual: FlashMoeStateBufferDescriptor,
    normed: FlashMoeStateBufferDescriptor,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeCmd3InputState {
    #[cfg(test)]
    pub(crate) fn cpu_normed_residual(
        layer: usize,
        normed_len: usize,
        residual_len: usize,
    ) -> Self {
        Self {
            layer,
            residual: FlashMoeStateBufferDescriptor::cpu(
                FlashMoeStateBufferRole::Residual,
                residual_len,
            ),
            normed: FlashMoeStateBufferDescriptor::cpu(FlashMoeStateBufferRole::Normed, normed_len),
            placement: FlashMoeStatePlacement::CpuVisible,
        }
    }

    pub(crate) fn metal_post_attention_prep(
        layer: usize,
        prep: FlashMoePostAttentionPrepState,
    ) -> Self {
        Self {
            layer,
            residual: FlashMoeStateBufferDescriptor::gpu(prep.residual()),
            normed: FlashMoeStateBufferDescriptor::gpu(prep.normed()),
            placement: FlashMoeStatePlacement::GpuResident,
        }
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn width(self) -> usize {
        self.residual.len()
    }

    pub(crate) fn residual(self) -> FlashMoeStateBufferDescriptor {
        self.residual
    }

    pub(crate) fn normed(self) -> FlashMoeStateBufferDescriptor {
        self.normed
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.residual.is_declared_graph_state()
            && self.normed.is_declared_graph_state()
            && self.residual.role() == FlashMoeStateBufferRole::Residual
            && self.normed.role() == FlashMoeStateBufferRole::Normed
            && self.residual.len() == self.normed.len()
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeCmd3OutputState {
    hidden: FlashMoeGpuBufferDescriptor,
    next_normed: Option<FlashMoeGpuBufferDescriptor>,
}

impl FlashMoeCmd3OutputState {
    pub(crate) fn gpu_resident(width: usize, next_normed: bool) -> Self {
        Self {
            hidden: FlashMoeGpuBufferDescriptor::hidden(width),
            next_normed: next_normed.then(|| FlashMoeGpuBufferDescriptor::next_layer_normed(width)),
        }
    }

    pub(crate) fn width(self) -> usize {
        self.hidden.len()
    }

    pub(crate) fn hidden(self) -> FlashMoeGpuBufferDescriptor {
        self.hidden
    }

    pub(crate) fn next_normed(self) -> Option<FlashMoeGpuBufferDescriptor> {
        self.next_normed
    }

    pub(crate) fn has_next_normed(self) -> bool {
        self.next_normed.is_some()
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.hidden.is_declared_graph_state()
            && self.hidden.role() == FlashMoeStateBufferRole::Hidden
            && self.next_normed.map_or(true, |next_normed| {
                next_normed.is_declared_graph_state()
                    && next_normed.role() == FlashMoeStateBufferRole::NextLayerNormed
                    && next_normed.len() == self.hidden.len()
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeCmd1InputState {
    layer: usize,
    role: FlashMoeStateBufferRole,
    len: usize,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeCmd1InputState {
    pub(crate) fn new(
        layer: usize,
        role: FlashMoeStateBufferRole,
        len: usize,
        placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            layer,
            role,
            len,
            placement,
        }
    }

    pub(crate) fn cpu_normed(layer: usize, len: usize) -> Self {
        Self::new(
            layer,
            FlashMoeStateBufferRole::Normed,
            len,
            FlashMoeStatePlacement::CpuVisible,
        )
    }

    pub(crate) fn gpu_next_layer_normed(
        layer: usize,
        descriptor: FlashMoeGpuBufferDescriptor,
    ) -> Self {
        Self::new(
            layer,
            descriptor.role(),
            descriptor.len(),
            descriptor.placement(),
        )
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        self.role
    }

    pub(crate) fn len(self) -> usize {
        self.len
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.len() > 0
            && matches!(
                (self.role(), self.placement()),
                (
                    FlashMoeStateBufferRole::Normed,
                    FlashMoeStatePlacement::CpuVisible
                ) | (
                    FlashMoeStateBufferRole::NextLayerNormed,
                    FlashMoeStatePlacement::GpuResident
                )
            )
            && FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

impl Deref for FlashMoeCpuBuffer {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl DerefMut for FlashMoeCpuBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}
