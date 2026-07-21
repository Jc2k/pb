use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeRecurrentState {
    value: u64,
}

impl FlashMoeRecurrentState {
    pub(crate) fn new(value: u64) -> Self {
        Self { value }
    }

    pub(crate) fn mix_active_expert(&mut self, expert_hash: u64, weight: f32) {
        self.value = self
            .value
            .wrapping_add(expert_hash.wrapping_mul((weight.to_bits() as u64).max(1)));
    }

    pub(crate) fn value(self) -> u64 {
        self.value
    }

    pub(crate) fn layer_state_record(
        self,
        position: usize,
        layer: usize,
    ) -> FlashMoeLayerStateRecord {
        FlashMoeLayerStateRecord {
            position,
            layer,
            recurrent_value: self.value(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeTokenState {
    hidden: FlashMoeCpuBuffer,
    next_layer_normed: Option<FlashMoeCpuBuffer>,
    recurrent: FlashMoeRecurrentState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashMoeExpertPhaseApplication {
    HiddenAndNextNormed,
    HiddenOnly,
}

impl FlashMoeTokenState {
    pub(crate) fn new(hidden_values: Vec<f32>, recurrent_seed: u64) -> Self {
        Self {
            hidden: FlashMoeCpuBuffer::hidden(hidden_values),
            next_layer_normed: None,
            recurrent: FlashMoeRecurrentState::new(recurrent_seed),
        }
    }

    pub(crate) fn hidden(&self) -> &FlashMoeCpuBuffer {
        &self.hidden
    }

    pub(crate) fn hidden_mut(&mut self) -> &mut FlashMoeCpuBuffer {
        &mut self.hidden
    }

    pub(crate) fn replace_hidden(&mut self, values: Vec<f32>) {
        self.hidden.replace_values(values);
    }

    #[cfg(test)]
    pub(crate) fn residual_snapshot(&self) -> FlashMoeCpuBuffer {
        FlashMoeCpuBuffer::residual(self.hidden.clone_values())
    }

    pub(crate) fn set_next_layer_normed(&mut self, values: Option<Vec<f32>>) {
        self.next_layer_normed = values.map(FlashMoeCpuBuffer::next_layer_normed);
    }

    pub(crate) fn clear_next_layer_normed(&mut self) {
        self.next_layer_normed = None;
    }

    pub(crate) fn apply_expert_phase_output(&mut self, output: FlashMoeExpertPhaseOutput) {
        let (hidden, next_normed) = output.into_hidden_and_next_normed();
        self.replace_hidden(hidden);
        self.set_next_layer_normed(next_normed);
    }

    pub(crate) fn apply_expert_phase_hidden_only(&mut self, output: FlashMoeExpertPhaseOutput) {
        let (hidden, _) = output.into_hidden_and_next_normed();
        self.replace_hidden(hidden);
        self.clear_next_layer_normed();
    }

    pub(crate) fn apply_declared_expert_phase(
        &mut self,
        output: FlashMoeExpertPhaseOutput,
        application: FlashMoeExpertPhaseApplication,
    ) -> Result<()> {
        if output.declared_cmd3_output().is_none() {
            bail!("FlashMoe token state refused undeclared expert phase output");
        }
        match application {
            FlashMoeExpertPhaseApplication::HiddenAndNextNormed => {
                self.apply_expert_phase_output(output)
            }
            FlashMoeExpertPhaseApplication::HiddenOnly => {
                self.apply_expert_phase_hidden_only(output)
            }
        }
        Ok(())
    }

    pub(crate) fn take_next_layer_normed_as_normed(&mut self) -> Option<FlashMoeCpuBuffer> {
        self.next_layer_normed
            .take()
            .map(|buffer| buffer.into_role(FlashMoeStateBufferRole::Normed))
    }

    pub(crate) fn mix_active_expert(&mut self, expert_hash: u64, weight: f32) {
        self.recurrent.mix_active_expert(expert_hash, weight);
    }

    #[cfg(test)]
    pub(crate) fn recurrent_value(&self) -> u64 {
        self.recurrent.value()
    }

    pub(crate) fn layer_state_record(
        &self,
        position: usize,
        layer: usize,
    ) -> FlashMoeLayerStateRecord {
        self.recurrent.layer_state_record(position, layer)
    }

    #[cfg(test)]
    pub(crate) fn into_hidden_values(self) -> Vec<f32> {
        self.hidden.into_values()
    }

    pub(crate) fn into_hidden_and_recurrent(self) -> (Vec<f32>, u64) {
        (self.hidden.into_values(), self.recurrent.value())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeLayerStateRecord {
    pub(super) position: usize,
    pub(super) layer: usize,
    pub(super) recurrent_value: u64,
}

impl FlashMoeLayerStateRecord {
    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn recurrent_value(self) -> u64 {
        self.recurrent_value
    }

    pub(crate) fn state(self, placement: FlashMoeStatePlacement) -> FlashMoeRecurrentLayerState {
        match placement {
            FlashMoeStatePlacement::CpuVisible => FlashMoeRecurrentLayerState::cpu_visible(
                self.position(),
                self.layer(),
                self.recurrent_value(),
            ),
            FlashMoeStatePlacement::GpuResident => FlashMoeRecurrentLayerState::new(
                self.position(),
                self.layer(),
                self.recurrent_value(),
                placement,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeRecurrentLayerState {
    position: usize,
    layer: usize,
    value: u64,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeRecurrentLayerState {
    pub(crate) fn new(
        position: usize,
        layer: usize,
        value: u64,
        placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            position,
            layer,
            value,
            placement,
        }
    }

    pub(crate) fn cpu_visible(position: usize, layer: usize, value: u64) -> Self {
        Self::new(position, layer, value, FlashMoeStatePlacement::CpuVisible)
    }

    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn value(self) -> u64 {
        self.value
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        FlashMoeStateBufferRole::Recurrent
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeLinearAttentionCacheState {
    layer: usize,
    conv_state_len: usize,
    ssm_state_len: usize,
    conv_output_len: usize,
    output_len: usize,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeLinearAttentionCacheState {
    pub(crate) fn new(
        layer: usize,
        conv_state_len: usize,
        ssm_state_len: usize,
        conv_output_len: usize,
        output_len: usize,
        placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            layer,
            conv_state_len,
            ssm_state_len,
            conv_output_len,
            output_len,
            placement,
        }
    }

    pub(crate) fn cpu_visible(
        layer: usize,
        conv_state_len: usize,
        ssm_state_len: usize,
        conv_output_len: usize,
        output_len: usize,
    ) -> Self {
        Self::new(
            layer,
            conv_state_len,
            ssm_state_len,
            conv_output_len,
            output_len,
            FlashMoeStatePlacement::CpuVisible,
        )
    }

    pub(crate) fn gpu_resident(
        layer: usize,
        conv_state_len: usize,
        ssm_state_len: usize,
        conv_output_len: usize,
        output_len: usize,
    ) -> Self {
        Self::new(
            layer,
            conv_state_len,
            ssm_state_len,
            conv_output_len,
            output_len,
            FlashMoeStatePlacement::GpuResident,
        )
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn conv_state_len(self) -> usize {
        self.conv_state_len
    }

    pub(crate) fn ssm_state_len(self) -> usize {
        self.ssm_state_len
    }

    pub(crate) fn conv_output_len(self) -> usize {
        self.conv_output_len
    }

    pub(crate) fn output_len(self) -> usize {
        self.output_len
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        FlashMoeStateBufferRole::Recurrent
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.conv_state_len() > 0
            && self.ssm_state_len() > 0
            && self.conv_output_len() > 0
            && self.output_len() > 0
            && FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeLinearAttentionCacheShape {
    pub(crate) conv_state_len: usize,
    pub(crate) ssm_state_len: usize,
    pub(crate) conv_output_len: usize,
    pub(crate) output_len: usize,
    pub(crate) value_scratch_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LinearAttentionLayout {
    pub(crate) num_value_heads: usize,
    pub(crate) num_key_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) value_dim: usize,
    pub(crate) total_key_width: usize,
    pub(crate) total_value_width: usize,
    pub(crate) conv_dim: usize,
    pub(crate) conv_kernel_size: usize,
}

impl LinearAttentionLayout {
    pub(crate) fn conv_state_len(self) -> usize {
        self.conv_kernel_size.saturating_sub(1) * self.conv_dim
    }

    pub(crate) fn ssm_state_len(self) -> usize {
        self.num_value_heads * self.value_dim * self.key_dim
    }

    pub(crate) fn value_heads_per_key_head(self) -> usize {
        (self.num_value_heads / self.num_key_heads).max(1)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeLinearAttentionLayerSnapshot {
    state: FlashMoeLinearAttentionCacheState,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

impl FlashMoeLinearAttentionLayerSnapshot {
    pub(crate) fn new(
        layer: usize,
        conv_state: Vec<f32>,
        ssm_state: Vec<f32>,
        conv_output_len: usize,
        output_len: usize,
    ) -> Result<Self> {
        let state = FlashMoeLinearAttentionCacheState::cpu_visible(
            layer,
            conv_state.len(),
            ssm_state.len(),
            conv_output_len,
            output_len,
        );
        if !state.is_declared_graph_state() {
            bail!(
                "FlashMoe linear-attention session snapshot for layer {layer} is not declared CPU-visible graph state"
            );
        }
        Ok(Self {
            state,
            conv_state,
            ssm_state,
        })
    }

    pub(crate) fn state(&self) -> FlashMoeLinearAttentionCacheState {
        self.state
    }

    pub(crate) fn conv_state(&self) -> &[f32] {
        &self.conv_state
    }

    pub(crate) fn ssm_state(&self) -> &[f32] {
        &self.ssm_state
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeLinearAttentionSessionSnapshot {
    layers: Box<[Option<FlashMoeLinearAttentionLayerSnapshot>]>,
}

impl FlashMoeLinearAttentionSessionSnapshot {
    pub(crate) fn new(layers: Vec<Option<FlashMoeLinearAttentionLayerSnapshot>>) -> Result<Self> {
        if layers.is_empty() {
            bail!("FlashMoe linear-attention session snapshot requires resolved model layers");
        }
        for (layer, snapshot) in layers.iter().enumerate() {
            if let Some(snapshot) = snapshot
                && (snapshot.state().layer() != layer
                    || snapshot.state().placement() != FlashMoeStatePlacement::CpuVisible
                    || !snapshot.state().is_declared_graph_state())
            {
                bail!(
                    "FlashMoe linear-attention session snapshot layer {layer} does not match its declared CPU-visible state"
                );
            }
        }
        Ok(Self {
            layers: layers.into_boxed_slice(),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn layer(&self, layer: usize) -> Option<&FlashMoeLinearAttentionLayerSnapshot> {
        self.layers.get(layer).and_then(Option::as_ref)
    }

    pub(crate) fn state_sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"pb.flashmoe.linear-attention-state.v1\0");
        digest.update((self.layers.len() as u64).to_le_bytes());
        for (layer, snapshot) in self.layers.iter().enumerate() {
            digest.update((layer as u64).to_le_bytes());
            match snapshot {
                Some(snapshot) => {
                    digest.update([1]);
                    let state = snapshot.state();
                    digest.update((state.layer() as u64).to_le_bytes());
                    digest.update((state.conv_state_len() as u64).to_le_bytes());
                    digest.update((state.ssm_state_len() as u64).to_le_bytes());
                    update_f32_digest(&mut digest, snapshot.conv_state());
                    update_f32_digest(&mut digest, snapshot.ssm_state());
                }
                None => digest.update([0]),
            }
        }
        format!("{:x}", digest.finalize())
    }

    pub(crate) fn layer_state_sha256(&self) -> Vec<Option<String>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer, snapshot)| {
                snapshot.as_ref().map(|snapshot| {
                    let mut digest = Sha256::new();
                    digest.update(b"pb.flashmoe.linear-attention-layer.v1\0");
                    digest.update((layer as u64).to_le_bytes());
                    update_f32_digest(&mut digest, snapshot.conv_state());
                    update_f32_digest(&mut digest, snapshot.ssm_state());
                    format!("{:x}", digest.finalize())
                })
            })
            .collect()
    }
}

pub(crate) fn update_f32_digest(digest: &mut Sha256, values: &[f32]) {
    digest.update((values.len() as u64).to_le_bytes());
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
}

#[cfg(test)]
impl FlashMoeLinearAttentionCacheShape {
    pub(crate) fn new(
        conv_state_len: usize,
        ssm_state_len: usize,
        conv_output_len: usize,
        output_len: usize,
        value_scratch_len: usize,
    ) -> Self {
        Self {
            conv_state_len,
            ssm_state_len,
            conv_output_len,
            output_len,
            value_scratch_len,
        }
    }

    pub(crate) fn state(
        self,
        layer: usize,
        placement: FlashMoeStatePlacement,
    ) -> FlashMoeLinearAttentionCacheState {
        FlashMoeLinearAttentionCacheState::new(
            layer,
            self.conv_state_len,
            self.ssm_state_len,
            self.conv_output_len,
            self.output_len,
            placement,
        )
    }

    pub(crate) fn is_declared_graph_shape(self) -> bool {
        self.conv_state_len > 0
            && self.ssm_state_len > 0
            && self.conv_output_len > 0
            && self.output_len > 0
            && self.value_scratch_len > 0
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct FlashMoeLinearAttentionStateData {
    pub(crate) conv_state: Vec<f32>,
    pub(crate) ssm_state: Vec<f32>,
    pub(crate) conv_out: Vec<f32>,
    pub(crate) out_values: Vec<f32>,
    pub(crate) kv_mem: Vec<f32>,
    pub(crate) delta: Vec<f32>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct FlashMoeLinearAttentionState {
    inner: Arc<FlashMoeLinearAttentionStateData>,
}

#[cfg(test)]
impl FlashMoeLinearAttentionState {
    pub(crate) fn new(shape: FlashMoeLinearAttentionCacheShape) -> Self {
        debug_assert!(shape.is_declared_graph_shape());
        Self {
            inner: Arc::new(FlashMoeLinearAttentionStateData {
                conv_state: vec![0.0; shape.conv_state_len],
                ssm_state: vec![0.0; shape.ssm_state_len],
                conv_out: vec![0.0; shape.conv_output_len],
                out_values: vec![0.0; shape.output_len],
                kv_mem: vec![0.0; shape.value_scratch_len],
                delta: vec![0.0; shape.value_scratch_len],
            }),
        }
    }

    pub(crate) fn expected_state(
        layer: usize,
        shape: FlashMoeLinearAttentionCacheShape,
        placement: FlashMoeStatePlacement,
    ) -> FlashMoeLinearAttentionCacheState {
        match placement {
            FlashMoeStatePlacement::CpuVisible => FlashMoeLinearAttentionCacheState::cpu_visible(
                layer,
                shape.conv_state_len,
                shape.ssm_state_len,
                shape.conv_output_len,
                shape.output_len,
            ),
            FlashMoeStatePlacement::GpuResident => FlashMoeLinearAttentionCacheState::gpu_resident(
                layer,
                shape.conv_state_len,
                shape.ssm_state_len,
                shape.conv_output_len,
                shape.output_len,
            ),
        }
    }

    pub(crate) fn state(
        &self,
        layer: usize,
        placement: FlashMoeStatePlacement,
    ) -> FlashMoeLinearAttentionCacheState {
        FlashMoeLinearAttentionCacheShape::new(
            self.conv_state.len(),
            self.ssm_state.len(),
            self.conv_out.len(),
            self.out_values.len(),
            self.kv_mem.len(),
        )
        .state(layer, placement)
    }

    pub(crate) fn matches_shape(&self, shape: FlashMoeLinearAttentionCacheShape) -> bool {
        self.conv_state.len() == shape.conv_state_len
            && self.ssm_state.len() == shape.ssm_state_len
            && self.conv_out.len() == shape.conv_output_len
            && self.out_values.len() == shape.output_len
            && self.kv_mem.len() == shape.value_scratch_len
            && self.delta.len() == shape.value_scratch_len
    }
}

#[cfg(test)]
impl Deref for FlashMoeLinearAttentionState {
    type Target = FlashMoeLinearAttentionStateData;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(test)]
impl DerefMut for FlashMoeLinearAttentionState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.inner)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeExpertPhaseOutput {
    hidden: Vec<f32>,
    next_normed: Option<Vec<f32>>,
    declared_cmd3_output: Option<FlashMoeCmd3OutputState>,
}

impl FlashMoeExpertPhaseOutput {
    pub(crate) fn new(hidden: Vec<f32>, next_normed: Option<Vec<f32>>) -> Self {
        Self {
            hidden,
            next_normed,
            declared_cmd3_output: None,
        }
    }

    pub(crate) fn hidden_len(&self) -> usize {
        self.hidden.len()
    }

    pub(crate) fn next_normed_len(&self) -> Option<usize> {
        self.next_normed.as_ref().map(Vec::len)
    }

    pub(crate) fn with_declared_cmd3_output(mut self, state: FlashMoeCmd3OutputState) -> Self {
        self.declared_cmd3_output = Some(state);
        self
    }

    pub(crate) fn declared_cmd3_output(&self) -> Option<FlashMoeCmd3OutputState> {
        self.declared_cmd3_output
    }

    pub(crate) fn into_hidden_and_next_normed(self) -> (Vec<f32>, Option<Vec<f32>>) {
        (self.hidden, self.next_normed)
    }
}
