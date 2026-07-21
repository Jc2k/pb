use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeGeneratedTokenRecord {
    position: usize,
    token: u32,
}

impl FlashMoeGeneratedTokenRecord {
    pub(crate) fn new(position: usize, token: u32) -> Self {
        Self { position, token }
    }

    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn token(self) -> u32 {
        self.token
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoePromptTokenRecord {
    position: usize,
    token: u32,
}

impl FlashMoePromptTokenRecord {
    pub(crate) fn new(position: usize, token: u32) -> Self {
        Self { position, token }
    }

    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn token(self) -> u32 {
        self.token
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashMoeFullAttentionKvRecord {
    position: usize,
    layer: usize,
    key: Vec<f32>,
    value: Vec<f32>,
}

impl FlashMoeFullAttentionKvRecord {
    pub(crate) fn new(position: usize, layer: usize, key: Vec<f32>, value: Vec<f32>) -> Self {
        Self {
            position,
            layer,
            key,
            value,
        }
    }

    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn layer(&self) -> usize {
        self.layer
    }

    #[cfg(test)]
    pub(crate) fn key(&self) -> &[f32] {
        &self.key
    }

    #[cfg(test)]
    pub(crate) fn value(&self) -> &[f32] {
        &self.value
    }

    pub(crate) fn state(&self, placement: FlashMoeStatePlacement) -> FlashMoeFullAttentionKvState {
        match placement {
            FlashMoeStatePlacement::CpuVisible => FlashMoeFullAttentionKvState::cpu_visible(
                self.position,
                self.layer,
                self.key.len(),
                self.value.len(),
            ),
            FlashMoeStatePlacement::GpuResident => FlashMoeFullAttentionKvState::gpu_resident(
                self.position,
                self.layer,
                self.key.len(),
                self.value.len(),
            ),
        }
    }

    pub(crate) fn into_key_value(self) -> (Vec<f32>, Vec<f32>) {
        (self.key, self.value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeFullAttentionKvState {
    position: usize,
    layer: usize,
    key_len: usize,
    value_len: usize,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeFullAttentionKvState {
    pub(crate) fn new(
        position: usize,
        layer: usize,
        key_len: usize,
        value_len: usize,
        placement: FlashMoeStatePlacement,
    ) -> Self {
        Self {
            position,
            layer,
            key_len,
            value_len,
            placement,
        }
    }

    pub(crate) fn cpu_visible(
        position: usize,
        layer: usize,
        key_len: usize,
        value_len: usize,
    ) -> Self {
        Self::new(
            position,
            layer,
            key_len,
            value_len,
            FlashMoeStatePlacement::CpuVisible,
        )
    }

    pub(crate) fn gpu_resident(
        position: usize,
        layer: usize,
        key_len: usize,
        value_len: usize,
    ) -> Self {
        Self::new(
            position,
            layer,
            key_len,
            value_len,
            FlashMoeStatePlacement::GpuResident,
        )
    }

    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn width(self) -> usize {
        self.key_len
    }

    pub(crate) fn key_len(self) -> usize {
        self.key_len
    }

    pub(crate) fn value_len(self) -> usize {
        self.value_len
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        FlashMoeStateBufferRole::Kv
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.width() > 0
            && self.key_len() == self.value_len()
            && FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashMoeMlaKvState {
    position: usize,
    layer: usize,
    latent_len: usize,
    rotary_len: usize,
    placement: FlashMoeStatePlacement,
}

impl FlashMoeMlaKvState {
    pub(crate) fn cpu_visible(
        position: usize,
        layer: usize,
        latent_len: usize,
        rotary_len: usize,
    ) -> Self {
        Self {
            position,
            layer,
            latent_len,
            rotary_len,
            placement: FlashMoeStatePlacement::CpuVisible,
        }
    }

    pub(crate) fn position(self) -> usize {
        self.position
    }

    pub(crate) fn layer(self) -> usize {
        self.layer
    }

    pub(crate) fn latent_len(self) -> usize {
        self.latent_len
    }

    pub(crate) fn rotary_len(self) -> usize {
        self.rotary_len
    }

    pub(crate) fn role(self) -> FlashMoeStateBufferRole {
        FlashMoeStateBufferRole::Kv
    }

    pub(crate) fn placement(self) -> FlashMoeStatePlacement {
        self.placement
    }

    pub(crate) fn is_declared_graph_state(self) -> bool {
        self.latent_len() > 0
            && self.rotary_len() > 0
            && FlashMoeStateBufferRole::GENERATION_ROLES.contains(&self.role())
            && FlashMoeStatePlacement::GRAPH_PLACEMENTS.contains(&self.placement())
    }
}

pub(in crate::inference::flashmoe) type KvEntry = (Arc<[f32]>, Arc<[f32]>);
pub(in crate::inference::flashmoe) type MlaKvEntry = (Arc<[f32]>, Arc<[f32]>);

#[derive(Debug, Clone)]
pub(in crate::inference::flashmoe) struct KvCache {
    pub(in crate::inference::flashmoe) layers: usize,
    pub(in crate::inference::flashmoe) capacity: usize,
    prompt_tokens: Vec<(usize, u32)>,
    generated_tokens: Vec<(usize, u32)>,
    pub(super) layer_states: Vec<(usize, usize, u64)>,
    pub(in crate::inference::flashmoe) kv: Vec<Vec<Option<KvEntry>>>,
    pub(in crate::inference::flashmoe) mla_kv: Vec<Vec<Option<MlaKvEntry>>>,
}

impl KvCache {
    pub(crate) fn new(layers: usize, capacity: usize) -> Self {
        Self {
            layers,
            capacity,
            prompt_tokens: Vec::new(),
            generated_tokens: Vec::new(),
            layer_states: Vec::new(),
            kv: vec![vec![None; capacity]; layers],
            mla_kv: vec![vec![None; capacity]; layers],
        }
    }

    pub(crate) fn shallow_snapshot(&self) -> Self {
        self.clone()
    }

    pub(crate) fn record_prompt_token(&mut self, position: usize, token: u32) -> Result<()> {
        self.ensure_position(position)?;
        self.prompt_tokens.push((position, token));
        Ok(())
    }

    pub(crate) fn record_prompt_token_record(
        &mut self,
        record: FlashMoePromptTokenRecord,
    ) -> Result<()> {
        self.record_prompt_token(record.position(), record.token())
    }

    pub(crate) fn resize_capacity(&mut self, capacity: usize) {
        if capacity <= self.capacity {
            return;
        }
        for layer in &mut self.kv {
            layer.resize_with(capacity, || None);
        }
        for layer in &mut self.mla_kv {
            layer.resize_with(capacity, || None);
        }
        self.capacity = capacity;
    }

    pub(crate) fn record_generated_token(&mut self, position: usize, token: u32) -> Result<()> {
        self.ensure_position(position)?;
        self.generated_tokens.push((position, token));
        Ok(())
    }

    pub(in crate::inference::flashmoe) fn record_generated_token_record(
        &mut self,
        record: FlashMoeGeneratedTokenRecord,
    ) -> Result<()> {
        self.record_generated_token(record.position(), record.token())
    }

    pub(crate) fn record_layer_state(
        &mut self,
        position: usize,
        layer: usize,
        state: u64,
    ) -> Result<()> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        self.layer_states.push((position, layer, state));
        Ok(())
    }

    /// Record one contiguous layer-major recurrent trace after validating the
    /// shared graph coordinates once. The scalar path retains the typed
    /// single-record API; a prefill matrix would otherwise repeat identical
    /// placement, layer, and capacity checks for every row in the layer.
    pub(crate) fn record_layer_state_values(
        &mut self,
        start_position: usize,
        layer: usize,
        states: impl ExactSizeIterator<Item = u64>,
    ) -> Result<()> {
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        let rows = states.len();
        let end_position = start_position
            .checked_add(rows)
            .context("layer-major recurrent trace position overflow")?;
        if end_position > self.capacity {
            bail!(
                "KV cache layer-major range {start_position}..{end_position} exceeds capacity {}",
                self.capacity
            );
        }
        self.layer_states.reserve(rows);
        self.layer_states.extend(
            states
                .enumerate()
                .map(|(row, value)| (start_position + row, layer, value)),
        );
        Ok(())
    }

    pub(in crate::inference::flashmoe) fn record_layer_state_record(
        &mut self,
        record: FlashMoeLayerStateRecord,
    ) -> Result<()> {
        self.record_recurrent_layer_state(record.state(FlashMoeStatePlacement::CpuVisible))
    }

    pub(crate) fn record_recurrent_layer_state(
        &mut self,
        state: FlashMoeRecurrentLayerState,
    ) -> Result<()> {
        if !state.is_declared_graph_state() {
            bail!("FlashMoe recurrent layer state is not declared graph state");
        }
        if state.placement() != FlashMoeStatePlacement::CpuVisible {
            bail!(
                "FlashMoe recurrent layer state recording requires CpuVisible placement, got {:?}",
                state.placement()
            );
        }
        self.record_layer_state(state.position(), state.layer(), state.value())
    }

    pub(crate) fn record_kv(
        &mut self,
        position: usize,
        layer: usize,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<()> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        self.kv[layer][position] = Some((Arc::from(key), Arc::from(value)));
        Ok(())
    }

    pub(in crate::inference::flashmoe) fn record_kv_record(
        &mut self,
        record: FlashMoeFullAttentionKvRecord,
    ) -> Result<()> {
        let position = record.position();
        let layer = record.layer();
        let (key, value) = record.into_key_value();
        self.record_kv(position, layer, key, value)
    }

    pub(in crate::inference::flashmoe) fn record_mla_kv(
        &mut self,
        position: usize,
        layer: usize,
        latent: Vec<f32>,
        rotary_key: Vec<f32>,
    ) -> Result<()> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!(
                "MLA KV cache layer {layer} exceeds layer count {}",
                self.layers
            );
        }
        self.mla_kv[layer][position] = Some((Arc::from(latent), Arc::from(rotary_key)));
        Ok(())
    }

    pub(in crate::inference::flashmoe) fn mla_records(
        &self,
        position: usize,
        layer: usize,
    ) -> Result<Vec<(&[f32], &[f32])>> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!(
                "MLA KV cache layer {layer} exceeds layer count {}",
                self.layers
            );
        }
        Ok(self.mla_kv[layer]
            .iter()
            .take(position + 1)
            .filter_map(|entry| {
                entry
                    .as_ref()
                    .map(|(latent, rotary)| (&latent[..], &rotary[..]))
            })
            .collect())
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    pub(in crate::inference::flashmoe) fn causal_attention(
        &self,
        position: usize,
        layer: usize,
        query: &[f32],
        num_q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        let keys_values: Vec<(&[f32], &[f32])> = self.kv[layer]
            .iter()
            .take(position + 1)
            .filter_map(|entry| entry.as_ref().map(|(key, value)| (&key[..], &value[..])))
            .collect();
        Ok(causal_attention(
            query,
            &keys_values,
            num_q_heads,
            kv_heads,
            head_dim,
        ))
    }

    #[allow(dead_code)]
    pub(crate) fn keys_values(
        &self,
        position: usize,
        layer: usize,
    ) -> Result<Vec<(&[f32], &[f32])>> {
        self.ensure_position(position)?;
        if layer >= self.layers {
            bail!("KV cache layer {layer} exceeds layer count {}", self.layers);
        }
        Ok(self.kv[layer]
            .iter()
            .take(position + 1)
            .filter_map(|entry| entry.as_ref().map(|(key, value)| (&key[..], &value[..])))
            .collect())
    }

    pub(crate) fn ensure_position(&self, position: usize) -> Result<()> {
        if position >= self.capacity {
            bail!(
                "KV cache position {position} exceeds capacity {}",
                self.capacity
            );
        }
        Ok(())
    }

    pub(crate) fn prefill_state_sha256(&self) -> (String, String) {
        let mut kv_digest = Sha256::new();
        kv_digest.update(b"pb.flashmoe.full-attention-kv.v1\0");
        kv_digest.update((self.layers as u64).to_le_bytes());
        for (layer, entries) in self.kv.iter().enumerate() {
            for (position, entry) in entries.iter().enumerate() {
                let Some((key, value)) = entry else {
                    continue;
                };
                kv_digest.update((layer as u64).to_le_bytes());
                kv_digest.update((position as u64).to_le_bytes());
                update_f32_digest(&mut kv_digest, key);
                update_f32_digest(&mut kv_digest, value);
            }
        }

        // Token-major and layer-major execution record the same states in a
        // different traversal order. Canonicalize by graph coordinates so the
        // digest measures state, not scheduling order.
        let mut layer_states = self.layer_states.clone();
        layer_states.sort_unstable_by_key(|(position, layer, _)| (*position, *layer));
        let mut trace_digest = Sha256::new();
        trace_digest.update(b"pb.flashmoe.router-recurrent-trace.v1\0");
        trace_digest.update((layer_states.len() as u64).to_le_bytes());
        for (position, layer, value) in layer_states {
            trace_digest.update((position as u64).to_le_bytes());
            trace_digest.update((layer as u64).to_le_bytes());
            trace_digest.update(value.to_le_bytes());
        }

        (
            format!("{:x}", kv_digest.finalize()),
            format!("{:x}", trace_digest.finalize()),
        )
    }

    pub(crate) fn prefill_layer_state_sha256(&self) -> (Vec<Option<String>>, Vec<Option<String>>) {
        let kv = self
            .kv
            .iter()
            .enumerate()
            .map(|(layer, entries)| {
                let present = entries.iter().any(Option::is_some);
                present.then(|| {
                    let mut digest = Sha256::new();
                    digest.update(b"pb.flashmoe.full-attention-kv-layer.v1\0");
                    digest.update((layer as u64).to_le_bytes());
                    for (position, entry) in entries.iter().enumerate() {
                        let Some((key, value)) = entry else {
                            continue;
                        };
                        digest.update((position as u64).to_le_bytes());
                        update_f32_digest(&mut digest, key);
                        update_f32_digest(&mut digest, value);
                    }
                    format!("{:x}", digest.finalize())
                })
            })
            .collect();
        let trace = (0..self.layers)
            .map(|layer| {
                let mut states = self
                    .layer_states
                    .iter()
                    .filter(|(_, state_layer, _)| *state_layer == layer)
                    .copied()
                    .collect::<Vec<_>>();
                if states.is_empty() {
                    return None;
                }
                states.sort_unstable_by_key(|(position, _, _)| *position);
                let mut digest = Sha256::new();
                digest.update(b"pb.flashmoe.router-recurrent-layer.v1\0");
                digest.update((layer as u64).to_le_bytes());
                for (position, _, value) in states {
                    digest.update((position as u64).to_le_bytes());
                    digest.update(value.to_le_bytes());
                }
                Some(format!("{:x}", digest.finalize()))
            })
            .collect();
        (kv, trace)
    }
}
