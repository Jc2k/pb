use std::collections::BTreeSet;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::collections::{BTreeMap, HashMap};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::ffi::{CStr, CString, c_char, c_void};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::ptr;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::{Arc, Mutex};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::thread;
use std::time::{Duration, Instant};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod deepseek_execution;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) use deepseek_execution::DeepSeekV4SessionSnapshot;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[derive(Debug)]
pub(super) struct DeepSeekV4SessionSnapshot;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::deepseek_metal::DEEPSEEK_V4_METAL_SHADERS;
use super::deepseek_metal::DEEPSEEK_V4_REQUIRED_METAL_KERNELS;
use super::state::{
    FlashMoeExpertPhaseOutput, FlashMoeGpuBufferDescriptor, FlashMoeGpuMatrixDescriptor,
    FlashMoeStateBufferRole,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::types::FlashMoeMetalResourceSnapshot;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::experts::{
    EXPERT_SCALE_BIAS_DTYPE_BF16, EXPERT_SCALE_BIAS_DTYPE_F32, ReusableExpertBytes,
    expert_scale_bias_dtype_size,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::math::{routing_sigmoid_noaux_top_k, routing_softmax_top_k};
#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
use super::scheduler::ScheduledQ4ExpertPhaseMlpPayload;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::scheduler::{
    ScheduledCmd3ExpertPayload, ScheduledCmd3MetalPostAttentionInput, ScheduledCmd3OutputState,
    ScheduledDenseExpertPhaseMlpPayload, ScheduledExpertPhaseMlpPayload, ScheduledExpertSlot,
    ScheduledLayerMajorExperts, ScheduledRoutingCandidateSource, ScheduledRoutingCommand,
    ScheduledSharedExpertPhaseRef,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::state::{
    FlashMoeCmd3OutputState, FlashMoeLinearAttentionCacheState,
    FlashMoeLinearAttentionLayerSnapshot, FlashMoeLinearAttentionSessionSnapshot,
    FlashMoePostAttentionPrepState, LinearAttentionLayout,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::weights::{
    Cmd2ResidentPostAttentionPrepProjections, DenseMmapMatvecProjection,
    DenseQ4MmapMatvecProjection, LinearAttentionResidentBindings, ResidentMmapMatvecProjection,
    ResidentStaticDtype, SharedExpertPhaseResidentProjections, SharedExpertPhaseWeights,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use anyhow::{Context as _, Result, bail};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) type MetalObjcId = *mut c_void;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalStateBuffer {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    buffer: MetalObjcId,
    state: FlashMoeGpuBufferDescriptor,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalMatrixBuffer {
    buffer: MetalObjcId,
    state: FlashMoeGpuMatrixDescriptor,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalMatrixBuffer {
    fn new(buffer: MetalObjcId, state: FlashMoeGpuMatrixDescriptor) -> anyhow::Result<Self> {
        if buffer.is_null() {
            anyhow::bail!("FlashMoe Metal matrix buffer requires a non-null buffer");
        }
        if !state.is_declared_graph_state() {
            anyhow::bail!("FlashMoe Metal matrix buffer requires declared GPU matrix state");
        }
        Ok(Self { buffer, state })
    }

    pub(crate) fn buffer(self) -> MetalObjcId {
        self.buffer
    }

    pub(crate) fn state(self) -> FlashMoeGpuMatrixDescriptor {
        self.state
    }

    pub(crate) fn rows(self) -> usize {
        self.state.rows()
    }

    pub(crate) fn cols(self) -> usize {
        self.state.cols()
    }

    pub(crate) fn values(self) -> usize {
        self.state.values()
    }
}

impl MetalStateBuffer {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn new(buffer: MetalObjcId, state: FlashMoeGpuBufferDescriptor) -> anyhow::Result<Self> {
        if buffer.is_null() {
            anyhow::bail!("FlashMoe Metal state buffer requires a non-null buffer");
        }
        if !state.is_declared_graph_state()
            || state.placement() != super::state::FlashMoeStatePlacement::GpuResident
        {
            anyhow::bail!("FlashMoe Metal state buffer requires declared GpuResident state");
        }
        Ok(Self { buffer, state })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn buffer(self) -> MetalObjcId {
        self.buffer
    }

    pub(crate) fn len(self) -> usize {
        self.state.len()
    }

    pub(crate) fn state(self) -> FlashMoeGpuBufferDescriptor {
        self.state
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) type MetalSelector = *mut c_void;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MetalMatvecTiming {
    pub(crate) buffer_upload: Duration,
    pub(crate) dispatch: Duration,
    pub(crate) readback: Duration,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) enum MetalBatchProjectionInput<'a> {
    Cpu(&'a [f32]),
    Buffer { buffer: MetalObjcId, len: usize },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalGlmMlaAbsorbedAttentionInput<'a> {
    pub(crate) heads: usize,
    pub(crate) latent_rank: usize,
    pub(crate) query_nope: &'a [f32],
    pub(crate) query_rope: &'a [f32],
    pub(crate) record_latents: &'a [f32],
    pub(crate) record_rotary: &'a [f32],
    pub(crate) sequence: usize,
    pub(crate) rope_dim: usize,
    pub(crate) scale: f32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalGlmMlaFusedAttentionInput<'a> {
    pub(crate) input: MetalBatchProjectionInput<'a>,
    pub(crate) heads: usize,
    pub(crate) latent_rank: usize,
    pub(crate) nope_dim: usize,
    pub(crate) rope_dim: usize,
    pub(crate) previous_record_latents: &'a [f32],
    pub(crate) previous_record_rotary: &'a [f32],
    pub(crate) rope_cos: &'a [f32],
    pub(crate) rope_sin: &'a [f32],
    pub(crate) scale: f32,
    pub(crate) post_attention: Option<MetalGlmMlaPostAttentionInput<'a>>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalGlmMlaPostAttentionInput<'a> {
    pub(crate) projections: &'a Cmd2ResidentPostAttentionPrepProjections,
    pub(crate) residual: MetalBatchProjectionInput<'a>,
    pub(crate) post_norm_weight: &'a [f32],
    pub(crate) router_correction_bias: Option<&'a [f32]>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) enum MetalGlmMlaFusedAttentionTerminal {
    Attention(Vec<f32>),
    PostAttention(MetalPostAttentionPrep),
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalGlmMlaFusedAttentionOutput {
    pub(crate) terminal: MetalGlmMlaFusedAttentionTerminal,
    pub(crate) latent: Vec<f32>,
    pub(crate) rotary: Vec<f32>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalBatchProjectionInput<'_> {
    pub(crate) fn len(self) -> usize {
        match self {
            Self::Cpu(input) => input.len(),
            Self::Buffer { len, .. } => len,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalExpertSourceBufferKey {
    ptr: usize,
    len: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
struct MetalExpertSourceBufferEntry {
    key: MetalExpertSourceBufferKey,
    buffer: MetalObjcId,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
struct MetalPersistentExpertBuffer {
    device: usize,
    buffer: usize,
    resources: Arc<MetalResourceLedger>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalPersistentExpertBuffer {
    fn new(
        device: MetalObjcId,
        buffer: MetalObjcId,
        len: usize,
        resources: Arc<MetalResourceLedger>,
    ) -> Self {
        resources.register_buffer(buffer, len, MetalTrackedBufferClass::ResidentExpertWrapper);
        Self {
            device: device as usize,
            buffer: buffer as usize,
            resources,
        }
    }

    fn buffer_for_device(&self, device: MetalObjcId) -> Option<MetalObjcId> {
        (self.device == device as usize).then_some(self.buffer as MetalObjcId)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalPersistentExpertBuffer {
    fn drop(&mut self) {
        unsafe {
            self.resources.release_buffer(self.buffer as MetalObjcId);
            release(self.buffer as MetalObjcId);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Default)]
pub(crate) struct MetalExpertSourceBufferCache {
    entries: Vec<MetalExpertSourceBufferEntry>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalExpertSourceBufferCache {
    fn key_for(bytes: &[u8]) -> MetalExpertSourceBufferKey {
        MetalExpertSourceBufferKey {
            ptr: bytes.as_ptr() as usize,
            len: bytes.len(),
        }
    }

    pub(crate) fn get(&self, bytes: &[u8]) -> Option<MetalObjcId> {
        let key = Self::key_for(bytes);
        self.entries
            .iter()
            .find_map(|entry| (entry.key == key).then_some(entry.buffer))
    }

    pub(crate) fn insert(&mut self, bytes: &[u8], buffer: MetalObjcId) {
        let key = Self::key_for(bytes);
        if self.entries.iter().any(|entry| entry.key == key) {
            return;
        }
        self.entries
            .push(MetalExpertSourceBufferEntry { key, buffer });
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalLinearAttentionStateCache {
    pub(crate) layers: Vec<Option<MetalLinearAttentionLayerState>>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalLinearAttentionStateCache {
    pub(crate) fn new(layers: Vec<Option<MetalLinearAttentionLayerState>>) -> Self {
        Self { layers }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalLinearAttentionLayerState {
    pub(crate) conv_state: MetalObjcId,
    pub(crate) ssm_state: MetalObjcId,
    pub(crate) conv_output: MetalObjcId,
    pub(crate) delta_output: MetalObjcId,
    pub(crate) g_decay: MetalObjcId,
    pub(crate) beta_gate: MetalObjcId,
    pub(crate) conv_state_len: usize,
    pub(crate) ssm_state_len: usize,
    pub(crate) conv_dim: usize,
    pub(crate) total_value_width: usize,
    pub(crate) num_value_heads: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalLinearAttentionLayerState {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        conv_state: MetalObjcId,
        ssm_state: MetalObjcId,
        conv_output: MetalObjcId,
        delta_output: MetalObjcId,
        g_decay: MetalObjcId,
        beta_gate: MetalObjcId,
        conv_state_len: usize,
        ssm_state_len: usize,
        conv_dim: usize,
        total_value_width: usize,
        num_value_heads: usize,
    ) -> Self {
        Self {
            conv_state,
            ssm_state,
            conv_output,
            delta_output,
            g_decay,
            beta_gate,
            conv_state_len,
            ssm_state_len,
            conv_dim,
            total_value_width,
            num_value_heads,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) const METAL_REUSABLE_BUFFER_POOL_LIMIT: usize = 64;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) const METAL_REUSABLE_EXPERT_STAGING_POOL_LIMIT: usize = 16;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const METAL_WORKING_SET_HEADROOM_PERCENT: usize = 10;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const METAL_MINIMUM_WORKING_SET_HEADROOM_BYTES: usize = 1024 * 1024 * 1024;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalTrackedBufferClass {
    ResidentExpertWrapper,
    ActiveGeneral,
    Pooled,
    TransientExpert,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
struct MetalTrackedBuffer {
    len: usize,
    class: MetalTrackedBufferClass,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Default, Clone, Copy)]
struct MetalResourceCounter {
    count: usize,
    bytes: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalResourceCounter {
    fn add(&mut self, bytes: usize) {
        self.count = self.count.saturating_add(1);
        self.bytes = self.bytes.saturating_add(bytes);
    }

    fn remove(&mut self, bytes: usize) {
        self.count = self.count.saturating_sub(1);
        self.bytes = self.bytes.saturating_sub(bytes);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
struct MetalResourceLedgerState {
    buffers: HashMap<usize, MetalTrackedBuffer>,
    active_general: MetalResourceCounter,
    pooled: MetalResourceCounter,
    transient_expert: MetalResourceCounter,
    resident_expert_wrapper: MetalResourceCounter,
    resident_dense_bytes: usize,
    recurrent_state_bytes: usize,
    ledger_high_water_bytes: usize,
    recommended_working_set_bytes: usize,
    working_set_limit_bytes: usize,
    current_allocated_bytes: usize,
    unobserved_allocated_bytes: usize,
    driver_high_water_bytes: usize,
    in_flight_commands: usize,
    command_high_water: usize,
    command_submissions: usize,
    host_upload_bytes: usize,
    host_readback_bytes: usize,
    token_boundaries: usize,
    pressure_recoveries: usize,
    resource_limit_aborts: usize,
    buffer_allocations: usize,
    buffer_reuses: usize,
    buffer_recycles: usize,
    buffer_releases: usize,
    phase_cleanup_calls: usize,
    phase_cleanup_buffers: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalResourceLedgerState {
    fn new(recommended_working_set_bytes: usize, current_allocated_bytes: usize) -> Self {
        Self {
            buffers: HashMap::new(),
            active_general: MetalResourceCounter::default(),
            pooled: MetalResourceCounter::default(),
            transient_expert: MetalResourceCounter::default(),
            resident_expert_wrapper: MetalResourceCounter::default(),
            resident_dense_bytes: 0,
            recurrent_state_bytes: 0,
            ledger_high_water_bytes: 0,
            recommended_working_set_bytes,
            working_set_limit_bytes: default_metal_working_set_limit(recommended_working_set_bytes),
            current_allocated_bytes,
            unobserved_allocated_bytes: 0,
            driver_high_water_bytes: current_allocated_bytes,
            in_flight_commands: 0,
            command_high_water: 0,
            command_submissions: 0,
            host_upload_bytes: 0,
            host_readback_bytes: 0,
            token_boundaries: 0,
            pressure_recoveries: 0,
            resource_limit_aborts: 0,
            buffer_allocations: 0,
            buffer_reuses: 0,
            buffer_recycles: 0,
            buffer_releases: 0,
            phase_cleanup_calls: 0,
            phase_cleanup_buffers: 0,
        }
    }

    fn ledger_live_bytes(&self) -> usize {
        self.resident_dense_bytes
            .saturating_add(self.recurrent_state_bytes)
            .saturating_add(self.active_general.bytes)
            .saturating_add(self.pooled.bytes)
            .saturating_add(self.transient_expert.bytes)
            .saturating_add(self.resident_expert_wrapper.bytes)
    }

    fn update_ledger_high_water(&mut self) {
        self.ledger_high_water_bytes = self.ledger_high_water_bytes.max(self.ledger_live_bytes());
    }

    fn counter_mut(&mut self, class: MetalTrackedBufferClass) -> &mut MetalResourceCounter {
        match class {
            MetalTrackedBufferClass::ResidentExpertWrapper => &mut self.resident_expert_wrapper,
            MetalTrackedBufferClass::ActiveGeneral => &mut self.active_general,
            MetalTrackedBufferClass::Pooled => &mut self.pooled,
            MetalTrackedBufferClass::TransientExpert => &mut self.transient_expert,
        }
    }

    fn register_buffer(&mut self, id: usize, len: usize, class: MetalTrackedBufferClass) {
        debug_assert!(!self.buffers.contains_key(&id));
        if let Some(previous) = self.buffers.insert(id, MetalTrackedBuffer { len, class }) {
            self.counter_mut(previous.class).remove(previous.len);
        }
        self.counter_mut(class).add(len);
        self.buffer_allocations = self.buffer_allocations.saturating_add(1);
        self.update_ledger_high_water();
    }

    fn transition_buffer(&mut self, id: usize, class: MetalTrackedBufferClass) {
        let Some(previous) = self.buffers.get(&id).copied() else {
            debug_assert!(false, "untracked Metal buffer transition: {id:#x}");
            return;
        };
        if previous.class == class {
            return;
        }
        self.counter_mut(previous.class).remove(previous.len);
        self.counter_mut(class).add(previous.len);
        if previous.class == MetalTrackedBufferClass::Pooled {
            self.buffer_reuses = self.buffer_reuses.saturating_add(1);
        }
        if class == MetalTrackedBufferClass::Pooled {
            self.buffer_recycles = self.buffer_recycles.saturating_add(1);
        }
        if let Some(buffer) = self.buffers.get_mut(&id) {
            buffer.class = class;
        }
        self.update_ledger_high_water();
    }

    fn release_buffer(&mut self, id: usize) {
        let Some(previous) = self.buffers.remove(&id) else {
            debug_assert!(false, "untracked Metal buffer release: {id:#x}");
            return;
        };
        self.counter_mut(previous.class).remove(previous.len);
        self.buffer_releases = self.buffer_releases.saturating_add(1);
    }

    fn snapshot(&self) -> FlashMoeMetalResourceSnapshot {
        FlashMoeMetalResourceSnapshot {
            recommended_working_set_bytes: self.recommended_working_set_bytes,
            working_set_limit_bytes: self.working_set_limit_bytes,
            current_allocated_bytes: self.current_allocated_bytes,
            driver_high_water_bytes: self.driver_high_water_bytes,
            ledger_live_bytes: self.ledger_live_bytes(),
            ledger_high_water_bytes: self.ledger_high_water_bytes,
            resident_dense_bytes: self.resident_dense_bytes,
            recurrent_state_bytes: self.recurrent_state_bytes,
            resident_expert_wrapper_buffers: self.resident_expert_wrapper.count,
            resident_expert_wrapper_bytes: self.resident_expert_wrapper.bytes,
            active_general_buffers: self.active_general.count,
            active_general_bytes: self.active_general.bytes,
            pooled_buffers: self.pooled.count,
            pooled_bytes: self.pooled.bytes,
            transient_expert_buffers: self.transient_expert.count,
            transient_expert_bytes: self.transient_expert.bytes,
            in_flight_commands: self.in_flight_commands,
            command_high_water: self.command_high_water,
            command_submissions: self.command_submissions,
            host_upload_bytes: self.host_upload_bytes,
            host_readback_bytes: self.host_readback_bytes,
            token_boundaries: self.token_boundaries,
            pressure_recoveries: self.pressure_recoveries,
            resource_limit_aborts: self.resource_limit_aborts,
            buffer_allocations: self.buffer_allocations,
            buffer_reuses: self.buffer_reuses,
            buffer_recycles: self.buffer_recycles,
            buffer_releases: self.buffer_releases,
            phase_cleanup_calls: self.phase_cleanup_calls,
            phase_cleanup_buffers: self.phase_cleanup_buffers,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn default_metal_working_set_limit(recommended: usize) -> usize {
    if recommended == 0 {
        return usize::MAX;
    }
    let percent_headroom = recommended / METAL_WORKING_SET_HEADROOM_PERCENT;
    let headroom = percent_headroom
        .max(METAL_MINIMUM_WORKING_SET_HEADROOM_BYTES)
        .min(recommended / 2);
    recommended.saturating_sub(headroom)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalResourceLedger {
    state: Mutex<MetalResourceLedgerState>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Default for MetalResourceLedger {
    fn default() -> Self {
        Self {
            state: Mutex::new(MetalResourceLedgerState::new(0, 0)),
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalResourceLedger {
    unsafe fn from_device(device: MetalObjcId) -> Self {
        unsafe {
            Self {
                state: Mutex::new(MetalResourceLedgerState::new(
                    msg_send_usize0(device, sel("recommendedMaxWorkingSetSize")),
                    msg_send_usize0(device, sel("currentAllocatedSize")),
                )),
            }
        }
    }

    fn register_buffer(&self, id: MetalObjcId, len: usize, class: MetalTrackedBufferClass) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.register_buffer(id as usize, len, class);
        state.unobserved_allocated_bytes = state.unobserved_allocated_bytes.saturating_add(len);
    }

    fn transition_buffer(&self, id: MetalObjcId, class: MetalTrackedBufferClass) {
        self.state
            .lock()
            .expect("Metal resource ledger poisoned")
            .transition_buffer(id as usize, class);
    }

    fn release_buffer(&self, id: MetalObjcId) {
        self.state
            .lock()
            .expect("Metal resource ledger poisoned")
            .release_buffer(id as usize);
    }

    fn record_resident_resources(&self, dense_bytes: usize, recurrent_bytes: usize) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.resident_dense_bytes = dense_bytes;
        state.recurrent_state_bytes = recurrent_bytes;
        state.update_ledger_high_water();
    }

    fn set_working_set_limit_bytes(&self, requested: usize) -> anyhow::Result<()> {
        if requested == 0 {
            anyhow::bail!("FlashMoe Metal working-set limit must be greater than zero");
        }
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        let default_limit = default_metal_working_set_limit(state.recommended_working_set_bytes);
        state.working_set_limit_bytes = requested.min(default_limit);
        Ok(())
    }

    unsafe fn sample_device(&self, device: MetalObjcId, token_boundary: bool) -> usize {
        unsafe {
            let current = msg_send_usize0(device, sel("currentAllocatedSize"));
            let mut state = self.state.lock().expect("Metal resource ledger poisoned");
            let previous_high_water = state.driver_high_water_bytes;
            state.current_allocated_bytes = current;
            state.unobserved_allocated_bytes = 0;
            state.driver_high_water_bytes = state.driver_high_water_bytes.max(current);
            if token_boundary {
                state.token_boundaries = state.token_boundaries.saturating_add(1);
            }
            let new_high_water = current > previous_high_water;
            let high_water = state.driver_high_water_bytes;
            drop(state);
            if new_high_water {
                tracing::debug!(
                    target: "flashmoe::resources",
                    current_allocated_bytes = current,
                    driver_high_water_bytes = high_water,
                    token_boundary,
                    "FlashMoe Metal driver working-set high-water changed"
                );
            }
            current
        }
    }

    fn allocation_would_exceed_limit(&self, current: usize, requested: usize) -> bool {
        let state = self.state.lock().expect("Metal resource ledger poisoned");
        current.saturating_add(requested) > state.working_set_limit_bytes
    }

    fn estimated_allocation_would_exceed_limit(&self, requested: usize) -> bool {
        let state = self.state.lock().expect("Metal resource ledger poisoned");
        state
            .current_allocated_bytes
            .saturating_add(state.unobserved_allocated_bytes)
            .saturating_add(requested)
            > state.working_set_limit_bytes
    }

    fn record_pressure_recovery(&self) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.pressure_recoveries = state.pressure_recoveries.saturating_add(1);
    }

    fn record_resource_limit_abort(&self) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.resource_limit_aborts = state.resource_limit_aborts.saturating_add(1);
    }

    fn record_phase_cleanup(&self, buffers: usize) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.phase_cleanup_calls = state.phase_cleanup_calls.saturating_add(1);
        state.phase_cleanup_buffers = state.phase_cleanup_buffers.saturating_add(buffers);
    }

    fn command_started(&self) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.command_submissions = state.command_submissions.saturating_add(1);
        state.in_flight_commands = state.in_flight_commands.saturating_add(1);
        state.command_high_water = state.command_high_water.max(state.in_flight_commands);
    }

    fn record_host_upload(&self, bytes: usize) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.host_upload_bytes = state.host_upload_bytes.saturating_add(bytes);
    }

    fn record_host_readback(&self, bytes: usize) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.host_readback_bytes = state.host_readback_bytes.saturating_add(bytes);
    }

    fn command_finished(&self) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.in_flight_commands = state.in_flight_commands.saturating_sub(1);
    }

    fn snapshot(&self) -> FlashMoeMetalResourceSnapshot {
        self.state
            .lock()
            .expect("Metal resource ledger poisoned")
            .snapshot()
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalReusableBuffer {
    pub(crate) id: MetalObjcId,
    pub(crate) len: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn best_fit_reusable_buffer_index(buffers: &[MetalReusableBuffer], len: usize) -> Option<usize> {
    buffers
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.len >= len)
        .min_by_key(|(_, buffer)| buffer.len)
        .map(|(index, _)| index)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn reusable_buffer_replacement_index(buffers: &[MetalReusableBuffer], len: usize) -> Option<usize> {
    buffers
        .iter()
        .enumerate()
        .min_by_key(|(_, buffer)| buffer.len)
        .and_then(|(index, buffer)| (buffer.len < len).then_some(index))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalReusableBuffer {
    pub(crate) fn new(id: MetalObjcId, len: usize) -> Self {
        Self { id, len }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalBufferPool {
    reusable: Mutex<Vec<MetalReusableBuffer>>,
    reusable_expert_staging: Mutex<Vec<MetalReusableBuffer>>,
    resources: Arc<MetalResourceLedger>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Default for MetalBufferPool {
    fn default() -> Self {
        Self::new(Arc::new(MetalResourceLedger::default()))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalBufferPool {
    fn new(resources: Arc<MetalResourceLedger>) -> Self {
        Self {
            reusable: Mutex::new(Vec::new()),
            reusable_expert_staging: Mutex::new(Vec::new()),
            resources,
        }
    }

    fn resources(&self) -> &Arc<MetalResourceLedger> {
        &self.resources
    }

    pub(crate) unsafe fn buffer_with_len(
        &self,
        device: MetalObjcId,
        len: usize,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe { self.buffer_with_len_class(device, len, MetalTrackedBufferClass::ActiveGeneral) }
    }

    unsafe fn buffer_with_len_class(
        &self,
        device: MetalObjcId,
        len: usize,
        class: MetalTrackedBufferClass,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            if class != MetalTrackedBufferClass::TransientExpert {
                let mut reusable = self.reusable.lock().expect("metal buffer pool poisoned");
                if let Some(index) = best_fit_reusable_buffer_index(&reusable, len) {
                    let buffer = reusable.swap_remove(index);
                    self.resources.transition_buffer(buffer.id, class);
                    return Ok(buffer.id);
                }
            }
            self.ensure_allocation_capacity(device, len)?;
            let buffer =
                msg_send_id2_usize_u64(device, sel("newBufferWithLength:options:"), len, 0);
            if !buffer.is_null() {
                self.resources.register_buffer(buffer, len, class);
                return Ok(buffer);
            }
            let (released_buffers, released_bytes) = self.release_idle_buffers();
            let retry = msg_send_id2_usize_u64(device, sel("newBufferWithLength:options:"), len, 0);
            if retry.is_null() {
                let current_allocated_bytes = msg_send_usize0(device, sel("currentAllocatedSize"));
                let recommended_working_set_bytes =
                    msg_send_usize0(device, sel("recommendedMaxWorkingSetSize"));
                anyhow::bail!(
                    "failed to allocate Flash-MoE Metal buffer: requested_bytes={len} released_pooled_buffers={released_buffers} released_pooled_bytes={released_bytes} current_allocated_bytes={current_allocated_bytes} recommended_working_set_bytes={recommended_working_set_bytes}"
                );
            }
            tracing::warn!(
                requested_bytes = len,
                released_pooled_buffers = released_buffers,
                released_pooled_bytes = released_bytes,
                "FlashMoe Metal allocation recovered after draining the reusable buffer pool"
            );
            self.resources.register_buffer(retry, len, class);
            Ok(retry)
        }
    }

    unsafe fn ensure_allocation_capacity(
        &self,
        device: MetalObjcId,
        requested_bytes: usize,
    ) -> anyhow::Result<()> {
        unsafe {
            if !self
                .resources
                .estimated_allocation_would_exceed_limit(requested_bytes)
            {
                return Ok(());
            }
            let current = self.resources.sample_device(device, false);
            if !self
                .resources
                .allocation_would_exceed_limit(current, requested_bytes)
            {
                return Ok(());
            }
            let (released_buffers, released_bytes) = self.release_idle_buffers();
            let after_drain = self.resources.sample_device(device, false);
            if self
                .resources
                .allocation_would_exceed_limit(after_drain, requested_bytes)
            {
                self.resources.record_resource_limit_abort();
                let snapshot = self.resources.snapshot();
                anyhow::bail!(
                    "FlashMoe Metal resource limit would be exceeded: requested_bytes={requested_bytes} current_allocated_bytes={after_drain} working_set_limit_bytes={} recommended_working_set_bytes={} released_pooled_buffers={released_buffers} released_pooled_bytes={released_bytes} driver_high_water_bytes={} ledger_live_bytes={}",
                    snapshot.working_set_limit_bytes,
                    snapshot.recommended_working_set_bytes,
                    snapshot.driver_high_water_bytes,
                    snapshot.ledger_live_bytes,
                );
            }
            self.resources.record_pressure_recovery();
            tracing::warn!(
                requested_bytes,
                current_allocated_bytes = current,
                current_after_drain_bytes = after_drain,
                released_pooled_buffers = released_buffers,
                released_pooled_bytes = released_bytes,
                "FlashMoe Metal working-set pressure recovered after draining idle buffers"
            );
            Ok(())
        }
    }

    fn release_idle_buffers(&self) -> (usize, usize) {
        let mut reusable = self.reusable.lock().expect("metal buffer pool poisoned");
        let mut released_buffers = reusable.len();
        let mut released_bytes = reusable
            .iter()
            .map(|buffer| buffer.len)
            .fold(0usize, usize::saturating_add);
        unsafe {
            for reusable_buffer in reusable.drain(..) {
                self.resources.release_buffer(reusable_buffer.id);
                release(reusable_buffer.id);
            }
        }
        drop(reusable);
        let mut expert_staging = self
            .reusable_expert_staging
            .lock()
            .expect("metal expert staging pool poisoned");
        released_buffers = released_buffers.saturating_add(expert_staging.len());
        released_bytes = released_bytes.saturating_add(
            expert_staging
                .iter()
                .map(|buffer| buffer.len)
                .fold(0usize, usize::saturating_add),
        );
        unsafe {
            for reusable_buffer in expert_staging.drain(..) {
                self.resources.release_buffer(reusable_buffer.id);
                purge_and_release_metal_buffer(reusable_buffer.id);
            }
        }
        (released_buffers, released_bytes)
    }

    pub(crate) unsafe fn buffer_with_bytes(
        &self,
        device: MetalObjcId,
        bytes: &[u8],
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            let buffer = self.buffer_with_len(device, bytes.len())?;
            let contents = msg_send_ptr0(buffer, sel("contents"));
            ptr::copy_nonoverlapping(bytes.as_ptr(), contents.cast::<u8>(), bytes.len());
            self.resources.record_host_upload(bytes.len());
            Ok(buffer)
        }
    }

    pub(crate) unsafe fn transient_expert_buffer_with_bytes(
        &self,
        device: MetalObjcId,
        bytes: &[u8],
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            let buffer = {
                let mut reusable = self
                    .reusable_expert_staging
                    .lock()
                    .expect("metal expert staging pool poisoned");
                best_fit_reusable_buffer_index(&reusable, bytes.len()).map(|index| {
                    let buffer = reusable.swap_remove(index);
                    self.resources
                        .transition_buffer(buffer.id, MetalTrackedBufferClass::TransientExpert);
                    buffer.id
                })
            };
            let buffer = match buffer {
                Some(buffer) => buffer,
                None => self.buffer_with_len_class(
                    device,
                    bytes.len(),
                    MetalTrackedBufferClass::TransientExpert,
                )?,
            };
            let contents = msg_send_ptr0(buffer, sel("contents"));
            ptr::copy_nonoverlapping(bytes.as_ptr(), contents.cast::<u8>(), bytes.len());
            self.resources.record_host_upload(bytes.len());
            Ok(buffer)
        }
    }

    unsafe fn tracked_buffer_with_len(
        &self,
        device: MetalObjcId,
        len: usize,
        owned: &mut Vec<MetalObjcId>,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            match self.buffer_with_len(device, len) {
                Ok(buffer) => {
                    owned.push(buffer);
                    Ok(buffer)
                }
                Err(error) => {
                    self.recycle_or_release(owned, true);
                    owned.clear();
                    Err(error)
                }
            }
        }
    }

    unsafe fn tracked_buffer_with_bytes(
        &self,
        device: MetalObjcId,
        bytes: &[u8],
        owned: &mut Vec<MetalObjcId>,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            let buffer = self.tracked_buffer_with_len(device, bytes.len(), owned)?;
            let contents = msg_send_ptr0(buffer, sel("contents"));
            ptr::copy_nonoverlapping(bytes.as_ptr(), contents.cast::<u8>(), bytes.len());
            self.resources.record_host_upload(bytes.len());
            Ok(buffer)
        }
    }

    unsafe fn read_f32_buffer(&self, buffer: MetalObjcId, len: usize) -> Vec<f32> {
        let bytes = len.saturating_mul(std::mem::size_of::<f32>());
        self.resources.record_host_readback(bytes);
        unsafe { read_f32_buffer(buffer, len) }
    }

    unsafe fn read_f32_buffer_offset(
        &self,
        buffer: MetalObjcId,
        offset: usize,
        len: usize,
    ) -> Vec<f32> {
        let bytes = len.saturating_mul(std::mem::size_of::<f32>());
        self.resources.record_host_readback(bytes);
        unsafe { read_f32_buffer_offset(buffer, offset, len) }
    }

    pub(crate) unsafe fn recycle(&self, buffer: MetalObjcId) {
        unsafe {
            let len = msg_send_usize0(buffer, sel("length"));
            let mut reusable = self.reusable.lock().expect("metal buffer pool poisoned");
            if reusable.len() < METAL_REUSABLE_BUFFER_POOL_LIMIT {
                self.resources
                    .transition_buffer(buffer, MetalTrackedBufferClass::Pooled);
                reusable.push(MetalReusableBuffer::new(buffer, len));
                return;
            }
            let Some(index) = reusable_buffer_replacement_index(&reusable, len) else {
                drop(reusable);
                self.resources.release_buffer(buffer);
                release(buffer);
                return;
            };
            self.resources
                .transition_buffer(buffer, MetalTrackedBufferClass::Pooled);
            let evicted =
                std::mem::replace(&mut reusable[index], MetalReusableBuffer::new(buffer, len));
            drop(reusable);
            self.resources.release_buffer(evicted.id);
            release(evicted.id);
        }
    }

    unsafe fn recycle_expert_staging(&self, buffer: MetalObjcId) {
        unsafe {
            let len = msg_send_usize0(buffer, sel("length"));
            let mut reusable = self
                .reusable_expert_staging
                .lock()
                .expect("metal expert staging pool poisoned");
            if reusable.len() < METAL_REUSABLE_EXPERT_STAGING_POOL_LIMIT {
                self.resources
                    .transition_buffer(buffer, MetalTrackedBufferClass::Pooled);
                reusable.push(MetalReusableBuffer::new(buffer, len));
                return;
            }
            let Some(index) = reusable_buffer_replacement_index(&reusable, len) else {
                drop(reusable);
                self.resources.release_buffer(buffer);
                purge_and_release_metal_buffer(buffer);
                return;
            };
            self.resources
                .transition_buffer(buffer, MetalTrackedBufferClass::Pooled);
            let evicted =
                std::mem::replace(&mut reusable[index], MetalReusableBuffer::new(buffer, len));
            drop(reusable);
            self.resources.release_buffer(evicted.id);
            purge_and_release_metal_buffer(evicted.id);
        }
    }

    pub(crate) fn recycle_or_release(&self, buffers: &[MetalObjcId], release_only: bool) {
        unsafe {
            for buffer in buffers.iter().copied() {
                if release_only {
                    self.resources.release_buffer(buffer);
                    release(buffer);
                } else {
                    self.recycle(buffer);
                }
            }
        }
    }

    pub(crate) fn recycle_or_release_phase(
        &self,
        buffers: Vec<MetalPhaseBuffer>,
        release_only: bool,
    ) {
        self.resources.record_phase_cleanup(buffers.len());
        unsafe {
            for buffer in buffers {
                match buffer.class {
                    MetalPhaseBufferClass::BorrowedExpert => release(buffer.id),
                    MetalPhaseBufferClass::TransientExpert => {
                        if release_only {
                            self.resources.release_buffer(buffer.id);
                            purge_and_release_metal_buffer(buffer.id);
                        } else {
                            self.recycle_expert_staging(buffer.id);
                        }
                    }
                    MetalPhaseBufferClass::General => {
                        if release_only {
                            self.resources.release_buffer(buffer.id);
                            release(buffer.id);
                        } else {
                            self.recycle(buffer.id);
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn release_all(&mut self) {
        let reusable = self.reusable.get_mut().expect("metal buffer pool poisoned");
        unsafe {
            for buffer in reusable.drain(..) {
                self.resources.release_buffer(buffer.id);
                release(buffer.id);
            }
        }
        let expert_staging = self
            .reusable_expert_staging
            .get_mut()
            .expect("metal expert staging pool poisoned");
        unsafe {
            for buffer in expert_staging.drain(..) {
                self.resources.release_buffer(buffer.id);
                purge_and_release_metal_buffer(buffer.id);
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalBufferPool {
    fn drop(&mut self) {
        self.release_all();
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalDenseWeights {
    pub(crate) buffer: MetalObjcId,
    _mmap: Arc<memmap2::Mmap>,
    pub(crate) len: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalDenseWeights {
    pub(crate) fn new(buffer: MetalObjcId, mmap: Arc<memmap2::Mmap>, len: usize) -> Self {
        Self {
            buffer,
            _mmap: mmap,
            len,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalDispatchSize {
    pub(crate) width: u64,
    pub(crate) height: u64,
    pub(crate) depth: u64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalDispatchSize {
    pub(crate) const fn new(width: u64, height: u64, depth: u64) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalDispatchMode {
    Threads,
    Threadgroups,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalDispatchPlan {
    pub(crate) mode: MetalDispatchMode,
    pub(crate) grid: MetalDispatchSize,
    pub(crate) threadgroup: MetalDispatchSize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalDispatchPlan {
    pub(crate) fn threads(threads: u64) -> Self {
        Self {
            mode: MetalDispatchMode::Threads,
            grid: MetalDispatchSize::new(threads, 1, 1),
            threadgroup: MetalDispatchSize::new(threads.clamp(1, 64), 1, 1),
        }
    }

    pub(crate) fn q4_threadgroups(rows: u64) -> Self {
        const Q4_ROWS_PER_THREADGROUP: u64 = 8;
        Self {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(rows.div_ceil(Q4_ROWS_PER_THREADGROUP).max(1), 1, 1),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    }

    pub(crate) fn q4_mmap_threadgroups(rows: u64) -> Self {
        const Q4_MMAP_ROWS_PER_THREADGROUP: u64 = 16;
        Self {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(rows.div_ceil(Q4_MMAP_ROWS_PER_THREADGROUP).max(1), 1, 1),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    }

    pub(crate) fn q4_mmap_matrix_threadgroups(rows: u64, input_rows: u64) -> Self {
        const Q4_MMAP_ROWS_PER_THREADGROUP: u64 = 16;
        Self {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(
                rows.div_ceil(Q4_MMAP_ROWS_PER_THREADGROUP).max(1),
                input_rows.max(1),
                1,
            ),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    }

    pub(crate) fn qwen_attention_threadgroups(query_rows: u64, query_heads: u64) -> Self {
        Self {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(query_rows.max(1), query_heads.max(1), 1),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    }

    pub(crate) fn single_threadgroup(threads: u64) -> Self {
        Self {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(1, 1, 1),
            threadgroup: MetalDispatchSize::new(threads.clamp(1, 256), 1, 1),
        }
    }
}

const DEFAULT_FLASHMOE_METAL_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_FLASHMOE_METAL_COMMAND_POLL_INTERVAL: Duration = Duration::from_micros(100);

pub(crate) mod kernels {
    pub(crate) const DEEPSEEK_IQ2_XXS_PAIR_SWIGLU: &str =
        "kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32";
    pub(crate) const DEEPSEEK_Q2_K_SUM6: &str = "kernel_mul_mv_slots6_q2_K_sum6_f32";
    pub(crate) const Q4_FMA_MATVEC: &str = "q4_fma_matvec";
    pub(crate) const Q4_FMA_MATVEC_BF16_SCALE_BIAS: &str = "q4_fma_matvec_bf16_scale_bias";
    pub(crate) const MXFP4_FMA_MATVEC_E8M0: &str = "mxfp4_fma_matvec_e8m0";
    pub(crate) const Q4_SWIGLU_FUSED: &str = "q4_swiglu_fused";
    pub(crate) const Q4_SWIGLU_FUSED_BF16_SCALE_BIAS: &str = "q4_swiglu_fused_bf16_scale_bias";
    pub(crate) const Q4_MMAP_FMA_MATVEC: &str = "q4_mmap_fma_matvec";
    pub(crate) const Q4_MMAP_FMA_MATVEC_BF16_SCALE_BIAS: &str =
        "q4_mmap_fma_matvec_bf16_scale_bias";
    pub(crate) const Q4_MMAP_FMA_MATVEC_BATCH: &str = "q4_mmap_fma_matvec_batch";
    pub(crate) const Q4_MMAP_FMA_MATVEC_BATCH_BF16_SCALE_BIAS: &str =
        "q4_mmap_fma_matvec_batch_bf16_scale_bias";
    pub(crate) const Q4_MMAP_FMA_MULTILINEAR_BF16_SCALE_BIAS: &str =
        "q4_mmap_fma_multilinear_bf16_scale_bias";
    pub(crate) const GLM_MLA_PREPARE_QUERY_KV: &str = "glm_mla_prepare_query_kv";
    pub(crate) const GLM_MLA_ABSORBED_SCORES: &str = "glm_mla_absorbed_scores";
    pub(crate) const GLM_MLA_SOFTMAX: &str = "glm_mla_softmax";
    pub(crate) const GLM_MLA_CONTEXT: &str = "glm_mla_context";
    pub(crate) const DENSE_MMAP_FMA_MATVEC_BF16: &str = "dense_mmap_fma_matvec_bf16";
    pub(crate) const DENSE_MMAP_FMA_MATVEC_F16: &str = "dense_mmap_fma_matvec_f16";
    pub(crate) const DENSE_MMAP_FMA_MATVEC_F32: &str = "dense_mmap_fma_matvec_f32";
    pub(crate) const DENSE_MMAP_FMA_MATRIX_BF16: &str = "dense_mmap_fma_matrix_bf16";
    pub(crate) const DENSE_MMAP_FMA_MATRIX_F16: &str = "dense_mmap_fma_matrix_f16";
    pub(crate) const DENSE_MMAP_FMA_MATRIX_F32: &str = "dense_mmap_fma_matrix_f32";
    pub(crate) const RMS_NORM_REDUCED: &str = "rms_norm_reduced";
    pub(crate) const RESIDUAL_ADD_RMS_NORM: &str = "residual_add_rms_norm";
    pub(crate) const ATTENTION_SCORES: &str = "attention_scores";
    pub(crate) const QWEN_CAUSAL_ATTENTION_ROWS: &str = "qwen_causal_attention_rows";
    pub(crate) const EXPERT_MLP_FUSED: &str = "expert_mlp_fused";
    pub(crate) const SILU_PRODUCT: &str = "silu_product";
    pub(crate) const SHARED_EXPERT_ACTIVATION: &str = "shared_expert_activation";
    pub(crate) const COMBINE_EXPERT_PHASE: &str = "combine_expert_phase";
    pub(crate) const QWEN_LAYER_MAJOR_GATHER: &str = "qwen_layer_major_gather";
    pub(crate) const QWEN_LAYER_MAJOR_COMBINE: &str = "qwen_layer_major_combine";
    pub(crate) const FILL_ZERO: &str = "fill_zero";
    pub(crate) const TOPK_VOCAB: &str = "topk_vocab";
    pub(crate) const LINEAR_CONV1D_STEP_BF16: &str = "linear_conv1d_step_bf16";
    pub(crate) const LINEAR_CONV1D_STEP_F16: &str = "linear_conv1d_step_f16";
    pub(crate) const LINEAR_CONV1D_STEP_F32: &str = "linear_conv1d_step_f32";
    pub(crate) const LINEAR_RMS_NORM_QK: &str = "linear_rms_norm_qk";
    pub(crate) const LINEAR_COMPUTE_DECAY_BETA_BF16: &str = "linear_compute_decay_beta_bf16";
    pub(crate) const LINEAR_COMPUTE_DECAY_BETA_F16: &str = "linear_compute_decay_beta_f16";
    pub(crate) const LINEAR_COMPUTE_DECAY_BETA_F32: &str = "linear_compute_decay_beta_f32";
    pub(crate) const LINEAR_GATED_DELTA_STEP: &str = "linear_gated_delta_step";
    pub(crate) const LINEAR_GATED_RMS_NORM_BF16: &str = "linear_gated_rms_norm_bf16";
    pub(crate) const LINEAR_GATED_RMS_NORM_F16: &str = "linear_gated_rms_norm_f16";
    pub(crate) const LINEAR_GATED_RMS_NORM_F32: &str = "linear_gated_rms_norm_f32";
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalRuntimeCapabilities {
    kernels: BTreeSet<&'static str>,
}

impl MetalRuntimeCapabilities {
    pub(crate) fn from_pipeline_names(names: MetalPipelineNameSet) -> Self {
        Self {
            kernels: names.kernel_names().into_iter().collect(),
        }
    }

    pub(crate) fn supports(&self, kernel: &str) -> bool {
        self.kernels.contains(kernel)
    }

    pub(crate) fn with_additional(mut self, kernels: &'static [&'static str]) -> Self {
        self.kernels.extend(kernels.iter().copied());
        self
    }

    #[cfg(test)]
    pub(crate) fn empty_for_test() -> Self {
        Self {
            kernels: BTreeSet::new(),
        }
    }

    pub(crate) fn require_all(&self, kernels: &[&'static str]) -> anyhow::Result<()> {
        let missing = kernels
            .iter()
            .copied()
            .filter(|kernel| !self.supports(kernel))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            anyhow::bail!("missing Metal kernels: {}", missing.join(", "));
        }
        Ok(())
    }
}

#[cfg(test)]
const REQUIRED_FORWARD_KERNELS: &[&str] = &[
    kernels::Q4_FMA_MATVEC,
    kernels::Q4_FMA_MATVEC_BF16_SCALE_BIAS,
    kernels::MXFP4_FMA_MATVEC_E8M0,
    kernels::Q4_SWIGLU_FUSED,
    kernels::Q4_SWIGLU_FUSED_BF16_SCALE_BIAS,
    kernels::Q4_MMAP_FMA_MATVEC,
    kernels::Q4_MMAP_FMA_MATVEC_BF16_SCALE_BIAS,
    kernels::Q4_MMAP_FMA_MATVEC_BATCH,
    kernels::Q4_MMAP_FMA_MATVEC_BATCH_BF16_SCALE_BIAS,
    kernels::Q4_MMAP_FMA_MULTILINEAR_BF16_SCALE_BIAS,
    kernels::GLM_MLA_PREPARE_QUERY_KV,
    kernels::GLM_MLA_ABSORBED_SCORES,
    kernels::GLM_MLA_SOFTMAX,
    kernels::GLM_MLA_CONTEXT,
    kernels::DENSE_MMAP_FMA_MATVEC_BF16,
    kernels::DENSE_MMAP_FMA_MATVEC_F16,
    kernels::DENSE_MMAP_FMA_MATVEC_F32,
    kernels::DENSE_MMAP_FMA_MATRIX_BF16,
    kernels::DENSE_MMAP_FMA_MATRIX_F16,
    kernels::DENSE_MMAP_FMA_MATRIX_F32,
    kernels::RMS_NORM_REDUCED,
    kernels::RESIDUAL_ADD_RMS_NORM,
    kernels::ATTENTION_SCORES,
    kernels::QWEN_CAUSAL_ATTENTION_ROWS,
    kernels::EXPERT_MLP_FUSED,
    kernels::SILU_PRODUCT,
    kernels::SHARED_EXPERT_ACTIVATION,
    kernels::COMBINE_EXPERT_PHASE,
    kernels::QWEN_LAYER_MAJOR_GATHER,
    kernels::QWEN_LAYER_MAJOR_COMBINE,
    kernels::FILL_ZERO,
    kernels::TOPK_VOCAB,
    kernels::LINEAR_CONV1D_STEP_BF16,
    kernels::LINEAR_CONV1D_STEP_F16,
    kernels::LINEAR_CONV1D_STEP_F32,
    kernels::LINEAR_RMS_NORM_QK,
    kernels::LINEAR_COMPUTE_DECAY_BETA_BF16,
    kernels::LINEAR_COMPUTE_DECAY_BETA_F16,
    kernels::LINEAR_COMPUTE_DECAY_BETA_F32,
    kernels::LINEAR_GATED_DELTA_STEP,
    kernels::LINEAR_GATED_RMS_NORM_BF16,
    kernels::LINEAR_GATED_RMS_NORM_F16,
    kernels::LINEAR_GATED_RMS_NORM_F32,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalPipelineNameSet {
    pub(crate) q4: &'static str,
    pub(crate) q4_bf16_scale_bias: &'static str,
    pub(crate) mxfp4_e8m0: &'static str,
    pub(crate) q4_swiglu: &'static str,
    pub(crate) q4_swiglu_bf16_scale_bias: &'static str,
    pub(crate) q4_mmap: &'static str,
    pub(crate) q4_mmap_bf16_scale_bias: &'static str,
    pub(crate) q4_mmap_batch: &'static str,
    pub(crate) q4_mmap_batch_bf16_scale_bias: &'static str,
    pub(crate) q4_mmap_multilinear_bf16_scale_bias: &'static str,
    pub(crate) glm_mla_prepare_query_kv: &'static str,
    pub(crate) glm_mla_absorbed_scores: &'static str,
    pub(crate) glm_mla_softmax: &'static str,
    pub(crate) glm_mla_context: &'static str,
    pub(crate) dense_mmap_bf16: &'static str,
    pub(crate) dense_mmap_f16: &'static str,
    pub(crate) dense_mmap_f32: &'static str,
    pub(crate) dense_matrix_bf16: &'static str,
    pub(crate) dense_matrix_f16: &'static str,
    pub(crate) dense_matrix_f32: &'static str,
    pub(crate) rms_norm_reduced: &'static str,
    pub(crate) residual_rms_norm: &'static str,
    pub(crate) attention: &'static str,
    pub(crate) qwen_causal_attention_rows: &'static str,
    pub(crate) expert_mlp: &'static str,
    pub(crate) silu_product: &'static str,
    pub(crate) shared_expert_activation: &'static str,
    pub(crate) combine_expert_phase: &'static str,
    pub(crate) qwen_layer_major_gather: &'static str,
    pub(crate) qwen_layer_major_combine: &'static str,
    pub(crate) fill_zero: &'static str,
    pub(crate) topk_vocab: &'static str,
    pub(crate) linear_conv1d_bf16: &'static str,
    pub(crate) linear_conv1d_f16: &'static str,
    pub(crate) linear_conv1d_f32: &'static str,
    pub(crate) linear_rms_norm_qk: &'static str,
    pub(crate) linear_decay_beta_bf16: &'static str,
    pub(crate) linear_decay_beta_f16: &'static str,
    pub(crate) linear_decay_beta_f32: &'static str,
    pub(crate) linear_delta_step: &'static str,
    pub(crate) linear_gated_rms_norm_bf16: &'static str,
    pub(crate) linear_gated_rms_norm_f16: &'static str,
    pub(crate) linear_gated_rms_norm_f32: &'static str,
}

impl MetalPipelineNameSet {
    pub(crate) fn new() -> Self {
        Self {
            q4: kernels::Q4_FMA_MATVEC,
            q4_bf16_scale_bias: kernels::Q4_FMA_MATVEC_BF16_SCALE_BIAS,
            mxfp4_e8m0: kernels::MXFP4_FMA_MATVEC_E8M0,
            q4_swiglu: kernels::Q4_SWIGLU_FUSED,
            q4_swiglu_bf16_scale_bias: kernels::Q4_SWIGLU_FUSED_BF16_SCALE_BIAS,
            q4_mmap: kernels::Q4_MMAP_FMA_MATVEC,
            q4_mmap_bf16_scale_bias: kernels::Q4_MMAP_FMA_MATVEC_BF16_SCALE_BIAS,
            q4_mmap_batch: kernels::Q4_MMAP_FMA_MATVEC_BATCH,
            q4_mmap_batch_bf16_scale_bias: kernels::Q4_MMAP_FMA_MATVEC_BATCH_BF16_SCALE_BIAS,
            q4_mmap_multilinear_bf16_scale_bias: kernels::Q4_MMAP_FMA_MULTILINEAR_BF16_SCALE_BIAS,
            glm_mla_prepare_query_kv: kernels::GLM_MLA_PREPARE_QUERY_KV,
            glm_mla_absorbed_scores: kernels::GLM_MLA_ABSORBED_SCORES,
            glm_mla_softmax: kernels::GLM_MLA_SOFTMAX,
            glm_mla_context: kernels::GLM_MLA_CONTEXT,
            dense_mmap_bf16: kernels::DENSE_MMAP_FMA_MATVEC_BF16,
            dense_mmap_f16: kernels::DENSE_MMAP_FMA_MATVEC_F16,
            dense_mmap_f32: kernels::DENSE_MMAP_FMA_MATVEC_F32,
            dense_matrix_bf16: kernels::DENSE_MMAP_FMA_MATRIX_BF16,
            dense_matrix_f16: kernels::DENSE_MMAP_FMA_MATRIX_F16,
            dense_matrix_f32: kernels::DENSE_MMAP_FMA_MATRIX_F32,
            rms_norm_reduced: kernels::RMS_NORM_REDUCED,
            residual_rms_norm: kernels::RESIDUAL_ADD_RMS_NORM,
            attention: kernels::ATTENTION_SCORES,
            qwen_causal_attention_rows: kernels::QWEN_CAUSAL_ATTENTION_ROWS,
            expert_mlp: kernels::EXPERT_MLP_FUSED,
            silu_product: kernels::SILU_PRODUCT,
            shared_expert_activation: kernels::SHARED_EXPERT_ACTIVATION,
            combine_expert_phase: kernels::COMBINE_EXPERT_PHASE,
            qwen_layer_major_gather: kernels::QWEN_LAYER_MAJOR_GATHER,
            qwen_layer_major_combine: kernels::QWEN_LAYER_MAJOR_COMBINE,
            fill_zero: kernels::FILL_ZERO,
            topk_vocab: kernels::TOPK_VOCAB,
            linear_conv1d_bf16: kernels::LINEAR_CONV1D_STEP_BF16,
            linear_conv1d_f16: kernels::LINEAR_CONV1D_STEP_F16,
            linear_conv1d_f32: kernels::LINEAR_CONV1D_STEP_F32,
            linear_rms_norm_qk: kernels::LINEAR_RMS_NORM_QK,
            linear_decay_beta_bf16: kernels::LINEAR_COMPUTE_DECAY_BETA_BF16,
            linear_decay_beta_f16: kernels::LINEAR_COMPUTE_DECAY_BETA_F16,
            linear_decay_beta_f32: kernels::LINEAR_COMPUTE_DECAY_BETA_F32,
            linear_delta_step: kernels::LINEAR_GATED_DELTA_STEP,
            linear_gated_rms_norm_bf16: kernels::LINEAR_GATED_RMS_NORM_BF16,
            linear_gated_rms_norm_f16: kernels::LINEAR_GATED_RMS_NORM_F16,
            linear_gated_rms_norm_f32: kernels::LINEAR_GATED_RMS_NORM_F32,
        }
    }

    pub(crate) fn kernel_names(self) -> Vec<&'static str> {
        vec![
            self.q4,
            self.q4_bf16_scale_bias,
            self.mxfp4_e8m0,
            self.q4_swiglu,
            self.q4_swiglu_bf16_scale_bias,
            self.q4_mmap,
            self.q4_mmap_bf16_scale_bias,
            self.q4_mmap_batch,
            self.q4_mmap_batch_bf16_scale_bias,
            self.q4_mmap_multilinear_bf16_scale_bias,
            self.glm_mla_prepare_query_kv,
            self.glm_mla_absorbed_scores,
            self.glm_mla_softmax,
            self.glm_mla_context,
            self.dense_mmap_bf16,
            self.dense_mmap_f16,
            self.dense_mmap_f32,
            self.dense_matrix_bf16,
            self.dense_matrix_f16,
            self.dense_matrix_f32,
            self.rms_norm_reduced,
            self.residual_rms_norm,
            self.attention,
            self.qwen_causal_attention_rows,
            self.expert_mlp,
            self.silu_product,
            self.shared_expert_activation,
            self.combine_expert_phase,
            self.qwen_layer_major_gather,
            self.qwen_layer_major_combine,
            self.fill_zero,
            self.topk_vocab,
            self.linear_conv1d_bf16,
            self.linear_conv1d_f16,
            self.linear_conv1d_f32,
            self.linear_rms_norm_qk,
            self.linear_decay_beta_bf16,
            self.linear_decay_beta_f16,
            self.linear_decay_beta_f32,
            self.linear_delta_step,
            self.linear_gated_rms_norm_bf16,
            self.linear_gated_rms_norm_f16,
            self.linear_gated_rms_norm_f32,
        ]
    }
}

#[derive(Debug)]
pub(crate) struct MetalPipelineSet<T> {
    pub(crate) q4_pipeline: T,
    pub(crate) q4_bf16_scale_bias_pipeline: T,
    pub(crate) mxfp4_e8m0_pipeline: T,
    pub(crate) q4_swiglu_pipeline: T,
    pub(crate) q4_swiglu_bf16_scale_bias_pipeline: T,
    pub(crate) q4_mmap_pipeline: T,
    pub(crate) q4_mmap_bf16_scale_bias_pipeline: T,
    pub(crate) q4_mmap_batch_pipeline: T,
    pub(crate) q4_mmap_batch_bf16_scale_bias_pipeline: T,
    pub(crate) q4_mmap_multilinear_bf16_scale_bias_pipeline: T,
    pub(crate) glm_mla_prepare_query_kv_pipeline: T,
    pub(crate) glm_mla_absorbed_scores_pipeline: T,
    pub(crate) glm_mla_softmax_pipeline: T,
    pub(crate) glm_mla_context_pipeline: T,
    pub(crate) dense_mmap_bf16_pipeline: T,
    pub(crate) dense_mmap_f16_pipeline: T,
    pub(crate) dense_mmap_f32_pipeline: T,
    pub(crate) dense_matrix_bf16_pipeline: T,
    pub(crate) dense_matrix_f16_pipeline: T,
    pub(crate) dense_matrix_f32_pipeline: T,
    pub(crate) rms_norm_reduced_pipeline: T,
    pub(crate) residual_rms_norm_pipeline: T,
    pub(crate) attention_pipeline: T,
    pub(crate) qwen_causal_attention_rows_pipeline: T,
    pub(crate) expert_mlp_pipeline: T,
    pub(crate) silu_product_pipeline: T,
    pub(crate) shared_expert_activation_pipeline: T,
    pub(crate) combine_expert_phase_pipeline: T,
    pub(crate) qwen_layer_major_gather_pipeline: T,
    pub(crate) qwen_layer_major_combine_pipeline: T,
    pub(crate) fill_zero_pipeline: T,
    pub(crate) topk_vocab_pipeline: T,
    pub(crate) linear_conv1d_bf16_pipeline: T,
    pub(crate) linear_conv1d_f16_pipeline: T,
    pub(crate) linear_conv1d_f32_pipeline: T,
    pub(crate) linear_rms_norm_qk_pipeline: T,
    pub(crate) linear_decay_beta_bf16_pipeline: T,
    pub(crate) linear_decay_beta_f16_pipeline: T,
    pub(crate) linear_decay_beta_f32_pipeline: T,
    pub(crate) linear_delta_step_pipeline: T,
    pub(crate) linear_gated_rms_norm_bf16_pipeline: T,
    pub(crate) linear_gated_rms_norm_f16_pipeline: T,
    pub(crate) linear_gated_rms_norm_f32_pipeline: T,
}

impl<T: Copy> MetalPipelineSet<T> {
    pub(crate) fn release_with(&self, mut release: impl FnMut(T)) {
        release(self.q4_pipeline);
        release(self.q4_bf16_scale_bias_pipeline);
        release(self.mxfp4_e8m0_pipeline);
        release(self.q4_swiglu_pipeline);
        release(self.q4_swiglu_bf16_scale_bias_pipeline);
        release(self.q4_mmap_pipeline);
        release(self.q4_mmap_bf16_scale_bias_pipeline);
        release(self.q4_mmap_batch_pipeline);
        release(self.q4_mmap_batch_bf16_scale_bias_pipeline);
        release(self.q4_mmap_multilinear_bf16_scale_bias_pipeline);
        release(self.glm_mla_prepare_query_kv_pipeline);
        release(self.glm_mla_absorbed_scores_pipeline);
        release(self.glm_mla_softmax_pipeline);
        release(self.glm_mla_context_pipeline);
        release(self.dense_mmap_bf16_pipeline);
        release(self.dense_mmap_f16_pipeline);
        release(self.dense_mmap_f32_pipeline);
        release(self.dense_matrix_bf16_pipeline);
        release(self.dense_matrix_f16_pipeline);
        release(self.dense_matrix_f32_pipeline);
        release(self.rms_norm_reduced_pipeline);
        release(self.residual_rms_norm_pipeline);
        release(self.attention_pipeline);
        release(self.qwen_causal_attention_rows_pipeline);
        release(self.expert_mlp_pipeline);
        release(self.silu_product_pipeline);
        release(self.shared_expert_activation_pipeline);
        release(self.combine_expert_phase_pipeline);
        release(self.qwen_layer_major_gather_pipeline);
        release(self.qwen_layer_major_combine_pipeline);
        release(self.fill_zero_pipeline);
        release(self.topk_vocab_pipeline);
        release(self.linear_conv1d_bf16_pipeline);
        release(self.linear_conv1d_f16_pipeline);
        release(self.linear_conv1d_f32_pipeline);
        release(self.linear_rms_norm_qk_pipeline);
        release(self.linear_decay_beta_bf16_pipeline);
        release(self.linear_decay_beta_f16_pipeline);
        release(self.linear_decay_beta_f32_pipeline);
        release(self.linear_delta_step_pipeline);
        release(self.linear_gated_rms_norm_bf16_pipeline);
        release(self.linear_gated_rms_norm_f16_pipeline);
        release(self.linear_gated_rms_norm_f32_pipeline);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
struct OwnedMetalObject(MetalObjcId);

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl OwnedMetalObject {
    fn new(id: MetalObjcId) -> anyhow::Result<Self> {
        if id.is_null() {
            anyhow::bail!("failed to create required Flash-MoE Metal object");
        }
        Ok(Self(id))
    }

    fn id(&self) -> MetalObjcId {
        self.0
    }

    fn into_raw(mut self) -> MetalObjcId {
        let id = self.0;
        self.0 = ptr::null_mut();
        id
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for OwnedMetalObject {
    fn drop(&mut self) {
        unsafe { release(self.0) }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
struct MetalCommandLease {
    resources: Arc<MetalResourceLedger>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCommandLease {
    fn new(resources: Arc<MetalResourceLedger>) -> Self {
        resources.command_started();
        Self { resources }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalCommandLease {
    fn drop(&mut self) {
        self.resources.command_finished();
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
struct MetalCommandEncoding {
    command_buffer: MetalObjcId,
    encoder: MetalObjcId,
    ended: bool,
    command_lease: Option<MetalCommandLease>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalCommandEncoding {
    unsafe fn new(
        command_queue: MetalObjcId,
        resources: Arc<MetalResourceLedger>,
        command_buffer_error: &'static str,
        encoder_error: &'static str,
    ) -> anyhow::Result<Self> {
        unsafe {
            // Every encoded resource is held explicitly until completion (or transferred to a
            // deferred submission), so the command buffer does not need to retain it again.
            let command_buffer = retain_autoreleased_return_value(msg_send_id0(
                command_queue,
                sel("commandBufferWithUnretainedReferences"),
            ));
            if command_buffer.is_null() {
                anyhow::bail!(command_buffer_error);
            }
            let encoder = retain_autoreleased_return_value(msg_send_id0(
                command_buffer,
                sel("computeCommandEncoder"),
            ));
            if encoder.is_null() {
                release(command_buffer);
                anyhow::bail!(encoder_error);
            }
            Ok(Self {
                command_buffer,
                encoder,
                ended: false,
                command_lease: Some(MetalCommandLease::new(resources)),
            })
        }
    }

    fn command_buffer(&self) -> MetalObjcId {
        self.command_buffer
    }

    fn encoder(&self) -> MetalObjcId {
        self.encoder
    }

    unsafe fn end_encoding(&mut self) {
        unsafe {
            if !self.ended {
                msg_send_void0(self.encoder, sel("endEncoding"));
                self.ended = true;
            }
        }
    }

    unsafe fn into_command_buffer(mut self) -> (MetalObjcId, MetalCommandLease) {
        unsafe {
            self.end_encoding();
            release(self.encoder);
            self.encoder = ptr::null_mut();
            let command_buffer = self.command_buffer;
            self.command_buffer = ptr::null_mut();
            let command_lease = self
                .command_lease
                .take()
                .expect("Metal command encoding is missing its resource lease");
            (command_buffer, command_lease)
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalCommandEncoding {
    fn drop(&mut self) {
        unsafe {
            if !self.encoder.is_null() {
                self.end_encoding();
                release(self.encoder);
            }
            release(self.command_buffer);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalRuntime {
    pub(crate) device: MetalObjcId,
    pub(crate) command_queue: MetalObjcId,
    pub(crate) pipelines: MetalPipelineSet<MetalObjcId>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalRuntime {
    pub(crate) fn compile(
        shader_source: &str,
        names: MetalPipelineNameSet,
    ) -> anyhow::Result<Self> {
        unsafe {
            let device = OwnedMetalObject::new(metal_default_device()).map_err(|_| {
                anyhow::anyhow!(
                    "Metal is required for Flash-MoE on ARM macOS, but no default Metal device is available"
                )
            })?;
            let source = OwnedMetalObject::new(ns_string(shader_source))?;
            let mut compile_error = ptr::null_mut();
            let library_id = msg_send_id2_id_error(
                device.id(),
                sel("newLibraryWithSource:options:error:"),
                source.id(),
                ptr::null_mut(),
                &mut compile_error,
            );
            if library_id.is_null() {
                let error = ns_error_localized_description(compile_error)
                    .unwrap_or_else(|| "unknown Metal compiler error".to_string());
                anyhow::bail!("failed to compile Flash-MoE Metal shader library: {error}");
            }
            let library = OwnedMetalObject::new(library_id)?;
            let mut compiled = BTreeMap::new();
            for name in names.kernel_names() {
                compiled.insert(
                    name,
                    OwnedMetalObject::new(compile_pipeline(device.id(), library.id(), name)?)?,
                );
            }
            let command_queue =
                OwnedMetalObject::new(msg_send_id0(device.id(), sel("newCommandQueue"))).map_err(
                    |_| anyhow::anyhow!("failed to create Flash-MoE Metal command queue"),
                )?;
            let mut take_pipeline = |name: &'static str| -> MetalObjcId {
                compiled
                    .remove(name)
                    .expect("compiled Metal pipeline name disappeared")
                    .into_raw()
            };
            let pipelines = MetalPipelineSet {
                q4_pipeline: take_pipeline(names.q4),
                q4_bf16_scale_bias_pipeline: take_pipeline(names.q4_bf16_scale_bias),
                mxfp4_e8m0_pipeline: take_pipeline(names.mxfp4_e8m0),
                q4_swiglu_pipeline: take_pipeline(names.q4_swiglu),
                q4_swiglu_bf16_scale_bias_pipeline: take_pipeline(names.q4_swiglu_bf16_scale_bias),
                q4_mmap_pipeline: take_pipeline(names.q4_mmap),
                q4_mmap_bf16_scale_bias_pipeline: take_pipeline(names.q4_mmap_bf16_scale_bias),
                q4_mmap_batch_pipeline: take_pipeline(names.q4_mmap_batch),
                q4_mmap_batch_bf16_scale_bias_pipeline: take_pipeline(
                    names.q4_mmap_batch_bf16_scale_bias,
                ),
                q4_mmap_multilinear_bf16_scale_bias_pipeline: take_pipeline(
                    names.q4_mmap_multilinear_bf16_scale_bias,
                ),
                glm_mla_prepare_query_kv_pipeline: take_pipeline(names.glm_mla_prepare_query_kv),
                glm_mla_absorbed_scores_pipeline: take_pipeline(names.glm_mla_absorbed_scores),
                glm_mla_softmax_pipeline: take_pipeline(names.glm_mla_softmax),
                glm_mla_context_pipeline: take_pipeline(names.glm_mla_context),
                dense_mmap_bf16_pipeline: take_pipeline(names.dense_mmap_bf16),
                dense_mmap_f16_pipeline: take_pipeline(names.dense_mmap_f16),
                dense_mmap_f32_pipeline: take_pipeline(names.dense_mmap_f32),
                dense_matrix_bf16_pipeline: take_pipeline(names.dense_matrix_bf16),
                dense_matrix_f16_pipeline: take_pipeline(names.dense_matrix_f16),
                dense_matrix_f32_pipeline: take_pipeline(names.dense_matrix_f32),
                rms_norm_reduced_pipeline: take_pipeline(names.rms_norm_reduced),
                residual_rms_norm_pipeline: take_pipeline(names.residual_rms_norm),
                attention_pipeline: take_pipeline(names.attention),
                qwen_causal_attention_rows_pipeline: take_pipeline(
                    names.qwen_causal_attention_rows,
                ),
                expert_mlp_pipeline: take_pipeline(names.expert_mlp),
                silu_product_pipeline: take_pipeline(names.silu_product),
                shared_expert_activation_pipeline: take_pipeline(names.shared_expert_activation),
                combine_expert_phase_pipeline: take_pipeline(names.combine_expert_phase),
                qwen_layer_major_gather_pipeline: take_pipeline(names.qwen_layer_major_gather),
                qwen_layer_major_combine_pipeline: take_pipeline(names.qwen_layer_major_combine),
                fill_zero_pipeline: take_pipeline(names.fill_zero),
                topk_vocab_pipeline: take_pipeline(names.topk_vocab),
                linear_conv1d_bf16_pipeline: take_pipeline(names.linear_conv1d_bf16),
                linear_conv1d_f16_pipeline: take_pipeline(names.linear_conv1d_f16),
                linear_conv1d_f32_pipeline: take_pipeline(names.linear_conv1d_f32),
                linear_rms_norm_qk_pipeline: take_pipeline(names.linear_rms_norm_qk),
                linear_decay_beta_bf16_pipeline: take_pipeline(names.linear_decay_beta_bf16),
                linear_decay_beta_f16_pipeline: take_pipeline(names.linear_decay_beta_f16),
                linear_decay_beta_f32_pipeline: take_pipeline(names.linear_decay_beta_f32),
                linear_delta_step_pipeline: take_pipeline(names.linear_delta_step),
                linear_gated_rms_norm_bf16_pipeline: take_pipeline(
                    names.linear_gated_rms_norm_bf16,
                ),
                linear_gated_rms_norm_f16_pipeline: take_pipeline(names.linear_gated_rms_norm_f16),
                linear_gated_rms_norm_f32_pipeline: take_pipeline(names.linear_gated_rms_norm_f32),
            };
            debug_assert!(compiled.is_empty());
            Ok(Self {
                device: device.into_raw(),
                command_queue: command_queue.into_raw(),
                pipelines,
            })
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalRuntime {
    fn drop(&mut self) {
        unsafe {
            self.pipelines.release_with(|pipeline| release(pipeline));
            release(self.command_queue);
            release(self.device);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct DeepSeekMetalPipelineSet {
    pipelines: BTreeMap<&'static str, MetalObjcId>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe impl Send for DeepSeekMetalPipelineSet {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe impl Sync for DeepSeekMetalPipelineSet {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl DeepSeekMetalPipelineSet {
    unsafe fn compile(device: MetalObjcId) -> anyhow::Result<Self> {
        unsafe {
            let source = OwnedMetalObject::new(ns_string(DEEPSEEK_V4_METAL_SHADERS))?;
            let mut compile_error = ptr::null_mut();
            let library_id = msg_send_id2_id_error(
                device,
                sel("newLibraryWithSource:options:error:"),
                source.id(),
                ptr::null_mut(),
                &mut compile_error,
            );
            if library_id.is_null() {
                let error = ns_error_localized_description(compile_error)
                    .unwrap_or_else(|| "unknown Metal compiler error".to_string());
                anyhow::bail!(
                    "failed to compile required DeepSeek V4 Flash Metal shader library: {error}"
                );
            }
            let library = OwnedMetalObject::new(library_id)?;
            let mut pipelines = BTreeMap::new();
            for &name in DEEPSEEK_V4_REQUIRED_METAL_KERNELS {
                let constants: &[(u64, u64, &[u8])] = match name {
                    "kernel_mul_mv_q8_0_f32"
                    | "kernel_mul_mv_f16_f32"
                    | "kernel_dsv4_shared_gate_up_swiglu_q8_0"
                    | "kernel_dsv4_shared_down_hc_expand4_q8_0"
                    | "kernel_dsv4_q8_hc_expand4_q8_0" => &[(600, 37, &4i16.to_ne_bytes())],
                    "kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32"
                    | "kernel_mul_mv_slots6_q2_K_sum6_f32" => &[(600, 37, &2i16.to_ne_bytes())],
                    "kernel_sum_rows_f32_f32" => &[(1400, 37, &10i16.to_ne_bytes())],
                    "kernel_mul_mm_q8_0_f32" | "kernel_mul_mm_f16_f32" => {
                        &[(700, 53, &[0]), (701, 53, &[1])]
                    }
                    "kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16" | "kernel_mul_mm_id_q2_K_f16" => {
                        &[(700, 53, &[0])]
                    }
                    "kernel_flash_attn_ext_pad" => {
                        &[(100, 53, &[1]), (125, 29, &64i32.to_ne_bytes())]
                    }
                    "kernel_flash_attn_ext_blk" => &[
                        (224, 29, &8i32.to_ne_bytes()),
                        (225, 29, &64i32.to_ne_bytes()),
                    ],
                    "kernel_flash_attn_ext_f16_dk512_dv512" => &[
                        (300, 53, &[1]),
                        (301, 53, &[1]),
                        (302, 53, &[0]),
                        (303, 53, &[0]),
                        (304, 53, &[1]),
                        (310, 53, &[1]),
                        (320, 29, &512i32.to_ne_bytes()),
                        (321, 29, &512i32.to_ne_bytes()),
                        (322, 29, &8i32.to_ne_bytes()),
                    ],
                    _ => &[],
                };
                let pipeline = OwnedMetalObject::new(if constants.is_empty() {
                    compile_pipeline(device, library.id(), name)?
                } else {
                    compile_pipeline_with_constants(device, library.id(), name, constants)?
                })?;
                pipelines.insert(name, pipeline.into_raw());
            }
            let no_pad_constants: &[(u64, u64, &[u8])] = &[
                (300, 53, &[1]),
                (301, 53, &[1]),
                (302, 53, &[0]),
                (303, 53, &[0]),
                (304, 53, &[0]),
                (310, 53, &[1]),
                (320, 29, &512i32.to_ne_bytes()),
                (321, 29, &512i32.to_ne_bytes()),
                (322, 29, &8i32.to_ne_bytes()),
            ];
            let no_pad = OwnedMetalObject::new(compile_pipeline_with_constants(
                device,
                library.id(),
                "kernel_flash_attn_ext_f16_dk512_dv512",
                no_pad_constants,
            )?)?;
            pipelines.insert(
                "kernel_flash_attn_ext_f16_dk512_dv512_nopad",
                no_pad.into_raw(),
            );
            Ok(Self { pipelines })
        }
    }

    pub(crate) fn require(&self, name: &'static str) -> anyhow::Result<MetalObjcId> {
        self.pipelines.get(name).copied().with_context(|| {
            format!("DeepSeek V4 Flash Metal graph is missing compiled kernel {name}")
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for DeepSeekMetalPipelineSet {
    fn drop(&mut self) {
        unsafe {
            for pipeline in self.pipelines.values().copied() {
                release(pipeline);
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalExecutionContext {
    runtime: MetalRuntime,
    deepseek_pipelines: Option<DeepSeekMetalPipelineSet>,
    dense_weights: Option<MetalDenseWeights>,
    linear_attention_state: Mutex<MetalLinearAttentionStateCache>,
    deepseek_state: Mutex<Option<deepseek_execution::DeepSeekV4MetalState>>,
    buffers: Arc<MetalBufferPool>,
    resources: Arc<MetalResourceLedger>,
    norm_epsilon: f32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe impl Send for MetalExecutionContext {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe impl Sync for MetalExecutionContext {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalExecutionContext {
    fn drop(&mut self) {
        unsafe {
            if let Some(dense_weights) = self.dense_weights.take() {
                release(dense_weights.buffer);
            }
            if let Ok(linear_state) = self.linear_attention_state.get_mut() {
                release_linear_attention_state(linear_state);
            }
            if let Ok(deepseek_state) = self.deepseek_state.get_mut()
                && let Some(mut state) = deepseek_state.take()
            {
                state.release();
            }
            self.resources.record_resident_resources(0, 0);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalExecutionContext {
    #[allow(dead_code)]
    pub(crate) fn compile(
        dense_mmap: Arc<memmap2::Mmap>,
        dense_len: u64,
        linear_layouts: &[Option<LinearAttentionLayout>],
        norm_epsilon: f32,
    ) -> anyhow::Result<Self> {
        Self::compile_resolved(dense_mmap, dense_len, linear_layouts, norm_epsilon, false)
    }

    pub(crate) fn compile_resolved(
        dense_mmap: Arc<memmap2::Mmap>,
        dense_len: u64,
        linear_layouts: &[Option<LinearAttentionLayout>],
        norm_epsilon: f32,
        deepseek_v4_flash: bool,
    ) -> anyhow::Result<Self> {
        let runtime = MetalRuntime::compile(METAL_SHADERS, MetalPipelineNameSet::new())?;
        let deepseek_pipelines = deepseek_v4_flash
            .then(|| unsafe { DeepSeekMetalPipelineSet::compile(runtime.device) })
            .transpose()?;
        let resources = Arc::new(unsafe { MetalResourceLedger::from_device(runtime.device) });
        let dense_weights = wrap_dense_mmap_as_metal_buffer(runtime.device, dense_mmap, dense_len)?;
        let linear_attention_state =
            allocate_linear_attention_state(runtime.device, linear_layouts)?;
        resources.record_resident_resources(
            dense_weights.as_ref().map_or(0, |weights| weights.len),
            linear_attention_state_bytes(&linear_attention_state),
        );
        unsafe {
            resources.sample_device(runtime.device, false);
        }
        Ok(Self {
            runtime,
            deepseek_pipelines,
            dense_weights,
            linear_attention_state: Mutex::new(linear_attention_state),
            deepseek_state: Mutex::new(None),
            buffers: Arc::new(MetalBufferPool::new(Arc::clone(&resources))),
            resources,
            norm_epsilon,
        })
    }

    pub(crate) fn runtime_capabilities(&self) -> MetalRuntimeCapabilities {
        let capabilities =
            MetalRuntimeCapabilities::from_pipeline_names(MetalPipelineNameSet::new());
        if self.deepseek_pipelines.is_some() {
            capabilities.with_additional(DEEPSEEK_V4_REQUIRED_METAL_KERNELS)
        } else {
            capabilities
        }
    }

    pub(crate) fn deepseek_pipelines(&self) -> anyhow::Result<&DeepSeekMetalPipelineSet> {
        self.deepseek_pipelines
            .as_ref()
            .context("DeepSeek V4 Flash Metal pipelines were not selected by the load-time graph")
    }

    pub(crate) fn runtime(&self) -> &MetalRuntime {
        &self.runtime
    }

    pub(crate) fn dense_weights(&self) -> Option<&MetalDenseWeights> {
        self.dense_weights.as_ref()
    }

    pub(crate) fn buffers(&self) -> &Arc<MetalBufferPool> {
        &self.buffers
    }

    pub(crate) fn norm_epsilon(&self) -> f32 {
        self.norm_epsilon
    }

    pub(crate) fn set_working_set_limit_bytes(&self, limit: usize) -> anyhow::Result<()> {
        self.resources.set_working_set_limit_bytes(limit)
    }

    pub(crate) fn resource_snapshot(&self) -> FlashMoeMetalResourceSnapshot {
        unsafe {
            self.resources.sample_device(self.runtime.device, false);
        }
        self.resources.snapshot()
    }

    pub(crate) fn prepare_resident_expert_backing(
        &self,
        bytes: &ReusableExpertBytes,
    ) -> anyhow::Result<()> {
        let buffer = unsafe {
            persistent_expert_source_buffer(
                self.runtime.device,
                bytes.as_slice(),
                bytes,
                self.buffers.as_ref(),
            )?
        };
        if buffer.is_none() {
            anyhow::bail!(
                "resident expert backing is not page-aligned for a persistent no-copy Metal buffer: bytes={}",
                bytes.len()
            );
        }
        Ok(())
    }

    pub(crate) fn finish_token_boundary(&self, position: usize) -> anyhow::Result<()> {
        let current = unsafe { self.resources.sample_device(self.runtime.device, true) };
        let mut snapshot = self.resources.snapshot();
        if snapshot.active_general_buffers != 0
            || snapshot.transient_expert_buffers != 0
            || snapshot.in_flight_commands != 0
        {
            anyhow::bail!(
                "FlashMoe Metal resource ownership imbalance at token boundary: position={position} active_general_buffers={} active_general_bytes={} pooled_buffers={} pooled_bytes={} transient_expert_buffers={} transient_expert_bytes={} in_flight_commands={} buffer_allocations={} buffer_reuses={} buffer_recycles={} buffer_releases={} phase_cleanup_calls={} phase_cleanup_buffers={}",
                snapshot.active_general_buffers,
                snapshot.active_general_bytes,
                snapshot.pooled_buffers,
                snapshot.pooled_bytes,
                snapshot.transient_expert_buffers,
                snapshot.transient_expert_bytes,
                snapshot.in_flight_commands,
                snapshot.buffer_allocations,
                snapshot.buffer_reuses,
                snapshot.buffer_recycles,
                snapshot.buffer_releases,
                snapshot.phase_cleanup_calls,
                snapshot.phase_cleanup_buffers,
            );
        }
        if current <= snapshot.working_set_limit_bytes {
            return Ok(());
        }

        let (released_pooled_buffers, released_pooled_bytes) = self.buffers.release_idle_buffers();
        let after_drain = unsafe { self.resources.sample_device(self.runtime.device, false) };
        snapshot = self.resources.snapshot();
        if after_drain > snapshot.working_set_limit_bytes {
            self.resources.record_resource_limit_abort();
            anyhow::bail!(
                "FlashMoe Metal resource limit exceeded at token boundary: position={position} current_allocated_bytes={after_drain} working_set_limit_bytes={} recommended_working_set_bytes={} driver_high_water_bytes={} released_pooled_buffers={released_pooled_buffers} released_pooled_bytes={released_pooled_bytes} ledger_live_bytes={}",
                snapshot.working_set_limit_bytes,
                snapshot.recommended_working_set_bytes,
                snapshot.driver_high_water_bytes,
                snapshot.ledger_live_bytes,
            );
        }
        self.resources.record_pressure_recovery();
        tracing::warn!(
            position,
            current_allocated_bytes = current,
            current_after_drain_bytes = after_drain,
            released_pooled_buffers,
            released_pooled_bytes,
            "FlashMoe Metal token boundary recovered from working-set pressure"
        );
        Ok(())
    }

    pub(crate) fn has_resident_dense_weights(&self) -> bool {
        self.dense_weights.is_some()
    }

    pub(crate) fn reset_linear_attention_state(&self) -> anyhow::Result<()> {
        let state = self.linear_attention_state.lock().map_err(|_| {
            anyhow::anyhow!("FlashMoe Metal linear-attention state lock is poisoned during reset")
        })?;
        unsafe {
            for layer in state.layers.iter().flatten() {
                zero_buffer(layer.conv_state, layer.conv_state_len);
                zero_buffer(layer.ssm_state, layer.ssm_state_len);
                zero_buffer(layer.conv_output, layer.conv_dim);
                zero_buffer(layer.delta_output, layer.total_value_width);
                zero_buffer(layer.g_decay, layer.num_value_heads);
                zero_buffer(layer.beta_gate, layer.num_value_heads);
            }
        }
        Ok(())
    }

    pub(crate) fn capture_linear_attention_session_state(
        &self,
    ) -> anyhow::Result<FlashMoeLinearAttentionSessionSnapshot> {
        let state = self.linear_attention_state.lock().map_err(|_| {
            anyhow::anyhow!(
                "FlashMoe Metal linear-attention state lock is poisoned during session capture"
            )
        })?;
        let readback_values = state
            .layers
            .iter()
            .flatten()
            .map(|layer| layer.conv_state_len.saturating_add(layer.ssm_state_len))
            .fold(0usize, usize::saturating_add);
        let snapshot = capture_linear_attention_session_snapshot(&state)?;
        self.buffers
            .resources
            .record_host_readback(readback_values.saturating_mul(std::mem::size_of::<f32>()));
        Ok(snapshot)
    }

    pub(crate) fn restore_linear_attention_session_state(
        &self,
        snapshot: &FlashMoeLinearAttentionSessionSnapshot,
    ) -> anyhow::Result<()> {
        let state = self.linear_attention_state.lock().map_err(|_| {
            anyhow::anyhow!(
                "FlashMoe Metal linear-attention state lock is poisoned during session restore"
            )
        })?;
        restore_linear_attention_session_snapshot(&state, snapshot)
    }

    pub(crate) fn resident_top_candidates(
        &self,
        projection: &ResidentMmapMatvecProjection,
        input: &[f32],
        output_rows: usize,
        top_k: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let dense_weights = self.dense_weights.as_ref().context(
            "FlashMoe unsupported resident topK path: resident dense Metal weights are unavailable",
        )?;
        MetalResidentTopKBuilder::new(
            self.runtime.device,
            self.runtime.command_queue,
            &self.runtime.pipelines,
            dense_weights,
            &self.buffers,
        )
        .execute(projection, input, output_rows, top_k)
    }

    pub(crate) fn resident_post_attention_prep_topk(
        &self,
        projections: &Cmd2ResidentPostAttentionPrepProjections,
        attention_output: &[f32],
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        router_correction_bias: Option<&[f32]>,
    ) -> anyhow::Result<MetalPostAttentionPrep> {
        let dense_weights = self.dense_weights.as_ref().context(
            "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: resident dense Metal weights are unavailable",
        )?;
        MetalResidentPostAttentionPrepBuilder::new(
            self.runtime.device,
            self.runtime.command_queue,
            &self.runtime.pipelines,
            dense_weights,
            &self.buffers,
            self.norm_epsilon,
        )
        .execute(
            projections,
            attention_output,
            residual,
            post_norm_weight,
            router_correction_bias,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_attention_post_attention_prep(
        &self,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        input: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        top_k: usize,
    ) -> anyhow::Result<MetalPostAttentionPrep> {
        MetalFusedLinearAttentionBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.linear_attention_state,
            &self.buffers,
        )
        .execute(layout, bindings, input, residual, post_norm_weight, top_k)
    }

    pub(crate) fn qwen_linear_attention_matrix(
        &self,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        rows: usize,
        qkv: &[f32],
        z: &[f32],
        beta: &[f32],
        alpha: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        MetalFusedLinearAttentionBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.linear_attention_state,
            &self.buffers,
        )
        .execute_recurrent_matrix(layout, bindings, rows, qkv, z, beta, alpha)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen_post_attention_matrix(
        &self,
        out_proj: &ResidentMmapMatvecProjection,
        router: &ResidentMmapMatvecProjection,
        rows: usize,
        attention_width: usize,
        width: usize,
        attention: &[f32],
        residual: &[f32],
        post_norm_weight: &[f32],
    ) -> anyhow::Result<MetalLayerMajorPostAttention> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_post_attention_matrix(
            out_proj,
            router,
            rows,
            attention_width,
            width,
            attention,
            residual,
            post_norm_weight,
            self.norm_epsilon,
        )?
        .context("Qwen layer-major post-attention matrix requires resident dense weights")
    }

    pub(crate) fn qwen_layer_major_experts(
        &self,
        scheduled: &ScheduledLayerMajorExperts,
        post_attention: &MetalLayerMajorPostAttention,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
    ) -> anyhow::Result<MetalQwenPrefillLayerOutput> {
        let dense_weights = self
            .dense_weights
            .as_ref()
            .context("Qwen layer-major expert graph requires resident dense Metal weights")?;
        MetalScheduledCmd3Builder::new(
            &self.runtime,
            dense_weights,
            Arc::clone(&self.buffers),
            self.norm_epsilon,
        )
        .execute_layer_major(scheduled, post_attention, shared, next_norm_weight)
    }

    #[cfg(test)]
    pub(crate) fn qwen_rms_norm_rows(
        &self,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        width: usize,
    ) -> anyhow::Result<Vec<f32>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_rms_norm_rows(input, weight, rows, width, self.norm_epsilon)
    }

    pub(crate) fn resident_projection_batch(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input: &[f32],
    ) -> anyhow::Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute(projections, input)
    }

    pub(crate) fn resident_projection_matrix_batch(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input_rows: usize,
        input_cols: usize,
        input: &[f32],
    ) -> anyhow::Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_matrix(projections, input_rows, input_cols, input)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen_causal_attention_rows(
        &self,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        query_rows: usize,
        prefix_rows: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let query_width = query_heads
            .checked_mul(head_dim)
            .context("Qwen attention query width overflow")?;
        let kv_width = kv_heads
            .checked_mul(head_dim)
            .context("Qwen attention KV width overflow")?;
        let key_rows = prefix_rows
            .checked_add(query_rows)
            .context("Qwen attention key row count overflow")?;
        if query_rows == 0
            || query_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || head_dim > 256
            || !query_heads.is_multiple_of(kv_heads)
            || queries.len() != query_rows.saturating_mul(query_width)
            || keys.len() != key_rows.saturating_mul(kv_width)
            || values.len() != key_rows.saturating_mul(kv_width)
        {
            anyhow::bail!(
                "Qwen causal attention matrix has incompatible geometry: queries={} keys={} values={} rows={} prefix={} q_heads={} kv_heads={} head_dim={}",
                queries.len(),
                keys.len(),
                values.len(),
                query_rows,
                prefix_rows,
                query_heads,
                kv_heads,
                head_dim
            );
        }
        let query_rows_u32 = u32::try_from(query_rows)?;
        let prefix_rows_u32 = u32::try_from(prefix_rows)?;
        let query_heads_u32 = u32::try_from(query_heads)?;
        let kv_heads_u32 = u32::try_from(kv_heads)?;
        let head_dim_u32 = u32::try_from(head_dim)?;
        unsafe {
            let mut buffers = Vec::with_capacity(4);
            let allocated = (|| -> anyhow::Result<_> {
                let query_buffer = self.buffers.tracked_buffer_with_bytes(
                    self.runtime.device,
                    f32_as_bytes(queries),
                    &mut buffers,
                )?;
                let key_buffer = self.buffers.tracked_buffer_with_bytes(
                    self.runtime.device,
                    f32_as_bytes(keys),
                    &mut buffers,
                )?;
                let value_buffer = self.buffers.tracked_buffer_with_bytes(
                    self.runtime.device,
                    f32_as_bytes(values),
                    &mut buffers,
                )?;
                let output_buffer = self.buffers.tracked_buffer_with_len(
                    self.runtime.device,
                    queries.len() * std::mem::size_of::<f32>(),
                    &mut buffers,
                )?;
                Ok((query_buffer, key_buffer, value_buffer, output_buffer))
            })();
            let (query_buffer, key_buffer, value_buffer, output_buffer) = match allocated {
                Ok(allocated) => allocated,
                Err(error) => {
                    self.buffers.recycle_or_release_phase(
                        buffers
                            .into_iter()
                            .map(MetalPhaseBuffer::recyclable)
                            .collect(),
                        true,
                    );
                    return Err(error);
                }
            };
            let mut encoding = match MetalCommandEncoding::new(
                self.runtime.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Qwen causal-attention row command buffer",
                "failed to create Qwen causal-attention row encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release_phase(
                        buffers
                            .into_iter()
                            .map(MetalPhaseBuffer::recyclable)
                            .collect(),
                        true,
                    );
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.runtime.pipelines.qwen_causal_attention_rows_pipeline,
            );
            set_buffer(encoder, query_buffer, 0);
            set_buffer(encoder, key_buffer, 1);
            set_buffer(encoder, value_buffer, 2);
            set_buffer(encoder, output_buffer, 3);
            set_bytes(encoder, u32_as_bytes(&query_rows_u32), 4);
            set_bytes(encoder, u32_as_bytes(&prefix_rows_u32), 5);
            set_bytes(encoder, u32_as_bytes(&query_heads_u32), 6);
            set_bytes(encoder, u32_as_bytes(&kv_heads_u32), 7);
            set_bytes(encoder, u32_as_bytes(&head_dim_u32), 8);
            dispatch_metal_plan(
                encoder,
                MetalDispatchPlan::qwen_attention_threadgroups(
                    query_rows as u64,
                    query_heads as u64,
                ),
            );
            encoding.end_encoding();
            let context = MetalCommandContext::new("qwen_causal_attention_rows")
                .with("query_rows", query_rows)
                .with("prefix_rows", prefix_rows)
                .with("query_heads", query_heads)
                .with("kv_heads", kv_heads)
                .with("head_dim", head_dim);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.buffers.recycle_or_release_phase(
                    buffers
                        .into_iter()
                        .map(MetalPhaseBuffer::recyclable)
                        .collect(),
                    error.should_release_buffers(),
                );
                return Err(error.into());
            }
            let output = self.buffers.read_f32_buffer(output_buffer, queries.len());
            drop(encoding);
            self.buffers.recycle_or_release_phase(
                buffers
                    .into_iter()
                    .map(MetalPhaseBuffer::recyclable)
                    .collect(),
                false,
            );
            Ok(output)
        }
    }

    pub(crate) fn resident_projection_batch_with_input_buffer(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input_buffer: MetalObjcId,
        input_len: usize,
    ) -> anyhow::Result<Option<(Vec<Vec<f32>>, MetalMatvecTiming, usize)>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_with_input_buffer(projections, input_buffer, input_len)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resident_glm_mla_input_projection_chain(
        &self,
        q_a: &ResidentMmapMatvecProjection,
        kv_a: &ResidentMmapMatvecProjection,
        q_b: &ResidentMmapMatvecProjection,
        input: MetalBatchProjectionInput<'_>,
        q_norm_weight: &[f32],
        kv_norm_weight: &[f32],
        kv_lora_rank: usize,
        norm_epsilon: f32,
    ) -> anyhow::Result<Option<(Vec<f32>, Vec<f32>)>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_glm_mla_input_projection_chain(
            q_a,
            kv_a,
            q_b,
            input,
            q_norm_weight,
            kv_norm_weight,
            kv_lora_rank,
            norm_epsilon,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resident_glm_mla_fused_attention(
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
    ) -> anyhow::Result<Option<MetalGlmMlaFusedAttentionOutput>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_glm_mla_fused_attention(
            q_a,
            kv_a,
            q_b,
            embed_q,
            unembed_out,
            input,
            q_norm_weight,
            kv_norm_weight,
            norm_epsilon,
        )
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn resident_q4_multilinear(
        &self,
        projection: &DenseQ4MmapMatvecProjection,
        heads: usize,
        rows_per_head: usize,
        inputs: &[f32],
    ) -> anyhow::Result<Option<Vec<f32>>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_q4_multilinear(projection, heads, rows_per_head, inputs)
    }

    pub(crate) fn resident_glm_mla_absorbed_attention(
        &self,
        embed_q: &DenseQ4MmapMatvecProjection,
        unembed_out: &DenseQ4MmapMatvecProjection,
        input: MetalGlmMlaAbsorbedAttentionInput<'_>,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        MetalResidentProjectionBatchBuilder::new(
            &self.runtime,
            self.dense_weights.as_ref(),
            &self.buffers,
        )
        .execute_glm_mla_absorbed_attention(embed_q, unembed_out, input)
    }

    #[cfg(test)]
    pub(crate) fn read_and_recycle_f32(&self, buffer: MetalObjcId, len: usize) -> Vec<f32> {
        unsafe {
            let values = self.buffers.read_f32_buffer(buffer, len);
            self.buffers.recycle(buffer);
            values
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn validate_linear_attention_session_snapshot(
    resident: &MetalLinearAttentionStateCache,
    snapshot: &FlashMoeLinearAttentionSessionSnapshot,
) -> anyhow::Result<()> {
    if snapshot.len() != resident.layers.len() {
        anyhow::bail!(
            "FlashMoe Metal recurrent session snapshot has {} layers, expected {}",
            snapshot.len(),
            resident.layers.len()
        );
    }
    for (layer, resident) in resident.layers.iter().enumerate() {
        match (resident, snapshot.layer(layer)) {
            (None, None) => {}
            (Some(resident), Some(snapshot)) => {
                let declared = snapshot.state();
                if declared.layer() != layer
                    || declared.conv_state_len() != resident.conv_state_len
                    || declared.ssm_state_len() != resident.ssm_state_len
                    || declared.conv_output_len() != resident.conv_dim
                    || declared.output_len() != resident.total_value_width
                {
                    anyhow::bail!(
                        "FlashMoe Metal recurrent session snapshot for layer {layer} does not match the resolved resident state"
                    );
                }
            }
            (Some(_), None) => {
                anyhow::bail!(
                    "FlashMoe Metal recurrent session snapshot is missing resolved linear-attention layer {layer}"
                );
            }
            (None, Some(_)) => {
                anyhow::bail!(
                    "FlashMoe Metal recurrent session snapshot unexpectedly contains full-attention layer {layer}"
                );
            }
        }
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn capture_linear_attention_session_snapshot(
    resident: &MetalLinearAttentionStateCache,
) -> anyhow::Result<FlashMoeLinearAttentionSessionSnapshot> {
    let layers = resident
        .layers
        .iter()
        .enumerate()
        .map(|(layer, state)| {
            state
                .as_ref()
                .map(|state| unsafe {
                    FlashMoeLinearAttentionLayerSnapshot::new(
                        layer,
                        read_f32_buffer(state.conv_state, state.conv_state_len),
                        read_f32_buffer(state.ssm_state, state.ssm_state_len),
                        state.conv_dim,
                        state.total_value_width,
                    )
                })
                .transpose()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    FlashMoeLinearAttentionSessionSnapshot::new(layers)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn restore_linear_attention_session_snapshot(
    resident: &MetalLinearAttentionStateCache,
    snapshot: &FlashMoeLinearAttentionSessionSnapshot,
) -> anyhow::Result<()> {
    validate_linear_attention_session_snapshot(resident, snapshot)?;
    for (layer, resident) in resident.layers.iter().enumerate() {
        if let (Some(resident), Some(snapshot)) = (resident, snapshot.layer(layer)) {
            unsafe {
                write_f32_buffer(resident.conv_state, snapshot.conv_state());
                write_f32_buffer(resident.ssm_state, snapshot.ssm_state());
                zero_buffer(resident.conv_output, resident.conv_dim);
                zero_buffer(resident.delta_output, resident.total_value_width);
                zero_buffer(resident.g_decay, resident.num_value_heads);
                zero_buffer(resident.beta_gate, resident.num_value_heads);
            }
        }
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalResidentTopKBuilder<'a> {
    device: MetalObjcId,
    command_queue: MetalObjcId,
    pipelines: &'a MetalPipelineSet<MetalObjcId>,
    topk_pipeline: MetalObjcId,
    dense_weights: &'a MetalDenseWeights,
    buffers: &'a MetalBufferPool,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<'a> MetalResidentTopKBuilder<'a> {
    pub(crate) fn new(
        device: MetalObjcId,
        command_queue: MetalObjcId,
        pipelines: &'a MetalPipelineSet<MetalObjcId>,
        dense_weights: &'a MetalDenseWeights,
        buffers: &'a MetalBufferPool,
    ) -> Self {
        Self {
            device,
            command_queue,
            pipelines,
            topk_pipeline: pipelines.topk_vocab_pipeline,
            dense_weights,
            buffers,
        }
    }

    pub(crate) fn execute(
        &self,
        projection: &ResidentMmapMatvecProjection,
        input: &[f32],
        output_rows: usize,
        top_k: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        if output_rows == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        if output_rows > projection.rows() {
            anyhow::bail!(
                "Metal resident topK output width {} exceeds tensor rows {} for {}",
                output_rows,
                projection.rows(),
                projection.tensor_name()
            );
        }
        validate_resident_projection(projection, input.len(), self.dense_weights.len)?;

        let top_k = top_k.min(output_rows).max(1);
        unsafe { self.encode_and_read(projection, input, output_rows, top_k) }
    }

    unsafe fn transient_buffer(
        &self,
        len: usize,
        owned: &mut Vec<MetalObjcId>,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            match self.buffers.buffer_with_len(self.device, len) {
                Ok(buffer) => {
                    owned.push(buffer);
                    Ok(buffer)
                }
                Err(error) => {
                    self.buffers.recycle_or_release(owned, true);
                    owned.clear();
                    Err(error)
                }
            }
        }
    }

    unsafe fn encode_and_read(
        &self,
        projection: &ResidentMmapMatvecProjection,
        input: &[f32],
        rows: usize,
        top_k: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        unsafe {
            let mut transient = Vec::with_capacity(4);
            let input_buffer =
                self.transient_buffer(std::mem::size_of_val(input), &mut transient)?;
            let input_contents = msg_send_ptr0(input_buffer, sel("contents"));
            ptr::copy_nonoverlapping(
                input.as_ptr().cast::<u8>(),
                input_contents.cast::<u8>(),
                std::mem::size_of_val(input),
            );
            let logits_buffer =
                self.transient_buffer(rows * std::mem::size_of::<f32>(), &mut transient)?;
            let indices_buffer =
                self.transient_buffer(top_k * std::mem::size_of::<u32>(), &mut transient)?;
            let values_buffer =
                self.transient_buffer(top_k * std::mem::size_of::<f32>(), &mut transient)?;

            let constants = (|| -> anyhow::Result<_> {
                Ok((
                    u32::try_from(rows).context("resident topK rows exceed u32")?,
                    u32::try_from(top_k).context("resident topK count exceeds u32")?,
                ))
            })();
            let (rows_u32, top_k_u32) = match constants {
                Ok(constants) => constants,
                Err(error) => {
                    self.buffers.recycle_or_release(&transient, true);
                    return Err(error);
                }
            };

            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE resident topK command buffer",
                "failed to create Flash-MoE resident topK compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release(&transient, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();

            if let Err(error) = encode_resident_projection_rows(
                self.pipelines,
                encoder,
                self.dense_weights,
                projection,
                rows,
                input_buffer,
                logits_buffer,
                0,
            ) {
                drop(encoding);
                self.buffers.recycle_or_release(&transient, true);
                return Err(error);
            }

            msg_send_void1_id(encoder, sel("setComputePipelineState:"), self.topk_pipeline);
            set_buffer(encoder, logits_buffer, 0);
            set_buffer(encoder, indices_buffer, 1);
            set_buffer(encoder, values_buffer, 2);
            set_bytes(encoder, u32_as_bytes(&rows_u32), 3);
            set_bytes(encoder, u32_as_bytes(&top_k_u32), 4);
            dispatch_threads(encoder, 1);
            encoding.end_encoding();

            let context = MetalCommandContext::new("resident_topk")
                .with("tensor", projection.tensor_name())
                .with("rows", rows)
                .with("physical_rows", projection.rows())
                .with("cols", projection.cols())
                .with("top_k", top_k);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.buffers
                    .recycle_or_release(&transient, error.should_release_buffers());
                return Err(error.into());
            }

            let indices_ptr = msg_send_ptr0(indices_buffer, sel("contents")).cast::<u32>();
            let values_ptr = msg_send_ptr0(values_buffer, sel("contents")).cast::<f32>();
            let candidates = (0..top_k)
                .map(|index| (*indices_ptr.add(index) as usize, *values_ptr.add(index)))
                .collect();

            drop(encoding);
            self.buffers.recycle_or_release(&transient, false);
            Ok(candidates)
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalResidentPostAttentionPrepBuilder<'a> {
    device: MetalObjcId,
    command_queue: MetalObjcId,
    pipelines: &'a MetalPipelineSet<MetalObjcId>,
    residual_rms_norm_pipeline: MetalObjcId,
    dense_weights: &'a MetalDenseWeights,
    buffers: &'a MetalBufferPool,
    norm_epsilon: f32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<'a> MetalResidentPostAttentionPrepBuilder<'a> {
    pub(crate) fn new(
        device: MetalObjcId,
        command_queue: MetalObjcId,
        pipelines: &'a MetalPipelineSet<MetalObjcId>,
        dense_weights: &'a MetalDenseWeights,
        buffers: &'a MetalBufferPool,
        norm_epsilon: f32,
    ) -> Self {
        Self {
            device,
            command_queue,
            pipelines,
            residual_rms_norm_pipeline: pipelines.residual_rms_norm_pipeline,
            dense_weights,
            buffers,
            norm_epsilon,
        }
    }

    pub(crate) fn execute(
        &self,
        projections: &Cmd2ResidentPostAttentionPrepProjections,
        attention_output: &[f32],
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        router_correction_bias: Option<&[f32]>,
    ) -> anyhow::Result<MetalPostAttentionPrep> {
        let plan = projections.resident_plan(
            attention_output.len(),
            residual.len(),
            post_norm_weight.len(),
        )?;
        validate_resident_projection(
            &projections.out_proj,
            attention_output.len(),
            self.dense_weights.len,
        )?;
        validate_resident_projection(&projections.router, residual.len(), self.dense_weights.len)?;
        let width_u32 =
            u32::try_from(plan.width).context("FlashMoe Metal CMD2 residual width exceeds u32")?;
        unsafe {
            let mut owned = Vec::with_capacity(7);
            let attention_buffer = self.buffers.tracked_buffer_with_bytes(
                self.device,
                f32_as_bytes(attention_output),
                &mut owned,
            )?;
            let (residual_input_buffer, owned_residual_input) = match residual {
                MetalBatchProjectionInput::Cpu(residual) => (
                    self.buffers.tracked_buffer_with_bytes(
                        self.device,
                        f32_as_bytes(residual),
                        &mut owned,
                    )?,
                    true,
                ),
                MetalBatchProjectionInput::Buffer { buffer, .. } => (buffer, false),
            };
            let norm_weight_buffer = self.buffers.tracked_buffer_with_bytes(
                self.device,
                f32_as_bytes(post_norm_weight),
                &mut owned,
            )?;
            let projected_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                plan.width * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let residual_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                plan.width * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let normed_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                plan.width * std::mem::size_of::<f32>(),
                &mut owned,
            )?;
            let router_logits_buffer = self.buffers.tracked_buffer_with_len(
                self.device,
                projections.router.rows() * std::mem::size_of::<f32>(),
                &mut owned,
            )?;

            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE CMD2 post-attention command buffer",
                "failed to create Flash-MoE CMD2 post-attention compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release(&owned, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();

            let encode_result = (|| -> Result<()> {
                encode_resident_projection(
                    self.pipelines,
                    encoder,
                    self.dense_weights,
                    &projections.out_proj,
                    attention_buffer,
                    projected_buffer,
                    0,
                )?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.residual_rms_norm_pipeline,
                );
                set_buffer(encoder, projected_buffer, 0);
                set_buffer(encoder, residual_input_buffer, 1);
                set_buffer(encoder, norm_weight_buffer, 2);
                set_buffer(encoder, residual_buffer, 3);
                set_buffer(encoder, normed_buffer, 4);
                set_bytes(encoder, u32_as_bytes(&width_u32), 5);
                set_bytes(
                    encoder,
                    f32_as_bytes(std::slice::from_ref(&self.norm_epsilon)),
                    6,
                );
                dispatch_single_threadgroup(encoder, 256);
                encode_resident_projection(
                    self.pipelines,
                    encoder,
                    self.dense_weights,
                    &projections.router,
                    normed_buffer,
                    router_logits_buffer,
                    0,
                )
            })();
            if let Err(error) = encode_result {
                drop(encoding);
                self.buffers.recycle_or_release(&owned, true);
                return Err(error);
            }
            encoding.end_encoding();

            let context = MetalCommandContext::new("cmd2_resident_post_attention")
                .with("layer", plan.layer)
                .with("width", plan.width)
                .with("attention_width", plan.attention_width)
                .with("experts", plan.experts)
                .with("top_k", plan.active_count)
                .with("routing_topk", "cpu")
                .with("out_proj", projections.out_proj.tensor_name())
                .with("router", projections.router.tensor_name());
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.buffers
                    .recycle_or_release(&owned, error.should_release_buffers());
                return Err(error.into());
            }

            let router_logits_ptr =
                msg_send_ptr0(router_logits_buffer, sel("contents")).cast::<f32>();
            let router_scores =
                std::slice::from_raw_parts(router_logits_ptr, projections.router.rows()).to_vec();
            let active = if let Some(correction_bias) = router_correction_bias {
                routing_sigmoid_noaux_top_k(&router_scores, correction_bias, plan.active_count)?
            } else {
                routing_softmax_top_k(&router_scores, plan.active_count)
            };
            let output = MetalPostAttentionPrep::new(
                plan.layer,
                plan.width,
                plan.experts,
                active,
                residual_buffer,
                normed_buffer,
            );

            drop(encoding);
            if output.is_err() {
                self.buffers.recycle_or_release(&owned, false);
                return output;
            }
            let transient = owned
                .into_iter()
                .filter(|buffer| *buffer != residual_buffer && *buffer != normed_buffer)
                .collect::<Vec<_>>();
            debug_assert_eq!(transient.len(), if owned_residual_input { 5 } else { 4 });
            self.buffers.recycle_or_release(&transient, false);
            output
        }
    }
}

#[derive(Debug)]
#[must_use = "scheduled Metal CMD3 submissions must be waited or explicitly finished"]
pub(crate) struct MetalScheduledCmd3Submission {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    buffers: Arc<MetalBufferPool>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    command_buffer: Option<MetalObjcId>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    _command_lease: MetalCommandLease,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    phase_buffers: Option<Vec<MetalPhaseBuffer>>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    _expert_slots: Arc<[Arc<ScheduledExpertSlot>]>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    output: MetalCmd3DeferredOutput,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    context: MetalCommandContext,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    scheduled_output: ScheduledCmd3OutputState,
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    unsupported: std::convert::Infallible,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalScheduledCmd3Submission {
    pub(crate) fn next_normed_input(&self) -> anyhow::Result<Option<MetalStateBuffer>> {
        self.output
            .next_normed_buffer
            .zip(self.output.output_state.next_normed())
            .map(|(buffer, state)| MetalStateBuffer::new(buffer, state))
            .transpose()
    }

    pub(crate) fn hidden_input(&self) -> anyhow::Result<MetalStateBuffer> {
        MetalStateBuffer::new(self.output.hidden_buffer, self.output.output_state.hidden())
    }

    pub(crate) fn finish_without_readback(mut self) -> anyhow::Result<()> {
        objc2::rc::autoreleasepool(|_| unsafe {
            let command_buffer = self
                .command_buffer
                .take()
                .context("FlashMoe Metal CMD3 submission has already been finished")?;
            let wait = wait_for_metal_command_buffer(command_buffer, &self.context);
            release(command_buffer);
            let phase_buffers = self.phase_buffers.take().unwrap_or_default();
            match wait {
                Ok(()) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, false);
                    Ok(())
                }
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    Err(error.into())
                }
            }
        })
    }

    pub(crate) fn wait(mut self) -> anyhow::Result<FlashMoeExpertPhaseOutput> {
        objc2::rc::autoreleasepool(|_| unsafe {
            let command_buffer = self
                .command_buffer
                .take()
                .context("FlashMoe Metal CMD3 submission has already been waited")?;
            if let Err(error) = wait_for_metal_command_buffer(command_buffer, &self.context) {
                release(command_buffer);
                self.buffers
                    .recycle_or_release_phase(self.phase_buffers.take().unwrap_or_default(), true);
                return Err(error.into());
            }
            let hidden = self.buffers.read_f32_buffer(
                self.output.hidden_buffer,
                self.output.output_state.hidden().len(),
            );
            let next_normed = self
                .output
                .next_normed_buffer
                .zip(self.output.output_state.next_normed())
                .map(|(buffer, state)| self.buffers.read_f32_buffer(buffer, state.len()));
            release(command_buffer);
            self.buffers
                .recycle_or_release_phase(self.phase_buffers.take().unwrap_or_default(), false);
            self.scheduled_output
                .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(hidden, next_normed))
        })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for MetalScheduledCmd3Submission {
    fn drop(&mut self) {
        let Some(command_buffer) = self.command_buffer.take() else {
            return;
        };
        objc2::rc::autoreleasepool(|_| unsafe {
            let wait = wait_for_metal_command_buffer(command_buffer, &self.context);
            release(command_buffer);
            self.buffers.recycle_or_release_phase(
                self.phase_buffers.take().unwrap_or_default(),
                wait.is_err(),
            );
            if let Err(error) = wait {
                tracing::warn!(
                    target: "flashmoe::resources",
                    error = %error,
                    "FlashMoe cleaned up an unfinished Metal CMD3 submission after a command failure"
                );
            } else {
                tracing::warn!(
                    target: "flashmoe::resources",
                    "FlashMoe cleaned up a Metal CMD3 submission that was dropped without an explicit wait"
                );
            }
        });
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl MetalScheduledCmd3Submission {
    pub(crate) fn next_normed_input(&self) -> anyhow::Result<Option<MetalStateBuffer>> {
        match self.unsupported {}
    }

    pub(crate) fn hidden_input(&self) -> anyhow::Result<MetalStateBuffer> {
        match self.unsupported {}
    }

    pub(crate) fn finish_without_readback(self) -> anyhow::Result<()> {
        match self.unsupported {}
    }

    pub(crate) fn wait(self) -> anyhow::Result<FlashMoeExpertPhaseOutput> {
        match self.unsupported {}
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalScheduledCmd3Builder<'a> {
    runtime: &'a MetalRuntime,
    dense_weights: &'a MetalDenseWeights,
    buffers: Arc<MetalBufferPool>,
    norm_epsilon: f32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Clone, Copy)]
struct MetalLayerMajorOneRowReference {
    expert_outputs: MetalObjcId,
    shared_output: MetalObjcId,
    shared_router: MetalObjcId,
    shared_router_width: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
type MetalLayerMajorGroupPlan = (Vec<u32>, Vec<u32>, Vec<(usize, usize)>);

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn qwen_layer_major_group_plan(
    route_slots: &[usize],
    unique_experts: usize,
    active_experts: usize,
) -> anyhow::Result<MetalLayerMajorGroupPlan> {
    if route_slots.is_empty() || unique_experts == 0 || active_experts == 0 {
        bail!("Qwen layer-major group plan requires non-empty routes and experts");
    }
    let mut routes_by_slot = vec![Vec::new(); unique_experts];
    for (route, slot) in route_slots.iter().copied().enumerate() {
        routes_by_slot
            .get_mut(slot)
            .with_context(|| {
                format!("Qwen layer-major route {route} references missing expert slot {slot}")
            })?
            .push(route);
    }
    if routes_by_slot.iter().any(Vec::is_empty) {
        bail!("Qwen layer-major unique expert union contains an unused slot");
    }

    let mut grouped_source_rows = Vec::with_capacity(route_slots.len());
    let mut grouped_output_indices = vec![0u32; route_slots.len()];
    let mut groups = Vec::with_capacity(routes_by_slot.len());
    for routes in &routes_by_slot {
        let start = grouped_source_rows.len();
        for route in routes {
            let row = route / active_experts;
            grouped_source_rows
                .push(u32::try_from(row).context("Qwen layer-major source row exceeds u32")?);
            grouped_output_indices[*route] = u32::try_from(grouped_source_rows.len() - 1)
                .context("Qwen layer-major grouped route index exceeds u32")?;
        }
        groups.push((start, routes.len()));
    }
    Ok((grouped_source_rows, grouped_output_indices, groups))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<'a> MetalScheduledCmd3Builder<'a> {
    pub(crate) fn new(
        runtime: &'a MetalRuntime,
        dense_weights: &'a MetalDenseWeights,
        buffers: Arc<MetalBufferPool>,
        norm_epsilon: f32,
    ) -> Self {
        Self {
            runtime,
            dense_weights,
            buffers,
            norm_epsilon,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn submit(
        &self,
        position: usize,
        layer: usize,
        experts: Arc<[Arc<ScheduledExpertSlot>]>,
        routing_weights: &[f32],
        input: MetalPostAttentionPrep,
        scheduled_output: ScheduledCmd3OutputState,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
        payloads: &[ScheduledExpertPhaseMlpPayload<'_>],
    ) -> anyhow::Result<MetalScheduledCmd3Submission> {
        if scheduled_output.layer != layer {
            self.recycle_input(input);
            anyhow::bail!(
                "FlashMoe Metal CMD3 output layer {} does not match scheduled layer {layer}",
                scheduled_output.layer
            );
        }
        if input.input.state() != input.state {
            self.recycle_input(input);
            anyhow::bail!(
                "FlashMoe Metal CMD3 typed input state does not match post-attention state for layer {layer}"
            );
        }
        if input.input.state().routing().layer() != layer || input.state.routing().layer() != layer
        {
            self.recycle_input(input);
            anyhow::bail!("FlashMoe Metal CMD3 input layer does not match scheduled layer {layer}");
        }
        let width = input.input.width();
        let output_state = scheduled_output.state();
        let command_plan = match MetalCmd3ExecutionPlan::new(
            position,
            layer,
            experts.len(),
            width,
            routing_weights.len(),
            output_state,
            shared,
            next_norm_weight.map(|weight| weight.len()),
            payloads,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                self.recycle_input(input);
                return Err(error.context(
                    "FlashMoe unsupported Metal CMD3 phase: invalid scheduled command plan",
                ));
            }
        };
        if let Err(error) = Self::require_shared_implementation(command_plan.shared.source) {
            self.recycle_input(input);
            return Err(error);
        }
        let input_buffers = match MetalCmd3InputBuffers::new(
            command_plan.phase,
            input.normed_buffer,
            input.residual_buffer,
        ) {
            Ok(input_buffers) => input_buffers,
            Err(error) => {
                self.recycle_input(input);
                return Err(error.context(
                    "FlashMoe unsupported Metal CMD3 phase: invalid scheduled input buffers",
                ));
            }
        };
        objc2::rc::autoreleasepool(|_| unsafe {
            self.encode_and_submit(
                command_plan,
                input_buffers,
                experts,
                routing_weights,
                scheduled_output,
                shared,
                next_norm_weight,
                payloads,
            )
        })
    }

    pub(crate) fn execute_layer_major(
        &self,
        scheduled: &ScheduledLayerMajorExperts,
        post_attention: &MetalLayerMajorPostAttention,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
    ) -> anyhow::Result<MetalQwenPrefillLayerOutput> {
        let rows = scheduled.rows();
        let active_experts = scheduled.active_experts();
        let normed = post_attention.normed();
        let residual = post_attention.residual();
        let width = residual.cols();
        let route_count = rows
            .checked_mul(active_experts)
            .context("Qwen layer-major route count overflow")?;
        if rows == 0
            || active_experts == 0
            || width == 0
            || residual.rows() != rows
            || residual.state().role() != FlashMoeStateBufferRole::Residual
            || normed.rows() != rows
            || normed.cols() != width
            || normed.state().role() != FlashMoeStateBufferRole::Normed
            || scheduled.route_slots().len() != route_count
            || scheduled.weights().len() != route_count
            || scheduled.experts().is_empty()
            || next_norm_weight.is_some_and(|weight| weight.len() != width)
        {
            bail!(
                "Qwen layer-major expert command has incompatible geometry layer={} rows={rows} width={width} active={active_experts} normed={} residual={} routes={} weights={} unique={}",
                scheduled.layer(),
                normed.values(),
                residual.values(),
                scheduled.route_slots().len(),
                scheduled.weights().len(),
                scheduled.experts().len()
            );
        }

        let (grouped_source_rows, grouped_output_indices, groups) = qwen_layer_major_group_plan(
            scheduled.route_slots(),
            scheduled.experts().len(),
            active_experts,
        )?;
        debug_assert_eq!(grouped_source_rows.len(), route_count);

        objc2::rc::autoreleasepool(|_| unsafe {
            self.encode_and_execute_layer_major(
                scheduled,
                width,
                normed,
                residual,
                &grouped_source_rows,
                &grouped_output_indices,
                &groups,
                shared,
                next_norm_weight,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_one_row_scalar_reference(
        &self,
        encoder: MetalObjcId,
        scheduled: &ScheduledLayerMajorExperts,
        width: usize,
        intermediate: usize,
        payloads: &[ScheduledExpertPhaseMlpPayload<'_>],
        normed_buffer: MetalObjcId,
        _residual_buffer: MetalObjcId,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        phase_buffers: &mut Vec<MetalPhaseBuffer>,
        source_buffers: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<MetalLayerMajorOneRowReference> {
        unsafe {
            let active_experts = scheduled.active_experts();
            let gate =
                self.phase_buffer(intermediate * std::mem::size_of::<f32>(), phase_buffers)?;
            let up = self.phase_buffer(intermediate * std::mem::size_of::<f32>(), phase_buffers)?;
            let activated =
                self.phase_buffer(intermediate * std::mem::size_of::<f32>(), phase_buffers)?;
            let expert_outputs = self.phase_buffer(
                active_experts * width * std::mem::size_of::<f32>(),
                phase_buffers,
            )?;
            for route in 0..active_experts {
                let slot = scheduled.route_slots()[route];
                let payload = payloads
                    .get(slot)
                    .and_then(ScheduledExpertPhaseMlpPayload::q4_checked)
                    .with_context(|| {
                        format!("one-row Qwen parity route {route} has no affine-Q4 payload")
                    })?;
                self.encode_q4_matvec(
                    encoder,
                    &payload.gate,
                    payload.gate_source(),
                    normed_buffer,
                    gate,
                    0,
                    phase_buffers,
                    source_buffers,
                )?;
                self.encode_q4_matvec(
                    encoder,
                    &payload.up,
                    payload.up_source(),
                    normed_buffer,
                    up,
                    0,
                    phase_buffers,
                    source_buffers,
                )?;
                let intermediate_u32 = u32::try_from(intermediate)
                    .context("one-row Qwen parity intermediate exceeds u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.runtime.pipelines.silu_product_pipeline,
                );
                set_buffer(encoder, gate, 0);
                set_buffer(encoder, up, 1);
                set_buffer(encoder, activated, 2);
                set_bytes(encoder, u32_as_bytes(&intermediate_u32), 3);
                dispatch_threads(encoder, intermediate as u64);
                self.encode_q4_matvec(
                    encoder,
                    &payload.down,
                    payload.down_source(),
                    activated,
                    expert_outputs,
                    (route * width * std::mem::size_of::<f32>()) as u64,
                    phase_buffers,
                    source_buffers,
                )?;
            }

            let (shared_output, shared_router, shared_router_width) = match shared {
                ScheduledSharedExpertPhaseRef::Resident(shared) => {
                    let shape = shared.validated_shape()?;
                    let gate = self.phase_buffer(
                        shape.total_intermediate * std::mem::size_of::<f32>(),
                        phase_buffers,
                    )?;
                    let up = self.phase_buffer(
                        shape.total_intermediate * std::mem::size_of::<f32>(),
                        phase_buffers,
                    )?;
                    let activated = self.phase_buffer(
                        shape.total_intermediate * std::mem::size_of::<f32>(),
                        phase_buffers,
                    )?;
                    let output =
                        self.phase_buffer(width * std::mem::size_of::<f32>(), phase_buffers)?;
                    let router_width = shared.router.as_ref().map_or(0, |router| router.rows());
                    let router = self.phase_buffer(
                        router_width.max(1) * std::mem::size_of::<f32>(),
                        phase_buffers,
                    )?;
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &shared.gate,
                        normed_buffer,
                        gate,
                        0,
                    )?;
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &shared.up,
                        normed_buffer,
                        up,
                        0,
                    )?;
                    if let Some(projection) = shared.router.as_ref() {
                        encode_resident_projection(
                            &self.runtime.pipelines,
                            encoder,
                            self.dense_weights,
                            projection,
                            normed_buffer,
                            router,
                            0,
                        )?;
                    }
                    let total_u32 = u32::try_from(shape.total_intermediate)
                        .context("one-row shared intermediate exceeds u32")?;
                    let intermediate_u32 = u32::try_from(shape.intermediate)
                        .context("one-row shared expert width exceeds u32")?;
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.shared_expert_activation_pipeline,
                    );
                    set_buffer(encoder, gate, 0);
                    set_buffer(encoder, up, 1);
                    set_buffer(encoder, router, 2);
                    set_buffer(encoder, activated, 3);
                    set_bytes(encoder, u32_as_bytes(&intermediate_u32), 4);
                    set_bytes(encoder, u32_as_bytes(&total_u32), 5);
                    dispatch_threads(encoder, shape.total_intermediate as u64);
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &shared.down,
                        activated,
                        output,
                        0,
                    )?;
                    (output, router, router_width)
                }
                ScheduledSharedExpertPhaseRef::None => {
                    let output =
                        self.phase_buffer(width * std::mem::size_of::<f32>(), phase_buffers)?;
                    let width_u32 =
                        u32::try_from(width).context("one-row shared fill width exceeds u32")?;
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.fill_zero_pipeline,
                    );
                    set_buffer(encoder, output, 0);
                    set_bytes(encoder, u32_as_bytes(&width_u32), 1);
                    dispatch_threads(encoder, width as u64);
                    let router = self.phase_buffer_with_bytes(
                        f32_as_bytes(std::slice::from_ref(&80.0f32)),
                        phase_buffers,
                    )?;
                    (output, router, 0)
                }
                ScheduledSharedExpertPhaseRef::Dense(_) => {
                    bail!("one-row Qwen parity requires resident shared projections")
                }
            };
            Ok(MetalLayerMajorOneRowReference {
                expert_outputs,
                shared_output,
                shared_router,
                shared_router_width,
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_and_execute_layer_major(
        &self,
        scheduled: &ScheduledLayerMajorExperts,
        width: usize,
        normed: MetalMatrixBuffer,
        residual: MetalMatrixBuffer,
        grouped_source_rows: &[u32],
        grouped_output_indices: &[u32],
        groups: &[(usize, usize)],
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
    ) -> anyhow::Result<MetalQwenPrefillLayerOutput> {
        unsafe {
            let rows = scheduled.rows();
            let active_experts = scheduled.active_experts();
            let route_count = rows * active_experts;
            let payloads = scheduled
                .experts()
                .iter()
                .map(|expert| expert.scheduled_cmd3_expert_phase_payload(width))
                .collect::<Result<Vec<_>>>()?;
            let intermediate = payloads
                .first()
                .and_then(ScheduledExpertPhaseMlpPayload::q4_checked)
                .map(|payload| payload.gate.rows)
                .context("Qwen layer-major expert graph requires fixed affine-Q4 payloads")?;
            for payload in &payloads {
                let q4 = payload
                    .q4_checked()
                    .context("Qwen layer-major expert graph requires fixed affine-Q4 payloads")?;
                if q4.gate.rows != intermediate
                    || !q4
                        .gate
                        .scale_bias_dtype
                        .eq_ignore_ascii_case(super::experts::EXPERT_SCALE_BIAS_DTYPE_BF16)
                {
                    bail!("Qwen layer-major expert graph requires uniform BF16 affine-Q4 experts");
                }
            }

            let mut phase_buffers = Vec::new();
            let setup = (|| -> anyhow::Result<_> {
                let gathered_buffer = self.phase_buffer(
                    route_count * width * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let grouped_source_rows_buffer = self.phase_buffer_with_bytes(
                    u32_as_bytes_slice(grouped_source_rows),
                    &mut phase_buffers,
                )?;
                let weights_buffer = self.phase_buffer_with_bytes(
                    f32_as_bytes(scheduled.weights()),
                    &mut phase_buffers,
                )?;
                let grouped_indices_buffer = self.phase_buffer_with_bytes(
                    u32_as_bytes_slice(grouped_output_indices),
                    &mut phase_buffers,
                )?;
                let route_intermediate_values = route_count
                    .checked_mul(intermediate)
                    .context("Qwen layer-major expert activation size overflow")?;
                let grouped_output_values = route_count
                    .checked_mul(width)
                    .context("Qwen layer-major expert output size overflow")?;
                let gate_buffer = self.phase_buffer(
                    route_intermediate_values * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let up_buffer = self.phase_buffer(
                    route_intermediate_values * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let activated_buffer = self.phase_buffer(
                    route_intermediate_values * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let grouped_output_buffer = self.phase_buffer(
                    grouped_output_values * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let shared_output_buffer = self.phase_buffer(
                    rows * width * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let hidden_buffer = self.phase_buffer(
                    rows * width * std::mem::size_of::<f32>(),
                    &mut phase_buffers,
                )?;
                let next_norm = next_norm_weight
                    .map(|weight| -> anyhow::Result<_> {
                        let weight_buffer =
                            self.phase_buffer_with_bytes(f32_as_bytes(weight), &mut phase_buffers)?;
                        let output_buffer = self.phase_buffer(
                            rows * width * std::mem::size_of::<f32>(),
                            &mut phase_buffers,
                        )?;
                        Ok((weight_buffer, output_buffer))
                    })
                    .transpose()?;
                Ok((
                    gathered_buffer,
                    grouped_source_rows_buffer,
                    weights_buffer,
                    grouped_indices_buffer,
                    gate_buffer,
                    up_buffer,
                    activated_buffer,
                    grouped_output_buffer,
                    shared_output_buffer,
                    hidden_buffer,
                    next_norm,
                ))
            })();
            let (
                gathered_buffer,
                grouped_source_rows_buffer,
                weights_buffer,
                grouped_indices_buffer,
                gate_buffer,
                up_buffer,
                activated_buffer,
                grouped_output_buffer,
                shared_output_buffer,
                hidden_buffer,
                next_norm,
            ) = match setup {
                Ok(values) => values,
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };

            let mut encoding = match MetalCommandEncoding::new(
                self.runtime.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Qwen layer-major expert command buffer",
                "failed to create Qwen layer-major expert encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            let encode_result = (||
             -> anyhow::Result<(
                MetalObjcId,
                MetalObjcId,
                MetalObjcId,
                MetalObjcId,
                usize,
                Option<MetalObjcId>,
                Option<MetalLayerMajorOneRowReference>,
            )> {
                let gathered_values = route_count
                    .checked_mul(width)
                    .context("Qwen layer-major gathered matrix size overflow")?;
                let gathered_values_u32 = u32::try_from(gathered_values)
                    .context("Qwen layer-major gathered matrix exceeds u32")?;
                let width_u32 =
                    u32::try_from(width).context("Qwen layer-major gather width exceeds u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.runtime.pipelines.qwen_layer_major_gather_pipeline,
                );
                set_buffer(encoder, normed.buffer(), 0);
                set_buffer(encoder, grouped_source_rows_buffer, 1);
                set_buffer(encoder, gathered_buffer, 2);
                set_bytes(encoder, u32_as_bytes(&width_u32), 3);
                set_bytes(encoder, u32_as_bytes(&gathered_values_u32), 4);
                dispatch_threads(encoder, gathered_values as u64);

                let mut source_buffers = MetalExpertSourceBufferCache::default();
                for (((payload, group), expert), slot_index) in payloads
                    .iter()
                    .zip(groups)
                    .zip(scheduled.experts().iter())
                    .zip(0usize..)
                {
                    let q4 = payload
                        .q4_checked()
                        .context("Qwen layer-major expert payload changed after validation")?;
                    let (start, count) = *group;
                    self.encode_layer_major_q4_matrix(
                        encoder,
                        &q4.gate,
                        q4.gate_source(),
                        gathered_buffer,
                        start * width,
                        gate_buffer,
                        start * intermediate,
                        count,
                        &mut phase_buffers,
                        &mut source_buffers,
                    )?;
                    self.encode_layer_major_q4_matrix(
                        encoder,
                        &q4.up,
                        q4.up_source(),
                        gathered_buffer,
                        start * width,
                        up_buffer,
                        start * intermediate,
                        count,
                        &mut phase_buffers,
                        &mut source_buffers,
                    )?;
                    debug_assert_eq!(expert.expert(), scheduled.experts()[slot_index].expert());
                }
                let activated_values = route_count * intermediate;
                let activated_u32 = u32::try_from(activated_values)
                    .context("Qwen layer-major activation width exceeds u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.runtime.pipelines.silu_product_pipeline,
                );
                set_buffer(encoder, gate_buffer, 0);
                set_buffer(encoder, up_buffer, 1);
                set_buffer(encoder, activated_buffer, 2);
                set_bytes(encoder, u32_as_bytes(&activated_u32), 3);
                dispatch_threads(encoder, activated_values as u64);
                for (payload, group) in payloads.iter().zip(groups) {
                    let q4 = payload
                        .q4_checked()
                        .context("Qwen layer-major expert payload changed after validation")?;
                    let (start, count) = *group;
                    self.encode_layer_major_q4_matrix(
                        encoder,
                        &q4.down,
                        q4.down_source(),
                        activated_buffer,
                        start * intermediate,
                        grouped_output_buffer,
                        start * width,
                        count,
                        &mut phase_buffers,
                        &mut source_buffers,
                    )?;
                }

                let (shared_router, shared_router_width) = match shared {
                    ScheduledSharedExpertPhaseRef::Resident(shared) => {
                        let shape = shared.validated_shape()?;
                        let shared_gate = self.phase_buffer(
                            rows * shape.total_intermediate * std::mem::size_of::<f32>(),
                            &mut phase_buffers,
                        )?;
                        let shared_up = self.phase_buffer(
                            rows * shape.total_intermediate * std::mem::size_of::<f32>(),
                            &mut phase_buffers,
                        )?;
                        let shared_activated = self.phase_buffer(
                            rows * shape.total_intermediate * std::mem::size_of::<f32>(),
                            &mut phase_buffers,
                        )?;
                        let shared_router_width = shared.router.as_ref().map_or(0, |p| p.rows());
                        let shared_router = self.phase_buffer(
                            rows * shared_router_width.max(1) * std::mem::size_of::<f32>(),
                            &mut phase_buffers,
                        )?;
                        self.encode_layer_major_resident_matrix(
                            encoder,
                            &shared.gate,
                            normed.buffer(),
                            shared_gate,
                            rows,
                        )?;
                        self.encode_layer_major_resident_matrix(
                            encoder,
                            &shared.up,
                            normed.buffer(),
                            shared_up,
                            rows,
                        )?;
                        if let Some(router) = shared.router.as_ref() {
                            self.encode_layer_major_resident_matrix(
                                encoder,
                                router,
                                normed.buffer(),
                                shared_router,
                                rows,
                            )?;
                        }
                        let shared_values = rows * shape.total_intermediate;
                        let shared_values_u32 = u32::try_from(shared_values)
                            .context("Qwen shared activation width exceeds u32")?;
                        msg_send_void1_id(
                            encoder,
                            sel("setComputePipelineState:"),
                            self.runtime.pipelines.silu_product_pipeline,
                        );
                        set_buffer(encoder, shared_gate, 0);
                        set_buffer(encoder, shared_up, 1);
                        set_buffer(encoder, shared_activated, 2);
                        set_bytes(encoder, u32_as_bytes(&shared_values_u32), 3);
                        dispatch_threads(encoder, shared_values as u64);
                        self.encode_layer_major_resident_matrix(
                            encoder,
                            &shared.down,
                            shared_activated,
                            shared_output_buffer,
                            rows,
                        )?;
                        (shared_router, shared_router_width)
                    }
                    ScheduledSharedExpertPhaseRef::None => {
                        let zero_width = u32::try_from(rows * width)
                            .context("Qwen shared zero-fill width exceeds u32")?;
                        msg_send_void1_id(
                            encoder,
                            sel("setComputePipelineState:"),
                            self.runtime.pipelines.fill_zero_pipeline,
                        );
                        set_buffer(encoder, shared_output_buffer, 0);
                        set_bytes(encoder, u32_as_bytes(&zero_width), 1);
                        dispatch_threads(encoder, (rows * width) as u64);
                        (shared_output_buffer, 0)
                    }
                    ScheduledSharedExpertPhaseRef::Dense(_) => {
                        bail!("Qwen layer-major shared expert requires resident projections")
                    }
                };

                let one_row_reference = (rows == 1)
                    .then(|| {
                        self.encode_one_row_scalar_reference(
                            encoder,
                            scheduled,
                            width,
                            intermediate,
                            &payloads,
                            normed.buffer(),
                            residual.buffer(),
                            shared,
                            &mut phase_buffers,
                            &mut source_buffers,
                        )
                    })
                    .transpose()?;

                let rows_u32 = u32::try_from(rows).context("Qwen batch rows exceed u32")?;
                let width_u32 = u32::try_from(width).context("Qwen batch width exceeds u32")?;
                let active_u32 = u32::try_from(active_experts)
                    .context("Qwen active expert count exceeds u32")?;
                let shared_router_width_u32 = u32::try_from(shared_router_width)
                    .context("Qwen shared router width exceeds u32")?;
                msg_send_void1_id(
                    encoder,
                    sel("setComputePipelineState:"),
                    self.runtime.pipelines.qwen_layer_major_combine_pipeline,
                );
                set_buffer(encoder, residual.buffer(), 0);
                set_buffer(encoder, shared_output_buffer, 1);
                set_buffer(encoder, grouped_output_buffer, 2);
                set_buffer(encoder, weights_buffer, 3);
                set_buffer(encoder, grouped_indices_buffer, 4);
                set_buffer(encoder, shared_router, 5);
                set_buffer(encoder, hidden_buffer, 6);
                set_bytes(encoder, u32_as_bytes(&rows_u32), 7);
                set_bytes(encoder, u32_as_bytes(&width_u32), 8);
                set_bytes(encoder, u32_as_bytes(&active_u32), 9);
                set_bytes(encoder, u32_as_bytes(&shared_router_width_u32), 10);
                dispatch_threads(encoder, (rows * width) as u64);
                let next_normed_buffer = if let Some((weight_buffer, output_buffer)) = next_norm {
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.rms_norm_reduced_pipeline,
                    );
                    set_buffer(encoder, hidden_buffer, 0);
                    set_buffer(encoder, weight_buffer, 1);
                    set_buffer(encoder, output_buffer, 2);
                    set_bytes(encoder, u32_as_bytes(&width_u32), 3);
                    set_bytes(
                        encoder,
                        f32_as_bytes(std::slice::from_ref(&self.norm_epsilon)),
                        4,
                    );
                    msg_send_void2_size(
                        encoder,
                        sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                        MetalDispatchSize::new(1, rows as u64, 1),
                        MetalDispatchSize::new(256, 1, 1),
                    );
                    Some(output_buffer)
                } else {
                    None
                };
                Ok((
                    hidden_buffer,
                    grouped_output_buffer,
                    shared_output_buffer,
                    shared_router,
                    shared_router_width,
                    next_normed_buffer,
                    one_row_reference,
                ))
            })();
            let (
                hidden_buffer,
                grouped_output_buffer,
                shared_output_buffer,
                shared_router,
                shared_router_width,
                next_normed_buffer,
                one_row_reference,
            ) = match encode_result {
                Ok(result) => result,
                Err(error) => {
                    drop(encoding);
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };
            encoding.end_encoding();
            let context = MetalCommandContext::new("qwen_layer_major_experts")
                .with("layer", scheduled.layer())
                .with("rows", rows)
                .with("active_experts", active_experts)
                .with("unique_experts", scheduled.experts().len());
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.buffers
                    .recycle_or_release_phase(phase_buffers, error.should_release_buffers());
                return Err(error.into());
            }
            if let Some(reference) = one_row_reference {
                let grouped = self
                    .buffers
                    .read_f32_buffer(grouped_output_buffer, route_count * width);
                let scalar = self
                    .buffers
                    .read_f32_buffer(reference.expert_outputs, route_count * width);
                for route in 0..route_count {
                    let grouped_route = grouped_output_indices[route] as usize;
                    for col in 0..width {
                        let actual = grouped[grouped_route * width + col];
                        let expected = scalar[route * width + col];
                        if actual.to_bits() != expected.to_bits() {
                            drop(encoding);
                            self.buffers.recycle_or_release_phase(phase_buffers, false);
                            bail!(
                                "Qwen one-row layer-major expert parity failed layer={} route={route} col={col}: matrix={actual} scalar={expected} delta={}",
                                scheduled.layer(),
                                (actual - expected).abs()
                            );
                        }
                    }
                }
                let shared_actual = self.buffers.read_f32_buffer(shared_output_buffer, width);
                let shared_expected = self.buffers.read_f32_buffer(reference.shared_output, width);
                for (col, (actual, expected)) in
                    shared_actual.iter().zip(shared_expected.iter()).enumerate()
                {
                    if actual.to_bits() != expected.to_bits() {
                        drop(encoding);
                        self.buffers.recycle_or_release_phase(phase_buffers, false);
                        bail!(
                            "Qwen one-row layer-major shared-expert parity failed layer={} col={col}: matrix={actual} scalar={expected} delta={}",
                            scheduled.layer(),
                            (actual - expected).abs()
                        );
                    }
                }
                if shared_router_width != reference.shared_router_width {
                    drop(encoding);
                    self.buffers.recycle_or_release_phase(phase_buffers, false);
                    bail!("Qwen one-row shared-router widths differ");
                }
                if shared_router_width > 0 {
                    let router_actual = self
                        .buffers
                        .read_f32_buffer(shared_router, shared_router_width);
                    let router_expected = self
                        .buffers
                        .read_f32_buffer(reference.shared_router, shared_router_width);
                    for (index, (actual, expected)) in
                        router_actual.iter().zip(router_expected.iter()).enumerate()
                    {
                        if actual.to_bits() != expected.to_bits() {
                            drop(encoding);
                            self.buffers.recycle_or_release_phase(phase_buffers, false);
                            bail!(
                                "Qwen one-row shared-router parity failed layer={} index={index}: matrix={actual} scalar={expected} delta={}",
                                scheduled.layer(),
                                (actual - expected).abs()
                            );
                        }
                    }
                }
            }
            drop(encoding);
            let output = MetalQwenPrefillLayerOutput::new(
                Arc::clone(&self.buffers),
                scheduled.layer(),
                hidden_buffer,
                next_normed_buffer,
                rows,
                width,
            );
            match output {
                Ok(output) => {
                    let transient = phase_buffers
                        .into_iter()
                        .filter(|phase| {
                            phase.id != hidden_buffer && next_normed_buffer != Some(phase.id)
                        })
                        .collect();
                    self.buffers.recycle_or_release_phase(transient, false);
                    Ok(output)
                }
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, false);
                    Err(error)
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_layer_major_q4_matrix(
        &self,
        encoder: MetalObjcId,
        payload: &super::experts::Q4MatvecPayload<'_>,
        source: super::experts::Q4MatvecSource<'_>,
        input: MetalObjcId,
        input_value_offset: usize,
        output: MetalObjcId,
        output_value_offset: usize,
        input_rows: usize,
        buffers: &mut Vec<MetalPhaseBuffer>,
        source_buffers: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<()> {
        unsafe {
            if input_rows == 0
                || !payload
                    .scale_bias_dtype
                    .eq_ignore_ascii_case(super::experts::EXPERT_SCALE_BIAS_DTYPE_BF16)
            {
                bail!("Qwen layer-major Q4 matrix requires non-empty BF16 affine rows");
            }
            let buffer = self.expert_source_buffer(
                source.bytes,
                source.reusable_bytes,
                buffers,
                source_buffers,
            )?;
            self.encode_layer_major_q4_matrix_from_buffer(
                encoder,
                buffer,
                source.packed_offset as u64,
                source.scale_offset as u64,
                source.bias_offset as u64,
                payload.rows,
                payload.cols,
                payload.group_size,
                input,
                input_value_offset,
                output,
                output_value_offset,
                input_rows,
            )
        }
    }

    unsafe fn encode_layer_major_resident_matrix(
        &self,
        encoder: MetalObjcId,
        projection: &ResidentMmapMatvecProjection,
        input: MetalObjcId,
        output: MetalObjcId,
        input_rows: usize,
    ) -> anyhow::Result<()> {
        unsafe {
            match projection {
                ResidentMmapMatvecProjection::Q4(projection) => {
                    if !projection
                        .scale_bias_dtype
                        .eq_ignore_ascii_case(super::experts::EXPERT_SCALE_BIAS_DTYPE_BF16)
                    {
                        bail!("Qwen layer-major resident matrix requires BF16 scale/bias weights");
                    }
                    self.encode_layer_major_q4_matrix_from_buffer(
                        encoder,
                        self.dense_weights.buffer,
                        projection.packed_byte_offset,
                        projection.scales_byte_offset,
                        projection.biases_byte_offset,
                        projection.rows,
                        projection.cols,
                        projection.group_size,
                        input,
                        0,
                        output,
                        0,
                        input_rows,
                    )
                }
                ResidentMmapMatvecProjection::Dense(projection) => encode_dense_resident_matrix(
                    &self.runtime.pipelines,
                    encoder,
                    self.dense_weights,
                    projection,
                    input,
                    output,
                    input_rows,
                ),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_layer_major_q4_matrix_from_buffer(
        &self,
        encoder: MetalObjcId,
        weights: MetalObjcId,
        packed_offset: u64,
        scale_offset: u64,
        bias_offset: u64,
        rows: usize,
        cols: usize,
        group_size: usize,
        input: MetalObjcId,
        input_value_offset: usize,
        output: MetalObjcId,
        output_value_offset: usize,
        input_rows: usize,
    ) -> anyhow::Result<()> {
        unsafe {
            let rows_u32 = u32::try_from(rows).context("Qwen matrix rows exceed u32")?;
            let cols_u32 = u32::try_from(cols).context("Qwen matrix cols exceed u32")?;
            let groups_u32 = u32::try_from(cols.div_ceil(group_size.max(1)))
                .context("Qwen matrix groups exceed u32")?;
            let group_size_u32 =
                u32::try_from(group_size).context("Qwen matrix group size exceeds u32")?;
            let projection_count = 1u32;
            let row_offset = 0u32;
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.runtime
                    .pipelines
                    .q4_mmap_batch_bf16_scale_bias_pipeline,
            );
            set_buffer(encoder, weights, 0);
            set_buffer_with_offset(
                encoder,
                input,
                (input_value_offset * std::mem::size_of::<f32>()) as u64,
                1,
            );
            set_buffer_with_offset(
                encoder,
                output,
                (output_value_offset * std::mem::size_of::<f32>()) as u64,
                2,
            );
            set_bytes(encoder, u64_as_bytes(&packed_offset), 3);
            set_bytes(encoder, u64_as_bytes(&scale_offset), 4);
            set_bytes(encoder, u64_as_bytes(&bias_offset), 5);
            set_bytes(encoder, u32_as_bytes(&row_offset), 6);
            set_bytes(encoder, u32_as_bytes(&rows_u32), 7);
            set_bytes(encoder, u32_as_bytes(&groups_u32), 8);
            set_bytes(encoder, u32_as_bytes(&projection_count), 9);
            set_bytes(encoder, u32_as_bytes(&cols_u32), 10);
            set_bytes(encoder, u32_as_bytes(&group_size_u32), 11);
            dispatch_q4_mmap_matrix_threadgroups(encoder, rows as u64, input_rows as u64);
            Ok(())
        }
    }

    fn recycle_input(&self, input: MetalPostAttentionPrep) {
        self.buffers
            .recycle_or_release(&[input.normed_buffer, input.residual_buffer], false);
    }

    fn require_shared_implementation(source: MetalCmd3SharedPhaseSource) -> anyhow::Result<()> {
        match source {
            MetalCmd3SharedPhaseSource::Resident | MetalCmd3SharedPhaseSource::None => Ok(()),
            MetalCmd3SharedPhaseSource::Dense => anyhow::bail!(
                "FlashMoe unsupported Metal CMD3 implementation: dense CPU shared-expert weights are not a declared implementation"
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_and_submit(
        &self,
        command_plan: MetalCmd3ExecutionPlan,
        input_buffers: MetalCmd3InputBuffers,
        experts: Arc<[Arc<ScheduledExpertSlot>]>,
        routing_weights: &[f32],
        scheduled_output: ScheduledCmd3OutputState,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        next_norm_weight: Option<&[f32]>,
        payloads: &[ScheduledExpertPhaseMlpPayload<'_>],
    ) -> anyhow::Result<MetalScheduledCmd3Submission> {
        unsafe {
            let mut phase_buffers = vec![
                MetalPhaseBuffer::recyclable(input_buffers.normed),
                MetalPhaseBuffer::recyclable(input_buffers.residual),
            ];
            let setup = (|| -> anyhow::Result<_> {
                let output_buffers = self.output_buffers(&command_plan, &mut phase_buffers)?;
                let combine_buffers = self.combine_buffers(
                    command_plan.combine,
                    routing_weights,
                    &mut phase_buffers,
                )?;
                Ok((output_buffers, combine_buffers))
            })();
            let (output_buffers, combine_buffers) = match setup {
                Ok(setup) => setup,
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };

            let mut encoding = match MetalCommandEncoding::new(
                self.runtime.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Flash-MoE scheduled CMD3 command buffer",
                "failed to create Flash-MoE scheduled CMD3 compute encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();

            let encode_result = (|| -> anyhow::Result<()> {
                let mut source_buffers = MetalExpertSourceBufferCache::default();
                let shared_router = self.encode_shared(
                    encoder,
                    &command_plan,
                    input_buffers,
                    &output_buffers,
                    combine_buffers,
                    shared,
                    &mut phase_buffers,
                )?;
                for (active_plan, payload) in command_plan.active_experts.iter().zip(payloads) {
                    let active_work = self.active_work_buffers(*active_plan, &mut phase_buffers)?;
                    let active_stage = MetalCmd3ActiveExpertStageBuffers::new(
                        *active_plan,
                        input_buffers,
                        &output_buffers,
                        active_work,
                    )?;
                    match payload {
                        ScheduledExpertPhaseMlpPayload::Q4(payload) => {
                            let gate_out = active_stage
                                .work
                                .gate_out
                                .context("Metal Q4 expert stage has no gate projection buffer")?;
                            let up_out = active_stage
                                .work
                                .up_out
                                .context("Metal Q4 expert stage has no up projection buffer")?;
                            self.encode_q4_matvec(
                                encoder,
                                &payload.gate,
                                payload.gate_source(),
                                active_stage.normed,
                                gate_out,
                                0,
                                &mut phase_buffers,
                                &mut source_buffers,
                            )?;
                            self.encode_q4_matvec(
                                encoder,
                                &payload.up,
                                payload.up_source(),
                                active_stage.normed,
                                up_out,
                                0,
                                &mut phase_buffers,
                                &mut source_buffers,
                            )?;
                            let intermediate = active_stage.plan.intermediate_u32()?;
                            msg_send_void1_id(
                                encoder,
                                sel("setComputePipelineState:"),
                                self.runtime.pipelines.silu_product_pipeline,
                            );
                            set_buffer(encoder, gate_out, 0);
                            set_buffer(encoder, up_out, 1);
                            set_buffer(encoder, active_stage.activated, 2);
                            set_bytes(encoder, u32_as_bytes(&intermediate), 3);
                            dispatch_threads(encoder, active_stage.plan.intermediate as u64);
                            self.encode_q4_matvec(
                                encoder,
                                &payload.down,
                                payload.down_source(),
                                active_stage.activated,
                                active_stage.expert_outputs,
                                active_stage.output_offset,
                                &mut phase_buffers,
                                &mut source_buffers,
                            )?;
                        }
                        ScheduledExpertPhaseMlpPayload::Dense(payload) => {
                            self.encode_dense_swiglu(
                                encoder,
                                payload,
                                active_stage,
                                &mut phase_buffers,
                                &mut source_buffers,
                            )?;
                        }
                        ScheduledExpertPhaseMlpPayload::DeepSeekGguf(_) => {
                            bail!(
                                "DeepSeek GGUF expert payload reached the Qwen/GLM CMD3 builder instead of its load-resolved fused Metal builder"
                            );
                        }
                    }
                }
                self.encode_combine(
                    encoder,
                    command_plan.combine,
                    command_plan.shared,
                    input_buffers,
                    &output_buffers,
                    combine_buffers,
                    shared_router,
                    &mut phase_buffers,
                )?;
                if let (Some(weight), Some(next_normed), Some(next_plan)) = (
                    next_norm_weight,
                    output_buffers.next_normed,
                    command_plan.next_norm,
                ) {
                    let next_buffers = self.next_norm_buffers(
                        next_plan,
                        weight,
                        output_buffers.hidden,
                        next_normed,
                        combine_buffers.width,
                        &mut phase_buffers,
                    )?;
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.rms_norm_reduced_pipeline,
                    );
                    set_buffer(encoder, next_buffers.hidden, 0);
                    set_buffer(encoder, next_buffers.weight, 1);
                    set_buffer(encoder, next_buffers.next_normed, 2);
                    set_buffer(encoder, next_buffers.width, 3);
                    set_bytes(
                        encoder,
                        f32_as_bytes(std::slice::from_ref(&self.norm_epsilon)),
                        4,
                    );
                    dispatch_single_threadgroup(encoder, next_plan.dispatch_threads);
                }
                Ok(())
            })();
            if let Err(error) = encode_result {
                drop(encoding);
                self.buffers.recycle_or_release_phase(phase_buffers, true);
                return Err(error);
            }
            encoding.end_encoding();

            let output = match MetalCmd3DeferredOutput::new(
                output_buffers.hidden,
                output_buffers.next_normed,
                command_plan.phase.output_state,
            ) {
                Ok(output) => output,
                Err(error) => {
                    drop(encoding);
                    self.buffers.recycle_or_release_phase(phase_buffers, true);
                    return Err(error);
                }
            };
            let expert_ids = experts
                .iter()
                .map(|slot| slot.expert().to_string())
                .collect::<Vec<_>>()
                .join(",");
            let context = command_plan.command_context(expert_ids);
            let (command_buffer, command_lease) = encoding.into_command_buffer();
            commit_metal_command_buffer(command_buffer, &context);
            Ok(MetalScheduledCmd3Submission {
                buffers: Arc::clone(&self.buffers),
                command_buffer: Some(command_buffer),
                _command_lease: command_lease,
                phase_buffers: Some(phase_buffers),
                _expert_slots: experts,
                output,
                context,
                scheduled_output,
            })
        }
    }

    unsafe fn output_buffers(
        &self,
        plan: &MetalCmd3ExecutionPlan,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalCmd3OutputBuffers> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let expert_outputs = self.phase_buffer(layout.expert_outputs_bytes, buffers)?;
            let shared_output = self.phase_buffer(layout.shared_output_bytes, buffers)?;
            let hidden = self.phase_buffer(layout.hidden_output_bytes, buffers)?;
            let next_normed = layout
                .next_normed_output_bytes
                .map(|bytes| self.phase_buffer(bytes, buffers))
                .transpose()?;
            MetalCmd3OutputBuffers::new(plan, expert_outputs, shared_output, hidden, next_normed)
        }
    }

    unsafe fn combine_buffers(
        &self,
        plan: MetalCmd3CombinePlan,
        routing_weights: &[f32],
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalCmd3CombineBuffers> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let weight_bytes = f32_as_bytes(routing_weights);
            if weight_bytes.len() != layout.routing_weights_bytes {
                anyhow::bail!(
                    "FlashMoe Metal CMD3 routing weights byte length {} does not match {}",
                    weight_bytes.len(),
                    layout.routing_weights_bytes
                );
            }
            let routing_weights = self.phase_buffer_with_bytes(weight_bytes, buffers)?;
            let width = self.phase_buffer_with_bytes(u32_as_bytes(&layout.width_u32), buffers)?;
            let active_count =
                self.phase_buffer_with_bytes(u32_as_bytes(&layout.active_count_u32), buffers)?;
            MetalCmd3CombineBuffers::new(plan, routing_weights, width, active_count)
        }
    }

    unsafe fn shared_work_buffers(
        &self,
        plan: MetalCmd3SharedPhasePlan,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalCmd3SharedWorkBuffers> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let gate = self.phase_buffer(layout.projection_output_bytes, buffers)?;
            let up = self.phase_buffer(layout.projection_output_bytes, buffers)?;
            let router = self.phase_buffer(layout.router_output_bytes, buffers)?;
            let activated = self.phase_buffer(layout.projection_output_bytes, buffers)?;
            let total = self
                .phase_buffer_with_bytes(u32_as_bytes(&layout.total_intermediate_u32), buffers)?;
            let intermediate =
                self.phase_buffer_with_bytes(u32_as_bytes(&layout.intermediate_u32), buffers)?;
            MetalCmd3SharedWorkBuffers::new(plan, gate, up, router, activated, total, intermediate)
        }
    }

    unsafe fn active_work_buffers(
        &self,
        plan: MetalCmd3ActiveExpertPlan,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalCmd3ActiveExpertWorkBuffers> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let gate_out = layout
                .projection_output_bytes
                .map(|bytes| self.phase_buffer(bytes, buffers))
                .transpose()?;
            let up_out = layout
                .projection_output_bytes
                .map(|bytes| self.phase_buffer(bytes, buffers))
                .transpose()?;
            let activated = self.phase_buffer(layout.activation_bytes, buffers)?;
            MetalCmd3ActiveExpertWorkBuffers::new(plan, gate_out, up_out, activated)
        }
    }

    unsafe fn next_norm_buffers(
        &self,
        plan: MetalCmd3NextNormPlan,
        weight: &[f32],
        hidden: MetalObjcId,
        next_normed: MetalObjcId,
        width: MetalObjcId,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalCmd3NextNormBuffers> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let bytes = f32_as_bytes(&weight[..plan.width]);
            if bytes.len() != layout.weight_bytes {
                anyhow::bail!(
                    "FlashMoe Metal CMD3 next-norm weight bytes {} do not match {}",
                    bytes.len(),
                    layout.weight_bytes
                );
            }
            let weight = self.phase_buffer_with_bytes(bytes, buffers)?;
            MetalCmd3NextNormBuffers::new(plan, hidden, weight, next_normed, width)
        }
    }

    unsafe fn encode_shared(
        &self,
        encoder: MetalObjcId,
        command: &MetalCmd3ExecutionPlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        combine: MetalCmd3CombineBuffers,
        shared: ScheduledSharedExpertPhaseRef<'_>,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            match command.shared.source {
                MetalCmd3SharedPhaseSource::Resident => {
                    let weights = shared.resident().context(
                        "FlashMoe Metal CMD3 resident shared stage has no resolved projections",
                    )?;
                    let work = self.shared_work_buffers(command.shared, buffers)?;
                    let stage = MetalCmd3SharedStageBuffers::projected(
                        command.shared,
                        inputs,
                        outputs,
                        combine,
                        work,
                    )?;
                    let work = stage
                        .work
                        .context("missing Metal CMD3 shared work buffers")?;
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &weights.gate,
                        stage.normed,
                        work.gate_out,
                        0,
                    )?;
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &weights.up,
                        stage.normed,
                        work.up_out,
                        0,
                    )?;
                    let shared_router = if let Some(router) = weights.router.as_ref() {
                        encode_resident_projection(
                            &self.runtime.pipelines,
                            encoder,
                            self.dense_weights,
                            router,
                            stage.normed,
                            work.router_out,
                            0,
                        )?;
                        work.router_out
                    } else {
                        self.phase_buffer_with_bytes(
                            f32_as_bytes(std::slice::from_ref(&80.0f32)),
                            buffers,
                        )?
                    };
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.shared_expert_activation_pipeline,
                    );
                    set_buffer(encoder, work.gate_out, 0);
                    set_buffer(encoder, work.up_out, 1);
                    set_buffer(encoder, shared_router, 2);
                    set_buffer(encoder, work.activated, 3);
                    set_buffer(encoder, work.intermediate, 4);
                    set_buffer(encoder, work.total_intermediate, 5);
                    dispatch_threads(encoder, command.shared.activation_dispatch_threads());
                    encode_resident_projection(
                        &self.runtime.pipelines,
                        encoder,
                        self.dense_weights,
                        &weights.down,
                        work.activated,
                        stage.shared_output,
                        0,
                    )?;
                    Ok(shared_router)
                }
                MetalCmd3SharedPhaseSource::None => {
                    let fill_width = u32::try_from(command.shared.fill_zero_width())
                        .context("Metal CMD3 shared fill width exceeds u32")?;
                    let stage = MetalCmd3SharedStageBuffers::fill_zero(
                        command.shared,
                        inputs,
                        outputs,
                        combine,
                    )?;
                    msg_send_void1_id(
                        encoder,
                        sel("setComputePipelineState:"),
                        self.runtime.pipelines.fill_zero_pipeline,
                    );
                    set_buffer(encoder, stage.shared_output, 0);
                    set_bytes(encoder, u32_as_bytes(&fill_width), 1);
                    dispatch_threads(encoder, command.shared.fill_zero_width() as u64);
                    Ok(stage.shared_output)
                }
                MetalCmd3SharedPhaseSource::Dense => unreachable!("rejected before encoding"),
            }
        }
    }

    unsafe fn encode_combine(
        &self,
        encoder: MetalObjcId,
        plan: MetalCmd3CombinePlan,
        shared_plan: MetalCmd3SharedPhasePlan,
        inputs: MetalCmd3InputBuffers,
        outputs: &MetalCmd3OutputBuffers,
        combine: MetalCmd3CombineBuffers,
        shared_router: MetalObjcId,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<()> {
        unsafe {
            let layout = plan.buffer_layout()?;
            let stage = MetalCmd3CombineStageBuffers::new(plan, inputs, outputs, combine)?;
            let grouped_indices = (0..plan.active_count)
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("Metal CMD3 active expert index exceeds u32")?;
            let grouped_indices =
                self.phase_buffer_with_bytes(u32_as_bytes_slice(&grouped_indices), buffers)?;
            let rows = 1u32;
            let shared_router_width = if shared_plan.source == MetalCmd3SharedPhaseSource::Resident
            {
                u32::try_from(shared_plan.shared_experts)
                    .context("Metal CMD3 shared router width exceeds u32")?
            } else {
                0
            };
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.runtime.pipelines.qwen_layer_major_combine_pipeline,
            );
            set_buffer(encoder, stage.residual, 0);
            set_buffer(encoder, stage.shared_output, 1);
            set_buffer(encoder, stage.expert_outputs, 2);
            set_buffer(encoder, stage.routing_weights, 3);
            set_buffer(encoder, grouped_indices, 4);
            set_buffer(encoder, shared_router, 5);
            set_buffer(encoder, stage.hidden, 6);
            set_bytes(encoder, u32_as_bytes(&rows), 7);
            set_bytes(encoder, u32_as_bytes(&layout.width_u32), 8);
            set_bytes(encoder, u32_as_bytes(&layout.active_count_u32), 9);
            set_bytes(encoder, u32_as_bytes(&shared_router_width), 10);
            dispatch_threads(encoder, stage.plan.dispatch_threads);
            Ok(())
        }
    }

    unsafe fn encode_dense_swiglu(
        &self,
        encoder: MetalObjcId,
        payload: &ScheduledDenseExpertPhaseMlpPayload<'_>,
        stage: MetalCmd3ActiveExpertStageBuffers,
        buffers: &mut Vec<MetalPhaseBuffer>,
        source_buffers: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<()> {
        unsafe {
            let gate_out = stage
                .work
                .gate_out
                .context("Metal dense expert stage has no gate projection buffer")?;
            let up_out = stage
                .work
                .up_out
                .context("Metal dense expert stage has no up projection buffer")?;
            self.encode_dense_expert_matvec(
                encoder,
                &payload.gate,
                stage.normed,
                gate_out,
                0,
                buffers,
                source_buffers,
            )?;
            self.encode_dense_expert_matvec(
                encoder,
                &payload.up,
                stage.normed,
                up_out,
                0,
                buffers,
                source_buffers,
            )?;
            let intermediate = stage.plan.intermediate_u32()?;
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                self.runtime.pipelines.silu_product_pipeline,
            );
            set_buffer(encoder, gate_out, 0);
            set_buffer(encoder, up_out, 1);
            set_buffer(encoder, stage.activated, 2);
            set_bytes(encoder, u32_as_bytes(&intermediate), 3);
            dispatch_threads(encoder, stage.plan.intermediate as u64);
            self.encode_dense_expert_matvec(
                encoder,
                &payload.down,
                stage.activated,
                stage.expert_outputs,
                stage.output_offset,
                buffers,
                source_buffers,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_dense_expert_matvec(
        &self,
        encoder: MetalObjcId,
        payload: &super::experts::DenseMatvecPayload<'_>,
        input: MetalObjcId,
        output: MetalObjcId,
        output_offset: u64,
        buffers: &mut Vec<MetalPhaseBuffer>,
        source_buffers: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<()> {
        unsafe {
            let source = payload.source;
            let buffer = self.expert_source_buffer(
                source.bytes,
                source.reusable_bytes,
                buffers,
                source_buffers,
            )?;
            let rows =
                u32::try_from(payload.rows).context("Metal CMD3 dense expert rows exceed u32")?;
            let cols =
                u32::try_from(payload.cols).context("Metal CMD3 dense expert cols exceed u32")?;
            let byte_offset = u64::try_from(source.byte_offset)
                .context("Metal CMD3 dense expert byte offset exceeds u64")?;
            let pipeline = match payload.dtype {
                super::experts::DenseExpertDtype::Bf16 => {
                    self.runtime.pipelines.dense_mmap_bf16_pipeline
                }
                super::experts::DenseExpertDtype::F16 => {
                    self.runtime.pipelines.dense_mmap_f16_pipeline
                }
            };
            msg_send_void1_id(encoder, sel("setComputePipelineState:"), pipeline);
            set_buffer(encoder, buffer, 0);
            set_buffer(encoder, input, 1);
            set_buffer_with_offset(encoder, output, output_offset, 2);
            set_bytes(encoder, u64_as_bytes(&byte_offset), 3);
            set_bytes(encoder, u32_as_bytes(&rows), 4);
            set_bytes(encoder, u32_as_bytes(&cols), 5);
            dispatch_threads(encoder, payload.rows as u64);
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_q4_matvec(
        &self,
        encoder: MetalObjcId,
        payload: &super::experts::Q4MatvecPayload<'_>,
        source: super::experts::Q4MatvecSource<'_>,
        input: MetalObjcId,
        output: MetalObjcId,
        output_offset: u64,
        buffers: &mut Vec<MetalPhaseBuffer>,
        source_buffers: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<()> {
        unsafe {
            let buffer = self.expert_source_buffer(
                source.bytes,
                source.reusable_bytes,
                buffers,
                source_buffers,
            )?;
            let rows = u32::try_from(payload.rows).context("Metal CMD3 expert rows exceed u32")?;
            let cols = u32::try_from(payload.cols).context("Metal CMD3 expert cols exceed u32")?;
            let groups = u32::try_from(payload.cols.div_ceil(payload.group_size).max(1))
                .context("Metal CMD3 expert groups exceed u32")?;
            let group_size = u32::try_from(payload.group_size)
                .context("Metal CMD3 expert group size exceeds u32")?;
            let pipeline = if payload
                .scale_bias_dtype
                .eq_ignore_ascii_case(super::experts::EXPERT_SCALE_DTYPE_E8M0)
            {
                self.runtime.pipelines.mxfp4_e8m0_pipeline
            } else {
                self.runtime.pipelines.q4_bf16_scale_bias_pipeline
            };
            msg_send_void1_id(encoder, sel("setComputePipelineState:"), pipeline);
            set_buffer_with_offset(encoder, buffer, source.packed_offset as u64, 0);
            set_buffer(encoder, input, 1);
            set_buffer_with_offset(encoder, buffer, source.scale_offset as u64, 2);
            set_buffer_with_offset(encoder, buffer, source.bias_offset as u64, 3);
            set_buffer_with_offset(encoder, output, output_offset, 4);
            set_bytes(encoder, u32_as_bytes(&rows), 5);
            set_bytes(encoder, u32_as_bytes(&cols), 6);
            set_bytes(encoder, u32_as_bytes(&groups), 7);
            set_bytes(encoder, u32_as_bytes(&group_size), 8);
            dispatch_q4_threadgroups(encoder, payload.rows as u64);
            Ok(())
        }
    }

    unsafe fn expert_source_buffer(
        &self,
        bytes: &[u8],
        reusable_bytes: Option<&ReusableExpertBytes>,
        buffers: &mut Vec<MetalPhaseBuffer>,
        cache: &mut MetalExpertSourceBufferCache,
    ) -> anyhow::Result<MetalObjcId> {
        if let Some(buffer) = cache.get(bytes) {
            return Ok(buffer);
        }
        if let Some(reusable_bytes) = reusable_bytes
            && let Some(buffer) = unsafe {
                persistent_expert_source_buffer(
                    self.runtime.device,
                    bytes,
                    reusable_bytes,
                    self.buffers.as_ref(),
                )?
            }
        {
            cache.insert(bytes, buffer);
            return Ok(buffer);
        }
        let phase = unsafe { self.expert_source_phase_buffer(bytes)? };
        let buffer = phase.id;
        buffers.push(phase);
        cache.insert(bytes, buffer);
        Ok(buffer)
    }

    unsafe fn expert_source_phase_buffer(&self, bytes: &[u8]) -> anyhow::Result<MetalPhaseBuffer> {
        unsafe {
            if let Some(buffer) = wrap_expert_slot_as_metal_buffer(self.runtime.device, bytes) {
                return Ok(MetalPhaseBuffer::borrowed_expert(buffer));
            }
            let buffer = self
                .buffers
                .transient_expert_buffer_with_bytes(self.runtime.device, bytes)?;
            Ok(MetalPhaseBuffer::transient_expert(buffer))
        }
    }

    unsafe fn phase_buffer(
        &self,
        bytes: usize,
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            let buffer = self.buffers.buffer_with_len(self.runtime.device, bytes)?;
            buffers.push(MetalPhaseBuffer::recyclable(buffer));
            Ok(buffer)
        }
    }

    unsafe fn phase_buffer_with_bytes(
        &self,
        bytes: &[u8],
        buffers: &mut Vec<MetalPhaseBuffer>,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe {
            let buffer = self.buffers.buffer_with_bytes(self.runtime.device, bytes)?;
            buffers.push(MetalPhaseBuffer::recyclable(buffer));
            Ok(buffer)
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalFusedLinearAttentionBuilder<'a> {
    runtime: &'a MetalRuntime,
    dense_weights: Option<&'a MetalDenseWeights>,
    linear_attention_state: &'a Mutex<MetalLinearAttentionStateCache>,
    buffers: &'a MetalBufferPool,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<'a> MetalFusedLinearAttentionBuilder<'a> {
    pub(crate) fn new(
        runtime: &'a MetalRuntime,
        dense_weights: Option<&'a MetalDenseWeights>,
        linear_attention_state: &'a Mutex<MetalLinearAttentionStateCache>,
        buffers: &'a MetalBufferPool,
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
fn validate_resident_projection(
    projection: &ResidentMmapMatvecProjection,
    input_len: usize,
    dense_len: usize,
) -> Result<()> {
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
    match projection {
        ResidentMmapMatvecProjection::Q4(projection) => {
            if projection.row_packed_bytes != projection.cols.div_ceil(2) {
                bail!(
                    "resident Q4 projection {} row packed bytes {} do not match cols {}",
                    projection.tensor_name,
                    projection.row_packed_bytes,
                    projection.cols
                );
            }
            let scale_bias_bytes = expert_scale_bias_dtype_size(&projection.scale_bias_dtype)?;
            if projection.scales_byte_offset % scale_bias_bytes as u64 != 0
                || projection.biases_byte_offset % scale_bias_bytes as u64 != 0
            {
                bail!(
                    "resident Q4 projection {} has unaligned scale/bias offsets",
                    projection.tensor_name
                );
            }
            let packed_len = projection
                .rows
                .checked_mul(projection.row_packed_bytes)
                .context("resident Q4 projection packed byte length overflow")?;
            let group_bytes = projection
                .rows
                .checked_mul(projection.groups_per_row)
                .and_then(|groups| groups.checked_mul(scale_bias_bytes))
                .context("resident Q4 projection group byte length overflow")?;
            for (offset, len, label) in [
                (projection.packed_byte_offset, packed_len, "packed"),
                (projection.scales_byte_offset, group_bytes, "scales"),
                (projection.biases_byte_offset, group_bytes, "biases"),
            ] {
                let offset = usize::try_from(offset).with_context(|| {
                    format!("resident Q4 projection {label} offset does not fit usize")
                })?;
                if offset.checked_add(len).map_or(true, |end| end > dense_len) {
                    bail!(
                        "resident Q4 projection {label} range for {} exceeds resident dense weights",
                        projection.tensor_name
                    );
                }
            }
        }
        ResidentMmapMatvecProjection::Dense(projection) => {
            let element_size = match projection.dtype.to_ascii_uppercase().as_str() {
                "BF16" | "BFLOAT16" | "F16" | "FLOAT16" | "FP16" => 2,
                "F32" | "FLOAT32" | "FP32" => 4,
                _ => bail!(
                    "resident dense projection {} has unsupported Metal dtype {}",
                    projection.tensor_name,
                    projection.dtype
                ),
            };
            if projection.byte_offset % element_size as u64 != 0 {
                bail!(
                    "resident dense projection {} offset {} is unaligned for dtype {}",
                    projection.tensor_name,
                    projection.byte_offset,
                    projection.dtype
                );
            }
            let byte_len = projection
                .rows
                .checked_mul(projection.cols)
                .and_then(|values| values.checked_mul(element_size))
                .context("resident dense projection byte length overflow")?;
            let offset = usize::try_from(projection.byte_offset)
                .context("resident dense projection offset does not fit usize")?;
            if offset
                .checked_add(byte_len)
                .map_or(true, |end| end > dense_len)
            {
                bail!(
                    "resident dense projection {} range exceeds resident dense weights",
                    projection.tensor_name
                );
            }
        }
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn encode_resident_projection(
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
unsafe fn encode_resident_projection_rows(
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
        set_buffer(encoder, dense_weights.buffer, 0);
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
                let pipeline = match projection.dtype.to_ascii_uppercase().as_str() {
                    "BF16" | "BFLOAT16" => pipelines.dense_mmap_bf16_pipeline,
                    "F16" | "FLOAT16" | "FP16" => pipelines.dense_mmap_f16_pipeline,
                    "F32" | "FLOAT32" | "FP32" => pipelines.dense_mmap_f32_pipeline,
                    _ => bail!(
                        "resident dense projection {} has unsupported Metal dtype {}",
                        projection.tensor_name,
                        projection.dtype
                    ),
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
unsafe fn encode_dense_resident_matrix(
    pipelines: &MetalPipelineSet<MetalObjcId>,
    encoder: MetalObjcId,
    dense_weights: &MetalDenseWeights,
    projection: &DenseMmapMatvecProjection,
    input_buffer: MetalObjcId,
    output_buffer: MetalObjcId,
    input_rows: usize,
) -> Result<()> {
    unsafe {
        let pipeline = match projection.dtype.to_ascii_uppercase().as_str() {
            "BF16" | "BFLOAT16" => pipelines.dense_matrix_bf16_pipeline,
            "F16" | "FLOAT16" | "FP16" => pipelines.dense_matrix_f16_pipeline,
            "F32" | "FLOAT32" | "FP32" => pipelines.dense_matrix_f32_pipeline,
            _ => bail!(
                "resident dense matrix {} has unsupported dtype {}",
                projection.tensor_name,
                projection.dtype
            ),
        };
        let rows =
            u32::try_from(projection.rows).context("resident dense matrix rows do not fit u32")?;
        let cols =
            u32::try_from(projection.cols).context("resident dense matrix cols do not fit u32")?;
        msg_send_void1_id(encoder, sel("setComputePipelineState:"), pipeline);
        set_buffer(encoder, dense_weights.buffer, 0);
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
    pub(crate) fn execute_recurrent_matrix(
        &self,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        rows: usize,
        qkv: &[f32],
        z: &[f32],
        beta: &[f32],
        alpha: &[f32],
    ) -> Result<Vec<f32>> {
        let layer = bindings.layer;
        let static_tensors = &bindings.static_tensors;
        let expected_qkv = rows
            .checked_mul(layout.conv_dim)
            .context("linear-attention QKV matrix size overflow")?;
        let expected_values = rows
            .checked_mul(layout.total_value_width)
            .context("linear-attention value matrix size overflow")?;
        let expected_heads = rows
            .checked_mul(layout.num_value_heads)
            .context("linear-attention gate matrix size overflow")?;
        if rows == 0
            || qkv.len() != expected_qkv
            || z.len() != expected_values
            || beta.len() != expected_heads
            || alpha.len() != expected_heads
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
                "Qwen layer-major linear-attention matrix has incompatible geometry at layer {layer}: rows={rows} qkv={} expected={expected_qkv} z={} expected_values={expected_values} beta={} alpha={} expected_heads={expected_heads}",
                qkv.len(),
                z.len(),
                beta.len(),
                alpha.len()
            );
        }
        let dense_weights = self
            .dense_weights
            .as_ref()
            .context("Qwen layer-major linear attention requires resident dense Metal weights")?;
        let mut state_guard = self
            .linear_attention_state
            .lock()
            .expect("metal linear attention state poisoned");
        let state = state_guard
            .layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .with_context(|| {
                format!("Qwen layer-major linear-attention state has no resolved layer {layer}")
            })?;
        if state.conv_dim != layout.conv_dim
            || state.total_value_width != layout.total_value_width
            || state.num_value_heads != layout.num_value_heads
            || state.conv_state_len != layout.conv_state_len()
            || state.ssm_state_len != layout.ssm_state_len()
        {
            bail!("Qwen layer-major linear-attention recurrent state does not match layer {layer}");
        }

        unsafe {
            let qkv_buffer = self.buffer_with_bytes(f32_as_bytes(qkv))?;
            let z_buffer = self.buffer_with_bytes(f32_as_bytes(z))?;
            let beta_buffer = self.buffer_with_bytes(f32_as_bytes(beta))?;
            let alpha_buffer = self.buffer_with_bytes(f32_as_bytes(alpha))?;
            let output_buffer =
                self.buffer_with_len(expected_values * std::mem::size_of::<f32>())?;
            let buffers = vec![
                qkv_buffer,
                z_buffer,
                beta_buffer,
                alpha_buffer,
                output_buffer,
            ];
            let mut encoding = match MetalCommandEncoding::new(
                self.command_queue,
                Arc::clone(self.buffers.resources()),
                "failed to create Qwen layer-major linear-attention command buffer",
                "failed to create Qwen layer-major linear-attention encoder",
            ) {
                Ok(encoding) => encoding,
                Err(error) => {
                    self.recycle_or_release_buffers(&buffers, true);
                    drop(state_guard);
                    return Err(error);
                }
            };
            let encoder = encoding.encoder();
            let encode_result = (|| -> Result<()> {
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
                    let qkv_offset = (row * layout.conv_dim * std::mem::size_of::<f32>()) as u64;
                    let z_offset =
                        (row * layout.total_value_width * std::mem::size_of::<f32>()) as u64;
                    let gate_offset =
                        (row * layout.num_value_heads * std::mem::size_of::<f32>()) as u64;
                    let output_offset = z_offset;

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
                    set_buffer_with_offset(encoder, qkv_buffer, qkv_offset, 1);
                    set_buffer_with_offset(
                        encoder,
                        dense_weights.buffer,
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
                    set_buffer_with_offset(encoder, alpha_buffer, gate_offset, 0);
                    set_buffer_with_offset(encoder, beta_buffer, gate_offset, 1);
                    set_buffer_with_offset(
                        encoder,
                        dense_weights.buffer,
                        static_tensors.a_log.byte_offset,
                        2,
                    );
                    set_buffer_with_offset(
                        encoder,
                        dense_weights.buffer,
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
                    set_buffer_with_offset(encoder, z_buffer, z_offset, 1);
                    set_buffer_with_offset(
                        encoder,
                        dense_weights.buffer,
                        static_tensors.norm_weight.byte_offset,
                        2,
                    );
                    set_buffer_with_offset(encoder, output_buffer, output_offset, 3);
                    set_bytes(encoder, u32_as_bytes(&value_dim_u32), 4);
                    set_bytes(encoder, f32_as_bytes(std::slice::from_ref(&eps)), 5);
                    msg_send_void2_size(
                        encoder,
                        sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                        MetalDispatchSize::new(layout.num_value_heads as u64, 1, 1),
                        MetalDispatchSize::new(layout.value_dim as u64, 1, 1),
                    );
                }
                Ok(())
            })();
            if let Err(error) = encode_result {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, true);
                drop(state_guard);
                return Err(error);
            }
            encoding.end_encoding();
            let context = MetalCommandContext::new("qwen_layer_major_linear_attention")
                .with("layer", layer)
                .with("rows", rows)
                .with("value_width", layout.total_value_width);
            if let Err(error) =
                commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)
            {
                drop(encoding);
                self.recycle_or_release_buffers(&buffers, error.should_release_buffers());
                drop(state_guard);
                return Err(error.into());
            }
            let output = self.buffers.read_f32_buffer(output_buffer, expected_values);
            drop(encoding);
            for buffer in buffers {
                self.recycle(buffer);
            }
            drop(state_guard);
            Ok(output)
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
                dense_weights.buffer,
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
                dense_weights.buffer,
                static_tensors.a_log.byte_offset,
                2,
            );
            set_buffer_with_offset(
                encoder,
                dense_weights.buffer,
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
                dense_weights.buffer,
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
impl MetalLayerMajorPostAttention {
    fn new(
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
    fn new(
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

    pub(crate) fn materialize(&self) -> (Vec<f32>, Option<Vec<f32>>) {
        unsafe {
            let hidden = self
                .buffers
                .read_f32_buffer(self.hidden.buffer(), self.hidden.values());
            let next_normed = self.next_normed.map(|matrix| {
                self.buffers
                    .read_f32_buffer(matrix.buffer(), matrix.values())
            });
            (hidden, next_normed)
        }
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
        set_buffer(encoder, dense_weights.buffer, 0);
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
        validate_resident_projection(
            &ResidentMmapMatvecProjection::Q4(embed_q.clone()),
            input.nope_dim,
            dense_weights.len,
        )?;
        validate_resident_projection(
            &ResidentMmapMatvecProjection::Q4(unembed_out.clone()),
            input.latent_rank,
            dense_weights.len,
        )?;

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
        validate_resident_projection(
            &ResidentMmapMatvecProjection::Q4(projection.clone()),
            projection.cols,
            dense_weights.len,
        )?;

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
        validate_resident_projection(
            &ResidentMmapMatvecProjection::Q4(embed_q.clone()),
            embed_q.cols,
            dense_weights.len,
        )?;
        validate_resident_projection(
            &ResidentMmapMatvecProjection::Q4(unembed_out.clone()),
            input.latent_rank,
            dense_weights.len,
        )?;

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

    unsafe fn try_encode_q4_mmap_projection_batch(
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
            set_buffer(encoder, dense_weights.buffer, 0);
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
            dispatch_q4_mmap_matrix_threadgroups(encoder, total_rows as u64, input_rows as u64);
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

    pub(crate) fn execute_matrix(
        &self,
        projections: &[ResidentMmapMatvecProjection],
        input_rows: usize,
        input_cols: usize,
        input: &[f32],
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
            let input_buffer = self.buffer_with_bytes(f32_as_bytes(input))?;
            let output_buffer =
                self.buffer_with_len(total_output_values * std::mem::size_of::<f32>())?;
            let mut buffers = vec![input_buffer, output_buffer];
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
        attention: &[f32],
        residual: &[f32],
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
            let attention_buffer = self.buffer_with_bytes(f32_as_bytes(attention))?;
            let residual_input_buffer = self.buffer_with_bytes(f32_as_bytes(residual))?;
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
                attention_buffer,
                residual_input_buffer,
                norm_weight_buffer,
                projected_buffer,
                hidden_buffer,
                normed_buffer,
                router_buffer,
            ];
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

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(crate) struct MetalPostAttentionPrep {
    pub(crate) residual_buffer: MetalObjcId,
    pub(crate) normed_buffer: MetalObjcId,
    pub(crate) input: ScheduledCmd3MetalPostAttentionInput,
    pub(crate) state: FlashMoePostAttentionPrepState,
    pub(crate) width: usize,
    pub(crate) active: Vec<(usize, f32)>,
    routing_command: Option<ScheduledRoutingCommand>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalPostAttentionPrep {
    pub(crate) fn new(
        layer: usize,
        width: usize,
        expert_count: usize,
        active: Vec<(usize, f32)>,
        residual_buffer: MetalObjcId,
        normed_buffer: MetalObjcId,
    ) -> anyhow::Result<Self> {
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

    fn from_shape(
        source: MetalCmd3SharedPhaseSource,
        width: usize,
        shape: super::weights::SharedExpertPhaseShape,
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

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalPhaseBufferClass {
    General,
    BorrowedExpert,
    TransientExpert,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalPhaseBuffer {
    pub(crate) id: MetalObjcId,
    pub(crate) class: MetalPhaseBufferClass,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalPhaseBuffer {
    pub(crate) fn recyclable(id: MetalObjcId) -> Self {
        Self {
            id,
            class: MetalPhaseBufferClass::General,
        }
    }

    pub(crate) fn transient_expert(id: MetalObjcId) -> Self {
        Self {
            id,
            class: MetalPhaseBufferClass::TransientExpert,
        }
    }

    pub(crate) fn borrowed_expert(id: MetalObjcId) -> Self {
        Self {
            id,
            class: MetalPhaseBufferClass::BorrowedExpert,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> MetalObjcId;
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[link(name = "objc")]
unsafe extern "C" {
    fn objc_getClass(name: *const c_char) -> MetalObjcId;
    fn objc_retainAutoreleasedReturnValue(value: MetalObjcId) -> MetalObjcId;
    fn sel_registerName(name: *const c_char) -> MetalSelector;
    fn objc_msgSend();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn metal_default_device() -> MetalObjcId {
    unsafe { MTLCreateSystemDefaultDevice() }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn sel(name: &str) -> MetalSelector {
    let name = CString::new(name).expect("selector contains nul");
    unsafe { sel_registerName(name.as_ptr()) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn class(name: &str) -> MetalObjcId {
    let name = CString::new(name).expect("class contains nul");
    unsafe { objc_getClass(name.as_ptr()) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn ns_string(value: &str) -> MetalObjcId {
    unsafe {
        let alloc = msg_send_id0(class("NSString"), sel("alloc"));
        msg_send_id3_ptr_usize_u64(
            alloc,
            sel("initWithBytes:length:encoding:"),
            value.as_ptr().cast(),
            value.len(),
            4,
        )
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn ns_error_localized_description(error: MetalObjcId) -> Option<String> {
    unsafe {
        if error.is_null() {
            return None;
        }
        let description = msg_send_id0(error, sel("localizedDescription"));
        if description.is_null() {
            return None;
        }
        let bytes = msg_send_const_char_ptr0(description, sel("UTF8String"));
        if bytes.is_null() {
            return None;
        }
        Some(CStr::from_ptr(bytes).to_string_lossy().into_owned())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn new_function(library: MetalObjcId, name: &str) -> anyhow::Result<MetalObjcId> {
    unsafe {
        let function_name = ns_string(name);
        let function = msg_send_id1_id(library, sel("newFunctionWithName:"), function_name);
        release(function_name);
        if function.is_null() {
            anyhow::bail!("compiled Flash-MoE Metal library is missing kernel `{name}`");
        }
        Ok(function)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn compile_pipeline(
    device: MetalObjcId,
    library: MetalObjcId,
    name: &str,
) -> anyhow::Result<MetalObjcId> {
    unsafe {
        let function = new_function(library, name)?;
        let pipeline = new_compute_pipeline(device, function)
            .with_context(|| format!("failed to create {name} Metal pipeline"))?;
        release(function);
        Ok(pipeline)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn compile_pipeline_with_constants(
    device: MetalObjcId,
    library: MetalObjcId,
    name: &str,
    constants: &[(u64, u64, &[u8])],
) -> anyhow::Result<MetalObjcId> {
    unsafe {
        let alloc = msg_send_id0(class("MTLFunctionConstantValues"), sel("alloc"));
        let values = OwnedMetalObject::new(msg_send_id0(alloc, sel("init")))?;
        for &(index, data_type, bytes) in constants {
            msg_send_void3_ptr_u64_u64(
                values.id(),
                sel("setConstantValue:type:atIndex:"),
                bytes.as_ptr().cast(),
                data_type,
                index,
            );
        }
        let function_name = OwnedMetalObject::new(ns_string(name))?;
        let mut error = ptr::null_mut();
        let function = OwnedMetalObject::new(msg_send_id2_id_error(
            library,
            sel("newFunctionWithName:constantValues:error:"),
            function_name.id(),
            values.id(),
            &mut error,
        ))
        .with_context(|| {
            let detail = ns_error_localized_description(error)
                .unwrap_or_else(|| "unknown Metal specialization error".to_string());
            format!("failed to specialize DeepSeek V4 Flash Metal kernel {name}: {detail}")
        })?;
        let pipeline = new_compute_pipeline(device, function.id())
            .with_context(|| format!("failed to create specialized {name} Metal pipeline"))?;
        Ok(pipeline)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn new_compute_pipeline(
    device: MetalObjcId,
    function: MetalObjcId,
) -> anyhow::Result<MetalObjcId> {
    unsafe {
        let pipeline = msg_send_id3(
            device,
            sel("newComputePipelineStateWithFunction:error:"),
            function,
        );
        if pipeline.is_null() {
            anyhow::bail!("failed to create Flash-MoE Metal compute pipeline");
        }
        Ok(pipeline)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn metal_page_size() -> usize {
    unsafe {
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        if page_size > 0 {
            page_size as usize
        } else {
            16 * 1024
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn wrap_expert_slot_as_metal_buffer(
    device: MetalObjcId,
    bytes: &[u8],
) -> Option<MetalObjcId> {
    if bytes.is_empty() {
        return None;
    }
    let page_size = metal_page_size();
    let ptr = bytes.as_ptr() as *mut c_void;
    if (ptr as usize) % page_size != 0 || bytes.len() % page_size != 0 {
        return None;
    }
    unsafe {
        let buffer = msg_send_id4_ptr_usize_u64_ptr(
            device,
            sel("newBufferWithBytesNoCopy:length:options:deallocator:"),
            ptr,
            bytes.len(),
            0,
            ptr::null_mut(),
        );
        (!buffer.is_null()).then_some(buffer)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn persistent_expert_source_buffer(
    device: MetalObjcId,
    bytes: &[u8],
    reusable_bytes: &ReusableExpertBytes,
    buffers: &MetalBufferPool,
) -> anyhow::Result<Option<MetalObjcId>> {
    let whole_slot = reusable_bytes.as_slice();
    if whole_slot.as_ptr() != bytes.as_ptr() || whole_slot.len() != bytes.len() {
        return Ok(None);
    }
    if let Some(attachment) = reusable_bytes.attachment::<MetalPersistentExpertBuffer>() {
        return Ok(attachment.buffer_for_device(device));
    }
    unsafe {
        buffers.ensure_allocation_capacity(device, bytes.len())?;
    }
    let Some(buffer) = (unsafe { wrap_expert_slot_as_metal_buffer(device, bytes) }) else {
        return Ok(None);
    };
    Ok(reusable_bytes
        .install_attachment(MetalPersistentExpertBuffer::new(
            device,
            buffer,
            bytes.len(),
            Arc::clone(buffers.resources()),
        ))
        .and_then(|attachment| attachment.buffer_for_device(device)))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn wrap_dense_mmap_as_metal_buffer(
    device: MetalObjcId,
    mmap: Arc<memmap2::Mmap>,
    len: u64,
) -> anyhow::Result<Option<MetalDenseWeights>> {
    let len = usize::try_from(len).context("dense mmap length does not fit usize")?;
    if len == 0 {
        return Ok(None);
    }
    let ptr = mmap.as_ptr() as *mut c_void;
    let page_size = metal_page_size();
    if (ptr as usize) % page_size != 0 {
        tracing::debug!(
            ptr = ?ptr,
            page_size,
            "dense mmap is not page-aligned; resident Metal dense buffer disabled"
        );
        return Ok(None);
    }
    unsafe {
        let buffer = msg_send_id4_ptr_usize_u64_ptr(
            device,
            sel("newBufferWithBytesNoCopy:length:options:deallocator:"),
            ptr,
            len,
            0,
            ptr::null_mut(),
        );
        if buffer.is_null() {
            tracing::debug!(len, "failed to wrap dense mmap as resident Metal buffer");
            return Ok(None);
        }
        Ok(Some(MetalDenseWeights::new(buffer, mmap, len)))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn allocate_zeroed_buffer(
    device: MetalObjcId,
    len: usize,
    label: &str,
) -> anyhow::Result<MetalObjcId> {
    let len = len.max(std::mem::size_of::<f32>());
    unsafe {
        let buffer = msg_send_id2_usize_u64(device, sel("newBufferWithLength:options:"), len, 0);
        if buffer.is_null() {
            anyhow::bail!("failed to allocate Flash-MoE Metal {label} buffer ({len} bytes)");
        }
        let contents = msg_send_ptr0(buffer, sel("contents"));
        ptr::write_bytes(contents.cast::<u8>(), 0, len);
        Ok(buffer)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn zero_buffer(buffer: MetalObjcId, f32_len: usize) {
    unsafe {
        let contents = msg_send_ptr0(buffer, sel("contents"));
        ptr::write_bytes(
            contents.cast::<u8>(),
            0,
            f32_len.saturating_mul(std::mem::size_of::<f32>()),
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn release_linear_attention_layer(layer: &MetalLinearAttentionLayerState) {
    unsafe {
        release(layer.conv_state);
        release(layer.ssm_state);
        release(layer.conv_output);
        release(layer.delta_output);
        release(layer.g_decay);
        release(layer.beta_gate);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn release_linear_attention_state(state: &mut MetalLinearAttentionStateCache) {
    unsafe {
        for layer in state.layers.iter_mut().filter_map(Option::take) {
            release_linear_attention_layer(&layer);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn linear_attention_state_bytes(state: &MetalLinearAttentionStateCache) -> usize {
    state
        .layers
        .iter()
        .flatten()
        .map(|layer| {
            layer
                .conv_state_len
                .saturating_add(layer.ssm_state_len)
                .saturating_add(layer.conv_dim)
                .saturating_add(layer.total_value_width)
                .saturating_add(layer.num_value_heads.saturating_mul(2))
                .saturating_mul(std::mem::size_of::<f32>())
        })
        .fold(0usize, usize::saturating_add)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn allocate_linear_attention_state(
    device: MetalObjcId,
    layouts: &[Option<LinearAttentionLayout>],
) -> anyhow::Result<MetalLinearAttentionStateCache> {
    let mut cache = MetalLinearAttentionStateCache::new(Vec::with_capacity(layouts.len()));
    for (layer, layout) in layouts.iter().copied().enumerate() {
        let Some(layout) = layout else {
            cache.layers.push(None);
            continue;
        };
        let state = FlashMoeLinearAttentionCacheState::gpu_resident(
            layer,
            layout.conv_state_len(),
            layout.ssm_state_len(),
            layout.conv_dim,
            layout.total_value_width,
        );
        if state.layer() != layer || !state.is_declared_graph_state() {
            unsafe { release_linear_attention_state(&mut cache) };
            anyhow::bail!(
                "FlashMoe Metal linear-attention cache state for layer {layer} is not declared graph state"
            );
        }

        let allocation = (|| -> anyhow::Result<MetalLinearAttentionLayerState> {
            let mut owned = Vec::with_capacity(6);
            let mut allocate = |len: usize, label: &str| -> anyhow::Result<MetalObjcId> {
                match allocate_zeroed_buffer(
                    device,
                    len.saturating_mul(std::mem::size_of::<f32>()),
                    label,
                ) {
                    Ok(buffer) => {
                        owned.push(buffer);
                        Ok(buffer)
                    }
                    Err(error) => {
                        unsafe {
                            for buffer in owned.drain(..) {
                                release(buffer);
                            }
                        }
                        Err(error)
                    }
                }
            };
            let conv_state = allocate(state.conv_state_len(), "linear conv state")?;
            let ssm_state = allocate(state.ssm_state_len(), "linear SSM state")?;
            let conv_output = allocate(state.conv_output_len(), "linear conv output")?;
            let delta_output = allocate(state.output_len(), "linear delta output")?;
            let g_decay = allocate(layout.num_value_heads, "linear decay")?;
            let beta_gate = allocate(layout.num_value_heads, "linear beta gate")?;
            Ok(MetalLinearAttentionLayerState::new(
                conv_state,
                ssm_state,
                conv_output,
                delta_output,
                g_decay,
                beta_gate,
                state.conv_state_len(),
                state.ssm_state_len(),
                layout.conv_dim,
                layout.total_value_width,
                layout.num_value_heads,
            ))
        })();
        match allocation {
            Ok(state) => cache.layers.push(Some(state)),
            Err(error) => {
                unsafe { release_linear_attention_state(&mut cache) };
                return Err(error).with_context(|| {
                    format!("failed to allocate linear-attention state for layer {layer}")
                });
            }
        }
    }
    Ok(cache)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn set_buffer(encoder: MetalObjcId, buffer: MetalObjcId, index: u64) {
    unsafe { set_buffer_with_offset(encoder, buffer, 0, index) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn set_buffer_with_offset(
    encoder: MetalObjcId,
    buffer: MetalObjcId,
    offset: u64,
    index: u64,
) {
    unsafe {
        msg_send_void4(
            encoder,
            sel("setBuffer:offset:atIndex:"),
            buffer,
            offset,
            index,
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn set_bytes(encoder: MetalObjcId, bytes: &[u8], index: u64) {
    unsafe {
        msg_send_void3_ptr_usize_u64(
            encoder,
            sel("setBytes:length:atIndex:"),
            bytes.as_ptr().cast(),
            bytes.len(),
            index,
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn read_f32_buffer(buffer: MetalObjcId, len: usize) -> Vec<f32> {
    unsafe {
        let contents = msg_send_ptr0(buffer, sel("contents"));
        let mut output = vec![0.0f32; len];
        ptr::copy_nonoverlapping(contents.cast::<f32>(), output.as_mut_ptr(), len);
        output
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn read_f32_buffer_offset(buffer: MetalObjcId, offset: usize, len: usize) -> Vec<f32> {
    unsafe {
        let contents = msg_send_ptr0(buffer, sel("contents")).cast::<f32>();
        let mut output = vec![0.0f32; len];
        ptr::copy_nonoverlapping(contents.add(offset), output.as_mut_ptr(), len);
        output
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn write_f32_buffer(buffer: MetalObjcId, values: &[f32]) {
    unsafe {
        let contents = msg_send_ptr0(buffer, sel("contents"));
        ptr::copy_nonoverlapping(values.as_ptr(), contents.cast::<f32>(), values.len());
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn dispatch_threads(encoder: MetalObjcId, threads: u64) {
    unsafe { dispatch_metal_plan(encoder, MetalDispatchPlan::threads(threads)) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn dispatch_q4_threadgroups(encoder: MetalObjcId, rows: u64) {
    unsafe { dispatch_metal_plan(encoder, MetalDispatchPlan::q4_threadgroups(rows)) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn dispatch_q4_mmap_threadgroups(encoder: MetalObjcId, rows: u64) {
    unsafe { dispatch_metal_plan(encoder, MetalDispatchPlan::q4_mmap_threadgroups(rows)) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn dispatch_q4_mmap_matrix_threadgroups(
    encoder: MetalObjcId,
    rows: u64,
    input_rows: u64,
) {
    unsafe {
        dispatch_metal_plan(
            encoder,
            MetalDispatchPlan::q4_mmap_matrix_threadgroups(rows, input_rows),
        )
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn dispatch_single_threadgroup(encoder: MetalObjcId, threads: u64) {
    unsafe { dispatch_metal_plan(encoder, MetalDispatchPlan::single_threadgroup(threads)) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn dispatch_metal_plan(encoder: MetalObjcId, plan: MetalDispatchPlan) {
    unsafe {
        let selector = match plan.mode {
            MetalDispatchMode::Threads => sel("dispatchThreads:threadsPerThreadgroup:"),
            MetalDispatchMode::Threadgroups => sel("dispatchThreadgroups:threadsPerThreadgroup:"),
        };
        msg_send_void2_size(encoder, selector, plan.grid, plan.threadgroup);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn f32_as_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn u32_as_bytes(value: &u32) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            (value as *const u32).cast::<u8>(),
            std::mem::size_of::<u32>(),
        )
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn u32_as_bytes_slice(values: &[u32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn u64_as_bytes(value: &u64) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            (value as *const u64).cast::<u8>(),
            std::mem::size_of::<u64>(),
        )
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn u64_as_bytes_slice(values: &[u64]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn commit_metal_command_buffer(
    command_buffer: MetalObjcId,
    context: &MetalCommandContext,
) {
    unsafe {
        set_metal_command_buffer_label(command_buffer, context);
        msg_send_void0(command_buffer, sel("commit"));
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn commit_and_wait_metal_command_buffer(
    command_buffer: MetalObjcId,
    context: &MetalCommandContext,
) -> std::result::Result<(), MetalCommandBufferFailure> {
    unsafe {
        commit_metal_command_buffer(command_buffer, context);
        wait_for_metal_command_buffer(command_buffer, context)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn set_metal_command_buffer_label(
    command_buffer: MetalObjcId,
    context: &MetalCommandContext,
) {
    unsafe {
        let label = ns_string(&context.label());
        msg_send_void1_id(command_buffer, sel("setLabel:"), label);
        release(label);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn wait_for_metal_command_buffer(
    command_buffer: MetalObjcId,
    context: &MetalCommandContext,
) -> std::result::Result<(), MetalCommandBufferFailure> {
    let started = Instant::now();
    let policy = MetalCommandWaitPolicy::default();
    loop {
        let status = unsafe { metal_command_buffer_status(command_buffer) };
        let elapsed = started.elapsed();
        let timed_out = elapsed >= policy.timeout;
        let metal_error = if status.is_terminal() || timed_out {
            unsafe { metal_command_buffer_error(command_buffer) }
        } else {
            None
        };
        match resolve_metal_command_wait(context, elapsed, status, metal_error, timed_out) {
            MetalCommandWaitResult::Pending => thread::sleep(policy.poll_interval),
            MetalCommandWaitResult::Finished(result) => return result,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn metal_command_buffer_status(command_buffer: MetalObjcId) -> MetalCommandStatus {
    unsafe { MetalCommandStatus::from_raw(msg_send_usize0(command_buffer, sel("status"))) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn metal_command_buffer_error(command_buffer: MetalObjcId) -> Option<String> {
    unsafe {
        let error = msg_send_id0(command_buffer, sel("error"));
        ns_error_localized_description(error)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn release(receiver: MetalObjcId) {
    unsafe {
        if !receiver.is_null() {
            msg_send_void0(receiver, sel("release"));
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn purge_and_release_metal_buffer(buffer: MetalObjcId) {
    unsafe {
        // MTLResourcePurgeableStateEmpty. Expert staging is never reused after its command;
        // marking it empty tells IOAccelerator to discard dirty backing and per-binding mappings.
        let _ = msg_send_usize1_usize(buffer, sel("setPurgeableState:"), 4);
        release(buffer);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn retain_autoreleased_return_value(receiver: MetalObjcId) -> MetalObjcId {
    unsafe { objc_retainAutoreleasedReturnValue(receiver) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_id0(receiver: MetalObjcId, selector: MetalSelector) -> MetalObjcId {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector) -> MetalObjcId =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_id1_id(
    receiver: MetalObjcId,
    selector: MetalSelector,
    arg: MetalObjcId,
) -> MetalObjcId {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector, MetalObjcId) -> MetalObjcId =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, arg)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_id3(
    receiver: MetalObjcId,
    selector: MetalSelector,
    arg: MetalObjcId,
) -> MetalObjcId {
    unsafe {
        let f: unsafe extern "C" fn(
            MetalObjcId,
            MetalSelector,
            MetalObjcId,
            *mut MetalObjcId,
        ) -> MetalObjcId = std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, arg, ptr::null_mut())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_id2_id_error(
    receiver: MetalObjcId,
    selector: MetalSelector,
    arg1: MetalObjcId,
    arg2: MetalObjcId,
    error: *mut MetalObjcId,
) -> MetalObjcId {
    unsafe {
        let f: unsafe extern "C" fn(
            MetalObjcId,
            MetalSelector,
            MetalObjcId,
            MetalObjcId,
            *mut MetalObjcId,
        ) -> MetalObjcId = std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, arg1, arg2, error)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_id2_usize_u64(
    receiver: MetalObjcId,
    selector: MetalSelector,
    len: usize,
    options: u64,
) -> MetalObjcId {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector, usize, u64) -> MetalObjcId =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, len, options)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_id3_ptr_usize_u64(
    receiver: MetalObjcId,
    selector: MetalSelector,
    bytes: *const c_void,
    len: usize,
    options: u64,
) -> MetalObjcId {
    unsafe {
        let f: unsafe extern "C" fn(
            MetalObjcId,
            MetalSelector,
            *const c_void,
            usize,
            u64,
        ) -> MetalObjcId = std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, bytes, len, options)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_id4_ptr_usize_u64_ptr(
    receiver: MetalObjcId,
    selector: MetalSelector,
    bytes: *mut c_void,
    len: usize,
    options: u64,
    deallocator: *mut c_void,
) -> MetalObjcId {
    unsafe {
        let f: unsafe extern "C" fn(
            MetalObjcId,
            MetalSelector,
            *mut c_void,
            usize,
            u64,
            *mut c_void,
        ) -> MetalObjcId = std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, bytes, len, options, deallocator)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_void0(receiver: MetalObjcId, selector: MetalSelector) {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector) =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_void1_id(
    receiver: MetalObjcId,
    selector: MetalSelector,
    arg: MetalObjcId,
) {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector, MetalObjcId) =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, arg);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_void2_size(
    receiver: MetalObjcId,
    selector: MetalSelector,
    a: MetalDispatchSize,
    b: MetalDispatchSize,
) {
    unsafe {
        let f: unsafe extern "C" fn(
            MetalObjcId,
            MetalSelector,
            MetalDispatchSize,
            MetalDispatchSize,
        ) = std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, a, b);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_void3_ptr_usize_u64(
    receiver: MetalObjcId,
    selector: MetalSelector,
    bytes: *const c_void,
    len: usize,
    index: u64,
) {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector, *const c_void, usize, u64) =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, bytes, len, index);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_void3_ptr_u64_u64(
    receiver: MetalObjcId,
    selector: MetalSelector,
    value: *const c_void,
    data_type: u64,
    index: u64,
) {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector, *const c_void, u64, u64) =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, value, data_type, index);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_void4(
    receiver: MetalObjcId,
    selector: MetalSelector,
    arg1: MetalObjcId,
    arg2: u64,
    arg3: u64,
) {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector, MetalObjcId, u64, u64) =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, arg1, arg2, arg3);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_ptr0(receiver: MetalObjcId, selector: MetalSelector) -> *mut c_void {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector) -> *mut c_void =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_const_char_ptr0(
    receiver: MetalObjcId,
    selector: MetalSelector,
) -> *const c_char {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector) -> *const c_char =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn msg_send_usize0(receiver: MetalObjcId, selector: MetalSelector) -> usize {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector) -> usize =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn msg_send_usize1_usize(
    receiver: MetalObjcId,
    selector: MetalSelector,
    arg: usize,
) -> usize {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector, usize) -> usize =
            std::mem::transmute(objc_msgSend as *const ());
        f(receiver, selector, arg)
    }
}

pub const METAL_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4_fma_matvec(
    device const uchar* packed [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& groups_per_row [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = scales[scale_row + group];
            float bias = biases[scale_row + group];

            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];

            acc += fma(float((word >>  0) & 0x0f), scale * x0, bias * x0);
            acc += fma(float((word >>  4) & 0x0f), scale * x1, bias * x1);
            acc += fma(float((word >>  8) & 0x0f), scale * x2, bias * x2);
            acc += fma(float((word >> 12) & 0x0f), scale * x3, bias * x3);
            acc += fma(float((word >> 16) & 0x0f), scale * x4, bias * x4);
            acc += fma(float((word >> 20) & 0x0f), scale * x5, bias * x5);
            acc += fma(float((word >> 24) & 0x0f), scale * x6, bias * x6);
            acc += fma(float((word >> 28) & 0x0f), scale * x7, bias * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float scale0 = scales[scale_row + group0];
            float bias0 = biases[scale_row + group0];
            acc += fma(float(byte & 0x0f), scale0 * x0, bias0 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float scale1 = scales[scale_row + group1];
                float bias1 = biases[scale_row + group1];
                acc += fma(float(byte >> 4), scale1 * x1, bias1 * x1);
            }
        }
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        output[row] = sum;
    }
}

inline float bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16u);
}

inline float q4_fma_row_bf16_ptrs(
    device const uchar* packed,
    device const ushort* scales,
    device const ushort* biases,
    device const float* input,
    threadgroup float* input_cache,
    uint row,
    uint cols,
    uint groups_per_row,
    uint group_size,
    bool word_aligned,
    bool use_input_cache,
    uint simd_lane) {
    uint packed_stride = (cols + 1) / 2;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = word_aligned && (cols % 8 == 0) && (group_size % 8 == 0);
    float acc = 0.0f;
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = bf16_to_float(scales[scale_row + group]);
            float bias = bf16_to_float(biases[scale_row + group]);
            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];
            acc += fma(float((word >>  0) & 0x0f), scale * x0, bias * x0);
            acc += fma(float((word >>  4) & 0x0f), scale * x1, bias * x1);
            acc += fma(float((word >>  8) & 0x0f), scale * x2, bias * x2);
            acc += fma(float((word >> 12) & 0x0f), scale * x3, bias * x3);
            acc += fma(float((word >> 16) & 0x0f), scale * x4, bias * x4);
            acc += fma(float((word >> 20) & 0x0f), scale * x5, bias * x5);
            acc += fma(float((word >> 24) & 0x0f), scale * x6, bias * x6);
            acc += fma(float((word >> 28) & 0x0f), scale * x7, bias * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float scale0 = bf16_to_float(scales[scale_row + group0]);
            float bias0 = bf16_to_float(biases[scale_row + group0]);
            acc += fma(float(byte & 0x0f), scale0 * x0, bias0 * x0);
            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float scale1 = bf16_to_float(scales[scale_row + group1]);
                float bias1 = bf16_to_float(biases[scale_row + group1]);
                acc += fma(float(byte >> 4), scale1 * x1, bias1 * x1);
            }
        }
    }
    return simd_sum(acc);
}

kernel void q4_fma_matvec_bf16_scale_bias(
    device const uchar* packed [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const ushort* scales [[buffer(2)]],
    device const ushort* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& groups_per_row [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float sum = q4_fma_row_bf16_ptrs(
        packed, scales, biases, input, input_cache,
        row, cols, groups_per_row, group_size, true, use_input_cache, simd_lane);
    if (simd_lane == 0) {
        output[row] = sum;
    }
}

inline float mxfp4_e2m1_to_float(uchar nibble) {
    float magnitude;
    switch (nibble & 0x07) {
        case 0: magnitude = 0.0f; break;
        case 1: magnitude = 0.5f; break;
        case 2: magnitude = 1.0f; break;
        case 3: magnitude = 1.5f; break;
        case 4: magnitude = 2.0f; break;
        case 5: magnitude = 3.0f; break;
        case 6: magnitude = 4.0f; break;
        default: magnitude = 6.0f; break;
    }
    return (nibble & 0x08) == 0 ? magnitude : -magnitude;
}

inline float mxfp4_e8m0_to_float(uchar bits) {
    return exp2(float(int(bits) - 127));
}

kernel void mxfp4_fma_matvec_e8m0(
    device const uchar* packed [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& groups_per_row [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = mxfp4_e8m0_to_float(scales[scale_row + group]);
            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];
            acc += mxfp4_e2m1_to_float(uchar((word >>  0) & 0x0f)) * scale * x0;
            acc += mxfp4_e2m1_to_float(uchar((word >>  4) & 0x0f)) * scale * x1;
            acc += mxfp4_e2m1_to_float(uchar((word >>  8) & 0x0f)) * scale * x2;
            acc += mxfp4_e2m1_to_float(uchar((word >> 12) & 0x0f)) * scale * x3;
            acc += mxfp4_e2m1_to_float(uchar((word >> 16) & 0x0f)) * scale * x4;
            acc += mxfp4_e2m1_to_float(uchar((word >> 20) & 0x0f)) * scale * x5;
            acc += mxfp4_e2m1_to_float(uchar((word >> 24) & 0x0f)) * scale * x6;
            acc += mxfp4_e2m1_to_float(uchar((word >> 28) & 0x0f)) * scale * x7;
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            float scale0 = mxfp4_e8m0_to_float(scales[scale_row + col0 / group_size]);
            acc += mxfp4_e2m1_to_float(byte & 0x0f) * scale0 * x0;
            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                float scale1 = mxfp4_e8m0_to_float(scales[scale_row + col1 / group_size]);
                acc += mxfp4_e2m1_to_float(byte >> 4) * scale1 * x1;
            }
        }
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        output[row] = sum;
    }
}

kernel void q4_swiglu_fused(
    device const uchar* gate_packed [[buffer(0)]],
    device const uchar* up_packed [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device const float* gate_scales [[buffer(3)]],
    device const float* gate_biases [[buffer(4)]],
    device const float* up_scales [[buffer(5)]],
    device const float* up_biases [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& groups_per_row [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* gate_words = reinterpret_cast<device const uint*>(gate_packed);
        device const uint* up_words = reinterpret_cast<device const uint*>(up_packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint gate_word = gate_words[word_row + packed_word];
            uint up_word = up_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float gate_scale = gate_scales[scale_row + group];
            float gate_bias = gate_biases[scale_row + group];
            float up_scale = up_scales[scale_row + group];
            float up_bias = up_biases[scale_row + group];

            for (uint i = 0; i < 8; i++) {
                uint shift = i * 4;
                float x = use_input_cache ? input_cache[col0 + i] : input[col0 + i];
                gate_acc += fma(float((gate_word >> shift) & 0x0f), gate_scale * x, gate_bias * x);
                up_acc += fma(float((up_word >> shift) & 0x0f), up_scale * x, up_bias * x);
            }
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar gate_byte = gate_packed[packed_row + packed_col];
            uchar up_byte = up_packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float gate_scale0 = gate_scales[scale_row + group0];
            float gate_bias0 = gate_biases[scale_row + group0];
            float up_scale0 = up_scales[scale_row + group0];
            float up_bias0 = up_biases[scale_row + group0];
            gate_acc += fma(float(gate_byte & 0x0f), gate_scale0 * x0, gate_bias0 * x0);
            up_acc += fma(float(up_byte & 0x0f), up_scale0 * x0, up_bias0 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float gate_scale1 = gate_scales[scale_row + group1];
                float gate_bias1 = gate_biases[scale_row + group1];
                float up_scale1 = up_scales[scale_row + group1];
                float up_bias1 = up_biases[scale_row + group1];
                gate_acc += fma(float(gate_byte >> 4), gate_scale1 * x1, gate_bias1 * x1);
                up_acc += fma(float(up_byte >> 4), up_scale1 * x1, up_bias1 * x1);
            }
        }
    }
    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0) {
        output[row] = (gate_sum / (1.0f + exp(-gate_sum))) * up_sum;
    }
}

kernel void q4_swiglu_fused_bf16_scale_bias(
    device const uchar* gate_packed [[buffer(0)]],
    device const uchar* up_packed [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device const ushort* gate_scales [[buffer(3)]],
    device const ushort* gate_biases [[buffer(4)]],
    device const ushort* up_scales [[buffer(5)]],
    device const ushort* up_biases [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& groups_per_row [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* gate_words = reinterpret_cast<device const uint*>(gate_packed);
        device const uint* up_words = reinterpret_cast<device const uint*>(up_packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint gate_word = gate_words[word_row + packed_word];
            uint up_word = up_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float gate_scale = bf16_to_float(gate_scales[scale_row + group]);
            float gate_bias = bf16_to_float(gate_biases[scale_row + group]);
            float up_scale = bf16_to_float(up_scales[scale_row + group]);
            float up_bias = bf16_to_float(up_biases[scale_row + group]);

            for (uint i = 0; i < 8; i++) {
                uint shift = i * 4;
                float x = use_input_cache ? input_cache[col0 + i] : input[col0 + i];
                gate_acc += fma(float((gate_word >> shift) & 0x0f), gate_scale * x, gate_bias * x);
                up_acc += fma(float((up_word >> shift) & 0x0f), up_scale * x, up_bias * x);
            }
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar gate_byte = gate_packed[packed_row + packed_col];
            uchar up_byte = up_packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float gate_scale0 = bf16_to_float(gate_scales[scale_row + group0]);
            float gate_bias0 = bf16_to_float(gate_biases[scale_row + group0]);
            float up_scale0 = bf16_to_float(up_scales[scale_row + group0]);
            float up_bias0 = bf16_to_float(up_biases[scale_row + group0]);
            gate_acc += fma(float(gate_byte & 0x0f), gate_scale0 * x0, gate_bias0 * x0);
            up_acc += fma(float(up_byte & 0x0f), up_scale0 * x0, up_bias0 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float gate_scale1 = bf16_to_float(gate_scales[scale_row + group1]);
                float gate_bias1 = bf16_to_float(gate_biases[scale_row + group1]);
                float up_scale1 = bf16_to_float(up_scales[scale_row + group1]);
                float up_bias1 = bf16_to_float(up_biases[scale_row + group1]);
                gate_acc += fma(float(gate_byte >> 4), gate_scale1 * x1, gate_bias1 * x1);
                up_acc += fma(float(up_byte >> 4), up_scale1 * x1, up_bias1 * x1);
            }
        }
    }
    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0) {
        output[row] = (gate_sum / (1.0f + exp(-gate_sum))) * up_sum;
    }
}

kernel void q4_mmap_fma_matvec(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& packed_byte_offset [[buffer(3)]],
    constant ulong& scales_byte_offset [[buffer(4)]],
    constant ulong& biases_byte_offset [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& groups_per_row [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const float* scales = reinterpret_cast<device const float*>(weight_bytes + scales_byte_offset);
    device const float* biases = reinterpret_cast<device const float*>(weight_bytes + biases_byte_offset);
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint row0 = tile * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[4096];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 >= rows) {
        return;
    }

    bool row1_valid = row1 < rows;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint packed_row0 = row0 * packed_stride;
    uint packed_row1 = row1 * packed_stride;
    uint scale_row0 = row0 * groups_per_row;
    uint scale_row1 = row1 * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0) && ((packed_byte_offset & 3ul) == 0ul);
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row0 = row0 * packed_words_per_row;
        uint word_row1 = row1 * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word0 = packed_words[word_row0 + packed_word];
            uint word1 = row1_valid ? packed_words[word_row1 + packed_word] : 0u;
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale0 = scales[scale_row0 + group];
            float bias0 = biases[scale_row0 + group];
            float scale1 = row1_valid ? scales[scale_row1 + group] : 0.0f;
            float bias1 = row1_valid ? biases[scale_row1 + group] : 0.0f;

            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];

            acc0 += fma(float((word0 >>  0) & 0x0f), scale0 * x0, bias0 * x0);
            acc0 += fma(float((word0 >>  4) & 0x0f), scale0 * x1, bias0 * x1);
            acc0 += fma(float((word0 >>  8) & 0x0f), scale0 * x2, bias0 * x2);
            acc0 += fma(float((word0 >> 12) & 0x0f), scale0 * x3, bias0 * x3);
            acc0 += fma(float((word0 >> 16) & 0x0f), scale0 * x4, bias0 * x4);
            acc0 += fma(float((word0 >> 20) & 0x0f), scale0 * x5, bias0 * x5);
            acc0 += fma(float((word0 >> 24) & 0x0f), scale0 * x6, bias0 * x6);
            acc0 += fma(float((word0 >> 28) & 0x0f), scale0 * x7, bias0 * x7);

            acc1 += fma(float((word1 >>  0) & 0x0f), scale1 * x0, bias1 * x0);
            acc1 += fma(float((word1 >>  4) & 0x0f), scale1 * x1, bias1 * x1);
            acc1 += fma(float((word1 >>  8) & 0x0f), scale1 * x2, bias1 * x2);
            acc1 += fma(float((word1 >> 12) & 0x0f), scale1 * x3, bias1 * x3);
            acc1 += fma(float((word1 >> 16) & 0x0f), scale1 * x4, bias1 * x4);
            acc1 += fma(float((word1 >> 20) & 0x0f), scale1 * x5, bias1 * x5);
            acc1 += fma(float((word1 >> 24) & 0x0f), scale1 * x6, bias1 * x6);
            acc1 += fma(float((word1 >> 28) & 0x0f), scale1 * x7, bias1 * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte0 = packed[packed_row0 + packed_col];
            uchar byte1 = row1_valid ? packed[packed_row1 + packed_col] : uchar(0);
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float scale00 = scales[scale_row0 + group0];
            float bias00 = biases[scale_row0 + group0];
            float scale10 = row1_valid ? scales[scale_row1 + group0] : 0.0f;
            float bias10 = row1_valid ? biases[scale_row1 + group0] : 0.0f;
            acc0 += fma(float(byte0 & 0x0f), scale00 * x0, bias00 * x0);
            acc1 += fma(float(byte1 & 0x0f), scale10 * x0, bias10 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float scale01 = scales[scale_row0 + group1];
                float bias01 = biases[scale_row0 + group1];
                float scale11 = row1_valid ? scales[scale_row1 + group1] : 0.0f;
                float bias11 = row1_valid ? biases[scale_row1 + group1] : 0.0f;
                acc0 += fma(float(byte0 >> 4), scale01 * x1, bias01 * x1);
                acc1 += fma(float(byte1 >> 4), scale11 * x1, bias11 * x1);
            }
        }
    }
    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        output[row0] = sum0;
        if (row1_valid) {
            output[row1] = sum1;
        }
    }
}

kernel void q4_mmap_fma_matvec_bf16_scale_bias(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& packed_byte_offset [[buffer(3)]],
    constant ulong& scales_byte_offset [[buffer(4)]],
    constant ulong& biases_byte_offset [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& groups_per_row [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const ushort* scales = reinterpret_cast<device const ushort*>(weight_bytes + scales_byte_offset);
    device const ushort* biases = reinterpret_cast<device const ushort*>(weight_bytes + biases_byte_offset);
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint row0 = tile * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[4096];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 >= rows) {
        return;
    }

    bool row1_valid = row1 < rows;
    bool word_aligned = (packed_byte_offset & 3ul) == 0ul;
    float sum0 = q4_fma_row_bf16_ptrs(
        packed, scales, biases, input, input_cache,
        row0, cols, groups_per_row, group_size,
        word_aligned, use_input_cache, simd_lane);
    float sum1 = row1_valid ? q4_fma_row_bf16_ptrs(
        packed, scales, biases, input, input_cache,
        row1, cols, groups_per_row, group_size,
        word_aligned, use_input_cache, simd_lane) : 0.0f;
    if (simd_lane == 0) {
        output[row0] = sum0;
        if (row1_valid) {
            output[row1] = sum1;
        }
    }
}

inline uint q4_batch_projection_for_row(
    uint row,
    device const uint* row_offsets,
    device const uint* rows,
    uint projection_count) {
    for (uint idx = 0; idx < projection_count; idx++) {
        uint start = row_offsets[idx];
        uint end = start + rows[idx];
        if (row >= start && row < end) {
            return idx;
        }
    }
    return projection_count;
}

inline float q4_mmap_fma_row_f32(
    device const uchar* weight_bytes,
    device const float* input,
    threadgroup float* input_cache,
    ulong packed_byte_offset,
    ulong scales_byte_offset,
    ulong biases_byte_offset,
    uint row,
    uint cols,
    uint groups_per_row,
    uint group_size,
    uint simd_lane) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const float* scales = reinterpret_cast<device const float*>(weight_bytes + scales_byte_offset);
    device const float* biases = reinterpret_cast<device const float*>(weight_bytes + biases_byte_offset);
    uint packed_stride = (cols + 1) / 2;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0) && ((packed_byte_offset & 3ul) == 0ul);
    float acc = 0.0f;
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = scales[scale_row + group];
            float bias = biases[scale_row + group];
            float x0 = input_cache[col0 + 0];
            float x1 = input_cache[col0 + 1];
            float x2 = input_cache[col0 + 2];
            float x3 = input_cache[col0 + 3];
            float x4 = input_cache[col0 + 4];
            float x5 = input_cache[col0 + 5];
            float x6 = input_cache[col0 + 6];
            float x7 = input_cache[col0 + 7];
            acc += fma(float((word >>  0) & 0x0f), scale * x0, bias * x0);
            acc += fma(float((word >>  4) & 0x0f), scale * x1, bias * x1);
            acc += fma(float((word >>  8) & 0x0f), scale * x2, bias * x2);
            acc += fma(float((word >> 12) & 0x0f), scale * x3, bias * x3);
            acc += fma(float((word >> 16) & 0x0f), scale * x4, bias * x4);
            acc += fma(float((word >> 20) & 0x0f), scale * x5, bias * x5);
            acc += fma(float((word >> 24) & 0x0f), scale * x6, bias * x6);
            acc += fma(float((word >> 28) & 0x0f), scale * x7, bias * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = input_cache[col0];
            uint group0 = col0 / group_size;
            float scale0 = scales[scale_row + group0];
            float bias0 = biases[scale_row + group0];
            acc += fma(float(byte & 0x0f), scale0 * x0, bias0 * x0);
            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = input_cache[col1];
                uint group1 = col1 / group_size;
                float scale1 = scales[scale_row + group1];
                float bias1 = biases[scale_row + group1];
                acc += fma(float(byte >> 4), scale1 * x1, bias1 * x1);
            }
        }
    }
    return simd_sum(acc);
}

inline float q4_mmap_fma_row_bf16(
    device const uchar* weight_bytes,
    device const float* input,
    threadgroup float* input_cache,
    ulong packed_byte_offset,
    ulong scales_byte_offset,
    ulong biases_byte_offset,
    uint row,
    uint cols,
    uint groups_per_row,
    uint group_size,
    uint simd_lane) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const ushort* scales = reinterpret_cast<device const ushort*>(weight_bytes + scales_byte_offset);
    device const ushort* biases = reinterpret_cast<device const ushort*>(weight_bytes + biases_byte_offset);
    bool use_input_cache = cols <= 4096;
    return q4_fma_row_bf16_ptrs(
        packed, scales, biases, input, input_cache,
        row, cols, groups_per_row, group_size,
        (packed_byte_offset & 3ul) == 0ul, use_input_cache, simd_lane);
}

kernel void q4_mmap_fma_multilinear_bf16_scale_bias(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* inputs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& packed_byte_offset [[buffer(3)]],
    constant ulong& scales_byte_offset [[buffer(4)]],
    constant ulong& biases_byte_offset [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& groups_per_row [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    constant uint& rows_per_head [[buffer(10)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint row0 = tile * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    uint head = (tile * rows_per_threadgroup) / rows_per_head;
    device const float* input = inputs + head * cols;
    threadgroup float input_cache[4096];
    for (uint col = lid; col < cols && col < input_cache_len; col += 256) {
        input_cache[col] = input[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 < rows) {
        float sum = q4_mmap_fma_row_bf16(
            weight_bytes, input, input_cache,
            packed_byte_offset, scales_byte_offset, biases_byte_offset,
            row0, cols, groups_per_row, group_size, simd_lane);
        if (simd_lane == 0) {
            output[row0] = sum;
        }
    }
    if (row1 < rows) {
        float sum = q4_mmap_fma_row_bf16(
            weight_bytes, input, input_cache,
            packed_byte_offset, scales_byte_offset, biases_byte_offset,
            row1, cols, groups_per_row, group_size, simd_lane);
        if (simd_lane == 0) {
            output[row1] = sum;
        }
    }
}

kernel void q4_mmap_fma_matvec_batch(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const ulong* packed_byte_offsets [[buffer(3)]],
    device const ulong* scales_byte_offsets [[buffer(4)]],
    device const ulong* biases_byte_offsets [[buffer(5)]],
    device const uint* row_offsets [[buffer(6)]],
    device const uint* rows [[buffer(7)]],
    device const uint* groups_per_rows [[buffer(8)]],
    constant uint& projection_count [[buffer(9)]],
    constant uint& cols [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint total_rows = row_offsets[projection_count - 1] + rows[projection_count - 1];
    uint input_row = tile.y;
    input += input_row * cols;
    output += input_row * total_rows;
    uint row0 = tile.x * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    threadgroup float input_cache[4096];
    for (uint col = lid.x; col < cols && col < input_cache_len; col += 256) {
        input_cache[col] = input[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint projection0 = q4_batch_projection_for_row(row0, row_offsets, rows, projection_count);
    if (projection0 < projection_count) {
        uint local_row = row0 - row_offsets[projection0];
        float sum = q4_mmap_fma_row_f32(
            weight_bytes, input, input_cache,
            packed_byte_offsets[projection0],
            scales_byte_offsets[projection0],
            biases_byte_offsets[projection0],
            local_row, cols, groups_per_rows[projection0], group_size, simd_lane);
        if (simd_lane == 0) {
            output[row0] = sum;
        }
    }
    uint projection1 = q4_batch_projection_for_row(row1, row_offsets, rows, projection_count);
    if (projection1 < projection_count) {
        uint local_row = row1 - row_offsets[projection1];
        float sum = q4_mmap_fma_row_f32(
            weight_bytes, input, input_cache,
            packed_byte_offsets[projection1],
            scales_byte_offsets[projection1],
            biases_byte_offsets[projection1],
            local_row, cols, groups_per_rows[projection1], group_size, simd_lane);
        if (simd_lane == 0) {
            output[row1] = sum;
        }
    }
}

kernel void q4_mmap_fma_matvec_batch_bf16_scale_bias(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const ulong* packed_byte_offsets [[buffer(3)]],
    device const ulong* scales_byte_offsets [[buffer(4)]],
    device const ulong* biases_byte_offsets [[buffer(5)]],
    device const uint* row_offsets [[buffer(6)]],
    device const uint* rows [[buffer(7)]],
    device const uint* groups_per_rows [[buffer(8)]],
    constant uint& projection_count [[buffer(9)]],
    constant uint& cols [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint total_rows = row_offsets[projection_count - 1] + rows[projection_count - 1];
    uint input_row = tile.y;
    input += input_row * cols;
    output += input_row * total_rows;
    uint row0 = tile.x * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    threadgroup float input_cache[4096];
    for (uint col = lid.x; col < cols && col < input_cache_len; col += 256) {
        input_cache[col] = input[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint projection0 = q4_batch_projection_for_row(row0, row_offsets, rows, projection_count);
    if (projection0 < projection_count) {
        uint local_row = row0 - row_offsets[projection0];
        float sum = q4_mmap_fma_row_bf16(
            weight_bytes, input, input_cache,
            packed_byte_offsets[projection0],
            scales_byte_offsets[projection0],
            biases_byte_offsets[projection0],
            local_row, cols, groups_per_rows[projection0], group_size, simd_lane);
        if (simd_lane == 0) {
            output[row0] = sum;
        }
    }
    uint projection1 = q4_batch_projection_for_row(row1, row_offsets, rows, projection_count);
    if (projection1 < projection_count) {
        uint local_row = row1 - row_offsets[projection1];
        float sum = q4_mmap_fma_row_bf16(
            weight_bytes, input, input_cache,
            packed_byte_offsets[projection1],
            scales_byte_offsets[projection1],
            biases_byte_offsets[projection1],
            local_row, cols, groups_per_rows[projection1], group_size, simd_lane);
        if (simd_lane == 0) {
            output[row1] = sum;
        }
    }
}

kernel void dense_mmap_fma_matvec_bf16(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) { return; }
    device const ushort* weights = reinterpret_cast<device const ushort*>(weight_bytes + weight_byte_offset);
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(bf16_to_float(weights[start + col]), input[col], acc);
    }
    output[row] = acc;
}

kernel void dense_mmap_fma_matvec_f16(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) { return; }
    device const half* weights = reinterpret_cast<device const half*>(weight_bytes + weight_byte_offset);
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(float(weights[start + col]), input[col], acc);
    }
    output[row] = acc;
}

kernel void dense_mmap_fma_matvec_f32(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) { return; }
    device const float* weights = reinterpret_cast<device const float*>(weight_bytes + weight_byte_offset);
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(weights[start + col], input[col], acc);
    }
    output[row] = acc;
}

kernel void dense_mmap_fma_matrix_bf16(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    uint row = position.x;
    uint input_row = position.y;
    if (row >= rows) { return; }
    device const ushort* weights = reinterpret_cast<device const ushort*>(weight_bytes + weight_byte_offset);
    input += input_row * cols;
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(bf16_to_float(weights[start + col]), input[col], acc);
    }
    output[input_row * rows + row] = acc;
}

kernel void dense_mmap_fma_matrix_f16(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    uint row = position.x;
    uint input_row = position.y;
    if (row >= rows) { return; }
    device const half* weights = reinterpret_cast<device const half*>(weight_bytes + weight_byte_offset);
    input += input_row * cols;
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(float(weights[start + col]), input[col], acc);
    }
    output[input_row * rows + row] = acc;
}

kernel void dense_mmap_fma_matrix_f32(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    uint row = position.x;
    uint input_row = position.y;
    if (row >= rows) { return; }
    device const float* weights = reinterpret_cast<device const float*>(weight_bytes + weight_byte_offset);
    input += input_row * cols;
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(weights[start + col], input[col], acc);
    }
    output[input_row * rows + row] = acc;
}

kernel void rms_norm_reduced(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint threads = 256;
    uint lid = thread_position.x;
    input += group.y * width;
    output += group.y * width;
    threadgroup float partial[32];
    float sum = 0.0f;
    for (uint i = lid; i < width; i += threads) {
        sum += input[i] * input[i];
    }
    float simd_value = simd_sum(sum);
    if (simd_lane == 0) {
        partial[simd_group] = simd_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float value = simd_lane < 8 ? partial[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            partial[0] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(partial[0] / float(max(width, 1u)) + epsilon);
    for (uint i = lid; i < width; i += threads) {
        output[i] = input[i] * scale * weight[i];
    }
}

kernel void residual_add_rms_norm(
    device const float* projected [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    device float* normed [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint threads = 256;
    uint lid = thread_position.x;
    projected += group.y * width;
    residual += group.y * width;
    hidden += group.y * width;
    normed += group.y * width;
    threadgroup float partial[32];
    float sum = 0.0f;
    for (uint i = lid; i < width; i += threads) {
        float value = projected[i] + residual[i];
        sum += value * value;
    }
    float simd_value = simd_sum(sum);
    if (simd_lane == 0) {
        partial[simd_group] = simd_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float value = simd_lane < 8 ? partial[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            partial[0] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(partial[0] / float(max(width, 1u)) + epsilon);
    for (uint i = lid; i < width; i += threads) {
        float value = projected[i] + residual[i];
        hidden[i] = value;
        normed[i] = value * scale * weight[i];
    }
}

kernel void attention_scores(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint token [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < width; ++i) {
        acc = fma(query[i], keys[token * width + i], acc);
    }
    scores[token] = acc * rsqrt(float(max(head_dim, 1u)));
}

kernel void qwen_causal_attention_rows(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& query_rows [[buffer(4)]],
    constant uint& prefix_rows [[buffer(5)]],
    constant uint& query_heads [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint thread_count = 256;
    threadgroup float partial_sums[8];
    threadgroup float shared_score;
    uint query_row = group.x;
    uint query_head = group.y;
    uint dimension = thread_position.x;
    if (query_row >= query_rows || query_head >= query_heads) {
        return;
    }
    uint groups_per_kv = max(query_heads / max(kv_heads, 1u), 1u);
    uint kv_head = min(query_head / groups_per_kv, max(kv_heads, 1u) - 1u);
    uint query_offset = (query_row * query_heads + query_head) * head_dim;
    uint key_count = prefix_rows + query_row + 1;
    float accumulator = 0.0f;
    float denominator = 0.0f;
    float maximum = -INFINITY;
    float query_value = dimension < head_dim ? queries[query_offset + dimension] : 0.0f;
    for (uint key_row = 0; key_row < key_count; ++key_row) {
        uint kv_offset = (key_row * kv_heads + kv_head) * head_dim;
        float product = dimension < head_dim ? query_value * keys[kv_offset + dimension] : 0.0f;
        float subgroup_sum = simd_sum(product);
        if (simd_lane == 0) {
            partial_sums[simd_group] = subgroup_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_position.x == 0) {
            float score = 0.0f;
            for (uint subgroup = 0; subgroup < thread_count / 32; ++subgroup) {
                score += partial_sums[subgroup];
            }
            shared_score = score * rsqrt(float(max(head_dim, 1u)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float next_maximum = max(maximum, shared_score);
        float previous_scale = isinf(maximum) ? 0.0f : exp(maximum - next_maximum);
        float current_scale = exp(shared_score - next_maximum);
        if (dimension < head_dim) {
            accumulator = accumulator * previous_scale
                + values[kv_offset + dimension] * current_scale;
        }
        denominator = denominator * previous_scale + current_scale;
        maximum = next_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (dimension < head_dim) {
        output[query_offset + dimension] = accumulator / max(denominator, 1.0e-20f);
    }
}

kernel void glm_mla_prepare_query_kv(
    device const float* query [[buffer(0)]],
    device const float* compressed [[buffer(1)]],
    device float* query_nope [[buffer(2)]],
    device float* query_rope [[buffer(3)]],
    device float* record_latents [[buffer(4)]],
    device float* record_rotary [[buffer(5)]],
    device const float* rope_cos [[buffer(6)]],
    device const float* rope_sin [[buffer(7)]],
    constant uint& heads [[buffer(8)]],
    constant uint& nope_dim [[buffer(9)]],
    constant uint& rope_dim [[buffer(10)]],
    constant uint& latent_rank [[buffer(11)]],
    constant uint& sequence [[buffer(12)]],
    uint index [[thread_position_in_grid]]) {
    uint nope_values = heads * nope_dim;
    uint query_rope_values = heads * rope_dim;
    uint rope_half = rope_dim / 2;
    if (index < nope_values) {
        uint head = index / nope_dim;
        uint dim = index - head * nope_dim;
        query_nope[index] = query[head * (nope_dim + rope_dim) + dim];
        return;
    }
    uint section = index - nope_values;
    if (section < query_rope_values) {
        uint head = section / rope_dim;
        uint dim = section - head * rope_dim;
        uint pair = dim < rope_half ? dim : dim - rope_half;
        uint input_offset = head * (nope_dim + rope_dim) + nope_dim + pair * 2;
        float a = query[input_offset];
        float b = query[input_offset + 1];
        query_rope[section] = dim < rope_half
            ? a * rope_cos[pair] - b * rope_sin[pair]
            : b * rope_cos[pair] + a * rope_sin[pair];
        return;
    }
    section -= query_rope_values;
    if (section < latent_rank) {
        record_latents[(sequence - 1) * latent_rank + section] = compressed[section];
        return;
    }
    section -= latent_rank;
    if (section < rope_dim) {
        uint pair = section < rope_half ? section : section - rope_half;
        float a = compressed[latent_rank + pair * 2];
        float b = compressed[latent_rank + pair * 2 + 1];
        record_rotary[(sequence - 1) * rope_dim + section] = section < rope_half
            ? a * rope_cos[pair] - b * rope_sin[pair]
            : b * rope_cos[pair] + a * rope_sin[pair];
    }
}

kernel void glm_mla_absorbed_scores(
    device const float* absorbed_queries [[buffer(0)]],
    device const float* query_rope [[buffer(1)]],
    device const float* record_latents [[buffer(2)]],
    device const float* record_rotary [[buffer(3)]],
    device float* scores [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& latent_rank [[buffer(6)]],
    constant uint& rope_dim [[buffer(7)]],
    constant uint& sequence [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    uint total = heads * sequence;
    if (index >= total) { return; }
    uint head = index / sequence;
    uint position = index - head * sequence;
    float score = 0.0f;
    uint query_latent_offset = head * latent_rank;
    uint record_latent_offset = position * latent_rank;
    for (uint dim = 0; dim < latent_rank; ++dim) {
        score = fma(
            absorbed_queries[query_latent_offset + dim],
            record_latents[record_latent_offset + dim],
            score);
    }
    uint query_rope_offset = head * rope_dim;
    uint record_rope_offset = position * rope_dim;
    for (uint dim = 0; dim < rope_dim; ++dim) {
        score = fma(
            query_rope[query_rope_offset + dim],
            record_rotary[record_rope_offset + dim],
            score);
    }
    scores[index] = score * scale;
}

kernel void glm_mla_softmax(
    device float* scores [[buffer(0)]],
    constant uint& heads [[buffer(1)]],
    constant uint& sequence [[buffer(2)]],
    uint head [[thread_position_in_grid]]) {
    if (head >= heads) { return; }
    uint offset = head * sequence;
    float maximum = scores[offset];
    for (uint position = 1; position < sequence; ++position) {
        maximum = max(maximum, scores[offset + position]);
    }
    float denominator = 0.0f;
    for (uint position = 0; position < sequence; ++position) {
        float value = exp(scores[offset + position] - maximum);
        scores[offset + position] = value;
        denominator += value;
    }
    float inverse = 1.0f / max(denominator, 1.0e-20f);
    for (uint position = 0; position < sequence; ++position) {
        scores[offset + position] *= inverse;
    }
}

kernel void glm_mla_context(
    device const float* scores [[buffer(0)]],
    device const float* record_latents [[buffer(1)]],
    device float* contexts [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& latent_rank [[buffer(4)]],
    constant uint& sequence [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    uint total = heads * latent_rank;
    if (index >= total) { return; }
    uint head = index / latent_rank;
    uint dim = index - head * latent_rank;
    float value = 0.0f;
    for (uint position = 0; position < sequence; ++position) {
        value = fma(
            scores[head * sequence + position],
            record_latents[position * latent_rank + dim],
            value);
    }
    contexts[index] = value;
}

kernel void expert_mlp_fused(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* down [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& intermediate [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < intermediate; ++i) {
        float g = gate[i] / (1.0f + exp(-gate[i]));
        acc = fma(down[row * intermediate + i], g * up[i], acc);
    }
    output[row] = acc * rsqrt(float(max(intermediate, 1u)));
}

kernel void silu_product(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    float g = gate[idx];
    output[idx] = (g / (1.0f + exp(-g))) * up[idx];
}

kernel void shared_expert_activation(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* router [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& intermediate [[buffer(4)]],
    constant uint& total_intermediate [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= total_intermediate) { return; }
    float g = gate[idx];
    output[idx] = (g / (1.0f + exp(-g))) * up[idx];
}

kernel void combine_expert_phase(
    device const float* residual [[buffer(0)]],
    device const float* shared [[buffer(1)]],
    device const float* expert_outputs [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device float* hidden [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& active_experts [[buffer(6)]],
    device const float* shared_router [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    float route = shared_router[0];
    float shared_weight = 1.0f / (1.0f + exp(-route));
    float moe = 0.0f;
    for (uint expert = 0; expert < active_experts; ++expert) {
        moe += weights[expert] * expert_outputs[expert * width + idx];
    }
    hidden[idx] = residual[idx] + moe + shared_weight * shared[idx];
}

kernel void qwen_layer_major_gather(
    device const float* input [[buffer(0)]],
    device const uint* source_rows [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& values [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= values) { return; }
    uint output_row = idx / width;
    uint col = idx - output_row * width;
    output[idx] = input[source_rows[output_row] * width + col];
}

kernel void qwen_layer_major_combine(
    device const float* residual [[buffer(0)]],
    device const float* shared [[buffer(1)]],
    device const float* grouped_expert_outputs [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const uint* grouped_output_indices [[buffer(4)]],
    device const float* shared_router [[buffer(5)]],
    device float* hidden [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    constant uint& active_experts [[buffer(9)]],
    constant uint& shared_router_width [[buffer(10)]],
    uint idx [[thread_position_in_grid]]) {
    uint total = rows * width;
    if (idx >= total) { return; }
    uint row = idx / width;
    uint col = idx - row * width;
    uint route_base = row * active_experts;
    float moe = 0.0f;
    for (uint active = 0; active < active_experts; ++active) {
        uint route = route_base + active;
        uint grouped = grouped_output_indices[route];
        moe += weights[route] * grouped_expert_outputs[grouped * width + col];
    }
    float shared_weight = 0.0f;
    if (shared_router_width > 0) {
        float route = shared_router[row * shared_router_width];
        shared_weight = 1.0f / (1.0f + exp(-route));
    }
    hidden[idx] = residual[idx] + moe + shared_weight * shared[idx];
}

kernel void fill_zero(
    device float* output [[buffer(0)]],
    constant uint& width [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    output[idx] = 0.0f;
}

kernel void topk_vocab(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* values [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    uint slot [[thread_position_in_grid]]) {
    if (slot != 0) { return; }
    uint limit = min(top_k, vocab);
    for (uint out = 0; out < limit; ++out) {
        float best = -INFINITY;
        uint best_i = 0;
        bool found = false;
        for (uint i = 0; i < vocab; ++i) {
            float raw_value = logits[i];
            float value = isfinite(raw_value) ? raw_value : -INFINITY;
            bool already_used = false;
            for (uint prev = 0; prev < out; ++prev) {
                already_used = already_used || (indices[prev] == i);
            }
            if (!already_used && (!found || value > best)) {
                best = value;
                best_i = i;
                found = true;
            }
        }
        indices[out] = best_i;
        values[out] = best;
    }
}

kernel void linear_conv1d_step_bf16(
    device float* conv_state [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const ushort* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& conv_dim [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= conv_dim || kernel_size == 0) {
        return;
    }
    float acc = 0.0f;
    uint w_base = idx * kernel_size;
    for (uint k = 0; k + 1 < kernel_size; ++k) {
        acc = fma(conv_state[k * conv_dim + idx], bf16_to_float(weights[w_base + k]), acc);
    }
    float inp = input[idx];
    acc = fma(inp, bf16_to_float(weights[w_base + kernel_size - 1]), acc);
    output[idx] = acc / (1.0f + exp(-acc));
    for (uint k = 0; k + 2 < kernel_size; ++k) {
        conv_state[k * conv_dim + idx] = conv_state[(k + 1) * conv_dim + idx];
    }
    if (kernel_size > 1) {
        conv_state[(kernel_size - 2) * conv_dim + idx] = inp;
    }
}

kernel void linear_conv1d_step_f16(
    device float* conv_state [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const half* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& conv_dim [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= conv_dim || kernel_size == 0) {
        return;
    }
    float acc = 0.0f;
    uint w_base = idx * kernel_size;
    for (uint k = 0; k + 1 < kernel_size; ++k) {
        acc = fma(conv_state[k * conv_dim + idx], float(weights[w_base + k]), acc);
    }
    float inp = input[idx];
    acc = fma(inp, float(weights[w_base + kernel_size - 1]), acc);
    output[idx] = acc / (1.0f + exp(-acc));
    for (uint k = 0; k + 2 < kernel_size; ++k) {
        conv_state[k * conv_dim + idx] = conv_state[(k + 1) * conv_dim + idx];
    }
    if (kernel_size > 1) {
        conv_state[(kernel_size - 2) * conv_dim + idx] = inp;
    }
}

kernel void linear_conv1d_step_f32(
    device float* conv_state [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& conv_dim [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= conv_dim || kernel_size == 0) {
        return;
    }
    float acc = 0.0f;
    uint w_base = idx * kernel_size;
    for (uint k = 0; k + 1 < kernel_size; ++k) {
        acc = fma(conv_state[k * conv_dim + idx], weights[w_base + k], acc);
    }
    float inp = input[idx];
    acc = fma(inp, weights[w_base + kernel_size - 1], acc);
    output[idx] = acc / (1.0f + exp(-acc));
    for (uint k = 0; k + 2 < kernel_size; ++k) {
        conv_state[k * conv_dim + idx] = conv_state[(k + 1) * conv_dim + idx];
    }
    if (kernel_size > 1) {
        conv_state[(kernel_size - 2) * conv_dim + idx] = inp;
    }
}

kernel void linear_rms_norm_qk(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant uint& key_dim [[buffer(2)]],
    constant float& inv_scale [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    uint base = head * key_dim;
    threadgroup float partial[256];
    float qval = (tid < key_dim) ? q[base + tid] : 0.0f;
    partial[tid] = qval * qval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < key_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(key_dim, 1u)) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < key_dim) {
        q[base + tid] = qval * partial[0] * inv_scale * inv_scale;
    }

    float kval = (tid < key_dim) ? k[base + tid] : 0.0f;
    partial[tid] = kval * kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < key_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(key_dim, 1u)) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < key_dim) {
        k[base + tid] = kval * partial[0] * inv_scale;
    }
}

kernel void linear_compute_decay_beta_bf16(
    device const float* alpha [[buffer(0)]],
    device const float* beta [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const ushort* dt_bias [[buffer(3)]],
    device float* g_decay [[buffer(4)]],
    device float* beta_gate [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= heads) {
        return;
    }
    float softplus_value = log(1.0f + exp(alpha[idx] + bf16_to_float(dt_bias[idx])));
    g_decay[idx] = exp(-exp(a_log[idx]) * softplus_value);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta[idx]));
}

kernel void linear_compute_decay_beta_f16(
    device const float* alpha [[buffer(0)]],
    device const float* beta [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const half* dt_bias [[buffer(3)]],
    device float* g_decay [[buffer(4)]],
    device float* beta_gate [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= heads) {
        return;
    }
    float softplus_value = log(1.0f + exp(alpha[idx] + float(dt_bias[idx])));
    g_decay[idx] = exp(-exp(a_log[idx]) * softplus_value);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta[idx]));
}

kernel void linear_compute_decay_beta_f32(
    device const float* alpha [[buffer(0)]],
    device const float* beta [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device float* g_decay [[buffer(4)]],
    device float* beta_gate [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= heads) {
        return;
    }
    float softplus_value = log(1.0f + exp(alpha[idx] + dt_bias[idx]));
    g_decay[idx] = exp(-exp(a_log[idx]) * softplus_value);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta[idx]));
}

kernel void linear_gated_delta_step(
    device float* state [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device const float* g_decay [[buffer(4)]],
    device const float* beta_gate [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& key_dim [[buffer(7)]],
    constant uint& value_dim [[buffer(8)]],
    constant uint& k_heads_per_v [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]) {
    if (vi >= value_dim || key_dim > 256) {
        return;
    }
    uint key_head = head / max(k_heads_per_v, 1u);
    uint state_base = head * value_dim * key_dim + vi * key_dim;
    uint key_base = key_head * key_dim;
    uint value_base = head * value_dim;
    float decay = g_decay[head];
    float beta = beta_gate[head];
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < key_dim; ++ki) {
        float s = state[state_base + ki] * decay;
        state[state_base + ki] = s;
        kv_mem = fma(s, k[key_base + ki], kv_mem);
    }
    float delta = (v[value_base + vi] - kv_mem) * beta;
    for (uint ki = 0; ki < key_dim; ++ki) {
        state[state_base + ki] = fma(k[key_base + ki], delta, state[state_base + ki]);
    }
    float out_value = 0.0f;
    for (uint ki = 0; ki < key_dim; ++ki) {
        out_value = fma(state[state_base + ki], q[key_base + ki], out_value);
    }
    output[value_base + vi] = out_value;
}

kernel void linear_gated_rms_norm_bf16(
    device const float* values [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const ushort* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& value_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    uint base = head * value_dim;
    threadgroup float partial[256];
    float value = (tid < value_dim) ? values[base + tid] : 0.0f;
    partial[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < value_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(value_dim, 1u)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < value_dim) {
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));
        output[base + tid] = value * partial[0] * gate * bf16_to_float(weight[tid]);
    }
}

kernel void linear_gated_rms_norm_f16(
    device const float* values [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& value_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    uint base = head * value_dim;
    threadgroup float partial[256];
    float value = (tid < value_dim) ? values[base + tid] : 0.0f;
    partial[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < value_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(value_dim, 1u)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < value_dim) {
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));
        output[base + tid] = value * partial[0] * gate * float(weight[tid]);
    }
}

kernel void linear_gated_rms_norm_f32(
    device const float* values [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& value_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    uint base = head * value_dim;
    threadgroup float partial[256];
    float value = (tid < value_dim) ? values[base + tid] : 0.0f;
    partial[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < value_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(value_dim, 1u)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < value_dim) {
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));
        output[base + tid] = value * partial[0] * gate * weight[tid];
    }
}
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalCommandContext {
    operation: String,
    details: Vec<(String, String)>,
}

impl MetalCommandContext {
    pub(crate) fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            details: Vec::new(),
        }
    }

    pub(crate) fn with(mut self, key: impl Into<String>, value: impl ToString) -> Self {
        self.details.push((key.into(), value.to_string()));
        self
    }

    pub(crate) fn label(&self) -> String {
        let mut label = format!("Flash-MoE {}", self.operation);
        for (key, value) in &self.details {
            label.push(' ');
            label.push_str(key);
            label.push('=');
            label.push_str(value);
        }
        label
    }

    pub(crate) fn detail_summary(&self) -> String {
        if self.details.is_empty() {
            "none".to_string()
        } else {
            self.details
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalCommandStatus {
    NotEnqueued,
    Enqueued,
    Committed,
    Scheduled,
    Completed,
    Error,
    Unknown(usize),
}

impl MetalCommandStatus {
    pub(crate) fn from_raw(raw: usize) -> Self {
        match raw {
            0 => Self::NotEnqueued,
            1 => Self::Enqueued,
            2 => Self::Committed,
            3 => Self::Scheduled,
            4 => Self::Completed,
            5 => Self::Error,
            value => Self::Unknown(value),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::NotEnqueued => "not_enqueued",
            Self::Enqueued => "enqueued",
            Self::Committed => "committed",
            Self::Scheduled => "scheduled",
            Self::Completed => "completed",
            Self::Error => "error",
            Self::Unknown(_) => "unknown",
        }
    }

    pub(crate) fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Error)
    }
}

impl std::fmt::Display for MetalCommandStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(raw) => write!(f, "unknown({raw})"),
            status => f.write_str(status.as_str()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalCommandFailureKind {
    Timeout,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalCommandBufferFailure {
    kind: MetalCommandFailureKind,
    message: String,
}

impl MetalCommandBufferFailure {
    pub(crate) fn timeout(
        context: &MetalCommandContext,
        elapsed: Duration,
        status: MetalCommandStatus,
        metal_error: Option<String>,
    ) -> Self {
        Self {
            kind: MetalCommandFailureKind::Timeout,
            message: format_metal_command_failure(
                MetalCommandFailureKind::Timeout,
                context,
                elapsed,
                status,
                metal_error.as_deref(),
            ),
        }
    }

    pub(crate) fn failed(
        context: &MetalCommandContext,
        elapsed: Duration,
        status: MetalCommandStatus,
        metal_error: Option<String>,
    ) -> Self {
        Self {
            kind: MetalCommandFailureKind::Failed,
            message: format_metal_command_failure(
                MetalCommandFailureKind::Failed,
                context,
                elapsed,
                status,
                metal_error.as_deref(),
            ),
        }
    }

    pub(crate) fn should_release_buffers(&self) -> bool {
        true
    }
}

impl std::fmt::Display for MetalCommandBufferFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MetalCommandBufferFailure {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCommandWaitPolicy {
    pub(crate) timeout: Duration,
    pub(crate) poll_interval: Duration,
}

impl Default for MetalCommandWaitPolicy {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_FLASHMOE_METAL_COMMAND_TIMEOUT,
            poll_interval: DEFAULT_FLASHMOE_METAL_COMMAND_POLL_INTERVAL,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MetalCommandWaitResult {
    Pending,
    Finished(std::result::Result<(), MetalCommandBufferFailure>),
}

pub(crate) fn resolve_metal_command_wait(
    context: &MetalCommandContext,
    elapsed: Duration,
    status: MetalCommandStatus,
    metal_error: Option<String>,
    timed_out: bool,
) -> MetalCommandWaitResult {
    if status.is_terminal() {
        return match status {
            MetalCommandStatus::Completed if metal_error.is_none() => {
                MetalCommandWaitResult::Finished(Ok(()))
            }
            _ => MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure::failed(
                context,
                elapsed,
                status,
                metal_error,
            ))),
        };
    }
    if timed_out {
        return MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure::timeout(
            context,
            elapsed,
            status,
            metal_error,
        )));
    }
    MetalCommandWaitResult::Pending
}

pub(crate) fn format_metal_command_failure(
    kind: MetalCommandFailureKind,
    context: &MetalCommandContext,
    elapsed: Duration,
    status: MetalCommandStatus,
    metal_error: Option<&str>,
) -> String {
    let action = match kind {
        MetalCommandFailureKind::Timeout => "timed out",
        MetalCommandFailureKind::Failed => "failed",
    };
    let error = metal_error
        .filter(|error| !error.trim().is_empty())
        .unwrap_or("none reported");
    format!(
        "Flash-MoE Metal command buffer {action}: label=\"{}\", elapsed={}ms, status={}, metal_error=\"{}\", details={}",
        context.label(),
        elapsed.as_millis(),
        status,
        error,
        context.detail_summary()
    )
}

#[cfg(test)]
mod tests {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use super::super::capabilities::{
        FlashMoeGraphStage, FlashMoeStageCapability, FlashMoeStageImplementation,
        FlashMoeStagePlacement,
    };
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use super::super::experts::Q4MatvecPayload;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use super::super::scheduler::ScheduledRoutingTopK;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use super::super::weights::DenseQ4MmapMatvecProjection;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use super::super::weights::SharedExpertPhaseShape;
    use super::*;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_working_set_limit_preserves_documented_headroom() {
        let gib = 1024 * 1024 * 1024;
        assert_eq!(default_metal_working_set_limit(10 * gib), 9 * gib);
        assert_eq!(default_metal_working_set_limit(4 * gib), 3 * gib);
        assert_eq!(default_metal_working_set_limit(gib), gib / 2);
        assert_eq!(default_metal_working_set_limit(0), usize::MAX);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_resource_ledger_balances_buffer_ownership_transitions() {
        let gib = 1024 * 1024 * 1024;
        let ledger = MetalResourceLedger {
            state: Mutex::new(MetalResourceLedgerState::new(10 * gib, 2 * gib)),
        };
        ledger.record_resident_resources(100, 200);
        ledger.register_buffer(
            0x1000usize as MetalObjcId,
            64,
            MetalTrackedBufferClass::ActiveGeneral,
        );
        ledger.register_buffer(
            0x2000usize as MetalObjcId,
            128,
            MetalTrackedBufferClass::TransientExpert,
        );
        ledger.register_buffer(
            0x3000usize as MetalObjcId,
            256,
            MetalTrackedBufferClass::ResidentExpertWrapper,
        );

        let active = ledger.snapshot();
        assert_eq!(active.active_general_buffers, 1);
        assert_eq!(active.transient_expert_buffers, 1);
        assert_eq!(active.resident_expert_wrapper_buffers, 1);
        assert_eq!(active.resident_expert_wrapper_bytes, 256);
        assert_eq!(active.ledger_live_bytes, 748);
        assert_eq!(active.driver_high_water_bytes, 2 * gib);

        ledger.transition_buffer(0x1000usize as MetalObjcId, MetalTrackedBufferClass::Pooled);
        ledger.release_buffer(0x2000usize as MetalObjcId);
        let pooled = ledger.snapshot();
        assert_eq!(pooled.active_general_buffers, 0);
        assert_eq!(pooled.transient_expert_buffers, 0);
        assert_eq!(pooled.pooled_buffers, 1);
        assert_eq!(pooled.ledger_live_bytes, 620);

        ledger.release_buffer(0x1000usize as MetalObjcId);
        ledger.release_buffer(0x3000usize as MetalObjcId);
        ledger.record_resident_resources(0, 0);
        let released = ledger.snapshot();
        assert_eq!(released.ledger_live_bytes, 0);
        assert_eq!(released.pooled_buffers, 0);
        assert_eq!(released.resident_expert_wrapper_buffers, 0);
        assert_eq!(released.ledger_high_water_bytes, 748);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_resource_ledger_balances_injected_error_cleanup_and_commands() {
        let resources = Arc::new(MetalResourceLedger::default());
        resources.register_buffer(
            0x1000usize as MetalObjcId,
            16,
            MetalTrackedBufferClass::ActiveGeneral,
        );
        resources.register_buffer(
            0x2000usize as MetalObjcId,
            32,
            MetalTrackedBufferClass::ActiveGeneral,
        );
        let first = MetalCommandLease::new(Arc::clone(&resources));
        let second = MetalCommandLease::new(Arc::clone(&resources));
        assert_eq!(resources.snapshot().in_flight_commands, 2);
        assert_eq!(resources.snapshot().command_high_water, 2);
        assert_eq!(resources.snapshot().command_submissions, 2);
        resources.record_host_upload(4_096);
        resources.record_host_readback(2_048);

        resources.release_buffer(0x1000usize as MetalObjcId);
        resources.release_buffer(0x2000usize as MetalObjcId);
        drop(first);
        drop(second);
        let cleaned = resources.snapshot();
        assert_eq!(cleaned.active_general_buffers, 0);
        assert_eq!(cleaned.in_flight_commands, 0);
        assert_eq!(cleaned.command_submissions, 2);
        assert_eq!(cleaned.host_upload_bytes, 4_096);
        assert_eq!(cleaned.host_readback_bytes, 2_048);
        assert_eq!(cleaned.ledger_live_bytes, 0);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn explicit_metal_limit_can_only_lower_default_policy() {
        let gib = 1024 * 1024 * 1024;
        let ledger = MetalResourceLedger {
            state: Mutex::new(MetalResourceLedgerState::new(10 * gib, 2 * gib)),
        };
        ledger.set_working_set_limit_bytes(6 * gib).unwrap();
        assert_eq!(ledger.snapshot().working_set_limit_bytes, 6 * gib);
        assert!(ledger.allocation_would_exceed_limit(6 * gib - 1, 2));

        ledger.set_working_set_limit_bytes(20 * gib).unwrap();
        assert_eq!(ledger.snapshot().working_set_limit_bytes, 9 * gib);
        assert!(ledger.set_working_set_limit_bytes(0).is_err());
    }

    #[test]
    fn command_context_label_includes_actionable_details() {
        let context = MetalCommandContext::new("deferred_expert_phase")
            .with("position", 17)
            .with("layer", 3)
            .with("experts", "1,7,9,11")
            .with("width", 4096);

        assert_eq!(
            context.label(),
            "Flash-MoE deferred_expert_phase position=17 layer=3 experts=1,7,9,11 width=4096"
        );
        assert_eq!(
            context.detail_summary(),
            "position=17, layer=3, experts=1,7,9,11, width=4096"
        );
    }

    #[test]
    fn command_status_names_known_and_unknown_values() {
        assert_eq!(MetalCommandStatus::from_raw(0).to_string(), "not_enqueued");
        assert_eq!(MetalCommandStatus::from_raw(3).to_string(), "scheduled");
        assert_eq!(MetalCommandStatus::from_raw(4).to_string(), "completed");
        assert_eq!(MetalCommandStatus::from_raw(5).to_string(), "error");
        assert_eq!(MetalCommandStatus::from_raw(99).to_string(), "unknown(99)");
        assert!(MetalCommandStatus::Completed.is_terminal());
        assert!(MetalCommandStatus::Error.is_terminal());
        assert!(!MetalCommandStatus::Scheduled.is_terminal());
    }

    #[test]
    fn command_failure_diagnostic_is_actionable() {
        let context = MetalCommandContext::new("gqa_attention_scores")
            .with("layer", 12)
            .with("position", 128)
            .with("tokens", 129)
            .with("q_heads", 32)
            .with("kv_heads", 8);

        let message = format_metal_command_failure(
            MetalCommandFailureKind::Timeout,
            &context,
            Duration::from_millis(1234),
            MetalCommandStatus::Scheduled,
            Some("GPU timeout"),
        );

        assert!(message.contains("timed out"));
        assert!(message.contains("label=\"Flash-MoE gqa_attention_scores"));
        assert!(message.contains("elapsed=1234ms"));
        assert!(message.contains("status=scheduled"));
        assert!(message.contains("metal_error=\"GPU timeout\""));
        assert!(message.contains("layer=12"));
        assert!(message.contains("position=128"));
        assert!(message.contains("tokens=129"));
    }

    #[test]
    fn command_failure_marks_buffers_for_release() {
        let context = MetalCommandContext::new("lm_head_topk").with("rows", 42);
        let error = MetalCommandBufferFailure::failed(
            &context,
            Duration::from_millis(7),
            MetalCommandStatus::Error,
            None,
        );
        assert!(error.should_release_buffers());
        assert!(error.to_string().contains("none reported"));
    }

    #[test]
    fn command_wait_policy_uses_upstream_shaped_timeout_defaults() {
        let policy = MetalCommandWaitPolicy::default();
        assert_eq!(policy.timeout, Duration::from_secs(120));
        assert_eq!(policy.poll_interval, Duration::from_micros(100));
    }

    #[test]
    fn command_wait_resolution_handles_completed_failed_timeout_and_pending() {
        let context = MetalCommandContext::new("cmd3");

        assert_eq!(
            resolve_metal_command_wait(
                &context,
                Duration::from_millis(4),
                MetalCommandStatus::Completed,
                None,
                false,
            ),
            MetalCommandWaitResult::Finished(Ok(()))
        );

        let failed = resolve_metal_command_wait(
            &context,
            Duration::from_millis(5),
            MetalCommandStatus::Error,
            Some("encoder failed".to_string()),
            false,
        );
        assert!(matches!(
            failed,
            MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure {
                kind: MetalCommandFailureKind::Failed,
                ..
            }))
        ));

        let timed_out = resolve_metal_command_wait(
            &context,
            Duration::from_secs(120),
            MetalCommandStatus::Scheduled,
            Some("still running".to_string()),
            true,
        );
        assert!(matches!(
            timed_out,
            MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure {
                kind: MetalCommandFailureKind::Timeout,
                ..
            }))
        ));

        assert_eq!(
            resolve_metal_command_wait(
                &context,
                Duration::from_millis(1),
                MetalCommandStatus::Scheduled,
                None,
                false,
            ),
            MetalCommandWaitResult::Pending
        );
    }

    #[test]
    fn shader_source_defines_full_forward_kernel_set() {
        for kernel in REQUIRED_FORWARD_KERNELS {
            assert!(
                METAL_SHADERS.contains(&format!("kernel void {kernel}")),
                "missing Metal kernel {kernel}"
            );
        }
        assert!(METAL_SHADERS.contains("threadgroup float input_cache"));
        assert!(METAL_SHADERS.contains("simd_sum(acc)"));
        assert!(METAL_SHADERS.contains("thread_index_in_simdgroup"));
        assert!(METAL_SHADERS.contains("constant uint& group_size"));
        assert!(METAL_SHADERS.contains("constant uint& top_k"));
        assert!(METAL_SHADERS.contains("fma(float(byte & 0x0f), scale0 * x0, bias0 * x0)"));
        assert!(
            !METAL_SHADERS.contains("uint half"),
            "`half` is a Metal scalar type and cannot be reused as a variable name"
        );
    }

    #[test]
    fn pipeline_name_set_matches_declared_forward_kernel_surface() {
        let mut compiled = MetalPipelineNameSet::new().kernel_names();
        compiled.sort_unstable();
        compiled.dedup();

        let mut required = REQUIRED_FORWARD_KERNELS.to_vec();
        required.sort_unstable();
        required.dedup();

        assert_eq!(compiled, required);
    }

    #[test]
    fn deepseek_shader_source_defines_load_resolved_kernel_surface() {
        for kernel in DEEPSEEK_V4_REQUIRED_METAL_KERNELS {
            assert!(
                super::super::deepseek_metal::DEEPSEEK_V4_METAL_SHADERS.contains(kernel),
                "missing DeepSeek V4 Metal kernel {kernel}"
            );
        }
        assert!(
            super::super::deepseek_metal::DEEPSEEK_V4_METAL_SHADERS
                .contains("kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32")
        );
        assert!(
            super::super::deepseek_metal::DEEPSEEK_V4_METAL_SHADERS
                .contains("kernel_mul_mv_slots6_q2_K_sum6_f32")
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn deepseek_shader_library_compiles_every_required_pipeline() {
        objc2::rc::autoreleasepool(|_| unsafe {
            let device = OwnedMetalObject::new(metal_default_device()).unwrap();
            let pipelines = DeepSeekMetalPipelineSet::compile(device.id()).unwrap();
            for &kernel in DEEPSEEK_V4_REQUIRED_METAL_KERNELS {
                pipelines.require(kernel).unwrap();
            }
        });
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn native_mxfp4_matvec_matches_e2m1_e8m0_reference() {
        objc2::rc::autoreleasepool(|_| unsafe {
            let runtime =
                MetalRuntime::compile(METAL_SHADERS, MetalPipelineNameSet::new()).unwrap();
            let packed = [vec![0x22u8; 16], vec![0x95u8; 16]].concat();
            let input = (1..=32).map(|value| value as f32).collect::<Vec<_>>();
            let scales = [127u8, 127u8];
            let packed_buffer = OwnedMetalObject::new(msg_send_id3_ptr_usize_u64(
                runtime.device,
                sel("newBufferWithBytes:length:options:"),
                packed.as_ptr().cast(),
                packed.len(),
                0,
            ))
            .unwrap();
            let input_bytes = f32_as_bytes(&input);
            let input_buffer = OwnedMetalObject::new(msg_send_id3_ptr_usize_u64(
                runtime.device,
                sel("newBufferWithBytes:length:options:"),
                input_bytes.as_ptr().cast(),
                input_bytes.len(),
                0,
            ))
            .unwrap();
            let scale_buffer = OwnedMetalObject::new(msg_send_id3_ptr_usize_u64(
                runtime.device,
                sel("newBufferWithBytes:length:options:"),
                scales.as_ptr().cast(),
                scales.len(),
                0,
            ))
            .unwrap();
            let output_buffer = OwnedMetalObject::new(msg_send_id2_usize_u64(
                runtime.device,
                sel("newBufferWithLength:options:"),
                2 * std::mem::size_of::<f32>(),
                0,
            ))
            .unwrap();
            let mut encoding = MetalCommandEncoding::new(
                runtime.command_queue,
                Arc::new(MetalResourceLedger::default()),
                "failed to create MXFP4 test command buffer",
                "failed to create MXFP4 test encoder",
            )
            .unwrap();
            let encoder = encoding.encoder();
            msg_send_void1_id(
                encoder,
                sel("setComputePipelineState:"),
                runtime.pipelines.mxfp4_e8m0_pipeline,
            );
            set_buffer(encoder, packed_buffer.id(), 0);
            set_buffer(encoder, input_buffer.id(), 1);
            set_buffer(encoder, scale_buffer.id(), 2);
            set_buffer(encoder, output_buffer.id(), 4);
            for (index, value) in [(5, 2u32), (6, 32), (7, 1), (8, 32)] {
                set_bytes(encoder, u32_as_bytes(&value), index);
            }
            dispatch_q4_threadgroups(encoder, 2);
            encoding.end_encoding();
            let (command_buffer, command_lease) = encoding.into_command_buffer();
            let context = MetalCommandContext::new("native MXFP4 reference test");
            commit_metal_command_buffer(command_buffer, &context);
            wait_for_metal_command_buffer(command_buffer, &context).unwrap();
            let actual = read_f32_buffer(output_buffer.id(), 2);
            release(command_buffer);
            drop(command_lease);
            assert!((actual[0] - 528.0).abs() < 1e-4, "{actual:?}");
            assert!((actual[1] - 632.0).abs() < 1e-4, "{actual:?}");
        });
    }

    #[test]
    fn pipeline_set_releases_every_resolved_pipeline() {
        let pipelines = test_pipeline_set();
        let mut released = Vec::new();
        pipelines.release_with(|pipeline| released.push(pipeline));
        assert_eq!(
            released,
            [
                vec![1, 2, 38, 3, 4, 5, 6, 7, 8],
                vec![33, 37, 34, 35, 36],
                vec![24, 25, 26, 41, 42, 43],
                vec![9, 10, 11, 39, 12, 13, 14, 15, 44, 40, 16],
                vec![18, 19, 27, 28, 20, 21, 29, 30, 22, 23, 31, 32],
            ]
            .concat()
        );
    }

    fn test_pipeline_set() -> MetalPipelineSet<i32> {
        MetalPipelineSet {
            q4_pipeline: 1,
            q4_bf16_scale_bias_pipeline: 2,
            mxfp4_e8m0_pipeline: 38,
            q4_swiglu_pipeline: 3,
            q4_swiglu_bf16_scale_bias_pipeline: 4,
            q4_mmap_pipeline: 5,
            q4_mmap_bf16_scale_bias_pipeline: 6,
            q4_mmap_batch_pipeline: 7,
            q4_mmap_batch_bf16_scale_bias_pipeline: 8,
            q4_mmap_multilinear_bf16_scale_bias_pipeline: 33,
            glm_mla_prepare_query_kv_pipeline: 37,
            glm_mla_absorbed_scores_pipeline: 34,
            glm_mla_softmax_pipeline: 35,
            glm_mla_context_pipeline: 36,
            dense_mmap_bf16_pipeline: 24,
            dense_mmap_f16_pipeline: 25,
            dense_mmap_f32_pipeline: 26,
            dense_matrix_bf16_pipeline: 41,
            dense_matrix_f16_pipeline: 42,
            dense_matrix_f32_pipeline: 43,
            rms_norm_reduced_pipeline: 9,
            residual_rms_norm_pipeline: 10,
            attention_pipeline: 11,
            qwen_causal_attention_rows_pipeline: 39,
            expert_mlp_pipeline: 12,
            silu_product_pipeline: 13,
            shared_expert_activation_pipeline: 14,
            combine_expert_phase_pipeline: 15,
            qwen_layer_major_gather_pipeline: 44,
            qwen_layer_major_combine_pipeline: 40,
            fill_zero_pipeline: 16,
            topk_vocab_pipeline: 18,
            linear_conv1d_bf16_pipeline: 19,
            linear_conv1d_f16_pipeline: 27,
            linear_conv1d_f32_pipeline: 28,
            linear_rms_norm_qk_pipeline: 20,
            linear_decay_beta_bf16_pipeline: 21,
            linear_decay_beta_f16_pipeline: 29,
            linear_decay_beta_f32_pipeline: 30,
            linear_delta_step_pipeline: 22,
            linear_gated_rms_norm_bf16_pipeline: 23,
            linear_gated_rms_norm_f16_pipeline: 31,
            linear_gated_rms_norm_f32_pipeline: 32,
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_objc_pipeline_set(id: MetalObjcId) -> MetalPipelineSet<MetalObjcId> {
        MetalPipelineSet {
            q4_pipeline: id,
            q4_bf16_scale_bias_pipeline: id,
            mxfp4_e8m0_pipeline: id,
            q4_swiglu_pipeline: id,
            q4_swiglu_bf16_scale_bias_pipeline: id,
            q4_mmap_pipeline: id,
            q4_mmap_bf16_scale_bias_pipeline: id,
            q4_mmap_batch_pipeline: id,
            q4_mmap_batch_bf16_scale_bias_pipeline: id,
            q4_mmap_multilinear_bf16_scale_bias_pipeline: id,
            glm_mla_prepare_query_kv_pipeline: id,
            glm_mla_absorbed_scores_pipeline: id,
            glm_mla_softmax_pipeline: id,
            glm_mla_context_pipeline: id,
            dense_mmap_bf16_pipeline: id,
            dense_mmap_f16_pipeline: id,
            dense_mmap_f32_pipeline: id,
            dense_matrix_bf16_pipeline: id,
            dense_matrix_f16_pipeline: id,
            dense_matrix_f32_pipeline: id,
            rms_norm_reduced_pipeline: id,
            residual_rms_norm_pipeline: id,
            attention_pipeline: id,
            qwen_causal_attention_rows_pipeline: id,
            expert_mlp_pipeline: id,
            silu_product_pipeline: id,
            shared_expert_activation_pipeline: id,
            combine_expert_phase_pipeline: id,
            qwen_layer_major_gather_pipeline: id,
            qwen_layer_major_combine_pipeline: id,
            fill_zero_pipeline: id,
            topk_vocab_pipeline: id,
            linear_conv1d_bf16_pipeline: id,
            linear_conv1d_f16_pipeline: id,
            linear_conv1d_f32_pipeline: id,
            linear_rms_norm_qk_pipeline: id,
            linear_decay_beta_bf16_pipeline: id,
            linear_decay_beta_f16_pipeline: id,
            linear_decay_beta_f32_pipeline: id,
            linear_delta_step_pipeline: id,
            linear_gated_rms_norm_bf16_pipeline: id,
            linear_gated_rms_norm_f16_pipeline: id,
            linear_gated_rms_norm_f32_pipeline: id,
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn batch_projection_input_reports_declared_input_len() {
        let values = [1.0, 2.0, 3.0];
        assert_eq!(MetalBatchProjectionInput::Cpu(&values).len(), values.len());

        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        assert_eq!(
            MetalBatchProjectionInput::Buffer { buffer: id, len: 7 }.len(),
            7
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn qwen_layer_major_group_plan_preserves_route_and_expert_order() {
        let (source_rows, output_indices, groups) =
            qwen_layer_major_group_plan(&[1, 0, 0, 1], 2, 2).unwrap();

        assert_eq!(source_rows, vec![0, 1, 0, 1]);
        assert_eq!(output_indices, vec![2, 0, 1, 3]);
        assert_eq!(groups, vec![(0, 2), (2, 2)]);

        assert!(qwen_layer_major_group_plan(&[0, 0], 2, 1).is_err());
        assert!(qwen_layer_major_group_plan(&[2], 2, 1).is_err());
        assert!(qwen_layer_major_group_plan(&[], 0, 0).is_err());
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn post_attention_prep_builds_declared_cmd3_metal_input() {
        let mut prep = MetalPostAttentionPrep::new(
            3,
            8,
            16,
            vec![(2, 0.75), (5, 0.25)],
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .unwrap();

        assert_eq!(prep.width, 8);
        assert_eq!(prep.active, vec![(2, 0.75), (5, 0.25)]);
        assert!(prep.routing_command().is_none());
        assert_eq!(prep.input.state(), prep.state);
        assert!(prep.state.is_declared_graph_state());

        let command = test_fused_prep_routing_command(3, 16, &prep.active);
        let attached = prep.attach_routing_command(command.clone()).unwrap();
        assert_eq!(attached, command);
        assert_eq!(prep.routing_command(), Some(&command));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn post_attention_prep_rejects_mismatched_routing_command() {
        let mut prep = MetalPostAttentionPrep::new(
            3,
            8,
            16,
            vec![(2, 0.75), (5, 0.25)],
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .unwrap();
        let command = test_fused_prep_routing_command(4, 16, &prep.active);

        let err = prep.attach_routing_command(command).unwrap_err();
        assert!(err.to_string().contains("routing layer 3"), "{err:#}");
        assert!(prep.routing_command().is_none());
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn post_attention_prep_rejects_undeclared_cmd3_metal_input() {
        let err = MetalPostAttentionPrep::new(
            3,
            0,
            16,
            vec![(2, 1.0)],
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("Metal post-attention input for layer 3")
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_deferred_output_declares_gpu_resident_buffers() {
        let hidden = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let next_normed = hidden;
        let output = MetalCmd3DeferredOutput::new(
            hidden,
            Some(next_normed),
            FlashMoeCmd3OutputState::gpu_resident(16, true),
        )
        .unwrap();

        assert_eq!(output.hidden_buffer, hidden);
        assert_eq!(output.next_normed_buffer, Some(next_normed));
        assert_eq!(output.output_state.width(), 16);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_deferred_output_rejects_undeclared_buffer_state() {
        let hidden = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();

        let missing_next = MetalCmd3DeferredOutput::new(
            hidden,
            None,
            FlashMoeCmd3OutputState::gpu_resident(16, true),
        )
        .unwrap_err();
        assert!(
            missing_next
                .to_string()
                .contains("next-norm buffer presence"),
            "{missing_next:#}"
        );

        let null_hidden = MetalCmd3DeferredOutput::new(
            std::ptr::null_mut(),
            None,
            FlashMoeCmd3OutputState::gpu_resident(16, false),
        )
        .unwrap_err();
        assert!(
            null_hidden.to_string().contains("non-null hidden buffer"),
            "{null_hidden:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_phase_plan_declares_supported_command_shape() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(16, true);

        let plan = MetalCmd3PhasePlan::new(9, 3, 4, 16, 4, 4, output_state, true).unwrap();

        assert_eq!(plan.position, 9);
        assert_eq!(plan.layer, 3);
        assert_eq!(plan.expert_count, 4);
        assert_eq!(plan.width, 16);
        assert_eq!(plan.output_state, output_state);
        assert!(plan.has_next_norm);
        assert_eq!(plan.width_u32(), 16);
        assert_eq!(plan.expert_outputs_bytes().unwrap(), 4 * 16 * 4);
        assert_eq!(plan.shared_output_bytes().unwrap(), 16 * 4);
        assert_eq!(plan.hidden_output_bytes().unwrap(), 16 * 4);
        assert_eq!(plan.next_normed_output_bytes().unwrap(), Some(16 * 4));
        assert_eq!(plan.expert_output_offset(0).unwrap(), 0);
        assert_eq!(plan.expert_output_offset(3).unwrap(), 3 * 16 * 4);

        let combine = MetalCmd3CombinePlan::new(plan);
        assert_eq!(combine.width, 16);
        assert_eq!(combine.active_count, 4);
        assert_eq!(combine.active_count_u32(), 4);
        assert_eq!(combine.dispatch_threads, 16);
        assert_eq!(combine.routing_weights_bytes().unwrap(), 4 * 4);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_phase_plan_rejects_unsupported_command_shape() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(16, true);

        let count_err = MetalCmd3PhasePlan::new(9, 3, 4, 16, 3, 4, output_state, true).unwrap_err();
        assert!(
            count_err.to_string().contains("expert count 4"),
            "{count_err:#}"
        );

        let output_err =
            MetalCmd3PhasePlan::new(9, 3, 4, 16, 4, 4, output_state, false).unwrap_err();
        assert!(
            output_err
                .to_string()
                .contains("next-norm output declaration"),
            "{output_err:#}"
        );

        let wide_state = FlashMoeCmd3OutputState::gpu_resident(u32::MAX as usize + 1, false);
        let width_err =
            MetalCmd3PhasePlan::new(9, 3, 4, u32::MAX as usize + 1, 4, 4, wide_state, false)
                .unwrap_err();
        assert!(
            width_err.to_string().contains("does not fit Metal u32"),
            "{width_err:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_input_buffers_carry_declared_phase_inputs() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();

        let inputs = MetalCmd3InputBuffers::new(
            phase,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
        )
        .unwrap();

        assert_eq!(inputs.normed, 0x1000usize as MetalObjcId);
        assert_eq!(inputs.residual, 0x2000usize as MetalObjcId);
        assert_eq!(inputs.phase, phase);

        let missing_normed =
            MetalCmd3InputBuffers::new(phase, std::ptr::null_mut(), inputs.residual).unwrap_err();
        assert!(
            missing_normed.to_string().contains("non-null normed"),
            "{missing_normed:#}"
        );

        let missing_residual =
            MetalCmd3InputBuffers::new(phase, inputs.normed, std::ptr::null_mut()).unwrap_err();
        assert!(
            missing_residual.to_string().contains("non-null residual"),
            "{missing_residual:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_combine_buffers_carry_declared_bindings() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
        let plan = MetalCmd3CombinePlan::new(phase);

        let buffers = MetalCmd3CombineBuffers::new(
            plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
        )
        .unwrap();

        assert_eq!(buffers.routing_weights, 0x1000usize as MetalObjcId);
        assert_eq!(buffers.width, 0x2000usize as MetalObjcId);
        assert_eq!(buffers.active_count, 0x3000usize as MetalObjcId);
        assert_eq!(buffers.layout.width_u32, 4);
        assert_eq!(buffers.layout.active_count_u32, 2);
        assert_eq!(buffers.layout.routing_weights_bytes, 2 * 4);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_combine_stage_buffers_match_declared_layout() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let payloads = vec![
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
        ];
        let execution = MetalCmd3ExecutionPlan::new(
            9,
            3,
            2,
            4,
            2,
            output_state,
            ScheduledSharedExpertPhaseRef::None,
            None,
            &payloads,
        )
        .unwrap();
        let inputs = MetalCmd3InputBuffers::new(
            execution.phase,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
        )
        .unwrap();
        let outputs = MetalCmd3OutputBuffers::new(
            &execution,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
            0x5000usize as MetalObjcId,
            None,
        )
        .unwrap();
        let combine = MetalCmd3CombineBuffers::new(
            execution.combine,
            0x6000usize as MetalObjcId,
            0x7000usize as MetalObjcId,
            0x8000usize as MetalObjcId,
        )
        .unwrap();

        let stage = MetalCmd3CombineStageBuffers::new(execution.combine, inputs, &outputs, combine)
            .unwrap();

        assert_eq!(stage.residual, inputs.residual);
        assert_eq!(stage.shared_output, outputs.shared_output);
        assert_eq!(stage.expert_outputs, outputs.expert_outputs);
        assert_eq!(stage.routing_weights, combine.routing_weights);
        assert_eq!(stage.hidden, outputs.hidden);
        assert_eq!(stage.width, combine.width);
        assert_eq!(stage.active_count, combine.active_count);
        assert_eq!(stage.plan, execution.combine);

        let stale_plan = MetalCmd3CombinePlan {
            width: execution.combine.width,
            active_count: execution.combine.active_count + 1,
            dispatch_threads: execution.combine.dispatch_threads,
        };
        let stale =
            MetalCmd3CombineStageBuffers::new(stale_plan, inputs, &outputs, combine).unwrap_err();
        assert!(
            stale.to_string().contains("constants do not match plan"),
            "{stale:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_next_norm_buffers_carry_declared_bindings() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, true);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, true).unwrap();
        let plan = MetalCmd3NextNormPlan::new(phase, Some(4)).unwrap().unwrap();

        let buffers = MetalCmd3NextNormBuffers::new(
            plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
        )
        .unwrap();

        assert_eq!(buffers.hidden, 0x1000usize as MetalObjcId);
        assert_eq!(buffers.weight, 0x2000usize as MetalObjcId);
        assert_eq!(buffers.next_normed, 0x3000usize as MetalObjcId);
        assert_eq!(buffers.width, 0x4000usize as MetalObjcId);
        assert_eq!(buffers.layout.width_u32, 4);
        assert_eq!(buffers.layout.weight_bytes, 4 * 4);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_shared_stage_buffers_require_declared_source() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let shared = SharedExpertPhaseResidentProjections {
            gate: test_resident_q4_projection("gate", 6, 4),
            up: test_resident_q4_projection("up", 6, 4),
            down: test_resident_q4_projection("down", 4, 6),
            router: Some(test_resident_q4_projection("router", 2, 4)),
            shared_experts: 2,
            intermediate: 3,
            width: 4,
        };
        let payloads = vec![
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
        ];
        let plan = MetalCmd3ExecutionPlan::new(
            9,
            3,
            2,
            4,
            2,
            output_state,
            ScheduledSharedExpertPhaseRef::Resident(&shared),
            None,
            &payloads,
        )
        .unwrap();
        let inputs = MetalCmd3InputBuffers::new(
            plan.phase,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
        )
        .unwrap();
        let outputs = MetalCmd3OutputBuffers::new(
            &plan,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
            0x5000usize as MetalObjcId,
            None,
        )
        .unwrap();
        let combine = MetalCmd3CombineBuffers::new(
            plan.combine,
            0x6000usize as MetalObjcId,
            0x7000usize as MetalObjcId,
            0x8000usize as MetalObjcId,
        )
        .unwrap();
        let work = MetalCmd3SharedWorkBuffers::new(
            plan.shared,
            0x9000usize as MetalObjcId,
            0xa000usize as MetalObjcId,
            0xb000usize as MetalObjcId,
            0xc000usize as MetalObjcId,
            0xd000usize as MetalObjcId,
            0xe000usize as MetalObjcId,
        )
        .unwrap();

        let projected =
            MetalCmd3SharedStageBuffers::projected(plan.shared, inputs, &outputs, combine, work)
                .unwrap();

        assert_eq!(projected.source, MetalCmd3SharedPhaseSource::Resident);
        assert_eq!(projected.normed, inputs.normed);
        assert_eq!(projected.width, combine.width);
        assert_eq!(projected.shared_output, outputs.shared_output);
        assert_eq!(projected.work, Some(work));

        let no_shared = MetalCmd3SharedPhasePlan::none(4);
        let fill_zero =
            MetalCmd3SharedStageBuffers::fill_zero(no_shared, inputs, &outputs, combine).unwrap();
        assert_eq!(fill_zero.source, MetalCmd3SharedPhaseSource::None);
        assert_eq!(fill_zero.work, None);
        assert_eq!(fill_zero.shared_output, outputs.shared_output);

        let projected_none =
            MetalCmd3SharedStageBuffers::projected(no_shared, inputs, &outputs, combine, work)
                .unwrap_err();
        assert!(
            projected_none
                .to_string()
                .contains("declared shared expert source"),
            "{projected_none:#}"
        );

        let fill_projected =
            MetalCmd3SharedStageBuffers::fill_zero(plan.shared, inputs, &outputs, combine)
                .unwrap_err();
        assert!(
            fill_projected
                .to_string()
                .contains("no shared expert source"),
            "{fill_projected:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_execution_plan_declares_full_command_topology() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, true);
        let shared = SharedExpertPhaseResidentProjections {
            gate: test_resident_q4_projection("gate", 6, 4),
            up: test_resident_q4_projection("up", 6, 4),
            down: test_resident_q4_projection("down", 4, 6),
            router: Some(test_resident_q4_projection("router", 2, 4)),
            shared_experts: 2,
            intermediate: 3,
            width: 4,
        };
        let payloads = vec![
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
        ];

        let plan = MetalCmd3ExecutionPlan::new(
            9,
            3,
            2,
            4,
            2,
            output_state,
            ScheduledSharedExpertPhaseRef::Resident(&shared),
            Some(4),
            &payloads,
        )
        .unwrap();

        assert_eq!(plan.phase.position, 9);
        assert_eq!(plan.phase.layer, 3);
        assert_eq!(plan.phase.output_state, output_state);
        assert_eq!(plan.shared.source, MetalCmd3SharedPhaseSource::Resident);
        assert_eq!(plan.shared.total_intermediate, 6);
        assert_eq!(plan.active_experts.len(), 2);
        assert_eq!(plan.active_experts[0].intermediate, 5);
        assert_eq!(plan.active_experts[0].output_offset, 0);
        assert_eq!(plan.active_experts[1].intermediate, 7);
        assert_eq!(plan.active_experts[1].output_offset, 4 * 4);
        assert_eq!(plan.combine.active_count, 2);
        assert_eq!(plan.next_norm.unwrap().width, 4);

        let layout = plan.buffer_layout().unwrap();
        assert_eq!(layout.width_u32, 4);
        assert_eq!(layout.active_count_u32, 2);
        assert_eq!(layout.expert_outputs_bytes, 2 * 4 * 4);
        assert_eq!(layout.shared_output_bytes, 4 * 4);
        assert_eq!(layout.hidden_output_bytes, 4 * 4);
        assert_eq!(layout.next_normed_output_bytes, Some(4 * 4));

        let context = plan.command_context("1,7");
        assert_eq!(
            context.label(),
            "Flash-MoE deferred_expert_phase_from_buffers position=9 layer=3 active_experts=2 experts=1,7 width=4 shared=true next_norm=true"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_output_buffers_match_declared_output_state() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, true);
        let payloads = vec![ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(
            5, 4,
        ))];
        let plan = MetalCmd3ExecutionPlan::new(
            9,
            3,
            1,
            4,
            1,
            output_state,
            ScheduledSharedExpertPhaseRef::None,
            Some(4),
            &payloads,
        )
        .unwrap();

        let buffers = MetalCmd3OutputBuffers::new(
            &plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            Some(0x4000usize as MetalObjcId),
        )
        .unwrap();

        assert_eq!(buffers.layout.width_u32, 4);
        assert_eq!(buffers.layout.active_count_u32, 1);
        assert_eq!(buffers.layout.expert_outputs_bytes, 4 * 4);
        assert_eq!(buffers.layout.shared_output_bytes, 4 * 4);
        assert_eq!(buffers.layout.hidden_output_bytes, 4 * 4);
        assert_eq!(buffers.layout.next_normed_output_bytes, Some(4 * 4));
        assert_eq!(buffers.hidden, 0x3000usize as MetalObjcId);

        let missing_next = MetalCmd3OutputBuffers::new(
            &plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            None,
        )
        .unwrap_err();
        assert!(
            missing_next
                .to_string()
                .contains("does not match declared output state"),
            "{missing_next:#}"
        );

        let no_next_plan = MetalCmd3ExecutionPlan::new(
            9,
            3,
            1,
            4,
            1,
            FlashMoeCmd3OutputState::gpu_resident(4, false),
            ScheduledSharedExpertPhaseRef::None,
            None,
            &payloads,
        )
        .unwrap();
        let unexpected_next = MetalCmd3OutputBuffers::new(
            &no_next_plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            Some(0x4000usize as MetalObjcId),
        )
        .unwrap_err();
        assert!(
            unexpected_next
                .to_string()
                .contains("does not match declared output state"),
            "{unexpected_next:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_active_expert_work_buffers_carry_staged_projection_layout() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
        let payload = ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(6, 4));
        let plan = MetalCmd3ActiveExpertPlan::new(phase, 1, &payload).unwrap();

        let staged = MetalCmd3ActiveExpertWorkBuffers::new(
            plan,
            Some(0x1100usize as MetalObjcId),
            Some(0x1200usize as MetalObjcId),
            0x1000usize as MetalObjcId,
        )
        .unwrap();

        assert_eq!(staged.gate_out, Some(0x1100usize as MetalObjcId));
        assert_eq!(staged.up_out, Some(0x1200usize as MetalObjcId));
        assert_eq!(staged.activated, 0x1000usize as MetalObjcId);
        assert_eq!(staged.layout.intermediate_u32, 6);
        assert_eq!(staged.layout.activation_bytes, 6 * 4);
        assert_eq!(staged.layout.projection_output_bytes, Some(6 * 4));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_active_expert_stage_buffers_match_declared_layout() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let payloads = vec![
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
            ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
        ];
        let execution = MetalCmd3ExecutionPlan::new(
            9,
            3,
            2,
            4,
            2,
            output_state,
            ScheduledSharedExpertPhaseRef::None,
            None,
            &payloads,
        )
        .unwrap();
        let inputs = MetalCmd3InputBuffers::new(
            execution.phase,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
        )
        .unwrap();
        let outputs = MetalCmd3OutputBuffers::new(
            &execution,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
            0x5000usize as MetalObjcId,
            None,
        )
        .unwrap();
        let active_plan = execution.active_experts[1];
        let work = MetalCmd3ActiveExpertWorkBuffers::new(
            active_plan,
            Some(0x6100usize as MetalObjcId),
            Some(0x6200usize as MetalObjcId),
            0x6000usize as MetalObjcId,
        )
        .unwrap();

        let stage =
            MetalCmd3ActiveExpertStageBuffers::new(active_plan, inputs, &outputs, work).unwrap();

        assert_eq!(stage.normed, inputs.normed);
        assert_eq!(stage.activated, work.activated);
        assert_eq!(stage.expert_outputs, outputs.expert_outputs);
        assert_eq!(stage.output_offset, active_plan.output_offset);
        assert_eq!(stage.plan, active_plan);
        assert_eq!(stage.work, work);

        let stale_plan = MetalCmd3ActiveExpertPlan {
            intermediate: active_plan.intermediate + 1,
            ..active_plan
        };
        let stale =
            MetalCmd3ActiveExpertStageBuffers::new(stale_plan, inputs, &outputs, work).unwrap_err();
        assert!(
            stale
                .to_string()
                .contains("work layout does not match plan"),
            "{stale:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_execution_plan_rejects_mismatched_subplans() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let payloads = vec![ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(
            5, 6,
        ))];

        let payload_err = MetalCmd3ExecutionPlan::new(
            9,
            3,
            1,
            4,
            1,
            output_state,
            ScheduledSharedExpertPhaseRef::None,
            None,
            &payloads,
        )
        .unwrap_err();
        assert!(
            payload_err
                .to_string()
                .contains("does not match phase width 4"),
            "{payload_err:#}"
        );

        let shared = SharedExpertPhaseWeights::new(
            Arc::new(vec![1.0; 24]),
            Arc::new(vec![2.0; 24]),
            Arc::new(vec![3.0; 24]),
            Arc::new(vec![4.0; 8]),
            2,
            3,
            4,
        )
        .unwrap();
        let payloads = vec![ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(
            5, 8,
        ))];
        let shared_err = MetalCmd3ExecutionPlan::new(
            9,
            3,
            1,
            8,
            1,
            FlashMoeCmd3OutputState::gpu_resident(8, false),
            ScheduledSharedExpertPhaseRef::Dense(&shared),
            None,
            &payloads,
        )
        .unwrap_err();
        assert!(
            shared_err.to_string().contains("shared expert width 4"),
            "{shared_err:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn scheduled_cmd3_builder_names_missing_shared_implementation() {
        assert!(
            MetalScheduledCmd3Builder::require_shared_implementation(
                MetalCmd3SharedPhaseSource::Resident
            )
            .is_ok()
        );
        assert!(
            MetalScheduledCmd3Builder::require_shared_implementation(
                MetalCmd3SharedPhaseSource::None
            )
            .is_ok()
        );
        let error = MetalScheduledCmd3Builder::require_shared_implementation(
            MetalCmd3SharedPhaseSource::Dense,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("dense CPU shared-expert weights are not a declared implementation"),
            "{error:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_next_norm_plan_declares_weight_slice_and_dispatch() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(16, true);
        let phase = MetalCmd3PhasePlan::new(9, 3, 4, 16, 4, 4, output_state, true).unwrap();

        let plan = MetalCmd3NextNormPlan::new(phase, Some(32))
            .unwrap()
            .unwrap();

        assert_eq!(plan.width, 16);
        assert_eq!(plan.dispatch_threads, 256);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_next_norm_plan_rejects_undeclared_or_short_weights() {
        let with_next = MetalCmd3PhasePlan::new(
            9,
            3,
            4,
            16,
            4,
            4,
            FlashMoeCmd3OutputState::gpu_resident(16, true),
            true,
        )
        .unwrap();
        let short = MetalCmd3NextNormPlan::new(with_next, Some(15)).unwrap_err();
        assert!(
            short.to_string().contains("smaller than width 16"),
            "{short:#}"
        );

        let missing = MetalCmd3NextNormPlan::new(with_next, None).unwrap_err();
        assert!(
            missing.to_string().contains("no next-norm weights"),
            "{missing:#}"
        );

        let without_next = MetalCmd3PhasePlan::new(
            9,
            3,
            4,
            16,
            4,
            4,
            FlashMoeCmd3OutputState::gpu_resident(16, false),
            false,
        )
        .unwrap();
        let unexpected = MetalCmd3NextNormPlan::new(without_next, Some(16)).unwrap_err();
        assert!(
            unexpected
                .to_string()
                .contains("provided for a no-next-norm phase"),
            "{unexpected:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_active_expert_plan_declares_payload_layout() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
        let payload = ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(6, 4));

        let plan = MetalCmd3ActiveExpertPlan::new(phase, 1, &payload).unwrap();

        assert_eq!(plan.index, 1);
        assert_eq!(plan.source, MetalCmd3ActiveExpertSource::Q4);
        assert_eq!(plan.intermediate, 6);
        assert_eq!(plan.intermediate_u32().unwrap(), 6);
        assert_eq!(plan.activation_bytes().unwrap(), 6 * 4);
        assert_eq!(plan.projection_output_bytes().unwrap(), 6 * 4);
        assert_eq!(plan.output_offset, 4 * 4);
        assert_eq!(
            plan.buffer_layout().unwrap(),
            MetalCmd3ActiveExpertBufferLayout {
                intermediate_u32: 6,
                activation_bytes: 6 * 4,
                projection_output_bytes: Some(6 * 4),
            }
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_active_expert_plan_rejects_mismatched_payload() {
        let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
        let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
        let payload = ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(6, 5));

        let err = MetalCmd3ActiveExpertPlan::new(phase, 0, &payload).unwrap_err();

        assert!(
            err.to_string().contains("does not match phase width 4"),
            "{err:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_shared_phase_plan_declares_dense_shape() {
        let shared = SharedExpertPhaseWeights::new(
            Arc::new(vec![1.0; 24]),
            Arc::new(vec![2.0; 24]),
            Arc::new(vec![3.0; 24]),
            Arc::new(vec![4.0; 8]),
            2,
            3,
            4,
        )
        .unwrap();

        let plan = MetalCmd3SharedPhasePlan::dense(4, &shared).unwrap();

        assert_eq!(plan.source, MetalCmd3SharedPhaseSource::Dense);
        assert_eq!(plan.width, 4);
        assert_eq!(plan.shared_experts, 2);
        assert_eq!(plan.intermediate, 3);
        assert_eq!(plan.total_intermediate, 6);
        assert_eq!(plan.total_intermediate_u32().unwrap(), 6);
        assert_eq!(plan.intermediate_u32().unwrap(), 3);
        assert_eq!(plan.projection_output_bytes().unwrap(), 6 * 4);
        assert_eq!(plan.router_output_bytes().unwrap(), 2 * 4);
        assert_eq!(plan.projection_rows(), 6);
        assert_eq!(plan.router_rows(), 2);
        assert_eq!(plan.activation_dispatch_threads(), 6);
        assert_eq!(
            plan.buffer_layout().unwrap(),
            MetalCmd3SharedBufferLayout {
                total_intermediate_u32: 6,
                intermediate_u32: 3,
                projection_output_bytes: 6 * 4,
                router_output_bytes: 2 * 4,
            }
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_shared_phase_plan_declares_resident_shape() {
        let shared = SharedExpertPhaseResidentProjections {
            gate: test_resident_q4_projection("gate", 6, 4),
            up: test_resident_q4_projection("up", 6, 4),
            down: test_resident_q4_projection("down", 4, 6),
            router: Some(test_resident_q4_projection("router", 2, 4)),
            shared_experts: 2,
            intermediate: 3,
            width: 4,
        };

        let plan = MetalCmd3SharedPhasePlan::resident(4, &shared).unwrap();

        assert_eq!(plan.source, MetalCmd3SharedPhaseSource::Resident);
        assert_eq!(plan.width, 4);
        assert_eq!(plan.shared_experts, 2);
        assert_eq!(plan.intermediate, 3);
        assert_eq!(plan.total_intermediate, 6);
        assert_eq!(plan.total_intermediate_u32().unwrap(), 6);
        assert_eq!(plan.intermediate_u32().unwrap(), 3);
        assert_eq!(plan.projection_output_bytes().unwrap(), 6 * 4);
        assert_eq!(plan.router_output_bytes().unwrap(), 2 * 4);
        assert_eq!(plan.projection_rows(), 6);
        assert_eq!(plan.router_rows(), 2);
        assert_eq!(plan.activation_dispatch_threads(), 6);
        assert_eq!(
            plan.buffer_layout().unwrap(),
            MetalCmd3SharedBufferLayout {
                total_intermediate_u32: 6,
                intermediate_u32: 3,
                projection_output_bytes: 6 * 4,
                router_output_bytes: 2 * 4,
            }
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_shared_work_buffers_carry_declared_layout() {
        let shared = SharedExpertPhaseResidentProjections {
            gate: test_resident_q4_projection("gate", 6, 4),
            up: test_resident_q4_projection("up", 6, 4),
            down: test_resident_q4_projection("down", 4, 6),
            router: Some(test_resident_q4_projection("router", 2, 4)),
            shared_experts: 2,
            intermediate: 3,
            width: 4,
        };
        let plan = MetalCmd3SharedPhasePlan::resident(4, &shared).unwrap();

        let buffers = MetalCmd3SharedWorkBuffers::new(
            plan,
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
            0x5000usize as MetalObjcId,
            0x6000usize as MetalObjcId,
        )
        .unwrap();

        assert_eq!(buffers.layout.total_intermediate_u32, 6);
        assert_eq!(buffers.layout.intermediate_u32, 3);
        assert_eq!(buffers.layout.projection_output_bytes, 6 * 4);
        assert_eq!(buffers.layout.router_output_bytes, 2 * 4);
        assert_eq!(buffers.gate_out, 0x1000usize as MetalObjcId);

        let err = MetalCmd3SharedWorkBuffers::new(
            MetalCmd3SharedPhasePlan::none(4),
            0x1000usize as MetalObjcId,
            0x2000usize as MetalObjcId,
            0x3000usize as MetalObjcId,
            0x4000usize as MetalObjcId,
            0x5000usize as MetalObjcId,
            0x6000usize as MetalObjcId,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("declared shared expert source"),
            "{err:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd3_shared_phase_plan_rejects_width_mismatch() {
        let shared = SharedExpertPhaseWeights::new(
            Arc::new(vec![1.0; 24]),
            Arc::new(vec![2.0; 24]),
            Arc::new(vec![3.0; 24]),
            Arc::new(vec![4.0; 8]),
            2,
            3,
            4,
        )
        .unwrap();

        let err = MetalCmd3SharedPhasePlan::dense(8, &shared).unwrap_err();

        assert!(err.to_string().contains("shared expert width 4"), "{err:#}");

        let huge_shape = SharedExpertPhaseShape::new(1, u32::MAX as usize + 1, 1).unwrap();
        let huge_err =
            MetalCmd3SharedPhasePlan::from_shape(MetalCmd3SharedPhaseSource::Dense, 1, huge_shape)
                .unwrap_err();
        assert!(
            huge_err.to_string().contains("does not fit Metal u32"),
            "{huge_err:#}"
        );

        let none = MetalCmd3SharedPhasePlan::none(4);
        assert_eq!(none.fill_zero_width(), 4);
        assert_eq!(none.activation_dispatch_threads(), 0);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_q4_expert_payload(
        intermediate: usize,
        width: usize,
    ) -> ScheduledQ4ExpertPhaseMlpPayload<'static> {
        let gate = test_q4_matvec_payload(intermediate, width);
        let up = test_q4_matvec_payload(intermediate, width);
        let down = test_q4_matvec_payload(width, intermediate);
        ScheduledQ4ExpertPhaseMlpPayload::new(3, 1, width, gate, up, down).unwrap()
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    static TEST_Q4_EXPERT_SLOT: [u8; 4096] = [0; 4096];

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_q4_matvec_payload(rows: usize, cols: usize) -> Q4MatvecPayload<'static> {
        let packed_bytes = rows * cols.div_ceil(2);
        let scale_bias_groups = rows * cols.div_ceil(16);
        let scale_bias_bytes = scale_bias_groups * 2;
        Q4MatvecPayload {
            rows,
            cols,
            group_size: 16,
            packed: &TEST_Q4_EXPERT_SLOT[..packed_bytes],
            scales: &[],
            biases: &[],
            scale_bias_groups,
            scale_bias_dtype: "BF16",
            scale_bytes: &TEST_Q4_EXPERT_SLOT[1024..1024 + scale_bias_bytes],
            bias_bytes: &TEST_Q4_EXPERT_SLOT[2048..2048 + scale_bias_bytes],
            source: Some(super::super::experts::Q4MatvecSource {
                bytes: &TEST_Q4_EXPERT_SLOT,
                packed_offset: 0,
                scale_offset: 1024,
                bias_offset: 2048,
                reusable_bytes: None,
            }),
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_q4_projection(
        tensor_name: &str,
        output_width: usize,
        input_width: usize,
    ) -> DenseQ4MmapMatvecProjection {
        DenseQ4MmapMatvecProjection {
            tensor_name: tensor_name.to_string(),
            packed_byte_offset: 0,
            scales_byte_offset: 64,
            biases_byte_offset: 96,
            rows: output_width,
            cols: input_width,
            output_width,
            row_packed_bytes: input_width.div_ceil(2),
            groups_per_row: input_width.div_ceil(16),
            group_size: 16,
            scale_bias_dtype: "F32".to_string(),
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_resident_q4_projection(
        tensor_name: &str,
        output_width: usize,
        input_width: usize,
    ) -> ResidentMmapMatvecProjection {
        test_q4_projection(tensor_name, output_width, input_width).into()
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_fused_prep_routing_command(
        layer: usize,
        experts: usize,
        routes: &[(usize, f32)],
    ) -> ScheduledRoutingCommand {
        let stage = FlashMoeStageCapability::new(
            FlashMoeGraphStage::RoutingSoftmaxTopK,
            FlashMoeStagePlacement::CpuDeclared,
            FlashMoeStageImplementation::CpuSoftmaxTopK,
        );
        let routing = ScheduledRoutingTopK {
            stage,
            layer,
            experts,
            active_experts: routes.len(),
            source: ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
        };
        ScheduledRoutingCommand {
            routing,
            layer,
            active_experts: routes.len(),
            source: ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
            routes: routes.to_vec(),
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_linear_attention_state_cache_preserves_gpu_buffer_roles() {
        let base = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let state = MetalLinearAttentionLayerState::new(
            base, base, base, base, base, base, 12, 20, 4, 8, 2,
        );
        let cache = MetalLinearAttentionStateCache::new(vec![None, Some(state)]);
        let layer = cache.layers[1].as_ref().unwrap();

        assert_eq!(layer.conv_state, base);
        assert_eq!(layer.ssm_state, base);
        assert_eq!(layer.conv_output, base);
        assert_eq!(layer.delta_output, base);
        assert_eq!(layer.g_decay, base);
        assert_eq!(layer.beta_gate, base);
        assert_eq!(layer.conv_state_len, 12);
        assert_eq!(layer.ssm_state_len, 20);
        assert_eq!(layer.conv_dim, 4);
        assert_eq!(layer.total_value_width, 8);
        assert_eq!(layer.num_value_heads, 2);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_state_buffer_carries_validated_gpu_state_with_raw_binding() {
        let buffer = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let hidden = MetalStateBuffer::new(buffer, FlashMoeGpuBufferDescriptor::hidden(8)).unwrap();
        assert_eq!(hidden.buffer(), buffer);
        assert_eq!(hidden.len(), 8);
        assert_eq!(hidden.state(), FlashMoeGpuBufferDescriptor::hidden(8));

        let null_err = MetalStateBuffer::new(
            std::ptr::null_mut(),
            FlashMoeGpuBufferDescriptor::next_layer_normed(8),
        )
        .unwrap_err();
        assert!(
            null_err.to_string().contains("non-null buffer"),
            "{null_err:#}"
        );

        let empty_err =
            MetalStateBuffer::new(buffer, FlashMoeGpuBufferDescriptor::hidden(0)).unwrap_err();
        assert!(
            empty_err.to_string().contains("declared GpuResident state"),
            "{empty_err:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn recurrent_session_snapshot_validates_complete_resident_layer_table() {
        let base = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let resident = MetalLinearAttentionStateCache::new(vec![
            None,
            Some(MetalLinearAttentionLayerState::new(
                base, base, base, base, base, base, 2, 3, 4, 5, 2,
            )),
        ]);
        let matching = FlashMoeLinearAttentionSessionSnapshot::new(vec![
            None,
            Some(
                FlashMoeLinearAttentionLayerSnapshot::new(1, vec![1.0; 2], vec![2.0; 3], 4, 5)
                    .unwrap(),
            ),
        ])
        .unwrap();
        validate_linear_attention_session_snapshot(&resident, &matching).unwrap();

        let missing = FlashMoeLinearAttentionSessionSnapshot::new(vec![None, None]).unwrap();
        let missing_err =
            validate_linear_attention_session_snapshot(&resident, &missing).unwrap_err();
        assert!(
            missing_err
                .to_string()
                .contains("missing resolved linear-attention layer 1"),
            "{missing_err:#}"
        );

        let wrong_shape = FlashMoeLinearAttentionSessionSnapshot::new(vec![
            None,
            Some(
                FlashMoeLinearAttentionLayerSnapshot::new(1, vec![1.0; 2], vec![2.0; 4], 4, 5)
                    .unwrap(),
            ),
        ])
        .unwrap();
        let shape_err =
            validate_linear_attention_session_snapshot(&resident, &wrong_shape).unwrap_err();
        assert!(
            shape_err
                .to_string()
                .contains("does not match the resolved resident state"),
            "{shape_err:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn metal_nested_autorelease_releases_completed_command_resources() {
        objc2::rc::autoreleasepool(|_| unsafe {
            let physical_footprint = || {
                let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v2>::uninit();
                let result = libc::proc_pid_rusage(
                    libc::getpid(),
                    libc::RUSAGE_INFO_V2,
                    usage.as_mut_ptr().cast(),
                );
                assert_eq!(result, 0, "proc_pid_rusage failed");
                usage.assume_init().ri_phys_footprint
            };
            let runtime =
                MetalRuntime::compile(METAL_SHADERS, MetalPipelineNameSet::new()).unwrap();
            let resources = Arc::new(MetalResourceLedger::default());
            let baseline = msg_send_usize0(runtime.device, sel("currentAllocatedSize"));
            let baseline_footprint = physical_footprint();

            for _ in 0..16 {
                let (command_buffer, command_lease, buffers) = objc2::rc::autoreleasepool(|_| {
                    let buffers = (0..30)
                        .map(|_| {
                            OwnedMetalObject::new(msg_send_id2_usize_u64(
                                runtime.device,
                                sel("newBufferWithLength:options:"),
                                1024 * 1024,
                                0,
                            ))
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    for buffer in &buffers {
                        ptr::write_bytes(
                            msg_send_ptr0(buffer.id(), sel("contents")).cast::<u8>(),
                            0xA5,
                            1024 * 1024,
                        );
                    }
                    let mut encoding = MetalCommandEncoding::new(
                        runtime.command_queue,
                        Arc::clone(&resources),
                        "test command buffer allocation failed",
                        "test command encoder allocation failed",
                    )
                    .unwrap();
                    msg_send_void1_id(
                        encoding.encoder(),
                        sel("setComputePipelineState:"),
                        runtime.pipelines.fill_zero_pipeline,
                    );
                    set_buffer(encoding.encoder(), buffers[0].id(), 0);
                    for (index, buffer) in buffers.iter().enumerate().skip(1) {
                        set_buffer(encoding.encoder(), buffer.id(), index as u64 + 1);
                    }
                    for width in 1..=64u32 {
                        set_bytes(encoding.encoder(), u32_as_bytes(&width), 1);
                    }
                    dispatch_threads(encoding.encoder(), 1);
                    encoding.end_encoding();
                    let (command_buffer, command_lease) = encoding.into_command_buffer();
                    commit_metal_command_buffer(
                        command_buffer,
                        &MetalCommandContext::new("nested autorelease test"),
                    );
                    (command_buffer, command_lease, buffers)
                });
                objc2::rc::autoreleasepool(|_| {
                    wait_for_metal_command_buffer(
                        command_buffer,
                        &MetalCommandContext::new("nested autorelease completion test"),
                    )
                    .unwrap();
                    release(command_buffer);
                    drop(command_lease);
                    for buffer in buffers {
                        purge_and_release_metal_buffer(buffer.into_raw());
                    }
                });
            }

            let allocated = msg_send_usize0(runtime.device, sel("currentAllocatedSize"));
            let footprint = physical_footprint();
            assert!(
                allocated.saturating_sub(baseline) < 64 * 1024 * 1024,
                "completed commands retained {} bytes across nested autorelease pools",
                allocated.saturating_sub(baseline)
            );
            assert!(
                footprint.saturating_sub(baseline_footprint) < 192 * 1024 * 1024,
                "completed commands grew physical footprint by {} bytes",
                footprint.saturating_sub(baseline_footprint)
            );
        });
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn resident_q4_multilinear_uses_each_heads_own_input() {
        let heads = 2usize;
        let rows_per_head = 16usize;
        let cols = 8usize;
        let rows = heads * rows_per_head;
        let packed_bytes = rows * cols.div_ceil(2);
        let scalar_bytes = rows * std::mem::size_of::<u16>();
        let scales_offset = packed_bytes;
        let biases_offset = scales_offset + scalar_bytes;
        let mut mmap = memmap2::MmapMut::map_anon(4096).unwrap();
        for row in 0..rows {
            let nibble = (row % 7 + 1) as u8;
            mmap[row * 4..row * 4 + 4].fill(nibble | (nibble << 4));
            mmap[scales_offset + row * 2..scales_offset + row * 2 + 2]
                .copy_from_slice(&0x3f80u16.to_le_bytes());
            mmap[biases_offset + row * 2..biases_offset + row * 2 + 2]
                .copy_from_slice(&0u16.to_le_bytes());
        }
        let mmap = Arc::new(mmap.make_read_only().unwrap());
        let context = MetalExecutionContext::compile(mmap, 4096, &[], 1e-6).unwrap();
        let projection = DenseQ4MmapMatvecProjection {
            tensor_name: "test.multilinear.weight".to_string(),
            packed_byte_offset: 0,
            scales_byte_offset: scales_offset as u64,
            biases_byte_offset: biases_offset as u64,
            rows,
            cols,
            output_width: rows,
            row_packed_bytes: cols.div_ceil(2),
            groups_per_row: 1,
            group_size: cols,
            scale_bias_dtype: "BF16".to_string(),
        };
        let inputs = [vec![1.0; cols], vec![2.0; cols]].concat();
        let actual = context
            .resident_q4_multilinear(&projection, heads, rows_per_head, &inputs)
            .unwrap()
            .unwrap();
        let expected = (0..rows)
            .map(|row| {
                let nibble = (row % 7 + 1) as f32;
                let input = if row < rows_per_head { 1.0 } else { 2.0 };
                nibble * cols as f32 * input
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn resident_glm_mla_absorbed_attention_matches_reference() {
        fn write_projection(
            mmap: &mut [u8],
            offset: &mut usize,
            name: &str,
            rows: usize,
            cols: usize,
            nibble: impl Fn(usize) -> u8,
        ) -> DenseQ4MmapMatvecProjection {
            let packed_byte_offset = *offset;
            let row_packed_bytes = cols.div_ceil(2);
            for row in 0..rows {
                let value = nibble(row) & 0x0f;
                mmap[packed_byte_offset + row * row_packed_bytes
                    ..packed_byte_offset + (row + 1) * row_packed_bytes]
                    .fill(value | (value << 4));
            }
            let scales_byte_offset = packed_byte_offset + rows * row_packed_bytes;
            for row in 0..rows {
                mmap[scales_byte_offset + row * 2..scales_byte_offset + row * 2 + 2]
                    .copy_from_slice(&0x3f80u16.to_le_bytes());
            }
            let biases_byte_offset = scales_byte_offset + rows * 2;
            mmap[biases_byte_offset..biases_byte_offset + rows * 2].fill(0);
            *offset = biases_byte_offset + rows * 2;
            DenseQ4MmapMatvecProjection {
                tensor_name: name.to_string(),
                packed_byte_offset: packed_byte_offset as u64,
                scales_byte_offset: scales_byte_offset as u64,
                biases_byte_offset: biases_byte_offset as u64,
                rows,
                cols,
                output_width: rows,
                row_packed_bytes,
                groups_per_row: 1,
                group_size: cols,
                scale_bias_dtype: "BF16".to_string(),
            }
        }

        let heads = 2usize;
        let latent_rank = 16usize;
        let nope_dim = 8usize;
        let rope_dim = 4usize;
        let output_per_head = 16usize;
        let sequence = 2usize;
        let mut mmap = memmap2::MmapMut::map_anon(64 * 1024).unwrap();
        let mut offset = 0usize;
        let embed_q = write_projection(
            &mut mmap,
            &mut offset,
            "test.embed_q",
            heads * latent_rank,
            nope_dim,
            |row| (row % 3 + 1) as u8,
        );
        let unembed_out = write_projection(
            &mut mmap,
            &mut offset,
            "test.unembed_out",
            heads * output_per_head,
            latent_rank,
            |row| (row % 5 + 1) as u8,
        );
        let mmap = Arc::new(mmap.make_read_only().unwrap());
        let context = MetalExecutionContext::compile(mmap, 64 * 1024, &[], 1e-6).unwrap();
        let query_nope = [vec![1.0; nope_dim], vec![2.0; nope_dim]].concat();
        let query_rope = vec![0.25, -0.5, 0.75, 1.0, -0.25, 0.5, -0.75, 1.5];
        let record_latents = (0..sequence)
            .flat_map(|position| {
                (0..latent_rank).map(move |dim| {
                    if position == 0 {
                        (dim + 1) as f32 / latent_rank as f32
                    } else {
                        (latent_rank - dim) as f32 / latent_rank as f32
                    }
                })
            })
            .collect::<Vec<_>>();
        let record_rotary = vec![0.5, 0.25, -0.5, 1.0, -0.25, 0.75, 0.5, -1.0];
        let scale = (nope_dim + rope_dim) as f32;
        let scale = scale.sqrt().recip();

        let actual = context
            .resident_glm_mla_absorbed_attention(
                &embed_q,
                &unembed_out,
                MetalGlmMlaAbsorbedAttentionInput {
                    heads,
                    latent_rank,
                    query_nope: &query_nope,
                    query_rope: &query_rope,
                    record_latents: &record_latents,
                    record_rotary: &record_rotary,
                    sequence,
                    rope_dim,
                    scale,
                },
            )
            .unwrap()
            .unwrap();

        let mut expected = Vec::with_capacity(heads * output_per_head);
        for head in 0..heads {
            let query_sum = query_nope[head * nope_dim..(head + 1) * nope_dim]
                .iter()
                .sum::<f32>();
            let absorbed = (0..latent_rank)
                .map(|dim| ((head * latent_rank + dim) % 3 + 1) as f32 * query_sum)
                .collect::<Vec<_>>();
            let mut scores = (0..sequence)
                .map(|position| {
                    let latent =
                        &record_latents[position * latent_rank..(position + 1) * latent_rank];
                    let rotary = &record_rotary[position * rope_dim..(position + 1) * rope_dim];
                    let latent_score = absorbed
                        .iter()
                        .zip(latent)
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    let rotary_score = query_rope[head * rope_dim..(head + 1) * rope_dim]
                        .iter()
                        .zip(rotary)
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    (latent_score + rotary_score) * scale
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .sum::<f32>();
            for score in &mut scores {
                *score = (*score - maximum).exp() / denominator;
            }
            let context_values = (0..latent_rank)
                .map(|dim| {
                    (0..sequence)
                        .map(|position| {
                            scores[position] * record_latents[position * latent_rank + dim]
                        })
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            let context_sum = context_values.iter().sum::<f32>();
            expected.extend(
                (0..output_per_head)
                    .map(|row| ((head * output_per_head + row) % 5 + 1) as f32 * context_sum),
            );
        }
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-3,
                "output {index}: actual={actual} expected={expected}"
            );
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn resident_glm_mla_input_projection_chain_matches_reference() {
        fn write_q4_projection(
            mmap: &mut [u8],
            offset: &mut usize,
            name: &str,
            rows: usize,
            cols: usize,
            nibble: impl Fn(usize) -> u8,
        ) -> ResidentMmapMatvecProjection {
            let packed_byte_offset = *offset;
            let packed_bytes = rows * cols.div_ceil(2);
            for row in 0..rows {
                let value = nibble(row) & 0x0f;
                mmap[packed_byte_offset + row * cols.div_ceil(2)
                    ..packed_byte_offset + (row + 1) * cols.div_ceil(2)]
                    .fill(value | (value << 4));
            }
            let scales_byte_offset = packed_byte_offset + packed_bytes;
            for row in 0..rows {
                mmap[scales_byte_offset + row * 2..scales_byte_offset + row * 2 + 2]
                    .copy_from_slice(&0x3f80u16.to_le_bytes());
            }
            let biases_byte_offset = scales_byte_offset + rows * 2;
            mmap[biases_byte_offset..biases_byte_offset + rows * 2].fill(0);
            *offset = biases_byte_offset + rows * 2;
            ResidentMmapMatvecProjection::Q4(DenseQ4MmapMatvecProjection {
                tensor_name: name.to_string(),
                packed_byte_offset: packed_byte_offset as u64,
                scales_byte_offset: scales_byte_offset as u64,
                biases_byte_offset: biases_byte_offset as u64,
                rows,
                cols,
                output_width: rows,
                row_packed_bytes: cols.div_ceil(2),
                groups_per_row: 1,
                group_size: cols,
                scale_bias_dtype: "BF16".to_string(),
            })
        }

        let input = (1..=8).map(|value| value as f32).collect::<Vec<_>>();
        let mut mmap = memmap2::MmapMut::map_anon(16 * 1024).unwrap();
        let mut offset = 0usize;
        let q_a = write_q4_projection(&mut mmap, &mut offset, "q_a", 16, 8, |row| {
            (row % 7 + 1) as u8
        });
        let kv_a = write_q4_projection(&mut mmap, &mut offset, "kv_a", 16, 8, |row| {
            (row % 5 + 1) as u8
        });
        let q_b = write_q4_projection(&mut mmap, &mut offset, "q_b", 16, 16, |_| 1);
        let mmap = Arc::new(mmap.make_read_only().unwrap());
        let context = MetalExecutionContext::compile(mmap, 16 * 1024, &[], 1e-6).unwrap();
        let q_norm = vec![1.0; 16];
        let kv_norm = vec![1.0; 8];
        let (query, compressed) = context
            .resident_glm_mla_input_projection_chain(
                &q_a,
                &kv_a,
                &q_b,
                MetalBatchProjectionInput::Cpu(&input),
                &q_norm,
                &kv_norm,
                8,
                1e-6,
            )
            .unwrap()
            .unwrap();

        let input_sum = input.iter().sum::<f32>();
        let mut q_a_reference = (0..16)
            .map(|row| (row % 7 + 1) as f32 * input_sum)
            .collect::<Vec<_>>();
        let q_scale = (q_a_reference.iter().map(|value| value * value).sum::<f32>()
            / q_a_reference.len() as f32
            + 1e-6)
            .sqrt()
            .recip();
        for value in &mut q_a_reference {
            *value *= q_scale;
        }
        let expected_query = vec![q_a_reference.iter().sum::<f32>(); 16];
        for (actual, expected) in query.iter().zip(expected_query) {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }

        let mut expected_compressed = (0..16)
            .map(|row| (row % 5 + 1) as f32 * input_sum)
            .collect::<Vec<_>>();
        let kv_scale = (expected_compressed[..8]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / 8.0
            + 1e-6)
            .sqrt()
            .recip();
        for value in &mut expected_compressed[..8] {
            *value *= kv_scale;
        }
        for (actual, expected) in compressed.iter().zip(expected_compressed) {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn resident_glm_mla_fused_attention_matches_two_command_reference() {
        fn write_q4_projection(
            mmap: &mut [u8],
            offset: &mut usize,
            name: &str,
            rows: usize,
            cols: usize,
            nibble: impl Fn(usize) -> u8,
        ) -> DenseQ4MmapMatvecProjection {
            let packed_byte_offset = *offset;
            let packed_bytes = rows * cols.div_ceil(2);
            for row in 0..rows {
                let value = nibble(row) & 0x0f;
                mmap[packed_byte_offset + row * cols.div_ceil(2)
                    ..packed_byte_offset + (row + 1) * cols.div_ceil(2)]
                    .fill(value | (value << 4));
            }
            let scales_byte_offset = packed_byte_offset + packed_bytes;
            for row in 0..rows {
                mmap[scales_byte_offset + row * 2..scales_byte_offset + row * 2 + 2]
                    .copy_from_slice(&0x3f80u16.to_le_bytes());
            }
            let biases_byte_offset = scales_byte_offset + rows * 2;
            mmap[biases_byte_offset..biases_byte_offset + rows * 2].fill(0);
            *offset = biases_byte_offset + rows * 2;
            DenseQ4MmapMatvecProjection {
                tensor_name: name.to_string(),
                packed_byte_offset: packed_byte_offset as u64,
                scales_byte_offset: scales_byte_offset as u64,
                biases_byte_offset: biases_byte_offset as u64,
                rows,
                cols,
                output_width: rows,
                row_packed_bytes: cols.div_ceil(2),
                groups_per_row: 1,
                group_size: cols,
                scale_bias_dtype: "BF16".to_string(),
            }
        }

        let input = (1..=8).map(|value| value as f32).collect::<Vec<_>>();
        let heads = 2usize;
        let q_rank = 16usize;
        let latent_rank = 16usize;
        let nope_dim = 8usize;
        let rope_dim = 4usize;
        let output_per_head = 16usize;
        let mut mmap = memmap2::MmapMut::map_anon(64 * 1024).unwrap();
        let mut offset = 0usize;
        let q_a_q4 = write_q4_projection(&mut mmap, &mut offset, "q_a", q_rank, 8, |row| {
            (row % 7 + 1) as u8
        });
        let kv_a_q4 = write_q4_projection(
            &mut mmap,
            &mut offset,
            "kv_a",
            latent_rank + rope_dim,
            8,
            |row| (row % 5 + 1) as u8,
        );
        let q_b_q4 = write_q4_projection(
            &mut mmap,
            &mut offset,
            "q_b",
            heads * (nope_dim + rope_dim),
            q_rank,
            |row| (row % 3 + 1) as u8,
        );
        let embed_q = write_q4_projection(
            &mut mmap,
            &mut offset,
            "embed_q",
            heads * latent_rank,
            nope_dim,
            |row| (row % 3 + 1) as u8,
        );
        let unembed_out = write_q4_projection(
            &mut mmap,
            &mut offset,
            "unembed_out",
            heads * output_per_head,
            latent_rank,
            |row| (row % 5 + 1) as u8,
        );
        let out_proj = ResidentMmapMatvecProjection::Q4(write_q4_projection(
            &mut mmap,
            &mut offset,
            "o_proj",
            16,
            heads * output_per_head,
            |row| (row % 3 + 1) as u8,
        ));
        let router = ResidentMmapMatvecProjection::Q4(write_q4_projection(
            &mut mmap,
            &mut offset,
            "router",
            4,
            16,
            |row| (row % 4 + 1) as u8,
        ));
        let post_projections = Cmd2ResidentPostAttentionPrepProjections::new(
            3,
            out_proj,
            router,
            4,
            16,
            heads * output_per_head,
            2,
        )
        .unwrap();
        let q_a = ResidentMmapMatvecProjection::Q4(q_a_q4);
        let kv_a = ResidentMmapMatvecProjection::Q4(kv_a_q4);
        let q_b = ResidentMmapMatvecProjection::Q4(q_b_q4);
        let mmap = Arc::new(mmap.make_read_only().unwrap());
        let context = MetalExecutionContext::compile(mmap, 64 * 1024, &[], 1e-6).unwrap();
        let q_norm = vec![1.0; q_rank];
        let kv_norm = vec![1.0; latent_rank];
        let previous_latent = (0..latent_rank)
            .map(|dim| (dim + 1) as f32 / latent_rank as f32)
            .collect::<Vec<_>>();
        let previous_rotary = vec![0.5, -0.25, 0.75, -1.0];
        let rope_position = 3usize;
        let theta = 10_000.0f64;
        let mut rope_cos = Vec::new();
        let mut rope_sin = Vec::new();
        for pair in 0..rope_dim / 2 {
            let frequency = theta.powf(-((2 * pair) as f64) / rope_dim as f64);
            let (sin, cos) = (rope_position as f64 * frequency).sin_cos();
            rope_cos.push(cos as f32);
            rope_sin.push(sin as f32);
        }

        let (mut query, mut compressed) = context
            .resident_glm_mla_input_projection_chain(
                &q_a,
                &kv_a,
                &q_b,
                MetalBatchProjectionInput::Cpu(&input),
                &q_norm,
                &kv_norm,
                latent_rank,
                1e-6,
            )
            .unwrap()
            .unwrap();
        for head in 0..heads {
            let start = head * (nope_dim + rope_dim) + nope_dim;
            super::super::math::apply_rotary_interleaved_to_split_half(
                &mut query[start..start + rope_dim],
                rope_position,
                rope_dim,
                theta,
            )
            .unwrap();
        }
        let mut current_rotary = compressed.split_off(latent_rank);
        super::super::math::apply_rotary_interleaved_to_split_half(
            &mut current_rotary,
            rope_position,
            rope_dim,
            theta,
        )
        .unwrap();
        let mut query_nope = Vec::new();
        let mut query_rope = Vec::new();
        for head in 0..heads {
            let start = head * (nope_dim + rope_dim);
            query_nope.extend_from_slice(&query[start..start + nope_dim]);
            query_rope.extend_from_slice(&query[start + nope_dim..start + nope_dim + rope_dim]);
        }
        let record_latents = [previous_latent.clone(), compressed.clone()].concat();
        let record_rotary = [previous_rotary.clone(), current_rotary.clone()].concat();
        let scale = ((nope_dim + rope_dim) as f32).sqrt().recip();
        let reference = context
            .resident_glm_mla_absorbed_attention(
                &embed_q,
                &unembed_out,
                MetalGlmMlaAbsorbedAttentionInput {
                    heads,
                    latent_rank,
                    query_nope: &query_nope,
                    query_rope: &query_rope,
                    record_latents: &record_latents,
                    record_rotary: &record_rotary,
                    sequence: 2,
                    rope_dim,
                    scale,
                },
            )
            .unwrap()
            .unwrap();
        let fused = context
            .resident_glm_mla_fused_attention(
                &q_a,
                &kv_a,
                &q_b,
                &embed_q,
                &unembed_out,
                MetalGlmMlaFusedAttentionInput {
                    input: MetalBatchProjectionInput::Cpu(&input),
                    heads,
                    latent_rank,
                    nope_dim,
                    rope_dim,
                    previous_record_latents: &previous_latent,
                    previous_record_rotary: &previous_rotary,
                    rope_cos: &rope_cos,
                    rope_sin: &rope_sin,
                    scale,
                    post_attention: None,
                },
                &q_norm,
                &kv_norm,
                1e-6,
            )
            .unwrap()
            .unwrap();

        let MetalGlmMlaFusedAttentionTerminal::Attention(fused_attention) = &fused.terminal else {
            panic!("reference-only fused MLA test unexpectedly produced post-attention state")
        };
        for (index, (actual, expected)) in fused_attention.iter().zip(&reference).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-3,
                "attention {index}: actual={actual} expected={expected}"
            );
        }
        for (actual, expected) in fused.latent.iter().zip(&compressed) {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }
        for (actual, expected) in fused.rotary.iter().zip(&current_rotary) {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }

        let residual = (0..16)
            .map(|index| (index as f32 - 7.0) / 8.0)
            .collect::<Vec<_>>();
        let post_norm = vec![1.0; 16];
        let correction_bias = vec![0.0, 0.2, -0.1, 0.1];
        let reference_post = context
            .resident_post_attention_prep_topk(
                &post_projections,
                &reference,
                MetalBatchProjectionInput::Cpu(&residual),
                &post_norm,
                Some(&correction_bias),
            )
            .unwrap();
        let fused_post = context
            .resident_glm_mla_fused_attention(
                &q_a,
                &kv_a,
                &q_b,
                &embed_q,
                &unembed_out,
                MetalGlmMlaFusedAttentionInput {
                    input: MetalBatchProjectionInput::Cpu(&input),
                    heads,
                    latent_rank,
                    nope_dim,
                    rope_dim,
                    previous_record_latents: &previous_latent,
                    previous_record_rotary: &previous_rotary,
                    rope_cos: &rope_cos,
                    rope_sin: &rope_sin,
                    scale,
                    post_attention: Some(MetalGlmMlaPostAttentionInput {
                        projections: &post_projections,
                        residual: MetalBatchProjectionInput::Cpu(&residual),
                        post_norm_weight: &post_norm,
                        router_correction_bias: Some(&correction_bias),
                    }),
                },
                &q_norm,
                &kv_norm,
                1e-6,
            )
            .unwrap()
            .unwrap();
        let MetalGlmMlaFusedAttentionTerminal::PostAttention(fused_post) = fused_post.terminal
        else {
            panic!("fused MLA post-attention test did not produce post-attention state")
        };
        assert_eq!(fused_post.active, reference_post.active);
        unsafe {
            let reference_residual = read_f32_buffer(reference_post.residual_buffer, 16);
            let fused_residual = read_f32_buffer(fused_post.residual_buffer, 16);
            let reference_normed = read_f32_buffer(reference_post.normed_buffer, 16);
            let fused_normed = read_f32_buffer(fused_post.normed_buffer, 16);
            for (actual, expected) in fused_residual.iter().zip(reference_residual) {
                assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
            }
            for (actual, expected) in fused_normed.iter().zip(reference_normed) {
                assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
            }
            for buffer in [
                reference_post.residual_buffer,
                reference_post.normed_buffer,
                fused_post.residual_buffer,
                fused_post.normed_buffer,
            ] {
                context.buffers.recycle(buffer);
            }
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn recurrent_session_snapshot_round_trips_metal_recurrent_buffers() {
        let device = unsafe { OwnedMetalObject::new(metal_default_device()).unwrap() };
        let allocate = |len: usize, label: &str| {
            OwnedMetalObject::new(allocate_zeroed_buffer(device.id(), len * 4, label).unwrap())
                .unwrap()
        };
        let conv_state = allocate(2, "test recurrent conv state");
        let ssm_state = allocate(3, "test recurrent SSM state");
        let conv_output = allocate(4, "test recurrent conv output");
        let delta_output = allocate(5, "test recurrent delta output");
        let g_decay = allocate(2, "test recurrent decay");
        let beta_gate = allocate(2, "test recurrent beta");
        let resident =
            MetalLinearAttentionStateCache::new(vec![Some(MetalLinearAttentionLayerState::new(
                conv_state.id(),
                ssm_state.id(),
                conv_output.id(),
                delta_output.id(),
                g_decay.id(),
                beta_gate.id(),
                2,
                3,
                4,
                5,
                2,
            ))]);
        unsafe {
            write_f32_buffer(conv_state.id(), &[1.0, 2.0]);
            write_f32_buffer(ssm_state.id(), &[3.0, 4.0, 5.0]);
            write_f32_buffer(conv_output.id(), &[6.0; 4]);
            write_f32_buffer(delta_output.id(), &[7.0; 5]);
            write_f32_buffer(g_decay.id(), &[8.0; 2]);
            write_f32_buffer(beta_gate.id(), &[9.0; 2]);
        }

        let snapshot = capture_linear_attention_session_snapshot(&resident).unwrap();
        unsafe {
            write_f32_buffer(conv_state.id(), &[10.0; 2]);
            write_f32_buffer(ssm_state.id(), &[11.0; 3]);
        }
        restore_linear_attention_session_snapshot(&resident, &snapshot).unwrap();

        unsafe {
            assert_eq!(read_f32_buffer(conv_state.id(), 2), vec![1.0, 2.0]);
            assert_eq!(read_f32_buffer(ssm_state.id(), 3), vec![3.0, 4.0, 5.0]);
            assert_eq!(read_f32_buffer(conv_output.id(), 4), vec![0.0; 4]);
            assert_eq!(read_f32_buffer(delta_output.id(), 5), vec![0.0; 5]);
            assert_eq!(read_f32_buffer(g_decay.id(), 2), vec![0.0; 2]);
            assert_eq!(read_f32_buffer(beta_gate.id(), 2), vec![0.0; 2]);
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn reusable_metal_buffer_selection_is_best_fit_and_size_aware() {
        let id = std::ptr::null_mut();
        let buffers = vec![
            MetalReusableBuffer::new(id, 2_654_208),
            MetalReusableBuffer::new(id, 16),
            MetalReusableBuffer::new(id, 4_096),
            MetalReusableBuffer::new(id, 8_192),
        ];

        assert_eq!(best_fit_reusable_buffer_index(&buffers, 1), Some(1));
        assert_eq!(best_fit_reusable_buffer_index(&buffers, 64), Some(2));
        assert_eq!(best_fit_reusable_buffer_index(&buffers, 2_654_208), Some(0));
        assert_eq!(best_fit_reusable_buffer_index(&buffers, 2_654_209), None);
        assert_eq!(reusable_buffer_replacement_index(&buffers, 32), Some(1));
        assert_eq!(reusable_buffer_replacement_index(&buffers, 16), None);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_dispatch_plans_preserve_command_geometry() {
        assert_eq!(
            MetalDispatchPlan::threads(96),
            MetalDispatchPlan {
                mode: MetalDispatchMode::Threads,
                grid: MetalDispatchSize::new(96, 1, 1),
                threadgroup: MetalDispatchSize::new(64, 1, 1),
            }
        );
        assert_eq!(
            MetalDispatchPlan::q4_threadgroups(17),
            MetalDispatchPlan {
                mode: MetalDispatchMode::Threadgroups,
                grid: MetalDispatchSize::new(3, 1, 1),
                threadgroup: MetalDispatchSize::new(256, 1, 1),
            }
        );
        assert_eq!(
            MetalDispatchPlan::q4_mmap_threadgroups(17),
            MetalDispatchPlan {
                mode: MetalDispatchMode::Threadgroups,
                grid: MetalDispatchSize::new(2, 1, 1),
                threadgroup: MetalDispatchSize::new(256, 1, 1),
            }
        );
        assert_eq!(
            MetalDispatchPlan::q4_mmap_matrix_threadgroups(17, 64),
            MetalDispatchPlan {
                mode: MetalDispatchMode::Threadgroups,
                grid: MetalDispatchSize::new(2, 64, 1),
                threadgroup: MetalDispatchSize::new(256, 1, 1),
            }
        );
        assert_eq!(
            MetalDispatchPlan::qwen_attention_threadgroups(64, 16),
            MetalDispatchPlan {
                mode: MetalDispatchMode::Threadgroups,
                grid: MetalDispatchSize::new(64, 16, 1),
                threadgroup: MetalDispatchSize::new(256, 1, 1),
            }
        );
        assert_eq!(
            MetalDispatchPlan::single_threadgroup(512),
            MetalDispatchPlan {
                mode: MetalDispatchMode::Threadgroups,
                grid: MetalDispatchSize::new(1, 1, 1),
                threadgroup: MetalDispatchSize::new(256, 1, 1),
            }
        );
        assert_eq!(MetalDispatchPlan::q4_threadgroups(0).grid.width, 1);
        assert_eq!(
            MetalDispatchPlan::q4_mmap_matrix_threadgroups(0, 0).grid,
            MetalDispatchSize::new(1, 1, 1)
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_dense_weights_hold_buffer_len_and_mmap_owner() {
        let mmap = Arc::new(
            memmap2::MmapMut::map_anon(16)
                .unwrap()
                .make_read_only()
                .unwrap(),
        );
        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let dense = MetalDenseWeights::new(id, Arc::clone(&mmap), 16);

        assert_eq!(dense.buffer, id);
        assert_eq!(dense.len, 16);
        assert_eq!(Arc::strong_count(&mmap), 2);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn resident_topk_builder_rejects_invalid_bindings_before_encoding() {
        let mmap = Arc::new(
            memmap2::MmapMut::map_anon(128)
                .unwrap()
                .make_read_only()
                .unwrap(),
        );
        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let dense = MetalDenseWeights::new(id, mmap, 128);
        let buffers = MetalBufferPool::default();
        let pipelines = test_objc_pipeline_set(id);
        let builder = MetalResidentTopKBuilder::new(id, id, &pipelines, &dense, &buffers);

        let projection =
            ResidentMmapMatvecProjection::Q4(test_q4_projection("lm_head.weight", 4, 16));
        let input_error = builder.execute(&projection, &[0.0; 15], 4, 2).unwrap_err();
        assert!(
            input_error.to_string().contains("input len 15"),
            "{input_error:#}"
        );

        let mut unsupported_dtype = test_q4_projection("lm_head.weight", 4, 16);
        unsupported_dtype.scale_bias_dtype = "F16".to_string();
        let dtype_error = builder
            .execute(
                &ResidentMmapMatvecProjection::Q4(unsupported_dtype),
                &[0.0; 16],
                4,
                2,
            )
            .unwrap_err();
        assert!(
            dtype_error.to_string().contains("scale/bias dtype F16"),
            "{dtype_error:#}"
        );

        let mut out_of_range = test_q4_projection("lm_head.weight", 4, 16);
        out_of_range.biases_byte_offset = 124;
        let range_error = builder
            .execute(
                &ResidentMmapMatvecProjection::Q4(out_of_range),
                &[0.0; 16],
                4,
                2,
            )
            .unwrap_err();
        assert!(
            range_error
                .to_string()
                .contains("biases range for lm_head.weight exceeds resident dense weights"),
            "{range_error:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cmd2_resident_builder_rejects_state_width_mismatch_before_encoding() {
        let mmap = Arc::new(
            memmap2::MmapMut::map_anon(4096)
                .unwrap()
                .make_read_only()
                .unwrap(),
        );
        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let dense = MetalDenseWeights::new(id, mmap, 4096);
        let buffers = MetalBufferPool::default();
        let pipelines = test_objc_pipeline_set(id);
        let builder =
            MetalResidentPostAttentionPrepBuilder::new(id, id, &pipelines, &dense, &buffers, 1e-6);
        let projections = Cmd2ResidentPostAttentionPrepProjections::new(
            7,
            ResidentMmapMatvecProjection::Q4(test_q4_projection(
                "model.layers.7.self_attn.o_proj.weight",
                4,
                16,
            )),
            ResidentMmapMatvecProjection::Q4(test_q4_projection(
                "model.layers.7.mlp.gate.weight",
                8,
                4,
            )),
            8,
            4,
            16,
            4,
        )
        .unwrap();

        let error = builder
            .execute(
                &projections,
                &[0.0; 15],
                MetalBatchProjectionInput::Cpu(&[0.0; 4]),
                &[1.0; 4],
                None,
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("projection shapes out=4x16 rows=4"),
            "{error:#}"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_reusable_buffer_records_pool_entry_shape() {
        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
        let buffer = MetalReusableBuffer::new(id, 4096);

        assert_eq!(buffer.id, id);
        assert_eq!(buffer.len, 4096);
        assert_eq!(METAL_REUSABLE_BUFFER_POOL_LIMIT, 64);
        assert_eq!(METAL_REUSABLE_EXPERT_STAGING_POOL_LIMIT, 16);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn metal_expert_staging_pool_reuses_completed_slot_allocation() {
        objc2::rc::autoreleasepool(|_| unsafe {
            let runtime =
                MetalRuntime::compile(METAL_SHADERS, MetalPipelineNameSet::new()).unwrap();
            let resources = Arc::new(MetalResourceLedger::from_device(runtime.device));
            let buffers = MetalBufferPool::new(Arc::clone(&resources));
            let bytes = vec![0x5au8; 64 * 1024];

            let first = buffers
                .transient_expert_buffer_with_bytes(runtime.device, &bytes)
                .unwrap();
            buffers
                .recycle_or_release_phase(vec![MetalPhaseBuffer::transient_expert(first)], false);
            let pooled = resources.snapshot();
            assert_eq!(pooled.transient_expert_buffers, 0);
            assert_eq!(pooled.pooled_buffers, 1);

            let second = buffers
                .transient_expert_buffer_with_bytes(runtime.device, &bytes)
                .unwrap();
            assert_eq!(second, first);
            let checked_out = resources.snapshot();
            assert_eq!(checked_out.transient_expert_buffers, 1);
            assert_eq!(checked_out.pooled_buffers, 0);

            buffers
                .recycle_or_release_phase(vec![MetalPhaseBuffer::transient_expert(second)], true);
            let released = resources.snapshot();
            assert_eq!(released.transient_expert_buffers, 0);
            assert_eq!(released.pooled_buffers, 0);
        });
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_expert_source_buffer_cache_reuses_same_fixed_payload_key() {
        let first = [1u8; 16];
        let second = [2u8; 16];
        let buffer = 0x1000usize as MetalObjcId;
        let mut cache = MetalExpertSourceBufferCache::default();

        cache.insert(&first, buffer);

        assert_eq!(cache.get(&first), Some(buffer));
        assert_eq!(cache.get(&second), None);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn phase_buffer_tracks_recycling_class() {
        let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();

        let recyclable = MetalPhaseBuffer::recyclable(id);
        assert_eq!(recyclable.id, id);
        assert_eq!(recyclable.class, MetalPhaseBufferClass::General);

        let expert = MetalPhaseBuffer::transient_expert(id);
        assert_eq!(expert.id, id);
        assert_eq!(expert.class, MetalPhaseBufferClass::TransientExpert);

        let borrowed = MetalPhaseBuffer::borrowed_expert(id);
        assert_eq!(borrowed.id, id);
        assert_eq!(borrowed.class, MetalPhaseBufferClass::BorrowedExpert);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn metal_wraps_page_aligned_expert_slot_without_copying() {
        objc2::rc::autoreleasepool(|_| unsafe {
            let device = metal_default_device();
            assert!(!device.is_null());
            let page_size = metal_page_size();
            let mut bytes = memmap2::MmapMut::map_anon(page_size).unwrap();
            bytes.fill(0x5a);

            let buffer = wrap_expert_slot_as_metal_buffer(device, &bytes).unwrap();

            assert_eq!(
                msg_send_ptr0(buffer, sel("contents")).cast::<u8>(),
                bytes.as_mut_ptr()
            );
            assert_eq!(msg_send_usize0(buffer, sel("length")), page_size);
            release(buffer);
            release(device);
        });
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn metal_reuses_wrapper_attached_to_page_aligned_expert_slot() {
        objc2::rc::autoreleasepool(|_| unsafe {
            let device = metal_default_device();
            assert!(!device.is_null());
            let resources = Arc::new(MetalResourceLedger::from_device(device));
            let buffers = MetalBufferPool::new(Arc::clone(&resources));
            let page_size = metal_page_size();
            let mut scratch = super::super::experts::ReusableExpertBuffer::default();
            scratch
                .prepare_payload(page_size, page_size)
                .unwrap()
                .fill(0x5a);
            let bytes = scratch.take_payload();

            let first = persistent_expert_source_buffer(device, &bytes, &bytes, &buffers)
                .unwrap()
                .unwrap();
            let second = persistent_expert_source_buffer(device, &bytes, &bytes, &buffers)
                .unwrap()
                .unwrap();

            assert_eq!(second, first);
            assert_eq!(
                msg_send_ptr0(first, sel("contents")).cast::<u8>(),
                bytes.as_ptr() as *mut u8
            );
            assert_eq!(msg_send_usize0(first, sel("length")), page_size);
            let attached = resources.snapshot();
            assert_eq!(attached.resident_expert_wrapper_buffers, 1);
            assert_eq!(attached.resident_expert_wrapper_bytes, page_size);
            assert_eq!(attached.active_general_buffers, 0);
            drop(bytes);
            let released = resources.snapshot();
            assert_eq!(released.resident_expert_wrapper_buffers, 0);
            assert_eq!(released.resident_expert_wrapper_bytes, 0);
            assert_eq!(released.buffer_releases, 1);
            drop(buffers);
            release(device);
        });
    }
}
