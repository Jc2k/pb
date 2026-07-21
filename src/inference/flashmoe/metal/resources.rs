use super::*;

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
pub(super) enum MetalTrackedBufferClass {
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
pub(super) struct MetalResourceCounter {
    count: usize,
    bytes: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalResourceCounter {
    pub(super) fn add(&mut self, bytes: usize) {
        self.count = self.count.saturating_add(1);
        self.bytes = self.bytes.saturating_add(bytes);
    }

    pub(super) fn remove(&mut self, bytes: usize) {
        self.count = self.count.saturating_sub(1);
        self.bytes = self.bytes.saturating_sub(bytes);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug)]
pub(super) struct MetalResourceLedgerState {
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
    pub(super) fn new(
        recommended_working_set_bytes: usize,
        current_allocated_bytes: usize,
    ) -> Self {
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

    pub(super) fn ledger_live_bytes(&self) -> usize {
        self.resident_dense_bytes
            .saturating_add(self.recurrent_state_bytes)
            .saturating_add(self.active_general.bytes)
            .saturating_add(self.pooled.bytes)
            .saturating_add(self.transient_expert.bytes)
            .saturating_add(self.resident_expert_wrapper.bytes)
    }

    pub(super) fn update_ledger_high_water(&mut self) {
        self.ledger_high_water_bytes = self.ledger_high_water_bytes.max(self.ledger_live_bytes());
    }

    pub(super) fn counter_mut(
        &mut self,
        class: MetalTrackedBufferClass,
    ) -> &mut MetalResourceCounter {
        match class {
            MetalTrackedBufferClass::ResidentExpertWrapper => &mut self.resident_expert_wrapper,
            MetalTrackedBufferClass::ActiveGeneral => &mut self.active_general,
            MetalTrackedBufferClass::Pooled => &mut self.pooled,
            MetalTrackedBufferClass::TransientExpert => &mut self.transient_expert,
        }
    }

    pub(super) fn register_buffer(
        &mut self,
        id: usize,
        len: usize,
        class: MetalTrackedBufferClass,
    ) {
        debug_assert!(!self.buffers.contains_key(&id));
        if let Some(previous) = self.buffers.insert(id, MetalTrackedBuffer { len, class }) {
            self.counter_mut(previous.class).remove(previous.len);
        }
        self.counter_mut(class).add(len);
        self.buffer_allocations = self.buffer_allocations.saturating_add(1);
        self.update_ledger_high_water();
    }

    pub(super) fn transition_buffer(&mut self, id: usize, class: MetalTrackedBufferClass) {
        if !self.transition_buffer_if_tracked(id, class) {
            debug_assert!(false, "untracked Metal buffer transition: {id:#x}");
        }
    }

    fn transition_buffer_if_tracked(&mut self, id: usize, class: MetalTrackedBufferClass) -> bool {
        let Some(previous) = self.buffers.get(&id).copied() else {
            return false;
        };
        if previous.class == class {
            return true;
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
        true
    }

    pub(super) fn release_buffer(&mut self, id: usize) {
        if !self.release_buffer_if_tracked(id) {
            debug_assert!(false, "untracked Metal buffer release: {id:#x}");
        }
    }

    fn release_buffer_if_tracked(&mut self, id: usize) -> bool {
        let Some(previous) = self.buffers.remove(&id) else {
            return false;
        };
        self.counter_mut(previous.class).remove(previous.len);
        self.buffer_releases = self.buffer_releases.saturating_add(1);
        true
    }

    pub(super) fn snapshot(&self) -> FlashMoeMetalResourceSnapshot {
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
pub(super) fn default_metal_working_set_limit(recommended: usize) -> usize {
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
    pub(super) state: Mutex<MetalResourceLedgerState>,
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
    pub(super) unsafe fn from_device(device: MetalObjcId) -> Self {
        unsafe {
            Self {
                state: Mutex::new(MetalResourceLedgerState::new(
                    msg_send_usize0(device, sel("recommendedMaxWorkingSetSize")),
                    msg_send_usize0(device, sel("currentAllocatedSize")),
                )),
            }
        }
    }

    pub(super) fn register_buffer(
        &self,
        id: MetalObjcId,
        len: usize,
        class: MetalTrackedBufferClass,
    ) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.register_buffer(id as usize, len, class);
        state.unobserved_allocated_bytes = state.unobserved_allocated_bytes.saturating_add(len);
    }

    pub(super) fn transition_buffer(&self, id: MetalObjcId, class: MetalTrackedBufferClass) {
        self.state
            .lock()
            .expect("Metal resource ledger poisoned")
            .transition_buffer(id as usize, class);
    }

    pub(super) fn release_buffer(&self, id: MetalObjcId) {
        self.state
            .lock()
            .expect("Metal resource ledger poisoned")
            .release_buffer(id as usize);
    }

    pub(super) fn release_buffer_on_drop(&self, id: MetalObjcId) {
        super::synchronization::lock_for_drop(&self.state).release_buffer_if_tracked(id as usize);
    }

    pub(super) fn transition_buffer_on_drop(
        &self,
        id: MetalObjcId,
        class: MetalTrackedBufferClass,
    ) -> bool {
        super::synchronization::lock_for_drop(&self.state)
            .transition_buffer_if_tracked(id as usize, class)
    }

    pub(super) fn record_resident_resources(&self, dense_bytes: usize, recurrent_bytes: usize) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.resident_dense_bytes = dense_bytes;
        state.recurrent_state_bytes = recurrent_bytes;
        state.update_ledger_high_water();
    }

    pub(super) fn clear_resident_resources_on_drop(&self) {
        let mut state = super::synchronization::lock_for_drop(&self.state);
        state.resident_dense_bytes = 0;
        state.recurrent_state_bytes = 0;
        state.update_ledger_high_water();
    }

    pub(super) fn set_working_set_limit_bytes(&self, requested: usize) -> anyhow::Result<()> {
        if requested == 0 {
            anyhow::bail!("FlashMoe Metal working-set limit must be greater than zero");
        }
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        let default_limit = default_metal_working_set_limit(state.recommended_working_set_bytes);
        state.working_set_limit_bytes = requested.min(default_limit);
        Ok(())
    }

    pub(super) unsafe fn sample_device(&self, device: MetalObjcId, token_boundary: bool) -> usize {
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

    pub(super) fn allocation_would_exceed_limit(&self, current: usize, requested: usize) -> bool {
        let state = self.state.lock().expect("Metal resource ledger poisoned");
        current.saturating_add(requested) > state.working_set_limit_bytes
    }

    pub(super) fn estimated_allocation_would_exceed_limit(&self, requested: usize) -> bool {
        let state = self.state.lock().expect("Metal resource ledger poisoned");
        state
            .current_allocated_bytes
            .saturating_add(state.unobserved_allocated_bytes)
            .saturating_add(requested)
            > state.working_set_limit_bytes
    }

    pub(super) fn record_pressure_recovery(&self) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.pressure_recoveries = state.pressure_recoveries.saturating_add(1);
    }

    pub(super) fn record_resource_limit_abort(&self) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.resource_limit_aborts = state.resource_limit_aborts.saturating_add(1);
    }

    pub(super) fn record_phase_cleanup(&self, buffers: usize) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.phase_cleanup_calls = state.phase_cleanup_calls.saturating_add(1);
        state.phase_cleanup_buffers = state.phase_cleanup_buffers.saturating_add(buffers);
    }

    pub(super) fn record_phase_cleanup_on_drop(&self, buffers: usize) {
        let mut state = super::synchronization::lock_for_drop(&self.state);
        state.phase_cleanup_calls = state.phase_cleanup_calls.saturating_add(1);
        state.phase_cleanup_buffers = state.phase_cleanup_buffers.saturating_add(buffers);
    }

    pub(super) fn command_started(&self) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.command_submissions = state.command_submissions.saturating_add(1);
        state.in_flight_commands = state.in_flight_commands.saturating_add(1);
        state.command_high_water = state.command_high_water.max(state.in_flight_commands);
    }

    pub(super) fn record_host_upload(&self, bytes: usize) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.host_upload_bytes = state.host_upload_bytes.saturating_add(bytes);
    }

    pub(super) fn record_host_readback(&self, bytes: usize) {
        let mut state = self.state.lock().expect("Metal resource ledger poisoned");
        state.host_readback_bytes = state.host_readback_bytes.saturating_add(bytes);
    }

    pub(super) fn command_finished_on_drop(&self) {
        let mut state = super::synchronization::lock_for_drop(&self.state);
        state.in_flight_commands = state.in_flight_commands.saturating_sub(1);
    }

    pub(super) fn snapshot(&self) -> FlashMoeMetalResourceSnapshot {
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
// SAFETY: `id` is a retained MTLBuffer handle, which is not thread-affine. The
// pool transfers only the handle and its immutable length between threads; all
// ownership transitions are serialized by `MetalBufferPool`'s mutexes, and
// command completion is observed before a buffer is recycled.
unsafe impl Send for MetalReusableBuffer {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn best_fit_reusable_buffer_index(
    buffers: &[MetalReusableBuffer],
    len: usize,
) -> Option<usize> {
    buffers
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.len >= len)
        .min_by_key(|(_, buffer)| buffer.len)
        .map(|(index, _)| index)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn reusable_buffer_replacement_index(
    buffers: &[MetalReusableBuffer],
    len: usize,
) -> Option<usize> {
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
    pub(super) resources: Arc<MetalResourceLedger>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Default for MetalBufferPool {
    fn default() -> Self {
        Self::new(Arc::new(MetalResourceLedger::default()))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalBufferPool {
    pub(super) fn new(resources: Arc<MetalResourceLedger>) -> Self {
        Self {
            reusable: Mutex::new(Vec::new()),
            reusable_expert_staging: Mutex::new(Vec::new()),
            resources,
        }
    }

    pub(super) fn resources(&self) -> &Arc<MetalResourceLedger> {
        &self.resources
    }

    pub(crate) unsafe fn buffer_with_len(
        &self,
        device: MetalObjcId,
        len: usize,
    ) -> anyhow::Result<MetalObjcId> {
        unsafe { self.buffer_with_len_class(device, len, MetalTrackedBufferClass::ActiveGeneral) }
    }

    pub(super) unsafe fn buffer_with_len_class(
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

    pub(super) unsafe fn ensure_allocation_capacity(
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

    pub(super) fn release_idle_buffers(&self) -> (usize, usize) {
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

    pub(super) unsafe fn tracked_buffer_with_len(
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

    pub(super) unsafe fn tracked_buffer_with_bytes(
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

    pub(super) unsafe fn read_f32_buffer(&self, buffer: MetalObjcId, len: usize) -> Vec<f32> {
        let bytes = len.saturating_mul(std::mem::size_of::<f32>());
        self.resources.record_host_readback(bytes);
        unsafe { read_f32_buffer(buffer, len) }
    }

    pub(super) unsafe fn read_f32_buffer_offset(
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

    pub(super) unsafe fn recycle_expert_staging(&self, buffer: MetalObjcId) {
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

    unsafe fn recycle_into_pool_on_drop<const LIMIT: usize, F>(
        &self,
        pool: &Mutex<Vec<MetalReusableBuffer>>,
        buffer: MetalObjcId,
        release_buffer: F,
    ) where
        F: Fn(MetalObjcId) + Copy,
    {
        let len = unsafe { msg_send_usize0(buffer, sel("length")) };
        let mut reusable = match pool.lock() {
            Ok(reusable) => reusable,
            Err(_) => {
                self.resources.release_buffer_on_drop(buffer);
                release_buffer(buffer);
                return;
            }
        };
        if reusable.len() < LIMIT {
            if !self
                .resources
                .transition_buffer_on_drop(buffer, MetalTrackedBufferClass::Pooled)
            {
                drop(reusable);
                release_buffer(buffer);
                return;
            }
            reusable.push(MetalReusableBuffer::new(buffer, len));
            return;
        }
        let Some(index) = reusable_buffer_replacement_index(&reusable, len) else {
            drop(reusable);
            self.resources.release_buffer_on_drop(buffer);
            release_buffer(buffer);
            return;
        };
        if !self
            .resources
            .transition_buffer_on_drop(buffer, MetalTrackedBufferClass::Pooled)
        {
            drop(reusable);
            release_buffer(buffer);
            return;
        }
        let evicted =
            std::mem::replace(&mut reusable[index], MetalReusableBuffer::new(buffer, len));
        drop(reusable);
        self.resources.release_buffer_on_drop(evicted.id);
        release_buffer(evicted.id);
    }

    unsafe fn recycle_on_drop(&self, buffer: MetalObjcId) {
        unsafe {
            self.recycle_into_pool_on_drop::<METAL_REUSABLE_BUFFER_POOL_LIMIT, _>(
                &self.reusable,
                buffer,
                |buffer| release(buffer),
            );
        }
    }

    unsafe fn recycle_expert_staging_on_drop(&self, buffer: MetalObjcId) {
        unsafe {
            self.recycle_into_pool_on_drop::<METAL_REUSABLE_EXPERT_STAGING_POOL_LIMIT, _>(
                &self.reusable_expert_staging,
                buffer,
                |buffer| purge_and_release_metal_buffer(buffer),
            );
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

    pub(crate) fn recycle_buffers_on_drop(&self, buffers: &[MetalObjcId]) {
        unsafe {
            for buffer in buffers.iter().copied() {
                self.recycle_on_drop(buffer);
            }
        }
    }

    pub(crate) fn recycle_or_release_phase_on_drop(
        &self,
        buffers: Vec<MetalPhaseBuffer>,
        release_only: bool,
    ) {
        self.resources.record_phase_cleanup_on_drop(buffers.len());
        unsafe {
            for buffer in buffers {
                match buffer.class {
                    MetalPhaseBufferClass::BorrowedExpert => release(buffer.id),
                    MetalPhaseBufferClass::TransientExpert => {
                        if release_only {
                            self.resources.release_buffer_on_drop(buffer.id);
                            purge_and_release_metal_buffer(buffer.id);
                        } else {
                            self.recycle_expert_staging_on_drop(buffer.id);
                        }
                    }
                    MetalPhaseBufferClass::General => {
                        if release_only {
                            self.resources.release_buffer_on_drop(buffer.id);
                            release(buffer.id);
                        } else {
                            self.recycle_on_drop(buffer.id);
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn release_all(&mut self) {
        let reusable = super::synchronization::get_mut_for_drop(&mut self.reusable);
        unsafe {
            for buffer in reusable.drain(..) {
                self.resources.release_buffer_on_drop(buffer.id);
                release(buffer.id);
            }
        }
        let expert_staging =
            super::synchronization::get_mut_for_drop(&mut self.reusable_expert_staging);
        unsafe {
            for buffer in expert_staging.drain(..) {
                self.resources.release_buffer_on_drop(buffer.id);
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

#[cfg(test)]
mod poison_tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;

    fn poison<T>(mutex: &Mutex<T>) {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _guard = mutex.lock().expect("test mutex should start healthy");
            panic!("intentional cleanup poison");
        }));
        assert!(result.is_err());
    }

    #[test]
    fn metal_buffer_pool_drop_recovers_poisoned_empty_pools() {
        let pool = MetalBufferPool::default();
        poison(&pool.reusable);
        poison(&pool.reusable_expert_staging);

        assert!(catch_unwind(AssertUnwindSafe(|| drop(pool))).is_ok());
    }

    #[test]
    fn metal_resource_drop_accounting_recovers_poisoned_ledger() {
        let resources = MetalResourceLedger::default();
        let buffer = 0x1000usize as MetalObjcId;
        resources.register_buffer(buffer, 64, MetalTrackedBufferClass::ActiveGeneral);
        resources.command_started();
        poison(&resources.state);

        assert!(resources.transition_buffer_on_drop(buffer, MetalTrackedBufferClass::Pooled));
        resources.command_finished_on_drop();
        resources.clear_resident_resources_on_drop();

        {
            let snapshot =
                super::super::synchronization::lock_for_drop(&resources.state).snapshot();
            assert_eq!(snapshot.pooled_buffers, 1);
        }
        resources.release_buffer_on_drop(buffer);
        let snapshot = super::super::synchronization::lock_for_drop(&resources.state).snapshot();
        assert_eq!(snapshot.ledger_live_bytes, 0);
        assert_eq!(snapshot.active_general_buffers, 0);
        assert_eq!(snapshot.in_flight_commands, 0);
    }
}
