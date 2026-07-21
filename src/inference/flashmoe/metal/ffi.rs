use super::*;

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
    pub(super) fn objc_msgSend();
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
pub(super) fn class(name: &str) -> MetalObjcId {
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
pub(super) unsafe fn new_function(library: MetalObjcId, name: &str) -> anyhow::Result<MetalObjcId> {
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
pub(super) unsafe fn compile_pipeline_with_constants(
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
pub(super) unsafe fn new_compute_pipeline(
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
pub(super) fn metal_page_size() -> usize {
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
pub(super) unsafe fn wrap_expert_slot_as_metal_buffer(
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
pub(super) unsafe fn persistent_expert_source_buffer(
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
        Ok(Some(MetalDenseWeights::new(buffer, mmap, len)?))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn allocate_zeroed_buffer(
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
pub(super) unsafe fn zero_buffer(buffer: MetalObjcId, f32_len: usize) {
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
pub(super) unsafe fn release_linear_attention_layer(layer: &MetalLinearAttentionLayerState) {
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
pub(super) unsafe fn release_linear_attention_state(state: &mut MetalLinearAttentionStateCache) {
    unsafe {
        for layer in state.layers.iter_mut().filter_map(Option::take) {
            release_linear_attention_layer(&layer);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn linear_attention_state_bytes(state: &MetalLinearAttentionStateCache) -> usize {
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
pub(super) fn allocate_linear_attention_state(
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
pub(super) unsafe fn read_f32_buffer_offset(
    buffer: MetalObjcId,
    offset: usize,
    len: usize,
) -> Vec<f32> {
    unsafe {
        let contents = msg_send_ptr0(buffer, sel("contents")).cast::<f32>();
        let mut output = vec![0.0f32; len];
        ptr::copy_nonoverlapping(contents.add(offset), output.as_mut_ptr(), len);
        output
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) unsafe fn write_f32_buffer(buffer: MetalObjcId, values: &[f32]) {
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
pub(crate) unsafe fn dispatch_q4_mmap_matrix_bf16_threadgroups(
    encoder: MetalObjcId,
    rows: u64,
    input_rows: u64,
    input_rows_per_threadgroup: u64,
) {
    unsafe {
        dispatch_metal_plan(
            encoder,
            MetalDispatchPlan::q4_mmap_matrix_bf16_threadgroups(
                rows,
                input_rows,
                input_rows_per_threadgroup,
            ),
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
pub(super) unsafe fn set_metal_command_buffer_label(
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
pub(super) unsafe fn metal_command_buffer_status(
    command_buffer: MetalObjcId,
) -> MetalCommandStatus {
    unsafe { MetalCommandStatus::from_raw(msg_send_usize0(command_buffer, sel("status"))) }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) unsafe fn metal_command_buffer_error(command_buffer: MetalObjcId) -> Option<String> {
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
pub(super) unsafe fn purge_and_release_metal_buffer(buffer: MetalObjcId) {
    unsafe {
        // MTLResourcePurgeableStateEmpty. Expert staging is never reused after its command;
        // marking it empty tells IOAccelerator to discard dirty backing and per-binding mappings.
        let _ = msg_send_usize1_usize(buffer, sel("setPurgeableState:"), 4);
        release(buffer);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) unsafe fn retain_autoreleased_return_value(receiver: MetalObjcId) -> MetalObjcId {
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
pub(super) unsafe fn msg_send_void1_bool(
    receiver: MetalObjcId,
    selector: MetalSelector,
    arg: bool,
) {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector, bool) =
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
pub(super) unsafe fn msg_send_void3_ptr_u64_u64(
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
pub(super) unsafe fn msg_send_const_char_ptr0(
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
pub(super) unsafe fn msg_send_usize1_usize(
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
