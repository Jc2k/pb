use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::sync::Arc;

use super::{
    MetalObjcId, MetalResourceLedger, msg_send_id0, msg_send_void0, release,
    retain_autoreleased_return_value, sel,
};

#[derive(Debug)]
pub(super) struct MetalObject;

#[derive(Debug)]
pub(super) struct MetalDevice;

#[derive(Debug)]
pub(super) struct MetalCommandQueue;

#[derive(Debug)]
pub(super) struct MetalPipelineState;

#[derive(Debug)]
pub(super) struct MetalBuffer;

#[repr(transparent)]
#[derive(Debug)]
pub(super) struct RetainedMetalObject<K> {
    id: NonNull<c_void>,
    _kind: PhantomData<K>,
}

pub(super) type OwnedMetalObject = RetainedMetalObject<MetalObject>;

impl<K> RetainedMetalObject<K> {
    /// # Safety
    ///
    /// `id` must be either null or a valid +1 retained Objective-C object.
    /// A successful value transfers that retain and sends `release` on drop.
    pub(super) unsafe fn new(id: MetalObjcId) -> anyhow::Result<Self> {
        let id = NonNull::new(id)
            .ok_or_else(|| anyhow::anyhow!("failed to create required Flash-MoE Metal object"))?;
        Ok(Self {
            id,
            _kind: PhantomData,
        })
    }

    pub(super) fn id(&self) -> MetalObjcId {
        self.id.as_ptr()
    }

    #[cfg(test)]
    pub(super) fn into_raw(self) -> MetalObjcId {
        let object = std::mem::ManuallyDrop::new(self);
        object.id()
    }
}

pub(super) fn keep_retained<K>(
    object: RetainedMetalObject<K>,
    owners: &mut Vec<RetainedMetalObject<K>>,
) -> MetalObjcId {
    let id = object.id();
    owners.push(object);
    id
}

impl<K> Drop for RetainedMetalObject<K> {
    fn drop(&mut self) {
        unsafe { release(self.id()) }
    }
}

#[derive(Debug)]
pub(super) struct MetalCommandLease {
    resources: Arc<MetalResourceLedger>,
}

impl MetalCommandLease {
    pub(super) fn new(resources: Arc<MetalResourceLedger>) -> Self {
        resources.command_started();
        Self { resources }
    }
}

impl Drop for MetalCommandLease {
    fn drop(&mut self) {
        self.resources.command_finished_on_drop();
    }
}

#[derive(Debug)]
pub(super) struct MetalCommandEncoding {
    command_buffer: MetalObjcId,
    encoder: MetalObjcId,
    ended: bool,
    command_lease: Option<MetalCommandLease>,
}

impl MetalCommandEncoding {
    pub(super) unsafe fn new(
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

    pub(super) fn command_buffer(&self) -> MetalObjcId {
        self.command_buffer
    }

    pub(super) fn encoder(&self) -> MetalObjcId {
        self.encoder
    }

    pub(super) unsafe fn end_encoding(&mut self) {
        unsafe {
            if !self.ended {
                msg_send_void0(self.encoder, sel("endEncoding"));
                self.ended = true;
            }
        }
    }

    pub(super) unsafe fn into_command_buffer(mut self) -> (MetalObjcId, MetalCommandLease) {
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
