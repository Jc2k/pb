use anyhow::{Context, Result, bail};
use std::sync::{Arc, Mutex};

use super::model_family::{
    QwenMoeExpertComponentKind, QwenMoeExpertComponentLayout, QwenMoeQ4ExpertLayout,
};
use super::types::ACTIVE_EXPERTS_PER_TOKEN;

pub type ReusableExpertBytePool = Arc<Mutex<Vec<Vec<u8>>>>;

const FIXED_Q4_EXPERT_BUFFER_POOL_LIMIT: usize = ACTIVE_EXPERTS_PER_TOKEN * 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertReadPath {
    PositionedRead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertIoPolicy {
    pub expert_read_path: ExpertReadPath,
    pub application_expert_cache: bool,
    pub lz4_expert_compression: bool,
    pub speculative_routing: bool,
    pub broad_ssd_gpu_overlap: bool,
}

// Expert scheduler policy guardrails:
// - read packed experts with positioned reads, not mmap;
// - do not add an application-level expert LRU/cache;
// - do not add LZ4 expert compression;
// - do not speculate future expert routes;
// - avoid broad SSD/GPU overlap beyond the existing narrow deferred expert phase.
//
// These choices follow Flash-MoE's "Trust the OS" result: the OS page cache plus
// parallel pread won over custom expert caches, mmap expert files, LZ4, prefetch
// hints, speculative routing, dispatch_io, and aggressive SSD/GPU overlap.
// See https://github.com/danveloper/flash-moe, especially the README "Trust the
// OS" notes and docs/optimization-experiments-q4.md.
pub const FLASHMOE_EXPERT_IO_POLICY: ExpertIoPolicy = ExpertIoPolicy {
    expert_read_path: ExpertReadPath::PositionedRead,
    application_expert_cache: false,
    lz4_expert_compression: false,
    speculative_routing: false,
    broad_ssd_gpu_overlap: false,
};

pub fn take_reusable_expert_bytes(
    pool: &ReusableExpertBytePool,
    min_capacity: usize,
) -> Option<Vec<u8>> {
    let mut pool = pool.lock().expect("fixed Q4 expert byte pool poisoned");
    let index = pool
        .iter()
        .position(|bytes| bytes.capacity() >= min_capacity)?;
    Some(pool.swap_remove(index))
}

pub fn recycle_reusable_expert_bytes(
    pool: &ReusableExpertBytePool,
    mut bytes: Vec<u8>,
    min_capacity: usize,
) {
    if bytes.capacity() < min_capacity {
        return;
    }
    bytes.clear();
    let mut pool = pool.lock().expect("fixed Q4 expert byte pool poisoned");
    if pool.len() < FIXED_Q4_EXPERT_BUFFER_POOL_LIMIT {
        pool.push(bytes);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlotDescriptor {
    pub layer: usize,
    pub expert: usize,
    pub slot_offset: u64,
    pub slot_capacity: usize,
    pub payload_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlotView<'a> {
    descriptor: ExpertSlotDescriptor,
    payload: &'a [u8],
}

impl<'a> ExpertSlotView<'a> {
    pub fn new(
        layer: usize,
        expert: usize,
        slot_offset: u64,
        slot_capacity: usize,
        payload: &'a [u8],
    ) -> Result<Self> {
        if payload.len() > slot_capacity {
            bail!(
                "expert slot payload length {} exceeds slot capacity {}",
                payload.len(),
                slot_capacity
            );
        }
        Ok(Self {
            descriptor: ExpertSlotDescriptor {
                layer,
                expert,
                slot_offset,
                slot_capacity,
                payload_len: payload.len(),
            },
            payload,
        })
    }

    pub fn descriptor(&self) -> ExpertSlotDescriptor {
        self.descriptor
    }

    pub fn payload(&self) -> &'a [u8] {
        self.payload
    }

    pub fn payload_prefix(&self, max_len: usize) -> &'a [u8] {
        &self.payload[..self.payload.len().min(max_len)]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedQ4ExpertSlotView<'a> {
    slot: ExpertSlotView<'a>,
    layout: QwenMoeQ4ExpertLayout,
}

impl<'a> FixedQ4ExpertSlotView<'a> {
    pub fn new(slot: ExpertSlotView<'a>, layout: QwenMoeQ4ExpertLayout) -> Result<Self> {
        layout.validate()?;
        let payload_len = slot.payload().len();
        if payload_len < layout.expert_bytes {
            bail!(
                "fixed Q4 expert slot payload length {payload_len} is shorter than layout size {}",
                layout.expert_bytes
            );
        }
        Ok(Self { slot, layout })
    }

    pub fn descriptor(&self) -> ExpertSlotDescriptor {
        self.slot.descriptor()
    }

    pub fn layout(&self) -> QwenMoeQ4ExpertLayout {
        self.layout
    }

    pub fn payload(&self) -> &'a [u8] {
        self.slot.payload()
    }

    pub fn component(&self, kind: QwenMoeExpertComponentKind) -> &'a [u8] {
        let component = self.layout.component(kind);
        self.component_bytes(component)
    }

    fn component_bytes(&self, component: QwenMoeExpertComponentLayout) -> &'a [u8] {
        let start = component.offset;
        let end = start + component.bytes;
        &self.slot.payload()[start..end]
    }
}

#[derive(Debug, Default)]
pub struct ReusableExpertBuffer {
    bytes: Vec<u8>,
}

impl ReusableExpertBuffer {
    pub fn prepare_payload(
        &mut self,
        slot_capacity: usize,
        payload_len: usize,
    ) -> Result<&mut [u8]> {
        if payload_len > slot_capacity {
            bail!("expert payload length {payload_len} exceeds slot capacity {slot_capacity}");
        }
        if self.bytes.capacity() < slot_capacity {
            self.bytes
                .try_reserve_exact(slot_capacity - self.bytes.capacity())
                .context("failed to reserve reusable expert buffer")?;
        }
        self.bytes.resize(payload_len, 0);
        Ok(&mut self.bytes)
    }

    pub fn slot_view(
        &self,
        layer: usize,
        expert: usize,
        slot_offset: u64,
        slot_capacity: usize,
    ) -> Result<ExpertSlotView<'_>> {
        ExpertSlotView::new(layer, expert, slot_offset, slot_capacity, &self.bytes)
    }

    pub fn take_payload(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.bytes)
    }

    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    pub fn adopt_buffer(&mut self, mut bytes: Vec<u8>) -> Vec<u8> {
        bytes.clear();
        std::mem::replace(&mut self.bytes, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_fixed_q4_layout() -> QwenMoeQ4ExpertLayout {
        use QwenMoeExpertComponentKind::*;
        QwenMoeQ4ExpertLayout {
            expert_bytes: 45,
            group_size: 2,
            components: [
                QwenMoeExpertComponentLayout {
                    kind: GateWeight,
                    offset: 0,
                    bytes: 8,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateScale,
                    offset: 8,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateBias,
                    offset: 12,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpWeight,
                    offset: 16,
                    bytes: 8,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpScale,
                    offset: 24,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpBias,
                    offset: 28,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownWeight,
                    offset: 32,
                    bytes: 8,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownScale,
                    offset: 40,
                    bytes: 3,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownBias,
                    offset: 43,
                    bytes: 2,
                },
            ],
        }
    }

    #[test]
    fn reusable_expert_buffer_keeps_capacity_across_smaller_reads() {
        let mut buffer = ReusableExpertBuffer::default();
        buffer.prepare_payload(128, 96).unwrap().fill(7);
        let initial_capacity = buffer.capacity();

        buffer.prepare_payload(128, 8).unwrap().fill(3);
        let slot = buffer.slot_view(2, 5, 1024, 128).unwrap();

        assert_eq!(buffer.capacity(), initial_capacity);
        assert_eq!(
            slot.descriptor(),
            ExpertSlotDescriptor {
                layer: 2,
                expert: 5,
                slot_offset: 1024,
                slot_capacity: 128,
                payload_len: 8,
            }
        );
        assert_eq!(slot.payload(), &[3; 8]);
    }

    #[test]
    fn reusable_expert_buffer_can_move_a_whole_slot_payload_without_copying() {
        let mut buffer = ReusableExpertBuffer::default();
        buffer.prepare_payload(128, 96).unwrap().fill(9);
        let initial_capacity = buffer.capacity();

        let payload = buffer.take_payload();

        assert_eq!(payload, vec![9; 96]);
        assert_eq!(payload.capacity(), initial_capacity);
        assert_eq!(buffer.capacity(), 0);
    }

    #[test]
    fn fixed_q4_expert_slot_view_slices_components_from_one_payload() {
        let payload: Vec<u8> = (0..45).collect();
        let slot = ExpertSlotView::new(4, 7, 4096, 45, &payload).unwrap();
        let view = FixedQ4ExpertSlotView::new(slot, tiny_fixed_q4_layout()).unwrap();

        assert_eq!(view.descriptor(), slot.descriptor());
        assert_eq!(view.payload(), payload.as_slice());
        assert_eq!(
            view.component(QwenMoeExpertComponentKind::GateWeight),
            &payload[0..8]
        );
        assert_eq!(
            view.component(QwenMoeExpertComponentKind::UpScale),
            &payload[24..28]
        );
        assert_eq!(
            view.component(QwenMoeExpertComponentKind::DownBias),
            &payload[43..45]
        );
    }

    #[test]
    fn fixed_q4_expert_slot_view_rejects_short_payloads() {
        let payload = [0u8; 44];
        let slot = ExpertSlotView::new(0, 0, 0, 45, &payload).unwrap();
        let err = FixedQ4ExpertSlotView::new(slot, tiny_fixed_q4_layout()).unwrap_err();

        assert!(
            err.to_string().contains("shorter than layout size 45"),
            "{err:#}"
        );
    }

    #[test]
    fn expert_slot_rejects_payloads_larger_than_the_slot() {
        let err = ExpertSlotView::new(0, 0, 0, 2, &[1, 2, 3]).unwrap_err();

        assert!(
            err.to_string()
                .contains("payload length 3 exceeds slot capacity 2"),
            "{err:#}"
        );
    }

    #[test]
    fn expert_io_policy_keeps_upstream_positioned_read_guardrails() {
        assert_eq!(
            FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
            ExpertReadPath::PositionedRead
        );
        assert!(!FLASHMOE_EXPERT_IO_POLICY.application_expert_cache);
        assert!(!FLASHMOE_EXPERT_IO_POLICY.lz4_expert_compression);
        assert!(!FLASHMOE_EXPERT_IO_POLICY.speculative_routing);
        assert!(!FLASHMOE_EXPERT_IO_POLICY.broad_ssd_gpu_overlap);
    }

    #[test]
    fn reusable_expert_byte_pool_reuses_capacity_qualified_buffers() {
        let pool: ReusableExpertBytePool = Arc::new(Mutex::new(Vec::new()));
        recycle_reusable_expert_bytes(&pool, Vec::with_capacity(64), 64);

        let returned = take_reusable_expert_bytes(&pool, 32).unwrap();

        assert!(returned.capacity() >= 64);
        assert!(pool.lock().unwrap().is_empty());
    }
}
