use anyhow::{Context, Result, bail};

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

    pub fn descriptor(self) -> ExpertSlotDescriptor {
        self.descriptor
    }

    pub fn payload(self) -> &'a [u8] {
        self.payload
    }

    pub fn payload_prefix(self, max_len: usize) -> &'a [u8] {
        &self.payload[..self.payload.len().min(max_len)]
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

    #[cfg(test)]
    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn expert_slot_rejects_payloads_larger_than_the_slot() {
        let err = ExpertSlotView::new(0, 0, 0, 2, &[1, 2, 3]).unwrap_err();

        assert!(
            err.to_string()
                .contains("payload length 3 exceeds slot capacity 2"),
            "{err:#}"
        );
    }
}
