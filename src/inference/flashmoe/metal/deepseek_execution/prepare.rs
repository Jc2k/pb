use std::ptr;
use std::sync::Arc;
use std::thread;

use anyhow::{Context, Result, bail};

use crate::inference::flashmoe::deepseek::{DeepSeekResidentRange, DeepSeekV4ExecutionGraph};
use crate::inference::flashmoe::scheduler::{
    FlashMoeExecutionScheduler, PendingScheduledExpertLayerPrepare,
};

use super::super::{MetalDenseWeights, metal_page_size};

const RESIDENT_LAYER_PREPARE_WORKERS: usize = 1;

#[derive(Debug, Clone, Copy)]
struct DeepSeekResidentPageRange {
    byte_offset: usize,
    byte_len: usize,
}

#[derive(Debug)]
struct PendingDeepSeekResidentLayerPrepare {
    layer: usize,
    bytes: usize,
    workers: Vec<thread::JoinHandle<Result<usize>>>,
}

impl PendingDeepSeekResidentLayerPrepare {
    fn finish(mut self) -> Result<()> {
        let mut bytes = 0usize;
        let mut first_error = None;
        for worker in self.workers.drain(..) {
            match worker.join() {
                Ok(Ok(worker_bytes)) => match bytes.checked_add(worker_bytes) {
                    Some(total) => bytes = total,
                    None if first_error.is_none() => {
                        first_error = Some(anyhow::anyhow!(
                            "DeepSeek resident layer preparation completed byte count overflow"
                        ));
                    }
                    None => {}
                },
                Ok(Err(error)) if first_error.is_none() => first_error = Some(error),
                Err(_) if first_error.is_none() => {
                    first_error = Some(anyhow::anyhow!(
                        "DeepSeek resident layer preparation worker panicked on layer {}",
                        self.layer
                    ));
                }
                Ok(Err(_)) | Err(_) => {}
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        if bytes != self.bytes {
            bail!(
                "DeepSeek resident layer preparation completed {bytes} bytes for layer {}, expected {}",
                self.layer,
                self.bytes
            );
        }
        Ok(())
    }
}

impl Drop for PendingDeepSeekResidentLayerPrepare {
    fn drop(&mut self) {
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[derive(Debug)]
pub(super) struct PendingDeepSeekLayerPrepare<'a> {
    expert: PendingScheduledExpertLayerPrepare<'a>,
    resident: PendingDeepSeekResidentLayerPrepare,
}

fn resident_page_ranges(
    ranges: &[DeepSeekResidentRange],
    mmap_len: usize,
    page_size: usize,
) -> Result<Vec<DeepSeekResidentPageRange>> {
    if ranges.is_empty() || mmap_len == 0 || page_size == 0 || !page_size.is_power_of_two() {
        bail!(
            "DeepSeek resident layer preparation requires non-empty ranges, mmap bytes, and a power-of-two page size"
        );
    }
    let mut pages = Vec::<DeepSeekResidentPageRange>::with_capacity(ranges.len());
    for range in ranges {
        let start = usize::try_from(range.byte_offset)?;
        let len = usize::try_from(range.byte_len)?;
        let end = start
            .checked_add(len)
            .context("DeepSeek resident layer preparation range overflow")?;
        if len == 0 || end > mmap_len {
            bail!(
                "DeepSeek resident layer preparation range {start}..{end} is outside {mmap_len}-byte mmap"
            );
        }
        let page_start = start & !(page_size - 1);
        let page_end = end
            .checked_add(page_size - 1)
            .context("DeepSeek resident layer preparation page alignment overflow")?
            & !(page_size - 1);
        let page_end = page_end.min(mmap_len);
        if let Some(previous) = pages.last_mut() {
            let previous_end = previous
                .byte_offset
                .checked_add(previous.byte_len)
                .context("DeepSeek resident layer preparation page range overflow")?;
            if page_start <= previous_end {
                previous.byte_len = page_end
                    .max(previous_end)
                    .checked_sub(previous.byte_offset)
                    .context("DeepSeek resident layer preparation page merge underflow")?;
                continue;
            }
        }
        pages.push(DeepSeekResidentPageRange {
            byte_offset: page_start,
            byte_len: page_end - page_start,
        });
    }
    Ok(pages)
}

fn touch_resident_pages(
    mmap: &memmap2::Mmap,
    ranges: &[DeepSeekResidentPageRange],
    page_size: usize,
) -> Result<usize> {
    let mut sink = 0u8;
    let mut bytes = 0usize;
    for range in ranges {
        let end = range
            .byte_offset
            .checked_add(range.byte_len)
            .context("DeepSeek resident layer preparation worker range overflow")?;
        if range.byte_len == 0 || end > mmap.len() {
            bail!(
                "DeepSeek resident layer preparation worker range {}..{end} is outside {}-byte mmap",
                range.byte_offset,
                mmap.len()
            );
        }
        let mut offset = range.byte_offset;
        while offset < end {
            // A volatile read from each file-backed VM page synchronously
            // establishes residency without retaining an application buffer.
            sink ^= unsafe { ptr::read_volatile(mmap.as_ptr().add(offset)) };
            offset = offset
                .checked_add(page_size)
                .context("DeepSeek resident layer preparation page step overflow")?;
        }
        bytes = bytes
            .checked_add(range.byte_len)
            .context("DeepSeek resident layer preparation byte count overflow")?;
    }
    std::hint::black_box(sink);
    Ok(bytes)
}

fn issue_resident_layer_prepare(
    dense: &MetalDenseWeights,
    graph: &DeepSeekV4ExecutionGraph,
    layer: usize,
) -> Result<PendingDeepSeekResidentLayerPrepare> {
    let declared = graph
        .prefill_resident_layer_ranges
        .get(layer)
        .with_context(|| format!("DeepSeek resident preparation layer {layer} is not resolved"))?;
    let page_size = metal_page_size();
    let ranges = resident_page_ranges(declared, dense.len, page_size)?;
    let bytes = ranges.iter().try_fold(0usize, |total, range| {
        total
            .checked_add(range.byte_len)
            .context("DeepSeek resident layer preparation byte count overflow")
    })?;
    let mmap = Arc::clone(&dense._mmap);
    let workers = vec![thread::spawn(move || {
        touch_resident_pages(&mmap, &ranges, page_size)
    })];
    debug_assert_eq!(workers.len(), RESIDENT_LAYER_PREPARE_WORKERS);
    Ok(PendingDeepSeekResidentLayerPrepare {
        layer,
        bytes,
        workers,
    })
}

pub(super) fn issue_layer_prepare<'a>(
    dense: &MetalDenseWeights,
    graph: &DeepSeekV4ExecutionGraph,
    scheduler: &mut FlashMoeExecutionScheduler,
    layer: usize,
    destination: &'a mut [u8],
) -> Result<PendingDeepSeekLayerPrepare<'a>> {
    let resident = issue_resident_layer_prepare(dense, graph, layer)?;
    match unsafe { scheduler.issue_expert_layer_prepare_into(layer, destination) } {
        Ok(expert) => Ok(PendingDeepSeekLayerPrepare { expert, resident }),
        Err(expert_error) => match resident.finish() {
            Ok(()) => Err(expert_error),
            Err(resident_error) => bail!(
                "DeepSeek layer {layer} expert preparation issue failed: {expert_error:#}; resident preparation cleanup also failed: {resident_error:#}"
            ),
        },
    }
}

pub(super) fn finish_layer_prepare(
    scheduler: &mut FlashMoeExecutionScheduler,
    pending: PendingDeepSeekLayerPrepare<'_>,
) -> Result<()> {
    let PendingDeepSeekLayerPrepare { expert, resident } = pending;
    let expert_result = scheduler.finish_expert_layer_prepare(expert);
    let resident_result = resident.finish();
    match (expert_result, resident_result) {
        (Ok(_), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(expert_error), Err(resident_error)) => bail!(
            "DeepSeek layer preparation failed for expert stream: {expert_error:#}; resident stream also failed: {resident_error:#}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_prepare_geometry_page_aligns_and_coalesces_declared_tensors() {
        let ranges = resident_page_ranges(
            &[
                DeepSeekResidentRange {
                    byte_offset: 4_096,
                    byte_len: 4_096,
                },
                DeepSeekResidentRange {
                    byte_offset: 12_288,
                    byte_len: 8_192,
                },
                DeepSeekResidentRange {
                    byte_offset: 40_960,
                    byte_len: 4_096,
                },
            ],
            65_536,
            16_384,
        )
        .unwrap();

        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].byte_offset, 0);
        assert_eq!(ranges[0].byte_len, 49_152);
    }
}
