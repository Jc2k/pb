use std::mem::size_of;
use std::ptr;

use anyhow::{Context, Result, bail};

use super::super::{MetalExecutionContext, MetalObjcId, msg_send_usize0, release, sel};
use super::buffers::{MetalBufferAllocation, allocate_owned_buffer, checked_bytes};
use super::{
    ACTIVE_EXPERTS, DeepSeekLayerState, DeepSeekScratch, DeepSeekV4MetalState, EXPERT_WIDTH,
    EXPERTS, HC, HC_WIDTH, HEAD_DIM, HIDDEN, INDEX_HEAD_DIM, INDEX_HEADS, INDEX_TOP_K, INDEX_WIDTH,
    OUTPUT_LOW, Q_RANK, Q_WIDTH, RAW_CAP, buffer_contents,
};
use crate::inference::flashmoe::deepseek::DeepSeekV4ExecutionGraph;

impl DeepSeekV4MetalState {
    unsafe fn release(&mut self) {
        unsafe {
            for buffer in self.owned.drain(..) {
                release(buffer);
            }
        }
        self.bytes = 0;
    }

    pub(super) fn session_buffer_specs(
        &self,
        frontier: usize,
    ) -> Result<Vec<(MetalObjcId, usize)>> {
        let mut buffers = Vec::with_capacity(self.layers.len() * 7 + 2);
        for layer in &self.layers {
            buffers.push((layer.raw, unsafe {
                msg_send_usize0(layer.raw, sel("length"))
            }));
            if let Some(comp) = layer.comp {
                let rows = frontier / layer.ratio;
                let bytes = rows
                    .checked_mul(HEAD_DIM * size_of::<f32>())
                    .context("DeepSeek V4 compressed session snapshot size overflow")?
                    .max(size_of::<f32>());
                buffers.push((comp, bytes));
                buffers.push((
                    layer.comp_state_kv.expect("compressed KV frontier"),
                    unsafe {
                        msg_send_usize0(
                            layer.comp_state_kv.expect("compressed KV frontier"),
                            sel("length"),
                        )
                    },
                ));
                buffers.push((
                    layer.comp_state_score.expect("compressed score frontier"),
                    unsafe {
                        msg_send_usize0(
                            layer.comp_state_score.expect("compressed score frontier"),
                            sel("length"),
                        )
                    },
                ));
            }
            if let Some(index_comp) = layer.index_comp {
                let rows = frontier / layer.ratio;
                let bytes = rows
                    .checked_mul(INDEX_HEAD_DIM * size_of::<f32>())
                    .context("DeepSeek V4 index session snapshot size overflow")?
                    .max(size_of::<f32>());
                buffers.push((index_comp, bytes));
                buffers.push((layer.index_state_kv.expect("index KV frontier"), unsafe {
                    msg_send_usize0(
                        layer.index_state_kv.expect("index KV frontier"),
                        sel("length"),
                    )
                }));
                buffers.push((
                    layer.index_state_score.expect("index score frontier"),
                    unsafe {
                        msg_send_usize0(
                            layer.index_state_score.expect("index score frontier"),
                            sel("length"),
                        )
                    },
                ));
            }
        }
        for buffer in [self.scratch.cur_hc, self.scratch.output_hidden] {
            buffers.push((buffer, unsafe { msg_send_usize0(buffer, sel("length")) }));
        }
        for (buffer, bytes) in &buffers {
            let available = unsafe { msg_send_usize0(*buffer, sel("length")) };
            if *bytes > available {
                bail!(
                    "DeepSeek V4 session snapshot needs {bytes} bytes from a {available}-byte buffer"
                );
            }
        }
        Ok(buffers)
    }

    pub(super) unsafe fn allocate(
        context: &MetalExecutionContext,
        graph: &DeepSeekV4ExecutionGraph,
        capacity: usize,
    ) -> Result<Self> {
        if capacity == 0 {
            bail!("DeepSeek V4 Metal state requires a non-zero context capacity");
        }
        let mut owned = MetalBufferAllocation::default();
        let allocation = (|| -> Result<Self> {
            let mut alloc = |elements: usize, label: &str| unsafe {
                allocate_owned_buffer(context, &mut owned, checked_bytes(elements, label)?, label)
            };
            let mut layers = Vec::with_capacity(graph.layers.len());
            for &ratio in &graph.config.compress_ratios {
                let comp_cap = if ratio == 0 { 0 } else { capacity / ratio };
                let raw = alloc(RAW_CAP * HEAD_DIM, "raw KV ring")?;
                let (comp, comp_state_kv, comp_state_score) = if ratio == 0 {
                    (None, None, None)
                } else {
                    let width = if ratio == 4 { 1024 } else { 512 };
                    let state_rows = if ratio == 4 { 8 } else { 128 };
                    (
                        Some(alloc(comp_cap.max(1) * HEAD_DIM, "compressed KV cache")?),
                        Some(alloc(state_rows * width, "compressor KV frontier")?),
                        Some(alloc(state_rows * width, "compressor score frontier")?),
                    )
                };
                let (index_comp, index_state_kv, index_state_score) = if ratio == 4 {
                    (
                        Some(alloc(comp_cap.max(1) * INDEX_HEAD_DIM, "indexer KV cache")?),
                        Some(alloc(8 * 256, "indexer KV frontier")?),
                        Some(alloc(8 * 256, "indexer score frontier")?),
                    )
                } else {
                    (None, None, None)
                };
                layers.push(DeepSeekLayerState {
                    ratio,
                    comp_cap,
                    raw,
                    comp,
                    comp_state_kv,
                    comp_state_score,
                    index_comp,
                    index_state_kv,
                    index_state_score,
                });
            }
            let scratch = DeepSeekScratch {
                cur_hc: alloc(HC_WIDTH, "current HC")?,
                attn_hc: alloc(HC_WIDTH, "attention HC")?,
                next_hc: alloc(HC_WIDTH, "next HC")?,
                flat_hc: alloc(HC_WIDTH, "normalized HC")?,
                hc_mix: alloc(24, "HC mix")?,
                hc_split: alloc(24, "HC split")?,
                attn_cur: alloc(HIDDEN, "attention current")?,
                attn_norm: alloc(HIDDEN, "attention norm")?,
                qr: alloc(Q_RANK, "query rank")?,
                qr_norm: alloc(Q_RANK, "normalized query rank")?,
                kv_raw: alloc(HEAD_DIM, "projected KV")?,
                kv: alloc(HEAD_DIM, "normalized KV")?,
                q: alloc(Q_WIDTH, "projected query")?,
                heads: alloc(Q_WIDTH, "attention heads")?,
                attn_low: alloc(OUTPUT_LOW, "attention low rank")?,
                attn_out: alloc(HIDDEN, "attention output")?,
                ffn_cur: alloc(HIDDEN, "FFN current")?,
                ffn_norm: alloc(HIDDEN, "FFN norm")?,
                router: alloc(EXPERTS, "router logits")?,
                shared_gate: alloc(EXPERT_WIDTH, "shared gate")?,
                shared_up: alloc(EXPERT_WIDTH, "shared up")?,
                shared_mid: alloc(EXPERT_WIDTH, "shared mid")?,
                shared_out: alloc(HIDDEN, "shared output")?,
                routed_gate: alloc(ACTIVE_EXPERTS * EXPERT_WIDTH, "routed gates")?,
                routed_up: alloc(ACTIVE_EXPERTS * EXPERT_WIDTH, "routed ups")?,
                routed_mid: alloc(ACTIVE_EXPERTS * EXPERT_WIDTH, "routed mids")?,
                routed_out: alloc(HIDDEN, "routed output")?,
                route_weights: alloc(ACTIVE_EXPERTS, "route weights")?,
                comp_kv: alloc(1024, "compressor projection")?,
                comp_score: alloc(1024, "compressor scores")?,
                index_q: alloc(INDEX_WIDTH, "index query")?,
                index_weights: alloc(INDEX_HEADS, "index weights")?,
                index_scores: alloc(capacity.div_ceil(4).max(1), "index scores")?,
                index_selected: alloc(INDEX_TOP_K, "index selection")?,
                index_topk_scratch: alloc(capacity.div_ceil(4).max(1) * 2, "index top-k scratch")?,
                output_pre: alloc(HC, "output HC mix")?,
                output_hidden: alloc(HIDDEN, "output hidden")?,
                logits: alloc(129_280, "output logits")?,
            };
            let bytes = owned.iter().try_fold(0usize, |sum, &buffer| unsafe {
                sum.checked_add(msg_send_usize0(buffer, sel("length")))
                    .context("DeepSeek V4 Metal resident-state byte count overflow")
            })?;
            Ok(Self {
                capacity,
                next_position: 0,
                layers,
                scratch,
                owned: owned.take(),
                bytes,
            })
        })();
        allocation
    }

    pub(super) unsafe fn reset(&mut self) {
        unsafe {
            for &buffer in &self.owned {
                let len = msg_send_usize0(buffer, sel("length"));
                ptr::write_bytes(buffer_contents(buffer), 0, len);
            }
            for layer in &self.layers {
                for score in [layer.comp_state_score, layer.index_state_score]
                    .into_iter()
                    .flatten()
                {
                    let len = msg_send_usize0(score, sel("length")) / size_of::<f32>();
                    let values =
                        std::slice::from_raw_parts_mut(buffer_contents(score).cast::<f32>(), len);
                    values.fill(f32::NEG_INFINITY);
                }
            }
        }
        self.next_position = 0;
    }
}

impl Drop for DeepSeekV4MetalState {
    fn drop(&mut self) {
        unsafe { self.release() }
    }
}
