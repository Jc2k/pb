use std::mem::size_of;
use std::ops::Deref;
use std::ptr;

use anyhow::{Context, Result, bail};

use crate::inference::flashmoe::experts::DeepSeekGgufExpertSlotSpec;

use super::super::{MetalExecutionContext, MetalObjcId, msg_send_id2_usize_u64, release, sel};
use super::{
    ACTIVE_EXPERTS, DeepSeekBatchScratch, EXPERT_WIDTH, EXPERTS, GROUP_WIDTH, HC, HC_WIDTH,
    HEAD_DIM, HIDDEN, INDEX_HEADS, INDEX_TOP_K, INDEX_WIDTH, OUTPUT_LOW, OUTPUT_RANK, Q_RANK,
    Q_WIDTH, RAW_CAP, buffer_contents,
};

#[derive(Debug, Default)]
pub(super) struct MetalBufferAllocation {
    buffers: Vec<MetalObjcId>,
}

impl MetalBufferAllocation {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            buffers: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, buffer: MetalObjcId) {
        self.buffers.push(buffer);
    }

    pub(super) fn take(&mut self) -> Vec<MetalObjcId> {
        std::mem::take(&mut self.buffers)
    }
}

impl Deref for MetalBufferAllocation {
    type Target = [MetalObjcId];

    fn deref(&self) -> &Self::Target {
        &self.buffers
    }
}

impl Drop for MetalBufferAllocation {
    fn drop(&mut self) {
        unsafe {
            for buffer in self.buffers.drain(..) {
                release(buffer);
            }
        }
    }
}

pub(super) fn checked_bytes(elements: usize, label: &str) -> Result<usize> {
    elements
        .checked_mul(size_of::<f32>())
        .with_context(|| format!("DeepSeek V4 {label} buffer size overflow"))
}

pub(super) unsafe fn allocate_owned_buffer(
    context: &MetalExecutionContext,
    owned: &mut MetalBufferAllocation,
    bytes: usize,
    label: &str,
) -> Result<MetalObjcId> {
    let bytes = bytes.max(size_of::<f32>());
    unsafe {
        context
            .buffers
            .ensure_allocation_capacity(context.runtime.device, bytes)?;
        let buffer = msg_send_id2_usize_u64(
            context.runtime.device,
            sel("newBufferWithLength:options:"),
            bytes,
            0,
        );
        if buffer.is_null() {
            bail!("failed to allocate DeepSeek V4 Metal {label} buffer ({bytes} bytes)");
        }
        ptr::write_bytes(buffer_contents(buffer), 0, bytes);
        context
            .resources
            .sample_device(context.runtime.device, false);
        owned.push(buffer);
        Ok(buffer)
    }
}

pub(super) unsafe fn allocate_owned_buffer_uninitialized(
    context: &MetalExecutionContext,
    owned: &mut MetalBufferAllocation,
    bytes: usize,
    label: &str,
) -> Result<MetalObjcId> {
    let bytes = bytes.max(size_of::<f32>());
    unsafe {
        context
            .buffers
            .ensure_allocation_capacity(context.runtime.device, bytes)?;
        let buffer = msg_send_id2_usize_u64(
            context.runtime.device,
            sel("newBufferWithLength:options:"),
            bytes,
            0,
        );
        if buffer.is_null() {
            bail!("failed to allocate DeepSeek V4 Metal {label} buffer ({bytes} bytes)");
        }
        context
            .resources
            .sample_device(context.runtime.device, false);
        owned.push(buffer);
        Ok(buffer)
    }
}

impl DeepSeekBatchScratch {
    pub(super) unsafe fn allocate(
        context: &MetalExecutionContext,
        tokens: usize,
        pos0: usize,
        double_expert_staging: bool,
    ) -> Result<Self> {
        if tokens == 0 {
            bail!("DeepSeek V4 batch prefill requires at least one token");
        }
        let expert_spec = DeepSeekGgufExpertSlotSpec::new(HIDDEN, EXPERT_WIDTH)?;
        let mut owned = MetalBufferAllocation::default();
        let allocation = (|| -> Result<Self> {
            let mut alloc_bytes = |bytes: usize, label: &str| unsafe {
                allocate_owned_buffer_uninitialized(context, &mut owned, bytes, label)
            };
            let f32_bytes = |rows: usize, width: usize, label: &str| -> Result<usize> {
                rows.checked_mul(width)
                    .and_then(|values| values.checked_mul(size_of::<f32>()))
                    .with_context(|| format!("DeepSeek V4 batch {label} size overflow"))
            };
            let hc_mix_width = 2 * HC + HC * HC;
            let prefix_raw = pos0.min(RAW_CAP);
            let raw_rows = prefix_raw
                .checked_add(tokens)
                .context("DeepSeek V4 raw batch context size overflow")?;
            let max_comp = pos0
                .checked_add(tokens)
                .context("DeepSeek V4 compressed frontier overflow")?
                / 4;
            let max_comp = max_comp.max(1);
            let max_flash_keys = raw_rows
                .checked_add(max_comp)
                .context("DeepSeek V4 FlashAttention key count overflow")?;
            let flash_mask_bytes = tokens
                .checked_mul(max_flash_keys)
                .and_then(|values| values.checked_mul(size_of::<u16>()))
                .context("DeepSeek V4 FlashAttention mask size overflow")?;
            let flash_kv_bytes = max_flash_keys
                .checked_mul(HEAD_DIM)
                .and_then(|values| values.checked_mul(size_of::<u16>()))
                .context("DeepSeek V4 FlashAttention KV staging size overflow")?;
            let flash_pad_bytes = 64usize
                .checked_mul(
                    2usize
                        .checked_mul(HEAD_DIM * size_of::<u16>())
                        .and_then(|bytes| bytes.checked_add(tokens * size_of::<u16>()))
                        .context("DeepSeek V4 FlashAttention pad row overflow")?,
                )
                .context("DeepSeek V4 FlashAttention pad size overflow")?;
            let flash_block_bytes = max_flash_keys
                .div_ceil(64)
                .checked_mul(tokens.div_ceil(8))
                .map(|bytes| bytes.next_multiple_of(32))
                .context("DeepSeek V4 FlashAttention block-map size overflow")?;
            let index_scratch_bytes = 2usize
                .checked_mul(tokens)
                .and_then(|values| values.checked_mul(max_comp.max(INDEX_TOP_K)))
                .and_then(|values| values.checked_mul(size_of::<i32>()))
                .context("DeepSeek V4 index top-k scratch size overflow")?;
            let moe_map_bytes = EXPERTS
                .checked_add(
                    EXPERTS
                        .checked_mul(tokens)
                        .context("MoE map size overflow")?,
                )
                .and_then(|values| values.checked_mul(size_of::<i32>()))
                .context("DeepSeek V4 MoE map byte size overflow")?;
            let expert_staging_bytes = EXPERTS
                .checked_mul(expert_spec.expert_bytes)
                .context("DeepSeek V4 expert staging size overflow")?;

            Ok(Self {
                tokens,
                token_ids: alloc_bytes(tokens * size_of::<u32>(), "batch token ids")?,
                cur_hc: alloc_bytes(
                    f32_bytes(tokens, HC_WIDTH, "current HC")?,
                    "batch current HC",
                )?,
                next_hc: alloc_bytes(f32_bytes(tokens, HC_WIDTH, "next HC")?, "batch next HC")?,
                flat_hc: alloc_bytes(f32_bytes(tokens, HC_WIDTH, "flat HC")?, "batch flat HC")?,
                hc_mix: alloc_bytes(f32_bytes(tokens, hc_mix_width, "HC mix")?, "batch HC mix")?,
                hc_split: alloc_bytes(
                    f32_bytes(tokens, hc_mix_width, "HC split")?,
                    "batch HC split",
                )?,
                attn_cur: alloc_bytes(
                    f32_bytes(tokens, HIDDEN, "attention current")?,
                    "batch attention current",
                )?,
                attn_norm: alloc_bytes(
                    f32_bytes(tokens, HIDDEN, "attention norm")?,
                    "batch attention norm",
                )?,
                qr: alloc_bytes(f32_bytes(tokens, Q_RANK, "query rank")?, "batch query rank")?,
                qr_norm: alloc_bytes(
                    f32_bytes(tokens, Q_RANK, "normalized query rank")?,
                    "batch normalized query rank",
                )?,
                kv_raw: alloc_bytes(f32_bytes(tokens, HEAD_DIM, "raw KV")?, "batch raw KV")?,
                kv: alloc_bytes(f32_bytes(tokens, HEAD_DIM, "KV")?, "batch KV")?,
                raw_context: alloc_bytes(
                    f32_bytes(
                        if pos0 == 0 { 1 } else { raw_rows },
                        HEAD_DIM,
                        "raw context",
                    )?,
                    "batch raw context",
                )?,
                q: alloc_bytes(f32_bytes(tokens, Q_WIDTH, "query")?, "batch query")?,
                heads: alloc_bytes(
                    f32_bytes(tokens, Q_WIDTH, "attention heads")?,
                    "batch attention heads",
                )?,
                attn_group: alloc_bytes(
                    f32_bytes(tokens, GROUP_WIDTH, "attention group")?,
                    "batch attention group",
                )?,
                attn_rank: alloc_bytes(
                    f32_bytes(tokens, OUTPUT_RANK, "attention rank")?,
                    "batch attention rank",
                )?,
                attn_low: alloc_bytes(
                    f32_bytes(tokens, OUTPUT_LOW, "attention low rank")?,
                    "batch attention low rank",
                )?,
                attn_out: alloc_bytes(
                    f32_bytes(tokens, HIDDEN, "attention output")?,
                    "batch attention output",
                )?,
                after_attn_hc: alloc_bytes(
                    f32_bytes(tokens, HC_WIDTH, "post-attention HC")?,
                    "batch post-attention HC",
                )?,
                ffn_cur: alloc_bytes(
                    f32_bytes(tokens, HIDDEN, "FFN current")?,
                    "batch FFN current",
                )?,
                ffn_norm: alloc_bytes(f32_bytes(tokens, HIDDEN, "FFN norm")?, "batch FFN norm")?,
                router: alloc_bytes(f32_bytes(tokens, EXPERTS, "router")?, "batch router")?,
                shared_gate: alloc_bytes(
                    f32_bytes(tokens, EXPERT_WIDTH, "shared gate")?,
                    "batch shared gate",
                )?,
                shared_up: alloc_bytes(
                    f32_bytes(tokens, EXPERT_WIDTH, "shared up")?,
                    "batch shared up",
                )?,
                shared_mid: alloc_bytes(
                    f32_bytes(tokens, EXPERT_WIDTH, "shared mid")?,
                    "batch shared mid",
                )?,
                shared_out: alloc_bytes(
                    f32_bytes(tokens, HIDDEN, "shared output")?,
                    "batch shared output",
                )?,
                route_selected: alloc_bytes(
                    tokens * ACTIVE_EXPERTS * size_of::<i32>(),
                    "batch selected experts",
                )?,
                route_weights: alloc_bytes(
                    f32_bytes(tokens, ACTIVE_EXPERTS, "route weights")?,
                    "batch route weights",
                )?,
                routed_mid: alloc_bytes(
                    tokens * ACTIVE_EXPERTS * EXPERT_WIDTH * size_of::<u16>(),
                    "batch routed mid",
                )?,
                routed_down: alloc_bytes(
                    f32_bytes(tokens * ACTIVE_EXPERTS, HIDDEN, "routed down")?,
                    "batch routed down",
                )?,
                routed_out: alloc_bytes(
                    f32_bytes(tokens, HIDDEN, "routed output")?,
                    "batch routed output",
                )?,
                moe_map: alloc_bytes(moe_map_bytes, "batch MoE expert map")?,
                expert_staging: alloc_bytes(expert_staging_bytes, "batch expert staging")?,
                expert_staging_alternate: double_expert_staging
                    .then(|| alloc_bytes(expert_staging_bytes, "batch alternate expert staging"))
                    .transpose()?,
                expert_spec,
                comp_kv: alloc_bytes(
                    f32_bytes(tokens, 1024, "compressor KV")?,
                    "batch compressor KV",
                )?,
                comp_score: alloc_bytes(
                    f32_bytes(tokens, 1024, "compressor score")?,
                    "batch compressor score",
                )?,
                index_q: alloc_bytes(
                    f32_bytes(tokens, INDEX_WIDTH, "index query")?,
                    "batch index query",
                )?,
                index_weights: alloc_bytes(
                    f32_bytes(tokens, INDEX_HEADS, "index weights")?,
                    "batch index weights",
                )?,
                index_scores: alloc_bytes(
                    f32_bytes(tokens, max_comp, "index scores")?,
                    "batch index scores",
                )?,
                index_selected: alloc_bytes(
                    tokens * INDEX_TOP_K * size_of::<i32>(),
                    "batch index selection",
                )?,
                index_sorted: alloc_bytes(
                    tokens * INDEX_TOP_K * size_of::<i32>(),
                    "batch sorted index selection",
                )?,
                index_topk_scratch: alloc_bytes(index_scratch_bytes, "batch index top-k scratch")?,
                rope_positions: alloc_bytes(tokens * size_of::<i32>(), "batch RoPE positions")?,
                comp_rope_positions: alloc_bytes(
                    max_comp * size_of::<i32>(),
                    "batch compressed RoPE positions",
                )?,
                flash_mask: alloc_bytes(flash_mask_bytes, "batch FlashAttention mask")?,
                flash_kv: alloc_bytes(flash_kv_bytes, "batch FlashAttention KV")?,
                flash_pad: alloc_bytes(flash_pad_bytes.max(1), "batch FlashAttention padding")?,
                flash_blocks: alloc_bytes(flash_block_bytes.max(1), "batch FlashAttention blocks")?,
                owned: owned.take(),
            })
        })();
        allocation
    }

    pub(super) fn alternate_expert_staging(&self) -> Result<MetalObjcId> {
        self.expert_staging_alternate
            .context("DeepSeek saturated batch graph did not allocate alternate expert staging")
    }

    pub(super) fn advance_expert_staging(&mut self) -> Result<()> {
        let alternate = self.alternate_expert_staging()?;
        self.expert_staging_alternate = Some(self.expert_staging);
        self.expert_staging = alternate;
        Ok(())
    }
}
