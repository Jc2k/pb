use std::mem::size_of;
use std::ptr;
use std::sync::Arc;

use anyhow::{Context, Result, bail};

use super::super::deepseek::{
    DeepSeekResidentDtype, DeepSeekResidentTensor, DeepSeekV4ExecutionGraph, DeepSeekV4LayerGraph,
    deepseek_v4_router_probabilities, deepseek_v4_select_routes,
};
use super::super::experts::ExpertMlpProjection;
use super::super::scheduler::{
    FlashMoeExecutionScheduler, ScheduledDeepSeekGgufExpertPhaseMlpPayload,
    ScheduledExpertPhaseMlpPayload, ScheduledExpertSet, ScheduledExpertSlot,
};
use super::*;

const HIDDEN: usize = 4096;
const HC: usize = 4;
const HC_WIDTH: usize = HIDDEN * HC;
const HEADS: usize = 64;
const HEAD_DIM: usize = 512;
const Q_RANK: usize = 1024;
const Q_WIDTH: usize = HEADS * HEAD_DIM;
const OUTPUT_GROUPS: usize = 8;
const GROUP_WIDTH: usize = 4096;
const OUTPUT_RANK: usize = 1024;
const OUTPUT_LOW: usize = OUTPUT_GROUPS * OUTPUT_RANK;
const EXPERTS: usize = 256;
const ACTIVE_EXPERTS: usize = 6;
const EXPERT_WIDTH: usize = 2048;
const RAW_CAP: usize = 128;
const INDEX_HEADS: usize = 64;
const INDEX_HEAD_DIM: usize = 128;
const INDEX_WIDTH: usize = INDEX_HEADS * INDEX_HEAD_DIM;
const INDEX_TOP_K: usize = 512;
const RMS_EPS: f32 = 1.0e-6;
const HC_EPS: f32 = 1.0e-6;

#[repr(C)]
#[derive(Clone, Copy)]
struct MatvecArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    nr0: i32,
    r2: i16,
    r3: i16,
}

impl MatvecArgs {
    fn q8(input: usize, output: usize) -> Result<Self> {
        if input % 32 != 0 {
            bail!("DeepSeek Q8_0 matvec input width {input} is not block aligned");
        }
        Self::new(input, output, (input / 32) * 34, 34)
    }

    fn f16(input: usize, output: usize) -> Result<Self> {
        Self::new(input, output, input * size_of::<u16>(), size_of::<u16>())
    }

    fn new(input: usize, output: usize, row_bytes: usize, block_bytes: usize) -> Result<Self> {
        Ok(Self {
            ne00: i32::try_from(input)?,
            ne01: i32::try_from(output)?,
            ne02: 1,
            nb00: block_bytes as u64,
            nb01: row_bytes as u64,
            nb02: (row_bytes * output) as u64,
            nb03: (row_bytes * output) as u64,
            ne10: i32::try_from(input)?,
            ne11: 1,
            ne12: 1,
            nb10: size_of::<f32>() as u64,
            nb11: (input * size_of::<f32>()) as u64,
            nb12: (input * size_of::<f32>()) as u64,
            nb13: (input * size_of::<f32>()) as u64,
            ne0: i32::try_from(output)?,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HcSplitNormArgs {
    n_embd: i64,
    n_hc: i32,
    sinkhorn_iters: i32,
    n_rows: i64,
    mix_hc: i64,
    nb_mix1: u64,
    nb_split1: u64,
    nb_x0: u64,
    nb_x1: u64,
    nb_x2: u64,
    nb0: u64,
    nb1: u64,
    nb_norm1: u64,
    eps: f32,
    norm_eps: f32,
}

impl HcSplitNormArgs {
    fn one() -> Self {
        Self {
            n_embd: HIDDEN as i64,
            n_hc: HC as i32,
            sinkhorn_iters: 20,
            n_rows: 1,
            mix_hc: 24,
            nb_mix1: (24 * size_of::<f32>()) as u64,
            nb_split1: (24 * size_of::<f32>()) as u64,
            nb_x0: size_of::<f32>() as u64,
            nb_x1: (HIDDEN * size_of::<f32>()) as u64,
            nb_x2: (HC_WIDTH * size_of::<f32>()) as u64,
            nb0: size_of::<f32>() as u64,
            nb1: (HIDDEN * size_of::<f32>()) as u64,
            nb_norm1: (HIDDEN * size_of::<f32>()) as u64,
            eps: HC_EPS,
            norm_eps: RMS_EPS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HcExpandArgs {
    n_embd: i64,
    n_hc: i64,
    n_tokens: i64,
    nb_block0: u64,
    nb_block1: u64,
    nb_add0: u64,
    nb_add1: u64,
    nb_res0: u64,
    nb_res1: u64,
    nb_res2: u64,
    nb_post0: u64,
    nb_post1: u64,
    nb_comb0: u64,
    nb_comb1: u64,
    nb_comb2: u64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    has_add: i32,
}

impl HcExpandArgs {
    fn one() -> Self {
        Self {
            n_embd: HIDDEN as i64,
            n_hc: HC as i64,
            n_tokens: 1,
            nb_block0: size_of::<f32>() as u64,
            nb_block1: (HIDDEN * size_of::<f32>()) as u64,
            nb_add0: size_of::<f32>() as u64,
            nb_add1: (HIDDEN * size_of::<f32>()) as u64,
            nb_res0: size_of::<f32>() as u64,
            nb_res1: (HIDDEN * size_of::<f32>()) as u64,
            nb_res2: (HC_WIDTH * size_of::<f32>()) as u64,
            nb_post0: size_of::<f32>() as u64,
            nb_post1: (HC * size_of::<f32>()) as u64,
            nb_comb0: size_of::<f32>() as u64,
            nb_comb1: (HC * size_of::<f32>()) as u64,
            nb_comb2: (HC * HC * size_of::<f32>()) as u64,
            nb0: size_of::<f32>() as u64,
            nb1: (HIDDEN * size_of::<f32>()) as u64,
            nb2: (HC_WIDTH * size_of::<f32>()) as u64,
            has_add: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsArgs {
    width: u32,
    rows: u32,
    weighted: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbeddingArgs {
    token: u32,
    hidden: u32,
    hc: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CompressorArgs {
    width: u32,
    head_dim: u32,
    ratio: u32,
    position: u32,
    emit_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttentionArgs {
    n_head: u32,
    head_dim: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    n_comp: u32,
    top_k: u32,
    use_top_k: u32,
    position: u32,
    window: u32,
    ratio: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct IndexedAttentionArgs {
    n_tokens: u32,
    n_head: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    n_comp: u32,
    top_k: u32,
    pos0: u32,
    window: u32,
    ratio: u32,
    comp_kv_f16: u32,
    pad0: u32,
    q_token_stride: u64,
    q_head_stride: u64,
    raw_row_stride: u64,
    comp_row_stride: u64,
    topk_token_stride: u64,
    dst_token_stride: u64,
    dst_head_stride: u64,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OutputCollapseArgs {
    hidden: u32,
    eps: f32,
    hc_eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct KvStoreArgs {
    head_dim: i32,
    n_rot: i32,
    raw_row: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Fp8Args {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    n_rot: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeArgs {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    n_dims: i32,
    mode: i32,
    n_ctx_orig: i32,
    inverse: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    src2: bool,
}

impl RopeArgs {
    fn one(heads: usize, width: usize, compressed: bool, inverse: bool) -> Result<Self> {
        let freq_scale = if compressed { 1.0 / 16.0 } else { 1.0 };
        let ext_factor = if compressed { 1.0 } else { 0.0 };
        let attn_factor = if compressed {
            1.0 / (1.0 + 0.1 * 16.0f32.ln())
        } else {
            1.0
        };
        Ok(Self {
            ne00: i64::try_from(width)?,
            ne01: i64::try_from(heads)?,
            ne02: 1,
            ne03: 1,
            nb00: size_of::<f32>() as u64,
            nb01: (width * size_of::<f32>()) as u64,
            nb02: (heads * width * size_of::<f32>()) as u64,
            nb03: (heads * width * size_of::<f32>()) as u64,
            nb0: size_of::<f32>() as u64,
            nb1: (width * size_of::<f32>()) as u64,
            nb2: (heads * width * size_of::<f32>()) as u64,
            nb3: (heads * width * size_of::<f32>()) as u64,
            n_dims: 64,
            mode: 0,
            n_ctx_orig: if compressed { 65_536 } else { 0 },
            inverse: i32::from(inverse),
            freq_base: if compressed { 160_000.0 } else { 10_000.0 },
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast: 32.0,
            beta_slow: 1.0,
            src2: false,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct IndexQatArgs {
    n_rows: u32,
    head_dim: u32,
    row_stride: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct IndexScoresArgs {
    n_comp: u32,
    n_tokens: u32,
    n_head: u32,
    head_dim: u32,
    pos0: u32,
    ratio: u32,
    q_token_stride: u64,
    q_head_stride: u64,
    weights_token_stride: u64,
    index_row_stride: u64,
    score_token_stride: u64,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ArgsortArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    top_k: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ArgsortMergeArgs {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    top_k: i32,
    len: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MoeMatvecArgs {
    nei0: i32,
    nei1: i32,
    nbi1: u64,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    ne0: i32,
    ne1: i32,
    nb1: u64,
    nr0: i32,
}

impl MoeMatvecArgs {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cols: usize,
        rows: usize,
        total_experts: usize,
        row_bytes: usize,
        expert_bytes: usize,
        rhs_rows: usize,
        nr0: usize,
    ) -> Result<Self> {
        let rhs_row_bytes = cols * size_of::<f32>();
        let blocks = cols / 256;
        if blocks == 0 || row_bytes % blocks != 0 {
            bail!("DeepSeek expert row layout is not GGUF block aligned");
        }
        Ok(Self {
            nei0: ACTIVE_EXPERTS as i32,
            nei1: 1,
            nbi1: (ACTIVE_EXPERTS * size_of::<i32>()) as u64,
            ne00: i32::try_from(cols)?,
            ne01: i32::try_from(rows)?,
            ne02: i32::try_from(total_experts)?,
            nb00: (row_bytes / blocks) as u64,
            nb01: row_bytes as u64,
            nb02: expert_bytes as u64,
            ne10: i32::try_from(cols)?,
            ne11: i32::try_from(rhs_rows)?,
            ne12: 1,
            ne13: 1,
            nb10: size_of::<f32>() as u64,
            nb11: rhs_row_bytes as u64,
            nb12: (rhs_rows * rhs_row_bytes) as u64,
            ne0: i32::try_from(rows)?,
            ne1: ACTIVE_EXPERTS as i32,
            nb1: (rows * size_of::<f32>()) as u64,
            nr0: i32::try_from(nr0)?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MoeActivationArgs {
    width: u32,
    rows: u32,
    gate_row_stride: u64,
    up_row_stride: u64,
    mid_row_stride: u64,
    weight_stride: u64,
    write_clamped: u32,
    clamp_value: f32,
}

#[derive(Debug)]
struct DeepSeekLayerState {
    ratio: usize,
    comp_cap: usize,
    raw: MetalObjcId,
    comp: Option<MetalObjcId>,
    comp_state_kv: Option<MetalObjcId>,
    comp_state_score: Option<MetalObjcId>,
    index_comp: Option<MetalObjcId>,
    index_state_kv: Option<MetalObjcId>,
    index_state_score: Option<MetalObjcId>,
}

#[derive(Debug)]
struct DeepSeekScratch {
    cur_hc: MetalObjcId,
    attn_hc: MetalObjcId,
    next_hc: MetalObjcId,
    flat_hc: MetalObjcId,
    hc_mix: MetalObjcId,
    hc_split: MetalObjcId,
    attn_cur: MetalObjcId,
    attn_norm: MetalObjcId,
    qr: MetalObjcId,
    qr_norm: MetalObjcId,
    kv_raw: MetalObjcId,
    kv: MetalObjcId,
    q: MetalObjcId,
    heads: MetalObjcId,
    attn_low: MetalObjcId,
    attn_out: MetalObjcId,
    ffn_cur: MetalObjcId,
    ffn_norm: MetalObjcId,
    router: MetalObjcId,
    shared_gate: MetalObjcId,
    shared_up: MetalObjcId,
    shared_mid: MetalObjcId,
    shared_out: MetalObjcId,
    routed_gate: MetalObjcId,
    routed_up: MetalObjcId,
    routed_mid: MetalObjcId,
    routed_out: MetalObjcId,
    route_weights: MetalObjcId,
    comp_kv: MetalObjcId,
    comp_score: MetalObjcId,
    index_q: MetalObjcId,
    index_weights: MetalObjcId,
    index_scores: MetalObjcId,
    index_selected: MetalObjcId,
    index_topk_scratch: MetalObjcId,
    output_pre: MetalObjcId,
    output_hidden: MetalObjcId,
    logits: MetalObjcId,
}

#[derive(Debug)]
pub(super) struct DeepSeekV4MetalState {
    capacity: usize,
    next_position: usize,
    layers: Vec<DeepSeekLayerState>,
    scratch: DeepSeekScratch,
    owned: Vec<MetalObjcId>,
    bytes: usize,
}

impl DeepSeekV4MetalState {
    pub(super) unsafe fn release(&mut self) {
        unsafe {
            for buffer in self.owned.drain(..) {
                super::release(buffer);
            }
        }
        self.bytes = 0;
    }
}

unsafe fn encode_matvec(
    pipelines: &DeepSeekMetalPipelineSet,
    encoder: MetalObjcId,
    dense: &MetalDenseWeights,
    tensor: &DeepSeekResidentTensor,
    weight_extra_offset: usize,
    input: MetalObjcId,
    input_offset: usize,
    output: MetalObjcId,
    output_offset: usize,
    cols: usize,
    rows: usize,
) -> Result<()> {
    let (kernel, args, row_bytes) = match tensor.dtype {
        DeepSeekResidentDtype::Q8_0 => (
            "kernel_mul_mv_q8_0_f32",
            MatvecArgs::q8(cols, rows)?,
            cols.checked_div(32)
                .and_then(|blocks| blocks.checked_mul(34))
                .context("DeepSeek Q8_0 matvec row size overflow")?,
        ),
        DeepSeekResidentDtype::F16 => (
            "kernel_mul_mv_f16_f32",
            MatvecArgs::f16(cols, rows)?,
            cols.checked_mul(size_of::<u16>())
                .context("DeepSeek F16 matvec row size overflow")?,
        ),
        dtype => bail!(
            "DeepSeek V4 tensor {} has unsupported {:?} matvec dtype",
            tensor.name,
            dtype
        ),
    };
    if weight_extra_offset % row_bytes != 0 {
        bail!(
            "DeepSeek matvec tensor {} submatrix offset {} is not row-aligned to {} bytes",
            tensor.name,
            weight_extra_offset,
            row_bytes
        );
    }
    let required_bytes = row_bytes
        .checked_mul(rows)
        .context("DeepSeek matvec submatrix size overflow")?;
    let tensor_len = usize::try_from(tensor.byte_len)?;
    let tensor_end = weight_extra_offset
        .checked_add(required_bytes)
        .context("DeepSeek matvec submatrix range overflow")?;
    if tensor_end > tensor_len {
        bail!(
            "DeepSeek matvec tensor {} submatrix {}..{} exceeds its {}-byte payload",
            tensor.name,
            weight_extra_offset,
            tensor_end,
            tensor_len
        );
    }
    let weight_offset = usize::try_from(tensor.byte_offset)?
        .checked_add(weight_extra_offset)
        .context("DeepSeek matvec weight offset overflow")?;
    let weight_end = weight_offset
        .checked_add(required_bytes)
        .context("DeepSeek matvec weight range overflow")?;
    if weight_end > dense.len {
        bail!(
            "DeepSeek matvec tensor {} is outside resident dense storage",
            tensor.name
        );
    }
    unsafe {
        set_pipeline(encoder, pipelines.require(kernel)?);
        set_bytes(encoder, bytes_of(&args), 0);
        set_buffer_with_offset(encoder, dense.buffer, weight_offset as u64, 1);
        set_buffer_with_offset(encoder, input, input_offset as u64, 2);
        set_buffer_with_offset(encoder, output, output_offset as u64, 3);
        set_threadgroup_memory(encoder, 256, 0);
        dispatch_groups(encoder, (rows.div_ceil(2) as u64, 1, 1), (32, 4, 1));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_rms(
    pipelines: &DeepSeekMetalPipelineSet,
    encoder: MetalObjcId,
    dense: &MetalDenseWeights,
    input: MetalObjcId,
    input_offset: usize,
    weight: Option<&DeepSeekResidentTensor>,
    output: MetalObjcId,
    output_offset: usize,
    width: usize,
    rows: usize,
) -> Result<()> {
    let args = RmsArgs {
        width: u32::try_from(width)?,
        rows: u32::try_from(rows)?,
        weighted: u32::from(weight.is_some()),
        eps: RMS_EPS,
    };
    unsafe {
        set_pipeline(encoder, pipelines.require("kernel_pb_dsv4_rms_norm_f32")?);
        set_bytes(encoder, bytes_of(&args), 0);
        set_buffer_with_offset(encoder, input, input_offset as u64, 1);
        if let Some(weight) = weight {
            set_buffer_with_offset(encoder, dense.buffer, weight.byte_offset, 2);
        } else {
            set_buffer(encoder, input, 2);
        }
        set_buffer_with_offset(encoder, output, output_offset as u64, 3);
        set_threadgroup_memory(encoder, 32, 0);
        dispatch_groups(encoder, (rows as u64, 1, 1), (256, 1, 1));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_rope(
    pipelines: &DeepSeekMetalPipelineSet,
    encoder: MetalObjcId,
    buffer: MetalObjcId,
    offset: usize,
    heads: usize,
    width: usize,
    position: usize,
    compressed: bool,
    inverse: bool,
) -> Result<()> {
    let args = RopeArgs::one(heads, width, compressed, inverse)?;
    let position = i32::try_from(position).context("DeepSeek RoPE position exceeds i32")?;
    unsafe {
        set_pipeline(encoder, pipelines.require("kernel_dsv4_rope_tail_f32")?);
        set_bytes(encoder, bytes_of(&args), 0);
        set_buffer_with_offset(encoder, buffer, offset as u64, 1);
        set_bytes(encoder, &position.to_ne_bytes(), 2);
        set_buffer_with_offset(encoder, buffer, offset as u64, 3);
        set_buffer_with_offset(encoder, buffer, offset as u64, 4);
        dispatch_groups(encoder, (heads as u64, 1, 1), (width.min(256) as u64, 1, 1));
    }
    Ok(())
}

unsafe fn encode_hc_pre(
    pipelines: &DeepSeekMetalPipelineSet,
    encoder: MetalObjcId,
    dense: &MetalDenseWeights,
    state: &DeepSeekScratch,
    input_hc: MetalObjcId,
    layer: &DeepSeekV4LayerGraph,
    attention: bool,
) -> Result<()> {
    let (function, scale, base, norm, current, normalized) = if attention {
        (
            &layer.hc_attn_fn,
            &layer.hc_attn_scale,
            &layer.hc_attn_base,
            &layer.attn_norm,
            state.attn_cur,
            state.attn_norm,
        )
    } else {
        (
            &layer.hc_ffn_fn,
            &layer.hc_ffn_scale,
            &layer.hc_ffn_base,
            &layer.ffn_norm,
            state.ffn_cur,
            state.ffn_norm,
        )
    };
    unsafe {
        encode_rms(
            pipelines,
            encoder,
            dense,
            input_hc,
            0,
            None,
            state.flat_hc,
            0,
            HC_WIDTH,
            1,
        )?;
        encode_matvec(
            pipelines,
            encoder,
            dense,
            function,
            0,
            state.flat_hc,
            0,
            state.hc_mix,
            0,
            HC_WIDTH,
            24,
        )?;
        set_pipeline(
            encoder,
            pipelines.require("kernel_dsv4_hc_split_weighted_sum_norm4")?,
        );
        set_bytes(encoder, bytes_of(&HcSplitNormArgs::one()), 0);
        set_buffer(encoder, state.hc_mix, 1);
        set_buffer_with_offset(encoder, dense.buffer, scale.byte_offset, 2);
        set_buffer_with_offset(encoder, dense.buffer, base.byte_offset, 3);
        set_buffer(encoder, input_hc, 4);
        set_buffer(encoder, state.hc_split, 5);
        set_buffer(encoder, current, 6);
        set_buffer_with_offset(encoder, dense.buffer, norm.byte_offset, 7);
        set_buffer(encoder, normalized, 8);
        set_threadgroup_memory(encoder, (HIDDEN + 4 + 32) * size_of::<f32>(), 0);
        dispatch_groups(encoder, (1, 1, 1), (1024, 1, 1));
    }
    Ok(())
}

unsafe fn encode_fp8_row(
    pipelines: &DeepSeekMetalPipelineSet,
    encoder: MetalObjcId,
    buffer: MetalObjcId,
    offset: usize,
    width: usize,
) -> Result<()> {
    let args = Fp8Args {
        ne00: i64::try_from(width)?,
        ne01: 1,
        ne02: 1,
        ne03: 1,
        nb00: size_of::<f32>() as u64,
        nb01: (width * size_of::<f32>()) as u64,
        nb02: (width * size_of::<f32>()) as u64,
        nb03: (width * size_of::<f32>()) as u64,
        nb0: size_of::<f32>() as u64,
        nb1: (width * size_of::<f32>()) as u64,
        nb2: (width * size_of::<f32>()) as u64,
        nb3: (width * size_of::<f32>()) as u64,
        n_rot: 64,
    };
    unsafe {
        set_pipeline(
            encoder,
            pipelines.require("kernel_dsv4_fp8_kv_quantize_f32")?,
        );
        set_bytes(encoder, bytes_of(&args), 0);
        set_buffer_with_offset(encoder, buffer, offset as u64, 1);
        set_buffer_with_offset(encoder, buffer, offset as u64, 2);
        set_threadgroup_memory(encoder, 64 * size_of::<f32>(), 0);
        dispatch_groups(encoder, (1, 1, 1), (64, 1, 1));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_compressor(
    pipelines: &DeepSeekMetalPipelineSet,
    encoder: MetalObjcId,
    dense: &MetalDenseWeights,
    source: MetalObjcId,
    kv_projection: &DeepSeekResidentTensor,
    gate_projection: &DeepSeekResidentTensor,
    ape: &DeepSeekResidentTensor,
    norm: &DeepSeekResidentTensor,
    ratio: usize,
    width: usize,
    head_dim: usize,
    position: usize,
    cache: MetalObjcId,
    state_kv: MetalObjcId,
    state_score: MetalObjcId,
    scratch: &DeepSeekScratch,
) -> Result<usize> {
    let n_comp = (position + 1) / ratio;
    let emit = (position + 1).is_multiple_of(ratio);
    unsafe {
        encode_matvec(
            pipelines,
            encoder,
            dense,
            kv_projection,
            0,
            source,
            0,
            scratch.comp_kv,
            0,
            HIDDEN,
            width,
        )?;
        encode_matvec(
            pipelines,
            encoder,
            dense,
            gate_projection,
            0,
            source,
            0,
            scratch.comp_score,
            0,
            HIDDEN,
            width,
        )?;
        let args = CompressorArgs {
            width: u32::try_from(width)?,
            head_dim: u32::try_from(head_dim)?,
            ratio: u32::try_from(ratio)?,
            position: u32::try_from(position)?,
            emit_row: u32::try_from(n_comp.saturating_sub(1))?,
        };
        set_pipeline(
            encoder,
            pipelines.require("kernel_pb_dsv4_compressor_step")?,
        );
        set_bytes(encoder, bytes_of(&args), 0);
        set_buffer(encoder, scratch.comp_kv, 1);
        set_buffer(encoder, scratch.comp_score, 2);
        set_buffer_with_offset(encoder, dense.buffer, ape.byte_offset, 3);
        set_buffer(encoder, state_kv, 4);
        set_buffer(encoder, state_score, 5);
        set_buffer(encoder, cache, 6);
        dispatch_groups(encoder, (1, 1, 1), (256, 1, 1));
        if emit {
            let row_offset = n_comp.saturating_sub(1) * head_dim * size_of::<f32>();
            encode_rms(
                pipelines,
                encoder,
                dense,
                cache,
                row_offset,
                Some(norm),
                cache,
                row_offset,
                head_dim,
                1,
            )?;
            encode_rope(
                pipelines,
                encoder,
                cache,
                row_offset,
                1,
                head_dim,
                position + 1 - ratio,
                true,
                false,
            )?;
            if head_dim == HEAD_DIM {
                encode_fp8_row(pipelines, encoder, cache, row_offset, head_dim)?;
            } else {
                let qat = IndexQatArgs {
                    n_rows: 1,
                    head_dim: u32::try_from(head_dim)?,
                    row_stride: (head_dim * size_of::<f32>()) as u64,
                };
                set_pipeline(
                    encoder,
                    pipelines.require("kernel_dsv4_indexer_hadamard_fp4_f32")?,
                );
                set_bytes(encoder, bytes_of(&qat), 0);
                set_buffer_with_offset(encoder, cache, row_offset as u64, 1);
                set_threadgroup_memory(encoder, 256 * size_of::<f32>(), 0);
                dispatch_groups(encoder, (1, 1, 1), (128, 1, 1));
            }
        }
    }
    Ok(n_comp)
}

unsafe fn pipeline_max_threads(pipeline: MetalObjcId) -> usize {
    unsafe { msg_send_usize0(pipeline, sel("maxTotalThreadsPerThreadgroup")) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IndexTopKGeometry {
    threads: usize,
    parts: usize,
    block_top: usize,
    work_width: usize,
}

fn index_topk_geometry(n_comp: usize, max_threads: usize) -> IndexTopKGeometry {
    let max_threads = max_threads.max(256);
    let mut threads = 1usize;
    while threads < n_comp && 2 * threads <= max_threads {
        threads *= 2;
    }
    let parts = n_comp.div_ceil(threads);
    let block_top = INDEX_TOP_K.min(threads);
    let last_block = n_comp - (parts - 1) * threads;
    let work_width = if parts > 1 {
        (parts - 1) * block_top + last_block.min(block_top)
    } else {
        INDEX_TOP_K
    };
    IndexTopKGeometry {
        threads,
        parts,
        block_top,
        work_width,
    }
}

unsafe fn encode_index_topk(
    pipelines: &DeepSeekMetalPipelineSet,
    encoder: MetalObjcId,
    scratch: &DeepSeekScratch,
    n_comp: usize,
) -> Result<()> {
    if n_comp <= INDEX_TOP_K {
        return Ok(());
    }
    let argsort_pipeline = pipelines.require("kernel_argsort_f32_i32_desc")?;
    let geometry = index_topk_geometry(n_comp, unsafe { pipeline_max_threads(argsort_pipeline) });
    let nth = geometry.threads;
    let npr = geometry.parts;
    let block_top = geometry.block_top;
    let work_width = geometry.work_width;
    let args = ArgsortArgs {
        ne00: i32::try_from(n_comp)?,
        ne01: 1,
        ne02: 1,
        ne03: 1,
        nb00: size_of::<f32>() as u64,
        nb01: (n_comp * size_of::<f32>()) as u64,
        nb02: (n_comp * size_of::<f32>()) as u64,
        nb03: (n_comp * size_of::<f32>()) as u64,
        ne0: i32::try_from(work_width)?,
        ne1: 1,
        ne2: 1,
        ne3: 1,
        top_k: i32::try_from(block_top)?,
    };
    let scratch_row_bytes = work_width * size_of::<i32>();
    let mut current_offset = 0usize;
    let mut next_offset = scratch_row_bytes;
    let single_pass = npr == 1;
    unsafe {
        set_pipeline(encoder, argsort_pipeline);
        set_bytes(encoder, bytes_of(&args), 0);
        set_buffer(encoder, scratch.index_scores, 1);
        if single_pass {
            set_buffer(encoder, scratch.index_selected, 2);
        } else {
            set_buffer_with_offset(
                encoder,
                scratch.index_topk_scratch,
                current_offset as u64,
                2,
            );
        }
        set_threadgroup_memory(encoder, (nth * size_of::<i32>()).next_multiple_of(16), 0);
        dispatch_groups(encoder, (npr as u64, 1, 1), (nth as u64, 1, 1));

        if single_pass {
            return Ok(());
        }
        let merge_pipeline = pipelines.require("kernel_argsort_merge_f32_i32_desc")?;
        let mut len = block_top;
        while len < work_width {
            let nm = work_width.div_ceil(2 * len);
            let final_merge = nm == 1;
            let merge_threads = pipeline_max_threads(merge_pipeline).clamp(1, 512).min(len);
            let merge_args = ArgsortMergeArgs {
                ne00: i64::try_from(n_comp)?,
                ne01: 1,
                ne02: 1,
                ne03: 1,
                nb00: size_of::<f32>() as u64,
                nb01: (n_comp * size_of::<f32>()) as u64,
                nb02: (n_comp * size_of::<f32>()) as u64,
                nb03: (n_comp * size_of::<f32>()) as u64,
                ne0: i32::try_from(work_width)?,
                ne1: 1,
                ne2: 1,
                ne3: 1,
                top_k: i32::try_from(if final_merge { INDEX_TOP_K } else { work_width })?,
                len: i32::try_from(len)?,
            };
            set_pipeline(encoder, merge_pipeline);
            set_bytes(encoder, bytes_of(&merge_args), 0);
            set_buffer(encoder, scratch.index_scores, 1);
            set_buffer_with_offset(
                encoder,
                scratch.index_topk_scratch,
                current_offset as u64,
                2,
            );
            if final_merge {
                set_buffer(encoder, scratch.index_selected, 3);
            } else {
                set_buffer_with_offset(encoder, scratch.index_topk_scratch, next_offset as u64, 3);
            }
            dispatch_groups(encoder, (nm as u64, 1, 1), (merge_threads as u64, 1, 1));
            std::mem::swap(&mut current_offset, &mut next_offset);
            len <<= 1;
        }
    }
    Ok(())
}

unsafe fn encode_attention(
    pipelines: &DeepSeekMetalPipelineSet,
    encoder: MetalObjcId,
    layer_state: &DeepSeekLayerState,
    scratch: &DeepSeekScratch,
    sinks: (&MetalDenseWeights, &DeepSeekResidentTensor),
    position: usize,
    n_comp: usize,
) -> Result<()> {
    let n_raw = (position + 1).min(RAW_CAP);
    let raw_start = (position + 1 - n_raw) % RAW_CAP;
    let use_top_k = layer_state.ratio == 4 && n_comp > INDEX_TOP_K;
    if use_top_k {
        let row_stride = (HEAD_DIM * size_of::<f32>()) as u64;
        let args = IndexedAttentionArgs {
            n_tokens: 1,
            n_head: HEADS as u32,
            n_raw: n_raw as u32,
            raw_cap: RAW_CAP as u32,
            raw_start: raw_start as u32,
            n_comp: n_comp as u32,
            top_k: INDEX_TOP_K as u32,
            pos0: position as u32,
            window: RAW_CAP as u32,
            ratio: layer_state.ratio as u32,
            comp_kv_f16: 0,
            pad0: 0,
            q_token_stride: (HEADS as u64) * row_stride,
            q_head_stride: row_stride,
            raw_row_stride: row_stride,
            comp_row_stride: row_stride,
            topk_token_stride: (INDEX_TOP_K * size_of::<i32>()) as u64,
            dst_token_stride: (HEADS as u64) * row_stride,
            dst_head_stride: row_stride,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        };
        unsafe {
            set_pipeline(
                encoder,
                pipelines.require("kernel_dsv4_indexed_mixed_attention_heads8_rb16")?,
            );
            set_bytes(encoder, bytes_of(&args), 0);
            set_buffer(encoder, scratch.q, 1);
            set_buffer(encoder, layer_state.raw, 2);
            set_buffer(
                encoder,
                layer_state
                    .comp
                    .context("indexed DeepSeek attention is missing compressed KV")?,
                3,
            );
            set_buffer(encoder, scratch.index_selected, 4);
            set_buffer_with_offset(encoder, sinks.0.buffer, sinks.1.byte_offset, 5);
            set_buffer(encoder, scratch.heads, 6);
            set_threadgroup_memory(encoder, 16 * 128 * size_of::<[u16; 4]>(), 0);
            dispatch_groups(encoder, (1, HEADS.div_ceil(8) as u64, 1), (32, 8, 1));
        }
        return Ok(());
    }
    let args = AttentionArgs {
        n_head: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        n_raw: n_raw as u32,
        raw_cap: RAW_CAP as u32,
        raw_start: raw_start as u32,
        n_comp: n_comp as u32,
        top_k: 0,
        use_top_k: 0,
        position: position as u32,
        window: RAW_CAP as u32,
        ratio: layer_state.ratio as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };
    let comp = layer_state.comp.unwrap_or(layer_state.raw);
    unsafe {
        set_pipeline(
            encoder,
            pipelines.require("kernel_pb_dsv4_decode_attention_h512")?,
        );
        set_bytes(encoder, bytes_of(&args), 0);
        set_buffer(encoder, scratch.q, 1);
        set_buffer(encoder, layer_state.raw, 2);
        set_buffer(encoder, comp, 3);
        set_buffer(encoder, scratch.index_selected, 4);
        set_buffer_with_offset(encoder, sinks.0.buffer, sinks.1.byte_offset, 5);
        set_buffer(encoder, scratch.heads, 6);
        set_threadgroup_memory(encoder, 12 * size_of::<f32>(), 0);
        dispatch_groups(encoder, (HEADS as u64, 1, 1), (128, 1, 1));
    }
    Ok(())
}

unsafe fn encode_layer_pre_expert(
    context: &MetalExecutionContext,
    encoding: &mut MetalCommandEncoding,
    dense: &MetalDenseWeights,
    graph_layer: &DeepSeekV4LayerGraph,
    layer_state: &DeepSeekLayerState,
    scratch: &DeepSeekScratch,
    position: usize,
) -> Result<()> {
    let pipelines = context.deepseek_pipelines()?;
    let encoder = encoding.encoder();
    let compressed = layer_state.ratio != 0;
    unsafe {
        encode_hc_pre(
            pipelines,
            encoder,
            dense,
            scratch,
            scratch.cur_hc,
            graph_layer,
            true,
        )?;
        encode_matvec(
            pipelines,
            encoder,
            dense,
            &graph_layer.attn_q_a,
            0,
            scratch.attn_norm,
            0,
            scratch.qr,
            0,
            HIDDEN,
            Q_RANK,
        )?;
        encode_matvec(
            pipelines,
            encoder,
            dense,
            &graph_layer.attn_kv,
            0,
            scratch.attn_norm,
            0,
            scratch.kv_raw,
            0,
            HIDDEN,
            HEAD_DIM,
        )?;
        encode_rms(
            pipelines,
            encoder,
            dense,
            scratch.qr,
            0,
            Some(&graph_layer.attn_q_a_norm),
            scratch.qr_norm,
            0,
            Q_RANK,
            1,
        )?;
        encode_rms(
            pipelines,
            encoder,
            dense,
            scratch.kv_raw,
            0,
            Some(&graph_layer.attn_kv_a_norm),
            scratch.kv,
            0,
            HEAD_DIM,
            1,
        )?;
        encode_matvec(
            pipelines,
            encoder,
            dense,
            &graph_layer.attn_q_b,
            0,
            scratch.qr_norm,
            0,
            scratch.q,
            0,
            Q_RANK,
            Q_WIDTH,
        )?;
        encode_rms(
            pipelines, encoder, dense, scratch.q, 0, None, scratch.q, 0, HEAD_DIM, HEADS,
        )?;
        encode_rope(
            pipelines, encoder, scratch.q, 0, HEADS, HEAD_DIM, position, compressed, false,
        )?;
        encode_rope(
            pipelines, encoder, scratch.kv, 0, 1, HEAD_DIM, position, compressed, false,
        )?;
        let kv_store = KvStoreArgs {
            head_dim: HEAD_DIM as i32,
            n_rot: 64,
            raw_row: (position % RAW_CAP) as i32,
        };
        set_pipeline(encoder, pipelines.require("kernel_dsv4_kv_fp8_store_f32")?);
        set_bytes(encoder, bytes_of(&kv_store), 0);
        set_buffer(encoder, scratch.kv, 1);
        set_buffer(encoder, layer_state.raw, 2);
        set_threadgroup_memory(encoder, 64 * size_of::<f32>(), 0);
        dispatch_groups(encoder, (1, 1, 1), (64, 1, 1));

        let n_comp = if let Some(compressor) = &graph_layer.compressor {
            let n_comp = encode_compressor(
                pipelines,
                encoder,
                dense,
                scratch.attn_norm,
                &compressor.kv,
                &compressor.gate,
                &compressor.ape,
                &compressor.norm,
                compressor.ratio,
                if compressor.ratio == 4 { 1024 } else { 512 },
                HEAD_DIM,
                position,
                layer_state
                    .comp
                    .context("compressed layer is missing KV cache")?,
                layer_state
                    .comp_state_kv
                    .context("compressed layer is missing KV frontier")?,
                layer_state
                    .comp_state_score
                    .context("compressed layer is missing score frontier")?,
                scratch,
            )?;
            if n_comp > layer_state.comp_cap {
                bail!(
                    "DeepSeek V4 compressed cache capacity exceeded at position {position}: {n_comp} > {}",
                    layer_state.comp_cap
                );
            }
            n_comp
        } else {
            0
        };

        if let Some(indexer) = &graph_layer.indexer {
            encode_matvec(
                pipelines,
                encoder,
                dense,
                &indexer.q_b,
                0,
                scratch.qr_norm,
                0,
                scratch.index_q,
                0,
                Q_RANK,
                INDEX_WIDTH,
            )?;
            encode_rope(
                pipelines,
                encoder,
                scratch.index_q,
                0,
                INDEX_HEADS,
                INDEX_HEAD_DIM,
                position,
                true,
                false,
            )?;
            let qat = IndexQatArgs {
                n_rows: INDEX_HEADS as u32,
                head_dim: INDEX_HEAD_DIM as u32,
                row_stride: (INDEX_HEAD_DIM * size_of::<f32>()) as u64,
            };
            set_pipeline(
                encoder,
                pipelines.require("kernel_dsv4_indexer_hadamard_fp4_f32")?,
            );
            set_bytes(encoder, bytes_of(&qat), 0);
            set_buffer(encoder, scratch.index_q, 1);
            set_threadgroup_memory(encoder, 256 * size_of::<f32>(), 0);
            dispatch_groups(encoder, (INDEX_HEADS as u64, 1, 1), (128, 1, 1));
            encode_matvec(
                pipelines,
                encoder,
                dense,
                &indexer.projection,
                0,
                scratch.attn_norm,
                0,
                scratch.index_weights,
                0,
                HIDDEN,
                INDEX_HEADS,
            )?;
            let index_n_comp = encode_compressor(
                pipelines,
                encoder,
                dense,
                scratch.attn_norm,
                &indexer.compressor_kv,
                &indexer.compressor_gate,
                &indexer.compressor_ape,
                &indexer.compressor_norm,
                4,
                256,
                INDEX_HEAD_DIM,
                position,
                layer_state
                    .index_comp
                    .context("ratio-4 layer is missing index cache")?,
                layer_state
                    .index_state_kv
                    .context("ratio-4 layer is missing index KV frontier")?,
                layer_state
                    .index_state_score
                    .context("ratio-4 layer is missing index score frontier")?,
                scratch,
            )?;
            if index_n_comp != n_comp {
                bail!("DeepSeek V4 attention/index compressor frontiers diverged");
            }
            if n_comp > 0 {
                let args = IndexScoresArgs {
                    n_comp: n_comp as u32,
                    n_tokens: 1,
                    n_head: INDEX_HEADS as u32,
                    head_dim: INDEX_HEAD_DIM as u32,
                    pos0: position as u32,
                    ratio: 4,
                    q_token_stride: (INDEX_WIDTH * size_of::<f32>()) as u64,
                    q_head_stride: (INDEX_HEAD_DIM * size_of::<f32>()) as u64,
                    weights_token_stride: (INDEX_HEADS * size_of::<f32>()) as u64,
                    index_row_stride: (INDEX_HEAD_DIM * size_of::<f32>()) as u64,
                    score_token_stride: (n_comp * size_of::<f32>()) as u64,
                    scale: 1.0 / (INDEX_WIDTH as f32).sqrt(),
                };
                set_pipeline(
                    encoder,
                    pipelines.require("kernel_dsv4_indexer_score_one_direct")?,
                );
                set_bytes(encoder, bytes_of(&args), 0);
                set_buffer(encoder, scratch.index_q, 1);
                set_buffer(encoder, scratch.index_weights, 2);
                set_buffer(
                    encoder,
                    layer_state.index_comp.expect("validated index cache"),
                    3,
                );
                set_buffer(encoder, scratch.index_scores, 4);
                set_threadgroup_memory(encoder, (INDEX_HEAD_DIM + 4) * size_of::<f32>(), 0);
                dispatch_groups(encoder, (n_comp as u64, 1, 1), (32, 4, 1));
                encode_index_topk(pipelines, encoder, scratch, n_comp)?;
            }
        }

        encode_attention(
            pipelines,
            encoder,
            layer_state,
            scratch,
            (dense, &graph_layer.attn_sinks),
            position,
            n_comp,
        )?;
        encode_rope(
            pipelines,
            encoder,
            scratch.heads,
            0,
            HEADS,
            HEAD_DIM,
            position,
            compressed,
            true,
        )?;
        let q8_row_bytes = (GROUP_WIDTH / 32) * 34;
        for group in 0..OUTPUT_GROUPS {
            encode_matvec(
                pipelines,
                encoder,
                dense,
                &graph_layer.attn_output_a,
                group * OUTPUT_RANK * q8_row_bytes,
                scratch.heads,
                group * GROUP_WIDTH * size_of::<f32>(),
                scratch.attn_low,
                group * OUTPUT_RANK * size_of::<f32>(),
                GROUP_WIDTH,
                OUTPUT_RANK,
            )?;
        }
        set_pipeline(
            encoder,
            pipelines.require("kernel_dsv4_q8_hc_expand4_q8_0")?,
        );
        set_bytes(encoder, bytes_of(&MatvecArgs::q8(OUTPUT_LOW, HIDDEN)?), 0);
        set_bytes(encoder, bytes_of(&HcExpandArgs::one()), 1);
        set_buffer_with_offset(
            encoder,
            dense.buffer,
            graph_layer.attn_output_b.byte_offset,
            2,
        );
        set_buffer(encoder, scratch.attn_low, 3);
        set_buffer(encoder, scratch.attn_out, 4);
        set_buffer(encoder, scratch.cur_hc, 5);
        set_buffer_with_offset(encoder, scratch.hc_split, (4 * size_of::<f32>()) as u64, 6);
        set_buffer_with_offset(encoder, scratch.hc_split, (8 * size_of::<f32>()) as u64, 7);
        set_buffer(encoder, scratch.attn_hc, 8);
        set_threadgroup_memory(encoder, 256, 0);
        dispatch_groups(encoder, (HIDDEN.div_ceil(2) as u64, 1, 1), (32, 4, 1));

        encode_hc_pre(
            pipelines,
            encoder,
            dense,
            scratch,
            scratch.attn_hc,
            graph_layer,
            false,
        )?;
        encode_matvec(
            pipelines,
            encoder,
            dense,
            &graph_layer.router,
            0,
            scratch.ffn_norm,
            0,
            scratch.router,
            0,
            HIDDEN,
            EXPERTS,
        )?;
        set_pipeline(
            encoder,
            pipelines.require("kernel_dsv4_shared_gate_up_swiglu_q8_0")?,
        );
        set_bytes(encoder, bytes_of(&MatvecArgs::q8(HIDDEN, EXPERT_WIDTH)?), 0);
        set_buffer_with_offset(
            encoder,
            dense.buffer,
            graph_layer.shared_gate.byte_offset,
            1,
        );
        set_buffer_with_offset(encoder, dense.buffer, graph_layer.shared_up.byte_offset, 2);
        set_buffer(encoder, scratch.ffn_norm, 3);
        set_buffer(encoder, scratch.shared_gate, 4);
        set_buffer(encoder, scratch.shared_up, 5);
        set_buffer(encoder, scratch.shared_mid, 6);
        set_bytes(encoder, &10.0f32.to_ne_bytes(), 7);
        set_threadgroup_memory(encoder, 512, 0);
        dispatch_groups(encoder, (EXPERT_WIDTH.div_ceil(2) as u64, 1, 1), (32, 4, 1));
    }
    Ok(())
}

fn read_resident_f32(
    dense: &MetalDenseWeights,
    tensor: &DeepSeekResidentTensor,
) -> Result<Vec<f32>> {
    if tensor.dtype != DeepSeekResidentDtype::F32 {
        bail!("DeepSeek resident tensor {} is not F32", tensor.name);
    }
    let start = usize::try_from(tensor.byte_offset)?;
    let end = start
        .checked_add(usize::try_from(tensor.byte_len)?)
        .context("DeepSeek resident F32 range overflow")?;
    let bytes = dense
        ._mmap
        .get(start..end)
        .with_context(|| format!("DeepSeek resident tensor {} is outside mmap", tensor.name))?;
    Ok(bytes
        .chunks_exact(size_of::<f32>())
        .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte chunk")))
        .collect())
}

fn read_hash_routes(
    dense: &MetalDenseWeights,
    tensor: &DeepSeekResidentTensor,
    token: u32,
) -> Result<[i32; ACTIVE_EXPERTS]> {
    if tensor.dtype != DeepSeekResidentDtype::I32 {
        bail!("DeepSeek hash tensor {} is not I32", tensor.name);
    }
    let token = usize::try_from(token)?;
    if token >= 129_280 {
        bail!("DeepSeek token {token} is outside the fixed vocabulary");
    }
    let start = usize::try_from(tensor.byte_offset)?
        .checked_add(token * ACTIVE_EXPERTS * size_of::<i32>())
        .context("DeepSeek hash-route offset overflow")?;
    let bytes = dense
        ._mmap
        .get(start..start + ACTIVE_EXPERTS * size_of::<i32>())
        .context("DeepSeek hash-route row is outside mmap")?;
    let mut selected = [0i32; ACTIVE_EXPERTS];
    for (slot, value) in selected
        .iter_mut()
        .zip(bytes.chunks_exact(size_of::<i32>()))
    {
        *slot = i32::from_le_bytes(value.try_into().expect("four-byte chunk"));
    }
    Ok(selected)
}

unsafe fn commit_deepseek_command(
    mut encoding: MetalCommandEncoding,
    phase: &'static str,
    position: usize,
    layer: Option<usize>,
) -> Result<()> {
    unsafe {
        encoding.end_encoding();
        let mut context = MetalCommandContext::new(phase).with("position", position);
        if let Some(layer) = layer {
            context = context.with("layer", layer);
        }
        commit_and_wait_metal_command_buffer(encoding.command_buffer(), &context)?;
    }
    Ok(())
}

fn deepseek_payloads(
    scheduled: &ScheduledExpertSet<Arc<ScheduledExpertSlot>>,
) -> Result<Vec<ScheduledDeepSeekGgufExpertPhaseMlpPayload<'_>>> {
    let payloads = scheduled.cmd3_expert_phase_payloads(HIDDEN)?;
    payloads
        .into_iter()
        .map(|payload| match payload {
            ScheduledExpertPhaseMlpPayload::DeepSeekGguf(payload) => Ok(payload),
            ScheduledExpertPhaseMlpPayload::Q4(_) | ScheduledExpertPhaseMlpPayload::Dense(_) => {
                bail!("DeepSeek V4 scheduler returned a non-GGUF expert payload")
            }
        })
        .collect()
}

unsafe fn expert_buffers(
    context: &MetalExecutionContext,
    payloads: &[ScheduledDeepSeekGgufExpertPhaseMlpPayload<'_>],
) -> Result<Vec<MetalObjcId>> {
    let mut buffers = Vec::with_capacity(ACTIVE_EXPERTS);
    for payload in payloads {
        if payload.layer >= 43 || payload.expert >= EXPERTS {
            bail!(
                "DeepSeek V4 scheduler returned invalid layer/expert {}/{}",
                payload.layer,
                payload.expert
            );
        }
        let bytes = payload.bytes.as_slice();
        let buffer = unsafe {
            persistent_expert_source_buffer(
                context.runtime.device,
                bytes,
                payload.bytes,
                &context.buffers,
            )?
        }
        .context("DeepSeek V4 scheduler expert slot is not page-aligned for Metal")?;
        buffers.push(buffer);
    }
    if buffers.len() != ACTIVE_EXPERTS {
        bail!(
            "DeepSeek V4 scheduler returned {} expert slots, expected {ACTIVE_EXPERTS}",
            buffers.len()
        );
    }
    Ok(buffers)
}

unsafe fn encode_layer_experts(
    context: &MetalExecutionContext,
    encoding: &mut MetalCommandEncoding,
    dense: &MetalDenseWeights,
    graph_layer: &DeepSeekV4LayerGraph,
    state: &mut DeepSeekV4MetalState,
    payloads: &[ScheduledDeepSeekGgufExpertPhaseMlpPayload<'_>],
    expert_buffers: &[MetalObjcId],
) -> Result<()> {
    let pipelines = context.deepseek_pipelines()?;
    let encoder = encoding.encoder();
    let scratch = &state.scratch;
    let spec = payloads
        .first()
        .context("DeepSeek V4 expert phase has no payloads")?
        .spec;
    if payloads.iter().any(|payload| payload.spec != spec) {
        bail!("DeepSeek V4 expert slots disagree on their fixed projection layout");
    }
    let gate = spec.projection(ExpertMlpProjection::Gate);
    let up = spec.projection(ExpertMlpProjection::Up);
    let down = spec.projection(ExpertMlpProjection::Down);
    let gate_args = MoeMatvecArgs::new(
        HIDDEN,
        EXPERT_WIDTH,
        EXPERTS,
        gate.bytes / EXPERT_WIDTH,
        gate.bytes,
        1,
        4,
    )?;
    let activation = MoeActivationArgs {
        width: EXPERT_WIDTH as u32,
        rows: ACTIVE_EXPERTS as u32,
        gate_row_stride: (EXPERT_WIDTH * size_of::<f32>()) as u64,
        up_row_stride: (EXPERT_WIDTH * size_of::<f32>()) as u64,
        mid_row_stride: (EXPERT_WIDTH * size_of::<f32>()) as u64,
        weight_stride: size_of::<f32>() as u64,
        write_clamped: 0,
        clamp_value: 10.0,
    };
    unsafe {
        set_pipeline(
            encoder,
            pipelines.require("kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32")?,
        );
        set_bytes(encoder, bytes_of(&gate_args), 0);
        set_bytes(encoder, bytes_of(&activation), 1);
        for (slot, &buffer) in expert_buffers.iter().enumerate() {
            set_buffer_with_offset(encoder, buffer, gate.offset as u64, 2 + slot as u64);
            set_buffer_with_offset(encoder, buffer, up.offset as u64, 8 + slot as u64);
        }
        set_buffer(encoder, scratch.ffn_norm, 14);
        set_buffer(encoder, scratch.routed_gate, 15);
        set_buffer(encoder, scratch.routed_up, 16);
        set_buffer(encoder, scratch.routed_mid, 17);
        set_buffer(encoder, scratch.route_weights, 18);
        set_threadgroup_memory(encoder, 2176, 0);
        dispatch_groups(encoder, (256, 1, ACTIVE_EXPERTS as u64), (32, 2, 1));

        let down_args = MoeMatvecArgs::new(
            EXPERT_WIDTH,
            HIDDEN,
            EXPERTS,
            down.bytes / HIDDEN,
            down.bytes,
            ACTIVE_EXPERTS,
            4,
        )?;
        set_pipeline(
            encoder,
            pipelines.require("kernel_mul_mv_slots6_q2_K_sum6_f32")?,
        );
        set_bytes(encoder, bytes_of(&down_args), 0);
        for (slot, &buffer) in expert_buffers.iter().enumerate() {
            set_buffer_with_offset(encoder, buffer, down.offset as u64, 1 + slot as u64);
        }
        set_buffer(encoder, scratch.routed_mid, 7);
        set_buffer(encoder, scratch.routed_out, 8);
        dispatch_groups(encoder, (512, 1, 1), (32, 2, 1));

        set_pipeline(
            encoder,
            pipelines.require("kernel_dsv4_shared_down_hc_expand4_q8_0")?,
        );
        set_bytes(encoder, bytes_of(&MatvecArgs::q8(EXPERT_WIDTH, HIDDEN)?), 0);
        set_bytes(encoder, bytes_of(&HcExpandArgs::one()), 1);
        set_buffer_with_offset(
            encoder,
            dense.buffer,
            graph_layer.shared_down.byte_offset,
            2,
        );
        set_buffer(encoder, scratch.shared_mid, 3);
        set_buffer(encoder, scratch.shared_out, 4);
        set_buffer(encoder, scratch.routed_out, 5);
        set_buffer(encoder, scratch.attn_hc, 6);
        set_buffer_with_offset(encoder, scratch.hc_split, (4 * size_of::<f32>()) as u64, 7);
        set_buffer_with_offset(encoder, scratch.hc_split, (8 * size_of::<f32>()) as u64, 8);
        set_buffer(encoder, scratch.next_hc, 9);
        set_threadgroup_memory(encoder, 256, 0);
        dispatch_groups(encoder, (HIDDEN.div_ceil(2) as u64, 1, 1), (32, 4, 1));
    }
    std::mem::swap(&mut state.scratch.cur_hc, &mut state.scratch.next_hc);
    Ok(())
}

fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
}

unsafe fn set_pipeline(encoder: MetalObjcId, pipeline: MetalObjcId) {
    unsafe { msg_send_void1_id(encoder, sel("setComputePipelineState:"), pipeline) }
}

unsafe fn set_threadgroup_memory(encoder: MetalObjcId, len: usize, index: u64) {
    unsafe {
        let f: unsafe extern "C" fn(MetalObjcId, MetalSelector, usize, u64) =
            std::mem::transmute(objc_msgSend as *const ());
        f(
            encoder,
            sel("setThreadgroupMemoryLength:atIndex:"),
            len,
            index,
        );
    }
}

unsafe fn dispatch_groups(encoder: MetalObjcId, grid: (u64, u64, u64), threads: (u64, u64, u64)) {
    unsafe {
        dispatch_metal_plan(
            encoder,
            MetalDispatchPlan {
                mode: MetalDispatchMode::Threadgroups,
                grid: MetalDispatchSize::new(grid.0.max(1), grid.1.max(1), grid.2.max(1)),
                threadgroup: MetalDispatchSize::new(
                    threads.0.max(1),
                    threads.1.max(1),
                    threads.2.max(1),
                ),
            },
        )
    }
}

unsafe fn buffer_contents(buffer: MetalObjcId) -> *mut u8 {
    unsafe { msg_send_ptr0(buffer, sel("contents")).cast::<u8>() }
}

fn checked_bytes(elements: usize, label: &str) -> Result<usize> {
    elements
        .checked_mul(size_of::<f32>())
        .with_context(|| format!("DeepSeek V4 {label} buffer size overflow"))
}

unsafe fn allocate_owned_buffer(
    context: &MetalExecutionContext,
    owned: &mut Vec<MetalObjcId>,
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

impl DeepSeekV4MetalState {
    unsafe fn allocate(
        context: &MetalExecutionContext,
        graph: &DeepSeekV4ExecutionGraph,
        capacity: usize,
    ) -> Result<Self> {
        if capacity == 0 {
            bail!("DeepSeek V4 Metal state requires a non-zero context capacity");
        }
        let mut owned = Vec::new();
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
                owned: std::mem::take(&mut owned),
                bytes,
            })
        })();
        if allocation.is_err() {
            for buffer in owned.drain(..) {
                unsafe { release(buffer) };
            }
        }
        allocation
    }

    unsafe fn reset(&mut self) {
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

impl MetalExecutionContext {
    pub(crate) fn prepare_deepseek_v4_state(
        &self,
        graph: &DeepSeekV4ExecutionGraph,
        capacity: usize,
    ) -> Result<()> {
        if graph.layers.len() != 43 || graph.config.block_count != 43 {
            bail!("DeepSeek V4 Metal state received a graph other than the fixed 43-layer profile");
        }
        let mut guard = self.deepseek_state.lock().map_err(|_| {
            anyhow::anyhow!("DeepSeek V4 Metal state lock is poisoned during preparation")
        })?;
        unsafe {
            if guard
                .as_ref()
                .is_none_or(|state| state.capacity != capacity)
            {
                if let Some(mut previous) = guard.take() {
                    previous.release();
                }
                *guard = Some(DeepSeekV4MetalState::allocate(self, graph, capacity)?);
            }
            guard.as_mut().expect("DeepSeek state allocated").reset();
        }
        let state_bytes = guard.as_ref().map_or(0, |state| state.bytes);
        let recurrent_bytes = self
            .linear_attention_state
            .lock()
            .map_err(|_| anyhow::anyhow!("Metal recurrent-state lock poisoned"))
            .map(|state| linear_attention_state_bytes(&state))?;
        self.resources.record_resident_resources(
            self.dense_weights.as_ref().map_or(0, |weights| weights.len),
            recurrent_bytes.saturating_add(state_bytes),
        );
        Ok(())
    }

    pub(crate) fn deepseek_v4_forward_token(
        &self,
        graph: &DeepSeekV4ExecutionGraph,
        scheduler: &mut FlashMoeExecutionScheduler,
        token: u32,
        position: usize,
    ) -> Result<Vec<f32>> {
        let dense = self
            .dense_weights
            .as_ref()
            .context("DeepSeek V4 requires resident mmap-backed Metal weights")?;
        let mut state_guard = self.deepseek_state.lock().map_err(|_| {
            anyhow::anyhow!("DeepSeek V4 Metal state lock is poisoned during token execution")
        })?;
        let state = state_guard
            .as_mut()
            .context("DeepSeek V4 Metal state was not prepared before prefill")?;
        if position != state.next_position {
            bail!(
                "DeepSeek V4 token position {position} does not match resident state frontier {}",
                state.next_position
            );
        }
        if position >= state.capacity {
            bail!(
                "DeepSeek V4 token position {position} exceeds prepared context capacity {}",
                state.capacity
            );
        }
        let embedding = EmbeddingArgs {
            token,
            hidden: HIDDEN as u32,
            hc: HC as u32,
        };
        unsafe {
            let encoding = MetalCommandEncoding::new(
                self.runtime.command_queue,
                Arc::clone(&self.resources),
                "failed to create DeepSeek V4 embedding command buffer",
                "failed to create DeepSeek V4 embedding encoder",
            )?;
            let encoder = encoding.encoder();
            set_pipeline(
                encoder,
                self.deepseek_pipelines()?
                    .require("kernel_pb_dsv4_embedding_hc4")?,
            );
            set_bytes(encoder, bytes_of(&embedding), 0);
            set_buffer_with_offset(encoder, dense.buffer, graph.embedding.byte_offset, 1);
            set_buffer(encoder, state.scratch.cur_hc, 2);
            dispatch_groups(encoder, (HC_WIDTH.div_ceil(256) as u64, 1, 1), (256, 1, 1));
            commit_deepseek_command(encoding, "deepseek_embedding", position, None)?;
        }

        for (layer, graph_layer) in graph.layers.iter().enumerate() {
            unsafe {
                let mut encoding = MetalCommandEncoding::new(
                    self.runtime.command_queue,
                    Arc::clone(&self.resources),
                    "failed to create DeepSeek V4 layer pre-expert command buffer",
                    "failed to create DeepSeek V4 layer pre-expert encoder",
                )?;
                encode_layer_pre_expert(
                    self,
                    &mut encoding,
                    dense,
                    graph_layer,
                    &state.layers[layer],
                    &state.scratch,
                    position,
                )?;
                commit_deepseek_command(
                    encoding,
                    "deepseek_layer_pre_expert",
                    position,
                    Some(layer),
                )?;
            }
            let logits = unsafe { read_f32_buffer(state.scratch.router, EXPERTS) };
            let probabilities = deepseek_v4_router_probabilities(&logits)?;
            let correction_bias = graph_layer
                .router_bias
                .as_ref()
                .map(|tensor| read_resident_f32(dense, tensor))
                .transpose()?;
            let hash_selected = graph_layer
                .token_hash_routes
                .as_ref()
                .map(|tensor| read_hash_routes(dense, tensor, token))
                .transpose()?;
            let routes = deepseek_v4_select_routes(
                &probabilities,
                correction_bias.as_deref(),
                hash_selected.as_ref().map(|values| values.as_slice()),
            )?;
            let scheduled = scheduler.read_preselected_experts(layer, &routes)?;
            if scheduled.layer != layer || scheduled.weights.len() != ACTIVE_EXPERTS {
                bail!(
                    "DeepSeek V4 scheduler returned inconsistent layer/weight count {}/{} for layer {layer}",
                    scheduled.layer,
                    scheduled.weights.len()
                );
            }
            unsafe {
                ptr::copy_nonoverlapping(
                    scheduled.weights.as_ptr(),
                    buffer_contents(state.scratch.route_weights).cast::<f32>(),
                    ACTIVE_EXPERTS,
                );
            }
            let payloads = deepseek_payloads(&scheduled)?;
            let expert_buffers = unsafe { expert_buffers(self, &payloads)? };
            unsafe {
                let mut encoding = MetalCommandEncoding::new(
                    self.runtime.command_queue,
                    Arc::clone(&self.resources),
                    "failed to create DeepSeek V4 streamed-expert command buffer",
                    "failed to create DeepSeek V4 streamed-expert encoder",
                )?;
                encode_layer_experts(
                    self,
                    &mut encoding,
                    dense,
                    graph_layer,
                    state,
                    &payloads,
                    &expert_buffers,
                )?;
                commit_deepseek_command(
                    encoding,
                    "deepseek_streamed_experts",
                    position,
                    Some(layer),
                )?;
            }
        }

        unsafe {
            let pipelines = self.deepseek_pipelines()?;
            let encoding = MetalCommandEncoding::new(
                self.runtime.command_queue,
                Arc::clone(&self.resources),
                "failed to create DeepSeek V4 output command buffer",
                "failed to create DeepSeek V4 output encoder",
            )?;
            let encoder = encoding.encoder();
            encode_rms(
                pipelines,
                encoder,
                dense,
                state.scratch.cur_hc,
                0,
                None,
                state.scratch.flat_hc,
                0,
                HC_WIDTH,
                1,
            )?;
            encode_matvec(
                pipelines,
                encoder,
                dense,
                &graph.output_hc_fn,
                0,
                state.scratch.flat_hc,
                0,
                state.scratch.output_pre,
                0,
                HC_WIDTH,
                HC,
            )?;
            let args = OutputCollapseArgs {
                hidden: HIDDEN as u32,
                eps: RMS_EPS,
                hc_eps: HC_EPS,
            };
            set_pipeline(
                encoder,
                pipelines.require("kernel_pb_dsv4_output_collapse_norm4")?,
            );
            set_bytes(encoder, bytes_of(&args), 0);
            set_buffer(encoder, state.scratch.cur_hc, 1);
            set_buffer(encoder, state.scratch.output_pre, 2);
            set_buffer_with_offset(encoder, dense.buffer, graph.output_hc_scale.byte_offset, 3);
            set_buffer_with_offset(encoder, dense.buffer, graph.output_hc_base.byte_offset, 4);
            set_buffer_with_offset(encoder, dense.buffer, graph.output_norm.byte_offset, 5);
            set_buffer(encoder, state.scratch.output_hidden, 6);
            set_threadgroup_memory(encoder, 32, 0);
            dispatch_groups(encoder, (1, 1, 1), (256, 1, 1));
            commit_deepseek_command(encoding, "deepseek_output", position, None)?;
        }
        state.next_position += 1;
        Ok(unsafe { read_f32_buffer(state.scratch.output_hidden, HIDDEN) })
    }

    pub(crate) fn deepseek_v4_logits(
        &self,
        graph: &DeepSeekV4ExecutionGraph,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        if hidden.len() != HIDDEN {
            bail!(
                "DeepSeek V4 output projection received hidden width {}, expected {HIDDEN}",
                hidden.len()
            );
        }
        let dense = self
            .dense_weights
            .as_ref()
            .context("DeepSeek V4 requires resident mmap-backed Metal weights")?;
        let state_guard = self.deepseek_state.lock().map_err(|_| {
            anyhow::anyhow!("DeepSeek V4 Metal state lock is poisoned during logits projection")
        })?;
        let state = state_guard
            .as_ref()
            .context("DeepSeek V4 Metal state was not prepared before sampling")?;
        unsafe {
            ptr::copy_nonoverlapping(
                hidden.as_ptr(),
                buffer_contents(state.scratch.output_hidden).cast::<f32>(),
                HIDDEN,
            );
            let encoding = MetalCommandEncoding::new(
                self.runtime.command_queue,
                Arc::clone(&self.resources),
                "failed to create DeepSeek V4 logits command buffer",
                "failed to create DeepSeek V4 logits encoder",
            )?;
            encode_matvec(
                self.deepseek_pipelines()?,
                encoding.encoder(),
                dense,
                &graph.output,
                0,
                state.scratch.output_hidden,
                0,
                state.scratch.logits,
                0,
                HIDDEN,
                129_280,
            )?;
            commit_deepseek_command(encoding, "deepseek_logits", state.next_position, None)?;
            Ok(read_f32_buffer(state.scratch.logits, 129_280))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_argument_layouts_match_vendored_metal_abi() {
        assert_eq!(size_of::<MatvecArgs>(), 112);
        assert_eq!(size_of::<HcSplitNormArgs>(), 104);
        assert_eq!(size_of::<HcExpandArgs>(), 152);
        assert_eq!(size_of::<RmsArgs>(), 16);
        assert_eq!(size_of::<EmbeddingArgs>(), 12);
        assert_eq!(size_of::<CompressorArgs>(), 20);
        assert_eq!(size_of::<AttentionArgs>(), 48);
        assert_eq!(size_of::<IndexedAttentionArgs>(), 112);
        assert_eq!(size_of::<OutputCollapseArgs>(), 12);
        assert_eq!(size_of::<KvStoreArgs>(), 12);
        assert_eq!(size_of::<Fp8Args>(), 104);
        assert_eq!(size_of::<RopeArgs>(), 144);
        assert_eq!(size_of::<IndexQatArgs>(), 16);
        assert_eq!(size_of::<IndexScoresArgs>(), 72);
        assert_eq!(size_of::<ArgsortArgs>(), 72);
        assert_eq!(size_of::<ArgsortMergeArgs>(), 88);
        assert_eq!(size_of::<MoeMatvecArgs>(), 120);
        assert_eq!(size_of::<MoeActivationArgs>(), 48);
    }

    #[test]
    fn rope_specialization_is_fixed_by_compression_mode() {
        let dense = RopeArgs::one(HEADS, HEAD_DIM, false, false).unwrap();
        assert_eq!(dense.freq_base, 10_000.0);
        assert_eq!(dense.freq_scale, 1.0);
        assert_eq!(dense.n_ctx_orig, 0);
        assert_eq!(dense.ext_factor, 0.0);
        assert_eq!(dense.inverse, 0);

        let compressed = RopeArgs::one(HEADS, HEAD_DIM, true, true).unwrap();
        assert_eq!(compressed.freq_base, 160_000.0);
        assert_eq!(compressed.freq_scale, 1.0 / 16.0);
        assert_eq!(compressed.n_ctx_orig, 65_536);
        assert_eq!(compressed.ext_factor, 1.0);
        assert_eq!(compressed.inverse, 1);
        assert_eq!(compressed.beta_fast, 32.0);
        assert_eq!(compressed.beta_slow, 1.0);
    }

    #[test]
    fn streamed_expert_arguments_are_six_slot_fixed_shapes() {
        let gate = MoeMatvecArgs::new(4096, 2048, 256, 1056, 2_162_688, 1, 4).unwrap();
        assert_eq!(gate.nei0, 6);
        assert_eq!(gate.ne00, 4096);
        assert_eq!(gate.ne01, 2048);
        assert_eq!(gate.ne10, 4096);
        assert_eq!(gate.ne11, 1);
        assert_eq!(gate.nr0, 4);

        let down = MoeMatvecArgs::new(2048, 4096, 256, 672, 2_752_512, 6, 4).unwrap();
        assert_eq!(down.nei0, 6);
        assert_eq!(down.ne00, 2048);
        assert_eq!(down.ne01, 4096);
        assert_eq!(down.ne10, 2048);
        assert_eq!(down.ne11, 6);
        assert_eq!(down.nr0, 4);
    }

    #[test]
    fn index_topk_geometry_publishes_single_pass_and_merges_multi_pass_ranges() {
        let single = index_topk_geometry(513, 1024);
        assert_eq!(single.threads, 1024);
        assert_eq!(single.parts, 1);
        assert_eq!(single.work_width, INDEX_TOP_K);

        let multi = index_topk_geometry(1025, 1024);
        assert_eq!(multi.threads, 1024);
        assert_eq!(multi.parts, 2);
        assert_eq!(multi.block_top, INDEX_TOP_K);
        assert_eq!(multi.work_width, INDEX_TOP_K + 1);
    }
}
