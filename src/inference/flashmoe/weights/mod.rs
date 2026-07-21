use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[cfg(target_os = "macos")]
use std::ffi::c_int;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::artifact::{
    AggregateExpertTensor, EXPERT_SCALE_BIAS_DTYPE_BF16, EXPERT_SCALE_BIAS_DTYPE_F32,
    ExpertSourceTensor, expert_scale_bias_dtype_size,
};
use super::math::{q4_dequantize_rows_with_group_size, quantize_q4, softmax_in_place};
#[cfg(test)]
use super::math::{q4_fma_matvec_with_group_size, rms_norm_with_weight_in_place};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::metal::{
    MetalBatchProjectionInput, MetalGlmMlaFusedAttentionInput, MetalGlmMlaFusedAttentionOutput,
    MetalGlmMlaPostAttentionInput, MetalObjcId as ObjcId, MetalPostAttentionPrep,
};
use super::metal::{MetalExecutionFacade, MetalGlmMlaAbsorbedAttentionInput};
use super::model_family::{
    QwenModelConfig, QwenMoeFamily, QwenMoeLayerKind, QwenNormWeightSemantics,
};
use super::safetensors::{SafetensorShard, parse_safetensors_header};
use super::state::{
    FlashMoeRoutingOutputSource, FlashMoeRoutingOutputState, LinearAttentionLayout,
};
#[cfg(test)]
use super::types::FlashMoeLayerKind;
use super::types::{
    ExpertQuantization, GROUP_SIZE, LINEAR_KEY_DIM, LINEAR_TOTAL_KEY, LINEAR_TOTAL_VALUE,
    LINEAR_VALUE_DIM,
};

#[cfg(test)]
fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}
use anyhow::{Context, Result, bail};

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: c_int = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: c_int = 111;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemv(
        order: c_int,
        trans_a: c_int,
        m: c_int,
        n: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        x: *const f32,
        inc_x: c_int,
        beta: f32,
        y: *mut f32,
        inc_y: c_int,
    );
}

mod conversion;
pub(crate) use conversion::*;

pub(super) fn dense_f32_matvec_rows(
    weights: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Option<Vec<f32>>> {
    let expected = rows
        .checked_mul(cols)
        .context("dense f32 matvec row-major weight size overflow")?;
    if weights.len() < expected || input.len() < cols {
        return Ok(None);
    }

    #[cfg(target_os = "macos")]
    {
        let Ok(m) = c_int::try_from(rows) else {
            return Ok(None);
        };
        let Ok(n) = c_int::try_from(cols) else {
            return Ok(None);
        };
        let mut out = vec![0.0f32; rows];
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                m,
                n,
                1.0,
                weights.as_ptr(),
                n,
                input.as_ptr(),
                1,
                0.0,
                out.as_mut_ptr(),
                1,
            );
        }
        return Ok(Some(out));
    }

    #[cfg(not(target_os = "macos"))]
    {
        let mut out = vec![0.0f32; rows];
        for row in 0..rows {
            let start = row
                .checked_mul(cols)
                .context("dense f32 matvec row offset overflow")?;
            let weights = &weights[start..start + cols];
            out[row] = weights
                .iter()
                .zip(input.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
        }
        Ok(Some(out))
    }
}

pub(super) fn validate_required_tensor_manifest(
    config: &QwenModelConfig,
    registry: &TensorRegistry,
) -> Result<()> {
    require_tensor_shape(
        registry,
        "model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
    )?;
    require_tensor_shape(registry, "model.norm.weight", &[config.hidden_size])?;
    // lm_head.weight is optional: when absent (or when tie_word_embeddings is true) the model
    // uses tied embeddings and reuses model.embed_tokens.weight (already validated above) for
    // the output projection.
    if registry.tensor("lm_head.weight").is_some() {
        require_tensor_shape(
            registry,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
    }
    for layer in 0..config.num_hidden_layers {
        match infer_attention_layer_type(registry, layer)? {
            AttentionLayerType::Full => {
                let layout = infer_full_attention_layout(config, registry, layer)?;
                for projection in ["q_norm", "k_norm"] {
                    require_tensor_shape(
                        registry,
                        &layer_norm_tensor_name(layer, &format!("self_attn.{projection}")),
                        &[layout.head_dim],
                    )?;
                }
            }
            AttentionLayerType::Mla => {
                let _ = infer_mla_attention_layout(config, registry, layer)?;
            }
            AttentionLayerType::Linear => {
                let _ = infer_linear_attention_layout(config, registry, layer)?;
            }
        }
        require_tensor_shape(
            registry,
            &layer_norm_tensor_name(layer, "input_layernorm"),
            &[config.hidden_size],
        )?;
        require_tensor_shape(
            registry,
            &layer_norm_tensor_name(layer, "post_attention_layernorm"),
            &[config.hidden_size],
        )?;
        if config.is_dense_mlp_layer(layer) {
            let intermediate = config.intermediate_size.with_context(|| {
                format!("GLM dense MLP layer {layer} is missing intermediate_size")
            })?;
            for (projection, shape) in [
                ("gate_proj", [intermediate, config.hidden_size]),
                ("up_proj", [intermediate, config.hidden_size]),
                ("down_proj", [config.hidden_size, intermediate]),
            ] {
                require_tensor_shape(
                    registry,
                    &format!("model.layers.{layer}.mlp.{projection}.weight"),
                    &shape,
                )?;
            }
            continue;
        }
        require_tensor_shape(
            registry,
            &router_tensor_name(layer),
            &[config.experts(), config.hidden_size],
        )?;
        if config.glm.is_some() {
            require_tensor_shape(
                registry,
                &format!("model.layers.{layer}.mlp.gate.e_score_correction_bias"),
                &[config.experts()],
            )?;
        }
        let shared_experts = config.shared_experts();
        if shared_experts > 0 {
            let shared_inter = config.shared_expert_intermediate_size();
            if shared_inter == 0 {
                bail!(
                    "Qwen config declares {shared_experts} shared expert(s) but no shared expert intermediate size"
                );
            }
            let total_shared_inter = shared_experts
                .checked_mul(shared_inter)
                .context("shared expert intermediate size overflow")?;
            require_tensor_shape(
                registry,
                &shared_expert_tensor_name(layer, "gate_proj"),
                &[total_shared_inter, config.hidden_size],
            )?;
            require_tensor_shape(
                registry,
                &shared_expert_tensor_name(layer, "up_proj"),
                &[total_shared_inter, config.hidden_size],
            )?;
            require_tensor_shape(
                registry,
                &shared_expert_tensor_name(layer, "down_proj"),
                &[config.hidden_size, total_shared_inter],
            )?;
            if config.glm.is_none() {
                require_tensor_shape(
                    registry,
                    &shared_expert_gate_tensor_name(layer),
                    &[shared_experts, config.hidden_size],
                )?;
            }
        }
        // Per-expert tensor presence is intentionally not validated here.
        //
        // Reasons:
        // 1. Expert MLP correctness (gate/up/down projection shapes) is enforced per-expert at
        //    pack time by `validate_expert_tensor_group`.
        // 2. At runtime the packed expert files are managed by `ExpertSlotStore`; the registry
        //    records their original source metadata but is not used for expert inference.
        // 3. Real Qwen3 revision checkpoints may differ in expert naming (e.g. shared experts)
        //    or use a naming scheme that doesn't match the exact pattern assumed here.
        //    A rigid per-name loop would cause false rejections for such models.
    }
    Ok(())
}

pub(super) fn validate_qwen_q4_graph_bindings(
    family: QwenMoeFamily,
    config: &QwenModelConfig,
    runtime: &DenseTransformerRuntime,
    registry: &TensorRegistry,
    store_len: u64,
) -> Result<()> {
    for layer in 0..config.num_hidden_layers {
        if runtime.is_mla_attention_layer(layer) {
            let layout = runtime.mla_attention_layout(layer)?;
            for (projection, output_width, input_width) in [
                ("q_a_proj", layout.q_lora_rank, runtime.width),
                ("q_b_proj", layout.q_width, layout.q_lora_rank),
                ("kv_a_proj_with_mqa", layout.kv_a_width, runtime.width),
                ("o_proj", runtime.width, layout.attention_output_width),
            ] {
                require_resident_q4_graph_projection(
                    family,
                    registry,
                    store_len,
                    "MLA projection",
                    &attention_tensor_name(layer, projection),
                    output_width,
                    input_width,
                )?;
            }
            match layout.kv_projection {
                MlaKvProjectionLayout::FusedKvB => {
                    require_resident_q4_graph_projection(
                        family,
                        registry,
                        store_len,
                        "MLA KV-B projection",
                        &attention_tensor_name(layer, "kv_b_proj"),
                        layout.kv_b_width,
                        layout.kv_lora_rank,
                    )?;
                }
                MlaKvProjectionLayout::AbsorbedMultiLinear => {
                    require_resident_q4_multilinear_projection(
                        family,
                        registry,
                        store_len,
                        "MLA absorbed query projection",
                        &attention_tensor_name(layer, "embed_q"),
                        layout.num_heads,
                        layout.kv_lora_rank,
                        layout.qk_nope_head_dim,
                    )?;
                    require_resident_q4_multilinear_projection(
                        family,
                        registry,
                        store_len,
                        "MLA absorbed output projection",
                        &attention_tensor_name(layer, "unembed_out"),
                        layout.num_heads,
                        layout.v_head_dim,
                        layout.kv_lora_rank,
                    )?;
                }
            }
        } else if !runtime.is_linear_attention_layer(layer) {
            let layout = runtime.full_attention_layout(layer)?;
            let requests = full_attention_input_projection_requests(
                layer,
                layout.q_projection_width,
                layout.kv_width,
            )?;
            for request in requests.requests() {
                require_resident_q4_graph_projection(
                    family,
                    registry,
                    store_len,
                    "CMD1 full-attention projection",
                    request.tensor_name,
                    request.output_width,
                    runtime.width,
                )?;
            }
            require_resident_q4_graph_projection(
                family,
                registry,
                store_len,
                "CMD2 full-attention output projection",
                &attention_tensor_name(layer, "o_proj"),
                runtime.width,
                layout.num_q_heads * layout.head_dim,
            )?;
        }

        if config.is_dense_mlp_layer(layer) {
            let intermediate = config.intermediate_size.with_context(|| {
                format!("GLM dense MLP layer {layer} is missing intermediate_size")
            })?;
            for (projection, output_width, input_width) in [
                ("gate_proj", intermediate, runtime.width),
                ("up_proj", intermediate, runtime.width),
                ("down_proj", runtime.width, intermediate),
            ] {
                require_resident_q4_graph_projection(
                    family,
                    registry,
                    store_len,
                    "dense lead-in MLP projection",
                    &format!("model.layers.{layer}.mlp.{projection}.weight"),
                    output_width,
                    input_width,
                )?;
            }
        } else {
            if config.glm.is_some() || config.is_qwen3_next() {
                require_resident_graph_projection(
                    family,
                    registry,
                    store_len,
                    "CMD2 router projection",
                    &router_tensor_name(layer),
                    config.experts(),
                    runtime.width,
                )?;
            } else {
                require_resident_q4_graph_projection(
                    family,
                    registry,
                    store_len,
                    "CMD2 router projection",
                    &router_tensor_name(layer),
                    config.experts(),
                    runtime.width,
                )?;
            }
        }
    }

    let lm_head_name = if registry.tensor("lm_head.weight").is_some() {
        "lm_head.weight"
    } else {
        "model.embed_tokens.weight"
    };
    if config.glm.is_some() {
        require_resident_graph_projection(
            family,
            registry,
            store_len,
            "GLM LM-head sampling projection",
            lm_head_name,
            config.vocab_size,
            runtime.width,
        )?;
    } else {
        require_resident_q4_graph_projection(
            family,
            registry,
            store_len,
            "LM-head sampling projection",
            lm_head_name,
            config.vocab_size,
            runtime.width,
        )?;
    }
    Ok(())
}

pub(super) fn require_resident_graph_projection(
    family: QwenMoeFamily,
    registry: &TensorRegistry,
    store_len: u64,
    stage: &str,
    tensor_name: &str,
    output_width: usize,
    input_width: usize,
) -> Result<()> {
    let entry = registry.require(tensor_name)?;
    ResidentMmapMatvecProjection::from_entry(
        tensor_name,
        entry,
        store_len,
        output_width,
        input_width,
    )
    .with_context(|| {
        format!(
            "FlashMoe unsupported resolved {family:?} {stage}: tensor {tensor_name} cannot bind resident shape {output_width}x{input_width}"
        )
    })?;
    Ok(())
}

pub(super) fn require_resident_q4_graph_projection(
    family: QwenMoeFamily,
    registry: &TensorRegistry,
    store_len: u64,
    stage: &str,
    tensor_name: &str,
    output_width: usize,
    input_width: usize,
) -> Result<()> {
    let entry = registry.require(tensor_name)?;
    DenseQ4MmapMatvecProjection::from_entry(
        tensor_name,
        entry,
        store_len,
        output_width,
        input_width,
    )?
    .with_context(|| {
        format!(
            "FlashMoe unsupported resolved {family:?} Q4 {stage}: tensor {tensor_name} cannot bind the resident projection for shape {output_width}x{input_width}"
        )
    })?;
    Ok(())
}

pub(super) fn require_resident_q4_multilinear_projection(
    family: QwenMoeFamily,
    registry: &TensorRegistry,
    store_len: u64,
    stage: &str,
    tensor_name: &str,
    heads: usize,
    output_width_per_head: usize,
    input_width: usize,
) -> Result<()> {
    let entry = registry.require(tensor_name)?;
    DenseQ4MmapMatvecProjection::from_multilinear_entry(
        tensor_name,
        entry,
        store_len,
        heads,
        output_width_per_head,
        input_width,
    )?
    .with_context(|| {
        format!(
            "FlashMoe unsupported resolved {family:?} Q4 {stage}: tensor {tensor_name} cannot bind the resident multilinear projection for shape {heads}x{output_width_per_head}x{input_width}"
        )
    })?;
    Ok(())
}

#[derive(Debug, Clone)]
pub(super) struct DenseTransformerRuntime {
    pub(super) width: usize,
    #[cfg(test)]
    pub(super) head_dim: usize,
    #[cfg(test)]
    pub(super) num_q_heads: usize,
    #[cfg(test)]
    pub(super) kv_heads: usize,
    pub(super) full_attention: Vec<Option<FullAttentionLayout>>,
    pub(super) mla_attention: Vec<Option<MlaAttentionLayout>>,
    pub(super) linear_attention: Vec<Option<LinearAttentionLayout>>,
}

impl DenseTransformerRuntime {
    pub(super) fn new(config: &QwenModelConfig) -> Self {
        #[cfg(test)]
        let head_dim = config.full_attention_head_dim();
        #[cfg(test)]
        let kv_heads = config.kv_heads();
        Self {
            width: config.hidden_size,
            #[cfg(test)]
            head_dim,
            #[cfg(test)]
            num_q_heads: config.num_attention_heads,
            #[cfg(test)]
            kv_heads,
            full_attention: vec![None; config.num_hidden_layers],
            mla_attention: vec![None; config.num_hidden_layers],
            linear_attention: vec![None; config.num_hidden_layers],
        }
    }

    pub(super) fn from_registry(
        config: &QwenModelConfig,
        registry: &TensorRegistry,
    ) -> Result<Self> {
        let mut runtime = Self::new(config);
        for layer in 0..config.num_hidden_layers {
            match infer_attention_layer_type(registry, layer)? {
                AttentionLayerType::Full => {
                    runtime.full_attention[layer] =
                        Some(infer_full_attention_layout(config, registry, layer)?);
                }
                AttentionLayerType::Mla => {
                    runtime.mla_attention[layer] =
                        Some(infer_mla_attention_layout(config, registry, layer)?);
                }
                AttentionLayerType::Linear => {
                    runtime.linear_attention[layer] =
                        Some(infer_linear_attention_layout(config, registry, layer)?);
                }
            }
        }

        Ok(runtime)
    }

    pub(super) fn full_attention_layout(&self, layer: usize) -> Result<FullAttentionLayout> {
        self.full_attention
            .get(layer)
            .copied()
            .flatten()
            .with_context(|| format!("missing full-attention runtime layout for layer {layer}"))
    }

    pub(super) fn linear_attention_layout(&self, layer: usize) -> Result<LinearAttentionLayout> {
        self.linear_attention
            .get(layer)
            .copied()
            .flatten()
            .with_context(|| format!("missing linear-attention runtime layout for layer {layer}"))
    }

    pub(super) fn mla_attention_layout(&self, layer: usize) -> Result<MlaAttentionLayout> {
        self.mla_attention
            .get(layer)
            .copied()
            .flatten()
            .with_context(|| format!("missing MLA runtime layout for layer {layer}"))
    }

    pub(super) fn is_mla_attention_layer(&self, layer: usize) -> bool {
        self.mla_attention
            .get(layer)
            .and_then(|layout| *layout)
            .is_some()
    }

    pub(super) fn is_linear_attention_layer(&self, layer: usize) -> bool {
        self.linear_attention
            .get(layer)
            .and_then(|layout| *layout)
            .is_some()
    }

    pub(super) fn resolved_attention_layers(&self) -> Result<Vec<QwenMoeLayerKind>> {
        self.full_attention
            .iter()
            .zip(&self.mla_attention)
            .zip(&self.linear_attention)
            .enumerate()
            .map(|(layer, ((full, mla), linear))| match (
                full.is_some(),
                mla.is_some(),
                linear.is_some(),
            ) {
                (true, false, false) | (false, true, false) => {
                    Ok(QwenMoeLayerKind::FullAttention)
                }
                (false, false, true) => Ok(QwenMoeLayerKind::LinearAttention),
                _ => bail!(
                    "FlashMoe dense runtime layer {layer} must resolve exactly one attention implementation"
                ),
            })
            .collect()
    }

    #[cfg(test)]
    pub(super) fn layer_kind(&self, layer: usize) -> FlashMoeLayerKind {
        if self.is_linear_attention_layer(layer) {
            FlashMoeLayerKind::LinearAttention
        } else if self.is_mla_attention_layer(layer)
            || self
                .full_attention
                .get(layer)
                .and_then(|layout| *layout)
                .is_some()
        {
            FlashMoeLayerKind::FullAttention
        } else {
            FlashMoeLayerKind::Unknown
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttentionLayerType {
    Full,
    Mla,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MlaKvProjectionLayout {
    FusedKvB,
    AbsorbedMultiLinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MlaAttentionLayout {
    pub(super) q_lora_rank: usize,
    pub(super) kv_lora_rank: usize,
    pub(super) qk_nope_head_dim: usize,
    pub(super) qk_rope_head_dim: usize,
    pub(super) qk_head_dim: usize,
    pub(super) v_head_dim: usize,
    pub(super) num_heads: usize,
    pub(super) q_width: usize,
    pub(super) kv_a_width: usize,
    pub(super) kv_b_width: usize,
    pub(super) attention_output_width: usize,
    pub(super) kv_projection: MlaKvProjectionLayout,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(super) struct GlmMlaPostAttentionRequest<'a> {
    pub(super) residual: MetalBatchProjectionInput<'a>,
    pub(super) post_norm_weight: &'a [f32],
    pub(super) router_correction_bias: Option<&'a [f32]>,
    pub(super) experts: usize,
    pub(super) active_experts: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FullAttentionQLayout {
    Standard,
    Gated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RotaryPairing {
    #[allow(dead_code)]
    Adjacent,
    SplitHalf,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FullAttentionLayout {
    pub(super) q_layout: FullAttentionQLayout,
    pub(super) q_projection_width: usize,
    pub(super) q_width: usize,
    pub(super) kv_width: usize,
    pub(super) head_dim: usize,
    pub(super) rotary_dim: usize,
    pub(super) num_q_heads: usize,
    pub(super) kv_heads: usize,
    pub(super) rotary_pairing: RotaryPairing,
}

pub(super) fn infer_linear_attention_layout(
    config: &QwenModelConfig,
    registry: &TensorRegistry,
    layer: usize,
) -> Result<LinearAttentionLayout> {
    let qkv_name = linear_attention_tensor_name(layer, "in_proj_qkv");
    let z_name = linear_attention_tensor_name(layer, "in_proj_z");
    let b_name = linear_attention_tensor_name(layer, "in_proj_b");
    let a_name = linear_attention_tensor_name(layer, "in_proj_a");
    let conv_name = linear_attention_tensor_name(layer, "conv1d");
    let a_log_name = linear_attention_scalar_tensor_name(layer, "A_log");
    let dt_bias_name = linear_attention_scalar_tensor_name(layer, "dt_bias");
    let norm_name = linear_attention_tensor_name(layer, "norm");
    let out_proj_name = linear_attention_tensor_name(layer, "out_proj");

    let (qkv_rows, qkv_cols) = require_2d_tensor_shape(registry, &qkv_name)?;
    let (z_rows, z_cols) = require_2d_tensor_shape(registry, &z_name)?;
    let (b_rows, b_cols) = require_2d_tensor_shape(registry, &b_name)?;
    let (a_rows, a_cols) = require_2d_tensor_shape(registry, &a_name)?;
    let (out_rows, out_cols) = require_2d_tensor_shape(registry, &out_proj_name)?;
    let a_log_len = require_1d_tensor_shape(registry, &a_log_name)?;
    let dt_bias_len = require_1d_tensor_shape(registry, &dt_bias_name)?;
    let value_dim = require_1d_tensor_shape(registry, &norm_name)?;
    let (conv_channels, conv_kernel_size) = require_conv1d_tensor_shape(registry, &conv_name)?;
    if conv_kernel_size < 2 {
        bail!(
            "linear-attention layer {layer} conv1d kernel size {conv_kernel_size} must be at least 2"
        );
    }

    if qkv_cols != config.hidden_size
        || z_cols != config.hidden_size
        || b_cols != config.hidden_size
        || a_cols != config.hidden_size
    {
        bail!(
            "linear-attention layer {layer} projection input widths are qkv={qkv_cols}, z={z_cols}, b={b_cols}, a={a_cols}; expected hidden_size {}",
            config.hidden_size
        );
    }
    if out_rows != config.hidden_size {
        bail!(
            "linear-attention layer {layer} out_proj output rows {out_rows}; expected hidden_size {}",
            config.hidden_size
        );
    }
    if z_rows != out_cols {
        bail!(
            "linear-attention layer {layer} z width {z_rows} does not match out_proj input width {out_cols}"
        );
    }
    if value_dim == 0 || z_rows % value_dim != 0 {
        bail!(
            "linear-attention layer {layer} value width {z_rows} is not divisible by norm/value dim {value_dim}"
        );
    }
    let num_value_heads = z_rows / value_dim;
    if b_rows != num_value_heads
        || a_rows != num_value_heads
        || a_log_len != num_value_heads
        || dt_bias_len != num_value_heads
    {
        bail!(
            "linear-attention layer {layer} value head counts disagree: inferred={num_value_heads}, b={b_rows}, a={a_rows}, A_log={a_log_len}, dt_bias={dt_bias_len}"
        );
    }
    if qkv_rows < z_rows {
        bail!(
            "linear-attention layer {layer} qkv width {qkv_rows} is smaller than value width {z_rows}"
        );
    }
    let paired_key_width = qkv_rows - z_rows;
    if paired_key_width % 2 != 0 {
        bail!(
            "linear-attention layer {layer} qkv non-value width {paired_key_width} cannot split evenly into Q and K"
        );
    }
    let total_key_width = paired_key_width / 2;
    let key_dim = infer_linear_attention_key_dim(config, total_key_width, z_rows, value_dim)
        .with_context(|| {
            format!(
                "linear-attention layer {layer} cannot infer key dimension from key width {total_key_width}, value width {z_rows}, value dim {value_dim}"
            )
        })?;
    let num_key_heads = total_key_width / key_dim;
    if num_key_heads == 0 {
        bail!("linear-attention layer {layer} inferred zero key heads");
    }
    if num_value_heads % num_key_heads != 0 {
        bail!(
            "linear-attention layer {layer} value heads {num_value_heads} must be divisible by key heads {num_key_heads}"
        );
    }
    if conv_channels != qkv_rows {
        bail!(
            "linear-attention layer {layer} conv1d channels {conv_channels} do not match qkv width {qkv_rows}"
        );
    }

    Ok(LinearAttentionLayout {
        num_value_heads,
        num_key_heads,
        key_dim,
        value_dim,
        total_key_width,
        total_value_width: z_rows,
        conv_dim: qkv_rows,
        conv_kernel_size,
    })
}

pub(super) fn infer_linear_attention_key_dim(
    config: &QwenModelConfig,
    total_key_width: usize,
    total_value_width: usize,
    value_dim: usize,
) -> Result<usize> {
    let config_head_dim = config.hidden_size / config.num_attention_heads.max(1);
    if config_head_dim > 0 && total_key_width.is_multiple_of(config_head_dim) {
        return Ok(config_head_dim);
    }
    if is_known_qwen35_linear_attention_shape(total_key_width, total_value_width, value_dim) {
        return Ok(LINEAR_KEY_DIM);
    }
    if total_key_width.is_multiple_of(value_dim) {
        return Ok(value_dim);
    }
    bail!(
        "key width is not divisible by config head_dim {config_head_dim} or manifest value_dim {value_dim}"
    )
}

fn is_known_qwen35_linear_attention_shape(
    total_key_width: usize,
    total_value_width: usize,
    value_dim: usize,
) -> bool {
    total_key_width == LINEAR_TOTAL_KEY
        && total_value_width == LINEAR_TOTAL_VALUE
        && value_dim == LINEAR_VALUE_DIM
}

pub(super) fn infer_full_attention_layout(
    config: &QwenModelConfig,
    registry: &TensorRegistry,
    layer: usize,
) -> Result<FullAttentionLayout> {
    let q_name = attention_tensor_name(layer, "q_proj");
    let k_name = attention_tensor_name(layer, "k_proj");
    let v_name = attention_tensor_name(layer, "v_proj");
    let o_name = attention_tensor_name(layer, "o_proj");

    let (q_rows, q_cols) = require_2d_tensor_shape(registry, &q_name)?;
    let (k_rows, k_cols) = require_2d_tensor_shape(registry, &k_name)?;
    let (v_rows, v_cols) = require_2d_tensor_shape(registry, &v_name)?;
    let (o_rows, o_cols) = require_2d_tensor_shape(registry, &o_name)?;

    let num_q_heads = config.num_attention_heads;
    let kv_heads = config.kv_heads();

    if q_cols != config.hidden_size || k_cols != config.hidden_size || v_cols != config.hidden_size
    {
        bail!(
            "full-attention layer {layer} projection input widths are q={q_cols}, k={k_cols}, v={v_cols}; expected hidden_size {}",
            config.hidden_size
        );
    }
    if o_rows != config.hidden_size {
        bail!(
            "full-attention layer {layer} o_proj output rows {o_rows}; expected hidden_size {}",
            config.hidden_size
        );
    }
    if k_rows != v_rows {
        bail!(
            "full-attention layer {layer} k_proj rows {k_rows} do not match v_proj rows {v_rows}"
        );
    }
    if k_rows == 0 || k_rows % kv_heads != 0 {
        bail!(
            "full-attention layer {layer} k/v width {k_rows} is not divisible by kv_heads {kv_heads}"
        );
    }

    let head_dim = k_rows / kv_heads;
    let q_width = num_q_heads
        .checked_mul(head_dim)
        .context("full-attention q_width overflow")?;
    let gated_q_width = q_width
        .checked_mul(2)
        .context("gated full-attention q_width overflow")?;

    if q_rows == q_width && o_cols == q_width {
        return Ok(FullAttentionLayout {
            q_layout: FullAttentionQLayout::Standard,
            q_projection_width: q_width,
            q_width,
            kv_width: k_rows,
            head_dim,
            rotary_dim: rotary_dim_for(config, head_dim, FullAttentionQLayout::Standard),
            num_q_heads,
            kv_heads,
            rotary_pairing: RotaryPairing::SplitHalf,
        });
    }

    if q_rows == gated_q_width && o_cols == q_width {
        return Ok(FullAttentionLayout {
            q_layout: FullAttentionQLayout::Gated,
            q_projection_width: gated_q_width,
            q_width,
            kv_width: k_rows,
            head_dim,
            rotary_dim: rotary_dim_for(config, head_dim, FullAttentionQLayout::Gated),
            num_q_heads,
            kv_heads,
            rotary_pairing: RotaryPairing::SplitHalf,
        });
    }

    bail!(
        "unsupported full-attention layer {layer} layout: q_proj rows={q_rows}, k/v rows={k_rows}, o_proj shape=[{o_rows},{o_cols}], num_q_heads={num_q_heads}, kv_heads={kv_heads}, inferred head_dim={head_dim}; expected standard q_rows={q_width}, o_cols={q_width}, or gated q_rows={gated_q_width}, o_cols={q_width}"
    )
}

pub(super) fn infer_mla_attention_layout(
    config: &QwenModelConfig,
    registry: &TensorRegistry,
    layer: usize,
) -> Result<MlaAttentionLayout> {
    let glm = config
        .glm
        .as_ref()
        .with_context(|| format!("MLA tensors at layer {layer} require a GLM config"))?;
    let qk_head_dim = glm
        .qk_nope_head_dim
        .checked_add(glm.qk_rope_head_dim)
        .context("MLA q/k head width overflow")?;
    let q_width = config
        .num_attention_heads
        .checked_mul(qk_head_dim)
        .context("MLA query width overflow")?;
    let kv_a_width = glm
        .kv_lora_rank
        .checked_add(glm.qk_rope_head_dim)
        .context("MLA compressed KV width overflow")?;
    let kv_b_head_width = glm
        .qk_nope_head_dim
        .checked_add(glm.v_head_dim)
        .context("MLA KV-B head width overflow")?;
    let kv_b_width = config
        .num_attention_heads
        .checked_mul(kv_b_head_width)
        .context("MLA KV-B width overflow")?;
    let attention_output_width = config
        .num_attention_heads
        .checked_mul(glm.v_head_dim)
        .context("MLA attention output width overflow")?;

    let expected = [
        ("q_a_proj", glm.q_lora_rank, config.hidden_size),
        ("q_b_proj", q_width, glm.q_lora_rank),
        ("kv_a_proj_with_mqa", kv_a_width, config.hidden_size),
        ("o_proj", config.hidden_size, attention_output_width),
    ];
    for (projection, expected_rows, expected_cols) in expected {
        let tensor_name = attention_tensor_name(layer, projection);
        let (rows, cols) = require_2d_tensor_shape(registry, &tensor_name)?;
        if rows != expected_rows || cols != expected_cols {
            bail!(
                "MLA layer {layer} projection {tensor_name} has shape [{rows},{cols}], expected [{expected_rows},{expected_cols}]"
            );
        }
    }
    let kv_b_name = attention_tensor_name(layer, "kv_b_proj");
    let embed_q_name = attention_tensor_name(layer, "embed_q");
    let unembed_out_name = attention_tensor_name(layer, "unembed_out");
    let kv_projection = if registry.tensor(&kv_b_name).is_some() {
        require_tensor_shape(registry, &kv_b_name, &[kv_b_width, glm.kv_lora_rank])?;
        MlaKvProjectionLayout::FusedKvB
    } else if registry.tensor(&embed_q_name).is_some()
        && registry.tensor(&unembed_out_name).is_some()
    {
        require_tensor_shape(
            registry,
            &embed_q_name,
            &[
                config.num_attention_heads,
                glm.kv_lora_rank,
                glm.qk_nope_head_dim,
            ],
        )?;
        require_tensor_shape(
            registry,
            &unembed_out_name,
            &[config.num_attention_heads, glm.v_head_dim, glm.kv_lora_rank],
        )?;
        MlaKvProjectionLayout::AbsorbedMultiLinear
    } else {
        bail!(
            "MLA layer {layer} is missing {kv_b_name} or the absorbed pair {embed_q_name} and {unembed_out_name}"
        );
    };
    require_tensor_shape(
        registry,
        &layer_norm_tensor_name(layer, "self_attn.q_a_layernorm"),
        &[glm.q_lora_rank],
    )?;
    require_tensor_shape(
        registry,
        &layer_norm_tensor_name(layer, "self_attn.kv_a_layernorm"),
        &[glm.kv_lora_rank],
    )?;

    Ok(MlaAttentionLayout {
        q_lora_rank: glm.q_lora_rank,
        kv_lora_rank: glm.kv_lora_rank,
        qk_nope_head_dim: glm.qk_nope_head_dim,
        qk_rope_head_dim: glm.qk_rope_head_dim,
        qk_head_dim,
        v_head_dim: glm.v_head_dim,
        num_heads: config.num_attention_heads,
        q_width,
        kv_a_width,
        kv_b_width,
        attention_output_width,
        kv_projection,
    })
}

pub(super) fn rotary_dim_for(
    config: &QwenModelConfig,
    head_dim: usize,
    q_layout: FullAttentionQLayout,
) -> usize {
    let factor = config.partial_rotary_factor.unwrap_or_else(|| {
        if q_layout == FullAttentionQLayout::Gated {
            0.25
        } else {
            1.0
        }
    });

    let mut rotary_dim = ((head_dim as f64) * factor).round() as usize;
    rotary_dim = rotary_dim.clamp(2, head_dim);
    rotary_dim - (rotary_dim % 2)
}

pub(super) fn require_2d_tensor_shape(
    registry: &TensorRegistry,
    canonical_name: &str,
) -> Result<(usize, usize)> {
    let tensor = registry.require(canonical_name)?;
    ensure_runtime_tensor_storage_supported(canonical_name, tensor)?;
    match tensor.shape.as_slice() {
        [rows, cols] if *rows > 0 && *cols > 0 => Ok((*rows, *cols)),
        shape => bail!(
            "Flash-MoE tensor {canonical_name} has shape {:?}; expected non-empty 2-D matrix",
            shape
        ),
    }
}

pub(super) fn require_tensor_shape(
    registry: &TensorRegistry,
    canonical_name: &str,
    expected_shape: &[usize],
) -> Result<()> {
    let tensor = registry.require(canonical_name)?;
    ensure_runtime_tensor_storage_supported(canonical_name, tensor)?;
    if tensor.shape.as_slice() != expected_shape {
        bail!(
            "Flash-MoE tensor {canonical_name} has shape {:?}; expected {:?}",
            tensor.shape,
            expected_shape
        );
    }
    Ok(())
}

pub(super) fn require_1d_tensor_shape(
    registry: &TensorRegistry,
    canonical_name: &str,
) -> Result<usize> {
    let tensor = registry.require(canonical_name)?;
    ensure_runtime_tensor_storage_supported(canonical_name, tensor)?;
    match tensor.shape.as_slice() {
        [width] if *width > 0 => Ok(*width),
        shape => bail!(
            "Flash-MoE tensor {canonical_name} has shape {:?}; expected non-empty 1-D vector",
            shape
        ),
    }
}

pub(super) fn require_conv1d_tensor_shape(
    registry: &TensorRegistry,
    canonical_name: &str,
) -> Result<(usize, usize)> {
    let tensor = registry.require(canonical_name)?;
    ensure_runtime_tensor_storage_supported(canonical_name, tensor)?;
    match tensor.shape.as_slice() {
        [channels, kernel_size] if *channels > 0 && *kernel_size > 0 => {
            Ok((*channels, *kernel_size))
        }
        [channels, 1, kernel_size] if *channels > 0 && *kernel_size > 0 => {
            Ok((*channels, *kernel_size))
        }
        [channels, kernel_size, 1] if *channels > 0 && *kernel_size > 0 => {
            Ok((*channels, *kernel_size))
        }
        shape => bail!(
            "Flash-MoE tensor {canonical_name} has shape {:?}; expected non-empty [channels, kernel], [channels, 1, kernel], or [channels, kernel, 1]",
            shape
        ),
    }
}

pub(super) fn ensure_runtime_tensor_storage_supported(
    canonical_name: &str,
    tensor: &RuntimeTensorEntry,
) -> Result<()> {
    match &tensor.quantization {
        TensorQuantization::None => {
            if dtype_size(&tensor.dtype).is_none() {
                bail!(
                    "Flash-MoE tensor {canonical_name} has unsupported dtype {}",
                    tensor.dtype
                );
            }
        }
        TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } => {
            if !(tensor.dtype.eq_ignore_ascii_case("U32")
                || tensor.dtype.eq_ignore_ascii_case("U8"))
            {
                bail!(
                    "Flash-MoE q4 tensor {canonical_name} has unsupported packed dtype {}",
                    tensor.dtype
                );
            }
            dense_q4_layout_with_scale_bias_dtype(&tensor.shape, *group_size, scale_bias_dtype)
                .with_context(|| {
                    format!("Flash-MoE q4 tensor {canonical_name} has unsupported runtime layout")
                })?;
        }
        TensorQuantization::Gguf {
            block_elements,
            block_bytes,
            ..
        } => {
            if *block_elements == 0 || *block_bytes == 0 {
                bail!(
                    "Flash-MoE GGUF tensor {canonical_name} has an invalid zero-sized block layout"
                );
            }
            let elements = tensor.shape.iter().try_fold(1u64, |count, dimension| {
                count.checked_mul(*dimension as u64)
            });
            let expected = elements
                .and_then(|elements| elements.checked_add(*block_elements - 1))
                .map(|elements| elements / *block_elements)
                .and_then(|blocks| blocks.checked_mul(*block_bytes))
                .context("Flash-MoE GGUF tensor byte length overflow")?;
            if expected != tensor.byte_len {
                bail!(
                    "Flash-MoE GGUF tensor {canonical_name} has {} bytes, expected {expected}",
                    tensor.byte_len
                );
            }
        }
    }
    Ok(())
}

pub(super) fn infer_attention_layer_type(
    registry: &TensorRegistry,
    layer: usize,
) -> Result<AttentionLayerType> {
    let linear_prefix = format!("model.layers.{layer}.linear_attn.");
    let has_linear = registry.has_tensor_with_prefix(&linear_prefix);
    let has_mla = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"]
        .iter()
        .any(|projection| {
            registry
                .tensor(&attention_tensor_name(layer, projection))
                .is_some()
        });
    let has_full = ["q_proj", "k_proj", "v_proj"].iter().any(|projection| {
        registry
            .tensor(&attention_tensor_name(layer, projection))
            .is_some()
    });

    match (has_linear, has_mla, has_full) {
        (true, false, true) => bail!(
            "Flash-MoE tensor manifest has both linear-attention tensors ({linear_prefix}*) and full-attention self_attn projection tensors for layer {layer}"
        ),
        (true, true, _) | (_, true, true) => bail!(
            "Flash-MoE tensor manifest resolves more than one attention implementation for layer {layer}"
        ),
        (true, false, false) => Ok(AttentionLayerType::Linear),
        (false, true, false) => Ok(AttentionLayerType::Mla),
        (false, false, true) => Ok(AttentionLayerType::Full),
        (false, false, false) => bail!(
            "Flash-MoE tensor manifest is missing attention tensors for layer {layer}; expected linear attention, MLA, or self_attn.{{q,k,v,o}}_proj tensors"
        ),
    }
}

mod manifest;
pub use manifest::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RouterScoreProjectionBinding {
    ResidentDense(DenseMmapMatvecProjection),
    ResidentQ4(DenseQ4MmapMatvecProjection),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RouterScoreProjectionDescriptor {
    pub(crate) layer: usize,
    pub(crate) tensor_name: String,
    pub(crate) experts: usize,
    pub(crate) hidden_width: usize,
    pub(crate) binding: RouterScoreProjectionBinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterScoreProjectionExecutionKind {
    ResidentDense,
    ResidentQ4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RouterScoreProjectionExecution<'a> {
    pub(crate) layer: usize,
    pub(crate) tensor_name: &'a str,
    pub(crate) experts: usize,
    pub(crate) hidden_width: usize,
    pub(crate) kind: RouterScoreProjectionExecutionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterScoreProjectionScoreSource {
    ResidentDenseFullTensor,
    DeclaredRows,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RouterScoreProjectionScorePlan<'a> {
    pub(crate) tensor_name: &'a str,
    pub(crate) experts: usize,
    pub(crate) hidden_width: usize,
    pub(crate) source: RouterScoreProjectionScoreSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum RouterScoreProjectionTopKSource {
    ResidentDense(DenseMmapMatvecProjection),
    ResidentQ4(DenseQ4MmapMatvecProjection),
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct RouterScoreProjectionTopKPlan {
    pub(crate) layer: usize,
    pub(crate) tensor_name: String,
    pub(crate) experts: usize,
    pub(crate) hidden_width: usize,
    pub(crate) active_experts: usize,
    pub(crate) source: RouterScoreProjectionTopKSource,
}

impl<'a> RouterScoreProjectionExecution<'a> {
    pub(crate) fn score_plan(
        self,
        hidden_len: usize,
    ) -> Result<RouterScoreProjectionScorePlan<'a>> {
        if self.hidden_width != hidden_len {
            bail!(
                "Flash-MoE router score projection hidden length {} does not match declared width {}",
                hidden_len,
                self.hidden_width
            );
        }
        let source = match self.kind {
            RouterScoreProjectionExecutionKind::ResidentDense => {
                RouterScoreProjectionScoreSource::ResidentDenseFullTensor
            }
            RouterScoreProjectionExecutionKind::ResidentQ4 => {
                RouterScoreProjectionScoreSource::DeclaredRows
            }
        };
        Ok(RouterScoreProjectionScorePlan {
            tensor_name: self.tensor_name,
            experts: self.experts,
            hidden_width: self.hidden_width,
            source,
        })
    }
}

impl RouterScoreProjectionDescriptor {
    pub(crate) fn from_entry(
        layer: usize,
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        experts: usize,
        hidden_width: usize,
    ) -> Result<Self> {
        match &entry.quantization {
            TensorQuantization::None => {
                let Some(element_size) = dense_dtype_size(&entry.dtype) else {
                    bail!(
                        "Flash-MoE router tensor {} has unsupported dtype {}",
                        tensor_name,
                        entry.dtype
                    );
                };
                let projection = DenseMmapMatvecProjection::from_entry(
                    tensor_name,
                    entry,
                    store_len,
                    experts,
                    hidden_width,
                    element_size,
                )?;
                Ok(Self {
                    layer,
                    tensor_name: tensor_name.to_string(),
                    experts,
                    hidden_width,
                    binding: RouterScoreProjectionBinding::ResidentDense(projection),
                })
            }
            TensorQuantization::Q4 { .. } => {
                let Some(projection) = DenseQ4MmapMatvecProjection::from_entry(
                    tensor_name,
                    entry,
                    store_len,
                    experts,
                    hidden_width,
                )?
                else {
                    bail!(
                        "Flash-MoE router tensor {tensor_name} cannot resolve a resident Q4 projection descriptor for shape [{experts}, {hidden_width}]"
                    );
                };
                Ok(Self {
                    layer,
                    tensor_name: tensor_name.to_string(),
                    experts,
                    hidden_width,
                    binding: RouterScoreProjectionBinding::ResidentQ4(projection),
                })
            }
            TensorQuantization::Gguf { .. } => bail!(
                "Flash-MoE router tensor {tensor_name} requires a model-family GGUF graph binding"
            ),
        }
    }

    pub(crate) fn execution(
        &self,
        layer: usize,
        experts: usize,
        hidden_width: usize,
    ) -> Result<RouterScoreProjectionExecution<'_>> {
        if self.layer != layer {
            bail!(
                "Flash-MoE router score projection execution layer {} does not match scheduled layer {}",
                self.layer,
                layer
            );
        }
        if self.experts != experts {
            bail!(
                "Flash-MoE router score projection execution experts {} does not match scheduled experts {}",
                self.experts,
                experts
            );
        }
        if self.hidden_width != hidden_width {
            bail!(
                "Flash-MoE router score projection execution hidden width {} does not match scheduled hidden width {}",
                self.hidden_width,
                hidden_width
            );
        }
        let kind = match self.binding {
            RouterScoreProjectionBinding::ResidentDense(_) => {
                RouterScoreProjectionExecutionKind::ResidentDense
            }
            RouterScoreProjectionBinding::ResidentQ4(_) => {
                RouterScoreProjectionExecutionKind::ResidentQ4
            }
        };
        Ok(RouterScoreProjectionExecution {
            layer: self.layer,
            tensor_name: &self.tensor_name,
            experts: self.experts,
            hidden_width: self.hidden_width,
            kind,
        })
    }

    #[cfg(test)]
    pub(crate) fn topk_plan(
        &self,
        hidden_len: usize,
        active_experts: usize,
    ) -> Result<RouterScoreProjectionTopKPlan> {
        if self.hidden_width != hidden_len {
            bail!(
                "Flash-MoE router score projection topK hidden length {} does not match declared width {}",
                hidden_len,
                self.hidden_width
            );
        }
        if active_experts == 0 || active_experts > self.experts {
            bail!(
                "Flash-MoE router score projection topK active experts {} is outside declared expert range 1..={}",
                active_experts,
                self.experts
            );
        }
        let source = match &self.binding {
            RouterScoreProjectionBinding::ResidentDense(projection) => {
                RouterScoreProjectionTopKSource::ResidentDense(projection.clone())
            }
            RouterScoreProjectionBinding::ResidentQ4(projection) => {
                RouterScoreProjectionTopKSource::ResidentQ4(projection.clone())
            }
        };
        Ok(RouterScoreProjectionTopKPlan {
            layer: self.layer,
            tensor_name: self.tensor_name.clone(),
            experts: self.experts,
            hidden_width: self.hidden_width,
            active_experts,
            source,
        })
    }
}

pub(crate) fn build_router_score_projection_descriptor<'a, F>(
    layer: usize,
    experts: usize,
    hidden_width: usize,
    store_len: u64,
    mut lookup: F,
) -> Result<Option<RouterScoreProjectionDescriptor>>
where
    F: FnMut(&str) -> Option<&'a RuntimeTensorEntry>,
{
    let tensor_name = router_tensor_name(layer);
    let Some(entry) = lookup(&tensor_name) else {
        return Ok(None);
    };
    RouterScoreProjectionDescriptor::from_entry(
        layer,
        &tensor_name,
        entry,
        store_len,
        experts,
        hidden_width,
    )
    .map(Some)
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RouterScoreBatch {
    state: FlashMoeRoutingOutputState,
    pub(crate) projection: Option<RouterScoreProjectionDescriptor>,
    pub(crate) scores: Vec<f32>,
}

impl RouterScoreBatch {
    pub(crate) fn new(
        state: FlashMoeRoutingOutputState,
        projection: Option<RouterScoreProjectionDescriptor>,
        scores: Vec<f32>,
    ) -> Result<Self> {
        if !state.is_declared_graph_state() {
            bail!("FlashMoe router score batch is not declared graph state");
        }
        if state.source() != FlashMoeRoutingOutputSource::CpuRouterScores {
            bail!(
                "FlashMoe router score batch source {:?} is not CPU router scores",
                state.source()
            );
        }
        if scores.len() != state.experts() {
            bail!(
                "FlashMoe router score batch has {} scores for {} declared experts",
                scores.len(),
                state.experts()
            );
        }
        if let Some(projection) = projection.as_ref() {
            if projection.layer != state.layer() {
                bail!(
                    "FlashMoe router score batch layer {} does not match projection layer {}",
                    state.layer(),
                    projection.layer
                );
            }
            if projection.experts != state.experts() {
                bail!(
                    "FlashMoe router score batch expert count {} does not match projection experts {}",
                    state.experts(),
                    projection.experts
                );
            }
        }
        Ok(Self {
            state,
            projection,
            scores,
        })
    }

    pub(crate) fn state(&self) -> FlashMoeRoutingOutputState {
        self.state
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Cmd2ResidentPostAttentionPrepProjections {
    pub(crate) layer: usize,
    pub(crate) out_proj: ResidentMmapMatvecProjection,
    pub(crate) router: ResidentMmapMatvecProjection,
    pub(crate) experts: usize,
    pub(crate) residual_width: usize,
    pub(crate) attention_width: usize,
    pub(crate) active_experts: usize,
}

impl Cmd2ResidentPostAttentionPrepProjections {
    pub(crate) fn new(
        layer: usize,
        out_proj: ResidentMmapMatvecProjection,
        router: ResidentMmapMatvecProjection,
        experts: usize,
        residual_width: usize,
        attention_width: usize,
        active_experts: usize,
    ) -> Result<Self> {
        if residual_width == 0 || attention_width == 0 || experts == 0 {
            bail!(
                "FlashMoe CMD2 resident post-attention prep requires non-zero experts, residual width, and attention width"
            );
        }
        if out_proj.output_width() != residual_width || out_proj.cols() != attention_width {
            bail!(
                "FlashMoe CMD2 resident post-attention output projection shape is invalid: output_width={} cols={} expected output_width={} cols={}",
                out_proj.output_width(),
                out_proj.cols(),
                residual_width,
                attention_width
            );
        }
        if router.output_width() != experts || router.cols() != residual_width {
            bail!(
                "FlashMoe CMD2 resident post-attention router projection shape is invalid: output_width={} cols={} expected output_width={} cols={}",
                router.output_width(),
                router.cols(),
                experts,
                residual_width
            );
        }
        Ok(Self {
            layer,
            out_proj,
            router,
            experts,
            residual_width,
            attention_width,
            active_experts,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cmd2ResidentPostAttentionPrepPlan {
    pub(crate) layer: usize,
    pub(crate) width: usize,
    pub(crate) attention_width: usize,
    pub(crate) experts: usize,
    pub(crate) active_count: usize,
}

impl Cmd2ResidentPostAttentionPrepProjections {
    pub(crate) fn resident_plan(
        &self,
        attention_width: usize,
        residual_width: usize,
        post_norm_weight_len: usize,
    ) -> Result<Cmd2ResidentPostAttentionPrepPlan> {
        if self.active_experts == 0 {
            bail!(
                "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: active expert count is zero"
            );
        }
        if attention_width == 0 || residual_width == 0 {
            bail!(
                "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: attention width {attention_width} and residual width {residual_width} must be non-zero"
            );
        }
        if residual_width != post_norm_weight_len {
            bail!(
                "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: post-attention norm weight length {post_norm_weight_len} does not match residual width {residual_width}"
            );
        }
        if self.out_proj.output_width() != residual_width
            || self.out_proj.rows() != residual_width
            || self.out_proj.cols() != attention_width
            || self.router.cols() != residual_width
            || self.router.output_width() != self.router.rows()
        {
            bail!(
                "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: projection shapes out={}x{} rows={} router={}x{} rows={} do not match attention width {} and residual width {}",
                self.out_proj.output_width(),
                self.out_proj.cols(),
                self.out_proj.rows(),
                self.router.output_width(),
                self.router.cols(),
                self.router.rows(),
                attention_width,
                residual_width
            );
        }
        Ok(Cmd2ResidentPostAttentionPrepPlan {
            layer: self.layer,
            width: residual_width,
            attention_width,
            experts: self.router.rows(),
            active_count: self.active_experts.min(self.router.rows()).max(1),
        })
    }
}

#[cfg(test)]
fn build_cmd2_resident_post_attention_prep_projections<F>(
    layer: usize,
    experts: usize,
    out_proj_name: &str,
    attention_width: usize,
    residual_width: usize,
    active_experts: usize,
    mut projection: F,
) -> Result<Option<Cmd2ResidentPostAttentionPrepProjections>>
where
    F: FnMut(&str, usize, usize) -> Result<Option<ResidentMmapMatvecProjection>>,
{
    if experts == 0 || attention_width == 0 || residual_width == 0 {
        return Ok(None);
    }
    let Some(out_proj) = projection(out_proj_name, residual_width, attention_width)? else {
        return Ok(None);
    };
    let router_name = router_tensor_name(layer);
    let Some(router) = projection(&router_name, experts, residual_width)? else {
        return Ok(None);
    };
    Cmd2ResidentPostAttentionPrepProjections::new(
        layer,
        out_proj,
        router,
        experts,
        residual_width,
        attention_width,
        active_experts,
    )
    .map(Some)
}

pub(crate) fn build_required_cmd2_resident_post_attention_prep_projections<F>(
    layer: usize,
    experts: usize,
    out_proj_name: &str,
    attention_width: usize,
    residual_width: usize,
    active_experts: usize,
    mut projection: F,
) -> Result<Cmd2ResidentPostAttentionPrepProjections>
where
    F: FnMut(&str, usize, usize) -> Result<Option<ResidentMmapMatvecProjection>>,
{
    if experts == 0 || attention_width == 0 || residual_width == 0 {
        bail!(
            "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: experts {experts}, attention width {attention_width}, and residual width {residual_width} must be non-zero"
        );
    }
    let out_proj = projection(out_proj_name, residual_width, attention_width)?.ok_or_else(|| {
        anyhow::anyhow!(
            "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: missing output projection {out_proj_name}"
        )
    })?;
    let router_name = router_tensor_name(layer);
    let router = projection(&router_name, experts, residual_width)?.ok_or_else(|| {
        anyhow::anyhow!(
            "FlashMoe unsupported scheduled CMD2 resident post-attention prep path: missing router projection {router_name}"
        )
    })?;
    Cmd2ResidentPostAttentionPrepProjections::new(
        layer,
        out_proj,
        router,
        experts,
        residual_width,
        attention_width,
        active_experts,
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuVisibleNextNormWeights<'a> {
    pub(crate) tensor_name: &'a str,
    pub(crate) width: usize,
    values: &'a [f32],
}

impl<'a> CpuVisibleNextNormWeights<'a> {
    pub(crate) fn new(tensor_name: &'a str, values: &'a [f32], width: usize) -> Result<Self> {
        if tensor_name.is_empty() {
            bail!("FlashMoe scheduled next-norm weights require a tensor name");
        }
        if width == 0 {
            bail!("FlashMoe scheduled next-norm weights require non-zero width");
        }
        if values.len() < width {
            bail!(
                "FlashMoe scheduled next-norm weight tensor {tensor_name} length {} is smaller than width {width}",
                values.len()
            );
        }
        Ok(Self {
            tensor_name,
            width,
            values,
        })
    }

    pub(crate) fn values(self) -> &'a [f32] {
        &self.values[..self.width]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScheduledNextNormWeights<'a> {
    None,
    CpuVisible(CpuVisibleNextNormWeights<'a>),
}

impl<'a> ScheduledNextNormWeights<'a> {
    pub(crate) fn none() -> Self {
        Self::None
    }

    pub(crate) fn cpu_visible(
        tensor_name: &'a str,
        values: &'a [f32],
        width: usize,
    ) -> Result<Self> {
        Ok(Self::CpuVisible(CpuVisibleNextNormWeights::new(
            tensor_name,
            values,
            width,
        )?))
    }

    pub(crate) fn is_none(self) -> bool {
        matches!(self, Self::None)
    }

    pub(crate) fn is_cpu_visible(self) -> bool {
        matches!(self, Self::CpuVisible(_))
    }

    pub(crate) fn width(self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::CpuVisible(weights) => Some(weights.width),
        }
    }

    pub(crate) fn values(self) -> Option<&'a [f32]> {
        match self {
            Self::None => None,
            Self::CpuVisible(weights) => Some(weights.values()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedScheduledNextNormWeights {
    tensor_name: Option<String>,
    values: Option<Vec<f32>>,
    width: usize,
}

impl PreparedScheduledNextNormWeights {
    pub(crate) fn none() -> Self {
        Self {
            tensor_name: None,
            values: None,
            width: 0,
        }
    }

    pub(crate) fn cpu_visible(tensor_name: String, values: Vec<f32>, width: usize) -> Result<Self> {
        CpuVisibleNextNormWeights::new(&tensor_name, &values, width)?;
        Ok(Self {
            tensor_name: Some(tensor_name),
            values: Some(values),
            width,
        })
    }

    pub(crate) fn scheduled(&self) -> Result<ScheduledNextNormWeights<'_>> {
        match (self.tensor_name.as_deref(), self.values.as_deref()) {
            (Some(tensor_name), Some(values)) => {
                ScheduledNextNormWeights::cpu_visible(tensor_name, values, self.width)
            }
            (None, None) => Ok(ScheduledNextNormWeights::none()),
            _ => bail!("FlashMoe scheduled next-norm weights have incomplete descriptor state"),
        }
    }
}

pub(crate) fn layer_norm_tensor_name(layer: usize, name: &str) -> String {
    format!("model.layers.{layer}.{name}.weight")
}

pub(crate) fn router_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.gate.weight")
}

pub(crate) fn shared_expert_tensor_name(layer: usize, projection: &str) -> String {
    format!("model.layers.{layer}.mlp.shared_expert.{projection}.weight")
}

pub(crate) fn shared_expert_gate_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.shared_expert_gate.weight")
}

pub(crate) fn attention_tensor_name(layer: usize, projection: &str) -> String {
    format!("model.layers.{layer}.self_attn.{projection}.weight")
}

pub(crate) fn linear_attention_tensor_name(layer: usize, projection: &str) -> String {
    format!("model.layers.{layer}.linear_attn.{projection}.weight")
}

pub(crate) fn linear_attention_scalar_tensor_name(layer: usize, name: &str) -> String {
    format!("model.layers.{layer}.linear_attn.{name}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseProjectionRequest<'a> {
    pub(crate) tensor_name: &'a str,
    pub(crate) output_width: usize,
}

impl<'a> DenseProjectionRequest<'a> {
    fn validate(tensor_name: &str, output_width: usize) -> Result<()> {
        if tensor_name.is_empty() {
            bail!("FlashMoe dense projection request requires a tensor name");
        }
        if output_width == 0 {
            bail!("FlashMoe dense projection request requires non-zero output width");
        }
        Ok(())
    }

    pub(crate) fn new(tensor_name: &'a str, output_width: usize) -> Result<Self> {
        Self::validate(tensor_name, output_width)?;
        Ok(Self {
            tensor_name,
            output_width,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseProjectionRequestGroup<const N: usize> {
    tensor_names: [String; N],
    output_widths: [usize; N],
}

impl<const N: usize> DenseProjectionRequestGroup<N> {
    fn new(tensor_names: [String; N], output_widths: [usize; N]) -> Result<Self> {
        for (tensor_name, output_width) in tensor_names.iter().zip(output_widths.iter().copied()) {
            DenseProjectionRequest::validate(tensor_name, output_width)?;
        }
        Ok(Self {
            tensor_names,
            output_widths,
        })
    }

    pub(crate) fn requests(&self) -> [DenseProjectionRequest<'_>; N] {
        std::array::from_fn(|idx| DenseProjectionRequest {
            tensor_name: &self.tensor_names[idx],
            output_width: self.output_widths[idx],
        })
    }

    #[cfg(test)]
    pub(crate) fn tensor_name(&self, idx: usize) -> &str {
        &self.tensor_names[idx]
    }
}

pub(crate) fn full_attention_input_projection_requests(
    layer: usize,
    q_projection_width: usize,
    kv_width: usize,
) -> Result<DenseProjectionRequestGroup<3>> {
    DenseProjectionRequestGroup::new(
        [
            attention_tensor_name(layer, "q_proj"),
            attention_tensor_name(layer, "k_proj"),
            attention_tensor_name(layer, "v_proj"),
        ],
        [q_projection_width, kv_width, kv_width],
    )
}

pub(crate) fn linear_attention_input_projection_requests(
    layer: usize,
    conv_dim: usize,
    total_value_width: usize,
    num_value_heads: usize,
) -> Result<DenseProjectionRequestGroup<4>> {
    DenseProjectionRequestGroup::new(
        [
            linear_attention_tensor_name(layer, "in_proj_qkv"),
            linear_attention_tensor_name(layer, "in_proj_z"),
            linear_attention_tensor_name(layer, "in_proj_b"),
            linear_attention_tensor_name(layer, "in_proj_a"),
        ],
        [
            conv_dim,
            total_value_width,
            num_value_heads,
            num_value_heads,
        ],
    )
}

pub(crate) fn prepare_scheduled_next_norm_weights<F>(
    layer: usize,
    total_layers: usize,
    width: usize,
    needs_next_layer_norm: bool,
    mut lookup: F,
) -> Result<PreparedScheduledNextNormWeights>
where
    F: FnMut(&str, usize) -> Result<Option<Vec<f32>>>,
{
    if !needs_next_layer_norm || layer + 1 >= total_layers {
        return Ok(PreparedScheduledNextNormWeights::none());
    }

    let name = layer_norm_tensor_name(layer + 1, "input_layernorm");
    let values = lookup(&name, width)?.with_context(|| {
        format!(
            "FlashMoe unsupported scheduled CMD3 path: missing next-layer norm weight {name} for layer {layer}"
        )
    })?;
    PreparedScheduledNextNormWeights::cpu_visible(name, values, width)
}

pub(crate) fn qwen_norm_uses_offset(
    semantics: QwenNormWeightSemantics,
    canonical_name: &str,
) -> bool {
    semantics == QwenNormWeightSemantics::Offset
        && (canonical_name == "model.norm.weight"
            || canonical_name.contains(".input_layernorm.weight")
            || canonical_name.contains(".post_attention_layernorm.weight")
            || canonical_name.contains(".self_attn.q_norm.weight")
            || canonical_name.contains(".self_attn.k_norm.weight"))
}

pub(crate) fn apply_qwen_norm_weight_semantics(
    semantics: QwenNormWeightSemantics,
    canonical_name: &str,
    weight: &mut [f32],
) {
    if qwen_norm_uses_offset(semantics, canonical_name) {
        for value in weight {
            *value += 1.0;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResidentStaticDtype {
    Bf16,
    F16,
    F32,
}

impl ResidentStaticDtype {
    fn from_declared(dtype: &str) -> Option<Self> {
        match dtype.to_ascii_uppercase().as_str() {
            "BF16" | "BFLOAT16" => Some(Self::Bf16),
            "F16" | "FLOAT16" | "FP16" => Some(Self::F16),
            "F32" | "FLOAT32" | "FP32" => Some(Self::F32),
            _ => None,
        }
    }

    pub(crate) const fn as_str(&self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }

    const fn element_size(&self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResidentStaticTensorRef {
    pub(crate) tensor_name: String,
    pub(crate) byte_offset: u64,
    pub(crate) dtype: ResidentStaticDtype,
    pub(crate) values: usize,
    pub(crate) element_size: usize,
}

impl ResidentStaticTensorRef {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        expected_values: usize,
        allowed_dtypes: &[ResidentStaticDtype],
    ) -> Result<Option<Self>> {
        if entry.quantization != TensorQuantization::None {
            return Ok(None);
        }
        let Some(dtype) = ResidentStaticDtype::from_declared(&entry.dtype) else {
            return Ok(None);
        };
        if !allowed_dtypes.contains(&dtype) {
            return Ok(None);
        }
        let element_size = dtype.element_size();
        let expected_bytes = expected_values
            .checked_mul(element_size)
            .context("resident static tensor byte length overflow")?;
        if entry.byte_len as usize != expected_bytes {
            return Ok(None);
        }
        if entry
            .byte_offset
            .checked_add(entry.byte_len)
            .map_or(true, |end| end > store_len)
        {
            return Ok(None);
        }
        if entry.byte_offset % element_size as u64 != 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            tensor_name: tensor_name.to_string(),
            byte_offset: entry.byte_offset,
            dtype,
            values: expected_values,
            element_size,
        }))
    }
}

fn dense_dtype_size(dtype: &str) -> Option<usize> {
    match dtype.to_ascii_uppercase().as_str() {
        "BF16" | "BFLOAT16" | "F16" | "FLOAT16" | "FP16" => Some(2),
        "F32" | "FLOAT32" | "FP32" => Some(4),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseMmapMatvecProjection {
    pub(crate) tensor_name: String,
    pub(crate) byte_offset: u64,
    pub(crate) dtype: String,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) output_width: usize,
}

impl DenseMmapMatvecProjection {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        output_width: usize,
        input_len: usize,
        element_size: usize,
    ) -> Result<Self> {
        let (rows, cols) =
            validate_dense_matvec_shape(entry, tensor_name, output_width, input_len)?;
        let row_bytes = cols
            .checked_mul(element_size)
            .context("dense tensor resident row byte length overflow")?;
        let byte_len = rows
            .checked_mul(row_bytes)
            .context("dense tensor resident byte length overflow")?;
        if entry
            .byte_offset
            .checked_add(byte_len as u64)
            .map_or(true, |end| end > store_len)
        {
            bail!(
                "Flash-MoE dense tensor {} byte range {}..{} exceeds dense store length {}",
                tensor_name,
                entry.byte_offset,
                entry.byte_offset.saturating_add(byte_len as u64),
                store_len
            );
        }
        Ok(Self {
            tensor_name: tensor_name.to_string(),
            byte_offset: entry.byte_offset,
            dtype: entry.dtype.clone(),
            rows,
            cols,
            output_width,
        })
    }

    #[cfg(test)]
    pub(crate) fn stride(&self) -> usize {
        self.cols
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseQ4MmapMatvecProjection {
    pub(crate) tensor_name: String,
    pub(crate) packed_byte_offset: u64,
    pub(crate) scales_byte_offset: u64,
    pub(crate) biases_byte_offset: u64,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) output_width: usize,
    pub(crate) row_packed_bytes: usize,
    pub(crate) groups_per_row: usize,
    pub(crate) group_size: usize,
    pub(crate) scale_bias_dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DenseQ4ProjectionKey {
    pub(crate) name: String,
    pub(crate) output_width: usize,
    pub(crate) input_len: usize,
}

impl DenseQ4ProjectionKey {
    pub(crate) fn new(name: &str, output_width: usize, input_len: usize) -> Self {
        Self {
            name: name.to_string(),
            output_width,
            input_len,
        }
    }
}

impl DenseQ4MmapMatvecProjection {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        output_width: usize,
        input_len: usize,
    ) -> Result<Option<Self>> {
        let TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } = &entry.quantization
        else {
            return Ok(None);
        };
        let (rows, cols) =
            validate_dense_matvec_shape(entry, tensor_name, output_width, input_len)?;
        Self::from_validated_entry(
            tensor_name,
            entry,
            store_len,
            rows,
            cols,
            output_width,
            *group_size,
            scale_bias_dtype,
        )
    }

    pub(crate) fn from_multilinear_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        heads: usize,
        output_width_per_head: usize,
        input_len: usize,
    ) -> Result<Option<Self>> {
        let TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } = &entry.quantization
        else {
            return Ok(None);
        };
        let expected_shape = [heads, output_width_per_head, input_len];
        if entry.shape.as_slice() != expected_shape {
            bail!(
                "Flash-MoE dense tensor {tensor_name} shape mismatch: expected multilinear shape {:?}, actual shape {:?}",
                expected_shape,
                entry.shape
            );
        }
        let rows = heads
            .checked_mul(output_width_per_head)
            .context("dense Q4 multilinear row count overflow")?;
        Self::from_validated_entry(
            tensor_name,
            entry,
            store_len,
            rows,
            input_len,
            rows,
            *group_size,
            scale_bias_dtype,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_validated_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        rows: usize,
        cols: usize,
        output_width: usize,
        group_size: usize,
        scale_bias_dtype: &str,
    ) -> Result<Option<Self>> {
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&entry.shape, group_size, scale_bias_dtype)?;
        if entry.byte_len as usize != layout.total_bytes
            || rows != layout.rows
            || cols != layout.cols
        {
            return Ok(None);
        }
        if entry
            .byte_offset
            .checked_add(entry.byte_len)
            .map_or(true, |end| end > store_len)
        {
            return Ok(None);
        }
        let packed_byte_offset = entry.byte_offset;
        let scales_byte_offset = entry
            .byte_offset
            .checked_add(layout.packed_bytes as u64)
            .context("dense q4 projection scales offset overflow")?;
        let biases_byte_offset = scales_byte_offset
            .checked_add(layout.scales_bytes as u64)
            .context("dense q4 projection biases offset overflow")?;
        Ok(Some(Self {
            tensor_name: tensor_name.to_string(),
            packed_byte_offset,
            scales_byte_offset,
            biases_byte_offset,
            rows,
            cols,
            output_width,
            row_packed_bytes: layout.row_packed_bytes,
            groups_per_row: layout.groups_per_row,
            group_size,
            scale_bias_dtype: scale_bias_dtype.to_string(),
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResidentMmapMatvecProjection {
    Q4(DenseQ4MmapMatvecProjection),
    Dense(DenseMmapMatvecProjection),
}

impl From<DenseQ4MmapMatvecProjection> for ResidentMmapMatvecProjection {
    fn from(projection: DenseQ4MmapMatvecProjection) -> Self {
        Self::Q4(projection)
    }
}

impl ResidentMmapMatvecProjection {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        output_width: usize,
        input_len: usize,
    ) -> Result<Self> {
        match &entry.quantization {
            TensorQuantization::Q4 { .. } => {
                let projection = DenseQ4MmapMatvecProjection::from_entry(
                    tensor_name,
                    entry,
                    store_len,
                    output_width,
                    input_len,
                )?
                .with_context(|| {
                    format!(
                        "resident Q4 projection {tensor_name} does not match its declared shape or byte range"
                    )
                })?;
                Ok(Self::Q4(projection))
            }
            TensorQuantization::None => {
                let element_size = dense_dtype_size(&entry.dtype).with_context(|| {
                    format!(
                        "resident dense projection {tensor_name} has unsupported dtype {}",
                        entry.dtype
                    )
                })?;
                Ok(Self::Dense(DenseMmapMatvecProjection::from_entry(
                    tensor_name,
                    entry,
                    store_len,
                    output_width,
                    input_len,
                    element_size,
                )?))
            }
            TensorQuantization::Gguf { .. } => bail!(
                "resident projection {tensor_name} requires a model-family GGUF graph binding"
            ),
        }
    }

    pub(crate) fn tensor_name(&self) -> &str {
        match self {
            Self::Q4(projection) => &projection.tensor_name,
            Self::Dense(projection) => &projection.tensor_name,
        }
    }

    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Q4(projection) => projection.rows,
            Self::Dense(projection) => projection.rows,
        }
    }

    pub(crate) fn cols(&self) -> usize {
        match self {
            Self::Q4(projection) => projection.cols,
            Self::Dense(projection) => projection.cols,
        }
    }

    pub(crate) fn output_width(&self) -> usize {
        match self {
            Self::Q4(projection) => projection.output_width,
            Self::Dense(projection) => projection.output_width,
        }
    }

    pub(crate) fn q4(&self) -> Option<&DenseQ4MmapMatvecProjection> {
        match self {
            Self::Q4(projection) => Some(projection),
            Self::Dense(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LinearAttentionStaticBindings {
    pub(crate) conv_weight: ResidentStaticTensorRef,
    pub(crate) a_log: ResidentStaticTensorRef,
    pub(crate) dt_bias: ResidentStaticTensorRef,
    pub(crate) norm_weight: ResidentStaticTensorRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LinearAttentionResidentBindings {
    pub(crate) layer: usize,
    pub(crate) input_projections: [ResidentMmapMatvecProjection; 4],
    pub(crate) static_tensors: LinearAttentionStaticBindings,
    pub(crate) out_proj: ResidentMmapMatvecProjection,
    pub(crate) router: ResidentMmapMatvecProjection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LinearAttentionWeightTable {
    layers: Vec<Option<LinearAttentionResidentBindings>>,
}

impl LinearAttentionWeightTable {
    pub(crate) fn empty(layers: usize) -> Self {
        Self {
            layers: vec![None; layers],
        }
    }

    pub(crate) fn require(&self, layer: usize) -> Result<&LinearAttentionResidentBindings> {
        self.layers
            .get(layer)
            .and_then(Option::as_ref)
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled linear-attention weight path: layer {layer} has no resolved resident bindings"
                )
            })
    }

    #[cfg(test)]
    pub(crate) fn layer(&self, layer: usize) -> Option<&LinearAttentionResidentBindings> {
        self.layers.get(layer).and_then(Option::as_ref)
    }
}

pub(crate) fn build_dense_q4_mmap_projection<'a, F>(
    tensor_name: &str,
    output_width: usize,
    input_len: usize,
    store_len: u64,
    mut lookup: F,
) -> Result<Option<DenseQ4MmapMatvecProjection>>
where
    F: FnMut(&str) -> Option<&'a RuntimeTensorEntry>,
{
    let Some(entry) = lookup(tensor_name) else {
        return Ok(None);
    };
    DenseQ4MmapMatvecProjection::from_entry(tensor_name, entry, store_len, output_width, input_len)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SharedExpertPhaseShape {
    pub(crate) width: usize,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) total_intermediate: usize,
}

impl SharedExpertPhaseShape {
    pub(crate) fn new(width: usize, shared_experts: usize, intermediate: usize) -> Result<Self> {
        if width == 0 || shared_experts == 0 || intermediate == 0 {
            bail!(
                "shared expert graph shape requires non-zero width, shared expert count, and intermediate width"
            );
        }
        let total_intermediate = shared_experts
            .checked_mul(intermediate)
            .context("shared expert intermediate width overflow")?;
        Ok(Self {
            width,
            shared_experts,
            intermediate,
            total_intermediate,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SharedExpertPhaseWeights {
    pub(crate) gate: Arc<Vec<f32>>,
    pub(crate) up: Arc<Vec<f32>>,
    pub(crate) down: Arc<Vec<f32>>,
    pub(crate) router: Arc<Vec<f32>>,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) width: usize,
}

impl SharedExpertPhaseWeights {
    #[cfg(test)]
    pub(crate) fn new(
        gate: Arc<Vec<f32>>,
        up: Arc<Vec<f32>>,
        down: Arc<Vec<f32>>,
        router: Arc<Vec<f32>>,
        shared_experts: usize,
        intermediate: usize,
        width: usize,
    ) -> Result<Self> {
        let weights = Self {
            gate,
            up,
            down,
            router,
            shared_experts,
            intermediate,
            width,
        };
        weights.validated_shape()?;
        Ok(weights)
    }

    pub(crate) fn validated_shape(&self) -> Result<SharedExpertPhaseShape> {
        let shape =
            SharedExpertPhaseShape::new(self.width, self.shared_experts, self.intermediate)?;
        let dense_len = shape
            .total_intermediate
            .checked_mul(shape.width)
            .context("shared expert dense projection width overflow")?;
        let router_len = shape
            .shared_experts
            .checked_mul(shape.width)
            .context("shared expert router projection width overflow")?;
        if self.gate.len() != dense_len
            || self.up.len() != dense_len
            || self.down.len() != dense_len
            || self.router.len() != router_len
        {
            bail!(
                "FlashMoe scheduled shared dense expert shape is invalid: width={} shared_experts={} intermediate={} gate={} up={} down={} router={}",
                self.width,
                self.shared_experts,
                self.intermediate,
                self.gate.len(),
                self.up.len(),
                self.down.len(),
                self.router.len()
            );
        }
        Ok(shape)
    }
}

#[cfg(test)]
pub(crate) fn build_shared_expert_phase_weights<F>(
    layer: usize,
    width: usize,
    shared_experts: usize,
    intermediate: usize,
    mut lookup: F,
) -> Result<Option<SharedExpertPhaseWeights>>
where
    F: FnMut(&str) -> Result<Option<Arc<Vec<f32>>>>,
{
    if shared_experts == 0 || intermediate == 0 {
        return Ok(None);
    }
    SharedExpertPhaseShape::new(width, shared_experts, intermediate)?;

    let gate_name = shared_expert_tensor_name(layer, "gate_proj");
    let up_name = shared_expert_tensor_name(layer, "up_proj");
    let down_name = shared_expert_tensor_name(layer, "down_proj");
    let shared_gate_name = shared_expert_gate_tensor_name(layer);
    let gate = lookup(&gate_name)?
        .with_context(|| format!("missing configured shared expert tensor {gate_name}"))?;
    let up = lookup(&up_name)?
        .with_context(|| format!("missing configured shared expert tensor {up_name}"))?;
    let down = lookup(&down_name)?
        .with_context(|| format!("missing configured shared expert tensor {down_name}"))?;
    let router = lookup(&shared_gate_name)?.with_context(|| {
        format!("missing configured shared expert gate tensor {shared_gate_name}")
    })?;

    SharedExpertPhaseWeights::new(gate, up, down, router, shared_experts, intermediate, width)
        .map(Some)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SharedExpertPhaseResidentProjections {
    pub(crate) gate: ResidentMmapMatvecProjection,
    pub(crate) up: ResidentMmapMatvecProjection,
    pub(crate) down: ResidentMmapMatvecProjection,
    pub(crate) router: Option<ResidentMmapMatvecProjection>,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) width: usize,
}

impl SharedExpertPhaseResidentProjections {
    pub(crate) fn validated_shape(&self) -> Result<SharedExpertPhaseShape> {
        let shape =
            SharedExpertPhaseShape::new(self.width, self.shared_experts, self.intermediate)?;
        if self.gate.cols() != shape.width
            || self.up.cols() != shape.width
            || self
                .router
                .as_ref()
                .is_some_and(|router| router.cols() != shape.width)
            || self.down.cols() != shape.total_intermediate
            || self.gate.output_width() != shape.total_intermediate
            || self.up.output_width() != shape.total_intermediate
            || self.down.output_width() != shape.width
            || self
                .router
                .as_ref()
                .is_some_and(|router| router.output_width() != shape.shared_experts)
        {
            bail!(
                "FlashMoe scheduled resident shared-expert shape is invalid: width={} shared_experts={} intermediate={} gate=({},{}) up=({},{}) down=({},{}) router=({},{})",
                self.width,
                self.shared_experts,
                self.intermediate,
                self.gate.output_width(),
                self.gate.cols(),
                self.up.output_width(),
                self.up.cols(),
                self.down.output_width(),
                self.down.cols(),
                self.router
                    .as_ref()
                    .map_or(0, |router| router.output_width()),
                self.router.as_ref().map_or(0, |router| router.cols())
            );
        }
        Ok(shape)
    }
}

#[cfg(test)]
pub(crate) fn build_shared_expert_resident_phase_projections<F, P>(
    layer: usize,
    width: usize,
    shared_experts: usize,
    intermediate: usize,
    mut projection: F,
) -> Result<Option<SharedExpertPhaseResidentProjections>>
where
    F: FnMut(&str, usize, usize) -> Result<Option<P>>,
    P: Into<ResidentMmapMatvecProjection>,
{
    if width == 0 || shared_experts == 0 || intermediate == 0 {
        return Ok(None);
    }
    let shape = SharedExpertPhaseShape::new(width, shared_experts, intermediate)?;

    let gate_name = shared_expert_tensor_name(layer, "gate_proj");
    let up_name = shared_expert_tensor_name(layer, "up_proj");
    let down_name = shared_expert_tensor_name(layer, "down_proj");
    let router_name = shared_expert_gate_tensor_name(layer);
    let Some(gate) = projection(&gate_name, shape.total_intermediate, shape.width)? else {
        return Ok(None);
    };
    let Some(up) = projection(&up_name, shape.total_intermediate, shape.width)? else {
        return Ok(None);
    };
    let Some(down) = projection(&down_name, shape.width, shape.total_intermediate)? else {
        return Ok(None);
    };
    let Some(router) = projection(&router_name, shape.shared_experts, shape.width)? else {
        return Ok(None);
    };

    let shared = SharedExpertPhaseResidentProjections {
        gate: gate.into(),
        up: up.into(),
        down: down.into(),
        router: Some(router.into()),
        shared_experts,
        intermediate,
        width,
    };
    shared.validated_shape()?;
    Ok(Some(shared))
}

#[cfg(test)]
pub(crate) fn build_required_shared_expert_resident_phase_projections<F, P>(
    layer: usize,
    width: usize,
    shared_experts: usize,
    intermediate: usize,
    projection: F,
) -> Result<Option<SharedExpertPhaseResidentProjections>>
where
    F: FnMut(&str, usize, usize) -> Result<Option<P>>,
    P: Into<ResidentMmapMatvecProjection>,
{
    build_required_shared_expert_resident_phase_projections_with_router(
        layer,
        width,
        shared_experts,
        intermediate,
        true,
        projection,
    )
}

pub(crate) fn build_required_shared_expert_resident_phase_projections_with_router<F, P>(
    layer: usize,
    width: usize,
    shared_experts: usize,
    intermediate: usize,
    requires_router: bool,
    mut projection: F,
) -> Result<Option<SharedExpertPhaseResidentProjections>>
where
    F: FnMut(&str, usize, usize) -> Result<Option<P>>,
    P: Into<ResidentMmapMatvecProjection>,
{
    if shared_experts == 0 {
        return Ok(None);
    }
    let shape = SharedExpertPhaseShape::new(width, shared_experts, intermediate)?;

    let gate_name = shared_expert_tensor_name(layer, "gate_proj");
    let up_name = shared_expert_tensor_name(layer, "up_proj");
    let down_name = shared_expert_tensor_name(layer, "down_proj");
    let router_name = shared_expert_gate_tensor_name(layer);
    let gate = projection(&gate_name, shape.total_intermediate, shape.width)?.ok_or_else(|| {
        anyhow::anyhow!(
            "FlashMoe unsupported scheduled CMD3 shared-expert path: missing resident shared gate projection {gate_name}"
        )
    })?;
    let up = projection(&up_name, shape.total_intermediate, shape.width)?.ok_or_else(|| {
        anyhow::anyhow!(
            "FlashMoe unsupported scheduled CMD3 shared-expert path: missing resident shared up projection {up_name}"
        )
    })?;
    let down = projection(&down_name, shape.width, shape.total_intermediate)?.ok_or_else(|| {
        anyhow::anyhow!(
            "FlashMoe unsupported scheduled CMD3 shared-expert path: missing resident shared down projection {down_name}"
        )
    })?;
    let router = if requires_router {
        Some(
            projection(&router_name, shape.shared_experts, shape.width)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "FlashMoe unsupported scheduled CMD3 shared-expert path: missing resident shared router projection {router_name}"
                )
            })?,
        )
    } else {
        None
    };

    let shared = SharedExpertPhaseResidentProjections {
        gate: gate.into(),
        up: up.into(),
        down: down.into(),
        router: router.map(Into::into),
        shared_experts,
        intermediate,
        width,
    };
    shared.validated_shape()?;
    Ok(Some(shared))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SharedExpertLayerWeights {
    None,
    Resident(SharedExpertPhaseResidentProjections),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SharedExpertWeightTable {
    layers: Vec<SharedExpertLayerWeights>,
}

impl SharedExpertWeightTable {
    pub(crate) fn none(layers: usize) -> Self {
        Self {
            layers: vec![SharedExpertLayerWeights::None; layers],
        }
    }

    pub(crate) fn layer(&self, layer: usize) -> Result<&SharedExpertLayerWeights> {
        self.layers.get(layer).with_context(|| {
            format!("FlashMoe scheduled shared-expert weight table has no entry for layer {layer}")
        })
    }
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct SharedExpertPhaseCache {
    dense: Mutex<BTreeMap<usize, Arc<SharedExpertPhaseWeights>>>,
}

#[cfg(test)]
impl SharedExpertPhaseCache {
    pub(crate) fn dense<F>(
        &self,
        layer: usize,
        width: usize,
        shared_experts: usize,
        intermediate: usize,
        lookup: F,
    ) -> Result<Option<Arc<SharedExpertPhaseWeights>>>
    where
        F: FnMut(&str) -> Result<Option<Arc<Vec<f32>>>>,
    {
        if shared_experts == 0 || intermediate == 0 {
            return Ok(None);
        }
        if let Some(shared) = self.cached_dense(layer, width)? {
            return Ok(Some(shared));
        }

        let Some(shared) =
            build_shared_expert_phase_weights(layer, width, shared_experts, intermediate, lookup)?
        else {
            return Ok(None);
        };
        let shared = Arc::new(shared);
        let mut cache = self.dense.lock().expect("shared expert cache poisoned");
        if let Some(existing) = cache.get(&layer).cloned() {
            validate_cached_shared_width("shared expert tensors", layer, existing.width, width)?;
            Ok(Some(existing))
        } else {
            cache.insert(layer, shared.clone());
            Ok(Some(shared))
        }
    }

    fn cached_dense(
        &self,
        layer: usize,
        width: usize,
    ) -> Result<Option<Arc<SharedExpertPhaseWeights>>> {
        let cache = self.dense.lock().expect("shared expert cache poisoned");
        let Some(shared) = cache.get(&layer).cloned() else {
            return Ok(None);
        };
        validate_cached_shared_width("shared expert tensors", layer, shared.width, width)?;
        Ok(Some(shared))
    }
}

#[cfg(test)]
fn validate_cached_shared_width(
    label: &str,
    layer: usize,
    cached_width: usize,
    requested_width: usize,
) -> Result<()> {
    if cached_width != requested_width {
        bail!(
            "cached {label} for layer {layer} have width {cached_width}, requested {requested_width}"
        );
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn dense_projection_tile_rows(cols: usize, rows: usize) -> usize {
    let bytes_per_row = cols.saturating_mul(std::mem::size_of::<f32>()).max(1);
    (DENSE_PROJECTION_TILE_BYTES / bytes_per_row)
        .max(1)
        .min(rows.max(1))
}

#[cfg(test)]
fn dense_mmap_dtype_supported(dtype: &str) -> bool {
    matches!(
        dtype.to_ascii_uppercase().as_str(),
        "F32" | "FLOAT32" | "FP32" | "BF16" | "BFLOAT16"
    )
}

#[cfg(test)]
pub(super) fn dense_projection_tile_rows_for_metal(
    dtype: &str,
    cols: usize,
    rows: usize,
    resident_mmap_available: bool,
) -> usize {
    if resident_mmap_available && dense_mmap_dtype_supported(dtype) {
        rows.max(1)
    } else {
        dense_projection_tile_rows(cols, rows)
    }
}

mod store;
pub use store::*;

#[cfg(test)]
#[path = "parity_tests.rs"]
mod parity_tests;

#[cfg(test)]
#[path = "../tests/weights.rs"]
mod tests;
