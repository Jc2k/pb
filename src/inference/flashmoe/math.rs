use anyhow::{Context, Result, bail};

#[cfg(test)]
use super::state::LinearAttentionLayout;
use super::types::GROUP_SIZE;
use super::vision::{MropeAxis, MropePosition};
use super::weights::{FullAttentionLayout, FullAttentionQLayout, RotaryPairing};

#[cfg(test)]
pub(super) fn reorder_grouped_linear_qkv_projection(
    qkv: &mut Vec<f32>,
    layout: LinearAttentionLayout,
) -> Result<()> {
    if qkv.len() != layout.conv_dim {
        bail!(
            "linear-attention qkv projection produced {} values; expected {}",
            qkv.len(),
            layout.conv_dim
        );
    }
    let value_heads_per_key = layout.value_heads_per_key_head();
    let value_width_per_key = value_heads_per_key
        .checked_mul(layout.value_dim)
        .context("linear-attention grouped value width overflow")?;
    let group_width = layout
        .key_dim
        .checked_mul(2)
        .and_then(|width| width.checked_add(value_width_per_key))
        .context("linear-attention grouped qkv width overflow")?;
    if group_width
        .checked_mul(layout.num_key_heads)
        .context("linear-attention grouped qkv total width overflow")?
        != layout.conv_dim
    {
        bail!(
            "linear-attention grouped qkv width {group_width} * heads {} does not match conv width {}",
            layout.num_key_heads,
            layout.conv_dim
        );
    }

    let mut reordered = vec![0.0f32; qkv.len()];
    for head in 0..layout.num_key_heads {
        let src = head * group_width;
        let src_q = src;
        let src_k = src_q + layout.key_dim;
        let src_v = src_k + layout.key_dim;

        let dst_q = head * layout.key_dim;
        let dst_k = layout.total_key_width + head * layout.key_dim;
        let dst_v = 2 * layout.total_key_width + head * value_width_per_key;

        reordered[dst_q..dst_q + layout.key_dim]
            .copy_from_slice(&qkv[src_q..src_q + layout.key_dim]);
        reordered[dst_k..dst_k + layout.key_dim]
            .copy_from_slice(&qkv[src_k..src_k + layout.key_dim]);
        reordered[dst_v..dst_v + value_width_per_key]
            .copy_from_slice(&qkv[src_v..src_v + value_width_per_key]);
    }
    *qkv = reordered;
    Ok(())
}

#[cfg(test)]
pub(super) fn normalize_linear_attention_qk_in_place(
    layout: LinearAttentionLayout,
    lin_q: &mut [f32],
    lin_k: &mut [f32],
) -> Result<()> {
    if lin_q.len() < layout.total_key_width || lin_k.len() < layout.total_key_width {
        bail!(
            "linear-attention q/k widths are q={}, k={}, expected at least {}",
            lin_q.len(),
            lin_k.len(),
            layout.total_key_width
        );
    }
    let inv_scale = 1.0f32 / (layout.key_dim as f32).sqrt();
    let q_scale = inv_scale * inv_scale;
    for head in 0..layout.num_key_heads {
        let start = head * layout.key_dim;
        let end = start + layout.key_dim;
        rms_norm_with_weight_in_place(&mut lin_q[start..end], None);
        for value in &mut lin_q[start..end] {
            *value *= q_scale;
        }
        rms_norm_with_weight_in_place(&mut lin_k[start..end], None);
        for value in &mut lin_k[start..end] {
            *value *= inv_scale;
        }
    }
    Ok(())
}

pub(super) fn split_q_projection(
    projected: Vec<f32>,
    layout: FullAttentionLayout,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    match layout.q_layout {
        FullAttentionQLayout::Standard => {
            if projected.len() != layout.q_width {
                bail!(
                    "standard q_proj produced {} values; expected {}",
                    projected.len(),
                    layout.q_width
                );
            }
            Ok((projected, None))
        }
        FullAttentionQLayout::Gated => {
            if projected.len() != layout.q_projection_width {
                bail!(
                    "gated q_proj produced {} values; expected {}",
                    projected.len(),
                    layout.q_projection_width
                );
            }

            let mut q = vec![0.0f32; layout.q_width];
            let mut gate = vec![0.0f32; layout.q_width];

            for head in 0..layout.num_q_heads {
                let src = head * 2 * layout.head_dim;
                let dst = head * layout.head_dim;

                q[dst..dst + layout.head_dim]
                    .copy_from_slice(&projected[src..src + layout.head_dim]);
                gate[dst..dst + layout.head_dim]
                    .copy_from_slice(&projected[src + layout.head_dim..src + 2 * layout.head_dim]);
            }

            Ok((q, Some(gate)))
        }
    }
}

#[cfg(test)]
pub(super) fn rms_norm_in_place(values: &mut [f32]) {
    rms_norm_with_weight_in_place(values, None)
}

pub(super) fn rms_norm_with_weight_in_place(values: &mut [f32], weight: Option<&[f32]>) {
    let mean_square =
        values.iter().map(|value| value * value).sum::<f32>() / values.len().max(1) as f32;
    let scale = (mean_square + 1e-6).sqrt().recip();
    for (idx, value) in values.iter_mut().enumerate() {
        *value *= scale;
        if let Some(weight) = weight
            && let Some(weight) = weight.get(idx)
        {
            *value *= *weight;
        }
    }
}

#[cfg(test)]
pub(super) fn apply_rotary(values: &mut [f32], position: usize, head_dim: usize, theta: f64) {
    apply_rotary_split_half(values, position, head_dim, head_dim, theta);
}

pub(super) fn apply_per_head_rms_norm(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    weight: Option<&[f32]>,
) -> Result<()> {
    if values.len() != heads.saturating_mul(head_dim) {
        bail!(
            "per-head RMSNorm got {} values; expected heads {heads} * head_dim {head_dim}",
            values.len()
        );
    }
    if let Some(weight) = weight
        && weight.len() < head_dim
    {
        bail!(
            "per-head RMSNorm weight has len {}; expected at least head_dim {head_dim}",
            weight.len()
        );
    }
    for head in values.chunks_mut(head_dim) {
        rms_norm_with_weight_in_place(head, weight.map(|w| &w[..head_dim]));
    }
    Ok(())
}

pub(super) fn apply_optional_per_head_rms_norm(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    weight: Option<&[f32]>,
) -> Result<()> {
    if let Some(weight) = weight {
        apply_per_head_rms_norm(values, heads, head_dim, Some(weight))?;
    }
    Ok(())
}

pub(super) fn apply_full_attention_qk_norm_and_rotary(
    q: &mut [f32],
    k: &mut [f32],
    layout: FullAttentionLayout,
    position: MropePosition,
    theta: f64,
    mrope_section: Option<[usize; 3]>,
    q_weight: Option<&[f32]>,
    k_weight: Option<&[f32]>,
) -> Result<()> {
    if layout.rotary_pairing == RotaryPairing::SplitHalf {
        apply_optional_per_head_rms_norm_and_split_half_rope(
            q,
            layout.num_q_heads,
            layout.head_dim,
            q_weight,
            position,
            layout.rotary_dim,
            theta,
            mrope_section,
        )?;
        apply_optional_per_head_rms_norm_and_split_half_rope(
            k,
            layout.kv_heads,
            layout.head_dim,
            k_weight,
            position,
            layout.rotary_dim,
            theta,
            mrope_section,
        )?;
        return Ok(());
    }

    apply_optional_per_head_rms_norm(q, layout.num_q_heads, layout.head_dim, q_weight)?;
    apply_optional_per_head_rms_norm(k, layout.kv_heads, layout.head_dim, k_weight)?;
    apply_rotary_for_layout(q, k, position, theta, layout, mrope_section);
    Ok(())
}

pub(super) fn apply_optional_per_head_rms_norm_and_split_half_rope(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    weight: Option<&[f32]>,
    position: MropePosition,
    rotary_dim: usize,
    theta: f64,
    mrope_section: Option<[usize; 3]>,
) -> Result<()> {
    if values.len() != heads.saturating_mul(head_dim) {
        bail!(
            "per-head RMSNorm/RoPE got {} values; expected heads {heads} * head_dim {head_dim}",
            values.len()
        );
    }
    if let Some(weight) = weight
        && weight.len() < head_dim
    {
        bail!(
            "per-head RMSNorm/RoPE weight has len {}; expected at least head_dim {head_dim}",
            weight.len()
        );
    }
    let rotations = split_half_rope_rotations(position, head_dim, rotary_dim, theta, mrope_section);
    let rotary_dim = rotations.len().saturating_mul(2);
    let half = rotations.len();
    for head in values.chunks_mut(head_dim) {
        let norm_scale = weight.map(|_| {
            let mean_square =
                head.iter().map(|value| value * value).sum::<f32>() / head.len().max(1) as f32;
            (mean_square + 1e-6).sqrt().recip()
        });
        if let (Some(scale), Some(weight)) = (norm_scale, weight) {
            for (idx, value) in head.iter_mut().enumerate() {
                *value *= scale * weight[idx];
            }
        }
        if head.len() < rotary_dim {
            continue;
        }
        for (idx, (sin, cos)) in rotations.iter().copied().enumerate() {
            let x0 = head[idx];
            let x1 = head[idx + half];
            head[idx] = x0 * cos - x1 * sin;
            head[idx + half] = x0 * sin + x1 * cos;
        }
    }
    Ok(())
}

pub(super) fn split_half_rope_rotations(
    position: MropePosition,
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
    mrope_section: Option<[usize; 3]>,
) -> Vec<(f32, f32)> {
    let theta = theta.max(1.0) as f32;
    let head_dim = head_dim.max(2);
    let rotary_dim = rotary_dim.min(head_dim) - (rotary_dim.min(head_dim) % 2);
    let half = rotary_dim / 2;
    (0..half)
        .map(|idx| {
            let axis_position = if let Some(section) = mrope_section {
                position.axis(mrope_axis_for_frequency(idx, section))
            } else {
                position.temporal
            };
            let inv_freq = theta.powf(-((2 * idx) as f32) / rotary_dim.max(1) as f32);
            let angle = (axis_position as f32) * inv_freq;
            angle.sin_cos()
        })
        .collect()
}

pub(super) fn apply_rotary_for_layout(
    q: &mut [f32],
    k: &mut [f32],
    position: MropePosition,
    theta: f64,
    layout: FullAttentionLayout,
    mrope_section: Option<[usize; 3]>,
) {
    match layout.rotary_pairing {
        RotaryPairing::Adjacent => {
            if let Some(section) = mrope_section {
                apply_rotary_adjacent_mrope(
                    q,
                    position,
                    layout.head_dim,
                    layout.rotary_dim,
                    theta,
                    section,
                );
                apply_rotary_adjacent_mrope(
                    k,
                    position,
                    layout.head_dim,
                    layout.rotary_dim,
                    theta,
                    section,
                );
            } else {
                apply_rotary_adjacent(
                    q,
                    position.temporal,
                    layout.head_dim,
                    layout.rotary_dim,
                    theta,
                );
                apply_rotary_adjacent(
                    k,
                    position.temporal,
                    layout.head_dim,
                    layout.rotary_dim,
                    theta,
                );
            }
        }
        RotaryPairing::SplitHalf => {
            if let Some(section) = mrope_section {
                apply_rotary_split_half_mrope(
                    q,
                    position,
                    layout.head_dim,
                    layout.rotary_dim,
                    theta,
                    section,
                );
                apply_rotary_split_half_mrope(
                    k,
                    position,
                    layout.head_dim,
                    layout.rotary_dim,
                    theta,
                    section,
                );
            } else {
                apply_rotary_split_half(
                    q,
                    position.temporal,
                    layout.head_dim,
                    layout.rotary_dim,
                    theta,
                );
                apply_rotary_split_half(
                    k,
                    position.temporal,
                    layout.head_dim,
                    layout.rotary_dim,
                    theta,
                );
            }
        }
    }
}

pub(super) fn apply_rotary_adjacent(
    values: &mut [f32],
    position: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
) {
    let theta = theta.max(1.0) as f32;
    let head_dim = head_dim.max(2);
    let rotary_dim = rotary_dim.min(head_dim) - (rotary_dim.min(head_dim) % 2);

    for head in values.chunks_mut(head_dim) {
        let rotary_dims = rotary_dim.min(head.len()) - (rotary_dim.min(head.len()) % 2);
        for pair_idx in (0..rotary_dims).step_by(2) {
            let inv_freq = theta.powf(-(pair_idx as f32) / head_dim as f32);
            let angle = (position as f32) * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x = head[pair_idx];
            let y = head[pair_idx + 1];
            head[pair_idx] = x * cos - y * sin;
            head[pair_idx + 1] = x * sin + y * cos;
        }
    }
}

pub(super) fn apply_rotary_adjacent_mrope(
    values: &mut [f32],
    position: MropePosition,
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
    mrope_section: [usize; 3],
) {
    let theta = theta.max(1.0) as f32;
    let head_dim = head_dim.max(2);
    let rotary_dim = rotary_dim.min(head_dim) - (rotary_dim.min(head_dim) % 2);

    for head in values.chunks_mut(head_dim) {
        let rotary_dims = rotary_dim.min(head.len()) - (rotary_dim.min(head.len()) % 2);
        for pair_idx in (0..rotary_dims).step_by(2) {
            let freq_idx = pair_idx / 2;
            let axis = mrope_axis_for_frequency(freq_idx, mrope_section);
            let inv_freq = theta.powf(-(pair_idx as f32) / head_dim as f32);
            let angle = (position.axis(axis) as f32) * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x = head[pair_idx];
            let y = head[pair_idx + 1];
            head[pair_idx] = x * cos - y * sin;
            head[pair_idx + 1] = x * sin + y * cos;
        }
    }
}

pub(super) fn apply_rotary_split_half(
    values: &mut [f32],
    position: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
) {
    let theta = theta.max(1.0) as f32;
    let head_dim = head_dim.max(2);
    let rotary_dim = rotary_dim.min(head_dim) - (rotary_dim.min(head_dim) % 2);
    let half = rotary_dim / 2;

    for head in values.chunks_mut(head_dim) {
        if head.len() < rotary_dim {
            continue;
        }
        for i in 0..half {
            let inv_freq = theta.powf(-((2 * i) as f32) / rotary_dim as f32);
            let angle = (position as f32) * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos - x1 * sin;
            head[i + half] = x0 * sin + x1 * cos;
        }
    }
}

pub(super) fn apply_rotary_split_half_mrope(
    values: &mut [f32],
    position: MropePosition,
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
    mrope_section: [usize; 3],
) {
    let theta = theta.max(1.0) as f32;
    let head_dim = head_dim.max(2);
    let rotary_dim = rotary_dim.min(head_dim) - (rotary_dim.min(head_dim) % 2);
    let half = rotary_dim / 2;

    for head in values.chunks_mut(head_dim) {
        if head.len() < rotary_dim {
            continue;
        }
        for i in 0..half {
            let axis = mrope_axis_for_frequency(i, mrope_section);
            let inv_freq = theta.powf(-((2 * i) as f32) / rotary_dim as f32);
            let angle = (position.axis(axis) as f32) * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos - x1 * sin;
            head[i + half] = x0 * sin + x1 * cos;
        }
    }
}

pub(super) fn mrope_axis_for_frequency(index: usize, section: [usize; 3]) -> MropeAxis {
    if index % 3 == 1 && index < section[1].saturating_mul(3) {
        MropeAxis::Height
    } else if index % 3 == 2 && index < section[2].saturating_mul(3) {
        MropeAxis::Width
    } else {
        MropeAxis::Temporal
    }
}

pub(super) fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
pub(super) fn cpu_dense_matvec(
    weights: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let used_cols = cols.min(input.len());
    let mut out = vec![0.0f32; rows];
    for (row, slot) in out.iter_mut().enumerate() {
        let start = row.saturating_mul(cols);
        let end = start.saturating_add(used_cols).min(weights.len());
        let acc = weights
            .get(start..end)
            .unwrap_or(&[])
            .iter()
            .zip(input.iter().take(used_cols))
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        *slot = acc;
    }
    out
}

pub(super) fn add_in_place(target: &mut [f32], update: &[f32]) {
    for (target, update) in target.iter_mut().zip(update) {
        *target += *update;
    }
}

pub(super) fn trace_layer_values(position: usize, layer: usize, stage: &str, values: &[f32]) {
    let _ = (position, layer, stage, values);
}

pub(crate) fn causal_attention(
    q: &[f32],
    keys_values: &[(&[f32], &[f32])],
    num_q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    if keys_values.is_empty() || num_q_heads == 0 || head_dim == 0 {
        return vec![0.0; q.len()];
    }
    let q_width = num_q_heads * head_dim;
    let groups_per_kv = num_q_heads / kv_heads.max(1);
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; q_width];
    for qh in 0..num_q_heads {
        let kv_head = qh / groups_per_kv.max(1);
        let q_slice = &q[qh * head_dim..(qh + 1) * head_dim];
        // score this Q head against every token's corresponding K head
        let mut scores: Vec<f32> = keys_values
            .iter()
            .map(|(k, _)| {
                let k_slice = &k[kv_head * head_dim..(kv_head + 1) * head_dim];
                q_slice.iter().zip(k_slice).map(|(a, b)| a * b).sum::<f32>() * scale
            })
            .collect();
        softmax_in_place(&mut scores);
        // weighted sum of corresponding V head
        let out_slice = &mut out[qh * head_dim..(qh + 1) * head_dim];
        for (weight, (_, value)) in scores.into_iter().zip(keys_values.iter()) {
            let v_slice = &value[kv_head * head_dim..(kv_head + 1) * head_dim];
            for (o, v) in out_slice.iter_mut().zip(v_slice) {
                *o += weight * v;
            }
        }
    }
    out
}

/// Sort by descending score, then ascending token id for stable tie-breaking.
pub(crate) fn compare_scored_tokens(
    left: &(usize, f32),
    right: &(usize, f32),
) -> std::cmp::Ordering {
    right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
}

pub fn top_k(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(compare_scored_tokens);
    indexed.truncate(k.min(indexed.len()));
    indexed
}

pub fn softmax_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 && sum.is_finite() {
        for value in values {
            *value /= sum;
        }
    }
}

pub fn q4_fma_matvec(
    packed: &[u8],
    input: &[f32],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    q4_fma_matvec_with_group_size(packed, input, scales, biases, rows, cols, GROUP_SIZE)
}

pub fn q4_fma_matvec_with_group_size(
    packed: &[u8],
    input: &[f32],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Result<Vec<f32>> {
    if group_size == 0 {
        bail!("group_size must be positive");
    }
    if input.len() != cols {
        bail!("input length {} does not match cols {cols}", input.len());
    }
    let groups_per_row = cols.div_ceil(group_size);
    if scales.len() < rows * groups_per_row || biases.len() < rows * groups_per_row {
        bail!("scale/bias arrays are too small for {rows}x{cols} with group size {group_size}");
    }
    let needed_packed = rows * cols.div_ceil(2);
    if packed.len() < needed_packed {
        bail!(
            "packed q4 data has {} bytes, needs at least {needed_packed}",
            packed.len()
        );
    }
    let mut out = vec![0.0f32; rows];
    let packed_stride = cols.div_ceil(2);
    debug_assert_eq!(groups_per_row, cols.div_ceil(group_size));
    for (row, out_value) in out.iter_mut().enumerate().take(rows) {
        let mut acc = 0.0f32;
        let packed_row = row * packed_stride;
        for group in 0..groups_per_row {
            let idx = row * groups_per_row + group;
            let scale = scales[idx];
            let bias = biases[idx];
            let start = group * group_size;
            let end = (start + group_size).min(cols);
            for col in start..end {
                let x = input[col];
                let scale_x = scale * x;
                let bias_x = bias * x;
                let byte = packed[packed_row + col / 2];
                let q = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                acc += q.mul_add(scale_x, bias_x);
            }
        }
        *out_value = acc;
    }
    Ok(out)
}

pub fn q4_dequantize_rows_with_group_size(
    packed: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Result<Vec<f32>> {
    if group_size == 0 {
        bail!("group_size must be positive");
    }
    let groups_per_row = cols.div_ceil(group_size);
    if scales.len() < rows * groups_per_row || biases.len() < rows * groups_per_row {
        bail!("scale/bias arrays are too small for {rows}x{cols} with group size {group_size}");
    }
    let needed_packed = rows * cols.div_ceil(2);
    if packed.len() < needed_packed {
        bail!(
            "packed q4 data has {} bytes, needs at least {needed_packed}",
            packed.len()
        );
    }
    let mut out = vec![0.0f32; rows * cols];
    let packed_stride = cols.div_ceil(2);
    for row in 0..rows {
        let packed_row = row * packed_stride;
        let out_row = row * cols;
        for group in 0..groups_per_row {
            let idx = row * groups_per_row + group;
            let scale = scales[idx];
            let bias = biases[idx];
            let start = group * group_size;
            let end = (start + group_size).min(cols);
            for col in start..end {
                let byte = packed[packed_row + col / 2];
                let q = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                out[out_row + col] = q.mul_add(scale, bias);
            }
        }
    }
    Ok(out)
}

pub(crate) struct QuantizedQ4 {
    pub(crate) values: Vec<u8>,
    pub(crate) scales: Vec<f32>,
    pub(crate) biases: Vec<f32>,
}

struct QuantizedQ4Group {
    codes: Vec<u8>,
    scale: f32,
    bias: f32,
    error: f32,
}

pub(crate) fn quantize_q4(
    values: &[f32],
    shape: &[usize],
    group_size: usize,
) -> Result<QuantizedQ4> {
    if group_size == 0 {
        bail!("group_size must be positive");
    }
    let cols = shape.last().copied().unwrap_or(values.len());
    if cols == 0 {
        bail!("cannot quantize q4 tensor with zero columns");
    }
    let rows = if shape.len() > 1 {
        shape[..shape.len() - 1].iter().product::<usize>().max(1)
    } else {
        1
    };
    let expected = rows
        .checked_mul(cols)
        .context("q4 tensor element count overflow")?;
    if expected != values.len() {
        bail!(
            "q4 tensor shape {:?} describes {expected} values but decoded tensor has {}",
            shape,
            values.len()
        );
    }
    let row_stride = cols.div_ceil(2);
    let mut packed_values = Vec::with_capacity(rows * row_stride);
    let mut scales = Vec::new();
    let mut biases = Vec::new();

    for row in values.chunks_exact(cols) {
        let mut pending_low: Option<u8> = None;
        let row_start_len = packed_values.len();
        for group in row.chunks(group_size) {
            let quantized = quantize_q4_group_affine_mse(group);
            scales.push(quantized.scale);
            biases.push(quantized.bias);
            for q in quantized.codes {
                if let Some(low) = pending_low.take() {
                    packed_values.push(low | (q << 4));
                } else {
                    pending_low = Some(q);
                }
            }
        }
        if let Some(low) = pending_low {
            packed_values.push(low);
        }
        while packed_values.len() - row_start_len < row_stride {
            packed_values.push(0);
        }
    }
    Ok(QuantizedQ4 {
        values: packed_values,
        scales,
        biases,
    })
}

fn quantize_q4_group_affine_mse(group: &[f32]) -> QuantizedQ4Group {
    if group.is_empty() {
        return QuantizedQ4Group {
            codes: Vec::new(),
            scale: 1.0,
            bias: 0.0,
            error: 0.0,
        };
    }

    let finite: Vec<f32> = group
        .iter()
        .map(|value| if value.is_finite() { *value } else { 0.0 })
        .collect();
    let min = finite.iter().copied().fold(f32::INFINITY, f32::min);
    let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    quantize_q4_group_from_range(&finite, min, max)
}

fn quantize_q4_group_from_range(values: &[f32], min: f32, max: f32) -> QuantizedQ4Group {
    let range = (max - min).abs();
    if !range.is_finite() || range <= f32::EPSILON {
        let bias = if min.is_finite() { min } else { 0.0 };
        let codes = vec![0; values.len()];
        let error = values
            .iter()
            .map(|value| {
                let delta = *value - bias;
                delta * delta
            })
            .sum();
        return QuantizedQ4Group {
            codes,
            scale: 1.0,
            bias,
            error,
        };
    }

    let mut scale = range / 15.0;
    let mut bias = min;
    let mut codes = quantize_q4_codes(values, scale, bias);
    let mut best = quantized_q4_group(values, scale, bias, codes.clone());

    for _ in 0..4 {
        if let Some((fit_scale, fit_bias)) = fit_q4_affine(values, &codes) {
            scale = fit_scale;
            bias = fit_bias;
        }
        codes = quantize_q4_codes(values, scale, bias);
        let candidate = quantized_q4_group(values, scale, bias, codes.clone());
        if candidate.error < best.error {
            best = candidate;
        }
    }
    best
}

fn quantize_q4_codes(values: &[f32], scale: f32, bias: f32) -> Vec<u8> {
    let scale = if scale.is_finite() && scale.abs() > f32::EPSILON {
        scale
    } else {
        1.0
    };
    values
        .iter()
        .map(|value| ((*value - bias) / scale).round().clamp(0.0, 15.0) as u8)
        .collect()
}

fn fit_q4_affine(values: &[f32], codes: &[u8]) -> Option<(f32, f32)> {
    if values.len() != codes.len() || values.is_empty() {
        return None;
    }
    let n = values.len() as f32;
    let mut sum_q = 0.0f32;
    let mut sum_x = 0.0f32;
    let mut sum_qq = 0.0f32;
    let mut sum_qx = 0.0f32;
    for (value, code) in values.iter().zip(codes) {
        let q = *code as f32;
        sum_q += q;
        sum_x += *value;
        sum_qq += q * q;
        sum_qx += q * *value;
    }
    let denom = n.mul_add(sum_qq, -(sum_q * sum_q));
    if !denom.is_finite() || denom.abs() <= f32::EPSILON {
        return None;
    }
    let scale = (n.mul_add(sum_qx, -(sum_q * sum_x))) / denom;
    let bias = (sum_x - scale * sum_q) / n;
    if scale.is_finite() && scale > f32::EPSILON && bias.is_finite() {
        Some((scale, bias))
    } else {
        None
    }
}

fn quantized_q4_group(values: &[f32], scale: f32, bias: f32, codes: Vec<u8>) -> QuantizedQ4Group {
    let error = values
        .iter()
        .zip(&codes)
        .map(|(value, code)| {
            let decoded = (*code as f32).mul_add(scale, bias);
            let delta = *value - decoded;
            delta * delta
        })
        .sum();
    QuantizedQ4Group {
        codes,
        scale,
        bias,
        error,
    }
}

#[cfg(test)]
#[path = "math_parity_tests.rs"]
mod parity_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    fn conv1d_reference_step(
        state: &[f32],
        input: &[f32],
        weight: &[f32],
        channels: usize,
        kernel_size: usize,
    ) -> Vec<f32> {
        (0..channels)
            .map(|channel| {
                let history = (0..kernel_size - 1)
                    .map(|tap| {
                        state[tap * channels + channel] * weight[channel * kernel_size + tap]
                    })
                    .sum::<f32>();
                silu(
                    history
                        + input[channel]
                            * weight[channel * kernel_size + kernel_size.saturating_sub(1)],
                )
            })
            .collect()
    }

    fn gated_delta_reference(
        layout: LinearAttentionLayout,
        state: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha: &[f32],
        beta: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
    ) -> Vec<f32> {
        let heads_per_key = layout.value_heads_per_key_head();
        let matrix_len = layout.value_dim * layout.key_dim;
        let mut output = vec![0.0; layout.num_value_heads * layout.value_dim];
        for value_head in 0..layout.num_value_heads {
            let key_head = value_head / heads_per_key;
            let decay = (-(a_log[value_head].exp())
                * (1.0 + (alpha[value_head] + dt_bias[value_head]).exp()).ln())
            .exp();
            let beta_gate = 1.0 / (1.0 + (-beta[value_head]).exp());
            let state_base = value_head * matrix_len;
            let key = &k[key_head * layout.key_dim..(key_head + 1) * layout.key_dim];
            let query = &q[key_head * layout.key_dim..(key_head + 1) * layout.key_dim];
            let value = &v[value_head * layout.value_dim..(value_head + 1) * layout.value_dim];
            for value_index in 0..layout.value_dim {
                let row_base = state_base + value_index * layout.key_dim;
                let row = &mut state[row_base..row_base + layout.key_dim];
                for slot in row.iter_mut() {
                    *slot *= decay;
                }
                let remembered = row
                    .iter()
                    .zip(key)
                    .map(|(state, key)| state * key)
                    .sum::<f32>();
                let delta = (value[value_index] - remembered) * beta_gate;
                for (state, key) in row.iter_mut().zip(key) {
                    *state += key * delta;
                }
                output[value_head * layout.value_dim + value_index] = row
                    .iter()
                    .zip(query)
                    .map(|(state, query)| state * query)
                    .sum();
            }
        }
        output
    }

    #[test]
    fn routing_top_k_is_stable_and_softmax_normalizes() {
        let selected = top_k(&[0.1, 0.9, 0.9, -1.0], 2);
        assert_eq!(selected, vec![(1, 0.9), (2, 0.9)]);
        let mut weights: Vec<f32> = selected.iter().map(|(_, score)| *score).collect();
        softmax_in_place(&mut weights);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn optional_qk_norm_absent_leaves_projection_unchanged() {
        let mut values = vec![3.0, 4.0, 5.0, 12.0];
        let original = values.clone();
        apply_optional_per_head_rms_norm(&mut values, 2, 2, None).unwrap();
        assert_eq!(values, original);

        apply_optional_per_head_rms_norm(&mut values, 2, 2, Some(&[1.0, 1.0])).unwrap();
        assert_ne!(values, original);
    }

    #[test]
    fn fused_qk_norm_rope_matches_separate_steps() {
        let layout = FullAttentionLayout {
            q_layout: FullAttentionQLayout::Standard,
            q_projection_width: 16,
            q_width: 16,
            kv_width: 8,
            head_dim: 8,
            rotary_dim: 8,
            num_q_heads: 2,
            kv_heads: 1,
            rotary_pairing: RotaryPairing::SplitHalf,
        };
        let q_weight = [0.5, 1.0, 1.5, 2.0, 0.75, 1.25, 1.75, 2.25];
        let k_weight = [1.25, 0.75, 1.5, 0.5, 2.0, 1.0, 0.875, 1.125];
        let mut expected_q: Vec<f32> = (0..layout.q_width)
            .map(|index| ((index as f32) * 0.31).sin() + 0.125)
            .collect();
        let mut expected_k: Vec<f32> = (0..layout.kv_width)
            .map(|index| ((index as f32) * 0.19).cos() - 0.25)
            .collect();
        let mut actual_q = expected_q.clone();
        let mut actual_k = expected_k.clone();
        let position = MropePosition {
            temporal: 7,
            height: 3,
            width: 5,
        };

        apply_optional_per_head_rms_norm(
            &mut expected_q,
            layout.num_q_heads,
            layout.head_dim,
            Some(&q_weight),
        )
        .unwrap();
        apply_optional_per_head_rms_norm(
            &mut expected_k,
            layout.kv_heads,
            layout.head_dim,
            Some(&k_weight),
        )
        .unwrap();
        apply_rotary_for_layout(
            &mut expected_q,
            &mut expected_k,
            position,
            1_000_000.0,
            layout,
            Some([1, 1, 1]),
        );
        apply_full_attention_qk_norm_and_rotary(
            &mut actual_q,
            &mut actual_k,
            layout,
            position,
            1_000_000.0,
            Some([1, 1, 1]),
            Some(&q_weight),
            Some(&k_weight),
        )
        .unwrap();

        for (actual, expected) in actual_q
            .iter()
            .chain(&actual_k)
            .zip(expected_q.iter().chain(&expected_k))
        {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn causal_attention_gqa_matches_independent_reference() {
        let query = [0.2, -0.1, 0.4, 0.3, -0.5, 0.7, 0.6, -0.2];
        let k0 = [0.1, 0.3, -0.2, 0.4];
        let v0 = [1.0, 2.0, 3.0, 4.0];
        let k1 = [0.5, -0.4, 0.6, 0.2];
        let v1 = [-1.0, 0.5, 2.0, -0.5];
        let keys_values = [(&k0[..], &v0[..]), (&k1[..], &v1[..])];

        let actual = causal_attention(&query, &keys_values, 4, 2, 2);
        let mut expected = vec![0.0f32; query.len()];
        let scale = (2.0f32).sqrt().recip();
        for query_head in 0..4 {
            let key_head = query_head / 2;
            let q = &query[query_head * 2..query_head * 2 + 2];
            let mut scores = keys_values
                .iter()
                .map(|(key, _)| {
                    let key = &key[key_head * 2..key_head * 2 + 2];
                    q.iter().zip(key).map(|(q, key)| q * key).sum::<f32>() * scale
                })
                .collect::<Vec<_>>();
            softmax_in_place(&mut scores);
            for (score, (_, value)) in scores.iter().zip(keys_values) {
                let value = &value[key_head * 2..key_head * 2 + 2];
                expected[query_head * 2] += score * value[0];
                expected[query_head * 2 + 1] += score * value[1];
            }
        }
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn causal_conv1d_reference_uses_chronological_state_order() {
        let actual = conv1d_reference_step(
            &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            &[4.0, 40.0],
            &[0.1, 0.2, 0.3, 0.4, -0.2, 0.05, 0.15, -0.1],
            2,
            4,
        );
        let expected = [
            silu(1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3 + 4.0 * 0.4),
            silu(10.0 * -0.2 + 20.0 * 0.05 + 30.0 * 0.15 + 40.0 * -0.1),
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn gated_delta_reference_preserves_value_major_state_layout() {
        let layout = LinearAttentionLayout {
            num_key_heads: 1,
            num_value_heads: 2,
            key_dim: 2,
            value_dim: 2,
            total_key_width: 2,
            total_value_width: 4,
            conv_dim: 6,
            conv_kernel_size: 2,
        };
        let mut state = vec![0.1, 0.2, 0.3, 0.4, -0.2, 0.5, 0.7, -0.1];
        let output = gated_delta_reference(
            layout,
            &mut state,
            &[0.7, -0.2],
            &[0.4, 0.1],
            &[0.2, -0.1, 0.6, 0.05],
            &[0.1, -0.3],
            &[0.2, 0.5],
            &[-0.2, 0.4],
            &[0.05, -0.15],
        );

        assert_eq!(output.len(), layout.total_value_width);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(state.iter().all(|value| value.is_finite()));
        assert_ne!(state, vec![0.1, 0.2, 0.3, 0.4, -0.2, 0.5, 0.7, -0.1]);
    }

    #[test]
    fn gated_q_projection_splits_query_and_gate_by_head() {
        let layout = FullAttentionLayout {
            q_layout: FullAttentionQLayout::Gated,
            q_projection_width: 8,
            q_width: 4,
            kv_width: 2,
            head_dim: 2,
            rotary_dim: 2,
            num_q_heads: 2,
            kv_heads: 1,
            rotary_pairing: RotaryPairing::SplitHalf,
        };

        let (query, gate) =
            split_q_projection((0..8).map(|value| value as f32).collect(), layout).unwrap();

        assert_eq!(query, vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(gate.unwrap(), vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn quantize_q4_packs_nibbles_and_group_metadata() {
        let packed = quantize_q4(&[0.0, 15.0, 30.0], &[1, 3], 2).unwrap();

        assert_eq!(packed.values.len(), 2);
        assert_eq!(packed.scales.len(), 2);
        assert_eq!(packed.biases.len(), 2);
    }

    #[test]
    fn q4_dequantize_rows_supports_variable_groups_and_odd_widths() {
        let packed = [
            0x10, 0x32, 0x04, // row 0: 0, 1, 2, 3, 4
            0x65, 0x87, 0x09, // row 1: 5, 6, 7, 8, 9
        ];
        let scales = [0.5, -0.25, 0.125, 0.75];
        let biases = [1.0, 2.0, -1.5, 0.25];

        let decoded =
            q4_dequantize_rows_with_group_size(&packed, &scales, &biases, 2, 5, 3).unwrap();

        assert_eq!(
            decoded,
            vec![1.0, 1.5, 2.0, 1.25, 1.0, -0.875, -0.75, -0.625, 6.25, 7.0]
        );
    }
}
