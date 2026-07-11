use anyhow::{Context, Result, bail};

use super::types::GROUP_SIZE;

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
mod tests {
    use super::*;

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
