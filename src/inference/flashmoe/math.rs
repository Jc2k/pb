use anyhow::{Result, bail};

use super::types::GROUP_SIZE;

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
