//! Owner-local routing, Q4 arithmetic, and rotary parity tests.

use super::*;
use crate::inference::flashmoe::model_family::QwenModelConfig;
use crate::inference::flashmoe::vision::MropePosition;
use crate::inference::flashmoe::weights::{
    FullAttentionLayout, FullAttentionQLayout, RotaryPairing, rotary_dim_for,
};

const DENSE_Q4_GROUP_SIZE: usize = 16;

#[test]
fn cpu_routing_topk_and_softmax_support_non_four_k() {
    let scores = [0.2, 1.0, 0.9, -1.0, 3.0, 2.0, 3.0, 1.5];
    let active = top_k(&scores, 5);
    let active_ids: Vec<_> = active.iter().map(|(expert, _)| *expert).collect();
    assert_eq!(active_ids, vec![4, 6, 5, 7, 1]);

    let mut weights: Vec<f32> = active.iter().map(|(_, score)| *score).collect();
    softmax_in_place(&mut weights);
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(
        weights
            .iter()
            .all(|weight| weight.is_finite() && *weight > 0.0)
    );
}

#[test]
fn routing_weights_match_flashmoe_softmax_then_topk_reference() {
    let router_scores = [0.25, -1.0, 3.5, 3.5, 0.0, 2.25, -0.5, 1.75];
    let k = 4;

    let active = top_k(&router_scores, k);
    let mut pb_weights: Vec<f32> = active.iter().map(|(_, score)| *score).collect();
    softmax_in_place(&mut pb_weights);

    let mut reference_scores = router_scores;
    softmax_in_place(&mut reference_scores);
    let reference_active = top_k(&reference_scores, k);
    let reference_sum: f32 = reference_active.iter().map(|(_, score)| *score).sum();
    let reference_weights: Vec<f32> = reference_active
        .iter()
        .map(|(_, score)| *score / reference_sum)
        .collect();

    let pb_ids: Vec<_> = active.iter().map(|(expert, _)| *expert).collect();
    let reference_ids: Vec<_> = reference_active.iter().map(|(expert, _)| *expert).collect();
    assert_eq!(pb_ids, reference_ids);
    for (idx, (actual, expected)) in pb_weights.iter().zip(reference_weights.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= 1e-6,
            "routing weight {idx} diverged: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn dense_q4_group16_reduces_projection_reconstruction_error() {
    let values: Vec<f32> = (0..128)
        .map(|idx| {
            let base = ((idx as f32) * 0.071).sin() * 0.35;
            let trend = ((idx % 17) as f32 - 8.0) * 0.013;
            if idx % 37 == 0 {
                base + trend + 1.15
            } else {
                base + trend
            }
        })
        .collect();
    let q64 = quantize_q4(&values, &[1, values.len()], GROUP_SIZE).unwrap();
    let q16 = quantize_q4(&values, &[1, values.len()], DENSE_Q4_GROUP_SIZE).unwrap();
    let reconstruction_error = |quantized: &QuantizedQ4, group_size: usize| -> f32 {
        values
            .iter()
            .enumerate()
            .map(|(col, value)| {
                let byte = quantized.values[col / 2];
                let code = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                let group = col / group_size;
                let decoded = code.mul_add(quantized.scales[group], quantized.biases[group]);
                let delta = *value - decoded;
                delta * delta
            })
            .sum()
    };
    let error64 = reconstruction_error(&q64, GROUP_SIZE);
    let error16 = reconstruction_error(&q16, DENSE_Q4_GROUP_SIZE);
    assert!(
        error16 < error64,
        "group16 reconstruction error {error16} was not below group64 error {error64}"
    );
}

#[test]
fn q4_fma_matvec_dequantizes_nibbles_by_group() {
    let packed = [0x21, 0x43];
    let input = [1.0, 2.0, 3.0, 4.0];
    let scales = [0.5];
    let biases = [1.0];
    let out = q4_fma_matvec(&packed, &input, &scales, &biases, 1, 4).unwrap();
    let expected = (1.0 * 0.5 + 1.0) * 1.0
        + (2.0 * 0.5 + 1.0) * 2.0
        + (3.0 * 0.5 + 1.0) * 3.0
        + (4.0 * 0.5 + 1.0) * 4.0;
    assert!((out[0] - expected).abs() < 1e-6);
}

#[test]
fn q4_fma_matvec_supports_variable_groups_and_odd_shapes() {
    let rows = 3;
    let cols = 5;
    let group_size = 3;
    let packed = [
        0x10, 0x32, 0x04, // row 0: 0, 1, 2, 3, 4
        0x65, 0x87, 0x09, // row 1: 5, 6, 7, 8, 9
        0xba, 0xdc, 0x0e, // row 2: 10, 11, 12, 13, 14
    ];
    let input = [0.25, -1.0, 2.0, 0.5, -0.75];
    let scales = [0.5, -0.25, 0.125, 0.75, -0.5, 0.25];
    let biases = [1.0, 2.0, -1.5, 0.25, 0.0, -0.5];
    let out =
        q4_fma_matvec_with_group_size(&packed, &input, &scales, &biases, rows, cols, group_size)
            .unwrap();

    let mut expected = [0.0f32; 3];
    let groups_per_row = cols.div_ceil(group_size);
    for row in 0..rows {
        for col in 0..cols {
            let byte = packed[row * cols.div_ceil(2) + col / 2];
            let q = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
            let group = col / group_size;
            let idx = row * groups_per_row + group;
            expected[row] += q.mul_add(scales[idx] * input[col], biases[idx] * input[col]);
        }
    }

    for (actual, expected) in out.iter().zip(expected) {
        assert!(
            (*actual - expected).abs() < 1e-6,
            "actual {actual} expected {expected}"
        );
    }
}

#[test]
fn q4_fma_matvec_matches_explicit_dequant_reference() {
    let rows = 2;
    let cols = 7;
    let group_size = 4;
    let packed = [
        0xf0, 0x21, 0x43, 0x06, // row 0: 0, 15, 1, 2, 3, 4, 6
        0x75, 0x98, 0xba, 0x0d, // row 1: 5, 7, 8, 9, 10, 11, 13
    ];
    let input = [0.5, -2.0, 1.25, 0.0, -0.75, 3.0, -1.5];
    let scales = [0.03125, -0.125, 0.5, -0.25];
    let biases = [-1.0, 2.0, 0.25, -0.5];

    let actual =
        q4_fma_matvec_with_group_size(&packed, &input, &scales, &biases, rows, cols, group_size)
            .unwrap();

    let packed_stride = cols.div_ceil(2);
    let groups_per_row = cols.div_ceil(group_size);
    let expected: Vec<f32> = (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| {
                    let byte = packed[row * packed_stride + col / 2];
                    let code = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                    let group = row * groups_per_row + col / group_size;
                    let decoded = code * scales[group] + biases[group];
                    decoded * input[col]
                })
                .sum()
        })
        .collect();

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (*actual - *expected).abs() <= 1e-6,
            "FMA matvec diverged from explicit dequant: actual={actual}, expected={expected}"
        );
    }
}
fn assert_close(left: f32, right: f32) {
    let diff = (left - right).abs();
    assert!(
        diff <= 1e-5,
        "values differ: left={left:.9}, right={right:.9}, diff={diff:.9}"
    );
}

#[test]
fn flashmoe_rope_split_half_matches_reference() {
    // Mirrors danveloper/flash-moe:
    //   half = rotary_dim / 2
    //   freq = 1 / pow(theta, 2*i / rotary_dim)
    //   pairs are (x[i], x[i + half]), not adjacent pairs.
    let position = 3usize;
    let head_dim = 8usize;
    let rotary_dim = 4usize;
    let theta = 10_000_000.0f64;

    let mut got = vec![1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0];
    apply_rotary_split_half(&mut got, position, head_dim, rotary_dim, theta);

    let mut expected = vec![1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0];
    let half = rotary_dim / 2;
    for i in 0..half {
        let freq = 1.0f32 / (theta as f32).powf((2 * i) as f32 / rotary_dim as f32);
        let angle = position as f32 * freq;
        let (sin_a, cos_a) = angle.sin_cos();

        let x0 = expected[i];
        let x1 = expected[i + half];
        expected[i] = x0 * cos_a - x1 * sin_a;
        expected[i + half] = x0 * sin_a + x1 * cos_a;
    }

    for (left, right) in got.iter().zip(expected.iter()) {
        assert_close(*left, *right);
    }

    // Non-rotary tail must be untouched.
    assert_eq!(&got[rotary_dim..], &[100.0, 200.0, 300.0, 400.0]);
}

#[test]
fn gated_flashmoe_rope_defaults_to_partial_split_half() {
    let config = QwenModelConfig {
        model_type: Some("qwen".to_string()),
        architectures: None,
        num_hidden_layers: 1,
        hidden_size: 4096,
        num_attention_heads: 32,
        head_dim: Some(256),
        num_key_value_heads: Some(2),
        vocab_size: 248320,
        rope_theta: None,
        partial_rotary_factor: None,
        torch_dtype: Some("bfloat16".to_string()),
        num_experts: Some(512),
        num_experts_per_tok: Some(4),
        norm_topk_prob: None,
        moe_intermediate_size: Some(1024),
        intermediate_size: None,
        max_position_embeddings: None,
        mrope_section: None,
        tie_word_embeddings: None,
        num_shared_experts: None,
        shared_expert_intermediate_size: None,
        vision_config: None,
    };

    let rotary_dim = rotary_dim_for(&config, 256, FullAttentionQLayout::Gated);
    assert_eq!(rotary_dim, 64);
}

#[test]
fn standard_qwen_rope_uses_split_half_pairing() {
    let layout = FullAttentionLayout {
        q_layout: FullAttentionQLayout::Standard,
        q_projection_width: 8,
        q_width: 8,
        kv_width: 8,
        head_dim: 8,
        rotary_dim: 8,
        num_q_heads: 1,
        kv_heads: 1,
        rotary_pairing: RotaryPairing::SplitHalf,
    };

    let mut q = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let mut k = q.clone();
    apply_rotary_for_layout(
        &mut q,
        &mut k,
        MropePosition::text(1),
        1_000_000.0,
        layout,
        None,
    );

    // Adjacent pairing would rotate (0,1), (2,3), ...
    // Split-half pairing rotates (0,4), (1,5), ...
    assert_ne!(q[1], 2.0);
    assert_ne!(q[5], 20.0);
}
