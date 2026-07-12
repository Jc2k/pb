//! Flash-MoE inspired inference backend for Qwen3.5-397B-A17B on Apple Silicon.
//!
//! The upstream flash-moe design is very different from llama.cpp: non-expert
//! tensors are mmap'd, routed expert tensors stay on SSD, and each token reads
//! only the active MoE experts with parallel `pread` before dispatching fused
//! Metal kernels.  This module captures that runtime contract in pb instead of
//! pretending a GGUF file is required for Qwen3.5.

#![allow(
    clippy::assertions_on_constants,
    clippy::collapsible_if,
    clippy::default_constructed_unit_structs,
    clippy::derivable_impls,
    clippy::enum_variant_names,
    clippy::explicit_counter_loop,
    clippy::manual_checked_ops,
    clippy::manual_inspect,
    clippy::manual_is_multiple_of,
    clippy::manual_saturating_arithmetic,
    clippy::manual_slice_size_calculation,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::needless_option_as_deref,
    clippy::ptr_arg,
    clippy::type_complexity,
    clippy::unnecessary_get_then_check,
    clippy::unnecessary_map_or,
    clippy::useless_format,
    clippy::useless_vec
)]

#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::path::Path;

#[cfg(test)]
use super::cache::*;
#[cfg(test)]
use super::experts::*;
use super::experts::{PackedExpertTensor, parse_pbq4_expert_pack};
#[cfg(test)]
use super::math::*;
#[cfg(test)]
use super::metal::MetalExecutionContext;
use super::model_family::QwenModelConfig;
#[cfg(test)]
use super::test_fixtures::*;
#[cfg(test)]
use super::text::*;
use super::types::*;
#[cfg(test)]
use super::vision::{MropePosition, qwen3vl_single_image_mrope_positions};
#[cfg(test)]
use super::weights::*;
#[cfg(test)]
const DENSE_Q4_GROUP_SIZE: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flashmoe_parity_tokenizer_chat_template_and_routing_goldens() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_tokenizer_config_json()),
        )
        .unwrap();

        let rendered = tokenizer
            .apply_chat_template_to_messages(&[ChatMessage::text(ChatRole::User, "hi")], &[], true)
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(tokenizer.encode("hi<|im_end|>").unwrap(), vec![3, 101]);
        assert_eq!(tokenizer.decode(&[3, 101, 4]).unwrap(), "hi");

        let routed = top_k(&[0.0, 2.0, 2.0, -1.0, 1.0], 3);
        assert_eq!(routed, vec![(1, 2.0), (2, 2.0), (4, 1.0)]);
        let mut weights: Vec<f32> = routed.iter().map(|(_, score)| *score).collect();
        softmax_in_place(&mut weights);
        for (actual, expected) in weights.iter().zip([0.42231882, 0.42231882, 0.15536241]) {
            assert_close(*actual, expected);
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    mod arm_macos_integration {
        use super::*;

        fn tiny_dense_store(root: &Path) -> DenseStore {
            let dense_path = root.join("model_weights.bin");
            let manifest_path = root.join("model_weights.json");
            std::fs::write(&dense_path, [0u8]).unwrap();
            std::fs::write(
                &manifest_path,
                serde_json::to_vec(&FlashMoeManifest {
                    model: QWEN35_MODEL.to_string(),
                    cache_version: CACHE_VERSION.to_string(),
                    dense_shards: Vec::new(),
                    expert_tensors: Vec::new(),
                    dense_tensors: Vec::new(),
                })
                .unwrap(),
            )
            .unwrap();
            DenseStore::open(dense_path, manifest_path).unwrap()
        }

        #[test]
        #[ignore = "requires Apple Silicon Metal; run on ARM macOS with `cargo test --all-targets -- --ignored`"]
        fn arm_macos_compiles_flashmoe_metal_kernels() {
            let temp = tempfile::tempdir().unwrap();
            let config: QwenModelConfig = serde_json::from_slice(
                br#"{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
            )
            .unwrap();
            let runtime = DenseTransformerRuntime::new(&config);
            let dense = tiny_dense_store(temp.path());
            let _executor = MetalExecutionContext::compile(
                dense.mmap.clone(),
                dense.len,
                &runtime.linear_attention,
            )
            .unwrap();
        }
    }

    #[test]
    fn qwen3next_plain_rms_norm_offsets_match_reference_module_types() {
        let qwen35: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_5_moe","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":2,"vocab_size":248320,"rope_theta":10000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();
        let legacy_qwen3: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":2}"#,
        )
        .unwrap();

        for name in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.3.self_attn.q_norm.weight",
            "model.layers.3.self_attn.k_norm.weight",
        ] {
            assert!(
                qwen3next_norm_uses_offset(qwen35.uses_qwen3next_norm_offsets(), name),
                "{name} should use Qwen3Next 1+weight RMSNorm semantics"
            );
        }

        for name in [
            "model.layers.0.linear_attn.norm.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
        ] {
            assert!(
                !qwen3next_norm_uses_offset(qwen35.uses_qwen3next_norm_offsets(), name),
                "{name} is not a plain Qwen3NextRMSNorm weight"
            );
        }

        assert!(!qwen3next_norm_uses_offset(
            legacy_qwen3.uses_qwen3next_norm_offsets(),
            "model.norm.weight"
        ));
    }

    #[test]
    fn resident_lm_head_candidate_superset_preserves_repeat_penalized_top_k() {
        let logits = vec![10.0, 9.99, 9.98, 9.97, 9.96, 9.0, 8.0];
        let sampler = TokenSampler::new(0.7, 2, 99);
        let prompt = vec![0, 1, 2];
        let repeated = sampler.repeated_tokens(&prompt, &[]);
        let raw_count = sampler.top_k + repeated.len();
        let raw_candidates = top_k(&logits, raw_count);

        let reranked = rerank_resident_lm_head_candidates(
            &raw_candidates,
            sampler.top_k,
            sampler.repeat_penalty,
            &repeated,
        );

        assert_eq!(reranked, sampler.top_candidates(&logits, &prompt, &[]));
        assert_eq!(
            reranked.iter().map(|(token, _)| *token).collect::<Vec<_>>(),
            vec![3, 4]
        );
    }

    #[test]
    fn top_k_candidates_matches_full_top_k_across_tiles() {
        let scores = [0.2, 1.0, 0.9, -1.0, 3.0, 2.0, 3.0];
        let mut candidates = TopKCandidates::new(3);
        for (offset, chunk) in scores.chunks(2).enumerate() {
            for (inner, score) in chunk.iter().copied().enumerate() {
                candidates.push(offset * 2 + inner, score);
            }
        }
        assert_eq!(candidates.into_sorted_vec(), top_k(&scores, 3));
    }

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
        for (idx, (actual, expected)) in pb_weights.iter().zip(reference_weights.iter()).enumerate()
        {
            assert!(
                (*actual - *expected).abs() <= 1e-6,
                "routing weight {idx} diverged: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn expert_tensor_classifier_ignores_mtp_speculative_layers() {
        assert!(is_expert_tensor_name(
            "model.layers.0.mlp.experts.gate_up_proj"
        ));
        assert!(is_expert_tensor_name(
            "model.layers.0.mlp.experts.7.gate_proj.weight"
        ));
        assert!(is_expert_tensor_name(
            "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        ));
        assert!(!is_expert_tensor_name(
            "mtp.layers.0.mlp.experts.7.gate_proj.weight"
        ));
    }

    #[test]
    fn expert_cache_requires_complete_qwen_expert_mlp_triplet() {
        let tensor = ExpertTensorRef {
            tensor: "model.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
            shard: "expert.safetensors".to_string(),
            layer: Some(0),
            expert: Some(0),
            dtype: Some("BF16".to_string()),
            shape: vec![16, 8],
            source_offsets: Some([0, 16]),
            q4_sources: None,
        };
        let err = validate_expert_tensor_group(0, 0, &[&tensor], None).unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required tensor up_proj.weight"),
            "{err:#}"
        );
    }

    #[test]
    fn expert_q4_payload_borrows_record_buffers() {
        let tensor = PackedExpertTensor {
            name: "model.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            source_offsets: [0, 0],
            source_hash: None,
            group_size: 2,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
            packed: vec![0x10, 0x32, 0x54, 0x76],
            scales: vec![0.5, 0.25, 0.125, 0.0625],
            biases: vec![1.0, 2.0, 3.0, 4.0],
            scale_bytes: vec![
                0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x80, 0x3e, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00,
                0x80, 0x3d,
            ],
            bias_bytes: vec![
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00,
                0x80, 0x40,
            ],
        };
        let payload = tensor.matvec_payload(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();

        assert_eq!(payload.rows, 2);
        assert_eq!(payload.cols, 4);
        assert_eq!(payload.packed.as_ptr(), tensor.packed.as_ptr());
        assert_eq!(payload.scales.as_ptr(), tensor.scales.as_ptr());
        assert_eq!(payload.biases.as_ptr(), tensor.biases.as_ptr());
    }

    #[test]
    fn dense_projection_rejects_input_width_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("dense.bin");
        let manifest_path = tmp.path().join("manifest.json");

        let mut bytes = Vec::new();
        for value in 0..6u32 {
            bytes.extend_from_slice(&(value as f32).to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![DenseTensorRef {
                    tensor: "proj.weight".to_string(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![2, 3],
                    source_offsets: [0, bytes.len() as u64],
                    runtime_offset: 0,
                    byte_len: bytes.len() as u64,
                    quantization: TensorQuantization::None,
                    q4_sources: None,
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let err = store
            .matvec_tensor_prefix("proj.weight", &[1.0, 1.0], 2)
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("proj.weight"), "{err:#}");
        assert!(message.contains("expected shape [2, 2]"), "{err:#}");
        assert!(message.contains("actual shape [2, 3]"), "{err:#}");
        assert!(message.contains("input length 2"), "{err:#}");
    }

    #[test]
    fn dense_projection_rejects_output_width_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("dense.bin");
        let manifest_path = tmp.path().join("manifest.json");

        let mut bytes = Vec::new();
        for value in 0..2u32 {
            bytes.extend_from_slice(&(value as f32).to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![DenseTensorRef {
                    tensor: "proj.weight".to_string(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![1, 2],
                    source_offsets: [0, bytes.len() as u64],
                    runtime_offset: 0,
                    byte_len: bytes.len() as u64,
                    quantization: TensorQuantization::None,
                    q4_sources: None,
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let err = store
            .project_dense_tensor_with_metal(None, "proj.weight", &[1.0, 1.0], 2)
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("proj.weight"), "{err:#}");
        assert!(message.contains("expected shape [2, 2]"), "{err:#}");
        assert!(message.contains("actual shape [1, 2]"), "{err:#}");
    }

    #[test]
    fn lm_head_logits_rejects_missing_vocab_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("dense.bin");
        let manifest_path = tmp.path().join("manifest.json");

        let tokenizer = QwenTokenizer::from_json_bytes(
            br#"{
  "added_tokens": [
    {"id": 2, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0,
      "a": 2
    },
    "unk_token": "<unk>"
  }
}"#,
        )
        .unwrap();

        let mut bytes = Vec::new();
        for row_idx in 0..2usize {
            let value = (row_idx as f32) + 1.0;
            bytes.extend_from_slice(&value.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![DenseTensorRef {
                    tensor: "lm_head.weight".to_string(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![2, 2],
                    source_offsets: [0, bytes.len() as u64],
                    runtime_offset: 0,
                    byte_len: bytes.len() as u64,
                    quantization: TensorQuantization::None,
                    q4_sources: None,
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let err = store
            .lm_head_logits("lm_head.weight", &[1.0, 1.0], &tokenizer)
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("lm_head.weight"), "{err:#}");
        assert!(message.contains("expected at least [3, 2]"), "{err:#}");
        assert!(message.contains("actual shape [2, 2]"), "{err:#}");
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
    fn expert_cache_quantizes_decoded_bf16_values_not_raw_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
        write_test_config(&snapshot);
        std::fs::write(
            snapshot.join("dense.safetensors"),
            make_safetensors(&[("model.layers.0.self_attn.q_proj.weight", b"dense")]),
        )
        .unwrap();
        let mut gate_bytes = Vec::new();
        for value in 1u32..=128 {
            gate_bytes.extend_from_slice(&(((value as f32).to_bits() >> 16) as u16).to_le_bytes());
        }
        let mut up_bytes = Vec::new();
        for _ in 0..(16 * 8) {
            up_bytes.extend_from_slice(&0x3f80u16.to_le_bytes());
        }
        let mut down_bytes = Vec::new();
        for _ in 0..(8 * 16) {
            down_bytes.extend_from_slice(&0x3f80u16.to_le_bytes());
        }
        std::fs::write(
            snapshot.join("expert.safetensors"),
            make_typed_safetensors(&[
                (
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    "BF16",
                    vec![16, 8],
                    &gate_bytes,
                ),
                (
                    "model.layers.0.mlp.experts.0.up_proj.weight",
                    "BF16",
                    vec![16, 8],
                    &up_bytes,
                ),
                (
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                    "BF16",
                    vec![8, 16],
                    &down_bytes,
                ),
            ]),
        )
        .unwrap();
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            expert_triplet_weight_map(0, 0),
        )
        .unwrap();

        let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot).unwrap();
        let records = read_pbq4_expert_records(&plan.experts_dir, 0, 0).unwrap();
        let record = records
            .iter()
            .find(|record| record.name.ends_with("gate_proj.weight"))
            .unwrap();
        let input = [1.0; 8];
        let payload = record.matvec_payload(&input, 1).unwrap();
        let out =
            q4_fma_matvec(payload.packed, &input, payload.scales, payload.biases, 1, 8).unwrap();
        assert!((out[0] - 36.0).abs() < 1.0, "decoded q4 sum was {}", out[0]);
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
        let out = q4_fma_matvec_with_group_size(
            &packed, &input, &scales, &biases, rows, cols, group_size,
        )
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

        let actual = q4_fma_matvec_with_group_size(
            &packed, &input, &scales, &biases, rows, cols, group_size,
        )
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

    #[test]
    fn q4_bf16_expert_pack_matches_flashmoe_uint32_nibble_reference() {
        let tensor = "model.layers.0.mlp.experts.0.gate_proj.weight";
        let input = [1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.75];
        let mut pack = Vec::new();
        pack.extend_from_slice(PBQ4_EXPERT_MAGIC);
        pack.extend_from_slice(&(tensor.len() as u32).to_le_bytes());
        pack.extend_from_slice(tensor.as_bytes());
        pack.extend_from_slice(&4u64.to_le_bytes());
        pack.extend_from_slice(&1u64.to_le_bytes());
        pack.extend_from_slice(&f32_to_bf16_bits(0.5).to_le_bytes());
        pack.extend_from_slice(&f32_to_bf16_bits(1.0).to_le_bytes());
        pack.extend_from_slice(&0x7654_3210u32.to_le_bytes());

        let metadata = ExpertPackMetadata {
            layer: 0,
            expert: 0,
            packed_bytes: pack.len() as u64,
            records: vec![ExpertPackRecord {
                tensor: tensor.to_string(),
                dtype: "Q4".to_string(),
                shape: vec![1, 8],
                source_offsets: [0, 8],
                source_hash: Some("synthetic".to_string()),
                record_offset: PBQ4_EXPERT_MAGIC.len() as u64,
                packed_bytes: 4,
                groups: 1,
                group_size: 8,
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            }],
        };
        let records = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap();
        let payload = records[0].matvec_payload(&input, 1).unwrap();
        let actual = q4_fma_matvec_with_group_size(
            payload.packed,
            &input,
            payload.scales,
            payload.biases,
            payload.rows,
            payload.cols,
            payload.group_size,
        )
        .unwrap();

        let packed_word = u32::from_le_bytes([
            payload.packed[0],
            payload.packed[1],
            payload.packed[2],
            payload.packed[3],
        ]);
        let expected: f32 = input
            .iter()
            .enumerate()
            .map(|(n, x)| {
                let nibble = ((packed_word >> (n * 4)) & 0x0f) as f32;
                (nibble * 0.5 + 1.0) * x
            })
            .sum();

        assert_eq!(payload.scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
        assert_eq!(payload.scale_bytes.len(), 2);
        assert_eq!(payload.bias_bytes.len(), 2);
        assert!(
            (actual[0] - expected).abs() <= 1e-6,
            "bf16 q4 matvec diverged from uint32 nibble reference: actual={} expected={expected}",
            actual[0]
        );
    }
}

#[cfg(test)]
mod flashmoe_rope_tests {
    use super::*;

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

    #[test]
    fn qwen3vl_mrope_interleaves_height_and_width_frequency_slots() {
        let position = MropePosition {
            temporal: 2,
            height: 5,
            width: 7,
        };
        let section = [2, 1, 1];
        let head_dim = 8usize;
        let theta = 10_000.0f64;

        let mut got = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        apply_rotary_split_half_mrope(&mut got, position, head_dim, head_dim, theta, section);

        let mut expected = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let half = head_dim / 2;
        for i in 0..half {
            let axis = match i {
                1 => position.height,
                2 => position.width,
                _ => position.temporal,
            };
            let freq = 1.0f32 / (theta as f32).powf((2 * i) as f32 / head_dim as f32);
            let angle = axis as f32 * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = expected[i];
            let x1 = expected[i + half];
            expected[i] = x0 * cos_a - x1 * sin_a;
            expected[i + half] = x0 * sin_a + x1 * cos_a;
        }

        for (left, right) in got.iter().zip(expected.iter()) {
            assert_close(*left, *right);
        }
    }

    #[test]
    fn qwen3vl_image_mrope_positions_match_single_image_get_rope_index_shape() {
        let tokens = [101, 999, 999, 999, 999, 102, 201, 202];
        let (positions, next_position) =
            qwen3vl_single_image_mrope_positions(&tokens, 999, 2, 2).unwrap();

        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(
            positions[1],
            MropePosition {
                temporal: 1,
                height: 1,
                width: 1,
            }
        );
        assert_eq!(
            positions[2],
            MropePosition {
                temporal: 1,
                height: 1,
                width: 2,
            }
        );
        assert_eq!(
            positions[3],
            MropePosition {
                temporal: 1,
                height: 2,
                width: 1,
            }
        );
        assert_eq!(
            positions[4],
            MropePosition {
                temporal: 1,
                height: 2,
                width: 2,
            }
        );
        assert_eq!(positions[5], MropePosition::text(3));
        assert_eq!(positions[6], MropePosition::text(4));
        assert_eq!(positions[7], MropePosition::text(5));
        assert_eq!(next_position, 6);
    }
}
