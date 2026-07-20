//! Owner-local cache assembly and manifest classification parity tests.

use super::*;
use crate::inference::flashmoe::test_fixtures::*;
use crate::inference::flashmoe::text::{test_tokenizer_config_json, test_tokenizer_json};
use crate::inference::flashmoe::types::*;

#[test]
fn build_cache_writes_runtime_metadata_and_metal_kernels() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
    std::fs::write(
        snapshot.join("tokenizer_config.json"),
        test_tokenizer_config_json(),
    )
    .unwrap();
    std::fs::write(snapshot.join("chat_template.jinja"), b"external-template").unwrap();
    write_test_config(&snapshot);
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        b"{\"weight_map\":{}}",
    )
    .unwrap();
    let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot).unwrap();
    assert!(plan.runtime_dir.join("kernels.metal").is_file());
    assert!(plan.tokenizer.is_file());
    assert!(plan.tokenizer_config.is_file());
    assert!(plan.chat_template.is_file());
    assert!(plan.tensor_manifest.is_file());
}

#[test]
fn tokenizer_artifacts_copy_optional_external_chat_template() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path().join("source");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
    std::fs::write(
        snapshot.join("tokenizer_config.json"),
        test_tokenizer_config_json(),
    )
    .unwrap();
    std::fs::write(snapshot.join("chat_template.jinja"), b"glm-template").unwrap();
    let plan = plan_unchecked(GLM52_MODEL, &tmp.path().join("models"));

    prepare_tokenizer_artifacts(&snapshot, &plan).unwrap();

    assert_eq!(std::fs::read(&plan.chat_template).unwrap(), b"glm-template");
}

#[test]
fn qwen3vl_cache_status_requires_vision_artifacts() {
    let tmp = tempfile::tempdir().unwrap();
    let plan = plan_unchecked(QWEN3_VL_MODEL, tmp.path());
    std::fs::create_dir_all(&plan.runtime_dir).unwrap();
    std::fs::create_dir_all(&plan.experts_dir).unwrap();
    std::fs::write(&plan.non_expert_weights, b"").unwrap();
    std::fs::write(
            &plan.tensor_manifest,
            br#"{"model":"hf://Qwen/Qwen3-VL-MoE-Instruct","cache_version":"flashmoe-v3","dense_shards":[],"expert_tensors":[],"dense_tensors":[]}"#,
        )
        .unwrap();
    std::fs::write(
        &plan.model_config,
        br#"{
                "model_type": "qwen3_vl",
                "text_config": {
                    "hidden_size": 8,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 1,
                    "vocab_size": 16,
                    "num_experts": 1,
                    "num_experts_per_tok": 1,
                    "moe_intermediate_size": 4
                },
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 4,
                    "num_heads": 1
                }
            }"#,
    )
    .unwrap();
    std::fs::write(&plan.tokenizer, b"{}").unwrap();

    let status = plan.cache_status().unwrap();
    assert!(
        status
            .missing
            .contains(plan.vision_weights.as_ref().unwrap())
    );
    assert!(
        status
            .missing
            .contains(plan.vision_manifest.as_ref().unwrap())
    );
    assert!(
        status
            .missing
            .contains(plan.vision_config_path.as_ref().unwrap())
    );
}

#[test]
fn build_cache_parses_safetensors_index_into_manifest() {
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
    std::fs::write(
        snapshot.join("expert.safetensors"),
        make_typed_safetensors(&typed_fixture_refs(&test_expert_triplet(2, 7))),
    )
    .unwrap();
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        expert_triplet_weight_map(2, 7),
    )
    .unwrap();
    let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot).unwrap();
    let manifest: FlashMoeManifest =
        serde_json::from_slice(&std::fs::read(&plan.tensor_manifest).unwrap()).unwrap();
    assert_eq!(manifest.dense_shards, vec!["dense.safetensors"]);
    assert_eq!(manifest.dense_tensors[0].dtype, "U8");
    assert_eq!(manifest.dense_tensors[0].shape, vec![5]);
    assert_eq!(manifest.dense_tensors[0].runtime_offset, 0);
    assert_eq!(std::fs::read(&plan.non_expert_weights).unwrap(), b"dense");
    assert_eq!(manifest.expert_tensors[0].layer, Some(2));
    assert_eq!(manifest.expert_tensors[0].expert, Some(7));
    assert!(plan.non_expert_weights.is_file());
    let expert_pack = expert_layer_path(&plan.experts_dir, 2);
    assert!(expert_pack.is_file());
    assert!(std::fs::metadata(&expert_pack).unwrap().len() > 0);
    let metadata = read_expert_pack_metadata(&plan.experts_dir, 2, 7)
        .unwrap()
        .unwrap();
    assert_eq!(metadata.expert, 7);

    let registry = TensorRegistry::load(&plan.tensor_manifest).unwrap();
    let dense = registry
        .require("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(dense.dtype, "U8");
    assert_eq!(dense.shape, vec![5]);
    assert_eq!(dense.byte_offset, 0);
    assert_eq!(dense.byte_len, 5);
    assert_eq!(dense.quantization, TensorQuantization::None);
    let expert = registry
        .require("model.layers.2.mlp.experts.7.gate_proj.weight")
        .unwrap();
    assert!(matches!(
        expert.quantization,
        TensorQuantization::Q4 {
            group_size: GROUP_SIZE,
            ..
        }
    ));
}

#[test]
fn colibri_unindexed_manifest_preserves_int8_io_as_resident_bf16() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path();
    let tensors = vec![
        (
            "model.embed_tokens.weight".to_string(),
            "U8".to_string(),
            vec![2],
            vec![(-2i8) as u8, 3],
        ),
        (
            "model.embed_tokens.weight.qs".to_string(),
            "F32".to_string(),
            vec![1],
            0.5f32.to_le_bytes().to_vec(),
        ),
    ];
    std::fs::write(
        snapshot.join("out-00000.safetensors"),
        make_typed_safetensors(&typed_fixture_refs(&tensors)),
    )
    .unwrap();
    let config: QwenModelConfig = serde_json::from_value(serde_json::json!({
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 2,
        "num_attention_heads": 1,
        "head_dim": 1,
        "vocab_size": 1,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "norm_topk_prob": true,
        "moe_intermediate_size": 1,
        "intermediate_size": 2,
        "first_k_dense_replace": 1,
        "q_lora_rank": 1,
        "kv_lora_rank": 1,
        "qk_nope_head_dim": 1,
        "qk_rope_head_dim": 2,
        "v_head_dim": 1,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "index_topk": 16
    }))
    .unwrap();

    let (manifest, visual) =
        build_unindexed_manifest(GLM52_MODEL, snapshot, Some(&config)).unwrap();

    assert!(visual.is_empty());
    assert_eq!(manifest.dense_shards, vec!["out-00000.safetensors"]);
    let embedding = &manifest.dense_tensors[0];
    assert_eq!(embedding.dtype, "BF16");
    assert_eq!(embedding.shape, vec![1, 2]);
    assert_eq!(embedding.byte_len, 4);
    assert_eq!(embedding.quantization, TensorQuantization::None);
    assert_eq!(
        embedding.q4_sources.as_ref().unwrap().source_format,
        DenseQ4SourceFormat::ColibriInt8
    );

    let destination = snapshot.join("dense.bin");
    write_dense_tensor_store(snapshot, &destination, &manifest.dense_tensors, None).unwrap();
    let bytes = std::fs::read(destination).unwrap();
    let first = u16::from_le_bytes(bytes[0..2].try_into().unwrap()) as u32;
    let second = u16::from_le_bytes(bytes[2..4].try_into().unwrap()) as u32;
    assert_eq!(f32::from_bits(first << 16), -1.0);
    assert_eq!(f32::from_bits(second << 16), 1.5);
}

#[test]
fn mlx_mxfp4_manifest_and_dense_writer_build_canonical_q4() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path();
    let weight = "model.layers.0.self_attn.q_proj.weight";
    let scales = "model.layers.0.self_attn.q_proj.scales";
    let mut packed = vec![0x91; 16];
    packed.extend(vec![0xe6; 16]);
    let tensors = vec![
        (weight.to_string(), "U32".to_string(), vec![1, 8], packed),
        (
            scales.to_string(),
            "U8".to_string(),
            vec![1, 2],
            vec![127, 128],
        ),
    ];
    std::fs::write(
        snapshot.join("model.safetensors"),
        make_typed_safetensors(&typed_fixture_refs(&tensors)),
    )
    .unwrap();
    let index_path = snapshot.join("model.safetensors.index.json");
    std::fs::write(
        &index_path,
        format!(
            r#"{{"weight_map":{{"{weight}":"model.safetensors","{scales}":"model.safetensors"}}}}"#
        ),
    )
    .unwrap();

    let (manifest, visual) = build_manifest(GLM52_MODEL, snapshot, &index_path, None).unwrap();
    assert!(visual.is_empty());
    assert_eq!(manifest.dense_tensors.len(), 1);
    let tensor = &manifest.dense_tensors[0];
    assert_eq!(tensor.shape, vec![1, 64]);
    assert_eq!(
        tensor.q4_sources.as_ref().unwrap().source_format,
        DenseQ4SourceFormat::MlxMxfp4
    );
    assert_eq!(
        tensor.quantization,
        TensorQuantization::Q4 {
            group_size: GROUP_SIZE,
            format: DENSE_Q4_MXFP4_FORMAT.to_string(),
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        }
    );

    let destination = snapshot.join("dense.bin");
    write_dense_tensor_store(snapshot, &destination, &manifest.dense_tensors, None).unwrap();
    assert_eq!(
        std::fs::metadata(destination).unwrap().len(),
        tensor.byte_len
    );
}

#[test]
fn build_cache_accepts_qwen3_style_index_with_qknorm_and_shared_expert() {
    // Fixture derived from the Qwen3 MoE architecture:
    //   - q_norm / k_norm per attention layer (Qwen3 QK-norm)
    //   - shared_expert MLP that is always active and gated by shared_expert_gate
    //   - separate lm_head.weight (tie_word_embeddings=false)
    //   - 4 routable experts per layer
    // All of these tensors should be classified correctly (dense vs expert) and the
    // validator should accept the resulting manifest.
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();

    // config.json with Qwen3-style extra fields
    std::fs::write(
        snapshot.join("config.json"),
        br#"{
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_hidden_layers": 1,
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 300,
                "rope_theta": 1000000.0,
                "torch_dtype": "bfloat16",
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
                "tie_word_embeddings": false,
                "num_shared_experts": 1,
                "shared_expert_intermediate_size": 16
            }"#,
    )
    .unwrap();

    // Dense shard: all non-expert tensors including Qwen3-specific q_norm/k_norm and
    // shared_expert projections.  Shapes are consistent with the config above.
    // kv_width = num_key_value_heads(1) * (hidden_size / num_attention_heads) = 1 * (8/2) = 4
    let dense_shard = make_typed_safetensors(&[
        (
            "model.embed_tokens.weight",
            "BF16",
            vec![300, 8],
            &vec![0u8; 300 * 8 * 2],
        ),
        (
            "lm_head.weight",
            "BF16",
            vec![300, 8],
            &vec![0u8; 300 * 8 * 2],
        ),
        ("model.norm.weight", "BF16", vec![8], &vec![0u8; 8 * 2]),
        (
            "model.layers.0.self_attn.q_proj.weight",
            "BF16",
            vec![8, 8],
            &vec![0u8; 8 * 8 * 2],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            "BF16",
            vec![4, 8],
            &vec![0u8; 4 * 8 * 2],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            "BF16",
            vec![4, 8],
            &vec![0u8; 4 * 8 * 2],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            "BF16",
            vec![8, 8],
            &vec![0u8; 8 * 8 * 2],
        ),
        // QK-norm tensors present in Qwen3 MoE checkpoints
        (
            "model.layers.0.self_attn.q_norm.weight",
            "BF16",
            vec![4],
            &vec![0u8; 4 * 2],
        ),
        (
            "model.layers.0.self_attn.k_norm.weight",
            "BF16",
            vec![4],
            &vec![0u8; 4 * 2],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            "BF16",
            vec![8],
            &vec![0u8; 8 * 2],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            "BF16",
            vec![8],
            &vec![0u8; 8 * 2],
        ),
        (
            "model.layers.0.mlp.gate.weight",
            "BF16",
            vec![4, 8],
            &vec![0u8; 4 * 8 * 2],
        ),
        // Shared expert (always active, not gated): treated as dense, not packed
        (
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "BF16",
            vec![16, 8],
            &vec![0u8; 16 * 8 * 2],
        ),
        (
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            "BF16",
            vec![16, 8],
            &vec![0u8; 16 * 8 * 2],
        ),
        (
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            "BF16",
            vec![8, 16],
            &vec![0u8; 8 * 16 * 2],
        ),
        (
            "model.layers.0.mlp.shared_expert_gate.weight",
            "BF16",
            vec![1, 8],
            &vec![0u8; 8 * 2],
        ),
    ]);
    std::fs::write(snapshot.join("dense.safetensors"), dense_shard).unwrap();

    // Expert shard: 4 routed experts, each with gate/up/down projections.
    let mut expert_entries: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = Vec::new();
    let gate_bytes = vec![0u8; 16 * 8 * 2];
    let down_bytes = vec![0u8; 8 * 16 * 2];
    let names: Vec<(String, String, String)> = (0..4)
        .flat_map(|e| {
            let pfx = format!("model.layers.0.mlp.experts.{e}");
            [
                (
                    format!("{pfx}.gate_proj.weight"),
                    "gate".to_string(),
                    format!("{e}-gate"),
                ),
                (
                    format!("{pfx}.up_proj.weight"),
                    "up".to_string(),
                    format!("{e}-up"),
                ),
                (
                    format!("{pfx}.down_proj.weight"),
                    "down".to_string(),
                    format!("{e}-down"),
                ),
            ]
        })
        .collect();
    for (name, proj, _) in &names {
        let (shape, data): (Vec<usize>, &[u8]) = if proj == "down" {
            (vec![8, 16], &down_bytes)
        } else {
            (vec![16, 8], &gate_bytes)
        };
        expert_entries.push((name.as_str(), "BF16", shape, data.to_vec()));
    }
    std::fs::write(
        snapshot.join("expert.safetensors"),
        make_typed_safetensors(
            &expert_entries
                .iter()
                .map(|(n, d, s, b)| (*n, *d, s.clone(), b.as_slice()))
                .collect::<Vec<_>>(),
        ),
    )
    .unwrap();

    // Build weight_map: all tensors → their shard file
    let mut weight_map = serde_json::Map::new();
    // dense tensors
    for name in [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
    ] {
        weight_map.insert(
            name.to_string(),
            serde_json::Value::String("dense.safetensors".to_string()),
        );
    }
    // expert tensors
    for (name, _, _) in &names {
        weight_map.insert(
            name.clone(),
            serde_json::Value::String("expert.safetensors".to_string()),
        );
    }
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        serde_json::to_string(&serde_json::json!({"weight_map": weight_map})).unwrap(),
    )
    .unwrap();

    let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot)
        .expect("build should succeed for Qwen3-style snapshot with qknorm and shared_expert");

    // Validate: manifest should classify shared_expert and q/k_norm as dense, not expert.
    let manifest: FlashMoeManifest =
        serde_json::from_slice(&std::fs::read(&plan.tensor_manifest).unwrap()).unwrap();
    assert!(
        manifest
            .dense_tensors
            .iter()
            .any(|t| t.tensor.contains("q_norm")),
        "q_norm should be a dense tensor"
    );
    assert!(
        manifest
            .dense_tensors
            .iter()
            .any(|t| t.tensor.contains("k_norm")),
        "k_norm should be a dense tensor"
    );
    assert!(
        manifest
            .dense_tensors
            .iter()
            .any(|t| t.tensor.contains("shared_expert")),
        "shared_expert should be a dense tensor"
    );
    // 4 experts × 3 projections = 12 expert tensor entries
    assert_eq!(manifest.expert_tensors.len(), 12);

    // The validator must accept the resulting registry.
    let config = QwenModelConfig::from_file(&plan.model_config).unwrap();
    let registry = TensorRegistry::load(&plan.tensor_manifest).unwrap();
    validate_required_tensor_manifest(&config, &registry)
        .expect("Qwen3-style manifest should pass validation");
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
