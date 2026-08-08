//! Owner-local dense registry, resident projection, and runtime parity tests.

use super::*;
use crate::inference::flashmoe::cache::*;
use crate::inference::flashmoe::capabilities::*;
use crate::inference::flashmoe::math::*;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::inference::flashmoe::metal::*;
use crate::inference::flashmoe::model_family::*;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::inference::flashmoe::planning::*;
use crate::inference::flashmoe::scheduler::*;
use crate::inference::flashmoe::state::*;
use crate::inference::flashmoe::test_fixtures::*;
use crate::inference::flashmoe::text::QwenTokenizer;
use crate::inference::flashmoe::types::*;
use std::sync::Arc;

#[test]
fn flashmoe_parity_attention_layout_and_prefix_reuse_goldens() {
    let (mut config, mut manifest) =
        tiny_attention_manifest(&[AttentionLayerType::Full, AttentionLayerType::Linear]);
    config.num_attention_heads = 2;
    config.num_key_value_heads = Some(1);

    let registry = TensorRegistry::from_manifest(&manifest);
    let runtime = DenseTransformerRuntime::from_registry(&config, &registry).unwrap();
    let full = runtime.full_attention_layout(0).unwrap();
    assert_eq!(full.q_layout, FullAttentionQLayout::Standard);
    assert_eq!(full.q_projection_width, 8);
    assert_eq!(full.q_width, 8);
    assert_eq!(full.kv_width, 4);
    assert_eq!(full.head_dim, 4);
    assert_eq!(full.rotary_dim, 4);
    assert_eq!(full.num_q_heads, 2);
    assert_eq!(full.kv_heads, 1);

    let linear = runtime.linear_attention_layout(1).unwrap();
    assert_eq!(
        linear,
        LinearAttentionLayout {
            num_value_heads: 2,
            num_key_heads: 1,
            key_dim: 4,
            value_dim: 2,
            total_key_width: 4,
            total_value_width: 4,
            conv_dim: 12,
            conv_kernel_size: 3,
        }
    );
    assert_eq!(linear.conv_state_len(), 24);
    assert_eq!(linear.ssm_state_len(), 16);

    let q_name = attention_tensor_name(0, "q_proj");
    manifest
        .dense_tensors
        .iter_mut()
        .find(|tensor| tensor.tensor == q_name)
        .unwrap()
        .shape = vec![16, 8];
    let gated =
        DenseTransformerRuntime::from_registry(&config, &TensorRegistry::from_manifest(&manifest))
            .unwrap()
            .full_attention_layout(0)
            .unwrap();
    assert_eq!(gated.q_layout, FullAttentionQLayout::Gated);
    assert_eq!(gated.q_projection_width, 16);
    assert_eq!(gated.rotary_dim, 2);

    assert_eq!(
        reusable_session_prefix_len(&[10, 20, 30], &[10, 20, 30, 40]),
        Some(3)
    );
    assert_eq!(reusable_session_prefix_len(&[10, 20, 30], &[10, 20]), None);
    assert_eq!(
        reusable_session_prefix_len(&[10, 20, 30], &[10, 20, 99, 40]),
        None
    );
}

#[test]
fn resident_dense_mmap_projection_uses_full_row_dispatch() {
    assert_eq!(dense_projection_tile_rows(8192, 4096), 2048);
    assert_eq!(
        dense_projection_tile_rows_for_metal("BF16", 8192, 4096, true),
        4096
    );
    assert_eq!(
        dense_projection_tile_rows_for_metal("BF16", 8192, 4096, false),
        2048
    );
    assert_eq!(
        dense_projection_tile_rows_for_metal("U8", 8192, 4096, true),
        2048
    );
}

#[test]
fn dense_registry_validation_rejects_missing_lm_head_and_transformer_tensors() {
    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":128,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: Vec::new(),
        expert_tensors: Vec::new(),
        dense_tensors: Vec::new(),
    };
    let registry = TensorRegistry::from_manifest(&manifest);
    let err = validate_required_tensor_manifest(&config, &registry).unwrap_err();
    assert!(
        err.to_string().contains("model.embed_tokens.weight"),
        "{err:#}"
    );
}

/// Build a `FlashMoeManifest` containing every dense tensor required by `validate_required_tensor_manifest`
/// for a tiny 1-layer, 8-hidden-dim, 2-head, 1-kv-head, 128-vocab, 4-expert model.
fn minimal_dense_manifest(with_lm_head: bool) -> (QwenModelConfig, FlashMoeManifest) {
    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":128,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
    // kv_width = num_key_value_heads(1) * (hidden_size / num_attention_heads) = 1 * (8/2) = 4
    let mut tensors = vec![
        ("model.embed_tokens.weight", vec![128usize, 8]),
        ("model.norm.weight", vec![8]),
        ("model.layers.0.self_attn.q_proj.weight", vec![8, 8]),
        ("model.layers.0.self_attn.k_proj.weight", vec![4, 8]),
        ("model.layers.0.self_attn.v_proj.weight", vec![4, 8]),
        ("model.layers.0.self_attn.o_proj.weight", vec![8, 8]),
        ("model.layers.0.self_attn.q_norm.weight", vec![4]),
        ("model.layers.0.self_attn.k_norm.weight", vec![4]),
        ("model.layers.0.input_layernorm.weight", vec![8]),
        ("model.layers.0.post_attention_layernorm.weight", vec![8]),
        ("model.layers.0.mlp.gate.weight", vec![4, 8]),
    ];
    if with_lm_head {
        tensors.push(("lm_head.weight", vec![128, 8]));
    }
    let dense_tensors = tensors
        .iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let byte_len: u64 = shape.iter().product::<usize>() as u64 * 2; // BF16 = 2 bytes/elem
            DenseTensorRef {
                tensor: name.to_string(),
                shard: "shard.safetensors".to_string(),
                dtype: "BF16".to_string(),
                shape: shape.clone(),
                source_offsets: [0, byte_len],
                runtime_offset: i as u64 * 4096,
                byte_len,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }
        })
        .collect();
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["shard.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors,
    };
    (config, manifest)
}

fn make_dense_ref(tensor: &str, shape: Vec<usize>, slot: usize) -> DenseTensorRef {
    let byte_len: u64 = shape.iter().product::<usize>() as u64 * 2;
    DenseTensorRef {
        tensor: tensor.to_string(),
        shard: "hybrid.safetensors".to_string(),
        dtype: "BF16".to_string(),
        shape,
        source_offsets: [0, byte_len],
        runtime_offset: slot as u64 * 4096,
        byte_len,
        quantization: TensorQuantization::None,
        q4_sources: None,
    }
}

fn tiny_attention_manifest(
    layer_types: &[AttentionLayerType],
) -> (QwenModelConfig, FlashMoeManifest) {
    let (mut config, _) = minimal_dense_manifest(true);
    config.num_hidden_layers = layer_types.len();
    let mut slot = 0usize;
    let mut tensors = Vec::new();
    let mut push = |name: String, shape: Vec<usize>| {
        tensors.push(make_dense_ref(&name, shape, slot));
        slot += 1;
    };
    push(
        "model.embed_tokens.weight".to_string(),
        vec![config.vocab_size, config.hidden_size],
    );
    push("model.norm.weight".to_string(), vec![config.hidden_size]);
    push(
        "lm_head.weight".to_string(),
        vec![config.vocab_size, config.hidden_size],
    );

    for (layer, layer_type) in layer_types.iter().copied().enumerate() {
        push(
            layer_norm_tensor_name(layer, "input_layernorm"),
            vec![config.hidden_size],
        );
        push(
            layer_norm_tensor_name(layer, "post_attention_layernorm"),
            vec![config.hidden_size],
        );
        push(
            router_tensor_name(layer),
            vec![config.experts(), config.hidden_size],
        );
        match layer_type {
            AttentionLayerType::Mla => {
                panic!("tiny Qwen attention fixture does not construct MLA layers")
            }
            AttentionLayerType::Full => {
                push(
                    attention_tensor_name(layer, "q_proj"),
                    vec![config.hidden_size, config.hidden_size],
                );
                push(
                    attention_tensor_name(layer, "k_proj"),
                    vec![4, config.hidden_size],
                );
                push(
                    attention_tensor_name(layer, "v_proj"),
                    vec![4, config.hidden_size],
                );
                push(
                    attention_tensor_name(layer, "o_proj"),
                    vec![config.hidden_size, config.hidden_size],
                );
                push(layer_norm_tensor_name(layer, "self_attn.q_norm"), vec![4]);
                push(layer_norm_tensor_name(layer, "self_attn.k_norm"), vec![4]);
            }
            AttentionLayerType::Linear => {
                push(
                    linear_attention_tensor_name(layer, "in_proj_qkv"),
                    vec![12, config.hidden_size],
                );
                push(
                    linear_attention_tensor_name(layer, "in_proj_z"),
                    vec![4, config.hidden_size],
                );
                push(
                    linear_attention_tensor_name(layer, "in_proj_b"),
                    vec![2, config.hidden_size],
                );
                push(
                    linear_attention_tensor_name(layer, "in_proj_a"),
                    vec![2, config.hidden_size],
                );
                push(linear_attention_tensor_name(layer, "conv1d"), vec![12, 3]);
                push(linear_attention_scalar_tensor_name(layer, "A_log"), vec![2]);
                push(
                    linear_attention_scalar_tensor_name(layer, "dt_bias"),
                    vec![2],
                );
                push(linear_attention_tensor_name(layer, "norm"), vec![2]);
                push(
                    linear_attention_tensor_name(layer, "out_proj"),
                    vec![config.hidden_size, 4],
                );
            }
        }
    }

    let manifest = FlashMoeManifest {
        model: "hf://example/tiny-attention".to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["tiny.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: tensors,
    };
    (config, manifest)
}

fn assert_manifest_attention_kinds(layer_types: &[AttentionLayerType]) {
    let (config, manifest) = tiny_attention_manifest(layer_types);
    let registry = TensorRegistry::from_manifest(&manifest);
    validate_required_tensor_manifest(&config, &registry)
        .expect("manifest-driven attention schedule should validate");
    let runtime = DenseTransformerRuntime::from_registry(&config, &registry)
        .expect("manifest-driven attention schedule should build runtime layouts");

    for (layer, layer_type) in layer_types.iter().copied().enumerate() {
        match layer_type {
            AttentionLayerType::Mla => {
                panic!("tiny Qwen attention fixture does not construct MLA layers")
            }
            AttentionLayerType::Full => {
                assert_eq!(runtime.layer_kind(layer), FlashMoeLayerKind::FullAttention);
                runtime
                    .full_attention_layout(layer)
                    .expect("full-attention layer should have full layout");
                assert!(
                    runtime.linear_attention_layout(layer).is_err(),
                    "full-attention layer {layer} should not have a linear layout"
                );
            }
            AttentionLayerType::Linear => {
                assert_eq!(
                    runtime.layer_kind(layer),
                    FlashMoeLayerKind::LinearAttention
                );
                runtime
                    .linear_attention_layout(layer)
                    .expect("linear-attention layer should have linear layout");
                assert!(
                    runtime.full_attention_layout(layer).is_err(),
                    "linear-attention layer {layer} should not have a full layout"
                );
            }
        }
    }
}

#[test]
fn full_attention_manifest_requires_qk_norm_bindings() {
    let (config, mut manifest) = minimal_dense_manifest(true);
    manifest
        .dense_tensors
        .retain(|tensor| tensor.tensor != "model.layers.0.self_attn.k_norm.weight");
    let registry = TensorRegistry::from_manifest(&manifest);

    let error = validate_required_tensor_manifest(&config, &registry).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("model.layers.0.self_attn.k_norm.weight"),
        "{error:#}"
    );
}

#[test]
fn validate_rejects_configured_shared_expert_without_gate() {
    let (mut config, mut manifest) = minimal_dense_manifest(true);
    config.num_shared_experts = Some(1);
    config.shared_expert_intermediate_size = Some(16);
    let mut slot = manifest.dense_tensors.len();
    for (name, shape) in [
        (
            shared_expert_tensor_name(0, "gate_proj"),
            vec![16, config.hidden_size],
        ),
        (
            shared_expert_tensor_name(0, "up_proj"),
            vec![16, config.hidden_size],
        ),
        (
            shared_expert_tensor_name(0, "down_proj"),
            vec![config.hidden_size, 16],
        ),
    ] {
        manifest
            .dense_tensors
            .push(make_dense_ref(&name, shape, slot));
        slot += 1;
    }

    let registry = TensorRegistry::from_manifest(&manifest);
    let err = validate_required_tensor_manifest(&config, &registry).unwrap_err();
    assert!(
        err.to_string().contains("shared_expert_gate.weight"),
        "{err:#}"
    );
}

#[test]
fn manifest_attention_detection_accepts_all_full_attention() {
    assert_manifest_attention_kinds(&[
        AttentionLayerType::Full,
        AttentionLayerType::Full,
        AttentionLayerType::Full,
        AttentionLayerType::Full,
    ]);
}

#[test]
fn manifest_attention_detection_accepts_qwen35_mixed_schedule() {
    assert_manifest_attention_kinds(&[
        AttentionLayerType::Linear,
        AttentionLayerType::Linear,
        AttentionLayerType::Linear,
        AttentionLayerType::Full,
        AttentionLayerType::Linear,
        AttentionLayerType::Linear,
        AttentionLayerType::Linear,
        AttentionLayerType::Full,
    ]);
}

#[test]
fn manifest_attention_detection_accepts_non_every_fourth_mixed_schedule() {
    assert_manifest_attention_kinds(&[
        AttentionLayerType::Full,
        AttentionLayerType::Linear,
        AttentionLayerType::Full,
        AttentionLayerType::Linear,
    ]);
}

#[test]
fn manifest_attention_detection_rejects_conflicting_layer_layouts() {
    let (config, mut manifest) = tiny_attention_manifest(&[AttentionLayerType::Full]);
    let slot = manifest.dense_tensors.len();
    manifest.dense_tensors.push(make_dense_ref(
        &linear_attention_tensor_name(0, "in_proj_qkv"),
        vec![12, config.hidden_size],
        slot,
    ));
    let registry = TensorRegistry::from_manifest(&manifest);
    let err = validate_required_tensor_manifest(&config, &registry).unwrap_err();

    assert!(err.to_string().contains("both linear-attention"), "{err:#}");
    assert!(err.to_string().contains("full-attention"), "{err:#}");
}

#[test]
fn qwen_q4_graph_binding_rejects_projection_outside_resident_store() {
    let shape = vec![4, 4];
    let group_size = 2;
    let layout =
        dense_q4_layout_with_scale_bias_dtype(&shape, group_size, EXPERT_SCALE_BIAS_DTYPE_BF16)
            .unwrap();
    let tensor_name = "model.layers.0.self_attn.q_proj.weight";
    let byte_offset = 64u64;
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["dense.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: tensor_name.to_string(),
            shard: "dense.safetensors".to_string(),
            dtype: "U32".to_string(),
            shape,
            source_offsets: [0, layout.total_bytes as u64],
            runtime_offset: byte_offset,
            byte_len: layout.total_bytes as u64,
            quantization: TensorQuantization::Q4 {
                group_size,
                format: DENSE_Q4_FORMAT.to_string(),
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            },
            q4_sources: None,
        }],
    };
    let registry = TensorRegistry::from_manifest(&manifest);
    let required_len = byte_offset + layout.total_bytes as u64;

    require_resident_q4_graph_projection(
        QwenMoeFamily::Qwen35A17B,
        &registry,
        required_len,
        "CMD1 full-attention projection",
        tensor_name,
        4,
        4,
    )
    .unwrap();

    let err = require_resident_q4_graph_projection(
        QwenMoeFamily::Qwen35A17B,
        &registry,
        required_len - 1,
        "CMD1 full-attention projection",
        tensor_name,
        4,
        4,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("unsupported resolved Qwen35A17B Q4 CMD1 full-attention projection"),
        "{err:#}"
    );
}

#[test]
fn validate_accepts_hybrid_gated_deltanet_manifest() {
    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_5_moe","architectures":["Qwen3_5MoeForCausalLM"],"num_hidden_layers":4,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":4,"moe_intermediate_size":1024}"#,
        )
        .unwrap();
    let model_layout = QwenMoeModelLayout::from_config(QWEN35_MODEL, &config).unwrap();
    let mut slot = 0usize;
    let mut tensors = Vec::new();
    let mut push = |name: String, shape: Vec<usize>| {
        tensors.push(make_dense_ref(&name, shape, slot));
        slot += 1;
    };
    push(
        "model.embed_tokens.weight".to_string(),
        vec![config.vocab_size, config.hidden_size],
    );
    push("model.norm.weight".to_string(), vec![config.hidden_size]);
    push(
        "lm_head.weight".to_string(),
        vec![config.vocab_size, config.hidden_size],
    );
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_width = config.kv_heads() * head_dim;
    for layer in 0..config.num_hidden_layers {
        push(
            layer_norm_tensor_name(layer, "input_layernorm"),
            vec![config.hidden_size],
        );
        push(
            layer_norm_tensor_name(layer, "post_attention_layernorm"),
            vec![config.hidden_size],
        );
        push(
            router_tensor_name(layer),
            vec![config.experts(), config.hidden_size],
        );
        if model_layout.layer_kind(layer) == QwenMoeLayerKind::FullAttention {
            push(
                attention_tensor_name(layer, "q_proj"),
                vec![config.hidden_size, config.hidden_size],
            );
            push(
                attention_tensor_name(layer, "k_proj"),
                vec![kv_width, config.hidden_size],
            );
            push(
                attention_tensor_name(layer, "v_proj"),
                vec![kv_width, config.hidden_size],
            );
            push(
                attention_tensor_name(layer, "o_proj"),
                vec![config.hidden_size, config.hidden_size],
            );
            push(
                layer_norm_tensor_name(layer, "self_attn.q_norm"),
                vec![head_dim],
            );
            push(
                layer_norm_tensor_name(layer, "self_attn.k_norm"),
                vec![head_dim],
            );
        } else {
            push(
                linear_attention_tensor_name(layer, "in_proj_qkv"),
                vec![LINEAR_CONV_DIM, config.hidden_size],
            );
            push(
                linear_attention_tensor_name(layer, "in_proj_z"),
                vec![LINEAR_TOTAL_VALUE, config.hidden_size],
            );
            push(
                linear_attention_tensor_name(layer, "in_proj_b"),
                vec![LINEAR_NUM_V_HEADS, config.hidden_size],
            );
            push(
                linear_attention_tensor_name(layer, "in_proj_a"),
                vec![LINEAR_NUM_V_HEADS, config.hidden_size],
            );
            push(
                linear_attention_tensor_name(layer, "conv1d"),
                vec![LINEAR_CONV_DIM, CONV_KERNEL_SIZE],
            );
            push(
                linear_attention_scalar_tensor_name(layer, "A_log"),
                vec![LINEAR_NUM_V_HEADS],
            );
            push(
                linear_attention_scalar_tensor_name(layer, "dt_bias"),
                vec![LINEAR_NUM_V_HEADS],
            );
            push(
                linear_attention_tensor_name(layer, "norm"),
                vec![LINEAR_VALUE_DIM],
            );
            push(
                linear_attention_tensor_name(layer, "out_proj"),
                vec![config.hidden_size, LINEAR_TOTAL_VALUE],
            );
        }
    }
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["hybrid.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: tensors,
    };
    let registry = TensorRegistry::from_manifest(&manifest);
    validate_required_tensor_manifest(&config, &registry)
        .expect("hybrid manifest should validate for GatedDeltaNet/full-attn layer mix");
}

#[test]
fn validate_accepts_hf_conv1d_singleton_axis_shape() {
    let tensor_name = linear_attention_tensor_name(0, "conv1d");
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["hybrid.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: tensor_name.clone(),
            shard: "hybrid.safetensors".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![LINEAR_CONV_DIM, 1, CONV_KERNEL_SIZE],
            source_offsets: [0, 0],
            runtime_offset: 0,
            byte_len: 0,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }],
    };
    let registry = TensorRegistry::from_manifest(&manifest);

    assert_eq!(
        require_conv1d_tensor_shape(&registry, &tensor_name)
            .expect("HF conv1d [channels, 1, kernel] shape should validate"),
        (LINEAR_CONV_DIM, CONV_KERNEL_SIZE)
    );
}

#[test]
fn linear_attention_layout_infers_non_qwen35_dimensions() {
    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":128,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
    let mut slot = 0usize;
    let mut tensors = Vec::new();
    let mut push = |name: String, shape: Vec<usize>| {
        tensors.push(make_dense_ref(&name, shape, slot));
        slot += 1;
    };
    push(
        "model.embed_tokens.weight".to_string(),
        vec![config.vocab_size, config.hidden_size],
    );
    push("model.norm.weight".to_string(), vec![config.hidden_size]);
    push(
        layer_norm_tensor_name(0, "input_layernorm"),
        vec![config.hidden_size],
    );
    push(
        layer_norm_tensor_name(0, "post_attention_layernorm"),
        vec![config.hidden_size],
    );
    push(
        router_tensor_name(0),
        vec![config.experts(), config.hidden_size],
    );
    push(
        linear_attention_tensor_name(0, "in_proj_qkv"),
        vec![12, config.hidden_size],
    );
    push(
        linear_attention_tensor_name(0, "in_proj_z"),
        vec![4, config.hidden_size],
    );
    push(
        linear_attention_tensor_name(0, "in_proj_b"),
        vec![2, config.hidden_size],
    );
    push(
        linear_attention_tensor_name(0, "in_proj_a"),
        vec![2, config.hidden_size],
    );
    push(linear_attention_tensor_name(0, "conv1d"), vec![12, 3]);
    push(linear_attention_scalar_tensor_name(0, "A_log"), vec![2]);
    push(linear_attention_scalar_tensor_name(0, "dt_bias"), vec![2]);
    push(linear_attention_tensor_name(0, "norm"), vec![2]);
    push(
        linear_attention_tensor_name(0, "out_proj"),
        vec![config.hidden_size, 4],
    );

    let manifest = FlashMoeManifest {
        model: "hf://example/tiny-linear".to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["tiny.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: tensors,
    };
    let registry = TensorRegistry::from_manifest(&manifest);
    validate_required_tensor_manifest(&config, &registry)
        .expect("variable linear attention layout should validate");
    let runtime = DenseTransformerRuntime::from_registry(&config, &registry).unwrap();
    let layout = runtime.linear_attention_layout(0).unwrap();

    assert_eq!(layout.num_value_heads, 2);
    assert_eq!(layout.num_key_heads, 1);
    assert_eq!(layout.key_dim, 4);
    assert_eq!(layout.value_dim, 2);
    assert_eq!(layout.conv_dim, 12);
    assert_eq!(layout.conv_kernel_size, 3);
    assert_eq!(layout.conv_state_len(), 24);
    assert_eq!(layout.ssm_state_len(), 16);
}

#[test]
fn linear_attention_key_dim_uses_qwen35_default_only_for_known_shape() {
    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":4096,"num_attention_heads":31,"num_key_value_heads":1,"vocab_size":128,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();

    let key_dim = infer_linear_attention_key_dim(
        &config,
        LINEAR_TOTAL_KEY,
        LINEAR_TOTAL_VALUE,
        LINEAR_VALUE_DIM,
    )
    .expect("exact Qwen3.5 linear-attention shape should allow the default key dim");
    assert_eq!(key_dim, LINEAR_KEY_DIM);

    let err = infer_linear_attention_key_dim(
        &config,
        LINEAR_TOTAL_KEY,
        LINEAR_TOTAL_VALUE,
        LINEAR_VALUE_DIM * 32,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("not divisible by config head_dim"),
        "{err:#}"
    );
}

#[test]
fn linear_attention_qk_normalization_matches_qwen35_reference_scaling() {
    let layout = LinearAttentionLayout {
        num_value_heads: 4,
        num_key_heads: 2,
        key_dim: 4,
        value_dim: 3,
        total_key_width: 8,
        total_value_width: 12,
        conv_dim: 28,
        conv_kernel_size: 4,
    };
    let mut q = vec![1.0, 2.0, -3.0, 4.0, -1e-6, 2e-6, -3e-6, 4e-6];
    let mut k = vec![0.5, -1.5, 2.5, -3.5, 4e-6, -2e-6, 1e-6, 0.5e-6];
    let mut expected_q = q.clone();
    let mut expected_k = k.clone();

    for head in 0..layout.num_key_heads {
        let start = head * layout.key_dim;
        let end = start + layout.key_dim;
        let q_sum_sq = expected_q[start..end]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let q_inv_rms = (q_sum_sq / layout.key_dim as f32 + 1e-6).sqrt().recip();
        let k_sum_sq = expected_k[start..end]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let k_inv_rms = (k_sum_sq / layout.key_dim as f32 + 1e-6).sqrt().recip();
        let inv_scale = 1.0f32 / (layout.key_dim as f32).sqrt();
        for value in &mut expected_q[start..end] {
            *value *= q_inv_rms * inv_scale * inv_scale;
        }
        for value in &mut expected_k[start..end] {
            *value *= k_inv_rms * inv_scale;
        }
    }

    normalize_linear_attention_qk_in_place(layout, &mut q, &mut k).unwrap();

    for (actual, expected) in q.iter().zip(expected_q.iter()) {
        assert_close(*actual, *expected);
    }
    for (actual, expected) in k.iter().zip(expected_k.iter()) {
        assert_close(*actual, *expected);
    }
}

#[test]
fn linear_attention_qkv_projection_reorders_key_head_groups_for_conv() {
    let layout = LinearAttentionLayout {
        num_value_heads: 4,
        num_key_heads: 2,
        key_dim: 2,
        value_dim: 3,
        total_key_width: 4,
        total_value_width: 12,
        conv_dim: 20,
        conv_kernel_size: 4,
    };
    let mut qkv = vec![
        10.0, 11.0, // head 0 q
        20.0, 21.0, // head 0 k
        30.0, 31.0, 32.0, 33.0, 34.0, 35.0, // head 0 value heads
        40.0, 41.0, // head 1 q
        50.0, 51.0, // head 1 k
        60.0, 61.0, 62.0, 63.0, 64.0, 65.0, // head 1 value heads
    ];

    reorder_grouped_linear_qkv_projection(&mut qkv, layout).unwrap();

    assert_eq!(
        qkv,
        vec![
            10.0, 11.0, 40.0, 41.0, // all q
            20.0, 21.0, 50.0, 51.0, // all k
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0,
        ]
    );
}

#[test]
fn validate_accepts_tied_lm_head() {
    // lm_head.weight absent → tied embeddings; validator should pass.
    let (config, manifest) = minimal_dense_manifest(false);
    let registry = TensorRegistry::from_manifest(&manifest);
    validate_required_tensor_manifest(&config, &registry)
        .expect("tied-embedding manifest should pass validation");
}

#[test]
fn validate_accepts_separate_lm_head() {
    // lm_head.weight present with correct shape → should pass.
    let (config, manifest) = minimal_dense_manifest(true);
    let registry = TensorRegistry::from_manifest(&manifest);
    validate_required_tensor_manifest(&config, &registry)
        .expect("manifest with separate lm_head should pass validation");
}

#[test]
fn dense_registry_validation_accepts_native_mlx_q4_dense_tensors() {
    let (config, mut manifest) = minimal_dense_manifest(true);
    let embed = manifest
        .dense_tensors
        .iter_mut()
        .find(|tensor| tensor.tensor == "model.embed_tokens.weight")
        .expect("minimal manifest should include embeddings");
    embed.dtype = "U32".to_string();
    embed.quantization = TensorQuantization::Q4 {
        group_size: GROUP_SIZE,
        format: DENSE_Q4_MLX_FORMAT.to_string(),
        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
    };
    let layout = dense_q4_layout_with_scale_bias_dtype(
        &embed.shape,
        GROUP_SIZE,
        EXPERT_SCALE_BIAS_DTYPE_BF16,
    )
    .unwrap();
    embed.byte_len = layout.total_bytes as u64;

    let registry = TensorRegistry::from_manifest(&manifest);

    validate_required_tensor_manifest(&config, &registry)
        .expect("native MLX q4 dense tensors should validate by quantization metadata");
}

#[test]
fn validate_rejects_misshapen_lm_head() {
    let (config, mut manifest) = minimal_dense_manifest(true);
    // Corrupt the lm_head shape so it has wrong dimensions.
    for t in &mut manifest.dense_tensors {
        if t.tensor == "lm_head.weight" {
            t.shape = vec![128, 16]; // should be [128, 8]
        }
    }
    let registry = TensorRegistry::from_manifest(&manifest);
    let err = validate_required_tensor_manifest(&config, &registry).unwrap_err();
    assert!(
        err.to_string().contains("lm_head.weight"),
        "expected lm_head shape error, got: {err:#}"
    );
    assert!(
        err.to_string().contains("expected"),
        "expected shape mismatch message, got: {err:#}"
    );
}

#[test]
fn validate_accepts_mlx_conv1d_trailing_singleton_axis_shape() {
    let tensor_name = linear_attention_tensor_name(0, "conv1d");
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["hybrid.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: tensor_name.clone(),
            shard: "hybrid.safetensors".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![LINEAR_CONV_DIM, CONV_KERNEL_SIZE, 1],
            source_offsets: [0, 0],
            runtime_offset: 0,
            byte_len: 0,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }],
    };
    let registry = TensorRegistry::from_manifest(&manifest);

    assert_eq!(
        require_conv1d_tensor_shape(&registry, &tensor_name)
            .expect("MLX conv1d [channels, kernel, 1] shape should validate"),
        (LINEAR_CONV_DIM, CONV_KERNEL_SIZE)
    );
}

#[test]
fn validate_accepts_expert_tensors_absent_from_registry() {
    // Expert tensors are packed into ExpertSlotStore files and need not all appear in the
    // dense registry.  The validator must not reject a registry that has no expert entries.
    let (config, manifest) = minimal_dense_manifest(false);
    assert!(manifest.expert_tensors.is_empty());
    let registry = TensorRegistry::from_manifest(&manifest);
    validate_required_tensor_manifest(&config, &registry)
        .expect("registry without expert tensors should still pass dense validation");
}

#[test]
fn tensor_registry_aliases_qwen35_language_model_prefix() {
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["dense.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: "model.language_model.embed_tokens.weight".to_string(),
            shard: "dense.safetensors".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![248320, 4096],
            source_offsets: [0, 0],
            runtime_offset: 0,
            byte_len: 0,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }],
    };
    let registry = TensorRegistry::from_manifest(&manifest);

    assert!(
        registry
            .tensor("model.language_model.embed_tokens.weight")
            .is_some()
    );
    assert!(registry.tensor("model.embed_tokens.weight").is_some());
}

#[test]
fn qwen_config_deserializes_qwen3_moe_extra_fields() {
    // Real Qwen3 MoE checkpoints include additional config fields that should be parsed
    // without error and reflected in the struct.
    let json = br#"{
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_hidden_layers": 60,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "torch_dtype": "bfloat16",
            "num_experts": 512,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1536,
            "tie_word_embeddings": false,
            "num_shared_experts": 1,
            "shared_expert_intermediate_size": 1536
        }"#;
    let config: QwenModelConfig = serde_json::from_slice(json).unwrap();
    assert_eq!(config.tie_word_embeddings, Some(false));
    assert_eq!(config.num_shared_experts, Some(1));
    assert_eq!(config.shared_expert_intermediate_size, Some(1536));
    assert_eq!(config.experts(), 512);
    config.validate().unwrap();
}

#[test]
fn qwen_config_deserializes_qwen35_nested_text_and_vision_fields() {
    let json = br#"{
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "image_token_id": 248056,
            "model_type": "qwen3_5_moe",
            "text_config": {
                "dtype": "bfloat16",
                "head_dim": 256,
                "hidden_size": 4096,
                "max_position_embeddings": 262144,
                "model_type": "qwen3_5_moe_text",
                "moe_intermediate_size": 1024,
                "num_attention_heads": 32,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "num_hidden_layers": 60,
                "num_key_value_heads": 2,
                "shared_expert_intermediate_size": 1024,
                "vocab_size": 248320,
                "rope_parameters": {
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25
                }
            },
            "tie_word_embeddings": false,
            "vision_config": {
                "depth": 27,
                "deepstack_visual_indexes": [5, 11, 17],
                "hidden_size": 1152,
                "in_channels": 3,
                "intermediate_size": 4304,
                "num_heads": 16,
                "out_hidden_size": 4096,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            },
            "vision_end_token_id": 248054,
            "vision_start_token_id": 248053
        }"#;

    let config: QwenModelConfig = serde_json::from_slice(json).unwrap();
    assert_eq!(config.num_hidden_layers, 60);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.head_dim, Some(256));
    assert_eq!(config.full_attention_head_dim(), 256);
    assert_eq!(config.derived_attention_head_dim(), 128);
    assert_eq!(config.num_key_value_heads, Some(2));
    assert_eq!(config.vocab_size, 248320);
    assert_eq!(config.rope_theta, Some(10000000.0));
    assert_eq!(config.partial_rotary_factor, Some(0.25));
    assert_eq!(config.torch_dtype.as_deref(), Some("bfloat16"));
    assert_eq!(config.num_experts_per_tok, Some(10));
    assert_eq!(config.tie_word_embeddings, Some(false));

    let vision = config.vision_config.as_ref().unwrap();
    assert_eq!(vision.depth, 27);
    assert_eq!(vision.embed_dim, 1152);
    assert_eq!(vision.num_heads, 16);
    assert_eq!(vision.patch_size, 16);
    assert_eq!(vision.merge_size, 2);
    assert_eq!(vision.temporal_patch_size, 2);
    assert_eq!(vision.in_chans, 3);
    assert_eq!(vision.deepstack_visual_indexes, vec![5, 11, 17]);
    assert_eq!(vision.out_hidden_size, Some(4096));
    assert_eq!(vision.patch_flat_dim(), 3 * 2 * 16 * 16);
    assert_eq!(vision.mlp_hidden_size(), 4304);

    config.validate().unwrap();
}

#[test]
fn qwen_config_deserializes_mrope_section_from_rope_scaling() {
    let json = br#"{
            "model_type": "qwen3_vl",
            "text_config": {
                "hidden_size": 128,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "vocab_size": 1024,
                "rope_scaling": {
                    "rope_theta": 1000000.0,
                    "mrope_section": [24, 20, 20]
                }
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 64,
                "num_heads": 4
            }
        }"#;

    let config: QwenModelConfig = serde_json::from_slice(json).unwrap();
    assert_eq!(config.rope_theta, Some(1_000_000.0));
    assert_eq!(config.mrope_section, Some(DEFAULT_MROPE_SECTION));
    assert_eq!(config.text_mrope_section(), Some(DEFAULT_MROPE_SECTION));
    config.validate().unwrap();
}

#[test]
fn dense_store_reads_registered_tensor_rows_by_dtype() {
    let tmp = tempfile::tempdir().unwrap();
    let dense_path = tmp.path().join("model_weights.bin");
    let manifest_path = tmp.path().join("model_weights.json");
    let mut bytes = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&dense_path, &bytes).unwrap();
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["dense.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: "model.layers.0.self_attn.q_proj.weight".to_string(),
            shard: "dense.safetensors".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 2],
            source_offsets: [0, bytes.len() as u64],
            runtime_offset: 0,
            byte_len: bytes.len() as u64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }],
    };
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let store = DenseStore::open(dense_path, manifest_path).unwrap();
    let row = store
        .read_tensor_row_f32("model.layers.0.self_attn.q_proj.weight", 1, 2)
        .unwrap()
        .unwrap();
    assert_eq!(row, vec![3.0, 4.0]);
    let tile = store
        .read_tensor_rows_f32("model.layers.0.self_attn.q_proj.weight", 0, 2)
        .unwrap();
    assert_eq!(tile, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        store
            .resident
            .lock()
            .expect("dense tensor cache poisoned")
            .bytes,
        0
    );
    let projected = store
        .project(0, "q_proj", &[1.0, 1.0], 2)
        .expect("registered dense projection should decode F32 weights");
    assert_eq!(projected.len(), 2);
    assert!(projected[1] > projected[0]);
}

#[test]
fn router_scores_use_cached_full_tensor_matvec() {
    let tmp = tempfile::tempdir().unwrap();
    let dense_path = tmp.path().join("model_weights.bin");
    let manifest_path = tmp.path().join("model_weights.json");
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&dense_path, &bytes).unwrap();
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["dense.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: router_tensor_name(0),
            shard: "dense.safetensors".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3],
            source_offsets: [0, bytes.len() as u64],
            runtime_offset: 0,
            byte_len: bytes.len() as u64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }],
    };
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let store = DenseStore::open(dense_path, manifest_path).unwrap();
    let config = QwenModelConfig {
        model_type: Some("qwen".to_string()),
        architectures: None,
        num_hidden_layers: 1,
        hidden_size: 3,
        num_attention_heads: 1,
        head_dim: None,
        num_key_value_heads: Some(1),
        vocab_size: 16,
        rope_theta: None,
        partial_rotary_factor: None,
        torch_dtype: Some("float32".to_string()),
        num_experts: Some(2),
        num_experts_per_tok: Some(1),
        norm_topk_prob: None,
        moe_intermediate_size: Some(4),
        intermediate_size: None,
        max_position_embeddings: Some(8),
        full_attention_interval: None,
        linear_attention: None,
        mrope_section: None,
        tie_word_embeddings: None,
        num_shared_experts: None,
        shared_expert_intermediate_size: None,
        vision_config: None,
        glm: None,
    };
    let mut graph_layout = QwenMoeModelLayout::from_config(QWEN35_MODEL, &config).unwrap();
    graph_layout.hidden_size = GROUP_SIZE;
    graph_layout.moe_intermediate_size = GROUP_SIZE;
    let capability_plan = FlashMoeCapabilityPlan::for_model_layout(&graph_layout).unwrap();
    let scheduled_graph = FlashMoeScheduledGraph::from_capabilities(&capability_plan).unwrap();
    let scheduled_routing = scheduled_graph
        .build_routing_topk(0, 2, 1, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap();
    let projection = store.router_score_projection_descriptor(0, 2, 3).unwrap();
    let projection_ref = projection
        .as_ref()
        .expect("registered router tensor should produce a typed projection descriptor");
    assert_eq!(projection_ref.layer, 0);
    assert_eq!(projection_ref.experts, 2);
    assert_eq!(projection_ref.hidden_width, 3);
    assert_eq!(projection_ref.tensor_name, router_tensor_name(0));
    let command = scheduled_routing
        .build_score_projection_command(projection, 3)
        .unwrap();

    let hidden = [0.5, -1.0, 2.0];
    let execution = command.projection_execution().unwrap();
    let score_plan = execution.score_plan(hidden.len()).unwrap();
    let scores = store.router_scores(score_plan, &hidden).unwrap();
    let routing_command = command.into_routing_command(scores).unwrap();

    assert_eq!(
        routing_command.source,
        ScheduledRoutingCandidateSource::CpuRouterScores
    );
    assert_eq!(routing_command.layer, 0);
    assert_eq!(routing_command.active_experts, 1);
    assert_eq!(routing_command.routes.len(), 1);
    assert_eq!(routing_command.routes[0].0, 1);
    let expected_probability = 1.0 / (1.0 + (-4.5f32).exp());
    assert!((routing_command.routes[0].1 - expected_probability).abs() < 1e-6);
    assert_eq!(
        store
            .decoded_tensor_tiles
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn dense_store_caches_small_norm_weights() {
    let tmp = tempfile::tempdir().unwrap();
    let dense_path = tmp.path().join("model_weights.bin");
    let manifest_path = tmp.path().join("model_weights.json");
    let mut bytes = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&dense_path, &bytes).unwrap();
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["dense.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: "model.layers.0.input_layernorm.weight".to_string(),
            shard: "dense.safetensors".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4],
            source_offsets: [0, bytes.len() as u64],
            runtime_offset: 0,
            byte_len: bytes.len() as u64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }],
    };
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let store = DenseStore::open(dense_path, manifest_path).unwrap();

    let first = store
        .norm_weight("model.layers.0.input_layernorm.weight", 4)
        .unwrap()
        .unwrap();
    let second = store
        .norm_weight("model.layers.0.input_layernorm.weight", 4)
        .unwrap()
        .unwrap();

    assert_eq!(first, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(second, first);
    assert_eq!(
        store
            .norm_weights
            .lock()
            .expect("dense norm weight cache poisoned")
            .len(),
        1
    );
}

#[test]
fn dense_store_rms_norm_uses_small_weight_cache_without_decoded_tile_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let dense_path = tmp.path().join("model_weights.bin");
    let manifest_path = tmp.path().join("model_weights.json");
    let weights = [0.5f32, 1.0, 1.5, 2.0];
    let mut bytes = Vec::new();
    for value in weights {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&dense_path, &bytes).unwrap();
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["dense.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: "model.layers.0.post_attention_layernorm.weight".to_string(),
            shard: "dense.safetensors".to_string(),
            dtype: "F32".to_string(),
            shape: vec![weights.len()],
            source_offsets: [0, bytes.len() as u64],
            runtime_offset: 0,
            byte_len: bytes.len() as u64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }],
    };
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let store = DenseStore::open(dense_path, manifest_path).unwrap();

    let input = [3.0f32, 4.0, -5.0, 12.0];
    let actual = store
        .rms_norm("model.layers.0.post_attention_layernorm.weight", &input)
        .unwrap();
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + 1e-6).sqrt().recip();
    let expected: Vec<f32> = input
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * scale * weight)
        .collect();

    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= 1e-6,
            "rms norm element {idx} diverged: actual={actual}, expected={expected}"
        );
    }
    assert_eq!(
        store
            .norm_weights
            .lock()
            .expect("dense norm weight cache poisoned")
            .len(),
        1
    );
    assert_eq!(
        store
            .decoded_tiles
            .lock()
            .expect("decoded tile cache poisoned")
            .bytes,
        0
    );
}

#[test]
fn dense_bf16_store_projects_synthetic_tensor_like_runtime_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let dense_path = tmp.path().join("model_weights.bin");
    let manifest_path = tmp.path().join("model_weights.json");
    let tensor_name = "model.layers.0.self_attn.o_proj.weight";
    let rows = 19;
    let cols = 7;
    let values: Vec<f32> = (0..rows * cols)
        .map(|idx| ((idx as f32) * 0.37).sin() * 0.75 - ((idx % cols) as f32) * 0.03125)
        .collect();
    let bytes = bf16_tensor_bytes(&values);
    fs::write(&dense_path, &bytes).unwrap();
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["dense.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: tensor_name.to_string(),
            shard: "dense.safetensors".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![rows, cols],
            source_offsets: [0, bytes.len() as u64],
            runtime_offset: 0,
            byte_len: bytes.len() as u64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }],
    };
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let store = DenseStore::open(dense_path, manifest_path).unwrap();
    let input = vec![0.25, -1.0, 0.5, 2.0, -0.75, 1.5, -0.125];
    let decoded = decode_dense_tensor_f32("BF16", &bytes).unwrap();
    let expected = cpu_dense_matvec(&decoded, &input, rows, cols);
    let projected = store
        .project_dense_tensor_with_metal(None, tensor_name, &input, rows)
        .unwrap()
        .unwrap();

    for (row, (actual, expected)) in projected.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= 1e-6,
            "row {row}: BF16 projection {actual} diverged from decoded reference {expected}"
        );
    }
    assert_eq!(store.decoded_full_tensor_count(), 1);
}

#[test]
fn dense_q4_store_projects_synthetic_tensor_like_runtime_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let dense_path = tmp.path().join("model_weights.bin");
    let manifest_path = tmp.path().join("model_weights.json");
    let tensor_name = "model.layers.0.self_attn.q_proj.weight";
    let values = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 1.5, -1.0, -0.25, 0.75];
    let shape = vec![2, 5];
    let group_size = 3;
    let quantized = quantize_q4(&values, &shape, group_size).unwrap();
    let layout = dense_q4_layout(&shape, group_size).unwrap();
    assert_eq!(layout.rows, 2);
    assert_eq!(layout.cols, 5);
    assert_eq!(layout.row_packed_bytes, 3);
    assert_eq!(layout.groups_per_row, 2);
    assert_eq!(quantized.values.len(), layout.packed_bytes);
    assert_eq!(
        quantized.scales.len() * std::mem::size_of::<f32>(),
        layout.scales_bytes
    );

    let mut bytes = quantized.values.clone();
    for scale in &quantized.scales {
        bytes.extend_from_slice(&scale.to_le_bytes());
    }
    for bias in &quantized.biases {
        bytes.extend_from_slice(&bias.to_le_bytes());
    }
    assert_eq!(bytes.len(), layout.total_bytes);
    fs::write(&dense_path, &bytes).unwrap();
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["dense.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: tensor_name.to_string(),
            shard: "dense.safetensors".to_string(),
            dtype: "F32".to_string(),
            shape: shape.clone(),
            source_offsets: [0, (values.len() * std::mem::size_of::<f32>()) as u64],
            runtime_offset: 0,
            byte_len: layout.total_bytes as u64,
            quantization: TensorQuantization::Q4 {
                group_size,
                format: DENSE_Q4_FORMAT.to_string(),
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
            },
            q4_sources: None,
        }],
    };
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let store = DenseStore::open(dense_path, manifest_path).unwrap();
    let entry = store.registry().tensor(tensor_name).unwrap();
    let (packed_row, scales_row, biases_row, timing) =
        store.read_dense_q4_rows(entry, 1, 1, group_size).unwrap();
    assert_eq!(
        packed_row,
        quantized.values[layout.row_packed_bytes..].to_vec()
    );
    assert_eq!(
        scales_row,
        quantized.scales[layout.groups_per_row..].to_vec()
    );
    assert_eq!(
        biases_row,
        quantized.biases[layout.groups_per_row..].to_vec()
    );
    assert_eq!(
        timing.bytes_read,
        (layout.row_packed_bytes + layout.groups_per_row * 2 * std::mem::size_of::<f32>()) as u64
    );
    let decoded_row = store
        .read_tensor_row_f32(tensor_name, 1, 5)
        .unwrap()
        .unwrap();
    let expected_row = q4_dequantize_rows_with_group_size(
        &quantized.values[layout.row_packed_bytes..],
        &quantized.scales[layout.groups_per_row..],
        &quantized.biases[layout.groups_per_row..],
        1,
        5,
        group_size,
    )
    .unwrap();
    assert_eq!(decoded_row, expected_row);

    let (packed_tile, scales_tile, biases_tile, tile_timing) =
        store.read_dense_q4_rows(entry, 0, 2, group_size).unwrap();
    assert_eq!(packed_tile, quantized.values);
    assert_eq!(scales_tile, quantized.scales);
    assert_eq!(biases_tile, quantized.biases);
    assert_eq!(
        tile_timing.bytes_read,
        (layout.packed_bytes + layout.scales_bytes * 2) as u64
    );

    let input = vec![1.0, -1.0, 0.5, 2.0, -0.25];
    let expected = q4_fma_matvec_with_group_size(
        &quantized.values,
        &input,
        &quantized.scales,
        &quantized.biases,
        2,
        5,
        group_size,
    )
    .unwrap();
    let projected = store
        .project_dense_tensor_with_metal(None, tensor_name, &input, 2)
        .unwrap()
        .unwrap();
    assert_eq!(projected, expected);
    let dense_expected = cpu_dense_matvec(&values, &input, 2, 5);
    for (actual, dense) in projected.iter().zip(dense_expected.iter()) {
        assert!(
            (*actual - *dense).abs() <= 0.12,
            "q4 projection drifted too far from dense matvec: actual={actual}, dense={dense}"
        );
    }
    let decoded = store
        .read_full_tensor_f32_cached(tensor_name)
        .unwrap()
        .unwrap();
    assert_eq!(decoded.len(), values.len());
    for (actual, dense) in decoded.iter().zip(values.iter()) {
        assert!(
            (*actual - *dense).abs() <= 0.12,
            "q4 full decode drifted too far from dense tensor: actual={actual}, dense={dense}"
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[ignore = "requires a local Metal device"]
fn arm_macos_dense_q4_mmap_batch_matches_cpu_reference() {
    struct BatchTensor {
        name: String,
        shape: Vec<usize>,
        values: Vec<f32>,
        quantized: QuantizedQ4,
        runtime_offset: u64,
        byte_len: u64,
    }

    fn make_values(rows: usize, cols: usize, seed: f32) -> Vec<f32> {
        (0..rows * cols)
            .map(|idx| {
                let wave = ((idx as f32 + seed) * 0.17).sin() * 0.625;
                let slope = ((idx % cols) as f32 - 7.5) * 0.025;
                wave + slope - seed * 0.03125
            })
            .collect()
    }

    fn append_q4_tensor(
        bytes: &mut Vec<u8>,
        name: &str,
        rows: usize,
        cols: usize,
        group_size: usize,
        values: Vec<f32>,
    ) -> BatchTensor {
        let shape = vec![rows, cols];
        let mut quantized = quantize_q4(&values, &shape, group_size).unwrap();
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&shape, group_size, EXPERT_SCALE_BIAS_DTYPE_BF16)
                .unwrap();
        let runtime_offset = bytes.len() as u64;
        bytes.extend_from_slice(&quantized.values);
        for scale in &mut quantized.scales {
            let bits = f32_to_bf16_bits(*scale);
            bytes.extend_from_slice(&bits.to_le_bytes());
            *scale = f32::from_bits((bits as u32) << 16);
        }
        for bias in &mut quantized.biases {
            let bits = f32_to_bf16_bits(*bias);
            bytes.extend_from_slice(&bits.to_le_bytes());
            *bias = f32::from_bits((bits as u32) << 16);
        }
        let byte_len = bytes.len() as u64 - runtime_offset;
        assert_eq!(byte_len as usize, layout.total_bytes);
        BatchTensor {
            name: name.to_string(),
            shape,
            values,
            quantized,
            runtime_offset,
            byte_len,
        }
    }

    let tmp = tempfile::tempdir().unwrap();
    let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
    fs::create_dir_all(&plan.runtime_dir).unwrap();
    let cols = 2_048;
    let group_size = 64;
    let mut bytes = Vec::new();
    let tensors = vec![
        append_q4_tensor(
            &mut bytes,
            "model.layers.0.self_attn.q_proj.weight",
            3,
            cols,
            group_size,
            make_values(3, cols, 1.0),
        ),
        append_q4_tensor(
            &mut bytes,
            "model.layers.0.self_attn.k_proj.weight",
            5,
            cols,
            group_size,
            make_values(5, cols, 2.0),
        ),
        append_q4_tensor(
            &mut bytes,
            "model.layers.0.self_attn.v_proj.weight",
            2,
            cols,
            group_size,
            make_values(2, cols, 3.0),
        ),
    ];
    fs::write(&plan.non_expert_weights, &bytes).unwrap();
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["dense.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: tensors
            .iter()
            .map(|tensor| DenseTensorRef {
                tensor: tensor.name.clone(),
                shard: "dense.safetensors".to_string(),
                dtype: "F32".to_string(),
                shape: tensor.shape.clone(),
                source_offsets: [
                    tensor.runtime_offset,
                    tensor.runtime_offset
                        + (tensor.values.len() * std::mem::size_of::<f32>()) as u64,
                ],
                runtime_offset: tensor.runtime_offset,
                byte_len: tensor.byte_len,
                quantization: TensorQuantization::Q4 {
                    group_size,
                    format: DENSE_Q4_FORMAT.to_string(),
                    scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                },
                q4_sources: None,
            })
            .collect(),
    };
    fs::write(
        &plan.tensor_manifest,
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let store = DenseStore::open(
        plan.non_expert_weights.clone(),
        plan.tensor_manifest.clone(),
    )
    .unwrap();
    let config = QwenModelConfig {
        model_type: Some("qwen".to_string()),
        architectures: None,
        num_hidden_layers: 1,
        hidden_size: cols,
        num_attention_heads: 1,
        head_dim: None,
        num_key_value_heads: Some(1),
        vocab_size: 32,
        rope_theta: None,
        partial_rotary_factor: None,
        torch_dtype: Some("float32".to_string()),
        num_experts: Some(1),
        num_experts_per_tok: Some(1),
        norm_topk_prob: None,
        moe_intermediate_size: Some(4),
        intermediate_size: None,
        max_position_embeddings: Some(4),
        full_attention_interval: None,
        linear_attention: None,
        mrope_section: None,
        tie_word_embeddings: None,
        num_shared_experts: None,
        shared_expert_intermediate_size: None,
        vision_config: None,
        glm: None,
    };
    let runtime = DenseTransformerRuntime::new(&config);
    let metal = MetalExecutionFacade::new(&plan, &config, &runtime, &store).unwrap();
    let input: Vec<f32> = (0..cols)
        .map(|idx| ((idx as f32) * 0.11).cos() - 0.1875)
        .collect();
    let projections: Vec<_> = tensors
        .iter()
        .map(|tensor| {
            store
                .dense_q4_mmap_projection(&tensor.name, tensor.shape[0], cols)
                .unwrap()
                .unwrap()
        })
        .collect();
    let projections = projections
        .into_iter()
        .map(ResidentMmapMatvecProjection::Q4)
        .collect::<Vec<_>>();
    let (actual, _timing, dispatches) = metal
        .resident_mmap_matvec_batch(&projections, &input)
        .unwrap();
    assert_eq!(dispatches, 1);
    assert_eq!(actual.len(), tensors.len());
    for (projection_idx, (actual, tensor)) in actual.iter().zip(tensors.iter()).enumerate() {
        let expected = q4_fma_matvec_with_group_size(
            &tensor.quantized.values,
            &input,
            &tensor.quantized.scales,
            &tensor.quantized.biases,
            tensor.shape[0],
            cols,
            group_size,
        )
        .unwrap();
        for (row, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= expected.abs().max(1.0) * 2e-6,
                "projection {projection_idx} row {row}: Metal q4 batch mmap {actual} diverged from CPU reference {expected}"
            );
        }
    }

    let second_input = input
        .iter()
        .enumerate()
        .map(|(index, value)| value * -0.375 + index as f32 * 0.0125)
        .collect::<Vec<_>>();
    let third_input = input
        .iter()
        .enumerate()
        .map(|(index, value)| value * 0.8125 - (index % 29) as f32 * 0.03125)
        .collect::<Vec<_>>();
    let scalar_rows = [
        input.as_slice(),
        second_input.as_slice(),
        third_input.as_slice(),
    ]
    .into_iter()
    .map(|row| {
        metal
            .resident_mmap_matvec_batch(&projections, row)
            .unwrap()
            .0
    })
    .collect::<Vec<_>>();
    let matrix_input = [
        input.as_slice(),
        second_input.as_slice(),
        third_input.as_slice(),
    ]
    .concat();
    let (matrix_actual, _, matrix_dispatches) = metal
        .resident_mmap_projection_matrix(&projections, 3, cols, &matrix_input)
        .unwrap();
    assert_eq!(matrix_dispatches, 1);
    for (projection_idx, (actual, tensor)) in matrix_actual.iter().zip(tensors.iter()).enumerate() {
        assert_eq!(actual.len(), 3 * tensor.shape[0]);
        for (input_row, row_input) in [
            input.as_slice(),
            second_input.as_slice(),
            third_input.as_slice(),
        ]
        .into_iter()
        .enumerate()
        {
            let row_start = input_row * tensor.shape[0];
            assert_eq!(
                actual[row_start..row_start + tensor.shape[0]]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                scalar_rows[input_row][projection_idx]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "projection {projection_idx} input {input_row} matrix must exactly match scalar batch"
            );
            let expected = q4_fma_matvec_with_group_size(
                &tensor.quantized.values,
                row_input,
                &tensor.quantized.scales,
                &tensor.quantized.biases,
                tensor.shape[0],
                cols,
                group_size,
            )
            .unwrap();
            for (row, (actual, expected)) in actual[row_start..row_start + tensor.shape[0]]
                .iter()
                .zip(expected.iter())
                .enumerate()
            {
                assert!(
                    (*actual - *expected).abs() <= expected.abs().max(1.0) * 2e-6,
                    "projection {projection_idx} input {input_row} row {row}: Metal q4 matrix {actual} diverged from CPU reference {expected}"
                );
            }
        }
    }

    let norm_weight = (0..cols)
        .map(|index| 0.75 + index as f32 * 0.03125)
        .collect::<Vec<_>>();
    let matrix_norm = metal
        .qwen_rms_norm_rows(&matrix_input, &norm_weight, 3, cols)
        .unwrap();
    for (row, row_input) in [
        input.as_slice(),
        second_input.as_slice(),
        third_input.as_slice(),
    ]
    .into_iter()
    .enumerate()
    {
        let scalar_norm = metal
            .qwen_rms_norm_rows(row_input, &norm_weight, 1, cols)
            .unwrap();
        assert_eq!(
            matrix_norm[row * cols..(row + 1) * cols]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            scalar_norm
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "Qwen RMS-normalization matrix row {row} must exactly match the scalar command"
        );
    }

    let query_rows = 2;
    let prefix_rows = 1;
    let query_heads = 2;
    let kv_heads = 1;
    let head_dim = 256;
    let queries = (0..query_rows * query_heads * head_dim)
        .map(|index| ((index as f32 + 0.5) * 0.013).sin() * 0.25)
        .collect::<Vec<_>>();
    let keys = (0..(prefix_rows + query_rows) * kv_heads * head_dim)
        .map(|index| ((index as f32 + 1.5) * 0.017).cos() * 0.375)
        .collect::<Vec<_>>();
    let values = (0..(prefix_rows + query_rows) * kv_heads * head_dim)
        .map(|index| ((index as f32 + 2.5) * 0.019).sin() * 0.5)
        .collect::<Vec<_>>();
    let attention = metal
        .qwen_causal_attention_rows(
            &queries,
            &keys,
            &values,
            query_rows,
            prefix_rows,
            query_heads,
            kv_heads,
            head_dim,
        )
        .unwrap();
    for query_row in 0..query_rows {
        let q_start = query_row * query_heads * head_dim;
        let records = (0..prefix_rows + query_row + 1)
            .map(|key_row| {
                let start = key_row * kv_heads * head_dim;
                (
                    &keys[start..start + kv_heads * head_dim],
                    &values[start..start + kv_heads * head_dim],
                )
            })
            .collect::<Vec<_>>();
        let expected = causal_attention(
            &queries[q_start..q_start + query_heads * head_dim],
            &records,
            query_heads,
            kv_heads,
            head_dim,
        );
        for (dimension, (actual, expected)) in attention[q_start..q_start + query_heads * head_dim]
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 2e-5,
                "Qwen causal attention row {query_row} dimension {dimension}: Metal {actual} diverged from CPU {expected}"
            );
        }
    }

    let query_gates = (0..queries.len())
        .map(|index| ((index as f32 + 0.25) * 0.071).sin() * 12.0)
        .collect::<Vec<_>>();
    let gated_attention = metal
        .qwen_causal_attention_rows_owned(
            &queries,
            &keys,
            &values,
            Some(&query_gates),
            query_rows,
            prefix_rows,
            query_heads,
            kv_heads,
            head_dim,
        )
        .unwrap()
        .materialize();
    let unit_values = vec![1.0; values.len()];
    let metal_sigmoid = metal
        .qwen_causal_attention_rows_owned(
            &queries,
            &keys,
            &unit_values,
            Some(&query_gates),
            query_rows,
            prefix_rows,
            query_heads,
            kv_heads,
            head_dim,
        )
        .unwrap()
        .materialize();
    for (index, (actual, gate)) in metal_sigmoid.iter().zip(query_gates.iter()).enumerate() {
        let expected = qwen_attention_sigmoid(*gate);
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "Qwen Metal sigmoid {index} must match Rust exactly: gate={gate:?} Metal={actual:?} CPU={expected:?}"
        );
    }
    for (index, ((actual, ungated), gate)) in gated_attention
        .iter()
        .zip(attention.iter())
        .zip(query_gates.iter())
        .enumerate()
    {
        let expected = *ungated * qwen_attention_sigmoid(*gate);
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "Qwen causal attention gate {index} must exactly match the scalar CPU boundary: gate={gate:?} ungated={ungated:?} Metal={actual:?} CPU={expected:?}"
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn arm_macos_resident_dense_mmap_batch_matches_cpu_reference() {
    fn f16_bits(value: f32) -> u16 {
        match value.to_bits() {
            bits if bits == 0.0f32.to_bits() => 0x0000,
            bits if bits == 0.25f32.to_bits() => 0x3400,
            bits if bits == 0.5f32.to_bits() => 0x3800,
            bits if bits == 1.0f32.to_bits() => 0x3c00,
            bits if bits == 2.0f32.to_bits() => 0x4000,
            bits if bits == (-0.5f32).to_bits() => 0xb800,
            bits if bits == (-1.0f32).to_bits() => 0xbc00,
            bits if bits == (-2.0f32).to_bits() => 0xc000,
            _ => panic!("test value {value} is not in the exact F16 fixture"),
        }
    }

    fn append_dense_tensor(
        bytes: &mut Vec<u8>,
        name: &str,
        dtype: &str,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> DenseTensorRef {
        while !bytes.len().is_multiple_of(TENSOR_ALIGNMENT as usize) {
            bytes.push(0);
        }
        let runtime_offset = bytes.len() as u64;
        match dtype {
            "BF16" => {
                for value in values {
                    bytes.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
                }
            }
            "F16" => {
                for value in values {
                    bytes.extend_from_slice(&f16_bits(*value).to_le_bytes());
                }
            }
            "F32" => {
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
        let byte_len = bytes.len() as u64 - runtime_offset;
        DenseTensorRef {
            tensor: name.to_string(),
            shard: "dense.safetensors".to_string(),
            dtype: dtype.to_string(),
            shape: vec![rows, cols],
            source_offsets: [0, byte_len],
            runtime_offset,
            byte_len,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }
    }

    let temp = tempfile::tempdir().unwrap();
    let plan = plan_unchecked(QWEN35_MODEL, temp.path());
    fs::create_dir_all(&plan.runtime_dir).unwrap();
    let rows = 3;
    let cols = 4;
    let values = [
        1.0, -0.5, 0.25, 2.0, -1.0, 0.5, 2.0, -2.0, 0.0, 1.0, -1.0, 0.5,
    ];
    let mut bytes = Vec::new();
    let tensors = [
        append_dense_tensor(&mut bytes, "dense_bf16", "BF16", &values, rows, cols),
        append_dense_tensor(&mut bytes, "dense_f16", "F16", &values, rows, cols),
        append_dense_tensor(&mut bytes, "dense_f32", "F32", &values, rows, cols),
    ];
    let tensor_names = tensors
        .iter()
        .map(|tensor| tensor.tensor.clone())
        .collect::<Vec<_>>();
    fs::write(&plan.non_expert_weights, &bytes).unwrap();
    fs::write(
        &plan.tensor_manifest,
        serde_json::to_vec(&FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: tensors.into_iter().collect(),
        })
        .unwrap(),
    )
    .unwrap();
    let store = DenseStore::open(
        plan.non_expert_weights.clone(),
        plan.tensor_manifest.clone(),
    )
    .unwrap();
    let config = QwenModelConfig {
        model_type: Some("qwen".to_string()),
        architectures: None,
        num_hidden_layers: 1,
        hidden_size: cols,
        num_attention_heads: 1,
        head_dim: None,
        num_key_value_heads: Some(1),
        vocab_size: 32,
        rope_theta: None,
        partial_rotary_factor: None,
        torch_dtype: Some("bfloat16".to_string()),
        num_experts: Some(1),
        num_experts_per_tok: Some(1),
        norm_topk_prob: None,
        moe_intermediate_size: Some(4),
        intermediate_size: None,
        max_position_embeddings: Some(4),
        full_attention_interval: None,
        linear_attention: None,
        mrope_section: None,
        tie_word_embeddings: None,
        num_shared_experts: None,
        shared_expert_intermediate_size: None,
        vision_config: None,
        glm: None,
    };
    let runtime = DenseTransformerRuntime::new(&config);
    let metal = MetalExecutionFacade::new(&plan, &config, &runtime, &store).unwrap();
    let input = [0.5, -1.0, 2.0, 0.25];
    let projections = tensor_names
        .iter()
        .map(|tensor_name| {
            store
                .resident_mmap_projection(tensor_name, rows, cols)
                .unwrap()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let (actual, _, dispatches) = metal
        .resident_mmap_matvec_batch(&projections, &input)
        .unwrap();

    let expected = values
        .chunks_exact(cols)
        .map(|weights| {
            weights
                .iter()
                .zip(input.iter())
                .map(|(weight, input)| weight * input)
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    assert_eq!(dispatches, 3);
    for (index, dtype) in ["BF16", "F16", "F32"].iter().enumerate() {
        let output = &actual[index];
        for (row, (actual, expected)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "{dtype} row {row}: Metal {actual} != CPU {expected}"
            );
        }
        let actual_candidates = metal
            .resident_top_candidates(&projections[index], &input, 2, 2)
            .unwrap();
        let expected_candidates = top_k(&expected[..2], 2);
        assert_eq!(actual_candidates.len(), expected_candidates.len());
        for ((actual_id, actual_score), (expected_id, expected_score)) in
            actual_candidates.iter().zip(&expected_candidates)
        {
            assert_eq!(actual_id, expected_id, "{dtype} topK id diverged");
            assert!(
                (actual_score - expected_score).abs() <= 1e-5,
                "{dtype} topK score {actual_score} != {expected_score}"
            );
        }
        let masked_candidates = metal
            .resident_top_candidates_masked(&projections[index], &input, 2, 1, &[0b10])
            .unwrap();
        assert_eq!(masked_candidates.len(), 1);
        assert_eq!(masked_candidates[0].0, 1, "{dtype} ignored vocabulary mask");
        assert!(
            (masked_candidates[0].1 - expected[1]).abs() <= 1e-5,
            "{dtype} masked topK score {} != {}",
            masked_candidates[0].1,
            expected[1]
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn arm_macos_post_attention_dense_prep_matches_cpu_reference() {
    fn f16_bits(value: f32) -> u16 {
        match value.to_bits() {
            bits if bits == 0.0f32.to_bits() => 0x0000,
            bits if bits == 0.25f32.to_bits() => 0x3400,
            bits if bits == 0.5f32.to_bits() => 0x3800,
            bits if bits == 1.0f32.to_bits() => 0x3c00,
            bits if bits == 2.0f32.to_bits() => 0x4000,
            bits if bits == (-0.5f32).to_bits() => 0xb800,
            bits if bits == (-1.0f32).to_bits() => 0xbc00,
            bits if bits == (-2.0f32).to_bits() => 0xc000,
            _ => panic!("test value {value} is not in the exact F16 fixture"),
        }
    }

    fn append_dense_tensor(
        bytes: &mut Vec<u8>,
        name: &str,
        dtype: &str,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> DenseTensorRef {
        while !bytes.len().is_multiple_of(TENSOR_ALIGNMENT as usize) {
            bytes.push(0);
        }
        let runtime_offset = bytes.len() as u64;
        match dtype {
            "BF16" => {
                for value in values {
                    bytes.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
                }
            }
            "F16" => {
                for value in values {
                    bytes.extend_from_slice(&f16_bits(*value).to_le_bytes());
                }
            }
            "F32" => {
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
        DenseTensorRef {
            tensor: name.to_string(),
            shard: "dense.safetensors".to_string(),
            dtype: dtype.to_string(),
            shape: vec![rows, cols],
            source_offsets: [0, (values.len() * std::mem::size_of::<f32>()) as u64],
            runtime_offset,
            byte_len: bytes.len() as u64 - runtime_offset,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }
    }

    fn matvec(weights: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        assert_eq!(weights.len(), rows * cols);
        weights
            .chunks_exact(cols)
            .map(|row| {
                row.iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect()
    }

    let layer = 0;
    let width = 4;
    let attention_width = 4;
    let experts = 3;
    let out_proj_name = attention_tensor_name(layer, "o_proj");
    let router_name = router_tensor_name(layer);
    let out_values = [
        1.0, -0.5, 0.25, 2.0, -1.0, 0.5, 2.0, -2.0, 0.0, 1.0, -1.0, 0.5, 0.25, -2.0, 1.0, 0.5,
    ];
    let router_values = [
        1.0, -0.5, 0.25, 2.0, -1.0, 0.5, 2.0, -2.0, 0.25, -2.0, 1.0, 0.5,
    ];
    let attention_output = [0.5, -1.0, 2.0, 0.25];
    let residual = [0.25, -0.5, 1.0, 2.0];
    let post_norm_weight = [1.0, 0.5, 2.0, 0.25];

    let expected_projected = matvec(&out_values, &attention_output, width, attention_width);
    let mut expected_residual = residual.to_vec();
    add_in_place(&mut expected_residual, &expected_projected);
    let mut expected_normed = expected_residual.clone();
    rms_norm_with_weight_in_place(&mut expected_normed, Some(&post_norm_weight));
    let expected_active = top_k(&matvec(&router_values, &expected_normed, experts, width), 2);

    for dtype in ["BF16", "F16", "F32"] {
        let temp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, temp.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        let mut bytes = Vec::new();
        let out_tensor = append_dense_tensor(
            &mut bytes,
            &out_proj_name,
            dtype,
            &out_values,
            width,
            attention_width,
        );
        let router_tensor = append_dense_tensor(
            &mut bytes,
            &router_name,
            dtype,
            &router_values,
            experts,
            width,
        );
        fs::write(&plan.non_expert_weights, &bytes).unwrap();
        fs::write(
            &plan.tensor_manifest,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![out_tensor, router_tensor],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(
            plan.non_expert_weights.clone(),
            plan.tensor_manifest.clone(),
        )
        .unwrap();
        let config = QwenModelConfig {
            model_type: Some("qwen3_moe".to_string()),
            architectures: None,
            num_hidden_layers: 1,
            hidden_size: width,
            num_attention_heads: 1,
            head_dim: None,
            num_key_value_heads: Some(1),
            vocab_size: 32,
            rope_theta: None,
            partial_rotary_factor: None,
            torch_dtype: Some(dtype.to_ascii_lowercase()),
            num_experts: Some(experts),
            num_experts_per_tok: Some(2),
            norm_topk_prob: None,
            moe_intermediate_size: Some(4),
            intermediate_size: None,
            max_position_embeddings: Some(4),
            full_attention_interval: None,
            linear_attention: None,
            mrope_section: None,
            tie_word_embeddings: None,
            num_shared_experts: None,
            shared_expert_intermediate_size: None,
            vision_config: None,
            glm: None,
        };
        let runtime = DenseTransformerRuntime::new(&config);
        let metal = MetalExecutionFacade::new(&plan, &config, &runtime, &store).unwrap();
        let prep = store
            .post_attention_prep_with_metal(
                &metal,
                layer,
                experts,
                &out_proj_name,
                &attention_output,
                MetalBatchProjectionInput::Cpu(&residual),
                &post_norm_weight,
                2,
                None,
            )
            .unwrap();

        assert_eq!(prep.active.len(), expected_active.len());
        for ((actual_id, actual_score), (expected_id, expected_score)) in
            prep.active.iter().zip(&expected_active)
        {
            assert_eq!(actual_id, expected_id, "{dtype} route id diverged");
            assert!(
                (actual_score - expected_score).abs() <= 1e-4,
                "{dtype} route score {actual_score} != {expected_score}"
            );
        }
        let actual_residual = metal
            .inner
            .read_and_recycle_f32(prep.residual_buffer, width);
        let actual_normed = metal.inner.read_and_recycle_f32(prep.normed_buffer, width);
        for index in 0..width {
            assert!(
                (actual_residual[index] - expected_residual[index]).abs() <= 1e-4,
                "{dtype} residual[{index}] {} != {}",
                actual_residual[index],
                expected_residual[index]
            );
            assert!(
                (actual_normed[index] - expected_normed[index]).abs() <= 1e-4,
                "{dtype} normed[{index}] {} != {}",
                actual_normed[index],
                expected_normed[index]
            );
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn arm_macos_post_attention_resident_q4_prep_matches_cpu_reference() {
    fn q4_bytes(
        values: &[f32],
        shape: &[usize],
        group_size: usize,
    ) -> (Vec<u8>, DenseQ4Layout, QuantizedQ4) {
        let quantized = quantize_q4(values, shape, group_size).unwrap();
        let layout = dense_q4_layout(shape, group_size).unwrap();
        let mut bytes = quantized.values.clone();
        for scale in &quantized.scales {
            bytes.extend_from_slice(&scale.to_le_bytes());
        }
        for bias in &quantized.biases {
            bytes.extend_from_slice(&bias.to_le_bytes());
        }
        assert_eq!(bytes.len(), layout.total_bytes);
        (bytes, layout, quantized)
    }

    let tmp = tempfile::tempdir().unwrap();
    let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
    fs::create_dir_all(&plan.runtime_dir).unwrap();
    let layer = 0;
    let width = 8;
    let attention_width = 16;
    let experts = 6;
    let group_size = 4;
    let out_proj_name = linear_attention_tensor_name(layer, "out_proj");
    let router_name = router_tensor_name(layer);
    let out_shape = vec![width, attention_width];
    let router_shape = vec![experts, width];
    let out_values: Vec<f32> = (0..width * attention_width)
        .map(|idx| ((idx as f32) * 0.17).sin() * 0.625 - ((idx % 5) as f32) * 0.03125)
        .collect();
    let router_values: Vec<f32> = (0..experts * width)
        .map(|idx| ((idx as f32) * 0.23).cos() * 0.375 + ((idx % 3) as f32) * 0.0625)
        .collect();
    let (out_bytes, out_layout, out_quantized) = q4_bytes(&out_values, &out_shape, group_size);
    let (router_bytes, router_layout, router_quantized) =
        q4_bytes(&router_values, &router_shape, group_size);
    let router_offset = out_bytes.len();
    let mut dense_bytes = out_bytes;
    dense_bytes.extend_from_slice(&router_bytes);
    fs::write(&plan.non_expert_weights, &dense_bytes).unwrap();

    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["dense.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![
            DenseTensorRef {
                tensor: out_proj_name.clone(),
                shard: "dense.safetensors".to_string(),
                dtype: "F32".to_string(),
                shape: out_shape.clone(),
                source_offsets: [0, (out_values.len() * std::mem::size_of::<f32>()) as u64],
                runtime_offset: 0,
                byte_len: out_layout.total_bytes as u64,
                quantization: TensorQuantization::Q4 {
                    group_size,
                    format: DENSE_Q4_FORMAT.to_string(),
                    scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                },
                q4_sources: None,
            },
            DenseTensorRef {
                tensor: router_name.clone(),
                shard: "dense.safetensors".to_string(),
                dtype: "F32".to_string(),
                shape: router_shape.clone(),
                source_offsets: [
                    (out_values.len() * std::mem::size_of::<f32>()) as u64,
                    ((out_values.len() + router_values.len()) * std::mem::size_of::<f32>()) as u64,
                ],
                runtime_offset: router_offset as u64,
                byte_len: router_layout.total_bytes as u64,
                quantization: TensorQuantization::Q4 {
                    group_size,
                    format: DENSE_Q4_FORMAT.to_string(),
                    scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                },
                q4_sources: None,
            },
        ],
    };
    fs::write(
        &plan.tensor_manifest,
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let store = DenseStore::open(
        plan.non_expert_weights.clone(),
        plan.tensor_manifest.clone(),
    )
    .unwrap();
    let config = QwenModelConfig {
        model_type: Some("qwen".to_string()),
        architectures: None,
        num_hidden_layers: 1,
        hidden_size: width,
        num_attention_heads: 1,
        head_dim: None,
        num_key_value_heads: Some(1),
        vocab_size: 32,
        rope_theta: None,
        partial_rotary_factor: None,
        torch_dtype: Some("float32".to_string()),
        num_experts: Some(experts),
        num_experts_per_tok: Some(3),
        norm_topk_prob: None,
        moe_intermediate_size: Some(4),
        intermediate_size: None,
        max_position_embeddings: Some(4),
        full_attention_interval: None,
        linear_attention: None,
        mrope_section: None,
        tie_word_embeddings: None,
        num_shared_experts: None,
        shared_expert_intermediate_size: None,
        vision_config: None,
        glm: None,
    };
    let runtime = DenseTransformerRuntime::new(&config);
    let metal = MetalExecutionFacade::new(&plan, &config, &runtime, &store).unwrap();

    let attention_output: Vec<f32> = (0..attention_width)
        .map(|idx| ((idx as f32) * 0.11).sin() - 0.375)
        .collect();
    let residual: Vec<f32> = (0..width)
        .map(|idx| ((idx as f32) * 0.29).cos() * 0.5)
        .collect();
    let post_norm_weight: Vec<f32> = (0..width)
        .map(|idx| 0.75 + (idx as f32) * 0.03125)
        .collect();

    let expected_projected = q4_fma_matvec_with_group_size(
        &out_quantized.values,
        &attention_output,
        &out_quantized.scales,
        &out_quantized.biases,
        width,
        attention_width,
        group_size,
    )
    .unwrap();
    let mut expected_residual = residual.clone();
    add_in_place(&mut expected_residual, &expected_projected);
    let mut expected_normed = expected_residual.clone();
    rms_norm_with_weight_in_place(&mut expected_normed, Some(&post_norm_weight));
    let expected_router = q4_fma_matvec_with_group_size(
        &router_quantized.values,
        &expected_normed,
        &router_quantized.scales,
        &router_quantized.biases,
        experts,
        width,
        group_size,
    )
    .unwrap();
    let expected_active = top_k(&expected_router, 3);

    let prep = store
        .post_attention_prep_with_metal(
            &metal,
            layer,
            experts,
            &out_proj_name,
            &attention_output,
            MetalBatchProjectionInput::Cpu(&residual),
            &post_norm_weight,
            3,
            None,
        )
        .unwrap();
    assert_eq!(prep.width, width);
    assert_eq!(prep.active.len(), expected_active.len());
    for (slot, ((actual_id, actual_score), (expected_id, expected_score))) in
        prep.active.iter().zip(expected_active.iter()).enumerate()
    {
        assert_eq!(
            actual_id, expected_id,
            "active expert id at slot {slot} diverged"
        );
        assert!(
            (*actual_score - *expected_score).abs() <= 1e-4,
            "active expert score at slot {slot} diverged: actual={actual_score}, expected={expected_score}"
        );
    }
    assert!(prep.routing_command().is_none());

    let actual_residual = metal
        .inner
        .read_and_recycle_f32(prep.residual_buffer, width);
    let actual_normed = metal.inner.read_and_recycle_f32(prep.normed_buffer, width);
    for idx in 0..width {
        assert!(
            (actual_residual[idx] - expected_residual[idx]).abs() <= 1e-4,
            "residual[{idx}] diverged: actual={} expected={}",
            actual_residual[idx],
            expected_residual[idx]
        );
        assert!(
            (actual_normed[idx] - expected_normed[idx]).abs() <= 1e-4,
            "normed[{idx}] diverged: actual={} expected={}",
            actual_normed[idx],
            expected_normed[idx]
        );
    }
}

#[test]
fn dense_manifest_preserves_non_native_dense_weights() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path();
    let q_proj = f32_tensor_bytes(&[0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0]);
    let lm_head = f32_tensor_bytes(&[0.25; 8]);
    let embed = f32_tensor_bytes(&[0.5; 8]);
    let mtp_q_proj = f32_tensor_bytes(&[0.75; 8]);
    let mtp_expert = f32_tensor_bytes(&[1.25; 8]);
    let tensors = vec![
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "F32".to_string(),
            vec![2, 4],
            q_proj,
        ),
        (
            "lm_head.weight".to_string(),
            "F32".to_string(),
            vec![2, 4],
            lm_head,
        ),
        (
            "model.embed_tokens.weight".to_string(),
            "F32".to_string(),
            vec![2, 4],
            embed,
        ),
        (
            "mtp.layers.0.self_attn.q_proj.weight".to_string(),
            "F32".to_string(),
            vec![2, 4],
            mtp_q_proj,
        ),
        (
            "mtp.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
            "F32".to_string(),
            vec![2, 4],
            mtp_expert,
        ),
    ];
    let fixture_refs = typed_fixture_refs(&tensors);
    fs::write(
        snapshot.join("dense.safetensors"),
        make_typed_safetensors(&fixture_refs),
    )
    .unwrap();
    let mut weight_map = serde_json::Map::new();
    for (name, _, _, _) in &tensors {
        weight_map.insert(
            name.clone(),
            serde_json::Value::String("dense.safetensors".to_string()),
        );
    }
    let index = serde_json::Value::Object(serde_json::Map::from_iter([(
        "weight_map".to_string(),
        serde_json::Value::Object(weight_map),
    )]));
    let index_path = snapshot.join("model.safetensors.index.json");
    fs::write(&index_path, index.to_string()).unwrap();

    let (manifest, visual_refs) =
        build_manifest(QWEN35_MODEL, snapshot, &index_path, None).unwrap();
    assert!(visual_refs.is_empty());
    assert!(manifest.expert_tensors.is_empty());
    let registry = TensorRegistry::from_manifest(&manifest);
    let q_proj_entry = registry
        .tensor("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(q_proj_entry.quantization, TensorQuantization::None);
    assert_eq!(q_proj_entry.byte_len, 2 * 4 * 4);
    assert_eq!(
        registry.tensor("lm_head.weight").unwrap().quantization,
        TensorQuantization::None
    );
    assert_eq!(
        registry
            .tensor("model.embed_tokens.weight")
            .unwrap()
            .quantization,
        TensorQuantization::None
    );
    assert!(
        registry
            .tensor("mtp.layers.0.self_attn.q_proj.weight")
            .is_none()
    );
    assert!(
        registry
            .tensor("mtp.layers.0.mlp.experts.0.gate_proj.weight")
            .is_none()
    );
}

#[test]
fn dense_manifest_imports_native_mlx_q4_triples() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path();
    let source_tensor_name = "language_model.model.layers.0.self_attn.q_proj.weight";
    let scales_name = "language_model.model.layers.0.self_attn.q_proj.scales";
    let biases_name = "language_model.model.layers.0.self_attn.q_proj.biases";
    let runtime_tensor_name = "model.layers.0.self_attn.q_proj.weight";
    let packed_word = 0x7654_3210u32.to_le_bytes().to_vec();
    let scales = bf16_tensor_bytes(&[0.5]);
    let biases = bf16_tensor_bytes(&[1.0]);
    let tensors = vec![
        (
            source_tensor_name.to_string(),
            "U32".to_string(),
            vec![1, 1],
            packed_word.clone(),
        ),
        (
            scales_name.to_string(),
            EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            vec![1, 1],
            scales.clone(),
        ),
        (
            biases_name.to_string(),
            EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            vec![1, 1],
            biases.clone(),
        ),
    ];
    let fixture_refs = typed_fixture_refs(&tensors);
    fs::write(
        snapshot.join("dense.safetensors"),
        make_typed_safetensors(&fixture_refs),
    )
    .unwrap();
    let mut weight_map = serde_json::Map::new();
    for (name, _, _, _) in &tensors {
        weight_map.insert(
            name.clone(),
            serde_json::Value::String("dense.safetensors".to_string()),
        );
    }
    let index = serde_json::Value::Object(serde_json::Map::from_iter([(
        "weight_map".to_string(),
        serde_json::Value::Object(weight_map),
    )]));
    let index_path = snapshot.join("model.safetensors.index.json");
    fs::write(&index_path, index.to_string()).unwrap();

    let (manifest, visual_refs) =
        build_manifest(QWEN35_MODEL, snapshot, &index_path, None).unwrap();
    assert!(visual_refs.is_empty());
    assert!(manifest.expert_tensors.is_empty());
    assert_eq!(manifest.dense_tensors.len(), 1);
    let dense_ref = &manifest.dense_tensors[0];
    assert_eq!(dense_ref.tensor, runtime_tensor_name);
    assert_eq!(dense_ref.dtype, "U32");
    assert_eq!(dense_ref.shape, vec![1, 8]);
    assert_eq!(
        dense_ref.quantization,
        TensorQuantization::Q4 {
            group_size: GROUP_SIZE,
            format: DENSE_Q4_MLX_FORMAT.to_string(),
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        }
    );
    assert!(dense_ref.q4_sources.is_some());
    let layout = dense_q4_layout_with_scale_bias_dtype(
        &dense_ref.shape,
        GROUP_SIZE,
        EXPERT_SCALE_BIAS_DTYPE_BF16,
    )
    .unwrap();
    assert_eq!(dense_ref.byte_len, layout.total_bytes as u64);
    assert_eq!(layout.packed_bytes, packed_word.len());
    assert_eq!(layout.scales_bytes, scales.len());

    let dense_path = snapshot.join("model_weights.bin");
    write_dense_tensor_store(snapshot, &dense_path, &manifest.dense_tensors, None).unwrap();
    let mut expected_bytes = packed_word.clone();
    expected_bytes.extend_from_slice(&scales);
    expected_bytes.extend_from_slice(&biases);
    assert_eq!(fs::read(&dense_path).unwrap(), expected_bytes);

    let manifest_path = snapshot.join("model_weights.json");
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let store = DenseStore::open(dense_path, manifest_path).unwrap();
    let entry = store.registry().tensor(runtime_tensor_name).unwrap();
    let (packed, decoded_scales, decoded_biases, timing) =
        store.read_dense_q4_rows(entry, 0, 1, GROUP_SIZE).unwrap();
    assert_eq!(packed, packed_word);
    assert_eq!(decoded_scales, vec![0.5]);
    assert_eq!(decoded_biases, vec![1.0]);
    assert_eq!(
        timing.bytes_read,
        (layout.packed_bytes + layout.scales_bytes * 2) as u64
    );

    let input = vec![1.0; 8];
    let projected = store
        .project_dense_tensor_with_metal(None, runtime_tensor_name, &input, 1)
        .unwrap()
        .unwrap();
    assert_eq!(projected, vec![22.0]);
}

#[test]
fn manifest_classifies_mlx_switch_mlp_tensors_as_aggregate_experts() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path();
    let tensors = vec![
        (
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight".to_string(),
            "U32".to_string(),
            vec![2, 2, 1],
            0x7654_3210u32.to_le_bytes().to_vec(),
        ),
        (
            "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight".to_string(),
            "U32".to_string(),
            vec![2, 2, 1],
            0x7654_3210u32.to_le_bytes().to_vec(),
        ),
        (
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight".to_string(),
            "U32".to_string(),
            vec![2, 1, 2],
            0x7654_3210u32.to_le_bytes().to_vec(),
        ),
    ];
    let fixture_refs = typed_fixture_refs(&tensors);
    fs::write(
        snapshot.join("experts.safetensors"),
        make_typed_safetensors(&fixture_refs),
    )
    .unwrap();
    let mut weight_map = serde_json::Map::new();
    for (name, _, _, _) in &tensors {
        weight_map.insert(
            name.clone(),
            serde_json::Value::String("experts.safetensors".to_string()),
        );
    }
    let index = serde_json::Value::Object(serde_json::Map::from_iter([(
        "weight_map".to_string(),
        serde_json::Value::Object(weight_map),
    )]));
    let index_path = snapshot.join("model.safetensors.index.json");
    fs::write(&index_path, index.to_string()).unwrap();

    let (manifest, visual_refs) =
        build_manifest(QWEN35_MODEL, snapshot, &index_path, None).unwrap();

    assert!(visual_refs.is_empty());
    assert!(manifest.dense_tensors.is_empty());
    assert_eq!(manifest.expert_tensors.len(), 3);
    let names = manifest
        .expert_tensors
        .iter()
        .map(|tensor| tensor.tensor.as_str())
        .collect::<Vec<_>>();
    assert!(names.contains(&"model.layers.0.mlp.switch_mlp.gate_proj.weight"));
    assert!(names.contains(&"model.layers.0.mlp.switch_mlp.up_proj.weight"));
    assert!(names.contains(&"model.layers.0.mlp.switch_mlp.down_proj.weight"));
    assert!(
        manifest
            .expert_tensors
            .iter()
            .all(|tensor| tensor.layer == Some(0) && tensor.expert.is_none())
    );
}

#[test]
fn dense_store_reuses_decoded_tiles_for_repeated_lm_head_sampling() {
    let tmp = tempfile::tempdir().unwrap();
    let dense_path = tmp.path().join("model_weights.bin");
    let manifest_path = tmp.path().join("model_weights.json");
    let mut bytes = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&dense_path, &bytes).unwrap();
    let manifest = FlashMoeManifest {
        model: QWEN35_MODEL.to_string(),
        cache_version: CACHE_VERSION.to_string(),
        dense_shards: vec!["dense.safetensors".to_string()],
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: "lm_head.weight".to_string(),
            shard: "dense.safetensors".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 2],
            source_offsets: [0, bytes.len() as u64],
            runtime_offset: 0,
            byte_len: bytes.len() as u64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }],
    };
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let store = DenseStore::open(dense_path, manifest_path).unwrap();

    let first = store
        .read_tensor_rows_f32_cached("lm_head.weight", 0, 2)
        .unwrap();
    assert_eq!(first.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(store.decoded_tensor_tile_count(), 1);

    let second = store
        .read_tensor_rows_f32_cached("lm_head.weight", 0, 2)
        .unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(
        store.decoded_tensor_tile_count(),
        1,
        "cached LM-head tile should not be decoded again for the next token"
    );

    let other = store
        .read_tensor_rows_f32_cached("lm_head.weight", 2, 2)
        .unwrap();
    assert_eq!(other.as_slice(), &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(store.decoded_tensor_tile_count(), 2);
}

#[test]
fn dense_store_reports_decoded_tile_cache_hit_and_miss_timing() {
    let tmp = tempfile::tempdir().unwrap();
    let dense_path = tmp.path().join("model_weights.bin");
    let manifest_path = tmp.path().join("model_weights.json");
    let mut bytes = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0] {
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
    let (_, miss) = store
        .read_tensor_rows_f32_cached_profiled("lm_head.weight", 0, 2)
        .unwrap();
    assert_eq!(miss.cache_misses, 1);
    assert_eq!(miss.cache_hits, 0);
    assert_eq!(miss.cache_inserts, 1);
    assert_eq!(miss.bytes_read, bytes.len() as u64);
    assert_eq!(miss.decoded_bytes, bytes.len() as u64);

    let (_, hit) = store
        .read_tensor_rows_f32_cached_profiled("lm_head.weight", 0, 2)
        .unwrap();
    assert_eq!(hit.cache_hits, 1);
    assert_eq!(hit.cache_misses, 0);
    assert_eq!(hit.bytes_read, 0);
    assert_eq!(hit.decoded_bytes, 0);
}

#[test]
fn dense_q4_mmap_projection_descriptors_are_cached() {
    let tmp = tempfile::tempdir().unwrap();
    let dense_path = tmp.path().join("model_weights.bin");
    let manifest_path = tmp.path().join("model_weights.json");
    let tensor_name = "model.layers.0.self_attn.q_proj.weight";
    let shape = vec![2, 4];
    let group_size = 2;
    let values = [0.25, -0.5, 1.0, 0.75, -0.125, 0.375, 0.625, -0.875];
    let quantized = quantize_q4(&values, &shape, group_size).unwrap();
    let layout = dense_q4_layout(&shape, group_size).unwrap();
    let mut bytes = quantized.values.clone();
    for scale in &quantized.scales {
        bytes.extend_from_slice(&scale.to_le_bytes());
    }
    for bias in &quantized.biases {
        bytes.extend_from_slice(&bias.to_le_bytes());
    }
    assert_eq!(bytes.len(), layout.total_bytes);
    fs::write(&dense_path, &bytes).unwrap();
    fs::write(
        &manifest_path,
        serde_json::to_vec(&FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: tensor_name.to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: "F32".to_string(),
                shape,
                source_offsets: [0, bytes.len() as u64],
                runtime_offset: 0,
                byte_len: bytes.len() as u64,
                quantization: TensorQuantization::Q4 {
                    group_size,
                    format: DENSE_Q4_FORMAT.to_string(),
                    scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                },
                q4_sources: None,
            }],
        })
        .unwrap(),
    )
    .unwrap();

    let store = DenseStore::open(dense_path, manifest_path).unwrap();
    let first = store
        .dense_q4_mmap_projection(tensor_name, 2, 4)
        .unwrap()
        .unwrap();
    assert_eq!(store.q4_mmap_projection_cache_len(), 1);
    let second = store
        .dense_q4_mmap_projection(tensor_name, 2, 4)
        .unwrap()
        .unwrap();

    assert_eq!(store.q4_mmap_projection_cache_len(), 1);
    assert_eq!(first.tensor_name, second.tensor_name);
    assert_eq!(first.packed_byte_offset, second.packed_byte_offset);
    assert_eq!(first.scales_byte_offset, second.scales_byte_offset);
    assert_eq!(first.biases_byte_offset, second.biases_byte_offset);
    assert_eq!(first.rows, second.rows);
    assert_eq!(first.cols, second.cols);
}

#[test]
fn dense_transformer_runtime_runs_core_blocks() {
    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2}"#,
        )
        .unwrap();
    let runtime = DenseTransformerRuntime::new(&config);
    let mut hidden = vec![1.0; runtime.width];
    rms_norm_in_place(&mut hidden);
    let before = hidden.clone();
    apply_rotary(&mut hidden, 4, runtime.head_dim, config.rope_theta.unwrap());
    let attended = causal_attention(
        &hidden,
        &[(&before, &before)],
        runtime.num_q_heads,
        runtime.kv_heads,
        runtime.head_dim,
    );
    assert_eq!(attended.len(), runtime.width);
}

#[test]
fn dense_transformer_runtime_uses_full_config_hidden_size() {
    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":4}"#,
        )
        .unwrap();
    let runtime = DenseTransformerRuntime::new(&config);
    assert_eq!(runtime.width, 4096);
    assert_eq!(runtime.head_dim, 128);
}

#[test]
fn lm_head_logits_scores_full_vocab_in_cpu_fallback() {
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
    assert_eq!(tokenizer.vocab_size(), 3);
    assert_eq!(tokenizer.candidate_token_ids(), &[0, 2]);

    let mut bytes = Vec::new();
    for row in 0..tokenizer.vocab_size() {
        let value = (row as f32) + 1.0;
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
                shape: vec![tokenizer.vocab_size(), 2],
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
    let logits = store
        .lm_head_logits("lm_head.weight", &[1.0, 1.0], tokenizer.vocab_size())
        .unwrap();

    assert_eq!(logits.len(), 3);
    assert!(logits.iter().all(|value| value.is_finite()));
}

#[test]
fn lm_head_logits_accepts_padded_vocab_rows() {
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
    for row_idx in 0..5usize {
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
                shape: vec![5, 2],
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
    let logits = store
        .lm_head_logits("lm_head.weight", &[1.0, 1.0], tokenizer.vocab_size())
        .unwrap();

    assert_eq!(logits, vec![2.0, 4.0, 6.0]);
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
        .lm_head_logits("lm_head.weight", &[1.0, 1.0], tokenizer.vocab_size())
        .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("lm_head.weight"), "{err:#}");
    assert!(message.contains("expected at least [3, 2]"), "{err:#}");
    assert!(message.contains("actual shape [2, 2]"), "{err:#}");
}
