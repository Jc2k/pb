use super::*;

#[test]
fn colibri_int4_import_preserves_nibbles_and_builds_affine_bias() {
    let layout =
        dense_q4_layout_with_scale_bias_dtype(&[1, 64], 64, EXPERT_SCALE_BIAS_DTYPE_BF16).unwrap();
    let packed = (0..32).map(|value| value as u8).collect::<Vec<_>>();
    let mut output = Vec::new();

    write_colibri_q4_affine_tensor(
        &mut output,
        "tiny.weight",
        &packed,
        &2.0f32.to_le_bytes(),
        4,
        64,
        layout,
    )
    .unwrap();

    assert_eq!(&output[..packed.len()], packed.as_slice());
    let scale = u16::from_le_bytes(output[32..34].try_into().unwrap()) as u32;
    let bias = u16::from_le_bytes(output[34..36].try_into().unwrap()) as u32;
    assert_eq!(f32::from_bits(scale << 16), 2.0);
    assert_eq!(f32::from_bits(bias << 16), -16.0);
}

#[test]
fn colibri_int8_import_preserves_source_precision_as_bf16() {
    let mut output = Vec::new();
    write_colibri_int8_bf16_tensor(
        &mut output,
        "lm_head.weight",
        &[(-2i8) as u8, 3],
        &0.5f32.to_le_bytes(),
        2,
        &[1, 2],
    )
    .unwrap();

    let first = u16::from_le_bytes(output[0..2].try_into().unwrap()) as u32;
    let second = u16::from_le_bytes(output[2..4].try_into().unwrap()) as u32;
    assert_eq!(f32::from_bits(first << 16), -1.0);
    assert_eq!(f32::from_bits(second << 16), 1.5);
}

#[test]
fn mlx_affine_int8_import_preserves_affine_values_as_bf16() {
    let mut output = Vec::new();
    write_mlx_affine8_bf16_tensor(
        &mut output,
        "model.layers.0.mlp.gate.weight",
        &[0, 1, 2, 255],
        &encode_bf16_bits(0.5).to_le_bytes(),
        &encode_bf16_bits(-1.0).to_le_bytes(),
        &[1, 4],
    )
    .unwrap();

    let decoded = decode_bf16_le(&output).unwrap();
    assert_eq!(decoded, vec![-1.0, -0.5, 0.0, 126.5]);
}

#[test]
fn qwen3_next_static_bf16_import_widens_to_f32() {
    let source = [
        encode_bf16_bits(-2.0).to_le_bytes(),
        encode_bf16_bits(0.25).to_le_bytes(),
    ]
    .concat();
    let mut output = Vec::new();

    write_bf16_as_f32_tensor(&mut output, "linear_attn.A_log", &source, 8).unwrap();

    assert_eq!(decode_f32_le(&output).unwrap(), vec![-2.0, 0.25]);
}

#[test]
fn mlx_mxfp4_import_decodes_e2m1_and_e8m0_before_runtime_q4() {
    let layout =
        dense_q4_layout_with_scale_bias_dtype(&[1, 64], 64, EXPERT_SCALE_BIAS_DTYPE_BF16).unwrap();
    let mut packed = vec![0x91; 16]; // +0.5, -0.5 at E8M0 scale 1.
    packed.extend(vec![0xe6; 16]); // +4, -4 at E8M0 scale 2.
    let mut output = Vec::new();

    write_mlx_mxfp4_affine_tensor(&mut output, "tiny.weight", &packed, &[127, 128], 32, layout)
        .unwrap();

    assert_eq!(output.len(), layout.total_bytes);
    let scale_bits = u16::from_le_bytes(output[32..34].try_into().unwrap()) as u32;
    let bias_bits = u16::from_le_bytes(output[34..36].try_into().unwrap()) as u32;
    let decoded = q4_dequantize_rows_with_group_size(
        &output[..32],
        &[f32::from_bits(scale_bits << 16)],
        &[f32::from_bits(bias_bits << 16)],
        1,
        64,
        64,
    )
    .unwrap();
    for (actual, expected) in decoded[..32].iter().zip([0.5f32, -0.5].into_iter().cycle()) {
        assert!((actual - expected).abs() < 0.6, "{actual} != {expected}");
    }
    for (actual, expected) in decoded[32..].iter().zip([8.0f32, -8.0].into_iter().cycle()) {
        assert!((actual - expected).abs() < 0.6, "{actual} != {expected}");
    }
}

#[test]
fn mla_weight_absorption_uses_compressed_latent_without_expanding_kv() {
    let temp = tempfile::tempdir().unwrap();
    let dense_path = temp.path().join("dense.bin");
    let manifest_path = temp.path().join("manifest.json");
    let tensor_name = attention_tensor_name(0, "kv_b_proj");
    let layout = dense_q4_layout_with_scale_bias_dtype(&[2, 2], 2, "F32").unwrap();
    let mut bytes = vec![0x01, 0x10]; // Wk=[1,0], Wv=[0,1]
    for value in [1.0f32, 1.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in [0.0f32, 0.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(bytes.len(), layout.total_bytes);
    fs::write(&dense_path, &bytes).unwrap();
    fs::write(
        &manifest_path,
        serde_json::to_vec(&FlashMoeManifest {
            model: "tiny-glm".to_string(),
            cache_version: "test".to_string(),
            dense_shards: vec!["tiny.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: tensor_name,
                shard: "tiny.safetensors".to_string(),
                dtype: "U32".to_string(),
                shape: vec![2, 2],
                source_offsets: [0, 2],
                runtime_offset: 0,
                byte_len: layout.total_bytes as u64,
                quantization: TensorQuantization::Q4 {
                    group_size: 2,
                    format: "test".to_string(),
                    scale_bias_dtype: "F32".to_string(),
                },
                q4_sources: None,
            }],
        })
        .unwrap(),
    )
    .unwrap();
    let store = DenseStore::open(dense_path, manifest_path).unwrap();
    let mla = MlaAttentionLayout {
        q_lora_rank: 2,
        kv_lora_rank: 2,
        qk_nope_head_dim: 1,
        qk_rope_head_dim: 2,
        qk_head_dim: 3,
        v_head_dim: 1,
        num_heads: 1,
        q_width: 3,
        kv_a_width: 4,
        kv_b_width: 2,
        attention_output_width: 1,
        kv_projection: MlaKvProjectionLayout::FusedKvB,
    };
    let latent = [1.0, 3.0];
    let rotary = [0.0, 0.0];
    let output = store
        .mla_absorbed_attention(0, mla, &[2.0, 0.0, 0.0], &[(&latent, &rotary)])
        .unwrap();

    assert_eq!(output, vec![3.0]);
}

#[test]
fn mla_weight_absorption_accepts_mlx_preabsorbed_multilinear_weights() {
    let temp = tempfile::tempdir().unwrap();
    let dense_path = temp.path().join("dense.bin");
    let manifest_path = temp.path().join("manifest.json");
    let embed_name = attention_tensor_name(0, "embed_q");
    let unembed_name = attention_tensor_name(0, "unembed_out");
    let embed_layout = dense_q4_layout_with_scale_bias_dtype(&[1, 2, 1], 1, "F32").unwrap();
    let unembed_layout = dense_q4_layout_with_scale_bias_dtype(&[1, 1, 2], 2, "F32").unwrap();
    let mut bytes = vec![0x01, 0x00]; // embed_q maps [q] to [q, 0].
    for value in [1.0f32, 1.0, 0.0, 0.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(bytes.len(), embed_layout.total_bytes);
    let unembed_offset = bytes.len() as u64;
    bytes.push(0x10); // unembed_out maps [x, y] to y.
    for value in [1.0f32, 0.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(
        bytes.len(),
        embed_layout.total_bytes + unembed_layout.total_bytes
    );
    fs::write(&dense_path, &bytes).unwrap();
    fs::write(
        &manifest_path,
        serde_json::to_vec(&FlashMoeManifest {
            model: "tiny-glm-mlx".to_string(),
            cache_version: "test".to_string(),
            dense_shards: vec!["tiny.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![
                DenseTensorRef {
                    tensor: embed_name,
                    shard: "tiny.safetensors".to_string(),
                    dtype: "U32".to_string(),
                    shape: vec![1, 2, 1],
                    source_offsets: [0, 2],
                    runtime_offset: 0,
                    byte_len: embed_layout.total_bytes as u64,
                    quantization: TensorQuantization::Q4 {
                        group_size: 1,
                        format: "test".to_string(),
                        scale_bias_dtype: "F32".to_string(),
                    },
                    q4_sources: None,
                },
                DenseTensorRef {
                    tensor: unembed_name,
                    shard: "tiny.safetensors".to_string(),
                    dtype: "U32".to_string(),
                    shape: vec![1, 1, 2],
                    source_offsets: [0, 1],
                    runtime_offset: unembed_offset,
                    byte_len: unembed_layout.total_bytes as u64,
                    quantization: TensorQuantization::Q4 {
                        group_size: 2,
                        format: "test".to_string(),
                        scale_bias_dtype: "F32".to_string(),
                    },
                    q4_sources: None,
                },
            ],
        })
        .unwrap(),
    )
    .unwrap();
    let store = DenseStore::open(dense_path, manifest_path).unwrap();
    let mla = MlaAttentionLayout {
        q_lora_rank: 2,
        kv_lora_rank: 2,
        qk_nope_head_dim: 1,
        qk_rope_head_dim: 2,
        qk_head_dim: 3,
        v_head_dim: 1,
        num_heads: 1,
        q_width: 3,
        kv_a_width: 4,
        kv_b_width: 2,
        attention_output_width: 1,
        kv_projection: MlaKvProjectionLayout::AbsorbedMultiLinear,
    };
    let latent = [1.0, 3.0];
    let rotary = [0.0, 0.0];
    let output = store
        .mla_absorbed_attention(0, mla, &[2.0, 0.0, 0.0], &[(&latent, &rotary)])
        .unwrap();

    assert_eq!(output, vec![3.0]);
}

#[test]
fn qwen_family_tensor_names_are_canonicalized_for_runtime() {
    assert_eq!(
        canonical_hf_tensor_name("model.language_model.embed_tokens.weight"),
        "model.embed_tokens.weight"
    );
    assert_eq!(
        canonical_hf_tensor_name("language_model.model.layers.3.self_attn.q_proj.weight"),
        "model.layers.3.self_attn.q_proj.weight"
    );
    assert_eq!(
        canonical_hf_tensor_name("language_model.lm_head.weight"),
        "lm_head.weight"
    );
    assert_eq!(
        canonical_hf_tensor_name("model.visual.patch_embed.proj.weight"),
        "visual.patch_embed.proj.weight"
    );
    assert_eq!(
        canonical_hf_tensor_name("vision_tower.blocks.7.mlp.linear_fc1.weight"),
        "visual.blocks.7.mlp.fc1.weight"
    );
    assert_eq!(
        canonical_hf_tensor_name("vision_tower.merger.linear_fc2.weight"),
        "visual.merger.linear_fc2.weight"
    );
    assert_eq!(canonical_hf_tensor_name("lm_head.weight"), "lm_head.weight");
}

fn layout_config() -> QwenModelConfig {
    QwenModelConfig {
        model_type: Some("qwen3_moe".to_string()),
        architectures: Some(vec!["Qwen3MoeForCausalLM".to_string()]),
        num_hidden_layers: 1,
        hidden_size: 8,
        num_attention_heads: 2,
        head_dim: Some(4),
        num_key_value_heads: Some(1),
        vocab_size: 32,
        rope_theta: Some(1_000_000.0),
        partial_rotary_factor: None,
        torch_dtype: Some("float32".to_string()),
        num_experts: Some(4),
        num_experts_per_tok: Some(2),
        norm_topk_prob: Some(true),
        moe_intermediate_size: Some(16),
        intermediate_size: None,
        max_position_embeddings: Some(1024),
        full_attention_interval: None,
        linear_attention: None,
        mrope_section: None,
        tie_word_embeddings: Some(true),
        num_shared_experts: None,
        shared_expert_intermediate_size: None,
        vision_config: None,
        glm: None,
    }
}

fn layout_tensor(name: &str, shape: &[usize]) -> RuntimeTensorEntry {
    RuntimeTensorEntry {
        name: name.to_string(),
        dtype: "F32".to_string(),
        shape: shape.to_vec(),
        byte_offset: 0,
        byte_len: shape.iter().product::<usize>() as u64 * 4,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    }
}

#[test]
fn dense_runtime_layout_resolves_full_attention_from_manifest_shapes() {
    let registry = TensorRegistry {
        tensors: BTreeMap::from([
            (
                attention_tensor_name(0, "q_proj"),
                layout_tensor(&attention_tensor_name(0, "q_proj"), &[8, 8]),
            ),
            (
                attention_tensor_name(0, "k_proj"),
                layout_tensor(&attention_tensor_name(0, "k_proj"), &[4, 8]),
            ),
            (
                attention_tensor_name(0, "v_proj"),
                layout_tensor(&attention_tensor_name(0, "v_proj"), &[4, 8]),
            ),
            (
                attention_tensor_name(0, "o_proj"),
                layout_tensor(&attention_tensor_name(0, "o_proj"), &[8, 8]),
            ),
        ]),
    };

    let runtime = DenseTransformerRuntime::from_registry(&layout_config(), &registry).unwrap();
    let layout = runtime.full_attention_layout(0).unwrap();
    assert_eq!(layout.q_layout, FullAttentionQLayout::Standard);
    assert_eq!(layout.q_width, 8);
    assert_eq!(layout.kv_width, 4);
    assert_eq!(layout.head_dim, 4);
    assert_eq!(layout.rotary_dim, 4);
}

#[test]
fn dense_runtime_layout_rejects_mixed_attention_implementations() {
    let mut tensors = BTreeMap::from([(
        attention_tensor_name(0, "q_proj"),
        layout_tensor(&attention_tensor_name(0, "q_proj"), &[8, 8]),
    )]);
    tensors.insert(
        linear_attention_tensor_name(0, "in_proj_qkv"),
        layout_tensor(&linear_attention_tensor_name(0, "in_proj_qkv"), &[8, 8]),
    );
    let error =
        DenseTransformerRuntime::from_registry(&layout_config(), &TensorRegistry { tensors })
            .unwrap_err();

    assert!(error.to_string().contains("both linear-attention tensors"));
}

#[test]
fn required_manifest_validation_resolves_complete_full_attention_layer() {
    let entries = [
        ("model.embed_tokens.weight".to_string(), vec![32, 8]),
        ("model.norm.weight".to_string(), vec![8]),
        (attention_tensor_name(0, "q_proj"), vec![8, 8]),
        (attention_tensor_name(0, "k_proj"), vec![4, 8]),
        (attention_tensor_name(0, "v_proj"), vec![4, 8]),
        (attention_tensor_name(0, "o_proj"), vec![8, 8]),
        (layer_norm_tensor_name(0, "self_attn.q_norm"), vec![4]),
        (layer_norm_tensor_name(0, "self_attn.k_norm"), vec![4]),
        (layer_norm_tensor_name(0, "input_layernorm"), vec![8]),
        (
            layer_norm_tensor_name(0, "post_attention_layernorm"),
            vec![8],
        ),
        (router_tensor_name(0), vec![4, 8]),
    ];
    let registry = TensorRegistry {
        tensors: entries
            .into_iter()
            .map(|(name, shape)| {
                let tensor = layout_tensor(&name, &shape);
                (name, tensor)
            })
            .collect(),
    };

    validate_required_tensor_manifest(&layout_config(), &registry).unwrap();
}

#[test]
fn dense_cache_conversion_resolves_native_mlx_q4_layout() {
    assert_eq!(logical_shape_for_mlx_q4(&[3, 4]).unwrap(), vec![3, 32]);
    let native = DenseQ4SourceRefs {
        scales_shard: "scales.safetensors".to_string(),
        scales_offsets: [0, 8],
        biases_shard: "biases.safetensors".to_string(),
        biases_offsets: [0, 8],
        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        source_format: DenseQ4SourceFormat::MlxAffine,
        source_group_size: None,
        source_row_order: None,
    };

    assert_eq!(
        dense_tensor_quantization(
            "model.layers.0.self_attn.q_proj.weight",
            "U32",
            &Some(native)
        ),
        TensorQuantization::Q4 {
            group_size: GROUP_SIZE,
            format: DENSE_Q4_MLX_FORMAT.to_string(),
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        }
    );

    let native_int8 = DenseQ4SourceRefs {
        scales_shard: "scales.safetensors".to_string(),
        scales_offsets: [0, 8],
        biases_shard: "biases.safetensors".to_string(),
        biases_offsets: [0, 8],
        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        source_format: DenseQ4SourceFormat::MlxAffine8,
        source_group_size: None,
        source_row_order: None,
    };
    assert_eq!(
        logical_shape_for_mlx_source(&[3, 4], &native_int8).unwrap(),
        vec![3, 16]
    );
    assert_eq!(
        dense_tensor_quantization("model.layers.0.mlp.gate.weight", "U32", &Some(native_int8)),
        TensorQuantization::None
    );
}

#[test]
fn cache_writer_reorders_complete_source_rows_without_value_conversion() {
    let source = (0u8..24).collect::<Vec<_>>();
    let mut output = Vec::new();

    write_rows_in_order(
        &mut output,
        &source,
        3,
        &[2, 0, 7, 3],
        "combined.weight",
        "packed weights",
    )
    .unwrap();

    assert_eq!(output, vec![6, 7, 8, 0, 1, 2, 21, 22, 23, 9, 10, 11]);
}

fn runtime_matrix(name: &str, dtype: &str, quantization: TensorQuantization) -> RuntimeTensorEntry {
    RuntimeTensorEntry {
        name: name.to_string(),
        dtype: dtype.to_string(),
        shape: vec![4, 8],
        byte_offset: 0,
        byte_len: 128,
        alignment: TENSOR_ALIGNMENT,
        quantization,
    }
}

#[test]
fn tensor_quantization_defaults_to_unquantized_dense() {
    assert_eq!(TensorQuantization::default(), TensorQuantization::None);
}

#[test]
fn tensor_quantization_q4_defaults_scale_bias_dtype_for_legacy_manifests() {
    let quantization: TensorQuantization =
        serde_json::from_str(r#"{"Q4":{"group_size":16,"format":"dense-q4"}}"#).unwrap();

    assert_eq!(
        quantization,
        TensorQuantization::Q4 {
            group_size: 16,
            format: "dense-q4".to_string(),
            scale_bias_dtype: "F32".to_string(),
        }
    );
}

#[test]
fn tensor_registry_resolves_one_concrete_dense_layout() {
    for (dtype, expected) in [
        ("BF16", ResidentDenseLayout::Bf16),
        ("F16", ResidentDenseLayout::F16),
        ("F32", ResidentDenseLayout::F32),
    ] {
        let registry = TensorRegistry {
            tensors: BTreeMap::from([(
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                runtime_matrix(
                    "model.layers.0.self_attn.q_proj.weight",
                    dtype,
                    TensorQuantization::None,
                ),
            )]),
        };

        assert_eq!(registry.resolve_resident_dense_layout().unwrap(), expected);
    }
}

#[test]
fn tensor_registry_resolves_q4_with_unquantized_auxiliary_matrices() {
    let registry = TensorRegistry {
        tensors: BTreeMap::from([
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                runtime_matrix(
                    "model.layers.0.self_attn.q_proj.weight",
                    "U32",
                    TensorQuantization::Q4 {
                        group_size: 64,
                        format: "mlx-q4".to_string(),
                        scale_bias_dtype: "BF16".to_string(),
                    },
                ),
            ),
            (
                "model.embed_tokens.weight".to_string(),
                runtime_matrix(
                    "model.embed_tokens.weight",
                    "BF16",
                    TensorQuantization::None,
                ),
            ),
        ]),
    };

    assert_eq!(
        registry.resolve_resident_dense_layout().unwrap(),
        ResidentDenseLayout::Q4
    );
}

#[test]
fn tensor_registry_ignores_routed_expert_storage_when_resolving_dense_layout() {
    let registry = TensorRegistry {
        tensors: BTreeMap::from([
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                runtime_matrix(
                    "model.layers.0.self_attn.q_proj.weight",
                    "BF16",
                    TensorQuantization::None,
                ),
            ),
            (
                "model.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
                runtime_matrix(
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    "U32",
                    TensorQuantization::Q4 {
                        group_size: 64,
                        format: "expert-q4".to_string(),
                        scale_bias_dtype: "F32".to_string(),
                    },
                ),
            ),
        ]),
    };

    assert_eq!(
        registry.resolve_resident_dense_layout().unwrap(),
        ResidentDenseLayout::Bf16
    );
}

#[test]
fn tensor_registry_rejects_mixed_unquantized_matrix_layouts() {
    let registry = TensorRegistry {
        tensors: BTreeMap::from([
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                runtime_matrix(
                    "model.layers.0.self_attn.q_proj.weight",
                    "BF16",
                    TensorQuantization::None,
                ),
            ),
            (
                "lm_head.weight".to_string(),
                runtime_matrix("lm_head.weight", "F32", TensorQuantization::None),
            ),
        ]),
    };

    let err = registry.resolve_resident_dense_layout().unwrap_err();
    assert!(
        err.to_string().contains("mixes resident matrix layouts"),
        "{err:#}"
    );
}

#[test]
fn dense_tensor_ref_preserves_runtime_binding_offsets() {
    let tensor = DenseTensorRef {
        tensor: "model.embed_tokens.weight".to_string(),
        shard: "model-00001.safetensors".to_string(),
        dtype: "BF16".to_string(),
        shape: vec![8, 4],
        source_offsets: [128, 192],
        runtime_offset: 4096,
        byte_len: 64,
        quantization: TensorQuantization::None,
        q4_sources: None,
    };

    assert_eq!(tensor.runtime_offset, 4096);
    assert_eq!(tensor.byte_len, 64);
    assert_eq!(tensor.quantization, TensorQuantization::None);
}

#[test]
fn tensor_registry_builds_dense_aliases_from_manifest() {
    let manifest = FlashMoeManifest {
        model: "fixture".to_string(),
        cache_version: "test".to_string(),
        dense_shards: Vec::new(),
        expert_tensors: Vec::new(),
        dense_tensors: vec![DenseTensorRef {
            tensor: "model.language_model.layers.7.self_attn.q_proj.weight".to_string(),
            shard: "model.safetensors".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![4, 8],
            source_offsets: [0, 64],
            runtime_offset: 4096,
            byte_len: 64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }],
    };

    let registry = TensorRegistry::from_manifest(&manifest);
    let alias = registry
        .tensor("model.layers.7.self_attn.q_proj.weight")
        .unwrap();

    assert_eq!(
        alias.name,
        "model.language_model.layers.7.self_attn.q_proj.weight"
    );
    assert_eq!(alias.byte_offset, 4096);
    assert_eq!(alias.alignment, TENSOR_ALIGNMENT);
    assert!(registry.has_tensor_with_prefix("model.layers.7"));
}

#[test]
fn tensor_registry_keeps_expert_manifest_refs_as_import_compatibility() {
    let manifest = FlashMoeManifest {
        model: "fixture".to_string(),
        cache_version: "test".to_string(),
        dense_shards: Vec::new(),
        dense_tensors: Vec::new(),
        expert_tensors: vec![ExpertTensorRef {
            tensor: "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
            shard: "model.safetensors".to_string(),
            layer: Some(0),
            expert: Some(1),
            dtype: Some("F32".to_string()),
            shape: vec![2, 4],
            source_offsets: Some([128, 256]),
            q4_sources: None,
        }],
    };

    let registry = TensorRegistry::from_manifest(&manifest);
    let tensor = registry
        .tensor("model.layers.0.mlp.experts.1.gate_proj.weight")
        .unwrap();

    assert_eq!(tensor.byte_offset, 128);
    assert_eq!(tensor.byte_len, 128);
    assert_eq!(
        tensor.quantization,
        TensorQuantization::Q4 {
            group_size: GROUP_SIZE,
            format: ExpertQuantization::FourBitProduction.as_str().to_string(),
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
        }
    );
}

#[test]
fn dense_mmap_projection_stride_uses_runtime_cols() {
    let projection = DenseMmapMatvecProjection {
        tensor_name: "model.layers.0.self_attn.q_proj.weight".to_string(),
        byte_offset: 4096,
        dtype: ResidentStaticDtype::Bf16,
        rows: 16,
        cols: 32,
        output_width: 64,
    };

    assert_eq!(projection.stride(), 32);
}

#[test]
fn dense_mmap_projection_descriptor_resolves_entry_bounds() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.0.self_attn.q_proj.weight".to_string(),
        dtype: "BF16".to_string(),
        shape: vec![4, 8],
        byte_offset: 64,
        byte_len: 64,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    };

    let projection = DenseMmapMatvecProjection::from_entry(
        "model.layers.0.self_attn.q_proj.weight",
        &entry,
        256,
        4,
        8,
    )
    .unwrap();

    assert_eq!(projection.byte_offset, 64);
    assert_eq!(projection.rows, 4);
    assert_eq!(projection.cols, 8);
    assert_eq!(projection.output_width, 4);
}

#[test]
fn resident_projection_binding_resolves_bf16_f16_and_f32_without_layout_probe() {
    for (dtype, element_size) in [("BF16", 2), ("F16", 2), ("F32", 4)] {
        let entry = RuntimeTensorEntry {
            name: format!("model.layers.0.self_attn.{dtype}_proj.weight"),
            dtype: dtype.to_string(),
            shape: vec![3, 4],
            byte_offset: 64,
            byte_len: (12 * element_size) as u64,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let projection =
            ResidentMmapMatvecProjection::from_entry(&entry.name, &entry, 256, 3, 4).unwrap();
        assert_eq!(projection.tensor_name(), entry.name);
        assert_eq!(projection.rows(), 3);
        assert_eq!(projection.cols(), 4);
        assert_eq!(projection.output_width(), 3);
        assert!(matches!(projection, ResidentMmapMatvecProjection::Dense(_)));
    }

    let unsupported = RuntimeTensorEntry {
        name: "model.layers.0.self_attn.i8_proj.weight".to_string(),
        dtype: "I8".to_string(),
        shape: vec![3, 4],
        byte_offset: 64,
        byte_len: 12,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    };
    let error =
        ResidentMmapMatvecProjection::from_entry(&unsupported.name, &unsupported, 256, 3, 4)
            .unwrap_err();
    assert!(
        error.to_string().contains("unsupported dtype I8"),
        "{error:#}"
    );
}

#[test]
fn resident_static_dtype_canonicalizes_supported_manifest_aliases() {
    for (declared, expected) in [
        ("bfloat16", ResidentStaticDtype::Bf16),
        ("fp16", ResidentStaticDtype::F16),
        ("float32", ResidentStaticDtype::F32),
    ] {
        assert_eq!(ResidentStaticDtype::from_declared(declared), Some(expected));
    }
    assert_eq!(ResidentStaticDtype::from_declared("I8"), None);
}

#[test]
fn router_score_projection_descriptor_resolves_dense_binding() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.3.mlp.gate.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![2, 4],
        byte_offset: 64,
        byte_len: 32,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    };

    let descriptor =
        RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 128, 2, 4).unwrap();

    assert_eq!(descriptor.layer, 3);
    assert_eq!(descriptor.experts, 2);
    assert_eq!(descriptor.hidden_width, 4);
    match descriptor.binding {
        RouterScoreProjectionBinding::ResidentDense(projection) => {
            assert_eq!(projection.tensor_name, entry.name);
            assert_eq!(projection.byte_offset, 64);
            assert_eq!(projection.output_width, 2);
        }
        RouterScoreProjectionBinding::ResidentQ4(_) => panic!("expected dense binding"),
    }
}

#[test]
fn router_score_projection_topk_plan_declares_dense_binding() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.3.mlp.gate.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![2, 4],
        byte_offset: 64,
        byte_len: 32,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    };
    let descriptor =
        RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 128, 2, 4).unwrap();

    let plan = descriptor.topk_plan(4, 1).unwrap();
    assert_eq!(plan.layer, 3);
    assert_eq!(plan.tensor_name, entry.name);
    assert_eq!(plan.experts, 2);
    assert_eq!(plan.hidden_width, 4);
    assert_eq!(plan.active_experts, 1);
    match plan.source {
        RouterScoreProjectionTopKSource::ResidentDense(projection) => {
            assert_eq!(projection.byte_offset, 64);
            assert_eq!(projection.rows, 2);
            assert_eq!(projection.cols, 4);
        }
        RouterScoreProjectionTopKSource::ResidentQ4(_) => panic!("expected dense topK plan"),
    }

    let hidden_err = descriptor.topk_plan(3, 1).unwrap_err();
    assert!(
        hidden_err
            .to_string()
            .contains("topK hidden length 3 does not match declared width 4"),
        "{hidden_err:#}"
    );
    let active_err = descriptor.topk_plan(4, 0).unwrap_err();
    assert!(
        active_err
            .to_string()
            .contains("active experts 0 is outside declared expert range 1..=2"),
        "{active_err:#}"
    );
}

#[test]
fn router_score_projection_descriptor_resolves_q4_binding() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.3.mlp.gate.weight".to_string(),
        dtype: "Q4".to_string(),
        shape: vec![2, 4],
        byte_offset: 128,
        byte_len: 12,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::Q4 {
            group_size: 16,
            format: "dense-q4".to_string(),
            scale_bias_dtype: "BF16".to_string(),
        },
    };

    let descriptor =
        RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 256, 2, 4).unwrap();

    match descriptor.binding {
        RouterScoreProjectionBinding::ResidentQ4(projection) => {
            assert_eq!(projection.packed_byte_offset, 128);
            assert_eq!(projection.output_width, 2);
            assert_eq!(projection.cols, 4);
        }
        RouterScoreProjectionBinding::ResidentDense(_) => panic!("expected q4 binding"),
    }
}

#[test]
fn router_score_projection_topk_plan_declares_q4_binding() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.3.mlp.gate.weight".to_string(),
        dtype: "Q4".to_string(),
        shape: vec![2, 4],
        byte_offset: 128,
        byte_len: 12,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::Q4 {
            group_size: 16,
            format: "dense-q4".to_string(),
            scale_bias_dtype: "BF16".to_string(),
        },
    };
    let descriptor =
        RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 256, 2, 4).unwrap();

    let plan = descriptor.topk_plan(4, 2).unwrap();
    assert_eq!(plan.active_experts, 2);
    match plan.source {
        RouterScoreProjectionTopKSource::ResidentQ4(projection) => {
            assert_eq!(projection.packed_byte_offset, 128);
            assert_eq!(projection.output_width, 2);
            assert_eq!(projection.cols, 4);
        }
        RouterScoreProjectionTopKSource::ResidentDense(_) => panic!("expected q4 topK plan"),
    }
}

#[test]
fn router_score_projection_builder_uses_canonical_layer_tensor_name() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.3.mlp.gate.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![2, 4],
        byte_offset: 64,
        byte_len: 32,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    };
    let mut seen_name = None;

    let descriptor = build_router_score_projection_descriptor(3, 2, 4, 128, |name| {
        seen_name = Some(name.to_string());
        (name == entry.name).then_some(&entry)
    })
    .unwrap()
    .unwrap();

    assert_eq!(seen_name.unwrap(), "model.layers.3.mlp.gate.weight");
    assert_eq!(descriptor.tensor_name, entry.name);
    assert_eq!(descriptor.experts, 2);
    assert_eq!(descriptor.hidden_width, 4);
}

#[test]
fn router_score_projection_builder_returns_none_for_missing_router() {
    let descriptor = build_router_score_projection_descriptor(3, 2, 4, 128, |_| None).unwrap();

    assert!(descriptor.is_none());
}

#[test]
fn router_score_projection_builder_rejects_wrong_shape_without_fallback() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.3.mlp.gate.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![3, 4],
        byte_offset: 0,
        byte_len: 48,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    };

    let err = build_router_score_projection_descriptor(3, 2, 4, 64, |name| {
        (name == entry.name).then_some(&entry)
    })
    .unwrap_err();

    assert!(err.to_string().contains("shape mismatch"), "{err:#}");
}

#[test]
fn router_score_projection_execution_declares_binding_without_fallback() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.3.mlp.gate.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![2, 4],
        byte_offset: 64,
        byte_len: 32,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    };
    let descriptor =
        RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 128, 2, 4).unwrap();

    let execution = descriptor.execution(3, 2, 4).unwrap();
    assert_eq!(execution.layer, 3);
    assert_eq!(execution.tensor_name, entry.name);
    assert_eq!(execution.experts, 2);
    assert_eq!(execution.hidden_width, 4);
    assert_eq!(
        execution.kind,
        RouterScoreProjectionExecutionKind::ResidentDense
    );
    let score_plan = execution.score_plan(4).unwrap();
    assert_eq!(score_plan.tensor_name, entry.name);
    assert_eq!(score_plan.experts, 2);
    assert_eq!(score_plan.hidden_width, 4);
    assert_eq!(
        score_plan.source,
        RouterScoreProjectionScoreSource::ResidentDenseFullTensor
    );

    let hidden_err = execution.score_plan(3).unwrap_err();
    assert!(
        hidden_err
            .to_string()
            .contains("hidden length 3 does not match declared width 4"),
        "{hidden_err:#}"
    );

    let err = descriptor.execution(3, 3, 4).unwrap_err();
    assert!(
        err.to_string()
            .contains("experts 2 does not match scheduled experts 3"),
        "{err:#}"
    );
}

#[test]
fn router_score_projection_score_plan_declares_q4_row_execution() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.3.mlp.gate.weight".to_string(),
        dtype: "Q4".to_string(),
        shape: vec![2, 32],
        byte_offset: 64,
        byte_len: 48,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::Q4 {
            group_size: 16,
            format: "dense-q4".to_string(),
            scale_bias_dtype: "BF16".to_string(),
        },
    };
    let descriptor =
        RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 512, 2, 32).unwrap();

    let execution = descriptor.execution(3, 2, 32).unwrap();
    assert_eq!(
        execution.kind,
        RouterScoreProjectionExecutionKind::ResidentQ4
    );
    assert_eq!(
        execution.score_plan(32).unwrap().source,
        RouterScoreProjectionScoreSource::DeclaredRows
    );
}

#[test]
fn router_score_projection_descriptor_rejects_wrong_shape_without_fallback() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.3.mlp.gate.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![3, 4],
        byte_offset: 0,
        byte_len: 48,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    };

    let err =
        RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 64, 2, 4).unwrap_err();

    assert!(err.to_string().contains("shape mismatch"), "{err:#}");
}

#[test]
fn cmd2_resident_post_attention_prep_projection_bundle_resolves_bindings() {
    let projections = build_cmd2_resident_post_attention_prep_projections(
        7,
        16,
        "model.layers.7.self_attn.o_proj.weight",
        24,
        32,
        4,
        |name, output_width, input_len| {
            Ok(Some(ResidentMmapMatvecProjection::Q4(
                DenseQ4MmapMatvecProjection {
                    tensor_name: name.to_string(),
                    packed_byte_offset: 128,
                    scales_byte_offset: 256,
                    biases_byte_offset: 512,
                    rows: output_width,
                    cols: input_len,
                    output_width,
                    row_packed_bytes: input_len.div_ceil(2),
                    groups_per_row: input_len.div_ceil(16),
                    group_size: 16,
                    scale_bias_dtype: "BF16".to_string(),
                },
            )))
        },
    )
    .unwrap()
    .unwrap();

    assert_eq!(projections.layer, 7);
    assert_eq!(projections.experts, 16);
    assert_eq!(projections.residual_width, 32);
    assert_eq!(projections.attention_width, 24);
    assert_eq!(projections.active_experts, 4);
    assert_eq!(
        projections.out_proj.tensor_name(),
        "model.layers.7.self_attn.o_proj.weight"
    );
    assert_eq!(
        projections.router.tensor_name(),
        "model.layers.7.mlp.gate.weight"
    );
    assert_eq!(projections.out_proj.output_width(), 32);
    assert_eq!(projections.out_proj.cols(), 24);
    assert_eq!(projections.router.output_width(), 16);
    assert_eq!(projections.router.cols(), 32);
}

#[test]
fn cmd2_resident_post_attention_prep_plan_declares_executable_shape() {
    let projections = build_cmd2_resident_post_attention_prep_projections(
        7,
        16,
        "model.layers.7.self_attn.o_proj.weight",
        24,
        32,
        20,
        |name, output_width, input_len| {
            Ok(Some(ResidentMmapMatvecProjection::Q4(
                DenseQ4MmapMatvecProjection {
                    tensor_name: name.to_string(),
                    packed_byte_offset: 128,
                    scales_byte_offset: 256,
                    biases_byte_offset: 512,
                    rows: output_width,
                    cols: input_len,
                    output_width,
                    row_packed_bytes: input_len.div_ceil(2),
                    groups_per_row: input_len.div_ceil(16),
                    group_size: 16,
                    scale_bias_dtype: "BF16".to_string(),
                },
            )))
        },
    )
    .unwrap()
    .unwrap();

    let plan = projections.resident_plan(24, 32, 32).unwrap();

    assert_eq!(
        plan,
        Cmd2ResidentPostAttentionPrepPlan {
            layer: 7,
            width: 32,
            attention_width: 24,
            experts: 16,
            active_count: 16,
        }
    );
}

#[test]
fn cmd2_resident_post_attention_prep_plan_rejects_undeclared_inputs() {
    let projections = build_cmd2_resident_post_attention_prep_projections(
        7,
        16,
        "model.layers.7.self_attn.o_proj.weight",
        24,
        32,
        4,
        |name, output_width, input_len| {
            Ok(Some(ResidentMmapMatvecProjection::Q4(
                DenseQ4MmapMatvecProjection {
                    tensor_name: name.to_string(),
                    packed_byte_offset: 128,
                    scales_byte_offset: 256,
                    biases_byte_offset: 512,
                    rows: output_width,
                    cols: input_len,
                    output_width,
                    row_packed_bytes: input_len.div_ceil(2),
                    groups_per_row: input_len.div_ceil(16),
                    group_size: 16,
                    scale_bias_dtype: "BF16".to_string(),
                },
            )))
        },
    )
    .unwrap()
    .unwrap();

    let norm_err = projections.resident_plan(24, 32, 31).unwrap_err();
    assert!(
        norm_err
            .to_string()
            .contains("norm weight length 31 does not match residual width 32"),
        "{norm_err:#}"
    );
}

#[test]
fn cmd2_resident_post_attention_prep_plan_errors_on_undeclared_shape() {
    let projections = Cmd2ResidentPostAttentionPrepProjections {
        layer: 7,
        out_proj: ResidentMmapMatvecProjection::Q4(DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.7.self_attn.o_proj.weight".to_string(),
            packed_byte_offset: 128,
            scales_byte_offset: 256,
            biases_byte_offset: 512,
            rows: 32,
            cols: 25,
            output_width: 32,
            row_packed_bytes: 13,
            groups_per_row: 2,
            group_size: 16,
            scale_bias_dtype: "BF16".to_string(),
        }),
        router: ResidentMmapMatvecProjection::Q4(DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.7.mlp.gate.weight".to_string(),
            packed_byte_offset: 128,
            scales_byte_offset: 256,
            biases_byte_offset: 512,
            rows: 16,
            cols: 32,
            output_width: 16,
            row_packed_bytes: 16,
            groups_per_row: 2,
            group_size: 16,
            scale_bias_dtype: "BF16".to_string(),
        }),
        experts: 16,
        residual_width: 32,
        attention_width: 24,
        active_experts: 4,
    };

    let err = projections.resident_plan(24, 32, 32).unwrap_err();
    assert!(
        err.to_string()
            .contains("projection shapes out=32x25 rows=32 router=16x32 rows=16"),
        "{err:#}"
    );
}

#[test]
fn cmd2_resident_post_attention_prep_projection_bundle_skips_missing_bindings() {
    let missing_out = build_cmd2_resident_post_attention_prep_projections(
        7,
        16,
        "model.layers.7.self_attn.o_proj.weight",
        24,
        32,
        4,
        |name, output_width, input_len| {
            if name.ends_with("o_proj.weight") {
                return Ok(None);
            }
            Ok(Some(ResidentMmapMatvecProjection::Q4(
                DenseQ4MmapMatvecProjection {
                    tensor_name: name.to_string(),
                    packed_byte_offset: 128,
                    scales_byte_offset: 256,
                    biases_byte_offset: 512,
                    rows: output_width,
                    cols: input_len,
                    output_width,
                    row_packed_bytes: input_len.div_ceil(2),
                    groups_per_row: input_len.div_ceil(16),
                    group_size: 16,
                    scale_bias_dtype: "BF16".to_string(),
                },
            )))
        },
    )
    .unwrap();
    assert!(missing_out.is_none());

    let disabled =
        build_cmd2_resident_post_attention_prep_projections(7, 0, "out", 24, 32, 4, |_, _, _| {
            panic!("disabled CMD2 prep must not request projections")
        })
        .unwrap();
    assert!(disabled.is_none());
}

#[test]
fn required_cmd2_resident_post_attention_prep_projection_errors_on_missing_bindings() {
    let missing_out = build_required_cmd2_resident_post_attention_prep_projections(
        7,
        16,
        "model.layers.7.self_attn.o_proj.weight",
        24,
        32,
        4,
        |name, output_width, input_len| {
            if name.ends_with("o_proj.weight") {
                return Ok(None);
            }
            Ok(Some(ResidentMmapMatvecProjection::Q4(
                DenseQ4MmapMatvecProjection {
                    tensor_name: name.to_string(),
                    packed_byte_offset: 128,
                    scales_byte_offset: 256,
                    biases_byte_offset: 512,
                    rows: output_width,
                    cols: input_len,
                    output_width,
                    row_packed_bytes: input_len.div_ceil(2),
                    groups_per_row: input_len.div_ceil(16),
                    group_size: 16,
                    scale_bias_dtype: "BF16".to_string(),
                },
            )))
        },
    )
    .unwrap_err();
    assert!(
        missing_out
            .to_string()
            .contains("missing output projection"),
        "{missing_out:#}"
    );

    let missing_router = build_required_cmd2_resident_post_attention_prep_projections(
        7,
        16,
        "model.layers.7.self_attn.o_proj.weight",
        24,
        32,
        4,
        |name, output_width, input_len| {
            if name.ends_with("mlp.gate.weight") {
                return Ok(None);
            }
            Ok(Some(ResidentMmapMatvecProjection::Q4(
                DenseQ4MmapMatvecProjection {
                    tensor_name: name.to_string(),
                    packed_byte_offset: 128,
                    scales_byte_offset: 256,
                    biases_byte_offset: 512,
                    rows: output_width,
                    cols: input_len,
                    output_width,
                    row_packed_bytes: input_len.div_ceil(2),
                    groups_per_row: input_len.div_ceil(16),
                    group_size: 16,
                    scale_bias_dtype: "BF16".to_string(),
                },
            )))
        },
    )
    .unwrap_err();
    assert!(
        missing_router
            .to_string()
            .contains("missing router projection model.layers.7.mlp.gate.weight"),
        "{missing_router:#}"
    );
}

#[test]
fn cmd2_resident_post_attention_prep_projection_bundle_rejects_mismatched_shape() {
    let err = build_cmd2_resident_post_attention_prep_projections(
        7,
        16,
        "model.layers.7.self_attn.o_proj.weight",
        24,
        32,
        4,
        |name, output_width, input_len| {
            Ok(Some(ResidentMmapMatvecProjection::Q4(
                DenseQ4MmapMatvecProjection {
                    tensor_name: name.to_string(),
                    packed_byte_offset: 128,
                    scales_byte_offset: 256,
                    biases_byte_offset: 512,
                    rows: output_width,
                    cols: input_len + usize::from(name.ends_with("o_proj.weight")),
                    output_width,
                    row_packed_bytes: input_len.div_ceil(2),
                    groups_per_row: input_len.div_ceil(16),
                    group_size: 16,
                    scale_bias_dtype: "BF16".to_string(),
                },
            )))
        },
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("CMD2 resident post-attention output projection shape is invalid")
    );
}

#[test]
fn router_score_batch_keeps_projection_with_scores() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.3.mlp.gate.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![2, 4],
        byte_offset: 64,
        byte_len: 32,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    };
    let projection =
        RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 128, 2, 4).unwrap();

    let state = FlashMoeRoutingOutputState::cpu_router_scores(3, 2, 1);
    let batch = RouterScoreBatch::new(state, Some(projection), vec![1.0, -2.0]).unwrap();

    assert_eq!(batch.state(), state);
    assert_eq!(batch.scores, vec![1.0, -2.0]);
    assert_eq!(batch.projection.as_ref().unwrap().layer, 3);
    assert_eq!(batch.projection.as_ref().unwrap().experts, 2);
}

#[test]
fn router_score_batch_rejects_scores_outside_declared_state() {
    let err = RouterScoreBatch::new(
        FlashMoeRoutingOutputState::cpu_router_scores(3, 2, 1),
        None,
        vec![1.0],
    )
    .unwrap_err();

    assert!(
        err.to_string().contains("1 scores for 2 declared experts"),
        "{err:#}"
    );
}

#[test]
fn resident_static_tensor_descriptor_resolves_offsets_and_dtype() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.0.self_attn.conv1d.weight".to_string(),
        dtype: "BF16".to_string(),
        shape: vec![8],
        byte_offset: 16,
        byte_len: 16,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    };

    let resident = ResidentStaticTensorRef::from_entry(
        &entry.name,
        &entry,
        64,
        8,
        &[ResidentStaticDtype::Bf16],
    )
    .unwrap()
    .unwrap();

    assert_eq!(resident.tensor_name, entry.name);
    assert_eq!(resident.byte_offset, 16);
    assert_eq!(resident.dtype, ResidentStaticDtype::Bf16);
    assert_eq!(resident.values, 8);
    assert_eq!(resident.element_size, 2);
}

#[test]
fn resident_static_tensor_descriptor_rejects_wrong_layout_without_fallback() {
    let mut entry = RuntimeTensorEntry {
        name: "model.layers.0.self_attn.A_log".to_string(),
        dtype: "F32".to_string(),
        shape: vec![4],
        byte_offset: 4,
        byte_len: 16,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::None,
    };

    assert!(
        ResidentStaticTensorRef::from_entry(
            &entry.name,
            &entry,
            32,
            4,
            &[ResidentStaticDtype::F32],
        )
        .unwrap()
        .is_some()
    );

    entry.byte_len = 12;
    assert!(
        ResidentStaticTensorRef::from_entry(
            &entry.name,
            &entry,
            32,
            4,
            &[ResidentStaticDtype::F32],
        )
        .unwrap()
        .is_none()
    );

    entry.byte_len = 16;
    entry.quantization = TensorQuantization::Q4 {
        group_size: GROUP_SIZE,
        format: "mlx".to_string(),
        scale_bias_dtype: "F32".to_string(),
    };
    assert!(
        ResidentStaticTensorRef::from_entry(
            &entry.name,
            &entry,
            32,
            4,
            &[ResidentStaticDtype::F32],
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn linear_attention_weight_table_resolves_all_resident_dense_layouts_at_load() {
    fn append_tensor(
        bytes: &mut Vec<u8>,
        tensors: &mut Vec<DenseTensorRef>,
        name: String,
        dtype: &str,
        shape: Vec<usize>,
    ) {
        while !(bytes.len() as u64).is_multiple_of(TENSOR_ALIGNMENT) {
            bytes.push(0);
        }
        let runtime_offset = bytes.len() as u64;
        let element_size = dense_dtype_size(dtype).unwrap();
        let byte_len = shape.iter().product::<usize>() * element_size;
        bytes.resize(bytes.len() + byte_len, 0);
        tensors.push(DenseTensorRef {
            tensor: name,
            shard: "fixture.safetensors".to_string(),
            dtype: dtype.to_string(),
            shape,
            source_offsets: [0, byte_len as u64],
            runtime_offset,
            byte_len: byte_len as u64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        });
    }

    fn fixture(dtype: &str, omit_norm: bool) -> (tempfile::TempDir, DenseStore) {
        let hidden = 4;
        let experts = 3;
        let layout = LinearAttentionLayout {
            num_value_heads: 2,
            num_key_heads: 1,
            key_dim: 2,
            value_dim: 2,
            total_key_width: 2,
            total_value_width: 4,
            conv_dim: 8,
            conv_kernel_size: 2,
        };
        let mut bytes = Vec::new();
        let mut tensors = Vec::new();
        for request in linear_attention_input_projection_requests(
            0,
            layout.conv_dim,
            layout.total_value_width,
            layout.num_value_heads,
        )
        .unwrap()
        .requests()
        {
            append_tensor(
                &mut bytes,
                &mut tensors,
                request.tensor_name.to_string(),
                dtype,
                vec![request.output_width, hidden],
            );
        }
        append_tensor(
            &mut bytes,
            &mut tensors,
            linear_attention_tensor_name(0, "conv1d"),
            dtype,
            vec![layout.conv_dim, layout.conv_kernel_size],
        );
        append_tensor(
            &mut bytes,
            &mut tensors,
            linear_attention_scalar_tensor_name(0, "A_log"),
            "F32",
            vec![layout.num_value_heads],
        );
        append_tensor(
            &mut bytes,
            &mut tensors,
            linear_attention_scalar_tensor_name(0, "dt_bias"),
            dtype,
            vec![layout.num_value_heads],
        );
        if !omit_norm {
            append_tensor(
                &mut bytes,
                &mut tensors,
                linear_attention_tensor_name(0, "norm"),
                dtype,
                vec![layout.value_dim],
            );
        }
        append_tensor(
            &mut bytes,
            &mut tensors,
            linear_attention_tensor_name(0, "out_proj"),
            dtype,
            vec![hidden, layout.total_value_width],
        );
        append_tensor(
            &mut bytes,
            &mut tensors,
            router_tensor_name(0),
            dtype,
            vec![experts, hidden],
        );

        let temp = tempfile::tempdir().unwrap();
        let dense_path = temp.path().join("non-expert.bin");
        let manifest_path = temp.path().join("manifest.json");
        fs::write(&dense_path, bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: "fixture".to_string(),
                cache_version: "test".to_string(),
                dense_shards: vec!["fixture.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: tensors,
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        (temp, store)
    }

    let layout = LinearAttentionLayout {
        num_value_heads: 2,
        num_key_heads: 1,
        key_dim: 2,
        value_dim: 2,
        total_key_width: 2,
        total_value_width: 4,
        conv_dim: 8,
        conv_kernel_size: 2,
    };
    for (dtype, expected_static_dtype) in [
        ("BF16", ResidentStaticDtype::Bf16),
        ("F16", ResidentStaticDtype::F16),
        ("F32", ResidentStaticDtype::F32),
    ] {
        let (_temp, store) = fixture(dtype, false);
        let table = store
            .resolve_linear_attention_weight_table(&[Some(layout)], 4, 3)
            .unwrap();
        let bindings = table.layer(0).unwrap();
        assert_eq!(bindings.layer, 0);
        assert_eq!(bindings.input_projections.len(), 4);
        assert_eq!(
            bindings.static_tensors.conv_weight.dtype,
            expected_static_dtype
        );
        assert_eq!(bindings.static_tensors.dt_bias.dtype, expected_static_dtype);
        assert_eq!(
            bindings.static_tensors.norm_weight.dtype,
            expected_static_dtype
        );
        assert_eq!(
            bindings.static_tensors.a_log.dtype,
            ResidentStaticDtype::F32
        );
        assert_eq!(bindings.out_proj.rows(), 4);
        assert_eq!(bindings.router.rows(), 3);
    }

    let (_temp, store) = fixture("BF16", true);
    let error = store
        .resolve_linear_attention_weight_table(&[Some(layout)], 4, 3)
        .unwrap_err();
    assert!(
        error.to_string().contains("static-weight path"),
        "{error:#}"
    );
    assert!(error.to_string().contains("linear_attn.norm"), "{error:#}");
}

#[test]
fn shared_expert_weight_table_resolves_all_resident_dense_layouts_at_load() {
    fn append_tensor(
        bytes: &mut Vec<u8>,
        tensors: &mut Vec<DenseTensorRef>,
        name: String,
        dtype: &str,
        shape: Vec<usize>,
    ) {
        while !(bytes.len() as u64).is_multiple_of(TENSOR_ALIGNMENT) {
            bytes.push(0);
        }
        let runtime_offset = bytes.len() as u64;
        let byte_len = shape.iter().product::<usize>() * dense_dtype_size(dtype).unwrap();
        bytes.resize(bytes.len() + byte_len, 0);
        tensors.push(DenseTensorRef {
            tensor: name,
            shard: "fixture.safetensors".to_string(),
            dtype: dtype.to_string(),
            shape,
            source_offsets: [0, byte_len as u64],
            runtime_offset,
            byte_len: byte_len as u64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        });
    }

    fn fixture(dtype: &str, omit_down_layer: Option<usize>) -> (tempfile::TempDir, DenseStore) {
        let width = 4;
        let shared_experts = 2;
        let intermediate = 3;
        let total_intermediate = shared_experts * intermediate;
        let mut bytes = Vec::new();
        let mut tensors = Vec::new();
        for layer in 0..2 {
            for projection in ["gate_proj", "up_proj"] {
                append_tensor(
                    &mut bytes,
                    &mut tensors,
                    shared_expert_tensor_name(layer, projection),
                    dtype,
                    vec![total_intermediate, width],
                );
            }
            if omit_down_layer != Some(layer) {
                append_tensor(
                    &mut bytes,
                    &mut tensors,
                    shared_expert_tensor_name(layer, "down_proj"),
                    dtype,
                    vec![width, total_intermediate],
                );
            }
            append_tensor(
                &mut bytes,
                &mut tensors,
                shared_expert_gate_tensor_name(layer),
                dtype,
                vec![shared_experts, width],
            );
        }

        let temp = tempfile::tempdir().unwrap();
        let dense_path = temp.path().join("non-expert.bin");
        let manifest_path = temp.path().join("manifest.json");
        fs::write(&dense_path, bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: "fixture".to_string(),
                cache_version: "test".to_string(),
                dense_shards: vec!["fixture.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: tensors,
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        (temp, store)
    }

    for dtype in ["BF16", "F16", "F32"] {
        let (_temp, store) = fixture(dtype, None);
        let table = store
            .resolve_shared_expert_weight_table(2, 4, 2, 3)
            .unwrap();
        for layer in 0..2 {
            let SharedExpertLayerWeights::Resident(shared) = table.layer(layer).unwrap() else {
                panic!("configured shared experts must resolve resident bindings");
            };
            for projection in [&shared.gate, &shared.up, &shared.down] {
                let ResidentMmapMatvecProjection::Dense(projection) = projection else {
                    panic!("{dtype} fixture resolved a Q4 projection");
                };
                assert_eq!(projection.dtype.as_str(), dtype);
            }
            let ResidentMmapMatvecProjection::Dense(router) = shared.router.as_ref().unwrap()
            else {
                panic!("{dtype} fixture resolved a Q4 router projection");
            };
            assert_eq!(router.dtype.as_str(), dtype);
            assert_eq!(
                shared.validated_shape().unwrap(),
                SharedExpertPhaseShape::new(4, 2, 3).unwrap()
            );
        }

        let disabled = store
            .resolve_shared_expert_weight_table(2, 4, 0, 0)
            .unwrap();
        assert!(matches!(
            disabled.layer(0).unwrap(),
            SharedExpertLayerWeights::None
        ));
    }

    let (_temp, store) = fixture("BF16", Some(1));
    let error = store
        .resolve_shared_expert_weight_table(2, 4, 2, 3)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("missing resident shared down projection"),
        "{error:#}"
    );
    assert!(
        error
            .to_string()
            .contains("model.layers.1.mlp.shared_expert.down_proj.weight"),
        "{error:#}"
    );
}

#[test]
fn dense_q4_projection_descriptor_carries_one_binding_shape() {
    let projection = DenseQ4MmapMatvecProjection {
        tensor_name: "model.layers.0.mlp.gate_proj.weight".to_string(),
        packed_byte_offset: 128,
        scales_byte_offset: 256,
        biases_byte_offset: 512,
        rows: 16,
        cols: 32,
        output_width: 16,
        row_packed_bytes: 16,
        groups_per_row: 2,
        group_size: 16,
        scale_bias_dtype: "BF16".to_string(),
    };

    assert_eq!(projection.row_packed_bytes, projection.cols.div_ceil(2));
    assert_eq!(projection.groups_per_row, 2);
    assert_eq!(projection.output_width, projection.rows);
}

#[test]
fn dense_q4_projection_key_names_cached_binding_shape() {
    let key = DenseQ4ProjectionKey::new("model.layers.0.mlp.gate_proj.weight", 16, 32);

    assert_eq!(key.name, "model.layers.0.mlp.gate_proj.weight");
    assert_eq!(key.output_width, 16);
    assert_eq!(key.input_len, 32);
}

#[test]
fn dense_q4_projection_builder_uses_lookup_callback() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.0.mlp.gate_proj.weight".to_string(),
        dtype: "Q4".to_string(),
        shape: vec![2, 4],
        byte_offset: 128,
        byte_len: 12,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::Q4 {
            group_size: 16,
            format: "dense-q4".to_string(),
            scale_bias_dtype: "BF16".to_string(),
        },
    };
    let mut seen_name = None;

    let projection =
        build_dense_q4_mmap_projection("model.layers.0.mlp.gate_proj.weight", 2, 4, 256, |name| {
            seen_name = Some(name.to_string());
            (name == entry.name).then_some(&entry)
        })
        .unwrap()
        .unwrap();

    assert_eq!(seen_name.unwrap(), entry.name);
    assert_eq!(projection.packed_byte_offset, 128);
    assert_eq!(projection.output_width, 2);
    assert_eq!(projection.cols, 4);

    let missing = build_dense_q4_mmap_projection("missing.weight", 2, 4, 256, |_| None).unwrap();
    assert!(missing.is_none());
}

#[test]
fn shared_expert_dense_descriptor_groups_projection_weights() {
    let shared = SharedExpertPhaseWeights::new(
        Arc::new(vec![1.0, 2.0]),
        Arc::new(vec![3.0, 4.0]),
        Arc::new(vec![5.0, 6.0]),
        Arc::new(vec![7.0]),
        1,
        2,
        1,
    )
    .unwrap();

    assert_eq!(shared.shared_experts, 1);
    assert_eq!(shared.intermediate, 2);
    assert_eq!(shared.width, 1);
    assert_eq!(shared.gate.as_slice(), &[1.0, 2.0]);
    assert_eq!(shared.router.as_slice(), &[7.0]);
    assert_eq!(
        shared.validated_shape().unwrap(),
        SharedExpertPhaseShape::new(1, 1, 2).unwrap()
    );
}

#[test]
fn shared_expert_weight_builder_loads_named_dense_tensors() {
    let mut tensors = BTreeMap::<String, Arc<Vec<f32>>>::new();
    tensors.insert(
        shared_expert_tensor_name(3, "gate_proj"),
        Arc::new(vec![1.0, 2.0, 3.0, 4.0]),
    );
    tensors.insert(
        shared_expert_tensor_name(3, "up_proj"),
        Arc::new(vec![5.0, 6.0, 7.0, 8.0]),
    );
    tensors.insert(
        shared_expert_tensor_name(3, "down_proj"),
        Arc::new(vec![9.0, 10.0, 11.0, 12.0]),
    );
    tensors.insert(
        shared_expert_gate_tensor_name(3),
        Arc::new(vec![13.0, 14.0]),
    );

    let shared =
        build_shared_expert_phase_weights(3, 2, 1, 2, |name| Ok(tensors.get(name).cloned()))
            .unwrap()
            .unwrap();

    assert_eq!(shared.width, 2);
    assert_eq!(shared.shared_experts, 1);
    assert_eq!(shared.intermediate, 2);
    assert_eq!(shared.gate.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(shared.router.as_slice(), &[13.0, 14.0]);
}

#[test]
fn shared_expert_weight_builder_skips_disabled_shared_experts() {
    let none = build_shared_expert_phase_weights(3, 2, 0, 2, |_| {
        panic!("disabled shared experts must not request tensors")
    })
    .unwrap();
    assert!(none.is_none());

    let none = build_shared_expert_phase_weights(3, 2, 1, 0, |_| {
        panic!("zero intermediate shared experts must not request tensors")
    })
    .unwrap();
    assert!(none.is_none());
}

#[test]
fn shared_expert_phase_cache_reuses_weight_owned_dense_phase() {
    let cache = SharedExpertPhaseCache::default();
    let mut tensors = BTreeMap::<String, Arc<Vec<f32>>>::new();
    tensors.insert(
        shared_expert_tensor_name(3, "gate_proj"),
        Arc::new(vec![1.0, 2.0, 3.0, 4.0]),
    );
    tensors.insert(
        shared_expert_tensor_name(3, "up_proj"),
        Arc::new(vec![5.0, 6.0, 7.0, 8.0]),
    );
    tensors.insert(
        shared_expert_tensor_name(3, "down_proj"),
        Arc::new(vec![9.0, 10.0, 11.0, 12.0]),
    );
    tensors.insert(
        shared_expert_gate_tensor_name(3),
        Arc::new(vec![13.0, 14.0]),
    );
    let mut lookup_count = 0usize;

    let first = cache
        .dense(3, 2, 1, 2, |name| {
            lookup_count += 1;
            Ok(tensors.get(name).cloned())
        })
        .unwrap()
        .unwrap();
    let second = cache
        .dense(3, 2, 1, 2, |_| {
            panic!("cached shared expert phase must not reload tensors")
        })
        .unwrap()
        .unwrap();

    assert_eq!(lookup_count, 4);
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn shared_expert_phase_cache_skips_disabled_shared_experts() {
    let cache = SharedExpertPhaseCache::default();

    let none = cache
        .dense(3, 2, 0, 2, |_| {
            panic!("disabled shared experts must not request tensors")
        })
        .unwrap();

    assert!(none.is_none());
}

#[test]
fn shared_expert_phase_cache_rejects_width_mismatch_without_reload() {
    let cache = SharedExpertPhaseCache::default();
    let mut tensors = BTreeMap::<String, Arc<Vec<f32>>>::new();
    tensors.insert(
        shared_expert_tensor_name(3, "gate_proj"),
        Arc::new(vec![1.0, 2.0, 3.0, 4.0]),
    );
    tensors.insert(
        shared_expert_tensor_name(3, "up_proj"),
        Arc::new(vec![5.0, 6.0, 7.0, 8.0]),
    );
    tensors.insert(
        shared_expert_tensor_name(3, "down_proj"),
        Arc::new(vec![9.0, 10.0, 11.0, 12.0]),
    );
    tensors.insert(
        shared_expert_gate_tensor_name(3),
        Arc::new(vec![13.0, 14.0]),
    );

    cache
        .dense(3, 2, 1, 2, |name| Ok(tensors.get(name).cloned()))
        .unwrap()
        .unwrap();
    let err = cache
        .dense(3, 4, 1, 2, |_| {
            panic!("mismatched cached shared expert phase must fail before reload")
        })
        .unwrap_err();

    assert!(
        err.to_string()
            .contains("cached shared expert tensors for layer 3 have width 2, requested 4")
    );
}

#[test]
fn shared_expert_weight_builder_rejects_missing_and_mismatched_tensors() {
    let missing = build_shared_expert_phase_weights(3, 2, 1, 2, |_| Ok(None)).unwrap_err();
    assert!(missing.to_string().contains(
        "missing configured shared expert tensor model.layers.3.mlp.shared_expert.gate_proj.weight"
    ));

    let mut tensors = BTreeMap::<String, Arc<Vec<f32>>>::new();
    tensors.insert(
        shared_expert_tensor_name(3, "gate_proj"),
        Arc::new(vec![1.0]),
    );
    tensors.insert(
        shared_expert_tensor_name(3, "up_proj"),
        Arc::new(vec![2.0; 4]),
    );
    tensors.insert(
        shared_expert_tensor_name(3, "down_proj"),
        Arc::new(vec![3.0; 4]),
    );
    tensors.insert(shared_expert_gate_tensor_name(3), Arc::new(vec![4.0; 2]));

    let mismatch =
        build_shared_expert_phase_weights(3, 2, 1, 2, |name| Ok(tensors.get(name).cloned()))
            .unwrap_err();
    assert!(mismatch.to_string().contains("shape is invalid"));
}

#[test]
fn shared_expert_resident_descriptor_groups_projection_bindings() {
    let gate = DenseQ4MmapMatvecProjection {
        tensor_name: "model.layers.0.mlp.shared_expert.gate_proj.weight".to_string(),
        packed_byte_offset: 128,
        scales_byte_offset: 256,
        biases_byte_offset: 512,
        rows: 16,
        cols: 32,
        output_width: 16,
        row_packed_bytes: 16,
        groups_per_row: 2,
        group_size: 16,
        scale_bias_dtype: "BF16".to_string(),
    };
    let up = DenseQ4MmapMatvecProjection {
        tensor_name: "model.layers.0.mlp.shared_expert.up_proj.weight".to_string(),
        ..gate.clone()
    };
    let down = DenseQ4MmapMatvecProjection {
        tensor_name: "model.layers.0.mlp.shared_expert.down_proj.weight".to_string(),
        rows: 32,
        cols: 16,
        output_width: 32,
        row_packed_bytes: 8,
        groups_per_row: 1,
        ..gate.clone()
    };
    let router = DenseQ4MmapMatvecProjection {
        tensor_name: "model.layers.0.mlp.shared_expert_gate.weight".to_string(),
        rows: 1,
        output_width: 1,
        ..gate.clone()
    };
    let shared = SharedExpertPhaseResidentProjections {
        gate: gate.into(),
        up: up.into(),
        down: down.into(),
        router: Some(router.into()),
        shared_experts: 1,
        intermediate: 16,
        width: 32,
    };

    assert_eq!(shared.gate.q4().unwrap().packed_byte_offset, 128);
    assert_eq!(shared.down.output_width(), 32);
    assert_eq!(shared.router.as_ref().unwrap().cols(), 32);
    assert_eq!(shared.shared_experts, 1);
    assert_eq!(shared.intermediate, 16);
    assert_eq!(shared.width, 32);
    assert_eq!(
        shared.validated_shape().unwrap(),
        SharedExpertPhaseShape::new(32, 1, 16).unwrap()
    );
}

#[test]
fn shared_expert_resident_builder_resolves_named_projection_bindings() {
    let shared = build_shared_expert_resident_phase_projections(
        4,
        32,
        2,
        16,
        |name, output_width, input_len| {
            Ok(Some(DenseQ4MmapMatvecProjection {
                tensor_name: name.to_string(),
                packed_byte_offset: 128,
                scales_byte_offset: 256,
                biases_byte_offset: 512,
                rows: output_width,
                cols: input_len,
                output_width,
                row_packed_bytes: input_len.div_ceil(2),
                groups_per_row: input_len.div_ceil(16),
                group_size: 16,
                scale_bias_dtype: "BF16".to_string(),
            }))
        },
    )
    .unwrap()
    .unwrap();

    assert_eq!(
        shared.gate.tensor_name(),
        "model.layers.4.mlp.shared_expert.gate_proj.weight"
    );
    assert_eq!(shared.gate.output_width(), 32);
    assert_eq!(shared.gate.cols(), 32);
    assert_eq!(shared.down.output_width(), 32);
    assert_eq!(shared.down.cols(), 32);
    assert_eq!(shared.router.as_ref().unwrap().output_width(), 2);
    assert_eq!(shared.router.as_ref().unwrap().cols(), 32);
    assert_eq!(
        shared.validated_shape().unwrap(),
        SharedExpertPhaseShape::new(32, 2, 16).unwrap()
    );
}

#[test]
fn shared_expert_resident_builder_skips_disabled_or_partial_bindings() {
    let disabled = build_shared_expert_resident_phase_projections(
        4,
        32,
        0,
        16,
        |_, _, _| -> Result<Option<DenseQ4MmapMatvecProjection>> {
            panic!("disabled shared experts must not request projections")
        },
    )
    .unwrap();
    assert!(disabled.is_none());

    let partial = build_shared_expert_resident_phase_projections(
        4,
        32,
        2,
        16,
        |name, output_width, input_len| {
            if name.ends_with("up_proj.weight") {
                return Ok(None);
            }
            Ok(Some(DenseQ4MmapMatvecProjection {
                tensor_name: name.to_string(),
                packed_byte_offset: 128,
                scales_byte_offset: 256,
                biases_byte_offset: 512,
                rows: output_width,
                cols: input_len,
                output_width,
                row_packed_bytes: input_len.div_ceil(2),
                groups_per_row: input_len.div_ceil(16),
                group_size: 16,
                scale_bias_dtype: "BF16".to_string(),
            }))
        },
    )
    .unwrap();
    assert!(partial.is_none());
}

#[test]
fn required_shared_expert_resident_builder_errors_on_missing_configured_binding() {
    let disabled = build_required_shared_expert_resident_phase_projections(
        4,
        32,
        0,
        16,
        |_, _, _| -> Result<Option<DenseQ4MmapMatvecProjection>> {
            panic!("disabled shared experts must not request projections")
        },
    )
    .unwrap();
    assert!(disabled.is_none());

    let invalid = build_required_shared_expert_resident_phase_projections(
        4,
        32,
        2,
        0,
        |_, _, _| -> Result<Option<DenseQ4MmapMatvecProjection>> {
            panic!("invalid shared-expert shape must fail before requesting projections")
        },
    )
    .unwrap_err();
    assert!(
        invalid.to_string().contains("requires non-zero width"),
        "{invalid:#}"
    );

    let missing = build_required_shared_expert_resident_phase_projections(
        4,
        32,
        2,
        16,
        |name, output_width, input_len| {
            if name.ends_with("down_proj.weight") {
                return Ok(None);
            }
            Ok(Some(DenseQ4MmapMatvecProjection {
                tensor_name: name.to_string(),
                packed_byte_offset: 128,
                scales_byte_offset: 256,
                biases_byte_offset: 512,
                rows: output_width,
                cols: input_len,
                output_width,
                row_packed_bytes: input_len.div_ceil(2),
                groups_per_row: input_len.div_ceil(16),
                group_size: 16,
                scale_bias_dtype: "BF16".to_string(),
            }))
        },
    )
    .unwrap_err();
    assert!(
        missing
            .to_string()
            .contains("missing resident shared down projection"),
        "{missing:#}"
    );
}

#[test]
fn shared_expert_resident_builder_rejects_mismatched_projection_shape() {
    let err = build_shared_expert_resident_phase_projections(
        4,
        32,
        2,
        16,
        |name, output_width, input_len| {
            Ok(Some(DenseQ4MmapMatvecProjection {
                tensor_name: name.to_string(),
                packed_byte_offset: 128,
                scales_byte_offset: 256,
                biases_byte_offset: 512,
                rows: output_width,
                cols: input_len + usize::from(name.ends_with("gate_proj.weight")),
                output_width,
                row_packed_bytes: input_len.div_ceil(2),
                groups_per_row: input_len.div_ceil(16),
                group_size: 16,
                scale_bias_dtype: "BF16".to_string(),
            }))
        },
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("resident shared-expert shape is invalid")
    );
}

#[test]
fn shared_expert_descriptors_reject_mismatched_graph_shape() {
    let shared = SharedExpertPhaseWeights {
        gate: Arc::new(vec![1.0, 2.0]),
        up: Arc::new(vec![3.0, 4.0]),
        down: Arc::new(vec![5.0, 6.0]),
        router: Arc::new(vec![7.0]),
        shared_experts: 1,
        intermediate: 2,
        width: 2,
    };

    let err = shared.validated_shape().unwrap_err();
    assert!(err.to_string().contains("shape is invalid"));
}

#[test]
fn scheduled_next_norm_weights_declare_cpu_visible_width() {
    let values = [1.0, 0.5, 0.25, 0.125];
    let weights =
        ScheduledNextNormWeights::cpu_visible("model.layers.1.input_layernorm.weight", &values, 3)
            .unwrap();

    assert!(weights.is_cpu_visible());
    assert_eq!(weights.width(), Some(3));
    assert_eq!(weights.values().unwrap(), &[1.0, 0.5, 0.25]);
    assert!(ScheduledNextNormWeights::none().is_none());

    let empty_name_err = ScheduledNextNormWeights::cpu_visible("", &values, 3).unwrap_err();
    assert!(empty_name_err.to_string().contains("require a tensor name"));

    let short_err = ScheduledNextNormWeights::cpu_visible(
        "model.layers.1.input_layernorm.weight",
        &values[..2],
        3,
    )
    .unwrap_err();
    assert!(short_err.to_string().contains("smaller than width 3"));
}

#[test]
fn prepared_next_norm_weights_resolve_declared_cmd3_descriptor() {
    let prepared = PreparedScheduledNextNormWeights::cpu_visible(
        "model.layers.1.input_layernorm.weight".to_string(),
        vec![1.0, 0.5, 0.25, 0.125],
        3,
    )
    .unwrap();
    let scheduled = prepared.scheduled().unwrap();

    assert!(scheduled.is_cpu_visible());
    assert_eq!(scheduled.width(), Some(3));
    assert_eq!(scheduled.values().unwrap(), &[1.0, 0.5, 0.25]);
}

#[test]
fn prepare_next_norm_weights_declares_only_non_terminal_cmd3_layers() {
    let prepared = prepare_scheduled_next_norm_weights(0, 2, 4, true, |name, width| {
        assert_eq!(name, "model.layers.1.input_layernorm.weight");
        assert_eq!(width, 4);
        Ok(Some(vec![1.0, 1.1, 1.2, 1.3]))
    })
    .unwrap();
    assert!(prepared.scheduled().unwrap().is_cpu_visible());

    let terminal = prepare_scheduled_next_norm_weights(1, 2, 4, true, |_, _| {
        panic!("terminal layer must not request next-layer norm weights")
    })
    .unwrap();
    assert!(terminal.scheduled().unwrap().is_none());

    let disabled = prepare_scheduled_next_norm_weights(0, 2, 4, false, |_, _| {
        panic!("disabled next-layer norm must not request weights")
    })
    .unwrap();
    assert!(disabled.scheduled().unwrap().is_none());
}

#[test]
fn prepare_next_norm_weights_reports_missing_scheduled_cmd3_weight() {
    let err = prepare_scheduled_next_norm_weights(2, 4, 8, true, |_, _| Ok(None)).unwrap_err();

    assert!(err.to_string().contains(
        "missing next-layer norm weight model.layers.3.input_layernorm.weight for layer 2"
    ));
}

#[test]
fn qwen_moe_weight_tensor_names_are_canonical_hf_paths() {
    assert_eq!(
        layer_norm_tensor_name(7, "post_attention_layernorm"),
        "model.layers.7.post_attention_layernorm.weight"
    );
    assert_eq!(
        attention_tensor_name(7, "q_proj"),
        "model.layers.7.self_attn.q_proj.weight"
    );
    assert_eq!(
        attention_tensor_name(7, "o_proj"),
        "model.layers.7.self_attn.o_proj.weight"
    );
    assert_eq!(router_tensor_name(7), "model.layers.7.mlp.gate.weight");
    assert_eq!(
        shared_expert_tensor_name(7, "gate_proj"),
        "model.layers.7.mlp.shared_expert.gate_proj.weight"
    );
    assert_eq!(
        shared_expert_gate_tensor_name(7),
        "model.layers.7.mlp.shared_expert_gate.weight"
    );
}

#[test]
fn linear_attention_weight_tensor_names_are_canonical_hf_paths() {
    assert_eq!(
        linear_attention_tensor_name(7, "in_proj_qkv"),
        "model.layers.7.linear_attn.in_proj_qkv.weight"
    );
    assert_eq!(
        linear_attention_tensor_name(7, "out_proj"),
        "model.layers.7.linear_attn.out_proj.weight"
    );
    assert_eq!(
        linear_attention_scalar_tensor_name(7, "A_log"),
        "model.layers.7.linear_attn.A_log"
    );
}

#[test]
fn dense_projection_request_requires_named_nonzero_output() {
    let request =
        DenseProjectionRequest::new("model.layers.7.linear_attn.in_proj_qkv.weight", 128).unwrap();

    assert_eq!(
        request.tensor_name,
        "model.layers.7.linear_attn.in_proj_qkv.weight"
    );
    assert_eq!(request.output_width, 128);

    let missing_name = DenseProjectionRequest::new("", 128).unwrap_err();
    assert!(missing_name.to_string().contains("requires a tensor name"));

    let zero_width =
        DenseProjectionRequest::new("model.layers.7.linear_attn.in_proj_qkv.weight", 0)
            .unwrap_err();
    assert!(zero_width.to_string().contains("non-zero output width"));
}

#[test]
fn full_attention_projection_requests_use_canonical_self_attention_names() {
    let requests = full_attention_input_projection_requests(3, 24, 8).unwrap();
    let specs = requests.requests();

    assert_eq!(
        specs
            .iter()
            .map(|spec| (spec.tensor_name, spec.output_width))
            .collect::<Vec<_>>(),
        vec![
            ("model.layers.3.self_attn.q_proj.weight", 24),
            ("model.layers.3.self_attn.k_proj.weight", 8),
            ("model.layers.3.self_attn.v_proj.weight", 8),
        ]
    );
    assert_eq!(
        requests.tensor_name(0),
        "model.layers.3.self_attn.q_proj.weight"
    );
}

#[test]
fn linear_attention_projection_requests_use_canonical_gated_delta_names() {
    let requests = linear_attention_input_projection_requests(5, 16, 32, 4).unwrap();
    let specs = requests.requests();

    assert_eq!(
        specs
            .iter()
            .map(|spec| (spec.tensor_name, spec.output_width))
            .collect::<Vec<_>>(),
        vec![
            ("model.layers.5.linear_attn.in_proj_qkv.weight", 16),
            ("model.layers.5.linear_attn.in_proj_z.weight", 32),
            ("model.layers.5.linear_attn.in_proj_b.weight", 4),
            ("model.layers.5.linear_attn.in_proj_a.weight", 4),
        ]
    );
    assert_eq!(
        requests.tensor_name(3),
        "model.layers.5.linear_attn.in_proj_a.weight"
    );
}

#[test]
fn projection_request_groups_reject_zero_width_without_fallback() {
    let full = full_attention_input_projection_requests(3, 0, 8).unwrap_err();
    assert!(full.to_string().contains("non-zero output width"));

    let linear = linear_attention_input_projection_requests(5, 16, 32, 0).unwrap_err();
    assert!(linear.to_string().contains("non-zero output width"));
}

#[test]
fn qwen_norm_offset_policy_matches_declared_semantics_and_reference_names() {
    for name in [
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.3.self_attn.q_norm.weight",
        "model.layers.3.self_attn.k_norm.weight",
    ] {
        assert!(
            qwen_norm_uses_offset(QwenNormWeightSemantics::Offset, name),
            "{name} should use Qwen3Next 1+weight RMSNorm semantics"
        );
    }

    for name in [
        "model.layers.0.linear_attn.norm.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
    ] {
        assert!(
            !qwen_norm_uses_offset(QwenNormWeightSemantics::Offset, name),
            "{name} is not a plain Qwen3NextRMSNorm weight"
        );
    }

    assert!(!qwen_norm_uses_offset(
        QwenNormWeightSemantics::Multiplicative,
        "model.norm.weight"
    ));
}

#[test]
fn qwen_norm_semantics_are_resolved_without_value_probing() {
    let mut offset = vec![0.6679, 0.7187, 0.7265, 0.7031];
    apply_qwen_norm_weight_semantics(
        QwenNormWeightSemantics::Offset,
        "model.layers.0.input_layernorm.weight",
        &mut offset,
    );
    for (actual, expected) in offset.iter().zip([1.6679, 1.7187, 1.7265, 1.7031]) {
        assert!((actual - expected).abs() < 1e-5);
    }

    let mut disabled = vec![0.6679, 0.7187, 0.7265, 0.7031];
    apply_qwen_norm_weight_semantics(
        QwenNormWeightSemantics::Multiplicative,
        "model.norm.weight",
        &mut disabled,
    );
    assert_eq!(disabled, vec![0.6679, 0.7187, 0.7265, 0.7031]);
}

#[test]
fn dense_q4_layout_accounts_for_scale_bias_dtype() {
    let layout = dense_q4_layout_with_scale_bias_dtype(&[2, 4], 16, "BF16").unwrap();

    assert_eq!(layout.rows, 2);
    assert_eq!(layout.cols, 4);
    assert_eq!(layout.row_packed_bytes, 2);
    assert_eq!(layout.groups_per_row, 1);
    assert_eq!(layout.packed_bytes, 4);
    assert_eq!(layout.scales_bytes, 4);
    assert_eq!(layout.scale_bias_bytes, 2);
    assert_eq!(layout.total_bytes, 12);
}

#[test]
fn dense_q4_projection_descriptor_resolves_offsets_from_entry() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.0.mlp.gate_proj.weight".to_string(),
        dtype: "Q4".to_string(),
        shape: vec![2, 4],
        byte_offset: 128,
        byte_len: 12,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::Q4 {
            group_size: 16,
            format: "dense-q4".to_string(),
            scale_bias_dtype: "BF16".to_string(),
        },
    };

    let projection = DenseQ4MmapMatvecProjection::from_entry(
        "model.layers.0.mlp.gate_proj.weight",
        &entry,
        256,
        2,
        4,
    )
    .unwrap()
    .unwrap();

    assert_eq!(projection.packed_byte_offset, 128);
    assert_eq!(projection.scales_byte_offset, 132);
    assert_eq!(projection.biases_byte_offset, 136);
    assert_eq!(projection.rows, 2);
    assert_eq!(projection.cols, 4);
    assert_eq!(projection.scale_bias_dtype, "BF16");
}

#[test]
fn dense_q4_projection_descriptor_rejects_missing_capacity() {
    let entry = RuntimeTensorEntry {
        name: "model.layers.0.mlp.gate_proj.weight".to_string(),
        dtype: "Q4".to_string(),
        shape: vec![2, 4],
        byte_offset: 128,
        byte_len: 12,
        alignment: TENSOR_ALIGNMENT,
        quantization: TensorQuantization::Q4 {
            group_size: 16,
            format: "dense-q4".to_string(),
            scale_bias_dtype: "BF16".to_string(),
        },
    };

    let projection = DenseQ4MmapMatvecProjection::from_entry(
        "model.layers.0.mlp.gate_proj.weight",
        &entry,
        139,
        2,
        4,
    )
    .unwrap();

    assert_eq!(projection, None);
}
