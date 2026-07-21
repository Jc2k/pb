use super::super::model_family::QwenModelConfig;
use super::*;

fn packing_layout_config() -> QwenModelConfig {
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
        torch_dtype: Some("bfloat16".to_string()),
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

#[test]
fn expert_packing_policy_resolves_storage_from_declared_variant() {
    let model = "hf://Qwen/Qwen3-30B-A3B";
    let experts_dir = Path::new("unused-experts");
    let config = packing_layout_config();

    let q4 = ExpertPackingPolicy::new(model, experts_dir, ExpertQuantization::FourBitProduction);
    assert_eq!(
        fixed_dense_expert_slot_spec_for_pack(q4, None).unwrap(),
        None
    );

    for (quantization, expected_dtype) in [
        (ExpertQuantization::Bf16, DenseExpertDtype::Bf16),
        (ExpertQuantization::F16, DenseExpertDtype::F16),
    ] {
        let policy = ExpertPackingPolicy::new(model, experts_dir, quantization);
        let spec = fixed_dense_expert_slot_spec_for_pack(policy, Some(&config))
            .unwrap()
            .unwrap();
        assert_eq!(spec.dtype, expected_dtype);
        assert_eq!(spec.hidden_size, config.hidden_size);
        assert_eq!(
            spec.intermediate_size,
            config.moe_intermediate_size.unwrap()
        );
    }

    let missing_config = ExpertPackingPolicy::new(model, experts_dir, ExpertQuantization::Bf16);
    assert_eq!(
        fixed_dense_expert_slot_spec_for_pack(missing_config, None)
            .unwrap_err()
            .to_string(),
        "Qwen config is required for fixed dense expert packing"
    );
}

#[derive(Debug)]
struct TestAggregateTensor {
    name: &'static str,
    shape: Vec<usize>,
    native_q4: bool,
}

impl AggregateExpertTensor for TestAggregateTensor {
    fn aggregate_tensor_name(&self) -> &str {
        self.name
    }

    fn aggregate_tensor_shape(&self) -> &[usize] {
        &self.shape
    }

    fn aggregate_tensor_has_native_q4(&self) -> bool {
        self.native_q4
    }
}

impl ExpertSourceTensor for TestAggregateTensor {
    fn expert_source_offsets(&self) -> Option<[u64; 2]> {
        Some([100, 200])
    }
}

#[test]
fn aggregate_expert_tensor_kind_classifies_qwen_and_mlx_names() {
    assert_eq!(
        aggregate_expert_tensor_kind("model.layers.0.mlp.experts.gate_up_proj"),
        Some(AggregateExpertTensorKind::GateUp)
    );
    assert_eq!(
        aggregate_expert_tensor_kind("model.layers.0.mlp.switch_mlp.gate_proj.weight"),
        Some(AggregateExpertTensorKind::Gate)
    );
    assert_eq!(
        aggregate_expert_tensor_kind("model.layers.0.mlp.switch_mlp.up_proj.weight"),
        Some(AggregateExpertTensorKind::Up)
    );
    assert_eq!(
        aggregate_expert_tensor_kind("model.layers.0.mlp.experts.down_proj.weight"),
        Some(AggregateExpertTensorKind::Down)
    );
    assert_eq!(
        aggregate_expert_tensor_kind("model.layers.0.self_attn.q_proj"),
        None
    );
}

#[test]
fn aggregate_expert_layout_computes_split_ranges_from_model_shape() {
    let layout = AggregateExpertLayout::new(64, 3584, 1024).unwrap();

    assert_eq!(layout.experts, 64);
    assert_eq!(layout.hidden, 3584);
    assert_eq!(layout.intermediate, 1024);
    assert_eq!(layout.single_projection_values, 1024 * 3584);
    assert_eq!(layout.gate_up_expert_values, 2 * 1024 * 3584);
    assert_eq!(layout.down_expert_values, 3584 * 1024);
}

#[test]
fn aggregate_expert_tensors_plan_combined_gate_up_slices() {
    let gate_up = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.gate_up_proj.weight",
        shape: vec![2, 4, 3],
        native_q4: false,
    };
    let down = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.down_proj.weight",
        shape: vec![2, 3, 2],
        native_q4: false,
    };
    let layout = AggregateExpertLayout::new(2, 3, 2).unwrap();
    let tensors = aggregate_expert_tensors(&[&gate_up, &down], 0, layout).unwrap();

    assert_eq!(tensors.gate.start(1).unwrap(), 12);
    assert_eq!(tensors.up.start(1).unwrap(), 18);
    assert_eq!(
        single_aggregate_expert_tensor(&[&gate_up, &down], AggregateExpertTensorKind::Down, 0)
            .unwrap()
            .name,
        down.name
    );
}

#[test]
fn aggregate_expert_tensors_plan_split_switch_mlp_slices() {
    let gate = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.gate_proj.weight",
        shape: vec![2, 2, 3],
        native_q4: true,
    };
    let up = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.up_proj.weight",
        shape: vec![2, 2, 3],
        native_q4: true,
    };
    let down = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.down_proj.weight",
        shape: vec![2, 3, 2],
        native_q4: true,
    };
    let layout = AggregateExpertLayout::new(2, 3, 2).unwrap();
    let tensors = aggregate_expert_tensors(&[&gate, &up, &down], 1, layout).unwrap();

    assert_eq!(tensors.gate.start(1).unwrap(), 6);
    assert_eq!(tensors.up.start(1).unwrap(), 6);
    assert!(aggregate_native_q4_enabled(&tensors, &down).unwrap());
}

#[test]
fn native_q4_layout_resolves_from_qwen_moe_dimensions() {
    let gate = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.gate_proj.weight",
        shape: vec![512, 1536, 4096],
        native_q4: true,
    };
    let up = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.up_proj.weight",
        shape: vec![512, 1536, 4096],
        native_q4: true,
    };
    let down = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.down_proj.weight",
        shape: vec![512, 4096, 1536],
        native_q4: true,
    };
    let aggregate = AggregateExpertLayout::new(512, 4096, 1536).unwrap();
    let tensors = aggregate_expert_tensors(&[&gate, &up, &down], 1, aggregate).unwrap();
    let fixed = fixed_native_q4_aggregate_layout(&tensors, &down, aggregate)
        .unwrap()
        .unwrap();

    assert_eq!(fixed.group_size, GROUP_SIZE);
    assert_eq!(fixed.expert_bytes, 10_616_832);
    assert_ne!(fixed, QwenMoeQ4ExpertLayout::qwen35_a17b());
}

#[test]
fn native_q4_layout_keeps_non_runtime_fixture_as_import_format() {
    let gate = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.gate_proj.weight",
        shape: vec![2, 2, 3],
        native_q4: true,
    };
    let up = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.up_proj.weight",
        shape: vec![2, 2, 3],
        native_q4: true,
    };
    let down = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.down_proj.weight",
        shape: vec![2, 3, 2],
        native_q4: true,
    };
    let aggregate = AggregateExpertLayout::new(2, 3, 2).unwrap();
    let tensors = aggregate_expert_tensors(&[&gate, &up, &down], 1, aggregate).unwrap();

    assert!(
        fixed_native_q4_aggregate_layout(&tensors, &down, aggregate)
            .unwrap()
            .is_none()
    );
}

#[test]
fn aggregate_native_q4_requires_consistent_sources() {
    let gate = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.gate_proj.weight",
        shape: vec![2, 2, 3],
        native_q4: true,
    };
    let up = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.up_proj.weight",
        shape: vec![2, 2, 3],
        native_q4: false,
    };
    let down = TestAggregateTensor {
        name: "model.layers.1.mlp.switch_mlp.down_proj.weight",
        shape: vec![2, 3, 2],
        native_q4: true,
    };
    let layout = AggregateExpertLayout::new(2, 3, 2).unwrap();
    let tensors = aggregate_expert_tensors(&[&gate, &up, &down], 1, layout).unwrap();
    let err = aggregate_native_q4_enabled(&tensors, &down).unwrap_err();

    assert!(
        err.to_string()
            .contains("aggregate expert tensors must be all native MLX Q4 or all decoded tensors")
    );
}

#[test]
fn q4_record_layout_counts_packed_bytes_and_groups_from_shape() {
    assert_eq!(q4_record_layout_for_shape(&[3, 5]).unwrap(), (9, 3));
    assert_eq!(q4_record_layout_for_shape(&[2, 3, 33]).unwrap(), (102, 6));
}

#[test]
fn q4_record_layout_rejects_zero_column_shape() {
    let err = q4_record_layout_for_shape(&[3, 0]).unwrap_err();
    assert!(
        err.to_string()
            .contains("cannot compute q4 layout for zero-column tensor")
    );
}

#[test]
fn direct_expert_tensor_group_accepts_gate_up_down_shapes() {
    let gate = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.7.gate_proj.weight",
        shape: vec![2, 3],
        native_q4: false,
    };
    let up = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.7.up_proj.weight",
        shape: vec![2, 3],
        native_q4: false,
    };
    let down = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.7.down_proj.weight",
        shape: vec![3, 2],
        native_q4: false,
    };

    validate_direct_expert_tensor_group(
        0,
        7,
        &[&gate, &up, &down],
        Some(DirectExpertTensorShape::new(3, 2).unwrap()),
    )
    .unwrap();
}

#[test]
fn direct_expert_tensor_group_rejects_missing_or_duplicate_projection() {
    let gate = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.7.gate_proj.weight",
        shape: vec![2, 3],
        native_q4: false,
    };
    let duplicate_gate = TestAggregateTensor {
        name: "alias.model.layers.0.mlp.experts.7.gate_proj.weight",
        shape: vec![2, 3],
        native_q4: false,
    };
    let up = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.7.up_proj.weight",
        shape: vec![2, 3],
        native_q4: false,
    };
    let down = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.7.down_proj.weight",
        shape: vec![3, 2],
        native_q4: false,
    };

    let missing = validate_direct_expert_tensor_group(0, 7, &[&gate, &up], None).unwrap_err();
    assert!(
        missing
            .to_string()
            .contains("is missing required tensor down_proj.weight")
    );

    let duplicate =
        validate_direct_expert_tensor_group(0, 7, &[&gate, &duplicate_gate, &up, &down], None)
            .unwrap_err();
    assert!(
        duplicate
            .to_string()
            .contains("has duplicate tensors ending in gate_proj.weight")
    );
}

#[test]
fn direct_expert_tensor_group_rejects_shape_mismatch() {
    let gate = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.7.gate_proj.weight",
        shape: vec![2, 3],
        native_q4: false,
    };
    let up = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.7.up_proj.weight",
        shape: vec![2, 4],
        native_q4: false,
    };
    let down = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.7.down_proj.weight",
        shape: vec![3, 2],
        native_q4: false,
    };

    let err = validate_direct_expert_tensor_group(
        0,
        7,
        &[&gate, &up, &down],
        Some(DirectExpertTensorShape::new(3, 2).unwrap()),
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("expected [2, 3] for up_proj.weight")
    );
}

#[test]
fn native_q4_slice_byte_ranges_plan_row_aligned_slices() {
    let source = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.gate_up_proj.weight",
        shape: vec![4, 8],
        native_q4: true,
    };
    let ranges = native_q4_slice_byte_ranges(&source, &[2, 8], "BF16", 8, 16).unwrap();

    assert_eq!(
        ranges,
        NativeQ4SliceByteRanges {
            packed_offset: 4,
            packed_bytes: 8,
            scale_bias_offset: 2,
            scale_bias_bytes: 4,
            groups: 2,
        }
    );
}

#[test]
fn native_q4_slice_byte_ranges_reject_unaligned_or_mismatched_slices() {
    let source = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.gate_up_proj.weight",
        shape: vec![4, 8],
        native_q4: true,
    };

    let unaligned = native_q4_slice_byte_ranges(&source, &[1, 8], "BF16", 1, 8).unwrap_err();
    assert!(
        unaligned
            .to_string()
            .contains("is not aligned to 8 columns")
    );

    let mismatch = native_q4_slice_byte_ranges(&source, &[3, 8], "BF16", 8, 16).unwrap_err();
    assert!(
        mismatch
            .to_string()
            .contains("slice shape [3, 8] does not match 2 rows x 8 cols")
    );
}

#[test]
fn expert_tensor_byte_range_uses_source_offsets_and_dtype_width() {
    let tensor = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.7.gate_proj.weight",
        shape: vec![2, 3],
        native_q4: false,
    };

    assert_eq!(
        expert_tensor_byte_range(&tensor, "BF16", 3, 4).unwrap(),
        [106, 114]
    );
    assert_eq!(
        expert_tensor_byte_range(&tensor, "U8", 3, 4).unwrap(),
        [103, 107]
    );
}

#[test]
fn expert_tensor_byte_range_rejects_unsupported_or_out_of_bounds_ranges() {
    let tensor = TestAggregateTensor {
        name: "model.layers.0.mlp.experts.7.gate_proj.weight",
        shape: vec![2, 3],
        native_q4: false,
    };

    let dtype = expert_tensor_byte_range(&tensor, "U32", 0, 1).unwrap_err();
    assert!(dtype.to_string().contains("has unsupported dtype U32"));

    let bounds = expert_tensor_byte_range(&tensor, "F32", 20, 10).unwrap_err();
    assert!(bounds.to_string().contains("exceeds source offsets"));
}

#[test]
fn expected_expert_record_from_source_uses_q4_layout_accounting() {
    let record = expected_expert_pack_record_from_source(
        "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
        "BF16".to_string(),
        vec![3, 5],
        [12, 42],
        "hash".to_string(),
    )
    .unwrap();

    assert_eq!(record.packed_bytes, 9);
    assert_eq!(record.groups, 3);
    assert_eq!(record.group_size, GROUP_SIZE);
    assert_eq!(record.scale_bias_dtype, EXPERT_PACK_SCALE_BIAS_DTYPE);
}

#[test]
fn expected_native_q4_record_requires_source_hash() {
    let input = NativeQ4ExpertRecordInput {
        tensor: "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
        dtype: "U32".to_string(),
        shape: vec![3, 5],
        source_offsets: [12, 42],
        source_hash: None,
        packed: vec![0; 9],
        scale_bytes: vec![0; 6],
        bias_bytes: vec![0; 6],
        groups: 3,
        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
    };

    let err = expected_native_q4_expert_record_from_input(input).unwrap_err();
    assert!(
        err.to_string()
            .contains("native q4 expert record is missing source hash")
    );
}

#[test]
fn expected_expert_pack_from_records_accounts_for_wire_size() {
    let record = expected_expert_pack_record_from_source(
        "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
        "BF16".to_string(),
        vec![3, 5],
        [12, 42],
        "hash".to_string(),
    )
    .unwrap();
    let expected_size = pbq4_expert_pack_wire_size(&[record.clone()]).unwrap();
    let pack = expected_expert_pack_from_records(1, vec![record]).unwrap();

    assert_eq!(pack.expert, 1);
    assert_eq!(pack.packed_bytes, expected_size);
    assert_eq!(pack.records.len(), 1);
}

fn tiny_fixed_q4_layout() -> QwenMoeQ4ExpertLayout {
    use QwenMoeExpertComponentKind::*;
    QwenMoeQ4ExpertLayout {
        expert_bytes: 45,
        group_size: 2,
        components: [
            QwenMoeExpertComponentLayout {
                kind: GateWeight,
                offset: 0,
                bytes: 8,
            },
            QwenMoeExpertComponentLayout {
                kind: GateScale,
                offset: 8,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: GateBias,
                offset: 12,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: UpWeight,
                offset: 16,
                bytes: 8,
            },
            QwenMoeExpertComponentLayout {
                kind: UpScale,
                offset: 24,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: UpBias,
                offset: 28,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: DownWeight,
                offset: 32,
                bytes: 8,
            },
            QwenMoeExpertComponentLayout {
                kind: DownScale,
                offset: 40,
                bytes: 3,
            },
            QwenMoeExpertComponentLayout {
                kind: DownBias,
                offset: 43,
                bytes: 2,
            },
        ],
    }
}

fn tiny_pbq4_expert_pack() -> (Vec<u8>, ExpertPackMetadata) {
    let tensor = "model.layers.0.mlp.experts.1.gate_proj.weight".to_string();
    let mut pack = PBQ4_EXPERT_MAGIC.to_vec();
    let record_offset = pack.len() as u64;
    pack.extend_from_slice(&(tensor.len() as u32).to_le_bytes());
    pack.extend_from_slice(tensor.as_bytes());
    pack.extend_from_slice(&2u64.to_le_bytes());
    pack.extend_from_slice(&2u64.to_le_bytes());
    for value in [1.0f32, 1.5] {
        pack.extend_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
    }
    for value in [0.0f32, 2.0] {
        pack.extend_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
    }
    pack.extend_from_slice(&[0xf0, 0x0f]);
    let packed_bytes = pack.len() as u64;
    (
        pack,
        ExpertPackMetadata {
            layer: 0,
            expert: 1,
            packed_bytes,
            records: vec![ExpertPackRecord {
                tensor,
                dtype: "F32".to_string(),
                shape: vec![2, 2],
                source_offsets: [16, 32],
                source_hash: Some("fixture".to_string()),
                record_offset,
                packed_bytes: 2,
                groups: 2,
                group_size: GROUP_SIZE,
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            }],
        },
    )
}

#[derive(Debug, Clone)]
struct TestExpertPackWireRecord {
    tensor: String,
    packed_bytes: u64,
    groups: usize,
    scale_bias_dtype: String,
}

impl ExpertPackWireRecord for TestExpertPackWireRecord {
    fn tensor_name(&self) -> &str {
        &self.tensor
    }

    fn packed_bytes(&self) -> u64 {
        self.packed_bytes
    }

    fn scale_bias_groups(&self) -> usize {
        self.groups
    }

    fn scale_bias_dtype(&self) -> &str {
        &self.scale_bias_dtype
    }
}

#[test]
fn pbq4_expert_wire_size_accounts_for_bf16_scale_bias_metadata() {
    let records: Vec<TestExpertPackWireRecord> = [
        ("gate_proj.weight", 2_097_152, 65_536),
        ("up_proj.weight", 2_097_152, 65_536),
        ("down_proj.weight", 2_097_152, 65_536),
    ]
    .into_iter()
    .map(|(tensor, packed_bytes, groups)| TestExpertPackWireRecord {
        tensor: tensor.to_string(),
        packed_bytes,
        groups,
        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
    })
    .collect();
    let f32_size = pbq4_expert_pack_wire_size(&records).unwrap();
    let mut bf16_records = records.clone();
    for record in &mut bf16_records {
        record.scale_bias_dtype = EXPERT_SCALE_BIAS_DTYPE_BF16.to_string();
    }
    let bf16_size = pbq4_expert_pack_wire_size(&bf16_records).unwrap();

    assert_eq!(f32_size - bf16_size, 786_432);
    assert!(bf16_size < f32_size);
}

#[test]
fn fixed_q4_expert_slot_records_are_derived_from_layout_offsets() {
    let layout = tiny_fixed_q4_layout();
    let mut payload: Vec<u8> = (0..45).collect();
    payload[8..12].copy_from_slice(
        &[
            f32_to_bf16_bits(0.5).to_le_bytes(),
            f32_to_bf16_bits(1.5).to_le_bytes(),
        ]
        .concat(),
    );
    payload[12..16].copy_from_slice(
        &[
            f32_to_bf16_bits(-1.0).to_le_bytes(),
            f32_to_bf16_bits(2.0).to_le_bytes(),
        ]
        .concat(),
    );
    payload[24..28].copy_from_slice(
        &[
            f32_to_bf16_bits(3.0).to_le_bytes(),
            f32_to_bf16_bits(4.0).to_le_bytes(),
        ]
        .concat(),
    );
    payload[28..32].copy_from_slice(
        &[
            f32_to_bf16_bits(5.0).to_le_bytes(),
            f32_to_bf16_bits(6.0).to_le_bytes(),
        ]
        .concat(),
    );

    let slot = ExpertSlotView::new(2, 9, 128, 45, &payload).unwrap();
    let view = FixedQ4ExpertSlotView::new(slot, layout).unwrap();
    let records = fixed_q4_expert_records(
        &view,
        FixedQ4ExpertSlotSpec {
            layout,
            hidden_size: 2,
            intermediate_size: 2,
            encoding: FixedQ4ExpertEncoding::AffineBf16,
        },
    )
    .unwrap();

    assert_eq!(records.len(), 3);
    assert_eq!(
        records[0].name,
        "model.layers.2.mlp.experts.9.gate_proj.weight"
    );
    assert_eq!(records[0].shape, vec![2, 2]);
    assert_eq!(records[0].packed, payload[0..8]);
    assert_eq!(records[0].scales, vec![0.5, 1.5]);
    assert_eq!(records[0].biases, vec![-1.0, 2.0]);
    assert_eq!(records[1].packed, payload[16..24]);
    assert_eq!(records[1].scales, vec![3.0, 4.0]);
    assert_eq!(records[1].biases, vec![5.0, 6.0]);
    assert_eq!(records[2].shape, vec![2, 2]);
    assert_eq!(records[2].scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
}

#[test]
fn pbq4_metadata_parser_matches_generic_parser() {
    let (pack, metadata) = tiny_pbq4_expert_pack();
    let generic = parse_pbq4_expert_pack_generic(&pack, Some(&metadata)).unwrap();
    let metadata_fast = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap();

    assert_eq!(metadata_fast, generic);
    assert_eq!(metadata_fast[0].packed, vec![0xf0, 0x0f]);
    assert_eq!(metadata_fast[0].scales, vec![1.0, 1.5]);
    assert_eq!(metadata_fast[0].biases, vec![0.0, 2.0]);
    assert_eq!(metadata_fast[0].source_offsets(), [16, 32]);
}

#[test]
fn pbq4_metadata_parser_rejects_record_offset_drift() {
    let (pack, mut metadata) = tiny_pbq4_expert_pack();
    metadata.records[0].record_offset += 1;

    let err = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap_err();

    assert!(
        err.to_string().contains("metadata offset mismatch"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn read_f32_vec_le_bulk_path_decodes_values_and_advances_cursor() {
    let mut bytes = vec![0xaa, 0xbb, 0xcc];
    for value in [1.0f32, -2.5, f32::from_bits(0x7fc0_1234)] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&[0xdd, 0xee]);

    let mut cursor = 3;
    let values = read_f32_vec_le(&bytes, &mut cursor, 3).unwrap();
    assert_eq!(cursor, 15);
    assert_eq!(values[0], 1.0);
    assert_eq!(values[1], -2.5);
    assert_eq!(values[2].to_bits(), 0x7fc0_1234);
}

#[test]
fn read_bf16_vec_le_decodes_values_and_advances_cursor() {
    let mut bytes = vec![0xaa];
    for value in [0.5f32, -2.0, 1.25] {
        bytes.extend_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
    }
    bytes.push(0xbb);

    let mut cursor = 1;
    let values = read_bf16_vec_le(&bytes, &mut cursor, 3).unwrap();
    assert_eq!(cursor, 7);
    assert_eq!(values, vec![0.5, -2.0, 1.25]);
}

#[test]
fn reusable_expert_buffer_keeps_capacity_across_smaller_reads() {
    let mut buffer = ReusableExpertBuffer::default();
    buffer.prepare_payload(128, 96).unwrap().fill(7);
    let initial_capacity = buffer.capacity();

    buffer.prepare_payload(128, 8).unwrap().fill(3);
    let slot = buffer.slot_view(2, 5, 1024, 128).unwrap();

    assert_eq!(buffer.capacity(), initial_capacity);
    assert_eq!(
        slot.descriptor(),
        ExpertSlotDescriptor {
            layer: 2,
            expert: 5,
            slot_offset: 1024,
            slot_capacity: 128,
            payload_len: 8,
        }
    );
    assert_eq!(slot.payload(), &[3; 8]);
}

#[test]
fn reusable_expert_buffer_can_move_a_whole_slot_payload_without_copying() {
    let mut buffer = ReusableExpertBuffer::default();
    buffer.prepare_payload(128, 96).unwrap().fill(9);
    let initial_capacity = buffer.capacity();

    let payload = buffer.take_payload();

    assert_eq!(payload, vec![9; 96]);
    assert_eq!(payload.capacity(), initial_capacity);
    assert_eq!(buffer.capacity(), 0);
}

#[test]
fn reusable_expert_buffer_uses_page_aligned_backing_for_a_whole_slot() {
    let mut buffer = ReusableExpertBuffer::default();
    let slot_bytes = 16 * 1024;
    buffer
        .prepare_payload(slot_bytes, slot_bytes)
        .unwrap()
        .fill(7);

    let payload = buffer.take_payload();

    assert!(matches!(
        &payload.backing,
        ReusableExpertBytesBacking::PageAligned(_)
    ));
    assert_eq!((payload.as_ptr() as usize) % 4096, 0);
    assert_eq!(payload.as_slice(), &[7; 16 * 1024]);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn reusable_expert_attachment_follows_identity_free_pool_slot() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct DropMarker(Arc<AtomicUsize>);

    impl Drop for DropMarker {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    let pool: ReusableExpertBytePool = Arc::new(Mutex::new(Vec::new()));
    let slot_bytes = 16 * 1024;
    let mut scratch = ReusableExpertBuffer::default();
    scratch.prepare_payload(slot_bytes, slot_bytes).unwrap();
    let payload = scratch.take_payload();
    let initial_ptr = payload.as_ptr();
    assert!(
        payload
            .install_attachment(DropMarker(Arc::clone(&drops)))
            .is_some()
    );

    recycle_reusable_expert_bytes(&pool, payload, slot_bytes);
    let reused = take_reusable_expert_bytes(&pool, slot_bytes).unwrap();
    assert_eq!(reused.as_ptr(), initial_ptr);
    assert!(reused.attachment::<DropMarker>().is_some());
    let previous = scratch.adopt_buffer(reused);
    assert_eq!(previous.capacity(), 0);
    scratch.prepare_payload(slot_bytes, slot_bytes).unwrap();
    assert_eq!(scratch.bytes.as_ptr(), initial_ptr);

    drop(scratch);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn fixed_q4_expert_slot_view_slices_components_from_one_payload() {
    let payload: Vec<u8> = (0..45).collect();
    let slot = ExpertSlotView::new(4, 7, 4096, 45, &payload).unwrap();
    let view = FixedQ4ExpertSlotView::new(slot, tiny_fixed_q4_layout()).unwrap();

    assert_eq!(view.descriptor(), slot.descriptor());
    assert_eq!(view.payload(), payload.as_slice());
    assert_eq!(
        view.component(QwenMoeExpertComponentKind::GateWeight),
        &payload[0..8]
    );
    assert_eq!(
        view.component(QwenMoeExpertComponentKind::UpScale),
        &payload[24..28]
    );
    assert_eq!(
        view.component(QwenMoeExpertComponentKind::DownBias),
        &payload[43..45]
    );
}

#[test]
fn fixed_q4_expert_slot_view_rejects_short_payloads() {
    let payload = [0u8; 44];
    let slot = ExpertSlotView::new(0, 0, 0, 45, &payload).unwrap();
    let err = FixedQ4ExpertSlotView::new(slot, tiny_fixed_q4_layout()).unwrap_err();

    assert!(
        err.to_string().contains("shorter than layout size 45"),
        "{err:#}"
    );
}

#[test]
fn fixed_q4_payload_requires_whole_slot_bytes() {
    let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
    let payload = FixedQ4ExpertPayload::from_whole_slot(spec, (0..45).collect(), None).unwrap();

    assert_eq!(
        payload.component(QwenMoeExpertComponentKind::GateWeight),
        &[0, 1, 2, 3, 4, 5, 6, 7]
    );

    let err = FixedQ4ExpertPayload::from_whole_slot(spec, vec![0; 44], None).unwrap_err();
    assert!(
        err.to_string().contains("whole-slot payload length 44"),
        "{err:#}"
    );
}

#[test]
fn expert_store_resolves_validated_fixed_q4_execution_storage() {
    let temp = tempfile::tempdir().unwrap();
    let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
    let experts = 2;
    let bytes = vec![0u8; spec.layout.expert_bytes * experts];
    fs::write(expert_layer_path(temp.path(), 0), bytes).unwrap();
    let metadata = ExpertLayerPackMetadata::new_fixed_q4(
        0,
        spec.layout.expert_bytes as u64,
        experts,
        (0..experts)
            .map(|expert| tiny_pack_metadata(0, expert, spec.layout.expert_bytes as u64))
            .collect(),
    );
    write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();
    let store = ExpertSlotStore::open_with_fixed_q4(temp.path().to_path_buf(), spec).unwrap();

    let descriptor = store.resolve_execution_descriptor(1, experts).unwrap();

    assert_eq!(descriptor.layout, ExpertStorageLayout::FixedQ4);
    assert_eq!(descriptor.slot_spec, ExpertSlotSpec::FixedQ4(spec));
    assert_eq!(descriptor.layers, 1);
    assert_eq!(descriptor.experts_per_layer, experts);
}

#[test]
fn expert_store_resolves_q4_layout_from_fixed_slot_metadata() {
    let layout = tiny_qwen_moe_layout();
    let temp = tempfile::tempdir().unwrap();
    let spec = FixedQ4ExpertSlotSpec::from_model_layout(&layout).unwrap();
    fs::write(
        expert_layer_path(temp.path(), 0),
        vec![0u8; spec.layout.expert_bytes * layout.experts_per_layer],
    )
    .unwrap();
    let metadata = ExpertLayerPackMetadata::new_fixed_q4(
        0,
        spec.layout.expert_bytes as u64,
        layout.experts_per_layer,
        (0..layout.experts_per_layer)
            .map(|expert| tiny_pack_metadata(0, expert, spec.layout.expert_bytes as u64))
            .collect(),
    );
    write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();

    let resolved = ExpertSlotStore::resolve_from_metadata(
        temp.path().to_path_buf(),
        &layout,
        ExpertQuantization::FourBitProduction,
    )
    .unwrap();

    assert_eq!(resolved.descriptor.slot_spec, ExpertSlotSpec::FixedQ4(spec));
    assert_eq!(resolved.upgraded_pbq4_layers, 0);

    let error = ExpertSlotStore::resolve_from_metadata(
        temp.path().to_path_buf(),
        &layout,
        ExpertQuantization::Bf16,
    )
    .unwrap_err();
    assert!(error.to_string().contains("requested BF16 expert weights"));
    assert!(error.to_string().contains("resolves fixed-Q4 expert slots"));
}

#[test]
fn expert_store_resolves_native_mxfp4_layout_from_fixed_slot_metadata() {
    let layout = tiny_qwen_moe_layout();
    let temp = tempfile::tempdir().unwrap();
    let spec = FixedQ4ExpertSlotSpec::mxfp4_from_model_layout(&layout).unwrap();
    fs::write(
        expert_layer_path(temp.path(), 0),
        vec![0u8; spec.layout.expert_bytes * layout.experts_per_layer],
    )
    .unwrap();
    let metadata = ExpertLayerPackMetadata::new_fixed_mxfp4(
        0,
        spec.layout.expert_bytes as u64,
        layout.experts_per_layer,
        (0..layout.experts_per_layer)
            .map(|expert| tiny_pack_metadata(0, expert, spec.layout.expert_bytes as u64))
            .collect(),
    );
    write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();

    let resolved = ExpertSlotStore::resolve_from_metadata(
        temp.path().to_path_buf(),
        &layout,
        ExpertQuantization::FourBitProduction,
    )
    .unwrap();

    assert_eq!(resolved.descriptor.layout, ExpertStorageLayout::FixedMxfp4);
    assert_eq!(resolved.descriptor.slot_spec, ExpertSlotSpec::FixedQ4(spec));
    assert_eq!(resolved.upgraded_pbq4_layers, 0);
}

#[test]
fn requested_expert_storage_rejects_every_mismatched_resolved_layout() {
    for resolved in [
        ExpertStorageLayout::FixedQ4,
        ExpertStorageLayout::FixedMxfp4,
        ExpertStorageLayout::FixedBf16,
        ExpertStorageLayout::FixedF16,
    ] {
        for requested in [
            ExpertQuantization::FourBitProduction,
            ExpertQuantization::Bf16,
            ExpertQuantization::F16,
        ] {
            let result = validate_requested_expert_storage(
                Path::new("/cache/packed_experts"),
                resolved,
                requested,
            );
            if resolved.quantization() == requested {
                result.unwrap();
            } else {
                let error = result.unwrap_err().to_string();
                assert!(
                    error.contains("unsupported expert storage policy"),
                    "{error}"
                );
                assert!(error.contains(requested.as_str()), "{error}");
                assert!(error.contains(resolved.as_str()), "{error}");
            }
        }
    }
}

#[test]
fn native_mxfp4_import_rejects_non_finite_e8m0_scales() {
    validate_mxfp4_e8m0_scales("gate_proj.weight", &[126, 127, 128]).unwrap();

    let error = validate_mxfp4_e8m0_scales("gate_proj.weight", &[127, u8::MAX]).unwrap_err();
    assert!(error.to_string().contains("gate_proj.weight"));
    assert!(error.to_string().contains("non-finite E8M0 scale 0xff"));
    assert!(error.to_string().contains("byte 1"));
}

#[test]
fn fixed_dense_expert_slots_resolve_offsets_reads_and_typed_payloads() {
    for (dtype, expected_layout) in [
        (DenseExpertDtype::Bf16, ExpertStorageLayout::FixedBf16),
        (DenseExpertDtype::F16, ExpertStorageLayout::FixedF16),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let spec = FixedDenseExpertSlotSpec::new(dtype, 2, 3).unwrap();
        assert_eq!(spec.gate.offset, 0);
        assert_eq!(spec.gate.rows, 3);
        assert_eq!(spec.gate.cols, 2);
        assert_eq!(spec.up.offset, EXPERT_COMPONENT_ALIGNMENT);
        assert_eq!(spec.down.offset, EXPERT_COMPONENT_ALIGNMENT * 2);
        assert_eq!(spec.expert_bytes, EXPERT_COMPONENT_ALIGNMENT * 3);

        let experts = 2;
        let mut bytes = vec![0u8; spec.expert_bytes * experts];
        bytes[spec.gate.offset] = 11;
        bytes[spec.up.offset] = 22;
        bytes[spec.down.offset] = 33;
        fs::write(expert_layer_path(temp.path(), 0), bytes).unwrap();
        let metadata = ExpertLayerPackMetadata::new_fixed_dense(
            0,
            spec.expert_bytes as u64,
            experts,
            (0..experts)
                .map(|expert| fixed_dense_test_pack(0, expert, spec, [dtype.as_str(); 3]))
                .collect(),
        );
        write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();
        let store =
            ExpertSlotStore::open_with_fixed_dense(temp.path().to_path_buf(), spec).unwrap();

        let descriptor = store.resolve_execution_descriptor(1, experts).unwrap();
        assert_eq!(descriptor.layout, expected_layout);
        assert_eq!(descriptor.slot_spec.fixed_dense(), Some(spec));

        let mut reads = store.read_many_raw(0, &[0]).unwrap();
        let raw = reads.pop().unwrap();
        assert_eq!(raw.slot.slot_capacity, spec.expert_bytes);
        let ExpertRawPayload::FixedDense(payload) = raw.payload else {
            panic!("fixed dense slot did not resolve a fixed dense payload");
        };
        let gate = payload
            .matvec_payload(ExpertMlpProjection::Gate, 2, 3)
            .unwrap();
        let up = payload
            .matvec_payload(ExpertMlpProjection::Up, 2, 3)
            .unwrap();
        let down = payload
            .matvec_payload(ExpertMlpProjection::Down, 3, 2)
            .unwrap();
        assert_eq!(gate.dtype, dtype);
        assert_eq!(gate.source.byte_offset, spec.gate.offset);
        assert_eq!(up.source.byte_offset, spec.up.offset);
        assert_eq!(down.source.byte_offset, spec.down.offset);
        assert_eq!(gate.source.bytes[gate.source.byte_offset], 11);
        assert_eq!(up.source.bytes[up.source.byte_offset], 22);
        assert_eq!(down.source.bytes[down.source.byte_offset], 33);
    }
}

#[test]
fn expert_store_resolves_typed_layout_from_fixed_slot_metadata() {
    let layout = tiny_qwen_moe_layout();
    for dtype in [DenseExpertDtype::Bf16, DenseExpertDtype::F16] {
        let temp = tempfile::tempdir().unwrap();
        let spec = FixedDenseExpertSlotSpec::from_model_layout(&layout, dtype).unwrap();
        fs::write(
            expert_layer_path(temp.path(), 0),
            vec![0u8; spec.expert_bytes * layout.experts_per_layer],
        )
        .unwrap();
        let metadata = ExpertLayerPackMetadata::new_fixed_dense(
            0,
            spec.expert_bytes as u64,
            layout.experts_per_layer,
            (0..layout.experts_per_layer)
                .map(|expert| fixed_dense_test_pack(0, expert, spec, [dtype.as_str(); 3]))
                .collect(),
        );
        write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();

        let quantization = match dtype {
            DenseExpertDtype::Bf16 => ExpertQuantization::Bf16,
            DenseExpertDtype::F16 => ExpertQuantization::F16,
        };
        let resolved = ExpertSlotStore::resolve_from_metadata(
            temp.path().to_path_buf(),
            &layout,
            quantization,
        )
        .unwrap();

        assert_eq!(resolved.descriptor.slot_spec.fixed_dense(), Some(spec));
        assert_eq!(resolved.upgraded_pbq4_layers, 0);

        let mismatched_quantization = match dtype {
            DenseExpertDtype::Bf16 => ExpertQuantization::F16,
            DenseExpertDtype::F16 => ExpertQuantization::Bf16,
        };
        let error = ExpertSlotStore::resolve_from_metadata(
            temp.path().to_path_buf(),
            &layout,
            mismatched_quantization,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains(mismatched_quantization.as_str()),
            "{error:#}"
        );
        assert!(
            error
                .to_string()
                .contains(resolved.descriptor.layout.as_str()),
            "{error:#}"
        );
    }
}

#[test]
fn expert_store_rejects_mixed_fixed_dense_metadata_before_scheduling() {
    let layout = tiny_qwen_moe_layout();
    let temp = tempfile::tempdir().unwrap();
    let spec =
        FixedDenseExpertSlotSpec::from_model_layout(&layout, DenseExpertDtype::Bf16).unwrap();
    fs::write(
        expert_layer_path(temp.path(), 0),
        vec![0u8; spec.expert_bytes * layout.experts_per_layer],
    )
    .unwrap();
    let metadata = ExpertLayerPackMetadata::new_fixed_dense(
        0,
        spec.expert_bytes as u64,
        layout.experts_per_layer,
        (0..layout.experts_per_layer)
            .map(|expert| fixed_dense_test_pack(0, expert, spec, ["BF16", "F16", "BF16"]))
            .collect(),
    );
    write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();

    let error = ExpertSlotStore::resolve_from_metadata(
        temp.path().to_path_buf(),
        &layout,
        ExpertQuantization::Bf16,
    )
    .unwrap_err();

    assert!(
        error.to_string().contains("mixes BF16 and F16"),
        "{error:#}"
    );
}

#[test]
fn expert_store_rejects_partial_fixed_q4_execution_storage() {
    let temp = tempfile::tempdir().unwrap();
    let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
    fs::write(
        expert_layer_path(temp.path(), 0),
        vec![0u8; spec.layout.expert_bytes],
    )
    .unwrap();
    let metadata = ExpertLayerPackMetadata::new_fixed_q4(
        0,
        spec.layout.expert_bytes as u64,
        2,
        vec![tiny_pack_metadata(0, 0, spec.layout.expert_bytes as u64)],
    );
    write_expert_metadata_atomically(temp.path(), 0, &metadata).unwrap();
    let store = ExpertSlotStore::open_with_fixed_q4(temp.path().to_path_buf(), spec).unwrap();

    let err = store.resolve_execution_descriptor(1, 2).unwrap_err();

    assert!(err.to_string().contains("1 records, expected 2"), "{err:#}");
}

#[test]
fn fixed_q4_payload_resolves_typed_matvec_offsets_from_whole_slot() {
    let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
    let payload = FixedQ4ExpertPayload::from_whole_slot(spec, (0..45).collect(), None).unwrap();

    let gate = payload
        .matvec_payload(ExpertMlpProjection::Gate, 2, 2)
        .unwrap();
    assert_eq!(gate.rows, 2);
    assert_eq!(gate.cols, 2);
    assert_eq!(gate.group_size, 2);
    assert_eq!(gate.scale_bias_groups, 2);
    assert_eq!(gate.scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
    assert_eq!(gate.packed, &[0, 1]);
    assert_eq!(gate.scale_bytes, &[8, 9, 10, 11]);
    assert_eq!(gate.bias_bytes, &[12, 13, 14, 15]);

    let source = gate.source.unwrap();
    assert_eq!(source.bytes, payload.payload_prefix(45));
    assert_eq!(source.packed_offset, 0);
    assert_eq!(source.scale_offset, 8);
    assert_eq!(source.bias_offset, 12);
    assert!(source.covers(&gate));
    assert!(source.offsets_are_metal_aligned());

    let up = payload
        .matvec_payload(ExpertMlpProjection::Up, 2, 2)
        .unwrap();
    let up_source = up.source.unwrap();
    assert!(source.same_buffer(up_source));
    assert_eq!(up_source.packed_offset, 16);
    assert_eq!(up_source.scale_offset, 24);
    assert_eq!(up_source.bias_offset, 28);
}

#[test]
fn fixed_q4_payload_rejects_partial_projection_bytes() {
    let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
    let payload = FixedQ4ExpertPayload::from_whole_slot(spec, (0..45).collect(), None).unwrap();

    assert!(
        payload
            .matvec_payload(ExpertMlpProjection::Down, 2, 2)
            .is_none()
    );
}

fn tiny_pack_metadata(layer: usize, expert: usize, packed_bytes: u64) -> ExpertPackMetadata {
    ExpertPackMetadata {
        layer,
        expert,
        packed_bytes,
        records: Vec::new(),
    }
}

fn tiny_qwen_moe_layout() -> QwenMoeModelLayout {
    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"num_hidden_layers":1,"hidden_size":64,"num_attention_heads":8,"num_key_value_heads":1,"vocab_size":128,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":64,"norm_topk_prob":true}"#,
        )
        .unwrap();
    QwenMoeModelLayout::from_config("hf://Qwen/tiny-moe", &config).unwrap()
}

fn fixed_dense_test_pack(
    layer: usize,
    expert: usize,
    spec: FixedDenseExpertSlotSpec,
    dtypes: [&str; 3],
) -> ExpertPackMetadata {
    let records = [
        ("gate_proj.weight", spec.gate, dtypes[0]),
        ("up_proj.weight", spec.up, dtypes[1]),
        ("down_proj.weight", spec.down, dtypes[2]),
    ]
    .into_iter()
    .map(|(suffix, projection, dtype)| ExpertPackRecord {
        tensor: format!("model.layers.{layer}.mlp.experts.{expert}.{suffix}"),
        dtype: dtype.to_string(),
        shape: vec![projection.rows, projection.cols],
        source_offsets: [0, projection.bytes as u64],
        source_hash: Some("fixture".to_string()),
        record_offset: projection.offset as u64,
        packed_bytes: projection.bytes as u64,
        groups: 0,
        group_size: 0,
        scale_bias_dtype: String::new(),
    })
    .collect();
    ExpertPackMetadata {
        layer,
        expert,
        packed_bytes: spec.expert_bytes as u64,
        records,
    }
}

#[test]
fn expert_layer_metadata_validates_supported_formats_and_slots() {
    let metadata = ExpertLayerPackMetadata::new_fixed_q4(
        3,
        128,
        2,
        vec![tiny_pack_metadata(3, 0, 128), tiny_pack_metadata(3, 1, 64)],
    );
    metadata.validate(Path::new("layer_03.json"), 3).unwrap();

    let duplicate = ExpertLayerPackMetadata::new_fixed_q4(
        3,
        128,
        2,
        vec![tiny_pack_metadata(3, 0, 128), tiny_pack_metadata(3, 0, 64)],
    );
    let err = duplicate
        .validate(Path::new("layer_03.json"), 3)
        .unwrap_err();
    assert!(err.to_string().contains("duplicate expert 0"), "{err:#}");

    let oversized =
        ExpertLayerPackMetadata::new_fixed_q4(3, 128, 2, vec![tiny_pack_metadata(3, 1, 129)]);
    let err = oversized
        .validate(Path::new("layer_03.json"), 3)
        .unwrap_err();
    assert!(
        err.to_string().contains("length 129 exceeds slot size 128"),
        "{err:#}"
    );
}

#[test]
fn expert_metadata_reader_uses_expert_owned_paths_and_validation() {
    let tmp = tempfile::tempdir().unwrap();
    let metadata = ExpertLayerPackMetadata::new(5, 256, 4, vec![tiny_pack_metadata(5, 2, 200)]);
    std::fs::write(
        expert_layer_metadata_path(tmp.path(), 5),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();

    let read_layer = read_expert_layer_pack_metadata(tmp.path(), 5)
        .unwrap()
        .unwrap();
    let read_pack = read_expert_pack_metadata(tmp.path(), 5, 2)
        .unwrap()
        .unwrap();

    assert_eq!(read_layer.format, PBQ4_EXPERT_LAYER_FORMAT_V2);
    assert_eq!(read_layer.pack_for(2), Some(&read_pack));
    assert_eq!(read_pack.packed_bytes, 200);
    assert_eq!(
        expert_layer_path(tmp.path(), 5),
        tmp.path().join("layer_05.bin")
    );
}

#[test]
fn expert_layer_reader_reads_fixed_q4_whole_slots() {
    let tmp = tempfile::tempdir().unwrap();
    let layout = tiny_fixed_q4_layout();
    let spec = FixedQ4ExpertSlotSpec::new(layout, 2, 2).unwrap();
    let payload: Vec<u8> = (0..45).collect();
    std::fs::write(expert_layer_path(tmp.path(), 0), &payload).unwrap();
    let metadata = ExpertLayerPackMetadata::new_fixed_q4(
        0,
        layout.expert_bytes as u64,
        1,
        vec![tiny_pack_metadata(0, 0, layout.expert_bytes as u64)],
    );
    std::fs::write(
        expert_layer_metadata_path(tmp.path(), 0),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();

    let reader = ExpertLayerReader::open(
        tmp.path(),
        0,
        ExpertSlotSpec::FixedQ4(spec),
        Arc::new(Mutex::new(Vec::new())),
    )
    .unwrap();
    let plan = reader.prepare_read(0).unwrap();
    let mut scratch = ReusableExpertBuffer::default();
    let raw = reader.read_prepared_into(0, plan, &mut scratch).unwrap();

    assert_eq!(raw.slot.layer, 0);
    assert_eq!(raw.slot.expert, 0);
    assert_eq!(raw.read_path, ExpertReadPath::PositionedRead);
    match raw.payload {
        ExpertRawPayload::FixedQ4(fixed) => {
            assert_eq!(
                fixed.component(QwenMoeExpertComponentKind::GateWeight),
                &[0, 1, 2, 3, 4, 5, 6, 7]
            );
        }
        ExpertRawPayload::Pbq4(_) => panic!("fixed slot classified as PBQ4"),
        ExpertRawPayload::FixedDense(_) => panic!("Q4 slot classified as fixed dense"),
        ExpertRawPayload::FixedDeepSeekGguf(_) => {
            panic!("Q4 slot classified as DeepSeek GGUF")
        }
    }
}

#[test]
fn expert_layer_reader_keeps_pbq4_as_import_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let layout = tiny_fixed_q4_layout();
    let spec = FixedQ4ExpertSlotSpec::new(layout, 2, 2).unwrap();
    let mut payload = PBQ4_EXPERT_MAGIC.to_vec();
    payload.extend_from_slice(&[1, 2, 3, 4]);
    std::fs::write(expert_layer_path(tmp.path(), 0), &payload).unwrap();
    let metadata = ExpertLayerPackMetadata::new(
        0,
        payload.len() as u64,
        1,
        vec![tiny_pack_metadata(0, 0, payload.len() as u64)],
    );
    std::fs::write(
        expert_layer_metadata_path(tmp.path(), 0),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();

    let reader = ExpertLayerReader::open(
        tmp.path(),
        0,
        ExpertSlotSpec::FixedQ4(spec),
        Arc::new(Mutex::new(Vec::new())),
    )
    .unwrap();
    let plan = reader.prepare_read(0).unwrap();
    let mut scratch = ReusableExpertBuffer::default();
    let raw = reader.read_prepared_into(0, plan, &mut scratch).unwrap();

    match raw.payload {
        ExpertRawPayload::Pbq4(bytes) => assert_eq!(bytes, payload),
        ExpertRawPayload::FixedQ4(_) => panic!("PBQ4 slot classified as fixed Q4"),
        ExpertRawPayload::FixedDense(_) => panic!("PBQ4 slot classified as fixed dense"),
        ExpertRawPayload::FixedDeepSeekGguf(_) => {
            panic!("PBQ4 slot classified as DeepSeek GGUF")
        }
    }
}

#[test]
fn expert_read_worker_pool_returns_raw_whole_slot_reads() {
    let tmp = tempfile::tempdir().unwrap();
    let layout = tiny_fixed_q4_layout();
    let spec = FixedQ4ExpertSlotSpec::new(layout, 2, 2).unwrap();
    let payload: Vec<u8> = (0..45).collect();
    std::fs::write(expert_layer_path(tmp.path(), 0), &payload).unwrap();
    let metadata = ExpertLayerPackMetadata::new_fixed_q4(
        0,
        layout.expert_bytes as u64,
        1,
        vec![tiny_pack_metadata(0, 0, layout.expert_bytes as u64)],
    );
    std::fs::write(
        expert_layer_metadata_path(tmp.path(), 0),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
    let reader = Arc::new(
        ExpertLayerReader::open(
            tmp.path(),
            0,
            ExpertSlotSpec::FixedQ4(spec),
            Arc::new(Mutex::new(Vec::new())),
        )
        .unwrap(),
    );
    let plan = reader.prepare_read(0).unwrap();
    let mut pool = ExpertReadWorkerPool::default();

    let rx = pool
        .submit_read(11, 0, reader, plan, false, Instant::now())
        .unwrap();
    let response = rx.recv().unwrap();
    let raw = response.result.unwrap();

    assert_eq!(pool.worker_count(), 1);
    assert_eq!(response.id, 11);
    assert_eq!(response.read_path, ExpertReadPath::PositionedRead);
    assert_eq!(response.bytes_read, layout.expert_bytes as u64);
    assert!(!response.warm);
    assert_eq!(raw.slot.layer, 0);
    assert_eq!(raw.slot.expert, 0);
    match raw.payload {
        ExpertRawPayload::FixedQ4(fixed) => {
            assert_eq!(
                fixed.component(QwenMoeExpertComponentKind::DownBias),
                &[43, 44]
            );
        }
        ExpertRawPayload::Pbq4(_) => panic!("fixed slot classified as PBQ4"),
        ExpertRawPayload::FixedDense(_) => panic!("Q4 slot classified as fixed dense"),
        ExpertRawPayload::FixedDeepSeekGguf(_) => {
            panic!("Q4 slot classified as DeepSeek GGUF")
        }
    }
}

#[test]
fn expert_slot_store_owns_layer_reader_cache_and_raw_reads() {
    let tmp = tempfile::tempdir().unwrap();
    let layout = tiny_fixed_q4_layout();
    let spec = FixedQ4ExpertSlotSpec::new(layout, 2, 2).unwrap();
    let payload: Vec<u8> = (0..45).collect();
    std::fs::write(expert_layer_path(tmp.path(), 0), &payload).unwrap();
    let metadata = ExpertLayerPackMetadata::new_fixed_q4(
        0,
        layout.expert_bytes as u64,
        1,
        vec![tiny_pack_metadata(0, 0, layout.expert_bytes as u64)],
    );
    std::fs::write(
        expert_layer_metadata_path(tmp.path(), 0),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();

    let store = ExpertSlotStore::open_with_fixed_q4(tmp.path().to_path_buf(), spec).unwrap();
    let first_reader = store.layer_reader(0).unwrap();
    let second_reader = store.layer_reader(0).unwrap();
    let reads = store.read_many_raw(0, &[0]).unwrap();

    assert!(Arc::ptr_eq(&first_reader, &second_reader));
    assert_eq!(reads.len(), 1);
    assert_eq!(reads[0].slot.payload_len, layout.expert_bytes);
    assert!(matches!(reads[0].payload, ExpertRawPayload::FixedQ4(_)));
}

#[test]
fn fixed_q4_slot_spec_validates_layout_and_dimensions() {
    let layout = tiny_fixed_q4_layout();
    let spec = FixedQ4ExpertSlotSpec::new(layout, 8, 4).unwrap();

    assert_eq!(spec.layout, layout);
    assert_eq!(spec.hidden_size, 8);
    assert_eq!(spec.intermediate_size, 4);
}

#[test]
fn fixed_q4_slot_spec_rejects_invalid_layout_offsets() {
    let mut layout = tiny_fixed_q4_layout();
    layout.components[1].offset += 1;

    let err = FixedQ4ExpertSlotSpec::new(layout, 8, 4).unwrap_err();

    assert!(
        err.to_string()
            .contains("GateScale starts at 9, expected 8"),
        "{err:#}"
    );
}

#[test]
fn fixed_q4_slot_spec_rejects_zero_dimensions() {
    let err = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 0, 4).unwrap_err();

    assert!(
        err.to_string().contains("requires non-zero dimensions"),
        "{err:#}"
    );
}

#[test]
fn expert_slot_rejects_payloads_larger_than_the_slot() {
    let err = ExpertSlotView::new(0, 0, 0, 2, &[1, 2, 3]).unwrap_err();

    assert!(
        err.to_string()
            .contains("payload length 3 exceeds slot capacity 2"),
        "{err:#}"
    );
}

#[test]
fn expert_io_policy_keeps_upstream_positioned_read_guardrails() {
    assert_eq!(
        FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
        ExpertReadPath::PositionedRead
    );
    assert!(!FLASHMOE_EXPERT_IO_POLICY.application_expert_cache);
    assert!(!FLASHMOE_EXPERT_IO_POLICY.lz4_expert_compression);
    assert!(!FLASHMOE_EXPERT_IO_POLICY.speculative_routing);
    assert!(!FLASHMOE_EXPERT_IO_POLICY.broad_ssd_gpu_overlap);
    assert!(FLASHMOE_EXPERT_IO_POLICY.layer_ahead_request_staging);
}

#[test]
fn pending_layer_prepare_drop_joins_direct_destination_workers() {
    let (worker_started_tx, worker_started_rx) = mpsc::channel();
    let (release_worker_tx, release_worker_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        worker_started_tx.send(()).unwrap();
        release_worker_rx.recv().unwrap();
        Ok(ExpertLayerPrepareWorkerSummary { bytes_read: 0 })
    });
    let pending = PendingExpertLayerPrepare {
        layer: 0,
        bytes: 0,
        workers: vec![worker],
        destination: std::marker::PhantomData,
    };
    let (drop_finished_tx, drop_finished_rx) = mpsc::channel();
    let dropper = thread::spawn(move || {
        drop(pending);
        drop_finished_tx.send(()).unwrap();
    });

    worker_started_rx.recv().unwrap();
    assert!(drop_finished_rx.try_recv().is_err());
    release_worker_tx.send(()).unwrap();
    drop_finished_rx.recv().unwrap();
    dropper.join().unwrap();
}

#[test]
fn fixed_q4_payload_drop_recycles_whole_slot_bytes() {
    let spec = FixedQ4ExpertSlotSpec {
        layout: QwenMoeQ4ExpertLayout::qwen35_a17b(),
        hidden_size: 3584,
        intermediate_size: 1024,
        encoding: FixedQ4ExpertEncoding::AffineBf16,
    };
    let pool = Arc::new(Mutex::new(Vec::new()));
    let mut bytes = Vec::with_capacity(spec.layout.expert_bytes);
    bytes.resize(spec.layout.expert_bytes, 0);

    {
        let _payload = FixedQ4ExpertPayload {
            spec,
            bytes: bytes.into(),
            recycle_pool: Some(Arc::clone(&pool)),
        };
    }

    assert_eq!(pool.lock().unwrap().len(), 1);
    let returned = take_reusable_expert_bytes(&pool, spec.layout.expert_bytes).unwrap();
    assert!(returned.capacity() >= spec.layout.expert_bytes);
    let mut scratch = ReusableExpertBuffer::default();
    let previous = scratch.adopt_buffer(returned);
    assert_eq!(previous.capacity(), 0);
    assert!(scratch.capacity() >= spec.layout.expert_bytes);
}

#[test]
fn cleanup_stale_expert_temp_files_preserves_committed_layers() {
    let temp = tempfile::tempdir().unwrap();
    let experts_dir = temp.path();
    let final_bin = experts_dir.join("layer_00.bin");
    let temp_bin = experts_dir.join("layer_00.bin.tmp-123-ThreadId(1)");
    let temp_json = experts_dir.join("layer_00.json.tmp-123-ThreadId(1)");

    fs::write(&final_bin, b"PBQ4EXPERT ").unwrap();
    fs::write(&temp_bin, b"partial").unwrap();
    fs::write(&temp_json, b"partial").unwrap();

    let deleted = cleanup_stale_expert_temp_files(experts_dir).unwrap();

    assert_eq!(deleted, 2);
    assert!(final_bin.is_file());
    assert!(!temp_bin.exists());
    assert!(!temp_json.exists());
}

#[test]
fn reusable_expert_byte_pool_reuses_capacity_qualified_buffers() {
    let pool: ReusableExpertBytePool = Arc::new(Mutex::new(Vec::new()));
    recycle_reusable_expert_bytes(&pool, Vec::with_capacity(64), 64);

    let returned = take_reusable_expert_bytes(&pool, 32).unwrap();

    assert!(returned.capacity() >= 64);
    assert!(pool.lock().unwrap().is_empty());
}
