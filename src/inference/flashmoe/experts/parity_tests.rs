//! Owner-local PBQ4 import and fixed-slot compatibility parity tests.

use super::*;
use crate::inference::flashmoe::cache::{build_cache_from_hf_snapshot, build_manifest};
use crate::inference::flashmoe::math::{q4_fma_matvec, q4_fma_matvec_with_group_size};
use crate::inference::flashmoe::model_family::{
    QwenMoeExpertComponentKind, QwenMoeModelLayout, QwenMoeQ4ExpertLayout,
};
use crate::inference::flashmoe::planning::plan_unchecked;
use crate::inference::flashmoe::test_fixtures::*;
use crate::inference::flashmoe::text::test_tokenizer_json;
use crate::inference::flashmoe::types::*;

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn tiny_q4_expert_pack() -> (Vec<u8>, ExpertPackMetadata) {
    let prefix = "model.layers.0.mlp.experts.1";
    build_expert_pack(
        0,
        1,
        vec![
            ExpertRecordInput {
                tensor: format!("{prefix}.gate_proj.weight"),
                dtype: "F32".to_string(),
                shape: vec![2, 2],
                source_offsets: [0, 16],
                source_hash: Some("fixture-gate".to_string()),
                values: vec![0.0, 15.0, 15.0, 0.0],
            },
            ExpertRecordInput {
                tensor: format!("{prefix}.up_proj.weight"),
                dtype: "F32".to_string(),
                shape: vec![2, 2],
                source_offsets: [16, 32],
                source_hash: Some("fixture-up".to_string()),
                values: vec![15.0, 15.0, 15.0, 15.0],
            },
            ExpertRecordInput {
                tensor: format!("{prefix}.down_proj.weight"),
                dtype: "F32".to_string(),
                shape: vec![2, 2],
                source_offsets: [32, 48],
                source_hash: Some("fixture-down".to_string()),
                values: vec![15.0, 0.0, 0.0, 15.0],
            },
        ],
    )
    .unwrap()
}

fn fixed_q4_test_layout(
    hidden_size: usize,
    intermediate_size: usize,
    group_size: usize,
) -> QwenMoeQ4ExpertLayout {
    use crate::inference::flashmoe::QwenMoeExpertComponentLayout;
    use QwenMoeExpertComponentKind::*;

    let packed_gate_up = intermediate_size * hidden_size.div_ceil(2);
    let gate_up_scale_bias = intermediate_size * hidden_size.div_ceil(group_size) * 2;
    let packed_down = hidden_size * intermediate_size.div_ceil(2);
    let down_scale_bias = hidden_size * intermediate_size.div_ceil(group_size) * 2;
    let mut offset = 0usize;
    let mut component = |kind, bytes| {
        let layout = QwenMoeExpertComponentLayout {
            kind,
            offset,
            bytes,
        };
        offset += bytes;
        layout
    };
    let components = [
        component(GateWeight, packed_gate_up),
        component(GateScale, gate_up_scale_bias),
        component(GateBias, gate_up_scale_bias),
        component(UpWeight, packed_gate_up),
        component(UpScale, gate_up_scale_bias),
        component(UpBias, gate_up_scale_bias),
        component(DownWeight, packed_down),
        component(DownScale, down_scale_bias),
        component(DownBias, down_scale_bias),
    ];
    QwenMoeQ4ExpertLayout {
        expert_bytes: offset,
        group_size,
        components,
    }
}

#[test]
fn flashmoe_parity_q4_expert_pack_and_mlp_goldens() {
    let (pack, metadata) = tiny_q4_expert_pack();
    let records = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap();

    assert_eq!(records.len(), 3);
    assert_eq!(records[0].shape, vec![2, 2]);
    assert_eq!(records[0].group_size, GROUP_SIZE);
    assert_eq!(records[0].packed, vec![0xf0, 0x0f]);
    assert_eq!(records[0].scales, vec![1.0, 1.0]);
    assert_eq!(records[0].biases, vec![0.0, 0.0]);
    assert_eq!(records[1].packed, vec![0x00, 0x00]);
    assert_eq!(records[1].biases, vec![15.0, 15.0]);

    let hidden = [1.0, 2.0];
    let project = |record: &PackedExpertTensor| {
        let payload = record.matvec_payload(&hidden, 2).unwrap();
        q4_fma_matvec_with_group_size(
            payload.packed,
            &hidden,
            payload.scales,
            payload.biases,
            payload.rows,
            payload.cols,
            payload.group_size,
        )
        .unwrap()
    };
    let gate = project(
        records
            .iter()
            .find(|record| record.name.ends_with("gate_proj.weight"))
            .unwrap(),
    );
    let up = project(
        records
            .iter()
            .find(|record| record.name.ends_with("up_proj.weight"))
            .unwrap(),
    );
    assert_eq!(gate, vec![30.0, 15.0]);
    assert_eq!(up, vec![45.0, 45.0]);

    let spec = FixedQ4ExpertSlotSpec {
        layout: fixed_q4_test_layout(2, 2, GROUP_SIZE),
        hidden_size: 2,
        intermediate_size: 2,
        encoding: FixedQ4ExpertEncoding::AffineBf16,
    };
    let fixed_q4 = fixed_q4_payload_from_pbq4_records(0, 1, spec, &records, None).unwrap();
    let intermediate = [silu(gate[0]) * up[0], silu(gate[1]) * up[1]];
    let down_input = fixed_q4
        .project_cpu(ExpertMlpProjection::Down, &intermediate, 2)
        .unwrap();
    let out = down_input;
    assert_close(out[0], 15.0 * intermediate[0]);
    assert_close(out[1], 15.0 * intermediate[1]);
}

#[test]
fn pbq4_records_are_adapted_to_fixed_q4_payload() {
    use crate::inference::flashmoe::QwenMoeExpertComponentLayout;
    use QwenMoeExpertComponentKind::*;

    fn bf16_values(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f32_to_bf16_bits(*value).to_le_bytes())
            .collect()
    }

    fn native_record(
        name: &str,
        shape: Vec<usize>,
        packed: Vec<u8>,
        scales: Vec<u8>,
        biases: Vec<u8>,
        groups: usize,
    ) -> NativeQ4ExpertRecordInput {
        NativeQ4ExpertRecordInput {
            tensor: name.to_string(),
            dtype: "q4".to_string(),
            shape,
            source_offsets: [0, 0],
            source_hash: Some(format!("hash-{name}")),
            packed,
            scale_bytes: scales,
            bias_bytes: biases,
            groups,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        }
    }

    let layout = QwenMoeQ4ExpertLayout {
        expert_bytes: 464,
        group_size: GROUP_SIZE,
        components: [
            QwenMoeExpertComponentLayout {
                kind: GateWeight,
                offset: 0,
                bytes: 64,
            },
            QwenMoeExpertComponentLayout {
                kind: GateScale,
                offset: 64,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: GateBias,
                offset: 68,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: UpWeight,
                offset: 72,
                bytes: 64,
            },
            QwenMoeExpertComponentLayout {
                kind: UpScale,
                offset: 136,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: UpBias,
                offset: 140,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: DownWeight,
                offset: 144,
                bytes: 64,
            },
            QwenMoeExpertComponentLayout {
                kind: DownScale,
                offset: 208,
                bytes: 128,
            },
            QwenMoeExpertComponentLayout {
                kind: DownBias,
                offset: 336,
                bytes: 128,
            },
        ],
    };
    let layer = 5;
    let expert = 7;
    let gate_name = format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight");
    let up_name = format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight");
    let down_name = format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight");
    let gate_packed = vec![0x10; 64];
    let up_packed = vec![0x54; 64];
    let down_packed = vec![0x98; 64];
    let gate_scales = bf16_values(&[0.5, 0.25]);
    let gate_biases = bf16_values(&[1.0, -1.0]);
    let up_scales = bf16_values(&[0.75, 0.125]);
    let up_biases = bf16_values(&[0.0, 0.5]);
    let down_scales = bf16_values(&vec![0.25; 64]);
    let down_biases = bf16_values(&vec![-0.5; 64]);
    let (pack, metadata) = build_native_q4_expert_pack(
        layer,
        expert,
        vec![
            native_record(
                &gate_name,
                vec![2, 64],
                gate_packed.clone(),
                gate_scales.clone(),
                gate_biases.clone(),
                2,
            ),
            native_record(
                &up_name,
                vec![2, 64],
                up_packed.clone(),
                up_scales.clone(),
                up_biases.clone(),
                2,
            ),
            native_record(
                &down_name,
                vec![64, 2],
                down_packed.clone(),
                down_scales.clone(),
                down_biases.clone(),
                64,
            ),
        ],
    )
    .unwrap();
    let records = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap();
    let spec = FixedQ4ExpertSlotSpec {
        layout,
        hidden_size: 64,
        intermediate_size: 2,
        encoding: FixedQ4ExpertEncoding::AffineBf16,
    };
    let fixed = fixed_q4_payload_from_pbq4_records(layer, expert, spec, &records, None).unwrap();

    assert!(!fixed.bytes.starts_with(PBQ4_EXPERT_MAGIC));
    assert_eq!(&fixed.bytes[0..64], gate_packed.as_slice());
    assert_eq!(&fixed.bytes[64..68], gate_scales.as_slice());
    assert_eq!(&fixed.bytes[68..72], gate_biases.as_slice());
    assert_eq!(&fixed.bytes[72..136], up_packed.as_slice());
    assert_eq!(&fixed.bytes[136..140], up_scales.as_slice());
    assert_eq!(&fixed.bytes[140..144], up_biases.as_slice());
    assert_eq!(&fixed.bytes[144..208], down_packed.as_slice());
    assert_eq!(&fixed.bytes[208..336], down_scales.as_slice());
    assert_eq!(&fixed.bytes[336..464], down_biases.as_slice());

    let hidden: Vec<f32> = (0..64).map(|value| value as f32 / 8.0 - 4.0).collect();
    let gate = fixed
        .project_cpu(ExpertMlpProjection::Gate, &hidden, spec.intermediate_size)
        .unwrap();
    let up = fixed
        .project_cpu(ExpertMlpProjection::Up, &hidden, spec.intermediate_size)
        .unwrap();
    let activated: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(gate, up)| silu(*gate) * up)
        .collect();
    let output = fixed
        .project_cpu(ExpertMlpProjection::Down, &activated, spec.hidden_size)
        .unwrap();
    assert_eq!(output.len(), 64);
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn pbq4_layer_cache_rewrites_to_fixed_q4_slots() {
    use crate::inference::flashmoe::QwenMoeExpertComponentLayout;
    use QwenMoeExpertComponentKind::*;

    fn bf16_values(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f32_to_bf16_bits(*value).to_le_bytes())
            .collect()
    }

    fn native_record(
        name: &str,
        shape: Vec<usize>,
        packed: Vec<u8>,
        scales: Vec<u8>,
        biases: Vec<u8>,
        groups: usize,
    ) -> NativeQ4ExpertRecordInput {
        NativeQ4ExpertRecordInput {
            tensor: name.to_string(),
            dtype: "q4".to_string(),
            shape,
            source_offsets: [11, 22],
            source_hash: Some(format!("hash-{name}")),
            packed,
            scale_bytes: scales,
            bias_bytes: biases,
            groups,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        }
    }

    let layout = QwenMoeQ4ExpertLayout {
        expert_bytes: 464,
        group_size: GROUP_SIZE,
        components: [
            QwenMoeExpertComponentLayout {
                kind: GateWeight,
                offset: 0,
                bytes: 64,
            },
            QwenMoeExpertComponentLayout {
                kind: GateScale,
                offset: 64,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: GateBias,
                offset: 68,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: UpWeight,
                offset: 72,
                bytes: 64,
            },
            QwenMoeExpertComponentLayout {
                kind: UpScale,
                offset: 136,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: UpBias,
                offset: 140,
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: DownWeight,
                offset: 144,
                bytes: 64,
            },
            QwenMoeExpertComponentLayout {
                kind: DownScale,
                offset: 208,
                bytes: 128,
            },
            QwenMoeExpertComponentLayout {
                kind: DownBias,
                offset: 336,
                bytes: 128,
            },
        ],
    };
    let layer = 0;
    let expert = 0;
    let gate_name = format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight");
    let up_name = format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight");
    let down_name = format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight");
    let (pack, metadata) = build_native_q4_expert_pack(
        layer,
        expert,
        vec![
            native_record(
                &gate_name,
                vec![2, 64],
                vec![0x10; 64],
                bf16_values(&[0.5, 0.25]),
                bf16_values(&[1.0, -1.0]),
                2,
            ),
            native_record(
                &up_name,
                vec![2, 64],
                vec![0x54; 64],
                bf16_values(&[0.75, 0.125]),
                bf16_values(&[0.0, 0.5]),
                2,
            ),
            native_record(
                &down_name,
                vec![64, 2],
                vec![0x98; 64],
                bf16_values(&vec![0.25; 64]),
                bf16_values(&vec![-0.5; 64]),
                64,
            ),
        ],
    )
    .unwrap();

    let tmp = tempfile::tempdir().unwrap();
    write_test_expert_layer(tmp.path(), layer, vec![(expert, pack, metadata)], 1).unwrap();
    let spec = FixedQ4ExpertSlotSpec {
        layout,
        hidden_size: 64,
        intermediate_size: 2,
        encoding: FixedQ4ExpertEncoding::AffineBf16,
    };
    assert!(rewrite_pbq4_layer_to_fixed_q4(tmp.path(), layer, 1, spec).unwrap());
    assert!(!rewrite_pbq4_layer_to_fixed_q4(tmp.path(), layer, 1, spec).unwrap());

    let metadata = read_expert_layer_pack_metadata(tmp.path(), layer)
        .unwrap()
        .unwrap();
    assert_eq!(metadata.format, FIXED_Q4_EXPERT_LAYER_FORMAT_V1);
    assert_eq!(metadata.expert_size, layout.expert_bytes as u64);
    assert_eq!(metadata.packs[0].packed_bytes, layout.expert_bytes as u64);
    assert_eq!(metadata.packs[0].records[0].record_offset, 0);
    assert_eq!(metadata.packs[0].records[1].record_offset, 72);
    assert_eq!(metadata.packs[0].records[2].record_offset, 144);

    let mut prefix = vec![0u8; PBQ4_EXPERT_MAGIC.len()];
    let file = fs::File::open(expert_layer_path(tmp.path(), layer)).unwrap();
    read_exact_at_positioned(&file, &mut prefix, 0).unwrap();
    assert_ne!(prefix, PBQ4_EXPERT_MAGIC);

    let store = ExpertSlotStore::open_with_fixed_q4(tmp.path().to_path_buf(), spec).unwrap();
    let raw = store
        .read_many_raw(layer, &[expert])
        .unwrap()
        .pop()
        .unwrap();
    let ExpertRawPayload::FixedQ4(fixed) = raw.payload else {
        panic!("rewritten layer did not return fixed-Q4 execution storage");
    };
    assert_eq!(fixed.bytes.len(), layout.expert_bytes);
}

#[test]
fn build_expert_pack_writes_bf16_scale_bias_metadata_and_stays_projectable() {
    let input_values: Vec<f32> = (0..64).map(|idx| (idx as f32 - 32.0) * 0.125).collect();
    let (pack, metadata) = build_expert_pack(
        0,
        0,
        vec![ExpertRecordInput {
            tensor: "model.layers.0.mlp.experts.0.down_proj.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![1, 64],
            source_offsets: [0, 256],
            source_hash: Some("fixture".to_string()),
            values: input_values,
        }],
    )
    .unwrap();
    let record = &metadata.records[0];
    assert_eq!(record.scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
    assert_eq!(record.groups, 1);
    assert_eq!(
        pack.len(),
        PBQ4_EXPERT_MAGIC.len()
            + 4
            + record.tensor.len()
            + 8
            + 8
            + 2
            + 2
            + record.packed_bytes as usize
    );

    let parsed = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].packed.len(), 32);
    assert_eq!(parsed[0].scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
    assert_eq!(parsed[0].scale_bytes.len(), 2);
    assert_eq!(parsed[0].bias_bytes.len(), 2);
    let scale_offset = record.record_offset as usize + 4 + record.tensor.len() + 8 + 8;
    assert_eq!(parsed[0].scale_bytes, pack[scale_offset..scale_offset + 2]);
    let out = q4_fma_matvec_with_group_size(
        &parsed[0].packed,
        &[1.0; 64],
        &parsed[0].scales,
        &parsed[0].biases,
        1,
        64,
        GROUP_SIZE,
    )
    .unwrap();
    assert_eq!(out.len(), 1);
    assert!(out[0].is_finite());
}

#[test]
fn expert_store_parses_pbq4expert_records_as_import_data_only() {
    let temp = tempfile::tempdir().unwrap();
    fs::create_dir_all(temp.path()).unwrap();
    let tensor = "model.layers.0.mlp.experts.2.down_proj.weight";
    let pack = test_expert_pack(tensor);
    let metadata = test_expert_pack_metadata(0, 2, tensor, pack.len());
    write_test_expert_layer(temp.path(), 0, vec![(2, pack, metadata)], 8).unwrap();
    let layer_metadata = read_expert_layer_pack_metadata(temp.path(), 0)
        .unwrap()
        .unwrap();
    let expected = ExpectedExpertPack {
        expert: 2,
        packed_bytes: layer_metadata.pack_for(2).unwrap().packed_bytes,
        records: layer_metadata
            .pack_for(2)
            .unwrap()
            .records
            .iter()
            .map(|record| ExpectedExpertPackRecord {
                tensor: record.tensor.clone(),
                dtype: record.dtype.clone(),
                shape: record.shape.clone(),
                source_offsets: record.source_offsets,
                source_hash: record.source_hash.clone().unwrap(),
                packed_bytes: record.packed_bytes,
                groups: record.groups,
                group_size: record.group_size,
                scale_bias_dtype: record.scale_bias_dtype.clone(),
            })
            .collect(),
    };
    assert!(
        expert_layer_slot_is_reusable(
            &expert_layer_path(temp.path(), 0),
            &layer_metadata,
            ExpertLayerStorageFormat::Pbq4Import,
            &expected
        )
        .unwrap()
    );

    let records = read_pbq4_expert_records(temp.path(), 0, 2).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].name, tensor);
    assert_eq!(records[0].scales, vec![0.5]);
    assert_eq!(records[0].biases, vec![1.0]);
    let out = project_packed_expert_record(&records[0], &[1.0, 2.0, 3.0, 4.0], 1).unwrap();
    let expected = (1.0 * 0.5 + 1.0) * 1.0
        + (2.0 * 0.5 + 1.0) * 2.0
        + (3.0 * 0.5 + 1.0) * 3.0
        + (4.0 * 0.5 + 1.0) * 4.0;
    assert!((out[0] - expected).abs() < 1e-6);
}

#[test]
fn packer_splits_qwen35_aggregate_expert_tensors() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
    fs::create_dir_all(&snapshot).unwrap();
    let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
    fs::create_dir_all(&plan.experts_dir).unwrap();

    let gate_up_bytes: Vec<u8> = (0u8..16).collect();
    let down_bytes: Vec<u8> = (16u8..24).collect();
    fs::write(
        snapshot.join("expert.safetensors"),
        make_typed_safetensors(&[
            (
                "model.layers.1.mlp.experts.gate_up_proj",
                "U8",
                vec![2, 4, 2],
                &gate_up_bytes,
            ),
            (
                "model.layers.1.mlp.experts.down_proj",
                "U8",
                vec![2, 2, 2],
                &down_bytes,
            ),
        ]),
    )
    .unwrap();

    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":2,"num_attention_heads":1,"vocab_size":16,"torch_dtype":"bfloat16","num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":2}"#,
        )
        .unwrap();
    let tensors = vec![
        ExpertTensorRef {
            tensor: "model.layers.1.mlp.experts.gate_up_proj".to_string(),
            shard: "expert.safetensors".to_string(),
            layer: Some(1),
            expert: None,
            dtype: Some("U8".to_string()),
            shape: vec![2, 4, 2],
            source_offsets: Some([0, 16]),
            q4_sources: None,
        },
        ExpertTensorRef {
            tensor: "model.layers.1.mlp.experts.down_proj".to_string(),
            shard: "expert.safetensors".to_string(),
            layer: Some(1),
            expert: None,
            dtype: Some("U8".to_string()),
            shape: vec![2, 2, 2],
            source_offsets: Some([16, 24]),
            q4_sources: None,
        },
    ];

    pack_expert_tensors(
        &snapshot,
        ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
        &tensors,
        Some(&config),
    )
    .unwrap();

    let layer_path = expert_layer_path(&plan.experts_dir, 1);
    let metadata = read_expert_layer_pack_metadata(&plan.experts_dir, 1)
        .unwrap()
        .unwrap();
    assert!(layer_path.is_file());
    assert_eq!(
        fs::metadata(&layer_path).unwrap().len(),
        metadata.expert_size * metadata.experts as u64
    );
    assert_eq!(metadata.experts, 2);
    assert_eq!(metadata.packs.len(), 2);
    assert!(metadata.pack_for(0).is_some());
    assert!(metadata.pack_for(1).is_some());

    let expert0 = read_pbq4_expert_records(&plan.experts_dir, 1, 0).unwrap();
    let expert1 = read_pbq4_expert_records(&plan.experts_dir, 1, 1).unwrap();
    assert!(expert_pack_is_complete(&plan.experts_dir, 1, 0));
    assert!(expert_pack_is_complete(&plan.experts_dir, 1, 1));
    for records in [&expert0, &expert1] {
        assert_eq!(records.len(), 3);
        assert!(packed_expert_record_suffix(records, "gate_proj.weight").is_some());
        assert!(packed_expert_record_suffix(records, "up_proj.weight").is_some());
        assert!(packed_expert_record_suffix(records, "down_proj.weight").is_some());
    }
    let input = [1.0, 1.0];
    let expert0_gate = project_packed_expert_record(
        packed_expert_record_suffix(&expert0, "gate_proj.weight").unwrap(),
        &input,
        2,
    )
    .unwrap();
    let expert0_up = project_packed_expert_record(
        packed_expert_record_suffix(&expert0, "up_proj.weight").unwrap(),
        &input,
        2,
    )
    .unwrap();
    let expert1_gate = project_packed_expert_record(
        packed_expert_record_suffix(&expert1, "gate_proj.weight").unwrap(),
        &input,
        2,
    )
    .unwrap();
    let expert1_down = project_packed_expert_record(
        packed_expert_record_suffix(&expert1, "down_proj.weight").unwrap(),
        &input,
        2,
    )
    .unwrap();
    for (actual, expected) in expert0_gate.iter().zip([1.0, 5.0]) {
        assert_close_with_tolerance(*actual, expected, 0.01);
    }
    for (actual, expected) in expert0_up.iter().zip([9.0, 13.0]) {
        assert_close_with_tolerance(*actual, expected, 0.01);
    }
    for (actual, expected) in expert1_gate.iter().zip([17.0, 21.0]) {
        assert_close_with_tolerance(*actual, expected, 0.01);
    }
    for (actual, expected) in expert1_down.iter().zip([41.0, 45.0]) {
        assert_close_with_tolerance(*actual, expected, 0.01);
    }
}

#[test]
fn packer_splits_mlx_switch_mlp_aggregate_expert_tensors() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
    fs::create_dir_all(&snapshot).unwrap();
    let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
    fs::create_dir_all(&plan.experts_dir).unwrap();

    let gate_bytes: Vec<u8> = (0u8..8).collect();
    let up_bytes: Vec<u8> = (8u8..16).collect();
    let down_bytes: Vec<u8> = (16u8..24).collect();
    fs::write(
        snapshot.join("expert.safetensors"),
        make_typed_safetensors(&[
            (
                "model.layers.1.mlp.switch_mlp.gate_proj.weight",
                "U8",
                vec![2, 2, 2],
                &gate_bytes,
            ),
            (
                "model.layers.1.mlp.switch_mlp.up_proj.weight",
                "U8",
                vec![2, 2, 2],
                &up_bytes,
            ),
            (
                "model.layers.1.mlp.switch_mlp.down_proj.weight",
                "U8",
                vec![2, 2, 2],
                &down_bytes,
            ),
        ]),
    )
    .unwrap();

    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":2,"num_attention_heads":1,"vocab_size":16,"torch_dtype":"bfloat16","num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":2}"#,
        )
        .unwrap();
    let tensors = vec![
        ExpertTensorRef {
            tensor: "model.layers.1.mlp.switch_mlp.gate_proj.weight".to_string(),
            shard: "expert.safetensors".to_string(),
            layer: Some(1),
            expert: None,
            dtype: Some("U8".to_string()),
            shape: vec![2, 2, 2],
            source_offsets: Some([0, 8]),
            q4_sources: None,
        },
        ExpertTensorRef {
            tensor: "model.layers.1.mlp.switch_mlp.up_proj.weight".to_string(),
            shard: "expert.safetensors".to_string(),
            layer: Some(1),
            expert: None,
            dtype: Some("U8".to_string()),
            shape: vec![2, 2, 2],
            source_offsets: Some([8, 16]),
            q4_sources: None,
        },
        ExpertTensorRef {
            tensor: "model.layers.1.mlp.switch_mlp.down_proj.weight".to_string(),
            shard: "expert.safetensors".to_string(),
            layer: Some(1),
            expert: None,
            dtype: Some("U8".to_string()),
            shape: vec![2, 2, 2],
            source_offsets: Some([16, 24]),
            q4_sources: None,
        },
    ];

    pack_expert_tensors(
        &snapshot,
        ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
        &tensors,
        Some(&config),
    )
    .unwrap();

    let metadata = read_expert_layer_pack_metadata(&plan.experts_dir, 1)
        .unwrap()
        .unwrap();
    assert_eq!(metadata.experts, 2);
    assert_eq!(metadata.packs.len(), 2);
    let expert0 = read_pbq4_expert_records(&plan.experts_dir, 1, 0).unwrap();
    let expert1 = read_pbq4_expert_records(&plan.experts_dir, 1, 1).unwrap();
    for records in [&expert0, &expert1] {
        assert_eq!(records.len(), 3);
        assert!(packed_expert_record_suffix(records, "gate_proj.weight").is_some());
        assert!(packed_expert_record_suffix(records, "up_proj.weight").is_some());
        assert!(packed_expert_record_suffix(records, "down_proj.weight").is_some());
    }
    let input = [1.0, 1.0];
    let expert0_gate = project_packed_expert_record(
        packed_expert_record_suffix(&expert0, "gate_proj.weight").unwrap(),
        &input,
        2,
    )
    .unwrap();
    let expert0_up = project_packed_expert_record(
        packed_expert_record_suffix(&expert0, "up_proj.weight").unwrap(),
        &input,
        2,
    )
    .unwrap();
    let expert1_gate = project_packed_expert_record(
        packed_expert_record_suffix(&expert1, "gate_proj.weight").unwrap(),
        &input,
        2,
    )
    .unwrap();
    let expert1_down = project_packed_expert_record(
        packed_expert_record_suffix(&expert1, "down_proj.weight").unwrap(),
        &input,
        2,
    )
    .unwrap();
    for (actual, expected) in expert0_gate.iter().zip([1.0, 5.0]) {
        assert_close_with_tolerance(*actual, expected, 0.01);
    }
    for (actual, expected) in expert0_up.iter().zip([17.0, 21.0]) {
        assert_close_with_tolerance(*actual, expected, 0.01);
    }
    for (actual, expected) in expert1_gate.iter().zip([9.0, 13.0]) {
        assert_close_with_tolerance(*actual, expected, 0.01);
    }
    for (actual, expected) in expert1_down.iter().zip([41.0, 45.0]) {
        assert_close_with_tolerance(*actual, expected, 0.01);
    }
}

#[test]
fn packer_copies_native_mlx_q4_switch_mlp_experts_without_requantizing() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
    fs::create_dir_all(&snapshot).unwrap();
    let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
    fs::create_dir_all(&plan.experts_dir).unwrap();

    let packed_words: Vec<u32> = (0..16).map(|_| 0x7654_3210).collect();
    let gate_packed = u32_tensor_bytes(&packed_words);
    let up_words: Vec<u32> = (0..16)
        .map(|row| 0x0123_4567u32.wrapping_add(row))
        .collect();
    let up_packed = u32_tensor_bytes(&up_words);
    let down_words: Vec<u32> = (0..16)
        .map(|row| 0x89ab_cdefu32.wrapping_add(row))
        .collect();
    let down_packed = u32_tensor_bytes(&down_words);
    let gate_scales = bf16_tensor_bytes(&[0.5; 16]);
    let gate_biases = bf16_tensor_bytes(&[1.0; 16]);
    let up_scales = bf16_tensor_bytes(&[0.25; 16]);
    let up_biases = bf16_tensor_bytes(&[2.0; 16]);
    let down_scales = bf16_tensor_bytes(&[0.125; 16]);
    let down_biases = bf16_tensor_bytes(&[3.0; 16]);
    let tensors = vec![
        (
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight".to_string(),
            "U32".to_string(),
            vec![2, 8, 1],
            gate_packed.clone(),
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.scales".to_string(),
            EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            vec![2, 8, 1],
            gate_scales.clone(),
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.biases".to_string(),
            EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            vec![2, 8, 1],
            gate_biases.clone(),
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.up_proj.weight".to_string(),
            "U32".to_string(),
            vec![2, 8, 1],
            up_packed.clone(),
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.up_proj.scales".to_string(),
            EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            vec![2, 8, 1],
            up_scales.clone(),
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.up_proj.biases".to_string(),
            EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            vec![2, 8, 1],
            up_biases.clone(),
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight".to_string(),
            "U32".to_string(),
            vec![2, 8, 1],
            down_packed.clone(),
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.down_proj.scales".to_string(),
            EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            vec![2, 8, 1],
            down_scales.clone(),
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.down_proj.biases".to_string(),
            EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            vec![2, 8, 1],
            down_biases.clone(),
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
        build_manifest(QWEN35_MODEL, &snapshot, &index_path, None).unwrap();
    assert!(visual_refs.is_empty());
    assert!(manifest.dense_tensors.is_empty());
    assert_eq!(manifest.expert_tensors.len(), 3);
    assert!(manifest.expert_tensors.iter().all(|tensor| {
        tensor.q4_sources.is_some()
            && tensor.shape == vec![2, 8, 8]
            && tensor.dtype.as_deref() == Some("U32")
    }));

    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":8,"num_attention_heads":1,"vocab_size":16,"torch_dtype":"bfloat16","num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":8}"#,
        )
        .unwrap();
    pack_expert_tensors(
        &snapshot,
        ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
        &manifest.expert_tensors,
        Some(&config),
    )
    .unwrap();

    let expert0 = read_pbq4_expert_records(&plan.experts_dir, 1, 0).unwrap();
    let gate0 = packed_expert_record_suffix(&expert0, "gate_proj.weight").unwrap();
    assert_eq!(gate0.dtype, "U32");
    assert_eq!(gate0.shape, vec![8, 8]);
    assert_eq!(gate0.scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
    assert_eq!(gate0.packed, gate_packed[..32]);
    assert_eq!(gate0.scale_bytes, gate_scales[..16]);
    assert_eq!(gate0.bias_bytes, gate_biases[..16]);
    let input = [1.0; 8];
    let projected = project_packed_expert_record(gate0, &input, 8).unwrap();
    assert_eq!(projected, vec![22.0; 8]);

    let expert1 = read_pbq4_expert_records(&plan.experts_dir, 1, 1).unwrap();
    let up1 = packed_expert_record_suffix(&expert1, "up_proj.weight").unwrap();
    assert_eq!(up1.packed, up_packed[32..]);
    assert_eq!(up1.scale_bytes, up_scales[16..]);
    assert_eq!(up1.bias_bytes, up_biases[16..]);
    let down1 = packed_expert_record_suffix(&expert1, "down_proj.weight").unwrap();
    assert_eq!(down1.packed, down_packed[32..]);
    assert_eq!(down1.scale_bytes, down_scales[16..]);
    assert_eq!(down1.bias_bytes, down_biases[16..]);
}

#[test]
fn packer_preserves_mlx_mxfp4_switch_mlp_experts_in_fixed_native_slots() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path().join(crate::cache_dir_name(GLM52_MODEL));
    fs::create_dir_all(&snapshot).unwrap();
    let plan = plan_unchecked(GLM52_MODEL, tmp.path());
    fs::create_dir_all(&plan.experts_dir).unwrap();

    let tensor = |name: &str, nibble_pair: u8| {
        vec![
            (
                format!("model.layers.3.mlp.switch_mlp.{name}.weight"),
                "U32".to_string(),
                vec![2, 32, 4],
                vec![nibble_pair; 2 * 32 * 16],
            ),
            (
                format!("model.layers.3.mlp.switch_mlp.{name}.scales"),
                "U8".to_string(),
                vec![2, 32, 1],
                vec![127; 2 * 32],
            ),
        ]
    };
    let tensors = tensor("gate_proj", 0x22)
        .into_iter()
        .chain(tensor("up_proj", 0x44))
        .chain(tensor("down_proj", 0x11))
        .collect::<Vec<_>>();
    let fixture_refs = typed_fixture_refs(&tensors);
    fs::write(
        snapshot.join("experts.safetensors"),
        make_typed_safetensors(&fixture_refs),
    )
    .unwrap();
    let weight_map = tensors
        .iter()
        .map(|(name, _, _, _)| {
            (
                name.clone(),
                serde_json::Value::String("experts.safetensors".to_string()),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let index = serde_json::json!({"weight_map": weight_map});
    let index_path = snapshot.join("model.safetensors.index.json");
    fs::write(&index_path, index.to_string()).unwrap();

    let config: QwenModelConfig = serde_json::from_value(serde_json::json!({
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 78,
        "hidden_size": 32,
        "num_attention_heads": 1,
        "head_dim": 32,
        "vocab_size": 128,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "norm_topk_prob": true,
        "moe_intermediate_size": 32,
        "intermediate_size": 32,
        "first_k_dense_replace": 3,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 32,
        "v_head_dim": 32,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "index_topk": 2048
    }))
    .unwrap();

    let (manifest, visual_refs) =
        build_manifest(GLM52_MODEL, &snapshot, &index_path, Some(&config)).unwrap();
    assert!(visual_refs.is_empty());
    assert_eq!(manifest.expert_tensors.len(), 3);
    assert!(manifest.expert_tensors.iter().all(|tensor| {
        tensor.shape == vec![2, 32, 32]
            && tensor.q4_sources.as_ref().is_some_and(|source| {
                source.source_format == DenseQ4SourceFormat::MlxMxfp4
                    && source.source_group_size == Some(32)
            })
    }));

    pack_expert_tensors(
        &snapshot,
        ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
        &manifest.expert_tensors,
        Some(&config),
    )
    .unwrap();

    let model_layout = QwenMoeModelLayout::from_config(GLM52_MODEL, &config).unwrap();
    let spec = FixedQ4ExpertSlotSpec::mxfp4_from_model_layout(&model_layout).unwrap();
    let metadata = read_expert_layer_pack_metadata(&plan.experts_dir, 3)
        .unwrap()
        .unwrap();
    assert_eq!(metadata.format, FIXED_MXFP4_EXPERT_LAYER_FORMAT_V1);
    assert_eq!(metadata.expert_size, spec.layout.expert_bytes as u64);
    let store = ExpertSlotStore::open_with_fixed_q4(plan.experts_dir.clone(), spec).unwrap();
    let mut reads = store.read_many_raw(3, &[0]).unwrap();
    let expert0 = match reads.remove(0).payload {
        ExpertRawPayload::FixedQ4(payload) => payload,
        other => panic!("expected fixed MXFP4 payload, found {other:?}"),
    };
    let input = [1.0f32; 32];
    for (projection, expected) in [
        (ExpertMlpProjection::Gate, 32.0),
        (ExpertMlpProjection::Up, 64.0),
        (ExpertMlpProjection::Down, 16.0),
    ] {
        let projected = expert0.project_cpu(projection, &input, 32).unwrap();
        for actual in projected {
            assert_close_with_tolerance(actual, expected, 0.01);
        }
    }
}

#[test]
fn native_q4_qwen35_expert_pack_uses_fixed_slot_layout() {
    let fixed = QwenMoeQ4ExpertLayout::qwen35_a17b();
    let native_input = |tensor: &str,
                        shape: Vec<usize>,
                        weight_kind: QwenMoeExpertComponentKind,
                        scale_kind: QwenMoeExpertComponentKind,
                        bias_kind: QwenMoeExpertComponentKind,
                        packed_byte: u8,
                        scale_byte: u8,
                        bias_byte: u8| {
        NativeQ4ExpertRecordInput {
            tensor: tensor.to_string(),
            dtype: "U32".to_string(),
            shape,
            source_offsets: [0, 1],
            source_hash: Some(format!("{tensor}:hash")),
            packed: vec![packed_byte; fixed.component(weight_kind).bytes],
            scale_bytes: vec![scale_byte; fixed.component(scale_kind).bytes],
            bias_bytes: vec![bias_byte; fixed.component(bias_kind).bytes],
            groups: fixed.component(scale_kind).bytes / 2,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        }
    };

    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_5_moe_text","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"head_dim":256,"num_key_value_heads":2,"vocab_size":248320,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":1024,"shared_expert_intermediate_size":1024}"#,
        )
        .unwrap();
    let layout = AggregateExpertLayout::new(
        config.experts(),
        config.hidden_size,
        config.moe_intermediate_size.unwrap(),
    )
    .unwrap();
    let (packed, metadata) = build_fixed_native_q4_expert_pack(
        0,
        7,
        FixedQ4ExpertSlotSpec::new(fixed, layout.hidden, layout.intermediate).unwrap(),
        vec![
            native_input(
                "model.layers.0.mlp.experts.7.gate_proj.weight",
                vec![1024, 4096],
                QwenMoeExpertComponentKind::GateWeight,
                QwenMoeExpertComponentKind::GateScale,
                QwenMoeExpertComponentKind::GateBias,
                0x11,
                0x22,
                0x33,
            ),
            native_input(
                "model.layers.0.mlp.experts.7.up_proj.weight",
                vec![1024, 4096],
                QwenMoeExpertComponentKind::UpWeight,
                QwenMoeExpertComponentKind::UpScale,
                QwenMoeExpertComponentKind::UpBias,
                0x44,
                0x55,
                0x66,
            ),
            native_input(
                "model.layers.0.mlp.experts.7.down_proj.weight",
                vec![4096, 1024],
                QwenMoeExpertComponentKind::DownWeight,
                QwenMoeExpertComponentKind::DownScale,
                QwenMoeExpertComponentKind::DownBias,
                0x77,
                0x88,
                0x99,
            ),
        ],
    )
    .unwrap();

    assert_eq!(layout.hidden, 4096);
    assert_eq!(layout.intermediate, 1024);
    assert_eq!(packed.len(), fixed.expert_bytes);
    assert!(!packed.starts_with(PBQ4_EXPERT_MAGIC));
    assert_eq!(metadata.layer, 0);
    assert_eq!(metadata.expert, 7);
    assert_eq!(metadata.packed_bytes, fixed.expert_bytes as u64);
    assert_eq!(metadata.records.len(), 3);
    assert_eq!(
        &packed[fixed
            .component(QwenMoeExpertComponentKind::GateWeight)
            .offset
            ..fixed
                .component(QwenMoeExpertComponentKind::GateWeight)
                .offset
                + 4],
        &[0x11; 4]
    );
    assert_eq!(
        &packed[fixed.component(QwenMoeExpertComponentKind::UpScale).offset
            ..fixed.component(QwenMoeExpertComponentKind::UpScale).offset + 4],
        &[0x55; 4]
    );
    assert_eq!(
        &packed[fixed.component(QwenMoeExpertComponentKind::DownBias).offset
            ..fixed.component(QwenMoeExpertComponentKind::DownBias).offset + 4],
        &[0x99; 4]
    );
}

#[test]
fn aggregate_expert_reuse_rejects_changed_source_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
    fs::create_dir_all(&snapshot).unwrap();
    let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
    fs::create_dir_all(&plan.experts_dir).unwrap();

    let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":2,"num_attention_heads":1,"vocab_size":16,"torch_dtype":"bfloat16","num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":2}"#,
        )
        .unwrap();
    let tensors = vec![
        ExpertTensorRef {
            tensor: "model.layers.1.mlp.experts.gate_up_proj".to_string(),
            shard: "expert.safetensors".to_string(),
            layer: Some(1),
            expert: None,
            dtype: Some("U8".to_string()),
            shape: vec![2, 4, 2],
            source_offsets: Some([0, 16]),
            q4_sources: None,
        },
        ExpertTensorRef {
            tensor: "model.layers.1.mlp.experts.down_proj".to_string(),
            shard: "expert.safetensors".to_string(),
            layer: Some(1),
            expert: None,
            dtype: Some("U8".to_string()),
            shape: vec![2, 2, 2],
            source_offsets: Some([16, 24]),
            q4_sources: None,
        },
    ];

    let write_expert_shard = |gate_up_bytes: Vec<u8>, down_bytes: Vec<u8>| {
        fs::write(
            snapshot.join("expert.safetensors"),
            make_typed_safetensors(&[
                (
                    "model.layers.1.mlp.experts.gate_up_proj",
                    "U8",
                    vec![2, 4, 2],
                    gate_up_bytes.as_slice(),
                ),
                (
                    "model.layers.1.mlp.experts.down_proj",
                    "U8",
                    vec![2, 2, 2],
                    down_bytes.as_slice(),
                ),
            ]),
        )
        .unwrap();
    };

    write_expert_shard((0u8..16).collect(), (16u8..24).collect());
    pack_expert_tensors(
        &snapshot,
        ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
        &tensors,
        Some(&config),
    )
    .unwrap();
    let before = read_expert_pack_metadata(&plan.experts_dir, 1, 0)
        .unwrap()
        .unwrap()
        .records[0]
        .source_hash
        .clone()
        .unwrap();

    write_expert_shard((100u8..116).collect(), (200u8..208).collect());
    pack_expert_tensors(
        &snapshot,
        ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
        &tensors,
        Some(&config),
    )
    .unwrap();
    let after = read_expert_pack_metadata(&plan.experts_dir, 1, 0)
        .unwrap()
        .unwrap()
        .records[0]
        .source_hash
        .clone()
        .unwrap();

    assert_ne!(before, after);
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
    let out = q4_fma_matvec(payload.packed, &input, payload.scales, payload.biases, 1, 8).unwrap();
    assert!((out[0] - 36.0).abs() < 1.0, "decoded q4 sum was {}", out[0]);
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
