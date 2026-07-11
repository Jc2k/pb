//! Owner-local PBQ4 import and fixed-slot compatibility parity tests.

use super::*;
use crate::inference::flashmoe::math::q4_fma_matvec_with_group_size;
use crate::inference::flashmoe::model_family::{QwenMoeExpertComponentKind, QwenMoeQ4ExpertLayout};
use crate::inference::flashmoe::test_fixtures::{
    assert_close, test_expert_pack, test_expert_pack_metadata, write_test_expert_layer,
};

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
