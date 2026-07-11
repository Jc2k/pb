//! Shared binary fixtures for owner-local FlashMoe tests.

use std::path::Path;

pub(super) fn make_safetensors(tensors: &[(&str, &[u8])]) -> Vec<u8> {
    let typed: Vec<(&str, &str, Vec<usize>, &[u8])> = tensors
        .iter()
        .map(|(name, bytes)| (*name, "U8", vec![bytes.len()], *bytes))
        .collect();
    make_typed_safetensors(&typed)
}

pub(super) fn make_typed_safetensors(tensors: &[(&str, &str, Vec<usize>, &[u8])]) -> Vec<u8> {
    let mut offset = 0usize;
    let mut entries = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let end = offset + bytes.len();
        entries.insert(
            (*name).to_string(),
            serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,end]}),
        );
        data.extend_from_slice(bytes);
        offset = end;
    }
    let header = serde_json::Value::Object(entries).to_string().into_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    out
}

pub(super) fn f32_tensor_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub(super) fn bf16_tensor_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
    for value in values {
        bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    bytes
}

pub(super) fn u32_tensor_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub(super) fn test_expert_triplet(
    layer: usize,
    expert: usize,
) -> Vec<(String, String, Vec<usize>, Vec<u8>)> {
    let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
    vec![
        (
            format!("{prefix}.gate_proj.weight"),
            "U8".to_string(),
            vec![16, 8],
            vec![1; 16 * 8],
        ),
        (
            format!("{prefix}.up_proj.weight"),
            "U8".to_string(),
            vec![16, 8],
            vec![2; 16 * 8],
        ),
        (
            format!("{prefix}.down_proj.weight"),
            "U8".to_string(),
            vec![8, 16],
            vec![3; 8 * 16],
        ),
    ]
}

pub(super) fn typed_fixture_refs(
    tensors: &[(String, String, Vec<usize>, Vec<u8>)],
) -> Vec<(&str, &str, Vec<usize>, &[u8])> {
    tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                name.as_str(),
                dtype.as_str(),
                shape.clone(),
                bytes.as_slice(),
            )
        })
        .collect()
}

pub(super) fn expert_triplet_weight_map(layer: usize, expert: usize) -> String {
    format!(
        r#"{{"weight_map":{{"model.layers.0.self_attn.q_proj.weight":"dense.safetensors","model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight":"expert.safetensors","model.layers.{layer}.mlp.experts.{expert}.up_proj.weight":"expert.safetensors","model.layers.{layer}.mlp.experts.{expert}.down_proj.weight":"expert.safetensors"}}}}"#
    )
}

pub(super) fn write_test_config(snapshot: &Path) {
    std::fs::write(
        snapshot.join("config.json"),
        br#"{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
    )
    .unwrap();
}

pub(super) fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-5,
        "{actual:.8} != {expected:.8}"
    );
}

pub(super) fn assert_close_with_tolerance(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:.8} != {expected:.8} within {tolerance:.8}"
    );
}
