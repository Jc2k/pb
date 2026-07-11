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
use std::sync::Arc;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use anyhow::{Context, Result};
#[cfg(test)]
use std::collections::BTreeMap;

#[cfg(test)]
use super::cache::*;
#[cfg(test)]
use super::capabilities::FlashMoeCapabilityPlan;
#[cfg(test)]
use super::experts::*;
use super::experts::{
    ExpertMlpProjection, ExpertRawPayload, PackedExpertTensor, fixed_q4_payload_from_pbq4_records,
    parse_pbq4_expert_pack,
};
#[cfg(test)]
use super::math::*;
#[cfg(test)]
use super::metal::MetalBatchProjectionInput;
#[cfg(test)]
use super::metal::MetalExecutionContext;
#[cfg(test)]
use super::model_family::QwenMoeExpertComponentKind;
#[cfg(test)]
use super::model_family::QwenMoeLayerKind;
#[cfg(test)]
use super::model_family::QwenMoeModelLayout;
#[cfg(test)]
use super::model_family::QwenMoeQ4ExpertLayout;
use super::model_family::{QwenModelConfig, QwenMoeFamily};
#[cfg(test)]
use super::planning::*;
#[cfg(test)]
use super::runtime::MetalExecutionFacade;
#[cfg(test)]
use super::scheduler::FlashMoeScheduledGraph;
#[cfg(test)]
use super::scheduler::ScheduledRoutingCandidateSource;
#[cfg(test)]
use super::state::KvCache;
use super::state::LinearAttentionLayout;
#[cfg(test)]
use super::state::{
    FlashMoeRecurrentLayerState, FlashMoeSessionState, FlashMoeStatePlacement,
    reusable_session_prefix_len, stable_session_cache_tokens, take_reusable_session_cache_entry,
};
#[cfg(test)]
use super::text::*;
use super::types::*;
#[cfg(test)]
use super::vision::{
    ExpandedVisionPrompt, ImagePlaceholderSpec, ImagePreprocessor, MropePosition, VisualTokenSpan,
    block_major_patch_coords, expand_multimodal_image_placeholders,
    expand_single_image_placeholders, qwen3vl_multimodal_mrope_positions,
    qwen3vl_single_image_mrope_positions, token_run_bounds,
};
#[cfg(test)]
use super::weights::*;
#[cfg(test)]
const DENSE_Q4_GROUP_SIZE: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_safetensors(tensors: &[(&str, &[u8])]) -> Vec<u8> {
        let typed: Vec<(&str, &str, Vec<usize>, &[u8])> = tensors
            .iter()
            .map(|(name, bytes)| (*name, "U8", vec![bytes.len()], *bytes))
            .collect();
        make_typed_safetensors(&typed)
    }

    fn make_typed_safetensors(tensors: &[(&str, &str, Vec<usize>, &[u8])]) -> Vec<u8> {
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

    fn f32_tensor_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn bf16_tensor_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
        for value in values {
            let bits = (value.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        bytes
    }

    fn u32_tensor_bytes(values: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn test_expert_triplet(
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

    fn typed_fixture_refs(
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

    fn expert_triplet_weight_map(layer: usize, expert: usize) -> String {
        format!(
            r#"{{"weight_map":{{"model.layers.0.self_attn.q_proj.weight":"dense.safetensors","model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight":"expert.safetensors","model.layers.{layer}.mlp.experts.{expert}.up_proj.weight":"expert.safetensors","model.layers.{layer}.mlp.experts.{expert}.down_proj.weight":"expert.safetensors"}}}}"#
        )
    }

    fn write_test_config(snapshot: &Path) {
        std::fs::write(
            snapshot.join("config.json"),
            br#"{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "{actual:.8} != {expected:.8}"
        );
    }

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    fn assert_close_with_tolerance(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual:.8} != {expected:.8} within {tolerance:.8}"
        );
    }

    #[derive(Debug, Clone, Copy)]
    enum FlashMoeFixtureFamily {
        Qwen35FlashMoe,
        Qwen3Moe,
        Qwen3VlMoe,
    }

    impl FlashMoeFixtureFamily {
        fn model(self) -> &'static str {
            match self {
                Self::Qwen35FlashMoe => QWEN35_MODEL,
                Self::Qwen3Moe => "hf://Qwen/Qwen3-30B-A3B",
                Self::Qwen3VlMoe => QWEN3_VL_MODEL,
            }
        }

        fn config_json(self) -> &'static [u8] {
            match self {
                Self::Qwen35FlashMoe => {
                    br#"{
  "model_type": "qwen3_5_moe",
  "architectures": ["Qwen3_5MoeForCausalLM"],
  "num_hidden_layers": 60,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 2,
  "vocab_size": 248320,
  "rope_theta": 10000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "moe_intermediate_size": 1024,
  "num_shared_experts": 1,
  "shared_expert_intermediate_size": 1024
}"#
                }
                Self::Qwen3Moe => {
                    br#"{
  "model_type": "qwen3_moe",
  "architectures": ["Qwen3MoeForCausalLM"],
  "num_hidden_layers": 2,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 151936,
  "rope_theta": 1000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 512,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 1536
}"#
                }
                Self::Qwen3VlMoe => {
                    br#"{
  "model_type": "qwen3_vl_moe",
  "architectures": ["Qwen3VLMoeForConditionalGeneration"],
  "num_hidden_layers": 2,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 248320,
  "rope_theta": 1000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 512,
  "num_experts_per_tok": 3,
  "moe_intermediate_size": 1536,
  "rope_scaling": {"mrope_section": [24, 20, 20]},
  "vision_config": {
    "depth": 1,
    "hidden_size": 64,
    "num_heads": 4,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "out_hidden_size": 4096
  }
}"#
                }
            }
        }

        fn config(self) -> QwenModelConfig {
            serde_json::from_slice(self.config_json()).unwrap()
        }
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
        let fixed =
            fixed_q4_payload_from_pbq4_records(layer, expert, spec, &records, None).unwrap();

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
        let gated = DenseTransformerRuntime::from_registry(
            &config,
            &TensorRegistry::from_manifest(&manifest),
        )
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
    fn qwen3vl_parity_multimodal_prompt_image_tokens_and_mrope_goldens() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_qwen3vl_tokenizer_json(),
            Some(test_qwen3vl_tool_tokenizer_config_json()),
        )
        .unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[ChatMessage {
                    role: ChatRole::User,
                    content: ChatMessageContent::Parts(vec![ChatContentPart::Image {
                        image: Some("fixture.png".to_string()),
                        placeholder_tokens: None,
                    }]),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                }],
                &[],
                true,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        );

        let temp = tempfile::tempdir().unwrap();
        let image_file = temp.path().join("qwen3vl_fixture.png");
        let image = image::RgbImage::from_fn(84, 56, |x, y| {
            image::Rgb([(x % 251) as u8, (y % 251) as u8, ((x + y) % 251) as u8])
        });
        image.save(&image_file).unwrap();

        let preprocessor = ImagePreprocessor::default_qwen3_vl();
        let (patch_grid_h, patch_grid_w, patches) = preprocessor.preprocess(&image_file).unwrap();
        assert_eq!((patch_grid_h, patch_grid_w), (4, 6));
        assert_eq!(
            patches.len(),
            patch_grid_h * patch_grid_w * preprocessor.patch_flat_dim()
        );
        let visual_grid_h = patch_grid_h / preprocessor.merge_size;
        let visual_grid_w = patch_grid_w / preprocessor.merge_size;
        let visual_tokens = visual_grid_h * visual_grid_w;
        assert_eq!((visual_grid_h, visual_grid_w, visual_tokens), (2, 3, 6));

        let vision_start = tokenizer.token_id("<|vision_start|>").unwrap();
        let vision_end = tokenizer.token_id("<|vision_end|>").unwrap();
        let image_pad = tokenizer.token_id("<|image_pad|>").unwrap();
        let prompt_tokens = tokenizer.encode(&rendered).unwrap();
        assert_eq!(token_run_bounds(&prompt_tokens, image_pad), vec![(3, 4, 1)]);

        let expanded = expand_multimodal_image_placeholders(
            prompt_tokens,
            vision_start,
            vision_end,
            image_pad,
            &[ImagePlaceholderSpec {
                token_count: visual_tokens,
                grid_h: visual_grid_h,
                grid_w: visual_grid_w,
            }],
        )
        .unwrap();
        assert_eq!(
            expanded.tokens,
            vec![100, 5, 200, 202, 202, 202, 202, 202, 202, 201, 101, 100, 6]
        );
        assert_eq!(
            expanded.visual_spans,
            vec![VisualTokenSpan::image(3, 9, 2, 3)]
        );

        let (positions, next_position) =
            qwen3vl_multimodal_mrope_positions(&expanded.tokens, image_pad, &expanded.visual_spans)
                .unwrap();
        assert_eq!(
            &positions[..3],
            &[
                MropePosition::text(0),
                MropePosition::text(1),
                MropePosition::text(2)
            ]
        );
        assert_eq!(
            &positions[3..9],
            &[
                MropePosition {
                    temporal: 3,
                    height: 3,
                    width: 3,
                },
                MropePosition {
                    temporal: 3,
                    height: 3,
                    width: 4,
                },
                MropePosition {
                    temporal: 3,
                    height: 3,
                    width: 5,
                },
                MropePosition {
                    temporal: 3,
                    height: 4,
                    width: 3,
                },
                MropePosition {
                    temporal: 3,
                    height: 4,
                    width: 4,
                },
                MropePosition {
                    temporal: 3,
                    height: 4,
                    width: 5,
                },
            ]
        );
        assert_eq!(
            &positions[9..],
            &[
                MropePosition::text(6),
                MropePosition::text(7),
                MropePosition::text(8),
                MropePosition::text(9)
            ]
        );
        assert_eq!(next_position, 10);
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
        assert!(plan.tensor_manifest.is_file());
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
    fn qwen_config_validates_runtime_dimensions() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":4}"#,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.kv_heads(), 8);
        assert_eq!(config.experts(), 512);
        assert_eq!(config.config_active_experts(), 4);
    }

    #[test]
    fn qwen_config_accepts_arbitrary_num_experts_per_tok() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.config_active_experts(), 10);
    }

    #[test]
    fn routing_policy_defaults_qwen35_flashmoe_profile_to_k4() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_5_moe","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":2,"vocab_size":248320,"rope_theta":10000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();

        let policy = FlashMoeRoutingPolicy::default()
            .resolve(QWEN35_MODEL, &config)
            .unwrap();

        assert_eq!(policy.active_experts, 4);
        assert_eq!(policy.source, ActiveExpertsSource::Qwen35FlashMoeProfile);
    }

    #[test]
    fn routing_policy_defaults_other_qwen_moe_to_model_config_k() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":2}"#,
        )
        .unwrap();

        let policy = FlashMoeRoutingPolicy::default()
            .resolve("hf://Qwen/Qwen3-30B-A3B", &config)
            .unwrap();

        assert_eq!(policy.active_experts, 2);
        assert_eq!(policy.source, ActiveExpertsSource::ModelConfig);
    }

    #[test]
    fn flashmoe_parity_routing_defaults_are_model_family_aware() {
        let qwen35 = FlashMoeRoutingPolicy::default()
            .resolve(
                FlashMoeFixtureFamily::Qwen35FlashMoe.model(),
                &FlashMoeFixtureFamily::Qwen35FlashMoe.config(),
            )
            .unwrap();
        assert_eq!(qwen35.active_experts, 4);
        assert_eq!(qwen35.source, ActiveExpertsSource::Qwen35FlashMoeProfile);

        for (family, expected_k) in [
            (FlashMoeFixtureFamily::Qwen3Moe, 2),
            (FlashMoeFixtureFamily::Qwen3VlMoe, 3),
        ] {
            let policy = FlashMoeRoutingPolicy::default()
                .resolve(family.model(), &family.config())
                .unwrap();
            assert_eq!(policy.active_experts, expected_k, "{family:?}");
            assert_eq!(policy.source, ActiveExpertsSource::ModelConfig);
        }
    }

    #[test]
    fn routing_policy_honors_explicit_active_expert_override() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":2}"#,
        )
        .unwrap();

        let policy = FlashMoeRoutingPolicy::new(Some(6), false)
            .resolve("hf://Qwen/Qwen3-30B-A3B", &config)
            .unwrap();

        assert_eq!(policy.active_experts, 6);
        assert_eq!(policy.source, ActiveExpertsSource::UserOverride);
    }

    #[test]
    fn routing_policy_guards_qwen35_k_below_four_unless_forced() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_5_moe","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":2,"vocab_size":248320,"rope_theta":10000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();

        let err = FlashMoeRoutingPolicy::new(Some(3), false)
            .resolve(QWEN35_MODEL, &config)
            .unwrap_err();
        assert!(err.to_string().contains("requires K >= 4"), "{err:#}");

        let forced = FlashMoeRoutingPolicy::new(Some(3), true)
            .resolve(QWEN35_MODEL, &config)
            .unwrap();
        assert_eq!(forced.active_experts, 3);
        assert!(forced.force_active_experts);
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

    fn byte_tokens(text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    fn weather_tool() -> ChatTool {
        ChatTool {
            name: "get_weather".to_string(),
            description: Some("Get weather.".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
        }
    }

    fn assistant_weather_tool_call(content: &str) -> ChatMessage {
        let mut assistant = ChatMessage::text(ChatRole::Assistant, content);
        assistant.tool_calls.push(ChatToolCall {
            id: None,
            name: "get_weather".to_string(),
            arguments: serde_json::json!({"city": "London"}),
        });
        assistant
    }

    fn weather_tool_result() -> ChatMessage {
        ChatMessage {
            role: ChatRole::Tool,
            content: ChatMessageContent::Text("{\"temp\":12}".to_string()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: Some("get_weather".to_string()),
        }
    }

    fn rendered_tool_prompt_pair(assistant: ChatMessage) -> (String, String) {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_qwen3_tool_tokenizer_config_json()),
        )
        .unwrap();
        let tool = weather_tool();
        let initial_messages = vec![
            ChatMessage::text(ChatRole::System, "be precise"),
            ChatMessage::text(ChatRole::User, "weather?"),
        ];
        let first_prompt = tokenizer
            .apply_chat_template_to_messages(&initial_messages, std::slice::from_ref(&tool), true)
            .unwrap();
        let mut next_messages = initial_messages;
        next_messages.push(assistant);
        next_messages.push(weather_tool_result());
        let next_prompt = tokenizer
            .apply_chat_template_to_messages(&next_messages, &[tool], true)
            .unwrap();
        (first_prompt, next_prompt)
    }

    #[test]
    fn session_cache_reuses_prompt_prefix_after_json_compat_tool_call() {
        let (first_prompt, next_prompt) =
            rendered_tool_prompt_pair(assistant_weather_tool_call(""));
        let first_prompt_tokens = byte_tokens(&first_prompt);
        let next_prompt_tokens = byte_tokens(&next_prompt);
        let mut old_cached_tokens = first_prompt_tokens.clone();
        old_cached_tokens.extend(byte_tokens(
            r#"{"type":"tool_call","tool":"get_weather","arguments":{"city":"London"},"thinking":"checking"}"#,
        ));

        assert_eq!(
            reusable_session_prefix_len(&old_cached_tokens, &next_prompt_tokens),
            None
        );
        let stable_cached_tokens = stable_session_cache_tokens(&first_prompt_tokens);
        assert_eq!(
            reusable_session_prefix_len(&stable_cached_tokens, &next_prompt_tokens),
            Some(first_prompt_tokens.len())
        );
    }

    #[test]
    fn session_cache_reuses_prompt_prefix_after_native_tool_call_rerender() {
        let (first_prompt, next_prompt) =
            rendered_tool_prompt_pair(assistant_weather_tool_call("checking"));
        let first_prompt_tokens = byte_tokens(&first_prompt);
        let next_prompt_tokens = byte_tokens(&next_prompt);
        let mut old_cached_tokens = first_prompt_tokens.clone();
        old_cached_tokens.extend(byte_tokens(
            "checking\n<tool_call>\n{\"arguments\":{\"city\":\"London\"},\"name\":\"get_weather\"}\n</tool_call>\n",
        ));

        assert_eq!(
            reusable_session_prefix_len(&old_cached_tokens, &next_prompt_tokens),
            None
        );
        let stable_cached_tokens = stable_session_cache_tokens(&first_prompt_tokens);
        assert_eq!(
            reusable_session_prefix_len(&stable_cached_tokens, &next_prompt_tokens),
            Some(first_prompt_tokens.len())
        );
    }

    #[test]
    fn session_cache_reuse_moves_state_and_shallow_snapshots_cpu_kv_cache() {
        let cached_tokens = vec![10, 20];
        let mut cache = KvCache::new(2, 2);
        for (position, token) in cached_tokens.iter().copied().enumerate() {
            cache.record_prompt_token(position, token).unwrap();
        }
        cache
            .record_kv(0, 0, vec![1.0, 1.1], vec![2.0, 2.1])
            .unwrap();
        cache
            .record_kv(1, 0, vec![3.0, 3.1], vec![4.0, 4.1])
            .unwrap();

        let mut sessions = BTreeMap::new();
        sessions.insert(
            "chat".to_string(),
            FlashMoeSessionState {
                tokens: cached_tokens,
                kv_cache: cache,
                last_hidden: vec![9.0, 9.1],
            },
        );

        let next_prompt = [10, 20, 30];
        let (prefix_len, state) =
            take_reusable_session_cache_entry(&mut sessions, "chat", &next_prompt).unwrap();
        assert!(sessions.is_empty());
        let FlashMoeSessionState {
            tokens,
            mut kv_cache,
            last_hidden,
        } = state;
        assert_eq!(prefix_len, tokens.len());
        assert_eq!(last_hidden, vec![9.0, 9.1]);

        kv_cache.resize_capacity(next_prompt.len());
        let snapshot = kv_cache.shallow_snapshot();
        assert_eq!(snapshot.keys_values(1, 0).unwrap().len(), 2);
        kv_cache
            .record_kv(2, 0, vec![5.0, 5.1], vec![6.0, 6.1])
            .unwrap();
        assert_eq!(snapshot.keys_values(2, 0).unwrap().len(), 2);
    }

    #[test]
    fn recurrent_layer_state_recording_rejects_gpu_placement_without_fallback() {
        let mut cache = KvCache::new(2, 2);

        cache
            .record_recurrent_layer_state(FlashMoeRecurrentLayerState::cpu_visible(1, 0, 99))
            .unwrap();

        let err = cache
            .record_recurrent_layer_state(FlashMoeRecurrentLayerState::new(
                1,
                0,
                99,
                FlashMoeStatePlacement::GpuResident,
            ))
            .unwrap_err();
        assert!(
            err.to_string().contains("requires CpuVisible placement"),
            "{err:#}"
        );
    }

    #[test]
    fn session_prefix_reuse_preserves_cpu_kv_boundaries() {
        let cached_tokens = vec![10, 20];
        let mut cache = KvCache::new(2, 2);
        for (position, token) in cached_tokens.iter().copied().enumerate() {
            cache.record_prompt_token(position, token).unwrap();
        }
        cache
            .record_kv(0, 0, vec![1.0, 1.1], vec![2.0, 2.1])
            .unwrap();
        cache
            .record_kv(1, 0, vec![3.0, 3.1], vec![4.0, 4.1])
            .unwrap();

        let session_state = FlashMoeSessionState {
            tokens: cached_tokens,
            kv_cache: cache,
            last_hidden: vec![9.0, 9.1],
        };
        let next_prompt = [10, 20, 30];
        let prefix_len = reusable_session_prefix_len(&session_state.tokens, &next_prompt).unwrap();
        assert_eq!(prefix_len, session_state.tokens.len());

        let mut reused = session_state.kv_cache.shallow_snapshot();
        reused.resize_capacity(next_prompt.len());
        assert_eq!(reused.keys_values(1, 0).unwrap().len(), prefix_len);
        assert_eq!(reused.keys_values(2, 0).unwrap().len(), prefix_len);
        assert_eq!(session_state.last_hidden, vec![9.0, 9.1]);

        assert_eq!(
            reusable_session_prefix_len(&session_state.tokens, &[10]),
            None
        );
        assert_eq!(
            reusable_session_prefix_len(&session_state.tokens, &[10, 99, 30]),
            None
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
    fn qwen35_linear_attention_keeps_direct_qkv_projection_order() {
        let qwen35: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_5_moe_text","num_hidden_layers":1,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":2,"vocab_size":248320,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();
        assert!(!qwen35.linear_attention_qkv_projection_requires_reorder());

        let qwen_next: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_next","num_hidden_layers":1,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":2,"vocab_size":248320,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();
        assert!(qwen_next.linear_attention_qkv_projection_requires_reorder());
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
    fn qwen3vl_config_rejects_out_of_range_deepstack_index() {
        let json = br#"{
            "model_type": "qwen3_vl",
            "text_config": {
                "hidden_size": 128,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 1024
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 64,
                "num_heads": 4,
                "deepstack_visual_indexes": [0, 2]
            }
        }"#;

        let config: QwenModelConfig = serde_json::from_slice(json).unwrap();
        let err = config.validate().unwrap_err();
        assert!(
            err.to_string().contains("deepstack_visual_indexes"),
            "expected deepstack bounds error, got: {err:#}"
        );
    }

    #[test]
    fn qwen3vl_single_image_placeholder_is_expanded_in_place() {
        assert_eq!(
            expand_single_image_placeholders(vec![1, 9, 2], 7, 8, 9, 4).unwrap(),
            vec![1, 7, 9, 9, 9, 9, 8, 2]
        );
        assert_eq!(
            expand_single_image_placeholders(vec![1, 7, 9, 9, 8, 2], 7, 8, 9, 2).unwrap(),
            vec![1, 7, 9, 9, 8, 2]
        );
        assert_eq!(
            expand_single_image_placeholders(vec![1, 9, 9, 2], 7, 8, 9, 2).unwrap(),
            vec![1, 7, 9, 9, 8, 2]
        );
        assert!(expand_single_image_placeholders(vec![1, 2], 7, 8, 9, 2).is_err());
        assert!(expand_single_image_placeholders(vec![1, 9, 2, 9], 7, 8, 9, 2).is_err());
        assert!(expand_single_image_placeholders(vec![1, 7, 9, 2], 7, 8, 9, 2).is_err());
        assert!(qwen3vl_single_image_mrope_positions(&[1, 9, 2, 9], 9, 1, 2).is_err());
    }

    #[test]
    fn qwen3vl_placeholder_expansion_handles_explicit_and_implicit_spans() {
        let expanded = expand_multimodal_image_placeholders(
            vec![1, 7, 9, 9, 9, 9, 8, 2, 9, 3],
            7,
            8,
            9,
            &[
                ImagePlaceholderSpec {
                    token_count: 4,
                    grid_h: 2,
                    grid_w: 2,
                },
                ImagePlaceholderSpec {
                    token_count: 2,
                    grid_h: 1,
                    grid_w: 2,
                },
            ],
        )
        .unwrap();

        assert_eq!(expanded.tokens, vec![1, 7, 9, 9, 9, 9, 8, 2, 7, 9, 9, 8, 3]);
        assert_eq!(
            expanded.visual_spans,
            vec![
                VisualTokenSpan::image(2, 6, 2, 2),
                VisualTokenSpan::image(9, 11, 1, 2),
            ]
        );
    }

    #[test]
    fn qwen3vl_placeholder_expansion_rejects_clear_mismatches() {
        let err = expand_multimodal_image_placeholders(
            vec![1, 9, 2],
            7,
            8,
            9,
            &[ImagePlaceholderSpec {
                token_count: 5,
                grid_h: 2,
                grid_w: 3,
            }],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("image 0 visual token count 5 does not match merged grid 2x3 (6 tokens)"),
            "{err:#}"
        );

        let err = expand_multimodal_image_placeholders(
            vec![1, 7, 9, 9, 9, 8, 2],
            7,
            8,
            9,
            &[ImagePlaceholderSpec {
                token_count: 4,
                grid_h: 2,
                grid_w: 2,
            }],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("image 0 placeholder span contains 3 <|image_pad|> tokens but the encoded image produced 4 visual tokens; use one placeholder for implicit expansion or exactly one per visual token"),
            "{err:#}"
        );

        let err = expand_multimodal_image_placeholders(
            vec![1, 7, 9, 2],
            7,
            8,
            9,
            &[ImagePlaceholderSpec {
                token_count: 2,
                grid_h: 1,
                grid_w: 2,
            }],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("must be wrapped by both <|vision_start|> and <|vision_end|>"),
            "{err:#}"
        );

        let err = qwen3vl_multimodal_mrope_positions(
            &[9, 9, 9],
            9,
            &[VisualTokenSpan::image(0, 3, 2, 2)],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("image span 0 does not match its declared 2x2 merged grid"),
            "{err:#}"
        );

        let err =
            qwen3vl_multimodal_mrope_positions(&[9, 1], 9, &[VisualTokenSpan::image(0, 2, 1, 2)])
                .unwrap_err();
        assert!(
            err.to_string()
                .contains("image placeholder count 1 does not match expected visual token count 2"),
            "{err:#}"
        );
    }

    fn expand_and_position_for_test(
        tokens: Vec<u32>,
        image_specs: &[ImagePlaceholderSpec],
    ) -> (ExpandedVisionPrompt, Vec<MropePosition>, usize) {
        let expanded = expand_multimodal_image_placeholders(tokens, 7, 8, 9, image_specs).unwrap();
        let (positions, next_position) =
            qwen3vl_multimodal_mrope_positions(&expanded.tokens, 9, &expanded.visual_spans)
                .unwrap();
        (expanded, positions, next_position)
    }

    #[test]
    fn qwen3vl_text_before_image_gets_own_visual_span() {
        let (expanded, positions, next_position) = expand_and_position_for_test(
            vec![1, 9],
            &[ImagePlaceholderSpec {
                token_count: 4,
                grid_h: 2,
                grid_w: 2,
            }],
        );

        assert_eq!(expanded.tokens, vec![1, 7, 9, 9, 9, 9, 8]);
        assert_eq!(
            expanded.visual_spans,
            vec![VisualTokenSpan::image(2, 6, 2, 2)]
        );
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(positions[1], MropePosition::text(1));
        assert_eq!(
            &positions[2..6],
            &[
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 2,
                },
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 3,
                },
                MropePosition {
                    temporal: 2,
                    height: 3,
                    width: 2,
                },
                MropePosition {
                    temporal: 2,
                    height: 3,
                    width: 3,
                },
            ]
        );
        assert_eq!(positions[6], MropePosition::text(4));
        assert_eq!(next_position, 5);
    }

    #[test]
    fn qwen3vl_image_before_text_gets_own_visual_span() {
        let (expanded, positions, next_position) = expand_and_position_for_test(
            vec![9, 2],
            &[ImagePlaceholderSpec {
                token_count: 2,
                grid_h: 1,
                grid_w: 2,
            }],
        );

        assert_eq!(expanded.tokens, vec![7, 9, 9, 8, 2]);
        assert_eq!(
            expanded.visual_spans,
            vec![VisualTokenSpan::image(1, 3, 1, 2)]
        );
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(
            &positions[1..3],
            &[
                MropePosition {
                    temporal: 1,
                    height: 1,
                    width: 1,
                },
                MropePosition {
                    temporal: 1,
                    height: 1,
                    width: 2,
                },
            ]
        );
        assert_eq!(positions[3], MropePosition::text(3));
        assert_eq!(positions[4], MropePosition::text(4));
        assert_eq!(next_position, 5);
    }

    #[test]
    fn qwen3vl_text_image_text_advances_after_visual_grid() {
        let (expanded, positions, next_position) = expand_and_position_for_test(
            vec![1, 9, 2],
            &[ImagePlaceholderSpec {
                token_count: 4,
                grid_h: 2,
                grid_w: 2,
            }],
        );

        assert_eq!(expanded.tokens, vec![1, 7, 9, 9, 9, 9, 8, 2]);
        assert_eq!(
            expanded.visual_spans,
            vec![VisualTokenSpan::image(2, 6, 2, 2)]
        );
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(positions[1], MropePosition::text(1));
        assert_eq!(positions[6], MropePosition::text(4));
        assert_eq!(positions[7], MropePosition::text(5));
        assert_eq!(next_position, 6);
    }

    #[test]
    fn qwen3vl_two_images_get_separate_visual_spans() {
        let (expanded, positions, next_position) = expand_and_position_for_test(
            vec![1, 9, 2, 9, 3],
            &[
                ImagePlaceholderSpec {
                    token_count: 2,
                    grid_h: 1,
                    grid_w: 2,
                },
                ImagePlaceholderSpec {
                    token_count: 2,
                    grid_h: 2,
                    grid_w: 1,
                },
            ],
        );

        assert_eq!(expanded.tokens, vec![1, 7, 9, 9, 8, 2, 7, 9, 9, 8, 3]);
        assert_eq!(
            expanded.visual_spans,
            vec![
                VisualTokenSpan::image(2, 4, 1, 2),
                VisualTokenSpan::image(7, 9, 2, 1),
            ]
        );
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(positions[1], MropePosition::text(1));
        assert_eq!(
            &positions[2..4],
            &[
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 2,
                },
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 3,
                },
            ]
        );
        assert_eq!(positions[4], MropePosition::text(4));
        assert_eq!(positions[5], MropePosition::text(5));
        assert_eq!(positions[6], MropePosition::text(6));
        assert_eq!(
            &positions[7..9],
            &[
                MropePosition {
                    temporal: 7,
                    height: 7,
                    width: 7,
                },
                MropePosition {
                    temporal: 7,
                    height: 8,
                    width: 7,
                },
            ]
        );
        assert_eq!(positions[9], MropePosition::text(9));
        assert_eq!(positions[10], MropePosition::text(10));
        assert_eq!(next_position, 11);
    }

    #[test]
    fn qwen3vl_multiple_image_grids_with_different_dimensions_are_positioned() {
        let (expanded, positions, next_position) = expand_and_position_for_test(
            vec![1, 9, 2, 9, 3],
            &[
                ImagePlaceholderSpec {
                    token_count: 6,
                    grid_h: 2,
                    grid_w: 3,
                },
                ImagePlaceholderSpec {
                    token_count: 4,
                    grid_h: 1,
                    grid_w: 4,
                },
            ],
        );

        assert_eq!(
            expanded.tokens,
            vec![1, 7, 9, 9, 9, 9, 9, 9, 8, 2, 7, 9, 9, 9, 9, 8, 3]
        );
        assert_eq!(
            expanded.visual_spans,
            vec![
                VisualTokenSpan::image(2, 8, 2, 3),
                VisualTokenSpan::image(11, 15, 1, 4),
            ]
        );
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(positions[1], MropePosition::text(1));
        assert_eq!(
            &positions[2..8],
            &[
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 2,
                },
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 3,
                },
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 4,
                },
                MropePosition {
                    temporal: 2,
                    height: 3,
                    width: 2,
                },
                MropePosition {
                    temporal: 2,
                    height: 3,
                    width: 3,
                },
                MropePosition {
                    temporal: 2,
                    height: 3,
                    width: 4,
                },
            ]
        );
        assert_eq!(positions[8], MropePosition::text(5));
        assert_eq!(positions[9], MropePosition::text(6));
        assert_eq!(positions[10], MropePosition::text(7));
        assert_eq!(
            &positions[11..15],
            &[
                MropePosition {
                    temporal: 8,
                    height: 8,
                    width: 8,
                },
                MropePosition {
                    temporal: 8,
                    height: 8,
                    width: 9,
                },
                MropePosition {
                    temporal: 8,
                    height: 8,
                    width: 10,
                },
                MropePosition {
                    temporal: 8,
                    height: 8,
                    width: 11,
                },
            ]
        );
        assert_eq!(positions[15], MropePosition::text(12));
        assert_eq!(positions[16], MropePosition::text(13));
        assert_eq!(next_position, 14);
    }

    #[test]
    fn qwen3vl_parity_multiple_images_render_expand_and_position() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_qwen3vl_tokenizer_json(),
            Some(test_qwen3vl_tool_tokenizer_config_json()),
        )
        .unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[ChatMessage {
                    role: ChatRole::User,
                    content: ChatMessageContent::Parts(vec![
                        ChatContentPart::Text {
                            text: "describe ".to_string(),
                        },
                        ChatContentPart::Image {
                            image: Some("first.png".to_string()),
                            placeholder_tokens: None,
                        },
                        ChatContentPart::Text {
                            text: " now ".to_string(),
                        },
                        ChatContentPart::Image {
                            image: Some("second.png".to_string()),
                            placeholder_tokens: None,
                        },
                    ]),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                }],
                &[],
                true,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\ndescribe <|vision_start|><|image_pad|><|vision_end|> now <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        );

        let vision_start = tokenizer.token_id("<|vision_start|>").unwrap();
        let vision_end = tokenizer.token_id("<|vision_end|>").unwrap();
        let image_pad = tokenizer.token_id("<|image_pad|>").unwrap();
        let prompt_tokens = tokenizer.encode(&rendered).unwrap();
        assert_eq!(
            token_run_bounds(&prompt_tokens, image_pad),
            vec![(4, 5, 1), (8, 9, 1)]
        );

        let expanded = expand_multimodal_image_placeholders(
            prompt_tokens,
            vision_start,
            vision_end,
            image_pad,
            &[
                ImagePlaceholderSpec {
                    token_count: 4,
                    grid_h: 2,
                    grid_w: 2,
                },
                ImagePlaceholderSpec {
                    token_count: 2,
                    grid_h: 1,
                    grid_w: 2,
                },
            ],
        )
        .unwrap();
        assert_eq!(
            expanded.tokens,
            vec![
                100, 5, 7, 200, 202, 202, 202, 202, 201, 8, 200, 202, 202, 201, 101, 100, 6
            ]
        );
        assert_eq!(
            expanded.visual_spans,
            vec![
                VisualTokenSpan::image(4, 8, 2, 2),
                VisualTokenSpan::image(11, 13, 1, 2),
            ]
        );

        let (positions, next_position) =
            qwen3vl_multimodal_mrope_positions(&expanded.tokens, image_pad, &expanded.visual_spans)
                .unwrap();
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(positions[3], MropePosition::text(3));
        assert_eq!(
            &positions[4..8],
            &[
                MropePosition {
                    temporal: 4,
                    height: 4,
                    width: 4,
                },
                MropePosition {
                    temporal: 4,
                    height: 4,
                    width: 5,
                },
                MropePosition {
                    temporal: 4,
                    height: 5,
                    width: 4,
                },
                MropePosition {
                    temporal: 4,
                    height: 5,
                    width: 5,
                },
            ]
        );
        assert_eq!(positions[8], MropePosition::text(6));
        assert_eq!(positions[10], MropePosition::text(8));
        assert_eq!(
            &positions[11..13],
            &[
                MropePosition {
                    temporal: 9,
                    height: 9,
                    width: 9,
                },
                MropePosition {
                    temporal: 9,
                    height: 9,
                    width: 10,
                },
            ]
        );
        assert_eq!(positions[16], MropePosition::text(14));
        assert_eq!(next_position, 15);
    }

    #[test]
    fn qwen3vl_smart_resize_obeys_pixel_budget_after_rounding() {
        let preprocessor = ImagePreprocessor::default_qwen3_vl();
        let (h, w) = preprocessor.smart_resize(10_000, 10_000);
        assert_eq!(h % VIT_SPATIAL_MERGE_SIZE as u32, 0);
        assert_eq!(w % VIT_SPATIAL_MERGE_SIZE as u32, 0);
        assert!((h as usize) * (w as usize) <= preprocessor.max_pixels);

        let (small_h, small_w) = preprocessor.smart_resize(1, 1);
        assert!((small_h as usize) * (small_w as usize) >= preprocessor.min_pixels);
    }

    #[test]
    fn qwen3vl_vision_patch_coords_are_block_major() {
        assert_eq!(
            block_major_patch_coords(4, 4, 2),
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 1),
                (2, 2),
                (2, 3),
                (3, 2),
                (3, 3),
            ]
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

        let (manifest, visual_refs) = build_manifest(QWEN35_MODEL, &snapshot, &index_path).unwrap();
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
            fixed,
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
            mrope_section: None,
            tie_word_embeddings: None,
            num_shared_experts: None,
            shared_expert_intermediate_size: None,
            vision_config: None,
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

        let routing_command = store
            .router_command_with_metal(None, command, &[0.5, -1.0, 2.0])
            .unwrap();

        assert_eq!(
            routing_command.source,
            ScheduledRoutingCandidateSource::CpuRouterScores
        );
        assert_eq!(routing_command.layer, 0);
        assert_eq!(routing_command.active_experts, 1);
        assert_eq!(routing_command.routes, vec![(1, 9.0)]);
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
            (layout.row_packed_bytes + layout.groups_per_row * 2 * std::mem::size_of::<f32>())
                as u64
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
            let quantized = quantize_q4(&values, &shape, group_size).unwrap();
            let layout = dense_q4_layout(&shape, group_size).unwrap();
            let runtime_offset = bytes.len() as u64;
            bytes.extend_from_slice(&quantized.values);
            for scale in &quantized.scales {
                bytes.extend_from_slice(&scale.to_le_bytes());
            }
            for bias in &quantized.biases {
                bytes.extend_from_slice(&bias.to_le_bytes());
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
        let cols = 16;
        let group_size = 8;
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
                        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
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
            mrope_section: None,
            tie_word_embeddings: None,
            num_shared_experts: None,
            shared_expert_intermediate_size: None,
            vision_config: None,
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
                    (*actual - *expected).abs() < 1e-4,
                    "projection {projection_idx} row {row}: Metal q4 batch mmap {actual} diverged from CPU reference {expected}"
                );
            }
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
            mrope_section: None,
            tie_word_embeddings: None,
            num_shared_experts: None,
            shared_expert_intermediate_size: None,
            vision_config: None,
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
                mrope_section: None,
                tie_word_embeddings: None,
                num_shared_experts: None,
                shared_expert_intermediate_size: None,
                vision_config: None,
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
                        ((out_values.len() + router_values.len()) * std::mem::size_of::<f32>())
                            as u64,
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
            mrope_section: None,
            tie_word_embeddings: None,
            num_shared_experts: None,
            shared_expert_intermediate_size: None,
            vision_config: None,
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

        let (manifest, visual_refs) = build_manifest(QWEN35_MODEL, snapshot, &index_path).unwrap();
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

        let (manifest, visual_refs) = build_manifest(QWEN35_MODEL, snapshot, &index_path).unwrap();
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
        write_dense_tensor_store(snapshot, &dense_path, &manifest.dense_tensors).unwrap();
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

        let (manifest, visual_refs) = build_manifest(QWEN35_MODEL, snapshot, &index_path).unwrap();

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
            .lm_head_logits("lm_head.weight", &[1.0, 1.0], &tokenizer)
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
            .lm_head_logits("lm_head.weight", &[1.0, 1.0], &tokenizer)
            .unwrap();

        assert_eq!(logits, vec![2.0, 4.0, 6.0]);
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

    fn test_expert_pack(name: &str) -> Vec<u8> {
        let mut pack = Vec::new();
        pack.extend_from_slice(PBQ4_EXPERT_MAGIC);
        pack.extend_from_slice(&(name.len() as u32).to_le_bytes());
        pack.extend_from_slice(name.as_bytes());
        pack.extend_from_slice(&2u64.to_le_bytes());
        pack.extend_from_slice(&1u64.to_le_bytes());
        pack.extend_from_slice(&0.5f32.to_le_bytes());
        pack.extend_from_slice(&1.0f32.to_le_bytes());
        pack.extend_from_slice(&[0x21, 0x43]);
        pack
    }

    fn test_expert_pack_metadata(
        layer: usize,
        expert: usize,
        tensor: &str,
        packed_bytes: usize,
    ) -> ExpertPackMetadata {
        ExpertPackMetadata {
            layer,
            expert,
            packed_bytes: packed_bytes as u64,
            records: vec![ExpertPackRecord {
                tensor: tensor.to_string(),
                dtype: "F32".to_string(),
                shape: vec![1, 4],
                source_offsets: [0, 4],
                source_hash: Some(format!("hash-{layer}-{expert}")),
                record_offset: PBQ4_EXPERT_MAGIC.len() as u64,
                packed_bytes: 2,
                groups: 1,
                group_size: GROUP_SIZE,
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
            }],
        }
    }

    fn write_test_expert_layer(
        root: &Path,
        layer: usize,
        packs: Vec<(usize, Vec<u8>, ExpertPackMetadata)>,
        experts: usize,
    ) -> Result<()> {
        let slot_size = packs
            .iter()
            .map(|(_, pack, _)| pack.len() as u64)
            .max()
            .unwrap_or(1)
            .max(1);
        let path = expert_layer_path(root, layer);
        let file = fs::File::create(&path)
            .with_context(|| format!("failed to create test layer {}", path.display()))?;
        file.set_len((experts as u64) * slot_size)?;
        let mut metadata = Vec::new();
        for (expert, pack, mut pack_metadata) in packs {
            pack_metadata.packed_bytes = pack.len() as u64;
            write_all_at_positioned(&file, &pack, expert_slot_offset(expert, slot_size)?)?;
            metadata.push(pack_metadata);
        }
        let layer_metadata = ExpertLayerPackMetadata::new(layer, slot_size, experts, metadata);
        fs::write(
            expert_layer_metadata_path(root, layer),
            serde_json::to_vec(&layer_metadata)?,
        )?;
        Ok(())
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
