use super::*;
use crate::inference::flashmoe::experts::{
    DenseExpertDtype, EXPERT_SCALE_BIAS_DTYPE_F32, ExpertLayerPackMetadata, ExpertPackMetadata,
    ExpertPackRecord, ExpertRawPayload, ExpertSlotSpec, ExpertStoreExecutionDescriptor,
    FixedDenseExpertPayload, FixedDenseExpertSlotSpec, FixedQ4ExpertPayload, FixedQ4ExpertSlotSpec,
    PBQ4_EXPERT_MAGIC, expert_layer_path, write_expert_metadata_atomically,
};
use crate::inference::flashmoe::math::causal_attention;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::inference::flashmoe::metal::{
    MetalExecutionContext, MetalPostAttentionPrep, MetalScheduledCmd3Builder,
};
use crate::inference::flashmoe::model_family::{
    QwenMoeExpertComponentKind, QwenMoeExpertComponentLayout, QwenMoeQ4ExpertLayout,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::inference::flashmoe::runtime::ExpertPhaseInput;
use crate::inference::flashmoe::state::{
    FlashMoeExpertPhaseApplication, FlashMoeGpuBufferDescriptor, FlashMoeTokenState,
};
use crate::inference::flashmoe::weights::{
    DenseMmapMatvecProjection, DenseQ4MmapMatvecProjection, ResidentMmapMatvecProjection,
    RouterScoreProjectionBinding, RouterScoreProjectionDescriptor,
    SharedExpertPhaseResidentProjections, SharedExpertPhaseWeights,
};
use crate::inference::flashmoe::{GROUP_SIZE, QWEN35_MODEL, QwenModelConfig, QwenMoeModelLayout};
use std::{fs, path::Path};

fn qwen35_layout() -> QwenMoeModelLayout {
    let config: QwenModelConfig = serde_json::from_slice(
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
}"#,
    )
    .unwrap();
    QwenMoeModelLayout::from_config(QWEN35_MODEL, &config).unwrap()
}

fn qwen3_moe_layout() -> QwenMoeModelLayout {
    let config: QwenModelConfig = serde_json::from_slice(
        br#"{
  "model_type": "qwen3_moe",
  "architectures": ["Qwen3MoeForCausalLM"],
  "num_hidden_layers": 48,
  "hidden_size": 2048,
  "num_attention_heads": 32,
  "head_dim": 128,
  "num_key_value_heads": 4,
  "vocab_size": 151936,
  "rope_theta": 1000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 768,
  "norm_topk_prob": true
}"#,
    )
    .unwrap();
    QwenMoeModelLayout::from_config("hf://Qwen/Qwen3-30B-A3B", &config).unwrap()
}

fn pbq4_import_store(experts: &[usize]) -> (tempfile::TempDir, ExpertSlotStore) {
    let temp = tempfile::tempdir().unwrap();
    let mut packs = Vec::new();
    for expert in experts {
        let tensor = format!("model.layers.0.mlp.experts.{expert}.down_proj.weight");
        let mut bytes = PBQ4_EXPERT_MAGIC.to_vec();
        let record_offset = bytes.len() as u64;
        bytes.extend_from_slice(&(tensor.len() as u32).to_le_bytes());
        bytes.extend_from_slice(tensor.as_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&[0x21, 0x43]);
        let metadata = ExpertPackMetadata {
            layer: 0,
            expert: *expert,
            packed_bytes: bytes.len() as u64,
            records: vec![ExpertPackRecord {
                tensor,
                dtype: "F32".to_string(),
                shape: vec![1, 4],
                source_offsets: [0, 4],
                source_hash: Some(format!("fixture-{expert}")),
                record_offset,
                packed_bytes: 2,
                groups: 1,
                group_size: GROUP_SIZE,
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
            }],
        };
        packs.push((*expert, bytes, metadata));
    }
    let slot_size = packs
        .iter()
        .map(|(_, bytes, _)| bytes.len())
        .max()
        .unwrap_or(1);
    let expert_count = experts.iter().copied().max().unwrap_or(0) + 1;
    let mut layer = vec![0; slot_size * expert_count];
    let mut metadata = Vec::new();
    for (expert, bytes, pack) in packs {
        let offset = expert * slot_size;
        layer[offset..offset + bytes.len()].copy_from_slice(&bytes);
        metadata.push(pack);
    }
    fs::write(expert_layer_path(temp.path(), 0), layer).unwrap();
    write_expert_metadata_atomically(
        temp.path(),
        0,
        &ExpertLayerPackMetadata::new(0, slot_size as u64, expert_count, metadata),
    )
    .unwrap();
    let store = ExpertSlotStore::open(temp.path().to_path_buf()).unwrap();
    (temp, store)
}

fn tiny_fixed_q4_layout() -> QwenMoeQ4ExpertLayout {
    use QwenMoeExpertComponentKind::*;
    QwenMoeQ4ExpertLayout {
        expert_bytes: 48,
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
                bytes: 4,
            },
            QwenMoeExpertComponentLayout {
                kind: DownBias,
                offset: 44,
                bytes: 4,
            },
        ],
    }
}

fn raw_pbq4_read(layer: usize, expert: usize, payload: Vec<u8>) -> ExpertRawRead {
    ExpertRawRead {
        layer,
        expert,
        slot: ExpertSlotDescriptor {
            layer,
            expert,
            slot_offset: 1024,
            slot_capacity: payload.len(),
            payload_len: payload.len(),
        },
        metadata: ExpertPackMetadata {
            layer,
            expert,
            packed_bytes: payload.len() as u64,
            records: Vec::new(),
        },
        payload: ExpertRawPayload::Pbq4(payload),
        read_latency: Duration::from_millis(7),
        read_path: ExpertReadPath::PositionedRead,
    }
}

fn raw_fixed_q4_read(layer: usize, expert: usize) -> ExpertRawRead {
    let fixed_q4 = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
    let payload = FixedQ4ExpertPayload::from_whole_slot(
        fixed_q4,
        vec![0; fixed_q4.layout.expert_bytes],
        None,
    )
    .unwrap();
    ExpertRawRead {
        layer,
        expert,
        slot: ExpertSlotDescriptor {
            layer,
            expert,
            slot_offset: 512,
            slot_capacity: fixed_q4.layout.expert_bytes,
            payload_len: fixed_q4.layout.expert_bytes,
        },
        metadata: ExpertPackMetadata {
            layer,
            expert,
            packed_bytes: fixed_q4.layout.expert_bytes as u64,
            records: Vec::new(),
        },
        payload: ExpertRawPayload::FixedQ4(payload),
        read_latency: Duration::from_millis(3),
        read_path: ExpertReadPath::PositionedRead,
    }
}

fn raw_fixed_dense_read(layer: usize, expert: usize, dtype: DenseExpertDtype) -> ExpertRawRead {
    let spec = FixedDenseExpertSlotSpec::new(dtype, 2, 2).unwrap();
    let payload =
        FixedDenseExpertPayload::from_whole_slot(spec, vec![0; spec.expert_bytes], None).unwrap();
    ExpertRawRead {
        layer,
        expert,
        slot: ExpertSlotDescriptor {
            layer,
            expert,
            slot_offset: 512,
            slot_capacity: spec.expert_bytes,
            payload_len: spec.expert_bytes,
        },
        metadata: ExpertPackMetadata {
            layer,
            expert,
            packed_bytes: spec.expert_bytes as u64,
            records: Vec::new(),
        },
        payload: ExpertRawPayload::FixedDense(payload),
        read_latency: Duration::from_millis(3),
        read_path: ExpertReadPath::PositionedRead,
    }
}

fn identity_fixed_q4_slot_bytes() -> Vec<u8> {
    let layout = tiny_fixed_q4_layout();
    let mut bytes = vec![0u8; layout.expert_bytes];
    let one_bf16 = 0x3f80u16.to_le_bytes();
    for (weight_offset, scale_offset) in [(0, 8), (16, 24), (32, 40)] {
        // Row-major 2x2 identity, low nibble first. Remaining component
        // bytes are fixed-slot padding and must stay addressable.
        bytes[weight_offset] = 0x01;
        bytes[weight_offset + 1] = 0x10;
        bytes[scale_offset..scale_offset + 2].copy_from_slice(&one_bf16);
        bytes[scale_offset + 2..scale_offset + 4].copy_from_slice(&one_bf16);
    }
    bytes
}

fn write_identity_fixed_q4_layer(root: &std::path::Path, layer: usize, experts: usize) {
    let slot = identity_fixed_q4_slot_bytes();
    let bytes = slot.repeat(experts);
    std::fs::write(expert_layer_path(root, layer), bytes).unwrap();
    let metadata = ExpertLayerPackMetadata::new_fixed_q4(
        layer,
        slot.len() as u64,
        experts,
        (0..experts)
            .map(|expert| ExpertPackMetadata {
                layer,
                expert,
                packed_bytes: slot.len() as u64,
                records: Vec::new(),
            })
            .collect(),
    );
    write_expert_metadata_atomically(root, layer, &metadata).unwrap();
}

fn reference_bf16(bytes: &[u8], group: usize) -> f32 {
    let offset = group * 2;
    f32::from_bits((u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as u32) << 16)
}

fn reference_q4_matvec(payload: &Q4MatvecPayload<'_>, input: &[f32]) -> Vec<f32> {
    let row_bytes = payload.cols.div_ceil(2);
    let groups_per_row = payload.cols.div_ceil(payload.group_size);
    (0..payload.rows)
        .map(|row| {
            (0..payload.cols)
                .map(|col| {
                    let byte = payload.packed[row * row_bytes + col / 2];
                    let quantized = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    let group = row * groups_per_row + col / payload.group_size;
                    let scale = reference_bf16(payload.scale_bytes, group);
                    let bias = reference_bf16(payload.bias_bytes, group);
                    (quantized as f32 * scale + bias) * input[col]
                })
                .sum()
        })
        .collect()
}

fn reference_q4_swiglu(payload: &ScheduledQ4ExpertPhaseMlpPayload<'_>, input: &[f32]) -> Vec<f32> {
    let gate = reference_q4_matvec(&payload.gate, input);
    let up = reference_q4_matvec(&payload.up, input);
    let intermediate: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect();
    reference_q4_matvec(&payload.down, &intermediate)
}

fn reference_rms_norm(values: &[f32], weights: &[f32]) -> Vec<f32> {
    let mean_square =
        values.iter().map(|value| value * value).sum::<f32>() / values.len().max(1) as f32;
    let scale = (mean_square + 1e-6).sqrt().recip();
    values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| value * scale * weight)
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct DummyCmd3Input {
    source: ScheduledCmd3InputSource,
    width: usize,
}

#[derive(Debug, Clone, Copy)]
struct DummyCmd3InputState {
    source: ScheduledCmd3InputSource,
    state: FlashMoeCmd3InputState,
}

impl ScheduledCmd3Input for DummyCmd3Input {
    fn scheduled_cmd3_input_source(&self) -> ScheduledCmd3InputSource {
        self.source
    }

    fn scheduled_cmd3_input_state(&self, layer: usize) -> FlashMoeCmd3InputState {
        match self.source {
            ScheduledCmd3InputSource::CpuNormedResidualUpload => {
                FlashMoeCmd3InputState::cpu_normed_residual(layer, self.width, self.width)
            }
            ScheduledCmd3InputSource::MetalPostAttentionPrep => {
                FlashMoeCmd3InputState::metal_post_attention_prep(
                    layer,
                    FlashMoePostAttentionPrepState::new(layer, self.width, 16, 4),
                )
            }
        }
    }
}

impl ScheduledCmd3Input for DummyCmd3InputState {
    fn scheduled_cmd3_input_source(&self) -> ScheduledCmd3InputSource {
        self.source
    }

    fn scheduled_cmd3_input_state(&self, _layer: usize) -> FlashMoeCmd3InputState {
        self.state
    }
}

#[derive(Debug, Clone, Copy)]
struct DummySharedExpert {
    source: ScheduledSharedExpertSource,
    shape: Option<ScheduledSharedExpertShape>,
}

impl ScheduledSharedExpert for DummySharedExpert {
    fn scheduled_shared_expert_descriptor(&self) -> Result<ScheduledSharedExpertDescriptor> {
        ScheduledSharedExpertDescriptor::new(self.source, self.shape)
    }
}

fn dummy_cmd3_input(source: ScheduledCmd3InputSource) -> DummyCmd3Input {
    DummyCmd3Input { source, width: 8 }
}

fn dummy_cmd3_input_with_width(source: ScheduledCmd3InputSource, width: usize) -> DummyCmd3Input {
    DummyCmd3Input { source, width }
}

fn dummy_shared_expert(source: ScheduledSharedExpertSource) -> DummySharedExpert {
    let shape = match source {
        ScheduledSharedExpertSource::None => None,
        ScheduledSharedExpertSource::DenseCpuWeights
        | ScheduledSharedExpertSource::ResidentProjections => {
            Some(ScheduledSharedExpertShape::new(8, 2, 2).unwrap())
        }
    };
    DummySharedExpert { source, shape }
}

fn dummy_shared_expert_with_shape(
    source: ScheduledSharedExpertSource,
    shape: Option<ScheduledSharedExpertShape>,
) -> DummySharedExpert {
    DummySharedExpert { source, shape }
}

fn test_execution_scheduler() -> (tempfile::TempDir, FlashMoeExecutionScheduler) {
    test_execution_scheduler_with_attention(vec![
        ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
        ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
        ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
        ScheduledLayerAttentionImplementation::FullAttentionCpuKv,
        ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
    ])
}

fn test_execution_scheduler_with_attention(
    attention_layers: Vec<ScheduledLayerAttentionImplementation>,
) -> (tempfile::TempDir, FlashMoeExecutionScheduler) {
    let temp = tempfile::tempdir().unwrap();
    let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
    let store = ExpertSlotStore::open_with_fixed_q4(temp.path().to_path_buf(), spec).unwrap();
    let mut layout = qwen35_layout();
    layout.layers = attention_layers.len();
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
    let resolved_attention = capabilities
        .attention_layers
        .iter()
        .copied()
        .map(ScheduledLayerAttentionImplementation::from)
        .collect::<Vec<_>>();
    assert_eq!(attention_layers, resolved_attention);
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    (temp, FlashMoeExecutionScheduler::new(graph, store).unwrap())
}

fn test_dense_execution_scheduler_with_attention(
    dtype: DenseExpertDtype,
    attention_layers: Vec<ScheduledLayerAttentionImplementation>,
) -> (tempfile::TempDir, FlashMoeExecutionScheduler) {
    let temp = tempfile::tempdir().unwrap();
    let tiny_spec = FixedDenseExpertSlotSpec::new(dtype, 2, 2).unwrap();
    let store =
        ExpertSlotStore::open_with_fixed_dense(temp.path().to_path_buf(), tiny_spec).unwrap();
    let mut layout = qwen35_layout();
    layout.layers = attention_layers.len();
    let graph_spec = FixedDenseExpertSlotSpec::from_model_layout(&layout, dtype).unwrap();
    let mut capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
    capabilities.expert_storage = ExpertStoreExecutionDescriptor {
        layout: match dtype {
            DenseExpertDtype::Bf16 => ExpertStorageLayout::FixedBf16,
            DenseExpertDtype::F16 => ExpertStorageLayout::FixedF16,
        },
        slot_spec: ExpertSlotSpec::FixedDense(graph_spec),
        layers: layout.layers,
        first_expert_layer: layout.first_sparse_layer,
        experts_per_layer: layout.experts_per_layer,
    };
    let resolved_attention = capabilities
        .attention_layers
        .iter()
        .copied()
        .map(ScheduledLayerAttentionImplementation::from)
        .collect::<Vec<_>>();
    assert_eq!(attention_layers, resolved_attention);
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    (temp, FlashMoeExecutionScheduler::new(graph, store).unwrap())
}

fn write_identity_fixed_dense_layer(
    root: &Path,
    layer: usize,
    experts: usize,
    dtype: DenseExpertDtype,
) {
    let spec = FixedDenseExpertSlotSpec::new(dtype, 2, 2).unwrap();
    let one = match dtype {
        DenseExpertDtype::Bf16 => 0x3f80u16.to_le_bytes(),
        DenseExpertDtype::F16 => 0x3c00u16.to_le_bytes(),
    };
    let mut bytes = vec![0u8; spec.expert_bytes * experts];
    for expert in 0..experts {
        let slot = expert * spec.expert_bytes;
        for projection in [spec.gate, spec.up, spec.down] {
            let start = slot + projection.offset;
            bytes[start..start + 2].copy_from_slice(&one);
            bytes[start + 6..start + 8].copy_from_slice(&one);
        }
    }
    fs::write(expert_layer_path(root, layer), bytes).unwrap();
    let metadata = ExpertLayerPackMetadata::new_fixed_dense(
        layer,
        spec.expert_bytes as u64,
        experts,
        (0..experts)
            .map(|expert| ExpertPackMetadata {
                layer,
                expert,
                packed_bytes: spec.expert_bytes as u64,
                records: Vec::new(),
            })
            .collect(),
    );
    write_expert_metadata_atomically(root, layer, &metadata).unwrap();
}

fn test_qwen3_execution_scheduler() -> (tempfile::TempDir, FlashMoeExecutionScheduler) {
    let temp = tempfile::tempdir().unwrap();
    let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
    let store = ExpertSlotStore::open_with_fixed_q4(temp.path().to_path_buf(), spec).unwrap();
    let mut layout = qwen3_moe_layout();
    layout.layers = 2;
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    (temp, FlashMoeExecutionScheduler::new(graph, store).unwrap())
}

#[allow(clippy::too_many_arguments)]
fn run_identity_k4_layer(
    scheduler: &mut FlashMoeExecutionScheduler,
    position: usize,
    layer: usize,
    layers: usize,
    previous: ScheduledPreviousCmd3Handoff,
    cmd1_input: ScheduledCmd1InputSource,
    cmd1_state: FlashMoeCmd1InputState,
    residual: &[f32],
    normed: &[f32],
    shared_output: &[f32],
    next_norm_weights: Option<&[f32]>,
) -> ScheduledLayerExecution<FlashMoeExpertPhaseOutput> {
    let active_experts = 4;
    let experts = 9;
    let width = residual.len();
    let scheduled = scheduler
        .begin_layer(position, layer, layers, active_experts, previous, true)
        .unwrap();
    let (_, scheduled) = scheduled
        .resolve(scheduler, cmd1_input, cmd1_state)
        .unwrap();
    let (cmd2, scheduled) = scheduled
        .resolve(
            scheduler,
            ScheduledCmd2PhaseInputs::from_inputs(
                ScheduledCmd2AttentionInput::metal_values(width),
                ScheduledCmd2ResidualInput::metal_buffer(width),
            ),
        )
        .unwrap();
    let router_scores = [0.1, 2.0, -1.0, 3.0, 0.5, 2.5, -0.2, 1.5, 4.0];
    let active = top_k(&router_scores, active_experts);
    let prep_state = FlashMoePostAttentionPrepState::new(layer, width, experts, active_experts);
    let routing = scheduler
        .routing_from_post_attention_prep(&cmd2, prep_state, &active)
        .unwrap();
    let pending = scheduled
        .resolve(&routing)
        .unwrap()
        .issue_cmd3(scheduler, &routing)
        .unwrap();
    let input = DummyCmd3InputState {
        source: ScheduledCmd3InputSource::MetalPostAttentionPrep,
        state: FlashMoeCmd3InputState::metal_post_attention_prep(layer, prep_state),
    };
    let shared = dummy_shared_expert_with_shape(
        ScheduledSharedExpertSource::ResidentProjections,
        Some(ScheduledSharedExpertShape::new(width, 1, width).unwrap()),
    );
    let next_norm = match next_norm_weights {
        Some(weights) => ScheduledNextNormWeights::cpu_visible(
            "model.layers.next.input_layernorm.weight",
            weights,
            width,
        )
        .unwrap(),
        None => ScheduledNextNormWeights::none(),
    };
    pending
        .finish(scheduler, input, shared, next_norm, |command| {
            let mut expert_output = vec![0.0f32; width];
            for (payload, weight) in command.payloads.iter().zip(command.weights.iter()) {
                let output = reference_q4_swiglu(payload.q4(), normed);
                for (combined, value) in expert_output.iter_mut().zip(output.iter()) {
                    *combined += value * weight;
                }
            }
            let hidden: Vec<f32> = residual
                .iter()
                .zip(expert_output.iter())
                .zip(shared_output.iter())
                .map(|((residual, expert), shared)| residual + expert + shared)
                .collect();
            let next_normed = command
                .next_norm_weights
                .values()
                .map(|weights| reference_rms_norm(&hidden, weights));
            command
                .resolve_output_state()?
                .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(hidden, next_normed))
        })
        .unwrap()
}

#[test]
fn execution_scheduler_resolves_cmd1_cmd2_and_routing_with_one_graph_owner() {
    let (_temp, scheduler) = test_execution_scheduler();
    let layer = scheduler
        .begin_layer(
            17,
            3,
            5,
            2,
            ScheduledPreviousCmd3Handoff::cpu_visible(2, 8),
            true,
        )
        .unwrap();
    let (cmd1, layer) = layer
        .resolve(
            &scheduler,
            ScheduledCmd1InputSource::CpuNormedHidden,
            FlashMoeCmd1InputState::cpu_normed(3, 8),
        )
        .unwrap();
    assert_eq!(cmd1.layer, 3);
    assert_eq!(cmd1.input_state.layer(), 3);

    let (cmd2, layer) = layer
        .resolve(
            &scheduler,
            ScheduledCmd2PhaseInputs::from_inputs(
                ScheduledCmd2AttentionInput::metal_values(8),
                ScheduledCmd2ResidualInput::metal_buffer(8),
            ),
        )
        .unwrap();
    let state = FlashMoePostAttentionPrepState::new(3, 8, 5, 2);
    let routing = scheduler
        .routing_from_post_attention_prep(&cmd2, state, &[(4, 3.0), (1, 2.0)])
        .unwrap();
    let routed = layer.resolve(&routing).unwrap();

    assert_eq!(routing.layer, 3);
    assert_eq!(routing.active_experts, 2);
    assert_eq!(routing.routes, vec![(4, 3.0), (1, 2.0)]);
    assert_eq!(
        routed.identity.output_handoff,
        ScheduledCmd3OutputHandoff::DeferredToNextLayer
    );

    let complete_here = scheduler
        .begin_layer(
            17,
            3,
            5,
            2,
            ScheduledPreviousCmd3Handoff::cpu_visible(2, 8),
            false,
        )
        .unwrap();
    assert_eq!(
        complete_here.identity.output_handoff,
        ScheduledCmd3OutputHandoff::CompleteHere
    );
}

#[test]
fn execution_scheduler_rejects_mismatched_previous_cmd3_handoffs() {
    let (_temp, scheduler) = test_execution_scheduler();
    let err = scheduler
        .begin_layer(
            17,
            3,
            5,
            2,
            ScheduledPreviousCmd3Handoff::cpu_visible(1, 8),
            true,
        )
        .unwrap_err();
    assert!(err.to_string().contains("does not feed layer 3"), "{err:#}");

    let layer = scheduler
        .begin_layer(
            17,
            3,
            5,
            2,
            ScheduledPreviousCmd3Handoff::deferred_gpu(
                2,
                FlashMoeGpuBufferDescriptor::hidden(8),
                FlashMoeGpuBufferDescriptor::next_layer_normed(8),
            ),
            true,
        )
        .unwrap();
    let err = layer
        .resolve(
            &scheduler,
            ScheduledCmd1InputSource::CpuNormedHidden,
            FlashMoeCmd1InputState::cpu_normed(3, 8),
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("requires CMD1 input DeferredMetalNextNormed"),
        "{err:#}"
    );
}

#[test]
fn execution_scheduler_finishes_whole_slot_reads_and_submits_cmd3_transaction() {
    let (_temp, mut scheduler) = test_execution_scheduler();
    let layer = scheduler
        .begin_layer(
            19,
            3,
            5,
            1,
            ScheduledPreviousCmd3Handoff::cpu_visible(2, 2),
            true,
        )
        .unwrap();
    let (_, layer) = layer
        .resolve(
            &scheduler,
            ScheduledCmd1InputSource::CpuNormedHidden,
            FlashMoeCmd1InputState::cpu_normed(3, 2),
        )
        .unwrap();
    let (cmd2, layer) = layer
        .resolve(
            &scheduler,
            ScheduledCmd2PhaseInputs::from_inputs(
                ScheduledCmd2AttentionInput::metal_values(2),
                ScheduledCmd2ResidualInput::metal_buffer(2),
            ),
        )
        .unwrap();
    let routing = scheduler
        .routing_from_post_attention_prep(
            &cmd2,
            FlashMoePostAttentionPrepState::new(3, 2, 9, 1),
            &[(8, 1.0)],
        )
        .unwrap();
    let routed = layer.resolve(&routing).unwrap();
    let routes = ScheduledExpertRoutes::from_routing_command(&routing, 1.0).unwrap();
    let (tx, rx) = mpsc::channel();
    let pending = PendingScheduledExpertSet::new(routes, vec![PendingScheduledRead::new(77, rx)]);
    tx.send(ExpertRawReadResponse {
        id: 77,
        queue_latency: Duration::from_millis(1),
        read_path: ExpertReadPath::PositionedRead,
        read_latency: Duration::from_millis(2),
        bytes_read: tiny_fixed_q4_layout().expert_bytes as u64,
        warm: false,
        result: Ok(raw_fixed_q4_read(3, 8)),
    })
    .unwrap();
    let transaction = PendingScheduledCmd3 {
        before: scheduler.snapshot(),
        pending: PendingScheduledExpertAccess::Streamed(pending),
        issue_elapsed: Duration::from_millis(1),
    };
    let shared = dummy_shared_expert_with_shape(
        ScheduledSharedExpertSource::ResidentProjections,
        Some(ScheduledSharedExpertShape::new(2, 1, 2).unwrap()),
    );

    let pending_layer = ScheduledLayerPendingCmd3 {
        identity: routed.identity,
        pending: transaction,
    };
    let execution = pending_layer
        .finish(
            &mut scheduler,
            dummy_cmd3_input_with_width(ScheduledCmd3InputSource::CpuNormedResidualUpload, 2),
            shared,
            ScheduledNextNormWeights::none(),
            |command| {
                assert_eq!(command.layer, 3);
                assert_eq!(command.experts.len(), 1);
                Ok(command.layer)
            },
        )
        .unwrap();
    let cmd3 = execution.cmd3;

    assert_eq!(
        execution.output_handoff,
        ScheduledCmd3OutputHandoff::DeferredToNextLayer
    );
    assert_eq!(cmd3.submission, 3);
    assert_eq!(cmd3.expert_delta.positioned_reads, 1);
    assert_eq!(
        cmd3.expert_delta.bytes_read,
        tiny_fixed_q4_layout().expert_bytes as u64
    );
    assert_eq!(cmd3.expert_mixes.len(), 1);
    assert_eq!(cmd3.expert_mixes[0].1, 1.0);
    assert!(cmd3.expert_io_elapsed >= Duration::from_millis(1));
}

#[test]
fn qwen35_q4_layer_parity_fixture_follows_resolved_k4_transaction() {
    // Golden values are independently derived from the Qwen3.5/Qwen3Next
    // equations: scaled dot-product attention, residual RMSNorm, router
    // topK then selected-score softmax, Q4 SwiGLU, shared addition, RMSNorm.
    let position = 7;
    let layer = 3;
    let experts = 9;
    let active_experts = 4;
    let width = 2;
    let query = [1.0, 0.0];
    let key_0 = [1.0, 0.0];
    let value_0 = [2.0, 1.0];
    let key_1 = [0.0, 1.0];
    let value_1 = [-1.0, 3.0];
    let attention = causal_attention(
        &query,
        &[(&key_0, &value_0), (&key_1, &value_1)],
        1,
        1,
        width,
    );
    for (actual, expected) in attention.iter().zip([1.0092846, 1.6604769]) {
        assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
    }
    let residual_input = [0.5, -1.0];
    let residual = [
        residual_input[0] + attention[0],
        residual_input[1] + attention[1],
    ];
    let normed = reference_rms_norm(&residual, &[1.0, 1.0]);
    for (actual, expected) in normed.iter().zip([1.2955897, 0.5669620]) {
        assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
    }

    let router_scores = [0.1, 2.0, -1.0, 3.0, 0.5, 2.5, -0.2, 1.5, 4.0];
    let mut router_probabilities = router_scores.to_vec();
    softmax_in_place(&mut router_probabilities);
    let active = routing_top_k(&router_probabilities, active_experts);
    assert_eq!(
        active.iter().map(|(expert, _)| *expert).collect::<Vec<_>>(),
        vec![5, 1, 8, 3]
    );

    let (temp, mut scheduler) = test_execution_scheduler();
    write_identity_fixed_q4_layer(temp.path(), layer, experts);
    let attention_math = scheduler
        .graph
        .build_attention_math(layer, position)
        .unwrap()
        .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(
            position, layer, width, width,
        ))
        .unwrap();
    assert_eq!(
        attention_math.implementation(),
        ScheduledAttentionMathImplementation::CpuKvCache
    );

    let scheduled = scheduler
        .begin_layer(
            position,
            layer,
            5,
            active_experts,
            ScheduledPreviousCmd3Handoff::deferred_gpu(
                layer - 1,
                FlashMoeGpuBufferDescriptor::hidden(width),
                FlashMoeGpuBufferDescriptor::next_layer_normed(width),
            ),
            true,
        )
        .unwrap();
    let (_, scheduled) = scheduled
        .resolve(
            &scheduler,
            ScheduledCmd1InputSource::DeferredMetalNextNormed,
            FlashMoeCmd1InputState::gpu_next_layer_normed(
                layer,
                FlashMoeGpuBufferDescriptor::next_layer_normed(width),
            ),
        )
        .unwrap();
    let (cmd2, scheduled) = scheduled
        .resolve(
            &scheduler,
            ScheduledCmd2PhaseInputs::from_inputs(
                ScheduledCmd2AttentionInput::metal_values(width),
                ScheduledCmd2ResidualInput::metal_buffer(width),
            ),
        )
        .unwrap();
    let prep_state = FlashMoePostAttentionPrepState::new(layer, width, experts, active_experts);
    let routing = scheduler
        .routing_from_post_attention_prep(&cmd2, prep_state, &active)
        .unwrap();
    let routed = scheduled.resolve(&routing).unwrap();
    let pending = routed.issue_cmd3(&mut scheduler, &routing).unwrap();
    let next_norm_weights = [1.0, 0.5];
    let shared_output = [0.25, -0.5];
    let input = DummyCmd3InputState {
        source: ScheduledCmd3InputSource::MetalPostAttentionPrep,
        state: FlashMoeCmd3InputState::metal_post_attention_prep(layer, prep_state),
    };
    let shared = dummy_shared_expert_with_shape(
        ScheduledSharedExpertSource::ResidentProjections,
        Some(ScheduledSharedExpertShape::new(width, 1, width).unwrap()),
    );
    let execution = pending
        .finish(
            &mut scheduler,
            input,
            shared,
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.4.input_layernorm.weight",
                &next_norm_weights,
                width,
            )
            .unwrap(),
            |command| {
                assert_eq!(
                    command
                        .experts
                        .iter()
                        .map(|expert| expert.expert())
                        .collect::<Vec<_>>(),
                    vec![5, 1, 8, 3]
                );
                for (actual, expected) in command
                    .weights
                    .iter()
                    .zip([0.12925005, 0.07839412, 0.57925856, 0.2130973])
                {
                    assert!((actual - expected).abs() <= 1e-6);
                }
                let mut expert_output = vec![0.0f32; width];
                for (payload, weight) in command.payloads.iter().zip(command.weights.iter()) {
                    let output = reference_q4_swiglu(payload.q4(), &normed);
                    for (combined, value) in expert_output.iter_mut().zip(output.iter()) {
                        *combined += value * weight;
                    }
                }
                let hidden: Vec<f32> = residual
                    .iter()
                    .zip(expert_output.iter())
                    .zip(shared_output.iter())
                    .map(|((residual, expert), shared)| residual + expert + shared)
                    .collect();
                for (actual, expected) in hidden.iter().zip([3.0771027, 0.36557937]) {
                    assert!(
                        (actual - expected).abs() <= 1e-5,
                        "{actual} != {expected}; hidden={hidden:?}"
                    );
                }
                let next_normed =
                    reference_rms_norm(&hidden, command.next_norm_weights.values().unwrap());
                command
                    .resolve_output_state()?
                    .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(
                        hidden,
                        Some(next_normed),
                    ))
            },
        )
        .unwrap();

    assert_eq!(
        execution.output_handoff,
        ScheduledCmd3OutputHandoff::DeferredToNextLayer
    );
    assert_eq!(execution.cmd3.expert_delta.positioned_reads, 4);
    assert_eq!(execution.cmd3.expert_delta.bytes_read, 4 * 48);
    let mut token_state = FlashMoeTokenState::new(vec![0.0; width], 0);
    token_state
        .apply_declared_expert_phase(
            execution.cmd3.submission,
            FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
        )
        .unwrap();
    for (actual, expected) in token_state.hidden().iter().zip([3.0771027, 0.36557937]) {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
    let next_normed = token_state.take_next_layer_normed_as_normed().unwrap();
    for (actual, expected) in next_normed.iter().zip([1.404337, 0.08342209]) {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
}

#[test]
fn qwen3_q4_layer_parity_follows_resolved_k8_no_shared_transaction() {
    // Golden values follow Qwen3 MoE's selected-route normalization with
    // scale 1.0, full attention, fixed-Q4 SwiGLU, and no shared expert.
    let position = 7;
    let layer = 0;
    let experts = 9;
    let active_experts = 8;
    let width = 2;
    let query = [1.0, 0.0];
    let key_0 = [1.0, 0.0];
    let value_0 = [2.0, 1.0];
    let key_1 = [0.0, 1.0];
    let value_1 = [-1.0, 3.0];
    let attention = causal_attention(
        &query,
        &[(&key_0, &value_0), (&key_1, &value_1)],
        1,
        1,
        width,
    );
    let residual = [0.5 + attention[0], -1.0 + attention[1]];
    let normed = reference_rms_norm(&residual, &[1.0, 1.0]);
    let router_scores = [0.1, 2.0, -1.0, 3.0, 0.5, 2.5, -0.2, 1.5, 4.0];
    let mut router_probabilities = router_scores.to_vec();
    softmax_in_place(&mut router_probabilities);
    let active = routing_top_k(&router_probabilities, active_experts);
    assert_eq!(
        active.iter().map(|(expert, _)| *expert).collect::<Vec<_>>(),
        vec![0, 1, 8, 3, 4, 5, 6, 7]
    );

    let (temp, mut scheduler) = test_qwen3_execution_scheduler();
    assert_eq!(scheduler.graph.family(), QwenMoeFamily::Qwen3Moe);
    assert_eq!(scheduler.experts_per_layer(), 128);
    assert_eq!(scheduler.active_experts(), 8);
    assert_eq!(scheduler.graph.routed_expert_scale(), 1.0);
    write_identity_fixed_q4_layer(temp.path(), layer, experts);

    let attention_math = scheduler
        .resolve_attention_math(layer, position)
        .unwrap()
        .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(
            position, layer, width, width,
        ))
        .unwrap();
    assert_eq!(
        attention_math.implementation(),
        ScheduledAttentionMathImplementation::CpuKvCache
    );

    let scheduled = scheduler
        .begin_resolved_layer(
            position,
            layer,
            2,
            ScheduledPreviousCmd3Handoff::initial(width),
            true,
        )
        .unwrap();
    let (_, scheduled) = scheduled
        .resolve(
            &scheduler,
            ScheduledCmd1InputSource::CpuNormedHidden,
            FlashMoeCmd1InputState::cpu_normed(layer, width),
        )
        .unwrap();
    let (cmd2, scheduled) = scheduled
        .resolve(
            &scheduler,
            ScheduledCmd2PhaseInputs::from_inputs(
                ScheduledCmd2AttentionInput::metal_values(width),
                ScheduledCmd2ResidualInput::metal_buffer(width),
            ),
        )
        .unwrap();
    assert_eq!(cmd2.active_experts, active_experts);
    let prep_state = FlashMoePostAttentionPrepState::new(layer, width, experts, active_experts);
    let routing = scheduler
        .routing_from_post_attention_prep(&cmd2, prep_state, &active)
        .unwrap();
    let routed = scheduled.resolve(&routing).unwrap();
    let pending = routed.issue_cmd3(&mut scheduler, &routing).unwrap();
    let next_norm_weights = [1.0, 0.5];
    let execution = pending
        .finish(
            &mut scheduler,
            DummyCmd3InputState {
                source: ScheduledCmd3InputSource::MetalPostAttentionPrep,
                state: FlashMoeCmd3InputState::metal_post_attention_prep(layer, prep_state),
            },
            dummy_shared_expert(ScheduledSharedExpertSource::None),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.1.input_layernorm.weight",
                &next_norm_weights,
                width,
            )
            .unwrap(),
            |command| {
                assert_eq!(command.cmd3.shared, ScheduledSharedExpertSource::None);
                assert_eq!(
                    command
                        .experts
                        .iter()
                        .map(|expert| expert.expert())
                        .collect::<Vec<_>>(),
                    vec![0, 1, 8, 3, 4, 5, 6, 7]
                );
                for (actual, expected) in command.weights.iter().zip([
                    0.010802226,
                    0.072222546,
                    0.5336564,
                    0.19632123,
                    0.016115028,
                    0.11907485,
                    0.008002486,
                    0.04380519,
                ]) {
                    assert!((actual - expected).abs() <= 1e-6);
                }
                let mut expert_output = vec![0.0f32; width];
                for (payload, weight) in command.payloads.iter().zip(command.weights.iter()) {
                    let output = reference_q4_swiglu(payload.q4(), &normed);
                    for (combined, value) in expert_output.iter_mut().zip(output.iter()) {
                        *combined += value * weight;
                    }
                }
                let hidden: Vec<f32> = residual
                    .iter()
                    .zip(expert_output.iter())
                    .map(|(residual, expert)| residual + expert)
                    .collect();
                for (actual, expected) in hidden.iter().zip([2.8271027, 0.86557925]) {
                    assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
                }
                let next_normed =
                    reference_rms_norm(&hidden, command.next_norm_weights.values().unwrap());
                command
                    .resolve_output_state()?
                    .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(
                        hidden,
                        Some(next_normed),
                    ))
            },
        )
        .unwrap();

    assert_eq!(execution.cmd3.expert_delta.positioned_reads, 8);
    assert_eq!(execution.cmd3.expert_delta.bytes_read, 8 * 48);
    let mut token_state = FlashMoeTokenState::new(vec![0.0; width], 0);
    token_state
        .apply_declared_expert_phase(
            execution.cmd3.submission,
            FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
        )
        .unwrap();
    for (actual, expected) in token_state.hidden().iter().zip([2.8271027, 0.86557925]) {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
    let next_normed = token_state.take_next_layer_normed_as_normed().unwrap();
    for (actual, expected) in next_normed.iter().zip([1.3522521, 0.20701078]) {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
}

#[test]
fn qwen35_q4_multi_linear_layer_parity_preserves_deferred_state_and_logits() {
    let (temp, mut scheduler) = test_execution_scheduler_with_attention(vec![
        ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
        ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
    ]);
    write_identity_fixed_q4_layer(temp.path(), 0, 9);
    write_identity_fixed_q4_layer(temp.path(), 1, 9);
    assert_eq!(
        scheduler.graph.attention_layers.as_ref(),
        [
            ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
            ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal,
        ]
    );

    let shared_output = [0.25, -0.5];
    let residual_0 = [0.5, -1.0];
    let normed_0 = reference_rms_norm(&residual_0, &[1.0, 1.0]);
    let next_norm_weights = [1.0, 0.5];
    let layer_0 = run_identity_k4_layer(
        &mut scheduler,
        11,
        0,
        2,
        ScheduledPreviousCmd3Handoff::initial(2),
        ScheduledCmd1InputSource::CpuNormedHidden,
        FlashMoeCmd1InputState::cpu_normed(0, 2),
        &residual_0,
        &normed_0,
        &shared_output,
        Some(&next_norm_weights),
    );
    assert_eq!(
        layer_0.output_handoff,
        ScheduledCmd3OutputHandoff::DeferredToNextLayer
    );
    let mut token_state = FlashMoeTokenState::new(vec![0.0; 2], 0);
    for (mix_hash, weight) in &layer_0.cmd3.expert_mixes {
        token_state.mix_active_expert(*mix_hash, *weight);
    }
    token_state
        .apply_declared_expert_phase(
            layer_0.cmd3.submission,
            FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
        )
        .unwrap();
    let hidden_0 = token_state.hidden().to_vec();
    for (actual, expected) in hidden_0.iter().zip([1.011218, -1.1477928]) {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "{actual} != {expected}; hidden={hidden_0:?}"
        );
    }
    let next_normed_0 = token_state
        .take_next_layer_normed_as_normed()
        .unwrap()
        .into_values();
    for (actual, expected) in next_normed_0.iter().zip([0.9348729, -0.5305683]) {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
    let recurrent_after_linear = token_state.recurrent_value();
    assert_ne!(recurrent_after_linear, 0);

    let attention_1 = [0.6036627f32, 0.7642249];
    for (actual, expected) in attention_1.iter().zip([0.6036627, 0.7642249]) {
        assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
    }
    let residual_1 = [hidden_0[0] + attention_1[0], hidden_0[1] + attention_1[1]];
    let normed_1 = reference_rms_norm(&residual_1, &[1.0, 1.0]);
    let layer_1 = run_identity_k4_layer(
        &mut scheduler,
        11,
        1,
        2,
        ScheduledPreviousCmd3Handoff::deferred_gpu(
            0,
            FlashMoeGpuBufferDescriptor::hidden(2),
            FlashMoeGpuBufferDescriptor::next_layer_normed(2),
        ),
        ScheduledCmd1InputSource::DeferredMetalNextNormed,
        FlashMoeCmd1InputState::gpu_next_layer_normed(
            1,
            FlashMoeGpuBufferDescriptor::next_layer_normed(2),
        ),
        &residual_1,
        &normed_1,
        &shared_output,
        None,
    );
    assert_eq!(
        layer_1.output_handoff,
        ScheduledCmd3OutputHandoff::CompleteHere
    );
    for (mix_hash, weight) in &layer_1.cmd3.expert_mixes {
        token_state.mix_active_expert(*mix_hash, *weight);
    }
    token_state
        .apply_declared_expert_phase(
            layer_1.cmd3.submission,
            FlashMoeExpertPhaseApplication::HiddenOnly,
        )
        .unwrap();
    assert_ne!(token_state.recurrent_value(), recurrent_after_linear);
    let hidden_1 = token_state.hidden().to_vec();
    for (actual, expected) in hidden_1.iter().zip([3.3762858, -0.8388142]) {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "{actual} != {expected}; hidden={hidden_1:?}"
        );
    }
    assert!(token_state.take_next_layer_normed_as_normed().is_none());

    let logits = [
        hidden_1[0],
        hidden_1[1],
        -hidden_1[0] + 0.5 * hidden_1[1],
        0.25 * hidden_1[0] - 0.75 * hidden_1[1],
    ];
    for (actual, expected) in logits
        .iter()
        .zip([3.3762858, -0.8388142, -3.795693, 1.4731821])
    {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
    let candidates = top_k(&logits, 2);
    assert_eq!(candidates[0].0, 0);
    assert_eq!(candidates[1].0, 3);
    assert!((candidates[0].1 - 3.3762858).abs() <= 1e-5);
    assert!((candidates[1].1 - 1.4731821).abs() <= 1e-5);
    let metrics = scheduler.snapshot();
    assert_eq!(metrics.positioned_reads, 8);
    assert_eq!(metrics.bytes_read, 8 * 48);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn qwen35_resident_shared_cmd3_metal_output_matches_layer_reference() {
    #[derive(Debug, Clone, Copy)]
    enum ActiveLayout {
        Q4,
        Dense(DenseExpertDtype),
    }

    #[derive(Debug, Clone, Copy)]
    enum SharedLayout {
        Q4,
        Dense(&'static str),
    }

    let position = 7;
    let layer = 0;
    let width = 2;
    let experts = 9;
    let active_experts = 4;
    let residual = [1.5092846, 0.6604769];
    let normed = [1.2955897, 0.5669620];
    let router_scores = [0.1, 2.0, -1.0, 3.0, 0.5, 2.5, -0.2, 1.5, 4.0];
    let active = top_k(&router_scores, active_experts);
    for (active_layout, shared_layout) in [
        (ActiveLayout::Q4, SharedLayout::Q4),
        (ActiveLayout::Q4, SharedLayout::Dense("BF16")),
        (ActiveLayout::Q4, SharedLayout::Dense("F16")),
        (ActiveLayout::Q4, SharedLayout::Dense("F32")),
        (
            ActiveLayout::Dense(DenseExpertDtype::Bf16),
            SharedLayout::Q4,
        ),
        (ActiveLayout::Dense(DenseExpertDtype::F16), SharedLayout::Q4),
    ] {
        let attention = vec![ScheduledLayerAttentionImplementation::FusedLinearAttentionMetal];
        let (temp, mut scheduler) = match active_layout {
            ActiveLayout::Q4 => test_execution_scheduler_with_attention(attention),
            ActiveLayout::Dense(dtype) => {
                test_dense_execution_scheduler_with_attention(dtype, attention)
            }
        };
        match active_layout {
            ActiveLayout::Q4 => write_identity_fixed_q4_layer(temp.path(), layer, experts),
            ActiveLayout::Dense(dtype) => {
                write_identity_fixed_dense_layer(temp.path(), layer, experts, dtype)
            }
        }

        let scheduled = scheduler
            .begin_layer(
                position,
                layer,
                1,
                active_experts,
                ScheduledPreviousCmd3Handoff::initial(width),
                true,
            )
            .unwrap();
        let (_, scheduled) = scheduled
            .resolve(
                &scheduler,
                ScheduledCmd1InputSource::CpuNormedHidden,
                FlashMoeCmd1InputState::cpu_normed(layer, width),
            )
            .unwrap();
        let (cmd2, scheduled) = scheduled
            .resolve(
                &scheduler,
                ScheduledCmd2PhaseInputs::from_inputs(
                    ScheduledCmd2AttentionInput::metal_values(width),
                    ScheduledCmd2ResidualInput::metal_buffer(width),
                ),
            )
            .unwrap();
        let prep_state = FlashMoePostAttentionPrepState::new(layer, width, experts, active_experts);
        let routing = scheduler
            .routing_from_post_attention_prep(&cmd2, prep_state, &active)
            .unwrap();
        let pending = scheduled
            .resolve(&routing)
            .unwrap()
            .issue_cmd3(&mut scheduler, &routing)
            .unwrap();

        let dense_path = temp.path().join("cmd3-parity-dense.bin");
        let mut dense_bytes = vec![0u8; 16 * 1024];
        let shared = match shared_layout {
            SharedLayout::Q4 => {
                let one_bf16 = 0x3f80u16.to_le_bytes();
                let mut write_identity_q4 = |packed: usize, scales: usize, rows: usize| {
                    dense_bytes[packed] = 0x01;
                    if rows > 1 {
                        dense_bytes[packed + 1] = 0x10;
                    }
                    for row in 0..rows {
                        dense_bytes[scales + row * 2..scales + row * 2 + 2]
                            .copy_from_slice(&one_bf16);
                    }
                };
                write_identity_q4(0, 16, 2);
                write_identity_q4(64, 80, 2);
                write_identity_q4(128, 144, 2);
                dense_bytes[208..210].copy_from_slice(&one_bf16);
                let projection = |tensor_name: &str,
                                  packed_byte_offset,
                                  scales_byte_offset,
                                  biases_byte_offset,
                                  rows| {
                    DenseQ4MmapMatvecProjection {
                        tensor_name: tensor_name.to_string(),
                        packed_byte_offset,
                        scales_byte_offset,
                        biases_byte_offset,
                        rows,
                        cols: width,
                        output_width: rows,
                        row_packed_bytes: 1,
                        groups_per_row: 1,
                        group_size: 2,
                        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                    }
                    .into()
                };
                SharedExpertPhaseResidentProjections {
                    gate: projection("shared.gate", 0, 16, 32, 2),
                    up: projection("shared.up", 64, 80, 96, 2),
                    down: projection("shared.down", 128, 144, 160, 2),
                    router: Some(projection("shared.router", 192, 208, 224, 1)),
                    shared_experts: 1,
                    intermediate: 2,
                    width,
                }
            }
            SharedLayout::Dense(dtype) => {
                let scalar_bytes = |value: f32| match dtype {
                    "BF16" => ((value.to_bits() >> 16) as u16).to_le_bytes().to_vec(),
                    "F16" => {
                        let bits = if value == 0.0 { 0u16 } else { 0x3c00u16 };
                        bits.to_le_bytes().to_vec()
                    }
                    "F32" => value.to_le_bytes().to_vec(),
                    _ => unreachable!(),
                };
                for offset in [0usize, 64, 128] {
                    let values = [1.0f32, 0.0, 0.0, 1.0];
                    let mut cursor = offset;
                    for value in values {
                        let bytes = scalar_bytes(value);
                        dense_bytes[cursor..cursor + bytes.len()].copy_from_slice(&bytes);
                        cursor += bytes.len();
                    }
                }
                let projection = |tensor_name: &str, byte_offset, rows| {
                    ResidentMmapMatvecProjection::Dense(DenseMmapMatvecProjection {
                        tensor_name: tensor_name.to_string(),
                        byte_offset,
                        dtype: dtype.to_string(),
                        rows,
                        cols: width,
                        output_width: rows,
                    })
                };
                SharedExpertPhaseResidentProjections {
                    gate: projection("shared.gate", 0, 2),
                    up: projection("shared.up", 64, 2),
                    down: projection("shared.down", 128, 2),
                    router: Some(projection("shared.router", 192, 1)),
                    shared_experts: 1,
                    intermediate: 2,
                    width,
                }
            }
        };
        std::fs::write(&dense_path, dense_bytes).unwrap();
        let dense_file = std::fs::File::open(&dense_path).unwrap();
        let dense_mmap = Arc::new(unsafe { memmap2::MmapOptions::new().map(&dense_file).unwrap() });
        let metal = MetalExecutionContext::compile(
            Arc::clone(&dense_mmap),
            dense_mmap.len() as u64,
            &[None],
            1e-6,
        )
        .unwrap();
        let f32_bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let normed_buffer = unsafe {
            metal
                .buffers()
                .buffer_with_bytes(metal.runtime().device, &f32_bytes(&normed))
                .unwrap()
        };
        let residual_buffer = unsafe {
            metal
                .buffers()
                .buffer_with_bytes(metal.runtime().device, &f32_bytes(&residual))
                .unwrap()
        };
        let mut prep = MetalPostAttentionPrep::new(
            layer,
            width,
            experts,
            active.clone(),
            residual_buffer,
            normed_buffer,
        )
        .unwrap();
        prep.attach_routing_command(routing.clone()).unwrap();
        let dense_weights = metal.dense_weights().unwrap();
        let execution = pending
            .finish(
                &mut scheduler,
                ExpertPhaseInput::MetalPostAttention(prep),
                ScheduledSharedExpertPhaseRef::Resident(&shared),
                ScheduledNextNormWeights::none(),
                |command| {
                    let output = command.resolve_output_state()?;
                    let ScheduledCmd3Command {
                        position,
                        layer,
                        experts,
                        weights,
                        input,
                        shared,
                        next_norm_weights,
                        payloads,
                        ..
                    } = command;
                    let ExpertPhaseInput::MetalPostAttention(input) = input;
                    MetalScheduledCmd3Builder::new(
                        metal.runtime(),
                        dense_weights,
                        Arc::clone(metal.buffers()),
                        metal.norm_epsilon(),
                    )
                    .submit(
                        position,
                        layer,
                        experts,
                        weights,
                        input,
                        output,
                        shared,
                        next_norm_weights.values(),
                        &payloads,
                    )
                },
            )
            .unwrap();
        let output = execution.cmd3.submission.wait().unwrap();
        let (hidden, next_normed) = output.into_hidden_and_next_normed();
        assert!(next_normed.is_none());
        for (actual, expected) in hidden.iter().zip([3.3542297, 0.9476203]) {
            assert!(
                (actual - expected).abs() <= 1e-4,
                "{actual} != {expected} for active={active_layout:?} shared={shared_layout:?}"
            );
        }
    }
}

fn dummy_shared_dense_phase() -> SharedExpertPhaseWeights {
    SharedExpertPhaseWeights {
        gate: Arc::new(vec![1.0, 2.0]),
        up: Arc::new(vec![3.0, 4.0]),
        down: Arc::new(vec![5.0, 6.0]),
        router: Arc::new(vec![7.0]),
        shared_experts: 1,
        intermediate: 2,
        width: 1,
    }
}

fn dummy_q4_projection(
    name: &str,
    output_width: usize,
    cols: usize,
) -> DenseQ4MmapMatvecProjection {
    DenseQ4MmapMatvecProjection {
        tensor_name: name.to_string(),
        packed_byte_offset: 128,
        scales_byte_offset: 256,
        biases_byte_offset: 512,
        rows: output_width,
        cols,
        output_width,
        row_packed_bytes: cols.div_ceil(2),
        groups_per_row: cols.div_ceil(16),
        group_size: 16,
        scale_bias_dtype: "BF16".to_string(),
    }
}

fn dummy_router_projection(
    layer: usize,
    experts: usize,
    hidden_width: usize,
) -> RouterScoreProjectionDescriptor {
    let tensor_name = format!("model.layers.{layer}.mlp.gate.weight");
    RouterScoreProjectionDescriptor {
        layer,
        tensor_name: tensor_name.clone(),
        experts,
        hidden_width,
        binding: RouterScoreProjectionBinding::ResidentDense(DenseMmapMatvecProjection {
            tensor_name,
            byte_offset: 4096,
            dtype: "F32".to_string(),
            rows: experts,
            cols: hidden_width,
            output_width: experts,
        }),
    }
}

fn dummy_shared_q4_phase() -> SharedExpertPhaseResidentProjections {
    SharedExpertPhaseResidentProjections {
        gate: dummy_q4_projection("shared.gate", 16, 32).into(),
        up: dummy_q4_projection("shared.up", 16, 32).into(),
        down: dummy_q4_projection("shared.down", 32, 16).into(),
        router: Some(dummy_q4_projection("shared.router", 1, 32).into()),
        shared_experts: 1,
        intermediate: 16,
        width: 32,
    }
}

#[test]
fn shared_expert_phase_ref_resolves_scheduler_source() {
    let dense = dummy_shared_dense_phase();
    let q4 = dummy_shared_q4_phase();

    let none = ScheduledSharedExpertPhaseRef::from_options(None, None);
    let none_descriptor = none.scheduled_shared_expert_descriptor().unwrap();
    assert_eq!(none_descriptor.source, ScheduledSharedExpertSource::None);
    assert_eq!(none_descriptor.shape, None);
    assert!(none.dense().is_none());
    assert!(none.resident().is_none());

    let dense_ref = ScheduledSharedExpertPhaseRef::from_options(Some(&dense), None);
    let dense_descriptor = dense_ref.scheduled_shared_expert_descriptor().unwrap();
    assert_eq!(
        dense_descriptor.source,
        ScheduledSharedExpertSource::DenseCpuWeights
    );
    assert!(dense_ref.dense().is_some());
    assert!(dense_ref.resident().is_none());
    assert_eq!(
        dense_descriptor.shape,
        Some(ScheduledSharedExpertShape::new(1, 1, 2).unwrap())
    );

    let q4_ref = ScheduledSharedExpertPhaseRef::from_options(Some(&dense), Some(&q4));
    let q4_descriptor = q4_ref.scheduled_shared_expert_descriptor().unwrap();
    assert_eq!(
        q4_descriptor.source,
        ScheduledSharedExpertSource::ResidentProjections
    );
    assert!(q4_ref.dense().is_none());
    assert!(q4_ref.resident().is_some());
    assert_eq!(
        q4_descriptor.shape,
        Some(ScheduledSharedExpertShape::new(32, 1, 16).unwrap())
    );
}

#[derive(Debug, Clone)]
struct DummyCmd3Expert {
    layer: usize,
    expert: usize,
    descriptor: ExpertSlotDescriptor,
}

impl DummyCmd3Expert {
    fn whole_slot(layer: usize, expert: usize) -> Self {
        Self {
            layer,
            expert,
            descriptor: ExpertSlotDescriptor {
                layer,
                expert,
                slot_offset: (expert as u64) * 128,
                slot_capacity: 128,
                payload_len: 128,
            },
        }
    }

    fn with_descriptor(mut self, descriptor: ExpertSlotDescriptor) -> Self {
        self.descriptor = descriptor;
        self
    }
}

impl ScheduledCmd3Expert for DummyCmd3Expert {
    fn scheduled_expert_layer(&self) -> usize {
        self.layer
    }

    fn scheduled_expert_id(&self) -> usize {
        self.expert
    }

    fn scheduled_expert_slot_descriptor(&self) -> ExpertSlotDescriptor {
        self.descriptor
    }
}

static DUMMY_Q4_SLOT: [u8; 512] = [0; 512];

fn dummy_q4_payload(rows: usize, cols: usize) -> Q4MatvecPayload<'static> {
    let packed_bytes = rows * cols.div_ceil(2);
    let scale_bias_bytes = rows * cols.div_ceil(8) * 2;
    Q4MatvecPayload {
        rows,
        cols,
        group_size: 8,
        packed: &DUMMY_Q4_SLOT[..packed_bytes],
        scales: &[],
        biases: &[],
        scale_bias_groups: rows * cols.div_ceil(8),
        scale_bias_dtype: "BF16",
        scale_bytes: &DUMMY_Q4_SLOT[128..128 + scale_bias_bytes],
        bias_bytes: &DUMMY_Q4_SLOT[256..256 + scale_bias_bytes],
        source: Some(Q4MatvecSource {
            bytes: &DUMMY_Q4_SLOT,
            packed_offset: 0,
            scale_offset: 128,
            bias_offset: 256,
            reusable_bytes: None,
        }),
    }
}

impl ScheduledCmd3ExpertPayload for DummyCmd3Expert {
    fn scheduled_cmd3_expert_phase_payload(
        &self,
        width: usize,
    ) -> Result<ScheduledExpertPhaseMlpPayload<'_>> {
        Ok(ScheduledExpertPhaseMlpPayload::Q4(
            ScheduledQ4ExpertPhaseMlpPayload::new(
                self.layer,
                self.expert,
                width,
                dummy_q4_payload(4, width),
                dummy_q4_payload(4, width),
                dummy_q4_payload(width, 4),
            )?,
        ))
    }
}

fn dummy_scheduled_experts(layer: usize, experts: usize) -> ScheduledExpertSet<DummyCmd3Expert> {
    let routes = (0..experts)
        .map(|expert| ExpertRoute {
            expert,
            score: expert as f32,
        })
        .collect::<Vec<_>>();
    let scheduled_routes = ScheduledExpertRoutes::from_routes(layer, routes, 1.0).unwrap();
    ScheduledExpertSet::from_parts(
        scheduled_routes,
        (0..experts)
            .map(|expert| DummyCmd3Expert::whole_slot(layer, expert))
            .collect(),
    )
    .unwrap()
}

#[test]
fn scheduled_graph_preserves_the_declared_stage_order() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

    let stages: Vec<_> = graph.stages().iter().map(|stage| stage.stage).collect();

    assert_eq!(stages, FlashMoeGraphStage::ALL);
    assert!(graph.declares_scheduler_owned_expert_reads());
}

#[test]
fn scheduled_graph_exposes_the_upstream_command_sequence() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd_sequence: Vec<_> = graph
        .cmd_sequence()
        .iter()
        .map(|stage| stage.stage)
        .collect();

    assert_eq!(
        cmd_sequence,
        vec![
            FlashMoeGraphStage::DeferredPreviousCmd3,
            FlashMoeGraphStage::Cmd1AttentionProjections,
            FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection,
        ]
    );
    assert_eq!(
        graph
            .stage(FlashMoeGraphStage::RoutingSoftmaxTopK)
            .placement,
        FlashMoeStagePlacement::CpuDeclared
    );
}

#[test]
fn scheduled_graph_builds_explicit_cmd1_cmd2_and_cmd3_descriptors() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

    let cmd1 = graph
        .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::DeferredMetalNextNormed)
        .unwrap();
    let cmd2 = graph
        .build_cmd2_post_attention(
            14,
            4,
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
        )
        .unwrap();
    let routing = graph
        .build_routing_topk(
            14,
            512,
            4,
            ScheduledRoutingCandidateSource::MetalRouterScoresReadback,
        )
        .unwrap();
    let cmd3 = graph
        .build_cmd3_expert_phase(
            14,
            4,
            ScheduledCmd3InputSource::MetalPostAttentionPrep,
            ScheduledSharedExpertSource::ResidentProjections,
            ScheduledNextNormSource::CpuVisibleWeights,
        )
        .unwrap();

    assert_eq!(
        cmd1.stage.stage,
        FlashMoeGraphStage::Cmd1AttentionProjections
    );
    assert_eq!(cmd1.stage.placement, FlashMoeStagePlacement::Metal);
    assert_eq!(cmd1.layer, 14);
    assert_eq!(
        cmd1.input,
        ScheduledCmd1InputSource::DeferredMetalNextNormed
    );
    assert_eq!(
        cmd2.stage.stage,
        FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection
    );
    assert_eq!(cmd2.stage.placement, FlashMoeStagePlacement::Metal);
    assert_eq!(cmd2.layer, 14);
    assert_eq!(cmd2.active_experts, 4);
    assert_eq!(
        cmd2.attention,
        ScheduledCmd2AttentionSource::MetalAttentionValues
    );
    assert_eq!(cmd2.residual, ScheduledCmd2ResidualSource::MetalBuffer);
    assert_eq!(routing.stage.stage, FlashMoeGraphStage::RoutingSoftmaxTopK);
    assert_eq!(routing.stage.placement, FlashMoeStagePlacement::CpuDeclared);
    assert_eq!(routing.layer, 14);
    assert_eq!(routing.experts, 512);
    assert_eq!(routing.active_experts, 4);
    assert_eq!(
        routing.source,
        ScheduledRoutingCandidateSource::MetalRouterScoresReadback
    );
    assert_eq!(
        cmd3.stage.stage,
        FlashMoeGraphStage::Cmd3ExpertAndSharedCombine
    );
    assert_eq!(cmd3.stage.placement, FlashMoeStagePlacement::Metal);
    assert_eq!(cmd3.layer, 14);
    assert_eq!(cmd3.expert_count, 4);
    assert_eq!(cmd3.input, ScheduledCmd3InputSource::MetalPostAttentionPrep);
    assert_eq!(
        cmd3.shared,
        ScheduledSharedExpertSource::ResidentProjections
    );
    assert_eq!(cmd3.next_norm, ScheduledNextNormSource::CpuVisibleWeights);
}

#[test]
fn scheduled_attention_math_resolves_declared_cpu_kv_state() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let attention = graph.build_attention_math(14, 9).unwrap();

    let output = attention
        .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(9, 14, 128, 128))
        .unwrap();

    assert_eq!(
        output.implementation(),
        ScheduledAttentionMathImplementation::CpuKvCache
    );
    assert_eq!(output.state().position(), 9);
    assert_eq!(output.state().layer(), 14);
    assert!(output.validate_execution_state(14, 9, 128).is_ok());
}

#[test]
fn scheduled_glm_attention_resolves_distinct_compressed_mla_state() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let mut graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let attention = graph
        .stages
        .iter_mut()
        .find(|stage| stage.stage == FlashMoeGraphStage::AttentionMath)
        .unwrap();
    attention.implementation = FlashMoeStageImplementation::GlmMlaCpuWeightAbsorption;
    let attention = graph.build_attention_math(14, 9).unwrap();

    let output = attention
        .resolve_mla_kv_state(FlashMoeMlaKvState::cpu_visible(9, 14, 512, 64))
        .unwrap();

    assert_eq!(
        output.implementation(),
        ScheduledAttentionMathImplementation::CpuGlmMlaWeightAbsorption
    );
    assert_eq!(output.state().latent_len(), 512);
    assert_eq!(output.state().rotary_len(), 64);
    assert!(
        attention
            .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(9, 14, 512, 512))
            .unwrap_err()
            .to_string()
            .contains("requires compressed MLA KV state")
    );
}

#[test]
fn scheduled_attention_math_rejects_stage_without_executor() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let mut graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let attention = graph
        .stages
        .iter_mut()
        .find(|stage| stage.stage == FlashMoeGraphStage::AttentionMath)
        .unwrap();
    attention.placement = FlashMoeStagePlacement::Metal;
    attention.implementation = FlashMoeStageImplementation::MetalResidentQ4AttentionProjections;

    let err = graph.build_attention_math(14, 9).unwrap_err();

    assert_eq!(err.stage, FlashMoeGraphStage::AttentionMath);
    assert!(
        err.to_string().contains("has no scheduled executor"),
        "{err}"
    );
}

#[test]
fn scheduled_attention_math_rejects_mismatched_kv_state_without_fallback() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let attention = graph.build_attention_math(14, 9).unwrap();

    let placement_err = attention
        .resolve_kv_state(FlashMoeFullAttentionKvState::gpu_resident(9, 14, 128, 128))
        .unwrap_err();
    assert!(
        placement_err
            .to_string()
            .contains("requires CpuVisible KV state"),
        "{placement_err:#}"
    );

    let layer_err = attention
        .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(9, 15, 128, 128))
        .unwrap_err();
    assert!(
        layer_err
            .to_string()
            .contains("does not match KV state layer"),
        "{layer_err:#}"
    );

    let position_err = attention
        .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(8, 14, 128, 128))
        .unwrap_err();
    assert!(
        position_err
            .to_string()
            .contains("does not match KV state position"),
        "{position_err:#}"
    );

    let width_err = attention
        .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(9, 14, 128, 127))
        .unwrap_err();
    assert!(
        width_err
            .to_string()
            .contains("KV state is not declared graph state"),
        "{width_err:#}"
    );

    let output = attention
        .resolve_kv_state(FlashMoeFullAttentionKvState::cpu_visible(9, 14, 128, 128))
        .unwrap();
    let execution_err = output.validate_execution_state(14, 9, 256).unwrap_err();
    assert!(
        execution_err
            .to_string()
            .contains("does not match execution width"),
        "{execution_err:#}"
    );
}

#[test]
fn scheduled_cmd1_submission_builds_resolved_command() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd1 = graph
        .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::DeferredMetalNextNormed)
        .unwrap();

    let command =
        ScheduledCmd1Submission::new(cmd1, ScheduledCmd1InputSource::DeferredMetalNextNormed)
            .unwrap()
            .into_cmd1_command();

    assert_eq!(command.layer, 14);
    assert_eq!(command.cmd1.layer, 14);
    assert_eq!(
        command.input.scheduled_cmd1_input_source(),
        ScheduledCmd1InputSource::DeferredMetalNextNormed
    );

    let resolved = command
        .into_resolved_command(FlashMoeCmd1InputState::gpu_next_layer_normed(
            14,
            FlashMoeGpuBufferDescriptor::next_layer_normed(4096),
        ))
        .unwrap();
    assert_eq!(resolved.layer, 14);
    assert_eq!(resolved.cmd1.layer, 14);
    assert_eq!(
        resolved.input.scheduled_cmd1_input_source(),
        ScheduledCmd1InputSource::DeferredMetalNextNormed
    );
    assert_eq!(resolved.input_state.layer(), 14);
    assert_eq!(resolved.input_state.len(), 4096);
    assert_eq!(
        resolved.input_state.placement(),
        FlashMoeStatePlacement::GpuResident
    );
}

#[test]
fn scheduled_graph_builds_cmd1_submission_and_rejects_stale_stage() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd1 = graph
        .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::CpuNormedHidden)
        .unwrap();

    graph
        .build_cmd1_submission(cmd1, ScheduledCmd1InputSource::CpuNormedHidden)
        .unwrap();

    let mut stale_graph = graph.clone();
    stale_graph
        .stages
        .iter_mut()
        .find(|stage| stage.stage == FlashMoeGraphStage::Cmd1AttentionProjections)
        .unwrap()
        .implementation = FlashMoeStageImplementation::DeferredMetalCmd3;

    let err = stale_graph
        .build_cmd1_submission(cmd1, ScheduledCmd1InputSource::CpuNormedHidden)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("does not match scheduled graph CMD1 stage"),
        "{err:#}"
    );
}

#[test]
fn scheduled_cmd1_resolves_declared_input_state() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cpu_cmd1 = graph
        .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::CpuNormedHidden)
        .unwrap();
    let cpu_command =
        ScheduledCmd1Submission::new(cpu_cmd1, ScheduledCmd1InputSource::CpuNormedHidden)
            .unwrap()
            .into_cmd1_command();

    let cpu_resolved = cpu_command
        .into_resolved_command(FlashMoeCmd1InputState::cpu_normed(14, 4096))
        .unwrap();
    assert_eq!(cpu_resolved.layer, 14);
    assert_eq!(cpu_resolved.input_state.len(), 4096);
    assert_eq!(
        cpu_resolved.input.scheduled_cmd1_input_source(),
        ScheduledCmd1InputSource::CpuNormedHidden
    );

    let gpu_cmd1 = graph
        .build_cmd1_attention_projections(15, ScheduledCmd1InputSource::DeferredMetalNextNormed)
        .unwrap();
    let gpu_command =
        ScheduledCmd1Submission::new(gpu_cmd1, ScheduledCmd1InputSource::DeferredMetalNextNormed)
            .unwrap()
            .into_cmd1_command();

    let gpu_resolved = gpu_command
        .into_resolved_command(FlashMoeCmd1InputState::gpu_next_layer_normed(
            15,
            FlashMoeGpuBufferDescriptor::next_layer_normed(4096),
        ))
        .unwrap();
    assert_eq!(gpu_resolved.layer, 15);
    assert_eq!(
        gpu_resolved.input_state.placement(),
        FlashMoeStatePlacement::GpuResident
    );
}

#[test]
fn scheduled_cmd1_rejects_mismatched_input_state_without_fallback() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cpu_cmd1 = graph
        .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::CpuNormedHidden)
        .unwrap();
    let cpu_command = || {
        ScheduledCmd1Submission::new(cpu_cmd1, ScheduledCmd1InputSource::CpuNormedHidden)
            .unwrap()
            .into_cmd1_command()
    };

    let layer_err = cpu_command()
        .into_resolved_command(FlashMoeCmd1InputState::cpu_normed(15, 4096))
        .unwrap_err();
    assert!(
        layer_err
            .to_string()
            .contains("does not match input state layer"),
        "{layer_err:#}"
    );

    let source_err = cpu_command()
        .into_resolved_command(FlashMoeCmd1InputState::gpu_next_layer_normed(
            14,
            FlashMoeGpuBufferDescriptor::next_layer_normed(4096),
        ))
        .unwrap_err();
    assert!(
        source_err
            .to_string()
            .contains("CPU input requires CpuVisible Normed state"),
        "{source_err:#}"
    );

    let empty_err = cpu_command()
        .into_resolved_command(FlashMoeCmd1InputState::cpu_normed(14, 0))
        .unwrap_err();
    assert!(
        empty_err
            .to_string()
            .contains("CMD1 input is not declared graph state"),
        "{empty_err:#}"
    );
}

#[test]
fn scheduled_cmd1_submission_rejects_mismatched_input_without_fallback() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd1 = graph
        .build_cmd1_attention_projections(14, ScheduledCmd1InputSource::CpuNormedHidden)
        .unwrap();

    let err = ScheduledCmd1Submission::new(cmd1, ScheduledCmd1InputSource::DeferredMetalNextNormed)
        .unwrap_err();

    assert!(err.to_string().contains("does not match submitted input"));
}

#[test]
fn scheduled_routing_selects_cpu_topk_from_declared_scores() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(3, 5, 3, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap();

    let logits = [0.0, 2.0, 2.0, -1.0, 1.0];
    let selected = routing
        .select_from_scores(&ScheduledRoutingScoreView::new(
            3,
            ScheduledRoutingCandidateSource::CpuRouterScores,
            &logits,
        ))
        .unwrap();

    let mut probabilities = logits.to_vec();
    softmax_in_place(&mut probabilities);
    assert_eq!(selected, routing_top_k(&probabilities, 3));

    let layer_err = routing
        .select_from_scores(&ScheduledRoutingScoreView::new(
            4,
            ScheduledRoutingCandidateSource::CpuRouterScores,
            &[0.0, 2.0, 2.0, -1.0, 1.0],
        ))
        .unwrap_err();
    assert!(
        layer_err
            .to_string()
            .contains("does not match submitted score layer"),
        "{layer_err:#}"
    );

    let source_err = routing
        .select_from_scores(&ScheduledRoutingScoreView::new(
            3,
            ScheduledRoutingCandidateSource::MetalRouterScoresReadback,
            &[0.0, 2.0, 2.0, -1.0, 1.0],
        ))
        .unwrap_err();
    assert!(
        source_err
            .to_string()
            .contains("does not match submitted score source"),
        "{source_err:#}"
    );

    let projection = dummy_router_projection(3, 5, 4096);
    let projected_selected = routing
        .select_from_scores(&ScheduledRoutingScoreView::from_router_projection(
            ScheduledRoutingCandidateSource::CpuRouterScores,
            &projection,
            &[0.0, 2.0, 2.0, -1.0, 1.0],
        ))
        .unwrap();
    assert_eq!(projected_selected, selected);

    let wrong_experts = dummy_router_projection(3, 4, 4096);
    let projection_err = routing
        .select_from_scores(&ScheduledRoutingScoreView::from_router_projection(
            ScheduledRoutingCandidateSource::CpuRouterScores,
            &wrong_experts,
            &[0.0, 2.0, 2.0, -1.0, 1.0],
        ))
        .unwrap_err();
    assert!(
        projection_err
            .to_string()
            .contains("does not match submitted router projection experts"),
        "{projection_err:#}"
    );
}

#[test]
fn scheduled_router_score_projection_command_declares_score_state() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(3, 5, 2, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap();
    let projection = dummy_router_projection(3, 5, 8);

    let command = routing
        .build_score_projection_command(Some(projection.clone()), 8)
        .unwrap();

    assert_eq!(command.routing, routing);
    assert_eq!(
        command.state,
        FlashMoeRoutingOutputState::cpu_router_scores(3, 5, 2)
    );
    assert_eq!(command.projection, Some(projection));
    assert_eq!(command.hidden_width, 8);
}

#[test]
fn scheduled_router_score_projection_command_finalizes_declared_batch() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(3, 5, 2, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap();
    let projection = dummy_router_projection(3, 5, 8);

    let batch = routing
        .build_score_projection_command(Some(projection.clone()), 8)
        .unwrap()
        .into_score_batch(vec![0.0, 1.0, 2.0, 3.0, 4.0])
        .unwrap();
    assert_eq!(
        batch.state(),
        FlashMoeRoutingOutputState::cpu_router_scores(3, 5, 2)
    );
    assert_eq!(batch.projection, Some(projection.clone()));
    assert_eq!(batch.scores, vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    let err = routing
        .build_score_projection_command(Some(projection), 8)
        .unwrap()
        .into_score_batch(vec![0.0, 1.0])
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("has 2 scores for 5 declared experts"),
        "{err:#}"
    );
}

#[test]
fn scheduled_router_score_projection_command_selects_routing_command() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(3, 5, 2, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap();
    let projection = dummy_router_projection(3, 5, 8);

    let command = routing
        .build_score_projection_command(Some(projection.clone()), 8)
        .unwrap();
    let execution = command.projection_execution().unwrap();
    assert_eq!(execution.layer, 3);
    assert_eq!(execution.experts, 5);
    assert_eq!(execution.hidden_width, 8);
    assert_eq!(execution.tensor_name, projection.tensor_name);

    let routed = command
        .into_routing_command(vec![0.5, 9.0, -1.0, 4.0, 3.0])
        .unwrap();
    assert_eq!(routed.layer, 3);
    assert_eq!(routed.active_experts, 2);
    assert_eq!(
        routed.source,
        ScheduledRoutingCandidateSource::CpuRouterScores
    );
    let mut probabilities = vec![0.5, 9.0, -1.0, 4.0, 3.0];
    softmax_in_place(&mut probabilities);
    assert_eq!(routed.routes, routing_top_k(&probabilities, 2));

    let err = routing
        .build_score_projection_command(Some(projection), 8)
        .unwrap()
        .into_routing_command(vec![0.5, f32::NAN, -1.0, 4.0, 3.0])
        .unwrap_err();
    assert!(
        err.to_string().contains("score for expert 1 is not finite"),
        "{err:#}"
    );

    let missing_projection_err = routing
        .build_score_projection_command(None, 8)
        .unwrap()
        .projection_execution()
        .unwrap_err();
    assert!(
        missing_projection_err
            .to_string()
            .contains("has no declared resident projection implementation"),
        "{missing_projection_err:#}"
    );
}

#[test]
fn scheduled_graph_builds_router_score_projection_command() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let projection = dummy_router_projection(3, 5, 8);

    let command = graph
        .build_router_score_projection(3, 5, 2, Some(projection.clone()), 8)
        .unwrap();

    assert_eq!(command.routing.layer, 3);
    assert_eq!(command.routing.experts, 5);
    assert_eq!(command.routing.active_experts, 2);
    assert_eq!(
        command.routing.source,
        ScheduledRoutingCandidateSource::CpuRouterScores
    );
    assert_eq!(
        command.state,
        FlashMoeRoutingOutputState::cpu_router_scores(3, 5, 2)
    );
    assert_eq!(command.projection, Some(projection));

    let err = graph
        .build_router_score_projection(3, 5, 2, Some(dummy_router_projection(3, 5, 4)), 8)
        .unwrap_err();
    assert_eq!(err.stage, FlashMoeGraphStage::RoutingSoftmaxTopK);
    assert!(
        err.to_string()
            .contains("invalid scheduled router score projection"),
        "{err:#}"
    );
}

#[test]
fn scheduled_router_score_projection_command_rejects_mismatched_projection() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(3, 5, 2, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap();

    let width_err = routing
        .build_score_projection_command(Some(dummy_router_projection(3, 5, 4)), 8)
        .unwrap_err();
    assert!(
        width_err
            .to_string()
            .contains("hidden width 4 does not match submitted hidden width 8"),
        "{width_err:#}"
    );

    let preselected = graph
        .build_routing_topk(
            3,
            5,
            2,
            ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
        )
        .unwrap();
    let source_err = preselected
        .build_score_projection_command(None, 8)
        .unwrap_err();
    assert!(
        source_err
            .to_string()
            .contains("requires CPU router-score routing"),
        "{source_err:#}"
    );
}

#[test]
fn scheduled_routing_builds_command_from_declared_scores() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(3, 5, 3, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap();
    let output = routing
        .validate_output_state(FlashMoeRoutingOutputState::cpu_router_scores(3, 5, 3))
        .unwrap();

    let command = routing
        .select_command_from_output_scores(
            output,
            &ScheduledRoutingScoreView::new(
                3,
                ScheduledRoutingCandidateSource::CpuRouterScores,
                &[0.1, 0.9, 0.2, 1.5, -0.2],
            ),
        )
        .unwrap();

    assert_eq!(command.layer, 3);
    assert_eq!(command.active_experts, 3);
    assert_eq!(
        command.source,
        ScheduledRoutingCandidateSource::CpuRouterScores
    );
    assert_eq!(command.routing.layer, 3);
    assert_eq!(command.routes.len(), 3);
    assert_eq!(command.routes[0].0, 3);
}

#[test]
fn scheduled_routing_command_validates_active_expert_issue_shape() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(3, 5, 3, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap();
    let output = routing
        .validate_output_state(FlashMoeRoutingOutputState::cpu_router_scores(3, 5, 3))
        .unwrap();
    let command = routing
        .select_command_from_output_scores(
            output,
            &ScheduledRoutingScoreView::new(
                3,
                ScheduledRoutingCandidateSource::CpuRouterScores,
                &[0.1, 0.9, 0.2, 1.5, -0.2],
            ),
        )
        .unwrap();

    command.validate_for_active_expert_issue().unwrap();

    let mut wrong_count = command.clone();
    wrong_count.active_experts = 2;
    let count_err = wrong_count.validate_for_active_expert_issue().unwrap_err();
    assert!(
        count_err
            .to_string()
            .contains("does not match routing descriptor active expert count"),
        "{count_err:#}"
    );

    let mut repeated = command;
    repeated.routes[1].0 = repeated.routes[0].0;
    let repeated_err = repeated.validate_for_active_expert_issue().unwrap_err();
    assert!(
        repeated_err
            .to_string()
            .contains("selected expert 3 more than once"),
        "{repeated_err:#}"
    );
}

#[test]
fn scheduled_routing_validates_preselected_fused_prep_candidates() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(
            3,
            8,
            4,
            ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
        )
        .unwrap();

    let selected = routing
        .validate_preselected(&[(7, 3.0), (1, 2.0), (3, 1.0), (5, 0.0)])
        .unwrap();
    assert_eq!(selected, vec![(7, 3.0), (1, 2.0), (3, 1.0), (5, 0.0)]);

    let duplicate_err = routing
        .validate_preselected(&[(7, 3.0), (1, 2.0), (7, 1.0), (5, 0.0)])
        .unwrap_err();
    assert!(
        duplicate_err
            .to_string()
            .contains("selected expert 7 more than once"),
        "{duplicate_err:#}"
    );

    let source_err = routing
        .select_from_scores(&ScheduledRoutingScoreView::new(
            3,
            ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ))
        .unwrap_err();
    assert!(
        source_err
            .to_string()
            .contains("must submit preselected CPU topK candidates"),
        "{source_err:#}"
    );
}

#[test]
fn scheduled_routing_validates_declared_cmd2_output_state() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(
            3,
            8,
            4,
            ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
        )
        .unwrap();

    let output = routing
        .validate_output_state(
            FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(3, 8, 4),
        )
        .unwrap();
    assert_eq!(output.routing, routing);
    assert_eq!(
        output.state().source(),
        FlashMoeRoutingOutputSource::FusedMetalPostAttentionPrepCpuTopK
    );

    let layer_err = routing
        .validate_output_state(
            FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(4, 8, 4),
        )
        .unwrap_err();
    assert!(
        layer_err
            .to_string()
            .contains("does not match submitted routing output layer"),
        "{layer_err:#}"
    );

    let source_err = routing
        .validate_output_state(FlashMoeRoutingOutputState::cpu_router_scores(3, 8, 4))
        .unwrap_err();
    assert!(
        source_err
            .to_string()
            .contains("does not match submitted routing output source"),
        "{source_err:#}"
    );
}

#[test]
fn scheduled_routing_builds_command_from_preselected_candidates() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(
            4,
            8,
            2,
            ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
        )
        .unwrap();
    let output = routing
        .validate_output_state(
            FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(4, 8, 2),
        )
        .unwrap();

    let command = routing
        .command_from_preselected_output(output, &[(7, 0.75), (1, 0.25)])
        .unwrap();

    assert_eq!(command.layer, 4);
    assert_eq!(command.active_experts, 2);
    assert_eq!(
        command.source,
        ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK
    );
    assert_eq!(command.routes, vec![(7, 0.75), (1, 0.25)]);
}

#[test]
fn scheduled_routing_rejects_wrong_stage_placement_or_bounds() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let mut graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .stages
        .iter_mut()
        .find(|stage| stage.stage == FlashMoeGraphStage::RoutingSoftmaxTopK)
        .unwrap();
    routing.placement = FlashMoeStagePlacement::Metal;

    let err = graph
        .build_routing_topk(0, 8, 4, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap_err();
    assert_eq!(err.family, graph.family());
    assert_eq!(err.stage, FlashMoeGraphStage::RoutingSoftmaxTopK);
    assert!(
        err.to_string()
            .contains("routing softmax/topK stage must be implemented"),
        "{err:#}"
    );

    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(0, 2, 4, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap();
    let bounds_err = routing
        .select_from_scores(&ScheduledRoutingScoreView::new(
            0,
            ScheduledRoutingCandidateSource::CpuRouterScores,
            &[1.0, 2.0],
        ))
        .unwrap_err();
    assert!(
        bounds_err
            .to_string()
            .contains("active expert count 4 exceeds expert count 2"),
        "{bounds_err:#}"
    );
}

#[test]
fn scheduled_graph_rejects_non_metal_cmd1_builder() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let mut graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd1 = graph
        .stages
        .iter_mut()
        .find(|stage| stage.stage == FlashMoeGraphStage::Cmd1AttentionProjections)
        .unwrap();
    cmd1.placement = FlashMoeStagePlacement::CpuDeclared;

    let err = graph
        .build_cmd1_attention_projections(0, ScheduledCmd1InputSource::CpuNormedHidden)
        .unwrap_err();

    assert_eq!(err.family, graph.family());
    assert_eq!(err.stage, FlashMoeGraphStage::Cmd1AttentionProjections);
    assert!(
        err.to_string()
            .contains("CMD1 attention projection stage must be implemented"),
        "{err:#}"
    );
}

#[test]
fn scheduled_cmd2_submission_validates_attention_and_residual_sources() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd2 = graph
        .build_cmd2_post_attention(
            11,
            4,
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
        )
        .unwrap();

    let submission = ScheduledCmd2Submission::new(
        cmd2,
        ScheduledCmd2PhaseInputs::new(
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
            4096,
            4096,
        ),
    )
    .unwrap();

    assert_eq!(submission.cmd2.layer, 11);
    assert_eq!(submission.cmd2.active_experts, 4);
}

#[test]
fn scheduled_cmd2_submission_builds_resolved_command() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd2 = graph
        .build_cmd2_post_attention(
            11,
            4,
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
        )
        .unwrap();

    let command = ScheduledCmd2Submission::new(
        cmd2,
        ScheduledCmd2PhaseInputs::new(
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
            4096,
            4096,
        ),
    )
    .unwrap()
    .into_cmd2_command();

    assert_eq!(command.layer, 11);
    assert_eq!(command.active_experts, 4);
    assert_eq!(
        command.inputs.scheduled_cmd2_attention_source(),
        ScheduledCmd2AttentionSource::MetalAttentionValues
    );
    assert_eq!(
        command.inputs.scheduled_cmd2_residual_source(),
        ScheduledCmd2ResidualSource::MetalBuffer
    );
    let input_state = command.input_state();
    assert_eq!(input_state.attention().len(), 4096);
    assert_eq!(
        input_state.attention().placement(),
        FlashMoeStatePlacement::GpuResident
    );
    assert_eq!(input_state.residual().len(), 4096);
    assert!(input_state.is_declared_graph_state());
}

#[test]
fn scheduled_cmd2_input_descriptors_build_declared_command() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

    let command = graph
        .build_cmd2_command(
            11,
            4,
            ScheduledCmd2PhaseInputs::from_inputs(
                ScheduledCmd2AttentionInput::metal_values(4096),
                ScheduledCmd2ResidualInput::cpu_hidden(4096),
            ),
        )
        .unwrap();

    assert_eq!(
        command.cmd2.attention,
        ScheduledCmd2AttentionSource::MetalAttentionValues
    );
    assert_eq!(
        command.cmd2.residual,
        ScheduledCmd2ResidualSource::CpuHidden
    );
    assert_eq!(
        command.input_state().attention().placement(),
        FlashMoeStatePlacement::GpuResident
    );
    assert_eq!(
        command.input_state().residual().placement(),
        FlashMoeStatePlacement::CpuVisible
    );
    assert_eq!(command.input_state().attention().len(), 4096);
    assert_eq!(command.input_state().residual().len(), 4096);
}

#[test]
fn scheduled_cmd2_input_descriptors_reject_empty_graph_state_without_fallback() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

    let err = graph
        .build_cmd2_command(
            11,
            4,
            ScheduledCmd2PhaseInputs::from_inputs(
                ScheduledCmd2AttentionInput::cpu_values(0),
                ScheduledCmd2ResidualInput::cpu_hidden(4096),
            ),
        )
        .unwrap_err();

    assert!(
        err.to_string()
            .contains("scheduled CMD2 input is not declared graph state"),
        "{err:#}"
    );
}

#[test]
fn scheduled_graph_builds_cmd2_submission_and_rejects_stale_stage() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd2 = graph
        .build_cmd2_post_attention(
            11,
            4,
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
        )
        .unwrap();

    graph
        .build_cmd2_submission(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
                4096,
                4096,
            ),
        )
        .unwrap();

    let mut stale_graph = graph.clone();
    stale_graph
        .stages
        .iter_mut()
        .find(|stage| stage.stage == FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection)
        .unwrap()
        .implementation = FlashMoeStageImplementation::QwenTextInput;

    let err = stale_graph
        .build_cmd2_submission(
            cmd2,
            ScheduledCmd2PhaseInputs::new(
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
                4096,
                4096,
            ),
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("does not match scheduled graph CMD2 stage"),
        "{err:#}"
    );
}

#[test]
fn scheduled_cmd2_resolves_declared_post_attention_prep_output() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd2 = graph
        .build_cmd2_post_attention(
            11,
            4,
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
        )
        .unwrap();
    let command = ScheduledCmd2Submission::new(
        cmd2,
        ScheduledCmd2PhaseInputs::new(
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
            4096,
            4096,
        ),
    )
    .unwrap()
    .into_cmd2_command();

    let output = command
        .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(11, 4096, 512, 4))
        .unwrap();

    assert_eq!(output.layer, 11);
    assert_eq!(output.active_experts, 4);
    assert_eq!(output.width(), 4096);
    assert_eq!(output.input_state, command.input_state());
    assert_eq!(output.routing().layer(), 11);
    assert_eq!(output.routing().experts(), 512);
}

#[test]
fn scheduled_cmd2_output_builds_preselected_routing_command() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd2 = graph
        .build_cmd2_post_attention(
            11,
            2,
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
        )
        .unwrap();
    let command = ScheduledCmd2Submission::new(
        cmd2,
        ScheduledCmd2PhaseInputs::new(
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
            4096,
            4096,
        ),
    )
    .unwrap()
    .into_cmd2_command();
    let output = command
        .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(11, 4096, 512, 2))
        .unwrap();

    let routing = output
        .command_from_preselected_routes(&graph, &[(7, 0.75), (3, 0.25)])
        .unwrap();

    assert_eq!(routing.layer, 11);
    assert_eq!(routing.active_experts, 2);
    assert_eq!(
        routing.source,
        ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK
    );
    assert_eq!(routing.routes, vec![(7, 0.75), (3, 0.25)]);
}

#[test]
fn scheduled_cmd2_command_builds_preselected_routing_command() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd2 = graph
        .build_cmd2_post_attention(
            11,
            2,
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
        )
        .unwrap();
    let command = ScheduledCmd2Submission::new(
        cmd2,
        ScheduledCmd2PhaseInputs::new(
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
            4096,
            4096,
        ),
    )
    .unwrap()
    .into_cmd2_command();

    let routing = command
        .command_from_post_attention_prep_routes(
            &graph,
            FlashMoePostAttentionPrepState::new(11, 4096, 512, 2),
            &[(7, 0.75), (3, 0.25)],
        )
        .unwrap();

    assert_eq!(routing.layer, 11);
    assert_eq!(routing.active_experts, 2);
    assert_eq!(
        routing.source,
        ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK
    );
    assert_eq!(routing.routes, vec![(7, 0.75), (3, 0.25)]);

    let err = command
        .command_from_post_attention_prep_routes(
            &graph,
            FlashMoePostAttentionPrepState::new(11, 4096, 512, 2),
            &[(7, 0.75)],
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("received 1 preselected experts; expected 2"),
        "{err:#}"
    );
}

#[test]
fn scheduled_cmd2_rejects_missing_post_attention_prep_without_cpu_fallback() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let command = graph
        .build_cmd2_command(
            11,
            4,
            ScheduledCmd2PhaseInputs::from_inputs(
                ScheduledCmd2AttentionInput::metal_values(4096),
                ScheduledCmd2ResidualInput::metal_buffer(4096),
            ),
        )
        .unwrap();

    let err = command
        .reject_missing_post_attention_prep("test missing prep")
        .unwrap_err();

    assert!(
            err.to_string().contains(
                "FlashMoe unsupported scheduled CMD2 path: layer 11 declares CMD2 post-attention and routing projection implementation"
            ),
            "{err:#}"
        );
    assert!(err.to_string().contains("test missing prep"), "{err:#}");
}

#[test]
fn scheduled_cmd2_output_rejects_mismatched_preselected_routes() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd2 = graph
        .build_cmd2_post_attention(
            11,
            2,
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
        )
        .unwrap();
    let command = ScheduledCmd2Submission::new(
        cmd2,
        ScheduledCmd2PhaseInputs::new(
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
            4096,
            4096,
        ),
    )
    .unwrap()
    .into_cmd2_command();
    let output = command
        .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(11, 4096, 512, 2))
        .unwrap();

    let err = output
        .command_from_preselected_routes(&graph, &[(7, 0.75)])
        .unwrap_err();

    assert!(
        err.to_string()
            .contains("received 1 preselected experts; expected 2"),
        "{err:#}"
    );
}

#[test]
fn scheduled_cmd2_rejects_mismatched_post_attention_prep_output() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd2 = graph
        .build_cmd2_post_attention(
            11,
            4,
            ScheduledCmd2AttentionSource::CpuAttentionValues,
            ScheduledCmd2ResidualSource::CpuHidden,
        )
        .unwrap();
    let command = ScheduledCmd2Submission::new(
        cmd2,
        ScheduledCmd2PhaseInputs::new(
            ScheduledCmd2AttentionSource::CpuAttentionValues,
            ScheduledCmd2ResidualSource::CpuHidden,
            4096,
            4096,
        ),
    )
    .unwrap()
    .into_cmd2_command();

    let layer_err = command
        .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(12, 4096, 512, 4))
        .unwrap_err();
    assert!(
        layer_err
            .to_string()
            .contains("does not match post-attention prep layer"),
        "{layer_err:#}"
    );

    let active_err = command
        .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(11, 4096, 512, 3))
        .unwrap_err();
    assert!(
        active_err
            .to_string()
            .contains("does not match post-attention prep active expert count"),
        "{active_err:#}"
    );

    let width_err = command
        .resolve_post_attention_prep(FlashMoePostAttentionPrepState::new(11, 2048, 512, 4))
        .unwrap_err();
    assert!(
        width_err
            .to_string()
            .contains("does not match residual input width"),
        "{width_err:#}"
    );
}

#[test]
fn scheduled_cmd2_submission_rejects_mismatched_sources_without_fallback() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd2 = graph
        .build_cmd2_post_attention(
            11,
            4,
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
        )
        .unwrap();

    let attention_err = ScheduledCmd2Submission::new(
        cmd2,
        ScheduledCmd2PhaseInputs::new(
            ScheduledCmd2AttentionSource::CpuAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
            4096,
            4096,
        ),
    )
    .unwrap_err();
    assert!(
        attention_err
            .to_string()
            .contains("does not match submitted source")
    );

    let residual_err = ScheduledCmd2Submission::new(
        cmd2,
        ScheduledCmd2PhaseInputs::new(
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::CpuHidden,
            4096,
            4096,
        ),
    )
    .unwrap_err();
    assert!(
        residual_err
            .to_string()
            .contains("does not match submitted source")
    );

    let invalid_state_err = ScheduledCmd2Submission::new(
        cmd2,
        ScheduledCmd2PhaseInputs::new(
            ScheduledCmd2AttentionSource::MetalAttentionValues,
            ScheduledCmd2ResidualSource::MetalBuffer,
            0,
            4096,
        ),
    )
    .unwrap_err();
    assert!(
        invalid_state_err
            .to_string()
            .contains("input is not declared graph state")
    );
}

#[test]
fn scheduled_cmd3_cpu_input_declares_whole_phase_or_errors() {
    let normed = [1.0f32, 2.0, 3.0];
    let residual = [4.0f32, 5.0, 6.0];
    let input = ScheduledCmd3CpuInput::new(9, &normed, &residual).unwrap();
    assert_eq!(input.width(), 3);
    assert_eq!(
        input.scheduled_cmd3_input_source(),
        ScheduledCmd3InputSource::CpuNormedResidualUpload
    );
    assert_eq!(
        input.scheduled_cmd3_input_state(9),
        FlashMoeCmd3InputState::cpu_normed_residual(9, 3, 3)
    );

    let mismatched = ScheduledCmd3CpuInput::new(9, &normed, &residual[..2]).unwrap_err();
    assert!(
        mismatched
            .to_string()
            .contains("is not a declared graph state"),
        "{mismatched:#}"
    );

    let empty = ScheduledCmd3CpuInput::new(9, &[], &[]).unwrap_err();
    assert!(
        empty.to_string().contains("is not a declared graph state"),
        "{empty:#}"
    );
}

#[test]
fn scheduled_cmd3_metal_post_attention_input_declares_prep_state_or_errors() {
    let state = FlashMoePostAttentionPrepState::new(4, 8, 16, 2);
    let input = ScheduledCmd3MetalPostAttentionInput::new(state, 2).unwrap();
    assert_eq!(input.width(), 8);
    assert_eq!(input.state(), state);
    assert_eq!(
        input.scheduled_cmd3_input_source(),
        ScheduledCmd3InputSource::MetalPostAttentionPrep
    );
    assert_eq!(
        input.scheduled_cmd3_input_state(4),
        FlashMoeCmd3InputState::metal_post_attention_prep(4, state)
    );

    let route_err = ScheduledCmd3MetalPostAttentionInput::new(state, 1).unwrap_err();
    assert!(
        route_err
            .to_string()
            .contains("state declares 2 active experts but prep carries 1 routes"),
        "{route_err:#}"
    );

    let empty = FlashMoePostAttentionPrepState::new(4, 0, 16, 2);
    let empty_err = ScheduledCmd3MetalPostAttentionInput::new(empty, 2).unwrap_err();
    assert!(
        empty_err
            .to_string()
            .contains("prep state is not a declared graph state"),
        "{empty_err:#}"
    );
}

#[test]
fn scheduled_cmd3_submission_validates_batch_and_sources() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let scheduled = dummy_scheduled_experts(7, 2);
    let cmd3 = graph
        .build_cmd3_expert_phase(
            7,
            2,
            ScheduledCmd3InputSource::CpuNormedResidualUpload,
            ScheduledSharedExpertSource::ResidentProjections,
            ScheduledNextNormSource::None,
        )
        .unwrap();

    let submission = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::none(),
    )
    .unwrap();

    assert_eq!(submission.position, 19);
    assert_eq!(submission.cmd3.layer, 7);
    assert_eq!(submission.scheduled.len(), 2);
    assert_eq!(submission.input_state.width(), 8);
    assert_eq!(
        submission.input_state.placement(),
        FlashMoeStatePlacement::CpuVisible
    );
}

#[test]
fn scheduled_expert_batch_resolves_cmd3_payloads_from_scheduled_experts() {
    let scheduled = dummy_scheduled_experts(7, 2);

    let payloads = scheduled.cmd3_expert_phase_payloads(8).unwrap();

    assert_eq!(payloads.len(), 2);
    let payload = payloads[0].q4();
    assert_eq!(payload.gate.rows, 4);
    assert_eq!(payload.gate.cols, 8);
    assert_eq!(payload.up.rows, 4);
    assert_eq!(payload.down.rows, 8);
    assert_eq!(payload.down.cols, 4);
}

#[test]
fn scheduled_cmd3_submission_builds_resolved_command_payloads() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let scheduled = dummy_scheduled_experts(7, 2);
    let next_norm = [1.0; 8];
    let cmd3 = graph
        .build_cmd3_expert_phase(
            7,
            2,
            ScheduledCmd3InputSource::CpuNormedResidualUpload,
            ScheduledSharedExpertSource::ResidentProjections,
            ScheduledNextNormSource::CpuVisibleWeights,
        )
        .unwrap();
    let submission = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::cpu_visible(
            "model.layers.8.input_layernorm.weight",
            &next_norm,
            8,
        )
        .unwrap(),
    )
    .unwrap();

    let command = submission.into_cmd3_command().unwrap();

    assert_eq!(command.position, 19);
    assert_eq!(command.layer, 7);
    assert_eq!(command.cmd3.layer, 7);
    assert_eq!(command.experts.len(), 2);
    assert_eq!(command.weights.len(), 2);
    assert_eq!(command.payloads.len(), 2);
    assert_eq!(
        command.input.scheduled_cmd3_input_source(),
        ScheduledCmd3InputSource::CpuNormedResidualUpload
    );
    assert_eq!(command.input_state.layer(), 7);
    assert_eq!(command.input_state.width(), 8);
    assert_eq!(
        command.input_state.placement(),
        FlashMoeStatePlacement::CpuVisible
    );
    assert_eq!(command.next_norm_weights.values().unwrap().len(), 8);
    assert_eq!(command.payloads[0].q4().gate.cols, 8);

    let output = command.resolve_output_state().unwrap();
    assert_eq!(output.cmd3, command.cmd3);
    assert_eq!(output.layer, 7);
    assert_eq!(output.input_state, command.input_state);
    let output_state = output.state();
    assert_eq!(output_state.width(), 8);
    assert!(output_state.has_next_normed());
    assert_eq!(output_state.hidden().len(), 8);
    assert_eq!(output_state.next_normed().unwrap().len(), 8);
}

#[test]
fn scheduled_graph_builds_cmd3_command_from_typed_descriptors() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let scheduled = dummy_scheduled_experts(7, 2);
    let next_norm = [1.0; 8];

    let command = graph
        .build_cmd3_command_from_descriptors(
            19,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &next_norm,
                8,
            )
            .unwrap(),
        )
        .unwrap();

    assert_eq!(command.position, 19);
    assert_eq!(command.layer, 7);
    assert_eq!(command.cmd3.layer, 7);
    assert_eq!(command.cmd3.expert_count, 2);
    assert_eq!(
        command.cmd3.input,
        ScheduledCmd3InputSource::CpuNormedResidualUpload
    );
    assert_eq!(
        command.cmd3.shared,
        ScheduledSharedExpertSource::ResidentProjections
    );
    assert_eq!(
        command.cmd3.next_norm,
        ScheduledNextNormSource::CpuVisibleWeights
    );
    assert_eq!(command.input_state.width(), 8);
    assert_eq!(command.payloads.len(), 2);
}

#[test]
fn scheduled_graph_cmd3_command_rejects_mismatched_typed_descriptor() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let scheduled = dummy_scheduled_experts(7, 2);

    let err = graph
        .build_cmd3_command_from_descriptors(
            19,
            &scheduled,
            dummy_cmd3_input_with_width(ScheduledCmd3InputSource::CpuNormedResidualUpload, 4),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::none(),
        )
        .unwrap_err();

    assert!(
        err.to_string()
            .contains("shared expert width 8 does not match input width 4"),
        "{err:#}"
    );
}

#[test]
fn scheduled_cmd3_output_state_tracks_absent_next_norm() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let scheduled = dummy_scheduled_experts(7, 2);
    let cmd3 = graph
        .build_cmd3_expert_phase(
            7,
            2,
            ScheduledCmd3InputSource::CpuNormedResidualUpload,
            ScheduledSharedExpertSource::ResidentProjections,
            ScheduledNextNormSource::None,
        )
        .unwrap();
    let command = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::none(),
    )
    .unwrap()
    .into_cmd3_command()
    .unwrap();

    let output = command.resolve_output_state().unwrap();
    assert_eq!(output.cmd3, command.cmd3);
    assert_eq!(output.layer, 7);
    assert_eq!(output.input_state, command.input_state);
    let output_state = output.state();

    assert_eq!(output_state.width(), 8);
    assert!(!output_state.has_next_normed());
    assert!(output_state.next_normed().is_none());
}

#[test]
fn scheduled_cmd3_output_accepts_declared_phase_output() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let scheduled = dummy_scheduled_experts(7, 2);
    let cmd3 = graph
        .build_cmd3_expert_phase(
            7,
            2,
            ScheduledCmd3InputSource::CpuNormedResidualUpload,
            ScheduledSharedExpertSource::ResidentProjections,
            ScheduledNextNormSource::CpuVisibleWeights,
        )
        .unwrap();
    let command = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::cpu_visible(
            "model.layers.8.input_layernorm.weight",
            &[1.0; 8],
            8,
        )
        .unwrap(),
    )
    .unwrap()
    .into_cmd3_command()
    .unwrap();
    let output_state = command.resolve_output_state().unwrap();

    let output = FlashMoeExpertPhaseOutput::new(vec![0.0; 8], Some(vec![1.0; 8]));

    let output = output_state.validate_expert_phase_output(output).unwrap();
    assert_eq!(output.declared_cmd3_output(), Some(output_state.state()));
}

#[test]
fn scheduled_cmd3_output_rejects_mismatched_phase_output_without_fallback() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let scheduled = dummy_scheduled_experts(7, 2);
    let cmd3 = graph
        .build_cmd3_expert_phase(
            7,
            2,
            ScheduledCmd3InputSource::CpuNormedResidualUpload,
            ScheduledSharedExpertSource::ResidentProjections,
            ScheduledNextNormSource::CpuVisibleWeights,
        )
        .unwrap();
    let command = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::cpu_visible(
            "model.layers.8.input_layernorm.weight",
            &[1.0; 8],
            8,
        )
        .unwrap(),
    )
    .unwrap()
    .into_cmd3_command()
    .unwrap();
    let output_state = command.resolve_output_state().unwrap();

    let hidden_err = output_state
        .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(
            vec![0.0; 7],
            Some(vec![1.0; 8]),
        ))
        .unwrap_err();
    assert!(
        hidden_err
            .to_string()
            .contains("hidden length 7 does not match declared hidden length 8"),
        "{hidden_err:#}"
    );

    let missing_next_norm_err = output_state
        .validate_expert_phase_output(FlashMoeExpertPhaseOutput::new(vec![0.0; 8], None))
        .unwrap_err();
    assert!(
        missing_next_norm_err
            .to_string()
            .contains("did not produce declared next-normed state"),
        "{missing_next_norm_err:#}"
    );
}

#[test]
fn scheduled_q4_cmd3_payload_rejects_mismatched_shapes_without_fallback() {
    let err = ScheduledQ4ExpertPhaseMlpPayload::new(
        7,
        3,
        8,
        dummy_q4_payload(4, 8),
        dummy_q4_payload(5, 8),
        dummy_q4_payload(8, 4),
    )
    .unwrap_err();

    assert!(err.to_string().contains("mismatched gate/up rows"));
}

#[test]
fn scheduled_q4_cmd3_payload_requires_fixed_whole_slot_source() {
    let mut gate = dummy_q4_payload(4, 8);
    gate.source = None;
    let err = ScheduledQ4ExpertPhaseMlpPayload::new(
        7,
        3,
        8,
        gate,
        dummy_q4_payload(4, 8),
        dummy_q4_payload(8, 4),
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("not backed by a scheduler-owned whole-expert slot"),
        "{err:#}"
    );
}

#[test]
fn scheduled_q4_cmd3_payload_rejects_unresolved_scale_layout() {
    let mut gate = dummy_q4_payload(4, 8);
    gate.scale_bias_dtype = "F32";
    let err = ScheduledQ4ExpertPhaseMlpPayload::new(
        7,
        3,
        8,
        gate,
        dummy_q4_payload(4, 8),
        dummy_q4_payload(8, 4),
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("resolved implementation requires BF16"),
        "{err:#}"
    );
}

#[test]
fn scheduled_cmd3_submission_rejects_mismatched_sources_without_fallback() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let scheduled = dummy_scheduled_experts(7, 2);
    let cmd3 = graph
        .build_cmd3_expert_phase(
            7,
            2,
            ScheduledCmd3InputSource::MetalPostAttentionPrep,
            ScheduledSharedExpertSource::ResidentProjections,
            ScheduledNextNormSource::CpuVisibleWeights,
        )
        .unwrap();

    let input_err = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::cpu_visible("model.layers.8.input_layernorm.weight", &[1.0], 1)
            .unwrap(),
    )
    .unwrap_err();
    assert!(input_err.to_string().contains("does not match phase input"));

    let shared_err = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
        dummy_shared_expert(ScheduledSharedExpertSource::DenseCpuWeights),
        ScheduledNextNormWeights::cpu_visible("model.layers.8.input_layernorm.weight", &[1.0], 1)
            .unwrap(),
    )
    .unwrap_err();
    assert!(
        shared_err
            .to_string()
            .contains("does not match phase shared source")
    );

    let missing_shared_shape_err = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
        dummy_shared_expert_with_shape(ScheduledSharedExpertSource::ResidentProjections, None),
        ScheduledNextNormWeights::cpu_visible(
            "model.layers.8.input_layernorm.weight",
            &[1.0; 8],
            8,
        )
        .unwrap(),
    )
    .unwrap_err();
    assert!(
        missing_shared_shape_err
            .to_string()
            .contains("requires a declared shape")
    );

    let next_norm_err = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::none(),
    )
    .unwrap_err();
    assert!(
        next_norm_err
            .to_string()
            .contains("requires next-norm weights")
    );

    let shared_width_err = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
        dummy_shared_expert_with_shape(
            ScheduledSharedExpertSource::ResidentProjections,
            Some(ScheduledSharedExpertShape::new(4, 2, 2).unwrap()),
        ),
        ScheduledNextNormWeights::cpu_visible(
            "model.layers.8.input_layernorm.weight",
            &[1.0; 8],
            8,
        )
        .unwrap(),
    )
    .unwrap_err();
    assert!(
        shared_width_err
            .to_string()
            .contains("does not match input width")
    );

    let shared_shape_err = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
        dummy_shared_expert_with_shape(
            ScheduledSharedExpertSource::ResidentProjections,
            Some(ScheduledSharedExpertShape {
                width: 8,
                shared_experts: 2,
                intermediate: 2,
                total_intermediate: 5,
            }),
        ),
        ScheduledNextNormWeights::cpu_visible(
            "model.layers.8.input_layernorm.weight",
            &[1.0; 8],
            8,
        )
        .unwrap(),
    )
    .unwrap_err();
    assert!(
        shared_shape_err
            .to_string()
            .contains("not declared graph shape")
    );

    let next_norm_width_err = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input_with_width(ScheduledCmd3InputSource::MetalPostAttentionPrep, 8),
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::cpu_visible(
            "model.layers.8.input_layernorm.weight",
            &[1.0; 4],
            4,
        )
        .unwrap(),
    )
    .unwrap_err();
    assert!(
        next_norm_width_err
            .to_string()
            .contains("width 4 does not match input width 8")
    );

    let invalid_input_state_err = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        DummyCmd3InputState {
            source: ScheduledCmd3InputSource::MetalPostAttentionPrep,
            state: FlashMoeCmd3InputState::cpu_normed_residual(7, 8, 4),
        },
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::cpu_visible(
            "model.layers.8.input_layernorm.weight",
            &[1.0; 8],
            8,
        )
        .unwrap(),
    )
    .unwrap_err();
    assert!(
        invalid_input_state_err
            .to_string()
            .contains("input is not declared graph state")
    );
}

#[test]
fn scheduled_cmd3_descriptor_carries_shared_expert_shape() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let scheduled = dummy_scheduled_experts(7, 2);
    let shared_descriptor = ScheduledSharedExpertDescriptor::new(
        ScheduledSharedExpertSource::ResidentProjections,
        Some(ScheduledSharedExpertShape::new(8, 2, 2).unwrap()),
    )
    .unwrap();
    let cmd3 = graph
        .build_cmd3_expert_phase_with_shared_descriptor(
            7,
            2,
            ScheduledCmd3InputSource::MetalPostAttentionPrep,
            shared_descriptor,
            ScheduledNextNormSource::CpuVisibleWeights,
        )
        .unwrap();

    assert_eq!(
        cmd3.shared,
        ScheduledSharedExpertSource::ResidentProjections
    );
    assert_eq!(cmd3.shared_descriptor, Some(shared_descriptor));
    ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::cpu_visible(
            "model.layers.8.input_layernorm.weight",
            &[1.0; 8],
            8,
        )
        .unwrap(),
    )
    .unwrap();

    let mismatch = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &scheduled,
        dummy_cmd3_input(ScheduledCmd3InputSource::MetalPostAttentionPrep),
        dummy_shared_expert_with_shape(
            ScheduledSharedExpertSource::ResidentProjections,
            Some(ScheduledSharedExpertShape::new(8, 1, 4).unwrap()),
        ),
        ScheduledNextNormWeights::cpu_visible(
            "model.layers.8.input_layernorm.weight",
            &[1.0; 8],
            8,
        )
        .unwrap(),
    )
    .unwrap_err();
    assert!(
        mismatch.to_string().contains("shared descriptor"),
        "{mismatch:#}"
    );
}

#[test]
fn scheduled_cmd3_builder_derives_next_norm_source_from_weights() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let shared_descriptor = ScheduledSharedExpertDescriptor::new(
        ScheduledSharedExpertSource::ResidentProjections,
        Some(ScheduledSharedExpertShape::new(8, 1, 2).unwrap()),
    )
    .unwrap();

    let no_next_norm = graph
        .build_cmd3_expert_phase_from_descriptors(
            7,
            2,
            ScheduledCmd3InputSource::CpuNormedResidualUpload,
            shared_descriptor,
            ScheduledNextNormWeights::none(),
        )
        .unwrap();
    assert_eq!(no_next_norm.next_norm, ScheduledNextNormSource::None);

    let cpu_next_norm = graph
        .build_cmd3_expert_phase_from_descriptors(
            7,
            2,
            ScheduledCmd3InputSource::CpuNormedResidualUpload,
            shared_descriptor,
            ScheduledNextNormWeights::cpu_visible(
                "model.layers.8.input_layernorm.weight",
                &[1.0; 8],
                8,
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        cpu_next_norm.next_norm,
        ScheduledNextNormSource::CpuVisibleWeights
    );
}

#[test]
fn scheduled_graph_builds_cmd3_submission_and_rejects_stale_stage() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let scheduled = dummy_scheduled_experts(7, 2);
    let cmd3 = graph
        .build_cmd3_expert_phase(
            7,
            2,
            ScheduledCmd3InputSource::CpuNormedResidualUpload,
            ScheduledSharedExpertSource::ResidentProjections,
            ScheduledNextNormSource::None,
        )
        .unwrap();

    graph
        .build_cmd3_submission(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::none(),
        )
        .unwrap();

    let mut stale_graph = graph.clone();
    stale_graph
        .stages
        .iter_mut()
        .find(|stage| stage.stage == FlashMoeGraphStage::Cmd3ExpertAndSharedCombine)
        .unwrap()
        .implementation = FlashMoeStageImplementation::QwenTextInput;

    let err = stale_graph
        .build_cmd3_submission(
            19,
            cmd3,
            &scheduled,
            dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
            dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
            ScheduledNextNormWeights::none(),
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("does not match scheduled graph CMD3 stage"),
        "{err:#}"
    );
}

#[test]
fn scheduled_cmd3_submission_rejects_mismatched_or_partial_expert_slots() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd3 = graph
        .build_cmd3_expert_phase(
            7,
            1,
            ScheduledCmd3InputSource::CpuNormedResidualUpload,
            ScheduledSharedExpertSource::ResidentProjections,
            ScheduledNextNormSource::None,
        )
        .unwrap();
    let routes = ScheduledExpertRoutes::from_routes(
        7,
        vec![ExpertRoute {
            expert: 0,
            score: 1.0,
        }],
        1.0,
    )
    .unwrap();
    let wrong_expert =
        ScheduledExpertSet::from_parts(routes.clone(), vec![DummyCmd3Expert::whole_slot(7, 9)])
            .unwrap();

    let err = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &wrong_expert,
        dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::none(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("does not match routed layer"));

    let partial_slot = DummyCmd3Expert::whole_slot(7, 0).with_descriptor(ExpertSlotDescriptor {
        layer: 7,
        expert: 0,
        slot_offset: 0,
        slot_capacity: 128,
        payload_len: 64,
    });
    let partial_expert = ScheduledExpertSet::from_parts(routes, vec![partial_slot]).unwrap();
    let err = ScheduledCmd3Submission::new(
        19,
        cmd3,
        &partial_expert,
        dummy_cmd3_input(ScheduledCmd3InputSource::CpuNormedResidualUpload),
        dummy_shared_expert(ScheduledSharedExpertSource::ResidentProjections),
        ScheduledNextNormWeights::none(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("must be a whole-expert slot"));
}

#[test]
fn fixed_q4_graph_rejects_dense_shared_weights_for_each_text_family() {
    for layout in [qwen35_layout(), qwen3_moe_layout()] {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

        let err = graph
            .build_cmd3_expert_phase(
                7,
                2,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::DenseCpuWeights,
                ScheduledNextNormSource::None,
            )
            .unwrap_err();

        assert_eq!(err.family, graph.family());
        assert_eq!(err.stage, FlashMoeGraphStage::Cmd3ExpertAndSharedCombine);
        assert!(
            err.to_string()
                .contains("requires resident shared projections"),
            "{err:#}"
        );
        assert!(
            err.to_string()
                .contains("not a declared graph-stage implementation"),
            "{err:#}"
        );
    }
}

#[test]
fn scheduled_graph_rejects_non_metal_cmd3_builder() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let mut graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let cmd3 = graph
        .stages
        .iter_mut()
        .find(|stage| stage.stage == FlashMoeGraphStage::Cmd3ExpertAndSharedCombine)
        .unwrap();
    cmd3.placement = FlashMoeStagePlacement::CpuDeclared;

    let err = graph
        .build_cmd3_expert_phase(
            0,
            4,
            ScheduledCmd3InputSource::CpuNormedResidualUpload,
            ScheduledSharedExpertSource::DenseCpuWeights,
            ScheduledNextNormSource::None,
        )
        .unwrap_err();

    assert_eq!(err.family, graph.family());
    assert_eq!(err.stage, FlashMoeGraphStage::Cmd3ExpertAndSharedCombine);
    assert!(
        err.to_string()
            .contains("CMD3 expert/shared combine must be implemented"),
        "{err:#}"
    );
}

#[test]
fn scheduled_expert_routes_renormalize_and_scale_full_softmax_probabilities() {
    let scheduled =
        ScheduledExpertRoutes::from_scores(12, &[(7, 0.2), (3, 0.6), (9, 0.1)], 0.25).unwrap();
    let expected = [0.25 * 2.0 / 9.0, 0.25 * 6.0 / 9.0, 0.25 * 1.0 / 9.0];

    assert_eq!(scheduled.layer, 12);
    assert_eq!(
        scheduled.routes,
        vec![
            ExpertRoute {
                expert: 7,
                score: 0.2,
            },
            ExpertRoute {
                expert: 3,
                score: 0.6,
            },
            ExpertRoute {
                expert: 9,
                score: 0.1,
            },
        ]
    );
    assert_eq!(scheduled.weights.len(), expected.len());
    for (actual, expected) in scheduled.weights.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn deepseek_routes_apply_the_fixed_selected_sum_floor_without_affecting_standard_routes() {
    let routes = ExpertRoute::from_scores(&[(7, 1.0e-6), (3, 2.0e-6)]).unwrap();
    let deepseek = ScheduledExpertRoutes::from_routes_with_policy(
        12,
        routes.clone(),
        QwenMoeRoutingWeightNormalization::DeepSeekRenormalizeSelectedWithFloor,
        1.5,
    )
    .unwrap();
    let standard = ScheduledExpertRoutes::from_routes_with_policy(
        12,
        routes,
        QwenMoeRoutingWeightNormalization::RenormalizeSelected,
        1.5,
    )
    .unwrap();

    assert!((deepseek.weights[0] - 1.5e-6 / 6.103515625e-5).abs() < 1.0e-7);
    assert!((deepseek.weights[1] - 3.0e-6 / 6.103515625e-5).abs() < 1.0e-7);
    assert!((standard.weights[0] - 0.5).abs() < 1.0e-6);
    assert!((standard.weights[1] - 1.0).abs() < 1.0e-6);
}

#[test]
fn scheduled_expert_routes_reject_unimplemented_full_softmax_weights() {
    let error = ScheduledExpertRoutes::from_routes_with_policy(
        12,
        ExpertRoute::from_scores(&[(7, 1.0), (3, 2.0)]).unwrap(),
        QwenMoeRoutingWeightNormalization::PreserveFullSoftmax,
        1.0,
    )
    .unwrap_err();

    assert!(error.to_string().contains("full expert softmax"));
}

#[test]
fn scheduled_expert_routes_resolve_from_routing_command() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(12, 10, 3, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap();
    let command = routing.command_from_routes(vec![(7, 0.2), (3, 0.6), (9, 0.1)]);

    let scheduled = ScheduledExpertRoutes::from_routing_command(&command, 0.25).unwrap();

    assert_eq!(scheduled.layer, 12);
    assert_eq!(scheduled.expert_ids().collect::<Vec<_>>(), vec![7, 3, 9]);
    let expected = [0.25 * 2.0 / 9.0, 0.25 * 6.0 / 9.0, 0.25 * 1.0 / 9.0];
    for (actual, expected) in scheduled.weights.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn active_expert_scheduler_issues_routed_read_set_from_command() {
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(12, 10, 2, ScheduledRoutingCandidateSource::CpuRouterScores)
        .unwrap();
    let command = routing.command_from_routes(vec![(7, 0.25), (3, 0.75)]);
    let mut scheduler = ActiveExpertReadScheduler::new(0.25);

    let issued = scheduler.issue_routed_reads(&command).unwrap();
    assert_eq!(issued.layer(), 12);
    assert_eq!(issued.len(), 2);
    assert_eq!(issued.issues()[0].id, 0);
    assert_eq!(issued.issues()[0].key.expert, 7);
    assert_eq!(issued.issues()[1].id, 1);
    assert_eq!(issued.issues()[1].key.expert, 3);
    assert!(!issued.issues()[0].warm);
    assert!(!issued.issues()[1].warm);
    let routes = issued.into_routes();
    let expected = [0.25 * 0.25, 0.25 * 0.75];
    for (actual, expected) in routes.weights.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }

    let repeated = scheduler.issue_routed_reads(&command).unwrap();
    assert_eq!(repeated.issues()[0].id, 2);
    assert_eq!(repeated.issues()[1].id, 3);
    assert!(repeated.issues()[0].warm);
    assert!(repeated.issues()[1].warm);
    assert_eq!(scheduler.snapshot().issued_reads, 4);
}

#[test]
fn scheduled_read_coordinator_streams_routed_slots_in_order() {
    let (_temp, store) = pbq4_import_store(&[1, 3, 7]);
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let routing = graph
        .build_routing_topk(
            0,
            graph.experts_per_layer(),
            3,
            ScheduledRoutingCandidateSource::CpuRouterScores,
        )
        .unwrap();
    let command = routing.command_from_routes(vec![(7, 0.6), (1, 0.3), (3, 0.1)]);
    let mut coordinator = ScheduledExpertReadCoordinator::new_with_routed_expert_scale(store, 0.9);

    let pending = coordinator.issue_routing_command(&command).unwrap();
    let scheduled = coordinator.finish_routes(pending).unwrap();

    assert_eq!(coordinator.worker_count(), 3);
    assert_eq!(
        scheduled
            .experts
            .iter()
            .map(|expert| expert.expert())
            .collect::<Vec<_>>(),
        vec![7, 1, 3]
    );
    let expected = [0.54, 0.27, 0.09];
    for (actual, expected) in scheduled.weights.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-6);
    }
    let first = coordinator.snapshot();
    assert_eq!(first.issued_reads, 3);
    assert_eq!(first.positioned_reads, 3);
    assert_eq!(first.read_failures, 0);
    assert_eq!(first.warm_reads, 0);

    let pending = coordinator.issue_experts(0, &[3]).unwrap();
    let repeated = coordinator.finish(pending).unwrap();
    assert_eq!(repeated[0].expert(), 3);
    let second = coordinator.snapshot();
    assert_eq!(second.issued_reads, 4);
    assert_eq!(second.positioned_reads, 4);
    assert_eq!(second.warm_reads, 1);
    assert!(second.warm_bytes_read > 0);
}

#[test]
fn execution_scheduler_streams_one_sorted_unique_batch_working_set() {
    let temp = tempfile::tempdir().unwrap();
    write_identity_fixed_q4_layer(temp.path(), 0, 512);
    let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
    let store = ExpertSlotStore::open_with_fixed_q4(temp.path().to_path_buf(), spec).unwrap();
    let mut layout = qwen35_layout();
    layout.layers = 1;
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let mut scheduler = FlashMoeExecutionScheduler::new(graph, store).unwrap();

    let stride = tiny_fixed_q4_layout().expert_bytes;
    let mut prepared_destination = vec![0; 512 * stride];
    let pending = unsafe {
        scheduler
            .issue_expert_layer_prepare_into(0, &mut prepared_destination)
            .unwrap()
    };
    let prepared = scheduler.finish_expert_layer_prepare(pending).unwrap();
    assert_eq!(prepared.bytes_read, (512 * stride) as u64);
    let expected = identity_fixed_q4_slot_bytes();
    assert!(
        prepared_destination
            .chunks_exact(stride)
            .all(|slot| slot == expected.as_slice())
    );

    let mut destination = vec![0; 512 * stride];
    let summary = scheduler
        .read_unique_experts_into(0, &[1, 2, 3, 7], &mut destination, stride)
        .unwrap();

    assert_eq!(summary.positioned_runs, 2);
    assert_eq!(summary.bytes_read, (4 * stride) as u64);
    for expert in [1, 2, 3, 7] {
        assert_eq!(
            &destination[expert * stride..(expert + 1) * stride],
            expected.as_slice()
        );
    }
    assert!(destination[..stride].iter().all(|byte| *byte == 0));
    let snapshot = scheduler.snapshot();
    assert_eq!(snapshot.issued_reads, 4);
    assert_eq!(snapshot.positioned_reads, 4);

    let duplicate = scheduler
        .read_unique_experts_into(0, &[1, 1], &mut destination, stride)
        .unwrap_err();
    assert!(duplicate.to_string().contains("sorted and unique"));
    let descending = scheduler
        .read_unique_experts_into(0, &[3, 1], &mut destination, stride)
        .unwrap_err();
    assert!(descending.to_string().contains("sorted and unique"));
}

#[test]
fn direct_batch_read_records_every_issued_failure() {
    let temp = tempfile::tempdir().unwrap();
    write_identity_fixed_q4_layer(temp.path(), 0, 512);
    let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
    let store = ExpertSlotStore::open_with_fixed_q4(temp.path().to_path_buf(), spec).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(expert_layer_path(temp.path(), 0))
        .unwrap()
        .set_len(0)
        .unwrap();
    let mut layout = qwen35_layout();
    layout.layers = 1;
    let capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let mut scheduler = FlashMoeExecutionScheduler::new(graph, store).unwrap();
    let stride = tiny_fixed_q4_layout().expert_bytes;
    let mut destination = vec![0; 512 * stride];

    let error = scheduler
        .read_unique_experts_into(0, &[1, 2, 3, 7], &mut destination, stride)
        .unwrap_err();

    assert!(
        error.to_string().contains("failed direct batch expert run"),
        "{error:#}"
    );
    let snapshot = scheduler.snapshot();
    assert_eq!(snapshot.issued_reads, 4);
    assert_eq!(snapshot.positioned_reads, 4);
    assert_eq!(snapshot.read_failures, 4);
    assert_eq!(snapshot.bytes_read, 0);
}

#[test]
fn layer_major_streaming_reads_each_unique_expert_once_per_layer_request() {
    let temp = tempfile::tempdir().unwrap();
    write_identity_fixed_q4_layer(temp.path(), 0, 512);
    let spec = FixedQ4ExpertSlotSpec::new(tiny_fixed_q4_layout(), 2, 2).unwrap();
    let store = ExpertSlotStore::open_with_fixed_q4(temp.path().to_path_buf(), spec).unwrap();
    let mut layout = qwen35_layout();
    layout.layers = 1;
    let mut capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
    capabilities.active_experts = 2;
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
    let mut scheduler = FlashMoeExecutionScheduler::new(graph, store).unwrap();
    let routes = vec![vec![(3, 0.8), (1, 0.2)], vec![(1, 0.7), (2, 0.3)]];

    let scheduled = scheduler.resolve_layer_major_experts(0, &routes).unwrap();

    assert_eq!(scheduled.layer(), 0);
    assert_eq!(scheduled.rows(), 2);
    assert_eq!(scheduled.active_experts(), 2);
    assert_eq!(scheduled.route_slots(), &[2, 0, 0, 1]);
    assert_eq!(scheduled.weights(), &[0.8, 0.2, 0.7, 0.3]);
    assert_eq!(
        scheduled
            .experts()
            .iter()
            .map(|slot| slot.expert())
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    let first = scheduler.snapshot();
    assert_eq!(first.issued_reads, 3);
    assert_eq!(first.positioned_reads, 3);
    assert_eq!(
        first.bytes_read,
        (3 * tiny_fixed_q4_layout().expert_bytes) as u64
    );

    let repeated = scheduler.resolve_layer_major_experts(0, &routes).unwrap();
    assert_eq!(repeated.route_slots(), scheduled.route_slots());
    let second = scheduler.snapshot();
    assert_eq!(second.issued_reads, 6);
    assert_eq!(second.positioned_reads, 6);
    assert_eq!(
        second.bytes_read,
        (6 * tiny_fixed_q4_layout().expert_bytes) as u64
    );
}

#[test]
fn execution_scheduler_resident_graph_maps_complete_table_without_positioned_reads() {
    let temp = tempfile::tempdir().unwrap();
    write_identity_fixed_dense_layer(temp.path(), 0, 1, DenseExpertDtype::Bf16);
    let spec = FixedDenseExpertSlotSpec::new(DenseExpertDtype::Bf16, 2, 2).unwrap();
    let store = ExpertSlotStore::open_with_fixed_dense(temp.path().to_path_buf(), spec).unwrap();

    let mut layout = qwen35_layout();
    layout.layers = 1;
    let mut capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
    capabilities.expert_storage = ExpertStoreExecutionDescriptor {
        layout: ExpertStorageLayout::FixedBf16,
        slot_spec: ExpertSlotSpec::FixedDense(spec),
        layers: 1,
        first_expert_layer: 0,
        experts_per_layer: 1,
    };
    capabilities.experts_per_layer = 1;
    capabilities.active_experts = 1;
    let stage = capabilities
        .stages
        .iter_mut()
        .find(|stage| stage.stage == FlashMoeGraphStage::ActiveExpertReads)
        .unwrap();
    stage.placement = FlashMoeStagePlacement::SchedulerMemory;
    stage.implementation = FlashMoeStageImplementation::ResidentMappedWholeExpertSlots;
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

    let mut bound = 0usize;
    let mut scheduler =
        FlashMoeExecutionScheduler::new_with_resident_binding(graph, store, |bytes| {
            bound += 1;
            assert_eq!(bytes.len(), spec.expert_bytes);
            Ok(())
        })
        .unwrap();
    assert_eq!(bound, 1);

    let routing = scheduler
        .graph
        .build_routing_topk(
            0,
            1,
            1,
            ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
        )
        .unwrap()
        .command_from_preselected(&[(0, 1.0)])
        .unwrap();
    let first = scheduler
        .expert_access
        .issue_routing_command(&routing)
        .unwrap();
    let first = scheduler.expert_access.finish_routes(first).unwrap();
    let second = scheduler
        .expert_access
        .issue_routing_command(&routing)
        .unwrap();
    let second = scheduler.expert_access.finish_routes(second).unwrap();

    assert_eq!(first.len(), 1);
    assert!(Arc::ptr_eq(&first.experts[0], &second.experts[0]));
    assert_eq!(
        first.experts[0].raw.read_path,
        ExpertReadPath::ResidentMapped
    );
    assert_eq!(scheduler.snapshot().positioned_reads, 0);
    assert_eq!(scheduler.snapshot().bytes_read, 0);
}

#[test]
fn layer_major_resident_graph_reuses_mapped_union_without_scheduler_reads() {
    let temp = tempfile::tempdir().unwrap();
    write_identity_fixed_dense_layer(temp.path(), 0, 4, DenseExpertDtype::Bf16);
    let spec = FixedDenseExpertSlotSpec::new(DenseExpertDtype::Bf16, 2, 2).unwrap();
    let store = ExpertSlotStore::open_with_fixed_dense(temp.path().to_path_buf(), spec).unwrap();

    let mut layout = qwen35_layout();
    layout.layers = 1;
    let mut capabilities = FlashMoeCapabilityPlan::for_model_layout(&layout).unwrap();
    capabilities.expert_storage = ExpertStoreExecutionDescriptor {
        layout: ExpertStorageLayout::FixedBf16,
        slot_spec: ExpertSlotSpec::FixedDense(spec),
        layers: 1,
        first_expert_layer: 0,
        experts_per_layer: 4,
    };
    capabilities.experts_per_layer = 4;
    capabilities.active_experts = 2;
    let stage = capabilities
        .stages
        .iter_mut()
        .find(|stage| stage.stage == FlashMoeGraphStage::ActiveExpertReads)
        .unwrap();
    stage.placement = FlashMoeStagePlacement::SchedulerMemory;
    stage.implementation = FlashMoeStageImplementation::ResidentMappedWholeExpertSlots;
    let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

    let mut bound = 0usize;
    let mut scheduler = FlashMoeExecutionScheduler::new_with_resident_binding(graph, store, |_| {
        bound += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(bound, 4);
    let routes = vec![vec![(3, 0.8), (1, 0.2)], vec![(1, 0.7), (2, 0.3)]];

    let first = scheduler.resolve_layer_major_experts(0, &routes).unwrap();
    let second = scheduler.resolve_layer_major_experts(0, &routes).unwrap();

    assert_eq!(first.route_slots(), &[2, 0, 0, 1]);
    assert_eq!(first.weights(), &[0.8, 0.2, 0.7, 0.3]);
    assert_eq!(
        first
            .experts()
            .iter()
            .map(|slot| slot.expert())
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert!(
        first
            .experts()
            .iter()
            .zip(second.experts().iter())
            .all(|(left, right)| Arc::ptr_eq(left, right))
    );
    let snapshot = scheduler.snapshot();
    assert_eq!(snapshot.issued_reads, 0);
    assert_eq!(snapshot.positioned_reads, 0);
    assert_eq!(snapshot.bytes_read, 0);
}

#[test]
fn scheduled_read_coordinator_records_positioned_read_failure() {
    let (temp, store) = pbq4_import_store(&[2]);
    fs::OpenOptions::new()
        .write(true)
        .open(expert_layer_path(temp.path(), 0))
        .unwrap()
        .set_len(0)
        .unwrap();
    let mut coordinator = ScheduledExpertReadCoordinator::new(store);

    let pending = coordinator.issue_experts(0, &[2]).unwrap();
    let error = coordinator.finish(pending).unwrap_err();

    assert!(
        error.to_string().contains("failed to read expert 2"),
        "{error:#}"
    );
    let snapshot = coordinator.snapshot();
    assert_eq!(snapshot.issued_reads, 1);
    assert_eq!(snapshot.positioned_reads, 1);
    assert_eq!(snapshot.read_failures, 1);
}

#[test]
fn scheduled_expert_routes_reject_non_finite_scores() {
    let err = ScheduledExpertRoutes::from_scores(0, &[(2, f32::NAN)], 1.0).unwrap_err();

    assert!(
        err.to_string()
            .contains("expert route score for expert 2 is not finite"),
        "{err:#}"
    );
}

#[test]
fn scheduled_expert_batch_validates_route_weight_and_expert_counts() {
    let routes = ScheduledExpertRoutes::from_scores(3, &[(8, 0.6), (4, 0.4)], 1.0).unwrap();
    let batch = ScheduledExpertBatch::from_parts(routes, vec!["expert-8", "expert-4"]).unwrap();

    assert_eq!(batch.layer, 3);
    assert_eq!(batch.len(), 2);
    assert!(!batch.is_empty());
    assert_eq!(batch.experts.as_ref(), ["expert-8", "expert-4"]);

    let routes = ScheduledExpertRoutes::from_scores(3, &[(8, 0.6), (4, 0.4)], 1.0).unwrap();
    let err = ScheduledExpertBatch::from_parts(routes, vec!["expert-8"]).unwrap_err();
    assert!(
        err.to_string()
            .contains("scheduled expert batch has 1 experts for 2 routes"),
        "{err:#}"
    );
}

#[test]
fn scheduled_expert_slot_resolves_cmd3_payload_without_legacy_adapter() {
    let slot = ScheduledExpertSlot::from_raw(raw_fixed_q4_read(3, 8));

    assert_eq!(slot.scheduled_expert_layer(), 3);
    assert_eq!(slot.scheduled_expert_id(), 8);
    assert_eq!(
        slot.scheduled_expert_slot_descriptor(),
        ExpertSlotDescriptor {
            layer: 3,
            expert: 8,
            slot_offset: 512,
            slot_capacity: tiny_fixed_q4_layout().expert_bytes,
            payload_len: tiny_fixed_q4_layout().expert_bytes,
        }
    );

    let payload = slot.scheduled_cmd3_expert_phase_payload(2).unwrap();
    let q4 = payload.q4();
    assert_eq!(q4.gate.rows, 2);
    assert_eq!(q4.gate.cols, 2);
    assert_eq!(q4.up.rows, 2);
    assert_eq!(q4.up.cols, 2);
    assert_eq!(q4.down.rows, 2);
    assert_eq!(q4.down.cols, 2);
    let source = q4
        .gate
        .source
        .expect("fixed slot should expose source offsets");
    assert_eq!(source.packed_offset, 0);
    assert_eq!(source.scale_offset, 8);
    assert_eq!(source.bias_offset, 12);
}

#[test]
fn scheduled_expert_slot_resolves_typed_dense_payload_from_same_lease() {
    for dtype in [DenseExpertDtype::Bf16, DenseExpertDtype::F16] {
        let spec = FixedDenseExpertSlotSpec::new(dtype, 2, 2).unwrap();
        let slot = ScheduledExpertSlot::from_raw(raw_fixed_dense_read(3, 8, dtype));
        let payload = slot.scheduled_cmd3_expert_phase_payload(2).unwrap();
        let ScheduledExpertPhaseMlpPayload::Dense(dense) = payload else {
            panic!("fixed dense slot resolved a Q4 payload");
        };
        assert_eq!(dense.gate.dtype, dtype);
        assert_eq!(dense.gate.rows, 2);
        assert_eq!(dense.gate.cols, 2);
        assert_eq!(
            dense.up.source.bytes.as_ptr(),
            dense.gate.source.bytes.as_ptr()
        );
        assert_eq!(
            dense.down.source.bytes.as_ptr(),
            dense.gate.source.bytes.as_ptr()
        );
        assert_eq!(dense.gate.source.byte_offset, 0);
        assert_eq!(dense.up.source.byte_offset, spec.up.offset);
        assert_eq!(dense.down.source.byte_offset, spec.down.offset);
    }
}

#[test]
fn scheduled_expert_slot_rejects_pbq4_component_payload_for_cmd3() {
    let slot = ScheduledExpertSlot::from_raw(raw_pbq4_read(3, 8, vec![1, 2, 3]));

    let err = slot.scheduled_cmd3_expert_phase_payload(2).unwrap_err();

    assert!(
        err.to_string().contains("PBQ4/component import data"),
        "{err:#}"
    );
}

#[test]
fn pending_scheduled_expert_set_owns_read_receivers_and_routes() {
    let (tx, rx) = mpsc::channel();
    let read = PendingScheduledRead::new(77, rx);
    assert_eq!(read.id(), 77);
    let scheduled_routes = ScheduledExpertRoutes::from_routes(
        5,
        vec![ExpertRoute {
            expert: 9,
            score: 1.25,
        }],
        1.0,
    )
    .unwrap();
    let pending = PendingScheduledExpertSet::new(scheduled_routes, vec![read]);

    tx.send("expert-9").unwrap();
    let (routes, reads) = pending.into_parts();

    assert_eq!(routes.layer, 5);
    assert_eq!(
        routes.routes,
        vec![ExpertRoute {
            expert: 9,
            score: 1.25
        }]
    );
    assert_eq!(reads.len(), 1);
    assert_eq!(
        reads.into_iter().next().unwrap().recv().unwrap(),
        "expert-9"
    );
}

#[test]
fn active_expert_scheduler_issues_ids_and_marks_repeated_reads_warm() {
    let mut scheduler = ActiveExpertReadScheduler::new(1.0);

    let cold = scheduler.issue_read(4, 7);
    let warm = scheduler.issue_read(4, 7);

    assert_eq!(cold.id, 0);
    assert_eq!(
        cold.key,
        ExpertReadKey {
            layer: 4,
            expert: 7
        }
    );
    assert!(!cold.warm);
    assert_eq!(warm.id, 1);
    assert_eq!(warm.key, cold.key);
    assert!(warm.warm);
    assert_eq!(scheduler.snapshot().issued_reads, 2);
}

#[test]
fn active_expert_scheduler_finishes_responses_and_records_failures() {
    let mut scheduler = ActiveExpertReadScheduler::new(0.5);
    let first = scheduler.issue_read(2, 9);
    let value = scheduler
        .finish_read(
            first.id,
            ScheduledExpertReadResponse {
                id: first.id,
                queue_latency: Duration::from_millis(2),
                read_path: ExpertReadPath::PositionedRead,
                read_latency: Duration::from_millis(5),
                bytes_read: 128,
                warm: first.warm,
                result: Ok("expert-9"),
            },
        )
        .unwrap();

    assert_eq!(value, "expert-9");
    let snapshot = scheduler.snapshot();
    assert_eq!(snapshot.positioned_reads, 1);
    assert_eq!(snapshot.bytes_read, 128);
    assert_eq!(snapshot.read_failures, 0);

    let second = scheduler.issue_read(2, 10);
    let err = scheduler
        .finish_read(
            second.id,
            ScheduledExpertReadResponse {
                id: second.id + 1,
                queue_latency: Duration::ZERO,
                read_path: ExpertReadPath::PositionedRead,
                read_latency: Duration::ZERO,
                bytes_read: 0,
                warm: false,
                result: Ok("wrong-id"),
            },
        )
        .unwrap_err();

    assert!(
        err.to_string()
            .contains("returned response 2 for pending read 1"),
        "{err:#}"
    );
    assert_eq!(scheduler.snapshot().read_failures, 1);
}

#[test]
fn active_expert_scheduler_finishes_raw_reads_as_scheduled_slots() {
    let mut scheduler = ActiveExpertReadScheduler::new(1.0);
    let issue = scheduler.issue_read(3, 8);

    let slot = scheduler
        .finish_slot_read(
            issue.id,
            ExpertRawReadResponse {
                id: issue.id,
                queue_latency: Duration::from_millis(1),
                read_path: ExpertReadPath::PositionedRead,
                read_latency: Duration::from_millis(7),
                bytes_read: 3,
                warm: issue.warm,
                result: Ok(raw_pbq4_read(3, 8, vec![1, 2, 3])),
            },
        )
        .unwrap();

    assert_eq!(slot.layer(), 3);
    assert_eq!(slot.expert(), 8);
    assert_eq!(
        slot.descriptor(),
        ExpertSlotDescriptor {
            layer: 3,
            expert: 8,
            slot_offset: 1024,
            slot_capacity: 3,
            payload_len: 3,
        }
    );
    assert!(
        slot.scheduled_cmd3_expert_phase_payload(2)
            .unwrap_err()
            .to_string()
            .contains("PBQ4/component import data instead of a resolved whole-expert payload")
    );
    let snapshot = scheduler.snapshot();
    assert_eq!(snapshot.issued_reads, 1);
    assert_eq!(snapshot.positioned_reads, 1);
    assert_eq!(snapshot.bytes_read, 3);
    assert_eq!(snapshot.read_failures, 0);
}

#[test]
fn expert_scheduler_metrics_snapshot_reports_saturating_delta() {
    let mut metrics = ExpertSchedulerMetrics::default();
    metrics.record_issued_read();
    metrics.record_positioned_read();
    metrics.record_queue_latency(Duration::from_millis(7));
    metrics.record_read_latency(Duration::from_millis(11));
    metrics.record_bytes_read(128);
    let before = metrics.snapshot();

    metrics.record_issued_read();
    metrics.record_read_failure();
    metrics.record_queue_latency(Duration::from_millis(3));
    metrics.record_read_latency(Duration::from_millis(5));
    metrics.record_bytes_read(32);
    metrics.record_warm_read(Duration::from_millis(5), 32);

    let delta = metrics.snapshot().saturating_delta(before);
    assert_eq!(delta.issued_reads, 1);
    assert_eq!(delta.positioned_reads, 0);
    assert_eq!(delta.read_failures, 1);
    assert_eq!(delta.total_queue_latency, Duration::from_millis(3));
    assert_eq!(delta.total_read_latency, Duration::from_millis(5));
    assert_eq!(delta.bytes_read, 32);
    assert_eq!(delta.warm_reads, 1);
    assert_eq!(delta.warm_bytes_read, 32);
}
