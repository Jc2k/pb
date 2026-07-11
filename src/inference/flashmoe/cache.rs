use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use super::experts::*;
use super::metal::METAL_SHADERS;
use super::model_family::{QwenModelConfig, QwenMoeModelLayout};
use super::planning::*;
use super::safetensors::*;
use super::types::*;
use super::weights::*;

#[cfg(test)]
pub(super) const DENSE_Q4_FORMAT: &str = "dense-q4-affine-mse-v3";

#[cfg(test)]
pub(super) fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

pub fn expected_hf_files() -> Vec<OsString> {
    [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    .into_iter()
    .map(OsString::from)
    .collect()
}

/// The minimum set of HuggingFace snapshot files required for a Qwen3-VL
/// (vision-language) FlashMoe model.  The ViT tensors are embedded in the
/// same shards as the text tensors and are split out during caching.
pub fn expected_vl_hf_files() -> Vec<OsString> {
    expected_hf_files()
}

pub fn build_cache_from_hf_snapshot(model: &str, snapshot_dir: &Path) -> Result<FlashMoePlan> {
    let plan = plan_unchecked(model, snapshot_dir.parent().unwrap_or(snapshot_dir));
    fs::create_dir_all(&plan.runtime_dir)
        .with_context(|| format!("failed to create {}", plan.runtime_dir.display()))?;
    fs::create_dir_all(&plan.experts_dir)
        .with_context(|| format!("failed to create {}", plan.experts_dir.display()))?;

    let config_json = snapshot_dir.join("config.json");
    let config = if config_json.is_file() {
        let config = QwenModelConfig::from_file(&config_json)?;
        fs::copy(&config_json, &plan.model_config).with_context(|| {
            format!(
                "failed to copy {} to {}",
                config_json.display(),
                plan.model_config.display()
            )
        })?;
        let routing_policy = plan.routing_policy.resolve(&plan.model, &config)?;
        tracing::debug!(
            layers = config.num_hidden_layers,
            hidden_size = config.hidden_size,
            attention_heads = config.num_attention_heads,
            kv_heads = config.kv_heads(),
            experts = config.experts(),
            active_experts = routing_policy.active_experts,
            active_experts_source = ?routing_policy.source,
            vocab_size = config.vocab_size,
            "validated Qwen Flash-MoE model config"
        );
        Some(config)
    } else {
        None
    };

    prepare_tokenizer_artifacts(snapshot_dir, &plan)?;

    let index_json = snapshot_dir.join("model.safetensors.index.json");
    let (manifest, visual_tensor_refs) = if index_json.is_file() {
        build_manifest(model, snapshot_dir, &index_json)?
    } else {
        (
            FlashMoeManifest {
                model: canonical_model(model),
                cache_version: cache_version_for_model(model).to_string(),
                dense_shards: Vec::new(),
                expert_tensors: Vec::new(),
                dense_tensors: Vec::new(),
            },
            Vec::new(),
        )
    };
    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).context("failed to encode Flash-MoE manifest")?;
    fs::write(&plan.tensor_manifest, manifest_bytes).with_context(|| {
        format!(
            "failed to write Flash-MoE tensor manifest {}",
            plan.tensor_manifest.display()
        )
    })?;

    write_dense_tensor_store(
        snapshot_dir,
        &plan.non_expert_weights,
        &manifest.dense_tensors,
    )?;
    pack_expert_tensors(
        snapshot_dir,
        &plan,
        &manifest.expert_tensors,
        config.as_ref(),
    )?;

    // For VL models, build and write the vision weights store.
    if let (Some(vision_weights), Some(vision_manifest)) =
        (plan.vision_weights.as_ref(), plan.vision_manifest.as_ref())
    {
        if !visual_tensor_refs.is_empty() {
            write_dense_tensor_store(snapshot_dir, vision_weights, &visual_tensor_refs)?;
            let vision_manifest_data = FlashMoeManifest {
                model: canonical_model(model),
                cache_version: cache_version_for_model(model).to_string(),
                dense_shards: Vec::new(),
                expert_tensors: Vec::new(),
                dense_tensors: visual_tensor_refs,
            };
            let vision_manifest_bytes = serde_json::to_vec_pretty(&vision_manifest_data)
                .context("failed to encode vision weights manifest")?;
            fs::write(vision_manifest, vision_manifest_bytes).with_context(|| {
                format!(
                    "failed to write vision weights manifest {}",
                    vision_manifest.display()
                )
            })?;
        }
        // Write vision_config.json (the nested vision_config object from config.json).
        if let (Some(vc), Some(vc_path)) = (
            config.as_ref().and_then(|c| c.vision_config.as_ref()),
            plan.vision_config_path.as_ref(),
        ) {
            let vc_bytes =
                serde_json::to_vec_pretty(vc).context("failed to encode vision config")?;
            fs::write(vc_path, vc_bytes)
                .with_context(|| format!("failed to write vision config {}", vc_path.display()))?;
        }
    }

    fs::write(plan.runtime_dir.join("kernels.metal"), METAL_SHADERS).with_context(|| {
        format!(
            "failed to write Metal kernels to {}",
            plan.runtime_dir.display()
        )
    })?;
    fs::write(plan.runtime_dir.join("README.txt"), plan.describe())
        .with_context(|| "failed to write Flash-MoE cache README".to_string())?;
    Ok(plan)
}

pub(super) fn prepare_tokenizer_artifacts(snapshot_dir: &Path, plan: &FlashMoePlan) -> Result<()> {
    let tokenizer_json = snapshot_dir.join("tokenizer.json");
    let tokenizer_config_json = snapshot_dir.join("tokenizer_config.json");
    if !tokenizer_json.is_file() {
        bail!(
            "Flash-MoE requires tokenizer.json from the active model directory; missing {}",
            tokenizer_json.display()
        );
    }
    fs::create_dir_all(&plan.model_cache_dir)
        .with_context(|| format!("failed to create {}", plan.model_cache_dir.display()))?;
    if tokenizer_json != plan.tokenizer {
        fs::copy(&tokenizer_json, &plan.tokenizer).with_context(|| {
            format!(
                "failed to copy {} to {}",
                tokenizer_json.display(),
                plan.tokenizer.display()
            )
        })?;
    }
    if tokenizer_config_json.is_file() && tokenizer_config_json != plan.tokenizer_config {
        fs::copy(&tokenizer_config_json, &plan.tokenizer_config).with_context(|| {
            format!(
                "failed to copy {} to {}",
                tokenizer_config_json.display(),
                plan.tokenizer_config.display()
            )
        })?;
    }
    Ok(())
}

pub(super) fn build_manifest(
    model: &str,
    snapshot_dir: &Path,
    index_json: &Path,
) -> Result<(FlashMoeManifest, Vec<DenseTensorRef>)> {
    let index: SafetensorsIndex = serde_json::from_slice(
        &fs::read(index_json)
            .with_context(|| format!("failed to read {}", index_json.display()))?,
    )
    .with_context(|| format!("failed to parse {}", index_json.display()))?;
    let mut dense_shards = BTreeSet::new();
    let mut dense_tensor_refs = Vec::new();
    let mut visual_tensor_refs = Vec::new();
    let mut expert_tensors = Vec::new();
    let mut shard_cache = BTreeMap::<String, SafetensorShard>::new();
    let mut runtime_offset = 0u64;
    let mut visual_offset = 0u64;
    for (tensor, shard) in &index.weight_map {
        let canonical_tensor = canonical_hf_tensor_name(&tensor);
        if skip_flashmoe_runtime_tensor(&canonical_tensor) {
            continue;
        }
        if is_q4_aux_tensor_name(&canonical_tensor)
            && index
                .weight_map
                .contains_key(q4_weight_name_for_aux(&tensor).as_str())
        {
            continue;
        }
        let shard_path = snapshot_dir.join(shard);
        if !shard_path.is_file() {
            bail!(
                "safetensors shard referenced by index is missing: {}",
                shard_path.display()
            );
        }
        if !shard_cache.contains_key(shard) {
            shard_cache.insert(shard.clone(), parse_safetensors_header(&shard_path)?);
        }
        let shard_info = shard_cache.get(shard).expect("inserted above");
        let tensor_info = shard_info.tensors.get(tensor).with_context(|| {
            format!("tensor {tensor} listed in index but missing from safetensors header {shard}")
        })?;
        let tensor_dtype = tensor_info.dtype.clone();
        let tensor_shape = tensor_info.shape.clone();
        let tensor_source_offsets = tensor_info.data_offsets;
        if is_expert_tensor_name(&canonical_tensor) {
            let (layer, expert) = parse_layer_expert(&canonical_tensor);
            let native_q4 = dense_native_q4_sources(
                snapshot_dir,
                &index.weight_map,
                &mut shard_cache,
                &tensor,
            )?;
            let runtime_shape = if native_q4.is_some() {
                logical_shape_for_mlx_q4(&tensor_shape)?
            } else {
                tensor_shape
            };
            expert_tensors.push(ExpertTensorRef {
                tensor: canonical_tensor,
                shard: shard.clone(),
                layer,
                expert,
                dtype: Some(tensor_dtype),
                shape: runtime_shape,
                source_offsets: Some(tensor_source_offsets),
                q4_sources: native_q4,
            });
        } else if canonical_tensor.starts_with("visual.") {
            // Vision encoder tensors go into a separate store.
            let byte_len = tensor_source_offsets[1]
                .checked_sub(tensor_source_offsets[0])
                .with_context(|| format!("invalid data_offsets for visual tensor {tensor}"))?;
            visual_offset = align_to(visual_offset, TENSOR_ALIGNMENT);
            visual_tensor_refs.push(DenseTensorRef {
                tensor: canonical_tensor,
                shard: shard.clone(),
                dtype: tensor_dtype,
                shape: tensor_shape,
                source_offsets: tensor_source_offsets,
                runtime_offset: visual_offset,
                byte_len,
                quantization: TensorQuantization::None,
                q4_sources: None,
            });
            visual_offset = visual_offset.saturating_add(byte_len);
        } else {
            dense_shards.insert(shard.clone());
            let source_byte_len = tensor_source_offsets[1]
                .checked_sub(tensor_source_offsets[0])
                .with_context(|| format!("invalid data_offsets for tensor {tensor}"))?;
            let native_q4 = dense_native_q4_sources(
                snapshot_dir,
                &index.weight_map,
                &mut shard_cache,
                &tensor,
            )?;
            let quantization =
                dense_tensor_quantization(&canonical_tensor, &tensor_dtype, &native_q4);
            let runtime_shape = if native_q4.is_some() {
                logical_shape_for_mlx_q4(&tensor_shape)?
            } else {
                tensor_shape.clone()
            };
            let byte_len = match &quantization {
                TensorQuantization::None => source_byte_len,
                TensorQuantization::Q4 {
                    group_size,
                    scale_bias_dtype,
                    ..
                } => {
                    dense_q4_layout_with_scale_bias_dtype(
                        &runtime_shape,
                        *group_size,
                        scale_bias_dtype,
                    )?
                    .total_bytes as u64
                }
            };
            runtime_offset = align_to(runtime_offset, TENSOR_ALIGNMENT);
            dense_tensor_refs.push(DenseTensorRef {
                tensor: canonical_tensor,
                shard: shard.clone(),
                dtype: tensor_dtype,
                shape: runtime_shape,
                source_offsets: tensor_source_offsets,
                runtime_offset,
                byte_len,
                quantization,
                q4_sources: native_q4,
            });
            runtime_offset = runtime_offset.saturating_add(byte_len);
        }
    }
    Ok((
        FlashMoeManifest {
            model: canonical_model(model),
            cache_version: cache_version_for_model(model).to_string(),
            dense_shards: dense_shards.into_iter().collect(),
            expert_tensors,
            dense_tensors: dense_tensor_refs,
        },
        visual_tensor_refs,
    ))
}

pub(super) fn pack_expert_tensors(
    snapshot_dir: &Path,
    plan: &FlashMoePlan,
    expert_tensors: &[ExpertTensorRef],
    config: Option<&QwenModelConfig>,
) -> Result<()> {
    let mut by_layer: BTreeMap<usize, BTreeMap<usize, Vec<&ExpertTensorRef>>> = BTreeMap::new();
    let mut aggregate_by_layer: BTreeMap<usize, Vec<&ExpertTensorRef>> = BTreeMap::new();
    for tensor in expert_tensors {
        if let (Some(layer), Some(expert)) = (tensor.layer, tensor.expert) {
            by_layer
                .entry(layer)
                .or_default()
                .entry(expert)
                .or_default()
                .push(tensor);
        } else if let Some(layer) = tensor.layer
            && aggregate_expert_tensor_kind(&tensor.tensor).is_some()
        {
            aggregate_by_layer.entry(layer).or_default().push(tensor);
        }
    }

    let deleted_temps = cleanup_stale_expert_temp_files(&plan.experts_dir)?;
    if deleted_temps > 0 {
        eprintln!(
            "deleted {deleted_temps} stale temporary expert pack file(s) from {}",
            plan.experts_dir.display()
        );
    }

    let aggregate_layers = aggregate_by_layer.len();
    if aggregate_layers > 0 {
        eprintln!("packing aggregate experts across {aggregate_layers} layer(s)");
    }
    let mut shard_cache = BTreeMap::<String, (memmap2::Mmap, SafetensorShard)>::new();
    for (layer_index, (layer, tensors)) in aggregate_by_layer.into_iter().enumerate() {
        pack_aggregate_expert_layer(
            snapshot_dir,
            plan,
            layer,
            layer_index + 1,
            aggregate_layers,
            &tensors,
            config,
        )?;
    }
    for (layer, experts) in by_layer {
        pack_direct_expert_layer(snapshot_dir, plan, layer, experts, config, &mut shard_cache)?;
    }
    Ok(())
}

pub(super) fn fixed_dense_expert_slot_spec_for_pack(
    plan: &FlashMoePlan,
    config: Option<&QwenModelConfig>,
) -> Result<Option<FixedDenseExpertSlotSpec>> {
    let dtype = match plan.quantization {
        ExpertQuantization::FourBitProduction => return Ok(None),
        ExpertQuantization::Bf16 => DenseExpertDtype::Bf16,
        ExpertQuantization::F16 => DenseExpertDtype::F16,
    };
    let config = config.context("Qwen config is required for fixed dense expert packing")?;
    let layout = QwenMoeModelLayout::from_config(&plan.model, config)?;
    FixedDenseExpertSlotSpec::from_model_layout(&layout, dtype).map(Some)
}

pub(super) fn pack_direct_expert_layer(
    snapshot_dir: &Path,
    plan: &FlashMoePlan,
    layer: usize,
    experts: BTreeMap<usize, Vec<&ExpertTensorRef>>,
    config: Option<&QwenModelConfig>,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
) -> Result<()> {
    let fixed_dense = fixed_dense_expert_slot_spec_for_pack(plan, config)?;
    let mut expected = Vec::with_capacity(experts.len());
    for (expert, tensors) in &experts {
        validate_expert_tensor_group(layer, *expert, tensors, config)?;
        expected.push(match fixed_dense {
            Some(spec) => expected_fixed_dense_expert_pack(
                snapshot_dir,
                shard_cache,
                layer,
                *expert,
                tensors,
                spec,
            )?,
            None => expected_expert_pack(snapshot_dir, shard_cache, *expert, tensors)?,
        });
    }
    let expert_count = layer_expert_count(config, &experts);
    rewrite_expert_layer_pack(
        &plan.experts_dir,
        layer,
        expert_count,
        match fixed_dense {
            Some(spec) => ExpertLayerStorageFormat::FixedDense(spec),
            None => ExpertLayerStorageFormat::Pbq4Import,
        },
        &expected,
        |expert| {
            let tensors = experts
                .get(&expert)
                .with_context(|| format!("missing expert {expert} tensors for layer {layer}"))?;
            build_direct_expert_pack(
                snapshot_dir,
                shard_cache,
                layer,
                expert,
                tensors,
                fixed_dense,
            )
        },
    )?;
    Ok(())
}

pub(super) fn layer_expert_count(
    config: Option<&QwenModelConfig>,
    experts: &BTreeMap<usize, Vec<&ExpertTensorRef>>,
) -> usize {
    let declared = config.map(|config| config.experts()).unwrap_or(0);
    let observed = experts
        .keys()
        .next_back()
        .map(|expert| expert + 1)
        .unwrap_or(0);
    declared.max(observed).max(1)
}

pub(super) fn build_direct_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    tensors: &[&ExpertTensorRef],
    fixed_dense: Option<FixedDenseExpertSlotSpec>,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    if let Some(spec) = fixed_dense {
        let inputs = tensors
            .iter()
            .map(|tensor| {
                fixed_dense_expert_record_input(
                    snapshot_dir,
                    shard_cache,
                    tensor,
                    tensor.tensor.clone(),
                    tensor.shape.clone(),
                    0,
                    tensor.shape.iter().product(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        return build_fixed_dense_expert_pack(layer, expert, spec, inputs);
    }
    let mut inputs = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let dtype = tensor.dtype.as_deref().unwrap_or("unknown");
        let (values, source_offsets, source_hash) = decode_expert_tensor_range(
            snapshot_dir,
            shard_cache,
            tensor,
            0,
            tensor.shape.iter().product(),
        )?;
        inputs.push(ExpertRecordInput {
            tensor: tensor.tensor.clone(),
            dtype: dtype.to_string(),
            shape: tensor.shape.clone(),
            source_offsets,
            source_hash: Some(source_hash),
            values,
        });
    }
    build_expert_pack(layer, expert, inputs)
}

pub(super) fn pack_aggregate_expert_layer(
    snapshot_dir: &Path,
    plan: &FlashMoePlan,
    layer: usize,
    layer_index: usize,
    layer_total: usize,
    tensors: &[&ExpertTensorRef],
    config: Option<&QwenModelConfig>,
) -> Result<()> {
    let config = config.context("Qwen config is required to split aggregate expert tensors")?;
    let intermediate = config
        .moe_intermediate_size
        .or(config.intermediate_size)
        .context("Qwen config is missing moe_intermediate_size/intermediate_size for aggregate expert packing")?;
    let layout = AggregateExpertLayout::new(config.experts(), config.hidden_size, intermediate)?;

    let aggregate_tensors = aggregate_expert_tensors(tensors, layer, layout)?;
    let down = single_aggregate_expert_tensor(tensors, AggregateExpertTensorKind::Down, layer)?;
    validate_aggregate_expert_tensor_shape(
        down,
        &[layout.experts, layout.hidden, layout.intermediate],
        "down_proj",
    )?;

    eprintln!(
        "packing aggregate experts for layer {layer} ({layer_index}/{layer_total}): {} experts",
        layout.experts
    );
    let fixed_dense = fixed_dense_expert_slot_spec_for_pack(plan, Some(config))?;
    let fixed_native_q4 = fixed_native_q4_aggregate_layout(&aggregate_tensors, down, layout)?;
    let mut shard_cache = BTreeMap::<String, (memmap2::Mmap, SafetensorShard)>::new();
    let mut expected = Vec::with_capacity(layout.experts);
    for expert in 0..layout.experts {
        let records = match fixed_dense {
            Some(spec) => expected_fixed_dense_aggregate_expert_records(
                snapshot_dir,
                &mut shard_cache,
                layer,
                expert,
                &aggregate_tensors,
                down,
                layout,
                spec,
            )?,
            None => expected_aggregate_expert_records(
                snapshot_dir,
                &mut shard_cache,
                layer,
                expert,
                &aggregate_tensors,
                down,
                layout,
            )?,
        };
        let packed_bytes = match (fixed_dense, fixed_native_q4) {
            (Some(spec), _) => spec.expert_bytes as u64,
            (None, Some(fixed)) => fixed.expert_bytes as u64,
            (None, None) => pbq4_expert_pack_wire_size(&records)?,
        };
        expected.push(ExpectedExpertPack {
            expert,
            packed_bytes,
            records,
        });
    }
    let skipped =
        rewrite_expert_layer_pack(
            &plan.experts_dir,
            layer,
            layout.experts,
            match (fixed_dense, fixed_native_q4) {
                (Some(spec), _) => ExpertLayerStorageFormat::FixedDense(spec),
                (None, Some(fixed)) => ExpertLayerStorageFormat::FixedQ4(
                    FixedQ4ExpertSlotSpec::new(fixed, layout.hidden, layout.intermediate)?,
                ),
                (None, None) => ExpertLayerStorageFormat::Pbq4Import,
            },
            &expected,
            |expert| {
                if let Some(spec) = fixed_dense {
                    build_fixed_dense_aggregate_expert_pack(
                        snapshot_dir,
                        &mut shard_cache,
                        layer,
                        expert,
                        &aggregate_tensors,
                        down,
                        layout,
                        spec,
                    )
                } else {
                    build_aggregate_expert_pack(
                        snapshot_dir,
                        &mut shard_cache,
                        layer,
                        expert,
                        &aggregate_tensors,
                        down,
                        layout,
                    )
                }
            },
        )?;
    eprintln!(
        "prepared aggregate experts for layer {layer} ({layer_index}/{layer_total}): {}/{} ({skipped} reused)",
        layout.experts, layout.experts,
    );
    Ok(())
}

pub(super) fn build_aggregate_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    if aggregate_native_q4_enabled(aggregate_tensors, down)? {
        return build_native_q4_aggregate_expert_pack(
            snapshot_dir,
            shard_cache,
            layer,
            expert,
            aggregate_tensors,
            down,
            layout,
        );
    }

    let mut inputs = Vec::with_capacity(3);
    let (gate_values, gate_offsets, gate_hash) = decode_expert_tensor_range(
        snapshot_dir,
        shard_cache,
        aggregate_tensors.gate.tensor,
        aggregate_tensors.gate.start(expert)?,
        layout.single_projection_values,
    )?;
    inputs.push(ExpertRecordInput {
        tensor: format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
        dtype: aggregate_tensors
            .gate
            .tensor
            .dtype
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        shape: vec![layout.intermediate, layout.hidden],
        source_offsets: gate_offsets,
        source_hash: Some(gate_hash),
        values: gate_values,
    });

    let (up_values, up_offsets, up_hash) = decode_expert_tensor_range(
        snapshot_dir,
        shard_cache,
        aggregate_tensors.up.tensor,
        aggregate_tensors.up.start(expert)?,
        layout.single_projection_values,
    )?;
    inputs.push(ExpertRecordInput {
        tensor: format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
        dtype: aggregate_tensors
            .up
            .tensor
            .dtype
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        shape: vec![layout.intermediate, layout.hidden],
        source_offsets: up_offsets,
        source_hash: Some(up_hash),
        values: up_values,
    });

    let down_base = expert
        .checked_mul(layout.down_expert_values)
        .context("aggregate down expert offset overflow")?;
    let (down_values, down_offsets, down_hash) = decode_expert_tensor_range(
        snapshot_dir,
        shard_cache,
        down,
        down_base,
        layout.down_expert_values,
    )?;
    inputs.push(ExpertRecordInput {
        tensor: format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
        dtype: down.dtype.clone().unwrap_or_else(|| "unknown".to_string()),
        shape: vec![layout.hidden, layout.intermediate],
        source_offsets: down_offsets,
        source_hash: Some(down_hash),
        values: down_values,
    });
    build_expert_pack(layer, expert, inputs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_fixed_dense_aggregate_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
    spec: FixedDenseExpertSlotSpec,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    let inputs = vec![
        fixed_dense_expert_record_input(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.gate.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.gate.start(expert)?,
            layout.single_projection_values,
        )?,
        fixed_dense_expert_record_input(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.up.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.up.start(expert)?,
            layout.single_projection_values,
        )?,
        fixed_dense_expert_record_input(
            snapshot_dir,
            shard_cache,
            down,
            format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
            vec![layout.hidden, layout.intermediate],
            expert
                .checked_mul(layout.down_expert_values)
                .context("aggregate down expert offset overflow")?,
            layout.down_expert_values,
        )?,
    ];
    build_fixed_dense_expert_pack(layer, expert, spec, inputs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn expected_fixed_dense_aggregate_expert_records(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
    spec: FixedDenseExpertSlotSpec,
) -> Result<Vec<ExpectedExpertPackRecord>> {
    let sources = [
        (
            aggregate_tensors.gate.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.gate.start(expert)?,
            layout.single_projection_values,
            ExpertMlpProjection::Gate,
        ),
        (
            aggregate_tensors.up.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.up.start(expert)?,
            layout.single_projection_values,
            ExpertMlpProjection::Up,
        ),
        (
            down,
            format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
            vec![layout.hidden, layout.intermediate],
            expert
                .checked_mul(layout.down_expert_values)
                .context("aggregate down expert offset overflow")?,
            layout.down_expert_values,
            ExpertMlpProjection::Down,
        ),
    ];
    sources
        .into_iter()
        .map(|(source, tensor, shape, offset, count, projection)| {
            let (source_offsets, source_hash) =
                expert_tensor_source_fingerprint(snapshot_dir, shard_cache, source, offset, count)?;
            let component = spec.projection(projection);
            Ok(ExpectedExpertPackRecord {
                tensor,
                dtype: source
                    .dtype
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                shape,
                source_offsets,
                source_hash,
                packed_bytes: component.bytes as u64,
                groups: 0,
                group_size: 0,
                scale_bias_dtype: spec.dtype.as_str().to_string(),
            })
        })
        .collect()
}

pub(super) fn expected_aggregate_expert_records(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
) -> Result<Vec<ExpectedExpertPackRecord>> {
    if aggregate_native_q4_enabled(aggregate_tensors, down)? {
        return expected_native_q4_aggregate_expert_records(
            snapshot_dir,
            shard_cache,
            layer,
            expert,
            aggregate_tensors,
            down,
            layout,
        );
    }

    let gate = expected_expert_pack_record(
        snapshot_dir,
        shard_cache,
        aggregate_tensors.gate.tensor,
        format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
        vec![layout.intermediate, layout.hidden],
        aggregate_tensors.gate.start(expert)?,
        layout.single_projection_values,
    )?;
    let up = expected_expert_pack_record(
        snapshot_dir,
        shard_cache,
        aggregate_tensors.up.tensor,
        format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
        vec![layout.intermediate, layout.hidden],
        aggregate_tensors.up.start(expert)?,
        layout.single_projection_values,
    )?;
    let down_base = expert
        .checked_mul(layout.down_expert_values)
        .context("aggregate down expert offset overflow")?;
    let down = expected_expert_pack_record(
        snapshot_dir,
        shard_cache,
        down,
        format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
        vec![layout.hidden, layout.intermediate],
        down_base,
        layout.down_expert_values,
    )?;
    Ok(vec![gate, up, down])
}

pub(super) fn build_native_q4_aggregate_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    let inputs = vec![
        native_q4_expert_record_input(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.gate.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.gate.start(expert)?,
            layout.single_projection_values,
        )?,
        native_q4_expert_record_input(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.up.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.up.start(expert)?,
            layout.single_projection_values,
        )?,
        native_q4_expert_record_input(
            snapshot_dir,
            shard_cache,
            down,
            format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
            vec![layout.hidden, layout.intermediate],
            expert
                .checked_mul(layout.down_expert_values)
                .context("aggregate down expert offset overflow")?,
            layout.down_expert_values,
        )?,
    ];
    if let Some(fixed) = fixed_native_q4_aggregate_layout(aggregate_tensors, down, layout)? {
        return build_fixed_native_q4_expert_pack(layer, expert, fixed, inputs);
    }
    build_native_q4_expert_pack(layer, expert, inputs)
}

pub(super) fn expected_native_q4_aggregate_expert_records(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
) -> Result<Vec<ExpectedExpertPackRecord>> {
    Ok(vec![
        expected_native_q4_expert_record(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.gate.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.gate.start(expert)?,
            layout.single_projection_values,
        )?,
        expected_native_q4_expert_record(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.up.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.up.start(expert)?,
            layout.single_projection_values,
        )?,
        expected_native_q4_expert_record(
            snapshot_dir,
            shard_cache,
            down,
            format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
            vec![layout.hidden, layout.intermediate],
            expert
                .checked_mul(layout.down_expert_values)
                .context("aggregate down expert offset overflow")?,
            layout.down_expert_values,
        )?,
    ])
}

pub(super) fn expected_native_q4_expert_record(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    source: &ExpertTensorRef,
    tensor: String,
    shape: Vec<usize>,
    element_offset: usize,
    element_count: usize,
) -> Result<ExpectedExpertPackRecord> {
    let input = native_q4_expert_record_input(
        snapshot_dir,
        shard_cache,
        source,
        tensor,
        shape,
        element_offset,
        element_count,
    )?;
    expected_native_q4_expert_record_from_input(input)
}

pub(super) fn expected_expert_pack_record(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    source: &ExpertTensorRef,
    tensor: String,
    shape: Vec<usize>,
    element_offset: usize,
    element_count: usize,
) -> Result<ExpectedExpertPackRecord> {
    let (source_offsets, source_hash) = expert_tensor_source_fingerprint(
        snapshot_dir,
        shard_cache,
        source,
        element_offset,
        element_count,
    )?;
    expected_expert_pack_record_from_source(
        tensor,
        source
            .dtype
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        shape,
        source_offsets,
        source_hash,
    )
}

pub(super) fn expected_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    expert: usize,
    tensors: &[&ExpertTensorRef],
) -> Result<ExpectedExpertPack> {
    let mut records = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let shape = tensor.shape.clone();
        let element_count = shape.iter().product();
        records.push(expected_expert_pack_record(
            snapshot_dir,
            shard_cache,
            tensor,
            tensor.tensor.clone(),
            shape,
            0,
            element_count,
        )?);
    }
    expected_expert_pack_from_records(expert, records)
}

pub(super) fn expected_fixed_dense_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    tensors: &[&ExpertTensorRef],
    spec: FixedDenseExpertSlotSpec,
) -> Result<ExpectedExpertPack> {
    let mut records = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let projection = if tensor.tensor.ends_with("gate_proj.weight") {
            ExpertMlpProjection::Gate
        } else if tensor.tensor.ends_with("up_proj.weight") {
            ExpertMlpProjection::Up
        } else if tensor.tensor.ends_with("down_proj.weight") {
            ExpertMlpProjection::Down
        } else {
            bail!(
                "fixed {} expert pack layer {layer} expert {expert} has unknown tensor {}",
                spec.dtype.as_str(),
                tensor.tensor
            );
        };
        let component = spec.projection(projection);
        if tensor.shape != [component.rows, component.cols] {
            bail!(
                "fixed {} expert tensor {} has shape {:?}, expected [{}, {}]",
                spec.dtype.as_str(),
                tensor.tensor,
                tensor.shape,
                component.rows,
                component.cols
            );
        }
        let (source_offsets, source_hash) = expert_tensor_source_fingerprint(
            snapshot_dir,
            shard_cache,
            tensor,
            0,
            tensor.shape.iter().product(),
        )?;
        records.push(ExpectedExpertPackRecord {
            tensor: tensor.tensor.clone(),
            dtype: tensor
                .dtype
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            shape: tensor.shape.clone(),
            source_offsets,
            source_hash,
            packed_bytes: component.bytes as u64,
            groups: 0,
            group_size: 0,
            scale_bias_dtype: spec.dtype.as_str().to_string(),
        });
    }
    Ok(ExpectedExpertPack {
        expert,
        packed_bytes: spec.expert_bytes as u64,
        records,
    })
}

pub(super) fn decode_expert_tensor_range(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    tensor: &ExpertTensorRef,
    element_offset: usize,
    element_count: usize,
) -> Result<(Vec<f32>, [u64; 2], String)> {
    with_expert_tensor_raw_range(
        snapshot_dir,
        shard_cache,
        tensor,
        element_offset,
        element_count,
        |raw, source_offsets, dtype| {
            let values = decode_dense_tensor_f32(dtype, raw).with_context(|| {
                format!(
                    "failed to decode expert tensor {} as {dtype} before q4 quantization",
                    tensor.tensor
                )
            })?;
            Ok((values, source_offsets, sha256_hex(raw)))
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fixed_dense_expert_record_input(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    source: &ExpertTensorRef,
    tensor: String,
    shape: Vec<usize>,
    element_offset: usize,
    element_count: usize,
) -> Result<FixedDenseExpertRecordInput> {
    with_expert_tensor_raw_range(
        snapshot_dir,
        shard_cache,
        source,
        element_offset,
        element_count,
        |raw, source_offsets, dtype| {
            Ok(FixedDenseExpertRecordInput {
                tensor,
                dtype: dtype.to_string(),
                shape,
                source_offsets,
                source_hash: Some(sha256_hex(raw)),
                bytes: raw.to_vec(),
            })
        },
    )
}

pub(super) fn expert_tensor_source_fingerprint(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    tensor: &ExpertTensorRef,
    element_offset: usize,
    element_count: usize,
) -> Result<([u64; 2], String)> {
    with_expert_tensor_raw_range(
        snapshot_dir,
        shard_cache,
        tensor,
        element_offset,
        element_count,
        |raw, source_offsets, _| Ok((source_offsets, sha256_hex(raw))),
    )
}

pub(super) fn with_expert_tensor_raw_range<R>(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    tensor: &ExpertTensorRef,
    element_offset: usize,
    element_count: usize,
    read: impl FnOnce(&[u8], [u64; 2], &str) -> Result<R>,
) -> Result<R> {
    if !shard_cache.contains_key(&tensor.shard) {
        let shard_path = snapshot_dir.join(&tensor.shard);
        let file = fs::File::open(&shard_path)
            .with_context(|| format!("failed to open shard {}", shard_path.display()))?;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .with_context(|| format!("failed to memory-map {}", shard_path.display()))?
        };
        shard_cache.insert(
            tensor.shard.clone(),
            (mmap, parse_safetensors_header(&shard_path)?),
        );
    }
    let (bytes, shard) = shard_cache.get(&tensor.shard).expect("inserted above");
    let dtype = tensor.dtype.as_deref().unwrap_or("unknown");
    let [byte_start, byte_end] =
        expert_tensor_byte_range(tensor, dtype, element_offset, element_count)?;
    let abs_start = shard.data_start + byte_start;
    let abs_end = shard.data_start + byte_end;
    let raw = &bytes[abs_start as usize..abs_end as usize];
    read(raw, [byte_start, byte_end], dtype)
}

pub(super) fn native_q4_expert_record_input(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    source: &ExpertTensorRef,
    tensor: String,
    shape: Vec<usize>,
    element_offset: usize,
    element_count: usize,
) -> Result<NativeQ4ExpertRecordInput> {
    let q4_sources = source
        .q4_sources
        .as_ref()
        .with_context(|| format!("expert tensor {} is not native MLX Q4", source.tensor))?;
    let slice = native_q4_slice_byte_ranges(
        source,
        shape.as_slice(),
        &q4_sources.scale_bias_dtype,
        element_offset,
        element_count,
    )?;
    let source_offsets = source
        .source_offsets
        .with_context(|| format!("expert tensor {} is missing source offsets", source.tensor))?;
    let (packed, packed_offsets) = read_safetensor_source_byte_range(
        snapshot_dir,
        shard_cache,
        &source.shard,
        source_offsets,
        slice.packed_offset,
        slice.packed_bytes,
    )?;
    let (scale_bytes, _) = read_safetensor_source_byte_range(
        snapshot_dir,
        shard_cache,
        &q4_sources.scales_shard,
        q4_sources.scales_offsets,
        slice.scale_bias_offset,
        slice.scale_bias_bytes,
    )?;
    let (bias_bytes, _) = read_safetensor_source_byte_range(
        snapshot_dir,
        shard_cache,
        &q4_sources.biases_shard,
        q4_sources.biases_offsets,
        slice.scale_bias_offset,
        slice.scale_bias_bytes,
    )?;
    let source_hash = sha256_hex_parts(&[&packed, &scale_bytes, &bias_bytes]);
    Ok(NativeQ4ExpertRecordInput {
        tensor,
        dtype: source
            .dtype
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        shape,
        source_offsets: packed_offsets,
        source_hash: Some(source_hash),
        packed,
        scale_bytes,
        bias_bytes,
        groups: slice.groups,
        scale_bias_dtype: q4_sources.scale_bias_dtype.clone(),
    })
}

pub(super) fn read_safetensor_source_byte_range(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    shard_name: &str,
    data_offsets: [u64; 2],
    relative_offset: usize,
    byte_len: usize,
) -> Result<(Vec<u8>, [u64; 2])> {
    if !shard_cache.contains_key(shard_name) {
        let shard_path = snapshot_dir.join(shard_name);
        let file = fs::File::open(&shard_path)
            .with_context(|| format!("failed to open shard {}", shard_path.display()))?;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .with_context(|| format!("failed to memory-map {}", shard_path.display()))?
        };
        shard_cache.insert(
            shard_name.to_string(),
            (mmap, parse_safetensors_header(&shard_path)?),
        );
    }
    let (bytes, shard) = shard_cache.get(shard_name).expect("inserted above");
    let byte_start = data_offsets[0]
        .checked_add(relative_offset as u64)
        .context("safetensor source byte offset overflow")?;
    let byte_end = byte_start
        .checked_add(byte_len as u64)
        .context("safetensor source byte range overflow")?;
    if byte_end > data_offsets[1] {
        bail!(
            "safetensor source range {byte_start}..{byte_end} exceeds offsets {:?} in {shard_name}",
            data_offsets
        );
    }
    let abs_start = shard
        .data_start
        .checked_add(byte_start)
        .context("safetensor absolute byte offset overflow")?;
    let abs_end = shard
        .data_start
        .checked_add(byte_end)
        .context("safetensor absolute byte range overflow")?;
    Ok((
        bytes[abs_start as usize..abs_end as usize].to_vec(),
        [byte_start, byte_end],
    ))
}

pub(super) fn validate_expert_tensor_group(
    layer: usize,
    expert: usize,
    tensors: &[&ExpertTensorRef],
    config: Option<&QwenModelConfig>,
) -> Result<()> {
    let shape = if let Some(config) = config {
        let intermediate = config
            .moe_intermediate_size
            .or(config.intermediate_size)
            .context("Qwen config is missing moe_intermediate_size/intermediate_size for expert validation")?;
        Some(DirectExpertTensorShape::new(
            config.hidden_size,
            intermediate,
        )?)
    } else {
        None
    };
    validate_direct_expert_tensor_group(layer, expert, tensors, shape)
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

pub(super) fn sha256_hex_parts(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

pub(super) fn is_expert_tensor_name(name: &str) -> bool {
    name.starts_with("model.layers.")
        && (name.contains(".experts.")
            || name.contains(".mlp.experts")
            || name.contains(".mlp.switch_mlp."))
}

pub(super) fn parse_layer_expert(name: &str) -> (Option<usize>, Option<usize>) {
    let parts: Vec<&str> = name.split('.').collect();
    let mut layer = None;
    let mut expert = None;
    for window in parts.windows(2) {
        match window[0] {
            "layers" => layer = window[1].parse().ok(),
            "experts" => expert = window[1].parse().ok(),
            _ => {}
        }
    }
    (layer, expert)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_owner_declares_required_huggingface_artifacts() {
        let files = expected_hf_files();
        assert!(files.contains(&OsString::from("config.json")));
        assert!(files.contains(&OsString::from("tokenizer_config.json")));
        assert!(files.contains(&OsString::from("model.safetensors.index.json")));
        assert_eq!(expected_vl_hf_files(), files);
    }

    #[test]
    fn cache_owner_parses_only_explicit_layer_and_expert_components() {
        assert_eq!(
            parse_layer_expert("model.layers.12.mlp.experts.7.gate_proj.weight"),
            (Some(12), Some(7))
        );
        assert_eq!(
            parse_layer_expert("model.layers.3.mlp.switch_mlp.gate_proj.weight"),
            (Some(3), None)
        );
    }
}
