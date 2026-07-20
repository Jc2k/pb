use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};

use super::experts::*;
use super::metal::METAL_SHADERS;
use super::model_family::{QwenModelConfig, is_glm52};
use super::planning::*;
use super::safetensors::*;
use super::types::ExpertQuantization;
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
        "chat_template.jinja",
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
    build_cache_from_hf_snapshot_with_quantization(
        model,
        snapshot_dir,
        default_expert_quantization(model),
    )
}

pub fn build_cache_from_hf_snapshot_with_quantization(
    model: &str,
    snapshot_dir: &Path,
    quantization: ExpertQuantization,
) -> Result<FlashMoePlan> {
    let plan = plan_unchecked_with_quantization(
        model,
        snapshot_dir.parent().unwrap_or(snapshot_dir),
        quantization,
    );
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
    let (mut manifest, visual_tensor_refs) = if index_json.is_file() {
        build_manifest(model, snapshot_dir, &index_json, config.as_ref())?
    } else {
        build_unindexed_manifest(model, snapshot_dir, config.as_ref())?
    };
    let runtime_cache_version =
        if is_glm52(&plan.model) && plan.quantization == ExpertQuantization::FourBitProduction {
            super::types::GLM52_CACHE_VERSION
        } else if config.as_ref().is_some_and(QwenModelConfig::is_qwen3_next)
            && plan.quantization == ExpertQuantization::FourBitProduction
        {
            super::types::QWEN3_NEXT_CACHE_VERSION
        } else {
            plan.quantization.cache_version()
        };
    manifest.cache_version = runtime_cache_version.to_string();
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
        config.as_ref(),
    )?;
    pack_expert_tensors(
        snapshot_dir,
        ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
        &manifest.expert_tensors,
        config.as_ref(),
    )?;

    // For VL models, build and write the vision weights store.
    if let (Some(vision_weights), Some(vision_manifest)) =
        (plan.vision_weights.as_ref(), plan.vision_manifest.as_ref())
    {
        if !visual_tensor_refs.is_empty() {
            write_dense_tensor_store(snapshot_dir, vision_weights, &visual_tensor_refs, None)?;
            let vision_manifest_data = FlashMoeManifest {
                model: canonical_model(model),
                cache_version: runtime_cache_version.to_string(),
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
    let chat_template = snapshot_dir.join("chat_template.jinja");
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
    if chat_template.is_file() && chat_template != plan.chat_template {
        fs::copy(&chat_template, &plan.chat_template).with_context(|| {
            format!(
                "failed to copy {} to {}",
                chat_template.display(),
                plan.chat_template.display()
            )
        })?;
    }
    Ok(())
}

pub(super) fn build_manifest(
    model: &str,
    snapshot_dir: &Path,
    index_json: &Path,
    config: Option<&QwenModelConfig>,
) -> Result<(FlashMoeManifest, Vec<DenseTensorRef>)> {
    let resolved_manifest = resolve_safetensors_manifest(snapshot_dir, index_json)?;
    if resolved_manifest.source == SafetensorsManifestSource::ActualShardHeaders {
        tracing::info!(
            snapshot = %snapshot_dir.display(),
            "resolved safetensors manifest from actual shard headers because the declared index references missing shards"
        );
    }
    build_manifest_from_resolved(model, snapshot_dir, config, resolved_manifest)
}

pub(super) fn build_unindexed_manifest(
    model: &str,
    snapshot_dir: &Path,
    config: Option<&QwenModelConfig>,
) -> Result<(FlashMoeManifest, Vec<DenseTensorRef>)> {
    let resolved_manifest = resolve_unindexed_safetensors_manifest(snapshot_dir)?;
    tracing::info!(
        snapshot = %snapshot_dir.display(),
        "resolved unindexed safetensors manifest from actual shard headers"
    );
    build_manifest_from_resolved(model, snapshot_dir, config, resolved_manifest)
}

fn build_manifest_from_resolved(
    model: &str,
    snapshot_dir: &Path,
    config: Option<&QwenModelConfig>,
    resolved_manifest: ResolvedSafetensorsManifest,
) -> Result<(FlashMoeManifest, Vec<DenseTensorRef>)> {
    let weight_map = resolved_manifest.weight_map;
    let mut dense_shards = BTreeSet::new();
    let mut dense_tensor_refs = Vec::new();
    let mut visual_tensor_refs = Vec::new();
    let mut expert_tensors = Vec::new();
    let mut shard_cache = resolved_manifest.shards;
    let mut runtime_offset = 0u64;
    let mut visual_offset = 0u64;
    for (tensor, shard) in &weight_map {
        let canonical_tensor = canonical_hf_tensor_name(tensor);
        if skip_flashmoe_runtime_tensor(&canonical_tensor) {
            continue;
        }
        if let Some(config) = config
            && config.glm.is_some()
            && (canonical_tensor.contains(".indexer.")
                || canonical_tensor.contains(".indexers_proj.")
                || canonical_tensor.contains(".shared_head.")
                || canonical_tensor.ends_with(".eh_proj.weight")
                || canonical_tensor.ends_with(".enorm.weight")
                || canonical_tensor.ends_with(".hnorm.weight")
                || tensor_layer(&canonical_tensor)
                    .is_some_and(|layer| layer >= config.num_hidden_layers))
        {
            continue;
        }
        if is_q4_aux_tensor_name(&canonical_tensor)
            && weight_map.contains_key(q4_weight_name_for_aux(tensor).as_str())
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
            let glm_shape =
                config.and_then(|config| config.glm_logical_tensor_shape(&canonical_tensor));
            let native_q4 = dense_native_q4_sources(
                snapshot_dir,
                &weight_map,
                &mut shard_cache,
                tensor,
                glm_shape.as_deref(),
            )?;
            let runtime_shape = if native_q4.is_some() {
                match glm_shape {
                    Some(shape) => shape,
                    None => logical_shape_for_mlx_source(
                        &tensor_shape,
                        native_q4.as_ref().expect("checked above"),
                    )?,
                }
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
            let glm_shape =
                config.and_then(|config| config.glm_logical_tensor_shape(&canonical_tensor));
            let native_q4 = dense_native_q4_sources(
                snapshot_dir,
                &weight_map,
                &mut shard_cache,
                tensor,
                glm_shape.as_deref(),
            )?;
            let quantization =
                dense_tensor_quantization(&canonical_tensor, &tensor_dtype, &native_q4);
            let runtime_shape = if native_q4.is_some() {
                match glm_shape {
                    Some(shape) => shape,
                    None => logical_shape_for_mlx_source(
                        &tensor_shape,
                        native_q4.as_ref().expect("checked above"),
                    )?,
                }
            } else {
                tensor_shape.clone()
            };
            if let Some(splits) =
                qwen3_next_grouped_projection_splits(config, &canonical_tensor, &runtime_shape)?
            {
                let native_q4 = native_q4.with_context(|| {
                    format!(
                        "Qwen3-Next grouped projection {canonical_tensor} requires its native affine-Q4 scale/bias tensors"
                    )
                })?;
                if native_q4.source_format != DenseQ4SourceFormat::MlxAffine {
                    bail!(
                        "Qwen3-Next grouped projection {canonical_tensor} requires MLX affine Q4 source, found {:?}",
                        native_q4.source_format
                    );
                }
                for split in splits {
                    let mut split_source = native_q4.clone();
                    split_source.source_row_order = Some(split.source_row_order);
                    let split_sources = Some(split_source);
                    let quantization =
                        dense_tensor_quantization(&split.tensor, &tensor_dtype, &split_sources);
                    let TensorQuantization::Q4 {
                        group_size,
                        scale_bias_dtype,
                        ..
                    } = &quantization
                    else {
                        bail!(
                            "Qwen3-Next grouped projection {} did not resolve affine Q4 storage",
                            split.tensor
                        );
                    };
                    let byte_len = dense_q4_layout_with_scale_bias_dtype(
                        &split.shape,
                        *group_size,
                        scale_bias_dtype,
                    )?
                    .total_bytes as u64;
                    runtime_offset = align_to(runtime_offset, TENSOR_ALIGNMENT);
                    dense_tensor_refs.push(DenseTensorRef {
                        tensor: split.tensor,
                        shard: shard.clone(),
                        dtype: tensor_dtype.clone(),
                        shape: split.shape,
                        source_offsets: tensor_source_offsets,
                        runtime_offset,
                        byte_len,
                        quantization,
                        q4_sources: split_sources,
                    });
                    runtime_offset = runtime_offset.saturating_add(byte_len);
                }
                continue;
            }
            let preserves_affine_int8 = native_q4.as_ref().is_some_and(|source| {
                matches!(
                    source.source_format,
                    DenseQ4SourceFormat::ColibriInt8 | DenseQ4SourceFormat::MlxAffine8
                )
            });
            let widens_qwen3_next_a_log = config.is_some_and(QwenModelConfig::is_qwen3_next)
                && canonical_tensor.ends_with(".linear_attn.A_log")
                && tensor_dtype.eq_ignore_ascii_case("BF16")
                && native_q4.is_none();
            let runtime_dtype = if preserves_affine_int8 {
                "BF16".to_string()
            } else if widens_qwen3_next_a_log {
                "F32".to_string()
            } else {
                tensor_dtype
            };
            let byte_len = match &quantization {
                TensorQuantization::None if preserves_affine_int8 => runtime_shape
                    .iter()
                    .try_fold(2u64, |bytes, dimension| {
                        bytes.checked_mul(*dimension as u64)
                    })
                    .context("Colibri int8-to-BF16 runtime tensor byte count overflow")?,
                TensorQuantization::None if widens_qwen3_next_a_log => runtime_shape
                    .iter()
                    .try_fold(4u64, |bytes, dimension| {
                        bytes.checked_mul(*dimension as u64)
                    })
                    .context("Qwen3-Next A_log BF16-to-F32 runtime byte count overflow")?,
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
                TensorQuantization::Gguf { .. } => source_byte_len,
            };
            runtime_offset = align_to(runtime_offset, TENSOR_ALIGNMENT);
            dense_tensor_refs.push(DenseTensorRef {
                tensor: canonical_tensor,
                shard: shard.clone(),
                dtype: runtime_dtype,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct Qwen3NextProjectionSplit {
    tensor: String,
    shape: Vec<usize>,
    source_row_order: Vec<usize>,
}

fn qwen3_next_grouped_projection_splits(
    config: Option<&QwenModelConfig>,
    tensor: &str,
    source_shape: &[usize],
) -> Result<Option<Vec<Qwen3NextProjectionSplit>>> {
    let Some(config) = config.filter(|config| config.is_qwen3_next()) else {
        return Ok(None);
    };
    let Some(base) = tensor
        .strip_suffix("in_proj_qkvz.weight")
        .or_else(|| tensor.strip_suffix("in_proj_ba.weight"))
    else {
        return Ok(None);
    };
    let [source_rows, source_cols] = source_shape else {
        bail!("Qwen3-Next grouped projection {tensor} must be a matrix, found {source_shape:?}");
    };
    if *source_cols != config.hidden_size {
        bail!(
            "Qwen3-Next grouped projection {tensor} input width {source_cols} does not match hidden size {}",
            config.hidden_size
        );
    }
    let linear = config.linear_attention.with_context(|| {
        format!("Qwen3-Next grouped projection {tensor} is missing linear-attention geometry")
    })?;
    let value_heads_per_key = linear
        .num_value_heads
        .checked_div(linear.num_key_heads)
        .filter(|value| *value > 0)
        .context("Qwen3-Next value heads must be divisible by key heads")?;
    let value_width_per_key = value_heads_per_key
        .checked_mul(linear.value_head_dim)
        .context("Qwen3-Next grouped value width overflow")?;

    if tensor.ends_with("in_proj_qkvz.weight") {
        let group_width = linear
            .key_head_dim
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width_per_key.checked_mul(2)?))
            .context("Qwen3-Next QKVZ group width overflow")?;
        let expected_rows = group_width
            .checked_mul(linear.num_key_heads)
            .context("Qwen3-Next QKVZ row count overflow")?;
        if *source_rows != expected_rows {
            bail!(
                "Qwen3-Next grouped projection {tensor} has {source_rows} rows, expected {expected_rows}"
            );
        }
        let mut query_rows = Vec::new();
        let mut key_rows = Vec::new();
        let mut value_rows = Vec::new();
        let mut gate_rows = Vec::new();
        for head in 0..linear.num_key_heads {
            let start = head * group_width;
            let key_start = start + linear.key_head_dim;
            let value_start = key_start + linear.key_head_dim;
            let gate_start = value_start + value_width_per_key;
            query_rows.extend(start..key_start);
            key_rows.extend(key_start..value_start);
            value_rows.extend(value_start..gate_start);
            gate_rows.extend(gate_start..gate_start + value_width_per_key);
        }
        let mut qkv_rows = query_rows;
        qkv_rows.extend(key_rows);
        qkv_rows.extend(value_rows);
        return Ok(Some(vec![
            Qwen3NextProjectionSplit {
                tensor: format!("{base}in_proj_qkv.weight"),
                shape: vec![qkv_rows.len(), *source_cols],
                source_row_order: qkv_rows,
            },
            Qwen3NextProjectionSplit {
                tensor: format!("{base}in_proj_z.weight"),
                shape: vec![gate_rows.len(), *source_cols],
                source_row_order: gate_rows,
            },
        ]));
    }

    let group_width = value_heads_per_key
        .checked_mul(2)
        .context("Qwen3-Next BA group width overflow")?;
    let expected_rows = group_width
        .checked_mul(linear.num_key_heads)
        .context("Qwen3-Next BA row count overflow")?;
    if *source_rows != expected_rows {
        bail!(
            "Qwen3-Next grouped projection {tensor} has {source_rows} rows, expected {expected_rows}"
        );
    }
    let mut beta_rows = Vec::new();
    let mut alpha_rows = Vec::new();
    for head in 0..linear.num_key_heads {
        let start = head * group_width;
        let alpha_start = start + value_heads_per_key;
        beta_rows.extend(start..alpha_start);
        alpha_rows.extend(alpha_start..alpha_start + value_heads_per_key);
    }
    Ok(Some(vec![
        Qwen3NextProjectionSplit {
            tensor: format!("{base}in_proj_b.weight"),
            shape: vec![beta_rows.len(), *source_cols],
            source_row_order: beta_rows,
        },
        Qwen3NextProjectionSplit {
            tensor: format!("{base}in_proj_a.weight"),
            shape: vec![alpha_rows.len(), *source_cols],
            source_row_order: alpha_rows,
        },
    ]))
}

fn tensor_layer(name: &str) -> Option<usize> {
    let parts = name.split('.').collect::<Vec<_>>();
    parts
        .windows(2)
        .find(|part| part[0] == "layers")
        .and_then(|part| part[1].parse().ok())
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
#[path = "cache_parity_tests.rs"]
mod parity_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_next_grouped_projections_split_into_canonical_runtime_rows() {
        let config: QwenModelConfig = serde_json::from_value(serde_json::json!({
            "model_type": "qwen3_next",
            "architectures": ["Qwen3NextForCausalLM"],
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "linear_key_head_dim": 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 3
        }))
        .unwrap();

        let qkvz = qwen3_next_grouped_projection_splits(
            Some(&config),
            "model.layers.0.linear_attn.in_proj_qkvz.weight",
            &[32, 8],
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            qkvz[0].tensor,
            "model.layers.0.linear_attn.in_proj_qkv.weight"
        );
        assert_eq!(qkvz[0].shape, vec![20, 8]);
        assert_eq!(
            qkvz[0].source_row_order,
            vec![
                0, 1, 16, 17, // query
                2, 3, 18, 19, // key
                4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25 // value
            ]
        );
        assert_eq!(
            qkvz[1].tensor,
            "model.layers.0.linear_attn.in_proj_z.weight"
        );
        assert_eq!(
            qkvz[1].source_row_order,
            vec![10, 11, 12, 13, 14, 15, 26, 27, 28, 29, 30, 31]
        );

        let ba = qwen3_next_grouped_projection_splits(
            Some(&config),
            "model.layers.0.linear_attn.in_proj_ba.weight",
            &[8, 8],
        )
        .unwrap()
        .unwrap();
        assert_eq!(ba[0].tensor, "model.layers.0.linear_attn.in_proj_b.weight");
        assert_eq!(ba[0].source_row_order, vec![0, 1, 4, 5]);
        assert_eq!(ba[1].tensor, "model.layers.0.linear_attn.in_proj_a.weight");
        assert_eq!(ba[1].source_row_order, vec![2, 3, 6, 7]);
    }

    #[test]
    fn cache_owner_declares_required_huggingface_artifacts() {
        let files = expected_hf_files();
        assert!(files.contains(&OsString::from("config.json")));
        assert!(files.contains(&OsString::from("tokenizer_config.json")));
        assert!(files.contains(&OsString::from("chat_template.jinja")));
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
