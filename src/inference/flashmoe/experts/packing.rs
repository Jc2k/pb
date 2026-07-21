use super::*;

#[derive(Debug, Clone, Copy)]
pub(in crate::inference::flashmoe) struct ExpertPackingPolicy<'a> {
    model: &'a str,
    experts_dir: &'a Path,
    quantization: ExpertQuantization,
}

impl<'a> ExpertPackingPolicy<'a> {
    pub(in crate::inference::flashmoe) fn new(
        model: &'a str,
        experts_dir: &'a Path,
        quantization: ExpertQuantization,
    ) -> Self {
        Self {
            model,
            experts_dir,
            quantization,
        }
    }
}

pub(in crate::inference::flashmoe) fn pack_expert_tensors(
    snapshot_dir: &Path,
    policy: ExpertPackingPolicy<'_>,
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

    let deleted_temps = cleanup_stale_expert_temp_files(policy.experts_dir)?;
    if deleted_temps > 0 {
        eprintln!(
            "deleted {deleted_temps} stale temporary expert pack file(s) from {}",
            policy.experts_dir.display()
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
            policy,
            layer,
            layer_index + 1,
            aggregate_layers,
            &tensors,
            config,
        )?;
    }
    for (layer, experts) in by_layer {
        pack_direct_expert_layer(
            snapshot_dir,
            policy,
            layer,
            experts,
            config,
            &mut shard_cache,
        )?;
    }
    Ok(())
}

pub(in crate::inference::flashmoe) fn fixed_dense_expert_slot_spec_for_pack(
    policy: ExpertPackingPolicy<'_>,
    config: Option<&QwenModelConfig>,
) -> Result<Option<FixedDenseExpertSlotSpec>> {
    let dtype = match policy.quantization {
        ExpertQuantization::FourBitProduction => return Ok(None),
        ExpertQuantization::Bf16 => DenseExpertDtype::Bf16,
        ExpertQuantization::F16 => DenseExpertDtype::F16,
    };
    let config = config.context("Qwen config is required for fixed dense expert packing")?;
    let layout = QwenMoeModelLayout::from_config(policy.model, config)?;
    FixedDenseExpertSlotSpec::from_model_layout(&layout, dtype).map(Some)
}

pub(in crate::inference::flashmoe) fn pack_direct_expert_layer(
    snapshot_dir: &Path,
    policy: ExpertPackingPolicy<'_>,
    layer: usize,
    experts: BTreeMap<usize, Vec<&ExpertTensorRef>>,
    config: Option<&QwenModelConfig>,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
) -> Result<()> {
    let fixed_dense = fixed_dense_expert_slot_spec_for_pack(policy, config)?;
    let fixed_native_q4 = if policy.quantization == ExpertQuantization::FourBitProduction
        && experts.values().flatten().all(|tensor| {
            tensor
                .q4_sources
                .as_ref()
                .is_some_and(|source| source.source_format != DenseQ4SourceFormat::MlxAffine8)
        }) {
        let config = config.context("model config is required for native Q4 expert packing")?;
        let layout = QwenMoeModelLayout::from_config(policy.model, config)?;
        let sources = experts
            .values()
            .flatten()
            .filter_map(|tensor| tensor.q4_sources.as_ref());
        let (mxfp4, total) = sources.fold((0usize, 0usize), |(mxfp4, total), source| {
            (
                mxfp4 + usize::from(source.source_format == DenseQ4SourceFormat::MlxMxfp4),
                total + 1,
            )
        });
        if mxfp4 != 0 && mxfp4 != total {
            bail!("direct expert tensors must be all MLX MXFP4 or all affine Q4");
        }
        Some(if mxfp4 == total && total != 0 {
            FixedQ4ExpertSlotSpec::mxfp4_from_model_layout(&layout)?
        } else {
            FixedQ4ExpertSlotSpec::from_model_layout(&layout)?
        })
    } else {
        None
    };
    let mut expected = Vec::with_capacity(experts.len());
    for (expert, tensors) in &experts {
        validate_expert_tensor_group(layer, *expert, tensors, config)?;
        expected.push(match (fixed_dense, fixed_native_q4) {
            (Some(spec), _) => expected_fixed_dense_expert_pack(
                snapshot_dir,
                shard_cache,
                layer,
                *expert,
                tensors,
                spec,
            )?,
            (None, Some(spec)) => expected_fixed_native_q4_direct_expert_pack(
                snapshot_dir,
                shard_cache,
                layer,
                *expert,
                tensors,
                spec,
            )?,
            (None, None) => expected_expert_pack(snapshot_dir, shard_cache, *expert, tensors)?,
        });
    }
    let expert_count = layer_expert_count(config, &experts);
    rewrite_expert_layer_pack(
        policy.experts_dir,
        layer,
        expert_count,
        match (fixed_dense, fixed_native_q4) {
            (Some(spec), _) => ExpertLayerStorageFormat::FixedDense(spec),
            (None, Some(spec)) => ExpertLayerStorageFormat::FixedQ4(spec),
            (None, None) => ExpertLayerStorageFormat::Pbq4Import,
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
                fixed_native_q4,
            )
        },
    )?;
    Ok(())
}

pub(in crate::inference::flashmoe) fn layer_expert_count(
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

pub(in crate::inference::flashmoe) fn build_direct_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    tensors: &[&ExpertTensorRef],
    fixed_dense: Option<FixedDenseExpertSlotSpec>,
    fixed_native_q4: Option<FixedQ4ExpertSlotSpec>,
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
    if let Some(spec) = fixed_native_q4 {
        let inputs = ordered_direct_expert_tensors(tensors)?
            .into_iter()
            .map(|tensor| {
                native_q4_expert_record_input(
                    snapshot_dir,
                    shard_cache,
                    tensor,
                    tensor.tensor.clone(),
                    tensor.shape.clone(),
                    0,
                    tensor.shape.iter().product(),
                    spec.encoding,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        return build_fixed_native_q4_expert_pack(layer, expert, spec, inputs);
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

fn ordered_direct_expert_tensors<'a>(
    tensors: &'a [&'a ExpertTensorRef],
) -> Result<[&'a ExpertTensorRef; 3]> {
    let find = |suffix: &str| {
        tensors
            .iter()
            .copied()
            .find(|tensor| tensor.tensor.ends_with(suffix))
            .with_context(|| format!("direct expert is missing {suffix}"))
    };
    Ok([
        find("gate_proj.weight")?,
        find("up_proj.weight")?,
        find("down_proj.weight")?,
    ])
}

fn expected_fixed_native_q4_direct_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    _layer: usize,
    expert: usize,
    tensors: &[&ExpertTensorRef],
    spec: FixedQ4ExpertSlotSpec,
) -> Result<ExpectedExpertPack> {
    let records = ordered_direct_expert_tensors(tensors)?
        .into_iter()
        .map(|tensor| {
            expected_native_q4_expert_record(
                snapshot_dir,
                shard_cache,
                tensor,
                tensor.tensor.clone(),
                tensor.shape.clone(),
                0,
                tensor.shape.iter().product(),
                spec.encoding,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ExpectedExpertPack {
        expert,
        packed_bytes: spec.layout.expert_bytes as u64,
        records,
    })
}

pub(in crate::inference::flashmoe) fn pack_aggregate_expert_layer(
    snapshot_dir: &Path,
    policy: ExpertPackingPolicy<'_>,
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
    let fixed_dense = fixed_dense_expert_slot_spec_for_pack(policy, Some(config))?;
    let fixed_native_q4 = fixed_native_q4_aggregate_spec(&aggregate_tensors, down, layout)?;
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
                fixed_native_q4
                    .map(|spec| spec.encoding)
                    .unwrap_or(FixedQ4ExpertEncoding::AffineBf16),
            )?,
        };
        let packed_bytes = match (fixed_dense, fixed_native_q4) {
            (Some(spec), _) => spec.expert_bytes as u64,
            (None, Some(spec)) => spec.layout.expert_bytes as u64,
            (None, None) => pbq4_expert_pack_wire_size(&records)?,
        };
        expected.push(ExpectedExpertPack {
            expert,
            packed_bytes,
            records,
        });
    }
    let skipped = rewrite_expert_layer_pack(
        policy.experts_dir,
        layer,
        layout.experts,
        match (fixed_dense, fixed_native_q4) {
            (Some(spec), _) => ExpertLayerStorageFormat::FixedDense(spec),
            (None, Some(spec)) => ExpertLayerStorageFormat::FixedQ4(spec),
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
                    fixed_native_q4,
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

pub(in crate::inference::flashmoe) fn build_aggregate_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
    fixed_native_q4: Option<FixedQ4ExpertSlotSpec>,
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
            fixed_native_q4,
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
pub(in crate::inference::flashmoe) fn build_fixed_dense_aggregate_expert_pack(
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
pub(in crate::inference::flashmoe) fn expected_fixed_dense_aggregate_expert_records(
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

pub(in crate::inference::flashmoe) fn expected_aggregate_expert_records(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
    encoding: FixedQ4ExpertEncoding,
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
            encoding,
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

pub(in crate::inference::flashmoe) fn build_native_q4_aggregate_expert_pack(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
    fixed_native_q4: Option<FixedQ4ExpertSlotSpec>,
) -> Result<(Vec<u8>, ExpertPackMetadata)> {
    let encoding = fixed_native_q4
        .map(|spec| spec.encoding)
        .unwrap_or(FixedQ4ExpertEncoding::AffineBf16);
    let inputs = vec![
        native_q4_expert_record_input(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.gate.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.gate.start(expert)?,
            layout.single_projection_values,
            encoding,
        )?,
        native_q4_expert_record_input(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.up.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.up.start(expert)?,
            layout.single_projection_values,
            encoding,
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
            encoding,
        )?,
    ];
    if let Some(spec) = fixed_native_q4 {
        return build_fixed_native_q4_expert_pack(layer, expert, spec, inputs);
    }
    build_native_q4_expert_pack(layer, expert, inputs)
}

pub(in crate::inference::flashmoe) fn expected_native_q4_aggregate_expert_records(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    layer: usize,
    expert: usize,
    aggregate_tensors: &AggregateExpertTensors<'_, ExpertTensorRef>,
    down: &ExpertTensorRef,
    layout: AggregateExpertLayout,
    encoding: FixedQ4ExpertEncoding,
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
            encoding,
        )?,
        expected_native_q4_expert_record(
            snapshot_dir,
            shard_cache,
            aggregate_tensors.up.tensor,
            format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
            vec![layout.intermediate, layout.hidden],
            aggregate_tensors.up.start(expert)?,
            layout.single_projection_values,
            encoding,
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
            encoding,
        )?,
    ])
}

pub(in crate::inference::flashmoe) fn expected_native_q4_expert_record(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    source: &ExpertTensorRef,
    tensor: String,
    shape: Vec<usize>,
    element_offset: usize,
    element_count: usize,
    encoding: FixedQ4ExpertEncoding,
) -> Result<ExpectedExpertPackRecord> {
    if source
        .q4_sources
        .as_ref()
        .is_some_and(|q4| q4.source_format == DenseQ4SourceFormat::MlxMxfp4)
    {
        let source_slice = mlx_mxfp4_expert_source_slice(
            snapshot_dir,
            shard_cache,
            source,
            &shape,
            element_offset,
            element_count,
        )?;
        if encoding == FixedQ4ExpertEncoding::MlxMxfp4 {
            return Ok(ExpectedExpertPackRecord {
                tensor,
                dtype: source
                    .dtype
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                shape,
                source_offsets: source_slice.packed_offsets,
                source_hash: sha256_hex_parts(&[&source_slice.weights, &source_slice.scales]),
                packed_bytes: source_slice.weights.len() as u64,
                groups: source_slice.scales.len(),
                group_size: source_slice.source_group_size,
                scale_bias_dtype: EXPERT_SCALE_DTYPE_E8M0.to_string(),
            });
        }
        let layout = dense_q4_layout_with_scale_bias_dtype(
            &shape,
            GROUP_SIZE,
            EXPERT_SCALE_BIAS_DTYPE_BF16,
        )?;
        return Ok(ExpectedExpertPackRecord {
            tensor,
            dtype: source
                .dtype
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            shape,
            source_offsets: source_slice.packed_offsets,
            source_hash: sha256_hex_parts(&[&source_slice.weights, &source_slice.scales]),
            packed_bytes: layout.packed_bytes as u64,
            groups: layout.rows * layout.groups_per_row,
            group_size: GROUP_SIZE,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        });
    }
    let input = native_q4_expert_record_input(
        snapshot_dir,
        shard_cache,
        source,
        tensor,
        shape,
        element_offset,
        element_count,
        encoding,
    )?;
    expected_native_q4_expert_record_from_input(input)
}

pub(in crate::inference::flashmoe) fn expected_expert_pack_record(
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

pub(in crate::inference::flashmoe) fn expected_expert_pack(
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

pub(in crate::inference::flashmoe) fn expected_fixed_dense_expert_pack(
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

pub(in crate::inference::flashmoe) fn decode_expert_tensor_range(
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
pub(in crate::inference::flashmoe) fn fixed_dense_expert_record_input(
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

pub(in crate::inference::flashmoe) fn expert_tensor_source_fingerprint(
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

struct MlxMxfp4ExpertSourceSlice {
    weights: Vec<u8>,
    scales: Vec<u8>,
    packed_offsets: [u64; 2],
    source_group_size: usize,
}

pub(in crate::inference::flashmoe) fn validate_mxfp4_e8m0_scales(
    tensor: &str,
    scales: &[u8],
) -> Result<()> {
    if let Some((index, _)) = scales
        .iter()
        .enumerate()
        .find(|(_, scale)| **scale == u8::MAX)
    {
        bail!(
            "MLX MXFP4 expert tensor {tensor} contains non-finite E8M0 scale 0xff at byte {index}"
        );
    }
    Ok(())
}

fn mlx_mxfp4_expert_source_slice(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    source: &ExpertTensorRef,
    shape: &[usize],
    element_offset: usize,
    element_count: usize,
) -> Result<MlxMxfp4ExpertSourceSlice> {
    let q4_sources = source.q4_sources.as_ref().with_context(|| {
        format!(
            "MLX MXFP4 expert tensor {} is missing source metadata",
            source.tensor
        )
    })?;
    if q4_sources.source_format != DenseQ4SourceFormat::MlxMxfp4 {
        bail!("expert tensor {} is not MLX MXFP4", source.tensor);
    }
    let source_cols = source.shape.last().copied().unwrap_or(0);
    let source_group_size = q4_sources.source_group_size.with_context(|| {
        format!(
            "MLX MXFP4 expert tensor {} is missing source group size",
            source.tensor
        )
    })?;
    if source_cols == 0
        || !element_offset.is_multiple_of(source_cols)
        || !element_count.is_multiple_of(source_cols)
    {
        bail!(
            "MLX MXFP4 expert tensor {} slice {element_offset}..{} is not row-aligned to {source_cols} columns",
            source.tensor,
            element_offset.saturating_add(element_count)
        );
    }
    let slice_rows = element_count / source_cols;
    let expected_rows =
        shape[..shape.len().saturating_sub(1)]
            .iter()
            .try_fold(1usize, |rows, dimension| {
                rows.checked_mul(*dimension)
                    .context("MLX MXFP4 expert slice row count overflow")
            })?;
    if shape.last().copied() != Some(source_cols) || expected_rows != slice_rows {
        bail!(
            "MLX MXFP4 expert tensor {} slice shape {shape:?} does not match {slice_rows} rows x {source_cols} columns",
            source.tensor
        );
    }
    let source_offsets = source
        .source_offsets
        .with_context(|| format!("expert tensor {} is missing source offsets", source.tensor))?;
    let row_start = element_offset / source_cols;
    let source_row_bytes = source_cols.div_ceil(2);
    let source_groups_per_row = source_cols.div_ceil(source_group_size);
    let packed_offset = row_start
        .checked_mul(source_row_bytes)
        .context("MLX MXFP4 expert packed offset overflow")?;
    let packed_bytes = slice_rows
        .checked_mul(source_row_bytes)
        .context("MLX MXFP4 expert packed byte count overflow")?;
    let scale_offset = row_start
        .checked_mul(source_groups_per_row)
        .context("MLX MXFP4 expert scale offset overflow")?;
    let scale_bytes = slice_rows
        .checked_mul(source_groups_per_row)
        .context("MLX MXFP4 expert scale byte count overflow")?;
    let (weights, packed_offsets) = read_safetensor_source_byte_range(
        snapshot_dir,
        shard_cache,
        &source.shard,
        source_offsets,
        packed_offset,
        packed_bytes,
    )?;
    let (scales, _) = read_safetensor_source_byte_range(
        snapshot_dir,
        shard_cache,
        &q4_sources.scales_shard,
        q4_sources.scales_offsets,
        scale_offset,
        scale_bytes,
    )?;
    validate_mxfp4_e8m0_scales(&source.tensor, &scales)?;
    Ok(MlxMxfp4ExpertSourceSlice {
        weights,
        scales,
        packed_offsets,
        source_group_size,
    })
}

pub(in crate::inference::flashmoe) fn with_expert_tensor_raw_range<R>(
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

pub(in crate::inference::flashmoe) fn native_q4_expert_record_input(
    snapshot_dir: &Path,
    shard_cache: &mut BTreeMap<String, (memmap2::Mmap, SafetensorShard)>,
    source: &ExpertTensorRef,
    tensor: String,
    shape: Vec<usize>,
    element_offset: usize,
    element_count: usize,
    encoding: FixedQ4ExpertEncoding,
) -> Result<NativeQ4ExpertRecordInput> {
    let q4_sources = source
        .q4_sources
        .as_ref()
        .with_context(|| format!("expert tensor {} is not native MLX Q4", source.tensor))?;
    if encoding == FixedQ4ExpertEncoding::MlxMxfp4
        && q4_sources.source_format != DenseQ4SourceFormat::MlxMxfp4
    {
        bail!(
            "native MXFP4 expert storage requires MLX MXFP4 source tensor {}, found {:?}",
            source.tensor,
            q4_sources.source_format
        );
    }
    if matches!(
        q4_sources.source_format,
        DenseQ4SourceFormat::ColibriInt4 | DenseQ4SourceFormat::ColibriInt8
    ) {
        let bits = if q4_sources.source_format == DenseQ4SourceFormat::ColibriInt4 {
            4
        } else {
            8
        };
        if element_offset != 0 || element_count != shape.iter().product::<usize>() {
            bail!(
                "Colibri direct expert tensor {} must be imported as one complete component",
                source.tensor
            );
        }
        let source_offsets = source.source_offsets.with_context(|| {
            format!("expert tensor {} is missing source offsets", source.tensor)
        })?;
        let source_weight_bytes = usize::try_from(source_offsets[1] - source_offsets[0])
            .context("Colibri expert weight byte count exceeds usize")?;
        let (source_weights, packed_offsets) = read_safetensor_source_byte_range(
            snapshot_dir,
            shard_cache,
            &source.shard,
            source_offsets,
            0,
            source_weight_bytes,
        )?;
        let source_scale_bytes =
            usize::try_from(q4_sources.scales_offsets[1] - q4_sources.scales_offsets[0])
                .context("Colibri expert scale byte count exceeds usize")?;
        let (source_scales, _) = read_safetensor_source_byte_range(
            snapshot_dir,
            shard_cache,
            &q4_sources.scales_shard,
            q4_sources.scales_offsets,
            0,
            source_scale_bytes,
        )?;
        let layout = dense_q4_layout_with_scale_bias_dtype(
            &shape,
            GROUP_SIZE,
            EXPERT_SCALE_BIAS_DTYPE_BF16,
        )?;
        let mut converted = Vec::with_capacity(layout.total_bytes);
        write_colibri_q4_affine_tensor(
            &mut converted,
            &tensor,
            &source_weights,
            &source_scales,
            bits,
            q4_sources.source_group_size.with_context(|| {
                format!(
                    "Colibri expert tensor {} is missing source group size",
                    source.tensor
                )
            })?,
            layout,
        )?;
        if converted.len() != layout.total_bytes {
            bail!(
                "Colibri expert tensor {tensor} converted to {} bytes, expected {}",
                converted.len(),
                layout.total_bytes
            );
        }
        let scale_start = layout.packed_bytes;
        let bias_start = scale_start + layout.scales_bytes;
        let packed = converted[..scale_start].to_vec();
        let scale_bytes = converted[scale_start..bias_start].to_vec();
        let bias_bytes = converted[bias_start..].to_vec();
        return Ok(NativeQ4ExpertRecordInput {
            tensor,
            dtype: source
                .dtype
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            shape,
            source_offsets: packed_offsets,
            source_hash: Some(sha256_hex_parts(&[&source_weights, &source_scales])),
            packed,
            scale_bytes,
            bias_bytes,
            groups: layout.rows * layout.groups_per_row,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        });
    }
    if q4_sources.source_format == DenseQ4SourceFormat::MlxMxfp4 {
        let source_slice = mlx_mxfp4_expert_source_slice(
            snapshot_dir,
            shard_cache,
            source,
            &shape,
            element_offset,
            element_count,
        )?;
        if encoding == FixedQ4ExpertEncoding::MlxMxfp4 {
            return Ok(NativeQ4ExpertRecordInput {
                tensor,
                dtype: source
                    .dtype
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                shape,
                source_offsets: source_slice.packed_offsets,
                source_hash: Some(sha256_hex_parts(&[
                    &source_slice.weights,
                    &source_slice.scales,
                ])),
                packed: source_slice.weights,
                scale_bytes: source_slice.scales,
                bias_bytes: Vec::new(),
                groups: element_count / source_slice.source_group_size,
                scale_bias_dtype: EXPERT_SCALE_DTYPE_E8M0.to_string(),
            });
        }
        let layout = dense_q4_layout_with_scale_bias_dtype(
            &shape,
            GROUP_SIZE,
            EXPERT_SCALE_BIAS_DTYPE_BF16,
        )?;
        let mut converted = Vec::with_capacity(layout.total_bytes);
        write_mlx_mxfp4_affine_tensor(
            &mut converted,
            &tensor,
            &source_slice.weights,
            &source_slice.scales,
            source_slice.source_group_size,
            layout,
        )?;
        if converted.len() != layout.total_bytes {
            bail!(
                "MLX MXFP4 expert tensor {tensor} converted to {} bytes, expected {}",
                converted.len(),
                layout.total_bytes
            );
        }
        let scale_start = layout.packed_bytes;
        let bias_start = scale_start + layout.scales_bytes;
        return Ok(NativeQ4ExpertRecordInput {
            tensor,
            dtype: source
                .dtype
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            shape,
            source_offsets: source_slice.packed_offsets,
            source_hash: Some(sha256_hex_parts(&[
                &source_slice.weights,
                &source_slice.scales,
            ])),
            packed: converted[..scale_start].to_vec(),
            scale_bytes: converted[scale_start..bias_start].to_vec(),
            bias_bytes: converted[bias_start..].to_vec(),
            groups: layout.rows * layout.groups_per_row,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        });
    }
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

pub(in crate::inference::flashmoe) fn read_safetensor_source_byte_range(
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

pub(in crate::inference::flashmoe) fn validate_expert_tensor_group(
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

pub(in crate::inference::flashmoe) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

pub(in crate::inference::flashmoe) fn sha256_hex_parts(parts: &[&[u8]]) -> String {
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

#[cfg(test)]
pub(in crate::inference::flashmoe) fn read_pbq4_expert_records(
    root: &Path,
    layer: usize,
    expert: usize,
) -> Result<Vec<PackedExpertTensor>> {
    let store = ExpertSlotStore::open(root.to_path_buf())?;
    let raw = store
        .read_many_raw(layer, &[expert])?
        .pop()
        .with_context(|| format!("expert layer {layer} returned no expert {expert}"))?;
    let ExpertRawPayload::Pbq4(bytes) = raw.payload else {
        bail!(
            "expert layer {layer} expert {expert} is fixed-slot execution storage, not PBQ4 import data"
        );
    };
    parse_pbq4_expert_pack(&bytes, Some(&raw.metadata))
}

#[cfg(test)]
pub(in crate::inference::flashmoe) fn packed_expert_record_suffix<'a>(
    records: &'a [PackedExpertTensor],
    suffix: &str,
) -> Option<&'a PackedExpertTensor> {
    records.iter().find(|record| record.name.ends_with(suffix))
}

#[cfg(test)]
pub(in crate::inference::flashmoe) fn project_packed_expert_record(
    record: &PackedExpertTensor,
    input: &[f32],
    output_width: usize,
) -> Result<Vec<f32>> {
    let payload = record
        .matvec_payload(input, output_width)
        .with_context(|| format!("PBQ4 record {} has no compatible Q4 payload", record.name))?;
    q4_fma_matvec_with_group_size(
        payload.packed,
        &input[..payload.cols],
        payload.scales,
        payload.biases,
        payload.rows,
        payload.cols,
        payload.group_size,
    )
    .with_context(|| format!("failed to project PBQ4 import record {}", record.name))
}
