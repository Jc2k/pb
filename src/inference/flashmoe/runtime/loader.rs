use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeLoadOptions {
    pub metal_working_set_limit_bytes: Option<usize>,
    pub session_cache: crate::config::ResolvedSessionCacheConfig,
    pub memory_sessions: usize,
}

impl Default for FlashMoeLoadOptions {
    fn default() -> Self {
        Self {
            metal_working_set_limit_bytes: None,
            session_cache: crate::config::ResolvedSessionCacheConfig {
                enabled: true,
                root: dirs::cache_dir().map(|root| root.join("pb")),
                max_bytes: crate::config::DEFAULT_SESSION_CACHE_MAX_BYTES,
            },
            memory_sessions: crate::config::DEFAULT_FLASHMOE_MEMORY_SESSIONS,
        }
    }
}

pub fn load(plan: &FlashMoePlan) -> Result<FlashMoeEngine> {
    load_with_options(plan, FlashMoeLoadOptions::default())
}

pub fn load_with_progress<F>(plan: &FlashMoePlan, progress: F) -> Result<FlashMoeEngine>
where
    F: FnMut(&'static str, Duration),
{
    load_with_options_and_progress(plan, FlashMoeLoadOptions::default(), progress)
}

pub fn load_with_options(
    plan: &FlashMoePlan,
    options: FlashMoeLoadOptions,
) -> Result<FlashMoeEngine> {
    load_with_options_and_progress(plan, options, |_, _| {})
}

pub fn load_with_options_and_progress<F>(
    plan: &FlashMoePlan,
    options: FlashMoeLoadOptions,
    mut progress: F,
) -> Result<FlashMoeEngine>
where
    F: FnMut(&'static str, Duration),
{
    let mut phase_started = Instant::now();
    let status = plan.cache_status()?;
    progress("cache_status", phase_started.elapsed());
    if !status.ready {
        bail!(
            "Flash-MoE cache is not ready for {}. Missing: {}. Found {} expert files totaling {} bytes. Run `pb pull {}` on ARM macOS to download and prepare the FlashMoe cache.",
            plan.model,
            format_missing(&status.missing),
            status.expert_files,
            status.expert_bytes,
            plan.model
        );
    }
    phase_started = Instant::now();
    let deepseek_config = if is_deepseek_v4_flash(&plan.model) {
        Some(DeepSeekV4Config::from_file(&plan.model_config)?)
    } else {
        None
    };
    let config = match &deepseek_config {
        Some(config) => config.shared_runtime_config(),
        None => QwenModelConfig::from_file(&plan.model_config)?,
    };
    progress("config", phase_started.elapsed());
    phase_started = Instant::now();
    let routing_policy = plan.routing_policy.resolve(&plan.model, &config)?;
    progress("routing_policy", phase_started.elapsed());
    phase_started = Instant::now();
    let model_layout = QwenMoeModelLayout::from_config(&plan.model, &config)?
        .with_scheduled_active_experts(routing_policy.active_experts)?;
    progress("model_layout", phase_started.elapsed());
    phase_started = Instant::now();
    let resolved_experts = ExpertSlotStore::resolve_from_metadata(
        plan.experts_dir.clone(),
        &model_layout,
        plan.quantization,
    )?;
    if resolved_experts.upgraded_pbq4_layers > 0 {
        tracing::info!(
            model = %plan.model,
            layers = resolved_experts.upgraded_pbq4_layers,
            "upgraded PBQ4 expert cache layers to fixed Q4 slots"
        );
    }
    progress("expert_cache_format", phase_started.elapsed());
    phase_started = Instant::now();
    let dense = DenseStore::open(
        plan.non_expert_weights.clone(),
        plan.tensor_manifest.clone(),
    )?;
    progress("dense_store", phase_started.elapsed());
    phase_started = Instant::now();
    let deepseek_graph = deepseek_config
        .map(|config| DeepSeekV4ExecutionGraph::from_registry(config, dense.registry(), dense.len))
        .transpose()?
        .map(Arc::new);
    if deepseek_graph.is_none() {
        validate_required_tensor_manifest(&config, dense.registry())?;
    }
    progress("manifest_validation", phase_started.elapsed());
    phase_started = Instant::now();
    let runtime = if deepseek_graph.is_some() {
        DenseTransformerRuntime::new(&config)
    } else {
        DenseTransformerRuntime::from_registry(&config, dense.registry())?
    };
    let attention_layers = if deepseek_graph.is_some() {
        vec![super::super::model_family::QwenMoeLayerKind::FullAttention; config.num_hidden_layers]
    } else {
        runtime.resolved_attention_layers()?
    };
    progress("runtime_layout", phase_started.elapsed());
    phase_started = Instant::now();
    let linear_attention_weights = if deepseek_graph.is_some() {
        LinearAttentionWeightTable::empty(config.num_hidden_layers)
    } else {
        dense.resolve_linear_attention_weight_table(
            &runtime.linear_attention,
            config.hidden_size,
            model_layout.experts_per_layer,
        )?
    };
    progress("linear_attention_weights", phase_started.elapsed());
    phase_started = Instant::now();
    let shared_expert_weights = if deepseek_graph.is_some() {
        SharedExpertWeightTable::none(config.num_hidden_layers)
    } else {
        dense.resolve_shared_expert_weight_table_from(
            config.num_hidden_layers,
            config.hidden_size,
            config.shared_experts(),
            config.shared_expert_intermediate_size(),
            config.first_sparse_layer(),
            config.glm.is_none(),
        )?
    };
    progress("shared_expert_weights", phase_started.elapsed());
    phase_started = Instant::now();
    let dense_layout = if deepseek_graph.is_some() {
        // DeepSeek's typed graph validates its exact F16/F32/I32/Q8 GGUF
        // mixture above. This legacy field is not consulted by its execution
        // implementation.
        ResidentDenseLayout::F16
    } else {
        dense.registry().resolve_resident_dense_layout()?
    };
    if matches!(
        model_layout.family,
        QwenMoeFamily::Qwen35A17B
            | QwenMoeFamily::Qwen3NextMoe
            | QwenMoeFamily::Qwen3Moe
            | QwenMoeFamily::Qwen3VlMoe
            | QwenMoeFamily::Glm52
    ) && dense_layout == ResidentDenseLayout::Q4
    {
        validate_qwen_q4_graph_bindings(
            model_layout.family,
            &config,
            &runtime,
            dense.registry(),
            dense.len,
        )?;
    }
    progress("dense_graph_bindings", phase_started.elapsed());
    phase_started = Instant::now();
    let input_adapter_executor =
        FlashMoeInputAdapterExecutor::from_plan(model_layout.family, plan, &config)?;
    let input_adapter = input_adapter_executor.capability()?;
    progress("vision_encoder", phase_started.elapsed());
    phase_started = Instant::now();
    let metal = MetalExecutionFacade::new(plan, &config, &runtime, &dense)?;
    if let Some(limit) = options.metal_working_set_limit_bytes {
        metal.set_working_set_limit_bytes(limit)?;
    }
    progress("metal_executor", phase_started.elapsed());
    phase_started = Instant::now();
    let experts = resolved_experts.store;
    let expert_storage = resolved_experts.descriptor;
    progress("expert_store", phase_started.elapsed());
    phase_started = Instant::now();
    let attention_math = if model_layout.family == QwenMoeFamily::DeepSeekV4Flash {
        FlashMoeAttentionMathCapability::DeepSeekV4HyperconnectionCompressedAttentionMetal
    } else if model_layout.family == QwenMoeFamily::Glm52
        && runtime.mla_attention.iter().all(|layout| {
            matches!(
                layout.map(|layout| layout.kv_projection),
                Some(MlaKvProjectionLayout::AbsorbedMultiLinear)
            )
        })
    {
        FlashMoeAttentionMathCapability::GlmMlaMetalQ4AbsorbedAttention
    } else if model_layout.family == QwenMoeFamily::Glm52 {
        FlashMoeAttentionMathCapability::GlmMlaCpuWeightAbsorption
    } else {
        FlashMoeAttentionMathCapability::QwenFullAttentionCpuKv
    };
    let resource_snapshot = metal.resource_snapshot();
    let expert_access = resolve_expert_access(
        model_layout.family,
        expert_storage,
        resource_snapshot.as_ref(),
    )?;
    tracing::info!(
        model = %plan.model,
        implementation = ?expert_access,
        expert_bytes = expert_storage.total_expert_bytes()?,
        working_set_limit_bytes = resource_snapshot
            .as_ref()
            .map_or(0, |snapshot| snapshot.working_set_limit_bytes),
        current_allocated_bytes = resource_snapshot
            .as_ref()
            .map_or(0, |snapshot| snapshot.current_allocated_bytes),
        "resolved FlashMoe expert access implementation"
    );
    let capability_plan = FlashMoeCapabilityPlan::resolve_with_attention_math_and_expert_access(
        &model_layout,
        input_adapter,
        dense_layout,
        expert_storage,
        &attention_layers,
        attention_math,
        expert_access,
        Some(metal.runtime_capabilities()),
    )?;
    let qwen_prefill_graph = capability_plan.qwen_prefill_graph;
    tracing::info!(
        model = %plan.model,
        implementation = qwen_prefill_graph.as_str(),
        dense_layout = dense_layout.as_str(),
        expert_layout = ?expert_storage.layout,
        "prepared FlashMoe Qwen prefill graph"
    );
    let scheduled_graph = FlashMoeScheduledGraph::from_capabilities(&capability_plan)?;
    let scheduler =
        FlashMoeExecutionScheduler::new_with_resident_binding(scheduled_graph, experts, |bytes| {
            metal.prepare_resident_expert_backing(bytes)
        })?;
    progress("capability_graph", phase_started.elapsed());
    phase_started = Instant::now();
    let tokenizer = QwenTokenizer::from_files(
        &plan.tokenizer,
        &plan.tokenizer_config,
        Some(&plan.chat_template),
    )?;
    progress("tokenizer", phase_started.elapsed());
    let session_cache = FlashMoeSessionCache::new(
        FlashMoeDiskCache::from_plan(plan, config.num_hidden_layers, &options.session_cache),
        options.memory_sessions,
    );
    Ok(FlashMoeEngine {
        plan: plan.clone(),
        scheduler,
        dense,
        tokenizer,
        metal,
        input_adapter_executor,
        config,
        model_layout,
        routing_policy,
        expert_access,
        qwen_prefill_graph,
        runtime,
        executor: deepseek_graph.map_or(
            ResolvedModelExecutor::Qwen,
            ResolvedModelExecutor::DeepSeekV4,
        ),
        linear_attention_weights,
        shared_expert_weights,
        session_cache,
        deepseek_sessions: DeepSeekV4SessionStore::default(),
    })
}

fn format_missing(paths: &[std::path::PathBuf]) -> String {
    if paths.is_empty() {
        "none".to_string()
    } else {
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}
