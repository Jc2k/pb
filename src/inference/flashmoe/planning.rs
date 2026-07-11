use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::experts::first_missing_expert_pack_for_shape;
use super::model_family::{QwenModelConfig, is_qwen3_moe, is_qwen3_vl, is_qwen35_or_legacy_alias};
use super::types::{
    ACTIVE_EXPERTS_PER_TOKEN, BackendSelection, CacheStatus, EXPECTED_EXPERT_BYTES,
    ExpertQuantization, HIDDEN_DIM, LEGACY_QWEN_CODER_MARKER, NUM_EXPERTS, NUM_LAYERS,
    QWEN35_BF16_MODEL, QWEN35_MODEL,
};

const QWEN35_MIN_ACTIVE_EXPERTS: usize = 4;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlashMoeRoutingPolicy {
    pub active_experts_override: Option<usize>,
    pub force_active_experts: bool,
}

impl FlashMoeRoutingPolicy {
    pub fn new(active_experts_override: Option<usize>, force_active_experts: bool) -> Self {
        Self {
            active_experts_override,
            force_active_experts,
        }
    }

    pub(crate) fn resolve(
        &self,
        model: &str,
        config: &QwenModelConfig,
    ) -> Result<ResolvedRoutingPolicy> {
        let qwen35_profile = is_qwen35_or_legacy_alias(model);
        let (source, active_experts) = if let Some(active_experts) = self.active_experts_override {
            (ActiveExpertsSource::UserOverride, active_experts)
        } else if qwen35_profile {
            (
                ActiveExpertsSource::Qwen35FlashMoeProfile,
                ACTIVE_EXPERTS_PER_TOKEN,
            )
        } else {
            (
                ActiveExpertsSource::ModelConfig,
                config.config_active_experts(),
            )
        };
        let experts = config.experts();
        if experts == 0 || active_experts == 0 || active_experts > experts {
            bail!(
                "invalid MoE routing policy: num_experts={experts}, active_experts={active_experts}"
            );
        }
        if qwen35_profile && active_experts < QWEN35_MIN_ACTIVE_EXPERTS {
            if self.force_active_experts {
                tracing::warn!(
                    model,
                    active_experts,
                    minimum = QWEN35_MIN_ACTIVE_EXPERTS,
                    "forcing Qwen3.5 Flash-MoE active-expert count below the quality guard"
                );
            } else {
                bail!(
                    "Qwen3.5 Flash-MoE routing requires K >= {QWEN35_MIN_ACTIVE_EXPERTS}; got K={active_experts}. Set model.flashmoe_force_active_experts=true or pass --flashmoe-force-active-experts to force this experimental routing."
                );
            }
        }
        Ok(ResolvedRoutingPolicy {
            active_experts,
            source,
            force_active_experts: self.force_active_experts,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveExpertsSource {
    ModelConfig,
    Qwen35FlashMoeProfile,
    UserOverride,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedRoutingPolicy {
    pub(crate) active_experts: usize,
    pub(crate) source: ActiveExpertsSource,
    pub(crate) force_active_experts: bool,
}

#[derive(Debug, Clone)]
pub struct FlashMoePlan {
    pub model: String,
    pub model_cache_dir: PathBuf,
    pub runtime_dir: PathBuf,
    pub non_expert_weights: PathBuf,
    pub tensor_manifest: PathBuf,
    pub model_config: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: PathBuf,
    pub experts_dir: PathBuf,
    pub uses_metal: bool,
    pub streams_experts_from_nand: bool,
    pub quantization: ExpertQuantization,
    pub routing_policy: FlashMoeRoutingPolicy,
    pub vision_weights: Option<PathBuf>,
    pub vision_manifest: Option<PathBuf>,
    pub vision_config_path: Option<PathBuf>,
}

pub fn select_backend(model: &str) -> BackendSelection {
    if supports_flashmoe(model) {
        BackendSelection::FlashMoePreferred
    } else {
        BackendSelection::LlamaCpp
    }
}

pub fn supports_flashmoe(model: &str) -> bool {
    is_arm_macos() && is_flashmoe_model_name(model)
}

pub fn is_arm_macos() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

fn is_flashmoe_model_name(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase();
    if normalized.contains("gguf") {
        return false;
    }
    is_qwen35_or_legacy_alias(model) || is_qwen3_vl(model) || is_qwen3_moe(model)
}

pub fn is_flashmoe_hf_model(model: &str) -> bool {
    model.starts_with("hf://") && is_flashmoe_model_name(model)
}

pub fn canonical_model(model: &str) -> String {
    if model
        .to_ascii_lowercase()
        .contains(LEGACY_QWEN_CODER_MARKER)
    {
        QWEN35_MODEL.to_string()
    } else {
        model.to_string()
    }
}

pub fn cache_version_for_model(model: &str) -> &'static str {
    default_expert_quantization(model).cache_version()
}

pub fn default_expert_quantization(model: &str) -> ExpertQuantization {
    if canonical_model(model) == QWEN35_BF16_MODEL {
        ExpertQuantization::Bf16
    } else {
        ExpertQuantization::FourBitProduction
    }
}

pub fn plan(model: &str, models_root: &Path) -> Option<FlashMoePlan> {
    plan_with_routing(model, models_root, FlashMoeRoutingPolicy::default())
}

pub fn plan_unchecked(model: &str, models_root: &Path) -> FlashMoePlan {
    plan_unchecked_with_routing(model, models_root, FlashMoeRoutingPolicy::default())
}

pub fn plan_unchecked_with_quantization(
    model: &str,
    models_root: &Path,
    quantization: ExpertQuantization,
) -> FlashMoePlan {
    plan_unchecked_with_routing_and_quantization(
        model,
        models_root,
        FlashMoeRoutingPolicy::default(),
        quantization,
    )
}

pub fn plan_with_routing(
    model: &str,
    models_root: &Path,
    routing_policy: FlashMoeRoutingPolicy,
) -> Option<FlashMoePlan> {
    supports_flashmoe(model)
        .then(|| plan_unchecked_with_routing(model, models_root, routing_policy))
}

pub fn plan_unchecked_with_routing(
    model: &str,
    models_root: &Path,
    routing_policy: FlashMoeRoutingPolicy,
) -> FlashMoePlan {
    plan_unchecked_with_routing_and_quantization(
        model,
        models_root,
        routing_policy,
        default_expert_quantization(model),
    )
}

pub fn plan_unchecked_with_routing_and_quantization(
    model: &str,
    models_root: &Path,
    routing_policy: FlashMoeRoutingPolicy,
    quantization: ExpertQuantization,
) -> FlashMoePlan {
    plan_unchecked_with_cache_version_and_quantization(
        model,
        models_root,
        routing_policy,
        quantization.cache_version(),
        quantization,
    )
}

pub fn plan_unchecked_with_cache_version(
    model: &str,
    models_root: &Path,
    routing_policy: FlashMoeRoutingPolicy,
    cache_version: &str,
) -> FlashMoePlan {
    plan_unchecked_with_cache_version_and_quantization(
        model,
        models_root,
        routing_policy,
        cache_version,
        default_expert_quantization(model),
    )
}

fn plan_unchecked_with_cache_version_and_quantization(
    model: &str,
    models_root: &Path,
    routing_policy: FlashMoeRoutingPolicy,
    cache_version: &str,
    quantization: ExpertQuantization,
) -> FlashMoePlan {
    let model = canonical_model(model);
    let model_cache_dir = models_root.join(crate::cache_dir_name(&model));
    let runtime_dir = model_cache_dir.join(cache_version);
    let vl = is_qwen3_vl(&model);
    FlashMoePlan {
        vision_weights: vl.then(|| runtime_dir.join("vision_weights.bin")),
        vision_manifest: vl.then(|| runtime_dir.join("vision_weights.json")),
        vision_config_path: vl.then(|| runtime_dir.join("vision_config.json")),
        non_expert_weights: runtime_dir.join("model_weights.bin"),
        tensor_manifest: runtime_dir.join("model_weights.json"),
        model_config: runtime_dir.join("config.json"),
        tokenizer: model_cache_dir.join("tokenizer.json"),
        tokenizer_config: model_cache_dir.join("tokenizer_config.json"),
        experts_dir: runtime_dir.join("packed_experts"),
        runtime_dir,
        model,
        model_cache_dir,
        uses_metal: true,
        streams_experts_from_nand: true,
        quantization,
        routing_policy,
    }
}

impl FlashMoePlan {
    pub fn cache_status(&self) -> Result<CacheStatus> {
        let mut required = vec![
            self.non_expert_weights.clone(),
            self.tensor_manifest.clone(),
            self.model_config.clone(),
            self.tokenizer.clone(),
        ];
        if is_qwen3_vl(&self.model) {
            required.extend(
                [
                    self.vision_weights.clone(),
                    self.vision_manifest.clone(),
                    self.vision_config_path.clone(),
                ]
                .into_iter()
                .flatten(),
            );
        }
        let mut missing: Vec<PathBuf> = required
            .into_iter()
            .filter(|path| !path.is_file())
            .collect();
        if !self.experts_dir.is_dir() {
            missing.push(self.experts_dir.clone());
        }

        let (expert_files, expert_bytes) = expert_store_size(&self.experts_dir)?;
        if self.experts_dir.is_dir() && self.model_config.is_file() {
            let config = QwenModelConfig::from_file(&self.model_config).with_context(|| {
                format!(
                    "cannot resolve expert cache coverage from {}",
                    self.model_config.display()
                )
            })?;
            if let Some(missing_expert) = first_missing_expert_pack_for_shape(
                &self.experts_dir,
                config.num_hidden_layers,
                config.experts(),
            )? {
                missing.push(missing_expert);
            }
        }
        let ready = missing.is_empty() && expert_bytes > 0;
        Ok(CacheStatus {
            ready,
            missing,
            expert_files,
            expert_bytes,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "Flash-MoE {} for {}: {} layers, {} experts/layer, K={}, hidden={}, cache={}, expert store={} (~{} GiB)",
            self.quantization.cache_version(),
            self.model,
            NUM_LAYERS,
            NUM_EXPERTS,
            ACTIVE_EXPERTS_PER_TOKEN,
            HIDDEN_DIM,
            self.runtime_dir.display(),
            self.experts_dir.display(),
            EXPECTED_EXPERT_BYTES / (1024 * 1024 * 1024)
        )
    }
}

fn expert_store_size(path: &Path) -> Result<(usize, u64)> {
    if !path.is_dir() {
        return Ok((0, 0));
    }
    let mut files = 0usize;
    let mut bytes = 0u64;
    for entry in fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file()
            && entry.path().extension().and_then(|ext| ext.to_str()) == Some("bin")
        {
            files += 1;
            bytes = bytes.saturating_add(metadata.len());
        }
    }
    Ok((files, bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashMoeCacheCleanupKind {
    StaleRuntimeDir,
    SourceShard,
}

impl FlashMoeCacheCleanupKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StaleRuntimeDir => "stale-runtime-dir",
            Self::SourceShard => "source-shard",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeCacheCleanupCandidate {
    pub path: PathBuf,
    pub kind: FlashMoeCacheCleanupKind,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeCacheCleanupReport {
    pub model: String,
    pub model_cache_dir: PathBuf,
    pub active_runtime_dir: PathBuf,
    pub include_source_shards: bool,
    pub deleted: bool,
    pub candidates: Vec<FlashMoeCacheCleanupCandidate>,
}

impl FlashMoeCacheCleanupReport {
    pub fn total_bytes(&self) -> u64 {
        self.candidates
            .iter()
            .map(|candidate| candidate.bytes)
            .sum()
    }
}

pub fn plan_cache_cleanup(
    plan: &FlashMoePlan,
    include_source_shards: bool,
) -> Result<FlashMoeCacheCleanupReport> {
    let mut candidates = Vec::new();
    if plan.model_cache_dir.is_dir() {
        for entry in fs::read_dir(&plan.model_cache_dir)
            .with_context(|| format!("failed to read {}", plan.model_cache_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            let file_type = entry.file_type()?;

            if file_type.is_dir() && file_name.starts_with("flashmoe-") && path != plan.runtime_dir
            {
                candidates.push(FlashMoeCacheCleanupCandidate {
                    bytes: cache_cleanup_path_size(&path)?,
                    kind: FlashMoeCacheCleanupKind::StaleRuntimeDir,
                    path,
                });
                continue;
            }
            if include_source_shards
                && file_type.is_file()
                && is_flashmoe_source_shard_name(&file_name)
            {
                candidates.push(FlashMoeCacheCleanupCandidate {
                    bytes: entry.metadata()?.len(),
                    kind: FlashMoeCacheCleanupKind::SourceShard,
                    path,
                });
            }
        }
    }
    candidates.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(FlashMoeCacheCleanupReport {
        model: plan.model.clone(),
        model_cache_dir: plan.model_cache_dir.clone(),
        active_runtime_dir: plan.runtime_dir.clone(),
        include_source_shards,
        deleted: false,
        candidates,
    })
}

pub fn clean_cache(
    plan: &FlashMoePlan,
    include_source_shards: bool,
    delete: bool,
) -> Result<FlashMoeCacheCleanupReport> {
    let mut report = plan_cache_cleanup(plan, include_source_shards)?;
    if delete {
        for candidate in &report.candidates {
            ensure_cache_cleanup_candidate_is_safe(&report.model_cache_dir, candidate)?;
            delete_cache_cleanup_candidate(candidate)?;
        }
        report.deleted = true;
    }
    Ok(report)
}

pub fn clean_source_shards(
    plan: &FlashMoePlan,
    delete: bool,
) -> Result<FlashMoeCacheCleanupReport> {
    let mut report = plan_cache_cleanup(plan, true)?;
    report
        .candidates
        .retain(|candidate| candidate.kind == FlashMoeCacheCleanupKind::SourceShard);
    if delete {
        for candidate in &report.candidates {
            ensure_cache_cleanup_candidate_is_safe(&report.model_cache_dir, candidate)?;
            delete_cache_cleanup_candidate(candidate)?;
        }
        report.deleted = true;
    }
    Ok(report)
}

fn is_flashmoe_source_shard_name(file_name: &str) -> bool {
    (file_name.starts_with("model.safetensors-") || file_name.starts_with("model-"))
        && file_name.contains("-of-")
        && file_name.ends_with(".safetensors")
}

fn ensure_cache_cleanup_candidate_is_safe(
    model_cache_dir: &Path,
    candidate: &FlashMoeCacheCleanupCandidate,
) -> Result<()> {
    if candidate.path.parent() != Some(model_cache_dir) {
        bail!(
            "refusing to clean cache path outside model cache root: {}",
            candidate.path.display()
        );
    }
    Ok(())
}

fn delete_cache_cleanup_candidate(candidate: &FlashMoeCacheCleanupCandidate) -> Result<()> {
    let metadata = fs::symlink_metadata(&candidate.path)
        .with_context(|| format!("failed to inspect {}", candidate.path.display()))?;
    if metadata.is_dir() {
        fs::remove_dir_all(&candidate.path)
            .with_context(|| format!("failed to delete {}", candidate.path.display()))?;
    } else {
        fs::remove_file(&candidate.path)
            .with_context(|| format!("failed to delete {}", candidate.path.display()))?;
    }
    Ok(())
}

fn cache_cleanup_path_size(path: &Path) -> Result<u64> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Ok(0);
    }
    let mut bytes = 0u64;
    for entry in fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))? {
        bytes = bytes.saturating_add(cache_cleanup_path_size(&entry?.path())?);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::flashmoe::experts::{
        ExpertLayerPackMetadata, ExpertPackMetadata, expert_layer_metadata_path, expert_layer_path,
    };
    use crate::inference::flashmoe::types::{
        BF16_CACHE_VERSION, CACHE_VERSION, F16_CACHE_VERSION, QWEN3_VL_MODEL,
    };

    fn qwen_config(model_type: &str, active_experts: usize) -> QwenModelConfig {
        serde_json::from_value(serde_json::json!({
            "model_type": model_type,
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_hidden_layers": 1,
            "hidden_size": 64,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "vocab_size": 128,
            "torch_dtype": "bfloat16",
            "num_experts": 8,
            "num_experts_per_tok": active_experts,
            "moe_intermediate_size": 64,
            "norm_topk_prob": true
        }))
        .unwrap()
    }

    #[test]
    fn routing_policy_resolves_family_specific_k_before_graph_construction() {
        let qwen35 = FlashMoeRoutingPolicy::default()
            .resolve(QWEN35_MODEL, &qwen_config("qwen3_5_moe", 8))
            .unwrap();
        assert_eq!(qwen35.active_experts, ACTIVE_EXPERTS_PER_TOKEN);
        assert_eq!(qwen35.source, ActiveExpertsSource::Qwen35FlashMoeProfile);

        let qwen = FlashMoeRoutingPolicy::default()
            .resolve("hf://Qwen/Qwen3-30B-A3B", &qwen_config("qwen3_moe", 6))
            .unwrap();
        assert_eq!(qwen.active_experts, 6);
        assert_eq!(qwen.source, ActiveExpertsSource::ModelConfig);
    }

    #[test]
    fn routing_policy_resolves_explicit_override_before_graph_construction() {
        let resolved = FlashMoeRoutingPolicy::new(Some(6), false)
            .resolve("hf://Qwen/Qwen3-30B-A3B", &qwen_config("qwen3_moe", 2))
            .unwrap();

        assert_eq!(resolved.active_experts, 6);
        assert_eq!(resolved.source, ActiveExpertsSource::UserOverride);
    }

    #[test]
    fn routing_policy_rejects_qwen35_k_below_four_unless_forced() {
        let config = qwen_config("qwen3_5_moe", 10);
        let error = FlashMoeRoutingPolicy::new(Some(3), false)
            .resolve(QWEN35_MODEL, &config)
            .unwrap_err();
        assert!(error.to_string().contains("requires K >= 4"), "{error:#}");

        let forced = FlashMoeRoutingPolicy::new(Some(3), true)
            .resolve(QWEN35_MODEL, &config)
            .unwrap();
        assert_eq!(forced.active_experts, 3);
        assert!(forced.force_active_experts);
    }

    #[test]
    fn planning_selects_typed_variant_artifacts_without_runtime_probe() {
        let temp = tempfile::tempdir().unwrap();
        let vl = plan_unchecked(QWEN3_VL_MODEL, temp.path());
        assert!(vl.vision_weights.is_some());
        assert!(vl.vision_manifest.is_some());
        assert!(vl.vision_config_path.is_some());
        assert_eq!(vl.quantization, ExpertQuantization::FourBitProduction);

        let qwen35 = plan_unchecked(QWEN35_MODEL, temp.path());
        assert!(qwen35.runtime_dir.ends_with(CACHE_VERSION));
        assert!(qwen35.non_expert_weights.ends_with("model_weights.bin"));
        assert!(qwen35.experts_dir.ends_with("packed_experts"));
        assert!(qwen35.uses_metal);
        assert!(qwen35.streams_experts_from_nand);
        assert!(qwen35.describe().contains("397B"));
    }

    #[test]
    fn qwen_moe_repositories_resolve_their_typed_pull_family() {
        assert!(is_flashmoe_hf_model("hf://Qwen/Qwen3-30B-A3B"));
        assert!(is_flashmoe_hf_model("hf://Qwen/Qwen3-235B-A22B-Instruct"));
        assert!(is_flashmoe_hf_model(QWEN3_VL_MODEL));
        assert!(is_flashmoe_hf_model("hf://Qwen/Qwen3-VL-30B-A3B-Instruct"));
        assert!(!is_flashmoe_hf_model("hf://Qwen/Qwen3-VL-8B-Instruct"));
        assert!(!is_flashmoe_hf_model("hf://Qwen/Qwen3-8B"));
        assert!(!is_flashmoe_hf_model("qwen3-30b-a3b"));
        assert_eq!(
            canonical_model("hf://unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M.gguf"),
            QWEN35_MODEL
        );
        assert_eq!(
            select_backend("qwen-vision.gguf"),
            BackendSelection::LlamaCpp
        );

        let vl = plan_unchecked(QWEN3_VL_MODEL, Path::new("/models"));
        assert!(vl.vision_weights.is_some());
        assert!(vl.vision_manifest.is_some());
        assert!(vl.vision_config_path.is_some());
    }

    #[test]
    fn explicit_storage_policy_selects_distinct_qwen_family_cache_namespaces() {
        let temp = tempfile::tempdir().unwrap();
        for model in ["hf://Qwen/Qwen3-30B-A3B", QWEN3_VL_MODEL] {
            let q4 = plan_unchecked_with_quantization(
                model,
                temp.path(),
                ExpertQuantization::FourBitProduction,
            );
            let bf16 =
                plan_unchecked_with_quantization(model, temp.path(), ExpertQuantization::Bf16);
            let f16 = plan_unchecked_with_quantization(model, temp.path(), ExpertQuantization::F16);

            assert_eq!(q4.model_cache_dir, bf16.model_cache_dir);
            assert_eq!(q4.model_cache_dir, f16.model_cache_dir);
            assert!(q4.runtime_dir.ends_with(CACHE_VERSION));
            assert!(bf16.runtime_dir.ends_with(BF16_CACHE_VERSION));
            assert!(f16.runtime_dir.ends_with(F16_CACHE_VERSION));
            assert_eq!(q4.quantization, ExpertQuantization::FourBitProduction);
            assert_eq!(bf16.quantization, ExpertQuantization::Bf16);
            assert_eq!(f16.quantization, ExpertQuantization::F16);
            assert!(bf16.describe().contains(BF16_CACHE_VERSION));
            assert!(f16.describe().contains(F16_CACHE_VERSION));
        }
    }

    #[test]
    fn default_storage_policy_preserves_q4_and_official_bf16_compatibility() {
        assert_eq!(
            default_expert_quantization("hf://Qwen/Qwen3-30B-A3B"),
            ExpertQuantization::FourBitProduction
        );
        assert_eq!(
            default_expert_quantization(QWEN3_VL_MODEL),
            ExpertQuantization::FourBitProduction
        );
        assert_eq!(
            default_expert_quantization(QWEN35_BF16_MODEL),
            ExpertQuantization::Bf16
        );
        assert_eq!(canonical_model(QWEN35_BF16_MODEL), QWEN35_BF16_MODEL);
        assert_eq!(cache_version_for_model(QWEN35_MODEL), CACHE_VERSION);
        assert_eq!(
            cache_version_for_model(QWEN35_BF16_MODEL),
            BF16_CACHE_VERSION
        );

        let plan = plan_unchecked(QWEN35_BF16_MODEL, Path::new("/models"));
        assert!(plan.runtime_dir.ends_with(BF16_CACHE_VERSION));
        assert_eq!(plan.model, QWEN35_BF16_MODEL);
    }

    #[test]
    fn cache_status_rejects_malformed_config_without_qwen35_shape_fallback() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, root.path());
        fs::create_dir_all(&plan.experts_dir).unwrap();
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        fs::write(&plan.model_config, b"not-json").unwrap();
        fs::write(plan.experts_dir.join("layer_00.bin"), b"expert").unwrap();

        let error = plan.cache_status().unwrap_err();

        assert!(
            error
                .to_string()
                .contains("cannot resolve expert cache coverage"),
            "{error:#}"
        );
    }

    #[test]
    fn cache_status_reports_missing_runtime_artifacts() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, root.path());

        let status = plan.cache_status().unwrap();

        assert!(!status.ready);
        assert!(
            status
                .missing
                .iter()
                .any(|path| path.ends_with("model_weights.bin"))
        );
        assert_eq!(status.expert_files, 0);
    }

    #[test]
    fn cache_status_rejects_partial_expert_layer_coverage() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, root.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        fs::create_dir_all(&plan.experts_dir).unwrap();
        fs::write(&plan.non_expert_weights, b"dense").unwrap();
        fs::write(
            &plan.tensor_manifest,
            br#"{"model":"","cache_version":"","dense_shards":[],"expert_tensors":[],"dense_tensors":[]}"#,
        )
        .unwrap();
        fs::write(
            &plan.model_config,
            br#"{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
        fs::write(&plan.tokenizer, b"{}").unwrap();

        let packs = (0..8)
            .map(|expert| ExpertPackMetadata {
                layer: 0,
                expert,
                packed_bytes: 1,
                records: Vec::new(),
            })
            .collect();
        let metadata = ExpertLayerPackMetadata::new(0, 1, 8, packs);
        fs::write(expert_layer_path(&plan.experts_dir, 0), vec![0; 8]).unwrap();
        fs::write(
            expert_layer_metadata_path(&plan.experts_dir, 0),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();

        let status = plan.cache_status().unwrap();

        assert!(!status.ready);
        assert!(
            status
                .missing
                .iter()
                .any(|path| path.ends_with("layer_01.bin"))
        );
    }

    #[test]
    fn cache_cleanup_preserves_active_runtime_during_dry_run() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_BF16_MODEL, root.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        fs::write(plan.runtime_dir.join("model_weights.bin"), b"active").unwrap();
        let stale_runtime = plan.model_cache_dir.join("flashmoe-v2-denseq4");
        fs::create_dir_all(stale_runtime.join("packed_experts")).unwrap();
        fs::write(stale_runtime.join("model_weights.bin"), b"stale").unwrap();

        let report = clean_cache(&plan, false, false).unwrap();

        assert!(!report.deleted);
        assert_eq!(report.candidates.len(), 1);
        assert_eq!(
            report.candidates[0].kind,
            FlashMoeCacheCleanupKind::StaleRuntimeDir
        );
        assert_eq!(report.candidates[0].path, stale_runtime);
        assert!(plan.runtime_dir.join("model_weights.bin").is_file());
        assert!(stale_runtime.is_dir());
    }

    #[test]
    fn cache_cleanup_deletes_only_selected_runtime_and_source_artifacts() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_BF16_MODEL, root.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        fs::write(plan.runtime_dir.join("model_weights.bin"), b"active").unwrap();
        let stale_runtime = plan.model_cache_dir.join("flashmoe-v1");
        let source_shard = plan
            .model_cache_dir
            .join("model.safetensors-00002-of-00094.safetensors");
        let unrelated_file = plan.model_cache_dir.join("config.json");
        fs::create_dir_all(&stale_runtime).unwrap();
        fs::write(stale_runtime.join("model_weights.bin"), b"stale").unwrap();
        fs::write(&source_shard, b"source").unwrap();
        fs::write(&unrelated_file, b"{}").unwrap();

        let runtimes_only = clean_cache(&plan, false, true).unwrap();
        assert_eq!(runtimes_only.candidates.len(), 1);
        assert!(!stale_runtime.exists());
        assert!(source_shard.is_file());

        let with_sources = clean_cache(&plan, true, true).unwrap();
        assert_eq!(with_sources.candidates.len(), 1);
        assert_eq!(
            with_sources.candidates[0].kind,
            FlashMoeCacheCleanupKind::SourceShard
        );
        assert!(!source_shard.exists());
        assert!(unrelated_file.is_file());
        assert!(plan.runtime_dir.join("model_weights.bin").is_file());
    }

    #[test]
    fn source_shard_cleanup_preserves_runtime_directories_and_matches_mlx_names() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, root.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        fs::write(plan.runtime_dir.join("model_weights.bin"), b"active").unwrap();
        let stale_runtime = plan.model_cache_dir.join("flashmoe-v1");
        fs::create_dir_all(&stale_runtime).unwrap();
        fs::write(stale_runtime.join("model_weights.bin"), b"stale").unwrap();
        let hf_shard = plan
            .model_cache_dir
            .join("model.safetensors-00001-of-00002.safetensors");
        let mlx_shard = plan
            .model_cache_dir
            .join("model-00001-of-00046.safetensors");
        fs::write(&hf_shard, b"source").unwrap();
        fs::write(&mlx_shard, b"source").unwrap();

        let report = clean_source_shards(&plan, true).unwrap();

        assert!(report.deleted);
        assert_eq!(report.candidates.len(), 2);
        assert!(!hf_shard.exists());
        assert!(!mlx_shard.exists());
        assert!(stale_runtime.is_dir());
        assert!(plan.runtime_dir.join("model_weights.bin").is_file());
    }
}
