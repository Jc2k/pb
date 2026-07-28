//! Rust-specific streaming control built directly on rust-analyzer internals.
//!
//! This crate owns Rust parsing, HIR name resolution, and Rust type shapes. The language-neutral
//! collar supplies virtual-source events and combines decisions; it does not flatten rust-analyzer
//! state into a cross-language type system.

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    path::{Path, PathBuf},
    time::Instant,
};

use pb_control_collar::analysis::{
    AnalyzerCapability, AnalyzerLayerDescriptor, LanguageId, LayerReadiness, LayerReadinessReceipt,
    ReadinessOrigin, SemanticCompleteness, SemanticWorldId,
};
use ra_ap_base_db::{
    CrateGraphBuilder, CrateOrigin, DependencyBuilder, SourceDatabase, all_crates,
};
use ra_ap_hir::{
    Crate, Function, Module, ModuleDef, ScopeDef, Semantics, Type, diagnostics::AnyDiagnostic,
};
use ra_ap_ide_db::{ChangeWithProcMacros, RootDatabase};
use ra_ap_load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};
use ra_ap_project_model::{CargoConfig, RustLibSource};
use ra_ap_vfs::{FileId, Vfs};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod stream;
pub use stream::{RustLayerCheckpoint, RustStreamingLayer, RustWorkspaceStreamingLayer};

pub const RUST_LAYER_CONTRACT_VERSION: u32 = 2;
pub const RUST_ANALYZER_VERSION: &str = "ra_ap_0.0.344";

/// The request-local rust-analyzer overlay may hard-reject only the promoted diagnostics in this
/// enum. The set is deliberately narrower than either rustc or rust-analyzer's full diagnostics:
/// every addition needs its own baseline-debt, repairability, macro/configuration, and final-replay
/// qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RustDeepDiagnostic {
    UnresolvedName,
    UnresolvedImport,
    MissingField,
    MissingMethod,
    Privacy,
    TypeMismatch,
    InvalidCall,
    Mutability,
    Ownership,
    TraitContract,
}

impl RustDeepDiagnostic {
    pub const fn id(self) -> &'static str {
        match self {
            Self::UnresolvedName => "unresolved_name",
            Self::UnresolvedImport => "unresolved_import",
            Self::MissingField => "missing_field",
            Self::MissingMethod => "missing_method",
            Self::Privacy => "privacy",
            Self::TypeMismatch => "type_mismatch",
            Self::InvalidCall => "invalid_call",
            Self::Mutability => "mutability",
            Self::Ownership => "ownership",
            Self::TraitContract => "trait_contract",
        }
    }

    fn obligation(self) -> &'static str {
        match self {
            Self::UnresolvedName => "rust_deep_unresolved_name",
            Self::UnresolvedImport => "rust_deep_unresolved_import",
            Self::MissingField => "rust_deep_missing_field",
            Self::MissingMethod => "rust_deep_missing_method",
            Self::Privacy => "rust_deep_privacy",
            Self::TypeMismatch => "rust_deep_type_mismatch",
            Self::InvalidCall => "rust_deep_invalid_call",
            Self::Mutability => "rust_deep_mutability",
            Self::Ownership => "rust_deep_ownership",
            Self::TraitContract => "rust_deep_trait_contract",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RustDeepUnknownReason {
    BuildScriptDisabled,
    ProceduralMacroDisabled,
    DependencyManifestUnavailable,
    ImportResolutionUnsupported,
    SourceTopologyChanged,
}

impl RustDeepUnknownReason {
    pub const fn id(self) -> &'static str {
        match self {
            Self::BuildScriptDisabled => "build_script_disabled",
            Self::ProceduralMacroDisabled => "procedural_macro_disabled",
            Self::DependencyManifestUnavailable => "dependency_manifest_unavailable",
            Self::ImportResolutionUnsupported => "import_resolution_unsupported",
            Self::SourceTopologyChanged => "source_topology_changed",
        }
    }

    fn obligation(self) -> &'static str {
        match self {
            Self::BuildScriptDisabled => "rust_deep_build_script_unknown",
            Self::ProceduralMacroDisabled => "rust_deep_proc_macro_unknown",
            Self::DependencyManifestUnavailable => "rust_deep_dependency_manifest_unknown",
            Self::ImportResolutionUnsupported => "rust_deep_import_resolution_unknown",
            Self::SourceTopologyChanged => "rust_deep_source_topology_unknown",
        }
    }
}

/// Completeness of the no-execution rust-analyzer profile for the promoted final-transaction
/// diagnostics. `Exact` is scoped to that allowlist; it is not a rustc compilation or borrow-check
/// guarantee.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "reasons")]
pub enum RustDeepProfile {
    Exact,
    Partial(Vec<RustDeepUnknownReason>),
}

impl RustDeepProfile {
    fn is_exact(&self) -> bool {
        matches!(self, Self::Exact)
    }
}

type RustDeepDiagnosticCounts = BTreeMap<RustDeepDiagnostic, usize>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustLayerConfig {
    pub contract_version: u32,
    /// A controller-created immutable shadow. The layer never loads the live workspace for a
    /// generation request.
    pub shadow_root: PathBuf,
    pub target_crate: String,
    pub world_sha256: String,
    pub configuration_sha256: String,
    pub dependency_sha256: String,
}

/// Identity and immutable shadow used to load one rust-analyzer database for an entire Cargo
/// project. Target selection is request-local so a multi-crate workspace does not pay the Cargo
/// load cost once per crate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustProjectConfig {
    pub contract_version: u32,
    pub shadow_root: PathBuf,
    pub world_sha256: String,
    pub configuration_sha256: String,
    pub dependency_sha256: String,
}

impl From<&RustLayerConfig> for RustProjectConfig {
    fn from(config: &RustLayerConfig) -> Self {
        Self {
            contract_version: config.contract_version,
            shadow_root: config.shadow_root.clone(),
            world_sha256: config.world_sha256.clone(),
            configuration_sha256: config.configuration_sha256.clone(),
            dependency_sha256: config.dependency_sha256.clone(),
        }
    }
}

impl RustProjectConfig {
    fn validate(&self) -> Result<(), RustLayerError> {
        if self.contract_version != RUST_LAYER_CONTRACT_VERSION {
            return Err(RustLayerError::InvalidConfig(format!(
                "Rust layer contract version must be {RUST_LAYER_CONTRACT_VERSION}"
            )));
        }
        if !self.shadow_root.is_absolute() {
            return Err(RustLayerError::InvalidConfig(
                "Rust project shadow root must be absolute".to_string(),
            ));
        }
        for (label, digest) in [
            ("world", self.world_sha256.as_str()),
            ("configuration", self.configuration_sha256.as_str()),
            ("dependency", self.dependency_sha256.as_str()),
        ] {
            if !is_lower_hex_digest(digest) {
                return Err(RustLayerError::InvalidConfig(format!(
                    "Rust layer {label} identity must be a lowercase SHA-256 digest"
                )));
            }
        }
        Ok(())
    }
}

impl RustLayerConfig {
    fn validate(&self) -> Result<(), RustLayerError> {
        RustProjectConfig::from(self).validate()?;
        if self.target_crate.trim().is_empty() || self.target_crate.len() > 512 {
            return Err(RustLayerError::InvalidConfig(
                "Rust layer target crate must be bounded and non-empty".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RustTargetDescriptor {
    /// Stable within the exact semantic world. Includes the crate root so same-named Cargo targets
    /// remain distinct.
    pub id: String,
    pub crate_name: String,
    /// Slash-normalized package directory relative to the immutable shadow root. Empty means the
    /// project root.
    pub package_scope: String,
    pub crate_root: String,
}

#[derive(Clone)]
struct RustTarget {
    descriptor: RustTargetDescriptor,
    krate: Crate,
}

/// Warm immutable rust-analyzer state for a whole Cargo project. This is the expensive lifecycle
/// object: construct and prime it before inference, retain it while its exact identity remains
/// current, and derive cheap request snapshots from it.
pub struct RustProjectWorld {
    descriptor: AnalyzerLayerDescriptor,
    receipt: LayerReadinessReceipt,
    readiness: LayerReadiness,
    db: RootDatabase,
    targets: Vec<RustTarget>,
    file_targets: BTreeMap<String, Vec<String>>,
    file_ids: BTreeMap<String, FileId>,
    deep_profile: RustDeepProfile,
    deep_state: Option<std::sync::Arc<std::sync::Mutex<RustDeepState>>>,
    request_epoch: std::sync::Arc<()>,
}

struct RustDeepState {
    db: RootDatabase,
    targets: Vec<RustTarget>,
    file_targets: BTreeMap<String, Vec<String>>,
    file_ids: BTreeMap<String, FileId>,
    baseline_sources: BTreeMap<String, String>,
    baseline_diagnostics: RustDeepDiagnosticCounts,
    baseline_imports: RustDeepImportStates,
}

type RustDeepImportStates = BTreeMap<String, Option<BTreeMap<Vec<String>, RustDeepImportCounts>>>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RustDeepImportCounts {
    resolved: usize,
    absent: usize,
    unknown: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustDeepImportResolution {
    Resolved,
    Absent,
    Unknown,
}

impl RustProjectWorld {
    pub fn load_and_prime(config: RustProjectConfig) -> Result<Self, RustLayerError> {
        config.validate()?;
        let world = semantic_world_id(&config);
        let load_started = Instant::now();
        let mut cargo = CargoConfig {
            sysroot: Some(RustLibSource::Discover),
            // Semantic preparation is a local correctness boundary, not an implicit dependency
            // installer. Missing registry/git sources fail pre-inference preparation instead of
            // causing an undeclared network path.
            metadata_extra_args: vec!["--offline".to_string()],
            ..CargoConfig::default()
        };
        cargo
            .extra_env
            .insert("CARGO_NET_OFFLINE".to_string(), Some("true".to_string()));
        cargo
            .extra_env
            .insert("RUSTUP_AUTO_INSTALL".to_string(), Some("0".to_string()));
        let load = LoadCargoConfig {
            load_out_dirs_from_check: false,
            with_proc_macro_server: ProcMacroServerChoice::None,
            prefill_caches: false,
            num_worker_threads: 1,
            proc_macro_processes: 0,
        };
        let (db, vfs, _proc_macro) =
            load_workspace_at(&config.shadow_root, &cargo, &load, &|_progress| {})
                .map_err(|error| RustLayerError::ProviderUnavailable(error.to_string()))?;
        let load_millis = elapsed_millis(load_started);
        let targets = local_targets(&db, &vfs, &config.shadow_root)?;
        if targets.is_empty() {
            return Err(RustLayerError::MissingTarget(
                "no local Cargo targets were loaded from the immutable shadow".to_string(),
            ));
        }
        let file_targets = local_file_targets(&db, &vfs, &config.shadow_root, &targets);
        let file_ids = local_file_ids(&vfs, &config.shadow_root);
        let deep_profile = rust_deep_profile(&db, &vfs, &targets);

        let prime_started = Instant::now();
        let mut primed_queries = 0u64;
        for target in &targets {
            let root = target.krate.root_module(&db);
            let _ = root.scope(&db, Some(root));
            primed_queries = primed_queries.saturating_add(1);
            for dependency in target.krate.dependencies(&db) {
                let module = dependency.krate.root_module(&db);
                let _ = module.scope(&db, Some(root));
                primed_queries = primed_queries.saturating_add(1);
            }
        }
        let deep_state = if deep_profile.is_exact() {
            let mut state = independent_deep_state(&db, &vfs, &config.shadow_root)?;
            let (diagnostics, diagnostic_queries) =
                rust_deep_diagnostic_counts(&state.db, &state.targets);
            primed_queries = primed_queries.saturating_add(diagnostic_queries);
            state.baseline_diagnostics = diagnostics;
            state.baseline_imports = rust_deep_import_states(
                &state.db,
                &state.targets,
                &state.file_targets,
                &state.baseline_sources,
            );
            Some(std::sync::Arc::new(std::sync::Mutex::new(state)))
        } else {
            None
        };
        let prime_millis = elapsed_millis(prime_started);
        let descriptor = AnalyzerLayerDescriptor {
            id: "rust-native-v2".to_string(),
            language: LanguageId("rust".to_string()),
            world: world.clone(),
            capabilities: vec![
                AnalyzerCapability::PrefixStructural,
                AnalyzerCapability::SymbolResolution,
                AnalyzerCapability::TypeChecking,
                AnalyzerCapability::OwnershipChecking,
                AnalyzerCapability::DependencyResolution,
                AnalyzerCapability::FinalWorkspaceGate,
            ],
        };
        let receipt = LayerReadinessReceipt {
            world,
            origin: ReadinessOrigin::ColdBuild,
            // This safe profile does not execute build scripts or procedural macros. Positive HIR
            // facts are usable; negative facts are qualified per query.
            completeness: SemanticCompleteness::Partial,
            load_millis,
            prime_millis,
            primed_queries,
        };
        Ok(Self {
            descriptor,
            receipt,
            readiness: LayerReadiness::Ready,
            db,
            targets,
            file_targets,
            file_ids,
            deep_profile,
            deep_state,
            request_epoch: std::sync::Arc::new(()),
        })
    }

    pub fn descriptor(&self) -> &AnalyzerLayerDescriptor {
        &self.descriptor
    }

    pub fn readiness(&self) -> LayerReadiness {
        self.readiness
    }

    pub fn readiness_receipt(&self) -> &LayerReadinessReceipt {
        &self.receipt
    }

    pub fn targets(&self) -> impl ExactSizeIterator<Item = &RustTargetDescriptor> {
        self.targets.iter().map(|target| &target.descriptor)
    }

    pub fn deep_profile(&self) -> &RustDeepProfile {
        &self.deep_profile
    }

    pub fn mark_stale(&mut self) {
        self.readiness = LayerReadiness::Stale;
    }

    pub fn snapshot_for_request(
        &self,
        expected_world: &SemanticWorldId,
    ) -> Result<RustProjectRequestWorld, RustLayerError> {
        ensure_ready_world(self.readiness, &self.descriptor.world, expected_world)?;
        Ok(RustProjectRequestWorld {
            descriptor: self.descriptor.clone(),
            receipt: warm_receipt(&self.receipt),
            db: self.db.clone(),
            targets: self.targets.clone(),
            file_targets: self.file_targets.clone(),
            deep_profile: self.deep_profile.clone(),
            deep_state: self.deep_state.clone(),
            _epoch_lease: std::sync::Arc::clone(&self.request_epoch),
        })
    }

    /// Derive a new exact world after modifications to already indexed Rust source files. Cargo
    /// metadata, dependency graph, VFS topology, and target configuration must be unchanged. Salsa
    /// invalidates affected HIR queries, avoiding a second Cargo load while preserving immutable
    /// request snapshots already handed to an inference.
    pub fn refresh_existing_sources(
        &mut self,
        config: RustProjectConfig,
        changes: &[(String, Vec<u8>)],
    ) -> Result<(), RustLayerError> {
        config.validate()?;
        ensure_ready_world(
            self.readiness,
            &self.descriptor.world,
            &self.descriptor.world,
        )?;
        if config.configuration_sha256 != self.descriptor.world.configuration_sha256
            || config.dependency_sha256 != self.descriptor.world.dependency_sha256
        {
            return Err(RustLayerError::IncrementalRefresh(
                "Cargo configuration or dependency identity changed".to_string(),
            ));
        }
        if changes.is_empty() {
            return Err(RustLayerError::IncrementalRefresh(
                "incremental Rust refresh requires at least one changed source".to_string(),
            ));
        }
        if std::sync::Arc::strong_count(&self.request_epoch) != 1 {
            return Err(RustLayerError::IncrementalRefresh(
                "an inference snapshot from the previous world is still active".to_string(),
            ));
        }
        let mut change = ChangeWithProcMacros::default();
        for (path, bytes) in changes {
            let file_id = self.file_ids.get(path).copied().ok_or_else(|| {
                RustLayerError::IncrementalRefresh(format!(
                    "Rust source {path:?} is new, deleted, or absent from the loaded VFS"
                ))
            })?;
            let text = std::str::from_utf8(bytes).map_err(|_| {
                RustLayerError::IncrementalRefresh(format!(
                    "Rust source {path:?} is not valid UTF-8"
                ))
            })?;
            change.change_file(file_id, Some(text.to_string()));
        }
        self.db.apply_change(change);
        let prime_started = Instant::now();
        let mut primed_queries = 0u64;
        for target in &self.targets {
            let root = target.krate.root_module(&self.db);
            let _ = root.scope(&self.db, Some(root));
            primed_queries = primed_queries.saturating_add(1);
            for dependency in target.krate.dependencies(&self.db) {
                let module = dependency.krate.root_module(&self.db);
                let _ = module.scope(&self.db, Some(root));
                primed_queries = primed_queries.saturating_add(1);
            }
        }
        if let Some(deep_state) = &self.deep_state {
            let mut deep_state = deep_state.lock().map_err(|_| {
                RustLayerError::IncrementalRefresh(
                    "Rust deep semantic state lock is poisoned".to_string(),
                )
            })?;
            let mut deep_change = ChangeWithProcMacros::default();
            for (path, bytes) in changes {
                let file_id = deep_state.file_ids.get(path).copied().ok_or_else(|| {
                    RustLayerError::IncrementalRefresh(format!(
                        "Rust deep semantic source {path:?} is absent from the loaded VFS"
                    ))
                })?;
                let text = std::str::from_utf8(bytes).map_err(|_| {
                    RustLayerError::IncrementalRefresh(format!(
                        "Rust deep semantic source {path:?} is not valid UTF-8"
                    ))
                })?;
                deep_change.change_file(file_id, Some(text.to_string()));
                deep_state
                    .baseline_sources
                    .insert(path.clone(), text.to_string());
            }
            deep_state.db.apply_change(deep_change);
            let (diagnostics, diagnostic_queries) =
                rust_deep_diagnostic_counts(&deep_state.db, &deep_state.targets);
            primed_queries = primed_queries.saturating_add(diagnostic_queries);
            deep_state.baseline_diagnostics = diagnostics;
            deep_state.baseline_imports = rust_deep_import_states(
                &deep_state.db,
                &deep_state.targets,
                &deep_state.file_targets,
                &deep_state.baseline_sources,
            );
        }
        let world = semantic_world_id(&config);
        self.descriptor.world = world.clone();
        self.receipt = LayerReadinessReceipt {
            world,
            origin: ReadinessOrigin::WarmCache,
            completeness: self.receipt.completeness,
            load_millis: 0,
            prime_millis: elapsed_millis(prime_started),
            primed_queries,
        };
        Ok(())
    }

    fn target_by_name(&self, name: &str) -> Result<&RustTarget, RustLayerError> {
        let targets = self
            .targets
            .iter()
            .filter(|target| target.descriptor.crate_name == name)
            .collect::<Vec<_>>();
        match targets.as_slice() {
            [target] => Ok(*target),
            [] => Err(RustLayerError::MissingTarget(name.to_string())),
            _ => Err(RustLayerError::MissingTarget(format!(
                "Cargo target {name:?} is ambiguous across {} loaded crates",
                targets.len()
            ))),
        }
    }
}

/// Cheap immutable request snapshot spanning all local targets. Its streaming layer selects the
/// exact target from rust-analyzer's file/module mapping; ambiguous new files degrade to Unknown
/// rather than guessing a crate and rejecting valid output.
pub struct RustProjectRequestWorld {
    descriptor: AnalyzerLayerDescriptor,
    receipt: LayerReadinessReceipt,
    db: RootDatabase,
    targets: Vec<RustTarget>,
    file_targets: BTreeMap<String, Vec<String>>,
    deep_profile: RustDeepProfile,
    deep_state: Option<std::sync::Arc<std::sync::Mutex<RustDeepState>>>,
    /// Prevents the cached database from being revised while any decoder can still query its
    /// Salsa snapshot. A later lifecycle stage falls back to an independent cold world instead.
    _epoch_lease: std::sync::Arc<()>,
}

impl RustProjectRequestWorld {
    pub fn descriptor(&self) -> &AnalyzerLayerDescriptor {
        &self.descriptor
    }

    pub fn readiness_receipt(&self) -> &LayerReadinessReceipt {
        &self.receipt
    }

    pub fn deep_profile(&self) -> &RustDeepProfile {
        &self.deep_profile
    }

    pub fn into_streaming_layer(
        self,
        max_source_bytes: usize,
        max_checkpoints: usize,
    ) -> pb_control_collar::CollarResult<RustWorkspaceStreamingLayer> {
        RustWorkspaceStreamingLayer::new(self, max_source_bytes, max_checkpoints)
    }

    fn target_for_path(&self, path: &pb_control_collar::mutation::LogicalPath) -> Option<Crate> {
        target_for_path_in_world(&self.db, &self.targets, &self.file_targets, path.as_str())
    }

    fn request_for_target(&self, target: Crate) -> RustRequestWorld {
        RustRequestWorld {
            descriptor: self.descriptor.clone(),
            receipt: self.receipt.clone(),
            db: self.db.clone(),
            target,
            deep_exact: self.deep_profile.is_exact(),
        }
    }

    fn deep_diagnostic_delta(
        &self,
        candidates: &[(pb_control_collar::mutation::LogicalPath, Vec<u8>)],
    ) -> Result<Vec<RustDeepDiagnostic>, RustDeepUnknownReason> {
        let RustDeepProfile::Exact = &self.deep_profile else {
            return Err(self.deep_profile_unknown_reason());
        };
        let deep_state = self
            .deep_state
            .as_ref()
            .ok_or_else(|| self.deep_profile_unknown_reason())?;
        let mut deep_state = deep_state
            .lock()
            .map_err(|_| RustDeepUnknownReason::DependencyManifestUnavailable)?;
        let mut candidate_change = ChangeWithProcMacros::default();
        let mut restore_change = ChangeWithProcMacros::default();
        let mut candidate_sources = deep_state.baseline_sources.clone();
        let mut changed = BTreeSet::new();
        for (path, bytes) in candidates {
            if !changed.insert(path.as_str()) {
                return Err(RustDeepUnknownReason::SourceTopologyChanged);
            }
            let Some(file_id) = deep_state.file_ids.get(path.as_str()).copied() else {
                // Adding/deleting source-root topology requires a cold project reload. The
                // request-local overlay is intentionally limited to already indexed files.
                return Err(RustDeepUnknownReason::SourceTopologyChanged);
            };
            let text = std::str::from_utf8(bytes)
                .map_err(|_| RustDeepUnknownReason::DependencyManifestUnavailable)?;
            let baseline = deep_state
                .baseline_sources
                .get(path.as_str())
                .ok_or(RustDeepUnknownReason::SourceTopologyChanged)?;
            candidate_change.change_file(file_id, Some(text.to_string()));
            restore_change.change_file(file_id, Some(baseline.clone()));
            candidate_sources.insert(path.as_str().to_string(), text.to_string());
        }
        deep_state.db.apply_change(candidate_change);
        let candidate_imports = rust_deep_import_states(
            &deep_state.db,
            &deep_state.targets,
            &deep_state.file_targets,
            &candidate_sources,
        );
        let import_delta = rust_deep_import_delta(
            &deep_state.baseline_sources,
            &candidate_sources,
            &deep_state.baseline_imports,
            &candidate_imports,
        );
        let (candidate, _) = rust_deep_diagnostic_counts(&deep_state.db, &deep_state.targets);
        deep_state.db.apply_change(restore_change);
        let mut diagnostics = candidate
            .into_iter()
            .filter_map(|(diagnostic, count)| {
                let baseline = deep_state
                    .baseline_diagnostics
                    .get(&diagnostic)
                    .copied()
                    .unwrap_or_default();
                (count > baseline).then_some(diagnostic)
            })
            .collect::<Vec<_>>();
        if import_delta? && !diagnostics.contains(&RustDeepDiagnostic::UnresolvedImport) {
            diagnostics.push(RustDeepDiagnostic::UnresolvedImport);
        }
        diagnostics.sort_unstable();
        Ok(diagnostics)
    }

    /// Content-free promotion evidence for the checked-in semantic qualifier. Production
    /// generation and execution use the same private implementation through `finalize`; this
    /// method exposes only diagnostic classes (or a conservative unknown reason), never source.
    pub fn qualification_diagnostic_delta(
        &self,
        candidates: &[(pb_control_collar::mutation::LogicalPath, Vec<u8>)],
    ) -> Result<Vec<RustDeepDiagnostic>, RustDeepUnknownReason> {
        self.deep_diagnostic_delta(candidates)
    }

    fn deep_profile_unknown_reason(&self) -> RustDeepUnknownReason {
        match &self.deep_profile {
            RustDeepProfile::Exact => RustDeepUnknownReason::DependencyManifestUnavailable,
            RustDeepProfile::Partial(reasons) => reasons
                .first()
                .copied()
                .unwrap_or(RustDeepUnknownReason::DependencyManifestUnavailable),
        }
    }
}

/// Warm immutable state for one explicitly selected target. Kept as a focused public API for
/// embedders that already know the Cargo target; pb's workspace lifecycle uses `RustProjectWorld`.
pub struct RustSemanticWorld {
    descriptor: AnalyzerLayerDescriptor,
    receipt: LayerReadinessReceipt,
    readiness: LayerReadiness,
    db: RootDatabase,
    target: Crate,
    deep_exact: bool,
}

impl RustSemanticWorld {
    pub fn load_and_prime(config: RustLayerConfig) -> Result<Self, RustLayerError> {
        config.validate()?;
        let project = RustProjectWorld::load_and_prime(RustProjectConfig::from(&config))?;
        let target = project.target_by_name(&config.target_crate)?.krate;
        let deep_exact = project.deep_profile.is_exact();
        Ok(Self {
            descriptor: project.descriptor,
            receipt: project.receipt,
            readiness: project.readiness,
            db: project.db,
            target,
            deep_exact,
        })
    }

    pub fn descriptor(&self) -> &AnalyzerLayerDescriptor {
        &self.descriptor
    }

    pub fn readiness(&self) -> LayerReadiness {
        self.readiness
    }

    pub fn readiness_receipt(&self) -> &LayerReadinessReceipt {
        &self.receipt
    }

    pub fn mark_stale(&mut self) {
        self.readiness = LayerReadiness::Stale;
    }

    pub fn snapshot_for_request(
        &self,
        expected_world: &SemanticWorldId,
    ) -> Result<RustRequestWorld, RustLayerError> {
        ensure_ready_world(self.readiness, &self.descriptor.world, expected_world)?;
        Ok(RustRequestWorld {
            descriptor: self.descriptor.clone(),
            receipt: warm_receipt(&self.receipt),
            db: self.db.clone(),
            target: self.target,
            deep_exact: self.deep_exact,
        })
    }
}

/// Request-local Salsa snapshot. Streaming checkpoints and generated-source overlays will live on
/// this object; the cached warm world is never mutated by inference.
pub struct RustRequestWorld {
    descriptor: AnalyzerLayerDescriptor,
    receipt: LayerReadinessReceipt,
    db: RootDatabase,
    target: Crate,
    deep_exact: bool,
}

impl RustRequestWorld {
    pub fn descriptor(&self) -> &AnalyzerLayerDescriptor {
        &self.descriptor
    }

    pub fn readiness_receipt(&self) -> &LayerReadinessReceipt {
        &self.receipt
    }

    /// Resolve a Rust 2018+ absolute import path in the target's extern prelude. A positive result
    /// comes directly from HIR. Absence is authoritative only for rust-analyzer builtin/sysroot
    /// crates in this no-build-script/no-proc-macro profile.
    pub fn resolve_import(&self, path: &[&str]) -> RustImportResolution {
        let Some((root, rest)) = path.split_first() else {
            return RustImportResolution::Unknown(RustUnknownReason::UnsupportedPath);
        };
        let dependency = self
            .target
            .dependencies(&self.db)
            .into_iter()
            .find(|dependency| {
                dependency
                    .name
                    .display(&self.db, self.target.edition(&self.db))
                    .to_string()
                    == *root
            });
        if root == &"crate" {
            if rest.is_empty() {
                return RustImportResolution::Resolved(RustSymbolShape::module(
                    RustSemanticCertainty::Partial,
                ));
            }
            return self.resolve_from_module(self.target.root_module(&self.db), rest, false);
        }
        let Some(dependency) = dependency else {
            return RustImportResolution::Unknown(RustUnknownReason::PartialScope);
        };
        let dependency_is_immutable = dependency.krate.is_builtin(&self.db)
            || self.deep_exact && !dependency.krate.origin(&self.db).is_local();
        let certainty = if dependency_is_immutable {
            RustSemanticCertainty::Exact
        } else {
            RustSemanticCertainty::Partial
        };
        if rest.is_empty() {
            return RustImportResolution::Resolved(RustSymbolShape::module(certainty));
        }

        self.resolve_from_module(
            dependency.krate.root_module(&self.db),
            rest,
            dependency_is_immutable,
        )
    }

    fn resolve_from_module(
        &self,
        mut module: Module,
        segments: &[&str],
        complete: bool,
    ) -> RustImportResolution {
        for (index, segment) in segments.iter().enumerate() {
            let alternatives = self.visible_definitions(module, segment);
            let final_segment = index.saturating_add(1) == segments.len();
            if final_segment {
                if alternatives.is_empty() {
                    return if complete {
                        RustImportResolution::Absent
                    } else {
                        RustImportResolution::Unknown(RustUnknownReason::PartialScope)
                    };
                }
                return RustImportResolution::Resolved(merge_shapes(alternatives.into_iter().map(
                    |definition| {
                        symbol_shape(
                            &self.db,
                            definition,
                            if complete {
                                RustSemanticCertainty::Exact
                            } else {
                                RustSemanticCertainty::Partial
                            },
                        )
                    },
                )));
            }
            let Some(next) = alternatives
                .into_iter()
                .find_map(|definition| match definition {
                    ModuleDef::Module(module) => Some(module),
                    _ => None,
                })
            else {
                return if complete {
                    RustImportResolution::Absent
                } else {
                    RustImportResolution::Unknown(RustUnknownReason::PartialScope)
                };
            };
            module = next;
        }
        RustImportResolution::Unknown(RustUnknownReason::UnsupportedPath)
    }

    fn visible_definitions(&self, module: Module, wanted: &str) -> Vec<ModuleDef> {
        let edition = module.krate(&self.db).edition(&self.db);
        module
            .scope(&self.db, Some(self.target.root_module(&self.db)))
            .into_iter()
            .filter_map(|(name, definition)| {
                (name.display(&self.db, edition).to_string() == wanted).then_some(definition)
            })
            .filter_map(|definition| match definition {
                ScopeDef::ModuleDef(definition) => Some(definition),
                _ => None,
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RustImportResolution {
    Resolved(RustSymbolShape),
    Absent,
    Unknown(RustUnknownReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RustUnknownReason {
    PartialScope,
    UnsupportedPath,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RustSymbolShape {
    pub certainty: RustSemanticCertainty,
    pub kinds: Vec<RustSymbolKind>,
    pub callables: Vec<RustCallableShape>,
}

impl RustSymbolShape {
    fn module(certainty: RustSemanticCertainty) -> Self {
        Self {
            certainty,
            kinds: vec![RustSymbolKind::Module],
            callables: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RustSemanticCertainty {
    /// Exact under the pinned project/configuration profile and safe to use for hard rejection.
    Exact,
    /// Positive HIR information that may vary under unexecuted build scripts, procedural macros,
    /// or other intentionally unsupported configuration inputs. It can steer but cannot prune.
    Partial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RustSymbolKind {
    Module,
    Function,
    Type,
    Trait,
    Constant,
    Static,
    Value,
    Macro,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RustCallableShape {
    pub parameters: Vec<RustTypeShape>,
    /// `None` means rust-analyzer's public HIR surface did not prove whether this is variadic.
    pub accepts_extra_arguments: Option<bool>,
    pub result: RustTypeShape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RustTypeShape {
    Boolean,
    Integer,
    Float,
    StringSlice,
    Unit,
    Unknown,
}

fn symbol_shape(
    db: &RootDatabase,
    definition: ModuleDef,
    certainty: RustSemanticCertainty,
) -> RustSymbolShape {
    let (kind, callable) = match definition {
        ModuleDef::Module(_) => (RustSymbolKind::Module, None),
        ModuleDef::Function(function) => {
            (RustSymbolKind::Function, Some(callable_shape(db, function)))
        }
        ModuleDef::Adt(_) | ModuleDef::TypeAlias(_) | ModuleDef::BuiltinType(_) => {
            (RustSymbolKind::Type, None)
        }
        ModuleDef::Trait(_) => (RustSymbolKind::Trait, None),
        ModuleDef::Const(_) => (RustSymbolKind::Constant, None),
        ModuleDef::Static(_) => (RustSymbolKind::Static, None),
        ModuleDef::EnumVariant(_) => (RustSymbolKind::Value, None),
        ModuleDef::Macro(_) => (RustSymbolKind::Macro, None),
    };
    RustSymbolShape {
        certainty,
        kinds: vec![kind],
        callables: callable.into_iter().collect(),
    }
}

fn callable_shape(db: &RootDatabase, function: Function) -> RustCallableShape {
    // Generic signatures in the next trait solver consult rust-analyzer's thread-local database
    // interner. Public HIR calls usually attach it through a query boundary, but direct shape
    // extraction is one of the APIs where the caller must establish that scope explicitly.
    ra_ap_hir_ty::attach_db(db, || RustCallableShape {
        parameters: function
            .params_without_self(db)
            .iter()
            .map(|parameter| type_shape(parameter.ty()))
            .collect(),
        accepts_extra_arguments: None,
        result: type_shape(&function.ret_type(db)),
    })
}

fn type_shape(ty: &Type<'_>) -> RustTypeShape {
    if ty.is_unit() {
        RustTypeShape::Unit
    } else if ty.is_bool() {
        RustTypeShape::Boolean
    } else if ty.is_int_or_uint() {
        RustTypeShape::Integer
    } else if ty.is_float() {
        RustTypeShape::Float
    } else if ty.is_str() || ty.strip_references().is_str() {
        RustTypeShape::StringSlice
    } else {
        RustTypeShape::Unknown
    }
}

fn merge_shapes(shapes: impl IntoIterator<Item = RustSymbolShape>) -> RustSymbolShape {
    let mut certainty = RustSemanticCertainty::Exact;
    let mut kinds = Vec::new();
    let mut callables = Vec::new();
    for shape in shapes {
        if shape.certainty == RustSemanticCertainty::Partial {
            certainty = RustSemanticCertainty::Partial;
        }
        kinds.extend(shape.kinds);
        callables.extend(shape.callables);
    }
    kinds.sort_unstable();
    kinds.dedup();
    RustSymbolShape {
        certainty,
        kinds,
        callables,
    }
}

fn semantic_world_id(config: &RustProjectConfig) -> SemanticWorldId {
    SemanticWorldId {
        provider: "rust-analyzer-native".to_string(),
        provider_version: RUST_ANALYZER_VERSION.to_string(),
        world_sha256: config.world_sha256.clone(),
        configuration_sha256: config.configuration_sha256.clone(),
        dependency_sha256: config.dependency_sha256.clone(),
    }
}

fn warm_receipt(receipt: &LayerReadinessReceipt) -> LayerReadinessReceipt {
    LayerReadinessReceipt {
        origin: ReadinessOrigin::WarmCache,
        load_millis: 0,
        prime_millis: 0,
        ..receipt.clone()
    }
}

fn ensure_ready_world(
    readiness: LayerReadiness,
    actual: &SemanticWorldId,
    expected: &SemanticWorldId,
) -> Result<(), RustLayerError> {
    if readiness != LayerReadiness::Ready {
        return Err(RustLayerError::NotReady(readiness));
    }
    if expected != actual {
        return Err(RustLayerError::WorldChanged);
    }
    Ok(())
}

/// Build one independently writable Salsa storage from an already loaded project. A
/// `RootDatabase::clone()` is a read snapshot sharing Salsa storage; mutating it while the warm
/// world remains alive deadlocks in Salsa's cancellation barrier. Copying inputs once during the
/// pre-inference readiness phase gives final transaction replay its own revision stream without
/// rerunning Cargo or touching the live workspace.
fn independent_deep_state(
    source: &RootDatabase,
    vfs: &Vfs,
    shadow_root: &Path,
) -> Result<RustDeepState, RustLayerError> {
    let mut root_ids = BTreeSet::new();
    for (file_id, _) in vfs.iter() {
        root_ids.insert(source.file_source_root(file_id).source_root_id(source));
    }
    if root_ids
        .iter()
        .enumerate()
        .any(|(index, root)| root.0 != u32::try_from(index).unwrap_or(u32::MAX))
    {
        return Err(RustLayerError::ProviderUnavailable(
            "rust-analyzer source-root identifiers are not contiguous".to_string(),
        ));
    }
    let roots = root_ids
        .into_iter()
        .map(|root| {
            source
                .source_root(root)
                .source_root(source)
                .as_ref()
                .clone()
        })
        .collect::<Vec<_>>();

    let source_crates = all_crates(source).to_vec();
    let mut graph = CrateGraphBuilder::default();
    let mut crate_ids = std::collections::HashMap::new();
    for &krate in &source_crates {
        let data = krate.data(source);
        let extra = krate.extra_data(source);
        let attrs = data
            .crate_attrs
            .iter()
            .map(|attr| {
                attr.strip_prefix("#![")
                    .and_then(|attr| attr.strip_suffix(']'))
                    .map(str::to_string)
                    .ok_or_else(|| {
                        RustLayerError::ProviderUnavailable(
                            "rust-analyzer returned a malformed crate attribute".to_string(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let builder_id = graph.add_crate_root(
            data.root_file_id,
            data.edition,
            extra.display_name.clone(),
            extra.version.clone(),
            krate.cfg_options(source).clone(),
            extra.potential_cfg_options.clone(),
            krate.env(source).clone(),
            data.origin.clone(),
            attrs,
            data.is_proc_macro,
            data.proc_macro_cwd.clone(),
            krate.workspace_data(source).clone(),
        );
        crate_ids.insert(krate, builder_id);
    }
    for &krate in &source_crates {
        let from = crate_ids[&krate];
        for dependency in &krate.data(source).dependencies {
            graph
                .add_dep(
                    from,
                    DependencyBuilder::with_prelude(
                        dependency.name.clone(),
                        crate_ids[&dependency.crate_id],
                        dependency.is_prelude(),
                        dependency.is_sysroot(),
                    ),
                )
                .map_err(|error| RustLayerError::ProviderUnavailable(error.to_string()))?;
        }
    }

    let mut change = ChangeWithProcMacros::default();
    change.set_roots(roots);
    for (file_id, _) in vfs.iter() {
        change.change_file(
            file_id,
            Some(source.file_text(file_id).text(source).to_string()),
        );
    }
    change.set_crate_graph(graph);
    let mut db = RootDatabase::new(None);
    db.apply_change(change);
    let targets = local_targets(&db, vfs, shadow_root)?;
    let file_targets = local_file_targets(&db, vfs, shadow_root, &targets);
    let file_ids = local_file_ids(vfs, shadow_root);
    let baseline_sources = file_ids
        .iter()
        .map(|(path, file_id)| (path.clone(), db.file_text(*file_id).text(&db).to_string()))
        .collect();
    Ok(RustDeepState {
        db,
        targets,
        file_targets,
        file_ids,
        baseline_sources,
        baseline_diagnostics: BTreeMap::new(),
        baseline_imports: BTreeMap::new(),
    })
}

fn rust_deep_profile(db: &RootDatabase, vfs: &Vfs, targets: &[RustTarget]) -> RustDeepProfile {
    let mut reasons = BTreeSet::new();
    let mut inspected_manifests = HashSet::new();
    let mut visited = HashSet::new();
    let mut pending = targets
        .iter()
        .map(|target| target.krate)
        .collect::<Vec<_>>();

    while let Some(krate) = pending.pop() {
        if !visited.insert(krate) {
            continue;
        }
        let data = krate.base().data(db);
        if data.is_proc_macro {
            reasons.insert(RustDeepUnknownReason::ProceduralMacroDisabled);
        }
        pending.extend(
            krate
                .dependencies(db)
                .into_iter()
                .map(|dependency| dependency.krate),
        );

        if matches!(krate.origin(db), CrateOrigin::Lang(_)) {
            continue;
        }
        let Some(root) = vfs
            .file_path(krate.root_file(db))
            .as_path()
            .map(|path| PathBuf::from(AsRef::<Path>::as_ref(path)))
        else {
            reasons.insert(RustDeepUnknownReason::DependencyManifestUnavailable);
            continue;
        };
        let Some(manifest) = nearest_cargo_manifest(&root) else {
            reasons.insert(RustDeepUnknownReason::DependencyManifestUnavailable);
            continue;
        };
        if !inspected_manifests.insert(manifest.clone()) {
            continue;
        }
        match manifest_uses_build_script(&manifest) {
            Ok(true) => {
                reasons.insert(RustDeepUnknownReason::BuildScriptDisabled);
            }
            Ok(false) => {}
            Err(()) => {
                reasons.insert(RustDeepUnknownReason::DependencyManifestUnavailable);
            }
        }
    }

    if reasons.is_empty() {
        RustDeepProfile::Exact
    } else {
        RustDeepProfile::Partial(reasons.into_iter().collect())
    }
}

fn nearest_cargo_manifest(crate_root: &Path) -> Option<PathBuf> {
    crate_root.parent()?.ancestors().find_map(|directory| {
        let manifest = directory.join("Cargo.toml");
        manifest.is_file().then_some(manifest)
    })
}

fn manifest_uses_build_script(manifest: &Path) -> Result<bool, ()> {
    let source = std::fs::read_to_string(manifest).map_err(|_| ())?;
    let document = source.parse::<toml::Value>().map_err(|_| ())?;
    let package = document
        .get("package")
        .and_then(toml::Value::as_table)
        .ok_or(())?;
    match package.get("build") {
        Some(toml::Value::Boolean(false)) => Ok(false),
        Some(toml::Value::String(_)) | Some(toml::Value::Boolean(true)) => Ok(true),
        Some(_) => Err(()),
        None => Ok(manifest
            .parent()
            .is_some_and(|directory| directory.join("build.rs").is_file())),
    }
}

fn rust_deep_diagnostic_counts(
    db: &RootDatabase,
    targets: &[RustTarget],
) -> (RustDeepDiagnosticCounts, u64) {
    ra_ap_hir::attach_db(db, || {
        let mut counts = BTreeMap::new();
        let mut queries = 0u64;
        for target in targets {
            for module in target.krate.modules(db) {
                let mut diagnostics = Vec::new();
                for definition in module.declarations(db) {
                    if matches!(definition, ModuleDef::Module(_)) {
                        continue;
                    }
                    diagnostics.extend(definition.diagnostics(db, false));
                    queries = queries.saturating_add(1);
                    if let ModuleDef::Trait(trait_) = definition {
                        for item in trait_.items(db) {
                            item.diagnostics(db, &mut diagnostics, false);
                            queries = queries.saturating_add(1);
                        }
                    }
                }
                // Module-level diagnostics intentionally stay out of this safe profile: the HIR
                // entry point also performs whole-module impl coherence and trait-contract work,
                // which is neither bounded enough for request closure nor complete without the
                // compiler/build/macro profile. Impl item bodies still receive the same promoted
                // expression/type checks as free functions.
                for impl_ in module.impl_defs(db) {
                    for item in impl_.items(db) {
                        item.diagnostics(db, &mut diagnostics, false);
                        queries = queries.saturating_add(1);
                    }
                }
                for diagnostic in diagnostics {
                    if let Some(diagnostic) = promoted_rust_diagnostic(&diagnostic) {
                        let count = counts.entry(diagnostic).or_insert(0usize);
                        *count = count.saturating_add(1);
                    }
                }
            }
        }
        (counts, queries)
    })
}

fn rust_deep_import_states(
    db: &RootDatabase,
    targets: &[RustTarget],
    file_targets: &BTreeMap<String, Vec<String>>,
    sources: &BTreeMap<String, String>,
) -> RustDeepImportStates {
    sources
        .iter()
        .map(|(path, source)| {
            let paths = stream::complete_use_paths(source.as_bytes())
                .ok()
                .map(|paths| {
                    let target = target_for_path_in_world(db, targets, file_targets, path);
                    let mut states = BTreeMap::new();
                    for path in paths {
                        let resolution = target
                            .map_or(RustDeepImportResolution::Unknown, |target| {
                                resolve_complete_import(db, target, &path)
                            });
                        let counts = states
                            .entry(path)
                            .or_insert_with(RustDeepImportCounts::default);
                        match resolution {
                            RustDeepImportResolution::Resolved => {
                                counts.resolved = counts.resolved.saturating_add(1);
                            }
                            RustDeepImportResolution::Absent => {
                                counts.absent = counts.absent.saturating_add(1);
                            }
                            RustDeepImportResolution::Unknown => {
                                counts.unknown = counts.unknown.saturating_add(1);
                            }
                        }
                    }
                    states
                });
            (path.clone(), paths)
        })
        .collect()
}

fn rust_deep_import_delta(
    baseline_sources: &BTreeMap<String, String>,
    candidate_sources: &BTreeMap<String, String>,
    baseline: &RustDeepImportStates,
    candidate: &RustDeepImportStates,
) -> Result<bool, RustDeepUnknownReason> {
    let mut introduced_absence = false;
    for (file, candidate_state) in candidate {
        let Some(candidate_state) = candidate_state else {
            if candidate_sources.get(file) != baseline_sources.get(file) {
                return Err(RustDeepUnknownReason::ImportResolutionUnsupported);
            }
            continue;
        };
        let Some(Some(baseline_state)) = baseline.get(file) else {
            // Existing syntax debt made the baseline import set unknowable. A repaired file may
            // still receive HIR diagnostics, but import absence cannot safely prune this request.
            continue;
        };
        for (path, candidate_counts) in candidate_state {
            let baseline_counts = baseline_state.get(path).copied().unwrap_or_default();
            if candidate_counts.unknown > baseline_counts.unknown {
                return Err(RustDeepUnknownReason::ImportResolutionUnsupported);
            }
            introduced_absence |= candidate_counts.absent > baseline_counts.absent;
        }
    }
    Ok(introduced_absence)
}

fn resolve_complete_import(
    db: &RootDatabase,
    target: Crate,
    segments: &[String],
) -> RustDeepImportResolution {
    let Some((root, rest)) = segments.split_first() else {
        return RustDeepImportResolution::Unknown;
    };
    if root == "self" || root == "super" {
        return RustDeepImportResolution::Unknown;
    }
    if root == "crate" {
        return resolve_complete_import_from_module(db, target, target.root_module(db), rest);
    }

    let local = visible_module_definitions(db, target, target.root_module(db), root);
    let dependency = target
        .dependencies(db)
        .into_iter()
        .find(|dependency| dependency.name.display(db, target.edition(db)).to_string() == *root);
    match (local.as_slice(), dependency) {
        ([], None) => RustDeepImportResolution::Absent,
        ([], Some(dependency)) => {
            resolve_complete_import_from_module(db, target, dependency.krate.root_module(db), rest)
        }
        ([definition], None) if rest.is_empty() => {
            let _ = definition;
            RustDeepImportResolution::Resolved
        }
        (definitions, None) => {
            let modules = definitions
                .iter()
                .filter_map(|definition| match definition {
                    ModuleDef::Module(module) => Some(*module),
                    _ => None,
                })
                .collect::<Vec<_>>();
            match modules.as_slice() {
                [module] => resolve_complete_import_from_module(db, target, *module, rest),
                [] => RustDeepImportResolution::Absent,
                _ => RustDeepImportResolution::Unknown,
            }
        }
        // A local item colliding with an extern-prelude name requires the compiler's exact path
        // precedence. Do not manufacture a negative fact at the final gate.
        (_, Some(_)) => RustDeepImportResolution::Unknown,
    }
}

fn resolve_complete_import_from_module(
    db: &RootDatabase,
    target: Crate,
    mut module: Module,
    segments: &[String],
) -> RustDeepImportResolution {
    if segments.is_empty() {
        return RustDeepImportResolution::Resolved;
    }
    for (index, segment) in segments.iter().enumerate() {
        let definitions = visible_module_definitions(db, target, module, segment);
        let final_segment = index.saturating_add(1) == segments.len();
        if final_segment {
            return if definitions.is_empty() {
                RustDeepImportResolution::Absent
            } else {
                RustDeepImportResolution::Resolved
            };
        }
        let modules = definitions
            .into_iter()
            .filter_map(|definition| match definition {
                ModuleDef::Module(module) => Some(module),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [next] = modules.as_slice() else {
            return if modules.is_empty() {
                RustDeepImportResolution::Absent
            } else {
                RustDeepImportResolution::Unknown
            };
        };
        module = *next;
    }
    RustDeepImportResolution::Unknown
}

fn visible_module_definitions(
    db: &RootDatabase,
    target: Crate,
    module: Module,
    wanted: &str,
) -> Vec<ModuleDef> {
    let edition = module.krate(db).edition(db);
    module
        .scope(db, Some(target.root_module(db)))
        .into_iter()
        .filter_map(|(name, definition)| {
            (name.display(db, edition).to_string() == wanted).then_some(definition)
        })
        .filter_map(|definition| match definition {
            ScopeDef::ModuleDef(definition) => Some(definition),
            _ => None,
        })
        .collect()
}

fn target_for_path_in_world(
    db: &RootDatabase,
    targets: &[RustTarget],
    file_targets: &BTreeMap<String, Vec<String>>,
    path: &str,
) -> Option<Crate> {
    if let Some(ids) = file_targets.get(path) {
        let matches = targets
            .iter()
            .filter(|target| ids.contains(&target.descriptor.id))
            .collect::<Vec<_>>();
        if let [target] = matches.as_slice() {
            return Some(target.krate);
        }
    }

    let mut scoped = targets
        .iter()
        .filter(|target| path_is_in_scope(path, &target.descriptor.package_scope))
        .collect::<Vec<_>>();
    let longest = scoped
        .iter()
        .map(|target| target.descriptor.package_scope.len())
        .max()?;
    scoped.retain(|target| target.descriptor.package_scope.len() == longest);
    let first = scoped.first()?;
    scoped
        .iter()
        .all(|target| same_dependency_surface(db, first.krate, target.krate))
        .then_some(first.krate)
}

fn promoted_rust_diagnostic(diagnostic: &AnyDiagnostic<'_>) -> Option<RustDeepDiagnostic> {
    Some(match diagnostic {
        AnyDiagnostic::UnresolvedIdent(_) => RustDeepDiagnostic::UnresolvedName,
        AnyDiagnostic::UnresolvedImport(_)
        | AnyDiagnostic::UnresolvedExternCrate(_)
        | AnyDiagnostic::UnresolvedModule(_) => RustDeepDiagnostic::UnresolvedImport,
        AnyDiagnostic::MissingFields(_)
        | AnyDiagnostic::NoSuchField(_)
        | AnyDiagnostic::UnresolvedField(_) => RustDeepDiagnostic::MissingField,
        AnyDiagnostic::UnresolvedAssocItem(_) | AnyDiagnostic::UnresolvedMethodCall(_) => {
            RustDeepDiagnostic::MissingMethod
        }
        AnyDiagnostic::PrivateAssocItem(_) | AnyDiagnostic::PrivateField(_) => {
            RustDeepDiagnostic::Privacy
        }
        AnyDiagnostic::CannotBeDereferenced(_)
        | AnyDiagnostic::CannotImplicitlyDerefTraitObject(_)
        | AnyDiagnostic::CannotIndexInto(_)
        | AnyDiagnostic::CastToUnsized(_)
        | AnyDiagnostic::ExpectedArrayOrSlicePat(_)
        | AnyDiagnostic::InvalidCast(_)
        | AnyDiagnostic::InvalidRangePatType(_)
        | AnyDiagnostic::MethodCallIllegalSizedBound(_)
        | AnyDiagnostic::TypeMismatch(_)
        | AnyDiagnostic::TypeMustBeKnown(_) => RustDeepDiagnostic::TypeMismatch,
        AnyDiagnostic::UnimplementedTrait(_) => RustDeepDiagnostic::TraitContract,
        AnyDiagnostic::ExpectedFunction(_)
        | AnyDiagnostic::GenericArgsProhibited(_)
        | AnyDiagnostic::IncorrectGenericsLen(_)
        | AnyDiagnostic::IncorrectGenericsOrder(_)
        | AnyDiagnostic::MismatchedArgCount(_)
        | AnyDiagnostic::ParenthesizedGenericArgsWithoutFnTrait(_) => {
            RustDeepDiagnostic::InvalidCall
        }
        AnyDiagnostic::InvalidLhsOfAssignment(_)
        | AnyDiagnostic::MutableRefBinding(_)
        | AnyDiagnostic::MutRefInImmRefPat(_)
        | AnyDiagnostic::NeedMut(_) => RustDeepDiagnostic::Mutability,
        AnyDiagnostic::MovedOutOfRef(_) => RustDeepDiagnostic::Ownership,
        _ => return None,
    })
}

fn local_targets(
    db: &RootDatabase,
    vfs: &Vfs,
    shadow_root: &Path,
) -> Result<Vec<RustTarget>, RustLayerError> {
    let mut targets = Vec::new();
    for krate in Crate::all(db) {
        let path = vfs
            .file_path(krate.root_file(db))
            .as_path()
            .map(|path| PathBuf::from(AsRef::<Path>::as_ref(path)));
        let Some(path) = path.filter(|path| path.starts_with(shadow_root)) else {
            continue;
        };
        let relative = relative_string(shadow_root, &path)?;
        let crate_name = krate
            .display_name(db)
            .map(|name| name.to_string())
            .unwrap_or_else(|| relative.clone());
        let package_scope = cargo_package_scope(shadow_root, &path)?;
        targets.push(RustTarget {
            descriptor: RustTargetDescriptor {
                id: format!("{crate_name}@{relative}"),
                crate_name,
                package_scope,
                crate_root: relative,
            },
            krate,
        });
    }
    targets.sort_by(|left, right| left.descriptor.id.cmp(&right.descriptor.id));
    Ok(targets)
}

fn local_file_targets(
    db: &RootDatabase,
    vfs: &Vfs,
    shadow_root: &Path,
    targets: &[RustTarget],
) -> BTreeMap<String, Vec<String>> {
    let semantics = Semantics::new(db);
    let mut files = BTreeMap::<String, Vec<String>>::new();
    for (file_id, vfs_path) in vfs.iter() {
        let Some(path) = vfs_path
            .as_path()
            .map(|path| PathBuf::from(AsRef::<Path>::as_ref(path)))
            .filter(|path| path.starts_with(shadow_root))
        else {
            continue;
        };
        let Ok(relative) = relative_string(shadow_root, &path) else {
            continue;
        };
        let target_ids = semantics
            .file_to_module_defs(file_id)
            .filter_map(|module| {
                let krate = module.krate(db);
                targets
                    .iter()
                    .find(|target| target.krate == krate)
                    .map(|target| target.descriptor.id.clone())
            })
            .collect::<Vec<_>>();
        if !target_ids.is_empty() {
            let entry = files.entry(relative).or_default();
            entry.extend(target_ids);
            entry.sort();
            entry.dedup();
        }
    }
    files
}

fn local_file_ids(vfs: &Vfs, shadow_root: &Path) -> BTreeMap<String, FileId> {
    vfs.iter()
        .filter_map(|(file_id, vfs_path)| {
            let path = vfs_path
                .as_path()
                .map(|path| PathBuf::from(AsRef::<Path>::as_ref(path)))?;
            path.starts_with(shadow_root).then(|| {
                relative_string(shadow_root, &path)
                    .ok()
                    .map(|path| (path, file_id))
            })?
        })
        .collect()
}

fn cargo_package_scope(shadow_root: &Path, crate_root: &Path) -> Result<String, RustLayerError> {
    let mut directory = crate_root.parent();
    while let Some(candidate) = directory.filter(|candidate| candidate.starts_with(shadow_root)) {
        if candidate.join("Cargo.toml").is_file() {
            return relative_string(shadow_root, candidate);
        }
        if candidate == shadow_root {
            break;
        }
        directory = candidate.parent();
    }
    Err(RustLayerError::MissingTarget(format!(
        "could not bind crate root {} to a Cargo manifest",
        crate_root.display()
    )))
}

fn relative_string(root: &Path, path: &Path) -> Result<String, RustLayerError> {
    let relative = path.strip_prefix(root).map_err(|_| {
        RustLayerError::InvalidConfig(format!(
            "Rust project path {} escaped immutable shadow {}",
            path.display(),
            root.display()
        ))
    })?;
    let relative = relative.to_str().ok_or_else(|| {
        RustLayerError::InvalidConfig("Rust project contains a non-UTF-8 path".to_string())
    })?;
    Ok(relative.replace('\\', "/"))
}

fn path_is_in_scope(path: &str, scope: &str) -> bool {
    scope.is_empty()
        || path == scope
        || path
            .strip_prefix(scope)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn same_dependency_surface(db: &RootDatabase, left: Crate, right: Crate) -> bool {
    fn names(db: &RootDatabase, krate: Crate) -> Vec<String> {
        let mut names = krate
            .dependencies(db)
            .into_iter()
            .map(|dependency| dependency.name.display(db, krate.edition(db)).to_string())
            .collect::<Vec<_>>();
        names.sort();
        names
    }
    left == right || names(db, left) == names(db, right)
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Debug, Error)]
pub enum RustLayerError {
    #[error("invalid Rust control-layer configuration: {0}")]
    InvalidConfig(String),
    #[error("rust-analyzer native world is unavailable: {0}")]
    ProviderUnavailable(String),
    #[error("rust-analyzer could not establish the requested target: {0}")]
    MissingTarget(String),
    #[error("Rust semantic world is not ready: {0:?}")]
    NotReady(LayerReadiness),
    #[error("Rust semantic world identity changed before request snapshot")]
    WorldChanged,
    #[error("Rust semantic world cannot be incrementally refreshed: {0}")]
    IncrementalRefresh(String),
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use pb_control_collar::{
        analysis::{
            AnalysisBoundary, ClosureVerdict, IncrementalAnalyzer, ProgramSnapshot, SourceEvent,
            SourceOrigin, Viability,
        },
        mutation::LogicalPath,
    };

    use super::*;

    fn write(path: &Path, contents: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    fn fixture() -> (tempfile::TempDir, RustLayerConfig) {
        let root = tempfile::tempdir().unwrap();
        write(
            &root.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"app\", \"dep\"]\nresolver = \"3\"\n",
        );
        write(
            &root.path().join("app/Cargo.toml"),
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\ndep = { path = \"../dep\" }\n",
        );
        write(
            &root.path().join("app/src/lib.rs"),
            "pub fn local() -> i32 { dep::api::add(1, 2) }\n",
        );
        write(
            &root.path().join("dep/Cargo.toml"),
            "[package]\nname = \"dep\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        );
        write(
            &root.path().join("dep/src/lib.rs"),
            concat!(
                "pub mod api {\n",
                "    pub fn add(left: i32, right: i32) -> i32 { left + right }\n",
                "    pub fn takes_int(value: i32) -> i32 { value }\n",
                "    pub struct Public { pub value: i32, hidden: i32 }\n",
                "    pub fn public() -> Public { Public { value: 1, hidden: 2 } }\n",
                "    pub trait Required { fn required(&self); }\n",
                "}\n",
            ),
        );
        let config = RustLayerConfig {
            contract_version: RUST_LAYER_CONTRACT_VERSION,
            shadow_root: root.path().to_path_buf(),
            target_crate: "app".to_string(),
            world_sha256: "a".repeat(64),
            configuration_sha256: "b".repeat(64),
            dependency_sha256: "c".repeat(64),
        };
        (root, config)
    }

    #[test]
    fn warm_world_resolves_dependency_shapes_without_incomplete_diagnostics() {
        let (_root, config) = fixture();
        let world = RustSemanticWorld::load_and_prime(config).unwrap();
        assert_eq!(world.readiness(), LayerReadiness::Ready);
        assert!(world.readiness_receipt().primed_queries > 0);

        let request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let resolution = request.resolve_import(&["dep", "api", "add"]);
        let RustImportResolution::Resolved(shape) = resolution else {
            panic!("expected resolved dependency function, got {resolution:?}");
        };
        assert_eq!(shape.kinds, vec![RustSymbolKind::Function]);
        assert_eq!(shape.certainty, RustSemanticCertainty::Partial);
        assert_eq!(shape.callables.len(), 1);
        assert_eq!(
            shape.callables[0].parameters,
            vec![RustTypeShape::Integer, RustTypeShape::Integer]
        );
        assert_eq!(shape.callables[0].result, RustTypeShape::Integer);

        assert_eq!(
            request.resolve_import(&["dep", "api", "missing"]),
            RustImportResolution::Unknown(RustUnknownReason::PartialScope)
        );
        let std_resolution = request.resolve_import(&["std", "collections", "BTreeMap"]);
        assert!(
            matches!(&std_resolution, RustImportResolution::Resolved(_)),
            "std resolution was {std_resolution:?}; dependencies were {:?}",
            request
                .target
                .dependencies(&request.db)
                .into_iter()
                .map(|dependency| dependency
                    .name
                    .display(&request.db, request.target.edition(&request.db))
                    .to_string())
                .collect::<Vec<_>>()
        );
        let RustImportResolution::Resolved(std_shape) = std_resolution else {
            unreachable!();
        };
        assert_eq!(std_shape.certainty, RustSemanticCertainty::Exact);
        assert_eq!(
            request.resolve_import(&["std", "collections", "DefinitelyMissing"]),
            RustImportResolution::Absent
        );
    }

    #[test]
    fn stale_or_identity_divergent_worlds_cannot_start_inference() {
        let (_root, config) = fixture();
        let mut world = RustSemanticWorld::load_and_prime(config).unwrap();
        let mut other = world.descriptor().world.clone();
        other.dependency_sha256 = "d".repeat(64);
        assert!(matches!(
            world.snapshot_for_request(&other),
            Err(RustLayerError::WorldChanged)
        ));

        world.mark_stale();
        assert!(matches!(
            world.snapshot_for_request(&world.descriptor().world),
            Err(RustLayerError::NotReady(LayerReadiness::Stale))
        ));
    }

    #[test]
    fn build_script_projects_defer_deep_rejection_without_executing_the_script() {
        let (root, config) = fixture();
        write(
            &root.path().join("dep/build.rs"),
            "fn main() { panic!(\"must never execute during semantic preparation\"); }\n",
        );
        let project = RustProjectWorld::load_and_prime(RustProjectConfig::from(&config)).unwrap();
        assert_eq!(
            project.deep_profile(),
            &RustDeepProfile::Partial(vec![RustDeepUnknownReason::BuildScriptDisabled])
        );
        let request = project
            .snapshot_for_request(&project.descriptor().world)
            .unwrap();
        assert_eq!(
            request.deep_diagnostic_delta(&[(
                LogicalPath::parse("app/src/lib.rs").unwrap(),
                b"pub fn broken() -> i32 { \"wrong\" }\n".to_vec(),
            )]),
            Err(RustDeepUnknownReason::BuildScriptDisabled)
        );
    }

    #[test]
    fn procedural_macro_projects_defer_deep_rejection_without_starting_a_macro_server() {
        let (root, config) = fixture();
        write(
            &root.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"app\", \"dep\", \"pm\"]\nresolver = \"3\"\n",
        );
        write(
            &root.path().join("app/Cargo.toml"),
            concat!(
                "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n",
                "[dependencies]\ndep = { path = \"../dep\" }\npm = { path = \"../pm\" }\n",
            ),
        );
        write(
            &root.path().join("pm/Cargo.toml"),
            concat!(
                "[package]\nname = \"pm\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n",
                "[lib]\nproc-macro = true\n",
            ),
        );
        write(
            &root.path().join("pm/src/lib.rs"),
            concat!(
                "extern crate proc_macro;\n",
                "use proc_macro::TokenStream;\n",
                "#[proc_macro]\n",
                "pub fn passthrough(input: TokenStream) -> TokenStream { input }\n",
            ),
        );
        let project = RustProjectWorld::load_and_prime(RustProjectConfig::from(&config)).unwrap();
        assert_eq!(
            project.deep_profile(),
            &RustDeepProfile::Partial(vec![RustDeepUnknownReason::ProceduralMacroDisabled])
        );
    }

    #[test]
    fn existing_sources_refresh_without_reloading_cargo() {
        let (_root, config) = fixture();
        let mut world = RustProjectWorld::load_and_prime((&config).into()).unwrap();
        let mut next = RustProjectConfig::from(&config);
        next.world_sha256 = "d".repeat(64);
        let active_request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        assert!(matches!(
            world.refresh_existing_sources(
                next.clone(),
                &[(
                    "app/src/lib.rs".to_string(),
                    b"pub fn local() -> i32 { 42 }\n".to_vec(),
                )],
            ),
            Err(RustLayerError::IncrementalRefresh(_))
        ));
        drop(active_request);
        world
            .refresh_existing_sources(
                next,
                &[(
                    "app/src/lib.rs".to_string(),
                    b"pub fn local() -> i32 { 42 }\n".to_vec(),
                )],
            )
            .unwrap();
        assert_eq!(world.readiness_receipt().origin, ReadinessOrigin::WarmCache);
        assert_eq!(world.readiness_receipt().load_millis, 0);
        assert_eq!(world.descriptor().world.world_sha256, "d".repeat(64));
    }

    #[test]
    fn stream_rejects_a_proven_invalid_import_and_rolls_back_the_candidate() {
        let (_root, config) = fixture();
        let world = RustSemanticWorld::load_and_prime(config).unwrap();
        let request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let mut layer = RustStreamingLayer::new(request, 16 * 1024, 16).unwrap();
        layer.begin(ProgramSnapshot::default()).unwrap();
        let path = LogicalPath::parse("src/lib.rs").unwrap();
        let language = LanguageId("rust".to_string());
        layer
            .apply(SourceEvent::BeginFile {
                path: &path,
                language: &language,
                mutation: pb_control_collar::mutation::MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"use std::collections::BTreeMap;\nuse dep::api::add;\nfn valid() { let _ = add(1, 2); }\n",
            })
            .unwrap();
        let valid = layer
            .apply(SourceEvent::Boundary(AnalysisBoundary::Statement))
            .unwrap();
        assert_eq!(valid.viability, Viability::Unknown);

        let checkpoint = layer.checkpoint().unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"fn wrong_type() { let _ = add(\"x\", 2); }\n",
            })
            .unwrap();
        let wrong_type = layer
            .apply(SourceEvent::Boundary(AnalysisBoundary::Statement))
            .unwrap();
        // A later file in the same patch can change a local dependency signature. Local-project
        // facts therefore steer during streaming and become authoritative only at tool closure.
        assert_eq!(wrong_type.viability, Viability::Unknown);

        layer.rollback(checkpoint).unwrap();
        let checkpoint = layer.checkpoint().unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"use std::collections::DefinitelyMissing;\n",
            })
            .unwrap();
        let rejected = layer
            .apply(SourceEvent::Boundary(AnalysisBoundary::Statement))
            .unwrap();
        assert_eq!(rejected.viability, Viability::Impossible);

        layer.rollback(checkpoint).unwrap();
        let restored = layer
            .apply(SourceEvent::Boundary(AnalysisBoundary::Statement))
            .unwrap();
        assert_eq!(restored.viability, Viability::Unknown);
    }

    #[test]
    fn project_layer_selects_file_target_and_rolls_back_across_file_boundaries() {
        let (_root, config) = fixture();
        let project = RustProjectWorld::load_and_prime(RustProjectConfig::from(&config)).unwrap();
        assert!(project.targets().len() >= 2);
        let request = project
            .snapshot_for_request(&project.descriptor().world)
            .unwrap();
        let mut layer = request.into_streaming_layer(16 * 1024, 32).unwrap();
        layer.begin(ProgramSnapshot::default()).unwrap();
        let baseline = layer.checkpoint().unwrap();
        let language = LanguageId("rust".to_string());
        let app = LogicalPath::parse("app/src/lib.rs").unwrap();
        layer
            .apply(SourceEvent::BeginFile {
                path: &app,
                language: &language,
                mutation: pb_control_collar::mutation::MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"use std::collections::DefinitelyMissing;\n",
            })
            .unwrap();
        let invalid = layer
            .apply(SourceEvent::Boundary(AnalysisBoundary::Statement))
            .unwrap();
        assert_eq!(invalid.viability, Viability::Impossible);

        layer.rollback(baseline).unwrap();
        let dependency = LogicalPath::parse("dep/src/lib.rs").unwrap();
        layer
            .apply(SourceEvent::BeginFile {
                path: &dependency,
                language: &language,
                mutation: pb_control_collar::mutation::MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"use std::collections::BTreeMap;\n",
            })
            .unwrap();
        let valid = layer
            .apply(SourceEvent::Boundary(AnalysisBoundary::Statement))
            .unwrap();
        assert_eq!(valid.viability, Viability::Valid);
    }

    #[test]
    fn complete_project_overlay_classifies_promoted_hir_diagnostics_and_repairs() {
        let (_root, config) = fixture();
        let project = RustProjectWorld::load_and_prime(RustProjectConfig::from(&config)).unwrap();
        assert_eq!(project.deep_profile(), &RustDeepProfile::Exact);
        let request = project
            .snapshot_for_request(&project.descriptor().world)
            .unwrap();
        let app = LogicalPath::parse("app/src/lib.rs").unwrap();
        let dep = LogicalPath::parse("dep/src/lib.rs").unwrap();

        let cases = [
            (
                "pub fn value() { definitely_missing(); }\n",
                RustDeepDiagnostic::UnresolvedName,
            ),
            (
                "pub fn value() { let _: i32 = \"text\"; }\n",
                RustDeepDiagnostic::TypeMismatch,
            ),
            (
                "pub fn value() { let _ = dep::api::takes_int(); }\n",
                RustDeepDiagnostic::InvalidCall,
            ),
            (
                "pub fn value() { let _ = \"text\".definitely_missing(); }\n",
                RustDeepDiagnostic::MissingMethod,
            ),
            (
                concat!(
                    "pub fn value() {\n",
                    "    let _ = dep::api::Public { value: 1, missing: 2 };\n",
                    "}\n",
                ),
                RustDeepDiagnostic::MissingField,
            ),
            (
                "pub fn value() { let _ = dep::api::public().hidden; }\n",
                RustDeepDiagnostic::Privacy,
            ),
            (
                "pub fn value() { let value = 1; value = 2; }\n",
                RustDeepDiagnostic::Mutability,
            ),
            (
                concat!(
                    "pub fn take(_: String) {}\n",
                    "pub fn value() {\n",
                    "    let value = String::new();\n",
                    "    let borrowed = &value;\n",
                    "    take(*borrowed);\n",
                    "}\n",
                ),
                RustDeepDiagnostic::Ownership,
            ),
            (
                concat!(
                    "pub fn needs<T: dep::api::Required>(_: T) {}\n",
                    "pub fn value() { needs(1_i32); }\n",
                ),
                RustDeepDiagnostic::TraitContract,
            ),
        ];
        for (source, expected) in cases {
            let diagnostics = request
                .deep_diagnostic_delta(&[(app.clone(), source.as_bytes().to_vec())])
                .unwrap();
            assert!(
                diagnostics.contains(&expected),
                "{source:?} produced {diagnostics:?}, expected {expected:?}"
            );
        }

        let app_source = b"use dep::api::new_api;\npub fn local() -> i32 { new_api() }\n".to_vec();
        let missing_import = request
            .deep_diagnostic_delta(&[(app.clone(), app_source.clone())])
            .unwrap();
        assert!(
            missing_import.contains(&RustDeepDiagnostic::UnresolvedImport),
            "missing import produced {missing_import:?}"
        );
        let dep_source = concat!(
            "pub mod api {\n",
            "    pub fn new_api() -> i32 { 42 }\n",
            "}\n",
        )
        .as_bytes()
        .to_vec();
        assert_eq!(
            request
                .deep_diagnostic_delta(&[(app, app_source), (dep, dep_source)])
                .unwrap(),
            Vec::<RustDeepDiagnostic>::new()
        );

        assert_eq!(
            request.deep_diagnostic_delta(&[(
                LogicalPath::parse("app/src/lib.rs").unwrap(),
                b"mod nested { use self::maybe_later; }\n".to_vec(),
            )]),
            Err(RustDeepUnknownReason::ImportResolutionUnsupported)
        );
    }

    #[test]
    fn complete_project_overlay_preserves_baseline_debt_after_native_refresh() {
        let (_root, config) = fixture();
        let mut project =
            RustProjectWorld::load_and_prime(RustProjectConfig::from(&config)).unwrap();
        let baseline = b"pub fn existing() -> i32 { \"baseline debt\" }\n".to_vec();
        let mut next = RustProjectConfig::from(&config);
        next.world_sha256 = "d".repeat(64);
        project
            .refresh_existing_sources(next, &[("app/src/lib.rs".to_string(), baseline.clone())])
            .unwrap();
        let request = project
            .snapshot_for_request(&project.descriptor().world)
            .unwrap();
        let app = LogicalPath::parse("app/src/lib.rs").unwrap();
        let preserved =
            b"pub fn existing() -> i32 { \"baseline debt\" }\npub fn valid() -> i32 { 42 }\n"
                .to_vec();
        assert!(
            request
                .deep_diagnostic_delta(&[(app.clone(), preserved)])
                .unwrap()
                .is_empty()
        );
        let introduced = b"pub fn existing() -> i32 { \"baseline debt\" }\npub fn new_error() -> i32 { \"new debt\" }\n".to_vec();
        assert_eq!(
            request.deep_diagnostic_delta(&[(app, introduced)]).unwrap(),
            vec![RustDeepDiagnostic::TypeMismatch]
        );
    }

    #[test]
    fn workspace_finalization_rejects_deep_errors_and_accepts_cross_file_repairs_after_rollback() {
        let (_root, config) = fixture();
        let project = RustProjectWorld::load_and_prime(RustProjectConfig::from(&config)).unwrap();
        let request = project
            .snapshot_for_request(&project.descriptor().world)
            .unwrap();
        let mut layer = request.into_streaming_layer(32 * 1024, 32).unwrap();
        layer.begin(ProgramSnapshot::default()).unwrap();
        let baseline = layer.checkpoint().unwrap();
        let language = LanguageId("rust".to_string());
        let app = LogicalPath::parse("app/src/lib.rs").unwrap();
        let dep = LogicalPath::parse("dep/src/lib.rs").unwrap();

        layer
            .apply(SourceEvent::BeginFile {
                path: &app,
                language: &language,
                mutation: pb_control_collar::mutation::MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"pub fn broken() -> i32 { \"wrong\" }\n",
            })
            .unwrap();
        layer.apply(SourceEvent::EndFile).unwrap();
        let rejected = layer.finalize().unwrap();
        assert_eq!(rejected.viability, Viability::Impossible);
        assert_eq!(rejected.closure, ClosureVerdict::Reject);
        assert!(
            rejected
                .obligations
                .iter()
                .any(|obligation| obligation.kind == "rust_deep_type_mismatch")
        );

        layer.rollback(baseline).unwrap();
        layer
            .apply(SourceEvent::BeginFile {
                path: &app,
                language: &language,
                mutation: pb_control_collar::mutation::MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"use dep::api::new_api;\npub fn local() -> i32 { new_api() }\n",
            })
            .unwrap();
        layer.apply(SourceEvent::EndFile).unwrap();
        layer
            .apply(SourceEvent::BeginFile {
                path: &dep,
                language: &language,
                mutation: pb_control_collar::mutation::MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"pub mod api { pub fn new_api() -> i32 { 42 } }\n",
            })
            .unwrap();
        layer.apply(SourceEvent::EndFile).unwrap();
        let repaired = layer.finalize().unwrap();
        assert_eq!(repaired.viability, Viability::Valid);
        assert_eq!(repaired.closure, ClosureVerdict::Allow);
    }
}
