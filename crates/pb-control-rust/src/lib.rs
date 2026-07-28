//! Rust-specific streaming control built directly on rust-analyzer internals.
//!
//! This crate owns Rust parsing, HIR name resolution, and Rust type shapes. The language-neutral
//! collar supplies virtual-source events and combines decisions; it does not flatten rust-analyzer
//! state into a cross-language type system.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

use pb_control_collar::analysis::{
    AnalyzerCapability, AnalyzerLayerDescriptor, LanguageId, LayerReadiness, LayerReadinessReceipt,
    ReadinessOrigin, SemanticCompleteness, SemanticWorldId,
};
use ra_ap_hir::{Crate, Function, Module, ModuleDef, ScopeDef, Semantics, Type};
use ra_ap_ide_db::{ChangeWithProcMacros, RootDatabase};
use ra_ap_load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};
use ra_ap_project_model::{CargoConfig, RustLibSource};
use ra_ap_vfs::{FileId, Vfs};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod stream;
pub use stream::{RustLayerCheckpoint, RustStreamingLayer, RustWorkspaceStreamingLayer};

pub const RUST_LAYER_CONTRACT_VERSION: u32 = 1;
pub const RUST_ANALYZER_VERSION: &str = "ra_ap_0.0.344";

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
    request_epoch: std::sync::Arc<()>,
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
        let prime_millis = elapsed_millis(prime_started);
        let descriptor = AnalyzerLayerDescriptor {
            id: "rust-native-v1".to_string(),
            language: LanguageId("rust".to_string()),
            world: world.clone(),
            capabilities: vec![
                AnalyzerCapability::PrefixStructural,
                AnalyzerCapability::SymbolResolution,
                AnalyzerCapability::TypeChecking,
                AnalyzerCapability::DependencyResolution,
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

    pub fn into_streaming_layer(
        self,
        max_source_bytes: usize,
        max_checkpoints: usize,
    ) -> pb_control_collar::CollarResult<RustWorkspaceStreamingLayer> {
        RustWorkspaceStreamingLayer::new(self, max_source_bytes, max_checkpoints)
    }

    fn target_for_path(&self, path: &pb_control_collar::mutation::LogicalPath) -> Option<Crate> {
        if let Some(ids) = self.file_targets.get(path.as_str()) {
            let matches = self
                .targets
                .iter()
                .filter(|target| ids.contains(&target.descriptor.id))
                .collect::<Vec<_>>();
            if let [target] = matches.as_slice() {
                return Some(target.krate);
            }
        }

        let mut scoped = self
            .targets
            .iter()
            .filter(|target| path_is_in_scope(path.as_str(), &target.descriptor.package_scope))
            .collect::<Vec<_>>();
        let longest = scoped
            .iter()
            .map(|target| target.descriptor.package_scope.len())
            .max()?;
        scoped.retain(|target| target.descriptor.package_scope.len() == longest);
        let first = scoped.first()?;
        scoped
            .iter()
            .all(|target| same_dependency_surface(&self.db, first.krate, target.krate))
            .then_some(first.krate)
    }

    fn request_for_target(&self, target: Crate) -> RustRequestWorld {
        RustRequestWorld {
            descriptor: self.descriptor.clone(),
            receipt: self.receipt.clone(),
            db: self.db.clone(),
            target,
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
}

impl RustSemanticWorld {
    pub fn load_and_prime(config: RustLayerConfig) -> Result<Self, RustLayerError> {
        config.validate()?;
        let project = RustProjectWorld::load_and_prime(RustProjectConfig::from(&config))?;
        let target = project.target_by_name(&config.target_crate)?.krate;
        Ok(Self {
            descriptor: project.descriptor,
            receipt: project.receipt,
            readiness: project.readiness,
            db: project.db,
            target,
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
        let certainty = if dependency.krate.is_builtin(&self.db) {
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
            dependency.krate.is_builtin(&self.db),
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
    RustCallableShape {
        parameters: function
            .params_without_self(db)
            .iter()
            .map(|parameter| type_shape(parameter.ty()))
            .collect(),
        accepts_extra_arguments: None,
        result: type_shape(&function.ret_type(db)),
    }
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
            AnalysisBoundary, IncrementalAnalyzer, ProgramSnapshot, SourceEvent, SourceOrigin,
            Viability,
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
            "pub mod api { pub fn add(left: i32, right: i32) -> i32 { left + right } }\n",
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
        // The no-build-script/no-proc-macro project profile has only partial certainty for
        // third-party signatures, so a mismatch can steer but cannot hard-prune.
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
}
