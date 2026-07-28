//! Python-specific streaming control built directly on Astral ty internals.
//!
//! The expensive project graph is loaded into a frozen in-memory filesystem and primed before
//! inference. Request checks use private copy-on-write overlays, so imports and type shapes are
//! resolved against one coherent project without querying a live LSP during token generation.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Instant,
};

use pb_control_collar::{
    CollarError, CollarResult,
    analysis::{
        Analysis, AnalysisBoundary, AnalyzerCapability, AnalyzerCheckpoint,
        AnalyzerLayerDescriptor, ClosureVerdict, IncrementalAnalyzer, LanguageId, LayerReadiness,
        LayerReadinessReceipt, ProgramSnapshot, ReadinessOrigin, SemanticCompleteness,
        SemanticObligation, SemanticWorldId, SourceEvent, SourceOrigin, Viability,
    },
    mutation::{LogicalPath, MutationKind},
};
use ruff_db::{
    Db as SourceDb,
    diagnostic::{Diagnostic, Severity},
    files::{File, Files, system_path_to_file},
    source::source_text,
    system::{MemoryFileSystem, System, SystemPathBuf},
    vendored::VendoredFileSystem,
};
use ruff_python_ast::PythonVersion;
use tree_sitter::{InputEdit, Node, Parser, Point, Tree};
use ty_module_resolver::{
    Db as ModuleResolverDb, FallibleStrategy, SearchPathSettings, SearchPaths,
};
use ty_python_core::{
    Db as PythonCoreDb,
    platform::PythonPlatform,
    program::{Program, ProgramSettings},
};
use ty_python_semantic::{
    AnalysisSettings, Db as SemanticDb, check_file_unwrap, default_lint_registry,
    lint::{LintRegistry, RuleSelection},
};
use ty_site_packages::{PythonVersionSource, PythonVersionWithSource};
use walkdir::WalkDir;

mod system;
use system::PythonSystem;

pub const PYTHON_LAYER_CONTRACT_VERSION: u32 = 2;
pub const TY_PROVIDER_VERSION: &str = "ty_0.0.6";
const VIRTUAL_PROJECT_ROOT: &str = "/workspace";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PythonProjectConfig {
    pub contract_version: u32,
    /// Controller-created immutable shadow of repository-owned files.
    pub shadow_root: PathBuf,
    /// First-party import roots relative to `shadow_root`. Empty selects the project root.
    pub first_party_roots: Vec<PathBuf>,
    /// Snapshotted dependency roots relative to `shadow_root` (for example `.venv/.../site-packages`).
    pub site_packages_roots: Vec<PathBuf>,
    /// True only when the controller proved that the selected static external import search space
    /// was captured completely. When false, missing absolute third-party imports remain unknown.
    pub external_imports_complete: bool,
    pub python_version: String,
    pub python_platform: String,
    pub world_sha256: String,
    pub configuration_sha256: String,
    pub dependency_sha256: String,
    pub max_files: usize,
    pub max_bytes: usize,
}

impl PythonProjectConfig {
    fn validate(&self) -> Result<(), PythonLayerError> {
        if self.contract_version != PYTHON_LAYER_CONTRACT_VERSION {
            return Err(PythonLayerError::InvalidConfig(format!(
                "Python layer contract version must be {PYTHON_LAYER_CONTRACT_VERSION}"
            )));
        }
        if !self.shadow_root.is_absolute() || !self.shadow_root.is_dir() {
            return Err(PythonLayerError::InvalidConfig(
                "Python shadow root must be an existing absolute directory".to_string(),
            ));
        }
        if self.max_files == 0 || self.max_bytes == 0 {
            return Err(PythonLayerError::InvalidConfig(
                "Python project bounds must be non-zero".to_string(),
            ));
        }
        for root in self
            .first_party_roots
            .iter()
            .chain(&self.site_packages_roots)
        {
            validate_relative_root(root)?;
            if !self.shadow_root.join(root).is_dir() {
                return Err(PythonLayerError::InvalidConfig(format!(
                    "Python search root {:?} is absent from the immutable shadow",
                    root
                )));
            }
        }
        let version = PythonVersion::from_str(&self.python_version)
            .map_err(|error| PythonLayerError::InvalidConfig(error.to_string()))?;
        if version < PythonVersion::PY310 || version > PythonVersion::latest_ty() {
            return Err(PythonLayerError::InvalidConfig(format!(
                "Python version {version} is outside the ty control profile"
            )));
        }
        if self.python_platform.is_empty() || self.python_platform.len() > 64 {
            return Err(PythonLayerError::InvalidConfig(
                "Python platform must be bounded and non-empty".to_string(),
            ));
        }
        for (label, digest) in [
            ("world", self.world_sha256.as_str()),
            ("configuration", self.configuration_sha256.as_str()),
            ("dependency", self.dependency_sha256.as_str()),
        ] {
            if !is_lower_hex_digest(digest) {
                return Err(PythonLayerError::InvalidConfig(format!(
                    "Python {label} identity must be a lowercase SHA-256 digest"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PythonDiagnostic {
    pub path: String,
    pub code: String,
    pub message: String,
    pub start: usize,
    pub end: usize,
    pub source_excerpt: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PythonCheckReport {
    pub baseline: Vec<PythonDiagnostic>,
    pub candidate: Vec<PythonDiagnostic>,
    pub introduced: Vec<PythonDiagnostic>,
}

pub struct PythonProjectWorld {
    descriptor: AnalyzerLayerDescriptor,
    receipt: LayerReadinessReceipt,
    readiness: LayerReadiness,
    factory: Arc<PythonDatabaseFactory>,
    baseline: Arc<BTreeMap<String, Vec<PythonDiagnostic>>>,
    first_party_files: Arc<BTreeMap<String, SystemPathBuf>>,
    dependency_files: Arc<BTreeMap<String, SystemPathBuf>>,
    promotion_policy: Arc<PythonPromotionPolicy>,
    request_epoch: Arc<()>,
}

impl PythonProjectWorld {
    pub fn load_and_prime(config: PythonProjectConfig) -> Result<Self, PythonLayerError> {
        config.validate()?;
        let load_started = Instant::now();
        let image = FrozenImage::capture(&config)?;
        let promotion_policy = Arc::new(PythonPromotionPolicy {
            controlled_modules: image.controlled_modules.clone(),
            external_imports_complete: config.external_imports_complete,
        });
        let version = PythonVersion::from_str(&config.python_version)
            .map_err(|error| PythonLayerError::InvalidConfig(error.to_string()))?;
        let factory = Arc::new(PythonDatabaseFactory {
            system: PythonSystem::from_base(image.fs),
            src_roots: virtual_roots(&config.first_party_roots),
            site_packages_roots: virtual_roots(&config.site_packages_roots),
            python_version: version,
            python_platform: PythonPlatform::from(config.python_platform.clone()),
        });
        let db = factory.build()?;
        let load_millis = elapsed_millis(load_started);

        let prime_started = Instant::now();
        let mut baseline = BTreeMap::new();
        let mut primed_queries = 0u64;
        for (logical, path) in &image.first_party_files {
            let file = system_path_to_file(&db, path).map_err(|error| {
                PythonLayerError::ProviderUnavailable(format!(
                    "failed to intern frozen Python file {logical}: {error}"
                ))
            })?;
            let diagnostics = diagnostic_records(&db, file, logical, check_file_unwrap(&db, file));
            baseline.insert(
                logical.clone(),
                promoted(diagnostics, promotion_policy.as_ref()),
            );
            primed_queries = primed_queries.saturating_add(1);
        }
        for (logical, path) in &image.dependency_files {
            let file = system_path_to_file(&db, path).map_err(|error| {
                PythonLayerError::ProviderUnavailable(format!(
                    "failed to intern frozen Python dependency {logical}: {error}"
                ))
            })?;
            let _ = check_file_unwrap(&db, file);
            primed_queries = primed_queries.saturating_add(1);
        }
        let prime_millis = elapsed_millis(prime_started);
        let world = SemanticWorldId {
            provider: "astral-ty-native".to_string(),
            provider_version: TY_PROVIDER_VERSION.to_string(),
            world_sha256: config.world_sha256,
            configuration_sha256: config.configuration_sha256,
            dependency_sha256: config.dependency_sha256,
        };
        let descriptor = AnalyzerLayerDescriptor {
            id: "python-native-v1".to_string(),
            language: LanguageId("python".to_string()),
            world: world.clone(),
            capabilities: vec![
                AnalyzerCapability::PrefixStructural,
                AnalyzerCapability::SymbolResolution,
                AnalyzerCapability::TypeChecking,
                AnalyzerCapability::DependencyResolution,
                AnalyzerCapability::FinalWorkspaceGate,
            ],
        };
        let receipt = LayerReadinessReceipt {
            world,
            origin: ReadinessOrigin::ColdBuild,
            completeness: SemanticCompleteness::Partial,
            load_millis,
            prime_millis,
            primed_queries,
        };
        Ok(Self {
            descriptor,
            receipt,
            readiness: LayerReadiness::Ready,
            factory,
            baseline: Arc::new(baseline),
            first_party_files: Arc::new(image.first_party_files),
            dependency_files: Arc::new(image.dependency_files),
            promotion_policy,
            request_epoch: Arc::new(()),
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
        self.request_epoch = Arc::new(());
    }

    pub fn snapshot_for_request(
        &self,
        expected: &SemanticWorldId,
    ) -> Result<PythonProjectRequestWorld, PythonLayerError> {
        if self.readiness != LayerReadiness::Ready || expected != &self.descriptor.world {
            return Err(PythonLayerError::StaleWorld);
        }
        // Salsa database clones are read snapshots and deliberately block writes. A request gets
        // an independently writable database over the already-frozen in-memory project instead;
        // all expensive queries are primed here, before inference, never at first mutation use.
        let db = self.factory.build()?;
        for (logical, path) in self.first_party_files.iter() {
            let file = system_path_to_file(&db, path).map_err(|error| {
                PythonLayerError::ProviderUnavailable(format!(
                    "failed to intern prepared Python file {logical}: {error}"
                ))
            })?;
            let _ = check_file_unwrap(&db, file);
        }
        for (logical, path) in self.dependency_files.iter() {
            let file = system_path_to_file(&db, path).map_err(|error| {
                PythonLayerError::ProviderUnavailable(format!(
                    "failed to intern prepared Python dependency {logical}: {error}"
                ))
            })?;
            let _ = check_file_unwrap(&db, file);
        }
        Ok(PythonProjectRequestWorld {
            descriptor: self.descriptor.clone(),
            receipt: self.receipt.clone(),
            db,
            baseline: Arc::clone(&self.baseline),
            first_party_files: Arc::clone(&self.first_party_files),
            promotion_policy: Arc::clone(&self.promotion_policy),
            applied_paths: BTreeSet::new(),
            request_epoch: Arc::clone(&self.request_epoch),
        })
    }
}

pub struct PythonProjectRequestWorld {
    descriptor: AnalyzerLayerDescriptor,
    receipt: LayerReadinessReceipt,
    db: PythonDatabase,
    baseline: Arc<BTreeMap<String, Vec<PythonDiagnostic>>>,
    first_party_files: Arc<BTreeMap<String, SystemPathBuf>>,
    promotion_policy: Arc<PythonPromotionPolicy>,
    applied_paths: BTreeSet<LogicalPath>,
    #[allow(dead_code)]
    request_epoch: Arc<()>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PythonCandidateMutation {
    Upsert(String),
    Delete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PythonCheckScope {
    Candidates,
    FrozenProject,
}

impl PythonProjectRequestWorld {
    #[cfg(test)]
    fn check_candidates(
        &mut self,
        candidates: &BTreeMap<LogicalPath, String>,
    ) -> Result<BTreeMap<String, PythonCheckReport>, PythonLayerError> {
        let candidates = candidates
            .iter()
            .map(|(path, source)| {
                (
                    path.clone(),
                    PythonCandidateMutation::Upsert(source.clone()),
                )
            })
            .collect();
        self.check_mutations(&candidates)
    }

    fn check_mutations(
        &mut self,
        candidates: &BTreeMap<LogicalPath, PythonCandidateMutation>,
    ) -> Result<BTreeMap<String, PythonCheckReport>, PythonLayerError> {
        self.check_mutations_in_scope(candidates, PythonCheckScope::Candidates)
    }

    fn check_project_mutations(
        &mut self,
        candidates: &BTreeMap<LogicalPath, PythonCandidateMutation>,
    ) -> Result<BTreeMap<String, PythonCheckReport>, PythonLayerError> {
        self.check_mutations_in_scope(candidates, PythonCheckScope::FrozenProject)
    }

    /// Return only the content-free promoted diagnostic codes introduced by a complete candidate
    /// transaction. This is the language-owned differential-qualification surface: production
    /// still authorizes execution through the ordinary streaming/final gates, while the harness
    /// can prove which exact promoted class caused that decision without persisting source,
    /// excerpts, messages, paths, or symbols.
    pub fn qualification_introduced_codes(
        &mut self,
        candidates: BTreeMap<LogicalPath, Option<String>>,
    ) -> Result<BTreeSet<String>, PythonLayerError> {
        let candidates = candidates
            .into_iter()
            .map(|(path, source)| {
                (
                    path,
                    source.map_or(
                        PythonCandidateMutation::Delete,
                        PythonCandidateMutation::Upsert,
                    ),
                )
            })
            .collect();
        Ok(self
            .check_project_mutations(&candidates)?
            .into_values()
            .flat_map(|report| report.introduced)
            .map(|diagnostic| diagnostic.code)
            .collect())
    }

    fn check_mutations_in_scope(
        &mut self,
        candidates: &BTreeMap<LogicalPath, PythonCandidateMutation>,
        scope: PythonCheckScope,
    ) -> Result<BTreeMap<String, PythonCheckReport>, PythonLayerError> {
        // Semantic probes are speculative. Restore every path touched by the previous probe to the
        // frozen base before applying the next complete candidate set, so rollback never leaks a
        // generated module or stale edit into another decoder branch.
        for logical in std::mem::take(&mut self.applied_paths) {
            let path = virtual_path(logical.as_str())?;
            self.db.system.reset(&path)?;
            File::sync_path(&mut self.db, &path);
        }
        for (logical, mutation) in candidates {
            let path = virtual_path(logical.as_str())?;
            match mutation {
                PythonCandidateMutation::Upsert(source) => {
                    self.db.system.put(&path, source.clone())?;
                }
                PythonCandidateMutation::Delete => self.db.system.delete(&path)?,
            }
            File::sync_path(&mut self.db, &path);
            self.applied_paths.insert(logical.clone());
        }
        let mut selected = BTreeMap::new();
        if scope == PythonCheckScope::FrozenProject {
            let deleted = candidates
                .iter()
                .filter_map(|(logical, mutation)| {
                    matches!(mutation, PythonCandidateMutation::Delete).then_some(logical.as_str())
                })
                .collect::<BTreeSet<_>>();
            for (logical, path) in self.first_party_files.iter() {
                if !deleted.contains(logical.as_str()) {
                    selected.insert(logical.clone(), path.clone());
                }
            }
        }
        for (logical, mutation) in candidates {
            if matches!(mutation, PythonCandidateMutation::Upsert(_)) {
                selected.insert(
                    logical.as_str().to_string(),
                    virtual_path(logical.as_str())?,
                );
            }
        }
        let mut files = BTreeMap::new();
        for (logical, path) in selected {
            let file = system_path_to_file(&self.db, &path).map_err(|error| {
                PythonLayerError::ProviderUnavailable(format!(
                    "failed to intern Python closure file {logical}: {error}",
                ))
            })?;
            files.insert(logical, file);
        }
        let mut reports = BTreeMap::new();
        for (logical, file) in files {
            let candidate = promoted(
                diagnostic_records(&self.db, file, &logical, check_file_unwrap(&self.db, file)),
                self.promotion_policy.as_ref(),
            );
            let baseline = self.baseline.get(&logical).cloned().unwrap_or_default();
            let introduced = multiset_difference(&candidate, &baseline);
            reports.insert(
                logical,
                PythonCheckReport {
                    baseline,
                    candidate,
                    introduced,
                },
            );
        }
        Ok(reports)
    }

    pub fn into_streaming_layer(
        self,
        max_source_bytes: usize,
        max_checkpoints: usize,
    ) -> CollarResult<PythonWorkspaceStreamingLayer> {
        PythonWorkspaceStreamingLayer::new(self, max_source_bytes, max_checkpoints)
    }
}

pub struct PythonWorkspaceStreamingLayer {
    project: PythonProjectRequestWorld,
    parser: Parser,
    active: Option<PythonFileState>,
    completed: BTreeMap<LogicalPath, PythonCandidateMutation>,
    snapshots: Vec<PythonStateSnapshot>,
    epoch: u64,
    max_source_bytes: usize,
    max_checkpoints: usize,
    generated_suppression: bool,
    last_analysis: Analysis,
}

impl PythonWorkspaceStreamingLayer {
    fn new(
        project: PythonProjectRequestWorld,
        max_source_bytes: usize,
        max_checkpoints: usize,
    ) -> CollarResult<Self> {
        if max_source_bytes == 0 || max_checkpoints == 0 {
            return Err(CollarError::Analysis(
                "Python streaming limits must be non-zero".to_string(),
            ));
        }
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .map_err(|error| {
                CollarError::Analysis(format!("failed to load pinned Python grammar: {error}"))
            })?;
        Ok(Self {
            project,
            parser,
            active: None,
            completed: BTreeMap::new(),
            snapshots: Vec::new(),
            epoch: 0,
            max_source_bytes,
            max_checkpoints,
            generated_suppression: false,
            last_analysis: repairable_analysis(),
        })
    }

    fn begin_file(&mut self, path: &LogicalPath, mutation: MutationKind) -> CollarResult<Analysis> {
        let tree = self.parser.parse(&[] as &[u8], None).ok_or_else(|| {
            CollarError::Analysis("pinned Python parser returned no initial tree".to_string())
        })?;
        self.active = Some(PythonFileState {
            path: path.clone(),
            mutation,
            source: Vec::new(),
            generated_ranges: Vec::new(),
            tree,
        });
        self.last_analysis = repairable_analysis();
        Ok(self.last_analysis.clone())
    }

    fn push_bytes(&mut self, origin: SourceOrigin, bytes: &[u8]) -> CollarResult<Analysis> {
        let state = self.active.as_mut().ok_or_else(|| {
            CollarError::Analysis("Python bytes arrived before BeginFile".to_string())
        })?;
        let old_len = state.source.len();
        let new_len = old_len
            .checked_add(bytes.len())
            .ok_or_else(|| CollarError::Analysis("Python source length overflowed".to_string()))?;
        let completed_bytes = self
            .completed
            .values()
            .try_fold(0usize, |total, mutation| {
                let len = match mutation {
                    PythonCandidateMutation::Upsert(source) => source.len(),
                    PythonCandidateMutation::Delete => 0,
                };
                total.checked_add(len)
            })
            .ok_or_else(|| {
                CollarError::Analysis("Python request byte count overflowed".to_string())
            })?;
        if completed_bytes.saturating_add(new_len) > self.max_source_bytes {
            return Err(CollarError::Analysis(format!(
                "Python request exceeds the {}-byte limit",
                self.max_source_bytes
            )));
        }
        state.source.extend_from_slice(bytes);
        if origin == SourceOrigin::Generated && old_len != new_len {
            if let Some((_, end)) = state.generated_ranges.last_mut()
                && *end == old_len
            {
                *end = new_len;
            } else {
                state.generated_ranges.push((old_len, new_len));
            }
        }
        let mut edited = state.tree.clone();
        let old_point = end_point(&state.source[..old_len]);
        edited.edit(&InputEdit {
            start_byte: old_len,
            old_end_byte: old_len,
            new_end_byte: new_len,
            start_position: old_point,
            old_end_position: old_point,
            new_end_position: end_point(&state.source),
        });
        state.tree = self
            .parser
            .parse(&state.source, Some(&edited))
            .ok_or_else(|| {
                CollarError::Analysis("pinned Python parser returned no updated tree".to_string())
            })?;
        self.last_analysis = repairable_analysis();
        Ok(self.last_analysis.clone())
    }

    fn analyze_boundary(&mut self, boundary: AnalysisBoundary) -> CollarResult<Analysis> {
        let Some(state) = self.active.as_ref() else {
            return Err(CollarError::Analysis(
                "Python boundary arrived before BeginFile".to_string(),
            ));
        };
        if state.mutation == MutationKind::Delete {
            self.last_analysis = repairable_analysis();
            return Ok(self.last_analysis.clone());
        }
        if state.tree.root_node().has_error() {
            self.last_analysis = unknown_analysis("python_incomplete_semantic_boundary", boundary);
            return Ok(self.last_analysis.clone());
        }
        let source = std::str::from_utf8(&state.source).map_err(|_| {
            CollarError::Analysis("Python generated source is not UTF-8".to_string())
        })?;
        if generated_suppression(source, &state.generated_ranges) {
            self.last_analysis = rejected_analysis("python_generated_type_suppression", boundary);
            return Ok(self.last_analysis.clone());
        }
        if matches!(
            boundary,
            AnalysisBoundary::Statement | AnalysisBoundary::File
        ) && diagnostic_profile("unsupported-operator").is_some_and(|profile| {
            profile.token_proof == Some(PythonTokenProof::GeneratedLiteralOperator)
        }) && literal_operator_contradiction(&state.tree, &state.source, &state.generated_ranges)
        {
            let mut candidates = self.completed.clone();
            candidates.insert(
                state.path.clone(),
                PythonCandidateMutation::Upsert(source.to_string()),
            );
            let reports = self
                .project
                .check_mutations(&candidates)
                .map_err(|error| CollarError::Analysis(error.to_string()))?;
            if reports.get(state.path.as_str()).is_some_and(|report| {
                report
                    .introduced
                    .iter()
                    .any(|diagnostic| diagnostic.code == "unsupported-operator")
            }) {
                self.last_analysis =
                    rejected_analysis("python_unsupported_literal_operator", boundary);
                return Ok(self.last_analysis.clone());
            }
        }
        self.last_analysis = repairable_analysis();
        Ok(self.last_analysis.clone())
    }

    fn end_file(&mut self) -> CollarResult<Analysis> {
        let state = self.active.take().ok_or_else(|| {
            CollarError::Analysis("Python EndFile arrived before BeginFile".to_string())
        })?;
        let mutation = if state.mutation == MutationKind::Delete {
            PythonCandidateMutation::Delete
        } else {
            self.generated_suppression |= generated_suppression(
                std::str::from_utf8(&state.source).map_err(|_| {
                    CollarError::Analysis("Python generated source is not UTF-8".to_string())
                })?,
                &state.generated_ranges,
            );
            PythonCandidateMutation::Upsert(String::from_utf8(state.source).map_err(|_| {
                CollarError::Analysis("Python generated source is not UTF-8".to_string())
            })?)
        };
        self.completed.insert(state.path, mutation);
        self.last_analysis = repairable_analysis();
        Ok(self.last_analysis.clone())
    }
}

impl IncrementalAnalyzer for PythonWorkspaceStreamingLayer {
    fn descriptor(&self) -> &AnalyzerLayerDescriptor {
        &self.project.descriptor
    }

    fn readiness(&self) -> LayerReadiness {
        LayerReadiness::Ready
    }

    fn readiness_receipt(&self) -> Option<&LayerReadinessReceipt> {
        Some(&self.project.receipt)
    }

    fn begin(&mut self, snapshot: ProgramSnapshot) -> CollarResult<()> {
        let python_bytes = snapshot
            .files
            .iter()
            .filter(|file| file.language == self.descriptor().language)
            .try_fold(0usize, |total, file| total.checked_add(file.bytes.len()))
            .ok_or_else(|| {
                CollarError::Analysis("Python program snapshot byte count overflowed".to_string())
            })?;
        if python_bytes > self.max_source_bytes {
            return Err(CollarError::Analysis(format!(
                "Python program snapshot exceeds the {}-byte request limit",
                self.max_source_bytes
            )));
        }
        self.active = None;
        self.completed.clear();
        self.snapshots.clear();
        self.epoch = self.epoch.wrapping_add(1);
        self.generated_suppression = false;
        self.last_analysis = repairable_analysis();
        Ok(())
    }

    fn checkpoint(&mut self) -> CollarResult<AnalyzerCheckpoint> {
        if self.snapshots.len() >= self.max_checkpoints {
            return Err(CollarError::Analysis(format!(
                "Python stream exceeds the {}-checkpoint limit",
                self.max_checkpoints
            )));
        }
        self.snapshots.push(PythonStateSnapshot {
            active: self.active.as_ref().map(PythonFileState::snapshot),
            completed: self.completed.clone(),
            generated_suppression: self.generated_suppression,
            last_analysis: self.last_analysis.clone(),
        });
        Ok(AnalyzerCheckpoint {
            epoch: self.epoch,
            revision: u64::try_from(self.snapshots.len().saturating_sub(1)).unwrap_or(u64::MAX),
        })
    }

    fn apply(&mut self, event: SourceEvent<'_>) -> CollarResult<Analysis> {
        match event {
            SourceEvent::BeginFile {
                path,
                language,
                mutation,
            } => {
                if language != &self.descriptor().language {
                    return Err(CollarError::Analysis(
                        "Python layer received a non-Python file".to_string(),
                    ));
                }
                self.begin_file(path, mutation)
            }
            SourceEvent::Bytes { origin, bytes } => self.push_bytes(origin, bytes),
            SourceEvent::DeleteKnownBytes(_) => Ok(self.last_analysis.clone()),
            SourceEvent::Boundary(boundary) => self.analyze_boundary(boundary),
            SourceEvent::EndFile => self.end_file(),
        }
    }

    fn rollback(&mut self, checkpoint: AnalyzerCheckpoint) -> CollarResult<()> {
        if checkpoint.epoch != self.epoch {
            return Err(CollarError::Analysis(
                "Python checkpoint belongs to another request".to_string(),
            ));
        }
        let revision = usize::try_from(checkpoint.revision).map_err(|_| {
            CollarError::Analysis("Python checkpoint revision does not fit usize".to_string())
        })?;
        let snapshot = self.snapshots.get(revision).ok_or_else(|| {
            CollarError::Analysis("Python checkpoint is not part of this stream".to_string())
        })?;
        match (&mut self.active, &snapshot.active) {
            (Some(active), Some(saved)) if active.path == saved.path => active.restore(saved)?,
            (active, None) => *active = None,
            (active, Some(saved)) => {
                // A multi-file patch may have ended this earlier file and moved to a later one
                // before a sibling-token probe rolls back across the file boundary. The complete
                // candidate is already retained under the same request byte bound, so reconstruct
                // the earlier append-only prefix from it instead of copying source into every
                // checkpoint.
                let completed = self.completed.get(&saved.path).ok_or_else(|| {
                    CollarError::Analysis(
                        "Python checkpoint lost its earlier completed file".to_string(),
                    )
                })?;
                *active = Some(saved.restore_from_completed(completed)?);
            }
        }
        self.completed.clone_from(&snapshot.completed);
        self.generated_suppression = snapshot.generated_suppression;
        self.last_analysis = snapshot.last_analysis.clone();
        self.snapshots.truncate(revision.saturating_add(1));
        Ok(())
    }

    fn finalize(&mut self) -> CollarResult<Analysis> {
        if self.active.is_some() {
            self.end_file()?;
        }
        if self.generated_suppression {
            self.last_analysis = rejected_analysis(
                "python_generated_type_suppression",
                AnalysisBoundary::ToolCall,
            );
            return Ok(self.last_analysis.clone());
        }
        if self.completed.is_empty() {
            return Ok(repairable_analysis());
        }
        let reports = self
            .project
            .check_project_mutations(&self.completed)
            .map_err(|error| CollarError::Analysis(error.to_string()))?;
        let introduced = reports
            .values()
            .flat_map(|report| &report.introduced)
            .collect::<Vec<_>>();
        if introduced.is_empty() {
            self.last_analysis = Analysis {
                viability: Viability::Valid,
                closure: ClosureVerdict::Allow,
                obligations: Vec::new(),
                biases: Vec::new(),
            };
        } else {
            self.last_analysis = Analysis {
                viability: Viability::Impossible,
                closure: ClosureVerdict::Reject,
                obligations: introduced
                    .iter()
                    .map(|diagnostic| SemanticObligation {
                        kind: format!("python_{}", diagnostic.code.replace('-', "_")),
                        boundary: AnalysisBoundary::ToolCall,
                    })
                    .collect(),
                biases: Vec::new(),
            };
        }
        Ok(self.last_analysis.clone())
    }
}

#[derive(Clone)]
struct PythonFileState {
    path: LogicalPath,
    mutation: MutationKind,
    source: Vec<u8>,
    generated_ranges: Vec<(usize, usize)>,
    tree: Tree,
}

impl PythonFileState {
    fn snapshot(&self) -> PythonFileSnapshot {
        PythonFileSnapshot {
            path: self.path.clone(),
            mutation: self.mutation,
            source_len: self.source.len(),
            generated_ranges: self.generated_ranges.clone(),
            tree: self.tree.clone(),
        }
    }

    fn restore(&mut self, saved: &PythonFileSnapshot) -> CollarResult<()> {
        if self.path != saved.path
            || self.mutation != saved.mutation
            || saved.source_len > self.source.len()
        {
            return Err(CollarError::Analysis(
                "Python checkpoint does not match the append-only source".to_string(),
            ));
        }
        self.source.truncate(saved.source_len);
        self.generated_ranges.clone_from(&saved.generated_ranges);
        self.tree = saved.tree.clone();
        Ok(())
    }
}

#[derive(Clone)]
struct PythonFileSnapshot {
    path: LogicalPath,
    mutation: MutationKind,
    source_len: usize,
    generated_ranges: Vec<(usize, usize)>,
    tree: Tree,
}

impl PythonFileSnapshot {
    fn restore_from_completed(
        &self,
        completed: &PythonCandidateMutation,
    ) -> CollarResult<PythonFileState> {
        let source = match (self.mutation, completed) {
            (MutationKind::Delete, PythonCandidateMutation::Delete) => Vec::new(),
            (
                MutationKind::Create | MutationKind::Modify,
                PythonCandidateMutation::Upsert(source),
            ) if self.source_len <= source.len() && source.is_char_boundary(self.source_len) => {
                source.as_bytes()[..self.source_len].to_vec()
            }
            _ => {
                return Err(CollarError::Analysis(
                    "Python completed candidate cannot restore the checkpointed file".to_string(),
                ));
            }
        };
        Ok(PythonFileState {
            path: self.path.clone(),
            mutation: self.mutation,
            source,
            generated_ranges: self.generated_ranges.clone(),
            tree: self.tree.clone(),
        })
    }
}

struct PythonStateSnapshot {
    active: Option<PythonFileSnapshot>,
    completed: BTreeMap<LogicalPath, PythonCandidateMutation>,
    generated_suppression: bool,
    last_analysis: Analysis,
}

#[salsa::db]
#[derive(Clone)]
struct PythonDatabase {
    storage: salsa::Storage<Self>,
    files: Files,
    system: PythonSystem,
    vendored: VendoredFileSystem,
    rule_selection: Arc<RuleSelection>,
    analysis_settings: Arc<AnalysisSettings>,
    open_files: BTreeSet<File>,
}

impl PythonDatabase {
    fn new(system: PythonSystem) -> Self {
        Self {
            storage: salsa::Storage::default(),
            files: Files::default(),
            system,
            vendored: ty_vendored::file_system().clone(),
            rule_selection: Arc::new(RuleSelection::from_registry(default_lint_registry())),
            analysis_settings: Arc::new(AnalysisSettings {
                respect_type_ignore_comments: false,
                ..AnalysisSettings::default()
            }),
            open_files: BTreeSet::new(),
        }
    }
}

struct PythonDatabaseFactory {
    system: PythonSystem,
    src_roots: Vec<SystemPathBuf>,
    site_packages_roots: Vec<SystemPathBuf>,
    python_version: PythonVersion,
    python_platform: PythonPlatform,
}

impl PythonDatabaseFactory {
    fn build(&self) -> Result<PythonDatabase, PythonLayerError> {
        let db = PythonDatabase::new(self.system.fork());
        let search_paths = SearchPathSettings {
            src_roots: self.src_roots.clone(),
            site_packages_paths: self.site_packages_roots.clone(),
            ..SearchPathSettings::empty()
        }
        .to_search_paths(db.system(), db.vendored(), &FallibleStrategy)
        .map_err(|error| PythonLayerError::InvalidConfig(error.to_string()))?;
        Program::from_settings(
            &db,
            ProgramSettings {
                python_version: PythonVersionWithSource {
                    version: self.python_version,
                    source: PythonVersionSource::default(),
                },
                python_platform: self.python_platform.clone(),
                search_paths,
            },
        );
        Ok(db)
    }
}

#[salsa::db]
impl SourceDb for PythonDatabase {
    fn vendored(&self) -> &VendoredFileSystem {
        &self.vendored
    }

    fn system(&self) -> &dyn System {
        &self.system
    }

    fn files(&self) -> &Files {
        &self.files
    }

    fn python_version(&self) -> PythonVersion {
        Program::get(self).python_version(self)
    }
}

#[salsa::db]
impl ModuleResolverDb for PythonDatabase {
    fn search_paths(&self) -> &SearchPaths {
        Program::get(self).search_paths(self)
    }
}

#[salsa::db]
impl PythonCoreDb for PythonDatabase {
    fn should_check_file(&self, file: File) -> bool {
        !file.path(self).is_vendored_path()
    }
}

#[salsa::db]
impl SemanticDb for PythonDatabase {
    fn check_file(&self, file: File) -> Vec<Diagnostic> {
        if self.should_check_file(file) {
            check_file_unwrap(self, file)
        } else {
            Vec::new()
        }
    }

    fn rule_selection(&self, _file: File) -> &RuleSelection {
        &self.rule_selection
    }

    fn lint_registry(&self) -> &LintRegistry {
        default_lint_registry()
    }

    fn analysis_settings(&self, _file: File) -> &AnalysisSettings {
        &self.analysis_settings
    }

    fn verbose(&self) -> bool {
        false
    }

    fn is_open_file(&self, file: File) -> bool {
        self.open_files.contains(&file)
    }

    fn dyn_clone(&self) -> Box<dyn SemanticDb> {
        Box::new(self.clone())
    }
}

#[salsa::db]
impl salsa::Database for PythonDatabase {}

struct FrozenImage {
    fs: MemoryFileSystem,
    first_party_files: BTreeMap<String, SystemPathBuf>,
    dependency_files: BTreeMap<String, SystemPathBuf>,
    controlled_modules: BTreeSet<String>,
}

impl FrozenImage {
    fn capture(config: &PythonProjectConfig) -> Result<Self, PythonLayerError> {
        let root = SystemPathBuf::from(VIRTUAL_PROJECT_ROOT);
        let fs = MemoryFileSystem::with_current_directory(&root);
        let first_roots = if config.first_party_roots.is_empty() {
            vec![PathBuf::new()]
        } else {
            config.first_party_roots.clone()
        };
        let mut first_party_files = BTreeMap::new();
        let mut dependency_files = BTreeMap::new();
        let mut controlled_modules = BTreeSet::new();
        let mut files = 0usize;
        let mut bytes = 0usize;
        for entry in WalkDir::new(&config.shadow_root).follow_links(false) {
            let entry = entry.map_err(|error| PythonLayerError::Snapshot(error.to_string()))?;
            if entry.file_type().is_symlink() {
                return Err(PythonLayerError::Snapshot(format!(
                    "Python semantic image refuses symlink {}",
                    entry.path().display()
                )));
            }
            if !entry.file_type().is_file() || !is_python_semantic_input(entry.path()) {
                continue;
            }
            let relative = entry
                .path()
                .strip_prefix(&config.shadow_root)
                .map_err(|error| PythonLayerError::Snapshot(error.to_string()))?;
            let logical = slash_path(relative)?;
            let contents = std::fs::read_to_string(entry.path()).map_err(|error| {
                PythonLayerError::Snapshot(format!("failed to read {logical}: {error}"))
            })?;
            files = files.saturating_add(1);
            bytes = bytes.saturating_add(contents.len());
            if files > config.max_files || bytes > config.max_bytes {
                return Err(PythonLayerError::Snapshot(
                    "Python semantic image exceeds configured bounds".to_string(),
                ));
            }
            let target = root.join(&logical);
            fs.write_file_all(&target, contents.as_bytes())?;
            if (logical.ends_with(".py") || logical.ends_with(".pyi"))
                && config
                    .site_packages_roots
                    .iter()
                    .any(|dependency| relative.starts_with(dependency))
            {
                dependency_files.insert(logical.clone(), target.clone());
            }
            if (logical.ends_with(".py") || logical.ends_with(".pyi"))
                && first_roots.iter().any(|first| relative.starts_with(first))
                && !config
                    .site_packages_roots
                    .iter()
                    .any(|dependency| relative.starts_with(dependency))
                && !relative
                    .components()
                    .any(|component| component.as_os_str().to_str() == Some("site-packages"))
            {
                for first in &first_roots {
                    let Ok(module_path) = relative.strip_prefix(first) else {
                        continue;
                    };
                    if let Some(module) = top_level_module(module_path) {
                        controlled_modules.insert(module);
                    }
                }
                first_party_files.insert(logical, target);
            }
        }
        Ok(Self {
            fs,
            first_party_files,
            dependency_files,
            controlled_modules,
        })
    }
}

fn diagnostic_records(
    db: &PythonDatabase,
    file: File,
    logical: &str,
    diagnostics: Vec<Diagnostic>,
) -> Vec<PythonDiagnostic> {
    let source = source_text(db, file);
    diagnostics
        .into_iter()
        .filter(|diagnostic| diagnostic.severity() == Severity::Error)
        .map(|diagnostic| {
            let range = diagnostic.primary_span_ref().and_then(|span| {
                (span.expect_ty_file() == file)
                    .then(|| span.range())
                    .flatten()
            });
            let (start, end) = range
                .map(|range| (usize::from(range.start()), usize::from(range.end())))
                .unwrap_or((0, 0));
            let excerpt = source.as_str().get(start..end).unwrap_or_default();
            PythonDiagnostic {
                path: logical.to_string(),
                code: diagnostic.id().as_str().to_string(),
                message: diagnostic.primary_message().to_string(),
                start,
                end,
                source_excerpt: excerpt.to_string(),
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
struct PythonPromotionPolicy {
    controlled_modules: BTreeSet<String>,
    external_imports_complete: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PythonTokenProof {
    GeneratedLiteralOperator,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PythonDiagnosticProfile {
    code: &'static str,
    /// `None` means that the diagnostic is authoritative only after every file in the mutation is
    /// known. A future token-time promotion must name and qualify a monotonic local proof here.
    token_proof: Option<PythonTokenProof>,
}

const PYTHON_DIAGNOSTIC_PROFILES: &[PythonDiagnosticProfile] = &[
    PythonDiagnosticProfile {
        code: "invalid-argument-type",
        token_proof: None,
    },
    PythonDiagnosticProfile {
        code: "invalid-assignment",
        token_proof: None,
    },
    PythonDiagnosticProfile {
        code: "invalid-return-type",
        token_proof: None,
    },
    PythonDiagnosticProfile {
        code: "unresolved-attribute",
        token_proof: None,
    },
    PythonDiagnosticProfile {
        code: "unresolved-import",
        token_proof: None,
    },
    PythonDiagnosticProfile {
        code: "unsupported-operator",
        token_proof: Some(PythonTokenProof::GeneratedLiteralOperator),
    },
];

fn diagnostic_profile(code: &str) -> Option<PythonDiagnosticProfile> {
    PYTHON_DIAGNOSTIC_PROFILES
        .iter()
        .copied()
        .find(|profile| profile.code == code)
}

fn promoted(
    diagnostics: Vec<PythonDiagnostic>,
    policy: &PythonPromotionPolicy,
) -> Vec<PythonDiagnostic> {
    diagnostics
        .into_iter()
        .filter(|diagnostic| {
            diagnostic_profile(&diagnostic.code).is_some()
                && (diagnostic.code != "unresolved-import"
                    || policy.external_imports_complete
                    || import_is_controlled(diagnostic, policy))
        })
        .collect()
}

fn import_is_controlled(diagnostic: &PythonDiagnostic, policy: &PythonPromotionPolicy) -> bool {
    let import = diagnostic.source_excerpt.trim();
    if import.starts_with('.') || diagnostic.message.contains("module `.") {
        return true;
    }
    let top_level = import
        .split(['.', ' ', '\t', '\n', ','])
        .next()
        .unwrap_or_default();
    policy.controlled_modules.contains(top_level)
}

fn multiset_difference(
    candidate: &[PythonDiagnostic],
    baseline: &[PythonDiagnostic],
) -> Vec<PythonDiagnostic> {
    let mut debt = baseline
        .iter()
        .cloned()
        .fold(BTreeMap::new(), |mut map, item| {
            *map.entry((item.code, item.message, item.source_excerpt))
                .or_insert(0usize) += 1;
            map
        });
    candidate
        .iter()
        .filter_map(|item| {
            let key = (
                item.code.clone(),
                item.message.clone(),
                item.source_excerpt.clone(),
            );
            match debt.get_mut(&key) {
                Some(count) if *count > 0 => {
                    *count -= 1;
                    None
                }
                _ => Some(item.clone()),
            }
        })
        .collect()
}

fn literal_operator_contradiction(
    tree: &Tree,
    source: &[u8],
    generated_ranges: &[(usize, usize)],
) -> bool {
    let mut found = false;
    visit_nodes(tree.root_node(), &mut |node| {
        if found || node.kind() != "binary_operator" || node.has_error() {
            return;
        }
        let range = (node.start_byte(), node.end_byte());
        if !range_intersects_generated(range, generated_ranges) {
            return;
        }
        let Some(left) = node.child_by_field_name("left") else {
            return;
        };
        let Some(right) = node.child_by_field_name("right") else {
            return;
        };
        let operator = node.child_by_field_name("operator").and_then(|operator| {
            source
                .get(operator.start_byte()..operator.end_byte())
                .and_then(|bytes| std::str::from_utf8(bytes).ok())
        });
        let incompatible = matches!(
            (left.kind(), right.kind()),
            ("string", "integer") | ("integer", "string")
        );
        found = operator == Some("+") && incompatible;
    });
    found
}

fn generated_suppression(source: &str, ranges: &[(usize, usize)]) -> bool {
    ranges.iter().any(|(start, end)| {
        source.get(*start..*end).is_some_and(|generated| {
            generated.contains("# ty: ignore")
                || generated.contains("#ty: ignore")
                || generated.contains("# type: ignore")
                || generated.contains("#type: ignore")
        })
    })
}

fn visit_nodes(node: Node<'_>, visit: &mut impl FnMut(Node<'_>)) {
    visit(node);
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        visit_nodes(child, visit);
    }
}

fn range_intersects_generated(range: (usize, usize), generated: &[(usize, usize)]) -> bool {
    generated
        .iter()
        .any(|candidate| range.0 < candidate.1 && candidate.0 < range.1)
}

fn repairable_analysis() -> Analysis {
    Analysis {
        viability: Viability::Repairable,
        closure: ClosureVerdict::Defer,
        obligations: Vec::new(),
        biases: Vec::new(),
    }
}

fn rejected_analysis(kind: &str, boundary: AnalysisBoundary) -> Analysis {
    Analysis {
        viability: Viability::Impossible,
        closure: ClosureVerdict::Reject,
        obligations: vec![SemanticObligation {
            kind: kind.to_string(),
            boundary,
        }],
        biases: Vec::new(),
    }
}

fn unknown_analysis(kind: &str, boundary: AnalysisBoundary) -> Analysis {
    Analysis {
        viability: Viability::Unknown,
        closure: ClosureVerdict::Defer,
        obligations: vec![SemanticObligation {
            kind: kind.to_string(),
            boundary,
        }],
        biases: Vec::new(),
    }
}

fn end_point(bytes: &[u8]) -> Point {
    let mut row = 0usize;
    let mut column = 0usize;
    for byte in bytes {
        if *byte == b'\n' {
            row = row.saturating_add(1);
            column = 0;
        } else {
            column = column.saturating_add(1);
        }
    }
    Point::new(row, column)
}

fn virtual_roots(roots: &[PathBuf]) -> Vec<SystemPathBuf> {
    if roots.is_empty() {
        return vec![SystemPathBuf::from(VIRTUAL_PROJECT_ROOT)];
    }
    roots
        .iter()
        .map(|root| {
            let relative = root.to_string_lossy().replace('\\', "/");
            SystemPathBuf::from(VIRTUAL_PROJECT_ROOT).join(relative)
        })
        .collect()
}

fn virtual_path(logical: &str) -> Result<SystemPathBuf, PythonLayerError> {
    let logical = LogicalPath::parse(logical.to_string())?;
    Ok(SystemPathBuf::from(VIRTUAL_PROJECT_ROOT).join(logical.as_str()))
}

fn validate_relative_root(root: &Path) -> Result<(), PythonLayerError> {
    if root.is_absolute()
        || root.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(PythonLayerError::InvalidConfig(format!(
            "Python search root {:?} must be project-relative",
            root
        )));
    }
    Ok(())
}

fn slash_path(path: &Path) -> Result<String, PythonLayerError> {
    let text = path
        .to_str()
        .ok_or_else(|| PythonLayerError::Snapshot("Python path is not UTF-8".to_string()))?;
    Ok(text.replace('\\', "/"))
}

fn top_level_module(path: &Path) -> Option<String> {
    let first = path.components().next()?.as_os_str().to_str()?;
    let module = first
        .strip_suffix(".pyi")
        .or_else(|| first.strip_suffix(".py"))
        .unwrap_or(first);
    (!module.is_empty() && module != "__init__").then(|| module.to_string())
}

fn is_python_semantic_input(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("py" | "pyi" | "pth")
    ) || path.file_name().and_then(|name| name.to_str()) == Some("py.typed")
}

fn is_lower_hex_digest(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn elapsed_millis(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[derive(Debug, thiserror::Error)]
pub enum PythonLayerError {
    #[error("invalid Python layer configuration: {0}")]
    InvalidConfig(String),
    #[error("Python semantic snapshot failed: {0}")]
    Snapshot(String),
    #[error("Python semantic provider is unavailable: {0}")]
    ProviderUnavailable(String),
    #[error("prepared Python semantic world is stale or has the wrong identity")]
    StaleWorld,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Collar(#[from] CollarError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use pb_control_collar::analysis::IncrementalAnalyzer;

    fn config(root: &Path) -> PythonProjectConfig {
        PythonProjectConfig {
            contract_version: PYTHON_LAYER_CONTRACT_VERSION,
            shadow_root: root.to_path_buf(),
            first_party_roots: Vec::new(),
            site_packages_roots: Vec::new(),
            external_imports_complete: false,
            python_version: "3.12".to_string(),
            python_platform: "linux".to_string(),
            world_sha256: "1".repeat(64),
            configuration_sha256: "2".repeat(64),
            dependency_sha256: "3".repeat(64),
            max_files: 1_000,
            max_bytes: 8 * 1024 * 1024,
        }
    }

    fn qualified_config(root: &Path) -> PythonProjectConfig {
        PythonProjectConfig {
            external_imports_complete: true,
            ..config(root)
        }
    }

    const PROMOTED_DIAGNOSTIC_FIXTURES: [(&str, &str); 6] = [
        (
            "invalid-argument-type",
            "def consume(value: int) -> None:\n    pass\nconsume(\"bad\")\n",
        ),
        ("invalid-assignment", "value: int = \"bad\"\n"),
        (
            "invalid-return-type",
            "def produce() -> int:\n    return \"bad\"\n",
        ),
        (
            "unresolved-attribute",
            "class Item:\n    pass\nvalue = Item().missing\n",
        ),
        (
            "unresolved-import",
            "import package_absent_from_the_complete_static_world\n",
        ),
        ("unsupported-operator", "value = \"bad\" + 1\n"),
    ];

    fn write(root: &Path, path: &str, source: &str) {
        let path = root.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, source).unwrap();
    }

    #[test]
    fn native_ty_rejects_string_plus_integer_but_allows_dynamic_any() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "main.py", "value: int = 1\n");
        let world = PythonProjectWorld::load_and_prime(config(root.path())).unwrap();
        let mut request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();

        let invalid = BTreeMap::from([(
            LogicalPath::parse("main.py".to_string()).unwrap(),
            "value = \"text\" + 1\n".to_string(),
        )]);
        let report = request.check_candidates(&invalid).unwrap();
        assert!(
            report["main.py"]
                .introduced
                .iter()
                .any(|diagnostic| diagnostic.code == "unsupported-operator")
        );

        let dynamic = BTreeMap::from([(
            LogicalPath::parse("main.py".to_string()).unwrap(),
            "from typing import Any\nvalue: Any = 1\nresult = value + \"text\"\n".to_string(),
        )]);
        assert!(
            request.check_candidates(&dynamic).unwrap()["main.py"]
                .introduced
                .is_empty()
        );
    }

    #[test]
    fn unresolved_absolute_import_stays_unknown_without_a_qualified_environment() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "main.py", "value = 1\n");
        let world = PythonProjectWorld::load_and_prime(config(root.path())).unwrap();
        let mut request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let reports = request
            .check_candidates(&BTreeMap::from([(
                LogicalPath::parse("main.py".to_string()).unwrap(),
                "import package_that_is_not_in_the_frozen_world\nvalue = 1\n".to_string(),
            )]))
            .unwrap();

        assert!(reports["main.py"].introduced.is_empty());
    }

    #[test]
    fn every_promoted_diagnostic_has_an_exact_complete_candidate_fixture() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "main.py", "value = 1\n");
        let world = PythonProjectWorld::load_and_prime(qualified_config(root.path())).unwrap();
        let mut request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let path = LogicalPath::parse("main.py".to_string()).unwrap();
        let cases = PROMOTED_DIAGNOSTIC_FIXTURES;
        assert_eq!(
            cases.iter().map(|(code, _)| *code).collect::<BTreeSet<_>>(),
            PYTHON_DIAGNOSTIC_PROFILES
                .iter()
                .map(|profile| profile.code)
                .collect::<BTreeSet<_>>()
        );

        for (expected, source) in cases {
            let report = request
                .check_candidates(&BTreeMap::from([(path.clone(), source.to_string())]))
                .unwrap();
            assert!(
                report["main.py"]
                    .introduced
                    .iter()
                    .any(|diagnostic| diagnostic.code == expected),
                "expected promoted diagnostic {expected}, got {:?}",
                report["main.py"].introduced
            );
        }

        let dynamic = "from typing import Any\n\nclass Dynamic:\n    def __getattr__(self, name: str) -> Any:\n        return 1\n\ndef consume(value: int) -> None:\n    pass\n\ndef produce(value: Any) -> int:\n    return value\n\ndynamic: Any = Dynamic().anything\nconsume(dynamic)\nassigned: int = dynamic\nattribute = dynamic.missing\noperator = dynamic + 1\nresult: int = produce(dynamic)\n";
        assert!(
            request
                .check_candidates(&BTreeMap::from([(path, dynamic.to_string())]))
                .unwrap()["main.py"]
                .introduced
                .is_empty()
        );
    }

    #[test]
    fn closure_only_diagnostics_never_hard_reject_a_statement_prefix() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "main.py", "value = 1\n");
        let world = PythonProjectWorld::load_and_prime(qualified_config(root.path())).unwrap();
        let language = LanguageId("python".to_string());
        let path = LogicalPath::parse("main.py".to_string()).unwrap();

        for (expected, source) in
            PROMOTED_DIAGNOSTIC_FIXTURES
                .iter()
                .copied()
                .filter(|(code, _)| {
                    diagnostic_profile(code).is_some_and(|profile| profile.token_proof.is_none())
                })
        {
            let request = world
                .snapshot_for_request(&world.descriptor().world)
                .unwrap();
            let mut layer = request.into_streaming_layer(1024 * 1024, 64).unwrap();
            layer.begin(ProgramSnapshot::default()).unwrap();
            layer
                .apply(SourceEvent::BeginFile {
                    path: &path,
                    language: &language,
                    mutation: MutationKind::Modify,
                })
                .unwrap();
            layer
                .apply(SourceEvent::Bytes {
                    origin: SourceOrigin::Generated,
                    bytes: source.as_bytes(),
                })
                .unwrap();
            let boundary = layer
                .apply(SourceEvent::Boundary(AnalysisBoundary::Statement))
                .unwrap();
            assert_ne!(
                boundary.viability,
                Viability::Impossible,
                "{expected} must remain repairable until transaction closure"
            );
            layer.apply(SourceEvent::EndFile).unwrap();
            let closure = layer.finalize().unwrap();
            assert_eq!(closure.viability, Viability::Impossible, "{expected}");
            assert_eq!(closure.closure, ClosureVerdict::Reject, "{expected}");
            assert!(closure.obligations.iter().any(|obligation| {
                obligation.kind == format!("python_{}", expected.replace('-', "_"))
            }));
        }
    }

    #[test]
    fn unresolved_import_stays_repairable_until_later_patch_files_are_known() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "main.py", "value = 1\n");
        let world = PythonProjectWorld::load_and_prime(qualified_config(root.path())).unwrap();
        let request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let mut layer = request.into_streaming_layer(1024 * 1024, 64).unwrap();
        layer.begin(ProgramSnapshot::default()).unwrap();
        let language = LanguageId("python".to_string());
        let main = LogicalPath::parse("main.py".to_string()).unwrap();
        layer
            .apply(SourceEvent::BeginFile {
                path: &main,
                language: &language,
                mutation: MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"from future_helper import answer\nvalue: int = answer()\n",
            })
            .unwrap();
        let boundary = layer
            .apply(SourceEvent::Boundary(AnalysisBoundary::Statement))
            .unwrap();
        assert_ne!(boundary.viability, Viability::Impossible);
        assert_ne!(boundary.closure, ClosureVerdict::Reject);
        layer.apply(SourceEvent::EndFile).unwrap();

        let helper = LogicalPath::parse("future_helper.py".to_string()).unwrap();
        layer
            .apply(SourceEvent::BeginFile {
                path: &helper,
                language: &language,
                mutation: MutationKind::Create,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"def answer() -> int:\n    return 42\n",
            })
            .unwrap();
        layer.apply(SourceEvent::EndFile).unwrap();

        let analysis = layer.finalize().unwrap();
        assert_eq!(analysis.viability, Viability::Valid);
        assert_eq!(analysis.closure, ClosureVerdict::Allow);
    }

    #[test]
    fn request_overlay_resolves_new_cross_file_imports_and_symbol_shapes() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "main.py", "value = 1\n");
        let world = PythonProjectWorld::load_and_prime(config(root.path())).unwrap();
        let mut request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let valid = BTreeMap::from([
            (
                LogicalPath::parse("helper.py".to_string()).unwrap(),
                "def add(value: int) -> int:\n    return value + 1\n".to_string(),
            ),
            (
                LogicalPath::parse("main.py".to_string()).unwrap(),
                "from helper import add\nvalue = add(1)\n".to_string(),
            ),
        ]);
        let valid_report = request.check_candidates(&valid).unwrap();
        assert!(
            valid_report
                .values()
                .all(|report| report.introduced.is_empty())
        );

        let invalid = BTreeMap::from([
            (
                LogicalPath::parse("helper.py".to_string()).unwrap(),
                "def add(value: int) -> int:\n    return value + 1\n".to_string(),
            ),
            (
                LogicalPath::parse("main.py".to_string()).unwrap(),
                "from helper import add\nvalue = add(\"bad\")\n".to_string(),
            ),
        ]);
        assert!(
            request.check_candidates(&invalid).unwrap()["main.py"]
                .introduced
                .iter()
                .any(|diagnostic| diagnostic.code == "invalid-argument-type")
        );
    }

    #[test]
    fn speculative_new_modules_do_not_leak_between_candidate_checks() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "main.py", "value = 1\n");
        let world = PythonProjectWorld::load_and_prime(qualified_config(root.path())).unwrap();
        let mut request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let with_helper = BTreeMap::from([
            (
                LogicalPath::parse("helper.py".to_string()).unwrap(),
                "def add(value: int) -> int:\n    return value + 1\n".to_string(),
            ),
            (
                LogicalPath::parse("main.py".to_string()).unwrap(),
                "from helper import add\nvalue = add(1)\n".to_string(),
            ),
        ]);
        assert!(
            request
                .check_candidates(&with_helper)
                .unwrap()
                .values()
                .all(|report| report.introduced.is_empty())
        );

        let without_helper = BTreeMap::from([(
            LogicalPath::parse("main.py".to_string()).unwrap(),
            "from helper import add\nvalue = add(1)\n".to_string(),
        )]);
        assert!(
            request.check_candidates(&without_helper).unwrap()["main.py"]
                .introduced
                .iter()
                .any(|diagnostic| diagnostic.code == "unresolved-import")
        );
    }

    #[test]
    fn streaming_rollback_clears_a_finalized_branch_semantic_overlay() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "main.py", "value = 1\n");
        let world = PythonProjectWorld::load_and_prime(qualified_config(root.path())).unwrap();
        let request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let mut layer = request.into_streaming_layer(1024 * 1024, 64).unwrap();
        layer.begin(ProgramSnapshot::default()).unwrap();
        let checkpoint = layer.checkpoint().unwrap();
        let language = LanguageId("python".to_string());
        let helper = LogicalPath::parse("helper.py".to_string()).unwrap();
        layer
            .apply(SourceEvent::BeginFile {
                path: &helper,
                language: &language,
                mutation: MutationKind::Create,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"def add(value: int) -> int:\n    return value + 1\n",
            })
            .unwrap();
        layer.apply(SourceEvent::EndFile).unwrap();
        assert_eq!(layer.finalize().unwrap().closure, ClosureVerdict::Allow);

        layer.rollback(checkpoint).unwrap();
        let main = LogicalPath::parse("main.py".to_string()).unwrap();
        layer
            .apply(SourceEvent::BeginFile {
                path: &main,
                language: &language,
                mutation: MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"from helper import add\nvalue = add(1)\n",
            })
            .unwrap();
        layer.apply(SourceEvent::EndFile).unwrap();
        let analysis = layer.finalize().unwrap();
        assert_eq!(analysis.viability, Viability::Impossible);
        assert_eq!(analysis.closure, ClosureVerdict::Reject);
    }

    #[test]
    fn project_closure_treats_deleted_modules_as_absent_for_untouched_dependants() {
        let root = tempfile::tempdir().unwrap();
        write(
            root.path(),
            "helper.py",
            "def add(value: int) -> int:\n    return value + 1\n",
        );
        write(
            root.path(),
            "main.py",
            "from helper import add\nvalue = add(1)\n",
        );
        let world = PythonProjectWorld::load_and_prime(config(root.path())).unwrap();
        let mut request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let candidates = BTreeMap::from([(
            LogicalPath::parse("helper.py".to_string()).unwrap(),
            PythonCandidateMutation::Delete,
        )]);

        assert!(request.check_mutations(&candidates).unwrap().is_empty());
        assert!(
            request.check_project_mutations(&candidates).unwrap()["main.py"]
                .introduced
                .iter()
                .any(|diagnostic| diagnostic.code == "unresolved-import")
        );
    }

    #[test]
    fn project_closure_rejects_a_public_shape_change_until_dependants_are_updated() {
        let root = tempfile::tempdir().unwrap();
        write(
            root.path(),
            "helper.py",
            "def render(value: str) -> str:\n    return value\n",
        );
        write(
            root.path(),
            "main.py",
            "from helper import render\nresult: str = render(\"ok\")\n",
        );
        let world = PythonProjectWorld::load_and_prime(config(root.path())).unwrap();
        let mut request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let helper = LogicalPath::parse("helper.py".to_string()).unwrap();
        let changed_helper = PythonCandidateMutation::Upsert(
            "def render(value: int) -> int:\n    return value\n".to_string(),
        );
        let changed = BTreeMap::from([(helper.clone(), changed_helper.clone())]);
        assert!(
            request.check_mutations(&changed).unwrap()["helper.py"]
                .introduced
                .is_empty()
        );
        assert!(
            request.check_project_mutations(&changed).unwrap()["main.py"]
                .introduced
                .iter()
                .any(|diagnostic| matches!(
                    diagnostic.code.as_str(),
                    "invalid-argument-type" | "invalid-assignment"
                ))
        );

        let coordinated = BTreeMap::from([
            (helper, changed_helper),
            (
                LogicalPath::parse("main.py".to_string()).unwrap(),
                PythonCandidateMutation::Upsert(
                    "from helper import render\nresult: int = render(1)\n".to_string(),
                ),
            ),
        ]);
        assert!(
            request
                .check_project_mutations(&coordinated)
                .unwrap()
                .values()
                .all(|report| report.introduced.is_empty())
        );
    }

    #[test]
    fn rollback_reconstructs_an_earlier_file_after_crossing_a_patch_boundary() {
        let root = tempfile::tempdir().unwrap();
        write(
            root.path(),
            "helper.py",
            "def render(value: str) -> str:\n    return value\n",
        );
        write(
            root.path(),
            "main.py",
            "from helper import render\nresult: str = render(\"ok\")\n",
        );
        let world = PythonProjectWorld::load_and_prime(config(root.path())).unwrap();
        let request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let mut layer = request.into_streaming_layer(1024 * 1024, 64).unwrap();
        layer.begin(ProgramSnapshot::default()).unwrap();
        let language = LanguageId("python".to_string());
        let helper = LogicalPath::parse("helper.py".to_string()).unwrap();
        let main = LogicalPath::parse("main.py".to_string()).unwrap();
        let helper_source = b"def render(value: int) -> int:\n    return value\n";
        let main_source = b"from helper import render\nresult: int = render(1)\n";

        layer
            .apply(SourceEvent::BeginFile {
                path: &helper,
                language: &language,
                mutation: MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: helper_source,
            })
            .unwrap();
        let earlier = layer.checkpoint().unwrap();
        layer.apply(SourceEvent::EndFile).unwrap();
        layer
            .apply(SourceEvent::BeginFile {
                path: &main,
                language: &language,
                mutation: MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: main_source,
            })
            .unwrap();

        layer.rollback(earlier).unwrap();
        layer.apply(SourceEvent::EndFile).unwrap();
        layer
            .apply(SourceEvent::BeginFile {
                path: &main,
                language: &language,
                mutation: MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: main_source,
            })
            .unwrap();
        layer.apply(SourceEvent::EndFile).unwrap();
        let closure = layer.finalize().unwrap();
        assert_eq!(closure.viability, Viability::Valid);
        assert_eq!(closure.closure, ClosureVerdict::Allow);
    }

    #[test]
    fn streaming_project_closure_propagates_a_module_deletion_to_untouched_files() {
        let root = tempfile::tempdir().unwrap();
        write(
            root.path(),
            "helper.py",
            "def add(value: int) -> int:\n    return value + 1\n",
        );
        write(
            root.path(),
            "main.py",
            "from helper import add\nvalue = add(1)\n",
        );
        let world = PythonProjectWorld::load_and_prime(config(root.path())).unwrap();
        let request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let mut layer = request.into_streaming_layer(1024 * 1024, 64).unwrap();
        layer.begin(ProgramSnapshot::default()).unwrap();
        let language = LanguageId("python".to_string());
        let helper = LogicalPath::parse("helper.py".to_string()).unwrap();
        layer
            .apply(SourceEvent::BeginFile {
                path: &helper,
                language: &language,
                mutation: MutationKind::Delete,
            })
            .unwrap();
        layer.apply(SourceEvent::EndFile).unwrap();

        let analysis = layer.finalize().unwrap();
        assert_eq!(analysis.viability, Viability::Impossible);
        assert_eq!(analysis.closure, ClosureVerdict::Reject);
    }

    #[test]
    fn baseline_debt_does_not_mask_a_second_error() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "main.py", "first = \"old\" + 1\n");
        let world = PythonProjectWorld::load_and_prime(config(root.path())).unwrap();
        let mut request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let candidate = BTreeMap::from([(
            LogicalPath::parse("main.py".to_string()).unwrap(),
            "first = \"old\" + 1\nsecond = \"new\" + 2\n".to_string(),
        )]);
        let report = &request.check_candidates(&candidate).unwrap()["main.py"];
        assert_eq!(report.baseline.len(), 1);
        assert_eq!(report.introduced.len(), 1);
        assert_eq!(report.introduced[0].source_excerpt, "\"new\" + 2");
    }

    #[test]
    fn streaming_only_hard_rejects_a_complete_generated_literal_contradiction() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "main.py", "value = 1\n");
        let world = PythonProjectWorld::load_and_prime(config(root.path())).unwrap();
        let request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let mut layer = request.into_streaming_layer(1024 * 1024, 64).unwrap();
        layer.begin(ProgramSnapshot::default()).unwrap();
        let path = LogicalPath::parse("main.py".to_string()).unwrap();
        let language = LanguageId("python".to_string());
        layer
            .apply(SourceEvent::BeginFile {
                path: &path,
                language: &language,
                mutation: MutationKind::Modify,
            })
            .unwrap();
        layer
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"value = \"text\" + 1\n",
            })
            .unwrap();
        let analysis = layer
            .apply(SourceEvent::Boundary(AnalysisBoundary::Statement))
            .unwrap();
        assert_eq!(analysis.viability, Viability::Impossible);
        assert_eq!(analysis.closure, ClosureVerdict::Reject);
    }

    #[test]
    fn generated_suppressions_reject_but_preserved_baseline_suppressions_do_not() {
        let root = tempfile::tempdir().unwrap();
        let baseline = "old = \"text\" + 1  # ty: ignore[unsupported-operator]\n";
        write(root.path(), "main.py", baseline);
        let world = PythonProjectWorld::load_and_prime(config(root.path())).unwrap();
        let language = LanguageId("python".to_string());
        let path = LogicalPath::parse("main.py".to_string()).unwrap();

        let request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let mut preserved = request.into_streaming_layer(1024 * 1024, 64).unwrap();
        preserved.begin(ProgramSnapshot::default()).unwrap();
        preserved
            .apply(SourceEvent::BeginFile {
                path: &path,
                language: &language,
                mutation: MutationKind::Modify,
            })
            .unwrap();
        preserved
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Known,
                bytes: baseline.as_bytes(),
            })
            .unwrap();
        preserved
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"value = 1\n",
            })
            .unwrap();
        preserved.apply(SourceEvent::EndFile).unwrap();
        assert_eq!(preserved.finalize().unwrap().closure, ClosureVerdict::Allow);

        let request = world
            .snapshot_for_request(&world.descriptor().world)
            .unwrap();
        let mut generated = request.into_streaming_layer(1024 * 1024, 64).unwrap();
        generated.begin(ProgramSnapshot::default()).unwrap();
        generated
            .apply(SourceEvent::BeginFile {
                path: &path,
                language: &language,
                mutation: MutationKind::Modify,
            })
            .unwrap();
        generated
            .apply(SourceEvent::Bytes {
                origin: SourceOrigin::Generated,
                bytes: b"value = \"text\" + 1  # ty: ignore[unsupported-operator]\n",
            })
            .unwrap();
        generated.apply(SourceEvent::EndFile).unwrap();
        let analysis = generated.finalize().unwrap();
        assert_eq!(analysis.viability, Viability::Impossible);
        assert_eq!(analysis.closure, ClosureVerdict::Reject);
    }
}
