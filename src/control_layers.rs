//! Pre-inference lifecycle for native streaming language layers.
//!
//! Expensive project worlds are built from verified immutable shadows before any model invocation
//! that can emit a matching-language mutation. Decoding receives only a cheap request snapshot;
//! it never loads Cargo metadata, starts a language server, or reads the live workspace.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    io::Read,
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        mpsc::{Receiver, RecvTimeoutError, sync_channel},
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use pb_control_collar::{
    CompletionDecision, MutationCompletionGate,
    analysis::{
        IncrementalAnalyzer, LanguageLayerStack, ProgramFile, ProgramSnapshot, SyntaxProfile,
    },
    mutation::WorkspaceSnapshot,
    protocol::ToolDialect,
    tool::{CollarLimits, CollarManifest, ExposedTool, MutationPolicy, ToolConstraintMode},
};
use pb_control_python::{PYTHON_LAYER_CONTRACT_VERSION, PythonProjectConfig, PythonProjectWorld};
use pb_control_rust::{RUST_LAYER_CONTRACT_VERSION, RustProjectConfig, RustProjectWorld};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    agent_core::BuiltInToolSchema,
    python_semantic_config::{PYTHON_SEMANTIC_CONFIG_PATH, PythonSemanticConfig},
    semantic::{SemanticShadowExtraFile, SemanticShadowWorkspace},
};

const RUST_LAYER_MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;
const RUST_LAYER_MAX_CHECKPOINTS: usize = 4_096;
const PYTHON_LAYER_MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;
const PYTHON_LAYER_MAX_CHECKPOINTS: usize = 4_096;
const PYTHON_LAYER_MAX_PROJECT_FILES: usize = 100_000;
const PYTHON_LAYER_MAX_PROJECT_BYTES: usize = 256 * 1024 * 1024;
const PYTHON_DEPENDENCY_SHADOW_PREFIX: &str = ".pb-semantic-dependencies/python";
const PYTHON_DEPENDENCY_MAX_CONFIG_BYTES: u64 = 64 * 1024;
const PYTHON_DEPENDENCY_MAX_SEARCH_ROOTS: usize = 32;
const PYTHON_DEPENDENCY_MAX_PTH_LINE_BYTES: usize = 4 * 1024;
const MAX_PROCESS_RUST_WORLDS: usize = 2;
const MAX_PROCESS_PYTHON_WORLDS: usize = 2;
// A cold rust-analyzer world can consume substantial memory. Detached cancellation must not let
// abandoned requests accumulate concurrent loaders for different workspaces.
const MAX_CONCURRENT_RUST_WORLD_PREPARATIONS: usize = 1;
const MAX_CONCURRENT_PYTHON_WORLD_PREPARATIONS: usize = 1;
const PREPARATION_WAIT_POLL: Duration = Duration::from_millis(100);

pub(crate) type SharedLanguageLayers = Arc<Mutex<LanguageLayerStack>>;

pub(crate) struct ControlLayerLifecycle {
    rust: Option<PreparedRustHandle>,
    python: Option<PreparedPythonHandle>,
    cold_builds: u64,
    warm_requests: u64,
    process_cache_hits: u64,
    python_cold_builds: u64,
    python_warm_requests: u64,
    python_process_cache_hits: u64,
    python_dependency_authority: crate::python_semantic_config::PythonExternalAuthority,
}

struct PreparedRustWorld {
    workspace_root: PathBuf,
    source_sha256: String,
    content: crate::workspace::ContentSnapshot,
    _shadow: Arc<SemanticShadowWorkspace>,
    world: RustProjectWorld,
}

struct PreparedPythonWorld {
    workspace_root: PathBuf,
    source_sha256: String,
    dependency_sha256: String,
    _shadow: Arc<SemanticShadowWorkspace>,
    world: PythonProjectWorld,
}

#[derive(Clone, Debug)]
struct PythonDependencySnapshot {
    fingerprint: String,
    python_version: String,
    first_party_roots: Vec<PathBuf>,
    site_packages_roots: Vec<PathBuf>,
    external_imports_complete: bool,
    files: Vec<SemanticShadowExtraFile>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PythonWorldQualificationObservation {
    pub(crate) world_sha256: String,
    pub(crate) configuration_sha256: String,
    pub(crate) dependency_sha256: String,
    pub(crate) provider_version: String,
    pub(crate) load_millis: u64,
    pub(crate) prime_millis: u64,
    pub(crate) primed_queries: u64,
    pub(crate) cold_millis: u64,
    pub(crate) warm_millis: u64,
    pub(crate) process_cache_hit_millis: u64,
    pub(crate) invalid_replay_millis: u64,
    pub(crate) valid_replay_millis: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PythonSemanticQualificationObservation {
    pub(crate) generation: CompletionDecision,
    pub(crate) final_replay: CompletionDecision,
    pub(crate) diagnostic_codes: BTreeSet<String>,
    pub(crate) warm_millis: u64,
    pub(crate) generation_millis: u64,
    pub(crate) final_replay_millis: u64,
    pub(crate) diagnostic_millis: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PythonSemanticWorldObservation {
    pub(crate) world_sha256: String,
    pub(crate) configuration_sha256: String,
    pub(crate) dependency_sha256: String,
    pub(crate) provider_version: String,
    pub(crate) load_millis: u64,
    pub(crate) prime_millis: u64,
    pub(crate) primed_queries: u64,
}

impl Default for ControlLayerLifecycle {
    fn default() -> Self {
        Self {
            rust: None,
            python: None,
            cold_builds: 0,
            warm_requests: 0,
            process_cache_hits: 0,
            python_cold_builds: 0,
            python_warm_requests: 0,
            python_process_cache_hits: 0,
            python_dependency_authority: Default::default(),
        }
    }
}

type PreparedRustHandle = Arc<Mutex<PreparedRustWorld>>;
type PreparedPythonHandle = Arc<Mutex<PreparedPythonWorld>>;

static RUST_WORLD_CACHE: OnceLock<Mutex<VecDeque<PreparedRustHandle>>> = OnceLock::new();
static PYTHON_WORLD_CACHE: OnceLock<Mutex<VecDeque<PreparedPythonHandle>>> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RustWorldKey {
    workspace_root: PathBuf,
    source_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PythonWorldKey {
    workspace_root: PathBuf,
    source_sha256: String,
    dependency_sha256: String,
}

#[derive(Default)]
struct RustWorldPreparationState {
    active: HashSet<RustWorldKey>,
    waiters: HashMap<RustWorldKey, usize>,
    completed: HashMap<RustWorldKey, PreparedRustHandle>,
}

#[derive(Default)]
struct RustWorldPreparationCoordinator {
    state: Mutex<RustWorldPreparationState>,
    ready: Condvar,
}

struct RustWorldPreparationGuard {
    coordinator: &'static RustWorldPreparationCoordinator,
    key: RustWorldKey,
}

enum RustWorldPreparation {
    Owner(RustWorldPreparationGuard),
    Shared(PreparedRustHandle),
}

struct RustWorldPreparationPublisher {
    coordinator: &'static RustWorldPreparationCoordinator,
    key: RustWorldKey,
}

static RUST_WORLD_PREPARATIONS: OnceLock<RustWorldPreparationCoordinator> = OnceLock::new();

#[derive(Default)]
struct PythonWorldPreparationState {
    active: HashSet<PythonWorldKey>,
    waiters: HashMap<PythonWorldKey, usize>,
    completed: HashMap<PythonWorldKey, PreparedPythonHandle>,
}

#[derive(Default)]
struct PythonWorldPreparationCoordinator {
    state: Mutex<PythonWorldPreparationState>,
    ready: Condvar,
}

struct PythonWorldPreparationGuard {
    coordinator: &'static PythonWorldPreparationCoordinator,
    key: PythonWorldKey,
}

enum PythonWorldPreparation {
    Owner(PythonWorldPreparationGuard),
    Shared(PreparedPythonHandle),
}

struct PythonWorldPreparationPublisher {
    coordinator: &'static PythonWorldPreparationCoordinator,
    key: PythonWorldKey,
}

static PYTHON_WORLD_PREPARATIONS: OnceLock<PythonWorldPreparationCoordinator> = OnceLock::new();

impl ControlLayerLifecycle {
    pub(crate) fn with_python_dependency_authority(
        authority: crate::python_semantic_config::PythonExternalAuthority,
    ) -> Self {
        Self {
            python_dependency_authority: authority,
            ..Self::default()
        }
    }

    /// Establish all expensive state before the caller reserves or records a model invocation.
    /// Returning successfully means the supplied stack is bound to the exact live-workspace
    /// identity observed after loading and priming completed.
    #[cfg(test)]
    fn prepare_for_inference(
        &mut self,
        workspace_root: &Path,
        tools: &[BuiltInToolSchema],
        mutation_snapshot: Option<&WorkspaceSnapshot>,
    ) -> Result<Option<SharedLanguageLayers>> {
        self.prepare_for_inference_cancellable(workspace_root, tools, mutation_snapshot, &|| Ok(()))
    }

    pub(crate) fn prepare_for_inference_cancellable(
        &mut self,
        workspace_root: &Path,
        tools: &[BuiltInToolSchema],
        mutation_snapshot: Option<&WorkspaceSnapshot>,
        cancellation: &dyn Fn() -> Result<()>,
    ) -> Result<Option<SharedLanguageLayers>> {
        let bound_path = mutation_snapshot.and_then(WorkspaceSnapshot::bound_mutation_path);
        let prepare_rust = tools_may_mutate_rust(tools, bound_path);
        let prepare_python = tools_may_mutate_python(tools, bound_path);
        if !prepare_rust && !prepare_python {
            return Ok(None);
        }
        cancellation()?;
        let snapshot = mutation_snapshot.context(
            "native-language-edit-capable inference requires a controller-authorized mutation snapshot",
        )?;
        let live = crate::workspace::ContentSnapshot::capture(workspace_root)
            .context("failed to identify the native semantic world before inference")?;
        let canonical_root = workspace_root
            .canonicalize()
            .context("failed to canonicalize the native semantic project root")?;

        // Rust is intentionally first: Cargo/rust-analyzer preparation is the slowest supported
        // lifecycle stage. No model reservation or inference begins until every requested layer
        // below has returned a readiness receipt for this exact workspace identity.
        let mut layers: Vec<Box<dyn IncrementalAnalyzer + Send>> = Vec::new();
        if prepare_rust
            && let Some(layer) = self.prepare_rust_layer(&canonical_root, &live, cancellation)?
        {
            layers.push(layer);
            self.warm_requests = self.warm_requests.saturating_add(1);
        }
        cancellation()?;
        if prepare_python
            && let Some(layer) = self.prepare_python_layer(&canonical_root, &live, cancellation)?
        {
            layers.push(layer);
            self.python_warm_requests = self.python_warm_requests.saturating_add(1);
        }
        if layers.is_empty() {
            return Ok(None);
        }
        let stack = LanguageLayerStack::new(layers, program_snapshot(snapshot)?)
            .context("failed to start the request-local native language-layer stack")?;
        Ok(Some(Arc::new(Mutex::new(stack))))
    }

    fn prepare_rust_layer(
        &mut self,
        canonical_root: &Path,
        live: &crate::workspace::ContentSnapshot,
        cancellation: &dyn Fn() -> Result<()>,
    ) -> Result<Option<Box<dyn IncrementalAnalyzer + Send>>> {
        if !live.paths.contains_key("Cargo.toml")
            && !live.paths.keys().any(|path| path.ends_with("/Cargo.toml"))
        {
            // A standalone .rs file still receives the collar's lexical/syntax layer. There is no
            // Cargo project whose dependencies can be loaded and resolved.
            return Ok(None);
        }
        let current_matches = self
            .rust
            .as_ref()
            .map(|prepared| {
                prepared
                    .lock()
                    .map(|prepared| {
                        prepared.workspace_root == canonical_root
                            && prepared.source_sha256 == live.fingerprint
                    })
                    .map_err(|_| anyhow::anyhow!("Rust semantic world lock is poisoned"))
            })
            .transpose()?
            == Some(true);
        if !current_matches {
            if let Some(cached) = process_cached_world(canonical_root, &live.fingerprint)? {
                self.rust = Some(cached);
                self.process_cache_hits = self.process_cache_hits.saturating_add(1);
            } else {
                let preparation = begin_rust_world_preparation(
                    RustWorldKey {
                        workspace_root: canonical_root.to_path_buf(),
                        source_sha256: live.fingerprint.clone(),
                    },
                    cancellation,
                )?;
                match preparation {
                    RustWorldPreparation::Shared(shared) => {
                        self.rust = Some(shared);
                        self.process_cache_hits = self.process_cache_hits.saturating_add(1);
                    }
                    RustWorldPreparation::Owner(preparation) => {
                        if let Some(cached) =
                            process_cached_world(canonical_root, &live.fingerprint)?
                        {
                            self.rust = Some(cached);
                            self.process_cache_hits = self.process_cache_hits.saturating_add(1);
                        } else if let Some(refreshed) =
                            self.try_incremental_refresh(canonical_root, live)?
                        {
                            insert_process_world(Arc::clone(&refreshed))?;
                            self.rust = Some(refreshed);
                        } else {
                            let receiver = spawn_rust_world_build(
                                preparation,
                                canonical_root.to_path_buf(),
                                live.clone(),
                            )?;
                            self.cold_builds = self.cold_builds.saturating_add(1);
                            self.rust = Some(wait_for_rust_preparation(&receiver, cancellation)?);
                        }
                    }
                }
            }
        }
        let prepared = self
            .rust
            .as_ref()
            .context("Rust control-layer lifecycle lost its prepared world")?;
        let prepared = prepared
            .lock()
            .map_err(|_| anyhow::anyhow!("Rust semantic world lock is poisoned"))?;
        let expected = &prepared.world.descriptor().world;
        let request = prepared
            .world
            .snapshot_for_request(expected)
            .context("prepared Rust semantic world was not ready before inference")?;
        let layer = request
            .into_streaming_layer(RUST_LAYER_MAX_SOURCE_BYTES, RUST_LAYER_MAX_CHECKPOINTS)
            .context("failed to create the request-local Rust streaming layer")?;
        Ok(Some(Box::new(layer)))
    }

    fn prepare_python_layer(
        &mut self,
        canonical_root: &Path,
        live: &crate::workspace::ContentSnapshot,
        cancellation: &dyn Fn() -> Result<()>,
    ) -> Result<Option<Box<dyn IncrementalAnalyzer + Send>>> {
        let dependencies = capture_python_dependencies(
            canonical_root,
            live,
            &self.python_dependency_authority,
            cancellation,
        )
        .context("failed to capture the local Python dependency world before inference")?;
        let dependency_sha256 = python_dependency_identity(live, &dependencies);
        let current_matches = self
            .python
            .as_ref()
            .map(|prepared| {
                prepared
                    .lock()
                    .map(|prepared| {
                        prepared.workspace_root == canonical_root
                            && prepared.source_sha256 == live.fingerprint
                            && prepared.dependency_sha256 == dependency_sha256
                    })
                    .map_err(|_| anyhow::anyhow!("Python semantic world lock is poisoned"))
            })
            .transpose()?
            == Some(true);
        if !current_matches {
            if let Some(cached) =
                process_cached_python_world(canonical_root, &live.fingerprint, &dependency_sha256)?
            {
                self.python = Some(cached);
                self.python_process_cache_hits = self.python_process_cache_hits.saturating_add(1);
            } else {
                let preparation = begin_python_world_preparation(
                    PythonWorldKey {
                        workspace_root: canonical_root.to_path_buf(),
                        source_sha256: live.fingerprint.clone(),
                        dependency_sha256: dependency_sha256.clone(),
                    },
                    cancellation,
                )?;
                match preparation {
                    PythonWorldPreparation::Shared(shared) => {
                        self.python = Some(shared);
                        self.python_process_cache_hits =
                            self.python_process_cache_hits.saturating_add(1);
                    }
                    PythonWorldPreparation::Owner(preparation) => {
                        if let Some(cached) = process_cached_python_world(
                            canonical_root,
                            &live.fingerprint,
                            &dependency_sha256,
                        )? {
                            self.python = Some(cached);
                            self.python_process_cache_hits =
                                self.python_process_cache_hits.saturating_add(1);
                        } else {
                            let receiver = spawn_python_world_build(
                                preparation,
                                canonical_root.to_path_buf(),
                                live.clone(),
                                dependencies,
                                dependency_sha256,
                                self.python_dependency_authority.clone(),
                            )?;
                            self.python_cold_builds = self.python_cold_builds.saturating_add(1);
                            self.python =
                                Some(wait_for_python_preparation(&receiver, cancellation)?);
                        }
                    }
                }
            }
        }
        let prepared = self
            .python
            .as_ref()
            .context("Python control-layer lifecycle lost its prepared world")?;
        let prepared = prepared
            .lock()
            .map_err(|_| anyhow::anyhow!("Python semantic world lock is poisoned"))?;
        let expected = &prepared.world.descriptor().world;
        let request = prepared
            .world
            .snapshot_for_request(expected)
            .context("prepared Python semantic world was not ready before inference")?;
        let layer = request
            .into_streaming_layer(PYTHON_LAYER_MAX_SOURCE_BYTES, PYTHON_LAYER_MAX_CHECKPOINTS)
            .context("failed to create the request-local Python streaming layer")?;
        Ok(Some(Box::new(layer)))
    }

    /// Independently replay a completed mutation immediately before executor entry. This never
    /// loads Cargo or revises a semantic world: inference must already have established an exact
    /// ready world, and any intervening workspace drift rejects the call so the next model turn can
    /// prepare again.
    pub(crate) fn validate_completed_mutation(
        &self,
        workspace_root: &Path,
        tools: &[BuiltInToolSchema],
        snapshot: &WorkspaceSnapshot,
        name: &str,
        arguments: &Value,
    ) -> Result<CompletionDecision> {
        if !matches!(
            name,
            "write_file" | "replace_file" | "edit_file" | "apply_patch"
        ) {
            return Ok(CompletionDecision::NotApplicable);
        }
        let canonical_root = workspace_root
            .canonicalize()
            .context("failed to canonicalize the semantic project before mutation execution")?;
        let live = crate::workspace::ContentSnapshot::capture(workspace_root)
            .context("failed to revalidate the semantic world before mutation execution")?;
        let bound_path = snapshot.bound_mutation_path();
        let validate_rust = tools_may_mutate_rust(tools, bound_path);
        let validate_python = tools_may_mutate_python(tools, bound_path);
        let current_python_dependency_sha256 = if validate_python {
            let dependencies = capture_python_dependencies(
                &canonical_root,
                &live,
                &self.python_dependency_authority,
                &|| Ok(()),
            )
            .context(
                "failed to revalidate the local Python dependency world before mutation execution",
            )?;
            Some(python_dependency_identity(&live, &dependencies))
        } else {
            None
        };
        let mut layers: Vec<Box<dyn IncrementalAnalyzer + Send>> = Vec::new();
        if validate_rust && let Some(prepared) = self.rust.as_ref() {
            let prepared = prepared
                .lock()
                .map_err(|_| anyhow::anyhow!("Rust semantic world lock is poisoned"))?;
            if prepared.workspace_root != canonical_root
                || prepared.source_sha256 != live.fingerprint
            {
                bail!(
                    "workspace changed after Rust semantic preparation; refusing to execute a mutation against a stale world"
                );
            }
            let expected = &prepared.world.descriptor().world;
            let request = prepared
                .world
                .snapshot_for_request(expected)
                .context("prepared Rust semantic world was not ready for execution replay")?;
            layers.push(Box::new(
                request
                    .into_streaming_layer(RUST_LAYER_MAX_SOURCE_BYTES, RUST_LAYER_MAX_CHECKPOINTS)
                    .context("failed to create the execution-time Rust replay layer")?,
            ));
        }
        if validate_python && let Some(prepared) = self.python.as_ref() {
            let prepared = prepared
                .lock()
                .map_err(|_| anyhow::anyhow!("Python semantic world lock is poisoned"))?;
            if prepared.workspace_root != canonical_root
                || prepared.source_sha256 != live.fingerprint
                || Some(prepared.dependency_sha256.as_str())
                    != current_python_dependency_sha256.as_deref()
            {
                bail!(
                    "workspace or local Python dependencies changed after semantic preparation; refusing to execute a mutation against a stale world"
                );
            }
            let expected = &prepared.world.descriptor().world;
            let request = prepared
                .world
                .snapshot_for_request(expected)
                .context("prepared Python semantic world was not ready for execution replay")?;
            layers.push(Box::new(
                request
                    .into_streaming_layer(
                        PYTHON_LAYER_MAX_SOURCE_BYTES,
                        PYTHON_LAYER_MAX_CHECKPOINTS,
                    )
                    .context("failed to create the execution-time Python replay layer")?,
            ));
        }
        if layers.is_empty() {
            return Ok(CompletionDecision::NotApplicable);
        }
        let stack = LanguageLayerStack::new(layers, program_snapshot(snapshot)?)
            .context("failed to start the execution-time native language-layer stack")?;
        let manifest = execution_manifest(tools, snapshot.clone());
        let gate = MutationCompletionGate::with_language_layers(manifest, stack)
            .context("failed to create the independent mutation replay gate")?;
        Ok(gate.evaluate_independent(name, arguments))
    }

    fn build_current_world(
        workspace_root: &Path,
        expected: crate::workspace::ContentSnapshot,
    ) -> Result<PreparedRustWorld> {
        let shadow = SemanticShadowWorkspace::capture(workspace_root, &expected)
            .context("failed to capture an immutable Rust semantic shadow")?;
        let configuration_sha256 = subset_identity(&expected, is_rust_configuration_input);
        let dependency_sha256 = subset_identity(&expected, is_rust_dependency_input);
        let config = RustProjectConfig {
            contract_version: RUST_LAYER_CONTRACT_VERSION,
            shadow_root: shadow.path().to_path_buf(),
            world_sha256: expected.fingerprint.clone(),
            configuration_sha256,
            dependency_sha256,
        };
        let world = RustProjectWorld::load_and_prime(config)
            .context("failed to load and prime rust-analyzer before inference")?;
        let after = crate::workspace::ContentSnapshot::capture(workspace_root)
            .context("failed to revalidate the Rust semantic world after priming")?;
        ensure_unchanged_during_rust_preparation(&expected, &after)?;
        tracing::info!(
            world_sha256 = %expected.fingerprint,
            load_millis = world.readiness_receipt().load_millis,
            prime_millis = world.readiness_receipt().prime_millis,
            targets = world.targets().len(),
            "Rust streaming semantic world is ready before inference"
        );
        Ok(PreparedRustWorld {
            workspace_root: workspace_root.to_path_buf(),
            source_sha256: expected.fingerprint.clone(),
            content: expected,
            _shadow: Arc::new(shadow),
            world,
        })
    }

    fn build_current_python_world(
        workspace_root: &Path,
        expected: crate::workspace::ContentSnapshot,
        dependencies: PythonDependencySnapshot,
        dependency_sha256: String,
        authority: crate::python_semantic_config::PythonExternalAuthority,
    ) -> Result<PreparedPythonWorld> {
        let shadow = SemanticShadowWorkspace::capture_with_extra_files(
            workspace_root,
            &expected,
            &dependencies.files,
        )
        .context("failed to capture an immutable Python semantic shadow")?;
        let config = PythonProjectConfig {
            contract_version: PYTHON_LAYER_CONTRACT_VERSION,
            shadow_root: shadow.path().to_path_buf(),
            first_party_roots: dependencies.first_party_roots.clone(),
            site_packages_roots: dependencies.site_packages_roots.clone(),
            external_imports_complete: dependencies.external_imports_complete,
            python_version: dependencies.python_version.clone(),
            python_platform: host_python_platform().to_string(),
            world_sha256: expected.fingerprint.clone(),
            configuration_sha256: subset_identity(&expected, is_python_configuration_input),
            dependency_sha256: dependency_sha256.clone(),
            max_files: PYTHON_LAYER_MAX_PROJECT_FILES,
            max_bytes: PYTHON_LAYER_MAX_PROJECT_BYTES,
        };
        let world = PythonProjectWorld::load_and_prime(config)
            .context("failed to load and prime Astral ty before inference")?;
        let after = crate::workspace::ContentSnapshot::capture(workspace_root)
            .context("failed to revalidate the Python semantic world after priming")?;
        ensure_unchanged_during_python_preparation(&expected, &after)?;
        let after_dependencies =
            capture_python_dependencies(workspace_root, &after, &authority, &|| Ok(()))
                .context("failed to revalidate the local Python dependency world after priming")?;
        if python_dependency_identity(&after, &after_dependencies) != dependency_sha256 {
            bail!(
                "local Python dependencies changed while Astral ty was loading; refusing Python-edit-capable inference until the controller recaptures one exact dependency image"
            );
        }
        tracing::info!(
            world_sha256 = %expected.fingerprint,
            dependency_sha256 = %dependency_sha256,
            dependency_files = dependencies.files.len(),
            dependency_roots = dependencies.site_packages_roots.len(),
            editable_first_party_roots = dependencies.first_party_roots.len(),
            external_imports_complete = dependencies.external_imports_complete,
            load_millis = world.readiness_receipt().load_millis,
            prime_millis = world.readiness_receipt().prime_millis,
            primed_queries = world.readiness_receipt().primed_queries,
            "Python streaming semantic world is ready before inference"
        );
        Ok(PreparedPythonWorld {
            workspace_root: workspace_root.to_path_buf(),
            source_sha256: expected.fingerprint,
            dependency_sha256,
            _shadow: Arc::new(shadow),
            world,
        })
    }

    fn try_incremental_refresh(
        &self,
        workspace_root: &Path,
        current: &crate::workspace::ContentSnapshot,
    ) -> Result<Option<PreparedRustHandle>> {
        let Some(previous) = self.rust.as_ref() else {
            return Ok(None);
        };
        let mut previous = previous
            .lock()
            .map_err(|_| anyhow::anyhow!("Rust semantic world lock is poisoned"))?;
        if previous.workspace_root != workspace_root {
            return Ok(None);
        }
        let changed_paths = previous.content.changed_paths(current);
        if changed_paths.is_empty()
            || changed_paths.iter().any(|path| {
                !path.ends_with(".rs")
                    || previous
                        .content
                        .paths
                        .get(path)
                        .is_none_or(|content| content.kind != "file")
                    || current
                        .paths
                        .get(path)
                        .is_none_or(|content| content.kind != "file")
            })
        {
            return Ok(None);
        }
        let configuration_sha256 = subset_identity(current, is_rust_configuration_input);
        let dependency_sha256 = subset_identity(current, is_rust_dependency_input);
        let changes = changed_paths
            .iter()
            .map(|path| {
                std::fs::read(workspace_root.join(path))
                    .with_context(|| format!("failed to read changed Rust source {path}"))
                    .map(|bytes| (path.clone(), bytes))
            })
            .collect::<Result<Vec<_>>>()?;
        let after = crate::workspace::ContentSnapshot::capture(workspace_root)
            .context("failed to revalidate incrementally refreshed Rust sources")?;
        if after.fingerprint != current.fingerprint {
            return Ok(None);
        }
        let config = RustProjectConfig {
            contract_version: RUST_LAYER_CONTRACT_VERSION,
            shadow_root: previous._shadow.path().to_path_buf(),
            world_sha256: current.fingerprint.clone(),
            configuration_sha256,
            dependency_sha256,
        };
        if let Err(error) = previous.world.refresh_existing_sources(config, &changes) {
            tracing::debug!(%error, "Rust streaming semantic world requires a cold rebuild");
            return Ok(None);
        }
        tracing::info!(
            world_sha256 = %current.fingerprint,
            changed_sources = changes.len(),
            prime_millis = previous.world.readiness_receipt().prime_millis,
            "Rust streaming semantic world was incrementally refreshed before inference"
        );
        previous.source_sha256 = current.fingerprint.clone();
        previous.content = current.clone();
        drop(previous);
        Ok(Some(Arc::clone(self.rust.as_ref().context(
            "incremental Rust refresh lost its prepared world",
        )?)))
    }

    #[cfg(test)]
    fn stats(&self) -> (u64, u64, u64) {
        (
            self.cold_builds,
            self.warm_requests,
            self.process_cache_hits,
        )
    }

    #[cfg(test)]
    fn python_stats(&self) -> (u64, u64, u64) {
        (
            self.python_cold_builds,
            self.python_warm_requests,
            self.python_process_cache_hits,
        )
    }
}

impl Drop for RustWorldPreparationGuard {
    fn drop(&mut self) {
        let mut state = self
            .coordinator
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.active.remove(&self.key);
        self.coordinator.ready.notify_all();
    }
}

impl Drop for PythonWorldPreparationGuard {
    fn drop(&mut self) {
        let mut state = self
            .coordinator
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.active.remove(&self.key);
        self.coordinator.ready.notify_all();
    }
}

impl RustWorldPreparationState {
    fn register_wait(&mut self, key: &RustWorldKey) -> Result<()> {
        let waiters = self.waiters.entry(key.clone()).or_default();
        *waiters = waiters
            .checked_add(1)
            .context("Rust semantic preparation waiter count overflowed")?;
        Ok(())
    }

    fn finish_wait(&mut self, key: &RustWorldKey) -> Option<PreparedRustHandle> {
        let completed = self.completed.get(key).cloned();
        if let Some(waiters) = self.waiters.get_mut(key) {
            *waiters = waiters.saturating_sub(1);
            if *waiters == 0 {
                self.waiters.remove(key);
                self.completed.remove(key);
            }
        }
        completed
    }
}

impl PythonWorldPreparationState {
    fn register_wait(&mut self, key: &PythonWorldKey) -> Result<()> {
        let waiters = self.waiters.entry(key.clone()).or_default();
        *waiters = waiters
            .checked_add(1)
            .context("Python semantic preparation waiter count overflowed")?;
        Ok(())
    }

    fn finish_wait(&mut self, key: &PythonWorldKey) -> Option<PreparedPythonHandle> {
        let completed = self.completed.get(key).cloned();
        if let Some(waiters) = self.waiters.get_mut(key) {
            *waiters = waiters.saturating_sub(1);
            if *waiters == 0 {
                self.waiters.remove(key);
                self.completed.remove(key);
            }
        }
        completed
    }
}

impl RustWorldPreparationPublisher {
    fn publish(&self, world: PreparedRustHandle) -> Result<()> {
        let mut state =
            self.coordinator.state.lock().map_err(|_| {
                anyhow::anyhow!("Rust semantic preparation coordinator is poisoned")
            })?;
        if state.waiters.get(&self.key).copied().unwrap_or(0) > 0 {
            state.completed.insert(self.key.clone(), world);
        }
        Ok(())
    }
}

impl PythonWorldPreparationPublisher {
    fn publish(&self, world: PreparedPythonHandle) -> Result<()> {
        let mut state =
            self.coordinator.state.lock().map_err(|_| {
                anyhow::anyhow!("Python semantic preparation coordinator is poisoned")
            })?;
        if state.waiters.get(&self.key).copied().unwrap_or(0) > 0 {
            state.completed.insert(self.key.clone(), world);
        }
        Ok(())
    }
}

fn begin_rust_world_preparation(
    key: RustWorldKey,
    cancellation: &dyn Fn() -> Result<()>,
) -> Result<RustWorldPreparation> {
    cancellation()?;
    let coordinator = RUST_WORLD_PREPARATIONS.get_or_init(Default::default);
    let mut state = coordinator
        .state
        .lock()
        .map_err(|_| anyhow::anyhow!("Rust semantic preparation coordinator is poisoned"))?;
    let mut registered = false;
    loop {
        if state.completed.contains_key(&key) {
            let completed = if registered {
                state.finish_wait(&key)
            } else {
                state.completed.get(&key).cloned()
            }
            .context("Rust semantic preparation handoff disappeared")?;
            return Ok(RustWorldPreparation::Shared(completed));
        }
        if !state.active.contains(&key)
            && state.active.len() < MAX_CONCURRENT_RUST_WORLD_PREPARATIONS
        {
            if registered {
                state.finish_wait(&key);
            }
            state.active.insert(key.clone());
            return Ok(RustWorldPreparation::Owner(RustWorldPreparationGuard {
                coordinator,
                key,
            }));
        }
        if !registered {
            state.register_wait(&key)?;
            registered = true;
        }
        let (next, _) = coordinator
            .ready
            .wait_timeout(state, PREPARATION_WAIT_POLL)
            .map_err(|_| anyhow::anyhow!("Rust semantic preparation coordinator is poisoned"))?;
        state = next;
        if let Err(error) = cancellation() {
            state.finish_wait(&key);
            return Err(error);
        }
    }
}

fn begin_python_world_preparation(
    key: PythonWorldKey,
    cancellation: &dyn Fn() -> Result<()>,
) -> Result<PythonWorldPreparation> {
    cancellation()?;
    let coordinator = PYTHON_WORLD_PREPARATIONS.get_or_init(Default::default);
    let mut state = coordinator
        .state
        .lock()
        .map_err(|_| anyhow::anyhow!("Python semantic preparation coordinator is poisoned"))?;
    let mut registered = false;
    loop {
        if state.completed.contains_key(&key) {
            let completed = if registered {
                state.finish_wait(&key)
            } else {
                state.completed.get(&key).cloned()
            }
            .context("Python semantic preparation handoff disappeared")?;
            return Ok(PythonWorldPreparation::Shared(completed));
        }
        if !state.active.contains(&key)
            && state.active.len() < MAX_CONCURRENT_PYTHON_WORLD_PREPARATIONS
        {
            if registered {
                state.finish_wait(&key);
            }
            state.active.insert(key.clone());
            return Ok(PythonWorldPreparation::Owner(PythonWorldPreparationGuard {
                coordinator,
                key,
            }));
        }
        if !registered {
            state.register_wait(&key)?;
            registered = true;
        }
        let (next, _) = coordinator
            .ready
            .wait_timeout(state, PREPARATION_WAIT_POLL)
            .map_err(|_| anyhow::anyhow!("Python semantic preparation coordinator is poisoned"))?;
        state = next;
        if let Err(error) = cancellation() {
            state.finish_wait(&key);
            return Err(error);
        }
    }
}

fn spawn_rust_world_build(
    preparation: RustWorldPreparationGuard,
    workspace_root: PathBuf,
    expected: crate::workspace::ContentSnapshot,
) -> Result<Receiver<Result<PreparedRustHandle>>> {
    let publisher = RustWorldPreparationPublisher {
        coordinator: preparation.coordinator,
        key: preparation.key.clone(),
    };
    spawn_rust_preparation_worker(preparation, move || {
        let prepared = Arc::new(Mutex::new(ControlLayerLifecycle::build_current_world(
            &workspace_root,
            expected,
        )?));
        insert_process_world(Arc::clone(&prepared))?;
        publisher.publish(Arc::clone(&prepared))?;
        Ok(prepared)
    })
}

fn spawn_python_world_build(
    preparation: PythonWorldPreparationGuard,
    workspace_root: PathBuf,
    expected: crate::workspace::ContentSnapshot,
    dependencies: PythonDependencySnapshot,
    dependency_sha256: String,
    authority: crate::python_semantic_config::PythonExternalAuthority,
) -> Result<Receiver<Result<PreparedPythonHandle>>> {
    let publisher = PythonWorldPreparationPublisher {
        coordinator: preparation.coordinator,
        key: preparation.key.clone(),
    };
    spawn_python_preparation_worker(preparation, move || {
        let prepared = Arc::new(Mutex::new(
            ControlLayerLifecycle::build_current_python_world(
                &workspace_root,
                expected,
                dependencies,
                dependency_sha256,
                authority,
            )?,
        ));
        insert_process_python_world(Arc::clone(&prepared))?;
        publisher.publish(Arc::clone(&prepared))?;
        Ok(prepared)
    })
}

fn spawn_rust_preparation_worker<T>(
    preparation: RustWorldPreparationGuard,
    prepare: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<Receiver<Result<T>>>
where
    T: Send + 'static,
{
    let (sender, receiver) = sync_channel(1);
    thread::Builder::new()
        .name("pb-rust-world-preparation".to_string())
        .spawn(move || {
            let result = prepare();
            drop(preparation);
            // A cancelled initiating request may have dropped its receiver. The worker still
            // finishes cache publication and releases the single-flight guard before this send.
            if let Err(disconnected) = sender.send(result)
                && let Err(error) = disconnected.0
            {
                tracing::warn!(
                    %error,
                    "detached Rust semantic preparation failed after its request stopped waiting"
                );
            }
        })
        .context("failed to start the Rust semantic preparation worker")?;
    Ok(receiver)
}

fn spawn_python_preparation_worker<T>(
    preparation: PythonWorldPreparationGuard,
    prepare: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<Receiver<Result<T>>>
where
    T: Send + 'static,
{
    let (sender, receiver) = sync_channel(1);
    thread::Builder::new()
        .name("pb-python-world-preparation".to_string())
        .spawn(move || {
            let result = prepare();
            drop(preparation);
            if let Err(disconnected) = sender.send(result)
                && let Err(error) = disconnected.0
            {
                tracing::warn!(
                    %error,
                    "detached Python semantic preparation failed after its request stopped waiting"
                );
            }
        })
        .context("failed to start the Python semantic preparation worker")?;
    Ok(receiver)
}

fn wait_for_rust_preparation<T>(
    receiver: &Receiver<Result<T>>,
    cancellation: &dyn Fn() -> Result<()>,
) -> Result<T> {
    loop {
        cancellation()?;
        match receiver.recv_timeout(PREPARATION_WAIT_POLL) {
            Ok(result) => {
                cancellation()?;
                return result;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                bail!("Rust semantic preparation worker stopped without a result")
            }
        }
    }
}

fn wait_for_python_preparation<T>(
    receiver: &Receiver<Result<T>>,
    cancellation: &dyn Fn() -> Result<()>,
) -> Result<T> {
    loop {
        cancellation()?;
        match receiver.recv_timeout(PREPARATION_WAIT_POLL) {
            Ok(result) => {
                cancellation()?;
                return result;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                bail!("Python semantic preparation worker stopped without a result")
            }
        }
    }
}

fn ensure_unchanged_during_rust_preparation(
    expected: &crate::workspace::ContentSnapshot,
    after: &crate::workspace::ContentSnapshot,
) -> Result<()> {
    if after.fingerprint != expected.fingerprint {
        bail!(
            "workspace changed while rust-analyzer was loading; refusing Rust-edit-capable inference until the controller recaptures one exact mutation and semantic snapshot"
        );
    }
    Ok(())
}

fn ensure_unchanged_during_python_preparation(
    expected: &crate::workspace::ContentSnapshot,
    after: &crate::workspace::ContentSnapshot,
) -> Result<()> {
    if after.fingerprint != expected.fingerprint {
        bail!(
            "workspace changed while Astral ty was loading; refusing Python-edit-capable inference until the controller recaptures one exact mutation and semantic snapshot"
        );
    }
    Ok(())
}

fn execution_manifest(tools: &[BuiltInToolSchema], workspace: WorkspaceSnapshot) -> CollarManifest {
    let exposed = |name: &str| tools.iter().any(|tool| tool.name == name);
    CollarManifest {
        contract_version: 1,
        dialect: ToolDialect::QwenJson,
        mode: ToolConstraintMode::ToolsAllowed,
        tools: tools
            .iter()
            .map(|tool| ExposedTool {
                name: tool.name.clone(),
                input_schema: tool.input_schema.clone(),
            })
            .collect(),
        terminal_tools: Vec::new(),
        mutation_policy: MutationPolicy {
            allow_write_file: exposed("write_file"),
            allow_replace_file: exposed("replace_file") || exposed("edit_file"),
            allow_apply_patch: exposed("apply_patch"),
            max_mutation_calls_per_batch: 1,
        },
        workspace,
        limits: CollarLimits {
            max_argument_bytes: crate::inference::flashmoe::constraints::MAX_COLLAR_ARGUMENT_BYTES,
            max_snapshot_bytes: crate::inference::flashmoe::constraints::MAX_COLLAR_SNAPSHOT_BYTES,
            max_files: crate::inference::flashmoe::constraints::MAX_COLLAR_FILES,
            max_patch_hunks: crate::inference::flashmoe::constraints::MAX_COLLAR_PATCH_HUNKS,
        },
    }
}

fn process_cached_world(
    workspace_root: &Path,
    source_sha256: &str,
) -> Result<Option<PreparedRustHandle>> {
    let mut cache = RUST_WORLD_CACHE
        .get_or_init(|| Mutex::new(VecDeque::new()))
        .lock()
        .map_err(|_| anyhow::anyhow!("Rust semantic world cache lock is poisoned"))?;
    let mut found = None;
    for (index, entry) in cache.iter().enumerate() {
        let entry = entry
            .lock()
            .map_err(|_| anyhow::anyhow!("Rust semantic world lock is poisoned"))?;
        if entry.workspace_root == workspace_root && entry.source_sha256 == source_sha256 {
            found = Some(index);
            break;
        }
    }
    let Some(index) = found else {
        return Ok(None);
    };
    let entry = cache
        .remove(index)
        .context("Rust semantic world cache entry disappeared")?;
    cache.push_back(Arc::clone(&entry));
    Ok(Some(entry))
}

fn insert_process_world(world: PreparedRustHandle) -> Result<()> {
    let mut cache = RUST_WORLD_CACHE
        .get_or_init(|| Mutex::new(VecDeque::new()))
        .lock()
        .map_err(|_| anyhow::anyhow!("Rust semantic world cache lock is poisoned"))?;
    let identity = {
        let world = world
            .lock()
            .map_err(|_| anyhow::anyhow!("Rust semantic world lock is poisoned"))?;
        (world.workspace_root.clone(), world.source_sha256.clone())
    };
    let mut retained = VecDeque::with_capacity(cache.len());
    while let Some(entry) = cache.pop_front() {
        let duplicate = {
            let entry = entry
                .lock()
                .map_err(|_| anyhow::anyhow!("Rust semantic world lock is poisoned"))?;
            entry.workspace_root == identity.0 && entry.source_sha256 == identity.1
        };
        if !duplicate {
            retained.push_back(entry);
        }
    }
    *cache = retained;
    while cache.len() >= MAX_PROCESS_RUST_WORLDS {
        cache.pop_front();
    }
    cache.push_back(world);
    Ok(())
}

fn process_cached_python_world(
    workspace_root: &Path,
    source_sha256: &str,
    dependency_sha256: &str,
) -> Result<Option<PreparedPythonHandle>> {
    let mut cache = PYTHON_WORLD_CACHE
        .get_or_init(|| Mutex::new(VecDeque::new()))
        .lock()
        .map_err(|_| anyhow::anyhow!("Python semantic world cache lock is poisoned"))?;
    let mut found = None;
    for (index, entry) in cache.iter().enumerate() {
        let entry = entry
            .lock()
            .map_err(|_| anyhow::anyhow!("Python semantic world lock is poisoned"))?;
        if entry.workspace_root == workspace_root
            && entry.source_sha256 == source_sha256
            && entry.dependency_sha256 == dependency_sha256
        {
            found = Some(index);
            break;
        }
    }
    let Some(index) = found else {
        return Ok(None);
    };
    let entry = cache
        .remove(index)
        .context("Python semantic world cache entry disappeared")?;
    cache.push_back(Arc::clone(&entry));
    Ok(Some(entry))
}

fn insert_process_python_world(world: PreparedPythonHandle) -> Result<()> {
    let mut cache = PYTHON_WORLD_CACHE
        .get_or_init(|| Mutex::new(VecDeque::new()))
        .lock()
        .map_err(|_| anyhow::anyhow!("Python semantic world cache lock is poisoned"))?;
    let identity = {
        let world = world
            .lock()
            .map_err(|_| anyhow::anyhow!("Python semantic world lock is poisoned"))?;
        (
            world.workspace_root.clone(),
            world.source_sha256.clone(),
            world.dependency_sha256.clone(),
        )
    };
    let mut retained = VecDeque::with_capacity(cache.len());
    while let Some(entry) = cache.pop_front() {
        let duplicate = {
            let entry = entry
                .lock()
                .map_err(|_| anyhow::anyhow!("Python semantic world lock is poisoned"))?;
            entry.workspace_root == identity.0
                && entry.source_sha256 == identity.1
                && entry.dependency_sha256 == identity.2
        };
        if !duplicate {
            retained.push_back(entry);
        }
    }
    *cache = retained;
    while cache.len() >= MAX_PROCESS_PYTHON_WORLDS {
        cache.pop_front();
    }
    cache.push_back(world);
    Ok(())
}

fn program_snapshot(snapshot: &WorkspaceSnapshot) -> Result<ProgramSnapshot> {
    let files = snapshot
        .entries()
        .filter_map(|entry| {
            let profile = SyntaxProfile::for_path(&entry.path)?;
            Some(ProgramFile {
                path: entry.path.clone(),
                language: profile.language_id(),
                bytes: entry.bytes.clone(),
            })
        })
        .collect();
    Ok(ProgramSnapshot { files })
}

fn tools_may_mutate_rust(
    tools: &[BuiltInToolSchema],
    bound_mutation_path: Option<&pb_control_collar::mutation::LogicalPath>,
) -> bool {
    tools.iter().any(|tool| match tool.name.as_str() {
        "apply_patch" => true,
        "write_file" | "replace_file" | "edit_file" => bound_mutation_path.map_or_else(
            || {
                constrained_paths(&tool.input_schema).is_none_or(|paths| {
                    paths.is_empty() || paths.iter().any(|path| path.ends_with(".rs"))
                })
            },
            |path| path.as_str().ends_with(".rs"),
        ),
        _ => false,
    })
}

fn tools_may_mutate_python(
    tools: &[BuiltInToolSchema],
    bound_mutation_path: Option<&pb_control_collar::mutation::LogicalPath>,
) -> bool {
    tools.iter().any(|tool| match tool.name.as_str() {
        "apply_patch" => true,
        "write_file" | "replace_file" | "edit_file" => bound_mutation_path.map_or_else(
            || {
                constrained_paths(&tool.input_schema).is_none_or(|paths| {
                    paths.is_empty()
                        || paths
                            .iter()
                            .any(|path| path.ends_with(".py") || path.ends_with(".pyi"))
                })
            },
            |path| path.as_str().ends_with(".py") || path.as_str().ends_with(".pyi"),
        ),
        _ => false,
    })
}

/// `Some` means the schema proves the complete finite path set. `None` means the model retains
/// authority to choose another path, so Rust must be considered reachable.
fn constrained_paths(schema: &Value) -> Option<Vec<&str>> {
    let path = schema.get("properties")?.get("path")?;
    if let Some(value) = path.get("const").and_then(Value::as_str) {
        return Some(vec![value]);
    }
    path.get("enum")?
        .as_array()
        .map(|values| values.iter().filter_map(Value::as_str).collect::<Vec<_>>())
}

fn capture_python_dependencies(
    workspace_root: &Path,
    snapshot: &crate::workspace::ContentSnapshot,
    external_authority: &crate::python_semantic_config::PythonExternalAuthority,
    cancellation: &dyn Fn() -> Result<()>,
) -> Result<PythonDependencySnapshot> {
    cancellation()?;
    if snapshot.paths.keys().any(|path| {
        path == PYTHON_DEPENDENCY_SHADOW_PREFIX
            || path.starts_with(&format!("{PYTHON_DEPENDENCY_SHADOW_PREFIX}/"))
    }) {
        bail!(
            "workspace path collides with the controller-reserved Python dependency shadow prefix"
        );
    }

    let project_policy = PythonSemanticConfig::load(workspace_root)?;
    let authority_path = workspace_root.join(PYTHON_SEMANTIC_CONFIG_PATH);
    if authority_path.exists()
        && !snapshot
            .paths
            .get(PYTHON_SEMANTIC_CONFIG_PATH)
            .is_some_and(|entry| entry.kind == "file")
    {
        bail!(
            "{} must be a controller-observed project file before it can grant native Python dependency authority",
            PYTHON_SEMANTIC_CONFIG_PATH
        );
    }
    let canonical_workspace = workspace_root
        .canonicalize()
        .context("failed to resolve the Python semantic workspace root")?;
    let allowed_editable_roots = external_authority
        .editable_roots
        .iter()
        .map(|path| resolve_declared_python_directory(workspace_root, path, "editable root"))
        .collect::<Result<BTreeSet<_>>>()?;

    if project_policy.environment.is_some() && external_authority.environment.is_some() {
        bail!("native Python environment is selected by both project policy and user authority");
    }
    let environment_candidates = if let Some(environment) = external_authority.environment.as_ref()
    {
        let resolved =
            resolve_declared_python_directory(workspace_root, environment, "environment")?;
        vec![(resolved.clone(), resolved)]
    } else if let Some(environment) = project_policy.environment.as_ref() {
        let resolved =
            resolve_declared_python_directory(workspace_root, environment, "environment")?;
        if !resolved.starts_with(&canonical_workspace) {
            bail!(
                "repository-owned {} may select only an environment inside the workspace",
                PYTHON_SEMANTIC_CONFIG_PATH
            );
        }
        let relative = resolved
            .strip_prefix(&canonical_workspace)
            .context("project Python environment escaped its workspace identity")?
            .to_path_buf();
        vec![(relative, resolved)]
    } else {
        let mut project_roots = BTreeSet::from([PathBuf::new()]);
        for path in snapshot
            .paths
            .keys()
            .filter(|path| is_python_project_marker(path))
        {
            let path = Path::new(path);
            project_roots.insert(path.parent().unwrap_or_else(|| Path::new("")).to_path_buf());
        }
        project_roots
            .into_iter()
            .flat_map(|project_root| {
                [".venv", "venv"].map(move |name| {
                    let relative = project_root.join(name);
                    let environment = workspace_root.join(&relative);
                    (relative, environment)
                })
            })
            .collect()
    };

    let mut environments = Vec::new();
    let mut observed = Vec::new();
    for (relative, environment) in environment_candidates {
        cancellation()?;
        let metadata = match std::fs::symlink_metadata(&environment) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                if external_authority.environment.is_some() || project_policy.environment.is_some()
                {
                    return Err(error).with_context(|| {
                        format!(
                            "failed to inspect configured Python environment {}",
                            environment.display()
                        )
                    });
                }
                continue;
            }
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            observed.push((relative, b"unsafe-environment-root".to_vec()));
            continue;
        }
        let config_path = environment.join("pyvenv.cfg");
        let config =
            match read_python_dependency_file(&config_path, PYTHON_DEPENDENCY_MAX_CONFIG_BYTES) {
                Ok(config) => config,
                Err(error) => {
                    tracing::debug!(
                        %error,
                        path = %config_path.display(),
                        "local Python environment metadata is not safely capturable"
                    );
                    observed.push((relative, b"unreadable-pyvenv-config".to_vec()));
                    continue;
                }
            };
        observed.push((relative.clone(), config.clone()));
        let python_version = match parse_pyvenv_python_version(&config) {
            Ok(version) => version,
            Err(error) => {
                tracing::debug!(
                    %error,
                    path = %config_path.display(),
                    "local Python environment version is not safely qualified"
                );
                continue;
            }
        };
        if pyvenv_includes_system_site_packages(&config)? {
            observed.push((relative, b"system-site-packages-enabled".to_vec()));
            continue;
        }
        let site_packages_relative = if cfg!(target_os = "windows") {
            relative.join("Lib").join("site-packages")
        } else {
            relative
                .join("lib")
                .join(format!("python{python_version}"))
                .join("site-packages")
        };
        let site_packages = workspace_root.join(&site_packages_relative);
        let site_metadata = match std::fs::symlink_metadata(&site_packages) {
            Ok(metadata) => metadata,
            Err(error) => {
                tracing::debug!(
                    %error,
                    path = %site_packages.display(),
                    "local Python environment has no safely capturable site-packages root"
                );
                continue;
            }
        };
        if site_metadata.file_type().is_symlink() || !site_metadata.is_dir() {
            tracing::debug!(
                path = %site_packages.display(),
                "local Python site-packages root is not a real directory"
            );
            continue;
        }
        environments.push((
            relative,
            config,
            python_version,
            site_packages_relative,
            site_packages,
        ));
    }

    if environments.len() != 1 {
        let marker = if environments.is_empty() {
            "no-qualified-local-environment"
        } else {
            "ambiguous-local-environments"
        };
        return Ok(unqualified_python_dependencies(marker, &observed));
    }

    let (environment_relative, config, python_version, site_source_relative, site_source) =
        environments
            .pop()
            .context("Python environment disappeared")?;
    let canonical_site_source = site_source.canonicalize().with_context(|| {
        format!(
            "failed to resolve local Python site-packages root {}",
            site_source.display()
        )
    })?;
    let site_shadow = PathBuf::from(PYTHON_DEPENDENCY_SHADOW_PREFIX)
        .join("site-packages")
        .join("0");
    let mut site_packages_roots = vec![site_shadow.clone()];
    let mut files = Vec::new();
    let mut first_party_roots = Vec::new();
    let mut selected_bytes = config.len();
    let mut scanned_entries = 0usize;
    let external_imports_complete = true;
    let mut unsafe_search_path = false;
    let mut pth_search_paths = Vec::new();
    let mut digest = Sha256::new();
    digest.update(b"pb-python-dependency-image-v1\0");
    hash_python_dependency_record(
        &mut digest,
        environment_relative.to_string_lossy().as_bytes(),
        &config,
    );
    files.push(SemanticShadowExtraFile {
        path: pb_control_collar::mutation::LogicalPath::parse(format!(
            "{PYTHON_DEPENDENCY_SHADOW_PREFIX}/environment/pyvenv.cfg"
        ))?,
        bytes: config,
    });

    // Dependency identity must not inherit filesystem enumeration order. `sort_by_file_name`
    // gives the capture, shadow image, and digest one stable traversal for the same exact tree.
    for entry in walkdir::WalkDir::new(&site_source)
        .follow_links(false)
        .sort_by_file_name()
    {
        cancellation()?;
        let entry = entry.with_context(|| {
            format!(
                "failed to walk local Python dependencies under {}",
                site_source.display()
            )
        })?;
        if entry.depth() == 0 {
            continue;
        }
        scanned_entries = scanned_entries.saturating_add(1);
        if scanned_entries > PYTHON_LAYER_MAX_PROJECT_FILES || entry.path().to_str().is_none() {
            unsafe_search_path = true;
            break;
        }
        let relative = entry.path().strip_prefix(&site_source).with_context(|| {
            format!(
                "Python dependency path escaped site-packages: {}",
                entry.path().display()
            )
        })?;
        if entry.file_type().is_symlink() {
            unsafe_search_path = true;
            break;
        }
        if !entry.file_type().is_file() {
            continue;
        }
        if is_python_native_module(relative) || is_unsupported_python_search_artifact(relative) {
            unsafe_search_path = true;
            break;
        }
        if !is_python_dependency_capture_file(relative) {
            continue;
        }
        let remaining = PYTHON_LAYER_MAX_PROJECT_BYTES.saturating_sub(selected_bytes);
        if entry
            .metadata()
            .with_context(|| {
                format!(
                    "failed to inspect local Python dependency {}",
                    entry.path().display()
                )
            })?
            .len()
            > remaining as u64
        {
            unsafe_search_path = true;
            break;
        }
        let bytes =
            read_python_dependency_file(entry.path(), remaining as u64).with_context(|| {
                format!(
                    "failed to snapshot local Python dependency {}",
                    entry.path().display()
                )
            })?;
        if relative.components().count() == 1
            && relative
                .extension()
                .and_then(|extension| extension.to_str())
                == Some("pth")
        {
            let pth = match python_pth_policy(&bytes) {
                Ok(policy) => policy,
                Err(_) => {
                    unsafe_search_path = true;
                    break;
                }
            };
            if pth.has_import_hook {
                unsafe_search_path = true;
                break;
            }
            pth_search_paths.extend(pth.search_paths);
        }
        selected_bytes = selected_bytes
            .checked_add(bytes.len())
            .context("local Python dependency byte count overflowed")?;
        if selected_bytes > PYTHON_LAYER_MAX_PROJECT_BYTES {
            bail!("local Python dependency image exceeds the bounded byte count");
        }
        let relative_text = slash_relative_path(relative)?;
        hash_python_dependency_record(&mut digest, relative_text.as_bytes(), &bytes);
        files.push(SemanticShadowExtraFile {
            path: pb_control_collar::mutation::LogicalPath::parse(
                site_shadow
                    .join(relative)
                    .to_string_lossy()
                    .replace('\\', "/"),
            )?,
            bytes,
        });
    }

    if !unsafe_search_path {
        let mut captured_external_roots = BTreeSet::new();
        for path in pth_search_paths {
            cancellation()?;
            let path = Path::new(&path);
            let candidate = if path.is_absolute() {
                path.to_path_buf()
            } else {
                site_source.join(path)
            };
            let metadata = match std::fs::symlink_metadata(&candidate) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(_) => {
                    unsafe_search_path = true;
                    break;
                }
            };
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                unsafe_search_path = true;
                break;
            }
            let target = candidate.canonicalize().with_context(|| {
                format!(
                    "failed to resolve Python editable root {}",
                    candidate.display()
                )
            })?;
            if target.starts_with(&canonical_site_source) {
                let relative = target
                    .strip_prefix(&canonical_site_source)
                    .with_context(|| {
                        format!(
                            "Python .pth target escaped site-packages identity: {}",
                            target.display()
                        )
                    })?;
                if relative.as_os_str().is_empty() {
                    continue;
                }
                let root_shadow = site_shadow.join(relative);
                let prefix = format!("{}/", root_shadow.to_string_lossy().replace('\\', "/"));
                if files
                    .iter()
                    .any(|file| file.path.as_str().starts_with(&prefix))
                    && !site_packages_roots.contains(&root_shadow)
                {
                    if site_packages_roots
                        .len()
                        .saturating_add(first_party_roots.len())
                        >= PYTHON_DEPENDENCY_MAX_SEARCH_ROOTS
                    {
                        unsafe_search_path = true;
                        break;
                    }
                    hash_python_dependency_record(
                        &mut digest,
                        b"site-packages-pth-root",
                        slash_relative_path(relative)?.as_bytes(),
                    );
                    site_packages_roots.push(root_shadow);
                }
                continue;
            }
            if target.starts_with(&canonical_workspace) {
                let relative = target.strip_prefix(&canonical_workspace).with_context(|| {
                    format!(
                        "Python editable root escaped workspace identity: {}",
                        target.display()
                    )
                })?;
                if relative.as_os_str().is_empty() {
                    continue;
                }
                if !validate_project_python_root(
                    &canonical_workspace,
                    relative,
                    snapshot,
                    cancellation,
                )? {
                    unsafe_search_path = true;
                    break;
                }
                let relative = PathBuf::from(slash_relative_path(relative)?);
                if !first_party_roots.contains(&relative) {
                    // Naming explicit first-party roots disables the layer's implicit project-root
                    // default, so the first editable consumes two entries: the project root and
                    // the editable root itself.
                    let additional_roots = if first_party_roots.is_empty() { 2 } else { 1 };
                    if site_packages_roots
                        .len()
                        .saturating_add(first_party_roots.len())
                        .saturating_add(additional_roots)
                        > PYTHON_DEPENDENCY_MAX_SEARCH_ROOTS
                    {
                        unsafe_search_path = true;
                        break;
                    }
                    if first_party_roots.is_empty() {
                        first_party_roots.push(PathBuf::new());
                    }
                    hash_python_dependency_record(
                        &mut digest,
                        b"editable-first-party-root",
                        relative.to_string_lossy().as_bytes(),
                    );
                    first_party_roots.push(relative);
                }
                continue;
            }
            if !allowed_editable_roots.contains(&target) {
                unsafe_search_path = true;
                break;
            }
            if !captured_external_roots.insert(target.clone()) {
                continue;
            }
            if site_packages_roots
                .len()
                .saturating_add(first_party_roots.len())
                >= PYTHON_DEPENDENCY_MAX_SEARCH_ROOTS
            {
                unsafe_search_path = true;
                break;
            }
            let root_index = captured_external_roots.len();
            let root_shadow = PathBuf::from(PYTHON_DEPENDENCY_SHADOW_PREFIX)
                .join("site-packages")
                .join(root_index.to_string());
            hash_python_dependency_record(
                &mut digest,
                b"editable-external-root",
                target.to_string_lossy().as_bytes(),
            );
            if !capture_external_python_root(
                &target,
                &root_shadow,
                &mut files,
                &mut digest,
                &mut selected_bytes,
                &mut scanned_entries,
                cancellation,
            )? {
                unsafe_search_path = true;
                break;
            }
            site_packages_roots.push(root_shadow);
        }
    }

    if unsafe_search_path {
        observed.push((environment_relative, b"unsupported-search-path".to_vec()));
        return Ok(unqualified_python_dependencies(
            "unsupported-local-environment-search-path",
            &observed,
        ));
    }
    digest.update([u8::from(external_imports_complete)]);
    digest.update(site_source_relative.to_string_lossy().as_bytes());
    Ok(PythonDependencySnapshot {
        fingerprint: format!("{:x}", digest.finalize()),
        python_version,
        first_party_roots,
        site_packages_roots,
        external_imports_complete,
        files,
    })
}

fn unqualified_python_dependencies(
    marker: &str,
    observed: &[(PathBuf, Vec<u8>)],
) -> PythonDependencySnapshot {
    let mut digest = Sha256::new();
    digest.update(b"pb-python-dependency-image-v1\0");
    digest.update(marker.as_bytes());
    for (path, bytes) in observed {
        hash_python_dependency_record(&mut digest, path.to_string_lossy().as_bytes(), bytes);
    }
    PythonDependencySnapshot {
        fingerprint: format!("{:x}", digest.finalize()),
        python_version: "3.12".to_string(),
        first_party_roots: Vec::new(),
        site_packages_roots: Vec::new(),
        external_imports_complete: false,
        files: Vec::new(),
    }
}

fn resolve_declared_python_directory(
    workspace_root: &Path,
    declared: &Path,
    label: &str,
) -> Result<PathBuf> {
    let candidate = if declared.is_absolute() {
        declared.to_path_buf()
    } else {
        workspace_root.join(declared)
    };
    let metadata = std::fs::symlink_metadata(&candidate).with_context(|| {
        format!(
            "failed to inspect configured Python {label} {}",
            candidate.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!(
            "configured Python {label} must be a real directory: {}",
            candidate.display()
        );
    }
    candidate.canonicalize().with_context(|| {
        format!(
            "failed to resolve configured Python {label} {}",
            candidate.display()
        )
    })
}

fn validate_project_python_root(
    workspace_root: &Path,
    relative_root: &Path,
    snapshot: &crate::workspace::ContentSnapshot,
    cancellation: &dyn Fn() -> Result<()>,
) -> Result<bool> {
    let source = workspace_root.join(relative_root);
    let mut entries = 0usize;
    for entry in walkdir::WalkDir::new(&source)
        .follow_links(false)
        .sort_by_file_name()
    {
        cancellation()?;
        let entry = entry.with_context(|| {
            format!(
                "failed to inspect project-local Python editable root {}",
                source.display()
            )
        })?;
        if entry.depth() == 0 {
            continue;
        }
        entries = entries.saturating_add(1);
        if entries > PYTHON_LAYER_MAX_PROJECT_FILES
            || entry.file_type().is_symlink()
            || entry.path().to_str().is_none()
        {
            return Ok(false);
        }
        if !entry.file_type().is_file() {
            continue;
        }
        let relative = entry.path().strip_prefix(workspace_root).with_context(|| {
            format!(
                "project-local Python editable path escaped the workspace: {}",
                entry.path().display()
            )
        })?;
        if is_python_native_module(relative) || is_unsupported_python_search_artifact(relative) {
            return Ok(false);
        }
        if !is_python_dependency_capture_file(relative) {
            continue;
        }
        let logical = slash_relative_path(relative)?;
        if !snapshot
            .paths
            .get(&logical)
            .is_some_and(|content| content.kind == "file")
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn capture_external_python_root(
    source: &Path,
    shadow: &Path,
    files: &mut Vec<SemanticShadowExtraFile>,
    digest: &mut Sha256,
    selected_bytes: &mut usize,
    scanned_entries: &mut usize,
    cancellation: &dyn Fn() -> Result<()>,
) -> Result<bool> {
    for entry in walkdir::WalkDir::new(source)
        .follow_links(false)
        .sort_by_file_name()
    {
        cancellation()?;
        let entry = entry.with_context(|| {
            format!(
                "failed to walk configured Python editable root {}",
                source.display()
            )
        })?;
        if entry.depth() == 0 {
            continue;
        }
        *scanned_entries = scanned_entries.saturating_add(1);
        if *scanned_entries > PYTHON_LAYER_MAX_PROJECT_FILES
            || entry.file_type().is_symlink()
            || entry.path().to_str().is_none()
        {
            return Ok(false);
        }
        if !entry.file_type().is_file() {
            continue;
        }
        let relative = entry.path().strip_prefix(source).with_context(|| {
            format!(
                "configured Python editable path escaped its root: {}",
                entry.path().display()
            )
        })?;
        if is_python_native_module(relative)
            || is_unsupported_python_search_artifact(relative)
            || relative
                .extension()
                .and_then(|extension| extension.to_str())
                == Some("pth")
        {
            return Ok(false);
        }
        if !is_python_dependency_capture_file(relative) {
            continue;
        }
        let remaining = PYTHON_LAYER_MAX_PROJECT_BYTES.saturating_sub(*selected_bytes);
        let bytes =
            read_python_dependency_file(entry.path(), remaining as u64).with_context(|| {
                format!(
                    "failed to snapshot configured Python editable dependency {}",
                    entry.path().display()
                )
            })?;
        *selected_bytes = selected_bytes
            .checked_add(bytes.len())
            .context("configured Python editable byte count overflowed")?;
        let relative_text = slash_relative_path(relative)?;
        hash_python_dependency_record(digest, relative_text.as_bytes(), &bytes);
        files.push(SemanticShadowExtraFile {
            path: pb_control_collar::mutation::LogicalPath::parse(
                shadow.join(relative).to_string_lossy().replace('\\', "/"),
            )?,
            bytes,
        });
    }
    Ok(true)
}

fn hash_python_dependency_record(digest: &mut Sha256, path: &[u8], bytes: &[u8]) {
    digest.update((path.len() as u64).to_le_bytes());
    digest.update(path);
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn read_python_dependency_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        bail!("Python dependency input is not a bounded regular file");
    }
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len())
            .unwrap_or(usize::MAX)
            .min(1024 * 1024),
    );
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != metadata.len() || bytes.len() as u64 > max_bytes {
        bail!("Python dependency input changed size while being captured");
    }
    Ok(bytes)
}

fn parse_pyvenv_python_version(config: &[u8]) -> Result<String> {
    let config = std::str::from_utf8(config).context("pyvenv.cfg is not UTF-8")?;
    for key in ["version", "version_info"] {
        if let Some(value) = config.lines().find_map(|line| {
            let (candidate, value) = line.split_once('=')?;
            candidate
                .trim()
                .eq_ignore_ascii_case(key)
                .then(|| value.trim())
        }) {
            let mut components = value
                .split(|character: char| !character.is_ascii_digit())
                .filter(|component| !component.is_empty());
            let major = components
                .next()
                .context("Python version has no major component")?;
            let minor = components
                .next()
                .context("Python version has no minor component")?;
            return Ok(format!("{major}.{minor}"));
        }
    }
    bail!("pyvenv.cfg contains neither version nor version_info")
}

fn pyvenv_includes_system_site_packages(config: &[u8]) -> Result<bool> {
    let config = std::str::from_utf8(config).context("pyvenv.cfg is not UTF-8")?;
    Ok(config.lines().any(|line| {
        line.split_once('=').is_some_and(|(key, value)| {
            key.trim()
                .eq_ignore_ascii_case("include-system-site-packages")
                && value.trim().eq_ignore_ascii_case("true")
        })
    }))
}

fn elapsed_millis(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Exercise the exact production lifecycle without invoking a model. The caller owns fixture
/// construction and process isolation; this function proves that cold preparation, warm request
/// construction, process-cache reuse, and both final replay outcomes cross the ordinary barriers.
pub(crate) fn qualify_python_world_fixture(
    workspace_root: &Path,
    target_path: &str,
    invalid_source: &str,
    valid_source: &str,
) -> Result<PythonWorldQualificationObservation> {
    let logical_path = pb_control_collar::mutation::LogicalPath::parse(target_path.to_string())?;
    let snapshot = WorkspaceSnapshot::new(vec![pb_control_collar::mutation::SnapshotEntry::new(
        logical_path.clone(),
        std::fs::read(workspace_root.join(target_path))
            .with_context(|| format!("failed to read Python qualification target {target_path}"))?,
    )])?
    .with_bound_mutation_path(logical_path);
    let tool = BuiltInToolSchema {
        name: "replace_file".to_string(),
        description: "Native Python world qualification mutation".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": { "path": { "const": target_path } }
        }),
    };

    let mut lifecycle = ControlLayerLifecycle::default();
    let cold_started = std::time::Instant::now();
    let cold_layers = lifecycle
        .prepare_for_inference_cancellable(
            workspace_root,
            std::slice::from_ref(&tool),
            Some(&snapshot),
            &|| Ok(()),
        )?
        .context("Python qualification fixture did not prepare a native layer")?;
    let cold_millis = elapsed_millis(cold_started);
    drop(cold_layers);

    let prepared = lifecycle
        .python
        .as_ref()
        .context("Python qualification lifecycle lost its prepared world")?;
    let prepared = prepared
        .lock()
        .map_err(|_| anyhow::anyhow!("Python qualification world lock is poisoned"))?;
    let receipt = prepared.world.readiness_receipt().clone();
    let world = prepared.world.descriptor().world.clone();
    drop(prepared);

    let invalid_started = std::time::Instant::now();
    let invalid = lifecycle.validate_completed_mutation(
        workspace_root,
        std::slice::from_ref(&tool),
        &snapshot,
        "replace_file",
        &serde_json::json!({ "path": target_path, "content": invalid_source }),
    )?;
    let invalid_replay_millis = elapsed_millis(invalid_started);
    if invalid != CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics) {
        bail!("Python qualification invalid replay was not rejected: {invalid:?}");
    }

    let valid_started = std::time::Instant::now();
    let valid = lifecycle.validate_completed_mutation(
        workspace_root,
        std::slice::from_ref(&tool),
        &snapshot,
        "replace_file",
        &serde_json::json!({ "path": target_path, "content": valid_source }),
    )?;
    let valid_replay_millis = elapsed_millis(valid_started);
    if valid != CompletionDecision::Accept {
        bail!("Python qualification valid replay was not accepted: {valid:?}");
    }

    let warm_started = std::time::Instant::now();
    let warm_layers = lifecycle
        .prepare_for_inference_cancellable(
            workspace_root,
            std::slice::from_ref(&tool),
            Some(&snapshot),
            &|| Ok(()),
        )?
        .context("Python qualification warm request did not prepare a native layer")?;
    let warm_millis = elapsed_millis(warm_started);
    drop(warm_layers);
    if (lifecycle.python_cold_builds, lifecycle.python_warm_requests) != (1, 2) {
        bail!(
            "Python qualification lifecycle did not record one cold build and two ready requests"
        );
    }

    let mut next_lifecycle = ControlLayerLifecycle::default();
    let cache_started = std::time::Instant::now();
    let cached_layers = next_lifecycle
        .prepare_for_inference_cancellable(
            workspace_root,
            std::slice::from_ref(&tool),
            Some(&snapshot),
            &|| Ok(()),
        )?
        .context("Python qualification process-cache request did not prepare a native layer")?;
    let process_cache_hit_millis = elapsed_millis(cache_started);
    drop(cached_layers);
    if (
        next_lifecycle.python_cold_builds,
        next_lifecycle.python_warm_requests,
        next_lifecycle.python_process_cache_hits,
    ) != (0, 1, 1)
    {
        bail!("Python qualification request did not use the exact process cache");
    }

    Ok(PythonWorldQualificationObservation {
        world_sha256: world.world_sha256,
        configuration_sha256: world.configuration_sha256,
        dependency_sha256: world.dependency_sha256,
        provider_version: world.provider_version,
        load_millis: receipt.load_millis,
        prime_millis: receipt.prime_millis,
        primed_queries: receipt.primed_queries,
        cold_millis,
        warm_millis,
        process_cache_hit_millis,
        invalid_replay_millis,
        valid_replay_millis,
    })
}

/// Qualify one complete Python mutation against the ordinary generation and execution gates, then
/// independently ask the language-owned layer for only its promoted diagnostic-code delta. The
/// caller supplies a frozen corpus workspace and never exposes the returned codes to inference.
pub(crate) fn qualify_python_semantic_case(
    lifecycle: &mut ControlLayerLifecycle,
    workspace_root: &Path,
    tool: &BuiltInToolSchema,
    snapshot: &WorkspaceSnapshot,
    name: &str,
    arguments: &Value,
) -> Result<PythonSemanticQualificationObservation> {
    let warm_started = std::time::Instant::now();
    let layers = lifecycle
        .prepare_for_inference_cancellable(
            workspace_root,
            std::slice::from_ref(tool),
            Some(snapshot),
            &|| Ok(()),
        )?
        .context("Python semantic qualification did not prepare a native layer")?;
    let warm_millis = elapsed_millis(warm_started);

    let gate = MutationCompletionGate::with_shared_language_layers(
        execution_manifest(std::slice::from_ref(tool), snapshot.clone()),
        layers,
    )
    .context("failed to create the Python semantic generation gate")?;
    let generation_started = std::time::Instant::now();
    let (closed_arguments, payload) = semantic_qualification_payload(name, arguments)?;
    let prefix = gate.evaluate_prefix(name, &closed_arguments, payload);
    let generation = match prefix {
        CompletionDecision::Reject(_) => prefix,
        CompletionDecision::Accept | CompletionDecision::NotApplicable => {
            gate.evaluate(name, arguments)
        }
    };
    let generation_millis = elapsed_millis(generation_started);

    let final_started = std::time::Instant::now();
    let final_replay = lifecycle.validate_completed_mutation(
        workspace_root,
        std::slice::from_ref(tool),
        snapshot,
        name,
        arguments,
    )?;
    let final_replay_millis = elapsed_millis(final_started);

    let diagnostic_started = std::time::Instant::now();
    let mutations = crate::semantic::semantic_mutations_from_call(snapshot, name, arguments)?;
    let candidates = mutations
        .into_iter()
        .map(|mutation| {
            Ok((
                pb_control_collar::mutation::LogicalPath::parse(mutation.path)?,
                mutation.result,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let prepared = lifecycle
        .python
        .as_ref()
        .context("Python semantic qualification lifecycle lost its prepared world")?;
    let prepared = prepared
        .lock()
        .map_err(|_| anyhow::anyhow!("Python semantic qualification world lock is poisoned"))?;
    let expected = &prepared.world.descriptor().world;
    let mut request = prepared
        .world
        .snapshot_for_request(expected)
        .context("Python semantic qualification could not snapshot the prepared world")?;
    let diagnostic_codes = request
        .qualification_introduced_codes(candidates)
        .context("Python semantic qualification diagnostic replay failed")?;
    let diagnostic_millis = elapsed_millis(diagnostic_started);

    Ok(PythonSemanticQualificationObservation {
        generation,
        final_replay,
        diagnostic_codes,
        warm_millis,
        generation_millis,
        final_replay_millis,
        diagnostic_millis,
    })
}

fn semantic_qualification_payload<'a>(
    name: &str,
    arguments: &'a Value,
) -> Result<(serde_json::Map<String, Value>, &'a str)> {
    let mut closed = arguments
        .as_object()
        .cloned()
        .context("Python semantic qualification arguments must be an object")?;
    let payload_name = match name {
        "write_file" | "replace_file" => "content",
        "edit_file" => "new_text",
        "apply_patch" => "patch",
        _ => bail!("Python semantic qualification does not cover tool {name}"),
    };
    let payload = arguments
        .get(payload_name)
        .and_then(Value::as_str)
        .with_context(|| {
            format!("Python semantic qualification tool {name} requires {payload_name}")
        })?;
    closed.remove(payload_name);
    Ok((closed, payload))
}

pub(crate) fn python_semantic_world_observation(
    lifecycle: &ControlLayerLifecycle,
) -> Result<PythonSemanticWorldObservation> {
    let prepared = lifecycle
        .python
        .as_ref()
        .context("Python semantic qualification lifecycle has no prepared world")?;
    let prepared = prepared
        .lock()
        .map_err(|_| anyhow::anyhow!("Python semantic qualification world lock is poisoned"))?;
    let world = &prepared.world.descriptor().world;
    let receipt = prepared.world.readiness_receipt();
    Ok(PythonSemanticWorldObservation {
        world_sha256: world.world_sha256.clone(),
        configuration_sha256: world.configuration_sha256.clone(),
        dependency_sha256: world.dependency_sha256.clone(),
        provider_version: world.provider_version.clone(),
        load_millis: receipt.load_millis,
        prime_millis: receipt.prime_millis,
        primed_queries: receipt.primed_queries,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PythonPthPolicy {
    search_paths: Vec<String>,
    has_import_hook: bool,
}

fn python_pth_policy(bytes: &[u8]) -> Result<PythonPthPolicy> {
    let contents = std::str::from_utf8(bytes).context("Python .pth file is not UTF-8")?;
    let mut policy = PythonPthPolicy::default();
    for line in contents.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.len() > PYTHON_DEPENDENCY_MAX_PTH_LINE_BYTES {
            bail!("Python .pth search line exceeds the bounded length");
        }
        if line.starts_with("import ") || line.starts_with("import\t") {
            policy.has_import_hook = true;
        } else {
            if line.contains('\0') {
                bail!("Python .pth search path contains a NUL byte");
            }
            policy.search_paths.push(line.to_string());
            if policy.search_paths.len() > PYTHON_DEPENDENCY_MAX_SEARCH_ROOTS {
                bail!("Python .pth file exceeds the bounded search-root count");
            }
        }
    }
    Ok(policy)
}

fn slash_relative_path(path: &Path) -> Result<String> {
    let path = path
        .to_str()
        .context("local Python dependency path is not UTF-8")?;
    Ok(path.replace('\\', "/"))
}

fn is_python_dependency_capture_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("py" | "pyi" | "pth")
    ) || path.file_name().and_then(|name| name.to_str()) == Some("py.typed")
        || is_python_distribution_metadata(path)
}

fn is_python_distribution_metadata(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let in_distribution_metadata = path.components().any(|component| {
        component.as_os_str().to_str().is_some_and(|component| {
            component.ends_with(".dist-info") || component.ends_with(".egg-info")
        })
    });
    in_distribution_metadata
        && matches!(
            name,
            "METADATA"
                | "WHEEL"
                | "INSTALLER"
                | "top_level.txt"
                | "namespace_packages.txt"
                | "direct_url.json"
        )
}

fn is_python_native_module(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("so" | "dylib" | "pyd" | "dll")
    )
}

fn is_unsupported_python_search_artifact(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("egg-link" | "egg" | "zip" | "pyc" | "pyo")
    ) || matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("sitecustomize.py" | "sitecustomize.pyi" | "usercustomize.py" | "usercustomize.pyi")
    )
}

fn is_python_project_marker(path: &str) -> bool {
    matches!(
        Path::new(path).file_name().and_then(|name| name.to_str()),
        Some(
            "pyproject.toml"
                | "ty.toml"
                | ".python-version"
                | "setup.py"
                | "setup.cfg"
                | "tox.ini"
                | "requirements.txt"
                | "requirements-dev.txt"
                | "uv.lock"
                | "poetry.lock"
                | "Pipfile"
                | "Pipfile.lock"
        )
    )
}

fn python_dependency_identity(
    snapshot: &crate::workspace::ContentSnapshot,
    dependencies: &PythonDependencySnapshot,
) -> String {
    let manifest = subset_identity(snapshot, is_python_dependency_input);
    let mut digest = Sha256::new();
    digest.update(b"pb-python-world-dependencies-v1\0");
    digest.update(manifest.as_bytes());
    digest.update(dependencies.fingerprint.as_bytes());
    digest.update(dependencies.python_version.as_bytes());
    digest.update([u8::from(dependencies.external_imports_complete)]);
    format!("{:x}", digest.finalize())
}

fn subset_identity(
    snapshot: &crate::workspace::ContentSnapshot,
    include: impl Fn(&str) -> bool,
) -> String {
    let mut digest = Sha256::new();
    for (path, content) in snapshot.paths.iter().filter(|(path, _)| include(path)) {
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path.as_bytes());
        digest.update(content.kind.as_bytes());
        digest.update(content.fingerprint.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn is_rust_configuration_input(path: &str) -> bool {
    matches!(
        path,
        "rust-toolchain" | "rust-toolchain.toml" | ".cargo/config" | ".cargo/config.toml"
    ) || path.ends_with("/.cargo/config")
        || path.ends_with("/.cargo/config.toml")
}

fn is_rust_dependency_input(path: &str) -> bool {
    path == "Cargo.toml"
        || path == "Cargo.lock"
        || path.ends_with("/Cargo.toml")
        || path.ends_with("/Cargo.lock")
}

fn is_python_configuration_input(path: &str) -> bool {
    matches!(
        path,
        "pyproject.toml"
            | "ty.toml"
            | ".python-version"
            | "setup.cfg"
            | "tox.ini"
            | "mypy.ini"
            | PYTHON_SEMANTIC_CONFIG_PATH
    ) || path.ends_with("/pyproject.toml")
        || path.ends_with("/ty.toml")
}

fn is_python_dependency_input(path: &str) -> bool {
    matches!(
        path,
        "pyproject.toml"
            | "requirements.txt"
            | "requirements-dev.txt"
            | "uv.lock"
            | "poetry.lock"
            | "Pipfile"
            | "Pipfile.lock"
            | PYTHON_SEMANTIC_CONFIG_PATH
    ) || path.ends_with("/requirements.txt")
        || path.ends_with("/uv.lock")
        || path.ends_with("/poetry.lock")
}

const fn host_python_platform() -> &'static str {
    if cfg!(target_os = "windows") {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(target_os = "android") {
        "android"
    } else if cfg!(target_os = "ios") {
        "ios"
    } else {
        "linux"
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        process::Command,
        sync::{
            Arc, Barrier,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use pb_control_collar::mutation::{LogicalPath, MutationKind, PatchStream, SnapshotEntry};
    use serde_json::json;

    use super::*;

    fn tool(name: &str, path: Value) -> BuiltInToolSchema {
        BuiltInToolSchema {
            name: name.to_string(),
            description: String::new(),
            input_schema: json!({
                "type": "object",
                "properties": { "path": path }
            }),
        }
    }

    #[test]
    fn exact_non_rust_path_does_not_trigger_expensive_rust_readiness() {
        let readme = LogicalPath::parse("README.md").unwrap();
        let rust = LogicalPath::parse("src/lib.rs").unwrap();
        assert!(!tools_may_mutate_rust(
            &[tool("write_file", json!({"const": "README.md"}))],
            None,
        ));
        assert!(tools_may_mutate_rust(
            &[tool("write_file", json!({"const": "src/lib.rs"}))],
            None,
        ));
        assert!(tools_may_mutate_rust(
            &[tool("write_file", json!({"type": "string"}))],
            None,
        ));
        assert!(!tools_may_mutate_rust(
            &[tool("replace_file", json!({"type": "string"}))],
            Some(&readme),
        ));
        assert!(tools_may_mutate_rust(
            &[tool("replace_file", json!({"type": "string"}))],
            Some(&rust),
        ));
        assert!(tools_may_mutate_rust(
            &[tool("apply_patch", json!({"const": "README.md"}))],
            Some(&readme),
        ));
        assert!(!tools_may_mutate_python(
            &[tool("write_file", json!({"const": "README.md"}))],
            None,
        ));
        assert!(tools_may_mutate_python(
            &[tool("write_file", json!({"const": "main.py"}))],
            None,
        ));
        assert!(tools_may_mutate_python(
            &[tool("apply_patch", json!({"const": "README.md"}))],
            Some(&readme),
        ));
    }

    #[test]
    fn python_dependency_capture_downgrades_native_hooks_and_undeclared_external_paths() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join(".gitignore"), ".venv/\n").unwrap();
        let site_packages = root.path().join(".venv/lib/python3.12/site-packages");
        fs::create_dir_all(&site_packages).unwrap();
        fs::write(root.path().join(".venv/pyvenv.cfg"), "version = 3.12.8\n").unwrap();
        fs::write(site_packages.join("native.abi3.so"), b"native").unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );
        let source = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        let native =
            capture_python_dependencies(root.path(), &source, &Default::default(), &|| Ok(()))
                .unwrap();
        assert!(native.site_packages_roots.is_empty());
        assert!(!native.external_imports_complete);
        assert!(native.files.is_empty());

        fs::remove_file(site_packages.join("native.abi3.so")).unwrap();
        fs::write(site_packages.join("editable.pth"), "import editable_hook\n").unwrap();
        let import_hook =
            capture_python_dependencies(root.path(), &source, &Default::default(), &|| Ok(()))
                .unwrap();
        assert!(import_hook.site_packages_roots.is_empty());
        assert!(!import_hook.external_imports_complete);
        assert!(import_hook.files.is_empty());

        let undeclared = site_packages.parent().unwrap().join("editable-source");
        fs::create_dir_all(&undeclared).unwrap();
        fs::write(undeclared.join("external_package.py"), "value: int = 1\n").unwrap();
        fs::write(site_packages.join("editable.pth"), "../editable-source\n").unwrap();
        let path_injected =
            capture_python_dependencies(root.path(), &source, &Default::default(), &|| Ok(()))
                .unwrap();
        assert!(path_injected.site_packages_roots.is_empty());
        assert!(!path_injected.external_imports_complete);
        assert!(path_injected.files.is_empty());
    }

    #[test]
    fn python_dependency_capture_downgrades_implicit_or_executable_search_state() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join(".gitignore"), ".venv/\n").unwrap();
        let site_packages = root.path().join(".venv/lib/python3.12/site-packages");
        fs::create_dir_all(&site_packages).unwrap();
        fs::write(
            root.path().join(".venv/pyvenv.cfg"),
            "version = 3.12.8\ninclude-system-site-packages = true\n",
        )
        .unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );
        let source = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        let capture = || {
            capture_python_dependencies(root.path(), &source, &Default::default(), &|| Ok(()))
                .unwrap()
        };
        assert!(capture().site_packages_roots.is_empty());

        fs::write(root.path().join(".venv/pyvenv.cfg"), "version = 3.12.8\n").unwrap();
        fs::write(
            site_packages.join("sitecustomize.py"),
            "import sys\nsys.path.append('/dynamic')\n",
        )
        .unwrap();
        assert!(capture().site_packages_roots.is_empty());

        fs::remove_file(site_packages.join("sitecustomize.py")).unwrap();
        fs::write(site_packages.join("sourceless.pyc"), b"bytecode").unwrap();
        assert!(capture().site_packages_roots.is_empty());
    }

    #[test]
    fn repository_python_policy_cannot_grant_external_or_ignored_authority() {
        let root = tempfile::tempdir().unwrap();
        let external = tempfile::tempdir().unwrap();
        PythonSemanticConfig {
            environment: Some(external.path().to_path_buf()),
            ..Default::default()
        }
        .save(root.path())
        .unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );
        let source = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        assert!(
            capture_python_dependencies(root.path(), &source, &Default::default(), &|| Ok(()),)
                .unwrap_err()
                .to_string()
                .contains("may select only an environment inside the workspace")
        );

        fs::write(root.path().join(".gitignore"), ".pb/python.toml\n").unwrap();
        let source = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        assert!(
            capture_python_dependencies(root.path(), &source, &Default::default(), &|| Ok(()),)
                .unwrap_err()
                .to_string()
                .contains("must be a controller-observed project file")
        );
    }

    #[test]
    fn repository_python_policy_can_disambiguate_an_in_workspace_environment() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join(".gitignore"), ".venv/\nvenv/\n").unwrap();
        for name in [".venv", "venv"] {
            let site_packages = root
                .path()
                .join(name)
                .join("lib/python3.12/site-packages/dependency");
            fs::create_dir_all(&site_packages).unwrap();
            fs::write(
                root.path().join(name).join("pyvenv.cfg"),
                "version = 3.12.8\n",
            )
            .unwrap();
            fs::write(
                site_packages.join("__init__.py"),
                "def parse(value: str) -> str:\n    return value\n",
            )
            .unwrap();
        }
        PythonSemanticConfig {
            environment: Some(PathBuf::from("venv")),
            ..Default::default()
        }
        .save(root.path())
        .unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );

        let source = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        let dependencies =
            capture_python_dependencies(root.path(), &source, &Default::default(), &|| Ok(()))
                .unwrap();
        assert_eq!(dependencies.site_packages_roots.len(), 1);
        assert!(dependencies.external_imports_complete);
        assert!(dependencies.files.iter().any(|file| {
            file.path.as_str()
                == ".pb-semantic-dependencies/python/site-packages/0/dependency/__init__.py"
        }));
    }

    #[test]
    fn project_local_plain_path_editable_is_a_frozen_first_party_root() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join(".gitignore"), ".venv/\n").unwrap();
        fs::write(
            root.path().join("pyproject.toml"),
            "[project]\nname = \"editable-project\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        fs::write(
            root.path().join("main.py"),
            "from dependency import parse\nresult: str = parse(\"ok\")\n",
        )
        .unwrap();
        fs::create_dir_all(root.path().join("src/dependency")).unwrap();
        fs::write(
            root.path().join("src/dependency/__init__.py"),
            "def parse(value: str) -> str:\n    return value\n",
        )
        .unwrap();
        fs::write(root.path().join("src/dependency/py.typed"), "").unwrap();
        let site_packages = root.path().join(".venv/lib/python3.12/site-packages");
        fs::create_dir_all(&site_packages).unwrap();
        fs::write(root.path().join(".venv/pyvenv.cfg"), "version = 3.12.8\n").unwrap();
        fs::write(
            site_packages.join("editable.pth"),
            format!("{}\n", root.path().join("src").display()),
        )
        .unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );

        let source = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        let dependencies =
            capture_python_dependencies(root.path(), &source, &Default::default(), &|| Ok(()))
                .unwrap();
        assert_eq!(
            dependencies.first_party_roots,
            vec![PathBuf::new(), PathBuf::from("src")]
        );
        assert_eq!(dependencies.site_packages_roots.len(), 1);
        assert!(dependencies.external_imports_complete);

        let python_tool = tool("replace_file", json!({"const": "main.py"}));
        let snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("main.py").unwrap(),
            fs::read(root.path().join("main.py")).unwrap(),
        )])
        .unwrap();
        let mut lifecycle = ControlLayerLifecycle::default();
        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    Some(&snapshot),
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    &snapshot,
                    "replace_file",
                    &json!({
                        "path": "main.py",
                        "content": "from dependency import parse\nresult: str = parse(1)\n"
                    }),
                )
                .unwrap(),
            CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics)
        );
    }

    #[test]
    fn project_editable_root_accounts_for_the_explicit_project_root_limit() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join(".gitignore"), ".venv/\n").unwrap();
        fs::create_dir_all(root.path().join("src/dependency")).unwrap();
        fs::write(
            root.path().join("src/dependency/__init__.py"),
            "value: int = 1\n",
        )
        .unwrap();
        let site_packages = root.path().join(".venv/lib/python3.12/site-packages");
        fs::create_dir_all(&site_packages).unwrap();
        fs::write(root.path().join(".venv/pyvenv.cfg"), "version = 3.12.8\n").unwrap();
        let mut search_paths = String::new();
        for index in 0..30 {
            let package = site_packages.join(format!("package-{index}"));
            fs::create_dir_all(&package).unwrap();
            fs::write(package.join("__init__.py"), "value: int = 1\n").unwrap();
            search_paths.push_str(&format!("package-{index}\n"));
        }
        search_paths.push_str(&format!("{}\n", root.path().join("src").display()));
        fs::write(site_packages.join("editable.pth"), search_paths).unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );

        let source = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        let dependencies =
            capture_python_dependencies(root.path(), &source, &Default::default(), &|| Ok(()))
                .unwrap();
        assert!(dependencies.first_party_roots.is_empty());
        assert!(dependencies.site_packages_roots.is_empty());
        assert!(!dependencies.external_imports_complete);
    }

    #[test]
    fn configured_external_plain_path_editable_is_frozen_and_revalidated() {
        let root = tempfile::tempdir().unwrap();
        let editable = tempfile::tempdir().unwrap();
        fs::write(root.path().join(".gitignore"), ".venv/\n").unwrap();
        fs::write(
            root.path().join("main.py"),
            "from dependency import parse\nresult: str = parse(\"ok\")\n",
        )
        .unwrap();
        fs::create_dir_all(editable.path().join("dependency")).unwrap();
        fs::write(
            editable.path().join("dependency/__init__.py"),
            "def parse(value: str) -> str:\n    return value\n",
        )
        .unwrap();
        fs::write(editable.path().join("dependency/py.typed"), "").unwrap();
        let site_packages = root.path().join(".venv/lib/python3.12/site-packages");
        fs::create_dir_all(&site_packages).unwrap();
        fs::write(root.path().join(".venv/pyvenv.cfg"), "version = 3.12.8\n").unwrap();
        fs::write(
            site_packages.join("editable.pth"),
            format!("{}\n", editable.path().display()),
        )
        .unwrap();
        let authority = crate::python_semantic_config::PythonExternalAuthority {
            environment: None,
            editable_roots: vec![editable.path().to_path_buf()],
        };
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );

        let source_before = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        let dependencies =
            capture_python_dependencies(root.path(), &source_before, &authority, &|| Ok(()))
                .unwrap();
        assert!(dependencies.first_party_roots.is_empty());
        assert_eq!(dependencies.site_packages_roots.len(), 2);
        assert!(dependencies.external_imports_complete);
        assert!(dependencies.files.iter().any(|file| {
            file.path.as_str()
                == ".pb-semantic-dependencies/python/site-packages/1/dependency/__init__.py"
        }));

        let python_tool = tool("replace_file", json!({"const": "main.py"}));
        let snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("main.py").unwrap(),
            fs::read(root.path().join("main.py")).unwrap(),
        )])
        .unwrap();
        let mut lifecycle =
            ControlLayerLifecycle::with_python_dependency_authority(authority.clone());
        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    Some(&snapshot),
                )
                .unwrap()
                .is_some()
        );
        let invalid = json!({
            "path": "main.py",
            "content": "from dependency import parse\nresult: str = parse(1)\n"
        });
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    &snapshot,
                    "replace_file",
                    &invalid,
                )
                .unwrap(),
            CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics)
        );

        fs::write(
            editable.path().join("dependency/__init__.py"),
            "def parse(value: int) -> int:\n    return value\n",
        )
        .unwrap();
        let source_after = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        assert_eq!(source_before.fingerprint, source_after.fingerprint);
        assert!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    &snapshot,
                    "replace_file",
                    &invalid,
                )
                .unwrap_err()
                .to_string()
                .contains("local Python dependencies changed")
        );
        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    Some(&snapshot),
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    &snapshot,
                    "replace_file",
                    &json!({
                        "path": "main.py",
                        "content": "from dependency import parse\nresult: int = parse(1)\n"
                    }),
                )
                .unwrap(),
            CompletionDecision::Accept
        );
    }

    #[test]
    fn configured_external_environment_resolves_dependency_shapes() {
        let root = tempfile::tempdir().unwrap();
        let environment_owner = tempfile::tempdir().unwrap();
        let environment = environment_owner.path().join("environment");
        let site_packages = environment.join("lib/python3.12/site-packages");
        fs::create_dir_all(site_packages.join("dependency")).unwrap();
        fs::write(environment.join("pyvenv.cfg"), "version = 3.12.8\n").unwrap();
        fs::write(
            site_packages.join("dependency/__init__.py"),
            "def parse(value: str) -> str:\n    return value\n",
        )
        .unwrap();
        fs::write(site_packages.join("dependency/py.typed"), "").unwrap();
        fs::write(
            root.path().join("main.py"),
            "from dependency import parse\nresult: str = parse(\"ok\")\n",
        )
        .unwrap();
        let authority = crate::python_semantic_config::PythonExternalAuthority {
            environment: Some(environment),
            editable_roots: Vec::new(),
        };
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );
        let source = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        let dependencies =
            capture_python_dependencies(root.path(), &source, &authority, &|| Ok(())).unwrap();
        assert_eq!(dependencies.site_packages_roots.len(), 1);
        assert!(dependencies.external_imports_complete);

        let python_tool = tool("replace_file", json!({"const": "main.py"}));
        let snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("main.py").unwrap(),
            fs::read(root.path().join("main.py")).unwrap(),
        )])
        .unwrap();
        let mut lifecycle = ControlLayerLifecycle::with_python_dependency_authority(authority);
        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    Some(&snapshot),
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    &[python_tool],
                    &snapshot,
                    "replace_file",
                    &json!({
                        "path": "main.py",
                        "content": "from dependency import parse\nresult: str = parse(1)\n"
                    }),
                )
                .unwrap(),
            CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics)
        );
    }

    #[test]
    fn python_world_is_prepared_before_inference_and_replayed_before_execution() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("main.py"), "value: int = 1\n").unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );
        let python_tool = tool("replace_file", json!({"const": "main.py"}));
        let snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("main.py").unwrap(),
            fs::read(root.path().join("main.py")).unwrap(),
        )])
        .unwrap();
        let mut lifecycle = ControlLayerLifecycle::default();
        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    Some(&snapshot),
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(lifecycle.stats(), (0, 0, 0));
        assert_eq!(lifecycle.python_stats(), (1, 1, 0));
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    &snapshot,
                    "replace_file",
                    &json!({"path": "main.py", "content": "value = \"text\" + 1\n"}),
                )
                .unwrap(),
            CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics)
        );
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    &snapshot,
                    "replace_file",
                    &json!({"path": "main.py", "content": "value = 2\n"}),
                )
                .unwrap(),
            CompletionDecision::Accept
        );
        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    Some(&snapshot),
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(lifecycle.python_stats(), (1, 2, 0));

        let mut next_stage = ControlLayerLifecycle::default();
        assert!(
            next_stage
                .prepare_for_inference(root.path(), &[python_tool], Some(&snapshot))
                .unwrap()
                .is_some()
        );
        let (cold, ready, cache_hits) = next_stage.python_stats();
        assert_eq!(ready, 1);
        assert!(
            (cold, cache_hits) == (0, 1) || (cold, cache_hits) == (1, 0),
            "a bounded process cache may be evicted by another concurrent project test"
        );
    }

    #[test]
    fn ignored_local_python_environment_is_frozen_before_inference_and_revalidated() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join(".gitignore"), ".venv/\n").unwrap();
        fs::write(
            root.path().join("main.py"),
            "from dependency import parse\nresult: str = parse(\"ok\")\n",
        )
        .unwrap();
        let site_packages = root.path().join(".venv/lib/python3.12/site-packages");
        fs::create_dir_all(site_packages.join("dependency")).unwrap();
        fs::create_dir_all(site_packages.join("dependency-1.0.dist-info")).unwrap();
        fs::write(root.path().join(".venv/pyvenv.cfg"), "version = 3.12.8\n").unwrap();
        fs::write(
            site_packages.join("dependency/__init__.py"),
            "def parse(value: str) -> str:\n    return value\n",
        )
        .unwrap();
        fs::write(site_packages.join("dependency/py.typed"), "").unwrap();
        fs::write(
            site_packages.join("dependency-1.0.dist-info/METADATA"),
            "Metadata-Version: 2.1\nName: dependency\nVersion: 1.0\n",
        )
        .unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );

        let python_tool = tool("replace_file", json!({"const": "main.py"}));
        let snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("main.py").unwrap(),
            fs::read(root.path().join("main.py")).unwrap(),
        )])
        .unwrap();
        let source_before = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        let mut lifecycle = ControlLayerLifecycle::default();
        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    Some(&snapshot),
                )
                .unwrap()
                .is_some()
        );
        assert!(
            lifecycle
                .python
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .world
                .readiness_receipt()
                .primed_queries
                >= 2
        );
        let invalid = json!({
            "path": "main.py",
            "content": "from dependency import parse\nresult: str = parse(1)\n"
        });
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    &snapshot,
                    "replace_file",
                    &invalid,
                )
                .unwrap(),
            CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics)
        );
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    &snapshot,
                    "replace_file",
                    &json!({
                        "path": "main.py",
                        "content": "import package_absent_from_the_qualified_environment\n"
                    }),
                )
                .unwrap(),
            CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics)
        );

        fs::write(
            site_packages.join("dependency/__init__.py"),
            "def parse(value: int) -> int:\n    return value\n",
        )
        .unwrap();
        let source_after = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        assert_eq!(source_before.fingerprint, source_after.fingerprint);
        assert!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    &snapshot,
                    "replace_file",
                    &invalid,
                )
                .unwrap_err()
                .to_string()
                .contains("local Python dependencies changed")
        );

        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    Some(&snapshot),
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(lifecycle.python_stats(), (2, 2, 0));
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    &snapshot,
                    "replace_file",
                    &json!({
                        "path": "main.py",
                        "content": "from dependency import parse\nresult: int = parse(1)\n"
                    }),
                )
                .unwrap(),
            CompletionDecision::Accept
        );
    }

    #[test]
    fn simultaneous_requests_share_one_exact_python_dependency_world() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("main.py"), "value: int = 1\n").unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );
        let python_tool = tool("replace_file", json!({"const": "main.py"}));
        let snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("main.py").unwrap(),
            fs::read(root.path().join("main.py")).unwrap(),
        )])
        .unwrap();
        let root = root.path().to_path_buf();
        let start = Arc::new(Barrier::new(3));
        let workers = (0..2)
            .map(|_| {
                let root = root.clone();
                let start = Arc::clone(&start);
                let python_tool = python_tool.clone();
                let snapshot = snapshot.clone();
                std::thread::spawn(move || {
                    let mut lifecycle = ControlLayerLifecycle::default();
                    start.wait();
                    let layers = lifecycle
                        .prepare_for_inference(&root, &[python_tool], Some(&snapshot))
                        .unwrap();
                    assert!(layers.is_some());
                    lifecycle.python_stats()
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        let stats = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(stats.iter().map(|stats| stats.0).sum::<u64>(), 1);
        assert_eq!(stats.iter().map(|stats| stats.1).sum::<u64>(), 2);
        assert_eq!(stats.iter().map(|stats| stats.2).sum::<u64>(), 1);
    }

    #[test]
    fn first_python_file_gets_native_semantics_before_inference() {
        let root = tempfile::tempdir().unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );
        let path = LogicalPath::parse("main.py").unwrap();
        let snapshot = WorkspaceSnapshot::new(Vec::new())
            .unwrap()
            .with_bound_mutation_path(path);
        let python_tool = tool("write_file", json!({"const": "main.py"}));
        let mut lifecycle = ControlLayerLifecycle::default();
        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    Some(&snapshot),
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(lifecycle.python_stats(), (1, 1, 0));
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&python_tool),
                    &snapshot,
                    "write_file",
                    &json!({"content": "value = \"text\" + 1\n"}),
                )
                .unwrap(),
            CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics)
        );
    }

    #[test]
    fn python_execution_replay_rejects_deletion_that_breaks_an_untouched_dependant() {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("helper.py"),
            "def add(value: int) -> int:\n    return value + 1\n",
        )
        .unwrap();
        fs::write(
            root.path().join("main.py"),
            "from helper import add\nvalue = add(1)\n",
        )
        .unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );
        let patch_tool = tool("apply_patch", json!({"type": "string"}));
        let snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("helper.py").unwrap(),
            fs::read(root.path().join("helper.py")).unwrap(),
        )])
        .unwrap();
        let mut lifecycle = ControlLayerLifecycle::default();
        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&patch_tool),
                    Some(&snapshot),
                )
                .unwrap()
                .is_some()
        );
        let patch = concat!(
            "diff --git a/helper.py b/helper.py\n",
            "deleted file mode 100644\n",
            "--- a/helper.py\n",
            "+++ /dev/null\n",
            "@@ -1,2 +0,0 @@\n",
            "-def add(value: int) -> int:\n",
            "-    return value + 1\n",
        );
        let mut stream = PatchStream::new(snapshot.clone(), patch.len(), 1, 1).unwrap();
        stream.push(patch.as_bytes()).unwrap();
        let (_, virtual_files) = stream.finish_with_virtual_files().unwrap();
        assert_eq!(virtual_files.len(), 1);
        assert_eq!(virtual_files[0].kind, MutationKind::Delete);

        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&patch_tool),
                    &snapshot,
                    "apply_patch",
                    &json!({"patch": patch}),
                )
                .unwrap(),
            CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics)
        );
    }

    #[test]
    fn stale_irrelevant_language_world_does_not_block_an_exact_other_language_mutation() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("main.py"), "value = 1\n").unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );
        let python_tool = tool("replace_file", json!({"const": "main.py"}));
        let python_snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("main.py").unwrap(),
            fs::read(root.path().join("main.py")).unwrap(),
        )])
        .unwrap();
        let mut lifecycle = ControlLayerLifecycle::default();
        lifecycle
            .prepare_for_inference(
                root.path(),
                std::slice::from_ref(&python_tool),
                Some(&python_snapshot),
            )
            .unwrap()
            .expect("the Python world should be prepared");
        fs::write(root.path().join("main.py"), "value = 2\n").unwrap();

        let javascript_path = LogicalPath::parse("new.js").unwrap();
        let javascript_snapshot = WorkspaceSnapshot::new(Vec::new())
            .unwrap()
            .with_bound_mutation_path(javascript_path);
        let javascript_tool = tool("write_file", json!({"const": "new.js"}));
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&javascript_tool),
                    &javascript_snapshot,
                    "write_file",
                    &json!({"content": "const value = 1;\n"}),
                )
                .unwrap(),
            CompletionDecision::NotApplicable
        );
    }

    #[test]
    fn workspace_drift_requires_a_fresh_controller_snapshot_instead_of_an_internal_retry() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("src")).unwrap();
        fs::write(
            root.path().join("Cargo.toml"),
            "[package]\nname = \"drift-fixture\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        fs::write(root.path().join("src/lib.rs"), "pub fn value() {}\n").unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );

        let expected = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        fs::write(
            root.path().join("src/lib.rs"),
            "pub fn changed_value() {}\n",
        )
        .unwrap();
        let after = crate::workspace::ContentSnapshot::capture(root.path()).unwrap();
        let error = ensure_unchanged_during_rust_preparation(&expected, &after)
            .unwrap_err()
            .to_string();
        assert!(error.contains("controller recaptures"), "{error}");
    }

    #[test]
    fn cancelled_waiter_does_not_wait_for_an_existing_world_preparation() {
        let key = RustWorldKey {
            workspace_root: PathBuf::from("/bounded/cancellation-fixture"),
            source_sha256: "a".repeat(64),
        };
        let active = begin_rust_world_preparation(key.clone(), &|| Ok(())).unwrap();
        let polls = AtomicUsize::new(0);
        let started = std::time::Instant::now();
        let error = begin_rust_world_preparation(key, &|| {
            if polls.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(())
            } else {
                bail!("cancelled while waiting for Rust preparation")
            }
        })
        .err()
        .expect("the waiter should observe cancellation");
        assert!(error.to_string().contains("cancelled while waiting"));
        assert!(started.elapsed() < Duration::from_secs(2));
        drop(active);
    }

    #[test]
    fn cancelled_python_waiter_does_not_wait_for_an_existing_dependency_world() {
        let key = PythonWorldKey {
            workspace_root: PathBuf::from("/bounded/python-cancellation-fixture"),
            source_sha256: "d".repeat(64),
            dependency_sha256: "e".repeat(64),
        };
        let active = begin_python_world_preparation(key.clone(), &|| Ok(())).unwrap();
        let polls = AtomicUsize::new(0);
        let started = std::time::Instant::now();
        let error = begin_python_world_preparation(key, &|| {
            if polls.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(())
            } else {
                bail!("cancelled while waiting for Python dependency preparation")
            }
        })
        .err()
        .expect("the Python waiter should observe cancellation");
        assert!(error.to_string().contains("cancelled while waiting"));
        assert!(started.elapsed() < Duration::from_secs(2));
        drop(active);
    }

    #[test]
    fn initiating_cold_world_wait_is_cancellable_while_worker_keeps_single_flight() {
        let key = RustWorldKey {
            workspace_root: PathBuf::from("/bounded/cold-owner-cancellation-fixture"),
            source_sha256: "b".repeat(64),
        };
        let preparation = match begin_rust_world_preparation(key.clone(), &|| Ok(())).unwrap() {
            RustWorldPreparation::Owner(preparation) => preparation,
            RustWorldPreparation::Shared(_) => {
                panic!("synthetic cold key unexpectedly had a world")
            }
        };
        let (started_sender, started_receiver) = sync_channel(1);
        let (release_sender, release_receiver) = sync_channel(1);
        let receiver = spawn_rust_preparation_worker(preparation, move || {
            started_sender.send(()).unwrap();
            release_receiver.recv().unwrap();
            Ok(())
        })
        .unwrap();
        started_receiver.recv().unwrap();

        let polls = AtomicUsize::new(0);
        let started = std::time::Instant::now();
        let error = wait_for_rust_preparation(&receiver, &|| {
            if polls.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(())
            } else {
                bail!("initiating request cancelled during cold Rust preparation")
            }
        })
        .unwrap_err();
        assert!(error.to_string().contains("initiating request cancelled"));
        assert!(started.elapsed() < Duration::from_secs(2));

        let coordinator = RUST_WORLD_PREPARATIONS.get().unwrap();
        assert!(coordinator.state.lock().unwrap().active.contains(&key));

        let other_key = RustWorldKey {
            workspace_root: PathBuf::from("/bounded/other-cold-world-fixture"),
            source_sha256: "c".repeat(64),
        };
        let capacity_polls = AtomicUsize::new(0);
        let capacity_result = begin_rust_world_preparation(other_key.clone(), &|| {
            if capacity_polls.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(())
            } else {
                bail!("cancelled while waiting for Rust preparation capacity")
            }
        });
        let error = match capacity_result {
            Ok(_) => panic!("the second cold world should wait for process capacity"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("preparation capacity"));

        release_sender.send(()).unwrap();
        receiver
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        let next = begin_rust_world_preparation(key, &|| Ok(())).unwrap();
        drop(next);
        let other = begin_rust_world_preparation(other_key, &|| Ok(())).unwrap();
        drop(other);
    }

    #[test]
    fn rust_world_is_warm_reused_and_incrementally_refreshed_after_source_drift() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("src")).unwrap();
        fs::write(
            root.path().join("Cargo.toml"),
            "[package]\nname = \"lifecycle-fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            root.path().join("src/lib.rs"),
            "pub fn value() -> i32 { 1 }\n",
        )
        .unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );

        let rust_tool = tool("replace_file", json!({"const": "src/lib.rs"}));
        let snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("src/lib.rs").unwrap(),
            fs::read(root.path().join("src/lib.rs")).unwrap(),
        )])
        .unwrap();
        let mut lifecycle = ControlLayerLifecycle::default();
        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&rust_tool),
                    Some(&snapshot)
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(lifecycle.stats(), (1, 1, 0));
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&rust_tool),
                    &snapshot,
                    "replace_file",
                    &json!({
                        "path": "src/lib.rs",
                        "content": "use std::collections::DefinitelyMissing;\npub fn value() -> i32 { 1 }\n"
                    }),
                )
                .unwrap(),
            CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics)
        );
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&rust_tool),
                    &snapshot,
                    "replace_file",
                    &json!({
                        "path": "src/lib.rs",
                        "content": "pub fn value() -> i32 { \"wrong\" }\n"
                    }),
                )
                .unwrap(),
            CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics)
        );
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root.path(),
                    std::slice::from_ref(&rust_tool),
                    &snapshot,
                    "replace_file",
                    &json!({
                        "path": "src/lib.rs",
                        "content": "use std::collections::BTreeMap;\npub fn value() -> i32 { BTreeMap::<i32, i32>::new().len() as i32 }\n"
                    }),
                )
                .unwrap(),
            CompletionDecision::Accept
        );

        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&rust_tool),
                    Some(&snapshot)
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(lifecycle.stats(), (1, 2, 0));

        fs::write(
            root.path().join("src/lib.rs"),
            "pub fn value() -> i32 { 2 }\n",
        )
        .unwrap();
        let changed = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("src/lib.rs").unwrap(),
            fs::read(root.path().join("src/lib.rs")).unwrap(),
        )])
        .unwrap();
        assert!(
            lifecycle
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&rust_tool),
                    Some(&changed)
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(lifecycle.stats(), (1, 3, 0));

        let mut next_stage = ControlLayerLifecycle::default();
        assert!(
            next_stage
                .prepare_for_inference(
                    root.path(),
                    std::slice::from_ref(&rust_tool),
                    Some(&changed)
                )
                .unwrap()
                .is_some()
        );
        assert_eq!(next_stage.stats(), (0, 1, 1));

        let active_layers = next_stage
            .prepare_for_inference(
                root.path(),
                std::slice::from_ref(&rust_tool),
                Some(&changed),
            )
            .unwrap()
            .expect("the exact warm Rust world should produce a request lease");
        assert_eq!(next_stage.stats(), (0, 2, 1));

        fs::write(
            root.path().join("src/lib.rs"),
            "pub fn value() -> i32 { 3 }\n",
        )
        .unwrap();
        let changed_while_active = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("src/lib.rs").unwrap(),
            fs::read(root.path().join("src/lib.rs")).unwrap(),
        )])
        .unwrap();
        let replacement_layers = next_stage
            .prepare_for_inference(root.path(), &[rust_tool], Some(&changed_while_active))
            .unwrap();
        assert!(replacement_layers.is_some());
        assert_eq!(next_stage.stats(), (1, 3, 1));

        // Both immutable request revisions can coexist. Dropping either request must not affect
        // the other, and the lifecycle never revises the Salsa database leased by active_layers.
        drop(replacement_layers);
        drop(active_layers);
    }

    #[test]
    fn simultaneous_requests_share_one_exact_rust_world_preparation() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("src")).unwrap();
        fs::write(
            root.path().join("Cargo.toml"),
            "[package]\nname = \"single-flight-fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            root.path().join("src/lib.rs"),
            "pub fn value() -> i32 { 1 }\n",
        )
        .unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .arg("--quiet")
                .current_dir(root.path())
                .status()
                .unwrap()
                .success()
        );

        let rust_tool = tool("replace_file", json!({"const": "src/lib.rs"}));
        let snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("src/lib.rs").unwrap(),
            fs::read(root.path().join("src/lib.rs")).unwrap(),
        )])
        .unwrap();
        let root = root.path().to_path_buf();
        let start = Arc::new(Barrier::new(3));
        let workers = (0..2)
            .map(|_| {
                let root = root.clone();
                let start = Arc::clone(&start);
                let rust_tool = rust_tool.clone();
                let snapshot = snapshot.clone();
                std::thread::spawn(move || {
                    let mut lifecycle = ControlLayerLifecycle::default();
                    start.wait();
                    let layers = lifecycle
                        .prepare_for_inference(&root, &[rust_tool], Some(&snapshot))
                        .unwrap();
                    assert!(layers.is_some());
                    lifecycle.stats()
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        let stats = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(stats.iter().map(|stats| stats.0).sum::<u64>(), 1);
        assert_eq!(stats.iter().map(|stats| stats.1).sum::<u64>(), 2);
        assert_eq!(stats.iter().map(|stats| stats.2).sum::<u64>(), 1);
    }

    #[test]
    #[ignore = "qualification probe loads this complete Cargo workspace through rust-analyzer"]
    fn rust_world_qualifies_current_workspace() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let rust_tool = tool("replace_file", json!({"const": "src/lib.rs"}));
        let snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            LogicalPath::parse("src/lib.rs").unwrap(),
            fs::read(root.join("src/lib.rs")).unwrap(),
        )])
        .unwrap();

        let mut lifecycle = ControlLayerLifecycle::default();
        let cold_started = std::time::Instant::now();
        let cold_layers = lifecycle
            .prepare_for_inference(root, std::slice::from_ref(&rust_tool), Some(&snapshot))
            .unwrap();
        let cold_elapsed = cold_started.elapsed();
        assert!(cold_layers.is_some());
        drop(cold_layers);

        let replay_started = std::time::Instant::now();
        assert_eq!(
            lifecycle
                .validate_completed_mutation(
                    root,
                    std::slice::from_ref(&rust_tool),
                    &snapshot,
                    "replace_file",
                    &json!({
                        "path": "src/lib.rs",
                        "content": "use std::collections::DefinitelyMissing;\n"
                    }),
                )
                .unwrap(),
            CompletionDecision::Reject(pb_control_collar::RejectionCode::InvalidSemantics)
        );
        let replay_elapsed = replay_started.elapsed();

        let warm_started = std::time::Instant::now();
        let warm_layers = lifecycle
            .prepare_for_inference(root, &[rust_tool], Some(&snapshot))
            .unwrap();
        let warm_elapsed = warm_started.elapsed();
        assert!(warm_layers.is_some());
        assert_eq!(lifecycle.stats(), (1, 2, 0));

        eprintln!(
            "rust-world qualification: cold_ms={} replay_ms={} warm_ms={}",
            cold_elapsed.as_millis(),
            replay_elapsed.as_millis(),
            warm_elapsed.as_millis()
        );
    }
}
