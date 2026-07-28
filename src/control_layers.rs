//! Pre-inference lifecycle for native streaming language layers.
//!
//! Expensive project worlds are built from verified immutable shadows before any model invocation
//! that can emit a matching-language mutation. Decoding receives only a cheap request snapshot;
//! it never loads Cargo metadata, starts a language server, or reads the live workspace.

use std::{
    collections::{HashSet, VecDeque},
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

use crate::{agent_core::BuiltInToolSchema, semantic::SemanticShadowWorkspace};

const RUST_LAYER_MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;
const RUST_LAYER_MAX_CHECKPOINTS: usize = 4_096;
const PYTHON_LAYER_MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;
const PYTHON_LAYER_MAX_CHECKPOINTS: usize = 4_096;
const PYTHON_LAYER_MAX_PROJECT_FILES: usize = 100_000;
const PYTHON_LAYER_MAX_PROJECT_BYTES: usize = 256 * 1024 * 1024;
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
    _shadow: Arc<SemanticShadowWorkspace>,
    world: PythonProjectWorld,
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
}

#[derive(Default)]
struct RustWorldPreparationCoordinator {
    active: Mutex<HashSet<RustWorldKey>>,
    ready: Condvar,
}

struct RustWorldPreparationGuard {
    coordinator: &'static RustWorldPreparationCoordinator,
    key: RustWorldKey,
}

static RUST_WORLD_PREPARATIONS: OnceLock<RustWorldPreparationCoordinator> = OnceLock::new();

#[derive(Default)]
struct PythonWorldPreparationCoordinator {
    active: Mutex<HashSet<PythonWorldKey>>,
    ready: Condvar,
}

struct PythonWorldPreparationGuard {
    coordinator: &'static PythonWorldPreparationCoordinator,
    key: PythonWorldKey,
}

static PYTHON_WORLD_PREPARATIONS: OnceLock<PythonWorldPreparationCoordinator> = OnceLock::new();

impl ControlLayerLifecycle {
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
                if let Some(cached) = process_cached_world(canonical_root, &live.fingerprint)? {
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
        let current_matches = self
            .python
            .as_ref()
            .map(|prepared| {
                prepared
                    .lock()
                    .map(|prepared| {
                        prepared.workspace_root == canonical_root
                            && prepared.source_sha256 == live.fingerprint
                    })
                    .map_err(|_| anyhow::anyhow!("Python semantic world lock is poisoned"))
            })
            .transpose()?
            == Some(true);
        if !current_matches {
            if let Some(cached) = process_cached_python_world(canonical_root, &live.fingerprint)? {
                self.python = Some(cached);
                self.python_process_cache_hits = self.python_process_cache_hits.saturating_add(1);
            } else {
                let preparation = begin_python_world_preparation(
                    PythonWorldKey {
                        workspace_root: canonical_root.to_path_buf(),
                        source_sha256: live.fingerprint.clone(),
                    },
                    cancellation,
                )?;
                if let Some(cached) =
                    process_cached_python_world(canonical_root, &live.fingerprint)?
                {
                    self.python = Some(cached);
                    self.python_process_cache_hits =
                        self.python_process_cache_hits.saturating_add(1);
                } else {
                    let receiver = spawn_python_world_build(
                        preparation,
                        canonical_root.to_path_buf(),
                        live.clone(),
                    )?;
                    self.python_cold_builds = self.python_cold_builds.saturating_add(1);
                    self.python = Some(wait_for_python_preparation(&receiver, cancellation)?);
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
            {
                bail!(
                    "workspace changed after Python semantic preparation; refusing to execute a mutation against a stale world"
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
    ) -> Result<PreparedPythonWorld> {
        let shadow = SemanticShadowWorkspace::capture(workspace_root, &expected)
            .context("failed to capture an immutable Python semantic shadow")?;
        let config = PythonProjectConfig {
            contract_version: PYTHON_LAYER_CONTRACT_VERSION,
            shadow_root: shadow.path().to_path_buf(),
            first_party_roots: Vec::new(),
            // Dependency snapshots are deliberately explicit. The shipped default resolves the
            // project plus bundled typeshed; configured virtual-environment capture is a later
            // lifecycle phase and must become part of dependency_sha256 before being enabled.
            site_packages_roots: Vec::new(),
            python_version: "3.12".to_string(),
            python_platform: host_python_platform().to_string(),
            world_sha256: expected.fingerprint.clone(),
            configuration_sha256: subset_identity(&expected, is_python_configuration_input),
            dependency_sha256: subset_identity(&expected, is_python_dependency_input),
            max_files: PYTHON_LAYER_MAX_PROJECT_FILES,
            max_bytes: PYTHON_LAYER_MAX_PROJECT_BYTES,
        };
        let world = PythonProjectWorld::load_and_prime(config)
            .context("failed to load and prime Astral ty before inference")?;
        let after = crate::workspace::ContentSnapshot::capture(workspace_root)
            .context("failed to revalidate the Python semantic world after priming")?;
        ensure_unchanged_during_python_preparation(&expected, &after)?;
        tracing::info!(
            world_sha256 = %expected.fingerprint,
            load_millis = world.readiness_receipt().load_millis,
            prime_millis = world.readiness_receipt().prime_millis,
            primed_queries = world.readiness_receipt().primed_queries,
            "Python streaming semantic world is ready before inference"
        );
        Ok(PreparedPythonWorld {
            workspace_root: workspace_root.to_path_buf(),
            source_sha256: expected.fingerprint,
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
        let mut active = self
            .coordinator
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        active.remove(&self.key);
        self.coordinator.ready.notify_all();
    }
}

impl Drop for PythonWorldPreparationGuard {
    fn drop(&mut self) {
        let mut active = self
            .coordinator
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        active.remove(&self.key);
        self.coordinator.ready.notify_all();
    }
}

fn begin_rust_world_preparation(
    key: RustWorldKey,
    cancellation: &dyn Fn() -> Result<()>,
) -> Result<RustWorldPreparationGuard> {
    cancellation()?;
    let coordinator = RUST_WORLD_PREPARATIONS.get_or_init(Default::default);
    let mut active = coordinator
        .active
        .lock()
        .map_err(|_| anyhow::anyhow!("Rust semantic preparation coordinator is poisoned"))?;
    while active.contains(&key) || active.len() >= MAX_CONCURRENT_RUST_WORLD_PREPARATIONS {
        let (next, _) = coordinator
            .ready
            .wait_timeout(active, PREPARATION_WAIT_POLL)
            .map_err(|_| anyhow::anyhow!("Rust semantic preparation coordinator is poisoned"))?;
        active = next;
        cancellation()?;
    }
    active.insert(key.clone());
    Ok(RustWorldPreparationGuard { coordinator, key })
}

fn begin_python_world_preparation(
    key: PythonWorldKey,
    cancellation: &dyn Fn() -> Result<()>,
) -> Result<PythonWorldPreparationGuard> {
    cancellation()?;
    let coordinator = PYTHON_WORLD_PREPARATIONS.get_or_init(Default::default);
    let mut active = coordinator
        .active
        .lock()
        .map_err(|_| anyhow::anyhow!("Python semantic preparation coordinator is poisoned"))?;
    while active.contains(&key) || active.len() >= MAX_CONCURRENT_PYTHON_WORLD_PREPARATIONS {
        let (next, _) = coordinator
            .ready
            .wait_timeout(active, PREPARATION_WAIT_POLL)
            .map_err(|_| anyhow::anyhow!("Python semantic preparation coordinator is poisoned"))?;
        active = next;
        cancellation()?;
    }
    active.insert(key.clone());
    Ok(PythonWorldPreparationGuard { coordinator, key })
}

fn spawn_rust_world_build(
    preparation: RustWorldPreparationGuard,
    workspace_root: PathBuf,
    expected: crate::workspace::ContentSnapshot,
) -> Result<Receiver<Result<PreparedRustHandle>>> {
    spawn_rust_preparation_worker(preparation, move || {
        let prepared = Arc::new(Mutex::new(ControlLayerLifecycle::build_current_world(
            &workspace_root,
            expected,
        )?));
        insert_process_world(Arc::clone(&prepared))?;
        Ok(prepared)
    })
}

fn spawn_python_world_build(
    preparation: PythonWorldPreparationGuard,
    workspace_root: PathBuf,
    expected: crate::workspace::ContentSnapshot,
) -> Result<Receiver<Result<PreparedPythonHandle>>> {
    spawn_python_preparation_worker(preparation, move || {
        let prepared = Arc::new(Mutex::new(
            ControlLayerLifecycle::build_current_python_world(&workspace_root, expected)?,
        ));
        insert_process_python_world(Arc::clone(&prepared))?;
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
        (world.workspace_root.clone(), world.source_sha256.clone())
    };
    let mut retained = VecDeque::with_capacity(cache.len());
    while let Some(entry) = cache.pop_front() {
        let duplicate = {
            let entry = entry
                .lock()
                .map_err(|_| anyhow::anyhow!("Python semantic world lock is poisoned"))?;
            entry.workspace_root == identity.0 && entry.source_sha256 == identity.1
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
        "pyproject.toml" | "ty.toml" | ".python-version" | "setup.cfg" | "tox.ini" | "mypy.ini"
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
        assert_eq!(next_stage.python_stats(), (0, 1, 1));
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
    fn python_execution_replay_resolves_deletions_across_one_patch_transaction() {
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
        let snapshot = WorkspaceSnapshot::new(vec![
            SnapshotEntry::new(
                LogicalPath::parse("helper.py").unwrap(),
                fs::read(root.path().join("helper.py")).unwrap(),
            ),
            SnapshotEntry::new(
                LogicalPath::parse("main.py").unwrap(),
                fs::read(root.path().join("main.py")).unwrap(),
            ),
        ])
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
            "diff --git a/main.py b/main.py\n",
            "--- a/main.py\n",
            "+++ b/main.py\n",
            "@@ -1,2 +1,2 @@\n",
            " from helper import add\n",
            "-value = add(1)\n",
            "+value = add(2)\n",
        );
        let mut stream = PatchStream::new(snapshot.clone(), patch.len(), 2, 2).unwrap();
        stream.push(patch.as_bytes()).unwrap();
        let (_, virtual_files) = stream.finish_with_virtual_files().unwrap();
        assert_eq!(virtual_files.len(), 2);
        assert_eq!(virtual_files[0].kind, MutationKind::Delete);
        assert_eq!(virtual_files[1].kind, MutationKind::Modify);
        assert_eq!(
            virtual_files[1]
                .segments
                .iter()
                .flat_map(|segment| &segment.bytes)
                .copied()
                .collect::<Vec<_>>(),
            b"from helper import add\nvalue = add(2)\n"
        );

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
    fn initiating_cold_world_wait_is_cancellable_while_worker_keeps_single_flight() {
        let key = RustWorldKey {
            workspace_root: PathBuf::from("/bounded/cold-owner-cancellation-fixture"),
            source_sha256: "b".repeat(64),
        };
        let preparation = begin_rust_world_preparation(key.clone(), &|| Ok(())).unwrap();
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
        assert!(coordinator.active.lock().unwrap().contains(&key));

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
