//! Pre-inference lifecycle for native streaming language layers.
//!
//! Expensive project worlds are built from verified immutable shadows before any model invocation
//! that can emit a matching-language mutation. Decoding receives only a cheap request snapshot;
//! it never loads Cargo metadata, starts a language server, or reads the live workspace.

use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use anyhow::{Context, Result, bail};
use pb_control_collar::{
    CompletionDecision, MutationCompletionGate,
    analysis::{LanguageLayerStack, ProgramFile, ProgramSnapshot, SyntaxProfile},
    mutation::WorkspaceSnapshot,
    protocol::ToolDialect,
    tool::{CollarLimits, CollarManifest, ExposedTool, MutationPolicy, ToolConstraintMode},
};
use pb_control_rust::{RUST_LAYER_CONTRACT_VERSION, RustProjectConfig, RustProjectWorld};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{agent_core::BuiltInToolSchema, semantic::SemanticShadowWorkspace};

const RUST_LAYER_MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;
const RUST_LAYER_MAX_CHECKPOINTS: usize = 4_096;
const MAX_WORLD_REBUILD_ATTEMPTS: usize = 2;
const MAX_PROCESS_RUST_WORLDS: usize = 2;

pub(crate) type SharedLanguageLayers = Arc<Mutex<LanguageLayerStack>>;

pub(crate) struct ControlLayerLifecycle {
    rust: Option<PreparedRustHandle>,
    cold_builds: u64,
    warm_requests: u64,
    process_cache_hits: u64,
}

struct PreparedRustWorld {
    workspace_root: PathBuf,
    source_sha256: String,
    content: crate::workspace::ContentSnapshot,
    _shadow: Arc<SemanticShadowWorkspace>,
    world: RustProjectWorld,
}

impl Default for ControlLayerLifecycle {
    fn default() -> Self {
        Self {
            rust: None,
            cold_builds: 0,
            warm_requests: 0,
            process_cache_hits: 0,
        }
    }
}

type PreparedRustHandle = Arc<Mutex<PreparedRustWorld>>;

static RUST_WORLD_CACHE: OnceLock<Mutex<VecDeque<PreparedRustHandle>>> = OnceLock::new();

impl ControlLayerLifecycle {
    /// Establish all expensive state before the caller reserves or records a model invocation.
    /// Returning successfully means the supplied stack is bound to the exact live-workspace
    /// identity observed after loading and priming completed.
    pub(crate) fn prepare_for_inference(
        &mut self,
        workspace_root: &Path,
        tools: &[BuiltInToolSchema],
        mutation_snapshot: Option<&WorkspaceSnapshot>,
    ) -> Result<Option<SharedLanguageLayers>> {
        if !tools_may_mutate_rust(
            tools,
            mutation_snapshot.and_then(WorkspaceSnapshot::bound_mutation_path),
        ) {
            return Ok(None);
        }
        let snapshot = mutation_snapshot.context(
            "Rust-edit-capable inference requires a controller-authorized mutation snapshot",
        )?;

        let live = crate::workspace::ContentSnapshot::capture(workspace_root)
            .context("failed to identify the Rust semantic world before inference")?;
        if !live.paths.contains_key("Cargo.toml")
            && !live.paths.keys().any(|path| path.ends_with("/Cargo.toml"))
        {
            // A standalone .rs file still receives the collar's lexical/syntax layer. There is no
            // Cargo project whose dependencies can be loaded and resolved.
            return Ok(None);
        }

        let canonical_root = workspace_root
            .canonicalize()
            .context("failed to canonicalize the Rust project root")?;

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
            if let Some(cached) = process_cached_world(&canonical_root, &live.fingerprint)? {
                self.rust = Some(cached);
                self.process_cache_hits = self.process_cache_hits.saturating_add(1);
            } else if let Some(refreshed) = self.try_incremental_refresh(&canonical_root, &live)? {
                insert_process_world(Arc::clone(&refreshed))?;
                self.rust = Some(refreshed);
            } else {
                let prepared =
                    Arc::new(Mutex::new(self.build_current_world(&canonical_root, live)?));
                self.cold_builds = self.cold_builds.saturating_add(1);
                insert_process_world(Arc::clone(&prepared))?;
                self.rust = Some(prepared);
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
        let program = program_snapshot(snapshot)?;
        let stack = LanguageLayerStack::new(vec![Box::new(layer)], program)
            .context("failed to start the request-local language-layer stack")?;
        self.warm_requests = self.warm_requests.saturating_add(1);
        Ok(Some(Arc::new(Mutex::new(stack))))
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
        let Some(prepared) = self.rust.as_ref() else {
            return Ok(CompletionDecision::NotApplicable);
        };
        if !matches!(
            name,
            "write_file" | "replace_file" | "edit_file" | "apply_patch"
        ) {
            return Ok(CompletionDecision::NotApplicable);
        }
        let canonical_root = workspace_root
            .canonicalize()
            .context("failed to canonicalize the Rust project before mutation execution")?;
        let live = crate::workspace::ContentSnapshot::capture(workspace_root)
            .context("failed to revalidate the Rust semantic world before mutation execution")?;
        let prepared = prepared
            .lock()
            .map_err(|_| anyhow::anyhow!("Rust semantic world lock is poisoned"))?;
        if prepared.workspace_root != canonical_root || prepared.source_sha256 != live.fingerprint {
            bail!(
                "workspace changed after Rust semantic preparation; refusing to execute a mutation against a stale world"
            );
        }
        let expected = &prepared.world.descriptor().world;
        let request = prepared
            .world
            .snapshot_for_request(expected)
            .context("prepared Rust semantic world was not ready for execution replay")?;
        let layer = request
            .into_streaming_layer(RUST_LAYER_MAX_SOURCE_BYTES, RUST_LAYER_MAX_CHECKPOINTS)
            .context("failed to create the execution-time Rust replay layer")?;
        let stack = LanguageLayerStack::new(vec![Box::new(layer)], program_snapshot(snapshot)?)
            .context("failed to start the execution-time language-layer stack")?;
        let manifest = execution_manifest(tools, snapshot.clone());
        let gate = MutationCompletionGate::with_language_layers(manifest, stack)
            .context("failed to create the independent mutation replay gate")?;
        Ok(gate.evaluate_independent(name, arguments))
    }

    fn build_current_world(
        &self,
        workspace_root: &Path,
        mut expected: crate::workspace::ContentSnapshot,
    ) -> Result<PreparedRustWorld> {
        for attempt in 1..=MAX_WORLD_REBUILD_ATTEMPTS {
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
            if after.fingerprint == expected.fingerprint {
                tracing::info!(
                    world_sha256 = %expected.fingerprint,
                    load_millis = world.readiness_receipt().load_millis,
                    prime_millis = world.readiness_receipt().prime_millis,
                    targets = world.targets().len(),
                    "Rust streaming semantic world is ready before inference"
                );
                return Ok(PreparedRustWorld {
                    workspace_root: workspace_root.to_path_buf(),
                    source_sha256: expected.fingerprint.clone(),
                    content: expected,
                    _shadow: Arc::new(shadow),
                    world,
                });
            }
            if attempt == MAX_WORLD_REBUILD_ATTEMPTS {
                bail!(
                    "workspace changed while rust-analyzer was loading; refusing Rust-edit-capable inference without an exact ready world"
                );
            }
            expected = after;
        }
        unreachable!("bounded Rust world rebuild loop always returns or fails")
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

#[cfg(test)]
mod tests {
    use std::{fs, process::Command};

    use pb_control_collar::mutation::{LogicalPath, SnapshotEntry};
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
