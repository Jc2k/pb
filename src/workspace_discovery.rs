use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use walkdir::{DirEntry, WalkDir};

use crate::environment::EnvironmentConfig;
use crate::workspace::{
    CargoWorkspace, CheckTrigger, Executor, ExecutorKind, WORKSPACE_CONFIG_VERSION, WorkspaceCheck,
    WorkspaceComponent, WorkspaceGraph, WorkspaceGraphSource,
};

const EXCLUDED_DIRECTORIES: &[&str] = &[
    ".git",
    ".pb",
    "target",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "vendor",
];

pub fn discover_workspace(
    repo_root: &Path,
    environment: Option<&EnvironmentConfig>,
) -> Result<WorkspaceGraph> {
    let repo_root = repo_root
        .canonicalize()
        .with_context(|| format!("failed to resolve repository root {}", repo_root.display()))?;
    let files = discover_manifests(&repo_root)?;
    let mut graph = WorkspaceGraph {
        version: WORKSPACE_CONFIG_VERSION,
        executors: BTreeMap::new(),
        components: BTreeMap::new(),
        checks: BTreeMap::new(),
        tasks: BTreeMap::new(),
        cargo_workspaces: BTreeMap::new(),
        discovery_warnings: Vec::new(),
        source: WorkspaceGraphSource::Discovered,
    };

    discover_cargo(&repo_root, &files.cargo, &mut graph);
    discover_javascript(&repo_root, &files, &mut graph);
    discover_go(&repo_root, &files.go, &mut graph)?;
    discover_python(&repo_root, &files.python, &mut graph)?;

    if graph.components.is_empty() {
        let executor = Executor {
            id: "project".to_string(),
            kind: ExecutorKind::Project,
            environment: None,
        };
        let component = WorkspaceComponent {
            id: "repository".to_string(),
            root: ".".to_string(),
            include: vec!["**".to_string()],
            exclude: Vec::new(),
            executor: executor.id.clone(),
            depends_on: Vec::new(),
        };
        graph.executors.insert(executor.id.clone(), executor);
        graph.components.insert(component.id.clone(), component);
    }

    let canonical_commands = canonical_commands(&repo_root, environment);
    apply_canonical_commands(&repo_root, &canonical_commands, &mut graph);

    let topology = graph.cargo_workspaces.clone();
    let warnings = graph.discovery_warnings.clone();
    let mut validated = graph.to_document().normalize()?;
    validated.cargo_workspaces = topology;
    validated.discovery_warnings = warnings;
    validated.source = WorkspaceGraphSource::Discovered;
    Ok(validated)
}

#[derive(Default)]
struct ManifestFiles {
    cargo: Vec<PathBuf>,
    package: Vec<PathBuf>,
    deno: Vec<PathBuf>,
    go: Vec<PathBuf>,
    python: Vec<PathBuf>,
}

fn discover_manifests(repo_root: &Path) -> Result<ManifestFiles> {
    let mut files = ManifestFiles::default();
    for entry in WalkDir::new(repo_root)
        .follow_links(false)
        .into_iter()
        .filter_entry(included_entry)
    {
        let entry = entry.with_context(|| {
            format!("failed to inspect workspace below {}", repo_root.display())
        })?;
        if !entry.file_type().is_file() {
            continue;
        }
        match entry.file_name().to_str() {
            Some("Cargo.toml") => files.cargo.push(entry.into_path()),
            Some("package.json") => files.package.push(entry.into_path()),
            Some("deno.json" | "deno.jsonc") => files.deno.push(entry.into_path()),
            Some("go.mod") => files.go.push(entry.into_path()),
            Some("pyproject.toml" | "requirements.txt") => files.python.push(entry.into_path()),
            _ => {}
        }
    }
    files.cargo.sort();
    files.package.sort();
    files.deno.sort();
    files.go.sort();
    files.python.sort();
    Ok(files)
}

fn included_entry(entry: &DirEntry) -> bool {
    entry.depth() == 0
        || !entry.file_type().is_dir()
        || !EXCLUDED_DIRECTORIES
            .iter()
            .any(|excluded| entry.file_name() == *excluded)
}

#[derive(Debug, Deserialize)]
struct CargoMetadata {
    packages: Vec<CargoPackageMetadata>,
    workspace_members: Vec<String>,
    workspace_default_members: Vec<String>,
    workspace_root: String,
}

#[derive(Debug, Deserialize)]
struct CargoPackageMetadata {
    id: String,
    name: String,
    manifest_path: String,
    #[serde(default)]
    dependencies: Vec<CargoDependencyMetadata>,
}

#[derive(Debug, Deserialize)]
struct CargoDependencyMetadata {
    name: String,
    path: Option<String>,
}

fn discover_cargo(repo_root: &Path, manifests: &[PathBuf], graph: &mut WorkspaceGraph) {
    let mut seen_workspaces = HashSet::new();
    let mut seen_member_manifests = HashSet::new();
    for manifest in manifests {
        let canonical_manifest = manifest.canonicalize().unwrap_or_else(|_| manifest.clone());
        if seen_member_manifests.contains(&canonical_manifest) {
            continue;
        }
        let output = Command::new("cargo")
            .args([
                "metadata",
                "--no-deps",
                "--format-version",
                "1",
                "--manifest-path",
            ])
            .arg(manifest)
            .current_dir(repo_root)
            .output();
        let output = match output {
            Ok(output) if output.status.success() => output,
            Ok(output) => {
                graph.discovery_warnings.push(format!(
                    "cargo metadata failed for {}: {}",
                    display_relative(repo_root, manifest),
                    String::from_utf8_lossy(&output.stderr).trim()
                ));
                add_fallback_cargo_package(repo_root, manifest, graph);
                continue;
            }
            Err(error) => {
                graph.discovery_warnings.push(format!(
                    "cargo metadata could not start for {}: {error}",
                    display_relative(repo_root, manifest)
                ));
                add_fallback_cargo_package(repo_root, manifest, graph);
                continue;
            }
        };
        let metadata: CargoMetadata = match serde_json::from_slice(&output.stdout) {
            Ok(metadata) => metadata,
            Err(error) => {
                graph.discovery_warnings.push(format!(
                    "cargo metadata returned invalid JSON for {}: {error}",
                    display_relative(repo_root, manifest)
                ));
                continue;
            }
        };
        let workspace_root = PathBuf::from(&metadata.workspace_root)
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(&metadata.workspace_root));
        if !workspace_root.starts_with(repo_root) {
            graph.discovery_warnings.push(format!(
                "cargo workspace {} escapes repository root and was ignored",
                workspace_root.display()
            ));
            continue;
        }
        if !seen_workspaces.insert(workspace_root.clone()) {
            continue;
        }
        for package in &metadata.packages {
            if let Ok(path) = PathBuf::from(&package.manifest_path).canonicalize() {
                seen_member_manifests.insert(path);
            }
        }
        add_cargo_workspace(repo_root, metadata, graph);
    }
}

fn add_cargo_workspace(repo_root: &Path, metadata: CargoMetadata, graph: &mut WorkspaceGraph) {
    let member_ids = metadata
        .workspace_members
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let packages = metadata
        .packages
        .iter()
        .filter(|package| member_ids.contains(package.id.as_str()))
        .collect::<Vec<_>>();
    if packages.is_empty() {
        return;
    }
    let workspace_root = PathBuf::from(&metadata.workspace_root)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(&metadata.workspace_root));
    let root = relative_path(repo_root, &workspace_root).unwrap_or_else(|| ".".to_string());
    let scope = scope_id(&root);
    let workspace_id = format!("cargo.{scope}");
    let executor_id = unique_id(&graph.executors, &workspace_id);
    graph.executors.insert(
        executor_id.clone(),
        Executor {
            id: executor_id.clone(),
            kind: ExecutorKind::Local,
            environment: None,
        },
    );

    let mut package_component_ids = HashMap::new();
    let mut package_roots = HashMap::new();
    let mut package_names = HashMap::new();
    for package in &packages {
        let manifest = PathBuf::from(&package.manifest_path);
        let package_root = manifest.parent().unwrap_or(&workspace_root);
        let Some(package_root_relative) = relative_path(repo_root, package_root) else {
            graph.discovery_warnings.push(format!(
                "cargo package '{}' is outside the repository and was ignored",
                package.name
            ));
            continue;
        };
        let base_id = format!("cargo.{scope}.{}", sanitize_id(&package.name));
        let component_id = unique_id(&graph.components, &base_id);
        package_component_ids.insert(package.id.clone(), component_id.clone());
        package_roots.insert(package.id.clone(), package_root_relative.clone());
        package_names.insert(package.name.clone(), component_id.clone());
        let manifest_relative = relative_path(repo_root, &manifest)
            .unwrap_or_else(|| join_relative(&package_root_relative, "Cargo.toml"));
        graph.components.insert(
            component_id.clone(),
            WorkspaceComponent {
                id: component_id,
                root: package_root_relative.clone(),
                include: cargo_package_inputs(&package_root_relative, &manifest_relative),
                exclude: Vec::new(),
                executor: executor_id.clone(),
                depends_on: Vec::new(),
            },
        );
    }

    let roots_to_components = package_component_ids
        .iter()
        .filter_map(|(package_id, component_id)| {
            package_roots
                .get(package_id)
                .map(|root| (root.clone(), component_id.clone()))
        })
        .collect::<HashMap<_, _>>();
    for package in &packages {
        let Some(component_id) = package_component_ids.get(&package.id) else {
            continue;
        };
        let mut dependencies = BTreeSet::new();
        for dependency in &package.dependencies {
            let by_path = dependency.path.as_ref().and_then(|path| {
                let path = PathBuf::from(path)
                    .canonicalize()
                    .unwrap_or_else(|_| PathBuf::from(path));
                relative_path(repo_root, &path)
                    .and_then(|root| roots_to_components.get(&root).cloned())
            });
            if let Some(dependency_id) =
                by_path.or_else(|| package_names.get(&dependency.name).cloned())
                && &dependency_id != component_id
            {
                dependencies.insert(dependency_id);
            }
        }
        if let Some(component) = graph.components.get_mut(component_id) {
            component.depends_on = dependencies.into_iter().collect();
        }
    }

    let workspace_manifest = join_relative(&root, "Cargo.toml");
    let lockfile = join_relative(&root, "Cargo.lock");
    let workspace_check_id = unique_id(&graph.checks, &format!("cargo.{scope}.workspace"));
    graph.checks.insert(
        workspace_check_id.clone(),
        WorkspaceCheck {
            id: workspace_check_id,
            label: format!("Test Cargo workspace {root}"),
            command: format!(
                "cargo test --manifest-path {} --workspace --all-targets",
                shell_quote(&workspace_manifest)
            ),
            cwd: ".".to_string(),
            executor: executor_id.clone(),
            components: Vec::new(),
            trigger: CheckTrigger::Changed,
            inputs: vec![workspace_manifest.clone(), lockfile],
            outputs: Vec::new(),
            depends_on: Vec::new(),
            timeout_seconds: 600,
        },
    );

    for package in &packages {
        let Some(component_id) = package_component_ids.get(&package.id) else {
            continue;
        };
        let check_id = unique_id(
            &graph.checks,
            &format!("{}.test", component_id.trim_end_matches(".test")),
        );
        let inputs = graph.components[component_id].include.clone();
        graph.checks.insert(
            check_id.clone(),
            WorkspaceCheck {
                id: check_id,
                label: format!("Test Cargo package {}", package.name),
                command: format!(
                    "cargo test --manifest-path {} -p {} --all-targets",
                    shell_quote(&workspace_manifest),
                    shell_quote(&package.name)
                ),
                cwd: ".".to_string(),
                executor: executor_id.clone(),
                components: vec![component_id.clone()],
                trigger: CheckTrigger::Changed,
                inputs,
                outputs: Vec::new(),
                depends_on: Vec::new(),
                timeout_seconds: 600,
            },
        );
    }

    let members = metadata
        .workspace_members
        .iter()
        .filter_map(|id| package_component_ids.get(id).cloned())
        .collect::<Vec<_>>();
    let default_members = metadata
        .workspace_default_members
        .iter()
        .filter_map(|id| package_component_ids.get(id).cloned())
        .collect::<Vec<_>>();
    graph.cargo_workspaces.insert(
        workspace_id.clone(),
        CargoWorkspace {
            id: workspace_id,
            root,
            manifest_path: workspace_manifest,
            members,
            default_members,
        },
    );
}

fn add_fallback_cargo_package(repo_root: &Path, manifest: &Path, graph: &mut WorkspaceGraph) {
    let root = manifest
        .parent()
        .and_then(|path| relative_path(repo_root, path))
        .unwrap_or_else(|| ".".to_string());
    let scope = scope_id(&root);
    let executor_id = unique_id(&graph.executors, &format!("cargo.{scope}"));
    graph.executors.insert(
        executor_id.clone(),
        Executor {
            id: executor_id.clone(),
            kind: ExecutorKind::Local,
            environment: None,
        },
    );
    let component_id = unique_id(&graph.components, &format!("cargo.{scope}.package"));
    let manifest_relative = join_relative(&root, "Cargo.toml");
    graph.components.insert(
        component_id.clone(),
        WorkspaceComponent {
            id: component_id.clone(),
            root: root.clone(),
            include: cargo_package_inputs(&root, &manifest_relative),
            exclude: Vec::new(),
            executor: executor_id.clone(),
            depends_on: Vec::new(),
        },
    );
    let check_id = unique_id(&graph.checks, &format!("cargo.{scope}.test"));
    graph.checks.insert(
        check_id.clone(),
        WorkspaceCheck {
            id: check_id,
            label: format!("Test Cargo package {root}"),
            command: format!(
                "cargo test --manifest-path {} --all-targets",
                shell_quote(&manifest_relative)
            ),
            cwd: ".".to_string(),
            executor: executor_id,
            components: vec![component_id],
            trigger: CheckTrigger::Changed,
            inputs: cargo_package_inputs(&root, &manifest_relative),
            outputs: Vec::new(),
            depends_on: Vec::new(),
            timeout_seconds: 600,
        },
    );
}

fn cargo_package_inputs(root: &str, manifest: &str) -> Vec<String> {
    [
        manifest.to_string(),
        join_relative(root, "build.rs"),
        join_relative(root, "src/**"),
        join_relative(root, "tests/**"),
        join_relative(root, "examples/**"),
        join_relative(root, "benches/**"),
    ]
    .into_iter()
    .collect()
}

fn discover_javascript(repo_root: &Path, files: &ManifestFiles, graph: &mut WorkspaceGraph) {
    let deno_roots = files
        .deno
        .iter()
        .filter_map(|path| path.parent().map(Path::to_path_buf))
        .collect::<HashSet<_>>();
    for deno in &files.deno {
        let root_path = deno.parent().unwrap_or(repo_root);
        let root = relative_path(repo_root, root_path).unwrap_or_else(|| ".".to_string());
        let scope = scope_id(&root);
        let executor_id = unique_id(&graph.executors, &format!("deno.{scope}"));
        graph.executors.insert(
            executor_id.clone(),
            Executor {
                id: executor_id.clone(),
                kind: ExecutorKind::Local,
                environment: None,
            },
        );
        let component_id = unique_id(&graph.components, &format!("deno.{scope}"));
        let mut include = web_component_inputs(repo_root, &root);
        include.push(join_relative(&root, "deno.json"));
        include.push(join_relative(&root, "deno.jsonc"));
        include.push(join_relative(&root, "deno.lock"));
        include.push(join_relative(&root, "package.json"));
        include.sort();
        include.dedup();
        graph.components.insert(
            component_id.clone(),
            WorkspaceComponent {
                id: component_id.clone(),
                root: root.clone(),
                include: include.clone(),
                exclude: vec![join_relative(&root, "node_modules/**")],
                executor: executor_id.clone(),
                depends_on: Vec::new(),
            },
        );
        let tasks = read_json(deno)
            .ok()
            .and_then(|value| value.get("tasks").and_then(Value::as_object).cloned())
            .unwrap_or_default();
        for task in tasks.keys().filter(|task| {
            task.as_str() == "test"
                || task.starts_with("test:")
                || task.as_str() == "build"
                || task.starts_with("build:")
        }) {
            let check_id = unique_id(
                &graph.checks,
                &format!("deno.{scope}.{}", sanitize_id(task)),
            );
            let builds_web = task.contains("build") && repo_root.join("webui").is_dir();
            graph.checks.insert(
                check_id.clone(),
                WorkspaceCheck {
                    id: check_id,
                    label: format!("Deno task {task}"),
                    command: format!("deno task {}", shell_quote(task)),
                    cwd: root.clone(),
                    executor: executor_id.clone(),
                    components: vec![component_id.clone()],
                    trigger: if builds_web {
                        CheckTrigger::Needed
                    } else {
                        CheckTrigger::Changed
                    },
                    inputs: include.clone(),
                    outputs: builds_web
                        .then(|| "webui/dist/**".to_string())
                        .into_iter()
                        .collect(),
                    depends_on: Vec::new(),
                    timeout_seconds: 600,
                },
            );
        }
    }

    for package in &files.package {
        let root_path = package.parent().unwrap_or(repo_root);
        if deno_roots.contains(root_path) {
            continue;
        }
        let root = relative_path(repo_root, root_path).unwrap_or_else(|| ".".to_string());
        let scope = scope_id(&root);
        let executor_id = unique_id(&graph.executors, &format!("node.{scope}"));
        graph.executors.insert(
            executor_id.clone(),
            Executor {
                id: executor_id.clone(),
                kind: ExecutorKind::Local,
                environment: None,
            },
        );
        let component_id = unique_id(&graph.components, &format!("node.{scope}"));
        let include = web_component_inputs(repo_root, &root);
        graph.components.insert(
            component_id.clone(),
            WorkspaceComponent {
                id: component_id.clone(),
                root: root.clone(),
                include: include.clone(),
                exclude: vec![
                    join_relative(&root, "node_modules/**"),
                    join_relative(&root, "dist/**"),
                ],
                executor: executor_id.clone(),
                depends_on: Vec::new(),
            },
        );
        let scripts = read_json(package)
            .ok()
            .and_then(|value| value.get("scripts").and_then(Value::as_object).cloned())
            .unwrap_or_default();
        for script in scripts.keys().filter(|script| {
            script.as_str() == "test"
                || script.starts_with("test:")
                || script.as_str() == "build"
                || script.starts_with("build:")
        }) {
            let check_id = unique_id(
                &graph.checks,
                &format!("node.{scope}.{}", sanitize_id(script)),
            );
            graph.checks.insert(
                check_id.clone(),
                WorkspaceCheck {
                    id: check_id,
                    label: format!("Node script {script}"),
                    command: format!("npm run {}", shell_quote(script)),
                    cwd: root.clone(),
                    executor: executor_id.clone(),
                    components: vec![component_id.clone()],
                    trigger: if script.starts_with("build") {
                        CheckTrigger::Needed
                    } else {
                        CheckTrigger::Changed
                    },
                    inputs: include.clone(),
                    outputs: Vec::new(),
                    depends_on: Vec::new(),
                    timeout_seconds: 600,
                },
            );
        }
    }
}

fn discover_go(repo_root: &Path, manifests: &[PathBuf], graph: &mut WorkspaceGraph) -> Result<()> {
    for manifest in manifests {
        let root_path = manifest.parent().unwrap_or(repo_root);
        let root = relative_path(repo_root, root_path)
            .with_context(|| format!("Go module {} escapes repository", manifest.display()))?;
        let scope = scope_id(&root);
        let executor_id = unique_id(&graph.executors, &format!("go.{scope}"));
        graph.executors.insert(
            executor_id.clone(),
            Executor {
                id: executor_id.clone(),
                kind: ExecutorKind::Local,
                environment: None,
            },
        );
        let component_id = unique_id(&graph.components, &format!("go.{scope}"));
        let inputs = vec![
            join_relative(&root, "go.mod"),
            join_relative(&root, "**/*.go"),
        ];
        graph.components.insert(
            component_id.clone(),
            WorkspaceComponent {
                id: component_id.clone(),
                root: root.clone(),
                include: inputs.clone(),
                exclude: Vec::new(),
                executor: executor_id.clone(),
                depends_on: Vec::new(),
            },
        );
        let check_id = unique_id(&graph.checks, &format!("go.{scope}.test"));
        graph.checks.insert(
            check_id.clone(),
            WorkspaceCheck {
                id: check_id,
                label: format!("Test Go module {root}"),
                command: "go test ./...".to_string(),
                cwd: root,
                executor: executor_id,
                components: vec![component_id],
                trigger: CheckTrigger::Changed,
                inputs,
                outputs: Vec::new(),
                depends_on: Vec::new(),
                timeout_seconds: 600,
            },
        );
    }
    Ok(())
}

fn discover_python(
    repo_root: &Path,
    manifests: &[PathBuf],
    graph: &mut WorkspaceGraph,
) -> Result<()> {
    let mut roots = BTreeSet::new();
    for manifest in manifests {
        roots.insert(manifest.parent().unwrap_or(repo_root).to_path_buf());
    }
    for root_path in roots {
        let root = relative_path(repo_root, &root_path).with_context(|| {
            format!("Python project {} escapes repository", root_path.display())
        })?;
        let scope = scope_id(&root);
        let executor_id = unique_id(&graph.executors, &format!("python.{scope}"));
        graph.executors.insert(
            executor_id.clone(),
            Executor {
                id: executor_id.clone(),
                kind: ExecutorKind::Local,
                environment: None,
            },
        );
        let component_id = unique_id(&graph.components, &format!("python.{scope}"));
        let inputs = vec![
            join_relative(&root, "pyproject.toml"),
            join_relative(&root, "requirements.txt"),
            join_relative(&root, "**/*.py"),
        ];
        graph.components.insert(
            component_id.clone(),
            WorkspaceComponent {
                id: component_id.clone(),
                root: root.clone(),
                include: inputs.clone(),
                exclude: vec![join_relative(&root, ".venv/**")],
                executor: executor_id.clone(),
                depends_on: Vec::new(),
            },
        );
        let check_id = unique_id(&graph.checks, &format!("python.{scope}.test"));
        graph.checks.insert(
            check_id.clone(),
            WorkspaceCheck {
                id: check_id,
                label: format!("Test Python project {root}"),
                command: "python -m pytest".to_string(),
                cwd: root,
                executor: executor_id,
                components: vec![component_id],
                trigger: CheckTrigger::Changed,
                inputs,
                outputs: Vec::new(),
                depends_on: Vec::new(),
                timeout_seconds: 600,
            },
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CommandKind {
    Cargo,
    Web,
    Go,
    Python,
    Other,
}

fn canonical_commands(repo_root: &Path, environment: Option<&EnvironmentConfig>) -> Vec<String> {
    let mut commands = environment
        .map(|config| config.guard_commands.clone())
        .unwrap_or_default();
    if let Ok(inspection) = crate::init::inspect(repo_root) {
        commands.extend(inspection.documented_guard_commands);
    }
    let mut seen = HashSet::new();
    commands.retain(|command| {
        let trimmed = command.trim().to_string();
        !trimmed.is_empty() && seen.insert(trimmed)
    });
    commands
}

fn apply_canonical_commands(repo_root: &Path, commands: &[String], graph: &mut WorkspaceGraph) {
    if commands.is_empty() {
        return;
    }
    graph
        .executors
        .entry("project".to_string())
        .or_insert_with(|| Executor {
            id: "project".to_string(),
            kind: ExecutorKind::Project,
            environment: None,
        });
    let kinds = commands
        .iter()
        .map(|command| command_kind(command))
        .collect::<HashSet<_>>();
    graph.checks.retain(|id, _| {
        !((kinds.contains(&CommandKind::Cargo) && id.starts_with("cargo."))
            || (kinds.contains(&CommandKind::Web)
                && (id.starts_with("deno.") || id.starts_with("node.")))
            || (kinds.contains(&CommandKind::Go) && id.starts_with("go."))
            || (kinds.contains(&CommandKind::Python) && id.starts_with("python.")))
    });

    let mut previous = None;
    for (index, command) in commands.iter().enumerate() {
        let kind = command_kind(command);
        let components = graph
            .components
            .keys()
            .filter(|id| component_matches_kind(id, kind))
            .cloned()
            .collect::<Vec<_>>();
        let components = if components.is_empty() {
            graph.components.keys().cloned().collect()
        } else {
            components
        };
        let mut inputs = components
            .iter()
            .filter_map(|id| graph.components.get(id))
            .flat_map(|component| component.include.iter().cloned())
            .collect::<Vec<_>>();
        if kind == CommandKind::Cargo {
            for workspace in graph.cargo_workspaces.values() {
                inputs.push(workspace.manifest_path.clone());
                inputs.push(join_relative(&workspace.root, "Cargo.lock"));
            }
        }
        if inputs.is_empty() {
            inputs.push("**".to_string());
        }
        inputs.sort();
        inputs.dedup();
        let id = format!("canonical-guard-{}", index + 1);
        let builds_web = kind == CommandKind::Web
            && command.contains("build")
            && repo_root.join("webui").is_dir();
        graph.checks.insert(
            id.clone(),
            WorkspaceCheck {
                id: id.clone(),
                label: command.clone(),
                command: command.clone(),
                cwd: ".".to_string(),
                executor: "project".to_string(),
                components,
                trigger: if builds_web {
                    CheckTrigger::Needed
                } else {
                    CheckTrigger::Changed
                },
                inputs,
                outputs: builds_web
                    .then(|| "webui/dist/**".to_string())
                    .into_iter()
                    .collect(),
                depends_on: previous.into_iter().collect(),
                timeout_seconds: 600,
            },
        );
        previous = Some(id);
    }
}

fn command_kind(command: &str) -> CommandKind {
    let command = command.trim_start();
    if command.starts_with("cargo ") {
        CommandKind::Cargo
    } else if command.starts_with("deno ")
        || command.starts_with("npm ")
        || command.starts_with("pnpm ")
        || command.starts_with("yarn ")
        || command.starts_with("bun ")
    {
        CommandKind::Web
    } else if command.starts_with("go ") {
        CommandKind::Go
    } else if command.starts_with("pytest") || command.starts_with("python ") {
        CommandKind::Python
    } else {
        CommandKind::Other
    }
}

fn component_matches_kind(id: &str, kind: CommandKind) -> bool {
    match kind {
        CommandKind::Cargo => id.starts_with("cargo."),
        CommandKind::Web => id.starts_with("deno.") || id.starts_with("node."),
        CommandKind::Go => id.starts_with("go."),
        CommandKind::Python => id.starts_with("python."),
        CommandKind::Other => true,
    }
}

fn web_component_inputs(repo_root: &Path, root: &str) -> Vec<String> {
    let mut inputs = vec![
        join_relative(root, "package.json"),
        join_relative(root, "deno.json"),
        join_relative(root, "deno.jsonc"),
        join_relative(root, "deno.lock"),
        join_relative(root, "src/**"),
    ];
    if root == "." && repo_root.join("webui").is_dir() {
        inputs.push("webui/**".to_string());
    } else {
        inputs.push(join_relative(root, "**/*.{js,jsx,ts,tsx,css,html}"));
    }
    inputs
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read JSON manifest {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse JSON manifest {}", path.display()))
}

fn relative_path(repo_root: &Path, path: &Path) -> Option<String> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let relative = canonical.strip_prefix(repo_root).ok()?;
    if relative.as_os_str().is_empty() {
        Some(".".to_string())
    } else {
        Some(relative.to_string_lossy().replace('\\', "/"))
    }
}

fn display_relative(repo_root: &Path, path: &Path) -> String {
    relative_path(repo_root, path).unwrap_or_else(|| path.display().to_string())
}

fn join_relative(root: &str, child: &str) -> String {
    if root == "." || root.is_empty() {
        child.to_string()
    } else {
        format!("{root}/{child}")
    }
}

fn scope_id(root: &str) -> String {
    if root == "." {
        "root".to_string()
    } else {
        sanitize_id(root)
    }
}

fn sanitize_id(value: &str) -> String {
    let mut result = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    while result.contains("--") {
        result = result.replace("--", "-");
    }
    result.trim_matches(['-', '.']).to_string()
}

fn unique_id<T>(items: &BTreeMap<String, T>, base: &str) -> String {
    if !items.contains_key(base) {
        return base.to_string();
    }
    (2..)
        .map(|suffix| format!("{base}-{suffix}"))
        .find(|candidate| !items.contains_key(candidate))
        .expect("unbounded unique workspace id")
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checks::plan_checks_for_paths;
    use crate::environment::{EnvironmentBackend, EnvironmentMode};

    fn write(root: &Path, relative: &str, content: &str) {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    fn package(root: &Path, relative: &str, name: &str, dependencies: &str) {
        write(
            root,
            &format!("{relative}/Cargo.toml"),
            &format!(
                "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n{dependencies}\n"
            ),
        );
        write(
            root,
            &format!("{relative}/src/lib.rs"),
            "pub fn value() {}\n",
        );
    }

    #[test]
    fn cargo_metadata_preserves_members_defaults_and_dependency_edges() {
        let repo = tempfile::tempdir().unwrap();
        write(
            repo.path(),
            "Cargo.toml",
            "[workspace]\nmembers = [\"shared\", \"api\", \"worker\"]\ndefault-members = [\"api\"]\nresolver = \"3\"\n",
        );
        package(repo.path(), "shared", "shared", "");
        package(
            repo.path(),
            "api",
            "api",
            "[dependencies]\nshared = { path = \"../shared\" }",
        );
        package(
            repo.path(),
            "worker",
            "worker",
            "[dependencies]\nshared = { path = \"../shared\" }",
        );
        write(repo.path(), "Cargo.lock", "# lock\n");

        let graph = discover_workspace(repo.path(), None).unwrap();
        assert_eq!(graph.cargo_workspaces.len(), 1);
        let workspace = graph.cargo_workspaces.values().next().unwrap();
        assert_eq!(workspace.members.len(), 3);
        assert_eq!(workspace.default_members.len(), 1);
        assert!(workspace.default_members[0].ends_with(".api"));
        let shared = workspace
            .members
            .iter()
            .find(|id| id.ends_with(".shared"))
            .unwrap();
        for name in ["api", "worker"] {
            let component = graph
                .components
                .values()
                .find(|component| component.id.ends_with(&format!(".{name}")))
                .unwrap();
            assert!(component.depends_on.contains(shared));
        }
        assert!(graph.checks.values().any(|check| {
            check.command.contains("--workspace")
                && check.inputs.contains(&"Cargo.lock".to_string())
        }));

        let document = graph.to_document();
        document.save(repo.path()).unwrap();
        let restored = crate::workspace::WorkspaceConfigDocument::load(repo.path())
            .unwrap()
            .unwrap()
            .normalize()
            .unwrap();
        assert_eq!(restored.cargo_workspaces, graph.cargo_workspaces);

        let shared_plan =
            plan_checks_for_paths(&graph, vec!["shared/src/lib.rs".to_string()]).unwrap();
        assert!(shared_plan.affected_components.contains(shared));
        for name in ["api", "worker"] {
            assert!(
                shared_plan
                    .affected_components
                    .iter()
                    .any(|id| id.ends_with(&format!(".{name}")))
            );
        }

        let lock_plan = plan_checks_for_paths(&graph, vec!["Cargo.lock".to_string()]).unwrap();
        assert!(
            lock_plan
                .checks
                .iter()
                .any(|id| { graph.checks[id].command.contains("--workspace") })
        );
    }

    #[test]
    fn single_rust_package_has_changed_and_unchanged_handoff_plans() {
        let repo = tempfile::tempdir().unwrap();
        package(repo.path(), ".", "single", "");

        let graph = discover_workspace(repo.path(), None).unwrap();
        assert_eq!(graph.cargo_workspaces.len(), 1);
        assert_eq!(
            graph
                .cargo_workspaces
                .values()
                .next()
                .unwrap()
                .members
                .len(),
            1
        );

        let changed = plan_checks_for_paths(&graph, vec!["src/lib.rs".to_string()]).unwrap();
        assert!(!changed.checks.is_empty());
        assert!(!changed.affected_components.is_empty());

        let unchanged = plan_checks_for_paths(&graph, Vec::new()).unwrap();
        assert!(unchanged.is_no_change());
        assert!(unchanged.checks.is_empty());
    }

    #[test]
    fn independent_cargo_workspaces_remain_separate() {
        let repo = tempfile::tempdir().unwrap();
        package(repo.path(), "services/api", "api", "");
        package(repo.path(), "tools/codegen", "codegen", "");

        let graph = discover_workspace(repo.path(), None).unwrap();
        assert_eq!(graph.cargo_workspaces.len(), 2);
        let roots = graph
            .cargo_workspaces
            .values()
            .map(|workspace| workspace.root.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(roots, BTreeSet::from(["services/api", "tools/codegen"]));
    }

    #[test]
    fn polyglot_services_receive_distinct_components_and_executors() {
        let repo = tempfile::tempdir().unwrap();
        write(
            repo.path(),
            "web/package.json",
            r#"{"name":"web","scripts":{"test":"vitest"}}"#,
        );
        write(
            repo.path(),
            "worker/pyproject.toml",
            "[project]\nname = \"worker\"\n",
        );
        write(
            repo.path(),
            "worker/test_worker.py",
            "def test_ok(): assert True\n",
        );
        write(repo.path(), "gateway/go.mod", "module example/gateway\n");
        write(repo.path(), "gateway/main.go", "package main\n");

        let graph = discover_workspace(repo.path(), None).unwrap();
        assert!(graph.components.keys().any(|id| id.starts_with("node.")));
        assert!(graph.components.keys().any(|id| id.starts_with("python.")));
        assert!(graph.components.keys().any(|id| id.starts_with("go.")));
        assert!(graph.executors.len() >= 3);

        let web_component = graph
            .components
            .values()
            .find(|component| component.root == "web")
            .unwrap();
        let web_plan = plan_checks_for_paths(&graph, vec!["web/src/app.ts".to_string()]).unwrap();
        assert_eq!(web_plan.affected_components, vec![web_component.id.clone()]);
        assert!(!web_plan.checks.is_empty());
        assert!(
            web_plan
                .checks
                .iter()
                .all(|id| { graph.checks[id].components.contains(&web_component.id) })
        );
        assert!(web_plan.checks.iter().all(|id| {
            let check = &graph.checks[id];
            !check.command.contains("pytest") && !check.command.starts_with("go test")
        }));
    }

    #[test]
    fn canonical_guards_replace_inferred_checks_and_preserve_order() {
        let repo = tempfile::tempdir().unwrap();
        package(repo.path(), ".", "app", "");
        write(
            repo.path(),
            "deno.json",
            r#"{"tasks":{"build:web":"echo build","test:web":"echo test"}}"#,
        );
        std::fs::create_dir_all(repo.path().join("webui")).unwrap();
        let environment = EnvironmentConfig {
            mode: EnvironmentMode::Local,
            backend: EnvironmentBackend::Local,
            image: "local".to_string(),
            init_commands: Vec::new(),
            setup_commands: Vec::new(),
            session_commands: Vec::new(),
            guard_commands: vec![
                "deno task build:web".to_string(),
                "cargo test --all-targets".to_string(),
            ],
            prepared_image: None,
            source: None,
            dockerfile: None,
        };

        let graph = discover_workspace(repo.path(), Some(&environment)).unwrap();
        assert_eq!(graph.checks.len(), 2);
        assert_eq!(
            graph.checks["canonical-guard-1"].command,
            environment.guard_commands[0]
        );
        assert_eq!(
            graph.checks["canonical-guard-2"].depends_on,
            vec!["canonical-guard-1"]
        );
        assert_eq!(
            graph.checks["canonical-guard-1"].trigger,
            CheckTrigger::Needed
        );
        assert_eq!(
            graph.checks["canonical-guard-1"].outputs,
            vec!["webui/dist/**"]
        );
    }
}
