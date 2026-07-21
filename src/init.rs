//! `pb init` — inspect a project and configure it for use with pb.
//!
//! The command walks the project root and detects:
//! - Container environments (devcontainer, Dockerfile, GitLab CI image)
//! - Language ecosystems (Rust, Python, Node/Deno, Go)
//! - Existing agent / coding-assistant documentation
//!
//! Based on the detections it writes (or confirms) the project environment,
//! workspace, strict workflow, and durable goal policy under `.pb/`, then
//! prints a summary of what it found and what it configured.
//! MCP tool-effect declarations such as `capabilities.read_only_tools` remain explicit operator
//! configuration: `pb init` never infers that an external operation is read-only.

use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::environment::{
    EnvironmentBackend, EnvironmentCache, EnvironmentConfig, EnvironmentMode,
};
use crate::environment_lock::{
    EnvironmentEvidence, EnvironmentEvidenceDocument, EnvironmentRequirement, HostCapability,
};

pub use crate::environment_lock::HostCapability as HostRequirement;

// ── detection results ────────────────────────────────────────────────────────

/// Everything we learn from inspecting the project root.
#[derive(Debug, Default)]
pub struct ProjectInspection {
    // Container / CI
    pub devcontainer_image: Option<String>,
    pub devcontainer_init_commands: Vec<String>,
    pub has_dockerfile: bool,
    pub gitlab_ci_image: Option<String>,
    pub has_github_workflows: bool,
    pub has_kubernetes_config: bool,

    // Language ecosystems
    pub has_cargo_toml: bool,
    pub has_pyproject_toml: bool,
    pub has_requirements_txt: bool,
    pub has_package_json: bool,
    pub has_deno_lock: bool,
    pub has_go_mod: bool,
    pub has_package_swift: bool,

    // Existing agent docs
    pub existing_agent_docs: Vec<PathBuf>,

    // Already configured?
    pub has_pb_environment: bool,
    pub has_pb_workspace: bool,
    pub has_pb_workflow: bool,
    pub has_pb_goal: bool,

    // Vision / image assets
    /// Project contains image files (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`).
    pub has_image_assets: bool,

    // Scout signals
    pub setup_commands: Vec<String>,
    pub session_commands: Vec<String>,
    pub guard_commands: Vec<String>,
    pub documented_guard_commands: Vec<String>,
    pub prefers_local_backend: bool,
    pub prefers_container_backend: bool,
    pub host_requirements: Vec<HostRequirement>,
    pub environment_evidence: Vec<EnvironmentEvidence>,
    pub dependency_key_files: Vec<PathBuf>,
    pub scout_sources: Vec<PathBuf>,
}

/// Inspect a project directory and return everything we detected.
pub fn inspect(root: &Path) -> Result<ProjectInspection> {
    let mut info = ProjectInspection::default();

    // --- devcontainer ---
    let dc_json = root.join(".devcontainer").join("devcontainer.json");
    let dc_json_top = root.join("devcontainer.json");
    for dc_path in [&dc_json, &dc_json_top] {
        if is_regular_project_file(dc_path) {
            record_container_signal(root, dc_path, &mut info, "devcontainer configuration");
            if let Ok(text) = std::fs::read_to_string(dc_path)
                && let Some((image, inits)) = parse_devcontainer_json(&text)
            {
                info.devcontainer_image = Some(image);
                info.devcontainer_init_commands = inits;
            }
            break;
        }
    }

    // --- Dockerfile ---
    if is_regular_project_file(&root.join("Dockerfile")) {
        info.has_dockerfile = true;
        record_container_signal(
            root,
            &root.join("Dockerfile"),
            &mut info,
            "Dockerfile build environment",
        );
    }

    // --- GitLab CI ---
    let gitlab_ci = root.join(".gitlab-ci.yml");
    if is_regular_project_file(&gitlab_ci)
        && let Ok(text) = std::fs::read_to_string(&gitlab_ci)
    {
        info.gitlab_ci_image = parse_gitlab_ci_image(&text);
        inspect_scout_text(root, &gitlab_ci, &text, &mut info);
    }

    // --- GitHub Actions and Kubernetes manifests ---
    let workflow_dir = root.join(".github").join("workflows");
    if let Ok(entries) = std::fs::read_dir(&workflow_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if entry.file_type().is_ok_and(|file_type| file_type.is_file())
                && matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("yml" | "yaml")
                )
            {
                info.has_github_workflows = true;
                if let Ok(text) = std::fs::read_to_string(&path) {
                    inspect_scout_text(root, &path, &text, &mut info);
                }
            }
        }
    }
    for rel in ["k8s", "kubernetes", "helm", "charts"] {
        if is_regular_project_directory(&root.join(rel)) {
            info.has_kubernetes_config = true;
            info.prefers_container_backend = true;
            record_container_signal(
                root,
                &root.join(rel),
                &mut info,
                "container deployment configuration",
            );
        }
    }

    // --- Language ecosystems ---
    inspect_language_manifests(root, root, 0, &mut info);

    // --- Existing agent docs ---
    let agent_doc_candidates = [
        "AGENT.md",
        "AGENTS.md",
        "README.md",
        ".github/copilot-instructions.md",
        "CLAUDE.md",
        ".cursor/rules",
        "GEMINI.md",
        ".aider.conf.yml",
        ".continue/config.json",
        "cline_docs/systemPrompt.md",
    ];
    for rel in &agent_doc_candidates {
        let p = root.join(rel);
        if is_regular_project_file(&p) {
            info.existing_agent_docs.push(p.clone());
            if let Ok(text) = std::fs::read_to_string(&p) {
                inspect_scout_text(root, &p, &text, &mut info);
            }
        }
    }

    let dockerfile = root.join("Dockerfile");
    if is_regular_project_file(&dockerfile)
        && let Ok(text) = std::fs::read_to_string(&dockerfile)
    {
        inspect_scout_text(root, &dockerfile, &text, &mut info);
    }

    // --- Already configured? ---
    info.has_pb_environment = root.join(".pb").join("environment.toml").exists();
    info.has_pb_workspace = root.join(".pb").join("workspace.toml").exists();
    info.has_pb_workflow = root.join(".pb").join("workflow.toml").exists();
    info.has_pb_goal = root.join(".pb").join("goal.toml").exists();

    info.environment_evidence.sort_by(|left, right| {
        left.component
            .cmp(&right.component)
            .then(left.source_path.cmp(&right.source_path))
            .then(left.detail.cmp(&right.detail))
    });

    Ok(info)
}

fn is_regular_project_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .is_ok_and(|metadata| metadata.file_type().is_file() && !metadata.file_type().is_symlink())
}

fn is_regular_project_directory(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .is_ok_and(|metadata| metadata.file_type().is_dir() && !metadata.file_type().is_symlink())
}

fn inspect_language_manifests(root: &Path, dir: &Path, depth: usize, info: &mut ProjectInspection) {
    if depth > 8 {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            if name.ends_with(".xcodeproj") {
                record_host_requirement(root, &path, info, HostRequirement::XcodeProject);
                continue;
            }
            if name.ends_with(".xcworkspace") {
                record_host_requirement(root, &path, info, HostRequirement::XcodeWorkspace);
                continue;
            }
            if matches!(
                name.as_ref(),
                ".git"
                    | ".pb"
                    | ".build"
                    | ".venv"
                    | "venv"
                    | "target"
                    | "node_modules"
                    | "dist"
                    | "build"
                    | "Pods"
                    | "DerivedData"
            ) {
                continue;
            }
            inspect_language_manifests(root, &path, depth + 1, info);
            continue;
        }
        let command = |cmd: &str| normalize_command(root, &path, cmd);
        match name.as_ref() {
            "Cargo.toml" => {
                info.has_cargo_toml = true;
                record_dependency_key_file(root, &path, info);
                record_setup_command(
                    root,
                    &path,
                    &command(
                        "if test -f Cargo.lock; then cargo fetch --locked; else cargo fetch; fi",
                    ),
                    info,
                );
                record_guard_command(root, &path, &command("cargo test"), info);
            }
            "pyproject.toml" => {
                info.has_pyproject_toml = true;
                record_dependency_key_file(root, &path, info);
                record_setup_command(root, &path, &command("pip install -e ."), info);
            }
            "requirements.txt" => {
                info.has_requirements_txt = true;
                record_dependency_key_file(root, &path, info);
                record_setup_command(
                    root,
                    &path,
                    &command("pip install -r requirements.txt"),
                    info,
                );
            }
            "package.json" => {
                info.has_package_json = true;
                record_dependency_key_file(root, &path, info);
                record_setup_command(root, &path, &command("npm ci"), info);
            }
            "deno.lock" => {
                info.has_deno_lock = true;
                record_dependency_key_file(root, &path, info);
                record_setup_command(root, &path, &command("deno install"), info);
            }
            "go.mod" => {
                info.has_go_mod = true;
                record_dependency_key_file(root, &path, info);
                record_setup_command(root, &path, &command("go mod download"), info);
            }
            "Package.swift" => {
                info.has_package_swift = true;
                record_dependency_key_file(root, &path, info);
                record_setup_command(root, &path, &command("swift package resolve"), info);
                record_guard_command(root, &path, &command("swift test"), info);
                if let Ok(text) = std::fs::read_to_string(&path)
                    && contains_apple_swift_package_signal(&text)
                {
                    record_host_requirement(root, &path, info, HostRequirement::AppleSwiftPackage);
                }
            }
            "Dockerfile" => {
                info.has_dockerfile = info.has_dockerfile || path == root.join("Dockerfile");
                info.prefers_container_backend = true;
                record_container_signal(root, &path, info, "Dockerfile build environment");
            }
            "Cargo.lock" | "uv.lock" | "poetry.lock" | "package-lock.json" | "pnpm-lock.yaml"
            | "yarn.lock" | "go.sum" | "Package.resolved" => {
                record_dependency_key_file(root, &path, info);
            }
            "flake.nix" | "flake.lock" | "shell.nix" | "default.nix" => {
                record_dependency_key_file(root, &path, info);
                record_toolchain_evidence(root, &path, info, "nix");
            }
            "mise.toml" | ".mise.toml" => {
                record_dependency_key_file(root, &path, info);
                record_toolchain_evidence(root, &path, info, "mise");
            }
            ".tool-versions" => {
                record_dependency_key_file(root, &path, info);
                record_toolchain_evidence(root, &path, info, "asdf");
            }
            "rust-toolchain" | "rust-toolchain.toml" => {
                record_dependency_key_file(root, &path, info);
                record_toolchain_evidence(root, &path, info, "rust");
            }
            ".nvmrc" | ".node-version" => {
                record_dependency_key_file(root, &path, info);
                record_toolchain_evidence(root, &path, info, "node");
            }
            ".python-version" => {
                record_dependency_key_file(root, &path, info);
                record_toolchain_evidence(root, &path, info, "python");
            }
            n if matches!(
                std::path::Path::new(n).extension().and_then(|e| e.to_str()),
                Some("png" | "jpg" | "jpeg" | "webp" | "gif")
            ) =>
            {
                info.has_image_assets = true;
            }
            n if std::path::Path::new(n).extension().and_then(|e| e.to_str())
                == Some("entitlements") =>
            {
                record_host_requirement(root, &path, info, HostRequirement::CodeSigning);
            }
            n if std::path::Path::new(n).extension().and_then(|e| e.to_str()) == Some("metal") => {
                record_host_requirement(root, &path, info, HostRequirement::Metal);
            }
            n if matches!(
                std::path::Path::new(n).extension().and_then(|e| e.to_str()),
                Some("swift" | "m" | "mm" | "h" | "hpp" | "c" | "cc" | "cpp" | "pch")
            ) =>
            {
                if let Ok(text) = std::fs::read_to_string(&path)
                    && contains_apple_framework_import(&text)
                {
                    record_host_requirement(
                        root,
                        &path,
                        info,
                        HostRequirement::AppleFrameworkSource,
                    );
                }
            }
            _ => {}
        }
    }
}

fn contains_apple_swift_package_signal(text: &str) -> bool {
    let compact = text.split_whitespace().collect::<String>();
    [
        ".macOS(",
        ".iOS(",
        ".tvOS(",
        ".watchOS(",
        ".visionOS(",
        ".linkedFramework(",
    ]
    .iter()
    .any(|signal| compact.contains(signal))
}

fn contains_apple_framework_import(text: &str) -> bool {
    const APPLE_ONLY_MODULES: &[&str] = &[
        "AppKit",
        "ARKit",
        "AuthenticationServices",
        "AVFoundation",
        "CloudKit",
        "Combine",
        "Contacts",
        "Containerization",
        "CoreBluetooth",
        "CoreData",
        "CoreGraphics",
        "CoreLocation",
        "CoreML",
        "CryptoTokenKit",
        "Darwin",
        "DriverKit",
        "EndpointSecurity",
        "IOKit",
        "LocalAuthentication",
        "MapKit",
        "Metal",
        "MetalKit",
        "NetworkExtension",
        "Photos",
        "QuartzCore",
        "RealityKit",
        "Security",
        "StoreKit",
        "SwiftUI",
        "UIKit",
        "UserNotifications",
        "Virtualization",
        "VisionKit",
        "WebKit",
    ];
    const SWIFT_DECLARATION_IMPORTS: &[&str] = &[
        "typealias",
        "struct",
        "class",
        "enum",
        "protocol",
        "let",
        "var",
        "func",
    ];

    text.lines().any(|line| {
        let trimmed = line.trim();
        let compact = trimmed.split_whitespace().collect::<String>();
        if APPLE_ONLY_MODULES.iter().any(|module| {
            compact.contains(&format!("<{module}/"))
                || compact.contains(&format!("\"{module}/"))
                || compact.contains(&format!("@import{module};"))
        }) {
            return true;
        }
        let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
        let Some(import_index) = tokens.iter().position(|token| *token == "import") else {
            return false;
        };
        let mut module_index = import_index + 1;
        if tokens
            .get(module_index)
            .is_some_and(|token| SWIFT_DECLARATION_IMPORTS.contains(token))
        {
            module_index += 1;
        }
        let Some(module) = tokens.get(module_index) else {
            return false;
        };
        let module = module
            .trim_matches(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .split('.')
            .next()
            .unwrap_or_default();
        APPLE_ONLY_MODULES.contains(&module)
    })
}

fn record_host_requirement(
    root: &Path,
    source: &Path,
    info: &mut ProjectInspection,
    requirement: HostCapability,
) {
    if !info.host_requirements.contains(&requirement) {
        info.host_requirements.push(requirement);
    }
    let relative = source.strip_prefix(root).unwrap_or(source).to_path_buf();
    let evidence = EnvironmentEvidence {
        source_path: relative,
        component: component_for_path(root, source),
        requirement: EnvironmentRequirement::HostCapability {
            capability: requirement,
        },
        detail: requirement.label().to_string(),
    };
    if !info.environment_evidence.contains(&evidence) {
        info.environment_evidence.push(evidence);
    }
    info.prefers_local_backend = true;
}

fn record_dependency_key_file(root: &Path, path: &Path, info: &mut ProjectInspection) {
    if let Ok(relative) = path.strip_prefix(root) {
        let relative = relative.to_path_buf();
        if !info.dependency_key_files.contains(&relative) {
            info.dependency_key_files.push(relative.clone());
        }
        let evidence = EnvironmentEvidence {
            source_path: relative.clone(),
            component: component_for_path(root, path),
            requirement: EnvironmentRequirement::DependencyInput {
                path: relative.clone(),
            },
            detail: format!("dependency input {}", relative.display()),
        };
        if !info.environment_evidence.contains(&evidence) {
            info.environment_evidence.push(evidence);
        }
    }
}

fn record_toolchain_evidence(root: &Path, path: &Path, info: &mut ProjectInspection, name: &str) {
    let relative = path.strip_prefix(root).unwrap_or(path).to_path_buf();
    let constraint = std::fs::read_to_string(path)
        .unwrap_or_default()
        .split_whitespace()
        .take(32)
        .collect::<Vec<_>>()
        .join(" ");
    let evidence = EnvironmentEvidence {
        source_path: relative,
        component: component_for_path(root, path),
        requirement: EnvironmentRequirement::Toolchain {
            name: name.to_string(),
            constraint: constraint.clone(),
        },
        detail: if constraint.is_empty() {
            format!("{name} toolchain declaration")
        } else {
            format!("{name} toolchain: {constraint}")
        },
    };
    if !info.environment_evidence.contains(&evidence) {
        info.environment_evidence.push(evidence);
    }
}

fn record_setup_command(root: &Path, source: &Path, command: &str, info: &mut ProjectInspection) {
    if !info
        .setup_commands
        .iter()
        .any(|existing| existing == command)
    {
        info.setup_commands.push(command.to_string());
    }
    record_environment_requirement(
        root,
        source,
        EnvironmentRequirement::SetupCommand {
            command: command.to_string(),
        },
        format!("bootstrap command {command}"),
        info,
    );
}

fn record_guard_command(root: &Path, source: &Path, command: &str, info: &mut ProjectInspection) {
    if !info
        .guard_commands
        .iter()
        .any(|existing| existing == command)
    {
        info.guard_commands.push(command.to_string());
    }
    record_environment_requirement(
        root,
        source,
        EnvironmentRequirement::GuardCommand {
            command: command.to_string(),
        },
        format!("validation command {command}"),
        info,
    );
}

fn record_container_signal(root: &Path, source: &Path, info: &mut ProjectInspection, detail: &str) {
    info.prefers_container_backend = true;
    record_environment_requirement(
        root,
        source,
        EnvironmentRequirement::ContainerSignal {
            detail: detail.to_string(),
        },
        detail.to_string(),
        info,
    );
}

fn record_environment_requirement(
    root: &Path,
    source: &Path,
    requirement: EnvironmentRequirement,
    detail: String,
    info: &mut ProjectInspection,
) {
    let evidence = EnvironmentEvidence {
        source_path: source.strip_prefix(root).unwrap_or(source).to_path_buf(),
        component: component_for_path(root, source),
        requirement,
        detail,
    };
    if !info.environment_evidence.contains(&evidence) {
        info.environment_evidence.push(evidence);
    }
}

fn component_for_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .ok()
        .and_then(|relative| relative.components().next())
        .and_then(|component| match component {
            std::path::Component::Normal(name) => name.to_str(),
            _ => None,
        })
        .filter(|name| !name.contains('.'))
        .unwrap_or("repository")
        .to_string()
}

// ── environment suggestion ────────────────────────────────────────────────────

/// Suggest an [`EnvironmentConfig`] based on the project inspection.
/// Returns `None` when nothing useful could be determined.
pub fn suggest_environment(info: &ProjectInspection) -> Option<EnvironmentConfig> {
    // 1. devcontainer takes highest priority
    if let Some(image) = &info.devcontainer_image {
        return Some(EnvironmentConfig {
            version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Pull,
            backend: EnvironmentBackend::AppleContainers,
            image: image.clone(),
            init_commands: persistent_container_setup_commands(
                info.devcontainer_init_commands.clone(),
            ),
            setup_commands: vec![],
            session_commands: vec![],
            env: discovered_container_env(info),
            bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
            runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
            resources: Default::default(),
            caches: discovered_container_caches(info),
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        });
    }

    // 2. Dockerfile in project root
    if info.has_dockerfile {
        return Some(EnvironmentConfig {
            version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Build,
            backend: EnvironmentBackend::AppleContainers,
            image: "pb-dev:latest".to_string(),
            init_commands: vec![],
            setup_commands: vec![],
            session_commands: vec![],
            env: discovered_container_env(info),
            bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
            runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
            resources: Default::default(),
            caches: discovered_container_caches(info),
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: Some(PathBuf::from("Dockerfile")),
        });
    }

    // 3. GitLab CI image
    if let Some(image) = &info.gitlab_ci_image {
        let init_commands = persistent_container_setup_commands(language_init_commands(info));
        return Some(EnvironmentConfig {
            version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Pull,
            backend: EnvironmentBackend::AppleContainers,
            image: image.clone(),
            init_commands,
            setup_commands: vec![],
            session_commands: vec![],
            env: discovered_container_env(info),
            bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
            runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
            resources: Default::default(),
            caches: discovered_container_caches(info),
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        });
    }

    // 4. Language-based well-known images
    if info.has_cargo_toml {
        return Some(EnvironmentConfig {
            version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Pull,
            backend: EnvironmentBackend::AppleContainers,
            image: "rust:latest".to_string(),
            init_commands: vec![
                "if test -f Cargo.lock; then cargo fetch --locked; else cargo fetch; fi"
                    .to_string(),
            ],
            setup_commands: vec![],
            session_commands: vec![],
            env: discovered_container_env(info),
            bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
            runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
            resources: Default::default(),
            caches: discovered_container_caches(info),
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        });
    }
    if info.has_pyproject_toml || info.has_requirements_txt {
        let init_commands = if info.has_pyproject_toml {
            vec!["pip install -e .".to_string()]
        } else {
            vec!["pip install -r requirements.txt".to_string()]
        };
        let init_commands = persistent_container_setup_commands(init_commands);
        return Some(EnvironmentConfig {
            version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Pull,
            backend: EnvironmentBackend::AppleContainers,
            image: "python:3-slim".to_string(),
            init_commands,
            setup_commands: vec![],
            session_commands: vec![],
            env: discovered_container_env(info),
            bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
            runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
            resources: Default::default(),
            caches: discovered_container_caches(info),
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        });
    }
    if info.has_deno_lock {
        return Some(EnvironmentConfig {
            version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Pull,
            backend: EnvironmentBackend::AppleContainers,
            image: "denoland/deno:latest".to_string(),
            init_commands: vec!["deno install".to_string()],
            setup_commands: vec![],
            session_commands: vec![],
            env: discovered_container_env(info),
            bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
            runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
            resources: Default::default(),
            caches: discovered_container_caches(info),
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        });
    }
    if info.has_package_json {
        return Some(EnvironmentConfig {
            version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Pull,
            backend: EnvironmentBackend::AppleContainers,
            image: "node:lts-slim".to_string(),
            init_commands: vec!["npm ci".to_string()],
            setup_commands: vec![],
            session_commands: vec![],
            env: discovered_container_env(info),
            bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
            runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
            resources: Default::default(),
            caches: discovered_container_caches(info),
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        });
    }
    if info.has_go_mod {
        return Some(EnvironmentConfig {
            version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Pull,
            backend: EnvironmentBackend::AppleContainers,
            image: "golang:latest".to_string(),
            init_commands: vec!["go mod download".to_string()],
            setup_commands: vec![],
            session_commands: vec![],
            env: discovered_container_env(info),
            bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
            runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
            resources: Default::default(),
            caches: discovered_container_caches(info),
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        });
    }
    if info.has_package_swift {
        return Some(EnvironmentConfig {
            version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Pull,
            backend: EnvironmentBackend::AppleContainers,
            image: "swift:latest".to_string(),
            init_commands: vec!["swift package resolve".to_string()],
            setup_commands: vec![],
            session_commands: vec![],
            env: discovered_container_env(info),
            bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
            runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
            resources: Default::default(),
            caches: discovered_container_caches(info),
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        });
    }

    None
}

/// Derive language-level init commands (used when an image is already known
/// from CI or devcontainer but we still need to install dependencies).
fn language_init_commands(info: &ProjectInspection) -> Vec<String> {
    let mut cmds = Vec::new();
    if info.has_cargo_toml {
        cmds.push(
            "if test -f Cargo.lock; then cargo fetch --locked; else cargo fetch; fi".to_string(),
        );
    }
    if info.has_pyproject_toml {
        cmds.push("pip install -e .".to_string());
    } else if info.has_requirements_txt {
        cmds.push("pip install -r requirements.txt".to_string());
    }
    if info.has_deno_lock {
        cmds.push("deno install".to_string());
    } else if info.has_package_json {
        cmds.push("npm ci".to_string());
    }
    if info.has_go_mod {
        cmds.push("go mod download".to_string());
    }
    if info.has_package_swift {
        cmds.push("swift package resolve".to_string());
    }
    cmds
}

fn persistent_container_setup_commands(commands: Vec<String>) -> Vec<String> {
    let uses_python = commands
        .iter()
        .any(|command| command.contains("pip install"));
    let mut persistent = Vec::new();
    if uses_python {
        persistent
            .push("test -x /opt/pb-venv/bin/python || python -m venv /opt/pb-venv".to_string());
    }
    persistent.extend(
        commands.into_iter().map(|command| {
            command.replace("pip install", "/opt/pb-venv/bin/python -m pip install")
        }),
    );
    persistent
}

fn discovered_container_env(info: &ProjectInspection) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    if info.has_pyproject_toml || info.has_requirements_txt {
        env.insert("VIRTUAL_ENV".to_string(), "/opt/pb-venv".to_string());
        env.insert(
            "PATH".to_string(),
            "/opt/pb-venv/bin:/usr/local/bin:/usr/bin:/bin".to_string(),
        );
    }
    if info.has_deno_lock {
        env.insert("DENO_DIR".to_string(), "/deno-dir".to_string());
    }
    env
}

fn discovered_container_caches(info: &ProjectInspection) -> Vec<EnvironmentCache> {
    let key_files = info.dependency_key_files.clone();
    let mut caches = Vec::new();
    let mut push = |id: &str, target: &str, trust| {
        caches.push(EnvironmentCache {
            id: id.to_string(),
            target: PathBuf::from(target),
            key_files: key_files.clone(),
            trust,
            max_bytes: None,
        });
    };
    if info.has_cargo_toml {
        push(
            "cargo-registry",
            "/usr/local/cargo/registry",
            crate::environment::CacheTrustClass::Download,
        );
        push(
            "cargo-git",
            "/usr/local/cargo/git",
            crate::environment::CacheTrustClass::Download,
        );
        push(
            "cargo-target",
            "/workspace/target",
            crate::environment::CacheTrustClass::ProjectExecutable,
        );
    }
    if info.has_pyproject_toml || info.has_requirements_txt {
        push(
            "python-venv",
            "/opt/pb-venv",
            crate::environment::CacheTrustClass::ProjectExecutable,
        );
    }
    if info.has_package_json {
        push(
            "npm-cache",
            "/root/.npm",
            crate::environment::CacheTrustClass::Download,
        );
        push(
            "node-modules",
            "/workspace/node_modules",
            crate::environment::CacheTrustClass::ProjectExecutable,
        );
    }
    if info.has_deno_lock {
        push(
            "deno-cache",
            "/deno-dir",
            crate::environment::CacheTrustClass::ProjectExecutable,
        );
    }
    if info.has_go_mod {
        push(
            "go-modules",
            "/go/pkg/mod",
            crate::environment::CacheTrustClass::Download,
        );
    }
    if info.has_package_swift {
        push(
            "swiftpm-cache",
            "/root/.cache/org.swift.swiftpm",
            crate::environment::CacheTrustClass::Download,
        );
    }
    if info.has_cargo_toml
        || info.has_pyproject_toml
        || info.has_requirements_txt
        || info.has_package_json
        || info.has_deno_lock
        || info.has_go_mod
        || info.has_package_swift
    {
        push(
            "lsp-index",
            "/var/cache/pb/lsp",
            crate::environment::CacheTrustClass::LspIndex,
        );
    }
    caches
}

pub fn local_environment(info: &ProjectInspection) -> EnvironmentConfig {
    EnvironmentConfig {
        version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
        mode: EnvironmentMode::Local,
        backend: EnvironmentBackend::Local,
        image: "local".to_string(),
        init_commands: language_init_commands(info),
        setup_commands: vec![],
        session_commands: vec![],
        env: Default::default(),
        bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
        runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
        resources: Default::default(),
        caches: vec![],
        guard_commands: vec![],
        prepared_image: None,
        source: None,
        dockerfile: None,
    }
}

// ── run ───────────────────────────────────────────────────────────────────────

/// Entry-point for `pb init`.
pub fn run_init(workdir: Option<PathBuf>, backend: Option<EnvironmentBackend>) -> Result<()> {
    let root = super::resolve_env_root(workdir)?;
    println!("Inspecting project at {}…", root.display());

    let info = inspect(&root)?;
    EnvironmentEvidenceDocument {
        version: crate::environment_lock::ENVIRONMENT_EVIDENCE_VERSION,
        evidence: info.environment_evidence.clone(),
    }
    .save_atomic(&root)?;

    // Print what was detected
    print_detection_summary(&info);

    // Existing agent docs
    if !info.existing_agent_docs.is_empty() {
        println!();
        println!("Found existing agent documentation:");
        for p in &info.existing_agent_docs {
            let rel = p
                .strip_prefix(&root)
                .map(|r| r.display().to_string())
                .unwrap_or_else(|_| p.display().to_string());
            println!("  {rel}");
        }
    }

    // Environment config
    println!();
    if info.has_pb_environment {
        println!(
            "Existing environment config found at {}.",
            root.join(".pb").join("environment.toml").display()
        );
        println!("Skipping environment setup (remove .pb/environment.toml to reconfigure).");
    } else {
        let env = match backend {
            Some(EnvironmentBackend::Local) => Some(local_environment(&info)),
            Some(EnvironmentBackend::AppleContainers) => suggest_environment(&info),
            None => suggest_scout_environment(&root, &info),
        };
        match env {
            Some(env) => {
                print_environment_config("Suggested environment", &env);
                env.save(&root)?;
                println!(
                    "Environment config written to {}.",
                    root.join(".pb").join("environment.toml").display()
                );
            }
            None => {
                println!(
                    "No environment could be automatically detected. \
                     Run `pb env pull <image>`, `pb env build`, or `pb env local` to configure one."
                );
            }
        }
    }

    if backend.is_none() {
        write_component_environment_configs(&root, &info)?;
    }

    println!();
    if info.has_pb_workspace {
        println!(
            "Existing workspace graph found at {}.",
            root.join(".pb").join("workspace.toml").display()
        );
    } else {
        let environment = EnvironmentConfig::load(&root)?;
        let graph =
            crate::workspace::WorkspaceGraph::load_or_discover(&root, environment.as_ref())?;
        graph.to_document().save(&root)?;
        println!(
            "Workspace graph written to {} ({} components, {} checks, {} executors).",
            root.join(".pb").join("workspace.toml").display(),
            graph.components.len(),
            graph.checks.len(),
            graph.executors.len()
        );
        for warning in &graph.discovery_warnings {
            println!("  warning: {warning}");
        }
    }

    println!();
    if info.has_pb_workflow {
        let policy = crate::workflow::WorkflowConfigDocument::load(&root)?
            .expect("workflow config exists after inspection")
            .compile()?;
        println!(
            "Existing strict delivery workflow found at {} (policy {}).",
            root.join(".pb").join("workflow.toml").display(),
            &policy.sha256[..12]
        );
    } else {
        crate::workflow::WorkflowConfigDocument::default().save(&root)?;
        println!(
            "Strict delivery workflow written to {}.",
            root.join(".pb").join("workflow.toml").display()
        );
    }

    println!();
    if info.has_pb_goal {
        let policy = crate::goal::GoalConfigDocument::load(&root)?
            .expect("goal config exists after inspection")
            .compile()?;
        println!(
            "Existing durable goal policy found at {} (policy {}).",
            root.join(".pb").join("goal.toml").display(),
            &policy.sha256[..12]
        );
    } else {
        crate::goal::GoalConfigDocument::default().save(&root)?;
        println!(
            "Durable goal policy written to {}.",
            root.join(".pb").join("goal.toml").display()
        );
    }

    Ok(())
}

fn write_component_environment_configs(root: &Path, info: &ProjectInspection) -> Result<()> {
    let components = info
        .environment_evidence
        .iter()
        .map(|evidence| evidence.component.as_str())
        .filter(|component| *component != "repository")
        .collect::<BTreeSet<_>>();
    for component in components {
        let component_root = root.join(component);
        if !is_regular_project_directory(&component_root) {
            continue;
        }
        let component_info = inspect(&component_root)?;
        let Some(environment) = suggest_scout_environment(&component_root, &component_info) else {
            continue;
        };
        let path = root
            .join(".pb")
            .join("environments")
            .join(format!("{component}.toml"));
        if path.exists() {
            continue;
        }
        environment.save_path(&path)?;
        println!(
            "Component environment for '{component}' written to {}.",
            path.display()
        );
    }
    Ok(())
}

fn print_environment_config(label: &str, env: &EnvironmentConfig) {
    let mode_str = match env.mode {
        EnvironmentMode::Pull => "pull",
        EnvironmentMode::Build => "build",
        EnvironmentMode::Local => "local",
    };
    let backend_str = match env.backend {
        EnvironmentBackend::AppleContainers => "apple-containers",
        EnvironmentBackend::Local => "local",
    };
    println!("{label}:");
    println!("  mode:  {mode_str}");
    println!("  backend: {backend_str}");
    if env.backend != EnvironmentBackend::Local {
        println!("  image: {}", env.image);
    }
    if let Some(df) = &env.dockerfile {
        println!("  dockerfile: {}", df.display());
    }
    let setup_commands = env.setup_commands();
    if !setup_commands.is_empty() {
        println!("  setup_commands:");
        for cmd in &setup_commands {
            println!("    - {cmd}");
        }
    }
    if !env.session_commands.is_empty() {
        println!("  session_commands:");
        for cmd in &env.session_commands {
            println!("    - {cmd}");
        }
    }
    if !env.guard_commands.is_empty() {
        println!("  guard_commands:");
        for cmd in &env.guard_commands {
            println!("    - {cmd}");
        }
    }
    if let Some(image) = &env.prepared_image {
        println!("  prepared_image: {image}");
    }
    if let Some(source) = &env.source {
        println!("  source: {source}");
    }
}

fn print_detection_summary(info: &ProjectInspection) {
    println!();
    println!("Detection results:");

    if let Some(img) = &info.devcontainer_image {
        println!("  devcontainer image: {img}");
        if !info.devcontainer_init_commands.is_empty() {
            for cmd in &info.devcontainer_init_commands {
                println!("    init: {cmd}");
            }
        }
    }
    if info.has_dockerfile {
        println!("  Dockerfile: found");
    }
    if let Some(img) = &info.gitlab_ci_image {
        println!("  GitLab CI image: {img}");
    }
    if info.has_github_workflows {
        println!("  GitHub Actions workflows: found");
    }
    if info.has_kubernetes_config {
        println!("  Kubernetes/deployment config: found");
    }
    if info.prefers_local_backend {
        println!("  scout backend signal: local (macOS-specific)");
        for requirement in &info.host_requirements {
            println!("    host requirement: {}", requirement.label());
        }
    }
    if info.prefers_container_backend {
        println!("  scout backend signal: containers (Linux/deployment)");
    }

    let mut langs = Vec::new();
    if info.has_cargo_toml {
        langs.push("Rust (Cargo.toml)");
    }
    if info.has_pyproject_toml {
        langs.push("Python (pyproject.toml)");
    }
    if info.has_requirements_txt {
        langs.push("Python (requirements.txt)");
    }
    if info.has_deno_lock {
        langs.push("Deno (deno.lock)");
    }
    if info.has_package_json {
        langs.push("Node.js (package.json)");
    }
    if info.has_go_mod {
        langs.push("Go (go.mod)");
    }
    if info.has_package_swift {
        langs.push("Swift (Package.swift)");
    }
    if langs.is_empty() {
        println!("  languages: (none detected)");
    } else {
        println!("  languages: {}", langs.join(", "));
    }
    if info.has_image_assets {
        println!(
            "  image assets: found (consider `pb env pull hf://Qwen/Qwen3-VL-MoE-Instruct` \
             for multimodal vision_describe support)"
        );
    }
}

/// Scout an environment without requiring a per-project config file.
pub fn scout_environment(root: &Path) -> Result<Option<EnvironmentConfig>> {
    let info = inspect(root)?;
    Ok(suggest_scout_environment(root, &info))
}

pub fn suggest_scout_environment(
    _root: &Path,
    info: &ProjectInspection,
) -> Option<EnvironmentConfig> {
    let base = suggest_environment(info).or_else(|| {
        if !info.prefers_container_backend && !info.prefers_local_backend {
            return None;
        }
        let container = info.prefers_container_backend;
        Some(EnvironmentConfig {
            version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
            mode: if container {
                EnvironmentMode::Pull
            } else {
                EnvironmentMode::Local
            },
            backend: if container {
                EnvironmentBackend::AppleContainers
            } else {
                EnvironmentBackend::Local
            },
            image: if container {
                "ubuntu:24.04".to_string()
            } else {
                "local".to_string()
            },
            init_commands: vec![],
            setup_commands: vec![],
            session_commands: vec![],
            env: Default::default(),
            bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
            runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
            resources: Default::default(),
            caches: vec![],
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        })
    })?;

    let backend = choose_backend(&base, info);
    let mut setup_commands = unique_commands(info.setup_commands.clone());
    if setup_commands.is_empty() {
        setup_commands = language_init_commands(info);
    }
    if backend == EnvironmentBackend::AppleContainers {
        setup_commands = persistent_container_setup_commands(setup_commands);
    }
    let guard_commands = unique_commands(info.guard_commands.clone());
    let source = scout_source_summary(info, backend);

    let image = if backend == EnvironmentBackend::Local {
        "local".to_string()
    } else {
        base.image.clone()
    };
    Some(EnvironmentConfig {
        version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
        mode: if backend == EnvironmentBackend::Local {
            EnvironmentMode::Local
        } else {
            base.mode
        },
        backend,
        image,
        init_commands: vec![],
        setup_commands,
        session_commands: unique_commands(info.session_commands.clone()),
        env: if backend == EnvironmentBackend::AppleContainers {
            discovered_container_env(info)
        } else {
            Default::default()
        },
        bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
        runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
        resources: Default::default(),
        caches: if backend == EnvironmentBackend::AppleContainers {
            discovered_container_caches(info)
        } else {
            vec![]
        },
        guard_commands,
        prepared_image: None,
        source: Some(source),
        dockerfile: if backend == EnvironmentBackend::Local {
            None
        } else {
            base.dockerfile
        },
    })
}

fn choose_backend(base: &EnvironmentConfig, info: &ProjectInspection) -> EnvironmentBackend {
    // Host requirements are capabilities a Linux VM cannot provide. They always take precedence
    // over Dockerfiles, Linux CI jobs, or deployment manifests for this whole-project resolver.
    if info.prefers_local_backend {
        return EnvironmentBackend::Local;
    }
    if info.prefers_container_backend
        || info.has_dockerfile
        || info.devcontainer_image.is_some()
        || info.gitlab_ci_image.is_some()
        || info.has_kubernetes_config
    {
        return EnvironmentBackend::AppleContainers;
    }
    base.backend
}

fn scout_source_summary(info: &ProjectInspection, backend: EnvironmentBackend) -> String {
    let mut parts = Vec::new();
    let backend_name = match backend {
        EnvironmentBackend::AppleContainers => "container backend",
        EnvironmentBackend::Local => "local backend",
    };
    parts.push(format!("scout selected {backend_name}"));
    if info.prefers_local_backend {
        parts.push("macOS-specific signals found".to_string());
        if !info.host_requirements.is_empty() {
            parts.push(format!(
                "host requirements: {}",
                info.host_requirements
                    .iter()
                    .map(|requirement| requirement.label())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    if info.prefers_container_backend {
        parts.push("Linux/container/deployment signals found".to_string());
    }
    if !info.scout_sources.is_empty() {
        let names: Vec<String> = info
            .scout_sources
            .iter()
            .take(6)
            .map(|p| p.display().to_string())
            .collect();
        parts.push(format!("sources: {}", names.join(", ")));
    }
    parts.join("; ")
}

fn inspect_scout_text(root: &Path, path: &Path, text: &str, info: &mut ProjectInspection) {
    if !info.scout_sources.iter().any(|p| p == path) {
        info.scout_sources.push(path.to_path_buf());
    }
    let lower = text.to_ascii_lowercase();
    if lower.contains("macos-latest") || lower.contains("macos-") {
        record_host_requirement(root, path, info, HostRequirement::MacosCi);
    }
    if lower.contains("xcodebuild") || lower.contains("xcrun") || lower.contains("launchd") {
        record_host_requirement(root, path, info, HostRequirement::XcodeToolchain);
    }
    if lower.contains("simctl") || lower.contains("ios simulator") {
        record_host_requirement(root, path, info, HostRequirement::Simulator);
    }
    if lower.contains("codesign")
        || lower.contains("notarytool")
        || lower.contains("cocoapods")
        || lower.contains("security find-identity")
    {
        record_host_requirement(root, path, info, HostRequirement::CodeSigning);
    }
    if lower.contains("aarch64-apple-darwin")
        || lower.contains("apple sdk")
        || lower.contains("sdkroot")
    {
        record_host_requirement(root, path, info, HostRequirement::AppleSdkTarget);
    }
    if contains_scout_token(&lower, "metal") || contains_scout_token(&lower, "metalkit") {
        record_host_requirement(root, path, info, HostRequirement::Metal);
    }
    if lower.contains("dockerfile")
        || lower.contains("docker build")
        || lower.contains("docker compose")
        || lower.contains("ubuntu")
        || lower.contains("debian")
        || lower.contains("alpine")
        || lower.contains("kubernetes")
        || lower.contains("kubectl")
        || lower.contains("helm")
        || lower.contains("linux")
    {
        record_container_signal(
            root,
            path,
            info,
            "documented Linux/container/deployment workflow",
        );
    }
    for line in text.lines() {
        let command = extract_documented_command(line);
        if let Some(command) = command {
            let normalized = normalize_command(root, path, &command);
            let cmd_lower = normalized.to_ascii_lowercase();
            if is_setup_command(&cmd_lower) {
                record_setup_command(root, path, &normalized, info);
            }
            if is_session_refresh_command(&cmd_lower, line) {
                info.session_commands.push(normalized.clone());
            }
            if is_guard_command(&cmd_lower, line) {
                if !info
                    .documented_guard_commands
                    .iter()
                    .any(|existing| existing == &normalized)
                {
                    info.documented_guard_commands.push(normalized.clone());
                }
                record_guard_command(root, path, &normalized, info);
            }
        }
    }
}

fn contains_scout_token(text: &str, expected: &str) -> bool {
    text.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .any(|token| token == expected)
}

fn extract_documented_command(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let mut candidate = trimmed
        .trim_start_matches(['-', '*'])
        .trim()
        .trim_start_matches("run ")
        .trim();
    if let Some(start) = candidate.find('`') {
        let rest = &candidate[start + 1..];
        if let Some(end) = rest.find('`') {
            candidate = &rest[..end];
        }
    } else if let Some(rest) = candidate.strip_prefix('$') {
        candidate = rest.trim();
    } else if candidate.contains(':') {
        candidate = candidate.split_once(':')?.1.trim();
    }
    let first = candidate.split_whitespace().next()?;
    const COMMAND_PREFIXES: &[&str] = &[
        "cargo",
        "npm",
        "pnpm",
        "yarn",
        "deno",
        "pip",
        "python",
        "uv",
        "poetry",
        "go",
        "make",
        "docker",
        "docker-compose",
        "xcodebuild",
        "swift",
    ];
    if COMMAND_PREFIXES.contains(&first) {
        Some(candidate.to_string())
    } else {
        None
    }
}

fn normalize_command(root: &Path, source: &Path, command: &str) -> String {
    let parent = source.parent().unwrap_or(root);
    if parent == root || command.starts_with("cd ") {
        return command.to_string();
    }
    if let Ok(rel) = parent.strip_prefix(root)
        && !rel.as_os_str().is_empty()
    {
        return format!("cd {} && {command}", shell_escape_path(rel));
    }
    command.to_string()
}

fn shell_escape_path(path: &Path) -> String {
    let s = path.to_string_lossy();
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-'))
    {
        s.into_owned()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

fn is_setup_command(command: &str) -> bool {
    command.contains("npm ci")
        || command.contains("npm install")
        || command.contains("pnpm install")
        || command.contains("yarn install")
        || command.contains("deno install")
        || command.contains("pip install")
        || command.contains("poetry install")
        || command.contains("uv sync")
        || command.contains("go mod download")
        || command.contains("cargo fetch")
        || command.contains("make deps")
        || command.contains("make setup")
}

fn is_session_refresh_command(command: &str, line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    (lower.contains("per session")
        || lower.contains("each session")
        || lower.contains("before every"))
        && (is_setup_command(command)
            || command.contains("generate")
            || command.contains("codegen"))
}

fn is_guard_command(command: &str, line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    command.contains(" test")
        || command.starts_with("cargo test")
        || command.contains(" check")
        || command.contains(" lint")
        || command.contains(" fmt")
        || command.contains(" build")
        || lower.contains("before commit")
        || lower.contains("required before")
        || lower.contains("guard")
}

fn unique_commands(commands: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for command in commands {
        if !out.contains(&command) {
            out.push(command);
        }
    }
    out
}

// ── devcontainer parsing ──────────────────────────────────────────────────────

/// Very lightweight devcontainer.json parser — we only care about `image` and
/// `postCreateCommand`. A full JSON parse is used so we're robust to comments
/// stripped by serde_json (devcontainer files may contain JSONC).
fn parse_devcontainer_json(text: &str) -> Option<(String, Vec<String>)> {
    // Strip single-line // comments before parsing as JSON
    let stripped = strip_jsonc_comments(text);
    let v: serde_json::Value = serde_json::from_str(&stripped).ok()?;

    let image = v.get("image").and_then(|i| i.as_str()).map(str::to_owned);
    let image = image?;

    let mut inits = Vec::new();
    if let Some(cmd) = v.get("postCreateCommand") {
        match cmd {
            serde_json::Value::String(s) => inits.push(s.clone()),
            serde_json::Value::Array(arr) => {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        inits.push(s.to_owned());
                    }
                }
            }
            _ => {}
        }
    }

    Some((image, inits))
}

/// Strip `// …` single-line comments from JSONC text.
fn strip_jsonc_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_string {
            out.push(ch);
            if ch == '\\' {
                // consume and pass through the escaped character — handles
                // all sequences: `\"`, `\\`, `\/`, etc. correctly including
                // `\\"` where the second `"` ends the string.
                if let Some(next) = chars.next() {
                    out.push(next);
                }
            } else if ch == '"' {
                in_string = false;
            }
        } else {
            match ch {
                '"' => {
                    in_string = true;
                    out.push(ch);
                }
                '/' => {
                    if chars.peek() == Some(&'/') {
                        // consume rest of line
                        for c in chars.by_ref() {
                            if c == '\n' {
                                out.push('\n');
                                break;
                            }
                        }
                    } else {
                        out.push(ch);
                    }
                }
                _ => out.push(ch),
            }
        }
    }
    out
}

// ── GitLab CI parsing ─────────────────────────────────────────────────────────

/// Extract a global `image:` value from a `.gitlab-ci.yml` file.
/// Uses a simple line scan — we don't pull in a full YAML parser.
fn parse_gitlab_ci_image(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("image:") {
            let value = rest.trim().trim_matches('"').trim_matches('\'').to_owned();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── helpers ──

    fn write(dir: &Path, rel: &str, content: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    // ── strip_jsonc_comments ──

    #[test]
    fn strip_jsonc_removes_line_comments() {
        let input = r#"{ // comment
  "image": "foo" // trailing
}"#;
        let stripped = strip_jsonc_comments(input);
        let v: serde_json::Value = serde_json::from_str(&stripped).unwrap();
        assert_eq!(v["image"], "foo");
    }

    #[test]
    fn strip_jsonc_handles_escaped_backslash_before_quote() {
        // "path" value contains `\\` (escaped backslash); the following `"`
        // must be treated as closing the string, not as an escaped quote.
        let input = r#"{ "path": "C:\\\\foo", "image": "bar" }"#;
        let stripped = strip_jsonc_comments(input);
        let v: serde_json::Value = serde_json::from_str(&stripped).unwrap();
        assert_eq!(v["image"], "bar");
    }

    // ── devcontainer parsing ──

    #[test]
    fn parse_devcontainer_basic() {
        let json = r#"{ "image": "mcr.microsoft.com/devcontainers/rust:latest" }"#;
        let (image, inits) = parse_devcontainer_json(json).unwrap();
        assert_eq!(image, "mcr.microsoft.com/devcontainers/rust:latest");
        assert!(inits.is_empty());
    }

    #[test]
    fn parse_devcontainer_with_post_create_string() {
        let json = r#"{
  "image": "node:lts",
  "postCreateCommand": "npm ci"
}"#;
        let (image, inits) = parse_devcontainer_json(json).unwrap();
        assert_eq!(image, "node:lts");
        assert_eq!(inits, vec!["npm ci"]);
    }

    #[test]
    fn parse_devcontainer_with_post_create_array() {
        let json = r#"{
  "image": "python:3",
  "postCreateCommand": ["pip install -e .", "pytest --collect-only"]
}"#;
        let (_, inits) = parse_devcontainer_json(json).unwrap();
        assert_eq!(inits, vec!["pip install -e .", "pytest --collect-only"]);
    }

    #[test]
    fn parse_devcontainer_no_image_returns_none() {
        let json = r#"{ "name": "My Container" }"#;
        assert!(parse_devcontainer_json(json).is_none());
    }

    // ── GitLab CI parsing ──

    #[test]
    fn parse_gitlab_ci_extracts_global_image() {
        let yaml = "image: ubuntu:22.04\n\njobs:\n  test:\n    script: cargo test\n";
        assert_eq!(
            parse_gitlab_ci_image(yaml),
            Some("ubuntu:22.04".to_string())
        );
    }

    #[test]
    fn parse_gitlab_ci_no_image_returns_none() {
        let yaml = "stages:\n  - test\n";
        assert!(parse_gitlab_ci_image(yaml).is_none());
    }

    // ── inspect ──

    #[test]
    fn inspect_empty_dir() {
        let dir = TempDir::new().unwrap();
        let info = inspect(dir.path()).unwrap();
        assert!(!info.has_cargo_toml);
        assert!(!info.has_pyproject_toml);
        assert!(!info.has_package_json);
        assert!(!info.has_deno_lock);
        assert!(!info.has_go_mod);
        assert!(info.devcontainer_image.is_none());
        assert!(!info.has_dockerfile);
        assert!(info.gitlab_ci_image.is_none());
        assert!(info.existing_agent_docs.is_empty());
        assert!(!info.has_pb_environment);
        assert!(!info.has_pb_workspace);
        assert!(!info.has_pb_workflow);
        assert!(!info.has_pb_goal);
    }

    #[test]
    fn inspect_rust_project() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"foo\"\n");
        let info = inspect(dir.path()).unwrap();
        assert!(info.has_cargo_toml);
        assert!(!info.has_pyproject_toml);
        assert!(info.environment_evidence.iter().any(|evidence| matches!(
            &evidence.requirement,
            EnvironmentRequirement::SetupCommand { command }
                if command.contains("cargo fetch")
        )));
        assert!(info.environment_evidence.iter().any(|evidence| matches!(
            &evidence.requirement,
            EnvironmentRequirement::GuardCommand { command } if command == "cargo test"
        )));
    }

    #[test]
    fn inspect_records_toolchain_managers_as_typed_lock_inputs() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "flake.nix", "{ outputs = inputs: {}; }");
        write(
            dir.path(),
            ".tool-versions",
            "nodejs 22.3.0\npython 3.12.4\n",
        );
        write(
            dir.path(),
            "rust-toolchain.toml",
            "[toolchain]\nchannel = \"1.88\"\n",
        );
        let info = inspect(dir.path()).unwrap();
        for path in ["flake.nix", ".tool-versions", "rust-toolchain.toml"] {
            assert!(info.dependency_key_files.contains(&PathBuf::from(path)));
        }
        assert!(info.environment_evidence.iter().any(|evidence| matches!(
            &evidence.requirement,
            EnvironmentRequirement::Toolchain { name, .. } if name == "nix"
        )));
        assert!(info.environment_evidence.iter().any(|evidence| matches!(
            &evidence.requirement,
            EnvironmentRequirement::Toolchain { name, .. } if name == "asdf"
        )));
        assert!(info.environment_evidence.iter().any(|evidence| matches!(
            &evidence.requirement,
            EnvironmentRequirement::Toolchain { name, .. } if name == "rust"
        )));
    }

    #[test]
    fn inspect_node_project() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "package.json", "{}");
        let info = inspect(dir.path()).unwrap();
        assert!(info.has_package_json);
        assert!(!info.has_deno_lock);
    }

    #[test]
    fn inspect_deno_project() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "deno.lock", "{}");
        write(dir.path(), "package.json", "{}");
        let info = inspect(dir.path()).unwrap();
        assert!(info.has_deno_lock);
        assert!(info.has_package_json);
    }

    #[cfg(unix)]
    #[test]
    fn inspect_does_not_follow_repository_symlinks() {
        use std::os::unix::fs::symlink;

        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        write(outside.path(), "package.json", "{}");
        symlink(outside.path(), dir.path().join("linked-outside")).unwrap();

        let info = inspect(dir.path()).unwrap();
        assert!(!info.has_package_json);
        assert!(info.environment_evidence.is_empty());
    }

    #[test]
    fn inspect_devcontainer() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            ".devcontainer/devcontainer.json",
            r#"{ "image": "mcr.microsoft.com/devcontainers/rust:latest" }"#,
        );
        let info = inspect(dir.path()).unwrap();
        assert_eq!(
            info.devcontainer_image,
            Some("mcr.microsoft.com/devcontainers/rust:latest".to_string())
        );
        assert!(info.environment_evidence.iter().any(|evidence| matches!(
            &evidence.requirement,
            EnvironmentRequirement::ContainerSignal { detail }
                if detail == "devcontainer configuration"
        )));
    }

    #[test]
    fn inspect_gitlab_ci() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), ".gitlab-ci.yml", "image: ubuntu:22.04\n");
        let info = inspect(dir.path()).unwrap();
        assert_eq!(info.gitlab_ci_image, Some("ubuntu:22.04".to_string()));
    }

    #[test]
    fn inspect_agent_docs() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), ".github/copilot-instructions.md", "# hello");
        write(dir.path(), "CLAUDE.md", "# hello");
        let info = inspect(dir.path()).unwrap();
        assert_eq!(info.existing_agent_docs.len(), 2);
    }

    #[test]
    fn inspect_existing_pb_environment() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            ".pb/environment.toml",
            "mode = \"pull\"\nimage = \"rust:latest\"\ninit_commands = []\n",
        );
        let info = inspect(dir.path()).unwrap();
        assert!(info.has_pb_environment);
    }

    #[test]
    fn inspect_existing_pb_workspace() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), ".pb/workspace.toml", "version = 1\n");
        let info = inspect(dir.path()).unwrap();
        assert!(info.has_pb_workspace);
    }

    #[test]
    fn inspect_existing_pb_workflow() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), ".pb/workflow.toml", "version = 1\n");
        let info = inspect(dir.path()).unwrap();
        assert!(info.has_pb_workflow);
    }

    #[test]
    fn inspect_existing_pb_goal() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), ".pb/goal.toml", "version = 1\n");
        let info = inspect(dir.path()).unwrap();
        assert!(info.has_pb_goal);
    }

    #[test]
    fn init_writes_normalized_workspace_graph_without_overwriting_environment_schema() {
        let dir = TempDir::new().unwrap();
        run_init(
            Some(dir.path().to_path_buf()),
            Some(EnvironmentBackend::Local),
        )
        .unwrap();

        assert!(dir.path().join(".pb/environment.toml").is_file());
        assert!(dir.path().join(".pb/workspace.toml").is_file());
        assert!(dir.path().join(".pb/workflow.toml").is_file());
        assert!(dir.path().join(".pb/goal.toml").is_file());
        let graph = crate::workspace::WorkspaceConfigDocument::load(dir.path())
            .unwrap()
            .unwrap()
            .normalize()
            .unwrap();
        assert_eq!(graph.components.len(), 1);
        assert_eq!(graph.components["repository"].root, ".");
        let policy = crate::workflow::WorkflowConfigDocument::load(dir.path())
            .unwrap()
            .unwrap()
            .compile()
            .unwrap();
        assert_eq!(policy.delivery, crate::workflow::DeliveryPolicy::Strict);
    }

    #[test]
    fn init_writes_scoped_environments_for_mixed_apple_and_linux_components() {
        let directory = TempDir::new().unwrap();
        write(
            directory.path(),
            "client/App.xcodeproj/project.pbxproj",
            "// !$*UTF8*$!\n",
        );
        write(
            directory.path(),
            "server/Cargo.toml",
            "[package]\nname = \"server\"\nversion = \"0.1.0\"\n",
        );

        run_init(Some(directory.path().to_path_buf()), None).unwrap();

        let repository = EnvironmentConfig::load(directory.path()).unwrap().unwrap();
        assert_eq!(repository.backend, EnvironmentBackend::Local);
        let client =
            EnvironmentConfig::load_path(&directory.path().join(".pb/environments/client.toml"))
                .unwrap();
        let server =
            EnvironmentConfig::load_path(&directory.path().join(".pb/environments/server.toml"))
                .unwrap();
        assert_eq!(client.backend, EnvironmentBackend::Local);
        assert_eq!(server.backend, EnvironmentBackend::AppleContainers);
        assert_eq!(server.image, "rust:latest");
    }

    // ── suggest_environment ──

    #[test]
    fn suggest_rust_project() {
        let info = ProjectInspection {
            has_cargo_toml: true,
            ..Default::default()
        };
        let env = suggest_environment(&info).unwrap();
        assert_eq!(env.image, "rust:latest");
        assert!(matches!(env.mode, EnvironmentMode::Pull));
        assert!(
            env.caches
                .iter()
                .any(|cache| cache.target == Path::new("/workspace/target"))
        );
        assert!(
            env.caches
                .iter()
                .all(|cache| cache.target != Path::new("/usr/local/rustup")),
            "an empty Apple named volume must not hide an image-owned Rust toolchain"
        );
    }

    #[test]
    fn suggest_python_pyproject() {
        let info = ProjectInspection {
            has_pyproject_toml: true,
            ..Default::default()
        };
        let env = suggest_environment(&info).unwrap();
        assert_eq!(env.image, "python:3-slim");
        assert_eq!(
            env.init_commands,
            vec![
                "test -x /opt/pb-venv/bin/python || python -m venv /opt/pb-venv",
                "/opt/pb-venv/bin/python -m pip install -e .",
            ]
        );
        assert_eq!(
            env.env.get("VIRTUAL_ENV").map(String::as_str),
            Some("/opt/pb-venv")
        );
        assert_eq!(env.caches[0].id, "python-venv");
    }

    #[test]
    fn suggest_python_requirements_txt() {
        let info = ProjectInspection {
            has_requirements_txt: true,
            ..Default::default()
        };
        let env = suggest_environment(&info).unwrap();
        assert_eq!(env.image, "python:3-slim");
        assert_eq!(
            env.init_commands,
            vec![
                "test -x /opt/pb-venv/bin/python || python -m venv /opt/pb-venv",
                "/opt/pb-venv/bin/python -m pip install -r requirements.txt",
            ]
        );
        assert_eq!(env.caches[0].target, PathBuf::from("/opt/pb-venv"));
    }

    #[test]
    fn suggest_deno_over_node() {
        let info = ProjectInspection {
            has_deno_lock: true,
            has_package_json: true,
            ..Default::default()
        };
        let env = suggest_environment(&info).unwrap();
        assert_eq!(env.image, "denoland/deno:latest");
    }

    #[test]
    fn suggest_node_project() {
        let info = ProjectInspection {
            has_package_json: true,
            ..Default::default()
        };
        let env = suggest_environment(&info).unwrap();
        assert_eq!(env.image, "node:lts-slim");
        assert_eq!(env.init_commands, vec!["npm ci"]);
    }

    #[test]
    fn suggest_go_project() {
        let info = ProjectInspection {
            has_go_mod: true,
            ..Default::default()
        };
        let env = suggest_environment(&info).unwrap();
        assert_eq!(env.image, "golang:latest");
        assert_eq!(env.init_commands, vec!["go mod download"]);
    }

    #[test]
    fn suggest_dockerfile_beats_language() {
        let info = ProjectInspection {
            has_dockerfile: true,
            has_cargo_toml: true,
            ..Default::default()
        };
        let env = suggest_environment(&info).unwrap();
        assert!(matches!(env.mode, EnvironmentMode::Build));
        assert_eq!(env.dockerfile, Some(PathBuf::from("Dockerfile")));
    }

    #[test]
    fn local_environment_forces_local_backend_with_language_init() {
        let info = ProjectInspection {
            has_package_json: true,
            ..Default::default()
        };
        let env = local_environment(&info);
        assert_eq!(env.mode, EnvironmentMode::Local);
        assert_eq!(env.backend, EnvironmentBackend::Local);
        assert_eq!(env.image, "local");
        assert_eq!(env.init_commands, vec!["npm ci"]);
    }

    #[test]
    fn suggest_devcontainer_beats_dockerfile() {
        let info = ProjectInspection {
            devcontainer_image: Some("ghcr.io/myorg/dev:latest".to_string()),
            devcontainer_init_commands: vec!["npm ci".to_string()],
            has_dockerfile: true,
            has_cargo_toml: true,
            ..Default::default()
        };
        let env = suggest_environment(&info).unwrap();
        assert_eq!(env.image, "ghcr.io/myorg/dev:latest");
        assert!(matches!(env.mode, EnvironmentMode::Pull));
        assert_eq!(env.init_commands, vec!["npm ci"]);
    }

    #[test]
    fn scout_prefers_local_for_macos_specific_docs() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            "AGENT.md",
            "Use `xcodebuild test` before commit on macos-latest.",
        );
        let info = inspect(dir.path()).unwrap();
        let env = suggest_scout_environment(dir.path(), &info).unwrap();
        assert_eq!(env.backend, EnvironmentBackend::Local);
        assert_eq!(env.guard_commands, vec!["xcodebuild test"]);
    }

    #[test]
    fn scout_prefers_containers_without_mutable_prepared_image() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "Dockerfile", "FROM node:lts\n");
        write(
            dir.path(),
            "README.md",
            "Run `npm ci`
Run `npm test` before commit.",
        );
        let info = inspect(dir.path()).unwrap();
        let env = suggest_scout_environment(dir.path(), &info).unwrap();
        assert_eq!(env.backend, EnvironmentBackend::AppleContainers);
        assert!(env.prepared_image.is_none());
        assert_eq!(env.setup_commands, vec!["npm ci"]);
        assert_eq!(env.guard_commands, vec!["npm test"]);
    }

    #[test]
    fn xcode_project_overrides_container_and_ci_signals() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "Dockerfile", "FROM swift:latest\n");
        write(
            dir.path(),
            ".github/workflows/build.yml",
            "jobs:\n  build:\n    runs-on: ubuntu-latest\n",
        );
        write(
            dir.path(),
            "Example.xcodeproj/project.pbxproj",
            "// !$*UTF8*$!\n",
        );

        let info = inspect(dir.path()).unwrap();
        let env = suggest_scout_environment(dir.path(), &info).unwrap();

        assert!(
            info.host_requirements
                .contains(&HostRequirement::XcodeProject)
        );
        assert_eq!(env.backend, EnvironmentBackend::Local);
        assert_eq!(env.image, "local");
    }

    #[test]
    fn generic_github_workflow_is_not_a_host_or_container_requirement() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            ".github/workflows/checks.yml",
            "name: checks\non: push\njobs: {}\n",
        );

        let info = inspect(dir.path()).unwrap();
        assert!(info.has_github_workflows);
        assert!(!info.prefers_container_backend);
        assert!(!info.prefers_local_backend);
        assert!(suggest_scout_environment(dir.path(), &info).is_none());
    }

    #[test]
    fn metadata_prose_does_not_require_metal_or_host_execution() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            "README.md",
            "The generated metadata is checked into the repository.\n",
        );

        let info = inspect(dir.path()).unwrap();

        assert!(!info.host_requirements.contains(&HostRequirement::Metal));
        assert!(!info.prefers_local_backend);
        assert!(suggest_scout_environment(dir.path(), &info).is_none());
    }

    #[test]
    fn plain_swift_package_can_use_linux_container() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            "Package.swift",
            "// swift-tools-version: 6.0\nimport PackageDescription\n",
        );

        let info = inspect(dir.path()).unwrap();
        let env = suggest_scout_environment(dir.path(), &info).unwrap();

        assert!(info.has_package_swift);
        assert!(info.host_requirements.is_empty());
        assert_eq!(env.backend, EnvironmentBackend::AppleContainers);
        assert_eq!(env.image, "swift:latest");
    }

    #[test]
    fn apple_platform_swift_package_requires_host() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            "Package.swift",
            "// swift-tools-version: 6.0\nimport PackageDescription\nlet package = Package(platforms: [.macOS(.v15)])\n",
        );

        let info = inspect(dir.path()).unwrap();
        let env = suggest_scout_environment(dir.path(), &info).unwrap();

        assert!(
            info.host_requirements
                .contains(&HostRequirement::AppleSwiftPackage)
        );
        assert_eq!(env.backend, EnvironmentBackend::Local);
    }

    #[test]
    fn apple_only_framework_sources_require_host_without_an_xcode_project() {
        let swift = TempDir::new().unwrap();
        write(
            swift.path(),
            "Package.swift",
            "// swift-tools-version: 6.0\nimport PackageDescription\n",
        );
        write(
            swift.path(),
            "Sources/VM/Runner.swift",
            "@preconcurrency import Virtualization\n",
        );
        let swift_info = inspect(swift.path()).unwrap();
        assert!(
            swift_info
                .host_requirements
                .contains(&HostRequirement::AppleFrameworkSource)
        );
        assert_eq!(
            suggest_scout_environment(swift.path(), &swift_info)
                .unwrap()
                .backend,
            EnvironmentBackend::Local
        );

        let objc = TempDir::new().unwrap();
        write(
            objc.path(),
            "client/main.m",
            "#import <AppKit/AppKit.h>\nint main(void) { return 0; }\n",
        );
        let objc_info = inspect(objc.path()).unwrap();
        assert!(
            objc_info
                .host_requirements
                .contains(&HostRequirement::AppleFrameworkSource)
        );
    }

    #[test]
    fn scout_detects_subproject_setup_commands() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            "crates/api/Cargo.toml",
            "[package]\nname = \"api\"\n",
        );
        write(dir.path(), "web/package.json", "{}");
        let info = inspect(dir.path()).unwrap();
        assert!(info.has_cargo_toml);
        assert!(info.has_package_json);
        let env = suggest_scout_environment(dir.path(), &info).unwrap();
        assert!(env.setup_commands.contains(&"cd web && npm ci".to_string()));
        assert!(
            env.guard_commands
                .contains(&"cd crates/api && cargo test".to_string())
        );
    }

    #[test]
    fn suggest_returns_none_for_empty_project() {
        let info = ProjectInspection::default();
        assert!(suggest_environment(&info).is_none());
    }
}
