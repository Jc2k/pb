//! `pb init` — inspect a project and configure it for use with pb.
//!
//! The command walks the project root and detects:
//! - Container environments (devcontainer, Dockerfile, GitLab CI image)
//! - Language ecosystems (Rust, Python, Node/Deno, Go)
//! - Existing agent / coding-assistant documentation
//!
//! Based on the detections it writes (or confirms) a `.pb/environment.toml` and
//! prints a summary of what it found and what it configured.

use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::environment::{EnvironmentConfig, EnvironmentMode};

// ── detection results ────────────────────────────────────────────────────────

/// Everything we learn from inspecting the project root.
#[derive(Debug, Default)]
pub struct ProjectInspection {
    // Container / CI
    pub devcontainer_image: Option<String>,
    pub devcontainer_init_commands: Vec<String>,
    pub has_dockerfile: bool,
    pub gitlab_ci_image: Option<String>,

    // Language ecosystems
    pub has_cargo_toml: bool,
    pub has_pyproject_toml: bool,
    pub has_requirements_txt: bool,
    pub has_package_json: bool,
    pub has_deno_lock: bool,
    pub has_go_mod: bool,

    // Existing agent docs
    pub existing_agent_docs: Vec<PathBuf>,

    // Already configured?
    pub has_pb_environment: bool,
}

/// Inspect a project directory and return everything we detected.
pub fn inspect(root: &Path) -> Result<ProjectInspection> {
    let mut info = ProjectInspection::default();

    // --- devcontainer ---
    let dc_json = root.join(".devcontainer").join("devcontainer.json");
    let dc_json_top = root.join("devcontainer.json");
    for dc_path in [&dc_json, &dc_json_top] {
        if dc_path.exists() {
            if let Ok(text) = std::fs::read_to_string(dc_path) {
                if let Some((image, inits)) = parse_devcontainer_json(&text) {
                    info.devcontainer_image = Some(image);
                    info.devcontainer_init_commands = inits;
                }
            }
            break;
        }
    }

    // --- Dockerfile ---
    if root.join("Dockerfile").exists() {
        info.has_dockerfile = true;
    }

    // --- GitLab CI ---
    let gitlab_ci = root.join(".gitlab-ci.yml");
    if gitlab_ci.exists() {
        if let Ok(text) = std::fs::read_to_string(&gitlab_ci) {
            info.gitlab_ci_image = parse_gitlab_ci_image(&text);
        }
    }

    // --- Language ecosystems ---
    info.has_cargo_toml = root.join("Cargo.toml").exists();
    info.has_pyproject_toml = root.join("pyproject.toml").exists();
    info.has_requirements_txt = root.join("requirements.txt").exists();
    info.has_package_json = root.join("package.json").exists();
    info.has_deno_lock = root.join("deno.lock").exists();
    info.has_go_mod = root.join("go.mod").exists();

    // --- Existing agent docs ---
    let agent_doc_candidates = [
        ".github/copilot-instructions.md",
        "CLAUDE.md",
        "AGENTS.md",
        ".cursor/rules",
        "GEMINI.md",
        ".aider.conf.yml",
        ".continue/config.json",
        "cline_docs/systemPrompt.md",
    ];
    for rel in &agent_doc_candidates {
        let p = root.join(rel);
        if p.exists() {
            info.existing_agent_docs.push(p);
        }
    }

    // --- Already configured? ---
    info.has_pb_environment = root.join(".pb").join("environment.toml").exists();

    Ok(info)
}

// ── environment suggestion ────────────────────────────────────────────────────

/// Suggest an [`EnvironmentConfig`] based on the project inspection.
/// Returns `None` when nothing useful could be determined.
pub fn suggest_environment(info: &ProjectInspection) -> Option<EnvironmentConfig> {
    // 1. devcontainer takes highest priority
    if let Some(image) = &info.devcontainer_image {
        return Some(EnvironmentConfig {
            mode: EnvironmentMode::Pull,
            image: image.clone(),
            init_commands: info.devcontainer_init_commands.clone(),
            dockerfile: None,
        });
    }

    // 2. Dockerfile in project root
    if info.has_dockerfile {
        return Some(EnvironmentConfig {
            mode: EnvironmentMode::Build,
            image: "pb-dev:latest".to_string(),
            init_commands: vec![],
            dockerfile: Some(PathBuf::from("Dockerfile")),
        });
    }

    // 3. GitLab CI image
    if let Some(image) = &info.gitlab_ci_image {
        let init_commands = language_init_commands(info);
        return Some(EnvironmentConfig {
            mode: EnvironmentMode::Pull,
            image: image.clone(),
            init_commands,
            dockerfile: None,
        });
    }

    // 4. Language-based well-known images
    if info.has_cargo_toml {
        return Some(EnvironmentConfig {
            mode: EnvironmentMode::Pull,
            image: "rust:latest".to_string(),
            init_commands: vec![],
            dockerfile: None,
        });
    }
    if info.has_pyproject_toml || info.has_requirements_txt {
        let init_commands = if info.has_pyproject_toml {
            vec!["pip install -e .".to_string()]
        } else {
            vec!["pip install -r requirements.txt".to_string()]
        };
        return Some(EnvironmentConfig {
            mode: EnvironmentMode::Pull,
            image: "python:3-slim".to_string(),
            init_commands,
            dockerfile: None,
        });
    }
    if info.has_deno_lock {
        return Some(EnvironmentConfig {
            mode: EnvironmentMode::Pull,
            image: "denoland/deno:latest".to_string(),
            init_commands: vec!["deno install".to_string()],
            dockerfile: None,
        });
    }
    if info.has_package_json {
        return Some(EnvironmentConfig {
            mode: EnvironmentMode::Pull,
            image: "node:lts-slim".to_string(),
            init_commands: vec!["npm ci".to_string()],
            dockerfile: None,
        });
    }
    if info.has_go_mod {
        return Some(EnvironmentConfig {
            mode: EnvironmentMode::Pull,
            image: "golang:latest".to_string(),
            init_commands: vec!["go mod download".to_string()],
            dockerfile: None,
        });
    }

    None
}

/// Derive language-level init commands (used when an image is already known
/// from CI or devcontainer but we still need to install dependencies).
fn language_init_commands(info: &ProjectInspection) -> Vec<String> {
    let mut cmds = Vec::new();
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
    cmds
}

// ── run ───────────────────────────────────────────────────────────────────────

/// Entry-point for `pb init`.
pub fn run_init(workdir: Option<PathBuf>) -> Result<()> {
    let root = super::resolve_env_root(workdir)?;
    println!("Inspecting project at {}…", root.display());

    let info = inspect(&root)?;

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
        match suggest_environment(&info) {
            Some(env) => {
                let mode_str = match env.mode {
                    EnvironmentMode::Pull => "pull",
                    EnvironmentMode::Build => "build",
                };
                println!("Suggested environment:");
                println!("  mode:  {mode_str}");
                println!("  image: {}", env.image);
                if let Some(df) = &env.dockerfile {
                    println!("  dockerfile: {}", df.display());
                }
                if !env.init_commands.is_empty() {
                    println!("  init_commands:");
                    for cmd in &env.init_commands {
                        println!("    - {cmd}");
                    }
                }
                env.save(&root)?;
                println!(
                    "Environment config written to {}.",
                    root.join(".pb").join("environment.toml").display()
                );
            }
            None => {
                println!(
                    "No environment could be automatically detected. \
                     Run `pb env pull <image>` or `pb env build` to configure one."
                );
            }
        }
    }

    Ok(())
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
    if langs.is_empty() {
        println!("  languages: (none detected)");
    } else {
        println!("  languages: {}", langs.join(", "));
    }
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
    }

    #[test]
    fn inspect_rust_project() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"foo\"\n");
        let info = inspect(dir.path()).unwrap();
        assert!(info.has_cargo_toml);
        assert!(!info.has_pyproject_toml);
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
    }

    #[test]
    fn suggest_python_pyproject() {
        let info = ProjectInspection {
            has_pyproject_toml: true,
            ..Default::default()
        };
        let env = suggest_environment(&info).unwrap();
        assert_eq!(env.image, "python:3-slim");
        assert_eq!(env.init_commands, vec!["pip install -e ."]);
    }

    #[test]
    fn suggest_python_requirements_txt() {
        let info = ProjectInspection {
            has_requirements_txt: true,
            ..Default::default()
        };
        let env = suggest_environment(&info).unwrap();
        assert_eq!(env.image, "python:3-slim");
        assert_eq!(env.init_commands, vec!["pip install -r requirements.txt"]);
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
    fn suggest_returns_none_for_empty_project() {
        let info = ProjectInspection::default();
        assert!(suggest_environment(&info).is_none());
    }
}
