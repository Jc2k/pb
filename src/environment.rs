use anyhow::{Context, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum EnvironmentMode {
    #[default]
    Pull,
    Build,
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "kebab-case")]
#[derive(Default)]
pub enum EnvironmentBackend {
    /// Execute commands in an Apple/container-backed project environment.
    #[default]
    AppleContainers,
    /// Execute commands directly on the host from the project root.
    Local,
}

/// Project environment configuration stored at `.pb/environment.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    /// Whether the image was pulled from a registry (`pull`), built locally (`build`), or run on the host (`local`).
    #[serde(default)]
    pub mode: EnvironmentMode,

    /// Execution backend for running project commands. Defaults to Apple containers.
    #[serde(default)]
    pub backend: EnvironmentBackend,

    /// Container image reference (e.g. `ghcr.io/myorg/dev:latest` or a locally built tag).
    pub image: String,

    /// Legacy setup commands. Kept for compatibility with existing `.pb/environment.toml` files.
    /// New scouted environments store one-time dependency installation in `setup_commands`.
    #[serde(default)]
    pub init_commands: Vec<String>,

    /// Commands that prepare a reusable development environment image.
    /// Container backends run these once, commit the result, and reuse the tagged image.
    /// Local backends only run them when the agent determines an environment refresh is needed.
    #[serde(default)]
    pub setup_commands: Vec<String>,

    /// Commands documented as per-session refresh steps. Most projects leave this empty.
    #[serde(default)]
    pub session_commands: Vec<String>,

    /// Commands that should pass before committing changes.
    #[serde(default)]
    pub guard_commands: Vec<String>,

    /// Image tag used for a prepared scout environment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prepared_image: Option<String>,

    /// Human-readable source/reason for the selected backend and commands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Path to the Dockerfile used for `build` mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dockerfile: Option<PathBuf>,
}

impl EnvironmentConfig {
    /// One-time setup commands, including legacy `init_commands` for old configs.
    pub fn setup_commands(&self) -> Vec<String> {
        let mut commands = self.init_commands.clone();
        commands.extend(self.setup_commands.clone());
        commands
    }

    /// Commands that should run for every fresh agent session.
    pub fn session_commands(&self) -> &[String] {
        &self.session_commands
    }
}

impl EnvironmentConfig {
    /// Load the project environment config from `<workspace_root>/.pb/environment.toml`.
    /// Returns `Ok(None)` when no config file exists.
    pub fn load(workspace_root: &Path) -> Result<Option<Self>> {
        let path = workspace_root.join(".pb").join("environment.toml");
        if !path.exists() {
            return Ok(None);
        }
        Self::load_path(&path).map(Some)
    }

    pub fn load_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
    }

    /// Persist the config to `<workspace_root>/.pb/environment.toml`, creating directories as needed.
    pub fn save(&self, workspace_root: &Path) -> Result<()> {
        let dir = workspace_root.join(".pb");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create directory {}", dir.display()))?;
        let path = dir.join("environment.toml");
        let text =
            toml::to_string_pretty(self).context("failed to serialize environment config")?;
        std::fs::write(&path, text)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trip_pull_config() {
        let dir = TempDir::new().unwrap();
        let config = EnvironmentConfig {
            mode: EnvironmentMode::Pull,
            backend: EnvironmentBackend::AppleContainers,
            image: "ghcr.io/example/dev:latest".to_string(),
            init_commands: vec!["npm ci".to_string()],
            setup_commands: vec![],
            session_commands: vec![],
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        };
        config.save(dir.path()).unwrap();
        let loaded = EnvironmentConfig::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.mode, EnvironmentMode::Pull);
        assert_eq!(loaded.image, "ghcr.io/example/dev:latest");
        assert_eq!(loaded.init_commands, vec!["npm ci"]);
        assert!(loaded.dockerfile.is_none());
    }

    #[test]
    fn round_trip_build_config() {
        let dir = TempDir::new().unwrap();
        let config = EnvironmentConfig {
            mode: EnvironmentMode::Build,
            backend: EnvironmentBackend::AppleContainers,
            image: "pb-dev:latest".to_string(),
            init_commands: vec![],
            setup_commands: vec![],
            session_commands: vec![],
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: Some(PathBuf::from("Dockerfile")),
        };
        config.save(dir.path()).unwrap();
        let loaded = EnvironmentConfig::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.mode, EnvironmentMode::Build);
        assert_eq!(loaded.dockerfile, Some(PathBuf::from("Dockerfile")));
    }

    #[test]
    fn round_trip_local_backend_config() {
        let dir = TempDir::new().unwrap();
        let config = EnvironmentConfig {
            mode: EnvironmentMode::Local,
            backend: EnvironmentBackend::Local,
            image: "local".to_string(),
            init_commands: vec!["cargo check".to_string()],
            setup_commands: vec![],
            session_commands: vec![],
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        };
        config.save(dir.path()).unwrap();
        let loaded = EnvironmentConfig::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.mode, EnvironmentMode::Local);
        assert_eq!(loaded.backend, EnvironmentBackend::Local);
        assert_eq!(loaded.init_commands, vec!["cargo check"]);
    }

    #[test]
    fn load_returns_none_when_no_file() {
        let dir = TempDir::new().unwrap();
        let result = EnvironmentConfig::load(dir.path()).unwrap();
        assert!(result.is_none());
    }
}
