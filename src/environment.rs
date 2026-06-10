use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum EnvironmentMode {
    #[default]
    Pull,
    Build,
}


/// Project environment configuration stored at `.pb/environment.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    /// Whether the image was pulled from a registry (`pull`) or built locally (`build`).
    #[serde(default)]
    pub mode: EnvironmentMode,

    /// Container image reference (e.g. `ghcr.io/myorg/dev:latest` or a locally built tag).
    pub image: String,

    /// Shell commands run inside the container after it is created, before agent work begins.
    #[serde(default)]
    pub init_commands: Vec<String>,

    /// Path to the Dockerfile used for `build` mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dockerfile: Option<PathBuf>,
}

impl EnvironmentConfig {
    /// Load the project environment config from `<workspace_root>/.pb/environment.toml`.
    /// Returns `Ok(None)` when no config file exists.
    pub fn load(workspace_root: &Path) -> Result<Option<Self>> {
        let path = workspace_root.join(".pb").join("environment.toml");
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Self = toml::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Some(config))
    }

    /// Persist the config to `<workspace_root>/.pb/environment.toml`, creating directories as needed.
    pub fn save(&self, workspace_root: &Path) -> Result<()> {
        let dir = workspace_root.join(".pb");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create directory {}", dir.display()))?;
        let path = dir.join("environment.toml");
        let text = toml::to_string_pretty(self).context("failed to serialize environment config")?;
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
            image: "ghcr.io/example/dev:latest".to_string(),
            init_commands: vec!["npm ci".to_string()],
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
            image: "pb-dev:latest".to_string(),
            init_commands: vec![],
            dockerfile: Some(PathBuf::from("Dockerfile")),
        };
        config.save(dir.path()).unwrap();
        let loaded = EnvironmentConfig::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.mode, EnvironmentMode::Build);
        assert_eq!(loaded.dockerfile, Some(PathBuf::from("Dockerfile")));
    }

    #[test]
    fn load_returns_none_when_no_file() {
        let dir = TempDir::new().unwrap();
        let result = EnvironmentConfig::load(dir.path()).unwrap();
        assert!(result.is_none());
    }
}
