//! Project-owned authority for native Python dependency discovery.
//!
//! The default profile reads only conventional repository-local virtual environments. Reading an
//! environment inside the repository may be selected by this versioned project document. Reading
//! an environment or editable source tree outside the repository requires an exact, workspace-bound
//! grant in the user-owned global configuration; paths are never inferred from the parent process
//! environment.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub(crate) const PYTHON_SEMANTIC_CONFIG_VERSION: u32 = 1;
pub(crate) const PYTHON_SEMANTIC_CONFIG_PATH: &str = ".pb/python.toml";

fn current_version() -> u32 {
    PYTHON_SEMANTIC_CONFIG_VERSION
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PythonSemanticConfig {
    #[serde(default = "current_version")]
    pub(crate) version: u32,
    /// Exact virtual-environment directory. Relative paths resolve from the repository root.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) environment: Option<PathBuf>,
}

impl Default for PythonSemanticConfig {
    fn default() -> Self {
        Self {
            version: PYTHON_SEMANTIC_CONFIG_VERSION,
            environment: None,
        }
    }
}

impl PythonSemanticConfig {
    pub(crate) fn load(workspace_root: &Path) -> Result<Self> {
        let path = workspace_root.join(PYTHON_SEMANTIC_CONFIG_PATH);
        if !path.exists() {
            return Ok(Self::default());
        }
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("failed to inspect {}", path.display()))?;
        if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 64 * 1024 {
            bail!(
                "native Python semantic configuration must be a bounded regular file: {}",
                path.display()
            );
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Self =
            toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        config.validate().with_context(|| {
            format!("invalid native Python configuration in {}", path.display())
        })?;
        Ok(config)
    }

    pub(crate) fn save(&self, workspace_root: &Path) -> Result<()> {
        self.validate()?;
        let path = workspace_root.join(PYTHON_SEMANTIC_CONFIG_PATH);
        let parent = path
            .parent()
            .context("native Python semantic config has no parent")?;
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        let bytes = toml::to_string_pretty(self)
            .context("failed to serialize native Python semantic configuration")?;
        crate::atomic_file::write(&path, bytes.as_bytes())
    }

    fn validate(&self) -> Result<()> {
        if self.version != PYTHON_SEMANTIC_CONFIG_VERSION {
            bail!(
                "native Python semantic config version must be {}",
                PYTHON_SEMANTIC_CONFIG_VERSION
            );
        }
        if let Some(environment) = &self.environment {
            validate_declared_path("environment", environment)?;
        }
        Ok(())
    }
}

fn validate_declared_path(label: &str, path: &Path) -> Result<()> {
    let text = path
        .to_str()
        .with_context(|| format!("native Python {label} path must be UTF-8"))?;
    if text.trim().is_empty() || text.contains(['\0', '\n', '\r']) {
        bail!("native Python {label} path must be a non-empty single-line path");
    }
    if path.parent().is_none() {
        bail!("native Python {label} path cannot name a filesystem root");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips_and_future_versions_fail_closed() {
        let root = tempfile::tempdir().unwrap();
        let config = PythonSemanticConfig::default();
        config.save(root.path()).unwrap();
        assert_eq!(PythonSemanticConfig::load(root.path()).unwrap(), config);

        std::fs::write(
            root.path().join(PYTHON_SEMANTIC_CONFIG_PATH),
            "version = 2\n",
        )
        .unwrap();
        assert!(
            PythonSemanticConfig::load(root.path())
                .unwrap_err()
                .to_string()
                .contains("invalid native Python configuration")
        );
    }

    #[test]
    fn unknown_external_authority_fields_fail_closed() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join(".pb")).unwrap();
        std::fs::write(
            root.path().join(PYTHON_SEMANTIC_CONFIG_PATH),
            "version = 1\nallowed_editable_roots = [\"../dependency\"]\n",
        )
        .unwrap();
        assert!(
            PythonSemanticConfig::load(root.path())
                .unwrap_err()
                .to_string()
                .contains("failed to parse")
        );
    }
}

/// User-owned, workspace-bound authority resolved from the global pb configuration. This value is
/// never deserialized from an agent request or repository-owned file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PythonExternalAuthority {
    pub(crate) environment: Option<PathBuf>,
    pub(crate) editable_roots: Vec<PathBuf>,
}
