use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const ENVIRONMENT_CONFIG_VERSION: u32 = 2;

fn current_environment_version() -> u32 {
    ENVIRONMENT_CONFIG_VERSION
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum EnvironmentNetworkMode {
    /// Attach the container to a pb-owned host-only internal network.
    #[default]
    Isolated,
    /// Use the runtime's default network with external egress.
    Egress,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnvironmentResources {
    pub cpus: u32,
    pub memory_mb: u64,
}

impl Default for EnvironmentResources {
    fn default() -> Self {
        Self {
            cpus: 4,
            memory_mb: 4096,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentCache {
    /// Stable logical cache identifier within this project environment.
    pub id: String,
    /// Absolute mount target inside the container.
    pub target: PathBuf,
    /// Project-relative files whose contents invalidate the cache.
    #[serde(default)]
    pub key_files: Vec<PathBuf>,
    /// Trust boundary for cache sharing and garbage collection.
    #[serde(default)]
    pub trust: CacheTrustClass,
    /// Optional soft quota recorded for cache accounting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTrustClass {
    Download,
    Toolchain,
    #[default]
    ProjectExecutable,
    LspIndex,
}

/// Project environment configuration stored at `.pb/environment.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentConfig {
    /// Schema version. Older files without the field are interpreted as the current compatible
    /// version; explicit future versions fail closed.
    #[serde(default = "current_environment_version")]
    pub version: u32,

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

    /// Commands that populate the workspace and declared cache volumes in a disposable bootstrap
    /// container. They must not rely on mutations to the container root filesystem persisting.
    #[serde(default)]
    pub setup_commands: Vec<String>,

    /// Commands documented as per-session refresh steps. Most projects leave this empty.
    #[serde(default)]
    pub session_commands: Vec<String>,

    /// Non-secret environment variables passed to bootstrap and runtime containers. Secrets must
    /// use a future secret-reference mechanism rather than being stored in project TOML.
    #[serde(default)]
    pub env: BTreeMap<String, String>,

    /// Network granted to the dependency bootstrap container.
    #[serde(default = "bootstrap_network_default")]
    pub bootstrap_network: EnvironmentNetworkMode,

    /// Network granted to the long-running agent command container.
    #[serde(default)]
    pub runtime_network: EnvironmentNetworkMode,

    /// Explicit VM/container resource bounds.
    #[serde(default)]
    pub resources: EnvironmentResources,

    /// Persistent project-scoped cache volumes.
    #[serde(default)]
    pub caches: Vec<EnvironmentCache>,

    /// Commands that should pass before committing changes.
    #[serde(default)]
    pub guard_commands: Vec<String>,

    /// Deprecated legacy tag from the removed mutable-container commit flow. It is read only for
    /// compatibility and ignored by current runtimes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prepared_image: Option<String>,

    /// Human-readable source/reason for the selected backend and commands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Path to the Dockerfile used for `build` mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dockerfile: Option<PathBuf>,
}

fn bootstrap_network_default() -> EnvironmentNetworkMode {
    EnvironmentNetworkMode::Egress
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

    pub fn validate(&self) -> Result<()> {
        if self.version == 0 || self.version > ENVIRONMENT_CONFIG_VERSION {
            bail!(
                "unsupported environment config version {}; supported versions are 1 through {}",
                self.version,
                ENVIRONMENT_CONFIG_VERSION
            );
        }
        if self.resources.cpus == 0 || self.resources.cpus > 64 {
            bail!("environment resources.cpus must be between 1 and 64");
        }
        if !(256..=262_144).contains(&self.resources.memory_mb) {
            bail!("environment resources.memory_mb must be between 256 and 262144");
        }
        if self.image.trim().is_empty()
            || self.image.starts_with('-')
            || self.image.contains(['\0', '\n', '\r'])
        {
            bail!("environment image must be a non-option, single-line reference");
        }
        for (key, value) in &self.env {
            let mut chars = key.chars();
            if !chars
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
                || !chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            {
                bail!("environment variable name '{key}' is invalid");
            }
            if value.contains('\0') {
                bail!("environment variable '{key}' contains a NUL byte");
            }
        }
        match (self.backend, &self.mode) {
            (EnvironmentBackend::Local, EnvironmentMode::Local) => {}
            (EnvironmentBackend::Local, _) => {
                bail!("local environment backend requires mode=local")
            }
            (EnvironmentBackend::AppleContainers, EnvironmentMode::Local) => {
                bail!("container environment backend cannot use mode=local")
            }
            (EnvironmentBackend::AppleContainers, EnvironmentMode::Build)
                if self.dockerfile.is_none() =>
            {
                bail!("build environment requires a dockerfile")
            }
            (EnvironmentBackend::AppleContainers, _) => {}
        }
        if let Some(dockerfile) = &self.dockerfile
            && (dockerfile.is_absolute()
                || dockerfile.components().any(|component| {
                    matches!(
                        component,
                        std::path::Component::ParentDir
                            | std::path::Component::RootDir
                            | std::path::Component::Prefix(_)
                    )
                }))
        {
            bail!("environment dockerfile must stay within the project");
        }
        let mut cache_ids = BTreeSet::new();
        let mut cache_targets = BTreeSet::new();
        for cache in &self.caches {
            if cache.id.is_empty()
                || !cache
                    .id
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
            {
                bail!(
                    "environment cache id '{}' must contain only ASCII letters, numbers, '-' or '_'",
                    cache.id
                );
            }
            if !cache_ids.insert(cache.id.as_str()) {
                bail!("duplicate environment cache id '{}'", cache.id);
            }
            if !cache.target.is_absolute()
                || cache
                    .target
                    .components()
                    .any(|part| part == std::path::Component::ParentDir)
                || cache.target.to_string_lossy().contains(':')
            {
                bail!(
                    "environment cache '{}' target must be an absolute, non-escaping, CLI-safe container path",
                    cache.id
                );
            }
            if matches!(cache.target.to_str(), Some("/") | Some("/workspace")) {
                bail!(
                    "environment cache '{}' cannot replace the container root or workspace root",
                    cache.id
                );
            }
            if !cache_targets.insert(cache.target.as_path()) {
                bail!(
                    "multiple environment caches target {}",
                    cache.target.display()
                );
            }
            for key_file in &cache.key_files {
                if key_file.is_absolute()
                    || key_file.components().any(|part| {
                        matches!(
                            part,
                            std::path::Component::ParentDir
                                | std::path::Component::RootDir
                                | std::path::Component::Prefix(_)
                        )
                    })
                {
                    bail!(
                        "environment cache '{}' key file must stay within the project: {}",
                        cache.id,
                        key_file.display()
                    );
                }
            }
            if cache.max_bytes == Some(0) {
                bail!(
                    "environment cache '{}' max_bytes must be greater than zero",
                    cache.id
                );
            }
        }
        Ok(())
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
        let config: Self =
            toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        config
            .validate()
            .with_context(|| format!("invalid environment configuration in {}", path.display()))?;
        Ok(config)
    }

    /// Persist the config to `<workspace_root>/.pb/environment.toml`, creating directories as needed.
    pub fn save(&self, workspace_root: &Path) -> Result<()> {
        self.validate()
            .context("invalid environment configuration")?;
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
            version: ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Pull,
            backend: EnvironmentBackend::AppleContainers,
            image: "ghcr.io/example/dev:latest".to_string(),
            init_commands: vec!["npm ci".to_string()],
            setup_commands: vec![],
            session_commands: vec![],
            env: BTreeMap::new(),
            bootstrap_network: EnvironmentNetworkMode::Egress,
            runtime_network: EnvironmentNetworkMode::Isolated,
            resources: EnvironmentResources::default(),
            caches: vec![],
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
            version: ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Build,
            backend: EnvironmentBackend::AppleContainers,
            image: "pb-dev:latest".to_string(),
            init_commands: vec![],
            setup_commands: vec![],
            session_commands: vec![],
            env: BTreeMap::new(),
            bootstrap_network: EnvironmentNetworkMode::Egress,
            runtime_network: EnvironmentNetworkMode::Isolated,
            resources: EnvironmentResources::default(),
            caches: vec![],
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
            version: ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Local,
            backend: EnvironmentBackend::Local,
            image: "local".to_string(),
            init_commands: vec!["cargo check".to_string()],
            setup_commands: vec![],
            session_commands: vec![],
            env: BTreeMap::new(),
            bootstrap_network: EnvironmentNetworkMode::Egress,
            runtime_network: EnvironmentNetworkMode::Isolated,
            resources: EnvironmentResources::default(),
            caches: vec![],
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

    #[test]
    fn rejects_future_versions_and_unknown_fields() {
        let future = r#"
version = 999
mode = "pull"
backend = "apple_containers"
image = "rust:latest"
"#;
        let config: EnvironmentConfig = toml::from_str(future).unwrap();
        assert!(config.validate().is_err());

        let unknown = r#"
version = 2
mode = "pull"
backend = "apple_containers"
image = "rust:latest"
network = "host"
"#;
        assert!(toml::from_str::<EnvironmentConfig>(unknown).is_err());
    }

    #[test]
    fn rejects_backend_mode_mismatch_and_workspace_replacement_cache() {
        let config = EnvironmentConfig {
            version: ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Local,
            backend: EnvironmentBackend::AppleContainers,
            image: "local".to_string(),
            init_commands: vec![],
            setup_commands: vec![],
            session_commands: vec![],
            env: BTreeMap::new(),
            bootstrap_network: EnvironmentNetworkMode::Egress,
            runtime_network: EnvironmentNetworkMode::Isolated,
            resources: EnvironmentResources::default(),
            caches: vec![],
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        };
        assert!(config.validate().is_err());

        let mut config = config;
        config.mode = EnvironmentMode::Pull;
        config.image = "rust:latest".to_string();
        config.caches.push(EnvironmentCache {
            id: "workspace".to_string(),
            target: PathBuf::from("/workspace"),
            key_files: vec![],
            trust: CacheTrustClass::ProjectExecutable,
            max_bytes: None,
        });
        assert!(config.validate().is_err());

        config.caches.clear();
        config.image = "--privileged".to_string();
        assert!(config.validate().is_err());

        config.image = "pb-dev:locked".to_string();
        config.mode = EnvironmentMode::Build;
        config.dockerfile = Some(PathBuf::from("../Dockerfile"));
        assert!(config.validate().is_err());
    }
}
