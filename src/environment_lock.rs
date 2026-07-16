use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use crate::container::RuntimeInfo;
use crate::environment::{EnvironmentBackend, EnvironmentConfig, EnvironmentMode};

pub const ENVIRONMENT_LOCK_VERSION: u32 = 1;
pub const ENVIRONMENT_RESOLVER_VERSION: u32 = 2;
pub const ENVIRONMENT_EVIDENCE_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostCapability {
    XcodeProject,
    XcodeWorkspace,
    AppleSwiftPackage,
    AppleFrameworkSource,
    XcodeToolchain,
    AppleSdkTarget,
    Simulator,
    CodeSigning,
    Metal,
    MacosCi,
}

impl HostCapability {
    pub fn label(self) -> &'static str {
        match self {
            Self::XcodeProject => "Xcode project",
            Self::XcodeWorkspace => "Xcode workspace",
            Self::AppleSwiftPackage => "Apple-platform Swift package",
            Self::AppleFrameworkSource => "Apple framework source import",
            Self::XcodeToolchain => "Xcode toolchain command",
            Self::AppleSdkTarget => "Apple SDK target",
            Self::Simulator => "Apple simulator",
            Self::CodeSigning => "Apple code signing/notarization",
            Self::Metal => "Metal",
            Self::MacosCi => "macOS CI runner",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EnvironmentRequirement {
    HostCapability { capability: HostCapability },
    DependencyInput { path: PathBuf },
    Toolchain { name: String, constraint: String },
    ContainerSignal { detail: String },
    SetupCommand { command: String },
    GuardCommand { command: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentEvidence {
    pub source_path: PathBuf,
    pub component: String,
    pub requirement: EnvironmentRequirement,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentEvidenceDocument {
    pub version: u32,
    pub evidence: Vec<EnvironmentEvidence>,
}

impl EnvironmentEvidenceDocument {
    pub fn path(workspace_root: &Path) -> PathBuf {
        workspace_root.join(".pb").join("environment.evidence.json")
    }

    pub fn load(workspace_root: &Path) -> Result<Option<Self>> {
        let path = Self::path(workspace_root);
        if !path.exists() {
            return Ok(None);
        }
        let document: Self = serde_json::from_slice(&std::fs::read(&path)?)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        if document.version != ENVIRONMENT_EVIDENCE_VERSION {
            bail!(
                "unsupported environment evidence version {}",
                document.version
            );
        }
        Ok(Some(document))
    }

    pub fn save_atomic(&self, workspace_root: &Path) -> Result<()> {
        if self.version != ENVIRONMENT_EVIDENCE_VERSION {
            bail!("unsupported environment evidence version {}", self.version);
        }
        let path = Self::path(workspace_root);
        let parent = path
            .parent()
            .context("environment evidence has no parent")?;
        std::fs::create_dir_all(parent)?;
        let temp = parent.join(format!(".environment.evidence.{}.tmp", std::process::id()));
        std::fs::write(&temp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&temp, &path)
            .with_context(|| format!("failed to replace {}", path.display()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedComponentAuthority {
    pub component: String,
    pub backend: EnvironmentBackend,
    pub host_capabilities: Vec<HostCapability>,
    pub evidence: Vec<EnvironmentEvidence>,
}

pub fn resolve_component_authority(
    component: &str,
    evidence: &[EnvironmentEvidence],
) -> ResolvedComponentAuthority {
    let selected = evidence
        .iter()
        .filter(|item| {
            component == "repository"
                || item.component == component
                || item.component == "repository"
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut host_capabilities = selected
        .iter()
        .filter_map(|item| match item.requirement {
            EnvironmentRequirement::HostCapability { capability } => Some(capability),
            _ => None,
        })
        .collect::<Vec<_>>();
    host_capabilities.sort();
    host_capabilities.dedup();
    ResolvedComponentAuthority {
        component: component.to_string(),
        backend: if host_capabilities.is_empty() {
            EnvironmentBackend::AppleContainers
        } else {
            EnvironmentBackend::Local
        },
        host_capabilities,
        evidence: selected,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentLock {
    pub version: u32,
    pub resolver_version: u32,
    pub config_sha256: String,
    pub backend: EnvironmentBackend,
    pub mode: EnvironmentMode,
    pub configured_image: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_image_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build_inputs_sha256: Option<String>,
    pub runtime: String,
    pub runtime_version: String,
    pub platform: String,
    pub dependency_inputs: BTreeMap<PathBuf, String>,
    pub cache_plan_sha256: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedEnvironment {
    pub config: EnvironmentConfig,
    pub lock: EnvironmentLock,
    pub lock_sha256: String,
}

impl EnvironmentLock {
    pub fn path(workspace_root: &Path) -> PathBuf {
        workspace_root.join(".pb").join("environment.lock")
    }

    pub fn load(workspace_root: &Path) -> Result<Option<Self>> {
        let path = Self::path(workspace_root);
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let lock: Self =
            toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        lock.validate()
            .with_context(|| format!("invalid environment lock {}", path.display()))?;
        Ok(Some(lock))
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != ENVIRONMENT_LOCK_VERSION {
            bail!(
                "unsupported environment lock version {}; expected {}",
                self.version,
                ENVIRONMENT_LOCK_VERSION
            );
        }
        if self.resolver_version != ENVIRONMENT_RESOLVER_VERSION {
            bail!(
                "environment lock resolver version {} is stale; expected {}",
                self.resolver_version,
                ENVIRONMENT_RESOLVER_VERSION
            );
        }
        for (label, value) in [
            ("config_sha256", self.config_sha256.as_str()),
            ("cache_plan_sha256", self.cache_plan_sha256.as_str()),
        ] {
            if !is_sha256(value) {
                bail!("environment lock {label} must be a lowercase SHA-256 digest");
            }
        }
        if self.backend == EnvironmentBackend::AppleContainers
            && self
                .local_image_sha256
                .as_deref()
                .is_none_or(|value| !is_sha256(value))
        {
            bail!("container environment lock requires local_image_sha256");
        }
        Ok(())
    }

    pub fn canonical_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).context("failed to serialize environment lock")
    }

    pub fn sha256(&self) -> Result<String> {
        Ok(sha256(self.canonical_toml()?.as_bytes()))
    }

    pub fn save_atomic(&self, workspace_root: &Path) -> Result<()> {
        self.validate()?;
        let path = Self::path(workspace_root);
        let parent = path.parent().context("environment lock has no parent")?;
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        let temp = parent.join(format!(".environment.lock.{}.tmp", std::process::id()));
        std::fs::write(&temp, self.canonical_toml()?)
            .with_context(|| format!("failed to write {}", temp.display()))?;
        std::fs::rename(&temp, &path)
            .with_context(|| format!("failed to replace {}", path.display()))?;
        Ok(())
    }
}

pub fn resolve_environment(
    config: &EnvironmentConfig,
    workspace_root: &Path,
    runtime: Option<&RuntimeInfo>,
    local_image_metadata: Option<&str>,
) -> Result<ResolvedEnvironment> {
    resolve_environment_inner(config, workspace_root, runtime, local_image_metadata, true)
}

pub(crate) fn resolve_environment_candidate(
    config: &EnvironmentConfig,
    workspace_root: &Path,
    runtime: Option<&RuntimeInfo>,
    local_image_metadata: Option<&str>,
) -> Result<ResolvedEnvironment> {
    resolve_environment_inner(config, workspace_root, runtime, local_image_metadata, false)
}

fn resolve_environment_inner(
    config: &EnvironmentConfig,
    workspace_root: &Path,
    runtime: Option<&RuntimeInfo>,
    local_image_metadata: Option<&str>,
    persist: bool,
) -> Result<ResolvedEnvironment> {
    config.validate()?;
    let config_toml =
        toml::to_string(config).context("failed to canonicalize environment config")?;
    let dependency_inputs = dependency_input_hashes(config, workspace_root)?;
    let cache_plan = serde_json::to_vec(&config.caches)
        .context("failed to canonicalize environment cache plan")?;
    let (runtime_name, runtime_version, local_image_sha256) = match config.backend {
        EnvironmentBackend::AppleContainers => {
            let runtime = runtime.context("container environment resolution requires a runtime")?;
            let metadata = local_image_metadata
                .context("container environment resolution requires inspected image metadata")?;
            (
                runtime.binary.clone(),
                runtime.version.clone(),
                Some(sha256(metadata.as_bytes())),
            )
        }
        EnvironmentBackend::Local => ("host".to_string(), "host".to_string(), None),
    };
    let lock = EnvironmentLock {
        version: ENVIRONMENT_LOCK_VERSION,
        resolver_version: ENVIRONMENT_RESOLVER_VERSION,
        config_sha256: sha256(config_toml.as_bytes()),
        backend: config.backend,
        mode: config.mode.clone(),
        configured_image: config.image.clone(),
        local_image_sha256,
        build_inputs_sha256: match config.mode {
            EnvironmentMode::Build => Some(hash_build_inputs(config, workspace_root)?),
            _ => None,
        },
        runtime: runtime_name,
        runtime_version,
        platform: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
        dependency_inputs,
        cache_plan_sha256: sha256(&cache_plan),
    };
    if persist {
        lock.save_atomic(workspace_root)?;
    }
    Ok(ResolvedEnvironment {
        config: config.clone(),
        lock_sha256: lock.sha256()?,
        lock,
    })
}

fn dependency_input_hashes(
    config: &EnvironmentConfig,
    workspace_root: &Path,
) -> Result<BTreeMap<PathBuf, String>> {
    let mut inputs = BTreeMap::new();
    for path in config
        .caches
        .iter()
        .flat_map(|cache| cache.key_files.iter())
    {
        if inputs.contains_key(path) {
            continue;
        }
        let digest = dependency_input_digest(workspace_root, path)?;
        inputs.insert(path.clone(), digest);
    }
    Ok(inputs)
}

fn hash_build_inputs(config: &EnvironmentConfig, workspace_root: &Path) -> Result<String> {
    let dockerfile = config
        .dockerfile
        .as_deref()
        .context("build environment is missing dockerfile")?;
    let context_root = workspace_root
        .join(dockerfile)
        .parent()
        .context("Dockerfile has no parent")?
        .to_path_buf();
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(&context_root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| !ignored_build_entry(&context_root, entry.path()))
    {
        let entry = entry.with_context(|| {
            format!(
                "failed to enumerate build context {}",
                context_root.display()
            )
        })?;
        if entry.file_type().is_file() || entry.file_type().is_symlink() {
            files.push(entry.path().to_path_buf());
        }
    }
    files.sort();
    let mut digest = Sha256::new();
    for path in files {
        let relative = path.strip_prefix(&context_root).unwrap_or(&path);
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update([0]);
        if path.is_symlink() {
            digest.update(std::fs::read_link(&path)?.to_string_lossy().as_bytes());
        } else {
            digest.update(
                std::fs::read(&path)
                    .with_context(|| format!("failed to hash build input {}", path.display()))?,
            );
        }
        digest.update([0]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn ignored_build_entry(root: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(root) else {
        return true;
    };
    relative.components().any(|component| {
        matches!(
            component,
            Component::Normal(name)
                if matches!(name.to_str(), Some(".git"))
        )
    }) || relative == Path::new(".pb/environment.lock")
}

pub(crate) fn dependency_input_digest(workspace_root: &Path, path: &Path) -> Result<String> {
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        bail!(
            "dependency input must stay within the project: {}",
            path.display()
        );
    }
    let canonical_root = workspace_root.canonicalize().with_context(|| {
        format!(
            "failed to resolve workspace root {}",
            workspace_root.display()
        )
    })?;
    let absolute = canonical_root.join(path);
    let canonical = match absolute.canonicalize() {
        Ok(canonical) => canonical,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(sha256(b"<missing>"));
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!("failed to resolve dependency input {}", absolute.display())
            });
        }
    };
    if !canonical.starts_with(&canonical_root) {
        bail!(
            "dependency input {} escapes workspace {}",
            absolute.display(),
            canonical_root.display()
        );
    }
    let bytes = std::fs::read(&canonical)
        .with_context(|| format!("failed to hash dependency input {}", canonical.display()))?;
    Ok(sha256(&bytes))
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{RuntimeCapabilities, RuntimeKind};
    use crate::environment::{EnvironmentCache, EnvironmentNetworkMode, EnvironmentResources};
    use tempfile::TempDir;

    fn config() -> EnvironmentConfig {
        EnvironmentConfig {
            version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
            mode: EnvironmentMode::Pull,
            backend: EnvironmentBackend::AppleContainers,
            image: "rust:latest".to_string(),
            init_commands: vec![],
            setup_commands: vec!["cargo fetch --locked".to_string()],
            session_commands: vec![],
            env: BTreeMap::new(),
            bootstrap_network: EnvironmentNetworkMode::Egress,
            runtime_network: EnvironmentNetworkMode::Isolated,
            resources: EnvironmentResources::default(),
            caches: vec![EnvironmentCache {
                id: "cargo".to_string(),
                target: PathBuf::from("/usr/local/cargo/registry"),
                key_files: vec![PathBuf::from("Cargo.lock")],
                trust: crate::environment::CacheTrustClass::ProjectExecutable,
                max_bytes: None,
            }],
            guard_commands: vec![],
            prepared_image: None,
            source: None,
            dockerfile: None,
        }
    }

    fn runtime() -> RuntimeInfo {
        RuntimeInfo {
            kind: RuntimeKind::Apple,
            binary: "container".to_string(),
            version: "1.0.0".to_string(),
            capabilities: RuntimeCapabilities {
                internal_networks: true,
                named_volumes: true,
                labels: true,
                resource_limits: true,
            },
        }
    }

    #[test]
    fn lock_changes_with_image_or_dependency_identity_and_round_trips() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("Cargo.lock"), "one").unwrap();
        let first = resolve_environment(&config(), dir.path(), Some(&runtime()), Some("image-one"))
            .unwrap();
        let loaded = EnvironmentLock::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded, first.lock);

        let second =
            resolve_environment(&config(), dir.path(), Some(&runtime()), Some("image-two"))
                .unwrap();
        assert_ne!(first.lock_sha256, second.lock_sha256);
        std::fs::write(dir.path().join("Cargo.lock"), "two").unwrap();
        let third = resolve_environment(&config(), dir.path(), Some(&runtime()), Some("image-two"))
            .unwrap();
        assert_ne!(second.lock_sha256, third.lock_sha256);
    }

    #[test]
    fn component_authority_fails_to_host_for_positive_capability_evidence() {
        let evidence = vec![EnvironmentEvidence {
            source_path: PathBuf::from("Client.xcodeproj"),
            component: "client".to_string(),
            requirement: EnvironmentRequirement::HostCapability {
                capability: HostCapability::XcodeProject,
            },
            detail: "Xcode project".to_string(),
        }];
        assert_eq!(
            resolve_component_authority("client", &evidence).backend,
            EnvironmentBackend::Local
        );
        assert_eq!(
            resolve_component_authority("server", &evidence).backend,
            EnvironmentBackend::AppleContainers
        );
        assert_eq!(
            resolve_component_authority("repository", &evidence).backend,
            EnvironmentBackend::Local
        );

        let dir = TempDir::new().unwrap();
        let document = EnvironmentEvidenceDocument {
            version: ENVIRONMENT_EVIDENCE_VERSION,
            evidence,
        };
        document.save_atomic(dir.path()).unwrap();
        assert_eq!(
            EnvironmentEvidenceDocument::load(dir.path())
                .unwrap()
                .unwrap(),
            document
        );
    }

    #[test]
    fn candidate_resolution_does_not_persist_a_preparation_state() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("Cargo.lock"), "one").unwrap();
        let candidate = resolve_environment_candidate(
            &config(),
            dir.path(),
            Some(&runtime()),
            Some("candidate-image"),
        )
        .unwrap();
        assert!(EnvironmentLock::load(dir.path()).unwrap().is_none());

        let resolved = resolve_environment(
            &config(),
            dir.path(),
            Some(&runtime()),
            Some("candidate-image"),
        )
        .unwrap();
        assert_eq!(resolved.lock_sha256, candidate.lock_sha256);
        assert!(EnvironmentLock::load(dir.path()).unwrap().is_some());
    }

    #[cfg(unix)]
    #[test]
    fn dependency_inputs_cannot_follow_symlinks_outside_the_workspace() {
        use std::os::unix::fs::symlink;

        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("lock"), "secret").unwrap();
        symlink(outside.path().join("lock"), dir.path().join("Cargo.lock")).unwrap();
        let error = dependency_input_digest(dir.path(), Path::new("Cargo.lock"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("escapes workspace"));
    }
}
