use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::Read;
use std::path::{Component as PathComponent, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use globset::Glob;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::environment::EnvironmentConfig;

pub const WORKSPACE_CONFIG_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepositoryContext {
    pub repo_root: PathBuf,
    pub focus_root: PathBuf,
    pub task_baseline: WorkspaceBaseline,
    pub invocation_baseline: WorkspaceBaseline,
}

impl RepositoryContext {
    pub fn capture(repo_root: &Path, focus_root: &Path) -> Result<Self> {
        let repo_root = repo_root.canonicalize().with_context(|| {
            format!("failed to resolve repository root {}", repo_root.display())
        })?;
        let focus_root = focus_root
            .canonicalize()
            .with_context(|| format!("failed to resolve focus root {}", focus_root.display()))?;
        if !focus_root.starts_with(&repo_root) {
            bail!(
                "focus root {} is outside repository root {}",
                focus_root.display(),
                repo_root.display()
            );
        }
        let baseline = WorkspaceBaseline::capture(&repo_root)?;
        Ok(Self {
            repo_root,
            focus_root,
            task_baseline: baseline.clone(),
            invocation_baseline: baseline,
        })
    }

    pub fn resume(
        repo_root: &Path,
        focus_root: &Path,
        task_baseline: WorkspaceBaseline,
    ) -> Result<Self> {
        let mut context = Self::capture(repo_root, focus_root)?;
        context.task_baseline = task_baseline;
        Ok(context)
    }

    pub fn refresh_invocation_baseline(&mut self) -> Result<()> {
        self.invocation_baseline = WorkspaceBaseline::capture(&self.repo_root)?;
        Ok(())
    }

    pub fn task_changed_paths(&self) -> Result<Vec<String>> {
        let current = ContentSnapshot::capture(&self.repo_root)?;
        Ok(self.task_baseline.content.changed_paths(&current))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceBaseline {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
    pub status: WorkspaceStatus,
    pub content: ContentSnapshot,
}

impl WorkspaceBaseline {
    pub fn capture(repo_root: &Path) -> Result<Self> {
        let head = git_optional(repo_root, &["rev-parse", "--verify", "HEAD"])?;
        let status = WorkspaceStatus::capture(repo_root)?;
        let content = ContentSnapshot::capture(repo_root)?;
        let mut digest = Sha256::new();
        digest.update(head.as_deref().unwrap_or("<unborn>").as_bytes());
        digest.update([0]);
        digest.update(status.porcelain.as_bytes());
        digest.update([0]);
        digest.update(content.fingerprint.as_bytes());
        let id = format!("{:x}", digest.finalize());
        Ok(Self {
            id,
            head,
            status,
            content,
        })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceStatus {
    pub porcelain: String,
    #[serde(default)]
    pub dirty_paths: Vec<String>,
}

impl WorkspaceStatus {
    fn capture(repo_root: &Path) -> Result<Self> {
        let porcelain = git_required(
            repo_root,
            &["status", "--porcelain=v1", "--untracked-files=all"],
        )?;
        let mut dirty_paths = BTreeSet::new();
        for args in [
            &["diff", "--name-only", "-z"][..],
            &["diff", "--cached", "--name-only", "-z"][..],
            &["ls-files", "--others", "--exclude-standard", "-z"][..],
        ] {
            let output = git_bytes(repo_root, args)?;
            for path in output
                .split(|byte| *byte == 0)
                .filter(|path| !path.is_empty())
            {
                dirty_paths.insert(
                    std::str::from_utf8(path)
                        .context("workspace contains a non-UTF-8 dirty path")?
                        .replace('\\', "/"),
                );
            }
        }
        Ok(Self {
            porcelain,
            dirty_paths: dirty_paths.into_iter().collect(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContentSnapshot {
    pub fingerprint: String,
    pub paths: BTreeMap<String, PathContent>,
}

impl ContentSnapshot {
    pub fn capture(repo_root: &Path) -> Result<Self> {
        let output = git_bytes(
            repo_root,
            &[
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
        )?;
        let mut raw_paths = output
            .split(|byte| *byte == 0)
            .filter(|path| !path.is_empty())
            .collect::<Vec<_>>();
        raw_paths.sort_unstable();

        let mut fingerprint = Sha256::new();
        let mut paths = BTreeMap::new();
        for raw_path in raw_paths {
            let relative = std::str::from_utf8(raw_path)
                .context("workspace contains a non-UTF-8 tracked or untracked path")?;
            let normalized = relative.replace('\\', "/");
            let (kind, bytes) = path_content(repo_root, relative)?;
            if kind == "missing" {
                continue;
            }
            fingerprint.update((raw_path.len() as u64).to_le_bytes());
            fingerprint.update(raw_path);
            fingerprint.update(kind.as_bytes());
            fingerprint.update(&bytes);
            let mut path_digest = Sha256::new();
            path_digest.update(kind.as_bytes());
            path_digest.update(&bytes);
            paths.insert(
                normalized,
                PathContent {
                    kind,
                    fingerprint: format!("{:x}", path_digest.finalize()),
                },
            );
        }
        Ok(Self {
            fingerprint: format!("{:x}", fingerprint.finalize()),
            paths,
        })
    }

    pub(crate) fn capture_until(
        repo_root: &Path,
        deadline: Instant,
        max_total_file_bytes: u64,
    ) -> Result<Self> {
        ensure_snapshot_time(deadline, "listing workspace paths")?;
        let output = git_bytes_until(
            repo_root,
            &[
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            deadline,
            16 * 1024 * 1024,
        )?;
        let mut raw_paths = output
            .split(|byte| *byte == 0)
            .filter(|path| !path.is_empty())
            .collect::<Vec<_>>();
        if raw_paths.len() > 100_000 {
            bail!("workspace snapshot exceeds the 100000-path proactive bound");
        }
        raw_paths.sort_unstable();

        let mut remaining_bytes = max_total_file_bytes;
        let mut fingerprint = Sha256::new();
        let mut paths = BTreeMap::new();
        for raw_path in raw_paths {
            ensure_snapshot_time(deadline, "reading workspace content")?;
            let relative = std::str::from_utf8(raw_path)
                .context("workspace contains a non-UTF-8 tracked or untracked path")?;
            let normalized = relative.replace('\\', "/");
            let (kind, bytes) =
                path_content_until(repo_root, relative, deadline, &mut remaining_bytes)?;
            if kind == "missing" {
                continue;
            }
            fingerprint.update((raw_path.len() as u64).to_le_bytes());
            fingerprint.update(raw_path);
            fingerprint.update(kind.as_bytes());
            fingerprint.update(&bytes);
            let mut path_digest = Sha256::new();
            path_digest.update(kind.as_bytes());
            path_digest.update(&bytes);
            paths.insert(
                normalized,
                PathContent {
                    kind,
                    fingerprint: format!("{:x}", path_digest.finalize()),
                },
            );
        }
        Ok(Self {
            fingerprint: format!("{:x}", fingerprint.finalize()),
            paths,
        })
    }

    pub fn changed_paths(&self, current: &Self) -> Vec<String> {
        self.paths
            .keys()
            .chain(current.paths.keys())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter(|path| self.paths.get(*path) != current.paths.get(*path))
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PathContent {
    pub kind: String,
    pub fingerprint: String,
}

fn path_content(repo_root: &Path, relative: &str) -> Result<(String, Vec<u8>)> {
    let path = repo_root.join(relative);
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Ok((
            "symlink".to_string(),
            std::fs::read_link(&path)
                .with_context(|| format!("failed to read symlink {}", path.display()))?
                .to_string_lossy()
                .as_bytes()
                .to_vec(),
        )),
        Ok(metadata) if metadata.is_file() => Ok((
            "file".to_string(),
            std::fs::read(&path)
                .with_context(|| format!("failed to read worktree file {}", path.display()))?,
        )),
        Ok(metadata) if metadata.is_dir() => Ok(("directory".to_string(), Vec::new())),
        Ok(_) => Ok(("other".to_string(), Vec::new())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(("missing".to_string(), Vec::new()))
        }
        Err(error) => {
            Err(error).with_context(|| format!("failed to inspect path {}", path.display()))
        }
    }
}

fn path_content_until(
    repo_root: &Path,
    relative: &str,
    deadline: Instant,
    remaining_bytes: &mut u64,
) -> Result<(String, Vec<u8>)> {
    ensure_snapshot_time(deadline, "inspecting workspace content")?;
    let path = repo_root.join(relative);
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            let bytes = std::fs::read_link(&path)
                .with_context(|| format!("failed to read symlink {}", path.display()))?
                .to_string_lossy()
                .as_bytes()
                .to_vec();
            consume_snapshot_bytes(remaining_bytes, bytes.len() as u64)?;
            Ok(("symlink".to_string(), bytes))
        }
        Ok(metadata) if metadata.is_file() => {
            consume_snapshot_bytes(remaining_bytes, metadata.len())?;
            let mut file = std::fs::File::open(&path)
                .with_context(|| format!("failed to open worktree file {}", path.display()))?;
            let mut bytes = Vec::with_capacity(
                usize::try_from(metadata.len())
                    .unwrap_or(usize::MAX)
                    .min(1024 * 1024),
            );
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                ensure_snapshot_time(deadline, "reading workspace content")?;
                let count = file
                    .read(&mut buffer)
                    .with_context(|| format!("failed to read worktree file {}", path.display()))?;
                if count == 0 {
                    break;
                }
                bytes.extend_from_slice(&buffer[..count]);
                if bytes.len() as u64 > metadata.len() {
                    bail!(
                        "worktree file {} grew during bounded snapshot",
                        path.display()
                    );
                }
            }
            Ok(("file".to_string(), bytes))
        }
        Ok(metadata) if metadata.is_dir() => Ok(("directory".to_string(), Vec::new())),
        Ok(_) => Ok(("other".to_string(), Vec::new())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(("missing".to_string(), Vec::new()))
        }
        Err(error) => {
            Err(error).with_context(|| format!("failed to inspect path {}", path.display()))
        }
    }
}

fn consume_snapshot_bytes(remaining: &mut u64, bytes: u64) -> Result<()> {
    if bytes > *remaining {
        bail!("workspace content exceeds the proactive snapshot byte bound");
    }
    *remaining -= bytes;
    Ok(())
}

fn ensure_snapshot_time(deadline: Instant, operation: &str) -> Result<()> {
    if Instant::now() >= deadline {
        bail!("proactive workspace snapshot time bound expired while {operation}");
    }
    Ok(())
}

fn git_bytes_until(
    repo_root: &Path,
    args: &[&str],
    deadline: Instant,
    max_output_bytes: usize,
) -> Result<Vec<u8>> {
    ensure_snapshot_time(deadline, "starting Git workspace discovery")?;
    let mut child = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    let stdout = child
        .stdout
        .take()
        .context("failed to capture git stdout")?;
    let stderr = child
        .stderr
        .take()
        .context("failed to capture git stderr")?;
    let stdout_thread = std::thread::spawn(move || read_bounded_stream(stdout, max_output_bytes));
    let stderr_thread = std::thread::spawn(move || read_bounded_stream(stderr, 64 * 1024));
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            bail!("proactive workspace snapshot time bound expired while running git");
        }
        std::thread::sleep(
            Duration::from_millis(5).min(deadline.saturating_duration_since(Instant::now())),
        );
    };
    let (stdout, stdout_truncated) = stdout_thread
        .join()
        .map_err(|_| anyhow::anyhow!("git stdout reader stopped unexpectedly"))??;
    let (stderr, _) = stderr_thread
        .join()
        .map_err(|_| anyhow::anyhow!("git stderr reader stopped unexpectedly"))??;
    if stdout_truncated {
        bail!("git workspace path listing exceeds the proactive output bound");
    }
    if !status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&stderr).trim()
        );
    }
    Ok(stdout)
}

fn read_bounded_stream(mut reader: impl Read, limit: usize) -> Result<(Vec<u8>, bool)> {
    let mut bytes = Vec::new();
    reader
        .by_ref()
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)?;
    let truncated = bytes.len() > limit;
    bytes.truncate(limit);
    Ok((bytes, truncated))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceConfigDocument {
    pub version: u32,
    #[serde(default)]
    pub executors: Vec<Executor>,
    #[serde(default)]
    pub components: Vec<WorkspaceComponent>,
    #[serde(default)]
    pub checks: Vec<WorkspaceCheck>,
    #[serde(default)]
    pub tasks: Vec<WorkspaceTask>,
    #[serde(default)]
    pub cargo_workspaces: Vec<CargoWorkspace>,
}

impl WorkspaceConfigDocument {
    pub fn from_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn load(repo_root: &Path) -> Result<Option<Self>> {
        let path = repo_root.join(".pb").join("workspace.toml");
        if !path.exists() {
            return Ok(None);
        }
        Self::from_path(&path).map(Some)
    }

    pub fn save(&self, repo_root: &Path) -> Result<()> {
        let dir = repo_root.join(".pb");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        let path = dir.join("workspace.toml");
        let text = toml::to_string_pretty(self).context("failed to serialize workspace config")?;
        std::fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn normalize(self) -> Result<WorkspaceGraph> {
        if self.version != WORKSPACE_CONFIG_VERSION {
            bail!(
                "unsupported workspace config version {}; expected {}",
                self.version,
                WORKSPACE_CONFIG_VERSION
            );
        }
        let executors = unique_by_id("executor", self.executors, |item| &item.id)?;
        let components = unique_by_id("component", self.components, |item| &item.id)?;
        let checks = unique_by_id("check", self.checks, |item| &item.id)?;
        let tasks = unique_by_id("task", self.tasks, |item| &item.id)?;
        let cargo_workspaces =
            unique_by_id("Cargo workspace", self.cargo_workspaces, |item| &item.id)?;

        for executor in executors.values() {
            validate_id("executor", &executor.id)?;
            if let Some(environment) = &executor.environment {
                validate_relative_path("executor environment", environment, false)?;
            }
        }
        for component in components.values() {
            validate_id("component", &component.id)?;
            validate_relative_path("component root", &component.root, true)?;
            validate_patterns("component include", &component.include)?;
            validate_patterns("component exclude", &component.exclude)?;
            if !executors.contains_key(&component.executor) {
                bail!(
                    "component '{}' references unknown executor '{}'",
                    component.id,
                    component.executor
                );
            }
            validate_references(
                "component",
                &component.id,
                &component.depends_on,
                &components,
            )?;
        }
        for check in checks.values() {
            validate_id("check", &check.id)?;
            if check.command.trim().is_empty() {
                bail!("check '{}' command must not be empty", check.id);
            }
            validate_relative_path("check cwd", &check.cwd, true)?;
            validate_patterns("check inputs", &check.inputs)?;
            validate_patterns("check outputs", &check.outputs)?;
            if check.timeout_seconds == 0 {
                bail!("check '{}' timeout_seconds must be positive", check.id);
            }
            if !executors.contains_key(&check.executor) {
                bail!(
                    "check '{}' references unknown executor '{}'",
                    check.id,
                    check.executor
                );
            }
            for component in &check.components {
                if !components.contains_key(component) {
                    bail!(
                        "check '{}' references unknown component '{}'",
                        check.id,
                        component
                    );
                }
            }
            validate_references("check", &check.id, &check.depends_on, &checks)?;
        }
        for task in tasks.values() {
            validate_id("task", &task.id)?;
            if task.command.trim().is_empty() {
                bail!("task '{}' command must not be empty", task.id);
            }
            validate_relative_path("task cwd", &task.cwd, true)?;
            validate_patterns("task allowed_changes", &task.allowed_changes)?;
            if task.timeout_seconds == 0 {
                bail!("task '{}' timeout_seconds must be positive", task.id);
            }
            if !executors.contains_key(&task.executor) {
                bail!(
                    "task '{}' references unknown executor '{}'",
                    task.id,
                    task.executor
                );
            }
        }
        validate_acyclic(
            "component",
            components
                .iter()
                .map(|(id, item)| (id.as_str(), item.depends_on.as_slice())),
        )?;
        validate_acyclic(
            "check",
            checks
                .iter()
                .map(|(id, item)| (id.as_str(), item.depends_on.as_slice())),
        )?;
        for workspace in cargo_workspaces.values() {
            validate_id("Cargo workspace", &workspace.id)?;
            validate_relative_path("Cargo workspace root", &workspace.root, true)?;
            validate_relative_path(
                "Cargo workspace manifest_path",
                &workspace.manifest_path,
                false,
            )?;
            for member in workspace
                .members
                .iter()
                .chain(workspace.default_members.iter())
            {
                if !components.contains_key(member) {
                    bail!(
                        "Cargo workspace '{}' references unknown component '{}'",
                        workspace.id,
                        member
                    );
                }
            }
            if workspace
                .default_members
                .iter()
                .any(|member| !workspace.members.contains(member))
            {
                bail!(
                    "Cargo workspace '{}' default_members must be workspace members",
                    workspace.id
                );
            }
        }

        Ok(WorkspaceGraph {
            version: self.version,
            executors,
            components,
            checks,
            tasks,
            cargo_workspaces,
            discovery_warnings: Vec::new(),
            source: WorkspaceGraphSource::Explicit,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceGraph {
    pub version: u32,
    pub executors: BTreeMap<String, Executor>,
    pub components: BTreeMap<String, WorkspaceComponent>,
    pub checks: BTreeMap<String, WorkspaceCheck>,
    #[serde(default)]
    pub tasks: BTreeMap<String, WorkspaceTask>,
    #[serde(default)]
    pub cargo_workspaces: BTreeMap<String, CargoWorkspace>,
    #[serde(default)]
    pub discovery_warnings: Vec<String>,
    pub source: WorkspaceGraphSource,
}

impl WorkspaceGraph {
    pub fn load_or_legacy(
        repo_root: &Path,
        environment: Option<&EnvironmentConfig>,
    ) -> Result<Self> {
        if let Some(document) = WorkspaceConfigDocument::load(repo_root)? {
            return document.normalize();
        }
        Ok(Self::legacy(
            environment
                .map(|config| config.guard_commands.as_slice())
                .unwrap_or_default(),
        ))
    }

    pub fn load_or_discover(
        repo_root: &Path,
        environment: Option<&EnvironmentConfig>,
    ) -> Result<Self> {
        if let Some(document) = WorkspaceConfigDocument::load(repo_root)? {
            return document.normalize();
        }
        crate::workspace_discovery::discover_workspace(repo_root, environment)
    }

    pub fn to_document(&self) -> WorkspaceConfigDocument {
        WorkspaceConfigDocument {
            version: self.version,
            executors: self.executors.values().cloned().collect(),
            components: self.components.values().cloned().collect(),
            checks: self.checks.values().cloned().collect(),
            tasks: self.tasks.values().cloned().collect(),
            cargo_workspaces: self.cargo_workspaces.values().cloned().collect(),
        }
    }

    pub fn legacy(guard_commands: &[String]) -> Self {
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
        let mut checks = BTreeMap::new();
        let mut previous = None;
        for (index, command) in guard_commands.iter().enumerate() {
            let id = format!("legacy-guard-{}", index + 1);
            checks.insert(
                id.clone(),
                WorkspaceCheck {
                    id: id.clone(),
                    label: command.clone(),
                    command: command.clone(),
                    cwd: ".".to_string(),
                    executor: executor.id.clone(),
                    components: vec![component.id.clone()],
                    trigger: CheckTrigger::Changed,
                    inputs: vec!["**".to_string()],
                    outputs: Vec::new(),
                    depends_on: previous.into_iter().collect(),
                    timeout_seconds: default_timeout_seconds(),
                },
            );
            previous = Some(id);
        }
        Self {
            version: WORKSPACE_CONFIG_VERSION,
            executors: BTreeMap::from([(executor.id.clone(), executor)]),
            components: BTreeMap::from([(component.id.clone(), component)]),
            checks,
            tasks: BTreeMap::new(),
            cargo_workspaces: BTreeMap::new(),
            discovery_warnings: Vec::new(),
            source: WorkspaceGraphSource::LegacyGuards,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceGraphSource {
    Explicit,
    LegacyGuards,
    Discovered,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Executor {
    pub id: String,
    #[serde(default)]
    pub kind: ExecutorKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutorKind {
    #[default]
    Project,
    Local,
    Container,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceComponent {
    pub id: String,
    #[serde(default = "dot_path")]
    pub root: String,
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
    pub executor: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceCheck {
    pub id: String,
    #[serde(default)]
    pub label: String,
    pub command: String,
    #[serde(default = "dot_path")]
    pub cwd: String,
    pub executor: String,
    #[serde(default)]
    pub components: Vec<String>,
    #[serde(default)]
    pub trigger: CheckTrigger,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<String>,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceTask {
    pub id: String,
    #[serde(default)]
    pub label: String,
    pub command: String,
    #[serde(default = "dot_path")]
    pub cwd: String,
    pub executor: String,
    #[serde(default)]
    pub allowed_changes: Vec<String>,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CargoWorkspace {
    pub id: String,
    pub root: String,
    pub manifest_path: String,
    pub members: Vec<String>,
    pub default_members: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CheckTrigger {
    #[default]
    Changed,
    Always,
    Needed,
}

fn unique_by_id<T, F>(kind: &str, items: Vec<T>, id: F) -> Result<BTreeMap<String, T>>
where
    F: Fn(&T) -> &String,
{
    let mut result = BTreeMap::new();
    for item in items {
        let key = id(&item).trim().to_string();
        if result.insert(key.clone(), item).is_some() {
            bail!("duplicate {kind} id '{key}'");
        }
    }
    Ok(result)
}

fn validate_id(kind: &str, id: &str) -> Result<()> {
    if id.is_empty()
        || !id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        bail!("{kind} id '{id}' must contain only ASCII letters, numbers, '-', '_' or '.'");
    }
    Ok(())
}

fn validate_relative_path(field: &str, raw: &str, allow_dot: bool) -> Result<()> {
    let path = Path::new(raw.trim());
    if path.as_os_str().is_empty() || path.is_absolute() {
        bail!("{field} must be a non-empty relative path");
    }
    let mut normal = false;
    for component in path.components() {
        match component {
            PathComponent::Normal(_) => normal = true,
            PathComponent::CurDir => {}
            PathComponent::ParentDir | PathComponent::RootDir | PathComponent::Prefix(_) => {
                bail!("{field} must stay inside the repository")
            }
        }
    }
    if !normal && !allow_dot {
        bail!("{field} must name a repository entry");
    }
    Ok(())
}

fn validate_patterns(field: &str, patterns: &[String]) -> Result<()> {
    for pattern in patterns {
        let normalized = pattern.replace('\\', "/");
        if normalized.starts_with('/') || normalized.split('/').any(|part| part == "..") {
            bail!("{field} pattern '{pattern}' must stay inside the repository");
        }
        Glob::new(&normalized).with_context(|| format!("invalid {field} pattern '{pattern}'"))?;
    }
    Ok(())
}

fn validate_references<T>(
    kind: &str,
    id: &str,
    dependencies: &[String],
    all: &BTreeMap<String, T>,
) -> Result<()> {
    let mut seen = HashSet::new();
    for dependency in dependencies {
        if dependency == id {
            bail!("{kind} '{id}' cannot depend on itself");
        }
        if !all.contains_key(dependency) {
            bail!("{kind} '{id}' references unknown dependency '{dependency}'");
        }
        if !seen.insert(dependency) {
            bail!("{kind} '{id}' repeats dependency '{dependency}'");
        }
    }
    Ok(())
}

fn validate_acyclic<'a>(
    kind: &str,
    entries: impl Iterator<Item = (&'a str, &'a [String])>,
) -> Result<()> {
    let graph = entries.collect::<BTreeMap<_, _>>();
    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    fn visit<'a>(
        id: &'a str,
        graph: &BTreeMap<&'a str, &'a [String]>,
        visiting: &mut HashSet<&'a str>,
        visited: &mut HashSet<&'a str>,
    ) -> bool {
        if visited.contains(id) {
            return false;
        }
        if !visiting.insert(id) {
            return true;
        }
        if graph.get(id).is_some_and(|dependencies| {
            dependencies
                .iter()
                .any(|dependency| visit(dependency, graph, visiting, visited))
        }) {
            return true;
        }
        visiting.remove(id);
        visited.insert(id);
        false
    }
    for id in graph.keys().copied() {
        if visit(id, &graph, &mut visiting, &mut visited) {
            bail!("{kind} dependency graph contains a cycle involving '{id}'");
        }
    }
    Ok(())
}

fn git_required(repo_root: &Path, args: &[&str]) -> Result<String> {
    String::from_utf8(git_bytes(repo_root, args)?)
        .context("git returned non-UTF-8 output")
        .map(|value| value.trim_end().to_string())
}

fn git_optional(repo_root: &Path, args: &[&str]) -> Result<Option<String>> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        return Ok(None);
    }
    Ok(Some(
        String::from_utf8(output.stdout)
            .context("git returned non-UTF-8 output")?
            .trim()
            .to_string(),
    ))
}

fn git_bytes(repo_root: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

fn dot_path() -> String {
    ".".to_string()
}

fn default_timeout_seconds() -> u64 {
    600
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proactive_snapshot_rejects_expired_deadlines() {
        let repo = tempfile::tempdir().unwrap();
        let error = ContentSnapshot::capture_until(repo.path(), Instant::now(), 1024)
            .unwrap_err()
            .to_string();
        assert!(error.contains("time bound expired"), "{error}");
    }

    #[test]
    fn proactive_snapshot_enforces_aggregate_file_byte_bound() {
        let repo = tempfile::tempdir().unwrap();
        Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(repo.path())
            .status()
            .unwrap();
        std::fs::write(repo.path().join("large.txt"), "four").unwrap();
        let error =
            ContentSnapshot::capture_until(repo.path(), Instant::now() + Duration::from_secs(2), 3)
                .unwrap_err()
                .to_string();
        assert!(error.contains("byte bound"), "{error}");
    }

    fn init_repo() -> tempfile::TempDir {
        let temp = tempfile::tempdir().unwrap();
        for args in [
            &["init", "--initial-branch=main"][..],
            &["config", "user.name", "pb workspace test"][..],
            &["config", "user.email", "workspace@pb.local"][..],
            &["commit", "--allow-empty", "-m", "test: initialize"][..],
        ] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(temp.path())
                    .status()
                    .unwrap()
                    .success()
            );
        }
        temp
    }

    #[test]
    fn repository_context_preserves_nested_focus_and_both_baselines() {
        let repo = init_repo();
        let focus = repo.path().join("services").join("api");
        std::fs::create_dir_all(&focus).unwrap();
        let context = RepositoryContext::capture(repo.path(), &focus).unwrap();

        assert_eq!(context.repo_root, repo.path().canonicalize().unwrap());
        assert_eq!(context.focus_root, focus.canonicalize().unwrap());
        assert_eq!(context.task_baseline, context.invocation_baseline);

        std::fs::write(focus.join("main.rs"), "fn main() {}\n").unwrap();
        assert_eq!(
            context.task_changed_paths().unwrap(),
            vec!["services/api/main.rs"]
        );
        std::fs::remove_file(focus.join("main.rs")).unwrap();
        assert!(context.task_changed_paths().unwrap().is_empty());
    }

    #[test]
    fn tracked_deletion_content_identity_survives_staging_and_commit() {
        let repo = init_repo();
        std::fs::write(repo.path().join("obsolete.txt"), "obsolete\n").unwrap();
        for args in [
            &["add", "obsolete.txt"][..],
            &["commit", "-m", "test: seed obsolete file"][..],
        ] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(repo.path())
                    .status()
                    .unwrap()
                    .success()
            );
        }
        let baseline = ContentSnapshot::capture(repo.path()).unwrap();

        std::fs::remove_file(repo.path().join("obsolete.txt")).unwrap();
        let deleted = ContentSnapshot::capture(repo.path()).unwrap();
        assert_ne!(deleted.fingerprint, baseline.fingerprint);
        assert!(!deleted.paths.contains_key("obsolete.txt"));

        for args in [
            &["add", "--all"][..],
            &["commit", "-m", "test: remove obsolete file"][..],
        ] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(repo.path())
                    .status()
                    .unwrap()
                    .success()
            );
        }
        assert_eq!(ContentSnapshot::capture(repo.path()).unwrap(), deleted);
    }

    #[test]
    fn explicit_polyglot_workspace_round_trips_and_validates() {
        let document: WorkspaceConfigDocument = toml::from_str(
            r#"
version = 1

[[executors]]
id = "rust"
kind = "local"

[[executors]]
id = "web"
kind = "container"
environment = "web"

[[components]]
id = "web"
root = "webui"
include = ["webui/**"]
executor = "web"

[[components]]
id = "app"
root = "."
include = ["src/**", "Cargo.toml"]
executor = "rust"
depends_on = ["web"]

[[checks]]
id = "web-bundle"
command = "deno task build:web"
executor = "web"
inputs = ["webui/src/**"]
outputs = ["webui/dist/**"]
trigger = "needed"

[[checks]]
id = "rust-tests"
command = "cargo test --all-targets"
executor = "rust"
inputs = ["src/**", "Cargo.toml"]
depends_on = ["web-bundle"]

[[tasks]]
id = "format-rust"
label = "Format Rust"
command = "cargo fmt"
executor = "rust"
allowed_changes = ["src/**"]
timeout_seconds = 30
"#,
        )
        .unwrap();
        let graph = document.clone().normalize().unwrap();
        assert_eq!(graph.components["app"].depends_on, vec!["web"]);
        assert_eq!(graph.checks["rust-tests"].depends_on, vec!["web-bundle"]);
        assert_eq!(graph.checks["web-bundle"].trigger, CheckTrigger::Needed);
        assert_eq!(graph.tasks["format-rust"].command, "cargo fmt");
        assert_eq!(graph.tasks["format-rust"].allowed_changes, vec!["src/**"]);

        let repo = init_repo();
        document.save(repo.path()).unwrap();
        let loaded = WorkspaceConfigDocument::load(repo.path()).unwrap().unwrap();
        assert_eq!(loaded, document);
    }

    #[test]
    fn legacy_guards_become_ordered_repository_checks() {
        let guards = vec!["deno task build:web".to_string(), "cargo test".to_string()];
        let graph = WorkspaceGraph::legacy(&guards);
        assert_eq!(graph.source, WorkspaceGraphSource::LegacyGuards);
        assert_eq!(graph.components.len(), 1);
        assert_eq!(graph.checks["legacy-guard-1"].command, guards[0]);
        assert_eq!(
            graph.checks["legacy-guard-2"].depends_on,
            vec!["legacy-guard-1"]
        );
    }

    #[test]
    fn workspace_config_rejects_escaping_paths_and_cycles() {
        let escaping: WorkspaceConfigDocument = toml::from_str(
            r#"
version = 1
[[executors]]
id = "local"
[[components]]
id = "bad"
root = "../outside"
executor = "local"
"#,
        )
        .unwrap();
        assert!(escaping.normalize().is_err());

        let cyclic: WorkspaceConfigDocument = toml::from_str(
            r#"
version = 1
[[executors]]
id = "local"
[[components]]
id = "a"
executor = "local"
depends_on = ["b"]
[[components]]
id = "b"
executor = "local"
depends_on = ["a"]
"#,
        )
        .unwrap();
        assert!(cyclic.normalize().is_err());
    }

    #[test]
    fn workspace_tasks_reject_unknown_executors_and_escaping_authority() {
        let unknown_executor: WorkspaceConfigDocument = toml::from_str(
            r#"
version = 1
[[executors]]
id = "local"
[[tasks]]
id = "format"
command = "cargo fmt"
executor = "missing"
"#,
        )
        .unwrap();
        assert!(unknown_executor.normalize().is_err());

        let escaping_changes: WorkspaceConfigDocument = toml::from_str(
            r#"
version = 1
[[executors]]
id = "local"
[[tasks]]
id = "generate"
command = "generate"
cwd = "../outside"
executor = "local"
allowed_changes = ["../outside/**"]
"#,
        )
        .unwrap();
        assert!(escaping_changes.normalize().is_err());
    }
}
