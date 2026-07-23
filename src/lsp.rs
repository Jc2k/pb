//! LSP server configuration and stdio-backed tool support.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const MAX_LSP_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_LSP_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_LSP_DOCUMENT_BYTES: u64 = 8 * 1024 * 1024;
const MAX_LSP_OPEN_DOCUMENTS: usize = 256;
const LSP_READ_TIMEOUT: Duration = Duration::from_secs(30);
const LSP_DIAGNOSTIC_TIMEOUT: Duration = Duration::from_secs(5);
const PROACTIVE_LSP_DIAGNOSTIC_TIMEOUT: Duration = Duration::from_secs(2);
const PROACTIVE_LSP_PASS_TIMEOUT: Duration = Duration::from_secs(12);
const LSP_DIAGNOSTIC_QUIET_PERIOD: Duration = Duration::from_millis(50);
const MAX_PROACTIVE_LSP_PATHS: usize = 8;
const MAX_PROACTIVE_LSP_CALLS: usize = 8;
const MAX_PROACTIVE_LSP_DIAGNOSTICS: usize = 64;
const MAX_PROACTIVE_LSP_MESSAGE_CHARS: usize = 500;
const MAX_PROACTIVE_LSP_SNAPSHOT_BYTES: u64 = 256 * 1024 * 1024;

pub const PROACTIVE_LSP_TOOL_NAME: &str = "lsp_proactive_diagnostics";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LspConfig {
    pub servers: BTreeMap<String, LspServerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LspServerConfig {
    pub command: Option<String>,
    pub container_image: Option<String>,
    pub source_container_image: Option<String>,
    pub verified_manifest_digest: Option<String>,
    pub container_runtime: Option<String>,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub working_directory: Option<PathBuf>,
    pub language_ids: Vec<String>,
    pub initialization_options: Option<Value>,
    #[serde(default = "default_lsp_workspace_access")]
    pub workspace_access: crate::session_environment::ServiceWorkspaceAccess,
    #[serde(default)]
    pub network_access: crate::session_environment::ServiceNetworkAccess,
    pub cache_ids: Vec<String>,
    pub disabled: bool,
}

impl Default for LspServerConfig {
    fn default() -> Self {
        Self {
            command: None,
            container_image: None,
            source_container_image: None,
            verified_manifest_digest: None,
            container_runtime: None,
            args: Vec::new(),
            env: BTreeMap::new(),
            working_directory: None,
            language_ids: Vec::new(),
            initialization_options: None,
            workspace_access: default_lsp_workspace_access(),
            network_access: crate::session_environment::ServiceNetworkAccess::None,
            cache_ids: Vec::new(),
            disabled: false,
        }
    }
}

fn default_lsp_workspace_access() -> crate::session_environment::ServiceWorkspaceAccess {
    crate::session_environment::ServiceWorkspaceAccess::ReadOnly
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectLspConfig {
    pub servers: BTreeMap<String, LspServerConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LspToolSpec {
    pub tool_name: String,
    pub server_name: String,
    pub operation: LspOperation,
    pub description: String,
    pub input_schema: Value,
    pub status_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LspOperation {
    Hover,
    Definition,
    References,
    DocumentSymbols,
    WorkspaceSymbols,
    Diagnostics,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ProactiveLspMode {
    Syntax,
    Settled,
}

impl ProactiveLspMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Syntax => "syntax",
            Self::Settled => "settled",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProactiveLspDiagnostic {
    pub server: String,
    pub path: String,
    pub path_fingerprint: String,
    pub line: u64,
    pub character: u64,
    pub severity: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProactiveLspFailure {
    pub server: String,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProactiveLspTarget {
    pub server: String,
    pub path: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ProactiveLspIncompleteReason {
    PathBound,
    CallBound,
    TimeBound,
    PathUnavailable,
    ServerFailure,
    PushOnlySnapshot,
    StaleWorkspace,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProactiveLspReport {
    pub mode: ProactiveLspMode,
    #[serde(default)]
    pub workspace_epoch: u64,
    pub workspace_fingerprint: String,
    pub requested_paths: Vec<String>,
    pub scanned_paths: Vec<String>,
    pub diagnostics: Vec<ProactiveLspDiagnostic>,
    pub suppressed_diagnostics: usize,
    #[serde(default)]
    pub omitted_paths: usize,
    pub failures: Vec<ProactiveLspFailure>,
    pub stale: bool,
    #[serde(default)]
    pub requested_targets: Vec<ProactiveLspTarget>,
    #[serde(default)]
    pub completed_targets: Vec<ProactiveLspTarget>,
    #[serde(default)]
    pub advisory_targets: Vec<ProactiveLspTarget>,
    #[serde(default)]
    pub deferred_targets: Vec<ProactiveLspTarget>,
    #[serde(default)]
    pub incomplete_reasons: BTreeSet<ProactiveLspIncompleteReason>,
    #[serde(default)]
    pub complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProactiveLspPathSelection {
    pub paths: Vec<String>,
    pub omitted_paths: usize,
}

impl ProactiveLspReport {
    pub fn blocking_paths(&self) -> BTreeSet<String> {
        self.diagnostics
            .iter()
            .map(|diagnostic| diagnostic.path.clone())
            .collect()
    }

    pub fn completed_paths(&self) -> BTreeSet<String> {
        self.paths_accounted_by(&self.completed_targets)
    }

    pub fn attempted_paths(&self) -> BTreeSet<String> {
        let mut targets = self.completed_targets.clone();
        targets.extend(self.advisory_targets.iter().cloned());
        self.paths_accounted_by(&targets)
    }

    fn paths_accounted_by(&self, targets: &[ProactiveLspTarget]) -> BTreeSet<String> {
        let requested = self.requested_targets.iter().fold(
            BTreeMap::<&str, usize>::new(),
            |mut counts, target| {
                *counts.entry(&target.path).or_default() += 1;
                counts
            },
        );
        let completed =
            targets
                .iter()
                .fold(BTreeMap::<&str, usize>::new(), |mut counts, target| {
                    *counts.entry(&target.path).or_default() += 1;
                    counts
                });
        requested
            .into_iter()
            .filter(|(path, count)| completed.get(path).copied() == Some(*count))
            .map(|(path, _)| path.to_string())
            .collect()
    }

    pub fn defer_omitted_paths(&mut self, omitted_paths: usize) {
        self.omitted_paths = self.omitted_paths.saturating_add(omitted_paths);
        if omitted_paths > 0 {
            self.incomplete_reasons
                .insert(ProactiveLspIncompleteReason::PathBound);
            self.complete = false;
        }
    }
}

#[derive(Clone, Default)]
pub struct LspToolRegistry {
    pub servers: BTreeMap<String, LspServerConfig>,
    pub tools: BTreeMap<String, LspToolSpec>,
    sessions: BTreeMap<String, Arc<Mutex<Option<LspClient>>>>,
    lease: Option<Arc<crate::session_environment::SessionEnvironmentLease>>,
}

impl LspToolRegistry {
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
    pub fn tool(&self, name: &str) -> Option<&LspToolSpec> {
        self.tools.get(name)
    }
}

impl ProjectLspConfig {
    pub fn load(workspace_root: &Path) -> Result<Option<Self>> {
        let path = project_lsp_config_path(workspace_root);
        if !path.exists() {
            return Ok(None);
        }
        let text = read_bounded_utf8(&path, MAX_LSP_CONFIG_BYTES, "LSP config")?;
        toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn save(&self, workspace_root: &Path) -> Result<()> {
        let path = project_lsp_config_path(workspace_root);
        let text =
            toml::to_string_pretty(self).context("failed to serialize project LSP config")?;
        crate::atomic_file::write(&path, text.as_bytes())
    }
}

pub fn project_lsp_config_path(workspace_root: &Path) -> PathBuf {
    workspace_root.join(".pb").join("lsp.toml")
}

fn read_bounded_utf8(path: &Path, max_bytes: u64, label: &str) -> Result<String> {
    let file =
        File::open(path).with_context(|| format!("failed to open {label} {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to stat {label} {}", path.display()))?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        bail!(
            "{label} {} exceeds the {max_bytes}-byte input bound",
            path.display()
        );
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {label} {}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        bail!(
            "{label} {} grew beyond the {max_bytes}-byte input bound",
            path.display()
        );
    }
    String::from_utf8(bytes).with_context(|| format!("{label} is not valid UTF-8"))
}

fn read_bounded_utf8_until(
    path: &Path,
    max_bytes: u64,
    label: &str,
    deadline: Instant,
) -> Result<String> {
    if Instant::now() >= deadline {
        bail!("timed out before reading {label} {}", path.display());
    }
    let mut file =
        File::open(path).with_context(|| format!("failed to open {label} {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to stat {label} {}", path.display()))?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        bail!(
            "{label} {} exceeds the {max_bytes}-byte input bound",
            path.display()
        );
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        if Instant::now() >= deadline {
            bail!("timed out reading {label} {}", path.display());
        }
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {label} {}", path.display()))?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() as u64 > max_bytes {
            bail!(
                "{label} {} grew beyond the {max_bytes}-byte input bound",
                path.display()
            );
        }
    }
    String::from_utf8(bytes).with_context(|| format!("{label} is not valid UTF-8"))
}

pub fn effective_servers(
    global: &LspConfig,
    project: Option<&ProjectLspConfig>,
) -> BTreeMap<String, LspServerConfig> {
    let mut servers = global.servers.clone();
    if let Some(project) = project {
        for (name, config) in &project.servers {
            if config.disabled {
                servers.remove(name);
            } else {
                servers.insert(name.clone(), config.clone());
            }
        }
    }
    servers
        .into_iter()
        .filter(|(_, c)| {
            !c.disabled
                && (c.command.as_deref().is_some_and(|s| !s.trim().is_empty())
                    || c.container_image
                        .as_deref()
                        .is_some_and(|s| !s.trim().is_empty()))
        })
        .collect()
}

pub fn discover_tools(
    servers: BTreeMap<String, LspServerConfig>,
    workspace_root: &Path,
) -> LspToolRegistry {
    discover_tools_inner(servers, workspace_root, None)
}

pub fn discover_tools_with_lease(
    servers: BTreeMap<String, LspServerConfig>,
    workspace_root: &Path,
    lease: Arc<crate::session_environment::SessionEnvironmentLease>,
) -> LspToolRegistry {
    discover_tools_inner(servers, workspace_root, Some(lease))
}

struct CachedLspRegistry {
    fingerprint: String,
    registry: LspToolRegistry,
}

static SESSION_LSP_REGISTRIES: OnceLock<Mutex<HashMap<String, CachedLspRegistry>>> =
    OnceLock::new();

pub fn discover_tools_for_session(
    session_id: &str,
    servers: BTreeMap<String, LspServerConfig>,
    workspace_root: &Path,
    lease: Arc<crate::session_environment::SessionEnvironmentLease>,
) -> LspToolRegistry {
    let fingerprint = crate::environment_lock::sha256(
        &serde_json::to_vec(&(workspace_root, &servers)).unwrap_or_default(),
    );
    let registries = SESSION_LSP_REGISTRIES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut registries) = registries.lock() {
        if let Some(cached) = registries.get(session_id)
            && cached.fingerprint == fingerprint
        {
            return cached.registry.clone();
        }
        let registry = discover_tools_inner(servers, workspace_root, Some(lease));
        registries.insert(
            session_id.to_string(),
            CachedLspRegistry {
                fingerprint,
                registry: registry.clone(),
            },
        );
        return registry;
    }
    LspToolRegistry::default()
}

pub fn shutdown_session_services(session_id: &str) {
    if let Some(registries) = SESSION_LSP_REGISTRIES.get()
        && let Ok(mut registries) = registries.lock()
    {
        registries.remove(session_id);
    }
}

fn discover_tools_inner(
    servers: BTreeMap<String, LspServerConfig>,
    _workspace_root: &Path,
    lease: Option<Arc<crate::session_environment::SessionEnvironmentLease>>,
) -> LspToolRegistry {
    let mut registry = LspToolRegistry {
        servers: servers.clone(),
        tools: BTreeMap::new(),
        sessions: BTreeMap::new(),
        lease: lease.clone(),
    };
    let mut collided_names = BTreeSet::new();
    for (server_name, _config) in servers {
        registry
            .sessions
            .insert(server_name.clone(), Arc::new(Mutex::new(None)));
        for op in [
            LspOperation::Hover,
            LspOperation::Definition,
            LspOperation::References,
            LspOperation::DocumentSymbols,
            LspOperation::WorkspaceSymbols,
            LspOperation::Diagnostics,
        ] {
            let tool_name = unique_tool_name(&server_name, op.name());
            if collided_names.contains(&tool_name) {
                continue;
            }
            if let Some(existing) = registry.tools.remove(&tool_name) {
                collided_names.insert(tool_name.clone());
                insert_status_tool(
                    &mut registry,
                    &server_name,
                    &format!(
                        "LSP tool-name collision: server '{}' and server '{}' both normalize operation '{}' to '{tool_name}'; neither tool was exposed",
                        existing.server_name,
                        server_name,
                        op.name(),
                    ),
                );
                continue;
            }
            registry.tools.insert(
                tool_name.clone(),
                LspToolSpec {
                    tool_name,
                    server_name: server_name.clone(),
                    operation: op,
                    description: op.description(&server_name),
                    input_schema: op.input_schema(),
                    status_error: None,
                },
            );
        }
    }
    registry
}

fn insert_status_tool(registry: &mut LspToolRegistry, server_name: &str, description: &str) {
    let digest = crate::environment_lock::sha256(description.as_bytes());
    let tool_name = format!("lsp_status_{}_status", &digest[..12]);
    registry.tools.insert(
        tool_name.clone(),
        LspToolSpec {
            tool_name,
            server_name: server_name.to_string(),
            operation: LspOperation::Diagnostics,
            description: description.to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            status_error: Some(description.to_string()),
        },
    );
}

pub fn call_tool(
    registry: &LspToolRegistry,
    workspace_root: &Path,
    tool_name: &str,
    arguments: &Value,
) -> Result<String> {
    call_tool_with_diagnostic_timeout(
        registry,
        workspace_root,
        tool_name,
        arguments,
        LSP_DIAGNOSTIC_TIMEOUT,
        None,
    )
    .map(|outcome| outcome.result)
}

#[derive(Debug)]
struct LspToolCallOutcome {
    result: String,
    diagnostic_complete: bool,
}

fn complete_tool_outcome(result: String) -> LspToolCallOutcome {
    LspToolCallOutcome {
        result,
        diagnostic_complete: true,
    }
}

fn call_tool_with_diagnostic_timeout(
    registry: &LspToolRegistry,
    workspace_root: &Path,
    tool_name: &str,
    arguments: &Value,
    diagnostic_timeout: Duration,
    overall_deadline: Option<Instant>,
) -> Result<LspToolCallOutcome> {
    let spec = registry
        .tool(tool_name)
        .with_context(|| format!("unknown LSP tool: {tool_name}"))?;
    if let Some(error) = &spec.status_error {
        bail!(error.clone());
    }
    let session = registry
        .sessions
        .get(&spec.server_name)
        .with_context(|| format!("LSP server '{}' is not running", spec.server_name))?;
    let mut slot = session
        .lock()
        .map_err(|_| anyhow!("LSP session for {} is poisoned", spec.server_name))?;
    if slot.is_none() {
        let config = registry
            .servers
            .get(&spec.server_name)
            .context("LSP server configuration disappeared during lazy startup")?;
        let initialize_timeout = remaining_timeout(
            overall_deadline,
            LSP_READ_TIMEOUT,
            "starting the language server",
        )?;
        *slot = Some(LspClient::connect(
            &spec.server_name,
            config,
            workspace_root,
            registry.lease.as_ref(),
            initialize_timeout,
        )?);
    }
    let client = slot
        .as_mut()
        .context("LSP server disappeared after lazy startup")?;
    let diagnostic_timeout = remaining_timeout(
        overall_deadline,
        diagnostic_timeout,
        "collecting language-server diagnostics",
    )?;
    match client.call_with_diagnostic_timeout(
        spec.operation,
        workspace_root,
        arguments,
        diagnostic_timeout,
    ) {
        Ok(result) => Ok(result),
        Err(first_error) => {
            if !recoverable_transport_error(&first_error) {
                return Err(first_error);
            }
            let config = registry
                .servers
                .get(&spec.server_name)
                .context("LSP server configuration disappeared during restart")?;
            let shutdown_timeout = overall_deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now()))
                .unwrap_or(Duration::from_secs(2))
                .min(Duration::from_secs(2));
            client.shutdown_transport(shutdown_timeout);
            let initialize_timeout = remaining_timeout(
                overall_deadline,
                LSP_READ_TIMEOUT,
                "restarting the language server",
            )?;
            *client = LspClient::connect(
                &spec.server_name,
                config,
                workspace_root,
                registry.lease.as_ref(),
                initialize_timeout,
            )
            .with_context(|| {
                format!("LSP request failed ({first_error:#}) and the bounded restart also failed")
            })?;
            let diagnostic_timeout = remaining_timeout(
                overall_deadline,
                diagnostic_timeout,
                "collecting diagnostics after language-server restart",
            )?;
            client.call_with_diagnostic_timeout(
                spec.operation,
                workspace_root,
                arguments,
                diagnostic_timeout,
            )
        }
    }
}

fn remaining_timeout(
    overall_deadline: Option<Instant>,
    requested: Duration,
    operation: &str,
) -> Result<Duration> {
    let Some(deadline) = overall_deadline else {
        return Ok(requested);
    };
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        bail!("proactive LSP time bound expired before {operation}");
    }
    Ok(requested.min(remaining))
}

pub fn proactive_diagnostics(
    registry: &LspToolRegistry,
    workspace_root: &Path,
    paths: impl IntoIterator<Item = String>,
    mode: ProactiveLspMode,
    workspace_epoch: u64,
) -> Result<ProactiveLspReport> {
    let pass_deadline = proactive_pass_deadline();
    let before = proactive_workspace_snapshot(workspace_root, pass_deadline)?;
    proactive_diagnostics_until(
        registry,
        workspace_root,
        paths,
        mode,
        workspace_epoch,
        before,
        pass_deadline,
    )
}

pub(crate) fn proactive_pass_deadline() -> Instant {
    Instant::now() + PROACTIVE_LSP_PASS_TIMEOUT
}

pub(crate) fn proactive_workspace_snapshot(
    workspace_root: &Path,
    deadline: Instant,
) -> Result<crate::workspace::ContentSnapshot> {
    crate::workspace::ContentSnapshot::capture_until(
        workspace_root,
        deadline,
        MAX_PROACTIVE_LSP_SNAPSHOT_BYTES,
    )
}

pub(crate) fn proactive_diagnostics_until(
    registry: &LspToolRegistry,
    workspace_root: &Path,
    paths: impl IntoIterator<Item = String>,
    mode: ProactiveLspMode,
    workspace_epoch: u64,
    before: crate::workspace::ContentSnapshot,
    pass_deadline: Instant,
) -> Result<ProactiveLspReport> {
    let all_paths = paths
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let omitted_paths = all_paths.len().saturating_sub(MAX_PROACTIVE_LSP_PATHS);
    let requested_paths = all_paths
        .into_iter()
        .take(MAX_PROACTIVE_LSP_PATHS)
        .collect::<Vec<_>>();
    let mut server_paths = BTreeMap::<String, Vec<String>>::new();
    let mut requested_targets = Vec::new();
    let mut unavailable_paths = Vec::new();
    for path in &requested_paths {
        let Some(path_state) = before.paths.get(path).filter(|state| state.kind == "file") else {
            unavailable_paths.push(path.clone());
            continue;
        };
        let Some(language_id) = inferred_language_id(Path::new(path)) else {
            unavailable_paths.push(path.clone());
            continue;
        };
        let target_count_before = requested_targets.len();
        for (server, config) in &registry.servers {
            if config
                .language_ids
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(language_id))
            {
                let _ = path_state;
                server_paths
                    .entry(server.clone())
                    .or_default()
                    .push(path.clone());
                requested_targets.push(ProactiveLspTarget {
                    server: server.clone(),
                    path: path.clone(),
                });
            }
        }
        if requested_targets.len() == target_count_before {
            unavailable_paths.push(path.clone());
        }
    }
    let mut report = ProactiveLspReport {
        mode,
        workspace_epoch,
        workspace_fingerprint: before.fingerprint.clone(),
        requested_paths: requested_paths.clone(),
        scanned_paths: Vec::new(),
        diagnostics: Vec::new(),
        suppressed_diagnostics: 0,
        omitted_paths,
        failures: Vec::new(),
        stale: false,
        requested_targets: requested_targets.clone(),
        completed_targets: Vec::new(),
        advisory_targets: Vec::new(),
        deferred_targets: Vec::new(),
        incomplete_reasons: if omitted_paths > 0 {
            BTreeSet::from([ProactiveLspIncompleteReason::PathBound])
        } else {
            BTreeSet::new()
        },
        complete: false,
    };
    if !unavailable_paths.is_empty() {
        report
            .incomplete_reasons
            .insert(ProactiveLspIncompleteReason::PathUnavailable);
        report
            .failures
            .extend(unavailable_paths.into_iter().map(|path| ProactiveLspFailure {
                server: "pb".to_string(),
                path,
                message: "selected path is missing, not a regular file, or no longer supported by a configured server".to_string(),
            }));
    }
    for (index, target) in requested_targets.iter().enumerate() {
        if index >= MAX_PROACTIVE_LSP_CALLS {
            report
                .deferred_targets
                .extend_from_slice(&requested_targets[index..]);
            report
                .incomplete_reasons
                .insert(ProactiveLspIncompleteReason::CallBound);
            break;
        }
        if Instant::now() >= pass_deadline {
            report
                .deferred_targets
                .extend_from_slice(&requested_targets[index..]);
            report
                .incomplete_reasons
                .insert(ProactiveLspIncompleteReason::TimeBound);
            break;
        }
        let server = &target.server;
        let path = &target.path;
        let path_state = &before.paths[path];
        let Some(spec) = registry.tools.values().find(|spec| {
            spec.server_name == *server && spec.operation == LspOperation::Diagnostics
        }) else {
            report.failures.push(ProactiveLspFailure {
                server: server.clone(),
                path: path.clone(),
                message: "diagnostics operation is unavailable".to_string(),
            });
            report
                .incomplete_reasons
                .insert(ProactiveLspIncompleteReason::ServerFailure);
            continue;
        };
        if let Some(error) = spec.status_error.as_deref() {
            report.failures.push(ProactiveLspFailure {
                server: server.clone(),
                path: path.clone(),
                message: bounded_proactive_text(error),
            });
            report
                .incomplete_reasons
                .insert(ProactiveLspIncompleteReason::ServerFailure);
            continue;
        }
        let result = call_tool_with_diagnostic_timeout(
            registry,
            workspace_root,
            &spec.tool_name,
            &json!({
                "path": path,
                "_pb_workspace_paths": server_paths.get(server).cloned().unwrap_or_default(),
            }),
            PROACTIVE_LSP_DIAGNOSTIC_TIMEOUT,
            Some(pass_deadline),
        );
        match result {
            Ok(outcome) => {
                if !report.scanned_paths.contains(&path) {
                    report.scanned_paths.push(path.clone());
                }
                let decoded: Value = match serde_json::from_str(&outcome.result) {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        report.failures.push(ProactiveLspFailure {
                            server: server.clone(),
                            path: path.clone(),
                            message: format!("invalid diagnostics response: {error}"),
                        });
                        report
                            .incomplete_reasons
                            .insert(ProactiveLspIncompleteReason::ServerFailure);
                        continue;
                    }
                };
                let Some(diagnostics) = decoded.as_array() else {
                    report.failures.push(ProactiveLspFailure {
                        server: server.clone(),
                        path: path.clone(),
                        message: "diagnostics response was not an array".to_string(),
                    });
                    report
                        .incomplete_reasons
                        .insert(ProactiveLspIncompleteReason::ServerFailure);
                    continue;
                };
                if outcome.diagnostic_complete {
                    report.completed_targets.push(target.clone());
                } else {
                    report.advisory_targets.push(target.clone());
                    report
                        .incomplete_reasons
                        .insert(ProactiveLspIncompleteReason::PushOnlySnapshot);
                }
                for diagnostic in diagnostics {
                    let Some(normalized) = normalize_proactive_diagnostic(
                        &server,
                        &path,
                        &path_state.fingerprint,
                        diagnostic,
                    ) else {
                        report.suppressed_diagnostics =
                            report.suppressed_diagnostics.saturating_add(1);
                        continue;
                    };
                    let admitted = match mode {
                        ProactiveLspMode::Syntax => {
                            proactive_diagnostic_is_syntax(diagnostic, &normalized)
                        }
                        ProactiveLspMode::Settled => normalized.severity == 1,
                    };
                    if !admitted || report.diagnostics.len() >= MAX_PROACTIVE_LSP_DIAGNOSTICS {
                        report.suppressed_diagnostics =
                            report.suppressed_diagnostics.saturating_add(1);
                        continue;
                    }
                    report.diagnostics.push(normalized);
                }
            }
            Err(error) => {
                report.failures.push(ProactiveLspFailure {
                    server: server.clone(),
                    path: path.clone(),
                    message: bounded_proactive_text(&format!("{error:#}")),
                });
                report
                    .incomplete_reasons
                    .insert(ProactiveLspIncompleteReason::ServerFailure);
            }
        }
    }

    match proactive_workspace_snapshot(workspace_root, pass_deadline) {
        Ok(after) if after.fingerprint == before.fingerprint => {}
        Ok(_) => {
            report.stale = true;
            report.diagnostics.clear();
            report.completed_targets.clear();
            report
                .incomplete_reasons
                .insert(ProactiveLspIncompleteReason::StaleWorkspace);
            report.failures.push(ProactiveLspFailure {
                server: "pb".to_string(),
                path: ".".to_string(),
                message:
                    "workspace content changed during proactive LSP collection; all diagnostics were discarded"
                        .to_string(),
            });
        }
        Err(error) => {
            report.stale = true;
            report.diagnostics.clear();
            report.completed_targets.clear();
            report
                .incomplete_reasons
                .insert(ProactiveLspIncompleteReason::TimeBound);
            report.failures.push(ProactiveLspFailure {
                server: "pb".to_string(),
                path: ".".to_string(),
                message: bounded_proactive_text(&format!(
                    "workspace could not be revalidated within the proactive bound: {error:#}"
                )),
            });
        }
    }
    report.scanned_paths.sort();
    report.scanned_paths.dedup();
    report.diagnostics.sort();
    report.diagnostics.dedup();
    report.failures.sort();
    report.failures.dedup();
    report.completed_targets.sort();
    report.completed_targets.dedup();
    report.advisory_targets.sort();
    report.advisory_targets.dedup();
    report.deferred_targets.sort();
    report.deferred_targets.dedup();
    report.complete = report.omitted_paths == 0
        && !report.stale
        && report.failures.is_empty()
        && report.deferred_targets.is_empty()
        && report.completed_targets.len() == report.requested_targets.len();
    Ok(report)
}

pub fn proactive_supported_paths(
    registry: &LspToolRegistry,
    paths: impl IntoIterator<Item = String>,
) -> ProactiveLspPathSelection {
    let supported = paths
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|path| {
            inferred_language_id(Path::new(path)).is_some_and(|language_id| {
                registry.servers.values().any(|config| {
                    config
                        .language_ids
                        .iter()
                        .any(|candidate| candidate.eq_ignore_ascii_case(language_id))
                })
            })
        })
        .collect::<Vec<_>>();
    ProactiveLspPathSelection {
        omitted_paths: supported.len().saturating_sub(MAX_PROACTIVE_LSP_PATHS),
        paths: supported
            .into_iter()
            .take(MAX_PROACTIVE_LSP_PATHS)
            .collect(),
    }
}

fn normalize_proactive_diagnostic(
    server: &str,
    path: &str,
    path_fingerprint: &str,
    diagnostic: &Value,
) -> Option<ProactiveLspDiagnostic> {
    let message = diagnostic.get("message")?.as_str()?.trim();
    if message.is_empty() {
        return None;
    }
    let severity = diagnostic.get("severity").and_then(Value::as_u64)?;
    let start = diagnostic.pointer("/range/start")?;
    let line = start.get("line").and_then(Value::as_u64)?.saturating_add(1);
    let character = start.get("character").and_then(Value::as_u64).unwrap_or(0);
    let code = diagnostic.get("code").and_then(|value| match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    });
    Some(ProactiveLspDiagnostic {
        server: server.to_string(),
        path: path.to_string(),
        path_fingerprint: path_fingerprint.to_string(),
        line,
        character,
        severity,
        code,
        message: bounded_proactive_text(message),
    })
}

fn proactive_diagnostic_is_syntax(raw: &Value, diagnostic: &ProactiveLspDiagnostic) -> bool {
    if diagnostic.severity != 1 {
        return false;
    }
    let code = diagnostic
        .code
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let source = raw
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    code.contains("syntax")
        || code.contains("parse")
        || source.contains("parser")
        || diagnostic
            .message
            .to_ascii_lowercase()
            .contains("syntax error")
}

fn bounded_proactive_text(value: &str) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= MAX_PROACTIVE_LSP_MESSAGE_CHARS {
        normalized
    } else {
        let mut bounded = normalized
            .chars()
            .take(MAX_PROACTIVE_LSP_MESSAGE_CHARS.saturating_sub(1))
            .collect::<String>();
        bounded.push('…');
        bounded
    }
}

#[derive(Debug)]
struct LspTransportFailure(String);

impl std::fmt::Display for LspTransportFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for LspTransportFailure {}

fn lsp_transport_failure(message: impl Into<String>) -> anyhow::Error {
    anyhow!(LspTransportFailure(message.into()))
}

fn recoverable_transport_error(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<LspTransportFailure>().is_some())
}

impl LspOperation {
    fn name(self) -> &'static str {
        match self {
            Self::Hover => "hover",
            Self::Definition => "definition",
            Self::References => "references",
            Self::DocumentSymbols => "document_symbols",
            Self::WorkspaceSymbols => "workspace_symbols",
            Self::Diagnostics => "diagnostics",
        }
    }
    fn description(self, server: &str) -> String {
        format!(
            "Query the {server} language server for {} information over LSP stdio.",
            self.name().replace('_', " ")
        )
    }
    fn input_schema(self) -> Value {
        match self {
            Self::WorkspaceSymbols => {
                json!({"type":"object","properties":{"query":{"type":"string","description":"Symbol query text.","maxLength":4096}},"required":["query"],"additionalProperties":false})
            }
            Self::DocumentSymbols | Self::Diagnostics => {
                json!({"type":"object","properties":{"path":{"type":"string","description":"Project-relative file path.","maxLength":4096}},"required":["path"],"additionalProperties":false})
            }
            _ => {
                json!({"type":"object","properties":{"path":{"type":"string","description":"Project-relative file path.","maxLength":4096},"line":{"type":"integer","description":"1-indexed line number.","minimum":1,"maximum":10000000},"character":{"type":"integer","description":"0-indexed UTF-16 character offset.","minimum":0,"maximum":10000000}},"required":["path","line","character"],"additionalProperties":false})
            }
        }
    }
}

struct LspClient {
    process: LspProcess,
    stdin_writer: LspStdinWriter,
    stdout_reader: crate::jsonrpc::FramedJsonReader,
    next_id: u64,
    diagnostic_serial: u64,
    diagnostics: BTreeMap<String, DiagnosticSnapshot>,
    supports_pull_diagnostics: bool,
    language_ids: Vec<String>,
    root_uri: String,
    root_name: String,
    open_documents: BTreeMap<String, OpenDocument>,
    stderr_tail: Arc<Mutex<VecDeque<u8>>>,
    stderr_thread: Option<std::thread::JoinHandle<()>>,
    shutdown_timeout: Duration,
    transport_shutdown: bool,
    active_deadline: Option<Instant>,
}

struct LspWriteCommand {
    frame: Vec<u8>,
    completion: SyncSender<std::result::Result<(), String>>,
}

struct LspStdinWriter {
    sender: Option<SyncSender<LspWriteCommand>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl LspStdinWriter {
    fn spawn(mut stdin: ChildStdin) -> Self {
        let (sender, receiver) = sync_channel::<LspWriteCommand>(1);
        let thread = std::thread::spawn(move || {
            while let Ok(command) = receiver.recv() {
                let result = stdin
                    .write_all(&command.frame)
                    .and_then(|()| stdin.flush())
                    .map_err(|error| error.to_string());
                let failed = result.is_err();
                let _ = command.completion.send(result);
                if failed {
                    break;
                }
            }
        });
        Self {
            sender: Some(sender),
            thread: Some(thread),
        }
    }

    fn write_until(&self, message: &Value, deadline: Instant) -> Result<()> {
        let body = serde_json::to_vec(message)?;
        let mut frame = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
        frame.extend_from_slice(&body);
        let (completion, completed) = sync_channel(1);
        let mut command = LspWriteCommand { frame, completion };
        let sender = self
            .sender
            .as_ref()
            .context("LSP stdin writer is already shut down")?;
        loop {
            match sender.try_send(command) {
                Ok(()) => break,
                Err(TrySendError::Full(returned)) => {
                    command = returned;
                    if Instant::now() >= deadline {
                        bail!("timed out queueing an LSP request");
                    }
                    std::thread::sleep(
                        Duration::from_millis(2)
                            .min(deadline.saturating_duration_since(Instant::now())),
                    );
                }
                Err(TrySendError::Disconnected(_)) => {
                    bail!("LSP stdin writer stopped unexpectedly")
                }
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!("timed out writing an LSP request");
        }
        completed
            .recv_timeout(remaining)
            .map_err(|_| anyhow!("timed out writing an LSP request"))?
            .map_err(anyhow::Error::msg)
    }

    fn join(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct OpenDocument {
    version: u64,
    content_sha256: String,
}

struct DiagnosticSnapshot {
    version: Option<u64>,
    serial: u64,
    diagnostics: Value,
}

struct OpenDocumentState {
    uri: String,
    version: u64,
}

enum LspProcess {
    Host(Child),
    Exec(crate::container::ManagedProcess),
    Service(crate::container::ManagedServiceProcess),
}

struct LspTransport {
    process: LspProcess,
    stdin: ChildStdin,
    stdout: ChildStdout,
    stderr: Option<ChildStderr>,
}

impl LspProcess {
    fn try_wait(&mut self) -> Result<Option<std::process::ExitStatus>> {
        match self {
            Self::Host(child) => Ok(child.try_wait()?),
            Self::Exec(process) => process.try_wait(),
            Self::Service(process) => process.try_wait(),
        }
    }

    fn shutdown(&mut self, timeout: Duration) {
        match self {
            Self::Host(child) => {
                terminate_host_process_group(child.id());
                let _ = child.kill();
                let _ = child.wait();
            }
            Self::Exec(process) => {
                let _ = process.shutdown(timeout);
            }
            Self::Service(process) => {
                let _ = process.shutdown(timeout);
            }
        }
    }
}

impl LspClient {
    fn connect(
        server_name: &str,
        config: &LspServerConfig,
        workspace_root: &Path,
        lease: Option<&Arc<crate::session_environment::SessionEnvironmentLease>>,
        initialize_timeout: Duration,
    ) -> Result<Self> {
        let initialize_deadline = Instant::now() + initialize_timeout;
        validate_packaged_image_authority(config)?;
        let cwd = match config.working_directory.as_deref() {
            Some(path) => resolve_workspace_path(workspace_root, &path.to_string_lossy())?,
            None => workspace_root.to_path_buf(),
        };
        let LspTransport {
            process,
            stdin,
            stdout,
            stderr,
        } = spawn_lsp_transport_until(
            server_name,
            config,
            workspace_root,
            &cwd,
            lease,
            initialize_deadline,
        )?;
        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        let stderr_thread = stderr.map(|stderr| drain_stderr(stderr, Arc::clone(&stderr_tail)));
        let stdout_reader =
            crate::jsonrpc::FramedJsonReader::spawn("LSP", stdout, MAX_LSP_RESPONSE_BYTES)?;
        let root_uri = path_to_uri(workspace_root)?;
        let root_name = workspace_root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("workspace")
            .to_string();
        let mut client = Self {
            process,
            stdin_writer: LspStdinWriter::spawn(stdin),
            stdout_reader,
            next_id: 1,
            diagnostic_serial: 0,
            diagnostics: BTreeMap::new(),
            supports_pull_diagnostics: false,
            language_ids: config.language_ids.clone(),
            root_uri: root_uri.clone(),
            root_name: root_name.clone(),
            open_documents: BTreeMap::new(),
            stderr_tail,
            stderr_thread,
            shutdown_timeout: Duration::ZERO,
            transport_shutdown: false,
            active_deadline: Some(initialize_deadline),
        };
        let mut initialize_params = json!({"processId": std::process::id(), "rootUri": root_uri, "workspaceFolders": [{"uri": root_uri, "name": root_name}], "capabilities": {"textDocument":{"synchronization":{"didSave":true,"dynamicRegistration":false},"hover":{},"definition":{},"references":{},"documentSymbol":{},"publishDiagnostics":{},"diagnostic":{"dynamicRegistration":false,"relatedDocumentSupport":false}},"workspace":{"symbol":{},"workspaceFolders":true,"configuration":true,"diagnostics":{"refreshSupport":false}}}});
        if let Some(options) = config.initialization_options.clone()
            && let Some(params) = initialize_params.as_object_mut()
        {
            params.insert("initializationOptions".to_string(), options);
        }
        let initialize_result =
            client.request_with_timeout("initialize", initialize_params, initialize_timeout)?;
        client.supports_pull_diagnostics = server_supports_pull_diagnostics(&initialize_result);
        client.notify("initialized", json!({}))?;
        client.active_deadline = None;
        client.shutdown_timeout = Duration::from_secs(2);
        Ok(client)
    }

    fn call_with_diagnostic_timeout(
        &mut self,
        op: LspOperation,
        workspace_root: &Path,
        args: &Value,
        diagnostic_timeout: Duration,
    ) -> Result<LspToolCallOutcome> {
        let operation_deadline = Instant::now() + diagnostic_timeout;
        let previous_deadline = self.active_deadline.replace(operation_deadline);
        let result =
            self.call_with_diagnostic_timeout_inner(op, workspace_root, args, diagnostic_timeout);
        self.active_deadline = previous_deadline;
        result
    }

    fn call_with_diagnostic_timeout_inner(
        &mut self,
        op: LspOperation,
        workspace_root: &Path,
        args: &Value,
        diagnostic_timeout: Duration,
    ) -> Result<LspToolCallOutcome> {
        match op {
            LspOperation::WorkspaceSymbols => self
                .request_text(
                    "workspace/symbol",
                    json!({"query": string_arg(args, "query")?}),
                )
                .map(complete_tool_outcome),
            LspOperation::DocumentSymbols => {
                let document = self.open_document(workspace_root, args, false)?;
                self.request_text(
                    "textDocument/documentSymbol",
                    json!({"textDocument":{"uri":document.uri}}),
                )
                .map(complete_tool_outcome)
            }
            LspOperation::Diagnostics => {
                if let Some(paths) = args.get("_pb_workspace_paths").and_then(Value::as_array) {
                    for path in paths.iter().filter_map(Value::as_str) {
                        self.open_document(workspace_root, &json!({"path": path}), false)?;
                    }
                }
                let after_serial = self.diagnostic_serial;
                let document = self.open_document(workspace_root, args, true)?;
                if self.supports_pull_diagnostics {
                    let diagnostics = self.pull_diagnostics(&document, diagnostic_timeout)?;
                    Ok(LspToolCallOutcome {
                        result: diagnostics.to_string(),
                        diagnostic_complete: true,
                    })
                } else {
                    Ok(LspToolCallOutcome {
                        result: self
                            .wait_for_diagnostics(&document, diagnostic_timeout, after_serial)?
                            .to_string(),
                        diagnostic_complete: false,
                    })
                }
            }
            LspOperation::Hover => self
                .position_request(workspace_root, args, "textDocument/hover")
                .map(complete_tool_outcome),
            LspOperation::Definition => self
                .position_request(workspace_root, args, "textDocument/definition")
                .map(complete_tool_outcome),
            LspOperation::References => {
                let (uri, pos) = self.document_position(workspace_root, args)?;
                self.request_text("textDocument/references", json!({"textDocument":{"uri":uri},"position":pos,"context":{"includeDeclaration":true}})).map(complete_tool_outcome)
            }
        }
    }

    fn pull_diagnostics(
        &mut self,
        document: &OpenDocumentState,
        timeout: Duration,
    ) -> Result<Value> {
        let report = self.request_with_timeout(
            "textDocument/diagnostic",
            json!({"textDocument":{"uri":document.uri}}),
            timeout,
        )?;
        diagnostic_items_from_pull_report(&report)
    }

    fn position_request(
        &mut self,
        workspace_root: &Path,
        args: &Value,
        method: &str,
    ) -> Result<String> {
        let (uri, pos) = self.document_position(workspace_root, args)?;
        self.request_text(method, json!({"textDocument":{"uri":uri},"position":pos}))
    }
    fn document_position(
        &mut self,
        workspace_root: &Path,
        args: &Value,
    ) -> Result<(String, Value)> {
        let document = self.open_document(workspace_root, args, false)?;
        let line = args
            .get("line")
            .and_then(Value::as_u64)
            .context("line is required")?;
        let character = args
            .get("character")
            .and_then(Value::as_u64)
            .context("character is required")?;
        Ok((
            document.uri,
            json!({"line": line.saturating_sub(1), "character": character}),
        ))
    }
    fn open_document(
        &mut self,
        workspace_root: &Path,
        args: &Value,
        force_refresh: bool,
    ) -> Result<OpenDocumentState> {
        let path = string_arg(args, "path")?;
        let full = resolve_workspace_path(workspace_root, path)?;
        let text = read_bounded_utf8_until(
            &full,
            MAX_LSP_DOCUMENT_BYTES,
            "LSP document",
            self.effective_deadline(LSP_READ_TIMEOUT),
        )?;
        let uri = path_to_uri(&full)?;
        let content_sha256 = crate::environment_lock::sha256(text.as_bytes());
        if let Some(document) = self.open_documents.get(&uri) {
            if document.content_sha256 != content_sha256 || force_refresh {
                let version = document.version.saturating_add(1);
                self.diagnostics.remove(&uri);
                self.notify(
                    "textDocument/didChange",
                    json!({"textDocument":{"uri":uri,"version":version},"contentChanges":[{"text":text}]}),
                )?;
                self.open_documents.insert(
                    uri.clone(),
                    OpenDocument {
                        version,
                        content_sha256,
                    },
                );
            }
        } else {
            if self.open_documents.len() >= MAX_LSP_OPEN_DOCUMENTS {
                bail!(
                    "LSP session reached the {MAX_LSP_OPEN_DOCUMENTS}-document bound; start a new session"
                );
            }
            self.diagnostics.remove(&uri);
            let language_id = language_id_for_path(&full, &self.language_ids);
            self.notify(
                "textDocument/didOpen",
                json!({"textDocument":{"uri":uri,"languageId":language_id,"version":1,"text":text}}),
            )?;
            self.open_documents.insert(
                uri.clone(),
                OpenDocument {
                    version: 1,
                    content_sha256,
                },
            );
        }
        let version = self
            .open_documents
            .get(&uri)
            .map(|document| document.version)
            .context("opened LSP document state disappeared")?;
        Ok(OpenDocumentState { uri, version })
    }

    fn wait_for_diagnostics(
        &mut self,
        document: &OpenDocumentState,
        timeout: Duration,
        after_serial: u64,
    ) -> Result<Value> {
        let deadline = self.effective_deadline(timeout);
        loop {
            if let Some(mut observed_serial) = self
                .diagnostics
                .get(&document.uri)
                .filter(|snapshot| {
                    snapshot.serial > after_serial
                        && diagnostic_snapshot_is_fresh(snapshot, document.version)
                })
                .map(|snapshot| snapshot.serial)
            {
                loop {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        break;
                    }
                    std::thread::sleep(LSP_DIAGNOSTIC_QUIET_PERIOD.min(remaining));
                    let mut publication_changed = false;
                    while let Some(message) = self.stdout_reader.try_recv()? {
                        self.handle_incoming_message(message)?;
                        if self
                            .diagnostics
                            .get(&document.uri)
                            .is_some_and(|latest| latest.serial > observed_serial)
                        {
                            observed_serial = self.diagnostics[&document.uri].serial;
                            publication_changed = true;
                        }
                    }
                    if !publication_changed {
                        break;
                    }
                }
                if let Some(latest) = self.diagnostics.get(&document.uri).filter(|snapshot| {
                    snapshot.serial > after_serial
                        && diagnostic_snapshot_is_fresh(snapshot, document.version)
                }) {
                    return Ok(latest.diagnostics.clone());
                }
            }
            if Instant::now() >= deadline {
                bail!(
                    "timed out waiting for fresh diagnostics for document version {}",
                    document.version
                );
            }
            let message = self.read_message(deadline)?;
            self.handle_incoming_message(message)?;
        }
    }

    fn handle_incoming_message(&mut self, message: Value) -> Result<()> {
        if let Some(method) = message.get("method").and_then(Value::as_str) {
            if let Some(id) = message.get("id").cloned() {
                self.handle_server_request(
                    id,
                    method,
                    message.get("params").cloned().unwrap_or(Value::Null),
                )?;
            } else {
                self.handle_notification(
                    method,
                    message.get("params").cloned().unwrap_or(Value::Null),
                );
            }
        }
        Ok(())
    }
    fn request_text(&mut self, method: &str, params: Value) -> Result<String> {
        Ok(self.request(method, params)?.to_string())
    }
    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.request_with_timeout(method, params, LSP_READ_TIMEOUT)
    }
    fn request_with_timeout(
        &mut self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let deadline = self.effective_deadline(timeout);
        self.write_until(
            &json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}),
            deadline,
        )
        .map_err(|error| {
            lsp_transport_failure(format!("failed to write LSP request {method}: {error:#}"))
        })?;
        loop {
            if Instant::now() > deadline {
                return Err(lsp_transport_failure(format!(
                    "timed out waiting for LSP response to {method}"
                )));
            }
            let msg = self.read_message(deadline)?;
            if let Some(method) = msg.get("method").and_then(Value::as_str) {
                if let Some(id) = msg.get("id").cloned() {
                    self.handle_server_request(
                        id,
                        method,
                        msg.get("params").cloned().unwrap_or(Value::Null),
                    )?;
                } else {
                    self.handle_notification(
                        method,
                        msg.get("params").cloned().unwrap_or(Value::Null),
                    );
                }
                continue;
            }
            if msg.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = msg.get("error") {
                bail!("LSP request {method} failed: {error}");
            }
            return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
        }
    }
    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.write(&json!({"jsonrpc":"2.0","method":method,"params":params}))
            .map_err(|error| {
                lsp_transport_failure(format!(
                    "failed to write LSP notification {method}: {error:#}"
                ))
            })
    }
    fn write(&mut self, message: &Value) -> Result<()> {
        self.write_until(message, self.effective_deadline(LSP_READ_TIMEOUT))
    }
    fn write_until(&mut self, message: &Value, deadline: Instant) -> Result<()> {
        self.stdin_writer.write_until(message, deadline)
    }
    fn effective_deadline(&self, timeout: Duration) -> Instant {
        let requested = Instant::now() + timeout;
        self.active_deadline
            .map_or(requested, |deadline| deadline.min(requested))
    }
    fn read_message(&mut self, deadline: Instant) -> Result<Value> {
        if let Some(status) = self.process.try_wait().map_err(|error| {
            lsp_transport_failure(format!("failed to poll LSP server: {error:#}"))
        })? {
            return Err(lsp_transport_failure(format!(
                "LSP server exited with status {status}: {}",
                self.stderr_text()
            )));
        }
        self.stdout_reader.recv_until(deadline).map_err(|error| {
            lsp_transport_failure(format!(
                "LSP response failed: {error:#}: {}",
                self.stderr_text()
            ))
        })
    }
    fn stderr_text(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|tail| String::from_utf8_lossy(&tail.iter().copied().collect::<Vec<_>>()).into())
            .unwrap_or_default()
    }
    fn handle_notification(&mut self, method: &str, params: Value) {
        if method == "textDocument/publishDiagnostics"
            && let Some(uri) = params.get("uri").and_then(Value::as_str)
            && self.open_documents.contains_key(uri)
        {
            self.diagnostic_serial = self.diagnostic_serial.saturating_add(1);
            self.diagnostics.insert(
                uri.to_string(),
                DiagnosticSnapshot {
                    version: params.get("version").and_then(Value::as_u64),
                    serial: self.diagnostic_serial,
                    diagnostics: params
                        .get("diagnostics")
                        .cloned()
                        .unwrap_or_else(|| json!([])),
                },
            );
        }
    }

    fn handle_server_request(&mut self, id: Value, method: &str, params: Value) -> Result<()> {
        let result = match method {
            "workspace/configuration" => {
                let count = params
                    .get("items")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len);
                Value::Array((0..count).map(|_| Value::Null).collect())
            }
            "workspace/workspaceFolders" => {
                json!([{"uri": self.root_uri, "name": self.root_name}])
            }
            "client/registerCapability"
            | "client/unregisterCapability"
            | "window/workDoneProgress/create" => Value::Null,
            _ => Value::Null,
        };
        self.write(&json!({"jsonrpc":"2.0","id":id,"result":result}))
            .map_err(|error| {
                lsp_transport_failure(format!(
                    "failed to answer LSP server request {method}: {error:#}"
                ))
            })
    }
}

fn spawn_lsp_transport_until(
    server_name: &str,
    config: &LspServerConfig,
    workspace_root: &Path,
    cwd: &Path,
    lease: Option<&Arc<crate::session_environment::SessionEnvironmentLease>>,
    deadline: Instant,
) -> Result<LspTransport> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        bail!("proactive LSP time bound expired before starting the language server");
    }
    let server_name = server_name.to_string();
    let config = config.clone();
    let workspace_root = workspace_root.to_path_buf();
    let cwd = cwd.to_path_buf();
    let lease = lease.cloned();
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let (sender, receiver) = sync_channel(0);
    std::thread::spawn(move || {
        let result =
            spawn_lsp_transport(&server_name, &config, &workspace_root, &cwd, lease.as_ref());
        if worker_cancelled.load(Ordering::Acquire) {
            if let Ok(mut transport) = result {
                transport.process.shutdown(Duration::from_secs(1));
            }
            return;
        }
        if let Err(error) = sender.send(result)
            && let Ok(mut transport) = error.0
        {
            transport.process.shutdown(Duration::from_secs(1));
        }
    });
    match receiver.recv_timeout(remaining) {
        Ok(result) => result,
        Err(_) => {
            cancelled.store(true, Ordering::Release);
            bail!("timed out starting the language server")
        }
    }
}

fn spawn_lsp_transport(
    server_name: &str,
    config: &LspServerConfig,
    workspace_root: &Path,
    cwd: &Path,
    lease: Option<&Arc<crate::session_environment::SessionEnvironmentLease>>,
) -> Result<LspTransport> {
    if let Some(lease) = lease {
        if let Some(image) = config
            .container_image
            .as_deref()
            .filter(|image| !image.trim().is_empty())
        {
            let actual_runtime = lease.record()?.runtime_binary;
            crate::container::ensure_service_runtime_matches(
                config.container_runtime.as_deref(),
                &actual_runtime,
                &format!("LSP server {server_name}"),
            )?;
            let mut service =
                lease.spawn_service(crate::session_environment::SessionServiceSpec {
                    service_name: server_name.to_string(),
                    role: format!("lsp:{server_name}"),
                    kind: crate::session_environment::LeaseResourceKind::LspProcess,
                    image: image.trim().to_string(),
                    args: config.args.clone(),
                    env: config.env.clone(),
                    working_directory: config.working_directory.clone(),
                    cache_scope_sha256: crate::environment_lock::sha256(&serde_json::to_vec(
                        config,
                    )?),
                    workspace_access: config.workspace_access,
                    network_access: config.network_access,
                    cache_ids: config.cache_ids.clone(),
                })?;
            let stdin = service.take_stdin()?;
            let stdout = service.take_stdout()?;
            let stderr = service.take_stderr()?;
            return Ok(LspTransport {
                process: LspProcess::Service(service),
                stdin,
                stdout,
                stderr: Some(stderr),
            });
        }
        let command = config.command.as_deref().with_context(|| {
            format!("LSP server {server_name} has no command or container_image")
        })?;
        let mut argv = vec![
            "sh".to_string(),
            "-lc".to_string(),
            "cd \"$1\" && shift && exec \"$@\"".to_string(),
            "pb-lsp".to_string(),
            cwd.to_string_lossy().into_owned(),
            command.to_string(),
        ];
        argv.extend(config.args.clone());
        let mut process = lease.spawn_exec_with_env(&argv, &config.env)?;
        let stdin = process.take_stdin()?;
        let stdout = process.take_stdout()?;
        let stderr = process.take_stderr()?;
        return Ok(LspTransport {
            process: LspProcess::Exec(process),
            stdin,
            stdout,
            stderr: Some(stderr),
        });
    }

    let (command, args) = stdio_command(server_name, config, workspace_root)?;
    let mut command_builder = Command::new(&command);
    command_builder
        .args(&args)
        .current_dir(cwd)
        .envs(&config.env)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command_builder.process_group(0);
    }
    let mut child = command_builder
        .spawn()
        .with_context(|| format!("failed to start LSP server {server_name}: {command}"))?;
    let stdin = child
        .stdin
        .take()
        .context("failed to open LSP server stdin")?;
    let stdout = child
        .stdout
        .take()
        .context("failed to open LSP server stdout")?;
    let stderr = child
        .stderr
        .take()
        .context("failed to open LSP server stderr")?;
    Ok(LspTransport {
        process: LspProcess::Host(child),
        stdin,
        stdout,
        stderr: Some(stderr),
    })
}

fn server_supports_pull_diagnostics(initialize_result: &Value) -> bool {
    initialize_result
        .pointer("/capabilities/diagnosticProvider")
        .is_some_and(|capability| !capability.is_null() && capability != &Value::Bool(false))
}

fn diagnostic_items_from_pull_report(report: &Value) -> Result<Value> {
    match report.get("kind").and_then(Value::as_str) {
        Some("full") => report
            .get("items")
            .filter(|items| items.is_array())
            .cloned()
            .context("full diagnostic pull response omitted its items array"),
        Some("unchanged") => {
            bail!("diagnostic pull returned unchanged without a previous result identifier")
        }
        _ => bail!("diagnostic pull returned an unsupported report"),
    }
}

fn validate_packaged_image_authority(config: &LspServerConfig) -> Result<()> {
    let Some(image) = config.container_image.as_deref() else {
        return Ok(());
    };
    if !crate::integrations::is_marketplace_container_image(image) {
        return Ok(());
    }
    let digest = config.verified_manifest_digest.as_deref().with_context(
        || "marketplace LSP installation is legacy-unverified; upgrade or reinstall it before use",
    )?;
    let expected = format!("@{digest}");
    if !image.ends_with(&expected) {
        bail!("marketplace LSP image does not match its verified manifest digest");
    }
    Ok(())
}

fn diagnostic_snapshot_is_fresh(snapshot: &DiagnosticSnapshot, version: u64) -> bool {
    snapshot
        .version
        .is_none_or(|published_version| published_version == version)
}

impl Drop for LspClient {
    fn drop(&mut self) {
        if self.transport_shutdown {
            return;
        }
        for uri in self.open_documents.keys().cloned().collect::<Vec<_>>() {
            let _ = self.notify("textDocument/didClose", json!({"textDocument":{"uri":uri}}));
        }
        let _ = self.request_with_timeout("shutdown", Value::Null, self.shutdown_timeout);
        let _ = self.notify("exit", json!(null));
        self.shutdown_transport(self.shutdown_timeout);
    }
}

impl LspClient {
    fn shutdown_transport(&mut self, timeout: Duration) {
        if self.transport_shutdown {
            return;
        }
        self.process.shutdown(timeout);
        self.stdin_writer.join();
        self.stdout_reader.join();
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
        self.transport_shutdown = true;
    }
}

fn terminate_host_process_group(process_id: u32) {
    #[cfg(unix)]
    // SAFETY: POSIX `kill` is called with a negated child process-group id and a valid signal
    // number. It does not dereference memory; failure is intentionally best-effort during cleanup.
    unsafe {
        unsafe extern "C" {
            fn kill(pid: i32, signal: i32) -> i32;
        }
        const SIGKILL: i32 = 9;
        let _ = kill(-(process_id as i32), SIGKILL);
    }
    #[cfg(not(unix))]
    let _ = process_id;
}

fn drain_stderr(
    mut stderr: ChildStderr,
    tail: Arc<Mutex<VecDeque<u8>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buffer = [0_u8; 4096];
        while let Ok(read) = stderr.read(&mut buffer) {
            if read == 0 {
                break;
            }
            if let Ok(mut tail) = tail.lock() {
                tail.extend(&buffer[..read]);
                while tail.len() > 64 * 1024 {
                    tail.pop_front();
                }
            }
        }
    })
}

fn stdio_command(
    server_name: &str,
    config: &LspServerConfig,
    _workspace_root: &Path,
) -> Result<(String, Vec<String>)> {
    if let Some(image) = config
        .container_image
        .as_deref()
        .filter(|i| !i.trim().is_empty())
    {
        bail!(
            "container LSP image '{}' requires an active session environment lease",
            image.trim()
        );
    }
    let command = config
        .command
        .as_deref()
        .with_context(|| format!("LSP server {server_name} has no command or container_image"))?;
    Ok((command.to_string(), config.args.clone()))
}

fn resolve_workspace_path(root: &Path, path: &str) -> Result<PathBuf> {
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve LSP workspace {}", root.display()))?;
    let candidate = root.join(path);
    let full = candidate
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", candidate.display()))?;
    if !full.starts_with(root) {
        bail!("path escapes workspace: {path}");
    }
    Ok(full)
}
fn path_to_uri(path: &Path) -> Result<String> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to resolve LSP URI path {}", path.display()))?;
    url::Url::from_file_path(&canonical)
        .map(|url| url.to_string())
        .map_err(|()| anyhow!("failed to encode LSP file URI for {}", canonical.display()))
}
fn string_arg<'a>(args: &'a Value, name: &str) -> Result<&'a str> {
    args.get(name)
        .and_then(Value::as_str)
        .with_context(|| format!("{name} is required"))
}

fn language_id_for_path(path: &Path, configured: &[String]) -> String {
    if configured.len() == 1 {
        return configured[0].clone();
    }
    let inferred = inferred_language_id(path);
    inferred
        .and_then(|inferred| {
            configured
                .iter()
                .find(|candidate| candidate.eq_ignore_ascii_case(inferred))
                .cloned()
        })
        .or_else(|| configured.first().cloned())
        .unwrap_or_else(|| inferred.unwrap_or("plaintext").to_string())
}

fn inferred_language_id(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("rs") => Some("rust"),
        Some("ts") => Some("typescript"),
        Some("tsx") => Some("typescriptreact"),
        Some("js") | Some("mjs") | Some("cjs") => Some("javascript"),
        Some("jsx") => Some("javascriptreact"),
        Some("py") => Some("python"),
        Some("go") => Some("go"),
        Some("java") => Some("java"),
        Some("kt") | Some("kts") => Some("kotlin"),
        Some("swift") => Some("swift"),
        Some("c") | Some("h") => Some("c"),
        Some("cc") | Some("cpp") | Some("cxx") | Some("hpp") => Some("cpp"),
        Some("json") => Some("json"),
        Some("yaml") | Some("yml") => Some("yaml"),
        Some("html") => Some("html"),
        Some("css") => Some("css"),
        _ => None,
    }
}

fn unique_tool_name(server: &str, op: &str) -> String {
    format!("lsp_{}_{}", sanitize(server), sanitize(op))
}
fn sanitize(name: &str) -> String {
    let out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let t = out.trim_matches('_');
    if t.is_empty() {
        "tool".to_string()
    } else {
        t.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_config_round_trips_initialization_and_sidecar_capabilities() {
        let dir = tempfile::tempdir().unwrap();
        let config = ProjectLspConfig {
            servers: BTreeMap::from([(
                "rust".to_string(),
                LspServerConfig {
                    container_image: Some("example/rust-analyzer:locked".to_string()),
                    initialization_options: Some(json!({"cargo":{"allFeatures":true}})),
                    cache_ids: vec!["rust-analyzer-index".to_string()],
                    ..Default::default()
                },
            )]),
        };
        config.save(dir.path()).unwrap();
        assert_eq!(ProjectLspConfig::load(dir.path()).unwrap().unwrap(), config);
    }

    #[test]
    fn discovery_is_lazy_even_for_an_unavailable_server() {
        let directory = tempfile::tempdir().unwrap();
        let registry = discover_tools(
            BTreeMap::from([(
                "rust".to_string(),
                LspServerConfig {
                    command: Some("/definitely/missing/pb-lsp".to_string()),
                    language_ids: vec!["rust".to_string()],
                    ..Default::default()
                },
            )]),
            directory.path(),
        );

        assert!(registry.tool("lsp_rust_diagnostics").is_some());
        assert!(registry.sessions["rust"].lock().unwrap().is_none());
    }

    #[test]
    fn proactive_report_accounts_for_every_server_path_target_at_the_call_bound() {
        let directory = tempfile::tempdir().unwrap();
        Command::new("git")
            .arg("init")
            .arg("--quiet")
            .current_dir(directory.path())
            .status()
            .unwrap();
        let mut paths = Vec::new();
        for index in 0..MAX_PROACTIVE_LSP_PATHS {
            let path = format!("file-{index}.rs");
            std::fs::write(directory.path().join(&path), "fn item() {}\n").unwrap();
            paths.push(path);
        }
        let config = || LspServerConfig {
            command: Some("/definitely/missing/pb-lsp".to_string()),
            language_ids: vec!["rust".to_string()],
            ..Default::default()
        };
        let registry = discover_tools(
            BTreeMap::from([
                ("first".to_string(), config()),
                ("second".to_string(), config()),
            ]),
            directory.path(),
        );

        let report = proactive_diagnostics(
            &registry,
            directory.path(),
            paths,
            ProactiveLspMode::Settled,
            7,
        )
        .unwrap();

        assert_eq!(report.requested_targets.len(), 16);
        assert_eq!(report.failures.len(), MAX_PROACTIVE_LSP_CALLS);
        assert_eq!(report.deferred_targets.len(), 8);
        assert!(
            report
                .incomplete_reasons
                .contains(&ProactiveLspIncompleteReason::CallBound)
        );
        assert!(!report.complete);
        assert!(report.completed_paths().is_empty());
    }

    #[test]
    fn completed_paths_require_every_matching_server_target() {
        let mut report = ProactiveLspReport {
            mode: ProactiveLspMode::Settled,
            workspace_epoch: 1,
            workspace_fingerprint: "workspace".to_string(),
            requested_paths: vec!["src/lib.rs".to_string()],
            scanned_paths: vec!["src/lib.rs".to_string()],
            diagnostics: Vec::new(),
            suppressed_diagnostics: 0,
            omitted_paths: 0,
            failures: Vec::new(),
            stale: false,
            requested_targets: vec![
                ProactiveLspTarget {
                    server: "a".to_string(),
                    path: "src/lib.rs".to_string(),
                },
                ProactiveLspTarget {
                    server: "b".to_string(),
                    path: "src/lib.rs".to_string(),
                },
            ],
            completed_targets: vec![ProactiveLspTarget {
                server: "a".to_string(),
                path: "src/lib.rs".to_string(),
            }],
            advisory_targets: Vec::new(),
            deferred_targets: Vec::new(),
            incomplete_reasons: BTreeSet::new(),
            complete: false,
        };
        assert!(report.completed_paths().is_empty());
        report.completed_targets.push(ProactiveLspTarget {
            server: "b".to_string(),
            path: "src/lib.rs".to_string(),
        });
        assert_eq!(
            report.completed_paths(),
            BTreeSet::from(["src/lib.rs".to_string()])
        );
    }

    #[test]
    fn push_only_targets_are_attempted_but_never_complete() {
        let target = ProactiveLspTarget {
            server: "legacy".to_string(),
            path: "src/lib.rs".to_string(),
        };
        let report = ProactiveLspReport {
            mode: ProactiveLspMode::Settled,
            workspace_epoch: 1,
            workspace_fingerprint: "workspace".to_string(),
            requested_paths: vec![target.path.clone()],
            scanned_paths: vec![target.path.clone()],
            diagnostics: Vec::new(),
            suppressed_diagnostics: 0,
            omitted_paths: 0,
            failures: Vec::new(),
            stale: false,
            requested_targets: vec![target.clone()],
            completed_targets: Vec::new(),
            advisory_targets: vec![target],
            deferred_targets: Vec::new(),
            incomplete_reasons: BTreeSet::from([ProactiveLspIncompleteReason::PushOnlySnapshot]),
            complete: false,
        };
        assert!(report.completed_paths().is_empty());
        assert_eq!(
            report.attempted_paths(),
            BTreeSet::from(["src/lib.rs".to_string()])
        );
    }

    #[test]
    fn diagnostic_pull_requires_an_explicit_full_report() {
        let diagnostics = diagnostic_items_from_pull_report(&json!({
            "kind": "full",
            "items": [{"message":"error"}]
        }))
        .unwrap();
        assert_eq!(diagnostics.as_array().unwrap().len(), 1);
        assert!(
            diagnostic_items_from_pull_report(&json!({
                "kind": "unchanged",
                "resultId": "old"
            }))
            .is_err()
        );
    }

    #[test]
    fn initialize_capability_controls_diagnostic_pull() {
        assert!(server_supports_pull_diagnostics(&json!({
            "capabilities": {"diagnosticProvider": {"interFileDependencies": true}}
        })));
        assert!(!server_supports_pull_diagnostics(&json!({
            "capabilities": {}
        })));
    }

    #[test]
    fn expired_proactive_deadline_fails_before_startup() {
        let error = remaining_timeout(Some(Instant::now()), Duration::from_secs(30), "starting")
            .unwrap_err()
            .to_string();
        assert!(error.contains("time bound expired"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn lsp_stdin_writes_are_deadline_bounded() {
        let mut child = Command::new("sh")
            .args(["-c", "sleep 5"])
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let mut writer = LspStdinWriter::spawn(stdin);
        let message = json!({"payload": "x".repeat(2 * 1024 * 1024)});
        let started = Instant::now();
        let error = writer
            .write_until(&message, Instant::now() + Duration::from_millis(100))
            .unwrap_err()
            .to_string();
        assert!(error.contains("timed out writing"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(2));
        let _ = child.kill();
        let _ = child.wait();
        writer.join();
    }

    #[cfg(unix)]
    #[test]
    fn proactive_deadline_bounds_an_unresponsive_initializer_and_its_cleanup() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("src")).unwrap();
        std::fs::write(directory.path().join("src/lib.rs"), "pub fn value() {}\n").unwrap();
        let registry = discover_tools(
            BTreeMap::from([(
                "rust".to_string(),
                LspServerConfig {
                    command: Some("sh".to_string()),
                    args: vec!["-c".to_string(), "sleep 5".to_string()],
                    language_ids: vec!["rust".to_string()],
                    ..Default::default()
                },
            )]),
            directory.path(),
        );
        let started = Instant::now();
        let error = call_tool_with_diagnostic_timeout(
            &registry,
            directory.path(),
            "lsp_rust_diagnostics",
            &json!({"path":"src/lib.rs"}),
            Duration::from_secs(2),
            Some(Instant::now() + Duration::from_millis(100)),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("LSP response failed"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn unavailable_selected_path_cannot_be_reported_as_clean() {
        let directory = tempfile::tempdir().unwrap();
        Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(directory.path())
            .status()
            .unwrap();
        let registry = discover_tools(
            BTreeMap::from([(
                "rust".to_string(),
                LspServerConfig {
                    command: Some("rust-analyzer".to_string()),
                    language_ids: vec!["rust".to_string()],
                    ..Default::default()
                },
            )]),
            directory.path(),
        );

        let report = proactive_diagnostics(
            &registry,
            directory.path(),
            ["src/missing.rs".to_string()],
            ProactiveLspMode::Settled,
            1,
        )
        .unwrap();

        assert!(!report.complete);
        assert!(report.completed_targets.is_empty());
        assert!(
            report
                .incomplete_reasons
                .contains(&ProactiveLspIncompleteReason::PathUnavailable)
        );
    }

    #[test]
    fn container_lsp_requires_a_session_owned_service() {
        let config = LspServerConfig {
            container_image: Some("example/lsp".to_string()),
            container_runtime: Some("podman".to_string()),
            args: vec!["server".to_string()],
            ..Default::default()
        };
        let error = stdio_command("rust", &config, Path::new("/workspace/pb"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("active session environment lease"));
    }

    #[test]
    fn marketplace_lsp_authority_requires_the_verified_digest() {
        let legacy = LspServerConfig {
            container_image: Some("ghcr.io/crunchy-pb/lsp-rust-analyzer:latest".to_string()),
            ..Default::default()
        };
        assert!(
            validate_packaged_image_authority(&legacy)
                .unwrap_err()
                .to_string()
                .contains("legacy-unverified")
        );

        let pinned = LspServerConfig {
            container_image: Some("ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:abc".to_string()),
            verified_manifest_digest: Some("sha256:abc".to_string()),
            ..Default::default()
        };
        validate_packaged_image_authority(&pinned).unwrap();
    }

    #[test]
    #[ignore = "requires GHCR access and a supported local container runtime"]
    fn live_marketplace_rust_analyzer_runs_in_a_service_only_lease() {
        let source = "ghcr.io/crunchy-pb/lsp-rust-analyzer:latest";
        let metadata = crate::integrations::fetch_config_schema(source).unwrap();
        let manifest = metadata.lsp_manifest.expect("LSP package manifest");
        let runtime = crate::container::detect_runtime().expect("supported container runtime");
        if !runtime.image_exists(&metadata.container_image).unwrap() {
            runtime.pull(&metadata.container_image).unwrap();
        }

        let directory = tempfile::Builder::new()
            .prefix(".pb-lsp-live-")
            .tempdir_in(std::env::current_dir().unwrap())
            .unwrap();
        Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(directory.path())
            .status()
            .unwrap();
        std::fs::create_dir_all(directory.path().join("src")).unwrap();
        std::fs::write(
            directory.path().join("Cargo.toml"),
            "[package]\nname = \"pb-lsp-live-smoke\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        std::fs::write(
            directory.path().join("src/lib.rs"),
            "pub fn answer() -> u32 { 42 }\n",
        )
        .unwrap();

        let state = tempfile::tempdir().unwrap();
        let supervisor = Box::leak(Box::new(
            crate::session_environment::EnvironmentSupervisor::new(state.path().to_path_buf()),
        ));
        let handle = supervisor
            .acquire_service_only(
                "live-marketplace-rust-analyzer",
                directory.path(),
                directory.path(),
                runtime,
                false,
            )
            .unwrap();
        let server = manifest.server;
        let registry = discover_tools_with_lease(
            BTreeMap::from([(
                "rust-analyzer".to_string(),
                LspServerConfig {
                    container_image: Some(metadata.container_image),
                    source_container_image: Some(metadata.source_container_image),
                    verified_manifest_digest: Some(metadata.manifest_digest),
                    args: server.args,
                    language_ids: server.language_ids,
                    initialization_options: server.initialization_options,
                    workspace_access: server.workspace_access,
                    network_access: server.network_access,
                    cache_ids: server.cache_ids,
                    ..Default::default()
                },
            )]),
            directory.path(),
            handle.lease(),
        );

        let outcome = call_tool_with_diagnostic_timeout(
            &registry,
            directory.path(),
            "lsp_rust_analyzer_diagnostics",
            &json!({"path":"src/lib.rs"}),
            LSP_DIAGNOSTIC_TIMEOUT,
            None,
        )
        .unwrap();
        assert!(outcome.diagnostic_complete);
        assert!(
            serde_json::from_str::<Value>(&outcome.result)
                .unwrap()
                .is_array()
        );
        drop(registry);
        drop(handle);
    }

    #[test]
    fn file_uris_encode_worktree_and_source_path_delimiters() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source #1.rs");
        std::fs::write(&source, "fn main() {}\n").unwrap();

        let uri = path_to_uri(&source).unwrap();
        assert!(uri.starts_with("file:///"));
        assert!(uri.ends_with("source%20%231.rs"));
        assert!(!uri.contains(' '));
    }

    #[cfg(unix)]
    #[test]
    fn lsp_path_resolution_accepts_a_symlinked_workspace_root() {
        let directory = tempfile::tempdir().unwrap();
        let real = directory.path().join("real");
        let linked = directory.path().join("linked");
        std::fs::create_dir(&real).unwrap();
        std::fs::write(real.join("lib.rs"), "fn main() {}\n").unwrap();
        std::os::unix::fs::symlink(&real, &linked).unwrap();

        assert_eq!(
            resolve_workspace_path(&linked, "lib.rs").unwrap(),
            real.canonicalize().unwrap().join("lib.rs")
        );
    }

    #[test]
    fn language_ids_are_selected_per_file_extension() {
        let configured = vec!["typescript".to_string(), "rust".to_string()];
        assert_eq!(
            language_id_for_path(Path::new("src/lib.rs"), &configured),
            "rust"
        );
        assert_eq!(
            language_id_for_path(Path::new("web/app.ts"), &configured),
            "typescript"
        );
    }

    #[test]
    fn diagnostics_with_an_old_document_version_are_not_fresh() {
        let stale = DiagnosticSnapshot {
            version: Some(1),
            serial: 1,
            diagnostics: json!([]),
        };
        let current = DiagnosticSnapshot {
            version: Some(2),
            serial: 2,
            diagnostics: json!([]),
        };
        assert!(!diagnostic_snapshot_is_fresh(&stale, 2));
        assert!(diagnostic_snapshot_is_fresh(&current, 2));
    }

    #[test]
    fn proactive_syntax_mode_admits_only_error_parse_diagnostics() {
        let syntax = json!({
            "range": {"start": {"line": 3, "character": 7}, "end": {"line": 3, "character": 8}},
            "severity": 1,
            "code": "syntax-error",
            "source": "rust-analyzer",
            "message": "expected expression"
        });
        let semantic = json!({
            "range": {"start": {"line": 4, "character": 0}, "end": {"line": 4, "character": 3}},
            "severity": 1,
            "code": "unresolved-ident",
            "source": "rust-analyzer",
            "message": "unresolved identifier"
        });
        let warning = json!({
            "range": {"start": {"line": 5, "character": 0}, "end": {"line": 5, "character": 3}},
            "severity": 2,
            "code": "syntax-warning",
            "source": "rust-analyzer",
            "message": "warning"
        });

        let normalized = normalize_proactive_diagnostic(
            "rust-analyzer",
            "src/lib.rs",
            "path-fingerprint",
            &syntax,
        )
        .unwrap();
        assert_eq!(normalized.line, 4);
        assert!(proactive_diagnostic_is_syntax(&syntax, &normalized));
        let normalized = normalize_proactive_diagnostic(
            "rust-analyzer",
            "src/lib.rs",
            "path-fingerprint",
            &semantic,
        )
        .unwrap();
        assert!(!proactive_diagnostic_is_syntax(&semantic, &normalized));
        let normalized = normalize_proactive_diagnostic(
            "rust-analyzer",
            "src/lib.rs",
            "path-fingerprint",
            &warning,
        )
        .unwrap();
        assert!(!proactive_diagnostic_is_syntax(&warning, &normalized));
    }

    #[test]
    fn proactive_path_selection_is_language_bound_and_bounded() {
        let registry = LspToolRegistry {
            servers: BTreeMap::from([(
                "rust-analyzer".to_string(),
                LspServerConfig {
                    language_ids: vec!["rust".to_string()],
                    command: Some("rust-analyzer".to_string()),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        };
        let mut paths = (0..12)
            .map(|index| format!("src/module_{index}.rs"))
            .collect::<Vec<_>>();
        paths.push("README.md".to_string());

        let selected = proactive_supported_paths(&registry, paths);

        assert_eq!(selected.paths.len(), MAX_PROACTIVE_LSP_PATHS);
        assert_eq!(selected.omitted_paths, 4);
        assert!(selected.paths.iter().all(|path| path.ends_with(".rs")));
        assert!(!selected.paths.contains(&"README.md".to_string()));
    }

    #[test]
    fn proactive_text_is_single_line_and_bounded() {
        let bounded = bounded_proactive_text(&format!("first\n{}", "x".repeat(800)));
        assert!(!bounded.contains('\n'));
        assert_eq!(bounded.chars().count(), MAX_PROACTIVE_LSP_MESSAGE_CHARS);
        assert!(bounded.ends_with('…'));
    }

    #[test]
    fn lsp_name_normalization_collisions_are_detectable() {
        assert_eq!(
            unique_tool_name("rust-analyzer", "diagnostics"),
            unique_tool_name("rust/analyzer", "diagnostics")
        );
    }
}
