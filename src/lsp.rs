//! LSP server configuration and stdio-backed tool support.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, Command, Stdio};
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
const MAX_PROACTIVE_LSP_PATHS: usize = 8;
const MAX_PROACTIVE_LSP_CALLS: usize = 8;
const MAX_PROACTIVE_LSP_DIAGNOSTICS: usize = 64;
const MAX_PROACTIVE_LSP_MESSAGE_CHARS: usize = 500;

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProactiveLspReport {
    pub mode: ProactiveLspMode,
    pub workspace_fingerprint: String,
    pub requested_paths: Vec<String>,
    pub scanned_paths: Vec<String>,
    pub diagnostics: Vec<ProactiveLspDiagnostic>,
    pub suppressed_diagnostics: usize,
    #[serde(default)]
    pub omitted_paths: usize,
    pub failures: Vec<ProactiveLspFailure>,
    pub stale: bool,
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
        self.scanned_paths
            .iter()
            .cloned()
            .chain(
                self.failures
                    .iter()
                    .filter(|failure| failure.path != ".")
                    .map(|failure| failure.path.clone()),
            )
            .collect()
    }
}

#[derive(Clone, Default)]
pub struct LspToolRegistry {
    pub servers: BTreeMap<String, LspServerConfig>,
    pub tools: BTreeMap<String, LspToolSpec>,
    sessions: BTreeMap<String, Arc<Mutex<LspClient>>>,
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
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let text =
            toml::to_string_pretty(self).context("failed to serialize project LSP config")?;
        std::fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))
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
    workspace_root: &Path,
    lease: Option<Arc<crate::session_environment::SessionEnvironmentLease>>,
) -> LspToolRegistry {
    let mut registry = LspToolRegistry {
        servers: servers.clone(),
        tools: BTreeMap::new(),
        sessions: BTreeMap::new(),
        lease: lease.clone(),
    };
    let mut collided_names = BTreeSet::new();
    for (server_name, config) in servers {
        match LspClient::connect(&server_name, &config, workspace_root, lease.as_ref()) {
            Ok(client) => {
                registry
                    .sessions
                    .insert(server_name.clone(), Arc::new(Mutex::new(client)));
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
            Err(err) => {
                insert_status_tool(
                    &mut registry,
                    &server_name,
                    &format!("LSP server {server_name} was configured but startup failed: {err:#}"),
                );
            }
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
    )
}

fn call_tool_with_diagnostic_timeout(
    registry: &LspToolRegistry,
    workspace_root: &Path,
    tool_name: &str,
    arguments: &Value,
    diagnostic_timeout: Duration,
) -> Result<String> {
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
    let mut client = session
        .lock()
        .map_err(|_| anyhow!("LSP session for {} is poisoned", spec.server_name))?;
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
            client.shutdown_transport();
            *client = LspClient::connect(
                &spec.server_name,
                config,
                workspace_root,
                registry.lease.as_ref(),
            )
            .with_context(|| {
                format!("LSP request failed ({first_error:#}) and the bounded restart also failed")
            })?;
            client.call_with_diagnostic_timeout(
                spec.operation,
                workspace_root,
                arguments,
                diagnostic_timeout,
            )
        }
    }
}

pub fn proactive_diagnostics(
    registry: &LspToolRegistry,
    workspace_root: &Path,
    paths: impl IntoIterator<Item = String>,
    mode: ProactiveLspMode,
) -> Result<ProactiveLspReport> {
    let before = crate::workspace::ContentSnapshot::capture(workspace_root)?;
    let requested_paths = paths
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .take(MAX_PROACTIVE_LSP_PATHS)
        .collect::<Vec<_>>();
    let mut report = ProactiveLspReport {
        mode,
        workspace_fingerprint: before.fingerprint.clone(),
        requested_paths: requested_paths.clone(),
        scanned_paths: Vec::new(),
        diagnostics: Vec::new(),
        suppressed_diagnostics: 0,
        omitted_paths: 0,
        failures: Vec::new(),
        stale: false,
    };
    let started = Instant::now();
    let mut calls = 0usize;

    'paths: for path in requested_paths {
        let Some(path_state) = before.paths.get(&path) else {
            continue;
        };
        if path_state.kind != "file" {
            continue;
        }
        let Some(language_id) = inferred_language_id(Path::new(&path)) else {
            continue;
        };
        let matching_servers = registry
            .servers
            .iter()
            .filter(|(_, config)| {
                config
                    .language_ids
                    .iter()
                    .any(|candidate| candidate.eq_ignore_ascii_case(language_id))
            })
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        for server in matching_servers {
            if calls >= MAX_PROACTIVE_LSP_CALLS || started.elapsed() >= PROACTIVE_LSP_PASS_TIMEOUT {
                break 'paths;
            }
            calls = calls.saturating_add(1);
            let Some(spec) = registry.tools.values().find(|spec| {
                spec.server_name == server && spec.operation == LspOperation::Diagnostics
            }) else {
                report.failures.push(ProactiveLspFailure {
                    server: server.clone(),
                    path: path.clone(),
                    message: "diagnostics operation is unavailable".to_string(),
                });
                continue;
            };
            if let Some(error) = spec.status_error.as_deref() {
                report.failures.push(ProactiveLspFailure {
                    server: server.clone(),
                    path: path.clone(),
                    message: bounded_proactive_text(error),
                });
                continue;
            }
            let result = call_tool_with_diagnostic_timeout(
                registry,
                workspace_root,
                &spec.tool_name,
                &json!({"path": path.clone()}),
                PROACTIVE_LSP_DIAGNOSTIC_TIMEOUT,
            );
            match result {
                Ok(result) => {
                    if !report.scanned_paths.contains(&path) {
                        report.scanned_paths.push(path.clone());
                    }
                    let decoded: Value = match serde_json::from_str(&result) {
                        Ok(decoded) => decoded,
                        Err(error) => {
                            report.failures.push(ProactiveLspFailure {
                                server: server.clone(),
                                path: path.clone(),
                                message: format!("invalid diagnostics response: {error}"),
                            });
                            continue;
                        }
                    };
                    let Some(diagnostics) = decoded.as_array() else {
                        report.failures.push(ProactiveLspFailure {
                            server: server.clone(),
                            path: path.clone(),
                            message: "diagnostics response was not an array".to_string(),
                        });
                        continue;
                    };
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
                Err(error) => report.failures.push(ProactiveLspFailure {
                    server: server.clone(),
                    path: path.clone(),
                    message: bounded_proactive_text(&format!("{error:#}")),
                }),
            }
        }
    }

    let after = crate::workspace::ContentSnapshot::capture(workspace_root)?;
    if after.fingerprint != before.fingerprint {
        report.stale = true;
        report.diagnostics.clear();
        report.failures.push(ProactiveLspFailure {
            server: "pb".to_string(),
            path: ".".to_string(),
            message:
                "workspace content changed during proactive LSP collection; all diagnostics were discarded"
                    .to_string(),
        });
    }
    report.scanned_paths.sort();
    report.scanned_paths.dedup();
    report.diagnostics.sort();
    report.diagnostics.dedup();
    report.failures.sort();
    report.failures.dedup();
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
    stdin: ChildStdin,
    stdout_reader: crate::jsonrpc::FramedJsonReader,
    next_id: u64,
    diagnostics: BTreeMap<String, DiagnosticSnapshot>,
    language_ids: Vec<String>,
    root_uri: String,
    root_name: String,
    open_documents: BTreeMap<String, OpenDocument>,
    stderr_tail: Arc<Mutex<VecDeque<u8>>>,
    stderr_thread: Option<std::thread::JoinHandle<()>>,
}

struct OpenDocument {
    version: u64,
    content_sha256: String,
}

struct DiagnosticSnapshot {
    version: Option<u64>,
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

impl LspProcess {
    fn try_wait(&mut self) -> Result<Option<std::process::ExitStatus>> {
        match self {
            Self::Host(child) => Ok(child.try_wait()?),
            Self::Exec(process) => process.try_wait(),
            Self::Service(process) => process.try_wait(),
        }
    }

    fn shutdown(&mut self) {
        match self {
            Self::Host(child) => {
                terminate_host_process_group(child.id());
                let _ = child.kill();
                let _ = child.wait();
            }
            Self::Exec(process) => {
                let _ = process.shutdown(Duration::from_secs(2));
            }
            Self::Service(process) => {
                let _ = process.shutdown(Duration::from_secs(2));
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
    ) -> Result<Self> {
        let cwd = match config.working_directory.as_deref() {
            Some(path) => resolve_workspace_path(workspace_root, &path.to_string_lossy())?,
            None => workspace_root.to_path_buf(),
        };
        let (process, stdin, stdout, stderr) = if let Some(lease) = lease {
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
                (LspProcess::Service(service), stdin, stdout, Some(stderr))
            } else {
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
                (LspProcess::Exec(process), stdin, stdout, Some(stderr))
            }
        } else {
            let (command, args) = stdio_command(server_name, config, workspace_root)?;
            let mut command_builder = Command::new(&command);
            command_builder
                .args(&args)
                .current_dir(&cwd)
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
            (LspProcess::Host(child), stdin, stdout, Some(stderr))
        };
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
            stdin,
            stdout_reader,
            next_id: 1,
            diagnostics: BTreeMap::new(),
            language_ids: config.language_ids.clone(),
            root_uri: root_uri.clone(),
            root_name: root_name.clone(),
            open_documents: BTreeMap::new(),
            stderr_tail,
            stderr_thread,
        };
        let mut initialize_params = json!({"processId": std::process::id(), "rootUri": root_uri, "workspaceFolders": [{"uri": root_uri, "name": root_name}], "capabilities": {"textDocument":{"synchronization":{"didSave":true,"dynamicRegistration":false},"hover":{},"definition":{},"references":{},"documentSymbol":{},"publishDiagnostics":{}},"workspace":{"symbol":{},"workspaceFolders":true,"configuration":true}}});
        if let Some(options) = config.initialization_options.clone()
            && let Some(params) = initialize_params.as_object_mut()
        {
            params.insert("initializationOptions".to_string(), options);
        }
        client.request("initialize", initialize_params)?;
        client.notify("initialized", json!({}))?;
        Ok(client)
    }

    fn call_with_diagnostic_timeout(
        &mut self,
        op: LspOperation,
        workspace_root: &Path,
        args: &Value,
        diagnostic_timeout: Duration,
    ) -> Result<String> {
        match op {
            LspOperation::WorkspaceSymbols => self.request_text(
                "workspace/symbol",
                json!({"query": string_arg(args, "query")?}),
            ),
            LspOperation::DocumentSymbols => {
                let document = self.open_document(workspace_root, args)?;
                self.request_text(
                    "textDocument/documentSymbol",
                    json!({"textDocument":{"uri":document.uri}}),
                )
            }
            LspOperation::Diagnostics => {
                let document = self.open_document(workspace_root, args)?;
                Ok(self
                    .wait_for_diagnostics(&document, diagnostic_timeout)?
                    .to_string())
            }
            LspOperation::Hover => {
                self.position_request(workspace_root, args, "textDocument/hover")
            }
            LspOperation::Definition => {
                self.position_request(workspace_root, args, "textDocument/definition")
            }
            LspOperation::References => {
                let (uri, pos) = self.document_position(workspace_root, args)?;
                self.request_text("textDocument/references", json!({"textDocument":{"uri":uri},"position":pos,"context":{"includeDeclaration":true}}))
            }
        }
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
        let document = self.open_document(workspace_root, args)?;
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
    fn open_document(&mut self, workspace_root: &Path, args: &Value) -> Result<OpenDocumentState> {
        let path = string_arg(args, "path")?;
        let full = resolve_workspace_path(workspace_root, path)?;
        let text = read_bounded_utf8(&full, MAX_LSP_DOCUMENT_BYTES, "LSP document")?;
        let uri = path_to_uri(&full)?;
        let content_sha256 = crate::environment_lock::sha256(text.as_bytes());
        if let Some(document) = self.open_documents.get(&uri) {
            if document.content_sha256 != content_sha256 {
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
    ) -> Result<Value> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(snapshot) = self.diagnostics.get(&document.uri)
                && diagnostic_snapshot_is_fresh(snapshot, document.version)
            {
                return Ok(snapshot.diagnostics.clone());
            }
            if Instant::now() >= deadline {
                bail!(
                    "timed out waiting for fresh diagnostics for document version {}",
                    document.version
                );
            }
            let message = self.read_message(deadline)?;
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
        }
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
        self.write(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))
            .map_err(|error| {
                lsp_transport_failure(format!("failed to write LSP request {method}: {error:#}"))
            })?;
        let deadline = Instant::now() + timeout;
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
        let body = serde_json::to_vec(message)?;
        write!(self.stdin, "Content-Length: {}\r\n\r\n", body.len())?;
        self.stdin.write_all(&body)?;
        self.stdin.flush()?;
        Ok(())
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
            self.diagnostics.insert(
                uri.to_string(),
                DiagnosticSnapshot {
                    version: params.get("version").and_then(Value::as_u64),
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

fn diagnostic_snapshot_is_fresh(snapshot: &DiagnosticSnapshot, version: u64) -> bool {
    snapshot
        .version
        .is_none_or(|published_version| published_version == version)
}

impl Drop for LspClient {
    fn drop(&mut self) {
        for uri in self.open_documents.keys().cloned().collect::<Vec<_>>() {
            let _ = self.notify("textDocument/didClose", json!({"textDocument":{"uri":uri}}));
        }
        let _ = self.request_with_timeout("shutdown", Value::Null, Duration::from_secs(2));
        let _ = self.notify("exit", json!(null));
        self.shutdown_transport();
    }
}

impl LspClient {
    fn shutdown_transport(&mut self) {
        self.process.shutdown();
        self.stdout_reader.join();
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
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
            diagnostics: json!([]),
        };
        let current = DiagnosticSnapshot {
            version: Some(2),
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
