//! MCP server configuration and session-owned stdio client support.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const MCP_PROTOCOL_VERSION: &str = "2025-11-25";
const SUPPORTED_MCP_PROTOCOL_VERSIONS: &[&str] = &[
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    MCP_PROTOCOL_VERSION,
];
const MAX_MCP_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_MCP_TOOL_PAGES: usize = 64;
const MAX_MCP_TOOLS: usize = 4096;
const MCP_READ_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct McpConfig {
    pub servers: BTreeMap<String, McpServerConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct McpServerConfig {
    pub command: Option<String>,
    pub url: Option<String>,
    pub container_image: Option<String>,
    pub container_runtime: Option<String>,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub working_directory: Option<PathBuf>,
    pub capabilities: McpCapabilities,
    pub disabled: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct McpCapabilities {
    pub workspace: crate::session_environment::ServiceWorkspaceAccess,
    pub network: crate::session_environment::ServiceNetworkAccess,
    pub cache_ids: Vec<String>,
    /// Container environment key -> host environment variable name. Values are resolved only at
    /// launch and are never written to project configuration, argv, or the session ledger.
    pub secret_env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectMcpConfig {
    pub servers: BTreeMap<String, McpServerConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct McpToolSpec {
    pub tool_name: String,
    pub server_name: String,
    pub server_tool_name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Clone, Default)]
pub struct McpToolRegistry {
    pub servers: BTreeMap<String, McpServerConfig>,
    pub tools: BTreeMap<String, McpToolSpec>,
    sessions: BTreeMap<String, Arc<Mutex<McpClient>>>,
    lease: Option<Arc<crate::session_environment::SessionEnvironmentLease>>,
    workspace_root: PathBuf,
}

impl McpToolRegistry {
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn tool(&self, name: &str) -> Option<&McpToolSpec> {
        self.tools.get(name)
    }
}

impl ProjectMcpConfig {
    pub fn load(workspace_root: &Path) -> Result<Option<Self>> {
        let path = project_mcp_config_path(workspace_root);
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn save(&self, workspace_root: &Path) -> Result<()> {
        let path = project_mcp_config_path(workspace_root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let text =
            toml::to_string_pretty(self).context("failed to serialize project MCP config")?;
        std::fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))
    }
}

pub fn project_mcp_config_path(workspace_root: &Path) -> PathBuf {
    workspace_root.join(".pb").join("mcp.toml")
}

pub fn effective_servers(
    global: &McpConfig,
    project: Option<&ProjectMcpConfig>,
) -> BTreeMap<String, McpServerConfig> {
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
        .filter(|(_, config)| {
            !config.disabled
                && (config
                    .command
                    .as_deref()
                    .is_some_and(|c| !c.trim().is_empty())
                    || config.url.as_deref().is_some_and(|u| !u.trim().is_empty())
                    || config
                        .container_image
                        .as_deref()
                        .is_some_and(|i| !i.trim().is_empty()))
        })
        .collect()
}

pub fn discover_tools(
    servers: BTreeMap<String, McpServerConfig>,
    workspace_root: &Path,
) -> McpToolRegistry {
    discover_tools_inner(servers, workspace_root, None)
}

struct CachedMcpRegistry {
    fingerprint: String,
    registry: McpToolRegistry,
}

static SESSION_MCP_REGISTRIES: OnceLock<Mutex<HashMap<String, CachedMcpRegistry>>> =
    OnceLock::new();

pub fn discover_tools_for_session(
    session_id: &str,
    servers: BTreeMap<String, McpServerConfig>,
    workspace_root: &Path,
    lease: Arc<crate::session_environment::SessionEnvironmentLease>,
) -> McpToolRegistry {
    let fingerprint = crate::environment_lock::sha256(
        &serde_json::to_vec(&(workspace_root, &servers)).unwrap_or_default(),
    );
    let registries = SESSION_MCP_REGISTRIES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut registries) = registries.lock() {
        if let Some(cached) = registries.get(session_id)
            && cached.fingerprint == fingerprint
        {
            return cached.registry.clone();
        }
        let registry = discover_tools_inner(servers, workspace_root, Some(lease));
        registries.insert(
            session_id.to_string(),
            CachedMcpRegistry {
                fingerprint,
                registry: registry.clone(),
            },
        );
        return registry;
    }
    McpToolRegistry::default()
}

pub fn shutdown_session_services(session_id: &str) {
    if let Some(registries) = SESSION_MCP_REGISTRIES.get()
        && let Ok(mut registries) = registries.lock()
    {
        registries.remove(session_id);
    }
}

fn discover_tools_inner(
    servers: BTreeMap<String, McpServerConfig>,
    workspace_root: &Path,
    lease: Option<Arc<crate::session_environment::SessionEnvironmentLease>>,
) -> McpToolRegistry {
    let mut registry = McpToolRegistry {
        servers: servers.clone(),
        tools: BTreeMap::new(),
        sessions: BTreeMap::new(),
        lease: lease.clone(),
        workspace_root: workspace_root.to_path_buf(),
    };
    for (server_name, server_config) in servers {
        match McpClient::connect(&server_name, &server_config, workspace_root, lease.as_ref())
            .and_then(|mut client| {
                let tools = client.list_tools()?;
                Ok((client, tools))
            }) {
            Ok((client, server_tools)) => {
                registry
                    .sessions
                    .insert(server_name.clone(), Arc::new(Mutex::new(client)));
                for tool in server_tools {
                    let unique_name = unique_tool_name(&server_name, &tool.name);
                    registry.tools.insert(
                        unique_name.clone(),
                        McpToolSpec {
                            tool_name: unique_name,
                            server_name: server_name.clone(),
                            server_tool_name: tool.name,
                            description: tool.description.unwrap_or_else(|| {
                                format!("Tool provided by MCP server {server_name}")
                            }),
                            input_schema: normalize_input_schema(tool.input_schema),
                        },
                    );
                }
            }
            Err(err) => {
                let tool_name = unique_tool_name(&server_name, "status");
                registry.tools.insert(
                    tool_name.clone(),
                    McpToolSpec {
                        tool_name,
                        server_name: server_name.clone(),
                        server_tool_name: "status".to_string(),
                        description: format!(
                            "MCP server {server_name} was configured but tool discovery failed: {err:#}"
                        ),
                        input_schema: json!({
                            "type": "object",
                            "properties": {},
                            "additionalProperties": false
                        }),
                    },
                );
            }
        }
    }
    registry
}

pub fn call_tool(registry: &McpToolRegistry, tool_name: &str, arguments: &Value) -> Result<String> {
    let spec = registry
        .tool(tool_name)
        .with_context(|| format!("unknown MCP tool: {tool_name}"))?;
    let server = registry
        .servers
        .get(&spec.server_name)
        .with_context(|| format!("MCP server '{}' is no longer configured", spec.server_name))?;
    if spec.server_tool_name == "status" {
        bail!(spec.description.clone());
    }
    if let Some(session) = registry.sessions.get(&spec.server_name) {
        let mut client = session
            .lock()
            .map_err(|_| anyhow!("MCP session for {} is poisoned", spec.server_name))?;
        match client.call_tool(&spec.server_tool_name, arguments) {
            Ok(result) => return Ok(result),
            Err(first_error) => {
                *client = McpClient::connect(
                    &spec.server_name,
                    server,
                    &registry.workspace_root,
                    registry.lease.as_ref(),
                )
                .with_context(|| {
                    format!(
                        "MCP request failed ({first_error:#}) and the bounded restart also failed"
                    )
                })?;
                return client.call_tool(&spec.server_tool_name, arguments);
            }
        }
    }
    let mut client = McpClient::connect(&spec.server_name, server, &registry.workspace_root, None)?;
    client.call_tool(&spec.server_tool_name, arguments)
}

fn unique_tool_name(server_name: &str, tool_name: &str) -> String {
    format!(
        "mcp_{}_{}",
        sanitize_tool_name(server_name),
        sanitize_tool_name(tool_name)
    )
}

fn sanitize_tool_name(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "tool".to_string()
    } else {
        trimmed.to_string()
    }
}

fn normalize_input_schema(schema: Option<Value>) -> Value {
    schema.unwrap_or_else(|| {
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true
        })
    })
}

#[derive(Debug, Deserialize)]
struct ServerTool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default, rename = "inputSchema")]
    input_schema: Option<Value>,
}

enum McpClient {
    Stdio(StdioMcpClient),
}

impl McpClient {
    fn connect(
        server_name: &str,
        config: &McpServerConfig,
        workspace_root: &Path,
        lease: Option<&Arc<crate::session_environment::SessionEnvironmentLease>>,
    ) -> Result<Self> {
        if let Some(url) = config.url.as_deref().filter(|u| !u.trim().is_empty()) {
            if config.capabilities.network
                != crate::session_environment::ServiceNetworkAccess::Egress
            {
                bail!("remote MCP server {server_name} requires capabilities.network = 'egress'");
            }
            bail!(
                "remote HTTP MCP server {server_name} ({}) is not supported by the production session control plane; package it as a capability-declared container_image",
                url.trim()
            );
        }

        let mut client = Self::Stdio(StdioMcpClient::spawn(
            server_name,
            config,
            workspace_root,
            lease,
        )?);
        client.initialize(server_name)?;
        Ok(client)
    }

    fn initialize(&mut self, server_name: &str) -> Result<()> {
        let result = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "pb", "version": env!("CARGO_PKG_VERSION")}
                }),
            )
            .with_context(|| format!("failed to initialize MCP server {server_name}"))?;
        let negotiated = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .context("MCP initialize response omitted protocolVersion")?;
        if !SUPPORTED_MCP_PROTOCOL_VERSIONS.contains(&negotiated) {
            bail!("MCP server {server_name} selected unsupported protocol version {negotiated}");
        }
        self.notify("notifications/initialized", json!({}))?;
        Ok(())
    }

    fn list_tools(&mut self) -> Result<Vec<ServerTool>> {
        let mut all_tools = Vec::new();
        let mut cursor = None;
        let mut seen_cursors = BTreeSet::new();
        loop {
            let params = cursor
                .as_ref()
                .map_or_else(|| json!({}), |cursor| json!({"cursor": cursor}));
            let result = self.request("tools/list", params)?;
            let tools = result
                .get("tools")
                .cloned()
                .unwrap_or_else(|| Value::Array(vec![]));
            let page = serde_json::from_value::<Vec<ServerTool>>(tools)
                .context("failed to parse MCP tools/list response")?;
            if all_tools.len().saturating_add(page.len()) > MAX_MCP_TOOLS {
                bail!("MCP tools/list exceeded the {MAX_MCP_TOOLS} tool limit");
            }
            all_tools.extend(page);
            let Some(next_cursor) = result
                .get("nextCursor")
                .and_then(Value::as_str)
                .filter(|cursor| !cursor.is_empty())
                .map(ToString::to_string)
            else {
                break;
            };
            if seen_cursors.len() >= MAX_MCP_TOOL_PAGES {
                bail!("MCP tools/list exceeded the {MAX_MCP_TOOL_PAGES} page limit");
            }
            if !seen_cursors.insert(next_cursor.clone()) {
                bail!("MCP tools/list returned a repeated pagination cursor");
            }
            cursor = Some(next_cursor);
        }
        Ok(all_tools)
    }

    fn call_tool(&mut self, name: &str, arguments: &Value) -> Result<String> {
        let result = self.request(
            "tools/call",
            json!({
                "name": name,
                "arguments": arguments,
            }),
        )?;
        Ok(format_tool_result(&result))
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        match self {
            Self::Stdio(client) => client.request(method, params),
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        match self {
            Self::Stdio(client) => client.notify(method, params),
        }
    }
}

struct StdioMcpClient {
    process: McpProcess,
    stdin: Option<ChildStdin>,
    stdout_reader: crate::jsonrpc::JsonLineReader,
    next_id: u64,
    stderr_tail: Arc<Mutex<VecDeque<u8>>>,
    stderr_thread: Option<std::thread::JoinHandle<()>>,
}

enum McpProcess {
    Host(Child),
    Service(crate::container::ManagedServiceProcess),
}

impl McpProcess {
    fn shutdown(&mut self) {
        match self {
            Self::Host(child) => {
                let deadline = Instant::now() + Duration::from_millis(500);
                while Instant::now() < deadline {
                    if child.try_wait().ok().flatten().is_some() {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                let _ = child.kill();
                let _ = child.wait();
            }
            Self::Service(process) => {
                let _ = process.shutdown(Duration::from_secs(2));
            }
        }
    }
}

impl StdioMcpClient {
    fn spawn(
        server_name: &str,
        config: &McpServerConfig,
        workspace_root: &Path,
        lease: Option<&Arc<crate::session_environment::SessionEnvironmentLease>>,
    ) -> Result<Self> {
        let (process, stdin, stdout, stderr) = if let Some(lease) = lease {
            if let Some(image) = config
                .container_image
                .as_deref()
                .filter(|image| !image.trim().is_empty())
            {
                let mut service_env = config.env.clone();
                for (container_key, host_key) in &config.capabilities.secret_env {
                    let value = std::env::var(host_key).with_context(|| {
                        format!(
                            "MCP server {server_name} requires host secret environment variable {host_key}"
                        )
                    })?;
                    service_env.insert(container_key.clone(), value);
                }
                let mut service =
                    lease.spawn_service(crate::session_environment::SessionServiceSpec {
                        service_name: server_name.to_string(),
                        role: format!("mcp:{server_name}"),
                        kind: crate::session_environment::LeaseResourceKind::McpService,
                        image: image.trim().to_string(),
                        args: config.args.clone(),
                        env: service_env,
                        working_directory: config.working_directory.clone(),
                        cache_scope_sha256: crate::environment_lock::sha256(&serde_json::to_vec(
                            config,
                        )?),
                        workspace_access: config.capabilities.workspace,
                        network_access: config.capabilities.network,
                        cache_ids: config.capabilities.cache_ids.clone(),
                    })?;
                let stdin = service.take_stdin()?;
                let stdout = service.take_stdout()?;
                let stderr = service.take_stderr()?;
                (McpProcess::Service(service), stdin, stdout, Some(stderr))
            } else {
                bail!(
                    "MCP server {server_name} uses a host command; container-backed sessions require container_image so workspace, cache, and network capabilities can be enforced"
                );
            }
        } else {
            let (command, args) = stdio_command(server_name, config)?;
            let cwd = resolve_host_workdir(workspace_root, config.working_directory.as_deref())?;
            let mut command_builder = Command::new(&command);
            command_builder.args(&args).current_dir(cwd);
            command_builder.envs(&config.env);
            command_builder
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let mut child = command_builder
                .spawn()
                .with_context(|| format!("failed to start MCP server {server_name}: {command}"))?;
            let stdin = child
                .stdin
                .take()
                .context("failed to open MCP server stdin")?;
            let stdout = child
                .stdout
                .take()
                .context("failed to open MCP server stdout")?;
            let stderr = child
                .stderr
                .take()
                .context("failed to open MCP server stderr")?;
            (McpProcess::Host(child), stdin, stdout, Some(stderr))
        };
        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        let stderr_thread = stderr.map(|stderr| drain_stderr(stderr, Arc::clone(&stderr_tail)));
        let stdout_reader =
            crate::jsonrpc::JsonLineReader::spawn("MCP", stdout, MAX_MCP_RESPONSE_BYTES)?;
        Ok(Self {
            process,
            stdin: Some(stdin),
            stdout_reader,
            next_id: 1,
            stderr_tail,
            stderr_thread,
        })
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.write_message(&message)?;
        let deadline = Instant::now() + MCP_READ_TIMEOUT;
        loop {
            if Instant::now() > deadline {
                bail!("timed out waiting for MCP response to {method}");
            }
            let message = self.read_message(deadline)?;
            if let Some(server_method) = message.get("method").and_then(Value::as_str) {
                if let Some(server_id) = message.get("id").cloned() {
                    self.respond_to_server_request(server_id, server_method)?;
                }
                continue;
            }
            if message.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = message.get("error") {
                bail!("MCP request {method} failed: {error}");
            }
            return Ok(message.get("result").cloned().unwrap_or(Value::Null));
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        let message = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.write_message(&message)
    }

    fn write_message(&mut self, message: &Value) -> Result<()> {
        let stdin = self
            .stdin
            .as_mut()
            .context("MCP server stdin is already closed")?;
        serde_json::to_writer(&mut *stdin, message).context("failed to serialize MCP message")?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        Ok(())
    }

    fn respond_to_server_request(&mut self, id: Value, method: &str) -> Result<()> {
        let response = if method == "ping" {
            json!({"jsonrpc":"2.0","id":id,"result":{}})
        } else {
            json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":format!("pb does not expose MCP client method {method}")}})
        };
        self.write_message(&response)
    }

    fn read_message(&mut self, deadline: Instant) -> Result<Value> {
        self.stdout_reader
            .recv_until(deadline)
            .with_context(|| format!("MCP response failed: {}", self.stderr_text()))
    }

    fn stderr_text(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|tail| String::from_utf8_lossy(&tail.iter().copied().collect::<Vec<_>>()).into())
            .unwrap_or_default()
    }
}

impl Drop for StdioMcpClient {
    fn drop(&mut self) {
        // MCP stdio shutdown is transport-owned: close input first, then give the server a bounded
        // grace period before terminating and removing its session-owned process/container.
        drop(self.stdin.take());
        self.process.shutdown();
        self.stdout_reader.join();
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
    }
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

fn stdio_command(server_name: &str, config: &McpServerConfig) -> Result<(String, Vec<String>)> {
    if let Some(image) = config
        .container_image
        .as_deref()
        .filter(|i| !i.trim().is_empty())
    {
        bail!(
            "container MCP image '{}' requires an active session environment lease",
            image.trim()
        );
    }

    let command = config.command.as_deref().with_context(|| {
        format!("MCP server {server_name} has no command, url, or container_image")
    })?;
    Ok((command.to_string(), config.args.clone()))
}

fn resolve_host_workdir(workspace_root: &Path, configured: Option<&Path>) -> Result<PathBuf> {
    let canonical_root = workspace_root.canonicalize().with_context(|| {
        format!(
            "failed to resolve MCP workspace {}",
            workspace_root.display()
        )
    })?;
    if let Some(path) = configured
        && (path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir)))
    {
        bail!(
            "MCP working directory must be a workspace-relative path without parent traversal: {}",
            path.display()
        );
    }
    let candidate =
        configured.map_or_else(|| canonical_root.clone(), |path| canonical_root.join(path));
    let candidate = candidate.canonicalize().with_context(|| {
        format!(
            "failed to resolve MCP working directory {}",
            candidate.display()
        )
    })?;
    if !candidate.starts_with(&canonical_root) {
        bail!(
            "MCP working directory {} escapes workspace {}",
            candidate.display(),
            canonical_root.display()
        );
    }
    Ok(candidate)
}

fn format_tool_result(result: &Value) -> String {
    if let Some(content) = result.get("content").and_then(Value::as_array) {
        let mut parts = Vec::new();
        for item in content {
            match item.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = item.get("text").and_then(Value::as_str) {
                        parts.push(text.to_string());
                    }
                }
                _ => parts.push(item.to_string()),
            }
        }
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }
    result.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_config_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = ProjectMcpConfig::default();
        config.servers.insert(
            "local".to_string(),
            McpServerConfig {
                command: Some("node".to_string()),
                args: vec!["server.js".to_string()],
                env: BTreeMap::from([("TOKEN".to_string(), "secret".to_string())]),
                working_directory: Some(PathBuf::from("tools")),
                disabled: false,
                ..Default::default()
            },
        );
        config.save(dir.path()).unwrap();
        let loaded = ProjectMcpConfig::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded, config);
    }

    #[test]
    fn project_servers_override_and_disable_global_servers() {
        let global = McpConfig {
            servers: BTreeMap::from([
                (
                    "docs".to_string(),
                    McpServerConfig {
                        command: Some("global-docs".to_string()),
                        ..Default::default()
                    },
                ),
                (
                    "github".to_string(),
                    McpServerConfig {
                        command: Some("github".to_string()),
                        ..Default::default()
                    },
                ),
            ]),
        };
        let project = ProjectMcpConfig {
            servers: BTreeMap::from([
                (
                    "docs".to_string(),
                    McpServerConfig {
                        command: Some("project-docs".to_string()),
                        args: vec!["--repo".to_string()],
                        ..Default::default()
                    },
                ),
                (
                    "github".to_string(),
                    McpServerConfig {
                        disabled: true,
                        ..Default::default()
                    },
                ),
            ]),
        };
        let effective = effective_servers(&global, Some(&project));
        assert_eq!(effective.len(), 1);
        assert_eq!(
            effective
                .get("docs")
                .and_then(|server| server.command.as_deref()),
            Some("project-docs")
        );
        assert!(!effective.contains_key("github"));
    }

    #[test]
    fn effective_servers_keeps_http_and_container_transports() {
        let global = McpConfig {
            servers: BTreeMap::from([
                (
                    "remote".to_string(),
                    McpServerConfig {
                        url: Some("https://mcp.example.test".to_string()),
                        ..Default::default()
                    },
                ),
                (
                    "boxed".to_string(),
                    McpServerConfig {
                        container_image: Some("ghcr.io/example/mcp:latest".to_string()),
                        ..Default::default()
                    },
                ),
            ]),
        };
        let effective = effective_servers(&global, None);
        assert_eq!(effective.len(), 2);
        assert_eq!(
            effective
                .get("remote")
                .and_then(|server| server.url.as_deref()),
            Some("https://mcp.example.test")
        );
        assert_eq!(
            effective
                .get("boxed")
                .and_then(|server| server.container_image.as_deref()),
            Some("ghcr.io/example/mcp:latest")
        );
    }

    #[test]
    fn container_mcp_requires_a_session_owned_service() {
        let config = McpServerConfig {
            container_image: Some("ghcr.io/example/mcp:latest".to_string()),
            container_runtime: Some("podman".to_string()),
            args: vec!["--flag".to_string()],
            env: BTreeMap::from([("TOKEN".to_string(), "secret".to_string())]),
            ..Default::default()
        };
        let error = stdio_command("boxed", &config).unwrap_err().to_string();
        assert!(error.contains("active session environment lease"));
    }

    #[test]
    fn stdio_uses_newline_delimited_mcp_and_negotiates_the_current_protocol() {
        let script = r#"
while IFS= read -r line; do
  case "$line" in
    *'"method":"initialize"'*)
      printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"1"}}}'
      ;;
    *'"cursor":"next"'*)
      printf '%s\n' '{"jsonrpc":"2.0","id":3,"result":{"tools":[{"name":"fixture_tool_two","description":"fixture","inputSchema":{"type":"object"}}]}}'
      ;;
    *'"method":"tools/list"'*)
      printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"fixture_tool","description":"fixture","inputSchema":{"type":"object"}}],"nextCursor":"next"}}'
      ;;
  esac
done
"#;
        let config = McpServerConfig {
            command: Some("sh".to_string()),
            args: vec!["-c".to_string(), script.to_string()],
            ..Default::default()
        };
        let workspace = tempfile::tempdir().unwrap();
        let mut client = McpClient::connect("fixture", &config, workspace.path(), None).unwrap();
        let tools = client.list_tools().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "fixture_tool");
        assert_eq!(tools[1].name, "fixture_tool_two");
    }

    #[test]
    fn remote_http_mcp_fails_closed_until_streamable_http_is_supervised() {
        let config = McpServerConfig {
            url: Some("https://mcp.example.test".to_string()),
            capabilities: McpCapabilities {
                network: crate::session_environment::ServiceNetworkAccess::Egress,
                ..Default::default()
            },
            ..Default::default()
        };
        let workspace = tempfile::tempdir().unwrap();
        let error = McpClient::connect("remote", &config, workspace.path(), None)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("not supported by the production session control plane"));
    }

    #[test]
    fn host_mcp_working_directory_is_confined_to_the_workspace() {
        let directory = tempfile::tempdir().unwrap();
        let workspace = directory.path().join("workspace");
        let tools = workspace.join("tools");
        let outside = directory.path().join("outside");
        std::fs::create_dir_all(&tools).unwrap();
        std::fs::create_dir_all(&outside).unwrap();

        assert_eq!(
            resolve_host_workdir(&workspace, Some(Path::new("tools"))).unwrap(),
            tools.canonicalize().unwrap()
        );
        let error = resolve_host_workdir(&workspace, Some(Path::new("../outside")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("workspace-relative"));

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, workspace.join("escape")).unwrap();
            let error = resolve_host_workdir(&workspace, Some(Path::new("escape")))
                .unwrap_err()
                .to_string();
            assert!(error.contains("escapes workspace"));
        }
    }

    #[test]
    fn sanitizes_names_for_llm_tool_use() {
        assert_eq!(
            unique_tool_name("GitHub Enterprise", "search/repo"),
            "mcp_github_enterprise_search_repo"
        );
    }
}
