//! LSP server configuration and stdio-backed tool support.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const MAX_LSP_RESPONSE_BYTES: usize = 1024 * 1024;
const LSP_READ_TIMEOUT: Duration = Duration::from_secs(30);

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
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
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
                    registry.tools.insert(
                        tool_name.clone(),
                        LspToolSpec {
                            tool_name,
                            server_name: server_name.clone(),
                            operation: op,
                            description: op.description(&server_name),
                            input_schema: op.input_schema(),
                        },
                    );
                }
            }
            Err(err) => {
                let tool_name = unique_tool_name(&server_name, "status");
                registry.tools.insert(tool_name.clone(), LspToolSpec { tool_name, server_name: server_name.clone(), operation: LspOperation::Diagnostics, description: format!("LSP server {server_name} was configured but startup failed: {err:#}"), input_schema: json!({"type":"object","properties":{},"additionalProperties":false}) });
            }
        }
    }
    registry
}

pub fn call_tool(
    registry: &LspToolRegistry,
    workspace_root: &Path,
    tool_name: &str,
    arguments: &Value,
) -> Result<String> {
    let spec = registry
        .tool(tool_name)
        .with_context(|| format!("unknown LSP tool: {tool_name}"))?;
    if spec.tool_name.ends_with("_status") {
        bail!(spec.description.clone());
    }
    let session = registry
        .sessions
        .get(&spec.server_name)
        .with_context(|| format!("LSP server '{}' is not running", spec.server_name))?;
    let mut client = session
        .lock()
        .map_err(|_| anyhow!("LSP session for {} is poisoned", spec.server_name))?;
    match client.call(spec.operation, workspace_root, arguments) {
        Ok(result) => Ok(result),
        Err(first_error) => {
            let config = registry
                .servers
                .get(&spec.server_name)
                .context("LSP server configuration disappeared during restart")?;
            *client = LspClient::connect(
                &spec.server_name,
                config,
                workspace_root,
                registry.lease.as_ref(),
            )
            .with_context(|| {
                format!("LSP request failed ({first_error:#}) and the bounded restart also failed")
            })?;
            client.call(spec.operation, workspace_root, arguments)
        }
    }
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
                json!({"type":"object","properties":{"query":{"type":"string","description":"Symbol query text."}},"required":["query"],"additionalProperties":false})
            }
            Self::DocumentSymbols | Self::Diagnostics => {
                json!({"type":"object","properties":{"path":{"type":"string","description":"Project-relative file path."}},"required":["path"],"additionalProperties":false})
            }
            _ => {
                json!({"type":"object","properties":{"path":{"type":"string","description":"Project-relative file path."},"line":{"type":"integer","description":"1-indexed line number.","minimum":1},"character":{"type":"integer","description":"0-indexed UTF-16 character offset.","minimum":0}},"required":["path","line","character"],"additionalProperties":false})
            }
        }
    }
}

struct LspClient {
    process: LspProcess,
    stdin: ChildStdin,
    stdout_reader: crate::jsonrpc::FramedJsonReader,
    next_id: u64,
    diagnostics: BTreeMap<String, Value>,
    language_id: String,
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
            let mut child = Command::new(&command)
                .args(&args)
                .current_dir(&cwd)
                .envs(&config.env)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
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
            (LspProcess::Host(child), stdin, stdout, None)
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
            language_id: config
                .language_ids
                .first()
                .cloned()
                .unwrap_or_else(|| "plaintext".to_string()),
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

    fn call(&mut self, op: LspOperation, workspace_root: &Path, args: &Value) -> Result<String> {
        match op {
            LspOperation::WorkspaceSymbols => self.request_text(
                "workspace/symbol",
                json!({"query": string_arg(args, "query")?}),
            ),
            LspOperation::DocumentSymbols => {
                let uri = self.open_document(workspace_root, args)?;
                self.request_text(
                    "textDocument/documentSymbol",
                    json!({"textDocument":{"uri":uri}}),
                )
            }
            LspOperation::Diagnostics => {
                let uri = self.open_document(workspace_root, args)?;
                Ok(self
                    .diagnostics
                    .get(&uri)
                    .cloned()
                    .unwrap_or_else(|| json!([]))
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
        let uri = self.open_document(workspace_root, args)?;
        let line = args
            .get("line")
            .and_then(Value::as_u64)
            .context("line is required")?;
        let character = args
            .get("character")
            .and_then(Value::as_u64)
            .context("character is required")?;
        Ok((
            uri,
            json!({"line": line.saturating_sub(1), "character": character}),
        ))
    }
    fn open_document(&mut self, workspace_root: &Path, args: &Value) -> Result<String> {
        let path = string_arg(args, "path")?;
        let full = resolve_workspace_path(workspace_root, path)?;
        let text = std::fs::read_to_string(&full)
            .with_context(|| format!("failed to read {}", full.display()))?;
        let uri = path_to_uri(&full)?;
        let content_sha256 = crate::environment_lock::sha256(text.as_bytes());
        if let Some(document) = self.open_documents.get(&uri) {
            if document.content_sha256 != content_sha256 {
                let version = document.version.saturating_add(1);
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
            self.notify(
                "textDocument/didOpen",
                json!({"textDocument":{"uri":uri,"languageId":&self.language_id,"version":1,"text":text}}),
            )?;
            self.open_documents.insert(
                uri.clone(),
                OpenDocument {
                    version: 1,
                    content_sha256,
                },
            );
        }
        Ok(uri)
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
        self.write(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))?;
        let deadline = Instant::now() + timeout;
        loop {
            if Instant::now() > deadline {
                bail!("timed out waiting for LSP response to {method}");
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
    }
    fn write(&mut self, message: &Value) -> Result<()> {
        let body = serde_json::to_vec(message)?;
        write!(self.stdin, "Content-Length: {}\r\n\r\n", body.len())?;
        self.stdin.write_all(&body)?;
        self.stdin.flush()?;
        Ok(())
    }
    fn read_message(&mut self, deadline: Instant) -> Result<Value> {
        if let Some(status) = self.process.try_wait()? {
            bail!(
                "LSP server exited with status {status}: {}",
                self.stderr_text()
            );
        }
        self.stdout_reader
            .recv_until(deadline)
            .with_context(|| format!("LSP response failed: {}", self.stderr_text()))
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
        {
            self.diagnostics.insert(
                uri.to_string(),
                params
                    .get("diagnostics")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
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
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        for uri in self.open_documents.keys().cloned().collect::<Vec<_>>() {
            let _ = self.notify("textDocument/didClose", json!({"textDocument":{"uri":uri}}));
        }
        let _ = self.request_with_timeout("shutdown", Value::Null, Duration::from_secs(2));
        let _ = self.notify("exit", json!(null));
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
    Ok(format!("file://{}", path.canonicalize()?.to_string_lossy()))
}
fn string_arg<'a>(args: &'a Value, name: &str) -> Result<&'a str> {
    args.get(name)
        .and_then(Value::as_str)
        .with_context(|| format!("{name} is required"))
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
}
