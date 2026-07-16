//! LSP server configuration and stdio-backed tool support.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_LSP_RESPONSE_BYTES: usize = 1024 * 1024;
const LSP_READ_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LspConfig {
    pub servers: BTreeMap<String, LspServerConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LspServerConfig {
    pub command: Option<String>,
    pub container_image: Option<String>,
    pub container_runtime: Option<String>,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub working_directory: Option<PathBuf>,
    pub language_ids: Vec<String>,
    pub disabled: bool,
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
    let mut registry = LspToolRegistry {
        servers: servers.clone(),
        tools: BTreeMap::new(),
        sessions: BTreeMap::new(),
    };
    for (server_name, config) in servers {
        match LspClient::connect(&server_name, &config, workspace_root) {
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
    client.call(spec.operation, workspace_root, arguments)
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
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    diagnostics: BTreeMap<String, Value>,
    language_id: String,
}

impl LspClient {
    fn connect(server_name: &str, config: &LspServerConfig, workspace_root: &Path) -> Result<Self> {
        let (command, args) = stdio_command(server_name, config, workspace_root)?;
        let cwd = config
            .working_directory
            .as_deref()
            .map(|p| {
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    workspace_root.join(p)
                }
            })
            .unwrap_or_else(|| workspace_root.to_path_buf());
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
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .context("failed to open LSP server stdout")?,
        );
        let mut client = Self {
            child,
            stdin,
            stdout,
            next_id: 1,
            diagnostics: BTreeMap::new(),
            language_id: config
                .language_ids
                .first()
                .cloned()
                .unwrap_or_else(|| "plaintext".to_string()),
        };
        let root_uri = path_to_uri(workspace_root)?;
        client.request("initialize", json!({"processId": std::process::id(), "rootUri": root_uri, "workspaceFolders": [{"uri": root_uri, "name": workspace_root.file_name().and_then(|n| n.to_str()).unwrap_or("workspace")}], "capabilities": {"textDocument":{"hover":{},"definition":{},"references":{},"documentSymbol":{},"publishDiagnostics":{}},"workspace":{"symbol":{}}}}))?;
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
        self.notify(
            "textDocument/didOpen",
            json!({"textDocument":{"uri":uri,"languageId":&self.language_id,"version":1,"text":text}}),
        )?;
        Ok(uri)
    }
    fn request_text(&mut self, method: &str, params: Value) -> Result<String> {
        Ok(self.request(method, params)?.to_string())
    }
    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        self.write(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))?;
        let deadline = Instant::now() + LSP_READ_TIMEOUT;
        loop {
            if Instant::now() > deadline {
                bail!("timed out waiting for LSP response to {method}");
            }
            let msg = self.read_message()?;
            if let Some(method) = msg.get("method").and_then(Value::as_str) {
                self.handle_notification(method, msg.get("params").cloned().unwrap_or(Value::Null));
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
    fn read_message(&mut self) -> Result<Value> {
        if let Some(status) = self.child.try_wait()? {
            bail!("LSP server exited with status {status}");
        }
        let mut len = None;
        loop {
            let mut line = String::new();
            if self.stdout.read_line(&mut line)? == 0 {
                bail!("LSP server exited before response");
            }
            let t = line.trim_end_matches(['\r', '\n']);
            if t.is_empty() {
                break;
            }
            if let Some(v) = t.strip_prefix("Content-Length:") {
                let parsed = v.trim().parse::<usize>()?;
                if parsed > MAX_LSP_RESPONSE_BYTES {
                    bail!("LSP response is too large: {parsed} bytes");
                }
                len = Some(parsed);
            }
        }
        let mut body = vec![0; len.context("LSP response missing Content-Length header")?];
        self.stdout.read_exact(&mut body)?;
        Ok(serde_json::from_slice(&body)?)
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
}

impl Drop for LspClient {
    fn drop(&mut self) {
        let _ = self.notify("exit", json!(null));
        let _ = self.child.kill();
    }
}

fn stdio_command(
    server_name: &str,
    config: &LspServerConfig,
    workspace_root: &Path,
) -> Result<(String, Vec<String>)> {
    if let Some(image) = config
        .container_image
        .as_deref()
        .filter(|i| !i.trim().is_empty())
    {
        let runtime =
            crate::container::resolve_runtime_binary(config.container_runtime.as_deref())?;
        let mut args = vec![
            "run".to_string(),
            "-i".to_string(),
            "--rm".to_string(),
            "-v".to_string(),
            format!("{}:{}", workspace_root.display(), workspace_root.display()),
            "-w".to_string(),
            workspace_root.display().to_string(),
        ];
        for key in config.env.keys() {
            args.push("-e".to_string());
            args.push(key.clone());
        }
        args.push(image.trim().to_string());
        args.extend(config.args.clone());
        return Ok((runtime, args));
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
    fn container_stdio_command_mounts_workspace_at_same_path() {
        let config = LspServerConfig {
            container_image: Some("example/lsp".to_string()),
            container_runtime: Some("podman".to_string()),
            args: vec!["server".to_string()],
            ..Default::default()
        };
        let (cmd, args) = stdio_command("rust", &config, Path::new("/workspace/pb")).unwrap();
        assert_eq!(cmd, "podman");
        assert!(
            args.windows(2)
                .any(|w| w == ["-v", "/workspace/pb:/workspace/pb"])
        );
        assert!(args.windows(2).any(|w| w == ["-w", "/workspace/pb"]));
    }
}
