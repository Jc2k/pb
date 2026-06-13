//! MCP server configuration and stdio client support.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const MAX_MCP_RESPONSE_BYTES: usize = 1024 * 1024;
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
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub working_directory: Option<PathBuf>,
    pub disabled: bool,
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

#[derive(Debug, Clone, Default)]
pub struct McpToolRegistry {
    pub servers: BTreeMap<String, McpServerConfig>,
    pub tools: BTreeMap<String, McpToolSpec>,
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
                && config
                    .command
                    .as_deref()
                    .is_some_and(|c| !c.trim().is_empty())
        })
        .collect()
}

pub fn discover_tools(servers: BTreeMap<String, McpServerConfig>) -> McpToolRegistry {
    let mut registry = McpToolRegistry {
        servers: servers.clone(),
        tools: BTreeMap::new(),
    };
    for (server_name, server_config) in servers {
        match McpClient::connect(&server_name, &server_config)
            .and_then(|mut client| client.list_tools())
        {
            Ok(server_tools) => {
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
    let mut client = McpClient::connect(&spec.server_name, server)?;
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

struct McpClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl McpClient {
    fn connect(server_name: &str, config: &McpServerConfig) -> Result<Self> {
        let command = config
            .command
            .as_deref()
            .with_context(|| format!("MCP server {server_name} has no command"))?;
        let mut command_builder = Command::new(command);
        command_builder.args(&config.args);
        if let Some(cwd) = &config.working_directory {
            command_builder.current_dir(cwd);
        }
        command_builder.envs(&config.env);
        command_builder
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
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
        let mut client = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        };
        client.initialize(server_name)?;
        Ok(client)
    }

    fn initialize(&mut self, server_name: &str) -> Result<()> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pb", "version": env!("CARGO_PKG_VERSION")}
            }),
        )
        .with_context(|| format!("failed to initialize MCP server {server_name}"))?;
        self.notify("notifications/initialized", json!({}))?;
        Ok(())
    }

    fn list_tools(&mut self) -> Result<Vec<ServerTool>> {
        let result = self.request("tools/list", json!({}))?;
        let tools = result
            .get("tools")
            .cloned()
            .unwrap_or_else(|| Value::Array(vec![]));
        serde_json::from_value(tools).context("failed to parse MCP tools/list response")
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
            let message = self.read_message()?;
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
        let body = serde_json::to_vec(message).context("failed to serialize MCP message")?;
        write!(self.stdin, "Content-Length: {}\r\n\r\n", body.len())?;
        self.stdin.write_all(&body)?;
        self.stdin.flush()?;
        Ok(())
    }

    fn read_message(&mut self) -> Result<Value> {
        let mut content_length = None;
        loop {
            let mut line = String::new();
            let read = self.stdout.read_line(&mut line)?;
            if read == 0 {
                bail!("MCP server exited before sending a complete response");
            }
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                break;
            }
            if let Some((name, value)) = trimmed.split_once(':') {
                if name.eq_ignore_ascii_case("Content-Length") {
                    let length = value
                        .trim()
                        .parse::<usize>()
                        .context("invalid MCP Content-Length header")?;
                    if length > MAX_MCP_RESPONSE_BYTES {
                        bail!("MCP response is too large: {length} bytes");
                    }
                    content_length = Some(length);
                }
            }
        }
        let length = content_length.context("MCP response missing Content-Length header")?;
        let mut body = vec![0; length];
        self.stdout.read_exact(&mut body)?;
        serde_json::from_slice(&body)
            .map_err(|err| anyhow!("failed to parse MCP response JSON: {err}"))
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
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
    fn sanitizes_names_for_llm_tool_use() {
        assert_eq!(
            unique_tool_name("GitHub Enterprise", "search/repo"),
            "mcp_github_enterprise_search_repo"
        );
    }
}
