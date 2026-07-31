//! Conservative ownership of one tailnet-only HTTPS endpoint through Tailscale Serve.

use anyhow::{Context, Result, anyhow, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ffi::OsStr;
use std::net::IpAddr;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

const MACOS_APP_CLI: &str = "/Applications/Tailscale.app/Contents/MacOS/Tailscale";
const MAX_COMMAND_ERROR_CHARS: usize = 2_000;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TailscaleState {
    Unavailable,
    Disconnected,
    Available,
    NeedsRepair,
    AuthorizationRequired,
    Conflict,
    Active,
    Error,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TailscaleStatus {
    pub state: TailscaleState,
    pub installed: bool,
    pub connected: bool,
    pub enabled: bool,
    pub active: bool,
    pub https_port: u16,
    pub backend_target: String,
    pub url: Option<String>,
    pub authorization_url: Option<String>,
    pub error: Option<String>,
    pub direct_lan_access: bool,
}

#[derive(Debug)]
pub struct TailscaleIntegration {
    web_port: u16,
    https_port: u16,
    enabled: bool,
    authorization_url: Option<String>,
    executable: Option<TailscaleExecutable>,
}

#[derive(Debug, Clone)]
struct TailscaleExecutable {
    path: PathBuf,
    force_cli_mode: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct RawStatus {
    #[serde(default)]
    backend_state: String,
    #[serde(default)]
    auth_url: String,
    #[serde(rename = "Self")]
    self_node: Option<RawSelf>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct RawSelf {
    #[serde(default)]
    dns_name: String,
    #[serde(default)]
    online: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndpointOwnership {
    Missing,
    Owned,
    Conflict,
}

#[derive(Debug)]
enum EnableOutcome {
    Active,
    AuthorizationRequired(String),
}

impl TailscaleIntegration {
    pub fn new(web_port: u16, https_port: u16, enabled: bool) -> Self {
        Self {
            web_port,
            https_port,
            enabled,
            authorization_url: None,
            executable: discover_executable(),
        }
    }

    #[cfg(test)]
    fn with_executable(web_port: u16, https_port: u16, enabled: bool, executable: PathBuf) -> Self {
        Self {
            web_port,
            https_port,
            enabled,
            authorization_url: None,
            executable: Some(TailscaleExecutable {
                path: executable,
                force_cli_mode: false,
            }),
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.authorization_url = None;
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn https_port(&self) -> u16 {
        self.https_port
    }

    pub fn status(&self, listen: &str) -> TailscaleStatus {
        match self.inspect() {
            Ok(inspected) => self.snapshot(listen, inspected),
            Err(error) => self.error_snapshot(listen, error),
        }
    }

    pub fn enable(&mut self, listen: &str) -> Result<TailscaleStatus> {
        let inspected = self.inspect()?;
        if inspected.ownership == EndpointOwnership::Conflict {
            bail!(
                "Tailscale HTTPS port {} already has a different Serve endpoint; pb left it unchanged",
                self.https_port
            );
        }
        if !inspected.installed {
            bail!("Tailscale is not installed");
        }
        if !inspected.connected {
            bail!("Tailscale is not connected to a tailnet");
        }
        if inspected.ownership == EndpointOwnership::Owned {
            self.enabled = true;
            self.authorization_url = None;
            return Ok(self.snapshot(listen, inspected));
        }

        match self.run_enable()? {
            EnableOutcome::Active => {
                self.enabled = true;
                self.authorization_url = None;
            }
            EnableOutcome::AuthorizationRequired(url) => {
                self.enabled = true;
                self.authorization_url = Some(url);
                let mut pending = inspected;
                pending.ownership = EndpointOwnership::Missing;
                return Ok(self.snapshot(listen, pending));
            }
        }
        let verified = self.inspect()?;
        if verified.ownership != EndpointOwnership::Owned {
            bail!("Tailscale reported success but did not retain pb's Serve endpoint");
        }
        Ok(self.snapshot(listen, verified))
    }

    pub fn disable(&mut self, listen: &str) -> Result<TailscaleStatus> {
        let inspected = self.inspect()?;
        if inspected.ownership == EndpointOwnership::Conflict {
            bail!(
                "Tailscale HTTPS port {} is not owned by pb; pb left it unchanged",
                self.https_port
            );
        }
        if inspected.ownership == EndpointOwnership::Owned {
            let executable = self
                .executable
                .as_ref()
                .context("Tailscale is not installed")?;
            let port = format!("--https={}", self.https_port);
            checked_output(executable, &["serve", &port, "off"])?;
        }
        self.enabled = false;
        self.authorization_url = None;
        let verified = self.inspect()?;
        if verified.ownership == EndpointOwnership::Owned {
            bail!("Tailscale did not remove pb's Serve endpoint");
        }
        Ok(self.snapshot(listen, verified))
    }

    /// Restore an explicitly enabled endpoint after Tailscale or the Mac restarts.
    pub fn reconcile(&mut self, listen: &str) -> TailscaleStatus {
        if !self.enabled {
            return self.status(listen);
        }
        self.enable(listen)
            .unwrap_or_else(|error| self.error_snapshot(listen, error))
    }

    fn run_enable(&self) -> Result<EnableOutcome> {
        let executable = self
            .executable
            .as_ref()
            .context("Tailscale is not installed")?;
        let port = format!("--https={}", self.https_port);
        let target = self.backend_target();
        let output = command_output(executable, &["serve", "--bg", &port, &target])?;
        let combined = combined_output(&output);
        if output.status.success() {
            return Ok(EnableOutcome::Active);
        }
        if let Some(url) = authorization_url(&combined) {
            return Ok(EnableOutcome::AuthorizationRequired(url));
        }
        Err(command_failure(executable, &output))
    }

    fn inspect(&self) -> Result<InspectedStatus> {
        let Some(executable) = &self.executable else {
            return Ok(InspectedStatus::unavailable());
        };
        let status_output = checked_output(executable, &["status", "--json", "--peers=false"])?;
        let raw: RawStatus = serde_json::from_slice(&status_output.stdout)
            .context("Tailscale returned invalid status JSON")?;
        let connected = raw.backend_state == "Running"
            && raw.self_node.as_ref().is_some_and(|node| node.online);
        let dns_name = raw
            .self_node
            .map(|node| node.dns_name.trim_end_matches('.').to_string())
            .filter(|name| !name.is_empty());
        let ownership = match checked_output(executable, &["serve", "status", "--json"]) {
            Ok(serve_output) => {
                let serve: Value = serde_json::from_slice(&serve_output.stdout)
                    .context("Tailscale returned invalid Serve status JSON")?;
                endpoint_ownership(&serve, self.https_port, &self.backend_target())
            }
            Err(_) if !connected => EndpointOwnership::Missing,
            Err(error) => return Err(error),
        };
        if !connected {
            return Ok(InspectedStatus {
                installed: true,
                connected: false,
                dns_name,
                auth_url: (!raw.auth_url.is_empty()).then_some(raw.auth_url),
                ownership,
            });
        }
        Ok(InspectedStatus {
            installed: true,
            connected: true,
            dns_name,
            auth_url: None,
            ownership,
        })
    }

    fn snapshot(&self, listen: &str, inspected: InspectedStatus) -> TailscaleStatus {
        let active = inspected.ownership == EndpointOwnership::Owned;
        let authorization_url = self.authorization_url.clone().or(inspected.auth_url);
        let state = if !inspected.installed {
            TailscaleState::Unavailable
        } else if !inspected.connected {
            TailscaleState::Disconnected
        } else if inspected.ownership == EndpointOwnership::Conflict {
            TailscaleState::Conflict
        } else if active {
            TailscaleState::Active
        } else if authorization_url.is_some() {
            TailscaleState::AuthorizationRequired
        } else if self.enabled {
            TailscaleState::NeedsRepair
        } else {
            TailscaleState::Available
        };
        let url = active
            .then(|| inspected.dns_name)
            .flatten()
            .map(|name| format!("https://{name}:{}/", self.https_port));
        TailscaleStatus {
            state,
            installed: inspected.installed,
            connected: inspected.connected,
            enabled: self.enabled,
            active,
            https_port: self.https_port,
            backend_target: self.backend_target(),
            url,
            authorization_url,
            error: None,
            direct_lan_access: direct_lan_access(listen),
        }
    }

    fn error_snapshot(&self, listen: &str, error: impl std::fmt::Display) -> TailscaleStatus {
        TailscaleStatus {
            state: TailscaleState::Error,
            installed: self.executable.is_some(),
            connected: false,
            enabled: self.enabled,
            active: false,
            https_port: self.https_port,
            backend_target: self.backend_target(),
            url: None,
            authorization_url: self.authorization_url.clone(),
            error: Some(truncate(&format!("{error:#}"), MAX_COMMAND_ERROR_CHARS)),
            direct_lan_access: direct_lan_access(listen),
        }
    }

    fn backend_target(&self) -> String {
        format!("http://127.0.0.1:{}", self.web_port)
    }
}

#[derive(Debug)]
struct InspectedStatus {
    installed: bool,
    connected: bool,
    dns_name: Option<String>,
    auth_url: Option<String>,
    ownership: EndpointOwnership,
}

impl InspectedStatus {
    fn unavailable() -> Self {
        Self {
            installed: false,
            connected: false,
            dns_name: None,
            auth_url: None,
            ownership: EndpointOwnership::Missing,
        }
    }
}

fn discover_executable() -> Option<TailscaleExecutable> {
    let app_cli = PathBuf::from(MACOS_APP_CLI);
    if app_cli.is_file() {
        return Some(TailscaleExecutable {
            path: app_cli,
            force_cli_mode: true,
        });
    }
    crate::host_environment::executable_in_path(OsStr::new("tailscale")).map(|path| {
        TailscaleExecutable {
            path,
            force_cli_mode: false,
        }
    })
}

fn command_output(executable: &TailscaleExecutable, args: &[&str]) -> Result<Output> {
    let mut command = Command::new(&executable.path);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if executable.force_cli_mode {
        command.env("TAILSCALE_BE_CLI", "1");
    }
    command
        .output()
        .with_context(|| format!("failed to run {}", executable.path.display()))
}

fn checked_output(executable: &TailscaleExecutable, args: &[&str]) -> Result<Output> {
    let output = command_output(executable, args)?;
    if output.status.success() {
        Ok(output)
    } else {
        Err(command_failure(executable, &output))
    }
}

fn command_failure(executable: &TailscaleExecutable, output: &Output) -> anyhow::Error {
    let detail = truncate(&combined_output(output), MAX_COMMAND_ERROR_CHARS);
    anyhow!(
        "{} exited with {}{}",
        executable.path.display(),
        output.status,
        if detail.is_empty() {
            String::new()
        } else {
            format!(": {detail}")
        }
    )
}

fn combined_output(output: &Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    format!("{stdout}\n{stderr}").trim().to_string()
}

fn authorization_url(output: &str) -> Option<String> {
    let re = Regex::new(r#"https://[^\s<>\"]+"#).expect("valid URL regex");
    re.find_iter(output)
        .map(|candidate| {
            candidate
                .as_str()
                .trim_end_matches(|character: char| {
                    matches!(character, '.' | ',' | ')' | ']' | '\'')
                })
                .to_string()
        })
        .find(|url| {
            url.contains("login.tailscale.com") || url.contains("login.tailscaleusercontent.com")
        })
}

fn endpoint_ownership(config: &Value, port: u16, target: &str) -> EndpointOwnership {
    let port_text = port.to_string();
    let mut port_sections = Vec::new();
    collect_port_sections(config, &port_text, &mut port_sections);
    if port_sections.is_empty() {
        EndpointOwnership::Missing
    } else {
        let mut handler_tables = Vec::new();
        for section in &port_sections {
            collect_handler_tables(section, &mut handler_tables);
        }
        if handler_tables.len() == 1
            && handler_tables[0].len() == 1
            && handler_tables[0]
                .get("/")
                .is_some_and(|handler| exact_proxy_handler(handler, target))
        {
            EndpointOwnership::Owned
        } else {
            EndpointOwnership::Conflict
        }
    }
}

fn collect_port_sections<'a>(value: &'a Value, port: &str, sections: &mut Vec<&'a Value>) {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                if key == port
                    || key == &format!("tcp:{port}")
                    || key.ends_with(&format!(":{port}"))
                {
                    sections.push(child);
                } else {
                    collect_port_sections(child, port, sections);
                }
            }
        }
        Value::Array(values) => {
            for child in values {
                collect_port_sections(child, port, sections);
            }
        }
        _ => {}
    }
}

fn collect_handler_tables<'a>(
    value: &'a Value,
    tables: &mut Vec<&'a serde_json::Map<String, Value>>,
) {
    match value {
        Value::Object(object) => {
            if let Some(Value::Object(handlers)) = object.get("Handlers") {
                tables.push(handlers);
            }
            for child in object.values() {
                collect_handler_tables(child, tables);
            }
        }
        Value::Array(values) => {
            for child in values {
                collect_handler_tables(child, tables);
            }
        }
        _ => {}
    }
}

fn exact_proxy_handler(value: &Value, expected: &str) -> bool {
    let Value::Object(handler) = value else {
        return false;
    };
    handler.len() == 1
        && handler.get("Proxy").is_some_and(|proxy| {
            proxy
                .as_str()
                .is_some_and(|proxy| proxy.trim_end_matches('/') == expected.trim_end_matches('/'))
        })
}

fn direct_lan_access(listen: &str) -> bool {
    listen
        .trim_matches(['[', ']'])
        .parse::<IpAddr>()
        .map(|ip| !ip.is_loopback())
        .unwrap_or(true)
}

fn truncate(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_config_ownership_is_scoped_to_one_https_port() {
        let owned = serde_json::json!({
            "TCP": {"8311": {"HTTPS": true}},
            "Web": {
                "clear.example.ts.net:8311": {
                    "Handlers": {"/": {"Proxy": "http://127.0.0.1:8311"}}
                },
                "clear.example.ts.net:443": {
                    "Handlers": {"/": {"Proxy": "http://127.0.0.1:3000"}}
                }
            }
        });
        assert_eq!(
            endpoint_ownership(&owned, 8311, "http://127.0.0.1:8311"),
            EndpointOwnership::Owned
        );
        assert_eq!(
            endpoint_ownership(&owned, 443, "http://127.0.0.1:8311"),
            EndpointOwnership::Conflict
        );
        assert_eq!(
            endpoint_ownership(&owned, 8443, "http://127.0.0.1:8311"),
            EndpointOwnership::Missing
        );

        let shared_port = serde_json::json!({
            "TCP": {"8311": {"HTTPS": true}},
            "Web": {
                "clear.example.ts.net:8311": {
                    "Handlers": {
                        "/": {"Proxy": "http://127.0.0.1:8311"},
                        "/other": {"Proxy": "http://127.0.0.1:3000"}
                    }
                }
            }
        });
        assert_eq!(
            endpoint_ownership(&shared_port, 8311, "http://127.0.0.1:8311"),
            EndpointOwnership::Conflict
        );
    }

    #[test]
    fn authorization_link_is_extracted_without_trailing_prose() {
        assert_eq!(
            authorization_url("Visit https://login.tailscale.com/admin/feature/abc, then retry.")
                .as_deref(),
            Some("https://login.tailscale.com/admin/feature/abc")
        );
    }

    #[test]
    fn loopback_and_network_listeners_remain_distinct() {
        assert!(!direct_lan_access("127.0.0.1"));
        assert!(!direct_lan_access("::1"));
        assert!(direct_lan_access("0.0.0.0"));
        assert!(direct_lan_access("192.168.1.10"));
    }

    #[test]
    fn unavailable_snapshot_never_claims_remote_access() {
        let manager = TailscaleIntegration {
            web_port: 8311,
            https_port: 8311,
            enabled: false,
            authorization_url: None,
            executable: None,
        };
        let status = manager.status("127.0.0.1");
        assert_eq!(status.state, TailscaleState::Unavailable);
        assert!(!status.active);
        assert_eq!(status.backend_target, "http://127.0.0.1:8311");
    }

    #[test]
    fn test_constructor_keeps_command_tests_explicit() {
        let manager = TailscaleIntegration::with_executable(
            8311,
            8443,
            false,
            PathBuf::from("/tmp/tailscale-test"),
        );
        assert_eq!(manager.https_port, 8443);
    }
}
