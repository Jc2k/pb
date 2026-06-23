//! Marketplace discovery and integration installation.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use crate::config::{self, UserConfig};
use crate::lsp::LspServerConfig;
use crate::mcp::{McpServerConfig, ProjectMcpConfig};

const MARKETPLACE_ORG: &str = "crunchy-pb";
pub const CONFIG_SCHEMA_ANNOTATION: &str = "io.github.crunchy-pb.integration.config-schema";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum IntegrationKind {
    Mcp,
    Lsp,
}

impl IntegrationKind {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "mcp" => Ok(Self::Mcp),
            "lsp" => Ok(Self::Lsp),
            _ => bail!("integration kind must be 'mcp' or 'lsp'"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mcp => "mcp",
            Self::Lsp => "lsp",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MarketplaceIntegration {
    pub name: String,
    pub kind: IntegrationKind,
    pub description: String,
    pub icon_url: String,
    pub repo_url: String,
    pub container_image: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstalledIntegration {
    pub name: String,
    pub kind: IntegrationKind,
    pub container_image: String,
    pub disabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationInstallRequest {
    pub kind: IntegrationKind,
    pub container_image: String,
    pub name: Option<String>,
    pub runtime: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub no_overwrite: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationInstallResponse {
    pub installed: InstalledIntegration,
    pub config_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntegrationConfigSchema {
    pub container_image: String,
    pub annotation: String,
    pub schema: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct GithubRepo {
    name: String,
    html_url: String,
    description: Option<String>,
    topics: Option<Vec<String>>,
    owner: GithubOwner,
}

#[derive(Debug, Deserialize)]
struct GithubOwner {
    avatar_url: String,
}

pub async fn list_marketplace() -> Result<Vec<MarketplaceIntegration>> {
    let url = format!("https://api.github.com/orgs/{MARKETPLACE_ORG}/repos?per_page=100");
    let repos = reqwest::Client::new()
        .get(url)
        .header(reqwest::header::USER_AGENT, "pb-integration-marketplace")
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .context("failed to query GitHub repositories")?
        .error_for_status()
        .context("GitHub repository query failed")?
        .json::<Vec<GithubRepo>>()
        .await
        .context("failed to decode GitHub repositories")?;

    let mut integrations = Vec::new();
    for repo in repos {
        let topics = repo.topics.unwrap_or_default();
        for kind in [IntegrationKind::Mcp, IntegrationKind::Lsp] {
            if topics
                .iter()
                .any(|topic| topic.eq_ignore_ascii_case(kind.as_str()))
            {
                integrations.push(MarketplaceIntegration {
                    container_image: marketplace_container_image(&repo.name),
                    name: repo.name.clone(),
                    kind,
                    description: repo.description.clone().unwrap_or_default(),
                    icon_url: repo.owner.avatar_url.clone(),
                    repo_url: repo.html_url.clone(),
                });
            }
        }
    }
    integrations.sort_by(|a, b| {
        a.kind
            .as_str()
            .cmp(b.kind.as_str())
            .then(a.name.cmp(&b.name))
    });
    Ok(integrations)
}

pub fn marketplace_container_image(repo_name: &str) -> String {
    format!("ghcr.io/{MARKETPLACE_ORG}/{repo_name}:latest")
}

pub fn fetch_config_schema(container_image: &str) -> Result<IntegrationConfigSchema> {
    if container_image.trim().is_empty() {
        bail!("container image cannot be empty");
    }
    let output = Command::new("docker")
        .args(["manifest", "inspect", container_image])
        .output()
        .with_context(|| format!("failed to inspect container image {container_image}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("failed to inspect container image {container_image}: {stderr}");
    }
    let manifest: Value = serde_json::from_slice(&output.stdout)
        .with_context(|| format!("failed to decode manifest for {container_image}"))?;
    let schema_text = find_annotation(&manifest, CONFIG_SCHEMA_ANNOTATION);
    let schema = schema_text
        .map(|text| {
            serde_json::from_str(text)
                .context("failed to parse integration config schema annotation")
        })
        .transpose()?;
    Ok(IntegrationConfigSchema {
        container_image: container_image.to_string(),
        annotation: CONFIG_SCHEMA_ANNOTATION.to_string(),
        schema,
    })
}

fn find_annotation<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    if let Some(text) = value
        .get("annotations")
        .and_then(Value::as_object)
        .and_then(|annotations| annotations.get(key))
        .and_then(Value::as_str)
    {
        return Some(text);
    }
    match value {
        Value::Array(items) => items.iter().find_map(|item| find_annotation(item, key)),
        Value::Object(map) => map.values().find_map(|item| find_annotation(item, key)),
        _ => None,
    }
}

pub fn list_project_installed(workspace_root: &Path) -> Result<Vec<InstalledIntegration>> {
    let mut installed = Vec::new();
    if let Some(config) = ProjectMcpConfig::load(workspace_root)? {
        installed.extend(config.servers.into_iter().filter_map(|(name, server)| {
            server
                .container_image
                .map(|container_image| InstalledIntegration {
                    name,
                    kind: IntegrationKind::Mcp,
                    container_image,
                    disabled: server.disabled,
                })
        }));
    }
    installed.sort_by(|a, b| {
        a.kind
            .as_str()
            .cmp(b.kind.as_str())
            .then(a.name.cmp(&b.name))
    });
    Ok(installed)
}

pub fn install_project(
    workspace_root: &Path,
    request: IntegrationInstallRequest,
) -> Result<IntegrationInstallResponse> {
    let name = request
        .name
        .unwrap_or_else(|| name_from_image(&request.container_image));
    if name.trim().is_empty() || name.contains(['\n', '\r']) {
        bail!("integration name cannot be empty or contain newlines");
    }
    let runtime = request.runtime.unwrap_or_else(|| "docker".to_string());
    match request.kind {
        IntegrationKind::Mcp => {
            let mut config = ProjectMcpConfig::load(workspace_root)?.unwrap_or_default();
            if request.no_overwrite && config.servers.contains_key(&name) {
                bail!("MCP integration '{name}' is already installed");
            }
            config.servers.insert(
                name.clone(),
                McpServerConfig {
                    container_image: Some(request.container_image.clone()),
                    container_runtime: Some(runtime),
                    env: request.env.clone(),
                    ..Default::default()
                },
            );
            config.save(workspace_root)?;
            Ok(IntegrationInstallResponse {
                installed: InstalledIntegration {
                    name,
                    kind: IntegrationKind::Mcp,
                    container_image: request.container_image,
                    disabled: false,
                },
                config_path: crate::mcp::project_mcp_config_path(workspace_root)
                    .display()
                    .to_string(),
            })
        }
        IntegrationKind::Lsp => bail!("LSP integrations are configured globally"),
    }
}

pub fn list_global_lsp_installed() -> Result<Vec<InstalledIntegration>> {
    let config = UserConfig::load()?;
    let mut installed: Vec<_> = config
        .lsp
        .servers
        .into_iter()
        .filter_map(|(name, server)| {
            server
                .container_image
                .map(|container_image| InstalledIntegration {
                    name,
                    kind: IntegrationKind::Lsp,
                    container_image,
                    disabled: server.disabled,
                })
        })
        .collect();
    installed.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(installed)
}

pub fn install_global_lsp(
    request: IntegrationInstallRequest,
) -> Result<IntegrationInstallResponse> {
    if request.kind != IntegrationKind::Lsp {
        bail!("only LSP integrations can be installed globally");
    }
    let name = request
        .name
        .unwrap_or_else(|| name_from_image(&request.container_image));
    if name.trim().is_empty() || name.contains(['\n', '\r']) {
        bail!("integration name cannot be empty or contain newlines");
    }
    let runtime = request.runtime.unwrap_or_else(|| "docker".to_string());
    let mut user_config = UserConfig::load()?;
    if request.no_overwrite && user_config.lsp.servers.contains_key(&name) {
        bail!("LSP integration '{name}' is already installed");
    }
    user_config.lsp.servers.insert(
        name.clone(),
        LspServerConfig {
            container_image: Some(request.container_image.clone()),
            container_runtime: Some(runtime),
            env: request.env.clone(),
            ..Default::default()
        },
    );
    user_config.save()?;
    Ok(IntegrationInstallResponse {
        installed: InstalledIntegration {
            name,
            kind: IntegrationKind::Lsp,
            container_image: request.container_image,
            disabled: false,
        },
        config_path: config::config_path()?.display().to_string(),
    })
}

fn name_from_image(image: &str) -> String {
    image
        .rsplit('/')
        .next()
        .unwrap_or(image)
        .split(':')
        .next()
        .unwrap_or(image)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marketplace_container_image_uses_crunchy_org() {
        assert_eq!(
            marketplace_container_image("sentry-mcp"),
            "ghcr.io/crunchy-pb/sentry-mcp:latest"
        );
    }

    #[test]
    fn install_writes_project_scoped_mcp_config() {
        let dir = tempfile::tempdir().unwrap();
        let response = install_project(
            dir.path(),
            IntegrationInstallRequest {
                kind: IntegrationKind::Mcp,
                container_image: "ghcr.io/crunchy-pb/sentry-mcp:latest".to_string(),
                name: None,
                runtime: Some("docker".to_string()),
                env: BTreeMap::from([("SENTRY_DSN".to_string(), "https://example".to_string())]),
                no_overwrite: false,
            },
        )
        .unwrap();
        assert_eq!(response.installed.name, "sentry-mcp");
        let config = ProjectMcpConfig::load(dir.path()).unwrap().unwrap();
        assert_eq!(
            config.servers["sentry-mcp"].container_image.as_deref(),
            Some("ghcr.io/crunchy-pb/sentry-mcp:latest")
        );
        assert_eq!(
            config.servers["sentry-mcp"]
                .env
                .get("SENTRY_DSN")
                .map(String::as_str),
            Some("https://example")
        );
    }
}
