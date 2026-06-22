//! Marketplace discovery and per-project integration installation.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::lsp::{LspServerConfig, ProjectLspConfig};
use crate::mcp::{McpServerConfig, ProjectMcpConfig};

const MARKETPLACE_ORG: &str = "crunchy-pb";

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
    pub no_overwrite: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationInstallResponse {
    pub installed: InstalledIntegration,
    pub config_path: String,
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

pub fn list_installed(workspace_root: &Path) -> Result<Vec<InstalledIntegration>> {
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
    if let Some(config) = ProjectLspConfig::load(workspace_root)? {
        installed.extend(config.servers.into_iter().filter_map(|(name, server)| {
            server
                .container_image
                .map(|container_image| InstalledIntegration {
                    name,
                    kind: IntegrationKind::Lsp,
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

pub fn install(
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
        IntegrationKind::Lsp => {
            let mut config = ProjectLspConfig::load(workspace_root)?.unwrap_or_default();
            if request.no_overwrite && config.servers.contains_key(&name) {
                bail!("LSP integration '{name}' is already installed");
            }
            config.servers.insert(
                name.clone(),
                LspServerConfig {
                    container_image: Some(request.container_image.clone()),
                    container_runtime: Some(runtime),
                    ..Default::default()
                },
            );
            config.save(workspace_root)?;
            Ok(IntegrationInstallResponse {
                installed: InstalledIntegration {
                    name,
                    kind: IntegrationKind::Lsp,
                    container_image: request.container_image,
                    disabled: false,
                },
                config_path: crate::lsp::project_lsp_config_path(workspace_root)
                    .display()
                    .to_string(),
            })
        }
    }
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
        let response = install(
            dir.path(),
            IntegrationInstallRequest {
                kind: IntegrationKind::Mcp,
                container_image: "ghcr.io/crunchy-pb/sentry-mcp:latest".to_string(),
                name: None,
                runtime: Some("docker".to_string()),
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
    }
}
