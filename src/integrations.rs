//! Marketplace discovery and integration installation.

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use reqwest::header::{ACCEPT, AUTHORIZATION, LOCATION, WWW_AUTHENTICATE};
use reqwest::{StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use crate::config::{self, UserConfig};
use crate::lsp::LspServerConfig;
use crate::mcp::{McpServerConfig, ProjectMcpConfig};

const MARKETPLACE_ORG: &str = "crunchy-pb";
pub const CONFIG_SCHEMA_ANNOTATION: &str = "uk.unrtd.pb.integration.config-schema";
pub const LSP_MANIFEST_ANNOTATION: &str = "uk.unrtd.pb.integration.lsp-manifest";
const LSP_PACKAGE_MANIFEST_VERSION: u32 = 1;
const MAX_LSP_PACKAGE_MANIFEST_BYTES: u64 = 64 * 1024;
const MAX_LSP_PACKAGE_ARGS: usize = 32;
const MAX_LSP_PACKAGE_LANGUAGE_IDS: usize = 32;
const MAX_LSP_PACKAGE_CACHE_IDS: usize = 16;
const MAX_REGISTRY_DOCUMENT_BYTES: u64 = 4 * 1024 * 1024;
const MAX_REGISTRY_TOKEN_BYTES: u64 = 64 * 1024;
const MAX_REGISTRY_REDIRECTS: usize = 5;
const MAX_REGISTRY_URL_CHARS: usize = 8_192;
const REGISTRY_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REGISTRY_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_container_image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verified_manifest_digest: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub disabled: bool,
    #[serde(default)]
    pub status: IntegrationStatus,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IntegrationStatus {
    #[default]
    Ready,
    Disabled,
    Unavailable,
    LegacyUnverified,
}

fn lsp_installation_status(config: &LspServerConfig) -> IntegrationStatus {
    let Some(image) = config.container_image.as_deref() else {
        return if config.disabled {
            IntegrationStatus::Disabled
        } else {
            IntegrationStatus::Ready
        };
    };
    container_installation_status(image, config.container_runtime.as_deref(), config.disabled)
}

fn container_installation_status(
    image: &str,
    configured_runtime: Option<&str>,
    disabled: bool,
) -> IntegrationStatus {
    if disabled {
        IntegrationStatus::Disabled
    } else if is_marketplace_container_image(image) && !image.contains("@sha256:") {
        IntegrationStatus::LegacyUnverified
    } else {
        let runtime = match configured_runtime {
            Some(binary) => crate::container::runtime_for_binary(binary).ok(),
            None => crate::container::detect_runtime(),
        };
        if runtime.is_some_and(|runtime| {
            if image.contains("@sha256:") {
                runtime.image_exists(image).unwrap_or(false)
            } else {
                runtime.info().is_ok()
            }
        }) {
            IntegrationStatus::Ready
        } else {
            IntegrationStatus::Unavailable
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntegrationInstallRequest {
    pub kind: IntegrationKind,
    pub container_image: String,
    #[serde(default)]
    pub source_container_image: Option<String>,
    pub name: Option<String>,
    pub runtime: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub lsp_manifest: Option<LspPackageManifest>,
    #[serde(default)]
    pub no_overwrite: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LspPackageManifest {
    pub version: u32,
    pub kind: IntegrationKind,
    pub server: LspPackageServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LspPackageServerConfig {
    pub args: Vec<String>,
    pub language_ids: Vec<String>,
    pub initialization_options: Option<Value>,
    #[serde(default = "default_lsp_package_workspace_access")]
    pub workspace_access: crate::session_environment::ServiceWorkspaceAccess,
    pub network_access: crate::session_environment::ServiceNetworkAccess,
    pub cache_ids: Vec<String>,
}

impl Default for LspPackageServerConfig {
    fn default() -> Self {
        Self {
            args: Vec::new(),
            language_ids: Vec::new(),
            initialization_options: None,
            workspace_access: default_lsp_package_workspace_access(),
            network_access: crate::session_environment::ServiceNetworkAccess::None,
            cache_ids: Vec::new(),
        }
    }
}

fn default_lsp_package_workspace_access() -> crate::session_environment::ServiceWorkspaceAccess {
    crate::session_environment::ServiceWorkspaceAccess::ReadOnly
}

impl LspPackageManifest {
    fn validate(&self) -> Result<()> {
        if self.version != LSP_PACKAGE_MANIFEST_VERSION {
            bail!(
                "unsupported LSP package manifest version {}; expected {}",
                self.version,
                LSP_PACKAGE_MANIFEST_VERSION
            );
        }
        if self.kind != IntegrationKind::Lsp {
            bail!("LSP package manifest kind must be 'lsp'");
        }
        if self.server.workspace_access
            != crate::session_environment::ServiceWorkspaceAccess::ReadOnly
        {
            bail!("packaged LSPs must use read_only workspace access");
        }
        if self.server.network_access != crate::session_environment::ServiceNetworkAccess::None {
            bail!("packaged LSPs must use network access 'none'");
        }
        validate_bounded_strings(
            "LSP package argument",
            &self.server.args,
            MAX_LSP_PACKAGE_ARGS,
            false,
        )?;
        validate_bounded_strings(
            "LSP package language id",
            &self.server.language_ids,
            MAX_LSP_PACKAGE_LANGUAGE_IDS,
            true,
        )?;
        validate_bounded_strings(
            "LSP package cache id",
            &self.server.cache_ids,
            MAX_LSP_PACKAGE_CACHE_IDS,
            true,
        )?;
        if self.server.language_ids.is_empty() {
            bail!("LSP package manifest must declare at least one language id");
        }
        if let Some(options) = &self.server.initialization_options
            && serde_json::to_vec(options)?.len() as u64 > MAX_LSP_PACKAGE_MANIFEST_BYTES
        {
            bail!("LSP package initialization options exceed the 65536-byte bound");
        }
        Ok(())
    }
}

fn validate_bounded_strings(
    label: &str,
    values: &[String],
    max_count: usize,
    identifier: bool,
) -> Result<()> {
    if values.len() > max_count {
        bail!("{label} list exceeds the {max_count}-entry bound");
    }
    for value in values {
        if value.is_empty()
            || value.len() > 4096
            || value.contains(['\0', '\n', '\r'])
            || (identifier
                && !value.chars().all(|character| {
                    character.is_ascii_alphanumeric() || "-_.".contains(character)
                }))
        {
            bail!("invalid {label}: {value:?}");
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationInstallResponse {
    pub installed: InstalledIntegration,
    pub config_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationRemoveResponse {
    pub removed: InstalledIntegration,
    pub config_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntegrationConfigSchema {
    pub container_image: String,
    pub source_container_image: String,
    pub manifest_digest: String,
    pub annotation: String,
    pub schema: Option<Value>,
    pub lsp_manifest_annotation: String,
    pub lsp_manifest: Option<LspPackageManifest>,
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
    crate::tls::install_default_crypto_provider();
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

pub fn is_marketplace_container_image(image: &str) -> bool {
    RegistryImage::parse(image).is_ok_and(|image| {
        image.registry == "ghcr.io"
            && image
                .repository
                .strip_prefix(MARKETPLACE_ORG)
                .is_some_and(|suffix| suffix.starts_with('/') && suffix.len() > 1)
    })
}

pub fn fetch_config_schema(container_image: &str) -> Result<IntegrationConfigSchema> {
    if container_image.trim().is_empty() {
        bail!("container image cannot be empty");
    }
    let image = RegistryImage::parse(container_image)?;
    let manifest = fetch_registry_manifest(&image)
        .with_context(|| format!("failed to inspect container image {container_image}"))?;
    if image.reference.starts_with("sha256:") && image.reference != manifest.digest {
        bail!(
            "registry returned manifest digest {} for requested digest {}",
            manifest.digest,
            image.reference
        );
    }
    let config = fetch_registry_config(&image, &manifest.value)
        .with_context(|| format!("failed to fetch image config for {container_image}"))?;
    let schema_text = find_annotation(&config, CONFIG_SCHEMA_ANNOTATION)
        .or_else(|| find_annotation(&manifest.value, CONFIG_SCHEMA_ANNOTATION));
    let schema = schema_text
        .map(|text| {
            serde_json::from_str(text)
                .context("failed to parse integration config schema annotation")
        })
        .transpose()?;
    let lsp_manifest_text = find_annotation(&config, LSP_MANIFEST_ANNOTATION)
        .or_else(|| find_annotation(&manifest.value, LSP_MANIFEST_ANNOTATION));
    let lsp_manifest = lsp_manifest_text
        .map(parse_lsp_package_manifest)
        .transpose()?;
    Ok(IntegrationConfigSchema {
        container_image: image.pinned_reference(&manifest.digest),
        source_container_image: container_image.to_string(),
        manifest_digest: manifest.digest,
        annotation: CONFIG_SCHEMA_ANNOTATION.to_string(),
        schema,
        lsp_manifest_annotation: LSP_MANIFEST_ANNOTATION.to_string(),
        lsp_manifest,
    })
}

fn parse_lsp_package_manifest(text: &str) -> Result<LspPackageManifest> {
    if text.len() as u64 > MAX_LSP_PACKAGE_MANIFEST_BYTES {
        bail!("LSP package manifest exceeds the 65536-byte bound");
    }
    let manifest: LspPackageManifest =
        serde_json::from_str(text).context("failed to parse LSP package manifest annotation")?;
    manifest.validate()?;
    Ok(manifest)
}

pub fn load_lsp_package_manifest(path: &Path) -> Result<LspPackageManifest> {
    let file = File::open(path)
        .with_context(|| format!("failed to open LSP package manifest {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to stat LSP package manifest {}", path.display()))?;
    if !metadata.is_file() || metadata.len() > MAX_LSP_PACKAGE_MANIFEST_BYTES {
        bail!(
            "LSP package manifest {} exceeds the {}-byte input bound",
            path.display(),
            MAX_LSP_PACKAGE_MANIFEST_BYTES
        );
    }
    let mut text = String::new();
    file.take(MAX_LSP_PACKAGE_MANIFEST_BYTES.saturating_add(1))
        .read_to_string(&mut text)
        .with_context(|| format!("failed to read LSP package manifest {}", path.display()))?;
    if text.len() as u64 > MAX_LSP_PACKAGE_MANIFEST_BYTES {
        bail!(
            "LSP package manifest {} grew beyond the {}-byte input bound",
            path.display(),
            MAX_LSP_PACKAGE_MANIFEST_BYTES
        );
    }
    parse_lsp_package_manifest(&text)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RegistryImage {
    registry: String,
    repository: String,
    reference: String,
}

impl RegistryImage {
    fn parse(image: &str) -> Result<Self> {
        let image = image.trim();
        if image.is_empty() {
            bail!("container image cannot be empty");
        }
        let (name, reference) = split_image_reference(image);
        let mut parts = name.splitn(2, '/');
        let registry = parts.next().unwrap_or_default();
        let repository = parts.next().unwrap_or_default();
        if registry.is_empty() || repository.is_empty() {
            bail!("container image must include a registry and repository");
        }
        Ok(Self {
            registry: registry.trim_end_matches('.').to_ascii_lowercase(),
            repository: repository.to_ascii_lowercase(),
            reference: reference.to_string(),
        })
    }

    fn manifest_url(&self) -> String {
        format!(
            "https://{}/v2/{}/manifests/{}",
            self.registry, self.repository, self.reference
        )
    }

    fn blob_url(&self, digest: &str) -> String {
        format!(
            "https://{}/v2/{}/blobs/{}",
            self.registry, self.repository, digest
        )
    }

    fn manifest_digest_url(&self, digest: &str) -> String {
        format!(
            "https://{}/v2/{}/manifests/{}",
            self.registry, self.repository, digest
        )
    }

    fn pinned_reference(&self, digest: &str) -> String {
        format!("{}/{}@{digest}", self.registry, self.repository)
    }
}

fn split_image_reference(image: &str) -> (&str, &str) {
    if let Some((name, digest)) = image.rsplit_once('@') {
        return (name, digest);
    }
    let last_slash = image.rfind('/');
    let last_colon = image.rfind(':');
    if let Some(colon) = last_colon
        && last_slash.is_none_or(|slash| colon > slash)
    {
        return (&image[..colon], &image[colon + 1..]);
    }
    (image, "latest")
}

fn validate_source_image_identity(request: &IntegrationInstallRequest) -> Result<()> {
    let Some(source) = request.source_container_image.as_deref() else {
        return Ok(());
    };
    let source = RegistryImage::parse(source)?;
    let installed = RegistryImage::parse(&request.container_image)?;
    if source.registry != installed.registry || source.repository != installed.repository {
        bail!("source container image does not name the installed image repository");
    }
    Ok(())
}

struct RegistryDocument {
    value: Value,
    digest: String,
}

fn fetch_registry_manifest(image: &RegistryImage) -> Result<RegistryDocument> {
    let accept = [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
    .join(", ");
    fetch_registry_json_document(image, &image.manifest_url(), &accept, "image manifest")
}

fn fetch_registry_config(image: &RegistryImage, manifest: &Value) -> Result<Value> {
    let image_manifest = if manifest.get("config").is_some() {
        manifest.clone()
    } else {
        let digest = select_runnable_manifest_digest(manifest)
            .context("image index does not contain a runnable manifest digest")?;
        fetch_registry_json_document_with_digest(
            image,
            &image.manifest_digest_url(digest),
            &[
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.v2+json",
            ]
            .join(", "),
            "platform image manifest",
            digest,
        )?
        .value
    };
    let digest = image_manifest
        .get("config")
        .and_then(|config| config.get("digest"))
        .and_then(Value::as_str)
        .context("image manifest does not contain a config digest")?;
    Ok(fetch_registry_json_document_with_digest(
        image,
        &image.blob_url(digest),
        &[
            "application/vnd.oci.image.config.v1+json",
            "application/vnd.docker.container.image.v1+json",
            "application/octet-stream",
        ]
        .join(", "),
        "image config",
        digest,
    )?
    .value)
}

fn fetch_registry_json_document_with_digest(
    image: &RegistryImage,
    url: &str,
    accept: &str,
    context: &str,
    expected_digest: &str,
) -> Result<RegistryDocument> {
    let document = fetch_registry_json_document(image, url, accept, context)?;
    verify_registry_document_digest(document, expected_digest, context)
}

fn verify_registry_document_digest(
    document: RegistryDocument,
    expected_digest: &str,
    context: &str,
) -> Result<RegistryDocument> {
    if !is_sha256_digest(expected_digest) {
        bail!("{context} uses an invalid sha256 digest");
    }
    if document.digest != expected_digest {
        bail!("{context} content does not match the digest selected by its parent manifest");
    }
    Ok(document)
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn select_runnable_manifest_digest(manifest: &Value) -> Option<&str> {
    let manifests = manifest.get("manifests")?.as_array()?;
    manifests
        .iter()
        .filter(|item| is_runnable_manifest(item))
        .filter(|item| manifest_platform_priority(item) > 0)
        .max_by_key(|item| manifest_platform_priority(item))
        .and_then(|item| item.get("digest"))
        .and_then(Value::as_str)
}

fn is_runnable_manifest(manifest: &Value) -> bool {
    let platform = manifest.get("platform");
    let os = platform
        .and_then(|platform| platform.get("os"))
        .and_then(Value::as_str);
    if matches!(os, Some("unknown")) {
        return false;
    }

    let media_type = manifest
        .get("mediaType")
        .and_then(Value::as_str)
        .unwrap_or_default();
    media_type.contains("manifest.v1") || media_type.contains("manifest.v2")
}

fn manifest_platform_priority(manifest: &Value) -> u8 {
    let platform = manifest.get("platform");
    let os = platform
        .and_then(|platform| platform.get("os"))
        .and_then(Value::as_str);
    let architecture = platform
        .and_then(|platform| platform.get("architecture"))
        .and_then(Value::as_str);

    let host_architecture = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else if cfg!(target_arch = "x86_64") {
        "amd64"
    } else {
        std::env::consts::ARCH
    };
    match (os, architecture) {
        (Some("linux"), Some(candidate)) if candidate == host_architecture => 3,
        (Some("linux"), None) => 1,
        _ => 0,
    }
}

fn fetch_registry_json_document(
    image: &RegistryImage,
    url: &str,
    accept: &str,
    context: &str,
) -> Result<RegistryDocument> {
    let url = parse_registry_url(url)?;
    let mut response = send_public_registry_get(url.clone(), Some(accept), None)?;
    if response.status() == StatusCode::UNAUTHORIZED {
        let challenge = response
            .headers()
            .get(WWW_AUTHENTICATE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        if let Some(challenge) = challenge
            && let Some(token) = fetch_bearer_token(&challenge, image)?
        {
            response = send_public_registry_get(url, Some(accept), Some(&token))?;
        }
    }
    read_bounded_registry_document(
        response.error_for_status()?,
        MAX_REGISTRY_DOCUMENT_BYTES,
        context,
    )
}

fn read_bounded_registry_document(
    response: reqwest::blocking::Response,
    max_bytes: u64,
    context: &str,
) -> Result<RegistryDocument> {
    let declared_digest = response
        .headers()
        .get("docker-content-digest")
        .and_then(|value| value.to_str().ok())
        .filter(|value| is_sha256_digest(value))
        .map(str::to_string);
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes)
    {
        bail!("{context} exceeds the {max_bytes}-byte response bound");
    }
    let mut bytes = Vec::new();
    response
        .take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {context}"))?;
    if bytes.len() as u64 > max_bytes {
        bail!("{context} exceeds the {max_bytes}-byte response bound");
    }
    let value =
        serde_json::from_slice(&bytes).with_context(|| format!("failed to decode {context}"))?;
    let computed_digest = format!("sha256:{}", crate::environment_lock::sha256(&bytes));
    if declared_digest
        .as_deref()
        .is_some_and(|declared| declared != computed_digest)
    {
        bail!("{context} content does not match its declared registry digest");
    }
    Ok(RegistryDocument {
        value,
        digest: computed_digest,
    })
}

fn fetch_bearer_token(challenge: &str, image: &RegistryImage) -> Result<Option<String>> {
    let Some(params) = challenge.strip_prefix("Bearer ") else {
        return Ok(None);
    };
    let mut realm = None;
    let mut service = None;
    for part in params.split(',') {
        let Some((key, value)) = part.trim().split_once('=') else {
            continue;
        };
        let value = value.trim_matches('"');
        match key {
            "realm" => realm = Some(value.to_string()),
            "service" => service = Some(value.to_string()),
            _ => {}
        }
    }
    let Some(realm) = realm else {
        return Ok(None);
    };
    let mut realm = parse_registry_url(&realm).context("registry auth realm is not safe")?;
    realm
        .query_pairs_mut()
        .append_pair("scope", &format!("repository:{}:pull", image.repository));
    if let Some(service) = service {
        realm.query_pairs_mut().append_pair("service", &service);
    }
    let token = read_bounded_registry_json(
        send_public_registry_get(realm, None, None)?.error_for_status()?,
        MAX_REGISTRY_TOKEN_BYTES,
        "registry auth token",
    )?;
    Ok(token
        .get("token")
        .or_else(|| token.get("access_token"))
        .and_then(Value::as_str)
        .map(str::to_string))
}

fn parse_registry_url(value: &str) -> Result<Url> {
    let url = Url::parse(value).with_context(|| format!("invalid registry URL: {value}"))?;
    validate_registry_url(&url)?;
    Ok(url)
}

fn validate_registry_url(url: &Url) -> Result<()> {
    if url.as_str().chars().count() > MAX_REGISTRY_URL_CHARS {
        bail!("registry URL exceeds {MAX_REGISTRY_URL_CHARS} characters");
    }
    crate::public_network::validate_public_http_url(url)?;
    if url.scheme() != "https" {
        bail!("registry and authentication URLs must use https");
    }
    Ok(())
}

fn registry_origin(url: &Url) -> Option<(String, String, u16)> {
    Some((
        url.scheme().to_string(),
        url.host_str()?.to_ascii_lowercase(),
        url.port_or_known_default()?,
    ))
}

fn send_public_registry_get(
    mut url: Url,
    accept: Option<&str>,
    bearer_token: Option<&str>,
) -> Result<reqwest::blocking::Response> {
    crate::tls::install_default_crypto_provider();
    let authenticated_origin = bearer_token.and_then(|_| registry_origin(&url));
    for redirect_count in 0..=MAX_REGISTRY_REDIRECTS {
        validate_registry_url(&url)?;
        let (host, address) = crate::public_network::resolve_public_target_blocking(&url)?;
        let client = Client::builder()
            .user_agent("pb-integration-config-schema")
            .connect_timeout(REGISTRY_CONNECT_TIMEOUT)
            .timeout(REGISTRY_REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .resolve(&host, address)
            .build()
            .context("failed to build pinned registry client")?;
        let mut request = client.get(url.clone());
        if let Some(accept) = accept {
            request = request.header(ACCEPT, accept);
        }
        if let (Some(token), Some(origin)) = (bearer_token, authenticated_origin.as_ref())
            && registry_origin(&url).as_ref() == Some(origin)
        {
            request = request.header(AUTHORIZATION, format!("Bearer {token}"));
        }
        let response = request
            .send()
            .with_context(|| format!("failed to fetch registry URL {url}"))?;
        if !response.status().is_redirection() {
            return Ok(response);
        }
        if redirect_count == MAX_REGISTRY_REDIRECTS {
            bail!("registry request exceeded {MAX_REGISTRY_REDIRECTS} redirects");
        }
        let location = response
            .headers()
            .get(LOCATION)
            .context("registry redirect omitted Location header")?
            .to_str()
            .context("registry redirect Location was not valid UTF-8")?;
        url = url
            .join(location)
            .with_context(|| format!("invalid registry redirect target: {location}"))?;
        validate_registry_url(&url).context("registry redirect target is not safe")?;
    }
    unreachable!("bounded registry redirect loop always returns or fails")
}

fn read_bounded_registry_json(
    response: reqwest::blocking::Response,
    max_bytes: u64,
    context: &str,
) -> Result<Value> {
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes)
    {
        bail!("{context} exceeds the {max_bytes}-byte response bound");
    }
    let mut bytes = Vec::new();
    response
        .take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {context}"))?;
    if bytes.len() as u64 > max_bytes {
        bail!("{context} exceeds the {max_bytes}-byte response bound");
    }
    serde_json::from_slice(&bytes).with_context(|| format!("failed to decode {context}"))
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
    if let Some(text) = value
        .get("config")
        .and_then(|config| config.get("Labels"))
        .or_else(|| value.get("Labels"))
        .and_then(Value::as_object)
        .and_then(|labels| labels.get(key))
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
            let status = server.container_image.as_deref().map(|image| {
                container_installation_status(
                    image,
                    server.container_runtime.as_deref(),
                    server.disabled,
                )
            });
            server
                .container_image
                .map(|container_image| InstalledIntegration {
                    name,
                    kind: IntegrationKind::Mcp,
                    container_image,
                    source_container_image: server.source_container_image,
                    verified_manifest_digest: server.verified_manifest_digest,
                    env: server.env,
                    disabled: server.disabled,
                    status: status.unwrap_or(IntegrationStatus::Unavailable),
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
    validate_source_image_identity(&request)?;
    let name = request
        .name
        .clone()
        .unwrap_or_else(|| name_from_image(&request.container_image));
    if name.trim().is_empty() || name.contains(['\n', '\r']) {
        bail!("integration name cannot be empty or contain newlines");
    }
    match request.kind {
        IntegrationKind::Mcp => {
            if request.lsp_manifest.is_some() {
                bail!("an LSP package manifest cannot configure an MCP integration");
            }
            if is_marketplace_container_image(&request.container_image)
                && !request.container_image.contains("@sha256:")
            {
                bail!("marketplace integrations must be installed by immutable OCI digest");
            }
            let container_runtime = requested_runtime(request.runtime.as_deref())?;
            if is_marketplace_container_image(&request.container_image) {
                let runtime = match container_runtime.as_deref() {
                    Some(binary) => Some(crate::container::runtime_for_binary(binary)?),
                    None => crate::container::detect_runtime(),
                };
                if let Some(runtime) = runtime {
                    let runtime_info = runtime.info()?;
                    let _image_operation = crate::container::acquire_image_operation_lock(
                        &runtime_info.binary,
                        &request.container_image,
                    )?;
                    if !runtime.image_exists(&request.container_image)? {
                        runtime.pull(&request.container_image).with_context(|| {
                            format!(
                                "failed to install pinned MCP image {}",
                                request.container_image
                            )
                        })?;
                    }
                }
            }
            let status = container_installation_status(
                &request.container_image,
                container_runtime.as_deref(),
                false,
            );
            let server_config = McpServerConfig {
                container_image: Some(request.container_image.clone()),
                source_container_image: request.source_container_image.clone(),
                verified_manifest_digest: request
                    .container_image
                    .rsplit_once("@sha256:")
                    .map(|(_, digest)| format!("sha256:{digest}")),
                container_runtime,
                env: request.env.clone(),
                ..Default::default()
            };
            let verified_manifest_digest = server_config.verified_manifest_digest.clone();
            ProjectMcpConfig::mutate(workspace_root, |config| {
                if request.no_overwrite && config.servers.contains_key(&name) {
                    bail!("MCP integration '{name}' is already installed");
                }
                config.servers.insert(name.clone(), server_config);
                Ok(())
            })?;
            Ok(IntegrationInstallResponse {
                installed: InstalledIntegration {
                    name,
                    kind: IntegrationKind::Mcp,
                    container_image: request.container_image,
                    source_container_image: request.source_container_image,
                    verified_manifest_digest,
                    env: request.env,
                    disabled: false,
                    status,
                },
                config_path: crate::mcp::project_mcp_config_path(workspace_root)
                    .display()
                    .to_string(),
            })
        }
        IntegrationKind::Lsp => bail!("LSP integrations are configured globally"),
    }
}

pub fn remove_project(
    workspace_root: &Path,
    kind: IntegrationKind,
    name: &str,
) -> Result<IntegrationRemoveResponse> {
    if kind != IntegrationKind::Mcp {
        bail!("LSP integrations are configured globally");
    }
    if name.trim().is_empty() || name.contains(['\n', '\r']) {
        bail!("integration name cannot be empty or contain newlines");
    }
    let server = ProjectMcpConfig::mutate(workspace_root, |config| {
        let server = config
            .servers
            .get(name)
            .with_context(|| format!("MCP integration '{name}' is not installed"))?;
        if server.container_image.is_none() {
            bail!("configured MCP server is not a container integration");
        }
        Ok(config
            .servers
            .remove(name)
            .expect("server was checked above"))
    })?;
    let status = server.container_image.as_deref().map(|image| {
        container_installation_status(image, server.container_runtime.as_deref(), server.disabled)
    });
    let container_image = server
        .container_image
        .context("configured MCP server is not a container integration")?;
    Ok(IntegrationRemoveResponse {
        removed: InstalledIntegration {
            name: name.to_string(),
            kind,
            container_image,
            source_container_image: server.source_container_image,
            verified_manifest_digest: server.verified_manifest_digest,
            env: server.env,
            disabled: server.disabled,
            status: status.unwrap_or(IntegrationStatus::Unavailable),
        },
        config_path: crate::mcp::project_mcp_config_path(workspace_root)
            .display()
            .to_string(),
    })
}

pub fn list_global_lsp_installed() -> Result<Vec<InstalledIntegration>> {
    let config = UserConfig::load()?;
    let mut installed: Vec<_> = config
        .lsp
        .servers
        .into_iter()
        .filter_map(|(name, server)| {
            let status = lsp_installation_status(&server);
            server
                .container_image
                .map(|container_image| InstalledIntegration {
                    name,
                    kind: IntegrationKind::Lsp,
                    container_image,
                    source_container_image: server.source_container_image,
                    verified_manifest_digest: server.verified_manifest_digest,
                    env: server.env,
                    disabled: server.disabled,
                    status,
                })
        })
        .collect();
    installed.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(installed)
}

pub fn install_global_lsp(
    request: IntegrationInstallRequest,
) -> Result<IntegrationInstallResponse> {
    validate_source_image_identity(&request)?;
    if request.kind != IntegrationKind::Lsp {
        bail!("only LSP integrations can be installed globally");
    }
    let name = request
        .name
        .clone()
        .unwrap_or_else(|| name_from_image(&request.container_image));
    if name.trim().is_empty() || name.contains(['\n', '\r']) {
        bail!("integration name cannot be empty or contain newlines");
    }
    if is_marketplace_container_image(&request.container_image) {
        if !request.container_image.contains("@sha256:") {
            bail!("marketplace LSP integrations must be installed by immutable OCI digest");
        }
        let verified = fetch_config_schema(&request.container_image)?;
        let verified_manifest = verified
            .lsp_manifest
            .context("marketplace LSP image does not publish a typed package manifest")?;
        if request.lsp_manifest.as_ref() != Some(&verified_manifest) {
            bail!("marketplace LSP manifest does not match the pinned OCI image");
        }
    }
    let server_config = lsp_server_config_from_install(&request)?;
    if is_marketplace_container_image(&request.container_image) {
        let runtime = match server_config.container_runtime.as_deref() {
            Some(binary) => Some(crate::container::runtime_for_binary(binary)?),
            None => crate::container::detect_runtime(),
        };
        if let Some(runtime) = runtime {
            let runtime_info = runtime.info()?;
            let _image_operation = crate::container::acquire_image_operation_lock(
                &runtime_info.binary,
                &request.container_image,
            )?;
            if !runtime.image_exists(&request.container_image)? {
                runtime.pull(&request.container_image).with_context(|| {
                    format!(
                        "failed to install pinned LSP image {}",
                        request.container_image
                    )
                })?;
            }
        }
    }
    let status = lsp_installation_status(&server_config);
    let verified_manifest_digest = server_config.verified_manifest_digest.clone();
    UserConfig::mutate(|user_config| {
        if request.no_overwrite && user_config.lsp.servers.contains_key(&name) {
            bail!("LSP integration '{name}' is already installed");
        }
        user_config.lsp.servers.insert(name.clone(), server_config);
        Ok(())
    })?;
    Ok(IntegrationInstallResponse {
        installed: InstalledIntegration {
            name,
            kind: IntegrationKind::Lsp,
            container_image: request.container_image,
            source_container_image: request.source_container_image,
            verified_manifest_digest,
            env: request.env,
            disabled: false,
            status,
        },
        config_path: config::config_path()?.display().to_string(),
    })
}

fn requested_runtime(runtime: Option<&str>) -> Result<Option<String>> {
    runtime
        .map(str::trim)
        .filter(|runtime| !runtime.is_empty())
        .map(|runtime| crate::container::resolve_runtime_binary(Some(runtime)).map(Some))
        .transpose()
        .map(Option::flatten)
}

fn lsp_server_config_from_install(request: &IntegrationInstallRequest) -> Result<LspServerConfig> {
    let manifest_digest = request
        .container_image
        .rsplit_once("@sha256:")
        .map(|(_, digest)| format!("sha256:{digest}"));
    let mut config = LspServerConfig {
        container_image: Some(request.container_image.clone()),
        source_container_image: request.source_container_image.clone(),
        verified_manifest_digest: manifest_digest,
        container_runtime: requested_runtime(request.runtime.as_deref())?,
        env: request.env.clone(),
        ..Default::default()
    };
    if let Some(manifest) = &request.lsp_manifest {
        manifest.validate()?;
        config.args = manifest.server.args.clone();
        config.language_ids = manifest.server.language_ids.clone();
        config.initialization_options = manifest.server.initialization_options.clone();
        config.workspace_access = manifest.server.workspace_access;
        config.network_access = manifest.server.network_access;
        config.cache_ids = manifest.server.cache_ids.clone();
    }
    Ok(config)
}

pub fn remove_global_lsp(name: &str) -> Result<IntegrationRemoveResponse> {
    if name.trim().is_empty() || name.contains(['\n', '\r']) {
        bail!("integration name cannot be empty or contain newlines");
    }
    let path = config::config_path()?;
    remove_global_lsp_from_path(&path, name)
}

fn remove_global_lsp_from_path(path: &Path, name: &str) -> Result<IntegrationRemoveResponse> {
    let server = UserConfig::mutate_path(path, |user_config| {
        let server = user_config
            .lsp
            .servers
            .get(name)
            .with_context(|| format!("LSP integration '{name}' is not installed"))?;
        if server.container_image.is_none() {
            bail!("configured LSP server is not a container integration");
        }
        Ok(user_config
            .lsp
            .servers
            .remove(name)
            .expect("server was checked above"))
    })?;
    let status = lsp_installation_status(&server);
    let source_container_image = server.source_container_image.clone();
    let verified_manifest_digest = server.verified_manifest_digest.clone();
    let container_image = server
        .container_image
        .context("configured LSP server is not a container integration")?;
    Ok(IntegrationRemoveResponse {
        removed: InstalledIntegration {
            name: name.to_string(),
            kind: IntegrationKind::Lsp,
            container_image,
            source_container_image,
            verified_manifest_digest,
            env: server.env,
            disabled: server.disabled,
            status,
        },
        config_path: path.display().to_string(),
    })
}

fn name_from_image(image: &str) -> String {
    let leaf = image.rsplit('/').next().unwrap_or(image);
    leaf.split('@')
        .next()
        .unwrap_or(leaf)
        .split(':')
        .next()
        .unwrap_or(leaf)
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
        assert!(is_marketplace_container_image(
            "ghcr.io/crunchy-pb/lsp-rust-analyzer:latest"
        ));
        assert!(is_marketplace_container_image(
            "GHCR.IO/crunchy-pb/lsp-rust-analyzer:latest"
        ));
        assert!(is_marketplace_container_image(
            "ghcr.io/CRUNCHY-PB/lsp-rust-analyzer:latest"
        ));
        assert!(is_marketplace_container_image(
            "ghcr.io./crunchy-pb/lsp-rust-analyzer:latest"
        ));
        assert!(!is_marketplace_container_image(
            "ghcr.io/example/lsp-rust-analyzer:latest"
        ));
        assert!(!is_marketplace_container_image(
            "ghcr.io/crunchy-pb.evil/lsp-rust-analyzer:latest"
        ));
    }

    #[test]
    fn registry_image_parse_splits_registry_repository_and_tag() {
        let image = RegistryImage::parse("GHCR.IO./crunchy-pb/mcp-sentry:latest").unwrap();
        assert_eq!(image.registry, "ghcr.io");
        assert_eq!(image.repository, "crunchy-pb/mcp-sentry");
        assert_eq!(image.reference, "latest");
        assert_eq!(
            image.manifest_url(),
            "https://ghcr.io/v2/crunchy-pb/mcp-sentry/manifests/latest"
        );
    }

    #[test]
    fn digest_addressed_registry_documents_must_match_the_parent_digest() {
        let bytes = br#"{"schemaVersion":2}"#;
        let digest = format!("sha256:{}", crate::environment_lock::sha256(bytes));
        let document = RegistryDocument {
            value: serde_json::from_slice(bytes).unwrap(),
            digest: digest.clone(),
        };
        assert!(verify_registry_document_digest(document, &digest, "image config").is_ok());

        let mismatch = RegistryDocument {
            value: serde_json::json!({"schemaVersion": 2}),
            digest,
        };
        assert!(
            verify_registry_document_digest(
                mismatch,
                &format!("sha256:{}", "0".repeat(64)),
                "image config"
            )
            .is_err()
        );
    }

    #[test]
    fn digest_addressed_registry_documents_reject_malformed_parent_digests() {
        let document = RegistryDocument {
            value: serde_json::json!({}),
            digest: format!("sha256:{}", "0".repeat(64)),
        };
        assert!(verify_registry_document_digest(document, "sha256:abc", "image config").is_err());
    }

    #[test]
    fn registry_image_parse_defaults_missing_tag_to_latest() {
        let image = RegistryImage::parse("ghcr.io/crunchy-pb/mcp-sentry").unwrap();
        assert_eq!(image.reference, "latest");
    }

    #[test]
    fn registry_image_preserves_and_renders_immutable_digest_references() {
        let image =
            RegistryImage::parse("ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:0123456789abcdef")
                .unwrap();
        assert_eq!(image.reference, "sha256:0123456789abcdef");
        assert_eq!(
            image.pinned_reference("sha256:fedcba9876543210"),
            "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:fedcba9876543210"
        );
        assert_eq!(
            name_from_image("ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:abc"),
            "lsp-rust-analyzer"
        );
    }

    #[test]
    fn registry_transport_rejects_local_insecure_and_credentialed_urls() {
        for value in [
            "https://127.0.0.1/v2/image/manifests/latest",
            "https://registry.local/v2/image/manifests/latest",
            "http://example.com/v2/image/manifests/latest",
            "https://token@example.com/v2/image/manifests/latest",
        ] {
            assert!(parse_registry_url(value).is_err(), "accepted {value}");
        }
        parse_registry_url("https://ghcr.io/v2/crunchy-pb/example/manifests/latest").unwrap();
    }

    #[test]
    fn find_annotation_reads_nested_manifest_annotations() {
        let manifest = serde_json::json!({
            "manifests": [{
                "annotations": {
                    CONFIG_SCHEMA_ANNOTATION: r#"{"type":"object"}"#
                }
            }]
        });

        assert_eq!(
            find_annotation(&manifest, CONFIG_SCHEMA_ANNOTATION),
            Some(r#"{"type":"object"}"#)
        );
    }

    #[test]
    fn find_annotation_reads_image_config_labels() {
        let config = serde_json::json!({
            "config": {
                "Labels": {
                    CONFIG_SCHEMA_ANNOTATION: r#"{"type":"object","required":["token"]}"#
                }
            }
        });

        assert_eq!(
            find_annotation(&config, CONFIG_SCHEMA_ANNOTATION),
            Some(r#"{"type":"object","required":["token"]}"#)
        );
    }

    #[test]
    fn select_runnable_manifest_digest_skips_attestation_manifests() {
        let manifest = serde_json::json!({
            "manifests": [
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": "sha256:attestation",
                    "platform": { "os": "unknown", "architecture": "unknown" }
                },
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": "sha256:arm64",
                    "platform": { "os": "linux", "architecture": "arm64" }
                },
                {
                    "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                    "digest": "sha256:amd64",
                    "platform": { "os": "linux", "architecture": "amd64" }
                }
            ]
        });

        let expected = if cfg!(target_arch = "aarch64") {
            "sha256:arm64"
        } else {
            "sha256:amd64"
        };
        assert_eq!(select_runnable_manifest_digest(&manifest), Some(expected));
    }

    #[test]
    fn remove_deletes_project_scoped_mcp_config() {
        let dir = tempfile::tempdir().unwrap();
        install_project(
            dir.path(),
            IntegrationInstallRequest {
                kind: IntegrationKind::Mcp,
                container_image: "ghcr.io/example/sentry-mcp:latest".to_string(),
                source_container_image: None,
                name: None,
                runtime: Some("docker".to_string()),
                env: BTreeMap::new(),
                lsp_manifest: None,
                no_overwrite: false,
            },
        )
        .unwrap();

        let response = remove_project(dir.path(), IntegrationKind::Mcp, "sentry-mcp").unwrap();

        assert_eq!(response.removed.name, "sentry-mcp");
        let config = ProjectMcpConfig::load(dir.path()).unwrap().unwrap();
        assert!(!config.servers.contains_key("sentry-mcp"));
    }

    #[test]
    fn remove_project_preserves_non_container_mcp_servers() {
        let dir = tempfile::tempdir().unwrap();
        ProjectMcpConfig::mutate(dir.path(), |config| {
            config.servers.insert(
                "local".to_string(),
                McpServerConfig {
                    command: Some("local-mcp".to_string()),
                    ..Default::default()
                },
            );
            Ok(())
        })
        .unwrap();

        assert!(remove_project(dir.path(), IntegrationKind::Mcp, "local").is_err());
        let config = ProjectMcpConfig::load(dir.path()).unwrap().unwrap();
        assert!(config.servers.contains_key("local"));
    }

    #[test]
    fn remove_global_lsp_preserves_non_container_servers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        UserConfig::mutate_path(&path, |config| {
            config.lsp.servers.insert(
                "local".to_string(),
                LspServerConfig {
                    command: Some("local-lsp".to_string()),
                    ..Default::default()
                },
            );
            Ok(())
        })
        .unwrap();

        assert!(remove_global_lsp_from_path(&path, "local").is_err());
        let config = UserConfig::load_from_path(&path).unwrap();
        assert!(config.lsp.servers.contains_key("local"));
    }

    #[test]
    fn install_writes_project_scoped_mcp_config() {
        let dir = tempfile::tempdir().unwrap();
        let response = install_project(
            dir.path(),
            IntegrationInstallRequest {
                kind: IntegrationKind::Mcp,
                container_image: "ghcr.io/example/sentry-mcp:latest".to_string(),
                source_container_image: None,
                name: None,
                runtime: Some("docker".to_string()),
                env: BTreeMap::from([("SENTRY_DSN".to_string(), "https://example".to_string())]),
                lsp_manifest: None,
                no_overwrite: false,
            },
        )
        .unwrap();
        assert_eq!(response.installed.name, "sentry-mcp");
        let config = ProjectMcpConfig::load(dir.path()).unwrap().unwrap();
        assert_eq!(
            config.servers["sentry-mcp"].container_image.as_deref(),
            Some("ghcr.io/example/sentry-mcp:latest")
        );
        assert_eq!(
            config.servers["sentry-mcp"]
                .env
                .get("SENTRY_DSN")
                .map(String::as_str),
            Some("https://example")
        );
    }

    #[test]
    fn project_mcp_install_preserves_the_upgrade_source_and_verified_digest() {
        let dir = tempfile::tempdir().unwrap();
        let response = install_project(
            dir.path(),
            IntegrationInstallRequest {
                kind: IntegrationKind::Mcp,
                container_image: "ghcr.io/example/sentry-mcp@sha256:abc".to_string(),
                source_container_image: Some("ghcr.io/example/sentry-mcp:stable".to_string()),
                name: Some("sentry-mcp".to_string()),
                runtime: Some("docker".to_string()),
                env: BTreeMap::new(),
                lsp_manifest: None,
                no_overwrite: false,
            },
        )
        .unwrap();

        assert_eq!(
            response.installed.source_container_image.as_deref(),
            Some("ghcr.io/example/sentry-mcp:stable")
        );
        assert_eq!(
            response.installed.verified_manifest_digest.as_deref(),
            Some("sha256:abc")
        );
        let installed = list_project_installed(dir.path()).unwrap();
        assert_eq!(installed.len(), 1);
        assert_eq!(
            installed[0].source_container_image.as_deref(),
            Some("ghcr.io/example/sentry-mcp:stable")
        );
        assert_eq!(
            installed[0].verified_manifest_digest.as_deref(),
            Some("sha256:abc")
        );
    }

    fn rust_analyzer_manifest() -> LspPackageManifest {
        LspPackageManifest {
            version: 1,
            kind: IntegrationKind::Lsp,
            server: LspPackageServerConfig {
                language_ids: vec!["rust".to_string()],
                initialization_options: Some(serde_json::json!({
                    "checkOnSave": false,
                    "cargo": {"buildScripts": {"enable": false}, "noDeps": true},
                    "procMacro": {"enable": false}
                })),
                ..Default::default()
            },
        }
    }

    #[test]
    fn packaged_lsp_manifest_applies_safe_server_defaults_without_pinning_runtime() {
        let request = IntegrationInstallRequest {
            kind: IntegrationKind::Lsp,
            container_image: "ghcr.io/crunchy-pb/lsp-rust-analyzer:latest".to_string(),
            source_container_image: None,
            name: Some("rust-analyzer".to_string()),
            runtime: None,
            env: BTreeMap::new(),
            lsp_manifest: Some(rust_analyzer_manifest()),
            no_overwrite: false,
        };

        let config = lsp_server_config_from_install(&request).unwrap();

        assert_eq!(config.container_runtime, None);
        assert_eq!(config.language_ids, vec!["rust"]);
        assert_eq!(
            config.workspace_access,
            crate::session_environment::ServiceWorkspaceAccess::ReadOnly
        );
        assert_eq!(
            config.network_access,
            crate::session_environment::ServiceNetworkAccess::None
        );
        assert_eq!(
            config
                .initialization_options
                .as_ref()
                .and_then(|options| options.get("checkOnSave"))
                .and_then(Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn packaged_lsp_manifest_cannot_grant_itself_network_or_write_access() {
        let mut manifest = rust_analyzer_manifest();
        manifest.server.network_access = crate::session_environment::ServiceNetworkAccess::Egress;
        assert!(manifest.validate().is_err());

        manifest.server.network_access = crate::session_environment::ServiceNetworkAccess::None;
        manifest.server.workspace_access =
            crate::session_environment::ServiceWorkspaceAccess::ReadWrite;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn lsp_manifest_annotation_is_parsed_and_validated() {
        let text = serde_json::to_string(&rust_analyzer_manifest()).unwrap();
        let parsed = parse_lsp_package_manifest(&text).unwrap();
        assert_eq!(parsed.kind, IntegrationKind::Lsp);
        assert_eq!(parsed.server.language_ids, vec!["rust"]);
    }
}
