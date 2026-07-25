//! User-level TOML configuration for pb.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::agent_core::AgentProfile;
use crate::lsp::LspConfig;
use crate::mcp::McpConfig;
use crate::{
    DEFAULT_AGENT_MAX_STEPS, DEFAULT_AGENT_MAX_TOKENS, DEFAULT_MODEL, daemon_client,
    default_gpu_layers,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct UserConfig {
    pub web: WebConfig,
    pub model: ModelConfig,
    /// Read-only upgrade compatibility. Retired controller-action settings are ignored and omitted
    /// the next time the configuration is saved.
    #[serde(rename = "agent", skip_serializing)]
    legacy_agent: LegacyAgentConfig,
    pub mcp: McpConfig,
    pub lsp: LspConfig,
    pub memory: MemoryConfig,
    pub storage: StorageConfig,
    pub inference: InferenceConfig,
    pub flashmoe: FlashMoeConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
struct LegacyAgentConfig {
    #[serde(rename = "action_elision")]
    _action_elision: Option<String>,
    #[serde(rename = "controller_delete_elision")]
    _controller_delete_elision: Option<bool>,
}

pub const DEFAULT_SESSION_CACHE_MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;
pub const DEFAULT_FLASHMOE_MEMORY_SESSIONS: usize = 2;
pub const DEFAULT_FLASHMOE_MEMORY_PROMPT_ROOT_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const DEFAULT_FLASHMOE_RESIDENT_MODELS: usize = 2;
pub const DEFAULT_FLASHMOE_IDLE_SECONDS: u64 = 15 * 60;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct StorageConfig {
    /// Root for pb-owned workspaces, leases, and managed cache records.
    pub state_dir: Option<PathBuf>,
    /// Root for prompt-derived inference caches.
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct InferenceConfig {
    pub llamacpp_session_cache_enabled: Option<bool>,
    pub llamacpp_session_cache_max_bytes: Option<u64>,
    pub flashmoe_session_cache_enabled: Option<bool>,
    pub flashmoe_session_cache_max_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct FlashMoeConfig {
    pub memory_sessions: Option<usize>,
    pub memory_prompt_root_max_bytes: Option<u64>,
    pub resident_models: Option<usize>,
    pub idle_seconds: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSessionCacheConfig {
    pub enabled: bool,
    pub root: Option<PathBuf>,
    pub max_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedFlashMoeConfig {
    pub session_cache: ResolvedSessionCacheConfig,
    pub memory_sessions: usize,
    pub memory_prompt_root_max_bytes: u64,
    pub resident_models: usize,
    pub idle_seconds: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    /// Optional separate personal memory repository for cross-project preferences.
    pub personal_repo: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct WebConfig {
    /// Address used by `pb serve`.
    pub listen: Option<String>,
    /// Port used by `pb serve`.
    pub port: Option<u16>,
    /// Unix socket path used by `pb serve` for local daemon clients.
    pub socket_path: Option<PathBuf>,
    /// Prevent idle system sleep while the work queue is actively processing.
    pub prevent_sleep_while_working: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub model: Option<String>,
    pub model_dir: Option<PathBuf>,
    pub workdir: Option<PathBuf>,
    pub max_steps: Option<usize>,
    pub max_tokens: Option<i32>,
    pub ctx_size: Option<u32>,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub gpu_layers: Option<u32>,
    pub temperature: Option<f32>,
    pub profile: Option<AgentProfile>,
    pub top_k: Option<i32>,
    pub seed: Option<u32>,
}

impl UserConfig {
    pub fn load() -> Result<Self> {
        let path = config_path()?;
        Self::load_from_path(&path)
    }

    pub fn load_from_path(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Self =
            toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        let path = config_path()?;
        self.save_to_path(&path)
    }

    pub fn save_to_path(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let text = toml::to_string_pretty(self).context("failed to serialize user config")?;
        crate::atomic_file::write(path, text.as_bytes())
    }

    pub fn mutate<R>(mutation: impl FnOnce(&mut Self) -> Result<R>) -> Result<R> {
        let path = config_path()?;
        Self::mutate_path(&path, mutation)
    }

    pub(crate) fn mutate_path<R>(
        path: &Path,
        mutation: impl FnOnce(&mut Self) -> Result<R>,
    ) -> Result<R> {
        let _lock = crate::state_lock::StateFileLock::acquire(
            config_lock_path(path),
            std::time::Duration::from_secs(10),
        )?;
        let mut config = Self::load_from_path(path)?;
        let result = mutation(&mut config)?;
        config.save_to_path(path)?;
        Ok(result)
    }

    pub fn get(&self, key: &str) -> Result<Option<String>> {
        Ok(match key {
            "web.listen" => self.web.listen.clone(),
            "web.port" => self.web.port.map(|value| value.to_string()),
            "web.socket_path" => self.web.socket_path.as_ref().map(|path| display_path(path)),
            "web.prevent_sleep_while_working" => self
                .web
                .prevent_sleep_while_working
                .map(|value| value.to_string()),
            "model.model" => self.model.model.clone(),
            "model.model_dir" => self.model.model_dir.as_ref().map(|path| display_path(path)),
            "model.workdir" => self.model.workdir.as_ref().map(|path| display_path(path)),
            "model.max_steps" => self.model.max_steps.map(|value| value.to_string()),
            "model.max_tokens" => self.model.max_tokens.map(|value| value.to_string()),
            "model.ctx_size" => self.model.ctx_size.map(|value| value.to_string()),
            "model.threads" => self.model.threads.map(|value| value.to_string()),
            "model.threads_batch" => self.model.threads_batch.map(|value| value.to_string()),
            "model.gpu_layers" => self.model.gpu_layers.map(|value| value.to_string()),
            "model.temperature" => self.model.temperature.map(|value| value.to_string()),
            "model.profile" => self.model.profile.map(|value| value.to_string()),
            "model.top_k" => self.model.top_k.map(|value| value.to_string()),
            "model.seed" => self.model.seed.map(|value| value.to_string()),
            "memory.personal_repo" => self
                .memory
                .personal_repo
                .as_ref()
                .map(|path| display_path(path)),
            "storage.state_dir" => self
                .storage
                .state_dir
                .as_ref()
                .map(|path| display_path(path)),
            "storage.cache_dir" => self
                .storage
                .cache_dir
                .as_ref()
                .map(|path| display_path(path)),
            "inference.llamacpp_session_cache_enabled" => self
                .inference
                .llamacpp_session_cache_enabled
                .map(|value| value.to_string()),
            "inference.llamacpp_session_cache_max_bytes" => self
                .inference
                .llamacpp_session_cache_max_bytes
                .map(|value| value.to_string()),
            "inference.flashmoe_session_cache_enabled" => self
                .inference
                .flashmoe_session_cache_enabled
                .map(|value| value.to_string()),
            "inference.flashmoe_session_cache_max_bytes" => self
                .inference
                .flashmoe_session_cache_max_bytes
                .map(|value| value.to_string()),
            "flashmoe.memory_sessions" => {
                self.flashmoe.memory_sessions.map(|value| value.to_string())
            }
            "flashmoe.memory_prompt_root_max_bytes" => self
                .flashmoe
                .memory_prompt_root_max_bytes
                .map(|value| value.to_string()),
            "flashmoe.resident_models" => {
                self.flashmoe.resident_models.map(|value| value.to_string())
            }
            "flashmoe.idle_seconds" => self.flashmoe.idle_seconds.map(|value| value.to_string()),
            _ => bail_unknown_key(key)?,
        })
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "web.listen" => self.web.listen = Some(value.to_string()),
            "web.port" => self.web.port = Some(parse_value(key, value)?),
            "web.socket_path" => self.web.socket_path = Some(PathBuf::from(value)),
            "web.prevent_sleep_while_working" => {
                self.web.prevent_sleep_while_working = Some(parse_value(key, value)?)
            }
            "model.model" => self.model.model = Some(value.to_string()),
            "model.model_dir" => self.model.model_dir = Some(PathBuf::from(value)),
            "model.workdir" => self.model.workdir = Some(PathBuf::from(value)),
            "model.max_steps" => self.model.max_steps = Some(parse_value(key, value)?),
            "model.max_tokens" => self.model.max_tokens = Some(parse_value(key, value)?),
            "model.ctx_size" => self.model.ctx_size = Some(parse_value(key, value)?),
            "model.threads" => self.model.threads = Some(parse_value(key, value)?),
            "model.threads_batch" => self.model.threads_batch = Some(parse_value(key, value)?),
            "model.gpu_layers" => self.model.gpu_layers = Some(parse_value(key, value)?),
            "model.temperature" => self.model.temperature = Some(parse_value(key, value)?),
            "model.profile" => {
                self.model.profile = Some(
                    toml::from_str(&format!("profile = {value:?}"))
                        .map(|wrapper: ProfileWrapper| wrapper.profile)
                        .with_context(|| format!("invalid value for {key}: {value}"))?,
                )
            }
            "model.top_k" => self.model.top_k = Some(parse_value(key, value)?),
            "model.seed" => self.model.seed = Some(parse_value(key, value)?),
            "memory.personal_repo" => self.memory.personal_repo = Some(PathBuf::from(value)),
            "storage.state_dir" => self.storage.state_dir = Some(parse_absolute_path(key, value)?),
            "storage.cache_dir" => self.storage.cache_dir = Some(parse_absolute_path(key, value)?),
            "inference.llamacpp_session_cache_enabled" => {
                self.inference.llamacpp_session_cache_enabled = Some(parse_value(key, value)?)
            }
            "inference.llamacpp_session_cache_max_bytes" => {
                self.inference.llamacpp_session_cache_max_bytes = Some(parse_positive(key, value)?)
            }
            "inference.flashmoe_session_cache_enabled" => {
                self.inference.flashmoe_session_cache_enabled = Some(parse_value(key, value)?)
            }
            "inference.flashmoe_session_cache_max_bytes" => {
                self.inference.flashmoe_session_cache_max_bytes = Some(parse_positive(key, value)?)
            }
            "flashmoe.memory_sessions" => {
                self.flashmoe.memory_sessions = Some(parse_positive(key, value)?)
            }
            "flashmoe.memory_prompt_root_max_bytes" => {
                self.flashmoe.memory_prompt_root_max_bytes = Some(parse_positive(key, value)?)
            }
            "flashmoe.resident_models" => {
                self.flashmoe.resident_models = Some(parse_positive(key, value)?)
            }
            "flashmoe.idle_seconds" => {
                self.flashmoe.idle_seconds = Some(parse_positive(key, value)?)
            }
            _ => return bail_unknown_key(key),
        }
        self.validate()?;
        Ok(())
    }

    pub fn to_pretty_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).context("failed to serialize user config")
    }

    pub fn effective_web_listen(&self) -> String {
        self.web
            .listen
            .clone()
            .unwrap_or_else(|| "127.0.0.1".to_string())
    }

    pub fn effective_web_port(&self) -> u16 {
        self.web.port.unwrap_or(8311)
    }

    pub fn effective_socket_path(&self) -> PathBuf {
        self.web
            .socket_path
            .clone()
            .unwrap_or_else(daemon_client::default_socket_path)
    }

    pub fn effective_prevent_sleep_while_working(&self) -> bool {
        self.web
            .prevent_sleep_while_working
            .unwrap_or(cfg!(target_os = "macos"))
    }

    pub fn effective_model(&self) -> String {
        self.model
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_MODEL.to_string())
    }

    pub fn effective_model_dir(&self) -> Option<PathBuf> {
        self.model.model_dir.clone()
    }

    pub fn effective_workdir(&self) -> Option<PathBuf> {
        self.model.workdir.clone()
    }

    pub fn effective_max_steps(&self) -> usize {
        self.model.max_steps.unwrap_or(DEFAULT_AGENT_MAX_STEPS)
    }

    pub fn effective_max_tokens(&self) -> i32 {
        self.model.max_tokens.unwrap_or(DEFAULT_AGENT_MAX_TOKENS)
    }

    pub fn effective_ctx_size(&self) -> u32 {
        self.model.ctx_size.unwrap_or(131_072)
    }

    pub fn effective_threads(&self) -> Option<i32> {
        self.model.threads
    }

    pub fn effective_threads_batch(&self) -> Option<i32> {
        self.model.threads_batch
    }

    pub fn effective_gpu_layers(&self) -> u32 {
        self.model.gpu_layers.unwrap_or_else(default_gpu_layers)
    }

    pub fn effective_temperature(&self) -> f32 {
        self.model.temperature.unwrap_or(0.2)
    }

    pub fn effective_profile(&self) -> AgentProfile {
        self.model.profile.unwrap_or_default()
    }

    pub fn effective_top_k(&self) -> i32 {
        self.model.top_k.unwrap_or(40)
    }

    pub fn effective_seed(&self) -> u32 {
        self.model.seed.unwrap_or(1337)
    }

    pub fn effective_personal_memory_repo(&self) -> Option<PathBuf> {
        self.memory.personal_repo.clone()
    }

    pub fn effective_state_dir(&self) -> Result<PathBuf> {
        if let Some(path) = &self.storage.state_dir {
            return Ok(path.clone());
        }
        let base = dirs::data_local_dir().context("cannot determine local pb data directory")?;
        Ok(base.join("pb").join("state"))
    }

    pub fn effective_cache_dir(&self) -> Option<PathBuf> {
        self.storage
            .cache_dir
            .clone()
            .or_else(|| dirs::cache_dir().map(|root| root.join("pb")))
    }

    pub fn effective_llamacpp_session_cache(&self) -> ResolvedSessionCacheConfig {
        ResolvedSessionCacheConfig {
            enabled: self
                .inference
                .llamacpp_session_cache_enabled
                .unwrap_or(true),
            root: self.effective_cache_dir(),
            max_bytes: self
                .inference
                .llamacpp_session_cache_max_bytes
                .unwrap_or(DEFAULT_SESSION_CACHE_MAX_BYTES),
        }
    }

    pub fn effective_flashmoe(&self) -> ResolvedFlashMoeConfig {
        ResolvedFlashMoeConfig {
            session_cache: ResolvedSessionCacheConfig {
                enabled: self
                    .inference
                    .flashmoe_session_cache_enabled
                    .unwrap_or(true),
                root: self.effective_cache_dir(),
                max_bytes: self
                    .inference
                    .flashmoe_session_cache_max_bytes
                    .unwrap_or(DEFAULT_SESSION_CACHE_MAX_BYTES),
            },
            memory_sessions: self
                .flashmoe
                .memory_sessions
                .unwrap_or(DEFAULT_FLASHMOE_MEMORY_SESSIONS),
            memory_prompt_root_max_bytes: self
                .flashmoe
                .memory_prompt_root_max_bytes
                .unwrap_or(DEFAULT_FLASHMOE_MEMORY_PROMPT_ROOT_MAX_BYTES),
            resident_models: self
                .flashmoe
                .resident_models
                .unwrap_or(DEFAULT_FLASHMOE_RESIDENT_MODELS),
            idle_seconds: self
                .flashmoe
                .idle_seconds
                .unwrap_or(DEFAULT_FLASHMOE_IDLE_SECONDS),
        }
    }

    fn validate(&self) -> Result<()> {
        validate_optional_absolute_path("storage.state_dir", self.storage.state_dir.as_deref())?;
        validate_optional_absolute_path("storage.cache_dir", self.storage.cache_dir.as_deref())?;
        validate_optional_positive(
            "inference.llamacpp_session_cache_max_bytes",
            self.inference.llamacpp_session_cache_max_bytes,
        )?;
        validate_optional_positive(
            "inference.flashmoe_session_cache_max_bytes",
            self.inference.flashmoe_session_cache_max_bytes,
        )?;
        validate_optional_positive("flashmoe.memory_sessions", self.flashmoe.memory_sessions)?;
        validate_optional_positive(
            "flashmoe.memory_prompt_root_max_bytes",
            self.flashmoe.memory_prompt_root_max_bytes,
        )?;
        validate_optional_positive("flashmoe.resident_models", self.flashmoe.resident_models)?;
        validate_optional_positive("flashmoe.idle_seconds", self.flashmoe.idle_seconds)?;
        Ok(())
    }
}

fn config_lock_path(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(".lock");
    PathBuf::from(value)
}

pub fn config_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir().context("cannot determine config directory")?;
    Ok(config_dir.join("pb").join("config.toml"))
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn parse_value<T>(key: &str, value: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid value for {key}: {value} ({err})"))
}

fn parse_positive<T>(key: &str, value: &str) -> Result<T>
where
    T: std::str::FromStr + PartialOrd + From<u8> + Copy,
    T::Err: std::fmt::Display,
{
    let parsed = parse_value::<T>(key, value)?;
    if parsed <= T::from(0) {
        bail!("invalid value for {key}: expected a positive value");
    }
    Ok(parsed)
}

fn parse_absolute_path(key: &str, value: &str) -> Result<PathBuf> {
    let path = PathBuf::from(value);
    validate_optional_absolute_path(key, Some(&path))?;
    Ok(path)
}

fn validate_optional_absolute_path(key: &str, path: Option<&Path>) -> Result<()> {
    if let Some(path) = path
        && (path.as_os_str().is_empty() || !path.is_absolute())
    {
        bail!("invalid value for {key}: expected a non-empty absolute path");
    }
    Ok(())
}

fn validate_optional_positive<T>(key: &str, value: Option<T>) -> Result<()>
where
    T: PartialOrd + From<u8> + Copy,
{
    if value.is_some_and(|value| value <= T::from(0)) {
        bail!("invalid value for {key}: expected a positive value");
    }
    Ok(())
}

fn bail_unknown_key<T>(key: &str) -> Result<T> {
    bail!(
        "unknown config key '{key}'; supported keys: web.listen, web.port, web.socket_path, web.prevent_sleep_while_working, model.model, model.model_dir, model.workdir, model.max_steps, model.max_tokens, model.ctx_size, model.threads, model.threads_batch, model.gpu_layers, model.temperature, model.profile, model.top_k, model.seed, memory.personal_repo, storage.state_dir, storage.cache_dir, inference.llamacpp_session_cache_enabled, inference.llamacpp_session_cache_max_bytes, inference.flashmoe_session_cache_enabled, inference.flashmoe_session_cache_max_bytes, flashmoe.memory_sessions, flashmoe.memory_prompt_root_max_bytes, flashmoe.resident_models, flashmoe.idle_seconds. MCP servers are configured in TOML as [mcp.servers.<name>] tables with command, url, container_image, source_container_image, verified_manifest_digest, container_runtime, args, env, working_directory, capabilities, and disabled fields. MCP capabilities default-deny workspace and network access and can declare workspace, network, cache_ids, secret_env, and operator-audited read_only_tools. LSP servers are configured in TOML as [lsp.servers.<name>] tables with command, container_image, container_runtime, args, env, working_directory, language_ids, workspace_access, network_access, cache_ids, and disabled fields"
    )
}

#[derive(Deserialize)]
struct ProfileWrapper {
    profile: AgentProfile,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use tempfile::tempdir;

    #[test]
    fn set_get_and_save_user_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut config = UserConfig::default();
        config.set("web.listen", "0.0.0.0").unwrap();
        config.set("web.port", "9999").unwrap();
        config
            .set("web.prevent_sleep_while_working", "false")
            .unwrap();
        config.set("model.temperature", "0.7").unwrap();
        config.set("model.profile", "review").unwrap();
        config.save_to_path(&path).unwrap();

        let loaded = UserConfig::load_from_path(&path).unwrap();
        assert_eq!(
            loaded.get("web.listen").unwrap(),
            Some("0.0.0.0".to_string())
        );
        assert_eq!(loaded.get("web.port").unwrap(), Some("9999".to_string()));
        assert_eq!(
            loaded.get("web.prevent_sleep_while_working").unwrap(),
            Some("false".to_string())
        );
        assert_eq!(loaded.model.temperature, Some(0.7));
        assert_eq!(loaded.model.profile, Some(AgentProfile::Review));
    }

    #[test]
    fn serialized_mutations_preserve_concurrent_config_fields() {
        let dir = tempdir().unwrap();
        let path = Arc::new(dir.path().join("config.toml"));
        UserConfig::default().save_to_path(&path).unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let first = {
            let path = Arc::clone(&path);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                UserConfig::mutate_path(&path, |config| {
                    config.web.port = Some(9001);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    Ok(())
                })
                .unwrap();
            })
        };
        let second = {
            let path = Arc::clone(&path);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                UserConfig::mutate_path(&path, |config| {
                    config.model.max_steps = Some(7);
                    Ok(())
                })
                .unwrap();
            })
        };
        barrier.wait();
        first.join().unwrap();
        second.join().unwrap();

        let config = UserConfig::load_from_path(&path).unwrap();
        assert_eq!(config.web.port, Some(9001));
        assert_eq!(config.model.max_steps, Some(7));
    }

    #[test]
    fn effective_values_use_config_over_defaults() {
        let mut config = UserConfig::default();
        config.set("web.listen", "0.0.0.0").unwrap();
        config.set("web.port", "9999").unwrap();
        config.set("model.max_steps", "3").unwrap();
        config
            .set("web.prevent_sleep_while_working", "false")
            .unwrap();

        assert_eq!(config.effective_web_listen(), "0.0.0.0");
        assert_eq!(config.effective_web_port(), 9999);
        assert_eq!(config.effective_max_steps(), 3);
        assert!(!config.effective_prevent_sleep_while_working());
        assert_eq!(
            UserConfig::default().effective_max_steps(),
            DEFAULT_AGENT_MAX_STEPS
        );
    }

    #[test]
    fn retired_action_elision_settings_load_but_are_no_longer_configurable_or_saved() {
        let config: UserConfig = toml::from_str(
            r#"
[agent]
action_elision = "safe"
controller_delete_elision = true
"#,
        )
        .unwrap();
        assert!(!config.to_pretty_toml().unwrap().contains("[agent]"));
        assert!(config.get("agent.action_elision").is_err());
        let mut config = config;
        assert!(config.set("agent.action_elision", "off").is_err());
    }

    #[test]
    fn runtime_storage_and_inference_settings_round_trip() {
        let dir = tempdir().unwrap();
        let state_dir = dir.path().join("state");
        let cache_dir = dir.path().join("cache");
        let path = dir.path().join("config.toml");
        let mut config = UserConfig::default();
        config
            .set("storage.state_dir", &state_dir.to_string_lossy())
            .unwrap();
        config
            .set("storage.cache_dir", &cache_dir.to_string_lossy())
            .unwrap();
        config
            .set("inference.llamacpp_session_cache_enabled", "false")
            .unwrap();
        config
            .set("inference.llamacpp_session_cache_max_bytes", "1024")
            .unwrap();
        config
            .set("inference.flashmoe_session_cache_enabled", "false")
            .unwrap();
        config
            .set("inference.flashmoe_session_cache_max_bytes", "2048")
            .unwrap();
        config.set("flashmoe.memory_sessions", "3").unwrap();
        config
            .set("flashmoe.memory_prompt_root_max_bytes", "4096")
            .unwrap();
        config.set("flashmoe.resident_models", "4").unwrap();
        config.set("flashmoe.idle_seconds", "60").unwrap();
        config.save_to_path(&path).unwrap();

        let loaded = UserConfig::load_from_path(&path).unwrap();
        assert_eq!(loaded.effective_state_dir().unwrap(), state_dir);
        let llama = loaded.effective_llamacpp_session_cache();
        assert!(!llama.enabled);
        assert_eq!(llama.root, Some(cache_dir.clone()));
        assert_eq!(llama.max_bytes, 1024);
        let flashmoe = loaded.effective_flashmoe();
        assert!(!flashmoe.session_cache.enabled);
        assert_eq!(flashmoe.session_cache.root, Some(cache_dir));
        assert_eq!(flashmoe.session_cache.max_bytes, 2048);
        assert_eq!(flashmoe.memory_sessions, 3);
        assert_eq!(flashmoe.memory_prompt_root_max_bytes, 4096);
        assert_eq!(flashmoe.resident_models, 4);
        assert_eq!(flashmoe.idle_seconds, 60);
    }

    #[test]
    fn runtime_settings_reject_relative_paths_and_zero_limits() {
        let mut config = UserConfig::default();
        assert!(config.set("storage.state_dir", "relative/state").is_err());
        assert!(
            config
                .set("inference.llamacpp_session_cache_max_bytes", "0")
                .is_err()
        );
        assert!(config.set("flashmoe.memory_sessions", "0").is_err());
        assert!(
            config
                .set("flashmoe.memory_prompt_root_max_bytes", "0")
                .is_err()
        );
        assert!(config.set("flashmoe.idle_seconds", "0").is_err());
    }

    #[test]
    fn effective_ctx_size_auto_detect() {
        let config = UserConfig::default();
        // When ctx_size not set, returns default of 131072 (128k)
        assert_eq!(config.effective_ctx_size(), 131_072);
    }

    #[test]
    fn effective_ctx_size_uses_config_when_set() {
        let mut config = UserConfig::default();
        config.set("model.ctx_size", "8192").unwrap();

        // When ctx_size is explicitly set in config, use that value
        assert_eq!(config.effective_ctx_size(), 8192);
    }
}
