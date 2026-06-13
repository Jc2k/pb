//! User-level TOML configuration for pb.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::agent_core::AgentProfile;
use crate::{DEFAULT_AGENT_MAX_STEPS, DEFAULT_AGENT_MAX_TOKENS, DEFAULT_MODEL, default_gpu_layers};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct UserConfig {
    pub web: WebConfig,
    pub model: ModelConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct WebConfig {
    /// Address used by `pb serve` when --host is omitted.
    pub listen: Option<String>,
    /// Port used by `pb serve` when --port is omitted.
    pub port: Option<u16>,
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
        toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn save(&self) -> Result<()> {
        let path = config_path()?;
        self.save_to_path(&path)
    }

    pub fn save_to_path(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let text = toml::to_string_pretty(self).context("failed to serialize user config")?;
        std::fs::write(path, text).with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn get(&self, key: &str) -> Result<Option<String>> {
        Ok(match key {
            "web.listen" => self.web.listen.clone(),
            "web.port" => self.web.port.map(|value| value.to_string()),
            "model.model" => self.model.model.clone(),
            "model.model_dir" => self.model.model_dir.as_ref().map(display_path),
            "model.workdir" => self.model.workdir.as_ref().map(display_path),
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
            _ => bail_unknown_key(key)?,
        })
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "web.listen" => self.web.listen = Some(value.to_string()),
            "web.port" => self.web.port = Some(parse_value(key, value)?),
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
            _ => return bail_unknown_key(key),
        }
        Ok(())
    }

    pub fn to_pretty_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).context("failed to serialize user config")
    }

    pub fn effective_web_listen(&self, cli_value: Option<String>) -> String {
        cli_value
            .or_else(|| self.web.listen.clone())
            .unwrap_or_else(|| "127.0.0.1".to_string())
    }

    pub fn effective_web_port(&self, cli_value: Option<u16>) -> u16 {
        cli_value.or(self.web.port).unwrap_or(8311)
    }

    pub fn effective_model(&self, cli_value: Option<String>) -> String {
        cli_value
            .or_else(|| self.model.model.clone())
            .unwrap_or_else(|| DEFAULT_MODEL.to_string())
    }

    pub fn effective_model_dir(&self, cli_value: Option<PathBuf>) -> Option<PathBuf> {
        cli_value.or_else(|| self.model.model_dir.clone())
    }

    pub fn effective_workdir(&self, cli_value: Option<PathBuf>) -> Option<PathBuf> {
        cli_value.or_else(|| self.model.workdir.clone())
    }

    pub fn effective_max_steps(&self, cli_value: Option<usize>) -> usize {
        cli_value
            .or(self.model.max_steps)
            .unwrap_or(DEFAULT_AGENT_MAX_STEPS)
    }

    pub fn effective_max_tokens(&self, cli_value: Option<i32>) -> i32 {
        cli_value
            .or(self.model.max_tokens)
            .unwrap_or(DEFAULT_AGENT_MAX_TOKENS)
    }

    pub fn effective_ctx_size(&self, cli_value: Option<u32>) -> u32 {
        cli_value.or(self.model.ctx_size).unwrap_or(8192)
    }

    pub fn effective_threads(&self, cli_value: Option<i32>) -> Option<i32> {
        cli_value.or(self.model.threads)
    }

    pub fn effective_threads_batch(&self, cli_value: Option<i32>) -> Option<i32> {
        cli_value.or(self.model.threads_batch)
    }

    pub fn effective_gpu_layers(&self, cli_value: Option<u32>) -> u32 {
        cli_value
            .or(self.model.gpu_layers)
            .unwrap_or_else(default_gpu_layers)
    }

    pub fn effective_temperature(&self, cli_value: Option<f32>) -> f32 {
        cli_value.or(self.model.temperature).unwrap_or(0.2)
    }

    pub fn effective_profile(&self, cli_value: Option<AgentProfile>) -> AgentProfile {
        cli_value.or(self.model.profile).unwrap_or_default()
    }

    pub fn effective_top_k(&self, cli_value: Option<i32>) -> i32 {
        cli_value.or(self.model.top_k).unwrap_or(40)
    }

    pub fn effective_seed(&self, cli_value: Option<u32>) -> u32 {
        cli_value.or(self.model.seed).unwrap_or(1337)
    }
}

pub fn config_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir().context("cannot determine config directory")?;
    Ok(config_dir.join("pb").join("config.toml"))
}

fn display_path(path: &PathBuf) -> String {
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

fn bail_unknown_key<T>(key: &str) -> Result<T> {
    bail!(
        "unknown config key '{key}'; supported keys: web.listen, web.port, model.model, model.model_dir, model.workdir, model.max_steps, model.max_tokens, model.ctx_size, model.threads, model.threads_batch, model.gpu_layers, model.temperature, model.profile, model.top_k, model.seed"
    )
}

#[derive(Deserialize)]
struct ProfileWrapper {
    profile: AgentProfile,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn set_get_and_save_user_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut config = UserConfig::default();
        config.set("web.listen", "0.0.0.0").unwrap();
        config.set("web.port", "9999").unwrap();
        config.set("model.temperature", "0.7").unwrap();
        config.set("model.profile", "review").unwrap();
        config.save_to_path(&path).unwrap();

        let loaded = UserConfig::load_from_path(&path).unwrap();
        assert_eq!(
            loaded.get("web.listen").unwrap(),
            Some("0.0.0.0".to_string())
        );
        assert_eq!(loaded.get("web.port").unwrap(), Some("9999".to_string()));
        assert_eq!(loaded.model.temperature, Some(0.7));
        assert_eq!(loaded.model.profile, Some(AgentProfile::Review));
    }

    #[test]
    fn effective_values_prefer_cli_over_config_over_defaults() {
        let mut config = UserConfig::default();
        config.set("web.listen", "0.0.0.0").unwrap();
        config.set("web.port", "9999").unwrap();
        config.set("model.max_steps", "3").unwrap();

        assert_eq!(config.effective_web_listen(None), "0.0.0.0");
        assert_eq!(
            config.effective_web_listen(Some("localhost".to_string())),
            "localhost"
        );
        assert_eq!(config.effective_web_port(None), 9999);
        assert_eq!(config.effective_web_port(Some(7777)), 7777);
        assert_eq!(config.effective_max_steps(None), 3);
        assert_eq!(
            UserConfig::default().effective_max_steps(None),
            DEFAULT_AGENT_MAX_STEPS
        );
    }
}
