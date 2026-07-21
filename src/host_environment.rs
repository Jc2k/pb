//! Audited access to the parent process environment.
//!
//! User-visible pb behavior must come from typed configuration or CLI arguments. This module is
//! limited to operating-system conventions and secret names explicitly declared in configuration.

use anyhow::{Context, Result};
use std::path::PathBuf;

pub fn runtime_dir() -> Option<PathBuf> {
    absolute_directory("XDG_RUNTIME_DIR")
}

pub fn cache_home() -> Option<PathBuf> {
    absolute_directory("XDG_CACHE_HOME")
}

pub fn config_home() -> Option<PathBuf> {
    absolute_directory("XDG_CONFIG_HOME")
}

pub fn state_home() -> Option<PathBuf> {
    absolute_directory("XDG_STATE_HOME")
}

pub fn data_home() -> Option<PathBuf> {
    absolute_directory("XDG_DATA_HOME")
}

fn absolute_directory(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
}

pub fn configured_secret(name: &str) -> Result<String> {
    std::env::var(name)
        .with_context(|| format!("configured secret environment variable {name} is not available"))
}
