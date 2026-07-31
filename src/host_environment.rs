//! Audited access to the parent process environment.
//!
//! User-visible pb behavior must come from typed configuration or CLI arguments. This module is
//! limited to operating-system conventions and secret names explicitly declared in configuration.

use anyhow::{Context, Result};
use std::ffi::OsStr;
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

/// Resolve an executable from the operating-system `PATH` convention.
pub fn executable_in_path(name: &OsStr) -> Option<PathBuf> {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .map(|directory| directory.join(name))
        .find(|candidate| is_executable_file(candidate))
}

#[cfg(unix)]
fn is_executable_file(path: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt as _;

    path.metadata()
        .map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable_file(path: &std::path::Path) -> bool {
    path.is_file()
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
