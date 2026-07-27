use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

use crate::{CollarError, CollarResult, mutation::WorkspaceSnapshot, protocol::ToolDialect};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolConstraintMode {
    #[default]
    Auto,
    ToolsAllowed,
    ToolRequired,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExposedTool {
    pub name: String,
    pub input_schema: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationPolicy {
    pub allow_write_file: bool,
    pub allow_replace_file: bool,
    pub allow_apply_patch: bool,
    pub max_mutation_calls_per_batch: usize,
}

impl Default for MutationPolicy {
    fn default() -> Self {
        Self {
            allow_write_file: false,
            allow_replace_file: false,
            allow_apply_patch: false,
            max_mutation_calls_per_batch: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollarLimits {
    pub max_argument_bytes: usize,
    pub max_snapshot_bytes: usize,
    pub max_files: usize,
    pub max_patch_hunks: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollarManifest {
    pub contract_version: u32,
    pub dialect: ToolDialect,
    pub mode: ToolConstraintMode,
    pub tools: Vec<ExposedTool>,
    pub terminal_tools: Vec<String>,
    pub mutation_policy: MutationPolicy,
    pub workspace: WorkspaceSnapshot,
    pub limits: CollarLimits,
}

impl CollarManifest {
    pub fn validate(&self) -> CollarResult<()> {
        if self.contract_version != 1 {
            return Err(CollarError::InvalidManifest(
                "contract version must be 1".to_string(),
            ));
        }
        if self.tools.is_empty() && self.mode != ToolConstraintMode::Auto {
            return Err(CollarError::InvalidManifest(
                "an explicit tool constraint mode requires exposed tools".to_string(),
            ));
        }
        if self.mutation_policy.max_mutation_calls_per_batch == 0 {
            return Err(CollarError::InvalidManifest(
                "mutation batch limit must be non-zero".to_string(),
            ));
        }
        if self.limits.max_argument_bytes == 0
            || self.limits.max_snapshot_bytes == 0
            || self.limits.max_files == 0
            || self.limits.max_patch_hunks == 0
        {
            return Err(CollarError::InvalidManifest(
                "every collar limit must be non-zero".to_string(),
            ));
        }
        if self.workspace.total_bytes() > self.limits.max_snapshot_bytes {
            return Err(CollarError::InvalidManifest(format!(
                "workspace snapshot exceeds the {}-byte limit",
                self.limits.max_snapshot_bytes
            )));
        }
        let mut names = BTreeSet::new();
        for tool in &self.tools {
            if tool.name.trim().is_empty() {
                return Err(CollarError::InvalidManifest(
                    "exposed tool names must be non-empty".to_string(),
                ));
            }
            if !names.insert(tool.name.as_str()) {
                return Err(CollarError::InvalidManifest(format!(
                    "exposed tool {:?} is duplicated",
                    tool.name
                )));
            }
        }
        let mut terminals = BTreeSet::new();
        for terminal in &self.terminal_tools {
            if !self.tools.iter().any(|tool| tool.name == *terminal) {
                return Err(CollarError::InvalidManifest(format!(
                    "terminal tool {terminal:?} is not exposed"
                )));
            }
            if !terminals.insert(terminal.as_str()) {
                return Err(CollarError::InvalidManifest(format!(
                    "terminal tool {terminal:?} is duplicated"
                )));
            }
        }
        for (allowed, names, label) in [
            (
                self.mutation_policy.allow_write_file,
                &["write_file"][..],
                "write_file",
            ),
            (
                self.mutation_policy.allow_replace_file,
                &["replace_file", "edit_file"][..],
                "replace_file/edit_file",
            ),
            (
                self.mutation_policy.allow_apply_patch,
                &["apply_patch"][..],
                "apply_patch",
            ),
        ] {
            if allowed
                && !names
                    .iter()
                    .any(|name| self.tools.iter().any(|tool| tool.name == *name))
            {
                return Err(CollarError::InvalidManifest(format!(
                    "mutation policy enables {label} without exposing that tool"
                )));
            }
        }
        Ok(())
    }
}
