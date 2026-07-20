use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const GOAL_CONFIG_VERSION: u32 = 1;

const HARD_MAX_MILESTONES: usize = 32;
const HARD_MAX_WORKFLOWS: usize = 64;
const HARD_MAX_MODEL_INVOCATIONS: usize = 1_024;
const HARD_MAX_GENERATED_TOKENS: usize = 4_000_000;
const HARD_MAX_WALL_TIME_MINUTES: u64 = 24 * 60;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GoalBudget {
    #[serde(default = "default_max_milestones")]
    pub max_milestones: usize,
    #[serde(default = "default_max_workflows")]
    pub max_workflows: usize,
    #[serde(default = "default_model_invocations")]
    pub total_model_invocations: usize,
    #[serde(default = "default_generated_tokens")]
    pub total_generated_tokens: usize,
    #[serde(default = "default_wall_time_minutes")]
    pub wall_time_minutes: u64,
}

impl Default for GoalBudget {
    fn default() -> Self {
        Self {
            max_milestones: default_max_milestones(),
            max_workflows: default_max_workflows(),
            total_model_invocations: default_model_invocations(),
            total_generated_tokens: default_generated_tokens(),
            wall_time_minutes: default_wall_time_minutes(),
        }
    }
}

impl GoalBudget {
    /// Default per-goal allowance. The project document's `GoalBudget::default()` remains the
    /// larger version-one ceiling so users can explicitly select Extended without editing config.
    pub const fn standard() -> Self {
        Self {
            max_milestones: 5,
            max_workflows: 8,
            total_model_invocations: 80,
            total_generated_tokens: 60_000,
            wall_time_minutes: 90,
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_limit("max_milestones", self.max_milestones, HARD_MAX_MILESTONES)?;
        validate_limit("max_workflows", self.max_workflows, HARD_MAX_WORKFLOWS)?;
        validate_limit(
            "total_model_invocations",
            self.total_model_invocations,
            HARD_MAX_MODEL_INVOCATIONS,
        )?;
        validate_limit(
            "total_generated_tokens",
            self.total_generated_tokens,
            HARD_MAX_GENERATED_TOKENS,
        )?;
        if self.wall_time_minutes == 0 || self.wall_time_minutes > HARD_MAX_WALL_TIME_MINUTES {
            bail!("wall_time_minutes must be between 1 and {HARD_MAX_WALL_TIME_MINUTES}");
        }
        if self.max_workflows < self.max_milestones {
            bail!("max_workflows must be at least max_milestones");
        }
        Ok(())
    }

    pub fn constrained_by(self, ceiling: Self) -> Result<Self> {
        self.validate()?;
        ceiling.validate()?;
        for (label, requested, maximum) in [
            (
                "max_milestones",
                self.max_milestones,
                ceiling.max_milestones,
            ),
            ("max_workflows", self.max_workflows, ceiling.max_workflows),
            (
                "total_model_invocations",
                self.total_model_invocations,
                ceiling.total_model_invocations,
            ),
            (
                "total_generated_tokens",
                self.total_generated_tokens,
                ceiling.total_generated_tokens,
            ),
        ] {
            if requested > maximum {
                bail!("requested {label} exceeds the project ceiling of {maximum}");
            }
        }
        if self.wall_time_minutes > ceiling.wall_time_minutes {
            bail!(
                "requested wall_time_minutes exceeds the project ceiling of {}",
                ceiling.wall_time_minutes
            );
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GoalConfigDocument {
    pub version: u32,
    #[serde(default)]
    pub limits: GoalBudget,
}

impl Default for GoalConfigDocument {
    fn default() -> Self {
        Self {
            version: GOAL_CONFIG_VERSION,
            limits: GoalBudget::default(),
        }
    }
}

impl GoalConfigDocument {
    pub fn load(repo_root: &Path) -> Result<Option<Self>> {
        let path = repo_root.join(".pb").join("goal.toml");
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        toml::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))
            .map(Some)
    }

    pub fn load_or_default(repo_root: &Path) -> Result<CompiledGoalPolicy> {
        Self::load(repo_root)?.unwrap_or_default().compile()
    }

    pub fn save(&self, repo_root: &Path) -> Result<()> {
        let dir = repo_root.join(".pb");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        let path = dir.join("goal.toml");
        let text = toml::to_string_pretty(self).context("failed to serialize goal config")?;
        std::fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn compile(self) -> Result<CompiledGoalPolicy> {
        if self.version != GOAL_CONFIG_VERSION {
            bail!(
                "unsupported goal config version {}; expected {}",
                self.version,
                GOAL_CONFIG_VERSION
            );
        }
        self.limits.validate()?;
        let bytes = serde_json::to_vec(&(self.version, self.limits))
            .context("failed to serialize goal policy")?;
        Ok(CompiledGoalPolicy {
            version: self.version,
            limits: self.limits,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompiledGoalPolicy {
    pub version: u32,
    pub limits: GoalBudget,
    pub sha256: String,
}

impl CompiledGoalPolicy {
    pub fn validate(&self) -> Result<()> {
        let expected = GoalConfigDocument {
            version: self.version,
            limits: self.limits,
        }
        .compile()?;
        if self.sha256 != expected.sha256 {
            bail!("compiled goal policy hash mismatch");
        }
        Ok(())
    }

    pub fn budget(&self, requested: Option<GoalBudget>) -> Result<GoalBudget> {
        requested
            .unwrap_or_else(GoalBudget::standard)
            .constrained_by(self.limits)
    }
}

fn validate_limit(label: &str, value: usize, maximum: usize) -> Result<()> {
    if value == 0 || value > maximum {
        bail!("{label} must be between 1 and {maximum}");
    }
    Ok(())
}

const fn default_max_milestones() -> usize {
    8
}
const fn default_max_workflows() -> usize {
    12
}
const fn default_model_invocations() -> usize {
    120
}
const fn default_generated_tokens() -> usize {
    100_000
}
const fn default_wall_time_minutes() -> u64 {
    120
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_hashes_and_rejects_expansion() {
        let policy = GoalConfigDocument::default().compile().unwrap();
        assert_eq!(policy.sha256.len(), 64);
        let mut expanded = policy.limits;
        expanded.max_milestones += 1;
        assert!(policy.budget(Some(expanded)).is_err());
    }

    #[test]
    fn repository_config_cannot_encode_activation_authority() {
        let input = "version = 1\nauto_start = true\n";
        assert!(toml::from_str::<GoalConfigDocument>(input).is_err());
    }
}
