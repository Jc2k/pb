use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::TaskEffort;

pub const TASK_CONFIG_VERSION: u32 = 1;
pub const TASK_PLANNER_QUALIFICATION_VERSION: u32 = 1;

const HARD_MAX_TASKS: usize = 32;
const HARD_MAX_WORKFLOWS: usize = 64;
const HARD_MAX_STAGE_STEPS: usize = 2_048;
const HARD_MAX_MODEL_INVOCATIONS: usize = 1_024;
const HARD_MAX_GENERATED_TOKENS: usize = 4_000_000;
const HARD_MAX_ADVISORY_CALLS: usize = 128;
const HARD_MAX_CYCLES: usize = 64;
const HARD_MAX_WALL_TIME_MINUTES: u64 = 24 * 60;
const HARD_MAX_PLANNING_ATTEMPTS: usize = 3;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskBudget {
    pub max_workflows: usize,
    pub stage_steps: usize,
    pub total_model_invocations: usize,
    pub total_generated_tokens: usize,
    pub advisory_calls: usize,
    pub plan_cycles: usize,
    pub repair_cycles: usize,
    pub wall_time_minutes: u64,
}

impl TaskBudget {
    pub fn validate(&self) -> Result<()> {
        validate_usize("max_workflows", self.max_workflows, HARD_MAX_WORKFLOWS)?;
        validate_usize("stage_steps", self.stage_steps, HARD_MAX_STAGE_STEPS)?;
        validate_usize(
            "total_model_invocations",
            self.total_model_invocations,
            HARD_MAX_MODEL_INVOCATIONS,
        )?;
        validate_usize(
            "total_generated_tokens",
            self.total_generated_tokens,
            HARD_MAX_GENERATED_TOKENS,
        )?;
        validate_usize(
            "advisory_calls",
            self.advisory_calls,
            HARD_MAX_ADVISORY_CALLS,
        )?;
        validate_usize("plan_cycles", self.plan_cycles, HARD_MAX_CYCLES)?;
        validate_usize("repair_cycles", self.repair_cycles, HARD_MAX_CYCLES)?;
        validate_u64(
            "wall_time_minutes",
            self.wall_time_minutes,
            HARD_MAX_WALL_TIME_MINUTES,
        )?;
        Ok(())
    }

    pub fn checked_add(self, other: Self) -> Result<Self> {
        let combined = Self {
            max_workflows: checked_add("max_workflows", self.max_workflows, other.max_workflows)?,
            stage_steps: checked_add("stage_steps", self.stage_steps, other.stage_steps)?,
            total_model_invocations: checked_add(
                "total_model_invocations",
                self.total_model_invocations,
                other.total_model_invocations,
            )?,
            total_generated_tokens: checked_add(
                "total_generated_tokens",
                self.total_generated_tokens,
                other.total_generated_tokens,
            )?,
            advisory_calls: checked_add(
                "advisory_calls",
                self.advisory_calls,
                other.advisory_calls,
            )?,
            plan_cycles: checked_add("plan_cycles", self.plan_cycles, other.plan_cycles)?,
            repair_cycles: checked_add("repair_cycles", self.repair_cycles, other.repair_cycles)?,
            wall_time_minutes: self
                .wall_time_minutes
                .checked_add(other.wall_time_minutes)
                .ok_or_else(|| anyhow::anyhow!("wall_time_minutes overflow"))?,
        };
        combined.validate()?;
        Ok(combined)
    }

    pub fn fits_within(&self, ceiling: &Self) -> bool {
        self.max_workflows <= ceiling.max_workflows
            && self.stage_steps <= ceiling.stage_steps
            && self.total_model_invocations <= ceiling.total_model_invocations
            && self.total_generated_tokens <= ceiling.total_generated_tokens
            && self.advisory_calls <= ceiling.advisory_calls
            && self.plan_cycles <= ceiling.plan_cycles
            && self.repair_cycles <= ceiling.repair_cycles
            && self.wall_time_minutes <= ceiling.wall_time_minutes
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MultiTaskBudget {
    pub max_tasks: usize,
    #[serde(flatten)]
    pub tasks: TaskBudget,
}

impl MultiTaskBudget {
    pub fn validate(&self) -> Result<()> {
        validate_usize("max_tasks", self.max_tasks, HARD_MAX_TASKS)?;
        self.tasks.validate()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskCoordinationBudget {
    pub planning_attempts: usize,
    pub model_invocations: usize,
    pub generated_tokens: usize,
    pub advisory_calls: usize,
    pub wall_time_minutes: u64,
}

impl TaskCoordinationBudget {
    pub fn validate(&self) -> Result<()> {
        validate_usize(
            "coordination.planning_attempts",
            self.planning_attempts,
            HARD_MAX_PLANNING_ATTEMPTS,
        )?;
        validate_usize(
            "coordination.model_invocations",
            self.model_invocations,
            HARD_MAX_MODEL_INVOCATIONS,
        )?;
        validate_usize(
            "coordination.generated_tokens",
            self.generated_tokens,
            HARD_MAX_GENERATED_TOKENS,
        )?;
        validate_usize(
            "coordination.advisory_calls",
            self.advisory_calls,
            HARD_MAX_ADVISORY_CALLS,
        )?;
        validate_u64(
            "coordination.wall_time_minutes",
            self.wall_time_minutes,
            HARD_MAX_WALL_TIME_MINUTES,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskConfigDocument {
    pub version: u32,
    #[serde(default = "default_small_budget")]
    pub small: TaskBudget,
    #[serde(default = "default_medium_budget")]
    pub medium: TaskBudget,
    #[serde(default = "default_large_budget")]
    pub large: TaskBudget,
    #[serde(default = "default_aggregate_budget")]
    pub aggregate: MultiTaskBudget,
    #[serde(default = "default_coordination_budget")]
    pub coordination: TaskCoordinationBudget,
}

impl Default for TaskConfigDocument {
    fn default() -> Self {
        Self {
            version: TASK_CONFIG_VERSION,
            small: default_small_budget(),
            medium: default_medium_budget(),
            large: default_large_budget(),
            aggregate: default_aggregate_budget(),
            coordination: default_coordination_budget(),
        }
    }
}

impl TaskConfigDocument {
    pub fn load(repo_root: &Path) -> Result<Option<Self>> {
        let path = repo_root.join(".pb").join("tasks.toml");
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        toml::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))
            .map(Some)
    }

    pub fn load_or_default(repo_root: &Path) -> Result<CompiledTaskPolicy> {
        Self::load(repo_root)?.unwrap_or_default().compile()
    }

    pub fn save(&self, repo_root: &Path) -> Result<()> {
        let dir = repo_root.join(".pb");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        let path = dir.join("tasks.toml");
        let text = toml::to_string_pretty(self).context("failed to serialize Task config")?;
        std::fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn compile(self) -> Result<CompiledTaskPolicy> {
        if self.version != TASK_CONFIG_VERSION {
            bail!(
                "unsupported Task config version {}; expected {}",
                self.version,
                TASK_CONFIG_VERSION
            );
        }
        for budget in [self.small, self.medium, self.large] {
            budget.validate()?;
            if !budget.fits_within(&self.aggregate.tasks) {
                bail!("Task effort budget exceeds the aggregate Task ceiling");
            }
        }
        self.aggregate.validate()?;
        self.coordination.validate()?;
        let normalized = NormalizedTaskPolicy {
            version: self.version,
            small: self.small,
            medium: self.medium,
            large: self.large,
            aggregate: self.aggregate,
            coordination: self.coordination,
        };
        let bytes = serde_json::to_vec(&normalized)
            .context("failed to serialize normalized Task policy")?;
        Ok(CompiledTaskPolicy {
            version: self.version,
            small: self.small,
            medium: self.medium,
            large: self.large,
            aggregate: self.aggregate,
            coordination: self.coordination,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
struct NormalizedTaskPolicy {
    version: u32,
    small: TaskBudget,
    medium: TaskBudget,
    large: TaskBudget,
    aggregate: MultiTaskBudget,
    coordination: TaskCoordinationBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompiledTaskPolicy {
    pub version: u32,
    pub small: TaskBudget,
    pub medium: TaskBudget,
    pub large: TaskBudget,
    pub aggregate: MultiTaskBudget,
    pub coordination: TaskCoordinationBudget,
    pub sha256: String,
}

impl CompiledTaskPolicy {
    pub fn validate(&self) -> Result<()> {
        let expected = TaskConfigDocument {
            version: self.version,
            small: self.small,
            medium: self.medium,
            large: self.large,
            aggregate: self.aggregate,
            coordination: self.coordination,
        }
        .compile()?;
        if expected.sha256 != self.sha256 {
            bail!("compiled Task policy hash mismatch");
        }
        Ok(())
    }

    pub const fn budget_for(&self, effort: TaskEffort) -> TaskBudget {
        match effort {
            TaskEffort::Small => self.small,
            TaskEffort::Medium => self.medium,
            TaskEffort::Large => self.large,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlannerQualification {
    pub version: u32,
    pub model_sha256: String,
    pub template_sha256: String,
    pub protocol_sha256: String,
    pub evidence_sha256: String,
    pub task_planning: bool,
    pub automatic_goal_selection: bool,
    pub sha256: String,
}

impl TaskPlannerQualification {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_sha256: impl Into<String>,
        template_sha256: impl Into<String>,
        protocol_sha256: impl Into<String>,
        evidence_sha256: impl Into<String>,
        task_planning: bool,
        automatic_goal_selection: bool,
    ) -> Result<Self> {
        let mut qualification = Self {
            version: TASK_PLANNER_QUALIFICATION_VERSION,
            model_sha256: model_sha256.into(),
            template_sha256: template_sha256.into(),
            protocol_sha256: protocol_sha256.into(),
            evidence_sha256: evidence_sha256.into(),
            task_planning,
            automatic_goal_selection,
            sha256: String::new(),
        };
        qualification.validate_fields()?;
        qualification.sha256 = qualification.expected_sha256()?;
        Ok(qualification)
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_fields()?;
        let expected = self.expected_sha256()?;
        if self.sha256 != expected {
            bail!(
                "Task planner qualification digest mismatch: expected {}, got {}",
                expected,
                self.sha256
            );
        }
        Ok(())
    }

    fn validate_fields(&self) -> Result<()> {
        if self.version != TASK_PLANNER_QUALIFICATION_VERSION {
            bail!(
                "unsupported Task planner qualification version {}; expected {}",
                self.version,
                TASK_PLANNER_QUALIFICATION_VERSION
            );
        }
        for (label, digest) in [
            ("model", &self.model_sha256),
            ("template", &self.template_sha256),
            ("protocol", &self.protocol_sha256),
            ("evidence", &self.evidence_sha256),
        ] {
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                bail!("Task planner {label} digest must be a 64-character hexadecimal SHA-256");
            }
        }
        if self.automatic_goal_selection && !self.task_planning {
            bail!("automatic Goal selection requires Task planning qualification");
        }
        Ok(())
    }

    fn expected_sha256(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&(
            self.version,
            &self.model_sha256,
            &self.template_sha256,
            &self.protocol_sha256,
            &self.evidence_sha256,
            self.task_planning,
            self.automatic_goal_selection,
        ))
        .context("failed to serialize Task planner qualification")?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

const fn default_small_budget() -> TaskBudget {
    TaskBudget {
        max_workflows: 1,
        stage_steps: 48,
        total_model_invocations: 40,
        total_generated_tokens: 24_000,
        advisory_calls: 4,
        plan_cycles: 2,
        repair_cycles: 2,
        wall_time_minutes: 30,
    }
}

const fn default_medium_budget() -> TaskBudget {
    TaskBudget {
        max_workflows: 2,
        stage_steps: 96,
        total_model_invocations: 64,
        total_generated_tokens: 40_000,
        advisory_calls: 6,
        plan_cycles: 3,
        repair_cycles: 3,
        wall_time_minutes: 60,
    }
}

const fn default_large_budget() -> TaskBudget {
    TaskBudget {
        max_workflows: 8,
        stage_steps: 192,
        total_model_invocations: 120,
        total_generated_tokens: 100_000,
        advisory_calls: 12,
        plan_cycles: 6,
        repair_cycles: 6,
        wall_time_minutes: 120,
    }
}

const fn default_aggregate_budget() -> MultiTaskBudget {
    MultiTaskBudget {
        max_tasks: 8,
        tasks: TaskBudget {
            max_workflows: 24,
            stage_steps: 768,
            total_model_invocations: 360,
            total_generated_tokens: 300_000,
            advisory_calls: 40,
            plan_cycles: 24,
            repair_cycles: 24,
            wall_time_minutes: 360,
        },
    }
}

const fn default_coordination_budget() -> TaskCoordinationBudget {
    TaskCoordinationBudget {
        planning_attempts: 3,
        model_invocations: 6,
        generated_tokens: 18_000,
        advisory_calls: 3,
        wall_time_minutes: 15,
    }
}

fn validate_usize(label: &str, value: usize, maximum: usize) -> Result<()> {
    if value == 0 || value > maximum {
        bail!("{label} must be between 1 and {maximum}");
    }
    Ok(())
}

fn validate_u64(label: &str, value: u64, maximum: u64) -> Result<()> {
    if value == 0 || value > maximum {
        bail!("{label} must be between 1 and {maximum}");
    }
    Ok(())
}

fn checked_add(label: &str, left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| anyhow::anyhow!("{label} overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip_and_hash_stably() {
        let document = TaskConfigDocument::default();
        let policy = document.clone().compile().unwrap();
        assert_eq!(policy.sha256.len(), 64);
        assert_eq!(policy, document.compile().unwrap());

        let dir = tempfile::tempdir().unwrap();
        TaskConfigDocument::default().save(dir.path()).unwrap();
        let loaded = TaskConfigDocument::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded, TaskConfigDocument::default());
    }

    #[test]
    fn config_cannot_grant_goal_selection_authority() {
        let input = "version = 1\nautomatic_goal_selection = true\n";
        assert!(toml::from_str::<TaskConfigDocument>(input).is_err());
    }

    #[test]
    fn qualification_rejects_tampering_and_impossible_capability() {
        let digest = "a".repeat(64);
        assert!(
            TaskPlannerQualification::new(
                digest.clone(),
                digest.clone(),
                digest.clone(),
                digest.clone(),
                false,
                true,
            )
            .is_err()
        );
        let mut qualification = TaskPlannerQualification::new(
            digest.clone(),
            digest.clone(),
            digest.clone(),
            digest,
            true,
            true,
        )
        .unwrap();
        assert!(qualification.validate().is_ok());
        qualification.model_sha256 = "bad".to_string();
        assert!(qualification.validate().is_err());
    }

    #[test]
    fn aggregate_ceiling_rejects_large_plan_sum() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let mut total = policy.large;
        for _ in 0..3 {
            total = total.checked_add(policy.large).unwrap();
        }
        assert!(!total.fits_within(&policy.aggregate.tasks));
    }
}
