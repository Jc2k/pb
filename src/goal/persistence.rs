use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{GoalCompletionBasis, GoalOutcome, GoalRun, GoalStage};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalCheckpoint {
    pub sha256: String,
    pub run: GoalRun,
}

impl GoalCheckpoint {
    pub fn new(run: GoalRun) -> Result<Self> {
        let sha256 = goal_digest(&run)?;
        let checkpoint = Self { sha256, run };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    pub fn validate(&self) -> Result<()> {
        let expected = goal_digest(&self.run)?;
        if self.sha256 != expected {
            bail!(
                "goal checkpoint digest mismatch: expected {}, got {}",
                expected,
                self.sha256
            );
        }
        self.run.validate()
    }
}

fn goal_digest(run: &GoalRun) -> Result<String> {
    let bytes = serde_json::to_vec(run).context("failed to serialize goal checkpoint")?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalSummary {
    pub id: String,
    pub objective: String,
    pub stage: GoalStage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<GoalOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_basis: Option<GoalCompletionBasis>,
    pub completed_milestones: usize,
    pub total_milestones: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_milestone_title: Option<String>,
    pub generated_tokens: usize,
    pub generated_token_limit: usize,
    pub workflows: usize,
    pub workflow_limit: usize,
    pub active: bool,
    pub plan_sha256: String,
}

impl From<&GoalRun> for GoalSummary {
    fn from(run: &GoalRun) -> Self {
        let counters = run.effective_counters();
        Self {
            id: run.id.clone(),
            objective: run.objective.clone(),
            stage: run.stage,
            outcome: run.outcome,
            completion_basis: run.completion_basis,
            completed_milestones: run
                .milestones
                .iter()
                .filter(|milestone| milestone.status.is_completed())
                .count(),
            total_milestones: run
                .milestones
                .iter()
                .filter(|milestone| !milestone.status.is_superseded())
                .count(),
            current_milestone_title: run
                .current_milestone()
                .map(|milestone| milestone.title.clone()),
            generated_tokens: counters.generated_tokens,
            generated_token_limit: run.budget.total_generated_tokens,
            workflows: counters.workflows,
            workflow_limit: run.budget.max_workflows,
            active: !run.stage.is_terminal(),
            plan_sha256: run.plan_sha256.clone(),
        }
    }
}
