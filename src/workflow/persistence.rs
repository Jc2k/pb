use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{WorkflowOutcome, WorkflowRun, WorkflowStage};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowCheckpoint {
    pub sha256: String,
    pub run: WorkflowRun,
}

impl WorkflowCheckpoint {
    pub fn new(run: WorkflowRun) -> Result<Self> {
        let sha256 = workflow_digest(&run)?;
        Ok(Self { sha256, run })
    }

    pub fn validate(&self) -> Result<()> {
        let expected = workflow_digest(&self.run)?;
        if self.sha256 != expected {
            bail!(
                "workflow checkpoint digest mismatch: expected {}, got {}",
                expected,
                self.sha256
            );
        }
        if self.run.policy.sha256 != self.run.policy_sha256 {
            bail!("workflow checkpoint policy hash does not match compiled policy");
        }
        self.run.policy.validate()?;
        Ok(())
    }
}

fn workflow_digest(run: &WorkflowRun) -> Result<String> {
    let bytes = serde_json::to_vec(run).context("failed to serialize workflow checkpoint")?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowSummary {
    pub id: String,
    pub source_turn_id: String,
    pub task: String,
    pub stage: WorkflowStage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<WorkflowOutcome>,
    pub policy_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_oid: Option<String>,
}

impl From<&WorkflowRun> for WorkflowSummary {
    fn from(run: &WorkflowRun) -> Self {
        Self {
            id: run.id.clone(),
            source_turn_id: run.source_turn_id.clone(),
            task: run.task.clone(),
            stage: run.stage,
            outcome: run.outcome,
            policy_sha256: run.policy_sha256.clone(),
            commit_oid: run.commit.as_ref().map(|commit| commit.oid.clone()),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversationHandoff {
    #[serde(default)]
    pub source_turn_ids: Vec<String>,
    pub task_summary: String,
    #[serde(default)]
    pub user_decisions: Vec<String>,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub unresolved_questions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeliveryProposal {
    pub id: String,
    pub source_turn_id: String,
    pub task_summary: String,
}

impl DeliveryProposal {
    pub fn handoff(&self) -> ConversationHandoff {
        ConversationHandoff {
            source_turn_ids: vec![self.source_turn_id.clone()],
            task_summary: self.task_summary.clone(),
            proposal_id: Some(self.id.clone()),
            ..ConversationHandoff::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadyEvidenceBundle {
    pub workflow_id: String,
    pub commit_oid: String,
    pub plan_sha256: String,
    pub review_sha256: String,
    #[serde(default)]
    pub check_evidence_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository_remote: Option<String>,
}

impl ReadyEvidenceBundle {
    pub fn from_run(run: &WorkflowRun, check_evidence_ids: Vec<String>) -> Result<Self> {
        if run.stage != WorkflowStage::Ready || run.outcome != Some(WorkflowOutcome::Ready) {
            bail!("ready evidence requires a successful delta-bearing workflow");
        }
        let commit = run
            .commit
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ready workflow has no managed commit evidence"))?;
        let plan = run
            .plan
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ready workflow has no accepted plan"))?;
        let review = run
            .code_review
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ready workflow has no code review evidence"))?;
        Ok(Self {
            workflow_id: run.id.clone(),
            commit_oid: commit.oid.clone(),
            plan_sha256: plan.sha256.clone(),
            review_sha256: review.sha256.clone(),
            check_evidence_ids,
            repository_remote: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::{WorkflowConfigDocument, WorkflowRun};
    use crate::workspace::RepositoryContext;

    #[test]
    fn checkpoint_digest_detects_tampering() {
        let dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .arg("init")
            .current_dir(dir.path())
            .output()
            .unwrap();
        let repository = RepositoryContext::capture(dir.path(), dir.path()).unwrap();
        let run = WorkflowRun::start(
            "workflow-1",
            "turn-1",
            "task",
            WorkflowConfigDocument::default().compile().unwrap(),
            repository,
        )
        .unwrap();
        let mut checkpoint = WorkflowCheckpoint::new(run).unwrap();
        checkpoint.run.task = "tampered".to_string();
        assert!(checkpoint.validate().is_err());
    }
}
