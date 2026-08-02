use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    ArtifactEnvelope, PlanArtifact, PlanReviewArtifact, WorkflowBlockCause, WorkflowOutcome,
    WorkflowRun, WorkflowStage,
};

pub const READY_EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowGitControlState {
    pub head: String,
    pub index_sha256: String,
    pub refs_sha256: String,
}

impl WorkflowGitControlState {
    pub fn difference(&self, current: &Self) -> String {
        let mut changed = Vec::new();
        if self.head != current.head {
            changed.push("HEAD");
        }
        if self.index_sha256 != current.index_sha256 {
            changed.push("index");
        }
        if self.refs_sha256 != current.refs_sha256 {
            changed.push("refs");
        }
        changed.join(", ")
    }
}

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
        if self.run.stage == WorkflowStage::Blocked && self.run.paused_stage.is_none() {
            bail!("blocked workflow checkpoint has no resumable prior stage");
        }
        if self.run.stage != WorkflowStage::Blocked && self.run.paused_stage.is_some() {
            bail!("non-blocked workflow checkpoint contains a paused stage");
        }
        if self.run.ready_evidence_schema > READY_EVIDENCE_SCHEMA_VERSION {
            bail!(
                "unsupported ready evidence schema {}; expected at most {}",
                self.run.ready_evidence_schema,
                READY_EVIDENCE_SCHEMA_VERSION
            );
        }
        self.run.stage_evidence.validate()?;
        self.run.work_units.validate()?;
        if let Some(plan) = &self.run.plan {
            self.run.work_units.validate_plan(&plan.id, &plan.sha256)?;
        } else if self.run.work_units.is_initialized() {
            bail!("workflow checkpoint has work units without an accepted plan");
        }
        match &self.run.ready_evidence {
            Some(_) if self.run.ready_evidence_schema == 0 => {
                bail!("legacy workflow checkpoint cannot contain current ready evidence");
            }
            Some(evidence) => evidence.validate_against(&self.run)?,
            None if self.run.ready_evidence_schema >= READY_EVIDENCE_SCHEMA_VERSION
                && self.run.stage == WorkflowStage::Ready
                && self.run.outcome == Some(WorkflowOutcome::Ready) =>
            {
                bail!("current ready workflow checkpoint has no reviewed delivery evidence");
            }
            None => {}
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ready_evidence: Option<ReadyEvidenceBundle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused_stage: Option<WorkflowStage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_cause: Option<WorkflowBlockCause>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery: Option<WorkflowRecovery>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<ArtifactEnvelope<PlanArtifact>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_review: Option<ArtifactEnvelope<PlanReviewArtifact>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowRecovery {
    Resume,
    RestartFromCurrentFiles,
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
            ready_evidence: run.ready_evidence.clone(),
            paused_stage: run.paused_stage,
            blocked_reason: run.blocked_reason.clone(),
            blocked_cause: run.blocked_cause.or_else(|| {
                Some(WorkflowBlockCause::classify(
                    run.outcome?,
                    run.blocked_reason.as_deref()?,
                ))
            }),
            recovery: match (run.stage, run.outcome) {
                (WorkflowStage::Blocked, Some(WorkflowOutcome::ExecutorUnavailable)) => {
                    Some(WorkflowRecovery::Resume)
                }
                (WorkflowStage::Blocked, Some(WorkflowOutcome::CommitBlocked)) => {
                    Some(WorkflowRecovery::RestartFromCurrentFiles)
                }
                _ => None,
            },
            plan: run.plan.clone(),
            plan_review: run.plan_review.clone(),
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
#[serde(deny_unknown_fields)]
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
    pub fn from_run(run: &WorkflowRun, repository_remote: Option<String>) -> Result<Self> {
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
        let mut check_evidence_ids = Vec::with_capacity(run.selected_checks.len());
        for check_id in &run.selected_checks {
            let evidence = run.checks.get(check_id).ok_or_else(|| {
                anyhow::anyhow!("ready workflow has no evidence for selected check '{check_id}'")
            })?;
            if !evidence.success || evidence.timed_out {
                bail!("ready workflow selected check '{check_id}' is not successful");
            }
            check_evidence_ids.push(format!("check:{check_id}"));
        }
        check_evidence_ids.sort();
        check_evidence_ids.dedup();
        let repository_remote = repository_remote
            .map(|remote| remote.trim().to_string())
            .filter(|remote| !remote.is_empty());
        Ok(Self {
            workflow_id: run.id.clone(),
            commit_oid: commit.oid.clone(),
            plan_sha256: plan.sha256.clone(),
            review_sha256: review.sha256.clone(),
            check_evidence_ids,
            repository_remote,
        })
    }

    pub fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("workflow id", self.workflow_id.as_str()),
            ("commit oid", self.commit_oid.as_str()),
        ] {
            if value.trim().is_empty() {
                bail!("ready evidence has an empty {label}");
            }
        }
        for (label, digest) in [
            ("plan", self.plan_sha256.as_str()),
            ("review", self.review_sha256.as_str()),
        ] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                bail!("ready evidence {label} digest is not a lowercase SHA-256");
            }
        }
        if self
            .repository_remote
            .as_ref()
            .is_some_and(|remote| remote.trim().is_empty())
        {
            bail!("ready evidence repository remote is empty");
        }
        if let Some(remote) = self.repository_remote.as_deref() {
            if let Ok(parsed) = url::Url::parse(remote) {
                if !parsed.username().is_empty() || parsed.password().is_some() {
                    bail!("ready evidence repository remote contains URL credentials");
                }
            } else if let Some((user, _)) = remote.split_once('@')
                && user != "git"
            {
                bail!("ready evidence repository remote contains an unsafe user component");
            }
        }
        if self.check_evidence_ids.iter().any(|evidence_id| {
            !evidence_id.starts_with("check:")
                || evidence_id == "check:"
                || evidence_id.chars().any(char::is_whitespace)
        }) {
            bail!("ready evidence contains an invalid check evidence id");
        }
        let mut normalized = self.check_evidence_ids.clone();
        normalized.sort();
        normalized.dedup();
        if normalized != self.check_evidence_ids {
            bail!("ready evidence check ids must be sorted and unique");
        }
        Ok(())
    }

    pub fn validate_against(&self, run: &WorkflowRun) -> Result<()> {
        self.validate()?;
        let expected = Self::from_run(run, self.repository_remote.clone())?;
        if self != &expected {
            bail!("ready evidence does not match the reviewed workflow artifacts");
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).context("failed to serialize ready evidence")?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::{WorkflowConfigDocument, WorkflowEvent, WorkflowRun};
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

    #[test]
    fn current_ready_checkpoint_requires_evidence_but_legacy_ready_remains_readable() {
        let dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .arg("init")
            .current_dir(dir.path())
            .output()
            .unwrap();
        let repository = RepositoryContext::capture(dir.path(), dir.path()).unwrap();
        let mut run = WorkflowRun::start(
            "workflow-1",
            "turn-1",
            "task",
            WorkflowConfigDocument::default().compile().unwrap(),
            repository,
        )
        .unwrap();
        run.stage = WorkflowStage::Ready;
        run.outcome = Some(WorkflowOutcome::Ready);
        assert!(
            WorkflowCheckpoint::new(run.clone())
                .unwrap()
                .validate()
                .is_err()
        );

        run.ready_evidence_schema = 0;
        let legacy = WorkflowCheckpoint::new(run).unwrap();
        legacy.validate().unwrap();
        assert!(WorkflowSummary::from(&legacy.run).ready_evidence.is_none());
    }

    #[test]
    fn blocked_summary_exposes_the_safe_recovery_path() {
        let dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .arg("init")
            .current_dir(dir.path())
            .output()
            .unwrap();
        let repository = RepositoryContext::capture(dir.path(), dir.path()).unwrap();
        let mut run = WorkflowRun::start(
            "workflow-recovery",
            "turn-recovery",
            "task",
            WorkflowConfigDocument::default().compile().unwrap(),
            repository,
        )
        .unwrap();
        run.apply(WorkflowEvent::Blocked {
            outcome: WorkflowOutcome::CommitBlocked,
            reason: "repository content changed during review".to_string(),
        })
        .unwrap();

        let summary = WorkflowSummary::from(&run);
        assert_eq!(summary.paused_stage, Some(WorkflowStage::Planning));
        assert_eq!(
            summary.recovery,
            Some(WorkflowRecovery::RestartFromCurrentFiles)
        );
        assert_eq!(
            summary.blocked_reason.as_deref(),
            Some("repository content changed during review")
        );
    }
}
