use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::checks::CheckEvidenceLedger;
use crate::events::HandoffCommitSummary;
use crate::workspace::{ContentSnapshot, RepositoryContext};

use super::{
    ArtifactEnvelope, CodeReviewArtifact, CompiledWorkflowPolicy, ImplementationArtifact,
    PlanArtifact, PlanReviewArtifact, ReviewVerdict, WorkflowLimits, WorkflowOutcome,
    WorkflowStage,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowCounters {
    #[serde(default)]
    pub stage_steps: BTreeMap<WorkflowStage, usize>,
    pub model_invocations: usize,
    pub generated_tokens: usize,
    pub advisory_calls: usize,
    pub plan_cycles: usize,
    pub repair_cycles: usize,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowUsage {
    pub stage_steps: usize,
    pub model_invocations: usize,
    pub generated_tokens: usize,
    pub advisory_calls: usize,
}

impl WorkflowCounters {
    fn record(
        &mut self,
        stage: WorkflowStage,
        usage: WorkflowUsage,
        limits: WorkflowLimits,
        earned_work_unit_progress: usize,
    ) -> Option<WorkflowOutcome> {
        let stage_steps = self.stage_steps.entry(stage).or_default();
        *stage_steps = stage_steps.saturating_add(usage.stage_steps);
        self.model_invocations = self
            .model_invocations
            .saturating_add(usage.model_invocations);
        self.generated_tokens = self.generated_tokens.saturating_add(usage.generated_tokens);
        self.advisory_calls = self.advisory_calls.saturating_add(usage.advisory_calls);
        let stage_step_limit = limits.stage_steps.saturating_add(
            if matches!(
                stage,
                WorkflowStage::Implementing | WorkflowStage::Repairing
            ) {
                earned_work_unit_progress.min(super::MAX_WORK_UNIT_PROGRESS_CREDITS)
            } else {
                0
            },
        );
        if *stage_steps > stage_step_limit {
            Some(WorkflowOutcome::StepLimit)
        } else if self.model_invocations > limits.total_model_invocations {
            Some(WorkflowOutcome::InvocationLimit)
        } else if self.generated_tokens > limits.total_generated_tokens {
            Some(WorkflowOutcome::TokenLimit)
        } else if self.advisory_calls > limits.advisory_calls {
            Some(WorkflowOutcome::InvocationLimit)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowRun {
    pub version: u32,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub ready_evidence_schema: u32,
    pub id: String,
    pub source_turn_id: String,
    pub task: String,
    pub stage: WorkflowStage,
    pub policy: CompiledWorkflowPolicy,
    pub policy_sha256: String,
    #[serde(default)]
    pub workspace_graph_sha256: String,
    pub repository: RepositoryContext,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planning_snapshot: Option<ContentSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_control: Option<super::WorkflowGitControlState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused_stage: Option<WorkflowStage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<ArtifactEnvelope<PlanArtifact>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_review: Option<ArtifactEnvelope<PlanReviewArtifact>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation: Option<ArtifactEnvelope<ImplementationArtifact>>,
    #[serde(default)]
    pub selected_checks: Vec<String>,
    #[serde(default)]
    pub checks: CheckEvidenceLedger,
    #[serde(default)]
    pub stage_evidence: super::StageEvidenceBundle,
    #[serde(default)]
    pub work_units: super::WorkUnitLedger,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_review: Option<ArtifactEnvelope<CodeReviewArtifact>>,
    #[serde(default)]
    pub counters: WorkflowCounters,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<HandoffCommitSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ready_evidence: Option<super::ReadyEvidenceBundle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<WorkflowOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_reason: Option<String>,
}

impl WorkflowRun {
    pub fn start(
        id: impl Into<String>,
        source_turn_id: impl Into<String>,
        task: impl Into<String>,
        policy: CompiledWorkflowPolicy,
        repository: RepositoryContext,
    ) -> Result<Self> {
        let id = required("workflow id", id.into())?;
        let source_turn_id = required("source turn id", source_turn_id.into())?;
        let task = required("workflow task", task.into())?;
        policy.validate()?;
        Ok(Self {
            version: policy.version,
            ready_evidence_schema: super::READY_EVIDENCE_SCHEMA_VERSION,
            id,
            source_turn_id,
            task,
            stage: WorkflowStage::Planning,
            policy_sha256: policy.sha256.clone(),
            workspace_graph_sha256: String::new(),
            policy,
            planning_snapshot: Some(repository.invocation_baseline.content.clone()),
            repository,
            git_control: None,
            paused_stage: None,
            plan: None,
            plan_review: None,
            implementation: None,
            selected_checks: Vec::new(),
            checks: CheckEvidenceLedger::default(),
            stage_evidence: super::StageEvidenceBundle::default(),
            work_units: super::WorkUnitLedger::default(),
            content_fingerprint: None,
            code_review: None,
            counters: WorkflowCounters::default(),
            commit: None,
            ready_evidence: None,
            outcome: None,
            blocked_reason: None,
        })
    }

    pub fn apply(&mut self, event: WorkflowEvent) -> Result<()> {
        *self = reduce(self.clone(), event)?;
        Ok(())
    }

    pub fn planning_content(&self) -> &ContentSnapshot {
        self.planning_snapshot
            .as_ref()
            .unwrap_or(&self.repository.task_baseline.content)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkflowEvent {
    UsageRecorded {
        usage: WorkflowUsage,
    },
    PlanSubmitted {
        plan: ArtifactEnvelope<PlanArtifact>,
    },
    PlanReviewSubmitted {
        review: ArtifactEnvelope<PlanReviewArtifact>,
    },
    ReplanRequested {
        reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        planning_snapshot: Option<ContentSnapshot>,
    },
    UserInterventionQueued {
        planning_snapshot: ContentSnapshot,
    },
    ImplementationSubmitted {
        implementation: ArtifactEnvelope<ImplementationArtifact>,
    },
    ChecksPassed {
        content_fingerprint: String,
        selected_checks: Vec<String>,
        evidence: CheckEvidenceLedger,
    },
    ChecksFailed {
        content_fingerprint: String,
        selected_checks: Vec<String>,
        evidence: CheckEvidenceLedger,
        failed_check_ids: Vec<String>,
    },
    CodeReviewSubmitted {
        review: ArtifactEnvelope<CodeReviewArtifact>,
    },
    MutationObserved {
        content_fingerprint: String,
    },
    CommitCompleted {
        content_fingerprint: String,
        commit: HandoffCommitSummary,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repository_remote: Option<String>,
    },
    Blocked {
        outcome: WorkflowOutcome,
        reason: String,
    },
    Failed {
        outcome: WorkflowOutcome,
        reason: String,
    },
    Cancelled {
        reason: String,
    },
    Resumed,
}

pub fn reduce(mut run: WorkflowRun, event: WorkflowEvent) -> Result<WorkflowRun> {
    if run.stage.is_terminal()
        && !matches!(
            (&event, run.stage),
            (WorkflowEvent::Resumed, WorkflowStage::Blocked)
        )
    {
        bail!(
            "workflow '{}' is already terminal at {:?}",
            run.id,
            run.stage
        );
    }
    match event {
        WorkflowEvent::UsageRecorded { usage } => {
            let earned_work_unit_progress = run.work_units.progress_credited_units.len();
            if let Some(outcome) = run.counters.record(
                run.stage,
                usage,
                run.policy.limits,
                earned_work_unit_progress,
            ) {
                run.stage = WorkflowStage::Failed;
                run.outcome = Some(outcome);
                run.blocked_reason = Some("workflow-wide budget exhausted".to_string());
            }
        }
        WorkflowEvent::PlanSubmitted { plan } => {
            require_stage(
                run.stage,
                &[WorkflowStage::Planning, WorkflowStage::PlanRevision],
                "submit plan",
            )?;
            plan.validate_digest()?;
            if run.stage == WorkflowStage::PlanRevision {
                let unresolved = run
                    .plan_review
                    .as_ref()
                    .map(|review| {
                        review
                            .artifact
                            .challenges
                            .iter()
                            .filter(|challenge| challenge.severity.is_blocking())
                            .map(|challenge| challenge.id.as_str())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let resolutions = plan
                    .artifact
                    .resolved_challenge_ids
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                if unresolved
                    .iter()
                    .any(|challenge| !resolutions.contains(challenge))
                {
                    bail!("revised plan does not account for every blocking challenge id");
                }
            }
            run.plan = Some(plan);
            run.plan_review = None;
            run.work_units = super::WorkUnitLedger::default();
            run.implementation = None;
            run.selected_checks.clear();
            run.checks = CheckEvidenceLedger::default();
            run.content_fingerprint = None;
            run.code_review = None;
            run.commit = None;
            run.ready_evidence = None;
            run.stage = WorkflowStage::PlanReview;
        }
        WorkflowEvent::PlanReviewSubmitted { review } => {
            require_stage(
                run.stage,
                &[WorkflowStage::PlanReview],
                "submit plan review",
            )?;
            let plan = run.plan.as_ref().ok_or_else(|| {
                anyhow::anyhow!("plan review cannot be accepted without a current plan")
            })?;
            review.validate_digest()?;
            review.artifact.validate(plan)?;
            let verdict = review.artifact.verdict;
            run.plan_review = Some(review);
            if verdict == ReviewVerdict::Pass {
                run.stage = WorkflowStage::Implementing;
            } else if run.counters.plan_cycles >= run.policy.limits.plan_cycles {
                run.stage = WorkflowStage::Failed;
                run.outcome = Some(WorkflowOutcome::PlanCyclesExhausted);
                run.blocked_reason = Some("plan revision cycle limit exhausted".to_string());
            } else {
                run.counters.plan_cycles = run.counters.plan_cycles.saturating_add(1);
                run.stage = WorkflowStage::PlanRevision;
            }
        }
        WorkflowEvent::ReplanRequested {
            reason,
            planning_snapshot,
        } => {
            require_stage(
                run.stage,
                &[WorkflowStage::Implementing, WorkflowStage::Repairing],
                "request replan",
            )?;
            required("replan reason", reason)?;
            if let Some(planning_snapshot) = planning_snapshot {
                run.planning_snapshot = Some(planning_snapshot);
            }
            run.plan = None;
            run.plan_review = None;
            run.work_units = super::WorkUnitLedger::default();
            run.implementation = None;
            run.selected_checks.clear();
            run.checks = CheckEvidenceLedger::default();
            run.content_fingerprint = None;
            run.code_review = None;
            run.commit = None;
            run.ready_evidence = None;
            run.stage = WorkflowStage::Planning;
        }
        WorkflowEvent::UserInterventionQueued { planning_snapshot } => {
            require_stage(
                run.stage,
                &[
                    WorkflowStage::PlanReview,
                    WorkflowStage::Implementing,
                    WorkflowStage::Repairing,
                    WorkflowStage::Checking,
                    WorkflowStage::CodeReview,
                    WorkflowStage::Committing,
                ],
                "route user intervention",
            )?;
            run.planning_snapshot = Some(planning_snapshot);
            run.plan_review = None;
            run.work_units = super::WorkUnitLedger::default();
            run.implementation = None;
            run.selected_checks.clear();
            run.checks = CheckEvidenceLedger::default();
            run.content_fingerprint = None;
            run.code_review = None;
            run.commit = None;
            run.ready_evidence = None;
            if run.stage == WorkflowStage::PlanReview {
                run.stage = WorkflowStage::PlanRevision;
            } else {
                run.plan = None;
                run.stage = WorkflowStage::Planning;
            }
        }
        WorkflowEvent::ImplementationSubmitted { implementation } => {
            require_stage(
                run.stage,
                &[WorkflowStage::Implementing, WorkflowStage::Repairing],
                "submit implementation",
            )?;
            let plan = run.plan.as_ref().ok_or_else(|| {
                anyhow::anyhow!("implementation cannot be accepted without a current plan")
            })?;
            implementation.validate_digest()?;
            implementation.artifact.validate(plan)?;
            run.content_fingerprint = Some(implementation.artifact.content_fingerprint.clone());
            run.implementation = Some(implementation);
            run.selected_checks.clear();
            run.checks = CheckEvidenceLedger::default();
            run.code_review = None;
            run.commit = None;
            run.ready_evidence = None;
            run.stage = WorkflowStage::Checking;
        }
        WorkflowEvent::ChecksPassed {
            content_fingerprint,
            mut selected_checks,
            evidence,
        } => {
            require_stage(run.stage, &[WorkflowStage::Checking], "accept checks")?;
            require_fingerprint(&run, &content_fingerprint, "check evidence")?;
            selected_checks.sort();
            selected_checks.dedup();
            for check_id in &selected_checks {
                let check = evidence.get(check_id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "passing check batch has no evidence for selected check '{check_id}'"
                    )
                })?;
                if !check.success || check.timed_out {
                    bail!("passing check batch contains unsuccessful check '{check_id}'");
                }
            }
            run.selected_checks = selected_checks;
            run.checks = evidence;
            if run
                .implementation
                .as_ref()
                .is_some_and(|implementation| implementation.artifact.no_change)
            {
                run.stage = WorkflowStage::Ready;
                run.outcome = Some(WorkflowOutcome::NoChange);
                run.ready_evidence = None;
            } else {
                run.stage = WorkflowStage::CodeReview;
            }
        }
        WorkflowEvent::ChecksFailed {
            content_fingerprint,
            mut selected_checks,
            evidence,
            failed_check_ids,
        } => {
            require_stage(
                run.stage,
                &[WorkflowStage::Checking],
                "record failed checks",
            )?;
            require_fingerprint(&run, &content_fingerprint, "failed check evidence")?;
            if failed_check_ids.is_empty() {
                bail!("failed check event must name at least one failed check");
            }
            selected_checks.sort();
            selected_checks.dedup();
            run.selected_checks = selected_checks;
            run.checks = evidence;
            enter_repair(&mut run, WorkflowOutcome::ChecksFailed)?;
        }
        WorkflowEvent::CodeReviewSubmitted { review } => {
            require_stage(
                run.stage,
                &[WorkflowStage::CodeReview],
                "submit code review",
            )?;
            let fingerprint = run
                .content_fingerprint
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("code review requires a content fingerprint"))?;
            review.validate_digest()?;
            review.artifact.validate(fingerprint)?;
            let verdict = review.artifact.verdict;
            run.code_review = Some(review);
            if verdict == ReviewVerdict::Pass {
                run.stage = WorkflowStage::Committing;
            } else {
                enter_repair(&mut run, WorkflowOutcome::ReviewFailed)?;
            }
        }
        WorkflowEvent::MutationObserved {
            content_fingerprint,
        } => {
            required("content fingerprint", content_fingerprint.clone())?;
            require_stage(
                run.stage,
                &[
                    WorkflowStage::Implementing,
                    WorkflowStage::Repairing,
                    WorkflowStage::Checking,
                    WorkflowStage::CodeReview,
                    WorkflowStage::Committing,
                ],
                "record mutation",
            )?;
            run.content_fingerprint = Some(content_fingerprint);
            run.selected_checks.clear();
            run.checks = CheckEvidenceLedger::default();
            run.code_review = None;
            run.commit = None;
            run.ready_evidence = None;
            if matches!(
                run.stage,
                WorkflowStage::Checking | WorkflowStage::CodeReview | WorkflowStage::Committing
            ) {
                run.stage = WorkflowStage::Checking;
            }
        }
        WorkflowEvent::CommitCompleted {
            content_fingerprint,
            commit,
            repository_remote,
        } => {
            require_stage(run.stage, &[WorkflowStage::Committing], "complete commit")?;
            require_fingerprint(&run, &content_fingerprint, "commit")?;
            if run.code_review.as_ref().is_none_or(|review| {
                review.artifact.content_fingerprint != content_fingerprint
                    || review.artifact.verdict != ReviewVerdict::Pass
            }) {
                bail!("commit requires a passing current code review");
            }
            required("commit oid", commit.oid.clone())?;
            required("commit subject", commit.subject.clone())?;
            run.commit = Some(commit);
            run.stage = WorkflowStage::Ready;
            run.outcome = Some(WorkflowOutcome::Ready);
            run.ready_evidence_schema = super::READY_EVIDENCE_SCHEMA_VERSION;
            run.ready_evidence = Some(super::ReadyEvidenceBundle::from_run(
                &run,
                repository_remote,
            )?);
        }
        WorkflowEvent::Blocked { outcome, reason } => {
            if !matches!(
                outcome,
                WorkflowOutcome::ExecutorUnavailable | WorkflowOutcome::CommitBlocked
            ) {
                bail!("blocked workflow requires a blocked outcome");
            }
            run.paused_stage = Some(run.stage);
            run.stage = WorkflowStage::Blocked;
            run.outcome = Some(outcome);
            run.blocked_reason = Some(required("blocked reason", reason)?);
        }
        WorkflowEvent::Failed { outcome, reason } => {
            if matches!(
                outcome,
                WorkflowOutcome::Ready | WorkflowOutcome::NoChange | WorkflowOutcome::Cancelled
            ) {
                bail!("failed workflow cannot use a success/cancel outcome");
            }
            run.stage = WorkflowStage::Failed;
            run.paused_stage = None;
            run.outcome = Some(outcome);
            run.blocked_reason = Some(required("failure reason", reason)?);
        }
        WorkflowEvent::Cancelled { reason } => {
            run.stage = WorkflowStage::Cancelled;
            run.paused_stage = None;
            run.outcome = Some(WorkflowOutcome::Cancelled);
            run.blocked_reason = Some(required("cancellation reason", reason)?);
        }
        WorkflowEvent::Resumed => {
            require_stage(run.stage, &[WorkflowStage::Blocked], "resume workflow")?;
            run.stage = run
                .paused_stage
                .take()
                .ok_or_else(|| anyhow::anyhow!("blocked workflow has no resumable prior stage"))?;
            run.outcome = None;
            run.blocked_reason = None;
        }
    }
    Ok(run)
}

fn enter_repair(run: &mut WorkflowRun, exhausted_outcome: WorkflowOutcome) -> Result<()> {
    if run.counters.repair_cycles >= run.policy.limits.repair_cycles {
        run.stage = WorkflowStage::Failed;
        run.outcome = Some(WorkflowOutcome::RepairCyclesExhausted);
        run.blocked_reason = Some(format!(
            "repair cycle limit exhausted after {:?}",
            exhausted_outcome
        ));
    } else {
        run.counters.repair_cycles = run.counters.repair_cycles.saturating_add(1);
        run.stage = WorkflowStage::Repairing;
    }
    Ok(())
}

fn require_stage(current: WorkflowStage, expected: &[WorkflowStage], action: &str) -> Result<()> {
    if !expected.contains(&current) {
        bail!("cannot {action} while workflow stage is {current:?}");
    }
    Ok(())
}

fn require_fingerprint(run: &WorkflowRun, supplied: &str, kind: &str) -> Result<()> {
    if run.content_fingerprint.as_deref() != Some(supplied) {
        bail!("{kind} does not reference the current content fingerprint");
    }
    Ok(())
}

fn required(field: &str, value: String) -> Result<String> {
    if value.trim().is_empty() {
        bail!("{field} must not be empty");
    }
    Ok(value)
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::{
        AssessmentStatus, CodeAssessment, CodeAssessmentKind, PlanAcceptance, PlanAssessment,
        PlanAssessmentKind, PlanPath, PlanRequirement, PlanStep, PlannedChange,
        REQUIRED_CODE_ASSESSMENTS, REQUIRED_PLAN_ASSESSMENTS, ReviewSeverity, WorkflowCheckpoint,
        WorkflowConfigDocument,
    };

    fn repository() -> (tempfile::TempDir, RepositoryContext) {
        let dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init", "-b", "main"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        let repository = RepositoryContext::capture(dir.path(), dir.path()).unwrap();
        (dir, repository)
    }

    fn run() -> WorkflowRun {
        let (_dir, repository) = repository();
        WorkflowRun::start(
            "workflow-1",
            "turn-1",
            "implement",
            WorkflowConfigDocument::default().compile().unwrap(),
            repository,
        )
        .unwrap()
    }

    #[test]
    fn new_workflow_plans_from_the_current_invocation_baseline() {
        let (dir, initial) = repository();
        std::fs::write(dir.path().join("adopted.txt"), "partial work\n").unwrap();
        let resumed =
            RepositoryContext::resume(dir.path(), dir.path(), initial.task_baseline.clone())
                .unwrap();
        assert_ne!(
            resumed.task_baseline.content.fingerprint,
            resumed.invocation_baseline.content.fingerprint
        );

        let run = WorkflowRun::start(
            "workflow-resumed-scratch",
            "turn-resumed-scratch",
            "repair adopted work",
            WorkflowConfigDocument::default().compile().unwrap(),
            resumed.clone(),
        )
        .unwrap();

        assert_eq!(
            run.planning_content().fingerprint,
            resumed.invocation_baseline.content.fingerprint
        );
    }

    fn plan() -> ArtifactEnvelope<PlanArtifact> {
        ArtifactEnvelope::new(
            "plan-1",
            PlanArtifact {
                summary: "plan".to_string(),
                requirements: vec![PlanRequirement {
                    id: "r1".to_string(),
                    description: "requirement".to_string(),
                    source: "user".to_string(),
                }],
                steps: vec![PlanStep {
                    id: "s1".to_string(),
                    requirement_ids: vec!["r1".to_string()],
                    component_ids: Vec::new(),
                    paths: vec![PlanPath {
                        path: "new.txt".to_string(),
                        change: PlannedChange::Create,
                    }],
                    description: "create file".to_string(),
                }],
                acceptance: vec![PlanAcceptance {
                    id: "a1".to_string(),
                    requirement_ids: vec!["r1".to_string()],
                    check_ids: Vec::new(),
                    description: "works".to_string(),
                }],
                risks: Vec::new(),
                assumptions: Vec::new(),
                open_questions: Vec::new(),
                resolved_challenge_ids: Vec::new(),
            },
        )
        .unwrap()
    }

    fn plan_review(plan: &ArtifactEnvelope<PlanArtifact>) -> ArtifactEnvelope<PlanReviewArtifact> {
        ArtifactEnvelope::new(
            "plan-review-1",
            PlanReviewArtifact {
                plan_id: plan.id.clone(),
                plan_sha256: plan.sha256.clone(),
                assessments: REQUIRED_PLAN_ASSESSMENTS
                    .into_iter()
                    .map(|kind| PlanAssessment {
                        kind,
                        status: AssessmentStatus::Pass,
                        evidence: Vec::new(),
                        explanation: "checked".to_string(),
                    })
                    .collect(),
                challenges: Vec::new(),
                verdict: ReviewVerdict::Pass,
            },
        )
        .unwrap()
    }

    fn implementation(
        plan: &ArtifactEnvelope<PlanArtifact>,
    ) -> ArtifactEnvelope<ImplementationArtifact> {
        ArtifactEnvelope::new(
            "implementation-1",
            ImplementationArtifact {
                plan_id: plan.id.clone(),
                plan_sha256: plan.sha256.clone(),
                content_fingerprint: "content-1".to_string(),
                steps: vec![super::super::ImplementationStep {
                    step_id: "s1".to_string(),
                    status: super::super::ImplementationStepStatus::Completed,
                    touched_paths: vec!["new.txt".to_string()],
                    summary: "created".to_string(),
                }],
                summary: "implemented".to_string(),
                no_change: false,
                semantic_commit_subject: "feat: add file".to_string(),
            },
        )
        .unwrap()
    }

    fn code_review() -> ArtifactEnvelope<CodeReviewArtifact> {
        ArtifactEnvelope::new(
            "code-review-1",
            CodeReviewArtifact {
                content_fingerprint: "content-1".to_string(),
                assessments: REQUIRED_CODE_ASSESSMENTS
                    .into_iter()
                    .map(|kind| CodeAssessment {
                        kind,
                        status: AssessmentStatus::Pass,
                        evidence: Vec::new(),
                        explanation: "checked".to_string(),
                    })
                    .collect(),
                findings: Vec::new(),
                verdict: ReviewVerdict::Pass,
            },
        )
        .unwrap()
    }

    #[test]
    fn happy_path_cannot_skip_challenged_plan_checks_review_or_commit() {
        let mut run = run();
        assert!(
            run.apply(WorkflowEvent::ImplementationSubmitted {
                implementation: implementation(&plan()),
            })
            .is_err()
        );

        let plan = plan();
        run.apply(WorkflowEvent::PlanSubmitted { plan: plan.clone() })
            .unwrap();
        run.apply(WorkflowEvent::PlanReviewSubmitted {
            review: plan_review(&plan),
        })
        .unwrap();
        run.apply(WorkflowEvent::ImplementationSubmitted {
            implementation: implementation(&plan),
        })
        .unwrap();
        let mut check_evidence = CheckEvidenceLedger::default();
        check_evidence.record(crate::checks::CheckEvidence {
            check_id: "test".to_string(),
            command: "cargo test".to_string(),
            cwd: ".".to_string(),
            command_fingerprint: "command-sha".to_string(),
            input_fingerprint: "input-sha".to_string(),
            dependency_outputs: BTreeMap::new(),
            output_fingerprint: None,
            exit_status: 0,
            success: true,
            timed_out: false,
            duration_ms: 1,
            output: String::new(),
            executor: "local".to_string(),
            source: crate::checks::EvidenceSource::Handoff,
        });
        run.apply(WorkflowEvent::ChecksPassed {
            content_fingerprint: "content-1".to_string(),
            selected_checks: vec!["test".to_string()],
            evidence: check_evidence,
        })
        .unwrap();
        assert_eq!(run.stage, WorkflowStage::CodeReview);
        run.apply(WorkflowEvent::CodeReviewSubmitted {
            review: code_review(),
        })
        .unwrap();
        assert_eq!(run.stage, WorkflowStage::Committing);
        run.apply(WorkflowEvent::CommitCompleted {
            content_fingerprint: "content-1".to_string(),
            commit: HandoffCommitSummary {
                oid: "abc123".to_string(),
                subject: "feat: add file".to_string(),
            },
            repository_remote: Some("git@example.test:team/project.git".to_string()),
        })
        .unwrap();
        assert_eq!(run.stage, WorkflowStage::Ready);
        assert_eq!(run.outcome, Some(WorkflowOutcome::Ready));
        let evidence = run.ready_evidence.as_ref().unwrap();
        assert_eq!(evidence.workflow_id, "workflow-1");
        assert_eq!(evidence.commit_oid, "abc123");
        assert_eq!(evidence.plan_sha256, plan.sha256);
        assert_eq!(
            evidence.review_sha256,
            run.code_review.as_ref().unwrap().sha256
        );
        assert_eq!(evidence.check_evidence_ids, vec!["check:test"]);
        assert_eq!(
            evidence.repository_remote.as_deref(),
            Some("git@example.test:team/project.git")
        );
        let summary = super::super::WorkflowSummary::from(&run);
        assert_eq!(summary.ready_evidence.as_ref(), Some(evidence));
        assert_eq!(summary.plan.as_ref(), Some(&plan));
        assert_eq!(summary.plan_review.as_ref(), run.plan_review.as_ref());
        WorkflowCheckpoint::new(run.clone())
            .unwrap()
            .validate()
            .unwrap();
        let mut tampered = run;
        tampered.ready_evidence.as_mut().unwrap().review_sha256 = "tampered".to_string();
        assert!(
            WorkflowCheckpoint::new(tampered)
                .unwrap()
                .validate()
                .is_err()
        );
    }

    #[test]
    fn stale_review_and_commit_fingerprints_are_rejected() {
        let mut run = run();
        let plan = plan();
        run.apply(WorkflowEvent::PlanSubmitted { plan: plan.clone() })
            .unwrap();
        run.apply(WorkflowEvent::PlanReviewSubmitted {
            review: plan_review(&plan),
        })
        .unwrap();
        run.apply(WorkflowEvent::ImplementationSubmitted {
            implementation: implementation(&plan),
        })
        .unwrap();
        assert!(
            run.apply(WorkflowEvent::ChecksPassed {
                content_fingerprint: "stale".to_string(),
                selected_checks: Vec::new(),
                evidence: CheckEvidenceLedger::default(),
            })
            .is_err()
        );
    }

    #[test]
    fn blocking_reviews_enter_bounded_repair() {
        let mut run = run();
        let plan = plan();
        run.apply(WorkflowEvent::PlanSubmitted { plan: plan.clone() })
            .unwrap();
        let mut review = plan_review(&plan).artifact;
        review.verdict = ReviewVerdict::Revise;
        review.assessments[0].status = AssessmentStatus::Concern;
        review.assessments[0].explanation.clear();
        review.challenges.push(super::super::ReviewChallenge {
            id: "c1".to_string(),
            severity: ReviewSeverity::P1,
            requirement_ids: vec!["r1".to_string()],
            description: "missing risk".to_string(),
            evidence: Vec::new(),
        });
        run.apply(WorkflowEvent::PlanReviewSubmitted {
            review: ArtifactEnvelope::new("review-revise", review).unwrap(),
        })
        .unwrap();
        assert_eq!(run.stage, WorkflowStage::PlanRevision);
        assert_eq!(run.counters.plan_cycles, 1);
    }

    #[test]
    fn user_intervention_during_plan_review_returns_to_the_planner() {
        let mut run = run();
        let plan = plan();
        run.apply(WorkflowEvent::PlanSubmitted { plan: plan.clone() })
            .unwrap();
        let snapshot = run.planning_content().clone();

        run.apply(WorkflowEvent::UserInterventionQueued {
            planning_snapshot: snapshot,
        })
        .unwrap();

        assert_eq!(run.stage, WorkflowStage::PlanRevision);
        assert_eq!(run.plan, Some(plan));
        assert!(run.plan_review.is_none());
        assert_eq!(run.counters.plan_cycles, 0);
    }

    #[test]
    fn late_build_feedback_invalidates_review_and_returns_to_planning() {
        let mut run = run();
        run.stage = WorkflowStage::CodeReview;
        run.plan = Some(plan());
        run.content_fingerprint = Some("checked".to_string());
        run.selected_checks.push("test".to_string());
        let snapshot = run.planning_content().clone();

        run.apply(WorkflowEvent::UserInterventionQueued {
            planning_snapshot: snapshot.clone(),
        })
        .unwrap();

        assert_eq!(run.stage, WorkflowStage::Planning);
        assert_eq!(run.planning_snapshot, Some(snapshot));
        assert!(run.plan.is_none());
        assert!(run.content_fingerprint.is_none());
        assert!(run.selected_checks.is_empty());
        assert!(run.code_review.is_none());
    }

    #[test]
    fn mutation_after_review_invalidates_evidence_and_returns_to_checks() {
        let mut run = run();
        run.stage = WorkflowStage::Committing;
        run.content_fingerprint = Some("old".to_string());
        run.code_review = Some(code_review());
        run.selected_checks.push("test".to_string());
        run.apply(WorkflowEvent::MutationObserved {
            content_fingerprint: "new".to_string(),
        })
        .unwrap();
        assert_eq!(run.stage, WorkflowStage::Checking);
        assert!(run.code_review.is_none());
        assert!(run.selected_checks.is_empty());
    }

    #[test]
    fn global_usage_limits_fail_deterministically() {
        let mut run = run();
        let limit = run.policy.limits.total_model_invocations;
        run.apply(WorkflowEvent::UsageRecorded {
            usage: WorkflowUsage {
                model_invocations: limit + 1,
                ..WorkflowUsage::default()
            },
        })
        .unwrap();
        assert_eq!(run.stage, WorkflowStage::Failed);
        assert_eq!(run.outcome, Some(WorkflowOutcome::InvocationLimit));
    }

    #[test]
    fn assessment_enums_remain_distinct() {
        assert_ne!(
            PlanAssessmentKind::Architecture as u8,
            PlanAssessmentKind::TestStrategy as u8
        );
        assert_ne!(
            CodeAssessmentKind::Architecture as u8,
            CodeAssessmentKind::Tests as u8
        );
    }
}
