use std::collections::HashSet;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{CompiledGoalPolicy, GoalBudget};

pub const GOAL_RUN_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalContinuationPolicy {
    #[default]
    ReviewPlanThenAutomatic,
    ManualMilestones,
    AutomaticWithinLimits,
}

impl GoalContinuationPolicy {
    pub const fn automatically_continues(self) -> bool {
        matches!(
            self,
            Self::ReviewPlanThenAutomatic | Self::AutomaticWithinLimits
        )
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalStage {
    Planning,
    PlanReview,
    PlanRevision,
    AwaitingPlanApproval,
    RunningMilestone,
    Evaluating,
    AwaitingUserReview,
    Paused,
    Blocked,
    Completed,
    Failed,
    Cancelled,
}

impl GoalStage {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalOutcome {
    Complete,
    BudgetExhausted,
    CriteriaUnsatisfied,
    MilestoneFailed,
    RepeatedNoProgress,
    AuthorityDenied,
    ContextLimit,
    EngineError,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalCompletionBasis {
    MachineVerified,
    UserAccepted,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalVerifier {
    WorkflowReady,
    #[default]
    ReviewRequired,
    UserConfirmation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalCriterionInput {
    pub text: String,
    #[serde(default)]
    pub verifier: GoalVerifier,
}

/// A read-only conversation artifact. Recording this never starts work or grants authority; an
/// explicit Goal UI action, API call, or Auto turn may cite it to create an approval-gated run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalProposal {
    pub id: String,
    pub source_turn_id: String,
    pub objective: String,
    pub criteria: Vec<GoalCriterionInput>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalCriterionStatus {
    Pending,
    EvidenceReady,
    MachineVerified,
    UserAccepted,
    Unsatisfied,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalCriterionState {
    pub id: String,
    pub text: String,
    pub verifier: GoalVerifier,
    pub status: GoalCriterionStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalMilestoneStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Superseded,
}

impl GoalMilestoneStatus {
    pub const fn is_completed(self) -> bool {
        matches!(self, Self::Completed)
    }

    pub const fn is_superseded(self) -> bool {
        matches!(self, Self::Superseded)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalMilestoneRun {
    pub id: String,
    pub title: String,
    pub description: String,
    pub criterion_ids: Vec<String>,
    pub plan_version: u32,
    pub status: GoalMilestoneStatus,
    pub attempts: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow: Option<crate::workflow::WorkflowCheckpoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow_summary: Option<crate::workflow::WorkflowSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalPlanArtifact {
    pub version: u32,
    pub objective: String,
    pub milestones: Vec<GoalMilestoneRun>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalCounters {
    pub workflows: usize,
    pub model_invocations: usize,
    pub generated_tokens: usize,
    pub advisory_calls: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalAuthorityEnvelope {
    pub workdir: String,
    pub publication: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalModelBrief {
    pub id: String,
    pub objective: String,
    pub stage: GoalStage,
    pub plan_sha256: String,
    pub budget: GoalBudget,
    pub counters: GoalCounters,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_milestone: Option<String>,
    pub completed_milestones: usize,
    pub total_milestones: usize,
}

impl GoalModelBrief {
    /// Validate a bounded, read-only Goal projection before it is supplied to a model or test
    /// harness. A brief never carries authority, but rejecting impossible counters and stages keeps
    /// experiments from drawing conclusions from controller states that could not exist.
    pub fn validate(&self) -> Result<()> {
        required("goal id", self.id.clone())?;
        required("goal objective", self.objective.clone())?;
        let plan_sha256 = required("goal plan digest", self.plan_sha256.clone())?;
        if plan_sha256.len() != 64 || !plan_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("goal plan digest must be a 64-character hexadecimal SHA-256");
        }
        self.budget.validate()?;
        if self.total_milestones == 0 || self.total_milestones > self.budget.max_milestones {
            bail!("goal milestone total must fit the goal budget");
        }
        if self.completed_milestones > self.total_milestones {
            bail!("completed goal milestones exceed the milestone total");
        }
        if self.counters.workflows > self.budget.max_workflows
            || self.counters.model_invocations > self.budget.total_model_invocations
            || self.counters.generated_tokens > self.budget.total_generated_tokens
        {
            bail!("goal counters exceed the supplied goal budget");
        }
        if self.stage == GoalStage::RunningMilestone && self.current_milestone.is_none() {
            bail!("running goal brief has no current milestone");
        }
        if self.stage.is_terminal() && self.current_milestone.is_some() {
            bail!("terminal goal brief retains a current milestone");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalAmendmentDraft {
    pub id: String,
    pub base_goal_sha256: String,
    pub objective: String,
    pub criteria: Vec<GoalCriterionInput>,
    pub continuation: GoalContinuationPolicy,
    pub budget: GoalBudget,
    pub replacement_milestones: Vec<GoalMilestoneRun>,
    pub replacement_plan_sha256: String,
    #[serde(default = "default_amendment_resume_stage")]
    pub resume_stage: GoalStage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalAmendmentRecord {
    pub id: String,
    pub base_goal_sha256: String,
    pub plan_sha256: String,
    pub accepted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalRun {
    pub version: u32,
    pub id: String,
    pub session_id: String,
    pub objective: String,
    pub stage: GoalStage,
    pub plan_version: u32,
    pub plan_sha256: String,
    pub policy: CompiledGoalPolicy,
    pub budget: GoalBudget,
    pub authority: GoalAuthorityEnvelope,
    pub continuation: GoalContinuationPolicy,
    pub criteria: Vec<GoalCriterionState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retired_criteria: Vec<GoalCriterionState>,
    pub milestones: Vec<GoalMilestoneRun>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_milestone_id: Option<String>,
    #[serde(default)]
    pub counters: GoalCounters,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused_stage: Option<GoalStage>,
    #[serde(default)]
    pub pause_requested: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_amendment: Option<GoalAmendmentDraft>,
    #[serde(default)]
    pub amendments: Vec<GoalAmendmentRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<GoalOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_basis: Option<GoalCompletionBasis>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl GoalRun {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        id: impl Into<String>,
        session_id: impl Into<String>,
        objective: impl Into<String>,
        criteria: Vec<GoalCriterionInput>,
        continuation: GoalContinuationPolicy,
        requested_budget: Option<GoalBudget>,
        policy: CompiledGoalPolicy,
        workdir: impl Into<String>,
        now_ms: u64,
    ) -> Result<Self> {
        policy.validate()?;
        let id = required("goal id", id.into())?;
        let session_id = required("session id", session_id.into())?;
        let objective = required("goal objective", objective.into())?;
        let workdir = required("goal workdir", workdir.into())?;
        let budget = policy.budget(requested_budget)?;
        let criteria = normalize_criteria(criteria, 1)?;
        if criteria.len() > budget.max_milestones {
            bail!(
                "goal requires {} milestones but the budget allows {}",
                criteria.len(),
                budget.max_milestones
            );
        }
        let milestones = milestones_for(&id, 1, &criteria);
        let plan_sha256 = plan_digest(1, &objective, &milestones)?;
        let run = Self {
            version: GOAL_RUN_VERSION,
            id,
            session_id,
            objective,
            stage: GoalStage::AwaitingPlanApproval,
            plan_version: 1,
            plan_sha256,
            policy,
            budget,
            authority: GoalAuthorityEnvelope {
                workdir,
                publication: false,
            },
            continuation,
            criteria,
            retired_criteria: Vec::new(),
            milestones,
            active_milestone_id: None,
            counters: GoalCounters::default(),
            paused_stage: None,
            pause_requested: false,
            blocked_reason: None,
            pending_amendment: None,
            amendments: Vec::new(),
            outcome: None,
            completion_basis: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };
        run.validate()?;
        Ok(run)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != GOAL_RUN_VERSION {
            bail!("unsupported goal run version {}", self.version);
        }
        self.policy.validate()?;
        self.budget.constrained_by(self.policy.limits)?;
        required("goal id", self.id.clone())?;
        required("session id", self.session_id.clone())?;
        required("goal objective", self.objective.clone())?;
        required("goal workdir", self.authority.workdir.clone())?;
        if self.authority.publication {
            bail!("goal publication authority must remain false");
        }
        let active_count = self
            .milestones
            .iter()
            .filter(|milestone| !milestone.status.is_superseded())
            .count();
        if active_count > self.budget.max_milestones {
            bail!("goal contains more active milestones than its budget permits");
        }
        let expected_plan = plan_digest(self.plan_version, &self.objective, &self.milestones)?;
        if self.plan_sha256 != expected_plan {
            bail!("goal plan digest mismatch");
        }
        let mut ids = HashSet::new();
        for milestone in &self.milestones {
            if !ids.insert(&milestone.id) {
                bail!("goal contains duplicate milestone id '{}'", milestone.id);
            }
            if let Some(workflow) = &milestone.workflow {
                workflow.validate()?;
            }
        }
        let mut criterion_ids = HashSet::new();
        for criterion in self.criteria.iter().chain(&self.retired_criteria) {
            if !criterion_ids.insert(criterion.id.as_str()) {
                bail!("goal contains duplicate criterion id '{}'", criterion.id);
            }
        }
        for milestone in &self.milestones {
            if milestone
                .criterion_ids
                .iter()
                .any(|criterion_id| !criterion_ids.contains(criterion_id.as_str()))
            {
                bail!("goal milestone references an unknown criterion");
            }
        }
        if let Some(active) = self.active_milestone_id.as_deref() {
            let milestone = self
                .milestones
                .iter()
                .find(|milestone| milestone.id == active)
                .ok_or_else(|| anyhow::anyhow!("active goal milestone does not exist"))?;
            if milestone.status != GoalMilestoneStatus::Running {
                bail!("active goal milestone is not marked running");
            }
        }
        if self.stage.is_terminal() && self.active_milestone_id.is_some() {
            bail!("terminal goal retains an active milestone");
        }
        if self.stage == GoalStage::Paused && self.paused_stage.is_none() {
            bail!("paused goal does not record its prior stage");
        }
        if self.stage != GoalStage::Paused && self.paused_stage.is_some() {
            bail!("non-paused goal records a paused stage");
        }
        Ok(())
    }

    pub fn plan(&self) -> GoalPlanArtifact {
        GoalPlanArtifact {
            version: self.plan_version,
            objective: self.objective.clone(),
            milestones: self.milestones.clone(),
        }
    }

    pub fn model_brief(&self) -> GoalModelBrief {
        let counters = self.effective_counters();
        GoalModelBrief {
            id: self.id.clone(),
            objective: self.objective.clone(),
            stage: self.stage,
            plan_sha256: self.plan_sha256.clone(),
            budget: self.budget,
            counters,
            current_milestone: self
                .current_milestone()
                .map(|milestone| milestone.title.clone()),
            completed_milestones: self
                .milestones
                .iter()
                .filter(|milestone| milestone.status == GoalMilestoneStatus::Completed)
                .count(),
            total_milestones: self
                .milestones
                .iter()
                .filter(|milestone| !milestone.status.is_superseded())
                .count(),
        }
    }

    pub fn current_milestone(&self) -> Option<&GoalMilestoneRun> {
        self.active_milestone_id
            .as_deref()
            .and_then(|id| self.milestones.iter().find(|milestone| milestone.id == id))
    }

    /// Includes the currently active workflow without mutating the durable completed-work totals.
    /// This keeps UI/API/model status truthful while a milestone is running or paused.
    pub fn effective_counters(&self) -> GoalCounters {
        let mut counters = self.counters.clone();
        if self.active_milestone_id.is_some() {
            counters.workflows = counters.workflows.saturating_add(1);
            if let Some(usage) = self.active_workflow_counters() {
                counters.model_invocations = counters
                    .model_invocations
                    .saturating_add(usage.model_invocations);
                counters.generated_tokens = counters
                    .generated_tokens
                    .saturating_add(usage.generated_tokens);
                counters.advisory_calls =
                    counters.advisory_calls.saturating_add(usage.advisory_calls);
            }
        }
        counters
    }

    pub fn approve_plan(&mut self, expected_plan_sha256: &str, now_ms: u64) -> Result<()> {
        if self.stage != GoalStage::AwaitingPlanApproval || self.pending_amendment.is_some() {
            bail!("goal is not awaiting initial plan approval");
        }
        if self.plan_sha256 != expected_plan_sha256 {
            bail!("goal plan changed before approval");
        }
        self.start_next_milestone(now_ms)
    }

    pub fn revise_initial_plan(
        &mut self,
        objective: impl Into<String>,
        criteria: Vec<GoalCriterionInput>,
        continuation: GoalContinuationPolicy,
        requested_budget: Option<GoalBudget>,
        now_ms: u64,
    ) -> Result<()> {
        if self.stage != GoalStage::AwaitingPlanApproval || self.pending_amendment.is_some() {
            bail!("goal initial draft is no longer editable");
        }
        let objective = required("goal objective", objective.into())?;
        let budget = self.policy.budget(requested_budget)?;
        let criteria = normalize_criteria(criteria, self.plan_version)?;
        if criteria.len() > budget.max_milestones {
            bail!("goal draft exceeds its milestone budget");
        }
        let milestones = milestones_for(&self.id, self.plan_version, &criteria);
        self.objective = objective;
        self.criteria = criteria;
        self.milestones = milestones;
        self.continuation = continuation;
        self.budget = budget;
        self.plan_sha256 = plan_digest(self.plan_version, &self.objective, &self.milestones)?;
        self.updated_at_ms = now_ms;
        Ok(())
    }

    pub fn start_next_milestone(&mut self, now_ms: u64) -> Result<()> {
        if self.active_milestone_id.is_some() {
            bail!("goal already has an active milestone");
        }
        if let Some(reason) = self.exhausted_budget_reason(now_ms) {
            self.fail(GoalOutcome::BudgetExhausted, &reason, now_ms);
            return Ok(());
        }
        let Some(milestone) = self
            .milestones
            .iter_mut()
            .find(|milestone| milestone.status == GoalMilestoneStatus::Pending)
        else {
            self.evaluate_completion(now_ms);
            return Ok(());
        };
        milestone.status = GoalMilestoneStatus::Running;
        milestone.attempts = milestone.attempts.saturating_add(1);
        self.active_milestone_id = Some(milestone.id.clone());
        self.stage = GoalStage::RunningMilestone;
        self.paused_stage = None;
        self.pause_requested = false;
        self.blocked_reason = None;
        self.updated_at_ms = now_ms;
        Ok(())
    }

    pub fn checkpoint_active_workflow(
        &mut self,
        checkpoint: crate::workflow::WorkflowCheckpoint,
        now_ms: u64,
    ) -> Result<()> {
        checkpoint.validate()?;
        self.active_milestone_mut()?.workflow = Some(checkpoint);
        self.updated_at_ms = now_ms;
        Ok(())
    }

    pub fn block_active_workflow(
        &mut self,
        checkpoint: crate::workflow::WorkflowCheckpoint,
        reason: impl Into<String>,
        now_ms: u64,
    ) -> Result<()> {
        checkpoint.validate()?;
        self.active_milestone_mut()?.workflow = Some(checkpoint);
        self.stage = GoalStage::Blocked;
        self.blocked_reason = Some(reason.into());
        self.updated_at_ms = now_ms;
        Ok(())
    }

    pub fn finish_active_workflow(
        &mut self,
        checkpoint: crate::workflow::WorkflowCheckpoint,
        now_ms: u64,
    ) -> Result<()> {
        checkpoint.validate()?;
        let summary = crate::workflow::WorkflowSummary::from(&checkpoint.run);
        let workflow_outcome = checkpoint.run.outcome;
        let usage = checkpoint.run.counters.clone();
        let completed_id = self
            .active_milestone_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("goal has no active milestone"))?;
        let criterion_ids = {
            let milestone = self.active_milestone_mut()?;
            milestone.workflow = Some(checkpoint);
            milestone.workflow_summary = Some(summary);
            milestone.status = if matches!(
                workflow_outcome,
                Some(
                    crate::workflow::WorkflowOutcome::Ready
                        | crate::workflow::WorkflowOutcome::NoChange
                )
            ) {
                GoalMilestoneStatus::Completed
            } else {
                GoalMilestoneStatus::Failed
            };
            milestone.criterion_ids.clone()
        };
        self.counters.workflows = self.counters.workflows.saturating_add(1);
        self.counters.model_invocations = self
            .counters
            .model_invocations
            .saturating_add(usage.model_invocations);
        self.counters.generated_tokens = self
            .counters
            .generated_tokens
            .saturating_add(usage.generated_tokens);
        self.counters.advisory_calls = self
            .counters
            .advisory_calls
            .saturating_add(usage.advisory_calls);
        self.active_milestone_id = None;
        if matches!(
            workflow_outcome,
            Some(
                crate::workflow::WorkflowOutcome::Ready
                    | crate::workflow::WorkflowOutcome::NoChange
            )
        ) {
            for criterion in &mut self.criteria {
                if criterion_ids.contains(&criterion.id) {
                    criterion
                        .evidence_ids
                        .push(format!("workflow:{completed_id}"));
                    criterion.status = match criterion.verifier {
                        GoalVerifier::WorkflowReady => GoalCriterionStatus::MachineVerified,
                        GoalVerifier::ReviewRequired | GoalVerifier::UserConfirmation => {
                            GoalCriterionStatus::EvidenceReady
                        }
                    };
                }
            }
            self.stage = GoalStage::Evaluating;
            self.updated_at_ms = now_ms;
            if self
                .milestones
                .iter()
                .any(|milestone| milestone.status == GoalMilestoneStatus::Pending)
            {
                if self.continuation.automatically_continues() && !self.pause_requested {
                    self.start_next_milestone(now_ms)?;
                } else {
                    self.paused_stage = Some(GoalStage::RunningMilestone);
                    self.stage = GoalStage::Paused;
                    self.pause_requested = false;
                }
            } else {
                self.evaluate_completion(now_ms);
            }
        } else {
            self.fail(
                map_workflow_failure(workflow_outcome),
                "strict milestone workflow did not reach Ready",
                now_ms,
            );
        }
        Ok(())
    }

    pub fn request_pause(&mut self, now_ms: u64) -> Result<bool> {
        if self.stage.is_terminal() {
            bail!("terminal goal cannot be paused");
        }
        self.updated_at_ms = now_ms;
        if matches!(
            self.stage,
            GoalStage::Paused
                | GoalStage::Blocked
                | GoalStage::AwaitingPlanApproval
                | GoalStage::AwaitingUserReview
        ) {
            self.pause_requested = false;
            return Ok(true);
        }
        self.pause_requested = true;
        if self.stage != GoalStage::RunningMilestone {
            let previous = self.stage;
            self.stage = GoalStage::Paused;
            self.paused_stage = Some(previous);
            return Ok(true);
        }
        Ok(false)
    }

    pub fn pause_at_boundary(&mut self, now_ms: u64) -> Result<()> {
        if self.stage.is_terminal() {
            bail!("terminal goal cannot be paused");
        }
        if self.stage != GoalStage::Paused {
            let previous = self.stage;
            self.stage = GoalStage::Paused;
            self.paused_stage = Some(previous);
        }
        self.pause_requested = false;
        self.updated_at_ms = now_ms;
        Ok(())
    }

    pub fn resume(&mut self, now_ms: u64) -> Result<()> {
        if let Some(reason) = self.exhausted_budget_reason(now_ms) {
            bail!("goal cannot resume because {reason}");
        }
        if self.stage == GoalStage::Blocked {
            self.stage = GoalStage::RunningMilestone;
            self.blocked_reason = None;
            self.pause_requested = false;
            self.updated_at_ms = now_ms;
            return Ok(());
        }
        if self.stage != GoalStage::Paused {
            bail!("goal is not paused or blocked");
        }
        let previous = self
            .paused_stage
            .take()
            .unwrap_or(GoalStage::RunningMilestone);
        self.stage = previous;
        self.pause_requested = false;
        self.blocked_reason = None;
        if self.active_milestone_id.is_none() && self.stage == GoalStage::RunningMilestone {
            self.start_next_milestone(now_ms)?;
        }
        self.updated_at_ms = now_ms;
        Ok(())
    }

    pub fn accept(
        &mut self,
        expected_goal_sha256: &str,
        current_sha256: &str,
        now_ms: u64,
    ) -> Result<()> {
        if self.stage != GoalStage::AwaitingUserReview {
            bail!("goal is not ready for user acceptance");
        }
        if expected_goal_sha256 != current_sha256 {
            bail!("goal evidence changed before acceptance");
        }
        for criterion in &mut self.criteria {
            if criterion.status == GoalCriterionStatus::EvidenceReady {
                criterion.status = GoalCriterionStatus::UserAccepted;
            }
        }
        self.stage = GoalStage::Completed;
        self.outcome = Some(GoalOutcome::Complete);
        self.completion_basis = Some(GoalCompletionBasis::UserAccepted);
        self.active_milestone_id = None;
        self.updated_at_ms = now_ms;
        Ok(())
    }

    pub fn cancel(&mut self, now_ms: u64) {
        self.absorb_active_workflow_usage();
        self.mark_active_milestone_failed();
        self.stage = GoalStage::Cancelled;
        self.outcome = Some(GoalOutcome::Cancelled);
        self.active_milestone_id = None;
        self.paused_stage = None;
        self.pause_requested = false;
        self.updated_at_ms = now_ms;
    }

    pub fn fail_external(&mut self, outcome: GoalOutcome, reason: impl Into<String>, now_ms: u64) {
        self.fail(outcome, &reason.into(), now_ms);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn propose_amendment(
        &mut self,
        amendment_id: impl Into<String>,
        base_goal_sha256: impl Into<String>,
        objective: impl Into<String>,
        criteria: Vec<GoalCriterionInput>,
        continuation: GoalContinuationPolicy,
        requested_budget: Option<GoalBudget>,
        now_ms: u64,
    ) -> Result<()> {
        if self.stage != GoalStage::Paused && self.stage != GoalStage::AwaitingUserReview {
            bail!("goal must be paused or ready for review before it can be amended");
        }
        let amendment_id = required("amendment id", amendment_id.into())?;
        let base_goal_sha256 = required("base goal digest", base_goal_sha256.into())?;
        let objective = required("goal objective", objective.into())?;
        let budget = self.policy.budget(requested_budget)?;
        let resume_stage = if self.stage == GoalStage::Paused {
            self.paused_stage.unwrap_or(GoalStage::RunningMilestone)
        } else {
            self.stage
        };
        let version = self.plan_version.saturating_add(1);
        let normalized = normalize_criteria(criteria, version)?;
        let replacement_milestones = milestones_for(&self.id, version, &normalized);
        if replacement_milestones.len() > budget.max_milestones {
            bail!("amended goal exceeds its milestone budget");
        }
        let replacement_plan_sha256 = plan_digest(version, &objective, &replacement_milestones)?;
        self.pending_amendment = Some(GoalAmendmentDraft {
            id: amendment_id,
            base_goal_sha256,
            objective,
            criteria: normalized
                .iter()
                .map(|criterion| GoalCriterionInput {
                    text: criterion.text.clone(),
                    verifier: criterion.verifier,
                })
                .collect(),
            continuation,
            budget,
            replacement_milestones,
            replacement_plan_sha256,
            resume_stage,
        });
        self.stage = GoalStage::AwaitingPlanApproval;
        self.paused_stage = None;
        self.updated_at_ms = now_ms;
        Ok(())
    }

    pub fn approve_amendment(&mut self, expected_plan_sha256: &str, now_ms: u64) -> Result<()> {
        let amendment = self
            .pending_amendment
            .take()
            .ok_or_else(|| anyhow::anyhow!("goal has no pending amendment"))?;
        if amendment.replacement_plan_sha256 != expected_plan_sha256 {
            self.pending_amendment = Some(amendment);
            bail!("amended goal plan changed before approval");
        }
        for milestone in &mut self.milestones {
            if !milestone.status.is_completed() {
                milestone.status = GoalMilestoneStatus::Superseded;
            }
        }
        self.objective = amendment.objective;
        self.continuation = amendment.continuation;
        self.budget = amendment.budget;
        self.plan_version = self.plan_version.saturating_add(1);
        self.milestones.extend(amendment.replacement_milestones);
        let previous_criteria = self.criteria.clone();
        self.retired_criteria.append(&mut self.criteria);
        self.criteria = normalize_criteria(amendment.criteria, self.plan_version)?;
        for criterion in &mut self.criteria {
            if let Some(previous) = previous_criteria.iter().find(|previous| {
                previous.text.eq_ignore_ascii_case(&criterion.text)
                    && previous.verifier == criterion.verifier
                    && !matches!(
                        previous.status,
                        GoalCriterionStatus::Pending | GoalCriterionStatus::Unsatisfied
                    )
            }) {
                criterion.status = previous.status;
                criterion.evidence_ids = previous.evidence_ids.clone();
                criterion.evidence_ids.push(format!(
                    "carry-forward:v{}:{}",
                    self.plan_version.saturating_sub(1),
                    previous.id
                ));
            }
        }
        for milestone in self
            .milestones
            .iter_mut()
            .filter(|milestone| milestone.plan_version == self.plan_version)
        {
            if milestone.criterion_ids.iter().all(|criterion_id| {
                self.criteria.iter().any(|criterion| {
                    &criterion.id == criterion_id
                        && !matches!(
                            criterion.status,
                            GoalCriterionStatus::Pending | GoalCriterionStatus::Unsatisfied
                        )
                })
            }) {
                milestone.status = GoalMilestoneStatus::Completed;
            }
        }
        self.plan_sha256 = plan_digest(self.plan_version, &self.objective, &self.milestones)?;
        self.active_milestone_id = None;
        self.amendments.push(GoalAmendmentRecord {
            id: amendment.id,
            base_goal_sha256: amendment.base_goal_sha256,
            plan_sha256: self.plan_sha256.clone(),
            accepted: true,
        });
        self.start_next_milestone(now_ms)
    }

    pub fn discard_amendment(&mut self, now_ms: u64) -> Result<()> {
        let amendment = self
            .pending_amendment
            .take()
            .ok_or_else(|| anyhow::anyhow!("goal has no pending amendment"))?;
        self.amendments.push(GoalAmendmentRecord {
            id: amendment.id,
            base_goal_sha256: amendment.base_goal_sha256,
            plan_sha256: amendment.replacement_plan_sha256,
            accepted: false,
        });
        self.stage = GoalStage::Paused;
        self.paused_stage = Some(amendment.resume_stage);
        self.updated_at_ms = now_ms;
        Ok(())
    }

    fn active_milestone_mut(&mut self) -> Result<&mut GoalMilestoneRun> {
        let id = self
            .active_milestone_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("goal has no active milestone"))?;
        self.milestones
            .iter_mut()
            .find(|milestone| milestone.id == id)
            .ok_or_else(|| anyhow::anyhow!("active goal milestone does not exist"))
    }

    fn active_workflow_counters(&self) -> Option<&crate::workflow::WorkflowCounters> {
        self.current_milestone()
            .and_then(|milestone| milestone.workflow.as_ref())
            .map(|checkpoint| &checkpoint.run.counters)
    }

    fn absorb_active_workflow_usage(&mut self) {
        if self.active_milestone_id.is_none() {
            return;
        }
        let usage = self.active_workflow_counters().cloned();
        self.counters.workflows = self.counters.workflows.saturating_add(1);
        if let Some(usage) = usage {
            self.counters.model_invocations = self
                .counters
                .model_invocations
                .saturating_add(usage.model_invocations);
            self.counters.generated_tokens = self
                .counters
                .generated_tokens
                .saturating_add(usage.generated_tokens);
            self.counters.advisory_calls = self
                .counters
                .advisory_calls
                .saturating_add(usage.advisory_calls);
        }
    }

    pub fn wall_time_exhausted(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.created_at_ms)
            >= self.budget.wall_time_minutes.saturating_mul(60_000)
    }

    fn exhausted_budget_reason(&self, now_ms: u64) -> Option<String> {
        if self.counters.workflows >= self.budget.max_workflows {
            Some("workflow budget is exhausted".to_string())
        } else if self.counters.model_invocations >= self.budget.total_model_invocations {
            Some("model invocation budget is exhausted".to_string())
        } else if self.counters.generated_tokens >= self.budget.total_generated_tokens {
            Some("generated-token budget is exhausted".to_string())
        } else if self.wall_time_exhausted(now_ms) {
            Some("wall-time budget is exhausted".to_string())
        } else {
            None
        }
    }

    fn evaluate_completion(&mut self, now_ms: u64) {
        self.active_milestone_id = None;
        if self.criteria.iter().all(|criterion| {
            matches!(
                criterion.status,
                GoalCriterionStatus::MachineVerified | GoalCriterionStatus::UserAccepted
            )
        }) {
            self.stage = GoalStage::Completed;
            self.outcome = Some(GoalOutcome::Complete);
            self.completion_basis = Some(
                if self
                    .criteria
                    .iter()
                    .any(|criterion| criterion.status == GoalCriterionStatus::UserAccepted)
                {
                    GoalCompletionBasis::UserAccepted
                } else {
                    GoalCompletionBasis::MachineVerified
                },
            );
        } else if self.criteria.iter().all(|criterion| {
            matches!(
                criterion.status,
                GoalCriterionStatus::MachineVerified | GoalCriterionStatus::EvidenceReady
            )
        }) {
            self.stage = GoalStage::AwaitingUserReview;
        } else {
            self.fail(
                GoalOutcome::CriteriaUnsatisfied,
                "one or more goal criteria have no current evidence",
                now_ms,
            );
        }
        self.updated_at_ms = now_ms;
    }

    fn fail(&mut self, outcome: GoalOutcome, reason: &str, now_ms: u64) {
        self.absorb_active_workflow_usage();
        self.mark_active_milestone_failed();
        self.stage = GoalStage::Failed;
        self.outcome = Some(outcome);
        self.blocked_reason = Some(reason.to_string());
        self.active_milestone_id = None;
        self.paused_stage = None;
        self.pause_requested = false;
        self.updated_at_ms = now_ms;
    }

    fn mark_active_milestone_failed(&mut self) {
        if let Some(active) = self.active_milestone_id.as_deref()
            && let Some(milestone) = self
                .milestones
                .iter_mut()
                .find(|milestone| milestone.id == active)
            && milestone.status == GoalMilestoneStatus::Running
        {
            milestone.status = GoalMilestoneStatus::Failed;
        }
    }
}

const fn default_amendment_resume_stage() -> GoalStage {
    GoalStage::RunningMilestone
}

fn normalize_criteria(
    inputs: Vec<GoalCriterionInput>,
    plan_version: u32,
) -> Result<Vec<GoalCriterionState>> {
    let inputs = if inputs.is_empty() {
        vec![GoalCriterionInput {
            text: "Review the completed implementation against the stated objective".to_string(),
            verifier: GoalVerifier::ReviewRequired,
        }]
    } else {
        inputs
    };
    let mut seen = HashSet::new();
    let mut criteria = Vec::with_capacity(inputs.len());
    for (index, input) in inputs.into_iter().enumerate() {
        let text = required("goal criterion", input.text)?;
        if !seen.insert(text.to_lowercase()) {
            bail!("goal contains a duplicate completion criterion");
        }
        criteria.push(GoalCriterionState {
            id: format!("v{plan_version}-criterion-{}", index + 1),
            text,
            verifier: input.verifier,
            status: GoalCriterionStatus::Pending,
            evidence_ids: Vec::new(),
        });
    }
    Ok(criteria)
}

fn milestones_for(
    goal_id: &str,
    plan_version: u32,
    criteria: &[GoalCriterionState],
) -> Vec<GoalMilestoneRun> {
    criteria
        .iter()
        .enumerate()
        .map(|(index, criterion)| GoalMilestoneRun {
            id: format!("{goal_id}-v{plan_version}-milestone-{}", index + 1),
            title: compact_title(&criterion.text, index + 1),
            description: format!(
                "Deliver the bounded repository change needed to satisfy: {}",
                criterion.text
            ),
            criterion_ids: vec![criterion.id.clone()],
            plan_version,
            status: GoalMilestoneStatus::Pending,
            attempts: 0,
            workflow: None,
            workflow_summary: None,
        })
        .collect()
}

fn compact_title(text: &str, index: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let title = compact.chars().take(80).collect::<String>();
    if title.is_empty() {
        format!("Milestone {index}")
    } else {
        title
    }
}

#[derive(Serialize)]
struct PlanDigestMilestone<'a> {
    id: &'a str,
    title: &'a str,
    description: &'a str,
    criterion_ids: &'a [String],
    plan_version: u32,
}

fn plan_digest(version: u32, objective: &str, milestones: &[GoalMilestoneRun]) -> Result<String> {
    let stable = milestones
        .iter()
        .filter(|milestone| milestone.plan_version == version)
        .map(|milestone| PlanDigestMilestone {
            id: &milestone.id,
            title: &milestone.title,
            description: &milestone.description,
            criterion_ids: &milestone.criterion_ids,
            plan_version: milestone.plan_version,
        })
        .collect::<Vec<_>>();
    let bytes = serde_json::to_vec(&(version, objective, stable))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn required(label: &str, value: String) -> Result<String> {
    let value = value.trim().to_string();
    if value.is_empty() {
        bail!("{label} must not be empty");
    }
    Ok(value)
}

fn map_workflow_failure(outcome: Option<crate::workflow::WorkflowOutcome>) -> GoalOutcome {
    match outcome {
        Some(
            crate::workflow::WorkflowOutcome::StepLimit
            | crate::workflow::WorkflowOutcome::InvocationLimit
            | crate::workflow::WorkflowOutcome::TokenLimit,
        ) => GoalOutcome::BudgetExhausted,
        Some(crate::workflow::WorkflowOutcome::ContextLimit) => GoalOutcome::ContextLimit,
        Some(crate::workflow::WorkflowOutcome::Cancelled) => GoalOutcome::Cancelled,
        Some(
            crate::workflow::WorkflowOutcome::ChecksFailed
            | crate::workflow::WorkflowOutcome::ReviewFailed
            | crate::workflow::WorkflowOutcome::RepairCyclesExhausted,
        ) => GoalOutcome::MilestoneFailed,
        _ => GoalOutcome::EngineError,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::{GoalCheckpoint, GoalConfigDocument};

    fn run() -> GoalRun {
        GoalRun::start(
            "goal-1",
            "session-1",
            "Ship goal mode",
            vec![GoalCriterionInput {
                text: "Persist goal state".to_string(),
                verifier: GoalVerifier::ReviewRequired,
            }],
            GoalContinuationPolicy::ReviewPlanThenAutomatic,
            None,
            GoalConfigDocument::default().compile().unwrap(),
            "/tmp/project",
            1,
        )
        .unwrap()
    }

    #[test]
    fn goal_requires_exact_plan_digest_and_has_one_active_milestone() {
        let mut run = run();
        assert!(run.approve_plan("stale", 2).is_err());
        let digest = run.plan_sha256.clone();
        run.approve_plan(&digest, 2).unwrap();
        assert_eq!(run.stage, GoalStage::RunningMilestone);
        assert!(run.current_milestone().is_some());
        assert!(run.start_next_milestone(3).is_err());
    }

    #[test]
    fn checkpoint_detects_goal_tampering() {
        let mut checkpoint = GoalCheckpoint::new(run()).unwrap();
        checkpoint.run.objective = "tampered".to_string();
        assert!(checkpoint.validate().is_err());
    }

    #[test]
    fn pause_and_amendment_preserve_old_plan_until_approval() {
        let mut run = run();
        let initial = run.plan_sha256.clone();
        run.approve_plan(&initial, 2).unwrap();
        assert!(!run.request_pause(3).unwrap());
        run.pause_at_boundary(4).unwrap();
        let old_plan = run.plan_sha256.clone();
        run.propose_amendment(
            "amendment-1",
            "base",
            "Ship smaller goal mode",
            vec![GoalCriterionInput {
                text: "Render goal state".to_string(),
                verifier: GoalVerifier::ReviewRequired,
            }],
            GoalContinuationPolicy::ManualMilestones,
            None,
            5,
        )
        .unwrap();
        assert_eq!(run.plan_sha256, old_plan);
        let replacement = run
            .pending_amendment
            .as_ref()
            .unwrap()
            .replacement_plan_sha256
            .clone();
        run.approve_amendment(&replacement, 6).unwrap();
        assert_ne!(run.plan_sha256, old_plan);
        assert_eq!(run.stage, GoalStage::RunningMilestone);
        assert_eq!(run.retired_criteria.len(), 1);
        assert_eq!(run.retired_criteria[0].text, "Persist goal state");
        assert_ne!(run.retired_criteria[0].id, run.criteria[0].id);
    }

    #[test]
    fn amendment_carries_unchanged_current_evidence_without_rerunning_its_milestone() {
        let mut run = run();
        let plan = run.plan_sha256.clone();
        run.approve_plan(&plan, 2).unwrap();
        run.criteria[0].status = GoalCriterionStatus::EvidenceReady;
        run.criteria[0].evidence_ids = vec!["workflow:old".to_string()];
        run.milestones[0].status = GoalMilestoneStatus::Completed;
        run.active_milestone_id = None;
        run.stage = GoalStage::AwaitingUserReview;
        run.propose_amendment(
            "amendment-carry",
            "base",
            "Ship goal mode with clearer copy",
            vec![GoalCriterionInput {
                text: "Persist goal state".to_string(),
                verifier: GoalVerifier::ReviewRequired,
            }],
            GoalContinuationPolicy::ReviewPlanThenAutomatic,
            None,
            3,
        )
        .unwrap();
        let replacement = run
            .pending_amendment
            .as_ref()
            .unwrap()
            .replacement_plan_sha256
            .clone();
        run.approve_amendment(&replacement, 4).unwrap();

        assert_eq!(run.stage, GoalStage::AwaitingUserReview);
        assert_eq!(run.criteria[0].status, GoalCriterionStatus::EvidenceReady);
        assert!(
            run.criteria[0]
                .evidence_ids
                .iter()
                .any(|evidence| evidence.starts_with("carry-forward:"))
        );
        assert_eq!(
            run.milestones
                .iter()
                .find(|milestone| milestone.plan_version == 2)
                .unwrap()
                .status,
            GoalMilestoneStatus::Completed
        );
    }

    #[test]
    fn active_and_cancelled_workflow_attempts_are_counted_once() {
        let mut run = run();
        let plan = run.plan_sha256.clone();
        run.approve_plan(&plan, 2).unwrap();

        assert_eq!(run.counters.workflows, 0);
        assert_eq!(run.effective_counters().workflows, 1);
        run.cancel(3);
        assert_eq!(run.counters.workflows, 1);
        assert_eq!(run.effective_counters().workflows, 1);
    }

    #[test]
    fn model_brief_validation_rejects_impossible_controller_state() {
        let mut run = run();
        let plan = run.plan_sha256.clone();
        run.approve_plan(&plan, 2).unwrap();
        let brief = run.model_brief();
        brief.validate().unwrap();

        let mut invalid = brief.clone();
        invalid.counters.model_invocations = invalid.budget.total_model_invocations + 1;
        assert!(invalid.validate().is_err());

        let mut invalid = brief;
        invalid.current_milestone = None;
        assert!(invalid.validate().is_err());
    }
}
