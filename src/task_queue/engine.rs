use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::goal::{GoalCheckpoint, GoalOutcome, GoalStage};
use crate::workflow::{ArtifactEnvelope, WorkflowCheckpoint, WorkflowOutcome, WorkflowStage};
use crate::workspace::WorkspaceBaseline;

use super::{
    CompiledTaskPolicy, MultiTaskBudget, TaskAcceptance, TaskBudget, TaskKind, TaskPlanArtifact,
    TaskPlanAuthority, TaskPlanReviewArtifact, TaskPlanReviewVerdict, TaskPlannerQualification,
    TaskRequirement, TaskSourceIntent, TaskSpec,
};

pub const MULTI_TASK_RUN_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MultiTaskStage {
    RunningTask,
    Evaluating,
    Paused,
    Blocked,
    Ready,
    Failed,
    Cancelled,
}

impl MultiTaskStage {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Ready | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MultiTaskOutcome {
    Ready,
    TaskBlocked,
    TaskFailed,
    BudgetExhausted,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Queued,
    Running,
    Committed,
    NoChange,
    Blocked,
    Failed,
    Cancelled,
    Superseded,
}

impl TaskState {
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Committed | Self::NoChange)
    }

    const fn is_current(self) -> bool {
        !matches!(self, Self::Superseded)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskAuthorityEnvelope {
    pub workdir: String,
    pub publication: bool,
    pub request_max_steps: usize,
    pub plan: TaskPlanAuthority,
    pub qualification: TaskPlannerQualification,
    pub workflow_policy: crate::workflow::CompiledWorkflowPolicy,
    pub goal_policy: crate::goal::CompiledGoalPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskRepositoryState {
    pub head: String,
    pub index_sha256: String,
    pub refs_sha256: String,
    pub content_sha256: String,
}

impl TaskRepositoryState {
    pub fn capture(repo_root: &Path) -> Result<Self> {
        let baseline = WorkspaceBaseline::capture(repo_root)?;
        let index = git_bytes(repo_root, &["ls-files", "--stage", "-z"])?;
        let refs = git_bytes(
            repo_root,
            &[
                "for-each-ref",
                "--format=%(refname)%00%(objectname)%00",
                "refs/heads",
                "refs/tags",
                "refs/remotes",
            ],
        )?;
        let state = Self {
            head: baseline.head.unwrap_or_else(|| "<unborn>".to_string()),
            index_sha256: format!("{:x}", Sha256::digest(index)),
            refs_sha256: format!("{:x}", Sha256::digest(refs)),
            content_sha256: baseline.content.fingerprint,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<()> {
        required("Task repository HEAD", &self.head)?;
        if self.head != "<unborn>" {
            validate_git_oid("Task repository HEAD", &self.head)?;
        }
        validate_sha256("Task repository index", &self.index_sha256)?;
        validate_sha256("Task repository refs", &self.refs_sha256)?;
        validate_sha256("Task repository content", &self.content_sha256)
    }

    pub fn difference(&self, observed: &Self) -> String {
        let mut changed = Vec::new();
        if self.head != observed.head {
            changed.push("HEAD");
        }
        if self.index_sha256 != observed.index_sha256 {
            changed.push("index");
        }
        if self.refs_sha256 != observed.refs_sha256 {
            changed.push("refs");
        }
        if self.content_sha256 != observed.content_sha256 {
            changed.push("content");
        }
        changed.join(", ")
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskCounters {
    pub workflows: usize,
    pub stage_steps: usize,
    pub model_invocations: usize,
    pub generated_tokens: usize,
    pub advisory_calls: usize,
    pub plan_cycles: usize,
    pub repair_cycles: usize,
    pub elapsed_ms: u64,
}

impl TaskCounters {
    fn checked_add(self, other: Self) -> Result<Self> {
        Ok(Self {
            workflows: checked_add("Task workflows", self.workflows, other.workflows)?,
            stage_steps: checked_add("Task stage steps", self.stage_steps, other.stage_steps)?,
            model_invocations: checked_add(
                "Task model invocations",
                self.model_invocations,
                other.model_invocations,
            )?,
            generated_tokens: checked_add(
                "Task generated tokens",
                self.generated_tokens,
                other.generated_tokens,
            )?,
            advisory_calls: checked_add(
                "Task advisory calls",
                self.advisory_calls,
                other.advisory_calls,
            )?,
            plan_cycles: checked_add("Task plan cycles", self.plan_cycles, other.plan_cycles)?,
            repair_cycles: checked_add(
                "Task repair cycles",
                self.repair_cycles,
                other.repair_cycles,
            )?,
            elapsed_ms: self
                .elapsed_ms
                .checked_add(other.elapsed_ms)
                .context("Task elapsed time overflow")?,
        })
    }

    fn is_monotonic_from(&self, prior: &Self) -> bool {
        self.workflows >= prior.workflows
            && self.stage_steps >= prior.stage_steps
            && self.model_invocations >= prior.model_invocations
            && self.generated_tokens >= prior.generated_tokens
            && self.advisory_calls >= prior.advisory_calls
            && self.plan_cycles >= prior.plan_cycles
            && self.repair_cycles >= prior.repair_cycles
            && self.elapsed_ms >= prior.elapsed_ms
    }

    pub fn fits_within(&self, budget: &TaskBudget) -> bool {
        self.workflows <= budget.max_workflows
            && self.stage_steps <= budget.stage_steps
            && self.model_invocations <= budget.total_model_invocations
            && self.generated_tokens <= budget.total_generated_tokens
            && self.advisory_calls <= budget.advisory_calls
            && self.plan_cycles <= budget.plan_cycles
            && self.repair_cycles <= budget.repair_cycles
            && self.elapsed_ms <= budget.wall_time_minutes.saturating_mul(60_000)
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskCoordinationCounters {
    pub planning_attempts: usize,
    pub model_invocations: usize,
    pub generated_tokens: usize,
    pub advisory_calls: usize,
    pub elapsed_ms: u64,
}

impl TaskCoordinationCounters {
    fn checked_add(self, other: Self) -> Result<Self> {
        Ok(Self {
            planning_attempts: checked_add(
                "Task planning attempts",
                self.planning_attempts,
                other.planning_attempts,
            )?,
            model_invocations: checked_add(
                "Task coordination model invocations",
                self.model_invocations,
                other.model_invocations,
            )?,
            generated_tokens: checked_add(
                "Task coordination generated tokens",
                self.generated_tokens,
                other.generated_tokens,
            )?,
            advisory_calls: checked_add(
                "Task coordination advisory calls",
                self.advisory_calls,
                other.advisory_calls,
            )?,
            elapsed_ms: self
                .elapsed_ms
                .checked_add(other.elapsed_ms)
                .context("Task coordination elapsed time overflow")?,
        })
    }

    pub fn fits_within(&self, budget: &super::TaskCoordinationBudget) -> bool {
        self.planning_attempts <= budget.planning_attempts
            && self.model_invocations <= budget.model_invocations
            && self.generated_tokens <= budget.generated_tokens
            && self.advisory_calls <= budget.advisory_calls
            && self.elapsed_ms <= budget.wall_time_minutes.saturating_mul(60_000)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MultiTaskCounters {
    pub tasks: TaskCounters,
    pub coordination: TaskCoordinationCounters,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskDirectUsage {
    pub model_invocations: usize,
    pub generated_tokens: usize,
    pub advisory_calls: usize,
    pub elapsed_ms: u64,
}

impl From<TaskDirectUsage> for TaskCounters {
    fn from(value: TaskDirectUsage) -> Self {
        Self {
            model_invocations: value.model_invocations,
            generated_tokens: value.generated_tokens,
            advisory_calls: value.advisory_calls,
            elapsed_ms: value.elapsed_ms,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskRequest {
    pub id: String,
    pub multi_task_id: String,
    pub source_turn_id: String,
    pub task_id: String,
    pub child_id: String,
    pub workdir: String,
    pub kind: TaskKind,
    pub title: String,
    pub objective: String,
    pub requirements: Vec<TaskRequirement>,
    pub acceptance: Vec<TaskAcceptance>,
    pub scope_hints: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_contract: Option<super::TaskGoalContract>,
    pub base_repository: TaskRepositoryState,
    pub budget: TaskBudget,
    pub attempt: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskResult {
    pub base_repository: TaskRepositoryState,
    pub terminal_repository: TaskRepositoryState,
    #[serde(default)]
    pub commits: Vec<String>,
    pub no_change: bool,
    pub satisfied_acceptance_ids: Vec<String>,
    pub evidence_refs: Vec<String>,
    pub summary: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskChildKind {
    Build,
    Goal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskChildWatermark {
    pub kind: TaskChildKind,
    pub child_id: String,
    pub checkpoint_sha256: String,
    pub usage: TaskCounters,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskRun {
    pub spec: TaskSpec,
    pub revision: u32,
    pub state: TaskState,
    pub attempts: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<TaskRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow: Option<WorkflowCheckpoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal: Option<GoalCheckpoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_contract_revision: Option<super::TaskGoalContract>,
    #[serde(default)]
    pub direct_counters: TaskCounters,
    #[serde(default)]
    pub child_watermarks: BTreeMap<String, TaskChildWatermark>,
    #[serde(default)]
    pub counters: TaskCounters,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository_checkpoint: Option<TaskRepositoryState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<TaskResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_reason: Option<String>,
}

impl TaskRun {
    fn queued(spec: TaskSpec, revision: u32) -> Self {
        Self {
            spec,
            revision,
            state: TaskState::Queued,
            attempts: 0,
            request: None,
            workflow: None,
            goal: None,
            goal_contract_revision: None,
            direct_counters: TaskCounters::default(),
            child_watermarks: BTreeMap::new(),
            counters: TaskCounters::default(),
            repository_checkpoint: None,
            result: None,
            blocked_reason: None,
        }
    }

    fn recompute_counters(&mut self) -> Result<()> {
        self.counters = self
            .child_watermarks
            .values()
            .try_fold(self.direct_counters, |total, watermark| {
                total.checked_add(watermark.usage)
            })?;
        Ok(())
    }

    fn effective_goal_contract(&self) -> Option<&super::TaskGoalContract> {
        self.goal_contract_revision
            .as_ref()
            .or(self.spec.goal_contract.as_ref())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MultiTaskRun {
    pub version: u32,
    pub id: String,
    pub session_id: String,
    pub source_turn_id: String,
    pub stage: MultiTaskStage,
    pub plan_revision: u32,
    pub plan: ArtifactEnvelope<TaskPlanArtifact>,
    pub plan_review: ArtifactEnvelope<TaskPlanReviewArtifact>,
    pub policy: CompiledTaskPolicy,
    pub policy_sha256: String,
    pub authority: TaskAuthorityEnvelope,
    pub budget: MultiTaskBudget,
    #[serde(default)]
    pub counters: MultiTaskCounters,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_task_id: Option<String>,
    pub tasks: Vec<TaskRun>,
    pub expected_repository: TaskRepositoryState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<MultiTaskOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "checkpoint", rename_all = "snake_case")]
pub enum TaskChildCheckpoint {
    Build(WorkflowCheckpoint),
    Goal(GoalCheckpoint),
}

impl TaskChildCheckpoint {
    fn id(&self) -> &str {
        match self {
            Self::Build(checkpoint) => &checkpoint.run.id,
            Self::Goal(checkpoint) => &checkpoint.run.id,
        }
    }

    fn sha256(&self) -> &str {
        match self {
            Self::Build(checkpoint) => &checkpoint.sha256,
            Self::Goal(checkpoint) => &checkpoint.sha256,
        }
    }

    fn kind(&self) -> TaskChildKind {
        match self {
            Self::Build(_) => TaskChildKind::Build,
            Self::Goal(_) => TaskChildKind::Goal,
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::Build(checkpoint) => checkpoint.validate(),
            Self::Goal(checkpoint) => checkpoint.validate(),
        }
    }

    fn usage(&self) -> Result<TaskCounters> {
        match self {
            Self::Build(checkpoint) => Ok(workflow_usage(checkpoint)),
            Self::Goal(checkpoint) => goal_usage(checkpoint),
        }
    }

    fn is_initial(&self) -> bool {
        match self {
            Self::Build(checkpoint) => checkpoint.run.stage == WorkflowStage::Planning,
            Self::Goal(checkpoint) => checkpoint.run.stage == GoalStage::AwaitingPlanApproval,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStopDisposition {
    Blocked,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MultiTaskEvent {
    CoordinationUsageRecorded {
        usage: TaskCoordinationCounters,
        now_ms: u64,
    },
    DirectUsageRecorded {
        task_id: String,
        usage: TaskDirectUsage,
        now_ms: u64,
    },
    ChildStarted {
        task_id: String,
        child: TaskChildCheckpoint,
        repository: TaskRepositoryState,
        now_ms: u64,
    },
    ChildCheckpointed {
        task_id: String,
        child: TaskChildCheckpoint,
        repository: TaskRepositoryState,
        now_ms: u64,
    },
    GoalContractRevised {
        task_id: String,
        child: GoalCheckpoint,
        repository: TaskRepositoryState,
        now_ms: u64,
    },
    TaskDelivered {
        task_id: String,
        result: TaskResult,
        repository: TaskRepositoryState,
        now_ms: u64,
    },
    TaskStopped {
        task_id: String,
        disposition: TaskStopDisposition,
        reason: String,
        now_ms: u64,
    },
    EvaluationCompleted {
        repository: TaskRepositoryState,
        now_ms: u64,
    },
    PendingTasksRevised {
        plan: ArtifactEnvelope<TaskPlanArtifact>,
        review: ArtifactEnvelope<TaskPlanReviewArtifact>,
        reason: String,
        user_approved_expansion: bool,
        repository: TaskRepositoryState,
        now_ms: u64,
    },
    PauseRequested {
        now_ms: u64,
    },
    ResumeRequested {
        repository: TaskRepositoryState,
        now_ms: u64,
    },
    RetryBlockedTask {
        repository: TaskRepositoryState,
        now_ms: u64,
    },
    Cancelled {
        reason: String,
        now_ms: u64,
    },
}

impl MultiTaskEvent {
    const fn now_ms(&self) -> u64 {
        match self {
            Self::CoordinationUsageRecorded { now_ms, .. }
            | Self::DirectUsageRecorded { now_ms, .. }
            | Self::ChildStarted { now_ms, .. }
            | Self::ChildCheckpointed { now_ms, .. }
            | Self::GoalContractRevised { now_ms, .. }
            | Self::TaskDelivered { now_ms, .. }
            | Self::TaskStopped { now_ms, .. }
            | Self::EvaluationCompleted { now_ms, .. }
            | Self::PendingTasksRevised { now_ms, .. }
            | Self::PauseRequested { now_ms }
            | Self::ResumeRequested { now_ms, .. }
            | Self::RetryBlockedTask { now_ms, .. }
            | Self::Cancelled { now_ms, .. } => *now_ms,
        }
    }
}

impl MultiTaskRun {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        id: impl Into<String>,
        session_id: impl Into<String>,
        source_turn_id: impl Into<String>,
        plan: ArtifactEnvelope<TaskPlanArtifact>,
        plan_review: ArtifactEnvelope<TaskPlanReviewArtifact>,
        policy: CompiledTaskPolicy,
        workflow_policy: crate::workflow::CompiledWorkflowPolicy,
        goal_policy: crate::goal::CompiledGoalPolicy,
        request_max_steps: usize,
        source_intent: TaskSourceIntent,
        qualification: TaskPlannerQualification,
        workdir: impl Into<String>,
        repository: TaskRepositoryState,
        coordination: TaskCoordinationCounters,
        now_ms: u64,
    ) -> Result<Self> {
        let id = required("multi-Task run id", &id.into())?.to_string();
        let session_id = required("multi-Task session id", &session_id.into())?.to_string();
        let source_turn_id =
            required("multi-Task source turn id", &source_turn_id.into())?.to_string();
        let workdir = required("multi-Task workdir", &workdir.into())?.to_string();
        repository.validate()?;
        policy.validate()?;
        workflow_policy.validate()?;
        goal_policy.validate()?;
        if request_max_steps == 0 {
            bail!("multi-Task source request must allow at least one model step");
        }
        qualification.validate()?;
        let plan_authority = TaskPlanAuthority {
            source_intent,
            task_planning_qualified: qualification.task_planning,
            automatic_goal_selection_qualified: qualification.automatic_goal_selection,
        };
        plan.validate_digest()?;
        plan.artifact.validate(plan_authority, &policy)?;
        plan_review.validate_digest()?;
        plan_review.artifact.validate(&plan)?;
        if plan_review.artifact.verdict != TaskPlanReviewVerdict::Pass {
            bail!("multi-Task run requires a passing Task-plan review");
        }
        if plan.artifact.tasks.len() < 2 {
            bail!("MultiTaskRun is created only for plans with at least two Tasks");
        }
        if !coordination.fits_within(&plan.artifact.coordination_budget) {
            bail!("Task-plan coordination usage exceeds its accepted budget");
        }
        let tasks = plan
            .artifact
            .tasks
            .iter()
            .cloned()
            .map(|spec| TaskRun::queued(spec, 1))
            .collect();
        let mut run = Self {
            version: MULTI_TASK_RUN_VERSION,
            id,
            session_id,
            source_turn_id,
            stage: MultiTaskStage::Evaluating,
            plan_revision: 1,
            policy_sha256: policy.sha256.clone(),
            authority: TaskAuthorityEnvelope {
                workdir,
                publication: false,
                request_max_steps,
                plan: plan_authority,
                qualification,
                workflow_policy,
                goal_policy,
            },
            budget: plan.artifact.allocated_budget,
            counters: MultiTaskCounters {
                tasks: TaskCounters::default(),
                coordination,
            },
            active_task_id: None,
            tasks,
            expected_repository: repository,
            outcome: None,
            reason: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            plan,
            plan_review,
            policy,
        };
        run.activate_next(now_ms)?;
        run.validate()?;
        Ok(run)
    }

    pub fn apply(&mut self, event: MultiTaskEvent) -> Result<()> {
        *self = reduce(self.clone(), event)?;
        Ok(())
    }

    pub fn validate_reconciliation(&self, observed: &TaskRepositoryState) -> Result<()> {
        observed.validate()?;
        let expected = self
            .active_task()
            .and_then(|task| task.repository_checkpoint.as_ref())
            .unwrap_or(&self.expected_repository);
        if expected != observed {
            bail!(
                "multi-Task repository state changed since its checkpoint: {}",
                expected.difference(observed)
            );
        }
        Ok(())
    }

    pub fn active_task(&self) -> Option<&TaskRun> {
        let id = self.active_task_id.as_deref()?;
        self.tasks
            .iter()
            .find(|task| task.state.is_current() && task.spec.id == id)
    }

    fn active_task_mut(&mut self) -> Result<&mut TaskRun> {
        let id = self
            .active_task_id
            .clone()
            .context("multi-Task run has no active Task")?;
        self.tasks
            .iter_mut()
            .find(|task| task.state.is_current() && task.spec.id == id)
            .with_context(|| format!("active Task '{id}' is missing"))
    }

    fn current_task(&self, id: &str) -> Option<&TaskRun> {
        self.tasks
            .iter()
            .find(|task| task.state.is_current() && task.spec.id == id)
    }

    fn current_task_mut(&mut self, id: &str) -> Option<&mut TaskRun> {
        self.tasks
            .iter_mut()
            .find(|task| task.state.is_current() && task.spec.id == id)
    }

    fn activate_next(&mut self, now_ms: u64) -> Result<()> {
        if self.active_task_id.is_some() {
            bail!("multi-Task run already has an active Task");
        }
        let next_id = self.plan.artifact.tasks.iter().find_map(|spec| {
            let task = self.current_task(&spec.id)?;
            if task.state != TaskState::Queued {
                return None;
            }
            spec.depends_on
                .iter()
                .all(|id| {
                    self.current_task(id)
                        .is_some_and(|task| task.state.is_success())
                })
                .then(|| spec.id.clone())
        });
        let Some(next_id) = next_id else {
            if self.plan.artifact.tasks.iter().all(|spec| {
                self.current_task(&spec.id)
                    .is_some_and(|task| task.state.is_success())
            }) {
                self.stage = MultiTaskStage::Ready;
                self.outcome = Some(MultiTaskOutcome::Ready);
                self.reason = None;
                self.updated_at_ms = now_ms;
                return Ok(());
            }
            self.stage = MultiTaskStage::Blocked;
            self.outcome = Some(MultiTaskOutcome::TaskBlocked);
            self.reason = Some("no dependency-ready Task remains".to_string());
            self.updated_at_ms = now_ms;
            return Ok(());
        };
        let request = self.build_request(&next_id)?;
        let repository = self.expected_repository.clone();
        let task = self
            .current_task_mut(&next_id)
            .with_context(|| format!("Task '{next_id}' disappeared during activation"))?;
        task.attempts = task.attempts.saturating_add(1);
        task.state = TaskState::Running;
        task.request = Some(request);
        task.repository_checkpoint = Some(repository);
        task.blocked_reason = None;
        self.active_task_id = Some(next_id);
        self.stage = MultiTaskStage::RunningTask;
        self.outcome = None;
        self.reason = None;
        self.updated_at_ms = now_ms;
        Ok(())
    }

    fn build_request(&self, task_id: &str) -> Result<TaskRequest> {
        let task = self
            .current_task(task_id)
            .with_context(|| format!("unknown Task '{task_id}'"))?;
        let attempt = task.attempts.saturating_add(1);
        let child_id = format!("{}:{}:{attempt}", self.id, task.spec.id);
        let request_id = format!("task-request:{child_id}");
        let requirements = selected_requirements(&self.plan.artifact, &task.spec.requirement_ids)?;
        let acceptance = selected_acceptance(&self.plan.artifact, &task.spec.acceptance_ids)?;
        Ok(TaskRequest {
            id: request_id,
            multi_task_id: self.id.clone(),
            source_turn_id: self.source_turn_id.clone(),
            task_id: task.spec.id.clone(),
            child_id,
            workdir: self.authority.workdir.clone(),
            kind: task.spec.kind,
            title: task.spec.title.clone(),
            objective: task.spec.description.clone(),
            requirements,
            acceptance,
            scope_hints: task.spec.scope_hints.clone(),
            goal_contract: task.effective_goal_contract().cloned(),
            base_repository: self.expected_repository.clone(),
            budget: task.spec.budget,
            attempt,
        })
    }

    fn recompute_counters(&mut self) -> Result<()> {
        self.counters.tasks = self
            .tasks
            .iter()
            .try_fold(TaskCounters::default(), |total, task| {
                total.checked_add(task.counters)
            })?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != MULTI_TASK_RUN_VERSION {
            bail!("unsupported multi-Task run version {}", self.version);
        }
        required("multi-Task run id", &self.id)?;
        required("multi-Task session id", &self.session_id)?;
        required("multi-Task source turn id", &self.source_turn_id)?;
        required("multi-Task workdir", &self.authority.workdir)?;
        if !Path::new(&self.authority.workdir).is_absolute() {
            bail!("multi-Task workdir must be absolute");
        }
        if self.authority.publication {
            bail!("multi-Task runs cannot carry publication authority");
        }
        if self.authority.request_max_steps == 0 {
            bail!("multi-Task source request has no model-step allowance");
        }
        self.authority.qualification.validate()?;
        self.authority.workflow_policy.validate()?;
        self.authority.goal_policy.validate()?;
        if self.authority.plan.task_planning_qualified != self.authority.qualification.task_planning
            || self.authority.plan.automatic_goal_selection_qualified
                != self.authority.qualification.automatic_goal_selection
        {
            bail!("multi-Task authority does not match its planner qualification");
        }
        self.expected_repository.validate()?;
        self.policy.validate()?;
        if self.policy.sha256 != self.policy_sha256 {
            bail!("multi-Task policy hash does not match its compiled policy");
        }
        self.plan.validate_digest()?;
        self.plan
            .artifact
            .validate(self.authority.plan, &self.policy)?;
        self.plan_review.validate_digest()?;
        self.plan_review.artifact.validate(&self.plan)?;
        if self.plan_review.artifact.verdict != TaskPlanReviewVerdict::Pass {
            bail!("multi-Task checkpoint does not contain a passing plan review");
        }
        if self.plan.artifact.tasks.len() < 2 {
            bail!("multi-Task checkpoint contains fewer than two Tasks");
        }
        if self.budget != self.plan.artifact.allocated_budget {
            bail!("multi-Task budget does not match the accepted plan");
        }
        if !self
            .counters
            .coordination
            .fits_within(&self.plan.artifact.coordination_budget)
            && !(self.stage == MultiTaskStage::Failed
                && self.outcome == Some(MultiTaskOutcome::BudgetExhausted))
        {
            bail!("multi-Task coordination counters exceed the accepted budget");
        }
        if self.plan_revision == 0 {
            bail!("multi-Task plan revision must be positive");
        }
        if self.updated_at_ms < self.created_at_ms {
            bail!("multi-Task update timestamp predates creation");
        }

        for spec in &self.plan.artifact.tasks {
            let matches = self
                .tasks
                .iter()
                .filter(|task| task.state.is_current() && task.spec.id == spec.id)
                .collect::<Vec<_>>();
            if matches.len() != 1 || matches[0].spec != *spec {
                bail!(
                    "current Task '{}' does not match the accepted plan",
                    spec.id
                );
            }
        }
        if self.tasks.iter().any(|task| {
            task.state.is_current()
                && !self
                    .plan
                    .artifact
                    .tasks
                    .iter()
                    .any(|spec| spec.id == task.spec.id)
        }) {
            bail!("multi-Task checkpoint contains a current Task outside its plan");
        }
        for task in &self.tasks {
            self.validate_task(task)?;
        }
        let expected_counters = self
            .tasks
            .iter()
            .try_fold(TaskCounters::default(), |total, task| {
                total.checked_add(task.counters)
            })?;
        if expected_counters != self.counters.tasks {
            bail!("multi-Task counters do not equal the Task counter rollup");
        }
        let tasks_fit = self.counters.tasks.fits_within(&self.budget.tasks);
        if !tasks_fit
            && !(self.stage == MultiTaskStage::Failed
                && self.outcome == Some(MultiTaskOutcome::BudgetExhausted))
        {
            bail!("multi-Task counters exceed budget without a budget-exhausted outcome");
        }

        let active = self.active_task();
        match self.stage {
            MultiTaskStage::RunningTask | MultiTaskStage::Paused => {
                let task = active.context("running multi-Task checkpoint has no active Task")?;
                if task.state != TaskState::Running {
                    bail!("running multi-Task checkpoint points to a non-running Task");
                }
                if self.outcome.is_some() {
                    bail!("running multi-Task checkpoint has a terminal outcome");
                }
                if self
                    .tasks
                    .iter()
                    .filter(|candidate| {
                        candidate.state.is_current() && candidate.state == TaskState::Running
                    })
                    .count()
                    != 1
                {
                    bail!("multi-Task checkpoint does not contain exactly one running Task");
                }
            }
            MultiTaskStage::Blocked => {
                let task = active.context("blocked multi-Task checkpoint has no active Task")?;
                if task.state != TaskState::Blocked {
                    bail!("blocked multi-Task checkpoint points to a non-blocked Task");
                }
                if self.outcome != Some(MultiTaskOutcome::TaskBlocked) {
                    bail!("blocked multi-Task checkpoint has an invalid outcome");
                }
            }
            MultiTaskStage::Evaluating => {
                if active.is_some() {
                    bail!("evaluating multi-Task checkpoint retains an active Task");
                }
                if self.outcome.is_some() {
                    bail!("evaluating multi-Task checkpoint has a terminal outcome");
                }
                if self.tasks.iter().any(|task| {
                    task.state.is_current()
                        && !matches!(
                            task.state,
                            TaskState::Queued | TaskState::Committed | TaskState::NoChange
                        )
                }) {
                    bail!("evaluating multi-Task checkpoint contains a non-evaluable Task");
                }
            }
            MultiTaskStage::Ready => {
                if active.is_some()
                    || self.outcome != Some(MultiTaskOutcome::Ready)
                    || self.plan.artifact.tasks.iter().any(|spec| {
                        !self
                            .current_task(&spec.id)
                            .is_some_and(|task| task.state.is_success())
                    })
                {
                    bail!("ready multi-Task checkpoint has incomplete Tasks");
                }
            }
            MultiTaskStage::Failed => {
                if active.is_some()
                    || !matches!(
                        self.outcome,
                        Some(MultiTaskOutcome::TaskFailed | MultiTaskOutcome::BudgetExhausted)
                    )
                {
                    bail!("failed multi-Task checkpoint has an invalid terminal state");
                }
                match self.outcome {
                    Some(MultiTaskOutcome::TaskFailed)
                        if !self.tasks.iter().any(|task| {
                            task.state.is_current() && task.state == TaskState::Failed
                        }) =>
                    {
                        bail!("failed multi-Task checkpoint has no failed Task");
                    }
                    Some(MultiTaskOutcome::BudgetExhausted)
                        if self.counters.tasks.fits_within(&self.budget.tasks)
                            && self
                                .counters
                                .coordination
                                .fits_within(&self.plan.artifact.coordination_budget)
                            && !self.tasks.iter().any(|task| {
                                task.state.is_current()
                                    && !task.counters.fits_within(&task.spec.budget)
                            }) =>
                    {
                        bail!("budget-exhausted multi-Task checkpoint has no exhausted budget");
                    }
                    _ => {}
                }
            }
            MultiTaskStage::Cancelled => {
                if active.is_some() || self.outcome != Some(MultiTaskOutcome::Cancelled) {
                    bail!("cancelled multi-Task checkpoint has an invalid terminal state");
                }
                if self.tasks.iter().any(|task| {
                    task.state.is_current()
                        && !task.state.is_success()
                        && task.state != TaskState::Cancelled
                }) {
                    bail!("cancelled multi-Task checkpoint retains an unfinished Task");
                }
            }
        }
        Ok(())
    }

    fn validate_task(&self, task: &TaskRun) -> Result<()> {
        if task.revision == 0 || task.revision > self.plan_revision {
            bail!("Task '{}' has an invalid plan revision", task.spec.id);
        }
        let expected =
            task.child_watermarks
                .values()
                .try_fold(task.direct_counters, |total, watermark| {
                    required("Task child watermark id", &watermark.child_id)?;
                    validate_sha256("Task child checkpoint", &watermark.checkpoint_sha256)?;
                    if task
                        .child_watermarks
                        .get(&watermark.child_id)
                        .is_none_or(|stored| stored != watermark)
                    {
                        bail!("Task child watermark key does not match its child id");
                    }
                    if matches!(
                        (task.spec.kind, watermark.kind),
                        (TaskKind::Build, TaskChildKind::Goal)
                            | (TaskKind::Goal, TaskChildKind::Build)
                    ) {
                        bail!("Task child watermark kind does not match its Task");
                    }
                    total.checked_add(watermark.usage)
                })?;
        if task.direct_counters.workflows != 0
            || task.direct_counters.stage_steps != 0
            || task.direct_counters.plan_cycles != 0
            || task.direct_counters.repair_cycles != 0
        {
            bail!("Task direct counters contain child-owned usage dimensions");
        }
        if expected != task.counters {
            bail!(
                "Task '{}' counters do not match its watermarks",
                task.spec.id
            );
        }
        if !task.counters.fits_within(&task.spec.budget)
            && !(task.state == TaskState::Failed
                && self.outcome == Some(MultiTaskOutcome::BudgetExhausted))
        {
            bail!(
                "Task '{}' counters exceed its accepted budget",
                task.spec.id
            );
        }
        if let Some(repository) = &task.repository_checkpoint {
            repository.validate()?;
        }
        if let Some(workflow) = &task.workflow {
            workflow.validate()?;
            if task.spec.kind != TaskKind::Build || task.goal.is_some() {
                bail!(
                    "Task '{}' has an incompatible Build checkpoint",
                    task.spec.id
                );
            }
            validate_current_watermark(
                task,
                &workflow.run.id,
                &workflow.sha256,
                TaskChildKind::Build,
                workflow_usage(workflow),
            )?;
        }
        if let Some(goal) = &task.goal {
            goal.validate()?;
            if task.spec.kind != TaskKind::Goal || task.workflow.is_some() {
                bail!(
                    "Task '{}' has an incompatible Goal checkpoint",
                    task.spec.id
                );
            }
            validate_current_watermark(
                task,
                &goal.run.id,
                &goal.sha256,
                TaskChildKind::Goal,
                goal_usage(goal)?,
            )?;
        }
        if task.goal_contract_revision.is_some() && task.spec.kind != TaskKind::Goal {
            bail!(
                "Build Task '{}' contains a Goal contract revision",
                task.spec.id
            );
        }
        match task.state {
            TaskState::Queued | TaskState::Superseded => {
                if task.attempts != 0
                    || task.request.is_some()
                    || task.workflow.is_some()
                    || task.goal.is_some()
                    || task.goal_contract_revision.is_some()
                    || task.repository_checkpoint.is_some()
                    || task.result.is_some()
                    || task.counters != TaskCounters::default()
                {
                    bail!("queued Task '{}' contains execution state", task.spec.id);
                }
            }
            TaskState::Running | TaskState::Blocked | TaskState::Failed => {
                let request = task
                    .request
                    .as_ref()
                    .with_context(|| format!("Task '{}' has no request", task.spec.id))?;
                validate_request(self, task, request)?;
                if task.result.is_some() {
                    bail!("unfinished Task '{}' contains a result", task.spec.id);
                }
            }
            TaskState::Committed | TaskState::NoChange => {
                let request = task
                    .request
                    .as_ref()
                    .with_context(|| format!("Task '{}' has no request", task.spec.id))?;
                validate_request(self, task, request)?;
                let result = task
                    .result
                    .as_ref()
                    .with_context(|| format!("Task '{}' has no result", task.spec.id))?;
                validate_result_shape(task, request, result)?;
                if (task.state == TaskState::NoChange) != result.no_change {
                    bail!("Task '{}' state disagrees with its result", task.spec.id);
                }
            }
            TaskState::Cancelled => {
                if let Some(request) = &task.request {
                    validate_request(self, task, request)?;
                } else if task.attempts != 0
                    || task.workflow.is_some()
                    || task.goal.is_some()
                    || task.goal_contract_revision.is_some()
                    || task.repository_checkpoint.is_some()
                    || task.counters != TaskCounters::default()
                {
                    bail!(
                        "cancelled queued Task '{}' contains execution state",
                        task.spec.id
                    );
                }
                if task.result.is_some() {
                    bail!("cancelled Task '{}' contains a result", task.spec.id);
                }
            }
        }
        if task.state.is_current()
            && !matches!(task.state, TaskState::Queued | TaskState::Cancelled)
            && task.spec.depends_on.iter().any(|dependency| {
                !self
                    .current_task(dependency)
                    .is_some_and(|candidate| candidate.state.is_success())
            })
        {
            bail!(
                "Task '{}' started before its dependencies completed",
                task.spec.id
            );
        }
        Ok(())
    }
}

pub fn reduce(mut run: MultiTaskRun, event: MultiTaskEvent) -> Result<MultiTaskRun> {
    if run.stage.is_terminal() {
        bail!("multi-Task run '{}' is already terminal", run.id);
    }
    if event.now_ms() < run.updated_at_ms {
        bail!("multi-Task event timestamp moved backwards");
    }
    match event {
        MultiTaskEvent::CoordinationUsageRecorded { usage, now_ms } => {
            if run.stage != MultiTaskStage::Evaluating {
                bail!("Task coordination usage can be recorded only while evaluating the queue");
            }
            run.counters.coordination = run.counters.coordination.checked_add(usage)?;
            run.updated_at_ms = now_ms;
            if !run
                .counters
                .coordination
                .fits_within(&run.plan.artifact.coordination_budget)
            {
                run.stage = MultiTaskStage::Failed;
                run.outcome = Some(MultiTaskOutcome::BudgetExhausted);
                run.reason = Some("Task coordination budget exhausted".to_string());
            }
        }
        MultiTaskEvent::DirectUsageRecorded {
            task_id,
            usage,
            now_ms,
        } => {
            require_running_task(&run, &task_id)?;
            let task = run
                .current_task_mut(&task_id)
                .context("active Task disappeared")?;
            task.direct_counters = task.direct_counters.checked_add(usage.into())?;
            task.recompute_counters()?;
            run.recompute_counters()?;
            run.updated_at_ms = now_ms;
            stop_for_budget_if_needed(&mut run, &task_id)?;
        }
        MultiTaskEvent::ChildStarted {
            task_id,
            child,
            repository,
            now_ms,
        } => {
            require_running_task(&run, &task_id)?;
            if !child.is_initial() {
                bail!("Task child must be checkpointed at its initial planning boundary");
            }
            let request = run
                .current_task(&task_id)
                .and_then(|task| task.request.as_ref())
                .context("active Task has no request")?;
            if repository != request.base_repository {
                bail!(
                    "Task child did not start from its reconciled repository state: {}",
                    request.base_repository.difference(&repository)
                );
            }
            checkpoint_child(&mut run, &task_id, child, repository, now_ms, true)?;
        }
        MultiTaskEvent::ChildCheckpointed {
            task_id,
            child,
            repository,
            now_ms,
        } => {
            require_running_task(&run, &task_id)?;
            checkpoint_child(&mut run, &task_id, child, repository, now_ms, false)?;
        }
        MultiTaskEvent::GoalContractRevised {
            task_id,
            child,
            repository,
            now_ms,
        } => {
            require_running_task(&run, &task_id)?;
            let task = run
                .current_task(&task_id)
                .context("active Goal Task disappeared")?;
            if task.spec.kind != TaskKind::Goal || task.goal.is_none() {
                bail!("only a started Goal Task contract can be revised");
            }
            let request = task
                .request
                .as_ref()
                .context("active Goal Task has no request")?;
            let contract = validate_goal_task_revision(request, &child)?;
            let task = run
                .current_task_mut(&task_id)
                .context("active Goal Task disappeared")?;
            task.goal_contract_revision = Some(contract.clone());
            task.request
                .as_mut()
                .context("active Goal Task has no request")?
                .goal_contract = Some(contract);
            checkpoint_child(
                &mut run,
                &task_id,
                TaskChildCheckpoint::Goal(child),
                repository,
                now_ms,
                false,
            )?;
        }
        MultiTaskEvent::TaskDelivered {
            task_id,
            result,
            repository,
            now_ms,
        } => {
            require_running_task(&run, &task_id)?;
            if repository != result.terminal_repository {
                bail!("Task result does not describe the observed terminal repository");
            }
            let task = run
                .current_task(&task_id)
                .context("active Task disappeared")?;
            let request = task
                .request
                .as_ref()
                .context("active Task has no request")?;
            validate_delivered_result(task, request, &result)?;
            let task = run
                .current_task_mut(&task_id)
                .context("active Task disappeared")?;
            task.state = if result.no_change {
                TaskState::NoChange
            } else {
                TaskState::Committed
            };
            task.repository_checkpoint = Some(repository.clone());
            task.result = Some(result);
            task.blocked_reason = None;
            run.expected_repository = repository;
            run.active_task_id = None;
            run.stage = MultiTaskStage::Evaluating;
            run.outcome = None;
            run.reason = None;
            run.updated_at_ms = now_ms;
        }
        MultiTaskEvent::TaskStopped {
            task_id,
            disposition,
            reason,
            now_ms,
        } => {
            require_running_task(&run, &task_id)?;
            let reason = required("Task stop reason", &reason)?.to_string();
            let task = run
                .current_task_mut(&task_id)
                .context("active Task disappeared")?;
            task.blocked_reason = Some(reason.clone());
            match disposition {
                TaskStopDisposition::Blocked => {
                    task.state = TaskState::Blocked;
                    run.stage = MultiTaskStage::Blocked;
                    run.outcome = Some(MultiTaskOutcome::TaskBlocked);
                }
                TaskStopDisposition::Failed => {
                    task.state = TaskState::Failed;
                    run.active_task_id = None;
                    run.stage = MultiTaskStage::Failed;
                    run.outcome = Some(MultiTaskOutcome::TaskFailed);
                }
            }
            run.reason = Some(reason);
            run.updated_at_ms = now_ms;
        }
        MultiTaskEvent::EvaluationCompleted { repository, now_ms } => {
            if run.stage != MultiTaskStage::Evaluating {
                bail!("multi-Task run is not evaluating its queue");
            }
            reconcile_exact(&run.expected_repository, &repository)?;
            run.activate_next(now_ms)?;
        }
        MultiTaskEvent::PendingTasksRevised {
            plan,
            review,
            reason,
            user_approved_expansion,
            repository,
            now_ms,
        } => {
            if run.stage != MultiTaskStage::Evaluating {
                bail!("pending Tasks can be revised only at an evaluation boundary");
            }
            reconcile_exact(&run.expected_repository, &repository)?;
            required("Task-plan revision reason", &reason)?;
            apply_pending_revision(&mut run, plan, review, user_approved_expansion, now_ms)?;
        }
        MultiTaskEvent::PauseRequested { now_ms } => {
            if run.stage != MultiTaskStage::RunningTask {
                bail!("multi-Task run is not at a pausable running boundary");
            }
            run.stage = MultiTaskStage::Paused;
            run.updated_at_ms = now_ms;
        }
        MultiTaskEvent::ResumeRequested { repository, now_ms } => {
            if !matches!(run.stage, MultiTaskStage::Paused | MultiTaskStage::Blocked) {
                bail!("multi-Task run is not paused or blocked");
            }
            run.validate_reconciliation(&repository)?;
            if run.stage == MultiTaskStage::Blocked {
                run.active_task_mut()?.state = TaskState::Running;
            }
            run.stage = MultiTaskStage::RunningTask;
            run.outcome = None;
            run.reason = None;
            run.updated_at_ms = now_ms;
        }
        MultiTaskEvent::RetryBlockedTask { repository, now_ms } => {
            if run.stage != MultiTaskStage::Blocked {
                bail!("only a blocked Task can be retried");
            }
            run.validate_reconciliation(&repository)?;
            let task_id = run
                .active_task_id
                .clone()
                .context("blocked multi-Task run has no active Task")?;
            let request = run.build_request(&task_id)?;
            let task = run.active_task_mut()?;
            task.attempts = task.attempts.saturating_add(1);
            task.state = TaskState::Running;
            task.request = Some(request);
            task.workflow = None;
            task.goal = None;
            task.repository_checkpoint = Some(repository);
            task.blocked_reason = None;
            run.stage = MultiTaskStage::RunningTask;
            run.outcome = None;
            run.reason = None;
            run.updated_at_ms = now_ms;
        }
        MultiTaskEvent::Cancelled { reason, now_ms } => {
            let reason = required("multi-Task cancellation reason", &reason)?.to_string();
            for task in &mut run.tasks {
                if task.state.is_current()
                    && matches!(
                        task.state,
                        TaskState::Queued | TaskState::Running | TaskState::Blocked
                    )
                {
                    task.state = TaskState::Cancelled;
                    task.blocked_reason = Some(reason.clone());
                }
            }
            run.active_task_id = None;
            run.stage = MultiTaskStage::Cancelled;
            run.outcome = Some(MultiTaskOutcome::Cancelled);
            run.reason = Some(reason);
            run.updated_at_ms = now_ms;
        }
    }
    run.validate()?;
    Ok(run)
}

fn checkpoint_child(
    run: &mut MultiTaskRun,
    task_id: &str,
    child: TaskChildCheckpoint,
    repository: TaskRepositoryState,
    now_ms: u64,
    starting: bool,
) -> Result<()> {
    child.validate()?;
    repository.validate()?;
    let task = run
        .current_task(task_id)
        .context("active Task disappeared")?;
    let request = task
        .request
        .as_ref()
        .context("active Task has no request")?;
    if child.id() != request.child_id {
        bail!("Task child id does not match the active Task request");
    }
    match (&child, task.spec.kind) {
        (TaskChildCheckpoint::Build(checkpoint), TaskKind::Build) => {
            if checkpoint.run.source_turn_id != request.id {
                bail!("Build Task checkpoint is not bound to its Task request");
            }
        }
        (TaskChildCheckpoint::Goal(checkpoint), TaskKind::Goal) => {
            validate_goal_task_checkpoint(
                request,
                task.effective_goal_contract(),
                task.goal_contract_revision.is_some(),
                checkpoint,
            )?;
        }
        _ => bail!("Task child kind does not match its accepted Task kind"),
    }
    if !starting {
        let existing_id = task
            .workflow
            .as_ref()
            .map(|checkpoint| checkpoint.run.id.as_str())
            .or_else(|| {
                task.goal
                    .as_ref()
                    .map(|checkpoint| checkpoint.run.id.as_str())
            })
            .context("Task child must be started before later checkpoints")?;
        if existing_id != child.id() {
            bail!("Task child checkpoint changed child identity");
        }
    } else if task.workflow.is_some() || task.goal.is_some() {
        bail!("Task already has a started child");
    }
    let usage = child.usage()?;
    if let Some(prior) = task.child_watermarks.get(child.id())
        && !usage.is_monotonic_from(&prior.usage)
    {
        bail!("Task child counters moved backwards across checkpoints");
    }
    let child_id = child.id().to_string();
    let watermark = TaskChildWatermark {
        kind: child.kind(),
        child_id: child_id.clone(),
        checkpoint_sha256: child.sha256().to_string(),
        usage,
    };
    let task = run
        .current_task_mut(task_id)
        .context("active Task disappeared")?;
    match child {
        TaskChildCheckpoint::Build(checkpoint) => task.workflow = Some(checkpoint),
        TaskChildCheckpoint::Goal(checkpoint) => task.goal = Some(checkpoint),
    }
    task.child_watermarks.insert(child_id, watermark);
    task.repository_checkpoint = Some(repository);
    task.recompute_counters()?;
    run.recompute_counters()?;
    run.updated_at_ms = now_ms;
    stop_for_budget_if_needed(run, task_id)
}

fn stop_for_budget_if_needed(run: &mut MultiTaskRun, task_id: &str) -> Result<()> {
    let task_exhausted = run
        .current_task(task_id)
        .is_some_and(|task| !task.counters.fits_within(&task.spec.budget));
    let aggregate_exhausted = !run.counters.tasks.fits_within(&run.budget.tasks);
    if task_exhausted || aggregate_exhausted {
        let reason = if task_exhausted {
            format!("Task '{task_id}' exhausted its accepted budget")
        } else {
            "multi-Task run exhausted its aggregate budget".to_string()
        };
        let task = run
            .current_task_mut(task_id)
            .context("budget-exhausted Task disappeared")?;
        task.state = TaskState::Failed;
        task.blocked_reason = Some(reason.clone());
        run.active_task_id = None;
        run.stage = MultiTaskStage::Failed;
        run.outcome = Some(MultiTaskOutcome::BudgetExhausted);
        run.reason = Some(reason);
    }
    Ok(())
}

fn validate_delivered_result(
    task: &TaskRun,
    request: &TaskRequest,
    result: &TaskResult,
) -> Result<()> {
    validate_result_shape(task, request, result)?;
    match task.spec.kind {
        TaskKind::Build => {
            let workflow = task
                .workflow
                .as_ref()
                .context("Build Task has no terminal workflow checkpoint")?;
            if workflow.run.stage != WorkflowStage::Ready {
                bail!("Build Task workflow has not reached Ready");
            }
            match workflow.run.outcome {
                Some(WorkflowOutcome::Ready) => {
                    let commit = workflow
                        .run
                        .commit
                        .as_ref()
                        .context("ready Build Task has no managed commit")?;
                    if result.no_change || result.commits != [commit.oid.clone()] {
                        bail!("Build Task result does not preserve its managed commit");
                    }
                }
                Some(WorkflowOutcome::NoChange) => {
                    if !result.no_change || !result.commits.is_empty() {
                        bail!("no-change Build Task contains commits");
                    }
                }
                _ => bail!("Build Task workflow has no successful terminal outcome"),
            }
        }
        TaskKind::Goal => {
            let goal = task
                .goal
                .as_ref()
                .context("Goal Task has no terminal Goal checkpoint")?;
            if goal.run.stage != GoalStage::Completed
                || goal.run.outcome != Some(GoalOutcome::Complete)
            {
                bail!("Goal Task has not reached accepted Goal completion");
            }
            let commits = goal_commits(goal);
            if result.commits != commits {
                bail!("Goal Task result does not preserve its child commit range");
            }
            if result.no_change != commits.is_empty() {
                bail!("Goal Task no-change result disagrees with its child commits");
            }
        }
    }
    Ok(())
}

fn validate_result_shape(task: &TaskRun, request: &TaskRequest, result: &TaskResult) -> Result<()> {
    result.base_repository.validate()?;
    result.terminal_repository.validate()?;
    if result.base_repository != request.base_repository {
        bail!("Task result base does not match the activated Task request");
    }
    if task.repository_checkpoint.as_ref() != Some(&result.terminal_repository) {
        bail!("Task result is not bound to the latest child repository checkpoint");
    }
    required("Task result summary", &result.summary)?;
    if result.evidence_refs.is_empty()
        || result
            .evidence_refs
            .iter()
            .any(|value| value.trim().is_empty())
    {
        bail!("Task result requires non-empty evidence references");
    }
    unique_values("Task result evidence", &result.evidence_refs)?;
    unique_values("Task result acceptance", &result.satisfied_acceptance_ids)?;
    let expected = task
        .spec
        .acceptance_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let actual = result
        .satisfied_acceptance_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if expected != actual {
        bail!("Task result does not satisfy its exact acceptance mapping");
    }
    if result.no_change {
        if !result.commits.is_empty() || result.terminal_repository != result.base_repository {
            bail!("no-change Task result changed repository state or recorded commits");
        }
    } else {
        if result.commits.is_empty() {
            bail!("changed Task result has no commits");
        }
        unique_values("Task result commit", &result.commits)?;
        for commit in &result.commits {
            validate_git_oid("Task result commit", commit)?;
        }
        if result.terminal_repository.head != *result.commits.last().unwrap() {
            bail!("Task terminal HEAD does not equal its final recorded commit");
        }
    }
    Ok(())
}

fn apply_pending_revision(
    run: &mut MultiTaskRun,
    plan: ArtifactEnvelope<TaskPlanArtifact>,
    review: ArtifactEnvelope<TaskPlanReviewArtifact>,
    user_approved_expansion: bool,
    now_ms: u64,
) -> Result<()> {
    plan.validate_digest()?;
    plan.artifact.validate(run.authority.plan, &run.policy)?;
    review.validate_digest()?;
    review.artifact.validate(&plan)?;
    if review.artifact.verdict != TaskPlanReviewVerdict::Pass {
        bail!("Task-plan revision requires a passing review");
    }
    if plan.artifact.tasks.len() < 2 {
        bail!("a multi-Task revision cannot unwrap into a single Task");
    }
    if plan.artifact.objective != run.plan.artifact.objective {
        bail!("Task-plan revision cannot change the source objective");
    }
    if plan.artifact.allocated_budget != run.budget
        || plan.artifact.coordination_budget != run.plan.artifact.coordination_budget
    {
        bail!("Task-plan revision cannot change accepted parent budgets");
    }
    for task in run
        .tasks
        .iter()
        .filter(|task| task.state.is_current() && task.state.is_success())
    {
        if !plan
            .artifact
            .tasks
            .iter()
            .any(|candidate| candidate == &task.spec)
        {
            bail!("Task-plan revision cannot change completed Task history");
        }
    }
    if revision_expands(&run.plan.artifact, &plan.artifact) && !user_approved_expansion {
        bail!("Task-plan expansion requires an explicit user decision");
    }

    let next_revision = run.plan_revision.saturating_add(1);
    for task in &mut run.tasks {
        if task.state == TaskState::Queued
            && !plan.artifact.tasks.iter().any(|spec| spec == &task.spec)
        {
            task.state = TaskState::Superseded;
        }
    }
    for spec in &plan.artifact.tasks {
        if !run
            .tasks
            .iter()
            .any(|task| task.state.is_current() && task.spec == *spec)
        {
            run.tasks.push(TaskRun::queued(spec.clone(), next_revision));
        }
    }
    run.plan = plan;
    run.plan_review = review;
    run.plan_revision = next_revision;
    run.updated_at_ms = now_ms;
    run.validate()
}

fn revision_expands(old: &TaskPlanArtifact, new: &TaskPlanArtifact) -> bool {
    let old_requirements = old
        .requirements
        .iter()
        .map(|item| (&item.id, &item.description))
        .collect::<BTreeSet<_>>();
    let old_acceptance = old
        .acceptance
        .iter()
        .map(|item| (&item.id, &item.description))
        .collect::<BTreeSet<_>>();
    if new
        .requirements
        .iter()
        .any(|item| !old_requirements.contains(&(&item.id, &item.description)))
        || new
            .acceptance
            .iter()
            .any(|item| !old_acceptance.contains(&(&item.id, &item.description)))
    {
        return true;
    }
    for task in &new.tasks {
        let Some(prior) = old.tasks.iter().find(|candidate| candidate.id == task.id) else {
            return true;
        };
        if task.kind != prior.kind
            || task.budget != prior.budget
            || task.goal_contract != prior.goal_contract
            || !is_subset(&task.requirement_ids, &prior.requirement_ids)
            || !is_subset(&task.acceptance_ids, &prior.acceptance_ids)
            || !is_subset(&task.scope_hints, &prior.scope_hints)
            || !is_subset(&prior.depends_on, &task.depends_on)
        {
            return true;
        }
    }
    false
}

fn is_subset(values: &[String], ceiling: &[String]) -> bool {
    values.iter().all(|value| ceiling.contains(value))
}

fn require_running_task(run: &MultiTaskRun, task_id: &str) -> Result<()> {
    if run.stage != MultiTaskStage::RunningTask {
        bail!("multi-Task run is not running a child Task");
    }
    if run.active_task_id.as_deref() != Some(task_id) {
        bail!("Task '{task_id}' is not the active Task");
    }
    if !run
        .current_task(task_id)
        .is_some_and(|task| task.state == TaskState::Running)
    {
        bail!("Task '{task_id}' is not running");
    }
    Ok(())
}

fn validate_request(run: &MultiTaskRun, task: &TaskRun, request: &TaskRequest) -> Result<()> {
    required("Task request id", &request.id)?;
    required("Task request child id", &request.child_id)?;
    required("Task request workdir", &request.workdir)?;
    if !Path::new(&request.workdir).is_absolute() {
        bail!("Task request workdir must be absolute");
    }
    let expected_child_id = format!("{}:{}:{}", run.id, task.spec.id, task.attempts);
    let expected_request_id = format!("task-request:{expected_child_id}");
    if request.multi_task_id != run.id
        || request.source_turn_id != run.source_turn_id
        || request.task_id != task.spec.id
        || request.workdir != run.authority.workdir
        || request.kind != task.spec.kind
        || request.title != task.spec.title
        || request.objective != task.spec.description
        || request.scope_hints != task.spec.scope_hints
        || request.goal_contract.as_ref() != task.effective_goal_contract()
        || request.budget != task.spec.budget
        || request.attempt != task.attempts
        || request.child_id != expected_child_id
        || request.id != expected_request_id
    {
        bail!("Task request no longer matches its accepted Task");
    }
    request.base_repository.validate()?;
    if request.requirements
        != selected_requirements(&run.plan.artifact, &task.spec.requirement_ids)?
        || request.acceptance != selected_acceptance(&run.plan.artifact, &task.spec.acceptance_ids)?
    {
        bail!("Task request traceability does not match the accepted plan");
    }
    if let Some(workflow) = &task.workflow
        && (workflow.run.id != request.child_id || workflow.run.source_turn_id != request.id)
    {
        bail!("Build checkpoint is not bound to its Task request");
    }
    if let Some(goal) = &task.goal
        && (goal.run.id != request.child_id
            || validate_goal_task_checkpoint(
                request,
                task.effective_goal_contract(),
                task.goal_contract_revision.is_some(),
                goal,
            )
            .is_err())
    {
        bail!("Goal checkpoint is not bound to its Task request");
    }
    Ok(())
}

fn validate_goal_task_checkpoint(
    request: &TaskRequest,
    contract: Option<&super::TaskGoalContract>,
    contract_was_revised: bool,
    checkpoint: &GoalCheckpoint,
) -> Result<()> {
    let contract = contract.context("Goal Task request has no Goal contract")?;
    if checkpoint.run.id != request.child_id
        || checkpoint.run.objective != contract.objective
        || checkpoint.run.continuation != contract.continuation
        || checkpoint.run.criteria.len() != contract.criteria.len()
        || checkpoint
            .run
            .criteria
            .iter()
            .zip(&contract.criteria)
            .any(|(actual, expected)| {
                actual.text.trim() != expected.text.trim() || actual.verifier != expected.verifier
            })
        || (!contract_was_revised && !checkpoint.run.retired_criteria.is_empty())
    {
        bail!("Goal checkpoint changed the accepted Goal Task contract");
    }
    if checkpoint.run.authority.publication || checkpoint.run.authority.workdir != request.workdir {
        bail!("Goal checkpoint changed the parent Task authority");
    }
    let budget = checkpoint.run.budget;
    if budget.max_milestones > request.budget.max_workflows
        || budget.max_workflows > request.budget.max_workflows
        || budget.total_model_invocations > request.budget.total_model_invocations
        || budget.total_generated_tokens > request.budget.total_generated_tokens
        || budget.wall_time_minutes > request.budget.wall_time_minutes
    {
        bail!("Goal checkpoint exceeds its parent Task budget");
    }
    Ok(())
}

fn validate_goal_task_revision(
    request: &TaskRequest,
    checkpoint: &GoalCheckpoint,
) -> Result<super::TaskGoalContract> {
    checkpoint.validate()?;
    let accepted = request
        .goal_contract
        .as_ref()
        .context("Goal Task request has no accepted Goal contract")?;
    if checkpoint.run.id != request.child_id
        || checkpoint.run.objective != accepted.objective
        || checkpoint.run.authority.publication
        || checkpoint.run.authority.workdir != request.workdir
    {
        bail!("Goal Task revision changed objective, identity, or authority");
    }
    if accepted.criteria.iter().any(|expected| {
        !checkpoint.run.criteria.iter().any(|actual| {
            actual.text.trim() == expected.text.trim() && actual.verifier == expected.verifier
        })
    }) {
        bail!("Goal Task revision removed an accepted criterion");
    }
    let budget = checkpoint.run.budget;
    if budget.max_milestones > request.budget.max_workflows
        || budget.max_workflows > request.budget.max_workflows
        || budget.total_model_invocations > request.budget.total_model_invocations
        || budget.total_generated_tokens > request.budget.total_generated_tokens
        || budget.wall_time_minutes > request.budget.wall_time_minutes
    {
        bail!("Goal Task revision exceeds its parent Task budget");
    }
    Ok(super::TaskGoalContract {
        objective: checkpoint.run.objective.clone(),
        criteria: checkpoint
            .run
            .criteria
            .iter()
            .map(|criterion| super::TaskGoalCriterion {
                text: criterion.text.clone(),
                verifier: criterion.verifier,
            })
            .collect(),
        continuation: checkpoint.run.continuation,
    })
}

fn validate_current_watermark(
    task: &TaskRun,
    child_id: &str,
    checkpoint_sha256: &str,
    kind: TaskChildKind,
    usage: TaskCounters,
) -> Result<()> {
    let watermark = task
        .child_watermarks
        .get(child_id)
        .with_context(|| format!("Task '{}' has no current child watermark", task.spec.id))?;
    if watermark.kind != kind
        || watermark.checkpoint_sha256 != checkpoint_sha256
        || watermark.usage != usage
    {
        bail!(
            "Task '{}' child checkpoint does not match its usage watermark",
            task.spec.id
        );
    }
    Ok(())
}

fn selected_requirements(plan: &TaskPlanArtifact, ids: &[String]) -> Result<Vec<TaskRequirement>> {
    ids.iter()
        .map(|id| {
            plan.requirements
                .iter()
                .find(|item| item.id == *id)
                .cloned()
                .with_context(|| format!("Task references missing requirement '{id}'"))
        })
        .collect()
}

fn selected_acceptance(plan: &TaskPlanArtifact, ids: &[String]) -> Result<Vec<TaskAcceptance>> {
    ids.iter()
        .map(|id| {
            plan.acceptance
                .iter()
                .find(|item| item.id == *id)
                .cloned()
                .with_context(|| format!("Task references missing acceptance '{id}'"))
        })
        .collect()
}

fn workflow_usage(checkpoint: &WorkflowCheckpoint) -> TaskCounters {
    let counters = &checkpoint.run.counters;
    TaskCounters {
        workflows: 1,
        stage_steps: counters.stage_steps.values().copied().sum(),
        model_invocations: counters.model_invocations,
        generated_tokens: counters.generated_tokens,
        advisory_calls: counters.advisory_calls,
        plan_cycles: counters.plan_cycles,
        repair_cycles: counters.repair_cycles,
        elapsed_ms: 0,
    }
}

fn goal_usage(checkpoint: &GoalCheckpoint) -> Result<TaskCounters> {
    let effective = checkpoint.run.effective_counters();
    let mut usage = TaskCounters {
        workflows: effective.workflows,
        model_invocations: effective.model_invocations,
        generated_tokens: effective.generated_tokens,
        advisory_calls: effective.advisory_calls,
        ..TaskCounters::default()
    };
    for workflow in checkpoint
        .run
        .milestones
        .iter()
        .filter_map(|milestone| milestone.workflow.as_ref())
    {
        let child = workflow_usage(workflow);
        usage.stage_steps = checked_add(
            "Goal Task stage steps",
            usage.stage_steps,
            child.stage_steps,
        )?;
        usage.plan_cycles = checked_add(
            "Goal Task plan cycles",
            usage.plan_cycles,
            child.plan_cycles,
        )?;
        usage.repair_cycles = checked_add(
            "Goal Task repair cycles",
            usage.repair_cycles,
            child.repair_cycles,
        )?;
    }
    Ok(usage)
}

fn goal_commits(checkpoint: &GoalCheckpoint) -> Vec<String> {
    checkpoint
        .run
        .milestones
        .iter()
        .filter_map(|milestone| milestone.workflow.as_ref())
        .filter_map(|workflow| workflow.run.commit.as_ref())
        .map(|commit| commit.oid.clone())
        .collect()
}

fn reconcile_exact(expected: &TaskRepositoryState, observed: &TaskRepositoryState) -> Result<()> {
    observed.validate()?;
    if expected != observed {
        bail!(
            "multi-Task repository reconciliation failed: {}",
            expected.difference(observed)
        );
    }
    Ok(())
}

fn unique_values(label: &str, values: &[String]) -> Result<()> {
    let mut seen = HashSet::new();
    for value in values {
        required(label, value)?;
        if !seen.insert(value) {
            bail!("{label} contains duplicate '{value}'");
        }
    }
    Ok(())
}

fn git_bytes(repo_root: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

fn checked_add(label: &str, left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .with_context(|| format!("{label} overflow"))
}

fn required<'a>(label: &str, value: &'a str) -> Result<&'a str> {
    if value.trim().is_empty() {
        bail!("{label} must not be empty");
    }
    Ok(value)
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} digest must be a 64-character hexadecimal SHA-256");
    }
    Ok(())
}

fn validate_git_oid(label: &str, value: &str) -> Result<()> {
    if !matches!(value.len(), 40 | 64) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} must be a 40- or 64-character hexadecimal Git object id");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::events::HandoffCommitSummary;
    use crate::task_queue::{
        MultiTaskCheckpoint, TaskConfigDocument, TaskPlanProposal, TaskPlanReviewArtifact,
        TaskPlanReviewVerdict, TaskSourceIntent,
    };
    use crate::workflow::{WorkflowConfigDocument, WorkflowCounters, WorkflowRun};
    use crate::workspace::{
        ContentSnapshot, RepositoryContext, WorkspaceBaseline, WorkspaceStatus,
    };

    use super::*;

    fn digest(value: &str) -> String {
        format!("{:x}", Sha256::digest(value.as_bytes()))
    }

    fn repository(label: &str) -> TaskRepositoryState {
        TaskRepositoryState {
            head: digest(&format!("{label}:head")),
            index_sha256: digest(&format!("{label}:index")),
            refs_sha256: digest(&format!("{label}:refs")),
            content_sha256: digest(&format!("{label}:content")),
        }
    }

    fn authority() -> TaskPlanAuthority {
        TaskPlanAuthority {
            source_intent: TaskSourceIntent::Build,
            task_planning_qualified: true,
            automatic_goal_selection_qualified: false,
        }
    }

    fn qualification() -> TaskPlannerQualification {
        let value = digest("qualification");
        TaskPlannerQualification::new(
            value.clone(),
            value.clone(),
            value.clone(),
            value,
            true,
            false,
        )
        .unwrap()
    }

    fn proposal(second_scope: &[&str]) -> TaskPlanProposal {
        serde_json::from_value(serde_json::json!({
            "objective": "Deliver two bounded changes",
            "requirements": [
                {"id": "r1", "description": "First outcome"},
                {"id": "r2", "description": "Second outcome"}
            ],
            "tasks": [
                {
                    "id": "t1",
                    "title": "First",
                    "description": "Deliver the first outcome",
                    "requirement_ids": ["r1"],
                    "acceptance_ids": ["a1"],
                    "scope_hints": ["src/first"],
                    "effort": "small",
                    "kind": "build"
                },
                {
                    "id": "t2",
                    "title": "Second",
                    "description": "Deliver the second outcome",
                    "requirement_ids": ["r2"],
                    "depends_on": ["t1"],
                    "acceptance_ids": ["a2"],
                    "scope_hints": second_scope,
                    "effort": "small",
                    "kind": "build"
                }
            ],
            "acceptance": [
                {"id": "a1", "description": "First evidence is current"},
                {"id": "a2", "description": "Second evidence is current"}
            ]
        }))
        .unwrap()
    }

    fn reviewed_plan(
        second_scope: &[&str],
        policy: &CompiledTaskPolicy,
        suffix: &str,
    ) -> (
        ArtifactEnvelope<TaskPlanArtifact>,
        ArtifactEnvelope<TaskPlanReviewArtifact>,
    ) {
        let artifact = proposal(second_scope)
            .validate_and_compile(authority(), policy)
            .unwrap();
        let plan = ArtifactEnvelope::new(format!("task-plan-{suffix}"), artifact).unwrap();
        let review = ArtifactEnvelope::new(
            format!("task-plan-review-{suffix}"),
            TaskPlanReviewArtifact {
                task_plan_sha256: plan.sha256.clone(),
                verdict: TaskPlanReviewVerdict::Pass,
                challenges: Vec::new(),
            },
        )
        .unwrap();
        (plan, review)
    }

    fn run() -> MultiTaskRun {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let (plan, review) = reviewed_plan(&["src/second"], &policy, "1");
        MultiTaskRun::start(
            "multi-1",
            "session-1",
            "turn-1",
            plan,
            review,
            policy,
            crate::workflow::WorkflowConfigDocument::default()
                .compile()
                .unwrap(),
            crate::goal::GoalConfigDocument::default()
                .compile()
                .unwrap(),
            32,
            TaskSourceIntent::Build,
            qualification(),
            "/workspace",
            repository("base"),
            TaskCoordinationCounters {
                planning_attempts: 1,
                model_invocations: 1,
                generated_tokens: 500,
                advisory_calls: 1,
                elapsed_ms: 1_000,
            },
            10,
        )
        .unwrap()
    }

    fn workspace_context(head: &str) -> RepositoryContext {
        let baseline = WorkspaceBaseline {
            id: digest("baseline"),
            head: Some(head.to_string()),
            status: WorkspaceStatus {
                porcelain: String::new(),
                dirty_paths: Vec::new(),
            },
            content: ContentSnapshot {
                fingerprint: digest("content"),
                paths: BTreeMap::new(),
            },
        };
        RepositoryContext {
            repo_root: PathBuf::from("/workspace"),
            focus_root: PathBuf::from("/workspace"),
            task_baseline: baseline.clone(),
            invocation_baseline: baseline,
        }
    }

    fn workflow_checkpoint(
        request: &TaskRequest,
        counters: WorkflowCounters,
        outcome: Option<WorkflowOutcome>,
        commit: Option<&str>,
    ) -> WorkflowCheckpoint {
        let policy = WorkflowConfigDocument::default().compile().unwrap();
        let mut workflow = WorkflowRun::start(
            request.child_id.clone(),
            request.id.clone(),
            request.objective.clone(),
            policy,
            workspace_context(&request.base_repository.head),
        )
        .unwrap();
        workflow.counters = counters;
        if let Some(outcome) = outcome {
            workflow.ready_evidence_schema = 0;
            workflow.stage = WorkflowStage::Ready;
            workflow.outcome = Some(outcome);
            workflow.commit = commit.map(|oid| HandoffCommitSummary {
                oid: oid.to_string(),
                subject: "test: deliver Task".to_string(),
            });
        }
        WorkflowCheckpoint::new(workflow).unwrap()
    }

    fn active_request(run: &MultiTaskRun) -> TaskRequest {
        run.active_task().unwrap().request.clone().unwrap()
    }

    fn goal_task_request() -> TaskRequest {
        TaskRequest {
            id: "task-request:multi-goal:g1:1".to_string(),
            multi_task_id: "multi-goal".to_string(),
            source_turn_id: "turn-goal".to_string(),
            task_id: "g1".to_string(),
            child_id: "multi-goal:g1:1".to_string(),
            workdir: "/workspace".to_string(),
            kind: TaskKind::Goal,
            title: "Prove recovery".to_string(),
            objective: "Deliver recovery evidence".to_string(),
            requirements: Vec::new(),
            acceptance: Vec::new(),
            scope_hints: Vec::new(),
            goal_contract: Some(super::super::TaskGoalContract {
                objective: "Prove restart recovery".to_string(),
                criteria: vec![super::super::TaskGoalCriterion {
                    text: "Recovery is machine verified".to_string(),
                    verifier: crate::goal::GoalVerifier::WorkflowReady,
                }],
                continuation: crate::goal::GoalContinuationPolicy::ReviewPlanThenAutomatic,
            }),
            base_repository: repository("goal-base"),
            budget: TaskBudget {
                max_workflows: 3,
                stage_steps: 20,
                total_model_invocations: 20,
                total_generated_tokens: 10_000,
                advisory_calls: 4,
                plan_cycles: 2,
                repair_cycles: 2,
                wall_time_minutes: 30,
            },
            attempt: 1,
        }
    }

    fn goal_checkpoint(request: &TaskRequest) -> GoalCheckpoint {
        let contract = request.goal_contract.as_ref().unwrap();
        let policy = crate::goal::GoalConfigDocument::default()
            .compile()
            .unwrap();
        let run = crate::goal::GoalRun::start(
            request.child_id.clone(),
            "session-1",
            contract.objective.clone(),
            contract
                .criteria
                .iter()
                .map(crate::goal::GoalCriterionInput::from)
                .collect(),
            contract.continuation,
            Some(crate::goal::GoalBudget {
                max_milestones: 2,
                max_workflows: 3,
                total_model_invocations: 20,
                total_generated_tokens: 10_000,
                wall_time_minutes: 30,
            }),
            policy,
            request.workdir.clone(),
            10,
        )
        .unwrap();
        GoalCheckpoint::new(run).unwrap()
    }

    fn start_child(run: &mut MultiTaskRun, counters: WorkflowCounters, now_ms: u64) {
        let request = active_request(run);
        run.apply(MultiTaskEvent::ChildStarted {
            task_id: request.task_id.clone(),
            child: TaskChildCheckpoint::Build(workflow_checkpoint(&request, counters, None, None)),
            repository: request.base_repository,
            now_ms,
        })
        .unwrap();
    }

    fn deliver_no_change(run: &mut MultiTaskRun, now_ms: u64) {
        let request = active_request(run);
        let counters = run
            .active_task()
            .unwrap()
            .workflow
            .as_ref()
            .map_or_else(WorkflowCounters::default, |checkpoint| {
                checkpoint.run.counters.clone()
            });
        let terminal =
            workflow_checkpoint(&request, counters, Some(WorkflowOutcome::NoChange), None);
        run.apply(MultiTaskEvent::ChildCheckpointed {
            task_id: request.task_id.clone(),
            child: TaskChildCheckpoint::Build(terminal),
            repository: request.base_repository.clone(),
            now_ms,
        })
        .unwrap();
        run.apply(MultiTaskEvent::TaskDelivered {
            task_id: request.task_id.clone(),
            result: TaskResult {
                base_repository: request.base_repository.clone(),
                terminal_repository: request.base_repository.clone(),
                commits: Vec::new(),
                no_change: true,
                satisfied_acceptance_ids: run.active_task().unwrap().spec.acceptance_ids.clone(),
                evidence_refs: vec![format!("workflow:{}", request.child_id)],
                summary: "Current evidence proves no repository change is needed".to_string(),
            },
            repository: request.base_repository,
            now_ms: now_ms + 1,
        })
        .unwrap();
    }

    #[test]
    fn start_activates_only_the_first_dependency_ready_task() {
        let run = run();
        assert_eq!(run.stage, MultiTaskStage::RunningTask);
        assert_eq!(run.active_task_id.as_deref(), Some("t1"));
        assert_eq!(run.current_task("t1").unwrap().attempts, 1);
        assert!(run.current_task("t1").unwrap().request.is_some());
        assert_eq!(run.current_task("t2").unwrap().state, TaskState::Queued);
        assert!(run.current_task("t2").unwrap().request.is_none());
        assert!(run.current_task("t2").unwrap().workflow.is_none());
    }

    #[test]
    fn explicit_goal_revision_may_add_bounded_criteria_but_cannot_weaken_contract() {
        let request = goal_task_request();
        let initial = goal_checkpoint(&request);
        validate_goal_task_checkpoint(&request, request.goal_contract.as_ref(), false, &initial)
            .unwrap();

        let mut revised = initial.run.clone();
        revised
            .revise_initial_plan(
                revised.objective.clone(),
                vec![
                    crate::goal::GoalCriterionInput {
                        text: "Recovery is machine verified".to_string(),
                        verifier: crate::goal::GoalVerifier::WorkflowReady,
                    },
                    crate::goal::GoalCriterionInput {
                        text: "The restart record is reviewed".to_string(),
                        verifier: crate::goal::GoalVerifier::ReviewRequired,
                    },
                ],
                crate::goal::GoalContinuationPolicy::ManualMilestones,
                Some(revised.budget),
                20,
            )
            .unwrap();
        let revised = GoalCheckpoint::new(revised).unwrap();
        let contract = validate_goal_task_revision(&request, &revised).unwrap();
        assert_eq!(contract.criteria.len(), 2);
        assert_eq!(
            contract.continuation,
            crate::goal::GoalContinuationPolicy::ManualMilestones
        );

        let mut weakened = initial.run;
        weakened
            .revise_initial_plan(
                weakened.objective.clone(),
                vec![crate::goal::GoalCriterionInput {
                    text: "A weaker replacement".to_string(),
                    verifier: crate::goal::GoalVerifier::ReviewRequired,
                }],
                weakened.continuation,
                Some(weakened.budget),
                21,
            )
            .unwrap();
        let weakened = GoalCheckpoint::new(weakened).unwrap();
        assert!(validate_goal_task_revision(&request, &weakened).is_err());
    }

    #[test]
    fn replayed_child_checkpoint_is_not_double_charged() {
        let mut run = run();
        start_child(&mut run, WorkflowCounters::default(), 20);
        let request = active_request(&run);
        let checkpoint = workflow_checkpoint(
            &request,
            WorkflowCounters {
                model_invocations: 3,
                generated_tokens: 900,
                ..WorkflowCounters::default()
            },
            None,
            None,
        );
        let event = MultiTaskEvent::ChildCheckpointed {
            task_id: "t1".to_string(),
            child: TaskChildCheckpoint::Build(checkpoint),
            repository: repository("during-build"),
            now_ms: 30,
        };
        run.apply(event.clone()).unwrap();
        let once = run.counters.tasks;
        run.apply(event).unwrap();
        assert_eq!(run.counters.tasks, once);
        assert_eq!(once.workflows, 1);
        assert_eq!(once.model_invocations, 3);
        assert_eq!(once.generated_tokens, 900);
    }

    #[test]
    fn child_counter_regression_fails_closed() {
        let mut run = run();
        start_child(
            &mut run,
            WorkflowCounters {
                model_invocations: 3,
                ..WorkflowCounters::default()
            },
            20,
        );
        let request = active_request(&run);
        let backwards = workflow_checkpoint(
            &request,
            WorkflowCounters {
                model_invocations: 2,
                ..WorkflowCounters::default()
            },
            None,
            None,
        );
        assert!(
            run.apply(MultiTaskEvent::ChildCheckpointed {
                task_id: "t1".to_string(),
                child: TaskChildCheckpoint::Build(backwards),
                repository: repository("during-build"),
                now_ms: 30,
            })
            .is_err()
        );
        assert_eq!(run.counters.tasks.model_invocations, 3);
    }

    #[test]
    fn next_task_is_created_only_after_delivery_and_reconciliation() {
        let mut run = run();
        start_child(&mut run, WorkflowCounters::default(), 20);
        deliver_no_change(&mut run, 30);
        assert_eq!(run.stage, MultiTaskStage::Evaluating);
        assert!(run.active_task_id.is_none());
        assert!(run.current_task("t2").unwrap().request.is_none());

        let stale = repository("stale");
        assert!(
            run.apply(MultiTaskEvent::EvaluationCompleted {
                repository: stale,
                now_ms: 40,
            })
            .is_err()
        );
        assert!(run.current_task("t2").unwrap().request.is_none());

        run.apply(MultiTaskEvent::EvaluationCompleted {
            repository: repository("base"),
            now_ms: 41,
        })
        .unwrap();
        assert_eq!(run.active_task_id.as_deref(), Some("t2"));
        let request = active_request(&run);
        assert_eq!(request.base_repository, repository("base"));
        assert_eq!(request.attempt, 1);
    }

    #[test]
    fn completing_every_task_reaches_ready_only_after_final_evaluation() {
        let mut run = run();
        start_child(&mut run, WorkflowCounters::default(), 20);
        deliver_no_change(&mut run, 30);
        run.apply(MultiTaskEvent::EvaluationCompleted {
            repository: repository("base"),
            now_ms: 40,
        })
        .unwrap();
        start_child(&mut run, WorkflowCounters::default(), 50);
        deliver_no_change(&mut run, 60);
        assert_eq!(run.stage, MultiTaskStage::Evaluating);
        run.apply(MultiTaskEvent::EvaluationCompleted {
            repository: repository("base"),
            now_ms: 70,
        })
        .unwrap();
        assert_eq!(run.stage, MultiTaskStage::Ready);
        assert_eq!(run.outcome, Some(MultiTaskOutcome::Ready));
    }

    #[test]
    fn changed_build_result_must_preserve_its_managed_commit_and_terminal_head() {
        let mut run = run();
        start_child(&mut run, WorkflowCounters::default(), 20);
        let request = active_request(&run);
        let commit = digest("task-commit");
        let mut terminal_repository = repository("after-build");
        terminal_repository.head = commit.clone();
        let terminal = workflow_checkpoint(
            &request,
            WorkflowCounters::default(),
            Some(WorkflowOutcome::Ready),
            Some(&commit),
        );
        run.apply(MultiTaskEvent::ChildCheckpointed {
            task_id: "t1".to_string(),
            child: TaskChildCheckpoint::Build(terminal),
            repository: terminal_repository.clone(),
            now_ms: 30,
        })
        .unwrap();
        run.apply(MultiTaskEvent::TaskDelivered {
            task_id: "t1".to_string(),
            result: TaskResult {
                base_repository: request.base_repository,
                terminal_repository: terminal_repository.clone(),
                commits: vec![commit],
                no_change: false,
                satisfied_acceptance_ids: vec!["a1".to_string()],
                evidence_refs: vec![format!("workflow:{}", request.child_id)],
                summary: "The first outcome is committed".to_string(),
            },
            repository: terminal_repository.clone(),
            now_ms: 31,
        })
        .unwrap();
        assert_eq!(run.current_task("t1").unwrap().state, TaskState::Committed);
        assert_eq!(run.expected_repository, terminal_repository);
        assert_eq!(run.stage, MultiTaskStage::Evaluating);
    }

    #[test]
    fn budget_exhaustion_preserves_usage_and_never_starts_a_dependent() {
        let mut run = run();
        start_child(
            &mut run,
            WorkflowCounters {
                model_invocations: 41,
                generated_tokens: 1_000,
                ..WorkflowCounters::default()
            },
            20,
        );
        assert_eq!(run.stage, MultiTaskStage::Failed);
        assert_eq!(run.outcome, Some(MultiTaskOutcome::BudgetExhausted));
        assert_eq!(run.current_task("t1").unwrap().state, TaskState::Failed);
        assert_eq!(
            run.current_task("t1").unwrap().counters.model_invocations,
            41
        );
        assert_eq!(run.current_task("t2").unwrap().state, TaskState::Queued);
        assert!(run.current_task("t2").unwrap().request.is_none());
        MultiTaskCheckpoint::new(run).unwrap();
    }

    #[test]
    fn retrying_a_blocked_task_keeps_prior_usage_watermarks() {
        let mut run = run();
        start_child(
            &mut run,
            WorkflowCounters {
                model_invocations: 2,
                ..WorkflowCounters::default()
            },
            20,
        );
        run.apply(MultiTaskEvent::TaskStopped {
            task_id: "t1".to_string(),
            disposition: TaskStopDisposition::Blocked,
            reason: "executor temporarily unavailable".to_string(),
            now_ms: 30,
        })
        .unwrap();
        let checkpointed_repository = run
            .active_task()
            .unwrap()
            .repository_checkpoint
            .clone()
            .unwrap();
        run.apply(MultiTaskEvent::RetryBlockedTask {
            repository: checkpointed_repository,
            now_ms: 40,
        })
        .unwrap();
        assert_eq!(active_request(&run).attempt, 2);
        assert_eq!(run.active_task().unwrap().counters.model_invocations, 2);
        assert_eq!(run.active_task().unwrap().child_watermarks.len(), 1);
        start_child(
            &mut run,
            WorkflowCounters {
                model_invocations: 1,
                ..WorkflowCounters::default()
            },
            50,
        );
        assert_eq!(run.stage, MultiTaskStage::Failed);
        assert_eq!(run.outcome, Some(MultiTaskOutcome::BudgetExhausted));
        let task = run.current_task("t1").unwrap();
        assert_eq!(task.counters.model_invocations, 3);
        assert_eq!(task.counters.workflows, 2);
        assert_eq!(task.child_watermarks.len(), 2);
    }

    #[test]
    fn compatible_revision_preserves_completed_history_and_supersedes_pending_spec() {
        let mut run = run();
        start_child(&mut run, WorkflowCounters::default(), 20);
        deliver_no_change(&mut run, 30);
        let completed = run.current_task("t1").unwrap().clone();
        let policy = run.policy.clone();
        let (plan, review) = reviewed_plan(&[], &policy, "2");
        run.apply(MultiTaskEvent::PendingTasksRevised {
            plan,
            review,
            reason: "The first delivery removed the second Task's path assumption".to_string(),
            user_approved_expansion: false,
            repository: repository("base"),
            now_ms: 40,
        })
        .unwrap();
        assert_eq!(run.plan_revision, 2);
        assert_eq!(run.current_task("t1").unwrap(), &completed);
        assert!(run.current_task("t2").unwrap().spec.scope_hints.is_empty());
        assert_eq!(
            run.tasks
                .iter()
                .filter(|task| task.spec.id == "t2" && task.state == TaskState::Superseded)
                .count(),
            1
        );
    }

    #[test]
    fn automatic_revision_cannot_expand_pending_scope() {
        let mut run = run();
        start_child(&mut run, WorkflowCounters::default(), 20);
        deliver_no_change(&mut run, 30);
        let policy = run.policy.clone();
        let (plan, review) = reviewed_plan(&["src/second", "docs"], &policy, "expanded");
        assert!(
            run.apply(MultiTaskEvent::PendingTasksRevised {
                plan,
                review,
                reason: "Add a new documentation scope".to_string(),
                user_approved_expansion: false,
                repository: repository("base"),
                now_ms: 40,
            })
            .is_err()
        );
        assert_eq!(run.plan_revision, 1);
    }

    #[test]
    fn coordination_revision_attempts_share_the_original_allowance() {
        let mut run = run();
        start_child(&mut run, WorkflowCounters::default(), 20);
        deliver_no_change(&mut run, 30);
        run.apply(MultiTaskEvent::CoordinationUsageRecorded {
            usage: TaskCoordinationCounters {
                planning_attempts: 2,
                model_invocations: 1,
                generated_tokens: 100,
                advisory_calls: 0,
                elapsed_ms: 100,
            },
            now_ms: 40,
        })
        .unwrap();
        assert_eq!(run.stage, MultiTaskStage::Failed);
        assert_eq!(run.outcome, Some(MultiTaskOutcome::BudgetExhausted));
        assert_eq!(run.counters.coordination.planning_attempts, 3);
        assert_eq!(run.current_task("t1").unwrap().state, TaskState::NoChange);
        assert_eq!(run.current_task("t2").unwrap().state, TaskState::Queued);
    }

    #[test]
    fn resume_requires_the_exact_last_child_repository_checkpoint() {
        let mut run = run();
        start_child(&mut run, WorkflowCounters::default(), 20);
        run.apply(MultiTaskEvent::PauseRequested { now_ms: 30 })
            .unwrap();
        assert!(
            run.apply(MultiTaskEvent::ResumeRequested {
                repository: repository("stale"),
                now_ms: 40,
            })
            .is_err()
        );
        run.apply(MultiTaskEvent::ResumeRequested {
            repository: repository("base"),
            now_ms: 41,
        })
        .unwrap();
        assert_eq!(run.stage, MultiTaskStage::RunningTask);
        assert_eq!(run.active_task_id.as_deref(), Some("t1"));
    }

    #[test]
    fn cancellation_preserves_completed_task_history() {
        let mut run = run();
        start_child(&mut run, WorkflowCounters::default(), 20);
        deliver_no_change(&mut run, 30);
        run.apply(MultiTaskEvent::EvaluationCompleted {
            repository: repository("base"),
            now_ms: 40,
        })
        .unwrap();
        run.apply(MultiTaskEvent::Cancelled {
            reason: "user stopped the project".to_string(),
            now_ms: 50,
        })
        .unwrap();
        assert_eq!(run.stage, MultiTaskStage::Cancelled);
        assert_eq!(run.current_task("t1").unwrap().state, TaskState::NoChange);
        assert_eq!(run.current_task("t2").unwrap().state, TaskState::Cancelled);
        MultiTaskCheckpoint::new(run).unwrap();
    }

    #[test]
    fn checkpoint_digest_detects_task_state_tampering() {
        let mut checkpoint = MultiTaskCheckpoint::new(run()).unwrap();
        checkpoint.run.tasks[0].spec.title = "tampered".to_string();
        assert!(checkpoint.validate().is_err());
    }
}
