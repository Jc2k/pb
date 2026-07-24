use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{CompiledTaskPolicy, MultiTaskBudget, TaskBudget, TaskCoordinationBudget};

pub const TASK_PLAN_VERSION: u32 = 1;
const MAX_TASK_PLAN_PROPOSAL_BYTES: usize = 256 * 1024;
const MAX_TASK_ID_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskEffort {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Build,
    Goal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskSourceIntent {
    Build,
    Goal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlanAuthority {
    pub source_intent: TaskSourceIntent,
    pub task_planning_qualified: bool,
    pub automatic_goal_selection_qualified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlanProposal {
    pub objective: String,
    pub requirements: Vec<TaskRequirement>,
    pub tasks: Vec<TaskProposal>,
    pub acceptance: Vec<TaskAcceptance>,
    #[serde(default)]
    pub risks: Vec<TaskRisk>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskRequirement {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskProposal {
    pub id: String,
    pub title: String,
    pub description: String,
    pub requirement_ids: Vec<String>,
    #[serde(default)]
    pub depends_on: Vec<String>,
    pub acceptance_ids: Vec<String>,
    #[serde(default)]
    pub scope_hints: Vec<String>,
    pub effort: TaskEffort,
    pub kind: TaskKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_contract: Option<TaskGoalContract>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskGoalContract {
    pub objective: String,
    pub criteria: Vec<TaskGoalCriterion>,
    pub continuation: crate::goal::GoalContinuationPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskGoalCriterion {
    pub text: String,
    #[serde(default)]
    pub verifier: crate::goal::GoalVerifier,
}

impl From<&TaskGoalCriterion> for crate::goal::GoalCriterionInput {
    fn from(value: &TaskGoalCriterion) -> Self {
        Self {
            text: value.text.clone(),
            verifier: value.verifier,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskAcceptance {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskRisk {
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlanArtifact {
    pub version: u32,
    pub objective: String,
    pub requirements: Vec<TaskRequirement>,
    pub tasks: Vec<TaskSpec>,
    pub acceptance: Vec<TaskAcceptance>,
    pub risks: Vec<TaskRisk>,
    pub allocated_budget: MultiTaskBudget,
    pub coordination_budget: TaskCoordinationBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskSpec {
    pub id: String,
    pub title: String,
    pub description: String,
    pub requirement_ids: Vec<String>,
    pub depends_on: Vec<String>,
    pub acceptance_ids: Vec<String>,
    pub scope_hints: Vec<String>,
    pub effort: TaskEffort,
    pub kind: TaskKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_contract: Option<TaskGoalContract>,
    pub budget: TaskBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskDispatch {
    Build { task: TaskSpec },
    Goal { task: TaskSpec },
    MultiTask,
}

impl TaskPlanProposal {
    pub fn from_model_json(text: &str) -> Result<Self, TaskPlanError> {
        if text.len() > MAX_TASK_PLAN_PROPOSAL_BYTES {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::SchemaInvalid,
                format!(
                    "Task-plan proposal is {} bytes; maximum is {}",
                    text.len(),
                    MAX_TASK_PLAN_PROPOSAL_BYTES
                ),
            ));
        }
        let value: serde_json::Value = serde_json::from_str(text).map_err(|error| {
            TaskPlanError::new(TaskPlanErrorCode::SchemaInvalid, error.to_string())
        })?;
        if value
            .get("tasks")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|tasks| tasks.iter().any(|task| task.get("budget").is_some()))
        {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::ModelAuthoredBudget,
                "model Task proposals cannot contain executable numeric budgets",
            ));
        }
        serde_json::from_value(value).map_err(|error| {
            TaskPlanError::new(TaskPlanErrorCode::SchemaInvalid, error.to_string())
        })
    }

    pub fn validate_and_compile(
        &self,
        authority: TaskPlanAuthority,
        policy: &CompiledTaskPolicy,
    ) -> Result<TaskPlanArtifact, TaskPlanError> {
        policy.validate().map_err(|error| {
            TaskPlanError::new(TaskPlanErrorCode::PolicyInvalid, error.to_string())
        })?;
        validate_proposal(self, authority, policy)?;

        let mut allocated: Option<TaskBudget> = None;
        let tasks = self
            .tasks
            .iter()
            .map(|task| {
                let budget = policy.budget_for(task.effort);
                if task
                    .goal_contract
                    .as_ref()
                    .is_some_and(|contract| contract.criteria.len() > budget.max_workflows)
                {
                    return Err(TaskPlanError::new(
                        TaskPlanErrorCode::InvalidGoalContract,
                        format!(
                            "Goal Task '{}' has more criteria than its workflow allowance",
                            task.id
                        ),
                    ));
                }
                allocated = Some(match allocated {
                    Some(total) => total.checked_add(budget).map_err(|error| {
                        TaskPlanError::new(TaskPlanErrorCode::AggregateBudget, error.to_string())
                    })?,
                    None => budget,
                });
                Ok(TaskSpec {
                    id: task.id.clone(),
                    title: task.title.clone(),
                    description: task.description.clone(),
                    requirement_ids: task.requirement_ids.clone(),
                    depends_on: task.depends_on.clone(),
                    acceptance_ids: task.acceptance_ids.clone(),
                    scope_hints: task.scope_hints.clone(),
                    effort: task.effort,
                    kind: task.kind,
                    goal_contract: task.goal_contract.clone(),
                    budget,
                })
            })
            .collect::<Result<Vec<_>, TaskPlanError>>()?;
        let allocated = allocated.expect("validated Task plan is non-empty");
        if !allocated.fits_within(&policy.aggregate.tasks) {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::AggregateBudget,
                "compiled Task allocations exceed the aggregate Task budget",
            ));
        }
        Ok(TaskPlanArtifact {
            version: TASK_PLAN_VERSION,
            objective: self.objective.clone(),
            requirements: self.requirements.clone(),
            tasks,
            acceptance: self.acceptance.clone(),
            risks: self.risks.clone(),
            allocated_budget: MultiTaskBudget {
                max_tasks: self.tasks.len(),
                tasks: allocated,
            },
            coordination_budget: policy.coordination,
        })
    }
}

impl TaskPlanArtifact {
    pub fn validate(
        &self,
        authority: TaskPlanAuthority,
        policy: &CompiledTaskPolicy,
    ) -> Result<(), TaskPlanError> {
        if self.version != TASK_PLAN_VERSION {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::VersionUnsupported,
                format!(
                    "unsupported Task plan version {}; expected {}",
                    self.version, TASK_PLAN_VERSION
                ),
            ));
        }
        let proposal = TaskPlanProposal {
            objective: self.objective.clone(),
            requirements: self.requirements.clone(),
            tasks: self
                .tasks
                .iter()
                .map(|task| TaskProposal {
                    id: task.id.clone(),
                    title: task.title.clone(),
                    description: task.description.clone(),
                    requirement_ids: task.requirement_ids.clone(),
                    depends_on: task.depends_on.clone(),
                    acceptance_ids: task.acceptance_ids.clone(),
                    scope_hints: task.scope_hints.clone(),
                    effort: task.effort,
                    kind: task.kind,
                    goal_contract: task.goal_contract.clone(),
                })
                .collect(),
            acceptance: self.acceptance.clone(),
            risks: self.risks.clone(),
        };
        let expected = proposal.validate_and_compile(authority, policy)?;
        if &expected != self {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::BudgetTampered,
                "Task plan budgets do not match controller projection",
            ));
        }
        Ok(())
    }

    pub fn dispatch(&self) -> Result<TaskDispatch, TaskPlanError> {
        if self.tasks.len() > 1 {
            return Ok(TaskDispatch::MultiTask);
        }
        let task = self.tasks.first().cloned().ok_or_else(|| {
            TaskPlanError::new(
                TaskPlanErrorCode::EmptyCollection,
                "Task plan has no Tasks to dispatch",
            )
        })?;
        Ok(match task.kind {
            TaskKind::Build => TaskDispatch::Build { task },
            TaskKind::Goal => TaskDispatch::Goal { task },
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskPlanReviewVerdict {
    Pass,
    Revise,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskPlanChallengeSeverity {
    Blocking,
    Advisory,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TaskPlanAuditCategory {
    RequestCoverage,
    TaskBoundaries,
    DependencyOrder,
    AcceptanceObservability,
    TestDocumentationOwnership,
    EffortGoalAuthority,
}

pub const REQUIRED_TASK_PLAN_AUDIT_CATEGORIES: [TaskPlanAuditCategory; 6] = [
    TaskPlanAuditCategory::RequestCoverage,
    TaskPlanAuditCategory::TaskBoundaries,
    TaskPlanAuditCategory::DependencyOrder,
    TaskPlanAuditCategory::AcceptanceObservability,
    TaskPlanAuditCategory::TestDocumentationOwnership,
    TaskPlanAuditCategory::EffortGoalAuthority,
];

#[cfg(test)]
pub(crate) fn passing_task_plan_audits() -> Vec<TaskPlanAudit> {
    REQUIRED_TASK_PLAN_AUDIT_CATEGORIES
        .into_iter()
        .map(|category| TaskPlanAudit {
            category,
            verdict: TaskPlanAuditVerdict::Pass,
            detail: format!("{category:?} passed"),
            task_ids: Vec::new(),
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskPlanAuditVerdict {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlanAudit {
    pub category: TaskPlanAuditCategory,
    pub verdict: TaskPlanAuditVerdict,
    pub detail: String,
    #[serde(default)]
    pub task_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlanChallenge {
    pub id: String,
    pub code: String,
    pub description: String,
    pub severity: TaskPlanChallengeSeverity,
    #[serde(default)]
    pub task_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlanReviewArtifact {
    pub task_plan_sha256: String,
    pub verdict: TaskPlanReviewVerdict,
    #[serde(default)]
    pub audits: Vec<TaskPlanAudit>,
    #[serde(default)]
    pub challenges: Vec<TaskPlanChallenge>,
}

impl TaskPlanReviewArtifact {
    pub fn validate(
        &self,
        plan: &crate::workflow::ArtifactEnvelope<TaskPlanArtifact>,
    ) -> Result<(), TaskPlanError> {
        plan.validate_digest().map_err(|error| {
            TaskPlanError::new(TaskPlanErrorCode::DigestMismatch, error.to_string())
        })?;
        if self.task_plan_sha256 != plan.sha256 {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::DigestMismatch,
                "Task-plan review does not target the current plan digest",
            ));
        }
        let task_ids = plan
            .artifact
            .tasks
            .iter()
            .map(|task| task.id.as_str())
            .collect::<HashSet<_>>();
        let required_audits = REQUIRED_TASK_PLAN_AUDIT_CATEGORIES
            .into_iter()
            .collect::<HashSet<_>>();
        let mut audit_categories = HashSet::new();
        for audit in &self.audits {
            required("Task-plan audit detail", &audit.detail)?;
            if !audit_categories.insert(audit.category) {
                return Err(TaskPlanError::new(
                    TaskPlanErrorCode::ReviewInvalid,
                    format!(
                        "Task-plan review repeats audit category {:?}",
                        audit.category
                    ),
                ));
            }
            for task_id in &audit.task_ids {
                if !task_ids.contains(task_id.as_str()) {
                    return Err(TaskPlanError::new(
                        TaskPlanErrorCode::UnknownTask,
                        format!("Task-plan audit references unknown Task '{task_id}'"),
                    ));
                }
            }
        }
        if audit_categories != required_audits {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::ReviewInvalid,
                "Task-plan review must contain every required audit category exactly once",
            ));
        }
        let mut challenge_ids = HashSet::new();
        for challenge in &self.challenges {
            required("Task-plan challenge id", &challenge.id)?;
            required("Task-plan challenge code", &challenge.code)?;
            required("Task-plan challenge description", &challenge.description)?;
            if !challenge_ids.insert(challenge.id.as_str()) {
                return Err(TaskPlanError::new(
                    TaskPlanErrorCode::DuplicateId,
                    format!("duplicate Task-plan challenge id '{}'", challenge.id),
                ));
            }
            for task_id in &challenge.task_ids {
                if !task_ids.contains(task_id.as_str()) {
                    return Err(TaskPlanError::new(
                        TaskPlanErrorCode::UnknownTask,
                        format!("Task-plan challenge references unknown Task '{task_id}'"),
                    ));
                }
            }
        }
        let blocking = self
            .challenges
            .iter()
            .any(|challenge| challenge.severity == TaskPlanChallengeSeverity::Blocking);
        let failed_audit = self
            .audits
            .iter()
            .any(|audit| audit.verdict == TaskPlanAuditVerdict::Fail);
        if self.verdict == TaskPlanReviewVerdict::Pass && (blocking || failed_audit) {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::ReviewInvalid,
                "Task-plan review cannot pass with a failed audit or blocking challenge",
            ));
        }
        if self.verdict == TaskPlanReviewVerdict::Revise && (!blocking || !failed_audit) {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::ReviewInvalid,
                "Task-plan revision requires a failed audit and at least one blocking challenge",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskPlanErrorCode {
    SchemaInvalid,
    ModelAuthoredBudget,
    VersionUnsupported,
    PolicyInvalid,
    EmptyField,
    EmptyCollection,
    DuplicateId,
    DuplicateReference,
    UnknownRequirement,
    UnknownAcceptance,
    UnknownDependency,
    UnknownTask,
    DependencyCycle,
    UncoveredRequirement,
    UnmappedAcceptance,
    TaskLimit,
    AggregateBudget,
    InvalidGoalContract,
    GoalSelectionUnqualified,
    ExplicitGoalShape,
    BudgetTampered,
    DigestMismatch,
    ReviewInvalid,
}

impl TaskPlanErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SchemaInvalid => "schema_invalid",
            Self::ModelAuthoredBudget => "model_authored_budget",
            Self::VersionUnsupported => "version_unsupported",
            Self::PolicyInvalid => "policy_invalid",
            Self::EmptyField => "empty_field",
            Self::EmptyCollection => "empty_collection",
            Self::DuplicateId => "duplicate_id",
            Self::DuplicateReference => "duplicate_reference",
            Self::UnknownRequirement => "unknown_requirement",
            Self::UnknownAcceptance => "unknown_acceptance",
            Self::UnknownDependency => "unknown_dependency",
            Self::UnknownTask => "unknown_task",
            Self::DependencyCycle => "dependency_cycle",
            Self::UncoveredRequirement => "uncovered_requirement",
            Self::UnmappedAcceptance => "unmapped_acceptance",
            Self::TaskLimit => "task_limit",
            Self::AggregateBudget => "aggregate_budget",
            Self::InvalidGoalContract => "invalid_goal_contract",
            Self::GoalSelectionUnqualified => "goal_selection_unqualified",
            Self::ExplicitGoalShape => "explicit_goal_shape",
            Self::BudgetTampered => "budget_tampered",
            Self::DigestMismatch => "digest_mismatch",
            Self::ReviewInvalid => "review_invalid",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Error)]
#[error("{code}: {message}")]
pub struct TaskPlanError {
    pub code: TaskPlanErrorCode,
    pub message: String,
}

impl TaskPlanError {
    fn new(code: TaskPlanErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for TaskPlanErrorCode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

fn validate_proposal(
    proposal: &TaskPlanProposal,
    authority: TaskPlanAuthority,
    policy: &CompiledTaskPolicy,
) -> Result<(), TaskPlanError> {
    let proposal_bytes = serde_json::to_vec(proposal)
        .map_err(|error| TaskPlanError::new(TaskPlanErrorCode::SchemaInvalid, error.to_string()))?;
    if proposal_bytes.len() > MAX_TASK_PLAN_PROPOSAL_BYTES {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::SchemaInvalid,
            format!(
                "Task-plan proposal is {} bytes; maximum is {}",
                proposal_bytes.len(),
                MAX_TASK_PLAN_PROPOSAL_BYTES
            ),
        ));
    }
    if !authority.task_planning_qualified {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::PolicyInvalid,
            "planner is not qualified for Task decomposition",
        ));
    }
    required("Task-plan objective", &proposal.objective)?;
    if proposal.requirements.is_empty()
        || proposal.tasks.is_empty()
        || proposal.acceptance.is_empty()
    {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::EmptyCollection,
            "Task plan requires requirements, Tasks, and acceptance facts",
        ));
    }
    if proposal.tasks.len() > policy.aggregate.max_tasks {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::TaskLimit,
            format!(
                "Task plan contains {} Tasks; maximum is {}",
                proposal.tasks.len(),
                policy.aggregate.max_tasks
            ),
        ));
    }

    let requirement_ids = unique_ids(
        "Task requirement",
        proposal.requirements.iter().map(|item| item.id.as_str()),
    )?;
    let acceptance_ids = unique_ids(
        "Task acceptance",
        proposal.acceptance.iter().map(|item| item.id.as_str()),
    )?;
    let task_ids = unique_ids("Task", proposal.tasks.iter().map(|item| item.id.as_str()))?;
    for requirement in &proposal.requirements {
        required("Task requirement description", &requirement.description)?;
    }
    for acceptance in &proposal.acceptance {
        required("Task acceptance description", &acceptance.description)?;
    }
    for risk in &proposal.risks {
        required("Task risk description", &risk.description)?;
    }

    let mut covered_requirements = HashSet::new();
    let mut mapped_acceptance = HashSet::new();
    let mut dependencies = BTreeMap::<&str, Vec<&str>>::new();
    let mut contains_goal = false;
    for task in &proposal.tasks {
        required("Task title", &task.title)?;
        required("Task description", &task.description)?;
        for scope_hint in &task.scope_hints {
            required("Task scope hint", scope_hint)?;
        }
        if task.requirement_ids.is_empty() || task.acceptance_ids.is_empty() {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::EmptyCollection,
                format!(
                    "Task '{}' must reference requirements and acceptance facts",
                    task.id
                ),
            ));
        }
        validate_unique_references("Task requirement", &task.id, &task.requirement_ids)?;
        validate_unique_references("Task acceptance", &task.id, &task.acceptance_ids)?;
        validate_unique_references("Task dependency", &task.id, &task.depends_on)?;
        for requirement_id in &task.requirement_ids {
            if !requirement_ids.contains(requirement_id.as_str()) {
                return Err(TaskPlanError::new(
                    TaskPlanErrorCode::UnknownRequirement,
                    format!(
                        "Task '{}' references unknown requirement '{requirement_id}'",
                        task.id
                    ),
                ));
            }
            covered_requirements.insert(requirement_id.as_str());
        }
        for acceptance_id in &task.acceptance_ids {
            if !acceptance_ids.contains(acceptance_id.as_str()) {
                return Err(TaskPlanError::new(
                    TaskPlanErrorCode::UnknownAcceptance,
                    format!(
                        "Task '{}' references unknown acceptance fact '{acceptance_id}'",
                        task.id
                    ),
                ));
            }
            mapped_acceptance.insert(acceptance_id.as_str());
        }
        for dependency in &task.depends_on {
            if !task_ids.contains(dependency.as_str()) {
                return Err(TaskPlanError::new(
                    TaskPlanErrorCode::UnknownDependency,
                    format!(
                        "Task '{}' references unknown dependency '{dependency}'",
                        task.id
                    ),
                ));
            }
        }
        dependencies.insert(
            task.id.as_str(),
            task.depends_on.iter().map(String::as_str).collect(),
        );
        match (task.kind, &task.goal_contract) {
            (TaskKind::Build, None) => {}
            (TaskKind::Build, Some(_)) => {
                return Err(TaskPlanError::new(
                    TaskPlanErrorCode::InvalidGoalContract,
                    format!("Build Task '{}' contains a Goal contract", task.id),
                ));
            }
            (TaskKind::Goal, None) => {
                return Err(TaskPlanError::new(
                    TaskPlanErrorCode::InvalidGoalContract,
                    format!("Goal Task '{}' has no Goal contract", task.id),
                ));
            }
            (TaskKind::Goal, Some(contract)) => {
                contains_goal = true;
                validate_goal_contract(&task.id, contract)?;
            }
        }
    }

    if let Some(id) = requirement_ids
        .iter()
        .find(|id| !covered_requirements.contains(**id))
    {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::UncoveredRequirement,
            format!("Task requirement '{id}' is not covered by any Task"),
        ));
    }
    if let Some(id) = acceptance_ids
        .iter()
        .find(|id| !mapped_acceptance.contains(**id))
    {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::UnmappedAcceptance,
            format!("Task acceptance fact '{id}' is not mapped to any Task"),
        ));
    }
    validate_acyclic(&dependencies)?;

    if authority.source_intent == TaskSourceIntent::Goal
        && (proposal.tasks.len() != 1 || proposal.tasks[0].kind != TaskKind::Goal)
    {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::ExplicitGoalShape,
            "explicit Goal mode must unwrap through exactly one Goal Task",
        ));
    }
    if contains_goal
        && authority.source_intent != TaskSourceIntent::Goal
        && !authority.automatic_goal_selection_qualified
    {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::GoalSelectionUnqualified,
            "planner is not qualified to select Goal Tasks automatically",
        ));
    }
    Ok(())
}

fn validate_goal_contract(task_id: &str, contract: &TaskGoalContract) -> Result<(), TaskPlanError> {
    required("Goal Task objective", &contract.objective)?;
    if contract.criteria.is_empty() {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::InvalidGoalContract,
            format!("Goal Task '{task_id}' has no criteria"),
        ));
    }
    let mut criteria = HashSet::new();
    for criterion in &contract.criteria {
        required("Goal Task criterion", &criterion.text)?;
        if !criteria.insert(criterion.text.trim()) {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::InvalidGoalContract,
                format!("Goal Task '{task_id}' contains a duplicate criterion"),
            ));
        }
    }
    Ok(())
}

fn validate_acyclic(dependencies: &BTreeMap<&str, Vec<&str>>) -> Result<(), TaskPlanError> {
    let mut indegree = dependencies
        .iter()
        .map(|(task, values)| (*task, values.len()))
        .collect::<BTreeMap<_, _>>();
    let mut dependents = BTreeMap::<&str, Vec<&str>>::new();
    for (task, values) in dependencies {
        for dependency in values {
            dependents.entry(dependency).or_default().push(task);
        }
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(task, degree)| (*degree == 0).then_some(*task))
        .collect::<VecDeque<_>>();
    let mut visited = 0;
    while let Some(task) = ready.pop_front() {
        visited += 1;
        for dependent in dependents.get(task).into_iter().flatten() {
            let degree = indegree
                .get_mut(dependent)
                .expect("validated dependent belongs to graph");
            *degree -= 1;
            if *degree == 0 {
                ready.push_back(dependent);
            }
        }
    }
    if visited != dependencies.len() {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::DependencyCycle,
            "Task dependency graph contains a cycle",
        ));
    }
    Ok(())
}

fn unique_ids<'a>(
    label: &str,
    values: impl Iterator<Item = &'a str>,
) -> Result<BTreeSet<&'a str>, TaskPlanError> {
    let mut ids = BTreeSet::new();
    for value in values {
        validate_id(label, value)?;
        if !ids.insert(value) {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::DuplicateId,
                format!("duplicate {label} id '{value}'"),
            ));
        }
    }
    Ok(ids)
}

fn validate_id(label: &str, value: &str) -> Result<(), TaskPlanError> {
    required(label, value)?;
    if value.len() > MAX_TASK_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::SchemaInvalid,
            format!(
                "{label} id must be at most {MAX_TASK_ID_BYTES} bytes of ASCII letters, digits, '-', '_', '.', or ':'"
            ),
        ));
    }
    Ok(())
}

fn validate_unique_references(
    label: &str,
    owner: &str,
    values: &[String],
) -> Result<(), TaskPlanError> {
    let mut seen = HashSet::new();
    for value in values {
        if !seen.insert(value.as_str()) {
            return Err(TaskPlanError::new(
                TaskPlanErrorCode::DuplicateReference,
                format!("{label} list for '{owner}' contains duplicate '{value}'"),
            ));
        }
    }
    Ok(())
}

fn required(label: &str, value: &str) -> Result<(), TaskPlanError> {
    if value.trim().is_empty() {
        return Err(TaskPlanError::new(
            TaskPlanErrorCode::EmptyField,
            format!("{label} must not be empty"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Corpus {
        cases: Vec<CorpusCase>,
    }

    #[derive(Deserialize)]
    struct CorpusCase {
        id: String,
        category: String,
        context: CorpusContext,
        attempts: Vec<serde_json::Value>,
        expected: CorpusExpected,
    }

    #[derive(Deserialize)]
    struct CorpusContext {
        intent: String,
        goal_selection_qualified: bool,
    }

    #[derive(Deserialize)]
    struct CorpusExpected {
        outcome: String,
        reason: String,
        dispatch: Option<String>,
    }

    fn authority(context: &CorpusContext) -> TaskPlanAuthority {
        TaskPlanAuthority {
            source_intent: match context.intent.as_str() {
                "goal" => TaskSourceIntent::Goal,
                _ => TaskSourceIntent::Build,
            },
            task_planning_qualified: true,
            automatic_goal_selection_qualified: context.goal_selection_qualified,
        }
    }

    #[test]
    fn locked_corpus_matches_production_validator_and_dispatch() {
        let corpus: Corpus = serde_json::from_str(include_str!(
            "../../fixtures/task-decomposition/corpus.json"
        ))
        .unwrap();
        let policy = crate::task_queue::TaskConfigDocument::default()
            .compile()
            .unwrap();
        for case in corpus.cases {
            if case.category == "semantic" {
                let proposal: TaskPlanProposal =
                    serde_json::from_value(case.attempts[0].clone()).unwrap();
                proposal
                    .validate_and_compile(authority(&case.context), &policy)
                    .unwrap_or_else(|error| panic!("{} should reach review: {error}", case.id));
                assert_eq!(case.expected.outcome, "review_rejected");
                continue;
            }
            if case.id == "two_invalid_attempts" {
                for attempt in case.attempts {
                    let proposal: TaskPlanProposal = serde_json::from_value(attempt).unwrap();
                    assert!(
                        proposal
                            .validate_and_compile(authority(&case.context), &policy)
                            .is_err()
                    );
                }
                assert_eq!(case.expected.outcome, "task_plan_rejected");
                continue;
            }
            let text = serde_json::to_string(&case.attempts[0]).unwrap();
            let result = TaskPlanProposal::from_model_json(&text).and_then(|proposal| {
                proposal.validate_and_compile(authority(&case.context), &policy)
            });
            match case.expected.outcome.as_str() {
                "accepted" => {
                    let artifact = result.unwrap_or_else(|error| panic!("{}: {error}", case.id));
                    let dispatch = match artifact.dispatch().unwrap() {
                        TaskDispatch::Build { .. } => "build",
                        TaskDispatch::Goal { .. } => "goal",
                        TaskDispatch::MultiTask => "multi_task",
                    };
                    assert_eq!(
                        case.expected.dispatch.as_deref(),
                        Some(dispatch),
                        "{}",
                        case.id
                    );
                }
                "validation_rejected" => {
                    let error = result.expect_err(&case.id);
                    assert_eq!(error.code.as_str(), case.expected.reason, "{}", case.id);
                }
                other => panic!("unsupported corpus outcome {other}"),
            }
        }
    }

    #[test]
    fn accepted_artifact_detects_controller_budget_tampering() {
        let policy = crate::task_queue::TaskConfigDocument::default()
            .compile()
            .unwrap();
        let proposal: TaskPlanProposal = serde_json::from_value(serde_json::json!({
            "objective":"Fix average",
            "requirements":[{"id":"r1","description":"Fix it"}],
            "tasks":[{"id":"t1","title":"Fix","description":"Fix it","requirement_ids":["r1"],"acceptance_ids":["a1"],"effort":"small","kind":"build"}],
            "acceptance":[{"id":"a1","description":"Tests pass"}]
        }))
        .unwrap();
        let authority = TaskPlanAuthority {
            source_intent: TaskSourceIntent::Build,
            task_planning_qualified: true,
            automatic_goal_selection_qualified: false,
        };
        let mut artifact = proposal.validate_and_compile(authority, &policy).unwrap();
        artifact.tasks[0].budget.total_generated_tokens += 1;
        assert_eq!(
            artifact.validate(authority, &policy).unwrap_err().code,
            TaskPlanErrorCode::BudgetTampered
        );
    }

    #[test]
    fn review_is_bound_to_plan_digest_and_blocking_challenges() {
        let policy = crate::task_queue::TaskConfigDocument::default()
            .compile()
            .unwrap();
        let proposal: TaskPlanProposal = serde_json::from_value(serde_json::json!({
            "objective":"Fix average",
            "requirements":[{"id":"r1","description":"Fix it"}],
            "tasks":[{"id":"t1","title":"Fix","description":"Fix it","requirement_ids":["r1"],"acceptance_ids":["a1"],"effort":"small","kind":"build"}],
            "acceptance":[{"id":"a1","description":"Tests pass"}]
        }))
        .unwrap();
        let artifact = proposal
            .validate_and_compile(
                TaskPlanAuthority {
                    source_intent: TaskSourceIntent::Build,
                    task_planning_qualified: true,
                    automatic_goal_selection_qualified: false,
                },
                &policy,
            )
            .unwrap();
        let plan = crate::workflow::ArtifactEnvelope::new("task-plan", artifact).unwrap();
        let review = TaskPlanReviewArtifact {
            task_plan_sha256: plan.sha256.clone(),
            verdict: TaskPlanReviewVerdict::Pass,
            audits: passing_task_plan_audits(),
            challenges: vec![TaskPlanChallenge {
                id: "c1".to_string(),
                code: "catch_all_task".to_string(),
                description: "Task is a catch-all".to_string(),
                severity: TaskPlanChallengeSeverity::Blocking,
                task_ids: vec!["t1".to_string()],
            }],
        };
        assert_eq!(
            review.validate(&plan).unwrap_err().code,
            TaskPlanErrorCode::ReviewInvalid
        );

        let mut missing_audit = review.clone();
        missing_audit.challenges.clear();
        missing_audit.audits.pop();
        assert_eq!(
            missing_audit.validate(&plan).unwrap_err().code,
            TaskPlanErrorCode::ReviewInvalid
        );

        let mut revised = review;
        revised.verdict = TaskPlanReviewVerdict::Revise;
        revised.audits[0].verdict = TaskPlanAuditVerdict::Fail;
        revised.validate(&plan).unwrap();
    }
}
