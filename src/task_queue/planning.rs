use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::workflow::ArtifactEnvelope;

use super::{
    CompiledTaskPolicy, TaskAcceptance, TaskCoordinationCounters, TaskEffort, TaskKind,
    TaskPlanArtifact, TaskPlanAuthority, TaskPlanProposal, TaskPlanReviewArtifact,
    TaskPlanReviewVerdict, TaskPlannerQualification, TaskProposal, TaskRequirement, TaskRisk,
    TaskSourceIntent,
};

const TASK_PLANNER_TEMPLATE_VERSION: &str = "pb-task-planner-template-v10";
const TASK_PLANNER_PROTOCOL_VERSION: &str = "pb-task-plan-proposal-review-v9";
const MAX_TASK_PLANNING_INPUT_BYTES: usize = 64 * 1024;
const MAX_TASK_PLANNER_OUTPUT_TOKENS: usize = 4_096;
const MAX_TASK_REVIEW_OUTPUT_TOKENS: usize = 2_048;
const MAX_RETRY_FEEDBACK_CHARS: usize = 4_000;
const MAX_PLANNING_FACTS: usize = 32;
const MAX_SCOPE_HINTS: usize = 16;
const MAX_RISKS: usize = 16;
const MAX_GOAL_CRITERIA: usize = 16;
const MAX_REVIEW_CHALLENGES: usize = 16;
const MAX_ID_CHARS: usize = 128;
const MAX_TITLE_CHARS: usize = 256;
// llama.cpp's grammar parser rejects bounded repetitions above 2,000. Keep the shared schema
// comfortably below that backend limit so the same contract compiles in llama.cpp and FlashMoe.
const MAX_DESCRIPTION_CHARS: usize = 1_024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskPlanningRole {
    Planner,
    Reviewer,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskModelOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TaskModelProposal {
    tasks: Vec<TaskModelTask>,
    #[serde(default)]
    risks: Vec<TaskRisk>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TaskModelTask {
    title: String,
    description: String,
    request_evidence: Vec<String>,
    acceptance: Vec<String>,
    tests: Vec<String>,
    documentation: Vec<String>,
    #[serde(default)]
    scope_hints: Vec<String>,
    kind: TaskKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    goal_contract: Option<super::TaskGoalContract>,
}

impl TaskModelProposal {
    fn into_controller_proposal(
        self,
        policy: &CompiledTaskPolicy,
        objective: &str,
    ) -> Result<TaskPlanProposal> {
        if self.tasks.len() > 1 {
            for task in &self.tasks {
                if task
                    .request_evidence
                    .iter()
                    .all(|evidence| is_decomposition_constraint(evidence))
                {
                    bail!(
                        "Task '{}' owns only decomposition constraints; tests, documentation, ordering, and final validation must stay with a behavior-owning Task",
                        task.title
                    );
                }
            }
        }
        let request_evidence = request_evidence_clauses(objective);
        let requirements = request_evidence
            .iter()
            .enumerate()
            .map(|(index, description)| TaskRequirement {
                id: format!("req-{:03}", index + 1),
                description: description.clone(),
            })
            .collect::<Vec<_>>();
        let requirement_ids = requirements
            .iter()
            .map(|requirement| (requirement.description.clone(), requirement.id.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut acceptance = Vec::new();
        let mut acceptance_ids = BTreeMap::<String, String>::new();
        let mut tasks = Vec::with_capacity(self.tasks.len());

        for (task_index, task) in self.tasks.into_iter().enumerate() {
            let id = format!("task-{:02}", task_index + 1);
            let mut task_requirement_ids = Vec::new();
            let mut seen_requirements = BTreeSet::new();
            for description in task.request_evidence {
                let description = description.trim();
                let requirement_id = requirement_ids.get(description).with_context(|| {
                    format!(
                        "Task '{}' cites request evidence not supplied by the controller",
                        task.title
                    )
                })?;
                if seen_requirements.insert(requirement_id.clone()) {
                    task_requirement_ids.push(requirement_id.clone());
                }
            }

            let mut task_acceptance_ids = Vec::new();
            let mut seen_acceptance = BTreeSet::new();
            let acceptance_facts = task
                .acceptance
                .into_iter()
                .chain(task.tests.into_iter().map(|fact| format!("Tests: {fact}")))
                .chain(
                    task.documentation
                        .into_iter()
                        .map(|fact| format!("Documentation: {fact}")),
                );
            for description in acceptance_facts {
                let description = description.trim().to_string();
                let acceptance_id = acceptance_ids
                    .entry(description.clone())
                    .or_insert_with(|| {
                        let acceptance_id = format!("accept-{:03}", acceptance.len() + 1);
                        acceptance.push(TaskAcceptance {
                            id: acceptance_id.clone(),
                            description,
                        });
                        acceptance_id
                    })
                    .clone();
                if seen_acceptance.insert(acceptance_id.clone()) {
                    task_acceptance_ids.push(acceptance_id);
                }
            }

            tasks.push(TaskProposal {
                id,
                title: task.title,
                description: task.description,
                requirement_ids: task_requirement_ids,
                depends_on: task_index
                    .checked_sub(1)
                    .map(|previous| vec![format!("task-{:02}", previous + 1)])
                    .unwrap_or_default(),
                acceptance_ids: task_acceptance_ids,
                scope_hints: task.scope_hints,
                effort: match task.kind {
                    TaskKind::Build => TaskEffort::Small,
                    TaskKind::Goal => TaskEffort::Large,
                },
                kind: task.kind,
                goal_contract: task.goal_contract,
            });
        }

        while !model_efforts_fit(&tasks, policy) {
            let candidate = tasks
                .iter()
                .rposition(|task| task.effort == TaskEffort::Large)
                .or_else(|| {
                    tasks
                        .iter()
                        .rposition(|task| task.effort == TaskEffort::Medium)
                });
            let Some(task_index) = candidate else {
                break;
            };
            let task = &mut tasks[task_index];
            task.effort = match task.effort {
                TaskEffort::Large => TaskEffort::Medium,
                TaskEffort::Medium => TaskEffort::Small,
                TaskEffort::Small => unreachable!("small Tasks are not normalization candidates"),
            };
        }

        Ok(TaskPlanProposal {
            objective: objective.to_string(),
            requirements,
            tasks,
            acceptance,
            risks: self.risks,
        })
    }
}

fn is_decomposition_constraint(evidence: &str) -> bool {
    let evidence = evidence.to_ascii_lowercase();
    evidence.contains("decompose this")
        || evidence.contains("behavior-owning task")
        || evidence.contains("behavior owning task")
        || evidence.contains("each task")
        || evidence.contains("every task")
        || evidence.contains("tasks suitable for")
}

fn model_efforts_fit(tasks: &[TaskProposal], policy: &CompiledTaskPolicy) -> bool {
    let Some((first, remaining)) = tasks.split_first() else {
        return false;
    };
    let allocated = remaining
        .iter()
        .try_fold(policy.budget_for(first.effort), |allocated, task| {
            allocated.checked_add(policy.budget_for(task.effort)).ok()
        });
    allocated.is_some_and(|allocated| allocated.fits_within(&policy.aggregate.tasks))
}

pub trait TaskPlanningModel {
    fn generate(
        &mut self,
        role: TaskPlanningRole,
        prompt: &str,
        max_tokens: usize,
        schema: &Value,
    ) -> Result<TaskModelOutput>;

    fn should_cancel(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub struct TaskPlanningInput<'a> {
    pub objective: &'a str,
    pub repository_context: &'a str,
    pub source_intent: TaskSourceIntent,
    pub model_sha256: &'a str,
    pub qualification: &'a TaskPlannerQualification,
    pub policy: &'a CompiledTaskPolicy,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskPlanRecoveryAction {
    RetryPlanning,
    EditRequest,
    RunAsOneBuild,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskPlanRejectionOutcome {
    AttemptsExhausted,
    BudgetExhausted,
    Cancelled,
    QualificationMismatch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlanAttemptFailure {
    pub attempt: usize,
    pub stage: TaskPlanningRole,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlanRejected {
    pub outcome: TaskPlanRejectionOutcome,
    pub attempts: usize,
    pub failures: Vec<TaskPlanAttemptFailure>,
    pub counters: TaskCoordinationCounters,
    pub recovery_actions: Vec<TaskPlanRecoveryAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AcceptedTaskPlan {
    pub plan: ArtifactEnvelope<TaskPlanArtifact>,
    pub review: ArtifactEnvelope<TaskPlanReviewArtifact>,
    pub counters: TaskCoordinationCounters,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum TaskPlanningOutcome {
    Accepted(AcceptedTaskPlan),
    Rejected(TaskPlanRejected),
}

pub fn task_planner_template_sha256() -> String {
    format!("{:x}", Sha256::digest(TASK_PLANNER_TEMPLATE_VERSION))
}

pub fn task_planner_protocol_sha256() -> String {
    format!("{:x}", Sha256::digest(TASK_PLANNER_PROTOCOL_VERSION))
}

pub fn plan_tasks(
    model: &mut dyn TaskPlanningModel,
    input: TaskPlanningInput<'_>,
) -> TaskPlanningOutcome {
    match plan_tasks_inner(model, &input) {
        Ok(outcome) => outcome,
        Err(error) => rejected(
            TaskPlanRejectionOutcome::QualificationMismatch,
            0,
            vec![TaskPlanAttemptFailure {
                attempt: 0,
                stage: TaskPlanningRole::Planner,
                reason: format!("{error:#}"),
            }],
            TaskCoordinationCounters::default(),
        ),
    }
}

fn plan_tasks_inner(
    model: &mut dyn TaskPlanningModel,
    input: &TaskPlanningInput<'_>,
) -> Result<TaskPlanningOutcome> {
    validate_input(input)?;
    let authority = TaskPlanAuthority {
        source_intent: input.source_intent,
        task_planning_qualified: input.qualification.task_planning,
        automatic_goal_selection_qualified: input.qualification.automatic_goal_selection,
    };
    let budget = input.policy.coordination;
    let mut counters = TaskCoordinationCounters::default();
    let mut failures = Vec::new();
    let mut feedback = String::new();

    for attempt in 1..=budget.planning_attempts {
        if model.should_cancel() {
            return Ok(rejected(
                TaskPlanRejectionOutcome::Cancelled,
                counters.planning_attempts,
                failures,
                counters,
            ));
        }
        counters.planning_attempts = counters.planning_attempts.saturating_add(1);
        let prompt = planner_prompt(input, attempt, &feedback);
        let schema = planner_schema(input);
        let remaining_tokens = budget
            .generated_tokens
            .saturating_sub(counters.generated_tokens);
        let max_tokens = remaining_tokens.min(MAX_TASK_PLANNER_OUTPUT_TOKENS);
        if max_tokens == 0 || counters.model_invocations >= budget.model_invocations {
            return Ok(rejected(
                TaskPlanRejectionOutcome::BudgetExhausted,
                counters.planning_attempts,
                failures,
                counters,
            ));
        }
        let started = Instant::now();
        let output = match model.generate(TaskPlanningRole::Planner, &prompt, max_tokens, &schema) {
            Ok(output) => output,
            Err(error) => {
                counters.model_invocations = counters.model_invocations.saturating_add(1);
                counters.elapsed_ms = counters
                    .elapsed_ms
                    .saturating_add(elapsed_ms(started.elapsed()));
                let reason = format!("planner invocation failed: {error:#}");
                failures.push(TaskPlanAttemptFailure {
                    attempt,
                    stage: TaskPlanningRole::Planner,
                    reason: reason.clone(),
                });
                feedback = bounded_feedback(&reason);
                continue;
            }
        };
        record_output(&mut counters, &output);
        if !counters.fits_within(&budget) {
            failures.push(TaskPlanAttemptFailure {
                attempt,
                stage: TaskPlanningRole::Planner,
                reason: "planner output exhausted the Task coordination budget".to_string(),
            });
            return Ok(rejected(
                TaskPlanRejectionOutcome::BudgetExhausted,
                counters.planning_attempts,
                failures,
                counters,
            ));
        }
        let proposal = match parse_json::<TaskModelProposal>(&output.text)
            .and_then(|proposal| proposal.into_controller_proposal(input.policy, input.objective))
            .and_then(|proposal| {
                proposal
                    .validate_and_compile(authority, input.policy)
                    .map_err(anyhow::Error::from)
            }) {
            Ok(plan) => plan,
            Err(error) => {
                let reason = format!("Task proposal rejected: {error:#}");
                failures.push(TaskPlanAttemptFailure {
                    attempt,
                    stage: TaskPlanningRole::Planner,
                    reason: reason.clone(),
                });
                feedback = bounded_feedback(&reason);
                continue;
            }
        };
        let plan = ArtifactEnvelope::new(format!("task-plan-{attempt}"), proposal)?;

        if model.should_cancel() {
            return Ok(rejected(
                TaskPlanRejectionOutcome::Cancelled,
                counters.planning_attempts,
                failures,
                counters,
            ));
        }
        let remaining_tokens = budget
            .generated_tokens
            .saturating_sub(counters.generated_tokens);
        let max_tokens = remaining_tokens.min(MAX_TASK_REVIEW_OUTPUT_TOKENS);
        if max_tokens == 0
            || counters.model_invocations >= budget.model_invocations
            || counters.advisory_calls >= budget.advisory_calls
        {
            return Ok(rejected(
                TaskPlanRejectionOutcome::BudgetExhausted,
                counters.planning_attempts,
                failures,
                counters,
            ));
        }
        let review_prompt = reviewer_prompt(input, &plan)?;
        let review_schema = reviewer_schema(input, &plan);
        let started = Instant::now();
        let output = match model.generate(
            TaskPlanningRole::Reviewer,
            &review_prompt,
            max_tokens,
            &review_schema,
        ) {
            Ok(output) => output,
            Err(error) => {
                counters.model_invocations = counters.model_invocations.saturating_add(1);
                counters.advisory_calls = counters.advisory_calls.saturating_add(1);
                counters.elapsed_ms = counters
                    .elapsed_ms
                    .saturating_add(elapsed_ms(started.elapsed()));
                let reason = format!("Task-plan review invocation failed: {error:#}");
                failures.push(TaskPlanAttemptFailure {
                    attempt,
                    stage: TaskPlanningRole::Reviewer,
                    reason: reason.clone(),
                });
                feedback = bounded_feedback(&reason);
                continue;
            }
        };
        counters.advisory_calls = counters.advisory_calls.saturating_add(1);
        record_output(&mut counters, &output);
        if !counters.fits_within(&budget) {
            failures.push(TaskPlanAttemptFailure {
                attempt,
                stage: TaskPlanningRole::Reviewer,
                reason: "Task-plan review exhausted the coordination budget".to_string(),
            });
            return Ok(rejected(
                TaskPlanRejectionOutcome::BudgetExhausted,
                counters.planning_attempts,
                failures,
                counters,
            ));
        }
        let review = match parse_json::<TaskPlanReviewArtifact>(&output.text).and_then(|review| {
            review.validate(&plan).map_err(anyhow::Error::from)?;
            validate_review_request_evidence(&review, input.objective)?;
            Ok(review)
        }) {
            Ok(review) => review,
            Err(error) => {
                let reason = format!("Task-plan review rejected: {error:#}");
                failures.push(TaskPlanAttemptFailure {
                    attempt,
                    stage: TaskPlanningRole::Reviewer,
                    reason: reason.clone(),
                });
                feedback = bounded_feedback(&reason);
                continue;
            }
        };
        let review_envelope = ArtifactEnvelope::new(format!("task-plan-review-{attempt}"), review)?;
        if review_envelope.artifact.verdict == TaskPlanReviewVerdict::Pass {
            return Ok(TaskPlanningOutcome::Accepted(AcceptedTaskPlan {
                plan,
                review: review_envelope,
                counters,
            }));
        }
        let reason = review_envelope
            .artifact
            .challenges
            .iter()
            .map(|challenge| format!("{}: {}", challenge.code, challenge.description))
            .collect::<Vec<_>>()
            .join("\n");
        failures.push(TaskPlanAttemptFailure {
            attempt,
            stage: TaskPlanningRole::Reviewer,
            reason: reason.clone(),
        });
        feedback = bounded_feedback(&reason);
    }

    Ok(rejected(
        TaskPlanRejectionOutcome::AttemptsExhausted,
        counters.planning_attempts,
        failures,
        counters,
    ))
}

fn validate_input(input: &TaskPlanningInput<'_>) -> Result<()> {
    input.policy.validate()?;
    input.qualification.validate()?;
    if !input.qualification.task_planning {
        bail!("planner qualification does not permit Task planning");
    }
    if input.qualification.model_sha256 != input.model_sha256 {
        bail!("planner qualification does not match the selected model digest");
    }
    if input.qualification.template_sha256 != task_planner_template_sha256() {
        bail!("planner qualification does not match the Task prompt template");
    }
    if input.qualification.protocol_sha256 != task_planner_protocol_sha256() {
        bail!("planner qualification does not match the Task artifact protocol");
    }
    if input.objective.trim().is_empty() {
        bail!("Task planning objective must not be empty");
    }
    if input
        .objective
        .len()
        .saturating_add(input.repository_context.len())
        > MAX_TASK_PLANNING_INPUT_BYTES
    {
        bail!("Task planning input exceeds its bounded context allowance");
    }
    Ok(())
}

fn planner_prompt(input: &TaskPlanningInput<'_>, attempt: usize, feedback: &str) -> String {
    let kind_rule = if input.qualification.automatic_goal_selection {
        "Choose build or goal. Use goal only when the outcome genuinely needs several evidence-driven Builds."
    } else if input.source_intent == TaskSourceIntent::Goal {
        "Return exactly one goal Task because Goal was explicitly selected."
    } else {
        "Every Task must use kind build; automatic Goal selection is not qualified."
    };
    let retry = if feedback.is_empty() {
        String::new()
    } else {
        format!(
            "\nThe previous attempt was rejected. Correct the grounded parts of this feedback, but do not add behavior that is absent from the original request:\n{feedback}\n"
        )
    };
    format!(
        "{TASK_PLANNER_TEMPLATE_VERSION}\nYou are the high-level Task planner. The runtime constrains your response to the exact Task-plan JSON schema; fill every required field and return no prose. Decompose the request into the smallest useful sequence of outcome-shaped Tasks for a smaller implementation model. Return between 1 and {} Tasks and combine closely coupled behavior instead of exceeding this ceiling. These are Tasks, not a Build's implementation plan: describe delivered outcomes and commit boundaries, not exact edits. Every Task must own concrete test work and documentation work or a concrete documentation-impact decision; standalone testing, documentation, validation, integration, or ordering Tasks are forbidden catch-alls. Never include an objective, IDs, references, dependencies, effort, or numeric budgets: pb owns them and makes the array a sequential queue. Array order is the dependency order and commit order: item 1 completes before item 2 can start. A later item cannot make an earlier consumer retroactively compatible, so put foundations and migrations before every service, API, or UI consumer that needs them.\n\n{kind_rule}\nA build Task must omit goal_contract. A goal Task must include it. Select each Task's request_evidence only from the controller choices, verbatim. Assign every choice to at least one behavior-owning Task and never weaken words such as existing, compatible, rollback-safe, before, without, or must in the Task outcome. A Task may not exist only to satisfy a decomposition, testing, documentation, ordering, or smaller-model instruction. Each Task also contains observable outcome acceptance, owned tests, and owned documentation.\n\nBefore responding, silently audit the artifact: cover every controller request-evidence choice, including compatibility, tests, documentation, ordering, rollback, explicit non-goals, and decomposition constraints. When one choice governs several behavior-owning Tasks, assign it to each of them. Check each Task can be independently delivered and committed, and that the complete ordered queue satisfies the request without catch-all work. For every before/after clause, compare the two affected array positions and correct any reversed order.\n\nAttempt: {attempt}\nOriginal request:\n{}\n\nController request evidence choices:\n{}\n\nBounded repository context:\n{}\n{retry}",
        input.policy.aggregate.max_tasks,
        input.objective,
        serde_json::to_string_pretty(&request_evidence_clauses(input.objective))
            .expect("request evidence serializes"),
        input.repository_context
    )
}

fn reviewer_prompt(
    input: &TaskPlanningInput<'_>,
    plan: &ArtifactEnvelope<TaskPlanArtifact>,
) -> Result<String> {
    let request_evidence = request_evidence_clauses(input.objective);
    Ok(format!(
        "{TASK_PLANNER_TEMPLATE_VERSION}\nYou are a fresh Task-plan critic with authority bounded by the original request. The runtime constrains your response to the exact review JSON schema; fill every required field and return no prose. Task IDs are the exact sequential execution order: task-01 completes before task-02 starts. First complete one request_assessment for every controller request-evidence choice exactly once. For each, cite the Tasks that preserve its exact meaning and fail contradictions such as new versus existing, breaking versus compatible, after versus before, or forward-only versus rollback-safe. For every before/after clause, name the earlier and later Task IDs in the detail and fail if the prerequisite has the larger number. Then complete all six audit categories exactly once with concise evidence. request_coverage includes compatibility, tests, documentation, ordering, rollback, explicit non-goals, and decomposition constraints even when they look like meta-instructions. test_documentation_ownership fails when requested tests or documentation are absent from any applicable behavior-owning Task or its acceptance facts. A standalone testing, documentation, validation, integration, ordering, or generic completion Task is a forbidden catch-all when its work belongs in behavior-owning Tasks. Also check each Task's size for a smaller implementation model, dependency and migration/rollback order, observable acceptance, commit boundaries, Build-versus-Goal choice, and qualitative effort. Treat genuinely equivalent wording as coverage, but do not treat a related outcome as equivalent to an exact constraint. Fail only for an omitted or contradicted request clause, an unobservable requested outcome, duplicated ownership, forbidden catch-all, reversed dependency, or a Task too broad to deliver independently. Every challenge must select request_evidence verbatim from the controller evidence list and ask for the minimum correction supported by that exact text. Do not claim the evidence says more than it does. Do not invent HTTP status codes, filenames, algorithms, integrity techniques, test names, documentation cross-links, failure modes, or other behavior absent from the request. Shared components do not imply overlap when Tasks own distinct outcomes. A pass means every request assessment and audit passes and the queue is executable as written and completely delivers the request. verdict pass must contain no blocking challenges; verdict revise must contain a failed request assessment or audit and at least one precise blocking challenge that tells the planner what fact to correct.\n\nOriginal request:\n{}\n\nController request evidence choices:\n{}\n\nTask plan digest: {}\nTask plan:\n{}",
        input.objective,
        serde_json::to_string_pretty(&request_evidence)?,
        plan.sha256,
        serde_json::to_string_pretty(&plan.artifact)?
    ))
}

fn planner_schema(input: &TaskPlanningInput<'_>) -> Value {
    let explicit_goal = input.source_intent == TaskSourceIntent::Goal;
    let goal_allowed = explicit_goal || input.qualification.automatic_goal_selection;
    let task_kinds = if explicit_goal {
        json!(["goal"])
    } else if goal_allowed {
        json!(["build", "goal"])
    } else {
        json!(["build"])
    };
    let mut task_properties = serde_json::Map::from_iter([
        ("title".to_string(), bounded_string(MAX_TITLE_CHARS)),
        (
            "description".to_string(),
            bounded_string(MAX_DESCRIPTION_CHARS),
        ),
        (
            "request_evidence".to_string(),
            json!({
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": request_evidence_clauses(input.objective)
                },
                "minItems": 1,
                "maxItems": MAX_PLANNING_FACTS
            }),
        ),
        (
            "acceptance".to_string(),
            bounded_nonempty_string_array(MAX_PLANNING_FACTS, MAX_DESCRIPTION_CHARS),
        ),
        (
            "tests".to_string(),
            bounded_nonempty_string_array(MAX_PLANNING_FACTS, MAX_DESCRIPTION_CHARS),
        ),
        (
            "documentation".to_string(),
            bounded_nonempty_string_array(MAX_PLANNING_FACTS, MAX_DESCRIPTION_CHARS),
        ),
        (
            "scope_hints".to_string(),
            bounded_string_array(MAX_SCOPE_HINTS, MAX_DESCRIPTION_CHARS),
        ),
        (
            "kind".to_string(),
            json!({"type": "string", "enum": task_kinds}),
        ),
    ]);
    if goal_allowed {
        task_properties.insert("goal_contract".to_string(), goal_contract_schema());
    }
    let mut required = vec![
        "title",
        "description",
        "request_evidence",
        "acceptance",
        "tests",
        "documentation",
        "scope_hints",
        "kind",
    ];
    if explicit_goal {
        required.push("goal_contract");
    }
    let max_tasks = if explicit_goal {
        1
    } else {
        input.policy.aggregate.max_tasks
    };
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": task_properties,
                    "required": required
                },
                "minItems": 1,
                "maxItems": max_tasks
            },
            "risks": object_array(
                json!({"description": bounded_string(MAX_DESCRIPTION_CHARS)}),
                &["description"],
                0,
                MAX_RISKS
            )
        },
        "required": ["tasks", "risks"]
    })
}

fn reviewer_schema(
    input: &TaskPlanningInput<'_>,
    plan: &ArtifactEnvelope<TaskPlanArtifact>,
) -> Value {
    let task_ids = plan
        .artifact
        .tasks
        .iter()
        .map(|task| Value::String(task.id.clone()))
        .collect::<Vec<_>>();
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "task_plan_sha256": {"type": "string", "enum": [plan.sha256.clone()]},
            "verdict": {"type": "string", "enum": ["pass", "revise"]},
            "request_assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "request_evidence": {
                            "type": "string",
                            "enum": request_evidence_clauses(input.objective)
                        },
                        "verdict": {"type": "string", "enum": ["pass", "fail"]},
                        "detail": bounded_string(MAX_DESCRIPTION_CHARS),
                        "task_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": task_ids.clone()},
                            "minItems": 0,
                            "maxItems": plan.artifact.tasks.len()
                        }
                    },
                    "required": ["request_evidence", "verdict", "detail", "task_ids"]
                },
                "minItems": request_evidence_clauses(input.objective).len(),
                "maxItems": request_evidence_clauses(input.objective).len()
            },
            "audits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "request_coverage",
                                "task_boundaries",
                                "dependency_order",
                                "acceptance_observability",
                                "test_documentation_ownership",
                                "effort_goal_authority"
                            ]
                        },
                        "verdict": {"type": "string", "enum": ["pass", "fail"]},
                        "detail": bounded_string(MAX_DESCRIPTION_CHARS),
                        "task_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": task_ids.clone()},
                            "minItems": 0,
                            "maxItems": plan.artifact.tasks.len()
                        }
                    },
                    "required": ["category", "verdict", "detail", "task_ids"]
                },
                "minItems": 6,
                "maxItems": 6
            },
            "challenges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "id": bounded_string(MAX_ID_CHARS),
                        "code": bounded_string(MAX_ID_CHARS),
                        "request_evidence": {
                            "type": "string",
                            "enum": request_evidence_clauses(input.objective)
                        },
                        "description": bounded_string(MAX_DESCRIPTION_CHARS),
                        "severity": {"type": "string", "enum": ["blocking", "advisory"]},
                        "task_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": task_ids},
                            "minItems": 0,
                            "maxItems": plan.artifact.tasks.len()
                        }
                    },
                    "required": ["id", "code", "request_evidence", "description", "severity", "task_ids"]
                },
                "minItems": 0,
                "maxItems": MAX_REVIEW_CHALLENGES
            }
        },
        "required": ["task_plan_sha256", "verdict", "request_assessments", "audits", "challenges"]
    })
}

fn request_evidence_clauses(objective: &str) -> Vec<String> {
    let mut clauses = Vec::new();
    for segment in objective.split(['.', ';', '\n']) {
        let characters = segment.trim().chars().collect::<Vec<_>>();
        for chunk in characters.chunks(MAX_DESCRIPTION_CHARS) {
            let clause = chunk.iter().collect::<String>().trim().to_string();
            if !clause.is_empty() && !clauses.contains(&clause) {
                clauses.push(clause);
                if clauses.len() == MAX_PLANNING_FACTS {
                    return clauses;
                }
            }
        }
    }
    if clauses.is_empty() {
        clauses.push(
            objective
                .trim()
                .chars()
                .take(MAX_DESCRIPTION_CHARS)
                .collect(),
        );
    }
    clauses
}

fn validate_review_request_evidence(
    review: &TaskPlanReviewArtifact,
    objective: &str,
) -> Result<()> {
    let clauses = request_evidence_clauses(objective);
    let allowed = clauses.iter().cloned().collect::<BTreeSet<_>>();
    let mut assessed = BTreeSet::new();
    for assessment in &review.request_assessments {
        if !allowed.contains(&assessment.request_evidence) {
            bail!("Task-plan request assessment does not cite controller request evidence");
        }
        if !assessed.insert(assessment.request_evidence.clone()) {
            bail!("Task-plan request assessment repeats controller request evidence");
        }
    }
    if assessed != allowed {
        bail!("Task-plan review must assess every controller request-evidence clause exactly once");
    }
    let clauses = clauses.into_iter().collect::<BTreeSet<_>>();
    for challenge in &review.challenges {
        if !clauses.contains(&challenge.request_evidence) {
            bail!(
                "Task-plan challenge '{}' does not cite controller request evidence",
                challenge.id
            );
        }
    }
    Ok(())
}

fn goal_contract_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "objective": bounded_string(MAX_DESCRIPTION_CHARS),
            "criteria": object_array(
                json!({
                    "text": bounded_string(MAX_DESCRIPTION_CHARS),
                    "verifier": {
                        "type": "string",
                        "enum": ["workflow_ready", "review_required", "user_confirmation"]
                    }
                }),
                &["text", "verifier"],
                1,
                MAX_GOAL_CRITERIA
            ),
            "continuation": {
                "type": "string",
                "enum": [
                    "review_plan_then_automatic",
                    "manual_milestones",
                    "automatic_within_limits"
                ]
            }
        },
        "required": ["objective", "criteria", "continuation"]
    })
}

fn bounded_string(max_chars: usize) -> Value {
    json!({"type": "string", "minLength": 1, "maxLength": max_chars})
}

fn bounded_string_array(max_items: usize, max_chars: usize) -> Value {
    json!({
        "type": "array",
        "items": bounded_string(max_chars),
        "minItems": 0,
        "maxItems": max_items
    })
}

fn bounded_nonempty_string_array(max_items: usize, max_chars: usize) -> Value {
    let mut schema = bounded_string_array(max_items, max_chars);
    schema["minItems"] = json!(1);
    schema
}

fn object_array(properties: Value, required: &[&str], min_items: usize, max_items: usize) -> Value {
    json!({
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": properties,
            "required": required
        },
        "minItems": min_items,
        "maxItems": max_items
    })
}

fn parse_json<T: DeserializeOwned>(text: &str) -> Result<T> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str(trimmed) {
        return Ok(value);
    }
    let start = trimmed
        .find('{')
        .context("model output has no JSON object")?;
    let end = trimmed
        .rfind('}')
        .context("model output has no complete JSON object")?;
    if start >= end {
        bail!("model output has no complete JSON object");
    }
    serde_json::from_str(&trimmed[start..=end]).context("failed to parse model JSON object")
}

fn record_output(counters: &mut TaskCoordinationCounters, output: &TaskModelOutput) {
    counters.model_invocations = counters.model_invocations.saturating_add(1);
    counters.generated_tokens = counters
        .generated_tokens
        .saturating_add(output.generated_tokens);
    counters.elapsed_ms = counters.elapsed_ms.saturating_add(output.duration_ms);
}

fn rejected(
    outcome: TaskPlanRejectionOutcome,
    attempts: usize,
    failures: Vec<TaskPlanAttemptFailure>,
    counters: TaskCoordinationCounters,
) -> TaskPlanningOutcome {
    TaskPlanningOutcome::Rejected(TaskPlanRejected {
        outcome,
        attempts,
        failures,
        counters,
        recovery_actions: vec![
            TaskPlanRecoveryAction::RetryPlanning,
            TaskPlanRecoveryAction::EditRequest,
            TaskPlanRecoveryAction::RunAsOneBuild,
        ],
    })
}

fn bounded_feedback(value: &str) -> String {
    value.chars().take(MAX_RETRY_FEEDBACK_CHARS).collect()
}

fn elapsed_ms(duration: std::time::Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::task_queue::TaskConfigDocument;

    struct ScriptedModel {
        outputs: VecDeque<Result<TaskModelOutput>>,
        calls: Vec<(TaskPlanningRole, usize)>,
        cancelled: bool,
    }

    impl ScriptedModel {
        fn new(outputs: impl IntoIterator<Item = String>) -> Self {
            Self {
                outputs: outputs
                    .into_iter()
                    .map(|text| {
                        Ok(TaskModelOutput {
                            text,
                            prompt_tokens: 200,
                            generated_tokens: 100,
                            duration_ms: 10,
                        })
                    })
                    .collect(),
                calls: Vec::new(),
                cancelled: false,
            }
        }
    }

    impl TaskPlanningModel for ScriptedModel {
        fn generate(
            &mut self,
            role: TaskPlanningRole,
            _prompt: &str,
            max_tokens: usize,
            _schema: &Value,
        ) -> Result<TaskModelOutput> {
            self.calls.push((role, max_tokens));
            self.outputs
                .pop_front()
                .context("scripted Task-planning output exhausted")?
        }

        fn should_cancel(&self) -> bool {
            self.cancelled
        }
    }

    fn model_digest() -> String {
        "a".repeat(64)
    }

    fn qualification(goal: bool) -> TaskPlannerQualification {
        TaskPlannerQualification::new(
            model_digest(),
            task_planner_template_sha256(),
            task_planner_protocol_sha256(),
            "b".repeat(64),
            true,
            goal,
        )
        .unwrap()
    }

    fn proposal(label: &str) -> serde_json::Value {
        serde_json::json!({
            "tasks": [{
                "title": format!("Fix average {label}"),
                "description": "Correct the divisor and add its regression test",
                "request_evidence": ["Fix average"],
                "acceptance": ["Regression test passes"],
                "tests": ["Run the average regression test"],
                "documentation": ["Record that no user-facing documentation changes are needed"],
                "scope_hints": ["src"],
                "kind": "build"
            }],
            "risks": []
        })
    }

    fn input<'a>(
        policy: &'a CompiledTaskPolicy,
        qualification: &'a TaskPlannerQualification,
    ) -> TaskPlanningInput<'a> {
        TaskPlanningInput {
            objective: "Fix average",
            repository_context: "Rust component with unit tests",
            source_intent: TaskSourceIntent::Build,
            model_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            qualification,
            policy,
        }
    }

    fn review(plan: &TaskPlanArtifact, attempt: usize, verdict: &str) -> String {
        let envelope = ArtifactEnvelope::new(format!("task-plan-{attempt}"), plan.clone()).unwrap();
        let audits = [
            "request_coverage",
            "task_boundaries",
            "dependency_order",
            "acceptance_observability",
            "test_documentation_ownership",
            "effort_goal_authority",
        ]
        .into_iter()
        .map(|category| {
            serde_json::json!({
                "category": category,
                "verdict": if verdict == "revise" && category == "task_boundaries" {
                    "fail"
                } else {
                    "pass"
                },
                "detail": format!("{category} was checked"),
                "task_ids": []
            })
        })
        .collect::<Vec<_>>();
        serde_json::json!({
            "task_plan_sha256": envelope.sha256,
            "verdict": verdict,
            "request_assessments": [{
                "request_evidence": "Fix average",
                "verdict": if verdict == "revise" { "fail" } else { "pass" },
                "detail": "The requested average correction is traced to task-01",
                "task_ids": ["task-01"]
            }],
            "audits": audits,
            "challenges": if verdict == "pass" { serde_json::json!([]) } else { serde_json::json!([{
                "id": "c1",
                "code": "task_too_broad",
                "request_evidence": "Fix average",
                "description": "Split the catch-all Task",
                "severity": "blocking",
                "task_ids": ["task-01"]
            }]) }
        })
        .to_string()
    }

    fn compiled(value: &serde_json::Value, policy: &CompiledTaskPolicy) -> TaskPlanArtifact {
        serde_json::from_value::<TaskModelProposal>(value.clone())
            .unwrap()
            .into_controller_proposal(policy, "Fix average")
            .unwrap()
            .validate_and_compile(
                TaskPlanAuthority {
                    source_intent: TaskSourceIntent::Build,
                    task_planning_qualified: true,
                    automatic_goal_selection_qualified: false,
                },
                policy,
            )
            .unwrap()
    }

    #[test]
    fn accepted_plan_is_reviewed_and_keeps_exact_coordination_usage() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = qualification(false);
        let value = proposal("accepted");
        let plan = compiled(&value, &policy);
        let mut model = ScriptedModel::new([value.to_string(), review(&plan, 1, "pass")]);
        let outcome = plan_tasks(&mut model, input(&policy, &qualification));
        let TaskPlanningOutcome::Accepted(accepted) = outcome else {
            panic!("plan should be accepted");
        };
        assert_eq!(accepted.plan.artifact.tasks.len(), 1);
        assert_eq!(accepted.counters.planning_attempts, 1);
        assert_eq!(accepted.counters.model_invocations, 2);
        assert_eq!(accepted.counters.advisory_calls, 1);
        assert_eq!(accepted.counters.generated_tokens, 200);
    }

    #[test]
    fn deterministic_rejection_gets_one_bounded_revision_attempt() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = qualification(false);
        let invalid = serde_json::json!({
            "tasks": [{"title": "Fix", "description": "Fix", "request_evidence": [], "acceptance": ["Pass"], "tests": ["Run tests"], "documentation": ["Record documentation impact"], "scope_hints": [], "kind": "build"}],
            "risks": []
        });
        let valid = proposal("revised");
        let plan = compiled(&valid, &policy);
        let mut model = ScriptedModel::new([
            invalid.to_string(),
            valid.to_string(),
            review(&plan, 2, "pass"),
        ]);
        let TaskPlanningOutcome::Accepted(accepted) =
            plan_tasks(&mut model, input(&policy, &qualification))
        else {
            panic!("revised plan should be accepted");
        };
        assert_eq!(accepted.counters.planning_attempts, 2);
        assert_eq!(accepted.counters.model_invocations, 3);
    }

    #[test]
    fn three_invalid_attempts_stop_with_explicit_recovery_actions() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = qualification(false);
        let invalid = "{\"objective\":\"bad\",\"tasks\":[],\"risks\":[]}";
        let mut model = ScriptedModel::new([
            invalid.to_string(),
            invalid.to_string(),
            invalid.to_string(),
        ]);
        let TaskPlanningOutcome::Rejected(rejected) =
            plan_tasks(&mut model, input(&policy, &qualification))
        else {
            panic!("invalid plans must be rejected");
        };
        assert_eq!(
            rejected.outcome,
            TaskPlanRejectionOutcome::AttemptsExhausted
        );
        assert_eq!(rejected.attempts, 3);
        assert_eq!(
            rejected.recovery_actions,
            vec![
                TaskPlanRecoveryAction::RetryPlanning,
                TaskPlanRecoveryAction::EditRequest,
                TaskPlanRecoveryAction::RunAsOneBuild
            ]
        );
    }

    #[test]
    fn blocking_review_forces_the_second_planning_attempt() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = qualification(false);
        let first = proposal("first");
        let second = proposal("second");
        let first_plan = compiled(&first, &policy);
        let second_plan = compiled(&second, &policy);
        let mut revise =
            serde_json::from_str::<serde_json::Value>(&review(&first_plan, 1, "revise")).unwrap();
        revise["challenges"][0]["task_ids"] = serde_json::json!(["task-01"]);
        let mut model = ScriptedModel::new([
            first.to_string(),
            revise.to_string(),
            second.to_string(),
            review(&second_plan, 2, "pass"),
        ]);
        let TaskPlanningOutcome::Accepted(accepted) =
            plan_tasks(&mut model, input(&policy, &qualification))
        else {
            panic!("second plan should pass");
        };
        assert_eq!(accepted.plan.artifact.tasks[0].id, "task-01");
        assert!(accepted.plan.artifact.tasks[0].title.ends_with("second"));
        assert_eq!(accepted.counters.model_invocations, 4);
        assert_eq!(accepted.counters.advisory_calls, 2);
    }

    #[test]
    fn qualification_must_match_the_exact_template_and_protocol() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = TaskPlannerQualification::new(
            model_digest(),
            "c".repeat(64),
            task_planner_protocol_sha256(),
            "b".repeat(64),
            true,
            false,
        )
        .unwrap();
        let mut model = ScriptedModel::new(Vec::<String>::new());
        let TaskPlanningOutcome::Rejected(rejected) =
            plan_tasks(&mut model, input(&policy, &qualification))
        else {
            panic!("mismatched qualification must fail");
        };
        assert_eq!(
            rejected.outcome,
            TaskPlanRejectionOutcome::QualificationMismatch
        );
        assert!(model.calls.is_empty());
    }

    #[test]
    fn controller_assigns_fact_ids_and_sequential_dependencies() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let value = serde_json::json!({
            "tasks": [
                {
                    "title": "Storage compatibility",
                    "description": "Deliver the compatibility boundary",
                    "request_evidence": ["Fix average"],
                    "acceptance": ["Storage compatibility tests pass"],
                    "tests": ["Run storage compatibility tests"],
                    "documentation": ["Document the storage compatibility boundary"],
                    "scope_hints": ["storage"],
                    "kind": "build"
                },
                {
                    "title": "API consumer",
                    "description": "Move API reads to the boundary",
                    "request_evidence": ["Fix average"],
                    "acceptance": ["API compatibility tests pass"],
                    "tests": ["Run API compatibility tests"],
                    "documentation": ["Document the API compatibility behavior"],
                    "scope_hints": ["api"],
                    "kind": "build"
                }
            ],
            "risks": []
        });

        let plan = compiled(&value, &policy);
        assert_eq!(plan.tasks[0].id, "task-01");
        assert_eq!(plan.tasks[0].effort, TaskEffort::Small);
        assert!(plan.tasks[0].depends_on.is_empty());
        assert_eq!(plan.tasks[1].id, "task-02");
        assert_eq!(plan.tasks[1].depends_on, ["task-01"]);
        assert_eq!(plan.requirements.len(), 1);
        assert_eq!(plan.acceptance.len(), 6);
        assert_eq!(
            plan.tasks[0].requirement_ids[0],
            plan.tasks[1].requirement_ids[0]
        );
        assert!(
            plan.acceptance
                .iter()
                .any(|fact| fact.description.starts_with("Tests: "))
        );
        assert!(
            plan.acceptance
                .iter()
                .any(|fact| fact.description.starts_with("Documentation: "))
        );
    }

    #[test]
    fn controller_keeps_the_verbatim_objective_and_requires_every_request_clause() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let objective = "Keep the existing API. Update its documentation.";
        let model: TaskModelProposal = serde_json::from_value(serde_json::json!({
            "tasks": [{
                "title": "Keep the API",
                "description": "Preserve the existing API behavior",
                "request_evidence": ["Keep the existing API"],
                "acceptance": ["Existing clients remain compatible"],
                "tests": ["Run API compatibility tests"],
                "documentation": ["Update the API documentation"],
                "scope_hints": ["api"],
                "kind": "build"
            }],
            "risks": []
        }))
        .unwrap();
        let proposal = model.into_controller_proposal(&policy, objective).unwrap();
        assert_eq!(proposal.objective, objective);
        let error = proposal
            .validate_and_compile(
                TaskPlanAuthority {
                    source_intent: TaskSourceIntent::Build,
                    task_planning_qualified: true,
                    automatic_goal_selection_qualified: false,
                },
                &policy,
            )
            .unwrap_err();
        assert_eq!(
            error.code,
            super::super::TaskPlanErrorCode::UncoveredRequirement
        );
    }

    #[test]
    fn controller_rejects_tasks_that_own_only_decomposition_constraints() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let objective = "Add durable storage. Each behavior-owning Task includes tests and documentation. Decompose this into bounded Tasks suitable for a smaller implementation model.";
        let model: TaskModelProposal = serde_json::from_value(serde_json::json!({
            "tasks": [
                {
                    "title": "Add durable storage",
                    "description": "Deliver durable storage",
                    "request_evidence": ["Add durable storage"],
                    "acceptance": ["Storage survives restart"],
                    "tests": ["Run restart tests"],
                    "documentation": ["Document durable storage"],
                    "scope_hints": ["storage"],
                    "kind": "build"
                },
                {
                    "title": "Test and document everything",
                    "description": "Run final tests and write final documentation",
                    "request_evidence": [
                        "Each behavior-owning Task includes tests and documentation",
                        "Decompose this into bounded Tasks suitable for a smaller implementation model"
                    ],
                    "acceptance": ["Everything is validated"],
                    "tests": ["Run all tests"],
                    "documentation": ["Write all documentation"],
                    "scope_hints": [],
                    "kind": "build"
                }
            ],
            "risks": []
        }))
        .unwrap();
        assert!(
            model
                .into_controller_proposal(&policy, objective)
                .unwrap_err()
                .to_string()
                .contains("owns only decomposition constraints")
        );
    }

    #[test]
    fn controller_normalizes_goal_effort_to_the_aggregate_budget() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let tasks = (1..=4)
            .map(|index| {
                serde_json::json!({
                    "title": format!("Goal {index}"),
                    "description": format!("Deliver Goal outcome {index}"),
                    "request_evidence": ["Deliver four evidence-driven outcomes"],
                    "acceptance": [format!("Outcome {index} is verified")],
                    "tests": [format!("Verify outcome {index}")],
                    "documentation": [format!("Document outcome {index}")],
                    "scope_hints": [],
                    "kind": "goal",
                    "goal_contract": {
                        "objective": format!("Deliver Goal outcome {index}"),
                        "criteria": [{
                            "text": format!("Outcome {index} is verified"),
                            "verifier": "workflow_ready"
                        }],
                        "continuation": "review_plan_then_automatic"
                    }
                })
            })
            .collect::<Vec<_>>();
        let model: TaskModelProposal = serde_json::from_value(serde_json::json!({
            "tasks": tasks,
            "risks": []
        }))
        .unwrap();

        let proposal = model
            .into_controller_proposal(&policy, "Deliver four evidence-driven outcomes")
            .unwrap();
        assert_eq!(
            proposal
                .tasks
                .iter()
                .map(|task| task.effort)
                .collect::<Vec<_>>(),
            [
                TaskEffort::Large,
                TaskEffort::Medium,
                TaskEffort::Medium,
                TaskEffort::Medium
            ]
        );
        assert!(model_efforts_fit(&proposal.tasks, &policy));
    }

    #[test]
    fn reviewer_challenges_must_cite_verbatim_request_evidence() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let plan = compiled(&proposal("evidence"), &policy);
        let mut review_artifact: TaskPlanReviewArtifact =
            serde_json::from_str(&review(&plan, 1, "revise")).unwrap();
        validate_review_request_evidence(&review_artifact, "Fix average").unwrap();

        review_artifact.challenges[0].request_evidence = "Invented requirement".to_string();
        assert!(
            validate_review_request_evidence(&review_artifact, "Fix average")
                .unwrap_err()
                .to_string()
                .contains("does not cite controller request evidence")
        );

        let mut missing_assessment: TaskPlanReviewArtifact =
            serde_json::from_str(&review(&plan, 1, "revise")).unwrap();
        missing_assessment.request_assessments.clear();
        assert!(
            validate_review_request_evidence(&missing_assessment, "Fix average")
                .unwrap_err()
                .to_string()
                .contains("assess every controller request-evidence clause")
        );
    }

    #[test]
    fn constrained_output_schemas_compile_for_both_inference_engines() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = qualification(false);
        let build_input = input(&policy, &qualification);
        let build_schema = planner_schema(&build_input);
        assert_eq!(
            build_schema.pointer("/properties/tasks/maxItems"),
            Some(&json!(policy.aggregate.max_tasks))
        );
        assert_eq!(
            build_schema.pointer("/properties/tasks/items/properties/kind/enum"),
            Some(&json!(["build"]))
        );
        assert!(
            build_schema
                .pointer("/properties/tasks/items/properties/goal_contract")
                .is_none()
        );
        assert!(
            build_schema
                .pointer("/properties/tasks/items/properties/id")
                .is_none()
        );
        assert!(
            build_schema
                .pointer("/properties/tasks/items/properties/depends_on")
                .is_none()
        );
        assert!(
            build_schema
                .pointer("/properties/tasks/items/properties/effort")
                .is_none()
        );
        assert_eq!(
            build_schema.pointer("/properties/tasks/items/properties/request_evidence/minItems"),
            Some(&json!(1))
        );
        assert!(build_schema.pointer("/properties/objective").is_none());
        assert_eq!(
            build_schema.pointer("/properties/tasks/items/properties/tests/minItems"),
            Some(&json!(1))
        );
        assert_eq!(
            build_schema.pointer("/properties/tasks/items/properties/documentation/minItems"),
            Some(&json!(1))
        );

        let mut goal_input = input(&policy, &qualification);
        goal_input.source_intent = TaskSourceIntent::Goal;
        let goal_schema = planner_schema(&goal_input);
        assert_eq!(
            goal_schema.pointer("/properties/tasks/maxItems"),
            Some(&json!(1))
        );
        assert!(
            goal_schema
                .pointer("/properties/tasks/items/required")
                .and_then(Value::as_array)
                .is_some_and(|required| required.contains(&json!("goal_contract")))
        );

        let value = proposal("schema");
        let plan = compiled(&value, &policy);
        let envelope = ArtifactEnvelope::new("task-plan-schema-test", plan).unwrap();
        let review_schema = reviewer_schema(&build_input, &envelope);
        assert_eq!(
            review_schema.pointer("/properties/task_plan_sha256/enum/0"),
            Some(&json!(envelope.sha256))
        );
        assert_eq!(
            review_schema.pointer("/properties/request_assessments/minItems"),
            Some(&json!(1))
        );
        assert_eq!(
            review_schema
                .pointer("/properties/challenges/items/properties/request_evidence/enum/0"),
            Some(&json!("Fix average"))
        );

        for schema in [build_schema, goal_schema, review_schema] {
            crate::inference::flashmoe::validate_native_tool_schema(&schema).unwrap();
            let schema = serde_json::to_string(&schema).unwrap();
            let grammar = llama_cpp_2::json_schema_to_grammar(&schema).unwrap();
            assert!(grammar.contains("root"));
            assert!(!grammar.contains("{1,2048}"));
        }
    }

    #[test]
    fn model_authored_numeric_budget_is_rejected_on_every_attempt() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = qualification(false);
        let mut value = proposal("budget");
        value["tasks"][0]["budget"] = serde_json::json!({"generated_tokens": 999999});
        let mut model =
            ScriptedModel::new([value.to_string(), value.to_string(), value.to_string()]);
        let TaskPlanningOutcome::Rejected(rejected) =
            plan_tasks(&mut model, input(&policy, &qualification))
        else {
            panic!("model budgets must fail");
        };
        assert_eq!(rejected.attempts, 3);
        assert!(
            rejected
                .failures
                .iter()
                .all(|failure| failure.reason.contains("unknown field `budget`"))
        );
    }

    #[test]
    fn cancellation_stops_without_invoking_the_model() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = qualification(false);
        let mut model = ScriptedModel::new(Vec::<String>::new());
        model.cancelled = true;
        let TaskPlanningOutcome::Rejected(rejected) =
            plan_tasks(&mut model, input(&policy, &qualification))
        else {
            panic!("cancelled planning must reject");
        };
        assert_eq!(rejected.outcome, TaskPlanRejectionOutcome::Cancelled);
        assert!(model.calls.is_empty());
    }
}
