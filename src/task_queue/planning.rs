use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::workflow::ArtifactEnvelope;

use super::{
    CompiledTaskPolicy, REQUIRED_TASK_PLAN_AUDIT_CATEGORIES, TaskAcceptance,
    TaskCoordinationCounters, TaskEffort, TaskKind, TaskPlanArtifact, TaskPlanAudit,
    TaskPlanAuditCategory, TaskPlanAuditVerdict, TaskPlanAuthority, TaskPlanProposal,
    TaskPlanRequestAssessment, TaskPlanReviewArtifact, TaskPlanReviewVerdict,
    TaskPlannerQualification, TaskProposal, TaskRequirement, TaskSourceIntent,
};

const TASK_PLANNER_TEMPLATE_VERSION: &str = "pb-task-planner-template-v13";
const TASK_PLANNER_PROTOCOL_VERSION: &str = "pb-task-partition-review-v12";
const MAX_TASK_PLANNING_INPUT_BYTES: usize = 64 * 1024;
const MAX_TASK_PLANNER_OUTPUT_TOKENS: usize = 1_024;
const MAX_TASK_REVIEW_OUTPUT_TOKENS: usize = 512;
const MAX_RETRY_FEEDBACK_CHARS: usize = 4_000;
const MAX_PLANNING_FACTS: usize = 32;
const MAX_TITLE_CHARS: usize = 256;
// llama.cpp's grammar parser rejects bounded repetitions above 2,000. Keep the shared schema
// comfortably below that backend limit so the same contract compiles in llama.cpp and FlashMoe.
const MAX_DESCRIPTION_CHARS: usize = 1_024;
const MAX_BUILD_TASK_BEHAVIOR_CLAUSES: usize = 3;
const MAX_TASK_PLANNING_ATTEMPTS: usize = 2;

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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TaskModelTask {
    title: String,
    covers: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TaskModelReviewDecision {
    Accept,
    Revise,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TaskModelReview {
    decision: TaskModelReviewDecision,
    issues: Vec<TaskModelReviewIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TaskModelReviewIssue {
    code: TaskModelReviewIssueCode,
    task_ids: Vec<String>,
    source_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TaskModelReviewIssueCode {
    BadOrder,
    TooBroad,
}

impl TaskModelProposal {
    fn into_controller_proposal(
        self,
        policy: &CompiledTaskPolicy,
        objective: &str,
    ) -> Result<TaskPlanProposal> {
        let request_evidence = request_evidence_clauses(objective);
        let requirements = request_evidence
            .iter()
            .enumerate()
            .map(|(index, description)| TaskRequirement {
                id: format!("req-{:03}", index + 1),
                description: description.clone(),
            })
            .collect::<Vec<_>>();
        let requirement_ids_by_description = requirements
            .iter()
            .map(|requirement| (requirement.description.clone(), requirement.id.clone()))
            .collect::<BTreeMap<_, _>>();
        let behavior_evidence = behavior_evidence_clauses(objective);
        let requirement_ids = behavior_evidence
            .iter()
            .enumerate()
            .map(|(index, description)| {
                (
                    source_id(index),
                    requirement_ids_by_description
                        .get(description)
                        .expect("behavior evidence is a request clause")
                        .clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let behavior_set = behavior_evidence.into_iter().collect::<BTreeSet<_>>();
        let constraint_requirement_ids = requirements
            .iter()
            .filter(|requirement| !behavior_set.contains(&requirement.description))
            .map(|requirement| requirement.id.clone())
            .collect::<Vec<_>>();
        let mut covered = BTreeSet::new();
        let mut source_task_positions = BTreeMap::new();
        let mut acceptance = Vec::new();
        let mut tasks = Vec::with_capacity(self.tasks.len());

        for (task_index, task) in self.tasks.into_iter().enumerate() {
            let id = format!("task-{:02}", task_index + 1);
            if task.covers.is_empty() {
                bail!(
                    "Task '{}' must cover at least one source clause",
                    task.title
                );
            }
            if task.covers.len() > MAX_BUILD_TASK_BEHAVIOR_CLAUSES {
                bail!(
                    "Task '{}' covers {} source clauses; maximum is {MAX_BUILD_TASK_BEHAVIOR_CLAUSES}",
                    task.title,
                    task.covers.len()
                );
            }
            let mut task_requirement_ids = constraint_requirement_ids.clone();
            let mut descriptions = Vec::with_capacity(task.covers.len());
            for source_id in task.covers {
                if !covered.insert(source_id.clone()) {
                    bail!("source clause '{source_id}' is assigned more than once");
                }
                source_task_positions.insert(source_id.clone(), task_index);
                let requirement_id = requirement_ids.get(&source_id).with_context(|| {
                    format!(
                        "Task '{}' cites unknown source clause '{source_id}'",
                        task.title
                    )
                })?;
                task_requirement_ids.push(requirement_id.clone());
                let requirement = requirements
                    .iter()
                    .find(|candidate| candidate.id == *requirement_id)
                    .expect("controller requirement exists");
                descriptions.push(requirement.description.clone());
            }
            let acceptance_id = format!("accept-{:03}", task_index + 1);
            let assigned_descriptions = task_requirement_ids
                .iter()
                .filter_map(|id| {
                    requirements
                        .iter()
                        .find(|requirement| requirement.id == *id)
                        .map(|requirement| requirement.description.as_str())
                })
                .collect::<Vec<_>>();
            acceptance.push(TaskAcceptance {
                id: acceptance_id.clone(),
                description: format!(
                    "The normal Build workflow verifies and commits these assigned source requirements: {}",
                    assigned_descriptions.join("; ")
                ),
            });

            tasks.push(TaskProposal {
                id,
                title: task.title,
                description: descriptions.join(". "),
                requirement_ids: task_requirement_ids,
                depends_on: task_index
                    .checked_sub(1)
                    .map(|previous| vec![format!("task-{:02}", previous + 1)])
                    .unwrap_or_default(),
                acceptance_ids: vec![acceptance_id],
                scope_hints: Vec::new(),
                effort: TaskEffort::Small,
                kind: TaskKind::Build,
                goal_contract: None,
            });
        }

        let expected = requirement_ids.keys().cloned().collect::<BTreeSet<_>>();
        if covered != expected {
            let missing = expected.difference(&covered).cloned().collect::<Vec<_>>();
            bail!(
                "Task partition does not cover source clauses: {}",
                missing.join(", ")
            );
        }
        for (earlier, later) in explicit_source_order_pairs(objective) {
            let earlier_position = source_task_positions
                .get(&earlier)
                .context("explicitly ordered source clause has no Task")?;
            let later_position = source_task_positions
                .get(&later)
                .context("explicitly ordered source clause has no Task")?;
            if earlier_position > later_position {
                bail!(
                    "explicit source order requires '{earlier}' before '{later}' in the Task queue"
                );
            }
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
            risks: Vec::new(),
        })
    }
}

fn source_id(index: usize) -> String {
    format!("source-{:03}", index + 1)
}

fn is_decomposition_constraint(evidence: &str) -> bool {
    let evidence = evidence.to_ascii_lowercase();
    evidence.contains("decompose this")
        || evidence.contains("behavior-owning task")
        || evidence.contains("behavior owning task")
        || evidence.contains("each task")
        || evidence.contains("every task")
        || evidence.contains("tasks suitable for")
        || evidence.contains("dependencies must")
        || ((evidence.contains("test") || evidence.contains("documentation"))
            && (evidence.contains("each behavior")
                || evidence.contains("every behavior")
                || evidence.contains("each task")
                || evidence.contains("every task")
                || evidence.contains("owned by")))
}

fn behavior_evidence_clauses(objective: &str) -> Vec<String> {
    let clauses = request_evidence_clauses(objective);
    let behavior = clauses
        .iter()
        .filter(|evidence| !is_decomposition_constraint(evidence))
        .cloned()
        .collect::<Vec<_>>();
    if behavior.is_empty() {
        clauses
    } else {
        behavior
    }
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskPlanningDecision {
    MultiTask,
    OneBuildSingleTask,
    OneBuildPlannerFallback,
    OneBuildBudgetFallback,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlanningTranscriptEntry {
    pub attempt: usize,
    pub stage: TaskPlanningRole,
    pub prompt: String,
    pub schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized_output: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlanningTranscript {
    pub decision: TaskPlanningDecision,
    pub summary: String,
    pub attempts: Vec<TaskPlanningTranscriptEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OneBuildTaskPlan {
    pub reason: String,
    pub counters: TaskCoordinationCounters,
    pub transcript: TaskPlanningTranscript,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlanRejected {
    pub outcome: TaskPlanRejectionOutcome,
    pub attempts: usize,
    pub failures: Vec<TaskPlanAttemptFailure>,
    pub counters: TaskCoordinationCounters,
    pub recovery_actions: Vec<TaskPlanRecoveryAction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<TaskPlanningTranscript>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AcceptedTaskPlan {
    pub plan: ArtifactEnvelope<TaskPlanArtifact>,
    pub review: ArtifactEnvelope<TaskPlanReviewArtifact>,
    pub counters: TaskCoordinationCounters,
    pub transcript: TaskPlanningTranscript,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum TaskPlanningOutcome {
    Accepted(AcceptedTaskPlan),
    OneBuild(OneBuildTaskPlan),
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
        Err(error) if input.source_intent == TaskSourceIntent::Build => one_build(
            TaskPlanningDecision::OneBuildPlannerFallback,
            format!("Task planning was unavailable: {error:#}"),
            TaskCoordinationCounters::default(),
            Vec::new(),
        ),
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
    let mut transcript = Vec::new();
    let max_attempts = budget.planning_attempts.min(MAX_TASK_PLANNING_ATTEMPTS);

    for attempt in 1..=max_attempts {
        if model.should_cancel() {
            return Ok(rejected_with_transcript(
                TaskPlanRejectionOutcome::Cancelled,
                counters.planning_attempts,
                failures,
                counters,
                TaskPlanningTranscript {
                    decision: TaskPlanningDecision::Cancelled,
                    summary: "Task planning was cancelled".to_string(),
                    attempts: transcript,
                },
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
            return Ok(build_fallback_or_rejected(
                input,
                TaskPlanRejectionOutcome::BudgetExhausted,
                TaskPlanningDecision::OneBuildBudgetFallback,
                "Task planning had no remaining coordination budget",
                failures,
                counters,
                transcript,
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
                transcript.push(failed_transcript_entry(
                    attempt,
                    TaskPlanningRole::Planner,
                    prompt,
                    schema,
                    reason.clone(),
                    elapsed_ms(started.elapsed()),
                ));
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
            transcript.push(output_transcript_entry(
                attempt,
                TaskPlanningRole::Planner,
                prompt,
                schema,
                &output,
                None,
                Some("planner output exhausted the Task coordination budget".to_string()),
            ));
            return Ok(build_fallback_or_rejected(
                input,
                TaskPlanRejectionOutcome::BudgetExhausted,
                TaskPlanningDecision::OneBuildBudgetFallback,
                "Task planning exhausted its coordination budget",
                failures,
                counters,
                transcript,
            ));
        }
        let model_proposal = match parse_json::<TaskModelProposal>(&output.text) {
            Ok(proposal) => proposal,
            Err(error) => {
                let reason = format!("Task partition rejected: {error:#}");
                failures.push(TaskPlanAttemptFailure {
                    attempt,
                    stage: TaskPlanningRole::Planner,
                    reason: reason.clone(),
                });
                transcript.push(output_transcript_entry(
                    attempt,
                    TaskPlanningRole::Planner,
                    prompt,
                    schema,
                    &output,
                    None,
                    Some(reason.clone()),
                ));
                feedback = bounded_feedback(&reason);
                continue;
            }
        };
        let normalized_proposal = serde_json::to_value(&model_proposal)?;
        transcript.push(output_transcript_entry(
            attempt,
            TaskPlanningRole::Planner,
            prompt,
            schema,
            &output,
            Some(normalized_proposal),
            None,
        ));
        if model_proposal.tasks.len() == 1 && input.source_intent == TaskSourceIntent::Build {
            return Ok(one_build(
                TaskPlanningDecision::OneBuildSingleTask,
                "The partition contained one Task, so pb kept the original Build request unchanged",
                counters,
                transcript,
            ));
        }
        let proposal = match model_proposal
            .into_controller_proposal(input.policy, input.objective)
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
                if let Some(entry) = transcript.last_mut() {
                    entry.failure = Some(reason.clone());
                }
                feedback = bounded_feedback(&reason);
                continue;
            }
        };
        let plan = ArtifactEnvelope::new(format!("task-plan-{attempt}"), proposal)?;

        if model.should_cancel() {
            return Ok(rejected_with_transcript(
                TaskPlanRejectionOutcome::Cancelled,
                counters.planning_attempts,
                failures,
                counters,
                TaskPlanningTranscript {
                    decision: TaskPlanningDecision::Cancelled,
                    summary: "Task planning was cancelled".to_string(),
                    attempts: transcript,
                },
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
            return accepted_task_plan(
                input,
                plan,
                counters,
                transcript,
                "The deterministic partition was accepted without optional criticism",
            );
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
                transcript.push(failed_transcript_entry(
                    attempt,
                    TaskPlanningRole::Reviewer,
                    review_prompt,
                    review_schema,
                    reason.clone(),
                    elapsed_ms(started.elapsed()),
                ));
                return accepted_task_plan(
                    input,
                    plan,
                    counters,
                    transcript,
                    "The deterministic partition was accepted; optional criticism was unavailable",
                );
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
            transcript.push(output_transcript_entry(
                attempt,
                TaskPlanningRole::Reviewer,
                review_prompt,
                review_schema,
                &output,
                None,
                Some("Task-plan review exhausted the coordination budget".to_string()),
            ));
            return Ok(build_fallback_or_rejected(
                input,
                TaskPlanRejectionOutcome::BudgetExhausted,
                TaskPlanningDecision::OneBuildBudgetFallback,
                "Task criticism exhausted its coordination budget",
                failures,
                counters,
                transcript,
            ));
        }
        let model_review = match parse_json::<TaskModelReview>(&output.text)
            .and_then(|review| validate_model_review(review, input, &plan))
        {
            Ok(review) => review,
            Err(error) => {
                let reason = format!("Task criticism rejected: {error:#}");
                failures.push(TaskPlanAttemptFailure {
                    attempt,
                    stage: TaskPlanningRole::Reviewer,
                    reason: reason.clone(),
                });
                transcript.push(output_transcript_entry(
                    attempt,
                    TaskPlanningRole::Reviewer,
                    review_prompt,
                    review_schema,
                    &output,
                    None,
                    Some(reason.clone()),
                ));
                return accepted_task_plan(
                    input,
                    plan,
                    counters,
                    transcript,
                    "The deterministic partition was accepted; optional criticism was invalid",
                );
            }
        };
        transcript.push(output_transcript_entry(
            attempt,
            TaskPlanningRole::Reviewer,
            review_prompt,
            review_schema,
            &output,
            Some(serde_json::to_value(&model_review)?),
            None,
        ));
        let summary = if model_review.decision == TaskModelReviewDecision::Accept {
            "The deterministic partition was accepted and optional criticism found no issue"
        } else {
            "The deterministic partition was accepted; model criticism is preserved as advisory evidence"
        };
        return accepted_task_plan(input, plan, counters, transcript, summary);
    }

    Ok(build_fallback_or_rejected(
        input,
        TaskPlanRejectionOutcome::AttemptsExhausted,
        TaskPlanningDecision::OneBuildPlannerFallback,
        "Task planning did not produce an accepted multi-Task partition",
        failures,
        counters,
        transcript,
    ))
}

fn accepted_task_plan(
    input: &TaskPlanningInput<'_>,
    plan: ArtifactEnvelope<TaskPlanArtifact>,
    counters: TaskCoordinationCounters,
    attempts: Vec<TaskPlanningTranscriptEntry>,
    summary: impl Into<String>,
) -> Result<TaskPlanningOutcome> {
    let review = controller_review(input.objective, &plan);
    let review_envelope = ArtifactEnvelope::new("task-plan-controller-review", review)?;
    Ok(TaskPlanningOutcome::Accepted(AcceptedTaskPlan {
        plan,
        review: review_envelope,
        counters,
        transcript: TaskPlanningTranscript {
            decision: TaskPlanningDecision::MultiTask,
            summary: summary.into(),
            attempts,
        },
    }))
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
    let retry = if feedback.is_empty() {
        String::new()
    } else {
        format!("\nPrevious partition feedback:\n{feedback}\n")
    };
    let sources = source_clauses(input.objective);
    format!(
        "{TASK_PLANNER_TEMPLATE_VERSION}\nReturn only JSON matching the schema. Partition the Build request into an ordered queue of independently deliverable, independently committed Tasks for a smaller implementation model. A Task is an outcome boundary, not a list of files or edits; its normal Build workflow will plan exact changes after the previous Task commits. Use every source ID exactly once. Put foundations and migrations before consumers. Do not create separate test, documentation, review, integration, or cleanup Tasks. If the work is tightly coupled or one Task is sufficient, return exactly one Task; pb will then run the original Build unchanged. For a multi-Task partition, each Task may cover at most {MAX_BUILD_TASK_BEHAVIOR_CLAUSES} source clauses. Return no fields except tasks, title, and covers.\n\nAttempt: {attempt}\nOriginal Build request:\n{}\n\nController source clauses:\n{}\n\nBounded repository context:\n{}\n{retry}",
        input.objective,
        serde_json::to_string_pretty(&sources).expect("source clauses serialize"),
        input.repository_context
    )
}

fn reviewer_prompt(
    input: &TaskPlanningInput<'_>,
    plan: &ArtifactEnvelope<TaskPlanArtifact>,
) -> Result<String> {
    let sources = source_clauses(input.objective);
    let tasks = plan
        .artifact
        .tasks
        .iter()
        .map(|task| {
            json!({
                "id": task.id,
                "title": task.title,
                "covers": task.requirement_ids.iter().filter_map(|requirement_id| {
                    plan.artifact.requirements.iter().position(|requirement| {
                        requirement.id == *requirement_id
                    }).map(source_id)
                }).collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    Ok(format!(
        "{TASK_PLANNER_TEMPLATE_VERSION}\nReturn only JSON matching the schema. Review this ordered Task partition, not the implementation details. Rust has already proven exact source coverage, disjoint ownership, sequential dependencies, budgets, and Build-only authority; do not re-audit or challenge those facts. Accept unless one of two semantic defects is present: bad_order (a prerequisite follows its consumer) or too_broad (a Task is not independently deliverable by a smaller model). A necessary foundation or migration is not an unnecessary Task. Use only the supplied Task IDs and source IDs. Do not invent missing requirements, files, tests, or implementation advice. decision=accept requires issues=[]; decision=revise requires at least one issue.\n\nOriginal Build request:\n{}\n\nController source clauses:\n{}\n\nOrdered partition:\n{}",
        input.objective,
        serde_json::to_string_pretty(&sources)?,
        serde_json::to_string_pretty(&tasks)?
    ))
}

fn planner_schema(input: &TaskPlanningInput<'_>) -> Value {
    let source_ids = source_clauses(input.objective)
        .into_iter()
        .map(|source| source["id"].clone())
        .collect::<Vec<_>>();
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "title": bounded_string(MAX_TITLE_CHARS),
                        "covers": {
                            "type": "array",
                            "items": {"type": "string", "enum": source_ids},
                            "minItems": 1,
                            "maxItems": MAX_PLANNING_FACTS
                        }
                    },
                    "required": ["title", "covers"]
                },
                "minItems": 1,
                "maxItems": input.policy.aggregate.max_tasks
            }
        },
        "required": ["tasks"]
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
    let source_ids = source_clauses(input.objective)
        .into_iter()
        .map(|source| source["id"].clone())
        .collect::<Vec<_>>();
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "decision": {"type": "string", "enum": ["accept", "revise"]},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "code": {"type": "string", "enum": ["bad_order", "too_broad"]},
                        "task_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": task_ids.clone()},
                            "minItems": 1,
                            "maxItems": plan.artifact.tasks.len()
                        },
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": source_ids},
                            "minItems": 0,
                            "maxItems": MAX_PLANNING_FACTS
                        }
                    },
                    "required": ["code", "task_ids", "source_ids"]
                },
                "minItems": 0,
                "maxItems": 8
            }
        },
        "required": ["decision", "issues"]
    })
}

fn source_clauses(objective: &str) -> Vec<Value> {
    behavior_evidence_clauses(objective)
        .into_iter()
        .enumerate()
        .map(|(index, text)| json!({"id": source_id(index), "text": text}))
        .collect()
}

fn request_evidence_clauses(objective: &str) -> Vec<String> {
    let mut clauses = Vec::new();
    for segment in objective.split(['.', ';', ',', '\n']) {
        for ordered in ordered_request_subclauses(segment.trim()) {
            let characters = ordered.chars().collect::<Vec<_>>();
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

fn ordered_request_subclauses(segment: &str) -> Vec<String> {
    for (separator, reverse) in [(" before ", false), (" after ", true), (" then ", false)] {
        let lower = segment.to_ascii_lowercase();
        if let Some(index) = lower.find(separator) {
            let left = segment[..index].trim();
            let right = segment[index + separator.len()..].trim();
            if !left.is_empty() && !right.is_empty() {
                let mut left = ordered_request_subclauses(left);
                let mut right = ordered_request_subclauses(right);
                return if reverse {
                    right.append(&mut left);
                    right
                } else {
                    left.append(&mut right);
                    left
                };
            }
        }
    }
    vec![segment.to_string()]
}

fn explicit_source_order_pairs(objective: &str) -> Vec<(String, String)> {
    let behavior = behavior_evidence_clauses(objective);
    let source_by_text = behavior
        .iter()
        .enumerate()
        .map(|(index, text)| (text.as_str(), source_id(index)))
        .collect::<BTreeMap<_, _>>();
    let mut pairs = Vec::new();
    for segment in objective.split(['.', ';', ',', '\n']) {
        let lower = segment.to_ascii_lowercase();
        for (separator, reverse) in [(" before ", false), (" after ", true), (" then ", false)] {
            let Some(index) = lower.find(separator) else {
                continue;
            };
            let left = ordered_request_subclauses(segment[..index].trim());
            let right = ordered_request_subclauses(segment[index + separator.len()..].trim());
            let (earlier, later) = if reverse {
                (&right, &left)
            } else {
                (&left, &right)
            };
            for earlier in earlier {
                let Some(earlier_id) = source_by_text.get(earlier.as_str()) else {
                    continue;
                };
                for later in later {
                    if let Some(later_id) = source_by_text.get(later.as_str()) {
                        pairs.push((earlier_id.clone(), later_id.clone()));
                    }
                }
            }
            break;
        }
    }
    pairs
}

fn validate_model_review(
    review: TaskModelReview,
    input: &TaskPlanningInput<'_>,
    plan: &ArtifactEnvelope<TaskPlanArtifact>,
) -> Result<TaskModelReview> {
    match review.decision {
        TaskModelReviewDecision::Accept if !review.issues.is_empty() => {
            bail!("an accepted Task partition cannot contain issues")
        }
        TaskModelReviewDecision::Revise if review.issues.is_empty() => {
            bail!("a Task partition revision must identify an issue")
        }
        _ => {}
    }
    let allowed_tasks = plan
        .artifact
        .tasks
        .iter()
        .map(|task| task.id.as_str())
        .collect::<BTreeSet<_>>();
    let allowed_sources = behavior_evidence_clauses(input.objective)
        .iter()
        .enumerate()
        .map(|(index, _)| source_id(index))
        .collect::<BTreeSet<_>>();
    for issue in &review.issues {
        if issue.task_ids.is_empty()
            || issue
                .task_ids
                .iter()
                .any(|task_id| !allowed_tasks.contains(task_id.as_str()))
        {
            bail!("Task criticism references an unknown or empty Task selection");
        }
        if issue
            .source_ids
            .iter()
            .any(|source| !allowed_sources.contains(source))
        {
            bail!("Task criticism references an unknown source clause");
        }
    }
    Ok(review)
}

fn controller_review(
    objective: &str,
    plan: &ArtifactEnvelope<TaskPlanArtifact>,
) -> TaskPlanReviewArtifact {
    let request_assessments = request_evidence_clauses(objective)
        .into_iter()
        .enumerate()
        .map(|(index, request_evidence)| {
            let requirement_id = format!("req-{:03}", index + 1);
            let task_ids = plan
                .artifact
                .tasks
                .iter()
                .filter(|task| task.requirement_ids.contains(&requirement_id))
                .map(|task| task.id.clone())
                .collect();
            TaskPlanRequestAssessment {
                detail: if is_decomposition_constraint(&request_evidence) {
                    "pb attached this decomposition-wide source constraint to every Task"
                        .to_string()
                } else {
                    "pb verified exact, single ownership of this source clause".to_string()
                },
                request_evidence,
                verdict: TaskPlanAuditVerdict::Pass,
                task_ids,
            }
        })
        .collect();
    let audits = REQUIRED_TASK_PLAN_AUDIT_CATEGORIES
        .into_iter()
        .map(|category| TaskPlanAudit {
            category,
            verdict: TaskPlanAuditVerdict::Pass,
            detail: match category {
                TaskPlanAuditCategory::RequestCoverage => {
                    "pb verified every controller source clause is assigned exactly once"
                }
                TaskPlanAuditCategory::TaskBoundaries => {
                    "the partition passed controller size bounds; model criticism is diagnostic"
                }
                TaskPlanAuditCategory::DependencyOrder => {
                    "pb created the sequential dependency and commit order from the partition"
                }
                TaskPlanAuditCategory::AcceptanceObservability => {
                    "each Task retains the normal Build planning, checks, review, and commit gates"
                }
                TaskPlanAuditCategory::TestDocumentationOwnership => {
                    "tests and documentation stay within each behavior-owning Build workflow"
                }
                TaskPlanAuditCategory::EffortGoalAuthority => {
                    "pb assigned bounded Build budgets without promoting Goal authority"
                }
            }
            .to_string(),
            task_ids: Vec::new(),
        })
        .collect();
    TaskPlanReviewArtifact {
        task_plan_sha256: plan.sha256.clone(),
        verdict: TaskPlanReviewVerdict::Pass,
        request_assessments,
        audits,
        challenges: Vec::new(),
    }
}

fn bounded_string(max_chars: usize) -> Value {
    json!({"type": "string", "minLength": 1, "maxLength": max_chars})
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

fn failed_transcript_entry(
    attempt: usize,
    stage: TaskPlanningRole,
    prompt: String,
    schema: Value,
    failure: String,
    duration_ms: u64,
) -> TaskPlanningTranscriptEntry {
    TaskPlanningTranscriptEntry {
        attempt,
        stage,
        prompt,
        schema,
        raw_output: None,
        normalized_output: None,
        failure: Some(failure),
        prompt_tokens: 0,
        generated_tokens: 0,
        duration_ms,
    }
}

fn output_transcript_entry(
    attempt: usize,
    stage: TaskPlanningRole,
    prompt: String,
    schema: Value,
    output: &TaskModelOutput,
    normalized_output: Option<Value>,
    failure: Option<String>,
) -> TaskPlanningTranscriptEntry {
    TaskPlanningTranscriptEntry {
        attempt,
        stage,
        prompt,
        schema,
        raw_output: Some(output.text.clone()),
        normalized_output,
        failure,
        prompt_tokens: output.prompt_tokens,
        generated_tokens: output.generated_tokens,
        duration_ms: output.duration_ms,
    }
}

fn one_build(
    decision: TaskPlanningDecision,
    reason: impl Into<String>,
    counters: TaskCoordinationCounters,
    attempts: Vec<TaskPlanningTranscriptEntry>,
) -> TaskPlanningOutcome {
    let reason = reason.into();
    TaskPlanningOutcome::OneBuild(OneBuildTaskPlan {
        reason: reason.clone(),
        counters,
        transcript: TaskPlanningTranscript {
            decision,
            summary: reason,
            attempts,
        },
    })
}

fn build_fallback_or_rejected(
    input: &TaskPlanningInput<'_>,
    outcome: TaskPlanRejectionOutcome,
    decision: TaskPlanningDecision,
    reason: impl Into<String>,
    failures: Vec<TaskPlanAttemptFailure>,
    counters: TaskCoordinationCounters,
    attempts: Vec<TaskPlanningTranscriptEntry>,
) -> TaskPlanningOutcome {
    let reason = reason.into();
    if input.source_intent == TaskSourceIntent::Build {
        return one_build(decision, reason, counters, attempts);
    }
    rejected_with_transcript(
        outcome,
        counters.planning_attempts,
        failures,
        counters,
        TaskPlanningTranscript {
            decision: TaskPlanningDecision::Rejected,
            summary: reason,
            attempts,
        },
    )
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
        transcript: None,
    })
}

fn rejected_with_transcript(
    outcome: TaskPlanRejectionOutcome,
    attempts: usize,
    failures: Vec<TaskPlanAttemptFailure>,
    counters: TaskCoordinationCounters,
    transcript: TaskPlanningTranscript,
) -> TaskPlanningOutcome {
    let TaskPlanningOutcome::Rejected(mut rejected) =
        rejected(outcome, attempts, failures, counters)
    else {
        unreachable!("rejected always returns a rejected outcome")
    };
    rejected.transcript = Some(transcript);
    TaskPlanningOutcome::Rejected(rejected)
}

fn bounded_feedback(value: &str) -> String {
    value.chars().take(MAX_RETRY_FEEDBACK_CHARS).collect()
}

fn elapsed_ms(duration: std::time::Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod compact_tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::task_queue::TaskConfigDocument;

    struct ScriptedModel {
        outputs: VecDeque<Result<TaskModelOutput>>,
        calls: Vec<TaskPlanningRole>,
        cancelled: bool,
    }

    impl ScriptedModel {
        fn new(outputs: impl IntoIterator<Item = &'static str>) -> Self {
            Self {
                outputs: outputs
                    .into_iter()
                    .map(|text| {
                        Ok(TaskModelOutput {
                            text: text.to_string(),
                            prompt_tokens: 40,
                            generated_tokens: 20,
                            duration_ms: 5,
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
            _max_tokens: usize,
            _schema: &Value,
        ) -> Result<TaskModelOutput> {
            self.calls.push(role);
            self.outputs
                .pop_front()
                .context("scripted Task planning output exhausted")?
        }

        fn should_cancel(&self) -> bool {
            self.cancelled
        }
    }

    fn qualification(source_intent: TaskSourceIntent) -> TaskPlannerQualification {
        TaskPlannerQualification::new(
            "a".repeat(64),
            task_planner_template_sha256(),
            task_planner_protocol_sha256(),
            "b".repeat(64),
            true,
            source_intent == TaskSourceIntent::Goal,
        )
        .unwrap()
    }

    fn run(
        model: &mut ScriptedModel,
        objective: &str,
        source_intent: TaskSourceIntent,
    ) -> TaskPlanningOutcome {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = qualification(source_intent);
        plan_tasks(
            model,
            TaskPlanningInput {
                objective,
                repository_context: "{}",
                source_intent,
                model_sha256: &"a".repeat(64),
                qualification: &qualification,
                policy: &policy,
            },
        )
    }

    #[test]
    fn one_task_keeps_the_original_build_without_criticism() {
        let mut model =
            ScriptedModel::new([r#"{"tasks":[{"title":"One Build","covers":["source-001"]}]}"#]);
        let TaskPlanningOutcome::OneBuild(bypass) =
            run(&mut model, "Fix average", TaskSourceIntent::Build)
        else {
            panic!("one Task must preserve the original Build");
        };
        assert_eq!(
            bypass.transcript.decision,
            TaskPlanningDecision::OneBuildSingleTask
        );
        assert_eq!(model.calls, vec![TaskPlanningRole::Planner]);
        assert_eq!(bypass.transcript.attempts.len(), 1);
    }

    #[test]
    fn compact_multi_task_partition_compiles_controller_owned_artifacts() {
        let mut model = ScriptedModel::new([
            r#"{"tasks":[{"title":"Add storage","covers":["source-001"]},{"title":"Expose health","covers":["source-002"]}]}"#,
            r#"{"decision":"accept","issues":[]}"#,
        ]);
        let TaskPlanningOutcome::Accepted(accepted) = run(
            &mut model,
            "Add durable storage. Expose health",
            TaskSourceIntent::Build,
        ) else {
            panic!("valid compact partition must pass");
        };
        assert_eq!(accepted.plan.artifact.tasks.len(), 2);
        assert_eq!(accepted.plan.artifact.tasks[0].kind, TaskKind::Build);
        assert_eq!(accepted.plan.artifact.tasks[1].depends_on, vec!["task-01"]);
        assert_eq!(accepted.plan.artifact.acceptance.len(), 2);
        assert_eq!(
            accepted.review.artifact.verdict,
            TaskPlanReviewVerdict::Pass
        );
        assert_eq!(accepted.transcript.attempts.len(), 2);
        accepted.review.artifact.validate(&accepted.plan).unwrap();
    }

    #[test]
    fn duplicate_or_missing_source_ownership_gets_one_revision() {
        let mut model = ScriptedModel::new([
            r#"{"tasks":[{"title":"First","covers":["source-001"]},{"title":"Duplicate","covers":["source-001"]}]}"#,
            r#"{"tasks":[{"title":"First","covers":["source-001"]},{"title":"Second","covers":["source-002"]}]}"#,
            r#"{"decision":"accept","issues":[]}"#,
        ]);
        let TaskPlanningOutcome::Accepted(accepted) = run(
            &mut model,
            "Add storage. Expose health",
            TaskSourceIntent::Build,
        ) else {
            panic!("corrected partition must pass");
        };
        assert_eq!(accepted.counters.planning_attempts, 2);
        assert_eq!(accepted.transcript.attempts.len(), 3);
        assert!(accepted.transcript.attempts[0].failure.is_some());
    }

    #[test]
    fn controller_attaches_decomposition_wide_constraints_to_each_task() {
        let mut model = ScriptedModel::new([
            r#"{"tasks":[{"title":"Storage","covers":["source-001"]},{"title":"Health","covers":["source-002"]}]}"#,
            r#"{"decision":"accept","issues":[]}"#,
        ]);
        let TaskPlanningOutcome::Accepted(accepted) = run(
            &mut model,
            "Add storage. Expose health. Each Task includes its applicable tests and documentation",
            TaskSourceIntent::Build,
        ) else {
            panic!("valid partition must pass");
        };
        let constraint = accepted
            .plan
            .artifact
            .requirements
            .iter()
            .find(|requirement| requirement.description.starts_with("Each Task"))
            .unwrap();
        assert!(
            accepted
                .plan
                .artifact
                .tasks
                .iter()
                .all(|task| task.requirement_ids.contains(&constraint.id))
        );
    }

    #[test]
    fn controller_atomizes_ordering_and_keeps_test_documentation_ownership_global() {
        let objective = "Add durable import lifecycle storage and a rollback-safe migration before exposing compatible status and cancel APIs, with tests and documentation owned by each behavior change.";
        assert_eq!(
            request_evidence_clauses(objective),
            vec![
                "Add durable import lifecycle storage and a rollback-safe migration",
                "exposing compatible status and cancel APIs",
                "with tests and documentation owned by each behavior change"
            ]
        );
        assert_eq!(behavior_evidence_clauses(objective).len(), 2);
        assert_eq!(
            ordered_request_subclauses("Expose health after adding storage"),
            vec!["adding storage", "Expose health"]
        );

        let mut model = ScriptedModel::new([
            r#"{"tasks":[{"title":"Storage and migration","covers":["source-001"]},{"title":"Compatible APIs","covers":["source-002"]}]}"#,
            r#"{"decision":"accept","issues":[]}"#,
        ]);
        let TaskPlanningOutcome::Accepted(accepted) =
            run(&mut model, objective, TaskSourceIntent::Build)
        else {
            panic!("ordered partition must pass");
        };
        let constraint_id = accepted.plan.artifact.requirements[2].id.clone();
        assert_eq!(accepted.plan.artifact.tasks[1].depends_on, vec!["task-01"]);
        assert!(
            accepted
                .plan
                .artifact
                .tasks
                .iter()
                .all(|task| task.requirement_ids.contains(&constraint_id))
        );
    }

    #[test]
    fn critic_findings_are_preserved_without_vetoing_a_valid_partition() {
        let partition = r#"{"tasks":[{"title":"Consumer","covers":["source-002"]},{"title":"Foundation","covers":["source-001"]}]}"#;
        let mut model = ScriptedModel::new([
            partition,
            r#"{"decision":"revise","issues":[{"code":"bad_order","task_ids":["task-01","task-02"],"source_ids":["source-001","source-002"]}]}"#,
        ]);
        let TaskPlanningOutcome::Accepted(accepted) = run(
            &mut model,
            "Add storage. Expose health",
            TaskSourceIntent::Build,
        ) else {
            panic!("valid deterministic partition must pass");
        };
        assert_eq!(accepted.counters.planning_attempts, 1);
        assert_eq!(accepted.counters.advisory_calls, 1);
        assert!(accepted.transcript.summary.contains("advisory evidence"));
        assert_eq!(accepted.transcript.attempts.len(), 2);
    }

    #[test]
    fn explicit_before_order_is_a_deterministic_gate() {
        let mut model = ScriptedModel::new([
            r#"{"tasks":[{"title":"API","covers":["source-002"]},{"title":"Storage","covers":["source-001"]}]}"#,
            r#"{"tasks":[{"title":"Storage","covers":["source-001"]},{"title":"API","covers":["source-002"]}]}"#,
            r#"{"decision":"accept","issues":[]}"#,
        ]);
        let TaskPlanningOutcome::Accepted(accepted) = run(
            &mut model,
            "Add storage before exposing the API",
            TaskSourceIntent::Build,
        ) else {
            panic!("corrected explicit order must pass");
        };
        assert_eq!(accepted.counters.planning_attempts, 2);
        assert!(
            accepted.transcript.attempts[0]
                .failure
                .as_deref()
                .unwrap()
                .contains("explicit source order")
        );
    }

    #[test]
    fn invalid_partitions_fail_soft_to_the_original_build() {
        let mut model = ScriptedModel::new(["{}", "{}"]);
        let TaskPlanningOutcome::OneBuild(fallback) =
            run(&mut model, "Fix average", TaskSourceIntent::Build)
        else {
            panic!("invalid automatic planning must fail soft");
        };
        assert_eq!(
            fallback.transcript.decision,
            TaskPlanningDecision::OneBuildPlannerFallback
        );
        assert_eq!(fallback.transcript.attempts.len(), 2);
    }

    #[test]
    fn compact_schemas_expose_only_partition_and_veto_fields() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = qualification(TaskSourceIntent::Build);
        let input = TaskPlanningInput {
            objective: "Add storage. Expose health",
            repository_context: "{}",
            source_intent: TaskSourceIntent::Build,
            model_sha256: &"a".repeat(64),
            qualification: &qualification,
            policy: &policy,
        };
        let schema = planner_schema(&input);
        crate::inference::flashmoe::validate_native_tool_schema(&schema).unwrap();
        crate::inference::flashmoe::validate_llguidance_json_schema(&schema).unwrap();
        let properties = schema["properties"]["tasks"]["items"]["properties"]
            .as_object()
            .unwrap();
        assert_eq!(
            properties.keys().cloned().collect::<BTreeSet<_>>(),
            BTreeSet::from(["covers".to_string(), "title".to_string()])
        );
    }

    #[test]
    fn cancellation_remains_terminal_instead_of_starting_a_build() {
        let mut model = ScriptedModel::new([]);
        model.cancelled = true;
        let TaskPlanningOutcome::Rejected(rejected) =
            run(&mut model, "Fix average", TaskSourceIntent::Build)
        else {
            panic!("cancellation must stop work");
        };
        assert_eq!(rejected.outcome, TaskPlanRejectionOutcome::Cancelled);
        assert!(model.calls.is_empty());
    }
}
