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

const TASK_PLANNER_TEMPLATE_VERSION: &str = "pb-task-planner-template-v14";
const TASK_PLANNER_PROTOCOL_VERSION: &str = "pb-task-request-list-v13";
const MAX_TASK_PLANNING_INPUT_BYTES: usize = 64 * 1024;
const MAX_TASK_PLANNER_OUTPUT_TOKENS: usize = 512;
const MAX_RETRY_FEEDBACK_CHARS: usize = 1_000;
const MAX_PLANNING_FACTS: usize = 32;
// llama.cpp's grammar parser rejects bounded repetitions above 2,000. Keep the shared schema
// comfortably below that backend limit so the same contract compiles in llama.cpp and FlashMoe.
const MAX_DESCRIPTION_CHARS: usize = 1_024;
const MAX_TASK_REQUEST_CHARS: usize = 1_024;
const MAX_DERIVED_TITLE_CHARS: usize = 96;
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
    tasks: Vec<String>,
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
        let behavior_sources = behavior_evidence
            .iter()
            .enumerate()
            .map(|(index, description)| {
                (
                    source_id(index),
                    description.clone(),
                    requirement_ids_by_description
                        .get(description)
                        .expect("behavior evidence is a request clause")
                        .clone(),
                )
            })
            .collect::<Vec<_>>();
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

        for (task_index, task_request) in self.tasks.into_iter().enumerate() {
            let id = format!("task-{:02}", task_index + 1);
            let task_request = task_request.trim().to_string();
            if task_request.is_empty() {
                bail!("Task request text must not be empty");
            }
            let mut task_requirement_ids = constraint_requirement_ids.clone();
            let mut matched_sources = Vec::new();
            for (source_id, description, requirement_id) in &behavior_sources {
                let occurrences = task_request.match_indices(description).count();
                if occurrences > 1 {
                    bail!(
                        "Task request repeats controller source clause '{description}' more than once"
                    );
                }
                if occurrences == 0 {
                    continue;
                }
                if !covered.insert(source_id.clone()) {
                    bail!("controller source clause '{description}' is assigned more than once");
                }
                source_task_positions.insert(source_id.clone(), task_index);
                task_requirement_ids.push(requirement_id.clone());
                matched_sources.push(description.clone());
            }
            if matched_sources.is_empty() {
                bail!(
                    "Each Task request in a multi-Task result must contain at least one controller source clause verbatim"
                );
            }
            if matched_sources.len() > MAX_BUILD_TASK_BEHAVIOR_CLAUSES {
                bail!(
                    "Task request covers {} source clauses; maximum is {MAX_BUILD_TASK_BEHAVIOR_CLAUSES}",
                    matched_sources.len()
                );
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
                title: derive_task_title(&task_request),
                description: task_request,
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

        let expected = behavior_sources
            .iter()
            .map(|(source_id, _, _)| source_id.clone())
            .collect::<BTreeSet<_>>();
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
        return accepted_task_plan(
            input,
            plan,
            counters,
            transcript,
            "The ordered Task requests passed controller validation",
        );
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
        "{TASK_PLANNER_TEMPLATE_VERSION}\nReturn only JSON matching the schema. tasks is an ordered list of requests that pb will execute and commit one at a time using the normal Build workflow. Each string must be an independently deliverable outcome for a smaller implementation model, not a list of files or edits. In a multi-Task result, copy every controller source clause verbatim into exactly one Task request; you may add only the context needed to make that request self-contained. Put foundations and migrations before consumers. Do not create separate test, documentation, review, integration, or cleanup Tasks because pb attaches request-wide constraints to the behavior-owning Tasks. If the work is tightly coupled or one Task is sufficient, return exactly one string; pb will then run the original Build unchanged. Each multi-Task request may contain at most {MAX_BUILD_TASK_BEHAVIOR_CLAUSES} source clauses.\n\nAttempt: {attempt}\nOriginal Build request:\n{}\n\nController source clauses to copy verbatim:\n{}\n\nRepository component outline:\n{}\n{retry}",
        input.objective,
        serde_json::to_string_pretty(&sources).expect("source clauses serialize"),
        input.repository_context
    )
}

fn planner_schema(input: &TaskPlanningInput<'_>) -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "tasks": {
                "type": "array",
                "items": bounded_string(MAX_TASK_REQUEST_CHARS),
                "minItems": 1,
                "maxItems": input.policy.aggregate.max_tasks
            }
        },
        "required": ["tasks"]
    })
}

fn source_clauses(objective: &str) -> Vec<String> {
    behavior_evidence_clauses(objective)
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
                    "each Task request owns bounded source clauses and passed controller validation"
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

fn derive_task_title(request: &str) -> String {
    let compact = request.split_whitespace().collect::<Vec<_>>().join(" ");
    let sentence = compact
        .split_once(". ")
        .map(|(first, _)| first)
        .unwrap_or(compact.as_str())
        .trim_end_matches(['.', ';', ':'])
        .trim();
    let mut title = sentence
        .chars()
        .take(MAX_DERIVED_TITLE_CHARS)
        .collect::<String>();
    if sentence.chars().count() > MAX_DERIVED_TITLE_CHARS {
        title = title
            .chars()
            .take(MAX_DERIVED_TITLE_CHARS.saturating_sub(1))
            .collect::<String>()
            .trim_end()
            .to_string();
        title.push('…');
    }
    let Some(first) = title.chars().next() else {
        return "Build Task".to_string();
    };
    first.to_uppercase().chain(title.chars().skip(1)).collect()
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
    fn one_task_keeps_the_original_build() {
        let mut model = ScriptedModel::new([r#"{"tasks":["Rewrite this request"]}"#]);
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
    fn task_request_list_compiles_controller_owned_artifacts() {
        let mut model = ScriptedModel::new([
            r#"{"tasks":["Add durable storage. Commit the storage foundation before API work.","Expose health using the committed storage."]}"#,
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
        assert_eq!(
            accepted.plan.artifact.tasks[0].description,
            "Add durable storage. Commit the storage foundation before API work."
        );
        assert_eq!(accepted.plan.artifact.tasks[0].title, "Add durable storage");
        assert_eq!(accepted.plan.artifact.acceptance.len(), 2);
        assert_eq!(
            accepted.review.artifact.verdict,
            TaskPlanReviewVerdict::Pass
        );
        assert_eq!(accepted.counters.model_invocations, 1);
        assert_eq!(accepted.counters.advisory_calls, 0);
        assert_eq!(accepted.transcript.attempts.len(), 1);
        accepted.review.artifact.validate(&accepted.plan).unwrap();
    }

    #[test]
    fn duplicate_or_missing_source_ownership_gets_one_revision() {
        let mut model = ScriptedModel::new([
            r#"{"tasks":["Add storage","Repeat Add storage"]}"#,
            r#"{"tasks":["Add storage","Expose health"]}"#,
        ]);
        let TaskPlanningOutcome::Accepted(accepted) = run(
            &mut model,
            "Add storage. Expose health",
            TaskSourceIntent::Build,
        ) else {
            panic!("corrected partition must pass");
        };
        assert_eq!(accepted.counters.planning_attempts, 2);
        assert_eq!(accepted.transcript.attempts.len(), 2);
        assert!(accepted.transcript.attempts[0].failure.is_some());
    }

    #[test]
    fn controller_attaches_decomposition_wide_constraints_to_each_task() {
        let mut model = ScriptedModel::new([r#"{"tasks":["Add storage","Expose health"]}"#]);
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
            r#"{"tasks":["Add durable import lifecycle storage and a rollback-safe migration. Commit the foundation first.","exposing compatible status and cancel APIs using the committed storage."]}"#,
        ]);
        let TaskPlanningOutcome::Accepted(accepted) =
            run(&mut model, objective, TaskSourceIntent::Build)
        else {
            panic!("ordered partition must pass");
        };
        let constraint_id = accepted.plan.artifact.requirements[2].id.clone();
        assert_eq!(accepted.plan.artifact.tasks[1].depends_on, vec!["task-01"]);
        assert_eq!(
            accepted.plan.artifact.tasks[1].title,
            "Exposing compatible status and cancel APIs using the committed storage"
        );
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
    fn valid_task_requests_do_not_invoke_a_model_critic() {
        let mut model = ScriptedModel::new([r#"{"tasks":["Add storage","Expose health"]}"#]);
        let TaskPlanningOutcome::Accepted(accepted) = run(
            &mut model,
            "Add storage. Expose health",
            TaskSourceIntent::Build,
        ) else {
            panic!("valid deterministic partition must pass");
        };
        assert_eq!(accepted.counters.planning_attempts, 1);
        assert_eq!(accepted.counters.advisory_calls, 0);
        assert_eq!(accepted.transcript.attempts.len(), 1);
        assert_eq!(model.calls, vec![TaskPlanningRole::Planner]);
    }

    #[test]
    fn explicit_before_order_is_a_deterministic_gate() {
        let mut model = ScriptedModel::new([
            r#"{"tasks":["exposing the API","Add storage"]}"#,
            r#"{"tasks":["Add storage","exposing the API"]}"#,
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
    fn compact_schema_is_only_a_bounded_list_of_request_strings() {
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
        assert_eq!(schema["properties"]["tasks"]["items"]["type"], "string");
        assert_eq!(
            schema["properties"]["tasks"]["items"]["maxLength"],
            MAX_TASK_REQUEST_CHARS
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
