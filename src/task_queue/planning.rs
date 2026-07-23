use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

use crate::workflow::ArtifactEnvelope;

use super::{
    CompiledTaskPolicy, TaskCoordinationCounters, TaskPlanArtifact, TaskPlanAuthority,
    TaskPlanProposal, TaskPlanReviewArtifact, TaskPlanReviewVerdict, TaskPlannerQualification,
    TaskSourceIntent,
};

const TASK_PLANNER_TEMPLATE_VERSION: &str = "pb-task-planner-template-v2";
const TASK_PLANNER_PROTOCOL_VERSION: &str = "pb-task-plan-proposal-review-v1";
const MAX_TASK_PLANNING_INPUT_BYTES: usize = 64 * 1024;
const MAX_TASK_PLANNER_OUTPUT_TOKENS: usize = 4_096;
const MAX_TASK_REVIEW_OUTPUT_TOKENS: usize = 2_048;
const MAX_RETRY_FEEDBACK_CHARS: usize = 4_000;

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

pub trait TaskPlanningModel {
    fn generate(
        &mut self,
        role: TaskPlanningRole,
        prompt: &str,
        max_tokens: usize,
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
        let output = match model.generate(TaskPlanningRole::Planner, &prompt, max_tokens) {
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
        let proposal = match parse_json::<TaskPlanProposal>(&output.text)
            .and_then(|proposal| {
                let canonical = serde_json::to_string(&proposal)?;
                TaskPlanProposal::from_model_json(&canonical).map_err(anyhow::Error::from)
            })
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
        let started = Instant::now();
        let output = match model.generate(TaskPlanningRole::Reviewer, &review_prompt, max_tokens) {
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
            review
                .validate(&plan)
                .map(|_| review)
                .map_err(anyhow::Error::from)
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
        format!("\nThe previous attempt was rejected. Correct these facts:\n{feedback}\n")
    };
    format!(
        "{TASK_PLANNER_TEMPLATE_VERSION}\nYou are the high-level Task planner. Decompose the request into the smallest useful sequence of outcome-shaped Tasks for a smaller implementation model. Return between 1 and {} Tasks; combine closely coupled behavior instead of exceeding this ceiling. These are Tasks, not Build plan steps: do not name exact edits or invent a final testing/documentation catch-all. Tests and docs belong with the behavior they verify. Return exactly one JSON object and no prose. Never include numeric budgets; use effort small, medium, or large.\n\n{kind_rule}\nEach requirement and acceptance id must be mapped by at least one Task. Dependencies must be acyclic and point to earlier prerequisites. A build Task has no goal_contract. A goal Task has goal_contract {{\"objective\":string,\"criteria\":[{{\"text\":string,\"verifier\":\"workflow_ready\"|\"review_required\"|\"user_confirmation\"}}],\"continuation\":\"review_plan_then_automatic\"|\"manual_milestones\"|\"automatic_within_limits\"}}.\n\nSchema: {{\"objective\":string,\"requirements\":[{{\"id\":string,\"description\":string}}],\"tasks\":[{{\"id\":string,\"title\":string,\"description\":string,\"requirement_ids\":[string],\"depends_on\":[string],\"acceptance_ids\":[string],\"scope_hints\":[string],\"effort\":\"small\"|\"medium\"|\"large\",\"kind\":\"build\"|\"goal\",\"goal_contract\":object|null}}],\"acceptance\":[{{\"id\":string,\"description\":string}}],\"risks\":[{{\"description\":string}}]}}\n\nAttempt: {attempt}\nRequest:\n{}\n\nBounded repository context:\n{}\n{retry}",
        input.policy.aggregate.max_tasks, input.objective, input.repository_context
    )
}

fn reviewer_prompt(
    input: &TaskPlanningInput<'_>,
    plan: &ArtifactEnvelope<TaskPlanArtifact>,
) -> Result<String> {
    Ok(format!(
        "{TASK_PLANNER_TEMPLATE_VERSION}\nYou are a fresh Task-plan critic. Assess requirement coverage, Task size for a smaller model, dependency and migration/rollback order, observable acceptance, commit boundaries, catch-all Tasks, Build-versus-Goal choice, and qualitative effort. Return exactly one JSON object and no prose. verdict pass cannot contain blocking challenges; verdict revise must contain at least one blocking challenge.\n\nSchema: {{\"task_plan_sha256\":\"{}\",\"verdict\":\"pass\"|\"revise\",\"challenges\":[{{\"id\":string,\"code\":string,\"description\":string,\"severity\":\"blocking\"|\"advisory\",\"task_ids\":[string]}}]}}\n\nOriginal request:\n{}\n\nTask plan:\n{}",
        plan.sha256,
        input.objective,
        serde_json::to_string_pretty(&plan.artifact)?
    ))
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

    fn proposal(id: &str) -> serde_json::Value {
        serde_json::json!({
            "objective": "Fix average",
            "requirements": [{"id": "r1", "description": "Use the correct divisor"}],
            "tasks": [{
                "id": id,
                "title": "Fix average",
                "description": "Correct the divisor and add its regression test",
                "requirement_ids": ["r1"],
                "depends_on": [],
                "acceptance_ids": ["a1"],
                "scope_hints": ["src"],
                "effort": "small",
                "kind": "build"
            }],
            "acceptance": [{"id": "a1", "description": "Regression test passes"}],
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
        serde_json::json!({
            "task_plan_sha256": envelope.sha256,
            "verdict": verdict,
            "challenges": if verdict == "pass" { serde_json::json!([]) } else { serde_json::json!([{
                "id": "c1",
                "code": "task_too_broad",
                "description": "Split the catch-all Task",
                "severity": "blocking",
                "task_ids": ["t1"]
            }]) }
        })
        .to_string()
    }

    fn compiled(value: &serde_json::Value, policy: &CompiledTaskPolicy) -> TaskPlanArtifact {
        serde_json::from_value::<TaskPlanProposal>(value.clone())
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
        let value = proposal("t1");
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
            "objective": "Fix average",
            "requirements": [{"id": "r1", "description": "Fix"}],
            "tasks": [{"id": "t1", "title": "Fix", "description": "Fix", "requirement_ids": ["missing"], "acceptance_ids": ["a1"], "effort": "small", "kind": "build"}],
            "acceptance": [{"id": "a1", "description": "Pass"}]
        });
        let valid = proposal("t1");
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
    fn two_invalid_attempts_stop_with_explicit_recovery_actions() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = qualification(false);
        let invalid = "{\"objective\":\"bad\",\"requirements\":[],\"tasks\":[],\"acceptance\":[]}";
        let mut model = ScriptedModel::new([invalid.to_string(), invalid.to_string()]);
        let TaskPlanningOutcome::Rejected(rejected) =
            plan_tasks(&mut model, input(&policy, &qualification))
        else {
            panic!("invalid plans must be rejected");
        };
        assert_eq!(
            rejected.outcome,
            TaskPlanRejectionOutcome::AttemptsExhausted
        );
        assert_eq!(rejected.attempts, 2);
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
        let first = proposal("t1");
        let second = proposal("t2");
        let first_plan = compiled(&first, &policy);
        let second_plan = compiled(&second, &policy);
        let mut revise =
            serde_json::from_str::<serde_json::Value>(&review(&first_plan, 1, "revise")).unwrap();
        revise["challenges"][0]["task_ids"] = serde_json::json!(["t1"]);
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
        assert_eq!(accepted.plan.artifact.tasks[0].id, "t2");
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
    fn model_authored_numeric_budget_is_rejected_on_both_attempts() {
        let policy = TaskConfigDocument::default().compile().unwrap();
        let qualification = qualification(false);
        let mut value = proposal("t1");
        value["tasks"][0]["budget"] = serde_json::json!({"generated_tokens": 999999});
        let mut model = ScriptedModel::new([value.to_string(), value.to_string()]);
        let TaskPlanningOutcome::Rejected(rejected) =
            plan_tasks(&mut model, input(&policy, &qualification))
        else {
            panic!("model budgets must fail");
        };
        assert_eq!(rejected.attempts, 2);
        assert!(rejected.failures.iter().all(|failure| {
            failure
                .reason
                .contains("model Task proposals cannot contain executable numeric budgets")
                || failure.reason.contains("unknown field `budget`")
        }));
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
