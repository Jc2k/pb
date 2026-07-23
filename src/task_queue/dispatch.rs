use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::workflow::{
    CompiledWorkflowPolicy, ConversationHandoff, DeliveryPolicy, TurnIntent, WorkflowCheckpoint,
    WorkflowConfigDocument, WorkflowLimits, WorkflowOutcome, WorkflowStage,
};

use super::{TaskKind, TaskRepositoryState, TaskRequest, TaskResult, TaskStopDisposition};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BuildTaskProjection {
    pub workflow_id: String,
    pub turn_id: String,
    pub task: String,
    pub workflow_policy: CompiledWorkflowPolicy,
    pub max_steps: usize,
    pub handoff: ConversationHandoff,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GoalTaskProjection {
    pub goal_id: String,
    pub objective: String,
    pub criteria: Vec<crate::goal::GoalCriterionInput>,
    pub continuation: crate::goal::GoalContinuationPolicy,
    pub budget: crate::goal::GoalBudget,
    pub policy: crate::goal::CompiledGoalPolicy,
}

pub fn project_goal_task(
    request: &TaskRequest,
    goal_policy: &crate::goal::CompiledGoalPolicy,
) -> Result<GoalTaskProjection> {
    if request.kind != TaskKind::Goal {
        bail!("only a Goal Task can be projected into a Goal run");
    }
    request.base_repository.validate()?;
    goal_policy.validate()?;
    let contract = request
        .goal_contract
        .as_ref()
        .context("Goal Task request has no Goal contract")?;
    let criteria = contract
        .criteria
        .iter()
        .map(crate::goal::GoalCriterionInput::from)
        .collect::<Vec<_>>();
    let constrained_policy = crate::goal::GoalConfigDocument {
        version: goal_policy.version,
        limits: crate::goal::GoalBudget {
            max_milestones: request
                .budget
                .max_workflows
                .min(goal_policy.limits.max_milestones),
            max_workflows: request
                .budget
                .max_workflows
                .min(goal_policy.limits.max_workflows),
            total_model_invocations: request
                .budget
                .total_model_invocations
                .min(goal_policy.limits.total_model_invocations),
            total_generated_tokens: request
                .budget
                .total_generated_tokens
                .min(goal_policy.limits.total_generated_tokens),
            wall_time_minutes: request
                .budget
                .wall_time_minutes
                .min(goal_policy.limits.wall_time_minutes),
        },
    }
    .compile()?;
    let requested = crate::goal::GoalBudget {
        max_milestones: criteria.len(),
        max_workflows: constrained_policy.limits.max_workflows,
        total_model_invocations: constrained_policy.limits.total_model_invocations,
        total_generated_tokens: constrained_policy.limits.total_generated_tokens,
        wall_time_minutes: constrained_policy.limits.wall_time_minutes,
    };
    let budget = constrained_policy.budget(Some(requested))?;
    Ok(GoalTaskProjection {
        goal_id: request.child_id.clone(),
        objective: contract.objective.clone(),
        criteria,
        continuation: contract.continuation,
        budget,
        policy: constrained_policy,
    })
}

pub fn project_build_task(
    request: &TaskRequest,
    workflow_policy: &CompiledWorkflowPolicy,
) -> Result<BuildTaskProjection> {
    if request.kind != TaskKind::Build {
        bail!("only a Build Task can be projected into a strict workflow");
    }
    request.base_repository.validate()?;
    workflow_policy.validate()?;
    let limits = WorkflowLimits {
        stage_steps: workflow_policy
            .limits
            .stage_steps
            .min(request.budget.stage_steps),
        total_model_invocations: workflow_policy
            .limits
            .total_model_invocations
            .min(request.budget.total_model_invocations),
        total_generated_tokens: workflow_policy
            .limits
            .total_generated_tokens
            .min(request.budget.total_generated_tokens),
        advisory_calls: workflow_policy
            .limits
            .advisory_calls
            .min(request.budget.advisory_calls),
        plan_cycles: workflow_policy
            .limits
            .plan_cycles
            .min(request.budget.plan_cycles),
        repair_cycles: workflow_policy
            .limits
            .repair_cycles
            .min(request.budget.repair_cycles),
        review_paths: workflow_policy.limits.review_paths,
        review_diff_bytes: workflow_policy.limits.review_diff_bytes,
    };
    let policy = WorkflowConfigDocument {
        version: workflow_policy.version,
        delivery: DeliveryPolicy::Strict,
        default_intent: TurnIntent::Deliver,
        limits,
    }
    .compile()?;
    let requirements = request
        .requirements
        .iter()
        .map(|requirement| format!("{}: {}", requirement.id, requirement.description))
        .collect::<Vec<_>>();
    let acceptance = request
        .acceptance
        .iter()
        .map(|fact| format!("{}: {}", fact.id, fact.description))
        .collect::<Vec<_>>();
    let task = render_build_task(request, &requirements, &acceptance);
    let mut constraints = vec![
        format!("High-level Task id: {}", request.task_id),
        format!("Requirements: {}", requirements.join(" | ")),
        format!("Acceptance facts: {}", acceptance.join(" | ")),
    ];
    if !request.scope_hints.is_empty() {
        constraints.push(format!(
            "Advisory scope hints (not path authority): {}",
            request.scope_hints.join(", ")
        ));
    }
    Ok(BuildTaskProjection {
        workflow_id: request.child_id.clone(),
        turn_id: request.id.clone(),
        task: task.clone(),
        workflow_policy: policy,
        max_steps: workflow_policy
            .limits
            .stage_steps
            .min(request.budget.stage_steps),
        handoff: ConversationHandoff {
            source_turn_ids: vec![request.source_turn_id.clone(), request.id.clone()],
            task_summary: task,
            constraints,
            ..ConversationHandoff::default()
        },
    })
}

pub fn build_task_result(
    request: &TaskRequest,
    checkpoint: &WorkflowCheckpoint,
    terminal_repository: TaskRepositoryState,
) -> Result<TaskResult> {
    checkpoint.validate()?;
    if checkpoint.run.id != request.child_id || checkpoint.run.source_turn_id != request.id {
        bail!("workflow checkpoint is not bound to the active Build Task request");
    }
    if checkpoint.run.stage != WorkflowStage::Ready {
        bail!("Build Task workflow has not reached Ready");
    }
    let (no_change, commits) = match checkpoint.run.outcome {
        Some(WorkflowOutcome::Ready) => (
            false,
            vec![
                checkpoint
                    .run
                    .commit
                    .as_ref()
                    .context("ready Build Task has no managed commit")?
                    .oid
                    .clone(),
            ],
        ),
        Some(WorkflowOutcome::NoChange) => (true, Vec::new()),
        other => bail!("Build Task workflow has no successful terminal outcome: {other:?}"),
    };
    let mut evidence_refs = vec![format!("workflow:sha256:{}", checkpoint.sha256)];
    if let Some(evidence) = &checkpoint.run.ready_evidence {
        evidence_refs.push(format!("ready-evidence:sha256:{}", evidence.sha256()?));
    }
    Ok(TaskResult {
        base_repository: request.base_repository.clone(),
        terminal_repository,
        commits,
        no_change,
        satisfied_acceptance_ids: request
            .acceptance
            .iter()
            .map(|fact| fact.id.clone())
            .collect(),
        evidence_refs,
        summary: if no_change {
            format!("Build Task '{}' verified no change", request.title)
        } else {
            format!(
                "Build Task '{}' delivered its managed commit",
                request.title
            )
        },
    })
}

pub fn build_task_stop(
    checkpoint: &WorkflowCheckpoint,
) -> Result<Option<(TaskStopDisposition, String)>> {
    checkpoint.validate()?;
    if checkpoint.run.stage == WorkflowStage::Blocked {
        return Ok(Some((
            TaskStopDisposition::Blocked,
            checkpoint
                .run
                .blocked_reason
                .clone()
                .unwrap_or_else(|| "Build Task workflow blocked".to_string()),
        )));
    }
    if checkpoint.run.stage == WorkflowStage::Failed {
        return Ok(Some((
            TaskStopDisposition::Failed,
            checkpoint
                .run
                .blocked_reason
                .clone()
                .unwrap_or_else(|| "Build Task workflow failed".to_string()),
        )));
    }
    Ok(None)
}

pub fn goal_task_result(
    request: &TaskRequest,
    checkpoint: &crate::goal::GoalCheckpoint,
    terminal_repository: TaskRepositoryState,
) -> Result<TaskResult> {
    checkpoint.validate()?;
    if request.kind != TaskKind::Goal || checkpoint.run.id != request.child_id {
        bail!("Goal checkpoint is not bound to the active Goal Task request");
    }
    if checkpoint.run.stage != crate::goal::GoalStage::Completed
        || checkpoint.run.outcome != Some(crate::goal::GoalOutcome::Complete)
    {
        bail!("Goal Task has not reached accepted Goal completion");
    }
    let commits = checkpoint
        .run
        .milestones
        .iter()
        .filter_map(|milestone| {
            milestone
                .workflow
                .as_ref()
                .and_then(|workflow| workflow.run.commit.as_ref())
                .map(|commit| commit.oid.clone())
        })
        .collect::<Vec<_>>();
    let no_change = commits.is_empty();
    Ok(TaskResult {
        base_repository: request.base_repository.clone(),
        terminal_repository,
        commits,
        no_change,
        satisfied_acceptance_ids: request
            .acceptance
            .iter()
            .map(|fact| fact.id.clone())
            .collect(),
        evidence_refs: vec![format!("goal:sha256:{}", checkpoint.sha256)],
        summary: format!("Goal Task '{}' reached accepted completion", request.title),
    })
}

pub fn goal_task_stop(
    checkpoint: &crate::goal::GoalCheckpoint,
) -> Result<Option<(TaskStopDisposition, String)>> {
    checkpoint.validate()?;
    let result = match checkpoint.run.stage {
        crate::goal::GoalStage::Blocked => Some((
            TaskStopDisposition::Blocked,
            checkpoint
                .run
                .blocked_reason
                .clone()
                .unwrap_or_else(|| "Goal Task blocked".to_string()),
        )),
        crate::goal::GoalStage::Failed => Some((
            TaskStopDisposition::Failed,
            checkpoint
                .run
                .blocked_reason
                .clone()
                .unwrap_or_else(|| "Goal Task failed".to_string()),
        )),
        _ => None,
    };
    Ok(result)
}

fn render_build_task(
    request: &TaskRequest,
    requirements: &[String],
    acceptance: &[String],
) -> String {
    let mut rendered = format!(
        "{}\n\nTask outcome:\n{}\n\nRequirements:\n- {}\n\nAcceptance facts:\n- {}",
        request.title,
        request.objective,
        requirements.join("\n- "),
        acceptance.join("\n- ")
    );
    if !request.scope_hints.is_empty() {
        rendered.push_str(&format!(
            "\n\nAdvisory scope hints (the Build plan must inspect current HEAD and name actual paths):\n- {}",
            request.scope_hints.join("\n- ")
        ));
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task_queue::{TaskBudget, TaskEffort, TaskSpec};

    fn repository() -> TaskRepositoryState {
        TaskRepositoryState {
            head: "a".repeat(40),
            index_sha256: "b".repeat(64),
            refs_sha256: "c".repeat(64),
            content_sha256: "d".repeat(64),
        }
    }

    fn request() -> TaskRequest {
        TaskRequest {
            id: "task-request:multi:t1:1".to_string(),
            multi_task_id: "multi".to_string(),
            source_turn_id: "turn-user".to_string(),
            task_id: "t1".to_string(),
            child_id: "multi:t1:1".to_string(),
            workdir: "/workspace".to_string(),
            kind: TaskKind::Build,
            title: "Persist imports".to_string(),
            objective: "Persist lifecycle state and migration".to_string(),
            requirements: vec![super::super::TaskRequirement {
                id: "r1".to_string(),
                description: "State survives restart".to_string(),
            }],
            acceptance: vec![super::super::TaskAcceptance {
                id: "a1".to_string(),
                description: "Storage tests pass".to_string(),
            }],
            scope_hints: vec!["src/storage".to_string()],
            goal_contract: None,
            base_repository: repository(),
            budget: TaskBudget {
                max_workflows: 1,
                stage_steps: 4,
                total_model_invocations: 6,
                total_generated_tokens: 3_000,
                advisory_calls: 2,
                plan_cycles: 1,
                repair_cycles: 1,
                wall_time_minutes: 20,
            },
            attempt: 1,
        }
    }

    #[test]
    fn build_projection_clamps_native_policy_and_keeps_scope_advisory() {
        let base = WorkflowConfigDocument::default().compile().unwrap();
        let projection = project_build_task(&request(), &base).unwrap();
        assert_eq!(projection.workflow_id, "multi:t1:1");
        assert_eq!(projection.turn_id, "task-request:multi:t1:1");
        assert_eq!(projection.workflow_policy.limits.stage_steps, 4);
        assert_eq!(projection.workflow_policy.limits.total_model_invocations, 6);
        assert_eq!(
            projection.workflow_policy.limits.total_generated_tokens,
            3_000
        );
        assert_eq!(projection.workflow_policy.limits.advisory_calls, 2);
        assert!(projection.task.contains("Storage tests pass"));
        assert!(projection.task.contains("inspect current HEAD"));
        assert!(
            projection
                .handoff
                .constraints
                .iter()
                .any(|value| value.contains("not path authority"))
        );
    }

    #[test]
    fn goal_task_cannot_enter_build_projection() {
        let mut request = request();
        request.kind = TaskKind::Goal;
        let base = WorkflowConfigDocument::default().compile().unwrap();
        assert!(project_build_task(&request, &base).is_err());
    }

    #[test]
    fn task_spec_remains_coarse_until_projection() {
        let spec = TaskSpec {
            id: "t1".to_string(),
            title: "Persist imports".to_string(),
            description: "Persist lifecycle state".to_string(),
            requirement_ids: vec!["r1".to_string()],
            depends_on: Vec::new(),
            acceptance_ids: vec!["a1".to_string()],
            scope_hints: vec!["storage".to_string()],
            effort: TaskEffort::Small,
            kind: TaskKind::Build,
            goal_contract: None,
            budget: request().budget,
        };
        let serialized = serde_json::to_value(&spec).unwrap();
        assert!(serialized.get("steps").is_none());
        assert!(serialized.get("work_units").is_none());
    }
}
