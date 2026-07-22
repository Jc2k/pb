//! Deterministic control-plane fixtures for the hidden harness.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::agent_core::{
    AgentProfile, AgentRequest, LocalModelEvalEngine, LocalModelEvalOutcome, ScriptedAgentOutcome,
    ScriptedCompletion, StageContext, run_local_model_eval_steps, run_scripted_agent_steps,
    run_scripted_delivery_workflow, run_scripted_stage,
};
use crate::events::{AgentEvent, ContractStatus};
use crate::{HarnessEvalArgs, HarnessEvalSuite, config::UserConfig};

const CONTROL_FIXTURES: &str = include_str!("../fixtures/harness-control-fixtures.json");
const HARNESS_EVAL_SCHEMA_VERSION: u32 = 3;
const SCRIPTED_EVAL_CONTEXT_SIZE: u32 = 8_192;
const MAX_TOOL_TRACE_NAME_CHARS: usize = 120;
const MAX_TOOL_TRACE_ARGUMENT_CHARS: usize = 600;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlFixtureCorpus {
    pub version: u32,
    pub fixtures: Vec<ControlFixture>,
    #[serde(default)]
    pub small_model_fixtures: Vec<String>,
    pub workflow_fixtures: Vec<WorkflowControlFixture>,
    #[serde(default)]
    pub goal_fixtures: Vec<GoalControlFixture>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalControlFixture {
    pub id: String,
    pub hypothesis: String,
    pub assertion: GoalControlAssertion,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalControlAssertion {
    ExactPlanApproval,
    ModelToolAuthorityBound,
    SequentialMilestones,
    PauseCheckpointResume,
    AmendmentPreservesEvidence,
    CompletionBasisBound,
    BudgetAndCancellationAccounting,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowControlFixture {
    pub id: String,
    pub hypothesis: String,
    pub assertion: WorkflowControlAssertion,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowControlAssertion {
    DiscussionNoBranch,
    DiscussionNoMutation,
    ExplicitDeliveryStartsPlanning,
    PlanningRequiresSubmission,
    PlanningAuthorityIsReadOnly,
    PlanStructureValidated,
    PlanReviewHashBound,
    PlanReviewEvidenceRequired,
    PlanChallengeForcesRevision,
    ImplementationRequiresAcceptedPlan,
    ImplementationCanReplan,
    RunCommandCannotBypassGates,
    CheckFailureForcesRepair,
    PostCheckMutationInvalidatesEvidence,
    CodeReviewFingerprintBound,
    CodeReviewPathEvidenceRequired,
    CodeFindingForcesRepair,
    PostReviewMutationBlocksCommit,
    DelegationCannotEscalateAuthority,
    WorkflowBudgetsAreGlobal,
    ManagedCommitIsTaskOwned,
    NoChangeCreatesNoCommit,
    ResumePreservesStageAndBudget,
    WebHarnessProjectionParity,
    LegacyStateHasNoStrictClaim,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlFixture {
    pub id: String,
    pub hypothesis: String,
    pub profile: AgentProfile,
    pub max_steps: usize,
    pub completion_supported: bool,
    pub expected: ControlFixtureExpectation,
    #[serde(default)]
    pub contract: Option<crate::harness_contract::HarnessContractDocument>,
    #[serde(default)]
    pub workspace: Option<crate::workspace::WorkspaceConfigDocument>,
    #[serde(default)]
    pub tool_allowlist: Vec<String>,
    #[serde(default)]
    pub initial_files: BTreeMap<String, String>,
    #[serde(default)]
    pub resumed_files: BTreeMap<String, String>,
    pub turns: Vec<ControlFixtureTurn>,
    #[serde(default)]
    pub observe_paths: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlFixtureExpectation {
    pub reached_final: bool,
    pub contract_status: ContractStatus,
    pub verified_completed: bool,
    pub termination_reason: String,
    pub llm_invocations: usize,
    pub tool_calls: usize,
    pub false_completion: bool,
    #[serde(default)]
    pub named_check_compliance: Option<bool>,
    #[serde(default)]
    pub observed_paths: BTreeMap<String, bool>,
    #[serde(default)]
    pub handoff_outcome: Option<crate::events::HandoffOutcome>,
    #[serde(default)]
    pub selected_checks: Option<Vec<String>>,
    #[serde(default)]
    pub executed_checks: Option<usize>,
    #[serde(default)]
    pub reused_checks: Option<usize>,
    #[serde(default)]
    pub executor_starts: Option<Vec<String>>,
    #[serde(default)]
    pub repair_turns: Option<usize>,
    #[serde(default)]
    pub commit_disposition: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HarnessEvalConfiguration {
    pub mode: String,
    pub backend: String,
    #[serde(default)]
    pub suite: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_dir: Option<String>,
    pub max_tokens: i32,
    pub ctx_size: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads_batch: Option<i32>,
    pub gpu_layers: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub seed: u32,
    pub flashmoe_resource_policy_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_config_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub executor_policy: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HarnessEvalRecord {
    pub schema_version: u32,
    pub fixture_version: u32,
    pub configuration: HarnessEvalConfiguration,
    pub protocol_pass: bool,
    pub protocol_failures: Vec<String>,
    pub result: ControlFixtureResult,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlFixtureTurn {
    pub content: String,
    #[serde(default)]
    pub truncated: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ContextEvalMetrics {
    #[serde(default)]
    pub invocations_observed: usize,
    #[serde(default)]
    pub context_capacity: usize,
    #[serde(default)]
    pub reserved_generation_tokens_high_water: usize,
    #[serde(default)]
    pub safety_margin_tokens_high_water: usize,
    #[serde(default)]
    pub usable_prompt_capacity_low_water: usize,
    #[serde(default)]
    pub preflight_prompt_tokens_high_water: usize,
    #[serde(default)]
    pub prompt_tokens_high_water: usize,
    #[serde(default)]
    pub prompt_utilization_bps_high_water: u32,
    #[serde(default)]
    pub message_chars_high_water: usize,
    #[serde(default)]
    pub tool_count_high_water: usize,
    #[serde(default)]
    pub tool_schema_chars_high_water: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_schema_tokens_high_water: Option<usize>,
    #[serde(default)]
    pub thinking_enabled_invocations: usize,
    #[serde(default)]
    pub thinking_disabled_invocations: usize,
    #[serde(default)]
    pub thinking_off_truncation_retries: usize,
    #[serde(default)]
    pub compact_mutation_truncation_retries: usize,
    #[serde(default)]
    pub larger_cap_truncation_retries: usize,
    #[serde(default)]
    pub compacted_messages: usize,
    #[serde(default)]
    pub omitted_tool_result_chars: usize,
    #[serde(default)]
    pub read_cache_hits: usize,
    #[serde(default)]
    pub closure_checkpoints: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HarnessEvalToolCall {
    pub tool: String,
    pub arguments_sha256: String,
    pub arguments_preview: String,
    pub arguments_truncated: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ControlFixtureResult {
    pub id: String,
    #[serde(default)]
    pub strict_goal: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_assertion_passed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_stage: Option<crate::goal::GoalStage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_outcome: Option<crate::goal::GoalOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_completion_basis: Option<crate::goal::GoalCompletionBasis>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_checkpoint_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_plan_sha256: Option<String>,
    #[serde(default)]
    pub goal_completed_milestones: usize,
    #[serde(default)]
    pub goal_total_milestones: usize,
    #[serde(default)]
    pub goal_workflows: usize,
    #[serde(default)]
    pub goal_model_invocations: usize,
    #[serde(default)]
    pub goal_generated_tokens: usize,
    #[serde(default)]
    pub strict_workflow: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow_assertion_passed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow_outcome: Option<crate::workflow::WorkflowOutcome>,
    #[serde(default)]
    pub workflow_stage_sequence: Vec<crate::workflow::WorkflowStage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow_plan_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow_plan_review_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow_code_review_sha256: Option<String>,
    #[serde(default)]
    pub workflow_plan_cycles: usize,
    #[serde(default)]
    pub workflow_repair_cycles: usize,
    #[serde(default)]
    pub workflow_advisory_calls: usize,
    #[serde(default)]
    pub workflow_rejected_actions: usize,
    #[serde(default)]
    pub workflow_evidence_invalidations: usize,
    pub reached_final: bool,
    #[serde(default)]
    pub contract_status: ContractStatus,
    #[serde(default)]
    pub verified_completed: bool,
    pub termination_reason: String,
    pub valid_actions: usize,
    pub llm_invocations: usize,
    pub tool_calls: usize,
    pub corrections: usize,
    pub gate_corrections: usize,
    pub errors: usize,
    pub blocked_tool_loops: usize,
    pub final_events: usize,
    pub executed_checks: usize,
    #[serde(default)]
    pub model_run_check_calls: usize,
    #[serde(default)]
    pub reused_checks: usize,
    #[serde(default)]
    pub failed_checks: usize,
    #[serde(default)]
    pub skipped_checks: usize,
    #[serde(default)]
    pub selected_checks: Vec<String>,
    #[serde(default)]
    pub affected_components: Vec<String>,
    #[serde(default)]
    pub executor_starts: Vec<String>,
    #[serde(default)]
    pub avoided_executor_starts: Vec<String>,
    #[serde(default)]
    pub team_messages: usize,
    #[serde(default)]
    pub repair_turns: usize,
    #[serde(default)]
    pub no_change: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff_outcome: Option<crate::events::HandoffOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_disposition: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_oid: Option<String>,
    #[serde(default)]
    pub output_fingerprints: Vec<String>,
    pub false_completion: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub named_check_compliance: Option<bool>,
    #[serde(default)]
    pub recovery_loop: bool,
    #[serde(default)]
    pub latency_ms: u64,
    #[serde(default)]
    pub prompt_tokens: usize,
    #[serde(default)]
    pub generated_tokens: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_trace: Vec<HarnessEvalToolCall>,
    #[serde(default)]
    pub context: ContextEvalMetrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_kwh: Option<f64>,
    pub remaining_completions: usize,
    pub observed_paths: BTreeMap<String, bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_quality: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_diagnostic: Option<String>,
}

pub fn control_fixture_corpus() -> Result<ControlFixtureCorpus> {
    parse_control_fixture_corpus(CONTROL_FIXTURES)
}

fn parse_control_fixture_corpus(contents: &str) -> Result<ControlFixtureCorpus> {
    let corpus: ControlFixtureCorpus = serde_json::from_str(contents)
        .context("failed to parse built-in harness control fixtures")?;
    if corpus.version != 3 {
        bail!(
            "unsupported harness control fixture version {}; expected 3",
            corpus.version
        );
    }
    if corpus.fixtures.is_empty() {
        bail!("harness control fixture corpus must not be empty");
    }
    let mut ids = std::collections::HashSet::new();
    for fixture in &corpus.fixtures {
        if fixture.id.trim().is_empty() {
            bail!("harness control fixture id must not be empty");
        }
        if !ids.insert(fixture.id.as_str()) {
            bail!("duplicate harness control fixture id '{}'", fixture.id);
        }
        if fixture.hypothesis.trim().is_empty() {
            bail!("harness control fixture '{}' has no hypothesis", fixture.id);
        }
        if fixture.max_steps == 0 || fixture.turns.is_empty() {
            bail!(
                "harness control fixture '{}' needs positive steps and at least one turn",
                fixture.id
            );
        }
    }
    if corpus.workflow_fixtures.len() != 25 {
        bail!(
            "harness workflow fixture corpus must contain the 25 required assertions; found {}",
            corpus.workflow_fixtures.len()
        );
    }
    for fixture in &corpus.workflow_fixtures {
        if fixture.id.trim().is_empty() || fixture.hypothesis.trim().is_empty() {
            bail!("harness workflow fixture id and hypothesis must not be empty");
        }
        if !ids.insert(fixture.id.as_str()) {
            bail!("duplicate harness control fixture id '{}'", fixture.id);
        }
    }
    if corpus.goal_fixtures.len() != 7 {
        bail!(
            "harness Goal fixture corpus must contain the 7 required assertions; found {}",
            corpus.goal_fixtures.len()
        );
    }
    for fixture in &corpus.goal_fixtures {
        if fixture.id.trim().is_empty() || fixture.hypothesis.trim().is_empty() {
            bail!("harness Goal fixture id and hypothesis must not be empty");
        }
        if !ids.insert(fixture.id.as_str()) {
            bail!("duplicate harness control fixture id '{}'", fixture.id);
        }
    }
    if corpus.small_model_fixtures.is_empty() {
        bail!("harness small-model fixture group must not be empty");
    }
    let mut grouped = HashSet::new();
    for id in &corpus.small_model_fixtures {
        if !grouped.insert(id.as_str()) {
            bail!("duplicate harness small-model fixture id '{id}'");
        }
        if !corpus.fixtures.iter().any(|fixture| fixture.id == *id) {
            bail!("unknown harness small-model fixture id '{id}'");
        }
    }
    Ok(corpus)
}

pub fn run_control_fixture(fixture: &ControlFixture) -> Result<ControlFixtureResult> {
    let scratch = tempfile::Builder::new()
        .prefix("pb-control-fixture-")
        .tempdir()
        .context("failed to create harness control fixture scratch directory")?;
    initialize_fixture_workspace(scratch.path(), &fixture.initial_files)?;
    let (contract, workspace_graph, repository_context) = fixture_runtime(fixture, scratch.path())?;

    let args = AgentRequest {
        task: fixture.hypothesis.clone(),
        turn_id: format!("control-fixture-turn-{}", fixture.id),
        intent: Some(crate::workflow::TurnIntent::Deliver),
        workflow_policy: None,
        workflow_stage: None,
        workflow_expected_content_fingerprint: None,
        workflow_action_first_turn: false,
        workflow_creation_path_order: Vec::new(),
        workflow_work_units: None,
        workflow_stage_evidence: None,
        workflow_checkpoint: None,
        conversation_handoff: None,
        legacy_prompt_owned_delivery: true,
        model: "scripted-control-fixture".to_string(),
        model_dir: None,
        workdir: Some(scratch.path().to_path_buf()),
        branch: None,
        max_steps: fixture.max_steps,
        max_tokens: 256,
        turn_max_tokens_cap: Some(256),
        tool_allowlist: Some(fixture.tool_allowlist.clone()),
        observation_rendering: crate::workflow::ObservationRendering::Native,
        controller_delete_elision: false,
        accept_existing_workspace_changes: false,
        ctx_size: SCRIPTED_EVAL_CONTEXT_SIZE,
        threads: None,
        threads_batch: None,
        gpu_layers: 0,
        temperature: 0.0,
        profile: fixture.profile,
        infer_profile: false,
        sub_agent_depth: 0,
        repository_less: false,
        top_k: 1,
        seed: 0,
        environment: None,
        environment_evidence_context: None,
        workspace_graph: Some(workspace_graph),
        repository_context: Some(repository_context),
        prior_check_evidence: crate::checks::CheckEvidenceLedger::default(),
        session_id: format!("control-fixture-{}", fixture.id),
        attachments: Vec::new(),
        goal_context: None,
        contract,
    };
    let completions = fixture
        .turns
        .iter()
        .map(|turn| ScriptedCompletion {
            content: turn.content.clone(),
            truncated: turn.truncated,
        })
        .collect();
    let mut events = Vec::new();
    let outcome = run_scripted_agent_steps(&args, completions, scratch.path(), &mut |event| {
        events.push(event)
    })?;

    summarize_fixture(fixture, scratch.path(), outcome.into(), &events, false)
}

struct WorkflowFixtureState {
    scratch: tempfile::TempDir,
    graph: crate::workspace::WorkspaceGraph,
    run: crate::workflow::WorkflowRun,
    plan: crate::workflow::ArtifactEnvelope<crate::workflow::PlanArtifact>,
}

fn workflow_fixture_request(root: &Path) -> Result<AgentRequest> {
    let repository = crate::workspace::RepositoryContext::capture(root, root)?;
    Ok(AgentRequest {
        task: "exercise strict workflow control".to_string(),
        turn_id: "turn-workflow-fixture".to_string(),
        intent: Some(crate::workflow::TurnIntent::Deliver),
        workflow_policy: Some(crate::workflow::WorkflowConfigDocument::default().compile()?),
        workflow_stage: None,
        workflow_expected_content_fingerprint: None,
        workflow_action_first_turn: false,
        workflow_creation_path_order: Vec::new(),
        workflow_work_units: None,
        workflow_stage_evidence: None,
        workflow_checkpoint: None,
        conversation_handoff: None,
        legacy_prompt_owned_delivery: false,
        model: "scripted-workflow-fixture".to_string(),
        model_dir: None,
        workdir: Some(root.to_path_buf()),
        branch: None,
        max_steps: 3,
        max_tokens: 256,
        turn_max_tokens_cap: Some(256),
        tool_allowlist: Some(vec![
            "read_file".to_string(),
            "write_file".to_string(),
            "run_command".to_string(),
            "sub_agent".to_string(),
        ]),
        observation_rendering: crate::workflow::ObservationRendering::Native,
        controller_delete_elision: false,
        accept_existing_workspace_changes: false,
        ctx_size: SCRIPTED_EVAL_CONTEXT_SIZE,
        threads: None,
        threads_batch: None,
        gpu_layers: 0,
        temperature: 0.0,
        profile: AgentProfile::Build,
        infer_profile: false,
        sub_agent_depth: 0,
        repository_less: false,
        top_k: 1,
        seed: 0,
        environment: None,
        environment_evidence_context: None,
        workspace_graph: Some(crate::workspace::WorkspaceGraph::legacy(&[])),
        repository_context: Some(repository),
        prior_check_evidence: crate::checks::CheckEvidenceLedger::default(),
        session_id: "workflow-fixture".to_string(),
        attachments: Vec::new(),
        goal_context: None,
        contract: None,
    })
}

fn workflow_fixture_state() -> Result<WorkflowFixtureState> {
    let scratch = tempfile::Builder::new()
        .prefix("pb-workflow-fixture-")
        .tempdir()?;
    initialize_fixture_workspace(
        scratch.path(),
        &BTreeMap::from([("existing.txt".to_string(), "baseline\n".to_string())]),
    )?;
    let graph = crate::workspace::WorkspaceGraph::legacy(&[]);
    let repository = crate::workspace::RepositoryContext::capture(scratch.path(), scratch.path())?;
    let mut run = crate::workflow::WorkflowRun::start(
        "workflow-control-fixture",
        "turn-workflow-fixture",
        "deliver the fixture change",
        crate::workflow::WorkflowConfigDocument::default().compile()?,
        repository,
    )?;
    let plan = workflow_fixture_plan(Vec::new())?;
    plan.artifact
        .validate(&graph, &run.repository.task_baseline.content)?;
    run.apply(crate::workflow::WorkflowEvent::PlanSubmitted { plan: plan.clone() })?;
    run.apply(crate::workflow::WorkflowEvent::PlanReviewSubmitted {
        review: workflow_fixture_plan_review(&plan, crate::workflow::ReviewVerdict::Pass)?,
    })?;
    Ok(WorkflowFixtureState {
        scratch,
        graph,
        run,
        plan,
    })
}

fn workflow_fixture_plan(
    resolved_challenge_ids: Vec<String>,
) -> Result<crate::workflow::ArtifactEnvelope<crate::workflow::PlanArtifact>> {
    crate::workflow::ArtifactEnvelope::new(
        "plan-workflow-fixture",
        crate::workflow::PlanArtifact {
            summary: "Modify the fixture and verify the result".to_string(),
            requirements: vec![crate::workflow::PlanRequirement {
                id: "req-fixture".to_string(),
                description: "Deliver the requested fixture change".to_string(),
                source: "current user task".to_string(),
            }],
            steps: vec![crate::workflow::PlanStep {
                id: "step-fixture".to_string(),
                requirement_ids: vec!["req-fixture".to_string()],
                component_ids: Vec::new(),
                paths: vec![crate::workflow::PlanPath {
                    path: "existing.txt".to_string(),
                    change: crate::workflow::PlannedChange::Modify,
                }],
                description: "Update the fixture".to_string(),
            }],
            acceptance: vec![crate::workflow::PlanAcceptance {
                id: "accept-fixture".to_string(),
                requirement_ids: vec!["req-fixture".to_string()],
                check_ids: Vec::new(),
                description: "The fixture content is updated".to_string(),
            }],
            risks: Vec::new(),
            assumptions: Vec::new(),
            open_questions: Vec::new(),
            resolved_challenge_ids,
        },
    )
}

fn workflow_fixture_plan_review(
    plan: &crate::workflow::ArtifactEnvelope<crate::workflow::PlanArtifact>,
    verdict: crate::workflow::ReviewVerdict,
) -> Result<crate::workflow::ArtifactEnvelope<crate::workflow::PlanReviewArtifact>> {
    let blocking = verdict == crate::workflow::ReviewVerdict::Revise;
    crate::workflow::ArtifactEnvelope::new(
        if blocking {
            "plan-review-revise"
        } else {
            "plan-review-pass"
        },
        crate::workflow::PlanReviewArtifact {
            plan_id: plan.id.clone(),
            plan_sha256: plan.sha256.clone(),
            assessments: crate::workflow::REQUIRED_PLAN_ASSESSMENTS
                .into_iter()
                .map(|kind| crate::workflow::PlanAssessment {
                    kind,
                    status: if blocking {
                        crate::workflow::AssessmentStatus::Concern
                    } else {
                        crate::workflow::AssessmentStatus::Pass
                    },
                    evidence: Vec::new(),
                    explanation: "reviewed in a fresh context".to_string(),
                })
                .collect(),
            challenges: blocking
                .then(|| crate::workflow::ReviewChallenge {
                    id: "challenge-fixture".to_string(),
                    severity: crate::workflow::ReviewSeverity::P1,
                    requirement_ids: vec!["req-fixture".to_string()],
                    description: "The plan must address a blocking risk".to_string(),
                    evidence: Vec::new(),
                })
                .into_iter()
                .collect(),
            verdict,
        },
    )
}

fn workflow_fixture_implementation(
    plan: &crate::workflow::ArtifactEnvelope<crate::workflow::PlanArtifact>,
    fingerprint: String,
    no_change: bool,
) -> Result<crate::workflow::ArtifactEnvelope<crate::workflow::ImplementationArtifact>> {
    crate::workflow::ArtifactEnvelope::new(
        if no_change {
            "implementation-no-change"
        } else {
            "implementation-change"
        },
        crate::workflow::ImplementationArtifact {
            plan_id: plan.id.clone(),
            plan_sha256: plan.sha256.clone(),
            content_fingerprint: fingerprint,
            steps: vec![crate::workflow::ImplementationStep {
                step_id: "step-fixture".to_string(),
                status: if no_change {
                    crate::workflow::ImplementationStepStatus::NoChange
                } else {
                    crate::workflow::ImplementationStepStatus::Completed
                },
                touched_paths: if no_change {
                    Vec::new()
                } else {
                    vec!["existing.txt".to_string()]
                },
                summary: "accounted for the fixture step".to_string(),
            }],
            summary: "implemented the fixture plan".to_string(),
            no_change,
            semantic_commit_subject: "feat: deliver workflow fixture".to_string(),
        },
    )
}

fn workflow_fixture_code_review(
    fingerprint: String,
    verdict: crate::workflow::ReviewVerdict,
) -> Result<crate::workflow::ArtifactEnvelope<crate::workflow::CodeReviewArtifact>> {
    let blocking = verdict == crate::workflow::ReviewVerdict::Revise;
    crate::workflow::ArtifactEnvelope::new(
        if blocking {
            "code-review-revise"
        } else {
            "code-review-pass"
        },
        crate::workflow::CodeReviewArtifact {
            content_fingerprint: fingerprint,
            assessments: crate::workflow::REQUIRED_CODE_ASSESSMENTS
                .into_iter()
                .map(|kind| crate::workflow::CodeAssessment {
                    kind,
                    status: if blocking {
                        crate::workflow::AssessmentStatus::Concern
                    } else {
                        crate::workflow::AssessmentStatus::Pass
                    },
                    evidence: Vec::new(),
                    explanation: "reviewed exact checked content".to_string(),
                })
                .collect(),
            findings: blocking
                .then(|| crate::workflow::CodeFinding {
                    id: "finding-fixture".to_string(),
                    severity: crate::workflow::ReviewSeverity::P1,
                    path: Some("existing.txt".to_string()),
                    line: Some(1),
                    requirement_ids: vec!["req-fixture".to_string()],
                    plan_step_ids: vec!["step-fixture".to_string()],
                    evidence: Vec::new(),
                    explanation: "the implementation requires repair".to_string(),
                })
                .into_iter()
                .collect(),
            verdict,
        },
    )
}

fn workflow_fixture_at_code_review() -> Result<WorkflowFixtureState> {
    let mut state = workflow_fixture_state()?;
    std::fs::write(state.scratch.path().join("existing.txt"), "delivered\n")?;
    let snapshot = crate::workspace::ContentSnapshot::capture(state.scratch.path())?;
    state
        .run
        .apply(crate::workflow::WorkflowEvent::ImplementationSubmitted {
            implementation: workflow_fixture_implementation(
                &state.plan,
                snapshot.fingerprint.clone(),
                false,
            )?,
        })?;
    state
        .run
        .apply(crate::workflow::WorkflowEvent::ChecksPassed {
            content_fingerprint: snapshot.fingerprint,
            selected_checks: Vec::new(),
            evidence: crate::checks::CheckEvidenceLedger::default(),
        })?;
    Ok(state)
}

#[derive(Default)]
struct WorkflowAssertionObservation {
    run: Option<crate::workflow::WorkflowRun>,
    stage_sequence: Vec<crate::workflow::WorkflowStage>,
    rejected_actions: usize,
    evidence_invalidations: usize,
    llm_invocations: usize,
    tool_calls: usize,
}

impl WorkflowAssertionObservation {
    fn from_run(run: crate::workflow::WorkflowRun) -> Self {
        Self {
            run: Some(run),
            ..Self::default()
        }
    }
}

fn require_workflow_fixture(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        bail!("{message}")
    }
}

fn plan_submission_completion(
    plan: &crate::workflow::ArtifactEnvelope<crate::workflow::PlanArtifact>,
) -> ScriptedCompletion {
    ScriptedCompletion {
        content: serde_json::json!({
            "type": "tool_call",
            "tool": "submit_plan",
            "arguments": {"id": plan.id, "plan": plan.artifact}
        })
        .to_string(),
        truncated: false,
    }
}

fn workflow_tool_completion(tool: &str, arguments: serde_json::Value) -> ScriptedCompletion {
    ScriptedCompletion {
        content: serde_json::json!({
            "type": "tool_call",
            "tool": tool,
            "arguments": arguments,
        })
        .to_string(),
        truncated: false,
    }
}

fn execute_workflow_assertion(
    assertion: WorkflowControlAssertion,
) -> Result<WorkflowAssertionObservation> {
    match assertion {
        WorkflowControlAssertion::DiscussionNoBranch => {
            let state = workflow_fixture_state()?;
            let mut request = workflow_fixture_request(state.scratch.path())?;
            request.intent = Some(crate::workflow::TurnIntent::Discuss);
            let before = git_output(state.scratch.path(), &["branch", "--show-current"])?;
            let outcome = run_scripted_agent_steps(
                &request,
                vec![ScriptedCompletion {
                    content:
                        r#"{"type":"final","content":"We can discuss this without building."}"#
                            .to_string(),
                    truncated: false,
                }],
                state.scratch.path(),
                &mut |_| {},
            )?;
            let after = git_output(state.scratch.path(), &["branch", "--show-current"])?;
            require_workflow_fixture(outcome.reached_final, "discussion did not finish normally")?;
            require_workflow_fixture(before == after, "discussion changed the task branch")?;
            Ok(WorkflowAssertionObservation {
                llm_invocations: outcome.llm_invocations,
                tool_calls: outcome.tool_calls,
                ..WorkflowAssertionObservation::default()
            })
        }
        WorkflowControlAssertion::DiscussionNoMutation => {
            let state = workflow_fixture_state()?;
            let mut request = workflow_fixture_request(state.scratch.path())?;
            request.intent = Some(crate::workflow::TurnIntent::Discuss);
            request.max_steps = 2;
            let outcome = run_scripted_agent_steps(
                &request,
                vec![
                    ScriptedCompletion {
                        content: r#"{"type":"tool_call","tool":"write_file","arguments":{"path":"forbidden.txt","content":"no"}}"#.to_string(),
                        truncated: false,
                    },
                    ScriptedCompletion {
                        content: r#"{"type":"final","content":"No repository mutation was performed."}"#.to_string(),
                        truncated: false,
                    },
                ],
                state.scratch.path(),
                &mut |_| {},
            )?;
            require_workflow_fixture(
                !state.scratch.path().join("forbidden.txt").exists(),
                "discussion mutated the repository",
            )?;
            Ok(WorkflowAssertionObservation {
                llm_invocations: outcome.llm_invocations,
                tool_calls: outcome.tool_calls,
                rejected_actions: 1,
                ..WorkflowAssertionObservation::default()
            })
        }
        WorkflowControlAssertion::ExplicitDeliveryStartsPlanning => {
            let state = workflow_fixture_state()?;
            let run = crate::workflow::WorkflowRun::start(
                "workflow-starts-planning",
                "turn-starts-planning",
                "deliver",
                state.run.policy.clone(),
                state.run.repository.clone(),
            )?;
            require_workflow_fixture(
                run.stage == crate::workflow::WorkflowStage::Planning,
                "explicit delivery did not start at planning",
            )?;
            Ok(WorkflowAssertionObservation::from_run(run))
        }
        WorkflowControlAssertion::PlanningRequiresSubmission => {
            let state = workflow_fixture_state()?;
            let request = workflow_fixture_request(state.scratch.path())?;
            let contract = crate::workflow::StageContract::strict(
                crate::workflow::WorkflowStage::Planning,
                request.workflow_policy.as_ref().unwrap().limits,
                request.max_tokens,
            )?;
            let mut events = Vec::new();
            let outcome = run_scripted_stage(
                &request,
                &contract,
                StageContext {
                    system_prompt: "plan".to_string(),
                    user_prompt: "plan".to_string(),
                    expected_content_fingerprint: None,
                    action_first_turn: false,
                    creation_path_order: Vec::new(),
                    work_units: None,
                },
                vec![
                    ScriptedCompletion {
                        content: r#"{"type":"final","content":"prose plan"}"#.to_string(),
                        truncated: false,
                    },
                    plan_submission_completion(&state.plan),
                ],
                state.scratch.path(),
                &mut |event| events.push(event),
            )?;
            require_workflow_fixture(
                matches!(
                    outcome.stage.submission,
                    Some(crate::workflow::StageSubmission::Plan { .. })
                ),
                "planning advanced without submit_plan",
            )?;
            require_workflow_fixture(
                events.iter().any(|event| {
                    matches!(
                        event,
                        AgentEvent::Correction { summary, .. }
                            if summary == "Workflow stage submission required"
                    )
                }),
                "prose final was not rejected",
            )?;
            Ok(WorkflowAssertionObservation {
                llm_invocations: outcome.stage.usage.model_invocations,
                rejected_actions: 1,
                ..WorkflowAssertionObservation::default()
            })
        }
        WorkflowControlAssertion::PlanningAuthorityIsReadOnly => {
            let state = workflow_fixture_state()?;
            let request = workflow_fixture_request(state.scratch.path())?;
            let capabilities = crate::workflow::StageCapabilities::for_stage(
                crate::workflow::WorkflowStage::Planning,
            );
            require_workflow_fixture(
                !capabilities.allows_tool("write_file")
                    && !capabilities.allows_tool("run_command")
                    && capabilities.allows_tool("sub_agent"),
                "planning capability surface grants mutation or shell authority",
            )?;
            let contract = crate::workflow::StageContract::strict(
                crate::workflow::WorkflowStage::Planning,
                request.workflow_policy.as_ref().unwrap().limits,
                request.max_tokens,
            )?;
            let outcome = run_scripted_stage(
                &request,
                &contract,
                StageContext {
                    system_prompt: "plan".to_string(),
                    user_prompt: "plan".to_string(),
                    expected_content_fingerprint: None,
                    action_first_turn: false,
                    creation_path_order: Vec::new(),
                    work_units: None,
                },
                vec![
                    ScriptedCompletion {
                        content: r#"{"type":"tool_call","tool":"run_command","arguments":{"cmd":"touch forbidden.txt"}}"#.to_string(),
                        truncated: false,
                    },
                    plan_submission_completion(&state.plan),
                ],
                state.scratch.path(),
                &mut |_| {},
            )?;
            require_workflow_fixture(
                !state.scratch.path().join("forbidden.txt").exists(),
                "hidden planning shell action executed",
            )?;
            Ok(WorkflowAssertionObservation {
                llm_invocations: outcome.stage.usage.model_invocations,
                rejected_actions: 1,
                ..WorkflowAssertionObservation::default()
            })
        }
        WorkflowControlAssertion::PlanStructureValidated => {
            let state = workflow_fixture_state()?;
            let invalid = crate::workflow::PlanArtifact {
                summary: "incomplete".to_string(),
                requirements: Vec::new(),
                steps: Vec::new(),
                acceptance: Vec::new(),
                risks: Vec::new(),
                assumptions: Vec::new(),
                open_questions: Vec::new(),
                resolved_challenge_ids: Vec::new(),
            };
            require_workflow_fixture(
                invalid
                    .validate(&state.graph, state.run.planning_content())
                    .is_err(),
                "malformed plan passed structural validation",
            )?;
            Ok(WorkflowAssertionObservation::default())
        }
        WorkflowControlAssertion::PlanReviewHashBound => {
            let state = workflow_fixture_state()?;
            let mut review =
                workflow_fixture_plan_review(&state.plan, crate::workflow::ReviewVerdict::Pass)?;
            review.artifact.plan_sha256 = "wrong".to_string();
            require_workflow_fixture(
                review.artifact.validate(&state.plan).is_err(),
                "plan review accepted the wrong plan hash",
            )?;
            Ok(WorkflowAssertionObservation::default())
        }
        WorkflowControlAssertion::PlanReviewEvidenceRequired => {
            let state = workflow_fixture_state()?;
            let mut review =
                workflow_fixture_plan_review(&state.plan, crate::workflow::ReviewVerdict::Pass)?;
            review.artifact.assessments[0].evidence = vec![crate::workflow::EvidenceReference {
                path: Some("existing.txt".to_string()),
                line: Some(1),
                check_id: None,
                description: "fixture evidence".to_string(),
            }];
            require_workflow_fixture(
                review
                    .artifact
                    .validate_observed_evidence(&state.graph, &HashSet::new())
                    .is_err(),
                "plan review accepted unread path evidence",
            )?;
            Ok(WorkflowAssertionObservation::default())
        }
        WorkflowControlAssertion::PlanChallengeForcesRevision => {
            let state = workflow_fixture_state()?;
            let mut run = crate::workflow::WorkflowRun::start(
                "workflow-plan-revision",
                "turn-plan-revision",
                "deliver",
                state.run.policy.clone(),
                state.run.repository.clone(),
            )?;
            run.apply(crate::workflow::WorkflowEvent::PlanSubmitted {
                plan: state.plan.clone(),
            })?;
            run.apply(crate::workflow::WorkflowEvent::PlanReviewSubmitted {
                review: workflow_fixture_plan_review(
                    &state.plan,
                    crate::workflow::ReviewVerdict::Revise,
                )?,
            })?;
            require_workflow_fixture(
                run.stage == crate::workflow::WorkflowStage::PlanRevision
                    && run.counters.plan_cycles == 1,
                "blocking plan challenge did not force bounded revision",
            )?;
            Ok(WorkflowAssertionObservation::from_run(run))
        }
        WorkflowControlAssertion::ImplementationRequiresAcceptedPlan => {
            let state = workflow_fixture_state()?;
            let mut run = crate::workflow::WorkflowRun::start(
                "workflow-no-plan-skip",
                "turn-no-plan-skip",
                "deliver",
                state.run.policy.clone(),
                state.run.repository.clone(),
            )?;
            let implementation = workflow_fixture_implementation(
                &state.plan,
                run.repository.task_baseline.content.fingerprint.clone(),
                true,
            )?;
            require_workflow_fixture(
                run.apply(crate::workflow::WorkflowEvent::ImplementationSubmitted {
                    implementation,
                })
                .is_err(),
                "implementation started before plan acceptance",
            )?;
            Ok(WorkflowAssertionObservation::from_run(run))
        }
        WorkflowControlAssertion::ImplementationCanReplan => {
            let mut state = workflow_fixture_state()?;
            std::fs::write(state.scratch.path().join("existing.txt"), "partial\n")?;
            let snapshot = crate::workspace::ContentSnapshot::capture(state.scratch.path())?;
            state
                .run
                .apply(crate::workflow::WorkflowEvent::ReplanRequested {
                    reason: "material architecture discovery".to_string(),
                    planning_snapshot: Some(snapshot.clone()),
                })?;
            require_workflow_fixture(
                state.run.stage == crate::workflow::WorkflowStage::Planning
                    && state.run.plan.is_none()
                    && state.run.planning_content() == &snapshot,
                "request_replan did not preserve and bind the current content snapshot",
            )?;
            Ok(WorkflowAssertionObservation::from_run(state.run))
        }
        WorkflowControlAssertion::RunCommandCannotBypassGates => {
            let state = workflow_fixture_state()?;
            let request = workflow_fixture_request(state.scratch.path())?;
            let contract = crate::workflow::StageContract::strict(
                crate::workflow::WorkflowStage::Implementing,
                request.workflow_policy.as_ref().unwrap().limits,
                request.max_tokens,
            )?;
            let outcome = run_scripted_stage(
                &request,
                &contract,
                StageContext {
                    system_prompt: "implement".to_string(),
                    user_prompt: "implement".to_string(),
                    expected_content_fingerprint: None,
                    action_first_turn: false,
                    creation_path_order: Vec::new(),
                    work_units: None,
                },
                vec![
                    ScriptedCompletion {
                        content: r#"{"type":"tool_call","tool":"run_command","arguments":{"cmd":"touch shell.txt"}}"#.to_string(),
                        truncated: false,
                    },
                    ScriptedCompletion {
                        content: r#"{"type":"final","content":"done"}"#.to_string(),
                        truncated: false,
                    },
                ],
                state.scratch.path(),
                &mut |_| {},
            )?;
            require_workflow_fixture(
                state.scratch.path().join("shell.txt").exists()
                    && outcome.stage.submission.is_none()
                    && outcome.stage.termination_reason != crate::events::TerminationReason::Final,
                "run_command either failed as an escape hatch or bypassed structured submission",
            )?;
            Ok(WorkflowAssertionObservation {
                llm_invocations: outcome.stage.usage.model_invocations,
                tool_calls: 1,
                rejected_actions: 1,
                ..WorkflowAssertionObservation::default()
            })
        }
        _ => execute_workflow_assertion_tail(assertion),
    }
}

fn workflow_fixture_at_checking() -> Result<WorkflowFixtureState> {
    let mut state = workflow_fixture_state()?;
    std::fs::write(state.scratch.path().join("existing.txt"), "delivered\n")?;
    let snapshot = crate::workspace::ContentSnapshot::capture(state.scratch.path())?;
    state
        .run
        .apply(crate::workflow::WorkflowEvent::ImplementationSubmitted {
            implementation: workflow_fixture_implementation(
                &state.plan,
                snapshot.fingerprint,
                false,
            )?,
        })?;
    Ok(state)
}

fn execute_workflow_assertion_tail(
    assertion: WorkflowControlAssertion,
) -> Result<WorkflowAssertionObservation> {
    match assertion {
        WorkflowControlAssertion::CheckFailureForcesRepair => {
            let mut state = workflow_fixture_at_checking()?;
            let fingerprint = state.run.content_fingerprint.clone().unwrap();
            state
                .run
                .apply(crate::workflow::WorkflowEvent::ChecksFailed {
                    content_fingerprint: fingerprint,
                    selected_checks: vec!["fixture-check".to_string()],
                    evidence: crate::checks::CheckEvidenceLedger::default(),
                    failed_check_ids: vec!["fixture-check".to_string()],
                })?;
            require_workflow_fixture(
                state.run.stage == crate::workflow::WorkflowStage::Repairing
                    && state.run.counters.repair_cycles == 1,
                "failed required check did not force repair",
            )?;
            Ok(WorkflowAssertionObservation::from_run(state.run))
        }
        WorkflowControlAssertion::PostCheckMutationInvalidatesEvidence => {
            let mut state = workflow_fixture_at_code_review()?;
            state
                .run
                .apply(crate::workflow::WorkflowEvent::MutationObserved {
                    content_fingerprint: "post-check-mutation".to_string(),
                })?;
            require_workflow_fixture(
                state.run.stage == crate::workflow::WorkflowStage::Checking
                    && state.run.checks == crate::checks::CheckEvidenceLedger::default()
                    && state.run.code_review.is_none(),
                "post-check mutation retained stale evidence",
            )?;
            Ok(WorkflowAssertionObservation {
                run: Some(state.run),
                evidence_invalidations: 1,
                ..WorkflowAssertionObservation::default()
            })
        }
        WorkflowControlAssertion::CodeReviewFingerprintBound => {
            let mut state = workflow_fixture_at_code_review()?;
            let review = workflow_fixture_code_review(
                "wrong-fingerprint".to_string(),
                crate::workflow::ReviewVerdict::Pass,
            )?;
            require_workflow_fixture(
                state
                    .run
                    .apply(crate::workflow::WorkflowEvent::CodeReviewSubmitted { review })
                    .is_err(),
                "code review accepted the wrong content fingerprint",
            )?;
            Ok(WorkflowAssertionObservation::from_run(state.run))
        }
        WorkflowControlAssertion::CodeReviewPathEvidenceRequired => {
            let state = workflow_fixture_at_code_review()?;
            let review = workflow_fixture_code_review(
                state.run.content_fingerprint.clone().unwrap(),
                crate::workflow::ReviewVerdict::Pass,
            )?;
            require_workflow_fixture(
                crate::agent_core::validate_code_review_submission(
                    &review,
                    &state.run,
                    &state.graph,
                    state.scratch.path(),
                    &HashSet::new(),
                )
                .is_err(),
                "code review passed without reading every changed path",
            )?;
            Ok(WorkflowAssertionObservation::from_run(state.run))
        }
        WorkflowControlAssertion::CodeFindingForcesRepair => {
            let mut state = workflow_fixture_at_code_review()?;
            let review = workflow_fixture_code_review(
                state.run.content_fingerprint.clone().unwrap(),
                crate::workflow::ReviewVerdict::Revise,
            )?;
            state
                .run
                .apply(crate::workflow::WorkflowEvent::CodeReviewSubmitted { review })?;
            require_workflow_fixture(
                state.run.stage == crate::workflow::WorkflowStage::Repairing
                    && state.run.counters.repair_cycles == 1,
                "blocking code finding did not force bounded repair",
            )?;
            Ok(WorkflowAssertionObservation::from_run(state.run))
        }
        WorkflowControlAssertion::PostReviewMutationBlocksCommit => {
            let mut state = workflow_fixture_at_code_review()?;
            let review = workflow_fixture_code_review(
                state.run.content_fingerprint.clone().unwrap(),
                crate::workflow::ReviewVerdict::Pass,
            )?;
            state
                .run
                .apply(crate::workflow::WorkflowEvent::CodeReviewSubmitted { review })?;
            require_workflow_fixture(
                state.run.stage == crate::workflow::WorkflowStage::Committing,
                "passing review did not reach committing",
            )?;
            state
                .run
                .apply(crate::workflow::WorkflowEvent::MutationObserved {
                    content_fingerprint: "post-review-mutation".to_string(),
                })?;
            require_workflow_fixture(
                state.run.stage == crate::workflow::WorkflowStage::Checking
                    && state.run.code_review.is_none(),
                "post-review mutation did not return to checks and clear review",
            )?;
            Ok(WorkflowAssertionObservation {
                run: Some(state.run),
                evidence_invalidations: 1,
                ..WorkflowAssertionObservation::default()
            })
        }
        WorkflowControlAssertion::DelegationCannotEscalateAuthority => {
            let state = workflow_fixture_state()?;
            let mut request = workflow_fixture_request(state.scratch.path())?;
            request.intent = Some(crate::workflow::TurnIntent::Discuss);
            request.max_steps = 2;
            let mut events = Vec::new();
            let outcome = run_scripted_agent_steps(
                &request,
                vec![
                    ScriptedCompletion {
                        content: r#"{"type":"tool_call","tool":"sub_agent","arguments":{"profile":"build","task":"mutate the project","max_steps":1}}"#.to_string(),
                        truncated: false,
                    },
                    ScriptedCompletion {
                        content: r#"{"type":"final","content":"No mutating teammate was started."}"#.to_string(),
                        truncated: false,
                    },
                ],
                state.scratch.path(),
                &mut |event| events.push(event),
            )?;
            require_workflow_fixture(
                !events
                    .iter()
                    .any(|event| matches!(event, AgentEvent::SubAgentStarted { .. })),
                "read-only parent delegated mutating authority",
            )?;
            Ok(WorkflowAssertionObservation {
                llm_invocations: outcome.llm_invocations,
                tool_calls: outcome.tool_calls,
                rejected_actions: 1,
                ..WorkflowAssertionObservation::default()
            })
        }
        WorkflowControlAssertion::WorkflowBudgetsAreGlobal => {
            let state = workflow_fixture_state()?;
            let mut policy_document = crate::workflow::WorkflowConfigDocument::default();
            policy_document.limits.advisory_calls = 1;
            let mut run = crate::workflow::WorkflowRun::start(
                "workflow-global-budget",
                "turn-global-budget",
                "deliver",
                policy_document.compile()?,
                state.run.repository.clone(),
            )?;
            run.apply(crate::workflow::WorkflowEvent::UsageRecorded {
                usage: crate::workflow::WorkflowUsage {
                    advisory_calls: 1,
                    ..crate::workflow::WorkflowUsage::default()
                },
            })?;
            run.apply(crate::workflow::WorkflowEvent::UsageRecorded {
                usage: crate::workflow::WorkflowUsage {
                    advisory_calls: 1,
                    ..crate::workflow::WorkflowUsage::default()
                },
            })?;
            require_workflow_fixture(
                run.stage == crate::workflow::WorkflowStage::Failed
                    && run.outcome == Some(crate::workflow::WorkflowOutcome::InvocationLimit),
                "workflow-wide advisory budget could be multiplied",
            )?;
            Ok(WorkflowAssertionObservation::from_run(run))
        }
        WorkflowControlAssertion::ManagedCommitIsTaskOwned => {
            let state = workflow_fixture_state()?;
            std::fs::write(state.scratch.path().join("unrelated.txt"), "preserve\n")?;
            let repository = crate::workspace::RepositoryContext::capture(
                state.scratch.path(),
                state.scratch.path(),
            )?;
            std::fs::write(state.scratch.path().join("existing.txt"), "owned\n")?;
            let outcome = crate::handoff::managed_commit(
                &repository,
                "feat: commit reviewed fixture",
                None,
                &mut |_| {},
            )?;
            require_workflow_fixture(
                matches!(outcome, crate::handoff::ManagedCommitOutcome::Created(_)),
                "managed commit was not created",
            )?;
            let changed = git_output(
                state.scratch.path(),
                &["show", "--pretty=", "--name-only", "HEAD"],
            )?;
            require_workflow_fixture(
                changed.trim() == "existing.txt"
                    && state.scratch.path().join("unrelated.txt").exists(),
                "managed commit included or removed unrelated content",
            )?;
            Ok(WorkflowAssertionObservation::default())
        }
        WorkflowControlAssertion::NoChangeCreatesNoCommit => {
            let mut state = workflow_fixture_state()?;
            let head_before = git_output(state.scratch.path(), &["rev-parse", "HEAD"])?;
            let fingerprint = state
                .run
                .repository
                .task_baseline
                .content
                .fingerprint
                .clone();
            state
                .run
                .apply(crate::workflow::WorkflowEvent::ImplementationSubmitted {
                    implementation: workflow_fixture_implementation(
                        &state.plan,
                        fingerprint.clone(),
                        true,
                    )?,
                })?;
            state
                .run
                .apply(crate::workflow::WorkflowEvent::ChecksPassed {
                    content_fingerprint: fingerprint,
                    selected_checks: Vec::new(),
                    evidence: crate::checks::CheckEvidenceLedger::default(),
                })?;
            let head_after = git_output(state.scratch.path(), &["rev-parse", "HEAD"])?;
            require_workflow_fixture(
                state.run.outcome == Some(crate::workflow::WorkflowOutcome::NoChange)
                    && state.run.commit.is_none()
                    && head_before == head_after,
                "no-change delivery invented a commit or skipped its challenged plan",
            )?;
            Ok(WorkflowAssertionObservation::from_run(state.run))
        }
        WorkflowControlAssertion::ResumePreservesStageAndBudget => {
            let mut state = workflow_fixture_state()?;
            state
                .run
                .apply(crate::workflow::WorkflowEvent::UsageRecorded {
                    usage: crate::workflow::WorkflowUsage {
                        stage_steps: 2,
                        model_invocations: 2,
                        generated_tokens: 17,
                        advisory_calls: 1,
                    },
                })?;
            state.run.apply(crate::workflow::WorkflowEvent::Blocked {
                outcome: crate::workflow::WorkflowOutcome::ExecutorUnavailable,
                reason: "pause for recovery".to_string(),
            })?;
            let encoded = serde_json::to_vec(&crate::workflow::WorkflowCheckpoint::new(
                state.run.clone(),
            )?)?;
            let checkpoint: crate::workflow::WorkflowCheckpoint = serde_json::from_slice(&encoded)?;
            checkpoint.validate()?;
            let mut resumed = checkpoint.run;
            resumed.apply(crate::workflow::WorkflowEvent::Resumed)?;
            require_workflow_fixture(
                resumed.stage == crate::workflow::WorkflowStage::Implementing
                    && resumed.counters.model_invocations == 2
                    && resumed.counters.generated_tokens == 17
                    && resumed.counters.advisory_calls == 1,
                "resume did not restore exact stage and global budgets",
            )?;
            Ok(WorkflowAssertionObservation::from_run(resumed))
        }
        WorkflowControlAssertion::WebHarnessProjectionParity => {
            let state = workflow_fixture_state()?;
            let request = workflow_fixture_request(state.scratch.path())?;
            std::fs::write(state.scratch.path().join("existing.txt"), "delivered\n")?;
            let fingerprint =
                crate::workspace::ContentSnapshot::capture(state.scratch.path())?.fingerprint;
            std::fs::write(state.scratch.path().join("existing.txt"), "baseline\n")?;
            let plan_review =
                workflow_fixture_plan_review(&state.plan, crate::workflow::ReviewVerdict::Pass)?;
            let implementation =
                workflow_fixture_implementation(&state.plan, fingerprint.clone(), false)?;
            let code_review =
                workflow_fixture_code_review(fingerprint, crate::workflow::ReviewVerdict::Pass)?;
            let mut events = Vec::new();
            let outcome = run_scripted_delivery_workflow(
                &request,
                vec![
                    plan_submission_completion(&state.plan),
                    workflow_tool_completion(
                        "submit_plan_review",
                        serde_json::json!({
                            "id": plan_review.id,
                            "review": plan_review.artifact,
                        }),
                    ),
                    workflow_tool_completion(
                        "read_file",
                        serde_json::json!({"path": "existing.txt"}),
                    ),
                    workflow_tool_completion(
                        "replace_file",
                        serde_json::json!({"path": "existing.txt", "content": "delivered\n"}),
                    ),
                    workflow_tool_completion(
                        "submit_implementation",
                        serde_json::json!({
                            "id": implementation.id,
                            "implementation": implementation.artifact,
                        }),
                    ),
                    workflow_tool_completion(
                        "read_file",
                        serde_json::json!({"path": "existing.txt"}),
                    ),
                    workflow_tool_completion(
                        "submit_code_review",
                        serde_json::json!({
                            "id": code_review.id,
                            "review": code_review.artifact,
                        }),
                    ),
                ],
                state.scratch.path(),
                &mut |event| events.push(event),
            )?;
            let web_projection = crate::workflow::WorkflowSummary::from(&outcome.workflow.run);
            let harness_outcome = events.iter().find_map(|event| match event {
                AgentEvent::WorkflowCompleted { outcome, .. } => Some(*outcome),
                _ => None,
            });
            let stage_sequence = events
                .iter()
                .filter_map(|event| match event {
                    AgentEvent::WorkflowStageStarted { stage, .. } => Some(*stage),
                    _ => None,
                })
                .collect::<Vec<_>>();
            require_workflow_fixture(
                web_projection.stage == crate::workflow::WorkflowStage::Ready
                    && web_projection.outcome == harness_outcome
                    && stage_sequence
                        == vec![
                            crate::workflow::WorkflowStage::Planning,
                            crate::workflow::WorkflowStage::PlanReview,
                            crate::workflow::WorkflowStage::Implementing,
                            crate::workflow::WorkflowStage::Checking,
                            crate::workflow::WorkflowStage::CodeReview,
                            crate::workflow::WorkflowStage::Committing,
                        ]
                    && outcome.remaining_completions == 0
                    && outcome.reached_final
                    && !outcome.verified_completed
                    && outcome.termination_reason == crate::events::TerminationReason::Final
                    && outcome.generation_tool_names.len() == 7,
                "web and harness workflow projections diverged",
            )?;
            Ok(WorkflowAssertionObservation {
                stage_sequence,
                run: Some(outcome.workflow.run),
                llm_invocations: outcome.llm_invocations,
                tool_calls: outcome.tool_calls,
                ..WorkflowAssertionObservation::default()
            })
        }
        WorkflowControlAssertion::LegacyStateHasNoStrictClaim => {
            let state = workflow_fixture_state()?;
            let request = workflow_fixture_request(state.scratch.path())?;
            let persisted = crate::session_store::PersistedSession::from_parts(
                "legacy-workflow-fixture".to_string(),
                request,
                None,
                Some(state.scratch.path().to_path_buf()),
                false,
                crate::session_store::SessionStatus::Completed,
                Vec::new(),
            );
            let mut value = serde_json::to_value(persisted)?;
            let object = value
                .as_object_mut()
                .context("persisted session is not an object")?;
            object.remove("workflow");
            object.remove("completed_workflows");
            if let Some(request) = object
                .get_mut("request_template")
                .and_then(serde_json::Value::as_object_mut)
            {
                request.remove("intent");
                request.remove("workflow_policy");
                request.remove("workflow_checkpoint");
                request.remove("turn_id");
            }
            let restored: crate::session_store::PersistedSession = serde_json::from_value(value)?;
            require_workflow_fixture(
                restored.workflow.is_none()
                    && restored.completed_workflows.is_empty()
                    && restored.request_template.workflow_policy.is_none()
                    && restored.request_template.workflow_checkpoint.is_none(),
                "legacy state acquired a strict workflow claim",
            )?;
            Ok(WorkflowAssertionObservation::default())
        }
        _ => bail!("workflow assertion was routed to the wrong evaluator"),
    }
}

fn run_workflow_control_fixture(fixture: &WorkflowControlFixture) -> Result<ControlFixtureResult> {
    let execution = execute_workflow_assertion(fixture.assertion);
    let (passed, observation, diagnostic) = match execution {
        Ok(observation) => (true, observation, None),
        Err(error) => (
            false,
            WorkflowAssertionObservation::default(),
            Some(format!("{error:#}")),
        ),
    };
    let run = observation.run.as_ref();
    Ok(ControlFixtureResult {
        id: fixture.id.clone(),
        strict_workflow: true,
        workflow_assertion_passed: Some(passed),
        workflow_outcome: run.and_then(|run| run.outcome),
        workflow_stage_sequence: observation.stage_sequence,
        workflow_plan_sha256: run.and_then(|run| run.plan.as_ref().map(|plan| plan.sha256.clone())),
        workflow_plan_review_sha256: run
            .and_then(|run| run.plan_review.as_ref().map(|review| review.sha256.clone())),
        workflow_code_review_sha256: run
            .and_then(|run| run.code_review.as_ref().map(|review| review.sha256.clone())),
        workflow_plan_cycles: run.map_or(0, |run| run.counters.plan_cycles),
        workflow_repair_cycles: run.map_or(0, |run| run.counters.repair_cycles),
        workflow_advisory_calls: run.map_or(0, |run| run.counters.advisory_calls),
        workflow_rejected_actions: observation.rejected_actions,
        workflow_evidence_invalidations: observation.evidence_invalidations,
        reached_final: passed,
        termination_reason: if passed {
            "fixture_pass".to_string()
        } else {
            "fixture_failed".to_string()
        },
        llm_invocations: observation.llm_invocations,
        tool_calls: observation.tool_calls,
        false_completion: false,
        artifact_quality: diagnostic,
        ..ControlFixtureResult::default()
    })
}

fn fixture_runtime(
    fixture: &ControlFixture,
    workspace: &Path,
) -> Result<(
    Option<crate::harness_contract::AgentContract>,
    crate::workspace::WorkspaceGraph,
    crate::workspace::RepositoryContext,
)> {
    let (contract, workspace_graph) = fixture_workspace_graph(fixture)?;
    let task_context = crate::workspace::RepositoryContext::capture(workspace, workspace)?;
    let repository_context = if fixture.resumed_files.is_empty() {
        task_context
    } else {
        write_fixture_files(workspace, &fixture.resumed_files)?;
        crate::workspace::RepositoryContext::resume(
            workspace,
            workspace,
            task_context.task_baseline,
        )?
    };
    Ok((contract, workspace_graph, repository_context))
}

fn fixture_workspace_graph(
    fixture: &ControlFixture,
) -> Result<(
    Option<crate::harness_contract::AgentContract>,
    crate::workspace::WorkspaceGraph,
)> {
    let contract = fixture
        .contract
        .clone()
        .map(crate::harness_contract::HarnessContractDocument::normalize)
        .transpose()?;
    let base_graph = fixture
        .workspace
        .clone()
        .map(crate::workspace::WorkspaceConfigDocument::normalize)
        .transpose()?
        .unwrap_or_else(|| crate::workspace::WorkspaceGraph::legacy(&[]));
    let workspace_graph = contract
        .as_ref()
        .map(|contract| contract.compile_workspace_graph(base_graph.clone()))
        .transpose()?
        .unwrap_or(base_graph);
    Ok((contract, workspace_graph))
}

#[derive(Default)]
struct GoalAssertionObservation {
    run: Option<crate::goal::GoalRun>,
    checkpoint_sha256: Option<String>,
    llm_invocations: usize,
    tool_calls: usize,
}

fn goal_fixture_run(
    criteria: Vec<crate::goal::GoalCriterionInput>,
    continuation: crate::goal::GoalContinuationPolicy,
    budget: Option<crate::goal::GoalBudget>,
) -> Result<crate::goal::GoalRun> {
    crate::goal::GoalRun::start(
        "goal-control-fixture",
        "goal-control-session",
        "Qualify durable Goal control",
        criteria,
        continuation,
        budget,
        crate::goal::GoalConfigDocument::default().compile()?,
        "/tmp/pb-goal-control-fixture",
        1,
    )
}

fn goal_criteria(
    verifier: crate::goal::GoalVerifier,
    count: usize,
) -> Vec<crate::goal::GoalCriterionInput> {
    (1..=count)
        .map(|index| crate::goal::GoalCriterionInput {
            text: format!("Satisfy bounded criterion {index}"),
            verifier,
        })
        .collect()
}

fn goal_no_change_checkpoint(
    id: &str,
    usage: crate::workflow::WorkflowUsage,
) -> Result<crate::workflow::WorkflowCheckpoint> {
    let mut state = workflow_fixture_state()?;
    state.run.id = id.to_string();
    state
        .run
        .apply(crate::workflow::WorkflowEvent::UsageRecorded { usage })?;
    let fingerprint = state
        .run
        .repository
        .task_baseline
        .content
        .fingerprint
        .clone();
    state
        .run
        .apply(crate::workflow::WorkflowEvent::ImplementationSubmitted {
            implementation: workflow_fixture_implementation(
                &state.plan,
                fingerprint.clone(),
                true,
            )?,
        })?;
    state
        .run
        .apply(crate::workflow::WorkflowEvent::ChecksPassed {
            content_fingerprint: fingerprint,
            selected_checks: Vec::new(),
            evidence: crate::checks::CheckEvidenceLedger::default(),
        })?;
    crate::workflow::WorkflowCheckpoint::new(state.run)
}

fn execute_goal_assertion(assertion: GoalControlAssertion) -> Result<GoalAssertionObservation> {
    use crate::goal::{
        GoalCheckpoint, GoalCompletionBasis, GoalContinuationPolicy, GoalCriterionStatus,
        GoalMilestoneStatus, GoalOutcome, GoalStage, GoalVerifier,
    };

    match assertion {
        GoalControlAssertion::ExactPlanApproval => {
            let mut run = goal_fixture_run(
                goal_criteria(GoalVerifier::ReviewRequired, 2),
                GoalContinuationPolicy::ReviewPlanThenAutomatic,
                None,
            )?;
            require_workflow_fixture(
                run.approve_plan("stale-plan", 2).is_err()
                    && run.stage == GoalStage::AwaitingPlanApproval,
                "stale Goal plan approval changed controller state",
            )?;
            let plan = run.plan_sha256.clone();
            run.approve_plan(&plan, 3)?;
            require_workflow_fixture(
                run.stage == GoalStage::RunningMilestone
                    && run.active_milestone_id.is_some()
                    && run
                        .milestones
                        .iter()
                        .filter(|milestone| milestone.status == GoalMilestoneStatus::Running)
                        .count()
                        == 1,
                "exact Goal approval did not start exactly one milestone",
            )?;
            let checkpoint = GoalCheckpoint::new(run.clone())?;
            Ok(GoalAssertionObservation {
                run: Some(run),
                checkpoint_sha256: Some(checkpoint.sha256),
                ..GoalAssertionObservation::default()
            })
        }
        GoalControlAssertion::ModelToolAuthorityBound => {
            let state = workflow_fixture_state()?;
            let mut discuss = workflow_fixture_request(state.scratch.path())?;
            discuss.intent = Some(crate::workflow::TurnIntent::Discuss);
            discuss.max_steps = 2;
            discuss.tool_allowlist =
                Some(vec!["propose_goal".to_string(), "start_goal".to_string()]);
            let proposed = run_scripted_agent_steps(
                &discuss,
                vec![
                    workflow_tool_completion(
                        "propose_goal",
                        serde_json::json!({
                            "objective": "Qualify Goal mode",
                            "criteria": ["Keep activation explicit"]
                        }),
                    ),
                    ScriptedCompletion {
                        content:
                            r#"{"type":"final","content":"The Goal still requires user review."}"#
                                .to_string(),
                        truncated: false,
                    },
                ],
                state.scratch.path(),
                &mut |_| {},
            )?;
            let mut auto = discuss.clone();
            auto.intent = Some(crate::workflow::TurnIntent::Auto);
            auto.turn_id = "goal-auto-current-turn".to_string();
            auto.max_steps = 1;
            let started = run_scripted_agent_steps(
                &auto,
                vec![workflow_tool_completion(
                    "start_goal",
                    serde_json::json!({
                        "source_turn_id": "goal-auto-current-turn",
                        "objective": "Qualify Goal mode",
                        "criteria": ["Keep activation explicit"]
                    }),
                )],
                state.scratch.path(),
                &mut |_| {},
            )?;
            require_workflow_fixture(
                proposed.goal_proposal.is_some()
                    && proposed.requested_goal.is_none()
                    && started.requested_goal.as_ref().is_some_and(|proposal| {
                        proposal.source_turn_id == "goal-auto-current-turn"
                    }),
                "model Goal tools granted authority or lost exact-turn binding",
            )?;
            Ok(GoalAssertionObservation {
                run: Some(goal_fixture_run(
                    goal_criteria(GoalVerifier::ReviewRequired, 1),
                    GoalContinuationPolicy::ReviewPlanThenAutomatic,
                    None,
                )?),
                llm_invocations: proposed.llm_invocations + started.llm_invocations,
                tool_calls: proposed.tool_calls + started.tool_calls,
                ..GoalAssertionObservation::default()
            })
        }
        GoalControlAssertion::SequentialMilestones => {
            let mut run = goal_fixture_run(
                goal_criteria(GoalVerifier::WorkflowReady, 2),
                GoalContinuationPolicy::ReviewPlanThenAutomatic,
                None,
            )?;
            let plan = run.plan_sha256.clone();
            run.approve_plan(&plan, 2)?;
            run.finish_active_workflow(
                goal_no_change_checkpoint(
                    "goal-sequential-1",
                    crate::workflow::WorkflowUsage::default(),
                )?,
                3,
            )?;
            require_workflow_fixture(
                run.stage == GoalStage::RunningMilestone
                    && run.counters.workflows == 1
                    && run
                        .milestones
                        .iter()
                        .filter(|milestone| milestone.status == GoalMilestoneStatus::Running)
                        .count()
                        == 1,
                "Goal did not advance sequentially after a Ready workflow",
            )?;
            run.finish_active_workflow(
                goal_no_change_checkpoint(
                    "goal-sequential-2",
                    crate::workflow::WorkflowUsage::default(),
                )?,
                4,
            )?;
            require_workflow_fixture(
                run.stage == GoalStage::Completed
                    && run.completion_basis == Some(GoalCompletionBasis::MachineVerified)
                    && run.counters.workflows == 2,
                "machine-verifiable Goal did not close after all sequential milestones",
            )?;
            Ok(GoalAssertionObservation {
                run: Some(run),
                ..GoalAssertionObservation::default()
            })
        }
        GoalControlAssertion::PauseCheckpointResume => {
            let mut run = goal_fixture_run(
                goal_criteria(GoalVerifier::ReviewRequired, 1),
                GoalContinuationPolicy::ReviewPlanThenAutomatic,
                None,
            )?;
            let plan = run.plan_sha256.clone();
            run.approve_plan(&plan, 2)?;
            let state = workflow_fixture_state()?;
            let mut workflow = state.run;
            workflow.apply(crate::workflow::WorkflowEvent::UsageRecorded {
                usage: crate::workflow::WorkflowUsage {
                    model_invocations: 2,
                    generated_tokens: 37,
                    ..crate::workflow::WorkflowUsage::default()
                },
            })?;
            run.checkpoint_active_workflow(crate::workflow::WorkflowCheckpoint::new(workflow)?, 3)?;
            require_workflow_fixture(!run.request_pause(4)?, "active Goal paused mid-stage")?;
            run.pause_at_boundary(5)?;
            let checkpoint = GoalCheckpoint::new(run)?;
            let before = checkpoint.run.effective_counters();
            let encoded = serde_json::to_vec(&checkpoint)?;
            let restored: GoalCheckpoint = serde_json::from_slice(&encoded)?;
            restored.validate()?;
            let mut run = restored.run;
            run.resume(6)?;
            require_workflow_fixture(
                run.stage == GoalStage::RunningMilestone
                    && run.effective_counters() == before
                    && run
                        .current_milestone()
                        .and_then(|item| item.workflow.as_ref())
                        .is_some(),
                "Goal resume lost its workflow checkpoint or effective counters",
            )?;
            Ok(GoalAssertionObservation {
                run: Some(run),
                ..GoalAssertionObservation::default()
            })
        }
        GoalControlAssertion::AmendmentPreservesEvidence => {
            let mut run = goal_fixture_run(
                goal_criteria(GoalVerifier::ReviewRequired, 1),
                GoalContinuationPolicy::ReviewPlanThenAutomatic,
                None,
            )?;
            let plan = run.plan_sha256.clone();
            run.approve_plan(&plan, 2)?;
            run.finish_active_workflow(
                goal_no_change_checkpoint(
                    "goal-amendment-evidence",
                    crate::workflow::WorkflowUsage::default(),
                )?,
                3,
            )?;
            let before = GoalCheckpoint::new(run.clone())?;
            run.propose_amendment(
                "goal-amendment-1",
                before.sha256,
                "Qualify durable Goal control with clearer wording",
                goal_criteria(GoalVerifier::ReviewRequired, 1),
                GoalContinuationPolicy::ManualMilestones,
                None,
                4,
            )?;
            let replacement = run
                .pending_amendment
                .as_ref()
                .context("Goal amendment draft missing")?
                .replacement_plan_sha256
                .clone();
            run.approve_amendment(&replacement, 5)?;
            require_workflow_fixture(
                run.stage == GoalStage::AwaitingUserReview
                    && run.plan_version == 2
                    && run.retired_criteria.len() == 1
                    && run.criteria[0].status == GoalCriterionStatus::EvidenceReady
                    && run.criteria[0]
                        .evidence_ids
                        .iter()
                        .any(|evidence| evidence.starts_with("carry-forward:")),
                "approved Goal amendment discarded compatible evidence or history",
            )?;
            Ok(GoalAssertionObservation {
                run: Some(run),
                ..GoalAssertionObservation::default()
            })
        }
        GoalControlAssertion::CompletionBasisBound => {
            let mut machine = goal_fixture_run(
                goal_criteria(GoalVerifier::WorkflowReady, 1),
                GoalContinuationPolicy::ReviewPlanThenAutomatic,
                None,
            )?;
            let plan = machine.plan_sha256.clone();
            machine.approve_plan(&plan, 2)?;
            machine.finish_active_workflow(
                goal_no_change_checkpoint(
                    "goal-machine-complete",
                    crate::workflow::WorkflowUsage::default(),
                )?,
                3,
            )?;
            let mut reviewed = goal_fixture_run(
                goal_criteria(GoalVerifier::ReviewRequired, 1),
                GoalContinuationPolicy::ReviewPlanThenAutomatic,
                None,
            )?;
            let plan = reviewed.plan_sha256.clone();
            reviewed.approve_plan(&plan, 2)?;
            reviewed.finish_active_workflow(
                goal_no_change_checkpoint(
                    "goal-user-complete",
                    crate::workflow::WorkflowUsage::default(),
                )?,
                3,
            )?;
            let checkpoint = GoalCheckpoint::new(reviewed.clone())?;
            reviewed.accept(&checkpoint.sha256, &checkpoint.sha256, 4)?;
            require_workflow_fixture(
                machine.completion_basis == Some(GoalCompletionBasis::MachineVerified)
                    && reviewed.completion_basis == Some(GoalCompletionBasis::UserAccepted)
                    && reviewed.stage == GoalStage::Completed,
                "Goal completion conflated machine evidence with explicit user acceptance",
            )?;
            Ok(GoalAssertionObservation {
                run: Some(reviewed),
                ..GoalAssertionObservation::default()
            })
        }
        GoalControlAssertion::BudgetAndCancellationAccounting => {
            let mut budget = crate::goal::GoalBudget::standard();
            budget.total_model_invocations = 1;
            let mut exhausted = goal_fixture_run(
                goal_criteria(GoalVerifier::WorkflowReady, 2),
                GoalContinuationPolicy::ReviewPlanThenAutomatic,
                Some(budget),
            )?;
            let plan = exhausted.plan_sha256.clone();
            exhausted.approve_plan(&plan, 2)?;
            exhausted.finish_active_workflow(
                goal_no_change_checkpoint(
                    "goal-budget-first",
                    crate::workflow::WorkflowUsage {
                        model_invocations: 1,
                        generated_tokens: 10,
                        ..crate::workflow::WorkflowUsage::default()
                    },
                )?,
                3,
            )?;
            let mut cancelled = goal_fixture_run(
                goal_criteria(GoalVerifier::ReviewRequired, 1),
                GoalContinuationPolicy::ReviewPlanThenAutomatic,
                None,
            )?;
            let plan = cancelled.plan_sha256.clone();
            cancelled.approve_plan(&plan, 2)?;
            let state = workflow_fixture_state()?;
            let mut workflow = state.run;
            workflow.apply(crate::workflow::WorkflowEvent::UsageRecorded {
                usage: crate::workflow::WorkflowUsage {
                    model_invocations: 2,
                    generated_tokens: 29,
                    ..crate::workflow::WorkflowUsage::default()
                },
            })?;
            cancelled.checkpoint_active_workflow(
                crate::workflow::WorkflowCheckpoint::new(workflow)?,
                3,
            )?;
            cancelled.cancel(4);
            require_workflow_fixture(
                exhausted.stage == GoalStage::Failed
                    && exhausted.outcome == Some(GoalOutcome::BudgetExhausted)
                    && exhausted.counters.model_invocations == 1
                    && cancelled.stage == GoalStage::Cancelled
                    && cancelled.counters.workflows == 1
                    && cancelled.counters.model_invocations == 2
                    && cancelled.effective_counters() == cancelled.counters,
                "Goal budgets reset between milestones or cancellation double-counted usage",
            )?;
            Ok(GoalAssertionObservation {
                run: Some(exhausted),
                ..GoalAssertionObservation::default()
            })
        }
    }
}

fn run_goal_control_fixture(fixture: &GoalControlFixture) -> Result<ControlFixtureResult> {
    let execution = execute_goal_assertion(fixture.assertion);
    let (passed, observation, diagnostic) = match execution {
        Ok(observation) => (true, observation, None),
        Err(error) => (
            false,
            GoalAssertionObservation::default(),
            Some(format!("{error:#}")),
        ),
    };
    let run = observation.run.as_ref();
    let counters = run.map(crate::goal::GoalRun::effective_counters);
    Ok(ControlFixtureResult {
        id: fixture.id.clone(),
        strict_goal: true,
        goal_assertion_passed: Some(passed),
        goal_stage: run.map(|run| run.stage),
        goal_outcome: run.and_then(|run| run.outcome),
        goal_completion_basis: run.and_then(|run| run.completion_basis),
        goal_checkpoint_sha256: observation.checkpoint_sha256,
        goal_plan_sha256: run.map(|run| run.plan_sha256.clone()),
        goal_completed_milestones: run.map_or(0, |run| {
            run.milestones
                .iter()
                .filter(|milestone| milestone.status.is_completed())
                .count()
        }),
        goal_total_milestones: run.map_or(0, |run| {
            run.milestones
                .iter()
                .filter(|milestone| !milestone.status.is_superseded())
                .count()
        }),
        goal_workflows: counters.as_ref().map_or(0, |counters| counters.workflows),
        goal_model_invocations: counters
            .as_ref()
            .map_or(0, |counters| counters.model_invocations),
        goal_generated_tokens: counters
            .as_ref()
            .map_or(0, |counters| counters.generated_tokens),
        reached_final: passed,
        termination_reason: if passed {
            "fixture_pass".to_string()
        } else {
            "fixture_failed".to_string()
        },
        llm_invocations: observation.llm_invocations,
        tool_calls: observation.tool_calls,
        artifact_quality: diagnostic,
        ..ControlFixtureResult::default()
    })
}

pub fn run_control_fixture_corpus() -> Result<Vec<ControlFixtureResult>> {
    let corpus = control_fixture_corpus()?;
    let mut results = corpus
        .fixtures
        .iter()
        .map(run_control_fixture)
        .collect::<Result<Vec<_>>>()?;
    results.extend(
        corpus
            .workflow_fixtures
            .iter()
            .map(run_workflow_control_fixture)
            .collect::<Result<Vec<_>>>()?,
    );
    results.extend(
        corpus
            .goal_fixtures
            .iter()
            .map(run_goal_control_fixture)
            .collect::<Result<Vec<_>>>()?,
    );
    Ok(results)
}

pub fn run_eval_command(args: HarnessEvalArgs) -> Result<()> {
    let corpus = control_fixture_corpus()?;
    let fixtures = selected_control_fixtures(&corpus, args.suite);
    let workflow_fixtures = selected_workflow_fixtures(&corpus, args.suite);
    let goal_fixtures = selected_goal_fixtures(&corpus, args.suite);
    let (configuration, results) = match args.model.as_deref() {
        Some(model) => {
            run_real_model_corpus(&args, model, &fixtures, &workflow_fixtures, &goal_fixtures)?
        }
        None => (
            scripted_configuration(args.suite),
            fixtures
                .iter()
                .map(|fixture| run_control_fixture(fixture))
                .chain(
                    workflow_fixtures
                        .iter()
                        .map(|fixture| run_workflow_control_fixture(fixture)),
                )
                .chain(
                    goal_fixtures
                        .iter()
                        .map(|fixture| run_goal_control_fixture(fixture)),
                )
                .collect::<Result<Vec<_>>>()?,
        ),
    };
    let records = build_selected_eval_records(
        corpus.version,
        &fixtures,
        &workflow_fixtures,
        &goal_fixtures,
        configuration,
        results,
    )?;
    write_eval_jsonl(args.jsonl.as_deref(), &records)?;
    let table = render_eval_table(&records);
    if args.jsonl.is_some() {
        print!("{table}");
    } else {
        eprint!("{table}");
    }
    let failed = records
        .iter()
        .filter(|record| !record.protocol_pass)
        .map(|record| record.result.id.as_str())
        .collect::<Vec<_>>();
    if !failed.is_empty() {
        bail!("harness protocol regressions: {}", failed.join(", "));
    }
    Ok(())
}

fn selected_control_fixtures(
    corpus: &ControlFixtureCorpus,
    suite: HarnessEvalSuite,
) -> Vec<&ControlFixture> {
    match suite {
        HarnessEvalSuite::Control => corpus.fixtures.iter().collect(),
        HarnessEvalSuite::SmallModel => corpus
            .small_model_fixtures
            .iter()
            .map(|id| {
                corpus
                    .fixtures
                    .iter()
                    .find(|fixture| fixture.id == *id)
                    .expect("validated small-model fixture id must resolve")
            })
            .collect(),
    }
}

fn selected_workflow_fixtures(
    corpus: &ControlFixtureCorpus,
    suite: HarnessEvalSuite,
) -> Vec<&WorkflowControlFixture> {
    match suite {
        HarnessEvalSuite::Control => corpus.workflow_fixtures.iter().collect(),
        HarnessEvalSuite::SmallModel => Vec::new(),
    }
}

fn selected_goal_fixtures(
    corpus: &ControlFixtureCorpus,
    suite: HarnessEvalSuite,
) -> Vec<&GoalControlFixture> {
    match suite {
        HarnessEvalSuite::Control => corpus.goal_fixtures.iter().collect(),
        HarnessEvalSuite::SmallModel => Vec::new(),
    }
}

fn scripted_configuration(suite: HarnessEvalSuite) -> HarnessEvalConfiguration {
    HarnessEvalConfiguration {
        mode: "scripted".to_string(),
        backend: "scripted".to_string(),
        suite: harness_eval_suite_name(suite).to_string(),
        model: None,
        model_dir: None,
        max_tokens: 256,
        ctx_size: SCRIPTED_EVAL_CONTEXT_SIZE,
        threads: None,
        threads_batch: None,
        gpu_layers: 0,
        temperature: 0.0,
        top_k: 1,
        seed: 0,
        flashmoe_resource_policy_version:
            crate::inference::flashmoe::HARNESS_RESOURCE_POLICY_VERSION,
        workspace_config_sha256: None,
        executor_policy: Vec::new(),
    }
}

fn run_real_model_corpus(
    args: &HarnessEvalArgs,
    model: &str,
    fixtures: &[&ControlFixture],
    workflow_fixtures: &[&WorkflowControlFixture],
    goal_fixtures: &[&GoalControlFixture],
) -> Result<(HarnessEvalConfiguration, Vec<ControlFixtureResult>)> {
    let user_config = UserConfig::load()?;
    let models_root = args
        .model_dir
        .clone()
        .or_else(|| user_config.effective_model_dir())
        .unwrap_or_else(crate::default_models_dir);
    let mut engine = LocalModelEvalEngine::load(model, &models_root, args.gpu_layers)?;
    ensure_flashmoe_eval_policy(
        engine.backend_name(),
        crate::inference::flashmoe::HARNESS_RESOURCE_POLICY_VERSION,
    )?;
    let configuration = HarnessEvalConfiguration {
        mode: "local_model".to_string(),
        backend: engine.backend_name().to_string(),
        suite: harness_eval_suite_name(args.suite).to_string(),
        model: Some(model.to_string()),
        model_dir: Some(
            models_root
                .canonicalize()
                .unwrap_or(models_root.clone())
                .display()
                .to_string(),
        ),
        max_tokens: args.max_tokens,
        ctx_size: args.ctx_size,
        threads: args.threads,
        threads_batch: args.threads_batch,
        gpu_layers: args.gpu_layers,
        temperature: args.temperature,
        top_k: args.top_k,
        seed: args.seed,
        flashmoe_resource_policy_version:
            crate::inference::flashmoe::HARNESS_RESOURCE_POLICY_VERSION,
        workspace_config_sha256: None,
        executor_policy: Vec::new(),
    };
    let mut results =
        Vec::with_capacity(fixtures.len() + workflow_fixtures.len() + goal_fixtures.len());
    for fixture in fixtures {
        results.push(run_real_model_fixture(
            fixture,
            &configuration,
            &mut engine,
        )?);
    }
    for fixture in workflow_fixtures {
        results.push(run_workflow_control_fixture(fixture)?);
    }
    for fixture in goal_fixtures {
        results.push(run_goal_control_fixture(fixture)?);
    }
    Ok((configuration, results))
}

const fn harness_eval_suite_name(suite: HarnessEvalSuite) -> &'static str {
    match suite {
        HarnessEvalSuite::Control => "control",
        HarnessEvalSuite::SmallModel => "small_model",
    }
}

fn ensure_flashmoe_eval_policy(backend: &str, policy_version: u32) -> Result<()> {
    if backend == "flashmoe" && policy_version == 0 {
        bail!("FlashMoe harness evaluation is disabled until a bounded resource policy is active");
    }
    Ok(())
}

fn run_real_model_fixture(
    fixture: &ControlFixture,
    configuration: &HarnessEvalConfiguration,
    engine: &mut LocalModelEvalEngine,
) -> Result<ControlFixtureResult> {
    let scratch = tempfile::Builder::new()
        .prefix("pb-model-control-fixture-")
        .tempdir()
        .context("failed to create model harness control fixture scratch directory")?;
    initialize_fixture_workspace(scratch.path(), &fixture.initial_files)?;
    let (contract, workspace_graph, repository_context) = fixture_runtime(fixture, scratch.path())?;
    let request = AgentRequest {
        task: format!(
            "Harness control evaluation. Complete the repository task implied by this control objective, using only the exposed tools: {}",
            fixture.hypothesis
        ),
        turn_id: format!("harness-eval-turn-{}", fixture.id),
        intent: Some(crate::workflow::TurnIntent::Deliver),
        workflow_policy: None,
        workflow_stage: None,
        workflow_expected_content_fingerprint: None,
        workflow_action_first_turn: false,
        workflow_creation_path_order: Vec::new(),
        workflow_work_units: None,
        workflow_stage_evidence: None,
        workflow_checkpoint: None,
        conversation_handoff: None,
        legacy_prompt_owned_delivery: true,
        model: configuration.model.clone().unwrap_or_default(),
        model_dir: configuration.model_dir.as_ref().map(PathBuf::from),
        workdir: Some(scratch.path().to_path_buf()),
        branch: None,
        max_steps: fixture.max_steps,
        max_tokens: configuration.max_tokens,
        turn_max_tokens_cap: Some(configuration.max_tokens),
        tool_allowlist: Some(fixture.tool_allowlist.clone()),
        observation_rendering: crate::workflow::ObservationRendering::Native,
        controller_delete_elision: false,
        accept_existing_workspace_changes: false,
        ctx_size: configuration.ctx_size,
        threads: configuration.threads,
        threads_batch: configuration.threads_batch,
        gpu_layers: configuration.gpu_layers,
        temperature: configuration.temperature,
        profile: fixture.profile,
        infer_profile: false,
        sub_agent_depth: 0,
        repository_less: false,
        top_k: configuration.top_k,
        seed: configuration.seed,
        environment: None,
        environment_evidence_context: None,
        workspace_graph: Some(workspace_graph),
        repository_context: Some(repository_context),
        prior_check_evidence: crate::checks::CheckEvidenceLedger::default(),
        session_id: format!("harness-eval-{}", fixture.id),
        attachments: Vec::new(),
        goal_context: None,
        contract,
    };
    let mut events = Vec::new();
    let outcome = run_local_model_eval_steps(engine, &request, scratch.path(), &mut |event| {
        events.push(event)
    })?;
    summarize_fixture(fixture, scratch.path(), outcome.into(), &events, true)
}

#[cfg(test)]
fn build_eval_records(
    corpus: &ControlFixtureCorpus,
    configuration: HarnessEvalConfiguration,
    results: Vec<ControlFixtureResult>,
) -> Result<Vec<HarnessEvalRecord>> {
    let fixtures = corpus.fixtures.iter().collect::<Vec<_>>();
    let workflow_fixtures = corpus.workflow_fixtures.iter().collect::<Vec<_>>();
    let goal_fixtures = corpus.goal_fixtures.iter().collect::<Vec<_>>();
    build_selected_eval_records(
        corpus.version,
        &fixtures,
        &workflow_fixtures,
        &goal_fixtures,
        configuration,
        results,
    )
}

fn build_selected_eval_records(
    fixture_version: u32,
    fixtures: &[&ControlFixture],
    workflow_fixtures: &[&WorkflowControlFixture],
    goal_fixtures: &[&GoalControlFixture],
    configuration: HarnessEvalConfiguration,
    results: Vec<ControlFixtureResult>,
) -> Result<Vec<HarnessEvalRecord>> {
    let expected_count = fixtures.len() + workflow_fixtures.len() + goal_fixtures.len();
    if results.len() != expected_count {
        bail!(
            "harness evaluation produced {} results for {} fixtures",
            results.len(),
            expected_count
        );
    }
    let mut records = Vec::with_capacity(expected_count);
    let mut results = results.into_iter();
    for fixture in fixtures {
        let result = results.next().context("missing legacy fixture result")?;
        if fixture.id != result.id {
            bail!(
                "harness evaluation result order mismatch: expected {}, got {}",
                fixture.id,
                result.id
            );
        }
        let protocol_failures = protocol_failures(&fixture.expected, &result);
        let mut record_configuration = configuration.clone();
        if let Some((sha256, executor_policy)) = fixture_workspace_metadata(fixture)? {
            record_configuration.workspace_config_sha256 = Some(sha256);
            record_configuration.executor_policy = executor_policy;
        }
        records.push(HarnessEvalRecord {
            schema_version: HARNESS_EVAL_SCHEMA_VERSION,
            fixture_version,
            configuration: record_configuration,
            protocol_pass: protocol_failures.is_empty(),
            protocol_failures,
            result,
        });
    }
    for fixture in workflow_fixtures {
        let result = results.next().context("missing workflow fixture result")?;
        if fixture.id != result.id {
            bail!(
                "harness evaluation result order mismatch: expected {}, got {}",
                fixture.id,
                result.id
            );
        }
        let protocol_failures = if result.workflow_assertion_passed == Some(true) {
            Vec::new()
        } else {
            vec![format!(
                "workflow assertion failed: {}",
                result
                    .artifact_quality
                    .as_deref()
                    .unwrap_or("no diagnostic was recorded")
            )]
        };
        records.push(HarnessEvalRecord {
            schema_version: HARNESS_EVAL_SCHEMA_VERSION,
            fixture_version,
            configuration: configuration.clone(),
            protocol_pass: protocol_failures.is_empty(),
            protocol_failures,
            result,
        });
    }
    for fixture in goal_fixtures {
        let result = results.next().context("missing Goal fixture result")?;
        if fixture.id != result.id {
            bail!(
                "harness evaluation result order mismatch: expected {}, got {}",
                fixture.id,
                result.id
            );
        }
        let protocol_failures = if result.goal_assertion_passed == Some(true) {
            Vec::new()
        } else {
            vec![format!(
                "Goal assertion failed: {}",
                result
                    .artifact_quality
                    .as_deref()
                    .unwrap_or("no diagnostic was recorded")
            )]
        };
        records.push(HarnessEvalRecord {
            schema_version: HARNESS_EVAL_SCHEMA_VERSION,
            fixture_version,
            configuration: configuration.clone(),
            protocol_pass: protocol_failures.is_empty(),
            protocol_failures,
            result,
        });
    }
    Ok(records)
}

fn validate_eval_record_schema(record: &HarnessEvalRecord) -> Result<()> {
    if record.schema_version != HARNESS_EVAL_SCHEMA_VERSION {
        bail!(
            "unsupported harness evaluation schema {}; expected {}",
            record.schema_version,
            HARNESS_EVAL_SCHEMA_VERSION
        );
    }
    Ok(())
}

fn protocol_failures(
    expected: &ControlFixtureExpectation,
    actual: &ControlFixtureResult,
) -> Vec<String> {
    let mut failures = Vec::new();
    macro_rules! compare {
        ($field:ident) => {
            if actual.$field != expected.$field {
                failures.push(format!(
                    "{} expected {:?}, got {:?}",
                    stringify!($field),
                    expected.$field,
                    actual.$field
                ));
            }
        };
    }
    compare!(reached_final);
    compare!(contract_status);
    compare!(verified_completed);
    compare!(termination_reason);
    compare!(llm_invocations);
    compare!(tool_calls);
    compare!(false_completion);
    compare!(named_check_compliance);
    compare!(observed_paths);
    if let Some(expected) = &expected.handoff_outcome
        && actual.handoff_outcome.as_ref() != Some(expected)
    {
        failures.push(format!(
            "handoff_outcome expected {:?}, got {:?}",
            expected, actual.handoff_outcome
        ));
    }
    if let Some(expected) = &expected.selected_checks
        && &actual.selected_checks != expected
    {
        failures.push(format!(
            "selected_checks expected {:?}, got {:?}",
            expected, actual.selected_checks
        ));
    }
    for (field, expected, actual) in [
        (
            "executed_checks",
            expected.executed_checks,
            actual.executed_checks,
        ),
        (
            "reused_checks",
            expected.reused_checks,
            actual.reused_checks,
        ),
        ("repair_turns", expected.repair_turns, actual.repair_turns),
    ] {
        if let Some(expected) = expected
            && actual != expected
        {
            failures.push(format!("{field} expected {expected}, got {actual}"));
        }
    }
    if let Some(expected) = &expected.executor_starts
        && &actual.executor_starts != expected
    {
        failures.push(format!(
            "executor_starts expected {:?}, got {:?}",
            expected, actual.executor_starts
        ));
    }
    if let Some(expected) = &expected.commit_disposition
        && actual.commit_disposition.as_ref() != Some(expected)
    {
        failures.push(format!(
            "commit_disposition expected {:?}, got {:?}",
            expected, actual.commit_disposition
        ));
    }
    failures
}

fn fixture_workspace_metadata(fixture: &ControlFixture) -> Result<Option<(String, Vec<String>)>> {
    let Some(document) = fixture.workspace.clone() else {
        return Ok(None);
    };
    let graph = document.normalize()?;
    let normalized = serde_json::to_vec(&graph.to_document())
        .context("failed to serialize normalized fixture workspace")?;
    let executor_policy = graph
        .executors
        .iter()
        .map(|(id, executor)| {
            let kind = match executor.kind {
                crate::workspace::ExecutorKind::Project => "project",
                crate::workspace::ExecutorKind::Local => "local",
                crate::workspace::ExecutorKind::Container => "container",
            };
            format!("{id}:{kind}")
        })
        .collect();
    Ok(Some((
        format!("{:x}", Sha256::digest(normalized)),
        executor_policy,
    )))
}

fn write_eval_jsonl(path: Option<&Path>, records: &[HarnessEvalRecord]) -> Result<()> {
    let mut writer: Box<dyn Write> = match path {
        Some(path) => Box::new(BufWriter::new(File::create(path).with_context(|| {
            format!(
                "failed to create harness evaluation JSONL {}",
                path.display()
            )
        })?)),
        None => Box::new(BufWriter::new(std::io::stdout().lock())),
    };
    for record in records {
        validate_eval_record_schema(record)?;
        serde_json::to_writer(&mut writer, record)
            .context("failed to encode harness evaluation JSONL")?;
        writer
            .write_all(b"\n")
            .context("failed to write harness evaluation JSONL")?;
    }
    writer
        .flush()
        .context("failed to flush harness evaluation JSONL")
}

fn render_eval_table(records: &[HarnessEvalRecord]) -> String {
    let mut table = String::from(
        "fixture                         pass handoff          checks reuse execs commit      valid named false loop turns ctx_hi schema_ch latency_ms tokens energy_kwh termination\n",
    );
    for record in records {
        let result = &record.result;
        let valid = if result.llm_invocations == 0 {
            "0/0".to_string()
        } else {
            format!("{}/{}", result.valid_actions, result.llm_invocations)
        };
        let named = result
            .named_check_compliance
            .map(|value| if value { "yes" } else { "no" })
            .unwrap_or("-");
        let energy = result
            .energy_kwh
            .map(|value| format!("{value:.3e}"))
            .unwrap_or_else(|| "-".to_string());
        let handoff = result
            .handoff_outcome
            .map(|outcome| format!("{outcome:?}").to_ascii_lowercase())
            .unwrap_or_else(|| "-".to_string());
        table.push_str(&format!(
            "{:<31} {:<4} {:<16} {:<6} {:<5} {:<5} {:<11} {:<5} {:<5} {:<5} {:<4} {:<5} {:<6.1} {:<9} {:<10} {:<6} {:<10} {}\n",
            result.id,
            if record.protocol_pass { "yes" } else { "no" },
            handoff,
            result.executed_checks,
            result.reused_checks,
            result.executor_starts.len(),
            result.commit_disposition.as_deref().unwrap_or("-"),
            valid,
            named,
            if result.false_completion { "yes" } else { "no" },
            if result.recovery_loop { "yes" } else { "no" },
            result.llm_invocations,
            result.context.prompt_utilization_bps_high_water as f64 / 100.0,
            result.context.tool_schema_chars_high_water,
            result.latency_ms,
            result.prompt_tokens.saturating_add(result.generated_tokens),
            energy,
            result.termination_reason,
        ));
    }
    table
}

struct FixtureOutcome {
    reached_final: bool,
    contract_status: ContractStatus,
    verified_completed: bool,
    termination_reason: String,
    remaining_completions: usize,
}

impl From<ScriptedAgentOutcome> for FixtureOutcome {
    fn from(outcome: ScriptedAgentOutcome) -> Self {
        Self {
            reached_final: outcome.reached_final,
            contract_status: outcome.contract_status,
            verified_completed: outcome.verified_completed,
            termination_reason: outcome.termination_reason.to_string(),
            remaining_completions: outcome.remaining_completions,
        }
    }
}

impl From<LocalModelEvalOutcome> for FixtureOutcome {
    fn from(outcome: LocalModelEvalOutcome) -> Self {
        Self {
            reached_final: outcome.reached_final,
            contract_status: outcome.contract_status,
            verified_completed: outcome.verified_completed,
            termination_reason: outcome.termination_reason.to_string(),
            remaining_completions: 0,
        }
    }
}

fn summarize_fixture(
    fixture: &ControlFixture,
    workspace: &Path,
    outcome: FixtureOutcome,
    events: &[AgentEvent],
    record_commit_oid: bool,
) -> Result<ControlFixtureResult> {
    let llm_invocations = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::LlmInvocation { .. }))
        .count();
    let invalid_actions = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::Error { summary, .. }
                    if summary.starts_with("Invalid pb JSON action")
            )
        })
        .count();
    let corrections = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::Correction { .. }))
        .count();
    let gate_corrections = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::Correction { summary, .. }
                    if summary == "Completion gate blocked final response"
                        || summary == "Acceptance contract rejected final response"
            )
        })
        .count();
    let errors = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::Error { .. }))
        .count();
    let blocked_tool_loops = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::Correction { summary, .. }
                    if summary == "Repeated tool call blocked"
            )
        })
        .count();
    let final_events = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::Final { .. }))
        .count();
    let model_run_check_calls = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::ToolCall { tool, .. } if tool == "run_check"
            )
        })
        .count();
    let executed_checks = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::CheckResult {
                    reused: false,
                    skip_reason: None,
                    ..
                }
            )
        })
        .count();
    let reused_checks = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::CheckResult { reused: true, .. }))
        .count();
    let failed_checks = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::CheckResult {
                    success: false,
                    skip_reason: None,
                    ..
                }
            )
        })
        .count();
    let skipped_checks = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::CheckResult {
                    skip_reason: Some(_),
                    ..
                }
            )
        })
        .count();
    let mut selected_checks = std::collections::BTreeSet::new();
    let mut affected_components = std::collections::BTreeSet::new();
    let mut handoff_outcome = None;
    for event in events {
        if let AgentEvent::HandoffSummary { summary, .. } = event {
            handoff_outcome = Some(summary.outcome);
            selected_checks.extend(summary.checks.iter().map(|check| check.check_id.clone()));
            affected_components.extend(summary.affected_components.iter().cloned());
        }
    }
    let executor_starts = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ExecutorStarted {
                executor_id,
                success: true,
                ..
            } => Some(executor_id.clone()),
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    let configured_executors = fixture
        .workspace
        .as_ref()
        .map(|workspace| {
            workspace
                .executors
                .iter()
                .map(|executor| executor.id.clone())
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();
    let avoided_executor_starts = configured_executors
        .difference(&executor_starts)
        .cloned()
        .collect::<Vec<_>>();
    let team_messages = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::TeamMessage { .. }))
        .count();
    let repair_turns = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::Correction { summary, .. }
                    if summary.contains("handoff teammate returned failed checks")
            )
        })
        .count();
    let commit = events.iter().rev().find_map(|event| match event {
        AgentEvent::CommitResult {
            success,
            created,
            reused,
            oid,
            ..
        } => Some((
            if *created {
                "created"
            } else if *reused {
                "reused"
            } else if *success {
                "not_needed"
            } else {
                "blocked"
            }
            .to_string(),
            oid.clone(),
        )),
        _ => None,
    });
    let output_fingerprints = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::CheckResult {
                output_fingerprint: Some(fingerprint),
                ..
            } => Some(fingerprint.clone()),
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let (latency_ms, prompt_tokens, generated_tokens, energy_kwh) = events.iter().fold(
        (0u64, 0usize, 0usize, 0.0f64),
        |(latency, prompt, generated, energy), event| match event {
            AgentEvent::LlmInvocation {
                duration_ms,
                prompt_tokens,
                generated_tokens,
                energy_kwh,
                ..
            } => (
                latency.saturating_add(*duration_ms),
                prompt.saturating_add(*prompt_tokens),
                generated.saturating_add(*generated_tokens),
                energy + energy_kwh.unwrap_or(0.0),
            ),
            _ => (latency, prompt, generated, energy),
        },
    );
    let context = summarize_context_metrics(events);
    let tool_trace = summarize_tool_trace(events);
    let required_check_ids = fixture
        .contract
        .as_ref()
        .map(|contract| {
            contract
                .checks
                .iter()
                .filter(|check| check.required)
                .map(|check| check.id.as_str())
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();
    let named_check_compliance = if required_check_ids.is_empty() {
        None
    } else {
        let (_, graph) = fixture_workspace_graph(fixture)?;
        let ledger = crate::checks::CheckEvidenceLedger::from_events(events);
        Some(required_check_ids.iter().all(|check_id| {
            crate::checks::check_evidence_is_current(workspace, &graph, &ledger, check_id)
                .unwrap_or(false)
        }))
    };
    let mut observed_paths = BTreeMap::new();
    for relative in &fixture.observe_paths {
        let path = fixture_path(workspace, relative)?;
        observed_paths.insert(relative.clone(), path.exists());
    }
    let runtime_diagnostic = matches!(
        outcome.termination_reason.as_str(),
        "engine_error" | "resource_limit" | "context_limit"
    )
    .then(|| {
        events.iter().rev().find_map(|event| match event {
            AgentEvent::Error { message, .. } => Some(message.clone()),
            _ => None,
        })
    })
    .flatten();

    Ok(ControlFixtureResult {
        id: fixture.id.clone(),
        strict_goal: false,
        goal_assertion_passed: None,
        goal_stage: None,
        goal_outcome: None,
        goal_completion_basis: None,
        goal_checkpoint_sha256: None,
        goal_plan_sha256: None,
        goal_completed_milestones: 0,
        goal_total_milestones: 0,
        goal_workflows: 0,
        goal_model_invocations: 0,
        goal_generated_tokens: 0,
        strict_workflow: false,
        workflow_assertion_passed: None,
        workflow_outcome: None,
        workflow_stage_sequence: Vec::new(),
        workflow_plan_sha256: None,
        workflow_plan_review_sha256: None,
        workflow_code_review_sha256: None,
        workflow_plan_cycles: 0,
        workflow_repair_cycles: 0,
        workflow_advisory_calls: 0,
        workflow_rejected_actions: 0,
        workflow_evidence_invalidations: 0,
        reached_final: outcome.reached_final,
        contract_status: outcome.contract_status,
        verified_completed: outcome.verified_completed,
        termination_reason: outcome.termination_reason.clone(),
        valid_actions: llm_invocations.saturating_sub(invalid_actions),
        llm_invocations,
        tool_calls: events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolCall { .. }))
            .count(),
        corrections,
        gate_corrections,
        errors,
        blocked_tool_loops,
        final_events,
        executed_checks,
        model_run_check_calls,
        reused_checks,
        failed_checks,
        skipped_checks,
        selected_checks: selected_checks.into_iter().collect(),
        affected_components: affected_components.into_iter().collect(),
        executor_starts: executor_starts.into_iter().collect(),
        avoided_executor_starts,
        team_messages,
        repair_turns,
        no_change: handoff_outcome == Some(crate::events::HandoffOutcome::NoChange),
        handoff_outcome,
        commit_disposition: commit.as_ref().map(|(disposition, _)| disposition.clone()),
        commit_oid: record_commit_oid
            .then(|| commit.and_then(|(_, oid)| oid))
            .flatten(),
        output_fingerprints,
        false_completion: outcome.reached_final
            && outcome.contract_status != ContractStatus::Unsatisfied
            && !fixture.completion_supported,
        named_check_compliance,
        recovery_loop: matches!(
            outcome.termination_reason.as_str(),
            "gate_loop" | "parse_loop"
        ),
        latency_ms,
        prompt_tokens,
        generated_tokens,
        tool_trace,
        context,
        energy_kwh: (energy_kwh > 0.0).then_some(energy_kwh),
        remaining_completions: outcome.remaining_completions,
        observed_paths,
        artifact_quality: None,
        runtime_diagnostic,
    })
}

fn summarize_tool_trace(events: &[AgentEvent]) -> Vec<HarnessEvalToolCall> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCall {
                tool, arguments, ..
            } => {
                let encoded = serde_json::to_string(arguments)
                    .unwrap_or_else(|_| "<arguments could not be encoded>".to_string());
                let preview = encoded
                    .chars()
                    .take(MAX_TOOL_TRACE_ARGUMENT_CHARS)
                    .collect::<String>();
                Some(HarnessEvalToolCall {
                    tool: tool.chars().take(MAX_TOOL_TRACE_NAME_CHARS).collect(),
                    arguments_sha256: crate::agent_context::normalized_arguments_sha256(arguments),
                    arguments_preview: preview,
                    arguments_truncated: encoded.chars().count() > MAX_TOOL_TRACE_ARGUMENT_CHARS,
                })
            }
            _ => None,
        })
        .collect()
}

fn summarize_context_metrics(events: &[AgentEvent]) -> ContextEvalMetrics {
    let mut summary = ContextEvalMetrics::default();
    for event in events {
        let AgentEvent::LlmInvocation {
            prompt_tokens,
            context: Some(context),
            ..
        } = event
        else {
            continue;
        };
        summary.invocations_observed = summary.invocations_observed.saturating_add(1);
        summary.context_capacity = summary.context_capacity.max(context.context_capacity);
        summary.reserved_generation_tokens_high_water = summary
            .reserved_generation_tokens_high_water
            .max(context.reserved_generation_tokens);
        summary.safety_margin_tokens_high_water = summary
            .safety_margin_tokens_high_water
            .max(context.safety_margin_tokens);
        summary.usable_prompt_capacity_low_water = if summary.usable_prompt_capacity_low_water == 0
        {
            context.usable_prompt_capacity
        } else {
            summary
                .usable_prompt_capacity_low_water
                .min(context.usable_prompt_capacity)
        };
        summary.prompt_tokens_high_water = summary.prompt_tokens_high_water.max(*prompt_tokens);
        summary.preflight_prompt_tokens_high_water = summary
            .preflight_prompt_tokens_high_water
            .max(context.preflight_prompt_tokens);
        summary.prompt_utilization_bps_high_water = summary
            .prompt_utilization_bps_high_water
            .max(context.prompt_utilization_bps);
        summary.message_chars_high_water =
            summary.message_chars_high_water.max(context.message_chars);
        summary.tool_count_high_water = summary.tool_count_high_water.max(context.tool_count);
        summary.tool_schema_chars_high_water = summary
            .tool_schema_chars_high_water
            .max(context.tool_schema_chars);
        summary.tool_schema_tokens_high_water = match (
            summary.tool_schema_tokens_high_water,
            context.tool_schema_tokens,
        ) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (left, right) => left.or(right),
        };
        match context.thinking_enabled {
            Some(true) => {
                summary.thinking_enabled_invocations =
                    summary.thinking_enabled_invocations.saturating_add(1);
            }
            Some(false) => {
                summary.thinking_disabled_invocations =
                    summary.thinking_disabled_invocations.saturating_add(1);
            }
            None => {}
        }
        match context.retry_reason {
            Some(crate::events::AgentRetryReason::ThinkingOffAfterTruncation) => {
                summary.thinking_off_truncation_retries =
                    summary.thinking_off_truncation_retries.saturating_add(1);
            }
            Some(crate::events::AgentRetryReason::CompactMutationAfterTruncation) => {
                summary.compact_mutation_truncation_retries = summary
                    .compact_mutation_truncation_retries
                    .saturating_add(1);
            }
            Some(crate::events::AgentRetryReason::LargerTokenCapAfterTruncation) => {
                summary.larger_cap_truncation_retries =
                    summary.larger_cap_truncation_retries.saturating_add(1);
            }
            None => {}
        }
        summary.compacted_messages = summary
            .compacted_messages
            .saturating_add(context.compacted_messages);
        summary.omitted_tool_result_chars = summary
            .omitted_tool_result_chars
            .saturating_add(context.omitted_tool_result_chars);
        summary.read_cache_hits = summary
            .read_cache_hits
            .saturating_add(context.read_cache_hits);
        summary.closure_checkpoints = summary
            .closure_checkpoints
            .saturating_add(context.closure_checkpoints);
    }
    summary
}

fn initialize_fixture_workspace(root: &Path, files: &BTreeMap<String, String>) -> Result<()> {
    run_git(root, &["init", "--initial-branch=main"])?;
    run_git(root, &["config", "user.name", "pb harness fixture"])?;
    run_git(root, &["config", "user.email", "fixture@pb.local"])?;
    write_fixture_files(root, files)?;
    run_git(root, &["add", "-A"])?;
    run_git(
        root,
        &["commit", "--allow-empty", "-m", "test: initialize fixture"],
    )?;
    run_git(root, &["checkout", "-b", "harness-eval"])?;
    Ok(())
}

fn write_fixture_files(root: &Path, files: &BTreeMap<String, String>) -> Result<()> {
    for (relative, content) in files {
        let path = fixture_path(root, relative)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        std::fs::write(&path, content)
            .with_context(|| format!("failed to write fixture file {}", path.display()))?;
    }
    Ok(())
}

fn fixture_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || path.starts_with(".git")
    {
        bail!("invalid harness control fixture path '{relative}'");
    }
    Ok(root.join(path))
}

fn run_git(root: &Path, args: &[&str]) -> Result<()> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

fn git_output(root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASELINE: &str = include_str!("../docs/harness-control-baseline.json");

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct BaselineReport {
        fixture_version: u32,
        captured_at: String,
        results: Vec<ControlFixtureResult>,
        observations: Vec<BaselineObservation>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct BaselineObservation {
        id: String,
        classification: String,
        priority: String,
        evidence: String,
    }

    #[test]
    fn control_fixture_corpus_matches_checked_in_baseline() {
        let corpus = control_fixture_corpus().unwrap();
        let actual = run_control_fixture_corpus().unwrap();
        let baseline: BaselineReport = serde_json::from_str(BASELINE).unwrap();

        assert_eq!(baseline.fixture_version, corpus.version);
        assert_eq!(baseline.captured_at, "2026-07-14");
        let fixture_ids = corpus
            .fixtures
            .iter()
            .map(|fixture| fixture.id.as_str())
            .chain(
                corpus
                    .workflow_fixtures
                    .iter()
                    .map(|fixture| fixture.id.as_str()),
            )
            .collect::<std::collections::BTreeSet<_>>();
        let observation_ids = baseline
            .observations
            .iter()
            .map(|observation| observation.id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(observation_ids, fixture_ids);
        for observation in &baseline.observations {
            assert!(
                matches!(
                    observation.classification.as_str(),
                    "pb_defect" | "model_limitation" | "experiment_error" | "positive_evidence"
                ),
                "invalid classification for {}",
                observation.id
            );
            assert!(!observation.priority.trim().is_empty());
            assert!(!observation.evidence.trim().is_empty());
        }
        let protocol_actual = actual
            .iter()
            .filter(|result| !result.strict_goal)
            .cloned()
            .map(|mut result| {
                // The v3 control baseline predates additive S0 context observations. Keep its
                // protocol comparison stable; the dedicated small-model baseline owns prompt
                // measurements. Scripted S1 preflight reports its deterministic rendered count
                // instead of the legacy one-token sentinel.
                result.context = ContextEvalMetrics::default();
                if !result.strict_workflow {
                    result.prompt_tokens = result.llm_invocations;
                }
                result.tool_trace.clear();
                result
            })
            .collect::<Vec<_>>();
        assert_eq!(baseline.results.len(), protocol_actual.len());
        for (expected, observed) in baseline.results.iter().zip(&protocol_actual) {
            assert_eq!(
                expected, observed,
                "control baseline mismatch for {}",
                expected.id
            );
        }
        let goal_results = actual
            .iter()
            .filter(|result| result.strict_goal)
            .collect::<Vec<_>>();
        assert_eq!(goal_results.len(), 7);
        assert!(
            goal_results
                .iter()
                .all(|result| result.goal_assertion_passed == Some(true))
        );
        for id in ["irrelevant_review_evidence", "check_then_mutation"] {
            let result = actual.iter().find(|result| result.id == id).unwrap();
            assert!(result.reached_final, "{id} emitted a final action");
            assert_eq!(result.contract_status, ContractStatus::Unsatisfied, "{id}");
            assert!(!result.verified_completed, "{id} must not be verified");
            assert_eq!(result.termination_reason, "contract_unsatisfied", "{id}");
            assert!(!result.false_completion, "{id} must not falsely complete");
            assert_eq!(result.gate_corrections, 1, "{id}");
        }
        assert_eq!(
            actual
                .iter()
                .find(|result| result.id == "irrelevant_review_evidence")
                .unwrap()
                .executed_checks,
            0,
            "the current gate rejects a missing named check instead of executing it"
        );
        assert_eq!(
            actual
                .iter()
                .find(|result| result.id == "check_then_mutation")
                .unwrap()
                .executed_checks,
            1
        );
        let current = actual
            .iter()
            .find(|result| result.id == "current_named_check")
            .unwrap();
        assert!(current.verified_completed);
        assert_eq!(current.executed_checks, 1);
        assert_eq!(current.named_check_compliance, Some(true));
        let repeated = actual
            .iter()
            .find(|result| result.id == "repeated_blocked_action")
            .unwrap();
        assert!(!repeated.reached_final);
        assert_eq!(repeated.termination_reason, "gate_loop");
        assert_eq!(repeated.llm_invocations, 3);
        assert_eq!(repeated.blocked_tool_loops, 1);
        assert_eq!(repeated.remaining_completions, 1);

        let handoff = actual
            .iter()
            .find(|result| result.id == "handoff_contract_check_commit")
            .unwrap();
        assert_eq!(handoff.model_run_check_calls, 0);
        assert_eq!(handoff.executed_checks, 1);
        assert_eq!(handoff.commit_disposition.as_deref(), Some("created"));
        let required_no_change = actual
            .iter()
            .find(|result| result.id == "handoff_required_mutation_no_change")
            .unwrap();
        assert_eq!(
            required_no_change.termination_reason,
            "contract_unsatisfied"
        );
        assert!(!required_no_change.verified_completed);
        let repaired = actual
            .iter()
            .find(|result| result.id == "handoff_repair_succeeds")
            .unwrap();
        assert_eq!(repaired.executed_checks, 2);
        assert_eq!(repaired.failed_checks, 1);
        assert_eq!(repaired.repair_turns, 1);
        assert_eq!(
            repaired.handoff_outcome,
            Some(crate::events::HandoffOutcome::Ready)
        );
        let resumed = actual
            .iter()
            .find(|result| result.id == "resumed_task_owned_change")
            .unwrap();
        assert_eq!(resumed.affected_components, vec!["api"]);
        assert_eq!(resumed.executed_checks, 1);
        assert_eq!(resumed.commit_disposition.as_deref(), Some("created"));
        let multi = actual
            .iter()
            .find(|result| result.id == "multi_executor_affected_selection")
            .unwrap();
        assert_eq!(multi.executor_starts, vec!["api"]);
        assert_eq!(multi.avoided_executor_starts, vec!["web"]);
        let bundle = actual
            .iter()
            .find(|result| result.id == "generated_bundle_dependency")
            .unwrap();
        assert_eq!(bundle.executed_checks, 2);
        assert_eq!(bundle.output_fingerprints.len(), 1);
    }

    #[test]
    fn control_fixture_paths_cannot_escape_scratch() {
        let root = tempfile::tempdir().unwrap();
        assert!(fixture_path(root.path(), "safe/nested.txt").is_ok());
        assert!(fixture_path(root.path(), "../escape.txt").is_err());
        assert!(fixture_path(root.path(), "/tmp/escape.txt").is_err());
        assert!(fixture_path(root.path(), ".git/config").is_err());
    }

    #[test]
    fn version_one_fixture_and_result_schemas_are_rejected_explicitly() {
        let error =
            parse_control_fixture_corpus(r#"{"version":1,"fixtures":[],"workflow_fixtures":[]}"#)
                .unwrap_err();
        assert!(error.to_string().contains("expected 3"));

        let corpus = control_fixture_corpus().unwrap();
        let mut record = build_eval_records(
            &corpus,
            scripted_configuration(HarnessEvalSuite::Control),
            run_control_fixture_corpus().unwrap(),
        )
        .unwrap()
        .remove(0);
        record.schema_version = 1;
        let error = validate_eval_record_schema(&record).unwrap_err();
        assert!(error.to_string().contains("expected 3"));
    }

    #[test]
    fn scripted_eval_report_is_deterministic_and_protocol_complete() {
        let corpus = control_fixture_corpus().unwrap();
        let build = || {
            build_eval_records(
                &corpus,
                scripted_configuration(HarnessEvalSuite::Control),
                run_control_fixture_corpus().unwrap(),
            )
            .unwrap()
        };
        let first = build();
        let second = build();

        assert_eq!(first, second);
        assert!(first.iter().all(|record| record.protocol_pass));
        let multi = first
            .iter()
            .find(|record| record.result.id == "multi_executor_affected_selection")
            .unwrap();
        assert_eq!(
            multi.configuration.executor_policy,
            vec!["api:local", "web:local"]
        );
        assert_eq!(
            multi
                .configuration
                .workspace_config_sha256
                .as_deref()
                .map(str::len),
            Some(64)
        );
        assert!(
            first
                .iter()
                .all(|record| record.result.artifact_quality.is_none())
        );
        let first_jsonl = first
            .iter()
            .map(|record| serde_json::to_string(record).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let second_jsonl = second
            .iter()
            .map(|record| serde_json::to_string(record).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(first_jsonl, second_jsonl);
        for line in first_jsonl.lines() {
            let record: HarnessEvalRecord = serde_json::from_str(line).unwrap();
            assert_eq!(record.schema_version, 3);
            assert_eq!(record.configuration.mode, "scripted");
        }
    }

    #[test]
    fn small_model_fixture_group_is_stable_and_reports_context_observations() {
        let corpus = control_fixture_corpus().unwrap();
        assert_eq!(
            corpus.small_model_fixtures,
            vec![
                "false_final_after_inspection",
                "repeated_blocked_action",
                "final_at_step_limit",
                "review_missing_check",
            ]
        );
        let fixtures = selected_control_fixtures(&corpus, HarnessEvalSuite::SmallModel);
        let workflows = selected_workflow_fixtures(&corpus, HarnessEvalSuite::SmallModel);
        let goals = selected_goal_fixtures(&corpus, HarnessEvalSuite::SmallModel);
        assert!(workflows.is_empty());
        assert!(goals.is_empty());
        let results = fixtures
            .iter()
            .map(|fixture| run_control_fixture(fixture))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let records = build_selected_eval_records(
            corpus.version,
            &fixtures,
            &workflows,
            &goals,
            scripted_configuration(HarnessEvalSuite::SmallModel),
            results,
        )
        .unwrap();
        assert_eq!(records.len(), corpus.small_model_fixtures.len());
        assert!(records.iter().all(|record| record.protocol_pass));
        for record in records {
            let context = &record.result.context;
            assert_eq!(context.invocations_observed, record.result.llm_invocations);
            assert_eq!(
                context.context_capacity,
                SCRIPTED_EVAL_CONTEXT_SIZE as usize
            );
            assert_eq!(context.reserved_generation_tokens_high_water, 256);
            assert_eq!(context.safety_margin_tokens_high_water, 32);
            assert_eq!(context.usable_prompt_capacity_low_water, 7_904);
            assert!(context.prompt_tokens_high_water > 1);
            assert_eq!(
                context.preflight_prompt_tokens_high_water,
                context.prompt_tokens_high_water
            );
            assert!(context.prompt_utilization_bps_high_water < 1_000);
            assert!(context.message_chars_high_water > 0);
            assert!(context.tool_count_high_water > 0);
            assert!(context.tool_schema_chars_high_water > 0);
            assert_eq!(context.compacted_messages, 0);
            assert_eq!(context.omitted_tool_result_chars, 0);
            assert_eq!(
                context.read_cache_hits,
                usize::from(record.result.id == "repeated_blocked_action")
            );
            assert_eq!(context.closure_checkpoints, 0);
        }
    }

    #[test]
    fn protocol_scoring_ignores_artifact_quality_but_detects_control_regressions() {
        let corpus = control_fixture_corpus().unwrap();
        let fixture = &corpus.fixtures[0];
        let mut result = run_control_fixture(fixture).unwrap();
        result.artifact_quality = Some("intentionally not scored".to_string());
        assert!(protocol_failures(&fixture.expected, &result).is_empty());

        result.termination_reason = "engine_error".to_string();
        let failures = protocol_failures(&fixture.expected, &result);
        assert!(
            failures
                .iter()
                .any(|failure| failure.starts_with("termination_reason"))
        );
    }

    #[test]
    fn eval_table_covers_control_and_resource_metrics() {
        let corpus = control_fixture_corpus().unwrap();
        let records = build_eval_records(
            &corpus,
            scripted_configuration(HarnessEvalSuite::Control),
            run_control_fixture_corpus().unwrap(),
        )
        .unwrap();
        let table = render_eval_table(&records);
        for heading in [
            "handoff",
            "checks",
            "reuse",
            "execs",
            "commit",
            "valid",
            "named",
            "false",
            "loop",
            "turns",
            "ctx_hi",
            "schema_ch",
            "latency_ms",
            "tokens",
            "energy_kwh",
            "termination",
        ] {
            assert!(table.contains(heading), "missing {heading}: {table}");
        }
    }

    #[test]
    fn real_model_tool_trace_hashes_and_bounds_arguments() {
        let arguments = serde_json::json!({"content": "x".repeat(10_000), "path": "result.txt"});
        let trace = summarize_tool_trace(&[AgentEvent::ToolCall {
            tool: "write_file".repeat(20),
            arguments: arguments.clone(),
            nesting_depth: None,
            timestamp_ms: None,
        }]);

        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].tool.chars().count(), MAX_TOOL_TRACE_NAME_CHARS);
        assert_eq!(trace[0].arguments_sha256.len(), 64);
        assert_eq!(
            trace[0].arguments_sha256,
            crate::agent_context::normalized_arguments_sha256(&arguments)
        );
        assert_eq!(
            trace[0].arguments_preview.chars().count(),
            MAX_TOOL_TRACE_ARGUMENT_CHARS
        );
        assert!(trace[0].arguments_truncated);
    }

    #[test]
    fn flashmoe_model_eval_requires_versioned_resource_policy() {
        assert!(ensure_flashmoe_eval_policy("flashmoe", 0).is_err());
        assert!(ensure_flashmoe_eval_policy("llama_cpp", 0).is_ok());
        assert!(
            ensure_flashmoe_eval_policy(
                "flashmoe",
                crate::inference::flashmoe::HARNESS_RESOURCE_POLICY_VERSION,
            )
            .is_ok()
        );
    }

    #[test]
    fn real_model_configuration_serializes_reproduction_parameters() {
        let configuration = HarnessEvalConfiguration {
            mode: "local_model".to_string(),
            backend: "llama_cpp".to_string(),
            suite: "small_model".to_string(),
            model: Some("model.gguf".to_string()),
            model_dir: Some("/models".to_string()),
            max_tokens: 512,
            ctx_size: 32768,
            threads: Some(8),
            threads_batch: Some(12),
            gpu_layers: 99,
            temperature: 0.0,
            top_k: 1,
            seed: 42,
            flashmoe_resource_policy_version: 1,
            workspace_config_sha256: Some("abc".to_string()),
            executor_policy: vec!["app:local".to_string()],
        };
        let value = serde_json::to_value(configuration).unwrap();
        for field in [
            "mode",
            "backend",
            "suite",
            "model",
            "model_dir",
            "max_tokens",
            "ctx_size",
            "threads",
            "threads_batch",
            "gpu_layers",
            "temperature",
            "top_k",
            "seed",
            "flashmoe_resource_policy_version",
            "workspace_config_sha256",
            "executor_policy",
        ] {
            assert!(value.get(field).is_some(), "missing {field}: {value}");
        }
    }
}
