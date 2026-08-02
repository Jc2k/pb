//! Direct, daemon-free harnesses for exercising pb internals.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::agent_core::{
    AgentRequest, AgentRunResult, EventSink, SessionAttachment, run_agent,
    run_agent_with_flashmoe_settings,
};
use crate::cli_ui::render_event;
use crate::config::UserConfig;
use crate::environment::{EnvironmentBackend, EnvironmentConfig, EnvironmentMode};
use crate::events::{AgentEvent, EventEnvelope, EvidenceRef};
use crate::session_store::now_millis;
use crate::{HarnessAgentArgs, HarnessCacheEvalArgs};

const HARNESS_GIT_NAME: &str = "pb harness";
const HARNESS_GIT_EMAIL: &str = "harness@pb.local";
const HARNESS_AGENT_TOOLS: &[&str] = &[
    "session_title",
    "run_command",
    "run_check",
    "read_file",
    "write_file",
    "replace_file",
    "edit_file",
    "apply_patch",
    "rm",
    "git_commit",
    "sub_agent",
    "propose_delivery",
    "start_delivery",
    "propose_goal",
    "start_goal",
    "goal_status",
    "goal_pause",
    "goal_request_amendment",
    "goal_request_budget",
];

#[derive(Debug)]
struct ScratchLayout {
    root: PathBuf,
    workspace: PathBuf,
    events: PathBuf,
    journal: PathBuf,
    run_index: PathBuf,
    run_id: String,
    run_events: PathBuf,
    run_journal: PathBuf,
    run_task_planning_transcript: PathBuf,
    task_baseline: PathBuf,
    adoptions: PathBuf,
    workflow_checkpoint: PathBuf,
    multi_task_checkpoint: PathBuf,
    resumed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
enum RunIndexRecord {
    Started {
        version: u32,
        run_id: String,
        timestamp_ms: u64,
        task: String,
        run_events: String,
        run_journal: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        workspace_config: Option<WorkspaceConfigMetadata>,
        #[serde(default)]
        workflow_config: WorkflowConfigMetadata,
    },
    Finished {
        version: u32,
        run_id: String,
        timestamp_ms: u64,
        status: String,
        reached_final: bool,
        contract_status: crate::events::ContractStatus,
        verified_completed: bool,
        termination_reason: Option<crate::events::TerminationReason>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        handoff_outcome: Option<crate::events::HandoffOutcome>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_baseline_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        invocation_baseline_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        workspace_config: Option<WorkspaceConfigMetadata>,
        #[serde(default)]
        workflow_config: WorkflowConfigMetadata,
        #[serde(default)]
        audit: HarnessRunAudit,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct WorkspaceConfigMetadata {
    source: String,
    sha256: String,
    #[serde(default)]
    executor_policy: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct WorkflowConfigMetadata {
    source: String,
    source_sha256: String,
    policy_sha256: String,
    delivery: crate::workflow::DeliveryPolicy,
    default_intent: crate::workflow::TurnIntent,
    limits: crate::workflow::WorkflowLimits,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct HarnessRunAudit {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    observation_rendering: Option<crate::workflow::ObservationRendering>,
    controller_observations: usize,
    controller_observation_prompt_bytes: usize,
    #[serde(default)]
    controller_observation_coverage: BTreeMap<String, usize>,
    controller_mutations: usize,
    controller_closures: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    goal_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    goal_stage: Option<crate::goal::GoalStage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    goal_plan_sha256: Option<String>,
    goal_completed_milestones: usize,
    goal_total_milestones: usize,
    goal_workflows: usize,
    goal_model_invocations: usize,
    goal_generated_tokens: usize,
    goal_pause_requests: usize,
    goal_amendment_requests: usize,
    goal_budget_requests: usize,
    strict_workflow: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    workflow_id: Option<String>,
    #[serde(default)]
    workflow_stage_sequence: Vec<crate::workflow::WorkflowStage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    workflow_outcome: Option<crate::workflow::WorkflowOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    workflow_checkpoint_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ready_evidence_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    repository_remote: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    plan_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    plan_review_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    code_review_sha256: Option<String>,
    plan_cycles: usize,
    repair_cycles: usize,
    workflow_model_invocations: usize,
    workflow_generated_tokens: usize,
    workflow_advisory_calls: usize,
    #[serde(default)]
    workflow_stage_steps: BTreeMap<crate::workflow::WorkflowStage, usize>,
    rejected_workflow_actions: usize,
    evidence_invalidations: usize,
    strict_workflow_satisfied: bool,
    #[serde(default)]
    affected_components: Vec<String>,
    #[serde(default)]
    checks_planned: Vec<String>,
    checks_reused: usize,
    checks_executed: usize,
    checks_passed: usize,
    checks_failed: usize,
    checks_skipped: usize,
    #[serde(default)]
    check_evidence_ids: Vec<String>,
    #[serde(default)]
    output_fingerprints: Vec<String>,
    #[serde(default)]
    executors_started: Vec<String>,
    #[serde(default)]
    executors_failed: Vec<String>,
    repair_turns: usize,
    team_messages: usize,
    #[serde(default)]
    feedback_evidence_ids: Vec<String>,
    no_change: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    commit_disposition: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    commit_oid: Option<String>,
    #[serde(default)]
    commit_changed_paths: Vec<String>,
}

#[derive(Debug, Clone)]
struct Observation {
    rank: u8,
    classification: &'static str,
    title: String,
    detail: String,
}

#[derive(Debug, Clone, Default)]
struct CapturedSummary {
    branch: String,
    commits: String,
    summary: String,
    diff_stat: String,
}

struct JournalState {
    cumulative_writer: BufWriter<File>,
    run_writer: BufWriter<File>,
    events: Vec<EventEnvelope>,
    observations: Vec<Observation>,
    summary: CapturedSummary,
    audit: HarnessRunAudit,
    write_error: Option<String>,
    terminal_precursor_keys: Vec<String>,
    workflow_checkpoint: PathBuf,
    multi_task_checkpoint: PathBuf,
}

#[derive(Clone)]
struct HarnessEventSink {
    state: Arc<Mutex<JournalState>>,
}

impl HarnessEventSink {
    fn new(cumulative_path: &Path, run_path: &Path, workflow_checkpoint: &Path) -> Result<Self> {
        let cumulative_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(cumulative_path)
            .with_context(|| {
                format!(
                    "failed to open cumulative harness event journal {}",
                    cumulative_path.display()
                )
            })?;
        let run_file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(run_path)
            .with_context(|| {
                format!(
                    "failed to create per-run harness event journal {}",
                    run_path.display()
                )
            })?;
        Ok(Self {
            state: Arc::new(Mutex::new(JournalState {
                cumulative_writer: BufWriter::new(cumulative_file),
                run_writer: BufWriter::new(run_file),
                events: Vec::new(),
                observations: Vec::new(),
                summary: CapturedSummary::default(),
                audit: HarnessRunAudit::default(),
                write_error: None,
                terminal_precursor_keys: Vec::new(),
                workflow_checkpoint: workflow_checkpoint.to_path_buf(),
                multi_task_checkpoint: workflow_checkpoint
                    .with_file_name("multi-task-checkpoint.json"),
            })),
        })
    }

    fn snapshot(&self) -> Result<(Vec<Observation>, CapturedSummary, HarnessRunAudit)> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("harness event journal lock was poisoned"))?;
        let mut flush_errors = Vec::new();
        if let Err(error) = state.cumulative_writer.flush() {
            flush_errors.push(format!("cumulative stream: {error}"));
        }
        if let Err(error) = state.run_writer.flush() {
            flush_errors.push(format!("per-run stream: {error}"));
        }
        if !flush_errors.is_empty() {
            bail!(
                "failed to flush harness event journals: {}",
                flush_errors.join("; ")
            );
        }
        if let Some(error) = state.write_error.as_deref() {
            bail!("failed to write harness event journal: {error}");
        }
        Ok((
            state.observations.clone(),
            state.summary.clone(),
            state.audit.clone(),
        ))
    }

    fn configure_goal_context(&self, goal: &crate::goal::GoalModelBrief) -> Result<()> {
        goal.validate()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("harness event journal lock was poisoned"))?;
        state.audit.goal_id = Some(goal.id.clone());
        state.audit.goal_stage = Some(goal.stage);
        state.audit.goal_plan_sha256 = Some(goal.plan_sha256.clone());
        state.audit.goal_completed_milestones = goal.completed_milestones;
        state.audit.goal_total_milestones = goal.total_milestones;
        state.audit.goal_workflows = goal.counters.workflows;
        state.audit.goal_model_invocations = goal.counters.model_invocations;
        state.audit.goal_generated_tokens = goal.counters.generated_tokens;
        Ok(())
    }
}

impl EventSink for HarnessEventSink {
    fn emit(&mut self, event: AgentEvent) {
        self.emit_superseding(event, Vec::new());
    }

    fn emit_superseding(&mut self, event: AgentEvent, supersedes: Vec<String>) {
        let Ok(mut state) = self.state.lock() else {
            eprintln!("pb harness: event journal lock was poisoned");
            return;
        };
        let mut envelope = EventEnvelope::new_superseding(event, supersedes);
        let sequence = state
            .events
            .last()
            .map_or(1, |entry| entry.transcript.sequence.saturating_add(1));
        envelope.assign_sequence(sequence);
        envelope.refresh_projections(&state.events);
        render_event(&envelope);

        match &envelope.event {
            AgentEvent::Started { branch, .. } => {
                state.summary.branch = branch.clone();
            }
            AgentEvent::HarnessExperimentConfigured {
                observation_rendering,
                ..
            } => {
                state.audit.observation_rendering = Some(*observation_rendering);
            }
            AgentEvent::ControllerObservation { receipt, .. } => {
                state.audit.controller_observations += 1;
                state.audit.controller_observation_prompt_bytes = state
                    .audit
                    .controller_observation_prompt_bytes
                    .saturating_add(receipt.prompt_bytes);
                *state
                    .audit
                    .controller_observation_coverage
                    .entry(receipt.coverage.as_str().to_string())
                    .or_default() += 1;
            }
            AgentEvent::ControllerMutation { .. } => {
                state.audit.controller_mutations += 1;
            }
            AgentEvent::ControllerClosure { .. } => {
                state.audit.controller_closures += 1;
            }
            AgentEvent::Error {
                detail, summary, ..
            } => {
                let bounded_stop = matches!(
                    summary.as_str(),
                    "Step limit reached"
                        | "No-progress tool loop"
                        | "Repeated tool call loop"
                        | "Contract unsatisfied"
                        | "Parse retry limit reached"
                ) || summary.starts_with("Invalid pb JSON action")
                    || summary.starts_with("Invalid structured workflow action");
                state.observations.push(Observation {
                    rank: 2,
                    classification: if bounded_stop {
                        "model_limitation"
                    } else {
                        "experiment_error"
                    },
                    title: nonempty_or(summary, "agent error"),
                    detail: compact_detail(detail),
                });
            }
            AgentEvent::Correction {
                message, summary, ..
            } => {
                state.observations.push(Observation {
                    rank: 3,
                    classification: correction_classification(message, summary),
                    title: nonempty_or(summary, "agent correction"),
                    detail: compact_detail(message),
                });
                if summary.contains("handoff teammate returned failed checks") {
                    state.audit.repair_turns += 1;
                }
                if matches!(
                    summary.as_str(),
                    "Workflow stage submission required"
                        | "Workflow artifact validation failed"
                        | "Tool not available"
                ) {
                    state.audit.rejected_workflow_actions += 1;
                }
            }
            AgentEvent::WorkflowStarted { workflow_id, .. } => {
                state.audit.strict_workflow = true;
                state.audit.workflow_id = Some(workflow_id.clone());
            }
            AgentEvent::GoalPauseRequested { .. } => {
                state.audit.goal_pause_requests += 1;
            }
            AgentEvent::GoalChangeRequested { kind, .. } => match kind {
                crate::events::GoalChangeKind::Amendment => {
                    state.audit.goal_amendment_requests += 1;
                }
                crate::events::GoalChangeKind::Budget => {
                    state.audit.goal_budget_requests += 1;
                }
            },
            AgentEvent::WorkflowStageStarted { stage, .. } => {
                state.audit.workflow_stage_sequence.push(*stage);
            }
            AgentEvent::WorkflowArtifactAccepted {
                artifact_kind,
                sha256,
                ..
            } => match artifact_kind.as_str() {
                "plan" => state.audit.plan_sha256 = Some(sha256.clone()),
                "plan_review" => state.audit.plan_review_sha256 = Some(sha256.clone()),
                "code_review" => state.audit.code_review_sha256 = Some(sha256.clone()),
                _ => {}
            },
            AgentEvent::WorkflowEvidenceInvalidated { .. } => {
                state.audit.evidence_invalidations += 1;
            }
            AgentEvent::WorkflowCompleted {
                workflow_id,
                outcome,
                checkpoint_sha256,
                ready_evidence_sha256,
                ..
            } => {
                state.audit.strict_workflow = true;
                state.audit.workflow_id = Some(workflow_id.clone());
                state.audit.workflow_outcome = Some(*outcome);
                state.audit.workflow_checkpoint_sha256 = Some(checkpoint_sha256.clone());
                state.audit.ready_evidence_sha256 = ready_evidence_sha256.clone();
                state.audit.strict_workflow_satisfied = match outcome {
                    crate::workflow::WorkflowOutcome::Ready => ready_evidence_sha256.is_some(),
                    crate::workflow::WorkflowOutcome::NoChange => true,
                    _ => false,
                };
            }
            AgentEvent::SessionSummary {
                branch,
                commits,
                summary,
                diff_stat,
                ..
            } => {
                state.summary = CapturedSummary {
                    branch: branch.clone(),
                    commits: commits
                        .iter()
                        .map(|commit| format!("{} {}", commit.oid, commit.subject))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    summary: summary.clone(),
                    diff_stat: diff_stat.clone(),
                };
            }
            AgentEvent::ExecutorStarted {
                executor_id,
                success,
                ..
            } => {
                let target = if *success {
                    &mut state.audit.executors_started
                } else {
                    &mut state.audit.executors_failed
                };
                if !target.contains(executor_id) {
                    target.push(executor_id.clone());
                }
            }
            AgentEvent::CheckResult {
                check_id,
                success,
                reused,
                skip_reason,
                output_fingerprint,
                ..
            } => {
                if !state.audit.checks_planned.contains(check_id) {
                    state.audit.checks_planned.push(check_id.clone());
                }
                if *reused {
                    state.audit.checks_reused += 1;
                } else if skip_reason.is_some() {
                    state.audit.checks_skipped += 1;
                } else {
                    state.audit.checks_executed += 1;
                }
                if *success {
                    state.audit.checks_passed += 1;
                } else if skip_reason.is_none() {
                    state.audit.checks_failed += 1;
                }
                let evidence_id = format!("check:{check_id}");
                if !state.audit.check_evidence_ids.contains(&evidence_id) {
                    state.audit.check_evidence_ids.push(evidence_id);
                }
                if let Some(fingerprint) = output_fingerprint
                    && !state.audit.output_fingerprints.contains(fingerprint)
                {
                    state.audit.output_fingerprints.push(fingerprint.clone());
                }
            }
            AgentEvent::TeamMessage { evidence, .. } => {
                state.audit.team_messages += 1;
                for reference in evidence {
                    let evidence_id = match reference {
                        EvidenceRef::Check { check_id } => format!("check:{check_id}"),
                        EvidenceRef::Commit { oid } => format!("commit:{oid}"),
                    };
                    if !state.audit.feedback_evidence_ids.contains(&evidence_id) {
                        state.audit.feedback_evidence_ids.push(evidence_id);
                    }
                }
            }
            AgentEvent::HandoffSummary { summary, .. } => {
                state.audit.affected_components = summary.affected_components.clone();
                state.audit.checks_planned = summary
                    .checks
                    .iter()
                    .map(|check| check.check_id.clone())
                    .collect();
                state.audit.no_change = summary.outcome == crate::events::HandoffOutcome::NoChange;
            }
            AgentEvent::CommitResult {
                success,
                created,
                reused,
                oid,
                changed_paths,
                ..
            } => {
                state.audit.commit_disposition = Some(
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
                );
                state.audit.commit_oid = oid.clone();
                state.audit.commit_changed_paths = changed_paths.clone();
            }
            _ => {}
        }

        if state.write_error.is_some() {
            return;
        }
        let encoded = match serde_json::to_vec(&envelope) {
            Ok(encoded) => encoded,
            Err(error) => {
                state.write_error = Some(format!("event serialization: {error}"));
                return;
            }
        };
        let mut write_errors = Vec::new();
        if let Err(error) = write_event_line(&mut state.cumulative_writer, &encoded) {
            write_errors.push(format!("cumulative stream: {error}"));
        }
        if let Err(error) = write_event_line(&mut state.run_writer, &encoded) {
            write_errors.push(format!("per-run stream: {error}"));
        }
        if !write_errors.is_empty() {
            state.write_error = Some(write_errors.join("; "));
        }
        state.events.push(envelope);
    }

    fn emit_terminal_precursor(&mut self, event: AgentEvent) {
        let entry_key = crate::events::event_entry_key(&event);
        self.emit(event);
        if let Ok(mut state) = self.state.lock() {
            state.terminal_precursor_keys.push(entry_key);
        }
    }

    fn take_terminal_precursor_keys(&mut self) -> Vec<String> {
        self.state
            .lock()
            .map(|mut state| std::mem::take(&mut state.terminal_precursor_keys))
            .unwrap_or_default()
    }

    fn checkpoint_workflow(
        &mut self,
        checkpoint: &crate::workflow::WorkflowCheckpoint,
    ) -> Result<()> {
        checkpoint.validate()?;
        let path = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("harness event journal lock was poisoned"))?;
            let run = &checkpoint.run;
            state.audit.strict_workflow = true;
            state.audit.workflow_id = Some(run.id.clone());
            state.audit.workflow_outcome = run.outcome;
            state.audit.workflow_checkpoint_sha256 = Some(checkpoint.sha256.clone());
            state.audit.ready_evidence_sha256 = run
                .ready_evidence
                .as_ref()
                .map(crate::workflow::ReadyEvidenceBundle::sha256)
                .transpose()?;
            state.audit.repository_remote = run
                .ready_evidence
                .as_ref()
                .and_then(|evidence| evidence.repository_remote.clone());
            state.audit.plan_sha256 = run.plan.as_ref().map(|plan| plan.sha256.clone());
            state.audit.plan_review_sha256 =
                run.plan_review.as_ref().map(|review| review.sha256.clone());
            state.audit.code_review_sha256 =
                run.code_review.as_ref().map(|review| review.sha256.clone());
            state.audit.plan_cycles = run.counters.plan_cycles;
            state.audit.repair_cycles = run.counters.repair_cycles;
            state.audit.workflow_model_invocations = run.counters.model_invocations;
            state.audit.workflow_generated_tokens = run.counters.generated_tokens;
            state.audit.workflow_advisory_calls = run.counters.advisory_calls;
            state.audit.workflow_stage_steps = run.counters.stage_steps.clone();
            state.audit.strict_workflow_satisfied = matches!(
                run.outcome,
                Some(
                    crate::workflow::WorkflowOutcome::Ready
                        | crate::workflow::WorkflowOutcome::NoChange
                )
            );
            state.workflow_checkpoint.clone()
        };
        let bytes = serde_json::to_vec_pretty(checkpoint)
            .context("failed to serialize harness workflow checkpoint")?;
        atomic_write(&path, &bytes)
    }

    fn checkpoint_multi_task(
        &mut self,
        checkpoint: &crate::task_queue::MultiTaskCheckpoint,
    ) -> Result<()> {
        checkpoint.validate()?;
        let path = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("harness event journal lock was poisoned"))?
            .multi_task_checkpoint
            .clone();
        let bytes = serde_json::to_vec_pretty(checkpoint)
            .context("failed to serialize harness multi-Task checkpoint")?;
        atomic_write(&path, &bytes)
    }

    fn request_goal_pause(&mut self, reason: &str) -> Result<String> {
        let goal_id = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("harness event journal lock was poisoned"))?
            .audit
            .goal_id
            .clone()
            .context("goal pause requires a configured harness Goal context")?;
        self.emit(AgentEvent::GoalPauseRequested {
            goal_id: goal_id.clone(),
            timestamp_ms: Some(now_millis()),
        });
        Ok(format!(
            "goal pause request recorded for {goal_id}: {}",
            compact_detail(reason)
        ))
    }

    fn request_goal_change(
        &mut self,
        kind: crate::events::GoalChangeKind,
        summary: &str,
    ) -> Result<String> {
        let goal_id = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("harness event journal lock was poisoned"))?
            .audit
            .goal_id
            .clone()
            .context("goal change requires a configured harness Goal context")?;
        self.emit(AgentEvent::GoalChangeRequested {
            goal_id: goal_id.clone(),
            kind,
            summary: compact_detail(summary),
            timestamp_ms: Some(now_millis()),
        });
        Ok(format!("goal {kind} request recorded for {goal_id}"))
    }
}

fn write_event_line(writer: &mut BufWriter<File>, encoded: &[u8]) -> std::io::Result<()> {
    writer.write_all(encoded)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn load_goal_context(path: &Path) -> Result<crate::goal::GoalModelBrief> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read harness Goal context {}", path.display()))?;
    let goal: crate::goal::GoalModelBrief = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse harness Goal context {}", path.display()))?;
    goal.validate()
        .with_context(|| format!("invalid harness Goal context {}", path.display()))?;
    Ok(goal)
}

fn resolve_harness_workspace_graph(
    workspace: &Path,
    trusted_workspace_graph: Option<crate::workspace::WorkspaceGraph>,
    contract: Option<&crate::harness_contract::AgentContract>,
) -> Result<crate::workspace::WorkspaceGraph> {
    let base = match trusted_workspace_graph {
        Some(graph) => graph,
        None => crate::workspace::WorkspaceGraph::load_or_discover(workspace, None)?,
    };
    contract
        .map(|contract| contract.compile_workspace_graph(base.clone()))
        .transpose()
        .map(|compiled| compiled.unwrap_or(base))
}

pub fn run_agent_task(args: HarnessAgentArgs) -> Result<()> {
    if args.task.trim().is_empty() {
        bail!("harness agent task must not be empty");
    }

    let goal_context = args
        .goal_context
        .as_deref()
        .map(load_goal_context)
        .transpose()?;

    let contract = args
        .contract
        .as_deref()
        .map(crate::harness_contract::HarnessContractDocument::from_path)
        .transpose()?
        .map(crate::harness_contract::HarnessContractDocument::normalize)
        .transpose()?;
    let trusted_workspace_graph = args
        .workspace_config
        .as_deref()
        .map(crate::workspace::WorkspaceConfigDocument::from_path)
        .transpose()?
        .map(crate::workspace::WorkspaceConfigDocument::normalize)
        .transpose()?;
    let workspace_config_metadata = args
        .workspace_config
        .as_deref()
        .zip(trusted_workspace_graph.as_ref())
        .map(|(path, graph)| workspace_config_metadata(path, graph))
        .transpose()?;
    let workflow_document = args
        .workflow_config
        .as_deref()
        .map(crate::workflow::WorkflowConfigDocument::from_path)
        .transpose()?
        .unwrap_or_default();
    let workflow_policy = workflow_document.clone().compile()?;
    let workflow_config_metadata = workflow_config_metadata(
        args.workflow_config.as_deref(),
        &workflow_document,
        &workflow_policy,
    )?;
    let layout = prepare_scratch(args.scratch_dir.as_deref())?;
    println!("pb harness: scratch={}", layout.root.display());
    println!("pb harness: workspace={}", layout.workspace.display());
    println!("pb harness: events={}", layout.events.display());
    println!("pb harness: journal={}", layout.journal.display());
    println!("pb harness: run_id={}", layout.run_id);
    println!("pb harness: run_events={}", layout.run_events.display());
    println!("pb harness: run_journal={}", layout.run_journal.display());
    println!(
        "pb harness: task_planning_transcript={}",
        layout.run_task_planning_transcript.display()
    );
    println!(
        "pb harness: multi_task_checkpoint={}",
        layout.multi_task_checkpoint.display()
    );
    println!("pb harness: resumed={}", layout.resumed);
    println!("pb harness: deterministic_actions=controller_block");

    let user_config = UserConfig::load()?;
    let model_dir = args
        .model_dir
        .clone()
        .or_else(|| user_config.effective_model_dir());
    let models_root = model_dir.clone().unwrap_or_else(crate::default_models_dir);
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| user_config.effective_model());
    let flashmoe_settings = args
        .cache_dir
        .as_ref()
        .map(|cache_dir| {
            if !cache_dir.is_absolute() {
                bail!(
                    "harness inference-cache root must be absolute: {}",
                    cache_dir.display()
                );
            }
            if crate::inference::flashmoe::plan(&model, &models_root).is_none() {
                bail!("--cache-dir is supported only by the FlashMoe harness backend");
            }
            let mut settings = user_config.effective_flashmoe();
            settings.session_cache.enabled = true;
            settings.session_cache.root = Some(cache_dir.clone());
            Ok(settings)
        })
        .transpose()?;
    let turn_max_tokens_cap = args.max_tokens;
    let repository_context = harness_repository_context(&layout)?;
    let prior_check_evidence = load_check_evidence(&layout.events)?;
    let workspace_graph = resolve_harness_workspace_graph(
        &layout.workspace,
        trusted_workspace_graph,
        contract.as_ref(),
    )?;
    let resumed_workflow = load_resumable_workflow_checkpoint(&layout.workflow_checkpoint)?;
    if let Some(checkpoint) = resumed_workflow.as_ref()
        && checkpoint.run.task != args.task
    {
        bail!(
            "active harness workflow task differs from --task; finish or cancel '{}' before starting '{}'",
            checkpoint.run.task,
            args.task
        );
    }
    let resumed_branch = resumable_workflow_branch(&layout.workspace, resumed_workflow.as_ref())?;
    let request = AgentRequest {
        task: args.task.clone(),
        turn_id: resumed_workflow
            .as_ref()
            .map(|checkpoint| checkpoint.run.source_turn_id.clone())
            .unwrap_or_else(|| format!("harness-turn-{}", layout.run_id)),
        intent: Some(args.intent),
        task_planning: crate::agent_core::TaskPlanningPreference::Auto,
        task_plan_rejected: None,
        task_planning_transcript: None,
        workflow_policy: Some(workflow_policy),
        workflow_stage: None,
        workflow_expected_content_fingerprint: None,
        workflow_plan_identity: None,
        workflow_plan_paths: Vec::new(),
        workflow_plan_revision_challenge_ids: Vec::new(),
        workflow_action_first_turn: false,
        workflow_creation_path_order: Vec::new(),
        workflow_work_units: None,
        workflow_repair_feedback: None,
        workflow_stage_evidence: None,
        workflow_checkpoint: resumed_workflow,
        conversation_handoff: None,
        legacy_prompt_owned_delivery: false,
        model,
        model_dir,
        workdir: Some(layout.workspace.clone()),
        branch: resumed_branch,
        max_steps: args
            .max_steps
            .unwrap_or_else(|| user_config.effective_max_steps()),
        max_tokens: args
            .max_tokens
            .unwrap_or_else(|| user_config.effective_max_tokens()),
        turn_max_tokens_cap,
        tool_allowlist: Some(harness_tool_allowlist(&args.exclude_tools)?),
        workflow_tool_exclusions: args.exclude_tools.clone(),
        observation_rendering: crate::workflow::ObservationRendering::ControllerBlock,
        accept_existing_workspace_changes: layout.resumed,
        ctx_size: args
            .ctx_size
            .unwrap_or_else(|| user_config.effective_ctx_size()),
        threads: args.threads.or_else(|| user_config.effective_threads()),
        threads_batch: args
            .threads_batch
            .or_else(|| user_config.effective_threads_batch()),
        gpu_layers: args
            .gpu_layers
            .unwrap_or_else(|| user_config.effective_gpu_layers()),
        temperature: args
            .temperature
            .unwrap_or_else(|| user_config.effective_temperature()),
        profile: args.profile,
        infer_profile: false,
        sub_agent_depth: 0,
        repository_less: false,
        top_k: args.top_k.unwrap_or_else(|| user_config.effective_top_k()),
        seed: args.seed.unwrap_or_else(|| user_config.effective_seed()),
        environment: Some(harness_environment()),
        python_dependency_authority: Default::default(),
        environment_evidence_context: None,
        workspace_graph: Some(workspace_graph),
        repository_context: Some(repository_context),
        prior_check_evidence,
        session_id: args
            .session_id
            .clone()
            .unwrap_or_else(|| format!("harness-{}", layout.run_id)),
        attachments: harness_attachments(&args.images)?,
        goal_context: goal_context.clone(),
        contract,
    };

    write_running_journal(
        &layout,
        &args.task,
        workspace_config_metadata.as_ref(),
        &workflow_config_metadata,
    )?;
    append_run_index_started(
        &layout,
        &args.task,
        workspace_config_metadata.as_ref(),
        &workflow_config_metadata,
    )?;
    let mut sink = HarnessEventSink::new(
        &layout.events,
        &layout.run_events,
        &layout.workflow_checkpoint,
    )?;
    sink.emit(AgentEvent::HarnessExperimentConfigured {
        observation_rendering: crate::workflow::ObservationRendering::ControllerBlock,
        timestamp_ms: Some(now_millis()),
    });
    if let Some(goal) = goal_context.as_ref() {
        sink.configure_goal_context(goal)?;
    }
    let run_result = match flashmoe_settings {
        Some(settings) => {
            run_agent_with_flashmoe_settings(request, &models_root, sink.clone(), settings)
        }
        None => run_agent(request, &models_root, sink.clone()),
    };
    if let Ok(result) = &run_result
        && let Some(transcript) = &result.task_planning_transcript
    {
        atomic_write(
            &layout.run_task_planning_transcript,
            &serde_json::to_vec_pretty(transcript)?,
        )?;
    }
    let (mut observations, summary, audit) = sink.snapshot()?;
    add_run_observations(&mut observations, &run_result, &layout.workspace, &summary);
    write_journal(
        &layout,
        &args.task,
        &run_result,
        &summary,
        &audit,
        workspace_config_metadata.as_ref(),
        &workflow_config_metadata,
        &mut observations,
    )?;
    append_run_index_finished(
        &layout,
        &run_result,
        &audit,
        workspace_config_metadata.as_ref(),
        &workflow_config_metadata,
    )?;

    match run_result {
        Ok(result) if harness_outcome_succeeded(&result) => {
            println!(
                "pb harness: reached_final={} handoff_outcome={} contract_status={} verified_completed={} termination_reason={} branch={} workspace={} journal={}",
                result.reached_final,
                result
                    .handoff_outcome
                    .map(|outcome| format!("{outcome:?}").to_ascii_lowercase())
                    .unwrap_or_else(|| "none".to_string()),
                result.contract_status,
                result.verified_completed,
                result.termination_reason,
                result.branch,
                result.workspace_root.display(),
                layout.journal.display()
            );
            Ok(())
        }
        Ok(result)
            if result.termination_reason
                == crate::events::TerminationReason::ContractUnsatisfied =>
        {
            bail!(
                "harness agent final did not satisfy its acceptance contract; reached_final={} contract_status={} verified_completed={} termination_reason={} workspace={} journal={}",
                result.reached_final,
                result.contract_status,
                result.verified_completed,
                result.termination_reason,
                result.workspace_root.display(),
                layout.journal.display()
            )
        }
        Ok(result) => bail!(
            "harness agent did not complete; reached_final={} contract_status={} verified_completed={} termination_reason={} workspace={} journal={}",
            result.reached_final,
            result.contract_status,
            result.verified_completed,
            result.termination_reason,
            result.workspace_root.display(),
            layout.journal.display()
        ),
        Err(error) => Err(error).with_context(|| {
            format!(
                "harness agent failed; workspace={} journal={}",
                layout.workspace.display(),
                layout.journal.display()
            )
        }),
    }
}

const CACHE_EVAL_ARM_IDS: [&str; 6] = [
    "cold_empty_storage",
    "warm_same_session_same_process",
    "warm_new_session_same_process",
    "changed_authority_same_process",
    "matching_authority_same_process",
    "persisted_root_new_process",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CacheEvalRootReport {
    backend: String,
    cache_format_version: String,
    model_namespace_sha256: String,
    source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    miss_reason: Option<crate::inference::PromptCacheMissReason>,
    rendered_token_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_schema_sha256: Option<String>,
    tokens: usize,
    reused_tokens: usize,
    workflow_stage: Option<crate::workflow::WorkflowStage>,
    authority_class: crate::inference::StageRootAuthorityClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    output_constraint_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CacheEvalArmReport {
    id: String,
    process: String,
    session_id: String,
    scratch_dir: String,
    agent_outcome_succeeded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    agent_error: Option<String>,
    invocations: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    planning_root: Option<CacheEvalRootReport>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct CacheEvalGates {
    all_planning_roots_observed: bool,
    cold_root_did_not_reuse: bool,
    same_session_reused_exact_root: bool,
    new_session_reused_exact_root: bool,
    changed_authority_rejected_old_root: bool,
    matching_authority_remained_reusable: bool,
    restart_restored_persisted_exact_root: bool,
}

impl CacheEvalGates {
    fn passed(&self) -> bool {
        self.all_planning_roots_observed
            && self.cold_root_did_not_reuse
            && self.same_session_reused_exact_root
            && self.new_session_reused_exact_root
            && self.changed_authority_rejected_old_root
            && self.matching_authority_remained_reusable
            && self.restart_restored_persisted_exact_root
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct CacheEvalReport {
    version: u32,
    started_at_ms: u64,
    finished_at_ms: u64,
    binary_sha256: String,
    model: String,
    model_dir: String,
    profile: crate::agent_core::AgentProfile,
    changed_authority_tool: String,
    contract_sha256: String,
    workspace_fingerprint: String,
    sampling: CacheEvalSamplingReport,
    cache_dir: String,
    report_path: String,
    released_parent_runtimes_before_restart: usize,
    arms: Vec<CacheEvalArmReport>,
    gates: CacheEvalGates,
    passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct CacheEvalSamplingReport {
    max_steps: usize,
    max_tokens: i32,
    ctx_size: u32,
    threads: Option<i32>,
    threads_batch: Option<i32>,
    gpu_layers: u32,
    temperature: f32,
    top_k: i32,
    seed: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheEvalInputIdentity {
    contract_sha256: String,
    workspace_fingerprint: String,
}

/// Run a durable cache-lifecycle qualification using five arms in this process and one arm in a
/// fresh child process. Agent artifact quality is recorded, but only exact prompt-root evidence
/// determines the cache result.
pub fn run_cache_eval(mut args: HarnessCacheEvalArgs) -> Result<()> {
    let started_at_ms = now_millis();
    resolve_cache_eval_args(&mut args)?;
    let input_identity = validate_cache_eval_args(&args)?;

    let baseline_excludes = vec!["sub_agent".to_string()];
    let changed_excludes = vec!["sub_agent".to_string(), args.changed_authority_tool.clone()];
    harness_tool_allowlist(&baseline_excludes)?;
    harness_tool_allowlist(&changed_excludes)?;

    let shared_session = "pb-cache-eval-shared";
    let sessions = [
        shared_session,
        shared_session,
        "pb-cache-eval-new-session",
        "pb-cache-eval-changed-authority",
        "pb-cache-eval-matching-authority",
        "pb-cache-eval-restart",
    ];
    let mut arms = Vec::with_capacity(CACHE_EVAL_ARM_IDS.len());
    for index in 0..5 {
        let excludes = if index == 3 {
            &changed_excludes
        } else {
            &baseline_excludes
        };
        let result = run_agent_task(cache_eval_agent_args(
            &args,
            index,
            sessions[index],
            excludes.clone(),
        ));
        arms.push(cache_eval_arm_report(
            CACHE_EVAL_ARM_IDS[index],
            "parent",
            sessions[index],
            &args.scratch_dirs[index],
            result,
        )?);
    }

    let released_parent_runtimes_before_restart =
        crate::inference::flashmoe::release_unleased_shared_runtimes()?;
    let restart_status = run_cache_eval_restart_child(&args, sessions[5], &baseline_excludes);
    arms.push(cache_eval_arm_report(
        CACHE_EVAL_ARM_IDS[5],
        "child",
        sessions[5],
        &args.scratch_dirs[5],
        restart_status,
    )?);

    let gates = evaluate_cache_eval_gates(&arms);
    let passed = gates.passed();
    let executable = std::env::current_exe().context("failed to resolve current pb executable")?;
    let binary_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&executable).with_context(|| format!(
            "failed to fingerprint current pb executable {}",
            executable.display()
        ))?)
    );
    let model = args
        .model
        .clone()
        .context("cache-eval model was not resolved")?;
    let model_dir = args
        .model_dir
        .as_deref()
        .context("cache-eval model directory was not resolved")?
        .display()
        .to_string();
    let sampling = CacheEvalSamplingReport {
        max_steps: args
            .max_steps
            .context("cache-eval max steps were not resolved")?,
        max_tokens: args
            .max_tokens
            .context("cache-eval max tokens were not resolved")?,
        ctx_size: args
            .ctx_size
            .context("cache-eval context size was not resolved")?,
        threads: args.threads,
        threads_batch: args.threads_batch,
        gpu_layers: args
            .gpu_layers
            .context("cache-eval GPU layers were not resolved")?,
        temperature: args
            .temperature
            .context("cache-eval temperature was not resolved")?,
        top_k: args.top_k.context("cache-eval top-k was not resolved")?,
        seed: args.seed.context("cache-eval seed was not resolved")?,
    };
    let report = CacheEvalReport {
        version: 1,
        started_at_ms,
        finished_at_ms: now_millis(),
        binary_sha256,
        model,
        model_dir,
        profile: args.profile,
        changed_authority_tool: args.changed_authority_tool.clone(),
        contract_sha256: input_identity.contract_sha256,
        workspace_fingerprint: input_identity.workspace_fingerprint,
        sampling,
        cache_dir: args.cache_dir.display().to_string(),
        report_path: args.output.display().to_string(),
        released_parent_runtimes_before_restart,
        arms,
        gates,
        passed,
    };
    atomic_write(&args.output, &serde_json::to_vec_pretty(&report)?)?;
    println!(
        "pb harness cache eval: passed={} report={}",
        report.passed,
        args.output.display()
    );
    if !report.passed {
        bail!(
            "prompt-cache lifecycle qualification failed; inspect {}",
            args.output.display()
        );
    }
    Ok(())
}

fn resolve_cache_eval_args(args: &mut HarnessCacheEvalArgs) -> Result<()> {
    let config = UserConfig::load()?;
    args.model.get_or_insert_with(|| config.effective_model());
    args.model_dir.get_or_insert_with(|| {
        config
            .effective_model_dir()
            .unwrap_or_else(crate::default_models_dir)
    });
    args.max_steps
        .get_or_insert_with(|| config.effective_max_steps());
    args.max_tokens
        .get_or_insert_with(|| config.effective_max_tokens());
    args.ctx_size
        .get_or_insert_with(|| config.effective_ctx_size());
    if args.threads.is_none() {
        args.threads = config.effective_threads();
    }
    if args.threads_batch.is_none() {
        args.threads_batch = config.effective_threads_batch();
    }
    args.gpu_layers
        .get_or_insert_with(|| config.effective_gpu_layers());
    args.temperature
        .get_or_insert_with(|| config.effective_temperature());
    args.top_k.get_or_insert_with(|| config.effective_top_k());
    args.seed.get_or_insert_with(|| config.effective_seed());
    Ok(())
}

fn validate_cache_eval_args(args: &HarnessCacheEvalArgs) -> Result<CacheEvalInputIdentity> {
    if args.task.trim().is_empty() {
        bail!("harness cache-eval task must not be empty");
    }
    if args.scratch_dirs.len() != CACHE_EVAL_ARM_IDS.len() {
        bail!(
            "harness cache-eval requires exactly {} --scratch-dir values in documented scenario order",
            CACHE_EVAL_ARM_IDS.len()
        );
    }
    if args.contracts.len() != CACHE_EVAL_ARM_IDS.len() {
        bail!(
            "harness cache-eval requires exactly {} --contract values in scratch order",
            CACHE_EVAL_ARM_IDS.len()
        );
    }
    if !args.cache_dir.is_absolute() {
        bail!(
            "harness cache-eval cache root must be absolute: {}",
            args.cache_dir.display()
        );
    }
    if !args.output.is_absolute() {
        bail!(
            "harness cache-eval output must be absolute: {}",
            args.output.display()
        );
    }
    if args.output.exists() {
        bail!(
            "harness cache-eval refuses to replace existing report {}",
            args.output.display()
        );
    }
    let output_parent = args
        .output
        .parent()
        .context("harness cache-eval output has no parent")?;
    if !output_parent.is_dir() {
        bail!(
            "harness cache-eval output parent does not exist: {}",
            output_parent.display()
        );
    }
    if args.cache_dir.exists() {
        let metadata = std::fs::symlink_metadata(&args.cache_dir).with_context(|| {
            format!(
                "failed to inspect harness cache-eval root {}",
                args.cache_dir.display()
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            bail!(
                "harness cache-eval cache root must be a real directory: {}",
                args.cache_dir.display()
            );
        }
        if std::fs::read_dir(&args.cache_dir)
            .with_context(|| {
                format!(
                    "failed to inspect harness cache-eval root {}",
                    args.cache_dir.display()
                )
            })?
            .next()
            .is_some()
        {
            bail!(
                "harness cache-eval cache root must be empty: {}",
                args.cache_dir.display()
            );
        }
    }

    let mut distinct = BTreeSet::new();
    let mut contract_sha256 = None;
    let mut workspace_fingerprint = None;
    for (index, (scratch, contract)) in args
        .scratch_dirs
        .iter()
        .zip(args.contracts.iter())
        .enumerate()
    {
        if !scratch.is_absolute() || !scratch.is_dir() {
            bail!(
                "cache-eval arm {} scratch must be an existing absolute directory: {}",
                CACHE_EVAL_ARM_IDS[index],
                scratch.display()
            );
        }
        let scratch = scratch.canonicalize().with_context(|| {
            format!("failed to resolve cache-eval scratch {}", scratch.display())
        })?;
        if !distinct.insert(scratch.clone()) {
            bail!(
                "cache-eval scratch roots must be distinct; duplicate {}",
                scratch.display()
            );
        }
        if scratch.join("run-index.jsonl").exists() || scratch.join("events.jsonl").exists() {
            bail!(
                "cache-eval arm {} scratch has prior harness state: {}",
                CACHE_EVAL_ARM_IDS[index],
                scratch.display()
            );
        }
        if !scratch.join("workspace/.git").is_dir() {
            bail!(
                "cache-eval arm {} scratch has no prepared git workspace: {}",
                CACHE_EVAL_ARM_IDS[index],
                scratch.display()
            );
        }
        if !contract.is_absolute() || !contract.is_file() {
            bail!(
                "cache-eval arm {} contract must be an existing absolute file: {}",
                CACHE_EVAL_ARM_IDS[index],
                contract.display()
            );
        }
        let contract = crate::harness_contract::HarnessContractDocument::from_path(contract)?
            .normalize()
            .with_context(|| {
                format!(
                    "invalid cache-eval contract for arm {}",
                    CACHE_EVAL_ARM_IDS[index]
                )
            })?;
        let digest = format!("{:x}", Sha256::digest(serde_json::to_vec(&contract)?));
        if contract_sha256
            .as_ref()
            .is_some_and(|expected| expected != &digest)
        {
            bail!(
                "cache-eval arm {} contract differs from the first arm",
                CACHE_EVAL_ARM_IDS[index]
            );
        }
        contract_sha256.get_or_insert(digest);

        let fingerprint =
            crate::workspace::ContentSnapshot::capture(&scratch.join("workspace"))?.fingerprint;
        if workspace_fingerprint
            .as_ref()
            .is_some_and(|expected| expected != &fingerprint)
        {
            bail!(
                "cache-eval arm {} workspace content differs from the first arm",
                CACHE_EVAL_ARM_IDS[index]
            );
        }
        workspace_fingerprint.get_or_insert(fingerprint);
    }
    if args.changed_authority_tool == "sub_agent" {
        bail!("--changed-authority-tool must differ from the baseline sub_agent exclusion");
    }
    if !HARNESS_AGENT_TOOLS.contains(&args.changed_authority_tool.as_str()) {
        bail!(
            "unknown changed-authority harness tool '{}'",
            args.changed_authority_tool
        );
    }
    Ok(CacheEvalInputIdentity {
        contract_sha256: contract_sha256.context("cache-eval contract identity is missing")?,
        workspace_fingerprint: workspace_fingerprint
            .context("cache-eval workspace identity is missing")?,
    })
}

fn cache_eval_agent_args(
    args: &HarnessCacheEvalArgs,
    index: usize,
    session_id: &str,
    exclude_tools: Vec<String>,
) -> HarnessAgentArgs {
    HarnessAgentArgs {
        task: args.task.clone(),
        intent: crate::workflow::TurnIntent::Deliver,
        scratch_dir: Some(args.scratch_dirs[index].clone()),
        cache_dir: Some(args.cache_dir.clone()),
        session_id: Some(session_id.to_string()),
        exclude_tools,
        contract: Some(args.contracts[index].clone()),
        workspace_config: None,
        workflow_config: None,
        goal_context: None,
        model: args.model.clone(),
        model_dir: args.model_dir.clone(),
        max_steps: args.max_steps,
        max_tokens: args.max_tokens,
        ctx_size: args.ctx_size,
        threads: args.threads,
        threads_batch: args.threads_batch,
        gpu_layers: args.gpu_layers,
        temperature: args.temperature,
        profile: args.profile,
        images: Vec::new(),
        top_k: args.top_k,
        seed: args.seed,
    }
}

fn run_cache_eval_restart_child(
    args: &HarnessCacheEvalArgs,
    session_id: &str,
    exclude_tools: &[String],
) -> Result<()> {
    let executable = std::env::current_exe().context("failed to resolve current pb executable")?;
    let mut command = Command::new(executable);
    command
        .arg("harness")
        .arg("agent")
        .arg("--scratch-dir")
        .arg(&args.scratch_dirs[5])
        .arg("--contract")
        .arg(&args.contracts[5])
        .arg("--cache-dir")
        .arg(&args.cache_dir)
        .arg("--session-id")
        .arg(session_id)
        .arg("--profile")
        .arg(args.profile.to_string());
    for tool in exclude_tools {
        command.arg("--exclude-tool").arg(tool);
    }
    if let Some(value) = args.model.as_deref() {
        command.arg("--model").arg(value);
    }
    if let Some(value) = args.model_dir.as_deref() {
        command.arg("--model-dir").arg(value);
    }
    if let Some(value) = args.max_steps {
        command.arg("--max-steps").arg(value.to_string());
    }
    if let Some(value) = args.max_tokens {
        command.arg("--max-tokens").arg(value.to_string());
    }
    if let Some(value) = args.ctx_size {
        command.arg("--ctx-size").arg(value.to_string());
    }
    if let Some(value) = args.threads {
        command.arg("--threads").arg(value.to_string());
    }
    if let Some(value) = args.threads_batch {
        command.arg("--threads-batch").arg(value.to_string());
    }
    if let Some(value) = args.gpu_layers {
        command.arg("--gpu-layers").arg(value.to_string());
    }
    if let Some(value) = args.temperature {
        command.arg("--temperature").arg(value.to_string());
    }
    if let Some(value) = args.top_k {
        command.arg("--top-k").arg(value.to_string());
    }
    if let Some(value) = args.seed {
        command.arg("--seed").arg(value.to_string());
    }
    command.arg(&args.task);
    let status = command
        .status()
        .context("failed to start cache-eval restart child")?;
    if !status.success() {
        bail!("cache-eval restart child exited with {status}");
    }
    Ok(())
}

fn cache_eval_arm_report(
    id: &str,
    process: &str,
    session_id: &str,
    scratch_dir: &Path,
    result: Result<()>,
) -> Result<CacheEvalArmReport> {
    let (agent_outcome_succeeded, agent_error) = match result {
        Ok(()) => (true, None),
        Err(error) => (false, Some(compact_detail(&format!("{error:#}")))),
    };
    let (invocations, planning_root) =
        read_cache_eval_planning_root(&scratch_dir.join("events.jsonl"))?;
    Ok(CacheEvalArmReport {
        id: id.to_string(),
        process: process.to_string(),
        session_id: session_id.to_string(),
        scratch_dir: scratch_dir.display().to_string(),
        agent_outcome_succeeded,
        agent_error,
        invocations,
        planning_root,
    })
}

fn read_cache_eval_planning_root(path: &Path) -> Result<(usize, Option<CacheEvalRootReport>)> {
    if !path.exists() {
        return Ok((0, None));
    }
    let file = File::open(path)
        .with_context(|| format!("failed to open cache-eval events {}", path.display()))?;
    let mut invocations = 0usize;
    let mut planning_root = None;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read cache-eval events {} line {}",
                path.display(),
                index + 1
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let envelope: EventEnvelope = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse cache-eval events {} line {}",
                path.display(),
                index + 1
            )
        })?;
        if let AgentEvent::LlmInvocation { prompt_cache, .. } = envelope.event {
            invocations += 1;
            if planning_root.is_none()
                && let Some(cache) = prompt_cache
                && let Some(root) = cache.root
                && root.workflow_stage == Some(crate::workflow::WorkflowStage::Planning)
            {
                planning_root = Some(CacheEvalRootReport {
                    backend: root.backend,
                    cache_format_version: root.cache_format_version,
                    model_namespace_sha256: root.model_namespace_sha256,
                    source: cache.source,
                    miss_reason: cache.miss_reason,
                    rendered_token_sha256: root.rendered_token_sha256,
                    tool_schema_sha256: root.tool_schema_sha256,
                    tokens: root.tokens,
                    reused_tokens: root.reused_tokens,
                    workflow_stage: root.workflow_stage,
                    authority_class: root.authority_class,
                    output_constraint_mode: root.output_constraint_mode,
                });
            }
        }
    }
    Ok((invocations, planning_root))
}

fn evaluate_cache_eval_gates(arms: &[CacheEvalArmReport]) -> CacheEvalGates {
    let roots = arms
        .iter()
        .map(|arm| arm.planning_root.as_ref())
        .collect::<Vec<_>>();
    let Some(cold) = roots.first().copied().flatten() else {
        return CacheEvalGates::default();
    };
    let exact_full_reuse = |root: Option<&CacheEvalRootReport>| {
        root.is_some_and(|root| {
            root.rendered_token_sha256 == cold.rendered_token_sha256
                && root.tokens == cold.tokens
                && root.reused_tokens == root.tokens
        })
    };
    let same_session = roots.get(1).copied().flatten();
    let new_session = roots.get(2).copied().flatten();
    let changed = roots.get(3).copied().flatten();
    let matching = roots.get(4).copied().flatten();
    let restart = roots.get(5).copied().flatten();
    CacheEvalGates {
        all_planning_roots_observed: roots.len() == CACHE_EVAL_ARM_IDS.len()
            && roots.iter().all(Option::is_some),
        cold_root_did_not_reuse: cold.source == "none"
            && cold.reused_tokens == 0
            && cold.miss_reason.is_some(),
        same_session_reused_exact_root: exact_full_reuse(same_session)
            && same_session.is_some_and(|root| {
                matches!(root.source.as_str(), "memory_session" | "memory_prefix")
            }),
        new_session_reused_exact_root: exact_full_reuse(new_session)
            && new_session.is_some_and(|root| root.source != "none"),
        changed_authority_rejected_old_root: changed.is_some_and(|root| {
            root.rendered_token_sha256 != cold.rendered_token_sha256
                && root.tool_schema_sha256 != cold.tool_schema_sha256
                && root.reused_tokens == 0
                && root.source == "none"
        }),
        matching_authority_remained_reusable: exact_full_reuse(matching)
            && matching.is_some_and(|root| root.source != "none"),
        restart_restored_persisted_exact_root: exact_full_reuse(restart)
            && restart.is_some_and(|root| root.source == "disk_prefix"),
    }
}

fn harness_tool_allowlist(exclude_tools: &[String]) -> Result<Vec<String>> {
    let mut excluded = BTreeSet::new();
    for tool in exclude_tools {
        if !HARNESS_AGENT_TOOLS.contains(&tool.as_str()) {
            bail!("unknown harness tool exclusion '{tool}'");
        }
        excluded.insert(tool.as_str());
    }
    let tools = HARNESS_AGENT_TOOLS
        .iter()
        .filter(|tool| !excluded.contains(**tool))
        .map(|tool| (*tool).to_string())
        .collect::<Vec<_>>();
    if tools.is_empty() {
        bail!("harness tool exclusions removed every available tool");
    }
    Ok(tools)
}

fn workspace_config_metadata(
    path: &Path,
    graph: &crate::workspace::WorkspaceGraph,
) -> Result<WorkspaceConfigMetadata> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read workspace config {}", path.display()))?;
    let source = path
        .canonicalize()
        .with_context(|| format!("failed to resolve workspace config {}", path.display()))?
        .to_string_lossy()
        .into_owned();
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
    Ok(WorkspaceConfigMetadata {
        source,
        sha256: format!("{:x}", Sha256::digest(bytes)),
        executor_policy,
    })
}

fn workflow_config_metadata(
    path: Option<&Path>,
    document: &crate::workflow::WorkflowConfigDocument,
    policy: &crate::workflow::CompiledWorkflowPolicy,
) -> Result<WorkflowConfigMetadata> {
    let (source, bytes) = if let Some(path) = path {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read workflow config {}", path.display()))?;
        let source = path
            .canonicalize()
            .with_context(|| format!("failed to resolve workflow config {}", path.display()))?
            .to_string_lossy()
            .into_owned();
        (source, bytes)
    } else {
        (
            "builtin:strict-default".to_string(),
            serde_json::to_vec(document)
                .context("failed to serialize built-in workflow configuration")?,
        )
    };
    Ok(WorkflowConfigMetadata {
        source,
        source_sha256: format!("{:x}", Sha256::digest(bytes)),
        policy_sha256: policy.sha256.clone(),
        delivery: policy.delivery,
        default_intent: policy.default_intent,
        limits: policy.limits,
    })
}

fn load_check_evidence(path: &Path) -> Result<crate::checks::CheckEvidenceLedger> {
    if !path.exists() {
        return Ok(crate::checks::CheckEvidenceLedger::default());
    }
    let file = File::open(path)
        .with_context(|| format!("failed to open harness event journal {}", path.display()))?;
    let mut events = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read harness event journal {} line {}",
                path.display(),
                index + 1
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let envelope: EventEnvelope = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse harness event journal {} line {}",
                path.display(),
                index + 1
            )
        })?;
        events.push(envelope.event);
    }
    Ok(crate::checks::CheckEvidenceLedger::from_events(&events))
}

fn load_resumable_workflow_checkpoint(
    path: &Path,
) -> Result<Option<crate::workflow::WorkflowCheckpoint>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path).with_context(|| {
        format!(
            "failed to read harness workflow checkpoint {}",
            path.display()
        )
    })?;
    let checkpoint: crate::workflow::WorkflowCheckpoint = serde_json::from_slice(&bytes)
        .with_context(|| {
            format!(
                "failed to parse harness workflow checkpoint {}",
                path.display()
            )
        })?;
    checkpoint.validate()?;
    if checkpoint.run.stage.is_terminal()
        && checkpoint.run.stage != crate::workflow::WorkflowStage::Blocked
    {
        Ok(None)
    } else {
        Ok(Some(checkpoint))
    }
}

fn resumable_workflow_branch(
    workspace: &Path,
    checkpoint: Option<&crate::workflow::WorkflowCheckpoint>,
) -> Result<Option<String>> {
    if checkpoint.is_none() {
        return Ok(None);
    }
    let branch = require_git_success(workspace, &["branch", "--show-current"])?;
    if branch.is_empty() {
        bail!("active harness workflow cannot resume from a detached HEAD");
    }
    Ok(Some(branch))
}

fn harness_outcome_succeeded(result: &AgentRunResult) -> bool {
    match result.handoff_outcome {
        Some(crate::events::HandoffOutcome::Ready | crate::events::HandoffOutcome::NoChange) => {
            result.contract_status != crate::events::ContractStatus::Unsatisfied
        }
        Some(_) => false,
        None => {
            result.verified_completed
                || (result.reached_final
                    && result.contract_status == crate::events::ContractStatus::Unspecified)
        }
    }
}

fn append_run_index_started(
    layout: &ScratchLayout,
    task: &str,
    workspace_config: Option<&WorkspaceConfigMetadata>,
    workflow_config: &WorkflowConfigMetadata,
) -> Result<()> {
    append_run_index_record(
        &layout.run_index,
        &RunIndexRecord::Started {
            version: 1,
            run_id: layout.run_id.clone(),
            timestamp_ms: now_millis(),
            task: task.to_string(),
            run_events: relative_to_root(&layout.root, &layout.run_events),
            run_journal: relative_to_root(&layout.root, &layout.run_journal),
            workspace_config: workspace_config.cloned(),
            workflow_config: workflow_config.clone(),
        },
    )
}

fn append_run_index_finished(
    layout: &ScratchLayout,
    result: &Result<AgentRunResult>,
    audit: &HarnessRunAudit,
    workspace_config: Option<&WorkspaceConfigMetadata>,
    workflow_config: &WorkflowConfigMetadata,
) -> Result<()> {
    let (status, reached_final, contract_status, verified_completed, termination_reason, error) =
        match result {
            Ok(result) if result.verified_completed => (
                "verified_completed",
                result.reached_final,
                result.contract_status,
                true,
                Some(result.termination_reason),
                None,
            ),
            Ok(result) if result.handoff_outcome == Some(crate::events::HandoffOutcome::Ready) => (
                "ready",
                result.reached_final,
                result.contract_status,
                false,
                Some(result.termination_reason),
                None,
            ),
            Ok(result)
                if result.handoff_outcome == Some(crate::events::HandoffOutcome::NoChange) =>
            {
                (
                    "no_change",
                    result.reached_final,
                    result.contract_status,
                    false,
                    Some(result.termination_reason),
                    None,
                )
            }
            Ok(result) if result.handoff_outcome.is_some() => (
                "incomplete",
                result.reached_final,
                result.contract_status,
                false,
                Some(result.termination_reason),
                None,
            ),
            Ok(result)
                if result.reached_final
                    && result.contract_status == crate::events::ContractStatus::Unspecified =>
            {
                (
                    "final_unverified",
                    true,
                    result.contract_status,
                    false,
                    Some(result.termination_reason),
                    None,
                )
            }
            Ok(result) => (
                "incomplete",
                result.reached_final,
                result.contract_status,
                false,
                Some(result.termination_reason),
                None,
            ),
            Err(error) => (
                "failed",
                false,
                crate::events::ContractStatus::Unspecified,
                false,
                Some(crate::events::TerminationReason::EngineError),
                Some(compact_detail(&format!("{error:#}"))),
            ),
        };
    append_run_index_record(
        &layout.run_index,
        &RunIndexRecord::Finished {
            version: 1,
            run_id: layout.run_id.clone(),
            timestamp_ms: now_millis(),
            status: status.to_string(),
            reached_final,
            contract_status,
            verified_completed,
            termination_reason,
            handoff_outcome: result
                .as_ref()
                .ok()
                .and_then(|result| result.handoff_outcome),
            task_baseline_id: result
                .as_ref()
                .ok()
                .and_then(|result| result.repository_context.as_ref())
                .map(|context| context.task_baseline.id.clone()),
            invocation_baseline_id: result
                .as_ref()
                .ok()
                .and_then(|result| result.repository_context.as_ref())
                .map(|context| context.invocation_baseline.id.clone()),
            workspace_config: workspace_config.cloned(),
            workflow_config: workflow_config.clone(),
            audit: audit.clone(),
            error,
        },
    )
}

fn append_run_index_record(path: &Path, record: &RunIndexRecord) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open harness run index {}", path.display()))?;
    serde_json::to_writer(&mut file, record)
        .with_context(|| format!("failed to encode harness run index {}", path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("failed to append harness run index {}", path.display()))?;
    file.sync_data()
        .with_context(|| format!("failed to sync harness run index {}", path.display()))
}

fn relative_to_root(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn prepare_scratch(requested: Option<&Path>) -> Result<ScratchLayout> {
    let (root, resumed) = match requested {
        Some(path) => {
            if path.exists() {
                if !path.is_dir() {
                    bail!(
                        "harness scratch path is not a directory: {}",
                        path.display()
                    );
                }
                let mut entries = std::fs::read_dir(path).with_context(|| {
                    format!("failed to inspect harness scratch {}", path.display())
                })?;
                (path.to_path_buf(), entries.next().is_some())
            } else {
                std::fs::create_dir_all(path).with_context(|| {
                    format!("failed to create harness scratch {}", path.display())
                })?;
                (path.to_path_buf(), false)
            }
        }
        None => (create_unique_scratch_root()?, false),
    };
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve harness scratch {}", root.display()))?;
    let workspace = root.join("workspace");
    if resumed {
        if !workspace.join(".git").is_dir() {
            bail!(
                "existing harness scratch has no git workspace: {}",
                workspace.display()
            );
        }
    } else {
        std::fs::create_dir(&workspace).with_context(|| {
            format!("failed to create harness workspace {}", workspace.display())
        })?;
        initialize_git_workspace(&workspace)?;
    }
    let runs = root.join("runs");
    std::fs::create_dir_all(&runs)
        .with_context(|| format!("failed to create harness runs directory {}", runs.display()))?;
    let (run_id, run_dir) = create_unique_run_dir(&runs)?;
    Ok(ScratchLayout {
        events: root.join("events.jsonl"),
        journal: root.join("journal.md"),
        run_index: root.join("run-index.jsonl"),
        run_events: run_dir.join("events.jsonl"),
        run_journal: run_dir.join("journal.md"),
        run_task_planning_transcript: run_dir.join("task-planning-transcript.json"),
        task_baseline: root.join("task-baseline.json"),
        adoptions: root.join("adoptions.jsonl"),
        workflow_checkpoint: root.join("workflow-checkpoint.json"),
        multi_task_checkpoint: root.join("multi-task-checkpoint.json"),
        run_id,
        root,
        workspace,
        resumed,
    })
}

fn create_unique_run_dir(runs: &Path) -> Result<(String, PathBuf)> {
    let timestamp = now_millis();
    let pid = std::process::id();
    for suffix in 0..1000u16 {
        let run_id = format!("{timestamp}-{pid}-{suffix}");
        let path = runs.join(&run_id);
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok((run_id, path)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create harness run directory {}", path.display())
                });
            }
        }
    }
    bail!("could not allocate a unique harness run directory")
}

fn harness_repository_context(
    layout: &ScratchLayout,
) -> Result<crate::workspace::RepositoryContext> {
    let task_baseline = if layout.task_baseline.exists() {
        let bytes = std::fs::read(&layout.task_baseline).with_context(|| {
            format!(
                "failed to read harness task baseline {}",
                layout.task_baseline.display()
            )
        })?;
        serde_json::from_slice::<crate::workspace::WorkspaceBaseline>(&bytes).with_context(
            || {
                format!(
                    "failed to parse harness task baseline {}",
                    layout.task_baseline.display()
                )
            },
        )?
    } else {
        let baseline = crate::workspace::WorkspaceBaseline::capture(&layout.workspace)?;
        let bytes = serde_json::to_vec_pretty(&baseline)
            .context("failed to serialize harness task baseline")?;
        atomic_write(&layout.task_baseline, &bytes)?;
        baseline
    };
    let context = crate::workspace::RepositoryContext::resume(
        &layout.workspace,
        &layout.workspace,
        task_baseline,
    )?;
    if layout.resumed {
        let adopted_paths = context.task_changed_paths()?;
        if !adopted_paths.is_empty() {
            let record = serde_json::json!({
                "version": 1,
                "type": "resume_adoption",
                "run_id": layout.run_id,
                "timestamp_ms": now_millis(),
                "task_baseline_id": context.task_baseline.id,
                "invocation_baseline_id": context.invocation_baseline.id,
                "paths": adopted_paths,
            });
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&layout.adoptions)
                .with_context(|| {
                    format!(
                        "failed to open harness adoption journal {}",
                        layout.adoptions.display()
                    )
                })?;
            serde_json::to_writer(&mut file, &record)
                .context("failed to encode harness adoption record")?;
            file.write_all(b"\n")
                .context("failed to append harness adoption record")?;
            file.sync_data()
                .context("failed to sync harness adoption record")?;
        }
    }
    Ok(context)
}

fn create_unique_scratch_root() -> Result<PathBuf> {
    let base = std::env::temp_dir();
    let timestamp = now_millis();
    let pid = std::process::id();
    for suffix in 0..100u8 {
        let path = base.join(format!("pb-harness-{timestamp}-{pid}-{suffix}"));
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create harness scratch {}", path.display())
                });
            }
        }
    }
    bail!("could not allocate a unique harness scratch directory")
}

fn initialize_git_workspace(workspace: &Path) -> Result<()> {
    let initialized = git_command(workspace, &["init", "-b", "main"])?;
    if !initialized.status.success() {
        require_git_success(workspace, &["init"])?;
        require_git_success(workspace, &["branch", "-M", "main"])?;
    }
    require_git_success(workspace, &["config", "user.name", HARNESS_GIT_NAME])?;
    require_git_success(workspace, &["config", "user.email", HARNESS_GIT_EMAIL])?;
    require_git_success(
        workspace,
        &[
            "commit",
            "--allow-empty",
            "-m",
            "chore: initialize harness workspace",
        ],
    )?;
    Ok(())
}

fn git_command(workspace: &Path, args: &[&str]) -> Result<std::process::Output> {
    Command::new("git")
        .args(args)
        .current_dir(workspace)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))
}

fn require_git_success(workspace: &Path, args: &[&str]) -> Result<String> {
    let output = git_command(workspace, args)?;
    if !output.status.success() {
        bail!(
            "git {} failed in {}: {}",
            args.join(" "),
            workspace.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn harness_environment() -> EnvironmentConfig {
    EnvironmentConfig {
        version: crate::environment::ENVIRONMENT_CONFIG_VERSION,
        mode: EnvironmentMode::Local,
        backend: EnvironmentBackend::Local,
        image: "local".to_string(),
        init_commands: Vec::new(),
        setup_commands: Vec::new(),
        session_commands: Vec::new(),
        env: Default::default(),
        bootstrap_network: crate::environment::EnvironmentNetworkMode::Egress,
        runtime_network: crate::environment::EnvironmentNetworkMode::Isolated,
        resources: Default::default(),
        caches: Vec::new(),
        guard_commands: Vec::new(),
        prepared_image: None,
        source: Some("pb harness scratch workspace".to_string()),
        dockerfile: None,
    }
}

fn harness_attachments(paths: &[PathBuf]) -> Result<Vec<SessionAttachment>> {
    paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let path = path
                .canonicalize()
                .with_context(|| format!("failed to resolve harness image {}", path.display()))?;
            let metadata = path
                .metadata()
                .with_context(|| format!("failed to inspect harness image {}", path.display()))?;
            Ok(SessionAttachment {
                id: format!("img{}", index + 1),
                name: path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("image")
                    .to_string(),
                mime: mime_guess::from_path(&path)
                    .first_or_octet_stream()
                    .to_string(),
                path: path.to_string_lossy().into_owned(),
                size: metadata.len(),
            })
        })
        .collect()
}

fn add_run_observations(
    observations: &mut Vec<Observation>,
    result: &Result<AgentRunResult>,
    workspace: &Path,
    summary: &CapturedSummary,
) {
    match result {
        Err(error) => observations.push(Observation {
            rank: 2,
            classification: "experiment_error",
            title: "agent run failed".to_string(),
            detail: compact_detail(&format!("{error:#}")),
        }),
        Ok(result) if !result.reached_final => observations.push(Observation {
            rank: 2,
            classification: "model_limitation",
            title: "agent did not reach a final answer".to_string(),
            detail: "The step budget ended before the full task completion contract was satisfied."
                .to_string(),
        }),
        Ok(result) if !result.verified_completed && result.contract_status != crate::events::ContractStatus::Unspecified => {
            observations.push(Observation {
                rank: 2,
                classification: "model_limitation",
                title: "acceptance contract was not satisfied".to_string(),
                detail: format!(
                    "The model emitted a final action, but the run terminated as {} with contract_status={}.",
                    result.termination_reason, result.contract_status
                ),
            })
        }
        Ok(_) => {}
    }

    let committed =
        require_git_success(workspace, &["log", "--oneline", "main..HEAD"]).unwrap_or_default();
    if matches!(result, Ok(result) if result.reached_final
        && result.handoff_outcome == Some(crate::events::HandoffOutcome::Ready))
        && committed.trim().is_empty()
    {
        observations.push(Observation {
            rank: 2,
            classification: "experiment_error",
            title: "completed run produced no commits".to_string(),
            detail: "Confirm whether the task genuinely required no repository changes."
                .to_string(),
        });
    }
    let status = require_git_success(workspace, &["status", "--short"]).unwrap_or_default();
    if !status.trim().is_empty() {
        let incomplete_delivery = matches!(result, Ok(result)
        if !result.verified_completed
            && result.handoff_outcome.is_some_and(|outcome| !matches!(
                outcome,
                crate::events::HandoffOutcome::Ready
                    | crate::events::HandoffOutcome::NoChange
            )));
        observations.push(Observation {
            rank: if incomplete_delivery { 3 } else { 2 },
            classification: if incomplete_delivery {
                "model_limitation"
            } else {
                "experiment_error"
            },
            title: if incomplete_delivery {
                "incomplete workspace preserves uncommitted changes"
            } else {
                "workspace has uncommitted changes"
            }
            .to_string(),
            detail: if incomplete_delivery {
                format!(
                    "Expected partial state was preserved and still fails any clean-workspace contract: {}",
                    compact_detail(&status)
                )
            } else {
                compact_detail(&status)
            },
        });
    }
    let normal_final_requires_summary = matches!(
        result,
        Ok(result)
            if result.reached_final
                && result.requested_delivery.is_none()
                && result.requested_goal.is_none()
    );
    if summary.summary.trim().is_empty() && normal_final_requires_summary {
        observations.push(Observation {
            rank: 3,
            classification: "experiment_error",
            title: "agent emitted no session summary".to_string(),
            detail: "Review the final event stream to determine the actual outcome.".to_string(),
        });
    }
    if observations.is_empty() {
        observations.push(Observation {
            rank: 3,
            classification: "positive_evidence",
            title: "no automatic runtime issues observed".to_string(),
            detail: "Manual review of the committed implementation and tests is still required."
                .to_string(),
        });
    }
}

fn write_journal(
    layout: &ScratchLayout,
    task: &str,
    result: &Result<AgentRunResult>,
    summary: &CapturedSummary,
    audit: &HarnessRunAudit,
    workspace_config: Option<&WorkspaceConfigMetadata>,
    workflow_config: &WorkflowConfigMetadata,
    observations: &mut Vec<Observation>,
) -> Result<()> {
    observations.sort_by(|left, right| {
        left.rank
            .cmp(&right.rank)
            .then_with(|| left.detail.cmp(&right.detail))
            .then_with(|| left.title.cmp(&right.title))
    });
    observations.dedup_by(|left, right| left.rank == right.rank && left.detail == right.detail);
    let status = match result {
        Ok(result) if result.verified_completed => "verified-completed",
        Ok(result) if result.handoff_outcome == Some(crate::events::HandoffOutcome::Ready) => {
            "ready"
        }
        Ok(result) if result.handoff_outcome == Some(crate::events::HandoffOutcome::NoChange) => {
            "no-change"
        }
        Ok(result) if result.handoff_outcome.is_some() => "incomplete",
        Ok(result)
            if result.reached_final
                && result.contract_status == crate::events::ContractStatus::Unspecified =>
        {
            "final-unverified"
        }
        Ok(result) if result.reached_final => "contract-unsatisfied",
        Ok(_) => "incomplete",
        Err(_) => "failed",
    };
    let branch = result
        .as_ref()
        .map(|result| result.branch.as_str())
        .unwrap_or(summary.branch.as_str());
    let committed = require_git_success(&layout.workspace, &["log", "--oneline", "main..HEAD"])
        .unwrap_or_else(|_| summary.commits.clone());

    let mut journal = String::new();
    journal.push_str("# pb harness journal\n\n");
    journal.push_str(&format!("- Status: `{status}`\n"));
    journal.push_str(&format!("- Run ID: `{}`\n", layout.run_id));
    if let Ok(result) = result {
        journal.push_str(&format!("- Reached final: `{}`\n", result.reached_final));
        journal.push_str(&format!(
            "- Contract status: `{}`\n",
            result.contract_status
        ));
        journal.push_str(&format!(
            "- Verified completed: `{}`\n",
            result.verified_completed
        ));
        journal.push_str(&format!(
            "- Termination reason: `{}`\n",
            result.termination_reason
        ));
        journal.push_str(&format!(
            "- Handoff outcome: `{}`\n",
            result
                .handoff_outcome
                .map(|outcome| format!("{outcome:?}").to_ascii_lowercase())
                .unwrap_or_else(|| "none".to_string())
        ));
        if let Some(context) = result.repository_context.as_ref() {
            journal.push_str(&format!(
                "- Task baseline: `{}`\n- Invocation baseline: `{}`\n",
                context.task_baseline.id, context.invocation_baseline.id
            ));
        }
    }
    if let Some(metadata) = workspace_config {
        journal.push_str(&format!(
            "- Workspace config: `{}`\n- Workspace config SHA-256: `{}`\n- Executor policy: `{}`\n",
            metadata.source,
            metadata.sha256,
            metadata.executor_policy.join(", ")
        ));
    }
    journal.push_str(&format!(
        "- Workflow config: `{}`\n- Workflow config SHA-256: `{}`\n- Workflow policy SHA-256: `{}`\n- Workflow policy: `{:?}` (default intent `{:?}`)\n",
        workflow_config.source,
        workflow_config.source_sha256,
        workflow_config.policy_sha256,
        workflow_config.delivery,
        workflow_config.default_intent,
    ));
    journal.push_str(&format!("- Task: {task}\n"));
    journal.push_str(&format!("- Workspace: `{}`\n", layout.workspace.display()));
    journal.push_str(&format!("- Branch: `{branch}`\n"));
    journal.push_str(&format!(
        "- Run events: `{}`\n",
        layout.run_events.display()
    ));
    journal.push_str(&format!(
        "- Cumulative events: `{}`\n",
        layout.events.display()
    ));
    journal.push_str("\n## Goal audit\n\n");
    journal.push_str(&format!(
        "- Goal ID: `{}`\n- Stage: `{}`\n- Plan SHA-256: `{}`\n- Milestones completed/total: `{}/{}`\n- Workflows / model invocations / generated tokens: `{}` / `{}` / `{}`\n- Pause / amendment / budget requests: `{}` / `{}` / `{}`\n",
        audit.goal_id.as_deref().unwrap_or("none"),
        audit
            .goal_stage
            .map(|stage| format!("{stage:?}").to_ascii_lowercase())
            .unwrap_or_else(|| "none".to_string()),
        audit.goal_plan_sha256.as_deref().unwrap_or("none"),
        audit.goal_completed_milestones,
        audit.goal_total_milestones,
        audit.goal_workflows,
        audit.goal_model_invocations,
        audit.goal_generated_tokens,
        audit.goal_pause_requests,
        audit.goal_amendment_requests,
        audit.goal_budget_requests,
    ));
    journal.push_str("\n## Deterministic controller actions\n\n");
    journal.push_str(&format!(
        "- Observation rendering: `{}`\n- Observations / prompt bytes: `{}` / `{}`\n- Observation coverage: `{}`\n- Controller mutations / closures: `{}` / `{}`\n",
        audit
            .observation_rendering
            .map(crate::workflow::ObservationRendering::as_str)
            .unwrap_or("none"),
        audit.controller_observations,
        audit.controller_observation_prompt_bytes,
        serde_json::to_string(&audit.controller_observation_coverage)
            .context("failed to serialize controller observation coverage audit")?,
        audit.controller_mutations,
        audit.controller_closures,
    ));
    journal.push_str("\n## Workflow audit\n\n");
    journal.push_str(&format!(
        "- Strict workflow enabled/satisfied: `{}` / `{}`\n- Workflow ID: `{}`\n- Outcome: `{}`\n- Stage sequence: `{}`\n- Plan / plan review / code review SHA-256: `{}` / `{}` / `{}`\n- Plan / repair cycles: `{}` / `{}`\n- Model invocations / generated tokens / advisory calls: `{}` / `{}` / `{}`\n- Stage steps: `{}`\n- Rejected workflow actions: `{}`\n- Evidence invalidations: `{}`\n- Checkpoint SHA-256: `{}`\n- Ready evidence SHA-256: `{}`\n- Repository remote: `{}`\n",
        audit.strict_workflow,
        audit.strict_workflow_satisfied,
        audit.workflow_id.as_deref().unwrap_or("none"),
        audit
            .workflow_outcome
            .map(|outcome| format!("{outcome:?}").to_ascii_lowercase())
            .unwrap_or_else(|| "none".to_string()),
        audit
            .workflow_stage_sequence
            .iter()
            .map(|stage| format!("{stage:?}").to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(" -> "),
        audit.plan_sha256.as_deref().unwrap_or("none"),
        audit.plan_review_sha256.as_deref().unwrap_or("none"),
        audit.code_review_sha256.as_deref().unwrap_or("none"),
        audit.plan_cycles,
        audit.repair_cycles,
        audit.workflow_model_invocations,
        audit.workflow_generated_tokens,
        audit.workflow_advisory_calls,
        serde_json::to_string(&audit.workflow_stage_steps)
            .context("failed to serialize workflow stage step audit")?,
        audit.rejected_workflow_actions,
        audit.evidence_invalidations,
        audit
            .workflow_checkpoint_sha256
            .as_deref()
            .unwrap_or("none"),
        audit.ready_evidence_sha256.as_deref().unwrap_or("none"),
        audit.repository_remote.as_deref().unwrap_or("none"),
    ));
    journal.push_str("\n## Handoff audit\n\n");
    journal.push_str(&format!(
        "- Affected components: `{}`\n- Checks planned: `{}`\n- Checks executed/reused/passed/failed/skipped: `{}/{}/{}/{}/{}`\n- Executors started/failed: `{}` / `{}`\n- Repair turns: `{}`\n- Team messages: `{}`\n- No change: `{}`\n- Commit disposition: `{}`\n- Commit OID: `{}`\n- Check evidence: `{}`\n- Output fingerprints: `{}`\n",
        audit.affected_components.join(", "),
        audit.checks_planned.join(", "),
        audit.checks_executed,
        audit.checks_reused,
        audit.checks_passed,
        audit.checks_failed,
        audit.checks_skipped,
        audit.executors_started.join(", "),
        audit.executors_failed.join(", "),
        audit.repair_turns,
        audit.team_messages,
        audit.no_change,
        audit.commit_disposition.as_deref().unwrap_or("none"),
        audit.commit_oid.as_deref().unwrap_or("none"),
        audit.check_evidence_ids.join(", "),
        audit.output_fingerprints.join(", "),
    ));
    journal.push_str("\n## Ranked observations\n\n");
    for observation in observations.iter() {
        journal.push_str(&format!(
            "1. **P{} — {} — {}.** {}\n",
            observation.rank, observation.classification, observation.title, observation.detail
        ));
    }
    journal.push_str("\n## Agent summary\n\n");
    if summary.summary.trim().is_empty() {
        journal.push_str("_No session summary was emitted._\n");
    } else {
        journal.push_str(summary.summary.trim());
        journal.push('\n');
    }
    journal.push_str("\n## Committed fixes\n\n");
    if committed.trim().is_empty() {
        journal.push_str("_No commits beyond the harness baseline._\n");
    } else {
        journal.push_str("```text\n");
        journal.push_str(committed.trim());
        journal.push_str("\n```\n");
    }
    if !summary.diff_stat.trim().is_empty() {
        journal.push_str("\n### Diff stat\n\n```text\n");
        journal.push_str(summary.diff_stat.trim());
        journal.push_str("\n```\n");
    }
    journal.push_str("\n## Follow-up improvement plan\n\n");
    journal.push_str("- [ ] Review the committed implementation and rerun its acceptance checks independently.\n");
    for observation in observations.iter().filter(|item| item.rank <= 1) {
        journal.push_str(&format!(
            "- [ ] Reproduce and address P{}: {}.\n",
            observation.rank, observation.title
        ));
    }
    journal.push_str(
        "- [ ] Convert validated, non-blocking observations into a prioritized improvement plan.\n",
    );
    atomic_write(&layout.run_journal, journal.as_bytes())?;
    atomic_write(&layout.journal, journal.as_bytes())
}

fn write_running_journal(
    layout: &ScratchLayout,
    task: &str,
    workspace_config: Option<&WorkspaceConfigMetadata>,
    workflow_config: &WorkflowConfigMetadata,
) -> Result<()> {
    let workspace_metadata = workspace_config
        .map(|metadata| {
            format!(
                "- Workspace config: `{}`\n- Workspace config SHA-256: `{}`\n- Executor policy: `{}`\n",
                metadata.source,
                metadata.sha256,
                metadata.executor_policy.join(", ")
            )
        })
        .unwrap_or_default();
    let workflow_metadata = format!(
        "- Workflow config: `{}`\n- Workflow config SHA-256: `{}`\n- Workflow policy SHA-256: `{}`\n",
        workflow_config.source, workflow_config.source_sha256, workflow_config.policy_sha256,
    );
    let journal = format!(
        "# pb harness journal\n\n\
         - Status: `running`\n\
         - Run ID: `{run_id}`\n\
         - Task: {task}\n\
         - Workspace: `{workspace}`\n\
         - Run events: `{run_events}`\n\
         - Cumulative events: `{events}`\n\
         {workspace_metadata}{workflow_metadata}\n\
         ## Ranked observations\n\n\
         1. **P2 — experiment_error — run has not finalized.** If the harness was interrupted, inspect the raw event stream and workspace before deciding whether to rerun.\n\n\
         ## Follow-up improvement plan\n\n\
         - [ ] Wait for the blocking agent run to finish, or diagnose why it was interrupted.\n\
         - [ ] Review the workspace and raw events before making changes to pb.\n",
        workspace = layout.workspace.display(),
        run_id = layout.run_id,
        run_events = layout.run_events.display(),
        events = layout.events.display(),
    );
    atomic_write(&layout.run_journal, journal.as_bytes())?;
    atomic_write(&layout.journal, journal.as_bytes())
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("harness audit path has no parent: {}", path.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("audit");
    for suffix in 0..100u8 {
        let temporary = parent.join(format!(
            ".{file_name}.{}.{}.{suffix}.tmp",
            std::process::id(),
            now_millis()
        ));
        let mut file = match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create atomic audit file {}", temporary.display())
                });
            }
        };
        let write_result = file
            .write_all(contents)
            .and_then(|_| file.sync_all())
            .and_then(|_| std::fs::rename(&temporary, path));
        if let Err(error) = write_result {
            let _ = std::fs::remove_file(&temporary);
            return Err(error)
                .with_context(|| format!("failed to atomically write {}", path.display()));
        }
        return Ok(());
    }
    bail!(
        "could not allocate atomic audit file for {}",
        path.display()
    )
}

fn nonempty_or(value: &str, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        fallback.to_string()
    } else {
        compact_detail(value)
    }
}

fn correction_classification(message: &str, summary: &str) -> &'static str {
    if matches!(
        summary,
        "Next accepted-plan creation work unit"
            | "Active accepted-plan work unit"
            | "Work-unit progress earned one bounded turn"
    ) || (summary == "Harness diagnostic preview"
        && message.starts_with("Harness diagnostic preview passed."))
    {
        "positive_evidence"
    } else {
        "model_limitation"
    }
}

fn compact_detail(value: &str) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let shortened = chars.by_ref().take(800).collect::<String>();
    if chars.next().is_some() {
        format!("{shortened}…")
    } else {
        shortened
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_agent_tool_surface_is_minimal_but_complete() {
        assert_eq!(
            HARNESS_AGENT_TOOLS,
            [
                "session_title",
                "run_command",
                "run_check",
                "read_file",
                "write_file",
                "replace_file",
                "edit_file",
                "apply_patch",
                "rm",
                "git_commit",
                "sub_agent",
                "propose_delivery",
                "start_delivery",
                "propose_goal",
                "start_goal",
                "goal_status",
                "goal_pause",
                "goal_request_amendment",
                "goal_request_budget"
            ]
        );
    }

    #[test]
    fn harness_tool_exclusions_are_validated_and_order_preserving() {
        let tools =
            harness_tool_allowlist(&["sub_agent".to_string(), "read_file".to_string()]).unwrap();
        assert!(!tools.iter().any(|tool| tool == "sub_agent"));
        assert!(!tools.iter().any(|tool| tool == "read_file"));
        assert_eq!(tools.first().map(String::as_str), Some("session_title"));
        assert!(harness_tool_allowlist(&["not_a_pb_tool".to_string()]).is_err());
    }

    fn cache_eval_test_args(parent: &Path) -> HarnessCacheEvalArgs {
        HarnessCacheEvalArgs {
            task: "Build the fixture".to_string(),
            scratch_dirs: (0..6)
                .map(|index| parent.join(format!("arm-{index}")))
                .collect(),
            contracts: (0..6)
                .map(|index| parent.join(format!("contract-{index}.json")))
                .collect(),
            cache_dir: parent.join("cache"),
            output: parent.join("report.json"),
            changed_authority_tool: "read_file".to_string(),
            model: Some("hf://example/model".to_string()),
            model_dir: Some(parent.join("models")),
            max_steps: Some(4),
            max_tokens: Some(64),
            ctx_size: Some(4096),
            threads: None,
            threads_batch: None,
            gpu_layers: Some(99),
            temperature: Some(0.0),
            profile: crate::agent_core::AgentProfile::Build,
            top_k: Some(1),
            seed: Some(0),
        }
    }

    #[test]
    fn cache_eval_refuses_existing_report_and_nonempty_cache_before_inference() {
        let parent = tempfile::tempdir().unwrap();
        let mut args = cache_eval_test_args(parent.path());
        std::fs::write(&args.output, b"preserve me").unwrap();
        let error = validate_cache_eval_args(&args).unwrap_err().to_string();
        assert!(
            error.contains("refuses to replace existing report"),
            "{error}"
        );
        assert_eq!(std::fs::read(&args.output).unwrap(), b"preserve me");

        std::fs::remove_file(&args.output).unwrap();
        std::fs::create_dir(&args.cache_dir).unwrap();
        std::fs::write(args.cache_dir.join("owned.bin"), b"preserve me").unwrap();
        args.output = parent.path().join("new-report.json");
        let error = validate_cache_eval_args(&args).unwrap_err().to_string();
        assert!(error.contains("cache root must be empty"), "{error}");
        assert_eq!(
            std::fs::read(args.cache_dir.join("owned.bin")).unwrap(),
            b"preserve me"
        );
    }

    fn cache_eval_test_root(
        digest: &str,
        schema: &str,
        source: &str,
        reused_tokens: usize,
    ) -> CacheEvalRootReport {
        CacheEvalRootReport {
            backend: "flashmoe".to_string(),
            cache_format_version: "test-v1".to_string(),
            model_namespace_sha256: "model-a".to_string(),
            source: source.to_string(),
            miss_reason: (source == "none")
                .then_some(crate::inference::PromptCacheMissReason::ColdSession),
            rendered_token_sha256: digest.to_string(),
            tool_schema_sha256: Some(schema.to_string()),
            tokens: 128,
            reused_tokens,
            workflow_stage: Some(crate::workflow::WorkflowStage::Planning),
            authority_class: crate::inference::StageRootAuthorityClass::Planning,
            output_constraint_mode: Some("tools_allowed".to_string()),
        }
    }

    fn cache_eval_test_arm(id: &str, root: CacheEvalRootReport) -> CacheEvalArmReport {
        CacheEvalArmReport {
            id: id.to_string(),
            process: "test".to_string(),
            session_id: id.to_string(),
            scratch_dir: format!("/tmp/{id}"),
            agent_outcome_succeeded: false,
            agent_error: Some("model artifact was incomplete".to_string()),
            invocations: 1,
            planning_root: Some(root),
        }
    }

    #[test]
    fn cache_eval_gates_exact_roots_independently_of_agent_outcomes() {
        let arms = vec![
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[0],
                cache_eval_test_root("root-a", "schema-a", "none", 0),
            ),
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[1],
                cache_eval_test_root("root-a", "schema-a", "memory_session", 128),
            ),
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[2],
                cache_eval_test_root("root-a", "schema-a", "memory_prefix", 128),
            ),
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[3],
                cache_eval_test_root("root-b", "schema-b", "none", 0),
            ),
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[4],
                cache_eval_test_root("root-a", "schema-a", "memory_prefix", 128),
            ),
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[5],
                cache_eval_test_root("root-a", "schema-a", "disk_prefix", 128),
            ),
        ];
        let gates = evaluate_cache_eval_gates(&arms);
        assert!(gates.passed());
        assert!(arms.iter().all(|arm| !arm.agent_outcome_succeeded));
    }

    #[test]
    fn cache_eval_rejects_authority_aliasing_and_partial_restart_reuse() {
        let mut arms = vec![
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[0],
                cache_eval_test_root("root-a", "schema-a", "none", 0),
            ),
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[1],
                cache_eval_test_root("root-a", "schema-a", "memory_session", 128),
            ),
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[2],
                cache_eval_test_root("root-a", "schema-a", "memory_prefix", 128),
            ),
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[3],
                cache_eval_test_root("root-a", "schema-a", "memory_prefix", 128),
            ),
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[4],
                cache_eval_test_root("root-a", "schema-a", "memory_prefix", 128),
            ),
            cache_eval_test_arm(
                CACHE_EVAL_ARM_IDS[5],
                cache_eval_test_root("root-a", "schema-a", "disk_prefix", 127),
            ),
        ];
        let gates = evaluate_cache_eval_gates(&arms);
        assert!(!gates.changed_authority_rejected_old_root);
        assert!(!gates.restart_restored_persisted_exact_root);
        assert!(!gates.passed());

        arms[3].planning_root = None;
        assert!(!evaluate_cache_eval_gates(&arms).all_planning_roots_observed);
    }

    #[test]
    fn harness_exit_success_accepts_handoff_ready_no_change_and_legacy_final() {
        let result = |reached_final, contract_status, verified_completed, termination_reason| {
            AgentRunResult {
                branch: "task".to_string(),
                workspace_root: PathBuf::from("/tmp/task"),
                focus_root: PathBuf::from("/tmp/task"),
                repository_context: None,
                workspace_graph: None,
                reached_final,
                contract_status,
                verified_completed,
                termination_reason,
                handoff_outcome: None,
                workflow: None,
                delivery_proposal: None,
                requested_delivery: None,
                goal_proposal: None,
                requested_goal: None,
                task_goal: None,
                task_plan_rejected: None,
                task_planning_transcript: None,
            }
        };

        assert!(harness_outcome_succeeded(&result(
            true,
            crate::events::ContractStatus::Unspecified,
            false,
            crate::events::TerminationReason::Final,
        )));
        assert!(harness_outcome_succeeded(&result(
            true,
            crate::events::ContractStatus::Satisfied,
            true,
            crate::events::TerminationReason::Final,
        )));
        assert!(!harness_outcome_succeeded(&result(
            true,
            crate::events::ContractStatus::Unsatisfied,
            false,
            crate::events::TerminationReason::ContractUnsatisfied,
        )));

        for outcome in [
            crate::events::HandoffOutcome::Ready,
            crate::events::HandoffOutcome::NoChange,
        ] {
            let mut handoff = result(
                true,
                crate::events::ContractStatus::Unspecified,
                false,
                crate::events::TerminationReason::Final,
            );
            handoff.handoff_outcome = Some(outcome);
            assert!(harness_outcome_succeeded(&handoff));
            assert!(!handoff.verified_completed);
        }
        let mut failed_handoff = result(
            true,
            crate::events::ContractStatus::Unspecified,
            false,
            crate::events::TerminationReason::ChecksFailed,
        );
        failed_handoff.handoff_outcome = Some(crate::events::HandoffOutcome::ChecksFailed);
        assert!(!harness_outcome_succeeded(&failed_handoff));
    }

    #[test]
    fn scratch_workspace_is_persistent_git_repository() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();

        assert_eq!(layout.root, root.canonicalize().unwrap());
        assert!(!layout.resumed);
        assert!(layout.workspace.join(".git").is_dir());
        assert_eq!(
            require_git_success(&layout.workspace, &["branch", "--show-current"]).unwrap(),
            "main"
        );
        assert_eq!(
            require_git_success(&layout.workspace, &["log", "-1", "--pretty=%s"]).unwrap(),
            "chore: initialize harness workspace"
        );
    }

    #[test]
    fn default_harness_workspace_graph_discovers_repository_checks() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();
        std::fs::create_dir_all(layout.workspace.join("webui/src/pages")).unwrap();
        std::fs::write(
            layout.workspace.join("deno.json"),
            r#"{"tasks":{"build:web":"echo build","test:web":"echo test"}}"#,
        )
        .unwrap();
        std::fs::write(
            layout.workspace.join("webui/src/pages/ProjectsPage.tsx"),
            "export const projectPage = true;\n",
        )
        .unwrap();

        let discovered = resolve_harness_workspace_graph(&layout.workspace, None, None).unwrap();
        let plan = crate::checks::plan_checks_for_paths(
            &discovered,
            vec!["webui/src/pages/ProjectsPage.tsx".to_string()],
        )
        .unwrap();
        let commands = plan
            .checks
            .iter()
            .map(|id| discovered.checks[id].command.as_str())
            .collect::<Vec<_>>();
        assert!(commands.contains(&"deno task 'build:web'"));
        assert!(commands.contains(&"deno task 'test:web'"));

        let explicit = crate::workspace::WorkspaceGraph::legacy(&[]);
        let resolved =
            resolve_harness_workspace_graph(&layout.workspace, Some(explicit.clone()), None)
                .unwrap();
        assert_eq!(resolved, explicit);
    }

    #[test]
    fn existing_empty_scratch_directory_is_initialized_as_a_new_run() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        std::fs::create_dir(&root).unwrap();

        let layout = prepare_scratch(Some(&root)).unwrap();

        assert!(!layout.resumed);
        assert!(layout.workspace.join(".git").is_dir());
    }

    #[test]
    fn existing_non_harness_scratch_directory_is_rejected() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("unrelated.txt"), "keep\n").unwrap();

        let error = prepare_scratch(Some(&root)).unwrap_err();

        assert!(error.to_string().contains("has no git workspace"));
        assert_eq!(
            std::fs::read_to_string(root.join("unrelated.txt")).unwrap(),
            "keep\n"
        );
    }

    #[test]
    fn existing_scratch_workspace_can_be_resumed() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let initial = prepare_scratch(Some(&root)).unwrap();
        std::fs::write(initial.workspace.join("work.txt"), "in progress\n").unwrap();

        let resumed = prepare_scratch(Some(&root)).unwrap();

        assert!(resumed.resumed);
        assert_eq!(
            std::fs::read_to_string(resumed.workspace.join("work.txt")).unwrap(),
            "in progress\n"
        );
    }

    #[test]
    fn harness_checkpoint_round_trip_resumes_active_and_blocked_workflows_only() {
        let parent = tempfile::tempdir().unwrap();
        let layout = prepare_scratch(Some(&parent.path().join("run"))).unwrap();
        let repository = harness_repository_context(&layout).unwrap();
        let policy = crate::workflow::WorkflowConfigDocument::default()
            .compile()
            .unwrap();
        let run = crate::workflow::WorkflowRun::start(
            "workflow-harness-resume",
            "turn-harness-resume",
            "deliver safely",
            policy,
            repository,
        )
        .unwrap();
        let mut sink = HarnessEventSink::new(
            &layout.events,
            &layout.run_events,
            &layout.workflow_checkpoint,
        )
        .unwrap();

        let active = crate::workflow::WorkflowCheckpoint::new(run.clone()).unwrap();
        sink.checkpoint_workflow(&active).unwrap();
        assert_eq!(
            load_resumable_workflow_checkpoint(&layout.workflow_checkpoint)
                .unwrap()
                .unwrap(),
            active
        );

        let mut blocked_run = run.clone();
        blocked_run
            .apply(crate::workflow::WorkflowEvent::Blocked {
                outcome: crate::workflow::WorkflowOutcome::ExecutorUnavailable,
                cause: crate::workflow::WorkflowBlockCause::ExecutorUnavailable,
                reason: "configured executor is unavailable".to_string(),
            })
            .unwrap();
        let blocked = crate::workflow::WorkflowCheckpoint::new(blocked_run).unwrap();
        sink.checkpoint_workflow(&blocked).unwrap();
        assert_eq!(
            load_resumable_workflow_checkpoint(&layout.workflow_checkpoint)
                .unwrap()
                .unwrap(),
            blocked
        );

        let mut cancelled_run = run;
        cancelled_run
            .apply(crate::workflow::WorkflowEvent::Cancelled {
                reason: "cancelled".to_string(),
            })
            .unwrap();
        sink.checkpoint_workflow(&crate::workflow::WorkflowCheckpoint::new(cancelled_run).unwrap())
            .unwrap();
        assert!(
            load_resumable_workflow_checkpoint(&layout.workflow_checkpoint)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn active_workflow_resume_reuses_the_checkpoint_workspace_branch() {
        let parent = tempfile::tempdir().unwrap();
        let layout = prepare_scratch(Some(&parent.path().join("run"))).unwrap();
        require_git_success(
            &layout.workspace,
            &["checkout", "-b", "pb/interrupted-task"],
        )
        .unwrap();
        let repository = harness_repository_context(&layout).unwrap();
        let policy = crate::workflow::WorkflowConfigDocument::default()
            .compile()
            .unwrap();
        let run = crate::workflow::WorkflowRun::start(
            "workflow-interrupted-task",
            "turn-interrupted-task",
            "resume safely",
            policy,
            repository,
        )
        .unwrap();
        let checkpoint = crate::workflow::WorkflowCheckpoint::new(run).unwrap();

        assert_eq!(
            resumable_workflow_branch(&layout.workspace, Some(&checkpoint)).unwrap(),
            Some("pb/interrupted-task".to_string())
        );
        assert_eq!(
            resumable_workflow_branch(&layout.workspace, None).unwrap(),
            None
        );
    }

    #[test]
    fn resumed_scratch_preserves_task_baseline_and_records_adoption() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let first = prepare_scratch(Some(&root)).unwrap();
        let first_context = harness_repository_context(&first).unwrap();
        std::fs::write(first.workspace.join("work.txt"), "in progress\n").unwrap();

        let resumed = prepare_scratch(Some(&root)).unwrap();
        let resumed_context = harness_repository_context(&resumed).unwrap();

        assert_eq!(
            first_context.task_baseline.id,
            resumed_context.task_baseline.id
        );
        assert_ne!(
            first_context.invocation_baseline.id,
            resumed_context.invocation_baseline.id
        );
        assert_eq!(
            resumed_context.task_changed_paths().unwrap(),
            vec!["work.txt"]
        );
        let adoption = std::fs::read_to_string(&resumed.adoptions).unwrap();
        assert!(adoption.contains("resume_adoption"));
        assert!(adoption.contains("work.txt"));
    }

    #[test]
    fn prior_check_evidence_loads_from_cumulative_events() {
        let parent = tempfile::tempdir().unwrap();
        let events = parent.path().join("events.jsonl");
        let run_events = parent.path().join("run-events.jsonl");
        let mut sink = HarnessEventSink::new(
            &events,
            &run_events,
            &events.with_extension("checkpoint.json"),
        )
        .unwrap();
        sink.emit(AgentEvent::CheckResult {
            check_id: "logic".to_string(),
            exit_status: 0,
            success: true,
            timed_out: false,
            output: "ok".to_string(),
            truncated: false,
            duration_ms: 3,
            fingerprint: "inputs".to_string(),
            command: Some("true".to_string()),
            cwd: Some(".".to_string()),
            executor: Some("project".to_string()),
            source: Some("handoff".to_string()),
            command_fingerprint: Some("command".to_string()),
            dependency_outputs: Default::default(),
            output_fingerprint: Some("output".to_string()),
            reused: false,
            skip_reason: None,
            nesting_depth: None,
            timestamp_ms: None,
        });

        let ledger = load_check_evidence(&events).unwrap();
        assert_eq!(
            ledger.get("logic").unwrap().output_fingerprint.as_deref(),
            Some("output")
        );
    }

    #[test]
    fn workspace_config_metadata_is_external_stable_and_executor_aware() {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("workspace.toml");
        std::fs::write(
            &path,
            "version = 1\n\n[[executors]]\nid = \"local\"\nkind = \"local\"\n\n[[components]]\nid = \"app\"\nroot = \".\"\nexecutor = \"local\"\n",
        )
        .unwrap();
        let graph = crate::workspace::WorkspaceConfigDocument::from_path(&path)
            .unwrap()
            .normalize()
            .unwrap();
        let metadata = workspace_config_metadata(&path, &graph).unwrap();

        assert_eq!(metadata.sha256.len(), 64);
        assert_eq!(metadata.executor_policy, vec!["local:local"]);
        assert_eq!(
            metadata.source,
            path.canonicalize().unwrap().display().to_string()
        );

        let layout = prepare_scratch(Some(&parent.path().join("run"))).unwrap();
        assert!(!layout.workspace.join("workspace.toml").exists());
        assert!(!layout.workspace.join(".pb/workspace.toml").exists());
        let repository =
            crate::workspace::RepositoryContext::capture(&layout.workspace, &layout.workspace)
                .unwrap();
        assert!(repository.task_changed_paths().unwrap().is_empty());
    }

    #[test]
    fn workflow_config_metadata_is_external_hash_bound_and_not_copied() {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("workflow.toml");
        let bytes = b"version = 1\ndefault_intent = \"deliver\"\n";
        std::fs::write(&path, bytes).unwrap();
        let document = crate::workflow::WorkflowConfigDocument::from_path(&path).unwrap();
        let policy = document.clone().compile().unwrap();

        let metadata = workflow_config_metadata(Some(&path), &document, &policy).unwrap();

        assert_eq!(
            metadata.source,
            path.canonicalize().unwrap().display().to_string()
        );
        assert_eq!(
            metadata.source_sha256,
            format!("{:x}", Sha256::digest(bytes))
        );
        assert_eq!(metadata.policy_sha256, policy.sha256);
        assert_eq!(metadata.delivery, policy.delivery);
        assert_eq!(
            metadata.default_intent,
            crate::workflow::TurnIntent::Deliver
        );
        assert_eq!(metadata.limits, policy.limits);

        let layout = prepare_scratch(Some(&parent.path().join("run"))).unwrap();
        assert!(!layout.workspace.join("workflow.toml").exists());
        assert!(!layout.workspace.join(".pb/workflow.toml").exists());
        let repository =
            crate::workspace::RepositoryContext::capture(&layout.workspace, &layout.workspace)
                .unwrap();
        assert!(repository.task_changed_paths().unwrap().is_empty());
    }

    #[test]
    fn workflow_checkpoint_audit_and_config_persist_exactly() {
        let parent = tempfile::tempdir().unwrap();
        let layout = prepare_scratch(Some(&parent.path().join("run"))).unwrap();
        let repository = harness_repository_context(&layout).unwrap();
        let document = crate::workflow::WorkflowConfigDocument::default();
        let policy = document.clone().compile().unwrap();
        let metadata = workflow_config_metadata(None, &document, &policy).unwrap();
        let mut run = crate::workflow::WorkflowRun::start(
            "workflow-audit",
            "turn-audit",
            "deliver audited change",
            policy,
            repository,
        )
        .unwrap();
        let plan = crate::workflow::ArtifactEnvelope::new(
            "plan-audit",
            crate::workflow::PlanArtifact {
                summary: "Audited plan".to_string(),
                requirements: Vec::new(),
                steps: Vec::new(),
                acceptance: Vec::new(),
                risks: Vec::new(),
                assumptions: Vec::new(),
                open_questions: Vec::new(),
                resolved_challenge_ids: Vec::new(),
            },
        )
        .unwrap();
        let plan_review = crate::workflow::ArtifactEnvelope::new(
            "plan-review-audit",
            crate::workflow::PlanReviewArtifact {
                plan_id: plan.id.clone(),
                plan_sha256: plan.sha256.clone(),
                assessments: Vec::new(),
                challenges: Vec::new(),
                verdict: crate::workflow::ReviewVerdict::Pass,
            },
        )
        .unwrap();
        let code_review = crate::workflow::ArtifactEnvelope::new(
            "code-review-audit",
            crate::workflow::CodeReviewArtifact {
                content_fingerprint: "content-audit".to_string(),
                assessments: Vec::new(),
                findings: Vec::new(),
                verdict: crate::workflow::ReviewVerdict::Pass,
            },
        )
        .unwrap();
        run.plan = Some(plan.clone());
        run.plan_review = Some(plan_review.clone());
        run.code_review = Some(code_review.clone());
        run.content_fingerprint = Some("content-audit".to_string());
        run.stage = crate::workflow::WorkflowStage::Ready;
        run.outcome = Some(crate::workflow::WorkflowOutcome::Ready);
        run.commit = Some(crate::events::HandoffCommitSummary {
            oid: "audit-commit".to_string(),
            subject: "feat: audited delivery".to_string(),
        });
        run.ready_evidence = Some(
            crate::workflow::ReadyEvidenceBundle::from_run(
                &run,
                Some("git@example.test:team/project.git".to_string()),
            )
            .unwrap(),
        );
        run.counters.plan_cycles = 1;
        run.counters.repair_cycles = 2;
        run.counters.model_invocations = 9;
        run.counters.generated_tokens = 1234;
        run.counters.advisory_calls = 3;
        run.counters
            .stage_steps
            .insert(crate::workflow::WorkflowStage::Planning, 2);
        run.counters
            .stage_steps
            .insert(crate::workflow::WorkflowStage::Repairing, 4);
        let checkpoint = crate::workflow::WorkflowCheckpoint::new(run.clone()).unwrap();
        let mut sink = HarnessEventSink::new(
            &layout.events,
            &layout.run_events,
            &layout.workflow_checkpoint,
        )
        .unwrap();
        sink.emit(AgentEvent::HarnessExperimentConfigured {
            observation_rendering: crate::workflow::ObservationRendering::ControllerBlock,
            timestamp_ms: None,
        });
        sink.emit(AgentEvent::ControllerObservation {
            receipt: crate::workflow::ControllerObservationReceipt {
                version: crate::workflow::CONTROLLER_OBSERVATION_SCHEMA_VERSION,
                action_id: "controller-observation-audit".to_string(),
                actual_origin: crate::workflow::ControllerObservationOrigin::Controller,
                prompt_representation: crate::workflow::ObservationRendering::ControllerBlock,
                stage: crate::workflow::WorkflowStage::CodeReview,
                work_unit_id: None,
                operation: crate::workflow::ControllerObservationOperation::InspectChange,
                path: "average.mjs".to_string(),
                workspace_fingerprint: "a".repeat(64),
                path_fingerprint: "b".repeat(64),
                content_sha256: "c".repeat(64),
                coverage: crate::workflow::ObservationCoverage::Full,
                observed_bytes: 40,
                prompt_bytes: 64,
                included_ranges: vec![crate::workflow::ObservationRange {
                    start_byte: 0,
                    end_byte: 40,
                    sha256: "d".repeat(64),
                }],
                included_in_prompt: true,
                authority_effects: vec![
                    crate::workflow::ControllerObservationAuthority::PromptContext,
                    crate::workflow::ControllerObservationAuthority::ReviewCoverage,
                ],
                fallback_reason: None,
            },
            actor: crate::events::TeamActor::workflow_steward(),
            assisting_profile: Some(crate::agent_core::AgentProfile::Build),
            nesting_depth: None,
            timestamp_ms: None,
        });
        for stage in [
            crate::workflow::WorkflowStage::Planning,
            crate::workflow::WorkflowStage::PlanReview,
            crate::workflow::WorkflowStage::Implementing,
            crate::workflow::WorkflowStage::Checking,
            crate::workflow::WorkflowStage::CodeReview,
            crate::workflow::WorkflowStage::Committing,
        ] {
            sink.emit(AgentEvent::WorkflowStageStarted {
                workflow_id: run.id.clone(),
                stage,
                timestamp_ms: None,
            });
        }
        sink.emit(AgentEvent::Correction {
            kind: crate::events::CorrectionKind::ToolUnavailable,
            message: "mutation denied".to_string(),
            summary: "Tool not available".to_string(),
            actor: crate::events::TeamActor::workflow_steward(),
            assisting_profile: Some(crate::agent_core::AgentProfile::Build),
            nesting_depth: None,
            timestamp_ms: None,
        });
        sink.emit(AgentEvent::WorkflowEvidenceInvalidated {
            workflow_id: run.id.clone(),
            previous_fingerprint: "before".to_string(),
            current_fingerprint: "after".to_string(),
            reason: "mutation".to_string(),
            timestamp_ms: None,
        });
        sink.checkpoint_workflow(&checkpoint).unwrap();

        let persisted: crate::workflow::WorkflowCheckpoint =
            serde_json::from_slice(&std::fs::read(&layout.workflow_checkpoint).unwrap()).unwrap();
        assert_eq!(persisted, checkpoint);
        let (_, summary, audit) = sink.snapshot().unwrap();
        assert!(audit.strict_workflow);
        assert!(audit.strict_workflow_satisfied);
        assert_eq!(audit.workflow_id.as_deref(), Some("workflow-audit"));
        assert_eq!(
            audit.workflow_outcome,
            Some(crate::workflow::WorkflowOutcome::Ready)
        );
        assert_eq!(
            audit.workflow_checkpoint_sha256.as_deref(),
            Some(checkpoint.sha256.as_str())
        );
        assert_eq!(
            audit.ready_evidence_sha256.as_deref(),
            Some(
                run.ready_evidence
                    .as_ref()
                    .unwrap()
                    .sha256()
                    .unwrap()
                    .as_str()
            )
        );
        assert_eq!(
            audit.repository_remote.as_deref(),
            Some("git@example.test:team/project.git")
        );
        assert_eq!(audit.plan_sha256.as_deref(), Some(plan.sha256.as_str()));
        assert_eq!(
            audit.plan_review_sha256.as_deref(),
            Some(plan_review.sha256.as_str())
        );
        assert_eq!(
            audit.code_review_sha256.as_deref(),
            Some(code_review.sha256.as_str())
        );
        assert_eq!(audit.plan_cycles, 1);
        assert_eq!(audit.repair_cycles, 2);
        assert_eq!(audit.workflow_model_invocations, 9);
        assert_eq!(audit.workflow_generated_tokens, 1234);
        assert_eq!(audit.workflow_advisory_calls, 3);
        assert_eq!(audit.workflow_stage_steps, run.counters.stage_steps);
        assert_eq!(audit.rejected_workflow_actions, 1);
        assert_eq!(audit.evidence_invalidations, 1);
        assert_eq!(
            audit.observation_rendering,
            Some(crate::workflow::ObservationRendering::ControllerBlock)
        );
        assert_eq!(audit.controller_observations, 1);
        assert_eq!(audit.controller_observation_prompt_bytes, 64);

        let result = Ok(AgentRunResult {
            branch: "main".to_string(),
            workspace_root: layout.workspace.clone(),
            focus_root: layout.workspace.clone(),
            repository_context: Some(run.repository.clone()),
            workspace_graph: None,
            reached_final: true,
            contract_status: crate::events::ContractStatus::Unspecified,
            verified_completed: false,
            termination_reason: crate::events::TerminationReason::Final,
            handoff_outcome: Some(crate::events::HandoffOutcome::Ready),
            workflow: Some(checkpoint.clone()),
            delivery_proposal: None,
            requested_delivery: None,
            goal_proposal: None,
            requested_goal: None,
            task_goal: None,
            task_plan_rejected: None,
            task_planning_transcript: None,
        });
        append_run_index_started(&layout, &run.task, None, &metadata).unwrap();
        append_run_index_finished(&layout, &result, &audit, None, &metadata).unwrap();
        write_journal(
            &layout,
            &run.task,
            &result,
            &summary,
            &audit,
            None,
            &metadata,
            &mut Vec::new(),
        )
        .unwrap();

        let records = std::fs::read_to_string(&layout.run_index).unwrap();
        let records = records
            .lines()
            .map(|line| serde_json::from_str::<RunIndexRecord>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(matches!(
            &records[0],
            RunIndexRecord::Started { workflow_config, .. } if workflow_config == &metadata
        ));
        assert!(matches!(
            &records[1],
            RunIndexRecord::Finished { workflow_config, audit: stored, .. }
                if workflow_config == &metadata && stored == &audit
        ));
        let journal = std::fs::read_to_string(&layout.run_journal).unwrap();
        assert!(journal.contains("Strict workflow enabled/satisfied: `true` / `true`"));
        assert!(journal.contains(&checkpoint.sha256));
        assert!(journal.contains(&plan.sha256));
        assert!(
            journal.contains(
                "Model invocations / generated tokens / advisory calls: `9` / `1234` / `3`"
            )
        );
        assert!(journal.contains("Rejected workflow actions: `1`"));
        assert!(journal.contains("Evidence invalidations: `1`"));
        assert!(journal.contains("## Deterministic controller actions"));
        assert!(journal.contains("Observation rendering: `controller_block`"));
        assert!(journal.contains("Observations / prompt bytes: `1` / `64`"));
        assert!(journal.contains("Observation coverage: `{"));
        assert!(journal.contains("Controller mutations / closures: `0` / `0`"));
        assert!(journal.contains("builtin:strict-default"));
        assert!(journal.contains(&metadata.policy_sha256));
    }

    #[test]
    fn valid_no_change_does_not_report_a_missing_commit_problem() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();
        let result = Ok(AgentRunResult {
            branch: "main".to_string(),
            workspace_root: layout.workspace.clone(),
            focus_root: layout.workspace.clone(),
            repository_context: None,
            workspace_graph: None,
            reached_final: true,
            contract_status: crate::events::ContractStatus::Unspecified,
            verified_completed: false,
            termination_reason: crate::events::TerminationReason::Final,
            handoff_outcome: Some(crate::events::HandoffOutcome::NoChange),
            workflow: None,
            delivery_proposal: None,
            requested_delivery: None,
            goal_proposal: None,
            requested_goal: None,
            task_goal: None,
            task_plan_rejected: None,
            task_planning_transcript: None,
        });
        let summary = CapturedSummary {
            summary: "No changes were needed.".to_string(),
            ..CapturedSummary::default()
        };
        let mut observations = Vec::new();

        add_run_observations(&mut observations, &result, &layout.workspace, &summary);

        assert!(
            !observations
                .iter()
                .any(|observation| observation.title == "completed run produced no commits")
        );
    }

    #[test]
    fn discussion_does_not_report_a_missing_commit_problem() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();
        let result = Ok(AgentRunResult {
            branch: "main".to_string(),
            workspace_root: layout.workspace.clone(),
            focus_root: layout.workspace.clone(),
            repository_context: None,
            workspace_graph: None,
            reached_final: true,
            contract_status: crate::events::ContractStatus::Unspecified,
            verified_completed: false,
            termination_reason: crate::events::TerminationReason::Final,
            handoff_outcome: None,
            workflow: None,
            delivery_proposal: None,
            requested_delivery: None,
            goal_proposal: None,
            requested_goal: None,
            task_goal: None,
            task_plan_rejected: None,
            task_planning_transcript: None,
        });
        let summary = CapturedSummary {
            summary: "Discussion answer.".to_string(),
            ..CapturedSummary::default()
        };
        let mut observations = Vec::new();

        add_run_observations(&mut observations, &result, &layout.workspace, &summary);

        assert!(
            !observations
                .iter()
                .any(|observation| observation.title == "completed run produced no commits")
        );
    }

    #[test]
    fn incomplete_delivery_does_not_report_a_missing_commit_problem() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();
        let result = Ok(AgentRunResult {
            branch: "main".to_string(),
            workspace_root: layout.workspace.clone(),
            focus_root: layout.workspace.clone(),
            repository_context: None,
            workspace_graph: None,
            reached_final: true,
            contract_status: crate::events::ContractStatus::Unspecified,
            verified_completed: false,
            termination_reason: crate::events::TerminationReason::EngineError,
            handoff_outcome: Some(crate::events::HandoffOutcome::Incomplete),
            workflow: None,
            delivery_proposal: None,
            requested_delivery: None,
            goal_proposal: None,
            requested_goal: None,
            task_goal: None,
            task_plan_rejected: None,
            task_planning_transcript: None,
        });
        let summary = CapturedSummary {
            summary: "Model setup failed before delivery began.".to_string(),
            ..CapturedSummary::default()
        };
        let mut observations = Vec::new();

        add_run_observations(&mut observations, &result, &layout.workspace, &summary);

        assert!(
            !observations
                .iter()
                .any(|observation| observation.title == "completed run produced no commits")
        );
    }

    #[test]
    fn incomplete_delivery_classifies_preserved_dirty_state_as_model_limitation() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();
        std::fs::write(layout.workspace.join("partial.txt"), "partial\n").unwrap();
        let result = Ok(AgentRunResult {
            branch: "main".to_string(),
            workspace_root: layout.workspace.clone(),
            focus_root: layout.workspace.clone(),
            repository_context: None,
            workspace_graph: None,
            reached_final: true,
            contract_status: crate::events::ContractStatus::Unspecified,
            verified_completed: false,
            termination_reason: crate::events::TerminationReason::StepLimit,
            handoff_outcome: Some(crate::events::HandoffOutcome::Incomplete),
            workflow: None,
            delivery_proposal: None,
            requested_delivery: None,
            goal_proposal: None,
            requested_goal: None,
            task_goal: None,
            task_plan_rejected: None,
            task_planning_transcript: None,
        });
        let summary = CapturedSummary {
            summary: "Implementation stopped at its step limit.".to_string(),
            ..CapturedSummary::default()
        };
        let mut observations = Vec::new();

        add_run_observations(&mut observations, &result, &layout.workspace, &summary);

        let dirty = observations
            .iter()
            .find(|observation| {
                observation.title == "incomplete workspace preserves uncommitted changes"
            })
            .expect("preserved incomplete workspace observation");
        assert_eq!(dirty.classification, "model_limitation");
        assert!(
            observations
                .iter()
                .all(|observation| observation.title != "workspace has uncommitted changes")
        );
    }

    #[test]
    fn compact_detail_is_single_line_and_bounded() {
        let detail = compact_detail(&format!("first\n{}", "x".repeat(900)));
        assert!(!detail.contains('\n'));
        assert!(detail.chars().count() <= 801);
        assert!(detail.ends_with('…'));
    }

    #[test]
    fn event_journal_captures_started_branch_before_failures() {
        let parent = tempfile::tempdir().unwrap();
        let events = parent.path().join("events.jsonl");
        let run_events = parent.path().join("run-events.jsonl");
        let sink = HarnessEventSink::new(
            &events,
            &run_events,
            &events.with_extension("checkpoint.json"),
        )
        .unwrap();
        let mut emitter = sink.clone();
        emitter.emit(AgentEvent::Started {
            task: "task".to_string(),
            model: "model".to_string(),
            profile: crate::agent_core::AgentProfile::Build,
            workspace: "/tmp/workspace".to_string(),
            focus_root: Some("/tmp/workspace".to_string()),
            branch: "pb/task-harness-1".to_string(),
            attachments: Vec::new(),
            timestamp_ms: None,
        });

        assert_eq!(std::fs::read_to_string(&events).unwrap().lines().count(), 1);
        let (_, summary, _) = sink.snapshot().unwrap();
        assert_eq!(summary.branch, "pb/task-harness-1");
        assert_eq!(std::fs::read_to_string(events).unwrap().lines().count(), 1);
        assert_eq!(
            std::fs::read_to_string(run_events).unwrap().lines().count(),
            1
        );
    }

    #[test]
    fn trusted_goal_context_is_validated_and_requests_are_audited() {
        let context_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/harness-goal-context.json");
        let goal = load_goal_context(&context_path).unwrap();
        let parent = tempfile::tempdir().unwrap();
        let events = parent.path().join("events.jsonl");
        let run_events = parent.path().join("run-events.jsonl");
        let mut sink = HarnessEventSink::new(
            &events,
            &run_events,
            &events.with_extension("checkpoint.json"),
        )
        .unwrap();
        sink.configure_goal_context(&goal).unwrap();
        sink.request_goal_pause("goal-harness-g8: inspect evidence")
            .unwrap();
        sink.request_goal_change(
            crate::events::GoalChangeKind::Amendment,
            "goal-harness-g8: narrow the remaining scope",
        )
        .unwrap();
        sink.request_goal_change(
            crate::events::GoalChangeKind::Budget,
            "goal-harness-g8: request ten more turns",
        )
        .unwrap();

        let (_, _, audit) = sink.snapshot().unwrap();
        assert_eq!(audit.goal_id.as_deref(), Some("goal-harness-g8"));
        assert_eq!(
            audit.goal_stage,
            Some(crate::goal::GoalStage::RunningMilestone)
        );
        assert_eq!(audit.goal_completed_milestones, 2);
        assert_eq!(audit.goal_pause_requests, 1);
        assert_eq!(audit.goal_amendment_requests, 1);
        assert_eq!(audit.goal_budget_requests, 1);
        let journal = std::fs::read_to_string(run_events).unwrap();
        assert!(journal.contains("goal_pause_requested"));
        assert!(journal.contains("goal_change_requested"));
    }

    #[test]
    fn structured_parse_failures_are_classified_as_model_limitations() {
        let parent = tempfile::tempdir().unwrap();
        let events = parent.path().join("events.jsonl");
        let run_events = parent.path().join("run-events.jsonl");
        let mut sink = HarnessEventSink::new(
            &events,
            &run_events,
            &events.with_extension("checkpoint.json"),
        )
        .unwrap();
        sink.emit(AgentEvent::Error {
            summary: "Invalid pb JSON action on step 1/3".to_string(),
            detail: "model returned prose instead of a bounded action".to_string(),
            nesting_depth: None,
            timestamp_ms: None,
        });
        sink.emit(AgentEvent::Error {
            summary: "Parse retry limit reached".to_string(),
            detail: "three equivalent parse failures".to_string(),
            nesting_depth: None,
            timestamp_ms: None,
        });

        let (observations, _, _) = sink.snapshot().unwrap();
        assert_eq!(observations.len(), 2);
        assert!(
            observations
                .iter()
                .all(|item| item.classification == "model_limitation")
        );
    }

    #[test]
    fn intentional_goal_handoff_does_not_require_a_session_summary() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();
        let result = Ok(AgentRunResult {
            branch: "main".to_string(),
            workspace_root: layout.workspace.clone(),
            focus_root: layout.workspace.clone(),
            repository_context: None,
            workspace_graph: None,
            reached_final: true,
            contract_status: crate::events::ContractStatus::Unspecified,
            verified_completed: false,
            termination_reason: crate::events::TerminationReason::Final,
            handoff_outcome: None,
            workflow: None,
            delivery_proposal: None,
            requested_delivery: None,
            goal_proposal: None,
            requested_goal: Some(crate::goal::GoalProposal {
                id: "proposal".to_string(),
                source_turn_id: "turn".to_string(),
                objective: "Qualify Goal mode".to_string(),
                criteria: Vec::new(),
            }),
            task_goal: None,
            task_plan_rejected: None,
            task_planning_transcript: None,
        });
        let mut observations = Vec::new();

        add_run_observations(
            &mut observations,
            &result,
            &layout.workspace,
            &CapturedSummary::default(),
        );

        assert!(
            observations
                .iter()
                .all(|item| item.title != "agent emitted no session summary")
        );
    }

    #[test]
    fn event_journal_summarizes_handoff_checks_team_feedback_and_commit() {
        let parent = tempfile::tempdir().unwrap();
        let events = parent.path().join("events.jsonl");
        let run_events = parent.path().join("run-events.jsonl");
        let mut sink = HarnessEventSink::new(
            &events,
            &run_events,
            &events.with_extension("checkpoint.json"),
        )
        .unwrap();
        sink.emit(AgentEvent::ExecutorStarted {
            executor_id: "web".to_string(),
            kind: "local".to_string(),
            success: true,
            detail: String::new(),
            timestamp_ms: None,
        });
        sink.emit(AgentEvent::TeamMessage {
            actor: crate::events::TeamActor::workflow_steward(),
            tone: crate::events::TeamMessageTone::Warning,
            purpose: crate::events::TeamMessagePurpose::General,
            handoff: None,
            message: "The web check needs attention.".to_string(),
            detail: Some("failed".to_string()),
            evidence: vec![EvidenceRef::Check {
                check_id: "web".to_string(),
            }],
            nesting_depth: None,
            timestamp_ms: None,
        });
        sink.emit(AgentEvent::HandoffSummary {
            summary: crate::events::HandoffSummary {
                outcome: crate::events::HandoffOutcome::Ready,
                affected_components: vec!["web".to_string()],
                checks: vec![crate::events::HandoffCheckSummary {
                    check_id: "web".to_string(),
                    status: "passed".to_string(),
                }],
                commit: None,
                changed_paths: vec!["web/app.ts".to_string()],
                detail: None,
            },
            nesting_depth: None,
            timestamp_ms: None,
        });
        sink.emit(AgentEvent::CommitResult {
            success: true,
            created: true,
            reused: false,
            oid: Some("abc".to_string()),
            subject: Some("feat: web".to_string()),
            changed_paths: vec!["web/app.ts".to_string()],
            detail: String::new(),
            nesting_depth: None,
            timestamp_ms: None,
        });

        let (_, _, audit) = sink.snapshot().unwrap();
        assert_eq!(audit.affected_components, vec!["web"]);
        assert_eq!(audit.checks_planned, vec!["web"]);
        assert_eq!(audit.executors_started, vec!["web"]);
        assert_eq!(audit.team_messages, 1);
        assert_eq!(audit.feedback_evidence_ids, vec!["check:web"]);
        assert_eq!(audit.commit_disposition.as_deref(), Some("created"));
        assert_eq!(audit.commit_oid.as_deref(), Some("abc"));
    }

    #[test]
    fn strict_check_and_work_unit_guidance_are_reported_truthfully() {
        let parent = tempfile::tempdir().unwrap();
        let events = parent.path().join("events.jsonl");
        let run_events = parent.path().join("run-events.jsonl");
        let mut sink = HarnessEventSink::new(
            &events,
            &run_events,
            &events.with_extension("checkpoint.json"),
        )
        .unwrap();
        sink.emit(AgentEvent::Correction {
            kind: crate::events::CorrectionKind::WorkUnit,
            message: "create alpha.txt now".to_string(),
            summary: "Next accepted-plan creation work unit".to_string(),
            actor: crate::events::TeamActor::workflow_steward(),
            assisting_profile: Some(crate::agent_core::AgentProfile::Build),
            nesting_depth: None,
            timestamp_ms: None,
        });
        sink.emit(AgentEvent::CheckResult {
            check_id: "exact_content".to_string(),
            exit_status: 0,
            success: true,
            timed_out: false,
            output: String::new(),
            truncated: false,
            duration_ms: 1,
            fingerprint: "content".to_string(),
            command: Some("test -f alpha.txt".to_string()),
            cwd: Some(".".to_string()),
            executor: Some("harness-contract".to_string()),
            source: Some("executed".to_string()),
            command_fingerprint: Some("command".to_string()),
            dependency_outputs: BTreeMap::new(),
            output_fingerprint: Some("output".to_string()),
            reused: false,
            skip_reason: None,
            nesting_depth: None,
            timestamp_ms: None,
        });

        let (observations, _, audit) = sink.snapshot().unwrap();
        assert_eq!(audit.checks_planned, vec!["exact_content"]);
        assert_eq!(audit.checks_executed, 1);
        assert_eq!(audit.checks_passed, 1);
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].classification, "positive_evidence");
    }

    #[test]
    fn successful_controller_corrections_are_positive_evidence() {
        let parent = tempfile::tempdir().unwrap();
        let events = parent.path().join("events.jsonl");
        let run_events = parent.path().join("run-events.jsonl");
        let mut sink = HarnessEventSink::new(
            &events,
            &run_events,
            &events.with_extension("checkpoint.json"),
        )
        .unwrap();
        for (kind, summary, message) in [
            (
                crate::events::CorrectionKind::WorkUnit,
                "Active accepted-plan work unit",
                "Harness work unit path:0: create slug.js now",
            ),
            (
                crate::events::CorrectionKind::WorkUnit,
                "Work-unit progress earned one bounded turn",
                "Harness progress credit: work unit path:0 produced a new transition",
            ),
            (
                crate::events::CorrectionKind::Diagnostics,
                "Harness diagnostic preview",
                "Harness diagnostic preview passed. This grants no final evidence.",
            ),
        ] {
            sink.emit(AgentEvent::Correction {
                kind,
                message: message.to_string(),
                summary: summary.to_string(),
                actor: crate::events::TeamActor::workflow_steward(),
                assisting_profile: Some(crate::agent_core::AgentProfile::Build),
                nesting_depth: None,
                timestamp_ms: None,
            });
        }

        let (observations, _, _) = sink.snapshot().unwrap();
        assert_eq!(observations.len(), 3);
        assert!(
            observations
                .iter()
                .all(|observation| observation.classification == "positive_evidence")
        );
        assert_eq!(
            correction_classification(
                "Harness diagnostic preview failed and grants no final evidence.",
                "Harness diagnostic preview"
            ),
            "model_limitation"
        );
    }

    #[test]
    fn legacy_run_index_records_deserialize_with_empty_audit_defaults() {
        let record: RunIndexRecord = serde_json::from_str(
            r#"{"state":"finished","version":1,"run_id":"old","timestamp_ms":1,"status":"final_unverified","reached_final":true,"contract_status":"unspecified","verified_completed":false,"termination_reason":"final"}"#,
        )
        .unwrap();
        let RunIndexRecord::Finished {
            handoff_outcome,
            audit,
            ..
        } = record
        else {
            panic!("expected finished record");
        };
        assert_eq!(handoff_outcome, None);
        assert_eq!(audit, HarnessRunAudit::default());
    }

    #[test]
    fn running_journal_exists_before_agent_completion() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();
        write_running_journal(
            &layout,
            "Build a test project",
            None,
            &WorkflowConfigMetadata::default(),
        )
        .unwrap();

        let journal = std::fs::read_to_string(layout.journal).unwrap();
        assert!(journal.contains("Status: `running`"));
        assert!(journal.contains("P2 — experiment_error — run has not finalized"));
        assert!(journal.contains("Build a test project"));
    }

    fn finish_test_run(layout: &ScratchLayout, task: &str, branch: &str) {
        write_running_journal(layout, task, None, &WorkflowConfigMetadata::default()).unwrap();
        append_run_index_started(layout, task, None, &WorkflowConfigMetadata::default()).unwrap();
        let sink = HarnessEventSink::new(
            &layout.events,
            &layout.run_events,
            &layout.workflow_checkpoint,
        )
        .unwrap();
        let mut emitter = sink.clone();
        emitter.emit(AgentEvent::Started {
            task: task.to_string(),
            model: "scripted".to_string(),
            profile: crate::agent_core::AgentProfile::Build,
            workspace: layout.workspace.display().to_string(),
            focus_root: Some(layout.workspace.display().to_string()),
            branch: branch.to_string(),
            attachments: Vec::new(),
            timestamp_ms: None,
        });
        let (_, summary, audit) = sink.snapshot().unwrap();
        let result = Ok(AgentRunResult {
            branch: branch.to_string(),
            workspace_root: layout.workspace.clone(),
            focus_root: layout.workspace.clone(),
            repository_context: None,
            workspace_graph: None,
            reached_final: true,
            contract_status: crate::events::ContractStatus::Unspecified,
            verified_completed: false,
            termination_reason: crate::events::TerminationReason::Final,
            handoff_outcome: Some(crate::events::HandoffOutcome::Ready),
            workflow: None,
            delivery_proposal: None,
            requested_delivery: None,
            goal_proposal: None,
            requested_goal: None,
            task_goal: None,
            task_plan_rejected: None,
            task_planning_transcript: None,
        });
        write_journal(
            layout,
            task,
            &result,
            &summary,
            &audit,
            None,
            &WorkflowConfigMetadata::default(),
            &mut Vec::new(),
        )
        .unwrap();
        append_run_index_finished(
            layout,
            &result,
            &audit,
            None,
            &WorkflowConfigMetadata::default(),
        )
        .unwrap();
    }

    #[test]
    fn resumed_invocations_preserve_immutable_per_run_audits() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let first = prepare_scratch(Some(&root)).unwrap();
        finish_test_run(&first, "first task", "task-first");
        let first_events = std::fs::read(&first.run_events).unwrap();
        let first_journal = std::fs::read(&first.run_journal).unwrap();

        let second = prepare_scratch(Some(&root)).unwrap();
        finish_test_run(&second, "second task", "task-second");

        assert_ne!(first.run_id, second.run_id);
        assert_eq!(std::fs::read(&first.run_events).unwrap(), first_events);
        assert_eq!(std::fs::read(&first.run_journal).unwrap(), first_journal);
        assert_eq!(
            std::fs::read_to_string(&second.events)
                .unwrap()
                .lines()
                .count(),
            2
        );
        assert_eq!(
            std::fs::read_to_string(&second.run_index)
                .unwrap()
                .lines()
                .count(),
            4
        );
        let latest = std::fs::read_to_string(&second.journal).unwrap();
        assert!(latest.contains(&format!("Run ID: `{}`", second.run_id)));
        assert!(latest.contains("second task"));
    }

    #[test]
    fn interrupted_run_is_discoverable_from_started_index_record() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();
        write_running_journal(
            &layout,
            "interrupted task",
            None,
            &WorkflowConfigMetadata::default(),
        )
        .unwrap();
        append_run_index_started(
            &layout,
            "interrupted task",
            None,
            &WorkflowConfigMetadata::default(),
        )
        .unwrap();

        let records = std::fs::read_to_string(&layout.run_index).unwrap();
        let records = records
            .lines()
            .map(|line| serde_json::from_str::<RunIndexRecord>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 1);
        assert!(matches!(
            &records[0],
            RunIndexRecord::Started { run_id, task, .. }
                if run_id == &layout.run_id && task == "interrupted task"
        ));
        let journal = std::fs::read_to_string(&layout.run_journal).unwrap();
        assert!(journal.contains("Status: `running`"));
        assert!(journal.contains(&format!("Run ID: `{}`", layout.run_id)));
        assert!(!layout.run_events.exists());
    }
}
