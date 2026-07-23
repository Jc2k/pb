use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::agent_core::{AgentProfile, SessionAttachment};
use crate::session_store::now_millis;

pub const EVENT_SCHEMA_VERSION: &str = "v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutcome {
    Succeeded,
    Failed,
    Rejected,
    TimedOut,
    Cancelled,
    CacheReplay,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractStatus {
    #[default]
    Unspecified,
    Unsatisfied,
    Satisfied,
}

impl ContractStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::Unsatisfied => "unsatisfied",
            Self::Satisfied => "satisfied",
        }
    }
}

impl std::fmt::Display for ContractStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminationReason {
    Final,
    StepLimit,
    GateLoop,
    ParseLoop,
    ContractUnsatisfied,
    ContextLimit,
    ResourceLimit,
    InvocationLimit,
    TokenLimit,
    EngineError,
    ChecksFailed,
    ExecutorUnavailable,
    RepairExhausted,
    CommitBlocked,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinalGraceStatus {
    Started,
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutomationActor {
    /// Legacy durable identity used by sessions written before automation was
    /// presented as a stable teammate.
    Handoff,
    Trinity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum TeamActor {
    Agent(AgentProfile),
    Automation(AutomationActor),
}

impl TeamActor {
    pub const fn agent(profile: AgentProfile) -> Self {
        Self::Agent(profile)
    }

    pub const fn workflow_steward() -> Self {
        Self::Automation(AutomationActor::Trinity)
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Agent(profile) => profile.teammate_name(),
            Self::Automation(AutomationActor::Handoff | AutomationActor::Trinity) => {
                "Trinity Walker"
            }
        }
    }
}

fn workflow_steward_actor() -> TeamActor {
    TeamActor::workflow_steward()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeamMessageTone {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffOutcome {
    Pending,
    Ready,
    NoChange,
    ChecksFailed,
    ExecutorUnavailable,
    CommitBlocked,
    RepairExhausted,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandoffCheckSummary {
    pub check_id: String,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandoffCommitSummary {
    pub oid: String,
    pub subject: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandoffSummary {
    pub outcome: HandoffOutcome,
    #[serde(default)]
    pub affected_components: Vec<String>,
    #[serde(default)]
    pub checks: Vec<HandoffCheckSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<HandoffCommitSummary>,
    #[serde(default)]
    pub changed_paths: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl TerminationReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Final => "final",
            Self::StepLimit => "step_limit",
            Self::GateLoop => "gate_loop",
            Self::ParseLoop => "parse_loop",
            Self::ContractUnsatisfied => "contract_unsatisfied",
            Self::ContextLimit => "context_limit",
            Self::ResourceLimit => "resource_limit",
            Self::InvocationLimit => "invocation_limit",
            Self::TokenLimit => "token_limit",
            Self::EngineError => "engine_error",
            Self::ChecksFailed => "checks_failed",
            Self::ExecutorUnavailable => "executor_unavailable",
            Self::RepairExhausted => "repair_exhausted",
            Self::CommitBlocked => "commit_blocked",
            Self::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for TerminationReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMetricsSnapshot {
    pub llm_invocations: usize,
    pub llm_runtime_ms: u64,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub tool_calls: usize,
    pub tool_runtime_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_energy_joules: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_energy_kwh: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_energy_joules: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_energy_kwh: Option<f64>,
    #[serde(default)]
    pub wall_runtime_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_energy_joules: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_energy_kwh: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gross_energy_joules: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjusted_energy_joules: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_power_watts: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_measured_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_coverage: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_source: Option<String>,
    #[serde(default)]
    pub display_energy_excluded: bool,
    #[serde(default)]
    pub idle_baseline_applied: bool,
    #[serde(default)]
    pub energy_complete: bool,
    #[serde(default)]
    pub energy_exclusive: bool,
}

impl SessionMetricsSnapshot {
    pub fn from_event(event: &AgentEvent) -> Option<Self> {
        if let AgentEvent::SessionMetrics {
            llm_invocations,
            llm_runtime_ms,
            prompt_tokens,
            generated_tokens,
            tool_calls,
            tool_runtime_ms,
            llm_energy_joules,
            llm_energy_kwh,
            tool_energy_joules,
            tool_energy_kwh,
            wall_runtime_ms,
            started_at_ms,
            ended_at_ms,
            total_energy_joules,
            total_energy_kwh,
            gross_energy_joules,
            adjusted_energy_joules,
            average_power_watts,
            energy_measured_ms,
            energy_coverage,
            energy_source,
            display_energy_excluded,
            idle_baseline_applied,
            energy_complete,
            energy_exclusive,
            ..
        } = event
        {
            Some(Self {
                llm_invocations: *llm_invocations,
                llm_runtime_ms: *llm_runtime_ms,
                prompt_tokens: *prompt_tokens,
                generated_tokens: *generated_tokens,
                tool_calls: *tool_calls,
                tool_runtime_ms: *tool_runtime_ms,
                llm_energy_joules: *llm_energy_joules,
                llm_energy_kwh: *llm_energy_kwh,
                tool_energy_joules: *tool_energy_joules,
                tool_energy_kwh: *tool_energy_kwh,
                wall_runtime_ms: *wall_runtime_ms,
                started_at_ms: *started_at_ms,
                ended_at_ms: *ended_at_ms,
                total_energy_joules: *total_energy_joules,
                total_energy_kwh: *total_energy_kwh,
                gross_energy_joules: *gross_energy_joules,
                adjusted_energy_joules: *adjusted_energy_joules,
                average_power_watts: *average_power_watts,
                energy_measured_ms: *energy_measured_ms,
                energy_coverage: *energy_coverage,
                energy_source: energy_source.clone(),
                display_energy_excluded: *display_energy_excluded,
                idle_baseline_applied: *idle_baseline_applied,
                energy_complete: *energy_complete,
                energy_exclusive: *energy_exclusive,
            })
        } else {
            None
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        self.llm_invocations = self.llm_invocations.saturating_add(other.llm_invocations);
        self.llm_runtime_ms = self.llm_runtime_ms.saturating_add(other.llm_runtime_ms);
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.generated_tokens = self.generated_tokens.saturating_add(other.generated_tokens);
        self.tool_calls = self.tool_calls.saturating_add(other.tool_calls);
        self.tool_runtime_ms = self.tool_runtime_ms.saturating_add(other.tool_runtime_ms);
        self.wall_runtime_ms = self.wall_runtime_ms.saturating_add(other.wall_runtime_ms);
        self.started_at_ms = match (self.started_at_ms, other.started_at_ms) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        };
        self.ended_at_ms = match (self.ended_at_ms, other.ended_at_ms) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (left, right) => left.or(right),
        };
        add_optional(&mut self.llm_energy_joules, other.llm_energy_joules);
        add_optional(&mut self.llm_energy_kwh, other.llm_energy_kwh);
        add_optional(&mut self.tool_energy_joules, other.tool_energy_joules);
        add_optional(&mut self.tool_energy_kwh, other.tool_energy_kwh);
        add_optional(&mut self.total_energy_joules, other.total_energy_joules);
        add_optional(&mut self.total_energy_kwh, other.total_energy_kwh);
        add_optional(&mut self.gross_energy_joules, other.gross_energy_joules);
        add_optional(
            &mut self.adjusted_energy_joules,
            other.adjusted_energy_joules,
        );
        self.energy_measured_ms = match (self.energy_measured_ms, other.energy_measured_ms) {
            (Some(left), Some(right)) => Some(left.saturating_add(right)),
            (left, right) => left.or(right),
        };
        self.energy_source = match (&self.energy_source, &other.energy_source) {
            (Some(left), Some(right)) if left != right => Some("mixed".to_string()),
            (None, Some(right)) => Some(right.clone()),
            (left, _) => left.clone(),
        };
        self.display_energy_excluded &= other.display_energy_excluded;
        self.idle_baseline_applied &= other.idle_baseline_applied;
        self.energy_complete &= other.energy_complete;
        self.energy_exclusive &= other.energy_exclusive;
        self.energy_coverage = self.energy_measured_ms.map(|measured| {
            if self.wall_runtime_ms == 0 {
                0.0
            } else {
                (measured as f64 / self.wall_runtime_ms as f64).clamp(0.0, 1.0)
            }
        });
        self.average_power_watts = self.total_energy_joules.and_then(|joules| {
            (self.wall_runtime_ms > 0).then_some(joules / (self.wall_runtime_ms as f64 / 1_000.0))
        });
    }
}

/// Observed prompt/context facts for one model invocation.
///
/// S0 records post-generation measurements without changing prompt construction. Later
/// small-model reliability milestones fill the compaction/cache/closure counters from their
/// deterministic control paths. Keeping this nested and optional on `LlmInvocation` preserves
/// compatibility with stored v1 events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRetryReason {
    ThinkingOffAfterTruncation,
    CompactMutationAfterTruncation,
    LargerTokenCapAfterTruncation,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentContextUsage {
    pub context_capacity: usize,
    pub reserved_generation_tokens: usize,
    #[serde(default)]
    pub safety_margin_tokens: usize,
    pub usable_prompt_capacity: usize,
    #[serde(default)]
    pub preflight_prompt_tokens: usize,
    pub prompt_utilization_bps: u32,
    pub message_chars: usize,
    pub tool_count: usize,
    pub tool_schema_chars: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_schema_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_reason: Option<AgentRetryReason>,
    #[serde(default)]
    pub compacted_messages: usize,
    #[serde(default)]
    pub omitted_tool_result_chars: usize,
    #[serde(default)]
    pub read_cache_hits: usize,
    #[serde(default)]
    pub closure_checkpoints: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mutation_payload_char_limit: Option<usize>,
    #[serde(default)]
    pub serialized_action_chars: usize,
    #[serde(default)]
    pub carried_evidence_entries: usize,
    #[serde(default)]
    pub carried_evidence_bytes: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCacheUsage {
    pub source: String,
    pub cached_tokens: usize,
    pub prefilled_tokens: usize,
    #[serde(default)]
    pub restore_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NativeGenerationUsage {
    pub fresh_prefill_tokens: usize,
    pub cached_tokens: usize,
    pub prefill_wall_ms: u64,
    pub prefill_tokens_per_second: f64,
    #[serde(default)]
    pub prefill_metal_commands: usize,
    #[serde(default)]
    pub prefill_host_upload_bytes: usize,
    #[serde(default)]
    pub prefill_host_readback_bytes: usize,
    pub decode_tokens: usize,
    pub decode_wall_ms: u64,
    pub decode_tokens_per_second: f64,
    pub model_family: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_experts_per_token: Option<usize>,
    pub expert_strategy: String,
    pub prefill_command_kind: String,
    pub thinking_enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_constraint_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_schema_sha256: Option<String>,
    #[serde(default)]
    pub rejected_constraint_candidates: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_terminal_state: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextSectionUsage {
    pub label: String,
    pub chars: usize,
}

fn add_optional(target: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        *target = Some(target.unwrap_or(0.0) + value);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    Started {
        task: String,
        model: String,
        workspace: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        focus_root: Option<String>,
        branch: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<SessionAttachment>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    HarnessExperimentConfigured {
        observation_rendering: crate::workflow::ObservationRendering,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ConversationTurnStarted {
        turn_id: String,
        intent: crate::workflow::TurnIntent,
        task: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    DeliveryProposed {
        proposal_id: String,
        source_turn_id: String,
        task_summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalProposed {
        proposal_id: String,
        source_turn_id: String,
        objective: String,
        criteria: Vec<crate::goal::GoalCriterionInput>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalStarted {
        goal_id: String,
        objective: String,
        plan_sha256: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalPlanAwaitingApproval {
        goal_id: String,
        plan_sha256: String,
        milestones: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalPlanApproved {
        goal_id: String,
        plan_sha256: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalMilestoneStarted {
        goal_id: String,
        milestone_id: String,
        title: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalMilestoneCompleted {
        goal_id: String,
        milestone_id: String,
        workflow_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalPauseRequested {
        goal_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalPaused {
        goal_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalResumed {
        goal_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalAmendmentRequested {
        goal_id: String,
        amendment_id: String,
        replacement_plan_sha256: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalChangeRequested {
        goal_id: String,
        kind: String,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalAmendmentResolved {
        goal_id: String,
        amendment_id: String,
        accepted: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalReadyForReview {
        goal_id: String,
        checkpoint_sha256: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalCompleted {
        goal_id: String,
        outcome: crate::goal::GoalOutcome,
        completion_basis: crate::goal::GoalCompletionBasis,
        checkpoint_sha256: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalFailed {
        goal_id: String,
        outcome: crate::goal::GoalOutcome,
        reason: String,
        checkpoint_sha256: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    GoalCancelled {
        goal_id: String,
        checkpoint_sha256: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    WorkflowStarted {
        workflow_id: String,
        source_turn_id: String,
        policy_sha256: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    WorkflowResumed {
        workflow_id: String,
        stage: crate::workflow::WorkflowStage,
        checkpoint_sha256: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    WorkflowStageStarted {
        workflow_id: String,
        stage: crate::workflow::WorkflowStage,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    WorkflowArtifactAccepted {
        workflow_id: String,
        artifact_kind: String,
        artifact_id: String,
        sha256: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    WorkflowChallengeRaised {
        workflow_id: String,
        challenge_id: String,
        severity: crate::workflow::ReviewSeverity,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    WorkflowEvidenceInvalidated {
        workflow_id: String,
        previous_fingerprint: String,
        current_fingerprint: String,
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    WorkflowStageCompleted {
        workflow_id: String,
        stage: crate::workflow::WorkflowStage,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    WorkflowBlocked {
        workflow_id: String,
        outcome: crate::workflow::WorkflowOutcome,
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    WorkflowCompleted {
        workflow_id: String,
        outcome: crate::workflow::WorkflowOutcome,
        checkpoint_sha256: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ready_evidence_sha256: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    StepStarted {
        step: usize,
        max_steps: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ModelLoading {
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    Reasoning {
        content: String,
        profile: AgentProfile,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ToolCall {
        tool: String,
        arguments: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        batch_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        actor: Option<TeamActor>,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ControllerObservation {
        receipt: crate::workflow::ControllerObservationReceipt,
        #[serde(default = "workflow_steward_actor")]
        actor: TeamActor,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        assisting_profile: Option<AgentProfile>,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ControllerClosure {
        workflow_id: String,
        stage: crate::workflow::WorkflowStage,
        reason: String,
        #[serde(default = "workflow_steward_actor")]
        actor: TeamActor,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        assisting_profile: Option<AgentProfile>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ControllerMutation {
        receipt: crate::workflow::ControllerMutationReceipt,
        #[serde(default = "workflow_steward_actor")]
        actor: TeamActor,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        assisting_profile: Option<AgentProfile>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ToolBatch {
        call_count: usize,
        parallel_safe_count: usize,
        useful_count: usize,
        bookkeeping_only_count: usize,
        rejected_as_dependent: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ToolResult {
        tool: String,
        result: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        batch_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outcome: Option<ToolOutcome>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        actor: Option<TeamActor>,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        energy_joules: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        energy_kwh: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        average_power_watts: Option<f64>,
        /// Number of concurrently executed calls covered by this single
        /// measurement. Present on one result in a parallel batch.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        energy_shared_calls: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ContextLimit {
        context_capacity: usize,
        reserved_generation_tokens: usize,
        safety_margin_tokens: usize,
        usable_prompt_capacity: usize,
        measured_prompt_tokens: usize,
        #[serde(default)]
        largest_sections: Vec<ContextSectionUsage>,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ExecutorStarted {
        executor_id: String,
        kind: String,
        success: bool,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        detail: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    CheckResult {
        check_id: String,
        exit_status: i32,
        success: bool,
        timed_out: bool,
        output: String,
        truncated: bool,
        duration_ms: u64,
        fingerprint: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        command: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        executor: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        command_fingerprint: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        dependency_outputs: BTreeMap<String, String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_fingerprint: Option<String>,
        #[serde(default, skip_serializing_if = "is_false")]
        reused: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        skip_reason: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    TeamMessage {
        actor: TeamActor,
        tone: TeamMessageTone,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        evidence_ids: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    HandoffSummary {
        summary: HandoffSummary,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    CommitResult {
        success: bool,
        created: bool,
        reused: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        oid: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        subject: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        changed_paths: Vec<String>,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        detail: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    UserQuestion {
        question_id: String,
        question: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        choices: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    UserAnswer {
        question_id: String,
        answer: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    Correction {
        message: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        summary: String,
        #[serde(default = "workflow_steward_actor")]
        actor: TeamActor,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        assisting_profile: Option<AgentProfile>,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    SubAgentStarted {
        profile: String,
        task: String,
        nesting_depth: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    SubAgentFinished {
        profile: String,
        result: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    Diff {
        path: String,
        diff: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    Final {
        content: String,
        profile: AgentProfile,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    FinalGrace {
        status: FinalGraceStatus,
        detail: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    SessionTitle {
        title: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    LlmInvocation {
        step: usize,
        duration_ms: u64,
        prompt_tokens: usize,
        generated_tokens: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt_cache: Option<PromptCacheUsage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context: Option<AgentContextUsage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        native: Option<NativeGenerationUsage>,
        #[serde(skip_serializing_if = "Option::is_none")]
        energy_joules: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        energy_kwh: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        average_power_watts: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    SessionMetrics {
        llm_invocations: usize,
        llm_runtime_ms: u64,
        prompt_tokens: usize,
        generated_tokens: usize,
        tool_calls: usize,
        tool_runtime_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        llm_energy_joules: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        llm_energy_kwh: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_energy_joules: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_energy_kwh: Option<f64>,
        #[serde(default)]
        wall_runtime_ms: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        started_at_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ended_at_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        total_energy_joules: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        total_energy_kwh: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gross_energy_joules: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        adjusted_energy_joules: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        average_power_watts: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        energy_measured_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        energy_coverage: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        energy_source: Option<String>,
        #[serde(default)]
        display_energy_excluded: bool,
        #[serde(default)]
        idle_baseline_applied: bool,
        #[serde(default)]
        energy_complete: bool,
        #[serde(default)]
        energy_exclusive: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    SessionSummary {
        branch: String,
        commits: String,
        #[serde(default)]
        reached_final: bool,
        #[serde(default)]
        contract_status: ContractStatus,
        #[serde(default)]
        verified_completed: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        termination_reason: Option<TerminationReason>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        handoff_outcome: Option<HandoffOutcome>,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        summary: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        power_summary: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        diff_stat: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        diff: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    Error {
        message: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub version: String,
    pub event: AgentEvent,
}

impl EventEnvelope {
    pub fn new(event: AgentEvent) -> Self {
        Self {
            version: EVENT_SCHEMA_VERSION.to_string(),
            event,
        }
    }

    pub fn with_timestamp(event: AgentEvent) -> Self {
        let now = now_millis();
        match event {
            AgentEvent::Started {
                task,
                model,
                workspace,
                focus_root,
                branch,
                attachments,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Started {
                    task,
                    model,
                    workspace,
                    focus_root,
                    branch,
                    attachments,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::HarnessExperimentConfigured {
                observation_rendering,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::HarnessExperimentConfigured {
                    observation_rendering,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ConversationTurnStarted {
                turn_id,
                intent,
                task,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ConversationTurnStarted {
                    turn_id,
                    intent,
                    task,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::DeliveryProposed {
                proposal_id,
                source_turn_id,
                task_summary,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::DeliveryProposed {
                    proposal_id,
                    source_turn_id,
                    task_summary,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::GoalProposed {
                proposal_id,
                source_turn_id,
                objective,
                criteria,
                ..
            } => Self::new(AgentEvent::GoalProposed {
                proposal_id,
                source_turn_id,
                objective,
                criteria,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalStarted {
                goal_id,
                objective,
                plan_sha256,
                ..
            } => Self::new(AgentEvent::GoalStarted {
                goal_id,
                objective,
                plan_sha256,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalPlanAwaitingApproval {
                goal_id,
                plan_sha256,
                milestones,
                ..
            } => Self::new(AgentEvent::GoalPlanAwaitingApproval {
                goal_id,
                plan_sha256,
                milestones,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalPlanApproved {
                goal_id,
                plan_sha256,
                ..
            } => Self::new(AgentEvent::GoalPlanApproved {
                goal_id,
                plan_sha256,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalMilestoneStarted {
                goal_id,
                milestone_id,
                title,
                ..
            } => Self::new(AgentEvent::GoalMilestoneStarted {
                goal_id,
                milestone_id,
                title,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalMilestoneCompleted {
                goal_id,
                milestone_id,
                workflow_id,
                ..
            } => Self::new(AgentEvent::GoalMilestoneCompleted {
                goal_id,
                milestone_id,
                workflow_id,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalPauseRequested { goal_id, .. } => {
                Self::new(AgentEvent::GoalPauseRequested {
                    goal_id,
                    timestamp_ms: Some(now),
                })
            }
            AgentEvent::GoalPaused { goal_id, .. } => Self::new(AgentEvent::GoalPaused {
                goal_id,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalResumed { goal_id, .. } => Self::new(AgentEvent::GoalResumed {
                goal_id,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalAmendmentRequested {
                goal_id,
                amendment_id,
                replacement_plan_sha256,
                ..
            } => Self::new(AgentEvent::GoalAmendmentRequested {
                goal_id,
                amendment_id,
                replacement_plan_sha256,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalChangeRequested {
                goal_id,
                kind,
                summary,
                ..
            } => Self::new(AgentEvent::GoalChangeRequested {
                goal_id,
                kind,
                summary,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalAmendmentResolved {
                goal_id,
                amendment_id,
                accepted,
                ..
            } => Self::new(AgentEvent::GoalAmendmentResolved {
                goal_id,
                amendment_id,
                accepted,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalReadyForReview {
                goal_id,
                checkpoint_sha256,
                ..
            } => Self::new(AgentEvent::GoalReadyForReview {
                goal_id,
                checkpoint_sha256,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalCompleted {
                goal_id,
                outcome,
                completion_basis,
                checkpoint_sha256,
                ..
            } => Self::new(AgentEvent::GoalCompleted {
                goal_id,
                outcome,
                completion_basis,
                checkpoint_sha256,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalFailed {
                goal_id,
                outcome,
                reason,
                checkpoint_sha256,
                ..
            } => Self::new(AgentEvent::GoalFailed {
                goal_id,
                outcome,
                reason,
                checkpoint_sha256,
                timestamp_ms: Some(now),
            }),
            AgentEvent::GoalCancelled {
                goal_id,
                checkpoint_sha256,
                ..
            } => Self::new(AgentEvent::GoalCancelled {
                goal_id,
                checkpoint_sha256,
                timestamp_ms: Some(now),
            }),
            AgentEvent::WorkflowStarted {
                workflow_id,
                source_turn_id,
                policy_sha256,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::WorkflowStarted {
                    workflow_id,
                    source_turn_id,
                    policy_sha256,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::WorkflowResumed {
                workflow_id,
                stage,
                checkpoint_sha256,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::WorkflowResumed {
                    workflow_id,
                    stage,
                    checkpoint_sha256,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::WorkflowStageStarted {
                workflow_id, stage, ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::WorkflowStageStarted {
                    workflow_id,
                    stage,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::WorkflowArtifactAccepted {
                workflow_id,
                artifact_kind,
                artifact_id,
                sha256,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::WorkflowArtifactAccepted {
                    workflow_id,
                    artifact_kind,
                    artifact_id,
                    sha256,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::WorkflowChallengeRaised {
                workflow_id,
                challenge_id,
                severity,
                summary,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::WorkflowChallengeRaised {
                    workflow_id,
                    challenge_id,
                    severity,
                    summary,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::WorkflowEvidenceInvalidated {
                workflow_id,
                previous_fingerprint,
                current_fingerprint,
                reason,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::WorkflowEvidenceInvalidated {
                    workflow_id,
                    previous_fingerprint,
                    current_fingerprint,
                    reason,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::WorkflowStageCompleted {
                workflow_id, stage, ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::WorkflowStageCompleted {
                    workflow_id,
                    stage,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::WorkflowBlocked {
                workflow_id,
                outcome,
                reason,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::WorkflowBlocked {
                    workflow_id,
                    outcome,
                    reason,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::WorkflowCompleted {
                workflow_id,
                outcome,
                checkpoint_sha256,
                ready_evidence_sha256,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::WorkflowCompleted {
                    workflow_id,
                    outcome,
                    checkpoint_sha256,
                    ready_evidence_sha256,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::StepStarted {
                step,
                max_steps,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::StepStarted {
                    step,
                    max_steps,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ModelLoading {
                model,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ModelLoading {
                    model,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::Reasoning {
                content,
                profile,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Reasoning {
                    content,
                    profile,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ToolCall {
                tool,
                arguments,
                call_id,
                batch_id,
                actor,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ToolCall {
                    tool,
                    arguments,
                    call_id,
                    batch_id,
                    actor,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ControllerObservation {
                receipt,
                actor,
                assisting_profile,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ControllerObservation {
                    receipt,
                    actor,
                    assisting_profile,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ControllerClosure {
                workflow_id,
                stage,
                reason,
                actor,
                assisting_profile,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ControllerClosure {
                    workflow_id,
                    stage,
                    reason,
                    actor,
                    assisting_profile,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ControllerMutation {
                receipt,
                actor,
                assisting_profile,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ControllerMutation {
                    receipt,
                    actor,
                    assisting_profile,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ToolBatch {
                call_count,
                parallel_safe_count,
                useful_count,
                bookkeeping_only_count,
                rejected_as_dependent,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ToolBatch {
                    call_count,
                    parallel_safe_count,
                    useful_count,
                    bookkeeping_only_count,
                    rejected_as_dependent,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ToolResult {
                tool,
                result,
                call_id,
                batch_id,
                outcome,
                actor,
                duration_ms,
                energy_joules,
                energy_kwh,
                average_power_watts,
                energy_shared_calls,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ToolResult {
                    tool,
                    result,
                    call_id,
                    batch_id,
                    outcome,
                    actor,
                    duration_ms,
                    energy_joules,
                    energy_kwh,
                    average_power_watts,
                    energy_shared_calls,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ContextLimit {
                context_capacity,
                reserved_generation_tokens,
                safety_margin_tokens,
                usable_prompt_capacity,
                measured_prompt_tokens,
                largest_sections,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ContextLimit {
                    context_capacity,
                    reserved_generation_tokens,
                    safety_margin_tokens,
                    usable_prompt_capacity,
                    measured_prompt_tokens,
                    largest_sections,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ExecutorStarted {
                executor_id,
                kind,
                success,
                detail,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ExecutorStarted {
                    executor_id,
                    kind,
                    success,
                    detail,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::CheckResult {
                check_id,
                exit_status,
                success,
                timed_out,
                output,
                truncated,
                duration_ms,
                fingerprint,
                command,
                cwd,
                executor,
                source,
                command_fingerprint,
                dependency_outputs,
                output_fingerprint,
                reused,
                skip_reason,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::CheckResult {
                    check_id,
                    exit_status,
                    success,
                    timed_out,
                    output,
                    truncated,
                    duration_ms,
                    fingerprint,
                    command,
                    cwd,
                    executor,
                    source,
                    command_fingerprint,
                    dependency_outputs,
                    output_fingerprint,
                    reused,
                    skip_reason,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::TeamMessage {
                actor,
                tone,
                message,
                detail,
                evidence_ids,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::TeamMessage {
                    actor,
                    tone,
                    message,
                    detail,
                    evidence_ids,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::HandoffSummary {
                summary,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::HandoffSummary {
                    summary,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::CommitResult {
                success,
                created,
                reused,
                oid,
                subject,
                changed_paths,
                detail,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::CommitResult {
                    success,
                    created,
                    reused,
                    oid,
                    subject,
                    changed_paths,
                    detail,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::UserQuestion {
                question_id,
                question,
                choices,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::UserQuestion {
                    question_id,
                    question,
                    choices,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::UserAnswer {
                question_id,
                answer,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::UserAnswer {
                    question_id,
                    answer,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::Correction {
                message,
                summary,
                actor,
                assisting_profile,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Correction {
                    message,
                    summary,
                    actor,
                    assisting_profile,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::SubAgentStarted {
                profile,
                task,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::SubAgentStarted {
                    profile,
                    task,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::SubAgentFinished {
                profile,
                result,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::SubAgentFinished {
                    profile,
                    result,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::Diff {
                path,
                diff,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Diff {
                    path,
                    diff,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::Final {
                content,
                profile,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Final {
                    content,
                    profile,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::FinalGrace {
                status,
                detail,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::FinalGrace {
                    status,
                    detail,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::LlmInvocation {
                step,
                duration_ms,
                prompt_tokens,
                generated_tokens,
                prompt_cache,
                context,
                native,
                energy_joules,
                energy_kwh,
                average_power_watts,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::LlmInvocation {
                    step,
                    duration_ms,
                    prompt_tokens,
                    generated_tokens,
                    prompt_cache,
                    context,
                    native,
                    energy_joules,
                    energy_kwh,
                    average_power_watts,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::SessionMetrics {
                llm_invocations,
                llm_runtime_ms,
                prompt_tokens,
                generated_tokens,
                tool_calls,
                tool_runtime_ms,
                llm_energy_joules,
                llm_energy_kwh,
                tool_energy_joules,
                tool_energy_kwh,
                wall_runtime_ms,
                started_at_ms,
                ended_at_ms,
                total_energy_joules,
                total_energy_kwh,
                gross_energy_joules,
                adjusted_energy_joules,
                average_power_watts,
                energy_measured_ms,
                energy_coverage,
                energy_source,
                display_energy_excluded,
                idle_baseline_applied,
                energy_complete,
                energy_exclusive,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::SessionMetrics {
                    llm_invocations,
                    llm_runtime_ms,
                    prompt_tokens,
                    generated_tokens,
                    tool_calls,
                    tool_runtime_ms,
                    llm_energy_joules,
                    llm_energy_kwh,
                    tool_energy_joules,
                    tool_energy_kwh,
                    wall_runtime_ms,
                    started_at_ms,
                    ended_at_ms,
                    total_energy_joules,
                    total_energy_kwh,
                    gross_energy_joules,
                    adjusted_energy_joules,
                    average_power_watts,
                    energy_measured_ms,
                    energy_coverage,
                    energy_source,
                    display_energy_excluded,
                    idle_baseline_applied,
                    energy_complete,
                    energy_exclusive,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::SessionTitle { title, .. } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::SessionTitle {
                    title,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::SessionSummary {
                branch,
                commits,
                reached_final,
                contract_status,
                verified_completed,
                termination_reason,
                handoff_outcome,
                summary,
                power_summary,
                diff_stat,
                diff,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::SessionSummary {
                    branch,
                    commits,
                    reached_final,
                    contract_status,
                    verified_completed,
                    termination_reason,
                    handoff_outcome,
                    summary,
                    power_summary,
                    diff_stat,
                    diff,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::Error {
                message,
                summary,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Error {
                    message,
                    summary,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_session_summary_deserializes_with_truthful_defaults() {
        let json = r#"{
            "version":"v1",
            "event":{
                "type":"session_summary",
                "branch":"task",
                "commits":"",
                "summary":"legacy"
            }
        }"#;
        let envelope: EventEnvelope = serde_json::from_str(json).unwrap();
        assert!(matches!(
            envelope.event,
            AgentEvent::SessionSummary {
                reached_final: false,
                contract_status: ContractStatus::Unspecified,
                verified_completed: false,
                termination_reason: None,
                ..
            }
        ));
    }

    #[test]
    fn action_actor_provenance_round_trips() {
        let envelope = EventEnvelope::with_timestamp(AgentEvent::ToolCall {
            tool: "read_file".to_string(),
            arguments: serde_json::json!({"path": "src/lib.rs"}),
            call_id: Some("call-1".to_string()),
            batch_id: Some("batch-1".to_string()),
            actor: Some(TeamActor::agent(AgentProfile::Review)),
            nesting_depth: None,
            timestamp_ms: None,
        });
        let restored: EventEnvelope =
            serde_json::from_str(&serde_json::to_string(&envelope).unwrap()).unwrap();
        assert!(matches!(
            restored.event,
            AgentEvent::ToolCall {
                call_id: Some(ref call_id),
                batch_id: Some(ref batch_id),
                actor: Some(TeamActor::Agent(AgentProfile::Review)),
                timestamp_ms: Some(_),
                ..
            } if call_id == "call-1" && batch_id == "batch-1"
        ));

        let result = EventEnvelope::with_timestamp(AgentEvent::ToolResult {
            tool: "read_file".to_string(),
            result: "contents".to_string(),
            call_id: Some("call-1".to_string()),
            batch_id: Some("batch-1".to_string()),
            outcome: Some(ToolOutcome::Succeeded),
            actor: Some(TeamActor::agent(AgentProfile::Review)),
            duration_ms: Some(1),
            energy_joules: None,
            energy_kwh: None,
            average_power_watts: None,
            energy_shared_calls: None,
            nesting_depth: None,
            timestamp_ms: None,
        });
        let restored: EventEnvelope =
            serde_json::from_str(&serde_json::to_string(&result).unwrap()).unwrap();
        assert!(matches!(
            restored.event,
            AgentEvent::ToolResult {
                call_id: Some(ref call_id),
                outcome: Some(ToolOutcome::Succeeded),
                ..
            } if call_id == "call-1"
        ));

        let envelope = EventEnvelope::with_timestamp(AgentEvent::ControllerClosure {
            workflow_id: "workflow-1".to_string(),
            stage: crate::workflow::WorkflowStage::Implementing,
            reason: "No change required".to_string(),
            actor: TeamActor::workflow_steward(),
            assisting_profile: Some(AgentProfile::Build),
            nesting_depth: Some(1),
            timestamp_ms: None,
        });
        let restored: EventEnvelope =
            serde_json::from_str(&serde_json::to_string(&envelope).unwrap()).unwrap();
        assert!(matches!(
            restored.event,
            AgentEvent::ControllerClosure {
                actor: TeamActor::Automation(AutomationActor::Trinity),
                assisting_profile: Some(AgentProfile::Build),
                nesting_depth: Some(1),
                timestamp_ms: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn legacy_actions_remain_readable_without_inventing_model_attribution() {
        let legacy_tool = r#"{
            "version":"v1",
            "event":{"type":"tool_call","tool":"read_file","arguments":{"path":"README.md"}}
        }"#;
        let envelope: EventEnvelope = serde_json::from_str(legacy_tool).unwrap();
        assert!(matches!(
            envelope.event,
            AgentEvent::ToolCall {
                actor: None,
                call_id: None,
                batch_id: None,
                ..
            }
        ));

        let legacy_result = r#"{
            "version":"v1",
            "event":{"type":"tool_result","tool":"read_file","result":"contents"}
        }"#;
        let envelope: EventEnvelope = serde_json::from_str(legacy_result).unwrap();
        assert!(matches!(
            envelope.event,
            AgentEvent::ToolResult {
                actor: None,
                call_id: None,
                outcome: None,
                ..
            }
        ));

        let legacy_controller = r#"{
            "version":"v1",
            "event":{
                "type":"controller_closure",
                "workflow_id":"workflow-1",
                "stage":"implementing",
                "reason":"No change required"
            }
        }"#;
        let envelope: EventEnvelope = serde_json::from_str(legacy_controller).unwrap();
        assert!(matches!(
            envelope.event,
            AgentEvent::ControllerClosure {
                actor: TeamActor::Automation(AutomationActor::Trinity),
                assisting_profile: None,
                nesting_depth: None,
                ..
            }
        ));

        let legacy_correction = r#"{
            "version":"v1",
            "event":{"type":"correction","message":"Try again","summary":"Retrying"}
        }"#;
        let envelope: EventEnvelope = serde_json::from_str(legacy_correction).unwrap();
        assert!(matches!(
            envelope.event,
            AgentEvent::Correction {
                actor: TeamActor::Automation(AutomationActor::Trinity),
                assisting_profile: None,
                ..
            }
        ));
    }

    #[test]
    fn legacy_session_metrics_deserialize_without_claiming_measurement_quality() {
        let json = r#"{
            "version":"v1",
            "event":{
                "type":"session_metrics",
                "llm_invocations":1,
                "llm_runtime_ms":10,
                "prompt_tokens":2,
                "generated_tokens":3,
                "tool_calls":0,
                "tool_runtime_ms":0
            }
        }"#;
        let envelope: EventEnvelope = serde_json::from_str(json).unwrap();
        let metrics = SessionMetricsSnapshot::from_event(&envelope.event).unwrap();
        assert_eq!(metrics.wall_runtime_ms, 0);
        assert!(metrics.total_energy_joules.is_none());
        assert!(!metrics.energy_complete);
        assert!(!metrics.energy_exclusive);
    }

    #[test]
    fn cumulative_metrics_use_authoritative_energy_and_wall_time() {
        let mut total = SessionMetricsSnapshot {
            prompt_tokens: 10,
            wall_runtime_ms: 1_000,
            total_energy_joules: Some(20.0),
            total_energy_kwh: Some(20.0 / 3_600_000.0),
            energy_measured_ms: Some(1_000),
            energy_source: Some("smc_system_total".into()),
            display_energy_excluded: true,
            idle_baseline_applied: true,
            energy_complete: true,
            energy_exclusive: true,
            ..Default::default()
        };
        total.add_assign(&SessionMetricsSnapshot {
            generated_tokens: 5,
            wall_runtime_ms: 2_000,
            total_energy_joules: Some(30.0),
            total_energy_kwh: Some(30.0 / 3_600_000.0),
            energy_measured_ms: Some(1_000),
            energy_source: Some("power_telemetry".into()),
            display_energy_excluded: true,
            idle_baseline_applied: false,
            energy_complete: false,
            energy_exclusive: true,
            ..Default::default()
        });

        assert_eq!(total.prompt_tokens, 10);
        assert_eq!(total.generated_tokens, 5);
        assert_eq!(total.wall_runtime_ms, 3_000);
        assert_eq!(total.total_energy_joules, Some(50.0));
        assert_eq!(total.average_power_watts, Some(50.0 / 3.0));
        assert_eq!(total.energy_coverage, Some(2.0 / 3.0));
        assert_eq!(total.energy_source.as_deref(), Some("mixed"));
        assert!(!total.idle_baseline_applied);
        assert!(!total.energy_complete);
    }

    #[test]
    fn legacy_started_event_deserializes_without_focus_root() {
        let json = r#"{
            "version":"v1",
            "event":{
                "type":"started",
                "task":"legacy",
                "model":"model",
                "workspace":"/repo",
                "branch":"pb/legacy",
                "attachments":[]
            }
        }"#;
        let envelope: EventEnvelope = serde_json::from_str(json).unwrap();
        assert!(matches!(
            envelope.event,
            AgentEvent::Started {
                focus_root: None,
                ..
            }
        ));
    }

    #[test]
    fn legacy_workflow_completion_has_no_publication_evidence_claim() {
        let json = r#"{
            "version":"v1",
            "event":{
                "type":"workflow_completed",
                "workflow_id":"legacy-workflow",
                "outcome":"ready",
                "checkpoint_sha256":"legacy-checkpoint"
            }
        }"#;
        let envelope: EventEnvelope = serde_json::from_str(json).unwrap();
        assert!(matches!(
            envelope.event,
            AgentEvent::WorkflowCompleted {
                ready_evidence_sha256: None,
                ..
            }
        ));
    }

    #[test]
    fn new_session_summary_serializes_explicit_outcome_fields() {
        let envelope = EventEnvelope::new(AgentEvent::SessionSummary {
            branch: "task".to_string(),
            commits: String::new(),
            reached_final: true,
            contract_status: ContractStatus::Unspecified,
            verified_completed: false,
            termination_reason: Some(TerminationReason::Final),
            handoff_outcome: Some(HandoffOutcome::Ready),
            summary: "done".to_string(),
            power_summary: String::new(),
            diff_stat: String::new(),
            diff: String::new(),
            timestamp_ms: None,
        });
        let value = serde_json::to_value(envelope).unwrap();
        assert_eq!(value["event"]["reached_final"], true);
        assert_eq!(value["event"]["contract_status"], "unspecified");
        assert_eq!(value["event"]["verified_completed"], false);
        assert_eq!(value["event"]["termination_reason"], "final");
        assert_eq!(value["event"]["handoff_outcome"], "ready");
    }

    #[test]
    fn legacy_llm_invocation_without_context_usage_remains_readable() {
        let json = r#"{
            "version":"v1",
            "event":{
                "type":"llm_invocation",
                "step":1,
                "duration_ms":10,
                "prompt_tokens":20,
                "generated_tokens":3
            }
        }"#;
        let envelope: EventEnvelope = serde_json::from_str(json).unwrap();
        assert!(matches!(
            envelope.event,
            AgentEvent::LlmInvocation { context: None, .. }
        ));
    }

    #[test]
    fn llm_invocation_context_usage_round_trips() {
        let context = AgentContextUsage {
            context_capacity: 8192,
            reserved_generation_tokens: 256,
            safety_margin_tokens: 32,
            usable_prompt_capacity: 7936,
            preflight_prompt_tokens: 4960,
            prompt_utilization_bps: 6250,
            message_chars: 4096,
            tool_count: 3,
            tool_schema_chars: 900,
            tool_schema_tokens: Some(225),
            thinking_enabled: Some(true),
            retry_reason: Some(AgentRetryReason::ThinkingOffAfterTruncation),
            compacted_messages: 2,
            omitted_tool_result_chars: 5000,
            read_cache_hits: 1,
            closure_checkpoints: 1,
            mutation_payload_char_limit: Some(2048),
            serialized_action_chars: 512,
            carried_evidence_entries: 2,
            carried_evidence_bytes: 4096,
        };
        let envelope = EventEnvelope::new(AgentEvent::LlmInvocation {
            step: 2,
            duration_ms: 10,
            prompt_tokens: 4960,
            generated_tokens: 3,
            prompt_cache: Some(PromptCacheUsage {
                source: "memory_prefix".to_string(),
                cached_tokens: 4096,
                prefilled_tokens: 864,
                restore_ms: 0,
            }),
            context: Some(context.clone()),
            native: Some(NativeGenerationUsage {
                fresh_prefill_tokens: 864,
                cached_tokens: 4096,
                prefill_wall_ms: 120,
                prefill_tokens_per_second: 7200.0,
                prefill_metal_commands: 48,
                prefill_host_upload_bytes: 1_024,
                prefill_host_readback_bytes: 512,
                decode_tokens: 3,
                decode_wall_ms: 30,
                decode_tokens_per_second: 100.0,
                model_family: "Qwen3NextMoe".to_string(),
                active_experts_per_token: Some(10),
                expert_strategy: "resident_complete_corpus".to_string(),
                prefill_command_kind: "qwen_chunked_token_batch".to_string(),
                thinking_enabled: false,
                tool_constraint_mode: Some("tool_required".to_string()),
                tool_schema_sha256: Some("abc".to_string()),
                rejected_constraint_candidates: 4,
                constraint_terminal_state: Some("complete_tool_call".to_string()),
            }),
            energy_joules: None,
            energy_kwh: None,
            average_power_watts: None,
            nesting_depth: None,
            timestamp_ms: None,
        });
        let json = serde_json::to_string(&envelope).unwrap();
        let restored: EventEnvelope = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            restored.event,
            AgentEvent::LlmInvocation {
                context: Some(restored),
                prompt_cache: Some(PromptCacheUsage { cached_tokens: 4096, .. }),
                native: Some(NativeGenerationUsage {
                    active_experts_per_token: Some(10),
                    prefill_metal_commands: 48,
                    prefill_host_upload_bytes: 1_024,
                    prefill_host_readback_bytes: 512,
                    ..
                }),
                ..
            } if restored == context
        ));

        let mut legacy: Value = serde_json::from_str(&json).unwrap();
        let native = legacy
            .pointer_mut("/event/native")
            .and_then(Value::as_object_mut)
            .unwrap();
        native.remove("prefill_metal_commands");
        native.remove("prefill_host_upload_bytes");
        native.remove("prefill_host_readback_bytes");
        let legacy: EventEnvelope = serde_json::from_value(legacy).unwrap();
        assert!(matches!(
            legacy.event,
            AgentEvent::LlmInvocation {
                native: Some(NativeGenerationUsage {
                    prefill_metal_commands: 0,
                    prefill_host_upload_bytes: 0,
                    prefill_host_readback_bytes: 0,
                    ..
                }),
                ..
            }
        ));
    }

    #[test]
    fn context_limit_event_round_trips_with_largest_sections() {
        let envelope = EventEnvelope::new(AgentEvent::ContextLimit {
            context_capacity: 512,
            reserved_generation_tokens: 128,
            safety_margin_tokens: 32,
            usable_prompt_capacity: 352,
            measured_prompt_tokens: 900,
            largest_sections: vec![ContextSectionUsage {
                label: "task_stage_anchor".to_string(),
                chars: 3_000,
            }],
            nesting_depth: None,
            timestamp_ms: None,
        });
        let json = serde_json::to_string(&envelope).unwrap();
        let restored: EventEnvelope = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            restored.event,
            AgentEvent::ContextLimit {
                measured_prompt_tokens: 900,
                largest_sections,
                ..
            } if largest_sections[0].label == "task_stage_anchor"
        ));
        assert_eq!(TerminationReason::ContextLimit.as_str(), "context_limit");
    }

    #[test]
    fn handoff_team_messages_round_trip_for_restored_sessions() {
        let envelope = EventEnvelope::new(AgentEvent::TeamMessage {
            actor: TeamActor::Automation(AutomationActor::Handoff),
            tone: TeamMessageTone::Warning,
            message: "The web checks need another pass.".to_string(),
            detail: Some("deno task test:web failed".to_string()),
            evidence_ids: vec!["check:web-test".to_string()],
            nesting_depth: None,
            timestamp_ms: None,
        });
        let json = serde_json::to_string(&envelope).unwrap();
        let restored: EventEnvelope = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            restored.event,
            AgentEvent::TeamMessage {
                actor: TeamActor::Automation(AutomationActor::Handoff),
                tone: TeamMessageTone::Warning,
                message,
                detail: Some(detail),
                evidence_ids,
                ..
            } if message == "The web checks need another pass."
                && detail == "deno task test:web failed"
                && evidence_ids == vec!["check:web-test"]
        ));
    }

    #[test]
    fn workflow_events_round_trip_with_typed_stage_and_outcome() {
        let envelope = EventEnvelope::with_timestamp(AgentEvent::WorkflowBlocked {
            workflow_id: "workflow-1".to_string(),
            outcome: crate::workflow::WorkflowOutcome::RepairCyclesExhausted,
            reason: "blocking findings remain".to_string(),
            timestamp_ms: None,
        });
        let json = serde_json::to_string(&envelope).unwrap();
        let restored: EventEnvelope = serde_json::from_str(&json).unwrap();

        assert!(matches!(
            restored.event,
            AgentEvent::WorkflowBlocked {
                workflow_id,
                outcome: crate::workflow::WorkflowOutcome::RepairCyclesExhausted,
                reason,
                timestamp_ms: Some(_),
            } if workflow_id == "workflow-1" && reason == "blocking findings remain"
        ));
    }
}
