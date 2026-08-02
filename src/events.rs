use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::agent_core::{AgentProfile, SessionAttachment};
use crate::inference::PromptCacheMissReason;
pub use crate::inference::StageRootAuthorityClass as PromptRootAuthorityClass;
use crate::session_store::now_millis;
pub use crate::workflow::WorkflowBlockCause;

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeamMessagePurpose {
    #[default]
    General,
    HandoffProgress,
    HandoffOutcome,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionKind {
    #[default]
    General,
    ArtifactValidation,
    RepositoryEvidence,
    ContractEvidence,
    WorkUnit,
    RuntimeFallback,
    RepeatedTool,
    NoProgress,
    DependentToolBatch,
    StageSubmission,
    InvalidAction,
    StepLimit,
    AdvisoryBudget,
    MissingEvidence,
    TruncatedAction,
    MutationRecovery,
    Lifecycle,
    TaskPlanningRecovery,
    ToolUnavailable,
    RequirementsRemain,
    Handoff,
    Diagnostics,
    WorkflowClosure,
    ToolFailure,
}

impl CorrectionKind {
    fn classify(summary: &str, detail: &str) -> Self {
        let summary = summary.trim();
        if summary == "Workflow artifact validation failed"
            || (summary.starts_with("submit_")
                && summary.ends_with(" tool call was not executed successfully"))
        {
            Self::ArtifactValidation
        } else if serde_json::from_str::<Value>(detail)
            .ok()
            .and_then(|value| {
                value
                    .get("type")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .as_deref()
            == Some("tool_failure")
        {
            Self::ToolFailure
        } else if summary == "Task-focused repository evidence" {
            Self::RepositoryEvidence
        } else if summary.contains("contract planning evidence")
            || summary.contains("contract review evidence")
            || summary.contains("proposed-path review evidence")
        {
            Self::ContractEvidence
        } else if summary.contains("accepted-plan")
            || summary == "Work-unit progress earned one bounded turn"
        {
            Self::WorkUnit
        } else if summary.contains("using host execution")
            || summary.contains("CPU-only llama.cpp fallback")
        {
            Self::RuntimeFallback
        } else if summary == "Repeated tool call detected"
            || summary == "Repeated tool call blocked"
            || summary.contains("repeated the same action")
            || summary.contains("reached the repeat limit")
        {
            Self::RepeatedTool
        } else if summary == "No-progress tool outcome detected" {
            Self::NoProgress
        } else if summary == "Dependent tool batch rejected" {
            Self::DependentToolBatch
        } else if summary == "Workflow stage submission required" {
            Self::StageSubmission
        } else if summary == "Teammate action retries exhausted"
            || summary.contains("Parse retry limit")
            || summary.contains("Invalid pb JSON action")
            || summary.to_lowercase().contains("unparsable")
        {
            Self::InvalidAction
        } else if summary.contains("bounded step limit") {
            Self::StepLimit
        } else if summary == "Advisory budget exhausted" {
            Self::AdvisoryBudget
        } else if summary == "Requesting missing bounded evidence" {
            Self::MissingEvidence
        } else if summary.contains("truncated action") {
            Self::TruncatedAction
        } else if summary.contains("mutation recovery")
            || summary.contains("compact atomic mutation")
            || summary.contains("constrained mutation")
            || summary.contains("payload-bound atomic mutation")
        {
            Self::MutationRecovery
        } else if matches!(
            summary,
            "Run cancelled" | "Goal pausing" | "Cancellation requested"
        ) {
            Self::Lifecycle
        } else if matches!(
            summary,
            "Restarting delivery from current files"
                | "Retrying Task planning"
                | "Running as one Build"
        ) {
            Self::TaskPlanningRecovery
        } else if summary == "Tool not available" {
            Self::ToolUnavailable
        } else if summary.contains("Task requirements remain") {
            Self::RequirementsRemain
        } else if summary.contains("handoff")
            || summary.starts_with("Agent terminated:")
            || summary.starts_with("Final grace terminated:")
            || summary.contains("Completion gate")
            || summary.contains("Acceptance contract")
        {
            Self::Handoff
        } else if summary.to_lowercase().contains("diagnostic") {
            Self::Diagnostics
        } else if summary == "Workflow closure checkpoint" {
            Self::WorkflowClosure
        } else {
            Self::General
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatterAudience {
    #[default]
    Team,
    CurrentUser,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventChatter {
    pub actor: TeamActor,
    pub tone: TeamMessageTone,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headline: Option<String>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default)]
    pub audience: ChatterAudience,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptVisibility {
    Visible,
    EvidenceOnly,
    Activity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptKind {
    Conversation,
    Activity,
    Evidence,
    Correction,
    RepeatedToolDetected,
    RepeatedToolCorrection,
    NoProgressCorrection,
    DependentToolBatchCorrection,
    HandoffCorrection,
    WorkflowClosureCheckpoint,
    WorkUnitProgress,
    TerminalToolLoopError,
    WorkflowBlocked,
    SessionSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptMetadata {
    pub visibility: TranscriptVisibility,
    pub kind: TranscriptKind,
    pub entry_key: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supersedes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dedupe_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub related_action_key: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub summary_redundant: bool,
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
    #[serde(default)]
    pub cache_persistence_queued_checkpoints: usize,
    #[serde(default)]
    pub cache_persistence_completed_checkpoints: usize,
    #[serde(default)]
    pub cache_persistence_wall_ms: u64,
    #[serde(default)]
    pub cache_persistence_failures: usize,
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
            cache_persistence_queued_checkpoints,
            cache_persistence_completed_checkpoints,
            cache_persistence_wall_ms,
            cache_persistence_failures,
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
                cache_persistence_queued_checkpoints: *cache_persistence_queued_checkpoints,
                cache_persistence_completed_checkpoints: *cache_persistence_completed_checkpoints,
                cache_persistence_wall_ms: *cache_persistence_wall_ms,
                cache_persistence_failures: *cache_persistence_failures,
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
        self.cache_persistence_queued_checkpoints = self
            .cache_persistence_queued_checkpoints
            .saturating_add(other.cache_persistence_queued_checkpoints);
        self.cache_persistence_completed_checkpoints = self
            .cache_persistence_completed_checkpoints
            .saturating_add(other.cache_persistence_completed_checkpoints);
        self.cache_persistence_wall_ms = self
            .cache_persistence_wall_ms
            .saturating_add(other.cache_persistence_wall_ms);
        self.cache_persistence_failures = self
            .cache_persistence_failures
            .saturating_add(other.cache_persistence_failures);
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
    BoundedReadAfterMutationDeadEnd,
    ExpandedMutationAfterPayloadLimit,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub miss_reason: Option<PromptCacheMissReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookup_detail: Option<crate::inference::PromptCacheLookupDetail>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<PromptRootUsage>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptRootUsage {
    pub descriptor_version: u32,
    pub backend: String,
    #[serde(default)]
    pub cache_format_version: String,
    pub model_namespace_sha256: String,
    pub rendered_token_sha256: String,
    pub tokens: usize,
    pub reused_tokens: usize,
    pub system_instruction_version: Option<String>,
    pub workflow_stage: Option<crate::workflow::WorkflowStage>,
    pub authority_class: PromptRootAuthorityClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_schema_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_constraint_mode: Option<String>,
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
    #[serde(default)]
    pub prefill_command_reason: String,
    pub thinking_enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refill: Option<NativeRefillUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_constraint_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_constraint_dialect: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_schema_sha256: Option<String>,
    #[serde(default)]
    pub rejected_constraint_candidates: usize,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mutation_constraint_rejections: BTreeMap<String, usize>,
    #[serde(default)]
    pub mutation_snapshot_files: usize,
    #[serde(default)]
    pub mutation_snapshot_bytes: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_terminal_state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_guarantee_rung: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_boundary: Option<crate::inference::SemanticBoundaryStats>,
    #[serde(default)]
    pub decode_recovery: crate::inference::DecodeRecovery,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeRefillUsage {
    pub cache_lookup_wall_ms: u64,
    #[serde(default)]
    pub disk_read_decode_wall_ms: u64,
    #[serde(default)]
    pub cpu_state_validation_allocation_wall_ms: u64,
    pub state_hydration_wall_ms: u64,
    pub fresh_suffix_prefill_wall_ms: u64,
    pub snapshot_capture_wall_ms: u64,
    #[serde(default)]
    pub persistence_queue_wall_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextSectionUsage {
    pub label: String,
    pub chars: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuedUserMessage {
    pub message_id: String,
    pub message: String,
}

fn add_optional(target: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        *target = Some(target.unwrap_or(0.0) + value);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelInvocationPurpose {
    #[default]
    Unclassified,
    Conversation,
    TaskPartitioning,
    WorkflowPlanning,
    WorkflowReview,
    WorkflowEvidence,
    WorkflowMutation,
    WorkflowClosure,
    WorkflowRecovery,
}

impl ModelInvocationPurpose {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unclassified => "unclassified",
            Self::Conversation => "conversation",
            Self::TaskPartitioning => "task partitioning",
            Self::WorkflowPlanning => "workflow planning",
            Self::WorkflowReview => "workflow review",
            Self::WorkflowEvidence => "workflow evidence",
            Self::WorkflowMutation => "workflow mutation",
            Self::WorkflowClosure => "workflow closure",
            Self::WorkflowRecovery => "workflow recovery",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    Started {
        task: String,
        model: String,
        #[serde(default)]
        profile: AgentProfile,
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
    TaskPlanAccepted {
        multi_task_id: String,
        plan_sha256: String,
        task_count: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    TaskPlanRejected {
        outcome: crate::task_queue::TaskPlanRejectionOutcome,
        attempts: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    TasksChanged {
        multi_task_id: String,
        stage: crate::task_queue::MultiTaskStage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outcome: Option<crate::task_queue::MultiTaskOutcome>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        active_task_id: Option<String>,
        checkpoint_sha256: String,
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
        #[serde(default)]
        cause: WorkflowBlockCause,
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
        #[serde(default)]
        profile: AgentProfile,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ModelLoading {
        model: String,
        #[serde(default)]
        profile: AgentProfile,
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
    SemanticGate {
        receipt: pb_control_collar::analysis::SemanticGateReceipt,
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
        #[serde(default)]
        purpose: TeamMessagePurpose,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        handoff: Option<HandoffSummary>,
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
        #[serde(default)]
        profile: AgentProfile,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    UserAnswer {
        question_id: String,
        answer: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    UserMessage {
        message_id: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    UserMessageApplied {
        message_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    Correction {
        message: String,
        #[serde(default)]
        kind: CorrectionKind,
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
        profile: AgentProfile,
        task: String,
        nesting_depth: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    SubAgentFinished {
        profile: AgentProfile,
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
        #[serde(default)]
        purpose: ModelInvocationPurpose,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        workflow_stage: Option<crate::workflow::WorkflowStage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        profile: Option<AgentProfile>,
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
        #[serde(default)]
        cache_persistence_queued_checkpoints: usize,
        #[serde(default)]
        cache_persistence_completed_checkpoints: usize,
        #[serde(default)]
        cache_persistence_wall_ms: u64,
        #[serde(default)]
        cache_persistence_failures: usize,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chatter: Vec<EventChatter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<TranscriptMetadata>,
}

impl EventEnvelope {
    pub fn new(mut event: AgentEvent) -> Self {
        normalize_event_semantics(&mut event);
        let mut envelope = Self {
            version: EVENT_SCHEMA_VERSION.to_string(),
            event,
            chatter: Vec::new(),
            transcript: None,
        };
        envelope.refresh_projections(&[]);
        envelope
    }

    pub fn refresh_projections(&mut self, history: &[EventEnvelope]) {
        normalize_event_semantics(&mut self.event);
        self.chatter = chatter_for_event(&self.event, history);
        self.transcript = Some(transcript_metadata_for_event(&self.event, history));
    }

    pub fn with_timestamp(event: AgentEvent) -> Self {
        let now = now_millis();
        let mut envelope = match event {
            AgentEvent::Started {
                task,
                model,
                profile,
                workspace,
                focus_root,
                branch,
                attachments,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::Started {
                    task,
                    model,
                    profile,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
            AgentEvent::TaskPlanAccepted {
                multi_task_id,
                plan_sha256,
                task_count,
                ..
            } => Self::new(AgentEvent::TaskPlanAccepted {
                multi_task_id,
                plan_sha256,
                task_count,
                timestamp_ms: Some(now),
            }),
            AgentEvent::TaskPlanRejected {
                outcome, attempts, ..
            } => Self::new(AgentEvent::TaskPlanRejected {
                outcome,
                attempts,
                timestamp_ms: Some(now),
            }),
            AgentEvent::TasksChanged {
                multi_task_id,
                stage,
                outcome,
                active_task_id,
                checkpoint_sha256,
                ..
            } => Self::new(AgentEvent::TasksChanged {
                multi_task_id,
                stage,
                outcome,
                active_task_id,
                checkpoint_sha256,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::WorkflowStageCompleted {
                    workflow_id,
                    stage,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::WorkflowBlocked {
                workflow_id,
                outcome,
                cause,
                reason,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::WorkflowBlocked {
                    workflow_id,
                    outcome,
                    cause,
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
                chatter: Vec::new(),
                transcript: None,
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
                profile,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::StepStarted {
                    step,
                    max_steps,
                    profile,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ModelLoading {
                model,
                profile,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::ModelLoading {
                    model,
                    profile,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                purpose,
                handoff,
                message,
                detail,
                evidence_ids,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::TeamMessage {
                    actor,
                    tone,
                    purpose,
                    handoff,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                profile,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::UserQuestion {
                    question_id,
                    question,
                    choices,
                    profile,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::UserAnswer {
                question_id,
                answer,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::UserAnswer {
                    question_id,
                    answer,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::UserMessage {
                message_id,
                message,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::UserMessage {
                    message_id,
                    message,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::UserMessageApplied { message_id, .. } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::UserMessageApplied {
                    message_id,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::Correction {
                message,
                kind,
                summary,
                actor,
                assisting_profile,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::Correction {
                    message,
                    kind,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::FinalGrace {
                    status,
                    detail,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::LlmInvocation {
                step,
                purpose,
                workflow_stage,
                profile,
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
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::LlmInvocation {
                    step,
                    purpose,
                    workflow_stage,
                    profile,
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
                cache_persistence_queued_checkpoints,
                cache_persistence_completed_checkpoints,
                cache_persistence_wall_ms,
                cache_persistence_failures,
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
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::SessionMetrics {
                    llm_invocations,
                    llm_runtime_ms,
                    prompt_tokens,
                    generated_tokens,
                    tool_calls,
                    tool_runtime_ms,
                    cache_persistence_queued_checkpoints,
                    cache_persistence_completed_checkpoints,
                    cache_persistence_wall_ms,
                    cache_persistence_failures,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
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
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::Error {
                    message,
                    summary,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::SemanticGate {
                receipt,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                chatter: Vec::new(),
                transcript: None,
                event: AgentEvent::SemanticGate {
                    receipt,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
        };
        normalize_event_semantics(&mut envelope.event);
        envelope.refresh_projections(&[]);
        envelope
    }
}

fn normalize_event_semantics(event: &mut AgentEvent) {
    match event {
        AgentEvent::Correction {
            message,
            kind,
            summary,
            ..
        } if *kind == CorrectionKind::General => {
            *kind = CorrectionKind::classify(summary, message);
        }
        AgentEvent::WorkflowBlocked {
            outcome,
            cause,
            reason,
            ..
        } if *cause == WorkflowBlockCause::Other => {
            *cause = WorkflowBlockCause::classify(*outcome, reason);
        }
        _ => {}
    }
}

pub fn refresh_event_projections(events: &mut [EventEnvelope]) {
    for index in 0..events.len() {
        let (history, current_and_rest) = events.split_at_mut(index);
        if let Some(current) = current_and_rest.first_mut() {
            current.refresh_projections(history);
        }
    }
}

pub fn hydrate_event_projections(events: &mut [EventEnvelope]) {
    for index in 0..events.len() {
        let (history, current_and_rest) = events.split_at_mut(index);
        let Some(current) = current_and_rest.first_mut() else {
            continue;
        };
        normalize_event_semantics(&mut current.event);
        let projected_chatter = chatter_for_event(&current.event, history);
        let projected_transcript = transcript_metadata_for_event(&current.event, history);
        if current.chatter.is_empty() {
            current.chatter = projected_chatter;
        }
        if current.transcript.is_none() {
            current.transcript = Some(projected_transcript);
        }
    }
}

fn transcript_metadata_for_event(
    event: &AgentEvent,
    history: &[EventEnvelope],
) -> TranscriptMetadata {
    let (visibility, kind, dedupe_key, related_action_key, summary_redundant) = match event {
        AgentEvent::Correction {
            message,
            kind,
            summary,
            ..
        } => {
            let visibility = if matches!(
                kind,
                CorrectionKind::Handoff
                    | CorrectionKind::WorkflowClosure
                    | CorrectionKind::WorkUnit
            ) {
                TranscriptVisibility::EvidenceOnly
            } else {
                TranscriptVisibility::Visible
            };
            let transcript_kind = match kind {
                CorrectionKind::Handoff => TranscriptKind::HandoffCorrection,
                CorrectionKind::WorkflowClosure => TranscriptKind::WorkflowClosureCheckpoint,
                CorrectionKind::WorkUnit => TranscriptKind::WorkUnitProgress,
                CorrectionKind::RepeatedTool => TranscriptKind::RepeatedToolCorrection,
                CorrectionKind::NoProgress => TranscriptKind::NoProgressCorrection,
                CorrectionKind::DependentToolBatch => TranscriptKind::DependentToolBatchCorrection,
                _ => TranscriptKind::Correction,
            };
            let dedupe_key = correction_dedupe_key(summary, message);
            let related_action_key = dedupe_key
                .as_ref()
                .and_then(|_| latest_tool_action_key(history));
            (
                visibility,
                transcript_kind,
                dedupe_key,
                related_action_key,
                false,
            )
        }
        AgentEvent::WorkflowBlocked { .. } => (
            TranscriptVisibility::Visible,
            TranscriptKind::WorkflowBlocked,
            None,
            None,
            false,
        ),
        AgentEvent::StepStarted { .. } | AgentEvent::ModelLoading { .. } => (
            TranscriptVisibility::Activity,
            TranscriptKind::Activity,
            None,
            None,
            false,
        ),
        AgentEvent::WorkflowArtifactAccepted { artifact_kind, .. } if artifact_kind == "plan" => (
            TranscriptVisibility::Visible,
            TranscriptKind::Conversation,
            None,
            None,
            false,
        ),
        AgentEvent::Started { .. }
        | AgentEvent::Reasoning { .. }
        | AgentEvent::ToolCall { .. }
        | AgentEvent::ToolResult { .. }
        | AgentEvent::ControllerObservation { .. }
        | AgentEvent::ControllerClosure { .. }
        | AgentEvent::ControllerMutation { .. }
        | AgentEvent::TeamMessage { .. }
        | AgentEvent::UserQuestion { .. }
        | AgentEvent::UserAnswer { .. }
        | AgentEvent::UserMessage { .. }
        | AgentEvent::Diff { .. }
        | AgentEvent::Final { .. }
        | AgentEvent::LlmInvocation { .. }
        | AgentEvent::WorkflowChallengeRaised { .. }
        | AgentEvent::WorkflowEvidenceInvalidated { .. }
        | AgentEvent::SessionMetrics { .. }
        | AgentEvent::Error { .. } => (
            TranscriptVisibility::Visible,
            TranscriptKind::Conversation,
            None,
            None,
            false,
        ),
        AgentEvent::SessionSummary { summary, .. } => (
            TranscriptVisibility::Visible,
            TranscriptKind::SessionSummary,
            None,
            None,
            session_summary_is_redundant(summary, history),
        ),
        _ => (
            TranscriptVisibility::EvidenceOnly,
            TranscriptKind::Evidence,
            None,
            None,
            false,
        ),
    };
    let entry_key = event_entry_key(event);
    TranscriptMetadata {
        visibility,
        kind,
        entry_key,
        supersedes: superseded_entry_keys(event, history),
        tool_summary: tool_summary_for_event(event, history),
        dedupe_key,
        related_action_key,
        summary_redundant,
    }
}

fn event_entry_key(event: &AgentEvent) -> String {
    let serialized = serde_json::to_vec(event).unwrap_or_default();
    format!("event:{}", crate::environment_lock::sha256(&serialized))
}

fn superseded_entry_keys(event: &AgentEvent, history: &[EventEnvelope]) -> Vec<String> {
    match event {
        AgentEvent::TeamMessage {
            actor,
            purpose: TeamMessagePurpose::HandoffOutcome,
            ..
        } => history
            .iter()
            .rev()
            .find_map(|envelope| match &envelope.event {
                AgentEvent::TeamMessage {
                    actor: prior_actor,
                    purpose: TeamMessagePurpose::HandoffProgress,
                    ..
                } if prior_actor == actor => envelope
                    .transcript
                    .as_ref()
                    .map(|metadata| metadata.entry_key.clone()),
                _ => None,
            })
            .into_iter()
            .collect(),
        AgentEvent::TeamMessage {
            actor: TeamActor::Automation(_),
            tone: TeamMessageTone::Info,
            purpose: TeamMessagePurpose::General,
            ..
        } => history
            .iter()
            .rev()
            .find_map(|envelope| match &envelope.event {
                AgentEvent::TeamMessage {
                    actor: TeamActor::Automation(_),
                    tone: TeamMessageTone::Info,
                    purpose: TeamMessagePurpose::General,
                    ..
                } => envelope
                    .transcript
                    .as_ref()
                    .map(|metadata| metadata.entry_key.clone()),
                _ => None,
            })
            .into_iter()
            .collect(),
        AgentEvent::WorkflowBlocked { .. } => history
            .iter()
            .rev()
            .take_while(|envelope| {
                envelope.transcript.as_ref().is_none_or(|metadata| {
                    metadata.visibility != TranscriptVisibility::Visible
                        || matches!(
                            envelope.event,
                            AgentEvent::Correction { .. } | AgentEvent::Error { .. }
                        )
                })
            })
            .filter_map(|envelope| match &envelope.event {
                AgentEvent::Correction { .. } | AgentEvent::Error { .. } => envelope
                    .transcript
                    .as_ref()
                    .map(|metadata| metadata.entry_key.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn tool_summary_for_event(event: &AgentEvent, history: &[EventEnvelope]) -> Option<String> {
    match event {
        AgentEvent::ToolCall {
            tool, arguments, ..
        } => Some(tool_summary(tool, arguments, None)),
        AgentEvent::ToolResult {
            tool,
            result,
            call_id,
            ..
        } => history.iter().rev().find_map(|envelope| {
            let AgentEvent::ToolCall {
                tool: called_tool,
                arguments,
                call_id: called_id,
                ..
            } = &envelope.event
            else {
                return None;
            };
            ((call_id.is_some() && call_id == called_id)
                || (call_id.is_none() && called_tool == tool))
                .then(|| tool_summary(tool, arguments, Some(result)))
        }),
        _ => None,
    }
}

fn tool_summary(tool: &str, arguments: &Value, result: Option<&str>) -> String {
    let string = |name: &str| {
        arguments
            .get(name)
            .and_then(Value::as_str)
            .unwrap_or_default()
    };
    let count = |name: &str| {
        arguments
            .get(name)
            .and_then(Value::as_array)
            .map_or(0, Vec::len)
    };
    let scoped = |value: &str, scope: &str, fallback: &str| {
        let label = if value.is_empty() { fallback } else { value };
        if scope.is_empty() {
            label.to_string()
        } else {
            format!("{label} · in {scope}")
        }
    };
    match tool {
        "read_file" | "inspect_change" | "rm" | "write_file" | "replace_file" | "apply_patch" => {
            let path = string("path");
            if path.is_empty() {
                "(no path)".to_string()
            } else {
                path.to_string()
            }
        }
        "edit_file" => {
            let path = string("path");
            if path.is_empty() {
                "(no path)".to_string()
            } else if arguments.get("diff").is_some() {
                format!("{path} (patch)")
            } else {
                path.to_string()
            }
        }
        "glob" => scoped(
            string("pattern"),
            if string("path").is_empty() {
                string("relative_path")
            } else {
                string("path")
            },
            "(no pattern)",
        ),
        "ripgrep" | "search" => scoped(string("pattern"), string("path"), "(no pattern)"),
        "web_search" => string_or(string("query"), "(no query)"),
        "web_fetch" => string_or(string("url"), "(no url)"),
        "run_command" => string_or(string("cmd"), "(no cmd)"),
        "run_task" | "run_check" => string_or(string("id"), "(no id)"),
        "session_changes" => {
            let mut filters = Vec::new();
            if !string("path").is_empty() {
                filters.push(format!("File: {}", string("path")));
            }
            if !string("commits").is_empty() {
                filters.push(format!("Commits: {}", string("commits")));
            }
            if filters.is_empty() {
                "Recent sessions and changes".to_string()
            } else {
                filters.join(" · ")
            }
        }
        "lsp_proactive_diagnostics" => lsp_tool_summary(arguments, result),
        "skill_search" => {
            let query = string("query");
            match result {
                None => format!("{query} (pending)"),
                Some(result) => format!("{query} ({} skills)", result.matches("name: ").count()),
            }
        }
        "skill" => match string("name") {
            "" => "(no name)".to_string(),
            "list" => "loaded skills list".to_string(),
            name => name.to_string(),
        },
        "mv" => format!("from {} to {}", string("source"), string("destination")),
        "git_commit" => string_or(string("message"), "(no message)"),
        "session_title" => string_or(string("title"), "(no title)"),
        "memory_search" => string_or(string("query"), "All relevant project memory"),
        "memory_read" | "memory_supersede" => string_or(string("id"), "(no memory id)"),
        "memory_propose" => string_or(
            string("title"),
            string_or(string("kind"), "New project memory").as_str(),
        ),
        "propose_delivery" | "start_delivery" => {
            string_or(string("task_summary"), "(no delivery summary)")
        }
        "propose_goal" | "start_goal" => string_or(string("objective"), "(no goal objective)"),
        "goal_pause" | "goal_request_budget" | "request_replan" => {
            string_or(string("reason"), "(no reason)")
        }
        "goal_request_amendment" => string_or(string("summary"), "(no change summary)"),
        "submit_plan" => {
            let requirements = count("requirements");
            let steps = count("steps");
            let acceptance = count("acceptance");
            if requirements == 0 || steps == 0 || acceptance == 0 {
                "Incomplete plan · missing required sections".to_string()
            } else {
                format!(
                    "{requirements} requirement{} · {steps} step{} · {acceptance} acceptance check{}",
                    plural(requirements),
                    plural(steps),
                    plural(acceptance)
                )
            }
        }
        "submit_plan_review" | "submit_code_review" => {
            let raw = string("verdict").replace('_', " ");
            let verdict = capitalize_or(&raw, "Review submitted");
            let concerns = arguments
                .get("challenges")
                .or_else(|| arguments.get("findings"))
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            if concerns == 0 {
                verdict
            } else {
                format!("{verdict} · {concerns} finding{}", plural(concerns))
            }
        }
        "submit_implementation" => {
            let steps = count("steps");
            format!("{steps} implementation step{}", plural(steps))
        }
        "git_revert" => string_or(string("commit"), "(no commit)"),
        _ => generic_tool_summary(result),
    }
}

fn string_or(value: &str, fallback: &str) -> String {
    if value.is_empty() {
        fallback.to_string()
    } else {
        value.to_string()
    }
}

fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

fn capitalize_or(value: &str, fallback: &str) -> String {
    let mut chars = value.chars();
    chars.next().map_or_else(
        || fallback.to_string(),
        |first| first.to_uppercase().collect::<String>() + chars.as_str(),
    )
}

fn generic_tool_summary(result: Option<&str>) -> String {
    let Some(result) = result else {
        return "(pending)".to_string();
    };
    match serde_json::from_str::<Value>(result) {
        Ok(Value::Array(items)) => format!("{} items", items.len()),
        Ok(Value::Object(fields)) => format!("result ({} fields)", fields.len()),
        _ if result.len() < 80 => result.replace('\n', " "),
        _ => String::new(),
    }
}

fn lsp_tool_summary(arguments: &Value, result: Option<&str>) -> String {
    let mode = arguments
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("automatic");
    let requested = arguments
        .get("paths")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let Some(result) = result else {
        return format!("{mode} · {requested} file{} (pending)", plural(requested));
    };
    let Ok(report) = serde_json::from_str::<Value>(result) else {
        return format!("{mode} · {requested} file{}", plural(requested));
    };
    let count = |name: &str| {
        report
            .get(name)
            .and_then(Value::as_array)
            .map_or(0, Vec::len)
    };
    let scanned = count("scanned_paths");
    let diagnostics = count("diagnostics");
    let failures = count("failures");
    let omitted = report
        .get("omitted_paths")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let deferred = if omitted > 0 {
        format!(" · {omitted} deferred")
    } else {
        String::new()
    };
    if report.get("stale").and_then(Value::as_bool) == Some(true) {
        return format!("{mode} · stale evidence discarded");
    }
    if diagnostics > 0 {
        return format!(
            "{mode} · {diagnostics} blocking diagnostic{} in {scanned} file{}{deferred}",
            plural(diagnostics),
            plural(scanned)
        );
    }
    if failures > 0 {
        return format!(
            "{mode} · {scanned}/{requested} files · {failures} server issue{}{deferred}",
            plural(failures)
        );
    }
    if report.get("complete").and_then(Value::as_bool) != Some(true) {
        return format!(
            "{mode} · incomplete evidence · {}/{} server/file targets{deferred}",
            count("completed_targets"),
            count("requested_targets")
        );
    }
    if omitted > 0 {
        return format!("{mode} · {scanned} files{deferred}");
    }
    format!("{mode} · {scanned} file{} · clean", plural(scanned))
}

fn latest_tool_action_key(history: &[EventEnvelope]) -> Option<String> {
    history.iter().rev().find_map(|envelope| {
        let AgentEvent::ToolCall {
            tool, arguments, ..
        } = &envelope.event
        else {
            return None;
        };
        Some(format!(
            "tool:{tool}:{}",
            crate::agent_context::normalized_arguments_sha256(arguments)
        ))
    })
}

fn session_summary_is_redundant(summary: &str, history: &[EventEnvelope]) -> bool {
    let summary = summary.trim();
    if summary.is_empty() {
        return false;
    }
    let latest_final = history.iter().rev().find_map(|envelope| {
        let AgentEvent::Final { content, .. } = &envelope.event else {
            return None;
        };
        Some(content.trim())
    });
    let latest_workflow_block = history.iter().rev().find_map(|envelope| {
        let AgentEvent::WorkflowBlocked { reason, .. } = &envelope.event else {
            return None;
        };
        Some(reason.trim())
    });
    latest_final == Some(summary) || latest_workflow_block == Some(summary)
}

fn correction_dedupe_key(summary: &str, detail: &str) -> Option<String> {
    if let Ok(failure) = serde_json::from_str::<Value>(detail)
        && failure.get("type").and_then(Value::as_str) == Some("tool_failure")
        && let (Some(tool), Some(message)) = (
            failure.get("tool").and_then(Value::as_str),
            failure.get("message").and_then(Value::as_str),
        )
    {
        return Some(
            extract_quoted_after(message, "failed to resolve path '").map_or_else(
                || format!("tool_failure:{tool}:{message}"),
                |path| format!("tool_failure:{tool}:missing:{path}"),
            ),
        );
    }
    (!summary.trim().is_empty()).then(|| format!("correction:{}", summary.trim()))
}

fn chatter_for_event(event: &AgentEvent, history: &[EventEnvelope]) -> Vec<EventChatter> {
    match event {
        AgentEvent::Correction {
            message,
            kind,
            summary,
            actor,
            assisting_profile,
            ..
        } => vec![correction_chatter(
            *actor,
            *assisting_profile,
            *kind,
            summary,
            message,
            history,
        )],
        AgentEvent::WorkflowBlocked {
            outcome,
            cause,
            reason,
            ..
        } => workflow_blocked_chatter(*outcome, *cause, reason, history),
        _ => Vec::new(),
    }
}

fn latest_agent_profile(history: &[EventEnvelope]) -> Option<AgentProfile> {
    history
        .iter()
        .rev()
        .find_map(|envelope| match &envelope.event {
            AgentEvent::LlmInvocation {
                profile: Some(profile),
                ..
            }
            | AgentEvent::StepStarted { profile, .. }
            | AgentEvent::ModelLoading { profile, .. }
            | AgentEvent::Reasoning { profile, .. }
            | AgentEvent::Final { profile, .. }
            | AgentEvent::Started { profile, .. } => Some(*profile),
            AgentEvent::ToolCall {
                actor: Some(TeamActor::Agent(profile)),
                ..
            } => Some(*profile),
            _ => None,
        })
}

fn first_name(name: &str) -> &str {
    name.split_whitespace().next().unwrap_or("Teammate")
}

fn chatter(
    actor: TeamActor,
    headline: Option<String>,
    message: String,
    detail: Option<String>,
    audience: ChatterAudience,
) -> EventChatter {
    EventChatter {
        actor,
        tone: TeamMessageTone::Warning,
        headline,
        message,
        detail,
        audience,
    }
}

fn correction_chatter(
    actor: TeamActor,
    assisting_profile: Option<AgentProfile>,
    kind: CorrectionKind,
    summary: &str,
    detail: &str,
    history: &[EventEnvelope],
) -> EventChatter {
    let profile = assisting_profile
        .or_else(|| latest_agent_profile(history))
        .or_else(|| detail.contains("Planning").then_some(AgentProfile::Plan));
    let teammate_name = profile
        .map(AgentProfile::teammate_name)
        .unwrap_or("the model");
    let teammate_first_name = first_name(teammate_name);
    let artifact_label = match profile {
        Some(AgentProfile::Build) => "implementation report",
        Some(AgentProfile::Review) => "review",
        _ => "plan",
    };
    let normalized_summary = summary.trim();

    let (headline, message) = if kind == CorrectionKind::ArtifactValidation {
        (
            Some(format!(
                "{teammate_name}’s {artifact_label} needs another pass"
            )),
            format!(
                "{} I sent it back so you can correct the submission before the team continues.",
                artifact_validation_problem(detail)
            ),
        )
    } else if kind == CorrectionKind::ToolFailure
        && let Some(feedback) = tool_failure_feedback(detail, teammate_name)
    {
        (None, feedback)
    } else if kind == CorrectionKind::RepositoryEvidence {
        (
            None,
            format!(
                "{teammate_first_name}, I found the task-relevant code and pulled out the strongest matching sections. Use them to finish the {artifact_label}. If one concrete fact is still missing, read only the relevant lines instead of rereading the whole file."
            ),
        )
    } else if kind == CorrectionKind::ContractEvidence {
        (
            None,
            format!(
                "{teammate_first_name}, I rechecked the exact code this stage depends on. You have enough evidence now—finish the {artifact_label} instead of rereading broad sections of the repository."
            ),
        )
    } else if kind == CorrectionKind::WorkUnit
        && normalized_summary == "Active accepted-plan work unit"
    {
        (
            None,
            format!(
                "{teammate_first_name}, I picked the next item from the accepted plan and confirmed exactly which file operation it needs. Complete only that item before moving on."
            ),
        )
    } else if kind == CorrectionKind::WorkUnit {
        (
            None,
            format!(
                "{teammate_first_name}, the next planned file does not exist yet. Create it now with one complete write, then move on to the next item."
            ),
        )
    } else if kind == CorrectionKind::RuntimeFallback
        && normalized_summary.contains("using host execution")
    {
        (
            None,
            "This task includes an Apple-only component, so I’m running that part directly on the Mac while keeping the rest of the session isolated."
                .to_string(),
        )
    } else if kind == CorrectionKind::RuntimeFallback {
        (
            None,
            "The preferred model runtime was unavailable, so I’m using the CPU-only model fallback for this session. Responses may take longer."
                .to_string(),
        )
    } else if kind == CorrectionKind::RepeatedTool
        && normalized_summary.contains("reached the repeat limit")
    {
        (
            None,
            format!(
                "{teammate_first_name}, you repeated the same action after guidance, so I blocked the duplicate before you spent more time on it. Choose a different approach or report the blocker."
            ),
        )
    } else if kind == CorrectionKind::RepeatedTool {
        let message = repeated_tool_name(detail).map_or_else(
            || {
                format!(
                    "{teammate_first_name}, you repeated the same action, so I blocked the duplicate before it ran. Change approach or report that you are blocked."
                )
            },
            |tool| {
                format!(
                    "{teammate_first_name}, you repeated the same `{tool}` call, so I blocked the duplicate before it ran. Change the path or action, or report that you are blocked."
                )
            },
        );
        (None, message)
    } else if kind == CorrectionKind::NoProgress {
        (
            None,
            format!(
                "{teammate_first_name}, that action returned the same outcome without adding new evidence. I stopped the loop; choose an action that changes the work or report the blocker."
            ),
        )
    } else if kind == CorrectionKind::DependentToolBatch {
        (
            None,
            format!(
                "{teammate_first_name}, those tool calls depend on one another, so I did not run them as one batch. Run the prerequisite first, wait for its result, then submit the dependent action."
            ),
        )
    } else if kind == CorrectionKind::StageSubmission {
        (
            None,
            format!(
                "{teammate_first_name}, a prose reply will not complete this stage. Submit the {artifact_label} in the required format so the team can continue."
            ),
        )
    } else if kind == CorrectionKind::InvalidAction
        && (normalized_summary == "Teammate action retries exhausted"
            || normalized_summary.contains("Parse retry limit"))
    {
        (
            None,
            format!(
                "{teammate_first_name}, your reply still did not form a valid action after several retries, so I stopped the pass instead of letting it loop. Start again with one small, complete action."
            ),
        )
    } else if kind == CorrectionKind::InvalidAction {
        (
            None,
            format!(
                "{teammate_first_name}, that reply was not a valid action, so nothing ran. Retry with one complete tool call or finish the stage in the required format."
            ),
        )
    } else if kind == CorrectionKind::StepLimit {
        (
            None,
            format!(
                "{teammate_first_name}, you reached this pass’s step limit before completing the work, so I stopped it instead of letting it run in circles. Continue with a tighter next action or report the blocker."
            ),
        )
    } else if kind == CorrectionKind::AdvisoryBudget {
        (
            None,
            "I skipped the optional step-limit review because its advisory budget was already used. The main work and repository were left unchanged."
                .to_string(),
        )
    } else if kind == CorrectionKind::MissingEvidence {
        (
            None,
            format!(
                "{teammate_first_name}, the edit stopped before it became a valid action because one small file excerpt is still missing. Read only the lines around that detail, then retry the edit."
            ),
        )
    } else if kind == CorrectionKind::TruncatedAction {
        (
            None,
            format!(
                "{teammate_first_name}, the action was cut off before it became valid, so nothing ran. Try again once with one concise, complete tool call."
            ),
        )
    } else if kind == CorrectionKind::MutationRecovery {
        (
            None,
            format!(
                "{teammate_first_name}, the edit was incomplete and was not executed. I’m giving you one fresh attempt for the smallest complete change; do not repeat the rejected payload."
            ),
        )
    } else if kind == CorrectionKind::Lifecycle && normalized_summary == "Run cancelled" {
        (
            None,
            "This run was cancelled. I preserved the repository and the evidence collected so far."
                .to_string(),
        )
    } else if kind == CorrectionKind::Lifecycle && normalized_summary == "Goal pausing" {
        (
            None,
            "I’m pausing the goal at a safe checkpoint before anyone starts another action."
                .to_string(),
        )
    } else if kind == CorrectionKind::Lifecycle {
        (
            None,
            "Cancellation is requested. I’m preserving the repository and the workflow evidence while the current work stops safely."
                .to_string(),
        )
    } else if kind == CorrectionKind::TaskPlanningRecovery
        && normalized_summary == "Restarting delivery from current files"
    {
        (
            None,
            "I kept the earlier plan and review in the transcript, accepted the project’s current files as the new baseline, and started a fresh planning pass."
                .to_string(),
        )
    } else if kind == CorrectionKind::TaskPlanningRecovery
        && normalized_summary == "Retrying Task planning"
    {
        (
            None,
            "I’m retrying task planning from the preserved repository state. No files or commits change until the new workflow begins delivery."
                .to_string(),
        )
    } else if kind == CorrectionKind::TaskPlanningRecovery {
        (
            None,
            "I’m keeping the repository as-is and retrying this request as one Build task instead of splitting it into several tasks."
                .to_string(),
        )
    } else if kind == CorrectionKind::ToolUnavailable {
        (
            None,
            format!(
                "{teammate_first_name}, that tool is not available in this stage, so the action did not run. Choose one of the available actions or report the blocker."
            ),
        )
    } else if kind == CorrectionKind::RequirementsRemain {
        (
            None,
            format!(
                "{teammate_first_name}, the handoff still leaves part of the user’s request unfinished. I sent the missing requirements back for one focused repair pass."
            ),
        )
    } else if kind == CorrectionKind::Handoff
        && normalized_summary == "Handoff executor unavailable"
    {
        (
            None,
            "I could not run the final handoff checks because their executor is unavailable. I preserved the work so the checks can be retried when it returns."
                .to_string(),
        )
    } else if kind == CorrectionKind::Handoff && normalized_summary == "Handoff commit blocked" {
        (
            None,
            "The final commit was blocked, so I left the completed changes uncommitted and preserved the handoff evidence for review."
                .to_string(),
        )
    } else if kind == CorrectionKind::Handoff {
        (
            None,
            format!(
                "{teammate_first_name}, your pass ended before it produced a usable result. I preserved every completed action and stopped at the current safe boundary."
            ),
        )
    } else if kind == CorrectionKind::Diagnostics
        && normalized_summary == "Harness diagnostic preview"
    {
        (
            None,
            format!(
                "{teammate_first_name}, I ran an early diagnostic check and found issues you should account for while you complete the current work item."
            ),
        )
    } else if kind == CorrectionKind::Diagnostics {
        (
            None,
            format!(
                "{teammate_first_name}, the automatic diagnostics found issues that need repair before this stage can continue. Fix the reported problems, then retry the handoff."
            ),
        )
    } else if let Some(excerpt) = correction_excerpt(detail) {
        (
            (!normalized_summary.is_empty()).then(|| normalized_summary.to_string()),
            excerpt,
        )
    } else {
        (
            Some(if normalized_summary.is_empty() {
                "Trinity update".to_string()
            } else {
                normalized_summary.to_string()
            }),
            format!(
                "{teammate_first_name}, I could not summarize this safely without losing the exact cause. Check the technical details before choosing your next action."
            ),
        )
    };

    chatter(
        actor,
        headline,
        message,
        Some(detail.to_string()),
        ChatterAudience::Team,
    )
}

fn tool_failure_feedback(detail: &str, teammate_name: &str) -> Option<String> {
    let failure = serde_json::from_str::<Value>(detail).ok()?;
    let failure = failure.as_object()?;
    if failure.get("type")?.as_str()? != "tool_failure" {
        return None;
    }
    let tool = failure.get("tool")?.as_str()?.trim();
    if tool.is_empty() {
        return None;
    }
    let failure_message = failure
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let problem = extract_quoted_after(failure_message, "failed to resolve path '").map_or_else(
        || {
            if failure_message.to_lowercase().contains("permission denied") {
                "The requested resource could not be accessed.".to_string()
            } else {
                "The action failed before it returned a result.".to_string()
            }
        },
        |path| format!("`{path}` does not exist."),
    );
    Some(format!(
        "{}, your call to the `{tool}` tool was not executed successfully. {problem} Fix the mistake, choose a different action, or report the blocker.",
        first_name(teammate_name)
    ))
}

fn extract_quoted_after<'a>(value: &'a str, prefix: &str) -> Option<&'a str> {
    let start = value.find(prefix)? + prefix.len();
    let remainder = value.get(start..)?;
    let end = remainder.find('\'')?;
    remainder.get(..end)
}

fn repeated_tool_name(detail: &str) -> Option<&str> {
    detail.lines().find_map(|line| {
        let line = line.trim().strip_prefix("- ")?;
        let (tool, _) = line.split_once(" with args")?;
        (!tool.is_empty()
            && tool.chars().all(|character| {
                character == '_' || character == '-' || character.is_alphanumeric()
            }))
        .then_some(tool)
    })
}

fn artifact_validation_problem(message: &str) -> &'static str {
    let lower = message.to_lowercase();
    if lower.contains("requires non-empty requirements, steps, and acceptance") {
        "The plan was missing its requirements, implementation steps, and acceptance checks."
    } else if lower.contains("requirements") && lower.contains("acceptance") {
        "The plan did not include all of the required planning and acceptance information."
    } else if lower.contains("fingerprint") {
        "The submission described an older version of the workspace, so it was not safe to accept."
    } else if lower.contains("revision") && lower.contains("every assessment passes") {
        "The review asked for changes but marked every review area as passing."
    } else {
        "The submission did not match the delivery structure the team needs to continue safely."
    }
}

static CORRECTION_PARAGRAPH_BREAK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n\s*\n").expect("valid paragraph-break regex"));
static TEMPORARY_PATH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:/private)?/tmp/[^\s,;:)]+").expect("valid temporary-path regex")
});
static USER_WORKSPACE_PATH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"/Users/[^/\s]+/[^\s,;:)]+").expect("valid workspace-path regex"));

fn correction_excerpt(detail: &str) -> Option<String> {
    let trimmed = detail.trim();
    if trimmed.is_empty() || trimmed.starts_with('{') || trimmed.starts_with('[') {
        return None;
    }
    let first_paragraph = CORRECTION_PARAGRAPH_BREAK
        .split(trimmed)
        .next()
        .unwrap_or_default();
    let without_temporary_paths =
        TEMPORARY_PATH.replace_all(first_paragraph, "the temporary workspace");
    let without_workspace_paths =
        USER_WORKSPACE_PATH.replace_all(&without_temporary_paths, "the workspace");
    let normalized = without_workspace_paths
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let characters = normalized.chars().collect::<Vec<_>>();
    if characters.len() <= 360 {
        return Some(normalized);
    }
    let end = characters
        .windows(2)
        .enumerate()
        .take(340)
        .filter_map(|(index, pair)| (pair == ['.', ' ']).then_some(index + 1))
        .filter(|end| *end >= 120)
        .next_back()
        .unwrap_or(340);
    Some(format!(
        "{}…",
        characters[..end].iter().collect::<String>().trim_end()
    ))
}

fn workflow_blocked_chatter(
    outcome: crate::workflow::WorkflowOutcome,
    cause: WorkflowBlockCause,
    reason: &str,
    history: &[EventEnvelope],
) -> Vec<EventChatter> {
    let recent_history = history.iter().rev().take(5).collect::<Vec<_>>();
    let repeated_action = recent_history.iter().find_map(|envelope| {
        let AgentEvent::Correction {
            message,
            kind,
            assisting_profile,
            ..
        } = &envelope.event
        else {
            return None;
        };
        (*kind == CorrectionKind::RepeatedTool).then_some((message, *assisting_profile))
    });
    let recent_profile = history
        .iter()
        .rev()
        .find_map(|envelope| match &envelope.event {
            AgentEvent::LlmInvocation {
                profile: Some(profile),
                ..
            }
            | AgentEvent::StepStarted { profile, .. }
            | AgentEvent::ModelLoading { profile, .. }
            | AgentEvent::Reasoning { profile, .. }
            | AgentEvent::Started { profile, .. } => Some(*profile),
            AgentEvent::ToolCall {
                actor: Some(TeamActor::Agent(profile)),
                ..
            } => Some(*profile),
            _ => None,
        });
    let repeated_profile = repeated_action
        .and_then(|(_, profile)| profile)
        .or(recent_profile);
    let repeated_name = repeated_profile
        .map(AgentProfile::teammate_name)
        .unwrap_or("A teammate");
    let repeated_first_name = first_name(repeated_name);
    let planning_first_name = first_name(AgentProfile::Plan.teammate_name());
    let tool_calls = history
        .iter()
        .filter_map(|envelope| match &envelope.event {
            AgentEvent::ToolCall {
                tool, arguments, ..
            } => Some((tool, arguments)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let repeated_path = tool_calls
        .last()
        .and_then(|(latest_tool, latest_arguments)| {
            let repeated = tool_calls[..tool_calls.len().saturating_sub(1)]
                .iter()
                .any(|(tool, arguments)| tool == latest_tool && arguments == latest_arguments);
            repeated
                .then(|| latest_arguments.get("path").and_then(Value::as_str))
                .flatten()
        });

    let planning_failure = cause == WorkflowBlockCause::PlanningRejected
        || outcome == crate::workflow::WorkflowOutcome::PlanRejected;
    let git_control_changed = cause == WorkflowBlockCause::GitControlChanged;
    let repository_content_changed = cause == WorkflowBlockCause::RepositoryContentChanged;
    let repeat_limit = cause == WorkflowBlockCause::DeterministicRepeatLimit;
    let executor_unavailable = cause == WorkflowBlockCause::ExecutorUnavailable
        || outcome == crate::workflow::WorkflowOutcome::ExecutorUnavailable;
    let needs_current_files_restart = matches!(
        cause,
        WorkflowBlockCause::GitControlChanged
            | WorkflowBlockCause::RepositoryContentChanged
            | WorkflowBlockCause::CommitBlocked
    ) || outcome
        == crate::workflow::WorkflowOutcome::CommitBlocked;
    let teammate_message = if planning_failure {
        format!(
            "{planning_first_name}, your plan was rejected after three attempts. {} Nothing changed, and this delivery is now on hold.",
            artifact_validation_problem(reason)
        )
    } else if git_control_changed && repeated_action.is_some() {
        format!(
            "{repeated_first_name}, you repeated the same action while the repository’s Git state was changing. I blocked the duplicate and put this delivery on hold so we would not commit or overwrite somebody else’s work."
        )
    } else if git_control_changed {
        "Team, I put this delivery on hold because the repository’s Git state changed during the pass. I preserved the current files so we would not commit or overwrite somebody else’s work."
            .to_string()
    } else if repository_content_changed {
        "Team, I put this delivery on hold because the project changed while you were reviewing an earlier snapshot. I kept that review tied to its exact files rather than risk overwriting the newer work."
            .to_string()
    } else if repeat_limit {
        repeated_path.map_or_else(
            || format!(
                "{repeated_first_name}, you repeated the same action after I flagged the failure. I blocked the duplicate, so your task—and this delivery—are now on hold."
            ),
            |path| format!(
                "{repeated_first_name}, `{path}` does not exist. You tried to read it again after I flagged that, then repeated the same action once more. I blocked that last attempt, so your review—and this delivery—are now on hold."
            ),
        )
    } else if executor_unavailable {
        "Team, this delivery is on hold because a required executor is unavailable. I preserved the current work so you can continue once that prerequisite is restored."
            .to_string()
    } else {
        "Team, I put this delivery on hold at a safe boundary because the reported problem needs a different approach."
            .to_string()
    };
    let user_request = if needs_current_files_restart {
        "restart this delivery with the current files so the team can plan against the right project state"
    } else if executor_unavailable {
        "restore the missing prerequisite, then resume this delivery"
    } else {
        "start a follow-up task here and add any context that could help the team find a different way forward"
    };
    let detail = repeated_action.map_or_else(
        || reason.to_string(),
        |(message, _)| format!("{message}\n{reason}"),
    );

    vec![
        chatter(
            TeamActor::workflow_steward(),
            None,
            teammate_message,
            Some(detail),
            ChatterAudience::Team,
        ),
        chatter(
            TeamActor::workflow_steward(),
            None,
            format!("Can you {user_request}?"),
            None,
            ChatterAudience::CurrentUser,
        ),
    ]
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
    fn semantic_gate_receipt_round_trips_without_source_content() {
        let receipt = pb_control_collar::analysis::SemanticGateReceipt {
            contract_version: pb_control_collar::analysis::SEMANTIC_EVIDENCE_CONTRACT_VERSION,
            stage: pb_control_collar::analysis::SemanticEvidenceStage::FinalExecutor,
            scope: pb_control_collar::analysis::SemanticEvidenceScope::Document,
            workspace_sha256: "a".repeat(64),
            affected_documents: 1,
            providers: Vec::new(),
            viability: pb_control_collar::analysis::Viability::Unknown,
            closure: pb_control_collar::analysis::ClosureVerdict::Reject,
            definite_errors: Vec::new(),
            unknown_reasons: vec![pb_control_collar::analysis::UnknownReason::ProviderUnavailable],
            wall_millis: 3,
            budget_millis: 8_000,
        };
        receipt.validate().unwrap();
        let envelope = EventEnvelope::with_timestamp(AgentEvent::SemanticGate {
            receipt: receipt.clone(),
            nesting_depth: Some(1),
            timestamp_ms: None,
        });

        let serialized = serde_json::to_string(&envelope).unwrap();
        assert!(!serialized.contains("src/"));
        let restored: EventEnvelope = serde_json::from_str(&serialized).unwrap();
        assert!(matches!(
            restored.event,
            AgentEvent::SemanticGate {
                receipt: restored,
                nesting_depth: Some(1),
                timestamp_ms: Some(_),
            } if restored == receipt
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
    fn legacy_activity_events_default_to_the_build_profile() {
        for event in [
            r#"{"type":"model_loading","model":"legacy"}"#,
            r#"{"type":"step_started","step":1,"max_steps":2}"#,
        ] {
            let json = format!(r#"{{"version":"v1","event":{event}}}"#);
            let envelope: EventEnvelope = serde_json::from_str(&json).unwrap();
            assert!(matches!(
                envelope.event,
                AgentEvent::ModelLoading {
                    profile: AgentProfile::Build,
                    ..
                } | AgentEvent::StepStarted {
                    profile: AgentProfile::Build,
                    ..
                }
            ));
        }
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
            purpose: ModelInvocationPurpose::WorkflowReview,
            workflow_stage: Some(crate::workflow::WorkflowStage::CodeReview),
            profile: Some(AgentProfile::Review),
            duration_ms: 10,
            prompt_tokens: 4960,
            generated_tokens: 3,
            prompt_cache: Some(PromptCacheUsage {
                source: "memory_prefix".to_string(),
                cached_tokens: 4096,
                prefilled_tokens: 864,
                restore_ms: 0,
                miss_reason: Some(PromptCacheMissReason::PromptDiverged),
                lookup_detail: Some(
                    crate::inference::PromptCacheLookupDetail::SessionDivergedRootHit,
                ),
                root: Some(PromptRootUsage {
                    descriptor_version: 1,
                    backend: "flashmoe".to_string(),
                    cache_format_version: "flashmoe-session-v1".to_string(),
                    model_namespace_sha256: "model".to_string(),
                    rendered_token_sha256: "root".to_string(),
                    tokens: 2048,
                    reused_tokens: 2048,
                    system_instruction_version: Some("agent-system-v1".to_string()),
                    workflow_stage: Some(crate::workflow::WorkflowStage::CodeReview),
                    authority_class: PromptRootAuthorityClass::CodeReview,
                    tool_schema_sha256: Some("abc".to_string()),
                    output_constraint_mode: Some("tool_required".to_string()),
                }),
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
                prefill_command_reason: "fresh_suffix_at_or_above_threshold".to_string(),
                thinking_enabled: false,
                refill: Some(NativeRefillUsage {
                    cache_lookup_wall_ms: 2,
                    disk_read_decode_wall_ms: 4,
                    cpu_state_validation_allocation_wall_ms: 5,
                    state_hydration_wall_ms: 3,
                    fresh_suffix_prefill_wall_ms: 100,
                    snapshot_capture_wall_ms: 15,
                    persistence_queue_wall_ms: 6,
                }),
                tool_constraint_mode: Some("tool_required".to_string()),
                tool_constraint_dialect: Some("qwen_json".to_string()),
                tool_schema_sha256: Some("abc".to_string()),
                rejected_constraint_candidates: 4,
                mutation_constraint_rejections: BTreeMap::from([("invalid_syntax".to_string(), 2)]),
                mutation_snapshot_files: 1,
                mutation_snapshot_bytes: 128,
                constraint_terminal_state: Some("complete_tool_call".to_string()),
                constraint_guarantee_rung: Some("prefix_syntax".to_string()),
                semantic_boundary: None,
                decode_recovery: crate::inference::DecodeRecovery::CandidateProbeOnly,
            }),
            energy_joules: None,
            energy_kwh: None,
            average_power_watts: None,
            nesting_depth: None,
            timestamp_ms: None,
        });
        let json = serde_json::to_string(&envelope).unwrap();
        let restored: EventEnvelope = serde_json::from_str(&json).unwrap();
        let AgentEvent::LlmInvocation {
            native: Some(native),
            ..
        } = &restored.event
        else {
            panic!("expected native invocation usage");
        };
        assert_eq!(native.tool_constraint_dialect.as_deref(), Some("qwen_json"));
        assert_eq!(native.mutation_snapshot_files, 1);
        assert_eq!(native.mutation_snapshot_bytes, 128);
        assert_eq!(
            native.constraint_guarantee_rung.as_deref(),
            Some("prefix_syntax")
        );
        assert_eq!(
            native.mutation_constraint_rejections.get("invalid_syntax"),
            Some(&2)
        );
        assert!(matches!(
            restored.event,
            AgentEvent::LlmInvocation {
                context: Some(restored),
                prompt_cache: Some(PromptCacheUsage {
                    cached_tokens: 4096,
                    miss_reason: Some(PromptCacheMissReason::PromptDiverged),
                    root: Some(PromptRootUsage {
                        reused_tokens: 2048,
                        authority_class: PromptRootAuthorityClass::CodeReview,
                        ..
                    }),
                    ..
                }),
                native: Some(NativeGenerationUsage {
                    active_experts_per_token: Some(10),
                    prefill_metal_commands: 48,
                    prefill_host_upload_bytes: 1_024,
                    prefill_host_readback_bytes: 512,
                    refill: Some(NativeRefillUsage {
                        fresh_suffix_prefill_wall_ms: 100,
                        ..
                    }),
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
        native.remove("tool_constraint_dialect");
        native.remove("mutation_constraint_rejections");
        native.remove("mutation_snapshot_files");
        native.remove("mutation_snapshot_bytes");
        native.remove("constraint_guarantee_rung");
        native.remove("semantic_boundary");
        native.remove("decode_recovery");
        let legacy: EventEnvelope = serde_json::from_value(legacy).unwrap();
        let AgentEvent::LlmInvocation {
            native: Some(legacy_native),
            ..
        } = &legacy.event
        else {
            panic!("expected legacy native invocation usage");
        };
        assert!(legacy_native.mutation_constraint_rejections.is_empty());
        assert!(matches!(
            legacy.event,
            AgentEvent::LlmInvocation {
                native: Some(NativeGenerationUsage {
                    prefill_metal_commands: 0,
                    prefill_host_upload_bytes: 0,
                    prefill_host_readback_bytes: 0,
                    tool_constraint_dialect: None,
                    mutation_snapshot_files: 0,
                    mutation_snapshot_bytes: 0,
                    constraint_guarantee_rung: None,
                    semantic_boundary: None,
                    decode_recovery: crate::inference::DecodeRecovery::CandidateProbeOnly,
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
            purpose: crate::events::TeamMessagePurpose::General,
            handoff: None,
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
    fn correction_chatter_is_server_authored_and_round_trips() {
        let envelope = EventEnvelope::new(AgentEvent::Correction {
            kind: crate::events::CorrectionKind::General,
            message: "technical controller guidance".to_string(),
            summary: "Task-focused repository evidence".to_string(),
            actor: TeamActor::workflow_steward(),
            assisting_profile: Some(AgentProfile::Plan),
            nesting_depth: None,
            timestamp_ms: None,
        });

        assert_eq!(envelope.chatter.len(), 1);
        assert!(matches!(
            envelope.event,
            AgentEvent::Correction {
                kind: CorrectionKind::RepositoryEvidence,
                ..
            }
        ));
        let transcript = envelope.transcript.as_ref().unwrap();
        assert_eq!(transcript.visibility, TranscriptVisibility::Visible);
        assert_eq!(transcript.kind, TranscriptKind::Correction);
        assert_eq!(
            transcript.dedupe_key.as_deref(),
            Some("correction:Task-focused repository evidence")
        );
        assert_eq!(
            envelope.chatter[0].message,
            "Dade, I found the task-relevant code and pulled out the strongest matching sections. Use them to finish the plan. If one concrete fact is still missing, read only the relevant lines instead of rereading the whole file."
        );
        assert_eq!(
            envelope.chatter[0].detail.as_deref(),
            Some("technical controller guidance")
        );

        let restored: EventEnvelope =
            serde_json::from_str(&serde_json::to_string(&envelope).unwrap()).unwrap();
        assert_eq!(restored.chatter, envelope.chatter);
    }

    #[test]
    fn correction_chatter_turns_tool_failures_into_teammate_guidance() {
        let envelope = EventEnvelope::new(AgentEvent::Correction {
            kind: crate::events::CorrectionKind::General,
            message: serde_json::json!({
                "type": "tool_failure",
                "tool": "read_file",
                "message": "failed to resolve path 'webui/src/components/SessionRows.tsx': failed to resolve path /private/tmp/workspace/webui/src/components/SessionRows.tsx"
            })
            .to_string(),
            summary: "Tool not available".to_string(),
            actor: TeamActor::workflow_steward(),
            assisting_profile: Some(AgentProfile::Review),
            nesting_depth: None,
            timestamp_ms: None,
        });

        assert_eq!(
            envelope.chatter[0].message,
            "Eugene, your call to the `read_file` tool was not executed successfully. `webui/src/components/SessionRows.tsx` does not exist. Fix the mistake, choose a different action, or report the blocker."
        );
        assert!(!envelope.chatter[0].message.contains("/private/tmp"));
        assert_eq!(
            envelope
                .transcript
                .as_ref()
                .and_then(|metadata| metadata.dedupe_key.as_deref()),
            Some("tool_failure:read_file:missing:webui/src/components/SessionRows.tsx")
        );
    }

    #[test]
    fn workflow_blocked_chatter_uses_prior_events_for_terminal_parity() {
        let repeated_arguments = serde_json::json!({
            "path": "webui/src/components/SessionRows.tsx"
        });
        let mut history = vec![
            EventEnvelope::new(AgentEvent::ToolCall {
                tool: "read_file".to_string(),
                arguments: repeated_arguments.clone(),
                call_id: Some("call-1".to_string()),
                batch_id: None,
                actor: Some(TeamActor::agent(AgentProfile::Review)),
                nesting_depth: None,
                timestamp_ms: None,
            }),
            EventEnvelope::new(AgentEvent::ToolCall {
                tool: "read_file".to_string(),
                arguments: repeated_arguments,
                call_id: Some("call-2".to_string()),
                batch_id: None,
                actor: Some(TeamActor::agent(AgentProfile::Review)),
                nesting_depth: None,
                timestamp_ms: None,
            }),
            EventEnvelope::new(AgentEvent::Correction {
                kind: crate::events::CorrectionKind::General,
                message: "The repeated path still does not exist.".to_string(),
                summary: "Eugene reached the repeat limit".to_string(),
                actor: TeamActor::workflow_steward(),
                assisting_profile: Some(AgentProfile::Review),
                nesting_depth: None,
                timestamp_ms: None,
            }),
        ];
        refresh_event_projections(&mut history);
        let repeated_action_key = history[2]
            .transcript
            .as_ref()
            .and_then(|metadata| metadata.related_action_key.as_deref());
        assert!(repeated_action_key.is_some_and(|key| key.starts_with("tool:read_file:")));
        let reason = "Eugene stopped making progress in the CodeReview stage and reached a deterministic repeat limit";
        let mut blocked = EventEnvelope::new(AgentEvent::WorkflowBlocked {
            workflow_id: "workflow-1".to_string(),
            outcome: crate::workflow::WorkflowOutcome::ReviewFailed,
            cause: WorkflowBlockCause::DeterministicRepeatLimit,
            reason: reason.to_string(),
            timestamp_ms: None,
        });
        blocked.refresh_projections(&history);

        assert_eq!(blocked.chatter.len(), 2);
        assert_eq!(
            blocked.transcript.as_ref().map(|metadata| metadata.kind),
            Some(TranscriptKind::WorkflowBlocked)
        );
        assert_eq!(
            blocked.chatter[0].message,
            "Eugene, `webui/src/components/SessionRows.tsx` does not exist. You tried to read it again after I flagged that, then repeated the same action once more. I blocked that last attempt, so your review—and this delivery—are now on hold."
        );
        assert_eq!(
            blocked.chatter[0].detail.as_deref(),
            Some(
                "The repeated path still does not exist.\nEugene stopped making progress in the CodeReview stage and reached a deterministic repeat limit"
            )
        );
        assert_eq!(
            blocked.chatter[1].message,
            "Can you start a follow-up task here and add any context that could help the team find a different way forward?"
        );
        assert_eq!(blocked.chatter[1].audience, ChatterAudience::CurrentUser);
    }

    #[test]
    fn session_summary_projection_marks_repeated_terminal_copy() {
        let mut history = vec![
            EventEnvelope::new(AgentEvent::Final {
                content: "Implemented the event boundary.".to_string(),
                profile: AgentProfile::Build,
                nesting_depth: None,
                timestamp_ms: None,
            }),
            EventEnvelope::new(AgentEvent::SessionSummary {
                branch: "main".to_string(),
                commits: String::new(),
                reached_final: true,
                contract_status: ContractStatus::Satisfied,
                verified_completed: true,
                termination_reason: Some(TerminationReason::Final),
                handoff_outcome: None,
                summary: " Implemented the event boundary. ".to_string(),
                power_summary: String::new(),
                diff_stat: String::new(),
                diff: String::new(),
                timestamp_ms: None,
            }),
        ];

        refresh_event_projections(&mut history);

        assert_eq!(
            history[1]
                .transcript
                .as_ref()
                .map(|metadata| (metadata.kind, metadata.summary_redundant)),
            Some((TranscriptKind::SessionSummary, true))
        );
    }

    #[test]
    fn workflow_events_round_trip_with_typed_stage_and_outcome() {
        let envelope = EventEnvelope::with_timestamp(AgentEvent::WorkflowBlocked {
            workflow_id: "workflow-1".to_string(),
            outcome: crate::workflow::WorkflowOutcome::RepairCyclesExhausted,
            cause: WorkflowBlockCause::Other,
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
                cause: WorkflowBlockCause::Other,
                reason,
                timestamp_ms: Some(_),
            } if workflow_id == "workflow-1" && reason == "blocking findings remain"
        ));
    }

    #[test]
    fn replay_hydration_preserves_persisted_server_projections() {
        let mut envelope = EventEnvelope::new(AgentEvent::Correction {
            kind: CorrectionKind::RepositoryEvidence,
            message: "technical detail".to_string(),
            summary: "Task-focused repository evidence".to_string(),
            actor: TeamActor::workflow_steward(),
            assisting_profile: Some(AgentProfile::Review),
            nesting_depth: None,
            timestamp_ms: None,
        });
        envelope.chatter = vec![EventChatter {
            actor: TeamActor::workflow_steward(),
            tone: TeamMessageTone::Success,
            headline: Some("Persisted wording".to_string()),
            message: "This exact authored message must survive replay.".to_string(),
            detail: None,
            audience: ChatterAudience::Team,
        }];
        envelope.transcript = Some(TranscriptMetadata {
            visibility: TranscriptVisibility::EvidenceOnly,
            kind: TranscriptKind::Evidence,
            entry_key: "persisted-entry".to_string(),
            supersedes: vec!["older-entry".to_string()],
            tool_summary: None,
            dedupe_key: None,
            related_action_key: None,
            summary_redundant: false,
        });
        let mut events = vec![envelope];

        hydrate_event_projections(&mut events);

        assert_eq!(
            events[0].chatter[0].headline.as_deref(),
            Some("Persisted wording")
        );
        assert_eq!(
            events[0]
                .transcript
                .as_ref()
                .map(|metadata| metadata.entry_key.as_str()),
            Some("persisted-entry")
        );
    }

    #[test]
    fn tool_projection_owns_result_presentation() {
        let mut events = vec![
            EventEnvelope::new(AgentEvent::ToolCall {
                tool: "lsp_proactive_diagnostics".to_string(),
                arguments: serde_json::json!({
                    "mode": "settled",
                    "paths": ["src/lib.rs", "src/main.rs"]
                }),
                call_id: Some("lsp-1".to_string()),
                batch_id: None,
                actor: Some(TeamActor::workflow_steward()),
                nesting_depth: None,
                timestamp_ms: None,
            }),
            EventEnvelope::new(AgentEvent::ToolResult {
                tool: "lsp_proactive_diagnostics".to_string(),
                result: serde_json::json!({
                    "scanned_paths": ["src/lib.rs", "src/main.rs"],
                    "diagnostics": [{"path": "src/lib.rs"}],
                    "failures": [],
                    "omitted_paths": 3,
                    "stale": false
                })
                .to_string(),
                call_id: Some("lsp-1".to_string()),
                batch_id: None,
                outcome: Some(ToolOutcome::Failed),
                actor: Some(TeamActor::workflow_steward()),
                duration_ms: None,
                energy_joules: None,
                energy_kwh: None,
                average_power_watts: None,
                energy_shared_calls: None,
                nesting_depth: None,
                timestamp_ms: None,
            }),
        ];

        refresh_event_projections(&mut events);

        assert_eq!(
            events[1]
                .transcript
                .as_ref()
                .and_then(|metadata| metadata.tool_summary.as_deref()),
            Some("settled · 1 blocking diagnostic in 2 files · 3 deferred")
        );
    }

    #[test]
    fn handoff_outcome_explicitly_supersedes_progress() {
        let summary = HandoffSummary {
            outcome: HandoffOutcome::Ready,
            affected_components: vec!["web".to_string()],
            checks: Vec::new(),
            commit: None,
            changed_paths: vec!["webui/src/App.tsx".to_string()],
            detail: None,
        };
        let mut events = vec![
            EventEnvelope::new(AgentEvent::TeamMessage {
                actor: TeamActor::workflow_steward(),
                tone: TeamMessageTone::Info,
                purpose: TeamMessagePurpose::HandoffProgress,
                handoff: None,
                message: "Checking the web app.".to_string(),
                detail: None,
                evidence_ids: Vec::new(),
                nesting_depth: None,
                timestamp_ms: None,
            }),
            EventEnvelope::new(AgentEvent::TeamMessage {
                actor: TeamActor::workflow_steward(),
                tone: TeamMessageTone::Success,
                purpose: TeamMessagePurpose::HandoffOutcome,
                handoff: Some(summary),
                message: "The web app is ready.".to_string(),
                detail: None,
                evidence_ids: Vec::new(),
                nesting_depth: None,
                timestamp_ms: None,
            }),
        ];

        refresh_event_projections(&mut events);
        let progress_key = events[0]
            .transcript
            .as_ref()
            .map(|metadata| metadata.entry_key.clone())
            .unwrap();
        assert_eq!(
            events[1]
                .transcript
                .as_ref()
                .map(|metadata| metadata.supersedes.as_slice()),
            Some([progress_key].as_slice())
        );
    }
}
