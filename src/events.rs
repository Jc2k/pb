use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::agent_core::{AgentProfile, SessionAttachment};
use crate::session_store::now_millis;

pub const EVENT_SCHEMA_VERSION: &str = "v1";

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
    ResourceLimit,
    InvocationLimit,
    TokenLimit,
    EngineError,
    ChecksFailed,
    ExecutorUnavailable,
    RepairExhausted,
    CommitBlocked,
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
    Handoff,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum TeamActor {
    Agent(AgentProfile),
    Automation(AutomationActor),
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
            Self::ResourceLimit => "resource_limit",
            Self::InvocationLimit => "invocation_limit",
            Self::TokenLimit => "token_limit",
            Self::EngineError => "engine_error",
            Self::ChecksFailed => "checks_failed",
            Self::ExecutorUnavailable => "executor_unavailable",
            Self::RepairExhausted => "repair_exhausted",
            Self::CommitBlocked => "commit_blocked",
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
            })
        } else {
            None
        }
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
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    ToolResult {
        tool: String,
        result: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
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
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ToolCall {
                    tool,
                    arguments,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ToolResult {
                tool,
                result,
                duration_ms,
                energy_joules,
                energy_kwh,
                average_power_watts,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ToolResult {
                    tool,
                    result,
                    duration_ms,
                    energy_joules,
                    energy_kwh,
                    average_power_watts,
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
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Correction {
                    message,
                    summary,
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
}
