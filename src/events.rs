use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent_core::{AgentProfile, SessionAttachment};
use crate::session_store::now_millis;

pub const EVENT_SCHEMA_VERSION: &str = "v1";

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
        #[serde(default, skip_serializing_if = "String::is_empty")]
        summary: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        diff_stat: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        diff: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
    Error {
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        nesting_depth: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u64>,
    },
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
                branch,
                attachments,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Started {
                    task,
                    model,
                    workspace,
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
                summary,
                diff_stat,
                diff,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::SessionSummary {
                    branch,
                    commits,
                    summary,
                    diff_stat,
                    diff,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::Error {
                message,
                nesting_depth,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Error {
                    message,
                    nesting_depth,
                    timestamp_ms: Some(now),
                },
            },
        }
    }
}
