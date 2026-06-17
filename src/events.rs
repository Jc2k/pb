use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent_core::AgentProfile;
use crate::session_store::now_millis;

pub const EVENT_SCHEMA_VERSION: &str = "v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    Started {
        task: String,
        model: String,
        workspace: String,
        branch: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    StepStarted {
        step: usize,
        max_steps: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    Reasoning {
        content: String,
        profile: AgentProfile,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    ToolCall {
        tool: String,
        arguments: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    ToolResult {
        tool: String,
        result: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    UserQuestion {
        question_id: String,
        question: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    UserAnswer {
        question_id: String,
        answer: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    SubAgentStarted {
        profile: String,
        task: String,
        nesting_depth: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    SubAgentFinished {
        profile: String,
        result: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    Diff {
        path: String,
        diff: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    Final {
        content: String,
        profile: AgentProfile,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    SessionSummary {
        branch: String,
        commits: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
    },
    Error {
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<u128>,
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
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Started {
                    task,
                    model,
                    workspace,
                    branch,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::StepStarted {
                step, max_steps, ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::StepStarted {
                    step,
                    max_steps,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::Reasoning {
                content, profile, ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Reasoning {
                    content,
                    profile,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ToolCall {
                tool, arguments, ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ToolCall {
                    tool,
                    arguments,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::ToolResult { tool, result, .. } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::ToolResult {
                    tool,
                    result,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::UserQuestion {
                question_id,
                question,
                ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::UserQuestion {
                    question_id,
                    question,
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
                profile, result, ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::SubAgentFinished {
                    profile,
                    result,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::Diff { path, diff, .. } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Diff {
                    path,
                    diff,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::Final {
                content, profile, ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Final {
                    content,
                    profile,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::SessionSummary {
                branch, commits, ..
            } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::SessionSummary {
                    branch,
                    commits,
                    timestamp_ms: Some(now),
                },
            },
            AgentEvent::Error { message, .. } => Self {
                version: EVENT_SCHEMA_VERSION.to_string(),
                event: AgentEvent::Error {
                    message,
                    timestamp_ms: Some(now),
                },
            },
        }
    }
}
