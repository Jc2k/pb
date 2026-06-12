use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const EVENT_SCHEMA_VERSION: &str = "v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    Started {
        task: String,
        model: String,
        workspace: String,
        branch: String,
    },
    StepStarted {
        step: usize,
        max_steps: usize,
    },
    Reasoning {
        content: String,
    },
    ToolCall {
        tool: String,
        arguments: Value,
    },
    ToolResult {
        tool: String,
        result: String,
    },
    SubAgentStarted {
        profile: String,
        task: String,
    },
    SubAgentFinished {
        profile: String,
        result: String,
    },
    Diff {
        path: String,
        diff: String,
    },
    Final {
        content: String,
    },
    SessionSummary {
        branch: String,
        commits: String,
    },
    Error {
        message: String,
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
}
