mod artifacts;
mod capabilities;
mod config;
mod engine;
mod persistence;

pub use artifacts::*;
pub use capabilities::*;
pub use config::*;
pub use engine::*;
pub use persistence::*;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TurnIntent {
    #[default]
    Discuss,
    Deliver,
    Auto,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStage {
    Planning,
    PlanReview,
    PlanRevision,
    Implementing,
    Checking,
    CodeReview,
    Repairing,
    Committing,
    Ready,
    Failed,
    Blocked,
    Cancelled,
}

impl WorkflowStage {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Ready | Self::Failed | Self::Blocked | Self::Cancelled
        )
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowOutcome {
    Ready,
    NoChange,
    PlanRejected,
    PlanCyclesExhausted,
    ChecksFailed,
    ReviewFailed,
    RepairCyclesExhausted,
    ExecutorUnavailable,
    CommitBlocked,
    StepLimit,
    InvocationLimit,
    TokenLimit,
    EngineError,
    Cancelled,
}
