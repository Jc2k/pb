pub mod backend;
pub mod chat_template;
pub mod flashmoe;
pub mod llamacpp;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PROMPT_ROOT_DESCRIPTOR_VERSION: u32 = 1;
pub const AGENT_SYSTEM_INSTRUCTION_VERSION: &str = "agent-system-v5";

pub(crate) fn rendered_token_sha256(tokens: impl IntoIterator<Item = u32>) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// Controller-owned, bounded authority represented by a stable prompt root.
///
/// This is diagnostic metadata only. Backends continue to require an exact rendered-token match
/// under the model namespace before restoring inference state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageRootAuthorityClass {
    #[default]
    Unclassified,
    Conversation,
    TaskArtifact,
    Planning,
    PlanningEvidence,
    PlanningClosure,
    PlanReview,
    PlanReviewEvidence,
    PlanReviewClosure,
    ImplementationRead,
    ImplementationMutation,
    ImplementationClosure,
    RepairRead,
    RepairMutation,
    RepairClosure,
    CodeReview,
    CodeReviewEvidence,
    CodeReviewClosure,
}

/// Decode-time constraint bound to a managed invocation independently of reusable KV state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageRootConstraintMode {
    #[default]
    None,
    Unconstrained,
    ToolsAllowed,
    ToolRequired,
    JsonSchema,
}

impl StageRootConstraintMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Unconstrained => "unconstrained",
            Self::ToolsAllowed => "tools_allowed",
            Self::ToolRequired => "tool_required",
            Self::JsonSchema => "json_schema",
        }
    }
}

/// Controller-owned description of the stable authority sent to an inference backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageRootDescriptor {
    pub descriptor_version: u32,
    pub system_instruction_version: String,
    pub workflow_stage: Option<crate::workflow::WorkflowStage>,
    pub authority_class: StageRootAuthorityClass,
    pub tool_schema_sha256: Option<String>,
    pub output_constraint_mode: StageRootConstraintMode,
}

/// Backend-owned identity for the exact rendered token prefix eligible for cross-session reuse.
///
/// The controller may attach diagnostic stage and authority labels to this identity, but cache
/// reuse remains owned by the backend's exact token comparison under its model namespace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendPromptRoot {
    pub descriptor_version: u32,
    pub backend: String,
    #[serde(default)]
    pub cache_format_version: String,
    pub model_namespace_sha256: String,
    pub rendered_token_sha256: String,
    pub tokens: usize,
    /// Present for managed controller invocations. Raw inference surfaces intentionally omit it.
    #[serde(default)]
    pub stage: Option<StageRootDescriptor>,
}

/// Backend-owned explanation for a generation that reused no prompt prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheMissReason {
    CacheDisabled,
    ColdSession,
    PromptDiverged,
    StablePrefixUnavailable,
    CacheUnreadable,
    ContextReset,
    RuntimeUnsupported,
}

impl PromptCacheMissReason {
    /// Stable event and UI label for this reason.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CacheDisabled => "cache_disabled",
            Self::ColdSession => "cold_session",
            Self::PromptDiverged => "prompt_diverged",
            Self::StablePrefixUnavailable => "stable_prefix_unavailable",
            Self::CacheUnreadable => "cache_unreadable",
            Self::ContextReset => "context_reset",
            Self::RuntimeUnsupported => "runtime_unsupported",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::rendered_token_sha256;

    #[test]
    fn rendered_root_identity_is_exact_at_one_token() {
        assert_eq!(
            rendered_token_sha256([1, 2, 3]),
            rendered_token_sha256([1, 2, 3])
        );
        assert_ne!(
            rendered_token_sha256([1, 2, 3]),
            rendered_token_sha256([1, 2, 4])
        );
    }
}

/// Bounded, privacy-safe cache lookup evidence used to diagnose a public miss reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheLookupDetail {
    SessionCheckpointMissing,
    SessionCheckpointDiverged,
    ExactRootCheckpointMissing,
    SessionDivergedRootMissing,
    SessionDivergedRootHit,
}
