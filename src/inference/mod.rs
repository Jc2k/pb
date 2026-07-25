pub mod backend;
pub mod chat_template;
pub mod flashmoe;
pub mod llamacpp;

use serde::{Deserialize, Serialize};

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
