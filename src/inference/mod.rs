pub mod backend;
pub mod chat_template;
pub mod flashmoe;
pub mod llamacpp;

use serde::{Deserialize, Serialize};

pub const PROMPT_ROOT_DESCRIPTOR_VERSION: u32 = 1;

/// Backend-owned identity for the exact rendered token prefix eligible for cross-session reuse.
///
/// The controller may attach diagnostic stage and authority labels to this identity, but cache
/// reuse remains owned by the backend's exact token comparison under its model namespace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendPromptRoot {
    pub descriptor_version: u32,
    pub backend: String,
    pub model_namespace_sha256: String,
    pub rendered_token_sha256: String,
    pub tokens: usize,
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
