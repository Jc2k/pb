pub mod backend;
pub mod chat_template;
pub mod flashmoe;
pub mod llamacpp;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PROMPT_ROOT_DESCRIPTOR_VERSION: u32 = 1;
pub const AGENT_SYSTEM_INSTRUCTION_VERSION: &str = "agent-system-v5";

/// Controller-owned blocking semantic probe. Backends call it only when a candidate closes a
/// mutation payload or another promoted semantic boundary. Implementations must be deterministic
/// for identical tool arguments and bind their result to an immutable semantic world.
pub trait SemanticBoundaryProvider: Send + Sync {
    fn probe(
        &self,
        tool: &str,
        arguments: &serde_json::Value,
    ) -> pb_control_collar::analysis::ProviderVerdict;

    fn evidence(&self) -> Option<pb_control_collar::analysis::SemanticGateReceipt> {
        None
    }
}

struct SemanticBoundaryInner {
    provider: Box<dyn SemanticBoundaryProvider>,
    probes: AtomicU64,
    allows: AtomicU64,
    rejects: AtomicU64,
    defers: AtomicU64,
    wall_nanos: AtomicU64,
}

#[derive(Clone)]
pub struct SemanticBoundaryControl(Arc<SemanticBoundaryInner>);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticBoundaryStats {
    pub probes: u64,
    pub allows: u64,
    pub rejects: u64,
    pub defers: u64,
    pub wall_millis: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt: Option<pb_control_collar::analysis::SemanticGateReceipt>,
}

/// Strongest decode-state recovery contract implemented and qualified by a backend. Candidate
/// probing is sufficient for the current blocking semantic boundary; the stronger variants are
/// reserved for bounded speculative decoding and must never be inferred from a KV cache alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeRecovery {
    #[default]
    CandidateProbeOnly,
    ReplayFromBoundary,
    SnapshotAndRestore,
}

impl SemanticBoundaryControl {
    pub fn new(provider: impl SemanticBoundaryProvider + 'static) -> Self {
        Self(Arc::new(SemanticBoundaryInner {
            provider: Box::new(provider),
            probes: AtomicU64::new(0),
            allows: AtomicU64::new(0),
            rejects: AtomicU64::new(0),
            defers: AtomicU64::new(0),
            wall_nanos: AtomicU64::new(0),
        }))
    }

    pub fn probe(
        &self,
        tool: &str,
        arguments: &serde_json::Value,
    ) -> pb_control_collar::analysis::ProviderVerdict {
        let started = Instant::now();
        let verdict = self.0.provider.probe(tool, arguments);
        self.0.probes.fetch_add(1, Ordering::Relaxed);
        match verdict.closure {
            pb_control_collar::analysis::ClosureVerdict::Allow => &self.0.allows,
            pb_control_collar::analysis::ClosureVerdict::Reject => &self.0.rejects,
            pb_control_collar::analysis::ClosureVerdict::Defer => &self.0.defers,
        }
        .fetch_add(1, Ordering::Relaxed);
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.0.wall_nanos.fetch_add(nanos, Ordering::Relaxed);
        verdict
    }

    pub fn stats(&self) -> SemanticBoundaryStats {
        SemanticBoundaryStats {
            probes: self.0.probes.load(Ordering::Relaxed),
            allows: self.0.allows.load(Ordering::Relaxed),
            rejects: self.0.rejects.load(Ordering::Relaxed),
            defers: self.0.defers.load(Ordering::Relaxed),
            wall_millis: self.0.wall_nanos.load(Ordering::Relaxed) / 1_000_000,
            receipt: self.0.provider.evidence(),
        }
    }
}

impl std::fmt::Debug for SemanticBoundaryControl {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SemanticBoundaryControl(..)")
    }
}

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
    use super::{SemanticBoundaryControl, SemanticBoundaryProvider, rendered_token_sha256};
    use pb_control_collar::analysis::{ClosureVerdict, ProviderVerdict, Viability};

    struct AllowBoundary;

    impl SemanticBoundaryProvider for AllowBoundary {
        fn probe(&self, _tool: &str, _arguments: &serde_json::Value) -> ProviderVerdict {
            ProviderVerdict {
                viability: Viability::Valid,
                closure: ClosureVerdict::Allow,
                definite_errors: Vec::new(),
                unknown_reasons: Vec::new(),
                obligations: Vec::new(),
                biases: Vec::new(),
            }
        }
    }

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

    #[test]
    fn semantic_boundary_records_content_free_outcome_counts() {
        let boundary = SemanticBoundaryControl::new(AllowBoundary);
        assert_eq!(
            boundary.probe("write_file", &serde_json::json!({})).closure,
            ClosureVerdict::Allow
        );
        let stats = boundary.stats();
        assert_eq!(stats.probes, 1);
        assert_eq!(stats.allows, 1);
        assert_eq!(stats.rejects, 0);
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
