//! Deterministic generation-time constraints for pb tool output.
//!
//! The collar owns token, wire-protocol, virtual-mutation, and analysis state. It deliberately does
//! not own model execution, live filesystem access, tool authority, or mutation publication.

#![forbid(unsafe_code)]

pub mod analysis;
pub mod diagnostics;
pub mod gate;
pub mod json;
pub mod mask;
pub mod mutation;
pub mod protocol;
pub mod receipt;
pub mod tool;
pub mod vocabulary;

pub use diagnostics::{CollarError, CollarResult};
pub use gate::{
    CompletionDecision, MutationCandidateCheckpoint, MutationCompletionGate, RejectionCode,
};
pub use json::{JsonConstraintFactory, JsonConstraintSession, validate_llguidance_json_schema};
pub use mask::TokenMask;
