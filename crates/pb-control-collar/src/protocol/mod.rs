use serde::{Deserialize, Serialize};

use crate::{CollarResult, TokenMask};

pub mod dsml;

pub use dsml::{CanonicalToolCall, DsmlConstraint, DsmlParseOutput, DsmlProbe, parse_dsml_output};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolDialect {
    QwenJson,
    DeepSeekDsml,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgumentEncoding {
    JsonString,
    JsonValue,
    DsmlRawString,
    DsmlJson,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolEvent {
    PreludeBytes(Vec<u8>),
    BeginCall {
        name: String,
    },
    BeginArgument {
        name: String,
        encoding: ArgumentEncoding,
    },
    ArgumentBytes(Vec<u8>),
    EndArgument,
    EndCall,
    EndBatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    TokenLimit,
    Cancelled,
    SemanticStop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    CompleteTerminalCall,
    CompleteBatch,
    MutationPayloadLimit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenBias {
    pub token: u32,
    pub logit_delta: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstraintStep {
    pub hard_mask: TokenMask,
    pub logit_biases: Vec<TokenBias>,
    pub stop: Option<StopReason>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenDecision {
    Allow,
    Reject,
    Defer,
}

pub trait ConstraintSession {
    type Receipt;

    fn next(&mut self) -> CollarResult<ConstraintStep>;
    fn probe(&mut self, token: u32) -> CollarResult<TokenDecision>;
    fn commit(&mut self, token: u32) -> CollarResult<Vec<ToolEvent>>;
    fn finish(&mut self, reason: FinishReason) -> CollarResult<Self::Receipt>;
}
