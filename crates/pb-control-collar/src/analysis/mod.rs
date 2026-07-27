use serde::{Deserialize, Serialize};

use crate::{CollarResult, mutation::LogicalPath};

mod syntax;
pub use syntax::{SyntaxProfile, SyntaxReport, validate_supported_syntax};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LanguageId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageProfile {
    pub id: LanguageId,
    pub version: String,
    pub extensions: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramFile {
    pub path: LogicalPath,
    pub language: LanguageId,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProgramSnapshot {
    pub files: Vec<ProgramFile>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceOrigin {
    Known,
    Generated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisBoundary {
    Expression,
    Statement,
    Item,
    Function,
    File,
    ToolArgument,
    ToolCall,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceEvent<'a> {
    BeginFile {
        path: &'a LogicalPath,
        language: &'a LanguageId,
    },
    Bytes {
        origin: SourceOrigin,
        bytes: &'a [u8],
    },
    DeleteKnownBytes(&'a [u8]),
    Boundary(AnalysisBoundary),
    EndFile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Viability {
    Valid,
    Repairable,
    Impossible,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClosureVerdict {
    Allow,
    Reject,
    Defer,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticObligation {
    pub kind: String,
    pub boundary: AnalysisBoundary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairIntent {
    pub kind: String,
    pub boundary: AnalysisBoundary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Analysis {
    pub viability: Viability,
    pub closure: ClosureVerdict,
    pub obligations: Vec<SemanticObligation>,
    pub biases: Vec<RepairIntent>,
}

pub trait IncrementalAnalyzer {
    type Checkpoint: Copy;

    fn begin(&mut self, snapshot: ProgramSnapshot) -> CollarResult<()>;
    fn checkpoint(&mut self) -> Self::Checkpoint;
    fn apply(&mut self, event: SourceEvent<'_>) -> CollarResult<Analysis>;
    fn rollback(&mut self, checkpoint: Self::Checkpoint) -> CollarResult<()>;
    fn finalize(&mut self) -> CollarResult<Analysis>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeRecovery {
    CandidateProbeOnly,
    ReplayFromBoundary,
    SnapshotAndRestore,
}
