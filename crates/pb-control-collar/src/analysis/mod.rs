use serde::{Deserialize, Serialize};

use crate::{
    CollarResult,
    mutation::{LogicalPath, MutationKind},
};

mod layers;
mod prefix;
mod semantic;
mod syntax;
pub use layers::{AnalyzerCheckpoint, LanguageLayerStack, LayerStackCheckpoint};
pub use prefix::{
    PrefixCheckpoint, PrefixReport, PrefixRule, SourcePrefixOracle, validate_supported_prefix,
};
pub use semantic::{
    BaselineCompleteness, DiagnosticDelta, DiagnosticIdentity, SEMANTIC_EVIDENCE_CONTRACT_VERSION,
    SEMANTIC_WORLD_CONTRACT_VERSION, SemanticDiagnosticSnapshot, SemanticEnforcement,
    SemanticEvidenceScope, SemanticEvidenceStage, SemanticFileBinding, SemanticGateReceipt,
    SemanticProviderEvidence, SemanticWorldSnapshot, diagnostic_delta,
};
pub use syntax::{
    IncrementalSyntaxCheckpoint, IncrementalSyntaxReport, IncrementalSyntaxTree, SyntaxProfile,
    SyntaxReport, validate_supported_syntax,
};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuaranteeRung {
    CompleteSyntax,
    ConservativePrefix,
    ScopedSymbols,
    ScopedTypes,
    BackendParity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalyzerCapability {
    PrefixLexical,
    PrefixStructural,
    SyntaxBoundary,
    SymbolResolution,
    TypeChecking,
    OwnershipChecking,
    DependencyResolution,
    FinalWorkspaceGate,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SemanticWorldId {
    pub provider: String,
    pub provider_version: String,
    pub world_sha256: String,
    pub configuration_sha256: String,
    pub dependency_sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefiniteErrorClass {
    UnresolvedName,
    UnresolvedImport,
    MissingField,
    MissingMethod,
    Privacy,
    TypeMismatch,
    InvalidCall,
    Mutability,
    Ownership,
    Configuration,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownReason {
    IncompleteBaseline,
    UnsupportedConstruct,
    DynamicType,
    MissingDependency,
    ProviderUnavailable,
    ProviderRestarted,
    StaleDocument,
    Timeout,
    BudgetExceeded,
    ConfigurationChanged,
    UnqualifiedProfile,
    UnclassifiedDiagnostic,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryProbe {
    pub world: SemanticWorldId,
    pub path: LogicalPath,
    pub content_sha256: String,
    pub boundary: AnalysisBoundary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderVerdict {
    pub viability: Viability,
    pub closure: ClosureVerdict,
    pub definite_errors: Vec<DefiniteErrorClass>,
    pub unknown_reasons: Vec<UnknownReason>,
    pub obligations: Vec<SemanticObligation>,
    pub biases: Vec<RepairIntent>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceEvent<'a> {
    BeginFile {
        path: &'a LogicalPath,
        language: &'a LanguageId,
        mutation: MutationKind,
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

impl Analysis {
    /// Compose independently owned language/syntax/policy layers. A proven impossibility or hard
    /// closure rejection dominates; an unknown layer prevents a stronger validity claim but never
    /// erases another layer's proof of impossibility.
    pub fn compose(layers: impl IntoIterator<Item = Self>) -> Self {
        let mut viability = Viability::Valid;
        let mut closure = ClosureVerdict::Allow;
        let mut obligations = Vec::new();
        let mut biases = Vec::new();
        for layer in layers {
            viability = compose_viability(viability, layer.viability);
            closure = compose_closure(closure, layer.closure);
            obligations.extend(layer.obligations);
            biases.extend(layer.biases);
        }
        Self {
            viability,
            closure,
            obligations,
            biases,
        }
    }
}

fn compose_viability(left: Viability, right: Viability) -> Viability {
    if left == Viability::Impossible || right == Viability::Impossible {
        Viability::Impossible
    } else if left == Viability::Unknown || right == Viability::Unknown {
        Viability::Unknown
    } else if left == Viability::Repairable || right == Viability::Repairable {
        Viability::Repairable
    } else {
        Viability::Valid
    }
}

fn compose_closure(left: ClosureVerdict, right: ClosureVerdict) -> ClosureVerdict {
    if left == ClosureVerdict::Reject || right == ClosureVerdict::Reject {
        ClosureVerdict::Reject
    } else if left == ClosureVerdict::Defer || right == ClosureVerdict::Defer {
        ClosureVerdict::Defer
    } else {
        ClosureVerdict::Allow
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerReadiness {
    Cold,
    Warming,
    Ready,
    Stale,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessOrigin {
    ColdBuild,
    WarmCache,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticCompleteness {
    Complete,
    Partial,
}

/// Content-free evidence that a language world was established before inference. The controller
/// may persist this receipt, but never analyzer databases, source names, or source text in events.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerReadinessReceipt {
    pub world: SemanticWorldId,
    pub origin: ReadinessOrigin,
    pub completeness: SemanticCompleteness,
    pub load_millis: u64,
    pub prime_millis: u64,
    pub primed_queries: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalyzerLayerDescriptor {
    pub id: String,
    pub language: LanguageId,
    pub world: SemanticWorldId,
    pub capabilities: Vec<AnalyzerCapability>,
}

pub trait IncrementalAnalyzer {
    fn descriptor(&self) -> &AnalyzerLayerDescriptor;
    fn readiness(&self) -> LayerReadiness;
    fn readiness_receipt(&self) -> Option<&LayerReadinessReceipt>;
    fn begin(&mut self, snapshot: ProgramSnapshot) -> CollarResult<()>;
    fn checkpoint(&mut self) -> CollarResult<AnalyzerCheckpoint>;
    fn apply(&mut self, event: SourceEvent<'_>) -> CollarResult<Analysis>;
    fn rollback(&mut self, checkpoint: AnalyzerCheckpoint) -> CollarResult<()>;
    fn finalize(&mut self) -> CollarResult<Analysis>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeRecovery {
    CandidateProbeOnly,
    ReplayFromBoundary,
    SnapshotAndRestore,
}

#[cfg(test)]
mod layer_tests {
    use super::*;

    fn analysis(viability: Viability, closure: ClosureVerdict) -> Analysis {
        Analysis {
            viability,
            closure,
            obligations: Vec::new(),
            biases: Vec::new(),
        }
    }

    #[test]
    fn layer_composition_preserves_rejection_and_unknown_evidence() {
        let rejected = Analysis::compose([
            analysis(Viability::Unknown, ClosureVerdict::Defer),
            analysis(Viability::Impossible, ClosureVerdict::Reject),
        ]);
        assert_eq!(rejected.viability, Viability::Impossible);
        assert_eq!(rejected.closure, ClosureVerdict::Reject);

        let unknown = Analysis::compose([
            analysis(Viability::Valid, ClosureVerdict::Allow),
            analysis(Viability::Unknown, ClosureVerdict::Allow),
        ]);
        assert_eq!(unknown.viability, Viability::Unknown);
        assert_eq!(unknown.closure, ClosureVerdict::Allow);
    }
}
