use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::{CollarError, CollarResult, mutation::LogicalPath, receipt::Digest};

use super::{
    AnalyzerCapability, ClosureVerdict, DefiniteErrorClass, SemanticWorldId, UnknownReason,
    Viability,
};

pub const SEMANTIC_WORLD_CONTRACT_VERSION: u32 = 1;
pub const SEMANTIC_EVIDENCE_CONTRACT_VERSION: u32 = 1;
const MAX_SEMANTIC_EVIDENCE_PROVIDERS: usize = 64;
const MAX_SEMANTIC_EVIDENCE_DOCUMENTS: usize = 1_024;
const MAX_SEMANTIC_EVIDENCE_CLASSES: usize = 64;
const MAX_SEMANTIC_PROVIDER_NAME_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticEvidenceStage {
    GenerationBoundary,
    FinalExecutor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticEvidenceScope {
    Document,
    AffectedTargets,
    CompleteProject,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticProviderEvidence {
    pub provider: String,
    pub provider_version: String,
    pub world_sha256: String,
    pub configuration_sha256: String,
    pub dependency_sha256: String,
    pub baseline: BaselineCompleteness,
    pub document_count: usize,
    pub introduced_diagnostics: usize,
    pub resolved_diagnostics: usize,
    pub unchanged_diagnostics: usize,
    pub authoritative: bool,
    pub definite_errors: Vec<DefiniteErrorClass>,
    pub unknown_reasons: Vec<UnknownReason>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticGateReceipt {
    pub contract_version: u32,
    pub stage: SemanticEvidenceStage,
    pub scope: SemanticEvidenceScope,
    pub workspace_sha256: String,
    pub affected_documents: usize,
    pub providers: Vec<SemanticProviderEvidence>,
    pub viability: Viability,
    pub closure: ClosureVerdict,
    pub definite_errors: Vec<DefiniteErrorClass>,
    pub unknown_reasons: Vec<UnknownReason>,
    pub wall_millis: u64,
    pub budget_millis: u64,
}

impl SemanticGateReceipt {
    pub fn validate(&self) -> CollarResult<()> {
        let provider_names = self
            .providers
            .iter()
            .map(|provider| provider.provider.as_str())
            .collect::<BTreeSet<_>>();
        if self.contract_version != SEMANTIC_EVIDENCE_CONTRACT_VERSION
            || !is_lower_hex_digest(&self.workspace_sha256)
            || self.affected_documents == 0
            || self.affected_documents > MAX_SEMANTIC_EVIDENCE_DOCUMENTS
            || self.providers.len() > MAX_SEMANTIC_EVIDENCE_PROVIDERS
            || provider_names.len() != self.providers.len()
            || self.definite_errors.len() > MAX_SEMANTIC_EVIDENCE_CLASSES
            || self.unknown_reasons.len() > MAX_SEMANTIC_EVIDENCE_CLASSES
            || self.providers.iter().any(|provider| {
                provider.provider.trim().is_empty()
                    || provider.provider.len() > MAX_SEMANTIC_PROVIDER_NAME_BYTES
                    || provider.provider_version.trim().is_empty()
                    || provider.provider_version.len() > MAX_SEMANTIC_PROVIDER_NAME_BYTES
                    || !is_lower_hex_digest(&provider.world_sha256)
                    || !is_lower_hex_digest(&provider.configuration_sha256)
                    || !is_lower_hex_digest(&provider.dependency_sha256)
                    || provider.document_count == 0
                    || provider.document_count > MAX_SEMANTIC_EVIDENCE_DOCUMENTS
                    || provider.definite_errors.len() > MAX_SEMANTIC_EVIDENCE_CLASSES
                    || provider.unknown_reasons.len() > MAX_SEMANTIC_EVIDENCE_CLASSES
            })
        {
            return Err(CollarError::Analysis(
                "semantic gate receipt is structurally invalid".to_string(),
            ));
        }
        if self.closure == ClosureVerdict::Allow
            && (self.viability != Viability::Valid
                || !self.definite_errors.is_empty()
                || !self.unknown_reasons.is_empty()
                || self.providers.is_empty()
                || self.providers.iter().any(|provider| {
                    !provider.authoritative
                        || provider.baseline != BaselineCompleteness::Complete
                        || provider.introduced_diagnostics != 0
                        || !provider.definite_errors.is_empty()
                        || !provider.unknown_reasons.is_empty()
                }))
        {
            return Err(CollarError::Analysis(
                "semantic allow receipt is not authoritative".to_string(),
            ));
        }
        if self.closure != ClosureVerdict::Allow && self.viability == Viability::Valid {
            return Err(CollarError::Analysis(
                "semantic non-allow receipt cannot claim valid viability".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticEnforcement {
    Advisory,
    Required,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineCompleteness {
    Complete,
    Incomplete,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticFileBinding {
    pub path: LogicalPath,
    pub sha256: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticWorldSnapshot {
    pub contract_version: u32,
    pub id: SemanticWorldId,
    pub language: String,
    /// Controller-captured identity of the complete workspace view analyzed by the provider.
    /// This may intentionally be broader than `files`, which lists only admitted overlay files.
    pub workspace_sha256: String,
    pub capabilities: BTreeSet<AnalyzerCapability>,
    pub files: Vec<SemanticFileBinding>,
    pub baseline: BaselineCompleteness,
}

#[derive(Serialize)]
struct SemanticWorldMaterial<'a> {
    contract_version: u32,
    provider: &'a str,
    provider_version: &'a str,
    language: &'a str,
    workspace_sha256: &'a str,
    configuration_sha256: &'a str,
    dependency_sha256: &'a str,
    capabilities: &'a BTreeSet<AnalyzerCapability>,
    files: &'a [SemanticFileBinding],
    baseline: BaselineCompleteness,
}

impl SemanticWorldSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: impl Into<String>,
        provider_version: impl Into<String>,
        language: impl Into<String>,
        workspace_sha256: impl Into<String>,
        configuration_sha256: impl Into<String>,
        dependency_sha256: impl Into<String>,
        capabilities: BTreeSet<AnalyzerCapability>,
        mut files: Vec<SemanticFileBinding>,
        baseline: BaselineCompleteness,
    ) -> CollarResult<Self> {
        let provider = provider.into();
        let provider_version = provider_version.into();
        let language = language.into();
        let workspace_sha256 = workspace_sha256.into();
        let configuration_sha256 = configuration_sha256.into();
        let dependency_sha256 = dependency_sha256.into();
        for (label, value) in [
            ("provider", provider.as_str()),
            ("provider version", provider_version.as_str()),
            ("language", language.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CollarError::Analysis(format!(
                    "semantic world {label} must be non-empty"
                )));
            }
        }
        for (label, value) in [
            ("workspace", workspace_sha256.as_str()),
            ("configuration", configuration_sha256.as_str()),
            ("dependency", dependency_sha256.as_str()),
        ] {
            if !is_lower_hex_digest(value) {
                return Err(CollarError::Analysis(format!(
                    "semantic world {label} identity must be a 64-character lowercase SHA-256"
                )));
            }
        }
        if capabilities.is_empty() {
            return Err(CollarError::Analysis(
                "semantic world requires at least one analyzer capability".to_string(),
            ));
        }
        files.sort_by(|left, right| left.path.cmp(&right.path));
        if files.windows(2).any(|pair| pair[0].path == pair[1].path) {
            return Err(CollarError::Analysis(
                "semantic world repeats a file binding".to_string(),
            ));
        }
        let material = SemanticWorldMaterial {
            contract_version: SEMANTIC_WORLD_CONTRACT_VERSION,
            provider: &provider,
            provider_version: &provider_version,
            language: &language,
            workspace_sha256: &workspace_sha256,
            configuration_sha256: &configuration_sha256,
            dependency_sha256: &dependency_sha256,
            capabilities: &capabilities,
            files: &files,
            baseline,
        };
        let bytes = serde_json::to_vec(&material).map_err(|error| {
            CollarError::Analysis(format!("failed to bind semantic world: {error}"))
        })?;
        let id = SemanticWorldId {
            provider,
            provider_version,
            world_sha256: format!("{:x}", Sha256::digest(bytes)),
            configuration_sha256,
            dependency_sha256,
        };
        Ok(Self {
            contract_version: SEMANTIC_WORLD_CONTRACT_VERSION,
            id,
            language,
            workspace_sha256,
            capabilities,
            files,
            baseline,
        })
    }

    pub fn validate(&self) -> CollarResult<()> {
        let rebuilt = Self::new(
            self.id.provider.clone(),
            self.id.provider_version.clone(),
            self.language.clone(),
            self.workspace_sha256.clone(),
            self.id.configuration_sha256.clone(),
            self.id.dependency_sha256.clone(),
            self.capabilities.clone(),
            self.files.clone(),
            self.baseline,
        )?;
        if self.contract_version != SEMANTIC_WORLD_CONTRACT_VERSION || rebuilt != *self {
            return Err(CollarError::Analysis(
                "semantic world snapshot is invalid".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DiagnosticIdentity {
    pub path: LogicalPath,
    pub start_line: u32,
    pub start_character: u32,
    pub end_line: u32,
    pub end_character: u32,
    pub class: DefiniteErrorClass,
    /// Hash of provider/source/code, never the source line or diagnostic message.
    pub provenance_sha256: String,
}

impl DiagnosticIdentity {
    pub fn validate(&self) -> CollarResult<()> {
        if !is_lower_hex_digest(&self.provenance_sha256) {
            return Err(CollarError::Analysis(
                "diagnostic provenance must be a lowercase SHA-256".to_string(),
            ));
        }
        if (self.end_line, self.end_character) < (self.start_line, self.start_character) {
            return Err(CollarError::Analysis(
                "diagnostic range ends before it starts".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticDiagnosticSnapshot {
    pub world: SemanticWorldId,
    pub document_versions: BTreeMap<LogicalPath, u64>,
    /// Exact bytes admitted to the provider for every document version in this snapshot.
    pub document_sha256: BTreeMap<LogicalPath, Digest>,
    pub completeness: BaselineCompleteness,
    pub diagnostics: BTreeSet<DiagnosticIdentity>,
    pub unknown_reasons: BTreeSet<UnknownReason>,
}

impl SemanticDiagnosticSnapshot {
    pub fn validate(&self) -> CollarResult<()> {
        if self
            .document_versions
            .keys()
            .ne(self.document_sha256.keys())
        {
            return Err(CollarError::Analysis(
                "semantic document versions and content bindings disagree".to_string(),
            ));
        }
        for diagnostic in &self.diagnostics {
            diagnostic.validate()?;
            if !self.document_versions.contains_key(&diagnostic.path) {
                return Err(CollarError::Analysis(format!(
                    "diagnostic names unversioned document {:?}",
                    diagnostic.path.as_str()
                )));
            }
        }
        if self.completeness == BaselineCompleteness::Complete && !self.unknown_reasons.is_empty() {
            return Err(CollarError::Analysis(
                "complete diagnostic snapshot cannot contain unknown reasons".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticDelta {
    pub introduced: BTreeSet<DiagnosticIdentity>,
    pub resolved: BTreeSet<DiagnosticIdentity>,
    pub unchanged: BTreeSet<DiagnosticIdentity>,
    pub authoritative: bool,
    pub unknown_reasons: BTreeSet<UnknownReason>,
}

pub fn diagnostic_delta(
    baseline: &SemanticDiagnosticSnapshot,
    candidate: &SemanticDiagnosticSnapshot,
) -> CollarResult<DiagnosticDelta> {
    baseline.validate()?;
    candidate.validate()?;
    if baseline.world != candidate.world {
        return Err(CollarError::Analysis(
            "semantic diagnostic snapshots belong to different worlds".to_string(),
        ));
    }
    let mut unknown_reasons = baseline.unknown_reasons.clone();
    unknown_reasons.extend(candidate.unknown_reasons.iter().copied());
    let authoritative = baseline.completeness == BaselineCompleteness::Complete
        && candidate.completeness == BaselineCompleteness::Complete
        && unknown_reasons.is_empty();
    Ok(DiagnosticDelta {
        introduced: candidate
            .diagnostics
            .difference(&baseline.diagnostics)
            .cloned()
            .collect(),
        resolved: baseline
            .diagnostics
            .difference(&candidate.diagnostics)
            .cloned()
            .collect(),
        unchanged: candidate
            .diagnostics
            .intersection(&baseline.diagnostics)
            .cloned()
            .collect(),
        authoritative,
        unknown_reasons,
    })
}

#[cfg(test)]
fn workspace_identity(files: &[SemanticFileBinding]) -> String {
    let mut digest = Sha256::new();
    for file in files {
        digest.update((file.path.as_str().len() as u64).to_le_bytes());
        digest.update(file.path.as_str().as_bytes());
        digest.update(file.sha256.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> SemanticWorldSnapshot {
        let file = SemanticFileBinding {
            path: LogicalPath::parse("src/lib.rs").unwrap(),
            sha256: Digest::of(b"fn main() {}\n"),
        };
        let workspace = workspace_identity(std::slice::from_ref(&file));
        SemanticWorldSnapshot::new(
            "rust-analyzer",
            "1.0.0",
            "rust",
            workspace,
            "a".repeat(64),
            "b".repeat(64),
            BTreeSet::from([
                AnalyzerCapability::SymbolResolution,
                AnalyzerCapability::TypeChecking,
            ]),
            vec![file],
            BaselineCompleteness::Complete,
        )
        .unwrap()
    }

    fn diagnostic(world: &SemanticWorldSnapshot) -> SemanticDiagnosticSnapshot {
        SemanticDiagnosticSnapshot {
            world: world.id.clone(),
            document_versions: BTreeMap::from([(LogicalPath::parse("src/lib.rs").unwrap(), 2)]),
            document_sha256: BTreeMap::from([(
                LogicalPath::parse("src/lib.rs").unwrap(),
                Digest::of(b"fn main() { missing(); }\n"),
            )]),
            completeness: BaselineCompleteness::Complete,
            diagnostics: BTreeSet::from([DiagnosticIdentity {
                path: LogicalPath::parse("src/lib.rs").unwrap(),
                start_line: 1,
                start_character: 0,
                end_line: 1,
                end_character: 4,
                class: DefiniteErrorClass::TypeMismatch,
                provenance_sha256: "c".repeat(64),
            }]),
            unknown_reasons: BTreeSet::new(),
        }
    }

    #[test]
    fn semantic_world_identity_changes_with_configuration_and_files() {
        let first = world();
        let mut second = first.clone();
        second.id.configuration_sha256 = "d".repeat(64);
        let rebuilt = SemanticWorldSnapshot::new(
            second.id.provider.clone(),
            second.id.provider_version.clone(),
            second.language.clone(),
            workspace_identity(&second.files),
            second.id.configuration_sha256.clone(),
            second.id.dependency_sha256.clone(),
            second.capabilities.clone(),
            second.files.clone(),
            second.baseline,
        )
        .unwrap();
        assert_ne!(first.id.world_sha256, rebuilt.id.world_sha256);
        let mut tampered = first;
        tampered.workspace_sha256 = "e".repeat(64);
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn diagnostic_debt_distinguishes_new_errors_from_repairs() {
        let world = world();
        let clean = SemanticDiagnosticSnapshot {
            world: world.id.clone(),
            document_versions: BTreeMap::from([(LogicalPath::parse("src/lib.rs").unwrap(), 1)]),
            document_sha256: BTreeMap::from([(
                LogicalPath::parse("src/lib.rs").unwrap(),
                Digest::of(b"fn main() {}\n"),
            )]),
            completeness: BaselineCompleteness::Complete,
            diagnostics: BTreeSet::new(),
            unknown_reasons: BTreeSet::new(),
        };
        let broken = diagnostic(&world);
        let introduced = diagnostic_delta(&clean, &broken).unwrap();
        assert!(introduced.authoritative);
        assert_eq!(introduced.introduced.len(), 1);

        let repaired = diagnostic_delta(&broken, &clean).unwrap();
        assert_eq!(repaired.resolved.len(), 1);
        assert!(repaired.introduced.is_empty());
    }

    #[test]
    fn incomplete_baselines_can_never_authorize_a_clean_gate() {
        let world = world();
        let mut baseline = diagnostic(&world);
        baseline.completeness = BaselineCompleteness::Incomplete;
        baseline
            .unknown_reasons
            .insert(UnknownReason::IncompleteBaseline);
        let candidate = baseline.clone();
        assert!(
            !diagnostic_delta(&baseline, &candidate)
                .unwrap()
                .authoritative
        );
    }

    #[test]
    fn semantic_receipts_require_content_free_authoritative_allow_evidence() {
        let world = world();
        let receipt = SemanticGateReceipt {
            contract_version: SEMANTIC_EVIDENCE_CONTRACT_VERSION,
            stage: SemanticEvidenceStage::FinalExecutor,
            scope: SemanticEvidenceScope::Document,
            workspace_sha256: world.workspace_sha256.clone(),
            affected_documents: 1,
            providers: vec![SemanticProviderEvidence {
                provider: world.id.provider.clone(),
                provider_version: world.id.provider_version.clone(),
                world_sha256: world.id.world_sha256.clone(),
                configuration_sha256: world.id.configuration_sha256.clone(),
                dependency_sha256: world.id.dependency_sha256.clone(),
                baseline: BaselineCompleteness::Complete,
                document_count: 1,
                introduced_diagnostics: 0,
                resolved_diagnostics: 0,
                unchanged_diagnostics: 0,
                authoritative: true,
                definite_errors: Vec::new(),
                unknown_reasons: Vec::new(),
            }],
            viability: Viability::Valid,
            closure: ClosureVerdict::Allow,
            definite_errors: Vec::new(),
            unknown_reasons: Vec::new(),
            wall_millis: 12,
            budget_millis: 8_000,
        };
        receipt.validate().unwrap();

        let mut unpinned = receipt;
        unpinned.providers[0].authoritative = false;
        assert!(unpinned.validate().is_err());

        let mut incomplete = unpinned.clone();
        incomplete.providers[0].authoritative = true;
        incomplete.providers[0].baseline = BaselineCompleteness::Incomplete;
        assert!(incomplete.validate().is_err());

        let mut inconsistent = unpinned;
        inconsistent.providers[0].authoritative = true;
        inconsistent.viability = Viability::Unknown;
        assert!(inconsistent.validate().is_err());
    }
}
