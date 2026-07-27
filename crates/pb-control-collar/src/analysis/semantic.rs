use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::{CollarError, CollarResult, mutation::LogicalPath, receipt::Digest};

use super::{AnalyzerCapability, DefiniteErrorClass, SemanticWorldId, UnknownReason};

pub const SEMANTIC_WORLD_CONTRACT_VERSION: u32 = 1;

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
}
