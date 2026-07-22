use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{PlannedChange, WorkflowStage};
use crate::workspace::ContentSnapshot;

pub const STAGE_EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const CONTROLLER_OBSERVATION_SCHEMA_VERSION: u32 = 1;
pub const CONTROLLER_MUTATION_SCHEMA_VERSION: u32 = 1;
const MAX_STAGE_EVIDENCE_ENTRIES: usize = 24;
const MAX_STAGE_EVIDENCE_BYTES: usize = 64 * 1024;
const MAX_CONTROLLER_OBSERVATIONS: usize = 64;
pub const MAX_STAGE_EVIDENCE_ENTRY_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ObservationRendering {
    #[default]
    Native,
    ControllerBlock,
    DisclosedToolTranscript,
    CompatibilityToolTranscript,
}

impl ObservationRendering {
    pub const fn is_controller(self) -> bool {
        !matches!(self, Self::Native)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::ControllerBlock => "controller_block",
            Self::DisclosedToolTranscript => "disclosed_tool_transcript",
            Self::CompatibilityToolTranscript => "compatibility_tool_transcript",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControllerObservationOrigin {
    Controller,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControllerObservationOperation {
    ReadFile,
    InspectChange,
}

impl ControllerObservationOperation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadFile => "read_file",
            Self::InspectChange => "inspect_change",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationCoverage {
    Full,
    Ranges,
    MetadataOnly,
    None,
}

impl ObservationCoverage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Ranges => "ranges",
            Self::MetadataOnly => "metadata_only",
            Self::None => "none",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControllerObservationAuthority {
    PromptContext,
    ReadBeforeWrite,
    ReviewCoverage,
}

impl ControllerObservationAuthority {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PromptContext => "prompt_context",
            Self::ReadBeforeWrite => "read_before_write",
            Self::ReviewCoverage => "review_coverage",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationRange {
    pub start_byte: usize,
    pub end_byte: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ControllerObservationReceipt {
    pub version: u32,
    pub action_id: String,
    pub actual_origin: ControllerObservationOrigin,
    pub prompt_representation: ObservationRendering,
    pub stage: WorkflowStage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_unit_id: Option<String>,
    pub operation: ControllerObservationOperation,
    pub path: String,
    pub workspace_fingerprint: String,
    pub path_fingerprint: String,
    pub content_sha256: String,
    pub coverage: ObservationCoverage,
    pub observed_bytes: usize,
    pub prompt_bytes: usize,
    #[serde(default)]
    pub included_ranges: Vec<ObservationRange>,
    pub included_in_prompt: bool,
    #[serde(default)]
    pub authority_effects: Vec<ControllerObservationAuthority>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ControllerMutationReceipt {
    pub version: u32,
    pub action_id: String,
    pub actual_origin: ControllerObservationOrigin,
    pub stage: WorkflowStage,
    pub work_unit_id: String,
    pub operation: PlannedChange,
    pub path: String,
    pub before_workspace_fingerprint: String,
    pub before_path_fingerprint: String,
    pub before_content_sha256: String,
    pub after_workspace_fingerprint: String,
    pub tracked: bool,
    pub adopted: bool,
    pub recovery: String,
}

impl ControllerMutationReceipt {
    pub fn validate(&self) -> Result<()> {
        if self.version != CONTROLLER_MUTATION_SCHEMA_VERSION {
            bail!("unsupported controller mutation schema");
        }
        if self.action_id.trim().is_empty()
            || self.work_unit_id.trim().is_empty()
            || self.path.trim().is_empty()
            || self.path.starts_with('/')
            || self.path.contains("..")
        {
            bail!("controller mutation identity or path is invalid");
        }
        if self.operation != PlannedChange::Delete {
            bail!("controller mutation receipt supports deletion only");
        }
        for digest in [
            self.before_workspace_fingerprint.as_str(),
            self.before_path_fingerprint.as_str(),
            self.before_content_sha256.as_str(),
            self.after_workspace_fingerprint.as_str(),
        ] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                bail!("controller mutation fingerprint is not lowercase SHA-256");
            }
        }
        if !self.tracked || self.adopted || self.recovery.trim().is_empty() {
            bail!("controller deletion lacks tracked, unchanged, recoverable authority");
        }
        Ok(())
    }
}

impl ControllerObservationReceipt {
    pub fn validate(&self) -> Result<()> {
        if self.version != CONTROLLER_OBSERVATION_SCHEMA_VERSION {
            bail!(
                "unsupported controller observation schema {}; expected {}",
                self.version,
                CONTROLLER_OBSERVATION_SCHEMA_VERSION
            );
        }
        if self.action_id.trim().is_empty() {
            bail!("controller observation action id is empty");
        }
        if self.path.trim().is_empty() || self.path.starts_with('/') || self.path.contains("..") {
            bail!("controller observation path must be normalized and repository-relative");
        }
        for (label, digest) in [
            ("workspace", self.workspace_fingerprint.as_str()),
            ("path", self.path_fingerprint.as_str()),
            ("content", self.content_sha256.as_str()),
        ] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                bail!("controller observation {label} fingerprint is not lowercase SHA-256");
            }
        }
        if self.prompt_representation == ObservationRendering::Native {
            bail!("a controller observation cannot use native prompt representation");
        }
        if self.included_in_prompt != (self.prompt_bytes > 0) {
            bail!("controller observation prompt inclusion metadata is inconsistent");
        }
        let mut previous_end = 0usize;
        for range in &self.included_ranges {
            if range.start_byte >= range.end_byte
                || range.end_byte > self.observed_bytes
                || range.start_byte < previous_end
            {
                bail!("controller observation ranges are invalid or overlap");
            }
            if range.sha256.len() != 64
                || !range
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                bail!("controller observation range digest is not lowercase SHA-256");
            }
            previous_end = range.end_byte;
        }
        match self.coverage {
            ObservationCoverage::Full
                if (self.observed_bytes == 0 && !self.included_ranges.is_empty())
                    || (self.observed_bytes > 0
                        && (self.included_ranges.len() != 1
                            || self.included_ranges[0].start_byte != 0
                            || self.included_ranges[0].end_byte != self.observed_bytes)) =>
            {
                bail!("full controller observation does not cover every observed byte");
            }
            ObservationCoverage::Ranges if self.included_ranges.is_empty() => {
                bail!("range controller observation has no included range");
            }
            ObservationCoverage::MetadataOnly | ObservationCoverage::None
                if !self.included_ranges.is_empty() =>
            {
                bail!("metadata-only controller observation contains byte ranges");
            }
            _ => {}
        }
        if self.authority_effects.is_empty()
            || !self
                .authority_effects
                .contains(&ControllerObservationAuthority::PromptContext)
        {
            bail!("controller observation lacks prompt-context provenance");
        }
        Ok(())
    }

    pub fn permits_read_before_write(&self) -> bool {
        self.authority_effects
            .contains(&ControllerObservationAuthority::ReadBeforeWrite)
    }
}

/// Harness-owned repository bytes that may be carried into a fresh workflow stage.
///
/// Entries are deliberately limited to complete UTF-8 file reads. A partial read remains valid
/// evidence in the invocation that performed it, but is not promoted into another model stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StageEvidenceEntry {
    pub path: String,
    pub path_fingerprint: String,
    pub content_sha256: String,
    pub workspace_fingerprint: String,
    pub source_stage: WorkflowStage,
    pub source_tool: String,
    pub arguments_sha256: String,
    pub total_lines: usize,
    pub raw_bytes: usize,
    pub content: String,
    pub observed_order: u64,
}

impl StageEvidenceEntry {
    #[allow(clippy::too_many_arguments)]
    pub fn complete_file(
        path: String,
        path_fingerprint: String,
        workspace_fingerprint: String,
        source_stage: WorkflowStage,
        source_tool: String,
        arguments_sha256: String,
        content: String,
        observed_order: u64,
    ) -> Result<Self> {
        if path.trim().is_empty() || path.starts_with('/') || path.contains("..") {
            bail!("stage evidence path must be normalized and repository-relative");
        }
        if path_fingerprint.trim().is_empty() || workspace_fingerprint.trim().is_empty() {
            bail!("stage evidence requires path and workspace fingerprints");
        }
        if content.len() > MAX_STAGE_EVIDENCE_ENTRY_BYTES {
            bail!(
                "stage evidence entry for {path} exceeds the {} byte bound",
                MAX_STAGE_EVIDENCE_ENTRY_BYTES
            );
        }
        let content_sha256 = format!("{:x}", Sha256::digest(content.as_bytes()));
        let total_lines = content.lines().count();
        let raw_bytes = content.len();
        let entry = Self {
            path,
            path_fingerprint,
            content_sha256,
            workspace_fingerprint,
            source_stage,
            source_tool,
            arguments_sha256,
            total_lines,
            raw_bytes,
            content,
            observed_order,
        };
        entry.validate()?;
        Ok(entry)
    }

    pub fn validate(&self) -> Result<()> {
        if self.path.trim().is_empty() || self.path.starts_with('/') || self.path.contains("..") {
            bail!("stage evidence path must be normalized and repository-relative");
        }
        if self.content.len() > MAX_STAGE_EVIDENCE_ENTRY_BYTES {
            bail!("stage evidence entry exceeds its byte bound");
        }
        let digest = format!("{:x}", Sha256::digest(self.content.as_bytes()));
        if digest != self.content_sha256 {
            bail!("stage evidence content digest mismatch for {}", self.path);
        }
        if self.raw_bytes != self.content.len() || self.total_lines != self.content.lines().count()
        {
            bail!("stage evidence size metadata mismatch for {}", self.path);
        }
        if self.path_fingerprint.trim().is_empty()
            || self.workspace_fingerprint.trim().is_empty()
            || self.source_tool.trim().is_empty()
            || self.arguments_sha256.trim().is_empty()
        {
            bail!("stage evidence provenance is incomplete for {}", self.path);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StageEvidenceBundle {
    pub version: u32,
    #[serde(default)]
    pub entries: Vec<StageEvidenceEntry>,
    #[serde(default)]
    pub controller_observations: Vec<ControllerObservationReceipt>,
}

impl Default for StageEvidenceBundle {
    fn default() -> Self {
        Self {
            version: STAGE_EVIDENCE_SCHEMA_VERSION,
            entries: Vec::new(),
            controller_observations: Vec::new(),
        }
    }
}

impl StageEvidenceBundle {
    pub fn validate(&self) -> Result<()> {
        if self.version != STAGE_EVIDENCE_SCHEMA_VERSION {
            bail!(
                "unsupported stage evidence schema {}; expected {}",
                self.version,
                STAGE_EVIDENCE_SCHEMA_VERSION
            );
        }
        if self.entries.len() > MAX_STAGE_EVIDENCE_ENTRIES {
            bail!("stage evidence entry count exceeds its bound");
        }
        if self.controller_observations.len() > MAX_CONTROLLER_OBSERVATIONS {
            bail!("controller observation count exceeds its bound");
        }
        let mut total_bytes = 0usize;
        let mut paths = std::collections::BTreeSet::new();
        for entry in &self.entries {
            entry.validate()?;
            if !paths.insert(entry.path.as_str()) {
                bail!("stage evidence repeats path {}", entry.path);
            }
            total_bytes = total_bytes.saturating_add(entry.content.len());
        }
        if total_bytes > MAX_STAGE_EVIDENCE_BYTES {
            bail!("stage evidence content exceeds its total byte bound");
        }
        let mut action_ids = std::collections::BTreeSet::new();
        for observation in &self.controller_observations {
            observation.validate()?;
            if !action_ids.insert(observation.action_id.as_str()) {
                bail!(
                    "stage evidence repeats controller observation {}",
                    observation.action_id
                );
            }
        }
        Ok(())
    }

    pub fn merge(&mut self, observed: Self) -> Result<()> {
        self.validate()?;
        observed.validate()?;
        let mut entries = self
            .entries
            .drain(..)
            .map(|entry| (entry.path.clone(), entry))
            .collect::<BTreeMap<_, _>>();
        for entry in observed.entries {
            entries.insert(entry.path.clone(), entry);
        }
        let mut entries = entries.into_values().collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.observed_order);
        while entries.len() > MAX_STAGE_EVIDENCE_ENTRIES
            || entries
                .iter()
                .map(|entry| entry.content.len())
                .sum::<usize>()
                > MAX_STAGE_EVIDENCE_BYTES
        {
            entries.remove(0);
        }
        self.entries = entries;
        let mut observations = self
            .controller_observations
            .drain(..)
            .map(|receipt| (receipt.action_id.clone(), receipt))
            .collect::<BTreeMap<_, _>>();
        for receipt in observed.controller_observations {
            observations.insert(receipt.action_id.clone(), receipt);
        }
        self.controller_observations = observations.into_values().collect();
        if self.controller_observations.len() > MAX_CONTROLLER_OBSERVATIONS {
            let remove = self
                .controller_observations
                .len()
                .saturating_sub(MAX_CONTROLLER_OBSERVATIONS);
            self.controller_observations.drain(..remove);
        }
        self.validate()
    }

    /// Return only complete entries whose repository path still has the exact captured bytes.
    pub fn current(&self, repo_root: &Path) -> Result<Self> {
        self.validate()?;
        if self.entries.is_empty() && self.controller_observations.is_empty() {
            return Ok(Self::default());
        }
        let snapshot = ContentSnapshot::capture(repo_root).with_context(|| {
            format!(
                "failed to validate stage evidence in {}",
                repo_root.display()
            )
        })?;
        let entries = self
            .entries
            .iter()
            .filter(|entry| {
                snapshot
                    .paths
                    .get(&entry.path)
                    .is_some_and(|path| path.fingerprint == entry.path_fingerprint)
            })
            .cloned()
            .collect();
        let controller_observations = self
            .controller_observations
            .iter()
            .filter(|receipt| {
                snapshot
                    .paths
                    .get(&receipt.path)
                    .is_some_and(|path| path.fingerprint == receipt.path_fingerprint)
            })
            .cloned()
            .collect();
        let current = Self {
            version: STAGE_EVIDENCE_SCHEMA_VERSION,
            entries,
            controller_observations,
        };
        current.validate()?;
        Ok(current)
    }

    pub fn read_paths(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|entry| entry.path.as_str()).chain(
            self.controller_observations
                .iter()
                .map(|receipt| receipt.path.as_str()),
        )
    }

    pub fn mutation_evidence_paths(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|entry| entry.path.as_str()).chain(
            self.controller_observations
                .iter()
                .filter(|receipt| receipt.permits_read_before_write())
                .map(|receipt| receipt.path.as_str()),
        )
    }

    pub fn prompt_json(&self) -> Result<String> {
        serde_json::to_string(self).context("failed to serialize carried stage evidence")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn repo() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        for args in [
            vec!["init", "-b", "main"],
            vec!["config", "user.email", "test@example.com"],
            vec!["config", "user.name", "Test"],
        ] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(tmp.path())
                    .status()
                    .unwrap()
                    .success()
            );
        }
        std::fs::write(tmp.path().join("small.txt"), "one\ntwo\n").unwrap();
        Command::new("git")
            .args(["add", "small.txt"])
            .current_dir(tmp.path())
            .status()
            .unwrap();
        tmp
    }

    fn entry(root: &Path, order: u64) -> StageEvidenceEntry {
        let snapshot = ContentSnapshot::capture(root).unwrap();
        StageEvidenceEntry::complete_file(
            "small.txt".to_string(),
            snapshot.paths["small.txt"].fingerprint.clone(),
            snapshot.fingerprint,
            WorkflowStage::Planning,
            "read_file".to_string(),
            "args".to_string(),
            "one\ntwo\n".to_string(),
            order,
        )
        .unwrap()
    }

    fn observation(root: &Path) -> ControllerObservationReceipt {
        let snapshot = ContentSnapshot::capture(root).unwrap();
        let content = std::fs::read(root.join("small.txt")).unwrap();
        ControllerObservationReceipt {
            version: CONTROLLER_OBSERVATION_SCHEMA_VERSION,
            action_id: "controller-read-small".to_string(),
            actual_origin: ControllerObservationOrigin::Controller,
            prompt_representation: ObservationRendering::ControllerBlock,
            stage: WorkflowStage::Implementing,
            work_unit_id: Some("step:0".to_string()),
            operation: ControllerObservationOperation::ReadFile,
            path: "small.txt".to_string(),
            workspace_fingerprint: snapshot.fingerprint,
            path_fingerprint: snapshot.paths["small.txt"].fingerprint.clone(),
            content_sha256: format!("{:x}", Sha256::digest(&content)),
            coverage: ObservationCoverage::Full,
            observed_bytes: content.len(),
            prompt_bytes: content.len(),
            included_ranges: vec![ObservationRange {
                start_byte: 0,
                end_byte: content.len(),
                sha256: format!("{:x}", Sha256::digest(&content)),
            }],
            included_in_prompt: true,
            authority_effects: vec![
                ControllerObservationAuthority::PromptContext,
                ControllerObservationAuthority::ReadBeforeWrite,
            ],
            fallback_reason: None,
        }
    }

    #[test]
    fn complete_unchanged_evidence_survives_checkpoint_round_trip() {
        let repo = repo();
        let bundle = StageEvidenceBundle {
            entries: vec![entry(repo.path(), 1)],
            ..StageEvidenceBundle::default()
        };
        let encoded = serde_json::to_string(&bundle).unwrap();
        let restored: StageEvidenceBundle = serde_json::from_str(&encoded).unwrap();
        assert_eq!(restored.current(repo.path()).unwrap(), bundle);
    }

    #[test]
    fn changed_path_is_not_carried_into_the_next_stage() {
        let repo = repo();
        let bundle = StageEvidenceBundle {
            entries: vec![entry(repo.path(), 1)],
            ..StageEvidenceBundle::default()
        };
        std::fs::write(repo.path().join("small.txt"), "changed\n").unwrap();
        assert!(bundle.current(repo.path()).unwrap().entries.is_empty());
    }

    #[test]
    fn tampered_content_is_rejected() {
        let repo = repo();
        let mut bundle = StageEvidenceBundle {
            entries: vec![entry(repo.path(), 1)],
            ..StageEvidenceBundle::default()
        };
        bundle.entries[0].content.push_str("tampered");
        assert!(bundle.validate().is_err());
    }

    #[test]
    fn oversized_or_partial_file_bytes_cannot_claim_complete_carried_evidence() {
        let repo = repo();
        let snapshot = ContentSnapshot::capture(repo.path()).unwrap();
        let oversized = "x".repeat(MAX_STAGE_EVIDENCE_ENTRY_BYTES + 1);
        assert!(
            StageEvidenceEntry::complete_file(
                "small.txt".to_string(),
                snapshot.paths["small.txt"].fingerprint.clone(),
                snapshot.fingerprint,
                WorkflowStage::Planning,
                "read_file".to_string(),
                "partial-args".to_string(),
                oversized,
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn controller_observation_round_trip_preserves_origin_and_current_path_binding() {
        let repo = repo();
        let receipt = observation(repo.path());
        receipt.validate().unwrap();
        let bundle = StageEvidenceBundle {
            controller_observations: vec![receipt.clone()],
            ..StageEvidenceBundle::default()
        };
        let restored: StageEvidenceBundle =
            serde_json::from_str(&serde_json::to_string(&bundle).unwrap()).unwrap();
        assert_eq!(restored.current(repo.path()).unwrap(), bundle);
        assert_eq!(
            restored.controller_observations[0].actual_origin,
            ControllerObservationOrigin::Controller
        );
        assert_eq!(
            restored.mutation_evidence_paths().collect::<Vec<_>>(),
            vec!["small.txt"]
        );

        std::fs::write(repo.path().join("small.txt"), "changed\n").unwrap();
        assert!(
            restored
                .current(repo.path())
                .unwrap()
                .controller_observations
                .is_empty()
        );
    }

    #[test]
    fn controller_observation_rejects_native_or_false_full_coverage() {
        let repo = repo();
        let mut receipt = observation(repo.path());
        receipt.prompt_representation = ObservationRendering::Native;
        assert!(receipt.validate().is_err());

        let mut receipt = observation(repo.path());
        receipt.included_ranges[0].end_byte -= 1;
        assert!(receipt.validate().is_err());
    }
}
