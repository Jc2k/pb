use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::WorkflowStage;
use crate::workspace::ContentSnapshot;

pub const STAGE_EVIDENCE_SCHEMA_VERSION: u32 = 1;
const MAX_STAGE_EVIDENCE_ENTRIES: usize = 24;
const MAX_STAGE_EVIDENCE_BYTES: usize = 64 * 1024;
pub const MAX_STAGE_EVIDENCE_ENTRY_BYTES: usize = 16 * 1024;

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
}

impl Default for StageEvidenceBundle {
    fn default() -> Self {
        Self {
            version: STAGE_EVIDENCE_SCHEMA_VERSION,
            entries: Vec::new(),
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
        self.validate()
    }

    /// Return only complete entries whose repository path still has the exact captured bytes.
    pub fn current(&self, repo_root: &Path) -> Result<Self> {
        self.validate()?;
        if self.entries.is_empty() {
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
        let current = Self {
            version: STAGE_EVIDENCE_SCHEMA_VERSION,
            entries,
        };
        current.validate()?;
        Ok(current)
    }

    pub fn read_paths(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|entry| entry.path.as_str())
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
}
