use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::mutation::LogicalPath;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Digest([u8; 32]);

impl Digest {
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDigest {
    pub path: LogicalPath,
    pub sha256: Digest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisStatus {
    Valid,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageResult {
    pub profile: String,
    pub status: AnalysisStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalState {
    Complete,
    Incomplete,
    PayloadLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollarReceipt {
    pub contract_version: u32,
    pub dialect_version: u32,
    pub manifest_sha256: Digest,
    pub transcript_sha256: Digest,
    pub base_files: Vec<FileDigest>,
    pub result_files: Vec<FileDigest>,
    pub patch_sha256: Option<Digest>,
    pub language_results: Vec<LanguageResult>,
    pub terminal_state: TerminalState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_display_is_stable_lower_hex() {
        assert_eq!(
            Digest::of(b"control-collar").to_string(),
            "15eb2bb5113c3dddfc72be43d48213f097a30b410e748901743ca580f4881286"
        );
    }
}
