use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{CollarError, CollarResult, receipt::Digest};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LogicalPath(String);

impl LogicalPath {
    pub fn parse(path: impl Into<String>) -> CollarResult<Self> {
        let path = path.into();
        if path.is_empty()
            || path.starts_with('/')
            || path.ends_with('/')
            || path.contains('\0')
            || path.contains('\\')
            || path
                .split('/')
                .any(|component| component.is_empty() || component == "." || component == "..")
        {
            return Err(CollarError::Mutation(format!(
                "path {path:?} is not a canonical repository-relative path"
            )));
        }
        Ok(Self(path))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotEntry {
    pub path: LogicalPath,
    pub bytes: Vec<u8>,
    pub sha256: Digest,
}

impl SnapshotEntry {
    pub fn new(path: LogicalPath, bytes: Vec<u8>) -> Self {
        let sha256 = Digest::of(&bytes);
        Self {
            path,
            bytes,
            sha256,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceSnapshot {
    entries: BTreeMap<LogicalPath, SnapshotEntry>,
}

impl WorkspaceSnapshot {
    pub fn new(entries: Vec<SnapshotEntry>) -> CollarResult<Self> {
        let mut indexed = BTreeMap::new();
        for entry in entries {
            if entry.sha256 != Digest::of(&entry.bytes) {
                return Err(CollarError::Mutation(format!(
                    "snapshot digest does not match bytes for {:?}",
                    entry.path.as_str()
                )));
            }
            let path = entry.path.clone();
            if indexed.insert(path.clone(), entry).is_some() {
                return Err(CollarError::Mutation(format!(
                    "snapshot repeats path {:?}",
                    path.as_str()
                )));
            }
        }
        Ok(Self { entries: indexed })
    }

    pub fn get(&self, path: &LogicalPath) -> Option<&SnapshotEntry> {
        self.entries.get(path)
    }

    pub fn contains(&self, path: &LogicalPath) -> bool {
        self.entries.contains_key(path)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn total_bytes(&self) -> usize {
        self.entries.values().map(|entry| entry.bytes.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_paths_reject_ambiguous_or_escaping_forms() {
        for path in ["", "/tmp/x", "a/../b", "a//b", "./a", "a\\b", "a/"] {
            assert!(LogicalPath::parse(path).is_err(), "accepted {path:?}");
        }
        assert_eq!(
            LogicalPath::parse("src/lib.rs").unwrap().as_str(),
            "src/lib.rs"
        );
    }

    #[test]
    fn snapshots_bind_exact_bytes() {
        let path = LogicalPath::parse("src/lib.rs").unwrap();
        let entry = SnapshotEntry::new(path.clone(), b"fn main() {}\n".to_vec());
        let snapshot = WorkspaceSnapshot::new(vec![entry.clone()]).unwrap();

        assert_eq!(snapshot.get(&path), Some(&entry));
        assert_eq!(snapshot.total_bytes(), 13);
    }
}
