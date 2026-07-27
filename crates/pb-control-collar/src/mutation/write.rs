use serde::{Deserialize, Serialize};

use crate::{
    CollarError, CollarResult,
    analysis::{SyntaxReport, validate_supported_syntax},
    mutation::{LogicalPath, WorkspaceSnapshot},
    receipt::Digest,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationKind {
    Create,
    Modify,
    Delete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileStreamMode {
    Create,
    Replace,
}

#[derive(Clone, Debug)]
pub struct VirtualFileStream {
    snapshot: WorkspaceSnapshot,
    path: LogicalPath,
    mode: FileStreamMode,
    bytes: Vec<u8>,
    max_bytes: usize,
}

impl VirtualFileStream {
    pub fn new(
        snapshot: WorkspaceSnapshot,
        path: LogicalPath,
        mode: FileStreamMode,
        max_bytes: usize,
    ) -> CollarResult<Self> {
        if max_bytes == 0 {
            return Err(CollarError::Mutation(
                "virtual file stream limit must be non-zero".to_string(),
            ));
        }
        match mode {
            FileStreamMode::Create if snapshot.contains(&path) => {
                return Err(CollarError::Mutation(format!(
                    "create target {:?} already exists in the snapshot",
                    path.as_str()
                )));
            }
            FileStreamMode::Replace if !snapshot.contains(&path) => {
                return Err(CollarError::Mutation(format!(
                    "replace target {:?} is missing from the snapshot",
                    path.as_str()
                )));
            }
            FileStreamMode::Create | FileStreamMode::Replace => {}
        }
        Ok(Self {
            snapshot,
            path,
            mode,
            bytes: Vec::new(),
            max_bytes,
        })
    }

    pub fn push(&mut self, bytes: &[u8]) -> CollarResult<()> {
        let next_len = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| CollarError::Mutation("file stream length overflow".to_string()))?;
        if next_len > self.max_bytes {
            return Err(CollarError::Mutation(format!(
                "file stream exceeds the {}-byte limit",
                self.max_bytes
            )));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    pub fn finish(self) -> CollarResult<PreparedFileMutation> {
        match self.mode {
            FileStreamMode::Create => prepare_create(&self.snapshot, self.path, self.bytes),
            FileStreamMode::Replace => prepare_replace(&self.snapshot, self.path, self.bytes),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedFileMutation {
    path: LogicalPath,
    kind: MutationKind,
    base_sha256: Option<Digest>,
    result_bytes: Option<Vec<u8>>,
    result_sha256: Option<Digest>,
    syntax: SyntaxReport,
}

impl PreparedFileMutation {
    pub(crate) fn new(
        path: LogicalPath,
        kind: MutationKind,
        base_bytes: Option<&[u8]>,
        result_bytes: Option<Vec<u8>>,
    ) -> CollarResult<Self> {
        match (kind, base_bytes, result_bytes.as_ref()) {
            (MutationKind::Create, None, Some(_))
            | (MutationKind::Modify, Some(_), Some(_))
            | (MutationKind::Delete, Some(_), None) => {}
            _ => {
                return Err(CollarError::Mutation(format!(
                    "mutation kind {kind:?} has inconsistent base/result state for {:?}",
                    path.as_str()
                )));
            }
        }
        if let (Some(base), Some(result)) = (base_bytes, result_bytes.as_deref())
            && base == result
        {
            return Err(CollarError::Mutation(format!(
                "mutation makes no content change to {:?}",
                path.as_str()
            )));
        }
        let syntax = match result_bytes.as_deref() {
            Some(result) => validate_supported_syntax(&path, result)?,
            None => SyntaxReport::Unsupported,
        };
        let base_sha256 = base_bytes.map(Digest::of);
        let result_sha256 = result_bytes.as_deref().map(Digest::of);
        Ok(Self {
            path,
            kind,
            base_sha256,
            result_bytes,
            result_sha256,
            syntax,
        })
    }

    pub fn path(&self) -> &LogicalPath {
        &self.path
    }

    pub fn kind(&self) -> MutationKind {
        self.kind
    }

    pub fn base_sha256(&self) -> Option<Digest> {
        self.base_sha256
    }

    pub fn result_bytes(&self) -> Option<&[u8]> {
        self.result_bytes.as_deref()
    }

    pub fn result_sha256(&self) -> Option<Digest> {
        self.result_sha256
    }

    pub fn syntax(&self) -> &SyntaxReport {
        &self.syntax
    }

    pub fn verify_live_base(&self, live: Option<&[u8]>) -> CollarResult<()> {
        let actual = live.map(Digest::of);
        if actual != self.base_sha256 {
            return Err(CollarError::Mutation(format!(
                "live base for {:?} drifted from the prepared snapshot",
                self.path.as_str()
            )));
        }
        Ok(())
    }

    pub fn revalidate_result(&self) -> CollarResult<()> {
        if let Some(result) = self.result_bytes() {
            validate_supported_syntax(&self.path, result)?;
        }
        Ok(())
    }
}

pub fn prepare_create(
    snapshot: &WorkspaceSnapshot,
    path: LogicalPath,
    content: Vec<u8>,
) -> CollarResult<PreparedFileMutation> {
    if snapshot.contains(&path) {
        return Err(CollarError::Mutation(format!(
            "create target {:?} already exists in the snapshot",
            path.as_str()
        )));
    }
    PreparedFileMutation::new(path, MutationKind::Create, None, Some(content))
}

pub fn prepare_replace(
    snapshot: &WorkspaceSnapshot,
    path: LogicalPath,
    content: Vec<u8>,
) -> CollarResult<PreparedFileMutation> {
    let base = snapshot.get(&path).ok_or_else(|| {
        CollarError::Mutation(format!(
            "replace target {:?} is missing from the snapshot",
            path.as_str()
        ))
    })?;
    PreparedFileMutation::new(path, MutationKind::Modify, Some(&base.bytes), Some(content))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutation::SnapshotEntry;

    #[test]
    fn creates_require_missing_paths_and_valid_supported_syntax() {
        let snapshot = WorkspaceSnapshot::default();
        let valid = prepare_create(
            &snapshot,
            LogicalPath::parse("src/lib.rs").unwrap(),
            b"pub fn value() -> i32 { 1 }\n".to_vec(),
        )
        .unwrap();
        assert_eq!(valid.kind(), MutationKind::Create);
        assert!(valid.verify_live_base(None).is_ok());

        let invalid = prepare_create(
            &snapshot,
            LogicalPath::parse("src/broken.rs").unwrap(),
            b"pub fn broken( {\n".to_vec(),
        );
        assert!(invalid.is_err());
    }

    #[test]
    fn replacements_bind_and_recheck_the_exact_base() {
        let path = LogicalPath::parse("main.py").unwrap();
        let base = b"value = 1\n".to_vec();
        let snapshot =
            WorkspaceSnapshot::new(vec![SnapshotEntry::new(path.clone(), base.clone())]).unwrap();
        let prepared = prepare_replace(&snapshot, path, b"value = 2\n".to_vec()).unwrap();

        assert!(prepared.verify_live_base(Some(&base)).is_ok());
        assert!(prepared.verify_live_base(Some(b"value = 3\n")).is_err());
        assert!(prepared.revalidate_result().is_ok());
    }

    #[test]
    fn virtual_file_stream_is_independent_of_wire_chunk_boundaries() {
        let content = b"fn main() {\n    println!(\"ok\");\n}\n";
        let expected = prepare_create(
            &WorkspaceSnapshot::default(),
            LogicalPath::parse("main.rs").unwrap(),
            content.to_vec(),
        )
        .unwrap();
        for width in 1..=content.len() {
            let mut stream = VirtualFileStream::new(
                WorkspaceSnapshot::default(),
                LogicalPath::parse("main.rs").unwrap(),
                FileStreamMode::Create,
                content.len(),
            )
            .unwrap();
            for chunk in content.chunks(width) {
                stream.push(chunk).unwrap();
            }
            assert_eq!(stream.finish().unwrap(), expected, "chunk width {width}");
        }
    }
}
