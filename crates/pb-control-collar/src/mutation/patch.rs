use std::collections::BTreeSet;

use crate::{
    CollarError, CollarResult,
    mutation::{LogicalPath, MutationKind, PreparedFileMutation, WorkspaceSnapshot},
    receipt::Digest,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedPatch {
    patch_sha256: Digest,
    files: Vec<PreparedFileMutation>,
    hunk_count: usize,
}

/// Chunk-boundary-independent logical patch stream. Wire adapters feed decoded argument bytes into
/// this owner; `finish` performs the same exact virtual transaction used by executor revalidation.
/// The buffered first implementation deliberately makes no unsound prefix claims, leaving room for
/// future hunk-boundary checkpoints without changing the protocol or executor interface.
#[derive(Clone, Debug)]
pub struct PatchStream {
    snapshot: WorkspaceSnapshot,
    bytes: Vec<u8>,
    max_bytes: usize,
    max_files: usize,
    max_hunks: usize,
}

impl PatchStream {
    pub fn new(
        snapshot: WorkspaceSnapshot,
        max_bytes: usize,
        max_files: usize,
        max_hunks: usize,
    ) -> CollarResult<Self> {
        if max_bytes == 0 || max_files == 0 || max_hunks == 0 {
            return Err(CollarError::Mutation(
                "patch stream limits must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            snapshot,
            bytes: Vec::new(),
            max_bytes,
            max_files,
            max_hunks,
        })
    }

    pub fn push(&mut self, bytes: &[u8]) -> CollarResult<()> {
        let next_len = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| CollarError::Mutation("patch stream length overflow".to_string()))?;
        if next_len > self.max_bytes {
            return Err(CollarError::Mutation(format!(
                "patch stream exceeds the {}-byte limit",
                self.max_bytes
            )));
        }
        if bytes.contains(&0) {
            return Err(CollarError::Mutation(
                "canonical patches cannot contain NUL bytes".to_string(),
            ));
        }
        if bytes.contains(&b'\r') {
            return Err(CollarError::Mutation(
                "canonical patches require LF line endings".to_string(),
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn finish(self) -> CollarResult<PreparedPatch> {
        let patch = std::str::from_utf8(&self.bytes).map_err(|error| {
            CollarError::Mutation(format!(
                "canonical patch is not UTF-8 at byte {}",
                error.valid_up_to()
            ))
        })?;
        prepare_patch(&self.snapshot, patch, self.max_files, self.max_hunks)
    }
}

impl PreparedPatch {
    pub fn patch_sha256(&self) -> Digest {
        self.patch_sha256
    }

    pub fn files(&self) -> &[PreparedFileMutation] {
        &self.files
    }

    pub fn hunk_count(&self) -> usize {
        self.hunk_count
    }

    pub fn verify_live_bases<'a>(
        &self,
        mut read: impl FnMut(&LogicalPath) -> CollarResult<Option<&'a [u8]>>,
    ) -> CollarResult<()> {
        for file in &self.files {
            file.verify_live_base(read(file.path())?)?;
            file.revalidate_result()?;
        }
        Ok(())
    }
}

pub fn prepare_patch(
    snapshot: &WorkspaceSnapshot,
    patch: &str,
    max_files: usize,
    max_hunks: usize,
) -> CollarResult<PreparedPatch> {
    if patch.is_empty() {
        return Err(CollarError::Mutation("patch must not be empty".to_string()));
    }
    if max_files == 0 || max_hunks == 0 {
        return Err(CollarError::Mutation(
            "patch limits must be non-zero".to_string(),
        ));
    }
    if patch.as_bytes().contains(&0) {
        return Err(CollarError::Mutation(
            "canonical patches cannot contain NUL bytes".to_string(),
        ));
    }
    if patch.contains('\r') {
        return Err(CollarError::Mutation(
            "canonical patches require LF line endings".to_string(),
        ));
    }
    let lines = patch.lines().collect::<Vec<_>>();
    let mut parser = PatchParser {
        snapshot,
        lines: &lines,
        position: 0,
        max_files,
        max_hunks,
        hunk_count: 0,
        paths: BTreeSet::new(),
    };
    let files = parser.parse_files()?;
    Ok(PreparedPatch {
        patch_sha256: Digest::of(patch.as_bytes()),
        files,
        hunk_count: parser.hunk_count,
    })
}

struct PatchParser<'a> {
    snapshot: &'a WorkspaceSnapshot,
    lines: &'a [&'a str],
    position: usize,
    max_files: usize,
    max_hunks: usize,
    hunk_count: usize,
    paths: BTreeSet<LogicalPath>,
}

impl<'input> PatchParser<'input> {
    fn parse_files(&mut self) -> CollarResult<Vec<PreparedFileMutation>> {
        let mut files = Vec::new();
        while self.position < self.lines.len() {
            if files.len() >= self.max_files {
                return Err(CollarError::Mutation(format!(
                    "patch exceeds the {0}-file limit",
                    self.max_files
                )));
            }
            files.push(self.parse_file()?);
        }
        if files.is_empty() {
            return Err(CollarError::Mutation(
                "patch does not contain a file diff".to_string(),
            ));
        }
        Ok(files)
    }

    fn parse_file(&mut self) -> CollarResult<PreparedFileMutation> {
        let diff_paths = if self
            .peek()
            .is_some_and(|line| line.starts_with("diff --git "))
        {
            Some(self.parse_diff_header()?)
        } else {
            None
        };
        let creation_mode = self.consume_exact("new file mode 100644");
        let deletion_mode = self.consume_exact("deleted file mode 100644");
        if creation_mode && deletion_mode {
            return Err(CollarError::Mutation(
                "one file diff cannot be both created and deleted".to_string(),
            ));
        }
        if self.peek().is_some_and(|line| {
            line.starts_with("old mode ")
                || line.starts_with("new mode ")
                || line.starts_with("similarity index ")
                || line.starts_with("rename from ")
                || line.starts_with("rename to ")
                || line.starts_with("copy from ")
                || line.starts_with("copy to ")
                || line.starts_with("GIT binary patch")
                || line.starts_with("Binary files ")
                || line.starts_with("index ")
        }) {
            return Err(CollarError::Mutation(format!(
                "unsupported canonical patch metadata {:?}",
                self.peek().unwrap_or_default()
            )));
        }
        let old_header = self
            .next()
            .ok_or_else(|| CollarError::Mutation("patch is missing --- header".to_string()))?;
        let new_header = self
            .next()
            .ok_or_else(|| CollarError::Mutation("patch is missing +++ header".to_string()))?;
        let old_path = parse_file_header(old_header, "--- ", 'a')?;
        let new_path = parse_file_header(new_header, "+++ ", 'b')?;
        let (path, kind) = match (&old_path, &new_path) {
            (None, Some(path)) => (path.clone(), MutationKind::Create),
            (Some(path), None) => (path.clone(), MutationKind::Delete),
            (Some(old), Some(new)) if old == new => (old.clone(), MutationKind::Modify),
            (Some(old), Some(new)) => {
                return Err(CollarError::Mutation(format!(
                    "canonical patch rename from {:?} to {:?} is unsupported",
                    old.as_str(),
                    new.as_str()
                )));
            }
            (None, None) => {
                return Err(CollarError::Mutation(
                    "patch file headers cannot both be /dev/null".to_string(),
                ));
            }
        };
        if creation_mode != (kind == MutationKind::Create)
            || deletion_mode != (kind == MutationKind::Delete)
        {
            return Err(CollarError::Mutation(format!(
                "patch metadata does not match {kind:?} headers for {:?}",
                path.as_str()
            )));
        }
        if let Some((diff_old, diff_new)) = diff_paths {
            let expected_old = old_path.as_ref().unwrap_or(&path);
            let expected_new = new_path.as_ref().unwrap_or(&path);
            if diff_old != *expected_old || diff_new != *expected_new {
                return Err(CollarError::Mutation(
                    "diff --git paths do not match file headers".to_string(),
                ));
            }
        }
        if !self.paths.insert(path.clone()) {
            return Err(CollarError::Mutation(format!(
                "patch repeats file {:?}",
                path.as_str()
            )));
        }
        let base = match kind {
            MutationKind::Create => {
                if self.snapshot.contains(&path) {
                    return Err(CollarError::Mutation(format!(
                        "patch create target {:?} already exists",
                        path.as_str()
                    )));
                }
                &[][..]
            }
            MutationKind::Modify | MutationKind::Delete => {
                &self
                    .snapshot
                    .get(&path)
                    .ok_or_else(|| {
                        CollarError::Mutation(format!(
                            "patch target {:?} is missing from the snapshot",
                            path.as_str()
                        ))
                    })?
                    .bytes
            }
        };
        let result = self.apply_hunks(&path, base)?;
        if kind == MutationKind::Delete && !result.is_empty() {
            return Err(CollarError::Mutation(format!(
                "delete patch for {:?} leaves {} bytes",
                path.as_str(),
                result.len()
            )));
        }
        PreparedFileMutation::new(
            path,
            kind,
            (kind != MutationKind::Create).then_some(base),
            (kind != MutationKind::Delete).then_some(result),
        )
    }

    fn parse_diff_header(&mut self) -> CollarResult<(LogicalPath, LogicalPath)> {
        let line = self.next().expect("peeked diff header");
        let parts = line.split_ascii_whitespace().collect::<Vec<_>>();
        if parts.len() != 4 || parts[0] != "diff" || parts[1] != "--git" {
            return Err(CollarError::Mutation(format!(
                "invalid canonical diff header {line:?}"
            )));
        }
        Ok((
            parse_prefixed_path(parts[2], 'a')?,
            parse_prefixed_path(parts[3], 'b')?,
        ))
    }

    fn apply_hunks(&mut self, path: &LogicalPath, base: &[u8]) -> CollarResult<Vec<u8>> {
        let base_lines = split_lines(base);
        let mut base_cursor = 0usize;
        let mut result_lines = Vec::<Vec<u8>>::new();
        let mut file_hunks = 0usize;
        while self.peek().is_some_and(|line| line.starts_with("@@ ")) {
            if self.hunk_count >= self.max_hunks {
                return Err(CollarError::Mutation(format!(
                    "patch exceeds the {}-hunk limit",
                    self.max_hunks
                )));
            }
            self.hunk_count += 1;
            file_hunks += 1;
            let header = parse_hunk_header(self.next().expect("peeked hunk"))?;
            let old_index = hunk_index(header.old_start, header.old_count)?;
            if old_index < base_cursor || old_index > base_lines.len() {
                return Err(CollarError::Mutation(format!(
                    "hunk old offset {} is out of order or outside {:?}",
                    header.old_start,
                    path.as_str()
                )));
            }
            result_lines.extend(base_lines[base_cursor..old_index].iter().cloned());
            base_cursor = old_index;
            let expected_new_start = if header.new_count == 0 {
                result_lines.len()
            } else {
                result_lines.len().saturating_add(1)
            };
            if header.new_start != expected_new_start {
                return Err(CollarError::Mutation(format!(
                    "hunk new offset {} does not match virtual result line {} for {:?}",
                    header.new_start,
                    expected_new_start,
                    path.as_str()
                )));
            }
            let mut old_seen = 0usize;
            let mut new_seen = 0usize;
            let mut previous: Option<(HunkLineKind, usize)> = None;
            while old_seen < header.old_count || new_seen < header.new_count {
                let line = self.next().ok_or_else(|| {
                    CollarError::Mutation(format!(
                        "hunk for {:?} ends before its declared counts",
                        path.as_str()
                    ))
                })?;
                if line == "\\ No newline at end of file" {
                    let (kind, index) = previous.ok_or_else(|| {
                        CollarError::Mutation(
                            "no-newline marker has no preceding hunk line".to_string(),
                        )
                    })?;
                    match kind {
                        HunkLineKind::Context | HunkLineKind::Addition => {
                            remove_final_newline(result_lines.get_mut(index).ok_or_else(|| {
                                CollarError::Mutation(
                                    "no-newline marker references missing result line".to_string(),
                                )
                            })?);
                        }
                        HunkLineKind::Deletion => {}
                    }
                    if matches!(kind, HunkLineKind::Context | HunkLineKind::Deletion) {
                        let old_index = base_cursor.saturating_sub(1);
                        let old = base_lines.get(old_index).ok_or_else(|| {
                            CollarError::Mutation(
                                "no-newline marker references missing base line".to_string(),
                            )
                        })?;
                        if old.ends_with(b"\n") {
                            return Err(CollarError::Mutation(format!(
                                "no-newline marker disagrees with the base file for {:?}",
                                path.as_str()
                            )));
                        }
                    }
                    previous = None;
                    continue;
                }
                let (kind, text) = parse_hunk_line(line)?;
                let mut bytes = text.as_bytes().to_vec();
                bytes.push(b'\n');
                if self.peek() == Some("\\ No newline at end of file") {
                    remove_final_newline(&mut bytes);
                }
                match kind {
                    HunkLineKind::Context => {
                        match_base_line(path, &base_lines, base_cursor, &bytes)?;
                        base_cursor += 1;
                        old_seen += 1;
                        new_seen += 1;
                        result_lines.push(bytes);
                        previous = Some((kind, result_lines.len() - 1));
                    }
                    HunkLineKind::Deletion => {
                        match_base_line(path, &base_lines, base_cursor, &bytes)?;
                        base_cursor += 1;
                        old_seen += 1;
                        previous = Some((kind, result_lines.len()));
                    }
                    HunkLineKind::Addition => {
                        new_seen += 1;
                        result_lines.push(bytes);
                        previous = Some((kind, result_lines.len() - 1));
                    }
                }
                if old_seen > header.old_count || new_seen > header.new_count {
                    return Err(CollarError::Mutation(format!(
                        "hunk body exceeds declared counts for {:?}",
                        path.as_str()
                    )));
                }
            }
            if self.peek() == Some("\\ No newline at end of file") {
                let line = self.next().expect("peeked no-newline marker");
                debug_assert_eq!(line, "\\ No newline at end of file");
                let (kind, index) = previous.ok_or_else(|| {
                    CollarError::Mutation(
                        "no-newline marker has no preceding hunk line".to_string(),
                    )
                })?;
                if matches!(kind, HunkLineKind::Context | HunkLineKind::Addition) {
                    remove_final_newline(result_lines.get_mut(index).ok_or_else(|| {
                        CollarError::Mutation(
                            "no-newline marker references missing result line".to_string(),
                        )
                    })?);
                }
                if matches!(kind, HunkLineKind::Context | HunkLineKind::Deletion) {
                    let old = base_lines
                        .get(base_cursor.saturating_sub(1))
                        .ok_or_else(|| {
                            CollarError::Mutation(
                                "no-newline marker references missing base line".to_string(),
                            )
                        })?;
                    if old.ends_with(b"\n") {
                        return Err(CollarError::Mutation(format!(
                            "no-newline marker disagrees with the base file for {:?}",
                            path.as_str()
                        )));
                    }
                }
            }
        }
        if file_hunks == 0 {
            return Err(CollarError::Mutation(format!(
                "file diff for {:?} has no hunks",
                path.as_str()
            )));
        }
        result_lines.extend(base_lines[base_cursor..].iter().cloned());
        Ok(result_lines.concat())
    }

    fn peek(&self) -> Option<&'input str> {
        self.lines.get(self.position).copied()
    }

    fn next(&mut self) -> Option<&'input str> {
        let line = self.peek()?;
        self.position += 1;
        Some(line)
    }

    fn consume_exact(&mut self, expected: &str) -> bool {
        if self.peek() == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Copy)]
struct HunkHeader {
    old_start: usize,
    old_count: usize,
    new_start: usize,
    new_count: usize,
}

fn parse_hunk_header(line: &str) -> CollarResult<HunkHeader> {
    let rest = line
        .strip_prefix("@@ -")
        .ok_or_else(|| CollarError::Mutation(format!("invalid canonical hunk header {line:?}")))?;
    let (old, rest) = rest
        .split_once(" +")
        .ok_or_else(|| CollarError::Mutation(format!("invalid canonical hunk header {line:?}")))?;
    let (new, suffix) = rest
        .split_once(" @@")
        .ok_or_else(|| CollarError::Mutation(format!("invalid canonical hunk header {line:?}")))?;
    if !suffix.is_empty() && !suffix.starts_with(' ') {
        return Err(CollarError::Mutation(format!(
            "invalid canonical hunk suffix {suffix:?}"
        )));
    }
    let (old_start, old_count) = parse_range(old)?;
    let (new_start, new_count) = parse_range(new)?;
    Ok(HunkHeader {
        old_start,
        old_count,
        new_start,
        new_count,
    })
}

fn parse_range(range: &str) -> CollarResult<(usize, usize)> {
    let (start, count) = range.split_once(',').unwrap_or((range, "1"));
    let start = start
        .parse::<usize>()
        .map_err(|_| CollarError::Mutation(format!("invalid hunk range start {start:?}")))?;
    let count = count
        .parse::<usize>()
        .map_err(|_| CollarError::Mutation(format!("invalid hunk range count {count:?}")))?;
    if count > 0 && start == 0 {
        return Err(CollarError::Mutation(
            "non-empty hunk ranges are one-based".to_string(),
        ));
    }
    Ok((start, count))
}

fn hunk_index(start: usize, count: usize) -> CollarResult<usize> {
    if count == 0 {
        Ok(start)
    } else {
        start.checked_sub(1).ok_or_else(|| {
            CollarError::Mutation("non-empty hunk starts cannot be zero".to_string())
        })
    }
}

#[derive(Clone, Copy)]
enum HunkLineKind {
    Context,
    Deletion,
    Addition,
}

fn parse_hunk_line(line: &str) -> CollarResult<(HunkLineKind, &str)> {
    let Some(prefix) = line.as_bytes().first().copied() else {
        return Err(CollarError::Mutation(
            "empty hunk line lacks a prefix".to_string(),
        ));
    };
    match prefix {
        b' ' => Ok((HunkLineKind::Context, &line[1..])),
        b'-' => Ok((HunkLineKind::Deletion, &line[1..])),
        b'+' => Ok((HunkLineKind::Addition, &line[1..])),
        _ => Err(CollarError::Mutation(format!(
            "invalid hunk line prefix in {line:?}"
        ))),
    }
}

fn match_base_line(
    path: &LogicalPath,
    base: &[Vec<u8>],
    index: usize,
    expected: &[u8],
) -> CollarResult<()> {
    let actual = base.get(index).ok_or_else(|| {
        CollarError::Mutation(format!(
            "hunk reads beyond {:?} at old line {}",
            path.as_str(),
            index.saturating_add(1)
        ))
    })?;
    if actual != expected {
        return Err(CollarError::Mutation(format!(
            "hunk context does not match {:?} at old line {}",
            path.as_str(),
            index.saturating_add(1)
        )));
    }
    Ok(())
}

fn split_lines(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut lines = Vec::new();
    let mut start = 0usize;
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'\n' {
            lines.push(bytes[start..=index].to_vec());
            start = index + 1;
        }
    }
    if start < bytes.len() {
        lines.push(bytes[start..].to_vec());
    }
    lines
}

fn remove_final_newline(line: &mut Vec<u8>) {
    if line.last() == Some(&b'\n') {
        line.pop();
    }
}

fn parse_file_header(line: &str, prefix: &str, side: char) -> CollarResult<Option<LogicalPath>> {
    let path = line.strip_prefix(prefix).ok_or_else(|| {
        CollarError::Mutation(format!("expected {prefix:?} file header, found {line:?}"))
    })?;
    if path == "/dev/null" {
        return Ok(None);
    }
    Ok(Some(parse_prefixed_path(path, side)?))
}

fn parse_prefixed_path(path: &str, side: char) -> CollarResult<LogicalPath> {
    if path.contains(['\t', ' ']) || path.starts_with('"') {
        return Err(CollarError::Mutation(format!(
            "canonical patch path {path:?} cannot be quoted or timestamped"
        )));
    }
    let prefix = format!("{side}/");
    let relative = path.strip_prefix(&prefix).ok_or_else(|| {
        CollarError::Mutation(format!(
            "canonical patch path {path:?} must start with {prefix:?}"
        ))
    })?;
    LogicalPath::parse(relative)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutation::SnapshotEntry;

    fn snapshot(path: &str, bytes: &[u8]) -> WorkspaceSnapshot {
        let path = LogicalPath::parse(path).unwrap();
        WorkspaceSnapshot::new(vec![SnapshotEntry::new(path, bytes.to_vec())]).unwrap()
    }

    #[test]
    fn applies_exact_multi_hunk_modification_in_memory() {
        let base = b"one\ntwo\nthree\nfour\nfive\n";
        let snapshot = snapshot("notes.txt", base);
        let patch = "diff --git a/notes.txt b/notes.txt\n--- a/notes.txt\n+++ b/notes.txt\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n@@ -4,2 +4,2 @@\n four\n-five\n+FIVE\n";

        let prepared = prepare_patch(&snapshot, patch, 1, 2).unwrap();

        assert_eq!(prepared.hunk_count(), 2);
        assert_eq!(
            prepared.files()[0].result_bytes().unwrap(),
            b"one\nTWO\nthree\nfour\nFIVE\n"
        );
    }

    #[test]
    fn creates_and_deletes_with_exact_counts() {
        let snapshot = snapshot("old.txt", b"old\n");
        let patch = "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,1 @@\n+value = 1\ndiff --git a/old.txt b/old.txt\ndeleted file mode 100644\n--- a/old.txt\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-old\n";

        let prepared = prepare_patch(&snapshot, patch, 2, 2).unwrap();

        assert_eq!(prepared.files()[0].kind(), MutationKind::Create);
        assert_eq!(
            prepared.files()[0].result_bytes(),
            Some(&b"value = 1\n"[..])
        );
        assert_eq!(prepared.files()[1].kind(), MutationKind::Delete);
        assert_eq!(prepared.files()[1].result_bytes(), None);
    }

    #[test]
    fn rejects_bad_counts_offsets_context_and_syntax() {
        let snapshot = snapshot("main.py", b"value = 1\n");
        for patch in [
            "--- a/main.py\n+++ b/main.py\n@@ -1,2 +1,1 @@\n-value = 1\n+value = 2\n",
            "--- a/main.py\n+++ b/main.py\n@@ -2,1 +1,1 @@\n-value = 1\n+value = 2\n",
            "--- a/main.py\n+++ b/main.py\n@@ -1,1 +1,1 @@\n-wrong\n+value = 2\n",
            "--- a/main.py\n+++ b/main.py\n@@ -1,1 +1,1 @@\n-value = 1\n+def broken(:\n",
        ] {
            assert!(
                prepare_patch(&snapshot, patch, 1, 1).is_err(),
                "accepted {patch}"
            );
        }
    }

    #[test]
    fn rejects_renames_metadata_and_repeated_files() {
        let snapshot = snapshot("a.txt", b"a\n");
        let rename =
            "diff --git a/a.txt b/b.txt\n--- a/a.txt\n+++ b/b.txt\n@@ -1,1 +1,1 @@\n-a\n+b\n";
        assert!(prepare_patch(&snapshot, rename, 1, 1).is_err());

        let indexed = "diff --git a/a.txt b/a.txt\nindex 123..456 100644\n--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-a\n+b\n";
        assert!(prepare_patch(&snapshot, indexed, 1, 1).is_err());
    }

    #[test]
    fn preserves_explicit_missing_final_newline() {
        let snapshot = snapshot("a.txt", b"old");
        let patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n";
        let prepared = prepare_patch(&snapshot, patch, 1, 1).unwrap();
        assert_eq!(prepared.files()[0].result_bytes(), Some(&b"new"[..]));
    }

    #[test]
    fn patch_stream_is_independent_of_wire_chunk_boundaries() {
        let path = LogicalPath::parse("main.py").unwrap();
        let snapshot = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            path,
            b"one = 1\ntwo = 2\n".to_vec(),
        )])
        .unwrap();
        let patch = concat!(
            "diff --git a/main.py b/main.py\n",
            "--- a/main.py\n",
            "+++ b/main.py\n",
            "@@ -1,2 +1,2 @@\n",
            " one = 1\n",
            "-two = 2\n",
            "+two = 3\n",
        );
        let expected = prepare_patch(&snapshot, patch, 4, 8).unwrap();
        for width in 1..=patch.len() {
            let mut stream = PatchStream::new(snapshot.clone(), patch.len(), 4, 8).unwrap();
            for chunk in patch.as_bytes().chunks(width) {
                stream.push(chunk).unwrap();
            }
            assert_eq!(stream.finish().unwrap(), expected, "chunk width {width}");
        }
    }
}
