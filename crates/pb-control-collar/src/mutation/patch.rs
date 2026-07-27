use std::{collections::BTreeSet, sync::Arc};

use crate::{
    CollarError, CollarResult,
    analysis::{SourcePrefixOracle, Viability},
    mutation::{LogicalPath, MutationKind, PreparedFileMutation, WorkspaceSnapshot},
    receipt::Digest,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedPatch {
    patch_sha256: Digest,
    files: Vec<PreparedFileMutation>,
    hunk_count: usize,
}

/// Chunk-boundary-independent logical patch stream. Completed patch lines are validated as soon as
/// they arrive, including canonical metadata, exact offsets/counts, context/deletion bytes, and
/// file/hunk bounds. `finish` still performs the authoritative batch transaction used by executor
/// revalidation, and tests require streaming/batch agreement.
#[derive(Clone, Debug)]
pub struct PatchStream {
    snapshot: WorkspaceSnapshot,
    bytes: Vec<u8>,
    processed_bytes: usize,
    prefix: PatchPrefixValidator,
    max_bytes: usize,
    max_files: usize,
    max_hunks: usize,
    stream_id: u64,
}

#[derive(Clone, Debug)]
pub struct PatchCheckpoint {
    stream_id: u64,
    bytes_len: usize,
    processed_bytes: usize,
    prefix: PatchPrefixValidator,
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
            processed_bytes: 0,
            prefix: PatchPrefixValidator::default(),
            max_bytes,
            max_files,
            max_hunks,
            stream_id: next_patch_stream_id(),
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
        while let Some(relative_end) = self.bytes[self.processed_bytes..]
            .iter()
            .position(|byte| *byte == b'\n')
        {
            let end = self.processed_bytes + relative_end;
            let line =
                std::str::from_utf8(&self.bytes[self.processed_bytes..end]).map_err(|error| {
                    CollarError::Mutation(format!(
                        "canonical patch line is not UTF-8 at byte {}",
                        self.processed_bytes.saturating_add(error.valid_up_to())
                    ))
                })?;
            self.prefix.push_line(
                &self.snapshot,
                line,
                self.max_bytes,
                self.max_files,
                self.max_hunks,
            )?;
            self.processed_bytes = end + 1;
        }
        self.prefix
            .probe_partial_line(&self.bytes[self.processed_bytes..])?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn checkpoint(&self) -> PatchCheckpoint {
        PatchCheckpoint {
            stream_id: self.stream_id,
            bytes_len: self.bytes.len(),
            processed_bytes: self.processed_bytes,
            prefix: self.prefix.clone(),
        }
    }

    pub fn rollback(&mut self, checkpoint: PatchCheckpoint) -> CollarResult<()> {
        if checkpoint.stream_id != self.stream_id {
            return Err(CollarError::Mutation(
                "patch checkpoint belongs to another stream".to_string(),
            ));
        }
        if checkpoint.bytes_len > self.bytes.len()
            || checkpoint.processed_bytes > checkpoint.bytes_len
        {
            return Err(CollarError::Mutation(
                "patch checkpoint is ahead of the current stream".to_string(),
            ));
        }
        self.bytes.truncate(checkpoint.bytes_len);
        self.processed_bytes = checkpoint.processed_bytes;
        self.prefix = checkpoint.prefix;
        Ok(())
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

#[derive(Clone, Debug, Default)]
struct PatchPrefixValidator {
    phase: PatchPrefixPhase,
    paths: BTreeSet<LogicalPath>,
    files: usize,
    hunks: usize,
}

#[derive(Clone, Debug, Default)]
enum PatchPrefixPhase {
    #[default]
    Start,
    Metadata(PrefixMetadata),
    NewHeader {
        metadata: PrefixMetadata,
        old_path: Option<LogicalPath>,
    },
    File(PrefixFile),
}

#[derive(Clone, Debug, Default)]
struct PrefixMetadata {
    diff_paths: Option<(LogicalPath, LogicalPath)>,
    creation_mode: bool,
    deletion_mode: bool,
}

#[derive(Clone, Debug)]
struct PrefixFile {
    path: LogicalPath,
    kind: MutationKind,
    base_lines: Arc<Vec<Vec<u8>>>,
    base_cursor: usize,
    result_lines: usize,
    result_oracle: SourcePrefixOracle,
    file_hunks: usize,
    hunk: Option<PrefixHunk>,
}

#[derive(Clone, Debug)]
struct PrefixHunk {
    header: HunkHeader,
    old_seen: usize,
    new_seen: usize,
    pending: Option<(HunkLineKind, Vec<u8>)>,
}

impl PatchPrefixValidator {
    fn probe_partial_line(&self, bytes: &[u8]) -> CollarResult<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        match std::str::from_utf8(bytes) {
            Ok(_) => {}
            Err(error) if error.error_len().is_none() => return Ok(()),
            Err(error) => {
                return Err(CollarError::Mutation(format!(
                    "canonical patch line prefix is not UTF-8 at byte {}",
                    error.valid_up_to()
                )));
            }
        }
        let PatchPrefixPhase::File(file) = &self.phase else {
            return Ok(());
        };
        file.probe_partial_hunk_line(bytes)
    }

    fn push_line(
        &mut self,
        snapshot: &WorkspaceSnapshot,
        line: &str,
        max_patch_bytes: usize,
        max_files: usize,
        max_hunks: usize,
    ) -> CollarResult<()> {
        let phase = std::mem::take(&mut self.phase);
        self.phase = match phase {
            PatchPrefixPhase::Start => self.start_file(line, max_files)?,
            PatchPrefixPhase::Metadata(metadata) => {
                self.consume_metadata(metadata, line, max_files)?
            }
            PatchPrefixPhase::NewHeader { metadata, old_path } => {
                self.consume_new_header(snapshot, metadata, old_path, line, max_patch_bytes)?
            }
            PatchPrefixPhase::File(mut file) => {
                if file.consume_line(line, &mut self.hunks, max_hunks)? {
                    PatchPrefixPhase::File(file)
                } else {
                    if file.file_hunks == 0 {
                        return Err(CollarError::Mutation(format!(
                            "file diff for {:?} has no hunks",
                            file.path.as_str()
                        )));
                    }
                    self.phase = self.start_file(line, max_files)?;
                    return Ok(());
                }
            }
        };
        Ok(())
    }

    fn start_file(&mut self, line: &str, max_files: usize) -> CollarResult<PatchPrefixPhase> {
        if self.files >= max_files {
            return Err(CollarError::Mutation(format!(
                "patch exceeds the {max_files}-file limit"
            )));
        }
        let mut metadata = PrefixMetadata::default();
        if line.starts_with("diff --git ") {
            metadata.diff_paths = Some(parse_diff_header_line(line)?);
            return Ok(PatchPrefixPhase::Metadata(metadata));
        }
        self.consume_metadata(metadata, line, max_files)
    }

    fn consume_metadata(
        &mut self,
        mut metadata: PrefixMetadata,
        line: &str,
        _max_files: usize,
    ) -> CollarResult<PatchPrefixPhase> {
        if line == "new file mode 100644" && !metadata.creation_mode && !metadata.deletion_mode {
            metadata.creation_mode = true;
            return Ok(PatchPrefixPhase::Metadata(metadata));
        }
        if line == "deleted file mode 100644" && !metadata.deletion_mode {
            metadata.deletion_mode = true;
            if metadata.creation_mode {
                return Err(CollarError::Mutation(
                    "one file diff cannot be both created and deleted".to_string(),
                ));
            }
            return Ok(PatchPrefixPhase::Metadata(metadata));
        }
        reject_unsupported_metadata(line)?;
        let old_path = parse_file_header(line, "--- ", 'a')?;
        Ok(PatchPrefixPhase::NewHeader { metadata, old_path })
    }

    fn consume_new_header(
        &mut self,
        snapshot: &WorkspaceSnapshot,
        metadata: PrefixMetadata,
        old_path: Option<LogicalPath>,
        line: &str,
        max_patch_bytes: usize,
    ) -> CollarResult<PatchPrefixPhase> {
        let new_path = parse_file_header(line, "+++ ", 'b')?;
        let (path, kind) = patch_path_kind(old_path.as_ref(), new_path.as_ref())?;
        if metadata.creation_mode != (kind == MutationKind::Create)
            || metadata.deletion_mode != (kind == MutationKind::Delete)
        {
            return Err(CollarError::Mutation(format!(
                "patch metadata does not match {kind:?} headers for {:?}",
                path.as_str()
            )));
        }
        if let Some((diff_old, diff_new)) = metadata.diff_paths {
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
                if snapshot.contains(&path) {
                    return Err(CollarError::Mutation(format!(
                        "patch create target {:?} already exists",
                        path.as_str()
                    )));
                }
                Vec::new()
            }
            MutationKind::Modify | MutationKind::Delete => split_lines(
                &snapshot
                    .get(&path)
                    .ok_or_else(|| {
                        CollarError::Mutation(format!(
                            "patch target {:?} is missing from the snapshot",
                            path.as_str()
                        ))
                    })?
                    .bytes,
            ),
        };
        let base_bytes = base.iter().try_fold(0usize, |total, line| {
            total
                .checked_add(line.len())
                .ok_or_else(|| CollarError::Mutation("virtual file length overflow".to_string()))
        })?;
        let result_limit = base_bytes.checked_add(max_patch_bytes).ok_or_else(|| {
            CollarError::Mutation("virtual result byte limit overflow".to_string())
        })?;
        let result_oracle = SourcePrefixOracle::new(path.clone(), result_limit.max(1))?;
        self.files = self.files.saturating_add(1);
        Ok(PatchPrefixPhase::File(PrefixFile {
            path,
            kind,
            base_lines: Arc::new(base),
            base_cursor: 0,
            result_lines: 0,
            result_oracle,
            file_hunks: 0,
            hunk: None,
        }))
    }
}

impl PrefixFile {
    fn probe_partial_hunk_line(&self, line: &[u8]) -> CollarResult<()> {
        let Some(hunk) = self.hunk.as_ref() else {
            return Ok(());
        };
        const NO_NEWLINE_MARKER: &[u8] = b"\\ No newline at end of file";
        if line.first() == Some(&b'\\') {
            if hunk.pending.is_some() && NO_NEWLINE_MARKER.starts_with(line) {
                return Ok(());
            }
            return Err(CollarError::Mutation(format!(
                "invalid no-final-newline marker prefix for {:?}",
                self.path.as_str()
            )));
        }

        let mut old_seen = hunk.old_seen;
        let mut new_seen = hunk.new_seen;
        let mut base_cursor = self.base_cursor;
        let mut oracle = self.result_oracle.clone();
        if let Some((kind, bytes)) = &hunk.pending {
            match kind {
                HunkLineKind::Context => {
                    let mut bytes = bytes.clone();
                    bytes.push(b'\n');
                    oracle.push(&bytes)?;
                    base_cursor = base_cursor.saturating_add(1);
                    old_seen = old_seen.saturating_add(1);
                    new_seen = new_seen.saturating_add(1);
                }
                HunkLineKind::Deletion => {
                    base_cursor = base_cursor.saturating_add(1);
                    old_seen = old_seen.saturating_add(1);
                }
                HunkLineKind::Addition => {
                    let mut bytes = bytes.clone();
                    bytes.push(b'\n');
                    oracle.push(&bytes)?;
                    new_seen = new_seen.saturating_add(1);
                }
            }
        }
        if old_seen == hunk.header.old_count && new_seen == hunk.header.new_count {
            return Ok(());
        }
        let (kind, content) = match line.split_first() {
            Some((b' ', content)) => (HunkLineKind::Context, content),
            Some((b'-', content)) => (HunkLineKind::Deletion, content),
            Some((b'+', content)) => (HunkLineKind::Addition, content),
            _ => {
                return Err(CollarError::Mutation(format!(
                    "invalid canonical hunk-line prefix for {:?}",
                    self.path.as_str()
                )));
            }
        };
        let next_old = old_seen.saturating_add(usize::from(matches!(
            kind,
            HunkLineKind::Context | HunkLineKind::Deletion
        )));
        let next_new = new_seen.saturating_add(usize::from(matches!(
            kind,
            HunkLineKind::Context | HunkLineKind::Addition
        )));
        if next_old > hunk.header.old_count || next_new > hunk.header.new_count {
            return Err(CollarError::Mutation(format!(
                "hunk-line prefix exceeds declared counts for {:?}",
                self.path.as_str()
            )));
        }
        if matches!(kind, HunkLineKind::Context | HunkLineKind::Deletion) {
            let base = self.base_lines.get(base_cursor).ok_or_else(|| {
                CollarError::Mutation(format!(
                    "hunk-line prefix is outside base {:?}",
                    self.path.as_str()
                ))
            })?;
            if !base.starts_with(content) {
                return Err(CollarError::Mutation(format!(
                    "hunk-line prefix does not match context in {:?}",
                    self.path.as_str()
                )));
            }
        }
        if matches!(kind, HunkLineKind::Context | HunkLineKind::Addition) {
            let report = oracle.push(content)?;
            if report.viability == Viability::Impossible {
                return Err(CollarError::Mutation(format!(
                    "canonical patch makes the generated virtual prefix impossible for {:?} ({:?})",
                    self.path.as_str(),
                    report.rule
                )));
            }
        }
        Ok(())
    }

    /// Returns false when `line` starts the next file diff and must be replayed by the outer state.
    fn consume_line(
        &mut self,
        line: &str,
        total_hunks: &mut usize,
        max_hunks: usize,
    ) -> CollarResult<bool> {
        if let Some(mut hunk) = self.hunk.take() {
            if let Some((kind, bytes)) = hunk.pending.take() {
                if line == "\\ No newline at end of file" {
                    self.commit_hunk_line(&mut hunk, kind, bytes)?;
                    self.hunk = Some(hunk);
                    return Ok(true);
                }
                let mut bytes = bytes;
                bytes.push(b'\n');
                self.commit_hunk_line(&mut hunk, kind, bytes)?;
            }
            if hunk.old_seen < hunk.header.old_count || hunk.new_seen < hunk.header.new_count {
                let (kind, text) = parse_hunk_line(line)?;
                let next_old = hunk.old_seen.saturating_add(usize::from(matches!(
                    kind,
                    HunkLineKind::Context | HunkLineKind::Deletion
                )));
                let next_new = hunk.new_seen.saturating_add(usize::from(matches!(
                    kind,
                    HunkLineKind::Context | HunkLineKind::Addition
                )));
                if next_old > hunk.header.old_count || next_new > hunk.header.new_count {
                    return Err(CollarError::Mutation(format!(
                        "hunk body exceeds declared counts for {:?}",
                        self.path.as_str()
                    )));
                }
                hunk.pending = Some((kind, text.as_bytes().to_vec()));
                self.hunk = Some(hunk);
                return Ok(true);
            }
            self.hunk = None;
        }

        if line.starts_with("@@ ") {
            if *total_hunks >= max_hunks {
                return Err(CollarError::Mutation(format!(
                    "patch exceeds the {max_hunks}-hunk limit"
                )));
            }
            let header = parse_hunk_header(line)?;
            let old_index = hunk_index(header.old_start, header.old_count)?;
            if old_index < self.base_cursor || old_index > self.base_lines.len() {
                return Err(CollarError::Mutation(format!(
                    "hunk old offset {} is out of order or outside {:?}",
                    header.old_start,
                    self.path.as_str()
                )));
            }
            self.append_untouched_base(old_index)?;
            self.result_lines = self
                .result_lines
                .saturating_add(old_index.saturating_sub(self.base_cursor));
            self.base_cursor = old_index;
            let expected_new_start = if header.new_count == 0 {
                self.result_lines
            } else {
                self.result_lines.saturating_add(1)
            };
            if header.new_start != expected_new_start {
                return Err(CollarError::Mutation(format!(
                    "hunk new offset {} does not match virtual result line {} for {:?}",
                    header.new_start,
                    expected_new_start,
                    self.path.as_str()
                )));
            }
            *total_hunks = total_hunks.saturating_add(1);
            self.file_hunks = self.file_hunks.saturating_add(1);
            self.hunk = Some(PrefixHunk {
                header,
                old_seen: 0,
                new_seen: 0,
                pending: None,
            });
            return Ok(true);
        }
        if line.starts_with("diff --git ")
            || line.starts_with("--- ")
            || line == "new file mode 100644"
            || line == "deleted file mode 100644"
        {
            self.append_untouched_base(self.base_lines.len())?;
            return Ok(false);
        }
        Err(CollarError::Mutation(format!(
            "expected a canonical hunk header for {:?}, found {line:?}",
            self.path.as_str()
        )))
    }

    fn commit_hunk_line(
        &mut self,
        hunk: &mut PrefixHunk,
        kind: HunkLineKind,
        bytes: Vec<u8>,
    ) -> CollarResult<()> {
        match kind {
            HunkLineKind::Context => {
                match_base_line(&self.path, &self.base_lines, self.base_cursor, &bytes)?;
                self.append_result_bytes(&bytes)?;
                self.base_cursor = self.base_cursor.saturating_add(1);
                self.result_lines = self.result_lines.saturating_add(1);
                hunk.old_seen = hunk.old_seen.saturating_add(1);
                hunk.new_seen = hunk.new_seen.saturating_add(1);
            }
            HunkLineKind::Deletion => {
                match_base_line(&self.path, &self.base_lines, self.base_cursor, &bytes)?;
                self.base_cursor = self.base_cursor.saturating_add(1);
                hunk.old_seen = hunk.old_seen.saturating_add(1);
            }
            HunkLineKind::Addition => {
                self.append_result_bytes(&bytes)?;
                self.result_lines = self.result_lines.saturating_add(1);
                hunk.new_seen = hunk.new_seen.saturating_add(1);
            }
        }
        Ok(())
    }

    fn append_untouched_base(&mut self, end: usize) -> CollarResult<()> {
        if end < self.base_cursor || end > self.base_lines.len() {
            return Err(CollarError::Mutation(format!(
                "virtual prefix range is invalid for {:?}",
                self.path.as_str()
            )));
        }
        if self.kind != MutationKind::Delete {
            for index in self.base_cursor..end {
                let bytes = self.base_lines[index].clone();
                self.append_result_bytes(&bytes)?;
            }
        }
        Ok(())
    }

    fn append_result_bytes(&mut self, bytes: &[u8]) -> CollarResult<()> {
        if self.kind != MutationKind::Delete {
            let report = self.result_oracle.push(bytes)?;
            if report.viability == Viability::Impossible {
                return Err(CollarError::Mutation(format!(
                    "canonical patch makes the committed virtual prefix impossible for {:?} ({:?})",
                    self.path.as_str(),
                    report.rule
                )));
            }
        }
        Ok(())
    }
}

fn parse_diff_header_line(line: &str) -> CollarResult<(LogicalPath, LogicalPath)> {
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

fn next_patch_stream_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed)
}

fn reject_unsupported_metadata(line: &str) -> CollarResult<()> {
    if line.starts_with("old mode ")
        || line.starts_with("new mode ")
        || line.starts_with("similarity index ")
        || line.starts_with("rename from ")
        || line.starts_with("rename to ")
        || line.starts_with("copy from ")
        || line.starts_with("copy to ")
        || line.starts_with("GIT binary patch")
        || line.starts_with("Binary files ")
        || line.starts_with("index ")
    {
        return Err(CollarError::Mutation(format!(
            "unsupported canonical patch metadata {line:?}"
        )));
    }
    Ok(())
}

fn patch_path_kind(
    old_path: Option<&LogicalPath>,
    new_path: Option<&LogicalPath>,
) -> CollarResult<(LogicalPath, MutationKind)> {
    match (old_path, new_path) {
        (None, Some(path)) => Ok((path.clone(), MutationKind::Create)),
        (Some(path), None) => Ok((path.clone(), MutationKind::Delete)),
        (Some(old), Some(new)) if old == new => Ok((old.clone(), MutationKind::Modify)),
        (Some(old), Some(new)) => Err(CollarError::Mutation(format!(
            "canonical patch rename from {:?} to {:?} is unsupported",
            old.as_str(),
            new.as_str()
        ))),
        (None, None) => Err(CollarError::Mutation(
            "patch file headers cannot both be /dev/null".to_string(),
        )),
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

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, Debug)]
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

    fn assert_stream_equivalent(
        snapshot: &WorkspaceSnapshot,
        patch: &str,
        max_files: usize,
        max_hunks: usize,
    ) {
        let expected = prepare_patch(snapshot, patch, max_files, max_hunks).unwrap();
        for width in 1..=patch.len() {
            let mut stream =
                PatchStream::new(snapshot.clone(), patch.len(), max_files, max_hunks).unwrap();
            for chunk in patch.as_bytes().chunks(width) {
                stream
                    .push(chunk)
                    .unwrap_or_else(|error| panic!("chunk width {width}: {error}"));
            }
            assert_eq!(stream.finish().unwrap(), expected, "chunk width {width}");
        }
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
        assert_stream_equivalent(&snapshot, patch, 4, 8);
    }

    #[test]
    fn patch_stream_accepts_every_complete_canonical_dialect_shape() {
        let base = b"one\ntwo\nthree\nfour\nfive\n";
        let notes_snapshot = snapshot("notes.txt", base);
        let multi_hunk = concat!(
            "diff --git a/notes.txt b/notes.txt\n",
            "--- a/notes.txt\n",
            "+++ b/notes.txt\n",
            "@@ -1,2 +1,2 @@\n",
            " one\n",
            "-two\n",
            "+TWO\n",
            "@@ -4,2 +4,2 @@\n",
            " four\n",
            "-five\n",
            "+FIVE\n",
        );
        assert_stream_equivalent(&notes_snapshot, multi_hunk, 1, 2);

        let create_delete_snapshot = snapshot("old.txt", b"old\n");
        let create_delete = concat!(
            "diff --git a/new.py b/new.py\n",
            "new file mode 100644\n",
            "--- /dev/null\n",
            "+++ b/new.py\n",
            "@@ -0,0 +1,1 @@\n",
            "+value = 1\n",
            "diff --git a/old.txt b/old.txt\n",
            "deleted file mode 100644\n",
            "--- a/old.txt\n",
            "+++ /dev/null\n",
            "@@ -1,1 +0,0 @@\n",
            "-old\n",
        );
        assert_stream_equivalent(&create_delete_snapshot, create_delete, 2, 2);

        let no_newline_snapshot = snapshot("a.txt", b"old");
        let no_newline = concat!(
            "--- a/a.txt\n",
            "+++ b/a.txt\n",
            "@@ -1,1 +1,1 @@\n",
            "-old\n",
            "\\ No newline at end of file\n",
            "+new\n",
            "\\ No newline at end of file\n",
        );
        assert_stream_equivalent(&no_newline_snapshot, no_newline, 1, 1);
    }

    #[test]
    fn patch_stream_rejects_offsets_and_context_before_payload_closure() {
        let snapshot = snapshot("main.py", b"value = 1\nnext = 2\n");
        let mut bad_offset = PatchStream::new(snapshot.clone(), 4096, 4, 8).unwrap();
        assert!(
            bad_offset
                .push(b"--- a/main.py\n+++ b/main.py\n@@ -3,1 +1,1 @@\n")
                .is_err()
        );

        let mut bad_context = PatchStream::new(snapshot, 4096, 4, 8).unwrap();
        bad_context
            .push(b"--- a/main.py\n+++ b/main.py\n@@ -1,1 +1,1 @@\n wrong\n")
            .unwrap();
        assert!(bad_context.push(b"--- a/next.py\n").is_err());
    }

    #[test]
    fn patch_stream_validates_the_committed_virtual_source_between_hunks() {
        let impossible_snapshot = snapshot("main.py", b"value = 1\nnext_value = 2\n");
        let mut impossible = PatchStream::new(impossible_snapshot, 4096, 1, 2).unwrap();
        let error = impossible
            .push(
                concat!(
                    "--- a/main.py\n",
                    "+++ b/main.py\n",
                    "@@ -1,1 +1,2 @@\n",
                    " value = 1\n",
                    "+)\n",
                    "@@ -2,1 +3,1 @@\n",
                )
                .as_bytes(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("virtual prefix impossible"));

        // A generated closer is viable when untouched source before the hunk supplied its opener.
        let snapshot = snapshot("main.py", b"values = [\nnext_value = 2\n");
        let patch = concat!(
            "--- a/main.py\n",
            "+++ b/main.py\n",
            "@@ -1,1 +1,2 @@\n",
            " values = [\n",
            "+]\n",
            "@@ -2,1 +3,1 @@\n",
            " next_value = 2\n",
        );
        assert_stream_equivalent(&snapshot, patch, 1, 2);
    }

    #[test]
    fn patch_stream_probes_partial_hunk_lines_before_their_newline() {
        let snapshot = snapshot("main.py", b"value = (1)\n");
        let header = concat!(
            "--- a/main.py\n",
            "+++ b/main.py\n",
            "@@ -1,1 +1,1 @@\n",
            "-value = (1)\n",
        );

        let mut impossible = PatchStream::new(snapshot.clone(), 4096, 1, 1).unwrap();
        impossible.push(header.as_bytes()).unwrap();
        let error = impossible.push(b"+value = ]").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("generated virtual prefix impossible")
        );

        let mut stale = PatchStream::new(snapshot.clone(), 4096, 1, 1).unwrap();
        stale
            .push(concat!("--- a/main.py\n", "+++ b/main.py\n", "@@ -1,1 +1,1 @@\n",).as_bytes())
            .unwrap();
        assert!(stale.push(b" wrong").is_err());

        let mut valid = PatchStream::new(snapshot, 4096, 1, 1).unwrap();
        valid.push(header.as_bytes()).unwrap();
        valid.push(b"+value = [").unwrap();
        valid.push(b"1, 2]\n").unwrap();
        assert_eq!(
            valid.finish().unwrap().files()[0].result_bytes(),
            Some(&b"value = [1, 2]\n"[..])
        );
    }

    #[test]
    fn patch_checkpoints_restore_exact_partial_line_and_stream_state() {
        let snapshot = snapshot("main.py", b"value = (1)\n");
        let header = concat!(
            "--- a/main.py\n",
            "+++ b/main.py\n",
            "@@ -1,1 +1,1 @@\n",
            "-value = (1)\n",
        );
        let mut stream = PatchStream::new(snapshot.clone(), 4096, 1, 1).unwrap();
        stream.push(header.as_bytes()).unwrap();
        let checkpoint = stream.checkpoint();
        assert!(stream.push(b"+value = ]").is_err());
        stream.rollback(checkpoint.clone()).unwrap();
        stream.push(b"+value = [1, 2]\n").unwrap();
        assert_eq!(
            stream.finish().unwrap().files()[0].result_bytes(),
            Some(&b"value = [1, 2]\n"[..])
        );

        let mut other = PatchStream::new(snapshot, 4096, 1, 1).unwrap();
        assert!(other.rollback(checkpoint).is_err());
    }
}
