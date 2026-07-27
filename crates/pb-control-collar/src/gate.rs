use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex};

use crate::{
    CollarError,
    analysis::{PrefixCheckpoint, PrefixReport, SourcePrefixOracle, Viability},
    mutation::{
        FileStreamMode, LogicalPath, PatchCheckpoint, PatchStream, VirtualFileStream,
        prepare_replace,
    },
    receipt::Digest,
    tool::CollarManifest,
};

const MAX_PREFIX_CACHE_CHECKPOINTS: usize = 4_096;

/// Content-free reason codes suitable for sampling statistics and durable telemetry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionCode {
    InvalidArguments,
    ArgumentLimit,
    NonCanonicalPath,
    ExistingCreateTarget,
    MissingSnapshot,
    NoContentChange,
    InvalidPrefix,
    InvalidSyntax,
    InvalidSemantics,
    InvalidPatch,
}

impl RejectionCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArguments => "invalid_arguments",
            Self::ArgumentLimit => "argument_limit",
            Self::NonCanonicalPath => "non_canonical_path",
            Self::ExistingCreateTarget => "existing_create_target",
            Self::MissingSnapshot => "missing_snapshot",
            Self::NoContentChange => "no_content_change",
            Self::InvalidPrefix => "invalid_prefix",
            Self::InvalidSyntax => "invalid_syntax",
            Self::InvalidSemantics => "invalid_semantics",
            Self::InvalidPatch => "invalid_patch",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionDecision {
    /// The call is not a mutation covered by this manifest.
    NotApplicable,
    Accept,
    Reject(RejectionCode),
}

/// Pure completion-time mutation policy. It owns no filesystem authority: every decision is made
/// from the controller-authorized snapshot embedded in the request manifest.
#[derive(Clone, Debug)]
pub struct MutationCompletionGate {
    manifest: CollarManifest,
    prefix_cache: Arc<Mutex<PrefixProbeCache>>,
    patch_cache: Arc<Mutex<PatchProbeCache>>,
}

impl MutationCompletionGate {
    pub fn new(manifest: CollarManifest) -> Result<Self, CollarError> {
        manifest.validate()?;
        if manifest.workspace.total_bytes() > manifest.limits.max_snapshot_bytes {
            return Err(CollarError::InvalidManifest(format!(
                "workspace snapshot exceeds the {}-byte limit",
                manifest.limits.max_snapshot_bytes
            )));
        }
        Ok(Self {
            manifest,
            prefix_cache: Arc::new(Mutex::new(PrefixProbeCache::default())),
            patch_cache: Arc::new(Mutex::new(PatchProbeCache::default())),
        })
    }

    pub fn manifest(&self) -> &CollarManifest {
        &self.manifest
    }

    pub fn evaluate(&self, name: &str, arguments: &Value) -> CompletionDecision {
        match name {
            "write_file" if self.manifest.mutation_policy.allow_write_file => {
                self.evaluate_write(arguments, false)
            }
            "replace_file" if self.manifest.mutation_policy.allow_replace_file => {
                self.evaluate_write(arguments, true)
            }
            "edit_file" if self.manifest.mutation_policy.allow_replace_file => {
                self.evaluate_edit(arguments)
            }
            "apply_patch" if self.manifest.mutation_policy.allow_apply_patch => {
                self.evaluate_patch(arguments)
            }
            _ => CompletionDecision::NotApplicable,
        }
    }

    /// Conservatively probes a decoded mutation payload before its enclosing string closes.
    /// `arguments` contains only parameters that have already closed on the wire; `payload_prefix`
    /// is the decoded logical prefix of the active mutation field.
    pub fn evaluate_prefix(
        &self,
        name: &str,
        arguments: &serde_json::Map<String, Value>,
        payload_prefix: &str,
    ) -> CompletionDecision {
        if payload_prefix.len() > self.manifest.limits.max_argument_bytes {
            return CompletionDecision::Reject(RejectionCode::ArgumentLimit);
        }
        match name {
            "write_file" if self.manifest.mutation_policy.allow_write_file => {
                self.evaluate_file_prefix(arguments, payload_prefix, false)
            }
            "replace_file" if self.manifest.mutation_policy.allow_replace_file => {
                self.evaluate_file_prefix(arguments, payload_prefix, true)
            }
            "edit_file" if self.manifest.mutation_policy.allow_replace_file => {
                self.evaluate_edit_prefix(arguments, payload_prefix)
            }
            "apply_patch" if self.manifest.mutation_policy.allow_apply_patch => {
                let result = self
                    .patch_cache
                    .lock()
                    .map_err(|_| {
                        CollarError::Mutation("patch prefix cache lock is poisoned".to_string())
                    })
                    .and_then(|mut cache| {
                        cache.probe(
                            &self.manifest,
                            payload_prefix.as_bytes(),
                            MAX_PREFIX_CACHE_CHECKPOINTS,
                        )
                    });
                match result {
                    Ok(()) => CompletionDecision::Accept,
                    Err(error) => CompletionDecision::Reject(classify_mutation_error(&error)),
                }
            }
            _ => CompletionDecision::NotApplicable,
        }
    }

    fn evaluate_file_prefix(
        &self,
        arguments: &serde_json::Map<String, Value>,
        payload_prefix: &str,
        replace: bool,
    ) -> CompletionDecision {
        let Some(path) = arguments.get("path").and_then(Value::as_str) else {
            return CompletionDecision::NotApplicable;
        };
        let path = match LogicalPath::parse(path.to_string()) {
            Ok(path) => path,
            Err(_) => return CompletionDecision::Reject(RejectionCode::NonCanonicalPath),
        };
        if (replace && !self.manifest.workspace.contains(&path))
            || (!replace && self.manifest.workspace.contains(&path))
        {
            return CompletionDecision::Reject(if replace {
                RejectionCode::MissingSnapshot
            } else {
                RejectionCode::ExistingCreateTarget
            });
        }
        self.prefix_decision(&path, &[], payload_prefix.as_bytes())
    }

    fn evaluate_edit_prefix(
        &self,
        arguments: &serde_json::Map<String, Value>,
        payload_prefix: &str,
    ) -> CompletionDecision {
        let (Some(path), Some(old_text)) = (
            arguments.get("path").and_then(Value::as_str),
            arguments.get("old_text").and_then(Value::as_str),
        ) else {
            return CompletionDecision::NotApplicable;
        };
        if old_text.is_empty() {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        }
        let path = match LogicalPath::parse(path.to_string()) {
            Ok(path) => path,
            Err(_) => return CompletionDecision::Reject(RejectionCode::NonCanonicalPath),
        };
        let Some(base) = self.manifest.workspace.get(&path) else {
            return CompletionDecision::Reject(RejectionCode::MissingSnapshot);
        };
        let Ok(base_text) = std::str::from_utf8(&base.bytes) else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        if base_text.matches(old_text).take(2).count() != 1 {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        }
        let Some(start) = base_text.find(old_text) else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        self.prefix_decision(&path, &base.bytes[..start], payload_prefix.as_bytes())
    }

    fn evaluate_write(&self, arguments: &Value, replace: bool) -> CompletionDecision {
        let Some(path) = arguments.get("path").and_then(Value::as_str) else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        let Some(content) = arguments.get("content").and_then(Value::as_str) else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        if content.len() > self.manifest.limits.max_argument_bytes {
            return CompletionDecision::Reject(RejectionCode::ArgumentLimit);
        }
        let path = match LogicalPath::parse(path.to_string()) {
            Ok(path) => path,
            Err(_) => return CompletionDecision::Reject(RejectionCode::NonCanonicalPath),
        };
        let mode = if replace {
            FileStreamMode::Replace
        } else {
            FileStreamMode::Create
        };
        let mut stream = match VirtualFileStream::new(
            self.manifest.workspace.clone(),
            path,
            mode,
            self.manifest.limits.max_argument_bytes,
        ) {
            Ok(stream) => stream,
            Err(error) => {
                return CompletionDecision::Reject(classify_mutation_error(&error));
            }
        };
        if let Err(error) = stream.push(content.as_bytes()) {
            return CompletionDecision::Reject(classify_mutation_error(&error));
        }
        let result = stream.finish();
        match result {
            Ok(_) => CompletionDecision::Accept,
            Err(error) => CompletionDecision::Reject(classify_mutation_error(&error)),
        }
    }

    fn evaluate_edit(&self, arguments: &Value) -> CompletionDecision {
        let Some(path) = arguments.get("path").and_then(Value::as_str) else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        let Some(old_text) = arguments.get("old_text").and_then(Value::as_str) else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        let Some(new_text) = arguments.get("new_text").and_then(Value::as_str) else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        if old_text.len().saturating_add(new_text.len()) > self.manifest.limits.max_argument_bytes
            || old_text.is_empty()
        {
            return CompletionDecision::Reject(RejectionCode::ArgumentLimit);
        }
        let path = match LogicalPath::parse(path.to_string()) {
            Ok(path) => path,
            Err(_) => return CompletionDecision::Reject(RejectionCode::NonCanonicalPath),
        };
        let Some(base) = self.manifest.workspace.get(&path) else {
            return CompletionDecision::Reject(RejectionCode::MissingSnapshot);
        };
        let Ok(base_text) = std::str::from_utf8(&base.bytes) else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        if base_text.matches(old_text).take(2).count() != 1 {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        }
        let result = base_text.replacen(old_text, new_text, 1);
        match prepare_replace(&self.manifest.workspace, path, result.into_bytes()) {
            Ok(_) => CompletionDecision::Accept,
            Err(error) => CompletionDecision::Reject(classify_mutation_error(&error)),
        }
    }

    fn evaluate_patch(&self, arguments: &Value) -> CompletionDecision {
        let Some(patch) = arguments.get("patch").and_then(Value::as_str) else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        if patch.len() > self.manifest.limits.max_argument_bytes {
            return CompletionDecision::Reject(RejectionCode::ArgumentLimit);
        }
        let mut stream = match PatchStream::new(
            self.manifest.workspace.clone(),
            self.manifest.limits.max_argument_bytes,
            self.manifest.limits.max_files,
            self.manifest.limits.max_patch_hunks,
        ) {
            Ok(stream) => stream,
            Err(error) => {
                return CompletionDecision::Reject(classify_mutation_error(&error));
            }
        };
        if let Err(error) = stream.push(patch.as_bytes()) {
            return CompletionDecision::Reject(classify_mutation_error(&error));
        }
        match stream.finish() {
            Ok(_) => CompletionDecision::Accept,
            Err(error) => CompletionDecision::Reject(classify_mutation_error(&error)),
        }
    }

    fn prefix_decision(
        &self,
        path: &LogicalPath,
        known_prefix: &[u8],
        generated_prefix: &[u8],
    ) -> CompletionDecision {
        let source_limit = match known_prefix
            .len()
            .checked_add(self.manifest.limits.max_argument_bytes)
        {
            Some(limit) => limit.max(1),
            None => return CompletionDecision::Reject(RejectionCode::InvalidPrefix),
        };
        let identity = PrefixProbeIdentity {
            path: path.clone(),
            known_prefix_sha256: Digest::of(known_prefix),
            known_prefix_len: known_prefix.len(),
            source_limit,
        };
        let report = self
            .prefix_cache
            .lock()
            .map_err(|_| CollarError::Analysis("source prefix cache lock is poisoned".to_string()))
            .and_then(|mut cache| cache.probe(identity, known_prefix, generated_prefix));
        match report {
            Ok(report) if report.viability == Viability::Impossible => {
                CompletionDecision::Reject(RejectionCode::InvalidPrefix)
            }
            Ok(report) if report.profile.is_some() => CompletionDecision::Accept,
            Ok(_) => CompletionDecision::NotApplicable,
            Err(_) => CompletionDecision::Reject(RejectionCode::InvalidPrefix),
        }
    }
}

fn classify_mutation_error(error: &CollarError) -> RejectionCode {
    match error {
        CollarError::Analysis(_) => RejectionCode::InvalidSyntax,
        CollarError::Mutation(message) if message.contains("already exists") => {
            RejectionCode::ExistingCreateTarget
        }
        CollarError::Mutation(message) if message.contains("missing from the snapshot") => {
            RejectionCode::MissingSnapshot
        }
        CollarError::Mutation(message) if message.contains("no content change") => {
            RejectionCode::NoContentChange
        }
        CollarError::Mutation(message)
            if message.contains("patch")
                || message.contains("hunk")
                || message.contains("diff")
                || message.contains("context")
                || message.contains("offset") =>
        {
            RejectionCode::InvalidPatch
        }
        CollarError::Mutation(_) => RejectionCode::InvalidArguments,
        _ => RejectionCode::InvalidArguments,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PrefixProbeIdentity {
    path: LogicalPath,
    known_prefix_sha256: Digest,
    known_prefix_len: usize,
    source_limit: usize,
}

#[derive(Debug, Default)]
struct PrefixProbeCache {
    entry: Option<CachedPrefixProbe>,
}

#[derive(Debug)]
struct CachedPrefixProbe {
    identity: PrefixProbeIdentity,
    oracle: SourcePrefixOracle,
    generated_prefix: Vec<u8>,
    checkpoints: Vec<(usize, PrefixCheckpoint)>,
}

impl PrefixProbeCache {
    fn probe(
        &mut self,
        identity: PrefixProbeIdentity,
        known_prefix: &[u8],
        generated_prefix: &[u8],
    ) -> Result<PrefixReport, CollarError> {
        if self
            .entry
            .as_ref()
            .is_none_or(|entry| entry.identity != identity)
        {
            let mut oracle = SourcePrefixOracle::new(identity.path.clone(), identity.source_limit)?;
            oracle.push(known_prefix)?;
            let checkpoint = oracle.checkpoint();
            self.entry = Some(CachedPrefixProbe {
                identity,
                oracle,
                generated_prefix: Vec::new(),
                checkpoints: vec![(0, checkpoint)],
            });
        }
        let entry = self.entry.as_mut().expect("prefix cache initialized above");
        let common = common_prefix_len(&entry.generated_prefix, generated_prefix);
        entry
            .checkpoints
            .retain(|(payload_len, _)| *payload_len <= common);
        let (checkpoint_len, checkpoint) = entry
            .checkpoints
            .last()
            .cloned()
            .expect("prefix cache always retains its zero checkpoint");
        entry.oracle.rollback(checkpoint)?;
        let report = entry.oracle.push(&generated_prefix[checkpoint_len..])?;
        entry.generated_prefix.clear();
        entry.generated_prefix.extend_from_slice(generated_prefix);
        if entry
            .checkpoints
            .last()
            .is_none_or(|(payload_len, _)| *payload_len != generated_prefix.len())
        {
            entry
                .checkpoints
                .push((generated_prefix.len(), entry.oracle.checkpoint()));
        }
        if entry.checkpoints.len() > MAX_PREFIX_CACHE_CHECKPOINTS {
            let remove = entry
                .checkpoints
                .len()
                .saturating_sub(MAX_PREFIX_CACHE_CHECKPOINTS);
            entry.checkpoints.drain(1..=remove);
        }
        Ok(report)
    }
}

fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

#[derive(Debug, Default)]
struct PatchProbeCache {
    entry: Option<CachedPatchProbe>,
}

#[derive(Debug)]
struct CachedPatchProbe {
    stream: PatchStream,
    generated_prefix: Vec<u8>,
    checkpoints: Vec<(usize, PatchCheckpoint)>,
}

impl PatchProbeCache {
    fn probe(
        &mut self,
        manifest: &CollarManifest,
        generated_prefix: &[u8],
        max_checkpoints: usize,
    ) -> Result<(), CollarError> {
        if self.entry.is_none() {
            let stream = PatchStream::new(
                manifest.workspace.clone(),
                manifest.limits.max_argument_bytes,
                manifest.limits.max_files,
                manifest.limits.max_patch_hunks,
            )?;
            let checkpoint = stream.checkpoint();
            self.entry = Some(CachedPatchProbe {
                stream,
                generated_prefix: Vec::new(),
                checkpoints: vec![(0, checkpoint)],
            });
        }
        let entry = self.entry.as_mut().expect("patch cache initialized above");
        let common = common_prefix_len(&entry.generated_prefix, generated_prefix);
        entry
            .checkpoints
            .retain(|(payload_len, _)| *payload_len <= common);
        let (checkpoint_len, checkpoint) = entry
            .checkpoints
            .last()
            .cloned()
            .expect("patch cache always retains its zero checkpoint");
        entry.stream.rollback(checkpoint.clone())?;
        if let Err(error) = entry.stream.push(&generated_prefix[checkpoint_len..]) {
            entry.stream.rollback(checkpoint)?;
            return Err(error);
        }
        entry.generated_prefix.clear();
        entry.generated_prefix.extend_from_slice(generated_prefix);
        if entry
            .checkpoints
            .last()
            .is_none_or(|(payload_len, _)| *payload_len != generated_prefix.len())
        {
            entry
                .checkpoints
                .push((generated_prefix.len(), entry.stream.checkpoint()));
        }
        if entry.checkpoints.len() > max_checkpoints {
            let remove = entry.checkpoints.len().saturating_sub(max_checkpoints);
            entry.checkpoints.drain(1..=remove);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        mutation::{SnapshotEntry, WorkspaceSnapshot},
        protocol::ToolDialect,
        tool::{CollarLimits, ExposedTool, MutationPolicy, ToolConstraintMode},
    };

    fn gate(entries: Vec<SnapshotEntry>) -> MutationCompletionGate {
        MutationCompletionGate::new(CollarManifest {
            contract_version: 1,
            dialect: ToolDialect::QwenJson,
            mode: ToolConstraintMode::ToolsAllowed,
            tools: ["write_file", "replace_file", "edit_file", "apply_patch"]
                .into_iter()
                .map(|name| ExposedTool {
                    name: name.to_string(),
                    input_schema: json!({"type":"object"}),
                })
                .collect(),
            terminal_tools: Vec::new(),
            mutation_policy: MutationPolicy {
                allow_write_file: true,
                allow_replace_file: true,
                allow_apply_patch: true,
                max_mutation_calls_per_batch: 1,
            },
            workspace: WorkspaceSnapshot::new(entries).unwrap(),
            limits: CollarLimits {
                max_argument_bytes: 64 * 1024,
                max_snapshot_bytes: 64 * 1024,
                max_files: 8,
                max_patch_hunks: 32,
            },
        })
        .unwrap()
    }

    #[test]
    fn write_completion_rejects_invalid_supported_syntax() {
        let gate = gate(Vec::new());
        assert_eq!(
            gate.evaluate(
                "write_file",
                &json!({"path":"src/lib.rs","content":"pub fn broken( {"})
            ),
            CompletionDecision::Reject(RejectionCode::InvalidSyntax)
        );
        assert_eq!(
            gate.evaluate(
                "write_file",
                &json!({"path":"src/lib.rs","content":"pub fn ok() {}\n"})
            ),
            CompletionDecision::Accept
        );
    }

    #[test]
    fn write_prefix_rejects_only_a_definite_impossible_transition() {
        let gate = gate(Vec::new());
        let arguments = json!({"path":"src/lib.rs"}).as_object().unwrap().clone();
        assert_eq!(
            gate.evaluate_prefix("write_file", &arguments, "fn value() { ]"),
            CompletionDecision::Reject(RejectionCode::InvalidPrefix)
        );
        assert_eq!(
            gate.evaluate_prefix("write_file", &arguments, "fn value() { let x = ("),
            CompletionDecision::Accept
        );
    }

    #[test]
    fn candidate_branch_caches_rollback_without_changing_decisions() {
        let path = LogicalPath::parse("main.py").unwrap();
        let gate = gate(vec![SnapshotEntry::new(path, b"value = (1)\n".to_vec())]);
        let write_arguments = json!({"path":"branch.rs"}).as_object().unwrap().clone();
        let common = "fn branch() { let values = [";
        assert_eq!(
            gate.evaluate_prefix("write_file", &write_arguments, &format!("{common}}}")),
            CompletionDecision::Reject(RejectionCode::InvalidPrefix)
        );
        assert_eq!(
            gate.evaluate_prefix("write_file", &write_arguments, &format!("{common}1]; }}")),
            CompletionDecision::Accept
        );

        let patch_arguments = serde_json::Map::new();
        let patch_common = concat!(
            "--- a/main.py\n",
            "+++ b/main.py\n",
            "@@ -1,1 +1,1 @@\n",
            "-value = (1)\n",
        );
        assert_eq!(
            gate.evaluate_prefix(
                "apply_patch",
                &patch_arguments,
                &format!("{patch_common}+value = ]")
            ),
            CompletionDecision::Reject(RejectionCode::InvalidPatch)
        );
        assert_eq!(
            gate.evaluate_prefix(
                "apply_patch",
                &patch_arguments,
                &format!("{patch_common}+value = [1, 2]")
            ),
            CompletionDecision::Accept
        );
    }

    #[test]
    fn patch_completion_uses_the_authorized_snapshot() {
        let path = LogicalPath::parse("main.py").unwrap();
        let gate = gate(vec![SnapshotEntry::new(path, b"value = 1\n".to_vec())]);
        let valid = concat!(
            "diff --git a/main.py b/main.py\n",
            "--- a/main.py\n",
            "+++ b/main.py\n",
            "@@ -1,1 +1,1 @@\n",
            "-value = 1\n",
            "+value = 2\n",
        );
        assert_eq!(
            gate.evaluate("apply_patch", &json!({"patch": valid})),
            CompletionDecision::Accept
        );
        assert_eq!(
            gate.evaluate(
                "apply_patch",
                &json!({"patch": valid.replace("value = 1", "value = 9")})
            ),
            CompletionDecision::Reject(RejectionCode::InvalidPatch)
        );
    }
}
