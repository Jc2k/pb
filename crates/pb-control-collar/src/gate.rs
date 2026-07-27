use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    CollarError,
    mutation::{FileStreamMode, LogicalPath, PatchStream, VirtualFileStream, prepare_replace},
    tool::CollarManifest,
};

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
    InvalidSyntax,
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
            Self::InvalidSyntax => "invalid_syntax",
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
        Ok(Self { manifest })
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
