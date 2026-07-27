use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
};

use crate::{
    CollarError,
    analysis::{
        Analysis, LanguageLayerStack, LayerStackCheckpoint, PrefixCheckpoint, PrefixReport,
        SourceEvent, SourceOrigin, SourcePrefixOracle, SyntaxProfile, Viability,
    },
    mutation::{
        FileStreamMode, LogicalPath, PatchCheckpoint, PatchStream, PatchVirtualFile,
        VirtualFileStream, prepare_replace,
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
    language_layers: Option<Arc<Mutex<LanguageLayerStack>>>,
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
            language_layers: None,
        })
    }

    pub fn with_language_layers(
        manifest: CollarManifest,
        language_layers: LanguageLayerStack,
    ) -> Result<Self, CollarError> {
        Self::with_shared_language_layers(manifest, Arc::new(Mutex::new(language_layers)))
    }

    /// Attach a request-local stack that may be handed through backend request objects before the
    /// mutation gate itself is compiled. The stack must not be reused by another inference.
    pub fn with_shared_language_layers(
        manifest: CollarManifest,
        language_layers: Arc<Mutex<LanguageLayerStack>>,
    ) -> Result<Self, CollarError> {
        let mut gate = Self::new(manifest)?;
        gate.language_layers = Some(language_layers);
        Ok(gate)
    }

    pub fn manifest(&self) -> &CollarManifest {
        &self.manifest
    }

    pub fn evaluate(&self, name: &str, arguments: &Value) -> CompletionDecision {
        let arguments = match self.arguments_with_bound_path(name, arguments) {
            Ok(arguments) => arguments,
            Err(code) => return CompletionDecision::Reject(code),
        };
        match name {
            "write_file" if self.manifest.mutation_policy.allow_write_file => {
                self.evaluate_write(&arguments, false)
            }
            "replace_file" if self.manifest.mutation_policy.allow_replace_file => {
                self.evaluate_write(&arguments, true)
            }
            "edit_file" if self.manifest.mutation_policy.allow_replace_file => {
                self.evaluate_edit(&arguments)
            }
            "apply_patch" if self.manifest.mutation_policy.allow_apply_patch => {
                self.evaluate_patch(&arguments)
            }
            _ => CompletionDecision::NotApplicable,
        }
    }

    /// Rebuild and replay a completed mutation through a fresh language-layer stack. This is an
    /// independent execution-time check, not a continuation of sampler state: the authoritative
    /// transaction is prepared again, unchanged bytes retain `Known` origin, deletions are
    /// explicit events, and every resulting file reaches `EndFile` before the call can execute.
    pub fn evaluate_independent(&self, name: &str, arguments: &Value) -> CompletionDecision {
        if self.language_layers.is_none() {
            return self.evaluate(name, arguments);
        }
        let arguments = match self.arguments_with_bound_path(name, arguments) {
            Ok(arguments) => arguments,
            Err(code) => return CompletionDecision::Reject(code),
        };
        let virtual_files = match name {
            "write_file" if self.manifest.mutation_policy.allow_write_file => {
                self.prepare_write_replay(&arguments, false)
            }
            "replace_file" if self.manifest.mutation_policy.allow_replace_file => {
                self.prepare_write_replay(&arguments, true)
            }
            "edit_file" if self.manifest.mutation_policy.allow_replace_file => {
                self.prepare_edit_replay(&arguments)
            }
            "apply_patch" if self.manifest.mutation_policy.allow_apply_patch => {
                self.prepare_patch_replay(&arguments)
            }
            _ => return CompletionDecision::NotApplicable,
        };
        let virtual_files = match virtual_files {
            Ok(files) => files,
            Err(error) => return CompletionDecision::Reject(classify_mutation_error(&error)),
        };
        match self.replay_virtual_files(&virtual_files) {
            Ok(analysis)
                if analysis.viability != Viability::Impossible
                    && analysis.closure != crate::analysis::ClosureVerdict::Reject =>
            {
                CompletionDecision::Accept
            }
            Ok(_) | Err(_) => CompletionDecision::Reject(RejectionCode::InvalidSemantics),
        }
    }

    fn prepare_write_replay(
        &self,
        arguments: &Value,
        replace: bool,
    ) -> Result<Vec<PatchVirtualFile>, CollarError> {
        let path = required_logical_path(arguments)?;
        let content = required_string(arguments, "content")?;
        if content.len() > self.manifest.limits.max_argument_bytes {
            return Err(CollarError::Mutation(format!(
                "file stream exceeds the {}-byte limit",
                self.manifest.limits.max_argument_bytes
            )));
        }
        let prepared = if replace {
            prepare_replace(
                &self.manifest.workspace,
                path.clone(),
                content.as_bytes().to_vec(),
            )?
        } else {
            crate::mutation::prepare_create(
                &self.manifest.workspace,
                path.clone(),
                content.as_bytes().to_vec(),
            )?
        };
        let result = prepared.result_bytes().ok_or_else(|| {
            CollarError::Mutation("file replay unexpectedly produced a deletion".to_string())
        })?;
        let base = replace
            .then(|| {
                self.manifest
                    .workspace
                    .get(&path)
                    .map(|entry| entry.bytes.as_slice())
            })
            .flatten();
        Ok(vec![replacement_virtual_file(path, base, result)])
    }

    fn prepare_edit_replay(&self, arguments: &Value) -> Result<Vec<PatchVirtualFile>, CollarError> {
        let path = required_logical_path(arguments)?;
        let old_text = required_string(arguments, "old_text")?;
        let new_text = required_string(arguments, "new_text")?;
        if old_text.is_empty()
            || old_text.len().saturating_add(new_text.len())
                > self.manifest.limits.max_argument_bytes
        {
            return Err(CollarError::Mutation(
                "edit arguments are empty or exceed the argument limit".to_string(),
            ));
        }
        let base = self.manifest.workspace.get(&path).ok_or_else(|| {
            CollarError::Mutation(format!(
                "replace target {:?} is missing from the snapshot",
                path.as_str()
            ))
        })?;
        let base_text = std::str::from_utf8(&base.bytes)
            .map_err(|_| CollarError::Mutation("edit target is not valid UTF-8".to_string()))?;
        if base_text.matches(old_text).take(2).count() != 1 {
            return Err(CollarError::Mutation(
                "edit old_text must match exactly once".to_string(),
            ));
        }
        let result = base_text.replacen(old_text, new_text, 1).into_bytes();
        let prepared = prepare_replace(&self.manifest.workspace, path.clone(), result)?;
        Ok(vec![replacement_virtual_file(
            path,
            Some(&base.bytes),
            prepared.result_bytes().ok_or_else(|| {
                CollarError::Mutation("edit replay unexpectedly produced a deletion".to_string())
            })?,
        )])
    }

    fn prepare_patch_replay(
        &self,
        arguments: &Value,
    ) -> Result<Vec<PatchVirtualFile>, CollarError> {
        let patch = required_string(arguments, "patch")?;
        if patch.len() > self.manifest.limits.max_argument_bytes {
            return Err(CollarError::Mutation(format!(
                "patch stream exceeds the {}-byte limit",
                self.manifest.limits.max_argument_bytes
            )));
        }
        let mut stream = PatchStream::new(
            self.manifest.workspace.clone(),
            self.manifest.limits.max_argument_bytes,
            self.manifest.limits.max_files,
            self.manifest.limits.max_patch_hunks,
        )?;
        stream.push(patch.as_bytes())?;
        let (_, files) = stream.finish_with_virtual_files()?;
        Ok(files)
    }

    fn replay_virtual_files(&self, files: &[PatchVirtualFile]) -> Result<Analysis, CollarError> {
        let mut layers = self
            .language_layers
            .as_ref()
            .expect("checked above")
            .lock()
            .map_err(|_| {
                CollarError::Analysis("language-layer stack lock is poisoned".to_string())
            })?;
        let mut analyses = Vec::new();
        for file in files {
            let Some(profile) = SyntaxProfile::for_path(&file.path) else {
                continue;
            };
            let language = profile.language_id();
            analyses.push(layers.apply(SourceEvent::BeginFile {
                path: &file.path,
                language: &language,
            })?);
            let mut cursor = 0usize;
            for deletion in &file.deletions {
                replay_result_range(
                    file,
                    cursor,
                    deletion.result_offset,
                    &mut layers,
                    &mut analyses,
                )?;
                analyses.push(layers.apply(SourceEvent::DeleteKnownBytes(&deletion.bytes))?);
                cursor = deletion.result_offset;
            }
            replay_result_range(file, cursor, file.len(), &mut layers, &mut analyses)?;
            analyses.push(layers.apply(SourceEvent::EndFile)?);
            if analyses.last().is_some_and(|analysis| {
                analysis.viability == Viability::Impossible
                    || analysis.closure == crate::analysis::ClosureVerdict::Reject
            }) {
                return Ok(Analysis::compose(analyses));
            }
        }
        analyses.push(layers.finalize()?);
        Ok(Analysis::compose(analyses))
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
        let arguments = self.prefix_arguments_with_bound_path(name, arguments);
        match name {
            "write_file" if self.manifest.mutation_policy.allow_write_file => {
                self.evaluate_file_prefix(&arguments, payload_prefix, false)
            }
            "replace_file" if self.manifest.mutation_policy.allow_replace_file => {
                self.evaluate_file_prefix(&arguments, payload_prefix, true)
            }
            "edit_file" if self.manifest.mutation_policy.allow_replace_file => {
                self.evaluate_edit_prefix(&arguments, payload_prefix)
            }
            "apply_patch" if self.manifest.mutation_policy.allow_apply_patch => {
                let result = (|| {
                    let mut cache = self.patch_cache.lock().map_err(|_| {
                        CollarError::Mutation("patch prefix cache lock is poisoned".to_string())
                    })?;
                    let mut layers = self
                        .language_layers
                        .as_ref()
                        .map(|layers| {
                            layers.lock().map_err(|_| {
                                CollarError::Analysis(
                                    "language-layer stack lock is poisoned".to_string(),
                                )
                            })
                        })
                        .transpose()?;
                    cache.probe(
                        &self.manifest,
                        payload_prefix.as_bytes(),
                        MAX_PREFIX_CACHE_CHECKPOINTS,
                        layers.as_deref_mut(),
                    )
                })();
                match result {
                    Ok(Some(analysis)) if analysis.viability == Viability::Impossible => {
                        CompletionDecision::Reject(RejectionCode::InvalidSemantics)
                    }
                    Ok(_) => CompletionDecision::Accept,
                    Err(error) => CompletionDecision::Reject(classify_mutation_error(&error)),
                }
            }
            _ => CompletionDecision::NotApplicable,
        }
    }

    fn arguments_with_bound_path<'a>(
        &self,
        name: &str,
        arguments: &'a Value,
    ) -> Result<Cow<'a, Value>, RejectionCode> {
        if !matches!(name, "write_file" | "replace_file" | "edit_file") {
            return Ok(Cow::Borrowed(arguments));
        }
        let Some(path) = self.manifest.workspace.bound_mutation_path() else {
            return Ok(Cow::Borrowed(arguments));
        };
        let Some(object) = arguments.as_object() else {
            return Err(RejectionCode::InvalidArguments);
        };
        let mut object = object.clone();
        object.insert("path".to_string(), Value::String(path.as_str().to_string()));
        Ok(Cow::Owned(Value::Object(object)))
    }

    fn prefix_arguments_with_bound_path<'a>(
        &self,
        name: &str,
        arguments: &'a serde_json::Map<String, Value>,
    ) -> Cow<'a, serde_json::Map<String, Value>> {
        if !matches!(name, "write_file" | "replace_file" | "edit_file") {
            return Cow::Borrowed(arguments);
        }
        let Some(path) = self.manifest.workspace.bound_mutation_path() else {
            return Cow::Borrowed(arguments);
        };
        let mut arguments = arguments.clone();
        arguments.insert("path".to_string(), Value::String(path.as_str().to_string()));
        Cow::Owned(arguments)
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
            Ok(_) => self.language_closure_decision(),
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
            Ok(_) => self.language_closure_decision(),
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
            Ok(_) => self.language_closure_decision(),
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
        let report = (|| {
            let mut cache = self.prefix_cache.lock().map_err(|_| {
                CollarError::Analysis("source prefix cache lock is poisoned".to_string())
            })?;
            let mut layers = self
                .language_layers
                .as_ref()
                .map(|layers| {
                    layers.lock().map_err(|_| {
                        CollarError::Analysis("language-layer stack lock is poisoned".to_string())
                    })
                })
                .transpose()?;
            cache.probe(
                identity,
                known_prefix,
                generated_prefix,
                layers.as_deref_mut(),
            )
        })();
        match report {
            Ok(report) if report.prefix.viability == Viability::Impossible => {
                CompletionDecision::Reject(RejectionCode::InvalidPrefix)
            }
            Ok(report)
                if report
                    .semantic
                    .as_ref()
                    .is_some_and(|analysis| analysis.viability == Viability::Impossible) =>
            {
                CompletionDecision::Reject(RejectionCode::InvalidSemantics)
            }
            Ok(report) if report.prefix.profile.is_some() || report.semantic.is_some() => {
                CompletionDecision::Accept
            }
            Ok(_) => CompletionDecision::NotApplicable,
            Err(_) => CompletionDecision::Reject(RejectionCode::InvalidPrefix),
        }
    }

    fn language_closure_decision(&self) -> CompletionDecision {
        let Some(layers) = self.language_layers.as_ref() else {
            return CompletionDecision::Accept;
        };
        let analysis = layers
            .lock()
            .map_err(|_| ())
            .and_then(|mut layers| layers.finalize().map_err(|_| ()));
        match analysis {
            Ok(analysis)
                if analysis.viability == Viability::Impossible
                    || analysis.closure == crate::analysis::ClosureVerdict::Reject =>
            {
                CompletionDecision::Reject(RejectionCode::InvalidSemantics)
            }
            Ok(_) => CompletionDecision::Accept,
            Err(()) => CompletionDecision::Reject(RejectionCode::InvalidSemantics),
        }
    }
}

fn required_logical_path(arguments: &Value) -> Result<LogicalPath, CollarError> {
    LogicalPath::parse(required_string(arguments, "path")?.to_string())
}

fn required_string<'a>(arguments: &'a Value, field: &str) -> Result<&'a str, CollarError> {
    arguments
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| CollarError::Mutation(format!("missing string argument {field:?}")))
}

fn replacement_virtual_file(
    path: LogicalPath,
    base: Option<&[u8]>,
    result: &[u8],
) -> PatchVirtualFile {
    let Some(base) = base else {
        return PatchVirtualFile {
            path,
            segments: vec![crate::mutation::PatchVirtualSegment {
                origin: SourceOrigin::Generated,
                bytes: result.to_vec(),
            }],
            deletions: Vec::new(),
            complete: true,
        };
    };
    let prefix = base
        .iter()
        .zip(result)
        .take_while(|(left, right)| left == right)
        .count();
    let max_suffix = base.len().min(result.len()).saturating_sub(prefix);
    let suffix = base
        .iter()
        .rev()
        .zip(result.iter().rev())
        .take(max_suffix)
        .take_while(|(left, right)| left == right)
        .count();
    let base_middle_end = base.len().saturating_sub(suffix);
    let result_middle_end = result.len().saturating_sub(suffix);
    let mut segments = Vec::new();
    push_replay_segment(&mut segments, SourceOrigin::Known, &result[..prefix]);
    push_replay_segment(
        &mut segments,
        SourceOrigin::Generated,
        &result[prefix..result_middle_end],
    );
    push_replay_segment(
        &mut segments,
        SourceOrigin::Known,
        &result[result_middle_end..],
    );
    let deletions = (base_middle_end > prefix)
        .then_some(crate::mutation::PatchVirtualDeletion {
            result_offset: prefix,
            bytes: base[prefix..base_middle_end].to_vec(),
        })
        .into_iter()
        .collect();
    PatchVirtualFile {
        path,
        segments,
        deletions,
        complete: true,
    }
}

fn push_replay_segment(
    segments: &mut Vec<crate::mutation::PatchVirtualSegment>,
    origin: SourceOrigin,
    bytes: &[u8],
) {
    if !bytes.is_empty() {
        segments.push(crate::mutation::PatchVirtualSegment {
            origin,
            bytes: bytes.to_vec(),
        });
    }
}

fn replay_result_range(
    file: &PatchVirtualFile,
    start: usize,
    end: usize,
    layers: &mut LanguageLayerStack,
    analyses: &mut Vec<Analysis>,
) -> Result<(), CollarError> {
    if start > end || end > file.len() {
        return Err(CollarError::Analysis(
            "independent replay byte range is invalid".to_string(),
        ));
    }
    let mut offset = 0usize;
    for segment in &file.segments {
        let segment_end = offset.saturating_add(segment.bytes.len());
        let range_start = start.max(offset);
        let range_end = end.min(segment_end);
        if range_start < range_end {
            analyses.push(layers.apply(SourceEvent::Bytes {
                origin: segment.origin,
                bytes: &segment.bytes[range_start - offset..range_end - offset],
            })?);
        }
        offset = segment_end;
        if offset >= end {
            break;
        }
    }
    if offset < end {
        return Err(CollarError::Analysis(
            "independent replay segments do not cover the result".to_string(),
        ));
    }
    Ok(())
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
    checkpoints: Vec<PrefixProbeCheckpoint>,
}

#[derive(Clone, Debug)]
struct PrefixProbeCheckpoint {
    payload_len: usize,
    prefix: PrefixCheckpoint,
    layers: Option<LayerStackCheckpoint>,
}

struct CombinedPrefixReport {
    prefix: PrefixReport,
    semantic: Option<Analysis>,
}

impl PrefixProbeCache {
    fn probe(
        &mut self,
        identity: PrefixProbeIdentity,
        known_prefix: &[u8],
        generated_prefix: &[u8],
        mut layers: Option<&mut LanguageLayerStack>,
    ) -> Result<CombinedPrefixReport, CollarError> {
        if self
            .entry
            .as_ref()
            .is_none_or(|entry| entry.identity != identity)
        {
            let mut oracle = SourcePrefixOracle::new(identity.path.clone(), identity.source_limit)?;
            let known_report = oracle.push(known_prefix)?;
            let layer_checkpoint = if let (Some(layers), Some(profile)) = (
                layers.as_deref_mut(),
                SyntaxProfile::for_path(&identity.path),
            ) {
                let language = profile.language_id();
                layers.apply(SourceEvent::BeginFile {
                    path: &identity.path,
                    language: &language,
                })?;
                if !known_prefix.is_empty() {
                    layers.apply(SourceEvent::Bytes {
                        origin: SourceOrigin::Known,
                        bytes: known_prefix,
                    })?;
                }
                if let Some(boundary) = known_report.boundary {
                    layers.apply(SourceEvent::Boundary(boundary))?;
                }
                Some(layers.checkpoint()?)
            } else {
                None
            };
            let prefix_checkpoint = oracle.checkpoint();
            self.entry = Some(CachedPrefixProbe {
                identity,
                oracle,
                generated_prefix: Vec::new(),
                checkpoints: vec![PrefixProbeCheckpoint {
                    payload_len: 0,
                    prefix: prefix_checkpoint,
                    layers: layer_checkpoint,
                }],
            });
        }
        let entry = self.entry.as_mut().expect("prefix cache initialized above");
        let common = common_prefix_len(&entry.generated_prefix, generated_prefix);
        entry
            .checkpoints
            .retain(|checkpoint| checkpoint.payload_len <= common);
        let checkpoint = entry
            .checkpoints
            .last()
            .cloned()
            .expect("prefix cache always retains its zero checkpoint");
        entry.oracle.rollback(checkpoint.prefix)?;
        if let (Some(layers), Some(layer_checkpoint)) =
            (layers.as_deref_mut(), checkpoint.layers.clone())
        {
            layers.rollback(layer_checkpoint)?;
        }
        let delta = &generated_prefix[checkpoint.payload_len..];
        let report = entry.oracle.push(delta)?;
        let semantic = if let Some(layers) = layers.as_deref_mut() {
            let mut analysis = if delta.is_empty() {
                None
            } else {
                Some(layers.apply(SourceEvent::Bytes {
                    origin: SourceOrigin::Generated,
                    bytes: delta,
                })?)
            };
            if let Some(boundary) = report.boundary {
                analysis = Some(layers.apply(SourceEvent::Boundary(boundary))?);
            }
            analysis
        } else {
            None
        };
        entry.generated_prefix.clear();
        entry.generated_prefix.extend_from_slice(generated_prefix);
        if entry
            .checkpoints
            .last()
            .is_none_or(|checkpoint| checkpoint.payload_len != generated_prefix.len())
        {
            entry.checkpoints.push(PrefixProbeCheckpoint {
                payload_len: generated_prefix.len(),
                prefix: entry.oracle.checkpoint(),
                layers: layers
                    .as_deref_mut()
                    .map(LanguageLayerStack::checkpoint)
                    .transpose()?,
            });
        }
        if entry.checkpoints.len() > MAX_PREFIX_CACHE_CHECKPOINTS {
            let remove = entry
                .checkpoints
                .len()
                .saturating_sub(MAX_PREFIX_CACHE_CHECKPOINTS);
            entry.checkpoints.drain(1..=remove);
        }
        Ok(CombinedPrefixReport {
            prefix: report,
            semantic,
        })
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
    mirror: PatchLayerMirror,
    checkpoints: Vec<PatchProbeCheckpoint>,
}

#[derive(Clone, Debug)]
struct PatchProbeCheckpoint {
    payload_len: usize,
    patch: PatchCheckpoint,
    mirror: PatchLayerMirror,
    layers: Option<LayerStackCheckpoint>,
}

#[derive(Clone, Debug, Default)]
struct PatchLayerMirror {
    files: Vec<PatchLayerFileMirror>,
}

#[derive(Clone, Debug)]
struct PatchLayerFileMirror {
    path: LogicalPath,
    bytes_len: usize,
    deletions_len: usize,
    oracle: Option<SourcePrefixOracle>,
    ended: bool,
}

impl PatchProbeCache {
    fn probe(
        &mut self,
        manifest: &CollarManifest,
        generated_prefix: &[u8],
        max_checkpoints: usize,
        mut layers: Option<&mut LanguageLayerStack>,
    ) -> Result<Option<Analysis>, CollarError> {
        if self.entry.is_none() {
            let stream = PatchStream::new(
                manifest.workspace.clone(),
                manifest.limits.max_argument_bytes,
                manifest.limits.max_files,
                manifest.limits.max_patch_hunks,
            )?;
            let checkpoint = stream.checkpoint();
            let layer_checkpoint = layers
                .as_deref_mut()
                .map(LanguageLayerStack::checkpoint)
                .transpose()?;
            self.entry = Some(CachedPatchProbe {
                stream,
                generated_prefix: Vec::new(),
                mirror: PatchLayerMirror::default(),
                checkpoints: vec![PatchProbeCheckpoint {
                    payload_len: 0,
                    patch: checkpoint,
                    mirror: PatchLayerMirror::default(),
                    layers: layer_checkpoint,
                }],
            });
        }
        let entry = self.entry.as_mut().expect("patch cache initialized above");
        let common = common_prefix_len(&entry.generated_prefix, generated_prefix);
        entry
            .checkpoints
            .retain(|checkpoint| checkpoint.payload_len <= common);
        let checkpoint = entry
            .checkpoints
            .last()
            .cloned()
            .expect("patch cache always retains its zero checkpoint");
        entry.stream.rollback(checkpoint.patch.clone())?;
        if let (Some(layers), Some(layer_checkpoint)) =
            (layers.as_deref_mut(), checkpoint.layers.clone())
        {
            layers.rollback(layer_checkpoint)?;
        }
        entry.mirror = checkpoint.mirror.clone();
        if let Err(error) = entry
            .stream
            .push(&generated_prefix[checkpoint.payload_len..])
        {
            entry.stream.rollback(checkpoint.patch)?;
            return Err(error);
        }
        let virtual_files = entry.stream.virtual_files()?;
        let semantic = layers
            .as_deref_mut()
            .map(|layers| {
                sync_patch_language_layers(
                    &mut entry.mirror,
                    &virtual_files,
                    layers,
                    manifest
                        .limits
                        .max_snapshot_bytes
                        .saturating_add(manifest.limits.max_argument_bytes)
                        .max(1),
                )
            })
            .transpose()?;
        entry.generated_prefix.clear();
        entry.generated_prefix.extend_from_slice(generated_prefix);
        if entry
            .checkpoints
            .last()
            .is_none_or(|checkpoint| checkpoint.payload_len != generated_prefix.len())
        {
            let layer_checkpoint = layers
                .as_deref_mut()
                .map(LanguageLayerStack::checkpoint)
                .transpose()?;
            entry.checkpoints.push(PatchProbeCheckpoint {
                payload_len: generated_prefix.len(),
                patch: entry.stream.checkpoint(),
                mirror: entry.mirror.clone(),
                layers: layer_checkpoint,
            });
        }
        if entry.checkpoints.len() > max_checkpoints {
            let remove = entry.checkpoints.len().saturating_sub(max_checkpoints);
            entry.checkpoints.drain(1..=remove);
        }
        Ok(semantic)
    }
}

fn sync_patch_language_layers(
    mirror: &mut PatchLayerMirror,
    files: &[PatchVirtualFile],
    layers: &mut LanguageLayerStack,
    source_limit: usize,
) -> Result<Analysis, CollarError> {
    if files.len() < mirror.files.len() {
        return Err(CollarError::Analysis(
            "virtual patch file sequence moved backwards without rollback".to_string(),
        ));
    }
    let mut analyses = Vec::new();
    for (index, file) in files.iter().enumerate() {
        if index == mirror.files.len() {
            let oracle = SyntaxProfile::for_path(&file.path)
                .map(|profile| {
                    let language = profile.language_id();
                    layers.apply(SourceEvent::BeginFile {
                        path: &file.path,
                        language: &language,
                    })?;
                    SourcePrefixOracle::new(file.path.clone(), source_limit)
                })
                .transpose()?;
            mirror.files.push(PatchLayerFileMirror {
                path: file.path.clone(),
                bytes_len: 0,
                deletions_len: 0,
                oracle,
                ended: false,
            });
        }
        let state = mirror
            .files
            .get_mut(index)
            .ok_or_else(|| CollarError::Analysis("virtual patch mirror lost a file".to_string()))?;
        if state.path != file.path
            || state.ended
                && (file.len() != state.bytes_len || file.deletions.len() != state.deletions_len)
        {
            return Err(CollarError::Analysis(
                "virtual patch file identity changed without rollback".to_string(),
            ));
        }
        if file.len() < state.bytes_len || file.deletions.len() < state.deletions_len {
            return Err(CollarError::Analysis(
                "virtual patch result prefix shrank without rollback".to_string(),
            ));
        }
        if let Some(oracle) = state.oracle.as_mut() {
            let mut cursor = state.bytes_len;
            for deletion in &file.deletions[state.deletions_len..] {
                if deletion.result_offset < cursor || deletion.result_offset > file.len() {
                    return Err(CollarError::Analysis(
                        "virtual patch deletion moved behind the mirrored result".to_string(),
                    ));
                }
                push_patch_result_range(
                    file,
                    cursor,
                    deletion.result_offset,
                    layers,
                    oracle,
                    &mut analyses,
                )?;
                analyses.push(layers.apply(SourceEvent::DeleteKnownBytes(&deletion.bytes))?);
                cursor = deletion.result_offset;
            }
            push_patch_result_range(file, cursor, file.len(), layers, oracle, &mut analyses)?;
            if file.complete && !state.ended {
                analyses.push(layers.apply(SourceEvent::EndFile)?);
                state.ended = true;
            }
        }
        state.bytes_len = file.len();
        state.deletions_len = file.deletions.len();
    }
    Ok(Analysis::compose(analyses))
}

fn push_patch_result_range(
    file: &PatchVirtualFile,
    start: usize,
    end: usize,
    layers: &mut LanguageLayerStack,
    oracle: &mut SourcePrefixOracle,
    analyses: &mut Vec<Analysis>,
) -> Result<(), CollarError> {
    if start > end || end > file.len() {
        return Err(CollarError::Analysis(
            "virtual patch byte range is invalid".to_string(),
        ));
    }
    let mut offset = 0usize;
    for segment in &file.segments {
        let segment_end = offset.saturating_add(segment.bytes.len());
        let range_start = start.max(offset);
        let range_end = end.min(segment_end);
        if range_start < range_end {
            let delta = &segment.bytes[range_start - offset..range_end - offset];
            analyses.push(layers.apply(SourceEvent::Bytes {
                origin: segment.origin,
                bytes: delta,
            })?);
            let report = oracle.push(delta)?;
            if let Some(boundary) = report.boundary {
                analyses.push(layers.apply(SourceEvent::Boundary(boundary))?);
            }
        }
        offset = segment_end;
        if offset >= end {
            break;
        }
    }
    if offset < end {
        return Err(CollarError::Analysis(
            "virtual patch segments do not cover the requested byte range".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        CollarResult,
        analysis::{
            AnalyzerCapability, AnalyzerCheckpoint, AnalyzerLayerDescriptor, ClosureVerdict,
            IncrementalAnalyzer, LanguageId, LayerReadiness, LayerReadinessReceipt,
            ProgramSnapshot, ReadinessOrigin, SemanticCompleteness, SemanticWorldId, SourceEvent,
        },
        mutation::{SnapshotEntry, WorkspaceSnapshot},
        protocol::ToolDialect,
        tool::{CollarLimits, ExposedTool, MutationPolicy, ToolConstraintMode},
    };

    fn manifest(entries: Vec<SnapshotEntry>) -> CollarManifest {
        CollarManifest {
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
        }
    }

    fn gate(entries: Vec<SnapshotEntry>) -> MutationCompletionGate {
        MutationCompletionGate::new(manifest(entries)).unwrap()
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
    fn controller_bound_path_is_injected_for_pathless_mutations() {
        let path = LogicalPath::parse("src/lib.rs").unwrap();
        let entry = SnapshotEntry::new(path.clone(), b"pub fn before() {}\n".to_vec());
        let mut manifest = manifest(vec![entry]);
        manifest.workspace = manifest.workspace.with_bound_mutation_path(path);
        let bound_gate = MutationCompletionGate::new(manifest).unwrap();

        assert_eq!(
            bound_gate.evaluate("replace_file", &json!({"content":"pub fn after() {}\n"})),
            CompletionDecision::Accept
        );
        assert_eq!(
            bound_gate.evaluate(
                "replace_file",
                &json!({
                    "path":"model/supplied/alternate.rs",
                    "content":"pub fn after() {}\n"
                })
            ),
            CompletionDecision::Accept
        );
        assert_eq!(
            bound_gate.evaluate("replace_file", &json!({"content":"pub fn broken( {"})),
            CompletionDecision::Reject(RejectionCode::InvalidSyntax)
        );
        assert_eq!(
            bound_gate.evaluate_prefix(
                "replace_file",
                &serde_json::Map::new(),
                "pub fn after() { ]"
            ),
            CompletionDecision::Reject(RejectionCode::InvalidPrefix)
        );

        let unbound = gate(vec![SnapshotEntry::new(
            LogicalPath::parse("src/lib.rs").unwrap(),
            b"pub fn before() {}\n".to_vec(),
        )]);
        assert_eq!(
            unbound.evaluate("replace_file", &json!({"content":"pub fn after() {}\n"})),
            CompletionDecision::Reject(RejectionCode::InvalidArguments)
        );
        assert_eq!(
            unbound.evaluate_prefix("replace_file", &serde_json::Map::new(), "pub fn after() {}"),
            CompletionDecision::NotApplicable
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
    fn language_layer_rejection_composes_with_prefix_cache_rollback() {
        let layer = MockRustLayer::new();
        let stack =
            LanguageLayerStack::new(vec![Box::new(layer)], ProgramSnapshot::default()).unwrap();
        let gate =
            MutationCompletionGate::with_language_layers(manifest(Vec::new()), stack).unwrap();
        let arguments = json!({"path":"src/lib.rs"}).as_object().unwrap().clone();
        let common = "use std::collections::";
        assert_eq!(
            gate.evaluate_prefix(
                "write_file",
                &arguments,
                &format!("{common}DefinitelyMissing;")
            ),
            CompletionDecision::Reject(RejectionCode::InvalidSemantics)
        );
        assert_eq!(
            gate.evaluate_prefix("write_file", &arguments, &format!("{common}BTreeMap;")),
            CompletionDecision::Accept
        );
    }

    #[test]
    fn independent_completion_replays_a_fresh_semantic_stack() {
        let stack = LanguageLayerStack::new(
            vec![Box::new(MockRustLayer::new())],
            ProgramSnapshot::default(),
        )
        .unwrap();
        let gate =
            MutationCompletionGate::with_language_layers(manifest(Vec::new()), stack).unwrap();
        assert_eq!(
            gate.evaluate_independent(
                "write_file",
                &json!({
                    "path":"src/lib.rs",
                    "content":"use std::collections::DefinitelyMissing;\n"
                })
            ),
            CompletionDecision::Reject(RejectionCode::InvalidSemantics)
        );

        let stack = LanguageLayerStack::new(
            vec![Box::new(MockRustLayer::new())],
            ProgramSnapshot::default(),
        )
        .unwrap();
        let gate =
            MutationCompletionGate::with_language_layers(manifest(Vec::new()), stack).unwrap();
        assert_eq!(
            gate.evaluate_independent(
                "write_file",
                &json!({
                    "path":"src/lib.rs",
                    "content":"use std::collections::BTreeMap;\n"
                })
            ),
            CompletionDecision::Accept
        );
    }

    #[test]
    fn patch_virtual_stream_reuses_language_layer_and_rolls_back_candidate_branches() {
        let path = LogicalPath::parse("src/lib.rs").unwrap();
        let entries = vec![SnapshotEntry::new(
            path,
            b"use std::collections::BTreeMap;\n".to_vec(),
        )];
        let layer = MockRustLayer::new();
        let stack =
            LanguageLayerStack::new(vec![Box::new(layer)], ProgramSnapshot::default()).unwrap();
        let gate = MutationCompletionGate::with_language_layers(manifest(entries), stack).unwrap();
        let arguments = serde_json::Map::new();
        let common = concat!(
            "--- a/src/lib.rs\n",
            "+++ b/src/lib.rs\n",
            "@@ -1,1 +1,1 @@\n",
            "-use std::collections::BTreeMap;\n",
            "+use std::collections::",
        );
        let invalid = format!("{common}DefinitelyMissing;");
        assert_eq!(
            gate.evaluate_prefix("apply_patch", &arguments, &invalid),
            CompletionDecision::Reject(RejectionCode::InvalidSemantics)
        );
        assert_eq!(
            gate.evaluate("apply_patch", &json!({"patch": format!("{invalid}\n")})),
            CompletionDecision::Reject(RejectionCode::InvalidSemantics)
        );
        assert_eq!(
            gate.evaluate_prefix("apply_patch", &arguments, &format!("{common}BTreeMap;")),
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

    struct MockRustLayer {
        descriptor: AnalyzerLayerDescriptor,
        receipt: LayerReadinessReceipt,
        source: Vec<u8>,
        snapshots: Vec<Vec<u8>>,
        epoch: u64,
    }

    impl MockRustLayer {
        fn new() -> Self {
            let world = SemanticWorldId {
                provider: "mock-rust".to_string(),
                provider_version: "v1".to_string(),
                world_sha256: "a".repeat(64),
                configuration_sha256: "b".repeat(64),
                dependency_sha256: "c".repeat(64),
            };
            Self {
                descriptor: AnalyzerLayerDescriptor {
                    id: "mock-rust".to_string(),
                    language: LanguageId("rust".to_string()),
                    world: world.clone(),
                    capabilities: vec![AnalyzerCapability::SymbolResolution],
                },
                receipt: LayerReadinessReceipt {
                    world,
                    origin: ReadinessOrigin::WarmCache,
                    completeness: SemanticCompleteness::Complete,
                    load_millis: 0,
                    prime_millis: 0,
                    primed_queries: 1,
                },
                source: Vec::new(),
                snapshots: Vec::new(),
                epoch: 0,
            }
        }

        fn verdict(&self, boundary: bool) -> Analysis {
            let rejected = boundary
                && self
                    .source
                    .windows(b"DefinitelyMissing;".len())
                    .any(|window| window == b"DefinitelyMissing;");
            Analysis {
                viability: if rejected {
                    Viability::Impossible
                } else if boundary {
                    Viability::Valid
                } else {
                    Viability::Repairable
                },
                closure: if rejected {
                    ClosureVerdict::Reject
                } else if boundary {
                    ClosureVerdict::Allow
                } else {
                    ClosureVerdict::Defer
                },
                obligations: Vec::new(),
                biases: Vec::new(),
            }
        }
    }

    impl IncrementalAnalyzer for MockRustLayer {
        fn descriptor(&self) -> &AnalyzerLayerDescriptor {
            &self.descriptor
        }

        fn readiness(&self) -> LayerReadiness {
            LayerReadiness::Ready
        }

        fn readiness_receipt(&self) -> Option<&LayerReadinessReceipt> {
            Some(&self.receipt)
        }

        fn begin(&mut self, _snapshot: ProgramSnapshot) -> CollarResult<()> {
            self.source.clear();
            self.snapshots.clear();
            self.epoch = self.epoch.wrapping_add(1);
            Ok(())
        }

        fn checkpoint(&mut self) -> CollarResult<AnalyzerCheckpoint> {
            self.snapshots.push(self.source.clone());
            Ok(AnalyzerCheckpoint {
                epoch: self.epoch,
                revision: u64::try_from(self.snapshots.len().saturating_sub(1)).unwrap(),
            })
        }

        fn apply(&mut self, event: SourceEvent<'_>) -> CollarResult<Analysis> {
            match event {
                SourceEvent::BeginFile { .. } => {
                    self.source.clear();
                    Ok(self.verdict(false))
                }
                SourceEvent::Bytes { bytes, .. } => {
                    self.source.extend_from_slice(bytes);
                    Ok(self.verdict(false))
                }
                SourceEvent::Boundary(_) | SourceEvent::EndFile => Ok(self.verdict(true)),
                SourceEvent::DeleteKnownBytes(_) => Ok(self.verdict(false)),
            }
        }

        fn rollback(&mut self, checkpoint: AnalyzerCheckpoint) -> CollarResult<()> {
            if checkpoint.epoch != self.epoch {
                return Err(CollarError::Analysis("mock epoch mismatch".to_string()));
            }
            let revision = usize::try_from(checkpoint.revision).unwrap();
            self.source = self.snapshots[revision].clone();
            self.snapshots.truncate(revision.saturating_add(1));
            Ok(())
        }

        fn finalize(&mut self) -> CollarResult<Analysis> {
            Ok(self.verdict(true))
        }
    }
}
