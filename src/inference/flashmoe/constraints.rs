use std::collections::{BTreeMap, BTreeSet, VecDeque};

use anyhow::{Context, Result, bail};
use pb_control_collar::analysis::ClosureVerdict;
use pb_control_collar::protocol::ToolDialect;
use pb_control_collar::tool::{
    CollarLimits, CollarManifest, ExposedTool, MutationPolicy, ToolConstraintMode,
};
use pb_control_collar::{CompletionDecision, MutationCompletionGate, RejectionCode};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::text::QwenTokenizer;
use super::types::{ChatTool, NativeToolConstraintMode};

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const CONSTRAINED_NO_REPEAT_NGRAM: usize = 32;
const MAX_STRUCTURAL_WHITESPACE_BYTES: usize = 32;
pub(crate) const MAX_COLLAR_ARGUMENT_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_COLLAR_SNAPSHOT_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const MAX_COLLAR_FILES: usize = 32;
pub(crate) const MAX_COLLAR_PATCH_HUNKS: usize = 256;

#[cfg(test)]
pub(crate) fn mutation_completion_gate(
    dialect: ToolDialect,
    mode: NativeToolConstraintMode,
    tools: &[ChatTool],
    terminal_tool_names: &[String],
    snapshot: Option<&pb_control_collar::mutation::WorkspaceSnapshot>,
) -> Result<Option<MutationCompletionGate>> {
    mutation_completion_gate_with_layers(dialect, mode, tools, terminal_tool_names, snapshot, None)
}

pub(crate) fn mutation_completion_gate_with_layers(
    dialect: ToolDialect,
    mode: NativeToolConstraintMode,
    tools: &[ChatTool],
    terminal_tool_names: &[String],
    snapshot: Option<&pb_control_collar::mutation::WorkspaceSnapshot>,
    language_layers: Option<crate::control_layers::SharedLanguageLayers>,
) -> Result<Option<MutationCompletionGate>> {
    let has_mutation = tools.iter().any(|tool| {
        matches!(
            tool.name.as_str(),
            "write_file" | "replace_file" | "edit_file" | "apply_patch"
        )
    });
    if !has_mutation {
        return Ok(None);
    }
    let Some(snapshot) = snapshot else {
        // Prompt measurement and identity calculation are deliberately workspace-data-free. The
        // generation entry point separately requires the snapshot before any model work begins.
        return Ok(None);
    };
    let manifest = collar_manifest(dialect, mode, tools, terminal_tool_names, snapshot.clone())?;
    Ok(Some(match language_layers {
        Some(layers) => MutationCompletionGate::with_shared_language_layers(manifest, layers)?,
        None => MutationCompletionGate::new(manifest)?,
    }))
}

pub(crate) fn collar_manifest(
    dialect: ToolDialect,
    mode: NativeToolConstraintMode,
    tools: &[ChatTool],
    terminal_tool_names: &[String],
    workspace: pb_control_collar::mutation::WorkspaceSnapshot,
) -> Result<CollarManifest> {
    let exposed = |name: &str| tools.iter().any(|tool| tool.name == name);
    Ok(CollarManifest {
        contract_version: 1,
        dialect,
        mode: match mode {
            NativeToolConstraintMode::Auto => ToolConstraintMode::Auto,
            NativeToolConstraintMode::ToolsAllowed => ToolConstraintMode::ToolsAllowed,
            NativeToolConstraintMode::ToolRequired => ToolConstraintMode::ToolRequired,
        },
        tools: tools
            .iter()
            .map(|tool| ExposedTool {
                name: tool.name.clone(),
                input_schema: tool.input_schema.clone(),
            })
            .collect(),
        terminal_tools: terminal_tool_names.to_vec(),
        mutation_policy: MutationPolicy {
            allow_write_file: exposed("write_file"),
            allow_replace_file: exposed("replace_file") || exposed("edit_file"),
            allow_apply_patch: exposed("apply_patch"),
            max_mutation_calls_per_batch: 1,
        },
        workspace,
        limits: CollarLimits {
            max_argument_bytes: MAX_COLLAR_ARGUMENT_BYTES,
            max_snapshot_bytes: MAX_COLLAR_SNAPSHOT_BYTES,
            max_files: MAX_COLLAR_FILES,
            max_patch_hunks: MAX_COLLAR_PATCH_HUNKS,
        },
    })
}

#[derive(Debug, Clone)]
pub(crate) struct NativeToolConstraint {
    mode: NativeToolConstraintMode,
    schemas: BTreeMap<String, Value>,
    mutation_gate: Option<MutationCompletionGate>,
    semantic_provider: Option<crate::inference::SemanticBoundaryControl>,
    terminal_tool_names: BTreeSet<String>,
    forced_tokens: VecDeque<u32>,
    payload_limit_stop: Option<String>,
    stopped_at_payload_limit: bool,
    schema_sha256: String,
    rejected_candidates: usize,
    mutation_rejections: BTreeMap<pb_control_collar::RejectionCode, usize>,
}

impl NativeToolConstraint {
    #[cfg(test)]
    pub(super) fn compile(
        mode: NativeToolConstraintMode,
        tools: &[ChatTool],
    ) -> Result<Option<Self>> {
        Self::compile_with_terminal_tools(mode, tools, &[])
    }

    #[cfg(test)]
    pub(super) fn compile_with_terminal_tools(
        mode: NativeToolConstraintMode,
        tools: &[ChatTool],
        terminal_tool_names: &[String],
    ) -> Result<Option<Self>> {
        Self::compile_with_mutation_gate(mode, tools, terminal_tool_names, None)
    }

    pub(crate) fn compile_with_mutation_gate(
        mode: NativeToolConstraintMode,
        tools: &[ChatTool],
        terminal_tool_names: &[String],
        mutation_gate: Option<MutationCompletionGate>,
    ) -> Result<Option<Self>> {
        let active_mode = match mode {
            NativeToolConstraintMode::Auto if tools.is_empty() => return Ok(None),
            NativeToolConstraintMode::Auto => NativeToolConstraintMode::ToolsAllowed,
            mode => mode,
        };
        if tools.is_empty() {
            bail!("native tool constraint mode {active_mode:?} requires at least one tool");
        }
        let mut schemas = BTreeMap::new();
        for tool in tools {
            if tool.name.trim().is_empty() {
                bail!("native tool constraints require non-empty tool names");
            }
            validate_supported_schema(&tool.input_schema, &format!("tool {}", tool.name))?;
            if schemas
                .insert(tool.name.clone(), tool.input_schema.clone())
                .is_some()
            {
                bail!(
                    "native tool constraints received duplicate tool {}",
                    tool.name
                );
            }
        }
        let terminal_tool_names = terminal_tool_names
            .iter()
            .map(|name| {
                if !schemas.contains_key(name) {
                    bail!("native terminal tool constraint names unexposed tool {name}");
                }
                Ok(name.clone())
            })
            .collect::<Result<BTreeSet<_>>>()?;
        let schema_bytes = serde_json::to_vec(&schemas)?;
        Ok(Some(Self {
            mode: active_mode,
            schemas,
            mutation_gate,
            semantic_provider: None,
            terminal_tool_names,
            forced_tokens: VecDeque::new(),
            payload_limit_stop: None,
            stopped_at_payload_limit: false,
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            rejected_candidates: 0,
            mutation_rejections: BTreeMap::new(),
        }))
    }

    pub(super) fn mode(&self) -> NativeToolConstraintMode {
        self.mode
    }

    pub(crate) fn set_semantic_provider(
        &mut self,
        provider: Option<crate::inference::SemanticBoundaryControl>,
    ) {
        self.semantic_provider = provider;
    }

    pub(super) fn schema_sha256(&self) -> &str {
        &self.schema_sha256
    }

    pub(super) fn rejected_candidates(&self) -> usize {
        self.rejected_candidates
    }

    /// Backend-neutral transcript probe used by llama.cpp's native full-vocabulary adapter. The
    /// Qwen tokenizer path retains its faster candidate batching, while both backends share this
    /// exact protocol and mutation policy.
    pub(crate) fn probe_transcript(&mut self, decoded: &str, at_eos: bool) -> bool {
        if self.output_prefix_is_valid(decoded, at_eos) {
            true
        } else {
            self.rejected_candidates = self.rejected_candidates.saturating_add(1);
            if let Some(code) = self.output_mutation_rejection(decoded) {
                *self.mutation_rejections.entry(code).or_default() += 1;
            }
            false
        }
    }

    pub(crate) fn transcript_has_complete_terminal_call(&self, decoded: &str) -> bool {
        self.output_has_complete_terminal_call(decoded)
    }

    pub(crate) fn stats(
        &self,
        decoded: &str,
    ) -> crate::inference::flashmoe::NativeToolConstraintStats {
        let (snapshot_files, snapshot_bytes) = self.mutation_gate.as_ref().map_or((0, 0), |gate| {
            let workspace = &gate.manifest().workspace;
            (workspace.len(), workspace.total_bytes())
        });
        let semantic_authorized = self.semantic_closure_authorized(decoded);
        let semantic_boundary = self
            .semantic_provider
            .as_ref()
            .map(crate::inference::SemanticBoundaryControl::stats);
        crate::inference::flashmoe::NativeToolConstraintStats {
            mode: self.mode,
            dialect: "qwen_json".to_string(),
            schema_sha256: self.schema_sha256.clone(),
            rejected_candidates: self.rejected_candidates,
            mutation_rejections: self
                .mutation_rejections
                .iter()
                .map(|(code, count)| (code.as_str().to_string(), *count))
                .collect(),
            snapshot_files,
            snapshot_bytes,
            terminal_state: self.terminal_state(decoded).to_string(),
            guarantee_rung: if semantic_authorized {
                "semantic_boundary"
            } else if self.mutation_gate.is_some() {
                "prefix_syntax"
            } else {
                "protocol_schema"
            }
            .to_string(),
            semantic_boundary,
            decode_recovery: crate::inference::DecodeRecovery::CandidateProbeOnly,
        }
    }

    fn semantic_closure_authorized(&self, decoded: &str) -> bool {
        let (Some(provider), Some(open)) = (
            self.semantic_provider.as_ref(),
            decoded.rfind(TOOL_CALL_OPEN),
        ) else {
            return false;
        };
        let body = &decoded[open + TOOL_CALL_OPEN.len()..];
        let Some(close) = body.find(TOOL_CALL_CLOSE) else {
            return false;
        };
        let Ok(value) = serde_json::from_str::<Value>(body[..close].trim()) else {
            return false;
        };
        let (Some(name), Some(arguments)) = (
            value.get("name").and_then(Value::as_str),
            value.get("arguments"),
        ) else {
            return false;
        };
        provider.probe(name, arguments).closure == ClosureVerdict::Allow
    }

    pub(super) fn terminal_state(&self, decoded: &str) -> &'static str {
        if self.stopped_at_payload_limit {
            "mutation_payload_limit"
        } else if self.output_has_complete_terminal_call(decoded) {
            "complete_terminal_tool_call"
        } else if decoded.contains(TOOL_CALL_CLOSE) {
            "complete_tool_call"
        } else if decoded.contains(TOOL_CALL_OPEN) {
            "in_tool_call"
        } else {
            "before_tool_call"
        }
    }

    pub(super) fn should_stop_after_token(
        &self,
        tokenizer: &QwenTokenizer,
        generated: &[u32],
        token: u32,
    ) -> Result<bool> {
        if self.terminal_tool_names.is_empty() {
            return Ok(false);
        }
        let mut trial = Vec::with_capacity(generated.len() + 1);
        trial.extend_from_slice(generated);
        trial.push(token);
        Ok(self.output_has_complete_terminal_call(&tokenizer.decode(&trial)?))
    }

    fn output_has_complete_terminal_call(&self, decoded: &str) -> bool {
        if self.terminal_tool_names.is_empty() {
            return false;
        }
        let mut remaining = decoded;
        while let Some(open) = remaining.find(TOOL_CALL_OPEN) {
            remaining = &remaining[open + TOOL_CALL_OPEN.len()..];
            if let Some(close) = remaining.find(TOOL_CALL_CLOSE) {
                if self.body_names_terminal_tool(remaining[..close].trim()) {
                    return true;
                }
                remaining = &remaining[close + TOOL_CALL_CLOSE.len()..];
                continue;
            }
            return match self.parse_tool_body(remaining) {
                PrefixStatus::Complete(position)
                    if self.close_suffix_is_valid(&remaining[position..]) =>
                {
                    self.body_names_terminal_tool(remaining[..position].trim())
                }
                _ => false,
            };
        }
        false
    }

    fn body_names_terminal_tool(&self, body: &str) -> bool {
        serde_json::from_str::<Value>(body)
            .ok()
            .and_then(|value| value.get("name")?.as_str().map(str::to_owned))
            .is_some_and(|name| self.terminal_tool_names.contains(&name))
    }

    fn close_suffix_is_valid(&self, suffix: &str) -> bool {
        TOOL_CALL_CLOSE.starts_with(suffix)
            || "\n</tool_call>".starts_with(suffix)
            || "\r\n</tool_call>".starts_with(suffix)
    }

    pub(super) fn forced_next_token(
        &mut self,
        tokenizer: &QwenTokenizer,
        generated: &[u32],
    ) -> Result<Option<u32>> {
        if self.forced_tokens.is_empty() {
            let decoded = tokenizer.decode(generated)?;
            if let Some(tool) = self.bounded_mutation_string_limit_tool(&decoded) {
                self.payload_limit_stop = Some(tool);
                self.stopped_at_payload_limit = true;
                return Ok(Some(tokenizer.eos_token_id()));
            }
            let remainder = self
                .unclosed_tool_call_close_remainder(&decoded)
                .or_else(|| self.bounded_string_structural_remainder(&decoded));
            if let Some(remainder) = remainder {
                let tokens = tokenizer.encode(&remainder)?;
                if tokens.is_empty() {
                    bail!("native tool constraint could not tokenize forced Qwen structure");
                }
                let mut trial = generated.to_vec();
                trial.extend_from_slice(&tokens);
                let decoded = tokenizer.decode(&trial)?;
                if !decoded.contains(TOOL_CALL_CLOSE)
                    || !self.output_prefix_is_valid(&decoded, false)
                {
                    bail!("native tool constraint could not force valid Qwen structure");
                }
                self.forced_tokens = tokens.into();
            }
        }
        Ok(self.forced_tokens.pop_front())
    }

    pub(super) fn take_payload_limit_stop(&mut self) -> Option<String> {
        self.payload_limit_stop.take()
    }

    fn unclosed_tool_call_close_remainder(&self, decoded: &str) -> Option<String> {
        let open = decoded.rfind(TOOL_CALL_OPEN)?;
        let body = &decoded[open + TOOL_CALL_OPEN.len()..];
        if body.contains(TOOL_CALL_CLOSE) {
            return None;
        }
        let PrefixStatus::Complete(position) = self.parse_tool_body(body) else {
            return None;
        };
        let suffix = &body[position..];
        [TOOL_CALL_CLOSE, "\n</tool_call>", "\r\n</tool_call>"]
            .into_iter()
            .find_map(|target| {
                target
                    .strip_prefix(suffix)
                    .map(|remainder| remainder.to_string())
            })
    }

    fn bounded_string_structural_remainder(&self, decoded: &str) -> Option<String> {
        let open = decoded.rfind(TOOL_CALL_OPEN)?;
        let body = &decoded[open + TOOL_CALL_OPEN.len()..];
        if body.contains(TOOL_CALL_CLOSE)
            || self.parse_tool_body(&format!("{body}x")) != PrefixStatus::Invalid
        {
            return None;
        }
        ["\"}}", "\"}]}", "\"}}}", "\"}]}}", "\"}}}}"]
            .into_iter()
            .find(|suffix| self.tool_body_is_complete(&format!("{body}{suffix}")))
            .map(|suffix| format!("{suffix}{TOOL_CALL_CLOSE}"))
    }

    fn bounded_mutation_string_limit_tool(&self, decoded: &str) -> Option<String> {
        let open = decoded.rfind(TOOL_CALL_OPEN)?;
        let body = &decoded[open + TOOL_CALL_OPEN.len()..];
        if body.contains(TOOL_CALL_CLOSE)
            || self.parse_tool_body(&format!("{body}x")) != PrefixStatus::Invalid
            || !["\"}}", "\"}]}", "\"}}}", "\"}]}}", "\"}}}}"]
                .into_iter()
                .any(|suffix| self.tool_body_is_complete(&format!("{body}{suffix}")))
        {
            return None;
        }
        let name = self.tool_name_from_body_prefix(body)?;
        matches!(name.as_str(), "write_file" | "replace_file").then_some(name)
    }

    fn tool_name_from_body_prefix(&self, body: &str) -> Option<String> {
        let mut position = skip_ws(body, 0);
        position = match consume_byte(body, position, b'{') {
            PrefixStatus::Complete(position) => position,
            _ => return None,
        };
        position = skip_ws(body, position);
        position = match parse_fixed_string(body, position, &["name"]) {
            StringStatus::Complete(name, position) if name == "name" => position,
            _ => return None,
        };
        position = skip_ws(body, position);
        position = match consume_byte(body, position, b':') {
            PrefixStatus::Complete(position) => position,
            _ => return None,
        };
        position = skip_ws(body, position);
        let names = self.schemas.keys().map(String::as_str).collect::<Vec<_>>();
        match parse_fixed_string(body, position, &names) {
            StringStatus::Complete(name, _) => Some(name),
            _ => None,
        }
    }

    pub(super) fn filter_candidates(
        &mut self,
        tokenizer: &QwenTokenizer,
        generated: &[u32],
        candidates: Vec<(usize, f32)>,
        keep: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let mut accepted = Vec::with_capacity(keep.min(candidates.len()));
        let decoded_prefix = tokenizer.decode(generated)?;
        let forbidden_repetition_tokens =
            repeated_ngram_forbidden_tokens(generated, CONSTRAINED_NO_REPEAT_NGRAM);
        for (token, score) in candidates {
            let token = u32::try_from(token).context("candidate token id does not fit u32")?;
            if forbidden_repetition_tokens.contains(&token) {
                self.rejected_candidates = self.rejected_candidates.saturating_add(1);
                continue;
            }
            let mut trial = generated.to_vec();
            trial.push(token);
            let decoded = tokenizer.decode(&trial)?;
            let is_eos = tokenizer.is_eos(token);
            if candidate_advances_visible_output(&decoded_prefix, &decoded, is_eos)
                && self.output_prefix_is_valid(&decoded, is_eos)
            {
                accepted.push((token as usize, score));
                if accepted.len() >= keep {
                    break;
                }
            } else {
                self.rejected_candidates = self.rejected_candidates.saturating_add(1);
                if let Some(code) = self.output_mutation_rejection(&decoded) {
                    *self.mutation_rejections.entry(code).or_default() += 1;
                }
            }
        }
        Ok(accepted)
    }

    fn output_prefix_is_valid(&self, decoded: &str, at_eos: bool) -> bool {
        let first_open = decoded.find(TOOL_CALL_OPEN);
        let start = match (self.mode, first_open) {
            (NativeToolConstraintMode::ToolRequired, Some(start)) => {
                if !decoded[..start].trim().is_empty() {
                    return false;
                }
                start
            }
            (NativeToolConstraintMode::ToolRequired, None) => {
                let prefix = decoded.trim_start();
                return !at_eos
                    && decoded.len().saturating_sub(prefix.len())
                        <= MAX_STRUCTURAL_WHITESPACE_BYTES
                    && TOOL_CALL_OPEN.starts_with(prefix);
            }
            (_, None) => return true,
            (_, Some(start)) => start,
        };

        if !structural_whitespace_is_bounded(&decoded[start..]) {
            return false;
        }

        let mut remaining = &decoded[start..];
        let mut mutation_calls = 0usize;
        let mutation_limit = self.mutation_gate.as_ref().map_or(usize::MAX, |gate| {
            gate.manifest().mutation_policy.max_mutation_calls_per_batch
        });
        loop {
            if !remaining.starts_with(TOOL_CALL_OPEN) {
                return !at_eos && TOOL_CALL_OPEN.starts_with(remaining.trim_start());
            }
            remaining = &remaining[TOOL_CALL_OPEN.len()..];
            let Some(close) = remaining.find(TOOL_CALL_CLOSE) else {
                if at_eos || !self.tool_body_prefix_is_valid(remaining) {
                    return false;
                }
                return !matches!(
                    self.mutation_payload_completion_decision(remaining),
                    CompletionDecision::Accept if mutation_calls >= mutation_limit
                );
            };
            let body = &remaining[..close];
            if !self.tool_body_is_complete(body) {
                return false;
            }
            if matches!(
                self.completed_tool_body_decision(body),
                CompletionDecision::Accept
            ) {
                mutation_calls = mutation_calls.saturating_add(1);
                if mutation_calls > mutation_limit {
                    return false;
                }
            }
            remaining = remaining[close + TOOL_CALL_CLOSE.len()..].trim_start();
            if remaining.is_empty() {
                return true;
            }
        }
    }

    fn tool_body_prefix_is_valid(&self, body: &str) -> bool {
        match self.parse_tool_body(body) {
            PrefixStatus::Incomplete => {
                !matches!(
                    self.mutation_payload_prefix_decision(body),
                    CompletionDecision::Reject(_)
                ) && !matches!(
                    self.mutation_payload_completion_decision(body),
                    CompletionDecision::Reject(_)
                )
            }
            PrefixStatus::Complete(position) => {
                self.close_suffix_is_valid(&body[position..])
                    && self.completed_tool_body_is_allowed(&body[..position])
            }
            PrefixStatus::Invalid => false,
        }
    }

    fn tool_body_is_complete(&self, body: &str) -> bool {
        matches!(self.parse_tool_body(body), PrefixStatus::Complete(position) if skip_ws(body, position) == body.len())
            && self.completed_tool_body_is_allowed(body)
    }

    fn completed_tool_body_is_allowed(&self, body: &str) -> bool {
        !matches!(
            self.completed_tool_body_decision(body),
            CompletionDecision::Reject(_)
        )
    }

    fn completed_tool_body_decision(&self, body: &str) -> CompletionDecision {
        let Some(gate) = self.mutation_gate.as_ref() else {
            return CompletionDecision::NotApplicable;
        };
        let Ok(value) = serde_json::from_str::<Value>(body.trim()) else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        let Some(name) = value.get("name").and_then(Value::as_str) else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        let Some(arguments) = value.get("arguments") else {
            return CompletionDecision::Reject(RejectionCode::InvalidArguments);
        };
        let decision = gate.evaluate(name, arguments);
        self.semantic_decision(name, arguments, decision)
    }

    fn completed_tool_body_rejection(&self, body: &str) -> Option<RejectionCode> {
        match self.completed_tool_body_decision(body) {
            CompletionDecision::Reject(code) => Some(code),
            CompletionDecision::Accept | CompletionDecision::NotApplicable => None,
        }
    }

    fn output_mutation_rejection(&self, decoded: &str) -> Option<RejectionCode> {
        let open = decoded.rfind(TOOL_CALL_OPEN)?;
        let body = &decoded[open + TOOL_CALL_OPEN.len()..];
        if let Some(close) = body.find(TOOL_CALL_CLOSE) {
            self.completed_tool_body_rejection(&body[..close])
        } else {
            let prefix = self.mutation_payload_prefix_decision(body);
            let decision = if matches!(prefix, CompletionDecision::Reject(_)) {
                prefix
            } else {
                self.mutation_payload_completion_decision(body)
            };
            match decision {
                CompletionDecision::Reject(code) => Some(code),
                CompletionDecision::Accept | CompletionDecision::NotApplicable => None,
            }
        }
    }

    fn mutation_payload_prefix_decision(&self, body: &str) -> CompletionDecision {
        let Some(gate) = self.mutation_gate.as_ref() else {
            return CompletionDecision::NotApplicable;
        };
        let Some((name, arguments, prefix)) = self.mutation_payload_prefix(body) else {
            return CompletionDecision::NotApplicable;
        };
        gate.evaluate_prefix(&name, &arguments, &prefix)
    }

    /// Evaluate a mutation as soon as its payload string closes. Waiting for the outer JSON object
    /// would commit an irreversible closing quote and leave the sampler no continuation that can
    /// repair invalid source or a stale patch.
    fn mutation_payload_completion_decision(&self, body: &str) -> CompletionDecision {
        let Some(gate) = self.mutation_gate.as_ref() else {
            return CompletionDecision::NotApplicable;
        };
        let Some((name, arguments)) = self.closed_mutation_payload(body) else {
            return CompletionDecision::NotApplicable;
        };
        let arguments = Value::Object(arguments);
        let decision = gate.evaluate(&name, &arguments);
        self.semantic_decision(&name, &arguments, decision)
    }

    fn semantic_decision(
        &self,
        name: &str,
        arguments: &Value,
        structural: CompletionDecision,
    ) -> CompletionDecision {
        if structural != CompletionDecision::Accept {
            return structural;
        }
        match self
            .semantic_provider
            .as_ref()
            .map(|provider| provider.probe(name, arguments).closure)
        {
            Some(ClosureVerdict::Reject) => {
                CompletionDecision::Reject(RejectionCode::InvalidSemantics)
            }
            Some(ClosureVerdict::Allow | ClosureVerdict::Defer) | None => structural,
        }
    }

    fn closed_mutation_payload(
        &self,
        body: &str,
    ) -> Option<(String, serde_json::Map<String, Value>)> {
        let mut position = skip_ws(body, 0);
        position = complete_position(consume_byte(body, position, b'{'))?;
        position = skip_ws(body, position);
        let (name_key, next) = complete_string(parse_fixed_string(body, position, &["name"]))?;
        if name_key != "name" {
            return None;
        }
        position = skip_ws(body, next);
        position = complete_position(consume_byte(body, position, b':'))?;
        position = skip_ws(body, position);
        let names = self.schemas.keys().map(String::as_str).collect::<Vec<_>>();
        let (name, next) = complete_string(parse_fixed_string(body, position, &names))?;
        let payload_name = match name.as_str() {
            "write_file" | "replace_file" => "content",
            "edit_file" => "new_text",
            "apply_patch" => "patch",
            _ => return None,
        };
        position = skip_ws(body, next);
        position = complete_position(consume_byte(body, position, b','))?;
        position = skip_ws(body, position);
        let (arguments_key, next) =
            complete_string(parse_fixed_string(body, position, &["arguments"]))?;
        if arguments_key != "arguments" {
            return None;
        }
        position = skip_ws(body, next);
        position = complete_position(consume_byte(body, position, b':'))?;
        position = skip_ws(body, position);
        position = complete_position(consume_byte(body, position, b'{'))?;

        let schema = self.schemas.get(&name)?;
        let properties = schema.get("properties")?.as_object()?;
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<BTreeSet<_>>();
        let order = mutation_argument_order(&name)?;
        let mut next_order_index = 0usize;
        let mut values = serde_json::Map::new();
        loop {
            position = skip_ws(body, position);
            let (key, next) = complete_string(parse_json_string(body, position))?;
            let order_index = order.iter().position(|expected| *expected == key)?;
            if order_index < next_order_index
                || !properties.contains_key(&key)
                || order[next_order_index..order_index]
                    .iter()
                    .any(|skipped| required.contains(skipped))
            {
                return None;
            }
            next_order_index = order_index.saturating_add(1);
            position = skip_ws(body, next);
            position = complete_position(consume_byte(body, position, b':'))?;
            position = skip_ws(body, position);
            let (value, next) = complete_string(parse_json_string(body, position))?;
            values.insert(key.clone(), Value::String(value));
            if key == payload_name {
                return Some((name, values));
            }
            position = skip_ws(body, next);
            position = complete_position(consume_byte(body, position, b','))?;
        }
    }

    fn mutation_payload_prefix(
        &self,
        body: &str,
    ) -> Option<(String, serde_json::Map<String, Value>, String)> {
        let mut position = skip_ws(body, 0);
        position = complete_position(consume_byte(body, position, b'{'))?;
        position = skip_ws(body, position);
        let (name_key, next) = complete_string(parse_fixed_string(body, position, &["name"]))?;
        if name_key != "name" {
            return None;
        }
        position = skip_ws(body, next);
        position = complete_position(consume_byte(body, position, b':'))?;
        position = skip_ws(body, position);
        let names = self.schemas.keys().map(String::as_str).collect::<Vec<_>>();
        let (name, next) = complete_string(parse_fixed_string(body, position, &names))?;
        let payload_name = match name.as_str() {
            "write_file" | "replace_file" => "content",
            "edit_file" => "new_text",
            "apply_patch" => "patch",
            _ => return None,
        };
        position = skip_ws(body, next);
        position = complete_position(consume_byte(body, position, b','))?;
        position = skip_ws(body, position);
        let (arguments_key, next) =
            complete_string(parse_fixed_string(body, position, &["arguments"]))?;
        if arguments_key != "arguments" {
            return None;
        }
        position = skip_ws(body, next);
        position = complete_position(consume_byte(body, position, b':'))?;
        position = skip_ws(body, position);
        position = complete_position(consume_byte(body, position, b'{'))?;

        let schema = self.schemas.get(&name)?;
        let properties = schema.get("properties")?.as_object()?;
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<BTreeSet<_>>();
        let order = mutation_argument_order(&name)?;
        let mut next_order_index = 0usize;
        let mut values = serde_json::Map::new();
        loop {
            position = skip_ws(body, position);
            let (key, next) = complete_string(parse_json_string(body, position))?;
            let order_index = order.iter().position(|expected| *expected == key)?;
            if order_index < next_order_index
                || !properties.contains_key(&key)
                || order[next_order_index..order_index]
                    .iter()
                    .any(|skipped| required.contains(skipped))
            {
                return None;
            }
            next_order_index = order_index.saturating_add(1);
            position = skip_ws(body, next);
            position = complete_position(consume_byte(body, position, b':'))?;
            position = skip_ws(body, position);
            match parse_json_string(body, position) {
                StringStatus::Complete(value, next) => {
                    values.insert(key.clone(), Value::String(value));
                    if key == payload_name {
                        return None;
                    }
                    position = skip_ws(body, next);
                    position = complete_position(consume_byte(body, position, b','))?;
                }
                StringStatus::Incomplete(Some(prefix)) if key == payload_name => {
                    return Some((name, values, prefix));
                }
                StringStatus::Incomplete(_) | StringStatus::Invalid => return None,
            }
        }
    }

    fn parse_tool_body(&self, body: &str) -> PrefixStatus {
        let mut position = skip_ws(body, 0);
        position = match consume_byte(body, position, b'{') {
            PrefixStatus::Complete(position) => position,
            status => return status,
        };
        position = skip_ws(body, position);
        position = match parse_fixed_string(body, position, &["name"]) {
            StringStatus::Complete(name, position) if name == "name" => position,
            StringStatus::Incomplete(Some(prefix)) if "name".starts_with(prefix.as_str()) => {
                return PrefixStatus::Incomplete;
            }
            StringStatus::Incomplete(None) => return PrefixStatus::Incomplete,
            _ => return PrefixStatus::Invalid,
        };
        position = skip_ws(body, position);
        position = match consume_byte(body, position, b':') {
            PrefixStatus::Complete(position) => position,
            status => return status,
        };
        position = skip_ws(body, position);
        let names = self.schemas.keys().map(String::as_str).collect::<Vec<_>>();
        let (name, next) = match parse_fixed_string(body, position, &names) {
            StringStatus::Complete(name, position) if self.schemas.contains_key(&name) => {
                (name, position)
            }
            StringStatus::Incomplete(Some(prefix))
                if names.iter().any(|name| name.starts_with(&prefix)) =>
            {
                return PrefixStatus::Incomplete;
            }
            StringStatus::Incomplete(None) => return PrefixStatus::Incomplete,
            _ => return PrefixStatus::Invalid,
        };
        position = skip_ws(body, next);
        position = match consume_byte(body, position, b',') {
            PrefixStatus::Complete(position) => position,
            status => return status,
        };
        position = skip_ws(body, position);
        position = match parse_fixed_string(body, position, &["arguments"]) {
            StringStatus::Complete(key, position) if key == "arguments" => position,
            StringStatus::Incomplete(Some(prefix)) if "arguments".starts_with(prefix.as_str()) => {
                return PrefixStatus::Incomplete;
            }
            StringStatus::Incomplete(None) => return PrefixStatus::Incomplete,
            _ => return PrefixStatus::Invalid,
        };
        position = skip_ws(body, position);
        position = match consume_byte(body, position, b':') {
            PrefixStatus::Complete(position) => position,
            status => return status,
        };
        position = skip_ws(body, position);
        let Some(schema) = self.schemas.get(&name) else {
            return PrefixStatus::Invalid;
        };
        let parser = JsonPrefixParser::new(body);
        let arguments = match mutation_argument_order(&name) {
            Some(order) => parser.parse_ordered_object(position, schema, order),
            None => parser.parse_value(position, schema),
        };
        position = match arguments {
            PrefixStatus::Complete(position) => position,
            status => return status,
        };
        position = skip_ws(body, position);
        consume_byte(body, position, b'}')
    }
}

#[derive(Debug, Clone)]
pub(super) struct DeepSeekToolConstraint {
    mode: NativeToolConstraintMode,
    collar: pb_control_collar::protocol::DsmlConstraint,
    schema_sha256: String,
    rejected_candidates: usize,
    mutation_rejections: BTreeMap<RejectionCode, usize>,
    snapshot_files: usize,
    snapshot_bytes: usize,
    mutation_enabled: bool,
    semantic_provider: Option<crate::inference::SemanticBoundaryControl>,
}

impl DeepSeekToolConstraint {
    fn compile(
        mode: NativeToolConstraintMode,
        tools: &[ChatTool],
        terminal_tool_names: &[String],
        snapshot: Option<&pb_control_collar::mutation::WorkspaceSnapshot>,
        language_layers: Option<crate::control_layers::SharedLanguageLayers>,
    ) -> Result<Option<Self>> {
        let active_mode = match mode {
            NativeToolConstraintMode::Auto if tools.is_empty() => return Ok(None),
            NativeToolConstraintMode::Auto => NativeToolConstraintMode::ToolsAllowed,
            mode => mode,
        };
        if tools.is_empty() {
            bail!("native DeepSeek tool constraint mode requires at least one tool");
        }
        let has_mutation = tools.iter().any(|tool| {
            matches!(
                tool.name.as_str(),
                "write_file" | "replace_file" | "edit_file" | "apply_patch"
            )
        });
        let workspace = match (has_mutation, snapshot) {
            (true, None) => {
                bail!("DeepSeek mutation constraints require a controller-authorized snapshot")
            }
            (_, Some(snapshot)) => snapshot.clone(),
            (false, None) => pb_control_collar::mutation::WorkspaceSnapshot::default(),
        };
        let snapshot_files = workspace.len();
        let snapshot_bytes = workspace.total_bytes();
        let manifest = collar_manifest(
            ToolDialect::DeepSeekDsml,
            active_mode,
            tools,
            terminal_tool_names,
            workspace,
        )?;
        let schema_bytes = serde_json::to_vec(
            &tools
                .iter()
                .map(|tool| (&tool.name, &tool.input_schema))
                .collect::<BTreeMap<_, _>>(),
        )?;
        Ok(Some(Self {
            mode: active_mode,
            collar: pb_control_collar::protocol::DsmlConstraint::compile_with_language_layers(
                manifest,
                language_layers,
            )?,
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            rejected_candidates: 0,
            mutation_rejections: BTreeMap::new(),
            snapshot_files,
            snapshot_bytes,
            mutation_enabled: has_mutation,
            semantic_provider: None,
        }))
    }

    fn semantic_payload_is_allowed(&self, transcript: &[u8]) -> bool {
        let (Some(provider), Some(call)) = (
            self.semantic_provider.as_ref(),
            self.collar.completed_mutation_payload(transcript),
        ) else {
            return true;
        };
        provider.probe(&call.name, &call.arguments).closure != ClosureVerdict::Reject
    }

    fn transcript_bytes(tokenizer: &QwenTokenizer, tokens: &[u32]) -> Result<(Vec<u8>, usize)> {
        let mut bytes = Vec::new();
        let mut dsml_controls = 0usize;
        for token in tokens {
            if tokenizer.is_eos(*token) {
                continue;
            }
            let surface = tokenizer
                .constraint_token_surface(*token)
                .with_context(|| {
                    format!("DeepSeek token {token} has no constraint vocabulary surface")
                })?;
            match surface {
                pb_control_collar::vocabulary::TokenSurface::Bytes(visible) => {
                    bytes.extend_from_slice(visible)
                }
                pb_control_collar::vocabulary::TokenSurface::Control {
                    identity,
                    visible_bytes,
                } => {
                    if identity.0 == "｜DSML｜" {
                        dsml_controls = dsml_controls.saturating_add(1);
                    }
                    bytes.extend_from_slice(visible_bytes);
                }
            }
        }
        Ok((bytes, dsml_controls))
    }

    fn identity_is_valid(bytes: &[u8], dsml_controls: usize) -> bool {
        bytes
            .windows("｜DSML｜".len())
            .filter(|window| *window == "｜DSML｜".as_bytes())
            .count()
            == dsml_controls
    }

    fn filter_candidates(
        &mut self,
        tokenizer: &QwenTokenizer,
        generated: &[u32],
        candidates: Vec<(usize, f32)>,
        keep: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let (prefix, prefix_controls) = Self::transcript_bytes(tokenizer, generated)?;
        let mut accepted = Vec::with_capacity(keep.min(candidates.len()));
        for (token, score) in candidates {
            let token_u32 = u32::try_from(token).context("candidate token id exceeds u32")?;
            let mut trial = prefix.clone();
            let mut controls = prefix_controls;
            if !tokenizer.is_eos(token_u32) {
                match tokenizer
                    .constraint_token_surface(token_u32)
                    .context("DeepSeek candidate has no constraint surface")?
                {
                    pb_control_collar::vocabulary::TokenSurface::Bytes(bytes) => {
                        trial.extend_from_slice(bytes)
                    }
                    pb_control_collar::vocabulary::TokenSurface::Control {
                        identity,
                        visible_bytes,
                    } => {
                        if identity.0 == "｜DSML｜" {
                            controls = controls.saturating_add(1);
                        }
                        trial.extend_from_slice(visible_bytes);
                    }
                }
            }
            let probe = self.collar.probe(&trial, tokenizer.is_eos(token_u32));
            let advances_visible_output = tokenizer.is_eos(token_u32) || trial.len() > prefix.len();
            let identity_valid = Self::identity_is_valid(&trial, controls);
            let semantic_allowed = probe.valid && self.semantic_payload_is_allowed(&trial);
            if advances_visible_output && identity_valid && semantic_allowed {
                accepted.push((token, score));
                if accepted.len() >= keep {
                    break;
                }
            } else {
                self.rejected_candidates = self.rejected_candidates.saturating_add(1);
                if let Some(code) = probe.rejection {
                    *self.mutation_rejections.entry(code).or_default() += 1;
                } else if probe.valid && identity_valid && !semantic_allowed {
                    *self
                        .mutation_rejections
                        .entry(RejectionCode::InvalidSemantics)
                        .or_default() += 1;
                }
            }
        }
        Ok(accepted)
    }

    fn should_stop_after_token(
        &self,
        tokenizer: &QwenTokenizer,
        generated: &[u32],
        token: u32,
    ) -> Result<bool> {
        let mut trial = generated.to_vec();
        trial.push(token);
        let (bytes, controls) = Self::transcript_bytes(tokenizer, &trial)?;
        Ok(Self::identity_is_valid(&bytes, controls) && self.collar.probe(&bytes, false).complete)
    }

    fn terminal_state(&self, decoded: &str) -> &'static str {
        self.collar.terminal_state(decoded.as_bytes())
    }
}

#[derive(Debug, Clone)]
pub(super) enum RuntimeToolConstraint {
    Qwen(NativeToolConstraint),
    DeepSeek(DeepSeekToolConstraint),
}

impl RuntimeToolConstraint {
    pub(super) fn compile_qwen(
        mode: NativeToolConstraintMode,
        tools: &[ChatTool],
        terminal_tool_names: &[String],
        mutation_gate: Option<MutationCompletionGate>,
    ) -> Result<Option<Self>> {
        Ok(NativeToolConstraint::compile_with_mutation_gate(
            mode,
            tools,
            terminal_tool_names,
            mutation_gate,
        )?
        .map(Self::Qwen))
    }

    pub(super) fn compile_deepseek(
        mode: NativeToolConstraintMode,
        tools: &[ChatTool],
        terminal_tool_names: &[String],
        snapshot: Option<&pb_control_collar::mutation::WorkspaceSnapshot>,
        language_layers: Option<crate::control_layers::SharedLanguageLayers>,
    ) -> Result<Option<Self>> {
        Ok(DeepSeekToolConstraint::compile(
            mode,
            tools,
            terminal_tool_names,
            snapshot,
            language_layers,
        )?
        .map(Self::DeepSeek))
    }

    pub(super) fn mode(&self) -> NativeToolConstraintMode {
        match self {
            Self::Qwen(constraint) => constraint.mode(),
            Self::DeepSeek(constraint) => constraint.mode,
        }
    }

    pub(super) fn set_semantic_provider(
        &mut self,
        provider: Option<crate::inference::SemanticBoundaryControl>,
    ) {
        match self {
            Self::Qwen(constraint) => constraint.set_semantic_provider(provider),
            Self::DeepSeek(constraint) => constraint.semantic_provider = provider,
        }
    }

    pub(super) fn schema_sha256(&self) -> &str {
        match self {
            Self::Qwen(constraint) => constraint.schema_sha256(),
            Self::DeepSeek(constraint) => &constraint.schema_sha256,
        }
    }

    pub(super) fn rejected_candidates(&self) -> usize {
        match self {
            Self::Qwen(constraint) => constraint.rejected_candidates(),
            Self::DeepSeek(constraint) => constraint.rejected_candidates,
        }
    }

    pub(super) fn dialect(&self) -> &'static str {
        match self {
            Self::Qwen(_) => "qwen_json",
            Self::DeepSeek(_) => "deepseek_dsml",
        }
    }

    pub(super) fn mutation_rejections(&self) -> BTreeMap<String, usize> {
        let source = match self {
            Self::Qwen(constraint) => &constraint.mutation_rejections,
            Self::DeepSeek(constraint) => &constraint.mutation_rejections,
        };
        source
            .iter()
            .map(|(code, count)| (code.as_str().to_string(), *count))
            .collect()
    }

    pub(super) fn snapshot_stats(&self) -> (usize, usize) {
        match self {
            Self::Qwen(constraint) => constraint.mutation_gate.as_ref().map_or((0, 0), |gate| {
                let workspace = &gate.manifest().workspace;
                (workspace.len(), workspace.total_bytes())
            }),
            Self::DeepSeek(constraint) => (constraint.snapshot_files, constraint.snapshot_bytes),
        }
    }

    pub(super) fn guarantee_rung(&self, decoded: &str) -> &'static str {
        match self {
            Self::Qwen(constraint) if constraint.semantic_closure_authorized(decoded) => {
                "semantic_boundary"
            }
            Self::DeepSeek(constraint)
                if constraint
                    .semantic_provider
                    .as_ref()
                    .is_some_and(|provider| {
                        constraint.collar.probe(decoded.as_bytes(), false).complete
                            && constraint
                                .collar
                                .completed_mutation_payload(decoded.as_bytes())
                                .is_some_and(|call| {
                                    provider.probe(&call.name, &call.arguments).closure
                                        == ClosureVerdict::Allow
                                })
                    }) =>
            {
                "semantic_boundary"
            }
            Self::Qwen(constraint) if constraint.mutation_gate.is_some() => "prefix_syntax",
            Self::DeepSeek(constraint) if constraint.mutation_enabled => "prefix_syntax",
            _ => "protocol_schema",
        }
    }

    pub(super) fn semantic_stats(&self) -> Option<crate::inference::SemanticBoundaryStats> {
        match self {
            Self::Qwen(constraint) => constraint
                .semantic_provider
                .as_ref()
                .map(crate::inference::SemanticBoundaryControl::stats),
            Self::DeepSeek(constraint) => constraint
                .semantic_provider
                .as_ref()
                .map(crate::inference::SemanticBoundaryControl::stats),
        }
    }

    pub(super) fn terminal_state(&self, decoded: &str) -> &'static str {
        match self {
            Self::Qwen(constraint) => constraint.terminal_state(decoded),
            Self::DeepSeek(constraint) => constraint.terminal_state(decoded),
        }
    }

    pub(super) fn forced_next_token(
        &mut self,
        tokenizer: &QwenTokenizer,
        generated: &[u32],
    ) -> Result<Option<u32>> {
        match self {
            Self::Qwen(constraint) => constraint.forced_next_token(tokenizer, generated),
            Self::DeepSeek(_) => Ok(None),
        }
    }

    pub(super) fn take_payload_limit_stop(&mut self) -> Option<String> {
        match self {
            Self::Qwen(constraint) => constraint.take_payload_limit_stop(),
            Self::DeepSeek(_) => None,
        }
    }

    pub(super) fn should_stop_after_token(
        &self,
        tokenizer: &QwenTokenizer,
        generated: &[u32],
        token: u32,
    ) -> Result<bool> {
        match self {
            Self::Qwen(constraint) => {
                constraint.should_stop_after_token(tokenizer, generated, token)
            }
            Self::DeepSeek(constraint) => {
                constraint.should_stop_after_token(tokenizer, generated, token)
            }
        }
    }

    pub(super) fn filter_candidates(
        &mut self,
        tokenizer: &QwenTokenizer,
        generated: &[u32],
        candidates: Vec<(usize, f32)>,
        keep: usize,
    ) -> Result<Vec<(usize, f32)>> {
        match self {
            Self::Qwen(constraint) => {
                constraint.filter_candidates(tokenizer, generated, candidates, keep)
            }
            Self::DeepSeek(constraint) => {
                constraint.filter_candidates(tokenizer, generated, candidates, keep)
            }
        }
    }
}

fn candidate_advances_visible_output(prefix: &str, candidate: &str, is_eos: bool) -> bool {
    is_eos || candidate.len() > prefix.len()
}

fn structural_whitespace_is_bounded(input: &str) -> bool {
    let mut in_string = false;
    let mut escaped = false;
    let mut whitespace_bytes = 0usize;
    for byte in input.bytes() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        if byte == b'"' {
            in_string = true;
            whitespace_bytes = 0;
        } else if byte.is_ascii_whitespace() {
            whitespace_bytes = whitespace_bytes.saturating_add(1);
            if whitespace_bytes > MAX_STRUCTURAL_WHITESPACE_BYTES {
                return false;
            }
        } else {
            whitespace_bytes = 0;
        }
    }
    true
}

fn repeated_ngram_forbidden_tokens(tokens: &[u32], width: usize) -> BTreeSet<u32> {
    if width < 2 || tokens.len() < width.saturating_sub(1) {
        return BTreeSet::new();
    }
    let prefix = &tokens[tokens.len() - (width - 1)..];
    let mut forbidden = BTreeSet::new();
    for start in 0..tokens.len().saturating_sub(width - 1) {
        if tokens[start..start + width - 1] == *prefix {
            forbidden.insert(tokens[start + width - 1]);
        }
    }
    forbidden
}

#[cfg(test)]
pub(crate) fn terminal_tool_output_is_complete(
    tools: &[ChatTool],
    terminal_tool: &str,
    output: &str,
) -> Result<bool> {
    let constraint = NativeToolConstraint::compile_with_terminal_tools(
        NativeToolConstraintMode::ToolsAllowed,
        tools,
        &[terminal_tool.to_string()],
    )?
    .context("test terminal constraint should be active")?;
    Ok(constraint.output_has_complete_terminal_call(output))
}

#[cfg(test)]
pub(crate) fn validate_native_tool_schema(schema: &Value) -> Result<()> {
    validate_supported_schema(schema, "test_tool")
}

fn validate_supported_schema(schema: &Value, location: &str) -> Result<()> {
    let object = schema
        .as_object()
        .with_context(|| format!("{location} schema must be an object"))?;
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "type"
                | "description"
                | "properties"
                | "required"
                | "additionalProperties"
                | "items"
                | "enum"
                | "const"
                | "maxLength"
                | "minLength"
                | "maxItems"
                | "minItems"
                | "minimum"
                | "maximum"
        ) {
            bail!("{location} schema uses unsupported native constraint keyword '{key}'");
        }
    }
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .with_context(|| format!("{location} schema requires one string type"))?;
    for key in object.keys() {
        let common = matches!(key.as_str(), "type" | "description" | "enum" | "const");
        let kind_specific = match kind {
            "object" => matches!(
                key.as_str(),
                "properties" | "required" | "additionalProperties"
            ),
            "array" => matches!(key.as_str(), "items" | "maxItems" | "minItems"),
            "string" => matches!(key.as_str(), "maxLength" | "minLength"),
            "integer" | "number" => matches!(key.as_str(), "minimum" | "maximum"),
            "boolean" => false,
            _ => false,
        };
        if !common && !kind_specific {
            bail!("{location} schema keyword '{key}' is not valid for declared type {kind}");
        }
    }
    match kind {
        "object" => {
            let properties = object
                .get("properties")
                .and_then(Value::as_object)
                .with_context(|| format!("{location} object schema requires properties"))?;
            if object.get("additionalProperties").and_then(Value::as_bool) != Some(false) {
                bail!("{location} object schema must set additionalProperties=false");
            }
            for (name, property) in properties {
                validate_supported_schema(property, &format!("{location}.{name}"))?;
            }
            if let Some(required) = object.get("required") {
                let mut required_names = BTreeSet::new();
                for field in required
                    .as_array()
                    .with_context(|| format!("{location}.required must be an array"))?
                {
                    let field = field
                        .as_str()
                        .with_context(|| format!("{location}.required entries must be strings"))?;
                    if !properties.contains_key(field) {
                        bail!("{location}.required names unknown property '{field}'");
                    }
                    if !required_names.insert(field) {
                        bail!("{location}.required repeats property '{field}'");
                    }
                }
            }
        }
        "array" => validate_supported_schema(
            object
                .get("items")
                .with_context(|| format!("{location} array schema requires items"))?,
            &format!("{location}[]"),
        )?,
        "string" | "integer" | "number" | "boolean" => {}
        other => bail!("{location} schema type '{other}' is unsupported by native constraints"),
    }
    if let Some(values) = object.get("enum") {
        let values = values
            .as_array()
            .with_context(|| format!("{location}.enum must be an array"))?;
        if values.is_empty() {
            bail!("{location}.enum must contain at least one value");
        }
        if values.iter().any(|value| !value_matches_kind(value, kind)) {
            bail!("{location}.enum contains a value outside declared type {kind}");
        }
    }
    if let Some(value) = object.get("const") {
        if matches!(kind, "object" | "array") {
            bail!("{location}.const is supported only for scalar native constraint types");
        }
        if !value_matches_kind(value, kind) {
            bail!("{location}.const contains a value outside declared type {kind}");
        }
        if object
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| !values.contains(value))
        {
            bail!("{location}.const is outside the declared enum");
        }
    }
    if kind == "string" {
        let minimum = object.get("minLength").map(|value| {
            value
                .as_u64()
                .with_context(|| format!("{location}.minLength must be a non-negative integer"))
        });
        let maximum = object.get("maxLength").map(|value| {
            value
                .as_u64()
                .with_context(|| format!("{location}.maxLength must be a non-negative integer"))
        });
        if minimum
            .transpose()?
            .zip(maximum.transpose()?)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            bail!("{location} has minLength greater than maxLength");
        }
    }
    if kind == "array" {
        let minimum = object.get("minItems").map(|value| {
            value
                .as_u64()
                .with_context(|| format!("{location}.minItems must be a non-negative integer"))
        });
        let maximum = object.get("maxItems").map(|value| {
            value
                .as_u64()
                .with_context(|| format!("{location}.maxItems must be a non-negative integer"))
        });
        if minimum
            .transpose()?
            .zip(maximum.transpose()?)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            bail!("{location} has minItems greater than maxItems");
        }
    }
    if matches!(kind, "integer" | "number") {
        let minimum = object.get("minimum").map(|value| {
            value
                .as_f64()
                .with_context(|| format!("{location}.minimum must be numeric"))
        });
        let maximum = object.get("maximum").map(|value| {
            value
                .as_f64()
                .with_context(|| format!("{location}.maximum must be numeric"))
        });
        if minimum
            .transpose()?
            .zip(maximum.transpose()?)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            bail!("{location} has minimum greater than maximum");
        }
    }
    Ok(())
}

fn value_matches_kind(value: &Value, kind: &str) -> bool {
    match kind {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixStatus {
    Complete(usize),
    Incomplete,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StringStatus {
    Complete(String, usize),
    Incomplete(Option<String>),
    Invalid,
}

fn complete_position(status: PrefixStatus) -> Option<usize> {
    match status {
        PrefixStatus::Complete(position) => Some(position),
        PrefixStatus::Incomplete | PrefixStatus::Invalid => None,
    }
}

fn complete_string(status: StringStatus) -> Option<(String, usize)> {
    match status {
        StringStatus::Complete(value, position) => Some((value, position)),
        StringStatus::Incomplete(_) | StringStatus::Invalid => None,
    }
}

fn mutation_argument_order(name: &str) -> Option<&'static [&'static str]> {
    match name {
        "write_file" | "replace_file" => Some(&["path", "content", "completion"]),
        "edit_file" => Some(&["path", "old_text", "new_text", "completion"]),
        "apply_patch" => Some(&["patch"]),
        _ => None,
    }
}

fn skip_ws(input: &str, mut position: usize) -> usize {
    while input
        .as_bytes()
        .get(position)
        .is_some_and(u8::is_ascii_whitespace)
    {
        position += 1;
    }
    position
}

fn consume_byte(input: &str, position: usize, expected: u8) -> PrefixStatus {
    match input.as_bytes().get(position) {
        Some(actual) if *actual == expected => PrefixStatus::Complete(position + 1),
        Some(_) => PrefixStatus::Invalid,
        None => PrefixStatus::Incomplete,
    }
}

fn parse_fixed_string(input: &str, position: usize, allowed: &[&str]) -> StringStatus {
    match parse_json_string(input, position) {
        StringStatus::Complete(value, end) if allowed.contains(&value.as_str()) => {
            StringStatus::Complete(value, end)
        }
        StringStatus::Incomplete(Some(prefix))
            if allowed.iter().any(|value| value.starts_with(&prefix)) =>
        {
            StringStatus::Incomplete(Some(prefix))
        }
        StringStatus::Incomplete(None) => StringStatus::Incomplete(None),
        _ => StringStatus::Invalid,
    }
}

fn parse_json_string(input: &str, position: usize) -> StringStatus {
    if input.as_bytes().get(position) != Some(&b'"') {
        return if position == input.len() {
            StringStatus::Incomplete(Some(String::new()))
        } else {
            StringStatus::Invalid
        };
    }
    let mut escaped = false;
    let mut unicode_escape_digits = 0u8;
    for (offset, character) in input[position + 1..].char_indices() {
        let absolute = position + 1 + offset;
        if unicode_escape_digits > 0 {
            if !character.is_ascii_hexdigit() {
                return StringStatus::Invalid;
            }
            unicode_escape_digits -= 1;
            continue;
        }
        if escaped {
            escaped = false;
            match character {
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {}
                'u' => unicode_escape_digits = 4,
                _ => return StringStatus::Invalid,
            }
            continue;
        }
        match character {
            '\\' => escaped = true,
            '"' => {
                let end = absolute + 1;
                return serde_json::from_str::<String>(&input[position..end])
                    .map(|value| StringStatus::Complete(value, end))
                    .unwrap_or(StringStatus::Invalid);
            }
            character if character.is_control() => return StringStatus::Invalid,
            _ => {}
        }
    }
    if escaped || unicode_escape_digits > 0 {
        return StringStatus::Incomplete(None);
    }
    let unterminated = &input[position..];
    serde_json::from_str::<String>(&format!("{unterminated}\""))
        .map(|value| StringStatus::Incomplete(Some(value)))
        .unwrap_or(StringStatus::Incomplete(None))
}

struct JsonPrefixParser<'a> {
    input: &'a str,
}

impl<'a> JsonPrefixParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input }
    }

    fn parse_value(&self, position: usize, schema: &Value) -> PrefixStatus {
        let position = skip_ws(self.input, position);
        let kind = match schema.get("type").and_then(Value::as_str) {
            Some(kind) => kind,
            None => return PrefixStatus::Invalid,
        };
        match kind {
            "object" => self.parse_object(position, schema),
            "array" => self.parse_array(position, schema),
            "string" => self.parse_string(position, schema),
            "integer" => self.parse_number(position, schema, true),
            "number" => self.parse_number(position, schema, false),
            "boolean" => self.parse_literal(position, schema, &["true", "false"]),
            _ => PrefixStatus::Invalid,
        }
    }

    fn parse_object(&self, position: usize, schema: &Value) -> PrefixStatus {
        let mut position = match consume_byte(self.input, position, b'{') {
            PrefixStatus::Complete(position) => position,
            status => return status,
        };
        let properties = match schema.get("properties").and_then(Value::as_object) {
            Some(properties) => properties,
            None => return PrefixStatus::Invalid,
        };
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        let mut can_close = true;
        loop {
            position = skip_ws(self.input, position);
            match self.input.as_bytes().get(position) {
                None => return PrefixStatus::Incomplete,
                Some(b'}') => {
                    return if can_close && required.iter().all(|field| seen.contains(*field)) {
                        PrefixStatus::Complete(position + 1)
                    } else {
                        PrefixStatus::Invalid
                    };
                }
                _ => {}
            }
            let available = properties
                .keys()
                .filter(|key| !seen.contains(key.as_str()))
                .map(String::as_str)
                .collect::<Vec<_>>();
            let (key, next) = match parse_fixed_string(self.input, position, &available) {
                StringStatus::Complete(key, next) => (key, next),
                StringStatus::Incomplete(_) => return PrefixStatus::Incomplete,
                StringStatus::Invalid => return PrefixStatus::Invalid,
            };
            if !seen.insert(key.clone()) {
                return PrefixStatus::Invalid;
            }
            position = skip_ws(self.input, next);
            position = match consume_byte(self.input, position, b':') {
                PrefixStatus::Complete(position) => position,
                status => return status,
            };
            let Some(property_schema) = properties.get(&key) else {
                return PrefixStatus::Invalid;
            };
            position = match self.parse_value(position, property_schema) {
                PrefixStatus::Complete(position) => position,
                status => return status,
            };
            position = skip_ws(self.input, position);
            match self.input.as_bytes().get(position) {
                Some(b',') => {
                    position += 1;
                    can_close = false;
                }
                Some(b'}') => {
                    return if required.iter().all(|field| seen.contains(*field)) {
                        PrefixStatus::Complete(position + 1)
                    } else {
                        PrefixStatus::Invalid
                    };
                }
                None => return PrefixStatus::Incomplete,
                _ => return PrefixStatus::Invalid,
            }
        }
    }

    fn parse_ordered_object(
        &self,
        position: usize,
        schema: &Value,
        order: &[&str],
    ) -> PrefixStatus {
        let mut position = match consume_byte(self.input, position, b'{') {
            PrefixStatus::Complete(position) => position,
            status => return status,
        };
        let properties = match schema.get("properties").and_then(Value::as_object) {
            Some(properties) => properties,
            None => return PrefixStatus::Invalid,
        };
        if properties
            .keys()
            .any(|property| !order.contains(&property.as_str()))
        {
            return PrefixStatus::Invalid;
        }
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        let mut last_order_index = None;
        let mut can_close = true;
        loop {
            position = skip_ws(self.input, position);
            match self.input.as_bytes().get(position) {
                None => return PrefixStatus::Incomplete,
                Some(b'}') => {
                    return if can_close && required.iter().all(|field| seen.contains(*field)) {
                        PrefixStatus::Complete(position + 1)
                    } else {
                        PrefixStatus::Invalid
                    };
                }
                _ => {}
            }
            let available = order
                .iter()
                .enumerate()
                .filter(|(index, name)| {
                    properties.contains_key(**name)
                        && !seen.contains(**name)
                        && last_order_index.is_none_or(|last| *index > last)
                        && order[..*index]
                            .iter()
                            .filter(|earlier| required.contains(**earlier))
                            .all(|earlier| seen.contains(*earlier))
                })
                .map(|(_, name)| *name)
                .collect::<Vec<_>>();
            let (key, next) = match parse_fixed_string(self.input, position, &available) {
                StringStatus::Complete(key, next) => (key, next),
                StringStatus::Incomplete(_) => return PrefixStatus::Incomplete,
                StringStatus::Invalid => return PrefixStatus::Invalid,
            };
            let Some(order_index) = order.iter().position(|name| *name == key) else {
                return PrefixStatus::Invalid;
            };
            last_order_index = Some(order_index);
            if !seen.insert(key.clone()) {
                return PrefixStatus::Invalid;
            }
            position = skip_ws(self.input, next);
            position = match consume_byte(self.input, position, b':') {
                PrefixStatus::Complete(position) => position,
                status => return status,
            };
            let Some(property_schema) = properties.get(&key) else {
                return PrefixStatus::Invalid;
            };
            position = match self.parse_value(position, property_schema) {
                PrefixStatus::Complete(position) => position,
                status => return status,
            };
            position = skip_ws(self.input, position);
            match self.input.as_bytes().get(position) {
                Some(b',') => {
                    position += 1;
                    can_close = false;
                }
                Some(b'}') => {
                    return if required.iter().all(|field| seen.contains(*field)) {
                        PrefixStatus::Complete(position + 1)
                    } else {
                        PrefixStatus::Invalid
                    };
                }
                None => return PrefixStatus::Incomplete,
                _ => return PrefixStatus::Invalid,
            }
        }
    }

    fn parse_array(&self, position: usize, schema: &Value) -> PrefixStatus {
        let mut position = match consume_byte(self.input, position, b'[') {
            PrefixStatus::Complete(position) => position,
            status => return status,
        };
        let Some(items) = schema.get("items") else {
            return PrefixStatus::Invalid;
        };
        let minimum = schema.get("minItems").and_then(Value::as_u64).unwrap_or(0);
        let maximum = schema.get("maxItems").and_then(Value::as_u64);
        let mut item_count = 0u64;
        let mut can_close = true;
        loop {
            position = skip_ws(self.input, position);
            match self.input.as_bytes().get(position) {
                None => return PrefixStatus::Incomplete,
                Some(b']') if can_close && item_count >= minimum => {
                    return PrefixStatus::Complete(position + 1);
                }
                Some(b']') => return PrefixStatus::Invalid,
                _ => {}
            }
            position = match self.parse_value(position, items) {
                PrefixStatus::Complete(position) => position,
                status => return status,
            };
            item_count = item_count.saturating_add(1);
            if maximum.is_some_and(|maximum| item_count > maximum) {
                return PrefixStatus::Invalid;
            }
            position = skip_ws(self.input, position);
            match self.input.as_bytes().get(position) {
                Some(b',') => {
                    if maximum.is_some_and(|maximum| item_count >= maximum) {
                        return PrefixStatus::Invalid;
                    }
                    position += 1;
                    can_close = false;
                }
                Some(b']') if item_count >= minimum => {
                    return PrefixStatus::Complete(position + 1);
                }
                Some(b']') => return PrefixStatus::Invalid,
                None => return PrefixStatus::Incomplete,
                _ => return PrefixStatus::Invalid,
            }
        }
    }

    fn parse_string(&self, position: usize, schema: &Value) -> PrefixStatus {
        match parse_json_string(self.input, position) {
            StringStatus::Complete(value, end) => {
                if schema
                    .get("maxLength")
                    .and_then(Value::as_u64)
                    .is_some_and(|max| value.chars().count() as u64 > max)
                    || schema
                        .get("minLength")
                        .and_then(Value::as_u64)
                        .is_some_and(|min| (value.chars().count() as u64) < min)
                    || !schema_accepts_value(schema, &Value::String(value))
                {
                    PrefixStatus::Invalid
                } else {
                    PrefixStatus::Complete(end)
                }
            }
            StringStatus::Incomplete(Some(prefix)) => {
                if schema
                    .get("maxLength")
                    .and_then(Value::as_u64)
                    .is_some_and(|max| prefix.chars().count() as u64 > max)
                    || schema
                        .get("enum")
                        .and_then(Value::as_array)
                        .is_some_and(|values| {
                            !values
                                .iter()
                                .filter_map(Value::as_str)
                                .any(|value| value.starts_with(&prefix))
                        })
                    || schema
                        .get("const")
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.starts_with(&prefix))
                {
                    PrefixStatus::Invalid
                } else {
                    PrefixStatus::Incomplete
                }
            }
            StringStatus::Incomplete(None) => PrefixStatus::Incomplete,
            StringStatus::Invalid => PrefixStatus::Invalid,
        }
    }

    fn parse_number(&self, position: usize, schema: &Value, integer: bool) -> PrefixStatus {
        if position == self.input.len() {
            return PrefixStatus::Incomplete;
        }
        let end = self.input[position..]
            .char_indices()
            .find(|(_, character)| matches!(character, ',' | '}' | ']' | ' ' | '\n' | '\r' | '\t'))
            .map(|(offset, _)| position + offset)
            .unwrap_or(self.input.len());
        let text = &self.input[position..end];
        if matches!(text, "" | "-" | "+") {
            return PrefixStatus::Incomplete;
        }
        let value = if integer {
            text.parse::<i64>().ok().map(Value::from)
        } else {
            text.parse::<f64>().ok().map(Value::from)
        };
        match value {
            Some(value)
                if schema_accepts_value(schema, &value)
                    && numeric_bounds_accept(schema, &value) =>
            {
                PrefixStatus::Complete(end)
            }
            Some(_) => PrefixStatus::Invalid,
            None if end == self.input.len() => PrefixStatus::Incomplete,
            None => PrefixStatus::Invalid,
        }
    }

    fn parse_literal(&self, position: usize, schema: &Value, allowed: &[&str]) -> PrefixStatus {
        let remaining = &self.input[position..];
        for literal in allowed {
            if remaining.starts_with(literal) {
                let value = Value::Bool(*literal == "true");
                return if schema_accepts_value(schema, &value) {
                    PrefixStatus::Complete(position + literal.len())
                } else {
                    PrefixStatus::Invalid
                };
            }
        }
        if allowed.iter().any(|literal| literal.starts_with(remaining)) {
            PrefixStatus::Incomplete
        } else {
            PrefixStatus::Invalid
        }
    }
}

fn schema_accepts_value(schema: &Value, value: &Value) -> bool {
    let enum_accepts = schema
        .get("enum")
        .and_then(Value::as_array)
        .is_none_or(|values| values.contains(value));
    let const_accepts = schema.get("const").is_none_or(|expected| expected == value);
    enum_accepts && const_accepts
}

fn numeric_bounds_accept(schema: &Value, value: &Value) -> bool {
    let Some(number) = value.as_f64() else {
        return false;
    };
    !schema
        .get("minimum")
        .and_then(Value::as_f64)
        .is_some_and(|minimum| number < minimum)
        && !schema
            .get("maximum")
            .and_then(Value::as_f64)
            .is_some_and(|maximum| number > maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pb_control_collar::analysis::{ProviderVerdict, Viability};
    use serde_json::json;

    struct RejectSemantics;

    impl crate::inference::SemanticBoundaryProvider for RejectSemantics {
        fn probe(&self, _tool: &str, _arguments: &Value) -> ProviderVerdict {
            ProviderVerdict {
                viability: Viability::Impossible,
                closure: ClosureVerdict::Reject,
                definite_errors: vec![
                    pb_control_collar::analysis::DefiniteErrorClass::TypeMismatch,
                ],
                unknown_reasons: Vec::new(),
                obligations: Vec::new(),
                biases: Vec::new(),
            }
        }
    }

    fn tools() -> Vec<ChatTool> {
        vec![ChatTool {
            name: "submit_review".to_string(),
            description: None,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["pass", "fail"]},
                    "notes": {"type": "array", "items": {"type": "string"}},
                    "detail": {"type": "string", "maxLength": 8}
                },
                "required": ["verdict", "notes"],
                "additionalProperties": false
            }),
        }]
    }

    fn mutation_tools() -> Vec<ChatTool> {
        vec![ChatTool {
            name: "write_file".to_string(),
            description: None,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "maxLength": 8}
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        }]
    }

    #[test]
    fn required_constraint_rejects_unexposed_names_and_wrong_arguments() {
        let constraint =
            NativeToolConstraint::compile(NativeToolConstraintMode::ToolRequired, &tools())
                .unwrap()
                .unwrap();
        assert!(constraint.output_prefix_is_valid("<tool_call>\n{\"name\": \"sub", false));
        assert!(
            !constraint.output_prefix_is_valid("<tool_call>\n{\"name\": \"write_file\"", false)
        );
        assert!(!constraint.output_prefix_is_valid(
            "<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"wrong\":1}}",
            false
        ));
        assert!(!constraint.output_prefix_is_valid(
            "<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"verdict\":\"maybe\",\"notes\":[]}}</tool_call>",
            true
        ));
        assert!(!constraint.output_prefix_is_valid(
            "<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"verdict\":\"pass\"}}</tool_call>",
            true
        ));
        assert!(constraint.output_prefix_is_valid(
            "<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"verdict\":\"pass\",\"notes\":[\"✓\"]}}</tool_call>",
            true
        ));
        assert!(constraint.output_prefix_is_valid(
            "<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"verdict\":\"pass\",\"notes\":[\"one\"]}}</tool_call>\n<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"verdict\":\"fail\",\"notes\":[]}}</tool_call>",
            true
        ));
        let complete_body = "<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"verdict\":\"pass\",\"notes\":[]}}";
        assert!(constraint.output_prefix_is_valid(complete_body, false));
        assert!(constraint.output_prefix_is_valid(&format!("{complete_body}\n"), false));
        assert!(!constraint.output_prefix_is_valid(&format!("{complete_body}\n\n"), false));
        assert!(!constraint.output_prefix_is_valid(&format!("{complete_body} "), false));
        assert!(constraint.output_prefix_is_valid(&format!("{complete_body}\n</tool"), false));
        assert!(
            !constraint.output_prefix_is_valid(&format!("{complete_body}\nPlease proceed"), false)
        );
        assert!(!constraint.output_prefix_is_valid(
            "<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"verdict\":\"pass\",\"notes\":[\"one\",]}}</tool_call>",
            true
        ));
    }

    #[test]
    fn scalar_const_schema_constrains_prefixes_and_completed_calls() {
        let tools = vec![ChatTool {
            name: "write_file".to_string(),
            description: None,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "const": "answer.py"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        }];
        let constraint =
            NativeToolConstraint::compile(NativeToolConstraintMode::ToolRequired, &tools)
                .unwrap()
                .unwrap();
        let valid = concat!(
            "<tool_call>{\"name\":\"write_file\",\"arguments\":{",
            "\"path\":\"answer.py\",\"content\":\"answer: int = 4\\n\"}}",
            "</tool_call>"
        );
        assert!(constraint.output_prefix_is_valid(valid, true));
        assert!(constraint.output_prefix_is_valid(
            "<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"ans",
            false
        ));
        assert!(!constraint.output_prefix_is_valid(
            "<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"other",
            false
        ));
        assert!(!constraint.output_prefix_is_valid(&valid.replace("answer.py", "other.py"), true));

        let mut conflicting = tools;
        conflicting[0].input_schema["properties"]["path"]["enum"] = json!(["different.py"]);
        assert!(
            NativeToolConstraint::compile(NativeToolConstraintMode::ToolRequired, &conflicting)
                .is_err()
        );
    }

    #[test]
    fn terminal_tool_completion_stops_only_on_the_named_workflow_submission() {
        let mut available = tools();
        available.push(ChatTool {
            name: "read_file".to_string(),
            description: None,
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": false
            }),
        });
        let terminal = vec!["submit_review".to_string()];
        let constraint = NativeToolConstraint::compile_with_terminal_tools(
            NativeToolConstraintMode::ToolsAllowed,
            &available,
            &terminal,
        )
        .unwrap()
        .unwrap();
        let read =
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"game.js\"}}</tool_call>";
        assert!(!constraint.output_has_complete_terminal_call(read));
        let submission = "<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"verdict\":\"pass\",\"notes\":[]}}</tool_call>";
        assert!(constraint.output_has_complete_terminal_call(submission));
        let unclosed_submission = submission.strip_suffix(TOOL_CALL_CLOSE).unwrap();
        assert!(constraint.output_has_complete_terminal_call(unclosed_submission));
        assert!(constraint.output_has_complete_terminal_call(&format!("{unclosed_submission}\n")));
        assert_eq!(
            constraint.terminal_state(unclosed_submission),
            "complete_terminal_tool_call"
        );
        assert!(constraint.output_has_complete_terminal_call(&format!("{read}\n{submission}")));

        let unknown = vec!["write_file".to_string()];
        assert!(
            NativeToolConstraint::compile_with_terminal_tools(
                NativeToolConstraintMode::ToolsAllowed,
                &available,
                &unknown,
            )
            .is_err()
        );
    }

    #[test]
    fn constrained_generation_rejects_invisible_non_eos_tokens() {
        assert!(!candidate_advances_visible_output("call", "call", false));
        assert!(!candidate_advances_visible_output("call", "all", false));
        assert!(!candidate_advances_visible_output("call", "wall", false));
        assert!(candidate_advances_visible_output("call", "call>", false));
        assert!(candidate_advances_visible_output("call", "call", true));
    }

    #[test]
    fn constrained_tool_structure_bounds_whitespace_without_limiting_string_payloads() {
        let constraint =
            NativeToolConstraint::compile(NativeToolConstraintMode::ToolRequired, &tools())
                .unwrap()
                .unwrap();
        let allowed = "\n".repeat(MAX_STRUCTURAL_WHITESPACE_BYTES);
        let rejected = "\n".repeat(MAX_STRUCTURAL_WHITESPACE_BYTES + 1);
        assert!(constraint.output_prefix_is_valid(&allowed, false));
        assert!(!constraint.output_prefix_is_valid(&rejected, false));
        assert!(constraint.output_prefix_is_valid(&format!("{TOOL_CALL_OPEN}{allowed}"), false));
        assert!(!constraint.output_prefix_is_valid(&format!("{TOOL_CALL_OPEN}{rejected}"), false));

        let string_payload = " ".repeat(MAX_STRUCTURAL_WHITESPACE_BYTES * 4);
        assert!(constraint.output_prefix_is_valid(
            &format!(
                "{TOOL_CALL_OPEN}{{\"name\":\"submit_review\",\"arguments\":{{\"verdict\":\"pass\",\"notes\":[\"{string_payload}\"]}}}}{TOOL_CALL_CLOSE}"
            ),
            true
        ));
    }

    #[test]
    fn constrained_generation_blocks_only_the_repeated_ngram_continuation() {
        let mut tokens = (0..40).collect::<Vec<u32>>();
        tokens.extend(8..39);
        let forbidden = repeated_ngram_forbidden_tokens(&tokens, 32);
        assert_eq!(forbidden, BTreeSet::from([39]));
        assert!(repeated_ngram_forbidden_tokens(&tokens, 1).is_empty());
    }

    #[test]
    fn bounded_array_constraints_compile_and_enforce_item_counts() {
        let schema = json!({
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {"type": "string"}
        });
        validate_supported_schema(&schema, "files").unwrap();
        let parser = |input| JsonPrefixParser::new(input).parse_array(0, &schema);
        assert_eq!(parser("[]"), PrefixStatus::Invalid);
        assert_eq!(parser("[\"one\"]"), PrefixStatus::Complete(7));
        assert_eq!(parser("[\"one\",\"two\"]"), PrefixStatus::Complete(13));
        assert_eq!(parser("[\"one\",\"two\","), PrefixStatus::Invalid);
        assert_eq!(parser("[\"one\",\"two\",\"three\"]"), PrefixStatus::Invalid);

        let invalid = json!({
            "type": "array",
            "minItems": 3,
            "maxItems": 2,
            "items": {"type": "string"}
        });
        assert!(validate_supported_schema(&invalid, "files").is_err());
    }

    #[test]
    fn escaped_incomplete_string_prefixes_still_enforce_decoded_length() {
        let schema = json!({"type": "string", "maxLength": 4});
        assert_eq!(
            JsonPrefixParser::new("\"a\\nbc").parse_string(0, &schema),
            PrefixStatus::Incomplete
        );
        assert_eq!(
            JsonPrefixParser::new("\"a\\nbcd").parse_string(0, &schema),
            PrefixStatus::Invalid
        );
    }

    #[test]
    fn malformed_json_escapes_are_rejected_before_the_tool_prefix_can_run_away() {
        assert_eq!(parse_json_string("\"summar\\\ny", 0), StringStatus::Invalid);
        assert_eq!(parse_json_string("\"bad\\q", 0), StringStatus::Invalid);
        assert_eq!(
            parse_json_string("\"unfinished\\", 0),
            StringStatus::Incomplete(None)
        );
        assert_eq!(
            parse_json_string("\"unfinished\\u12", 0),
            StringStatus::Incomplete(None)
        );
        assert_eq!(
            parse_json_string("\"snowman \\u2603\"", 0),
            StringStatus::Complete("snowman ☃".to_string(), 16)
        );

        let constraint =
            NativeToolConstraint::compile(NativeToolConstraintMode::ToolRequired, &tools())
                .unwrap()
                .unwrap();
        let malformed = concat!(
            "<tool_call>{\"name\":\"submit_review\",\"arguments\":{",
            "\"verdict\":\"pass\",\"notes\":[],\"detai\\\nl\":\"runaway"
        );
        assert!(!constraint.output_prefix_is_valid(malformed, false));
    }

    #[test]
    fn complete_nonterminal_tool_body_forces_only_the_missing_close_suffix() {
        let constraint =
            NativeToolConstraint::compile(NativeToolConstraintMode::ToolsAllowed, &tools())
                .unwrap()
                .unwrap();
        let body = "<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"verdict\":\"pass\",\"notes\":[]}}";
        assert_eq!(
            constraint.unclosed_tool_call_close_remainder(body),
            Some(TOOL_CALL_CLOSE.to_string())
        );
        assert_eq!(
            constraint.unclosed_tool_call_close_remainder(&format!("{body}\n")),
            Some(TOOL_CALL_CLOSE.to_string())
        );
        assert_eq!(
            constraint.unclosed_tool_call_close_remainder(&format!("{body}</tool")),
            Some("_call>".to_string())
        );
        assert_eq!(
            constraint.unclosed_tool_call_close_remainder(&format!("{body}{TOOL_CALL_CLOSE}")),
            None
        );
    }

    #[test]
    fn bounded_string_at_its_limit_forces_a_unique_valid_structural_suffix() {
        let constraint =
            NativeToolConstraint::compile(NativeToolConstraintMode::ToolsAllowed, &tools())
                .unwrap()
                .unwrap();
        let body = "<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"verdict\":\"pass\",\"notes\":[],\"detail\":\"12345678";
        assert_eq!(
            constraint.bounded_string_structural_remainder(body),
            Some("\"}}</tool_call>".to_string())
        );
        assert_eq!(
            constraint.bounded_string_structural_remainder(&body.replace("12345678", "1234")),
            None
        );
    }

    #[test]
    fn bounded_file_content_stops_as_a_truncated_named_mutation() {
        let constraint = NativeToolConstraint::compile(
            NativeToolConstraintMode::ToolsAllowed,
            &mutation_tools(),
        )
        .unwrap()
        .unwrap();
        let body = "<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"game.js\",\"content\":\"12345678";

        assert_eq!(
            constraint.bounded_mutation_string_limit_tool(body),
            Some("write_file".to_string())
        );
        assert_eq!(
            constraint.bounded_mutation_string_limit_tool(&body.replace("12345678", "1234")),
            None
        );
    }

    #[test]
    fn mutation_constraint_rejects_the_invalid_payload_closing_quote() {
        let mut mutation_tools = mutation_tools();
        mutation_tools[0].input_schema["properties"]["content"]["maxLength"] = json!(256);
        let snapshot = pb_control_collar::mutation::WorkspaceSnapshot::default();
        let gate = mutation_completion_gate(
            ToolDialect::QwenJson,
            NativeToolConstraintMode::ToolsAllowed,
            &mutation_tools,
            &[],
            Some(&snapshot),
        )
        .unwrap();
        let constraint = NativeToolConstraint::compile_with_mutation_gate(
            NativeToolConstraintMode::ToolsAllowed,
            &mutation_tools,
            &[],
            gate,
        )
        .unwrap()
        .unwrap();
        let invalid_open = concat!(
            "<tool_call>{\"name\":\"write_file\",\"arguments\":{",
            "\"path\":\"lib.rs\",\"content\":\"pub fn broken( {"
        );
        assert!(constraint.output_prefix_is_valid(invalid_open, false));
        assert!(!constraint.output_prefix_is_valid(&format!("{invalid_open}\""), false));
        let impossible_open = invalid_open.replace("pub fn broken( {", "pub fn broken() { ]");
        assert!(!constraint.output_prefix_is_valid(&impossible_open, false));
        assert_eq!(
            constraint.output_mutation_rejection(&impossible_open),
            Some(RejectionCode::InvalidPrefix)
        );

        let valid_open = invalid_open.replace("pub fn broken( {", "pub fn ok() {}");
        assert!(constraint.output_prefix_is_valid(&format!("{valid_open}\""), false));
        let valid_call = format!("{valid_open}\"}}}}{TOOL_CALL_CLOSE}");
        assert!(constraint.output_prefix_is_valid(&valid_call, false));
        assert!(!constraint.output_prefix_is_valid(&format!("{valid_call}{valid_call}"), false));
        assert!(!constraint.output_prefix_is_valid(
            "<tool_call>{\"name\":\"write_file\",\"arguments\":{\"content\"",
            false
        ));
    }

    #[test]
    fn pathless_bound_mutation_streams_its_first_payload_field() {
        let mut tools = mutation_tools();
        tools[0].name = "replace_file".to_string();
        tools[0].input_schema["properties"]["content"]["maxLength"] = json!(256);
        tools[0].input_schema["required"] = json!(["content"]);
        let path = pb_control_collar::mutation::LogicalPath::parse("src/lib.rs").unwrap();
        let snapshot = pb_control_collar::mutation::WorkspaceSnapshot::new(vec![
            pb_control_collar::mutation::SnapshotEntry::new(
                path.clone(),
                b"pub fn before() {}\n".to_vec(),
            ),
        ])
        .unwrap()
        .with_bound_mutation_path(path);
        let gate = mutation_completion_gate(
            ToolDialect::QwenJson,
            NativeToolConstraintMode::ToolsAllowed,
            &tools,
            &[],
            Some(&snapshot),
        )
        .unwrap();
        let constraint = NativeToolConstraint::compile_with_mutation_gate(
            NativeToolConstraintMode::ToolsAllowed,
            &tools,
            &[],
            gate,
        )
        .unwrap()
        .unwrap();

        let valid_open = concat!(
            "<tool_call>{\"name\":\"replace_file\",\"arguments\":{",
            "\"content\":\"pub fn after() {}"
        );
        assert!(constraint.output_prefix_is_valid(valid_open, false));
        assert!(
            constraint
                .output_prefix_is_valid(&format!("{valid_open}\"}}}}{TOOL_CALL_CLOSE}"), false)
        );
        let closed_body = format!("{}\"", valid_open.strip_prefix(TOOL_CALL_OPEN).unwrap());
        let (_, closed) = constraint.closed_mutation_payload(&closed_body).unwrap();
        assert_eq!(closed.get("path"), None);
        assert_eq!(closed.get("content"), Some(&json!("pub fn after() {}")));

        let alternate_path = concat!(
            "<tool_call>{\"name\":\"replace_file\",\"arguments\":{",
            "\"path\":\"model/supplied/alternate.rs\",",
            "\"content\":\"pub fn after() {}\"}}</tool_call>"
        );
        assert!(constraint.output_prefix_is_valid(alternate_path, false));

        let invalid_open = valid_open.replace("pub fn after() {}", "pub fn after() { ]");
        assert!(!constraint.output_prefix_is_valid(&invalid_open, false));
        assert_eq!(
            constraint.output_mutation_rejection(&invalid_open),
            Some(RejectionCode::InvalidPrefix)
        );
    }

    #[test]
    fn semantic_provider_rejects_the_payload_quote_while_the_prefix_remains_repairable() {
        let mut mutation_tools = mutation_tools();
        mutation_tools[0].input_schema["properties"]["content"]["maxLength"] = json!(256);
        let snapshot = pb_control_collar::mutation::WorkspaceSnapshot::default();
        let gate = mutation_completion_gate(
            ToolDialect::QwenJson,
            NativeToolConstraintMode::ToolsAllowed,
            &mutation_tools,
            &[],
            Some(&snapshot),
        )
        .unwrap();
        let mut constraint = NativeToolConstraint::compile_with_mutation_gate(
            NativeToolConstraintMode::ToolsAllowed,
            &mutation_tools,
            &[],
            gate,
        )
        .unwrap()
        .unwrap();
        constraint.set_semantic_provider(Some(crate::inference::SemanticBoundaryControl::new(
            RejectSemantics,
        )));
        let open = concat!(
            "<tool_call>{\"name\":\"write_file\",\"arguments\":{",
            "\"path\":\"lib.rs\",\"content\":\"pub fn value() -> i32 { 1 }"
        );
        assert!(constraint.output_prefix_is_valid(open, false));
        let closed = format!("{open}\"");
        assert!(!constraint.output_prefix_is_valid(&closed, false));
        assert_eq!(
            constraint.output_mutation_rejection(&closed),
            Some(RejectionCode::InvalidSemantics)
        );
    }

    #[test]
    fn deepseek_semantic_provider_observes_the_same_payload_quote_boundary() {
        let mut mutation_tools = mutation_tools();
        mutation_tools[0].input_schema["properties"]["content"]["maxLength"] = json!(256);
        let snapshot = pb_control_collar::mutation::WorkspaceSnapshot::default();
        let mut constraint = DeepSeekToolConstraint::compile(
            NativeToolConstraintMode::ToolsAllowed,
            &mutation_tools,
            &[],
            Some(&snapshot),
            None,
        )
        .unwrap()
        .unwrap();
        constraint.semantic_provider = Some(crate::inference::SemanticBoundaryControl::new(
            RejectSemantics,
        ));
        let transcript = format!(
            "{}{}write_file\">{}path\" string=\"true\">lib.rs{}{}content\" string=\"false\">\"pub fn value() -> i32 {{ 1 }}\"",
            pb_control_collar::protocol::dsml::CALLS_OPEN,
            pb_control_collar::protocol::dsml::INVOKE_OPEN,
            pb_control_collar::protocol::dsml::PARAMETER_OPEN,
            pb_control_collar::protocol::dsml::PARAMETER_CLOSE,
            pb_control_collar::protocol::dsml::PARAMETER_OPEN,
        );
        assert!(constraint.collar.probe(transcript.as_bytes(), false).valid);
        assert!(!constraint.semantic_payload_is_allowed(transcript.as_bytes()));
    }

    #[test]
    fn unsupported_schema_fails_before_generation() {
        let supported = tools();
        assert!(
            NativeToolConstraint::compile(NativeToolConstraintMode::ToolsAllowed, &supported)
                .is_ok()
        );
        let mut unsupported = supported;
        unsupported[0].input_schema["oneOf"] = json!([]);
        assert!(
            NativeToolConstraint::compile(NativeToolConstraintMode::ToolsAllowed, &unsupported)
                .is_err()
        );

        let mut wrong_enum = tools();
        wrong_enum[0].input_schema["properties"]["verdict"]["enum"] = json!([1]);
        assert!(
            NativeToolConstraint::compile(NativeToolConstraintMode::ToolsAllowed, &wrong_enum)
                .is_err()
        );

        let mut impossible_bounds = tools();
        impossible_bounds[0].input_schema["properties"]["detail"]["minLength"] = json!(9);
        assert!(
            NativeToolConstraint::compile(
                NativeToolConstraintMode::ToolsAllowed,
                &impossible_bounds
            )
            .is_err()
        );
    }
}
