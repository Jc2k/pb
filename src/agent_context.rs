use std::fmt;

use anyhow::Result;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::agent_core::{AgentToolCall, ChatMessage, PromptToolResultMetadata};

pub(crate) const CONTEXT_SAFETY_MARGIN_TOKENS: usize = 32;
const PROMPT_SOFT_LIMIT_PERCENT: usize = 70;
const PROMPT_COMPACTION_TARGET_PERCENT: usize = 60;
const RECEIPT_EXCERPT_CHARS: usize = 320;
const MAX_TOOL_RESULT_PROMPT_CHARS: usize = 16_000;
const MIN_TOOL_RESULT_PROMPT_CHARS: usize = 512;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PromptMeasurement {
    pub prompt_tokens: usize,
    pub tool_schema_tokens: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedPrompt {
    pub messages: Vec<ChatMessage>,
    pub measurement: PromptMeasurement,
    pub context_capacity: usize,
    pub reserved_generation_tokens: usize,
    pub safety_margin_tokens: usize,
    pub usable_prompt_capacity: usize,
    pub compacted_messages: usize,
    pub omitted_tool_result_chars: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContextLimitError {
    pub context_capacity: usize,
    pub reserved_generation_tokens: usize,
    pub safety_margin_tokens: usize,
    pub usable_prompt_capacity: usize,
    pub measured_prompt_tokens: usize,
    pub largest_sections: Vec<(String, usize)>,
}

impl fmt::Display for ContextLimitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sections = self
            .largest_sections
            .iter()
            .map(|(label, chars)| format!("{label}={chars} chars"))
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            formatter,
            "context limit: measured prompt {} tokens exceeds usable prompt capacity {} (context {}, generation reserve {}, safety margin {}); largest prompt sections: {}",
            self.measured_prompt_tokens,
            self.usable_prompt_capacity,
            self.context_capacity,
            self.reserved_generation_tokens,
            self.safety_margin_tokens,
            sections
        )
    }
}

impl std::error::Error for ContextLimitError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BoundedPromptResult {
    pub content: String,
    pub omitted_chars: usize,
    pub omitted_bytes: usize,
    pub omitted_lines: usize,
}

pub(crate) fn usable_prompt_capacity(
    context_capacity: usize,
    reserved_generation_tokens: usize,
) -> usize {
    context_capacity
        .saturating_sub(reserved_generation_tokens)
        .saturating_sub(CONTEXT_SAFETY_MARGIN_TOKENS)
}

pub(crate) fn tool_result_char_budget(
    context_capacity: usize,
    reserved_generation_tokens: usize,
) -> usize {
    usable_prompt_capacity(context_capacity, reserved_generation_tokens)
        .saturating_mul(3)
        .saturating_div(8)
        .clamp(MIN_TOOL_RESULT_PROMPT_CHARS, MAX_TOOL_RESULT_PROMPT_CHARS)
}

pub(crate) fn bound_tool_result_for_prompt(result: &str, max_chars: usize) -> BoundedPromptResult {
    let raw_chars = result.chars().count();
    if raw_chars <= max_chars {
        return BoundedPromptResult {
            content: result.to_string(),
            omitted_chars: 0,
            omitted_bytes: 0,
            omitted_lines: 0,
        };
    }

    let sha256 = format!("{:x}", Sha256::digest(result.as_bytes()));
    let raw_lines = result.lines().count();
    let marker_reserve = 256.min(max_chars.saturating_div(2).max(1));
    let available = max_chars.saturating_sub(marker_reserve);
    let suffix_chars = available / 3;
    let suffix = result
        .chars()
        .rev()
        .take(suffix_chars)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    let mut prefix_chars = available.saturating_sub(suffix_chars);
    loop {
        let prefix = result.chars().take(prefix_chars).collect::<String>();
        let kept_chars = prefix
            .chars()
            .count()
            .saturating_add(suffix.chars().count());
        let omitted_chars = raw_chars.saturating_sub(kept_chars);
        let kept_bytes = prefix.len().saturating_add(suffix.len());
        let omitted_bytes = result.len().saturating_sub(kept_bytes);
        let kept_lines = prefix
            .lines()
            .count()
            .saturating_add(suffix.lines().count());
        let omitted_lines = raw_lines.saturating_sub(kept_lines);
        let marker = format!(
            "\n\n[tool result shortened for prompt: omitted {omitted_chars} chars/{omitted_bytes} bytes/{omitted_lines} lines; raw_sha256={sha256}]\n\n"
        );
        let content = format!("{prefix}{marker}{suffix}");
        let rendered_chars = content.chars().count();
        if rendered_chars <= max_chars || prefix_chars == 0 {
            return BoundedPromptResult {
                content,
                omitted_chars,
                omitted_bytes,
                omitted_lines,
            };
        }
        prefix_chars = prefix_chars.saturating_sub(rendered_chars.saturating_sub(max_chars));
    }
}

pub(crate) fn normalized_arguments_sha256(arguments: &Value) -> String {
    let mut canonical = String::new();
    write_canonical_json(arguments, &mut canonical);
    format!("{:x}", Sha256::digest(canonical.as_bytes()))
}

fn write_canonical_json(value: &Value, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&value.to_string()),
        Value::String(value) => output.push_str(&serde_json::to_string(value).unwrap()),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical_json(value, output);
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key).unwrap());
                output.push(':');
                write_canonical_json(&values[key], output);
            }
            output.push('}');
        }
    }
}

pub(crate) fn prepare_prompt<F>(
    messages: &[ChatMessage],
    context_capacity: usize,
    reserved_generation_tokens: usize,
    tool_schema_chars: usize,
    mut measure: F,
) -> Result<PreparedPrompt>
where
    F: FnMut(&[ChatMessage]) -> Result<PromptMeasurement>,
{
    let usable_prompt_capacity =
        usable_prompt_capacity(context_capacity, reserved_generation_tokens);
    let soft_limit = usable_prompt_capacity.saturating_mul(PROMPT_SOFT_LIMIT_PERCENT) / 100;
    let target = usable_prompt_capacity.saturating_mul(PROMPT_COMPACTION_TARGET_PERCENT) / 100;
    let mut prepared = messages.to_vec();
    let mut measurement = measure(&prepared)?;
    let mut compacted_messages = 0usize;

    if measurement.prompt_tokens > soft_limit {
        while measurement.prompt_tokens > target {
            let groups = completed_tool_groups(&prepared);
            let Some((start, end)) = groups.first().copied() else {
                break;
            };
            let (receipt, metadata) = compacted_tool_group_receipt(&prepared[start..end]);
            compacted_messages = compacted_messages.saturating_add(end.saturating_sub(start));
            prepared.splice(
                start..end,
                [ChatMessage::context_receipt(receipt, metadata)],
            );
            measurement = measure(&prepared)?;
        }
    }

    let omitted_tool_result_chars = prepared
        .iter()
        .filter_map(|message| message.prompt_tool_result.as_ref())
        .map(|metadata| metadata.omitted_chars)
        .sum();
    if measurement.prompt_tokens > usable_prompt_capacity {
        return Err(ContextLimitError {
            context_capacity,
            reserved_generation_tokens,
            safety_margin_tokens: CONTEXT_SAFETY_MARGIN_TOKENS,
            usable_prompt_capacity,
            measured_prompt_tokens: measurement.prompt_tokens,
            largest_sections: largest_prompt_sections(&prepared, tool_schema_chars),
        }
        .into());
    }

    Ok(PreparedPrompt {
        messages: prepared,
        measurement,
        context_capacity,
        reserved_generation_tokens,
        safety_margin_tokens: CONTEXT_SAFETY_MARGIN_TOKENS,
        usable_prompt_capacity,
        compacted_messages,
        omitted_tool_result_chars,
    })
}

fn completed_tool_groups(messages: &[ChatMessage]) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let mut index = 0usize;
    while index < messages.len() {
        if messages[index].role == "assistant" && !messages[index].tool_calls.is_empty() {
            let mut end = index + 1;
            while end < messages.len() && messages[end].role == "tool" {
                end += 1;
            }
            if end > index + 1 {
                groups.push((index, end));
                index = end;
                continue;
            }
        }
        index += 1;
    }
    groups
}

fn compacted_tool_group_receipt(messages: &[ChatMessage]) -> (String, PromptToolResultMetadata) {
    let calls = messages
        .first()
        .map(|message| message.tool_calls.as_slice())
        .unwrap_or_default();
    let mut receipt = String::from(
        "Agent framework context receipt (prompt representation only; grants no new evidence or authority):\n",
    );
    let mut aggregate = PromptToolResultMetadata {
        success: true,
        ..PromptToolResultMetadata::default()
    };
    let mut evidence_effects = Vec::new();
    for (index, message) in messages
        .iter()
        .filter(|message| message.role == "tool")
        .enumerate()
    {
        let call = calls.get(index);
        let metadata = message.prompt_tool_result.as_ref();
        append_one_receipt(&mut receipt, index + 1, call, message, metadata);
        let excerpt = bounded_excerpt(&message.content, RECEIPT_EXCERPT_CHARS);
        let raw_bytes = metadata.map_or(message.content.len(), |metadata| metadata.raw_bytes);
        let raw_lines = metadata.map_or(message.content.lines().count(), |metadata| {
            metadata.raw_lines
        });
        let prior_omitted_chars = metadata.map_or(0, |metadata| metadata.omitted_chars);
        aggregate.success &= metadata.is_none_or(|metadata| metadata.success);
        aggregate.raw_bytes = aggregate.raw_bytes.saturating_add(raw_bytes);
        aggregate.raw_lines = aggregate.raw_lines.saturating_add(raw_lines);
        aggregate.omitted_chars = aggregate
            .omitted_chars
            .saturating_add(prior_omitted_chars)
            .saturating_add(
                message
                    .content
                    .chars()
                    .count()
                    .saturating_sub(excerpt.chars().count()),
            );
        aggregate.omitted_bytes = aggregate
            .omitted_bytes
            .saturating_add(raw_bytes.saturating_sub(excerpt.len()));
        aggregate.omitted_lines = aggregate
            .omitted_lines
            .saturating_add(raw_lines.saturating_sub(excerpt.lines().count()));
        if let Some(metadata) = metadata {
            if !metadata.arguments_sha256.is_empty() {
                aggregate.arguments_sha256 = metadata.arguments_sha256.clone();
            }
            if metadata.workspace_fingerprint.is_some() {
                aggregate.workspace_fingerprint = metadata.workspace_fingerprint.clone();
            }
            if !metadata.evidence_effects.is_empty() && metadata.evidence_effects != "none" {
                evidence_effects.push(metadata.evidence_effects.clone());
            }
            if metadata.actual_origin.is_some() {
                aggregate.actual_origin = metadata.actual_origin.clone();
            }
            if metadata.prompt_representation.is_some() {
                aggregate.prompt_representation = metadata.prompt_representation.clone();
            }
            if metadata.observation_coverage.is_some() {
                aggregate.observation_coverage = metadata.observation_coverage.clone();
            }
            if metadata.observation_action_id.is_some() {
                aggregate.observation_action_id = metadata.observation_action_id.clone();
            }
        }
    }
    evidence_effects.sort();
    evidence_effects.dedup();
    aggregate.evidence_effects = if evidence_effects.is_empty() {
        "none".to_string()
    } else {
        evidence_effects.join(",")
    };
    (receipt, aggregate)
}

fn append_one_receipt(
    receipt: &mut String,
    index: usize,
    call: Option<&AgentToolCall>,
    message: &ChatMessage,
    metadata: Option<&PromptToolResultMetadata>,
) {
    let tool = message
        .name
        .as_deref()
        .or_else(|| call.map(|call| call.tool.as_str()))
        .unwrap_or("unknown");
    let argument_hash = metadata
        .map(|metadata| metadata.arguments_sha256.clone())
        .or_else(|| call.map(|call| normalized_arguments_sha256(&call.arguments)))
        .unwrap_or_else(|| normalized_arguments_sha256(&Value::Null));
    let success = metadata.is_none_or(|metadata| metadata.success);
    let raw_bytes = metadata.map_or(message.content.len(), |metadata| metadata.raw_bytes);
    let raw_lines = metadata.map_or(message.content.lines().count(), |metadata| {
        metadata.raw_lines
    });
    let excerpt = bounded_excerpt(&message.content, RECEIPT_EXCERPT_CHARS);
    let excerpt_bytes = excerpt.len();
    let excerpt_lines = excerpt.lines().count();
    let omitted_bytes = raw_bytes.saturating_sub(excerpt_bytes);
    let omitted_lines = raw_lines.saturating_sub(excerpt_lines);
    let workspace_fingerprint = metadata
        .and_then(|metadata| metadata.workspace_fingerprint.as_deref())
        .unwrap_or("none");
    let evidence_effects = metadata
        .map(|metadata| metadata.evidence_effects.as_str())
        .filter(|effects| !effects.is_empty())
        .unwrap_or("none");
    let actual_origin = metadata
        .and_then(|metadata| metadata.actual_origin.as_deref())
        .unwrap_or("model");
    let prompt_representation = metadata
        .and_then(|metadata| metadata.prompt_representation.as_deref())
        .unwrap_or("native");
    let observation_coverage = metadata
        .and_then(|metadata| metadata.observation_coverage.as_deref())
        .unwrap_or("none");
    let observation_action_id = metadata
        .and_then(|metadata| metadata.observation_action_id.as_deref())
        .unwrap_or("none");
    receipt.push_str(&format!(
        "receipt[{index}]: tool={tool}; arguments_sha256={argument_hash}; outcome={}; omitted_bytes={omitted_bytes}; omitted_lines={omitted_lines}; workspace_fingerprint={workspace_fingerprint}; evidence_effects={evidence_effects}; actual_origin={actual_origin}; prompt_representation={prompt_representation}; observation_coverage={observation_coverage}; observation_action_id={observation_action_id}\nexcerpt:\n{excerpt}\n",
        if success { "success" } else { "failure" }
    ));
}

fn bounded_excerpt(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let suffix = extract_workspace_fingerprint(value)
        .map(|fingerprint| format!("\nHarness current content fingerprint: {fingerprint}"))
        .unwrap_or_default();
    let prefix_limit = max_chars.saturating_sub(suffix.chars().count());
    format!(
        "{}{}",
        value.chars().take(prefix_limit).collect::<String>(),
        suffix
    )
}

fn extract_workspace_fingerprint(value: &str) -> Option<&str> {
    value
        .rsplit_once("Harness current content fingerprint: ")
        .map(|(_, fingerprint)| fingerprint.lines().next().unwrap_or_default().trim())
        .filter(|fingerprint| !fingerprint.is_empty())
}

fn largest_prompt_sections(
    messages: &[ChatMessage],
    tool_schema_chars: usize,
) -> Vec<(String, usize)> {
    let mut sections = messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            let label = if index == 0 && message.role == "system" {
                "system_anchor".to_string()
            } else if index == 1 && message.role == "user" {
                "task_stage_anchor".to_string()
            } else {
                format!("message_{index}_{}", message.role)
            };
            (label, message.content.chars().count())
        })
        .collect::<Vec<_>>();
    if tool_schema_chars > 0 {
        sections.push(("tool_schemas".to_string(), tool_schema_chars));
    }
    sections.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    sections.truncate(4);
    sections
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn measured(messages: &[ChatMessage]) -> Result<PromptMeasurement> {
        Ok(PromptMeasurement {
            prompt_tokens: messages
                .iter()
                .map(|message| message.content.chars().count().div_ceil(4).max(1))
                .sum(),
            tool_schema_tokens: 0,
        })
    }

    #[test]
    fn normalized_argument_hash_ignores_object_key_order() {
        assert_eq!(
            normalized_arguments_sha256(&json!({"path":"a", "start":1})),
            normalized_arguments_sha256(&json!({"start":1, "path":"a"}))
        );
    }

    #[test]
    fn bounded_prompt_result_preserves_prefix_suffix_and_reports_omission() {
        let value = format!("prefix-{}-suffix", "x".repeat(2_000));
        let bounded = bound_tool_result_for_prompt(&value, 500);
        assert!(bounded.content.starts_with("prefix-"));
        assert!(bounded.content.ends_with("-suffix"));
        assert!(bounded.content.contains("raw_sha256="));
        assert!(bounded.omitted_chars > 0);
        assert!(bounded.content.chars().count() <= 500);
    }

    #[test]
    fn cb2_compacts_completed_tool_results_to_deterministic_receipts() {
        let tool_call = |id: &str| AgentToolCall {
            id: Some(id.to_string()),
            tool: "read_file".to_string(),
            arguments: json!({"path": format!("{id}.txt")}),
        };
        let tool_result = |id: &str| {
            ChatMessage::tool_result_with_metadata(
                "read_file".to_string(),
                Some(id.to_string()),
                "result line\n".repeat(300),
                PromptToolResultMetadata {
                    arguments_sha256: normalized_arguments_sha256(
                        &json!({"path": format!("{id}.txt")}),
                    ),
                    success: true,
                    raw_bytes: 3_600,
                    raw_lines: 300,
                    omitted_chars: 0,
                    omitted_bytes: 0,
                    omitted_lines: 0,
                    workspace_fingerprint: Some("abc123".to_string()),
                    evidence_effects: format!("read_path:{id}.txt"),
                    ..PromptToolResultMetadata::default()
                },
            )
        };
        let messages = vec![
            ChatMessage::text("system", "SYSTEM ANCHOR"),
            ChatMessage::text("user", "TASK ANCHOR"),
            ChatMessage::assistant_with_tool_calls("", vec![tool_call("one")]),
            tool_result("one"),
            ChatMessage::assistant_with_tool_calls("", vec![tool_call("two")]),
            tool_result("two"),
        ];
        let prepared = prepare_prompt(&messages, 1_024, 128, 400, measured).unwrap();
        assert!(prepared.compacted_messages >= 2);
        assert_eq!(prepared.messages[0].content, "SYSTEM ANCHOR");
        assert_eq!(prepared.messages[1].content, "TASK ANCHOR");
        let rendered = prepared
            .messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("arguments_sha256="));
        assert!(rendered.contains("workspace_fingerprint=abc123"));
        assert!(rendered.contains("evidence_effects=read_path:one.txt"));
    }

    #[test]
    fn truthful_controller_context_block_is_not_a_model_tool_group() {
        let messages = vec![
            ChatMessage::text("system", "SYSTEM"),
            ChatMessage::text("user", "TASK"),
            ChatMessage::context_receipt(
                "x".repeat(1_600),
                PromptToolResultMetadata {
                    arguments_sha256: normalized_arguments_sha256(&json!({"path":"small.txt"})),
                    success: true,
                    raw_bytes: 1_600,
                    raw_lines: 1,
                    workspace_fingerprint: Some("a".repeat(64)),
                    evidence_effects: "read_before_write".to_string(),
                    actual_origin: Some("controller".to_string()),
                    prompt_representation: Some("controller_block".to_string()),
                    observation_coverage: Some("full".to_string()),
                    observation_action_id: Some("controller-read".to_string()),
                    ..PromptToolResultMetadata::default()
                },
            ),
        ];
        let prepared = prepare_prompt(&messages, 2_048, 100, 0, measured).unwrap();
        assert_eq!(prepared.compacted_messages, 0);
        assert_eq!(prepared.messages.len(), messages.len());
        assert_eq!(prepared.messages[2].role, "user");
        assert!(prepared.messages[2].tool_calls.is_empty());
        assert!(prepared.messages[2].tool_call_id.is_none());
        assert_eq!(prepared.messages[2].content, messages[2].content);
        assert_eq!(
            prepared.messages[2]
                .prompt_tool_result
                .as_ref()
                .unwrap()
                .actual_origin
                .as_deref(),
            Some("controller")
        );
    }

    #[test]
    fn cb3_authoritative_anchors_survive_compaction_byte_for_byte() {
        let anchor = "TASK\naccepted_plan=plan-sha\nfingerprint=content-sha\nchecks=[cargo-test]\nterminal=submit_implementation";
        let messages = vec![
            ChatMessage::text("system", "STAGE CONTRACT\nnever weaken"),
            ChatMessage::text("user", anchor),
            ChatMessage::assistant_with_tool_calls(
                "",
                vec![AgentToolCall {
                    id: Some("read".to_string()),
                    tool: "read_file".to_string(),
                    arguments: json!({"path":"large.txt"}),
                }],
            ),
            ChatMessage::tool_result(
                "read_file".to_string(),
                Some("read".to_string()),
                "x".repeat(8_000),
            ),
        ];
        let prepared = prepare_prompt(&messages, 800, 128, 0, measured).unwrap();
        assert_eq!(prepared.messages[0].content, messages[0].content);
        assert_eq!(prepared.messages[1].content, anchor);
    }

    #[test]
    fn cb4_anchor_overflow_fails_before_generation() {
        let messages = vec![
            ChatMessage::text("system", "S".repeat(2_000)),
            ChatMessage::text("user", "U".repeat(2_000)),
        ];
        let error = prepare_prompt(&messages, 512, 128, 0, measured).unwrap_err();
        let limit = error.downcast_ref::<ContextLimitError>().unwrap();
        assert_eq!(limit.context_capacity, 512);
        assert!(limit.measured_prompt_tokens > limit.usable_prompt_capacity);
        assert_eq!(limit.largest_sections[0].1, 2_000);
    }
}
