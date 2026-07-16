use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) const MAX_TOOL_FAILURE_ENVELOPE_CHARS: usize = 2_400;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ToolFailureReason {
    UnknownTool,
    ToolNotExposed,
    MissingArgument,
    InvalidArgumentType,
    UnknownArgument,
    PreconditionUnmet,
    ReadRequired,
    TargetNotFound,
    PolicyDenied,
    ApprovalDenied,
    Timeout,
    ToolUnavailable,
    ExecutionFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArgumentIssue {
    pub(crate) reason: ToolFailureReason,
    pub(crate) message: String,
    pub(crate) suggested_next_action: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ToolFailureEnvelope {
    #[serde(rename = "type")]
    pub(crate) type_name: String,
    pub(crate) reason_code: ToolFailureReason,
    pub(crate) tool: String,
    pub(crate) message: String,
    pub(crate) retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) valid_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) suggested_tool: Option<String>,
    pub(crate) suggested_next_action: String,
}

pub(crate) fn render_tool_failure(
    reason: ToolFailureReason,
    tool: &str,
    message: &str,
    retryable: bool,
    valid_signature: Option<&str>,
    suggested_tool: Option<&str>,
    suggested_next_action: &str,
) -> String {
    let mut envelope = ToolFailureEnvelope {
        type_name: "tool_failure".to_string(),
        reason_code: reason,
        tool: sanitize_bounded(tool, 80),
        message: sanitize_bounded(message, 280),
        retryable,
        valid_signature: valid_signature.map(|value| sanitize_bounded(value, 420)),
        suggested_tool: suggested_tool.map(|value| sanitize_bounded(value, 80)),
        suggested_next_action: sanitize_bounded(suggested_next_action, 360),
    };
    let mut rendered = serde_json::to_string(&envelope)
        .expect("serializing a bounded tool failure envelope cannot fail");
    if rendered.chars().count() > MAX_TOOL_FAILURE_ENVELOPE_CHARS {
        envelope.message = sanitize_bounded(message, 120);
        envelope.valid_signature = valid_signature.map(|value| sanitize_bounded(value, 220));
        envelope.suggested_next_action = sanitize_bounded(suggested_next_action, 160);
        rendered = serde_json::to_string(&envelope)
            .expect("serializing a reduced tool failure envelope cannot fail");
    }
    debug_assert!(rendered.chars().count() <= MAX_TOOL_FAILURE_ENVELOPE_CHARS);
    rendered
}

pub(crate) fn is_tool_failure_envelope(value: &str) -> bool {
    serde_json::from_str::<ToolFailureEnvelope>(value)
        .is_ok_and(|envelope| envelope.type_name == "tool_failure")
}

pub(crate) fn classify_error(message: &str) -> ToolFailureReason {
    let lower = message.to_ascii_lowercase();
    if lower.contains("must read") || lower.contains("without reading") {
        ToolFailureReason::ReadRequired
    } else if lower.contains("not found") || lower.contains("does not exist") {
        ToolFailureReason::TargetNotFound
    } else if lower.contains("timed out") || lower.contains("timeout") {
        ToolFailureReason::Timeout
    } else if lower.contains("denied by policy") {
        ToolFailureReason::PolicyDenied
    } else if lower.contains("not approved") {
        ToolFailureReason::ApprovalDenied
    } else if lower.contains("not available") || lower.contains("unavailable") {
        ToolFailureReason::ToolUnavailable
    } else {
        ToolFailureReason::ExecutionFailed
    }
}

pub(crate) fn schema_signature(name: &str, schema: &Value) -> String {
    let required_order = schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    let required = required_order.iter().copied().collect::<HashSet<_>>();
    let arguments = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            let mut fields = required_order
                .iter()
                .filter_map(|field| properties.get(*field).map(|schema| (*field, schema)))
                .collect::<Vec<_>>();
            fields.extend(
                properties
                    .iter()
                    .filter(|(field, _)| !required.contains(field.as_str()))
                    .map(|(field, schema)| (field.as_str(), schema)),
            );
            fields
                .into_iter()
                .map(|(field, field_schema)| {
                    let optional = if required.contains(field) { "" } else { "?" };
                    format!("{field}{optional}: {}", schema_type_label(field_schema))
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!("{name}({arguments})")
}

pub(crate) fn validate_arguments(schema: &Value, arguments: &Value) -> Option<ArgumentIssue> {
    let Some(object) = arguments.as_object() else {
        return Some(ArgumentIssue {
            reason: ToolFailureReason::InvalidArgumentType,
            message: "tool arguments must be a JSON object".to_string(),
            suggested_next_action: "Call the tool again with one JSON object matching the valid signature; no arguments were guessed or executed.".to_string(),
        });
    };
    let properties = schema.get("properties").and_then(Value::as_object);
    for field in schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        if !object.contains_key(field) {
            return Some(ArgumentIssue {
                reason: ToolFailureReason::MissingArgument,
                message: format!("required argument '{field}' is missing"),
                suggested_next_action: format!(
                    "Call the same tool again with '{field}' supplied using the valid signature; pb did not invent a value."
                ),
            });
        }
    }
    if let Some(properties) = properties {
        for (field, value) in object {
            let Some(field_schema) = properties.get(field) else {
                if schema.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
                    return Some(ArgumentIssue {
                        reason: ToolFailureReason::UnknownArgument,
                        message: format!("argument '{field}' is not declared by this tool"),
                        suggested_next_action: format!(
                            "Remove '{field}' and call the tool with only fields from the valid signature; pb did not reinterpret it."
                        ),
                    });
                }
                continue;
            };
            if !value_matches_schema_type(value, field_schema) {
                return Some(ArgumentIssue {
                    reason: ToolFailureReason::InvalidArgumentType,
                    message: format!(
                        "argument '{field}' must be {}, not {}",
                        schema_type_label(field_schema),
                        value_type_label(value)
                    ),
                    suggested_next_action: format!(
                        "Call the same tool again with '{field}' using the type in the valid signature; pb did not coerce or execute the value."
                    ),
                });
            }
        }
    }
    None
}

pub(crate) fn nearest_tool<'a>(requested: &str, exposed: &'a [&str]) -> Option<&'a str> {
    let mut candidates = exposed
        .iter()
        .copied()
        .map(|candidate| (edit_distance(requested, candidate), candidate))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.cmp(right));
    let (distance, candidate) = candidates.into_iter().next()?;
    let threshold = requested.chars().count().max(candidate.chars().count()) / 3;
    (distance > 0 && distance <= 2 && distance <= threshold.max(1)).then_some(candidate)
}

fn value_matches_schema_type(value: &Value, schema: &Value) -> bool {
    let Some(schema_type) = schema.get("type") else {
        return true;
    };
    match schema_type {
        Value::String(kind) => value_matches_type(value, kind),
        Value::Array(kinds) => kinds
            .iter()
            .filter_map(Value::as_str)
            .any(|kind| value_matches_type(value, kind)),
        _ => true,
    }
}

fn value_matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn schema_type_label(schema: &Value) -> String {
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        let labels = values
            .iter()
            .filter_map(Value::as_str)
            .map(|value| format!("'{value}'"))
            .collect::<Vec<_>>();
        if !labels.is_empty() {
            return labels.join(" | ");
        }
    }
    match schema.get("type") {
        Some(Value::String(kind)) => kind.clone(),
        Some(Value::Array(kinds)) => kinds
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" | "),
        _ => "value".to_string(),
    }
}

fn value_type_label(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn sanitize_bounded(value: &str, max_chars: usize) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .take(max_chars)
        .collect()
}

fn edit_distance(left: &str, right: &str) -> usize {
    let right = right.chars().collect::<Vec<_>>();
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_character) in left.chars().enumerate() {
        let mut current = vec![left_index + 1];
        for (right_index, right_character) in right.iter().enumerate() {
            current.push(
                (previous[right_index + 1] + 1)
                    .min(current[right_index] + 1)
                    .min(previous[right_index] + usize::from(left_character != *right_character)),
            );
        }
        previous = current;
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ar1_nearest_tool_suggests_but_does_not_rewrite() {
        let tools = ["read_file", "ripgrep", "write_file"];
        assert_eq!(nearest_tool("read_flie", &tools), Some("read_file"));
        assert_eq!(nearest_tool("totally_different", &tools), None);
    }

    #[test]
    fn ar2_schema_validation_reports_exact_required_and_typed_arguments() {
        let schema = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer"}
            },
            "required": ["path"],
            "additionalProperties": false
        });
        assert_eq!(
            schema_signature("read_file", &schema),
            "read_file(path: string, start?: integer)"
        );
        assert_eq!(
            validate_arguments(&schema, &json!({})).unwrap().reason,
            ToolFailureReason::MissingArgument
        );
        assert_eq!(
            validate_arguments(&schema, &json!({"path": 7}))
                .unwrap()
                .reason,
            ToolFailureReason::InvalidArgumentType
        );
    }

    #[test]
    fn long_failure_envelopes_are_bounded_valid_json() {
        let rendered = render_tool_failure(
            ToolFailureReason::ExecutionFailed,
            &"x".repeat(10_000),
            &"error \\\"".repeat(10_000),
            true,
            Some(&"field: string, ".repeat(10_000)),
            None,
            &"retry \\\"".repeat(10_000),
        );
        assert!(rendered.chars().count() <= MAX_TOOL_FAILURE_ENVELOPE_CHARS);
        let envelope: ToolFailureEnvelope = serde_json::from_str(&rendered).unwrap();
        assert_eq!(envelope.type_name, "tool_failure");
    }
}
