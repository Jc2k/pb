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
    InvalidArgumentValue,
    UnknownArgument,
    PreconditionUnmet,
    ReadRequired,
    TargetNotFound,
    PolicyDenied,
    ApprovalDenied,
    Timeout,
    Cancelled,
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
    } else if lower.contains("cancelled by user") || lower.contains("canceled by user") {
        ToolFailureReason::Cancelled
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
            if let Some(message) = schema_value_violation(value, field_schema, field) {
                return Some(ArgumentIssue {
                    reason: ToolFailureReason::InvalidArgumentValue,
                    message,
                    suggested_next_action: format!(
                        "Call the same tool again with '{field}' inside the bounds and values declared by the exposed schema; the rejected action was not executed."
                    ),
                });
            }
        }
    }
    None
}

/// Validate that a dynamic tool schema uses only the JSON Schema subset enforced by
/// `validate_arguments`. Rejecting unsupported validation keywords is safer than exposing a schema
/// to the model while silently accepting arguments that violate it at execution time.
pub(crate) fn validate_supported_schema(schema: &Value) -> Result<(), String> {
    validate_supported_schema_at(schema, "$")?;
    if !schema_declares_type(schema, "object") {
        return Err("tool input schema at $ must declare type 'object'".to_string());
    }
    Ok(())
}

fn validate_supported_schema_at(schema: &Value, path: &str) -> Result<(), String> {
    let object = schema
        .as_object()
        .ok_or_else(|| format!("schema at {path} must be an object"))?;
    const SUPPORTED: &[&str] = &[
        "type",
        "description",
        "title",
        "default",
        "enum",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
    ];
    for keyword in object.keys() {
        if !SUPPORTED.contains(&keyword.as_str()) {
            return Err(format!(
                "schema at {path} uses unsupported validation keyword '{keyword}'"
            ));
        }
    }
    const TYPES: &[&str] = &[
        "string", "integer", "number", "boolean", "array", "object", "null",
    ];
    if let Some(schema_type) = object.get("type") {
        let types = match schema_type {
            Value::String(kind) => vec![kind.as_str()],
            Value::Array(kinds) if !kinds.is_empty() => kinds
                .iter()
                .map(|kind| {
                    kind.as_str()
                        .ok_or_else(|| format!("schema type array at {path} must contain strings"))
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => {
                return Err(format!(
                    "schema type at {path} must be a string or non-empty string array"
                ));
            }
        };
        for kind in types {
            if !TYPES.contains(&kind) {
                return Err(format!(
                    "schema at {path} declares unsupported type '{kind}'"
                ));
            }
        }
    }
    for keyword in ["description", "title"] {
        if object.get(keyword).is_some_and(|value| !value.is_string()) {
            return Err(format!("schema {keyword} at {path} must be a string"));
        }
    }
    if let Some(properties) = object.get("properties") {
        let properties = properties
            .as_object()
            .ok_or_else(|| format!("schema properties at {path} must be an object"))?;
        for (name, child) in properties {
            validate_supported_schema_at(child, &format!("{path}.properties.{name}"))?;
        }
    }
    if let Some(required) = object.get("required") {
        let required = required
            .as_array()
            .ok_or_else(|| format!("schema required at {path} must be an array"))?;
        let properties = object
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("schema required at {path} needs object properties"))?;
        for field in required {
            let field = field
                .as_str()
                .ok_or_else(|| format!("schema required at {path} must contain strings"))?;
            if !properties.contains_key(field) {
                return Err(format!(
                    "schema required field '{field}' at {path} is not declared in properties"
                ));
            }
        }
    }
    if let Some(items) = object.get("items") {
        validate_supported_schema_at(items, &format!("{path}.items"))?;
    }
    if let Some(additional) = object.get("additionalProperties")
        && !additional.is_boolean()
    {
        return Err(format!(
            "schema additionalProperties at {path} must be a boolean"
        ));
    }
    for keyword in ["minLength", "maxLength", "minItems", "maxItems"] {
        if object
            .get(keyword)
            .is_some_and(|value| value.as_u64().is_none())
        {
            return Err(format!(
                "schema {keyword} at {path} must be a non-negative integer"
            ));
        }
    }
    for (minimum, maximum) in [("minLength", "maxLength"), ("minItems", "maxItems")] {
        if let (Some(minimum), Some(maximum)) = (
            object.get(minimum).and_then(Value::as_u64),
            object.get(maximum).and_then(Value::as_u64),
        ) && minimum > maximum
        {
            return Err(format!(
                "schema bounds at {path} have minimum above maximum"
            ));
        }
    }
    for keyword in ["minimum", "maximum"] {
        if object.get(keyword).is_some_and(|value| !value.is_number()) {
            return Err(format!("schema {keyword} at {path} must be a number"));
        }
    }
    if let (Some(minimum), Some(maximum)) = (
        object.get("minimum").and_then(Value::as_f64),
        object.get("maximum").and_then(Value::as_f64),
    ) && minimum > maximum
    {
        return Err(format!(
            "schema numeric bounds at {path} have minimum above maximum"
        ));
    }
    if let Some(values) = object.get("enum")
        && !values.is_array()
    {
        return Err(format!("schema enum at {path} must be an array"));
    }
    Ok(())
}

fn schema_declares_type(schema: &Value, expected: &str) -> bool {
    match schema.get("type") {
        Some(Value::String(kind)) => kind == expected,
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some(expected)),
        _ => false,
    }
}

fn schema_value_violation(value: &Value, schema: &Value, path: &str) -> Option<String> {
    if let Some(values) = schema.get("enum").and_then(Value::as_array)
        && !values.contains(value)
    {
        return Some(format!(
            "argument '{path}' is not one of the declared enum values"
        ));
    }
    if let Some(text) = value.as_str() {
        let chars = text.chars().count() as u64;
        if schema
            .get("maxLength")
            .and_then(Value::as_u64)
            .is_some_and(|maximum| chars > maximum)
        {
            return Some(format!(
                "argument '{path}' has {chars} characters and exceeds maxLength {}",
                schema["maxLength"]
            ));
        }
        if schema
            .get("minLength")
            .and_then(Value::as_u64)
            .is_some_and(|minimum| chars < minimum)
        {
            return Some(format!(
                "argument '{path}' has {chars} characters and is below minLength {}",
                schema["minLength"]
            ));
        }
    }
    if let Some(number) = value.as_f64() {
        if schema
            .get("minimum")
            .and_then(Value::as_f64)
            .is_some_and(|minimum| number < minimum)
        {
            return Some(format!("argument '{path}' is below its declared minimum"));
        }
        if schema
            .get("maximum")
            .and_then(Value::as_f64)
            .is_some_and(|maximum| number > maximum)
        {
            return Some(format!("argument '{path}' exceeds its declared maximum"));
        }
    }
    if let Some(values) = value.as_array() {
        if schema
            .get("maxItems")
            .and_then(Value::as_u64)
            .is_some_and(|maximum| values.len() as u64 > maximum)
        {
            return Some(format!("argument '{path}' exceeds its declared maxItems"));
        }
        if schema
            .get("minItems")
            .and_then(Value::as_u64)
            .is_some_and(|minimum| (values.len() as u64) < minimum)
        {
            return Some(format!("argument '{path}' is below its declared minItems"));
        }
        let Some(items) = schema.get("items") else {
            return None;
        };
        for (index, value) in values.iter().enumerate() {
            if !value_matches_schema_type(value, items) {
                return Some(format!(
                    "argument '{path}[{index}]' must be {}",
                    schema_type_label(items)
                ));
            }
            if let Some(issue) = schema_value_violation(value, items, &format!("{path}[{index}]")) {
                return Some(issue);
            }
        }
    }
    if let Some(object) = value.as_object()
        && let Some(properties) = schema.get("properties").and_then(Value::as_object)
    {
        for required in schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if !object.contains_key(required) {
                return Some(format!(
                    "argument '{path}' is missing required property '{required}'"
                ));
            }
        }
        for (field, value) in object {
            let child_path = format!("{path}.{field}");
            let Some(child_schema) = properties.get(field) else {
                if schema.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
                    return Some(format!(
                        "argument '{path}' contains undeclared property '{field}'"
                    ));
                }
                continue;
            };
            if !value_matches_schema_type(value, child_schema) {
                return Some(format!(
                    "argument '{child_path}' must be {}",
                    schema_type_label(child_schema)
                ));
            }
            if let Some(issue) = schema_value_violation(value, child_schema, &child_path) {
                return Some(issue);
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
        _ => false,
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
        _ => false,
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
    fn cancellation_has_a_distinct_failure_reason() {
        assert_eq!(
            classify_error("run_command cancelled by user"),
            ToolFailureReason::Cancelled
        );
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
    fn schema_validation_enforces_string_bounds_and_nested_values() {
        let schema = json!({
            "type": "object",
            "properties": {
                "content": {"type": "string", "maxLength": 4},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"status": {"type": "string", "enum": ["pass"]}},
                        "required": ["status"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["content", "items"],
            "additionalProperties": false
        });
        assert_eq!(
            validate_arguments(&schema, &json!({"content": "large", "items": []}))
                .unwrap()
                .reason,
            ToolFailureReason::InvalidArgumentValue
        );
        assert_eq!(
            validate_arguments(
                &schema,
                &json!({"content": "okay", "items": [{"status": "fail"}]})
            )
            .unwrap()
            .reason,
            ToolFailureReason::InvalidArgumentValue
        );
        assert!(
            validate_arguments(
                &schema,
                &json!({"content": "okay", "items": [{"status": "pass"}]})
            )
            .is_none()
        );
    }

    #[test]
    fn dynamic_schema_rejects_keywords_the_runtime_cannot_enforce() {
        let error = validate_supported_schema(&json!({
            "type": "object",
            "properties": {
                "target": {"type": "string", "pattern": "^[a-z]+$"}
            }
        }))
        .unwrap_err();
        assert!(error.contains("unsupported validation keyword 'pattern'"));
    }

    #[test]
    fn dynamic_schema_rejects_unknown_types_and_invalid_required_fields() {
        let unknown_type = validate_supported_schema(&json!({
            "type": "object",
            "properties": {"target": {"type": "path"}}
        }))
        .unwrap_err();
        assert!(unknown_type.contains("unsupported type 'path'"));

        let invalid_required = validate_supported_schema(&json!({
            "type": "object",
            "properties": {},
            "required": ["missing"]
        }))
        .unwrap_err();
        assert!(invalid_required.contains("is not declared in properties"));
    }

    #[test]
    fn schema_validation_enforces_array_bounds() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}, "maxItems": 1}
            },
            "required": ["items"],
            "additionalProperties": false
        });
        validate_supported_schema(&schema).unwrap();
        assert_eq!(
            validate_arguments(&schema, &json!({"items": ["one", "two"]}))
                .unwrap()
                .reason,
            ToolFailureReason::InvalidArgumentValue
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
