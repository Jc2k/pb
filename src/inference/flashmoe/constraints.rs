use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::text::QwenTokenizer;
use super::types::{ChatTool, NativeToolConstraintMode};

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

#[derive(Debug, Clone)]
pub(super) struct NativeToolConstraint {
    mode: NativeToolConstraintMode,
    schemas: BTreeMap<String, Value>,
    schema_sha256: String,
    rejected_candidates: usize,
}

impl NativeToolConstraint {
    pub(super) fn compile(
        mode: NativeToolConstraintMode,
        tools: &[ChatTool],
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
        let schema_bytes = serde_json::to_vec(&schemas)?;
        Ok(Some(Self {
            mode: active_mode,
            schemas,
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            rejected_candidates: 0,
        }))
    }

    pub(super) fn mode(&self) -> NativeToolConstraintMode {
        self.mode
    }

    pub(super) fn schema_sha256(&self) -> &str {
        &self.schema_sha256
    }

    pub(super) fn rejected_candidates(&self) -> usize {
        self.rejected_candidates
    }

    pub(super) fn terminal_state(&self, decoded: &str) -> &'static str {
        if decoded.contains(TOOL_CALL_CLOSE) {
            "complete_tool_call"
        } else if decoded.contains(TOOL_CALL_OPEN) {
            "in_tool_call"
        } else {
            "before_tool_call"
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
        for (token, score) in candidates {
            let token = u32::try_from(token).context("candidate token id does not fit u32")?;
            let mut trial = generated.to_vec();
            trial.push(token);
            let decoded = tokenizer.decode(&trial)?;
            if self.output_prefix_is_valid(&decoded, tokenizer.is_eos(token)) {
                accepted.push((token as usize, score));
                if accepted.len() >= keep {
                    break;
                }
            } else {
                self.rejected_candidates = self.rejected_candidates.saturating_add(1);
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
                return !at_eos && TOOL_CALL_OPEN.starts_with(prefix);
            }
            (_, None) => return true,
            (_, Some(start)) => start,
        };

        let mut remaining = &decoded[start..];
        loop {
            if !remaining.starts_with(TOOL_CALL_OPEN) {
                return !at_eos && TOOL_CALL_OPEN.starts_with(remaining.trim_start());
            }
            remaining = &remaining[TOOL_CALL_OPEN.len()..];
            let Some(close) = remaining.find(TOOL_CALL_CLOSE) else {
                return !at_eos && self.tool_body_prefix_is_valid(remaining);
            };
            let body = &remaining[..close];
            if !self.tool_body_is_complete(body) {
                return false;
            }
            remaining = remaining[close + TOOL_CALL_CLOSE.len()..].trim_start();
            if remaining.is_empty() {
                return true;
            }
        }
    }

    fn tool_body_prefix_is_valid(&self, body: &str) -> bool {
        matches!(
            self.parse_tool_body(body),
            PrefixStatus::Incomplete | PrefixStatus::Complete(_)
        )
    }

    fn tool_body_is_complete(&self, body: &str) -> bool {
        matches!(self.parse_tool_body(body), PrefixStatus::Complete(position) if skip_ws(body, position) == body.len())
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
        position = match JsonPrefixParser::new(body).parse_value(position, schema) {
            PrefixStatus::Complete(position) => position,
            status => return status,
        };
        position = skip_ws(body, position);
        consume_byte(body, position, b'}')
    }
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
                | "maxLength"
                | "minLength"
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
        let common = matches!(key.as_str(), "type" | "description" | "enum");
        let kind_specific = match kind {
            "object" => matches!(
                key.as_str(),
                "properties" | "required" | "additionalProperties"
            ),
            "array" => key == "items",
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
    for (offset, character) in input[position + 1..].char_indices() {
        let absolute = position + 1 + offset;
        if escaped {
            escaped = false;
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
    if escaped || input[position + 1..].contains('\\') {
        StringStatus::Incomplete(None)
    } else {
        StringStatus::Incomplete(Some(input[position + 1..].to_string()))
    }
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

    fn parse_array(&self, position: usize, schema: &Value) -> PrefixStatus {
        let mut position = match consume_byte(self.input, position, b'[') {
            PrefixStatus::Complete(position) => position,
            status => return status,
        };
        let Some(items) = schema.get("items") else {
            return PrefixStatus::Invalid;
        };
        let mut can_close = true;
        loop {
            position = skip_ws(self.input, position);
            match self.input.as_bytes().get(position) {
                None => return PrefixStatus::Incomplete,
                Some(b']') if can_close => return PrefixStatus::Complete(position + 1),
                Some(b']') => return PrefixStatus::Invalid,
                _ => {}
            }
            position = match self.parse_value(position, items) {
                PrefixStatus::Complete(position) => position,
                status => return status,
            };
            position = skip_ws(self.input, position);
            match self.input.as_bytes().get(position) {
                Some(b',') => {
                    position += 1;
                    can_close = false;
                }
                Some(b']') => return PrefixStatus::Complete(position + 1),
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
                    || !enum_accepts(schema, &Value::String(value))
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
                if enum_accepts(schema, &value) && numeric_bounds_accept(schema, &value) =>
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
                return if enum_accepts(schema, &value) {
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

fn enum_accepts(schema: &Value, value: &Value) -> bool {
    schema
        .get("enum")
        .and_then(Value::as_array)
        .is_none_or(|values| values.contains(value))
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
    use serde_json::json;

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
        assert!(!constraint.output_prefix_is_valid(
            "<tool_call>{\"name\":\"submit_review\",\"arguments\":{\"verdict\":\"pass\",\"notes\":[\"one\",]}}</tool_call>",
            true
        ));
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
