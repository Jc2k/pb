use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use serde_json::{Map, Value};

use crate::{
    CollarError, CollarResult, CompletionDecision, MutationCompletionGate, RejectionCode,
    analysis::LanguageLayerStack,
    tool::{CollarManifest, ToolConstraintMode},
};

pub const CALLS_OPEN: &str = "<｜DSML｜tool_calls>";
pub const CALLS_CLOSE: &str = "</｜DSML｜tool_calls>";
pub const INVOKE_OPEN: &str = "<｜DSML｜invoke name=\"";
pub const INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
pub const PARAMETER_OPEN: &str = "<｜DSML｜parameter name=\"";
pub const PARAMETER_CLOSE: &str = "</｜DSML｜parameter>";
pub const ESCAPED_PARAMETER_CLOSE: &str = "&lt;/｜DSML｜parameter>";

const MAX_STRUCTURAL_WHITESPACE_BYTES: usize = 32;

#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalToolCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DsmlParseOutput {
    pub text: String,
    pub calls: Vec<CanonicalToolCall>,
    pub incomplete: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DsmlProbe {
    pub valid: bool,
    pub complete: bool,
    pub complete_terminal_call: bool,
    pub rejection: Option<RejectionCode>,
}

#[derive(Clone, Debug)]
pub struct DsmlConstraint {
    mode: ToolConstraintMode,
    schemas: BTreeMap<String, Value>,
    terminal_tools: BTreeSet<String>,
    mutation_gate: Option<MutationCompletionGate>,
}

impl DsmlConstraint {
    pub fn compile(manifest: CollarManifest) -> CollarResult<Self> {
        Self::compile_with_language_layers(manifest, None)
    }

    pub fn compile_with_language_layers(
        manifest: CollarManifest,
        language_layers: Option<Arc<Mutex<LanguageLayerStack>>>,
    ) -> CollarResult<Self> {
        manifest.validate()?;
        let mutation_gate = Some(match language_layers {
            Some(layers) => {
                MutationCompletionGate::with_shared_language_layers(manifest.clone(), layers)?
            }
            None => MutationCompletionGate::new(manifest.clone())?,
        });
        let mut schemas = BTreeMap::new();
        for tool in &manifest.tools {
            validate_schema(&tool.input_schema, &format!("tool {}", tool.name))?;
            if schemas
                .insert(tool.name.clone(), tool.input_schema.clone())
                .is_some()
            {
                return Err(CollarError::InvalidManifest(format!(
                    "duplicate exposed tool {:?}",
                    tool.name
                )));
            }
        }
        Ok(Self {
            mode: manifest.mode,
            schemas,
            terminal_tools: manifest.terminal_tools.into_iter().collect(),
            mutation_gate,
        })
    }

    pub fn probe(&self, transcript: &[u8], at_eos: bool) -> DsmlProbe {
        let decoded = match std::str::from_utf8(transcript) {
            Ok(decoded) => decoded,
            Err(error) if error.error_len().is_none() && !at_eos => {
                let valid = &transcript[..error.valid_up_to()];
                let Ok(decoded) = std::str::from_utf8(valid) else {
                    return invalid_probe(None);
                };
                return self.probe(decoded.as_bytes(), false);
            }
            Err(_) => return invalid_probe(None),
        };
        let Some(start) = decoded.find(CALLS_OPEN) else {
            if let Some(marker) = decoded.rfind("｜DSML｜") {
                let structural_start = decoded[..marker].rfind('<').unwrap_or(marker);
                let suffix = &decoded[structural_start..];
                if !CALLS_OPEN.starts_with(suffix) {
                    return invalid_probe(None);
                }
            }
            return DsmlProbe {
                valid: !at_eos || self.mode != ToolConstraintMode::ToolRequired,
                complete: false,
                complete_terminal_call: false,
                rejection: None,
            };
        };
        let mut parser = PrefixParser {
            input: decoded,
            position: start + CALLS_OPEN.len(),
            schemas: &self.schemas,
            terminal_tools: &self.terminal_tools,
            mutation_gate: self.mutation_gate.as_ref(),
            calls: Vec::new(),
            mutation_calls: 0,
            saw_terminal: false,
            completed_mutation_payload: None,
        };
        match parser.parse() {
            PrefixResult::Incomplete => DsmlProbe {
                valid: !at_eos,
                complete: false,
                complete_terminal_call: false,
                rejection: None,
            },
            PrefixResult::Complete => DsmlProbe {
                valid: true,
                complete: true,
                complete_terminal_call: parser.saw_terminal,
                rejection: None,
            },
            PrefixResult::Invalid => invalid_probe(None),
            PrefixResult::MutationRejected(code) => invalid_probe(Some(code)),
        }
    }

    /// Returns the first syntactically and mutation-valid payload whose JSON string has closed,
    /// even when the following DSML parameter/invoke delimiters are still only prefixes. This is
    /// the last repairable sampling boundary for a semantic provider: rejecting the closing quote
    /// lets the model continue the string, while waiting for the invoke to close does not.
    pub fn completed_mutation_payload(&self, transcript: &[u8]) -> Option<CanonicalToolCall> {
        let decoded = std::str::from_utf8(transcript).ok()?;
        let start = decoded.find(CALLS_OPEN)?;
        let mut parser = PrefixParser {
            input: decoded,
            position: start + CALLS_OPEN.len(),
            schemas: &self.schemas,
            terminal_tools: &self.terminal_tools,
            mutation_gate: self.mutation_gate.as_ref(),
            calls: Vec::new(),
            mutation_calls: 0,
            saw_terminal: false,
            completed_mutation_payload: None,
        };
        let result = parser.parse();
        if matches!(
            result,
            PrefixResult::Invalid | PrefixResult::MutationRejected(_)
        ) {
            None
        } else {
            parser.completed_mutation_payload
        }
    }

    pub fn terminal_state(&self, transcript: &[u8]) -> &'static str {
        let probe = self.probe(transcript, false);
        if probe.complete_terminal_call {
            "complete_terminal_tool_call"
        } else if probe.complete {
            "complete_tool_batch"
        } else if transcript
            .windows(CALLS_OPEN.len())
            .any(|window| window == CALLS_OPEN.as_bytes())
        {
            "in_tool_call"
        } else {
            "before_tool_call"
        }
    }
}

fn invalid_probe(rejection: Option<RejectionCode>) -> DsmlProbe {
    DsmlProbe {
        valid: false,
        complete: false,
        complete_terminal_call: false,
        rejection,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixResult {
    Incomplete,
    Complete,
    Invalid,
    MutationRejected(RejectionCode),
}

struct PrefixParser<'a> {
    input: &'a str,
    position: usize,
    schemas: &'a BTreeMap<String, Value>,
    terminal_tools: &'a BTreeSet<String>,
    mutation_gate: Option<&'a MutationCompletionGate>,
    calls: Vec<CanonicalToolCall>,
    mutation_calls: usize,
    saw_terminal: bool,
    completed_mutation_payload: Option<CanonicalToolCall>,
}

impl PrefixParser<'_> {
    fn parse(&mut self) -> PrefixResult {
        loop {
            if !self.skip_structural_whitespace() {
                return PrefixResult::Invalid;
            }
            match self.literal_choice(&[CALLS_CLOSE, INVOKE_OPEN]) {
                LiteralChoice::Incomplete => return PrefixResult::Incomplete,
                LiteralChoice::Invalid => return PrefixResult::Invalid,
                LiteralChoice::Match(CALLS_CLOSE) => {
                    if self.calls.is_empty() {
                        return PrefixResult::Invalid;
                    }
                    self.position += CALLS_CLOSE.len();
                    if !self.skip_structural_whitespace() {
                        return PrefixResult::Invalid;
                    }
                    return if self.position == self.input.len() {
                        PrefixResult::Complete
                    } else {
                        PrefixResult::Invalid
                    };
                }
                LiteralChoice::Match(INVOKE_OPEN) => match self.parse_invoke() {
                    PrefixResult::Complete => {}
                    result => return result,
                },
                LiteralChoice::Match(_) => unreachable!(),
            }
        }
    }

    fn parse_invoke(&mut self) -> PrefixResult {
        self.position += INVOKE_OPEN.len();
        let name = match self.fixed_attribute_value(self.schemas.keys().map(String::as_str)) {
            AttributeValue::Incomplete => return PrefixResult::Incomplete,
            AttributeValue::Invalid => return PrefixResult::Invalid,
            AttributeValue::Complete(value) => value,
        };
        match self.consume_literal("\">") {
            PrefixResult::Complete => {}
            result => return result,
        }
        let Some(schema) = self.schemas.get(&name) else {
            return PrefixResult::Invalid;
        };
        let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
            return PrefixResult::Invalid;
        };
        let required = required_fields(schema);
        let order = protocol_property_order(&name, properties);
        let mut seen = BTreeSet::new();
        let mut arguments = Map::new();
        let mut last_index = None;

        loop {
            if !self.skip_structural_whitespace() {
                return PrefixResult::Invalid;
            }
            let eligible = eligible_properties(&order, &required, &seen, last_index);
            match self.literal_choice(&[INVOKE_CLOSE, PARAMETER_OPEN]) {
                LiteralChoice::Incomplete => return PrefixResult::Incomplete,
                LiteralChoice::Invalid => return PrefixResult::Invalid,
                LiteralChoice::Match(INVOKE_CLOSE) => {
                    if !required.iter().all(|field| seen.contains(*field)) {
                        return PrefixResult::Invalid;
                    }
                    if validate_value(schema, &Value::Object(arguments.clone())).is_err() {
                        return PrefixResult::Invalid;
                    }
                    if let Some(gate) = self.mutation_gate {
                        match gate.evaluate(&name, &Value::Object(arguments.clone())) {
                            CompletionDecision::Reject(code) => {
                                return PrefixResult::MutationRejected(code);
                            }
                            CompletionDecision::Accept => {
                                self.mutation_calls = self.mutation_calls.saturating_add(1);
                                if self.mutation_calls
                                    > gate.manifest().mutation_policy.max_mutation_calls_per_batch
                                {
                                    return PrefixResult::Invalid;
                                }
                            }
                            CompletionDecision::NotApplicable => {}
                        }
                    }
                    self.position += INVOKE_CLOSE.len();
                    self.saw_terminal |= self.terminal_tools.contains(&name);
                    self.calls.push(CanonicalToolCall {
                        name,
                        arguments: Value::Object(arguments),
                    });
                    return PrefixResult::Complete;
                }
                LiteralChoice::Match(PARAMETER_OPEN) => {
                    self.position += PARAMETER_OPEN.len();
                    let parameter = match self.fixed_attribute_value(eligible.iter().copied()) {
                        AttributeValue::Incomplete => return PrefixResult::Incomplete,
                        AttributeValue::Invalid => return PrefixResult::Invalid,
                        AttributeValue::Complete(value) => value,
                    };
                    let Some(index) = order.iter().position(|field| *field == parameter) else {
                        return PrefixResult::Invalid;
                    };
                    last_index = Some(index);
                    let Some(parameter_schema) = properties.get(&parameter) else {
                        return PrefixResult::Invalid;
                    };
                    let mutation_payload =
                        mutation_payload_field(&name) == Some(parameter.as_str());
                    let string_mode = !mutation_payload
                        && parameter_schema.get("type").and_then(Value::as_str) == Some("string");
                    let header_end = if string_mode {
                        "\" string=\"true\">"
                    } else {
                        "\" string=\"false\">"
                    };
                    match self.consume_literal(header_end) {
                        PrefixResult::Complete => {}
                        result => return result,
                    }
                    let value = match self.parse_parameter_value(
                        &name,
                        &parameter,
                        parameter_schema,
                        string_mode,
                        mutation_payload,
                        &arguments,
                    ) {
                        ParameterValue::Incomplete => return PrefixResult::Incomplete,
                        ParameterValue::Invalid => return PrefixResult::Invalid,
                        ParameterValue::MutationRejected(code) => {
                            return PrefixResult::MutationRejected(code);
                        }
                        ParameterValue::Complete(value) => value,
                    };
                    if !seen.insert(parameter.clone())
                        || arguments.insert(parameter, value).is_some()
                    {
                        return PrefixResult::Invalid;
                    }
                }
                LiteralChoice::Match(_) => unreachable!(),
            }
        }
    }

    fn parse_parameter_value(
        &mut self,
        tool: &str,
        parameter: &str,
        schema: &Value,
        string_mode: bool,
        mutation_payload: bool,
        prior_arguments: &Map<String, Value>,
    ) -> ParameterValue {
        if string_mode {
            let Some(relative_end) = self.input[self.position..].find(PARAMETER_CLOSE) else {
                return ParameterValue::Incomplete;
            };
            let end = self.position + relative_end;
            let value =
                self.input[self.position..end].replace(ESCAPED_PARAMETER_CLOSE, PARAMETER_CLOSE);
            self.position = end + PARAMETER_CLOSE.len();
            let value = Value::String(value);
            return if validate_value(schema, &value).is_ok() {
                ParameterValue::Complete(value)
            } else {
                ParameterValue::Invalid
            };
        }

        if mutation_payload {
            let (value, consumed) = match parse_json_string_prefix(&self.input[self.position..]) {
                JsonStringPrefix::Incomplete(prefix) => {
                    if let (Some(prefix), Some(gate)) = (prefix, self.mutation_gate)
                        && let CompletionDecision::Reject(code) =
                            gate.evaluate_prefix(tool, prior_arguments, &prefix)
                    {
                        return ParameterValue::MutationRejected(code);
                    }
                    return ParameterValue::Incomplete;
                }
                JsonStringPrefix::Invalid => return ParameterValue::Invalid,
                JsonStringPrefix::Complete(value, consumed) => (value, consumed),
            };
            if validate_value(schema, &Value::String(value.clone())).is_err() {
                return ParameterValue::Invalid;
            }
            let mut arguments = prior_arguments.clone();
            arguments.insert(parameter.to_string(), Value::String(value.clone()));
            if let Some(gate) = self.mutation_gate
                && let CompletionDecision::Reject(code) =
                    gate.evaluate(tool, &Value::Object(arguments.clone()))
            {
                return ParameterValue::MutationRejected(code);
            }
            self.completed_mutation_payload = Some(CanonicalToolCall {
                name: tool.to_string(),
                arguments: Value::Object(arguments),
            });
            self.position += consumed;
            match self.consume_literal(PARAMETER_CLOSE) {
                PrefixResult::Complete => ParameterValue::Complete(Value::String(value)),
                PrefixResult::Incomplete => ParameterValue::Incomplete,
                PrefixResult::Invalid | PrefixResult::MutationRejected(_) => {
                    ParameterValue::Invalid
                }
            }
        } else {
            let Some((value, consumed)) =
                json_value_before_parameter_close(&self.input[self.position..])
            else {
                return ParameterValue::Incomplete;
            };
            if validate_value(schema, &value).is_err() {
                return ParameterValue::Invalid;
            }
            self.position += consumed;
            match self.consume_literal(PARAMETER_CLOSE) {
                PrefixResult::Complete => ParameterValue::Complete(value),
                PrefixResult::Incomplete => ParameterValue::Incomplete,
                PrefixResult::Invalid | PrefixResult::MutationRejected(_) => {
                    ParameterValue::Invalid
                }
            }
        }
    }

    fn skip_structural_whitespace(&mut self) -> bool {
        let start = self.position;
        while self
            .input
            .as_bytes()
            .get(self.position)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.position += 1;
        }
        self.position.saturating_sub(start) <= MAX_STRUCTURAL_WHITESPACE_BYTES
    }

    fn consume_literal(&mut self, literal: &str) -> PrefixResult {
        let remaining = &self.input[self.position..];
        if remaining.starts_with(literal) {
            self.position += literal.len();
            PrefixResult::Complete
        } else if literal.starts_with(remaining) {
            PrefixResult::Incomplete
        } else {
            PrefixResult::Invalid
        }
    }

    fn literal_choice<'a>(&self, literals: &'a [&'a str]) -> LiteralChoice<'a> {
        let remaining = &self.input[self.position..];
        if let Some(literal) = literals
            .iter()
            .copied()
            .find(|literal| remaining.starts_with(*literal))
        {
            LiteralChoice::Match(literal)
        } else if literals
            .iter()
            .any(|literal| literal.starts_with(remaining))
        {
            LiteralChoice::Incomplete
        } else {
            LiteralChoice::Invalid
        }
    }

    fn fixed_attribute_value<'a>(
        &mut self,
        values: impl Iterator<Item = &'a str>,
    ) -> AttributeValue {
        let values = values.collect::<Vec<_>>();
        let remaining = &self.input[self.position..];
        let Some(end) = remaining.find('"') else {
            return if values.iter().any(|value| value.starts_with(remaining)) {
                AttributeValue::Incomplete
            } else {
                AttributeValue::Invalid
            };
        };
        let value = &remaining[..end];
        if !values.contains(&value) {
            return AttributeValue::Invalid;
        }
        self.position += end;
        AttributeValue::Complete(value.to_string())
    }
}

enum LiteralChoice<'a> {
    Match(&'a str),
    Incomplete,
    Invalid,
}

enum AttributeValue {
    Complete(String),
    Incomplete,
    Invalid,
}

enum ParameterValue {
    Complete(Value),
    Incomplete,
    Invalid,
    MutationRejected(RejectionCode),
}

enum JsonStringPrefix {
    Complete(String, usize),
    Incomplete(Option<String>),
    Invalid,
}

fn json_value_before_parameter_close(input: &str) -> Option<(Value, usize)> {
    let mut search_start = 0usize;
    while let Some(relative_end) = input[search_start..].find(PARAMETER_CLOSE) {
        let end = search_start + relative_end;
        if let Ok(value) = serde_json::from_str::<Value>(&input[..end]) {
            return Some((value, end));
        }
        search_start = end.saturating_add(PARAMETER_CLOSE.len());
    }
    None
}

fn parse_json_string_prefix(input: &str) -> JsonStringPrefix {
    if !input.starts_with('"') {
        return if input.is_empty() {
            JsonStringPrefix::Incomplete(Some(String::new()))
        } else {
            JsonStringPrefix::Invalid
        };
    }
    let mut escaped = false;
    let mut unicode_digits = 0u8;
    for (offset, character) in input[1..].char_indices() {
        let absolute = 1 + offset;
        if unicode_digits > 0 {
            if !character.is_ascii_hexdigit() {
                return JsonStringPrefix::Invalid;
            }
            unicode_digits -= 1;
            continue;
        }
        if escaped {
            escaped = false;
            match character {
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {}
                'u' => unicode_digits = 4,
                _ => return JsonStringPrefix::Invalid,
            }
            continue;
        }
        match character {
            '\\' => escaped = true,
            '"' => {
                let end = absolute + 1;
                return serde_json::from_str::<String>(&input[..end])
                    .map(|value| JsonStringPrefix::Complete(value, end))
                    .unwrap_or(JsonStringPrefix::Invalid);
            }
            character if character.is_control() => return JsonStringPrefix::Invalid,
            _ => {}
        }
    }
    if escaped || unicode_digits > 0 {
        JsonStringPrefix::Incomplete(None)
    } else {
        serde_json::from_str::<String>(&format!("{input}\""))
            .map(|value| JsonStringPrefix::Incomplete(Some(value)))
            .unwrap_or(JsonStringPrefix::Incomplete(None))
    }
}

fn mutation_payload_field(tool: &str) -> Option<&'static str> {
    match tool {
        "write_file" | "replace_file" => Some("content"),
        "edit_file" => Some("new_text"),
        "apply_patch" => Some("patch"),
        _ => None,
    }
}

fn protocol_property_order<'a>(tool: &str, properties: &'a Map<String, Value>) -> Vec<&'a str> {
    let preferred: &[&str] = match tool {
        "write_file" | "replace_file" => &["path", "content", "completion"],
        "edit_file" => &["path", "old_text", "new_text", "completion"],
        "apply_patch" => &["patch"],
        _ => return properties.keys().map(String::as_str).collect(),
    };
    if properties
        .keys()
        .all(|property| preferred.contains(&property.as_str()))
    {
        preferred
            .iter()
            .copied()
            .filter(|field| properties.contains_key(*field))
            .collect()
    } else {
        properties.keys().map(String::as_str).collect()
    }
}

fn required_fields(schema: &Value) -> BTreeSet<&str> {
    schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect()
}

fn eligible_properties<'a>(
    order: &'a [&'a str],
    required: &BTreeSet<&str>,
    seen: &BTreeSet<String>,
    last_index: Option<usize>,
) -> Vec<&'a str> {
    order
        .iter()
        .enumerate()
        .filter(|(index, field)| {
            !seen.contains(**field)
                && last_index.is_none_or(|last| *index > last)
                && order[..*index]
                    .iter()
                    .filter(|earlier| required.contains(**earlier))
                    .all(|earlier| seen.contains(*earlier))
        })
        .map(|(_, field)| *field)
        .collect()
}

fn validate_schema(schema: &Value, location: &str) -> CollarResult<()> {
    let object = schema
        .as_object()
        .ok_or_else(|| CollarError::Protocol(format!("{location} schema must be an object")))?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| CollarError::Protocol(format!("{location} schema requires a type")))?;
    match kind {
        "object" => {
            let properties = object
                .get("properties")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    CollarError::Protocol(format!("{location} object requires properties"))
                })?;
            for (name, property) in properties {
                validate_schema(property, &format!("{location}.{name}"))?;
            }
            for required in required_fields(schema) {
                if !properties.contains_key(required) {
                    return Err(CollarError::Protocol(format!(
                        "{location} requires unknown property {required:?}"
                    )));
                }
            }
        }
        "array" => validate_schema(
            object
                .get("items")
                .ok_or_else(|| CollarError::Protocol(format!("{location} array requires items")))?,
            &format!("{location}[]"),
        )?,
        "string" | "integer" | "number" | "boolean" => {}
        _ => {
            return Err(CollarError::Protocol(format!(
                "{location} has unsupported type {kind:?}"
            )));
        }
    }
    Ok(())
}

fn validate_value(schema: &Value, value: &Value) -> CollarResult<()> {
    let kind = schema
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| CollarError::Protocol("schema value is missing type".to_string()))?;
    let valid_kind = match kind {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        _ => false,
    };
    if !valid_kind {
        return Err(CollarError::Protocol(format!(
            "value does not match schema type {kind}"
        )));
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array)
        && !values.contains(value)
    {
        return Err(CollarError::Protocol(
            "value is outside schema enum".to_string(),
        ));
    }
    match kind {
        "object" => {
            let object = value.as_object().expect("validated object");
            let properties = schema
                .get("properties")
                .and_then(Value::as_object)
                .expect("validated schema");
            if object.keys().any(|key| !properties.contains_key(key)) {
                return Err(CollarError::Protocol(
                    "object contains an unknown property".to_string(),
                ));
            }
            if !required_fields(schema)
                .iter()
                .all(|required| object.contains_key(*required))
            {
                return Err(CollarError::Protocol(
                    "object omits a required property".to_string(),
                ));
            }
            for (name, child) in object {
                validate_value(&properties[name], child)?;
            }
        }
        "array" => {
            let array = value.as_array().expect("validated array");
            let minimum = schema.get("minItems").and_then(Value::as_u64).unwrap_or(0);
            let maximum = schema.get("maxItems").and_then(Value::as_u64);
            if array.len() < minimum as usize
                || maximum.is_some_and(|maximum| array.len() > maximum as usize)
            {
                return Err(CollarError::Protocol(
                    "array length is outside schema bounds".to_string(),
                ));
            }
            let items = &schema["items"];
            for child in array {
                validate_value(items, child)?;
            }
        }
        "string" => {
            let length = value.as_str().expect("validated string").chars().count();
            if schema
                .get("minLength")
                .and_then(Value::as_u64)
                .is_some_and(|minimum| length < minimum as usize)
                || schema
                    .get("maxLength")
                    .and_then(Value::as_u64)
                    .is_some_and(|maximum| length > maximum as usize)
            {
                return Err(CollarError::Protocol(
                    "string length is outside schema bounds".to_string(),
                ));
            }
        }
        "integer" | "number" => {
            let number = value.as_f64().expect("validated number");
            if schema
                .get("minimum")
                .and_then(Value::as_f64)
                .is_some_and(|minimum| number < minimum)
                || schema
                    .get("maximum")
                    .and_then(Value::as_f64)
                    .is_some_and(|maximum| number > maximum)
            {
                return Err(CollarError::Protocol(
                    "number is outside schema bounds".to_string(),
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

pub fn parse_dsml_output(content: &str, allow_incomplete: bool) -> CollarResult<DsmlParseOutput> {
    let mut remaining = content;
    let mut text = String::new();
    let mut calls = Vec::new();
    while let Some(start) = remaining.find(CALLS_OPEN) {
        text.push_str(&remaining[..start]);
        let block_start = start + CALLS_OPEN.len();
        let block_tail = &remaining[block_start..];
        let mut search_start = 0usize;
        let mut parsed_block = None;
        let mut last_error = None;
        while let Some(relative_end) = block_tail[search_start..].find(CALLS_CLOSE) {
            let relative_end = search_start + relative_end;
            match parse_calls_block(&block_tail[..relative_end]) {
                Ok(block_calls) => {
                    parsed_block = Some((relative_end, block_calls));
                    break;
                }
                Err(error) => last_error = Some(error),
            }
            search_start = relative_end.saturating_add(CALLS_CLOSE.len());
        }
        let Some((relative_end, block_calls)) = parsed_block else {
            if allow_incomplete {
                text.push_str(&remaining[start..]);
                return Ok(DsmlParseOutput {
                    text: text.trim().to_string(),
                    calls,
                    incomplete: true,
                });
            }
            return Err(last_error.unwrap_or_else(|| {
                CollarError::Incomplete(format!("DeepSeek DSML tool call is missing {CALLS_CLOSE}"))
            }));
        };
        let block_end = block_start + relative_end;
        calls.extend(block_calls);
        remaining = &remaining[block_end + CALLS_CLOSE.len()..];
    }
    text.push_str(remaining);
    Ok(DsmlParseOutput {
        text: text.trim().to_string(),
        calls,
        incomplete: false,
    })
}

fn parse_calls_block(mut block: &str) -> CollarResult<Vec<CanonicalToolCall>> {
    let mut calls = Vec::new();
    loop {
        let Some(start) = block.find(INVOKE_OPEN) else {
            if !block.trim().is_empty() {
                return Err(CollarError::Protocol(
                    "DeepSeek DSML tool_calls contains text outside invoke".to_string(),
                ));
            }
            break;
        };
        if !block[..start].trim().is_empty() {
            return Err(CollarError::Protocol(
                "DeepSeek DSML tool_calls contains text before invoke".to_string(),
            ));
        }
        let name_start = start + INVOKE_OPEN.len();
        let name_end = block[name_start..]
            .find("\">")
            .map(|offset| name_start + offset)
            .ok_or_else(|| {
                CollarError::Incomplete("DSML invoke header is incomplete".to_string())
            })?;
        let name = block[name_start..name_end].to_string();
        let body_start = name_end + 2;
        let body_tail = &block[body_start..];
        let mut search_start = 0usize;
        let mut parsed_body = None;
        let mut last_error = None;
        while let Some(relative_end) = body_tail[search_start..].find(INVOKE_CLOSE) {
            let relative_end = search_start + relative_end;
            match parse_invoke_body(&body_tail[..relative_end]) {
                Ok(arguments) => {
                    parsed_body = Some((relative_end, arguments));
                    break;
                }
                Err(error) => last_error = Some(error),
            }
            search_start = relative_end.saturating_add(INVOKE_CLOSE.len());
        }
        let Some((relative_end, arguments)) = parsed_body else {
            return Err(last_error.unwrap_or_else(|| {
                CollarError::Incomplete("DSML invoke is not closed".to_string())
            }));
        };
        let body_end = body_start + relative_end;
        calls.push(CanonicalToolCall {
            name,
            arguments: Value::Object(arguments),
        });
        block = &block[body_end + INVOKE_CLOSE.len()..];
    }
    Ok(calls)
}

fn parse_invoke_body(mut body: &str) -> CollarResult<Map<String, Value>> {
    let mut arguments = Map::new();
    loop {
        let Some(parameter_start) = body.find(PARAMETER_OPEN) else {
            if !body.trim().is_empty() {
                return Err(CollarError::Protocol(
                    "DSML invoke contains text outside parameter".to_string(),
                ));
            }
            break;
        };
        if !body[..parameter_start].trim().is_empty() {
            return Err(CollarError::Protocol(
                "DSML invoke contains text before parameter".to_string(),
            ));
        }
        let name_start = parameter_start + PARAMETER_OPEN.len();
        let name_end = body[name_start..]
            .find('"')
            .map(|offset| name_start + offset)
            .ok_or_else(|| {
                CollarError::Incomplete("DSML parameter name is incomplete".to_string())
            })?;
        let parameter = &body[name_start..name_end];
        let header = &body[name_end..];
        let (string_mode, value_start) = if header.starts_with("\" string=\"true\">") {
            (true, name_end + "\" string=\"true\">".len())
        } else if header.starts_with("\" string=\"false\">") {
            (false, name_end + "\" string=\"false\">".len())
        } else {
            return Err(CollarError::Protocol(
                "DSML parameter has an invalid string attribute".to_string(),
            ));
        };
        let (value, value_end) = if string_mode {
            let value_end = body[value_start..]
                .find(PARAMETER_CLOSE)
                .map(|offset| value_start + offset)
                .ok_or_else(|| {
                    CollarError::Incomplete("DSML parameter is not closed".to_string())
                })?;
            (
                Value::String(
                    body[value_start..value_end].replace(ESCAPED_PARAMETER_CLOSE, PARAMETER_CLOSE),
                ),
                value_end,
            )
        } else {
            let (value, consumed) = json_value_before_parameter_close(&body[value_start..])
                .ok_or_else(|| {
                    CollarError::Protocol(format!(
                        "DSML non-string parameter {parameter:?} is not JSON"
                    ))
                })?;
            let value_end = value_start + consumed;
            if !body[value_end..].starts_with(PARAMETER_CLOSE) {
                return Err(CollarError::Protocol(format!(
                    "DSML non-string parameter {parameter:?} has trailing text"
                )));
            }
            (value, value_end)
        };
        if arguments.insert(parameter.to_string(), value).is_some() {
            return Err(CollarError::Protocol(format!(
                "DSML invoke repeats parameter {parameter:?}"
            )));
        }
        body = &body[value_end + PARAMETER_CLOSE.len()..];
    }
    Ok(arguments)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        mutation::{LogicalPath, SnapshotEntry, WorkspaceSnapshot},
        protocol::ToolDialect,
        tool::{CollarLimits, ExposedTool, MutationPolicy},
    };

    fn constraint() -> DsmlConstraint {
        DsmlConstraint::compile(CollarManifest {
            contract_version: 1,
            dialect: ToolDialect::DeepSeekDsml,
            mode: ToolConstraintMode::ToolRequired,
            tools: vec![ExposedTool {
                name: "write_file".to_string(),
                input_schema: json!({
                    "type":"object",
                    "properties":{
                        "path":{"type":"string"},
                        "content":{"type":"string","maxLength":256}
                    },
                    "required":["path","content"],
                    "additionalProperties":false
                }),
            }],
            terminal_tools: vec!["write_file".to_string()],
            mutation_policy: MutationPolicy {
                allow_write_file: true,
                allow_replace_file: false,
                allow_apply_patch: false,
                max_mutation_calls_per_batch: 1,
            },
            workspace: WorkspaceSnapshot::default(),
            limits: CollarLimits {
                max_argument_bytes: 4096,
                max_snapshot_bytes: 4096,
                max_files: 4,
                max_patch_hunks: 16,
            },
        })
        .unwrap()
    }

    fn write_prefix(content: &str) -> String {
        format!(
            "{CALLS_OPEN}\n{INVOKE_OPEN}write_file\">\n{PARAMETER_OPEN}path\" string=\"true\">lib.rs{PARAMETER_CLOSE}\n{PARAMETER_OPEN}content\" string=\"false\">{content}"
        )
    }

    #[test]
    fn mutation_json_quote_is_the_syntax_closure_boundary() {
        let constraint = constraint();
        let invalid = write_prefix("\"pub fn broken( {");
        assert!(constraint.probe(invalid.as_bytes(), false).valid);
        assert_eq!(
            constraint
                .probe(format!("{invalid}\"").as_bytes(), false)
                .rejection,
            Some(RejectionCode::InvalidSyntax)
        );
        let valid = write_prefix("\"pub fn ok() {}\\n\"");
        assert!(constraint.probe(valid.as_bytes(), false).valid);
        let complete = format!("{valid}{PARAMETER_CLOSE}\n{INVOKE_CLOSE}\n{CALLS_CLOSE}");
        let probe = constraint.probe(complete.as_bytes(), false);
        assert!(probe.valid && probe.complete && probe.complete_terminal_call);
    }

    #[test]
    fn controller_bound_path_allows_a_pathless_dsml_mutation() {
        let path = LogicalPath::parse("src/lib.rs").unwrap();
        let workspace = WorkspaceSnapshot::new(vec![SnapshotEntry::new(
            path.clone(),
            b"pub fn before() {}\n".to_vec(),
        )])
        .unwrap()
        .with_bound_mutation_path(path);
        let constraint = DsmlConstraint::compile(CollarManifest {
            contract_version: 1,
            dialect: ToolDialect::DeepSeekDsml,
            mode: ToolConstraintMode::ToolRequired,
            tools: vec![ExposedTool {
                name: "replace_file".to_string(),
                input_schema: json!({
                    "type":"object",
                    "properties":{
                        "path":{"type":"string"},
                        "content":{"type":"string","maxLength":256}
                    },
                    "required":["content"],
                    "additionalProperties":false
                }),
            }],
            terminal_tools: vec!["replace_file".to_string()],
            mutation_policy: MutationPolicy {
                allow_write_file: false,
                allow_replace_file: true,
                allow_apply_patch: false,
                max_mutation_calls_per_batch: 1,
            },
            workspace,
            limits: CollarLimits {
                max_argument_bytes: 4096,
                max_snapshot_bytes: 4096,
                max_files: 4,
                max_patch_hunks: 16,
            },
        })
        .unwrap();
        let open = format!(
            "{CALLS_OPEN}{INVOKE_OPEN}replace_file\">{PARAMETER_OPEN}content\" string=\"false\">\"pub fn after() {{}}\\n\""
        );
        assert!(constraint.probe(open.as_bytes(), false).valid);
        let complete = format!("{open}{PARAMETER_CLOSE}{INVOKE_CLOSE}{CALLS_CLOSE}");
        let probe = constraint.probe(complete.as_bytes(), false);
        assert!(probe.valid && probe.complete && probe.complete_terminal_call);
    }

    #[test]
    fn completed_mutation_payload_is_visible_before_dsml_delimiters_commit() {
        let constraint = constraint();
        let transcript = write_prefix("\"pub fn ok() {}\\n\"");
        let call = constraint
            .completed_mutation_payload(transcript.as_bytes())
            .expect("closed JSON string should expose a semantic boundary");
        assert_eq!(call.name, "write_file");
        assert_eq!(call.arguments["path"], "lib.rs");
        assert_eq!(call.arguments["content"], "pub fn ok() {}\n");
        assert!(
            constraint
                .completed_mutation_payload(write_prefix("\"pub fn pending() {").as_bytes())
                .is_none()
        );
    }

    #[test]
    fn mutation_prefix_rejects_a_definite_impossible_transition_before_quote_close() {
        let constraint = constraint();
        let impossible = write_prefix("\"pub fn broken() { ]");
        assert_eq!(
            constraint.probe(impossible.as_bytes(), false).rejection,
            Some(RejectionCode::InvalidPrefix)
        );
        let repairable = write_prefix("\"pub fn pending() { let value = (");
        assert!(constraint.probe(repairable.as_bytes(), false).valid);
    }

    #[test]
    fn final_parser_accepts_json_encoded_string_parameters() {
        let transcript = format!(
            "before{CALLS_OPEN}{INVOKE_OPEN}write_file\">{PARAMETER_OPEN}content\" string=\"false\">\"a\\nb\"{PARAMETER_CLOSE}{INVOKE_CLOSE}{CALLS_CLOSE}"
        );
        let parsed = parse_dsml_output(&transcript, false).unwrap();
        assert_eq!(parsed.text, "before");
        assert_eq!(parsed.calls[0].arguments["content"], json!("a\nb"));

        let nested = format!(
            "{CALLS_OPEN}{INVOKE_OPEN}tool\">{PARAMETER_OPEN}value\" string=\"false\">{{\"literal\":\"{PARAMETER_CLOSE}\"}}{PARAMETER_CLOSE}{INVOKE_CLOSE}{CALLS_CLOSE}"
        );
        let parsed = parse_dsml_output(&nested, false).unwrap();
        assert_eq!(
            parsed.calls[0].arguments["value"]["literal"],
            PARAMETER_CLOSE
        );
    }

    #[test]
    fn final_parser_ignores_dsml_closers_inside_json_parameters() {
        let payload = format!("before {INVOKE_CLOSE} {CALLS_CLOSE} {PARAMETER_CLOSE} after");
        let encoded = serde_json::to_string(&payload).unwrap();
        let transcript = format!(
            "{CALLS_OPEN}{INVOKE_OPEN}tool\">{PARAMETER_OPEN}value\" string=\"false\">{encoded}{PARAMETER_CLOSE}{INVOKE_CLOSE}{CALLS_CLOSE}"
        );
        let parsed = parse_dsml_output(&transcript, false).unwrap();
        assert_eq!(parsed.calls[0].arguments["value"], payload);

        let incomplete = format!(
            "{CALLS_OPEN}{INVOKE_OPEN}tool\">{PARAMETER_OPEN}value\" string=\"false\">{encoded}"
        );
        let parsed = parse_dsml_output(&incomplete, true).unwrap();
        assert!(parsed.incomplete);
        assert!(parsed.calls.is_empty());
    }
}
