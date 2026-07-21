use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::agent_core::AgentProfile;

const MAX_POLICY_CONFIG_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyOutcome {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
    #[serde(default)]
    pub rules: Vec<PolicyRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyRule {
    pub outcome: PolicyOutcome,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub profiles: Vec<AgentProfile>,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub params: BTreeMap<String, ParamMatcher>,
    #[serde(default)]
    pub question: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ParamMatcher {
    Pattern(ParamPattern),
    Literal(Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamPattern {
    #[serde(default)]
    equals: Option<Value>,
    #[serde(default)]
    contains: Option<String>,
    #[serde(default)]
    glob: Option<String>,
    #[serde(default)]
    regex: Option<String>,
    #[serde(default)]
    exists: Option<bool>,
}

impl<'de> Deserialize<'de> for ParamMatcher {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        if value.is_object() {
            serde_json::from_value(value)
                .map(Self::Pattern)
                .map_err(D::Error::custom)
        } else {
            Ok(Self::Literal(value))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDecision {
    pub outcome: PolicyOutcome,
    pub rule_name: Option<String>,
    pub question: Option<String>,
}

impl PolicyConfig {
    pub fn load(workspace_root: &Path) -> Result<Option<Self>> {
        let path = workspace_root.join(".pb/policy.toml");
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(&path)
            .with_context(|| format!("failed to open policy config at {}", path.display()))?;
        let metadata = file
            .metadata()
            .with_context(|| format!("failed to stat policy config at {}", path.display()))?;
        if !metadata.is_file() || metadata.len() > MAX_POLICY_CONFIG_BYTES {
            bail!(
                "policy config at {} exceeds the {}-byte input bound",
                path.display(),
                MAX_POLICY_CONFIG_BYTES
            );
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_POLICY_CONFIG_BYTES.saturating_add(1))
            .read_to_end(&mut bytes)
            .with_context(|| format!("failed to read policy config at {}", path.display()))?;
        if bytes.len() as u64 > MAX_POLICY_CONFIG_BYTES {
            bail!(
                "policy config at {} grew beyond the {}-byte input bound",
                path.display(),
                MAX_POLICY_CONFIG_BYTES
            );
        }
        let raw = String::from_utf8(bytes).context("policy config is not valid UTF-8")?;
        let config = toml::from_str(&raw)
            .with_context(|| format!("failed to parse policy config at {}", path.display()))?;
        Ok(Some(config))
    }

    pub fn decide(&self, profile: AgentProfile, tool: &str, arguments: &Value) -> PolicyDecision {
        for rule in &self.rules {
            if !rule.matches(profile, tool, arguments) {
                continue;
            }
            return PolicyDecision {
                outcome: rule.outcome,
                rule_name: rule.name.clone(),
                question: rule.question.clone(),
            };
        }
        PolicyDecision {
            outcome: PolicyOutcome::Allow,
            rule_name: None,
            question: None,
        }
    }

    /// Validate policy rules against the exact tool schemas exposed for this session.
    ///
    /// Policy is allow-by-default, so a misspelled tool argument would otherwise turn a deny/ask
    /// rule into a silent allow. Exact tool names therefore fail closed when either the tool or a
    /// referenced argument path is unknown. Wildcard rules retain their intentional cross-tool
    /// matching semantics, but their regular expressions are still validated eagerly.
    pub fn validate_tool_schemas(&self, schemas: &BTreeMap<String, Value>) -> Result<()> {
        for (index, rule) in self.rules.iter().enumerate() {
            let label = rule
                .name
                .as_deref()
                .map(str::to_string)
                .unwrap_or_else(|| format!("rule {}", index + 1));
            for matcher in rule.params.values() {
                matcher
                    .validate()
                    .with_context(|| format!("invalid matcher in policy {label}"))?;
            }
            for tool_pattern in &rule.tools {
                let matching = schemas
                    .iter()
                    .filter(|(tool, _)| wildcard_match(tool_pattern, tool))
                    .collect::<Vec<_>>();
                if matching.is_empty() {
                    bail!(
                        "policy {label} references unknown tool or unmatched pattern '{tool_pattern}'"
                    );
                }
                for path in rule.params.keys() {
                    let valid = if tool_pattern.contains('*') {
                        matching
                            .iter()
                            .any(|(_, schema)| schema_has_argument_path(schema, path))
                    } else {
                        matching
                            .iter()
                            .all(|(_, schema)| schema_has_argument_path(schema, path))
                    };
                    if !valid {
                        bail!(
                            "policy {label} references unknown argument '{path}' for tool selector '{tool_pattern}'"
                        );
                    }
                }
            }
            if rule.tools.is_empty() && !rule.params.is_empty() {
                for path in rule.params.keys() {
                    if !schemas
                        .values()
                        .any(|schema| schema_has_argument_path(schema, path))
                    {
                        bail!(
                            "policy {label} references argument '{path}' that no exposed tool accepts"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

impl PolicyRule {
    fn matches(&self, profile: AgentProfile, tool: &str, arguments: &Value) -> bool {
        (self.profiles.is_empty() || self.profiles.contains(&profile))
            && (self.tools.is_empty()
                || self
                    .tools
                    .iter()
                    .any(|candidate| wildcard_match(candidate, tool)))
            && self
                .params
                .iter()
                .all(|(path, matcher)| matcher.matches(argument_path(arguments, path)))
    }
}

impl ParamMatcher {
    fn validate(&self) -> Result<()> {
        if let Self::Pattern(pattern) = self {
            if pattern.equals.is_none()
                && pattern.contains.is_none()
                && pattern.glob.is_none()
                && pattern.regex.is_none()
                && pattern.exists.is_none()
            {
                bail!("policy parameter matcher must declare at least one predicate");
            }
            if let Some(regex) = &pattern.regex {
                Regex::new(regex)
                    .with_context(|| format!("invalid policy regular expression '{regex}'"))?;
            }
        }
        Ok(())
    }

    fn matches(&self, value: Option<&Value>) -> bool {
        match self {
            Self::Literal(expected) => value.is_some_and(|actual| actual == expected),
            Self::Pattern(pattern) => {
                if let Some(expected_exists) = &pattern.exists
                    && value.is_some() != *expected_exists
                {
                    return false;
                }
                if let Some(expected) = &pattern.equals
                    && !value.is_some_and(|actual| actual == expected)
                {
                    return false;
                }
                if let Some(needle) = &pattern.contains
                    && !value_to_match_text(value).is_some_and(|actual| actual.contains(needle))
                {
                    return false;
                }
                if let Some(pattern) = &pattern.glob
                    && !value_to_match_text(value)
                        .is_some_and(|actual| wildcard_match(pattern, &actual))
                {
                    return false;
                }
                if let Some(pattern) = &pattern.regex {
                    let Ok(re) = Regex::new(pattern) else {
                        return false;
                    };
                    if !value_to_match_text(value).is_some_and(|actual| re.is_match(&actual)) {
                        return false;
                    }
                }
                true
            }
        }
    }
}

fn schema_has_argument_path(schema: &Value, path: &str) -> bool {
    let mut current = schema;
    for segment in path.split('.') {
        if segment.is_empty() {
            return false;
        }
        if segment.parse::<usize>().is_ok() {
            let Some(items) = current.get("items") else {
                return false;
            };
            current = items;
        } else {
            let Some(property) = current
                .get("properties")
                .and_then(Value::as_object)
                .and_then(|properties| properties.get(segment))
            else {
                return false;
            };
            current = property;
        }
    }
    true
}

fn argument_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for segment in path.split('.') {
        if segment.is_empty() {
            return None;
        }
        current = match current {
            Value::Object(map) => map.get(segment)?,
            Value::Array(items) => items.get(segment.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(current)
}

fn value_to_match_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

fn wildcard_match(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let regex = format!("^{}$", regex::escape(pattern).replace("\\*", ".*"));
    Regex::new(&regex).is_ok_and(|re| re.is_match(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn policy_matches_profile_tool_and_nested_parameters() {
        let config = toml::from_str::<PolicyConfig>(
            r#"
            [[rules]]
            name = "block dangerous build command"
            outcome = "deny"
            profiles = ["build"]
            tools = ["run_command"]
            [rules.params]
            cmd = { regex = "rm\\s+-rf" }
        "#,
        )
        .unwrap();
        let decision = config.decide(
            AgentProfile::Build,
            "run_command",
            &json!({"cmd":"rm -rf target"}),
        );
        assert_eq!(decision.outcome, PolicyOutcome::Deny);
        assert_eq!(
            config
                .decide(
                    AgentProfile::Plan,
                    "run_command",
                    &json!({"cmd":"rm -rf target"})
                )
                .outcome,
            PolicyOutcome::Allow
        );
    }

    #[test]
    fn policy_schema_validation_rejects_unknown_exact_argument() {
        let config = toml::from_str::<PolicyConfig>(
            r#"
            [[rules]]
            outcome = "deny"
            tools = ["run_command"]
            [rules.params]
            command = { regex = "rm" }
        "#,
        )
        .unwrap();
        let schemas = BTreeMap::from([(
            "run_command".to_string(),
            json!({
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
                "additionalProperties": false
            }),
        )]);

        let error = config
            .validate_tool_schemas(&schemas)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown argument 'command'"), "{error}");
    }

    #[test]
    fn policy_schema_validation_accepts_nested_array_argument() {
        let config = toml::from_str::<PolicyConfig>(
            r#"
            [[rules]]
            outcome = "ask"
            tools = ["example"]
            [rules.params]
            "items.0.name" = { equals = "publish" }
        "#,
        )
        .unwrap();
        let schemas = BTreeMap::from([(
            "example".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {
                        "type": "object", "properties": {"name": {"type": "string"}}
                    }}
                }
            }),
        )]);

        config.validate_tool_schemas(&schemas).unwrap();
    }

    #[test]
    fn policy_schema_validation_rejects_unmatched_tool_pattern() {
        let config = toml::from_str::<PolicyConfig>(
            r#"
            [[rules]]
            outcome = "deny"
            tools = ["missing_*"]
        "#,
        )
        .unwrap();
        let schemas = BTreeMap::from([("run_command".to_string(), json!({"type":"object"}))]);

        let error = config
            .validate_tool_schemas(&schemas)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unmatched pattern 'missing_*'"), "{error}");
    }

    #[test]
    fn policy_rejects_unknown_rule_and_matcher_fields() {
        let rule_typo = toml::from_str::<PolicyConfig>(
            r#"
            [[rules]]
            outcome = "deny"
            tool = ["run_command"]
        "#,
        )
        .unwrap_err()
        .to_string();
        assert!(rule_typo.contains("unknown field `tool`"), "{rule_typo}");

        let matcher_typo = toml::from_str::<PolicyConfig>(
            r#"
            [[rules]]
            outcome = "deny"
            tools = ["run_command"]
            [rules.params]
            cmd = { regexp = "rm" }
        "#,
        )
        .unwrap_err()
        .to_string();
        assert!(
            matcher_typo.contains("unknown field `regexp`"),
            "{matcher_typo}"
        );
    }
}
