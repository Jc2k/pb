use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

use crate::agent_core::AgentProfile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyOutcome {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyConfig {
    #[serde(default)]
    pub rules: Vec<PolicyRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamMatcher {
    Pattern {
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
    },
    Literal(Value),
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
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read policy config at {}", path.display()))?;
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
    fn matches(&self, value: Option<&Value>) -> bool {
        match self {
            Self::Literal(expected) => value.is_some_and(|actual| actual == expected),
            Self::Pattern {
                equals,
                contains,
                glob,
                regex,
                exists,
            } => {
                if let Some(expected_exists) = exists {
                    if value.is_some() != *expected_exists {
                        return false;
                    }
                }
                if let Some(expected) = equals {
                    if !value.is_some_and(|actual| actual == expected) {
                        return false;
                    }
                }
                if let Some(needle) = contains {
                    if !value_to_match_text(value).is_some_and(|actual| actual.contains(needle)) {
                        return false;
                    }
                }
                if let Some(pattern) = glob {
                    if !value_to_match_text(value)
                        .is_some_and(|actual| wildcard_match(pattern, &actual))
                    {
                        return false;
                    }
                }
                if let Some(pattern) = regex {
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
            command = { regex = "rm\\s+-rf" }
        "#,
        )
        .unwrap();
        let decision = config.decide(
            AgentProfile::Build,
            "run_command",
            &json!({"command":"rm -rf target"}),
        );
        assert_eq!(decision.outcome, PolicyOutcome::Deny);
        assert_eq!(
            config
                .decide(
                    AgentProfile::Plan,
                    "run_command",
                    &json!({"command":"rm -rf target"})
                )
                .outcome,
            PolicyOutcome::Allow
        );
    }
}
