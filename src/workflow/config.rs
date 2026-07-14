use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::TurnIntent;

pub const WORKFLOW_CONFIG_VERSION: u32 = 1;

const MAX_STAGE_STEPS: usize = 64;
const MAX_MODEL_INVOCATIONS: usize = 256;
const MAX_GENERATED_TOKENS: usize = 1_000_000;
const MAX_ADVISORY_CALLS: usize = 32;
const MAX_CYCLES: usize = 16;
const MAX_REVIEW_PATHS: usize = 2_000;
const MAX_REVIEW_DIFF_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryPolicy {
    #[default]
    Strict,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkflowConfigDocument {
    pub version: u32,
    #[serde(default)]
    pub delivery: DeliveryPolicy,
    #[serde(default)]
    pub default_intent: TurnIntent,
    #[serde(default)]
    pub limits: WorkflowLimits,
}

impl Default for WorkflowConfigDocument {
    fn default() -> Self {
        Self {
            version: WORKFLOW_CONFIG_VERSION,
            delivery: DeliveryPolicy::Strict,
            default_intent: TurnIntent::Discuss,
            limits: WorkflowLimits::default(),
        }
    }
}

impl WorkflowConfigDocument {
    pub fn from_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn load(repo_root: &Path) -> Result<Option<Self>> {
        let path = repo_root.join(".pb").join("workflow.toml");
        if !path.exists() {
            return Ok(None);
        }
        Self::from_path(&path).map(Some)
    }

    pub fn load_or_default(repo_root: &Path) -> Result<CompiledWorkflowPolicy> {
        Self::load(repo_root)?.unwrap_or_default().compile()
    }

    pub fn save(&self, repo_root: &Path) -> Result<()> {
        let dir = repo_root.join(".pb");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        let path = dir.join("workflow.toml");
        let text = toml::to_string_pretty(self).context("failed to serialize workflow config")?;
        std::fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn compile(self) -> Result<CompiledWorkflowPolicy> {
        if self.version != WORKFLOW_CONFIG_VERSION {
            bail!(
                "unsupported workflow config version {}; expected {}",
                self.version,
                WORKFLOW_CONFIG_VERSION
            );
        }
        self.limits.validate()?;
        let normalized = NormalizedWorkflowPolicy {
            version: self.version,
            delivery: self.delivery,
            default_intent: self.default_intent,
            limits: self.limits,
        };
        let bytes = serde_json::to_vec(&normalized)
            .context("failed to serialize normalized workflow policy")?;
        let sha256 = format!("{:x}", Sha256::digest(bytes));
        Ok(CompiledWorkflowPolicy {
            version: normalized.version,
            delivery: normalized.delivery,
            default_intent: normalized.default_intent,
            limits: normalized.limits,
            sha256,
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkflowLimits {
    #[serde(default = "default_stage_steps")]
    pub stage_steps: usize,
    #[serde(default = "default_model_invocations")]
    pub total_model_invocations: usize,
    #[serde(default = "default_generated_tokens")]
    pub total_generated_tokens: usize,
    #[serde(default = "default_advisory_calls")]
    pub advisory_calls: usize,
    #[serde(default = "default_plan_cycles")]
    pub plan_cycles: usize,
    #[serde(default = "default_repair_cycles")]
    pub repair_cycles: usize,
    #[serde(default = "default_review_paths")]
    pub review_paths: usize,
    #[serde(default = "default_review_diff_bytes")]
    pub review_diff_bytes: usize,
}

impl Default for WorkflowLimits {
    fn default() -> Self {
        Self {
            stage_steps: default_stage_steps(),
            total_model_invocations: default_model_invocations(),
            total_generated_tokens: default_generated_tokens(),
            advisory_calls: default_advisory_calls(),
            plan_cycles: default_plan_cycles(),
            repair_cycles: default_repair_cycles(),
            review_paths: default_review_paths(),
            review_diff_bytes: default_review_diff_bytes(),
        }
    }
}

impl WorkflowLimits {
    fn validate(self) -> Result<()> {
        validate_limit("limits.stage_steps", self.stage_steps, MAX_STAGE_STEPS)?;
        validate_limit(
            "limits.total_model_invocations",
            self.total_model_invocations,
            MAX_MODEL_INVOCATIONS,
        )?;
        validate_limit(
            "limits.total_generated_tokens",
            self.total_generated_tokens,
            MAX_GENERATED_TOKENS,
        )?;
        validate_limit(
            "limits.advisory_calls",
            self.advisory_calls,
            MAX_ADVISORY_CALLS,
        )?;
        validate_limit("limits.plan_cycles", self.plan_cycles, MAX_CYCLES)?;
        validate_limit("limits.repair_cycles", self.repair_cycles, MAX_CYCLES)?;
        validate_limit("limits.review_paths", self.review_paths, MAX_REVIEW_PATHS)?;
        validate_limit(
            "limits.review_diff_bytes",
            self.review_diff_bytes,
            MAX_REVIEW_DIFF_BYTES,
        )?;
        Ok(())
    }
}

fn validate_limit(path: &str, value: usize, maximum: usize) -> Result<()> {
    if value == 0 {
        bail!("{path} must be positive");
    }
    if value > maximum {
        bail!("{path} exceeds the hard runtime ceiling of {maximum}");
    }
    Ok(())
}

const fn default_stage_steps() -> usize {
    8
}
const fn default_model_invocations() -> usize {
    40
}
const fn default_generated_tokens() -> usize {
    24_000
}
const fn default_advisory_calls() -> usize {
    4
}
const fn default_plan_cycles() -> usize {
    2
}
const fn default_repair_cycles() -> usize {
    2
}
const fn default_review_paths() -> usize {
    40
}
const fn default_review_diff_bytes() -> usize {
    200_000
}

#[derive(Debug, Clone, Copy, Serialize)]
struct NormalizedWorkflowPolicy {
    version: u32,
    delivery: DeliveryPolicy,
    default_intent: TurnIntent,
    limits: WorkflowLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompiledWorkflowPolicy {
    pub version: u32,
    pub delivery: DeliveryPolicy,
    pub default_intent: TurnIntent,
    pub limits: WorkflowLimits,
    pub sha256: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_compile_to_strict_discussion_first_policy() {
        let policy = WorkflowConfigDocument::default().compile().unwrap();
        assert_eq!(policy.delivery, DeliveryPolicy::Strict);
        assert_eq!(policy.default_intent, TurnIntent::Discuss);
        assert_eq!(policy.limits.stage_steps, 8);
        assert_eq!(policy.sha256.len(), 64);
    }

    #[test]
    fn equivalent_documents_have_the_same_policy_hash() {
        let first = WorkflowConfigDocument::default().compile().unwrap();
        let text = r#"
            version = 1
            delivery = "strict"
            default_intent = "discuss"

            [limits]
            stage_steps = 8
            total_model_invocations = 40
            total_generated_tokens = 24000
            advisory_calls = 4
            plan_cycles = 2
            repair_cycles = 2
            review_paths = 40
            review_diff_bytes = 200000
        "#;
        let second: WorkflowConfigDocument = toml::from_str(text).unwrap();
        assert_eq!(first.sha256, second.compile().unwrap().sha256);
    }

    #[test]
    fn unknown_fields_versions_and_unsafe_limits_fail_closed() {
        assert!(toml::from_str::<WorkflowConfigDocument>("version=1\nrelaxed=true").is_err());
        let mut wrong_version = WorkflowConfigDocument::default();
        wrong_version.version = 2;
        assert!(
            wrong_version
                .compile()
                .unwrap_err()
                .to_string()
                .contains("version")
        );
        let mut zero = WorkflowConfigDocument::default();
        zero.limits.repair_cycles = 0;
        assert!(
            zero.compile()
                .unwrap_err()
                .to_string()
                .contains("repair_cycles")
        );
        let mut excessive = WorkflowConfigDocument::default();
        excessive.limits.total_model_invocations = MAX_MODEL_INVOCATIONS + 1;
        assert!(
            excessive
                .compile()
                .unwrap_err()
                .to_string()
                .contains("ceiling")
        );
    }

    #[test]
    fn save_load_and_compile_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        WorkflowConfigDocument::default().save(dir.path()).unwrap();
        let loaded = WorkflowConfigDocument::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded, WorkflowConfigDocument::default());
        assert_eq!(loaded.clone().compile().unwrap(), loaded.compile().unwrap());
    }
}
