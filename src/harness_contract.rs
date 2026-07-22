use std::collections::HashSet;
use std::path::{Component, Path};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

const HARNESS_CONTRACT_VERSION: u32 = 1;
const DEFAULT_CHECK_TIMEOUT_SECONDS: u64 = 60;
const MAX_CHECK_TIMEOUT_SECONDS: u64 = 3600;
const MAX_DIAGNOSTIC_TIMEOUT_SECONDS: u64 = 60;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutationRequirement {
    Required,
    Forbidden,
    #[default]
    Optional,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HarnessCommitContract {
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub semantic: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HarnessReviewContract {
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub read_paths: Vec<String>,
    #[serde(default)]
    pub check_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HarnessCheckDocument {
    pub id: String,
    pub command: String,
    #[serde(default = "default_check_cwd")]
    pub cwd: String,
    #[serde(default = "default_true")]
    pub required: bool,
    #[serde(default)]
    pub diagnostic_eligible: bool,
    #[serde(default = "default_check_timeout_seconds")]
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HarnessContractDocument {
    pub version: u32,
    #[serde(default)]
    pub mutation: MutationRequirement,
    #[serde(default)]
    pub allowed_paths: Vec<String>,
    #[serde(default)]
    pub checks: Vec<HarnessCheckDocument>,
    #[serde(default)]
    pub commit: HarnessCommitContract,
    #[serde(default)]
    pub review: HarnessReviewContract,
    #[serde(default)]
    pub workspace_clean: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentCheckContract {
    pub id: String,
    pub command: String,
    pub cwd: String,
    pub required: bool,
    pub diagnostic_eligible: bool,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentContract {
    pub version: u32,
    pub mutation: MutationRequirement,
    pub allowed_paths: Vec<String>,
    pub checks: Vec<AgentCheckContract>,
    pub commit: HarnessCommitContract,
    pub review: HarnessReviewContract,
    pub workspace_clean: bool,
}

impl HarnessContractDocument {
    pub fn from_path(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read harness contract {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse harness contract {}", path.display()))
    }

    pub fn normalize(self) -> Result<AgentContract> {
        if self.version != HARNESS_CONTRACT_VERSION {
            bail!(
                "unsupported harness contract version {}; expected {}",
                self.version,
                HARNESS_CONTRACT_VERSION
            );
        }

        let allowed_paths = normalize_unique_paths("allowed_paths", self.allowed_paths, false)?;
        let review_read_paths =
            normalize_unique_paths("review.read_paths", self.review.read_paths, false)?;
        let mut check_ids = HashSet::new();
        let mut checks = Vec::with_capacity(self.checks.len());
        for (index, check) in self.checks.into_iter().enumerate() {
            let id = check.id.trim().to_string();
            if id.is_empty()
                || !id
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
            {
                bail!("checks[{index}].id must contain only ASCII letters, numbers, '-' or '_'");
            }
            if !check_ids.insert(id.clone()) {
                bail!("duplicate harness check id '{id}'");
            }
            let command = check.command.trim().to_string();
            if command.is_empty() {
                bail!("checks[{index}].command must not be empty");
            }
            let cwd = normalize_relative_path(&check.cwd, true)
                .with_context(|| format!("invalid checks[{index}].cwd"))?;
            if check.timeout_seconds == 0 || check.timeout_seconds > MAX_CHECK_TIMEOUT_SECONDS {
                bail!(
                    "checks[{index}].timeout_seconds must be between 1 and {MAX_CHECK_TIMEOUT_SECONDS}"
                );
            }
            if check.diagnostic_eligible && check.timeout_seconds > MAX_DIAGNOSTIC_TIMEOUT_SECONDS {
                bail!(
                    "checks[{index}] diagnostic_eligible timeout must not exceed {MAX_DIAGNOSTIC_TIMEOUT_SECONDS} seconds"
                );
            }
            checks.push(AgentCheckContract {
                id,
                command,
                cwd,
                required: check.required,
                diagnostic_eligible: check.diagnostic_eligible,
                timeout_seconds: check.timeout_seconds,
            });
        }

        let review_check_ids = normalize_unique_ids("review.check_ids", self.review.check_ids)?;
        for id in &review_check_ids {
            if !check_ids.contains(id) {
                bail!("review.check_ids references unknown check '{id}'");
            }
        }

        Ok(AgentContract {
            version: self.version,
            mutation: self.mutation,
            allowed_paths,
            checks,
            commit: self.commit,
            review: HarnessReviewContract {
                required: self.review.required,
                read_paths: review_read_paths,
                check_ids: review_check_ids,
            },
            workspace_clean: self.workspace_clean,
        })
    }
}

impl AgentContract {
    pub fn check(&self, id: &str) -> Option<&AgentCheckContract> {
        self.checks.iter().find(|check| check.id == id)
    }

    pub fn named_check_ids(&self) -> Vec<String> {
        self.checks.iter().map(|check| check.id.clone()).collect()
    }

    pub fn prompt_summary(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|_| "(contract could not be rendered)".to_string())
    }

    pub fn compile_workspace_graph(
        &self,
        mut graph: crate::workspace::WorkspaceGraph,
    ) -> Result<crate::workspace::WorkspaceGraph> {
        use crate::workspace::{
            CheckTrigger, Executor, ExecutorKind, WorkspaceCheck, WorkspaceComponent,
        };

        if self.checks.is_empty() {
            return Ok(graph);
        }
        const EXECUTOR_ID: &str = "harness-contract";
        const COMPONENT_ID: &str = "harness-contract";
        match graph.executors.get(EXECUTOR_ID) {
            Some(executor) if executor.kind != ExecutorKind::Project => {
                bail!("workspace executor id '{EXECUTOR_ID}' is reserved by the harness contract")
            }
            Some(_) => {}
            None => {
                graph.executors.insert(
                    EXECUTOR_ID.to_string(),
                    Executor {
                        id: EXECUTOR_ID.to_string(),
                        kind: ExecutorKind::Project,
                        environment: None,
                    },
                );
            }
        }
        if let Some(component) = graph.components.get(COMPONENT_ID) {
            if component.root != "." || component.executor != EXECUTOR_ID {
                bail!(
                    "workspace component id '{COMPONENT_ID}' is reserved by the harness contract"
                );
            }
        } else {
            graph.components.insert(
                COMPONENT_ID.to_string(),
                WorkspaceComponent {
                    id: COMPONENT_ID.to_string(),
                    root: ".".to_string(),
                    include: vec!["**".to_string()],
                    exclude: Vec::new(),
                    executor: EXECUTOR_ID.to_string(),
                    depends_on: Vec::new(),
                },
            );
        }
        let mut previous_required = None;
        for check in &self.checks {
            if let Some(existing) = graph.checks.get_mut(&check.id) {
                if existing.command != check.command || existing.cwd != check.cwd {
                    bail!(
                        "harness contract check '{}' conflicts with the workspace check of the same id",
                        check.id
                    );
                }
                if check.required {
                    existing.trigger = CheckTrigger::Always;
                    if let Some(previous) = &previous_required
                        && !existing.depends_on.contains(previous)
                    {
                        existing.depends_on.push(previous.clone());
                    }
                    previous_required = Some(check.id.clone());
                }
                continue;
            }
            let depends_on = if check.required {
                previous_required.iter().cloned().collect()
            } else {
                Vec::new()
            };
            graph.checks.insert(
                check.id.clone(),
                WorkspaceCheck {
                    id: check.id.clone(),
                    label: check.id.clone(),
                    command: check.command.clone(),
                    cwd: check.cwd.clone(),
                    executor: EXECUTOR_ID.to_string(),
                    components: vec![COMPONENT_ID.to_string()],
                    trigger: if check.required {
                        CheckTrigger::Always
                    } else {
                        CheckTrigger::Changed
                    },
                    inputs: vec!["**".to_string()],
                    outputs: Vec::new(),
                    depends_on,
                    timeout_seconds: check.timeout_seconds,
                },
            );
            if check.required {
                previous_required = Some(check.id.clone());
            }
        }
        graph.to_document().normalize()
    }
}

fn normalize_unique_paths(field: &str, paths: Vec<String>, allow_dot: bool) -> Result<Vec<String>> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::with_capacity(paths.len());
    for (index, path) in paths.into_iter().enumerate() {
        let path = normalize_relative_path(&path, allow_dot)
            .with_context(|| format!("invalid {field}[{index}]"))?;
        if seen.insert(path.clone()) {
            normalized.push(path);
        }
    }
    Ok(normalized)
}

fn normalize_unique_ids(field: &str, ids: Vec<String>) -> Result<Vec<String>> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::with_capacity(ids.len());
    for (index, id) in ids.into_iter().enumerate() {
        let id = id.trim().to_string();
        if id.is_empty() {
            bail!("{field}[{index}] must not be empty");
        }
        if seen.insert(id.clone()) {
            normalized.push(id);
        }
    }
    Ok(normalized)
}

fn normalize_relative_path(raw: &str, allow_dot: bool) -> Result<String> {
    let trimmed = raw.trim().replace('\\', "/");
    if trimmed.is_empty() {
        bail!("path must not be empty");
    }
    let path = Path::new(&trimmed);
    if path.is_absolute() {
        bail!("path must be relative");
    }
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("path must stay inside the workspace")
            }
        }
    }
    if parts.is_empty() {
        if allow_dot {
            return Ok(".".to_string());
        }
        bail!("path must name a workspace entry");
    }
    Ok(parts.join("/"))
}

fn default_true() -> bool {
    true
}

fn default_check_cwd() -> String {
    ".".to_string()
}

fn default_check_timeout_seconds() -> u64 {
    DEFAULT_CHECK_TIMEOUT_SECONDS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_normalization_rejects_unknown_review_checks_and_escaping_paths() {
        let escaping: HarnessContractDocument =
            serde_json::from_str(r#"{"version":1,"allowed_paths":["../escape"],"checks":[]}"#)
                .unwrap();
        assert!(
            escaping
                .normalize()
                .unwrap_err()
                .to_string()
                .contains("allowed_paths")
        );

        let unknown: HarnessContractDocument = serde_json::from_str(
            r#"{"version":1,"checks":[],"review":{"required":true,"check_ids":["missing"]}}"#,
        )
        .unwrap();
        assert!(
            unknown
                .normalize()
                .unwrap_err()
                .to_string()
                .contains("unknown check")
        );
    }

    #[test]
    fn contract_normalization_applies_check_defaults() {
        let document: HarnessContractDocument = serde_json::from_str(
            r#"{"version":1,"mutation":"required","checks":[{"id":"logic","command":"deno test"}]}"#,
        )
        .unwrap();
        let contract = document.normalize().unwrap();
        assert_eq!(contract.mutation, MutationRequirement::Required);
        assert_eq!(contract.checks[0].cwd, ".");
        assert_eq!(contract.checks[0].timeout_seconds, 60);
        assert!(contract.checks[0].required);
        assert!(!contract.checks[0].diagnostic_eligible);
    }

    #[test]
    fn diagnostic_preview_requires_an_explicit_bounded_check() {
        let document: HarnessContractDocument = serde_json::from_str(
            r#"{"version":1,"checks":[{"id":"slow","command":"slow test","diagnostic_eligible":true,"timeout_seconds":61}]}"#,
        )
        .unwrap();
        assert!(
            document
                .normalize()
                .unwrap_err()
                .to_string()
                .contains("diagnostic_eligible timeout")
        );
    }

    #[test]
    fn contract_checks_compile_into_always_ordered_workspace_checks() {
        let document: HarnessContractDocument = serde_json::from_str(
            r#"{"version":1,"checks":[{"id":"build","command":"make build"},{"id":"test","command":"make test"}]}"#,
        )
        .unwrap();
        let contract = document.normalize().unwrap();
        let graph = contract
            .compile_workspace_graph(crate::workspace::WorkspaceGraph::legacy(&[]))
            .unwrap();

        assert_eq!(
            graph.checks["build"].trigger,
            crate::workspace::CheckTrigger::Always
        );
        assert_eq!(graph.checks["test"].depends_on, vec!["build"]);
        assert_eq!(graph.checks["test"].executor, "harness-contract");
        assert_eq!(graph.components["harness-contract"].root, ".");
    }

    #[test]
    fn contract_check_conflicts_with_trusted_workspace_definition() {
        let document: HarnessContractDocument =
            serde_json::from_str(r#"{"version":1,"checks":[{"id":"test","command":"make test"}]}"#)
                .unwrap();
        let contract = document.normalize().unwrap();
        let mut graph = contract
            .compile_workspace_graph(crate::workspace::WorkspaceGraph::legacy(&[]))
            .unwrap();
        graph.checks.get_mut("test").unwrap().command = "other test".to_string();

        let error = contract.compile_workspace_graph(graph).unwrap_err();
        assert!(error.to_string().contains("conflicts"));
    }

    #[test]
    fn checked_in_task_completion_contracts_normalize() {
        let fixtures = [
            (
                "tc1",
                include_str!("../fixtures/harness-task-completion/tc1-contract.json"),
                2,
                1,
            ),
            (
                "tc2",
                include_str!("../fixtures/harness-task-completion/tc2-contract.json"),
                3,
                3,
            ),
        ];

        for (name, fixture, expected_paths, expected_checks) in fixtures {
            let document: HarnessContractDocument = serde_json::from_str(fixture)
                .unwrap_or_else(|error| panic!("{name} contract must parse: {error}"));
            let contract = document
                .normalize()
                .unwrap_or_else(|error| panic!("{name} contract must normalize: {error}"));
            assert_eq!(contract.mutation, MutationRequirement::Required);
            assert_eq!(contract.allowed_paths.len(), expected_paths);
            assert_eq!(contract.checks.len(), expected_checks);
            assert!(contract.commit.required);
            assert!(contract.commit.semantic);
            assert!(contract.review.required);
            assert!(contract.workspace_clean);
        }
    }

    #[test]
    fn checked_in_task_corpus_contracts_normalize() {
        let corpus: serde_json::Value = serde_json::from_str(include_str!(
            "../fixtures/harness-task-completion/corpus.json"
        ))
        .unwrap();
        assert_eq!(corpus["version"], 1);
        let cases = corpus["cases"].as_array().expect("corpus cases");
        assert!((10..=20).contains(&cases.len()));
        let mut case_ids = HashSet::new();

        for fixture in cases {
            let case_id = fixture["id"].as_str().expect("case id");
            assert!(case_ids.insert(case_id), "duplicate case id {case_id}");
            let document: HarnessContractDocument =
                serde_json::from_value(fixture["contract"].clone())
                    .unwrap_or_else(|error| panic!("{case_id} contract must parse: {error}"));
            let contract = document
                .normalize()
                .unwrap_or_else(|error| panic!("{case_id} contract must normalize: {error}"));
            assert!(!contract.checks.is_empty(), "{case_id} checks");
            assert!(contract.workspace_clean, "{case_id} clean worktree");
            if contract.mutation == MutationRequirement::Required {
                assert!(
                    !contract.allowed_paths.is_empty(),
                    "{case_id} allowed paths"
                );
                assert!(contract.commit.required, "{case_id} required commit");
                assert!(contract.commit.semantic, "{case_id} semantic commit");
                assert!(contract.review.required, "{case_id} required review");
            }
        }
    }
}
