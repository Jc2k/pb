//! Deterministic control-plane fixtures for the hidden harness.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::agent_core::{
    AgentProfile, AgentRequest, ScriptedAgentOutcome, ScriptedCompletion, run_scripted_agent_steps,
};
use crate::events::AgentEvent;

const CONTROL_FIXTURES: &str = include_str!("../fixtures/harness-control-fixtures.json");

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlFixtureCorpus {
    pub version: u32,
    pub fixtures: Vec<ControlFixture>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlFixture {
    pub id: String,
    pub hypothesis: String,
    pub profile: AgentProfile,
    pub max_steps: usize,
    pub completion_supported: bool,
    #[serde(default)]
    pub tool_allowlist: Vec<String>,
    #[serde(default)]
    pub initial_files: BTreeMap<String, String>,
    pub turns: Vec<ControlFixtureTurn>,
    #[serde(default)]
    pub observe_paths: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlFixtureTurn {
    pub content: String,
    #[serde(default)]
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ControlFixtureResult {
    pub id: String,
    pub reached_final: bool,
    pub termination_reason: String,
    pub valid_actions: usize,
    pub llm_invocations: usize,
    pub tool_calls: usize,
    pub corrections: usize,
    pub gate_corrections: usize,
    pub errors: usize,
    pub blocked_tool_loops: usize,
    pub final_events: usize,
    pub executed_checks: usize,
    pub false_completion: bool,
    pub remaining_completions: usize,
    pub observed_paths: BTreeMap<String, bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_quality: Option<String>,
}

pub fn control_fixture_corpus() -> Result<ControlFixtureCorpus> {
    let corpus: ControlFixtureCorpus = serde_json::from_str(CONTROL_FIXTURES)
        .context("failed to parse built-in harness control fixtures")?;
    if corpus.version != 1 {
        bail!(
            "unsupported harness control fixture version {}; expected 1",
            corpus.version
        );
    }
    if corpus.fixtures.is_empty() {
        bail!("harness control fixture corpus must not be empty");
    }
    let mut ids = std::collections::HashSet::new();
    for fixture in &corpus.fixtures {
        if fixture.id.trim().is_empty() {
            bail!("harness control fixture id must not be empty");
        }
        if !ids.insert(fixture.id.as_str()) {
            bail!("duplicate harness control fixture id '{}'", fixture.id);
        }
        if fixture.hypothesis.trim().is_empty() {
            bail!("harness control fixture '{}' has no hypothesis", fixture.id);
        }
        if fixture.max_steps == 0 || fixture.turns.is_empty() {
            bail!(
                "harness control fixture '{}' needs positive steps and at least one turn",
                fixture.id
            );
        }
    }
    Ok(corpus)
}

pub fn run_control_fixture(fixture: &ControlFixture) -> Result<ControlFixtureResult> {
    let scratch = tempfile::Builder::new()
        .prefix("pb-control-fixture-")
        .tempdir()
        .context("failed to create harness control fixture scratch directory")?;
    initialize_fixture_workspace(scratch.path(), &fixture.initial_files)?;

    let args = AgentRequest {
        task: fixture.hypothesis.clone(),
        model: "scripted-control-fixture".to_string(),
        model_dir: None,
        workdir: Some(scratch.path().to_path_buf()),
        branch: None,
        max_steps: fixture.max_steps,
        max_tokens: 256,
        turn_max_tokens_cap: Some(256),
        tool_allowlist: Some(fixture.tool_allowlist.clone()),
        accept_existing_workspace_changes: false,
        ctx_size: 1024,
        threads: None,
        threads_batch: None,
        gpu_layers: 0,
        temperature: 0.0,
        profile: fixture.profile,
        infer_profile: false,
        sub_agent_depth: 0,
        repository_less: false,
        top_k: 1,
        seed: 0,
        environment: None,
        session_id: format!("control-fixture-{}", fixture.id),
        attachments: Vec::new(),
    };
    let completions = fixture
        .turns
        .iter()
        .map(|turn| ScriptedCompletion {
            content: turn.content.clone(),
            truncated: turn.truncated,
        })
        .collect();
    let mut events = Vec::new();
    let outcome = run_scripted_agent_steps(&args, completions, scratch.path(), &mut |event| {
        events.push(event)
    })?;

    summarize_fixture(fixture, scratch.path(), outcome, &events)
}

pub fn run_control_fixture_corpus() -> Result<Vec<ControlFixtureResult>> {
    control_fixture_corpus()?
        .fixtures
        .iter()
        .map(run_control_fixture)
        .collect()
}

fn summarize_fixture(
    fixture: &ControlFixture,
    workspace: &Path,
    outcome: ScriptedAgentOutcome,
    events: &[AgentEvent],
) -> Result<ControlFixtureResult> {
    let llm_invocations = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::LlmInvocation { .. }))
        .count();
    let invalid_actions = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::Error { summary, .. }
                    if summary.starts_with("Invalid pb JSON action")
            )
        })
        .count();
    let corrections = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::Correction { .. }))
        .count();
    let gate_corrections = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::Correction { summary, .. }
                    if summary == "Completion gate blocked final response"
            )
        })
        .count();
    let errors = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::Error { .. }))
        .count();
    let blocked_tool_loops = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::Correction { summary, .. }
                    if summary == "Repeated tool call blocked"
            )
        })
        .count();
    let final_events = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::Final { .. }))
        .count();
    let executed_checks = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::ToolCall { tool, .. } if tool == "run_check"
            )
        })
        .count();
    let mut observed_paths = BTreeMap::new();
    for relative in &fixture.observe_paths {
        let path = fixture_path(workspace, relative)?;
        observed_paths.insert(relative.clone(), path.exists());
    }

    Ok(ControlFixtureResult {
        id: fixture.id.clone(),
        reached_final: outcome.reached_final,
        termination_reason: if outcome.reached_final {
            "final".to_string()
        } else {
            "step_limit".to_string()
        },
        valid_actions: llm_invocations.saturating_sub(invalid_actions),
        llm_invocations,
        tool_calls: outcome.tool_calls,
        corrections,
        gate_corrections,
        errors,
        blocked_tool_loops,
        final_events,
        executed_checks,
        false_completion: outcome.reached_final && !fixture.completion_supported,
        remaining_completions: outcome.remaining_completions,
        observed_paths,
        artifact_quality: None,
    })
}

fn initialize_fixture_workspace(root: &Path, files: &BTreeMap<String, String>) -> Result<()> {
    run_git(root, &["init", "--initial-branch=main"])?;
    run_git(root, &["config", "user.name", "pb harness fixture"])?;
    run_git(root, &["config", "user.email", "fixture@pb.local"])?;
    for (relative, content) in files {
        let path = fixture_path(root, relative)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        std::fs::write(&path, content)
            .with_context(|| format!("failed to write fixture file {}", path.display()))?;
    }
    run_git(root, &["add", "-A"])?;
    run_git(
        root,
        &["commit", "--allow-empty", "-m", "test: initialize fixture"],
    )?;
    Ok(())
}

fn fixture_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || path.starts_with(".git")
    {
        bail!("invalid harness control fixture path '{relative}'");
    }
    Ok(root.join(path))
}

fn run_git(root: &Path, args: &[&str]) -> Result<()> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASELINE: &str = include_str!("../docs/harness-control-baseline.json");

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct BaselineReport {
        fixture_version: u32,
        captured_at: String,
        results: Vec<ControlFixtureResult>,
        observations: Vec<BaselineObservation>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct BaselineObservation {
        id: String,
        classification: String,
        priority: String,
        evidence: String,
    }

    #[test]
    fn control_fixture_corpus_matches_checked_in_baseline() {
        let corpus = control_fixture_corpus().unwrap();
        let actual = run_control_fixture_corpus().unwrap();
        let baseline: BaselineReport = serde_json::from_str(BASELINE).unwrap();

        assert_eq!(baseline.fixture_version, corpus.version);
        assert_eq!(baseline.captured_at, "2026-07-13");
        let fixture_ids = corpus
            .fixtures
            .iter()
            .map(|fixture| fixture.id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let observation_ids = baseline
            .observations
            .iter()
            .map(|observation| observation.id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(observation_ids, fixture_ids);
        for observation in &baseline.observations {
            assert!(
                matches!(
                    observation.classification.as_str(),
                    "pb_defect" | "model_limitation" | "experiment_error" | "positive_evidence"
                ),
                "invalid classification for {}",
                observation.id
            );
            assert!(!observation.priority.trim().is_empty());
            assert!(!observation.evidence.trim().is_empty());
        }
        assert_eq!(
            actual, baseline.results,
            "deterministic harness control behavior changed; update runtime assertions first and preserve this historical baseline separately when the change is intentional"
        );
    }

    #[test]
    fn control_fixture_paths_cannot_escape_scratch() {
        let root = tempfile::tempdir().unwrap();
        assert!(fixture_path(root.path(), "safe/nested.txt").is_ok());
        assert!(fixture_path(root.path(), "../escape.txt").is_err());
        assert!(fixture_path(root.path(), "/tmp/escape.txt").is_err());
        assert!(fixture_path(root.path(), ".git/config").is_err());
    }
}
