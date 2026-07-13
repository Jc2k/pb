//! Deterministic control-plane fixtures for the hidden harness.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::agent_core::{
    AgentProfile, AgentRequest, LocalModelEvalEngine, LocalModelEvalOutcome, ScriptedAgentOutcome,
    ScriptedCompletion, git_worktree_content_fingerprint, run_local_model_eval_steps,
    run_scripted_agent_steps,
};
use crate::events::{AgentEvent, ContractStatus};
use crate::{HarnessEvalArgs, config::UserConfig};

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
    pub expected: ControlFixtureExpectation,
    #[serde(default)]
    pub contract: Option<crate::harness_contract::HarnessContractDocument>,
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
pub struct ControlFixtureExpectation {
    pub reached_final: bool,
    pub contract_status: ContractStatus,
    pub verified_completed: bool,
    pub termination_reason: String,
    pub llm_invocations: usize,
    pub tool_calls: usize,
    pub false_completion: bool,
    #[serde(default)]
    pub named_check_compliance: Option<bool>,
    #[serde(default)]
    pub observed_paths: BTreeMap<String, bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HarnessEvalConfiguration {
    pub mode: String,
    pub backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_dir: Option<String>,
    pub max_tokens: i32,
    pub ctx_size: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads_batch: Option<i32>,
    pub gpu_layers: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub seed: u32,
    pub flashmoe_resource_policy_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HarnessEvalRecord {
    pub schema_version: u32,
    pub fixture_version: u32,
    pub configuration: HarnessEvalConfiguration,
    pub protocol_pass: bool,
    pub protocol_failures: Vec<String>,
    pub result: ControlFixtureResult,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlFixtureTurn {
    pub content: String,
    #[serde(default)]
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlFixtureResult {
    pub id: String,
    pub reached_final: bool,
    #[serde(default)]
    pub contract_status: ContractStatus,
    #[serde(default)]
    pub verified_completed: bool,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub named_check_compliance: Option<bool>,
    #[serde(default)]
    pub recovery_loop: bool,
    #[serde(default)]
    pub latency_ms: u64,
    #[serde(default)]
    pub prompt_tokens: usize,
    #[serde(default)]
    pub generated_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_kwh: Option<f64>,
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
        contract: fixture
            .contract
            .clone()
            .map(crate::harness_contract::HarnessContractDocument::normalize)
            .transpose()?,
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

    summarize_fixture(fixture, scratch.path(), outcome.into(), &events)
}

pub fn run_control_fixture_corpus() -> Result<Vec<ControlFixtureResult>> {
    control_fixture_corpus()?
        .fixtures
        .iter()
        .map(run_control_fixture)
        .collect()
}

pub fn run_eval_command(args: HarnessEvalArgs) -> Result<()> {
    let corpus = control_fixture_corpus()?;
    let (configuration, results) = match args.model.as_deref() {
        Some(model) => run_real_model_corpus(&args, model, &corpus)?,
        None => (
            scripted_configuration(),
            corpus
                .fixtures
                .iter()
                .map(run_control_fixture)
                .collect::<Result<Vec<_>>>()?,
        ),
    };
    let records = build_eval_records(&corpus, configuration, results)?;
    write_eval_jsonl(args.jsonl.as_deref(), &records)?;
    let table = render_eval_table(&records);
    if args.jsonl.is_some() {
        print!("{table}");
    } else {
        eprint!("{table}");
    }
    let failed = records
        .iter()
        .filter(|record| !record.protocol_pass)
        .map(|record| record.result.id.as_str())
        .collect::<Vec<_>>();
    if !failed.is_empty() {
        bail!("harness protocol regressions: {}", failed.join(", "));
    }
    Ok(())
}

fn scripted_configuration() -> HarnessEvalConfiguration {
    HarnessEvalConfiguration {
        mode: "scripted".to_string(),
        backend: "scripted".to_string(),
        model: None,
        model_dir: None,
        max_tokens: 256,
        ctx_size: 1024,
        threads: None,
        threads_batch: None,
        gpu_layers: 0,
        temperature: 0.0,
        top_k: 1,
        seed: 0,
        flashmoe_resource_policy_version:
            crate::inference::flashmoe::HARNESS_RESOURCE_POLICY_VERSION,
    }
}

fn run_real_model_corpus(
    args: &HarnessEvalArgs,
    model: &str,
    corpus: &ControlFixtureCorpus,
) -> Result<(HarnessEvalConfiguration, Vec<ControlFixtureResult>)> {
    let user_config = UserConfig::load()?;
    let models_root = args
        .model_dir
        .clone()
        .or_else(|| user_config.effective_model_dir())
        .unwrap_or_else(crate::default_models_dir);
    let mut engine = LocalModelEvalEngine::load(model, &models_root, args.gpu_layers)?;
    ensure_flashmoe_eval_policy(
        engine.backend_name(),
        crate::inference::flashmoe::HARNESS_RESOURCE_POLICY_VERSION,
    )?;
    let configuration = HarnessEvalConfiguration {
        mode: "local_model".to_string(),
        backend: engine.backend_name().to_string(),
        model: Some(model.to_string()),
        model_dir: Some(
            models_root
                .canonicalize()
                .unwrap_or(models_root.clone())
                .display()
                .to_string(),
        ),
        max_tokens: args.max_tokens,
        ctx_size: args.ctx_size,
        threads: args.threads,
        threads_batch: args.threads_batch,
        gpu_layers: args.gpu_layers,
        temperature: args.temperature,
        top_k: args.top_k,
        seed: args.seed,
        flashmoe_resource_policy_version:
            crate::inference::flashmoe::HARNESS_RESOURCE_POLICY_VERSION,
    };
    let mut results = Vec::with_capacity(corpus.fixtures.len());
    for fixture in &corpus.fixtures {
        results.push(run_real_model_fixture(
            fixture,
            &configuration,
            &mut engine,
        )?);
    }
    Ok((configuration, results))
}

fn ensure_flashmoe_eval_policy(backend: &str, policy_version: u32) -> Result<()> {
    if backend == "flashmoe" && policy_version == 0 {
        bail!("FlashMoe harness evaluation is disabled until a bounded resource policy is active");
    }
    Ok(())
}

fn run_real_model_fixture(
    fixture: &ControlFixture,
    configuration: &HarnessEvalConfiguration,
    engine: &mut LocalModelEvalEngine,
) -> Result<ControlFixtureResult> {
    let scratch = tempfile::Builder::new()
        .prefix("pb-model-control-fixture-")
        .tempdir()
        .context("failed to create model harness control fixture scratch directory")?;
    initialize_fixture_workspace(scratch.path(), &fixture.initial_files)?;
    let request = AgentRequest {
        task: format!(
            "Harness control evaluation. Complete the repository task implied by this control objective, using only the exposed tools: {}",
            fixture.hypothesis
        ),
        model: configuration.model.clone().unwrap_or_default(),
        model_dir: configuration.model_dir.as_ref().map(PathBuf::from),
        workdir: Some(scratch.path().to_path_buf()),
        branch: None,
        max_steps: fixture.max_steps,
        max_tokens: configuration.max_tokens,
        turn_max_tokens_cap: Some(configuration.max_tokens),
        tool_allowlist: Some(fixture.tool_allowlist.clone()),
        accept_existing_workspace_changes: false,
        ctx_size: configuration.ctx_size,
        threads: configuration.threads,
        threads_batch: configuration.threads_batch,
        gpu_layers: configuration.gpu_layers,
        temperature: configuration.temperature,
        profile: fixture.profile,
        infer_profile: false,
        sub_agent_depth: 0,
        repository_less: false,
        top_k: configuration.top_k,
        seed: configuration.seed,
        environment: None,
        session_id: format!("harness-eval-{}", fixture.id),
        attachments: Vec::new(),
        contract: fixture
            .contract
            .clone()
            .map(crate::harness_contract::HarnessContractDocument::normalize)
            .transpose()?,
    };
    let mut events = Vec::new();
    let outcome = run_local_model_eval_steps(engine, &request, scratch.path(), &mut |event| {
        events.push(event)
    })?;
    summarize_fixture(fixture, scratch.path(), outcome.into(), &events)
}

fn build_eval_records(
    corpus: &ControlFixtureCorpus,
    configuration: HarnessEvalConfiguration,
    results: Vec<ControlFixtureResult>,
) -> Result<Vec<HarnessEvalRecord>> {
    if results.len() != corpus.fixtures.len() {
        bail!(
            "harness evaluation produced {} results for {} fixtures",
            results.len(),
            corpus.fixtures.len()
        );
    }
    corpus
        .fixtures
        .iter()
        .zip(results)
        .map(|(fixture, result)| {
            if fixture.id != result.id {
                bail!(
                    "harness evaluation result order mismatch: expected {}, got {}",
                    fixture.id,
                    result.id
                );
            }
            let protocol_failures = protocol_failures(&fixture.expected, &result);
            Ok(HarnessEvalRecord {
                schema_version: 1,
                fixture_version: corpus.version,
                configuration: configuration.clone(),
                protocol_pass: protocol_failures.is_empty(),
                protocol_failures,
                result,
            })
        })
        .collect()
}

fn protocol_failures(
    expected: &ControlFixtureExpectation,
    actual: &ControlFixtureResult,
) -> Vec<String> {
    let mut failures = Vec::new();
    macro_rules! compare {
        ($field:ident) => {
            if actual.$field != expected.$field {
                failures.push(format!(
                    "{} expected {:?}, got {:?}",
                    stringify!($field),
                    expected.$field,
                    actual.$field
                ));
            }
        };
    }
    compare!(reached_final);
    compare!(contract_status);
    compare!(verified_completed);
    compare!(termination_reason);
    compare!(llm_invocations);
    compare!(tool_calls);
    compare!(false_completion);
    compare!(named_check_compliance);
    compare!(observed_paths);
    failures
}

fn write_eval_jsonl(path: Option<&Path>, records: &[HarnessEvalRecord]) -> Result<()> {
    let mut writer: Box<dyn Write> = match path {
        Some(path) => Box::new(BufWriter::new(File::create(path).with_context(|| {
            format!(
                "failed to create harness evaluation JSONL {}",
                path.display()
            )
        })?)),
        None => Box::new(BufWriter::new(std::io::stdout().lock())),
    };
    for record in records {
        serde_json::to_writer(&mut writer, record)
            .context("failed to encode harness evaluation JSONL")?;
        writer
            .write_all(b"\n")
            .context("failed to write harness evaluation JSONL")?;
    }
    writer
        .flush()
        .context("failed to flush harness evaluation JSONL")
}

fn render_eval_table(records: &[HarnessEvalRecord]) -> String {
    let mut table = String::from(
        "fixture                         pass valid named false loop turns latency_ms tokens energy_kwh termination\n",
    );
    for record in records {
        let result = &record.result;
        let valid = if result.llm_invocations == 0 {
            "0/0".to_string()
        } else {
            format!("{}/{}", result.valid_actions, result.llm_invocations)
        };
        let named = result
            .named_check_compliance
            .map(|value| if value { "yes" } else { "no" })
            .unwrap_or("-");
        let energy = result
            .energy_kwh
            .map(|value| format!("{value:.3e}"))
            .unwrap_or_else(|| "-".to_string());
        table.push_str(&format!(
            "{:<31} {:<4} {:<5} {:<5} {:<5} {:<4} {:<5} {:<10} {:<6} {:<10} {}\n",
            result.id,
            if record.protocol_pass { "yes" } else { "no" },
            valid,
            named,
            if result.false_completion { "yes" } else { "no" },
            if result.recovery_loop { "yes" } else { "no" },
            result.llm_invocations,
            result.latency_ms,
            result.prompt_tokens.saturating_add(result.generated_tokens),
            energy,
            result.termination_reason,
        ));
    }
    table
}

struct FixtureOutcome {
    reached_final: bool,
    contract_status: ContractStatus,
    verified_completed: bool,
    termination_reason: String,
    remaining_completions: usize,
}

impl From<ScriptedAgentOutcome> for FixtureOutcome {
    fn from(outcome: ScriptedAgentOutcome) -> Self {
        Self {
            reached_final: outcome.reached_final,
            contract_status: outcome.contract_status,
            verified_completed: outcome.verified_completed,
            termination_reason: outcome.termination_reason.to_string(),
            remaining_completions: outcome.remaining_completions,
        }
    }
}

impl From<LocalModelEvalOutcome> for FixtureOutcome {
    fn from(outcome: LocalModelEvalOutcome) -> Self {
        Self {
            reached_final: outcome.reached_final,
            contract_status: outcome.contract_status,
            verified_completed: outcome.verified_completed,
            termination_reason: outcome.termination_reason.to_string(),
            remaining_completions: 0,
        }
    }
}

fn summarize_fixture(
    fixture: &ControlFixture,
    workspace: &Path,
    outcome: FixtureOutcome,
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
                        || summary == "Acceptance contract rejected final response"
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
    let (latency_ms, prompt_tokens, generated_tokens, energy_kwh) = events.iter().fold(
        (0u64, 0usize, 0usize, 0.0f64),
        |(latency, prompt, generated, energy), event| match event {
            AgentEvent::LlmInvocation {
                duration_ms,
                prompt_tokens,
                generated_tokens,
                energy_kwh,
                ..
            } => (
                latency.saturating_add(*duration_ms),
                prompt.saturating_add(*prompt_tokens),
                generated.saturating_add(*generated_tokens),
                energy + energy_kwh.unwrap_or(0.0),
            ),
            _ => (latency, prompt, generated, energy),
        },
    );
    let required_check_ids = fixture
        .contract
        .as_ref()
        .map(|contract| {
            contract
                .checks
                .iter()
                .filter(|check| check.required)
                .map(|check| check.id.as_str())
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();
    let named_check_compliance = if required_check_ids.is_empty() {
        None
    } else {
        let fingerprint = git_worktree_content_fingerprint(workspace)?;
        let successful = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::CheckResult {
                    check_id,
                    success: true,
                    fingerprint: check_fingerprint,
                    ..
                } if check_fingerprint == &fingerprint => Some(check_id.as_str()),
                _ => None,
            })
            .collect::<HashSet<_>>();
        Some(required_check_ids.is_subset(&successful))
    };
    let mut observed_paths = BTreeMap::new();
    for relative in &fixture.observe_paths {
        let path = fixture_path(workspace, relative)?;
        observed_paths.insert(relative.clone(), path.exists());
    }

    Ok(ControlFixtureResult {
        id: fixture.id.clone(),
        reached_final: outcome.reached_final,
        contract_status: outcome.contract_status,
        verified_completed: outcome.verified_completed,
        termination_reason: outcome.termination_reason.clone(),
        valid_actions: llm_invocations.saturating_sub(invalid_actions),
        llm_invocations,
        tool_calls: events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolCall { .. }))
            .count(),
        corrections,
        gate_corrections,
        errors,
        blocked_tool_loops,
        final_events,
        executed_checks,
        false_completion: outcome.reached_final
            && outcome.contract_status != ContractStatus::Unsatisfied
            && !fixture.completion_supported,
        named_check_compliance,
        recovery_loop: matches!(
            outcome.termination_reason.as_str(),
            "gate_loop" | "parse_loop"
        ),
        latency_ms,
        prompt_tokens,
        generated_tokens,
        energy_kwh: (energy_kwh > 0.0).then_some(energy_kwh),
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
        assert_eq!(baseline.results.len(), actual.len());
        assert!(
            baseline
                .results
                .iter()
                .find(|result| result.id == "irrelevant_review_evidence")
                .is_some_and(|result| result.false_completion),
            "historical baseline must preserve the pre-contract false completion"
        );
        for id in ["irrelevant_review_evidence", "check_then_mutation"] {
            let result = actual.iter().find(|result| result.id == id).unwrap();
            assert!(result.reached_final, "{id} emitted a final action");
            assert_eq!(result.contract_status, ContractStatus::Unsatisfied, "{id}");
            assert!(!result.verified_completed, "{id} must not be verified");
            assert_eq!(result.termination_reason, "contract_unsatisfied", "{id}");
            assert!(!result.false_completion, "{id} must not falsely complete");
            assert_eq!(result.gate_corrections, 1, "{id}");
        }
        assert_eq!(
            actual
                .iter()
                .find(|result| result.id == "check_then_mutation")
                .unwrap()
                .executed_checks,
            1
        );
        let repeated = actual
            .iter()
            .find(|result| result.id == "repeated_blocked_action")
            .unwrap();
        assert!(!repeated.reached_final);
        assert_eq!(repeated.termination_reason, "gate_loop");
        assert_eq!(repeated.llm_invocations, 3);
        assert_eq!(repeated.blocked_tool_loops, 1);
        assert_eq!(repeated.remaining_completions, 1);
    }

    #[test]
    fn control_fixture_paths_cannot_escape_scratch() {
        let root = tempfile::tempdir().unwrap();
        assert!(fixture_path(root.path(), "safe/nested.txt").is_ok());
        assert!(fixture_path(root.path(), "../escape.txt").is_err());
        assert!(fixture_path(root.path(), "/tmp/escape.txt").is_err());
        assert!(fixture_path(root.path(), ".git/config").is_err());
    }

    #[test]
    fn scripted_eval_report_is_deterministic_and_protocol_complete() {
        let corpus = control_fixture_corpus().unwrap();
        let build = || {
            build_eval_records(
                &corpus,
                scripted_configuration(),
                run_control_fixture_corpus().unwrap(),
            )
            .unwrap()
        };
        let first = build();
        let second = build();

        assert_eq!(first, second);
        assert!(first.iter().all(|record| record.protocol_pass));
        assert!(
            first
                .iter()
                .all(|record| record.result.artifact_quality.is_none())
        );
        let first_jsonl = first
            .iter()
            .map(|record| serde_json::to_string(record).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let second_jsonl = second
            .iter()
            .map(|record| serde_json::to_string(record).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(first_jsonl, second_jsonl);
        for line in first_jsonl.lines() {
            let record: HarnessEvalRecord = serde_json::from_str(line).unwrap();
            assert_eq!(record.schema_version, 1);
            assert_eq!(record.configuration.mode, "scripted");
        }
    }

    #[test]
    fn protocol_scoring_ignores_artifact_quality_but_detects_control_regressions() {
        let corpus = control_fixture_corpus().unwrap();
        let fixture = &corpus.fixtures[0];
        let mut result = run_control_fixture(fixture).unwrap();
        result.artifact_quality = Some("intentionally not scored".to_string());
        assert!(protocol_failures(&fixture.expected, &result).is_empty());

        result.termination_reason = "engine_error".to_string();
        let failures = protocol_failures(&fixture.expected, &result);
        assert!(
            failures
                .iter()
                .any(|failure| failure.starts_with("termination_reason"))
        );
    }

    #[test]
    fn eval_table_covers_control_and_resource_metrics() {
        let corpus = control_fixture_corpus().unwrap();
        let records = build_eval_records(
            &corpus,
            scripted_configuration(),
            run_control_fixture_corpus().unwrap(),
        )
        .unwrap();
        let table = render_eval_table(&records);
        for heading in [
            "valid",
            "named",
            "false",
            "loop",
            "turns",
            "latency_ms",
            "tokens",
            "energy_kwh",
            "termination",
        ] {
            assert!(table.contains(heading), "missing {heading}: {table}");
        }
    }

    #[test]
    fn flashmoe_model_eval_requires_versioned_resource_policy() {
        assert!(ensure_flashmoe_eval_policy("flashmoe", 0).is_err());
        assert!(ensure_flashmoe_eval_policy("llama_cpp", 0).is_ok());
        assert!(
            ensure_flashmoe_eval_policy(
                "flashmoe",
                crate::inference::flashmoe::HARNESS_RESOURCE_POLICY_VERSION,
            )
            .is_ok()
        );
    }

    #[test]
    fn real_model_configuration_serializes_reproduction_parameters() {
        let configuration = HarnessEvalConfiguration {
            mode: "local_model".to_string(),
            backend: "llama_cpp".to_string(),
            model: Some("model.gguf".to_string()),
            model_dir: Some("/models".to_string()),
            max_tokens: 512,
            ctx_size: 32768,
            threads: Some(8),
            threads_batch: Some(12),
            gpu_layers: 99,
            temperature: 0.0,
            top_k: 1,
            seed: 42,
            flashmoe_resource_policy_version: 1,
        };
        let value = serde_json::to_value(configuration).unwrap();
        for field in [
            "mode",
            "backend",
            "model",
            "model_dir",
            "max_tokens",
            "ctx_size",
            "threads",
            "threads_batch",
            "gpu_layers",
            "temperature",
            "top_k",
            "seed",
            "flashmoe_resource_policy_version",
        ] {
            assert!(value.get(field).is_some(), "missing {field}: {value}");
        }
    }
}
