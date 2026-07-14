//! Deterministic control-plane fixtures for the hidden harness.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::agent_core::{
    AgentProfile, AgentRequest, LocalModelEvalEngine, LocalModelEvalOutcome, ScriptedAgentOutcome,
    ScriptedCompletion, run_local_model_eval_steps, run_scripted_agent_steps,
};
use crate::events::{AgentEvent, ContractStatus};
use crate::{HarnessEvalArgs, config::UserConfig};

const CONTROL_FIXTURES: &str = include_str!("../fixtures/harness-control-fixtures.json");
const HARNESS_EVAL_SCHEMA_VERSION: u32 = 2;

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
    pub workspace: Option<crate::workspace::WorkspaceConfigDocument>,
    #[serde(default)]
    pub tool_allowlist: Vec<String>,
    #[serde(default)]
    pub initial_files: BTreeMap<String, String>,
    #[serde(default)]
    pub resumed_files: BTreeMap<String, String>,
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
    #[serde(default)]
    pub handoff_outcome: Option<crate::events::HandoffOutcome>,
    #[serde(default)]
    pub selected_checks: Option<Vec<String>>,
    #[serde(default)]
    pub executed_checks: Option<usize>,
    #[serde(default)]
    pub reused_checks: Option<usize>,
    #[serde(default)]
    pub executor_starts: Option<Vec<String>>,
    #[serde(default)]
    pub repair_turns: Option<usize>,
    #[serde(default)]
    pub commit_disposition: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_config_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub executor_policy: Vec<String>,
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
    #[serde(default)]
    pub model_run_check_calls: usize,
    #[serde(default)]
    pub reused_checks: usize,
    #[serde(default)]
    pub failed_checks: usize,
    #[serde(default)]
    pub skipped_checks: usize,
    #[serde(default)]
    pub selected_checks: Vec<String>,
    #[serde(default)]
    pub affected_components: Vec<String>,
    #[serde(default)]
    pub executor_starts: Vec<String>,
    #[serde(default)]
    pub avoided_executor_starts: Vec<String>,
    #[serde(default)]
    pub team_messages: usize,
    #[serde(default)]
    pub repair_turns: usize,
    #[serde(default)]
    pub no_change: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff_outcome: Option<crate::events::HandoffOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_disposition: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_oid: Option<String>,
    #[serde(default)]
    pub output_fingerprints: Vec<String>,
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
    parse_control_fixture_corpus(CONTROL_FIXTURES)
}

fn parse_control_fixture_corpus(contents: &str) -> Result<ControlFixtureCorpus> {
    let corpus: ControlFixtureCorpus = serde_json::from_str(contents)
        .context("failed to parse built-in harness control fixtures")?;
    if corpus.version != 2 {
        bail!(
            "unsupported harness control fixture version {}; expected 2",
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
    let (contract, workspace_graph, repository_context) = fixture_runtime(fixture, scratch.path())?;

    let args = AgentRequest {
        task: fixture.hypothesis.clone(),
        turn_id: format!("control-fixture-turn-{}", fixture.id),
        intent: Some(crate::workflow::TurnIntent::Deliver),
        workflow_policy: None,
        workflow_stage: None,
        conversation_handoff: None,
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
        workspace_graph: Some(workspace_graph),
        repository_context: Some(repository_context),
        prior_check_evidence: crate::checks::CheckEvidenceLedger::default(),
        session_id: format!("control-fixture-{}", fixture.id),
        attachments: Vec::new(),
        contract,
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

    summarize_fixture(fixture, scratch.path(), outcome.into(), &events, false)
}

fn fixture_runtime(
    fixture: &ControlFixture,
    workspace: &Path,
) -> Result<(
    Option<crate::harness_contract::AgentContract>,
    crate::workspace::WorkspaceGraph,
    crate::workspace::RepositoryContext,
)> {
    let (contract, workspace_graph) = fixture_workspace_graph(fixture)?;
    let task_context = crate::workspace::RepositoryContext::capture(workspace, workspace)?;
    let repository_context = if fixture.resumed_files.is_empty() {
        task_context
    } else {
        write_fixture_files(workspace, &fixture.resumed_files)?;
        crate::workspace::RepositoryContext::resume(
            workspace,
            workspace,
            task_context.task_baseline,
        )?
    };
    Ok((contract, workspace_graph, repository_context))
}

fn fixture_workspace_graph(
    fixture: &ControlFixture,
) -> Result<(
    Option<crate::harness_contract::AgentContract>,
    crate::workspace::WorkspaceGraph,
)> {
    let contract = fixture
        .contract
        .clone()
        .map(crate::harness_contract::HarnessContractDocument::normalize)
        .transpose()?;
    let base_graph = fixture
        .workspace
        .clone()
        .map(crate::workspace::WorkspaceConfigDocument::normalize)
        .transpose()?
        .unwrap_or_else(|| crate::workspace::WorkspaceGraph::legacy(&[]));
    let workspace_graph = contract
        .as_ref()
        .map(|contract| contract.compile_workspace_graph(base_graph.clone()))
        .transpose()?
        .unwrap_or(base_graph);
    Ok((contract, workspace_graph))
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
        workspace_config_sha256: None,
        executor_policy: Vec::new(),
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
        workspace_config_sha256: None,
        executor_policy: Vec::new(),
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
    let (contract, workspace_graph, repository_context) = fixture_runtime(fixture, scratch.path())?;
    let request = AgentRequest {
        task: format!(
            "Harness control evaluation. Complete the repository task implied by this control objective, using only the exposed tools: {}",
            fixture.hypothesis
        ),
        turn_id: format!("harness-eval-turn-{}", fixture.id),
        intent: Some(crate::workflow::TurnIntent::Deliver),
        workflow_policy: None,
        workflow_stage: None,
        conversation_handoff: None,
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
        workspace_graph: Some(workspace_graph),
        repository_context: Some(repository_context),
        prior_check_evidence: crate::checks::CheckEvidenceLedger::default(),
        session_id: format!("harness-eval-{}", fixture.id),
        attachments: Vec::new(),
        contract,
    };
    let mut events = Vec::new();
    let outcome = run_local_model_eval_steps(engine, &request, scratch.path(), &mut |event| {
        events.push(event)
    })?;
    summarize_fixture(fixture, scratch.path(), outcome.into(), &events, true)
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
            let mut record_configuration = configuration.clone();
            if let Some((sha256, executor_policy)) = fixture_workspace_metadata(fixture)? {
                record_configuration.workspace_config_sha256 = Some(sha256);
                record_configuration.executor_policy = executor_policy;
            }
            Ok(HarnessEvalRecord {
                schema_version: HARNESS_EVAL_SCHEMA_VERSION,
                fixture_version: corpus.version,
                configuration: record_configuration,
                protocol_pass: protocol_failures.is_empty(),
                protocol_failures,
                result,
            })
        })
        .collect()
}

fn validate_eval_record_schema(record: &HarnessEvalRecord) -> Result<()> {
    if record.schema_version != HARNESS_EVAL_SCHEMA_VERSION {
        bail!(
            "unsupported harness evaluation schema {}; expected {}",
            record.schema_version,
            HARNESS_EVAL_SCHEMA_VERSION
        );
    }
    Ok(())
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
    if let Some(expected) = &expected.handoff_outcome
        && actual.handoff_outcome.as_ref() != Some(expected)
    {
        failures.push(format!(
            "handoff_outcome expected {:?}, got {:?}",
            expected, actual.handoff_outcome
        ));
    }
    if let Some(expected) = &expected.selected_checks
        && &actual.selected_checks != expected
    {
        failures.push(format!(
            "selected_checks expected {:?}, got {:?}",
            expected, actual.selected_checks
        ));
    }
    for (field, expected, actual) in [
        (
            "executed_checks",
            expected.executed_checks,
            actual.executed_checks,
        ),
        (
            "reused_checks",
            expected.reused_checks,
            actual.reused_checks,
        ),
        ("repair_turns", expected.repair_turns, actual.repair_turns),
    ] {
        if let Some(expected) = expected
            && actual != expected
        {
            failures.push(format!("{field} expected {expected}, got {actual}"));
        }
    }
    if let Some(expected) = &expected.executor_starts
        && &actual.executor_starts != expected
    {
        failures.push(format!(
            "executor_starts expected {:?}, got {:?}",
            expected, actual.executor_starts
        ));
    }
    if let Some(expected) = &expected.commit_disposition
        && actual.commit_disposition.as_ref() != Some(expected)
    {
        failures.push(format!(
            "commit_disposition expected {:?}, got {:?}",
            expected, actual.commit_disposition
        ));
    }
    failures
}

fn fixture_workspace_metadata(fixture: &ControlFixture) -> Result<Option<(String, Vec<String>)>> {
    let Some(document) = fixture.workspace.clone() else {
        return Ok(None);
    };
    let graph = document.normalize()?;
    let normalized = serde_json::to_vec(&graph.to_document())
        .context("failed to serialize normalized fixture workspace")?;
    let executor_policy = graph
        .executors
        .iter()
        .map(|(id, executor)| {
            let kind = match executor.kind {
                crate::workspace::ExecutorKind::Project => "project",
                crate::workspace::ExecutorKind::Local => "local",
                crate::workspace::ExecutorKind::Container => "container",
            };
            format!("{id}:{kind}")
        })
        .collect();
    Ok(Some((
        format!("{:x}", Sha256::digest(normalized)),
        executor_policy,
    )))
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
        validate_eval_record_schema(record)?;
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
        "fixture                         pass handoff          checks reuse execs commit      valid named false loop turns latency_ms tokens energy_kwh termination\n",
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
        let handoff = result
            .handoff_outcome
            .map(|outcome| format!("{outcome:?}").to_ascii_lowercase())
            .unwrap_or_else(|| "-".to_string());
        table.push_str(&format!(
            "{:<31} {:<4} {:<16} {:<6} {:<5} {:<5} {:<11} {:<5} {:<5} {:<5} {:<4} {:<5} {:<10} {:<6} {:<10} {}\n",
            result.id,
            if record.protocol_pass { "yes" } else { "no" },
            handoff,
            result.executed_checks,
            result.reused_checks,
            result.executor_starts.len(),
            result.commit_disposition.as_deref().unwrap_or("-"),
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
    record_commit_oid: bool,
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
    let model_run_check_calls = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::ToolCall { tool, .. } if tool == "run_check"
            )
        })
        .count();
    let executed_checks = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::CheckResult {
                    reused: false,
                    skip_reason: None,
                    ..
                }
            )
        })
        .count();
    let reused_checks = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::CheckResult { reused: true, .. }))
        .count();
    let failed_checks = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::CheckResult {
                    success: false,
                    skip_reason: None,
                    ..
                }
            )
        })
        .count();
    let skipped_checks = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::CheckResult {
                    skip_reason: Some(_),
                    ..
                }
            )
        })
        .count();
    let mut selected_checks = std::collections::BTreeSet::new();
    let mut affected_components = std::collections::BTreeSet::new();
    let mut handoff_outcome = None;
    for event in events {
        if let AgentEvent::HandoffSummary { summary, .. } = event {
            handoff_outcome = Some(summary.outcome);
            selected_checks.extend(summary.checks.iter().map(|check| check.check_id.clone()));
            affected_components.extend(summary.affected_components.iter().cloned());
        }
    }
    let executor_starts = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ExecutorStarted {
                executor_id,
                success: true,
                ..
            } => Some(executor_id.clone()),
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    let configured_executors = fixture
        .workspace
        .as_ref()
        .map(|workspace| {
            workspace
                .executors
                .iter()
                .map(|executor| executor.id.clone())
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();
    let avoided_executor_starts = configured_executors
        .difference(&executor_starts)
        .cloned()
        .collect::<Vec<_>>();
    let team_messages = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::TeamMessage { .. }))
        .count();
    let repair_turns = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::Correction { summary, .. }
                    if summary.contains("handoff teammate returned failed checks")
            )
        })
        .count();
    let commit = events.iter().rev().find_map(|event| match event {
        AgentEvent::CommitResult {
            success,
            created,
            reused,
            oid,
            ..
        } => Some((
            if *created {
                "created"
            } else if *reused {
                "reused"
            } else if *success {
                "not_needed"
            } else {
                "blocked"
            }
            .to_string(),
            oid.clone(),
        )),
        _ => None,
    });
    let output_fingerprints = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::CheckResult {
                output_fingerprint: Some(fingerprint),
                ..
            } => Some(fingerprint.clone()),
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
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
        let (_, graph) = fixture_workspace_graph(fixture)?;
        let ledger = crate::checks::CheckEvidenceLedger::from_events(events);
        Some(required_check_ids.iter().all(|check_id| {
            crate::checks::check_evidence_is_current(workspace, &graph, &ledger, check_id)
                .unwrap_or(false)
        }))
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
        model_run_check_calls,
        reused_checks,
        failed_checks,
        skipped_checks,
        selected_checks: selected_checks.into_iter().collect(),
        affected_components: affected_components.into_iter().collect(),
        executor_starts: executor_starts.into_iter().collect(),
        avoided_executor_starts,
        team_messages,
        repair_turns,
        no_change: handoff_outcome == Some(crate::events::HandoffOutcome::NoChange),
        handoff_outcome,
        commit_disposition: commit.as_ref().map(|(disposition, _)| disposition.clone()),
        commit_oid: record_commit_oid
            .then(|| commit.and_then(|(_, oid)| oid))
            .flatten(),
        output_fingerprints,
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
    write_fixture_files(root, files)?;
    run_git(root, &["add", "-A"])?;
    run_git(
        root,
        &["commit", "--allow-empty", "-m", "test: initialize fixture"],
    )?;
    run_git(root, &["checkout", "-b", "harness-eval"])?;
    Ok(())
}

fn write_fixture_files(root: &Path, files: &BTreeMap<String, String>) -> Result<()> {
    for (relative, content) in files {
        let path = fixture_path(root, relative)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        std::fs::write(&path, content)
            .with_context(|| format!("failed to write fixture file {}", path.display()))?;
    }
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
        assert_eq!(baseline.results, actual);
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
                .find(|result| result.id == "irrelevant_review_evidence")
                .unwrap()
                .executed_checks,
            0,
            "the current gate rejects a missing named check instead of executing it"
        );
        assert_eq!(
            actual
                .iter()
                .find(|result| result.id == "check_then_mutation")
                .unwrap()
                .executed_checks,
            1
        );
        let current = actual
            .iter()
            .find(|result| result.id == "current_named_check")
            .unwrap();
        assert!(current.verified_completed);
        assert_eq!(current.executed_checks, 1);
        assert_eq!(current.named_check_compliance, Some(true));
        let repeated = actual
            .iter()
            .find(|result| result.id == "repeated_blocked_action")
            .unwrap();
        assert!(!repeated.reached_final);
        assert_eq!(repeated.termination_reason, "gate_loop");
        assert_eq!(repeated.llm_invocations, 3);
        assert_eq!(repeated.blocked_tool_loops, 1);
        assert_eq!(repeated.remaining_completions, 1);

        let handoff = actual
            .iter()
            .find(|result| result.id == "handoff_contract_check_commit")
            .unwrap();
        assert_eq!(handoff.model_run_check_calls, 0);
        assert_eq!(handoff.executed_checks, 1);
        assert_eq!(handoff.commit_disposition.as_deref(), Some("created"));
        let required_no_change = actual
            .iter()
            .find(|result| result.id == "handoff_required_mutation_no_change")
            .unwrap();
        assert_eq!(
            required_no_change.termination_reason,
            "contract_unsatisfied"
        );
        assert!(!required_no_change.verified_completed);
        let repaired = actual
            .iter()
            .find(|result| result.id == "handoff_repair_succeeds")
            .unwrap();
        assert_eq!(repaired.executed_checks, 2);
        assert_eq!(repaired.failed_checks, 1);
        assert_eq!(repaired.repair_turns, 1);
        assert_eq!(
            repaired.handoff_outcome,
            Some(crate::events::HandoffOutcome::Ready)
        );
        let resumed = actual
            .iter()
            .find(|result| result.id == "resumed_task_owned_change")
            .unwrap();
        assert_eq!(resumed.affected_components, vec!["api"]);
        assert_eq!(resumed.executed_checks, 1);
        assert_eq!(resumed.commit_disposition.as_deref(), Some("created"));
        let multi = actual
            .iter()
            .find(|result| result.id == "multi_executor_affected_selection")
            .unwrap();
        assert_eq!(multi.executor_starts, vec!["api"]);
        assert_eq!(multi.avoided_executor_starts, vec!["web"]);
        let bundle = actual
            .iter()
            .find(|result| result.id == "generated_bundle_dependency")
            .unwrap();
        assert_eq!(bundle.executed_checks, 2);
        assert_eq!(bundle.output_fingerprints.len(), 1);
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
    fn version_one_fixture_and_result_schemas_are_rejected_explicitly() {
        let error = parse_control_fixture_corpus(r#"{"version":1,"fixtures":[]}"#).unwrap_err();
        assert!(error.to_string().contains("expected 2"));

        let corpus = control_fixture_corpus().unwrap();
        let mut record = build_eval_records(
            &corpus,
            scripted_configuration(),
            run_control_fixture_corpus().unwrap(),
        )
        .unwrap()
        .remove(0);
        record.schema_version = 1;
        let error = validate_eval_record_schema(&record).unwrap_err();
        assert!(error.to_string().contains("expected 2"));
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
        let multi = first
            .iter()
            .find(|record| record.result.id == "multi_executor_affected_selection")
            .unwrap();
        assert_eq!(
            multi.configuration.executor_policy,
            vec!["api:local", "web:local"]
        );
        assert_eq!(
            multi
                .configuration
                .workspace_config_sha256
                .as_deref()
                .map(str::len),
            Some(64)
        );
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
            assert_eq!(record.schema_version, 2);
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
            "handoff",
            "checks",
            "reuse",
            "execs",
            "commit",
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
            workspace_config_sha256: Some("abc".to_string()),
            executor_policy: vec!["app:local".to_string()],
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
            "workspace_config_sha256",
            "executor_policy",
        ] {
            assert!(value.get(field).is_some(), "missing {field}: {value}");
        }
    }
}
