//! Rust-owned semantic and prefix-monotonicity qualification through the production lifecycle.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use anyhow::{Context, Result, bail};
use clap::Args;
use pb_control_collar::{
    CompletionDecision, RejectionCode,
    mutation::{LogicalPath, SnapshotEntry, WorkspaceSnapshot},
};
use pb_control_rust::{RustDeepDiagnostic, RustDeepProfile, RustDeepUnknownReason};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    agent_core::BuiltInToolSchema,
    control_layers::{
        ControlLayerLifecycle, RustSemanticQualificationObservation, qualify_rust_semantic_case,
        rust_semantic_world_observation,
    },
};

const CORPUS_VERSION: u32 = 2;
const REPORT_VERSION: u32 = 1;
const MAX_CORPUS_BYTES: u64 = 4 * 1024 * 1024;
const MAX_FILES: usize = 128;
const MAX_CASES: usize = 256;
const MAX_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_MAX_COLD_MILLIS: u64 = 120_000;
const DEFAULT_MAX_CASE_MILLIS: u64 = 20_000;
const PROMOTED_DIAGNOSTICS: [RustDeepDiagnostic; 10] = [
    RustDeepDiagnostic::UnresolvedName,
    RustDeepDiagnostic::UnresolvedImport,
    RustDeepDiagnostic::MissingField,
    RustDeepDiagnostic::MissingMethod,
    RustDeepDiagnostic::Privacy,
    RustDeepDiagnostic::TypeMismatch,
    RustDeepDiagnostic::InvalidCall,
    RustDeepDiagnostic::Mutability,
    RustDeepDiagnostic::Ownership,
    RustDeepDiagnostic::TraitContract,
];
const REQUIRED_UNKNOWN_REASONS: [RustDeepUnknownReason; 2] = [
    RustDeepUnknownReason::ImportResolutionUnsupported,
    RustDeepUnknownReason::SourceTopologyChanged,
];
const REQUIRED_CATEGORIES: [RustCorpusCategory; 5] = [
    RustCorpusCategory::Positive,
    RustCorpusCategory::Diagnostic,
    RustCorpusCategory::BaselineDebt,
    RustCorpusCategory::Transaction,
    RustCorpusCategory::Unknown,
];
const REQUIRED_TOOLS: [&str; 4] = ["write_file", "replace_file", "edit_file", "apply_patch"];

#[derive(Args, Debug, Clone)]
pub struct HarnessRustSemanticQualifyArgs {
    /// Versioned Rust semantic and prefix-monotonicity corpus
    #[arg(long)]
    pub(crate) corpus: PathBuf,

    /// Maximum accepted initial native-world preparation time
    #[arg(long, default_value_t = DEFAULT_MAX_COLD_MILLIS)]
    pub(crate) max_cold_millis: u64,

    /// Maximum accepted time for each warm preparation, generation gate, final replay, or
    /// diagnostic replay
    #[arg(long, default_value_t = DEFAULT_MAX_CASE_MILLIS)]
    pub(crate) max_case_millis: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum RustCorpusCategory {
    Positive,
    Diagnostic,
    BaselineDebt,
    Transaction,
    Unknown,
}

impl RustCorpusCategory {
    const fn id(self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Diagnostic => "diagnostic",
            Self::BaselineDebt => "baseline_debt",
            Self::Transaction => "transaction",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RustCorpusOutcome {
    Allow,
    Reject,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RustSemanticCorpus {
    version: u32,
    files: Vec<RustCorpusFile>,
    cases: Vec<RustCorpusCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RustCorpusFile {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RustCorpusCase {
    id: String,
    category: RustCorpusCategory,
    tool: String,
    arguments: Value,
    expected: RustCorpusExpectation,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RustCorpusExpectation {
    outcome: RustCorpusOutcome,
    #[serde(default)]
    diagnostics: Vec<RustDeepDiagnostic>,
    #[serde(default)]
    unknown_reason: Option<RustDeepUnknownReason>,
}

#[derive(Clone, Debug, Serialize)]
struct RustSemanticQualificationReport {
    version: u32,
    corpus_sha256: String,
    provider_version: String,
    world_sha256: String,
    configuration_sha256: String,
    dependency_sha256: String,
    file_count: usize,
    target_count: usize,
    case_count: usize,
    category_counts: BTreeMap<String, usize>,
    diagnostic_counts: BTreeMap<String, usize>,
    unknown_counts: BTreeMap<String, usize>,
    allow_count: usize,
    reject_count: usize,
    generation_final_parity_count: usize,
    prefix_probe_count: u64,
    rollback_replay_count: u64,
    cold_millis: u64,
    load_millis: u64,
    prime_millis: u64,
    primed_queries: u64,
    p50_case_millis: u64,
    p95_case_millis: u64,
    p99_case_millis: u64,
    max_case_millis: u64,
    cold_budget_millis: u64,
    case_budget_millis: u64,
    passed: bool,
}

pub(crate) fn run(args: HarnessRustSemanticQualifyArgs) -> Result<()> {
    let report = qualify(&args.corpus, args.max_cold_millis, args.max_case_millis)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn qualify(
    corpus_path: &Path,
    max_cold_millis: u64,
    max_case_millis: u64,
) -> Result<RustSemanticQualificationReport> {
    if max_cold_millis == 0 || max_case_millis == 0 {
        bail!("Rust semantic qualification budgets must be non-zero");
    }
    let bytes = read_bounded(corpus_path, MAX_CORPUS_BYTES)?;
    let corpus_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let corpus: RustSemanticCorpus = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "failed to parse Rust semantic corpus {}",
            corpus_path.display()
        )
    })?;
    validate_corpus(&corpus)?;

    let fixture = materialize_corpus(&corpus)?;
    let base_snapshot = workspace_snapshot(&corpus.files)?;
    let mut lifecycle = ControlLayerLifecycle::default();
    let cold_tool = tool_schema("apply_patch", None)?;
    let cold_started = Instant::now();
    let cold_layers = lifecycle
        .prepare_for_inference_cancellable(
            &fixture.root,
            std::slice::from_ref(&cold_tool),
            Some(&base_snapshot),
            &|| Ok(()),
        )?
        .context("Rust semantic corpus did not prepare a native world")?;
    let cold_millis = elapsed_millis(cold_started);
    drop(cold_layers);
    if cold_millis > max_cold_millis {
        bail!(
            "Rust semantic corpus exceeded its cold budget: {cold_millis} > {max_cold_millis} ms"
        );
    }
    let world = rust_semantic_world_observation(&lifecycle)?;
    if world.deep_profile != RustDeepProfile::Exact {
        bail!(
            "Rust semantic corpus requires the exact no-execution profile, got {:?}",
            world.deep_profile
        );
    }

    let mut category_counts = BTreeMap::new();
    let mut diagnostic_counts = BTreeMap::new();
    let mut unknown_counts = BTreeMap::new();
    let mut allow_count = 0usize;
    let mut reject_count = 0usize;
    let mut generation_final_parity_count = 0usize;
    let mut prefix_probe_count = 0u64;
    let mut rollback_replay_count = 0u64;
    let mut case_latencies = Vec::with_capacity(corpus.cases.len() * 4);
    let mut failures = Vec::new();

    for case in &corpus.cases {
        let bound_path = case
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .map(|path| LogicalPath::parse(path.to_string()))
            .transpose()?;
        let snapshot = if case.tool == "apply_patch" {
            base_snapshot.clone()
        } else {
            base_snapshot
                .clone()
                .with_bound_mutation_path(bound_path.context("mutation case has no path")?)
        };
        let tool = tool_schema(&case.tool, snapshot.bound_mutation_path())?;
        let observation = qualify_rust_semantic_case(
            &mut lifecycle,
            &fixture.root,
            &tool,
            &snapshot,
            &case.tool,
            &case.arguments,
        )
        .with_context(|| format!("Rust semantic case {} failed", case.id))?;
        case_latencies.extend([
            observation.warm_millis,
            observation.generation_millis,
            observation.final_replay_millis,
            observation.diagnostic_millis,
        ]);
        prefix_probe_count = prefix_probe_count.saturating_add(observation.prefix_probes);
        rollback_replay_count = rollback_replay_count.saturating_add(observation.rollback_replays);
        if let Err(error) = validate_observation(case, &observation) {
            failures.push(format!("{}: {error:#}", case.id));
        }
        if observation.generation == observation.final_replay {
            generation_final_parity_count = generation_final_parity_count.saturating_add(1);
        }
        *category_counts
            .entry(case.category.id().to_string())
            .or_insert(0usize) += 1;
        for diagnostic in &case.expected.diagnostics {
            *diagnostic_counts
                .entry(diagnostic.id().to_string())
                .or_insert(0usize) += 1;
        }
        if let Some(reason) = case.expected.unknown_reason {
            *unknown_counts
                .entry(reason.id().to_string())
                .or_insert(0usize) += 1;
        }
        match case.expected.outcome {
            RustCorpusOutcome::Allow => allow_count = allow_count.saturating_add(1),
            RustCorpusOutcome::Reject => reject_count = reject_count.saturating_add(1),
        }
    }

    case_latencies.sort_unstable();
    let p50_case_millis = percentile(&case_latencies, 50);
    let p95_case_millis = percentile(&case_latencies, 95);
    let p99_case_millis = percentile(&case_latencies, 99);
    let observed_max_case_millis = case_latencies.last().copied().unwrap_or_default();
    if observed_max_case_millis > max_case_millis {
        failures.push(format!(
            "maximum case stage latency {observed_max_case_millis} ms exceeded {max_case_millis} ms"
        ));
    }
    if !failures.is_empty() {
        bail!(
            "Rust semantic qualification failed {} observation(s): {}",
            failures.len(),
            failures.join("; ")
        );
    }

    Ok(RustSemanticQualificationReport {
        version: REPORT_VERSION,
        corpus_sha256,
        provider_version: world.provider_version,
        world_sha256: world.world_sha256,
        configuration_sha256: world.configuration_sha256,
        dependency_sha256: world.dependency_sha256,
        file_count: corpus.files.len(),
        target_count: world.target_count,
        case_count: corpus.cases.len(),
        category_counts,
        diagnostic_counts,
        unknown_counts,
        allow_count,
        reject_count,
        generation_final_parity_count,
        prefix_probe_count,
        rollback_replay_count,
        cold_millis,
        load_millis: world.load_millis,
        prime_millis: world.prime_millis,
        primed_queries: world.primed_queries,
        p50_case_millis,
        p95_case_millis,
        p99_case_millis,
        max_case_millis: observed_max_case_millis,
        cold_budget_millis: max_cold_millis,
        case_budget_millis: max_case_millis,
        passed: true,
    })
}

fn validate_observation(
    case: &RustCorpusCase,
    observation: &RustSemanticQualificationObservation,
) -> Result<()> {
    let expected_decision = match case.expected.outcome {
        RustCorpusOutcome::Allow => CompletionDecision::Accept,
        RustCorpusOutcome::Reject => CompletionDecision::Reject(RejectionCode::InvalidSemantics),
    };
    if observation.generation != expected_decision {
        bail!(
            "generation decision {:?} did not match {:?}",
            observation.generation,
            expected_decision
        );
    }
    if observation.final_replay != expected_decision {
        bail!(
            "final decision {:?} did not match {:?}",
            observation.final_replay,
            expected_decision
        );
    }
    let expected_diagnostics = case
        .expected
        .diagnostics
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    match (&observation.diagnostic_result, case.expected.unknown_reason) {
        (Ok(actual), None) if actual == &expected_diagnostics => Ok(()),
        (Err(actual), Some(expected)) if *actual == expected => Ok(()),
        (actual, expected_unknown) => bail!(
            "diagnostic result {actual:?} did not match diagnostics {expected_diagnostics:?} and unknown {expected_unknown:?}"
        ),
    }
}

fn validate_corpus(corpus: &RustSemanticCorpus) -> Result<()> {
    if corpus.version != CORPUS_VERSION {
        bail!("Rust semantic corpus version must be {CORPUS_VERSION}");
    }
    if corpus.files.is_empty() || corpus.files.len() > MAX_FILES {
        bail!("Rust semantic corpus must contain 1..={MAX_FILES} files");
    }
    if corpus.cases.is_empty() || corpus.cases.len() > MAX_CASES {
        bail!("Rust semantic corpus must contain 1..={MAX_CASES} cases");
    }

    let mut total_bytes = 0usize;
    let mut paths = BTreeSet::new();
    let mut has_root_manifest = false;
    let mut has_rust_source = false;
    for file in &corpus.files {
        let path = LogicalPath::parse(file.path.clone())?;
        if !paths.insert(file.path.as_str()) {
            bail!("Rust semantic corpus has a repeated file path");
        }
        has_root_manifest |= path.as_str() == "Cargo.toml";
        has_rust_source |= is_rust_path(path.as_str());
        total_bytes = checked_workspace_bytes(total_bytes, file.content.len())?;
    }
    if !has_root_manifest || !has_rust_source {
        bail!("Rust semantic corpus requires a root Cargo.toml and at least one Rust source");
    }
    if total_bytes > MAX_WORKSPACE_BYTES {
        bail!("Rust semantic corpus exceeds the {MAX_WORKSPACE_BYTES}-byte workspace bound");
    }

    let promoted = PROMOTED_DIAGNOSTICS.into_iter().collect::<BTreeSet<_>>();
    let required_unknown = REQUIRED_UNKNOWN_REASONS
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut observed_diagnostics = BTreeSet::new();
    let mut observed_unknown = BTreeSet::new();
    let mut ids = BTreeSet::new();
    let mut categories = BTreeSet::new();
    let mut tools = BTreeSet::new();
    for case in &corpus.cases {
        if case.id.is_empty() || case.id.len() > 128 || !ids.insert(case.id.as_str()) {
            bail!("Rust semantic corpus case ids must be bounded, non-empty, and unique");
        }
        if !REQUIRED_TOOLS.contains(&case.tool.as_str()) {
            bail!("Rust semantic case {} uses an unsupported tool", case.id);
        }
        tools.insert(case.tool.as_str());
        categories.insert(case.category);
        let diagnostics = case
            .expected
            .diagnostics
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if diagnostics.len() != case.expected.diagnostics.len()
            || !diagnostics.is_subset(&promoted)
            || (case.expected.outcome == RustCorpusOutcome::Allow && !diagnostics.is_empty())
            || (case.expected.outcome == RustCorpusOutcome::Reject && diagnostics.is_empty())
            || (case.expected.unknown_reason.is_some() && !diagnostics.is_empty())
            || (case.expected.unknown_reason.is_some()
                && case.expected.outcome != RustCorpusOutcome::Allow)
        {
            bail!(
                "Rust semantic case {} has an invalid diagnostic/unknown expectation",
                case.id
            );
        }
        observed_diagnostics.extend(diagnostics);
        observed_unknown.extend(case.expected.unknown_reason);
        validate_case_arguments(case, &paths)?;
    }
    if observed_diagnostics != promoted {
        bail!("Rust semantic corpus must exercise every promoted diagnostic class");
    }
    if !required_unknown.is_subset(&observed_unknown) {
        bail!("Rust semantic corpus must exercise every required conservative unknown reason");
    }
    if tools != REQUIRED_TOOLS.into_iter().collect::<BTreeSet<_>>() {
        bail!("Rust semantic corpus must exercise every supported mutation tool");
    }
    if categories != REQUIRED_CATEGORIES.into_iter().collect::<BTreeSet<_>>() {
        bail!("Rust semantic corpus must exercise every required category");
    }
    Ok(())
}

fn validate_case_arguments(case: &RustCorpusCase, paths: &BTreeSet<&str>) -> Result<()> {
    if case.tool == "apply_patch" {
        if case
            .arguments
            .get("patch")
            .and_then(Value::as_str)
            .is_none()
        {
            bail!("Rust semantic patch case {} has no patch", case.id);
        }
        return Ok(());
    }
    let path = case
        .arguments
        .get("path")
        .and_then(Value::as_str)
        .with_context(|| format!("Rust semantic case {} has no path", case.id))?;
    let logical = LogicalPath::parse(path.to_string())?;
    if !is_rust_path(logical.as_str()) {
        bail!("Rust semantic case {} targets a non-Rust file", case.id);
    }
    match case.tool.as_str() {
        "write_file" if paths.contains(path) => {
            bail!(
                "Rust semantic write case {} targets an existing file",
                case.id
            )
        }
        "replace_file" | "edit_file" if !paths.contains(path) => {
            bail!(
                "Rust semantic modify case {} targets a missing file",
                case.id
            )
        }
        _ => {}
    }
    Ok(())
}

fn materialize_corpus(corpus: &RustSemanticCorpus) -> Result<RustCorpusFixture> {
    let owner = tempfile::Builder::new()
        .prefix("pb-rust-semantic-")
        .tempdir()
        .context("failed to create Rust semantic corpus workspace")?;
    let root = owner.path().to_path_buf();
    initialize_git(&root)?;
    for file in &corpus.files {
        write_fixture_file(&root, file)?;
    }
    Ok(RustCorpusFixture {
        _owner: owner,
        root,
    })
}

struct RustCorpusFixture {
    _owner: tempfile::TempDir,
    root: PathBuf,
}

fn write_fixture_file(root: &Path, file: &RustCorpusFile) -> Result<()> {
    let path = root.join(&file.path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, file.content.as_bytes())
        .with_context(|| format!("failed to write Rust semantic fixture {}", file.path))
}

fn initialize_git(root: &Path) -> Result<()> {
    let status = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(root)
        .status()
        .context("failed to initialize Rust semantic Git fixture")?;
    if !status.success() {
        bail!("failed to initialize Rust semantic Git fixture");
    }
    Ok(())
}

fn workspace_snapshot(files: &[RustCorpusFile]) -> Result<WorkspaceSnapshot> {
    WorkspaceSnapshot::new(
        files
            .iter()
            .map(|file| {
                Ok(SnapshotEntry::new(
                    LogicalPath::parse(file.path.clone())?,
                    file.content.as_bytes().to_vec(),
                ))
            })
            .collect::<Result<Vec<_>>>()?,
    )
    .map_err(Into::into)
}

fn tool_schema(name: &str, bound_path: Option<&LogicalPath>) -> Result<BuiltInToolSchema> {
    let path_schema = bound_path.map_or_else(
        || serde_json::json!({ "type": "string" }),
        |path| serde_json::json!({ "const": path.as_str() }),
    );
    let input_schema = match name {
        "write_file" | "replace_file" => serde_json::json!({
            "type": "object",
            "properties": {
                "path": path_schema,
                "content": { "type": "string" }
            },
            "required": ["path", "content"],
            "additionalProperties": false
        }),
        "edit_file" => serde_json::json!({
            "type": "object",
            "properties": {
                "path": path_schema,
                "old_text": { "type": "string" },
                "new_text": { "type": "string" }
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": false
        }),
        "apply_patch" => serde_json::json!({
            "type": "object",
            "properties": { "patch": { "type": "string" } },
            "required": ["patch"],
            "additionalProperties": false
        }),
        _ => bail!("unsupported Rust semantic qualification tool {name}"),
    };
    Ok(BuiltInToolSchema {
        name: name.to_string(),
        description: "Rust semantic qualification mutation".to_string(),
        input_schema,
    })
}

fn is_rust_path(path: &str) -> bool {
    path.ends_with(".rs")
}

fn checked_workspace_bytes(total: usize, next: usize) -> Result<usize> {
    total
        .checked_add(next)
        .context("Rust semantic corpus workspace size overflowed")
}

fn read_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open Rust semantic corpus {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect Rust semantic corpus {}", path.display()))?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        bail!("Rust semantic corpus must be a regular file of at most {max_bytes} bytes");
    }
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or_default());
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes || bytes.len() as u64 != metadata.len() {
        bail!("Rust semantic corpus changed or exceeded its byte bound while reading");
    }
    Ok(bytes)
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn percentile(values: &[u64], percentile: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let index = values
        .len()
        .saturating_mul(percentile)
        .saturating_add(99)
        .saturating_div(100)
        .saturating_sub(1)
        .min(values.len().saturating_sub(1));
    values[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHECKED_CORPUS: &str = include_str!("../fixtures/control-collar/semantic-rust-v2.json");

    #[test]
    fn checked_rust_semantic_corpus_is_complete() {
        let corpus: RustSemanticCorpus = serde_json::from_str(CHECKED_CORPUS).unwrap();
        validate_corpus(&corpus).unwrap();
    }

    #[test]
    fn corpus_contract_rejects_missing_diagnostics_and_unknown_coverage() {
        let mut missing_diagnostic: Value = serde_json::from_str(CHECKED_CORPUS).unwrap();
        for case in missing_diagnostic["cases"].as_array_mut().unwrap() {
            if case["expected"]["diagnostics"] == serde_json::json!(["ownership"]) {
                case["expected"]["diagnostics"] = serde_json::json!(["type_mismatch"]);
            }
        }
        let corpus: RustSemanticCorpus = serde_json::from_value(missing_diagnostic).unwrap();
        assert!(validate_corpus(&corpus).is_err());

        let mut missing_unknown: Value = serde_json::from_str(CHECKED_CORPUS).unwrap();
        for case in missing_unknown["cases"].as_array_mut().unwrap() {
            if case["expected"]["unknown_reason"] == "import_resolution_unsupported" {
                case["expected"]["unknown_reason"] =
                    Value::String("source_topology_changed".to_string());
            }
        }
        let corpus: RustSemanticCorpus = serde_json::from_value(missing_unknown).unwrap();
        assert!(validate_corpus(&corpus).is_err());
    }

    #[test]
    fn checked_rust_semantic_corpus_passes_production_gates() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/control-collar/semantic-rust-v2.json");
        let report = qualify(&path, DEFAULT_MAX_COLD_MILLIS, DEFAULT_MAX_CASE_MILLIS).unwrap();
        assert_eq!(report.generation_final_parity_count, report.case_count);
        assert_eq!(report.diagnostic_counts.len(), PROMOTED_DIAGNOSTICS.len());
        assert!(report.prefix_probe_count >= report.case_count as u64);
        assert_eq!(report.rollback_replay_count, report.case_count as u64 * 64);
        assert!(report.passed);
    }
}
