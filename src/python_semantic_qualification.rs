//! Python-owned semantic qualification through the production mutation lifecycle.

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
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    agent_core::BuiltInToolSchema,
    control_layers::{
        ControlLayerLifecycle, PythonSemanticQualificationObservation,
        python_semantic_world_observation, qualify_python_semantic_case,
    },
};

const CORPUS_VERSION: u32 = 1;
const REPORT_VERSION: u32 = 1;
const MAX_CORPUS_BYTES: u64 = 4 * 1024 * 1024;
const MAX_FILES: usize = 64;
const MAX_DEPENDENCY_FILES: usize = 256;
const MAX_CASES: usize = 256;
const MAX_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_MAX_COLD_MILLIS: u64 = 60_000;
const DEFAULT_MAX_CASE_MILLIS: u64 = 20_000;
const PYTHON_ENVIRONMENT_CONFIG: &str = "version = 3.12.8\n";
const PYTHON_GITIGNORE: &str = ".venv/\n";
const PROMOTED_CODES: [&str; 6] = [
    "invalid-argument-type",
    "invalid-assignment",
    "invalid-return-type",
    "unresolved-attribute",
    "unresolved-import",
    "unsupported-operator",
];
const REQUIRED_CATEGORIES: [PythonCorpusCategory; 3] = [
    PythonCorpusCategory::Annotated,
    PythonCorpusCategory::Unannotated,
    PythonCorpusCategory::ThirdParty,
];
const REQUIRED_TOOLS: [&str; 4] = ["write_file", "replace_file", "edit_file", "apply_patch"];

#[derive(Args, Debug, Clone)]
pub struct HarnessPythonSemanticQualifyArgs {
    /// Versioned Python semantic corpus
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
enum PythonCorpusCategory {
    Annotated,
    Unannotated,
    ThirdParty,
    BaselineDebt,
    Transaction,
    DynamicUnknown,
}

impl PythonCorpusCategory {
    fn id(self) -> &'static str {
        match self {
            Self::Annotated => "annotated",
            Self::Unannotated => "unannotated",
            Self::ThirdParty => "third_party",
            Self::BaselineDebt => "baseline_debt",
            Self::Transaction => "transaction",
            Self::DynamicUnknown => "dynamic_unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PythonCorpusOutcome {
    Allow,
    Reject,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PythonSemanticCorpus {
    version: u32,
    python_version: String,
    files: Vec<PythonCorpusFile>,
    dependencies: Vec<PythonCorpusFile>,
    cases: Vec<PythonCorpusCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PythonCorpusFile {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PythonCorpusCase {
    id: String,
    category: PythonCorpusCategory,
    tool: String,
    arguments: Value,
    expected: PythonCorpusExpectation,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PythonCorpusExpectation {
    outcome: PythonCorpusOutcome,
    #[serde(default)]
    diagnostic_codes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct PythonSemanticQualificationReport {
    version: u32,
    corpus_sha256: String,
    provider_version: String,
    world_sha256: String,
    configuration_sha256: String,
    dependency_sha256: String,
    file_count: usize,
    dependency_file_count: usize,
    case_count: usize,
    category_counts: BTreeMap<String, usize>,
    diagnostic_counts: BTreeMap<String, usize>,
    allow_count: usize,
    reject_count: usize,
    generation_final_parity_count: usize,
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

pub(crate) fn run(args: HarnessPythonSemanticQualifyArgs) -> Result<()> {
    let report = qualify(&args.corpus, args.max_cold_millis, args.max_case_millis)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn qualify(
    corpus_path: &Path,
    max_cold_millis: u64,
    max_case_millis: u64,
) -> Result<PythonSemanticQualificationReport> {
    if max_cold_millis == 0 || max_case_millis == 0 {
        bail!("Python semantic qualification budgets must be non-zero");
    }
    let bytes = read_bounded(corpus_path, MAX_CORPUS_BYTES)?;
    let corpus_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let corpus: PythonSemanticCorpus = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "failed to parse Python semantic corpus {}",
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
        .context("Python semantic corpus did not prepare a native world")?;
    let cold_millis = elapsed_millis(cold_started);
    drop(cold_layers);
    if cold_millis > max_cold_millis {
        bail!(
            "Python semantic corpus exceeded its cold budget: {cold_millis} > {max_cold_millis} ms"
        );
    }
    let world = python_semantic_world_observation(&lifecycle)?;

    let mut category_counts = BTreeMap::new();
    let mut diagnostic_counts = BTreeMap::new();
    let mut allow_count = 0usize;
    let mut reject_count = 0usize;
    let mut generation_final_parity_count = 0usize;
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
        let observation = qualify_python_semantic_case(
            &mut lifecycle,
            &fixture.root,
            &tool,
            &snapshot,
            &case.tool,
            &case.arguments,
        )
        .with_context(|| format!("Python semantic case {} failed", case.id))?;
        case_latencies.extend([
            observation.warm_millis,
            observation.generation_millis,
            observation.final_replay_millis,
            observation.diagnostic_millis,
        ]);
        if let Err(error) = validate_observation(case, &observation) {
            failures.push(format!("{}: {error:#}", case.id));
        }
        if observation.generation == observation.final_replay {
            generation_final_parity_count = generation_final_parity_count.saturating_add(1);
        }
        *category_counts
            .entry(case.category.id().to_string())
            .or_insert(0usize) += 1;
        for code in &case.expected.diagnostic_codes {
            *diagnostic_counts.entry(code.clone()).or_insert(0usize) += 1;
        }
        match case.expected.outcome {
            PythonCorpusOutcome::Allow => allow_count = allow_count.saturating_add(1),
            PythonCorpusOutcome::Reject => reject_count = reject_count.saturating_add(1),
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
            "Python semantic qualification failed {} observation(s): {}",
            failures.len(),
            failures.join("; ")
        );
    }

    Ok(PythonSemanticQualificationReport {
        version: REPORT_VERSION,
        corpus_sha256,
        provider_version: world.provider_version,
        world_sha256: world.world_sha256,
        configuration_sha256: world.configuration_sha256,
        dependency_sha256: world.dependency_sha256,
        file_count: corpus.files.len(),
        dependency_file_count: corpus.dependencies.len(),
        case_count: corpus.cases.len(),
        category_counts,
        diagnostic_counts,
        allow_count,
        reject_count,
        generation_final_parity_count,
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
    case: &PythonCorpusCase,
    observation: &PythonSemanticQualificationObservation,
) -> Result<()> {
    let expected_decision = match case.expected.outcome {
        PythonCorpusOutcome::Allow => CompletionDecision::Accept,
        PythonCorpusOutcome::Reject => CompletionDecision::Reject(RejectionCode::InvalidSemantics),
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
    let expected_codes = case
        .expected
        .diagnostic_codes
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if observation.diagnostic_codes != expected_codes {
        bail!(
            "diagnostic codes {:?} did not match {:?}",
            observation.diagnostic_codes,
            expected_codes
        );
    }
    Ok(())
}

fn validate_corpus(corpus: &PythonSemanticCorpus) -> Result<()> {
    if corpus.version != CORPUS_VERSION {
        bail!("Python semantic corpus version must be {CORPUS_VERSION}");
    }
    if corpus.python_version != "3.12" {
        bail!("Python semantic corpus must target the promoted Python 3.12 profile");
    }
    if corpus.files.is_empty() || corpus.files.len() > MAX_FILES {
        bail!("Python semantic corpus must contain 1..={MAX_FILES} first-party files");
    }
    if corpus.dependencies.is_empty() || corpus.dependencies.len() > MAX_DEPENDENCY_FILES {
        bail!("Python semantic corpus must contain 1..={MAX_DEPENDENCY_FILES} dependency files");
    }
    if corpus.cases.is_empty() || corpus.cases.len() > MAX_CASES {
        bail!("Python semantic corpus must contain 1..={MAX_CASES} cases");
    }

    let mut total_bytes = 0usize;
    let mut first_party_paths = BTreeSet::new();
    for file in &corpus.files {
        let path = LogicalPath::parse(file.path.clone())?;
        if !is_python_path(path.as_str()) || !first_party_paths.insert(file.path.as_str()) {
            bail!("Python semantic corpus has a repeated or non-Python first-party path");
        }
        total_bytes = checked_workspace_bytes(total_bytes, file.content.len())?;
    }
    let mut dependency_paths = BTreeSet::new();
    for file in &corpus.dependencies {
        let path = LogicalPath::parse(file.path.clone())?;
        if forbidden_dependency_path(path.as_str()) || !dependency_paths.insert(file.path.as_str())
        {
            bail!("Python semantic corpus has a repeated or unsafe dependency path");
        }
        total_bytes = checked_workspace_bytes(total_bytes, file.content.len())?;
    }
    if total_bytes > MAX_WORKSPACE_BYTES {
        bail!("Python semantic corpus exceeds the {MAX_WORKSPACE_BYTES}-byte workspace bound");
    }

    let promoted = PROMOTED_CODES.into_iter().collect::<BTreeSet<_>>();
    let mut observed_codes = BTreeSet::new();
    let mut ids = BTreeSet::new();
    let mut category_outcomes = BTreeSet::new();
    let mut tools = BTreeSet::new();
    for case in &corpus.cases {
        if case.id.is_empty() || case.id.len() > 128 || !ids.insert(case.id.as_str()) {
            bail!("Python semantic corpus case ids must be bounded, non-empty, and unique");
        }
        if !REQUIRED_TOOLS.contains(&case.tool.as_str()) {
            bail!("Python semantic case {} uses an unsupported tool", case.id);
        }
        tools.insert(case.tool.as_str());
        category_outcomes.insert((
            case.category,
            case.expected.outcome == PythonCorpusOutcome::Reject,
        ));
        let codes = case
            .expected
            .diagnostic_codes
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if codes.len() != case.expected.diagnostic_codes.len()
            || !codes.is_subset(&promoted)
            || (case.expected.outcome == PythonCorpusOutcome::Allow && !codes.is_empty())
            || (case.expected.outcome == PythonCorpusOutcome::Reject && codes.is_empty())
        {
            bail!(
                "Python semantic case {} has an invalid expected diagnostic set",
                case.id
            );
        }
        observed_codes.extend(codes);
        validate_case_arguments(case, &first_party_paths)?;
    }
    if observed_codes != promoted {
        bail!("Python semantic corpus must exercise every promoted diagnostic code");
    }
    if tools != REQUIRED_TOOLS.into_iter().collect::<BTreeSet<_>>() {
        bail!("Python semantic corpus must exercise every supported mutation tool");
    }
    for category in REQUIRED_CATEGORIES {
        if !category_outcomes.contains(&(category, false))
            || !category_outcomes.contains(&(category, true))
        {
            bail!(
                "Python semantic corpus category {} requires allow and reject cases",
                category.id()
            );
        }
    }
    Ok(())
}

fn validate_case_arguments(
    case: &PythonCorpusCase,
    first_party_paths: &BTreeSet<&str>,
) -> Result<()> {
    if case.tool == "apply_patch" {
        if case
            .arguments
            .get("patch")
            .and_then(Value::as_str)
            .is_none()
        {
            bail!("Python semantic patch case {} has no patch", case.id);
        }
        return Ok(());
    }
    let path = case
        .arguments
        .get("path")
        .and_then(Value::as_str)
        .with_context(|| format!("Python semantic case {} has no path", case.id))?;
    let logical = LogicalPath::parse(path.to_string())?;
    if !is_python_path(logical.as_str()) {
        bail!("Python semantic case {} targets a non-Python file", case.id);
    }
    match case.tool.as_str() {
        "write_file" if first_party_paths.contains(path) => {
            bail!(
                "Python semantic write case {} targets an existing file",
                case.id
            )
        }
        "replace_file" | "edit_file" if !first_party_paths.contains(path) => {
            bail!(
                "Python semantic modify case {} targets a missing file",
                case.id
            )
        }
        _ => {}
    }
    Ok(())
}

fn materialize_corpus(corpus: &PythonSemanticCorpus) -> Result<PythonCorpusFixture> {
    let owner = tempfile::Builder::new()
        .prefix("pb-python-semantic-")
        .tempdir()
        .context("failed to create Python semantic corpus workspace")?;
    let root = owner.path().to_path_buf();
    initialize_git(&root)?;
    fs::write(root.join(".gitignore"), PYTHON_GITIGNORE)?;
    let site_packages = root.join(".venv/lib/python3.12/site-packages");
    fs::create_dir_all(&site_packages)?;
    fs::write(root.join(".venv/pyvenv.cfg"), PYTHON_ENVIRONMENT_CONFIG)?;
    for file in &corpus.files {
        write_fixture_file(&root, file)?;
    }
    for file in &corpus.dependencies {
        write_fixture_file(&site_packages, file)?;
    }
    Ok(PythonCorpusFixture {
        _owner: owner,
        root,
    })
}

struct PythonCorpusFixture {
    _owner: tempfile::TempDir,
    root: PathBuf,
}

fn write_fixture_file(root: &Path, file: &PythonCorpusFile) -> Result<()> {
    let path = root.join(&file.path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, file.content.as_bytes())
        .with_context(|| format!("failed to write Python semantic fixture {}", file.path))
}

fn initialize_git(root: &Path) -> Result<()> {
    let status = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(root)
        .status()
        .context("failed to initialize Python semantic Git fixture")?;
    if !status.success() {
        bail!("failed to initialize Python semantic Git fixture");
    }
    Ok(())
}

fn workspace_snapshot(files: &[PythonCorpusFile]) -> Result<WorkspaceSnapshot> {
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
        _ => bail!("unsupported Python semantic qualification tool {name}"),
    };
    Ok(BuiltInToolSchema {
        name: name.to_string(),
        description: "Python semantic qualification mutation".to_string(),
        input_schema,
    })
}

fn forbidden_dependency_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.ends_with(".pth")
        || lower.ends_with(".so")
        || lower.ends_with(".dylib")
        || lower.ends_with(".dll")
        || lower.ends_with(".pyd")
        || lower.contains("__pycache__")
}

fn is_python_path(path: &str) -> bool {
    path.ends_with(".py") || path.ends_with(".pyi")
}

fn checked_workspace_bytes(total: usize, next: usize) -> Result<usize> {
    total
        .checked_add(next)
        .context("Python semantic corpus workspace size overflowed")
}

fn read_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open Python semantic corpus {}", path.display()))?;
    let metadata = file.metadata().with_context(|| {
        format!(
            "failed to inspect Python semantic corpus {}",
            path.display()
        )
    })?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        bail!("Python semantic corpus must be a regular file of at most {max_bytes} bytes");
    }
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or_default());
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes || bytes.len() as u64 != metadata.len() {
        bail!("Python semantic corpus changed or exceeded its byte bound while reading");
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

    const CHECKED_CORPUS: &str = include_str!("../fixtures/control-collar/semantic-python-v1.json");

    #[test]
    fn checked_python_semantic_corpus_is_complete() {
        let corpus: PythonSemanticCorpus = serde_json::from_str(CHECKED_CORPUS).unwrap();
        validate_corpus(&corpus).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(CHECKED_CORPUS.as_bytes())),
            "e073204a58aab27a8f52c17912734f138e7a6a239c2999d95841ece93f1f0b40"
        );
    }

    #[test]
    fn corpus_contract_rejects_unsafe_dependencies_and_missing_profile_arms() {
        let mut unsafe_dependency: Value = serde_json::from_str(CHECKED_CORPUS).unwrap();
        unsafe_dependency["dependencies"][0]["path"] = Value::String("../outside.py".to_string());
        let corpus: PythonSemanticCorpus = serde_json::from_value(unsafe_dependency).unwrap();
        assert!(validate_corpus(&corpus).is_err());

        let mut missing_profile: Value = serde_json::from_str(CHECKED_CORPUS).unwrap();
        for case in missing_profile["cases"].as_array_mut().unwrap() {
            if case["category"] == "third_party" {
                case["category"] = Value::String("annotated".to_string());
            }
        }
        let corpus: PythonSemanticCorpus = serde_json::from_value(missing_profile).unwrap();
        assert!(validate_corpus(&corpus).is_err());
    }

    #[test]
    fn corpus_contract_rejects_unpromoted_or_duplicate_diagnostic_expectations() {
        let mut unpromoted: Value = serde_json::from_str(CHECKED_CORPUS).unwrap();
        unpromoted["cases"][1]["expected"]["diagnostic_codes"][0] =
            Value::String("not-promoted".to_string());
        let corpus: PythonSemanticCorpus = serde_json::from_value(unpromoted).unwrap();
        assert!(validate_corpus(&corpus).is_err());

        let mut duplicate: Value = serde_json::from_str(CHECKED_CORPUS).unwrap();
        duplicate["cases"][1]["expected"]["diagnostic_codes"] =
            serde_json::json!(["invalid-argument-type", "invalid-argument-type"]);
        let corpus: PythonSemanticCorpus = serde_json::from_value(duplicate).unwrap();
        assert!(validate_corpus(&corpus).is_err());
    }

    #[test]
    fn checked_python_semantic_corpus_passes_production_gates() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/control-collar/semantic-python-v1.json");
        let report = qualify(&path, DEFAULT_MAX_COLD_MILLIS, DEFAULT_MAX_CASE_MILLIS).unwrap();
        assert_eq!(report.case_count, 24);
        assert_eq!(report.generation_final_parity_count, report.case_count);
        assert_eq!(report.diagnostic_counts.len(), PROMOTED_CODES.len());
        assert!(report.passed);
    }
}
