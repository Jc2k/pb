//! Model-free qualification for digest-pinned semantic provider profiles.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use pb_control_collar::analysis::{
    BaselineCompleteness, ClosureVerdict, DefiniteErrorClass, SemanticEvidenceStage,
    SemanticGateReceipt, UnknownReason,
};
use pb_control_collar::mutation::{LogicalPath, SnapshotEntry, WorkspaceSnapshot};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::lsp::{LspSemanticEnforcement, LspServerConfig};

const CORPUS_VERSION: u32 = 1;
const REPORT_VERSION: u32 = 1;
const MAX_CORPUS_BYTES: u64 = 4 * 1024 * 1024;
const MAX_FILES: usize = 512;
const MAX_CASES: usize = 256;
const MAX_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticQualificationCorpus {
    version: u32,
    files: Vec<QualificationFile>,
    cases: Vec<QualificationCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualificationFile {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualificationCase {
    id: String,
    tool: String,
    arguments: Value,
    expected: QualificationExpectation,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualificationExpectation {
    closure: ClosureVerdict,
    #[serde(default)]
    definite_errors: Vec<DefiniteErrorClass>,
    #[serde(default)]
    unknown_reasons: Vec<UnknownReason>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SemanticQualificationReport {
    version: u32,
    corpus_sha256: String,
    provider: String,
    provider_version: String,
    configuration_sha256: String,
    case_count: usize,
    generation_probe_count: usize,
    final_transaction_count: usize,
    allow_count: usize,
    reject_count: usize,
    p50_millis: u64,
    p95_millis: u64,
    p99_millis: u64,
    max_millis: u64,
    latency_budget_millis: u64,
}

pub(crate) fn qualify(
    provider_name: &str,
    mut config: LspServerConfig,
    corpus_path: &Path,
    latency_budget_millis: u64,
) -> Result<SemanticQualificationReport> {
    if latency_budget_millis == 0 {
        bail!("semantic qualification latency budget must be positive");
    }
    let provider_version = crate::semantic::pinned_provider_version(&config).with_context(|| {
        format!(
            "semantic qualification requires {provider_name} to use one digest-pinned image whose verified manifest digest matches"
        )
    })?;
    if config.workspace_access != crate::session_environment::ServiceWorkspaceAccess::ReadOnly
        || config.network_access != crate::session_environment::ServiceNetworkAccess::None
    {
        bail!("semantic qualification requires a read-only workspace and network_access=none");
    }
    config.semantic_enforcement = LspSemanticEnforcement::Required;
    if !crate::semantic::qualified_semantic_profile(&config, "rust") {
        bail!(
            "semantic qualification requires an exact built-in Rust provider candidate profile; the image, verified digest, arguments, environment, language IDs, initialization options, workspace/network authority, and cache policy must match"
        );
    }

    let bytes = read_bounded(corpus_path, MAX_CORPUS_BYTES)?;
    let corpus_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let corpus: SemanticQualificationCorpus = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse semantic corpus {}", corpus_path.display()))?;
    validate_corpus(&corpus)?;
    let configuration_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&config)?));

    let workspace = tempfile::Builder::new()
        .prefix("pb-semantic-qualification-")
        .tempdir()
        .context("failed to create semantic qualification workspace")?;
    materialize_workspace(workspace.path(), &corpus.files)?;
    initialize_git(workspace.path())?;
    let snapshot = workspace_snapshot(&corpus.files)?;

    let runtime = crate::container::detect_runtime()
        .context("semantic qualification requires a supported local container runtime")?;
    let session_id = qualification_session_id(provider_name)?;
    let lease = crate::session_environment::global_supervisor().acquire_service_only(
        &session_id,
        workspace.path(),
        workspace.path(),
        runtime,
        false,
    )?;
    let registry = crate::lsp::discover_tools_with_lease(
        BTreeMap::from([(provider_name.to_string(), config)]),
        workspace.path(),
        Arc::clone(&lease.lease()),
    );
    let boundary =
        crate::semantic::semantic_boundary_control(&registry, workspace.path(), &snapshot)?
            .context("qualified provider was not enabled for semantic boundaries")?;

    let mut latencies = Vec::with_capacity(corpus.cases.len() * 2);
    let mut allow_count = 0usize;
    let mut reject_count = 0usize;
    let mut failures = Vec::new();
    for case in &corpus.cases {
        let started = Instant::now();
        let generation = boundary.probe(&case.tool, &case.arguments);
        latencies.push(elapsed_millis(started));
        if !generation.unknown_reasons.is_empty() {
            bail!(
                "semantic qualification provider was unavailable or non-authoritative at case {} ({:?})",
                case.id,
                generation.unknown_reasons
            );
        }
        if let Err(error) = assert_expectation(
            case,
            generation.closure,
            &generation.definite_errors,
            &generation.unknown_reasons,
        ) {
            failures.push(format!("{} generation: {error:#}", case.id));
        }
        let generation_receipt = boundary
            .stats()
            .receipt
            .with_context(|| format!("case {} did not produce generation evidence", case.id))?;
        if let Err(error) = validate_authoritative_receipt(
            case,
            &generation_receipt,
            SemanticEvidenceStage::GenerationBoundary,
            &provider_version,
        ) {
            failures.push(format!("{} generation evidence: {error:#}", case.id));
        }

        let mutations =
            crate::semantic::semantic_mutations_from_call(&snapshot, &case.tool, &case.arguments)
                .with_context(|| format!("case {} has invalid mutation arguments", case.id))?;
        let started = Instant::now();
        let (final_report, required) = crate::semantic::configured_transaction_report(
            &registry,
            workspace.path(),
            &mutations,
        )?
        .with_context(|| format!("case {} did not select a semantic provider", case.id))?;
        latencies.push(elapsed_millis(started));
        if !required {
            bail!("case {} did not run under required enforcement", case.id);
        }
        if let Err(error) = assert_expectation(
            case,
            final_report.verdict.closure,
            &final_report.verdict.definite_errors,
            &final_report.verdict.unknown_reasons,
        ) {
            failures.push(format!("{} final: {error:#}", case.id));
        }
        let final_receipt = final_report.receipt(
            SemanticEvidenceStage::FinalExecutor,
            Duration::from_millis(latency_budget_millis),
        )?;
        if let Err(error) = validate_authoritative_receipt(
            case,
            &final_receipt,
            SemanticEvidenceStage::FinalExecutor,
            &provider_version,
        ) {
            failures.push(format!("{} final evidence: {error:#}", case.id));
        }
        match final_report.verdict.closure {
            ClosureVerdict::Allow => allow_count += 1,
            ClosureVerdict::Reject => reject_count += 1,
            ClosureVerdict::Defer => {
                bail!("case {} unexpectedly deferred in required mode", case.id)
            }
        }
    }

    latencies.sort_unstable();
    let p50_millis = percentile(&latencies, 50);
    let p95_millis = percentile(&latencies, 95);
    let p99_millis = percentile(&latencies, 99);
    let max_millis = latencies.last().copied().unwrap_or_default();
    if !failures.is_empty() {
        bail!(
            "semantic qualification failed {} observation(s): {}",
            failures.len(),
            failures.join("; ")
        );
    }
    if p95_millis > latency_budget_millis {
        bail!(
            "semantic qualification p95 latency {p95_millis} ms exceeds the {latency_budget_millis} ms budget"
        );
    }

    drop(boundary);
    drop(registry);
    drop(lease);
    Ok(SemanticQualificationReport {
        version: REPORT_VERSION,
        corpus_sha256,
        provider: provider_name.to_string(),
        provider_version,
        configuration_sha256,
        case_count: corpus.cases.len(),
        generation_probe_count: corpus.cases.len(),
        final_transaction_count: corpus.cases.len(),
        allow_count,
        reject_count,
        p50_millis,
        p95_millis,
        p99_millis,
        max_millis,
        latency_budget_millis,
    })
}

fn validate_corpus(corpus: &SemanticQualificationCorpus) -> Result<()> {
    if corpus.version != CORPUS_VERSION {
        bail!("semantic qualification corpus version must be {CORPUS_VERSION}");
    }
    if corpus.files.is_empty() || corpus.files.len() > MAX_FILES {
        bail!("semantic qualification corpus must contain 1..={MAX_FILES} files");
    }
    if corpus.cases.is_empty() || corpus.cases.len() > MAX_CASES {
        bail!("semantic qualification corpus must contain 1..={MAX_CASES} cases");
    }
    let mut paths = BTreeSet::new();
    let mut total_bytes = 0usize;
    for file in &corpus.files {
        LogicalPath::parse(file.path.clone())?;
        if !paths.insert(file.path.as_str()) {
            bail!("semantic qualification corpus repeats file {}", file.path);
        }
        total_bytes = total_bytes
            .checked_add(file.content.len())
            .context("semantic qualification workspace size overflowed")?;
    }
    if total_bytes > MAX_WORKSPACE_BYTES {
        bail!("semantic qualification workspace exceeds {MAX_WORKSPACE_BYTES} bytes");
    }
    let mut ids = BTreeSet::new();
    for case in &corpus.cases {
        if case.id.trim().is_empty() || case.id.len() > 128 || !ids.insert(case.id.as_str()) {
            bail!("semantic qualification case ids must be non-empty and unique");
        }
        if !matches!(
            case.tool.as_str(),
            "write_file" | "replace_file" | "edit_file" | "apply_patch"
        ) {
            bail!(
                "case {} uses unsupported mutation tool {}",
                case.id,
                case.tool
            );
        }
        if case.expected.closure == ClosureVerdict::Defer {
            bail!(
                "case {} cannot expect Defer from a required profile",
                case.id
            );
        }
        if !case.expected.unknown_reasons.is_empty() {
            bail!(
                "case {} cannot expect Unknown from a required qualification profile",
                case.id
            );
        }
        let unique_errors = case
            .expected
            .definite_errors
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if unique_errors.len() != case.expected.definite_errors.len()
            || (case.expected.closure == ClosureVerdict::Allow && !unique_errors.is_empty())
            || (case.expected.closure == ClosureVerdict::Reject && unique_errors.is_empty())
        {
            bail!(
                "case {} must use unique definite errors consistent with its closure",
                case.id
            );
        }
    }
    Ok(())
}

fn materialize_workspace(root: &Path, files: &[QualificationFile]) -> Result<()> {
    for file in files {
        let logical = LogicalPath::parse(file.path.clone())?;
        let destination = root.join(logical.as_str());
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create semantic fixture parent for {}", file.path)
            })?;
        }
        std::fs::write(&destination, file.content.as_bytes())
            .with_context(|| format!("failed to write semantic fixture {}", file.path))?;
    }
    Ok(())
}

fn initialize_git(root: &Path) -> Result<()> {
    let status = std::process::Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(root)
        .status()
        .context("failed to start git for semantic qualification")?;
    if !status.success() {
        bail!("git init failed for semantic qualification ({status})");
    }
    Ok(())
}

fn workspace_snapshot(files: &[QualificationFile]) -> Result<WorkspaceSnapshot> {
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

fn assert_expectation(
    case: &QualificationCase,
    closure: ClosureVerdict,
    definite_errors: &[DefiniteErrorClass],
    unknown_reasons: &[UnknownReason],
) -> Result<()> {
    let mut actual_errors = definite_errors.to_vec();
    actual_errors.sort_unstable();
    actual_errors.dedup();
    let mut expected_errors = case.expected.definite_errors.clone();
    expected_errors.sort_unstable();
    expected_errors.dedup();
    let mut actual_unknown = unknown_reasons.to_vec();
    actual_unknown.sort_unstable();
    actual_unknown.dedup();
    let mut expected_unknown = case.expected.unknown_reasons.clone();
    expected_unknown.sort_unstable();
    expected_unknown.dedup();
    if closure != case.expected.closure
        || actual_errors != expected_errors
        || actual_unknown != expected_unknown
    {
        bail!(
            "semantic qualification case {} mismatch: closure={closure:?}, errors={actual_errors:?}, unknown={actual_unknown:?}",
            case.id
        );
    }
    Ok(())
}

fn validate_authoritative_receipt(
    case: &QualificationCase,
    receipt: &SemanticGateReceipt,
    stage: SemanticEvidenceStage,
    provider_version: &str,
) -> Result<()> {
    receipt.validate()?;
    assert_expectation(
        case,
        receipt.closure,
        &receipt.definite_errors,
        &receipt.unknown_reasons,
    )?;
    if receipt.stage != stage
        || receipt.providers.is_empty()
        || receipt.providers.iter().any(|provider| {
            provider.provider_version != provider_version
                || provider.baseline != BaselineCompleteness::Complete
                || !provider.authoritative
        })
    {
        bail!(
            "semantic qualification case {} did not produce authoritative {stage:?} evidence",
            case.id
        );
    }
    Ok(())
}

fn read_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open semantic corpus {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect semantic corpus {}", path.display()))?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        bail!(
            "semantic corpus {} exceeds the {max_bytes}-byte bound",
            path.display()
        );
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read semantic corpus {}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        bail!(
            "semantic corpus {} grew beyond the {max_bytes}-byte bound",
            path.display()
        );
    }
    Ok(bytes)
}

fn qualification_session_id(provider: &str) -> Result<String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    let safe_provider = provider
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    Ok(format!(
        "semantic-qualify-{safe_provider}-{}-{nanos}",
        std::process::id()
    ))
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = percentile
        .saturating_mul(sorted.len())
        .saturating_add(99)
        .saturating_div(100)
        .max(1);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_validation_rejects_duplicate_paths_and_non_authoritative_expectations() {
        let duplicate = SemanticQualificationCorpus {
            version: CORPUS_VERSION,
            files: vec![
                QualificationFile {
                    path: "src/lib.rs".to_string(),
                    content: String::new(),
                },
                QualificationFile {
                    path: "src/lib.rs".to_string(),
                    content: String::new(),
                },
            ],
            cases: vec![QualificationCase {
                id: "case".to_string(),
                tool: "replace_file".to_string(),
                arguments: serde_json::json!({}),
                expected: QualificationExpectation {
                    closure: ClosureVerdict::Allow,
                    definite_errors: Vec::new(),
                    unknown_reasons: Vec::new(),
                },
            }],
        };
        assert!(validate_corpus(&duplicate).is_err());

        let deferred = SemanticQualificationCorpus {
            version: CORPUS_VERSION,
            files: vec![QualificationFile {
                path: "src/lib.rs".to_string(),
                content: String::new(),
            }],
            cases: vec![QualificationCase {
                id: "case".to_string(),
                tool: "replace_file".to_string(),
                arguments: serde_json::json!({}),
                expected: QualificationExpectation {
                    closure: ClosureVerdict::Defer,
                    definite_errors: Vec::new(),
                    unknown_reasons: Vec::new(),
                },
            }],
        };
        assert!(validate_corpus(&deferred).is_err());

        let unknown = SemanticQualificationCorpus {
            version: CORPUS_VERSION,
            files: vec![QualificationFile {
                path: "src/lib.rs".to_string(),
                content: String::new(),
            }],
            cases: vec![QualificationCase {
                id: "case".to_string(),
                tool: "replace_file".to_string(),
                arguments: serde_json::json!({}),
                expected: QualificationExpectation {
                    closure: ClosureVerdict::Reject,
                    definite_errors: Vec::new(),
                    unknown_reasons: vec![UnknownReason::ProviderUnavailable],
                },
            }],
        };
        assert!(validate_corpus(&unknown).is_err());
    }

    #[test]
    fn percentile_uses_a_nearest_rank_upper_bound() {
        assert_eq!(percentile(&[1, 2, 3, 4, 5], 50), 3);
        assert_eq!(percentile(&[1, 2, 3, 4, 5], 95), 5);
        assert_eq!(percentile(&[], 95), 0);
    }
}
