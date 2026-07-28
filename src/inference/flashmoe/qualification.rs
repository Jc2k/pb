use std::{fs::File, io::Read, path::Path, time::Instant};

use anyhow::{Context, Result, bail};
use pb_control_collar::{
    analysis::{PrefixRule, SourcePrefixOracle, Viability, validate_supported_syntax},
    mutation::LogicalPath,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::{planning::FlashMoePlan, text::QwenTokenizer};

const PREFIX_QUALIFICATION_CONTRACT_VERSION: u32 = 1;
const MAX_PREFIX_CORPUS_BYTES: u64 = 4 * 1024 * 1024;
const MAX_PREFIX_CASES: usize = 512;
const MAX_TOKENIZER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_TOKENIZER_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
const MAX_RANDOM_CHUNK_REPLAYS_PER_CASE: usize = 65_536;

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExpectedPrefix {
    ValidComplete,
    Repairable,
    Impossible,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefixCorpus {
    version: u32,
    cases: Vec<PrefixCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefixCase {
    path: String,
    source: String,
    expected: ExpectedPrefix,
    rule: Option<PrefixRule>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PrefixQualificationReport {
    pub contract_version: u32,
    pub corpus_sha256: String,
    pub tokenizer_sha256: String,
    pub tokenizer_config_sha256: String,
    pub cases: usize,
    pub tokenizer_tokens: usize,
    pub token_prefix_probes: usize,
    pub rollback_probes: usize,
    pub random_chunk_replays_per_case: usize,
    pub random_chunk_replays: usize,
    pub probe_p50_micros: u64,
    pub probe_p95_micros: u64,
    pub probe_p99_micros: u64,
    pub probe_max_micros: u64,
    pub latency_budget_micros: u64,
    pub passed: bool,
}

pub fn qualify_prefix_tokenizer(
    plan: &FlashMoePlan,
    corpus_path: &Path,
    latency_budget_micros: u64,
    random_chunk_replays_per_case: usize,
) -> Result<PrefixQualificationReport> {
    if latency_budget_micros == 0 {
        bail!("prefix qualification latency budget must be non-zero");
    }
    if !(1..=MAX_RANDOM_CHUNK_REPLAYS_PER_CASE).contains(&random_chunk_replays_per_case) {
        bail!(
            "prefix qualification random chunk replays must be in 1..={MAX_RANDOM_CHUNK_REPLAYS_PER_CASE} per case"
        );
    }
    let corpus_bytes = read_bounded(corpus_path, MAX_PREFIX_CORPUS_BYTES, "prefix corpus")?;
    let corpus: PrefixCorpus = serde_json::from_slice(&corpus_bytes)
        .with_context(|| format!("failed to parse prefix corpus {}", corpus_path.display()))?;
    if corpus.version != PREFIX_QUALIFICATION_CONTRACT_VERSION
        || corpus.cases.is_empty()
        || corpus.cases.len() > MAX_PREFIX_CASES
    {
        bail!(
            "prefix corpus must use contract version {PREFIX_QUALIFICATION_CONTRACT_VERSION} and contain 1..={MAX_PREFIX_CASES} cases"
        );
    }
    let tokenizer_bytes = read_bounded(&plan.tokenizer, MAX_TOKENIZER_BYTES, "tokenizer")?;
    let tokenizer_config_bytes = read_bounded(
        &plan.tokenizer_config,
        MAX_TOKENIZER_CONFIG_BYTES,
        "tokenizer config",
    )?;
    let tokenizer = QwenTokenizer::from_files(
        &plan.tokenizer,
        &plan.tokenizer_config,
        plan.chat_template
            .is_file()
            .then_some(plan.chat_template.as_path()),
    )?;

    let mut tokenizer_tokens = 0usize;
    let mut token_prefix_probes = 0usize;
    let mut rollback_probes = 0usize;
    let mut random_chunk_replays = 0usize;
    let mut probe_nanos = Vec::new();
    for case in &corpus.cases {
        let path = LogicalPath::parse(case.path.clone())?;
        let source = case.source.as_bytes();
        match case.expected {
            ExpectedPrefix::Impossible if case.rule.is_none() => {
                bail!("impossible prefix case {:?} must name its rule", case.path)
            }
            ExpectedPrefix::ValidComplete | ExpectedPrefix::Repairable if case.rule.is_some() => {
                bail!(
                    "non-impossible prefix case {:?} cannot name a rule",
                    case.path
                )
            }
            _ => {}
        }
        if matches!(case.expected, ExpectedPrefix::ValidComplete) {
            validate_supported_syntax(&path, source).with_context(|| {
                format!(
                    "valid-complete prefix corpus case {:?} is not valid",
                    case.path
                )
            })?;
        }
        let tokens = tokenizer.encode(&case.source)?;
        if tokens.is_empty() {
            bail!("prefix corpus case {:?} tokenized to no tokens", case.path);
        }
        tokenizer_tokens = tokenizer_tokens.saturating_add(tokens.len());
        let decoded = tokenizer.decode(&tokens)?;
        if decoded != case.source {
            bail!(
                "tokenizer round trip changed prefix corpus case {:?}: decoded {} bytes, expected {}",
                case.path,
                decoded.len(),
                source.len()
            );
        }

        let mut oracle = SourcePrefixOracle::new(path.clone(), source.len().max(1))?;
        let mut prior = String::new();
        for end in 1..=tokens.len() {
            let decoded = tokenizer.decode(&tokens[..end])?;
            let Some(delta) = decoded.strip_prefix(&prior) else {
                bail!(
                    "tokenizer decoding is not prefix-monotonic at token {end} for {:?}",
                    case.path
                );
            };
            let started = Instant::now();
            let checkpoint = oracle.checkpoint();
            let probe = oracle.push(delta.as_bytes())?;
            oracle.rollback(checkpoint)?;
            probe_nanos.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
            let committed = oracle.push(delta.as_bytes())?;
            if committed != probe {
                bail!(
                    "rollback replay diverged at token {end} for {:?}",
                    case.path
                );
            }
            if !matches!(case.expected, ExpectedPrefix::Impossible)
                && committed.viability == Viability::Impossible
            {
                bail!(
                    "valid or repairable case {:?} became impossible at token {end} ({:?})",
                    case.path,
                    committed.rule
                );
            }
            prior = decoded;
            token_prefix_probes = token_prefix_probes.saturating_add(1);
            rollback_probes = rollback_probes.saturating_add(1);
        }
        let final_report = oracle.report()?;
        match case.expected {
            ExpectedPrefix::ValidComplete | ExpectedPrefix::Repairable
                if final_report.viability != Viability::Repairable =>
            {
                bail!("prefix corpus case {:?} unexpectedly rejected", case.path)
            }
            ExpectedPrefix::Impossible
                if final_report.viability != Viability::Impossible
                    || final_report.rule != case.rule =>
            {
                bail!(
                    "prefix corpus case {:?} expected {:?}, got {:?}",
                    case.path,
                    case.rule,
                    final_report.rule
                )
            }
            _ => {}
        }

        for seed in 1usize..=random_chunk_replays_per_case {
            let mut chunked = SourcePrefixOracle::new(path.clone(), source.len().max(1))?;
            let mut state = seed;
            let mut cursor = 0usize;
            while cursor < source.len() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let end = cursor.saturating_add(state % 31 + 1).min(source.len());
                chunked.push(&source[cursor..end])?;
                cursor = end;
            }
            if chunked.report()? != final_report {
                bail!(
                    "random chunk replay seed {seed} diverged for {:?}",
                    case.path
                );
            }
            random_chunk_replays = random_chunk_replays.saturating_add(1);
        }
    }

    if probe_nanos.is_empty() {
        bail!("prefix qualification produced no latency samples");
    }
    probe_nanos.sort_unstable();
    let percentile_micros = |percentile: usize| -> u64 {
        let rank = percentile
            .saturating_mul(probe_nanos.len())
            .saturating_add(99)
            .saturating_div(100)
            .max(1);
        let index = rank.saturating_sub(1).min(probe_nanos.len() - 1);
        probe_nanos[index].saturating_add(999) / 1_000
    };
    let p95 = percentile_micros(95);
    if p95 > latency_budget_micros {
        bail!("real-token prefix probe p95 {p95} us exceeds the {latency_budget_micros} us budget");
    }
    Ok(PrefixQualificationReport {
        contract_version: PREFIX_QUALIFICATION_CONTRACT_VERSION,
        corpus_sha256: lower_sha256(&corpus_bytes),
        tokenizer_sha256: lower_sha256(&tokenizer_bytes),
        tokenizer_config_sha256: lower_sha256(&tokenizer_config_bytes),
        cases: corpus.cases.len(),
        tokenizer_tokens,
        token_prefix_probes,
        rollback_probes,
        random_chunk_replays_per_case,
        random_chunk_replays,
        probe_p50_micros: percentile_micros(50),
        probe_p95_micros: p95,
        probe_p99_micros: percentile_micros(99),
        probe_max_micros: probe_nanos
            .last()
            .copied()
            .unwrap_or_default()
            .saturating_add(999)
            / 1_000,
        latency_budget_micros,
        passed: true,
    })
}

fn lower_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_bounded(path: &Path, max_bytes: u64, label: &str) -> Result<Vec<u8>> {
    let file =
        File::open(path).with_context(|| format!("failed to open {label} {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        bail!(
            "{label} {} exceeds the {max_bytes}-byte bound",
            path.display()
        );
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {label} {}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        bail!(
            "{label} {} grew beyond the {max_bytes}-byte bound",
            path.display()
        );
    }
    Ok(bytes)
}
