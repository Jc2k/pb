//! Process-isolated qualification for native language-world lifecycle and resource behavior.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::mem::MaybeUninit;

use anyhow::{Context, Result, bail};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::control_layers::{
    PythonWorldQualificationObservation, RustWorldQualificationObservation,
    qualify_python_world_fixture, qualify_rust_world_fixture,
};

const NATIVE_WORLD_QUALIFICATION_VERSION: u32 = 2;
const PYTHON_DEFAULT_MAX_COLD_MILLIS: u64 = 60_000;
const RUST_DEFAULT_MAX_COLD_MILLIS: u64 = 180_000;
const DEFAULT_MAX_WARM_MILLIS: u64 = 20_000;
const PYTHON_DEFAULT_MAX_REPLAY_MILLIS: u64 = 20_000;
const RUST_DEFAULT_MAX_REPLAY_MILLIS: u64 = 30_000;
const PYTHON_DEFAULT_MAX_STRESS_MILLIS: u64 = 120_000;
const RUST_DEFAULT_MAX_STRESS_MILLIS: u64 = 240_000;
const PYTHON_DEFAULT_MAX_PEAK_RESIDENT_BYTES: u64 = 1_073_741_824;
const RUST_DEFAULT_MAX_PEAK_RESIDENT_BYTES: u64 = 4_294_967_296;
const DEFAULT_MAX_RETAINED_GROWTH_BYTES: u64 = 536_870_912;
const PYTHON_FIXTURE_GITIGNORE: &str = ".venv/\n";
const PYTHON_FIXTURE_ENVIRONMENT: &str = "version = 3.12.8\n";
const PYTHON_FIXTURE_PACKAGE_INIT: &str = "";
const PYTHON_FIXTURE_PY_TYPED: &str = "";
const PYTHON_FIXTURE_METADATA: &str = "Metadata-Version: 2.1\nName: dependency-pkg\nVersion: 1.0\n";
const PYTHON_FIXTURE_MAIN: &str =
    "from dependency_pkg.module_00000 import parse\nresult: str = parse(\"ok\")\n";
const PYTHON_FIXTURE_INVALID: &str =
    "from dependency_pkg.module_00000 import parse\nresult: str = parse(1)\n";
const PYTHON_FIXTURE_VALID: &str =
    "from dependency_pkg.module_00000 import parse\nresult: str = parse(\"qualified\")\n";
const RUST_FIXTURE_TARGET: &str = "app/src/lib.rs";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum NativeWorldLanguage {
    Python,
    Rust,
}

impl NativeWorldLanguage {
    const fn id(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Rust => "rust",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum NativeWorldCase {
    Tiny,
    Representative,
    Large,
}

impl NativeWorldCase {
    const ALL: [Self; 3] = [Self::Tiny, Self::Representative, Self::Large];

    fn id(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Representative => "representative",
            Self::Large => "large",
        }
    }

    fn python_spec(self) -> NativeFixtureSpec {
        match self {
            Self::Tiny => NativeFixtureSpec {
                first_party_files: 4,
                dependency_modules: 4,
                stress_workers: 4,
                stress_replays_per_worker: 16,
            },
            Self::Representative => NativeFixtureSpec {
                first_party_files: 1_024,
                dependency_modules: 512,
                stress_workers: 4,
                stress_replays_per_worker: 8,
            },
            Self::Large => NativeFixtureSpec {
                first_party_files: 10_000,
                dependency_modules: 5_000,
                stress_workers: 2,
                stress_replays_per_worker: 4,
            },
        }
    }

    fn rust_spec(self) -> NativeFixtureSpec {
        match self {
            Self::Tiny => NativeFixtureSpec {
                first_party_files: 4,
                dependency_modules: 4,
                stress_workers: 4,
                stress_replays_per_worker: 16,
            },
            Self::Representative => NativeFixtureSpec {
                first_party_files: 256,
                dependency_modules: 128,
                stress_workers: 4,
                stress_replays_per_worker: 8,
            },
            Self::Large => NativeFixtureSpec {
                first_party_files: 2_048,
                dependency_modules: 1_024,
                stress_workers: 2,
                stress_replays_per_worker: 4,
            },
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct HarnessNativeWorldQualifyArgs {
    /// Native language implementation to qualify
    #[arg(long, value_enum, default_value_t = NativeWorldLanguage::Python)]
    pub(crate) language: NativeWorldLanguage,

    /// Override the language-profile cold preparation ceiling
    #[arg(long)]
    pub(crate) max_cold_millis: Option<u64>,

    /// Maximum accepted warm or exact process-cache request time for each matrix case
    #[arg(long)]
    pub(crate) max_warm_millis: Option<u64>,

    /// Maximum accepted independent final replay time for each matrix case
    #[arg(long)]
    pub(crate) max_replay_millis: Option<u64>,

    /// Override the aggregate serialized-overlay stress ceiling
    #[arg(long)]
    pub(crate) max_stress_millis: Option<u64>,

    /// Maximum accepted process peak resident memory for each isolated matrix case
    #[arg(long)]
    pub(crate) max_peak_resident_bytes: Option<u64>,

    /// Override the maximum current-resident growth retained after overlay stress
    #[arg(long)]
    pub(crate) max_retained_growth_bytes: Option<u64>,

    /// Internal process-isolated matrix arm
    #[arg(long, value_enum, hide = true)]
    pub(crate) case: Option<NativeWorldCase>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct NativeWorldQualificationLimits {
    max_cold_millis: u64,
    max_warm_millis: u64,
    max_replay_millis: u64,
    max_stress_millis: u64,
    max_peak_resident_bytes: u64,
    max_retained_growth_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct NativeFixtureSpec {
    first_party_files: usize,
    dependency_modules: usize,
    stress_workers: usize,
    stress_replays_per_worker: usize,
}

#[derive(Debug)]
struct PythonFixture {
    _root_owner: tempfile::TempDir,
    root: PathBuf,
    first_party_files: usize,
    dependency_files: usize,
    first_party_bytes: u64,
    dependency_bytes: u64,
}

#[derive(Debug)]
struct RustFixture {
    _root_owner: tempfile::TempDir,
    root: PathBuf,
    first_party_files: usize,
    dependency_files: usize,
    first_party_bytes: u64,
    dependency_bytes: u64,
    invalid_source: String,
    valid_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct NativeWorldCaseReport {
    case: String,
    first_party_files: usize,
    dependency_files: usize,
    first_party_bytes: u64,
    dependency_bytes: u64,
    provider_version: String,
    world_sha256: String,
    configuration_sha256: String,
    dependency_sha256: String,
    target_count: usize,
    semantic_profile: String,
    load_millis: u64,
    prime_millis: u64,
    primed_queries: u64,
    cold_millis: u64,
    warm_millis: u64,
    process_cache_hit_millis: u64,
    invalid_replay_millis: u64,
    valid_replay_millis: u64,
    stress_workers: usize,
    stress_replay_count: u64,
    stress_millis: u64,
    max_stress_replay_millis: u64,
    baseline_peak_resident_bytes: u64,
    peak_resident_bytes: u64,
    incremental_peak_resident_bytes: u64,
    pre_stress_resident_bytes: u64,
    post_stress_resident_bytes: u64,
    retained_growth_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct NativeWorldQualificationReport {
    version: u32,
    language: String,
    profile_sha256: String,
    limits: NativeWorldQualificationLimits,
    cases: Vec<NativeWorldCaseReport>,
    passed: bool,
}

pub(crate) fn run(args: HarnessNativeWorldQualifyArgs) -> Result<()> {
    if let Some(case) = args.case {
        let report = match args.language {
            NativeWorldLanguage::Python => qualify_python_case(case)?,
            NativeWorldLanguage::Rust => qualify_rust_case(case)?,
        };
        println!("{}", serde_json::to_string(&report)?);
        return Ok(());
    }

    let limits = qualification_limits(&args);
    validate_limits(limits)?;
    let cases = NativeWorldCase::ALL
        .into_iter()
        .map(|case| run_isolated_case(args.language, case, limits))
        .collect::<Result<Vec<_>>>()?;
    for case in &cases {
        validate_case(args.language, case, limits)?;
    }
    let report = NativeWorldQualificationReport {
        version: NATIVE_WORLD_QUALIFICATION_VERSION,
        language: args.language.id().to_string(),
        profile_sha256: qualification_profile_sha256(args.language),
        limits,
        cases,
        passed: true,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn qualification_limits(args: &HarnessNativeWorldQualifyArgs) -> NativeWorldQualificationLimits {
    let (cold, replay, stress, peak) = match args.language {
        NativeWorldLanguage::Python => (
            PYTHON_DEFAULT_MAX_COLD_MILLIS,
            PYTHON_DEFAULT_MAX_REPLAY_MILLIS,
            PYTHON_DEFAULT_MAX_STRESS_MILLIS,
            PYTHON_DEFAULT_MAX_PEAK_RESIDENT_BYTES,
        ),
        NativeWorldLanguage::Rust => (
            RUST_DEFAULT_MAX_COLD_MILLIS,
            RUST_DEFAULT_MAX_REPLAY_MILLIS,
            RUST_DEFAULT_MAX_STRESS_MILLIS,
            RUST_DEFAULT_MAX_PEAK_RESIDENT_BYTES,
        ),
    };
    NativeWorldQualificationLimits {
        max_cold_millis: args.max_cold_millis.unwrap_or(cold),
        max_warm_millis: args.max_warm_millis.unwrap_or(DEFAULT_MAX_WARM_MILLIS),
        max_replay_millis: args.max_replay_millis.unwrap_or(replay),
        max_stress_millis: args.max_stress_millis.unwrap_or(stress),
        max_peak_resident_bytes: args.max_peak_resident_bytes.unwrap_or(peak),
        max_retained_growth_bytes: args
            .max_retained_growth_bytes
            .unwrap_or(DEFAULT_MAX_RETAINED_GROWTH_BYTES),
    }
}

fn validate_limits(limits: NativeWorldQualificationLimits) -> Result<()> {
    if limits.max_cold_millis == 0
        || limits.max_warm_millis == 0
        || limits.max_replay_millis == 0
        || limits.max_stress_millis == 0
        || limits.max_peak_resident_bytes == 0
        || limits.max_retained_growth_bytes == 0
    {
        bail!("native-world qualification limits must all be non-zero");
    }
    Ok(())
}

fn run_isolated_case(
    language: NativeWorldLanguage,
    case: NativeWorldCase,
    limits: NativeWorldQualificationLimits,
) -> Result<NativeWorldCaseReport> {
    let executable = std::env::current_exe()
        .context("failed to locate the current pb executable for isolated qualification")?;
    let mut child = Command::new(&executable)
        .args([
            "harness",
            "native-world-qualify",
            "--language",
            language.id(),
            "--case",
            case.id(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "failed to start isolated {} native-world case {}",
                language.id(),
                case.id()
            )
        })?;
    let timeout = isolated_case_timeout(limits)?;
    let started = Instant::now();
    loop {
        if child
            .try_wait()
            .context("failed to poll isolated native-world qualification")?
            .is_some()
        {
            break;
        }
        if started.elapsed() >= timeout {
            child
                .kill()
                .context("failed to terminate timed-out native-world qualification")?;
            let output = child
                .wait_with_output()
                .context("failed to reap timed-out native-world qualification")?;
            bail!(
                "isolated {} native-world case {} exceeded its {} ms aggregate deadline: {}",
                language.id(),
                case.id(),
                timeout.as_millis(),
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let output = child
        .wait_with_output()
        .context("failed to collect isolated native-world qualification")?;
    if !output.status.success() {
        bail!(
            "isolated {} native-world case {} failed: {}",
            language.id(),
            case.id(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    serde_json::from_slice(&output.stdout).with_context(|| {
        format!(
            "isolated {} native-world case {} returned an invalid report",
            language.id(),
            case.id()
        )
    })
}

fn isolated_case_timeout(limits: NativeWorldQualificationLimits) -> Result<Duration> {
    let millis = limits
        .max_cold_millis
        .checked_add(
            limits
                .max_warm_millis
                .checked_mul(2)
                .context("native-world warm deadline overflowed")?,
        )
        .and_then(|total| {
            limits
                .max_replay_millis
                .checked_mul(2)
                .and_then(|replay| total.checked_add(replay))
        })
        .and_then(|total| total.checked_add(limits.max_stress_millis))
        .and_then(|total| total.checked_add(30_000))
        .context("native-world aggregate deadline overflowed")?;
    Ok(Duration::from_millis(millis))
}

fn qualify_python_case(case: NativeWorldCase) -> Result<NativeWorldCaseReport> {
    let fixture = build_python_fixture(case)?;
    let spec = case.python_spec();
    let baseline_peak_resident_bytes = process_peak_resident_bytes()?;
    let observation = qualify_python_world_fixture(
        &fixture.root,
        "main.py",
        PYTHON_FIXTURE_INVALID,
        PYTHON_FIXTURE_VALID,
        spec.stress_workers,
        spec.stress_replays_per_worker,
        &process_current_resident_bytes,
    )?;
    let peak_resident_bytes = process_peak_resident_bytes()?;
    Ok(python_case_report(
        case,
        spec,
        fixture,
        observation,
        baseline_peak_resident_bytes,
        peak_resident_bytes,
    ))
}

fn python_case_report(
    case: NativeWorldCase,
    spec: NativeFixtureSpec,
    fixture: PythonFixture,
    observation: PythonWorldQualificationObservation,
    baseline_peak_resident_bytes: u64,
    peak_resident_bytes: u64,
) -> NativeWorldCaseReport {
    NativeWorldCaseReport {
        case: case.id().to_string(),
        first_party_files: fixture.first_party_files,
        dependency_files: fixture.dependency_files,
        first_party_bytes: fixture.first_party_bytes,
        dependency_bytes: fixture.dependency_bytes,
        provider_version: observation.provider_version,
        world_sha256: observation.world_sha256,
        configuration_sha256: observation.configuration_sha256,
        dependency_sha256: observation.dependency_sha256,
        target_count: 0,
        semantic_profile: "exact-ty".to_string(),
        load_millis: observation.load_millis,
        prime_millis: observation.prime_millis,
        primed_queries: observation.primed_queries,
        cold_millis: observation.cold_millis,
        warm_millis: observation.warm_millis,
        process_cache_hit_millis: observation.process_cache_hit_millis,
        invalid_replay_millis: observation.invalid_replay_millis,
        valid_replay_millis: observation.valid_replay_millis,
        stress_workers: spec.stress_workers,
        stress_replay_count: observation.stress_replay_count,
        stress_millis: observation.stress_millis,
        max_stress_replay_millis: observation.max_stress_replay_millis,
        baseline_peak_resident_bytes,
        peak_resident_bytes,
        incremental_peak_resident_bytes: peak_resident_bytes
            .saturating_sub(baseline_peak_resident_bytes),
        pre_stress_resident_bytes: observation.pre_stress_resident_bytes,
        post_stress_resident_bytes: observation.post_stress_resident_bytes,
        retained_growth_bytes: observation
            .post_stress_resident_bytes
            .saturating_sub(observation.pre_stress_resident_bytes),
    }
}

fn qualify_rust_case(case: NativeWorldCase) -> Result<NativeWorldCaseReport> {
    let fixture = build_rust_fixture(case)?;
    let spec = case.rust_spec();
    let baseline_peak_resident_bytes = process_peak_resident_bytes()?;
    let observation = qualify_rust_world_fixture(
        &fixture.root,
        RUST_FIXTURE_TARGET,
        &fixture.invalid_source,
        &fixture.valid_source,
        spec.stress_workers,
        spec.stress_replays_per_worker,
        &process_current_resident_bytes,
    )?;
    let peak_resident_bytes = process_peak_resident_bytes()?;
    Ok(rust_case_report(
        case,
        spec,
        fixture,
        observation,
        baseline_peak_resident_bytes,
        peak_resident_bytes,
    ))
}

fn rust_case_report(
    case: NativeWorldCase,
    spec: NativeFixtureSpec,
    fixture: RustFixture,
    observation: RustWorldQualificationObservation,
    baseline_peak_resident_bytes: u64,
    peak_resident_bytes: u64,
) -> NativeWorldCaseReport {
    NativeWorldCaseReport {
        case: case.id().to_string(),
        first_party_files: fixture.first_party_files,
        dependency_files: fixture.dependency_files,
        first_party_bytes: fixture.first_party_bytes,
        dependency_bytes: fixture.dependency_bytes,
        provider_version: observation.provider_version,
        world_sha256: observation.world_sha256,
        configuration_sha256: observation.configuration_sha256,
        dependency_sha256: observation.dependency_sha256,
        target_count: observation.target_count,
        semantic_profile: match observation.deep_profile {
            pb_control_rust::RustDeepProfile::Exact => "exact-rust-v2".to_string(),
            profile => format!("unexpected-{profile:?}"),
        },
        load_millis: observation.load_millis,
        prime_millis: observation.prime_millis,
        primed_queries: observation.primed_queries,
        cold_millis: observation.cold_millis,
        warm_millis: observation.warm_millis,
        process_cache_hit_millis: observation.process_cache_hit_millis,
        invalid_replay_millis: observation.invalid_replay_millis,
        valid_replay_millis: observation.valid_replay_millis,
        stress_workers: spec.stress_workers,
        stress_replay_count: observation.stress_replay_count,
        stress_millis: observation.stress_millis,
        max_stress_replay_millis: observation.max_stress_replay_millis,
        baseline_peak_resident_bytes,
        peak_resident_bytes,
        incremental_peak_resident_bytes: peak_resident_bytes
            .saturating_sub(baseline_peak_resident_bytes),
        pre_stress_resident_bytes: observation.pre_stress_resident_bytes,
        post_stress_resident_bytes: observation.post_stress_resident_bytes,
        retained_growth_bytes: observation
            .post_stress_resident_bytes
            .saturating_sub(observation.pre_stress_resident_bytes),
    }
}

fn validate_case(
    language: NativeWorldLanguage,
    case: &NativeWorldCaseReport,
    limits: NativeWorldQualificationLimits,
) -> Result<()> {
    if case.cold_millis > limits.max_cold_millis {
        bail!(
            "{} native-world case {} exceeded cold budget: {} > {} ms",
            language.id(),
            case.case,
            case.cold_millis,
            limits.max_cold_millis
        );
    }
    for (label, actual) in [
        ("warm", case.warm_millis),
        ("process-cache", case.process_cache_hit_millis),
    ] {
        if actual > limits.max_warm_millis {
            bail!(
                "{} native-world case {} exceeded {label} budget: {actual} > {} ms",
                language.id(),
                case.case,
                limits.max_warm_millis
            );
        }
    }
    for (label, actual) in [
        ("invalid replay", case.invalid_replay_millis),
        ("valid replay", case.valid_replay_millis),
    ] {
        if actual > limits.max_replay_millis {
            bail!(
                "{} native-world case {} exceeded {label} budget: {actual} > {} ms",
                language.id(),
                case.case,
                limits.max_replay_millis
            );
        }
    }
    if case.peak_resident_bytes > limits.max_peak_resident_bytes {
        bail!(
            "{} native-world case {} exceeded peak resident budget: {} > {} bytes",
            language.id(),
            case.case,
            case.peak_resident_bytes,
            limits.max_peak_resident_bytes
        );
    }
    if case.stress_millis > limits.max_stress_millis {
        bail!(
            "{} native-world case {} exceeded stress budget: {} > {} ms",
            language.id(),
            case.case,
            case.stress_millis,
            limits.max_stress_millis
        );
    }
    if case.max_stress_replay_millis > limits.max_replay_millis {
        bail!(
            "{} native-world case {} exceeded per-replay stress budget: {} > {} ms",
            language.id(),
            case.case,
            case.max_stress_replay_millis,
            limits.max_replay_millis
        );
    }
    if case.retained_growth_bytes > limits.max_retained_growth_bytes {
        bail!(
            "{} native-world case {} retained {} bytes after stress, above {} bytes",
            language.id(),
            case.case,
            case.retained_growth_bytes,
            limits.max_retained_growth_bytes
        );
    }
    let spec = match language {
        NativeWorldLanguage::Python => NativeWorldCase::ALL
            .into_iter()
            .find(|candidate| candidate.id() == case.case)
            .context("Python native-world report has an unknown case")?
            .python_spec(),
        NativeWorldLanguage::Rust => NativeWorldCase::ALL
            .into_iter()
            .find(|candidate| candidate.id() == case.case)
            .context("Rust native-world report has an unknown case")?
            .rust_spec(),
    };
    let expected_replays = spec
        .stress_workers
        .saturating_mul(spec.stress_replays_per_worker)
        .saturating_add(1) as u64;
    if case.stress_workers != spec.stress_workers || case.stress_replay_count != expected_replays {
        bail!(
            "{} native-world case {} did not finish its complete serialized-overlay matrix",
            language.id(),
            case.case
        );
    }
    match language {
        NativeWorldLanguage::Python => {
            let minimum_queries = case
                .first_party_files
                .saturating_add(case.dependency_files.saturating_sub(2));
            if case.primed_queries < minimum_queries as u64 {
                bail!(
                    "Python native-world case {} did not prime every source module before inference",
                    case.case
                );
            }
        }
        NativeWorldLanguage::Rust => {
            if case.target_count != 2
                || case.semantic_profile != "exact-rust-v2"
                || case.primed_queries < 2
            {
                bail!(
                    "Rust native-world case {} did not prepare its exact two-target semantic world",
                    case.case
                );
            }
        }
    }
    Ok(())
}

fn build_python_fixture(case: NativeWorldCase) -> Result<PythonFixture> {
    let spec = case.python_spec();
    let owner = tempfile::Builder::new()
        .prefix(&format!("pb-python-world-{}-", case.id()))
        .tempdir()
        .context("failed to create Python native-world qualification fixture")?;
    let root = owner.path().to_path_buf();
    let status = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(&root)
        .status()
        .context("failed to initialize Python qualification Git fixture")?;
    if !status.success() {
        bail!("failed to initialize Python qualification Git fixture");
    }
    fs::write(root.join(".gitignore"), PYTHON_FIXTURE_GITIGNORE)?;

    let site_packages = root.join(".venv/lib/python3.12/site-packages");
    fs::create_dir_all(site_packages.join("dependency_pkg"))?;
    fs::create_dir_all(site_packages.join("dependency-pkg-1.0.dist-info"))?;
    fs::write(root.join(".venv/pyvenv.cfg"), PYTHON_FIXTURE_ENVIRONMENT)?;

    let mut dependency_bytes = 0u64;
    dependency_bytes = dependency_bytes.saturating_add(write_counted(
        &site_packages.join("dependency_pkg/__init__.py"),
        PYTHON_FIXTURE_PACKAGE_INIT,
    )?);
    dependency_bytes = dependency_bytes.saturating_add(write_counted(
        &site_packages.join("dependency_pkg/py.typed"),
        PYTHON_FIXTURE_PY_TYPED,
    )?);
    dependency_bytes = dependency_bytes.saturating_add(write_counted(
        &site_packages.join("dependency-pkg-1.0.dist-info/METADATA"),
        PYTHON_FIXTURE_METADATA,
    )?);
    for index in 0..spec.dependency_modules {
        let source = python_dependency_source(index);
        dependency_bytes = dependency_bytes.saturating_add(write_counted(
            &site_packages.join(python_dependency_path(index)),
            &source,
        )?);
    }

    let mut first_party_bytes = write_counted(&root.join("main.py"), PYTHON_FIXTURE_MAIN)?;
    for index in 1..spec.first_party_files {
        let dependency = index % spec.dependency_modules;
        let source = python_first_party_source(index, dependency);
        first_party_bytes = first_party_bytes.saturating_add(write_counted(
            &root.join(python_first_party_path(index)),
            &source,
        )?);
    }

    Ok(PythonFixture {
        _root_owner: owner,
        root,
        first_party_files: spec.first_party_files,
        dependency_files: spec.dependency_modules.saturating_add(3),
        first_party_bytes,
        dependency_bytes,
    })
}

fn build_rust_fixture(case: NativeWorldCase) -> Result<RustFixture> {
    let spec = case.rust_spec();
    let owner = tempfile::Builder::new()
        .prefix(&format!("pb-rust-world-{}-", case.id()))
        .tempdir()
        .context("failed to create Rust native-world qualification fixture")?;
    let root = owner.path().to_path_buf();
    let status = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(&root)
        .status()
        .context("failed to initialize Rust qualification Git fixture")?;
    if !status.success() {
        bail!("failed to initialize Rust qualification Git fixture");
    }
    fs::create_dir_all(root.join("app/src"))?;
    fs::create_dir_all(root.join("dep/src"))?;

    let workspace_manifest = "[workspace]\nmembers = [\"app\", \"dep\"]\nresolver = \"3\"\n";
    let app_manifest = "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\ndep = { path = \"../dep\" }\n";
    let dep_manifest = "[package]\nname = \"dep\"\nversion = \"0.1.0\"\nedition = \"2024\"\n";
    let valid_source = rust_app_root_source(spec.first_party_files, false);
    let baseline_source = valid_source.replace("parse(41)", "parse(40)");
    let invalid_source = rust_app_root_source(spec.first_party_files, true);
    let dependency_root = rust_dependency_root_source(spec.dependency_modules);

    let mut first_party_bytes = write_counted(&root.join("Cargo.toml"), workspace_manifest)?;
    first_party_bytes = first_party_bytes
        .saturating_add(write_counted(&root.join("app/Cargo.toml"), app_manifest)?);
    first_party_bytes = first_party_bytes.saturating_add(write_counted(
        &root.join(RUST_FIXTURE_TARGET),
        &baseline_source,
    )?);
    for index in 1..spec.first_party_files {
        first_party_bytes = first_party_bytes.saturating_add(write_counted(
            &root.join(format!("app/src/module_{index:05}.rs")),
            &rust_first_party_source(index, index % spec.dependency_modules),
        )?);
    }

    let mut dependency_bytes = write_counted(&root.join("dep/Cargo.toml"), dep_manifest)?;
    dependency_bytes = dependency_bytes.saturating_add(write_counted(
        &root.join("dep/src/lib.rs"),
        &dependency_root,
    )?);
    for index in 0..spec.dependency_modules {
        dependency_bytes = dependency_bytes.saturating_add(write_counted(
            &root.join(format!("dep/src/module_{index:05}.rs")),
            &rust_dependency_source(index),
        )?);
    }

    Ok(RustFixture {
        _root_owner: owner,
        root,
        first_party_files: spec.first_party_files.saturating_add(2),
        dependency_files: spec.dependency_modules.saturating_add(2),
        first_party_bytes,
        dependency_bytes,
        invalid_source,
        valid_source,
    })
}

fn write_counted(path: &Path, contents: &str) -> Result<u64> {
    fs::write(path, contents)
        .with_context(|| format!("failed to write qualification fixture {}", path.display()))?;
    Ok(contents.len() as u64)
}

fn python_dependency_path(index: usize) -> String {
    format!("dependency_pkg/module_{index:05}.py")
}

fn python_dependency_source(index: usize) -> String {
    format!("def parse(value: str) -> str:\n    return value + \"-{index:05}\"\n")
}

fn python_first_party_path(index: usize) -> String {
    format!("module_{index:05}.py")
}

fn python_first_party_source(index: usize, dependency: usize) -> String {
    format!(
        "from dependency_pkg.module_{dependency:05} import parse\n\ndef transform_{index:05}(value: str) -> str:\n    return parse(value)\n"
    )
}

fn rust_app_root_source(first_party_files: usize, invalid: bool) -> String {
    let mut source = String::new();
    for index in 1..first_party_files {
        source.push_str(&format!("pub mod module_{index:05};\n"));
    }
    if invalid {
        source.push_str("pub fn qualified() -> i32 { dep::module_00000::parse(\"wrong type\") }\n");
    } else {
        source.push_str("pub fn qualified() -> i32 { dep::module_00000::parse(41) }\n");
    }
    source
}

fn rust_dependency_root_source(dependency_modules: usize) -> String {
    let mut source = String::new();
    for index in 0..dependency_modules {
        source.push_str(&format!("pub mod module_{index:05};\n"));
    }
    source
}

fn rust_dependency_source(index: usize) -> String {
    format!("pub fn parse(value: i32) -> i32 {{ value + {index} }}\n")
}

fn rust_first_party_source(index: usize, dependency: usize) -> String {
    format!(
        "pub fn transform_{index:05}(value: i32) -> i32 {{ dep::module_{dependency:05}::parse(value) }}\n"
    )
}

fn qualification_profile_sha256(language: NativeWorldLanguage) -> String {
    let mut digest = Sha256::new();
    digest.update(b"pb-native-world-qualification-v2\0");
    digest.update(language.id().as_bytes());
    match language {
        NativeWorldLanguage::Python => {
            digest.update(b"\0ty_0.0.6\0");
            for case in NativeWorldCase::ALL {
                let spec = case.python_spec();
                hash_native_spec(&mut digest, case, spec);
                hash_qualification_record(&mut digest, ".gitignore", PYTHON_FIXTURE_GITIGNORE);
                hash_qualification_record(
                    &mut digest,
                    ".venv/pyvenv.cfg",
                    PYTHON_FIXTURE_ENVIRONMENT,
                );
                hash_qualification_record(
                    &mut digest,
                    ".venv/lib/python3.12/site-packages/dependency_pkg/__init__.py",
                    PYTHON_FIXTURE_PACKAGE_INIT,
                );
                hash_qualification_record(
                    &mut digest,
                    ".venv/lib/python3.12/site-packages/dependency_pkg/py.typed",
                    PYTHON_FIXTURE_PY_TYPED,
                );
                hash_qualification_record(
                    &mut digest,
                    ".venv/lib/python3.12/site-packages/dependency-pkg-1.0.dist-info/METADATA",
                    PYTHON_FIXTURE_METADATA,
                );
                for index in 0..spec.dependency_modules {
                    hash_qualification_record(
                        &mut digest,
                        &format!(
                            ".venv/lib/python3.12/site-packages/{}",
                            python_dependency_path(index)
                        ),
                        &python_dependency_source(index),
                    );
                }
                hash_qualification_record(&mut digest, "main.py", PYTHON_FIXTURE_MAIN);
                for index in 1..spec.first_party_files {
                    hash_qualification_record(
                        &mut digest,
                        &python_first_party_path(index),
                        &python_first_party_source(index, index % spec.dependency_modules),
                    );
                }
            }
            hash_qualification_record(&mut digest, "<invalid-replay>", PYTHON_FIXTURE_INVALID);
            hash_qualification_record(&mut digest, "<valid-replay>", PYTHON_FIXTURE_VALID);
        }
        NativeWorldLanguage::Rust => {
            digest.update(b"\0rust-analyzer_0.0.344\0exact-v2\0");
            let workspace_manifest =
                "[workspace]\nmembers = [\"app\", \"dep\"]\nresolver = \"3\"\n";
            let app_manifest = "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\ndep = { path = \"../dep\" }\n";
            let dep_manifest =
                "[package]\nname = \"dep\"\nversion = \"0.1.0\"\nedition = \"2024\"\n";
            for case in NativeWorldCase::ALL {
                let spec = case.rust_spec();
                hash_native_spec(&mut digest, case, spec);
                hash_qualification_record(&mut digest, "Cargo.toml", workspace_manifest);
                hash_qualification_record(&mut digest, "app/Cargo.toml", app_manifest);
                hash_qualification_record(
                    &mut digest,
                    RUST_FIXTURE_TARGET,
                    &rust_app_root_source(spec.first_party_files, false)
                        .replace("parse(41)", "parse(40)"),
                );
                for index in 1..spec.first_party_files {
                    hash_qualification_record(
                        &mut digest,
                        &format!("app/src/module_{index:05}.rs"),
                        &rust_first_party_source(index, index % spec.dependency_modules),
                    );
                }
                hash_qualification_record(&mut digest, "dep/Cargo.toml", dep_manifest);
                hash_qualification_record(
                    &mut digest,
                    "dep/src/lib.rs",
                    &rust_dependency_root_source(spec.dependency_modules),
                );
                for index in 0..spec.dependency_modules {
                    hash_qualification_record(
                        &mut digest,
                        &format!("dep/src/module_{index:05}.rs"),
                        &rust_dependency_source(index),
                    );
                }
                hash_qualification_record(
                    &mut digest,
                    "<invalid-replay>",
                    &rust_app_root_source(spec.first_party_files, true),
                );
                hash_qualification_record(
                    &mut digest,
                    "<valid-replay>",
                    &rust_app_root_source(spec.first_party_files, false),
                );
            }
        }
    }
    format!("{:x}", digest.finalize())
}

fn hash_native_spec(digest: &mut Sha256, case: NativeWorldCase, spec: NativeFixtureSpec) {
    digest.update(case.id().as_bytes());
    digest.update((spec.first_party_files as u64).to_le_bytes());
    digest.update((spec.dependency_modules as u64).to_le_bytes());
    digest.update((spec.stress_workers as u64).to_le_bytes());
    digest.update((spec.stress_replays_per_worker as u64).to_le_bytes());
}

fn hash_qualification_record(digest: &mut Sha256, path: &str, contents: &str) {
    digest.update((path.len() as u64).to_le_bytes());
    digest.update(path.as_bytes());
    digest.update((contents.len() as u64).to_le_bytes());
    digest.update(contents.as_bytes());
}

#[cfg(unix)]
fn process_peak_resident_bytes() -> Result<u64> {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the provided rusage on a zero return code.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return Err(std::io::Error::last_os_error())
            .context("failed to read process peak resident memory");
    }
    // SAFETY: the successful call above initialized usage.
    let raw = unsafe { usage.assume_init() }.ru_maxrss;
    let raw = u64::try_from(raw).context("process peak resident memory was negative")?;
    #[cfg(target_vendor = "apple")]
    let bytes = raw;
    #[cfg(not(target_vendor = "apple"))]
    let bytes = raw.saturating_mul(1024);
    Ok(bytes)
}

#[cfg(not(unix))]
fn process_peak_resident_bytes() -> Result<u64> {
    bail!("native-world peak resident qualification is currently supported only on Unix hosts")
}

#[cfg(target_vendor = "apple")]
fn process_current_resident_bytes() -> Result<u64> {
    let mut usage = MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: proc_pid_rusage initializes the versioned structure on a zero return code.
    let result = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast(),
        )
    };
    if result != 0 {
        return Err(std::io::Error::last_os_error())
            .context("failed to read current process resident memory");
    }
    // SAFETY: the successful call above initialized usage.
    Ok(unsafe { usage.assume_init() }.ri_resident_size)
}

#[cfg(all(unix, not(target_vendor = "apple"), target_os = "linux"))]
fn process_current_resident_bytes() -> Result<u64> {
    let statm = fs::read_to_string("/proc/self/statm")
        .context("failed to read current process resident memory")?;
    let resident_pages = statm
        .split_whitespace()
        .nth(1)
        .context("process statm has no resident-page count")?
        .parse::<u64>()
        .context("process statm resident-page count is invalid")?;
    // SAFETY: sysconf is a read-only process query.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size = u64::try_from(page_size).context("process page size was invalid")?;
    resident_pages
        .checked_mul(page_size)
        .context("current process resident memory overflowed")
}

#[cfg(any(
    not(unix),
    all(unix, not(target_vendor = "apple"), not(target_os = "linux"))
))]
fn process_current_resident_bytes() -> Result<u64> {
    bail!("native-world current resident qualification is supported only on macOS and Linux")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualification_profile_is_stable_and_bounded() {
        assert_eq!(
            qualification_profile_sha256(NativeWorldLanguage::Python),
            "a6093b4fa3f4b762d432291226aee27c111afae72745aa0dede09a64ee0ac0bc"
        );
        assert_eq!(
            qualification_profile_sha256(NativeWorldLanguage::Rust),
            "738a7ceef37d5c020f4ec89b67d306448ff01cd8aba9fabd44da98c1baeddd42"
        );
        assert_eq!(
            NativeWorldCase::Large.python_spec().first_party_files,
            10_000
        );
        assert_eq!(
            NativeWorldCase::Large.python_spec().dependency_modules,
            5_000
        );
        assert_eq!(NativeWorldCase::Large.rust_spec().first_party_files, 2_048);
        assert_eq!(NativeWorldCase::Large.rust_spec().dependency_modules, 1_024);
    }

    #[cfg(unix)]
    #[test]
    fn tiny_python_case_crosses_every_production_lifecycle_barrier() {
        let isolated_test = "native_qualification::tests::isolated_tiny_python_case_crosses_every_production_lifecycle_barrier";
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", isolated_test, "--nocapture"])
            .status()
            .unwrap();
        assert!(
            status.success(),
            "process-isolated native Python qualification test failed"
        );
    }

    #[cfg(unix)]
    #[test]
    #[ignore = "invoked in a dedicated process by the production lifecycle barrier test"]
    fn isolated_tiny_python_case_crosses_every_production_lifecycle_barrier() {
        let report = qualify_python_case(NativeWorldCase::Tiny).unwrap();
        assert_eq!(report.first_party_files, 4);
        assert_eq!(report.dependency_files, 7);
        assert!(report.primed_queries >= 9);
        assert!(report.peak_resident_bytes > 0);
        validate_case(
            NativeWorldLanguage::Python,
            &report,
            NativeWorldQualificationLimits {
                max_cold_millis: PYTHON_DEFAULT_MAX_COLD_MILLIS,
                max_warm_millis: DEFAULT_MAX_WARM_MILLIS,
                max_replay_millis: PYTHON_DEFAULT_MAX_REPLAY_MILLIS,
                max_stress_millis: PYTHON_DEFAULT_MAX_STRESS_MILLIS,
                max_peak_resident_bytes: PYTHON_DEFAULT_MAX_PEAK_RESIDENT_BYTES,
                max_retained_growth_bytes: DEFAULT_MAX_RETAINED_GROWTH_BYTES,
            },
        )
        .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn tiny_rust_case_crosses_every_production_lifecycle_barrier() {
        let isolated_test = "native_qualification::tests::isolated_tiny_rust_case_crosses_every_production_lifecycle_barrier";
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", isolated_test, "--nocapture"])
            .status()
            .unwrap();
        assert!(
            status.success(),
            "process-isolated native Rust qualification test failed"
        );
    }

    #[cfg(unix)]
    #[test]
    #[ignore = "invoked in a dedicated process by the production lifecycle barrier test"]
    fn isolated_tiny_rust_case_crosses_every_production_lifecycle_barrier() {
        let report = qualify_rust_case(NativeWorldCase::Tiny).unwrap();
        assert_eq!(report.first_party_files, 6);
        assert_eq!(report.dependency_files, 6);
        assert_eq!(report.target_count, 2);
        assert_eq!(report.semantic_profile, "exact-rust-v2");
        assert!(report.peak_resident_bytes > 0);
        validate_case(
            NativeWorldLanguage::Rust,
            &report,
            NativeWorldQualificationLimits {
                max_cold_millis: RUST_DEFAULT_MAX_COLD_MILLIS,
                max_warm_millis: DEFAULT_MAX_WARM_MILLIS,
                max_replay_millis: RUST_DEFAULT_MAX_REPLAY_MILLIS,
                max_stress_millis: RUST_DEFAULT_MAX_STRESS_MILLIS,
                max_peak_resident_bytes: RUST_DEFAULT_MAX_PEAK_RESIDENT_BYTES,
                max_retained_growth_bytes: DEFAULT_MAX_RETAINED_GROWTH_BYTES,
            },
        )
        .unwrap();
    }
}
