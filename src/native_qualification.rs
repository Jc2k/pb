//! Process-isolated qualification for native language-world lifecycle and resource behavior.

use std::{
    fs,
    mem::MaybeUninit,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::control_layers::{PythonWorldQualificationObservation, qualify_python_world_fixture};

const NATIVE_WORLD_QUALIFICATION_VERSION: u32 = 1;
const DEFAULT_MAX_COLD_MILLIS: u64 = 60_000;
const DEFAULT_MAX_WARM_MILLIS: u64 = 20_000;
const DEFAULT_MAX_REPLAY_MILLIS: u64 = 20_000;
const DEFAULT_MAX_PEAK_RESIDENT_BYTES: u64 = 1_073_741_824;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum NativeWorldLanguage {
    Python,
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

    fn spec(self) -> PythonFixtureSpec {
        match self {
            Self::Tiny => PythonFixtureSpec {
                first_party_files: 4,
                dependency_modules: 4,
            },
            Self::Representative => PythonFixtureSpec {
                first_party_files: 1_024,
                dependency_modules: 512,
            },
            Self::Large => PythonFixtureSpec {
                first_party_files: 10_000,
                dependency_modules: 5_000,
            },
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct HarnessNativeWorldQualifyArgs {
    /// Native language implementation to qualify
    #[arg(long, value_enum, default_value_t = NativeWorldLanguage::Python)]
    pub(crate) language: NativeWorldLanguage,

    /// Maximum accepted cold preparation time for each matrix case
    #[arg(long, default_value_t = DEFAULT_MAX_COLD_MILLIS)]
    pub(crate) max_cold_millis: u64,

    /// Maximum accepted warm or exact process-cache request time for each matrix case
    #[arg(long, default_value_t = DEFAULT_MAX_WARM_MILLIS)]
    pub(crate) max_warm_millis: u64,

    /// Maximum accepted independent final replay time for each matrix case
    #[arg(long, default_value_t = DEFAULT_MAX_REPLAY_MILLIS)]
    pub(crate) max_replay_millis: u64,

    /// Maximum accepted process peak resident memory for each isolated matrix case
    #[arg(long, default_value_t = DEFAULT_MAX_PEAK_RESIDENT_BYTES)]
    pub(crate) max_peak_resident_bytes: u64,

    /// Internal process-isolated matrix arm
    #[arg(long, value_enum, hide = true)]
    pub(crate) case: Option<NativeWorldCase>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct NativeWorldQualificationLimits {
    max_cold_millis: u64,
    max_warm_millis: u64,
    max_replay_millis: u64,
    max_peak_resident_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct PythonFixtureSpec {
    first_party_files: usize,
    dependency_modules: usize,
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
    load_millis: u64,
    prime_millis: u64,
    primed_queries: u64,
    cold_millis: u64,
    warm_millis: u64,
    process_cache_hit_millis: u64,
    invalid_replay_millis: u64,
    valid_replay_millis: u64,
    baseline_peak_resident_bytes: u64,
    peak_resident_bytes: u64,
    incremental_peak_resident_bytes: u64,
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
    if args.language != NativeWorldLanguage::Python {
        bail!("the requested native language qualifier is not implemented");
    }
    if let Some(case) = args.case {
        let report = qualify_python_case(case)?;
        println!("{}", serde_json::to_string(&report)?);
        return Ok(());
    }

    let limits = NativeWorldQualificationLimits {
        max_cold_millis: args.max_cold_millis,
        max_warm_millis: args.max_warm_millis,
        max_replay_millis: args.max_replay_millis,
        max_peak_resident_bytes: args.max_peak_resident_bytes,
    };
    validate_limits(limits)?;
    let cases = NativeWorldCase::ALL
        .into_iter()
        .map(|case| run_isolated_python_case(case, limits))
        .collect::<Result<Vec<_>>>()?;
    for case in &cases {
        validate_case(case, limits)?;
    }
    let report = NativeWorldQualificationReport {
        version: NATIVE_WORLD_QUALIFICATION_VERSION,
        language: "python".to_string(),
        profile_sha256: qualification_profile_sha256(),
        limits,
        cases,
        passed: true,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn validate_limits(limits: NativeWorldQualificationLimits) -> Result<()> {
    if limits.max_cold_millis == 0
        || limits.max_warm_millis == 0
        || limits.max_replay_millis == 0
        || limits.max_peak_resident_bytes == 0
    {
        bail!("native-world qualification limits must all be non-zero");
    }
    Ok(())
}

fn run_isolated_python_case(
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
            "python",
            "--case",
            case.id(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "failed to start isolated Python native-world case {}",
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
                "isolated Python native-world case {} exceeded its {} ms aggregate deadline: {}",
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
            "isolated Python native-world case {} failed: {}",
            case.id(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    serde_json::from_slice(&output.stdout).with_context(|| {
        format!(
            "isolated Python native-world case {} returned an invalid report",
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
        .and_then(|total| total.checked_add(30_000))
        .context("native-world aggregate deadline overflowed")?;
    Ok(Duration::from_millis(millis))
}

fn qualify_python_case(case: NativeWorldCase) -> Result<NativeWorldCaseReport> {
    let fixture = build_python_fixture(case)?;
    let baseline_peak_resident_bytes = process_peak_resident_bytes()?;
    let observation = qualify_python_world_fixture(
        &fixture.root,
        "main.py",
        PYTHON_FIXTURE_INVALID,
        PYTHON_FIXTURE_VALID,
    )?;
    let peak_resident_bytes = process_peak_resident_bytes()?;
    Ok(case_report(
        case,
        fixture,
        observation,
        baseline_peak_resident_bytes,
        peak_resident_bytes,
    ))
}

fn case_report(
    case: NativeWorldCase,
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
        load_millis: observation.load_millis,
        prime_millis: observation.prime_millis,
        primed_queries: observation.primed_queries,
        cold_millis: observation.cold_millis,
        warm_millis: observation.warm_millis,
        process_cache_hit_millis: observation.process_cache_hit_millis,
        invalid_replay_millis: observation.invalid_replay_millis,
        valid_replay_millis: observation.valid_replay_millis,
        baseline_peak_resident_bytes,
        peak_resident_bytes,
        incremental_peak_resident_bytes: peak_resident_bytes
            .saturating_sub(baseline_peak_resident_bytes),
    }
}

fn validate_case(
    case: &NativeWorldCaseReport,
    limits: NativeWorldQualificationLimits,
) -> Result<()> {
    if case.cold_millis > limits.max_cold_millis {
        bail!(
            "Python native-world case {} exceeded cold budget: {} > {} ms",
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
                "Python native-world case {} exceeded {label} budget: {actual} > {} ms",
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
                "Python native-world case {} exceeded {label} budget: {actual} > {} ms",
                case.case,
                limits.max_replay_millis
            );
        }
    }
    if case.peak_resident_bytes > limits.max_peak_resident_bytes {
        bail!(
            "Python native-world case {} exceeded peak resident budget: {} > {} bytes",
            case.case,
            case.peak_resident_bytes,
            limits.max_peak_resident_bytes
        );
    }
    let minimum_queries = case
        .first_party_files
        .saturating_add(case.dependency_files.saturating_sub(2));
    if case.primed_queries < minimum_queries as u64 {
        bail!(
            "Python native-world case {} did not prime every source module before inference",
            case.case
        );
    }
    Ok(())
}

fn build_python_fixture(case: NativeWorldCase) -> Result<PythonFixture> {
    let spec = case.spec();
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

fn qualification_profile_sha256() -> String {
    let mut digest = Sha256::new();
    digest.update(b"pb-native-world-qualification-v1\0python\0ty_0.0.6\0");
    for case in NativeWorldCase::ALL {
        let spec = case.spec();
        digest.update(case.id().as_bytes());
        digest.update((spec.first_party_files as u64).to_le_bytes());
        digest.update((spec.dependency_modules as u64).to_le_bytes());
        hash_qualification_record(&mut digest, ".gitignore", PYTHON_FIXTURE_GITIGNORE);
        hash_qualification_record(&mut digest, ".venv/pyvenv.cfg", PYTHON_FIXTURE_ENVIRONMENT);
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
    format!("{:x}", digest.finalize())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualification_profile_is_stable_and_bounded() {
        assert_eq!(
            qualification_profile_sha256(),
            "aeeca87bd8032bede70a174dd124ee341de33beeed1739998c11d94b8134966e"
        );
        assert_eq!(NativeWorldCase::Large.spec().first_party_files, 10_000);
        assert_eq!(NativeWorldCase::Large.spec().dependency_modules, 5_000);
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
            &report,
            NativeWorldQualificationLimits {
                max_cold_millis: DEFAULT_MAX_COLD_MILLIS,
                max_warm_millis: DEFAULT_MAX_WARM_MILLIS,
                max_replay_millis: DEFAULT_MAX_REPLAY_MILLIS,
                max_peak_resident_bytes: DEFAULT_MAX_PEAK_RESIDENT_BYTES,
            },
        )
        .unwrap();
    }
}
