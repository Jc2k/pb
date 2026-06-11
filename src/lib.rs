use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use futures::{StreamExt, stream};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{ACCEPT, CONTENT_LENGTH, RANGE};
use reqwest::StatusCode;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tokio::time::{Duration, sleep};

use crate::environment::{EnvironmentConfig, EnvironmentMode};

pub mod agent_core;
pub mod cli_ui;
pub mod container;
pub mod environment;
pub mod events;
pub mod init;
pub mod service;
pub mod web;

pub const DEFAULT_MODEL: &str =
    "hf://unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M.gguf";
const OLLAMA_REGISTRY: &str = "https://registry.ollama.ai";
const HF_ENDPOINT: &str = "https://huggingface.co";
const PROGRESS_BAR_WIDTH: usize = 40;
const PROGRESS_TICK_MS: u64 = 120;
pub const DEFAULT_AGENT_MAX_STEPS: usize = 12;
pub const DEFAULT_AGENT_MAX_TOKENS: i32 = 384;
const GPU_FULL_OFFLOAD: u32 = 999;

#[derive(Parser, Debug)]
#[command(name = "pb", about = "A local coding agent CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Self-management commands
    #[command(name = "self")]
    SelfCmd {
        #[command(subcommand)]
        command: SelfCommand,
    },
    /// Pull model blobs from the Ollama-compatible registry or Hugging Face (hf://owner/repo)
    Pull(PullArgs),
    /// Run a local in-process coding agent and stream progress
    Agent(AgentArgs),
    /// Start the web UI server
    Serve(ServeArgs),
    /// Manage the per-project container environment for sandboxed task execution
    #[command(name = "env")]
    Env {
        #[command(subcommand)]
        command: EnvCommand,
    },
    /// Manage the pb serve launchd service (macOS)
    #[command(name = "service")]
    Service {
        #[command(subcommand)]
        command: ServiceCommand,
    },
    /// Inspect a project and configure it for use with pb
    Init(InitArgs),
}

#[derive(Subcommand, Debug)]
pub enum SelfCommand {
    /// Update pb from the latest GitHub release
    Update,
}

#[derive(Subcommand, Debug)]
pub enum EnvCommand {
    /// Pull a ready-made dev-environment image and save it as the project environment
    Pull(EnvPullArgs),
    /// Build a dev-environment image from a Dockerfile and save it as the project environment
    Build(EnvBuildArgs),
    /// Verify the configured environment by creating a test container and running init commands
    Start(EnvWorkdirArgs),
    /// Show the current project environment configuration
    Status(EnvWorkdirArgs),
}

#[derive(Subcommand, Debug)]
pub enum ServiceCommand {
    /// Write a LaunchAgent plist and load it so pb serve runs on login
    Enable(ServeArgs),
    /// Unload the LaunchAgent and remove the plist
    Disable,
    /// Start the pb serve service immediately
    Start,
    /// Stop the pb serve service
    Stop,
}

#[derive(Args, Debug)]
pub struct EnvPullArgs {
    /// Container image reference to pull (e.g. ghcr.io/myorg/dev:latest)
    pub image: String,

    /// Shell commands to run inside the container after creation (may be repeated)
    #[arg(long = "init", value_name = "CMD")]
    pub init_commands: Vec<String>,

    /// Project root; defaults to the nearest git repository root
    #[arg(long)]
    pub workdir: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct EnvBuildArgs {
    /// Path to the Dockerfile
    #[arg(long, default_value = "Dockerfile")]
    pub dockerfile: PathBuf,

    /// Tag for the built image (e.g. pb-dev:latest)
    #[arg(long, default_value = "pb-dev:latest")]
    pub tag: String,

    /// Shell commands to run inside the container after creation (may be repeated)
    #[arg(long = "init", value_name = "CMD")]
    pub init_commands: Vec<String>,

    /// Project root; defaults to the nearest git repository root
    #[arg(long)]
    pub workdir: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct EnvWorkdirArgs {
    /// Project root; defaults to the nearest git repository root
    #[arg(long)]
    pub workdir: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct InitArgs {
    /// Project root; defaults to the nearest git repository root
    #[arg(long)]
    pub workdir: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct PullArgs {
    /// Model to pull: Ollama library name (e.g. qwen3-coder-next) or Hugging Face URI
    /// (e.g. hf://unsloth/Qwen3-Coder-Next-GGUF or hf://owner/repo/filename.gguf)
    #[arg(default_value = DEFAULT_MODEL)]
    pub model: String,

    /// Number of blobs to process in each recovery-safe batch
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    /// Maximum parallel blob downloads within each batch
    #[arg(long, default_value_t = default_parallelism())]
    pub parallel: usize,

    /// Number of retries per blob download
    #[arg(long, default_value_t = 4)]
    pub retries: u32,

    /// Output directory for downloaded blobs
    #[arg(long)]
    pub out_dir: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
pub struct AgentArgs {
    /// Task to execute
    pub task: String,

    /// Model identifier: Ollama name (e.g. qwen3-coder-next) or Hugging Face URI
    /// (e.g. hf://unsloth/Qwen3-Coder-Next-GGUF); looked up in the pull cache
    #[arg(long, default_value = DEFAULT_MODEL)]
    pub model: String,

    /// Directory containing pulled model blobs; defaults to the XDG data directory
    #[arg(long)]
    pub model_dir: Option<PathBuf>,

    /// Working directory where tools can read/search/edit
    #[arg(long)]
    pub workdir: Option<PathBuf>,

    /// Continue work on an existing branch
    #[arg(long)]
    pub branch: Option<String>,

    /// Maximum number of think/tool iterations
    #[arg(long, default_value_t = DEFAULT_AGENT_MAX_STEPS)]
    pub max_steps: usize,

    /// Maximum new tokens per model turn
    #[arg(long, default_value_t = DEFAULT_AGENT_MAX_TOKENS)]
    pub max_tokens: i32,

    /// Context size
    #[arg(long, default_value_t = 8192)]
    pub ctx_size: u32,

    /// Number of CPU threads for decoding
    #[arg(long)]
    pub threads: Option<i32>,

    /// Number of CPU threads for prompt processing
    #[arg(long)]
    pub threads_batch: Option<i32>,

    /// Number of transformer layers to offload to GPU
    #[arg(long, default_value_t = default_gpu_layers())]
    pub gpu_layers: u32,

    /// Temperature
    #[arg(long, default_value_t = 0.2)]
    pub temperature: f32,

    /// Top-k for sampling
    #[arg(long, default_value_t = 40)]
    pub top_k: i32,

    /// RNG seed
    #[arg(long, default_value_t = 1337)]
    pub seed: u32,
}

#[derive(Args, Debug, Clone)]
pub struct ServeArgs {
    /// Bind host
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Bind port
    #[arg(long, default_value_t = 8311)]
    pub port: u16,

    /// Default model for API sessions
    #[arg(long, default_value = DEFAULT_MODEL)]
    pub model: String,

    /// Directory containing pulled model blobs
    #[arg(long)]
    pub model_dir: Option<PathBuf>,

    /// Working directory where tools can read/search/edit
    #[arg(long)]
    pub workdir: Option<PathBuf>,

    /// Default max steps per run
    #[arg(long, default_value_t = DEFAULT_AGENT_MAX_STEPS)]
    pub max_steps: usize,

    /// Default max new tokens per model turn
    #[arg(long, default_value_t = DEFAULT_AGENT_MAX_TOKENS)]
    pub max_tokens: i32,

    /// Default context size
    #[arg(long, default_value_t = 8192)]
    pub ctx_size: u32,

    /// Default number of CPU threads for decoding
    #[arg(long)]
    pub threads: Option<i32>,

    /// Default number of CPU threads for prompt processing
    #[arg(long)]
    pub threads_batch: Option<i32>,

    /// Default number of transformer layers to offload to GPU
    #[arg(long, default_value_t = default_gpu_layers())]
    pub gpu_layers: u32,

    /// Default temperature
    #[arg(long, default_value_t = 0.2)]
    pub temperature: f32,

    /// Default top-k for sampling
    #[arg(long, default_value_t = 40)]
    pub top_k: i32,

    /// Default RNG seed
    #[arg(long, default_value_t = 1337)]
    pub seed: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct ManifestDescriptor {
    digest: String,
    size: u64,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    config: ManifestDescriptor,
    layers: Vec<ManifestDescriptor>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    rfilename: String,
    size: Option<u64>,
    lfs: Option<HfLfs>,
}

#[derive(Debug, Deserialize)]
struct HfLfs {
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HfModelInfo {
    siblings: Vec<HfSibling>,
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::SelfCmd { command } => match command {
            SelfCommand::Update => run_self_update(),
        },
        Commands::Pull(args) => pull_model(&args).await,
        Commands::Agent(args) => {
            let request = to_agent_request(args.clone());
            let models_root = args.model_dir.clone().unwrap_or_else(default_models_dir);
            cli_ui::run_agent_cli(request, &models_root).await
        }
        Commands::Serve(args) => {
            let defaults = agent_core::AgentRequest {
                task: String::new(),
                model: args.model.clone(),
                model_dir: args.model_dir.clone(),
                workdir: args.workdir.clone(),
                branch: None,
                max_steps: args.max_steps,
                max_tokens: args.max_tokens,
                ctx_size: args.ctx_size,
                threads: args.threads,
                threads_batch: args.threads_batch,
                gpu_layers: args.gpu_layers,
                temperature: args.temperature,
                top_k: args.top_k,
                seed: args.seed,
                environment: None,
            };
            web::run_server(
                web::ServeArgs {
                    host: args.host,
                    port: args.port,
                },
                defaults,
            )
            .await
        }
        Commands::Env { command } => run_env_command(command),
        Commands::Service { command } => run_service_command(command),
        Commands::Init(args) => init::run_init(args.workdir),
    }
}

fn to_agent_request(args: AgentArgs) -> agent_core::AgentRequest {
    agent_core::AgentRequest {
        task: args.task,
        model: args.model,
        model_dir: args.model_dir,
        workdir: args.workdir,
        branch: args.branch,
        max_steps: args.max_steps,
        max_tokens: args.max_tokens,
        ctx_size: args.ctx_size,
        threads: args.threads,
        threads_batch: args.threads_batch,
        gpu_layers: args.gpu_layers,
        temperature: args.temperature,
        top_k: args.top_k,
        seed: args.seed,
        environment: None,
    }
}

fn run_self_update() -> Result<()> {
    let target = release_target();
    let status = self_update::backends::github::Update::configure()
        .repo_owner("Jc2k")
        .repo_name("pb")
        .bin_name("pb")
        .target(&target)
        .show_download_progress(true)
        .current_version(self_update::cargo_crate_version!())
        .build()
        .context("failed to build self-update configuration")?
        .update()
        .context("self-update failed")?;

    println!("Updated to {}", status.version());
    Ok(())
}

fn run_env_command(command: EnvCommand) -> Result<()> {
    match command {
        EnvCommand::Pull(args) => env_pull(args),
        EnvCommand::Build(args) => env_build(args),
        EnvCommand::Start(args) => env_start(args),
        EnvCommand::Status(args) => env_status(args),
    }
}

fn run_service_command(command: ServiceCommand) -> Result<()> {
    match command {
        ServiceCommand::Enable(args) => service::enable(&args),
        ServiceCommand::Disable => service::disable(),
        ServiceCommand::Start => service::start(),
        ServiceCommand::Stop => service::stop(),
    }
}

/// Resolve the project root from an optional `--workdir` flag.
/// Walks up from the given directory (or CWD) to the nearest `.git` ancestor.
fn resolve_env_root(workdir: Option<PathBuf>) -> Result<PathBuf> {
    let start = workdir
        .map(|p| p.canonicalize().context("failed to resolve --workdir"))
        .transpose()?
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    Ok(agent_core::find_git_root(&start).unwrap_or(start))
}

fn env_pull(args: EnvPullArgs) -> Result<()> {
    let root = resolve_env_root(args.workdir)?;
    let runtime =
        container::detect_runtime().context("no container runtime found; install docker, podman, or apple/container")?;
    println!("Pulling image {}…", args.image);
    runtime.pull(&args.image)?;
    let config = EnvironmentConfig {
        mode: EnvironmentMode::Pull,
        image: args.image.clone(),
        init_commands: args.init_commands,
        dockerfile: None,
    };
    config.save(&root)?;
    println!(
        "Environment saved to {}",
        root.join(".pb").join("environment.toml").display()
    );
    Ok(())
}

fn env_build(args: EnvBuildArgs) -> Result<()> {
    let root = resolve_env_root(args.workdir)?;
    let dockerfile = if args.dockerfile.is_absolute() {
        args.dockerfile.clone()
    } else {
        root.join(&args.dockerfile)
    };
    if !dockerfile.exists() {
        bail!("Dockerfile not found: {}", dockerfile.display());
    }
    let runtime =
        container::detect_runtime().context("no container runtime found; install docker, podman, or apple/container")?;
    println!("Building image {} from {}…", args.tag, dockerfile.display());
    runtime.build(&dockerfile, &args.tag)?;
    let config = EnvironmentConfig {
        mode: EnvironmentMode::Build,
        image: args.tag.clone(),
        init_commands: args.init_commands,
        dockerfile: Some(args.dockerfile),
    };
    config.save(&root)?;
    println!(
        "Environment saved to {}",
        root.join(".pb").join("environment.toml").display()
    );
    Ok(())
}

fn env_start(args: EnvWorkdirArgs) -> Result<()> {
    let root = resolve_env_root(args.workdir)?;
    let config = EnvironmentConfig::load(&root)?
        .context("no environment configured; run `pb env pull` or `pb env build` first")?;
    let runtime =
        container::detect_runtime().context("no container runtime found; install docker, podman, or apple/container")?;
    println!("Creating test container from {}…", config.image);
    let container_id = runtime.create(&config.image, &root)?;
    println!("Container {} started", container_id);
    for cmd in &config.init_commands {
        println!("Running init command: {cmd}");
        let output = runtime.exec(&container_id, cmd)?;
        if !output.is_empty() {
            println!("{output}");
        }
    }
    println!("Removing test container…");
    runtime.remove(&container_id)?;
    println!("Environment verified successfully.");
    Ok(())
}

fn env_status(args: EnvWorkdirArgs) -> Result<()> {
    let root = resolve_env_root(args.workdir)?;
    match EnvironmentConfig::load(&root)? {
        None => println!(
            "No environment configured at {}\nRun `pb env pull <image>` or `pb env build` to set one up.",
            root.join(".pb").join("environment.toml").display()
        ),
        Some(config) => {
            let mode = match config.mode {
                EnvironmentMode::Pull => "pull",
                EnvironmentMode::Build => "build",
            };
            println!("mode:  {mode}");
            println!("image: {}", config.image);
            if let Some(df) = &config.dockerfile {
                println!("dockerfile: {}", df.display());
            }
            if config.init_commands.is_empty() {
                println!("init_commands: (none)");
            } else {
                println!("init_commands:");
                for cmd in &config.init_commands {
                    println!("  - {cmd}");
                }
            }
        }
    }
    Ok(())
}

pub async fn pull_model(args: &PullArgs) -> Result<()> {
    if args.batch_size == 0 {
        bail!("batch-size must be greater than 0");
    }
    if args.parallel == 0 {
        bail!("parallel must be greater than 0");
    }

    let output_root = args.out_dir.clone().unwrap_or_else(default_models_dir);
    tokio::fs::create_dir_all(&output_root)
        .await
        .with_context(|| {
            format!(
                "failed to create output directory {}",
                output_root.display()
            )
        })?;

    let client = reqwest::Client::builder()
        .user_agent("pb/0.1.0")
        .build()
        .context("failed to build HTTP client")?;

    if args.model.starts_with("hf://") {
        pull_from_hf(
            &client,
            &args.model,
            &output_root,
            args.parallel,
            args.retries,
        )
        .await
    } else {
        pull_from_ollama(&client, &args.model, &output_root, args.batch_size, args.parallel, args.retries).await
    }
}

/// Convert a model URI to a filesystem-safe cache directory name.
///
/// `hf://owner/repo` and `hf://owner/repo/filename.gguf` both map to `owner_repo`.
/// Plain Ollama model names are returned unchanged.
pub fn cache_dir_name(model: &str) -> String {
    if let Some(path) = model.strip_prefix("hf://") {
        let mut parts = path.splitn(3, '/');
        let owner = parts.next().unwrap_or(path);
        let repo = parts.next().unwrap_or("");
        if repo.is_empty() {
            owner.to_owned()
        } else {
            format!("{owner}_{repo}")
        }
    } else {
        model.to_owned()
    }
}

fn parse_hf_uri(uri: &str) -> Option<(String, String, Option<String>)> {
    let path = uri.strip_prefix("hf://")?;
    let mut parts = path.splitn(3, '/');
    let owner = parts.next()?.to_owned();
    let repo = parts.next()?.to_owned();
    let filename = parts.next().map(str::to_owned);
    Some((owner, repo, filename))
}

/// Given a list of GGUF filenames, return those belonging to the best available quantization.
///
/// Prefers `Q4_K_M` → `Q4_K_S` → `Q5_K_M` → `Q4_0` → `Q8_0`; falls back to the full list.
fn select_hf_gguf_files(files: &[String]) -> Vec<String> {
    const PREFS: &[&str] = &["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q8_0"];
    for quant in PREFS {
        let matches: Vec<String> = files.iter().filter(|f| f.contains(quant)).cloned().collect();
        if !matches.is_empty() {
            return matches;
        }
    }
    files.to_vec()
}

async fn list_hf_gguf_files(
    client: &reqwest::Client,
    owner: &str,
    repo: &str,
) -> Result<Vec<HfSibling>> {
    let url = format!("{HF_ENDPOINT}/api/models/{owner}/{repo}");
    let info: HfModelInfo = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to query {owner}/{repo} on Hugging Face"))?
        .error_for_status()
        .with_context(|| format!("Hugging Face API returned an error for {owner}/{repo}"))?
        .json()
        .await
        .with_context(|| format!("failed to parse Hugging Face model info for {owner}/{repo}"))?;

    Ok(info
        .siblings
        .into_iter()
        .filter(|s| s.rfilename.ends_with(".gguf"))
        .collect())
}

/// Return the size for a Hugging Face sibling, preferring top-level `size`
/// and falling back to `lfs.size` when needed.
fn hf_sibling_size(sibling: &HfSibling) -> Option<u64> {
    sibling.size.or_else(|| sibling.lfs.as_ref().and_then(|lfs| lfs.size))
}

/// Issue a HEAD request and return the `Content-Length` value, if available.
async fn fetch_content_length(client: &reqwest::Client, url: &str) -> Option<u64> {
    let resp = client.head(url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.headers()
        .get(CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()
}

/// Build a temp file path by appending `.tmp` to the target filename.
/// Falls back to `download.tmp` when no filename component exists.
fn download_tmp_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .map(|n| format!("{}.tmp", n.to_string_lossy()))
        .unwrap_or_else(|| "download.tmp".to_owned());
    path.with_file_name(file_name)
}

/// Return already-downloaded bytes for a target path.
/// Prefers a complete destination file, then a valid partial `.tmp` file.
fn existing_bytes(path: &Path, expected_size: Option<u64>) -> Result<u64> {
    let Some(expected_size) = expected_size else {
        return Ok(0);
    };

    if path.exists() {
        let size = std::fs::metadata(path)
            .with_context(|| format!("failed to stat {}", path.display()))?
            .len();
        if size == expected_size {
            return Ok(size);
        }
    }

    let tmp_path = download_tmp_path(path);
    if tmp_path.exists() {
        let size = std::fs::metadata(&tmp_path)
            .with_context(|| format!("failed to stat {}", tmp_path.display()))?
            .len();
        if size < expected_size {
            return Ok(size);
        }
    }

    Ok(0)
}

/// Build a byte-oriented progress bar with elapsed time, percentage, and ETA.
/// Validates that the starting position does not exceed the total.
/// When `total` is 0 (size unknown), uses a spinner with a bytes-only display.
fn build_progress_bar(total: u64, initial: u64) -> Result<ProgressBar> {
    if total > 0 && initial > total {
        bail!("initial progress {initial} is greater than total {total}");
    }
    let (pb, template) = if total > 0 {
        let pb = ProgressBar::new(total);
        let template = format!(
            "{{spinner:.green}} [{{elapsed_precise}}] [{{bar:{PROGRESS_BAR_WIDTH}.cyan/blue}}] {{bytes}}/{{total_bytes}} ({{percent}}%) ETA {{eta_precise}}"
        );
        (pb, template)
    } else {
        let pb = ProgressBar::new_spinner();
        let template = format!("{{spinner:.green}} [{{elapsed_precise}}] {{bytes}} downloaded");
        (pb, template)
    };
    let style = ProgressStyle::with_template(&template)
        .context("failed to configure progress bar style")?
        .progress_chars("=>-");
    pb.set_style(style);
    if total > 0 {
        pb.set_position(initial);
    }
    pb.enable_steady_tick(Duration::from_millis(PROGRESS_TICK_MS));
    Ok(pb)
}

async fn pull_from_hf(
    client: &reqwest::Client,
    hf_uri: &str,
    output_root: &Path,
    parallel: usize,
    retries: u32,
) -> Result<()> {
    let (owner, repo, explicit_filename) = parse_hf_uri(hf_uri)
        .with_context(|| format!("invalid Hugging Face URI: {hf_uri}"))?;

    let siblings = list_hf_gguf_files(client, &owner, &repo).await?;
    if siblings.is_empty() {
        bail!("no GGUF files found in {owner}/{repo} on Hugging Face");
    }

    let files: Vec<(String, Option<u64>)> = if let Some(f) = explicit_filename {
        let sibling = siblings
            .iter()
            .find(|s| s.rfilename == f)
            .with_context(|| format!("GGUF file {f} not found in {owner}/{repo} on Hugging Face"))?;
        let size = match hf_sibling_size(sibling) {
            s @ Some(_) => s,
            None => {
                let url = format!("{HF_ENDPOINT}/{owner}/{repo}/resolve/main/{f}");
                fetch_content_length(client, &url).await
            }
        };
        vec![(f, size)]
    } else {
        let all_names: Vec<String> = siblings.iter().map(|s| s.rfilename.clone()).collect();
        let selected = select_hf_gguf_files(&all_names);
        let mut result = Vec::new();
        for filename in selected {
            let sibling = siblings
                .iter()
                .find(|s| s.rfilename == filename)
                .with_context(|| format!("file metadata missing for {filename}"))?;
            let size = match hf_sibling_size(sibling) {
                s @ Some(_) => s,
                None => {
                    let url = format!("{HF_ENDPOINT}/{owner}/{repo}/resolve/main/{filename}");
                    fetch_content_length(client, &url).await
                }
            };
            result.push((filename, size));
        }
        result
    };

    let cache_dir = output_root.join(cache_dir_name(hf_uri));
    tokio::fs::create_dir_all(&cache_dir).await.with_context(|| {
        format!("failed to create cache directory {}", cache_dir.display())
    })?;

    let total_bytes: u64 = files.iter().map(|(_, size)| size.unwrap_or(0)).sum();
    let initial_bytes = files.iter().try_fold(0u64, |acc, (filename, size)| {
        existing_bytes(&cache_dir.join(filename), *size).map(|n| acc + n)
    })?;
    let progress = build_progress_bar(total_bytes, initial_bytes)?;

    let total_files = files.len();
    let tasks = stream::iter(files.into_iter().map(|(filename, size)| {
        let client = client.clone();
        let url = format!("{HF_ENDPOINT}/{owner}/{repo}/resolve/main/{filename}");
        let dest = cache_dir.join(&filename);
        let progress = progress.clone();
        async move {
            download_file_with_retry(&client, &url, &dest, size, &progress, &filename, retries).await
        }
    }))
    .buffer_unordered(parallel)
    .collect::<Vec<_>>();

    let results = tasks.await;
    for result in results {
        result?;
    }

    progress.finish_with_message("download complete");
    println!(
        "Pull complete: {total_files} file(s) available in {}",
        cache_dir.display()
    );
    Ok(())
}

async fn download_file_with_retry(
    client: &reqwest::Client,
    url: &str,
    path: &Path,
    expected_size: Option<u64>,
    progress: &ProgressBar,
    label: &str,
    retries: u32,
) -> Result<()> {
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match download_file_with_resume(client, url, path, expected_size, progress).await {
            Ok(()) => return Ok(()),
            Err(err) if attempt <= retries => {
                eprintln!(
                    "Download of {label} failed (attempt {attempt}/{retries}): {err}"
                );
                sleep(Duration::from_millis(500 * attempt as u64)).await;
            }
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("download of {label} failed after {attempt} attempts")
                });
            }
        }
    }
}

/// Download a file to `path` with resume support via HTTP range requests.
/// If the server ignores ranges and returns a full `200 OK`, the partial
/// state is discarded and the transfer restarts from zero.
/// When `expected_size` is `None` the file size is unknown: an existing
/// destination file is assumed complete, resume is skipped, and the final
/// size validation is omitted.
async fn download_file_with_resume(
    client: &reqwest::Client,
    url: &str,
    path: &Path,
    expected_size: Option<u64>,
    progress: &ProgressBar,
) -> Result<()> {
    if let Some(size) = expected_size {
        if path.exists() {
            let metadata = std::fs::metadata(path)
                .with_context(|| format!("failed to stat {}", path.display()))?;
            if metadata.len() == size {
                return Ok(());
            }
            std::fs::remove_file(path)
                .with_context(|| format!("failed to remove stale file {}", path.display()))?;
        }
    } else if path.exists() {
        // Size unknown: cannot verify the file is intact, but assume it is
        // complete to avoid re-downloading a potentially large file.
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let tmp_path = download_tmp_path(path);
    let mut resume_from = if let Some(size) = expected_size {
        let partial = if tmp_path.exists() {
            std::fs::metadata(&tmp_path)
                .with_context(|| format!("failed to stat {}", tmp_path.display()))?
                .len()
        } else {
            0
        };
        if partial > size {
            std::fs::remove_file(&tmp_path)
                .with_context(|| format!("failed to remove stale temp file {}", tmp_path.display()))?;
            0
        } else if partial == size {
            tokio::fs::rename(&tmp_path, path).await.with_context(|| {
                format!(
                    "failed to rename {} to {}",
                    tmp_path.display(),
                    path.display()
                )
            })?;
            return Ok(());
        } else {
            partial
        }
    } else {
        // No resume when size is unknown: discard any stale partial download.
        if tmp_path.exists() {
            std::fs::remove_file(&tmp_path)
                .with_context(|| format!("failed to remove stale temp file {}", tmp_path.display()))?;
        }
        0
    };

    let mut request = client.get(url);
    if resume_from > 0 {
        request = request.header(RANGE, format!("bytes={resume_from}-"));
    }

    let response = request
        .send()
        .await
        .with_context(|| format!("failed to request {url}"))?
        .error_for_status()
        .with_context(|| format!("download request failed for {url}"))?;

    // Some servers ignore `Range` and return full content with `200 OK`.
    // In that case we restart from zero and remove previously-accounted partial progress.
    if resume_from > 0 && response.status() == StatusCode::OK {
        progress.dec(resume_from);
        resume_from = 0;
    }

    let mut file = if resume_from > 0 {
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(&tmp_path)
            .await
            .with_context(|| format!("failed to open {}", tmp_path.display()))?
    } else {
        tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
            .await
            .with_context(|| format!("failed to create {}", tmp_path.display()))?
    };

    let mut bytes_written = resume_from;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("failed reading stream from {url}"))?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("failed writing to {}", tmp_path.display()))?;
        let chunk_len = chunk.len() as u64;
        bytes_written += chunk_len;
        progress.inc(chunk_len);
    }

    file.flush().await.context("failed to flush file")?;
    drop(file);

    if let Some(size) = expected_size {
        if bytes_written != size {
            bail!(
                "size mismatch for {}: expected {size}, wrote {bytes_written}",
                path.display()
            );
        }
    }

    tokio::fs::rename(&tmp_path, path).await.with_context(|| {
        format!(
            "failed to rename {} to {}",
            tmp_path.display(),
            path.display()
        )
    })?;

    Ok(())
}

async fn pull_from_ollama(
    client: &reqwest::Client,
    model: &str,
    output_root: &Path,
    batch_size: usize,
    parallel: usize,
    retries: u32,
) -> Result<()> {
    let manifest = fetch_manifest(client, model).await?;
    let descriptors = descriptors_from_manifest(manifest);

    let total_bytes: u64 = descriptors.iter().map(|d| d.size).sum();
    let initial_bytes = descriptors.iter().try_fold(0u64, |acc, descriptor| {
        existing_bytes(&blob_path(output_root, model, &descriptor.digest), Some(descriptor.size))
            .map(|n| acc + n)
    })?;
    let progress = build_progress_bar(total_bytes, initial_bytes)?;

    for batch in descriptors.chunks(batch_size) {
        let futures = stream::iter(batch.iter().cloned().map(|descriptor| {
            let client = client.clone();
            let model = model.to_owned();
            let output_root = output_root.to_owned();
            let progress = progress.clone();
            async move {
                download_blob_with_retry(
                    &client,
                    &model,
                    &output_root,
                    &descriptor,
                    &progress,
                    retries,
                )
                    .await
            }
        }))
        .buffer_unordered(parallel)
        .collect::<Vec<_>>();

        let results = futures.await;
        for result in results {
            result?;
        }
    }

    progress.finish_with_message("download complete");
    println!(
        "Pull complete: {} blobs available in {}",
        descriptors.len(),
        output_root.display()
    );
    Ok(())
}

async fn fetch_manifest(client: &reqwest::Client, model: &str) -> Result<Manifest> {
    let url = format!("{OLLAMA_REGISTRY}/v2/library/{model}/manifests/latest");
    let response = client
        .get(&url)
        .header(ACCEPT, "application/vnd.oci.image.manifest.v1+json")
        .send()
        .await
        .with_context(|| format!("failed to fetch manifest for model {model}"))?
        .error_for_status()
        .with_context(|| format!("manifest request failed for model {model}"))?;

    response
        .json::<Manifest>()
        .await
        .with_context(|| format!("failed to decode manifest for model {model}"))
}

fn descriptors_from_manifest(manifest: Manifest) -> Vec<ManifestDescriptor> {
    let mut descriptors = Vec::with_capacity(1 + manifest.layers.len());
    descriptors.push(manifest.config);
    descriptors.extend(manifest.layers);
    descriptors
}

async fn download_blob_with_retry(
    client: &reqwest::Client,
    model: &str,
    output_root: &Path,
    descriptor: &ManifestDescriptor,
    progress: &ProgressBar,
    retries: u32,
) -> Result<()> {
    let digest = descriptor.digest.clone();
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match download_blob(client, model, output_root, &digest, descriptor.size, progress).await {
            Ok(()) => return Ok(()),
            Err(err) if attempt <= retries => {
                eprintln!(
                    "Blob {} download failed (attempt {attempt}/{retries}): {err}",
                    digest
                );
                sleep(Duration::from_millis(500 * attempt as u64)).await;
            }
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("blob {} failed after {attempt} attempts", digest));
            }
        }
    }
}

async fn download_blob(
    client: &reqwest::Client,
    model: &str,
    output_root: &Path,
    digest: &str,
    expected_size: u64,
    progress: &ProgressBar,
) -> Result<()> {
    let path = blob_path(output_root, model, digest);
    let url = format!("{OLLAMA_REGISTRY}/v2/library/{model}/blobs/{digest}");
    download_file_with_resume(client, &url, &path, Some(expected_size), progress)
        .await
        .with_context(|| format!("blob request failed for {digest}"))
}

fn release_target() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match (os, arch) {
        ("macos", "aarch64") => "aarch64-apple-darwin".to_owned(),
        ("macos", "x86_64") => "x86_64-apple-darwin".to_owned(),
        ("linux", "aarch64") => "aarch64-unknown-linux-musl".to_owned(),
        ("linux", "x86_64") => "x86_64-unknown-linux-musl".to_owned(),
        ("windows", "x86_64") => "x86_64-pc-windows-msvc".to_owned(),
        _ => format!("{arch}-{os}"),
    }
}

fn default_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(32))
        .unwrap_or(8)
}

fn default_gpu_layers() -> u32 {
    if cfg!(target_os = "macos") {
        GPU_FULL_OFFLOAD
    } else {
        0
    }
}

fn default_data_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME")
        && !xdg.is_empty() {
            return PathBuf::from(xdg).join("pb");
        }
    if let Some(home) = dirs::home_dir() {
        return home.join(".local").join("share").join("pb");
    }
    PathBuf::from(".pb")
}

pub fn default_models_dir() -> PathBuf {
    default_data_dir().join("models")
}

fn blob_path(root: &Path, model: &str, digest: &str) -> PathBuf {
    let mut path = root.join(model);
    path.push(digest.replace(':', "_"));
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn descriptors_include_config_then_layers() {
        let manifest = Manifest {
            config: ManifestDescriptor {
                digest: "sha256:config".to_string(),
                size: 1,
            },
            layers: vec![
                ManifestDescriptor {
                    digest: "sha256:layer1".to_string(),
                    size: 2,
                },
                ManifestDescriptor {
                    digest: "sha256:layer2".to_string(),
                    size: 3,
                },
            ],
        };

        let descriptors = descriptors_from_manifest(manifest);
        assert_eq!(descriptors.len(), 3);
        assert_eq!(descriptors[0].digest, "sha256:config");
        assert_eq!(descriptors[1].digest, "sha256:layer1");
        assert_eq!(descriptors[2].digest, "sha256:layer2");
    }

    #[test]
    fn blob_path_sanitizes_digest() {
        let path = blob_path(Path::new("/tmp/output"), "qwen3-coder-next", "sha256:abcd");
        assert!(path.ends_with("qwen3-coder-next/sha256_abcd"));
    }

    #[test]
    fn default_parallelism_is_non_zero() {
        assert!(default_parallelism() > 0);
    }

    #[test]
    fn release_target_is_not_empty() {
        assert!(!release_target().is_empty());
    }

    #[test]
    fn default_model_pins_q4_k_m_quantization() {
        assert_eq!(
            parse_hf_uri(DEFAULT_MODEL),
            Some((
                "unsloth".to_owned(),
                "Qwen3-Coder-Next-GGUF".to_owned(),
                Some("Qwen3-Coder-Next-Q4_K_M.gguf".to_owned())
            ))
        );
    }

    #[test]
    fn cache_dir_name_passthrough_for_ollama() {
        assert_eq!(cache_dir_name("qwen3-coder-next"), "qwen3-coder-next");
    }

    #[test]
    fn cache_dir_name_hf_uri_without_filename() {
        assert_eq!(
            cache_dir_name("hf://unsloth/Qwen3-Coder-Next-GGUF"),
            "unsloth_Qwen3-Coder-Next-GGUF"
        );
    }

    #[test]
    fn cache_dir_name_hf_uri_with_filename() {
        assert_eq!(
            cache_dir_name("hf://unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M.gguf"),
            "unsloth_Qwen3-Coder-Next-GGUF"
        );
    }

    #[test]
    fn select_hf_gguf_files_prefers_q4_k_m() {
        let files = vec![
            "model-Q8_0.gguf".to_owned(),
            "model-Q4_K_M.gguf".to_owned(),
            "model-Q4_K_S.gguf".to_owned(),
        ];
        let selected = select_hf_gguf_files(&files);
        assert_eq!(selected, vec!["model-Q4_K_M.gguf"]);
    }

    #[test]
    fn select_hf_gguf_files_falls_back_to_all() {
        let files = vec!["model-IQ3_XS.gguf".to_owned(), "model-IQ4_XS.gguf".to_owned()];
        let selected = select_hf_gguf_files(&files);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn select_hf_gguf_files_returns_all_shards_of_quant() {
        let files = vec![
            "model-Q4_K_M-00001-of-00002.gguf".to_owned(),
            "model-Q4_K_M-00002-of-00002.gguf".to_owned(),
            "model-Q8_0.gguf".to_owned(),
        ];
        let selected = select_hf_gguf_files(&files);
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|f| f.contains("Q4_K_M")));
    }

    #[test]
    fn parse_hf_uri_without_filename() {
        let result = parse_hf_uri("hf://unsloth/Qwen3-Coder-Next-GGUF");
        assert_eq!(
            result,
            Some(("unsloth".to_owned(), "Qwen3-Coder-Next-GGUF".to_owned(), None))
        );
    }

    #[test]
    fn parse_hf_uri_with_filename() {
        let result = parse_hf_uri("hf://unsloth/Qwen3-Coder-Next-GGUF/model.gguf");
        assert_eq!(
            result,
            Some((
                "unsloth".to_owned(),
                "Qwen3-Coder-Next-GGUF".to_owned(),
                Some("model.gguf".to_owned())
            ))
        );
    }

    #[test]
    fn parse_hf_uri_returns_none_for_invalid() {
        assert!(parse_hf_uri("ollama://model").is_none());
        assert!(parse_hf_uri("hf://only-owner").is_none());
    }

    #[test]
    fn hf_sibling_size_prefers_top_level_size() {
        let sibling = HfSibling {
            rfilename: "model.gguf".to_owned(),
            size: Some(42),
            lfs: Some(HfLfs { size: Some(99) }),
        };
        assert_eq!(hf_sibling_size(&sibling), Some(42));
    }

    #[test]
    fn hf_sibling_size_falls_back_to_lfs_size() {
        let sibling = HfSibling {
            rfilename: "model.gguf".to_owned(),
            size: None,
            lfs: Some(HfLfs { size: Some(99) }),
        };
        assert_eq!(hf_sibling_size(&sibling), Some(99));
    }

    #[test]
    fn hf_sibling_size_returns_none_when_unknown() {
        let sibling = HfSibling {
            rfilename: "model.gguf".to_owned(),
            size: None,
            lfs: None,
        };
        assert_eq!(hf_sibling_size(&sibling), None);
    }

    #[test]
    fn download_tmp_path_adds_tmp_suffix() {
        let path = Path::new("/tmp/model.gguf");
        assert_eq!(download_tmp_path(path), PathBuf::from("/tmp/model.gguf.tmp"));
    }

    #[test]
    fn download_tmp_path_uses_default_when_no_filename() {
        let path = Path::new("/");
        assert_eq!(download_tmp_path(path), PathBuf::from("/download.tmp"));
    }

    #[test]
    fn existing_bytes_prefers_complete_destination_file() {
        let dir = tempdir().unwrap();
        let dest = dir.path().join("model.gguf");
        fs::write(&dest, vec![0u8; 16]).unwrap();
        assert_eq!(existing_bytes(&dest, Some(16)).unwrap(), 16);
    }

    #[test]
    fn existing_bytes_uses_partial_tmp_file() {
        let dir = tempdir().unwrap();
        let dest = dir.path().join("model.gguf");
        let tmp = download_tmp_path(&dest);
        fs::write(&tmp, vec![0u8; 8]).unwrap();
        assert_eq!(existing_bytes(&dest, Some(16)).unwrap(), 8);
    }

    #[test]
    fn existing_bytes_returns_zero_when_no_state() {
        let dir = tempdir().unwrap();
        let dest = dir.path().join("model.gguf");
        assert_eq!(existing_bytes(&dest, Some(16)).unwrap(), 0);
    }

    #[test]
    fn existing_bytes_ignores_oversized_tmp_file() {
        let dir = tempdir().unwrap();
        let dest = dir.path().join("model.gguf");
        let tmp = download_tmp_path(&dest);
        fs::write(&tmp, vec![0u8; 32]).unwrap();
        assert_eq!(existing_bytes(&dest, Some(16)).unwrap(), 0);
    }

    #[test]
    fn existing_bytes_returns_zero_when_size_unknown() {
        let dir = tempdir().unwrap();
        let dest = dir.path().join("model.gguf");
        // Even with a partial tmp file present, unknown size yields 0.
        let tmp = download_tmp_path(&dest);
        fs::write(&tmp, vec![0u8; 8]).unwrap();
        assert_eq!(existing_bytes(&dest, None).unwrap(), 0);
    }

    #[test]
    fn build_progress_bar_initializes_position() {
        let pb = build_progress_bar(100, 25).unwrap();
        assert_eq!(pb.length(), Some(100));
        assert_eq!(pb.position(), 25);
        pb.finish_and_clear();
    }

    #[test]
    fn build_progress_bar_uses_spinner_when_total_is_zero() {
        let pb = build_progress_bar(0, 0).unwrap();
        // Spinner has no fixed length.
        assert_eq!(pb.length(), None);
        // Incrementing should not panic.
        pb.inc(1024);
        pb.finish_and_clear();
    }
}
