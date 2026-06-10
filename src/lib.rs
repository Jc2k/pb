use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use futures::{StreamExt, stream};
use reqwest::header::ACCEPT;
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

pub const DEFAULT_MODEL: &str = "qwen3-coder-next";
const OLLAMA_REGISTRY: &str = "https://registry.ollama.ai";
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
    /// Pull model blobs from the Ollama-compatible registry
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
    /// Model name in Ollama library
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

    /// Model name in Ollama format (e.g. qwen3-coder-next); looked up in the pull cache
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

    let manifest = fetch_manifest(&client, &args.model).await?;
    let descriptors = descriptors_from_manifest(manifest);

    let total = descriptors.len();
    let mut completed = 0usize;

    for batch in descriptors.chunks(args.batch_size) {
        println!("Starting batch with {} blobs", batch.len());

        let futures = stream::iter(batch.iter().cloned().map(|descriptor| {
            let client = client.clone();
            let model = args.model.clone();
            let output_root = output_root.clone();
            async move {
                download_blob_with_retry(&client, &model, &output_root, &descriptor, args.retries)
                    .await
            }
        }))
        .buffer_unordered(args.parallel)
        .collect::<Vec<_>>();

        let results = futures.await;
        for result in results {
            result?;
            completed += 1;
            println!("Progress: {completed}/{total} blobs downloaded");
        }
    }

    println!(
        "Pull complete: {} blobs available in {}",
        total,
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
    retries: u32,
) -> Result<()> {
    let digest = descriptor.digest.clone();
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match download_blob(client, model, output_root, &digest, descriptor.size).await {
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
) -> Result<()> {
    let path = blob_path(output_root, model, digest);
    if path.exists() {
        let metadata = std::fs::metadata(&path)
            .with_context(|| format!("failed to stat existing blob {}", path.display()))?;
        if metadata.len() == expected_size {
            println!("Skipping existing blob {}", path.display());
            return Ok(());
        }
        eprintln!("Existing blob has wrong size, re-downloading: {}", path.display());
        std::fs::remove_file(&path)
            .with_context(|| format!("failed to remove stale blob {}", path.display()))?;
    }

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create blob directory {}", parent.display()))?;
    }

    let url = format!("{OLLAMA_REGISTRY}/v2/library/{model}/blobs/{digest}");
    let response = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to request blob {digest}"))?
        .error_for_status()
        .with_context(|| format!("blob request failed for {digest}"))?;

    let mut file = tokio::fs::File::create(&path)
        .await
        .with_context(|| format!("failed to create blob file {}", path.display()))?;

    let mut bytes_written = 0u64;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("failed reading stream for {digest}"))?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("failed writing blob {digest}"))?;
        bytes_written += chunk.len() as u64;
    }

    file.flush().await.context("failed to flush blob file")?;

    if bytes_written != expected_size {
        bail!("blob {digest} size mismatch: expected {expected_size}, wrote {bytes_written}");
    }

    Ok(())
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
}
