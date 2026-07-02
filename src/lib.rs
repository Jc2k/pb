use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use futures::{StreamExt, stream};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::StatusCode;
use reqwest::header::{ACCEPT, CONTENT_LENGTH, RANGE};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tokio::time::{Duration, sleep};

use crate::agent_core::AgentProfile;
use crate::config::UserConfig;
use crate::environment::{EnvironmentBackend, EnvironmentConfig, EnvironmentMode};
use crate::integrations::{IntegrationInstallRequest, IntegrationKind};
use crate::mcp::{McpServerConfig, ProjectMcpConfig};
use base64::Engine as _;

pub mod agent_core;
pub mod browser_tools;
pub mod cli_ui;
pub mod config;
pub mod container;
pub mod daemon_client;
pub mod energy;
pub mod environment;
pub mod events;
mod github_oauth;
pub mod inference;
pub mod init;
pub mod integrations;
pub mod lsp;
pub mod mcp;
pub mod memory;
pub mod policy;
pub mod projects;
pub mod service;
pub mod session_power;
pub mod session_store;
pub mod tray;
pub mod user;
pub mod web;

pub const DEFAULT_MODEL: &str = "hf://unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M.gguf";
const OLLAMA_REGISTRY: &str = "https://registry.ollama.ai";
const HF_ENDPOINT: &str = "https://huggingface.co";
const PROGRESS_BAR_WIDTH: usize = 40;
const PROGRESS_TICK_MS: u64 = 120;
pub const DEFAULT_AGENT_MAX_STEPS: usize = 12;
pub const DEFAULT_AGENT_MAX_TOKENS: i32 = 2048;
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
    /// Submit or attach to a daemon-backed session over the pb unix socket
    Queue(QueueArgs),
    /// Start the web UI server
    Serve,
    /// Manage user-level configuration
    #[command(name = "config")]
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Manage named projects in the user-global registry
    #[command(name = "projects", alias = "project")]
    Projects {
        #[command(subcommand)]
        command: ProjectsCommand,
    },
    /// Manage the per-project container environment for sandboxed task execution
    #[command(name = "env")]
    Env {
        #[command(subcommand)]
        command: EnvCommand,
    },
    /// Set up common MCP servers for this project
    #[command(name = "mcp")]
    Mcp {
        #[command(subcommand)]
        command: McpCommand,
    },
    /// List and install project-scoped MCP/LSP integrations
    #[command(name = "integrations", alias = "integration")]
    Integrations {
        #[command(subcommand)]
        command: IntegrationsCommand,
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
    /// Install pb into ~/.local/bin and register the launchd service
    Install,
    /// Stop the launchd service, remove its configuration, and delete the installed binary
    Uninstall(SelfUninstallArgs),
    /// Update pb from the latest GitHub release
    Update,
    /// Refresh launchd service configuration after a self-update
    #[command(name = "refresh-service", hide = true)]
    RefreshService,
}

#[derive(Args, Debug)]
pub struct SelfUninstallArgs {
    /// Also delete pb data, cache, config, state, and log files
    #[arg(long = "delete-data")]
    pub delete_data: bool,
}

#[derive(Subcommand, Debug)]
pub enum EnvCommand {
    /// Pull a ready-made dev-environment image and save it as the project environment
    Pull(EnvPullArgs),
    /// Build a dev-environment image from a Dockerfile and save it as the project environment
    Build(EnvBuildArgs),
    /// Configure this project to run commands directly on the local host
    Local(EnvLocalArgs),
    /// Verify the configured environment by creating a test container or running local init commands
    Start(EnvWorkdirArgs),
    /// Show the current project environment configuration
    Status(EnvWorkdirArgs),
}

#[derive(Subcommand, Debug)]
pub enum ConfigCommand {
    /// Set a user configuration value, for example: pb config set web.listen 0.0.0.0
    Set(ConfigSetArgs),
    /// Get a user configuration value
    Get(ConfigGetArgs),
    /// Show the user configuration TOML
    Show,
}

#[derive(Subcommand, Debug)]
pub enum McpCommand {
    /// Set up an MCP server
    Setup {
        #[command(subcommand)]
        command: McpSetupCommand,
    },
}

#[derive(Subcommand, Debug)]
pub enum IntegrationsCommand {
    /// List marketplace and installed integrations for this project
    List(IntegrationsListArgs),
    /// Add a project-scoped MCP or LSP integration by container image
    Add(IntegrationsAddArgs),
    /// Remove a configured project MCP or global LSP integration
    Remove(IntegrationsRemoveArgs),
}

#[derive(Args, Debug, Clone)]
pub struct IntegrationsListArgs {
    /// Project root; defaults to the nearest git repository root
    #[arg(long)]
    pub workdir: Option<PathBuf>,

    /// Only show integrations of this kind: mcp or lsp
    #[arg(long)]
    pub kind: Option<String>,

    /// Include marketplace entries from the crunchy-pb GitHub org
    #[arg(long)]
    pub marketplace: bool,
}

#[derive(Args, Debug, Clone)]
pub struct IntegrationsAddArgs {
    /// Integration kind: mcp or lsp
    pub kind: String,

    /// Container image to configure, for example ghcr.io/crunchy-pb/sentry-mcp:latest
    pub container_image: String,

    /// Project root; defaults to the nearest git repository root
    #[arg(long)]
    pub workdir: Option<PathBuf>,

    /// Server name to write in the per-project config; defaults to the image name
    #[arg(long)]
    pub name: Option<String>,

    /// Container runtime command used to run the integration
    #[arg(long, default_value = "docker")]
    pub runtime: String,

    /// Do not overwrite an existing integration with the same name
    #[arg(long)]
    pub no_overwrite: bool,
}

#[derive(Args, Debug, Clone)]
pub struct IntegrationsRemoveArgs {
    /// Integration kind: mcp or lsp
    pub kind: String,

    /// Configured integration/server name to remove
    pub name: String,

    /// Project root for MCP integrations; defaults to the nearest git repository root
    #[arg(long)]
    pub workdir: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum McpSetupCommand {
    /// Configure the official GitHub MCP server for the current project
    Github(McpSetupGithubArgs),
}

#[derive(Args, Debug, Clone)]
pub struct McpSetupGithubArgs {
    /// Project root; defaults to the nearest git repository root
    #[arg(long)]
    pub workdir: Option<PathBuf>,

    /// MCP server name to write under [servers.<name>]
    #[arg(long, default_value = "github")]
    pub server_name: String,

    /// Container runtime command used to run ghcr.io/github/github-mcp-server
    #[arg(long, default_value = "docker")]
    pub runtime: String,

    /// Print the GitHub authorization URL instead of opening a browser
    #[arg(long)]
    pub no_open: bool,

    /// Do not overwrite an existing server with the same name
    #[arg(long)]
    pub no_overwrite: bool,
}

#[derive(Args, Debug, Clone)]
pub struct ConfigSetArgs {
    /// Dot-separated config key, such as web.listen or model.temperature
    pub key: String,

    /// Value to store
    pub value: String,
}

#[derive(Args, Debug, Clone)]
pub struct ConfigGetArgs {
    /// Dot-separated config key, such as web.listen or model.temperature
    pub key: String,
}

#[derive(Subcommand, Debug)]
pub enum ProjectsCommand {
    /// Add a project to the user-global registry
    Add(ProjectAddArgs),
    /// List registered projects
    List(ProjectListArgs),
    /// Remove a project from the registry by name
    #[command(alias = "remove")]
    Rm(ProjectRemoveArgs),
}

#[derive(Args, Debug, Clone)]
pub struct ProjectAddArgs {
    /// Project directory; defaults to the current directory
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Project name; defaults to the directory name
    #[arg(long)]
    pub name: Option<String>,

    /// Unix socket path for the pb daemon
    #[arg(long)]
    pub socket_path: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
pub struct ProjectListArgs {
    /// Unix socket path for the pb daemon
    #[arg(long)]
    pub socket_path: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
pub struct ProjectRemoveArgs {
    /// Registered project name to remove
    pub name: String,

    /// Unix socket path for the pb daemon
    #[arg(long)]
    pub socket_path: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum ServiceCommand {
    /// Start the pb serve service immediately
    Start,
    /// Stop the pb serve service
    Stop,
    /// Restart the pb serve service
    Restart,
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
pub struct EnvLocalArgs {
    /// Shell commands to run locally before agent work begins (may be repeated)
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

    /// Force the project execution backend; defaults to Apple containers
    #[arg(long, value_enum)]
    pub backend: Option<EnvironmentBackend>,
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
pub struct QueueArgs {
    /// Task to execute. Omit with --session to attach to an existing session.
    pub task: Option<String>,

    /// Model identifier: Ollama name (e.g. qwen3-coder-next) or Hugging Face URI
    /// (e.g. hf://unsloth/Qwen3-Coder-Next-GGUF); looked up in the pull cache
    #[arg(long)]
    pub model: Option<String>,

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
    #[arg(long)]
    pub max_steps: Option<usize>,

    /// Maximum new tokens per model turn
    #[arg(long)]
    pub max_tokens: Option<i32>,

    /// Context size
    #[arg(long)]
    pub ctx_size: Option<u32>,

    /// Number of CPU threads for decoding
    #[arg(long)]
    pub threads: Option<i32>,

    /// Number of CPU threads for prompt processing
    #[arg(long)]
    pub threads_batch: Option<i32>,

    /// Number of transformer layers to offload to GPU
    #[arg(long)]
    pub gpu_layers: Option<u32>,

    /// Temperature
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Agent profile for the primary session
    #[arg(long, value_enum)]
    pub profile: Option<AgentProfile>,

    /// Attach an image file to the task (may be repeated)
    #[arg(long = "image", value_name = "PATH")]
    pub images: Vec<PathBuf>,

    /// Top-k for sampling
    #[arg(long)]
    pub top_k: Option<i32>,

    /// RNG seed
    #[arg(long)]
    pub seed: Option<u32>,

    /// Attach to an existing daemon session instead of starting a new one
    #[arg(long)]
    pub session: Option<String>,

    /// List daemon sessions instead of starting or attaching
    #[arg(long)]
    pub list: bool,

    /// Delete a finished daemon session and its git notes ref
    #[arg(long = "delete-session", value_name = "ID")]
    pub delete_session: Option<String>,

    /// Resume a restored paused queue entry
    #[arg(long = "resume-session", value_name = "ID")]
    pub resume_session: Option<String>,

    /// Answer a pending planning question for a daemon session
    #[arg(long = "answer-question", value_name = "QUESTION_ID")]
    pub answer_question: Option<String>,

    /// Answer text to send with --answer-question
    #[arg(long, requires = "answer_question")]
    pub answer: Option<String>,

    /// Submit the session without streaming events
    #[arg(long)]
    pub no_follow: bool,

    /// Unix socket path for the pb daemon
    #[arg(long)]
    pub socket_path: Option<PathBuf>,
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
            SelfCommand::Install => run_self_install(),
            SelfCommand::Uninstall(args) => run_self_uninstall(&args),
            SelfCommand::Update => run_self_update(),
            SelfCommand::RefreshService => run_self_refresh_service(),
        },
        Commands::Pull(args) => pull_model(&args).await,
        Commands::Queue(args) => run_queue(args).await,
        Commands::Serve => run_serve().await,
        Commands::Config { command } => run_config_command(command),
        Commands::Projects { command } => run_projects_command(command).await,
        Commands::Env { command } => run_env_command(command),
        Commands::Mcp { command } => run_mcp_command(command).await,
        Commands::Integrations { command } => run_integrations_command(command).await,
        Commands::Service { command } => run_service_command(command),
        Commands::Init(args) => init::run_init(args.workdir, args.backend),
    }
}

async fn run_serve() -> Result<()> {
    let user_config = UserConfig::load()?;
    let resolved_host = user_config.effective_web_listen();
    let resolved_port = user_config.effective_web_port();
    let defaults = agent_core::AgentRequest {
        task: String::new(),
        model: user_config.effective_model(),
        model_dir: user_config.effective_model_dir(),
        workdir: user_config.effective_workdir(),
        branch: None,
        max_steps: user_config.effective_max_steps(),
        max_tokens: user_config.effective_max_tokens(),
        ctx_size: user_config.effective_ctx_size(),
        threads: user_config.effective_threads(),
        threads_batch: user_config.effective_threads_batch(),
        gpu_layers: user_config.effective_gpu_layers(),
        temperature: user_config.effective_temperature(),
        profile: user_config.effective_profile(),
        infer_profile: false,
        sub_agent_depth: 0,
        repository_less: false,
        top_k: user_config.effective_top_k(),
        seed: user_config.effective_seed(),
        environment: None,
        session_id: String::new(),
        attachments: Vec::new(),
    };
    let server_args = web::ServeArgs {
        host: resolved_host.clone(),
        port: resolved_port,
        socket_path: user_config.effective_socket_path(),
    };

    run_serve_platform(server_args, defaults, resolved_host, resolved_port).await
}

#[cfg(not(target_os = "macos"))]
async fn run_serve_platform(
    server_args: web::ServeArgs,
    defaults: agent_core::AgentRequest,
    _resolved_host: String,
    _resolved_port: u16,
) -> Result<()> {
    web::run_server(server_args, defaults).await
}

#[cfg(target_os = "macos")]
async fn run_serve_platform(
    server_args: web::ServeArgs,
    defaults: agent_core::AgentRequest,
    resolved_host: String,
    resolved_port: u16,
) -> Result<()> {
    use std::sync::mpsc;

    let (ready_tx, ready_rx) = mpsc::channel();
    let server_thread = std::thread::Builder::new()
        .name("pb-web-server".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            runtime.block_on(web::run_server_with_ready(
                server_args,
                defaults,
                Some(ready_tx),
            ))
        })
        .context("failed to start pb web server thread")?;

    match ready_rx
        .recv()
        .context("pb web server exited before startup")?
    {
        Ok(_) => tray::run(tray::TrayArgs {
            host: resolved_host,
            port: resolved_port,
        }),
        Err(err) => {
            let _ = server_thread.join();
            bail!(err);
        }
    }
}

fn run_config_command(command: ConfigCommand) -> Result<()> {
    match command {
        ConfigCommand::Set(args) => {
            let mut config = UserConfig::load()?;
            config.set(&args.key, &args.value)?;
            config.save()?;
            println!(
                "{} = {}",
                args.key,
                config.get(&args.key)?.unwrap_or_default()
            );
        }
        ConfigCommand::Get(args) => {
            let config = UserConfig::load()?;
            match config.get(&args.key)? {
                Some(value) => println!("{value}"),
                None => bail!("config key '{}' is not set", args.key),
            }
        }
        ConfigCommand::Show => {
            let config = UserConfig::load()?;
            print!("{}", config.to_pretty_toml()?);
        }
    }
    Ok(())
}

async fn run_integrations_command(command: IntegrationsCommand) -> Result<()> {
    match command {
        IntegrationsCommand::List(args) => {
            let root = resolve_env_root(args.workdir)?;
            let filter = args
                .kind
                .as_deref()
                .map(IntegrationKind::parse)
                .transpose()?;
            let installed = integrations::list_project_installed(&root)?;
            if installed.is_empty() {
                println!("no project integrations installed");
            } else {
                for item in installed
                    .iter()
                    .filter(|item| filter.is_none_or(|kind| item.kind == kind))
                {
                    println!(
                        "installed	{}	{}	{}",
                        item.kind.as_str(),
                        item.name,
                        item.container_image
                    );
                }
            }
            if args.marketplace {
                for item in integrations::list_marketplace()
                    .await?
                    .into_iter()
                    .filter(|item| filter.is_none_or(|kind| item.kind == kind))
                {
                    println!(
                        "marketplace	{}	{}	{}	{}",
                        item.kind.as_str(),
                        item.name,
                        item.container_image,
                        item.description
                    );
                }
            }
        }
        IntegrationsCommand::Add(args) => {
            let root = resolve_env_root(args.workdir)?;
            let kind = IntegrationKind::parse(&args.kind)?;
            let request = IntegrationInstallRequest {
                kind,
                container_image: args.container_image,
                name: args.name,
                runtime: Some(args.runtime),
                env: Default::default(),
                no_overwrite: args.no_overwrite,
            };
            let response = match kind {
                IntegrationKind::Mcp => integrations::install_project(&root, request),
                IntegrationKind::Lsp => integrations::install_global_lsp(request),
            }?;
            println!(
                "installed {} integration '{}' from {} in {}",
                response.installed.kind.as_str(),
                response.installed.name,
                response.installed.container_image,
                response.config_path
            );
        }
        IntegrationsCommand::Remove(args) => {
            let kind = IntegrationKind::parse(&args.kind)?;
            let response = match kind {
                IntegrationKind::Mcp => {
                    let root = resolve_env_root(args.workdir)?;
                    integrations::remove_project(&root, kind, &args.name)
                }
                IntegrationKind::Lsp => integrations::remove_global_lsp(&args.name),
            }?;
            println!(
                "removed {} integration '{}' from {}",
                response.removed.kind.as_str(),
                response.removed.name,
                response.config_path
            );
        }
    }
    Ok(())
}

async fn run_mcp_command(command: McpCommand) -> Result<()> {
    match command {
        McpCommand::Setup { command } => match command {
            McpSetupCommand::Github(args) => mcp_setup_github(args).await,
        },
    }
}

async fn mcp_setup_github(args: McpSetupGithubArgs) -> Result<()> {
    let root = resolve_env_root(args.workdir)?;
    if args.server_name.trim().is_empty() {
        bail!("--server-name cannot be empty");
    }

    let client_id = non_empty_baked_value(BAKED_GITHUB_CLIENT_ID).ok_or_else(|| {
        anyhow::anyhow!(
            "GitHub OAuth client id was not baked into this binary; rebuild with PB_GITHUB_CLIENT_ID set"
        )
    })?;
    let user_config = UserConfig::load()?;
    let listen = user_config.effective_web_listen();
    let port = user_config.effective_web_port();
    let redirect_uri = github_oauth::redirect_uri(&listen, port);
    let request = github_oauth::begin(client_id, &redirect_uri, &["repo", "read:org"])?;
    github_oauth::clear_callback(&request.state)?;

    let callback_listener =
        github_oauth::try_start_callback_listener(github_oauth::callback_bind_addr(&listen, port)?);
    if callback_listener.is_none() {
        println!(
            "Using existing pb serve callback endpoint at {}.",
            request.redirect_uri
        );
    } else {
        println!(
            "Listening for GitHub OAuth callback at {}.",
            request.redirect_uri
        );
    }

    println!("Opening GitHub authorization in your browser…");
    println!("{}", request.authorize_url);
    if !args.no_open
        && let Err(err) = open_browser(&request.authorize_url)
    {
        eprintln!("Could not open a browser automatically: {err:#}");
        eprintln!("Open the printed GitHub authorization URL manually to continue.");
    }

    let callback = github_oauth::wait_for_callback(&request.state, callback_listener)?;
    let code = callback
        .code
        .as_deref()
        .context("GitHub OAuth callback did not include an authorization code")?;
    let token = github_oauth::exchange_code(
        client_id,
        code,
        &request.code_verifier,
        &request.redirect_uri,
    )
    .await?;
    let token_path = github_oauth::write_token(&token)?;

    let mut config = ProjectMcpConfig::load(&root)?.unwrap_or_default();
    if args.no_overwrite && config.servers.contains_key(&args.server_name) {
        bail!(
            "MCP server '{}' already exists in {}; remove --no-overwrite to replace it",
            args.server_name,
            mcp::project_mcp_config_path(&root).display()
        );
    }

    config.servers.insert(
        args.server_name.clone(),
        github_mcp_server_config(&args.runtime, &token_path),
    );
    config.save(&root)?;

    println!(
        "GitHub MCP server '{}' saved to {}.",
        args.server_name,
        mcp::project_mcp_config_path(&root).display()
    );
    println!("GitHub OAuth token saved to {}.", token_path.display());
    if let Some(repo) = current_github_repo(&root)? {
        println!("Detected GitHub repository: {repo}");
    }
    Ok(())
}

const BAKED_GITHUB_CLIENT_ID: Option<&str> = option_env!("PB_GITHUB_CLIENT_ID");

fn github_mcp_server_config(runtime: &str, token_path: &Path) -> McpServerConfig {
    let token_path = shell_single_quote(&token_path.to_string_lossy());
    McpServerConfig {
        command: Some("sh".to_string()),
        args: vec![
            "-c".to_string(),
            format!(
                "GITHUB_PERSONAL_ACCESS_TOKEN=\"$(cat {token_path})\" exec {runtime} run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server"
            ),
        ],
        env: Default::default(),
        working_directory: None,
        disabled: false,
        ..Default::default()
    }
}

fn non_empty_baked_value(value: Option<&'static str>) -> Option<&'static str> {
    value.filter(|secret| !secret.trim().is_empty())
}

fn shell_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    let mut command = std::process::Command::new("open");
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = std::process::Command::new("cmd");
        command.args(["/C", "start", ""]);
        command
    };
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    let mut command = std::process::Command::new("xdg-open");

    command.arg(url);
    let status = command
        .status()
        .context("failed to open browser for GitHub OAuth authorization")?;
    if !status.success() {
        bail!("browser command failed; open the printed GitHub authorization URL manually");
    }
    Ok(())
}

fn current_github_repo(root: &Path) -> Result<Option<String>> {
    let output = std::process::Command::new("git")
        .args(["config", "--get", "remote.origin.url"])
        .current_dir(root)
        .output()
        .context("failed to read git remote.origin.url")?;
    if !output.status.success() {
        return Ok(None);
    }
    Ok(parse_github_repo_url(
        String::from_utf8_lossy(&output.stdout).trim(),
    ))
}

fn parse_github_repo_url(url: &str) -> Option<String> {
    let trimmed = url.trim().trim_end_matches(".git");
    if let Some(rest) = trimmed.strip_prefix("git@github.com:") {
        return Some(rest.to_string());
    }
    for prefix in ["https://github.com/", "ssh://git@github.com/"] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return Some(rest.to_string());
        }
    }
    None
}

async fn run_projects_command(command: ProjectsCommand) -> Result<()> {
    match command {
        ProjectsCommand::Add(args) => {
            let socket_path = args
                .socket_path
                .clone()
                .unwrap_or_else(daemon_client::default_socket_path);
            let entry = daemon_client::add_project(
                &socket_path,
                projects::AddProjectRequest {
                    name: args.name,
                    path: args.path.to_string_lossy().into_owned(),
                },
            )
            .await?;
            println!("added project {}\t{}", entry.name, entry.path);
        }
        ProjectsCommand::List(args) => {
            let socket_path = args
                .socket_path
                .clone()
                .unwrap_or_else(daemon_client::default_socket_path);
            let projects = daemon_client::list_projects(&socket_path).await?;
            if projects.is_empty() {
                println!("no projects registered");
                return Ok(());
            }
            for project in projects {
                println!("{}\t{}", project.name, project.path);
            }
        }
        ProjectsCommand::Rm(args) => {
            let socket_path = args
                .socket_path
                .clone()
                .unwrap_or_else(daemon_client::default_socket_path);
            let entry = daemon_client::remove_project(
                &socket_path,
                projects::RemoveProjectRequest { name: args.name },
            )
            .await?;
            println!("removed project {}\t{}", entry.name, entry.path);
        }
    }
    Ok(())
}

async fn run_queue(args: QueueArgs) -> Result<()> {
    let socket_path = args
        .socket_path
        .clone()
        .unwrap_or_else(daemon_client::default_socket_path);

    if let Some(session_id) = args.delete_session.clone() {
        let deleted = daemon_client::delete_session(&socket_path, session_id).await?;
        println!("deleted session {}", deleted.session_id);
        return Ok(());
    }

    if let Some(session_id) = args.resume_session.clone() {
        daemon_client::resume_session(&socket_path, session_id.clone()).await?;
        println!("resumed queued session {session_id}");
        if !args.no_follow {
            daemon_client::watch_session(&socket_path, session_id).await?;
        }
        return Ok(());
    }

    if let Some(question_id) = args.answer_question.clone() {
        let session_id = args
            .session
            .clone()
            .context("--answer-question requires --session <id>")?;
        let answer = args
            .answer
            .clone()
            .context("--answer-question requires --answer <text>")?;
        daemon_client::answer_question(
            &socket_path,
            session_id.clone(),
            web::AnswerQuestionRequest {
                question_id,
                answer,
            },
        )
        .await?;
        println!("answered question for session {session_id}");
        return Ok(());
    }

    if args.list {
        let sessions = daemon_client::list_sessions(&socket_path).await?;
        if sessions.is_empty() {
            println!("no sessions queued in daemon");
            return Ok(());
        }
        for session in sessions {
            let status = match session.status {
                session_store::SessionStatus::Queued => "queued",
                session_store::SessionStatus::Running => "running",
                session_store::SessionStatus::Paused => {
                    if session.paused {
                        "paused"
                    } else {
                        "paused-restored"
                    }
                }
                session_store::SessionStatus::Completed => "completed",
                session_store::SessionStatus::Failed => "failed",
            };
            let workdir = session.workdir.unwrap_or_else(|| "-".to_string());
            println!(
                "{}\t{}\t{}\t{}",
                session.session_id, status, workdir, session.task
            );
        }
        return Ok(());
    }

    let session_id = if let Some(session_id) = args.session.clone() {
        session_id
    } else {
        let task = args.task.clone().context(
            "missing task; pass a task, --session <id>, --delete-session <id>, --answer-question <id>, or --list",
        )?;
        let response = daemon_client::start_session(
            &socket_path,
            web::StartSessionRequest {
                task,
                model: args.model.clone(),
                model_dir: args
                    .model_dir
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned()),
                workdir: args
                    .workdir
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned()),
                branch: args.branch.clone(),
                max_steps: args.max_steps,
                max_tokens: args.max_tokens,
                ctx_size: args.ctx_size,
                threads: args.threads,
                threads_batch: args.threads_batch,
                gpu_layers: args.gpu_layers,
                temperature: args.temperature,
                profile: args.profile,
                top_k: args.top_k,
                seed: args.seed,
                attachments: args
                    .images
                    .iter()
                    .map(|path| cli_attachment(path))
                    .collect::<Result<Vec<_>>>()?,
            },
        )
        .await?;
        println!("queued session {}", response.session_id);
        response.session_id
    };

    if !args.no_follow {
        daemon_client::watch_session(&socket_path, session_id).await?;
    }

    Ok(())
}

fn run_self_install() -> Result<()> {
    ensure_macos_launchd()?;

    let source = std::env::current_exe().context("cannot determine path to pb binary")?;
    let destination = installed_binary_path()?;
    confirm(&format!(
        "Install pb by moving {} to {} and starting the launchd service?",
        source.display(),
        destination.display()
    ))?;

    let bin_dir = destination
        .parent()
        .context("installed binary path has no parent directory")?;
    std::fs::create_dir_all(bin_dir)
        .with_context(|| format!("failed to create {}", bin_dir.display()))?;

    if source != destination {
        if destination.exists() {
            std::fs::remove_file(&destination)
                .with_context(|| format!("failed to replace {}", destination.display()))?;
        }
        std::fs::rename(&source, &destination).or_else(|rename_err| {
            std::fs::copy(&source, &destination).with_context(|| {
                format!(
                    "failed to move {} to {} ({rename_err})",
                    source.display(),
                    destination.display()
                )
            })?;
            std::fs::remove_file(&source)
                .with_context(|| format!("failed to remove {}", source.display()))?;
            Ok::<(), anyhow::Error>(())
        })?;
    }

    service::install(&destination)?;
    service::start()?;
    println!("pb installed at {}", destination.display());
    Ok(())
}

fn run_self_uninstall(args: &SelfUninstallArgs) -> Result<()> {
    ensure_macos_launchd()?;

    let destination = installed_binary_path()?;
    let data_note = if args.delete_data {
        " This will also delete pb data, cache, config, state, and log files."
    } else {
        ""
    };
    confirm(&format!(
        "Uninstall pb, stop the launchd service, remove its configuration, and delete {}?{data_note}",
        destination.display()
    ))?;

    service::remove()?;

    if destination.exists() {
        std::fs::remove_file(&destination)
            .with_context(|| format!("failed to remove {}", destination.display()))?;
        println!("Removed {}", destination.display());
    } else {
        println!("No installed binary found at {}", destination.display());
    }

    if args.delete_data {
        remove_self_data()?;
    }

    Ok(())
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
    run_updated_binary_service_refresh(&installed_binary_path()?)?;
    Ok(())
}

fn run_self_refresh_service() -> Result<()> {
    service::refresh_plist_and_reload_if_changed(&installed_binary_path()?)
}

fn run_updated_binary_service_refresh(binary: &Path) -> Result<()> {
    use std::process::Command;

    let status = Command::new(binary)
        .args(["self", "refresh-service"])
        .status()
        .with_context(|| {
            format!(
                "failed to run updated pb binary at {} to refresh launchd service",
                binary.display()
            )
        })?;
    if !status.success() {
        bail!("updated pb binary failed to refresh launchd service (exit {status})");
    }
    Ok(())
}

fn ensure_macos_launchd() -> Result<()> {
    if !cfg!(target_os = "macos") {
        bail!("pb self install/uninstall is only supported on macOS");
    }
    Ok(())
}

fn installed_binary_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home.join(".local").join("bin").join("pb"))
}

fn confirm(prompt: &str) -> Result<()> {
    use std::io::{self, Write};

    print!("{prompt} [y/N] ");
    io::stdout().flush().context("failed to flush stdout")?;

    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .context("failed to read confirmation")?;
    match answer.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => Ok(()),
        _ => bail!("cancelled"),
    }
}

fn remove_self_data() -> Result<()> {
    for path in self_data_paths()? {
        remove_path_if_exists(&path)?;
    }
    Ok(())
}

fn self_data_paths() -> Result<Vec<PathBuf>> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".cache"))
        .join("pb");
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".config"))
        .join("pb");
    let state = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".local").join("state"))
        .join("pb");
    Ok(vec![
        default_data_dir(),
        cache,
        config,
        state,
        home.join("Library").join("Logs").join("pb.stdout.log"),
        home.join("Library").join("Logs").join("pb.stderr.log"),
        home.join("Library").join("Logs").join("pb.tray.stdout.log"),
        home.join("Library").join("Logs").join("pb.tray.stderr.log"),
    ])
}

fn remove_path_if_exists(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_dir() {
        std::fs::remove_dir_all(path)
            .with_context(|| format!("failed to remove {}", path.display()))?;
    } else {
        std::fs::remove_file(path)
            .with_context(|| format!("failed to remove {}", path.display()))?;
    }
    println!("Removed {}", path.display());
    Ok(())
}

fn run_env_command(command: EnvCommand) -> Result<()> {
    match command {
        EnvCommand::Pull(args) => env_pull(args),
        EnvCommand::Build(args) => env_build(args),
        EnvCommand::Local(args) => env_local(args),
        EnvCommand::Start(args) => env_start(args),
        EnvCommand::Status(args) => env_status(args),
    }
}

fn run_service_command(command: ServiceCommand) -> Result<()> {
    match command {
        ServiceCommand::Start => service::start(),
        ServiceCommand::Stop => service::stop(),
        ServiceCommand::Restart => service::restart(),
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
    let runtime = container::detect_runtime()
        .context("no container runtime found; install docker, podman, or apple/container")?;
    println!("Pulling image {}…", args.image);
    runtime.pull(&args.image)?;
    let config = EnvironmentConfig {
        mode: EnvironmentMode::Pull,
        backend: EnvironmentBackend::AppleContainers,
        image: args.image.clone(),
        init_commands: args.init_commands,
        setup_commands: vec![],
        session_commands: vec![],
        guard_commands: vec![],
        prepared_image: None,
        source: None,
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
    let runtime = container::detect_runtime()
        .context("no container runtime found; install docker, podman, or apple/container")?;
    println!("Building image {} from {}…", args.tag, dockerfile.display());
    runtime.build(&dockerfile, &args.tag)?;
    let config = EnvironmentConfig {
        mode: EnvironmentMode::Build,
        backend: EnvironmentBackend::AppleContainers,
        image: args.tag.clone(),
        init_commands: args.init_commands,
        setup_commands: vec![],
        session_commands: vec![],
        guard_commands: vec![],
        prepared_image: None,
        source: None,
        dockerfile: Some(args.dockerfile),
    };
    config.save(&root)?;
    println!(
        "Environment saved to {}",
        root.join(".pb").join("environment.toml").display()
    );
    Ok(())
}

fn env_local(args: EnvLocalArgs) -> Result<()> {
    let root = resolve_env_root(args.workdir)?;
    let config = EnvironmentConfig {
        mode: EnvironmentMode::Local,
        backend: EnvironmentBackend::Local,
        image: "local".to_string(),
        init_commands: args.init_commands,
        setup_commands: vec![],
        session_commands: vec![],
        guard_commands: vec![],
        prepared_image: None,
        source: None,
        dockerfile: None,
    };
    config.save(&root)?;
    println!(
        "Local environment saved to {}",
        root.join(".pb").join("environment.toml").display()
    );
    Ok(())
}

fn env_start(args: EnvWorkdirArgs) -> Result<()> {
    let root = resolve_env_root(args.workdir)?;
    let config = EnvironmentConfig::load(&root)?
        .context("no environment configured; run `pb env pull` or `pb env build` first")?;
    match config.backend {
        EnvironmentBackend::AppleContainers => {
            let runtime = container::detect_runtime().context(
                "no container runtime found; install docker, podman, or apple/container",
            )?;
            println!("Creating test container from {}…", config.image);
            let container_id = runtime.create(&config.image, &root)?;
            println!("Container {} started", container_id);
            for cmd in config.setup_commands() {
                println!("Running setup command: {cmd}");
                let output = runtime.exec(&container_id, &cmd)?;
                if !output.is_empty() {
                    println!("{output}");
                }
            }
            for cmd in config.session_commands() {
                println!("Running session command: {cmd}");
                let output = runtime.exec(&container_id, cmd)?;
                if !output.is_empty() {
                    println!("{output}");
                }
            }
            println!("Removing test container…");
            runtime.remove(&container_id)?;
        }
        EnvironmentBackend::Local => {
            println!("Verifying local environment at {}…", root.display());
            for cmd in config.setup_commands() {
                println!("Running setup command locally: {cmd}");
                let output = run_local_command(&cmd, &root)?;
                if !output.is_empty() {
                    println!("{output}");
                }
            }
            for cmd in config.session_commands() {
                println!("Running session command locally: {cmd}");
                let output = run_local_command(cmd, &root)?;
                if !output.is_empty() {
                    println!("{output}");
                }
            }
        }
    }
    println!("Environment verified successfully.");
    Ok(())
}

fn run_local_command(cmd: &str, workdir: &Path) -> Result<String> {
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(workdir)
        .output()
        .with_context(|| format!("failed to spawn local shell for command: {cmd}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("local command failed: {stderr}");
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
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
                EnvironmentMode::Local => "local",
            };
            let backend = match config.backend {
                EnvironmentBackend::AppleContainers => "apple-containers",
                EnvironmentBackend::Local => "local",
            };
            println!("mode:  {mode}");
            println!("backend: {backend}");
            if config.backend != EnvironmentBackend::Local {
                println!("image: {}", config.image);
            }
            if let Some(df) = &config.dockerfile {
                println!("dockerfile: {}", df.display());
            }
            let setup_commands = config.setup_commands();
            if setup_commands.is_empty() {
                println!("setup_commands: (none)");
            } else {
                println!("setup_commands:");
                for cmd in &setup_commands {
                    println!("  - {cmd}");
                }
            }
            if config.session_commands.is_empty() {
                println!("session_commands: (none)");
            } else {
                println!("session_commands:");
                for cmd in &config.session_commands {
                    println!("  - {cmd}");
                }
            }
            if config.guard_commands.is_empty() {
                println!("guard_commands: (none)");
            } else {
                println!("guard_commands:");
                for cmd in &config.guard_commands {
                    println!("  - {cmd}");
                }
            }
            if let Some(image) = &config.prepared_image {
                println!("prepared_image: {image}");
            }
            if let Some(source) = &config.source {
                println!("source: {source}");
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
        pull_from_ollama(
            &client,
            &args.model,
            &output_root,
            args.batch_size,
            args.parallel,
            args.retries,
        )
        .await
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
        let matches: Vec<String> = files
            .iter()
            .filter(|f| f.contains(quant))
            .cloned()
            .collect();
        if !matches.is_empty() {
            return matches;
        }
    }
    files.to_vec()
}

async fn list_hf_files(
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

    Ok(info.siblings)
}

async fn list_hf_gguf_files(
    client: &reqwest::Client,
    owner: &str,
    repo: &str,
) -> Result<Vec<HfSibling>> {
    Ok(list_hf_files(client, owner, repo)
        .await?
        .into_iter()
        .filter(|s| s.rfilename.ends_with(".gguf"))
        .collect())
}

/// Return the size for a Hugging Face sibling, preferring top-level `size`
/// and falling back to `lfs.size` when needed.
fn hf_sibling_size(sibling: &HfSibling) -> Option<u64> {
    sibling
        .size
        .or_else(|| sibling.lfs.as_ref().and_then(|lfs| lfs.size))
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
        let template = "{spinner:.green} [{elapsed_precise}] {bytes} downloaded".to_string();
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

async fn pull_flashmoe_from_hf(
    client: &reqwest::Client,
    hf_uri: &str,
    owner: &str,
    repo: &str,
    output_root: &Path,
    parallel: usize,
    retries: u32,
) -> Result<()> {
    let siblings = list_hf_files(client, owner, repo).await?;
    let wanted: Vec<&HfSibling> = siblings
        .iter()
        .filter(|s| {
            s.rfilename.ends_with(".safetensors")
                || crate::inference::flashmoe::expected_hf_files()
                    .iter()
                    .any(|name| name == &std::ffi::OsString::from(&s.rfilename))
        })
        .collect();
    if wanted.is_empty() {
        bail!("no Qwen3.5 safetensors or tokenizer/config files found in {owner}/{repo}");
    }

    let cache_dir = output_root.join(cache_dir_name(hf_uri));
    tokio::fs::create_dir_all(&cache_dir)
        .await
        .with_context(|| format!("failed to create cache directory {}", cache_dir.display()))?;

    let mut files = Vec::with_capacity(wanted.len());
    for sibling in wanted {
        let filename = sibling.rfilename.clone();
        let size = match hf_sibling_size(sibling) {
            s @ Some(_) => s,
            None => {
                let url = format!("{HF_ENDPOINT}/{owner}/{repo}/resolve/main/{filename}");
                fetch_content_length(client, &url).await
            }
        };
        files.push((filename, size));
    }

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
            download_file_with_retry(&client, &url, &dest, size, &progress, &filename, retries)
                .await
        }
    }))
    .buffer_unordered(parallel)
    .collect::<Vec<_>>();

    let results = tasks.await;
    for result in results {
        result?;
    }
    progress.finish_with_message("download complete");

    let plan = crate::inference::flashmoe::build_cache_from_hf_snapshot(hf_uri, &cache_dir)?;
    println!(
        "Pull complete: {total_files} Hugging Face file(s) available in {}; Flash-MoE cache prepared at {}",
        cache_dir.display(),
        plan.runtime_dir.display()
    );
    Ok(())
}

async fn pull_from_hf(
    client: &reqwest::Client,
    hf_uri: &str,
    output_root: &Path,
    parallel: usize,
    retries: u32,
) -> Result<()> {
    let (owner, repo, explicit_filename) =
        parse_hf_uri(hf_uri).with_context(|| format!("invalid Hugging Face URI: {hf_uri}"))?;

    if crate::inference::flashmoe::is_flashmoe_hf_model(hf_uri) {
        return pull_flashmoe_from_hf(
            client,
            hf_uri,
            &owner,
            &repo,
            output_root,
            parallel,
            retries,
        )
        .await;
    }

    let siblings = list_hf_gguf_files(client, &owner, &repo).await?;
    if siblings.is_empty() {
        bail!("no GGUF files found in {owner}/{repo} on Hugging Face");
    }

    let files: Vec<(String, Option<u64>)> = if let Some(f) = explicit_filename {
        let sibling = siblings
            .iter()
            .find(|s| s.rfilename == f)
            .with_context(|| {
                format!("GGUF file {f} not found in {owner}/{repo} on Hugging Face")
            })?;
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
    tokio::fs::create_dir_all(&cache_dir)
        .await
        .with_context(|| format!("failed to create cache directory {}", cache_dir.display()))?;

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
            download_file_with_retry(&client, &url, &dest, size, &progress, &filename, retries)
                .await
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
                eprintln!("Download of {label} failed (attempt {attempt}/{retries}): {err}");
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
            std::fs::remove_file(&tmp_path).with_context(|| {
                format!("failed to remove stale temp file {}", tmp_path.display())
            })?;
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
            std::fs::remove_file(&tmp_path).with_context(|| {
                format!("failed to remove stale temp file {}", tmp_path.display())
            })?;
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

    if let Some(size) = expected_size
        && bytes_written != size
    {
        bail!(
            "size mismatch for {}: expected {size}, wrote {bytes_written}",
            path.display()
        );
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
        existing_bytes(
            &blob_path(output_root, model, &descriptor.digest),
            Some(descriptor.size),
        )
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
        match download_blob(
            client,
            model,
            output_root,
            &digest,
            descriptor.size,
            progress,
        )
        .await
        {
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
        && !xdg.is_empty()
    {
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
        let files = vec![
            "model-IQ3_XS.gguf".to_owned(),
            "model-IQ4_XS.gguf".to_owned(),
        ];
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
            Some((
                "unsloth".to_owned(),
                "Qwen3-Coder-Next-GGUF".to_owned(),
                None
            ))
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
        assert_eq!(
            download_tmp_path(path),
            PathBuf::from("/tmp/model.gguf.tmp")
        );
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

    #[test]
    fn parse_github_repo_url_accepts_common_origin_formats() {
        assert_eq!(
            parse_github_repo_url("git@github.com:owner/repo.git"),
            Some("owner/repo".to_string())
        );
        assert_eq!(
            parse_github_repo_url("https://github.com/owner/repo.git"),
            Some("owner/repo".to_string())
        );
        assert_eq!(
            parse_github_repo_url("ssh://git@github.com/owner/repo.git"),
            Some("owner/repo".to_string())
        );
    }

    #[test]
    fn github_mcp_server_config_reads_token_file_without_gh_dependency() {
        let token_path = Path::new("/tmp/pb-github-token");
        let config = github_mcp_server_config("docker", token_path);
        let command = config.args.join(" ");
        assert_eq!(config.command.as_deref(), Some("sh"));
        assert!(config.env.is_empty());
        assert!(!command.contains("gh auth token"));
        assert!(command.contains("cat '/tmp/pb-github-token'"));
        assert!(command.contains("GITHUB_PERSONAL_ACCESS_TOKEN"));
        assert!(command.contains("ghcr.io/github/github-mcp-server"));
    }

    #[test]
    fn shell_single_quote_escapes_embedded_single_quotes() {
        assert_eq!(shell_single_quote("abc'def"), "'abc'\\''def'");
    }
}

fn cli_attachment(path: &Path) -> Result<web::InlineAttachment> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read image {}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("image")
        .to_string();
    let mime = mime_guess::from_path(path)
        .first_or_octet_stream()
        .to_string();
    Ok(web::InlineAttachment {
        name,
        mime,
        base64: base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}
