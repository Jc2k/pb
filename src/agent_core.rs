use anyhow::{Context, Result, anyhow, bail};
use clap::ValueEnum;
use encoding_rs::UTF_8;
use futures::StreamExt;
use globset::GlobBuilder;
use grep_regex::RegexMatcher;
use grep_searcher::{Searcher, sinks::UTF8 as GrepUtf8};
use ignore::WalkBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use similar::TextDiff;
use std::cell::RefCell;
use std::collections::{HashMap, hash_map::DefaultHasher};
use std::fmt;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::num::NonZeroU32;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

use crate::container;
use crate::environment::{EnvironmentBackend, EnvironmentConfig};
use crate::events::AgentEvent;
use crate::mcp::{self, McpToolRegistry};

const LLAMA_BATCH_SIZE: usize = 512;
const MIN_GENERATION_CONTEXT_TOKENS: usize = 1;
const MAX_SEARCH_RESULTS: usize = 200;
const MAX_GLOB_RESULTS: usize = 200;
const MAX_WEB_SEARCH_RESULTS: usize = 8;
const MAX_WEB_RESPONSE_BYTES: usize = 512 * 1024;
const MAX_WEB_RESULT_CHARS: usize = 20_000;
const MAX_SKILL_SEARCH_RESULTS: usize = 50;
const MAX_SKILL_TEXT_CHARS: usize = 40_000;
const SEARCH_EXCLUDED_DIRS: &[&str] = &[".git", "target"];
const TOOL_USER_AGENT: &str = "pb-agent/1.0";
const MAX_SUB_AGENT_DEPTH: usize = 1;
const DEFAULT_SUB_AGENT_MAX_STEPS: usize = 6;

fn suppress_llama_logs() {
    static LLAMA_LOGS_SUPPRESSED: OnceLock<()> = OnceLock::new();
    LLAMA_LOGS_SUPPRESSED.get_or_init(|| {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
    });
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WebSearchResult {
    title: String,
    url: String,
}

pub trait EventSink {
    fn emit(&mut self, event: AgentEvent);

    fn ask_user(&mut self, _question: &str) -> Result<String> {
        bail!("ask_user is not available in this execution context")
    }
}

impl<F> EventSink for F
where
    F: FnMut(AgentEvent),
{
    fn emit(&mut self, event: AgentEvent) {
        self(event)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum AgentProfile {
    Build,
    Scout,
    Review,
    Explore,
    Plan,
    Ask,
    Research,
}

impl Default for AgentProfile {
    fn default() -> Self {
        Self::Build
    }
}

impl fmt::Display for AgentProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Ask the model to choose the primary agent profile from a user message so
/// callers do not need to choose one explicitly. The prompt describes the
/// profile responsibilities and lets the model classify the request instead of
/// relying on hard-coded trigger phrases.
pub fn infer_agent_profile(
    backend: &LlamaBackend,
    model: &LlamaModel,
    args: &AgentRequest,
    task: &str,
) -> Result<AgentProfile> {
    let mut inference_args = args.clone();
    inference_args.max_tokens = args.max_tokens.clamp(8, 32);
    inference_args.temperature = 0.0;
    inference_args.top_k = 1;

    let prompt = profile_inference_prompt(task);
    let output = generate_completion(backend, model, &inference_args, &prompt)?;
    parse_inferred_agent_profile(&output)
        .with_context(|| format!("failed to infer an agent profile from model output: {output}"))
}

fn profile_inference_prompt(task: &str) -> String {
    format!(
        "<conversation>\n\
[system]\n\
You choose the best pb agent profile for a user's task. Return exactly one JSON object in the form {{\"profile\":\"build\"}} and no other text.\n\
Profiles:\n\
- build: make, change, fix, refactor, or otherwise implement changes in the repository.\n\
- scout: establish or refresh the development environment, dependency setup, or runnable project configuration.\n\
- review: inspect existing code or diffs for correctness, risks, regressions, or test gaps without editing.\n\
- explore: investigate how the codebase works or where behavior lives without editing.\n\
- plan: produce an implementation plan or roadmap without editing.\n\
- ask: answer a focused question that does not require codebase investigation or edits.\n\
- research: deep dive into external information, documentation, errors, ecosystem behavior, or public sources to inform a plan, answer, review, or fix without editing.\n\
Choose one of: build, scout, review, explore, plan, ask, research.\n\n\
[user]\n\
Task:\n{task}\n\n\
[assistant]\n"
    )
}

fn parse_inferred_agent_profile(output: &str) -> Result<AgentProfile> {
    let trimmed = output.trim();
    if let Some(profile) = parse_profile_json(trimmed)? {
        return Ok(profile);
    }
    if let Some((start, end)) = trimmed.find('{').zip(trimmed.rfind('}'))
        && start < end
        && let Some(profile) = parse_profile_json(&trimmed[start..=end])?
    {
        return Ok(profile);
    }

    if let Ok(profile) = AgentProfile::parse(
        trimmed
            .trim_matches(|c: char| c == '`' || c == '"' || c == '\'' || c.is_ascii_whitespace()),
    ) {
        return Ok(profile);
    }

    bail!("expected JSON object with profile field or a profile name")
}

fn parse_profile_json(output: &str) -> Result<Option<AgentProfile>> {
    match serde_json::from_str::<Value>(output) {
        Ok(value) => value
            .get("profile")
            .and_then(Value::as_str)
            .map(AgentProfile::parse)
            .transpose(),
        Err(_) => Ok(None),
    }
}

impl AgentProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Scout => "scout",
            Self::Review => "review",
            Self::Explore => "explore",
            Self::Plan => "plan",
            Self::Ask => "ask",
            Self::Research => "research",
        }
    }

    fn parse(input: &str) -> Result<Self> {
        match input.trim().to_ascii_lowercase().as_str() {
            "build" => Ok(Self::Build),
            "scout" => Ok(Self::Scout),
            "review" => Ok(Self::Review),
            "explore" => Ok(Self::Explore),
            "plan" => Ok(Self::Plan),
            "ask" => Ok(Self::Ask),
            "research" => Ok(Self::Research),
            other => bail!(
                "unknown agent profile '{other}'; expected one of: build, scout, review, explore, plan, ask, research"
            ),
        }
    }

    fn instructions(self) -> &'static str {
        match self {
            Self::Build => {
                "Profile: build. Orchestrate implementation work for requests that make, change, or fix something. First call a plan sub-agent to break the request into concrete build tasks. Then call one or more build sub-agents to implement those tasks, requiring each build sub-agent to git_commit after its logical change. Automatically call a scout sub-agent when you need to establish or refresh a working development environment. After implementation, call a review sub-agent to inspect the committed work before finalizing. Use todo(action=list) or todo(action=next) to inspect shared task memory, todo(action=complete,...) when a task is finished, and todo(action=add,...) when implementation reveals follow-up work. You may edit files and commit logical changes, but prefer delegating planned implementation to build sub-agents."
            }
            Self::Scout => {
                "Profile: scout. First scout the repository's AGENT.md/AGENTS.md, README files, CI workflows, Dockerfiles, and language manifests for dev-environment setup, per-session refresh steps, and commit guard rails. Prefer run_command in the scouted backend. Before committing, run the discovered guard commands and only skip them with a clear reason. You may edit files and commit logical changes."
            }
            Self::Review => {
                "Profile: review. Inspect the current workspace and recent changes for correctness, missing requirements, regressions, and test gaps. Run checks when available. Use todo(action=add,...) for required follow-up work found during review. Do not edit files or create commits. Return concise findings with severity and evidence."
            }
            Self::Explore => {
                "Profile: explore. Investigate the codebase as it pertains to the task. Prefer search/read_file and targeted commands. Do not edit files or create commits. Return a compact map of relevant files, behaviors, and recommendations."
            }
            Self::Plan => {
                "Profile: plan. Produce an actionable implementation plan from the available context and use todo(action=add,...) to create concrete build tasks for each actionable step. Use ask_user(question) only when a human decision or missing requirement blocks a safe plan; the session pauses until the human answers, and you must incorporate the answer before finalizing. Use skill_search to find relevant reusable workflows or framework guidance; either incorporate invoked skills into the plan or plan explicit skill invocations for build/research agents. Do not edit files or create commits. Keep the plan concise and call out assumptions or risks."
            }
            Self::Ask => {
                "Profile: ask. Answer the focused question using repository context and, when necessary, public web research. Launch a research sub-agent when the answer depends on deeper external knowledge, current documentation, ecosystem behavior, or non-trivial source synthesis. Do not edit files or create commits. Return a direct answer with supporting evidence."
            }
            Self::Research => {
                "Profile: research. Deep dive into external knowledge needed for the task: current documentation, public sources, ecosystem behavior, error messages, build failures, API details, or domain background. Use skill_search to find targeted research workflows before broad web searches when the repository provides skills. Prefer web_search and web_fetch, combine findings with targeted repository reads or commands when useful, and clearly separate sourced facts from inferences. Do not edit files, create commits, or launch sub-agents. Return concise findings, source URLs or file evidence, confidence, and how the primary agent should integrate the research."
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TodoStatus {
    Pending,
    InProgress,
    Completed,
    Blocked,
}

impl TodoStatus {
    fn parse(input: &str) -> Result<Self> {
        match input.trim().to_ascii_lowercase().as_str() {
            "pending" => Ok(Self::Pending),
            "in_progress" | "in-progress" | "started" => Ok(Self::InProgress),
            "completed" | "complete" | "done" => Ok(Self::Completed),
            "blocked" => Ok(Self::Blocked),
            other => bail!(
                "unknown todo status '{other}'; expected one of: pending, in_progress, completed, blocked"
            ),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TodoTask {
    id: usize,
    title: String,
    description: String,
    status: TodoStatus,
    parent_id: Option<usize>,
    notes: Vec<String>,
}

#[derive(Debug, Default)]
struct TodoMemory {
    next_id: usize,
    tasks: Vec<TodoTask>,
}

impl TodoMemory {
    fn add(&mut self, title: String, description: String, parent_id: Option<usize>) -> &TodoTask {
        self.next_id += 1;
        self.tasks.push(TodoTask {
            id: self.next_id,
            title,
            description,
            status: TodoStatus::Pending,
            parent_id,
            notes: Vec::new(),
        });
        self.tasks.last().expect("todo was just pushed")
    }

    fn update(
        &mut self,
        id: usize,
        status: Option<TodoStatus>,
        title: Option<String>,
        description: Option<String>,
        note: Option<String>,
    ) -> Result<&TodoTask> {
        let task = self
            .tasks
            .iter_mut()
            .find(|task| task.id == id)
            .with_context(|| format!("todo id {id} was not found"))?;
        if let Some(status) = status {
            task.status = status;
        }
        if let Some(title) = title {
            task.title = title;
        }
        if let Some(description) = description {
            task.description = description;
        }
        if let Some(note) = note
            && !note.trim().is_empty()
        {
            task.notes.push(note);
        }
        Ok(task)
    }

    fn pending_tasks(&self) -> Vec<&TodoTask> {
        self.tasks
            .iter()
            .filter(|task| task.status == TodoStatus::Pending)
            .collect()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentRequest {
    pub task: String,
    pub model: String,
    pub model_dir: Option<PathBuf>,
    pub workdir: Option<PathBuf>,
    pub branch: Option<String>,
    pub max_steps: usize,
    pub max_tokens: i32,
    pub ctx_size: u32,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub gpu_layers: u32,
    pub temperature: f32,
    #[serde(default)]
    pub profile: AgentProfile,
    #[serde(default)]
    pub infer_profile: bool,
    #[serde(default)]
    pub sub_agent_depth: usize,
    pub top_k: i32,
    pub seed: u32,
    /// Optional environment config; when `None`, loaded from `.pb/environment.toml` at runtime.
    pub environment: Option<EnvironmentConfig>,
    #[serde(default)]
    pub session_id: String,
}

#[derive(Debug, Clone)]
pub struct AgentRunResult {
    pub branch: String,
    pub workspace_root: PathBuf,
    pub reached_final: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AgentAction {
    ToolCall {
        tool: String,
        #[serde(default)]
        arguments: Value,
        #[serde(default)]
        thinking: Option<String>,
    },
    Final {
        content: String,
        #[serde(default)]
        thinking: Option<String>,
    },
}

/// Walk up from `start` to find the nearest ancestor directory that contains a `.git` entry.
/// Returns `None` when no git root is found (e.g. the path is outside any repository).
pub fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        if current.join(".git").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandBackendKind {
    Container,
    Local,
}

enum CommandBackend {
    Container(container::ContainerHandle),
    Local { workspace_root: PathBuf },
}

impl CommandBackend {
    fn start(config: &EnvironmentConfig, workspace_root: &Path) -> Result<Self> {
        match config.backend {
            EnvironmentBackend::AppleContainers => {
                let runtime = container::detect_runtime().context(
                    "no container runtime found; install docker, podman, or apple/container",
                )?;
                let image = prepare_container_image(config, workspace_root, runtime.as_ref())?;
                let container_id = runtime
                    .create(&image, workspace_root)
                    .or_else(|create_err| {
                        if config.prepared_image.is_some() {
                            let rebuilt = rebuild_container_image(config, workspace_root, runtime.as_ref())?;
                            runtime.create(&rebuilt, workspace_root).with_context(|| {
                                format!(
                                    "failed to create task container after rebuilding environment; original error: {create_err:#}"
                                )
                            })
                        } else {
                            Err(create_err).context("failed to create task container")
                        }
                    })?;
                let handle = container::ContainerHandle {
                    runtime,
                    container_id,
                };
                for cmd in config.session_commands() {
                    handle
                        .exec(cmd)
                        .with_context(|| format!("container session command failed: {cmd}"))?;
                }
                Ok(CommandBackend::Container(handle))
            }
            EnvironmentBackend::Local => {
                let backend = CommandBackend::Local {
                    workspace_root: workspace_root.to_path_buf(),
                };
                // Local setups are intentionally not refreshed for every session; the
                // scout assumes host dependencies are present unless a later command failure
                // shows otherwise. Only documented per-session refresh commands run here.
                for cmd in config.session_commands() {
                    backend
                        .exec(cmd)
                        .with_context(|| format!("local session command failed: {cmd}"))?;
                }
                Ok(backend)
            }
        }
    }

    fn kind(&self) -> CommandBackendKind {
        match self {
            CommandBackend::Container(_) => CommandBackendKind::Container,
            CommandBackend::Local { .. } => CommandBackendKind::Local,
        }
    }

    fn exec(&self, cmd: &str) -> Result<String> {
        match self {
            CommandBackend::Container(handle) => handle.exec(cmd),
            CommandBackend::Local { workspace_root } => {
                run_local_shell_command(cmd, workspace_root)
            }
        }
    }
}

fn prepare_container_image(
    config: &EnvironmentConfig,
    workspace_root: &Path,
    runtime: &dyn container::ContainerRuntime,
) -> Result<String> {
    if let Some(image) = &config.prepared_image {
        if runtime.image_exists(image)? {
            return Ok(image.clone());
        }
        return rebuild_container_image(config, workspace_root, runtime);
    }
    Ok(config.image.clone())
}

fn rebuild_container_image(
    config: &EnvironmentConfig,
    workspace_root: &Path,
    runtime: &dyn container::ContainerRuntime,
) -> Result<String> {
    let image = config
        .prepared_image
        .clone()
        .unwrap_or_else(|| scouted_image_tag(workspace_root, config));
    let container_id = runtime
        .create(&config.image, workspace_root)
        .with_context(|| format!("failed to create setup container from {}", config.image))?;
    let setup_result = (|| -> Result<()> {
        for cmd in config.setup_commands() {
            runtime
                .exec(&container_id, &cmd)
                .with_context(|| format!("container setup command failed: {cmd}"))?;
        }
        runtime
            .commit(&container_id, &image)
            .with_context(|| format!("failed to tag prepared environment as {image}"))?;
        Ok(())
    })();
    let _ = runtime.remove(&container_id);
    setup_result?;
    Ok(image)
}

fn scouted_image_tag(workspace_root: &Path, config: &EnvironmentConfig) -> String {
    let mut hasher = DefaultHasher::new();
    workspace_root.to_string_lossy().hash(&mut hasher);
    config.image.hash(&mut hasher);
    config.setup_commands().hash(&mut hasher);
    format!("pb-scout:{:016x}", hasher.finish())
}

fn run_local_shell_command(cmd: &str, workdir: &Path) -> Result<String> {
    let output = Command::new("sh")
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

fn determine_branch_for_request(args: &AgentRequest, workspace_root: &Path) -> String {
    if let Some(b) = &args.branch {
        return b.clone();
    }

    // Defensive: first check if a branch exists for this task (possibly without session_id from older runs)
    let base_branch = branch_name_from_task(&args.task);
    if git_branch_exists(&base_branch, workspace_root) {
        return base_branch;
    }

    let mut branch = base_branch.clone();
    if !args.session_id.is_empty() {
        branch.push_str(&format!("-{}", args.session_id));

        // If suffix version exists, use that
        if git_branch_exists(&branch, workspace_root) {
            return branch;
        }
    }

    branch
}

fn git_branch_exists(name: &str, workdir: &Path) -> bool {
    git_run(&["rev-parse", "--verify", name], workdir).is_ok()
}

pub fn run_agent<S: EventSink>(
    mut args: AgentRequest,
    models_root: &Path,
    mut sink: S,
) -> Result<AgentRunResult> {
    let model_path = find_model_in_cache_in(models_root, &args.model)?;
    let workdir = args
        .workdir
        .clone()
        .unwrap_or(std::env::current_dir().context("failed to get current working directory")?);
    let workdir_canonical = workdir
        .canonicalize()
        .with_context(|| format!("failed to resolve workdir {}", workdir.display()))?;
    // Anchor to the git project root so tools cannot escape the repository boundary.
    let workspace_root = find_git_root(&workdir_canonical).unwrap_or(workdir_canonical);

    let branch = determine_branch_for_request(&args, &workspace_root);

    let is_continuation = git_checkout_branch(&branch, &workspace_root).is_ok();
    if !is_continuation {
        git_create_branch(&branch, &workspace_root)
            .with_context(|| format!("failed to create branch '{branch}'"))?;
    }

    suppress_llama_logs();
    let mut backend = LlamaBackend::init().context("failed to initialize llama backend")?;
    backend.void_logs();
    let model_params = LlamaModelParams::default().with_n_gpu_layers(args.gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| format!("failed to load model {}", model_path.display()))?;

    if args.infer_profile {
        args.profile = infer_agent_profile(&backend, &model, &args, &args.task)?;
        args.infer_profile = false;
    }

    // Load environment config (explicit arg takes precedence over file on disk).
    let env_config = args.environment.clone().or_else(|| {
        if args.profile == AgentProfile::Scout {
            crate::init::scout_environment(&workspace_root)
                .ok()
                .flatten()
        } else {
            EnvironmentConfig::load(&workspace_root).ok().flatten()
        }
    });

    // If an environment is configured, prepare the requested command backend for this task.
    let command_backend = if let Some(ref config) = env_config {
        Some(CommandBackend::start(config, &workspace_root)?)
    } else {
        None
    };

    let user_config =
        crate::config::UserConfig::load().context("failed to load user MCP config")?;
    let project_mcp_config = mcp::ProjectMcpConfig::load(&workspace_root)
        .context("failed to load project MCP config")?;
    let mcp_servers = mcp::effective_servers(&user_config.mcp, project_mcp_config.as_ref());
    let mcp_registry = mcp::discover_tools(mcp_servers);

    sink.emit(AgentEvent::Started {
        task: args.task.clone(),
        model: model_path.display().to_string(),
        workspace: workspace_root.display().to_string(),
        branch: branch.clone(),
    });

    let instructions = build_agent_instructions(
        &workspace_root,
        &branch,
        is_continuation,
        command_backend.as_ref().map(CommandBackend::kind),
        env_config.as_ref(),
        args.profile,
        args.sub_agent_depth < MAX_SUB_AGENT_DEPTH,
        &mcp_registry,
    )?;

    let todo_memory = RefCell::new(TodoMemory::default());

    let mut messages = vec![
        ChatMessage {
            role: "system",
            content: instructions,
        },
        ChatMessage {
            role: "user",
            content: args.task.clone(),
        },
    ];

    let outcome = run_agent_steps(
        &backend,
        &model,
        &args,
        &mut messages,
        &workspace_root,
        command_backend.as_ref(),
        env_config.as_ref(),
        &todo_memory,
        &mcp_registry,
        0,
        &mut sink,
    )?;
    let reached_final = outcome.reached_final;

    if git_has_changes(&workspace_root).unwrap_or(false) {
        let summary: String = args.task.chars().take(60).collect();
        let commit_msg = format!("refactor(agent): {summary}");
        let _ = git_commit_all(&commit_msg, &workspace_root);
    }

    let commits = git_log_recent(&workspace_root, 5).unwrap_or_default();
    sink.emit(AgentEvent::SessionSummary {
        branch: branch.clone(),
        commits,
    });

    // `command_backend` is dropped here, which removes task containers when used.

    Ok(AgentRunResult {
        branch,
        workspace_root,
        reached_final,
    })
}

fn build_agent_instructions(
    workspace_root: &Path,
    branch: &str,
    continuing: bool,
    command_backend_kind: Option<CommandBackendKind>,
    env_config: Option<&EnvironmentConfig>,
    profile: AgentProfile,
    allow_sub_agents: bool,
    mcp_registry: &McpToolRegistry,
) -> Result<String> {
    let mut instructions = String::from(
        "You are pb, a local coding agent. Always respond with one JSON object and nothing else.\n",
    );
    instructions.push_str(
        "Use {\"type\":\"tool_call\",\"tool\":\"...\",\"arguments\":{...},\"thinking\":\"...\"} for actions, or {\"type\":\"final\",\"content\":\"...\",\"thinking\":\"...\"} when done.\n",
    );
    instructions.push_str(profile.instructions());
    instructions.push('\n');
    let available_tools = available_tool_specs(
        profile,
        command_backend_kind,
        allow_sub_agents,
        mcp_registry,
    );
    let available_tool_signatures = available_tool_signatures(
        profile,
        command_backend_kind,
        allow_sub_agents,
        mcp_registry,
    );
    let tool_schema_json = serde_json::to_string_pretty(&available_tools)
        .context("failed to serialize tool schemas")?;
    instructions.push_str(&format!(
        "Available tools: {}.\n",
        available_tool_signatures.join(", ")
    ));
    instructions.push_str(
        "Tool schemas use the MCP tool shape with name, description, and inputSchema JSON Schema fields. Pass arguments that conform to the selected tool's inputSchema.\n",
    );
    instructions.push_str("Tool schemas:\n");
    instructions.push_str(&tool_schema_json);
    instructions.push('\n');
    if allow_sub_agents && profile != AgentProfile::Research {
        instructions.push_str(
            "Use sub_agent(profile,task,max_steps) to delegate bounded work into a fresh context. Supported profiles are explore, review, plan, ask, research, scout, and build. Launch a research sub-agent when you need external knowledge, current documentation, ecosystem context, or deeper source synthesis to make a better plan, answer a question, research a build failure, review risk, or implement a fix. The sub-agent result is summarized back to you so large investigation transcripts do not bloat your primary context.\n",
        );
    }
    if matches!(profile, AgentProfile::Build | AgentProfile::Scout) {
        instructions.push_str(
            "When editing, keep changes minimal and safe. Use edit_file for exact replacements, apply_patch(patch) for unified diffs, mv(source,destination) to rename files, and rm(path,recursive) to remove files or directories. Use git_commit with a semantic commit message after each logical change.\n",
        );
    } else {
        instructions.push_str(
            "This profile is read-only: do not call edit_file, apply_patch, mv, rm, git_commit, or git_revert.\n",
        );
    }
    instructions.push_str(
        "Use web_search for general internet research and web_fetch for reading a specific URL. Only use public http/https URLs.\n",
    );
    if !mcp_registry.is_empty() {
        instructions.push_str(
            "Configured MCP tools are exposed with mcp_<server>_<tool> names. Use them when they are the most direct way to access configured external context or services.\n",
        );
    }
    instructions.push_str(
        "Skills are discovered from repo Codex, Claude, OpenCode, and Copilot locations by metadata only. Use skill_search(query,max_results) to find relevant skills without loading full bodies, then skill(name) to load one selected skill when it applies. Build agents can use framework skills to improve implementation; plan agents can plan skill invocations; research agents can use research skills for targeted source gathering.\n",
    );
    instructions.push_str(&format!(
        "Reading and writing is only permitted within the project root: {}.\n",
        workspace_root.display()
    ));
    match command_backend_kind {
        Some(CommandBackendKind::Container) => instructions.push_str(
            "Use run_command(cmd) to execute shell commands inside the sandboxed container environment. The project root is mounted at /workspace inside the container.\n",
        ),
        Some(CommandBackendKind::Local) => instructions.push_str(
            "Use run_command(cmd) to execute shell commands locally from the project root on the host machine.\n",
        ),
        None => {}
    }
    if let Some(config) = env_config {
        if let Some(source) = &config.source {
            instructions.push_str("Scouted environment: ");
            instructions.push_str(source);
            instructions.push('\n');
        }
        if !config.guard_commands.is_empty() {
            instructions.push_str(
                "Commit guard commands to run before git_commit unless clearly impossible:\n",
            );
            for cmd in &config.guard_commands {
                instructions.push_str("- ");
                instructions.push_str(cmd);
                instructions.push('\n');
            }
        }
    }

    if let Ok(copilot_instructions) =
        std::fs::read_to_string(workspace_root.join(".github/copilot-instructions.md"))
    {
        instructions.push_str("Repository instructions:\n");
        instructions.push_str(&copilot_instructions);
        instructions.push('\n');
    }

    if continuing {
        instructions.push_str(&format!(
            "You are continuing work on branch '{branch}'. Review the recent commits below before proceeding.\n"
        ));
        match git_log_recent(workspace_root, 10) {
            Ok(log) if !log.is_empty() => {
                instructions.push_str("Recent commits:\n");
                instructions.push_str(&log);
                instructions.push('\n');
            }
            _ => {}
        }
    } else {
        instructions.push_str(&format!("You are working on branch '{branch}'.\n"));
    }

    Ok(instructions)
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct BuiltInToolSchema {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

fn available_tool_specs(
    profile: AgentProfile,
    command_backend_kind: Option<CommandBackendKind>,
    allow_sub_agents: bool,
    mcp_registry: &McpToolRegistry,
) -> Vec<BuiltInToolSchema> {
    let mut tools: Vec<_> = all_builtin_tool_specs()
        .into_iter()
        .filter(|tool| tool_allowed(&tool.name, profile, command_backend_kind, allow_sub_agents))
        .collect();
    tools.extend(mcp_registry.tools.values().map(|tool| BuiltInToolSchema {
        name: tool.tool_name.clone(),
        description: format!(
            "{} (MCP server: {}, tool: {})",
            tool.description, tool.server_name, tool.server_tool_name
        ),
        input_schema: tool.input_schema.clone(),
    }));
    tools
}

fn available_tool_signatures(
    profile: AgentProfile,
    command_backend_kind: Option<CommandBackendKind>,
    allow_sub_agents: bool,
    mcp_registry: &McpToolRegistry,
) -> Vec<String> {
    available_tool_specs(
        profile,
        command_backend_kind,
        allow_sub_agents,
        mcp_registry,
    )
    .into_iter()
    .map(|tool| tool.signature())
    .collect()
}

impl BuiltInToolSchema {
    fn signature(&self) -> String {
        let signature = match self.name.as_str() {
            "read_file" => "read_file(path,start,end)",
            "glob" => "glob(pattern,path,max_results)",
            "ripgrep" => "ripgrep(pattern,path,max_results)",
            "search" => "search(pattern,path)",
            "web_search" => "web_search(query)",
            "web_fetch" => "web_fetch(url)",
            "git_log" => "git_log()",
            "todo" => "todo(action,id,title,description,status,parent_id,note)",
            "skill_search" => "skill_search(query,max_results)",
            "skill" => "skill(name)",
            "ask_user" => "ask_user(question)",
            "run_command" => "run_command(cmd)",
            "edit_file" => "edit_file(path,old_text,new_text)",
            "apply_patch" => "apply_patch(patch)",
            "mv" => "mv(source,destination)",
            "rm" => "rm(path,recursive)",
            "git_commit" => "git_commit(message)",
            "git_revert" => "git_revert(commit)",
            "sub_agent" => "sub_agent(profile,task,max_steps)",
            _ => return format!("{}(arguments)", self.name),
        };
        signature.to_string()
    }
}

fn all_builtin_tool_specs() -> Vec<BuiltInToolSchema> {
    vec![
        builtin_tool(
            "read_file",
            "Read a UTF-8 text file inside the project root, optionally limiting the returned line range.",
            object_schema(
                [
                    string_property("path", "Project-relative file path to read."),
                    integer_property("start", "First 1-indexed line to include; defaults to 1."),
                    integer_property(
                        "end",
                        "Last 1-indexed line to include; defaults to the end of the file.",
                    ),
                ],
                ["path"],
            ),
        ),
        builtin_tool(
            "glob",
            "List project files matching a glob pattern.",
            object_schema(
                [
                    string_property("pattern", "Glob pattern to match."),
                    string_property(
                        "path",
                        "Optional project-relative directory to search within.",
                    ),
                    integer_property("max_results", "Maximum number of matches to return."),
                ],
                ["pattern"],
            ),
        ),
        builtin_tool(
            "ripgrep",
            "Search project files with ripgrep-compatible regular expressions.",
            object_schema(
                [
                    string_property("pattern", "Regular expression to search for."),
                    string_property(
                        "path",
                        "Optional project-relative file or directory to search within.",
                    ),
                    integer_property("max_results", "Maximum number of matches to return."),
                ],
                ["pattern"],
            ),
        ),
        builtin_tool(
            "search",
            "Alias for ripgrep; search project files with a regular expression.",
            object_schema(
                [
                    string_property("pattern", "Regular expression to search for."),
                    string_property(
                        "path",
                        "Optional project-relative file or directory to search within.",
                    ),
                    integer_property("max_results", "Maximum number of matches to return."),
                ],
                ["pattern"],
            ),
        ),
        builtin_tool(
            "web_search",
            "Search the public web for current or external information.",
            object_schema([string_property("query", "Search query.")], ["query"]),
        ),
        builtin_tool(
            "web_fetch",
            "Fetch and extract text from a public http or https URL.",
            object_schema(
                [string_property("url", "Public http or https URL to fetch.")],
                ["url"],
            ),
        ),
        builtin_tool(
            "git_log",
            "Show recent commits in the project repository.",
            object_schema([], []),
        ),
        builtin_tool(
            "todo",
            "Manage shared agent todo memory.",
            object_schema(
                [
                    enum_property(
                        "action",
                        "Todo operation to perform.",
                        [
                            "list", "next", "add", "update", "complete", "block", "start",
                        ],
                    ),
                    integer_property(
                        "id",
                        "Todo id for update, complete, block, or start actions.",
                    ),
                    string_property("title", "Todo title for add or update actions."),
                    string_property("description", "Todo description for add or update actions."),
                    enum_property(
                        "status",
                        "Todo status for update actions.",
                        ["pending", "in_progress", "completed", "blocked"],
                    ),
                    integer_property("parent_id", "Optional parent todo id for add actions."),
                    string_property("note", "Optional note to append while updating a todo."),
                ],
                [],
            ),
        ),
        builtin_tool(
            "skill_search",
            "Search discovered skill metadata without loading full skill bodies.",
            object_schema(
                [
                    string_property(
                        "query",
                        "Skill search query; empty string lists broadly relevant skills.",
                    ),
                    integer_property("max_results", "Maximum number of skills to return."),
                ],
                [],
            ),
        ),
        builtin_tool(
            "skill",
            "Load the body of a selected skill by name.",
            object_schema(
                [string_property(
                    "name",
                    "Skill name returned by skill_search.",
                )],
                ["name"],
            ),
        ),
        builtin_tool(
            "ask_user",
            "Ask the human user a blocking clarification question.",
            object_schema(
                [string_property(
                    "question",
                    "Question to present to the user.",
                )],
                ["question"],
            ),
        ),
        builtin_tool(
            "run_command",
            "Execute a shell command in the configured project environment.",
            object_schema(
                [string_property(
                    "cmd",
                    "Shell command to execute from the project root.",
                )],
                ["cmd"],
            ),
        ),
        builtin_tool(
            "edit_file",
            "Replace an exact text occurrence in a project file.",
            object_schema(
                [
                    string_property("path", "Project-relative file path to edit."),
                    string_property("old_text", "Exact text to replace."),
                    string_property("new_text", "Replacement text."),
                ],
                ["path", "old_text", "new_text"],
            ),
        ),
        builtin_tool(
            "apply_patch",
            "Apply a unified diff patch to project files.",
            object_schema(
                [string_property("patch", "Unified diff patch text.")],
                ["patch"],
            ),
        ),
        builtin_tool(
            "mv",
            "Move or rename a file or directory inside the project root.",
            object_schema(
                [
                    string_property("source", "Existing project-relative source path."),
                    string_property("destination", "New project-relative destination path."),
                ],
                ["source", "destination"],
            ),
        ),
        builtin_tool(
            "rm",
            "Remove a file or directory inside the project root.",
            object_schema(
                [
                    string_property("path", "Project-relative path to remove."),
                    boolean_property("recursive", "Whether to recursively remove a directory."),
                ],
                ["path"],
            ),
        ),
        builtin_tool(
            "git_commit",
            "Commit all current project changes with a semantic commit message.",
            object_schema(
                [string_property("message", "Semantic commit message.")],
                ["message"],
            ),
        ),
        builtin_tool(
            "git_revert",
            "Revert a commit by hash or revision.",
            object_schema(
                [string_property(
                    "commit",
                    "Commit hash or revision to revert.",
                )],
                ["commit"],
            ),
        ),
        builtin_tool(
            "sub_agent",
            "Delegate bounded work to another agent profile in a fresh context.",
            object_schema(
                [
                    enum_property(
                        "profile",
                        "Profile for the delegated agent.",
                        [
                            "explore", "review", "plan", "ask", "research", "scout", "build",
                        ],
                    ),
                    string_property("task", "Concrete task for the delegated agent."),
                    integer_property(
                        "max_steps",
                        "Maximum tool/final iterations for the delegated agent.",
                    ),
                ],
                ["profile", "task"],
            ),
        ),
    ]
}

fn builtin_tool(
    name: &'static str,
    description: &'static str,
    input_schema: Value,
) -> BuiltInToolSchema {
    BuiltInToolSchema {
        name: name.to_string(),
        description: description.to_string(),
        input_schema,
    }
}

fn object_schema<const P: usize, const R: usize>(
    properties: [(&'static str, Value); P],
    required: [&'static str; R],
) -> Value {
    let mut property_map = Map::new();
    for (name, schema) in properties {
        property_map.insert(name.to_string(), schema);
    }
    json!({
        "type": "object",
        "properties": property_map,
        "required": required.to_vec(),
        "additionalProperties": false,
    })
}

fn string_property(name: &'static str, description: &'static str) -> (&'static str, Value) {
    (
        name,
        json!({ "type": "string", "description": description }),
    )
}

fn integer_property(name: &'static str, description: &'static str) -> (&'static str, Value) {
    (
        name,
        json!({ "type": "integer", "description": description, "minimum": 1 }),
    )
}

fn boolean_property(name: &'static str, description: &'static str) -> (&'static str, Value) {
    (
        name,
        json!({ "type": "boolean", "description": description }),
    )
}

fn enum_property<const N: usize>(
    name: &'static str,
    description: &'static str,
    values: [&'static str; N],
) -> (&'static str, Value) {
    (
        name,
        json!({ "type": "string", "description": description, "enum": values.to_vec() }),
    )
}

fn tool_allowed(
    tool: &str,
    profile: AgentProfile,
    command_backend_kind: Option<CommandBackendKind>,
    allow_sub_agents: bool,
) -> bool {
    match tool {
        "read_file" | "glob" | "ripgrep" | "search" | "web_search" | "web_fetch" | "git_log"
        | "todo" | "skill_search" | "skill" => true,
        "ask_user" => profile == AgentProfile::Plan,
        "run_command" => command_backend_kind.is_some(),
        "edit_file" | "apply_patch" | "mv" | "rm" | "git_commit" | "git_revert" => {
            matches!(profile, AgentProfile::Build | AgentProfile::Scout)
        }
        "sub_agent" => allow_sub_agents && profile != AgentProfile::Research,
        _ => false,
    }
}

#[derive(Debug, Clone)]
struct StepRunOutcome {
    reached_final: bool,
    final_content: Option<String>,
}

fn run_agent_steps(
    backend: &LlamaBackend,
    model: &LlamaModel,
    args: &AgentRequest,
    messages: &mut Vec<ChatMessage>,
    workspace_root: &Path,
    command_backend: Option<&CommandBackend>,
    env_config: Option<&EnvironmentConfig>,
    todo_memory: &RefCell<TodoMemory>,
    mcp_registry: &McpToolRegistry,
    nesting_depth: usize,
    sink: &mut dyn EventSink,
) -> Result<StepRunOutcome> {
    for step in 1..=args.max_steps {
        sink.emit(AgentEvent::StepStarted {
            step,
            max_steps: args.max_steps,
        });

        let prompt = render_prompt(messages);
        let output = generate_completion(backend, model, args, &prompt)?;
        let action = parse_action(&output)?;

        match action {
            AgentAction::Final { content, thinking } => {
                if let Some(reasoning) = thinking {
                    sink.emit(AgentEvent::Reasoning {
                        content: reasoning,
                        profile: args.profile,
                    });
                }
                sink.emit(AgentEvent::Final {
                    content: content.clone(),
                    profile: args.profile,
                });
                return Ok(StepRunOutcome {
                    reached_final: true,
                    final_content: Some(content),
                });
            }
            AgentAction::ToolCall {
                tool,
                arguments,
                thinking,
            } => {
                if let Some(reasoning) = thinking {
                    sink.emit(AgentEvent::Reasoning {
                        content: reasoning,
                        profile: args.profile,
                    });
                }
                sink.emit(AgentEvent::ToolCall {
                    tool: tool.clone(),
                    arguments: arguments.clone(),
                });
                let tool_context = ToolContext {
                    backend,
                    model,
                    request: args,
                    workspace_root,
                    command_backend,
                    env_config,
                    todo_memory,
                    mcp_registry,
                };
                let tool_result = run_tool(&tool, &arguments, &tool_context, sink)?;
                sink.emit(AgentEvent::ToolResult {
                    tool: tool.clone(),
                    result: tool_result.clone(),
                });

                messages.push(ChatMessage {
                    role: "assistant",
                    content: output,
                });
                messages.push(ChatMessage {
                    role: "tool",
                    content: format!("tool={tool}\nargs={arguments}\nresult={tool_result}"),
                });
            }
        }
    }

    Ok(StepRunOutcome {
        reached_final: false,
        final_content: None,
    })
}

fn render_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    prompt.push_str("<conversation>\n");
    for message in messages {
        prompt.push('[');
        prompt.push_str(message.role);
        prompt.push_str("]\n");
        prompt.push_str(&message.content);
        prompt.push_str("\n\n");
    }
    prompt.push_str("[assistant]\n");
    prompt
}

fn generate_completion(
    backend: &LlamaBackend,
    model: &LlamaModel,
    args: &AgentRequest,
    prompt: &str,
) -> Result<String> {
    let n_ctx = NonZeroU32::new(args.ctx_size).context("ctx-size must be > 0")?;
    let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
    if let Some(threads) = args.threads {
        ctx_params = ctx_params.with_n_threads(threads);
    }
    if let Some(threads_batch) = args.threads_batch.or(args.threads) {
        ctx_params = ctx_params.with_n_threads_batch(threads_batch);
    }

    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("failed to create llama context")?;

    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| "failed to tokenize prompt")?;

    ensure_prompt_fits_context(tokens.len(), args.max_tokens, ctx.n_ctx())?;

    let mut batch = LlamaBatch::new(LLAMA_BATCH_SIZE, 1);
    for range in prompt_batch_ranges(tokens.len(), LLAMA_BATCH_SIZE) {
        batch.clear();
        let is_final_batch = range.end == tokens.len();
        for token_index in range.clone() {
            let is_last_prompt_token = is_final_batch && token_index + 1 == tokens.len();
            batch
                .add(tokens[token_index], token_index as i32, &[0], is_last_prompt_token)
                .with_context(|| {
                    format!(
                        "failed to add prompt token {token_index} to batch (batch capacity: {LLAMA_BATCH_SIZE}, prompt tokens: {})",
                        tokens.len()
                    )
                })?;
        }

        ctx.decode(&mut batch).with_context(|| {
            format!(
                "failed to decode prompt batch {}..{}",
                range.start, range.end
            )
        })?;
    }

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::top_k(args.top_k),
        LlamaSampler::temp(args.temperature),
        LlamaSampler::dist(args.seed),
    ]);

    let mut decoder = UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = i32::try_from(tokens.len()).context("prompt token count exceeds i32::MAX")?;
    let mut generated_tokens = 0;

    while generated_tokens < args.max_tokens {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .context("failed to decode output token")?;
        output.push_str(&piece);

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .context("failed to queue generated token")?;
        ctx.decode(&mut batch)
            .context("failed to decode generated token")?;
        n_cur += 1;
        generated_tokens += 1;
    }

    Ok(output)
}

fn ensure_prompt_fits_context(prompt_tokens: usize, max_tokens: i32, n_ctx: u32) -> Result<()> {
    let n_ctx = usize::try_from(n_ctx).context("context size does not fit usize")?;
    let requested_generation_tokens = usize::try_from(max_tokens.max(0))
        .context("requested generation token count does not fit usize")?;
    let reserved_generation_tokens = requested_generation_tokens.max(MIN_GENERATION_CONTEXT_TOKENS);
    if prompt_tokens + reserved_generation_tokens > n_ctx {
        bail!(
            "prompt is too long for the configured context: {prompt_tokens} prompt tokens + {reserved_generation_tokens} reserved generation tokens exceeds ctx-size {n_ctx}. Increase --ctx-size or reduce the task/history size."
        );
    }
    Ok(())
}

fn prompt_batch_ranges(token_count: usize, batch_size: usize) -> Vec<Range<usize>> {
    assert!(batch_size > 0, "batch_size must be greater than zero");
    (0..token_count)
        .step_by(batch_size)
        .map(|start| start..std::cmp::min(start + batch_size, token_count))
        .collect()
}

fn parse_action(output: &str) -> Result<AgentAction> {
    if let Ok(action) = serde_json::from_str::<AgentAction>(output.trim()) {
        return Ok(action);
    }

    let json_candidate = extract_json_object(output)
        .with_context(|| format!("model output did not contain a valid JSON action:\n{output}"))?;
    serde_json::from_str::<AgentAction>(&json_candidate)
        .with_context(|| format!("failed to parse agent JSON action:\n{json_candidate}"))
}

fn extract_json_object(input: &str) -> Option<String> {
    let mut start = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;

    for (i, ch) in input.char_indices() {
        if start.is_none() {
            if ch == '{' {
                start = Some(i);
                depth = 1;
            }
            continue;
        }

        if in_string {
            if escape {
                escape = false;
                continue;
            }
            if ch == '\\' {
                escape = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let s = start?;
                    return Some(input[s..=i].to_string());
                }
            }
            _ => {}
        }
    }

    if let Some(s) = start
        && depth > 0
        && !in_string
    {
        let mut candidate = input[s..].trim_end().to_string();
        candidate.extend(std::iter::repeat_n('}', depth));
        return Some(candidate);
    }

    None
}

struct ToolContext<'a> {
    backend: &'a LlamaBackend,
    model: &'a LlamaModel,
    request: &'a AgentRequest,
    workspace_root: &'a Path,
    command_backend: Option<&'a CommandBackend>,
    env_config: Option<&'a EnvironmentConfig>,
    todo_memory: &'a RefCell<TodoMemory>,
    mcp_registry: &'a McpToolRegistry,
}

fn tool_result_limit(arguments: &Value, tool: &str, default_limit: usize) -> Result<usize> {
    let Some(value) = arguments.get("max_results") else {
        return Ok(default_limit);
    };
    let requested = value
        .as_u64()
        .with_context(|| format!("{tool} max_results must be an integer"))?
        as usize;
    Ok(requested.clamp(1, default_limit))
}

fn workspace_walk(root: &Path) -> WalkBuilder {
    let mut builder = WalkBuilder::new(root);
    builder.hidden(false).filter_entry(|entry| {
        !SEARCH_EXCLUDED_DIRS
            .iter()
            .any(|excluded| entry.file_name() == std::ffi::OsStr::new(excluded))
    });
    builder
}

fn run_glob(
    pattern: &str,
    relative_path: Option<&str>,
    limit: usize,
    workspace_root: &Path,
) -> Result<String> {
    let search_root = if let Some(path) = relative_path {
        resolve_workspace_path(workspace_root, path, true)?
    } else {
        workspace_root.to_path_buf()
    };
    let matcher = GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .with_context(|| format!("invalid glob pattern: {pattern}"))?
        .compile_matcher();

    let mut matches = Vec::new();
    for entry in workspace_walk(&search_root).build() {
        let entry = entry.with_context(|| format!("failed to walk {}", search_root.display()))?;
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }
        let path = entry.path();
        let rel = path.strip_prefix(workspace_root).unwrap_or(path);
        let file_name = path.file_name().unwrap_or_default();
        if matcher.is_match(rel) || matcher.is_match(Path::new(file_name)) {
            matches.push(rel.display().to_string());
            if matches.len() >= limit {
                break;
            }
        }
    }

    if matches.is_empty() {
        Ok("no matches".to_string())
    } else {
        Ok(matches.join("\n"))
    }
}

fn run_ripgrep(
    pattern: &str,
    relative_path: Option<&str>,
    limit: usize,
    workspace_root: &Path,
) -> Result<String> {
    let search_root = if let Some(path) = relative_path {
        resolve_workspace_path(workspace_root, path, true)?
    } else {
        workspace_root.to_path_buf()
    };
    let matcher = RegexMatcher::new_line_matcher(pattern)
        .with_context(|| format!("invalid regex pattern: {pattern}"))?;
    let mut searcher = Searcher::new();
    let mut hits = Vec::new();

    for entry in workspace_walk(&search_root).build() {
        let entry = entry.with_context(|| format!("failed to walk {}", search_root.display()))?;
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }
        let path = entry.path();
        let rel = path
            .strip_prefix(workspace_root)
            .unwrap_or(path)
            .to_path_buf();
        let result = searcher.search_path(
            matcher.clone(),
            path,
            GrepUtf8(|line_number, line| {
                hits.push(format!("{}:{}:{}", rel.display(), line_number, line.trim()));
                Ok::<bool, std::io::Error>(hits.len() < limit)
            }),
        );
        match result {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => continue,
            Err(error) => {
                return Err(anyhow!(error))
                    .with_context(|| format!("failed to search {}", path.display()));
            }
        }
        if hits.len() >= limit {
            break;
        }
    }

    if hits.is_empty() {
        Ok("no matches".to_string())
    } else {
        Ok(hits.join("\n"))
    }
}

fn run_tool(
    tool: &str,
    arguments: &Value,
    context: &ToolContext<'_>,
    sink: &mut dyn EventSink,
) -> Result<String> {
    if context.mcp_registry.tool(tool).is_some() {
        return mcp::call_tool(context.mcp_registry, tool, arguments);
    }
    if !tool_allowed(
        tool,
        context.request.profile,
        context.command_backend.map(CommandBackend::kind),
        context.request.sub_agent_depth < MAX_SUB_AGENT_DEPTH,
    ) {
        bail!(
            "tool '{tool}' is not available for the {} profile",
            context.request.profile.as_str()
        );
    }
    let workspace_root = context.workspace_root;
    let command_backend = context.command_backend;
    match tool {
        "read_file" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .context("read_file requires string argument: path")?;
            let start = arguments.get("start").and_then(Value::as_u64).unwrap_or(1) as usize;
            let end = arguments.get("end").and_then(Value::as_u64);
            let resolved = resolve_workspace_path(workspace_root, path, true)?;
            let text = match std::fs::read_to_string(&resolved) {
                Ok(t) => t,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(format!("file not found: {}", resolved.display()));
                }
                Err(e) => {
                    return Err(anyhow!(e))
                        .context(format!("failed to read {}", resolved.display()));
                }
            };

            let lines: Vec<_> = text.lines().collect();
            if let Some(end) = end
                && (end as usize) < start
            {
                return Ok("(no content in requested range)".to_string());
            }
            let end_line = end.map_or(lines.len(), |v| v as usize).max(start);
            let mut out = String::new();
            for (idx, line) in lines
                .iter()
                .enumerate()
                .take(lines.len().min(end_line))
                .skip(start.saturating_sub(1))
            {
                out.push_str(&format!("{}: {}\n", idx + 1, line));
            }
            if out.is_empty() {
                out.push_str("(no content in requested range)");
            }
            Ok(out)
        }
        "glob" => {
            let pattern = arguments
                .get("pattern")
                .and_then(Value::as_str)
                .context("glob requires string argument: pattern")?;
            let relative_path = arguments.get("path").and_then(Value::as_str);
            let limit = tool_result_limit(arguments, "glob", MAX_GLOB_RESULTS)?;
            run_glob(pattern, relative_path, limit, workspace_root)
        }
        "ripgrep" | "search" => {
            let pattern = arguments
                .get("pattern")
                .and_then(Value::as_str)
                .with_context(|| format!("{tool} requires string argument: pattern"))?;
            let relative_path = arguments.get("path").and_then(Value::as_str);
            let limit = tool_result_limit(arguments, tool, MAX_SEARCH_RESULTS)?;
            run_ripgrep(pattern, relative_path, limit, workspace_root)
        }
        "edit_file" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .context("edit_file requires string argument: path")?;
            let old_text = arguments
                .get("old_text")
                .and_then(Value::as_str)
                .context("edit_file requires string argument: old_text")?;
            let new_text = arguments
                .get("new_text")
                .and_then(Value::as_str)
                .context("edit_file requires string argument: new_text")?;

            let resolved = resolve_workspace_path(workspace_root, path, true)?;
            let existing = std::fs::read_to_string(&resolved)
                .with_context(|| format!("failed to read {}", resolved.display()))?;

            if !existing.contains(old_text) {
                bail!("old_text not found in file");
            }

            let updated = existing.replacen(old_text, new_text, 1);
            std::fs::write(&resolved, &updated)
                .with_context(|| format!("failed to write {}", resolved.display()))?;

            let diff = unified_diff(&existing, &updated, path);
            sink.emit(AgentEvent::Diff {
                path: path.to_string(),
                diff,
            });
            Ok(format!("updated {}", resolved.display()))
        }
        "apply_patch" => {
            let patch = arguments
                .get("patch")
                .and_then(Value::as_str)
                .context("apply_patch requires string argument: patch")?;
            let changed_paths = validate_patch_paths(patch, workspace_root)?;
            run_git_apply_patch(patch, workspace_root)?;
            let diff = git_diff_paths(workspace_root, &changed_paths)?;
            if !diff.trim().is_empty() {
                sink.emit(AgentEvent::Diff {
                    path: "apply_patch".to_string(),
                    diff,
                });
            }
            Ok(format!("applied patch to {}", changed_paths.join(", ")))
        }
        "mv" => {
            let source = arguments
                .get("source")
                .and_then(Value::as_str)
                .context("mv requires string argument: source")?;
            let destination = arguments
                .get("destination")
                .and_then(Value::as_str)
                .context("mv requires string argument: destination")?;
            let source_path = resolve_workspace_path(workspace_root, source, true)?;
            let destination_path = resolve_workspace_path(workspace_root, destination, false)?;
            if source_path == workspace_root {
                bail!("mv cannot move the workspace root");
            }
            if destination_path.exists() {
                bail!(
                    "mv destination already exists: {}",
                    destination_path.display()
                );
            }
            std::fs::rename(&source_path, &destination_path).with_context(|| {
                format!(
                    "failed to move {} to {}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
            Ok(format!(
                "moved {} to {}",
                source_path.display(),
                destination_path.display()
            ))
        }
        "rm" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .context("rm requires string argument: path")?;
            let recursive = arguments
                .get("recursive")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let resolved = resolve_workspace_path(workspace_root, path, true)?;
            if resolved == workspace_root {
                bail!("rm cannot remove the workspace root");
            }
            let metadata = std::fs::symlink_metadata(&resolved)
                .with_context(|| format!("failed to stat {}", resolved.display()))?;
            if metadata.is_dir() {
                if recursive {
                    std::fs::remove_dir_all(&resolved)
                } else {
                    std::fs::remove_dir(&resolved)
                }
            } else {
                std::fs::remove_file(&resolved)
            }
            .with_context(|| format!("failed to remove {}", resolved.display()))?;
            Ok(format!("removed {}", resolved.display()))
        }
        "web_search" => {
            let query = arguments
                .get("query")
                .and_then(Value::as_str)
                .context("web_search requires string argument: query")?;
            block_on_tool(run_web_search(query))
        }
        "web_fetch" => {
            let url = arguments
                .get("url")
                .and_then(Value::as_str)
                .context("web_fetch requires string argument: url")?;
            block_on_tool(run_web_fetch(url))
        }
        "git_commit" => {
            let message = arguments
                .get("message")
                .and_then(Value::as_str)
                .context("git_commit requires string argument: message")?;
            if let Some(config) = context.env_config {
                run_guard_commands(config, command_backend, workspace_root)?;
            }
            match git_commit_all(message, workspace_root)? {
                true => Ok(format!("committed: {message}")),
                false => Ok("nothing to commit".to_string()),
            }
        }
        "git_log" => {
            let log = git_log_recent(workspace_root, 10)?;
            if log.is_empty() {
                Ok("no commits yet".to_string())
            } else {
                Ok(log)
            }
        }
        "todo" => run_todo_tool(arguments, context.todo_memory),
        "git_revert" => {
            let commit = arguments
                .get("commit")
                .and_then(Value::as_str)
                .context("git_revert requires string argument: commit")?;
            git_revert(commit, workspace_root)
        }
        "skill_search" => {
            let query = arguments.get("query").and_then(Value::as_str).unwrap_or("");
            let limit = tool_result_limit(arguments, "skill_search", MAX_SKILL_SEARCH_RESULTS)?;
            run_skill_search(query, limit, workspace_root)
        }
        "skill" => {
            let name = arguments
                .get("name")
                .and_then(Value::as_str)
                .context("skill requires string argument: name")?;
            run_skill_tool(name, workspace_root)
        }
        "ask_user" => {
            let question = arguments
                .get("question")
                .and_then(Value::as_str)
                .context("ask_user requires string argument: question")?;
            sink.ask_user(question)
        }
        "sub_agent" => run_sub_agent(arguments, context, sink),
        "run_command" => {
            let cmd = arguments
                .get("cmd")
                .and_then(Value::as_str)
                .context("run_command requires string argument: cmd")?;
            let backend = command_backend
                .context("run_command is not available: no project environment is configured")?;
            backend.exec(cmd)
        }
        _ => bail!("unknown tool: {tool}"),
    }
}

#[derive(Default)]
struct SubAgentEventCollector {
    final_content: Option<String>,
    errors: Vec<String>,
    diffs: usize,
}

impl SubAgentEventCollector {
    fn collect(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Final {
                content,
                profile: _,
            } => self.final_content = Some(content),
            AgentEvent::Error { message } => self.errors.push(message),
            AgentEvent::Diff { .. } => self.diffs += 1,
            _ => {}
        }
    }
}

impl EventSink for SubAgentEventCollector {
    fn emit(&mut self, event: AgentEvent) {
        self.collect(event);
    }
}

struct SubAgentSink<'a> {
    parent: &'a mut dyn EventSink,
    collector: SubAgentEventCollector,
}

impl EventSink for SubAgentSink<'_> {
    fn emit(&mut self, event: AgentEvent) {
        self.collector.collect(event);
    }

    fn ask_user(&mut self, question: &str) -> Result<String> {
        self.parent.ask_user(question)
    }
}

fn run_guard_commands(
    config: &EnvironmentConfig,
    command_backend: Option<&CommandBackend>,
    workspace_root: &Path,
) -> Result<()> {
    for cmd in &config.guard_commands {
        match command_backend {
            Some(backend) => {
                backend
                    .exec(cmd)
                    .with_context(|| format!("commit guard command failed: {cmd}"))?;
            }
            None => {
                run_local_shell_command(cmd, workspace_root)
                    .with_context(|| format!("commit guard command failed: {cmd}"))?;
            }
        }
    }
    Ok(())
}

fn run_sub_agent(
    arguments: &Value,
    context: &ToolContext<'_>,
    sink: &mut dyn EventSink,
) -> Result<String> {
    let profile_name = arguments
        .get("profile")
        .and_then(Value::as_str)
        .context("sub_agent requires string argument: profile")?;
    let profile = AgentProfile::parse(profile_name)?;
    let task = arguments
        .get("task")
        .and_then(Value::as_str)
        .context("sub_agent requires string argument: task")?;
    let max_steps = arguments
        .get("max_steps")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_SUB_AGENT_MAX_STEPS)
        .clamp(1, context.request.max_steps.max(1));

    sink.emit(AgentEvent::SubAgentStarted {
        profile: profile.as_str().to_string(),
        task: task.to_string(),
        nesting_depth: context.request.sub_agent_depth + 1,
    });

    let instructions = build_agent_instructions(
        context.workspace_root,
        context.request.branch.as_deref().unwrap_or("sub-agent"),
        true,
        context.command_backend.map(CommandBackend::kind),
        context.env_config,
        profile,
        false,
        context.mcp_registry,
    )?;
    let mut messages = vec![
        ChatMessage {
            role: "system",
            content: instructions,
        },
        ChatMessage {
            role: "user",
            content: task.to_string(),
        },
    ];

    let mut sub_request = context.request.clone();
    sub_request.task = task.to_string();
    sub_request.profile = profile;
    sub_request.max_steps = max_steps;
    sub_request.sub_agent_depth = context.request.sub_agent_depth + 1;

    let outcome = run_agent_steps(
        context.backend,
        context.model,
        &sub_request,
        &mut messages,
        context.workspace_root,
        context.command_backend,
        context.env_config,
        context.todo_memory,
        context.mcp_registry,
        context.request.sub_agent_depth + 1,
        sink,
    )?;

    let result = if outcome.reached_final {
        "sub-agent completed successfully".to_string()
    } else {
        "sub-agent reached its step limit before finalizing".to_string()
    };

    sink.emit(AgentEvent::SubAgentFinished {
        profile: profile.as_str().to_string(),
        result: result.clone(),
    });
    Ok(result)
}

fn run_todo_tool(arguments: &Value, todo_memory: &RefCell<TodoMemory>) -> Result<String> {
    let action = arguments
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("list")
        .trim()
        .to_ascii_lowercase();

    match action.as_str() {
        "list" => {
            let memory = todo_memory.borrow();
            format_todo_tasks(&memory.tasks)
        }
        "next" => {
            let memory = todo_memory.borrow();
            let pending = memory.pending_tasks();
            if pending.is_empty() {
                Ok("no pending todos".to_string())
            } else {
                serde_json::to_string_pretty(&pending).context("failed to serialize pending todos")
            }
        }
        "add" => {
            let title = arguments
                .get("title")
                .and_then(Value::as_str)
                .context("todo add requires string argument: title")?
                .trim();
            if title.is_empty() {
                bail!("todo add title must not be empty");
            }
            let description = arguments
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            let parent_id = arguments
                .get("parent_id")
                .and_then(Value::as_u64)
                .map(|value| value as usize);
            let mut memory = todo_memory.borrow_mut();
            if let Some(parent_id) = parent_id
                && !memory.tasks.iter().any(|task| task.id == parent_id)
            {
                bail!("parent todo id {parent_id} was not found");
            }
            let task = memory
                .add(title.to_string(), description, parent_id)
                .clone();
            serde_json::to_string_pretty(&json!({ "added": task }))
                .context("failed to serialize added todo")
        }
        "update" | "complete" | "block" | "start" => {
            let id = arguments
                .get("id")
                .and_then(Value::as_u64)
                .context("todo update/complete requires integer argument: id")?
                as usize;
            let status = match action.as_str() {
                "complete" => Some(TodoStatus::Completed),
                "block" => Some(TodoStatus::Blocked),
                "start" => Some(TodoStatus::InProgress),
                _ => arguments
                    .get("status")
                    .and_then(Value::as_str)
                    .map(TodoStatus::parse)
                    .transpose()?,
            };
            let title = arguments
                .get("title")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
            let description = arguments
                .get("description")
                .and_then(Value::as_str)
                .map(str::trim)
                .map(str::to_string);
            let note = arguments
                .get("note")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
            let mut memory = todo_memory.borrow_mut();
            let task = memory.update(id, status, title, description, note)?.clone();
            serde_json::to_string_pretty(&json!({ "updated": task }))
                .context("failed to serialize updated todo")
        }
        other => bail!(
            "unknown todo action '{other}'; expected one of: list, next, add, update, start, complete, block"
        ),
    }
}

fn format_todo_tasks(tasks: &[TodoTask]) -> Result<String> {
    if tasks.is_empty() {
        return Ok("no todos".to_string());
    }
    serde_json::to_string_pretty(tasks).context("failed to serialize todos")
}

fn validate_patch_paths(patch: &str, workspace_root: &Path) -> Result<Vec<String>> {
    if patch.trim().is_empty() {
        bail!("apply_patch patch must not be empty");
    }

    let mut paths = Vec::<String>::new();
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            let mut parts = rest.split_whitespace();
            if let Some(path) = parts.next() {
                collect_patch_path(path, workspace_root, &mut paths)?;
            }
            if let Some(path) = parts.next() {
                collect_patch_path(path, workspace_root, &mut paths)?;
            }
        } else if let Some(path) = line.strip_prefix("--- ") {
            collect_patch_path(path, workspace_root, &mut paths)?;
        } else if let Some(path) = line.strip_prefix("+++ ") {
            collect_patch_path(path, workspace_root, &mut paths)?;
        } else if let Some(path) = line.strip_prefix("*** Add File: ") {
            collect_patch_path(path, workspace_root, &mut paths)?;
        } else if let Some(path) = line.strip_prefix("*** Update File: ") {
            collect_patch_path(path, workspace_root, &mut paths)?;
        } else if let Some(path) = line.strip_prefix("*** Delete File: ") {
            collect_patch_path(path, workspace_root, &mut paths)?;
        } else if let Some(path) = line.strip_prefix("*** Move to: ") {
            collect_patch_path(path, workspace_root, &mut paths)?;
        }
    }

    if paths.is_empty() {
        bail!("apply_patch patch does not declare any file paths");
    }
    Ok(paths)
}

fn collect_patch_path(path: &str, workspace_root: &Path, paths: &mut Vec<String>) -> Result<()> {
    let path = path.split_whitespace().next().unwrap_or(path).trim();
    if path.is_empty() || path == "/dev/null" {
        return Ok(());
    }
    let path = path
        .strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path);
    let resolved = resolve_workspace_path(workspace_root, path, false)?;
    if resolved == workspace_root {
        bail!("patch path targets the workspace root");
    }
    if !paths.iter().any(|existing| existing == path) {
        paths.push(path.to_string());
    }
    Ok(())
}

fn run_git_apply_patch(patch: &str, workspace_root: &Path) -> Result<()> {
    git_apply_stdin(&["apply", "--check", "-"], patch, workspace_root)?;
    git_apply_stdin(&["apply", "-"], patch, workspace_root)?;
    Ok(())
}

fn git_apply_stdin(args: &[&str], input: &str, workdir: &Path) -> Result<String> {
    let mut child = Command::new("git")
        .args(args)
        .current_dir(workdir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to run git apply")?;
    child
        .stdin
        .as_mut()
        .context("failed to open git apply stdin")?
        .write_all(input.as_bytes())
        .context("failed to write patch to git apply")?;
    let output = child
        .wait_with_output()
        .context("failed to wait for git apply")?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        bail!("git {} failed: {}", args.join(" "), stderr)
    }
}

fn git_diff_paths(workdir: &Path, paths: &[String]) -> Result<String> {
    let mut command = Command::new("git");
    command
        .arg("diff")
        .arg("--")
        .args(paths)
        .current_dir(workdir);
    let output = command.output().context("failed to run git diff")?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        bail!("git diff failed: {stderr}")
    }
}

fn lexical_normalize(path: &Path) -> Result<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            std::path::Component::RootDir => normalized.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::Normal(part) => normalized.push(part),
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    bail!("path escapes workspace root: {}", path.display());
                }
            }
        }
    }
    Ok(normalized)
}

fn resolve_workspace_path(workspace_root: &Path, input: &str, must_exist: bool) -> Result<PathBuf> {
    let workspace_root = workspace_root.canonicalize().with_context(|| {
        format!(
            "failed to resolve workspace root {}",
            workspace_root.display()
        )
    })?;
    let candidate = if Path::new(input).is_absolute() {
        PathBuf::from(input)
    } else {
        workspace_root.join(input)
    };
    let candidate = lexical_normalize(&candidate)?;

    let normalized = if must_exist {
        candidate
            .canonicalize()
            .with_context(|| format!("failed to resolve path {}", candidate.display()))?
    } else {
        let mut ancestor = candidate.as_path();
        while !ancestor.exists() {
            ancestor = ancestor.parent().with_context(|| {
                format!("failed to resolve ancestor for {}", candidate.display())
            })?;
        }
        let ancestor_canonical = ancestor
            .canonicalize()
            .with_context(|| format!("failed to resolve ancestor {}", ancestor.display()))?;
        let suffix = candidate
            .strip_prefix(ancestor)
            .with_context(|| format!("failed to normalize path {}", candidate.display()))?;
        ancestor_canonical.join(suffix)
    };

    if !normalized.starts_with(&workspace_root) {
        bail!(
            "path escapes workspace root: {} not under {}",
            normalized.display(),
            workspace_root.display()
        );
    }

    Ok(normalized)
}

fn unified_diff(old: &str, new: &str, path: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut out = format!("--- a/{path}\n+++ b/{path}\n");
    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            similar::ChangeTag::Delete => "-",
            similar::ChangeTag::Insert => "+",
            similar::ChangeTag::Equal => " ",
        };
        out.push_str(sign);
        out.push_str(change.value());
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SkillMetadata {
    provider: &'static str,
    name: String,
    description: String,
    relative_path: String,
    kind: SkillKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkillKind {
    AgentSkill,
    CopilotInstructions,
    CopilotInstructionFile,
    CopilotPromptFile,
}

impl SkillKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::AgentSkill => "agent_skill",
            Self::CopilotInstructions => "copilot_instructions",
            Self::CopilotInstructionFile => "copilot_instruction_file",
            Self::CopilotPromptFile => "copilot_prompt_file",
        }
    }
}

fn run_skill_search(query: &str, limit: usize, workspace_root: &Path) -> Result<String> {
    let mut skills = discover_skills(workspace_root)?;
    let query = query.trim().to_ascii_lowercase();
    if !query.is_empty() {
        let terms: Vec<&str> = query.split_whitespace().collect();
        skills.retain(|skill| skill_matches(skill, &terms));
        skills.sort_by_key(|skill| std::cmp::Reverse(skill_score(skill, &terms)));
    }

    if skills.is_empty() {
        return Ok("no matching skills found in repo skill locations".to_string());
    }

    let mut out = String::from(
        "Skill metadata search results (full skill bodies are not loaded until skill(name) is called):\n",
    );
    for skill in skills.iter().take(limit) {
        out.push_str(&format!(
            "- name: {}\n  provider: {}\n  kind: {}\n  path: {}\n  description: {}\n",
            skill.name,
            skill.provider,
            skill.kind.as_str(),
            skill.relative_path,
            skill.description
        ));
    }
    Ok(out)
}

fn run_skill_tool(name: &str, workspace_root: &Path) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        bail!("skill name must not be empty");
    }

    let skills = discover_skills(workspace_root)?;
    if name.eq_ignore_ascii_case("list") {
        return format_skill_list(&skills);
    }

    let matches: Vec<_> = skills
        .iter()
        .filter(|skill| skill_identifier_matches(skill, name))
        .collect();

    match matches.as_slice() {
        [] => Ok(format!(
            "unknown skill '{name}'. Use skill_search(query,max_results) or skill(name=\"list\") to find available repo skills."
        )),
        [skill] => load_skill_body(skill, workspace_root),
        many => {
            let choices = many
                .iter()
                .map(|skill| {
                    format!(
                        "{}/{} ({})",
                        skill.provider, skill.name, skill.relative_path
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            Ok(format!(
                "skill name '{name}' is ambiguous. Invoke one of:\n{choices}"
            ))
        }
    }
}

fn format_skill_list(skills: &[SkillMetadata]) -> Result<String> {
    if skills.is_empty() {
        return Ok("no repo skills found".to_string());
    }
    let mut out = String::from("Available repo skills:\n");
    for skill in skills {
        out.push_str(&format!(
            "- {}/{} — {} ({})\n",
            skill.provider, skill.name, skill.description, skill.relative_path
        ));
    }
    Ok(out)
}

fn skill_identifier_matches(skill: &SkillMetadata, requested: &str) -> bool {
    let requested = requested.trim();
    skill.name.eq_ignore_ascii_case(requested)
        || skill
            .relative_path
            .eq_ignore_ascii_case(requested.trim_start_matches("./"))
        || format!("{}/{}", skill.provider, skill.name).eq_ignore_ascii_case(requested)
}

fn load_skill_body(skill: &SkillMetadata, workspace_root: &Path) -> Result<String> {
    let resolved = resolve_workspace_path(workspace_root, &skill.relative_path, true)?;
    let text = std::fs::read_to_string(&resolved)
        .with_context(|| format!("failed to read skill {}", resolved.display()))?;
    let resource_hint = skill_resource_hint(&resolved, workspace_root)?;
    let truncated = truncate_chars(&text, MAX_SKILL_TEXT_CHARS);
    let truncation_note = if text.chars().count() > MAX_SKILL_TEXT_CHARS {
        format!("\n\n[truncated to {MAX_SKILL_TEXT_CHARS} characters]")
    } else {
        String::new()
    };
    Ok(format!(
        "Skill: {}\nProvider: {}\nKind: {}\nPath: {}\nDescription: {}\n{}\n---\n{}{}",
        skill.name,
        skill.provider,
        skill.kind.as_str(),
        skill.relative_path,
        skill.description,
        resource_hint,
        truncated,
        truncation_note
    ))
}

fn skill_resource_hint(skill_path: &Path, workspace_root: &Path) -> Result<String> {
    let Some(skill_dir) = skill_path.parent() else {
        return Ok("Resources: none".to_string());
    };
    let mut resources = Vec::new();
    for resource_dir in ["scripts", "references", "assets", "agents"] {
        let dir = skill_dir.join(resource_dir);
        if !dir.is_dir() {
            continue;
        }
        for entry in workspace_walk(&dir).max_depth(Some(3)).build() {
            let entry = entry.with_context(|| format!("failed to walk {}", dir.display()))?;
            if entry
                .file_type()
                .is_some_and(|file_type| file_type.is_file())
            {
                let path = entry
                    .path()
                    .strip_prefix(workspace_root)
                    .unwrap_or(entry.path());
                resources.push(path.display().to_string());
                if resources.len() >= 30 {
                    break;
                }
            }
        }
    }
    if resources.is_empty() {
        Ok("Resources: none".to_string())
    } else {
        Ok(format!(
            "Resources (load with read_file only as needed):\n- {}",
            resources.join("\n- ")
        ))
    }
}

fn discover_skills(workspace_root: &Path) -> Result<Vec<SkillMetadata>> {
    let mut skills = Vec::new();
    for entry in workspace_walk(workspace_root).build() {
        let entry =
            entry.with_context(|| format!("failed to walk {}", workspace_root.display()))?;
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }
        let path = entry.path();
        let rel = path.strip_prefix(workspace_root).unwrap_or(path);
        let rel_string = rel.display().to_string();
        if let Some(skill) = parse_repo_skill_file(path, rel, &rel_string)? {
            skills.push(skill);
        }
    }
    skills.sort_by(|left, right| {
        left.provider
            .cmp(right.provider)
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.relative_path.cmp(&right.relative_path))
    });
    Ok(skills)
}

fn parse_repo_skill_file(
    path: &Path,
    rel: &Path,
    rel_string: &str,
) -> Result<Option<SkillMetadata>> {
    let components = rel_components(rel);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");

    if file_name == "SKILL.md" {
        if let Some(provider) = agent_skill_provider(&components) {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read skill metadata {}", path.display()))?;
            let (metadata, _) = parse_markdown_frontmatter(&text);
            let fallback_name = path
                .parent()
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                .unwrap_or("skill")
                .to_string();
            return Ok(Some(SkillMetadata {
                provider,
                name: metadata.get("name").cloned().unwrap_or(fallback_name),
                description: metadata
                    .get("description")
                    .or_else(|| metadata.get("summary"))
                    .cloned()
                    .unwrap_or_else(|| first_heading_or_default(&text, "Agent skill")),
                relative_path: rel_string.to_string(),
                kind: SkillKind::AgentSkill,
            }));
        }
    }

    if components == [".github", "copilot-instructions.md"] {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read skill metadata {}", path.display()))?;
        return Ok(Some(SkillMetadata {
            provider: "copilot",
            name: "copilot-instructions".to_string(),
            description: first_heading_or_default(&text, "Repository Copilot custom instructions"),
            relative_path: rel_string.to_string(),
            kind: SkillKind::CopilotInstructions,
        }));
    }

    if components.first().is_some_and(|part| *part == ".github")
        && components.iter().any(|part| *part == "instructions")
        && file_name.ends_with(".instructions.md")
    {
        return parse_copilot_markdown_skill(
            path,
            rel_string,
            SkillKind::CopilotInstructionFile,
            "Copilot instruction file",
        )
        .map(Some);
    }

    if components.first().is_some_and(|part| *part == ".github")
        && components.iter().any(|part| *part == "prompts")
        && file_name.ends_with(".prompt.md")
    {
        return parse_copilot_markdown_skill(
            path,
            rel_string,
            SkillKind::CopilotPromptFile,
            "Copilot prompt file",
        )
        .map(Some);
    }

    Ok(None)
}

fn parse_copilot_markdown_skill(
    path: &Path,
    rel_string: &str,
    kind: SkillKind,
    default_description: &str,
) -> Result<SkillMetadata> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read skill metadata {}", path.display()))?;
    let (metadata, _) = parse_markdown_frontmatter(&text);
    let name = metadata.get("name").cloned().unwrap_or_else(|| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("copilot-skill")
            .trim_end_matches(".instructions")
            .trim_end_matches(".prompt")
            .to_string()
    });
    let description = metadata
        .get("description")
        .or_else(|| metadata.get("summary"))
        .or_else(|| metadata.get("applyTo"))
        .cloned()
        .unwrap_or_else(|| first_heading_or_default(&text, default_description));
    Ok(SkillMetadata {
        provider: "copilot",
        name,
        description,
        relative_path: rel_string.to_string(),
        kind,
    })
}

fn agent_skill_provider(components: &[&str]) -> Option<&'static str> {
    if path_contains_sequence(components, &[".codex", "skills"]) {
        return Some("codex");
    }
    if path_contains_sequence(components, &[".agents", "skills"]) {
        return Some("codex");
    }
    if path_contains_sequence(components, &[".claude", "skills"]) {
        return Some("claude");
    }
    if path_contains_sequence(components, &[".opencode", "skill"])
        || path_contains_sequence(components, &[".opencode", "skills"])
    {
        return Some("opencode");
    }
    None
}

fn path_contains_sequence(components: &[&str], sequence: &[&str]) -> bool {
    !sequence.is_empty()
        && components
            .windows(sequence.len())
            .any(|window| window == sequence)
}

fn rel_components(path: &Path) -> Vec<&str> {
    path.components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => part.to_str(),
            _ => None,
        })
        .collect()
}

fn parse_markdown_frontmatter(text: &str) -> (HashMap<String, String>, &str) {
    let mut metadata = HashMap::new();
    let Some(rest) = text.strip_prefix("---") else {
        return (metadata, text);
    };
    let rest = rest
        .strip_prefix('\n')
        .or_else(|| rest.strip_prefix("\r\n"));
    let Some(rest) = rest else {
        return (metadata, text);
    };
    let mut body_start = None;
    let mut offset = text.len() - rest.len();
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed == "---" || trimmed == "..." {
            body_start = Some(offset + line.len());
            break;
        }
        if !line.starts_with(' ')
            && !line.starts_with('\t')
            && let Some((key, value)) = line.split_once(':')
        {
            let key = key.trim();
            if !key.is_empty() {
                metadata.insert(key.to_string(), clean_frontmatter_value(value));
            }
        }
        offset += line.len();
    }
    if let Some(start) = body_start {
        (metadata, &text[start..])
    } else {
        (HashMap::new(), text)
    }
}

fn clean_frontmatter_value(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch| ch == '"' || ch == '\'')
        .to_string()
}

fn first_heading_or_default(text: &str, default: &str) -> String {
    text.lines()
        .find_map(|line| line.trim().strip_prefix("# "))
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .unwrap_or(default)
        .to_string()
}

fn skill_matches(skill: &SkillMetadata, terms: &[&str]) -> bool {
    let haystack = format!(
        "{} {} {} {} {}",
        skill.provider,
        skill.name,
        skill.description,
        skill.relative_path,
        skill.kind.as_str()
    )
    .to_ascii_lowercase();
    terms.iter().all(|term| haystack.contains(term))
}

fn skill_score(skill: &SkillMetadata, terms: &[&str]) -> usize {
    let name = skill.name.to_ascii_lowercase();
    let description = skill.description.to_ascii_lowercase();
    let path = skill.relative_path.to_ascii_lowercase();
    terms
        .iter()
        .map(|term| {
            let mut score = 0;
            if name == *term {
                score += 20;
            }
            if name.contains(term) {
                score += 10;
            }
            if description.contains(term) {
                score += 5;
            }
            if path.contains(term) {
                score += 2;
            }
            score
        })
        .sum()
}

fn block_on_tool<F>(future: F) -> Result<String>
where
    F: Future<Output = Result<String>>,
{
    tool_runtime()?.block_on(future)
}

fn tool_runtime() -> Result<&'static tokio::runtime::Runtime> {
    static RUNTIME: OnceLock<std::result::Result<tokio::runtime::Runtime, String>> =
        OnceLock::new();
    match RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to start runtime for web tool")
            .map_err(|error| error.to_string())
    }) {
        Ok(runtime) => Ok(runtime),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

fn http_client() -> Result<&'static reqwest::Client> {
    static CLIENT: OnceLock<std::result::Result<reqwest::Client, String>> = OnceLock::new();
    match CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(TOOL_USER_AGENT)
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(20))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .context("failed to build web client")
            .map_err(|error| error.to_string())
    }) {
        Ok(client) => Ok(client),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

async fn run_web_search(query: &str) -> Result<String> {
    let query = query.trim();
    if query.is_empty() {
        bail!("web_search query must not be empty");
    }

    let response = http_client()?
        .get("https://duckduckgo.com/html/")
        .query(&[("q", query)])
        .send()
        .await
        .with_context(|| format!("failed to search the web for '{query}'"))?;
    let body = read_response_text(response).await?;
    let results = parse_duckduckgo_results(&body);

    if results.is_empty() {
        return Ok(format!("No public web results found for: {query}"));
    }

    let mut out = format!("Web search results for: {query}\n");
    for (index, result) in results.iter().take(MAX_WEB_SEARCH_RESULTS).enumerate() {
        out.push_str(&format!(
            "{}. {}\n   URL: {}\n",
            index + 1,
            result.title,
            result.url
        ));
    }
    Ok(out)
}

async fn run_web_fetch(url: &str) -> Result<String> {
    let url = parse_public_web_url(url)?;
    let response = http_client()?
        .get(url.clone())
        .send()
        .await
        .with_context(|| format!("failed to fetch {}", url))?;
    let final_url = response.url().clone();
    validate_public_web_url(&final_url)?;
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    let body = read_response_text(response).await?;
    let content = normalize_web_content(&body, &content_type);

    Ok(format!(
        "Fetched: {final_url}\nContent-Type: {content_type}\n\n{}",
        truncate_chars(&content, MAX_WEB_RESULT_CHARS)
    ))
}

async fn read_response_text(response: reqwest::Response) -> Result<String> {
    let status = response.status();
    if !status.is_success() {
        bail!("request failed with status {status}");
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_WEB_RESPONSE_BYTES as u64)
    {
        bail!("response exceeded {} bytes", MAX_WEB_RESPONSE_BYTES);
    }

    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("failed to read response body")?;
        if body.len() >= MAX_WEB_RESPONSE_BYTES {
            break;
        }
        let remaining = MAX_WEB_RESPONSE_BYTES - body.len();
        let take = remaining.min(chunk.len());
        body.extend_from_slice(&chunk[..take]);
    }

    Ok(String::from_utf8_lossy(&body).into_owned())
}

fn parse_public_web_url(input: &str) -> Result<Url> {
    let url = Url::parse(input).with_context(|| format!("invalid URL: {input}"))?;
    validate_public_web_url(&url)?;
    Ok(url)
}

fn validate_public_web_url(url: &Url) -> Result<()> {
    if !matches!(url.scheme(), "http" | "https") {
        bail!("only http and https URLs are supported");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("URLs with embedded credentials are not allowed");
    }
    let host = url.host_str().context("URL is missing a host")?;
    if host.eq_ignore_ascii_case("localhost")
        || host.ends_with(".localhost")
        || host.eq_ignore_ascii_case("local")
        || host.ends_with(".local")
    {
        bail!("local network URLs are not allowed");
    }
    if let Ok(ip) = host.parse::<IpAddr>()
        && is_private_ip(ip)
    {
        bail!("private or loopback IP URLs are not allowed");
    }
    Ok(())
}

fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.is_unspecified()
                || is_shared_v4(ip)
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
                || is_documentation_v6(ip)
        }
    }
}

fn is_shared_v4(ip: Ipv4Addr) -> bool {
    matches!(ip.octets(), [100, second_octet, ..] if (64..=127).contains(&second_octet))
}

fn is_documentation_v6(ip: Ipv6Addr) -> bool {
    matches!(ip.segments(), [0x2001, 0x0db8, ..])
}

fn normalize_web_content(body: &str, content_type: &str) -> String {
    if content_type.contains("html") {
        let text = html_to_text(body);
        if text.is_empty() {
            "(empty HTML response)".to_string()
        } else {
            text
        }
    } else {
        let text = body.trim();
        if text.is_empty() {
            "(empty response body)".to_string()
        } else {
            text.to_string()
        }
    }
}

fn parse_duckduckgo_results(html: &str) -> Vec<WebSearchResult> {
    static RESULT_RE: OnceLock<regex::Regex> = OnceLock::new();
    let regex = RESULT_RE.get_or_init(|| {
        regex::Regex::new(
            r#"(?is)<a[^>]*class=['"][^'"]*(?:result__a|result-link)[^'"]*['"][^>]*href=['"]([^'"]+)['"][^>]*>(.*?)</a>"#,
        )
        .expect("valid search result regex")
    });

    let mut results = Vec::new();
    for capture in regex.captures_iter(html) {
        let Some(raw_url) = capture.get(1).map(|m| m.as_str()) else {
            continue;
        };
        let Some(raw_title) = capture.get(2).map(|m| m.as_str()) else {
            continue;
        };
        let title = html_to_text(raw_title);
        let url = normalize_search_result_url(raw_url);
        if title.is_empty() || url.is_empty() {
            continue;
        }
        if results
            .iter()
            .any(|existing: &WebSearchResult| existing.url == url)
        {
            continue;
        }
        results.push(WebSearchResult { title, url });
        if results.len() >= MAX_WEB_SEARCH_RESULTS {
            break;
        }
    }
    results
}

fn normalize_search_result_url(raw_url: &str) -> String {
    let decoded = decode_html_entities(raw_url.trim());
    let joined = match Url::parse(&decoded) {
        Ok(url) => url,
        Err(_) => {
            match Url::parse("https://duckduckgo.com/").and_then(|base| base.join(&decoded)) {
                Ok(url) => url,
                Err(_) => return decoded,
            }
        }
    };

    if joined
        .host_str()
        .is_some_and(|host| host.ends_with("duckduckgo.com"))
        && let Some(target) = joined
            .query_pairs()
            .find(|(key, _)| key == "uddg")
            .map(|(_, value)| value.into_owned())
    {
        return target;
    }

    joined.to_string()
}

fn html_to_text(html: &str) -> String {
    static SCRIPT_STYLE_RE: OnceLock<regex::Regex> = OnceLock::new();
    static TAG_RE: OnceLock<regex::Regex> = OnceLock::new();
    let script_style = SCRIPT_STYLE_RE.get_or_init(|| {
        regex::Regex::new(r"(?is)<(script|style)[^>]*>.*?</(script|style)>")
            .expect("valid script/style regex")
    });
    let tag = TAG_RE.get_or_init(|| regex::Regex::new(r"(?is)<[^>]+>").expect("valid tag regex"));

    let without_scripts = script_style.replace_all(html, " ");
    let without_tags = tag.replace_all(&without_scripts, " ");
    collapse_whitespace(&decode_html_entities(&without_tags))
}

fn decode_html_entities(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index] == b'&'
            && let Some(end) = input[index..].find(';')
        {
            let entity = &input[index + 1..index + end];
            if let Some(decoded) = decode_html_entity(entity) {
                output.push(decoded);
                index += end + 1;
                continue;
            }
        }
        if let Some(ch) = input[index..].chars().next() {
            output.push(ch);
            index += ch.len_utf8();
        } else {
            break;
        }
    }

    output
}

fn decode_html_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" | "#39" => Some('\''),
        "nbsp" => Some(' '),
        _ => {
            let number = entity
                .strip_prefix("#x")
                .or_else(|| entity.strip_prefix("#X"));
            if let Some(number) = number {
                u32::from_str_radix(number, 16)
                    .ok()
                    .and_then(char::from_u32)
            } else if let Some(number) = entity.strip_prefix('#') {
                number.parse::<u32>().ok().and_then(char::from_u32)
            } else {
                None
            }
        }
    }
}

fn collapse_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    let truncated: String = input.chars().take(max_chars).collect();
    if input.chars().count() > max_chars {
        format!("{truncated}\n\n[truncated]")
    } else {
        truncated
    }
}

pub fn branch_name_from_task(task: &str) -> String {
    let slug: String = task
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let truncated: String = slug.chars().take(50).collect();
    format!("pb/{truncated}")
}

fn git_run(args: &[&str], workdir: &Path) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(workdir)
        .output()
        .context("failed to run git")?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        bail!("git {} failed: {}", args.join(" "), stderr)
    }
}

fn git_create_branch(name: &str, workdir: &Path) -> Result<()> {
    git_run(&["checkout", "-b", name], workdir)?;
    Ok(())
}

fn git_checkout_branch(name: &str, workdir: &Path) -> Result<()> {
    git_run(&["checkout", name], workdir)?;
    Ok(())
}

fn git_has_changes(workdir: &Path) -> Result<bool> {
    let out = git_run(&["status", "--porcelain"], workdir)?;
    Ok(!out.is_empty())
}

fn git_commit_all(message: &str, workdir: &Path) -> Result<bool> {
    if !git_has_changes(workdir)? {
        return Ok(false);
    }
    git_run(&["add", "-A"], workdir)?;
    git_run(&["commit", "-m", message], workdir)?;
    Ok(true)
}

fn git_log_recent(workdir: &Path, n: usize) -> Result<String> {
    git_run(&["log", "--oneline", &format!("-{n}")], workdir)
}

fn git_revert(commit: &str, workdir: &Path) -> Result<String> {
    let commit = commit.trim();
    git_run(&["revert", "--no-edit", commit], workdir)?;
    Ok(format!("reverted commit: {commit}"))
}

pub fn find_model_in_cache_in(pull_root: &Path, model: &str) -> Result<PathBuf> {
    let model_dir = pull_root.join(crate::cache_dir_name(model));

    if !model_dir.exists() {
        bail!(
            "model '{}' not found in pull cache. Run: pb pull {}",
            model,
            model
        );
    }

    const GGUF_MAGIC: &[u8] = b"GGUF";
    let mut gguf_files: Vec<PathBuf> = std::fs::read_dir(&model_dir)
        .with_context(|| format!("failed to read model directory {}", model_dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            let mut buf = [0u8; 4];
            std::fs::File::open(p)
                .and_then(|mut f| {
                    use std::io::Read;
                    f.read_exact(&mut buf)
                })
                .map(|_| buf == GGUF_MAGIC)
                .unwrap_or(false)
        })
        .collect();

    if gguf_files.is_empty() {
        bail!(
            "model '{}' cache is incomplete (no GGUF blobs found). Run: pb pull {}",
            model,
            model
        );
    }

    gguf_files.retain(|p| std::fs::metadata(p).is_ok());
    if gguf_files.is_empty() {
        bail!(
            "model '{}' cache is incomplete (GGUF blobs are inaccessible). Run: pb pull {}",
            model,
            model
        );
    }
    gguf_files
        .sort_by_key(|p| std::cmp::Reverse(std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)));

    Ok(gguf_files.into_iter().next().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_object_handles_noise() {
        let output = "hello {\"type\":\"final\",\"content\":\"ok\"} trailing";
        let extracted = extract_json_object(output).expect("json should be extracted");
        assert_eq!(extracted, "{\"type\":\"final\",\"content\":\"ok\"}");
    }

    #[test]
    fn parse_action_repairs_missing_closing_brace() {
        let output = r#"{"type":"tool_call","tool":"git_revert","arguments":{"commit":"a707c16"},"thinking":"I will revert this commit.""#;
        let action = parse_action(output).expect("truncated JSON action should be repaired");

        let AgentAction::ToolCall {
            tool, arguments, ..
        } = action
        else {
            panic!("expected tool call");
        };
        assert_eq!(tool, "git_revert");
        assert_eq!(arguments["commit"], "a707c16");
    }

    #[test]
    fn prompt_batch_ranges_splits_prompts_larger_than_batch_capacity() {
        assert_eq!(
            prompt_batch_ranges(1_025, 512),
            vec![0..512, 512..1_024, 1_024..1_025]
        );
    }

    #[test]
    fn ensure_prompt_fits_context_allows_generation_room() {
        ensure_prompt_fits_context(8_000, 128, 8_192).unwrap();
    }

    #[test]
    fn ensure_prompt_fits_context_rejects_overflow_with_actionable_message() {
        let err = ensure_prompt_fits_context(8_100, 128, 8_192)
            .unwrap_err()
            .to_string();

        assert!(err.contains("prompt is too long"), "error was: {err}");
        assert!(err.contains("--ctx-size"), "error was: {err}");
    }

    #[test]
    fn profile_inference_prompt_describes_profiles_and_task() {
        let prompt = profile_inference_prompt("Fix the login bug");

        assert!(prompt.contains("Return exactly one JSON object"));
        assert!(prompt.contains("build"));
        assert!(prompt.contains("scout"));
        assert!(prompt.contains("review"));
        assert!(prompt.contains("explore"));
        assert!(prompt.contains("plan"));
        assert!(prompt.contains("ask"));
        assert!(prompt.contains("research"));
        assert!(prompt.contains("Fix the login bug"));
    }

    #[test]
    fn parse_inferred_agent_profile_accepts_json_and_plain_profile() {
        assert_eq!(
            parse_inferred_agent_profile(r#"{"profile":"build"}"#).unwrap(),
            AgentProfile::Build
        );
        assert_eq!(
            parse_inferred_agent_profile("plan").unwrap(),
            AgentProfile::Plan
        );
        assert_eq!(
            parse_inferred_agent_profile("```json\n{\"profile\":\"review\"}\n```").unwrap(),
            AgentProfile::Review
        );
        assert_eq!(
            parse_inferred_agent_profile(r#"{"profile":"research"}"#).unwrap(),
            AgentProfile::Research
        );
    }

    #[test]
    fn parse_inferred_agent_profile_rejects_unknown_output() {
        let err = parse_inferred_agent_profile("not a profile")
            .unwrap_err()
            .to_string();

        assert!(err.contains("expected JSON object"), "error was: {err}");
    }

    #[test]
    fn local_backend_instructions_describe_host_commands() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let instructions = build_agent_instructions(
            tmp.path(),
            "test-branch",
            false,
            Some(CommandBackendKind::Local),
            None,
            AgentProfile::Build,
            true,
            &McpToolRegistry::default(),
        )
        .unwrap();
        assert!(instructions.contains("run_command(cmd)"));
        assert!(instructions.contains("locally from the project root on the host machine"));
    }

    #[test]
    fn build_profile_instructions_include_sub_agent_tool() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let instructions = build_agent_instructions(
            tmp.path(),
            "test-branch",
            false,
            Some(CommandBackendKind::Local),
            None,
            AgentProfile::Build,
            true,
            &McpToolRegistry::default(),
        )
        .unwrap();
        assert!(instructions.contains("Profile: build"));
        assert!(instructions.contains("sub_agent(profile,task,max_steps)"));
        assert!(instructions.contains("edit_file(path,old_text,new_text)"));
    }

    #[test]
    fn build_profile_instructions_include_mcp_shaped_tool_schemas() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let instructions = build_agent_instructions(
            tmp.path(),
            "test-branch",
            false,
            Some(CommandBackendKind::Local),
            None,
            AgentProfile::Build,
            true,
            &McpToolRegistry::default(),
        )
        .unwrap();
        assert!(instructions.contains("Tool schemas use the MCP tool shape"));
        assert!(instructions.contains(r#""inputSchema": {"#));
        assert!(instructions.contains(r#""name": "read_file""#));
        assert!(instructions.contains(r#""required": ["#));
        assert!(instructions.contains(r#""additionalProperties": false"#));
    }

    #[test]
    fn available_tool_specs_filter_by_profile_and_backend() {
        let review_tools = available_tool_specs(
            AgentProfile::Review,
            None,
            false,
            &McpToolRegistry::default(),
        );
        assert!(review_tools.iter().any(|tool| tool.name == "read_file"));
        assert!(!review_tools.iter().any(|tool| tool.name == "edit_file"));
        assert!(!review_tools.iter().any(|tool| tool.name == "run_command"));

        let build_tools = available_tool_specs(
            AgentProfile::Build,
            Some(CommandBackendKind::Local),
            true,
            &McpToolRegistry::default(),
        );
        let run_command = build_tools
            .iter()
            .find(|tool| tool.name == "run_command")
            .expect("run_command should be available with a backend");
        assert_eq!(run_command.input_schema["type"], "object");
        assert_eq!(run_command.input_schema["required"], json!(["cmd"]));
        assert!(build_tools.iter().any(|tool| tool.name == "sub_agent"));
    }

    #[test]
    fn review_profile_instructions_are_read_only() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let instructions = build_agent_instructions(
            tmp.path(),
            "test-branch",
            false,
            Some(CommandBackendKind::Local),
            None,
            AgentProfile::Review,
            false,
            &McpToolRegistry::default(),
        )
        .unwrap();
        assert!(instructions.contains("Profile: review"));
        assert!(instructions.contains("This profile is read-only"));
        assert!(!instructions.contains("edit_file(path,old_text,new_text)"));
        assert!(!instructions.contains("sub_agent(profile,task,max_steps)"));
    }

    #[test]
    fn non_research_profiles_can_delegate_to_research() {
        let tmp = tempfile::tempdir().expect("tempdir");
        for profile in [
            AgentProfile::Build,
            AgentProfile::Scout,
            AgentProfile::Review,
            AgentProfile::Explore,
            AgentProfile::Plan,
            AgentProfile::Ask,
        ] {
            let instructions = build_agent_instructions(
                tmp.path(),
                "test-branch",
                false,
                Some(CommandBackendKind::Local),
                None,
                profile,
                true,
                &McpToolRegistry::default(),
            )
            .unwrap();
            assert!(instructions.contains("sub_agent(profile,task,max_steps)"));
            assert!(instructions.contains("research"));
            assert!(tool_allowed("sub_agent", profile, None, true));
        }
    }

    #[test]
    fn research_profile_is_read_only_and_cannot_delegate() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let instructions = build_agent_instructions(
            tmp.path(),
            "test-branch",
            false,
            Some(CommandBackendKind::Local),
            None,
            AgentProfile::Research,
            true,
            &McpToolRegistry::default(),
        )
        .unwrap();
        assert!(instructions.contains("Profile: research"));
        assert!(instructions.contains("This profile is read-only"));
        assert!(!instructions.contains("edit_file(path,old_text,new_text)"));
        assert!(!instructions.contains("sub_agent(profile,task,max_steps)"));
        assert!(!tool_allowed(
            "sub_agent",
            AgentProfile::Research,
            None,
            true
        ));
    }

    #[test]
    fn todo_tool_adds_lists_and_completes_tasks() {
        let todo_memory = RefCell::new(TodoMemory::default());

        let added = run_todo_tool(
            &json!({
                "action": "add",
                "title": "Implement parser",
                "description": "Add task memory support"
            }),
            &todo_memory,
        )
        .unwrap();
        assert!(added.contains("Implement parser"));

        let pending = run_todo_tool(&json!({ "action": "next" }), &todo_memory).unwrap();
        assert!(pending.contains("pending"));

        let completed = run_todo_tool(
            &json!({
                "action": "complete",
                "id": 1,
                "note": "done by build sub-agent"
            }),
            &todo_memory,
        )
        .unwrap();
        assert!(completed.contains("completed"));
        assert!(completed.contains("done by build sub-agent"));

        let pending = run_todo_tool(&json!({ "action": "next" }), &todo_memory).unwrap();
        assert_eq!(pending, "no pending todos");
    }

    #[test]
    fn todo_tool_validates_parent_tasks() {
        let todo_memory = RefCell::new(TodoMemory::default());
        let err = run_todo_tool(
            &json!({
                "action": "add",
                "title": "Follow-up",
                "parent_id": 99
            }),
            &todo_memory,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("parent todo id 99 was not found"));
    }

    #[test]
    fn local_shell_command_runs_from_workspace_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("marker.txt"), "ok").unwrap();
        let output = run_local_shell_command("cat marker.txt", tmp.path()).unwrap();
        assert_eq!(output, "ok");
    }

    #[test]
    fn find_model_in_cache_missing_dir_suggests_pull() {
        let err = find_model_in_cache_in(Path::new("/tmp/pb-test-nonexistent-dir"), "mymodel")
            .unwrap_err()
            .to_string();
        assert!(err.contains("pb pull mymodel"), "error was: {err}");
    }

    #[test]
    fn find_model_in_cache_empty_dir_suggests_pull() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model_dir = tmp.path().join("mymodel");
        std::fs::create_dir_all(&model_dir).unwrap();
        let err = find_model_in_cache_in(tmp.path(), "mymodel")
            .unwrap_err()
            .to_string();
        assert!(err.contains("pb pull mymodel"), "error was: {err}");
    }

    #[test]
    fn find_model_in_cache_finds_gguf_by_magic() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model_dir = tmp.path().join("mymodel");
        std::fs::create_dir_all(&model_dir).unwrap();

        std::fs::write(model_dir.join("sha256_config"), b"{}").unwrap();
        let mut gguf_data = b"GGUF".to_vec();
        gguf_data.extend_from_slice(&[0u8; 16]);
        std::fs::write(model_dir.join("sha256_layer1"), &gguf_data).unwrap();

        let path = find_model_in_cache_in(tmp.path(), "mymodel").expect("should find GGUF");
        assert_eq!(path.file_name().unwrap(), "sha256_layer1");
    }

    #[test]
    fn find_model_in_cache_picks_largest_gguf() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model_dir = tmp.path().join("mymodel");
        std::fs::create_dir_all(&model_dir).unwrap();

        let small: Vec<u8> = b"GGUF".iter().chain(&[0u8; 4]).copied().collect();
        let large: Vec<u8> = b"GGUF".iter().chain(&[0u8; 100]).copied().collect();
        std::fs::write(model_dir.join("sha256_small"), &small).unwrap();
        std::fs::write(model_dir.join("sha256_large"), &large).unwrap();

        let path = find_model_in_cache_in(tmp.path(), "mymodel").expect("should find GGUF");
        assert_eq!(path.file_name().unwrap(), "sha256_large");
    }

    #[test]
    fn branch_name_from_task_basic() {
        assert_eq!(
            branch_name_from_task("Fix the login bug"),
            "pb/fix-the-login-bug"
        );
    }

    #[test]
    fn branch_name_from_task_special_chars() {
        assert_eq!(
            branch_name_from_task("Add feat: update foo/bar!"),
            "pb/add-feat-update-foo-bar"
        );
    }

    #[test]
    fn branch_name_from_task_truncates_at_50() {
        let long_task = "a".repeat(200);
        let name = branch_name_from_task(&long_task);
        assert!(name.len() <= "pb/".len() + 50);
        assert!(name.starts_with("pb/"));
    }

    #[test]
    fn branch_includes_session_id() {
        let args = AgentRequest {
            task: "Fix login bug".to_string(),
            model: "model.gguf".to_string(),
            model_dir: None,
            workdir: None,
            branch: None,
            max_steps: 10,
            max_tokens: 2048,
            ctx_size: 4096,
            threads: None,
            threads_batch: None,
            gpu_layers: 0,
            temperature: 0.7,
            profile: AgentProfile::Build,
            infer_profile: false,
            sub_agent_depth: 0,
            top_k: 40,
            seed: 42,
            environment: None,
            session_id: "session-123".to_string(),
        };

        let workdir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(workdir.path())
            .output()
            .unwrap();

        let branch = determine_branch_for_request(&args, workdir.path());
        assert_eq!(branch, "pb/fix-login-bug-session-123");
    }

    #[test]
    fn branch_defensive_checkout_existing_with_session_id() {
        let workdir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(workdir.path())
            .output()
            .unwrap();

        // First create a commit (required for branch)
        std::process::Command::new("git")
            .args(["commit", "--allow-empty", "-m", "initial"])
            .current_dir(workdir.path())
            .output()
            .unwrap();

        // Create a branch WITH session_id suffix (as would be created by new code)
        std::process::Command::new("git")
            .args(["checkout", "-b", "pb/another-task-session-456"])
            .current_dir(workdir.path())
            .output()
            .unwrap();

        let args = AgentRequest {
            task: "Another task".to_string(),
            model: "model.gguf".to_string(),
            model_dir: None,
            workdir: Some(workdir.path().to_path_buf()),
            branch: None,
            max_steps: 10,
            max_tokens: 2048,
            ctx_size: 4096,
            threads: None,
            threads_batch: None,
            gpu_layers: 0,
            temperature: 0.7,
            profile: AgentProfile::Build,
            infer_profile: false,
            sub_agent_depth: 0,
            top_k: 40,
            seed: 42,
            environment: None,
            session_id: "session-456".to_string(),
        };

        // Should find and use existing branch with same session_id
        let branch = determine_branch_for_request(&args, workdir.path());
        assert_eq!(
            branch, "pb/another-task-session-456",
            "Should find existing branch with matching session_id"
        );
    }

    #[test]
    fn git_commit_all_no_changes_returns_false() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(tmp.path())
            .output()
            .unwrap();
        let committed = git_commit_all("test commit", tmp.path()).unwrap();
        assert!(!committed);
    }

    #[test]
    fn git_revert_creates_revert_commit() {
        let tmp = tempfile::TempDir::new_in("/tmp").expect("tempdir");
        let dir = tmp.path();

        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir)
            .output()
            .unwrap();

        // Configure a minimal git identity so git commit works in CI.
        for (key, val) in [("user.email", "test@example.com"), ("user.name", "Test")] {
            std::process::Command::new("git")
                .args(["config", key, val])
                .current_dir(dir)
                .output()
                .unwrap();
        }

        // Create an initial commit so the repo has a HEAD.
        std::fs::write(dir.join("base.txt"), "base").unwrap();
        std::process::Command::new("git")
            .args(["add", "-A"])
            .current_dir(dir)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "initial"])
            .current_dir(dir)
            .output()
            .unwrap();

        // Create the commit we want to revert.
        std::fs::write(dir.join("change.txt"), "change").unwrap();
        git_commit_all("add change", dir).unwrap();

        // Capture the SHA we are reverting.
        let sha = git_run(&["rev-parse", "HEAD"], dir).unwrap();

        // Revert it.
        let result = git_revert(&sha, dir).unwrap();
        assert!(result.contains("reverted commit"), "result: {result}");

        // The reverted file should no longer exist (git revert removes it).
        assert!(!dir.join("change.txt").exists());
    }

    #[test]
    fn glob_finds_files_by_name_and_path() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src/nested")).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(root.join("src/nested/lib.rs"), "pub fn lib() {}\n").unwrap();
        std::fs::write(root.join("README.md"), "docs\n").unwrap();

        let by_path = run_glob("**/*.rs", None, MAX_GLOB_RESULTS, root).unwrap();
        assert!(by_path.contains("src/main.rs"), "result: {by_path}");
        assert!(by_path.contains("src/nested/lib.rs"), "result: {by_path}");

        let by_name = run_glob("README.md", None, MAX_GLOB_RESULTS, root).unwrap();
        assert_eq!(by_name, "README.md");
    }

    #[test]
    fn ripgrep_finds_regex_matches_with_locations() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/main.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .unwrap();
        std::fs::write(root.join("notes.txt"), "nothing here\n").unwrap();

        let result = run_ripgrep("println!", None, MAX_SEARCH_RESULTS, root).unwrap();
        assert_eq!(result, "src/main.rs:2:println!(\"hello\");");
    }

    #[test]
    fn ripgrep_limits_results() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::write(root.join("one.txt"), "needle\nneedle\n").unwrap();

        let result = run_ripgrep("needle", None, 1, root).unwrap();
        assert_eq!(result.lines().count(), 1);
    }

    #[test]
    fn parse_duckduckgo_results_extracts_titles_and_links() {
        let html = r#"
        <html>
          <body>
            <a class="result__a" href="https://example.com/one">First &amp; Result</a>
            <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ftwo">Second Result</a>
          </body>
        </html>
        "#;

        let results = parse_duckduckgo_results(html);

        assert_eq!(
            results,
            vec![
                WebSearchResult {
                    title: "First & Result".to_string(),
                    url: "https://example.com/one".to_string(),
                },
                WebSearchResult {
                    title: "Second Result".to_string(),
                    url: "https://example.com/two".to_string(),
                }
            ]
        );
    }

    #[test]
    fn parse_duckduckgo_results_deduplicates_urls() {
        let html = r#"
        <a class="result__a" href="https://example.com/one">First</a>
        <a class="result__a" href="https://example.com/one">Duplicate</a>
        "#;

        let results = parse_duckduckgo_results(html);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "First");
        assert_eq!(results[0].url, "https://example.com/one");
    }

    #[test]
    fn normalize_search_result_url_unwraps_duckduckgo_redirect() {
        let url = normalize_search_result_url(
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs",
        );
        assert_eq!(url, "https://example.com/docs");
    }

    #[test]
    fn html_to_text_strips_tags_and_decodes_entities() {
        let html = "<div>Hello <strong>world</strong> &amp; universe</div>";
        assert_eq!(html_to_text(html), "Hello world & universe");
    }

    #[test]
    fn validate_public_web_url_rejects_localhost() {
        let err = validate_public_web_url(&Url::parse("http://localhost:8080").unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("local network URLs are not allowed"));
    }

    #[test]
    fn validate_public_web_url_rejects_embedded_credentials() {
        let err = validate_public_web_url(&Url::parse("http://user@example.com").unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("embedded credentials"));
    }

    #[test]
    fn validate_public_web_url_rejects_shared_ipv4_range() {
        let err = validate_public_web_url(&Url::parse("https://100.64.0.1").unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("private or loopback IP"));
    }

    #[test]
    fn validate_public_web_url_allows_public_https() {
        validate_public_web_url(&Url::parse("https://example.com/docs").unwrap()).unwrap();
    }

    // -----------------------------------------------------------------------
    // find_git_root tests
    // -----------------------------------------------------------------------

    #[test]
    fn find_git_root_finds_direct_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join(".git")).unwrap();
        let result = find_git_root(tmp.path()).unwrap();
        assert_eq!(result, tmp.path());
    }

    #[test]
    fn find_git_root_finds_ancestor_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join(".git")).unwrap();
        let nested = tmp.path().join("a").join("b").join("c");
        std::fs::create_dir_all(&nested).unwrap();
        let result = find_git_root(&nested).unwrap();
        assert_eq!(result, tmp.path());
    }

    #[test]
    fn find_git_root_returns_none_when_no_git() {
        // Create a temp directory tree that has no .git entry anywhere within it.
        // We verify this by creating the tree under /tmp directly so it cannot
        // accidentally be inside the test runner's own git repository.
        let tmp = tempfile::TempDir::new_in("/tmp").expect("tempdir");
        let nested = tmp.path().join("a").join("b");
        std::fs::create_dir_all(&nested).unwrap();
        // /tmp itself does not contain .git, so walking up from our nested dir
        // should exhaust the filesystem without finding one.
        let result = find_git_root(&nested);
        assert!(result.is_none(), "expected None but got {result:?}");
    }

    #[test]
    fn find_git_root_finds_shallow_stop() {
        // Ensure a nested .git doesn't shadow the outer root when there is no inner one.
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join(".git")).unwrap();
        let sub = tmp.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        let result = find_git_root(&sub).unwrap();
        assert_eq!(result, tmp.path());
    }

    // -----------------------------------------------------------------------
    // resolve_workspace_path tests
    // -----------------------------------------------------------------------

    #[test]
    fn resolve_workspace_path_blocks_traversal() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let workspace = tmp.path().join("project");
        std::fs::create_dir_all(&workspace).unwrap();
        let err = resolve_workspace_path(&workspace, "../secret", false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("escapes workspace root"), "error was: {err}");
    }

    #[test]
    fn resolve_workspace_path_allows_missing_nested_subpath() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let workspace = tmp.path().join("project");
        std::fs::create_dir_all(&workspace).unwrap();
        let resolved = resolve_workspace_path(&workspace, "new/dir/file.txt", false).unwrap();
        assert_eq!(
            resolved.canonicalize().unwrap_or(resolved),
            workspace
                .join("new/dir/file.txt")
                .canonicalize()
                .unwrap_or(workspace.join("new/dir/file.txt"))
        );
    }

    #[test]
    fn validate_patch_paths_blocks_traversal() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let workspace = tmp.path().join("project");
        std::fs::create_dir_all(&workspace).unwrap();
        let patch = "diff --git a/../secret b/../secret\n--- a/../secret\n+++ b/../secret\n";
        let err = validate_patch_paths(patch, &workspace)
            .unwrap_err()
            .to_string();
        assert!(err.contains("escapes workspace root"), "error was: {err}");
    }

    #[test]
    fn validate_patch_paths_accepts_project_paths() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let workspace = tmp.path().join("project");
        std::fs::create_dir_all(&workspace).unwrap();
        let patch = "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n";
        let paths = validate_patch_paths(patch, &workspace).unwrap();
        assert_eq!(paths, vec!["src/lib.rs".to_string()]);
    }

    #[test]
    fn write_tools_are_only_available_for_write_profiles() {
        for tool in ["apply_patch", "mv", "rm"] {
            assert!(tool_allowed(tool, AgentProfile::Build, None, false));
            assert!(tool_allowed(tool, AgentProfile::Scout, None, false));
            assert!(!tool_allowed(tool, AgentProfile::Review, None, false));
            assert!(!tool_allowed(tool, AgentProfile::Explore, None, false));
            assert!(!tool_allowed(tool, AgentProfile::Research, None, false));
        }
    }

    #[test]
    fn resolve_workspace_path_allows_subpath() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let file = tmp.path().join("hello.txt");
        std::fs::write(&file, "hi").unwrap();
        let resolved = resolve_workspace_path(tmp.path(), "hello.txt", true).unwrap();
        assert_eq!(resolved, file.canonicalize().unwrap());
    }

    #[test]
    fn discover_skills_reads_agent_skill_metadata() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let skill_dir = tmp.path().join(".claude/skills/rust-web");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: rust-web\ndescription: Build Rust web handlers safely\n---\n# Rust Web\nFull body\n",
        )
        .unwrap();

        let skills = discover_skills(tmp.path()).unwrap();

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].provider, "claude");
        assert_eq!(skills[0].name, "rust-web");
        assert_eq!(skills[0].description, "Build Rust web handlers safely");
        assert_eq!(skills[0].relative_path, ".claude/skills/rust-web/SKILL.md");
    }

    #[test]
    fn skill_search_matches_open_code_and_copilot_locations() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let opencode_skill = tmp.path().join(".opencode/skill/research/SKILL.md");
        std::fs::create_dir_all(opencode_skill.parent().unwrap()).unwrap();
        std::fs::write(
            &opencode_skill,
            "---\nname: targeted-research\ndescription: Find authoritative sources\n---\nbody",
        )
        .unwrap();
        let copilot_prompt = tmp.path().join(".github/prompts/review.prompt.md");
        std::fs::create_dir_all(copilot_prompt.parent().unwrap()).unwrap();
        std::fs::write(
            &copilot_prompt,
            "---\ndescription: Review changed code\n---\n# Review Prompt\nbody",
        )
        .unwrap();

        let result = run_skill_search("research", 10, tmp.path()).unwrap();

        assert!(result.contains("targeted-research"), "{result}");
        assert!(result.contains("provider: opencode"), "{result}");
        assert!(!result.contains("Review changed code"), "{result}");
    }

    #[test]
    fn skill_tool_loads_selected_skill_body_and_resource_hint() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let skill_dir = tmp.path().join(".codex/skills/frontend");
        std::fs::create_dir_all(skill_dir.join("references")).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: frontend\ndescription: Build frontend features\n---\n# Frontend\nUse framework conventions.\n",
        )
        .unwrap();
        std::fs::write(skill_dir.join("references/patterns.md"), "patterns").unwrap();

        let result = run_skill_tool("codex/frontend", tmp.path()).unwrap();

        assert!(result.contains("Skill: frontend"), "{result}");
        assert!(result.contains("Use framework conventions."), "{result}");
        assert!(
            result.contains(".codex/skills/frontend/references/patterns.md"),
            "{result}"
        );
    }
}
