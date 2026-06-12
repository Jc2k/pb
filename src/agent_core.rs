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
use serde::Deserialize;
use serde_json::{Value, json};
use similar::TextDiff;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::num::NonZeroU32;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use crate::container;
use crate::environment::{EnvironmentBackend, EnvironmentConfig};
use crate::events::AgentEvent;

const LLAMA_BATCH_SIZE: usize = 512;
const MIN_GENERATION_CONTEXT_TOKENS: usize = 1;
const MAX_SEARCH_RESULTS: usize = 200;
const MAX_GLOB_RESULTS: usize = 200;
const MAX_WEB_SEARCH_RESULTS: usize = 8;
const MAX_WEB_RESPONSE_BYTES: usize = 512 * 1024;
const MAX_WEB_RESULT_CHARS: usize = 20_000;
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

impl AgentProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Scout => "scout",
            Self::Review => "review",
            Self::Explore => "explore",
            Self::Plan => "plan",
            Self::Ask => "ask",
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
            other => bail!(
                "unknown agent profile '{other}'; expected one of: build, scout, review, explore, plan, ask"
            ),
        }
    }

    fn instructions(self) -> &'static str {
        match self {
            Self::Build => {
                "Profile: build. Implement the requested change with minimal safe edits. Use todo(action=list) or todo(action=next) to inspect shared task memory, todo(action=complete,...) when a task is finished, and todo(action=add,...) when implementation reveals follow-up work. Use explore sub-agents to gather context before invasive work and review sub-agents to check your result before finalizing when useful. You may edit files and commit logical changes."
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
                "Profile: plan. Produce an actionable implementation plan from the available context and use todo(action=add,...) to create concrete build tasks for each actionable step. Do not edit files or create commits. Keep the plan concise and call out assumptions or risks."
            }
            Self::Ask => {
                "Profile: ask. Answer the focused question using repository context and, when necessary, public web research. Do not edit files or create commits. Return a direct answer with supporting evidence."
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
    pub sub_agent_depth: usize,
    pub top_k: i32,
    pub seed: u32,
    /// Optional environment config; when `None`, loaded from `.pb/environment.toml` at runtime.
    pub environment: Option<EnvironmentConfig>,
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

pub fn run_agent<S: EventSink>(
    args: AgentRequest,
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

    let (branch, is_continuation) = if let Some(b) = &args.branch {
        git_checkout_branch(b, &workspace_root)
            .with_context(|| format!("failed to checkout branch '{b}'"))?;
        (b.clone(), true)
    } else {
        let b = branch_name_from_task(&args.task);
        git_create_branch(&b, &workspace_root)
            .with_context(|| format!("failed to create branch '{b}'"))?;
        (b, false)
    };

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

    sink.emit(AgentEvent::Started {
        task: args.task.clone(),
        model: model_path.display().to_string(),
        workspace: workspace_root.display().to_string(),
        branch: branch.clone(),
    });

    suppress_llama_logs();
    let mut backend = LlamaBackend::init().context("failed to initialize llama backend")?;
    backend.void_logs();
    let model_params = LlamaModelParams::default().with_n_gpu_layers(args.gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| format!("failed to load model {}", model_path.display()))?;

    let instructions = build_agent_instructions(
        &workspace_root,
        &branch,
        is_continuation,
        command_backend.as_ref().map(CommandBackend::kind),
        env_config.as_ref(),
        args.profile,
        args.sub_agent_depth < MAX_SUB_AGENT_DEPTH,
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
) -> Result<String> {
    let mut instructions = String::from(
        "You are pb, a local coding agent. Always respond with one JSON object and nothing else.\n",
    );
    instructions.push_str(
        "Use {\"type\":\"tool_call\",\"tool\":\"...\",\"arguments\":{...},\"thinking\":\"...\"} for actions, or {\"type\":\"final\",\"content\":\"...\",\"thinking\":\"...\"} when done.\n",
    );
    instructions.push_str(profile.instructions());
    instructions.push('\n');
    let available_tools = available_tool_specs(profile, command_backend_kind, allow_sub_agents);
    instructions.push_str(&format!(
        "Available tools: {}.\n",
        available_tools.join(", ")
    ));
    if allow_sub_agents {
        instructions.push_str(
            "Use sub_agent(profile,task,max_steps) to delegate bounded work into a fresh context. Supported profiles are explore, review, plan, ask, scout, and build. The sub-agent result is summarized back to you so large investigation transcripts do not bloat your primary context.\n",
        );
    }
    if matches!(profile, AgentProfile::Build | AgentProfile::Scout) {
        instructions.push_str(
            "When editing, keep changes minimal and safe. Use git_commit with a semantic commit message after each logical change.\n",
        );
    } else {
        instructions.push_str(
            "This profile is read-only: do not call edit_file, git_commit, or git_revert.\n",
        );
    }
    instructions.push_str(
        "Use web_search for general internet research and web_fetch for reading a specific URL. Only use public http/https URLs.\n",
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

fn available_tool_specs(
    profile: AgentProfile,
    command_backend_kind: Option<CommandBackendKind>,
    allow_sub_agents: bool,
) -> Vec<&'static str> {
    let mut tools = vec![
        "read_file(path,start,end)",
        "glob(pattern,path,max_results)",
        "ripgrep(pattern,path,max_results)",
        "search(pattern,path)",
        "web_search(query)",
        "web_fetch(url)",
        "git_log()",
        "todo(action,id,title,description,status,parent_id,note)",
        "skill(name)",
    ];
    if command_backend_kind.is_some() {
        tools.push("run_command(cmd)");
    }
    if matches!(profile, AgentProfile::Build | AgentProfile::Scout) {
        tools.push("edit_file(path,old_text,new_text)");
        tools.push("git_commit(message)");
        tools.push("git_revert(commit)");
    }
    if allow_sub_agents {
        tools.push("sub_agent(profile,task,max_steps)");
    }
    tools
}

fn tool_allowed(
    tool: &str,
    profile: AgentProfile,
    command_backend_kind: Option<CommandBackendKind>,
    allow_sub_agents: bool,
) -> bool {
    match tool {
        "read_file" | "glob" | "ripgrep" | "search" | "web_search" | "web_fetch" | "git_log"
        | "todo" | "skill" => true,
        "run_command" => command_backend_kind.is_some(),
        "edit_file" | "git_commit" | "git_revert" => {
            matches!(profile, AgentProfile::Build | AgentProfile::Scout)
        }
        "sub_agent" => allow_sub_agents,
        _ => false,
    }
}

#[derive(Debug, Clone)]
struct StepRunOutcome {
    reached_final: bool,
    final_content: Option<String>,
}

fn run_agent_steps<S: EventSink>(
    backend: &LlamaBackend,
    model: &LlamaModel,
    args: &AgentRequest,
    messages: &mut Vec<ChatMessage>,
    workspace_root: &Path,
    command_backend: Option<&CommandBackend>,
    env_config: Option<&EnvironmentConfig>,
    todo_memory: &RefCell<TodoMemory>,
    sink: &mut S,
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
                    sink.emit(AgentEvent::Reasoning { content: reasoning });
                }
                sink.emit(AgentEvent::Final {
                    content: content.clone(),
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
                    sink.emit(AgentEvent::Reasoning { content: reasoning });
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

fn run_tool<S: EventSink>(
    tool: &str,
    arguments: &Value,
    context: &ToolContext<'_>,
    sink: &mut S,
) -> Result<String> {
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
            let text = std::fs::read_to_string(&resolved)
                .with_context(|| format!("failed to read {}", resolved.display()))?;

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
        "skill" => {
            let name = arguments
                .get("name")
                .and_then(Value::as_str)
                .context("skill requires string argument: name")?;
            Ok(skill_text(name))
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

impl EventSink for SubAgentEventCollector {
    fn emit(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Final { content } => self.final_content = Some(content),
            AgentEvent::Error { message } => self.errors.push(message),
            AgentEvent::Diff { .. } => self.diffs += 1,
            _ => {}
        }
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

fn run_sub_agent<S: EventSink>(
    arguments: &Value,
    context: &ToolContext<'_>,
    sink: &mut S,
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
    });

    let instructions = build_agent_instructions(
        context.workspace_root,
        context.request.branch.as_deref().unwrap_or("sub-agent"),
        true,
        context.command_backend.map(CommandBackend::kind),
        context.env_config,
        profile,
        false,
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

    let mut collector = SubAgentEventCollector::default();
    let outcome = run_agent_steps(
        context.backend,
        context.model,
        &sub_request,
        &mut messages,
        context.workspace_root,
        context.command_backend,
        context.env_config,
        context.todo_memory,
        &mut collector,
    )?;

    let mut result = String::new();
    if outcome.reached_final {
        result.push_str(
            collector
                .final_content
                .as_deref()
                .or(outcome.final_content.as_deref())
                .unwrap_or("sub-agent finished without a final message"),
        );
    } else {
        result.push_str("sub-agent reached its step limit before finalizing");
    }
    if collector.diffs > 0 {
        result.push_str(&format!(
            "

Workspace edits emitted: {} diff(s).",
            collector.diffs
        ));
    }
    if !collector.errors.is_empty() {
        result.push_str(
            "

Errors:
",
        );
        result.push_str(&collector.errors.join(
            "
",
        ));
    }

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

fn resolve_workspace_path(workspace_root: &Path, input: &str, must_exist: bool) -> Result<PathBuf> {
    let candidate = if Path::new(input).is_absolute() {
        PathBuf::from(input)
    } else {
        workspace_root.join(input)
    };

    let normalized = if must_exist {
        candidate
            .canonicalize()
            .with_context(|| format!("failed to resolve path {}", candidate.display()))?
    } else if let Some(parent) = candidate.parent() {
        let parent = parent
            .canonicalize()
            .with_context(|| format!("failed to resolve parent {}", parent.display()))?;
        parent.join(candidate.file_name().unwrap_or_default())
    } else {
        candidate
    };

    if !normalized.starts_with(workspace_root) {
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

fn skill_text(name: &str) -> String {
    match name {
        "copilot" => {
            "Use repository instructions first; keep edits minimal; run tests before finalizing."
                .to_string()
        }
        "codex" => {
            "Prefer structured tool calls, verify edits with diffs, and keep responses concise."
                .to_string()
        }
        "claude-code" => {
            "Think in small steps, use safe file boundaries, and report reasoning clearly."
                .to_string()
        }
        "list" => "Available skills: copilot, codex, claude-code".to_string(),
        _ => format!("unknown skill '{name}'. Try: copilot, codex, claude-code, list"),
    }
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
        )
        .unwrap();
        assert!(instructions.contains("Profile: build"));
        assert!(instructions.contains("sub_agent(profile,task,max_steps)"));
        assert!(instructions.contains("edit_file(path,old_text,new_text)"));
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
        )
        .unwrap();
        assert!(instructions.contains("Profile: review"));
        assert!(instructions.contains("This profile is read-only"));
        assert!(!instructions.contains("edit_file(path,old_text,new_text)"));
        assert!(!instructions.contains("sub_agent(profile,task,max_steps)"));
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
    fn resolve_workspace_path_allows_subpath() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let file = tmp.path().join("hello.txt");
        std::fs::write(&file, "hi").unwrap();
        let resolved = resolve_workspace_path(tmp.path(), "hello.txt", true).unwrap();
        assert_eq!(resolved, file.canonicalize().unwrap());
    }
}
