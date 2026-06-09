use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use futures::{StreamExt, stream};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use reqwest::header::ACCEPT;
use serde::Deserialize;
use serde_json::Value;
use similar::TextDiff;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tokio::time::{Duration, sleep};
use walkdir::WalkDir;

const DEFAULT_MODEL: &str = "qwen3-coder-next";
const OLLAMA_REGISTRY: &str = "https://registry.ollama.ai";
const DEFAULT_AGENT_MAX_STEPS: usize = 12;
const DEFAULT_AGENT_MAX_TOKENS: i32 = 384;
const LLAMA_BATCH_SIZE: usize = 512;
const MAX_SEARCH_RESULTS: usize = 200;
const SEARCH_EXCLUDED_DIRS: &[&str] = &[".git", "target"];

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
}

#[derive(Subcommand, Debug)]
pub enum SelfCommand {
    /// Update pb from the latest GitHub release
    Update,
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

    /// Path to a local GGUF model file
    #[arg(long)]
    pub model_path: Option<PathBuf>,

    /// Working directory where tools can read/search/edit
    #[arg(long)]
    pub workdir: Option<PathBuf>,

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

#[derive(Debug, Clone)]
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

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::SelfCmd { command } => match command {
            SelfCommand::Update => run_self_update(),
        },
        Commands::Pull(args) => pull_model(&args).await,
        Commands::Agent(args) => run_agent(args).await,
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

pub async fn pull_model(args: &PullArgs) -> Result<()> {
    if args.batch_size == 0 {
        bail!("batch-size must be greater than 0");
    }
    if args.parallel == 0 {
        bail!("parallel must be greater than 0");
    }

    let output_root = args.out_dir.clone().unwrap_or_else(default_pull_dir);
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

    client
        .get(url)
        .header(
            ACCEPT,
            "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json",
        )
        .send()
        .await
        .context("failed to request model manifest")?
        .error_for_status()
        .context("failed to fetch model manifest")?
        .json::<Manifest>()
        .await
        .context("failed to decode model manifest")
}

fn descriptors_from_manifest(manifest: Manifest) -> Vec<ManifestDescriptor> {
    let mut descriptors = Vec::with_capacity(manifest.layers.len() + 1);
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
    let blob_path = blob_path(output_root, model, &descriptor.digest);

    if tokio::fs::try_exists(&blob_path).await.unwrap_or(false) {
        println!("Skipping existing blob {}", descriptor.digest);
        return Ok(());
    }

    if let Some(parent) = blob_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    let mut attempt = 0u32;
    loop {
        attempt += 1;

        let result = download_blob(
            client,
            model,
            &descriptor.digest,
            descriptor.size,
            &blob_path,
        )
        .await;
        match result {
            Ok(_) => return Ok(()),
            Err(err) if attempt <= retries => {
                let backoff = Duration::from_millis(250 * u64::from(attempt));
                eprintln!(
                    "Blob {} failed (attempt {attempt}/{retries}), retrying in {:?}: {err}",
                    descriptor.digest, backoff
                );
                sleep(backoff).await;
            }
            Err(err) => return Err(err),
        }
    }
}

async fn download_blob(
    client: &reqwest::Client,
    model: &str,
    digest: &str,
    expected_size: u64,
    destination: &Path,
) -> Result<()> {
    let url = format!("{OLLAMA_REGISTRY}/v2/library/{model}/blobs/{digest}");
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("failed to request blob {digest}"))?
        .error_for_status()
        .with_context(|| format!("failed to download blob {digest}"))?;

    let mut file = tokio::fs::File::create(destination)
        .await
        .with_context(|| format!("failed to create {}", destination.display()))?;

    let mut stream = response.bytes_stream();
    let mut bytes_written: u64 = 0;
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

pub async fn run_agent(args: AgentArgs) -> Result<()> {
    let model_path = resolve_model_path(args.model_path.as_deref())?;
    let workdir = args
        .workdir
        .clone()
        .unwrap_or(std::env::current_dir().context("failed to get current working directory")?);
    let workspace_root = workdir
        .canonicalize()
        .with_context(|| format!("failed to resolve workdir {}", workdir.display()))?;

    print_header("local agent", &format!("task: {}", args.task));
    print_header("model", &model_path.display().to_string());
    print_header("workspace", &workspace_root.display().to_string());

    let mut instructions = String::from(
        "You are pb, a local coding agent. Always respond with one JSON object and nothing else.\n",
    );
    instructions.push_str(
        "Use {\"type\":\"tool_call\",\"tool\":\"...\",\"arguments\":{...},\"thinking\":\"...\"} \
for actions, or {\"type\":\"final\",\"content\":\"...\",\"thinking\":\"...\"} when done.\n",
    );
    instructions.push_str("Available tools: read_file(path,start,end), search(pattern,path), edit_file(path,old_text,new_text), skill(name).\n");
    instructions.push_str("When editing, keep changes minimal and safe.\n");
    if let Ok(copilot_instructions) = std::fs::read_to_string(".github/copilot-instructions.md") {
        instructions.push_str("Repository instructions:\n");
        instructions.push_str(&copilot_instructions);
        instructions.push('\n');
    }

    print_header("started", &args.task);

    let backend = LlamaBackend::init().context("failed to initialize llama backend")?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(args.gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| format!("failed to load model {}", model_path.display()))?;

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

    for step in 1..=args.max_steps {
        print_header("step", &format!("{step}/{}", args.max_steps));
        let prompt = render_prompt(&messages);
        let output = generate_completion(&backend, &model, &args, &prompt)?;
        let action = parse_action(&output)?;

        match action {
            AgentAction::Final { content, thinking } => {
                if let Some(reasoning) = thinking {
                    print_header("reasoning", &reasoning);
                }
                print_header("final", &content);
                break;
            }
            AgentAction::ToolCall {
                tool,
                arguments,
                thinking,
            } => {
                if let Some(reasoning) = thinking {
                    print_header("reasoning", &reasoning);
                }
                print_header("tool", &tool);
                let tool_result = run_tool(&tool, &arguments, &workspace_root)?;
                print_block("tool result", &tool_result);

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

    Ok(())
}

fn resolve_model_path(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }

    if let Ok(from_env) = std::env::var("PB_MODEL_PATH") {
        return Ok(PathBuf::from(from_env));
    }

    bail!(
        "model path is required: pass --model-path <file.gguf> or set PB_MODEL_PATH"
    )
}

fn render_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    prompt.push_str("<conversation>\n");
    for message in messages {
        prompt.push_str("[");
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
    args: &AgentArgs,
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

    let mut batch = LlamaBatch::new(LLAMA_BATCH_SIZE, 1);
    let last_index = (tokens.len().saturating_sub(1)) as i32;
    for (i, token) in (0_i32..).zip(tokens.into_iter()) {
        let is_last = i == last_index;
        batch
            .add(token, i, &[0], is_last)
            .context("failed to add prompt token to batch")?;
    }

    ctx.decode(&mut batch)
        .context("failed to decode prompt batch")?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(args.seed),
        LlamaSampler::top_k(args.top_k),
        LlamaSampler::temp(args.temperature),
    ]);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = batch.n_tokens();

    print!("\x1b[1;33massistant > \x1b[0m");
    std::io::stdout().flush().ok();

    while n_cur <= args.max_tokens {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .context("failed to decode output token")?;
        output.push_str(&piece);
        print!("{piece}");
        std::io::stdout().flush().ok();

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .context("failed to queue generated token")?;
        ctx.decode(&mut batch)
            .context("failed to decode generated token")?;
        n_cur += 1;
    }
    println!();

    Ok(output)
}

fn parse_action(output: &str) -> Result<AgentAction> {
    if let Ok(action) = serde_json::from_str::<AgentAction>(output.trim()) {
        return Ok(action);
    }

    let json_candidate = extract_json_object(output)
        .with_context(|| "model output did not contain a valid JSON action")?;
    serde_json::from_str::<AgentAction>(&json_candidate)
        .with_context(|| "failed to parse agent JSON action")
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

    None
}

fn run_tool(tool: &str, arguments: &Value, workspace_root: &Path) -> Result<String> {
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
            // Keep end_line >= start so reversed ranges safely produce no output.
            let end_line = end.map_or(lines.len(), |v| v as usize).max(start);
            let mut out = String::new();
            for idx in (start.saturating_sub(1))..lines.len().min(end_line) {
                out.push_str(&format!("{}: {}\n", idx + 1, lines[idx]));
            }
            if out.is_empty() {
                out.push_str("(no content in requested range)");
            }
            Ok(out)
        }
        "search" => {
            let pattern = arguments
                .get("pattern")
                .and_then(Value::as_str)
                .context("search requires string argument: pattern")?;
            let relative_path = arguments.get("path").and_then(Value::as_str);
            let search_root = if let Some(path) = relative_path {
                resolve_workspace_path(workspace_root, path, true)?
            } else {
                workspace_root.to_path_buf()
            };

            let regex = regex::Regex::new(pattern)
                .with_context(|| format!("invalid regex pattern: {pattern}"))?;

            let mut hits = Vec::new();
            for entry in WalkDir::new(&search_root)
                .into_iter()
                .filter_entry(|e| {
                    !SEARCH_EXCLUDED_DIRS
                        .iter()
                        .any(|excluded| e.file_name() == std::ffi::OsStr::new(excluded))
                })
                .filter_map(Result::ok)
                .filter(|e| e.file_type().is_file())
            {
                let path = entry.path();
                let Ok(content) = std::fs::read_to_string(path) else {
                    continue;
                };
                for (line_idx, line) in content.lines().enumerate() {
                    if regex.is_match(line) {
                        let rel = path.strip_prefix(workspace_root).unwrap_or(path);
                        hits.push(format!("{}:{}:{}", rel.display(), line_idx + 1, line.trim()));
                        if hits.len() >= MAX_SEARCH_RESULTS {
                            break;
                        }
                    }
                }
                if hits.len() >= MAX_SEARCH_RESULTS {
                    break;
                }
            }

            if hits.is_empty() {
                Ok("no matches".to_string())
            } else {
                Ok(hits.join("\n"))
            }
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

            // Replace only the first match to keep edits targeted and predictable.
            let updated = existing.replacen(old_text, new_text, 1);
            std::fs::write(&resolved, &updated)
                .with_context(|| format!("failed to write {}", resolved.display()))?;

            let diff = unified_diff(&existing, &updated, path);
            print_block("diff", &diff);
            Ok(format!("updated {}", resolved.display()))
        }
        "skill" => {
            let name = arguments
                .get("name")
                .and_then(Value::as_str)
                .context("skill requires string argument: name")?;
            Ok(skill_text(name))
        }
        _ => bail!("unknown tool: {tool}"),
    }
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
        "copilot" => "Use repository instructions first; keep edits minimal; run tests before finalizing.".to_string(),
        "codex" => "Prefer structured tool calls, verify edits with diffs, and keep responses concise.".to_string(),
        "claude-code" => "Think in small steps, use safe file boundaries, and report reasoning clearly.".to_string(),
        "list" => "Available skills: copilot, codex, claude-code".to_string(),
        _ => format!("unknown skill '{name}'. Try: copilot, codex, claude-code, list"),
    }
}

fn print_header(label: &str, value: &str) {
    println!("\x1b[1;36m[{label}]\x1b[0m {value}");
}

fn print_block(label: &str, content: &str) {
    println!("\x1b[1;35m[{label}]\x1b[0m\n{content}");
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
        // Large value requests full offload; llama.cpp clamps to model layer count.
        999
    } else {
        0
    }
}

fn default_pull_dir() -> PathBuf {
    let mut root = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    root.push(".pb");
    root.push("models");
    root
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

    #[test]
    fn extract_json_object_handles_noise() {
        let output = "hello {\"type\":\"final\",\"content\":\"ok\"} trailing";
        let extracted = extract_json_object(output).expect("json should be extracted");
        assert_eq!(extracted, "{\"type\":\"final\",\"content\":\"ok\"}");
    }
}
