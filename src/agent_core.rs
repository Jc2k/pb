use anyhow::{Context, Result, bail};
use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use serde::Deserialize;
use serde_json::Value;
use similar::TextDiff;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::events::AgentEvent;

const LLAMA_BATCH_SIZE: usize = 512;
const MAX_SEARCH_RESULTS: usize = 200;
const SEARCH_EXCLUDED_DIRS: &[&str] = &[".git", "target"];

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

#[derive(Debug, Clone)]
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
    pub top_k: i32,
    pub seed: u32,
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
    let workspace_root = workdir
        .canonicalize()
        .with_context(|| format!("failed to resolve workdir {}", workdir.display()))?;

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

    sink.emit(AgentEvent::Started {
        task: args.task.clone(),
        model: model_path.display().to_string(),
        workspace: workspace_root.display().to_string(),
        branch: branch.clone(),
    });

    let backend = LlamaBackend::init().context("failed to initialize llama backend")?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(args.gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| format!("failed to load model {}", model_path.display()))?;

    let instructions = build_agent_instructions(&workspace_root, &branch, is_continuation)?;

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

    let reached_final =
        run_agent_steps(&backend, &model, &args, &mut messages, &workspace_root, &mut sink)?;

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

    Ok(AgentRunResult {
        branch,
        workspace_root,
        reached_final,
    })
}

fn build_agent_instructions(workspace_root: &Path, branch: &str, continuing: bool) -> Result<String> {
    let mut instructions = String::from(
        "You are pb, a local coding agent. Always respond with one JSON object and nothing else.\n",
    );
    instructions.push_str(
        "Use {\"type\":\"tool_call\",\"tool\":\"...\",\"arguments\":{...},\"thinking\":\"...\"} for actions, or {\"type\":\"final\",\"content\":\"...\",\"thinking\":\"...\"} when done.\n",
    );
    instructions.push_str(
        "Available tools: read_file(path,start,end), search(pattern,path), edit_file(path,old_text,new_text), git_commit(message), git_log(), skill(name).\n",
    );
    instructions.push_str(
        "When editing, keep changes minimal and safe. Use git_commit with a semantic commit message after each logical change.\n",
    );

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

fn run_agent_steps<S: EventSink>(
    backend: &LlamaBackend,
    model: &LlamaModel,
    args: &AgentRequest,
    messages: &mut Vec<ChatMessage>,
    workspace_root: &Path,
    sink: &mut S,
) -> Result<bool> {
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
                sink.emit(AgentEvent::Final { content });
                return Ok(true);
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
                let tool_result = run_tool(&tool, &arguments, workspace_root, sink)?;
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

    Ok(false)
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

    let mut decoder = UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = batch.n_tokens();

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

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .context("failed to queue generated token")?;
        ctx.decode(&mut batch)
            .context("failed to decode generated token")?;
        n_cur += 1;
    }

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

fn run_tool<S: EventSink>(
    tool: &str,
    arguments: &Value,
    workspace_root: &Path,
    sink: &mut S,
) -> Result<String> {
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
            if let Some(end) = end {
                if (end as usize) < start {
                    return Ok("(no content in requested range)".to_string());
                }
            }
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
        "git_commit" => {
            let message = arguments
                .get("message")
                .and_then(Value::as_str)
                .context("git_commit requires string argument: message")?;
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
        "copilot" => {
            "Use repository instructions first; keep edits minimal; run tests before finalizing."
                .to_string()
        }
        "codex" => {
            "Prefer structured tool calls, verify edits with diffs, and keep responses concise."
                .to_string()
        }
        "claude-code" => {
            "Think in small steps, use safe file boundaries, and report reasoning clearly.".to_string()
        }
        "list" => "Available skills: copilot, codex, claude-code".to_string(),
        _ => format!("unknown skill '{name}'. Try: copilot, codex, claude-code, list"),
    }
}

pub fn branch_name_from_task(task: &str) -> String {
    let slug: String = task
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '-' })
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

pub fn find_model_in_cache_in(pull_root: &Path, model: &str) -> Result<PathBuf> {
    let model_dir = pull_root.join(model);

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
    gguf_files.sort_by_key(|p| std::cmp::Reverse(std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)));

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
        assert_eq!(branch_name_from_task("Fix the login bug"), "pb/fix-the-login-bug");
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
}
