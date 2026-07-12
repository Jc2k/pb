//! Direct, daemon-free harnesses for exercising pb internals.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};

use crate::HarnessAgentArgs;
use crate::agent_core::{AgentRequest, AgentRunResult, EventSink, SessionAttachment, run_agent};
use crate::cli_ui::render_event;
use crate::config::UserConfig;
use crate::environment::{EnvironmentBackend, EnvironmentConfig, EnvironmentMode};
use crate::events::{AgentEvent, EventEnvelope};
use crate::session_store::now_millis;

const HARNESS_GIT_NAME: &str = "pb harness";
const HARNESS_GIT_EMAIL: &str = "harness@pb.local";
const HARNESS_AGENT_TOOLS: &[&str] = &[
    "session_title",
    "todo",
    "read_file",
    "glob",
    "ripgrep",
    "search",
    "git_log",
    "session_changes",
    "run_command",
    "edit_file",
    "apply_patch",
    "mv",
    "rm",
    "git_commit",
    "git_revert",
    "sub_agent",
];

#[derive(Debug)]
struct ScratchLayout {
    root: PathBuf,
    workspace: PathBuf,
    events: PathBuf,
    journal: PathBuf,
}

#[derive(Debug, Clone)]
struct Observation {
    rank: u8,
    title: String,
    detail: String,
}

#[derive(Debug, Clone, Default)]
struct CapturedSummary {
    branch: String,
    commits: String,
    summary: String,
    diff_stat: String,
}

struct JournalState {
    writer: BufWriter<File>,
    observations: Vec<Observation>,
    summary: CapturedSummary,
    write_error: Option<String>,
}

#[derive(Clone)]
struct HarnessEventSink {
    state: Arc<Mutex<JournalState>>,
}

impl HarnessEventSink {
    fn new(path: &Path) -> Result<Self> {
        let file = File::create(path).with_context(|| {
            format!("failed to create harness event journal {}", path.display())
        })?;
        Ok(Self {
            state: Arc::new(Mutex::new(JournalState {
                writer: BufWriter::new(file),
                observations: Vec::new(),
                summary: CapturedSummary::default(),
                write_error: None,
            })),
        })
    }

    fn snapshot(&self) -> Result<(Vec<Observation>, CapturedSummary)> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("harness event journal lock was poisoned"))?;
        state
            .writer
            .flush()
            .context("failed to flush harness event journal")?;
        if let Some(error) = state.write_error.as_deref() {
            bail!("failed to write harness event journal: {error}");
        }
        Ok((state.observations.clone(), state.summary.clone()))
    }
}

impl EventSink for HarnessEventSink {
    fn emit(&mut self, event: AgentEvent) {
        render_event(&event);
        let Ok(mut state) = self.state.lock() else {
            eprintln!("pb harness: event journal lock was poisoned");
            return;
        };

        match &event {
            AgentEvent::Started { branch, .. } => {
                state.summary.branch = branch.clone();
            }
            AgentEvent::Error {
                message, summary, ..
            } => state.observations.push(Observation {
                rank: 0,
                title: nonempty_or(summary, "agent error"),
                detail: compact_detail(message),
            }),
            AgentEvent::Correction {
                message, summary, ..
            } => state.observations.push(Observation {
                rank: 1,
                title: nonempty_or(summary, "agent correction"),
                detail: compact_detail(message),
            }),
            AgentEvent::SessionSummary {
                branch,
                commits,
                summary,
                diff_stat,
                ..
            } => {
                state.summary = CapturedSummary {
                    branch: branch.clone(),
                    commits: commits.clone(),
                    summary: summary.clone(),
                    diff_stat: diff_stat.clone(),
                };
            }
            _ => {}
        }

        if state.write_error.is_some() {
            return;
        }
        let envelope = EventEnvelope::new(event);
        let write_result = serde_json::to_writer(&mut state.writer, &envelope)
            .map_err(|error| error.to_string())
            .and_then(|_| {
                state
                    .writer
                    .write_all(b"\n")
                    .map_err(|error| error.to_string())
            })
            .and_then(|_| state.writer.flush().map_err(|error| error.to_string()));
        if let Err(error) = write_result {
            state.write_error = Some(error);
        }
    }
}

pub fn run_agent_task(args: HarnessAgentArgs) -> Result<()> {
    if args.task.trim().is_empty() {
        bail!("harness agent task must not be empty");
    }

    let layout = prepare_scratch(args.scratch_dir.as_deref())?;
    println!("pb harness: scratch={}", layout.root.display());
    println!("pb harness: workspace={}", layout.workspace.display());
    println!("pb harness: events={}", layout.events.display());
    println!("pb harness: journal={}", layout.journal.display());
    write_running_journal(&layout, &args.task)?;

    let user_config = UserConfig::load()?;
    let model_dir = args
        .model_dir
        .clone()
        .or_else(|| user_config.effective_model_dir());
    let models_root = model_dir.clone().unwrap_or_else(crate::default_models_dir);
    let turn_max_tokens_cap = args.max_tokens;
    let request = AgentRequest {
        task: args.task.clone(),
        model: args
            .model
            .clone()
            .unwrap_or_else(|| user_config.effective_model()),
        model_dir,
        workdir: Some(layout.workspace.clone()),
        branch: None,
        max_steps: args
            .max_steps
            .unwrap_or_else(|| user_config.effective_max_steps()),
        max_tokens: args
            .max_tokens
            .unwrap_or_else(|| user_config.effective_max_tokens()),
        turn_max_tokens_cap,
        tool_allowlist: Some(
            HARNESS_AGENT_TOOLS
                .iter()
                .map(|tool| (*tool).to_string())
                .collect(),
        ),
        ctx_size: args
            .ctx_size
            .unwrap_or_else(|| user_config.effective_ctx_size()),
        threads: args.threads.or_else(|| user_config.effective_threads()),
        threads_batch: args
            .threads_batch
            .or_else(|| user_config.effective_threads_batch()),
        gpu_layers: args
            .gpu_layers
            .unwrap_or_else(|| user_config.effective_gpu_layers()),
        temperature: args
            .temperature
            .unwrap_or_else(|| user_config.effective_temperature()),
        profile: args.profile,
        infer_profile: false,
        sub_agent_depth: 0,
        repository_less: false,
        top_k: args.top_k.unwrap_or_else(|| user_config.effective_top_k()),
        seed: args.seed.unwrap_or_else(|| user_config.effective_seed()),
        environment: Some(harness_environment()),
        session_id: format!("harness-{}", now_millis()),
        attachments: harness_attachments(&args.images)?,
    };

    let sink = HarnessEventSink::new(&layout.events)?;
    let run_result = run_agent(request, &models_root, sink.clone());
    let (mut observations, summary) = sink.snapshot()?;
    add_run_observations(&mut observations, &run_result, &layout.workspace, &summary);
    write_journal(
        &layout,
        &args.task,
        &run_result,
        &summary,
        &mut observations,
    )?;

    match run_result {
        Ok(result) if result.reached_final => {
            println!(
                "pb harness: completed=true reached_final=true branch={} workspace={} journal={}",
                result.branch,
                result.workspace_root.display(),
                layout.journal.display()
            );
            Ok(())
        }
        Ok(result) => bail!(
            "harness agent exhausted its run without a final answer; workspace={} journal={}",
            result.workspace_root.display(),
            layout.journal.display()
        ),
        Err(error) => Err(error).with_context(|| {
            format!(
                "harness agent failed; workspace={} journal={}",
                layout.workspace.display(),
                layout.journal.display()
            )
        }),
    }
}

fn prepare_scratch(requested: Option<&Path>) -> Result<ScratchLayout> {
    let root = match requested {
        Some(path) => {
            if path.exists() {
                bail!("harness scratch path already exists: {}", path.display());
            }
            std::fs::create_dir_all(path)
                .with_context(|| format!("failed to create harness scratch {}", path.display()))?;
            path.to_path_buf()
        }
        None => create_unique_scratch_root()?,
    };
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve harness scratch {}", root.display()))?;
    let workspace = root.join("workspace");
    std::fs::create_dir(&workspace)
        .with_context(|| format!("failed to create harness workspace {}", workspace.display()))?;
    initialize_git_workspace(&workspace)?;
    Ok(ScratchLayout {
        events: root.join("events.jsonl"),
        journal: root.join("journal.md"),
        root,
        workspace,
    })
}

fn create_unique_scratch_root() -> Result<PathBuf> {
    let base = std::env::temp_dir();
    let timestamp = now_millis();
    let pid = std::process::id();
    for suffix in 0..100u8 {
        let path = base.join(format!("pb-harness-{timestamp}-{pid}-{suffix}"));
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create harness scratch {}", path.display())
                });
            }
        }
    }
    bail!("could not allocate a unique harness scratch directory")
}

fn initialize_git_workspace(workspace: &Path) -> Result<()> {
    let initialized = git_command(workspace, &["init", "-b", "main"])?;
    if !initialized.status.success() {
        require_git_success(workspace, &["init"])?;
        require_git_success(workspace, &["branch", "-M", "main"])?;
    }
    require_git_success(workspace, &["config", "user.name", HARNESS_GIT_NAME])?;
    require_git_success(workspace, &["config", "user.email", HARNESS_GIT_EMAIL])?;
    require_git_success(
        workspace,
        &[
            "commit",
            "--allow-empty",
            "-m",
            "chore: initialize harness workspace",
        ],
    )?;
    Ok(())
}

fn git_command(workspace: &Path, args: &[&str]) -> Result<std::process::Output> {
    Command::new("git")
        .args(args)
        .current_dir(workspace)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))
}

fn require_git_success(workspace: &Path, args: &[&str]) -> Result<String> {
    let output = git_command(workspace, args)?;
    if !output.status.success() {
        bail!(
            "git {} failed in {}: {}",
            args.join(" "),
            workspace.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn harness_environment() -> EnvironmentConfig {
    EnvironmentConfig {
        mode: EnvironmentMode::Local,
        backend: EnvironmentBackend::Local,
        image: "local".to_string(),
        init_commands: Vec::new(),
        setup_commands: Vec::new(),
        session_commands: Vec::new(),
        guard_commands: Vec::new(),
        prepared_image: None,
        source: Some("pb harness scratch workspace".to_string()),
        dockerfile: None,
    }
}

fn harness_attachments(paths: &[PathBuf]) -> Result<Vec<SessionAttachment>> {
    paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let path = path
                .canonicalize()
                .with_context(|| format!("failed to resolve harness image {}", path.display()))?;
            let metadata = path
                .metadata()
                .with_context(|| format!("failed to inspect harness image {}", path.display()))?;
            Ok(SessionAttachment {
                id: format!("img{}", index + 1),
                name: path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("image")
                    .to_string(),
                mime: mime_guess::from_path(&path)
                    .first_or_octet_stream()
                    .to_string(),
                path: path.to_string_lossy().into_owned(),
                size: metadata.len(),
            })
        })
        .collect()
}

fn add_run_observations(
    observations: &mut Vec<Observation>,
    result: &Result<AgentRunResult>,
    workspace: &Path,
    summary: &CapturedSummary,
) {
    match result {
        Err(error) => observations.push(Observation {
            rank: 0,
            title: "agent run failed".to_string(),
            detail: compact_detail(&format!("{error:#}")),
        }),
        Ok(result) if !result.reached_final => observations.push(Observation {
            rank: 0,
            title: "agent did not reach a final answer".to_string(),
            detail: "The step budget ended before the full task completion contract was satisfied."
                .to_string(),
        }),
        Ok(_) => {}
    }

    let committed =
        require_git_success(workspace, &["log", "--oneline", "main..HEAD"]).unwrap_or_default();
    if matches!(result, Ok(result) if result.reached_final) && committed.trim().is_empty() {
        observations.push(Observation {
            rank: 1,
            title: "completed run produced no commits".to_string(),
            detail: "Confirm whether the task genuinely required no repository changes."
                .to_string(),
        });
    }
    let status = require_git_success(workspace, &["status", "--short"]).unwrap_or_default();
    if !status.trim().is_empty() {
        observations.push(Observation {
            rank: 1,
            title: "workspace has uncommitted changes".to_string(),
            detail: compact_detail(&status),
        });
    }
    if summary.summary.trim().is_empty() {
        observations.push(Observation {
            rank: 2,
            title: "agent emitted no session summary".to_string(),
            detail: "Review the final event stream to determine the actual outcome.".to_string(),
        });
    }
    if observations.is_empty() {
        observations.push(Observation {
            rank: 3,
            title: "no automatic runtime issues observed".to_string(),
            detail: "Manual review of the committed implementation and tests is still required."
                .to_string(),
        });
    }
}

fn write_journal(
    layout: &ScratchLayout,
    task: &str,
    result: &Result<AgentRunResult>,
    summary: &CapturedSummary,
    observations: &mut Vec<Observation>,
) -> Result<()> {
    observations.sort_by(|left, right| {
        left.rank
            .cmp(&right.rank)
            .then_with(|| left.detail.cmp(&right.detail))
            .then_with(|| left.title.cmp(&right.title))
    });
    observations.dedup_by(|left, right| left.rank == right.rank && left.detail == right.detail);
    let status = match result {
        Ok(result) if result.reached_final => "completed",
        Ok(_) => "incomplete",
        Err(_) => "failed",
    };
    let branch = result
        .as_ref()
        .map(|result| result.branch.as_str())
        .unwrap_or(summary.branch.as_str());
    let committed = require_git_success(&layout.workspace, &["log", "--oneline", "main..HEAD"])
        .unwrap_or_else(|_| summary.commits.clone());

    let mut journal = String::new();
    journal.push_str("# pb harness journal\n\n");
    journal.push_str(&format!("- Status: `{status}`\n"));
    journal.push_str(&format!("- Task: {task}\n"));
    journal.push_str(&format!("- Workspace: `{}`\n", layout.workspace.display()));
    journal.push_str(&format!("- Branch: `{branch}`\n"));
    journal.push_str(&format!("- Raw events: `{}`\n", layout.events.display()));
    journal.push_str("\n## Ranked observations\n\n");
    for observation in observations.iter() {
        journal.push_str(&format!(
            "1. **P{} — {}.** {}\n",
            observation.rank, observation.title, observation.detail
        ));
    }
    journal.push_str("\n## Agent summary\n\n");
    if summary.summary.trim().is_empty() {
        journal.push_str("_No session summary was emitted._\n");
    } else {
        journal.push_str(summary.summary.trim());
        journal.push('\n');
    }
    journal.push_str("\n## Committed fixes\n\n");
    if committed.trim().is_empty() {
        journal.push_str("_No commits beyond the harness baseline._\n");
    } else {
        journal.push_str("```text\n");
        journal.push_str(committed.trim());
        journal.push_str("\n```\n");
    }
    if !summary.diff_stat.trim().is_empty() {
        journal.push_str("\n### Diff stat\n\n```text\n");
        journal.push_str(summary.diff_stat.trim());
        journal.push_str("\n```\n");
    }
    journal.push_str("\n## Follow-up improvement plan\n\n");
    journal.push_str("- [ ] Review the committed implementation and rerun its acceptance checks independently.\n");
    for observation in observations.iter().filter(|item| item.rank <= 1) {
        journal.push_str(&format!(
            "- [ ] Reproduce and address P{}: {}.\n",
            observation.rank, observation.title
        ));
    }
    journal.push_str(
        "- [ ] Convert validated, non-blocking observations into a prioritized improvement plan.\n",
    );
    std::fs::write(&layout.journal, journal).with_context(|| {
        format!(
            "failed to write harness journal {}",
            layout.journal.display()
        )
    })
}

fn write_running_journal(layout: &ScratchLayout, task: &str) -> Result<()> {
    let journal = format!(
        "# pb harness journal\n\n\
         - Status: `running`\n\
         - Task: {task}\n\
         - Workspace: `{workspace}`\n\
         - Raw events: `{events}`\n\n\
         ## Ranked observations\n\n\
         1. **P1 — run has not finalized.** If the harness was interrupted, inspect the raw event stream and workspace before deciding whether to rerun.\n\n\
         ## Follow-up improvement plan\n\n\
         - [ ] Wait for the blocking agent run to finish, or diagnose why it was interrupted.\n\
         - [ ] Review the workspace and raw events before making changes to pb.\n",
        workspace = layout.workspace.display(),
        events = layout.events.display(),
    );
    std::fs::write(&layout.journal, journal).with_context(|| {
        format!(
            "failed to initialize harness journal {}",
            layout.journal.display()
        )
    })
}

fn nonempty_or(value: &str, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        fallback.to_string()
    } else {
        compact_detail(value)
    }
}

fn compact_detail(value: &str) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let shortened = chars.by_ref().take(800).collect::<String>();
    if chars.next().is_some() {
        format!("{shortened}…")
    } else {
        shortened
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_workspace_is_persistent_git_repository() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();

        assert_eq!(layout.root, root.canonicalize().unwrap());
        assert!(layout.workspace.join(".git").is_dir());
        assert_eq!(
            require_git_success(&layout.workspace, &["branch", "--show-current"]).unwrap(),
            "main"
        );
        assert_eq!(
            require_git_success(&layout.workspace, &["log", "-1", "--pretty=%s"]).unwrap(),
            "chore: initialize harness workspace"
        );
    }

    #[test]
    fn compact_detail_is_single_line_and_bounded() {
        let detail = compact_detail(&format!("first\n{}", "x".repeat(900)));
        assert!(!detail.contains('\n'));
        assert!(detail.chars().count() <= 801);
        assert!(detail.ends_with('…'));
    }

    #[test]
    fn event_journal_captures_started_branch_before_failures() {
        let parent = tempfile::tempdir().unwrap();
        let events = parent.path().join("events.jsonl");
        let sink = HarnessEventSink::new(&events).unwrap();
        let mut emitter = sink.clone();
        emitter.emit(AgentEvent::Started {
            task: "task".to_string(),
            model: "model".to_string(),
            workspace: "/tmp/workspace".to_string(),
            branch: "pb/task-harness-1".to_string(),
            attachments: Vec::new(),
            timestamp_ms: None,
        });

        assert_eq!(std::fs::read_to_string(&events).unwrap().lines().count(), 1);
        let (_, summary) = sink.snapshot().unwrap();
        assert_eq!(summary.branch, "pb/task-harness-1");
        assert_eq!(std::fs::read_to_string(events).unwrap().lines().count(), 1);
    }

    #[test]
    fn running_journal_exists_before_agent_completion() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();
        write_running_journal(&layout, "Build a test project").unwrap();

        let journal = std::fs::read_to_string(layout.journal).unwrap();
        assert!(journal.contains("Status: `running`"));
        assert!(journal.contains("P1 — run has not finalized"));
        assert!(journal.contains("Build a test project"));
    }
}
