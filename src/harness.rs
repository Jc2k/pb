//! Direct, daemon-free harnesses for exercising pb internals.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

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
    "run_command",
    "run_check",
    "read_file",
    "write_file",
    "replace_file",
    "edit_file",
    "apply_patch",
    "rm",
    "git_commit",
    "sub_agent",
];

#[derive(Debug)]
struct ScratchLayout {
    root: PathBuf,
    workspace: PathBuf,
    events: PathBuf,
    journal: PathBuf,
    run_index: PathBuf,
    run_id: String,
    run_events: PathBuf,
    run_journal: PathBuf,
    resumed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
enum RunIndexRecord {
    Started {
        version: u32,
        run_id: String,
        timestamp_ms: u64,
        task: String,
        run_events: String,
        run_journal: String,
    },
    Finished {
        version: u32,
        run_id: String,
        timestamp_ms: u64,
        status: String,
        reached_final: bool,
        contract_status: crate::events::ContractStatus,
        verified_completed: bool,
        termination_reason: Option<crate::events::TerminationReason>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
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
    cumulative_writer: BufWriter<File>,
    run_writer: BufWriter<File>,
    observations: Vec<Observation>,
    summary: CapturedSummary,
    write_error: Option<String>,
}

#[derive(Clone)]
struct HarnessEventSink {
    state: Arc<Mutex<JournalState>>,
}

impl HarnessEventSink {
    fn new(cumulative_path: &Path, run_path: &Path) -> Result<Self> {
        let cumulative_file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(cumulative_path)
            .with_context(|| {
                format!(
                    "failed to open cumulative harness event journal {}",
                    cumulative_path.display()
                )
            })?;
        let run_file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(run_path)
            .with_context(|| {
                format!(
                    "failed to create per-run harness event journal {}",
                    run_path.display()
                )
            })?;
        Ok(Self {
            state: Arc::new(Mutex::new(JournalState {
                cumulative_writer: BufWriter::new(cumulative_file),
                run_writer: BufWriter::new(run_file),
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
        let mut flush_errors = Vec::new();
        if let Err(error) = state.cumulative_writer.flush() {
            flush_errors.push(format!("cumulative stream: {error}"));
        }
        if let Err(error) = state.run_writer.flush() {
            flush_errors.push(format!("per-run stream: {error}"));
        }
        if !flush_errors.is_empty() {
            bail!(
                "failed to flush harness event journals: {}",
                flush_errors.join("; ")
            );
        }
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
        let encoded = match serde_json::to_vec(&envelope) {
            Ok(encoded) => encoded,
            Err(error) => {
                state.write_error = Some(format!("event serialization: {error}"));
                return;
            }
        };
        let mut write_errors = Vec::new();
        if let Err(error) = write_event_line(&mut state.cumulative_writer, &encoded) {
            write_errors.push(format!("cumulative stream: {error}"));
        }
        if let Err(error) = write_event_line(&mut state.run_writer, &encoded) {
            write_errors.push(format!("per-run stream: {error}"));
        }
        if !write_errors.is_empty() {
            state.write_error = Some(write_errors.join("; "));
        }
    }
}

fn write_event_line(writer: &mut BufWriter<File>, encoded: &[u8]) -> std::io::Result<()> {
    writer.write_all(encoded)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

pub fn run_agent_task(args: HarnessAgentArgs) -> Result<()> {
    if args.task.trim().is_empty() {
        bail!("harness agent task must not be empty");
    }

    let contract = args
        .contract
        .as_deref()
        .map(crate::harness_contract::HarnessContractDocument::from_path)
        .transpose()?
        .map(crate::harness_contract::HarnessContractDocument::normalize)
        .transpose()?;
    let layout = prepare_scratch(args.scratch_dir.as_deref())?;
    println!("pb harness: scratch={}", layout.root.display());
    println!("pb harness: workspace={}", layout.workspace.display());
    println!("pb harness: events={}", layout.events.display());
    println!("pb harness: journal={}", layout.journal.display());
    println!("pb harness: run_id={}", layout.run_id);
    println!("pb harness: run_events={}", layout.run_events.display());
    println!("pb harness: run_journal={}", layout.run_journal.display());
    println!("pb harness: resumed={}", layout.resumed);

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
        accept_existing_workspace_changes: layout.resumed,
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
        session_id: format!("harness-{}", layout.run_id),
        attachments: harness_attachments(&args.images)?,
        contract,
    };

    write_running_journal(&layout, &args.task)?;
    append_run_index_started(&layout, &args.task)?;
    let sink = HarnessEventSink::new(&layout.events, &layout.run_events)?;
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
    append_run_index_finished(&layout, &run_result)?;

    match run_result {
        Ok(result) if harness_outcome_succeeded(&result) => {
            println!(
                "pb harness: reached_final={} contract_status={} verified_completed={} termination_reason={} branch={} workspace={} journal={}",
                result.reached_final,
                result.contract_status,
                result.verified_completed,
                result.termination_reason,
                result.branch,
                result.workspace_root.display(),
                layout.journal.display()
            );
            Ok(())
        }
        Ok(result)
            if result.termination_reason
                == crate::events::TerminationReason::ContractUnsatisfied =>
        {
            bail!(
                "harness agent final did not satisfy its acceptance contract; reached_final={} contract_status={} verified_completed={} termination_reason={} workspace={} journal={}",
                result.reached_final,
                result.contract_status,
                result.verified_completed,
                result.termination_reason,
                result.workspace_root.display(),
                layout.journal.display()
            )
        }
        Ok(result) => bail!(
            "harness agent did not complete; reached_final={} contract_status={} verified_completed={} termination_reason={} workspace={} journal={}",
            result.reached_final,
            result.contract_status,
            result.verified_completed,
            result.termination_reason,
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

fn harness_outcome_succeeded(result: &AgentRunResult) -> bool {
    result.verified_completed
        || (result.reached_final
            && result.contract_status == crate::events::ContractStatus::Unspecified)
}

fn append_run_index_started(layout: &ScratchLayout, task: &str) -> Result<()> {
    append_run_index_record(
        &layout.run_index,
        &RunIndexRecord::Started {
            version: 1,
            run_id: layout.run_id.clone(),
            timestamp_ms: now_millis(),
            task: task.to_string(),
            run_events: relative_to_root(&layout.root, &layout.run_events),
            run_journal: relative_to_root(&layout.root, &layout.run_journal),
        },
    )
}

fn append_run_index_finished(
    layout: &ScratchLayout,
    result: &Result<AgentRunResult>,
) -> Result<()> {
    let (status, reached_final, contract_status, verified_completed, termination_reason, error) =
        match result {
            Ok(result) if result.verified_completed => (
                "verified_completed",
                result.reached_final,
                result.contract_status,
                true,
                Some(result.termination_reason),
                None,
            ),
            Ok(result)
                if result.reached_final
                    && result.contract_status == crate::events::ContractStatus::Unspecified =>
            {
                (
                    "final_unverified",
                    true,
                    result.contract_status,
                    false,
                    Some(result.termination_reason),
                    None,
                )
            }
            Ok(result) => (
                "incomplete",
                result.reached_final,
                result.contract_status,
                false,
                Some(result.termination_reason),
                None,
            ),
            Err(error) => (
                "failed",
                false,
                crate::events::ContractStatus::Unspecified,
                false,
                Some(crate::events::TerminationReason::EngineError),
                Some(compact_detail(&format!("{error:#}"))),
            ),
        };
    append_run_index_record(
        &layout.run_index,
        &RunIndexRecord::Finished {
            version: 1,
            run_id: layout.run_id.clone(),
            timestamp_ms: now_millis(),
            status: status.to_string(),
            reached_final,
            contract_status,
            verified_completed,
            termination_reason,
            error,
        },
    )
}

fn append_run_index_record(path: &Path, record: &RunIndexRecord) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open harness run index {}", path.display()))?;
    serde_json::to_writer(&mut file, record)
        .with_context(|| format!("failed to encode harness run index {}", path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("failed to append harness run index {}", path.display()))?;
    file.sync_data()
        .with_context(|| format!("failed to sync harness run index {}", path.display()))
}

fn relative_to_root(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn prepare_scratch(requested: Option<&Path>) -> Result<ScratchLayout> {
    let (root, resumed) = match requested {
        Some(path) => {
            if path.exists() {
                if !path.is_dir() {
                    bail!(
                        "harness scratch path is not a directory: {}",
                        path.display()
                    );
                }
                (path.to_path_buf(), true)
            } else {
                std::fs::create_dir_all(path).with_context(|| {
                    format!("failed to create harness scratch {}", path.display())
                })?;
                (path.to_path_buf(), false)
            }
        }
        None => (create_unique_scratch_root()?, false),
    };
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve harness scratch {}", root.display()))?;
    let workspace = root.join("workspace");
    if resumed {
        if !workspace.join(".git").is_dir() {
            bail!(
                "existing harness scratch has no git workspace: {}",
                workspace.display()
            );
        }
    } else {
        std::fs::create_dir(&workspace).with_context(|| {
            format!("failed to create harness workspace {}", workspace.display())
        })?;
        initialize_git_workspace(&workspace)?;
    }
    let runs = root.join("runs");
    std::fs::create_dir_all(&runs)
        .with_context(|| format!("failed to create harness runs directory {}", runs.display()))?;
    let (run_id, run_dir) = create_unique_run_dir(&runs)?;
    Ok(ScratchLayout {
        events: root.join("events.jsonl"),
        journal: root.join("journal.md"),
        run_index: root.join("run-index.jsonl"),
        run_events: run_dir.join("events.jsonl"),
        run_journal: run_dir.join("journal.md"),
        run_id,
        root,
        workspace,
        resumed,
    })
}

fn create_unique_run_dir(runs: &Path) -> Result<(String, PathBuf)> {
    let timestamp = now_millis();
    let pid = std::process::id();
    for suffix in 0..1000u16 {
        let run_id = format!("{timestamp}-{pid}-{suffix}");
        let path = runs.join(&run_id);
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok((run_id, path)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create harness run directory {}", path.display())
                });
            }
        }
    }
    bail!("could not allocate a unique harness run directory")
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
        Ok(result) if !result.verified_completed && result.contract_status != crate::events::ContractStatus::Unspecified => {
            observations.push(Observation {
                rank: 0,
                title: "acceptance contract was not satisfied".to_string(),
                detail: format!(
                    "The model emitted a final action, but the run terminated as {} with contract_status={}.",
                    result.termination_reason, result.contract_status
                ),
            })
        }
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
        Ok(result) if result.verified_completed => "verified-completed",
        Ok(result)
            if result.reached_final
                && result.contract_status == crate::events::ContractStatus::Unspecified =>
        {
            "final-unverified"
        }
        Ok(result) if result.reached_final => "contract-unsatisfied",
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
    journal.push_str(&format!("- Run ID: `{}`\n", layout.run_id));
    if let Ok(result) = result {
        journal.push_str(&format!("- Reached final: `{}`\n", result.reached_final));
        journal.push_str(&format!(
            "- Contract status: `{}`\n",
            result.contract_status
        ));
        journal.push_str(&format!(
            "- Verified completed: `{}`\n",
            result.verified_completed
        ));
        journal.push_str(&format!(
            "- Termination reason: `{}`\n",
            result.termination_reason
        ));
    }
    journal.push_str(&format!("- Task: {task}\n"));
    journal.push_str(&format!("- Workspace: `{}`\n", layout.workspace.display()));
    journal.push_str(&format!("- Branch: `{branch}`\n"));
    journal.push_str(&format!(
        "- Run events: `{}`\n",
        layout.run_events.display()
    ));
    journal.push_str(&format!(
        "- Cumulative events: `{}`\n",
        layout.events.display()
    ));
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
    atomic_write(&layout.run_journal, journal.as_bytes())?;
    atomic_write(&layout.journal, journal.as_bytes())
}

fn write_running_journal(layout: &ScratchLayout, task: &str) -> Result<()> {
    let journal = format!(
        "# pb harness journal\n\n\
         - Status: `running`\n\
         - Run ID: `{run_id}`\n\
         - Task: {task}\n\
         - Workspace: `{workspace}`\n\
         - Run events: `{run_events}`\n\
         - Cumulative events: `{events}`\n\n\
         ## Ranked observations\n\n\
         1. **P1 — run has not finalized.** If the harness was interrupted, inspect the raw event stream and workspace before deciding whether to rerun.\n\n\
         ## Follow-up improvement plan\n\n\
         - [ ] Wait for the blocking agent run to finish, or diagnose why it was interrupted.\n\
         - [ ] Review the workspace and raw events before making changes to pb.\n",
        workspace = layout.workspace.display(),
        run_id = layout.run_id,
        run_events = layout.run_events.display(),
        events = layout.events.display(),
    );
    atomic_write(&layout.run_journal, journal.as_bytes())?;
    atomic_write(&layout.journal, journal.as_bytes())
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("harness audit path has no parent: {}", path.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("audit");
    for suffix in 0..100u8 {
        let temporary = parent.join(format!(
            ".{file_name}.{}.{}.{suffix}.tmp",
            std::process::id(),
            now_millis()
        ));
        let mut file = match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create atomic audit file {}", temporary.display())
                });
            }
        };
        let write_result = file
            .write_all(contents)
            .and_then(|_| file.sync_all())
            .and_then(|_| std::fs::rename(&temporary, path));
        if let Err(error) = write_result {
            let _ = std::fs::remove_file(&temporary);
            return Err(error)
                .with_context(|| format!("failed to atomically write {}", path.display()));
        }
        return Ok(());
    }
    bail!(
        "could not allocate atomic audit file for {}",
        path.display()
    )
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
    fn direct_agent_tool_surface_is_minimal_but_complete() {
        assert_eq!(
            HARNESS_AGENT_TOOLS,
            [
                "session_title",
                "run_command",
                "run_check",
                "read_file",
                "write_file",
                "replace_file",
                "edit_file",
                "apply_patch",
                "rm",
                "git_commit",
                "sub_agent"
            ]
        );
    }

    #[test]
    fn harness_exit_success_requires_legacy_final_or_verified_contract() {
        let result = |reached_final, contract_status, verified_completed, termination_reason| {
            AgentRunResult {
                branch: "task".to_string(),
                workspace_root: PathBuf::from("/tmp/task"),
                reached_final,
                contract_status,
                verified_completed,
                termination_reason,
            }
        };

        assert!(harness_outcome_succeeded(&result(
            true,
            crate::events::ContractStatus::Unspecified,
            false,
            crate::events::TerminationReason::Final,
        )));
        assert!(harness_outcome_succeeded(&result(
            true,
            crate::events::ContractStatus::Satisfied,
            true,
            crate::events::TerminationReason::Final,
        )));
        assert!(!harness_outcome_succeeded(&result(
            true,
            crate::events::ContractStatus::Unsatisfied,
            false,
            crate::events::TerminationReason::ContractUnsatisfied,
        )));
    }

    #[test]
    fn scratch_workspace_is_persistent_git_repository() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();

        assert_eq!(layout.root, root.canonicalize().unwrap());
        assert!(!layout.resumed);
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
    fn existing_scratch_workspace_can_be_resumed() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let initial = prepare_scratch(Some(&root)).unwrap();
        std::fs::write(initial.workspace.join("work.txt"), "in progress\n").unwrap();

        let resumed = prepare_scratch(Some(&root)).unwrap();

        assert!(resumed.resumed);
        assert_eq!(
            std::fs::read_to_string(resumed.workspace.join("work.txt")).unwrap(),
            "in progress\n"
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
        let run_events = parent.path().join("run-events.jsonl");
        let sink = HarnessEventSink::new(&events, &run_events).unwrap();
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
        assert_eq!(
            std::fs::read_to_string(run_events).unwrap().lines().count(),
            1
        );
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

    fn finish_test_run(layout: &ScratchLayout, task: &str, branch: &str) {
        write_running_journal(layout, task).unwrap();
        append_run_index_started(layout, task).unwrap();
        let sink = HarnessEventSink::new(&layout.events, &layout.run_events).unwrap();
        let mut emitter = sink.clone();
        emitter.emit(AgentEvent::Started {
            task: task.to_string(),
            model: "scripted".to_string(),
            workspace: layout.workspace.display().to_string(),
            branch: branch.to_string(),
            attachments: Vec::new(),
            timestamp_ms: None,
        });
        let (_, summary) = sink.snapshot().unwrap();
        let result = Ok(AgentRunResult {
            branch: branch.to_string(),
            workspace_root: layout.workspace.clone(),
            reached_final: true,
            contract_status: crate::events::ContractStatus::Unspecified,
            verified_completed: false,
            termination_reason: crate::events::TerminationReason::Final,
        });
        write_journal(layout, task, &result, &summary, &mut Vec::new()).unwrap();
        append_run_index_finished(layout, &result).unwrap();
    }

    #[test]
    fn resumed_invocations_preserve_immutable_per_run_audits() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let first = prepare_scratch(Some(&root)).unwrap();
        finish_test_run(&first, "first task", "task-first");
        let first_events = std::fs::read(&first.run_events).unwrap();
        let first_journal = std::fs::read(&first.run_journal).unwrap();

        let second = prepare_scratch(Some(&root)).unwrap();
        finish_test_run(&second, "second task", "task-second");

        assert_ne!(first.run_id, second.run_id);
        assert_eq!(std::fs::read(&first.run_events).unwrap(), first_events);
        assert_eq!(std::fs::read(&first.run_journal).unwrap(), first_journal);
        assert_eq!(
            std::fs::read_to_string(&second.events)
                .unwrap()
                .lines()
                .count(),
            2
        );
        assert_eq!(
            std::fs::read_to_string(&second.run_index)
                .unwrap()
                .lines()
                .count(),
            4
        );
        let latest = std::fs::read_to_string(&second.journal).unwrap();
        assert!(latest.contains(&format!("Run ID: `{}`", second.run_id)));
        assert!(latest.contains("second task"));
    }

    #[test]
    fn interrupted_run_is_discoverable_from_started_index_record() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("run");
        let layout = prepare_scratch(Some(&root)).unwrap();
        write_running_journal(&layout, "interrupted task").unwrap();
        append_run_index_started(&layout, "interrupted task").unwrap();

        let records = std::fs::read_to_string(&layout.run_index).unwrap();
        let records = records
            .lines()
            .map(|line| serde_json::from_str::<RunIndexRecord>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 1);
        assert!(matches!(
            &records[0],
            RunIndexRecord::Started { run_id, task, .. }
                if run_id == &layout.run_id && task == "interrupted task"
        ));
        let journal = std::fs::read_to_string(&layout.run_journal).unwrap();
        assert!(journal.contains("Status: `running`"));
        assert!(journal.contains(&format!("Run ID: `{}`", layout.run_id)));
        assert!(!layout.run_events.exists());
    }
}
