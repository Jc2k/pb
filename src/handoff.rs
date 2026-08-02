use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::agent_core::EventSink;
use crate::checks::{CheckRunSummary, EvidenceSource, WorkspaceCheckRuntime, plan_checks};
use crate::events::{
    AgentEvent, AutomationActor, HandoffCheckSummary, HandoffOutcome, HandoffSummary, TeamActor,
    TeamMessageTone,
};
use crate::session_store::now_millis;
use crate::workspace::{RepositoryContext, WorkspaceGraph};

#[derive(Debug, Clone)]
pub enum HandoffAttempt {
    Ready(HandoffSummary),
    NoChange(HandoffSummary),
    NeedsRepair {
        summary: HandoffSummary,
        feedback: String,
        failure_signature: String,
    },
    ExecutorUnavailable {
        summary: HandoffSummary,
        detail: String,
    },
    CommitBlocked {
        summary: HandoffSummary,
        detail: String,
    },
}

impl HandoffAttempt {
    pub fn summary(&self) -> &HandoffSummary {
        match self {
            Self::Ready(summary)
            | Self::NoChange(summary)
            | Self::NeedsRepair { summary, .. }
            | Self::ExecutorUnavailable { summary, .. }
            | Self::CommitBlocked { summary, .. } => summary,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManagedCommitOutcome {
    Created(crate::events::HandoffCommitSummary),
    Reused(crate::events::HandoffCommitSummary),
    NoChange,
    Blocked(String),
}

pub fn run_handoff(
    repository: &RepositoryContext,
    graph: &WorkspaceGraph,
    runtime: &mut WorkspaceCheckRuntime<'_>,
    commit_subject: &str,
    nesting_depth: usize,
    sink: &mut dyn EventSink,
) -> Result<HandoffAttempt> {
    let plan = plan_checks(graph, repository)?;
    let event_nesting_depth = (nesting_depth > 0).then_some(nesting_depth);
    if plan.changed_paths.is_empty() && plan.checks.is_empty() {
        let summary = HandoffSummary {
            outcome: HandoffOutcome::NoChange,
            affected_components: plan.affected_components,
            checks: Vec::new(),
            commit: None,
            changed_paths: plan.changed_paths,
            detail: None,
        };
        sink.emit(AgentEvent::TeamMessage {
            actor: handoff_actor(),
            tone: TeamMessageTone::Info,
            purpose: crate::events::TeamMessagePurpose::HandoffOutcome,
            handoff: Some(summary.clone()),
            message: "There’s no repository change to hand off, so I don’t have anything to test or commit."
                .to_string(),
            detail: None,
            evidence_ids: Vec::new(),
            nesting_depth: event_nesting_depth,
            timestamp_ms: Some(now_millis()),
        });
        emit_summary(sink, summary.clone(), event_nesting_depth);
        return Ok(HandoffAttempt::NoChange(summary));
    }

    let progress_entry_key = if !plan.checks.is_empty() {
        let labels = plan
            .checks
            .iter()
            .filter_map(|id| graph.checks.get(id))
            .map(|check| check.label.as_str())
            .collect::<Vec<_>>();
        Some(sink.emit_keyed(AgentEvent::TeamMessage {
            actor: handoff_actor(),
            tone: TeamMessageTone::Info,
            purpose: crate::events::TeamMessagePurpose::HandoffProgress,
            handoff: None,
            message: format!(
                "I’m checking the affected parts before we wrap this up: {}.",
                natural_list(&labels)
            ),
            detail: None,
            evidence_ids: Vec::new(),
            nesting_depth: event_nesting_depth,
            timestamp_ms: Some(now_millis()),
        }))
    } else {
        None
    };

    let run = match runtime.run_plan(&plan, EvidenceSource::Handoff, nesting_depth, sink) {
        Ok(run) => run,
        Err(error) => {
            let detail = format!("{error:#}");
            let summary = HandoffSummary {
                outcome: HandoffOutcome::ExecutorUnavailable,
                affected_components: plan.affected_components,
                checks: plan
                    .checks
                    .iter()
                    .map(|check_id| HandoffCheckSummary {
                        check_id: check_id.clone(),
                        status: "not_run".to_string(),
                    })
                    .collect(),
                commit: None,
                changed_paths: plan.changed_paths,
                detail: Some(detail.clone()),
            };
            sink.emit_superseding(AgentEvent::TeamMessage {
                actor: handoff_actor(),
                tone: TeamMessageTone::Error,
                purpose: crate::events::TeamMessagePurpose::HandoffOutcome,
                handoff: Some(summary.clone()),
                message: format!(
                    "I couldn’t run the affected checks because their environment is unavailable. The team may need help setting it up before we can finish. {detail}"
                ),
                detail: Some(detail.clone()),
                evidence_ids: Vec::new(),
                nesting_depth: event_nesting_depth,
                timestamp_ms: Some(now_millis()),
            }, progress_entry_key.iter().cloned().collect());
            emit_summary(sink, summary.clone(), event_nesting_depth);
            return Ok(HandoffAttempt::ExecutorUnavailable { summary, detail });
        }
    };
    let checks = handoff_check_summaries(&plan.checks, &run);
    let mut evidence_ids = plan
        .checks
        .iter()
        .map(|id| format!("check:{id}"))
        .collect::<Vec<_>>();

    if run.all_succeeded() {
        let current_changed_paths = repository.task_changed_paths()?;
        let commit = match managed_commit(repository, commit_subject, event_nesting_depth, sink)? {
            ManagedCommitOutcome::Created(commit) | ManagedCommitOutcome::Reused(commit) => {
                Some(commit)
            }
            ManagedCommitOutcome::NoChange => None,
            ManagedCommitOutcome::Blocked(detail) => {
                let summary = HandoffSummary {
                    outcome: HandoffOutcome::CommitBlocked,
                    affected_components: plan.affected_components,
                    checks,
                    commit: None,
                    changed_paths: current_changed_paths,
                    detail: Some(detail.clone()),
                };
                sink.emit_superseding(AgentEvent::TeamMessage {
                    actor: handoff_actor(),
                    tone: TeamMessageTone::Error,
                    purpose: crate::events::TeamMessagePurpose::HandoffOutcome,
                    handoff: Some(summary.clone()),
                    message: format!(
                        "Everything affected passed, but I couldn’t create a safe commit. I left the workspace intact: {detail}"
                    ),
                    detail: Some(detail.clone()),
                    evidence_ids,
                    nesting_depth: event_nesting_depth,
                    timestamp_ms: Some(now_millis()),
                }, progress_entry_key.iter().cloned().collect());
                emit_summary(sink, summary.clone(), event_nesting_depth);
                return Ok(HandoffAttempt::CommitBlocked { summary, detail });
            }
        };
        let outcome = if current_changed_paths.is_empty() {
            HandoffOutcome::NoChange
        } else {
            HandoffOutcome::Ready
        };
        if let Some(commit) = &commit {
            evidence_ids.push(format!("commit:{}", commit.oid));
        }
        let summary = HandoffSummary {
            outcome,
            affected_components: plan.affected_components,
            checks,
            commit: commit.clone(),
            changed_paths: current_changed_paths,
            detail: None,
        };
        let message = if outcome == HandoffOutcome::NoChange {
            "There’s no repository change to hand off. The checks that always run passed, and there’s nothing to commit."
                .to_string()
        } else if plan.checks.is_empty() {
            "I found repository changes, but this project has no applicable checks configured. I’m ready to hand them back."
                .to_string()
        } else if let Some(commit) = &commit {
            format!(
                "Everything affected passed. Kate committed the changes as {}.",
                short_oid(&commit.oid)
            )
        } else {
            "Everything affected passed. The task’s existing commit is ready to hand back."
                .to_string()
        };
        sink.emit_superseding(
            AgentEvent::TeamMessage {
                actor: handoff_actor(),
                tone: TeamMessageTone::Success,
                purpose: crate::events::TeamMessagePurpose::HandoffOutcome,
                handoff: Some(summary.clone()),
                message,
                detail: None,
                evidence_ids,
                nesting_depth: event_nesting_depth,
                timestamp_ms: Some(now_millis()),
            },
            progress_entry_key.iter().cloned().collect(),
        );
        emit_summary(sink, summary.clone(), event_nesting_depth);
        return Ok(if outcome == HandoffOutcome::NoChange {
            HandoffAttempt::NoChange(summary)
        } else {
            HandoffAttempt::Ready(summary)
        });
    }

    let failed = run
        .failed
        .iter()
        .chain(run.skipped.iter())
        .cloned()
        .collect::<Vec<_>>();
    let feedback = repair_feedback(&run);
    let failure_signature = failed
        .iter()
        .map(|id| {
            runtime.ledger().get(id).map_or_else(
                || id.clone(),
                |evidence| {
                    format!(
                        "{}:{}:{}:{}",
                        id,
                        evidence.input_fingerprint,
                        evidence.command_fingerprint,
                        evidence.exit_status
                    )
                },
            )
        })
        .collect::<Vec<_>>()
        .join("|");
    let summary = HandoffSummary {
        outcome: HandoffOutcome::ChecksFailed,
        affected_components: plan.affected_components,
        checks,
        commit: None,
        changed_paths: plan.changed_paths,
        detail: Some(feedback.clone()),
    };
    sink.emit_superseding(
        AgentEvent::TeamMessage {
            actor: handoff_actor(),
            tone: TeamMessageTone::Warning,
            purpose: crate::events::TeamMessagePurpose::HandoffOutcome,
            handoff: Some(summary.clone()),
            message: format!(
                "{} failed. I’ve sent that back to Kate for another pass.",
                natural_list(&failed.iter().map(String::as_str).collect::<Vec<_>>())
            ),
            detail: Some(feedback.clone()),
            evidence_ids,
            nesting_depth: event_nesting_depth,
            timestamp_ms: Some(now_millis()),
        },
        progress_entry_key.iter().cloned().collect(),
    );
    emit_summary(sink, summary.clone(), event_nesting_depth);
    Ok(HandoffAttempt::NeedsRepair {
        summary,
        feedback,
        failure_signature,
    })
}

pub fn managed_commit(
    repository: &RepositoryContext,
    subject: &str,
    nesting_depth: Option<usize>,
    sink: &mut dyn EventSink,
) -> Result<ManagedCommitOutcome> {
    if !crate::agent_core::is_semantic_commit_message(subject) {
        let detail = format!("managed commit subject is not semantic: {subject}");
        emit_commit_result(
            sink,
            false,
            false,
            false,
            None,
            Some(subject.to_string()),
            Vec::new(),
            detail.clone(),
            nesting_depth,
        );
        return Ok(ManagedCommitOutcome::Blocked(detail));
    }

    let changed_paths = repository.task_changed_paths()?;
    if changed_paths.is_empty() {
        return Ok(ManagedCommitOutcome::NoChange);
    }
    let changed = changed_paths.iter().cloned().collect::<BTreeSet<_>>();
    let baseline_dirty = repository
        .task_baseline
        .status
        .dirty_paths
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let overlap = changed
        .intersection(&baseline_dirty)
        .cloned()
        .collect::<Vec<_>>();
    if !overlap.is_empty() {
        let detail = format!(
            "task changes overlap path(s) that were already dirty when the task started: {}",
            overlap.join(", ")
        );
        emit_commit_result(
            sink,
            false,
            false,
            false,
            None,
            Some(subject.to_string()),
            overlap,
            detail.clone(),
            nesting_depth,
        );
        return Ok(ManagedCommitOutcome::Blocked(detail));
    }

    let dirty = current_dirty_paths(&repository.repo_root)?;
    let owned_dirty = dirty
        .into_iter()
        .filter(|path| changed.contains(path))
        .collect::<BTreeSet<_>>();
    let staged = git_path_list(
        &repository.repo_root,
        &["diff", "--cached", "--name-only", "-z"],
    )?;
    let unexpected_staged = staged.difference(&owned_dirty).cloned().collect::<Vec<_>>();
    if !unexpected_staged.is_empty() {
        let detail = format!(
            "the index contains staged path(s) not owned by this task: {}",
            unexpected_staged.join(", ")
        );
        emit_commit_result(
            sink,
            false,
            false,
            false,
            None,
            Some(subject.to_string()),
            unexpected_staged,
            detail.clone(),
            nesting_depth,
        );
        return Ok(ManagedCommitOutcome::Blocked(detail));
    }

    if owned_dirty.is_empty() {
        let commit = current_commit(&repository.repo_root)?.with_context(|| {
            "task content differs from its baseline but no task-owned uncommitted path or commit exists"
        })?;
        if repository.task_baseline.head.as_deref() == Some(commit.oid.as_str()) {
            let detail =
                "task content differs from its baseline but HEAD did not advance".to_string();
            emit_commit_result(
                sink,
                false,
                false,
                false,
                Some(commit.oid),
                Some(commit.subject),
                changed_paths,
                detail.clone(),
                nesting_depth,
            );
            return Ok(ManagedCommitOutcome::Blocked(detail));
        }
        emit_commit_result(
            sink,
            true,
            false,
            true,
            Some(commit.oid.clone()),
            Some(commit.subject.clone()),
            changed_paths,
            "reused the task’s existing commit".to_string(),
            nesting_depth,
        );
        return Ok(ManagedCommitOutcome::Reused(commit));
    }

    let owned_dirty = owned_dirty.into_iter().collect::<Vec<_>>();
    let add = Command::new("git")
        .arg("add")
        .arg("--")
        .args(&owned_dirty)
        .current_dir(&repository.repo_root)
        .output()
        .context("failed to stage task-owned paths")?;
    if !add.status.success() {
        let detail = format!(
            "failed to stage task-owned paths: {}",
            String::from_utf8_lossy(&add.stderr).trim()
        );
        emit_commit_result(
            sink,
            false,
            false,
            false,
            None,
            Some(subject.to_string()),
            owned_dirty,
            detail.clone(),
            nesting_depth,
        );
        return Ok(ManagedCommitOutcome::Blocked(detail));
    }
    let commit_output = Command::new("git")
        .args(["commit", "-m", subject])
        .current_dir(&repository.repo_root)
        .output()
        .context("failed to create managed commit")?;
    if !commit_output.status.success() {
        unstage_task_paths(&repository.repo_root, &owned_dirty);
        let detail = format!(
            "managed commit failed: {}",
            String::from_utf8_lossy(&commit_output.stderr).trim()
        );
        emit_commit_result(
            sink,
            false,
            false,
            false,
            None,
            Some(subject.to_string()),
            owned_dirty,
            detail.clone(),
            nesting_depth,
        );
        return Ok(ManagedCommitOutcome::Blocked(detail));
    }
    let commit = current_commit(&repository.repo_root)?
        .context("managed commit succeeded but HEAD is unavailable")?;
    let remaining = current_dirty_paths(&repository.repo_root)?
        .into_iter()
        .filter(|path| changed.contains(path))
        .collect::<Vec<_>>();
    if !remaining.is_empty() {
        bail!(
            "managed commit left task-owned path(s) uncommitted: {}",
            remaining.join(", ")
        );
    }
    emit_commit_result(
        sink,
        true,
        true,
        false,
        Some(commit.oid.clone()),
        Some(commit.subject.clone()),
        owned_dirty,
        "created a task-owned commit after checks passed".to_string(),
        nesting_depth,
    );
    Ok(ManagedCommitOutcome::Created(commit))
}

fn current_dirty_paths(repo_root: &Path) -> Result<BTreeSet<String>> {
    let mut dirty = BTreeSet::new();
    for args in [
        &["diff", "--name-only", "-z"][..],
        &["diff", "--cached", "--name-only", "-z"][..],
        &["ls-files", "--others", "--exclude-standard", "-z"][..],
    ] {
        dirty.extend(git_path_list(repo_root, args)?);
    }
    Ok(dirty)
}

fn git_path_list(repo_root: &Path, args: &[&str]) -> Result<BTreeSet<String>> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            std::str::from_utf8(path)
                .context("git returned a non-UTF-8 path")
                .map(|path| path.replace('\\', "/"))
        })
        .collect()
}

fn current_commit(repo_root: &Path) -> Result<Option<crate::events::HandoffCommitSummary>> {
    let output = Command::new("git")
        .args(["show", "-s", "--format=%H%x00%s", "HEAD"])
        .current_dir(repo_root)
        .output()
        .context("failed to inspect current commit")?;
    if !output.status.success() {
        return Ok(None);
    }
    let text =
        String::from_utf8(output.stdout).context("git returned non-UTF-8 commit metadata")?;
    let (oid, subject) = text
        .trim_end()
        .split_once('\0')
        .context("git returned malformed commit metadata")?;
    Ok(Some(crate::events::HandoffCommitSummary {
        oid: oid.to_string(),
        subject: subject.to_string(),
    }))
}

fn unstage_task_paths(repo_root: &Path, paths: &[String]) {
    let _ = Command::new("git")
        .arg("reset")
        .arg("--mixed")
        .arg("HEAD")
        .arg("--")
        .args(paths)
        .current_dir(repo_root)
        .output();
}

#[allow(clippy::too_many_arguments)]
fn emit_commit_result(
    sink: &mut dyn EventSink,
    success: bool,
    created: bool,
    reused: bool,
    oid: Option<String>,
    subject: Option<String>,
    changed_paths: Vec<String>,
    detail: String,
    nesting_depth: Option<usize>,
) {
    sink.emit(AgentEvent::CommitResult {
        success,
        created,
        reused,
        oid,
        subject,
        changed_paths,
        detail,
        nesting_depth,
        timestamp_ms: Some(now_millis()),
    });
}

fn short_oid(oid: &str) -> &str {
    oid.get(..8).unwrap_or(oid)
}

fn repair_feedback(run: &CheckRunSummary) -> String {
    let mut feedback = String::from(
        "The handoff teammate found check failures. Fix these concrete failures, rerun only what is useful while working, then return a final handoff response again:\n",
    );
    for failure in &run.failures {
        feedback.push_str(&format!(
            "\n- {} (exit {}, timed_out={}):\n",
            failure.check_id, failure.exit_status, failure.timed_out
        ));
        if let Some(reason) = &failure.skip_reason {
            feedback.push_str(reason);
        } else {
            feedback.push_str(failure.output.trim());
        }
        feedback.push('\n');
    }
    truncate_chars(&feedback, 12_000)
}

fn handoff_check_summaries(ids: &[String], run: &CheckRunSummary) -> Vec<HandoffCheckSummary> {
    ids.iter()
        .map(|check_id| {
            let status = if run.failed.contains(check_id) {
                "failed"
            } else if run.skipped.contains(check_id) {
                "skipped"
            } else if run.reused.contains(check_id) {
                "reused"
            } else {
                "passed"
            };
            HandoffCheckSummary {
                check_id: check_id.clone(),
                status: status.to_string(),
            }
        })
        .collect()
}

fn emit_summary(sink: &mut dyn EventSink, summary: HandoffSummary, nesting_depth: Option<usize>) {
    sink.emit(AgentEvent::HandoffSummary {
        summary,
        nesting_depth,
        timestamp_ms: Some(now_millis()),
    });
}

fn handoff_actor() -> TeamActor {
    TeamActor::Automation(AutomationActor::Trinity)
}

fn natural_list(values: &[&str]) -> String {
    match values {
        [] => "the configured project checks".to_string(),
        [one] => (*one).to_string(),
        [left, right] => format!("{left} and {right}"),
        many => format!(
            "{}, and {}",
            many[..many.len() - 1].join(", "),
            many[many.len() - 1]
        ),
    }
}

fn truncate_chars(value: &str, limit: usize) -> String {
    let mut chars = value.chars();
    let prefix = chars.by_ref().take(limit).collect::<String>();
    if chars.next().is_some() {
        format!("{prefix}\n[handoff feedback truncated]")
    } else {
        prefix
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::process::Command;

    use super::*;
    use crate::checks::CheckEvidenceLedger;
    use crate::events::AgentEvent;
    use crate::workspace::{
        CheckTrigger, Executor, ExecutorKind, WorkspaceCheck, WorkspaceComponent,
        WorkspaceGraphSource,
    };

    fn init_repo() -> tempfile::TempDir {
        let repo = tempfile::tempdir().unwrap();
        for args in [
            &["init", "--initial-branch=main"][..],
            &["config", "user.name", "pb handoff test"][..],
            &["config", "user.email", "handoff@pb.local"][..],
        ] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(repo.path())
                    .status()
                    .unwrap()
                    .success()
            );
        }
        std::fs::write(repo.path().join("app.txt"), "one\n").unwrap();
        assert!(
            Command::new("git")
                .args(["add", "app.txt"])
                .current_dir(repo.path())
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .args(["commit", "-m", "test: fixture"])
                .current_dir(repo.path())
                .status()
                .unwrap()
                .success()
        );
        repo
    }

    fn graph(command: &str, trigger: CheckTrigger) -> WorkspaceGraph {
        WorkspaceGraph {
            version: 1,
            executors: BTreeMap::from([(
                "local".to_string(),
                Executor {
                    id: "local".to_string(),
                    kind: ExecutorKind::Local,
                    environment: None,
                },
            )]),
            components: BTreeMap::from([(
                "app".to_string(),
                WorkspaceComponent {
                    id: "app".to_string(),
                    root: ".".to_string(),
                    include: vec!["**".to_string()],
                    exclude: Vec::new(),
                    executor: "local".to_string(),
                    depends_on: Vec::new(),
                },
            )]),
            checks: BTreeMap::from([(
                "app-test".to_string(),
                WorkspaceCheck {
                    id: "app-test".to_string(),
                    label: "app tests".to_string(),
                    command: command.to_string(),
                    cwd: ".".to_string(),
                    executor: "local".to_string(),
                    components: vec!["app".to_string()],
                    trigger,
                    inputs: vec!["**".to_string()],
                    outputs: Vec::new(),
                    depends_on: Vec::new(),
                    timeout_seconds: 5,
                },
            )]),
            tasks: BTreeMap::new(),
            cargo_workspaces: BTreeMap::new(),
            discovery_warnings: Vec::new(),
            source: WorkspaceGraphSource::Explicit,
        }
    }

    fn runtime<'a>(root: &'a Path, graph: &'a WorkspaceGraph) -> WorkspaceCheckRuntime<'a> {
        WorkspaceCheckRuntime::new(root, graph, None, None, CheckEvidenceLedger::default())
    }

    fn git(root: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    }

    #[test]
    fn unchanged_task_stays_out_of_the_way_when_no_always_check_applies() {
        let repo = init_repo();
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        let graph = graph("true", CheckTrigger::Changed);
        let mut runtime = runtime(repo.path(), &graph);
        let mut events = Vec::new();
        let attempt = run_handoff(
            &repository,
            &graph,
            &mut runtime,
            "test: hand off fixture",
            0,
            &mut |event| events.push(event),
        )
        .unwrap();
        assert!(matches!(attempt, HandoffAttempt::NoChange(_)));
        assert!(!events.iter().any(|event| matches!(
            event,
            AgentEvent::ExecutorStarted { .. } | AgentEvent::CheckResult { .. }
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::TeamMessage { message, .. } if message.contains("anything to test or commit")
        )));
    }

    #[test]
    fn failed_check_returns_bounded_actionable_feedback() {
        let repo = init_repo();
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        std::fs::write(repo.path().join("app.txt"), "two\n").unwrap();
        let graph = graph("echo broken >&2; exit 7", CheckTrigger::Changed);
        let mut runtime = runtime(repo.path(), &graph);
        let mut events = Vec::new();
        let attempt = run_handoff(
            &repository,
            &graph,
            &mut runtime,
            "test: hand off fixture",
            0,
            &mut |event| events.push(event),
        )
        .unwrap();
        let HandoffAttempt::NeedsRepair { feedback, .. } = attempt else {
            panic!("expected repair handoff");
        };
        assert!(feedback.contains("app-test"));
        assert!(feedback.contains("broken"));
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::TeamMessage { message, .. } if message.contains("Kate")
        )));
        assert_eq!(
            git(repo.path(), &["rev-parse", "HEAD"]).trim(),
            repository.task_baseline.head.as_deref().unwrap()
        );
    }

    #[test]
    fn always_check_runs_for_no_change_without_requesting_a_commit() {
        let repo = init_repo();
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        let graph = graph("true", CheckTrigger::Always);
        let mut runtime = runtime(repo.path(), &graph);
        let attempt = run_handoff(
            &repository,
            &graph,
            &mut runtime,
            "test: hand off fixture",
            0,
            &mut |_| {},
        )
        .unwrap();
        let HandoffAttempt::NoChange(summary) = attempt else {
            panic!("expected no-change handoff");
        };
        assert_eq!(summary.checks[0].status, "passed");
        assert!(summary.commit.is_none());
    }

    #[test]
    fn successful_handoff_creates_one_task_owned_commit() {
        let repo = init_repo();
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        std::fs::write(repo.path().join("app.txt"), "two\n").unwrap();
        let graph = graph("true", CheckTrigger::Changed);
        let mut runtime = runtime(repo.path(), &graph);
        let attempt = run_handoff(
            &repository,
            &graph,
            &mut runtime,
            "feat: update app fixture",
            0,
            &mut |_| {},
        )
        .unwrap();
        let HandoffAttempt::Ready(summary) = attempt else {
            panic!("expected ready handoff");
        };
        assert_eq!(summary.commit.unwrap().subject, "feat: update app fixture");
        assert!(git(repo.path(), &["status", "--porcelain"]).is_empty());
        assert_eq!(
            git(repo.path(), &["show", "--pretty=", "--name-only", "HEAD"]).trim(),
            "app.txt"
        );
    }

    #[test]
    fn managed_commit_preserves_preexisting_unstaged_and_untracked_work() {
        let repo = init_repo();
        std::fs::write(repo.path().join("user.txt"), "base\n").unwrap();
        git(repo.path(), &["add", "user.txt"]);
        git(repo.path(), &["commit", "-m", "test: add user fixture"]);
        std::fs::write(repo.path().join("user.txt"), "user work\n").unwrap();
        std::fs::write(repo.path().join("notes.txt"), "private notes\n").unwrap();
        std::fs::create_dir_all(repo.path().join(".pb")).unwrap();
        std::fs::write(repo.path().join(".pb/private.toml"), "user_owned = true\n").unwrap();
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        std::fs::write(repo.path().join("task.txt"), "task work\n").unwrap();

        let outcome =
            managed_commit(&repository, "feat: add task fixture", None, &mut |_| {}).unwrap();
        assert!(matches!(outcome, ManagedCommitOutcome::Created(_)));
        assert_eq!(
            git(repo.path(), &["show", "--pretty=", "--name-only", "HEAD"]).trim(),
            "task.txt"
        );
        assert_eq!(
            std::fs::read_to_string(repo.path().join("user.txt")).unwrap(),
            "user work\n"
        );
        assert_eq!(
            std::fs::read_to_string(repo.path().join("notes.txt")).unwrap(),
            "private notes\n"
        );
        assert_eq!(
            std::fs::read_to_string(repo.path().join(".pb/private.toml")).unwrap(),
            "user_owned = true\n"
        );
        let status = git(repo.path(), &["status", "--porcelain"]);
        assert!(status.contains("user.txt"));
        assert!(status.contains("notes.txt"));
        assert!(status.contains(".pb/"));
        assert!(!status.contains("task.txt"));
    }

    #[test]
    fn resumed_uncommitted_work_remains_task_owned_checkable_and_committable() {
        let repo = init_repo();
        let task_context = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        std::fs::write(repo.path().join("task.txt"), "work from an earlier run\n").unwrap();

        let repository =
            RepositoryContext::resume(repo.path(), repo.path(), task_context.task_baseline)
                .unwrap();
        assert_ne!(
            repository.task_baseline.id,
            repository.invocation_baseline.id
        );
        assert_eq!(repository.task_changed_paths().unwrap(), vec!["task.txt"]);

        let graph = graph("test -f task.txt", CheckTrigger::Changed);
        let mut runtime = runtime(repo.path(), &graph);
        let mut events = Vec::new();
        let attempt = run_handoff(
            &repository,
            &graph,
            &mut runtime,
            "feat: finish resumed task fixture",
            0,
            &mut |event| events.push(event),
        )
        .unwrap();

        let HandoffAttempt::Ready(summary) = attempt else {
            panic!("expected ready handoff");
        };
        assert_eq!(summary.changed_paths, vec!["task.txt"]);
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::CheckResult {
                success: true,
                reused: false,
                ..
            }
        )));
        assert_eq!(
            git(repo.path(), &["show", "--pretty=", "--name-only", "HEAD"]).trim(),
            "task.txt"
        );
    }

    #[test]
    fn unrelated_staged_content_blocks_commit_without_changing_the_index() {
        let repo = init_repo();
        std::fs::write(repo.path().join("staged.txt"), "base\n").unwrap();
        git(repo.path(), &["add", "staged.txt"]);
        git(repo.path(), &["commit", "-m", "test: add staged fixture"]);
        std::fs::write(repo.path().join("staged.txt"), "user staged\n").unwrap();
        git(repo.path(), &["add", "staged.txt"]);
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        std::fs::write(repo.path().join("task.txt"), "task work\n").unwrap();
        let index_before = git(repo.path(), &["diff", "--cached", "--binary"]);
        let head_before = git(repo.path(), &["rev-parse", "HEAD"]);

        let outcome =
            managed_commit(&repository, "feat: add task fixture", None, &mut |_| {}).unwrap();
        let ManagedCommitOutcome::Blocked(detail) = outcome else {
            panic!("expected blocked commit");
        };
        assert!(detail.contains("staged.txt"));
        assert_eq!(
            index_before,
            git(repo.path(), &["diff", "--cached", "--binary"])
        );
        assert_eq!(head_before, git(repo.path(), &["rev-parse", "HEAD"]));
    }

    #[test]
    fn overlapping_dirty_path_blocks_commit() {
        let repo = init_repo();
        std::fs::write(repo.path().join("app.txt"), "user work\n").unwrap();
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        std::fs::write(repo.path().join("app.txt"), "task overwrite\n").unwrap();
        let outcome = managed_commit(
            &repository,
            "feat: overwrite app fixture",
            None,
            &mut |_| {},
        )
        .unwrap();
        let ManagedCommitOutcome::Blocked(detail) = outcome else {
            panic!("expected overlapping path to block commit");
        };
        assert!(detail.contains("app.txt"));
    }

    #[test]
    fn existing_task_commit_is_reused_without_duplication() {
        let repo = init_repo();
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        std::fs::write(repo.path().join("task.txt"), "task work\n").unwrap();
        git(repo.path(), &["add", "task.txt"]);
        git(repo.path(), &["commit", "-m", "feat: agent task commit"]);
        let head_before = git(repo.path(), &["rev-parse", "HEAD"]);

        let outcome = managed_commit(
            &repository,
            "feat: redundant handoff commit",
            None,
            &mut |_| {},
        )
        .unwrap();
        let ManagedCommitOutcome::Reused(commit) = outcome else {
            panic!("expected existing commit reuse");
        };
        assert_eq!(commit.subject, "feat: agent task commit");
        assert_eq!(head_before, git(repo.path(), &["rev-parse", "HEAD"]));
    }
}
