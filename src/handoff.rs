use anyhow::Result;

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
}

impl HandoffAttempt {
    pub fn summary(&self) -> &HandoffSummary {
        match self {
            Self::Ready(summary)
            | Self::NoChange(summary)
            | Self::NeedsRepair { summary, .. }
            | Self::ExecutorUnavailable { summary, .. } => summary,
        }
    }
}

pub fn run_handoff(
    repository: &RepositoryContext,
    graph: &WorkspaceGraph,
    runtime: &mut WorkspaceCheckRuntime<'_>,
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
            message: "There’s no repository change to hand off, so I don’t have anything to test or commit."
                .to_string(),
            evidence_ids: Vec::new(),
            nesting_depth: event_nesting_depth,
            timestamp_ms: Some(now_millis()),
        });
        emit_summary(sink, summary.clone(), event_nesting_depth);
        return Ok(HandoffAttempt::NoChange(summary));
    }

    if !plan.checks.is_empty() {
        let labels = plan
            .checks
            .iter()
            .filter_map(|id| graph.checks.get(id))
            .map(|check| check.label.as_str())
            .collect::<Vec<_>>();
        sink.emit(AgentEvent::TeamMessage {
            actor: handoff_actor(),
            tone: TeamMessageTone::Info,
            message: format!(
                "I’m checking the affected parts before we wrap this up: {}.",
                natural_list(&labels)
            ),
            evidence_ids: Vec::new(),
            nesting_depth: event_nesting_depth,
            timestamp_ms: Some(now_millis()),
        });
    }

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
            sink.emit(AgentEvent::TeamMessage {
                actor: handoff_actor(),
                tone: TeamMessageTone::Error,
                message: format!(
                    "I couldn’t run the affected checks because their environment is unavailable. The team may need help setting it up before we can finish. {detail}"
                ),
                evidence_ids: Vec::new(),
                nesting_depth: event_nesting_depth,
                timestamp_ms: Some(now_millis()),
            });
            emit_summary(sink, summary.clone(), event_nesting_depth);
            return Ok(HandoffAttempt::ExecutorUnavailable { summary, detail });
        }
    };
    let checks = handoff_check_summaries(&plan.checks, &run);
    let evidence_ids = plan
        .checks
        .iter()
        .map(|id| format!("check:{id}"))
        .collect::<Vec<_>>();

    if run.all_succeeded() {
        let outcome = if plan.changed_paths.is_empty() {
            HandoffOutcome::NoChange
        } else {
            HandoffOutcome::Ready
        };
        let summary = HandoffSummary {
            outcome,
            affected_components: plan.affected_components,
            checks,
            commit: None,
            changed_paths: plan.changed_paths,
            detail: None,
        };
        let message = if outcome == HandoffOutcome::NoChange {
            "There’s no repository change to hand off. The checks that always run passed, and there’s nothing to commit."
                .to_string()
        } else if plan.checks.is_empty() {
            "I found repository changes, but this project has no applicable checks configured. I’m ready to hand them back."
                .to_string()
        } else {
            "Everything affected passed. I’m ready to hand the repository changes back.".to_string()
        };
        sink.emit(AgentEvent::TeamMessage {
            actor: handoff_actor(),
            tone: TeamMessageTone::Success,
            message,
            evidence_ids,
            nesting_depth: event_nesting_depth,
            timestamp_ms: Some(now_millis()),
        });
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
    sink.emit(AgentEvent::TeamMessage {
        actor: handoff_actor(),
        tone: TeamMessageTone::Warning,
        message: format!(
            "{} failed. I’ve sent that back to Kate for another pass.",
            natural_list(&failed.iter().map(String::as_str).collect::<Vec<_>>())
        ),
        evidence_ids,
        nesting_depth: event_nesting_depth,
        timestamp_ms: Some(now_millis()),
    });
    emit_summary(sink, summary.clone(), event_nesting_depth);
    Ok(HandoffAttempt::NeedsRepair {
        summary,
        feedback,
        failure_signature,
    })
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
    TeamActor::Automation(AutomationActor::Handoff)
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
            cargo_workspaces: BTreeMap::new(),
            discovery_warnings: Vec::new(),
            source: WorkspaceGraphSource::Explicit,
        }
    }

    fn runtime<'a>(root: &'a Path, graph: &'a WorkspaceGraph) -> WorkspaceCheckRuntime<'a> {
        WorkspaceCheckRuntime::new(root, graph, None, None, CheckEvidenceLedger::default())
    }

    #[test]
    fn unchanged_task_stays_out_of_the_way_when_no_always_check_applies() {
        let repo = init_repo();
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        let graph = graph("true", CheckTrigger::Changed);
        let mut runtime = runtime(repo.path(), &graph);
        let mut events = Vec::new();
        let attempt = run_handoff(&repository, &graph, &mut runtime, 0, &mut |event| {
            events.push(event)
        })
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
        let attempt = run_handoff(&repository, &graph, &mut runtime, 0, &mut |event| {
            events.push(event)
        })
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
    }

    #[test]
    fn always_check_runs_for_no_change_without_requesting_a_commit() {
        let repo = init_repo();
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        let graph = graph("true", CheckTrigger::Always);
        let mut runtime = runtime(repo.path(), &graph);
        let attempt = run_handoff(&repository, &graph, &mut runtime, 0, &mut |_| {}).unwrap();
        let HandoffAttempt::NoChange(summary) = attempt else {
            panic!("expected no-change handoff");
        };
        assert_eq!(summary.checks[0].status, "passed");
        assert!(summary.commit.is_none());
    }
}
