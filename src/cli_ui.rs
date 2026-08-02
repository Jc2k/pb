use crate::events::{AgentEvent, EventChatter, EventEnvelope, TeamActor};

pub fn render_event(envelope: &EventEnvelope) {
    let event = &envelope.event;
    match event {
        AgentEvent::Started {
            task,
            model,
            workspace,
            focus_root,
            branch,
            ..
        } => {
            print_header("local agent", &format!("task: {task}"));
            print_header("model", model);
            print_header("workspace", workspace);
            if let Some(focus_root) = focus_root
                && focus_root != workspace
            {
                print_header("focus", focus_root);
            }
            print_header("branch", branch);
        }
        AgentEvent::HarnessExperimentConfigured {
            observation_rendering,
            ..
        } => print_header(
            "harness experiment",
            &format!("observation_rendering={}", observation_rendering.as_str()),
        ),
        AgentEvent::ConversationTurnStarted { intent, task, .. } => {
            print_header("turn", &format!("{intent:?}: {task}"));
        }
        AgentEvent::DeliveryProposed { task_summary, .. } => {
            print_header("delivery proposed", task_summary);
        }
        AgentEvent::GoalProposed {
            objective,
            criteria,
            ..
        } => print_header(
            "goal proposed",
            &format!("{objective} ({} criteria)", criteria.len()),
        ),
        AgentEvent::TaskPlanAccepted { task_count, .. } => {
            print_header("Tasks", &format!("{task_count} accepted"));
        }
        AgentEvent::TaskPlanRejected {
            outcome, attempts, ..
        } => print_header(
            "Task planning",
            &format!("stopped after {attempts} attempt(s): {outcome:?}"),
        ),
        AgentEvent::TasksChanged {
            stage,
            active_task_id,
            ..
        } => print_header(
            "Tasks",
            &active_task_id.as_ref().map_or_else(
                || format!("{stage:?}"),
                |task_id| format!("{stage:?}: {task_id}"),
            ),
        ),
        AgentEvent::GoalStarted { objective, .. } => print_header("goal", objective),
        AgentEvent::GoalPlanAwaitingApproval { milestones, .. } => {
            print_header(
                "goal plan",
                &format!("{milestones} milestones awaiting approval"),
            );
        }
        AgentEvent::GoalPlanApproved { .. } => print_header("goal plan", "approved"),
        AgentEvent::GoalMilestoneStarted { title, .. } => {
            print_header("goal milestone", &format!("started: {title}"));
        }
        AgentEvent::GoalMilestoneCompleted { milestone_id, .. } => {
            print_header("goal milestone", &format!("completed: {milestone_id}"));
        }
        AgentEvent::GoalPauseRequested { .. } => print_header("goal", "pause requested"),
        AgentEvent::GoalPaused { .. } => print_header("goal", "paused"),
        AgentEvent::GoalResumed { .. } => print_header("goal", "resumed"),
        AgentEvent::GoalAmendmentRequested { amendment_id, .. } => {
            print_header("goal amendment", &format!("review {amendment_id}"));
        }
        AgentEvent::GoalChangeRequested { kind, summary, .. } => {
            print_header(&format!("goal {kind} request"), summary);
        }
        AgentEvent::GoalAmendmentResolved {
            amendment_id,
            accepted,
            ..
        } => print_header(
            "goal amendment",
            &format!(
                "{amendment_id}: {}",
                if *accepted { "accepted" } else { "discarded" }
            ),
        ),
        AgentEvent::GoalReadyForReview { .. } => print_header("goal", "ready for review"),
        AgentEvent::GoalCompleted {
            completion_basis, ..
        } => print_header("goal", &format!("completed: {completion_basis:?}")),
        AgentEvent::GoalFailed {
            outcome, reason, ..
        } => print_header("goal", &format!("failed: {outcome:?}: {reason}")),
        AgentEvent::GoalCancelled { .. } => print_header("goal", "cancelled"),
        AgentEvent::WorkflowStarted { workflow_id, .. } => {
            print_header("workflow", &format!("{workflow_id} started"));
        }
        AgentEvent::WorkflowResumed { stage, .. } => {
            print_header("workflow", &format!("resumed at {stage:?}"));
        }
        AgentEvent::WorkflowStageStarted { stage, .. } => {
            print_header("workflow stage", &format!("{stage:?}"));
        }
        AgentEvent::WorkflowArtifactAccepted {
            artifact_kind,
            artifact_id,
            ..
        } => print_header(
            "workflow artifact",
            &format!("{artifact_kind}: {artifact_id}"),
        ),
        AgentEvent::WorkflowChallengeRaised {
            severity, summary, ..
        } => print_header("workflow challenge", &format!("{severity:?}: {summary}")),
        AgentEvent::WorkflowEvidenceInvalidated { reason, .. } => {
            print_header("workflow evidence", &format!("invalidated: {reason}"));
        }
        AgentEvent::WorkflowStageCompleted { stage, .. } => {
            print_header("workflow stage", &format!("{stage:?} completed"));
        }
        AgentEvent::WorkflowBlocked { .. } => render_chatter(&envelope.chatter),
        AgentEvent::WorkflowCompleted { outcome, .. } => {
            print_header("workflow", &format!("completed: {outcome:?}"));
        }
        AgentEvent::StepStarted {
            step, max_steps, ..
        } => {
            print_header("step", &format!("{step}/{max_steps}"));
        }
        AgentEvent::ModelLoading { model, .. } => {
            print_header("model", &format!("loading {model}"));
        }
        AgentEvent::Reasoning {
            content, profile, ..
        } => print_header(profile.teammate_name(), content),
        AgentEvent::ToolCall { tool, actor, .. } => print_header(&action_label(*actor), tool),
        AgentEvent::ControllerObservation { receipt, actor, .. } => print_header(
            &action_label(*actor),
            &format_controller_observation(
                receipt.operation.as_str(),
                &receipt.path,
                receipt.coverage.as_str(),
                receipt.observed_bytes,
            ),
        ),
        AgentEvent::ControllerClosure { reason, actor, .. } => {
            print_header(&action_label(*actor), reason)
        }
        AgentEvent::ControllerMutation { receipt, actor, .. } => print_header(
            &action_label(*actor),
            &format_controller_delete(&receipt.path),
        ),
        AgentEvent::ToolBatch {
            call_count,
            useful_count,
            rejected_as_dependent,
            ..
        } => print_header(
            "tool batch",
            &format!(
                "{call_count} call(s), {useful_count} useful{}",
                if *rejected_as_dependent {
                    ", rejected dependency"
                } else {
                    ""
                }
            ),
        ),
        AgentEvent::ToolResult {
            result,
            actor,
            duration_ms,
            energy_joules,
            average_power_watts,
            energy_shared_calls,
            ..
        } => {
            let mut details = vec![format!("{duration_ms} ms")];
            if let Some(joules) = energy_joules {
                details.push(format_energy(*joules));
            }
            if let Some(watts) = average_power_watts {
                details.push(format!("{watts:.1} W average"));
            }
            if let Some(calls) = energy_shared_calls.filter(|calls| *calls > 1) {
                details.push(format!("shared across {calls} parallel calls"));
            }
            let actor_label = action_result_label(*actor);
            let label = if details.is_empty() {
                actor_label
            } else {
                format!("{actor_label} ({})", details.join(", "))
            };
            print_block(&label, result);
        }
        AgentEvent::ContextLimit {
            measured_prompt_tokens,
            usable_prompt_capacity,
            largest_sections,
            ..
        } => {
            let sections = largest_sections
                .iter()
                .map(|section| format!("{}={} chars", section.label, section.chars))
                .collect::<Vec<_>>()
                .join(", ");
            print_header(
                "context limit",
                &format!(
                    "measured {measured_prompt_tokens} prompt tokens; usable {usable_prompt_capacity}; largest sections: {sections}"
                ),
            );
        }
        AgentEvent::ExecutorStarted {
            executor_id,
            kind,
            success,
            detail,
            ..
        } => {
            let state = if *success { "ready" } else { "unavailable" };
            print_header(
                "executor",
                &format!("{executor_id} ({kind}, {state}) {detail}"),
            );
        }
        AgentEvent::TeamMessage { actor, message, .. } => {
            print_block(&actor_message_label(*actor), message);
        }
        AgentEvent::HandoffSummary { summary, .. } => {
            print_header("handoff", &format!("{:?}", summary.outcome));
        }
        AgentEvent::CommitResult {
            success,
            oid,
            subject,
            detail,
            ..
        } => {
            let state = if *success { "ready" } else { "blocked" };
            print_header(
                "commit",
                &format!(
                    "{state} {} {} {detail}",
                    oid.as_deref().unwrap_or_default(),
                    subject.as_deref().unwrap_or_default()
                ),
            );
        }
        AgentEvent::CheckResult {
            check_id,
            exit_status,
            success,
            timed_out,
            output,
            duration_ms,
            reused,
            skip_reason,
            ..
        } => {
            let disposition = if skip_reason.is_some() {
                "skipped"
            } else if *reused {
                "reused"
            } else if *timed_out {
                "timed out"
            } else if *success {
                "passed"
            } else {
                "failed"
            };
            print_block(
                &format!(
                    "check {check_id} ({disposition}, status {exit_status}, {duration_ms} ms)"
                ),
                output,
            );
        }
        AgentEvent::UserQuestion {
            question_id,
            question,
            choices,
            ..
        } => {
            let body = if choices.is_empty() {
                question.clone()
            } else {
                format!(
                    "{}\n{}",
                    question,
                    choices
                        .iter()
                        .map(|choice| format!("- {choice}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                )
            };
            print_block(&format!("question {question_id}"), &body);
        }
        AgentEvent::UserAnswer {
            question_id,
            answer,
            ..
        } => print_block(&format!("answer {question_id}"), answer),
        AgentEvent::UserMessage {
            message_id,
            message,
            ..
        } => print_block(&format!("user message {message_id}"), message),
        AgentEvent::UserMessageApplied { message_id, .. } => {
            print_header("user message", &format!("{message_id} applied"));
        }
        AgentEvent::Correction { .. } => render_chatter(&envelope.chatter),
        AgentEvent::SubAgentStarted {
            profile,
            task,
            nesting_depth: _,
            ..
        } => {
            print_header("sub-agent", &format!("{profile}: {task}"));
        }
        AgentEvent::SubAgentFinished {
            profile, result, ..
        } => {
            print_block(&format!("sub-agent {profile}"), result);
        }
        AgentEvent::Diff { diff, .. } => print_block("diff", diff),
        AgentEvent::Final {
            content,
            profile: _,
            ..
        } => print_header("final", content),
        AgentEvent::FinalGrace { status, detail, .. } => {
            print_header("final grace", &format!("{status:?}: {detail}"));
        }
        AgentEvent::LlmInvocation {
            step,
            purpose,
            duration_ms,
            prompt_tokens,
            generated_tokens,
            prompt_cache,
            native,
            energy_joules,
            average_power_watts,
            ..
        } => {
            let energy = energy_joules
                .map(|joules| {
                    let power = average_power_watts
                        .map(|watts| format!(" at {watts:.1} W"))
                        .unwrap_or_default();
                    format!(", {}{power}", format_energy(joules))
                })
                .unwrap_or_default();
            let cache = prompt_cache
                .as_ref()
                .map(|cache| {
                    let miss = cache
                        .miss_reason
                        .map(|reason| format!(", miss={}", reason.as_str()))
                        .unwrap_or_default();
                    let detail = cache
                        .lookup_detail
                        .map(|detail| format!(", detail={detail:?}"))
                        .unwrap_or_default();
                    let root = cache
                        .root
                        .as_ref()
                        .map(|root| {
                            format!(
                                ", root={}/{} {:?}",
                                root.reused_tokens, root.tokens, root.authority_class
                            )
                        })
                        .unwrap_or_default();
                    format!(
                        ", cache={} reused/{} fresh via {}{miss}{detail}{root}",
                        cache.cached_tokens, cache.prefilled_tokens, cache.source
                    )
                })
                .unwrap_or_default();
            let refill = native
                .as_ref()
                .and_then(|native| native.refill.as_ref())
                .map(|refill| {
                    format!(
                        ", refill={} lookup/{} disk/{} validate-allocate/{} hydrate/{} suffix/{} snapshot/{} queue ms",
                        refill.cache_lookup_wall_ms,
                        refill.disk_read_decode_wall_ms,
                        refill.cpu_state_validation_allocation_wall_ms,
                        refill.state_hydration_wall_ms,
                        refill.fresh_suffix_prefill_wall_ms,
                        refill.snapshot_capture_wall_ms,
                        refill.persistence_queue_wall_ms,
                    )
                })
                .unwrap_or_default();
            let prefill_command = native
                .as_ref()
                .map(|native| {
                    let reason = if native.prefill_command_reason.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", native.prefill_command_reason)
                    };
                    format!(", prefill={}{}", native.prefill_command_kind, reason)
                })
                .unwrap_or_default();
            print_header(
                "llm",
                &format!(
                    "{} step {step}: {duration_ms} ms, {prompt_tokens} prompt tokens, {generated_tokens} generated tokens{cache}{prefill_command}{refill}{energy}",
                    purpose.as_str()
                ),
            );
        }
        AgentEvent::SessionMetrics {
            llm_invocations,
            llm_runtime_ms,
            prompt_tokens,
            generated_tokens,
            tool_calls,
            tool_runtime_ms,
            cache_persistence_queued_checkpoints,
            cache_persistence_completed_checkpoints,
            cache_persistence_wall_ms,
            cache_persistence_failures,
            llm_energy_joules,
            tool_energy_joules,
            wall_runtime_ms,
            total_energy_joules,
            gross_energy_joules,
            adjusted_energy_joules,
            energy_coverage,
            energy_source,
            display_energy_excluded,
            idle_baseline_applied,
            ..
        } => {
            let energy = total_energy_joules.map(|joules| {
                let coverage = energy_coverage
                    .map(|coverage| format!(", {:.0}% coverage", coverage * 100.0))
                    .unwrap_or_default();
                let source = energy_source
                    .as_deref()
                    .map(|source| format!(", source {source}"))
                    .unwrap_or_default();
                format!(
                    "; estimated task energy: {} (gross {}, display-adjusted {}{coverage}{source}; display excluded: {display_energy_excluded}, idle baseline: {idle_baseline_applied}); diagnostic llm {}, tools {}",
                    format_energy(joules),
                    gross_energy_joules.map_or_else(|| "n/a".into(), |value| format_energy(value)),
                    adjusted_energy_joules.map_or_else(|| "n/a".into(), |value| format_energy(value)),
                    llm_energy_joules.map_or_else(|| "n/a".into(), |value| format_energy(value)),
                    tool_energy_joules.map_or_else(|| "n/a".into(), |value| format_energy(value)),
                )
            }).unwrap_or_default();
            let persistence = if *cache_persistence_queued_checkpoints > 0
                || *cache_persistence_failures > 0
            {
                format!(
                    "; cache persistence: {cache_persistence_completed_checkpoints}/{cache_persistence_queued_checkpoints} checkpoints in {cache_persistence_wall_ms} ms, {cache_persistence_failures} failures"
                )
            } else {
                String::new()
            };
            print_header(
                "metrics",
                &format!(
                    "wall: {wall_runtime_ms} ms; llm: {llm_invocations} calls, {llm_runtime_ms} ms, {prompt_tokens} prompt tokens, {generated_tokens} generated tokens; tools: {tool_calls} calls, {tool_runtime_ms} ms{persistence}{energy}"
                ),
            );
        }
        AgentEvent::SessionTitle { title, .. } => print_header("session title", title),
        AgentEvent::SessionStateChanged { .. } => {}
        AgentEvent::SessionSummary {
            branch,
            commits,
            reached_final,
            contract_status,
            verified_completed,
            termination_reason,
            summary,
            power_summary,
            diff_stat,
            diff,
            ..
        } => {
            let termination = termination_reason
                .map(|reason| reason.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            print_header(
                "session",
                &format!(
                    "branch: {branch}; reached_final: {reached_final}; contract_status: {contract_status}; verified_completed: {verified_completed}; termination_reason: {termination}"
                ),
            );
            if !summary.trim().is_empty() {
                print_block("session summary", summary);
            }
            if !power_summary.trim().is_empty() {
                print_block("session power", power_summary);
            }
            if !commits.is_empty() {
                let commits = commits
                    .iter()
                    .map(|commit| format!("{} {}", commit.oid, commit.subject))
                    .collect::<Vec<_>>()
                    .join("\n");
                print_block("session commits", &commits);
            }
            if !diff_stat.trim().is_empty() {
                print_block("diff stat from main", diff_stat);
            }
            if !diff.trim().is_empty() {
                print_block("diff from main", diff);
            }
        }
        AgentEvent::SemanticGate { receipt, .. } => print_header(
            "semantic gate",
            &format!(
                "stage={:?}; scope={:?}; providers={}; closure={:?}; definite_errors={}; unknown_reasons={}; wall={} ms",
                receipt.stage,
                receipt.scope,
                receipt.providers.len(),
                receipt.closure,
                receipt.definite_errors.len(),
                receipt.unknown_reasons.len(),
                receipt.wall_millis
            ),
        ),
        AgentEvent::Error {
            summary, detail, ..
        } => eprintln!("error: {summary}: {detail}"),
    }
}

fn render_chatter(messages: &[EventChatter]) {
    for message in messages {
        let body = message.headline.as_ref().map_or_else(
            || message.message.clone(),
            |headline| format!("{headline}\n{}", message.message),
        );
        print_block(&actor_message_label(message.actor), &body);
    }
}

fn print_header(label: &str, value: &str) {
    println!("\x1b[1;36m[{label}]\x1b[0m {value}");
}

fn print_block(label: &str, content: &str) {
    println!("\x1b[1;35m[{label}]\x1b[0m\n{content}");
}

fn format_energy(joules: f64) -> String {
    if joules < 1_000.0 {
        format!("{joules:.2} J")
    } else if joules < 3_600_000.0 {
        format!("{:.2} Wh", joules / 3_600.0)
    } else {
        format!("{:.3} kWh", joules / 3_600_000.0)
    }
}

fn format_controller_observation(
    operation: &str,
    path: &str,
    coverage: &str,
    observed_bytes: usize,
) -> String {
    format!("{operation} {path} · {coverage} coverage · {observed_bytes} bytes")
}

fn format_controller_delete(path: &str) -> String {
    format!("deleted {path} · tracked and Git-recoverable")
}

fn action_label(actor: TeamActor) -> String {
    match actor {
        actor @ TeamActor::Agent(_) => {
            format!("{} action · model", actor.display_name())
        }
        actor @ TeamActor::Automation(_) => {
            format!("{} action · harness", actor.display_name())
        }
    }
}

fn action_result_label(actor: TeamActor) -> String {
    match actor {
        actor @ TeamActor::Agent(_) => {
            format!("{} action result · model", actor.display_name())
        }
        actor @ TeamActor::Automation(_) => {
            format!("{} action result · harness", actor.display_name())
        }
    }
}

fn actor_message_label(actor: TeamActor) -> String {
    match actor {
        TeamActor::Agent(_) => actor.display_name().to_string(),
        TeamActor::Automation(_) => format!("{} · harness", actor.display_name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_core::AgentProfile;

    #[test]
    fn terminal_controller_actions_are_explicit_and_recovery_aware() {
        assert_eq!(
            format_controller_observation("read_file", "src/lib.rs", "full", 42),
            "read_file src/lib.rs · full coverage · 42 bytes"
        );
        assert_eq!(
            format_controller_delete("obsolete.rs"),
            "deleted obsolete.rs · tracked and Git-recoverable"
        );
    }

    #[test]
    fn terminal_action_labels_use_the_responsible_teammate() {
        assert_eq!(
            action_label(TeamActor::agent(AgentProfile::Build)),
            "Kate Libby action · model"
        );
        assert_eq!(
            action_label(TeamActor::workflow_steward()),
            "Trinity Walker action · harness"
        );
        assert_eq!(
            action_result_label(TeamActor::workflow_steward()),
            "Trinity Walker action result · harness"
        );
        assert_eq!(
            actor_message_label(TeamActor::workflow_steward()),
            "Trinity Walker · harness"
        );
    }
}
