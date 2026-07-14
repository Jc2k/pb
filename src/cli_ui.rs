use crate::events::AgentEvent;

pub fn render_event(event: &AgentEvent) {
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
        AgentEvent::ConversationTurnStarted { intent, task, .. } => {
            print_header("turn", &format!("{intent:?}: {task}"));
        }
        AgentEvent::DeliveryProposed { task_summary, .. } => {
            print_header("delivery proposed", task_summary);
        }
        AgentEvent::WorkflowStarted { workflow_id, .. } => {
            print_header("workflow", &format!("{workflow_id} started"));
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
        AgentEvent::WorkflowBlocked {
            outcome, reason, ..
        } => print_header("workflow blocked", &format!("{outcome:?}: {reason}")),
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
            content,
            profile: _,
            ..
        } => print_header("reasoning", content),
        AgentEvent::ToolCall { tool, .. } => print_header("tool", tool),
        AgentEvent::ToolResult {
            result,
            duration_ms,
            energy_kwh,
            ..
        } => {
            let label = match (duration_ms, energy_kwh) {
                (Some(ms), Some(kwh)) => format!("tool result ({ms} ms, {kwh:.6e} kWh)"),
                (Some(ms), None) => format!("tool result ({ms} ms)"),
                (None, Some(kwh)) => format!("tool result ({kwh:.6e} kWh)"),
                (None, None) => "tool result".to_string(),
            };
            print_block(&label, result);
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
        AgentEvent::TeamMessage { message, .. } => {
            print_block("team", message);
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
        AgentEvent::Correction {
            summary, message, ..
        } => {
            let body = if summary.trim().is_empty() {
                message
            } else {
                summary
            };
            print_header("correction", body);
        }
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
            duration_ms,
            prompt_tokens,
            generated_tokens,
            energy_kwh,
            ..
        } => {
            let energy = energy_kwh
                .map(|kwh| format!(", {kwh:.6e} kWh"))
                .unwrap_or_default();
            print_header(
                "llm",
                &format!(
                    "step {step}: {duration_ms} ms, {prompt_tokens} prompt tokens, {generated_tokens} generated tokens{energy}"
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
            llm_energy_kwh,
            tool_energy_kwh,
            ..
        } => {
            let energy = match (llm_energy_kwh, tool_energy_kwh) {
                (Some(llm), Some(tool)) => {
                    format!("; energy: llm {llm:.6e} kWh, tools {tool:.6e} kWh")
                }
                (Some(llm), None) => format!("; energy: llm {llm:.6e} kWh"),
                (None, Some(tool)) => format!("; energy: tools {tool:.6e} kWh"),
                (None, None) => String::new(),
            };
            print_header(
                "metrics",
                &format!(
                    "llm: {llm_invocations} calls, {llm_runtime_ms} ms, {prompt_tokens} prompt tokens, {generated_tokens} generated tokens; tools: {tool_calls} calls, {tool_runtime_ms} ms{energy}"
                ),
            );
        }
        AgentEvent::SessionTitle { title, .. } => print_header("session title", title),
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
            if !commits.trim().is_empty() {
                print_block("session commits", commits);
            }
            if !diff_stat.trim().is_empty() {
                print_block("diff stat from main", diff_stat);
            }
            if !diff.trim().is_empty() {
                print_block("diff from main", diff);
            }
        }
        AgentEvent::Error { message, .. } => eprintln!("error: {message}"),
    }
}

fn print_header(label: &str, value: &str) {
    println!("\x1b[1;36m[{label}]\x1b[0m {value}");
}

fn print_block(label: &str, content: &str) {
    println!("\x1b[1;35m[{label}]\x1b[0m\n{content}");
}
