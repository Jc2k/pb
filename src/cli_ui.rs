use crate::events::AgentEvent;

pub fn render_event(event: &AgentEvent) {
    match event {
        AgentEvent::Started {
            task,
            model,
            workspace,
            branch,
            ..
        } => {
            print_header("local agent", &format!("task: {task}"));
            print_header("model", model);
            print_header("workspace", workspace);
            print_header("branch", branch);
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
            summary,
            diff_stat,
            diff,
            ..
        } => {
            print_header("session", &format!("branch: {branch}"));
            if !summary.trim().is_empty() {
                print_block("session summary", summary);
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
