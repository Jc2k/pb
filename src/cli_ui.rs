use anyhow::{Context, Result};
use std::io::Write;

use crate::agent_core::{AgentRequest, run_agent};
use crate::events::AgentEvent;

pub fn render_event(event: &AgentEvent) {
    match event {
        AgentEvent::Started {
            task,
            model,
            workspace,
            branch,
        } => {
            print_header("local agent", &format!("task: {task}"));
            print_header("model", model);
            print_header("workspace", workspace);
            print_header("branch", branch);
        }
        AgentEvent::StepStarted { step, max_steps } => {
            print_header("step", &format!("{step}/{max_steps}"));
        }
        AgentEvent::Reasoning { content } => print_header("reasoning", content),
        AgentEvent::ToolCall { tool, .. } => print_header("tool", tool),
        AgentEvent::ToolResult { result, .. } => print_block("tool result", result),
        AgentEvent::Diff { diff, .. } => print_block("diff", diff),
        AgentEvent::Final { content } => print_header("final", content),
        AgentEvent::SessionSummary { branch, commits } => {
            print_header("session", &format!("branch: {branch}"));
            if !commits.trim().is_empty() {
                print_block("session commits", commits);
            }
        }
        AgentEvent::Error { message } => eprintln!("error: {message}"),
    }
}

pub async fn run_agent_cli(mut request: AgentRequest, models_root: &std::path::Path) -> Result<()> {
    loop {
        let result = run_agent(request.clone(), models_root, |event| render_event(&event))?;

        if !result.reached_final {
            break;
        }

        print!("\x1b[1;32m[session]\x1b[0m Follow-up prompt (or Enter to finish): ");
        std::io::stdout().flush().ok();

        let mut next_task = String::new();
        std::io::stdin()
            .read_line(&mut next_task)
            .context("failed to read follow-up prompt")?;
        let next_task = next_task.trim().to_string();
        if next_task.is_empty() {
            break;
        }

        request.task = next_task;
        request.branch = Some(result.branch);
        request.workdir = Some(result.workspace_root);
    }

    Ok(())
}

fn print_header(label: &str, value: &str) {
    println!("\x1b[1;36m[{label}]\x1b[0m {value}");
}

fn print_block(label: &str, content: &str) {
    println!("\x1b[1;35m[{label}]\x1b[0m\n{content}");
}
