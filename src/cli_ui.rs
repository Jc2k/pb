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
        AgentEvent::Reasoning {
            content,
            profile: _,
            ..
        } => print_header("reasoning", content),
        AgentEvent::ToolCall { tool, .. } => print_header("tool", tool),
        AgentEvent::ToolResult { result, .. } => print_block("tool result", result),
        AgentEvent::UserQuestion {
            question_id,
            question,
            ..
        } => print_block(&format!("question {question_id}"), question),
        AgentEvent::UserAnswer {
            question_id,
            answer,
            ..
        } => print_block(&format!("answer {question_id}"), answer),
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
        AgentEvent::SessionSummary {
            branch, commits, ..
        } => {
            print_header("session", &format!("branch: {branch}"));
            if !commits.trim().is_empty() {
                print_block("session commits", commits);
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
