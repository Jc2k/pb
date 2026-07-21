use serde::Serialize;

use crate::workflow::WorkflowStage;

pub(crate) const MAX_CLOSURE_CHECKPOINT_CHARS: usize = 4_000;
const MAX_MISSING_PRECONDITIONS: usize = 6;
const MAX_PRECONDITION_CHARS: usize = 140;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolExposureState {
    Authorized,
    Closing {
        stage: WorkflowStage,
        terminal_tool: &'static str,
        terminal_ready: bool,
    },
    TerminalOnly {
        terminal_tool: &'static str,
    },
}

impl ToolExposureState {
    pub(crate) fn for_turn(
        stage: Option<WorkflowStage>,
        ordinary_steps_remaining: usize,
        terminal_tool: Option<&'static str>,
        terminal_ready: bool,
    ) -> Self {
        let (Some(stage), Some(terminal_tool)) = (stage, terminal_tool) else {
            return Self::Authorized;
        };
        if ordinary_steps_remaining > 2 {
            return Self::Authorized;
        }
        if matches!(
            stage,
            WorkflowStage::Implementing | WorkflowStage::Repairing
        ) {
            return if ordinary_steps_remaining == 1 && terminal_ready {
                Self::TerminalOnly { terminal_tool }
            } else {
                Self::Authorized
            };
        }
        if !matches!(
            stage,
            WorkflowStage::Planning
                | WorkflowStage::PlanRevision
                | WorkflowStage::PlanReview
                | WorkflowStage::CodeReview
        ) {
            return Self::Authorized;
        }
        if ordinary_steps_remaining == 1 && terminal_ready {
            Self::TerminalOnly { terminal_tool }
        } else {
            Self::Closing {
                stage,
                terminal_tool,
                terminal_ready,
            }
        }
    }

    pub(crate) fn allows(self, tool: &str) -> bool {
        match self {
            Self::Authorized => true,
            Self::TerminalOnly { terminal_tool } => tool == terminal_tool,
            Self::Closing {
                stage,
                terminal_tool,
                terminal_ready,
            } => {
                (terminal_ready && tool == terminal_tool)
                    || match stage {
                        WorkflowStage::Planning | WorkflowStage::PlanRevision => {
                            is_planning_evidence_tool(tool)
                        }
                        WorkflowStage::PlanReview => is_plan_review_evidence_tool(tool),
                        WorkflowStage::CodeReview => is_code_review_evidence_tool(tool),
                        _ => false,
                    }
            }
        }
    }

    pub(crate) const fn is_closing(self) -> bool {
        !matches!(self, Self::Authorized)
    }
}

fn is_planning_evidence_tool(tool: &str) -> bool {
    matches!(
        tool,
        "read_file" | "glob" | "ripgrep" | "search" | "git_log" | "session_changes"
    )
}

fn is_plan_review_evidence_tool(tool: &str) -> bool {
    matches!(
        tool,
        "read_file" | "glob" | "ripgrep" | "search" | "git_log" | "session_changes"
    )
}

fn is_code_review_evidence_tool(tool: &str) -> bool {
    matches!(
        tool,
        "inspect_change" | "read_file" | "glob" | "ripgrep" | "search"
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct ClosureCheckpoint {
    #[serde(rename = "type")]
    type_name: &'static str,
    stage: WorkflowStage,
    ordinary_steps_remaining: usize,
    current_content_fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_content_fingerprint: Option<String>,
    terminal_tool: String,
    terminal_signature: String,
    terminal_status: &'static str,
    missing_terminal_preconditions: Vec<String>,
    instruction: &'static str,
}

impl ClosureCheckpoint {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render(
        stage: WorkflowStage,
        ordinary_steps_remaining: usize,
        current_content_fingerprint: &str,
        expected_content_fingerprint: Option<&str>,
        terminal_tool: &str,
        terminal_signature: &str,
        missing_terminal_preconditions: &[String],
    ) -> String {
        let omitted = missing_terminal_preconditions
            .len()
            .saturating_sub(MAX_MISSING_PRECONDITIONS);
        let mut missing = missing_terminal_preconditions
            .iter()
            .take(MAX_MISSING_PRECONDITIONS)
            .map(|item| sanitize_bounded(item, MAX_PRECONDITION_CHARS))
            .collect::<Vec<_>>();
        if omitted > 0 {
            missing.push(format!(
                "{omitted} additional deterministic precondition(s) omitted from this bounded checkpoint"
            ));
        }
        let terminal_ready = missing_terminal_preconditions.is_empty();
        let mut checkpoint = Self {
            type_name: "workflow_closure_checkpoint",
            stage,
            ordinary_steps_remaining,
            current_content_fingerprint: sanitize_bounded(current_content_fingerprint, 96),
            expected_content_fingerprint: expected_content_fingerprint
                .map(|value| sanitize_bounded(value, 96)),
            terminal_tool: sanitize_bounded(terminal_tool, 80),
            terminal_signature: sanitize_bounded(terminal_signature, 600),
            terminal_status: if terminal_ready { "eligible" } else { "hidden" },
            missing_terminal_preconditions: missing,
            instruction: if terminal_ready {
                "Finish with the exact terminal tool; do not start a new evidence chain unless a listed fact changes."
            } else {
                "Resolve only the listed deterministic facts with an exposed evidence tool; proximity to the step limit does not bypass them."
            },
        };
        let mut rendered = serde_json::to_string(&checkpoint)
            .expect("serializing a bounded closure checkpoint cannot fail");
        if rendered.chars().count() > MAX_CLOSURE_CHECKPOINT_CHARS {
            checkpoint.missing_terminal_preconditions = checkpoint
                .missing_terminal_preconditions
                .iter()
                .take(2)
                .map(|item| sanitize_bounded(item, 80))
                .collect();
            checkpoint.terminal_signature = sanitize_bounded(terminal_signature, 300);
            checkpoint.instruction = if terminal_ready {
                "Call the exact terminal tool now."
            } else {
                "Resolve the listed fact; the terminal tool remains hidden."
            };
            rendered = serde_json::to_string(&checkpoint)
                .expect("serializing a reduced closure checkpoint cannot fail");
        }
        debug_assert!(rendered.chars().count() <= MAX_CLOSURE_CHECKPOINT_CHARS);
        rendered
    }
}

fn sanitize_bounded(value: &str, max_chars: usize) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .take(max_chars)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn se1_closing_states_only_keep_stage_evidence_and_eligible_terminal_tools() {
        let planning = ToolExposureState::for_turn(
            Some(WorkflowStage::Planning),
            2,
            Some("submit_plan"),
            true,
        );
        assert!(planning.allows("read_file"));
        assert!(planning.allows("submit_plan"));
        assert!(!planning.allows("write_file"));
        assert!(!planning.allows("sub_agent"));

        let review = ToolExposureState::for_turn(
            Some(WorkflowStage::CodeReview),
            2,
            Some("submit_code_review"),
            false,
        );
        assert!(review.allows("inspect_change"));
        assert!(review.allows("ripgrep"));
        assert!(!review.allows("submit_code_review"));
        assert!(!review.allows("web_fetch"));
    }

    #[test]
    fn se2_last_ready_turn_is_terminal_only_including_implementation() {
        let closure = ToolExposureState::for_turn(
            Some(WorkflowStage::PlanReview),
            1,
            Some("submit_plan_review"),
            true,
        );
        assert!(closure.allows("submit_plan_review"));
        assert!(!closure.allows("read_file"));

        let implementation = ToolExposureState::for_turn(
            Some(WorkflowStage::Implementing),
            1,
            Some("submit_implementation"),
            true,
        );
        assert!(implementation.allows("submit_implementation"));
        assert!(!implementation.allows("run_check"));
        assert!(!implementation.allows("write_file"));

        let unfinished_implementation = ToolExposureState::for_turn(
            Some(WorkflowStage::Implementing),
            1,
            Some("submit_implementation"),
            false,
        );
        assert_eq!(unfinished_implementation, ToolExposureState::Authorized);
    }

    #[test]
    fn closure_checkpoint_is_structured_bounded_and_truthful_about_hidden_terminal() {
        let missing = (0..100)
            .map(|index| format!("missing path {index}: {}", "x".repeat(1_000)))
            .collect::<Vec<_>>();
        let rendered = ClosureCheckpoint::render(
            WorkflowStage::CodeReview,
            2,
            "current",
            Some("expected"),
            "submit_code_review",
            "submit_code_review(review: object)",
            &missing,
        );
        assert!(rendered.chars().count() <= MAX_CLOSURE_CHECKPOINT_CHARS);
        assert!(rendered.contains("\"terminal_status\":\"hidden\""));
        assert!(rendered.contains("additional deterministic precondition"));
    }
}
