use serde::{Deserialize, Serialize};

use super::WorkflowStage;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalActionKind {
    DiscussFinal,
    SubmitPlan,
    SubmitPlanReview,
    SubmitImplementation,
    SubmitCodeReview,
    None,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageCapabilities {
    pub repository_read: bool,
    pub public_research: bool,
    pub repository_mutation: bool,
    pub run_command: bool,
    pub run_task: bool,
    pub run_check: bool,
    pub advisory: bool,
    pub terminal_action: TerminalActionKind,
}

impl StageCapabilities {
    pub const fn discuss() -> Self {
        Self {
            repository_read: true,
            public_research: true,
            repository_mutation: false,
            run_command: false,
            run_task: false,
            run_check: false,
            advisory: true,
            terminal_action: TerminalActionKind::DiscussFinal,
        }
    }

    pub const fn for_stage(stage: WorkflowStage) -> Self {
        match stage {
            WorkflowStage::Planning | WorkflowStage::PlanRevision => Self {
                repository_read: true,
                public_research: true,
                repository_mutation: false,
                run_command: false,
                run_task: false,
                run_check: false,
                advisory: true,
                terminal_action: TerminalActionKind::SubmitPlan,
            },
            WorkflowStage::PlanReview => Self {
                repository_read: true,
                public_research: true,
                repository_mutation: false,
                run_command: false,
                run_task: false,
                run_check: false,
                advisory: true,
                terminal_action: TerminalActionKind::SubmitPlanReview,
            },
            WorkflowStage::Implementing | WorkflowStage::Repairing => Self {
                repository_read: true,
                public_research: true,
                repository_mutation: true,
                run_command: true,
                run_task: true,
                run_check: true,
                advisory: true,
                terminal_action: TerminalActionKind::SubmitImplementation,
            },
            WorkflowStage::CodeReview => Self {
                repository_read: true,
                public_research: true,
                repository_mutation: false,
                run_command: false,
                run_task: false,
                run_check: false,
                advisory: true,
                terminal_action: TerminalActionKind::SubmitCodeReview,
            },
            WorkflowStage::Checking
            | WorkflowStage::Committing
            | WorkflowStage::Ready
            | WorkflowStage::Failed
            | WorkflowStage::Blocked
            | WorkflowStage::Cancelled => Self {
                repository_read: false,
                public_research: false,
                repository_mutation: false,
                run_command: false,
                run_task: false,
                run_check: false,
                advisory: false,
                terminal_action: TerminalActionKind::None,
            },
        }
    }

    pub fn allows_tool(self, tool: &str) -> bool {
        match tool {
            "read_file" | "glob" | "ripgrep" | "search" | "git_log" | "session_changes"
            | "memory_search" | "memory_read" => self.repository_read,
            "web_search" | "web_fetch" => self.public_research,
            "write_file" | "replace_file" | "edit_file" | "apply_patch" | "mv" | "rm" => {
                self.repository_mutation
            }
            "run_command" => self.run_command,
            "run_task" => self.run_task,
            "run_check" => self.run_check,
            "sub_agent" => self.advisory,
            "submit_plan" => self.terminal_action == TerminalActionKind::SubmitPlan,
            "submit_plan_review" => self.terminal_action == TerminalActionKind::SubmitPlanReview,
            "submit_implementation" | "request_replan" => {
                self.terminal_action == TerminalActionKind::SubmitImplementation
            }
            "submit_code_review" => self.terminal_action == TerminalActionKind::SubmitCodeReview,
            "git_commit" | "git_revert" => false,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdvisoryRole {
    Explore,
    Research,
    FocusedReview,
    Monitor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implementation_keeps_shell_escape_hatch_but_not_commit_authority() {
        let capabilities = StageCapabilities::for_stage(WorkflowStage::Implementing);
        assert!(capabilities.allows_tool("run_command"));
        assert!(capabilities.allows_tool("apply_patch"));
        assert!(capabilities.allows_tool("submit_implementation"));
        assert!(!capabilities.allows_tool("git_commit"));
        assert!(!capabilities.allows_tool("submit_code_review"));
    }

    #[test]
    fn read_only_stages_have_distinct_terminal_actions_and_no_shell() {
        let plan = StageCapabilities::for_stage(WorkflowStage::Planning);
        assert!(plan.allows_tool("read_file"));
        assert!(plan.allows_tool("submit_plan"));
        assert!(!plan.allows_tool("run_command"));
        assert!(!plan.allows_tool("apply_patch"));

        let review = StageCapabilities::for_stage(WorkflowStage::CodeReview);
        assert!(review.allows_tool("submit_code_review"));
        assert!(!review.allows_tool("submit_plan"));
        assert!(!review.allows_tool("run_command"));
    }

    #[test]
    fn deterministic_stages_expose_no_model_tools() {
        for stage in [
            WorkflowStage::Checking,
            WorkflowStage::Committing,
            WorkflowStage::Ready,
            WorkflowStage::Failed,
        ] {
            let capabilities = StageCapabilities::for_stage(stage);
            assert!(!capabilities.repository_read);
            assert_eq!(capabilities.terminal_action, TerminalActionKind::None);
        }
    }
}
