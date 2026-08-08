use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::agent_core::AgentProfile;

use super::{
    ArtifactEnvelope, CodeReviewArtifact, ImplementationArtifact, PlanArtifact, PlanReviewArtifact,
    WorkflowLimits, WorkflowStage,
};

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageContract {
    pub stage: WorkflowStage,
    pub profile: AgentProfile,
    pub capabilities: StageCapabilities,
    pub max_steps: usize,
    pub max_tokens_per_turn: i32,
    pub terminal_action: TerminalActionKind,
}

impl StageContract {
    pub fn strict(stage: WorkflowStage, limits: WorkflowLimits, max_tokens: i32) -> Result<Self> {
        let profile = match stage {
            WorkflowStage::Planning | WorkflowStage::PlanRevision => AgentProfile::Plan,
            WorkflowStage::PlanReview | WorkflowStage::CodeReview => AgentProfile::Review,
            WorkflowStage::Implementing | WorkflowStage::Repairing => AgentProfile::Build,
            _ => bail!("stage {stage:?} is deterministic and cannot run a model stage"),
        };
        let capabilities = StageCapabilities::for_stage(stage);
        let contract = Self {
            stage,
            profile,
            capabilities,
            max_steps: limits.stage_steps,
            max_tokens_per_turn: max_tokens,
            terminal_action: capabilities.terminal_action,
        };
        contract.validate(limits)?;
        Ok(contract)
    }

    pub fn validate(&self, limits: WorkflowLimits) -> Result<()> {
        if self.stage.is_terminal()
            || matches!(
                self.stage,
                WorkflowStage::Checking | WorkflowStage::Committing
            )
        {
            bail!("stage {:?} is not model-driven", self.stage);
        }
        let expected = StageCapabilities::for_stage(self.stage);
        if self.capabilities != expected || self.terminal_action != expected.terminal_action {
            bail!(
                "stage contract capabilities must exactly match the harness policy for {:?}",
                self.stage
            );
        }
        let max_stage_steps = limits.stage_steps.saturating_add(
            if matches!(
                self.stage,
                WorkflowStage::Implementing | WorkflowStage::Repairing
            ) {
                super::MAX_WORK_UNIT_PROGRESS_CREDITS
            } else {
                0
            },
        );
        if self.max_steps == 0 || self.max_steps > max_stage_steps {
            bail!("stage max_steps must be between 1 and {}", max_stage_steps);
        }
        if self.max_tokens_per_turn <= 0
            || self.max_tokens_per_turn as usize > limits.total_generated_tokens
        {
            bail!("stage max_tokens_per_turn must fit the remaining workflow token policy");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageSubmission {
    Plan {
        plan: ArtifactEnvelope<PlanArtifact>,
    },
    PlanReview {
        review: ArtifactEnvelope<PlanReviewArtifact>,
    },
    Implementation {
        implementation: ArtifactEnvelope<ImplementationArtifact>,
    },
    Replan {
        reason: String,
    },
    CodeReview {
        review: ArtifactEnvelope<CodeReviewArtifact>,
    },
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
            "read_file" | "glob" | "ripgrep" | "search" | "git_log" | "memory_search"
            | "memory_read" => self.repository_read,
            // Strict stages have durable plan, evidence, and checkpoint state. Invocation-local
            // model summaries are neither repository evidence nor a workflow authority source.
            "session_changes" => false,
            "inspect_change" => {
                self.repository_read && self.terminal_action == TerminalActionKind::SubmitCodeReview
            }
            "web_search" | "web_fetch" => self.public_research,
            "attachments" | "vision_describe" => self.repository_read,
            // Strict delivery already has the user's task as its stable title fallback. Spending
            // a dedicated model turn on cosmetic title rewriting adds no workflow evidence or
            // authority, so title changes remain a discussion-only capability.
            "session_title" => self.terminal_action == TerminalActionKind::DiscussFinal,
            "ask_user" => {
                matches!(self.terminal_action, TerminalActionKind::SubmitPlan)
            }
            // Strict delivery already has durable typed stage state. Exposing the legacy,
            // invocation-local todo protocol only adds another control loop for the model.
            "todo" => false,
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
            "answer" => self.terminal_action == TerminalActionKind::DiscussFinal,
            "propose_delivery" | "start_delivery" | "propose_goal" | "start_goal" => {
                self.terminal_action == TerminalActionKind::DiscussFinal
            }
            "goal_status" | "goal_pause" | "goal_request_amendment" | "goal_request_budget" => {
                self.terminal_action != TerminalActionKind::None
            }
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
        assert!(!capabilities.allows_tool("todo"));
        assert!(!capabilities.allows_tool("git_commit"));
        assert!(!capabilities.allows_tool("submit_code_review"));
    }

    #[test]
    fn read_only_stages_have_distinct_terminal_actions_and_no_shell() {
        let plan = StageCapabilities::for_stage(WorkflowStage::Planning);
        assert!(plan.allows_tool("read_file"));
        assert!(!plan.allows_tool("inspect_change"));
        assert!(plan.allows_tool("submit_plan"));
        assert!(!plan.allows_tool("run_command"));
        assert!(!plan.allows_tool("apply_patch"));
        assert!(!plan.allows_tool("session_title"));
        assert!(!plan.allows_tool("session_changes"));
        assert!(!plan.allows_tool("answer"));

        let review = StageCapabilities::for_stage(WorkflowStage::CodeReview);
        assert!(review.allows_tool("inspect_change"));
        assert!(review.allows_tool("submit_code_review"));
        assert!(!review.allows_tool("submit_plan"));
        assert!(!review.allows_tool("run_command"));
        assert!(!review.allows_tool("session_title"));
        assert!(!review.allows_tool("session_changes"));

        let discuss = StageCapabilities::discuss();
        assert!(discuss.allows_tool("session_title"));
        assert!(discuss.allows_tool("answer"));
        assert!(!discuss.allows_tool("apply_patch"));
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

    #[test]
    fn stage_contract_cannot_smuggle_extra_capabilities_or_budget() {
        let limits = WorkflowLimits::default();
        let mut contract = StageContract::strict(WorkflowStage::Planning, limits, 512).unwrap();
        contract.capabilities.run_command = true;
        assert!(contract.validate(limits).is_err());

        let mut contract = StageContract::strict(WorkflowStage::Implementing, limits, 512).unwrap();
        contract.max_steps = limits.stage_steps + crate::workflow::MAX_WORK_UNIT_PROGRESS_CREDITS;
        assert!(contract.validate(limits).is_ok());
        contract.max_steps += 1;
        assert!(contract.validate(limits).is_err());

        let mut contract = StageContract::strict(WorkflowStage::Planning, limits, 512).unwrap();
        contract.max_steps = limits.stage_steps + 1;
        assert!(contract.validate(limits).is_err());
        assert!(StageContract::strict(WorkflowStage::Checking, limits, 512).is_err());
    }
}
