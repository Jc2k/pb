use std::collections::VecDeque;

use sha2::{Digest, Sha256};

const MAX_PROGRESS_WINDOW: usize = 8;
const MAX_OUTCOME_SUMMARY_CHARS: usize = 180;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProgressState {
    pub(crate) workspace_fingerprint: String,
    pub(crate) evidence_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProgressObservation {
    pub(crate) tool_family: String,
    pub(crate) call_fingerprint: String,
    pub(crate) outcome_fingerprint: String,
    pub(crate) outcome_summary: String,
    pub(crate) success: bool,
    pub(crate) state: ProgressState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProgressDecision {
    Continue,
    Warn(String),
    Block(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FailedOutcome {
    signature: String,
    call_fingerprint: String,
    family: String,
    summary: String,
}

#[derive(Debug, Default)]
pub(crate) struct ProgressGuard {
    state: Option<ProgressState>,
    failures: VecDeque<FailedOutcome>,
    warned: bool,
}

impl ProgressGuard {
    pub(crate) fn preflight(
        &mut self,
        tool_family: &str,
        call_fingerprint: &str,
        state: ProgressState,
    ) -> Option<String> {
        if self.state.as_ref() != Some(&state) {
            self.failures.clear();
            self.warned = false;
            self.state = Some(state);
            return None;
        }
        if self.failures.len() < 2 {
            let repeated_known_empty_read = self.failures.back().is_some_and(|previous| {
                tool_family == "repository_read"
                    && known_empty_read_path(call_fingerprint).is_some_and(|path| {
                        prior_empty_read_path(&previous.call_fingerprint) == Some(path)
                    })
            });
            return repeated_known_empty_read.then(|| {
                format!(
                    "No-progress guard blocked a repeated read range already known to be empty on the unchanged file. Previous outcome: {}. The blocked call consumed no tool runtime and earned no new evidence. {}",
                    self.failures
                        .back()
                        .map(|failure| failure.summary.as_str())
                        .unwrap_or("the requested range contained no content"),
                    alternatives_for_family(tool_family),
                )
            });
        }
        let previous = &self.failures[self.failures.len() - 1];
        let before_previous = &self.failures[self.failures.len() - 2];
        let same_outcome_edit = previous.signature == before_previous.signature
            && previous.family == "workspace_edit"
            && tool_family == "workspace_edit";
        let alternating_call = before_previous.call_fingerprint == call_fingerprint
            && previous.call_fingerprint != call_fingerprint;
        if !same_outcome_edit && !alternating_call {
            return None;
        }
        let pattern = if alternating_call {
            "an A-B-A call cycle"
        } else {
            "two equivalent unchanged failures"
        };
        Some(format!(
            "No-progress guard blocked this call before execution after {pattern}. Previous outcome: {}. The blocked call consumed no tool runtime and made no repository change. {}",
            previous.summary,
            alternatives_for_family(tool_family),
        ))
    }

    pub(crate) fn record(&mut self, observation: ProgressObservation) -> ProgressDecision {
        if self.state.as_ref() != Some(&observation.state) {
            self.failures.clear();
            self.warned = false;
            self.state = Some(observation.state.clone());
        }
        if observation.success {
            self.failures.clear();
            self.warned = false;
            return ProgressDecision::Continue;
        }

        let failure = FailedOutcome {
            signature: format!(
                "{}:{}",
                observation.tool_family, observation.outcome_fingerprint
            ),
            call_fingerprint: observation.call_fingerprint,
            family: observation.tool_family,
            summary: observation.outcome_summary,
        };
        self.failures.push_back(failure.clone());
        while self.failures.len() > MAX_PROGRESS_WINDOW {
            self.failures.pop_front();
        }

        let same_outcome_count = self
            .failures
            .iter()
            .filter(|candidate| candidate.signature == failure.signature)
            .count();
        let alternating_cycle = self.failures.len() >= 3
            && self.failures[self.failures.len() - 3].signature == failure.signature
            && self.failures[self.failures.len() - 2].signature != failure.signature;
        if same_outcome_count >= 3 || alternating_cycle {
            let pattern = if alternating_cycle {
                "an A-B-A failure cycle"
            } else {
                "the same normalized failure"
            };
            return ProgressDecision::Block(format!(
                "No-progress guard stopped the run after {pattern} repeated without any workspace or evidence transition. Latest outcome: {}. No additional model turn or tool call will run. {}",
                failure.summary,
                alternatives_for_family(&failure.family),
            ));
        }

        if !self.warned && (same_outcome_count >= 2 || self.failures.len() >= 2) {
            self.warned = true;
            return ProgressDecision::Warn(format!(
                "No-progress guard: two failed tool outcomes occurred without any workspace or evidence transition. Latest outcome: {}. Do not repeat the same approach. {}",
                failure.summary,
                alternatives_for_family(&failure.family),
            ));
        }
        ProgressDecision::Continue
    }
}

fn known_empty_read_path(fingerprint: &str) -> Option<&str> {
    fingerprint.strip_prefix("read_file:known_empty:")
}

fn prior_empty_read_path(fingerprint: &str) -> Option<&str> {
    known_empty_read_path(fingerprint)
        .or_else(|| fingerprint.strip_prefix("read_file:cache_replay:"))
}

pub(crate) fn tool_family(tool: &str) -> String {
    match tool {
        "write_file" | "replace_file" | "edit_file" | "apply_patch" | "mv" | "rm" => {
            "workspace_edit"
        }
        "read_file" | "glob" | "ripgrep" | "search" | "git_log" | "session_changes" => {
            "repository_read"
        }
        "run_command" | "run_task" | "run_check" => "execution",
        "submit_plan"
        | "submit_plan_review"
        | "submit_implementation"
        | "request_replan"
        | "submit_code_review"
        | "start_delivery" => "workflow_transition",
        _ => tool,
    }
    .to_string()
}

pub(crate) fn outcome_identity(tool: &str, success: bool, result: &str) -> (String, String) {
    let summary = normalized_outcome_summary(tool, success, result);
    let fingerprint = format!("{:x}", Sha256::digest(summary.as_bytes()));
    (fingerprint, summary)
}

fn normalized_outcome_summary(tool: &str, success: bool, result: &str) -> String {
    if success {
        return format!(
            "success:{}",
            format!("{:x}", Sha256::digest(result.as_bytes()))
        );
    }
    let lower = result.to_ascii_lowercase();
    let reason = if lower.contains("patch mismatch diagnostic")
        || lower.contains("patch does not apply")
        || lower.contains("patch failed")
    {
        "stale patch context"
    } else if lower.contains("old_text not found") {
        "stale edit context"
    } else if lower.contains("must read") || lower.contains("without reading") {
        "required read evidence is missing"
    } else if lower.contains("not available") {
        "tool or executor is unavailable"
    } else if lower.contains("not found") {
        "requested target was not found"
    } else if lower.contains("timed out") || lower.contains("timeout") {
        "tool execution timed out"
    } else if lower.contains("denied by policy") || lower.contains("not approved") {
        "tool authorization was denied"
    } else {
        result
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("tool failed")
    };
    format!(
        "failure:{tool}:{}",
        truncate_and_normalize_numbers(reason, MAX_OUTCOME_SUMMARY_CHARS)
    )
}

fn truncate_and_normalize_numbers(value: &str, max_chars: usize) -> String {
    let mut normalized = String::new();
    let mut in_digits = false;
    for character in value.trim().chars() {
        if character.is_ascii_digit() {
            if !in_digits {
                normalized.push_str("<n>");
                in_digits = true;
            }
        } else {
            in_digits = false;
            normalized.push(character);
        }
        if normalized.chars().count() >= max_chars {
            break;
        }
    }
    normalized
}

fn alternatives_for_family(family: &str) -> &'static str {
    match family {
        "workspace_edit" => {
            "Re-read the exact target context, use a different bounded edit method, or request a replan when the accepted approach is no longer valid."
        }
        "repository_read" => {
            "Change the path or query, inspect a related manifest or entry point, or proceed using the evidence already collected."
        }
        "execution" => {
            "Inspect the first actionable diagnostic, change the implementation or check input, or report the external blocker instead of rerunning unchanged work."
        }
        "workflow_transition" => {
            "Correct the rejected artifact from the harness feedback; do not resubmit an unchanged transition."
        }
        _ => "Choose a different authorized action or report the blocker truthfully.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed(family: &str, outcome: &str, state: &str) -> ProgressObservation {
        ProgressObservation {
            tool_family: family.to_string(),
            call_fingerprint: outcome.to_string(),
            outcome_fingerprint: outcome.to_string(),
            outcome_summary: format!("failure {outcome}"),
            success: false,
            state: ProgressState {
                workspace_fingerprint: state.to_string(),
                evidence_fingerprint: state.to_string(),
            },
        }
    }

    #[test]
    fn pg2_warns_then_blocks_an_unchanged_alternating_cycle() {
        let mut guard = ProgressGuard::default();
        assert_eq!(
            guard.record(failed("execution", "a", "state-1")),
            ProgressDecision::Continue
        );
        assert!(matches!(
            guard.record(failed("execution", "b", "state-1")),
            ProgressDecision::Warn(_)
        ));
        assert!(
            guard
                .preflight(
                    "execution",
                    "a",
                    ProgressState {
                        workspace_fingerprint: "state-1".to_string(),
                        evidence_fingerprint: "state-1".to_string(),
                    },
                )
                .is_some()
        );
    }

    #[test]
    fn pg3_real_state_transition_resets_the_failure_sequence() {
        let mut guard = ProgressGuard::default();
        assert_eq!(
            guard.record(failed("workspace_edit", "stale", "state-1")),
            ProgressDecision::Continue
        );
        assert!(matches!(
            guard.record(failed("workspace_edit", "stale", "state-1")),
            ProgressDecision::Warn(_)
        ));
        assert_eq!(
            guard.record(failed("workspace_edit", "stale", "state-2")),
            ProgressDecision::Continue
        );
        assert!(
            guard
                .preflight(
                    "workspace_edit",
                    "stale",
                    ProgressState {
                        workspace_fingerprint: "state-2".to_string(),
                        evidence_fingerprint: "state-2".to_string(),
                    },
                )
                .is_none()
        );
    }

    #[test]
    fn equivalent_read_failures_do_not_block_a_different_recovery_read() {
        let mut guard = ProgressGuard::default();
        assert_eq!(
            guard.record(failed("repository_read", "missing", "state-1")),
            ProgressDecision::Continue
        );
        assert!(matches!(
            guard.record(failed("repository_read", "missing", "state-1")),
            ProgressDecision::Warn(_)
        ));
        assert_eq!(
            guard.preflight(
                "repository_read",
                "different-path",
                ProgressState {
                    workspace_fingerprint: "state-1".to_string(),
                    evidence_fingerprint: "state-1".to_string(),
                },
            ),
            None
        );
    }

    #[test]
    fn known_empty_read_range_is_blocked_after_one_unchanged_empty_result() {
        let state = ProgressState {
            workspace_fingerprint: "state-1".to_string(),
            evidence_fingerprint: "state-1".to_string(),
        };
        let mut guard = ProgressGuard::default();
        let mut observation = failed("repository_read", "empty", "state-1");
        observation.call_fingerprint = "read_file:known_empty:path-hash".to_string();
        assert_eq!(guard.record(observation), ProgressDecision::Continue);
        assert!(
            guard
                .preflight("repository_read", "read_file:known_empty:path-hash", state,)
                .is_some()
        );
    }

    #[test]
    fn pg4_varied_stale_patch_results_share_one_outcome_identity() {
        let first = outcome_identity(
            "apply_patch",
            false,
            "tool 'apply_patch' failed: error: patch failed: src/a.rs:10\nPatch mismatch diagnostic",
        );
        let second = outcome_identity(
            "apply_patch",
            false,
            "tool 'apply_patch' failed: error: patch failed: src/b.rs:200\nPatch mismatch diagnostic",
        );
        assert_eq!(first, second);

        let mut guard = ProgressGuard::default();
        assert_eq!(
            guard.record(failed("workspace_edit", &first.0, "state")),
            ProgressDecision::Continue
        );
        assert!(matches!(
            guard.record(failed("workspace_edit", &second.0, "state")),
            ProgressDecision::Warn(_)
        ));
        assert!(
            guard
                .preflight(
                    "workspace_edit",
                    "third-varied-call",
                    ProgressState {
                        workspace_fingerprint: "state".to_string(),
                        evidence_fingerprint: "state".to_string(),
                    },
                )
                .is_some()
        );
    }
}
