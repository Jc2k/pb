use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalCommandContext {
    operation: String,
    details: Vec<(String, String)>,
}

impl MetalCommandContext {
    pub(crate) fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            details: Vec::new(),
        }
    }

    pub(crate) fn with(mut self, key: impl Into<String>, value: impl ToString) -> Self {
        self.details.push((key.into(), value.to_string()));
        self
    }

    pub(crate) fn label(&self) -> String {
        let mut label = format!("Flash-MoE {}", self.operation);
        for (key, value) in &self.details {
            label.push(' ');
            label.push_str(key);
            label.push('=');
            label.push_str(value);
        }
        label
    }

    pub(crate) fn detail_summary(&self) -> String {
        if self.details.is_empty() {
            "none".to_string()
        } else {
            self.details
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalCommandStatus {
    NotEnqueued,
    Enqueued,
    Committed,
    Scheduled,
    Completed,
    Error,
    Unknown(usize),
}

impl MetalCommandStatus {
    pub(crate) fn from_raw(raw: usize) -> Self {
        match raw {
            0 => Self::NotEnqueued,
            1 => Self::Enqueued,
            2 => Self::Committed,
            3 => Self::Scheduled,
            4 => Self::Completed,
            5 => Self::Error,
            value => Self::Unknown(value),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::NotEnqueued => "not_enqueued",
            Self::Enqueued => "enqueued",
            Self::Committed => "committed",
            Self::Scheduled => "scheduled",
            Self::Completed => "completed",
            Self::Error => "error",
            Self::Unknown(_) => "unknown",
        }
    }

    pub(crate) fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Error)
    }
}

impl std::fmt::Display for MetalCommandStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(raw) => write!(f, "unknown({raw})"),
            status => f.write_str(status.as_str()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalCommandFailureKind {
    Timeout,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalCommandBufferFailure {
    kind: MetalCommandFailureKind,
    message: String,
}

impl MetalCommandBufferFailure {
    pub(crate) fn timeout(
        context: &MetalCommandContext,
        elapsed: Duration,
        status: MetalCommandStatus,
        metal_error: Option<String>,
    ) -> Self {
        Self {
            kind: MetalCommandFailureKind::Timeout,
            message: format_metal_command_failure(
                MetalCommandFailureKind::Timeout,
                context,
                elapsed,
                status,
                metal_error.as_deref(),
            ),
        }
    }

    pub(crate) fn failed(
        context: &MetalCommandContext,
        elapsed: Duration,
        status: MetalCommandStatus,
        metal_error: Option<String>,
    ) -> Self {
        Self {
            kind: MetalCommandFailureKind::Failed,
            message: format_metal_command_failure(
                MetalCommandFailureKind::Failed,
                context,
                elapsed,
                status,
                metal_error.as_deref(),
            ),
        }
    }

    pub(crate) fn should_release_buffers(&self) -> bool {
        true
    }
}

impl std::fmt::Display for MetalCommandBufferFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MetalCommandBufferFailure {}

pub(crate) fn metal_command_failure_requires_release(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<MetalCommandBufferFailure>()
        .is_some_and(MetalCommandBufferFailure::should_release_buffers)
}

pub(crate) fn format_metal_command_failure(
    kind: MetalCommandFailureKind,
    context: &MetalCommandContext,
    elapsed: Duration,
    status: MetalCommandStatus,
    metal_error: Option<&str>,
) -> String {
    let action = match kind {
        MetalCommandFailureKind::Timeout => "timed out",
        MetalCommandFailureKind::Failed => "failed",
    };
    let error = metal_error
        .filter(|error| !error.trim().is_empty())
        .unwrap_or("none reported");
    format!(
        "Flash-MoE Metal command buffer {action}: label=\"{}\", elapsed={}ms, status={}, metal_error=\"{}\", details={}",
        context.label(),
        elapsed.as_millis(),
        status,
        error,
        context.detail_summary()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_context_label_includes_actionable_details() {
        let context = MetalCommandContext::new("deferred_expert_phase")
            .with("position", 17)
            .with("layer", 3)
            .with("experts", "1,7,9,11")
            .with("width", 4096);

        assert_eq!(
            context.label(),
            "Flash-MoE deferred_expert_phase position=17 layer=3 experts=1,7,9,11 width=4096"
        );
        assert_eq!(
            context.detail_summary(),
            "position=17, layer=3, experts=1,7,9,11, width=4096"
        );
    }

    #[test]
    fn command_status_names_known_and_unknown_values() {
        assert_eq!(MetalCommandStatus::from_raw(0).to_string(), "not_enqueued");
        assert_eq!(MetalCommandStatus::from_raw(3).to_string(), "scheduled");
        assert_eq!(MetalCommandStatus::from_raw(4).to_string(), "completed");
        assert_eq!(MetalCommandStatus::from_raw(5).to_string(), "error");
        assert_eq!(MetalCommandStatus::from_raw(99).to_string(), "unknown(99)");
        assert!(MetalCommandStatus::Completed.is_terminal());
        assert!(MetalCommandStatus::Error.is_terminal());
        assert!(!MetalCommandStatus::Scheduled.is_terminal());
    }

    #[test]
    fn command_failure_diagnostic_is_actionable() {
        let context = MetalCommandContext::new("gqa_attention_scores")
            .with("layer", 12)
            .with("position", 128)
            .with("tokens", 129)
            .with("q_heads", 32)
            .with("kv_heads", 8);

        let message = format_metal_command_failure(
            MetalCommandFailureKind::Timeout,
            &context,
            Duration::from_millis(1234),
            MetalCommandStatus::Scheduled,
            Some("GPU timeout"),
        );

        assert!(message.contains("timed out"));
        assert!(message.contains("label=\"Flash-MoE gqa_attention_scores"));
        assert!(message.contains("elapsed=1234ms"));
        assert!(message.contains("status=scheduled"));
        assert!(message.contains("metal_error=\"GPU timeout\""));
        assert!(message.contains("layer=12"));
        assert!(message.contains("position=128"));
        assert!(message.contains("tokens=129"));
    }

    #[test]
    fn command_failure_marks_buffers_for_release() {
        let context = MetalCommandContext::new("lm_head_topk").with("rows", 42);
        let error = MetalCommandBufferFailure::failed(
            &context,
            Duration::from_millis(7),
            MetalCommandStatus::Error,
            None,
        );
        let anyhow_error = anyhow::Error::from(error.clone());

        assert!(error.should_release_buffers());
        assert!(metal_command_failure_requires_release(&anyhow_error));
        assert!(error.to_string().contains("none reported"));
    }
}
