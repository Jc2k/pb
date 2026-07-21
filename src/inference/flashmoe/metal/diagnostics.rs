use super::*;

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
    pub(super) kind: MetalCommandFailureKind,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalCommandWaitPolicy {
    pub(crate) timeout: Duration,
    pub(crate) poll_interval: Duration,
}

impl Default for MetalCommandWaitPolicy {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_FLASHMOE_METAL_COMMAND_TIMEOUT,
            poll_interval: DEFAULT_FLASHMOE_METAL_COMMAND_POLL_INTERVAL,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MetalCommandWaitResult {
    Pending,
    Finished(std::result::Result<(), MetalCommandBufferFailure>),
}

pub(crate) fn resolve_metal_command_wait(
    context: &MetalCommandContext,
    elapsed: Duration,
    status: MetalCommandStatus,
    metal_error: Option<String>,
    timed_out: bool,
) -> MetalCommandWaitResult {
    if status.is_terminal() {
        return match status {
            MetalCommandStatus::Completed if metal_error.is_none() => {
                MetalCommandWaitResult::Finished(Ok(()))
            }
            _ => MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure::failed(
                context,
                elapsed,
                status,
                metal_error,
            ))),
        };
    }
    if timed_out {
        return MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure::timeout(
            context,
            elapsed,
            status,
            metal_error,
        )));
    }
    MetalCommandWaitResult::Pending
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
