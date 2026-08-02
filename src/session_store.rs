use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agent_core::{AgentRequest, find_git_root};
use crate::events::{AgentEvent, EventEnvelope, GoalChangeKind, SessionMetricsSnapshot};
use crate::projects::ProjectEntry;

const NOTES_NAMESPACE: &str = "refs/notes/pb/sessions";
const MAX_RESTORED_HISTORY_EVENTS: usize = 1_000;
const SESSION_GIT_NAME: &str = "pb";
const SESSION_GIT_EMAIL: &str = "pb@localhost";
pub const SESSION_SCHEMA_VERSION: &str = "v4";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionProject {
    pub name: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingGoalChange {
    pub goal_id: String,
    pub kind: GoalChangeKind,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistedSession {
    pub schema_version: String,
    pub session_id: String,
    pub task: String,
    pub title: Option<String>,
    pub branch: Option<String>,
    pub workdir: Option<PathBuf>,
    pub project: Option<SessionProject>,
    pub pending_delivery_proposal: Option<crate::workflow::DeliveryProposal>,
    pub pending_goal_proposal: Option<crate::goal::GoalProposal>,
    pub pending_goal_change: Option<PendingGoalChange>,
    pub request_template: AgentRequest,
    pub running: bool,
    pub status: SessionStatus,
    pub started_at_ms: u64,
    pub updated_at_ms: u64,
    pub metrics: Option<SessionMetricsSnapshot>,
    pub usage_records: Vec<SessionMetricsSnapshot>,
    pub workflow: Option<crate::workflow::WorkflowCheckpoint>,
    pub completed_workflows: Vec<crate::workflow::WorkflowSummary>,
    pub goal: Option<crate::goal::GoalCheckpoint>,
    pub completed_goals: Vec<crate::goal::GoalCheckpoint>,
    pub multi_task: Option<crate::task_queue::MultiTaskCheckpoint>,
    pub completed_multi_tasks: Vec<crate::task_queue::MultiTaskCheckpoint>,
    pub pending_user_messages: Vec<crate::events::QueuedUserMessage>,
    pub events: Vec<EventEnvelope>,
}

impl PersistedSession {
    pub fn from_parts(
        session_id: String,
        mut request_template: AgentRequest,
        branch: Option<String>,
        workdir: Option<PathBuf>,
        running: bool,
        status: SessionStatus,
        mut events: Vec<EventEnvelope>,
    ) -> Self {
        request_template.session_id.clone_from(&session_id);
        let mut last_sequence = 0;
        for envelope in &mut events {
            if envelope.transcript.sequence <= last_sequence {
                envelope.assign_sequence(last_sequence.saturating_add(1));
            }
            last_sequence = envelope.transcript.sequence;
        }
        let now = now_millis();
        let started_at_ms = events
            .iter()
            .find_map(|envelope| match &envelope.event {
                AgentEvent::Started { timestamp_ms, .. } => *timestamp_ms,
                _ => None,
            })
            .or_else(|| {
                session_id
                    .strip_prefix("session-")
                    .and_then(|timestamp| timestamp.parse().ok())
            })
            .unwrap_or(now);
        let usage_records = events
            .iter()
            .filter_map(|envelope| SessionMetricsSnapshot::from_event(&envelope.event))
            .collect::<Vec<_>>();
        Self {
            schema_version: SESSION_SCHEMA_VERSION.to_string(),
            session_id,
            task: request_template.task.clone(),
            title: latest_session_title(&events),
            branch,
            workdir,
            project: None,
            pending_delivery_proposal: None,
            pending_goal_proposal: None,
            pending_goal_change: None,
            request_template,
            running,
            status,
            started_at_ms,
            updated_at_ms: now,
            metrics: latest_session_metrics(&events),
            usage_records,
            workflow: None,
            completed_workflows: Vec::new(),
            goal: None,
            completed_goals: Vec::new(),
            multi_task: None,
            completed_multi_tasks: Vec::new(),
            pending_user_messages: Vec::new(),
            events: trim_events(events),
        }
    }
}

pub fn save_session(session: &PersistedSession) -> Result<()> {
    let Some(workspace_root) = workspace_root(session) else {
        return Ok(());
    };
    ensure_git_repository(&workspace_root)?;
    let note_ref = session_note_ref(&session.session_id)?;
    let target = note_target(&workspace_root, &session.session_id)?;
    let payload = serde_json::to_string_pretty(session).context("failed to serialize session")?;
    let note_file = write_temp_note(&session.session_id, &payload)?;
    let result = run_git(
        &workspace_root,
        [
            "notes",
            "--ref",
            note_ref.as_str(),
            "add",
            "-f",
            "-F",
            note_file.to_string_lossy().as_ref(),
            target.trim(),
        ],
    );
    let _ = std::fs::remove_file(&note_file);
    result.map(|_| ())
}

pub fn delete_session(workdir: &Path, session_id: &str) -> Result<()> {
    let workspace_root = find_git_root(workdir).unwrap_or_else(|| workdir.to_path_buf());
    let note_ref = session_note_ref(session_id)?;
    run_git(&workspace_root, ["update-ref", "-d", note_ref.as_str()]).map(|_| ())
}

pub fn restore_registered_sessions(projects: &[ProjectEntry]) -> Vec<PersistedSession> {
    let mut sessions = Vec::new();
    let mut restored_roots = HashSet::new();
    for project in projects {
        let path = PathBuf::from(&project.path);
        let Ok(root) = path.canonicalize() else {
            continue;
        };
        let root = find_git_root(&root).unwrap_or(root);
        if !restored_roots.insert(root.clone()) {
            continue;
        }
        match restore_project_sessions(&root) {
            Ok(mut restored) => sessions.append(&mut restored),
            Err(err) => eprintln!(
                "failed to restore pb sessions for {}: {err:#}",
                root.display()
            ),
        }
    }
    sessions.sort_by_key(|session| std::cmp::Reverse(session.updated_at_ms));
    sessions
}

pub fn restore_project_sessions(workspace_root: &Path) -> Result<Vec<PersistedSession>> {
    ensure_git_repository(workspace_root)?;
    let refs = run_git(
        workspace_root,
        ["for-each-ref", NOTES_NAMESPACE, "--format=%(refname)"],
    )?;
    let mut sessions = Vec::new();
    for note_ref in refs.lines().map(str::trim).filter(|line| !line.is_empty()) {
        match read_note(workspace_root, note_ref).and_then(|payload| parse_session(&payload)) {
            Ok(mut session) => {
                if let Some(checkpoint) = session.goal.take() {
                    let mut run = checkpoint.run;
                    if matches!(
                        run.stage,
                        crate::goal::GoalStage::Planning
                            | crate::goal::GoalStage::PlanReview
                            | crate::goal::GoalStage::PlanRevision
                            | crate::goal::GoalStage::RunningMilestone
                            | crate::goal::GoalStage::Evaluating
                    ) {
                        if let Err(error) = run.pause_at_boundary(now_millis()) {
                            eprintln!(
                                "failed to pause restored goal '{}' from {note_ref}: {error:#}",
                                run.id
                            );
                        }
                    }
                    session.goal = crate::goal::GoalCheckpoint::new(run).ok();
                }
                if let Some(checkpoint) = session.multi_task.take() {
                    let mut run = checkpoint.run;
                    if run.stage == crate::task_queue::MultiTaskStage::RunningTask {
                        if let Err(error) =
                            run.apply(crate::task_queue::MultiTaskEvent::PauseRequested {
                                now_ms: now_millis(),
                            })
                        {
                            eprintln!(
                                "failed to pause restored multi-Task run '{}' from {note_ref}: {error:#}",
                                run.id
                            );
                        }
                    }
                    session.multi_task = crate::task_queue::MultiTaskCheckpoint::new(run).ok();
                }
                session.status = match session.status {
                    SessionStatus::Running | SessionStatus::Queued | SessionStatus::Paused => {
                        SessionStatus::Paused
                    }
                    SessionStatus::Completed => SessionStatus::Completed,
                    SessionStatus::Failed => SessionStatus::Failed,
                };
                session.running = false;
                session.events = trim_events(session.events);
                sessions.push(session);
            }
            Err(err) => eprintln!(
                "failed to restore pb session note {note_ref} in {}: {err:#}",
                workspace_root.display()
            ),
        }
    }
    Ok(sessions)
}

fn read_note(workspace_root: &Path, note_ref: &str) -> Result<String> {
    let notes = run_git(workspace_root, ["notes", "--ref", note_ref, "list"])?;
    let Some(note_oid) = notes
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .next()
    else {
        bail!("session note ref has no note entries");
    };
    run_git(workspace_root, ["cat-file", "-p", note_oid])
}

fn parse_session(payload: &str) -> Result<PersistedSession> {
    let value: serde_json::Value =
        serde_json::from_str(payload).context("failed to parse session note")?;
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .context("session note has no schema_version")?;
    if schema_version != SESSION_SCHEMA_VERSION {
        bail!("unsupported session schema '{schema_version}'; expected '{SESSION_SCHEMA_VERSION}'");
    }
    for field in [
        "title",
        "branch",
        "workdir",
        "project",
        "pending_delivery_proposal",
        "pending_goal_proposal",
        "pending_goal_change",
        "metrics",
        "workflow",
        "goal",
        "multi_task",
    ] {
        if value.get(field).is_none() {
            bail!("session note has no {field}");
        }
    }
    let session: PersistedSession =
        serde_json::from_value(value).context("failed to parse session note")?;
    validate_authoritative_session_state(&session)?;
    let mut entry_keys = HashSet::new();
    let mut last_sequence = 0;
    for (index, envelope) in session.events.iter().enumerate() {
        envelope
            .validate_persisted_with_history(&session.events[..index])
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("session event {index} has invalid server projections"))?;
        if envelope.transcript.sequence <= last_sequence {
            bail!(
                "session event {index} has non-monotonic transcript sequence {}",
                envelope.transcript.sequence
            );
        }
        for superseded in &envelope.transcript.supersedes {
            if !entry_keys.contains(superseded.as_str()) {
                bail!(
                    "session event {index} supersedes unknown or future transcript entry '{superseded}'"
                );
            }
        }
        if !entry_keys.insert(envelope.transcript.entry_key.as_str()) {
            bail!(
                "session event {index} repeats transcript entry key '{}'",
                envelope.transcript.entry_key
            );
        }
        last_sequence = envelope.transcript.sequence;
    }
    if let Some(goal) = session.goal.as_ref() {
        goal.validate()
            .context("active goal checkpoint is invalid")?;
    }
    for goal in &session.completed_goals {
        goal.validate()
            .context("completed goal checkpoint is invalid")?;
    }
    if let Some(multi_task) = session.multi_task.as_ref() {
        multi_task
            .validate()
            .context("active multi-Task checkpoint is invalid")?;
    }
    for multi_task in &session.completed_multi_tasks {
        multi_task
            .validate()
            .context("completed multi-Task checkpoint is invalid")?;
    }
    Ok(session)
}

fn validate_authoritative_session_state(session: &PersistedSession) -> Result<()> {
    if session.session_id.trim().is_empty()
        || session.request_template.session_id != session.session_id
    {
        bail!("session identity does not match its request template");
    }
    if session.running != (session.status == SessionStatus::Running) {
        bail!("session running flag does not match its status");
    }
    if let Some(project) = session.project.as_ref()
        && (project.name.trim().is_empty()
            || project.path.trim().is_empty()
            || !Path::new(&project.path).is_absolute())
    {
        bail!("session project identity is invalid");
    }
    if let Some(proposal) = session.pending_delivery_proposal.as_ref()
        && (proposal.id.trim().is_empty()
            || proposal.source_turn_id.trim().is_empty()
            || proposal.task_summary.trim().is_empty())
    {
        bail!("pending delivery proposal is invalid");
    }
    if let Some(proposal) = session.pending_goal_proposal.as_ref()
        && (proposal.id.trim().is_empty()
            || proposal.source_turn_id.trim().is_empty()
            || proposal.objective.trim().is_empty()
            || proposal
                .criteria
                .iter()
                .any(|criterion| criterion.text.trim().is_empty()))
    {
        bail!("pending Goal proposal is invalid");
    }
    if let Some(change) = session.pending_goal_change.as_ref()
        && (change.goal_id.trim().is_empty()
            || change.summary.trim().is_empty()
            || session
                .goal
                .as_ref()
                .is_none_or(|goal| goal.run.id != change.goal_id))
    {
        bail!("pending Goal change does not match the active Goal");
    }
    let mut message_ids = HashSet::new();
    if session.pending_user_messages.iter().any(|message| {
        message.message_id.trim().is_empty()
            || message.message.trim().is_empty()
            || !message_ids.insert(message.message_id.as_str())
    }) {
        bail!("pending user messages are invalid");
    }
    if session
        .usage_records
        .iter()
        .any(|metrics| metrics.ended_at_ms < metrics.started_at_ms)
    {
        bail!("session usage record ends before it starts");
    }
    let mut combined: Option<SessionMetricsSnapshot> = None;
    for metrics in &session.usage_records {
        if let Some(existing) = combined.as_mut() {
            existing.add_assign(metrics);
        } else {
            combined = Some(metrics.clone());
        }
    }
    if session.metrics != combined {
        bail!("session metrics do not match its usage records");
    }
    Ok(())
}

fn latest_session_metrics(events: &[EventEnvelope]) -> Option<SessionMetricsSnapshot> {
    let mut total: Option<SessionMetricsSnapshot> = None;
    for metrics in events
        .iter()
        .filter_map(|envelope| SessionMetricsSnapshot::from_event(&envelope.event))
    {
        if let Some(existing) = total.as_mut() {
            existing.add_assign(&metrics);
        } else {
            total = Some(metrics);
        }
    }
    total
}

pub fn latest_session_title(events: &[EventEnvelope]) -> Option<String> {
    events
        .iter()
        .rev()
        .find_map(|envelope| match &envelope.event {
            AgentEvent::SessionTitle { title, .. } if !title.trim().is_empty() => {
                Some(title.trim().to_string())
            }
            _ => None,
        })
}

fn trim_events(events: Vec<EventEnvelope>) -> Vec<EventEnvelope> {
    trim_event_history(events, MAX_RESTORED_HISTORY_EVENTS)
}

pub(crate) fn trim_event_history(events: Vec<EventEnvelope>, maximum: usize) -> Vec<EventEnvelope> {
    if events.len() <= maximum {
        return events;
    }
    if maximum == 0 {
        return Vec::new();
    }
    let mut retained = HashSet::new();
    if matches!(
        events.first().map(|entry| &entry.event),
        Some(AgentEvent::Started { .. })
    ) {
        retained.insert(0);
        let keep_tail = maximum.saturating_sub(1);
        retained.extend(events.len().saturating_sub(keep_tail)..events.len());
    } else {
        retained.extend(events.len() - maximum..events.len());
    }

    // Event projections are authoritative persisted data. Retaining an event therefore also
    // retains every earlier event needed to recompute its projections during strict restore.
    // The configured maximum is deliberately soft when preserving that dependency closure.
    let entry_indices = events
        .iter()
        .enumerate()
        .map(|(index, event)| (event.transcript.entry_key.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut pending = retained.iter().copied().collect::<VecDeque<_>>();
    while let Some(index) = pending.pop_front() {
        let envelope = &events[index];
        let mut dependencies = envelope
            .transcript
            .supersedes
            .iter()
            .filter_map(|entry_key| entry_indices.get(entry_key.as_str()).copied())
            .collect::<Vec<_>>();

        match &envelope.event {
            AgentEvent::ToolResult { tool, call_id, .. } => {
                if let Some(dependency) = events[..index].iter().rposition(|candidate| {
                    matches!(
                        &candidate.event,
                        AgentEvent::ToolCall {
                            tool: called_tool,
                            call_id: called_id,
                            ..
                        } if called_tool == tool && called_id == call_id
                    )
                }) {
                    dependencies.push(dependency);
                }
            }
            AgentEvent::SessionSummary { .. } => {
                if let Some(dependency) = events[..index]
                    .iter()
                    .rposition(|candidate| matches!(candidate.event, AgentEvent::Final { .. }))
                {
                    dependencies.push(dependency);
                }
                if let Some(dependency) = events[..index].iter().rposition(|candidate| {
                    matches!(candidate.event, AgentEvent::WorkflowBlocked { .. })
                }) {
                    dependencies.push(dependency);
                }
            }
            AgentEvent::TeamMessage { evidence, .. } => {
                for reference in evidence {
                    let dependency = match reference {
                        crate::events::EvidenceRef::Check { check_id } => {
                            events[..index].iter().rposition(|candidate| {
                                matches!(
                                    &candidate.event,
                                    AgentEvent::CheckResult { check_id: candidate, .. }
                                        if candidate == check_id
                                )
                            })
                        }
                        crate::events::EvidenceRef::Commit { oid } => {
                            events[..index].iter().rposition(|candidate| {
                                matches!(
                                    &candidate.event,
                                    AgentEvent::CommitResult { oid: Some(candidate), .. }
                                        if candidate == oid
                                )
                            })
                        }
                    };
                    dependencies.extend(dependency);
                }
            }
            _ => {}
        }

        if envelope.transcript.related_action_key.is_some()
            && let Some(dependency) = events[..index]
                .iter()
                .rposition(|candidate| matches!(candidate.event, AgentEvent::ToolCall { .. }))
        {
            dependencies.push(dependency);
        }

        for dependency in dependencies {
            if retained.insert(dependency) {
                pending.push_back(dependency);
            }
        }
    }

    events
        .into_iter()
        .enumerate()
        .filter_map(|(index, event)| retained.contains(&index).then_some(event))
        .collect()
}

fn workspace_root(session: &PersistedSession) -> Option<PathBuf> {
    session
        .workdir
        .as_deref()
        .or(session.request_template.workdir.as_deref())
        .and_then(find_git_root)
}

fn ensure_git_repository(workspace_root: &Path) -> Result<()> {
    run_git(workspace_root, ["rev-parse", "--git-dir"]).map(|_| ())
}

fn note_target(workspace_root: &Path, session_id: &str) -> Result<String> {
    let mut child = session_git_command()
        .args(["hash-object", "-w", "--stdin"])
        .current_dir(workspace_root)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "failed to run git hash-object in {}",
                workspace_root.display()
            )
        })?;
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().context("failed to open git stdin")?;
        stdin
            .write_all(format!("pb-session:{session_id}\n").as_bytes())
            .context("failed to write git hash-object input")?;
    }
    let output = child
        .wait_with_output()
        .context("failed to wait for git hash-object")?;
    command_output(output, workspace_root, "git hash-object -w --stdin")
}

fn session_note_ref(session_id: &str) -> Result<String> {
    if session_id.is_empty()
        || session_id
            .chars()
            .any(|ch| !(ch.is_ascii_alphanumeric() || ch == '-' || ch == '_'))
    {
        bail!("invalid session id for git notes ref: {session_id}");
    }
    Ok(format!("{NOTES_NAMESPACE}/{session_id}"))
}

fn write_temp_note(session_id: &str, payload: &str) -> Result<PathBuf> {
    let path = std::env::temp_dir().join(format!(
        "pb-session-note-{session_id}-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, payload)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

fn run_git<const N: usize>(workspace_root: &Path, args: [&str; N]) -> Result<String> {
    let output = session_git_command()
        .args(args)
        .current_dir(workspace_root)
        .output()
        .with_context(|| format!("failed to run git in {}", workspace_root.display()))?;
    command_output(output, workspace_root, "git")
}

fn session_git_command() -> Command {
    // Git notes writes create commits, so provide an internal identity instead of
    // requiring CI or user machines to configure one globally.
    let mut command = Command::new("git");
    command
        .env("GIT_AUTHOR_NAME", SESSION_GIT_NAME)
        .env("GIT_AUTHOR_EMAIL", SESSION_GIT_EMAIL)
        .env("GIT_COMMITTER_NAME", SESSION_GIT_NAME)
        .env("GIT_COMMITTER_EMAIL", SESSION_GIT_EMAIL);
    command
}

fn command_output(
    output: std::process::Output,
    workspace_root: &Path,
    command: &str,
) -> Result<String> {
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("{command} failed in {}: {stderr}", workspace_root.display());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_core::AgentProfile;
    use crate::environment::EnvironmentConfig;
    use crate::events::AgentEvent;
    use tempfile::TempDir;

    fn request(workdir: &Path) -> AgentRequest {
        AgentRequest {
            task: "test task".to_string(),
            turn_id: "turn-test".to_string(),
            intent: Some(crate::workflow::TurnIntent::Discuss),
            task_planning: crate::agent_core::TaskPlanningPreference::Auto,
            task_plan_rejected: None,
            task_planning_transcript: None,
            workflow_policy: None,
            workflow_stage: None,
            workflow_expected_content_fingerprint: None,
            workflow_plan_identity: None,
            workflow_plan_paths: Vec::new(),
            workflow_plan_revision_challenge_ids: Vec::new(),
            workflow_action_first_turn: false,
            workflow_creation_path_order: Vec::new(),
            workflow_work_units: None,
            workflow_repair_feedback: None,
            workflow_stage_evidence: None,
            workflow_checkpoint: None,
            conversation_handoff: None,
            legacy_prompt_owned_delivery: false,
            model: "model.gguf".to_string(),
            model_dir: None,
            workdir: Some(workdir.to_path_buf()),
            branch: Some("pb/test".to_string()),
            max_steps: 1,
            max_tokens: 1,
            turn_max_tokens_cap: None,
            tool_allowlist: None,
            workflow_tool_exclusions: Vec::new(),
            observation_rendering: crate::workflow::ObservationRendering::Native,
            accept_existing_workspace_changes: false,
            ctx_size: 128,
            threads: None,
            threads_batch: None,
            gpu_layers: 0,
            temperature: 0.0,
            profile: crate::agent_core::AgentProfile::Build,
            infer_profile: false,
            sub_agent_depth: 0,
            repository_less: false,
            top_k: 1,
            seed: 0,
            environment: None::<EnvironmentConfig>,
            python_dependency_authority: Default::default(),
            environment_evidence_context: None,
            workspace_graph: None,
            repository_context: None,
            prior_check_evidence: crate::checks::CheckEvidenceLedger::default(),
            session_id: String::new(),
            attachments: Vec::new(),
            goal_context: None,
            contract: None,
        }
    }

    fn init_repo() -> TempDir {
        let dir = TempDir::new().unwrap();
        run_git(dir.path(), ["init"]).unwrap();
        dir
    }

    fn multi_task_checkpoint(repo: &Path) -> crate::task_queue::MultiTaskCheckpoint {
        let policy = crate::task_queue::TaskConfigDocument::default()
            .compile()
            .unwrap();
        let authority = crate::task_queue::TaskPlanAuthority {
            source_intent: crate::task_queue::TaskSourceIntent::Build,
            task_planning_qualified: true,
            automatic_goal_selection_qualified: false,
        };
        let proposal: crate::task_queue::TaskPlanProposal = serde_json::from_value(
            serde_json::json!({
                "objective": "Deliver two changes",
                "requirements": [
                    {"id": "r1", "description": "First"},
                    {"id": "r2", "description": "Second"}
                ],
                "tasks": [
                    {"id": "t1", "title": "First", "description": "First", "requirement_ids": ["r1"], "acceptance_ids": ["a1"], "effort": "small", "kind": "build"},
                    {"id": "t2", "title": "Second", "description": "Second", "requirement_ids": ["r2"], "depends_on": ["t1"], "acceptance_ids": ["a2"], "effort": "small", "kind": "build"}
                ],
                "acceptance": [
                    {"id": "a1", "description": "First is current"},
                    {"id": "a2", "description": "Second is current"}
                ]
            }),
        )
        .unwrap();
        let plan = crate::workflow::ArtifactEnvelope::new(
            "persisted-task-plan",
            proposal.validate_and_compile(authority, &policy).unwrap(),
        )
        .unwrap();
        let review = crate::workflow::ArtifactEnvelope::new(
            "persisted-task-plan-review",
            crate::task_queue::TaskPlanReviewArtifact {
                task_plan_sha256: plan.sha256.clone(),
                verdict: crate::task_queue::TaskPlanReviewVerdict::Pass,
                request_assessments: Vec::new(),
                audits: crate::task_queue::passing_task_plan_audits(),
                challenges: Vec::new(),
            },
        )
        .unwrap();
        let qualification_digest = "a".repeat(64);
        let qualification = crate::task_queue::TaskPlannerQualification::new(
            qualification_digest.clone(),
            qualification_digest.clone(),
            qualification_digest.clone(),
            qualification_digest,
            true,
            false,
        )
        .unwrap();
        let run = crate::task_queue::MultiTaskRun::start(
            "persisted-multi-task",
            "session-multi-task",
            "turn-multi-task",
            plan,
            review,
            policy,
            crate::workflow::WorkflowConfigDocument::default()
                .compile()
                .unwrap(),
            crate::goal::GoalConfigDocument::default()
                .compile()
                .unwrap(),
            32,
            crate::task_queue::TaskSourceIntent::Build,
            qualification,
            repo.to_string_lossy(),
            crate::task_queue::TaskRepositoryState::capture(repo).unwrap(),
            crate::task_queue::TaskCoordinationCounters {
                planning_attempts: 1,
                model_invocations: 1,
                generated_tokens: 100,
                advisory_calls: 0,
                elapsed_ms: 10,
            },
            10,
        )
        .unwrap();
        crate::task_queue::MultiTaskCheckpoint::new(run).unwrap()
    }

    fn metrics_event(tokens: usize, joules: f64, started_at_ms: u64) -> EventEnvelope {
        EventEnvelope::new(AgentEvent::SessionMetrics {
            llm_invocations: 1,
            llm_runtime_ms: 1_000,
            prompt_tokens: tokens,
            generated_tokens: 0,
            tool_calls: 0,
            tool_runtime_ms: 0,
            cache_persistence_queued_checkpoints: 0,
            cache_persistence_completed_checkpoints: 0,
            cache_persistence_wall_ms: 0,
            cache_persistence_failures: 0,
            llm_energy_joules: None,
            llm_energy_kwh: None,
            tool_energy_joules: None,
            tool_energy_kwh: None,
            wall_runtime_ms: 1_000,
            started_at_ms,
            ended_at_ms: started_at_ms + 1_000,
            total_energy_joules: Some(joules),
            total_energy_kwh: Some(joules / 3_600_000.0),
            gross_energy_joules: Some(joules + 2.0),
            adjusted_energy_joules: Some(joules + 1.0),
            average_power_watts: Some(joules),
            energy_measured_ms: Some(1_000),
            energy_coverage: Some(1.0),
            energy_source: Some("smc_system_total".to_string()),
            display_energy_excluded: true,
            idle_baseline_applied: true,
            energy_complete: true,
            energy_exclusive: true,
            nesting_depth: None,
            timestamp_ms: Some(started_at_ms + 1_000),
        })
    }

    #[test]
    fn session_note_ref_rejects_path_separators() {
        assert!(session_note_ref("session-123").is_ok());
        assert!(session_note_ref("../bad").is_err());
    }

    #[test]
    fn restore_preserves_and_pauses_active_multi_task_checkpoint() {
        let repo = init_repo();
        let mut persisted = PersistedSession::from_parts(
            "session-multi-task".to_string(),
            request(repo.path()),
            Some("pb/test".to_string()),
            Some(repo.path().to_path_buf()),
            true,
            SessionStatus::Running,
            Vec::new(),
        );
        let original = multi_task_checkpoint(repo.path());
        let request_id = original
            .run
            .active_task()
            .unwrap()
            .request
            .as_ref()
            .unwrap()
            .id
            .clone();
        persisted.multi_task = Some(original);
        save_session(&persisted).unwrap();

        let restored = restore_project_sessions(repo.path()).unwrap();
        assert_eq!(restored.len(), 1);
        let checkpoint = restored[0].multi_task.as_ref().unwrap();
        assert_eq!(
            checkpoint.run.stage,
            crate::task_queue::MultiTaskStage::Paused
        );
        assert_eq!(
            checkpoint
                .run
                .active_task()
                .unwrap()
                .request
                .as_ref()
                .unwrap()
                .id,
            request_id
        );
        assert_eq!(checkpoint.run.counters.coordination.planning_attempts, 1);
        checkpoint.validate().unwrap();
    }

    #[test]
    fn restore_preserves_plan_approval_and_pauses_only_active_goal_work() {
        let repo = init_repo();
        let request = request(repo.path());
        let make_run = |id: &str, session_id: &str| {
            crate::goal::GoalRun::start(
                id,
                session_id,
                "Ship durable goals",
                Vec::new(),
                crate::goal::GoalContinuationPolicy::ReviewPlanThenAutomatic,
                None,
                crate::goal::GoalConfigDocument::default()
                    .compile()
                    .unwrap(),
                repo.path().to_string_lossy(),
                1,
            )
            .unwrap()
        };

        let mut awaiting = PersistedSession::from_parts(
            "session-awaiting-plan".to_string(),
            request.clone(),
            None,
            Some(repo.path().to_path_buf()),
            false,
            SessionStatus::Paused,
            Vec::new(),
        );
        awaiting.goal = Some(
            crate::goal::GoalCheckpoint::new(make_run(
                "goal-awaiting-plan",
                "session-awaiting-plan",
            ))
            .unwrap(),
        );
        save_session(&awaiting).unwrap();

        let mut active_run = make_run("goal-active", "session-active");
        let plan = active_run.plan_sha256.clone();
        active_run.approve_plan(&plan, 2).unwrap();
        let mut active = PersistedSession::from_parts(
            "session-active".to_string(),
            request,
            None,
            Some(repo.path().to_path_buf()),
            true,
            SessionStatus::Running,
            Vec::new(),
        );
        active.goal = Some(crate::goal::GoalCheckpoint::new(active_run).unwrap());
        save_session(&active).unwrap();

        let restored = restore_project_sessions(repo.path()).unwrap();
        let awaiting = restored
            .iter()
            .find(|session| session.session_id == "session-awaiting-plan")
            .unwrap();
        assert_eq!(
            awaiting.goal.as_ref().unwrap().run.stage,
            crate::goal::GoalStage::AwaitingPlanApproval
        );
        let active = restored
            .iter()
            .find(|session| session.session_id == "session-active")
            .unwrap();
        assert_eq!(
            active.goal.as_ref().unwrap().run.stage,
            crate::goal::GoalStage::Paused
        );
        assert_eq!(
            active.goal.as_ref().unwrap().run.paused_stage,
            Some(crate::goal::GoalStage::RunningMilestone)
        );
        assert_eq!(active.status, SessionStatus::Paused);
    }

    #[test]
    fn continued_turn_metrics_are_persisted_cumulatively() {
        let metrics = latest_session_metrics(&[
            metrics_event(10, 20.0, 1_000),
            metrics_event(15, 30.0, 3_000),
        ])
        .unwrap();

        assert_eq!(metrics.prompt_tokens, 25);
        assert_eq!(metrics.wall_runtime_ms, 2_000);
        assert_eq!(metrics.total_energy_joules, Some(50.0));
        assert_eq!(metrics.started_at_ms, 1_000);
        assert_eq!(metrics.ended_at_ms, 4_000);
    }

    #[test]
    fn usage_records_outlive_the_trimmed_chat_history() {
        let dir = init_repo();
        let events = (0..1_005)
            .map(|index| metrics_event(1, 1.0, index * 2_000))
            .collect::<Vec<_>>();
        let session = PersistedSession::from_parts(
            "long-session".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            events,
        );

        assert_eq!(session.events.len(), MAX_RESTORED_HISTORY_EVENTS);
        assert_eq!(session.usage_records.len(), 1_005);
        assert_eq!(session.metrics.unwrap().prompt_tokens, 1_005);
    }

    #[test]
    fn latest_session_title_uses_last_non_empty_title() {
        let events = vec![
            EventEnvelope::new(AgentEvent::SessionTitle {
                title: " First title ".to_string(),
                timestamp_ms: None,
            }),
            EventEnvelope::new(AgentEvent::Final {
                content: "done".to_string(),
                profile: AgentProfile::Build,
                nesting_depth: None,
                timestamp_ms: None,
            }),
            EventEnvelope::new(AgentEvent::SessionTitle {
                title: " Updated title ".to_string(),
                timestamp_ms: None,
            }),
        ];

        assert_eq!(
            latest_session_title(&events).as_deref(),
            Some("Updated title")
        );
    }

    #[test]
    fn persisted_session_title_tracks_session_title_events() {
        let dir = init_repo();
        let request = request(dir.path());
        let session = PersistedSession::from_parts(
            "session-title".to_string(),
            request,
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            true,
            SessionStatus::Running,
            vec![EventEnvelope::new(AgentEvent::SessionTitle {
                title: " Tool supplied title ".to_string(),
                timestamp_ms: None,
            })],
        );

        assert_eq!(session.title.as_deref(), Some("Tool supplied title"));
    }

    #[test]
    fn persisted_session_without_v4_schema_is_rejected() {
        let dir = init_repo();
        let session = PersistedSession::from_parts(
            "old-session".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            Vec::new(),
        );
        let mut value = serde_json::to_value(session).unwrap();
        value.as_object_mut().unwrap().remove("schema_version");

        let error = parse_session(&serde_json::to_string(&value).unwrap()).unwrap_err();
        assert!(error.to_string().contains("has no schema_version"));
    }

    #[test]
    fn persisted_v4_session_requires_authoritative_state_fields() {
        let dir = init_repo();
        let session = PersistedSession::from_parts(
            "malformed-session".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            Vec::new(),
        );
        let value = serde_json::to_value(session).unwrap();

        for field in [
            "title",
            "branch",
            "workdir",
            "status",
            "started_at_ms",
            "usage_records",
            "pending_user_messages",
            "project",
            "pending_delivery_proposal",
            "pending_goal_proposal",
            "pending_goal_change",
            "metrics",
            "workflow",
            "goal",
            "multi_task",
        ] {
            let mut missing = value.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(
                parse_session(&serde_json::to_string(&missing).unwrap()).is_err(),
                "v4 session unexpectedly accepted without {field}"
            );
        }
    }

    #[test]
    fn old_sessions_are_rejected_without_migration() {
        let dir = init_repo();
        let session = PersistedSession::from_parts(
            "old-session".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            Vec::new(),
        );
        let value = serde_json::to_value(session).unwrap();
        for version in ["v2", "v3"] {
            let mut old = value.clone();
            old["schema_version"] = serde_json::Value::String(version.to_string());
            let error = parse_session(&serde_json::to_string(&old).unwrap()).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(&format!("unsupported session schema '{version}'"))
            );
        }
    }

    #[test]
    fn started_time_is_persisted_independently_of_trimmed_event_history() {
        let dir = init_repo();
        let started = EventEnvelope::new(AgentEvent::Started {
            task: "Review the boundary".to_string(),
            model: "local-model".to_string(),
            profile: crate::agent_core::AgentProfile::Review,
            workspace: dir.path().to_string_lossy().into_owned(),
            focus_root: Some(dir.path().to_string_lossy().into_owned()),
            branch: "pb/test".to_string(),
            attachments: Vec::new(),
            timestamp_ms: Some(42),
        });
        let session = PersistedSession::from_parts(
            "session-100".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            vec![started],
        );

        assert_eq!(session.started_at_ms, 42);
    }

    #[test]
    fn trimmed_history_preserves_the_started_anchor() {
        let started = EventEnvelope::new(AgentEvent::Started {
            task: "Review the boundary".to_string(),
            model: "local-model".to_string(),
            profile: crate::agent_core::AgentProfile::Review,
            workspace: "/workspace".to_string(),
            focus_root: Some("/workspace".to_string()),
            branch: "pb/test".to_string(),
            attachments: Vec::new(),
            timestamp_ms: Some(1),
        });
        let events = vec![
            started,
            EventEnvelope::new(AgentEvent::SessionTitle {
                title: "one".to_string(),
                timestamp_ms: Some(2),
            }),
            EventEnvelope::new(AgentEvent::SessionTitle {
                title: "two".to_string(),
                timestamp_ms: Some(3),
            }),
        ];

        let trimmed = trim_event_history(events, 2);
        assert!(matches!(trimmed[0].event, AgentEvent::Started { .. }));
        assert!(matches!(
            &trimmed[1].event,
            AgentEvent::SessionTitle { title, .. } if title == "two"
        ));
    }

    #[test]
    fn trimmed_history_keeps_projection_dependencies_and_strictly_restores() {
        let dir = init_repo();
        let mut events = vec![EventEnvelope::new(AgentEvent::Started {
            task: "Review the boundary".to_string(),
            model: "local-model".to_string(),
            profile: crate::agent_core::AgentProfile::Review,
            workspace: dir.path().to_string_lossy().into_owned(),
            focus_root: Some(dir.path().to_string_lossy().into_owned()),
            branch: "pb/test".to_string(),
            attachments: Vec::new(),
            timestamp_ms: Some(1),
        })];
        let mut push = |mut envelope: EventEnvelope| {
            envelope.assign_sequence(events.len() as u64 + 1);
            envelope.refresh_projections(&events);
            events.push(envelope);
        };
        push(EventEnvelope::new(AgentEvent::ToolCall {
            tool: "read_file".to_string(),
            arguments: serde_json::json!({"path": "src/lib.rs"}),
            call_id: "call-1".to_string(),
            batch_id: "batch-1".to_string(),
            actor: crate::events::TeamActor::agent(crate::agent_core::AgentProfile::Review),
            nesting_depth: None,
            timestamp_ms: Some(2),
        }));
        push(EventEnvelope::new(AgentEvent::ToolResult {
            tool: "read_file".to_string(),
            result: "contents".to_string(),
            call_id: "call-1".to_string(),
            batch_id: "batch-1".to_string(),
            outcome: crate::events::ToolOutcome::Succeeded,
            actor: crate::events::TeamActor::agent(crate::agent_core::AgentProfile::Review),
            duration_ms: 1,
            energy_joules: None,
            energy_kwh: None,
            average_power_watts: None,
            energy_shared_calls: None,
            nesting_depth: None,
            timestamp_ms: Some(3),
        }));
        push(EventEnvelope::new(AgentEvent::SessionTitle {
            title: "Retained tail".to_string(),
            timestamp_ms: Some(4),
        }));

        let trimmed = trim_event_history(events, 3);
        assert_eq!(
            trimmed.len(),
            4,
            "the history limit is soft for dependencies"
        );
        assert!(matches!(trimmed[1].event, AgentEvent::ToolCall { .. }));

        let session = PersistedSession::from_parts(
            "session-projection-closure".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            trimmed,
        );
        parse_session(&serde_json::to_string(&session).unwrap()).unwrap();
    }

    #[test]
    fn session_parser_rejects_invalid_projection_shapes_and_duplicate_entry_keys() {
        let dir = init_repo();
        let correction = EventEnvelope::new(AgentEvent::Correction {
            kind: crate::events::CorrectionKind::RepositoryEvidence,
            message: "detail".to_string(),
            summary: "Repository evidence".to_string(),
            actor: crate::events::TeamActor::workflow_steward(),
            assisting_profile: Some(crate::agent_core::AgentProfile::Review),
            nesting_depth: None,
            timestamp_ms: Some(1),
        });
        let session = PersistedSession::from_parts(
            "malformed-projections".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            vec![correction.clone()],
        );

        let mut invalid_projection = serde_json::to_value(&session).unwrap();
        invalid_projection["events"][0]["chatter"] = serde_json::json!([]);
        assert!(parse_session(&serde_json::to_string(&invalid_projection).unwrap()).is_err());

        let mut duplicate = session;
        let mut repeated_key = correction;
        repeated_key.assign_sequence(2);
        duplicate.events.push(repeated_key);
        let error = parse_session(&serde_json::to_string(&duplicate).unwrap()).unwrap_err();
        assert!(error.to_string().contains("repeats transcript entry key"));
    }

    #[test]
    fn session_parser_rejects_dangling_supersession_and_non_monotonic_sequence() {
        let dir = init_repo();
        let title = EventEnvelope::new(AgentEvent::SessionTitle {
            title: "Authoritative state".to_string(),
            timestamp_ms: Some(1),
        });
        let mut dangling = PersistedSession::from_parts(
            "dangling-supersession".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            vec![title],
        );
        let own_key = dangling.events[0].transcript.entry_key.clone();
        dangling.events[0].transcript.supersedes = vec![own_key];
        let error = parse_session(&serde_json::to_string(&dangling).unwrap()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unknown or future transcript entry")
        );

        let mut unordered = PersistedSession::from_parts(
            "unordered-sequence".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            vec![
                EventEnvelope::new(AgentEvent::SessionTitle {
                    title: "First".to_string(),
                    timestamp_ms: Some(1),
                }),
                EventEnvelope::new(AgentEvent::SessionTitle {
                    title: "Second".to_string(),
                    timestamp_ms: Some(2),
                }),
            ],
        );
        unordered.events[1].assign_sequence(1);
        let error = parse_session(&serde_json::to_string(&unordered).unwrap()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("non-monotonic transcript sequence")
        );
    }

    #[test]
    fn session_parser_rejects_contradictory_authoritative_state() {
        let dir = init_repo();
        let mut status = PersistedSession::from_parts(
            "contradictory-state".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            Vec::new(),
        );
        status.running = true;
        assert!(parse_session(&serde_json::to_string(&status).unwrap()).is_err());

        let mut project = status;
        project.running = false;
        project.project = Some(SessionProject {
            name: "pb".to_string(),
            path: "relative/path".to_string(),
        });
        assert!(parse_session(&serde_json::to_string(&project).unwrap()).is_err());

        let mut metrics = PersistedSession::from_parts(
            "contradictory-metrics".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            vec![metrics_event(1, 1.0, 1_000)],
        );
        metrics.usage_records.clear();
        assert!(parse_session(&serde_json::to_string(&metrics).unwrap()).is_err());

        let mut proposal = PersistedSession::from_parts(
            "invalid-proposal".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            Vec::new(),
        );
        proposal.pending_delivery_proposal = Some(crate::workflow::DeliveryProposal {
            id: String::new(),
            source_turn_id: "turn-1".to_string(),
            task_summary: "Deliver".to_string(),
        });
        assert!(parse_session(&serde_json::to_string(&proposal).unwrap()).is_err());

        let mut unknown = serde_json::to_value(PersistedSession::from_parts(
            "unknown-state".to_string(),
            request(dir.path()),
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            false,
            SessionStatus::Completed,
            Vec::new(),
        ))
        .unwrap();
        unknown["legacy_running"] = serde_json::Value::Bool(false);
        assert!(parse_session(&serde_json::to_string(&unknown).unwrap()).is_err());
    }

    #[test]
    fn save_restore_and_delete_session_note() {
        let dir = init_repo();
        let request = request(dir.path());
        let check = EventEnvelope::new(AgentEvent::CheckResult {
            check_id: "rust".to_string(),
            exit_status: 0,
            success: true,
            timed_out: false,
            output: "all tests passed".to_string(),
            truncated: false,
            duration_ms: 1,
            fingerprint: "check-fingerprint".to_string(),
            command: Some("cargo test --all-targets".to_string()),
            cwd: Some(dir.path().to_string_lossy().into_owned()),
            executor: Some("local".to_string()),
            source: None,
            command_fingerprint: None,
            dependency_outputs: std::collections::BTreeMap::new(),
            output_fingerprint: None,
            reused: false,
            skip_reason: None,
            nesting_depth: None,
            timestamp_ms: None,
        });
        let mut team_message = EventEnvelope::new(AgentEvent::TeamMessage {
            actor: crate::events::TeamActor::workflow_steward(),
            tone: crate::events::TeamMessageTone::Success,
            purpose: crate::events::TeamMessagePurpose::General,
            handoff: None,
            message: "Everything affected passed.".to_string(),
            detail: Some("cargo test --all-targets".to_string()),
            evidence: vec![crate::events::EvidenceRef::Check {
                check_id: "rust".to_string(),
            }],
            nesting_depth: None,
            timestamp_ms: None,
        });
        team_message.refresh_projections(std::slice::from_ref(&check));
        let session = PersistedSession::from_parts(
            "session-123".to_string(),
            request,
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            true,
            SessionStatus::Running,
            vec![
                EventEnvelope::new(AgentEvent::Final {
                    content: "done".to_string(),
                    profile: AgentProfile::Build,
                    nesting_depth: None,
                    timestamp_ms: None,
                }),
                check,
                team_message,
            ],
        );

        save_session(&session).unwrap();
        let restored = restore_project_sessions(dir.path()).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].session_id, "session-123");
        assert!(!restored[0].running);
        assert!(!restored[0].request_template.legacy_prompt_owned_delivery);
        assert_eq!(restored[0].events.len(), 3);
        assert!(matches!(
            &restored[0].events[2].event,
            AgentEvent::TeamMessage {
                actor: crate::events::TeamActor::Automation(
                    crate::events::AutomationActor::Trinity
                ),
                message,
                evidence,
                ..
            } if message == "Everything affected passed."
                && evidence
                    == &vec![crate::events::EvidenceRef::Check {
                        check_id: "rust".to_string(),
                    }]
        ));

        delete_session(dir.path(), "session-123").unwrap();
        assert!(restore_project_sessions(dir.path()).unwrap().is_empty());
    }
}
