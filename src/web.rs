use anyhow::{Context, Result, bail};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::{delete, get, patch, post};
use axum::{Json, Router};
use base64::Engine as _;
use futures::StreamExt;
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, mpsc};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, broadcast};
use tokio_stream::wrappers::BroadcastStream;

use crate::agent_core::{
    AgentProfile, AgentRequest, EventSink, SessionAttachment, run_agent_managed,
};
use crate::daemon_protocol::RpcFrame;
use crate::events::{
    AgentEvent, EventEnvelope, HandoffOutcome, QueuedUserMessage, SessionMetricsSnapshot,
};
use crate::integrations::{
    self, InstalledIntegration, IntegrationConfigSchema, IntegrationInstallRequest,
    MarketplaceIntegration,
};
use crate::projects::{
    self, AddProjectRequest, ProjectEntry, RemoveProjectRequest, UpdateProjectNotificationsRequest,
};
use crate::session_store::{
    self, PendingGoalChange, PersistedSession, SessionProject, SessionStatus, latest_session_title,
};
use crate::sleep_prevention::{SleepPrevention, SleepPreventionStatus};

const MAX_HISTORY_EVENTS: usize = 1_000;
const SESSION_HISTORY_RESPONSE_LIMIT: usize = 300;
const MAX_USER_MESSAGE_CHARS: usize = 8_000;
const MAX_PENDING_USER_MESSAGES: usize = 32;
const MAX_PROJECT_SESSION_TERMINAL_TRANSITIONS: usize = 256;
static USER_MESSAGE_SEQUENCE: AtomicU64 = AtomicU64::new(1);
static ID_FALLBACK_SEQUENCE: AtomicU64 = AtomicU64::new(1);
static SESSION_PERSISTENCE_LOCK: StdMutex<()> = StdMutex::new(());

const SERVICE_WORKER_JS: &str = r#"self.addEventListener("install",(event)=>{self.skipWaiting();});
self.addEventListener("activate",(event)=>{event.waitUntil(self.clients.claim());});
self.addEventListener("notificationclick",(event)=>{
  event.notification.close();
  const url=(event.notification.data&&event.notification.data.url)||"/";
  event.waitUntil(self.clients.matchAll({type:"window",includeUncontrolled:true}).then((clients)=>{
    for (const client of clients) {
      if ("focus" in client) {
        client.navigate(url);
        return client.focus();
      }
    }
    if (self.clients.openWindow) return self.clients.openWindow(url);
  }));
});
"#;

#[derive(Debug, Clone)]
pub struct ServeArgs {
    pub host: String,
    pub port: u16,
    pub socket_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InlineAttachment {
    pub name: String,
    pub mime: String,
    pub base64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartSessionRequest {
    pub task: String,
    #[serde(default)]
    pub intent: Option<crate::workflow::TurnIntent>,
    #[serde(default)]
    pub proposal_id: Option<String>,
    pub model: Option<String>,
    pub model_dir: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    pub workdir: Option<String>,
    pub branch: Option<String>,
    pub max_steps: Option<usize>,
    pub max_tokens: Option<i32>,
    pub ctx_size: Option<u32>,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub gpu_layers: Option<u32>,
    pub temperature: Option<f32>,
    pub profile: Option<AgentProfile>,
    pub top_k: Option<i32>,
    pub seed: Option<u32>,
    #[serde(default)]
    pub attachments: Vec<InlineAttachment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinueSessionRequest {
    pub task: String,
    #[serde(default)]
    pub intent: Option<crate::workflow::TurnIntent>,
    #[serde(default)]
    pub proposal_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SendSessionMessageRequest {
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerQuestionRequest {
    pub question_id: String,
    pub answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerSessionQuestionRequest {
    pub session_id: String,
    pub question_id: String,
    pub answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartGoalRequest {
    #[serde(default)]
    pub session_id: Option<String>,
    pub objective: String,
    #[serde(default)]
    pub criteria: Vec<crate::goal::GoalCriterionInput>,
    #[serde(default)]
    pub continuation: crate::goal::GoalContinuationPolicy,
    #[serde(default)]
    pub budget: Option<crate::goal::GoalBudget>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub workdir: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalDigestRequest {
    pub goal_sha256: String,
    #[serde(default)]
    pub plan_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GoalRpcMutationRequest {
    pub goal_id: String,
    pub goal_sha256: String,
    #[serde(default)]
    pub plan_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalAmendmentRequest {
    pub goal_sha256: String,
    pub objective: String,
    #[serde(default)]
    pub criteria: Vec<crate::goal::GoalCriterionInput>,
    pub continuation: crate::goal::GoalContinuationPolicy,
    #[serde(default)]
    pub budget: Option<crate::goal::GoalBudget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalResponse {
    pub session_id: String,
    pub goal_id: String,
    pub goal_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionResponse {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionListItem {
    pub session_id: String,
    pub task: String,
    pub title: Option<String>,
    pub status: SessionStatus,
    pub intent: Option<crate::workflow::TurnIntent>,
    pub branch: Option<String>,
    pub workdir: Option<String>,
    pub project: Option<SessionProject>,
    pub handoff_outcome: Option<HandoffOutcome>,
    pub pending_question: Option<PendingQuestionView>,
    pub started_at_ms: u64,
    pub updated_at_ms: u64,
    pub workflow_id: Option<String>,
    pub workflow_stage: Option<crate::workflow::WorkflowStage>,
    pub workflow_outcome: Option<crate::workflow::WorkflowOutcome>,
    pub strict_workflow: bool,
    pub goal: Option<crate::goal::GoalSummary>,
    pub active_goal: bool,
    pub multi_task: Option<MultiTaskSummary>,
    pub active_multi_task: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectSessionSnapshot {
    pub stream_id: String,
    pub revision: u64,
    pub usage_window_start_ms: u64,
    pub usage_window_end_ms: u64,
    pub terminal_transition_floor: u64,
    pub terminal_transitions: Vec<ProjectSessionTerminalTransition>,
    pub projects: Vec<ProjectEntry>,
    pub sessions: Vec<SessionListItem>,
    pub overall_usage: ProjectUsageSummary,
    pub project_usage: HashMap<String, ProjectUsageSummary>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectMutationReceipt<T> {
    pub value: T,
    pub snapshot: ProjectSessionSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectSessionTerminalTransition {
    pub entry_key: String,
    pub revision: u64,
    pub session_id: String,
    pub status: SessionStatus,
    pub task: String,
    pub title: Option<String>,
    pub handoff_outcome: Option<HandoffOutcome>,
    pub project: Option<ProjectEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiTaskSummary {
    pub id: String,
    pub stage: crate::task_queue::MultiTaskStage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<crate::task_queue::MultiTaskOutcome>,
    pub completed_tasks: usize,
    pub total_tasks: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_task_title: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectUsageStats {
    pub tokens: usize,
    pub runtime_ms: u64,
    pub tool_calls: usize,
    pub energy_kwh: Option<f64>,
    pub energy_joules: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectUsageSummary {
    pub total: ProjectUsageStats,
    pub today: ProjectUsageStats,
}

impl ProjectUsageStats {
    fn add_metrics(&mut self, metrics: &SessionMetricsSnapshot) {
        self.tokens = self.tokens.saturating_add(
            metrics
                .prompt_tokens
                .saturating_add(metrics.generated_tokens),
        );
        self.runtime_ms = self.runtime_ms.saturating_add(metrics_runtime_ms(metrics));
        self.tool_calls = self.tool_calls.saturating_add(metrics.tool_calls);
        if let Some(energy_joules) = metrics_energy_joules(metrics) {
            self.energy_joules = Some(self.energy_joules.unwrap_or(0.0) + energy_joules);
            self.energy_kwh = Some(self.energy_joules.unwrap_or(0.0) / 3_600_000.0);
        }
    }
}

fn metrics_energy_joules(metrics: &SessionMetricsSnapshot) -> Option<f64> {
    if let Some(joules) = metrics.total_energy_joules {
        return (joules.is_finite() && joules >= 0.0).then_some(joules);
    }
    if let Some(kwh) = metrics.total_energy_kwh {
        return (kwh.is_finite() && kwh >= 0.0).then_some(kwh * 3_600_000.0);
    }
    if metrics.wall_runtime_ms > 0 {
        return None;
    }
    let diagnostic_joules =
        metrics.llm_energy_joules.unwrap_or(0.0) + metrics.tool_energy_joules.unwrap_or(0.0);
    if diagnostic_joules.is_finite() && diagnostic_joules > 0.0 {
        Some(diagnostic_joules)
    } else {
        let diagnostic_kwh =
            metrics.llm_energy_kwh.unwrap_or(0.0) + metrics.tool_energy_kwh.unwrap_or(0.0);
        (diagnostic_kwh.is_finite() && diagnostic_kwh > 0.0).then_some(diagnostic_kwh * 3_600_000.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionDetails {
    pub session_id: String,
    pub task: String,
    pub title: Option<String>,
    pub cancel_requested: bool,
    pub status: SessionStatus,
    pub intent: Option<crate::workflow::TurnIntent>,
    pub branch: Option<String>,
    pub workdir: Option<String>,
    pub project: Option<SessionProject>,
    pub handoff_outcome: Option<HandoffOutcome>,
    pub pending_question: Option<PendingQuestionView>,
    pub events: Vec<EventEnvelope>,
    pub started_at_ms: u64,
    pub updated_at_ms: u64,
    pub metrics: Option<SessionMetricsSnapshot>,
    pub usage_records: Vec<SessionMetricsSnapshot>,
    pub workflow: Option<crate::workflow::WorkflowSummary>,
    pub strict_workflow: bool,
    pub goal: Option<crate::goal::GoalCheckpoint>,
    pub active_goal: bool,
    pub multi_task: Option<crate::task_queue::MultiTaskCheckpoint>,
    pub active_multi_task: bool,
    pub task_plan_rejected: Option<crate::task_queue::TaskPlanRejected>,
    pub task_planning_transcript: Option<crate::task_queue::TaskPlanningTranscript>,
    pub pending_delivery_proposal: Option<crate::workflow::DeliveryProposal>,
    pub pending_goal_proposal: Option<crate::goal::GoalProposal>,
    pub pending_goal_change: Option<PendingGoalChange>,
    pub revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionStreamSnapshot {
    pub session: SessionDetails,
    pub reset_history: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionMutationReceipt {
    pub response: SessionResponse,
    pub snapshot: SessionStreamSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalMutationReceipt {
    pub response: GoalResponse,
    pub snapshot: SessionStreamSnapshot,
}

struct SessionCommit<T> {
    value: T,
    snapshot: SessionStreamSnapshot,
    terminal_entry_key: Option<String>,
    watch_finished: bool,
    effects: SessionCommitEffects,
}

struct StagedSessionChange<T> {
    value: T,
    effects: SessionCommitEffects,
}

#[derive(Debug, Default)]
struct SessionCommitEffects {
    terminate_environment: bool,
    dispatch: bool,
    publish_collection: bool,
}

struct SessionCommitFinalization {
    dispatch: bool,
    terminate_environment: bool,
    publish_finished: bool,
    sender: SessionEventSender,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionCommitPolicy {
    Full,
    ProjectionOnly,
}

impl SessionCommitEffects {
    fn is_empty(&self) -> bool {
        !self.terminate_environment && !self.dispatch && !self.publish_collection
    }
}

impl<T> StagedSessionChange<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            effects: SessionCommitEffects::default(),
        }
    }

    fn terminate_environment(mut self) -> Self {
        self.effects.terminate_environment = true;
        self
    }
}

#[derive(Debug, thiserror::Error)]
#[error("session not found: {0}")]
struct GoalSessionNotFound(String);

#[derive(Debug, thiserror::Error)]
#[error("session not found: {0}")]
struct SessionNotFoundError(String);

#[derive(Debug, thiserror::Error)]
#[error("session persistence failed: {0:#}")]
struct SessionPersistenceError(#[source] anyhow::Error);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PendingQuestionView {
    pub question_id: String,
    pub question: String,
    pub choices: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeleteSessionResponse {
    pub session_id: String,
    pub deleted: bool,
    pub cleanup_warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeleteSessionMutationResponse {
    pub deletion: DeleteSessionResponse,
    pub snapshot: ProjectSessionSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub busy: bool,
    pub running_sessions: usize,
    pub queued_sessions: usize,
    pub paused_sessions: usize,
    pub completed_sessions: usize,
    pub failed_sessions: usize,
    pub total_sessions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSettingsResponse {
    pub prevent_sleep_while_working: bool,
    pub prevent_sleep_supported: bool,
    pub prevent_sleep_active: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prevent_sleep_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateWebSettingsRequest {
    pub prevent_sleep_while_working: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateTailscaleSettingsRequest {
    pub enabled: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RpcRequest {
    id: u64,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyRpcRequest {}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GoalLookupRpcRequest {
    goal_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectNotificationsRpcRequest {
    name: String,
    notify_on_finish: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WatchSessionRequest {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WatchSessionAcknowledgement {
    pub session_id: String,
}

#[derive(Debug, Clone)]
struct PendingQuestionState {
    question_id: String,
    question: String,
    choices: Vec<String>,
    responder: std::sync::mpsc::Sender<String>,
}

#[derive(Debug, Clone, Default)]
struct DurableSessionProjection {
    project: Option<SessionProject>,
    pending_delivery_proposal: Option<crate::workflow::DeliveryProposal>,
    pending_goal_proposal: Option<crate::goal::GoalProposal>,
    pending_goal_change: Option<PendingGoalChange>,
}

const MAX_PROJECT_USAGE_WINDOW_CACHE_ENTRIES: usize = 16;

#[derive(Debug)]
struct CachedProjectUsageWindow {
    window: UsageWindow,
    overall_today: WindowedUsageAccumulator,
    project_today: HashMap<String, WindowedUsageAccumulator>,
}

#[derive(Debug, Default)]
struct ProjectUsageWindowCache {
    entries: VecDeque<CachedProjectUsageWindow>,
}

#[derive(Debug, Clone)]
struct SessionRecord {
    revision: u64,
    task: String,
    title: Option<String>,
    branch: Option<String>,
    workdir: Option<PathBuf>,
    durable: DurableSessionProjection,
    request_template: AgentRequest,
    status: SessionStatus,
    metrics: Option<SessionMetricsSnapshot>,
    workflow: Option<crate::workflow::WorkflowCheckpoint>,
    completed_workflows: Vec<crate::workflow::WorkflowSummary>,
    goal: Option<crate::goal::GoalCheckpoint>,
    completed_goals: Vec<crate::goal::GoalCheckpoint>,
    multi_task: Option<crate::task_queue::MultiTaskCheckpoint>,
    completed_multi_tasks: Vec<crate::task_queue::MultiTaskCheckpoint>,
    started_at_ms: u64,
    updated_at_ms: u64,
}

#[derive(Debug)]
struct SessionRuntime {
    pending_question: Option<PendingQuestionState>,
    sender: SessionEventSender,
    history: Arc<StdMutex<Vec<EventEnvelope>>>,
    usage_records: Arc<StdMutex<Vec<SessionMetricsSnapshot>>>,
    pending_user_messages: Arc<StdMutex<VecDeque<QueuedUserMessage>>>,
    accepting_user_messages: Arc<AtomicBool>,
    pause_token: Arc<AtomicBool>,
    cancel_token: Arc<AtomicBool>,
}

#[derive(Debug)]
struct SessionState {
    record: SessionRecord,
    runtime: SessionRuntime,
}

impl std::ops::Deref for SessionState {
    type Target = SessionRecord;

    fn deref(&self) -> &Self::Target {
        &self.record
    }
}

impl std::ops::DerefMut for SessionState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.record
    }
}

fn session_watch_active(session: &SessionState) -> bool {
    matches!(
        session.status,
        SessionStatus::Queued | SessionStatus::Running
    ) || session.runtime.pending_question.is_some()
}

#[derive(Debug, Clone)]
struct SessionEventSender {
    sender: broadcast::Sender<SessionStreamPublication>,
}

#[derive(Debug, Clone)]
enum SessionStreamPublication {
    Event(EventEnvelope),
    Snapshot(SessionStreamSnapshot),
    Finished,
}

impl SessionEventSender {
    fn new(capacity: usize) -> Self {
        Self {
            sender: broadcast::channel(capacity).0,
        }
    }

    fn subscribe(&self) -> broadcast::Receiver<SessionStreamPublication> {
        self.sender.subscribe()
    }

    fn publish_committed(&self, publication: SessionStreamPublication) {
        let _ = self.sender.send(publication);
    }
}

#[derive(Debug, Default)]
struct ProjectSessionPublication {
    terminal_transitions: VecDeque<ProjectSessionTerminalTransition>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct ProjectSessionChangePublisher {
    revision: Arc<AtomicU64>,
    publication: Arc<StdMutex<ProjectSessionPublication>>,
    sender: broadcast::Sender<u64>,
}

impl ProjectSessionChangePublisher {
    fn publish_update(&self, terminal_transition: Option<ProjectSessionTerminalTransition>) -> u64 {
        self.publish_update_with_warnings(terminal_transition, None, true)
    }

    fn publish_reconciliation(&self, changed: bool, warnings: Vec<String>) -> u64 {
        self.publish_update_with_warnings(None, Some(warnings), changed)
    }

    fn publish_update_with_warnings(
        &self,
        mut terminal_transition: Option<ProjectSessionTerminalTransition>,
        warnings: Option<Vec<String>>,
        force: bool,
    ) -> u64 {
        let mut publication = self
            .publication
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let warnings_changed = warnings
            .as_ref()
            .is_some_and(|warnings| *warnings != publication.warnings);
        if let Some(warnings) = warnings {
            publication.warnings = warnings;
        }
        if !force && !warnings_changed {
            return self.revision.load(Ordering::SeqCst);
        }
        let revision = self
            .revision
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                Some(current.saturating_add(1))
            })
            .unwrap_or(u64::MAX)
            .saturating_add(1)
            .min(u64::MAX);
        if let Some(transition) = terminal_transition.as_mut() {
            transition.revision = revision;
        }
        if let Some(transition) = terminal_transition {
            publication.terminal_transitions.push_back(transition);
            while publication.terminal_transitions.len() > MAX_PROJECT_SESSION_TERMINAL_TRANSITIONS
            {
                publication.terminal_transitions.pop_front();
            }
        }
        let _ = self.sender.send(revision);
        revision
    }
}

#[derive(Debug, Clone)]
struct AppState {
    sessions: Arc<Mutex<HashMap<String, SessionState>>>,
    session_repository: Arc<dyn session_store::SessionRepository>,
    projects: Arc<Mutex<Vec<ProjectEntry>>>,
    project_usage_windows: Arc<Mutex<ProjectUsageWindowCache>>,
    project_session_stream_id: Arc<String>,
    project_session_revision: Arc<AtomicU64>,
    project_session_publication: Arc<StdMutex<ProjectSessionPublication>>,
    project_session_sender: broadcast::Sender<u64>,
    sleep_prevention: Arc<StdMutex<SleepPrevention>>,
    tailscale: Arc<StdMutex<crate::tailscale::TailscaleIntegration>>,
    web_listen: String,
}

impl AppState {
    async fn commit_session_projection_change_owned<T, F>(
        &self,
        session_id: String,
        change: F,
    ) -> Result<SessionCommit<T>>
    where
        T: Send + 'static,
        F: FnOnce(&mut SessionState) -> Result<StagedSessionChange<T>> + Send + 'static,
    {
        let state = self.clone();
        let runtime = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            runtime.block_on(async move {
                let mut sessions = state.sessions.lock().await;
                let session = sessions
                    .get_mut(&session_id)
                    .ok_or_else(|| SessionNotFoundError(session_id.clone()))?;
                state.project_usage_windows.lock().await.invalidate();
                let commit = commit_session_change(
                    state.session_repository.as_ref(),
                    &session_id,
                    session,
                    SessionCommitPolicy::ProjectionOnly,
                    change,
                )?;
                Ok::<_, anyhow::Error>(commit)
            })
        })
        .await
        .context("owned session projection transaction failed")?
    }

    async fn commit_goal_change_owned<T, F>(
        &self,
        goal_id: String,
        change: F,
    ) -> Result<(String, SessionCommit<T>)>
    where
        T: Send + 'static,
        F: FnOnce(&mut SessionState) -> Result<StagedSessionChange<T>> + Send + 'static,
    {
        let state = self.clone();
        let runtime = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            runtime.block_on(async move {
                let projects = state.projects.lock().await.clone();
                let mut sessions = state.sessions.lock().await;
                let (session_id, session) = sessions
                    .iter_mut()
                    .find(|(_, session)| {
                        session
                            .goal
                            .as_ref()
                            .is_some_and(|checkpoint| checkpoint.run.id == goal_id)
                    })
                    .map(|(session_id, session)| (session_id.clone(), session))
                    .ok_or_else(|| GoalSessionNotFound(goal_id.clone()))?;
                state.project_usage_windows.lock().await.invalidate();
                let mut commit = commit_session_change(
                    state.session_repository.as_ref(),
                    &session_id,
                    session,
                    SessionCommitPolicy::Full,
                    change,
                )?;
                let finalization =
                    state.publish_session_commit(&projects, &session_id, session, &commit);
                drop(sessions);
                let dispatch =
                    state.finish_session_commit(&session_id, &mut commit, finalization, Vec::new());
                if dispatch {
                    dispatch_next_session(state.clone());
                }
                Ok::<_, anyhow::Error>((session_id, commit))
            })
        })
        .await
        .context("owned Goal transaction task failed")?
    }

    async fn claim_next_session_owned(&self) -> Result<(Option<(String, AgentRequest)>, bool)> {
        let state = self.clone();
        let runtime = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            runtime.block_on(async move {
                let projects = state.projects.lock().await.clone();
                let mut sessions = state.sessions.lock().await;
                let has_active = sessions.values().any(|session| {
                    session.status == SessionStatus::Running
                        || session.runtime.pending_question.is_some()
                });
                if has_active {
                    let working = sessions
                        .values()
                        .any(|session| session.status == SessionStatus::Running);
                    return Ok((None, working));
                }
                let Some(session_id) = sessions
                    .iter()
                    .filter(|(_, session)| session.status == SessionStatus::Queued)
                    .min_by_key(|(_, session)| session.updated_at_ms)
                    .map(|(session_id, _)| session_id.clone())
                else {
                    return Ok((None, false));
                };
                state.project_usage_windows.lock().await.invalidate();
                let session = sessions
                    .get_mut(&session_id)
                    .expect("queued session selected from sessions map");
                let mut commit = commit_session_change(
                    state.session_repository.as_ref(),
                    &session_id,
                    session,
                    SessionCommitPolicy::Full,
                    |staged| {
                        staged.status = SessionStatus::Running;
                        staged
                            .runtime
                            .accepting_user_messages
                            .store(true, Ordering::SeqCst);
                        staged.updated_at_ms = now_millis();
                        Ok(StagedSessionChange::new(staged.request_template.clone()))
                    },
                )?;
                let finalization =
                    state.publish_session_commit(&projects, &session_id, session, &commit);
                drop(sessions);
                let dispatch =
                    state.finish_session_commit(&session_id, &mut commit, finalization, Vec::new());
                debug_assert!(!dispatch, "a dispatch claim must not recursively dispatch");
                Ok::<_, anyhow::Error>((Some((session_id, commit.value)), true))
            })
        })
        .await
        .context("owned session dispatch task failed")?
    }

    async fn create_session_owned(
        &self,
        session_id: String,
        session: SessionState,
    ) -> Result<SessionStreamSnapshot> {
        let state = self.clone();
        let runtime = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            runtime.block_on(async move {
                let mut sessions = state.sessions.lock().await;
                if sessions.contains_key(&session_id) {
                    bail!("generated session id collision");
                }
                let dispatch = session.status == SessionStatus::Queued;
                persist_exact_session_state(
                    state.session_repository.as_ref(),
                    &session_id,
                    &session,
                )?;
                state.project_usage_windows.lock().await.invalidate();
                sessions.insert(session_id.clone(), session);
                state.publish_project_session_change();
                let snapshot = session_mutation_snapshot_for_session(
                    &session_id,
                    sessions
                        .get(&session_id)
                        .expect("committed session was inserted"),
                );
                drop(sessions);
                if dispatch {
                    dispatch_next_session(state);
                }
                Ok::<SessionStreamSnapshot, anyhow::Error>(snapshot)
            })
        })
        .await
        .context("owned session creation task failed")?
    }

    async fn commit_session_change_owned<T, F>(
        &self,
        session_id: String,
        change: F,
    ) -> Result<SessionCommit<T>>
    where
        T: Send + 'static,
        F: FnOnce(&mut SessionState) -> Result<StagedSessionChange<T>> + Send + 'static,
    {
        self.commit_session_change_owned_after(session_id, change, |_| Vec::new())
            .await
    }

    async fn commit_session_change_owned_after<T, F, A>(
        &self,
        session_id: String,
        change: F,
        after_commit: A,
    ) -> Result<SessionCommit<T>>
    where
        T: Send + 'static,
        F: FnOnce(&mut SessionState) -> Result<StagedSessionChange<T>> + Send + 'static,
        A: FnOnce(&T) -> Vec<String> + Send + 'static,
    {
        let state = self.clone();
        let runtime = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            runtime.block_on(async move {
                // Capture the collection lookup before the session write. The committed terminal
                // transition is built from the exact installed session while the session lock is
                // still held, so later registry/session changes cannot alter its payload.
                let projects = state.projects.lock().await.clone();
                let mut sessions = state.sessions.lock().await;
                let session = sessions
                    .get_mut(&session_id)
                    .ok_or_else(|| SessionNotFoundError(session_id.clone()))?;
                // Requested-window usage is a derived cache. Clear it before the durable commit so
                // the revision broadcast can never expose an old aggregate for the new record.
                // A failed save may cause an unnecessary recomputation but cannot change state.
                state.project_usage_windows.lock().await.invalidate();
                let mut commit = commit_session_change(
                    state.session_repository.as_ref(),
                    &session_id,
                    session,
                    SessionCommitPolicy::Full,
                    change,
                )?;
                let finalization =
                    state.publish_session_commit(&projects, &session_id, session, &commit);
                drop(sessions);
                let warnings = after_commit(&commit.value);
                let dispatch =
                    state.finish_session_commit(&session_id, &mut commit, finalization, warnings);
                if dispatch {
                    dispatch_next_session(state.clone());
                }
                Ok::<SessionCommit<T>, anyhow::Error>(commit)
            })
        })
        .await
        .context("owned session transaction task failed")?
    }

    fn publish_session_commit<T>(
        &self,
        projects: &[ProjectEntry],
        session_id: &str,
        session: &SessionState,
        commit: &SessionCommit<T>,
    ) -> SessionCommitFinalization {
        if let Some(entry_key) = commit.terminal_entry_key.clone() {
            publish_terminal_session_transition(self, projects, session_id, session, entry_key);
        } else if commit.effects.publish_collection {
            self.publish_project_session_change();
        }
        SessionCommitFinalization {
            dispatch: commit.effects.dispatch,
            terminate_environment: commit.effects.terminate_environment,
            publish_finished: commit.watch_finished,
            sender: session.runtime.sender.clone(),
        }
    }

    fn finish_session_commit<T>(
        &self,
        session_id: &str,
        commit: &mut SessionCommit<T>,
        finalization: SessionCommitFinalization,
        mut warnings: Vec<String>,
    ) -> bool {
        if finalization.terminate_environment
            && let Err(error) = crate::session_environment::terminate_global_session(session_id)
        {
            let warning =
                format!("the session was committed, but its environment cleanup failed: {error}");
            tracing::warn!(%error, %session_id, %warning);
            warnings.push(warning);
        }
        if !warnings.is_empty() {
            commit.snapshot.warnings.extend(warnings);
            finalization
                .sender
                .publish_committed(SessionStreamPublication::Snapshot(commit.snapshot.clone()));
        }
        if finalization.publish_finished {
            finalization
                .sender
                .publish_committed(SessionStreamPublication::Finished);
        }
        finalization.dispatch
    }

    async fn seal_session_messages_owned(&self, session_id: String) -> Result<bool> {
        let state = self.clone();
        let runtime = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            runtime.block_on(async move {
                let sessions = state.sessions.lock().await;
                let session = sessions
                    .get(&session_id)
                    .ok_or_else(|| SessionNotFoundError(session_id.clone()))?;
                let pending = session
                    .runtime
                    .pending_user_messages
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session message queue is unavailable"))?;
                if !pending.is_empty() {
                    return Ok(false);
                }
                session
                    .runtime
                    .accepting_user_messages
                    .store(false, Ordering::SeqCst);
                Ok(true)
            })
        })
        .await
        .context("owned session message seal task failed")?
    }

    async fn open_session_messages_owned(&self, session_id: String) -> Result<()> {
        let state = self.clone();
        let runtime = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            runtime.block_on(async move {
                let sessions = state.sessions.lock().await;
                let session = sessions
                    .get(&session_id)
                    .ok_or_else(|| SessionNotFoundError(session_id.clone()))?;
                session
                    .runtime
                    .accepting_user_messages
                    .store(true, Ordering::SeqCst);
                Ok(())
            })
        })
        .await
        .context("owned session message open task failed")?
    }

    async fn delete_session_owned(
        &self,
        id: String,
        usage_window: UsageWindow,
    ) -> Result<DeleteSessionMutationResponse, DeleteSessionError> {
        // The task, rather than the HTTP/RPC caller, owns every durable and published phase.
        let state = self.clone();
        tokio::spawn(async move {
            let snapshot = {
                let projects = state.projects.lock().await;
                let mut sessions = state.sessions.lock().await;
                let Some(session) = sessions.get(&id) else {
                    return Err(DeleteSessionError::NotFound(format!(
                        "session not found: {id}"
                    )));
                };
                if session.status == SessionStatus::Running
                    || session.status == SessionStatus::Queued
                    || session.runtime.pending_question.is_some()
                {
                    return Err(DeleteSessionError::Active(format!(
                        "session is active: {id}"
                    )));
                }
                let persisted_workdir = session
                    .workdir
                    .clone()
                    .or_else(|| session.request_template.workdir.clone());
                if let Some(workdir) = persisted_workdir {
                    let session_id = id.clone();
                    let repository = Arc::clone(&state.session_repository);
                    tokio::task::spawn_blocking(move || {
                        repository.delete(&workdir, &session_id)
                    })
                    .await
                    .map_err(|error| {
                        anyhow::anyhow!("session persistence deletion task failed: {error}")
                    })?
                    .map_err(anyhow::Error::from)?;
                }
                sessions.remove(&id).expect("session exists");
                let mut usage_windows = state.project_usage_windows.lock().await;
                usage_windows.invalidate();
                let revision = state.publish_project_session_change();
                let publication = state
                    .project_session_publication
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                project_session_snapshot_from_locked(
                    &state,
                    &projects,
                    &sessions,
                    &mut usage_windows,
                    &publication,
                    revision,
                    revision,
                    usage_window,
                )
            };
            let cleanup_session_id = id.clone();
            let cleanup_warnings = match tokio::task::spawn_blocking(move || {
                cleanup_deleted_session_resources(&cleanup_session_id)
            })
            .await
            {
                Ok(warnings) => warnings,
                Err(error) => vec![format!("session resource cleanup task failed: {error}")],
            };
            for warning in &cleanup_warnings {
                tracing::warn!(session_id = %id, %warning, "session was deleted with cleanup warnings");
            }
            Ok(DeleteSessionMutationResponse {
                deletion: DeleteSessionResponse {
                    session_id: id,
                    deleted: true,
                    cleanup_warnings,
                },
                snapshot,
            })
        })
        .await
        .map_err(|error| anyhow::anyhow!("session deletion transaction failed: {error}"))?
    }

    async fn commit_project_registry_owned<T, F>(
        &self,
        usage_window: UsageWindow,
        transaction: F,
    ) -> Result<ProjectMutationReceipt<T>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<projects::ProjectRegistryMutation<T>> + Send + 'static,
    {
        let state = self.clone();
        tokio::spawn(async move {
            let mut current_projects = state.projects.lock().await;
            let mutation = tokio::task::spawn_blocking(transaction)
                .await
                .context("project registry mutation task failed")??;
            reconcile_project_registry(&state, &mut current_projects, mutation.projects).await;

            // Capture the request-specific response while the registry transaction still owns the
            // project lock. The response therefore names this commit's collection revision rather
            // than whichever later mutation happens to win a second read.
            let sessions = state.sessions.lock().await;
            let mut usage_windows = state.project_usage_windows.lock().await;
            let publication = state
                .project_session_publication
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let revision = state.project_session_revision.load(Ordering::SeqCst);
            let snapshot = project_session_snapshot_from_locked(
                &state,
                &current_projects,
                &sessions,
                &mut usage_windows,
                &publication,
                revision,
                revision,
                usage_window,
            );
            Ok(ProjectMutationReceipt {
                value: mutation.value,
                snapshot,
            })
        })
        .await
        .context("project registry transaction task failed")?
    }

    fn project_session_change_publisher(&self) -> ProjectSessionChangePublisher {
        ProjectSessionChangePublisher {
            revision: Arc::clone(&self.project_session_revision),
            publication: Arc::clone(&self.project_session_publication),
            sender: self.project_session_sender.clone(),
        }
    }

    fn publish_project_session_update(
        &self,
        terminal_transition: Option<ProjectSessionTerminalTransition>,
    ) -> u64 {
        self.project_session_change_publisher()
            .publish_update(terminal_transition)
    }

    fn publish_project_session_change(&self) -> u64 {
        self.publish_project_session_update(None)
    }

    fn publish_project_session_reconciliation(&self, changed: bool, warnings: Vec<String>) -> u64 {
        self.project_session_change_publisher()
            .publish_reconciliation(changed, warnings)
    }

    fn subscribe_project_session_changes(&self) -> (broadcast::Receiver<u64>, u64) {
        let _publication = self
            .project_session_publication
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (
            self.project_session_sender.subscribe(),
            self.project_session_revision.load(Ordering::SeqCst),
        )
    }

    fn project_session_revision_baseline(&self) -> u64 {
        let _publication = self
            .project_session_publication
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.project_session_revision.load(Ordering::SeqCst)
    }

    fn terminal_transition(
        projects: &[ProjectEntry],
        session_id: &str,
        entry_key: String,
        status: SessionStatus,
        session: &SessionState,
    ) -> ProjectSessionTerminalTransition {
        let (title, handoff_outcome) = session_history_summary(session);
        let project = session.durable.project.as_ref().and_then(|stored| {
            projects
                .iter()
                .find(|project| project.id == stored.id)
                .cloned()
        });
        ProjectSessionTerminalTransition {
            entry_key,
            revision: 0,
            session_id: session_id.to_string(),
            status,
            task: session.task.clone(),
            title,
            handoff_outcome,
            project,
        }
    }

    fn update_sleep_prevention_working(&self, working: bool) {
        self.sleep_prevention
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .set_working(working);
    }

    fn update_sleep_prevention_enabled(&self, enabled: bool) {
        self.sleep_prevention
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .set_enabled(enabled);
    }

    fn sleep_prevention_status(&self) -> SleepPreventionStatus {
        self.sleep_prevention
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .status()
    }
}

// `deno task build:web` refreshes these assets before Cargo embeds them.
#[derive(RustEmbed)]
#[folder = "webui/dist"]
struct WebAssets;

pub async fn run_server(args: ServeArgs, defaults: AgentRequest) -> Result<()> {
    run_server_with_ready(args, defaults, None).await
}

pub async fn run_server_with_ready(
    args: ServeArgs,
    defaults: AgentRequest,
    ready: Option<mpsc::Sender<Result<SocketAddr, String>>>,
) -> Result<()> {
    let mut ready = ready;
    let project_entries = match projects::load_projects() {
        Ok(project_entries) => project_entries,
        Err(err) => {
            notify_ready(&mut ready, Err(err.to_string()));
            return Err(err);
        }
    };
    let restored_sessions = restore_sessions(&project_entries);
    let user_config = match crate::config::UserConfig::load() {
        Ok(config) => config,
        Err(error) => {
            notify_ready(&mut ready, Err(error.to_string()));
            return Err(error);
        }
    };
    if let Err(error) = crate::session_environment::initialize_global_supervisor()
        .context("failed to reconcile session environments at startup")
    {
        notify_ready(&mut ready, Err(format!("{error:#}")));
        return Err(error);
    }
    tokio::spawn(async {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            let cleanup = tokio::task::spawn_blocking(|| -> Result<()> {
                crate::session_environment::global_supervisor().reap_expired()?;
                crate::session_environment::global_supervisor().retry_failed_cleanup()?;
                crate::inference::flashmoe::reap_idle_shared_runtimes()?;
                if let Some(runtime) = crate::container::detect_runtime() {
                    crate::cache_manager::global_cache_manager().gc(
                        runtime.as_ref(),
                        Duration::from_secs(30 * 24 * 60 * 60),
                        50 * 1024 * 1024 * 1024,
                    )?;
                }
                Ok(())
            })
            .await;
            match cleanup {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    eprintln!("failed to clean session environments: {error:#}")
                }
                Err(error) => eprintln!("session environment cleanup task failed: {error}"),
            }
        }
    });
    let (project_session_sender, _) = broadcast::channel(256);
    let state = AppState {
        sessions: Arc::new(Mutex::new(restored_sessions)),
        session_repository: Arc::new(session_store::GitNoteSessionRepository),
        projects: Arc::new(Mutex::new(project_entries)),
        project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
        project_session_stream_id: Arc::new(new_durable_id("project-session-stream")),
        project_session_revision: Arc::new(AtomicU64::new(0)),
        project_session_publication: Arc::new(StdMutex::new(ProjectSessionPublication::default())),
        project_session_sender,
        sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(
            user_config.effective_prevent_sleep_while_working(),
        ))),
        tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
            args.port,
            user_config.effective_tailscale_https_port(),
            user_config.effective_tailscale_enabled(),
        ))),
        web_listen: args.host.clone(),
    };

    let project_watch_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;
            if let Err(error) = reload_projects(&project_watch_state).await {
                tracing::warn!(%error, "failed to refresh the project registry");
            }
        }
    });

    let app = Router::new()
        .route("/api/sessions", post(start_session).get(list_sessions))
        .route(
            "/api/sessions/{id}",
            get(get_session).delete(delete_session),
        )
        .route("/api/sessions/{id}/continue", post(continue_session))
        .route("/api/sessions/{id}/message", post(send_session_message))
        .route("/api/sessions/{id}/resume", post(resume_session))
        .route(
            "/api/sessions/{id}/restart-delivery",
            post(restart_delivery),
        )
        .route(
            "/api/sessions/{id}/retry-task-planning",
            post(retry_task_planning),
        )
        .route(
            "/api/sessions/{id}/run-as-one-build",
            post(run_as_one_build),
        )
        .route("/api/sessions/{id}/cancel", post(cancel_session))
        .route("/api/sessions/{id}/answer", post(answer_question))
        .route("/api/sessions/{id}/goal", post(start_session_goal))
        .route("/api/sessions/{id}/events", get(session_events))
        .route("/api/goals", post(start_goal))
        .route("/api/goals/{id}", get(get_goal))
        .route("/api/goals/{id}/draft", patch(revise_goal_draft))
        .route("/api/goals/{id}/approve-plan", post(approve_goal_plan))
        .route("/api/goals/{id}/pause", post(pause_goal))
        .route("/api/goals/{id}/resume", post(resume_goal))
        .route("/api/goals/{id}/cancel", post(cancel_goal))
        .route("/api/goals/{id}/accept", post(accept_goal))
        .route("/api/goals/{id}/amendments", post(amend_goal))
        .route(
            "/api/goals/{id}/amendments/{amendment_id}/approve",
            post(approve_goal_amendment),
        )
        .route(
            "/api/goals/{id}/amendments/{amendment_id}/discard",
            post(discard_goal_amendment),
        )
        .route("/api/project-sessions", get(list_project_sessions))
        .route("/api/project-sessions/events", get(project_session_events))
        .route("/api/projects", get(list_projects))
        .route(
            "/api/integrations/marketplace",
            get(list_marketplace_integrations),
        )
        .route(
            "/api/integrations/config-schema",
            get(get_integration_config_schema),
        )
        .route(
            "/api/projects/{id}/integrations",
            get(list_project_integrations).post(install_project_integration),
        )
        .route(
            "/api/projects/{id}/integrations/{integration_name}",
            delete(remove_project_integration),
        )
        .route(
            "/api/integrations/lsp",
            get(list_global_lsp_integrations).post(install_global_lsp_integration),
        )
        .route(
            "/api/integrations/lsp/{integration_name}",
            delete(remove_global_lsp_integration),
        )
        .route(
            "/api/projects/{id}/notifications",
            patch(update_project_notifications),
        )
        .route("/api/settings", get(get_settings).patch(update_settings))
        .route(
            "/api/settings/tailscale",
            get(get_tailscale_settings).patch(update_tailscale_settings),
        )
        .route("/api/status", get(status))
        .route("/api/current-user.png", get(crate::user::avatar_png))
        .route("/api/current-user", get(crate::user::user_info))
        .route("/auth/github/callback", get(github_oauth_callback))
        .route("/pb-sw.js", get(service_worker))
        .route("/", get(index))
        .route("/{*path}", get(static_asset))
        .with_state((state.clone(), defaults.clone()));

    if let Err(err) = spawn_unix_rpc_server(args.socket_path.clone(), state.clone(), defaults).await
    {
        notify_ready(&mut ready, Err(err.to_string()));
        return Err(err);
    }

    let addr: SocketAddr = match crate::http_listener::socket_addr(&args.host, args.port) {
        Ok(addr) => addr,
        Err(err) => {
            notify_ready(&mut ready, Err(err.to_string()));
            return Err(err);
        }
    };
    let acquired = match crate::http_listener::acquire(addr).await {
        Ok(acquired) => acquired,
        Err(err) => {
            notify_ready(&mut ready, Err(err.to_string()));
            return Err(err);
        }
    };
    let bound_addr = acquired
        .listener
        .local_addr()
        .context("failed to inspect the pb HTTP listener")?;
    let source = match acquired.source {
        crate::http_listener::ListenerSource::Direct => "direct bind",
        crate::http_listener::ListenerSource::Launchd => "launchd socket activation",
    };
    println!("pb serve listening on http://{bound_addr} ({source})");
    if acquired.wake_advertised {
        println!(
            "pb wake-on-HTTP advertised through Bonjour as {}",
            crate::http_listener::BONJOUR_SERVICE_TYPE
        );
    }
    let tailscale = Arc::clone(&state.tailscale);
    let web_listen = state.web_listen.clone();
    tokio::spawn(async move {
        match tokio::task::spawn_blocking(move || {
            tailscale
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .reconcile(&web_listen)
        })
        .await
        {
            Ok(status) if status.enabled && !status.active => {
                eprintln!(
                    "pb Tailscale access needs attention: {}",
                    status.error.as_deref().unwrap_or("endpoint is not active")
                );
            }
            Err(error) => eprintln!("pb Tailscale reconciliation task failed: {error}"),
            _ => {}
        }
    });
    notify_ready(&mut ready, Ok(bound_addr));
    axum::serve(acquired.listener, app).await?;
    Ok(())
}

fn notify_ready(
    ready: &mut Option<mpsc::Sender<Result<SocketAddr, String>>>,
    result: Result<SocketAddr, String>,
) {
    if let Some(sender) = ready.take() {
        let _ = sender.send(result);
    }
}

async fn list_marketplace_integrations()
-> Result<Json<Vec<MarketplaceIntegration>>, IntegrationApiError> {
    integrations::list_marketplace()
        .await
        .map(Json)
        .map_err(|error| IntegrationApiError::new(StatusCode::BAD_GATEWAY, format!("{error:#}")))
}

#[derive(Debug, Deserialize)]
struct IntegrationSchemaQuery {
    image: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ProjectSessionQuery {
    usage_window_start_ms: u64,
    usage_window_end_ms: u64,
    #[serde(default)]
    last_event_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UsageWindow {
    start_ms: u64,
    end_ms: u64,
}

impl UsageWindow {
    fn current_utc_day() -> Self {
        const DAY_MS: u64 = 86_400_000;
        let now_ms = now_millis();
        let start_ms = now_ms - (now_ms % DAY_MS);
        Self {
            start_ms,
            end_ms: start_ms.saturating_add(DAY_MS),
        }
    }
}

impl ProjectSessionQuery {
    fn usage_window(&self) -> Result<UsageWindow, ApiError> {
        self.usage_window_end_ms
            .checked_sub(self.usage_window_start_ms)
            .filter(|duration| *duration > 0 && *duration <= 172_800_000)
            .ok_or_else(|| {
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_usage_window",
                    "usage window must be a positive interval no longer than 48 hours",
                )
            })?;
        Ok(UsageWindow {
            start_ms: self.usage_window_start_ms,
            end_ms: self.usage_window_end_ms,
        })
    }
}

#[derive(Debug, Serialize)]
struct IntegrationApiErrorBody {
    error: String,
}

struct IntegrationApiError {
    status: StatusCode,
    message: String,
}

impl IntegrationApiError {
    fn new(status: StatusCode, error: impl std::fmt::Display) -> Self {
        Self {
            status,
            message: error.to_string().chars().take(2_000).collect(),
        }
    }
}

impl IntoResponse for IntegrationApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(IntegrationApiErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

#[derive(Debug, Serialize)]
struct ApiErrorBody {
    code: &'static str,
    error: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, error: impl std::fmt::Display) -> Self {
        Self {
            status,
            code,
            message: error.to_string().chars().take(2_000).collect(),
        }
    }
}

impl From<StatusCode> for ApiError {
    fn from(status: StatusCode) -> Self {
        let (code, message) = match status {
            StatusCode::BAD_REQUEST => ("invalid_request", "The request is invalid."),
            StatusCode::NOT_FOUND => ("not_found", "The requested resource was not found."),
            StatusCode::CONFLICT => ("state_conflict", "The resource state has changed."),
            StatusCode::PAYLOAD_TOO_LARGE => ("payload_too_large", "The request is too large."),
            StatusCode::TOO_MANY_REQUESTS => ("queue_full", "The request queue is full."),
            _ => ("internal_error", "pb could not complete the request."),
        };
        Self::new(status, code, message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ApiErrorBody {
                code: self.code,
                error: self.message,
            }),
        )
            .into_response()
    }
}

async fn get_integration_config_schema(
    Query(query): Query<IntegrationSchemaQuery>,
) -> Result<Json<IntegrationConfigSchema>, IntegrationApiError> {
    tokio::task::spawn_blocking(move || integrations::fetch_config_schema(&query.image))
        .await
        .map_err(|error| IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error))?
        .map(Json)
        .map_err(|error| IntegrationApiError::new(StatusCode::BAD_GATEWAY, format!("{error:#}")))
}

async fn list_project_integrations(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<Vec<InstalledIntegration>>, IntegrationApiError> {
    let project_path = registered_project_path(&state, &id).await?;
    tokio::task::spawn_blocking(move || integrations::list_project_installed(&project_path))
        .await
        .map_err(|error| IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error))?
        .map(Json)
        .map_err(|error| {
            IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
        })
}

async fn install_project_integration(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<IntegrationInstallRequest>,
) -> Result<Json<Vec<InstalledIntegration>>, IntegrationApiError> {
    let project_path = registered_project_path(&state, &id).await?;
    tokio::task::spawn_blocking(move || {
        integrations::install_project(&project_path, req)?;
        integrations::list_project_installed(&project_path)
    })
    .await
    .map_err(|error| IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error))?
    .map(Json)
    .map_err(|error| {
        IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
    })
}

async fn remove_project_integration(
    Path((id, integration_name)): Path<(String, String)>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<Vec<InstalledIntegration>>, IntegrationApiError> {
    let project_path = registered_project_path(&state, &id).await?;
    tokio::task::spawn_blocking(move || {
        integrations::remove_project(
            &project_path,
            crate::integrations::IntegrationKind::Mcp,
            &integration_name,
        )?;
        integrations::list_project_installed(&project_path)
    })
    .await
    .map_err(|error| IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error))?
    .map(Json)
    .map_err(|error| {
        IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
    })
}

async fn registered_project_path(
    state: &AppState,
    id: &str,
) -> Result<PathBuf, IntegrationApiError> {
    reload_projects(state).await.map_err(|error| {
        IntegrationApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to reload project registry: {error:#}"),
        )
    })?;
    state
        .projects
        .lock()
        .await
        .iter()
        .find(|project| project.id == id)
        .map(|project| PathBuf::from(&project.path))
        .ok_or_else(|| IntegrationApiError::new(StatusCode::NOT_FOUND, "project not found"))
}

async fn list_global_lsp_integrations()
-> Result<Json<Vec<InstalledIntegration>>, IntegrationApiError> {
    tokio::task::spawn_blocking(integrations::list_global_lsp_installed)
        .await
        .map_err(|error| IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error))?
        .map(Json)
        .map_err(|error| {
            IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
        })
}

async fn install_global_lsp_integration(
    Json(req): Json<IntegrationInstallRequest>,
) -> Result<Json<Vec<InstalledIntegration>>, IntegrationApiError> {
    tokio::task::spawn_blocking(move || {
        integrations::install_global_lsp(req)?;
        integrations::list_global_lsp_installed()
    })
    .await
    .map_err(|error| IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error))?
    .map(Json)
    .map_err(|error| {
        IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
    })
}

async fn remove_global_lsp_integration(
    Path(integration_name): Path<String>,
) -> Result<Json<Vec<InstalledIntegration>>, IntegrationApiError> {
    tokio::task::spawn_blocking(move || {
        integrations::remove_global_lsp(&integration_name)?;
        integrations::list_global_lsp_installed()
    })
    .await
    .map_err(|error| IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error))?
    .map(Json)
    .map_err(|error| {
        IntegrationApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
    })
}

async fn start_session(
    State((state, defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<StartSessionRequest>,
) -> Result<Json<SessionResponse>, ApiError> {
    start_session_inner(state, defaults, req)
        .await
        .map(Json)
        .map_err(session_start_api_error)
}

fn session_start_api_error(error: anyhow::Error) -> ApiError {
    if error.downcast_ref::<SessionPersistenceError>().is_some() {
        tracing::error!(%error, "failed to persist session start");
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "session_persistence_failed",
            error,
        )
    } else {
        ApiError::new(StatusCode::BAD_REQUEST, "session_start_rejected", error)
    }
}

async fn start_session_inner(
    state: AppState,
    defaults: AgentRequest,
    req: StartSessionRequest,
) -> Result<SessionResponse> {
    let session_id = new_session_id();
    let sender = SessionEventSender::new(256);
    let task = req.task.trim().to_string();
    if task.is_empty() {
        bail!("session task must not be empty");
    }
    if req.project_id.is_some() && req.workdir.is_some() {
        bail!("choose project_id or workdir, not both");
    }
    reload_projects(&state).await?;
    let mut registered_projects = project_list_snapshot(&state).await;
    let named_project = req
        .project_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(|id| registered_session_project(&registered_projects, id))
        .transpose()?;

    let mut request = defaults.clone();
    request.task = task.clone();
    if let Some(model) = req.model {
        request.model = model;
    }
    if let Some(model_dir) = req.model_dir {
        request.model_dir = Some(PathBuf::from(model_dir));
    }
    let explicit_workdir = named_project
        .as_ref()
        .map(|(workdir, _)| workdir.as_path())
        .or_else(|| {
            req.workdir
                .as_deref()
                .filter(|workdir| !workdir.trim().is_empty())
                .map(std::path::Path::new)
        });
    if let Some(workdir) = explicit_workdir {
        request.workdir = Some(PathBuf::from(workdir));
        request.repository_less = false;
    } else if let Some(bootstrap) = maybe_bootstrap_project(&state, &task).await? {
        request.workdir = Some(bootstrap);
        request.repository_less = false;
        registered_projects = project_list_snapshot(&state).await;
    } else {
        request.workdir = None;
        request.repository_less = true;
        if !matches!(req.profile, Some(AgentProfile::Build | AgentProfile::Scout)) {
            request.profile = req.profile.unwrap_or(AgentProfile::Ask);
            request.infer_profile = req.profile.is_none();
        }
    }
    let workflow_policy = workflow_policy_for_request(request.workdir.as_deref())?;
    request.intent = Some(req.intent.unwrap_or(workflow_policy.default_intent));
    request.task_planning = crate::agent_core::TaskPlanningPreference::Auto;
    request.task_plan_rejected = None;
    request.task_planning_transcript = None;
    // A new user turn in a restored conversation always uses current workflow policy. The
    // compatibility marker applies only to resuming the exact pre-intent invocation.
    request.legacy_prompt_owned_delivery = false;
    request.workflow_policy = Some(workflow_policy);
    request.workflow_stage = None;
    request.workflow_checkpoint = None;
    request.turn_id = new_turn_id(&session_id);
    if req.proposal_id.is_some() {
        bail!("a new session cannot cite a proposal from an unrelated conversation");
    }
    request.conversation_handoff =
        delivery_handoff_for_turn(request.intent, &request.turn_id, &request.task, None);
    request.branch = if request.repository_less {
        None
    } else {
        req.branch.clone()
    };
    request.session_id = session_id.clone();
    request.max_steps = req.max_steps.unwrap_or(request.max_steps);
    request.max_tokens = req.max_tokens.unwrap_or(request.max_tokens);
    request.ctx_size = req.ctx_size.unwrap_or(request.ctx_size);
    request.threads = req.threads.or(request.threads);
    request.threads_batch = req.threads_batch.or(request.threads_batch);
    request.gpu_layers = req.gpu_layers.unwrap_or(request.gpu_layers);
    request.temperature = req.temperature.unwrap_or(request.temperature);
    if !request.repository_less
        || matches!(req.profile, Some(AgentProfile::Build | AgentProfile::Scout))
    {
        request.profile = req.profile.unwrap_or(request.profile);
        request.infer_profile = req.profile.is_none();
    }
    request.top_k = req.top_k.unwrap_or(request.top_k);
    request.seed = req.seed.unwrap_or(request.seed);
    request.attachments =
        materialize_attachments(&session_id, request.workdir.as_deref(), req.attachments)?;

    let project = named_project
        .map(|(_, project)| project)
        .or_else(|| resolve_session_project(&registered_projects, request.workdir.as_deref()));
    let durable = DurableSessionProjection {
        project,
        ..DurableSessionProjection::default()
    };

    let now = now_millis();
    let usage_records = Arc::new(StdMutex::new(Vec::new()));
    let session = SessionState {
        record: SessionRecord {
            revision: 0,
            task: request.task.clone(),
            title: None,
            branch: request.branch.clone(),
            workdir: request.workdir.clone(),
            durable: durable.clone(),
            request_template: request.clone(),
            status: SessionStatus::Queued,
            metrics: None,
            workflow: None,
            completed_workflows: Vec::new(),
            goal: None,
            completed_goals: Vec::new(),
            multi_task: None,
            completed_multi_tasks: Vec::new(),
            started_at_ms: now,
            updated_at_ms: now,
        },
        runtime: SessionRuntime {
            pending_question: None,
            sender,
            history: Arc::new(StdMutex::new(Vec::new())),
            usage_records: Arc::clone(&usage_records),
            pending_user_messages: Arc::new(StdMutex::new(VecDeque::new())),
            accepting_user_messages: Arc::new(AtomicBool::new(false)),
            pause_token: Arc::new(AtomicBool::new(false)),
            cancel_token: Arc::new(AtomicBool::new(false)),
        },
    };
    state
        .create_session_owned(session_id.clone(), session)
        .await?;

    Ok(SessionResponse { session_id })
}

async fn start_goal(
    State((state, defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<StartGoalRequest>,
) -> Result<Json<GoalResponse>, ApiError> {
    if req.session_id.is_some() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "session_goal_route_required",
            "start a goal in an existing session through that session's goal endpoint",
        ));
    }
    start_goal_inner(state, defaults, req)
        .await
        .map(|result| Json(result.response))
        .map_err(goal_start_api_error)
}

async fn start_session_goal(
    Path(id): Path<String>,
    State((state, defaults)): State<(AppState, AgentRequest)>,
    Json(mut req): Json<StartGoalRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    if req
        .session_id
        .as_deref()
        .is_some_and(|requested| requested != id)
    {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "session_id_mismatch",
            "the goal request names a different session",
        ));
    }
    req.session_id = Some(id);
    let result = start_goal_inner(state, defaults, req)
        .await
        .map_err(goal_start_api_error)?;
    Ok(Json(result.snapshot))
}

async fn start_goal_inner(
    state: AppState,
    defaults: AgentRequest,
    req: StartGoalRequest,
) -> Result<GoalMutationReceipt> {
    let now = now_millis();
    let goal_id = new_durable_id("goal");
    let objective = req.objective.trim().to_string();
    if objective.is_empty() {
        bail!("goal objective must not be empty");
    }

    // Snapshot the registry before taking the session lock. Goal activation never waits for a
    // second application lock while it is changing durable session state.
    if req.project_id.is_some() {
        reload_projects(&state).await?;
    }
    let registered_projects = state.projects.lock().await.clone();
    if let Some(session_id) = req.session_id.clone() {
        let overrides_project = req.project_id.is_some() || req.workdir.is_some();
        let transaction_session_id = session_id.clone();
        let transaction_goal_id = goal_id.clone();
        let criteria = req.criteria;
        let continuation = req.continuation;
        let budget = req.budget;
        let commit = state
            .commit_session_change_owned(session_id.clone(), move |session| {
                if overrides_project {
                    bail!("an existing session already determines the goal project");
                }
                if session.goal.is_some()
                    || session.status == SessionStatus::Running
                    || session.runtime.pending_question.is_some()
                {
                    bail!("session already has active work");
                }
                if !matches!(
                    session.status,
                    SessionStatus::Completed | SessionStatus::Failed | SessionStatus::Paused
                ) {
                    bail!("session is not ready to start a goal");
                }
                let workdir = session
                    .workdir
                    .clone()
                    .context("goal mode requires a registered repository")?;
                ensure_goal_workdir_registered(&registered_projects, &workdir)?;
                let policy = goal_policy_for_request(Some(&workdir))?;
                let run = crate::goal::GoalRun::start(
                    transaction_goal_id,
                    transaction_session_id,
                    objective.clone(),
                    criteria,
                    continuation,
                    budget,
                    policy,
                    workdir.to_string_lossy(),
                    now,
                )?;
                let checkpoint = crate::goal::GoalCheckpoint::new(run)?;
                session.task = objective.clone();
                session.title = Some(objective.clone());
                session.request_template.task = objective;
                session.request_template.intent = Some(crate::workflow::TurnIntent::Discuss);
                session.request_template.goal_context = None;
                session.request_template.workflow_checkpoint = None;
                session.request_template.workflow_stage = None;
                session.durable.pending_goal_proposal = None;
                session.durable.pending_goal_change = None;
                session.goal = Some(checkpoint.clone());
                session.status = SessionStatus::Paused;
                session.updated_at_ms = now;
                publish_goal_started(session, &checkpoint);
                Ok(StagedSessionChange::new(checkpoint))
            })
            .await?;
        let checkpoint = commit.value;
        return Ok(GoalMutationReceipt {
            response: GoalResponse {
                session_id,
                goal_id,
                goal_sha256: checkpoint.sha256,
            },
            snapshot: commit.snapshot,
        });
    }

    let session_id = new_session_id();

    if req.project_id.is_some() && req.workdir.is_some() {
        bail!("choose project_id or workdir, not both");
    }
    let named_project = req
        .project_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(|id| registered_session_project(&registered_projects, id))
        .transpose()?;
    let workdir = named_project
        .as_ref()
        .map(|(workdir, _)| workdir.clone())
        .or_else(|| {
            req.workdir
                .as_deref()
                .filter(|path| !path.trim().is_empty())
                .map(PathBuf::from)
        })
        .context("goal mode requires a registered repository")?;
    ensure_goal_workdir_registered(&registered_projects, &workdir)?;
    let policy = goal_policy_for_request(Some(&workdir))?;
    let run = crate::goal::GoalRun::start(
        goal_id.clone(),
        session_id.clone(),
        objective.clone(),
        req.criteria,
        req.continuation,
        req.budget,
        policy,
        workdir.to_string_lossy(),
        now,
    )?;
    let checkpoint = crate::goal::GoalCheckpoint::new(run)?;
    let sender = SessionEventSender::new(256);
    let history = Arc::new(StdMutex::new(Vec::new()));
    let usage_records = Arc::new(StdMutex::new(Vec::new()));
    let mut request = defaults;
    request.session_id = session_id.clone();
    request.task = objective.clone();
    request.workdir = Some(workdir.clone());
    request.repository_less = false;
    request.intent = Some(crate::workflow::TurnIntent::Discuss);
    request.goal_context = None;
    request.workflow_policy = Some(workflow_policy_for_request(Some(&workdir))?);
    request.workflow_stage = None;
    request.workflow_checkpoint = None;
    request.turn_id = new_turn_id(&session_id);
    request.branch = None;
    request.conversation_handoff = None;
    if let Some(model) = req.model {
        request.model = model;
    }
    let project = named_project
        .map(|(_, project)| project)
        .or_else(|| resolve_session_project(&registered_projects, Some(&workdir)));
    let mut session = SessionState {
        record: SessionRecord {
            revision: 0,
            task: objective.clone(),
            title: Some(objective),
            branch: None,
            workdir: Some(workdir),
            durable: DurableSessionProjection {
                project,
                ..DurableSessionProjection::default()
            },
            request_template: request,
            status: SessionStatus::Paused,
            metrics: None,
            workflow: None,
            completed_workflows: Vec::new(),
            goal: Some(checkpoint.clone()),
            completed_goals: Vec::new(),
            multi_task: None,
            completed_multi_tasks: Vec::new(),
            started_at_ms: now,
            updated_at_ms: now,
        },
        runtime: SessionRuntime {
            pending_question: None,
            sender,
            history,
            usage_records,
            pending_user_messages: Arc::new(StdMutex::new(VecDeque::new())),
            accepting_user_messages: Arc::new(AtomicBool::new(false)),
            pause_token: Arc::new(AtomicBool::new(false)),
            cancel_token: Arc::new(AtomicBool::new(false)),
        },
    };
    publish_goal_started(&mut session, &checkpoint);
    let snapshot = state
        .create_session_owned(session_id.clone(), session)
        .await?;
    Ok(GoalMutationReceipt {
        response: GoalResponse {
            session_id,
            goal_id,
            goal_sha256: checkpoint.sha256,
        },
        snapshot,
    })
}

fn goal_start_api_error(error: anyhow::Error) -> ApiError {
    if error.downcast_ref::<GoalSessionNotFound>().is_some()
        || error.downcast_ref::<SessionNotFoundError>().is_some()
    {
        return ApiError::new(StatusCode::NOT_FOUND, "session_not_found", error);
    }
    if error.downcast_ref::<SessionPersistenceError>().is_some() {
        tracing::error!(%error, "failed to persist goal start");
        return ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "session_persistence_failed",
            error,
        );
    }
    tracing::warn!(%error, "failed to start goal");
    ApiError::new(StatusCode::BAD_REQUEST, "goal_start_rejected", error)
}

fn ensure_goal_workdir_registered(
    projects: &[ProjectEntry],
    workdir: &std::path::Path,
) -> Result<()> {
    let requested = crate::agent_core::find_git_root(workdir)
        .unwrap_or_else(|| workdir.to_path_buf())
        .canonicalize()
        .with_context(|| format!("goal repository does not exist: {}", workdir.display()))?;
    let registered = projects.iter().any(|project| {
        let path = project
            .repository_root
            .as_deref()
            .unwrap_or(project.path.as_str());
        let path = PathBuf::from(path);
        let root = crate::agent_core::find_git_root(&path).unwrap_or(path);
        root.canonicalize().is_ok_and(|root| root == requested)
    });
    if !registered {
        bail!(
            "goal mode requires a registered repository; add {} to pb first",
            requested.display()
        );
    }
    Ok(())
}

fn resolve_session_project(
    projects: &[ProjectEntry],
    workdir: Option<&std::path::Path>,
) -> Option<SessionProject> {
    let workdir = workdir?.canonicalize().ok()?;
    if let Some(project) = projects.iter().find(|project| {
        PathBuf::from(&project.path)
            .canonicalize()
            .is_ok_and(|path| path == workdir)
    }) {
        return Some(SessionProject {
            id: project.id.clone(),
            name: project.name.clone(),
            path: project.path.clone(),
        });
    }
    let workdir_root = crate::agent_core::find_git_root(&workdir)
        .unwrap_or_else(|| workdir.clone())
        .canonicalize()
        .ok()?;
    let mut matching = projects.iter().filter(|project| {
        let root = project
            .repository_root
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(&project.path));
        crate::agent_core::find_git_root(&root)
            .unwrap_or(root)
            .canonicalize()
            .is_ok_and(|root| root == workdir_root)
    });
    let project = matching.next()?;
    if matching.next().is_some() {
        return None;
    }
    Some(SessionProject {
        id: project.id.clone(),
        name: project.name.clone(),
        path: project.path.clone(),
    })
}

fn registered_session_project(
    projects: &[ProjectEntry],
    project_id: &str,
) -> Result<(PathBuf, SessionProject)> {
    let project = projects
        .iter()
        .find(|project| project.id == project_id)
        .with_context(|| format!("registered project not found: {project_id}"))?;
    Ok((
        PathBuf::from(&project.path),
        SessionProject {
            id: project.id.clone(),
            name: project.name.clone(),
            path: project.path.clone(),
        },
    ))
}

fn publish_goal_started(session: &mut SessionState, checkpoint: &crate::goal::GoalCheckpoint) {
    stage_event(
        &session.runtime.history,
        AgentEvent::GoalStarted {
            goal_id: checkpoint.run.id.clone(),
            objective: checkpoint.run.objective.clone(),
            plan_sha256: checkpoint.run.plan_sha256.clone(),
            timestamp_ms: Some(now_millis()),
        },
    );
    stage_event(
        &session.runtime.history,
        AgentEvent::GoalPlanAwaitingApproval {
            goal_id: checkpoint.run.id.clone(),
            plan_sha256: checkpoint.run.plan_sha256.clone(),
            milestones: checkpoint.run.milestones.len(),
            timestamp_ms: Some(now_millis()),
        },
    );
}

async fn get_goal(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<crate::goal::GoalCheckpoint>, ApiError> {
    let sessions = state.sessions.lock().await;
    sessions
        .values()
        .find_map(|session| {
            session
                .goal
                .as_ref()
                .filter(|checkpoint| checkpoint.run.id == id)
                .or_else(|| {
                    session
                        .completed_goals
                        .iter()
                        .find(|checkpoint| checkpoint.run.id == id)
                })
                .cloned()
        })
        .map(Json)
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "goal_not_found",
                format!("goal not found: {id}"),
            )
        })
}

async fn approve_goal_plan(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let expected_sha256 = req.goal_sha256.clone();
    let result = mutate_active_goal(&state, &id, &expected_sha256, move |session, run| {
        let plan_sha256 = req
            .plan_sha256
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("plan_sha256 is required"))?;
        run.approve_plan(plan_sha256, now_millis())?;
        configure_goal_milestone_request(session, run)?;
        session.status = SessionStatus::Queued;
        stage_event(
            &session.runtime.history,
            AgentEvent::GoalPlanApproved {
                goal_id: run.id.clone(),
                plan_sha256: run.plan_sha256.clone(),
                timestamp_ms: Some(now_millis()),
            },
        );
        publish_current_goal_milestone(session, run);
        Ok(())
    })
    .await?;
    Ok(Json(result.snapshot))
}

async fn revise_goal_draft(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalAmendmentRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let expected_sha256 = req.goal_sha256.clone();
    let result = mutate_active_goal(&state, &id, &expected_sha256, move |session, run| {
        validate_goal_task_amendment(session, &req.objective, &req.criteria, req.budget)?;
        run.revise_initial_plan(
            req.objective.clone(),
            req.criteria.clone(),
            req.continuation,
            req.budget,
            now_millis(),
        )?;
        session.task = run.objective.clone();
        session.title = Some(run.objective.clone());
        session.request_template.task = run.objective.clone();
        stage_event(
            &session.runtime.history,
            AgentEvent::GoalPlanAwaitingApproval {
                goal_id: run.id.clone(),
                plan_sha256: run.plan_sha256.clone(),
                milestones: run.milestones.len(),
                timestamp_ms: Some(now_millis()),
            },
        );
        Ok(())
    })
    .await?;
    Ok(Json(result.snapshot))
}

async fn pause_goal(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    pause_goal_inner(&state, &id, &req)
        .await
        .map(|result| Json(result.snapshot))
}

async fn pause_goal_inner(
    state: &AppState,
    id: &str,
    req: &GoalDigestRequest,
) -> Result<GoalMutationReceipt, ApiError> {
    mutate_active_goal(state, id, &req.goal_sha256, |session, run| {
        let paused = run.request_pause(now_millis())?;
        session.runtime.pause_token.store(true, Ordering::SeqCst);
        stage_event(
            &session.runtime.history,
            AgentEvent::GoalPauseRequested {
                goal_id: run.id.clone(),
                timestamp_ms: Some(now_millis()),
            },
        );
        if paused {
            session.status = SessionStatus::Paused;
            stage_event(
                &session.runtime.history,
                AgentEvent::GoalPaused {
                    goal_id: run.id.clone(),
                    timestamp_ms: Some(now_millis()),
                },
            );
        }
        Ok(())
    })
    .await
}

async fn resume_goal(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    resume_goal_inner(&state, &id, &req)
        .await
        .map(|result| Json(result.snapshot))
}

async fn resume_goal_inner(
    state: &AppState,
    id: &str,
    req: &GoalDigestRequest,
) -> Result<GoalMutationReceipt, ApiError> {
    let result = mutate_active_goal(state, id, &req.goal_sha256, |session, run| {
        run.resume(now_millis())?;
        session.durable.pending_goal_change = None;
        session.runtime.pause_token.store(false, Ordering::SeqCst);
        configure_goal_milestone_request(session, run)?;
        session.status = SessionStatus::Queued;
        stage_event(
            &session.runtime.history,
            AgentEvent::GoalResumed {
                goal_id: run.id.clone(),
                timestamp_ms: Some(now_millis()),
            },
        );
        publish_current_goal_milestone(session, run);
        Ok(())
    })
    .await?;
    Ok(result)
}

async fn cancel_goal(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    cancel_goal_inner(&state, &id, &req)
        .await
        .map(|result| Json(result.snapshot))
}

async fn cancel_goal_inner(
    state: &AppState,
    id: &str,
    req: &GoalDigestRequest,
) -> Result<GoalMutationReceipt, ApiError> {
    let goal_id = id.to_string();
    let expected_sha256 = req.goal_sha256.clone();
    let requested_goal_id = goal_id.clone();
    let (session_id, commit) = state
        .commit_goal_change_owned(goal_id.clone(), move |session| {
            let checkpoint = session
                .goal
                .as_ref()
                .ok_or_else(|| GoalSessionNotFound(requested_goal_id.clone()))?;
            if checkpoint.sha256 != expected_sha256 {
                anyhow::bail!(GoalRevisionConflict);
            }
            let goal_sha256 = checkpoint.sha256.clone();
            if session.status == SessionStatus::Running {
                session.runtime.pause_token.store(false, Ordering::SeqCst);
                session.runtime.cancel_token.store(true, Ordering::SeqCst);
                session.updated_at_ms = now_millis();
                stage_event(
                                        &session.runtime.history,
                    AgentEvent::Correction {
                        kind: crate::events::CorrectionKind::Lifecycle,
                        message: "Cancellation requested. Completed goal milestones and repository evidence will be preserved."
                            .to_string(),
                        summary: "Cancellation requested".to_string(),
                        actor: crate::events::TeamActor::workflow_steward(),
                        assisting_profile: Some(session.request_template.profile),
                        nesting_depth: None,
                        timestamp_ms: Some(now_millis()),
                    },
                );
                return Ok(StagedSessionChange::new(goal_sha256));
            }
            let mut run = session
                .goal
                .take()
                .ok_or_else(|| GoalSessionNotFound(requested_goal_id.clone()))?
                .run;
            run.cancel(now_millis());
            session.durable.pending_goal_change = None;
            let checkpoint = crate::goal::GoalCheckpoint::new(run)?;
            stage_event(
                                &session.runtime.history,
                AgentEvent::GoalCancelled {
                    goal_id: checkpoint.run.id.clone(),
                    checkpoint_sha256: checkpoint.sha256.clone(),
                    timestamp_ms: Some(now_millis()),
                },
            );
            session.completed_goals.push(checkpoint.clone());
            session.status = fold_terminal_goal_task(session, &checkpoint)?;
            session.updated_at_ms = now_millis();
            let change = StagedSessionChange::new(checkpoint.sha256);
            Ok(change)
        })
        .await
        .map_err(goal_mutation_api_error)?;
    Ok(GoalMutationReceipt {
        response: GoalResponse {
            session_id,
            goal_id,
            goal_sha256: commit.value,
        },
        snapshot: commit.snapshot,
    })
}

async fn accept_goal(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    accept_goal_inner(&state, &id, &req)
        .await
        .map(|result| Json(result.snapshot))
}

async fn accept_goal_inner(
    state: &AppState,
    id: &str,
    req: &GoalDigestRequest,
) -> Result<GoalMutationReceipt, ApiError> {
    let goal_id = id.to_string();
    let expected_sha256 = req.goal_sha256.clone();
    let requested_goal_id = goal_id.clone();
    let (session_id, commit) = state
        .commit_goal_change_owned(goal_id.clone(), move |session| {
            let mut checkpoint = session
                .goal
                .take()
                .ok_or_else(|| GoalSessionNotFound(requested_goal_id.clone()))?;
            if checkpoint.sha256 != expected_sha256 {
                anyhow::bail!(GoalRevisionConflict);
            }
            checkpoint
                .run
                .accept(&expected_sha256, &checkpoint.sha256, now_millis())
                .map_err(GoalMutationRejected)?;
            session.durable.pending_goal_change = None;
            checkpoint =
                crate::goal::GoalCheckpoint::new(checkpoint.run).map_err(GoalInvariantError)?;
            stage_event(
                &session.runtime.history,
                AgentEvent::GoalCompleted {
                    goal_id: checkpoint.run.id.clone(),
                    outcome: crate::goal::GoalOutcome::Complete,
                    completion_basis: crate::goal::GoalCompletionBasis::UserAccepted,
                    checkpoint_sha256: checkpoint.sha256.clone(),
                    timestamp_ms: Some(now_millis()),
                },
            );
            session.completed_goals.push(checkpoint.clone());
            session.status = fold_terminal_goal_task(session, &checkpoint)?;
            session.updated_at_ms = now_millis();
            let change = StagedSessionChange::new(checkpoint.sha256);
            Ok(change)
        })
        .await
        .map_err(goal_mutation_api_error)?;
    Ok(GoalMutationReceipt {
        response: GoalResponse {
            session_id,
            goal_id,
            goal_sha256: commit.value,
        },
        snapshot: commit.snapshot,
    })
}

async fn amend_goal(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalAmendmentRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let expected_sha256 = req.goal_sha256.clone();
    let result = mutate_active_goal(&state, &id, &expected_sha256, move |session, run| {
        validate_goal_task_amendment(session, &req.objective, &req.criteria, req.budget)?;
        let amendment_id = new_durable_id("amendment");
        run.propose_amendment(
            amendment_id.clone(),
            req.goal_sha256.clone(),
            req.objective.clone(),
            req.criteria.clone(),
            req.continuation,
            req.budget,
            now_millis(),
        )?;
        let replacement_plan_sha256 = run
            .pending_amendment
            .as_ref()
            .map(|amendment| amendment.replacement_plan_sha256.clone())
            .unwrap_or_default();
        stage_event(
            &session.runtime.history,
            AgentEvent::GoalAmendmentRequested {
                goal_id: run.id.clone(),
                amendment_id,
                replacement_plan_sha256,
                timestamp_ms: Some(now_millis()),
            },
        );
        session.durable.pending_goal_change = None;
        session.status = SessionStatus::Paused;
        Ok(())
    })
    .await?;
    Ok(Json(result.snapshot))
}

async fn approve_goal_amendment(
    Path((id, amendment_id)): Path<(String, String)>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let expected_sha256 = req.goal_sha256.clone();
    let result = mutate_active_goal(&state, &id, &expected_sha256, move |session, run| {
        let pending = run
            .pending_amendment
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("goal has no pending amendment"))?;
        if pending.id != amendment_id {
            bail!("amendment id is stale");
        }
        validate_goal_task_amendment(
            session,
            &pending.objective,
            &pending.criteria,
            Some(pending.budget),
        )?;
        let plan_sha256 = req
            .plan_sha256
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("plan_sha256 is required"))?;
        run.approve_amendment(plan_sha256, now_millis())?;
        configure_goal_milestone_request(session, run)?;
        session.status = SessionStatus::Queued;
        stage_event(
            &session.runtime.history,
            AgentEvent::GoalAmendmentResolved {
                goal_id: run.id.clone(),
                amendment_id: amendment_id.clone(),
                accepted: true,
                timestamp_ms: Some(now_millis()),
            },
        );
        publish_current_goal_milestone(session, run);
        Ok(())
    })
    .await?;
    Ok(Json(result.snapshot))
}

async fn discard_goal_amendment(
    Path((id, amendment_id)): Path<(String, String)>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let expected_sha256 = req.goal_sha256.clone();
    let result = mutate_active_goal(&state, &id, &expected_sha256, move |session, run| {
        if run
            .pending_amendment
            .as_ref()
            .is_none_or(|pending| pending.id != amendment_id)
        {
            bail!("amendment id is stale");
        }
        run.discard_amendment(now_millis())?;
        session.status = SessionStatus::Paused;
        stage_event(
            &session.runtime.history,
            AgentEvent::GoalAmendmentResolved {
                goal_id: run.id.clone(),
                amendment_id: amendment_id.clone(),
                accepted: false,
                timestamp_ms: Some(now_millis()),
            },
        );
        Ok(())
    })
    .await?;
    Ok(Json(result.snapshot))
}

fn stage_session_state(session: &SessionState) -> Result<(SessionState, u64)> {
    let history = session
        .runtime
        .history
        .lock()
        .map_err(|_| anyhow::anyhow!("session history is unavailable"))?
        .clone();
    let base_revision = history
        .last()
        .map(|envelope| envelope.transcript.sequence)
        .unwrap_or(0);
    let pending_user_messages = session
        .runtime
        .pending_user_messages
        .lock()
        .map_err(|_| anyhow::anyhow!("session message queue is unavailable"))?
        .clone();
    let usage_records = session
        .runtime
        .usage_records
        .lock()
        .map_err(|_| anyhow::anyhow!("session usage history is unavailable"))?
        .clone();
    Ok((
        SessionState {
            record: session.record.clone(),
            runtime: SessionRuntime {
                pending_question: session.runtime.pending_question.clone(),
                sender: SessionEventSender::new(16),
                history: Arc::new(StdMutex::new(history)),
                usage_records: Arc::new(StdMutex::new(usage_records)),
                pending_user_messages: Arc::new(StdMutex::new(pending_user_messages)),
                accepting_user_messages: Arc::new(AtomicBool::new(
                    session
                        .runtime
                        .accepting_user_messages
                        .load(Ordering::SeqCst),
                )),
                pause_token: Arc::new(AtomicBool::new(
                    session.runtime.pause_token.load(Ordering::SeqCst),
                )),
                cancel_token: Arc::new(AtomicBool::new(
                    session.runtime.cancel_token.load(Ordering::SeqCst),
                )),
            },
        },
        base_revision,
    ))
}

fn commit_session_change<T>(
    repository: &dyn session_store::SessionRepository,
    session_id: &str,
    session: &mut SessionState,
    policy: SessionCommitPolicy,
    change: impl FnOnce(&mut SessionState) -> Result<StagedSessionChange<T>>,
) -> Result<SessionCommit<T>> {
    let lifecycle_before = session.status;
    let watch_active_before = session_watch_active(session);
    let was_terminal = matches!(
        session.status,
        SessionStatus::Completed | SessionStatus::Failed
    );
    let collection_row_before = (policy == SessionCommitPolicy::Full)
        .then(|| serde_json::to_vec(&session_list_item(session_id, session)))
        .transpose()
        .context("failed to capture the session collection row before mutation")?;
    let (mut staged, base_revision) = stage_session_state(session)?;
    let mut change = change(&mut staged)?;
    let usage_changed = {
        let live_usage = session
            .runtime
            .usage_records
            .lock()
            .map_err(|_| anyhow::anyhow!("session usage history is unavailable"))?;
        let staged_usage = staged
            .runtime
            .usage_records
            .lock()
            .map_err(|_| anyhow::anyhow!("staged session usage history is unavailable"))?;
        *live_usage != *staged_usage
    };
    change.effects.publish_collection |= usage_changed;
    let lifecycle_changed = lifecycle_before != staged.status;
    let became_terminal = !was_terminal
        && matches!(
            staged.status,
            SessionStatus::Completed | SessionStatus::Failed
        );
    change.effects.dispatch |=
        lifecycle_before != SessionStatus::Queued && staged.status == SessionStatus::Queued;
    let watch_finished = watch_active_before && !session_watch_active(&staged);
    if policy == SessionCommitPolicy::ProjectionOnly
        && (lifecycle_changed || watch_finished || !change.effects.is_empty())
    {
        bail!("projection-only session changes cannot request coordinator effects");
    }
    let lifecycle_entry_key = lifecycle_changed.then(|| stage_session_state_changed(&staged));
    let terminal_entry_key = became_terminal.then(|| {
        lifecycle_entry_key
            .clone()
            .expect("a terminal transition must change lifecycle state")
    });
    if let Some(before) = collection_row_before {
        let after = serde_json::to_vec(&session_list_item(session_id, &staged))
            .context("failed to capture the session collection row after mutation")?;
        change.effects.publish_collection |= before != after;
    }
    staged.revision = session.revision.saturating_add(1);
    change.effects.publish_collection |= commit_staged_session_state(
        repository,
        session_id,
        session,
        staged,
        base_revision,
        policy,
        terminal_entry_key.as_deref(),
    )?;
    Ok(SessionCommit {
        value: change.value,
        snapshot: session_mutation_snapshot_for_session(session_id, session),
        terminal_entry_key,
        watch_finished,
        effects: change.effects,
    })
}

fn commit_staged_session_state(
    repository: &dyn session_store::SessionRepository,
    session_id: &str,
    session: &mut SessionState,
    staged: SessionState,
    base_revision: u64,
    policy: SessionCommitPolicy,
    terminal_entry_key: Option<&str>,
) -> Result<bool> {
    let staged_events = staged
        .runtime
        .history
        .lock()
        .map_err(|_| anyhow::anyhow!("staged session history is unavailable"))?
        .iter()
        .filter(|envelope| envelope.transcript.sequence > base_revision)
        .cloned()
        .collect::<Vec<_>>();
    if policy == SessionCommitPolicy::ProjectionOnly
        && staged_events
            .iter()
            .any(EventEnvelope::affects_project_session_snapshot)
    {
        bail!("projection-only session changes cannot publish collection-affecting events");
    }
    commit_staged_session_state_with_terminal(
        repository,
        session_id,
        session,
        staged,
        staged_events,
        terminal_entry_key,
    )
}

fn commit_staged_session_state_with_terminal(
    repository: &dyn session_store::SessionRepository,
    session_id: &str,
    session: &mut SessionState,
    mut staged: SessionState,
    staged_events: Vec<EventEnvelope>,
    terminal_entry_key: Option<&str>,
) -> Result<bool> {
    if terminal_entry_key.is_some_and(|entry_key| {
        staged_events
            .iter()
            .filter(|envelope| envelope.transcript.entry_key == entry_key)
            .count()
            != 1
    }) {
        bail!("staged terminal event is missing or duplicated");
    }

    // Serialize snapshot capture with every other Git-note write so an older autonomous snapshot
    // cannot overwrite this control after it commits.
    let _persistence = SESSION_PERSISTENCE_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("session persistence lock is unavailable"))?;

    // Freeze live history while rebasing the staged envelopes. Producers that do not need the
    // session lock may still append transcript-only events while a control is being validated; the
    // durable projection and the live publication must include those events in the same order.
    let mut live_history = session
        .runtime
        .history
        .lock()
        .map_err(|_| anyhow::anyhow!("session history is unavailable"))?;
    let mut live_pending_user_messages = session
        .runtime
        .pending_user_messages
        .lock()
        .map_err(|_| anyhow::anyhow!("session message queue is unavailable"))?;
    let mut live_usage_records = session
        .runtime
        .usage_records
        .lock()
        .map_err(|_| anyhow::anyhow!("session usage history is unavailable"))?;
    let staged_pending_user_messages = staged
        .runtime
        .pending_user_messages
        .lock()
        .map_err(|_| anyhow::anyhow!("staged session message queue is unavailable"))?
        .clone();
    let staged_usage_records = staged
        .runtime
        .usage_records
        .lock()
        .map_err(|_| anyhow::anyhow!("staged session usage history is unavailable"))?
        .clone();
    staged.metrics = combined_metrics(&staged_usage_records);
    let mut committed_history = live_history.clone();
    let mut committed_events = Vec::with_capacity(staged_events.len());
    for mut envelope in staged_events {
        if committed_history
            .iter()
            .any(|existing| existing.transcript.entry_key == envelope.transcript.entry_key)
        {
            bail!("staged event entry key already exists in live history");
        }
        let sequence = committed_history
            .last()
            .map_or(1, |entry| entry.transcript.sequence.saturating_add(1));
        envelope.assign_sequence(sequence);
        envelope.refresh_projections(&committed_history);
        committed_history.push(envelope.clone());
        if committed_history.len() > MAX_HISTORY_EVENTS {
            committed_history = session_store::trim_event_history(
                std::mem::take(&mut committed_history),
                MAX_HISTORY_EVENTS,
            );
        }
        committed_events.push(envelope);
    }

    let persisted = exact_persisted_session(
        session_id,
        &staged,
        committed_history.clone(),
        staged_usage_records.clone(),
        staged_pending_user_messages.iter().cloned().collect(),
    );
    repository
        .save(&persisted)
        .map_err(|error| anyhow::Error::new(SessionPersistenceError(error)))?;

    session.record = staged.record;
    session.runtime.pending_question = staged.runtime.pending_question;
    *live_pending_user_messages = staged_pending_user_messages;
    *live_usage_records = staged_usage_records;
    session.runtime.accepting_user_messages.store(
        staged
            .runtime
            .accepting_user_messages
            .load(Ordering::SeqCst),
        Ordering::SeqCst,
    );
    session.runtime.pause_token.store(
        staged.runtime.pause_token.load(Ordering::SeqCst),
        Ordering::SeqCst,
    );
    session.runtime.cancel_token.store(
        staged.runtime.cancel_token.load(Ordering::SeqCst),
        Ordering::SeqCst,
    );
    *live_history = committed_history;
    drop(live_pending_user_messages);
    drop(live_usage_records);
    let publish_collection = terminal_entry_key.is_none()
        && committed_events
            .iter()
            .any(EventEnvelope::affects_project_session_snapshot);
    for envelope in committed_events {
        session
            .runtime
            .sender
            .publish_committed(SessionStreamPublication::Event(envelope));
    }
    session
        .runtime
        .sender
        .publish_committed(SessionStreamPublication::Snapshot(SessionStreamSnapshot {
            session: session_details_from_history(session_id, session, &live_history),
            reset_history: false,
            warnings: Vec::new(),
        }));
    Ok(publish_collection)
}

async fn mutate_active_goal(
    state: &AppState,
    goal_id: &str,
    expected_sha256: &str,
    mutate: impl FnOnce(&mut SessionState, &mut crate::goal::GoalRun) -> Result<()> + Send + 'static,
) -> Result<GoalMutationReceipt, ApiError> {
    let requested_goal_id = goal_id.to_string();
    let expected_sha256 = expected_sha256.to_string();
    let (session_id, commit) = state
        .commit_goal_change_owned(requested_goal_id.clone(), move |session| {
            let mut checkpoint = session
                .goal
                .take()
                .ok_or_else(|| GoalSessionNotFound(requested_goal_id.clone()))?;
            if checkpoint.sha256 != expected_sha256 {
                anyhow::bail!(GoalRevisionConflict);
            }
            mutate(session, &mut checkpoint.run).map_err(GoalMutationRejected)?;
            checkpoint = crate::goal::GoalCheckpoint::new(checkpoint.run)?;
            session.updated_at_ms = now_millis();
            session.goal = Some(checkpoint.clone());
            sync_multi_task_goal_checkpoint(session).map_err(GoalInvariantError)?;
            let change = StagedSessionChange::new(checkpoint);
            Ok(change)
        })
        .await
        .map_err(goal_mutation_api_error)?;
    let checkpoint = commit.value;
    Ok(GoalMutationReceipt {
        response: GoalResponse {
            session_id: session_id.clone(),
            goal_id: goal_id.to_string(),
            goal_sha256: checkpoint.sha256,
        },
        snapshot: commit.snapshot,
    })
}

#[derive(Debug, thiserror::Error)]
#[error("the goal changed; refresh before retrying")]
struct GoalRevisionConflict;

#[derive(Debug, thiserror::Error)]
#[error("goal mutation rejected: {0:#}")]
struct GoalMutationRejected(#[source] anyhow::Error);

#[derive(Debug, thiserror::Error)]
#[error("Goal invariant failed: {0:#}")]
struct GoalInvariantError(#[source] anyhow::Error);

fn goal_mutation_api_error(error: anyhow::Error) -> ApiError {
    if error.downcast_ref::<GoalSessionNotFound>().is_some() {
        ApiError::new(StatusCode::NOT_FOUND, "goal_not_found", error)
    } else if error.downcast_ref::<GoalRevisionConflict>().is_some() {
        ApiError::new(StatusCode::CONFLICT, "goal_revision_conflict", error)
    } else if error.downcast_ref::<SessionPersistenceError>().is_some() {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "session_persistence_failed",
            error,
        )
    } else if error.downcast_ref::<GoalInvariantError>().is_some() {
        tracing::error!(%error, "Goal invariant failed before commit");
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "goal_invariant_failed",
            error,
        )
    } else if error.downcast_ref::<GoalMutationRejected>().is_some() {
        tracing::warn!(%error, "goal mutation rejected");
        ApiError::new(StatusCode::CONFLICT, "goal_mutation_rejected", error)
    } else {
        tracing::error!(%error, "unexpected Goal transaction failure");
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "goal_transaction_failed",
            error,
        )
    }
}

fn sync_multi_task_goal_checkpoint(session: &mut SessionState) -> Result<()> {
    let (Some(parent), Some(goal)) = (session.multi_task.clone(), session.goal.clone()) else {
        return Ok(());
    };
    if parent.run.stage != crate::task_queue::MultiTaskStage::RunningTask {
        return Ok(());
    }
    let mut run = parent.run;
    let task_id = run
        .active_task_id
        .clone()
        .context("Goal Task parent has no active Task")?;
    let repository = crate::task_queue::TaskRepositoryState::capture(std::path::Path::new(
        &run.authority.workdir,
    ))?;
    let now = now_millis().max(run.updated_at_ms);
    let contract_matches = run
        .active_task()
        .and_then(|task| task.request.as_ref())
        .and_then(|request| request.goal_contract.as_ref())
        .is_some_and(|contract| goal_matches_task_contract(&goal, contract));
    let event = if contract_matches {
        crate::task_queue::MultiTaskEvent::ChildCheckpointed {
            task_id,
            child: crate::task_queue::TaskChildCheckpoint::Goal(goal),
            repository,
            now_ms: now,
        }
    } else {
        crate::task_queue::MultiTaskEvent::GoalContractRevised {
            task_id,
            child: goal,
            repository,
            now_ms: now,
        }
    };
    run.apply(event)?;
    let checkpoint = crate::task_queue::MultiTaskCheckpoint::new(run)?;
    session.multi_task = Some(checkpoint.clone());
    publish_multi_task_changed(session, &checkpoint);
    Ok(())
}

fn goal_matches_task_contract(
    goal: &crate::goal::GoalCheckpoint,
    contract: &crate::task_queue::TaskGoalContract,
) -> bool {
    goal.run.objective == contract.objective
        && goal.run.continuation == contract.continuation
        && goal.run.criteria.len() == contract.criteria.len()
        && goal
            .run
            .criteria
            .iter()
            .zip(&contract.criteria)
            .all(|(actual, expected)| {
                actual.text.trim() == expected.text.trim() && actual.verifier == expected.verifier
            })
}

fn validate_goal_task_amendment(
    session: &SessionState,
    objective: &str,
    criteria: &[crate::goal::GoalCriterionInput],
    budget: Option<crate::goal::GoalBudget>,
) -> Result<()> {
    let Some(request) = session
        .multi_task
        .as_ref()
        .and_then(|checkpoint| checkpoint.run.active_task())
        .and_then(|task| task.request.as_ref())
    else {
        return Ok(());
    };
    if request.kind != crate::task_queue::TaskKind::Goal {
        bail!("active Task is not a Goal Task");
    }
    let accepted = request
        .goal_contract
        .as_ref()
        .context("Goal Task request has no accepted contract")?;
    if objective.trim() != accepted.objective {
        bail!("Goal Task amendments cannot change the accepted Task objective");
    }
    if accepted.criteria.iter().any(|expected| {
        !criteria.iter().any(|actual| {
            actual.text.trim() == expected.text.trim() && actual.verifier == expected.verifier
        })
    }) {
        bail!("Goal Task amendments cannot remove an accepted criterion");
    }
    if criteria.len() > request.budget.max_workflows {
        bail!("Goal Task amendment exceeds its milestone allowance");
    }
    if let Some(budget) = budget
        && (budget.max_milestones > request.budget.max_workflows
            || budget.max_workflows > request.budget.max_workflows
            || budget.total_model_invocations > request.budget.total_model_invocations
            || budget.total_generated_tokens > request.budget.total_generated_tokens
            || budget.wall_time_minutes > request.budget.wall_time_minutes)
    {
        bail!("Goal Task amendment exceeds its parent Task budget");
    }
    Ok(())
}

fn fold_terminal_goal_task(
    session: &mut SessionState,
    goal: &crate::goal::GoalCheckpoint,
) -> Result<SessionStatus> {
    let Some(checkpoint) = session.multi_task.clone() else {
        return match goal.run.stage {
            crate::goal::GoalStage::Completed | crate::goal::GoalStage::Cancelled => {
                Ok(SessionStatus::Completed)
            }
            crate::goal::GoalStage::Failed => Ok(SessionStatus::Failed),
            stage => bail!("non-terminal standalone Goal reached terminal folding at {stage:?}"),
        };
    };
    let mut parent = checkpoint.run;
    let task_id = parent
        .active_task_id
        .clone()
        .context("terminal Goal Task parent has no active Task")?;
    let repository = crate::task_queue::TaskRepositoryState::capture(std::path::Path::new(
        &parent.authority.workdir,
    ))?;
    let mut now = now_millis().max(parent.updated_at_ms);
    parent.apply(crate::task_queue::MultiTaskEvent::ChildCheckpointed {
        task_id: task_id.clone(),
        child: crate::task_queue::TaskChildCheckpoint::Goal(goal.clone()),
        repository: repository.clone(),
        now_ms: now,
    })?;
    match goal.run.stage {
        crate::goal::GoalStage::Completed => {
            let request = parent
                .active_task()
                .and_then(|task| task.request.as_ref())
                .context("terminal Goal Task lost its request")?;
            let result = crate::task_queue::goal_task_result(request, goal, repository.clone())?;
            now = now_millis().max(parent.updated_at_ms);
            parent.apply(crate::task_queue::MultiTaskEvent::TaskDelivered {
                task_id,
                result,
                repository: repository.clone(),
                now_ms: now,
            })?;
            now = now_millis().max(parent.updated_at_ms);
            parent.apply(crate::task_queue::MultiTaskEvent::EvaluationCompleted {
                repository,
                now_ms: now,
            })?;
            let checkpoint = crate::task_queue::MultiTaskCheckpoint::new(parent)?;
            match checkpoint.run.stage {
                crate::task_queue::MultiTaskStage::Ready => {
                    archive_multi_task(session, checkpoint);
                    Ok(SessionStatus::Completed)
                }
                crate::task_queue::MultiTaskStage::RunningTask => {
                    dispatch_multi_task_active(session, &checkpoint)
                }
                stage => bail!("terminal Goal Task evaluation reached unexpected stage {stage:?}"),
            }
        }
        crate::goal::GoalStage::Cancelled => {
            now = now_millis().max(parent.updated_at_ms);
            parent.apply(crate::task_queue::MultiTaskEvent::Cancelled {
                reason: "Goal Task was cancelled; completed Task commits were preserved"
                    .to_string(),
                now_ms: now,
            })?;
            archive_multi_task(
                session,
                crate::task_queue::MultiTaskCheckpoint::new(parent)?,
            );
            Ok(SessionStatus::Completed)
        }
        crate::goal::GoalStage::Failed => {
            let (_, reason) = crate::task_queue::goal_task_stop(goal)?
                .context("failed Goal Task has no stop disposition")?;
            now = now_millis().max(parent.updated_at_ms);
            parent.apply(crate::task_queue::MultiTaskEvent::TaskStopped {
                task_id,
                disposition: crate::task_queue::TaskStopDisposition::Failed,
                reason,
                now_ms: now,
            })?;
            archive_multi_task(
                session,
                crate::task_queue::MultiTaskCheckpoint::new(parent)?,
            );
            Ok(SessionStatus::Failed)
        }
        stage => bail!("Goal Task checkpoint is not terminal: {stage:?}"),
    }
}

fn configure_goal_milestone_request(
    session: &mut SessionState,
    run: &crate::goal::GoalRun,
) -> Result<()> {
    let milestone = run
        .current_milestone()
        .context("goal has no milestone ready to run")?;
    let mut request = session.request_template.clone();
    request.task = format!(
        "Goal: {}\n\nCurrent milestone: {}\n\n{}",
        run.objective, milestone.title, milestone.description
    );
    request.intent = Some(crate::workflow::TurnIntent::Deliver);
    request.workflow_policy = Some(if let Some(checkpoint) = milestone.workflow.as_ref() {
        checkpoint.run.policy.clone()
    } else {
        let base = session
            .multi_task
            .as_ref()
            .map(|checkpoint| checkpoint.run.authority.workflow_policy.clone())
            .map_or_else(
                || workflow_policy_for_request(session.workdir.as_deref()),
                Ok,
            )?;
        let mut limits = base.limits;
        limits.total_model_invocations = limits.total_model_invocations.min(
            run.budget
                .total_model_invocations
                .saturating_sub(run.counters.model_invocations)
                .max(1),
        );
        limits.total_generated_tokens = limits.total_generated_tokens.min(
            run.budget
                .total_generated_tokens
                .saturating_sub(run.counters.generated_tokens)
                .max(1),
        );
        if let Some(task) = session
            .multi_task
            .as_ref()
            .and_then(|checkpoint| checkpoint.run.active_task())
        {
            limits.stage_steps = limits.stage_steps.min(task.spec.budget.stage_steps);
            limits.total_model_invocations = limits.total_model_invocations.min(
                task.spec
                    .budget
                    .total_model_invocations
                    .saturating_sub(task.counters.model_invocations)
                    .max(1),
            );
            limits.total_generated_tokens = limits.total_generated_tokens.min(
                task.spec
                    .budget
                    .total_generated_tokens
                    .saturating_sub(task.counters.generated_tokens)
                    .max(1),
            );
            limits.advisory_calls = limits.advisory_calls.min(
                task.spec
                    .budget
                    .advisory_calls
                    .saturating_sub(task.counters.advisory_calls)
                    .max(1),
            );
            limits.plan_cycles = limits.plan_cycles.min(
                task.spec
                    .budget
                    .plan_cycles
                    .saturating_sub(task.counters.plan_cycles)
                    .max(1),
            );
            limits.repair_cycles = limits.repair_cycles.min(
                task.spec
                    .budget
                    .repair_cycles
                    .saturating_sub(task.counters.repair_cycles)
                    .max(1),
            );
        }
        crate::workflow::WorkflowConfigDocument {
            version: base.version,
            delivery: base.delivery,
            default_intent: base.default_intent,
            limits,
        }
        .compile()?
    });
    request.workflow_stage = None;
    request.workflow_checkpoint = milestone.workflow.clone();
    request.goal_context = Some(run.model_brief());
    if let Some(parent) = session.multi_task.as_ref() {
        request.max_steps = request
            .max_steps
            .min(parent.run.authority.request_max_steps)
            .max(1);
    }
    request.turn_id = milestone
        .workflow
        .as_ref()
        .map(|checkpoint| checkpoint.run.source_turn_id.clone())
        .unwrap_or_else(|| new_turn_id(&session_id_for_goal(run)));
    request.conversation_handoff =
        delivery_handoff_for_turn(request.intent, &request.turn_id, &request.task, None);
    request.branch = session.branch.clone();
    request.workdir = session.workdir.clone();
    session.task = run.objective.clone();
    session.request_template = request;
    Ok(())
}

fn session_id_for_goal(run: &crate::goal::GoalRun) -> String {
    run.session_id.clone()
}

fn publish_current_goal_milestone(session: &SessionState, run: &crate::goal::GoalRun) {
    if let Some(milestone) = run.current_milestone() {
        stage_event(
            &session.runtime.history,
            AgentEvent::GoalMilestoneStarted {
                goal_id: run.id.clone(),
                milestone_id: milestone.id.clone(),
                title: milestone.title.clone(),
                timestamp_ms: Some(now_millis()),
            },
        );
    }
}

fn materialize_attachments(
    session_id: &str,
    workdir: Option<&std::path::Path>,
    attachments: Vec<InlineAttachment>,
) -> Result<Vec<SessionAttachment>> {
    if attachments.is_empty() {
        return Ok(Vec::new());
    }
    let root = workdir
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = root
        .join(".pb")
        .join("sessions")
        .join(session_id)
        .join("attachments");
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    attachments
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            let id = format!("img{}", index + 1);
            let safe_name = item
                .name
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                        c
                    } else {
                        '_'
                    }
                })
                .collect::<String>();
            let filename = if safe_name.is_empty() {
                format!("{id}.img")
            } else {
                format!("{id}-{safe_name}")
            };
            let path = dir.join(filename);
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(item.base64.trim())
                .context("invalid base64 attachment")?;
            std::fs::write(&path, &bytes)
                .with_context(|| format!("failed to write {}", path.display()))?;
            let stored_path = if let Some(workdir) = workdir {
                path.strip_prefix(workdir)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .into_owned()
            } else {
                path.to_string_lossy().into_owned()
            };
            Ok(SessionAttachment {
                id,
                name: item.name,
                mime: item.mime,
                path: stored_path,
                size: bytes.len() as u64,
            })
        })
        .collect()
}

async fn maybe_bootstrap_project(state: &AppState, task: &str) -> Result<Option<PathBuf>> {
    let Some(name) = parse_bootstrap_project_name(task) else {
        return Ok(None);
    };
    let (project_dir, request) = tokio::task::spawn_blocking(move || {
        let home =
            dirs::home_dir().context("cannot determine home directory for project bootstrap")?;
        let projects_dir = home.join("Projects");
        std::fs::create_dir_all(&projects_dir)
            .with_context(|| format!("failed to create {}", projects_dir.display()))?;
        let project_dir = projects_dir.join(&name);
        if project_dir.exists() {
            bail!(
                "bootstrap project already exists: {}",
                project_dir.display()
            );
        }
        std::fs::create_dir(&project_dir)
            .with_context(|| format!("failed to create {}", project_dir.display()))?;
        let output = Command::new("git")
            .arg("init")
            .current_dir(&project_dir)
            .output()
            .context("failed to run git init for bootstrap project")?;
        if !output.status.success() {
            bail!(
                "git init failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let request = AddProjectRequest {
            name: Some(name),
            path: project_dir.to_string_lossy().to_string(),
        };
        Ok::<_, anyhow::Error>((project_dir, request))
    })
    .await
    .context("project bootstrap task failed")??;
    state
        .commit_project_registry_owned(UsageWindow::current_utc_day(), move || {
            projects::add_project(request)
        })
        .await?;
    Ok(Some(project_dir))
}

fn parse_bootstrap_project_name(task: &str) -> Option<String> {
    let lower = task.to_ascii_lowercase();
    for marker in [
        "new repo called ",
        "new repository called ",
        "repo called ",
        "repository called ",
    ] {
        if let Some(index) = lower.find(marker) {
            let start = index + marker.len();
            let raw = task[start..]
                .split_whitespace()
                .next()?
                .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_');
            let name = raw
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
                .collect::<String>();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    None
}

async fn continue_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<ContinueSessionRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let task = req.task.trim().to_string();
    if task.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_task",
            "session task must not be empty",
        ));
    }
    let session_id = id.clone();
    let commit = state
        .commit_session_change_owned(id, move |session| {
            if !matches!(
                session.status,
                SessionStatus::Completed | SessionStatus::Failed
            ) {
                anyhow::bail!(ContinueSessionError::NotContinuable);
            }

            let mut request = session.request_template.clone();
            request.task = task;
            let workflow_policy = workflow_policy_for_request(session.workdir.as_deref())
                .map_err(ContinueSessionError::WorkflowPolicy)?;
            request.intent = Some(req.intent.unwrap_or(workflow_policy.default_intent));
            request.workflow_policy = Some(workflow_policy);
            request.workflow_stage = None;
            request.workflow_checkpoint = None;
            request.goal_context = None;
            request.turn_id = new_turn_id(&session_id);
            if req.proposal_id.is_some()
                && request.intent != Some(crate::workflow::TurnIntent::Deliver)
            {
                anyhow::bail!(ContinueSessionError::InvalidProposal(
                    "only a delivery turn can cite a delivery proposal"
                ));
            }
            let cited_proposal = match req.proposal_id.as_deref() {
                Some(proposal_id) => Some(
                    session
                        .durable
                        .pending_delivery_proposal
                        .as_ref()
                        .filter(|proposal| proposal.id == proposal_id)
                        .cloned()
                        .ok_or(ContinueSessionError::InvalidProposal(
                            "delivery proposal is missing or stale",
                        ))?,
                ),
                None => None,
            };
            request.conversation_handoff = delivery_handoff_for_turn(
                request.intent,
                &request.turn_id,
                &request.task,
                cited_proposal.as_ref(),
            );
            request.infer_profile = true;
            request.branch = session.branch.clone();
            request.workdir = session.workdir.clone();
            request.prior_check_evidence = {
                let history = session
                    .runtime
                    .history
                    .lock()
                    .map_err(|_| ContinueSessionError::HistoryUnavailable)?;
                let events = history
                    .iter()
                    .map(|envelope| envelope.event.clone())
                    .collect::<Vec<_>>();
                crate::checks::CheckEvidenceLedger::from_events(&events)
            };
            session.task = request.task.clone();
            session.title = None;
            session.request_template = request.clone();
            if request.intent == Some(crate::workflow::TurnIntent::Deliver) {
                session.durable.pending_delivery_proposal = None;
            }
            session.status = SessionStatus::Queued;
            session.runtime.pending_question = None;
            session.workflow = None;
            if let Some(multi_task) = session.multi_task.take()
                && session
                    .completed_multi_tasks
                    .last()
                    .is_none_or(|existing| existing.run.id != multi_task.run.id)
            {
                session.completed_multi_tasks.push(multi_task);
            }
            session.runtime.cancel_token.store(false, Ordering::SeqCst);
            session.updated_at_ms = now_millis();
            Ok(StagedSessionChange::new(()))
        })
        .await
        .map_err(continue_session_api_error)?;
    Ok(Json(commit.snapshot))
}

#[derive(Debug, thiserror::Error)]
enum ContinueSessionError {
    #[error("only a completed or failed session can be continued")]
    NotContinuable,
    #[error("{0}")]
    InvalidProposal(&'static str),
    #[error("failed to load workflow policy: {0:#}")]
    WorkflowPolicy(#[source] anyhow::Error),
    #[error("session history is unavailable")]
    HistoryUnavailable,
}

fn continue_session_api_error(error: anyhow::Error) -> ApiError {
    if error.downcast_ref::<SessionNotFoundError>().is_some() {
        ApiError::new(StatusCode::NOT_FOUND, "session_not_found", error)
    } else if error.downcast_ref::<SessionPersistenceError>().is_some() {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "session_persistence_failed",
            error,
        )
    } else {
        match error.downcast_ref::<ContinueSessionError>() {
            Some(ContinueSessionError::NotContinuable) => {
                ApiError::new(StatusCode::CONFLICT, "session_not_continuable", error)
            }
            Some(ContinueSessionError::InvalidProposal(_)) => {
                ApiError::new(StatusCode::BAD_REQUEST, "invalid_proposal", error)
            }
            Some(ContinueSessionError::WorkflowPolicy(_)) => ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "workflow_policy_failed",
                error,
            ),
            Some(ContinueSessionError::HistoryUnavailable) | None => ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "session_history_unavailable",
                error,
            ),
        }
    }
}

async fn send_session_message(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<SendSessionMessageRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let message = req.message.trim().to_string();
    if message.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_message",
            "message must not be empty",
        ));
    }
    if message.chars().count() > MAX_USER_MESSAGE_CHARS {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "message_too_large",
            format!("message must contain at most {MAX_USER_MESSAGE_CHARS} characters"),
        ));
    }

    let message_id = new_user_message_id(&id);
    let commit = state
        .commit_session_change_owned(id, move |session| {
            if session.status != SessionStatus::Running {
                anyhow::bail!(SendSessionMessageError::NotRunning);
            }
            if !session
                .runtime
                .accepting_user_messages
                .load(Ordering::SeqCst)
            {
                anyhow::bail!(SendSessionMessageError::WindowClosed);
            }
            let mut pending = session
                .runtime
                .pending_user_messages
                .lock()
                .map_err(|_| SendSessionMessageError::QueueUnavailable)?;
            if pending.len() >= MAX_PENDING_USER_MESSAGES {
                anyhow::bail!(SendSessionMessageError::QueueFull);
            }
            pending.push_back(QueuedUserMessage {
                message_id: message_id.clone(),
                message: message.clone(),
            });
            drop(pending);
            session.updated_at_ms = now_millis();
            stage_event(
                &session.runtime.history,
                AgentEvent::UserMessage {
                    message_id,
                    message,
                    timestamp_ms: Some(now_millis()),
                },
            );
            Ok(StagedSessionChange::new(()))
        })
        .await
        .map_err(send_session_message_api_error)?;
    Ok(Json(commit.snapshot))
}

#[derive(Debug, thiserror::Error)]
enum SendSessionMessageError {
    #[error("messages can only be sent to a running session")]
    NotRunning,
    #[error("the agent is no longer accepting messages for this turn")]
    WindowClosed,
    #[error("the pending message queue is full")]
    QueueFull,
    #[error("the session message queue is unavailable")]
    QueueUnavailable,
}

fn send_session_message_api_error(error: anyhow::Error) -> ApiError {
    if error.downcast_ref::<SessionNotFoundError>().is_some() {
        ApiError::new(StatusCode::NOT_FOUND, "session_not_found", error)
    } else if error.downcast_ref::<SessionPersistenceError>().is_some() {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "session_persistence_failed",
            error,
        )
    } else {
        match error.downcast_ref::<SendSessionMessageError>() {
            Some(SendSessionMessageError::NotRunning) => {
                ApiError::new(StatusCode::CONFLICT, "session_not_running", error)
            }
            Some(SendSessionMessageError::WindowClosed) => {
                ApiError::new(StatusCode::CONFLICT, "message_window_closed", error)
            }
            Some(SendSessionMessageError::QueueFull) => {
                ApiError::new(StatusCode::TOO_MANY_REQUESTS, "message_queue_full", error)
            }
            Some(SendSessionMessageError::QueueUnavailable) | None => ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "message_queue_unavailable",
                error,
            ),
        }
    }
}

async fn resume_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let result = resume_session_inner(state.clone(), id)
        .await
        .map_err(|err| match err.downcast_ref::<ResumeSessionError>() {
            Some(ResumeSessionError::Conflict) => {
                ApiError::new(StatusCode::CONFLICT, "session_not_resumable", err)
            }
            None if err.downcast_ref::<SessionNotFoundError>().is_some() => {
                ApiError::new(StatusCode::NOT_FOUND, "session_not_found", err)
            }
            None if err.downcast_ref::<SessionPersistenceError>().is_some() => ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "session_persistence_failed",
                err,
            ),
            None => ApiError::new(StatusCode::CONFLICT, "session_not_resumable", err),
        })?;
    Ok(Json(result.snapshot))
}

#[derive(Debug)]
enum ResumeSessionError {
    Conflict,
}

impl std::fmt::Display for ResumeSessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conflict => f.write_str("session is not a resumable restored queue"),
        }
    }
}

impl std::error::Error for ResumeSessionError {}

async fn resume_session_inner(state: AppState, id: String) -> Result<SessionMutationReceipt> {
    let commit = state
        .commit_session_change_owned(id.clone(), |session| {
            let blocked_workflow = session.workflow.as_ref().is_some_and(|checkpoint| {
                checkpoint.run.stage == crate::workflow::WorkflowStage::Blocked
            });
            let blocked_workflow_requires_restart =
                session.workflow.as_ref().is_some_and(|checkpoint| {
                    checkpoint.run.stage == crate::workflow::WorkflowStage::Blocked
                        && checkpoint.run.outcome
                            == Some(crate::workflow::WorkflowOutcome::CommitBlocked)
                });
            let resumable_multi_task = session.multi_task.as_ref().is_some_and(|checkpoint| {
                matches!(
                    checkpoint.run.stage,
                    crate::task_queue::MultiTaskStage::Paused
                        | crate::task_queue::MultiTaskStage::Blocked
                )
            });
            if (!matches!(session.status, SessionStatus::Paused)
                && !(session.status == SessionStatus::Failed
                    && (blocked_workflow || resumable_multi_task)))
                || session.runtime.pending_question.is_some()
            {
                anyhow::bail!(ResumeSessionError::Conflict);
            }
            if blocked_workflow_requires_restart {
                anyhow::bail!(ResumeSessionError::Conflict);
            }
            let mut request = session.request_template.clone();
            if let Some(checkpoint) = session.multi_task.clone() {
                let mut run = checkpoint.run;
                let repository = crate::task_queue::TaskRepositoryState::capture(
                    std::path::Path::new(&run.authority.workdir),
                )?;
                let now = now_millis().max(run.updated_at_ms);
                run.apply(crate::task_queue::MultiTaskEvent::ResumeRequested {
                    repository: repository.clone(),
                    now_ms: now,
                })?;
                let is_goal_task = run
                    .active_task()
                    .is_some_and(|task| task.spec.kind == crate::task_queue::TaskKind::Goal);
                if is_goal_task {
                    let mut goal = session
                        .goal
                        .clone()
                        .context("resumable Goal Task lost its Goal checkpoint")?
                        .run;
                    goal.resume(now)?;
                    let goal = crate::goal::GoalCheckpoint::new(goal)?;
                    let task_id = run
                        .active_task_id
                        .clone()
                        .context("resumable Goal Task has no active Task")?;
                    run.apply(crate::task_queue::MultiTaskEvent::ChildCheckpointed {
                        task_id,
                        child: crate::task_queue::TaskChildCheckpoint::Goal(goal.clone()),
                        repository,
                        now_ms: now_millis().max(run.updated_at_ms),
                    })?;
                    let parent = crate::task_queue::MultiTaskCheckpoint::new(run)?;
                    let previous_goal = session.goal.clone();
                    let previous_parent = session.multi_task.clone();
                    let previous_request = session.request_template.clone();
                    let previous_task = session.task.clone();
                    let previous_workflow = session.workflow.clone();
                    session.goal = Some(goal.clone());
                    session.multi_task = Some(parent);
                    if let Err(error) = configure_goal_milestone_request(session, &goal.run) {
                        session.goal = previous_goal;
                        session.multi_task = previous_parent;
                        session.request_template = previous_request;
                        session.task = previous_task;
                        session.workflow = previous_workflow;
                        return Err(error);
                    }
                    stage_event(
                        &session.runtime.history,
                        AgentEvent::GoalResumed {
                            goal_id: goal.run.id.clone(),
                            timestamp_ms: Some(now_millis()),
                        },
                    );
                    publish_current_goal_milestone(session, &goal.run);
                } else {
                    let checkpoint = crate::task_queue::MultiTaskCheckpoint::new(run)?;
                    let status = dispatch_multi_task_active(session, &checkpoint)?;
                    if status != SessionStatus::Queued {
                        anyhow::bail!(ResumeSessionError::Conflict);
                    }
                }
                request = session.request_template.clone();
            } else {
                request.workflow_checkpoint = session.workflow.clone();
            }
            session.status = SessionStatus::Queued;
            session.updated_at_ms = now_millis();
            session.runtime.cancel_token.store(false, Ordering::SeqCst);
            session.request_template = request.clone();
            Ok(StagedSessionChange::new(()))
        })
        .await?;
    Ok(SessionMutationReceipt {
        response: SessionResponse { session_id: id },
        snapshot: commit.snapshot,
    })
}

async fn restart_delivery(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let result = restart_delivery_inner(state.clone(), id)
        .await
        .map_err(|error| session_control_api_error(error, "delivery_not_restartable"))?;
    Ok(Json(result.snapshot))
}

fn prepare_blocked_workflow_restart(session: &mut SessionState, session_id: &str) -> Result<()> {
    let checkpoint = session
        .workflow
        .as_ref()
        .context("session has no blocked delivery to restart")?;
    if session.status != SessionStatus::Failed
        || session.runtime.pending_question.is_some()
        || session.multi_task.is_some()
        || session.goal.is_some()
        || checkpoint.run.stage != crate::workflow::WorkflowStage::Blocked
        || checkpoint.run.outcome != Some(crate::workflow::WorkflowOutcome::CommitBlocked)
    {
        bail!("session has no content-sensitive blocked delivery to restart");
    }

    let summary = crate::workflow::WorkflowSummary::from(&checkpoint.run);
    if session
        .completed_workflows
        .last()
        .is_none_or(|existing| existing.id != summary.id)
    {
        session.completed_workflows.push(summary);
    }
    session.workflow = None;

    let turn_id = new_turn_id(session_id);
    let request = &mut session.request_template;
    request.turn_id = turn_id.clone();
    request.intent = Some(crate::workflow::TurnIntent::Deliver);
    request.workflow_stage = None;
    request.workflow_expected_content_fingerprint = None;
    request.workflow_plan_identity = None;
    request.workflow_action_first_turn = false;
    request.workflow_creation_path_order.clear();
    request.workflow_work_units = None;
    request.workflow_stage_evidence = None;
    request.workflow_checkpoint = None;
    request.repository_context = None;
    request.prior_check_evidence = crate::checks::CheckEvidenceLedger::default();
    request.conversation_handoff =
        delivery_handoff_for_turn(request.intent, &turn_id, &request.task, None);
    session.status = SessionStatus::Queued;
    session.updated_at_ms = now_millis();
    session.runtime.cancel_token.store(false, Ordering::SeqCst);
    session.runtime.pause_token.store(false, Ordering::SeqCst);
    stage_event(
                &session.runtime.history,
        AgentEvent::Correction {
            kind: crate::events::CorrectionKind::TaskPlanningRecovery,
            message: "I kept the blocked plan and review in the session history, accepted the repository's current files as the new baseline, and started a fresh planning pass."
                .to_string(),
            summary: "Restarting delivery from current files".to_string(),
            actor: crate::events::TeamActor::workflow_steward(),
            assisting_profile: Some(crate::agent_core::AgentProfile::Plan),
            nesting_depth: None,
            timestamp_ms: Some(now_millis()),
        },
    );
    Ok(())
}

async fn restart_delivery_inner(state: AppState, id: String) -> Result<SessionMutationReceipt> {
    let session_id = id.clone();
    let commit = state
        .commit_session_change_owned(id.clone(), move |session| {
            prepare_blocked_workflow_restart(session, &session_id)?;
            Ok(StagedSessionChange::new(()))
        })
        .await?;
    Ok(SessionMutationReceipt {
        response: SessionResponse { session_id: id },
        snapshot: commit.snapshot,
    })
}

async fn retry_task_planning(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let result = recover_task_plan(
        state.clone(),
        id,
        crate::agent_core::TaskPlanningPreference::Auto,
    )
    .await
    .map_err(|error| session_control_api_error(error, "task_plan_not_recoverable"))?;
    Ok(Json(result.snapshot))
}

async fn run_as_one_build(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let result = recover_task_plan(
        state.clone(),
        id,
        crate::agent_core::TaskPlanningPreference::OneBuild,
    )
    .await
    .map_err(|error| session_control_api_error(error, "task_plan_not_recoverable"))?;
    Ok(Json(result.snapshot))
}

async fn recover_task_plan(
    state: AppState,
    id: String,
    preference: crate::agent_core::TaskPlanningPreference,
) -> Result<SessionMutationReceipt> {
    let session_id = id.clone();
    let commit = state
        .commit_session_change_owned(id.clone(), move |session| {
            if session.status != SessionStatus::Failed
                || session.request_template.task_plan_rejected.is_none()
                || session.multi_task.is_some()
                || session.goal.is_some()
            {
                bail!("session has no recoverable Task-planning failure");
            }
            let action = match preference {
                crate::agent_core::TaskPlanningPreference::Auto => "Retrying Task planning",
                crate::agent_core::TaskPlanningPreference::OneBuild => "Running as one Build",
            };
            session.request_template.task_planning = preference;
            session.request_template.task_plan_rejected = None;
            session.request_template.task_planning_transcript = None;
            session.request_template.turn_id = new_turn_id(&session_id);
            session.request_template.workflow_stage = None;
            session.request_template.workflow_checkpoint = None;
            session.request_template.repository_context = None;
            session.request_template.intent = Some(crate::workflow::TurnIntent::Deliver);
            session.request_template.conversation_handoff = delivery_handoff_for_turn(
                session.request_template.intent,
                &session.request_template.turn_id,
                &session.request_template.task,
                None,
            );
            session.status = SessionStatus::Queued;
            session.runtime.cancel_token.store(false, Ordering::SeqCst);
            session.updated_at_ms = now_millis();
            stage_event(
                                &session.runtime.history,
                AgentEvent::Correction {
                    kind: crate::events::CorrectionKind::TaskPlanningRecovery,
                    message: format!(
                        "{action}. Repository state and existing commits remain unchanged until the selected workflow delivers."
                    ),
                    summary: action.to_string(),
                    actor: crate::events::TeamActor::workflow_steward(),
                    assisting_profile: Some(crate::agent_core::AgentProfile::Plan),
                    nesting_depth: None,
                    timestamp_ms: Some(now_millis()),
                },
            );
            Ok(StagedSessionChange::new(()))
        })
        .await?;
    Ok(SessionMutationReceipt {
        response: SessionResponse { session_id: id },
        snapshot: commit.snapshot,
    })
}

fn session_control_api_error(error: anyhow::Error, conflict_code: &'static str) -> ApiError {
    if error.downcast_ref::<SessionNotFoundError>().is_some() {
        ApiError::new(StatusCode::NOT_FOUND, "session_not_found", error)
    } else if error.downcast_ref::<SessionPersistenceError>().is_some() {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "session_persistence_failed",
            error,
        )
    } else {
        ApiError::new(StatusCode::CONFLICT, conflict_code, error)
    }
}

async fn cancel_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let result = cancel_session_inner(state.clone(), id)
        .await
        .map_err(|error| {
            if error.downcast_ref::<SessionPersistenceError>().is_some() {
                ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "session_persistence_failed",
                    error,
                )
            } else {
                ApiError::new(StatusCode::CONFLICT, "session_not_cancellable", error)
            }
        })?;
    Ok(Json(result.snapshot))
}

async fn cancel_session_inner(state: AppState, id: String) -> Result<SessionMutationReceipt> {
    let commit = state
        .commit_session_change_owned(id.clone(), |staged| {
            let blocked_multi_task = staged.multi_task.as_ref().is_some_and(|checkpoint| {
                checkpoint.run.stage == crate::task_queue::MultiTaskStage::Blocked
            });
            if matches!(staged.status, SessionStatus::Completed)
                || (staged.status == SessionStatus::Failed
                    && !blocked_multi_task
                    && staged.workflow.as_ref().is_none_or(|checkpoint| {
                        checkpoint.run.stage != crate::workflow::WorkflowStage::Blocked
                    }))
            {
                bail!("session is not cancellable");
            }
            let terminate_environment = staged.status != SessionStatus::Running;
            staged.runtime.cancel_token.store(true, Ordering::SeqCst);
            staged.runtime.pending_question.take();
            stage_event(
                                &staged.runtime.history,
                AgentEvent::Correction {
                    kind: crate::events::CorrectionKind::Lifecycle,
                    message: "Cancellation requested. Repository content and workflow evidence will be preserved."
                        .to_string(),
                    summary: "Cancellation requested".to_string(),
                    actor: crate::events::TeamActor::workflow_steward(),
                    assisting_profile: Some(staged.request_template.profile),
                    nesting_depth: None,
                    timestamp_ms: Some(now_millis()),
                },
            );
            if terminate_environment {
                if let Some(checkpoint) = staged.multi_task.take() {
                    let mut run = checkpoint.run;
                    if !run.stage.is_terminal() {
                        let now = now_millis().max(run.updated_at_ms);
                        run.apply(crate::task_queue::MultiTaskEvent::Cancelled {
                            reason: "cancelled by user; completed Task commits were preserved"
                                .to_string(),
                            now_ms: now,
                        })?;
                    }
                    archive_multi_task(staged, crate::task_queue::MultiTaskCheckpoint::new(run)?);
                }
                if let Some(checkpoint) = staged.workflow.take() {
                    let mut run = checkpoint.run;
                    if run.stage == crate::workflow::WorkflowStage::Blocked {
                        run.apply(crate::workflow::WorkflowEvent::Resumed)?;
                    }
                    if !run.stage.is_terminal() {
                        run.apply(crate::workflow::WorkflowEvent::Cancelled {
                            reason:
                                "cancelled by user; repository content and evidence were preserved"
                                    .to_string(),
                        })?;
                    }
                    let checkpoint = crate::workflow::WorkflowCheckpoint::new(run)?;
                    let summary = crate::workflow::WorkflowSummary::from(&checkpoint.run);
                    if staged
                        .completed_workflows
                        .last()
                        .is_none_or(|existing| existing.id != summary.id)
                    {
                        staged.completed_workflows.push(summary);
                    }
                    stage_event(
                                                &staged.runtime.history,
                        AgentEvent::WorkflowCompleted {
                            workflow_id: checkpoint.run.id,
                            outcome: crate::workflow::WorkflowOutcome::Cancelled,
                            checkpoint_sha256: checkpoint.sha256,
                            ready_evidence_sha256: None,
                            timestamp_ms: Some(now_millis()),
                        },
                    );
                }
                staged.request_template.workflow_checkpoint = None;
                staged.status = SessionStatus::Completed;
                staged.updated_at_ms = now_millis();
                Ok(StagedSessionChange::new(terminate_environment).terminate_environment())
            } else {
                staged.updated_at_ms = now_millis();
                Ok(StagedSessionChange::new(terminate_environment))
            }
        })
        .await?;
    let terminate_environment = commit.value;
    let snapshot = commit.snapshot;
    debug_assert_eq!(
        terminate_environment,
        snapshot.session.status != SessionStatus::Running
    );
    Ok(SessionMutationReceipt {
        response: SessionResponse { session_id: id },
        snapshot,
    })
}

async fn answer_question(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<AnswerQuestionRequest>,
) -> Result<Json<SessionStreamSnapshot>, ApiError> {
    let result = answer_question_inner(state.clone(), id, req)
        .await
        .map_err(|err| {
            if err.downcast_ref::<SessionNotFoundError>().is_some() {
                ApiError::new(StatusCode::NOT_FOUND, "session_not_found", err)
            } else if err.downcast_ref::<SessionPersistenceError>().is_some() {
                ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "session_persistence_failed",
                    err,
                )
            } else {
                ApiError::new(StatusCode::CONFLICT, "question_mismatch", err)
            }
        })?;
    Ok(Json(result.snapshot))
}

#[derive(Debug)]
enum AnswerQuestionError {
    Conflict,
}

impl std::fmt::Display for AnswerQuestionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conflict => f.write_str(
                "session does not have a pending question matching the requested question id",
            ),
        }
    }
}

impl std::error::Error for AnswerQuestionError {}

async fn answer_question_inner(
    state: AppState,
    id: String,
    req: AnswerQuestionRequest,
) -> Result<SessionMutationReceipt> {
    let requested_question_id = req.question_id;
    let requested_answer = req.answer.trim().to_string();
    let wake_state = state.clone();
    let wake_session_id = id.clone();
    let commit = state
        .commit_session_change_owned_after(
            id.clone(),
            move |session| {
                let Some(pending) = session.runtime.pending_question.take() else {
                    anyhow::bail!(AnswerQuestionError::Conflict);
                };
                if pending.question_id != requested_question_id {
                    anyhow::bail!(AnswerQuestionError::Conflict);
                }
                if requested_answer.is_empty()
                    || (!pending.choices.is_empty() && !pending.choices.contains(&requested_answer))
                {
                    anyhow::bail!(AnswerQuestionError::Conflict);
                }
                session.status = SessionStatus::Running;
                session
                    .runtime
                    .accepting_user_messages
                    .store(true, Ordering::SeqCst);
                session.updated_at_ms = now_millis();
                stage_event(
                    &session.runtime.history,
                    AgentEvent::UserAnswer {
                        question_id: requested_question_id,
                        answer: requested_answer.clone(),
                        timestamp_ms: Some(now_millis()),
                    },
                );
                Ok(StagedSessionChange::new((
                    pending.responder,
                    requested_answer,
                )))
            },
            move |(responder, answer)| {
                wake_state.update_sleep_prevention_working(true);
                if responder.send(answer.clone()).is_err() {
                    let warning =
                    "the answer was committed, but the completed session runner could not be woken"
                        .to_string();
                    tracing::warn!(session_id = %wake_session_id, %warning);
                    vec![warning]
                } else {
                    Vec::new()
                }
            },
        )
        .await?;
    Ok(SessionMutationReceipt {
        response: SessionResponse { session_id: id },
        snapshot: commit.snapshot,
    })
}

fn session_history_summary(session: &SessionState) -> (Option<String>, Option<HandoffOutcome>) {
    let Ok(history) = session.runtime.history.lock() else {
        return (session.title.clone(), None);
    };
    (
        latest_session_title(&history).or_else(|| session.title.clone()),
        handoff_outcome_from_history(&history),
    )
}

fn handoff_outcome_from_history(history: &[EventEnvelope]) -> Option<HandoffOutcome> {
    history.iter().rev().find_map(|envelope| {
        if let AgentEvent::HandoffSummary { summary, .. } = &envelope.event {
            Some(summary.outcome)
        } else {
            None
        }
    })
}

fn session_usage_records(session: &SessionState) -> Vec<SessionMetricsSnapshot> {
    session
        .runtime
        .usage_records
        .lock()
        .map(|records| records.clone())
        .unwrap_or_default()
}

fn latest_workflow_summary(session: &SessionState) -> Option<crate::workflow::WorkflowSummary> {
    if let Some(goal) = session.goal.as_ref() {
        return goal.run.current_milestone().and_then(|milestone| {
            milestone
                .workflow
                .as_ref()
                .map(|checkpoint| crate::workflow::WorkflowSummary::from(&checkpoint.run))
                .or_else(|| milestone.workflow_summary.clone())
        });
    }
    session
        .workflow
        .as_ref()
        .map(|checkpoint| crate::workflow::WorkflowSummary::from(&checkpoint.run))
        .or_else(|| session.completed_workflows.last().cloned())
}

fn strict_workflow_enabled(session: &SessionState) -> bool {
    (session.request_template.intent == Some(crate::workflow::TurnIntent::Deliver)
        && session.request_template.workflow_policy.is_some()
        && !session.request_template.turn_id.trim().is_empty())
        || session.goal.is_some()
}

fn latest_goal_checkpoint(session: &SessionState) -> Option<&crate::goal::GoalCheckpoint> {
    session
        .goal
        .as_ref()
        .or_else(|| session.completed_goals.last())
}

fn latest_multi_task_checkpoint(
    session: &SessionState,
) -> Option<&crate::task_queue::MultiTaskCheckpoint> {
    session
        .multi_task
        .as_ref()
        .or_else(|| session.completed_multi_tasks.last())
}

fn multi_task_summary(checkpoint: &crate::task_queue::MultiTaskCheckpoint) -> MultiTaskSummary {
    MultiTaskSummary {
        id: checkpoint.run.id.clone(),
        stage: checkpoint.run.stage,
        outcome: checkpoint.run.outcome,
        completed_tasks: checkpoint
            .run
            .tasks
            .iter()
            .filter(|task| task.state.is_success())
            .count(),
        total_tasks: checkpoint.run.plan.artifact.tasks.len(),
        active_task_title: checkpoint
            .run
            .active_task()
            .map(|task| task.spec.title.clone()),
    }
}

fn task_deadline_ms(checkpoint: &crate::task_queue::MultiTaskCheckpoint) -> Option<u64> {
    let task = checkpoint.run.active_task()?;
    let allowance = task.spec.budget.wall_time_minutes.saturating_mul(60_000);
    let remaining = allowance.saturating_sub(task.counters.elapsed_ms);
    Some(now_millis().saturating_add(remaining))
}

async fn list_sessions(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Json<Vec<SessionListItem>> {
    Json(session_list_snapshot(&state).await)
}

async fn get_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionDetails>, ApiError> {
    session_details_snapshot(&state, &id)
        .await
        .map(Json)
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "session_not_found",
                format!("session not found: {id}"),
            )
        })
}

async fn status(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Json<StatusResponse> {
    let sessions = state.sessions.lock().await;
    let running_sessions = sessions
        .values()
        .filter(|session| session.status == SessionStatus::Running)
        .count();
    let queued_sessions = sessions
        .values()
        .filter(|session| session.status == SessionStatus::Queued)
        .count();
    let paused_sessions = sessions
        .values()
        .filter(|session| session.status == SessionStatus::Paused)
        .count();
    let completed_sessions = sessions
        .values()
        .filter(|session| session.status == SessionStatus::Completed)
        .count();
    let failed_sessions = sessions
        .values()
        .filter(|session| session.status == SessionStatus::Failed)
        .count();
    Json(StatusResponse {
        busy: running_sessions > 0,
        running_sessions,
        queued_sessions,
        paused_sessions,
        completed_sessions,
        failed_sessions,
        total_sessions: sessions.len(),
    })
}

async fn delete_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Query(query): Query<ProjectSessionQuery>,
) -> Result<Json<DeleteSessionMutationResponse>, ApiError> {
    let usage_window = query.usage_window()?;
    state
        .delete_session_owned(id, usage_window)
        .await
        .map(Json)
        .map_err(|error| match error {
            DeleteSessionError::NotFound(message) => {
                ApiError::new(StatusCode::NOT_FOUND, "session_not_found", message)
            }
            DeleteSessionError::Active(message) => {
                ApiError::new(StatusCode::CONFLICT, "session_active", message)
            }
            DeleteSessionError::Internal(error) => ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "session_delete_failed",
                error,
            ),
        })
}

#[derive(Debug, thiserror::Error)]
enum DeleteSessionError {
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Active(String),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

async fn delete_session_inner(
    state: AppState,
    id: &str,
) -> Result<DeleteSessionMutationResponse, DeleteSessionError> {
    state
        .delete_session_owned(
            id.to_string(),
            UsageWindow {
                start_ms: 0,
                end_ms: 1,
            },
        )
        .await
}

fn cleanup_deleted_session_resources(id: &str) -> Vec<String> {
    let mut warnings = Vec::new();
    if let Err(error) = crate::session_environment::terminate_global_session(id) {
        warnings.push(format!(
            "could not terminate the session environment: {error:#}"
        ));
    }
    match crate::session_workspace::WorkspaceManager::persistent() {
        Ok(workspace_manager) => match workspace_manager.find_record_by_session(id) {
            Ok(Some(record)) => {
                if let Err(error) = workspace_manager.remove(&record, true) {
                    warnings.push(format!("could not remove the session workspace: {error:#}"));
                }
            }
            Ok(None) => {}
            Err(error) => warnings.push(format!(
                "could not inspect the session workspace record: {error:#}"
            )),
        },
        Err(error) => warnings.push(format!(
            "could not initialize session workspace cleanup: {error:#}"
        )),
    }
    warnings
}

async fn list_projects(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<Vec<ProjectEntry>>, ApiError> {
    reload_projects(&state).await.map_err(|error| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "project_registry_unavailable",
            error,
        )
    })?;
    Ok(Json(project_list_snapshot(&state).await))
}

async fn list_project_sessions(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Query(query): Query<ProjectSessionQuery>,
) -> Result<Json<ProjectSessionSnapshot>, ApiError> {
    let usage_window = query.usage_window()?;
    reload_projects(&state).await.map_err(|error| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "project_registry_unavailable",
            error,
        )
    })?;
    let transition_floor = state.project_session_revision_baseline();
    Ok(Json(
        project_session_snapshot(&state, transition_floor, usage_window).await,
    ))
}

async fn project_session_events(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    headers: HeaderMap,
    Query(query): Query<ProjectSessionQuery>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let usage_window = query.usage_window()?;
    reload_projects(&state).await.map_err(|error| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "project_registry_unavailable",
            error,
        )
    })?;
    let (receiver, subscribed_revision) = state.subscribe_project_session_changes();
    let transition_floor =
        project_session_last_event_revision(&headers, state.project_session_stream_id.as_str())
            .or_else(|| {
                query.last_event_id.as_deref().and_then(|value| {
                    project_session_event_revision(value, state.project_session_stream_id.as_str())
                })
            })
            .map_or(subscribed_revision, |revision| {
                revision.min(subscribed_revision)
            });
    let initial_snapshot = project_session_snapshot(&state, transition_floor, usage_window).await;
    let live_floor = initial_snapshot.revision;
    let initial =
        futures::stream::iter(project_session_snapshot_sse_event(&initial_snapshot).map(Ok));
    let live = futures::stream::unfold(
        (receiver, state, usage_window, live_floor),
        |(mut receiver, state, usage_window, mut terminal_transition_floor)| async move {
            loop {
                let snapshot = next_project_session_snapshot(
                    &mut receiver,
                    &state,
                    terminal_transition_floor,
                    usage_window,
                )
                .await?;
                terminal_transition_floor = snapshot.revision;
                if let Some(event) = project_session_snapshot_sse_event(&snapshot) {
                    return Some((
                        Ok(event),
                        (receiver, state, usage_window, terminal_transition_floor),
                    ));
                }
            }
        },
    );
    Ok(Sse::new(initial.chain(live)).keep_alive(KeepAlive::default()))
}

async fn next_project_session_snapshot(
    receiver: &mut broadcast::Receiver<u64>,
    state: &AppState,
    terminal_transition_floor: u64,
    usage_window: UsageWindow,
) -> Option<ProjectSessionSnapshot> {
    loop {
        match receiver.recv().await {
            Ok(revision) if revision <= terminal_transition_floor => continue,
            Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {
                let snapshot =
                    project_session_snapshot(state, terminal_transition_floor, usage_window).await;
                if snapshot.revision > terminal_transition_floor {
                    return Some(snapshot);
                }
            }
            Err(broadcast::error::RecvError::Closed) => return None,
        }
    }
}

fn project_session_last_event_revision(headers: &HeaderMap, stream_id: &str) -> Option<u64> {
    let value = headers.get("last-event-id")?.to_str().ok()?.trim();
    project_session_event_revision(value, stream_id)
}

fn project_session_event_revision(value: &str, stream_id: &str) -> Option<u64> {
    let (cursor_stream_id, revision) = value.rsplit_once(':')?;
    (cursor_stream_id == stream_id)
        .then(|| revision.parse::<u64>().ok())
        .flatten()
}

fn project_session_snapshot_sse_event(snapshot: &ProjectSessionSnapshot) -> Option<Event> {
    let data = serde_json::to_string(snapshot).ok()?;
    Some(
        Event::default()
            .id(format!("{}:{}", snapshot.stream_id, snapshot.revision))
            .event("project_session_snapshot")
            .data(data),
    )
}

async fn update_project_notifications(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Query(query): Query<ProjectSessionQuery>,
    Json(req): Json<UpdateProjectNotificationsRequest>,
) -> Result<Json<ProjectSessionSnapshot>, ApiError> {
    let usage_window = query.usage_window()?;
    let notify_on_finish = req.notify_on_finish;
    state
        .commit_project_registry_owned(usage_window, move || {
            projects::set_project_notifications_by_id(&id, notify_on_finish)
        })
        .await
        .map(|receipt| Json(receipt.snapshot))
        .map_err(|error| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "project_notification_update_failed",
                error,
            )
        })
}

async fn get_settings(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Json<WebSettingsResponse> {
    Json(web_settings_snapshot(&state))
}

async fn update_settings(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<UpdateWebSettingsRequest>,
) -> Result<Json<WebSettingsResponse>, ApiError> {
    let previous_enabled = state.sleep_prevention_status().enabled;
    state.update_sleep_prevention_enabled(req.prevent_sleep_while_working);

    let enabled = req.prevent_sleep_while_working;
    let persisted = tokio::task::spawn_blocking(move || -> Result<()> {
        crate::config::UserConfig::mutate(|config| {
            config.web.prevent_sleep_while_working = Some(enabled);
            Ok(())
        })
    })
    .await;

    let persist_error = match persisted {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(format!("{error:#}")),
        Err(error) => Some(format!("settings persistence task failed: {error}")),
    };
    if let Some(error) = persist_error {
        state.update_sleep_prevention_enabled(previous_enabled);
        return Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "settings_persist_failed",
            error,
        ));
    }

    Ok(Json(web_settings_snapshot(&state)))
}

async fn get_tailscale_settings(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<crate::tailscale::TailscaleStatus>, IntegrationApiError> {
    let tailscale = Arc::clone(&state.tailscale);
    let web_listen = state.web_listen.clone();
    tokio::task::spawn_blocking(move || {
        tailscale
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .status(&web_listen)
    })
    .await
    .map(Json)
    .map_err(|error| {
        IntegrationApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to inspect Tailscale: {error}"),
        )
    })
}

async fn update_tailscale_settings(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<UpdateTailscaleSettingsRequest>,
) -> Result<Json<crate::tailscale::TailscaleStatus>, IntegrationApiError> {
    let tailscale = Arc::clone(&state.tailscale);
    let web_listen = state.web_listen.clone();
    let enabled = req.enabled;
    let result = tokio::task::spawn_blocking(move || -> Result<_> {
        let mut tailscale = tailscale
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous_enabled = tailscale.enabled();
        let previous_status = tailscale.status(&web_listen);
        let https_port = tailscale.https_port();
        let status = if enabled {
            tailscale.enable(&web_listen)?
        } else {
            tailscale.disable(&web_listen)?
        };

        if let Err(persist_error) = crate::config::UserConfig::mutate(|config| {
            config.web.tailscale.enabled = Some(enabled);
            config.web.tailscale.https_port = Some(https_port);
            Ok(())
        }) {
            let rollback = if enabled && !previous_status.active {
                tailscale.disable(&web_listen).map(|_| ())
            } else if !enabled && previous_status.active {
                tailscale.enable(&web_listen).map(|_| ())
            } else {
                tailscale.set_enabled(previous_enabled);
                Ok(())
            };
            tailscale.set_enabled(previous_enabled);
            return match rollback {
                Ok(()) => Err(persist_error.context(
                    "Tailscale changed successfully, but pb could not save the setting; the Tailscale change was rolled back",
                )),
                Err(rollback_error) => Err(persist_error.context(format!(
                    "Tailscale changed successfully, but pb could not save the setting; rollback also failed: {rollback_error:#}"
                ))),
            };
        }
        Ok(status)
    })
    .await
    .map_err(|error| {
        IntegrationApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Tailscale settings task failed: {error}"),
        )
    })?;

    result.map(Json).map_err(|error| {
        let message = format!("{error:#}");
        let status = if message.contains("already has a different Serve endpoint")
            || message.contains("not owned by pb")
        {
            StatusCode::CONFLICT
        } else if message.contains("not installed") || message.contains("not connected") {
            StatusCode::SERVICE_UNAVAILABLE
        } else {
            StatusCode::BAD_GATEWAY
        };
        IntegrationApiError::new(status, message)
    })
}

fn web_settings_snapshot(state: &AppState) -> WebSettingsResponse {
    let status = state.sleep_prevention_status();
    WebSettingsResponse {
        prevent_sleep_while_working: status.enabled,
        prevent_sleep_supported: status.supported,
        prevent_sleep_active: status.active,
        prevent_sleep_error: status.error,
    }
}

struct WebEventSink {
    state: AppState,
    session_id: String,
    request_template: AgentRequest,
    persisted_branch: Option<String>,
    persisted_workdir: Option<PathBuf>,
    workflow: Option<crate::workflow::WorkflowCheckpoint>,
    completed_workflows: Vec<crate::workflow::WorkflowSummary>,
    goal_deadline_ms: Option<u64>,
    task_deadline_ms: Option<u64>,
    goal: Option<crate::goal::GoalCheckpoint>,
    completed_goals: Vec<crate::goal::GoalCheckpoint>,
    multi_task: Option<crate::task_queue::MultiTaskCheckpoint>,
    durable: DurableSessionProjection,
    pending_user_messages: Arc<StdMutex<VecDeque<QueuedUserMessage>>>,
    pause_token: Arc<AtomicBool>,
    cancel_token: Arc<AtomicBool>,
    terminal_precursor_keys: Vec<String>,
}

impl WebEventSink {
    fn emit_timestamped(&mut self, event: AgentEvent, supersedes: Vec<String>) -> String {
        let envelope = EventEnvelope::with_timestamp(event);
        let entry_key = envelope.transcript.entry_key.clone();
        let started_location = started_session_location(&envelope.event);
        let metrics = SessionMetricsSnapshot::from_event(&envelope.event);
        let title = match &envelope.event {
            AgentEvent::SessionTitle { title, .. } => {
                let title = title.trim();
                (!title.is_empty()).then(|| title.to_string())
            }
            _ => None,
        };
        let delivery_proposal = if let AgentEvent::DeliveryProposed {
            proposal_id,
            source_turn_id,
            task_summary,
            ..
        } = &envelope.event
        {
            Some(crate::workflow::DeliveryProposal {
                id: proposal_id.clone(),
                source_turn_id: source_turn_id.clone(),
                task_summary: task_summary.clone(),
            })
        } else {
            None
        };
        let goal_proposal = if let AgentEvent::GoalProposed {
            proposal_id,
            source_turn_id,
            objective,
            criteria,
            ..
        } = &envelope.event
        {
            Some(crate::goal::GoalProposal {
                id: proposal_id.clone(),
                source_turn_id: source_turn_id.clone(),
                objective: objective.clone(),
                criteria: criteria.clone(),
            })
        } else {
            None
        };
        let session_id = self.session_id.clone();
        let commit = tokio::runtime::Handle::current().block_on(
            self.state
                .commit_session_change_owned(session_id.clone(), move |session| {
                    apply_started_session_location(session, started_location.as_ref());
                    if let Some(metrics) = metrics.as_ref() {
                        session
                            .runtime
                            .usage_records
                            .lock()
                            .map_err(|_| anyhow::anyhow!("session usage history is unavailable"))?
                            .push(metrics.clone());
                    }
                    if let Some(title) = title.as_ref() {
                        session.title = Some(title.clone());
                    }
                    if let Some(proposal) = delivery_proposal.as_ref() {
                        session.durable.pending_delivery_proposal = Some(proposal.clone());
                    }
                    if let Some(proposal) = goal_proposal.as_ref() {
                        session.durable.pending_goal_proposal = Some(proposal.clone());
                    }
                    session.updated_at_ms = now_millis();
                    if !stage_event_envelope_linked(&session.runtime.history, envelope, supersedes)
                    {
                        bail!("session event entry key already exists");
                    }
                    Ok(StagedSessionChange::new((
                        session.durable.clone(),
                        session.goal.clone(),
                        session.multi_task.clone(),
                        session.branch.clone(),
                        session.workdir.clone(),
                    )))
                }),
        );
        match commit {
            Ok(commit) => {
                let (durable, goal, multi_task, branch, workdir) = commit.value;
                self.durable = durable;
                if self.goal.is_some() {
                    self.goal = goal;
                }
                if self.multi_task.is_some() {
                    self.multi_task = multi_task;
                }
                self.persisted_branch = branch;
                self.persisted_workdir = workdir;
            }
            Err(error) => {
                self.cancel_token.store(true, Ordering::SeqCst);
                tracing::error!(
                    %error,
                    session_id = %session_id,
                    entry_key = %entry_key,
                    "session event was rejected before publication"
                );
                return String::new();
            }
        }
        entry_key
    }
}

fn started_session_location(event: &AgentEvent) -> Option<(String, PathBuf)> {
    let AgentEvent::Started {
        workspace,
        focus_root,
        branch,
        ..
    } = event
    else {
        return None;
    };
    Some((
        branch.clone(),
        PathBuf::from(focus_root.as_deref().unwrap_or(workspace)),
    ))
}

fn apply_started_session_location(
    session: &mut SessionState,
    location: Option<&(String, PathBuf)>,
) {
    if let Some((branch, workdir)) = location {
        session.branch = Some(branch.clone());
        session.workdir = Some(workdir.clone());
    }
}

impl EventSink for WebEventSink {
    fn supports_user_questions(&self) -> bool {
        true
    }

    fn emit(&mut self, event: AgentEvent) {
        self.emit_superseding(event, Vec::new());
    }

    fn emit_keyed(&mut self, event: AgentEvent) -> String {
        self.emit_timestamped(event, Vec::new())
    }

    fn emit_superseding(&mut self, event: AgentEvent, supersedes: Vec<String>) {
        self.emit_timestamped(event, supersedes);
    }

    fn emit_terminal_precursor(&mut self, event: AgentEvent) {
        let entry_key = self.emit_timestamped(event, Vec::new());
        if !entry_key.is_empty() {
            self.terminal_precursor_keys.push(entry_key);
        }
    }

    fn take_terminal_precursor_keys(&mut self) -> Vec<String> {
        std::mem::take(&mut self.terminal_precursor_keys)
    }

    fn has_user_messages(&self) -> bool {
        self.pending_user_messages
            .lock()
            .map(|pending| !pending.is_empty())
            .unwrap_or(true)
    }

    fn take_user_messages(&mut self) -> Vec<QueuedUserMessage> {
        let session_id = self.session_id.clone();
        let commit = tokio::runtime::Handle::current().block_on(
            self.state
                .commit_session_change_owned(session_id.clone(), |session| {
                    let messages = session
                        .runtime
                        .pending_user_messages
                        .lock()
                        .map_err(|_| anyhow::anyhow!("session message queue is unavailable"))?
                        .drain(..)
                        .collect::<Vec<_>>();
                    for message in &messages {
                        stage_event(
                            &session.runtime.history,
                            AgentEvent::UserMessageApplied {
                                message_id: message.message_id.clone(),
                                timestamp_ms: Some(now_millis()),
                            },
                        );
                    }
                    Ok(StagedSessionChange::new(messages))
                }),
        );
        match commit {
            Ok(commit) => commit.value,
            Err(error) => {
                self.cancel_token.store(true, Ordering::SeqCst);
                tracing::error!(
                    %error,
                    session_id = %session_id,
                    "pending user messages could not be durably consumed"
                );
                Vec::new()
            }
        }
    }

    fn seal_user_messages(&mut self) -> bool {
        let session_id = self.session_id.clone();
        match tokio::runtime::Handle::current()
            .block_on(self.state.seal_session_messages_owned(session_id.clone()))
        {
            Ok(sealed) => sealed,
            Err(error) => {
                self.cancel_token.store(true, Ordering::SeqCst);
                tracing::error!(
                    %error,
                    %session_id,
                    "session user-message seal could not cross the coordinator boundary"
                );
                false
            }
        }
    }

    fn open_user_messages(&mut self) {
        let session_id = self.session_id.clone();
        if let Err(error) = tokio::runtime::Handle::current()
            .block_on(self.state.open_session_messages_owned(session_id.clone()))
        {
            self.cancel_token.store(true, Ordering::SeqCst);
            tracing::error!(
                %error,
                %session_id,
                "session user-message opening could not cross the coordinator boundary"
            );
        }
    }

    fn checkpoint_workflow(
        &mut self,
        checkpoint: &crate::workflow::WorkflowCheckpoint,
    ) -> Result<()> {
        checkpoint.validate()?;
        let checkpoint = checkpoint.clone();
        let session_id = self.session_id.clone();
        let commit = tokio::runtime::Handle::current().block_on(
            self.state
                .commit_session_change_owned(session_id, move |session| {
                    if session.multi_task.is_some() && session.goal.is_some() {
                        let mut goal = session
                            .goal
                            .take()
                            .context("Goal Task workflow lost its Goal checkpoint")?
                            .run;
                        goal.checkpoint_active_workflow(checkpoint.clone(), now_millis())?;
                        let goal = crate::goal::GoalCheckpoint::new(goal)?;
                        let mut parent = session
                            .multi_task
                            .take()
                            .context("Goal Task workflow lost its parent checkpoint")?
                            .run;
                        let task_id = parent
                            .active_task_id
                            .clone()
                            .context("Goal Task parent has no active Task")?;
                        let repository = crate::task_queue::TaskRepositoryState::capture(
                            &checkpoint.run.repository.repo_root,
                        )?;
                        parent.apply(crate::task_queue::MultiTaskEvent::ChildCheckpointed {
                            task_id,
                            child: crate::task_queue::TaskChildCheckpoint::Goal(goal.clone()),
                            repository,
                            now_ms: now_millis().max(parent.updated_at_ms),
                        })?;
                        session.goal = Some(goal);
                        session.multi_task =
                            Some(crate::task_queue::MultiTaskCheckpoint::new(parent)?);
                        session.workflow = None;
                    } else if let Some(parent) = session.multi_task.take() {
                        let mut run = parent.run;
                        let task_id = run
                            .active_task_id
                            .clone()
                            .context("multi-Task workflow checkpoint has no active Task")?;
                        let repository = crate::task_queue::TaskRepositoryState::capture(
                            &checkpoint.run.repository.repo_root,
                        )?;
                        let child =
                            crate::task_queue::TaskChildCheckpoint::Build(checkpoint.clone());
                        let event = if run
                            .active_task()
                            .is_some_and(|task| task.workflow.is_none())
                        {
                            crate::task_queue::MultiTaskEvent::ChildStarted {
                                task_id,
                                child,
                                repository,
                                now_ms: now_millis(),
                            }
                        } else {
                            crate::task_queue::MultiTaskEvent::ChildCheckpointed {
                                task_id,
                                child,
                                repository,
                                now_ms: now_millis(),
                            }
                        };
                        run.apply(event)?;
                        session.multi_task =
                            Some(crate::task_queue::MultiTaskCheckpoint::new(run)?);
                        session.workflow = None;
                    } else if let Some(goal) = session.goal.take() {
                        let mut run = goal.run;
                        run.checkpoint_active_workflow(checkpoint.clone(), now_millis())?;
                        session.goal = Some(crate::goal::GoalCheckpoint::new(run)?);
                        session.workflow = None;
                    } else {
                        let summary = crate::workflow::WorkflowSummary::from(&checkpoint.run);
                        if checkpoint.run.stage.is_terminal()
                            && checkpoint.run.stage != crate::workflow::WorkflowStage::Blocked
                        {
                            if session
                                .completed_workflows
                                .last()
                                .is_none_or(|existing| existing.id != summary.id)
                            {
                                session.completed_workflows.push(summary);
                            }
                            session.workflow = None;
                        } else {
                            session.workflow = Some(checkpoint);
                        }
                    }
                    session.request_template.workflow_checkpoint = if session.goal.is_some() {
                        None
                    } else {
                        session.workflow.clone()
                    };
                    session.updated_at_ms = now_millis();
                    Ok(StagedSessionChange::new((
                        session.workflow.clone(),
                        session.completed_workflows.clone(),
                        session.goal.clone(),
                        session.completed_goals.clone(),
                        session.multi_task.clone(),
                    )))
                }),
        )?;
        (
            self.workflow,
            self.completed_workflows,
            self.goal,
            self.completed_goals,
            self.multi_task,
        ) = commit.value;
        Ok(())
    }

    fn checkpoint_multi_task(
        &mut self,
        checkpoint: &crate::task_queue::MultiTaskCheckpoint,
    ) -> Result<()> {
        checkpoint.validate()?;
        let checkpoint = checkpoint.clone();
        let session_id = self.session_id.clone();
        let commit = tokio::runtime::Handle::current().block_on(
            self.state
                .commit_session_change_owned(session_id, move |session| {
                    session.multi_task = Some(checkpoint);
                    session.workflow = None;
                    session.request_template.workflow_checkpoint = None;
                    session.updated_at_ms = now_millis();
                    Ok(StagedSessionChange::new(session.multi_task.clone()))
                }),
        )?;
        self.multi_task = commit.value;
        self.workflow = None;
        Ok(())
    }

    fn should_cancel(&self) -> bool {
        self.cancel_token.load(Ordering::SeqCst)
    }

    fn should_pause(&self) -> bool {
        self.pause_token.load(Ordering::SeqCst)
            || self
                .goal_deadline_ms
                .is_some_and(|deadline| now_millis() >= deadline)
            || self
                .task_deadline_ms
                .is_some_and(|deadline| now_millis() >= deadline)
    }

    fn request_goal_pause(&mut self, reason: &str) -> Result<String> {
        let reason = reason.to_string();
        let session_id = self.session_id.clone();
        let commit = tokio::runtime::Handle::current().block_on(
            self.state
                .commit_session_change_owned(session_id, |session| {
                    let mut run = session
                        .goal
                        .as_ref()
                        .context("goal pause request lost its durable goal")?
                        .run
                        .clone();
                    let goal_id = run.id.clone();
                    let paused = run.request_pause(now_millis())?;
                    session.goal = Some(crate::goal::GoalCheckpoint::new(run)?);
                    sync_multi_task_goal_checkpoint(session)?;
                    session.runtime.pause_token.store(true, Ordering::SeqCst);
                    stage_event(
                        &session.runtime.history,
                        AgentEvent::GoalPauseRequested {
                            goal_id: goal_id.clone(),
                            timestamp_ms: Some(now_millis()),
                        },
                    );
                    if paused {
                        session.status = SessionStatus::Paused;
                        stage_event(
                            &session.runtime.history,
                            AgentEvent::GoalPaused {
                                goal_id: goal_id.clone(),
                                timestamp_ms: Some(now_millis()),
                            },
                        );
                    }
                    session.updated_at_ms = now_millis();
                    Ok(StagedSessionChange::new((
                        goal_id,
                        session.goal.clone(),
                        session.multi_task.clone(),
                    )))
                }),
        )?;
        let (goal_id, goal, multi_task) = commit.value;
        self.goal = goal;
        self.multi_task = multi_task;
        Ok(format!(
            "safe-boundary pause requested for {goal_id}; reason recorded: {reason}"
        ))
    }

    fn request_goal_change(
        &mut self,
        kind: crate::events::GoalChangeKind,
        summary: &str,
    ) -> Result<String> {
        let summary = summary.to_string();
        let session_id = self.session_id.clone();
        let commit = tokio::runtime::Handle::current().block_on(
            self.state
                .commit_session_change_owned(session_id, move |session| {
                    let goal_id = session
                        .goal
                        .as_ref()
                        .context("goal change request lost its durable goal")?
                        .run
                        .id
                        .clone();
                    let pending = PendingGoalChange {
                        goal_id: goal_id.clone(),
                        kind: kind.clone(),
                        summary: summary.clone(),
                    };
                    session.durable.pending_goal_change = Some(pending.clone());
                    stage_event(
                        &session.runtime.history,
                        AgentEvent::GoalChangeRequested {
                            goal_id: goal_id.clone(),
                            kind: kind.clone(),
                            summary: summary.clone(),
                            timestamp_ms: Some(now_millis()),
                        },
                    );
                    let mut run = session
                        .goal
                        .take()
                        .context("goal change request lost its durable goal")?
                        .run;
                    let paused = run.request_pause(now_millis())?;
                    session.goal = Some(crate::goal::GoalCheckpoint::new(run)?);
                    sync_multi_task_goal_checkpoint(session)?;
                    session.runtime.pause_token.store(true, Ordering::SeqCst);
                    stage_event(
                        &session.runtime.history,
                        AgentEvent::GoalPauseRequested {
                            goal_id: goal_id.clone(),
                            timestamp_ms: Some(now_millis()),
                        },
                    );
                    if paused {
                        session.status = SessionStatus::Paused;
                        stage_event(
                            &session.runtime.history,
                            AgentEvent::GoalPaused {
                                goal_id,
                                timestamp_ms: Some(now_millis()),
                            },
                        );
                    }
                    session.updated_at_ms = now_millis();
                    Ok(StagedSessionChange::new((
                        pending,
                        session.goal.clone(),
                        session.multi_task.clone(),
                    )))
                }),
        )?;
        let (pending, goal, multi_task) = commit.value;
        self.durable.pending_goal_change = Some(pending);
        self.goal = goal;
        self.multi_task = multi_task;
        Ok(format!(
            "{kind} request recorded and the goal will pause for human review"
        ))
    }

    fn ask_user(&mut self, question: &str) -> Result<String> {
        self.ask_multiple_choice(question, &[])
    }

    fn ask_multiple_choice(&mut self, question: &str, choices: &[String]) -> Result<String> {
        let question = question.trim().to_string();
        if question.is_empty() {
            anyhow::bail!("ask_user question must not be empty");
        }
        let choices = choices.to_vec();
        let question_id = new_durable_id("question");
        let (tx, rx) = std::sync::mpsc::channel();
        let event = AgentEvent::UserQuestion {
            question_id: question_id.clone(),
            question: question.clone(),
            choices: choices.clone(),
            profile: self.request_template.profile,
            timestamp_ms: Some(now_millis()),
        };

        let session_id = self.session_id.clone();
        tokio::runtime::Handle::current().block_on(self.state.commit_session_change_owned(
            session_id,
            move |session| {
                session.status = SessionStatus::Paused;
                session.runtime.pending_question = Some(PendingQuestionState {
                    question_id: question_id.clone(),
                    question,
                    choices,
                    responder: tx,
                });
                session.updated_at_ms = now_millis();
                stage_event(&session.runtime.history, event);
                Ok(StagedSessionChange::new(()))
            },
        ))?;
        self.state.update_sleep_prevention_working(false);

        rx.recv().map_err(|error| {
            if self.cancel_token.load(Ordering::SeqCst) {
                anyhow::anyhow!("workflow cancellation requested")
            } else {
                anyhow::anyhow!(error)
                    .context("session stopped before the user answered the question")
            }
        })
    }
}

fn session_state_changed_event(session: &SessionState) -> AgentEvent {
    let status = match session.status {
        SessionStatus::Queued => crate::events::SessionLifecycleStatus::Queued,
        SessionStatus::Running => crate::events::SessionLifecycleStatus::Running,
        SessionStatus::Paused => crate::events::SessionLifecycleStatus::Paused,
        SessionStatus::Completed => crate::events::SessionLifecycleStatus::Completed,
        SessionStatus::Failed => crate::events::SessionLifecycleStatus::Failed,
    };
    AgentEvent::SessionStateChanged {
        status,
        timestamp_ms: Some(now_millis()),
    }
}

fn stage_session_state_changed(session: &SessionState) -> String {
    let envelope = EventEnvelope::with_timestamp(session_state_changed_event(session));
    let entry_key = envelope.transcript.entry_key.clone();
    let published = stage_event_envelope_linked(&session.runtime.history, envelope, Vec::new());
    debug_assert!(published, "staged history must accept its lifecycle event");
    entry_key
}

fn publish_terminal_session_transition(
    state: &AppState,
    projects: &[ProjectEntry],
    session_id: &str,
    session: &SessionState,
    entry_key: String,
) {
    let transition =
        AppState::terminal_transition(projects, session_id, entry_key, session.status, session);
    state.publish_project_session_update(Some(transition));
}

fn dispatch_next_session(state: AppState) {
    tokio::spawn(async move {
        let mut retry = 0_u8;
        let (next, working) = loop {
            match state.claim_next_session_owned().await {
                Ok(outcome) => break outcome,
                Err(error) => {
                    retry = retry.saturating_add(1);
                    let delay_ms = 100_u64
                        .saturating_mul(4_u64.saturating_pow(u32::from(retry.saturating_sub(1))))
                        .min(5_000);
                    tracing::warn!(
                        %error,
                        attempt = retry,
                        delay_ms,
                        "queued session dispatch could not cross the durable boundary; retrying"
                    );
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
            }
        };

        state.update_sleep_prevention_working(working);
        let Some(next) = next else {
            return;
        };

        let (session_id, request) = next;
        spawn_agent_run(state, session_id, request);
    });
}

fn spawn_agent_run(state: AppState, session_id: String, request: AgentRequest) {
    tokio::spawn(async move {
        let (
            models_root,
            workflow,
            completed_workflows,
            goal,
            completed_goals,
            multi_task,
            durable,
            pending_user_messages,
            pause_token,
            cancel_token,
        ) = {
            let sessions = state.sessions.lock().await;
            let Some(session) = sessions.get(&session_id) else {
                return;
            };
            (
                request
                    .model_dir
                    .clone()
                    .unwrap_or_else(crate::default_models_dir),
                session.workflow.clone(),
                session.completed_workflows.clone(),
                session.goal.clone(),
                session.completed_goals.clone(),
                session.multi_task.clone(),
                session.durable.clone(),
                Arc::clone(&session.runtime.pending_user_messages),
                Arc::clone(&session.runtime.pause_token),
                Arc::clone(&session.runtime.cancel_token),
            )
        };

        let request_for_run = request.clone();
        let state_for_run = state.clone();
        let session_id_for_run = session_id.clone();
        let result = tokio::task::spawn_blocking(move || {
            let sink = WebEventSink {
                state: state_for_run,
                session_id: session_id_for_run,
                request_template: request_for_run.clone(),
                persisted_branch: request_for_run.branch.clone(),
                persisted_workdir: request_for_run.workdir.clone(),
                workflow,
                completed_workflows,
                goal_deadline_ms: goal.as_ref().map(|checkpoint| {
                    checkpoint.run.created_at_ms.saturating_add(
                        checkpoint
                            .run
                            .budget
                            .wall_time_minutes
                            .saturating_mul(60_000),
                    )
                }),
                task_deadline_ms: multi_task.as_ref().and_then(task_deadline_ms),
                goal,
                completed_goals,
                multi_task,
                durable,
                pending_user_messages,
                pause_token,
                cancel_token,
                terminal_precursor_keys: Vec::new(),
            };
            run_agent_managed(request_for_run.clone(), &models_root, sink)
        })
        .await;

        let completion = match result {
            Ok(Ok(run_result)) => AgentRunCompletion::Finished(run_result),
            Ok(Err(error)) => AgentRunCompletion::RunnerFailed(format!("{error:#}")),
            Err(error) => AgentRunCompletion::TaskFailed(format!("{error:#}")),
        };

        let goal_projects = if matches!(
            &completion,
            AgentRunCompletion::Finished(run_result) if run_result.requested_goal.is_some()
        ) {
            Some(state.projects.lock().await.clone())
        } else {
            None
        };
        let transaction_session_id = session_id.clone();
        let mut retry = 0_u32;
        loop {
            let completion = completion.clone();
            let goal_projects = goal_projects.clone();
            let attempt_session_id = session_id.clone();
            let completion_session_id = attempt_session_id.clone();
            let commit = state
                .commit_session_change_owned(session_id.clone(), move |session| {
                    let mut terminate_environment = false;
                    session.runtime.pending_question = None;
                    session.updated_at_ms = now_millis();
                    let mut final_status = SessionStatus::Completed;
                    match completion {
                        AgentRunCompletion::Finished(run_result) => {
                            if let Some(proposal) = run_result.delivery_proposal.clone() {
                                session.durable.pending_delivery_proposal = Some(proposal);
                            }
                            if let Some(proposal) = run_result.goal_proposal.clone() {
                                session.durable.pending_goal_proposal = Some(proposal);
                            }
                            session.request_template.task_plan_rejected =
                                run_result.task_plan_rejected.clone();
                            session.request_template.task_planning_transcript =
                                run_result.task_planning_transcript.clone();
                            session.request_template.repository_context =
                                run_result.repository_context.clone();
                            session.request_template.workspace_graph =
                                run_result.workspace_graph.clone();
                            session.branch = Some(run_result.branch.clone());
                            session.workdir = Some(run_result.focus_root.clone());
                            if session.multi_task.is_some() {
                                match apply_multi_task_run_result(session, &run_result) {
                                    Ok((status, terminate)) => {
                                        final_status = status;
                                        terminate_environment = terminate;
                                    }
                                    Err(error) => {
                                        final_status = SessionStatus::Failed;
                                        stage_event(
                                            &session.runtime.history,
                                            AgentEvent::Error {
                                                summary: "Task controller failed".to_string(),
                                                detail: format!("{error:#}"),
                                                nesting_depth: None,
                                                timestamp_ms: Some(now_millis()),
                                            },
                                        );
                                    }
                                }
                            } else if session.goal.is_some() {
                                match apply_goal_run_result(session, &run_result) {
                                    Ok((status, terminate)) => {
                                        final_status = status;
                                        terminate_environment = terminate;
                                    }
                                    Err(error) => {
                                        final_status = SessionStatus::Failed;
                                        fail_active_goal_engine(
                                            session,
                                            format!("goal controller failed: {error:#}"),
                                        );
                                        stage_event(
                                            &session.runtime.history,
                                            AgentEvent::Error {
                                                summary: "Goal controller failed".to_string(),
                                                detail: format!("{error:#}"),
                                                nesting_depth: None,
                                                timestamp_ms: Some(now_millis()),
                                            },
                                        );
                                    }
                                }
                            } else if let Some(checkpoint) = run_result.workflow.clone() {
                                if checkpoint.run.stage == crate::workflow::WorkflowStage::Blocked {
                                    session.workflow = Some(checkpoint.clone());
                                    session.request_template.workflow_checkpoint = Some(checkpoint);
                                    final_status = SessionStatus::Failed;
                                } else {
                                    let summary =
                                        crate::workflow::WorkflowSummary::from(&checkpoint.run);
                                    if session
                                        .completed_workflows
                                        .last()
                                        .is_none_or(|existing| existing.id != summary.id)
                                    {
                                        session.completed_workflows.push(summary);
                                    }
                                    session.workflow = None;
                                    session.request_template.workflow_checkpoint = None;
                                }
                            } else if let Some(task_goal) = run_result.task_goal {
                                session.durable.pending_goal_proposal = None;
                                session.durable.pending_goal_change = None;
                                match activate_single_task_goal(
                                    session,
                                    &completion_session_id,
                                    task_goal,
                                ) {
                                    Ok(()) => {
                                        final_status = SessionStatus::Paused;
                                    }
                                    Err(error) => {
                                        final_status = SessionStatus::Failed;
                                        stage_event(
                                            &session.runtime.history,
                                            AgentEvent::Error {
                                                summary: "Goal Task activation failed".to_string(),
                                                detail: format!("{error:#}"),
                                                nesting_depth: None,
                                                timestamp_ms: Some(now_millis()),
                                            },
                                        );
                                    }
                                }
                            } else if let Some(proposal) = run_result.requested_goal {
                                session.durable.pending_goal_proposal = None;
                                session.durable.pending_goal_change = None;
                                match activate_requested_goal(
                                    session,
                                    &completion_session_id,
                                    proposal,
                                    goal_projects.as_deref().unwrap_or(&[]),
                                ) {
                                    Ok(()) => {
                                        final_status = SessionStatus::Paused;
                                    }
                                    Err(error) => {
                                        final_status = SessionStatus::Failed;
                                        stage_event(
                                            &session.runtime.history,
                                            AgentEvent::Error {
                                                summary: "Goal activation failed".to_string(),
                                                detail: format!("{error:#}"),
                                                nesting_depth: None,
                                                timestamp_ms: Some(now_millis()),
                                            },
                                        );
                                    }
                                }
                            } else if let Some(handoff) = run_result.requested_delivery {
                                session.durable.pending_delivery_proposal = None;
                                session.request_template.intent =
                                    Some(crate::workflow::TurnIntent::Deliver);
                                session.request_template.task = handoff.task_summary.clone();
                                session.request_template.conversation_handoff = Some(handoff);
                                session.request_template.workflow_stage = None;
                                session.task = session.request_template.task.clone();
                                final_status = SessionStatus::Queued;
                            }
                            if session.goal.is_none()
                                && session.multi_task.is_none()
                                && run_result.termination_reason
                                    == crate::events::TerminationReason::Cancelled
                            {
                                final_status = SessionStatus::Completed;
                                terminate_environment = true;
                            } else if session.goal.is_none()
                                && session.multi_task.is_none()
                                && (!run_result.reached_final
                                    || run_result.termination_reason
                                        != crate::events::TerminationReason::Final)
                            {
                                final_status = SessionStatus::Failed;
                            }
                        }
                        AgentRunCompletion::RunnerFailed(error) => {
                            final_status = SessionStatus::Failed;
                            fail_active_goal_engine(
                                session,
                                format!("milestone runner failed: {error}"),
                            );
                            stage_event(
                                &session.runtime.history,
                                AgentEvent::Error {
                                    summary: "Session failed".to_string(),
                                    detail: error,
                                    nesting_depth: None,
                                    timestamp_ms: Some(now_millis()),
                                },
                            );
                        }
                        AgentRunCompletion::TaskFailed(error) => {
                            final_status = SessionStatus::Failed;
                            fail_active_goal_engine(
                                session,
                                format!("milestone runner task failed: {error}"),
                            );
                            stage_event(
                                &session.runtime.history,
                                AgentEvent::Error {
                                    summary: "Session failed".to_string(),
                                    detail: error,
                                    nesting_depth: None,
                                    timestamp_ms: Some(now_millis()),
                                },
                            );
                        }
                    }
                    session.runtime.cancel_token.store(false, Ordering::SeqCst);
                    session.runtime.pause_token.store(false, Ordering::SeqCst);
                    session.status = final_status;
                    let change = StagedSessionChange::new(());
                    let change = if terminate_environment {
                        change.terminate_environment()
                    } else {
                        change
                    };
                    Ok(change)
                })
                .await;
            match commit {
                Ok(_) => break,
                Err(error) if error.downcast_ref::<SessionPersistenceError>().is_some() => {
                    retry = retry.saturating_add(1);
                    let delay_ms = 100_u64
                        .saturating_mul(2_u64.saturating_pow(retry.min(6)))
                        .min(5_000);
                    tracing::error!(
                        %error,
                        session_id = %attempt_session_id,
                        retry,
                        delay_ms,
                        "agent completion is waiting for durable session storage"
                    );
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                Err(error) => {
                    tracing::error!(
                        %error,
                        session_id = %transaction_session_id,
                        "agent completion could not cross the durable session boundary"
                    );
                    break;
                }
            }
        }
    });
}

#[derive(Debug, Clone)]
enum AgentRunCompletion {
    Finished(crate::agent_core::AgentRunResult),
    RunnerFailed(String),
    TaskFailed(String),
}

fn activate_single_task_goal(
    session: &mut SessionState,
    session_id: &str,
    projection: crate::task_queue::GoalTaskProjection,
) -> Result<()> {
    let workdir = session
        .workdir
        .clone()
        .context("Goal Task activation requires a repository-backed session")?;
    let root = crate::agent_core::find_git_root(&workdir).unwrap_or(workdir);
    let now = now_millis();
    let run = crate::goal::GoalRun::start(
        projection.goal_id,
        session_id.to_string(),
        projection.objective,
        projection.criteria,
        projection.continuation,
        Some(projection.budget),
        projection.policy,
        root.to_string_lossy(),
        now,
    )?;
    let checkpoint = crate::goal::GoalCheckpoint::new(run)?;
    session.goal = Some(checkpoint.clone());
    session.workflow = None;
    session.request_template.workflow_checkpoint = None;
    session.runtime.pause_token.store(false, Ordering::SeqCst);
    publish_goal_started(session, &checkpoint);
    Ok(())
}

fn configure_multi_task_build_request(
    session: &mut SessionState,
    checkpoint: &crate::task_queue::MultiTaskCheckpoint,
) -> Result<()> {
    checkpoint.validate()?;
    let task = checkpoint
        .run
        .active_task()
        .context("multi-Task run has no active Task to dispatch")?;
    let request = task
        .request
        .as_ref()
        .context("active Task has no controller request")?;
    if request.kind != crate::task_queue::TaskKind::Build {
        bail!("active Task requires Goal dispatch");
    }
    let projection =
        crate::task_queue::project_build_task(request, &checkpoint.run.authority.workflow_policy)?;
    let mut next = session.request_template.clone();
    next.task = projection.task;
    next.turn_id = projection.turn_id;
    next.intent = Some(crate::workflow::TurnIntent::Deliver);
    next.workflow_policy = Some(projection.workflow_policy);
    next.workflow_stage = None;
    next.workflow_checkpoint = task.workflow.clone();
    next.conversation_handoff = Some(projection.handoff);
    next.max_steps = checkpoint
        .run
        .authority
        .request_max_steps
        .min(projection.max_steps)
        .max(1);
    next.profile = crate::agent_core::AgentProfile::Build;
    next.infer_profile = false;
    next.repository_context = None;
    next.prior_check_evidence = crate::checks::CheckEvidenceLedger::default();
    next.goal_context = None;
    next.branch = session.branch.clone();
    next.workdir = session.workdir.clone();
    session.request_template = next;
    session.workflow = None;
    Ok(())
}

fn activate_multi_task_goal(
    session: &mut SessionState,
    checkpoint: &crate::task_queue::MultiTaskCheckpoint,
) -> Result<crate::task_queue::MultiTaskCheckpoint> {
    checkpoint.validate()?;
    let mut parent = checkpoint.run.clone();
    let task = parent
        .active_task()
        .context("multi-Task run has no active Goal Task")?;
    let request = task
        .request
        .as_ref()
        .context("active Goal Task has no controller request")?;
    let projection = crate::task_queue::project_goal_task(request, &parent.authority.goal_policy)?;
    let now = now_millis().max(parent.updated_at_ms);
    let goal = crate::goal::GoalRun::start(
        projection.goal_id,
        parent.session_id.clone(),
        projection.objective,
        projection.criteria,
        projection.continuation,
        Some(projection.budget),
        projection.policy,
        parent.authority.workdir.clone(),
        now,
    )?;
    let goal = crate::goal::GoalCheckpoint::new(goal)?;
    let repository = crate::task_queue::TaskRepositoryState::capture(std::path::Path::new(
        &parent.authority.workdir,
    ))?;
    parent.apply(crate::task_queue::MultiTaskEvent::ChildStarted {
        task_id: request.task_id.clone(),
        child: crate::task_queue::TaskChildCheckpoint::Goal(goal.clone()),
        repository,
        now_ms: now,
    })?;
    let parent = crate::task_queue::MultiTaskCheckpoint::new(parent)?;
    session.goal = Some(goal.clone());
    session.workflow = None;
    session.request_template.workflow_checkpoint = None;
    session.runtime.pause_token.store(false, Ordering::SeqCst);
    publish_goal_started(session, &goal);
    Ok(parent)
}

fn dispatch_multi_task_active(
    session: &mut SessionState,
    checkpoint: &crate::task_queue::MultiTaskCheckpoint,
) -> Result<SessionStatus> {
    let kind = checkpoint
        .run
        .active_task()
        .and_then(|task| task.request.as_ref())
        .map(|request| request.kind)
        .context("multi-Task run has no active request")?;
    match kind {
        crate::task_queue::TaskKind::Build => {
            configure_multi_task_build_request(session, checkpoint)?;
            session.multi_task = Some(checkpoint.clone());
            publish_multi_task_changed(session, checkpoint);
            Ok(SessionStatus::Queued)
        }
        crate::task_queue::TaskKind::Goal => {
            let checkpoint = activate_multi_task_goal(session, checkpoint)?;
            session.multi_task = Some(checkpoint.clone());
            publish_multi_task_changed(session, &checkpoint);
            Ok(SessionStatus::Paused)
        }
    }
}

fn archive_multi_task(
    session: &mut SessionState,
    checkpoint: crate::task_queue::MultiTaskCheckpoint,
) {
    publish_multi_task_changed(session, &checkpoint);
    if session
        .completed_multi_tasks
        .last()
        .is_none_or(|existing| existing.run.id != checkpoint.run.id)
    {
        session.completed_multi_tasks.push(checkpoint);
    }
    session.multi_task = None;
    session.workflow = None;
    session.request_template.workflow_checkpoint = None;
}

fn publish_multi_task_changed(
    session: &SessionState,
    checkpoint: &crate::task_queue::MultiTaskCheckpoint,
) {
    stage_event(
        &session.runtime.history,
        AgentEvent::TasksChanged {
            multi_task_id: checkpoint.run.id.clone(),
            stage: checkpoint.run.stage,
            outcome: checkpoint.run.outcome,
            active_task_id: checkpoint.run.active_task_id.clone(),
            checkpoint_sha256: checkpoint.sha256.clone(),
            timestamp_ms: Some(now_millis()),
        },
    );
}

fn apply_multi_task_run_result(
    session: &mut SessionState,
    run_result: &crate::agent_core::AgentRunResult,
) -> Result<(SessionStatus, bool)> {
    if session.goal.is_some() {
        return apply_multi_task_goal_run_result(session, run_result);
    }
    let checkpoint = session
        .multi_task
        .clone()
        .context("session lost its active multi-Task checkpoint")?;
    checkpoint.validate()?;
    let mut run = checkpoint.run;
    let now = now_millis().max(run.updated_at_ms);

    if run.stage.is_terminal() {
        let status = if run.stage == crate::task_queue::MultiTaskStage::Ready
            || run.stage == crate::task_queue::MultiTaskStage::Cancelled
        {
            SessionStatus::Completed
        } else {
            SessionStatus::Failed
        };
        archive_multi_task(session, crate::task_queue::MultiTaskCheckpoint::new(run)?);
        return Ok((status, status == SessionStatus::Completed));
    }

    let active = run
        .active_task()
        .context("running multi-Task checkpoint has no active Task")?;
    if active.workflow.is_none() && active.goal.is_none() && run_result.workflow.is_none() {
        let checkpoint = crate::task_queue::MultiTaskCheckpoint::new(run)?;
        let status = dispatch_multi_task_active(session, &checkpoint)?;
        return Ok((status, false));
    }

    let task_id = active.spec.id.clone();
    if session.runtime.cancel_token.load(Ordering::SeqCst)
        || run_result.termination_reason == crate::events::TerminationReason::Cancelled
    {
        run.apply(crate::task_queue::MultiTaskEvent::Cancelled {
            reason: "cancelled by user; completed Task commits were preserved".to_string(),
            now_ms: now,
        })?;
        archive_multi_task(session, crate::task_queue::MultiTaskCheckpoint::new(run)?);
        return Ok((SessionStatus::Completed, true));
    }

    let child_wall_ms = session
        .runtime
        .usage_records
        .lock()
        .ok()
        .and_then(|records| records.last().map(|metrics| metrics.wall_runtime_ms))
        .unwrap_or_default();
    if child_wall_ms > 0 {
        run.apply(crate::task_queue::MultiTaskEvent::DirectUsageRecorded {
            task_id: task_id.clone(),
            usage: crate::task_queue::TaskDirectUsage {
                elapsed_ms: child_wall_ms,
                ..crate::task_queue::TaskDirectUsage::default()
            },
            now_ms: now,
        })?;
    }
    if run.stage.is_terminal() {
        archive_multi_task(session, crate::task_queue::MultiTaskCheckpoint::new(run)?);
        return Ok((SessionStatus::Failed, false));
    }

    let workflow = run_result
        .workflow
        .clone()
        .or_else(|| run.active_task().and_then(|task| task.workflow.clone()))
        .context("Build Task ended without a workflow checkpoint")?;
    let summary = crate::workflow::WorkflowSummary::from(&workflow.run);
    if session
        .completed_workflows
        .last()
        .is_none_or(|existing| existing.id != summary.id)
        && workflow.run.stage.is_terminal()
    {
        session.completed_workflows.push(summary);
    }

    if workflow.run.stage == crate::workflow::WorkflowStage::Ready {
        let repository =
            crate::task_queue::TaskRepositoryState::capture(&workflow.run.repository.repo_root)?;
        let request = run
            .active_task()
            .and_then(|task| task.request.as_ref())
            .context("delivered Build Task lost its request")?;
        let result = crate::task_queue::build_task_result(request, &workflow, repository.clone())?;
        run.apply(crate::task_queue::MultiTaskEvent::TaskDelivered {
            task_id,
            result,
            repository: repository.clone(),
            now_ms: now,
        })?;
        run.apply(crate::task_queue::MultiTaskEvent::EvaluationCompleted {
            repository,
            now_ms: now,
        })?;
        let checkpoint = crate::task_queue::MultiTaskCheckpoint::new(run)?;
        match checkpoint.run.stage {
            crate::task_queue::MultiTaskStage::Ready => {
                archive_multi_task(session, checkpoint);
                Ok((SessionStatus::Completed, true))
            }
            crate::task_queue::MultiTaskStage::RunningTask => {
                let status = dispatch_multi_task_active(session, &checkpoint)?;
                Ok((status, false))
            }
            crate::task_queue::MultiTaskStage::Blocked => {
                session.multi_task = Some(checkpoint.clone());
                publish_multi_task_changed(session, &checkpoint);
                Ok((SessionStatus::Failed, false))
            }
            stage => bail!("Task evaluation reached unexpected stage {stage:?}"),
        }
    } else if let Some((disposition, reason)) = crate::task_queue::build_task_stop(&workflow)? {
        run.apply(crate::task_queue::MultiTaskEvent::TaskStopped {
            task_id,
            disposition,
            reason,
            now_ms: now,
        })?;
        let checkpoint = crate::task_queue::MultiTaskCheckpoint::new(run)?;
        session.multi_task = Some(checkpoint.clone());
        publish_multi_task_changed(session, &checkpoint);
        session.workflow = None;
        session.request_template.workflow_checkpoint = None;
        Ok((SessionStatus::Failed, false))
    } else {
        run.apply(crate::task_queue::MultiTaskEvent::TaskStopped {
            task_id,
            disposition: crate::task_queue::TaskStopDisposition::Failed,
            reason: format!(
                "Build Task stopped at {:?}: {}",
                workflow.run.stage, run_result.termination_reason
            ),
            now_ms: now,
        })?;
        let checkpoint = crate::task_queue::MultiTaskCheckpoint::new(run)?;
        archive_multi_task(session, checkpoint);
        Ok((SessionStatus::Failed, false))
    }
}

fn apply_multi_task_goal_run_result(
    session: &mut SessionState,
    run_result: &crate::agent_core::AgentRunResult,
) -> Result<(SessionStatus, bool)> {
    let parent_checkpoint = session
        .multi_task
        .clone()
        .context("Goal Task session lost its parent checkpoint")?;
    let task_id = parent_checkpoint
        .run
        .active_task_id
        .clone()
        .context("Goal Task parent has no active Task")?;
    let child_id = parent_checkpoint
        .run
        .active_task()
        .and_then(|task| task.request.as_ref())
        .map(|request| request.child_id.clone())
        .context("Goal Task parent has no active request")?;

    let (goal_status, _) = apply_goal_run_result(session, run_result)?;
    let goal = session
        .goal
        .clone()
        .or_else(|| {
            session
                .completed_goals
                .iter()
                .rev()
                .find(|checkpoint| checkpoint.run.id == child_id)
                .cloned()
        })
        .context("Goal Task controller did not preserve its Goal checkpoint")?;
    let mut parent = session.multi_task.clone().unwrap_or(parent_checkpoint).run;
    if parent.stage.is_terminal() {
        if let Some(active_goal) = session.goal.take() {
            let mut goal_run = active_goal.run;
            if !goal_run.stage.is_terminal() {
                goal_run.cancel(now_millis().max(goal_run.updated_at_ms));
            }
            session
                .completed_goals
                .push(crate::goal::GoalCheckpoint::new(goal_run)?);
        }
        archive_multi_task(
            session,
            crate::task_queue::MultiTaskCheckpoint::new(parent)?,
        );
        return Ok((SessionStatus::Failed, false));
    }

    let repository = crate::task_queue::TaskRepositoryState::capture(std::path::Path::new(
        &parent.authority.workdir,
    ))?;
    let mut now = now_millis().max(parent.updated_at_ms);
    parent.apply(crate::task_queue::MultiTaskEvent::ChildCheckpointed {
        task_id: task_id.clone(),
        child: crate::task_queue::TaskChildCheckpoint::Goal(goal.clone()),
        repository: repository.clone(),
        now_ms: now,
    })?;
    let child_wall_ms = session
        .runtime
        .usage_records
        .lock()
        .ok()
        .and_then(|records| records.last().map(|metrics| metrics.wall_runtime_ms))
        .unwrap_or_default();
    if child_wall_ms > 0 && !parent.stage.is_terminal() {
        now = now_millis().max(parent.updated_at_ms);
        parent.apply(crate::task_queue::MultiTaskEvent::DirectUsageRecorded {
            task_id: task_id.clone(),
            usage: crate::task_queue::TaskDirectUsage {
                elapsed_ms: child_wall_ms,
                ..crate::task_queue::TaskDirectUsage::default()
            },
            now_ms: now,
        })?;
    }
    if parent.stage.is_terminal() {
        archive_multi_task(
            session,
            crate::task_queue::MultiTaskCheckpoint::new(parent)?,
        );
        return Ok((SessionStatus::Failed, false));
    }

    match goal.run.stage {
        crate::goal::GoalStage::Completed => {
            let request = parent
                .active_task()
                .and_then(|task| task.request.as_ref())
                .context("completed Goal Task lost its request")?;
            let result = crate::task_queue::goal_task_result(request, &goal, repository.clone())?;
            now = now_millis().max(parent.updated_at_ms);
            parent.apply(crate::task_queue::MultiTaskEvent::TaskDelivered {
                task_id,
                result,
                repository: repository.clone(),
                now_ms: now,
            })?;
            now = now_millis().max(parent.updated_at_ms);
            parent.apply(crate::task_queue::MultiTaskEvent::EvaluationCompleted {
                repository,
                now_ms: now,
            })?;
            let checkpoint = crate::task_queue::MultiTaskCheckpoint::new(parent)?;
            match checkpoint.run.stage {
                crate::task_queue::MultiTaskStage::Ready => {
                    archive_multi_task(session, checkpoint);
                    Ok((SessionStatus::Completed, true))
                }
                crate::task_queue::MultiTaskStage::RunningTask => {
                    let status = dispatch_multi_task_active(session, &checkpoint)?;
                    Ok((status, false))
                }
                stage => bail!("Goal Task evaluation reached unexpected stage {stage:?}"),
            }
        }
        crate::goal::GoalStage::Blocked => {
            let (_, reason) = crate::task_queue::goal_task_stop(&goal)?
                .context("blocked Goal Task has no stop disposition")?;
            now = now_millis().max(parent.updated_at_ms);
            parent.apply(crate::task_queue::MultiTaskEvent::TaskStopped {
                task_id,
                disposition: crate::task_queue::TaskStopDisposition::Blocked,
                reason,
                now_ms: now,
            })?;
            let checkpoint = crate::task_queue::MultiTaskCheckpoint::new(parent)?;
            session.multi_task = Some(checkpoint.clone());
            publish_multi_task_changed(session, &checkpoint);
            Ok((SessionStatus::Paused, false))
        }
        crate::goal::GoalStage::Failed => {
            let (_, reason) = crate::task_queue::goal_task_stop(&goal)?
                .context("failed Goal Task has no stop disposition")?;
            now = now_millis().max(parent.updated_at_ms);
            parent.apply(crate::task_queue::MultiTaskEvent::TaskStopped {
                task_id,
                disposition: crate::task_queue::TaskStopDisposition::Failed,
                reason,
                now_ms: now,
            })?;
            archive_multi_task(
                session,
                crate::task_queue::MultiTaskCheckpoint::new(parent)?,
            );
            Ok((SessionStatus::Failed, false))
        }
        crate::goal::GoalStage::Cancelled => {
            now = now_millis().max(parent.updated_at_ms);
            parent.apply(crate::task_queue::MultiTaskEvent::Cancelled {
                reason: "Goal Task was cancelled; completed Task commits were preserved"
                    .to_string(),
                now_ms: now,
            })?;
            archive_multi_task(
                session,
                crate::task_queue::MultiTaskCheckpoint::new(parent)?,
            );
            Ok((SessionStatus::Completed, true))
        }
        _ => {
            let checkpoint = crate::task_queue::MultiTaskCheckpoint::new(parent)?;
            session.multi_task = Some(checkpoint.clone());
            publish_multi_task_changed(session, &checkpoint);
            Ok((goal_status, false))
        }
    }
}

fn activate_requested_goal(
    session: &mut SessionState,
    session_id: &str,
    proposal: crate::goal::GoalProposal,
    registered_projects: &[ProjectEntry],
) -> Result<()> {
    let workdir = session
        .workdir
        .clone()
        .context("goal activation requires a repository-backed session")?;
    ensure_goal_workdir_registered(registered_projects, &workdir)?;
    let now = now_millis();
    let run = crate::goal::GoalRun::start(
        new_durable_id("goal"),
        session_id.to_string(),
        proposal.objective.clone(),
        proposal.criteria,
        crate::goal::GoalContinuationPolicy::ReviewPlanThenAutomatic,
        None,
        goal_policy_for_request(Some(&workdir))?,
        workdir.to_string_lossy(),
        now,
    )?;
    let checkpoint = crate::goal::GoalCheckpoint::new(run)?;
    session.task = proposal.objective.clone();
    session.title = Some(proposal.objective.clone());
    session.request_template.task = proposal.objective;
    session.request_template.intent = Some(crate::workflow::TurnIntent::Discuss);
    session.request_template.workflow_stage = None;
    session.request_template.workflow_checkpoint = None;
    session.goal = Some(checkpoint.clone());
    session.runtime.pause_token.store(false, Ordering::SeqCst);
    publish_goal_started(session, &checkpoint);
    Ok(())
}

fn apply_goal_run_result(
    session: &mut SessionState,
    run_result: &crate::agent_core::AgentRunResult,
) -> Result<(SessionStatus, bool)> {
    let now = now_millis();
    let mut run = session
        .goal
        .as_ref()
        .context("goal session lost its active checkpoint")?
        .run
        .clone();
    let cancel_requested = session.runtime.cancel_token.load(Ordering::SeqCst)
        || run_result.termination_reason == crate::events::TerminationReason::Cancelled;
    if cancel_requested {
        run.cancel(now);
        let checkpoint = crate::goal::GoalCheckpoint::new(run)?;
        stage_event(
            &session.runtime.history,
            AgentEvent::GoalCancelled {
                goal_id: checkpoint.run.id.clone(),
                checkpoint_sha256: checkpoint.sha256.clone(),
                timestamp_ms: Some(now),
            },
        );
        session.completed_goals.push(checkpoint);
        session.goal = None;
        session.request_template.workflow_checkpoint = None;
        return Ok((SessionStatus::Completed, true));
    }

    let Some(workflow) = run_result.workflow.clone() else {
        run.fail_external(
            crate::goal::GoalOutcome::EngineError,
            format!(
                "milestone runner ended without a workflow checkpoint: {}",
                run_result.termination_reason
            ),
            now,
        );
        let checkpoint = crate::goal::GoalCheckpoint::new(run)?;
        publish_goal_failed(session, &checkpoint);
        session.completed_goals.push(checkpoint);
        session.goal = None;
        return Ok((SessionStatus::Failed, false));
    };

    if run.wall_time_exhausted(now) && !workflow.run.stage.is_terminal() {
        run.checkpoint_active_workflow(workflow, now)?;
        run.fail_external(
            crate::goal::GoalOutcome::BudgetExhausted,
            "goal wall-time budget was reached at a safe workflow boundary",
            now,
        );
        let checkpoint = crate::goal::GoalCheckpoint::new(run)?;
        publish_goal_failed(session, &checkpoint);
        session.completed_goals.push(checkpoint);
        session.goal = None;
        session.request_template.workflow_checkpoint = None;
        return Ok((SessionStatus::Failed, false));
    }

    if run.pause_requested && !workflow.run.stage.is_terminal() {
        run.checkpoint_active_workflow(workflow, now)?;
        run.pause_at_boundary(now)?;
        let checkpoint = crate::goal::GoalCheckpoint::new(run)?;
        stage_event(
            &session.runtime.history,
            AgentEvent::GoalPaused {
                goal_id: checkpoint.run.id.clone(),
                timestamp_ms: Some(now),
            },
        );
        session.goal = Some(checkpoint);
        session.request_template.workflow_checkpoint = None;
        return Ok((SessionStatus::Paused, false));
    }

    if workflow.run.stage == crate::workflow::WorkflowStage::Blocked {
        let reason = workflow
            .run
            .blocked_reason
            .clone()
            .unwrap_or_else(|| "strict milestone workflow is blocked".to_string());
        run.block_active_workflow(workflow, reason, now)?;
        session.goal = Some(crate::goal::GoalCheckpoint::new(run)?);
        session.request_template.workflow_checkpoint = None;
        return Ok((SessionStatus::Paused, false));
    }

    let milestone_id = run
        .active_milestone_id
        .clone()
        .context("goal workflow completed without an active milestone")?;
    let workflow_id = workflow.run.id.clone();
    run.finish_active_workflow(workflow, now)?;
    stage_event(
        &session.runtime.history,
        AgentEvent::GoalMilestoneCompleted {
            goal_id: run.id.clone(),
            milestone_id,
            workflow_id,
            timestamp_ms: Some(now),
        },
    );
    let checkpoint = crate::goal::GoalCheckpoint::new(run)?;
    match checkpoint.run.stage {
        crate::goal::GoalStage::RunningMilestone => {
            configure_goal_milestone_request(session, &checkpoint.run)?;
            publish_current_goal_milestone(session, &checkpoint.run);
            session.goal = Some(checkpoint);
            Ok((SessionStatus::Queued, false))
        }
        crate::goal::GoalStage::Paused => {
            stage_event(
                &session.runtime.history,
                AgentEvent::GoalPaused {
                    goal_id: checkpoint.run.id.clone(),
                    timestamp_ms: Some(now),
                },
            );
            session.goal = Some(checkpoint);
            session.request_template.workflow_checkpoint = None;
            Ok((SessionStatus::Paused, false))
        }
        crate::goal::GoalStage::AwaitingUserReview => {
            stage_event(
                &session.runtime.history,
                AgentEvent::GoalReadyForReview {
                    goal_id: checkpoint.run.id.clone(),
                    checkpoint_sha256: checkpoint.sha256.clone(),
                    timestamp_ms: Some(now),
                },
            );
            session.goal = Some(checkpoint);
            session.request_template.workflow_checkpoint = None;
            Ok((SessionStatus::Paused, false))
        }
        crate::goal::GoalStage::Completed => {
            let basis = checkpoint
                .run
                .completion_basis
                .unwrap_or(crate::goal::GoalCompletionBasis::MachineVerified);
            stage_event(
                &session.runtime.history,
                AgentEvent::GoalCompleted {
                    goal_id: checkpoint.run.id.clone(),
                    outcome: crate::goal::GoalOutcome::Complete,
                    completion_basis: basis,
                    checkpoint_sha256: checkpoint.sha256.clone(),
                    timestamp_ms: Some(now),
                },
            );
            session.completed_goals.push(checkpoint);
            session.goal = None;
            session.durable.pending_goal_change = None;
            session.request_template.workflow_checkpoint = None;
            Ok((SessionStatus::Completed, false))
        }
        crate::goal::GoalStage::Failed => {
            publish_goal_failed(session, &checkpoint);
            session.completed_goals.push(checkpoint);
            session.goal = None;
            session.durable.pending_goal_change = None;
            session.request_template.workflow_checkpoint = None;
            Ok((SessionStatus::Failed, false))
        }
        stage => {
            session.goal = Some(checkpoint);
            bail!("goal controller produced unsupported post-workflow stage {stage:?}")
        }
    }
}

fn fail_active_goal_engine(session: &mut SessionState, reason: String) {
    let Some(checkpoint) = session.goal.take() else {
        return;
    };
    session.durable.pending_goal_change = None;
    let mut run = checkpoint.run;
    run.fail_external(crate::goal::GoalOutcome::EngineError, reason, now_millis());
    match crate::goal::GoalCheckpoint::new(run) {
        Ok(checkpoint) => {
            publish_goal_failed(session, &checkpoint);
            session.completed_goals.push(checkpoint);
            session.request_template.workflow_checkpoint = None;
        }
        Err(error) => {
            tracing::error!(%error, "failed to terminalize goal after controller error");
        }
    }
}

fn publish_goal_failed(session: &SessionState, checkpoint: &crate::goal::GoalCheckpoint) {
    stage_event(
        &session.runtime.history,
        AgentEvent::GoalFailed {
            goal_id: checkpoint.run.id.clone(),
            outcome: checkpoint
                .run
                .outcome
                .unwrap_or(crate::goal::GoalOutcome::EngineError),
            reason: checkpoint
                .run
                .blocked_reason
                .clone()
                .unwrap_or_else(|| "goal failed".to_string()),
            checkpoint_sha256: checkpoint.sha256.clone(),
            timestamp_ms: Some(now_millis()),
        },
    );
}

async fn session_events(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    headers: HeaderMap,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let last_event_id = headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let (receiver, replay, covered_keys, snapshot) = {
        let sessions = state.sessions.lock().await;
        let session = sessions.get(&id).ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "session_not_found",
                format!("session not found: {id}"),
            )
        })?;
        let receiver = session.runtime.sender.subscribe();
        let history = session.runtime.history.lock().map_err(|_| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "session_history_unavailable",
                "session history is unavailable",
            )
        })?;
        let window = session_replay_window(&history, last_event_id.as_deref());
        let replay = history[window.start..].to_vec();
        let covered_keys = history
            .iter()
            .map(|envelope| envelope.transcript.entry_key.clone())
            .collect::<HashSet<_>>();
        let snapshot = SessionStreamSnapshot {
            session: session_details_from_history(&id, session, &history),
            reset_history: window.reset_history,
            warnings: Vec::new(),
        };
        (receiver, replay, covered_keys, snapshot)
    };

    let covered_keys = Arc::new(StdMutex::new(covered_keys));
    let initial = futures::stream::iter(
        replay
            .into_iter()
            .filter_map(session_event_sse_event)
            .map(Ok),
    )
    .chain(futures::stream::iter(
        session_snapshot_sse_event(&snapshot).map(Ok),
    ));
    let live_covered_keys = Arc::clone(&covered_keys);
    let live = BroadcastStream::new(receiver)
        .take_while(|message| futures::future::ready(message.is_ok()))
        .filter_map(move |message| {
            let covered_keys = Arc::clone(&live_covered_keys);
            async move {
                let publication = message.ok()?;
                let mut events = Vec::new();
                match publication {
                    SessionStreamPublication::Event(envelope) => {
                        let entry_key = envelope.transcript.entry_key.clone();
                        let was_covered = covered_keys
                            .lock()
                            .map(|mut keys| keys.remove(&entry_key))
                            .unwrap_or(false);
                        if !was_covered && let Some(event) = session_event_sse_event(envelope) {
                            events.push(Ok(event));
                        }
                    }
                    SessionStreamPublication::Snapshot(snapshot) => {
                        if let Some(event) = session_snapshot_sse_event(&snapshot) {
                            events.push(Ok(event));
                        }
                    }
                    SessionStreamPublication::Finished => {}
                }
                (!events.is_empty()).then_some(futures::stream::iter(events))
            }
        })
        .flatten();
    let stream = initial.chain(live);

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

fn session_event_sse_event(envelope: EventEnvelope) -> Option<Event> {
    let data = serde_json::to_string(&envelope).ok()?;
    Some(
        Event::default()
            .id(envelope.transcript.entry_key)
            .data(data),
    )
}

fn session_snapshot_sse_event(snapshot: &SessionStreamSnapshot) -> Option<Event> {
    let data = serde_json::to_string(snapshot).ok()?;
    let mut event = Event::default().event("session_snapshot").data(data);
    if let Some(entry_key) = snapshot
        .session
        .events
        .last()
        .map(|envelope| envelope.transcript.entry_key.as_str())
    {
        event = event.id(entry_key);
    }
    Some(event)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SessionReplayWindow {
    start: usize,
    reset_history: bool,
}

fn session_replay_window(
    history: &[EventEnvelope],
    last_event_id: Option<&str>,
) -> SessionReplayWindow {
    match last_event_id {
        Some(entry_key) => history
            .iter()
            .rposition(|envelope| envelope.transcript.entry_key == entry_key)
            .map_or(
                SessionReplayWindow {
                    start: history.len(),
                    reset_history: true,
                },
                |index| SessionReplayWindow {
                    start: index + 1,
                    reset_history: false,
                },
            ),
        None => SessionReplayWindow {
            start: history.len(),
            reset_history: true,
        },
    }
}

async fn spawn_unix_rpc_server(
    socket_path: PathBuf,
    state: AppState,
    defaults: AgentRequest,
) -> Result<()> {
    if let Some(parent) = socket_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    if socket_path.exists() {
        let _ = tokio::fs::remove_file(&socket_path).await;
    }
    let listener = tokio::net::UnixListener::bind(&socket_path)
        .with_context(|| format!("failed to bind unix socket {}", socket_path.display()))?;
    println!("pb daemon socket listening on {}", socket_path.display());

    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let state = state.clone();
                    let defaults = defaults.clone();
                    tokio::spawn(async move {
                        if let Err(err) = handle_rpc_connection(stream, state, defaults).await {
                            eprintln!("pb daemon rpc error: {err:#}");
                        }
                    });
                }
                Err(err) => eprintln!("pb daemon socket accept error: {err}"),
            }
        }
    });

    Ok(())
}

async fn handle_rpc_connection(
    stream: tokio::net::UnixStream,
    state: AppState,
    defaults: AgentRequest,
) -> Result<()> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line).await? == 0 {
        return Ok(());
    }
    let request: RpcRequest = serde_json::from_str(&line).context("invalid daemon request")?;
    let RpcRequest { id, method, params } = request;
    if method == "pb.session.watch" {
        match serde_json::from_value::<WatchSessionRequest>(params) {
            Ok(params) => {
                handle_session_watch(reader.get_mut(), id, state, params.session_id).await?
            }
            Err(error) => write_rpc_error(reader.get_mut(), id, error.to_string()).await?,
        }
        return Ok(());
    }

    match dispatch_unary_rpc(state, defaults, &method, params).await {
        Ok(result) => write_json_line(reader.get_mut(), &RpcFrame::Response { id, result }).await?,
        Err(error) => write_rpc_error(reader.get_mut(), id, format!("{error:#}")).await?,
    }
    Ok(())
}

async fn dispatch_unary_rpc(
    state: AppState,
    defaults: AgentRequest,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value> {
    let result = match method {
        "pb.session.start" => {
            let params: StartSessionRequest = serde_json::from_value(params)?;
            serde_json::to_value(start_session_inner(state, defaults, params).await?)?
        }
        "pb.goal.start" => {
            let params: StartGoalRequest = serde_json::from_value(params)?;
            serde_json::to_value(start_goal_inner(state, defaults, params).await?)?
        }
        "pb.goal.get" => {
            let params: GoalLookupRpcRequest = serde_json::from_value(params)?;
            match get_goal(Path(params.goal_id), State((state, defaults))).await {
                Ok(Json(result)) => serde_json::to_value(result)?,
                Err(error) => return Err(rpc_api_error("goal request", error)),
            }
        }
        "pb.goal.pause" | "pb.goal.resume" | "pb.goal.cancel" | "pb.goal.accept" => {
            let params: GoalRpcMutationRequest = serde_json::from_value(params)?;
            let goal_id = params.goal_id;
            let digest = GoalDigestRequest {
                goal_sha256: params.goal_sha256,
                plan_sha256: params.plan_sha256,
            };
            let result = match method {
                "pb.goal.pause" => pause_goal_inner(&state, &goal_id, &digest).await,
                "pb.goal.resume" => resume_goal_inner(&state, &goal_id, &digest).await,
                "pb.goal.cancel" => cancel_goal_inner(&state, &goal_id, &digest).await,
                "pb.goal.accept" => accept_goal_inner(&state, &goal_id, &digest).await,
                _ => unreachable!(),
            };
            match result {
                Ok(result) => serde_json::to_value(result)?,
                Err(error) => return Err(rpc_api_error("goal request", error)),
            }
        }
        "pb.session.list" => {
            let _: EmptyRpcRequest = serde_json::from_value(params)?;
            serde_json::to_value(session_list_snapshot(&state).await)?
        }
        "pb.projects.add" => {
            let params: AddProjectRequest = serde_json::from_value(params)?;
            serde_json::to_value(
                state
                    .commit_project_registry_owned(UsageWindow::current_utc_day(), move || {
                        projects::add_project(params)
                    })
                    .await?,
            )?
        }
        "pb.projects.list" => {
            let _: EmptyRpcRequest = serde_json::from_value(params)?;
            reload_projects(&state).await?;
            serde_json::to_value(project_list_snapshot(&state).await)?
        }
        "pb.projects.rm" => {
            let params: RemoveProjectRequest = serde_json::from_value(params)?;
            serde_json::to_value(
                state
                    .commit_project_registry_owned(UsageWindow::current_utc_day(), move || {
                        projects::remove_project(&params.name)
                    })
                    .await?,
            )?
        }
        "pb.projects.notifications" => {
            let params: ProjectNotificationsRpcRequest = serde_json::from_value(params)?;
            let name = params.name;
            let notify_on_finish = params.notify_on_finish;
            let result = state
                .commit_project_registry_owned(UsageWindow::current_utc_day(), move || {
                    projects::set_project_notifications(&name, notify_on_finish)
                })
                .await?;
            serde_json::to_value(result)?
        }
        "pb.session.resume" => {
            let params: WatchSessionRequest = serde_json::from_value(params)?;
            serde_json::to_value(resume_session_inner(state, params.session_id).await?)?
        }
        "pb.session.answer" => {
            let params: AnswerSessionQuestionRequest = serde_json::from_value(params)?;
            serde_json::to_value(
                answer_question_inner(
                    state,
                    params.session_id,
                    AnswerQuestionRequest {
                        question_id: params.question_id,
                        answer: params.answer,
                    },
                )
                .await?,
            )?
        }
        "pb.session.delete" => {
            let params: WatchSessionRequest = serde_json::from_value(params)?;
            serde_json::to_value(delete_session_inner(state, &params.session_id).await?)?
        }
        "pb.session.get" => {
            let params: WatchSessionRequest = serde_json::from_value(params)?;
            serde_json::to_value(
                session_details_snapshot(&state, &params.session_id)
                    .await
                    .with_context(|| format!("session not found: {}", params.session_id))?,
            )?
        }
        other => bail!("unknown method: {other}"),
    };
    Ok(result)
}

fn rpc_api_error(context: &str, error: ApiError) -> anyhow::Error {
    anyhow::anyhow!(
        "{context} failed with HTTP status {}: {}",
        error.status.as_u16(),
        error.message
    )
}

async fn write_rpc_response<T: Serialize>(
    stream: &mut tokio::net::UnixStream,
    id: u64,
    result: T,
) -> Result<()> {
    write_json_line(stream, &RpcFrame::response(id, result)?).await
}

async fn write_rpc_error(
    stream: &mut tokio::net::UnixStream,
    id: u64,
    error: String,
) -> Result<()> {
    write_json_line(stream, &RpcFrame::Error { id, error }).await
}

async fn write_json_line<T: Serialize>(
    stream: &mut tokio::net::UnixStream,
    value: &T,
) -> Result<()> {
    let mut data = serde_json::to_vec(value)?;
    data.push(b'\n');
    stream.write_all(&data).await?;
    stream.flush().await?;
    Ok(())
}

async fn handle_session_watch(
    stream: &mut tokio::net::UnixStream,
    id: u64,
    state: AppState,
    session_id: String,
) -> Result<()> {
    let prepared: Result<_> = {
        let sessions = state.sessions.lock().await;
        (|| {
            let session = sessions
                .get(&session_id)
                .ok_or_else(|| anyhow::anyhow!("session not found: {session_id}"))?;
            let history =
                session.runtime.history.lock().map_err(|_| {
                    anyhow::anyhow!("session history lock is poisoned: {session_id}")
                })?;
            // Publishers append while holding this same lock and only then broadcast. Subscribing
            // before cloning the locked history makes the snapshot/live handoff atomic.
            let receiver = session.runtime.sender.subscribe();
            Ok((receiver, history.clone(), session_watch_active(session)))
        })()
    };
    let (receiver, history, active) = match prepared {
        Ok(prepared) => prepared,
        Err(error) => {
            write_rpc_error(stream, id, error.to_string()).await?;
            return Ok(());
        }
    };

    write_rpc_response(
        stream,
        id,
        WatchSessionAcknowledgement {
            session_id: session_id.clone(),
        },
    )
    .await?;

    if let Err(error) =
        stream_session_watch(stream, &state, &session_id, receiver, history, active).await
    {
        let frame = RpcFrame::StreamError {
            session_id,
            error: format!("{error:#}"),
        };
        // A failed socket cannot receive its terminal error, but we must never fall back to a
        // second response frame after the watch acknowledgement.
        let _ = write_json_line(stream, &frame).await;
    }
    Ok(())
}

async fn stream_session_watch(
    stream: &mut tokio::net::UnixStream,
    state: &AppState,
    session_id: &str,
    mut receiver: broadcast::Receiver<SessionStreamPublication>,
    history: Vec<EventEnvelope>,
    active: bool,
) -> Result<()> {
    let mut last_sequence = 0;
    let initial_replay = terminal_replay_after(&history, last_sequence)?;
    write_terminal_replay(stream, session_id, &mut last_sequence, initial_replay).await?;

    if !active {
        write_session_finished(stream, session_id).await?;
        return Ok(());
    }

    loop {
        match receiver.recv().await {
            Ok(SessionStreamPublication::Snapshot(_)) => {}
            Ok(SessionStreamPublication::Event(envelope))
                if envelope.transcript.sequence <= last_sequence => {}
            Ok(SessionStreamPublication::Event(envelope))
                if envelope.transcript.sequence == last_sequence.saturating_add(1) =>
            {
                last_sequence = envelope.transcript.sequence;
                write_session_event(stream, session_id, envelope).await?;
            }
            Ok(SessionStreamPublication::Event(_))
            | Err(broadcast::error::RecvError::Lagged(_)) => {
                let (is_active, replay) =
                    terminal_session_snapshot(state, session_id, last_sequence).await?;
                write_terminal_replay(stream, session_id, &mut last_sequence, replay).await?;
                if !is_active {
                    write_session_finished(stream, session_id).await?;
                    break;
                }
            }
            Ok(SessionStreamPublication::Finished) => {
                let (_, replay) =
                    terminal_session_snapshot(state, session_id, last_sequence).await?;
                write_terminal_replay(stream, session_id, &mut last_sequence, replay).await?;
                write_session_finished(stream, session_id).await?;
                break;
            }
            Err(broadcast::error::RecvError::Closed) => {
                let (is_active, replay) =
                    terminal_session_snapshot(state, session_id, last_sequence).await?;
                write_terminal_replay(stream, session_id, &mut last_sequence, replay).await?;
                if is_active {
                    bail!("session event stream closed while its watch is active: {session_id}");
                }
                write_session_finished(stream, session_id).await?;
                break;
            }
        }
    }

    Ok(())
}

async fn write_session_finished(
    stream: &mut tokio::net::UnixStream,
    session_id: &str,
) -> Result<()> {
    write_json_line(
        stream,
        &RpcFrame::SessionFinished {
            session_id: session_id.to_string(),
        },
    )
    .await
}

async fn write_session_event(
    stream: &mut tokio::net::UnixStream,
    session_id: &str,
    envelope: EventEnvelope,
) -> Result<()> {
    write_json_line(
        stream,
        &RpcFrame::SessionEvent {
            session_id: session_id.to_string(),
            event: envelope,
        },
    )
    .await
}

#[derive(Debug, Clone)]
enum TerminalReplay {
    Delta(Vec<EventEnvelope>),
    Reset {
        after_sequence: u64,
        events: Vec<EventEnvelope>,
    },
}

async fn write_terminal_replay(
    stream: &mut tokio::net::UnixStream,
    session_id: &str,
    last_sequence: &mut u64,
    replay: TerminalReplay,
) -> Result<()> {
    match replay {
        TerminalReplay::Delta(events) => {
            for envelope in events {
                *last_sequence = envelope.transcript.sequence;
                write_session_event(stream, session_id, envelope).await?;
            }
        }
        TerminalReplay::Reset {
            after_sequence,
            events,
        } => {
            *last_sequence = events
                .last()
                .map_or(after_sequence, |event| event.transcript.sequence);
            write_json_line(
                stream,
                &RpcFrame::ReplayReset {
                    session_id: session_id.to_string(),
                    after_sequence,
                    events,
                },
            )
            .await?;
        }
    }
    Ok(())
}

async fn terminal_session_snapshot(
    state: &AppState,
    session_id: &str,
    last_sequence: u64,
) -> Result<(bool, TerminalReplay)> {
    let sessions = state.sessions.lock().await;
    let session = sessions
        .get(session_id)
        .ok_or_else(|| anyhow::anyhow!("session not found: {session_id}"))?;
    let history = session
        .runtime
        .history
        .lock()
        .map_err(|_| anyhow::anyhow!("session history lock is poisoned: {session_id}"))?;
    let replay = terminal_replay_after(&history, last_sequence)?;
    let is_active = session_watch_active(session);
    Ok((is_active, replay))
}

fn terminal_replay_after(history: &[EventEnvelope], last_sequence: u64) -> Result<TerminalReplay> {
    let replay = history
        .iter()
        .filter(|envelope| envelope.transcript.sequence > last_sequence)
        .cloned()
        .collect::<Vec<_>>();
    let starts_after_gap = replay
        .first()
        .is_some_and(|first| first.transcript.sequence != last_sequence.saturating_add(1));
    let contains_gap = replay
        .windows(2)
        .any(|pair| pair[1].transcript.sequence != pair[0].transcript.sequence.saturating_add(1));
    if starts_after_gap || contains_gap {
        return Ok(TerminalReplay::Reset {
            after_sequence: last_sequence,
            events: replay,
        });
    }
    Ok(TerminalReplay::Delta(replay))
}

fn restore_sessions(project_entries: &[ProjectEntry]) -> HashMap<String, SessionState> {
    session_store::restore_registered_sessions(project_entries)
        .into_iter()
        .map(session_from_persisted)
        .map(|(session_id, mut session)| {
            apply_intrinsic_controller_actions(&mut session.request_template);
            (session_id, session)
        })
        .collect()
}

fn apply_intrinsic_controller_actions(request: &mut AgentRequest) {
    request.observation_rendering = crate::workflow::ObservationRendering::ControllerBlock;
}

fn session_from_persisted(mut persisted: PersistedSession) -> (String, SessionState) {
    let sender = SessionEventSender::new(256);
    let session_id = persisted.session_id.clone();
    let title = persisted.title;
    let usage_records = persisted.usage_records.clone();
    let metrics = persisted.metrics.clone();
    let pending_user_messages = std::mem::take(&mut persisted.pending_user_messages);
    let durable = DurableSessionProjection {
        project: persisted.project,
        pending_delivery_proposal: persisted.pending_delivery_proposal,
        pending_goal_proposal: persisted.pending_goal_proposal,
        pending_goal_change: persisted.pending_goal_change,
    };
    let history = Arc::new(StdMutex::new(persisted.events));
    let pending_user_messages =
        Arc::new(StdMutex::new(pending_user_messages.into_iter().collect()));
    let interrupted = persisted.status == SessionStatus::Running;
    let status = if interrupted {
        SessionStatus::Paused
    } else {
        persisted.status
    };
    let mut request_template = persisted.request_template;
    request_template.workflow_checkpoint = persisted.workflow.clone();
    (
        session_id,
        SessionState {
            record: SessionRecord {
                revision: persisted.revision,
                task: persisted.task,
                title,
                branch: persisted.branch,
                workdir: persisted.workdir,
                durable,
                request_template,
                status,
                metrics,
                workflow: persisted.workflow,
                completed_workflows: persisted.completed_workflows,
                goal: persisted.goal,
                completed_goals: persisted.completed_goals,
                multi_task: persisted.multi_task,
                completed_multi_tasks: persisted.completed_multi_tasks,
                started_at_ms: persisted.started_at_ms,
                updated_at_ms: persisted.updated_at_ms,
            },
            runtime: SessionRuntime {
                pending_question: None,
                sender,
                history,
                usage_records: Arc::new(StdMutex::new(usage_records)),
                pending_user_messages,
                accepting_user_messages: Arc::new(AtomicBool::new(false)),
                pause_token: Arc::new(AtomicBool::new(false)),
                cancel_token: Arc::new(AtomicBool::new(false)),
            },
        },
    )
}

fn pending_question_view(pending: &PendingQuestionState) -> PendingQuestionView {
    PendingQuestionView {
        question_id: pending.question_id.clone(),
        question: pending.question.clone(),
        choices: pending.choices.clone(),
    }
}

fn persist_exact_session_state(
    repository: &dyn session_store::SessionRepository,
    session_id: &str,
    session: &SessionState,
) -> std::result::Result<(), SessionPersistenceError> {
    let _persistence = SESSION_PERSISTENCE_LOCK.lock().map_err(|_| {
        SessionPersistenceError(anyhow::anyhow!("session persistence lock is unavailable"))
    })?;
    let events = session
        .runtime
        .history
        .lock()
        .map_err(|_| SessionPersistenceError(anyhow::anyhow!("session history is unavailable")))?
        .clone();
    let usage_records = session
        .runtime
        .usage_records
        .lock()
        .map_err(|_| {
            SessionPersistenceError(anyhow::anyhow!("session usage history is unavailable"))
        })?
        .clone();
    let pending_user_messages = session
        .runtime
        .pending_user_messages
        .lock()
        .map_err(|_| {
            SessionPersistenceError(anyhow::anyhow!("session message queue is unavailable"))
        })?
        .iter()
        .cloned()
        .collect();
    let persisted = exact_persisted_session(
        session_id,
        session,
        events,
        usage_records,
        pending_user_messages,
    );
    repository.save(&persisted).map_err(SessionPersistenceError)
}

fn exact_persisted_session(
    session_id: &str,
    session: &SessionState,
    events: Vec<EventEnvelope>,
    usage_records: Vec<SessionMetricsSnapshot>,
    pending_user_messages: Vec<QueuedUserMessage>,
) -> PersistedSession {
    let mut persisted = PersistedSession::from_parts(
        session_id.to_string(),
        session.request_template.clone(),
        session.branch.clone(),
        session.workdir.clone(),
        session.status,
        events,
    );
    persisted.task.clone_from(&session.task);
    persisted.revision = session.revision;
    persisted.title.clone_from(&session.title);
    persisted.started_at_ms = session.started_at_ms;
    persisted.updated_at_ms = session.updated_at_ms;
    persisted.metrics = combined_metrics(&usage_records);
    persisted.usage_records = usage_records;
    persisted.workflow.clone_from(&session.workflow);
    persisted
        .completed_workflows
        .clone_from(&session.completed_workflows);
    persisted.goal.clone_from(&session.goal);
    persisted
        .completed_goals
        .clone_from(&session.completed_goals);
    persisted.multi_task.clone_from(&session.multi_task);
    persisted
        .completed_multi_tasks
        .clone_from(&session.completed_multi_tasks);
    persisted.pending_user_messages = pending_user_messages;
    persisted.project.clone_from(&session.durable.project);
    persisted
        .pending_delivery_proposal
        .clone_from(&session.durable.pending_delivery_proposal);
    persisted
        .pending_goal_proposal
        .clone_from(&session.durable.pending_goal_proposal);
    persisted
        .pending_goal_change
        .clone_from(&session.durable.pending_goal_change);
    persisted
}

fn combined_metrics(records: &[SessionMetricsSnapshot]) -> Option<SessionMetricsSnapshot> {
    let mut combined: Option<SessionMetricsSnapshot> = None;
    for metrics in records {
        if let Some(existing) = combined.as_mut() {
            existing.add_assign(metrics);
        } else {
            combined = Some(metrics.clone());
        }
    }
    combined
}

async fn session_list_snapshot(state: &AppState) -> Vec<SessionListItem> {
    let sessions = state.sessions.lock().await;
    session_list_items(&sessions)
}

fn session_list_items(sessions: &HashMap<String, SessionState>) -> Vec<SessionListItem> {
    let mut items = sessions
        .iter()
        .map(|(session_id, session)| session_list_item(session_id, session))
        .collect::<Vec<_>>();
    items.sort_by_key(|b| std::cmp::Reverse(b.updated_at_ms));
    items
}

fn session_list_item(session_id: &str, session: &SessionState) -> SessionListItem {
    let workflow = latest_workflow_summary(session);
    let goal = latest_goal_checkpoint(session);
    let (title, handoff_outcome) = session_history_summary(session);
    SessionListItem {
        session_id: session_id.to_string(),
        task: session.task.clone(),
        title,
        status: session.status,
        intent: session.request_template.intent,
        branch: session.branch.clone(),
        workdir: session
            .workdir
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        project: session.durable.project.clone(),
        handoff_outcome,
        pending_question: session
            .runtime
            .pending_question
            .as_ref()
            .map(pending_question_view),
        started_at_ms: session.started_at_ms,
        updated_at_ms: session.updated_at_ms,
        workflow_id: workflow.as_ref().map(|workflow| workflow.id.clone()),
        workflow_stage: workflow.as_ref().map(|workflow| workflow.stage),
        workflow_outcome: workflow.as_ref().and_then(|workflow| workflow.outcome),
        strict_workflow: strict_workflow_enabled(session),
        goal: goal.map(|checkpoint| crate::goal::GoalSummary::from(&checkpoint.run)),
        active_goal: session.goal.is_some(),
        multi_task: latest_multi_task_checkpoint(session).map(multi_task_summary),
        active_multi_task: session.multi_task.is_some(),
    }
}

#[derive(Debug, Clone, Default)]
struct WindowedUsageAccumulator {
    tokens: f64,
    runtime_ms: f64,
    tool_calls: f64,
    energy_joules: Option<f64>,
}

impl WindowedUsageAccumulator {
    fn add_metrics(&mut self, metrics: &SessionMetricsSnapshot, window: UsageWindow) {
        let interval_ms = metrics
            .ended_at_ms
            .saturating_sub(metrics.started_at_ms)
            .max(1);
        let overlap_start = metrics.started_at_ms.max(window.start_ms);
        let overlap_end = metrics.ended_at_ms.min(window.end_ms);
        if overlap_end <= overlap_start {
            return;
        }
        let share = (overlap_end - overlap_start) as f64 / interval_ms as f64;
        self.tokens += metrics
            .prompt_tokens
            .saturating_add(metrics.generated_tokens) as f64
            * share;
        self.runtime_ms += metrics_runtime_ms(metrics) as f64 * share;
        self.tool_calls += metrics.tool_calls as f64 * share;
        if let Some(energy_joules) = metrics_energy_joules(metrics) {
            self.energy_joules = Some(self.energy_joules.unwrap_or(0.0) + energy_joules * share);
        }
    }

    fn finish(&self) -> ProjectUsageStats {
        let energy_joules = self.energy_joules;
        ProjectUsageStats {
            tokens: self.tokens.round() as usize,
            runtime_ms: self.runtime_ms.round() as u64,
            tool_calls: self.tool_calls.round() as usize,
            energy_kwh: energy_joules.map(|joules| joules / 3_600_000.0),
            energy_joules,
        }
    }
}

impl CachedProjectUsageWindow {
    fn from_sessions(
        window: UsageWindow,
        projects: &[ProjectEntry],
        sessions: &HashMap<String, SessionState>,
    ) -> Self {
        let mut cached = Self {
            window,
            overall_today: WindowedUsageAccumulator::default(),
            project_today: projects
                .iter()
                .map(|project| (project.id.clone(), WindowedUsageAccumulator::default()))
                .collect(),
        };
        for session in sessions.values() {
            let project_id = session
                .durable
                .project
                .as_ref()
                .map(|project| project.id.as_str());
            let records = session
                .runtime
                .usage_records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for metrics in records.iter() {
                cached.add_metrics(project_id, metrics);
            }
        }
        cached
    }

    fn add_metrics(&mut self, project_id: Option<&str>, metrics: &SessionMetricsSnapshot) {
        self.overall_today.add_metrics(metrics, self.window);
        if let Some(accumulator) = project_id.and_then(|id| self.project_today.get_mut(id)) {
            accumulator.add_metrics(metrics, self.window);
        }
    }
}

impl ProjectUsageWindowCache {
    fn invalidate(&mut self) {
        self.entries.clear();
    }

    fn summaries(
        &mut self,
        projects: &[ProjectEntry],
        sessions: &HashMap<String, SessionState>,
        window: UsageWindow,
    ) -> (ProjectUsageStats, HashMap<String, ProjectUsageStats>) {
        let position = self
            .entries
            .iter()
            .position(|cached| cached.window == window);
        let cached = if let Some(position) = position {
            let cached = self.entries.remove(position).expect("cached window exists");
            self.entries.push_back(cached);
            self.entries.back().expect("cached window was restored")
        } else {
            if self.entries.len() == MAX_PROJECT_USAGE_WINDOW_CACHE_ENTRIES {
                self.entries.pop_front();
            }
            self.entries
                .push_back(CachedProjectUsageWindow::from_sessions(
                    window, projects, sessions,
                ));
            self.entries.back().expect("cached window was inserted")
        };
        let project_today = projects
            .iter()
            .map(|project| {
                (
                    project.id.clone(),
                    cached
                        .project_today
                        .get(&project.id)
                        .map(WindowedUsageAccumulator::finish)
                        .unwrap_or_default(),
                )
            })
            .collect();
        (cached.overall_today.finish(), project_today)
    }
}

fn metrics_runtime_ms(metrics: &SessionMetricsSnapshot) -> u64 {
    if metrics.wall_runtime_ms > 0 {
        metrics.wall_runtime_ms
    } else {
        metrics
            .llm_runtime_ms
            .saturating_add(metrics.tool_runtime_ms)
    }
}

fn usage_summaries(
    projects: &[ProjectEntry],
    sessions: &HashMap<String, SessionState>,
    usage_windows: &mut ProjectUsageWindowCache,
    window: UsageWindow,
) -> (ProjectUsageSummary, HashMap<String, ProjectUsageSummary>) {
    let mut overall_total = ProjectUsageStats::default();
    let mut project_totals = projects
        .iter()
        .map(|project| (project.id.clone(), ProjectUsageStats::default()))
        .collect::<HashMap<_, _>>();

    for session in sessions.values() {
        let Some(metrics) = &session.metrics else {
            continue;
        };
        overall_total.add_metrics(metrics);
        if let Some(total) = session
            .durable
            .project
            .as_ref()
            .and_then(|project| project_totals.get_mut(&project.id))
        {
            total.add_metrics(metrics);
        }
    }

    let (overall_today, mut project_today) = usage_windows.summaries(projects, sessions, window);

    let project_usage = projects
        .iter()
        .map(|project| {
            (
                project.id.clone(),
                ProjectUsageSummary {
                    total: project_totals.remove(&project.id).unwrap_or_default(),
                    today: project_today.remove(&project.id).unwrap_or_default(),
                },
            )
        })
        .collect();
    (
        ProjectUsageSummary {
            total: overall_total,
            today: overall_today,
        },
        project_usage,
    )
}

async fn project_session_snapshot(
    state: &AppState,
    terminal_transition_floor: u64,
    usage_window: UsageWindow,
) -> ProjectSessionSnapshot {
    let projects = state.projects.lock().await;
    let sessions = state.sessions.lock().await;
    let mut usage_windows = state.project_usage_windows.lock().await;
    let publication = state
        .project_session_publication
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let revision = state.project_session_revision.load(Ordering::SeqCst);
    project_session_snapshot_from_locked(
        state,
        &projects,
        &sessions,
        &mut usage_windows,
        &publication,
        revision,
        terminal_transition_floor,
        usage_window,
    )
}

fn project_session_snapshot_from_locked(
    state: &AppState,
    projects: &[ProjectEntry],
    sessions: &HashMap<String, SessionState>,
    usage_windows: &mut ProjectUsageWindowCache,
    publication: &ProjectSessionPublication,
    revision: u64,
    terminal_transition_floor: u64,
    usage_window: UsageWindow,
) -> ProjectSessionSnapshot {
    let (overall_usage, project_usage) =
        usage_summaries(projects, sessions, usage_windows, usage_window);
    ProjectSessionSnapshot {
        stream_id: (*state.project_session_stream_id).clone(),
        revision,
        usage_window_start_ms: usage_window.start_ms,
        usage_window_end_ms: usage_window.end_ms,
        terminal_transition_floor,
        terminal_transitions: publication
            .terminal_transitions
            .iter()
            .filter(|transition| {
                transition.revision > terminal_transition_floor && transition.revision <= revision
            })
            .cloned()
            .collect(),
        projects: projects.to_vec(),
        sessions: session_list_items(sessions),
        overall_usage,
        project_usage,
        warnings: publication.warnings.clone(),
    }
}

async fn reload_projects(state: &AppState) -> Result<()> {
    state
        .commit_project_registry_owned(UsageWindow::current_utc_day(), || {
            Ok(projects::ProjectRegistryMutation {
                value: (),
                projects: projects::load_projects()?,
            })
        })
        .await?;
    Ok(())
}

async fn reconcile_project_registry(
    state: &AppState,
    current_projects: &mut Vec<ProjectEntry>,
    projects: Vec<ProjectEntry>,
) {
    let session_ids = state
        .sessions
        .lock()
        .await
        .iter()
        .filter_map(|(session_id, session)| {
            let stored = session.durable.project.as_ref()?;
            let current = projects.iter().find(|project| project.id == stored.id)?;
            let previous_path = PathBuf::from(&stored.path);
            let project_changed = stored.name != current.name || stored.path != current.path;
            let workdir_changes = stored.path != current.path
                && session.status != SessionStatus::Running
                && (session.workdir.as_ref() == Some(&previous_path)
                    || session.request_template.workdir.as_ref() == Some(&previous_path));
            (project_changed || workdir_changes).then(|| session_id.clone())
        })
        .collect::<Vec<_>>();
    let mut session_projection_changed = false;
    let mut warnings = Vec::new();
    let registry_changed = *current_projects != projects;
    *current_projects = projects.clone();
    for session_id in session_ids {
        let registry = projects.clone();
        match state
            .commit_session_projection_change_owned(session_id.clone(), move |session| {
                let Some(stored_project) = session.durable.project.as_ref() else {
                    return Ok(StagedSessionChange::new(false));
                };
                let Some(current_project) = registry
                    .iter()
                    .find(|project| project.id == stored_project.id)
                else {
                    return Ok(StagedSessionChange::new(false));
                };
                let previous_path = PathBuf::from(&stored_project.path);
                let current_path = PathBuf::from(&current_project.path);
                let previous_project = session.durable.project.clone();
                let previous_workdir = session.workdir.clone();
                let previous_request_workdir = session.request_template.workdir.clone();
                session.durable.project = Some(SessionProject {
                    id: current_project.id.clone(),
                    name: current_project.name.clone(),
                    path: current_project.path.clone(),
                });
                if session.status != SessionStatus::Running
                    && session.workdir.as_ref() == Some(&previous_path)
                {
                    session.workdir = Some(current_path.clone());
                }
                if session.status != SessionStatus::Running
                    && session.request_template.workdir.as_ref() == Some(&previous_path)
                {
                    session.request_template.workdir = Some(current_path);
                }
                let changed = session.durable.project != previous_project
                    || session.workdir != previous_workdir
                    || session.request_template.workdir != previous_request_workdir;
                Ok(StagedSessionChange::new(changed))
            })
            .await
        {
            Ok(commit) => session_projection_changed |= commit.value,
            Err(error) => {
                let warning = format!(
                    "the project registry was committed, but session {session_id} could not be refreshed: {error:#}"
                );
                tracing::warn!(%error, %session_id, %warning);
                warnings.push(warning);
            }
        }
    }
    if registry_changed {
        state.project_usage_windows.lock().await.invalidate();
    }
    state.publish_project_session_reconciliation(
        registry_changed || session_projection_changed,
        warnings,
    );
}

async fn project_list_snapshot(state: &AppState) -> Vec<ProjectEntry> {
    state.projects.lock().await.clone()
}

async fn session_details_snapshot(state: &AppState, id: &str) -> Option<SessionDetails> {
    let sessions = state.sessions.lock().await;
    let session = sessions.get(id)?;
    let history = session.runtime.history.lock().ok()?;
    Some(session_details_from_history(id, session, &history))
}

fn session_mutation_snapshot_for_session(
    id: &str,
    session: &SessionState,
) -> SessionStreamSnapshot {
    let history = session
        .runtime
        .history
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    SessionStreamSnapshot {
        session: session_details_from_history(id, session, &history),
        reset_history: false,
        warnings: Vec::new(),
    }
}

fn session_details_from_history(
    id: &str,
    session: &SessionState,
    history: &[EventEnvelope],
) -> SessionDetails {
    let start = history.len().saturating_sub(SESSION_HISTORY_RESPONSE_LIMIT);
    let events = history[start..].to_vec();
    let title = latest_session_title(history).or_else(|| session.title.clone());
    let handoff_outcome = handoff_outcome_from_history(history);
    let workflow = latest_workflow_summary(session);
    let goal = latest_goal_checkpoint(session).cloned();
    SessionDetails {
        session_id: id.to_string(),
        task: session.task.clone(),
        title,
        cancel_requested: session.status == SessionStatus::Running
            && session.runtime.cancel_token.load(Ordering::SeqCst),
        status: session.status,
        intent: session.request_template.intent,
        branch: session.branch.clone(),
        workdir: session
            .workdir
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        project: session.durable.project.clone(),
        handoff_outcome,
        pending_question: session
            .runtime
            .pending_question
            .as_ref()
            .map(pending_question_view),
        events,
        started_at_ms: session.started_at_ms,
        updated_at_ms: session.updated_at_ms,
        metrics: session.metrics.clone(),
        usage_records: session_usage_records(session),
        workflow,
        strict_workflow: strict_workflow_enabled(session),
        goal,
        active_goal: session.goal.is_some(),
        multi_task: latest_multi_task_checkpoint(session).cloned(),
        active_multi_task: session.multi_task.is_some(),
        task_plan_rejected: session.request_template.task_plan_rejected.clone(),
        task_planning_transcript: session.request_template.task_planning_transcript.clone(),
        pending_delivery_proposal: session.durable.pending_delivery_proposal.clone(),
        pending_goal_proposal: session.durable.pending_goal_proposal.clone(),
        pending_goal_change: session.durable.pending_goal_change.clone(),
        revision: session.revision,
    }
}

async fn github_oauth_callback(Query(query): Query<HashMap<String, String>>) -> Response {
    crate::github_oauth::persist_callback_from_query(&query).into_response()
}

async fn service_worker() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        SERVICE_WORKER_JS,
    )
}

async fn index() -> impl IntoResponse {
    serve_asset("index.html")
}

async fn static_asset(Path(path): Path<String>) -> impl IntoResponse {
    let target = if path.is_empty() { "index.html" } else { &path };
    serve_asset(target)
}

fn serve_asset(path: &str) -> Response {
    let file = WebAssets::get(path).or_else(|| WebAssets::get("index.html"));
    if let Some(content) = file {
        let mime = if path.ends_with(".html") || !path.contains('.') {
            mime_guess::mime::TEXT_HTML
        } else {
            mime_guess::from_path(path).first_or_octet_stream()
        };
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, mime.to_string().as_str())],
            content.data,
        )
            .into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

fn new_session_id() -> String {
    new_durable_id("session")
}

fn new_turn_id(session_id: &str) -> String {
    new_durable_id(&format!("turn-{session_id}"))
}

fn new_durable_id(prefix: &str) -> String {
    let mut random = [0_u8; 16];
    if getrandom::getrandom(&mut random).is_ok() {
        use std::fmt::Write as _;
        let mut suffix = String::with_capacity(random.len() * 2);
        for byte in random {
            write!(&mut suffix, "{byte:02x}").expect("writing to a String cannot fail");
        }
        return format!("{prefix}-{suffix}");
    }
    let sequence = ID_FALLBACK_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!(
        "{prefix}-{}-{}-{sequence}",
        std::process::id(),
        now_millis()
    )
}

fn new_user_message_id(session_id: &str) -> String {
    let sequence = USER_MESSAGE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("message-{session_id}-{}-{sequence}", now_millis())
}

fn workflow_policy_for_request(
    workdir: Option<&std::path::Path>,
) -> Result<crate::workflow::CompiledWorkflowPolicy> {
    if let Some(workdir) = workdir {
        let root =
            crate::agent_core::find_git_root(workdir).unwrap_or_else(|| workdir.to_path_buf());
        crate::workflow::WorkflowConfigDocument::load_or_default(&root)
    } else {
        crate::workflow::WorkflowConfigDocument::default().compile()
    }
}

fn goal_policy_for_request(
    workdir: Option<&std::path::Path>,
) -> Result<crate::goal::CompiledGoalPolicy> {
    let workdir = workdir.context("goal mode requires a repository")?;
    let root = crate::agent_core::find_git_root(workdir).unwrap_or_else(|| workdir.to_path_buf());
    crate::goal::GoalConfigDocument::load_or_default(&root)
}

fn delivery_handoff_for_turn(
    intent: Option<crate::workflow::TurnIntent>,
    turn_id: &str,
    task: &str,
    proposal: Option<&crate::workflow::DeliveryProposal>,
) -> Option<crate::workflow::ConversationHandoff> {
    if intent != Some(crate::workflow::TurnIntent::Deliver) {
        return None;
    }
    let mut handoff = proposal.map_or_else(
        || crate::workflow::ConversationHandoff {
            source_turn_ids: Vec::new(),
            task_summary: bounded_handoff_text(task),
            ..crate::workflow::ConversationHandoff::default()
        },
        crate::workflow::DeliveryProposal::handoff,
    );
    if !handoff.source_turn_ids.iter().any(|id| id == turn_id) {
        handoff.source_turn_ids.push(turn_id.to_string());
    }
    Some(handoff)
}

fn bounded_handoff_text(text: &str) -> String {
    const MAX_HANDOFF_CHARS: usize = 4_000;
    let mut chars = text.trim().chars();
    let bounded = chars.by_ref().take(MAX_HANDOFF_CHARS).collect::<String>();
    if chars.next().is_some() {
        format!("{bounded}…")
    } else {
        bounded
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn stage_event(history: &StdMutex<Vec<EventEnvelope>>, event: AgentEvent) -> bool {
    stage_event_linked(history, event, Vec::new())
}

fn stage_event_linked(
    history: &StdMutex<Vec<EventEnvelope>>,
    event: AgentEvent,
    supersedes: Vec<String>,
) -> bool {
    let envelope = EventEnvelope::with_timestamp(event);
    stage_event_envelope_linked(history, envelope, supersedes)
}

fn stage_event_envelope_linked(
    history: &StdMutex<Vec<EventEnvelope>>,
    envelope: EventEnvelope,
    supersedes: Vec<String>,
) -> bool {
    stage_event_envelope(history, envelope, supersedes)
}

fn stage_event_envelope(
    history: &StdMutex<Vec<EventEnvelope>>,
    mut envelope: EventEnvelope,
    supersedes: Vec<String>,
) -> bool {
    envelope.transcript.supersedes = supersedes;
    let Ok(mut entries) = history.lock() else {
        tracing::error!(
            entry_key = %envelope.transcript.entry_key,
            "refusing to publish an event because session history is poisoned"
        );
        return false;
    };
    let sequence = entries
        .last()
        .map_or(1, |entry| entry.transcript.sequence.saturating_add(1));
    envelope.assign_sequence(sequence);
    envelope.refresh_projections(&entries);
    entries.push(envelope.clone());
    if entries.len() > MAX_HISTORY_EVENTS {
        *entries =
            session_store::trim_event_history(std::mem::take(&mut *entries), MAX_HISTORY_EVENTS);
    }
    true
}

#[cfg(test)]
mod workflow_tests {
    use super::*;

    #[derive(Debug, Default)]
    struct FaultInjectingSessionRepository {
        fail_next_save: AtomicBool,
        saved: StdMutex<Vec<PersistedSession>>,
        deleted: StdMutex<Vec<(PathBuf, String)>>,
    }

    impl FaultInjectingSessionRepository {
        fn fail_next_save(&self) {
            self.fail_next_save.store(true, Ordering::SeqCst);
        }

        fn saved(&self) -> Vec<PersistedSession> {
            self.saved
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl session_store::SessionRepository for FaultInjectingSessionRepository {
        fn save(&self, session: &PersistedSession) -> Result<()> {
            if self.fail_next_save.swap(false, Ordering::SeqCst) {
                bail!("injected session persistence failure");
            }
            self.saved
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(session.clone());
            Ok(())
        }

        fn delete(&self, workdir: &std::path::Path, session_id: &str) -> Result<()> {
            self.deleted
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((workdir.to_path_buf(), session_id.to_string()));
            Ok(())
        }
    }

    #[derive(Debug)]
    struct BlockingSessionRepository {
        started: StdMutex<Option<std::sync::mpsc::Sender<()>>>,
        release: (StdMutex<bool>, std::sync::Condvar),
        saved: StdMutex<Vec<PersistedSession>>,
    }

    impl BlockingSessionRepository {
        fn new(started: std::sync::mpsc::Sender<()>) -> Self {
            Self {
                started: StdMutex::new(Some(started)),
                release: (StdMutex::new(false), std::sync::Condvar::new()),
                saved: StdMutex::new(Vec::new()),
            }
        }

        fn release(&self) {
            *self
                .release
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
            self.release.1.notify_all();
        }

        fn saved(&self) -> Vec<PersistedSession> {
            self.saved
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl session_store::SessionRepository for BlockingSessionRepository {
        fn save(&self, session: &PersistedSession) -> Result<()> {
            if let Some(started) = self
                .started
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
            {
                let _ = started.send(());
            }
            let (lock, ready) = &self.release;
            let mut released = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while !*released {
                released = ready
                    .wait(released)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            self.saved
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(session.clone());
            Ok(())
        }

        fn delete(&self, _workdir: &std::path::Path, _session_id: &str) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn coordinator_derives_queue_dispatch_from_committed_status() {
        let mut request = request(std::path::Path::new("."));
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Paused,
            Vec::new(),
        );
        let (session_id, mut session) = session_from_persisted(persisted);
        let repository = FaultInjectingSessionRepository::default();

        let commit = commit_session_change(
            &repository,
            &session_id,
            &mut session,
            SessionCommitPolicy::Full,
            |staged| {
                staged.status = SessionStatus::Queued;
                Ok(StagedSessionChange::new(()))
            },
        )
        .unwrap();

        assert!(commit.effects.dispatch);
        assert_eq!(commit.snapshot.session.status, SessionStatus::Queued);
    }

    #[tokio::test]
    async fn committed_session_change_is_the_persistence_publication_barrier() {
        let mut request = request(std::path::Path::new("."));
        request.session_id = "session-transaction-barrier".to_string();
        request.task = "test transaction barrier".to_string();
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Queued,
            Vec::new(),
        );
        let (session_id, session) = session_from_persisted(persisted);
        let repository = Arc::new(FaultInjectingSessionRepository::default());
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: repository.clone(),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let mut session_events = state.sessions.lock().await[&session_id]
            .runtime
            .sender
            .subscribe();
        let mut collection_events = state.project_session_sender.subscribe();
        repository.fail_next_save();

        let before = {
            let sessions = state.sessions.lock().await;
            session_mutation_snapshot_for_session(&session_id, &sessions[&session_id])
        };
        let failed = state
            .commit_session_change_owned(session_id.clone(), |staged| {
                staged.status = SessionStatus::Running;
                staged
                    .runtime
                    .pending_user_messages
                    .lock()
                    .unwrap()
                    .push_back(QueuedUserMessage {
                        message_id: "message-failed-save".to_string(),
                        message: "must stay staged".to_string(),
                    });
                staged
                    .runtime
                    .usage_records
                    .lock()
                    .unwrap()
                    .push(SessionMetricsSnapshot::default());
                staged.metrics = Some(SessionMetricsSnapshot::default());
                Ok(StagedSessionChange::new(()))
            })
            .await;
        assert!(failed.is_err());
        let sessions = state.sessions.lock().await;
        let session = &sessions[&session_id];
        let after = session_mutation_snapshot_for_session(&session_id, session);
        assert_eq!(after.session.status, before.session.status);
        assert_eq!(after.session.revision, before.session.revision);
        assert!(repository.saved().is_empty());
        assert!(session_events.try_recv().is_err());
        assert!(collection_events.try_recv().is_err());
        assert!(
            session
                .runtime
                .pending_user_messages
                .lock()
                .unwrap()
                .is_empty()
        );
        assert!(session.runtime.usage_records.lock().unwrap().is_empty());
        assert!(session.metrics.is_none());
        assert_eq!(state.project_session_revision.load(Ordering::SeqCst), 0);
        drop(sessions);

        let committed = state
            .commit_session_change_owned(session_id.clone(), |staged| {
                staged.status = SessionStatus::Running;
                staged
                    .runtime
                    .usage_records
                    .lock()
                    .unwrap()
                    .push(SessionMetricsSnapshot {
                        prompt_tokens: 7,
                        ..SessionMetricsSnapshot::default()
                    });
                staged.metrics = Some(SessionMetricsSnapshot::default());
                Ok(StagedSessionChange::new(()))
            })
            .await
            .unwrap();
        let saved = repository.saved();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].status, SessionStatus::Running);
        assert!(
            serde_json::to_value(&saved[0])
                .unwrap()
                .get("running")
                .is_none()
        );
        assert_eq!(saved[0].metrics.as_ref().unwrap().prompt_tokens, 7);
        assert_eq!(saved[0].events.len(), 1);
        assert_eq!(committed.snapshot.session.revision, 1);
        assert_eq!(committed.snapshot.session.status, SessionStatus::Running);
        assert_eq!(
            committed
                .snapshot
                .session
                .metrics
                .as_ref()
                .unwrap()
                .prompt_tokens,
            7
        );
        let SessionStreamPublication::Event(envelope) = session_events.try_recv().unwrap() else {
            panic!("the committed event must precede its exact snapshot");
        };
        assert_eq!(envelope.transcript.sequence, 1);
        assert!(matches!(
            envelope.event,
            AgentEvent::SessionStateChanged {
                status: crate::events::SessionLifecycleStatus::Running,
                ..
            }
        ));
        let SessionStreamPublication::Snapshot(snapshot) = session_events.try_recv().unwrap()
        else {
            panic!("the committed event must be followed by its exact snapshot");
        };
        assert_eq!(snapshot.session.revision, 1);
        assert_eq!(collection_events.try_recv().unwrap(), 1);
    }

    #[tokio::test]
    async fn cancelled_caller_cannot_split_an_owned_session_commit() {
        let mut request = request(std::path::Path::new("."));
        request.session_id = "session-owned-cancellation".to_string();
        request.task = "before cancellation".to_string();
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Paused,
            Vec::new(),
        );
        let (session_id, session) = session_from_persisted(persisted);
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let repository = Arc::new(BlockingSessionRepository::new(started_tx));
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: repository.clone(),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };

        let transaction_state = state.clone();
        let transaction_session_id = session_id.clone();
        let caller = tokio::spawn(async move {
            transaction_state
                .commit_session_change_owned(transaction_session_id, |staged| {
                    staged.task = "after cancellation".to_string();
                    staged.updated_at_ms = now_millis();
                    Ok(StagedSessionChange::new(()))
                })
                .await
        });
        tokio::task::spawn_blocking(move || started_rx.recv())
            .await
            .unwrap()
            .unwrap();
        caller.abort();
        repository.release();

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let committed =
                    state.sessions.lock().await[&session_id].task == "after cancellation";
                if committed && repository.saved().len() == 1 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("detached transaction should finish after its caller is cancelled");
        assert_eq!(repository.saved()[0].task, "after cancellation");
    }

    #[tokio::test]
    async fn cancelled_answer_caller_still_wakes_or_warns_after_the_durable_commit() {
        let mut request = request(std::path::Path::new("."));
        request.session_id = "session-owned-answer".to_string();
        request.task = "answer durably".to_string();
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Paused,
            Vec::new(),
        );
        let (session_id, mut session) = session_from_persisted(persisted);
        let (answer_tx, answer_rx) = std::sync::mpsc::channel();
        drop(answer_rx);
        session.runtime.pending_question = Some(PendingQuestionState {
            question_id: "question-1".to_string(),
            question: "Continue?".to_string(),
            choices: vec!["Yes".to_string()],
            responder: answer_tx,
        });
        let mut publications = session.runtime.sender.subscribe();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let repository = Arc::new(BlockingSessionRepository::new(started_tx));
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: repository.clone(),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let caller_state = state.clone();
        let caller_session_id = session_id.clone();
        let caller = tokio::spawn(async move {
            answer_question_inner(
                caller_state,
                caller_session_id,
                AnswerQuestionRequest {
                    question_id: "question-1".to_string(),
                    answer: "Yes".to_string(),
                },
            )
            .await
        });
        tokio::task::spawn_blocking(move || started_rx.recv())
            .await
            .unwrap()
            .unwrap();
        caller.abort();
        repository.release();

        let warning_snapshot = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let SessionStreamPublication::Snapshot(snapshot) =
                    publications.recv().await.unwrap()
                    && !snapshot.warnings.is_empty()
                {
                    break snapshot;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(warning_snapshot.session.revision, 1);
        assert!(warning_snapshot.warnings[0].contains("answer was committed"));
        let sessions = state.sessions.lock().await;
        assert_eq!(sessions[&session_id].status, SessionStatus::Running);
        assert!(sessions[&session_id].runtime.pending_question.is_none());
        assert_eq!(repository.saved().len(), 1);
    }

    #[tokio::test]
    async fn eventless_mutations_advance_and_publish_session_revisions() {
        let mut request = request(std::path::Path::new("."));
        request.session_id = "session-eventless-revision".to_string();
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Paused,
            Vec::new(),
        );
        let (session_id, session) = session_from_persisted(persisted);
        let mut publications = session.runtime.sender.subscribe();
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(FaultInjectingSessionRepository::default()),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };

        for (expected_revision, task) in [(1, "first"), (2, "second")] {
            let task = task.to_string();
            let commit = state
                .commit_session_change_owned(session_id.clone(), move |staged| {
                    staged.task = task;
                    Ok(StagedSessionChange::new(()))
                })
                .await
                .unwrap();
            assert_eq!(commit.snapshot.session.revision, expected_revision);
            let SessionStreamPublication::Snapshot(snapshot) = publications.recv().await.unwrap()
            else {
                panic!("an eventless commit must publish its exact snapshot");
            };
            assert_eq!(snapshot.session.revision, expected_revision);
            assert_eq!(snapshot.session.task, commit.snapshot.session.task);
        }
    }

    #[tokio::test]
    async fn failed_session_creation_has_no_live_or_collection_side_effect() {
        let mut request = request(std::path::Path::new("."));
        request.session_id = "session-create-failure".to_string();
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Queued,
            Vec::new(),
        );
        let (session_id, session) = session_from_persisted(persisted);
        let repository = Arc::new(FaultInjectingSessionRepository::default());
        repository.fail_next_save();
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            session_repository: repository.clone(),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };

        let error = state
            .create_session_owned(session_id, session)
            .await
            .unwrap_err();
        assert!(error.downcast_ref::<SessionPersistenceError>().is_some());
        assert!(state.sessions.lock().await.is_empty());
        assert_eq!(state.project_session_revision.load(Ordering::SeqCst), 0);
        assert!(repository.saved().is_empty());
    }

    #[tokio::test]
    async fn dispatch_claim_publishes_its_committed_collection_revision() {
        let mut request = request(std::path::Path::new("."));
        request.session_id = "session-dispatch-publication".to_string();
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Queued,
            Vec::new(),
        );
        let (session_id, session) = session_from_persisted(persisted);
        let (project_session_sender, _) = broadcast::channel(16);
        let mut revisions = project_session_sender.subscribe();
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(FaultInjectingSessionRepository::default()),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };

        let (claimed, working) = state.claim_next_session_owned().await.unwrap();
        assert_eq!(claimed.unwrap().0, session_id);
        assert!(working);
        assert_eq!(revisions.recv().await.unwrap(), 1);
        assert!(revisions.try_recv().is_err());
        assert_eq!(
            state.sessions.lock().await[&session_id].status,
            SessionStatus::Running
        );
    }

    #[tokio::test]
    async fn message_seal_and_enqueue_have_one_serial_order() {
        let mut request = request(std::path::Path::new("."));
        request.session_id = "session-message-seal".to_string();
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Running,
            Vec::new(),
        );
        let (session_id, session) = session_from_persisted(persisted);
        session
            .runtime
            .accepting_user_messages
            .store(true, Ordering::SeqCst);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(FaultInjectingSessionRepository::default()),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };

        let seal_state = state.clone();
        let seal_id = session_id.clone();
        let seal =
            tokio::spawn(async move { seal_state.seal_session_messages_owned(seal_id).await });
        let enqueue_state = state.clone();
        let enqueue_id = session_id.clone();
        let enqueue = tokio::spawn(async move {
            enqueue_state
                .commit_session_change_owned(enqueue_id, |staged| {
                    if !staged
                        .runtime
                        .accepting_user_messages
                        .load(Ordering::SeqCst)
                    {
                        bail!("session message queue is sealed");
                    }
                    staged
                        .runtime
                        .pending_user_messages
                        .lock()
                        .unwrap()
                        .push_back(QueuedUserMessage {
                            message_id: "message-race".to_string(),
                            message: "serialize me".to_string(),
                        });
                    Ok(StagedSessionChange::new(()))
                })
                .await
        });

        let sealed = seal.await.unwrap().unwrap();
        let enqueued = enqueue.await.unwrap().is_ok();
        assert_eq!(sealed, !enqueued);
        let sessions = state.sessions.lock().await;
        let session = &sessions[&session_id];
        assert_eq!(
            session.runtime.pending_user_messages.lock().unwrap().len(),
            usize::from(enqueued)
        );
        assert_eq!(
            session
                .runtime
                .accepting_user_messages
                .load(Ordering::SeqCst),
            !sealed
        );
    }

    #[tokio::test]
    async fn project_session_stream_advances_revision_for_session_events() {
        let mut request = request(std::path::Path::new("."));
        request.session_id = "session-collection-publication".to_string();
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Paused,
            Vec::new(),
        );
        let (session_id, session) = session_from_persisted(persisted);
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(FaultInjectingSessionRepository::default()),
            projects: Arc::new(Mutex::new(vec![ProjectEntry {
                id: "project-1".to_string(),
                name: "pb".to_string(),
                path: "/workspace/pb".to_string(),
                repository_root: None,
                notify_on_finish: false,
            }])),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        assert!(
            !EventEnvelope::new(AgentEvent::Final {
                content: "done".to_string(),
                profile: AgentProfile::Build,
                nesting_depth: None,
                timestamp_ms: None,
            })
            .affects_project_session_snapshot()
        );
        let (mut changes, transition_floor) = state.subscribe_project_session_changes();
        assert_eq!(transition_floor, 0);
        state
            .commit_session_change_owned(session_id, |staged| {
                staged.title = Some("Live title".to_string());
                stage_event(
                    &staged.runtime.history,
                    AgentEvent::SessionTitle {
                        title: "Live title".to_string(),
                        timestamp_ms: None,
                    },
                );
                Ok(StagedSessionChange::new(()))
            })
            .await
            .unwrap();

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), changes.recv())
                .await
                .unwrap()
                .unwrap(),
            1
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(50), changes.recv())
                .await
                .is_err(),
            "one state-affecting event must produce exactly one collection revision"
        );
        let terminal_transition = ProjectSessionTerminalTransition {
            entry_key: "terminal-entry".to_string(),
            revision: 0,
            session_id: "session-1".to_string(),
            status: SessionStatus::Completed,
            task: "finish the boundary".to_string(),
            title: Some("Boundary complete".to_string()),
            handoff_outcome: Some(HandoffOutcome::Ready),
            project: None,
        };
        assert_eq!(
            state.publish_project_session_update(Some(terminal_transition)),
            2
        );
        assert_eq!(changes.recv().await.unwrap(), 2);

        let usage_window = UsageWindow {
            start_ms: 0,
            end_ms: 86_400_000,
        };
        let snapshot = project_session_snapshot(&state, transition_floor, usage_window).await;
        assert_eq!(snapshot.stream_id, "test-stream");
        assert_eq!(snapshot.revision, 2);
        assert_eq!(snapshot.terminal_transition_floor, 0);
        assert_eq!(snapshot.terminal_transitions.len(), 1);
        assert_eq!(snapshot.terminal_transitions[0].revision, 2);
        assert_eq!(snapshot.project_usage["project-1"].total.tokens, 0);
        assert_eq!(snapshot.project_usage["project-1"].today.tokens, 0);
        assert_eq!(snapshot.overall_usage.total.tokens, 0);
        assert!(
            project_session_snapshot(&state, snapshot.revision, usage_window)
                .await
                .terminal_transitions
                .is_empty()
        );
        let event = project_session_snapshot_sse_event(&snapshot).unwrap();
        assert!(format!("{event:?}").contains("project_session_snapshot"));
    }

    #[tokio::test]
    async fn project_stream_coalesces_publications_into_advancing_transition_deltas() {
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let transition = |entry_key: &str| ProjectSessionTerminalTransition {
            entry_key: entry_key.to_string(),
            revision: 0,
            session_id: format!("session-{entry_key}"),
            status: SessionStatus::Completed,
            task: "finish the boundary".to_string(),
            title: None,
            handoff_outcome: Some(HandoffOutcome::Ready),
            project: None,
        };
        let usage_window = UsageWindow {
            start_ms: 0,
            end_ms: 86_400_000,
        };
        let (mut receiver, floor) = state.subscribe_project_session_changes();
        state.publish_project_session_update(Some(transition("one")));
        state.publish_project_session_update(Some(transition("two")));

        let first = next_project_session_snapshot(&mut receiver, &state, floor, usage_window)
            .await
            .unwrap();
        assert_eq!(first.revision, 2);
        assert_eq!(first.terminal_transition_floor, 0);
        assert_eq!(
            first
                .terminal_transitions
                .iter()
                .map(|transition| transition.entry_key.as_str())
                .collect::<Vec<_>>(),
            vec!["one", "two"]
        );

        state.publish_project_session_update(Some(transition("three")));
        let second =
            next_project_session_snapshot(&mut receiver, &state, first.revision, usage_window)
                .await
                .unwrap();
        assert_eq!(second.revision, 3);
        assert_eq!(second.terminal_transition_floor, 2);
        assert_eq!(second.terminal_transitions.len(), 1);
        assert_eq!(second.terminal_transitions[0].entry_key, "three");
    }

    #[tokio::test]
    async fn cancelled_project_registry_caller_cannot_split_committed_projection() {
        let (project_session_sender, _) = broadcast::channel(16);
        let mut changes = project_session_sender.subscribe();
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let session_guard = state.sessions.lock().await;
        let project = ProjectEntry {
            id: "project-1".to_string(),
            name: "pb".to_string(),
            path: "/workspace/pb".to_string(),
            repository_root: None,
            notify_on_finish: false,
        };
        let caller_state = state.clone();
        let caller = tokio::spawn(async move {
            caller_state
                .commit_project_registry_owned(
                    UsageWindow {
                        start_ms: 0,
                        end_ms: 86_400_000,
                    },
                    move || {
                        Ok(projects::ProjectRegistryMutation {
                            value: (),
                            projects: vec![project],
                        })
                    },
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if state.projects.try_lock().is_err() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        caller.abort();
        drop(session_guard);

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), changes.recv())
                .await
                .unwrap()
                .unwrap(),
            1
        );
        assert_eq!(state.projects.lock().await[0].id, "project-1");
    }

    #[tokio::test]
    async fn project_mutation_receipt_captures_its_own_revision_before_a_later_commit() {
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let session_guard = state.sessions.lock().await;
        let first_state = state.clone();
        let first = tokio::spawn(async move {
            first_state
                .commit_project_registry_owned(
                    UsageWindow {
                        start_ms: 0,
                        end_ms: 86_400_000,
                    },
                    || {
                        Ok(projects::ProjectRegistryMutation {
                            value: (),
                            projects: vec![ProjectEntry {
                                id: "project-1".to_string(),
                                name: "first".to_string(),
                                path: "/workspace/first".to_string(),
                                repository_root: None,
                                notify_on_finish: false,
                            }],
                        })
                    },
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if state.projects.try_lock().is_err() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let second_state = state.clone();
        let second = tokio::spawn(async move {
            second_state
                .commit_project_registry_owned(
                    UsageWindow {
                        start_ms: 0,
                        end_ms: 86_400_000,
                    },
                    || {
                        Ok(projects::ProjectRegistryMutation {
                            value: (),
                            projects: vec![ProjectEntry {
                                id: "project-2".to_string(),
                                name: "second".to_string(),
                                path: "/workspace/second".to_string(),
                                repository_root: None,
                                notify_on_finish: false,
                            }],
                        })
                    },
                )
                .await
        });
        drop(session_guard);

        let first_receipt = first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
        assert_eq!(first_receipt.snapshot.revision, 1);
        assert_eq!(first_receipt.snapshot.projects[0].id, "project-1");
        assert_eq!(state.project_session_revision.load(Ordering::SeqCst), 2);
        assert_eq!(state.projects.lock().await[0].id, "project-2");
    }

    #[tokio::test]
    async fn project_reconciliation_failure_is_revisioned_until_a_retry_clears_it() {
        let old_project = ProjectEntry {
            id: "project-1".to_string(),
            name: "old-name".to_string(),
            path: "/workspace/old".to_string(),
            repository_root: None,
            notify_on_finish: false,
        };
        let new_project = ProjectEntry {
            id: "project-1".to_string(),
            name: "new-name".to_string(),
            path: "/workspace/new".to_string(),
            repository_root: None,
            notify_on_finish: false,
        };
        let mut request = request(std::path::Path::new("/workspace/old"));
        request.workdir = Some(PathBuf::from("/workspace/old"));
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            Some(PathBuf::from("/workspace/old")),
            SessionStatus::Completed,
            Vec::new(),
        );
        let (session_id, mut session) = session_from_persisted(persisted);
        session.durable.project = Some(SessionProject {
            id: old_project.id.clone(),
            name: old_project.name.clone(),
            path: old_project.path.clone(),
        });
        let repository = Arc::new(FaultInjectingSessionRepository::default());
        repository.fail_next_save();
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: repository,
            projects: Arc::new(Mutex::new(vec![old_project])),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let usage_window = UsageWindow {
            start_ms: 0,
            end_ms: 86_400_000,
        };

        let failed_reconciliation = state
            .commit_project_registry_owned(usage_window, {
                let new_project = new_project.clone();
                move || {
                    Ok(projects::ProjectRegistryMutation {
                        value: (),
                        projects: vec![new_project],
                    })
                }
            })
            .await
            .unwrap();
        assert_eq!(failed_reconciliation.snapshot.revision, 1);
        assert_eq!(failed_reconciliation.snapshot.warnings.len(), 1);
        assert_eq!(
            state.sessions.lock().await[&session_id]
                .durable
                .project
                .as_ref()
                .unwrap()
                .name,
            "old-name"
        );

        let repaired = state
            .commit_project_registry_owned(usage_window, move || {
                Ok(projects::ProjectRegistryMutation {
                    value: (),
                    projects: vec![new_project],
                })
            })
            .await
            .unwrap();
        assert_eq!(repaired.snapshot.revision, 2);
        assert!(repaired.snapshot.warnings.is_empty());
        assert_eq!(
            state.sessions.lock().await[&session_id]
                .durable
                .project
                .as_ref()
                .unwrap()
                .name,
            "new-name"
        );
    }

    #[test]
    fn project_session_reconnect_cursor_is_scoped_to_the_server_process() {
        let mut headers = HeaderMap::new();
        headers.insert("last-event-id", "stream-a:7".parse().unwrap());
        assert_eq!(
            project_session_last_event_revision(&headers, "stream-a"),
            Some(7)
        );
        assert_eq!(
            project_session_last_event_revision(&headers, "stream-b"),
            None
        );
        assert_eq!(
            project_session_event_revision("stream-a:9", "stream-a"),
            Some(9)
        );
        assert_eq!(
            project_session_event_revision("stream-a:9", "stream-b"),
            None
        );
    }

    #[tokio::test]
    async fn inactive_pause_publishes_watch_finish_without_a_terminal_transition() {
        let mut request = request(std::path::Path::new("."));
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Running,
            Vec::new(),
        );
        let (session_id, mut session) = session_from_persisted(persisted);
        session.status = SessionStatus::Running;
        let mut publications = session.runtime.sender.subscribe();
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(FaultInjectingSessionRepository::default()),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };

        state
            .commit_session_change_owned(session_id, |session| {
                session.status = SessionStatus::Paused;
                Ok(StagedSessionChange::new(()))
            })
            .await
            .unwrap();

        assert!(matches!(
            publications.recv().await.unwrap(),
            SessionStreamPublication::Event(_)
        ));
        assert!(matches!(
            publications.recv().await.unwrap(),
            SessionStreamPublication::Snapshot(_)
        ));
        assert!(matches!(
            publications.recv().await.unwrap(),
            SessionStreamPublication::Finished
        ));
        let publication = state
            .project_session_publication
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(publication.terminal_transitions.is_empty());
    }

    #[tokio::test]
    async fn terminal_session_events_publish_explicit_finish_transitions() {
        let mut request = request(std::path::Path::new("."));
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Running,
            Vec::new(),
        );
        let (session_id, mut session) = session_from_persisted(persisted);
        session.status = SessionStatus::Running;
        let mut session_publications = session.runtime.sender.subscribe();
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let (mut changes, transition_floor) = state.subscribe_project_session_changes();
        state
            .commit_session_change_owned(session_id.clone(), |session| {
                session.status = SessionStatus::Completed;
                Ok(StagedSessionChange::new(()))
            })
            .await
            .unwrap();
        assert!(matches!(
            session_publications.recv().await.unwrap(),
            SessionStreamPublication::Event(_)
        ));
        assert!(matches!(
            session_publications.recv().await.unwrap(),
            SessionStreamPublication::Snapshot(_)
        ));
        assert!(matches!(
            session_publications.recv().await.unwrap(),
            SessionStreamPublication::Finished
        ));
        {
            let mut sessions = state.sessions.lock().await;
            sessions.remove(&session_id);
        }

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), changes.recv())
                .await
                .unwrap()
                .unwrap(),
            1
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(50), changes.recv())
                .await
                .is_err(),
            "one terminal event must produce exactly one collection revision"
        );
        let snapshot = project_session_snapshot(
            &state,
            transition_floor,
            UsageWindow {
                start_ms: 0,
                end_ms: 86_400_000,
            },
        )
        .await;
        assert_eq!(snapshot.terminal_transitions.len(), 1);
        assert_eq!(
            snapshot.terminal_transitions[0].status,
            SessionStatus::Completed
        );
        assert_eq!(snapshot.terminal_transitions[0].session_id, session_id);
        assert!(!snapshot.terminal_transitions[0].task.is_empty());
        assert!(snapshot.terminal_transitions[0].revision > transition_floor);
    }

    #[test]
    fn terminal_replay_detects_eviction_gaps_and_resumes_contiguously() {
        let mut history = (1..=3)
            .map(|sequence| {
                let mut envelope = EventEnvelope::new(AgentEvent::SessionTitle {
                    title: format!("title {sequence}"),
                    timestamp_ms: Some(sequence),
                });
                envelope.assign_sequence(sequence);
                envelope
            })
            .collect::<Vec<_>>();
        let TerminalReplay::Delta(replay) = terminal_replay_after(&history, 1).unwrap() else {
            panic!("known cursor should produce a delta");
        };
        assert_eq!(
            replay
                .iter()
                .map(|event| event.transcript.sequence)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );

        history.remove(1);
        let TerminalReplay::Reset {
            after_sequence,
            events,
        } = terminal_replay_after(&history, 1).unwrap()
        else {
            panic!("evicted cursor should produce a reset");
        };
        assert_eq!(after_sequence, 1);
        assert_eq!(events[0].transcript.sequence, 3);

        let TerminalReplay::Reset {
            after_sequence,
            events,
        } = terminal_replay_after(&history, 0).unwrap()
        else {
            panic!("an internal retained-history gap should produce a reset");
        };
        assert_eq!(after_sequence, 0);
        assert_eq!(
            events
                .iter()
                .map(|event| event.transcript.sequence)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[tokio::test]
    async fn lagged_terminal_watch_recovers_the_finish_phase_without_polling() {
        let mut request = request(std::path::Path::new("."));
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Completed,
            vec![EventEnvelope::new(AgentEvent::SessionTitle {
                title: "Finished boundary".to_string(),
                timestamp_ms: Some(1),
            })],
        );
        let (session_id, mut session) = session_from_persisted(persisted);
        let sender = SessionEventSender::new(2);
        session.runtime.sender = sender.clone();
        let snapshot = session_mutation_snapshot_for_session(&session_id, &session);
        let receiver = sender.subscribe();
        for _ in 0..3 {
            sender.publish_committed(SessionStreamPublication::Snapshot(snapshot.clone()));
        }
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(FaultInjectingSessionRepository::default()),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let (client, mut server) = tokio::net::UnixStream::pair().unwrap();
        let stream_state = state.clone();
        let stream_session_id = session_id.clone();
        let stream = tokio::spawn(async move {
            stream_session_watch(
                &mut server,
                &stream_state,
                &stream_session_id,
                receiver,
                Vec::new(),
                true,
            )
            .await
        });
        let mut client = BufReader::new(client);
        let mut line = String::new();
        client.read_line(&mut line).await.unwrap();
        assert!(matches!(
            serde_json::from_str::<RpcFrame>(&line).unwrap(),
            RpcFrame::SessionEvent { .. }
        ));
        line.clear();
        client.read_line(&mut line).await.unwrap();
        assert!(matches!(
            serde_json::from_str::<RpcFrame>(&line).unwrap(),
            RpcFrame::SessionFinished { .. }
        ));
        stream.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn acknowledged_watch_failures_use_stream_frames_not_second_responses() {
        let mut request = request(std::path::Path::new("."));
        request.session_id = "session-watch-phase".to_string();
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Paused,
            Vec::new(),
        );
        let (session_id, mut session) = session_from_persisted(persisted);
        session.status = SessionStatus::Running;
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(FaultInjectingSessionRepository::default()),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let (client, mut server) = tokio::net::UnixStream::pair().unwrap();
        let watch_state = state.clone();
        let watch_session_id = session_id.clone();
        let watch = tokio::spawn(async move {
            handle_session_watch(&mut server, 7, watch_state, watch_session_id).await
        });
        let mut client = BufReader::new(client);
        let mut line = String::new();
        client.read_line(&mut line).await.unwrap();
        assert!(matches!(
            serde_json::from_str::<RpcFrame>(&line).unwrap(),
            RpcFrame::Response { id: 7, .. }
        ));

        state.sessions.lock().await.remove(&session_id);
        line.clear();
        tokio::time::timeout(Duration::from_secs(1), client.read_line(&mut line))
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            serde_json::from_str::<RpcFrame>(&line).unwrap(),
            RpcFrame::StreamError { session_id: id, .. } if id == session_id
        ));
        watch.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn unary_rpc_failures_always_return_one_tagged_error() {
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            session_repository: Arc::new(FaultInjectingSessionRepository::default()),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let defaults = request(std::path::Path::new("."));
        assert!(
            dispatch_unary_rpc(
                state.clone(),
                defaults.clone(),
                "pb.session.list",
                serde_json::json!({ "unexpected": true }),
            )
            .await
            .is_err(),
            "no-argument RPC methods must reject surplus parameters"
        );
        let (client, server) = tokio::net::UnixStream::pair().unwrap();
        let rpc = tokio::spawn(handle_rpc_connection(server, state, defaults));
        let mut client = BufReader::new(client);
        client
            .get_mut()
            .write_all(
                format!(
                    "{}\n",
                    serde_json::json!({
                        "id": 23,
                        "method": "pb.session.start",
                        "params": []
                    })
                )
                .as_bytes(),
            )
            .await
            .unwrap();

        let mut line = String::new();
        client.read_line(&mut line).await.unwrap();
        assert!(matches!(
            serde_json::from_str::<RpcFrame>(&line).unwrap(),
            RpcFrame::Error { id: 23, .. }
        ));
        line.clear();
        assert_eq!(client.read_line(&mut line).await.unwrap(), 0);
        rpc.await.unwrap().unwrap();
    }

    #[test]
    fn durable_ids_are_collision_safe_and_namespaced() {
        let ids = (0..1_000).map(|_| new_session_id()).collect::<HashSet<_>>();
        assert_eq!(ids.len(), 1_000);
        assert!(ids.iter().all(|id| id.starts_with("session-")));
    }

    #[test]
    fn event_replay_resumes_after_a_known_cursor_and_resets_an_evicted_cursor() {
        let history = (0..305)
            .map(|index| {
                EventEnvelope::new(AgentEvent::UserMessage {
                    message_id: format!("message-{index}"),
                    message: format!("message {index}"),
                    timestamp_ms: Some(index),
                })
            })
            .collect::<Vec<_>>();
        let cursor = history[200].transcript.entry_key.as_str();

        assert_eq!(
            session_replay_window(&history, Some(cursor)),
            SessionReplayWindow {
                start: 201,
                reset_history: false,
            }
        );
        assert_eq!(
            session_replay_window(&history, Some("missing")),
            SessionReplayWindow {
                start: history.len(),
                reset_history: true,
            }
        );
        assert_eq!(
            session_replay_window(&history, None),
            SessionReplayWindow {
                start: history.len(),
                reset_history: true,
            }
        );
    }

    #[test]
    fn session_project_resolution_uses_registered_identity_without_path_guessing() {
        let repo = tempfile::tempdir().unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .current_dir(repo.path())
                .output()
                .unwrap()
                .status
                .success()
        );
        let nested = repo.path().join("packages/web");
        std::fs::create_dir_all(&nested).unwrap();
        let registered = ProjectEntry {
            id: "project-pb".to_string(),
            name: "pb".to_string(),
            path: repo.path().to_string_lossy().into_owned(),
            repository_root: None,
            notify_on_finish: false,
        };

        assert_eq!(
            resolve_session_project(std::slice::from_ref(&registered), Some(&nested)),
            Some(SessionProject {
                id: registered.id.clone(),
                name: "pb".to_string(),
                path: registered.path.clone(),
            })
        );

        let duplicate = ProjectEntry {
            id: "project-same-repository".to_string(),
            name: "same-repository".to_string(),
            ..registered.clone()
        };
        assert!(resolve_session_project(&[registered, duplicate], Some(&nested)).is_none());
    }

    #[test]
    fn staged_event_preserves_its_history_key_without_publishing() {
        let sender = SessionEventSender::new(4);
        let mut receiver = sender.subscribe();
        let history = StdMutex::new(Vec::new());
        let envelope = EventEnvelope::with_timestamp(AgentEvent::UserMessage {
            message_id: "message-1".to_string(),
            message: "Keep the boundary exact.".to_string(),
            timestamp_ms: None,
        });
        let entry_key = envelope.transcript.entry_key.clone();

        stage_event_envelope_linked(&history, envelope, vec!["older-entry".to_string()]);

        assert!(receiver.try_recv().is_err());
        let history = history.lock().unwrap();
        assert_eq!(history[0].transcript.entry_key, entry_key);
        assert_eq!(history[0].transcript.sequence, 1);
        assert_eq!(history[0].transcript.supersedes, vec!["older-entry"]);
    }

    #[test]
    fn poisoned_history_never_publishes_an_unsequenced_event() {
        let sender = SessionEventSender::new(4);
        let mut receiver = sender.subscribe();
        let history = StdMutex::new(Vec::new());
        let _ = std::panic::catch_unwind(|| {
            let _guard = history.lock().unwrap();
            panic!("poison event history");
        });

        stage_event(
            &history,
            AgentEvent::SessionTitle {
                title: "must not escape".to_string(),
                timestamp_ms: None,
            },
        );

        assert!(matches!(
            receiver.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn queued_session_transition_is_published_with_snapshot_semantics() {
        let mut request = request(std::path::Path::new("."));
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Completed,
            Vec::new(),
        );
        let (_session_id, mut session) = session_from_persisted(persisted);
        session.status = SessionStatus::Queued;
        let mut receiver = session.runtime.sender.subscribe();

        stage_session_state_changed(&session);

        assert!(receiver.try_recv().is_err());
        let history = session.runtime.history.lock().unwrap();
        let envelope = &history[0];
        assert!(envelope.requires_session_snapshot());
        assert!(matches!(
            &envelope.event,
            AgentEvent::SessionStateChanged {
                status: crate::events::SessionLifecycleStatus::Queued,
                ..
            }
        ));
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn running_session_message_endpoint_queues_conversation_input() {
        let mut request = request(std::path::Path::new("."));
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request.clone(),
            None,
            None,
            SessionStatus::Completed,
            Vec::new(),
        );
        let (session_id, mut session) = session_from_persisted(persisted);
        session.status = SessionStatus::Running;
        session
            .runtime
            .accepting_user_messages
            .store(true, Ordering::SeqCst);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let mut collection_changes = state.project_session_sender.subscribe();

        let response = send_session_message(
            Path(session_id.clone()),
            State((state.clone(), request.clone())),
            Json(SendSessionMessageRequest {
                message: "  Keep the API stable.  ".to_string(),
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(collection_changes.recv().await.unwrap(), 1);

        let accepting_user_messages = {
            let sessions = state.sessions.lock().await;
            let session = sessions.get(&session_id).unwrap();
            let pending = session.runtime.pending_user_messages.lock().unwrap();
            assert_eq!(pending.len(), 1);
            assert!(response.session.events.iter().any(|envelope| matches!(
                &envelope.event,
                AgentEvent::UserMessage { message_id, .. }
                    if message_id == &pending[0].message_id
            )));
            assert_eq!(pending[0].message, "Keep the API stable.");
            assert!(session.runtime.history.lock().unwrap().iter().any(|envelope| {
                matches!(
                    &envelope.event,
                    AgentEvent::UserMessage { message, .. } if message == "Keep the API stable."
                )
            }));
            Arc::clone(&session.runtime.accepting_user_messages)
        };
        accepting_user_messages.store(false, Ordering::SeqCst);

        let rejected = send_session_message(
            Path(session_id.clone()),
            State((state.clone(), request.clone())),
            Json(SendSessionMessageRequest {
                message: "This arrived after the final boundary.".to_string(),
            }),
        )
        .await;
        assert_eq!(rejected.unwrap_err().status, StatusCode::CONFLICT);

        let cancellation = cancel_session(Path(session_id), State((state, request)))
            .await
            .unwrap()
            .0;
        assert!(cancellation.session.cancel_requested);
        assert!(cancellation.session.events.iter().any(|envelope| matches!(
            envelope.event,
            AgentEvent::Correction {
                kind: crate::events::CorrectionKind::Lifecycle,
                ..
            }
        )));
    }

    #[tokio::test]
    async fn failed_persisted_delete_keeps_the_authoritative_session() {
        let not_a_repository = tempfile::tempdir().unwrap();
        let request = request(not_a_repository.path());
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            Some(not_a_repository.path().to_path_buf()),
            SessionStatus::Completed,
            Vec::new(),
        );
        let (session_id, session) = session_from_persisted(persisted);
        let (project_session_sender, _) = broadcast::channel(16);
        let mut changes = project_session_sender.subscribe();
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };

        assert!(matches!(
            delete_session_inner(state.clone(), &session_id).await,
            Err(DeleteSessionError::Internal(_))
        ));
        assert!(state.sessions.lock().await.contains_key(&session_id));
        assert!(matches!(
            changes.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));
        assert_eq!(state.project_session_revision.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn delete_returns_the_exact_committed_collection_projection() {
        let mut defaults = request(std::path::Path::new("."));
        defaults.workdir = None;
        defaults.branch = None;
        defaults.repository_less = true;
        let persisted = PersistedSession::from_parts(
            defaults.session_id.clone(),
            defaults.clone(),
            None,
            None,
            SessionStatus::Completed,
            Vec::new(),
        );
        let (session_id, session) = session_from_persisted(persisted);
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };

        let response = delete_session(
            Path(session_id.clone()),
            State((state.clone(), defaults)),
            Query(ProjectSessionQuery {
                usage_window_start_ms: 1,
                usage_window_end_ms: 86_400_001,
                last_event_id: None,
            }),
        )
        .await
        .unwrap()
        .0;

        assert_eq!(response.deletion.session_id, session_id);
        assert!(response.deletion.deleted);
        assert_eq!(response.snapshot.stream_id, "test-stream");
        assert_eq!(response.snapshot.revision, 1);
        assert_eq!(response.snapshot.terminal_transition_floor, 1);
        assert!(response.snapshot.terminal_transitions.is_empty());
        assert!(response.snapshot.sessions.is_empty());
        assert_eq!(response.snapshot.overall_usage.total.tokens, 0);
        assert!(!state.sessions.lock().await.contains_key(&session_id));
    }

    #[tokio::test]
    async fn cancelled_delete_caller_cannot_split_authoritative_state() {
        let mut request = request(std::path::Path::new("."));
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let mut persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Completed,
            Vec::new(),
        );
        persisted.project = Some(SessionProject {
            id: "project-1".to_string(),
            name: "pb".to_string(),
            path: "/workspace/pb".to_string(),
        });
        let metrics = SessionMetricsSnapshot {
            prompt_tokens: 7,
            generated_tokens: 3,
            ..Default::default()
        };
        persisted.metrics = Some(metrics.clone());
        persisted.usage_records = vec![metrics];
        let (session_id, session) = session_from_persisted(persisted);
        let (project_session_sender, _) = broadcast::channel(16);
        let mut changes = project_session_sender.subscribe();
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(vec![ProjectEntry {
                id: "project-1".to_string(),
                name: "pb".to_string(),
                path: "/workspace/pb".to_string(),
                repository_root: None,
                notify_on_finish: false,
            }])),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };

        let usage_guard = state.project_usage_windows.lock().await;
        let caller_state = state.clone();
        let caller_session_id = session_id.clone();
        let caller =
            tokio::spawn(
                async move { delete_session_inner(caller_state, &caller_session_id).await },
            );
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if state.sessions.try_lock().is_err() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("owned deletion should reach the accounting boundary");
        assert!(matches!(
            changes.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));
        let snapshot_state = state.clone();
        let mut snapshot_caller = tokio::spawn(async move {
            project_session_snapshot(
                &snapshot_state,
                1,
                UsageWindow {
                    start_ms: 0,
                    end_ms: 86_400_000,
                },
            )
            .await
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut snapshot_caller)
                .await
                .is_err(),
            "snapshots must wait for the complete deletion projection"
        );
        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        drop(usage_guard);

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), changes.recv())
                .await
                .unwrap()
                .unwrap(),
            1
        );
        assert!(!state.sessions.lock().await.contains_key(&session_id));
        let snapshot = tokio::time::timeout(Duration::from_secs(1), snapshot_caller)
            .await
            .unwrap()
            .unwrap();
        assert!(snapshot.sessions.is_empty());
        assert_eq!(snapshot.project_usage["project-1"].total.tokens, 0);
        assert_eq!(snapshot.project_usage["project-1"].today.tokens, 0);
    }

    fn request(workdir: &std::path::Path) -> AgentRequest {
        AgentRequest {
            task: "deliver safely".to_string(),
            turn_id: "turn-web-workflow".to_string(),
            intent: Some(crate::workflow::TurnIntent::Deliver),
            task_planning: crate::agent_core::TaskPlanningPreference::Auto,
            task_plan_rejected: None,
            task_planning_transcript: None,
            workflow_policy: Some(
                crate::workflow::WorkflowConfigDocument::default()
                    .compile()
                    .unwrap(),
            ),
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
            profile: AgentProfile::Build,
            infer_profile: false,
            sub_agent_depth: 0,
            repository_less: false,
            top_k: 1,
            seed: 0,
            environment: None,
            python_dependency_authority: Default::default(),
            environment_evidence_context: None,
            workspace_graph: None,
            repository_context: None,
            prior_check_evidence: crate::checks::CheckEvidenceLedger::default(),
            session_id: "session-web-workflow".to_string(),
            attachments: Vec::new(),
            goal_context: None,
            contract: None,
        }
    }

    fn two_goal_task_run(
        session_id: &str,
        workdir: &std::path::Path,
    ) -> crate::task_queue::MultiTaskRun {
        let policy = crate::task_queue::TaskConfigDocument::default()
            .compile()
            .unwrap();
        let authority = crate::task_queue::TaskPlanAuthority {
            source_intent: crate::task_queue::TaskSourceIntent::Build,
            task_planning_qualified: true,
            automatic_goal_selection_qualified: true,
        };
        let proposal: crate::task_queue::TaskPlanProposal =
            serde_json::from_value(serde_json::json!({
                "objective": "Complete two sequential Goals",
                "requirements": [
                    {"id": "r1", "description": "First Goal completes"},
                    {"id": "r2", "description": "Second Goal starts afterwards"}
                ],
                "tasks": [
                    {
                        "id": "goal-1",
                        "title": "First Goal",
                        "description": "Complete the first Goal",
                        "requirement_ids": ["r1"],
                        "acceptance_ids": ["a1"],
                        "scope_hints": ["src/first"],
                        "effort": "small",
                        "kind": "goal",
                        "goal_contract": {
                            "objective": "Complete the first Goal",
                            "criteria": [{
                                "text": "First Goal evidence is accepted",
                                "verifier": "review_required"
                            }],
                            "continuation": "review_plan_then_automatic"
                        }
                    },
                    {
                        "id": "goal-2",
                        "title": "Second Goal",
                        "description": "Complete the second Goal",
                        "requirement_ids": ["r2"],
                        "depends_on": ["goal-1"],
                        "acceptance_ids": ["a2"],
                        "scope_hints": ["src/second"],
                        "effort": "small",
                        "kind": "goal",
                        "goal_contract": {
                            "objective": "Complete the second Goal",
                            "criteria": [{
                                "text": "Second Goal evidence is accepted",
                                "verifier": "review_required"
                            }],
                            "continuation": "review_plan_then_automatic"
                        }
                    }
                ],
                "acceptance": [
                    {"id": "a1", "description": "First Goal is accepted"},
                    {"id": "a2", "description": "Second Goal is accepted"}
                ]
            }))
            .unwrap();
        let artifact = proposal.validate_and_compile(authority, &policy).unwrap();
        let plan = crate::workflow::ArtifactEnvelope::new("two-goal-plan", artifact).unwrap();
        let review = crate::workflow::ArtifactEnvelope::new(
            "two-goal-review",
            crate::task_queue::TaskPlanReviewArtifact {
                task_plan_sha256: plan.sha256.clone(),
                verdict: crate::task_queue::TaskPlanReviewVerdict::Pass,
                request_assessments: Vec::new(),
                audits: crate::task_queue::passing_task_plan_audits(),
                challenges: Vec::new(),
            },
        )
        .unwrap();
        let qualification_digest = "1".repeat(64);
        let qualification = crate::task_queue::TaskPlannerQualification::new(
            qualification_digest.clone(),
            qualification_digest.clone(),
            qualification_digest.clone(),
            qualification_digest,
            true,
            true,
        )
        .unwrap();
        crate::task_queue::MultiTaskRun::start(
            "two-goal-task",
            session_id,
            "turn-two-goal",
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
            workdir.to_string_lossy(),
            crate::task_queue::TaskRepositoryState::capture(workdir).unwrap(),
            crate::task_queue::TaskCoordinationCounters::default(),
            10,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn rpc_goal_accept_acknowledges_the_completed_goal_after_next_goal_activation() {
        let repo = tempfile::tempdir().unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .current_dir(repo.path())
                .output()
                .unwrap()
                .status
                .success()
        );
        let defaults = request(repo.path());
        let persisted = PersistedSession::from_parts(
            defaults.session_id.clone(),
            defaults.clone(),
            defaults.branch.clone(),
            defaults.workdir.clone(),
            SessionStatus::Paused,
            Vec::new(),
        );
        let (session_id, mut session) = session_from_persisted(persisted);
        let parent = crate::task_queue::MultiTaskCheckpoint::new(two_goal_task_run(
            &session_id,
            repo.path(),
        ))
        .unwrap();
        session.status = dispatch_multi_task_active(&mut session, &parent).unwrap();
        let mut first_goal = session.goal.take().unwrap().run;
        for criterion in &mut first_goal.criteria {
            criterion.status = crate::goal::GoalCriterionStatus::EvidenceReady;
            criterion.evidence_ids = vec!["review:first-goal".to_string()];
        }
        for milestone in &mut first_goal.milestones {
            milestone.status = crate::goal::GoalMilestoneStatus::Completed;
        }
        first_goal.active_milestone_id = None;
        first_goal.stage = crate::goal::GoalStage::AwaitingUserReview;
        first_goal.updated_at_ms = now_millis();
        let first_goal = crate::goal::GoalCheckpoint::new(first_goal).unwrap();
        let first_goal_id = first_goal.run.id.clone();
        let first_goal_sha256 = first_goal.sha256.clone();
        session.goal = Some(first_goal);
        sync_multi_task_goal_checkpoint(&mut session).unwrap();

        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let (client, server) = tokio::net::UnixStream::pair().unwrap();
        let rpc_state = state.clone();
        let rpc_defaults = defaults.clone();
        let rpc =
            tokio::spawn(
                async move { handle_rpc_connection(server, rpc_state, rpc_defaults).await },
            );
        let mut client = BufReader::new(client);
        client
            .get_mut()
            .write_all(
                format!(
                    "{}\n",
                    serde_json::json!({
                        "id": 41,
                        "method": "pb.goal.accept",
                        "params": {
                            "goal_id": first_goal_id,
                            "goal_sha256": first_goal_sha256
                        }
                    })
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        let mut response_line = String::new();
        client.read_line(&mut response_line).await.unwrap();
        rpc.await.unwrap().unwrap();
        let response: serde_json::Value = serde_json::from_str(&response_line).unwrap();
        assert_eq!(response["id"], 41);
        assert_eq!(response["frame"], "response");
        assert_eq!(response["result"]["response"]["goal_id"], first_goal_id);

        let sessions = state.sessions.lock().await;
        let session = &sessions[&session_id];
        let completed = session
            .completed_goals
            .iter()
            .find(|goal| goal.run.id == first_goal_id)
            .unwrap();
        assert_eq!(
            response["result"]["response"]["goal_sha256"],
            completed.sha256
        );
        assert_eq!(
            response["result"]["snapshot"]["session"]["session_id"],
            session_id
        );
        assert_ne!(session.goal.as_ref().unwrap().run.id, first_goal_id);
        assert_eq!(
            session.goal.as_ref().unwrap().run.objective,
            "Complete the second Goal"
        );
    }

    #[test]
    fn persisted_or_client_fields_cannot_disable_intrinsic_controller_actions() {
        let directory = tempfile::tempdir().unwrap();
        let template = request(directory.path());
        let mut value = serde_json::to_value(&template).unwrap();
        assert!(value.get("observation_rendering").is_none());
        value["action_elision"] = serde_json::json!("safe");
        value["observation_rendering"] = serde_json::json!("native");
        value["controller_delete_elision"] = serde_json::json!(false);

        let mut restored: AgentRequest = serde_json::from_value(value).unwrap();
        assert_eq!(
            restored.observation_rendering,
            crate::workflow::ObservationRendering::Native
        );

        apply_intrinsic_controller_actions(&mut restored);
        assert_eq!(
            restored.observation_rendering,
            crate::workflow::ObservationRendering::ControllerBlock
        );
    }

    #[test]
    fn project_usage_uses_authoritative_task_energy_and_wall_time_once() {
        let mut usage = ProjectUsageStats::default();
        usage.add_metrics(&SessionMetricsSnapshot {
            llm_runtime_ms: 8_000,
            tool_runtime_ms: 9_000,
            wall_runtime_ms: 10_000,
            llm_energy_joules: Some(80.0),
            tool_energy_joules: Some(70.0),
            total_energy_joules: Some(100.0),
            ..Default::default()
        });
        assert_eq!(usage.runtime_ms, 10_000);
        assert_eq!(usage.energy_joules, Some(100.0));

        usage.add_metrics(&SessionMetricsSnapshot {
            wall_runtime_ms: 1_000,
            llm_energy_joules: Some(20.0),
            tool_energy_joules: Some(20.0),
            ..Default::default()
        });
        assert_eq!(usage.energy_joules, Some(100.0));

        let mut measured_zero = ProjectUsageStats::default();
        measured_zero.add_metrics(&SessionMetricsSnapshot {
            wall_runtime_ms: 1_000,
            total_energy_joules: Some(0.0),
            ..Default::default()
        });
        assert_eq!(measured_zero.energy_joules, Some(0.0));
    }

    #[test]
    fn usage_summaries_cover_every_project_and_apportion_the_requested_day() {
        let projects = vec![
            ProjectEntry {
                id: "project-1".to_string(),
                name: "pb".to_string(),
                path: "/workspace/pb".to_string(),
                repository_root: None,
                notify_on_finish: false,
            },
            ProjectEntry {
                id: "project-empty".to_string(),
                name: "empty".to_string(),
                path: "/workspace/empty".to_string(),
                repository_root: None,
                notify_on_finish: false,
            },
        ];
        let mut request = request(std::path::Path::new("."));
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let mut persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Completed,
            Vec::new(),
        );
        persisted.project = Some(SessionProject {
            id: "project-1".to_string(),
            name: "pb".to_string(),
            path: "/workspace/pb".to_string(),
        });
        let metrics = SessionMetricsSnapshot {
            prompt_tokens: 80,
            generated_tokens: 20,
            tool_calls: 4,
            wall_runtime_ms: 120_000,
            started_at_ms: 60_000,
            ended_at_ms: 180_000,
            total_energy_joules: Some(120.0),
            ..Default::default()
        };
        persisted.metrics = Some(metrics.clone());
        persisted.usage_records = vec![metrics];
        let (session_id, session) = session_from_persisted(persisted);
        let sessions = HashMap::from([(session_id, session)]);
        let mut usage_windows = ProjectUsageWindowCache::default();

        let (overall, project_usage) = usage_summaries(
            &projects,
            &sessions,
            &mut usage_windows,
            UsageWindow {
                start_ms: 120_000,
                end_ms: 240_000,
            },
        );

        assert_eq!(overall.total.tokens, 100);
        assert_eq!(overall.today.tokens, 50);
        assert_eq!(overall.today.runtime_ms, 60_000);
        assert_eq!(overall.today.tool_calls, 2);
        assert_eq!(overall.today.energy_joules, Some(60.0));
        assert_eq!(project_usage["project-1"].total.tokens, 100);
        assert_eq!(project_usage["project-1"].today.tokens, 50);
        assert_eq!(
            project_usage["project-empty"].total.tokens,
            ProjectUsageStats::default().tokens
        );
        assert_eq!(project_usage["project-empty"].today.tokens, 0);
    }

    #[test]
    fn usage_window_cache_invalidates_and_stays_bounded() {
        let projects = vec![ProjectEntry {
            id: "project-1".to_string(),
            name: "pb".to_string(),
            path: "/workspace/pb".to_string(),
            repository_root: None,
            notify_on_finish: false,
        }];
        let sessions = HashMap::new();
        let mut usage_windows = ProjectUsageWindowCache::default();
        let window = UsageWindow {
            start_ms: 0,
            end_ms: 86_400_000,
        };
        let (_, initial) = usage_summaries(&projects, &sessions, &mut usage_windows, window);
        assert_eq!(initial["project-1"].today.tokens, 0);

        usage_windows.invalidate();
        assert!(usage_windows.entries.is_empty());

        for offset in 1..=MAX_PROJECT_USAGE_WINDOW_CACHE_ENTRIES + 4 {
            let start_ms = offset as u64 * 1_000;
            usage_summaries(
                &projects,
                &sessions,
                &mut usage_windows,
                UsageWindow {
                    start_ms,
                    end_ms: start_ms + 86_400_000,
                },
            );
        }
        assert_eq!(
            usage_windows.entries.len(),
            MAX_PROJECT_USAGE_WINDOW_CACHE_ENTRIES
        );
    }

    #[tokio::test]
    async fn restored_workflow_keeps_exact_checkpoint_and_projects_stage_separately() {
        let repo = tempfile::tempdir().unwrap();
        let output = Command::new("git")
            .arg("init")
            .current_dir(repo.path())
            .output()
            .unwrap();
        assert!(output.status.success());
        let request = request(repo.path());
        let repository =
            crate::workspace::RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        let run = crate::workflow::WorkflowRun::start(
            "workflow-web-restore",
            request.turn_id.clone(),
            request.task.clone(),
            request.workflow_policy.clone().unwrap(),
            repository,
        )
        .unwrap();
        let checkpoint = crate::workflow::WorkflowCheckpoint::new(run).unwrap();
        let mut persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request.clone(),
            request.branch.clone(),
            request.workdir.clone(),
            SessionStatus::Running,
            Vec::new(),
        );
        persisted.workflow = Some(checkpoint.clone());
        let (session_id, mut restored) = session_from_persisted(persisted);
        let (question_sender, _question_receiver) = std::sync::mpsc::channel();
        restored.runtime.pending_question = Some(PendingQuestionState {
            question_id: "question-web-restore".to_string(),
            question: "Which release should I target?".to_string(),
            choices: vec!["Next".to_string(), "Current".to_string()],
            responder: question_sender,
        });

        assert_eq!(restored.status, SessionStatus::Paused);
        assert_eq!(restored.workflow.as_ref(), Some(&checkpoint));
        assert_eq!(
            restored.request_template.workflow_checkpoint.as_ref(),
            Some(&checkpoint)
        );

        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), restored)]))),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let list = session_list_snapshot(&state).await;
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].status, SessionStatus::Paused);
        assert_eq!(
            list[0].workflow_stage,
            Some(crate::workflow::WorkflowStage::Planning)
        );
        assert_eq!(
            list[0].pending_question.as_ref().unwrap().question_id,
            "question-web-restore"
        );
        assert!(list[0].strict_workflow);
        let details = session_details_snapshot(&state, &session_id).await.unwrap();
        assert_eq!(
            details.pending_question.as_ref().unwrap().choices,
            ["Next", "Current"]
        );
        assert_eq!(details.workflow.unwrap().id, "workflow-web-restore");
        assert!(details.strict_workflow);
    }

    #[tokio::test]
    async fn cancelling_a_blocked_workflow_preserves_content_and_records_terminal_outcome() {
        let repo = tempfile::tempdir().unwrap();
        let output = Command::new("git")
            .arg("init")
            .current_dir(repo.path())
            .output()
            .unwrap();
        assert!(output.status.success());
        let request = request(repo.path());
        let repository =
            crate::workspace::RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        let mut run = crate::workflow::WorkflowRun::start(
            "workflow-web-cancel",
            request.turn_id.clone(),
            request.task.clone(),
            request.workflow_policy.clone().unwrap(),
            repository,
        )
        .unwrap();
        run.apply(crate::workflow::WorkflowEvent::Blocked {
            outcome: crate::workflow::WorkflowOutcome::ExecutorUnavailable,
            cause: crate::workflow::WorkflowBlockCause::ExecutorUnavailable,
            reason: "executor unavailable".to_string(),
        })
        .unwrap();
        let checkpoint = crate::workflow::WorkflowCheckpoint::new(run).unwrap();
        let mut persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request.clone(),
            request.branch.clone(),
            request.workdir.clone(),
            SessionStatus::Failed,
            Vec::new(),
        );
        persisted.workflow = Some(checkpoint);
        let (session_id, restored) = session_from_persisted(persisted);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), restored)]))),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        std::fs::write(repo.path().join("in-progress.txt"), "preserve me\n").unwrap();

        cancel_session_inner(state.clone(), session_id.clone())
            .await
            .unwrap();

        let sessions = state.sessions.lock().await;
        let session = sessions.get(&session_id).unwrap();
        assert_eq!(session.status, SessionStatus::Completed);
        assert!(session.workflow.is_none());
        assert_eq!(
            session.completed_workflows.last().unwrap().outcome,
            Some(crate::workflow::WorkflowOutcome::Cancelled)
        );
        assert_eq!(
            std::fs::read_to_string(repo.path().join("in-progress.txt")).unwrap(),
            "preserve me\n"
        );
        assert!(
            session
                .runtime
                .history
                .lock()
                .unwrap()
                .iter()
                .any(|envelope| matches!(
                    envelope.event,
                    AgentEvent::WorkflowCompleted {
                        outcome: crate::workflow::WorkflowOutcome::Cancelled,
                        ..
                    }
                ))
        );
    }

    #[tokio::test]
    async fn content_sensitive_blocks_require_a_fresh_current_files_restart() {
        let repo = tempfile::tempdir().unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .current_dir(repo.path())
                .output()
                .unwrap()
                .status
                .success()
        );
        let request = request(repo.path());
        let original_turn = request.turn_id.clone();
        let repository =
            crate::workspace::RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        let mut run = crate::workflow::WorkflowRun::start(
            "workflow-web-restart",
            request.turn_id.clone(),
            request.task.clone(),
            request.workflow_policy.clone().unwrap(),
            repository,
        )
        .unwrap();
        run.apply(crate::workflow::WorkflowEvent::Blocked {
            outcome: crate::workflow::WorkflowOutcome::CommitBlocked,
            cause: crate::workflow::WorkflowBlockCause::RepositoryContentChanged,
            reason: "repository content changed while the read-only PlanReview stage was running"
                .to_string(),
        })
        .unwrap();
        let checkpoint = crate::workflow::WorkflowCheckpoint::new(run).unwrap();
        let mut persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request.clone(),
            request.branch.clone(),
            request.workdir.clone(),
            SessionStatus::Failed,
            Vec::new(),
        );
        persisted.workflow = Some(checkpoint);
        let (session_id, restored) = session_from_persisted(persisted);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), restored)]))),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(Vec::new())),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };

        assert!(
            resume_session_inner(state.clone(), session_id.clone())
                .await
                .is_err()
        );
        {
            let mut sessions = state.sessions.lock().await;
            let session = sessions.get_mut(&session_id).unwrap();
            prepare_blocked_workflow_restart(session, &session_id).unwrap();
            assert_eq!(session.status, SessionStatus::Queued);
            assert!(session.workflow.is_none());
            assert_eq!(session.completed_workflows.len(), 1);
            assert_eq!(
                session.completed_workflows[0].recovery,
                Some(crate::workflow::WorkflowRecovery::RestartFromCurrentFiles)
            );
            assert_ne!(session.request_template.turn_id, original_turn);
            assert!(session.request_template.workflow_checkpoint.is_none());
            assert!(session.request_template.repository_context.is_none());
            assert!(
                session
                    .runtime
                    .history
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|envelope| {
                        matches!(
                            &envelope.event,
                            AgentEvent::Correction { summary, .. }
                                if summary == "Restarting delivery from current files"
                        )
                    })
            );
        }
    }

    #[tokio::test]
    async fn goal_start_projects_approval_state_and_rejects_stale_mutations() {
        let repo = tempfile::tempdir().unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .current_dir(repo.path())
                .output()
                .unwrap()
                .status
                .success()
        );
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(vec![ProjectEntry {
                id: "project-test".to_string(),
                name: "test".to_string(),
                path: repo.path().to_string_lossy().into_owned(),
                repository_root: None,
                notify_on_finish: false,
            }])),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let defaults = request(repo.path());
        let missing_session = start_goal_inner(
            state.clone(),
            defaults.clone(),
            StartGoalRequest {
                session_id: Some("missing-session".to_string()),
                objective: "Do not create an implicit session".to_string(),
                criteria: Vec::new(),
                continuation: crate::goal::GoalContinuationPolicy::ReviewPlanThenAutomatic,
                budget: None,
                project_id: None,
                workdir: Some(repo.path().to_string_lossy().into_owned()),
                model: None,
            },
        )
        .await;
        let Err(missing_session) = missing_session else {
            panic!("a targeted goal must not create a missing session");
        };
        assert!(missing_session.to_string().contains("session not found"));
        assert!(state.sessions.lock().await.is_empty());
        let missing_http = start_session_goal(
            Path("missing-session".to_string()),
            State((state.clone(), defaults.clone())),
            Json(StartGoalRequest {
                session_id: None,
                objective: "Do not create an implicit session".to_string(),
                criteria: Vec::new(),
                continuation: crate::goal::GoalContinuationPolicy::ReviewPlanThenAutomatic,
                budget: None,
                project_id: None,
                workdir: None,
                model: None,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(missing_http.status, StatusCode::NOT_FOUND);
        assert_eq!(missing_http.code, "session_not_found");
        assert!(state.sessions.lock().await.is_empty());

        let response = start_goal_inner(
            state.clone(),
            defaults.clone(),
            StartGoalRequest {
                session_id: None,
                objective: "Ship durable goals".to_string(),
                criteria: vec![crate::goal::GoalCriterionInput {
                    text: "Persist checkpoints".to_string(),
                    verifier: crate::goal::GoalVerifier::ReviewRequired,
                }],
                continuation: crate::goal::GoalContinuationPolicy::ReviewPlanThenAutomatic,
                budget: None,
                project_id: None,
                workdir: Some(repo.path().to_string_lossy().into_owned()),
                model: None,
            },
        )
        .await
        .unwrap()
        .response;

        let details = session_details_snapshot(&state, &response.session_id)
            .await
            .unwrap();
        assert!(details.active_goal);
        assert_eq!(details.status, SessionStatus::Paused);
        let goal = details.goal.as_ref().unwrap();
        assert_eq!(goal.run.stage, crate::goal::GoalStage::AwaitingPlanApproval);
        assert_eq!(goal.run.budget, crate::goal::GoalBudget::standard());
        assert_eq!(goal.sha256, response.goal_sha256);

        let stale = mutate_active_goal(&state, &response.goal_id, "stale", |_, _| Ok(())).await;
        assert_eq!(stale.unwrap_err().status, StatusCode::CONFLICT);

        let before_rejected_mutation = session_details_snapshot(&state, &response.session_id)
            .await
            .unwrap();
        let rejected = mutate_active_goal(
            &state,
            &response.goal_id,
            &response.goal_sha256,
            |session, run| {
                session.task = "partially mutated task".to_string();
                stage_event(
                    &session.runtime.history,
                    AgentEvent::GoalPauseRequested {
                        goal_id: run.id.clone(),
                        timestamp_ms: Some(now_millis()),
                    },
                );
                bail!("reject the staged mutation")
            },
        )
        .await;
        assert_eq!(rejected.unwrap_err().status, StatusCode::CONFLICT);
        let after_rejected_mutation = session_details_snapshot(&state, &response.session_id)
            .await
            .unwrap();
        assert_eq!(after_rejected_mutation.task, before_rejected_mutation.task);
        assert_eq!(
            after_rejected_mutation.events.len(),
            before_rejected_mutation.events.len()
        );
        assert_eq!(
            after_rejected_mutation.revision,
            before_rejected_mutation.revision
        );
        assert_eq!(
            after_rejected_mutation.goal.as_ref().unwrap().sha256,
            response.goal_sha256
        );

        let invalid_checkpoint = mutate_active_goal(
            &state,
            &response.goal_id,
            &response.goal_sha256,
            |session, run| {
                run.objective.clear();
                stage_event(
                    &session.runtime.history,
                    AgentEvent::GoalPauseRequested {
                        goal_id: run.id.clone(),
                        timestamp_ms: Some(now_millis()),
                    },
                );
                Ok(())
            },
        )
        .await;
        assert_eq!(
            invalid_checkpoint.unwrap_err().status,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        let after_invalid_checkpoint = session_details_snapshot(&state, &response.session_id)
            .await
            .unwrap();
        assert_eq!(
            after_invalid_checkpoint.events.len(),
            before_rejected_mutation.events.len()
        );
        assert_eq!(
            after_invalid_checkpoint.revision,
            before_rejected_mutation.revision
        );
        assert_eq!(
            after_invalid_checkpoint.goal.as_ref().unwrap().sha256,
            response.goal_sha256
        );

        let invalid_persistence_session_id = "invalid/session".to_string();
        {
            let mut sessions = state.sessions.lock().await;
            let session = sessions.remove(&response.session_id).unwrap();
            sessions.insert(invalid_persistence_session_id.clone(), session);
        }
        let before_persistence_failure =
            session_details_snapshot(&state, &invalid_persistence_session_id)
                .await
                .unwrap();
        let persistence_failure = mutate_active_goal(
            &state,
            &response.goal_id,
            &response.goal_sha256,
            |session, run| {
                session.title = Some("must not become live".to_string());
                stage_event(
                    &session.runtime.history,
                    AgentEvent::GoalPauseRequested {
                        goal_id: run.id.clone(),
                        timestamp_ms: Some(now_millis()),
                    },
                );
                Ok(())
            },
        )
        .await;
        assert_eq!(
            persistence_failure.unwrap_err().status,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        let after_persistence_failure =
            session_details_snapshot(&state, &response.session_id).await;
        assert!(after_persistence_failure.is_none());
        let after_persistence_failure =
            session_details_snapshot(&state, &invalid_persistence_session_id)
                .await
                .unwrap();
        assert_eq!(
            after_persistence_failure.title,
            before_persistence_failure.title
        );
        assert_eq!(
            after_persistence_failure
                .events
                .iter()
                .map(|event| event.transcript.entry_key.as_str())
                .collect::<Vec<_>>(),
            before_persistence_failure
                .events
                .iter()
                .map(|event| event.transcript.entry_key.as_str())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            after_persistence_failure.revision,
            before_persistence_failure.revision
        );
        {
            let mut sessions = state.sessions.lock().await;
            let session = sessions.remove(&invalid_persistence_session_id).unwrap();
            sessions.insert(response.session_id.clone(), session);
        }

        let rejected_accept = accept_goal_inner(
            &state,
            &response.goal_id,
            &GoalDigestRequest {
                goal_sha256: response.goal_sha256.clone(),
                plan_sha256: None,
            },
        )
        .await;
        assert_eq!(rejected_accept.unwrap_err().status, StatusCode::CONFLICT);
        let after_rejected_accept = session_details_snapshot(&state, &response.session_id)
            .await
            .unwrap();
        assert!(after_rejected_accept.active_goal);
        assert_eq!(
            after_rejected_accept.goal.as_ref().unwrap().sha256,
            response.goal_sha256
        );
        assert_eq!(
            after_rejected_accept.revision,
            before_rejected_mutation.revision
        );

        let unchanged = get_goal(Path(response.goal_id), State((state, defaults)))
            .await
            .unwrap()
            .0;
        assert_eq!(unchanged.sha256, response.goal_sha256);
    }

    #[tokio::test]
    async fn initial_goal_draft_can_change_before_approval_and_cancel_archives_it() {
        let repo = tempfile::tempdir().unwrap();
        assert!(
            Command::new("git")
                .arg("init")
                .current_dir(repo.path())
                .output()
                .unwrap()
                .status
                .success()
        );
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            session_repository: Arc::new(session_store::GitNoteSessionRepository),
            projects: Arc::new(Mutex::new(vec![ProjectEntry {
                id: "project-test".to_string(),
                name: "test".to_string(),
                path: repo.path().to_string_lossy().into_owned(),
                repository_root: None,
                notify_on_finish: false,
            }])),
            project_usage_windows: Arc::new(Mutex::new(ProjectUsageWindowCache::default())),
            project_session_stream_id: Arc::new("test-stream".to_string()),
            project_session_revision: Arc::new(AtomicU64::new(0)),
            project_session_publication: Arc::new(StdMutex::new(
                ProjectSessionPublication::default(),
            )),
            project_session_sender: broadcast::channel(16).0,
            sleep_prevention: Arc::new(StdMutex::new(SleepPrevention::new(false))),
            tailscale: Arc::new(StdMutex::new(crate::tailscale::TailscaleIntegration::new(
                8311, 8311, false,
            ))),
            web_listen: "127.0.0.1".to_string(),
        };
        let defaults = request(repo.path());
        let started = start_goal_inner(
            state.clone(),
            defaults.clone(),
            StartGoalRequest {
                session_id: None,
                objective: "Original goal".to_string(),
                criteria: Vec::new(),
                continuation: crate::goal::GoalContinuationPolicy::ReviewPlanThenAutomatic,
                budget: None,
                project_id: None,
                workdir: Some(repo.path().to_string_lossy().into_owned()),
                model: None,
            },
        )
        .await
        .unwrap()
        .response;
        let revised = revise_goal_draft(
            Path(started.goal_id.clone()),
            State((state.clone(), defaults.clone())),
            Json(GoalAmendmentRequest {
                goal_sha256: started.goal_sha256.clone(),
                objective: "Revised goal".to_string(),
                criteria: vec![crate::goal::GoalCriterionInput {
                    text: "Revised evidence".to_string(),
                    verifier: crate::goal::GoalVerifier::ReviewRequired,
                }],
                continuation: crate::goal::GoalContinuationPolicy::ManualMilestones,
                budget: Some(crate::goal::GoalBudget::standard()),
            }),
        )
        .await
        .unwrap()
        .0;
        let revised_goal_sha256 = revised.session.goal.as_ref().unwrap().sha256.clone();
        assert_ne!(revised_goal_sha256, started.goal_sha256);

        let cancelled = cancel_goal(
            Path(started.goal_id.clone()),
            State((state.clone(), defaults.clone())),
            Json(GoalDigestRequest {
                goal_sha256: revised_goal_sha256,
                plan_sha256: None,
            }),
        )
        .await
        .unwrap()
        .0;
        let details = session_details_snapshot(&state, &cancelled.session.session_id)
            .await
            .unwrap();
        assert!(!details.active_goal);
        assert_eq!(details.status, SessionStatus::Completed);
        assert!(details.events.iter().any(|envelope| matches!(
            envelope.event,
            AgentEvent::SessionStateChanged {
                status: crate::events::SessionLifecycleStatus::Completed,
                ..
            }
        )));
        let snapshot = project_session_snapshot(
            &state,
            0,
            UsageWindow {
                start_ms: 0,
                end_ms: 86_400_000,
            },
        )
        .await;
        assert!(snapshot.terminal_transitions.iter().any(|transition| {
            transition.session_id == cancelled.session.session_id
                && transition.status == SessionStatus::Completed
        }));
        let restarted = start_session_goal(
            Path(cancelled.session.session_id.clone()),
            State((state.clone(), defaults.clone())),
            Json(StartGoalRequest {
                session_id: None,
                objective: "Follow-up goal".to_string(),
                criteria: Vec::new(),
                continuation: crate::goal::GoalContinuationPolicy::ReviewPlanThenAutomatic,
                budget: None,
                project_id: None,
                workdir: None,
                model: None,
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(restarted.session.session_id, cancelled.session.session_id);
        assert!(restarted.session.active_goal);
        assert_eq!(
            restarted.session.goal.as_ref().unwrap().run.objective,
            "Follow-up goal"
        );
        let archived = get_goal(Path(started.goal_id), State((state, defaults)))
            .await
            .unwrap()
            .0;
        assert_eq!(archived.run.stage, crate::goal::GoalStage::Cancelled);
        assert_eq!(archived.run.objective, "Revised goal");
    }

    #[test]
    fn started_event_projects_the_resolved_focus_root_into_live_session_state() {
        let mut request = request(std::path::Path::new("."));
        request.workdir = None;
        request.branch = None;
        request.repository_less = true;
        let persisted = PersistedSession::from_parts(
            request.session_id.clone(),
            request,
            None,
            None,
            SessionStatus::Queued,
            Vec::new(),
        );
        let (_session_id, mut session) = session_from_persisted(persisted);
        let event = AgentEvent::Started {
            task: "review the boundary".to_string(),
            model: "local".to_string(),
            profile: AgentProfile::Review,
            workspace: "/workspace".to_string(),
            focus_root: Some("/workspace/project".to_string()),
            branch: "feature/boundary".to_string(),
            attachments: Vec::new(),
            timestamp_ms: Some(1),
        };
        let location = started_session_location(&event);

        apply_started_session_location(&mut session, location.as_ref());

        assert_eq!(session.branch.as_deref(), Some("feature/boundary"));
        assert_eq!(
            session.workdir.as_deref(),
            Some(std::path::Path::new("/workspace/project"))
        );
    }
}
