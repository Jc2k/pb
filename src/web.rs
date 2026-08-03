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
pub struct InlineAttachment {
    pub name: String,
    pub mime: String,
    pub base64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub struct ContinueSessionRequest {
    pub task: String,
    #[serde(default)]
    pub intent: Option<crate::workflow::TurnIntent>,
    #[serde(default)]
    pub proposal_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendSessionMessageRequest {
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendSessionMessageResponse {
    pub message_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerQuestionRequest {
    pub question_id: String,
    pub answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerSessionQuestionRequest {
    pub session_id: String,
    pub question_id: String,
    pub answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub struct GoalDigestRequest {
    pub goal_sha256: String,
    #[serde(default)]
    pub plan_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoalRpcMutationRequest {
    pub goal_id: String,
    pub goal_sha256: String,
    #[serde(default)]
    pub plan_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub struct GoalResponse {
    pub session_id: String,
    pub goal_id: String,
    pub goal_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionResponse {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionListItem {
    pub session_id: String,
    pub task: String,
    pub title: Option<String>,
    pub running: bool,
    pub paused: bool,
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
    pub revision: u64,
}

#[derive(Debug, Clone, Serialize)]
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
}

#[derive(Debug, Clone, Serialize)]
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
pub struct ProjectUsageStats {
    pub tokens: usize,
    pub runtime_ms: u64,
    pub tool_calls: usize,
    pub energy_kwh: Option<f64>,
    pub energy_joules: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
pub struct SessionDetails {
    pub session_id: String,
    pub task: String,
    pub title: Option<String>,
    pub running: bool,
    pub paused: bool,
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

#[derive(Debug, Clone, Serialize)]
struct SessionStreamSnapshot {
    session: SessionDetails,
    reset_history: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingQuestionView {
    pub question_id: String,
    pub question: String,
    pub choices: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteSessionResponse {
    pub session_id: String,
    pub deleted: bool,
    pub cleanup_warnings: Vec<String>,
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
struct RpcRequest {
    id: u64,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct RpcResponse<T: Serialize> {
    id: u64,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct RpcNotification<T: Serialize> {
    method: &'static str,
    params: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchSessionRequest {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionFinished {
    pub session_id: String,
    pub running: bool,
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

#[derive(Debug)]
struct SessionState {
    task: String,
    title: Option<String>,
    branch: Option<String>,
    workdir: Option<PathBuf>,
    durable: DurableSessionProjection,
    request_template: AgentRequest,
    running: bool,
    paused: bool,
    status: SessionStatus,
    pending_question: Option<PendingQuestionState>,
    sender: broadcast::Sender<EventEnvelope>,
    history: Arc<StdMutex<Vec<EventEnvelope>>>,
    metrics: Option<SessionMetricsSnapshot>,
    usage_records: Arc<StdMutex<Vec<SessionMetricsSnapshot>>>,
    workflow: Option<crate::workflow::WorkflowCheckpoint>,
    completed_workflows: Vec<crate::workflow::WorkflowSummary>,
    goal: Option<crate::goal::GoalCheckpoint>,
    completed_goals: Vec<crate::goal::GoalCheckpoint>,
    multi_task: Option<crate::task_queue::MultiTaskCheckpoint>,
    completed_multi_tasks: Vec<crate::task_queue::MultiTaskCheckpoint>,
    pending_user_messages: Arc<StdMutex<VecDeque<QueuedUserMessage>>>,
    accepting_user_messages: Arc<AtomicBool>,
    pause_token: Arc<AtomicBool>,
    cancel_token: Arc<AtomicBool>,
    started_at_ms: u64,
    updated_at_ms: u64,
}

#[derive(Debug, Default)]
struct ProjectSessionPublication {
    terminal_transitions: VecDeque<ProjectSessionTerminalTransition>,
}

#[derive(Debug, Clone)]
struct AppState {
    sessions: Arc<Mutex<HashMap<String, SessionState>>>,
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
    fn publish_project_session_update(
        &self,
        mut terminal_transition: Option<ProjectSessionTerminalTransition>,
    ) -> u64 {
        let mut publication = self
            .project_session_publication
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let revision = self
            .project_session_revision
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
        let _ = self.project_session_sender.send(revision);
        revision
    }

    fn publish_project_session_change(&self) -> u64 {
        self.publish_project_session_update(None)
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

    async fn terminal_transition(
        &self,
        session_id: &str,
        entry_key: String,
        status: SessionStatus,
    ) -> Option<ProjectSessionTerminalTransition> {
        let projects = self.projects.lock().await;
        let sessions = self.sessions.lock().await;
        let session = sessions.get(session_id)?;
        let (title, handoff_outcome, _) = session_history_summary(session);
        let project = session.durable.project.as_ref().and_then(|stored| {
            projects
                .iter()
                .find(|project| project.id == stored.id)
                .cloned()
        });
        Some(ProjectSessionTerminalTransition {
            entry_key,
            revision: 0,
            session_id: session_id.to_string(),
            status,
            task: session.task.clone(),
            title,
            handoff_outcome,
            project,
        })
    }

    fn watch_session_changes(
        &self,
        session_id: String,
        mut receiver: broadcast::Receiver<EventEnvelope>,
    ) {
        let state = self.clone();
        tokio::spawn(async move {
            loop {
                match receiver.recv().await {
                    Ok(envelope) if envelope.affects_project_session_snapshot() => {
                        let status = match envelope.event {
                            AgentEvent::SessionStateChanged {
                                status: crate::events::SessionLifecycleStatus::Completed,
                                ..
                            } => Some(SessionStatus::Completed),
                            AgentEvent::SessionStateChanged {
                                status: crate::events::SessionLifecycleStatus::Failed,
                                ..
                            } => Some(SessionStatus::Failed),
                            _ => None,
                        };
                        let transition = if let Some(status) = status {
                            state
                                .terminal_transition(
                                    &session_id,
                                    envelope.transcript.entry_key,
                                    status,
                                )
                                .await
                        } else {
                            None
                        };
                        state.publish_project_session_update(transition);
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        state.publish_project_session_change();
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
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

    {
        let sessions = state.sessions.lock().await;
        for (session_id, session) in sessions.iter() {
            state.watch_session_changes(session_id.clone(), session.sender.subscribe());
        }
    }
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
        .map_err(|error| ApiError::new(StatusCode::BAD_REQUEST, "session_start_rejected", error))
}

async fn start_session_inner(
    state: AppState,
    defaults: AgentRequest,
    req: StartSessionRequest,
) -> Result<SessionResponse> {
    let session_id = new_session_id();
    let (sender, _) = broadcast::channel(256);
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
        task: request.task.clone(),
        title: None,
        branch: request.branch.clone(),
        workdir: request.workdir.clone(),
        durable: durable.clone(),
        request_template: request.clone(),
        running: false,
        paused: false,
        status: SessionStatus::Queued,
        pending_question: None,
        sender: sender.clone(),
        history: Arc::new(StdMutex::new(Vec::new())),
        metrics: None,
        usage_records: Arc::clone(&usage_records),
        workflow: None,
        completed_workflows: Vec::new(),
        goal: None,
        completed_goals: Vec::new(),
        multi_task: None,
        completed_multi_tasks: Vec::new(),
        pending_user_messages: Arc::new(StdMutex::new(VecDeque::new())),
        accepting_user_messages: Arc::new(AtomicBool::new(false)),
        pause_token: Arc::new(AtomicBool::new(false)),
        cancel_token: Arc::new(AtomicBool::new(false)),
        started_at_ms: now,
        updated_at_ms: now,
    };

    {
        let mut sessions = state.sessions.lock().await;
        if sessions.contains_key(&session_id) {
            bail!("generated session id collision");
        }
        sessions.insert(session_id.clone(), session);
    }
    state.watch_session_changes(session_id.clone(), sender.subscribe());
    state.publish_project_session_change();

    let empty_history = StdMutex::new(Vec::new());
    persist_session_snapshot(
        &session_id,
        &request,
        request.branch.clone(),
        request.workdir.clone(),
        SessionStatus::Queued,
        &empty_history,
        &usage_records,
        None,
        Vec::new(),
        None,
        Vec::new(),
        None,
        Vec::new(),
        durable,
    );

    dispatch_next_session(state.clone());

    Ok(SessionResponse { session_id })
}

async fn start_goal(
    State((state, defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<StartGoalRequest>,
) -> Result<Json<GoalResponse>, ApiError> {
    start_goal_inner(state, defaults, req)
        .await
        .map(Json)
        .map_err(|error| {
            tracing::warn!(%error, "failed to start goal");
            ApiError::new(StatusCode::BAD_REQUEST, "goal_start_rejected", error)
        })
}

async fn start_goal_inner(
    state: AppState,
    defaults: AgentRequest,
    req: StartGoalRequest,
) -> Result<GoalResponse> {
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
    let mut sessions = state.sessions.lock().await;
    let requested_existing_session = req.session_id.is_some();
    let session_id = req.session_id.clone().unwrap_or_else(|| {
        loop {
            let candidate = new_session_id();
            if !sessions.contains_key(&candidate) {
                break candidate;
            }
        }
    });
    if let Some(session) = sessions.get_mut(&session_id) {
        if !requested_existing_session {
            bail!("generated session id collision");
        }
        if req.project_id.is_some() || req.workdir.is_some() {
            bail!("an existing session already determines the goal project");
        }
        if session.goal.is_some() || session.running || session.pending_question.is_some() {
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
        session.task = objective.clone();
        session.title = Some(objective.clone());
        session.request_template.task = objective.clone();
        session.request_template.intent = Some(crate::workflow::TurnIntent::Discuss);
        session.request_template.goal_context = None;
        session.request_template.workflow_checkpoint = None;
        session.request_template.workflow_stage = None;
        session.durable.pending_goal_proposal = None;
        session.durable.pending_goal_change = None;
        session.goal = Some(checkpoint.clone());
        session.running = false;
        session.paused = true;
        session.status = SessionStatus::Paused;
        session.updated_at_ms = now;
        publish_goal_started(session, &checkpoint);
        persist_live_session(&session_id, session);
        return Ok(GoalResponse {
            session_id,
            goal_id,
            goal_sha256: checkpoint.sha256,
        });
    }

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
    let (sender, _) = broadcast::channel(256);
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
    let session_sender = sender.clone();
    let mut session = SessionState {
        task: objective.clone(),
        title: Some(objective),
        branch: None,
        workdir: Some(workdir),
        durable: DurableSessionProjection {
            project,
            ..DurableSessionProjection::default()
        },
        request_template: request,
        running: false,
        paused: true,
        status: SessionStatus::Paused,
        pending_question: None,
        sender,
        history,
        metrics: None,
        usage_records,
        workflow: None,
        completed_workflows: Vec::new(),
        goal: Some(checkpoint.clone()),
        completed_goals: Vec::new(),
        multi_task: None,
        completed_multi_tasks: Vec::new(),
        pending_user_messages: Arc::new(StdMutex::new(VecDeque::new())),
        accepting_user_messages: Arc::new(AtomicBool::new(false)),
        pause_token: Arc::new(AtomicBool::new(false)),
        cancel_token: Arc::new(AtomicBool::new(false)),
        started_at_ms: now,
        updated_at_ms: now,
    };
    publish_goal_started(&mut session, &checkpoint);
    persist_live_session(&session_id, &session);
    sessions.insert(session_id.clone(), session);
    drop(sessions);
    state.watch_session_changes(session_id.clone(), session_sender.subscribe());
    state.publish_project_session_change();
    Ok(GoalResponse {
        session_id,
        goal_id,
        goal_sha256: checkpoint.sha256,
    })
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
    publish_event(
        &session.sender,
        &session.history,
        AgentEvent::GoalStarted {
            goal_id: checkpoint.run.id.clone(),
            objective: checkpoint.run.objective.clone(),
            plan_sha256: checkpoint.run.plan_sha256.clone(),
            timestamp_ms: Some(now_millis()),
        },
    );
    publish_event(
        &session.sender,
        &session.history,
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
) -> Result<Json<GoalResponse>, ApiError> {
    let response = mutate_active_goal(&state, &id, &req.goal_sha256, |session, run| {
        let plan_sha256 = req
            .plan_sha256
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("plan_sha256 is required"))?;
        run.approve_plan(plan_sha256, now_millis())?;
        configure_goal_milestone_request(session, run)?;
        session.status = SessionStatus::Queued;
        session.paused = false;
        publish_event(
            &session.sender,
            &session.history,
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
    dispatch_next_session(state);
    Ok(Json(response))
}

async fn revise_goal_draft(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalAmendmentRequest>,
) -> Result<Json<GoalResponse>, ApiError> {
    mutate_active_goal(&state, &id, &req.goal_sha256, |session, run| {
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
        publish_event(
            &session.sender,
            &session.history,
            AgentEvent::GoalPlanAwaitingApproval {
                goal_id: run.id.clone(),
                plan_sha256: run.plan_sha256.clone(),
                milestones: run.milestones.len(),
                timestamp_ms: Some(now_millis()),
            },
        );
        Ok(())
    })
    .await
    .map(Json)
}

async fn pause_goal(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<GoalResponse>, ApiError> {
    let response = mutate_active_goal(&state, &id, &req.goal_sha256, |session, run| {
        let paused = run.request_pause(now_millis())?;
        session.pause_token.store(true, Ordering::SeqCst);
        publish_event(
            &session.sender,
            &session.history,
            AgentEvent::GoalPauseRequested {
                goal_id: run.id.clone(),
                timestamp_ms: Some(now_millis()),
            },
        );
        if paused {
            session.status = SessionStatus::Paused;
            session.paused = true;
            publish_event(
                &session.sender,
                &session.history,
                AgentEvent::GoalPaused {
                    goal_id: run.id.clone(),
                    timestamp_ms: Some(now_millis()),
                },
            );
        }
        Ok(())
    })
    .await?;
    Ok(Json(response))
}

async fn resume_goal(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<GoalResponse>, ApiError> {
    let response = mutate_active_goal(&state, &id, &req.goal_sha256, |session, run| {
        run.resume(now_millis())?;
        session.durable.pending_goal_change = None;
        session.pause_token.store(false, Ordering::SeqCst);
        configure_goal_milestone_request(session, run)?;
        session.status = SessionStatus::Queued;
        session.paused = false;
        publish_event(
            &session.sender,
            &session.history,
            AgentEvent::GoalResumed {
                goal_id: run.id.clone(),
                timestamp_ms: Some(now_millis()),
            },
        );
        publish_current_goal_milestone(session, run);
        Ok(())
    })
    .await?;
    dispatch_next_session(state);
    Ok(Json(response))
}

async fn cancel_goal(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<GoalResponse>, ApiError> {
    let mut sessions = state.sessions.lock().await;
    let (session_id, session) = find_active_goal_session_mut(&mut sessions, &id)?;
    let checkpoint = session.goal.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    if checkpoint.sha256 != req.goal_sha256 {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "goal_revision_conflict",
            "the goal changed; refresh before retrying",
        ));
    }
    if session.running {
        session.pause_token.store(false, Ordering::SeqCst);
        session.cancel_token.store(true, Ordering::SeqCst);
        return Ok(Json(GoalResponse {
            session_id,
            goal_id: id,
            goal_sha256: checkpoint.sha256.clone(),
        }));
    }
    let mut run = session.goal.take().unwrap().run;
    run.cancel(now_millis());
    session.durable.pending_goal_change = None;
    let checkpoint = crate::goal::GoalCheckpoint::new(run).map_err(internal_status)?;
    publish_event(
        &session.sender,
        &session.history,
        AgentEvent::GoalCancelled {
            goal_id: checkpoint.run.id.clone(),
            checkpoint_sha256: checkpoint.sha256.clone(),
            timestamp_ms: Some(now_millis()),
        },
    );
    session.completed_goals.push(checkpoint.clone());
    session.status = fold_terminal_goal_task(session, &checkpoint).map_err(internal_status)?;
    session.paused = false;
    session.updated_at_ms = now_millis();
    persist_live_session(&session_id, session);
    Ok(Json(GoalResponse {
        session_id,
        goal_id: id,
        goal_sha256: checkpoint.sha256,
    }))
}

async fn accept_goal(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<GoalResponse>, ApiError> {
    let mut sessions = state.sessions.lock().await;
    let (session_id, session) = find_active_goal_session_mut(&mut sessions, &id)?;
    let mut checkpoint = session.goal.take().ok_or(StatusCode::NOT_FOUND)?;
    if checkpoint.sha256 != req.goal_sha256 {
        session.goal = Some(checkpoint);
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "goal_revision_conflict",
            "the goal changed; refresh before retrying",
        ));
    }
    checkpoint
        .run
        .accept(&req.goal_sha256, &checkpoint.sha256, now_millis())
        .map_err(conflict_status)?;
    session.durable.pending_goal_change = None;
    checkpoint = crate::goal::GoalCheckpoint::new(checkpoint.run).map_err(internal_status)?;
    publish_event(
        &session.sender,
        &session.history,
        AgentEvent::GoalCompleted {
            goal_id: checkpoint.run.id.clone(),
            outcome: crate::goal::GoalOutcome::Complete,
            completion_basis: crate::goal::GoalCompletionBasis::UserAccepted,
            checkpoint_sha256: checkpoint.sha256.clone(),
            timestamp_ms: Some(now_millis()),
        },
    );
    session.completed_goals.push(checkpoint.clone());
    session.status = fold_terminal_goal_task(session, &checkpoint).map_err(internal_status)?;
    session.paused = false;
    let dispatch = session.status == SessionStatus::Queued;
    persist_live_session(&session_id, session);
    let response = Json(GoalResponse {
        session_id,
        goal_id: id,
        goal_sha256: checkpoint.sha256,
    });
    drop(sessions);
    if dispatch {
        dispatch_next_session(state);
    }
    Ok(response)
}

async fn amend_goal(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalAmendmentRequest>,
) -> Result<Json<GoalResponse>, ApiError> {
    mutate_active_goal(&state, &id, &req.goal_sha256, |session, run| {
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
        publish_event(
            &session.sender,
            &session.history,
            AgentEvent::GoalAmendmentRequested {
                goal_id: run.id.clone(),
                amendment_id,
                replacement_plan_sha256,
                timestamp_ms: Some(now_millis()),
            },
        );
        session.durable.pending_goal_change = None;
        session.status = SessionStatus::Paused;
        session.paused = true;
        Ok(())
    })
    .await
    .map(Json)
}

async fn approve_goal_amendment(
    Path((id, amendment_id)): Path<(String, String)>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<GoalResponse>, ApiError> {
    let response = mutate_active_goal(&state, &id, &req.goal_sha256, |session, run| {
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
        session.paused = false;
        publish_event(
            &session.sender,
            &session.history,
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
    dispatch_next_session(state);
    Ok(Json(response))
}

async fn discard_goal_amendment(
    Path((id, amendment_id)): Path<(String, String)>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<GoalDigestRequest>,
) -> Result<Json<GoalResponse>, ApiError> {
    mutate_active_goal(&state, &id, &req.goal_sha256, |session, run| {
        if run
            .pending_amendment
            .as_ref()
            .is_none_or(|pending| pending.id != amendment_id)
        {
            bail!("amendment id is stale");
        }
        run.discard_amendment(now_millis())?;
        session.status = SessionStatus::Paused;
        session.paused = true;
        publish_event(
            &session.sender,
            &session.history,
            AgentEvent::GoalAmendmentResolved {
                goal_id: run.id.clone(),
                amendment_id: amendment_id.clone(),
                accepted: false,
                timestamp_ms: Some(now_millis()),
            },
        );
        Ok(())
    })
    .await
    .map(Json)
}

async fn mutate_active_goal(
    state: &AppState,
    goal_id: &str,
    expected_sha256: &str,
    mutate: impl FnOnce(&mut SessionState, &mut crate::goal::GoalRun) -> Result<()>,
) -> Result<GoalResponse, ApiError> {
    let mut sessions = state.sessions.lock().await;
    let (session_id, session) = find_active_goal_session_mut(&mut sessions, goal_id)?;
    let mut checkpoint = session.goal.take().ok_or(StatusCode::NOT_FOUND)?;
    if checkpoint.sha256 != expected_sha256 {
        session.goal = Some(checkpoint);
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "goal_revision_conflict",
            "the goal changed; refresh before retrying",
        ));
    }
    if let Err(error) = mutate(session, &mut checkpoint.run) {
        session.goal = Some(checkpoint);
        tracing::warn!(%error, "goal mutation rejected");
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "goal_mutation_rejected",
            error,
        ));
    }
    checkpoint = crate::goal::GoalCheckpoint::new(checkpoint.run).map_err(internal_status)?;
    session.updated_at_ms = now_millis();
    session.goal = Some(checkpoint.clone());
    sync_multi_task_goal_checkpoint(session).map_err(internal_status)?;
    persist_live_session(&session_id, session);
    Ok(GoalResponse {
        session_id,
        goal_id: goal_id.to_string(),
        goal_sha256: checkpoint.sha256,
    })
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

fn find_active_goal_session_mut<'a>(
    sessions: &'a mut HashMap<String, SessionState>,
    goal_id: &str,
) -> Result<(String, &'a mut SessionState), StatusCode> {
    sessions
        .iter_mut()
        .find(|(_, session)| {
            session
                .goal
                .as_ref()
                .is_some_and(|checkpoint| checkpoint.run.id == goal_id)
        })
        .map(|(session_id, session)| (session_id.clone(), session))
        .ok_or(StatusCode::NOT_FOUND)
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
        publish_event(
            &session.sender,
            &session.history,
            AgentEvent::GoalMilestoneStarted {
                goal_id: run.id.clone(),
                milestone_id: milestone.id.clone(),
                title: milestone.title.clone(),
                timestamp_ms: Some(now_millis()),
            },
        );
    }
}

fn conflict_status(_error: anyhow::Error) -> StatusCode {
    StatusCode::CONFLICT
}

fn internal_status(error: anyhow::Error) -> StatusCode {
    tracing::error!(%error, "goal persistence invariant failed");
    StatusCode::INTERNAL_SERVER_ERROR
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
    mutate_project_registry(state, move || projects::add_project(request)).await?;
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
) -> Result<Json<SessionResponse>, ApiError> {
    let task = req.task.trim().to_string();
    if task.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_task",
            "session task must not be empty",
        ));
    }
    let mut sessions = state.sessions.lock().await;
    let session = sessions.get_mut(&id).ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            "session_not_found",
            "session not found",
        )
    })?;
    if !matches!(
        session.status,
        SessionStatus::Completed | SessionStatus::Failed
    ) {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "session_not_continuable",
            "only a completed or failed session can be continued",
        ));
    }

    let mut request = session.request_template.clone();
    request.task = task;
    let workflow_policy =
        workflow_policy_for_request(session.workdir.as_deref()).map_err(|error| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "workflow_policy_failed",
                error,
            )
        })?;
    request.intent = Some(req.intent.unwrap_or(workflow_policy.default_intent));
    request.workflow_policy = Some(workflow_policy);
    request.workflow_stage = None;
    request.workflow_checkpoint = None;
    request.goal_context = None;
    request.turn_id = new_turn_id(&id);
    if req.proposal_id.is_some() && request.intent != Some(crate::workflow::TurnIntent::Deliver) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_proposal",
            "only a delivery turn can cite a delivery proposal",
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
                .ok_or_else(|| {
                    ApiError::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_proposal",
                        "delivery proposal is missing or stale",
                    )
                })?,
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
        let history = session.history.lock().map_err(|_| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "session_history_unavailable",
                "session history is unavailable",
            )
        })?;
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
    session.running = false;
    session.paused = false;
    session.status = SessionStatus::Queued;
    session.pending_question = None;
    session.workflow = None;
    if let Some(multi_task) = session.multi_task.take() {
        if session
            .completed_multi_tasks
            .last()
            .is_none_or(|existing| existing.run.id != multi_task.run.id)
        {
            session.completed_multi_tasks.push(multi_task);
        }
    }
    session.cancel_token.store(false, Ordering::SeqCst);
    session.updated_at_ms = now_millis();
    publish_session_state_changed(session);

    let history = Arc::clone(&session.history);
    let usage_records = Arc::clone(&session.usage_records);
    let branch = session.branch.clone();
    let workdir = session.workdir.clone();
    let completed_workflows = session.completed_workflows.clone();
    let goal = session.goal.clone();
    let completed_goals = session.completed_goals.clone();
    let multi_task = session.multi_task.clone();
    let completed_multi_tasks = session.completed_multi_tasks.clone();
    let durable = session.durable.clone();
    drop(sessions);
    persist_session_snapshot(
        &id,
        &request,
        branch,
        workdir,
        SessionStatus::Queued,
        &history,
        &usage_records,
        None,
        completed_workflows,
        goal,
        completed_goals,
        multi_task,
        completed_multi_tasks,
        durable,
    );
    dispatch_next_session(state.clone());

    Ok(Json(SessionResponse { session_id: id }))
}

async fn send_session_message(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<SendSessionMessageRequest>,
) -> Result<Json<SendSessionMessageResponse>, ApiError> {
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

    let mut sessions = state.sessions.lock().await;
    let session = sessions.get_mut(&id).ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            "session_not_found",
            "session not found",
        )
    })?;
    if session.status != SessionStatus::Running || !session.running {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "session_not_running",
            "messages can only be sent to a running session",
        ));
    }
    let message_id = new_user_message_id(&id);
    let mut pending = session.pending_user_messages.lock().map_err(|_| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "message_queue_unavailable",
            "message queue is unavailable",
        )
    })?;
    if !session.accepting_user_messages.load(Ordering::SeqCst) {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "message_window_closed",
            "the agent is no longer accepting messages for this turn",
        ));
    }
    if pending.len() >= MAX_PENDING_USER_MESSAGES {
        return Err(ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "message_queue_full",
            "the pending message queue is full",
        ));
    }
    pending.push_back(QueuedUserMessage {
        message_id: message_id.clone(),
        message: message.clone(),
    });
    publish_event(
        &session.sender,
        &session.history,
        AgentEvent::UserMessage {
            message_id: message_id.clone(),
            message,
            timestamp_ms: Some(now_millis()),
        },
    );
    drop(pending);
    session.updated_at_ms = now_millis();
    persist_live_session(&id, session);
    Ok(Json(SendSessionMessageResponse { message_id }))
}

async fn resume_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionResponse>, ApiError> {
    resume_session_inner(state, id)
        .await
        .map(Json)
        .map_err(|err| match err.downcast_ref::<ResumeSessionError>() {
            Some(ResumeSessionError::NotFound) => {
                ApiError::new(StatusCode::NOT_FOUND, "session_not_found", err)
            }
            Some(ResumeSessionError::Conflict) | None => {
                ApiError::new(StatusCode::CONFLICT, "session_not_resumable", err)
            }
        })
}

#[derive(Debug)]
enum ResumeSessionError {
    NotFound,
    Conflict,
}

impl std::fmt::Display for ResumeSessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => f.write_str("session not found"),
            Self::Conflict => f.write_str("session is not a resumable restored queue"),
        }
    }
}

impl std::error::Error for ResumeSessionError {}

async fn resume_session_inner(state: AppState, id: String) -> Result<SessionResponse> {
    let mut sessions = state.sessions.lock().await;
    let session = sessions.get_mut(&id).ok_or(ResumeSessionError::NotFound)?;
    let blocked_workflow = session
        .workflow
        .as_ref()
        .is_some_and(|checkpoint| checkpoint.run.stage == crate::workflow::WorkflowStage::Blocked);
    let blocked_workflow_requires_restart = session.workflow.as_ref().is_some_and(|checkpoint| {
        checkpoint.run.stage == crate::workflow::WorkflowStage::Blocked
            && checkpoint.run.outcome == Some(crate::workflow::WorkflowOutcome::CommitBlocked)
    });
    let resumable_multi_task = session.multi_task.as_ref().is_some_and(|checkpoint| {
        matches!(
            checkpoint.run.stage,
            crate::task_queue::MultiTaskStage::Paused | crate::task_queue::MultiTaskStage::Blocked
        )
    });
    if (!matches!(session.status, SessionStatus::Paused)
        && !(session.status == SessionStatus::Failed && (blocked_workflow || resumable_multi_task)))
        || session.pending_question.is_some()
    {
        anyhow::bail!(ResumeSessionError::Conflict);
    }
    if blocked_workflow_requires_restart {
        anyhow::bail!(ResumeSessionError::Conflict);
    }
    let mut request = session.request_template.clone();
    if let Some(checkpoint) = session.multi_task.clone() {
        let mut run = checkpoint.run;
        let repository = crate::task_queue::TaskRepositoryState::capture(std::path::Path::new(
            &run.authority.workdir,
        ))?;
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
            publish_event(
                &session.sender,
                &session.history,
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
    session.running = false;
    session.paused = false;
    session.status = SessionStatus::Queued;
    session.updated_at_ms = now_millis();
    session.cancel_token.store(false, Ordering::SeqCst);
    session.request_template = request.clone();
    publish_session_state_changed(session);
    let branch = session.branch.clone();
    let workdir = session.workdir.clone();
    let history = Arc::clone(&session.history);
    let usage_records = Arc::clone(&session.usage_records);
    let workflow = session.workflow.clone();
    let completed_workflows = session.completed_workflows.clone();
    let goal = session.goal.clone();
    let completed_goals = session.completed_goals.clone();
    let multi_task = session.multi_task.clone();
    let completed_multi_tasks = session.completed_multi_tasks.clone();
    let durable = session.durable.clone();
    drop(sessions);
    persist_session_snapshot(
        &id,
        &request,
        branch,
        workdir,
        SessionStatus::Queued,
        &history,
        &usage_records,
        workflow,
        completed_workflows,
        goal,
        completed_goals,
        multi_task,
        completed_multi_tasks,
        durable,
    );
    dispatch_next_session(state.clone());
    Ok(SessionResponse { session_id: id })
}

async fn restart_delivery(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionResponse>, ApiError> {
    restart_delivery_inner(state, id)
        .await
        .map(Json)
        .map_err(|error| ApiError::new(StatusCode::CONFLICT, "delivery_not_restartable", error))
}

fn prepare_blocked_workflow_restart(session: &mut SessionState, session_id: &str) -> Result<()> {
    let checkpoint = session
        .workflow
        .as_ref()
        .context("session has no blocked delivery to restart")?;
    if session.status != SessionStatus::Failed
        || session.running
        || session.pending_question.is_some()
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

    session.running = false;
    session.paused = false;
    session.status = SessionStatus::Queued;
    session.updated_at_ms = now_millis();
    session.cancel_token.store(false, Ordering::SeqCst);
    session.pause_token.store(false, Ordering::SeqCst);
    publish_event(
        &session.sender,
        &session.history,
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
    publish_session_state_changed(session);
    Ok(())
}

async fn restart_delivery_inner(state: AppState, id: String) -> Result<SessionResponse> {
    let mut sessions = state.sessions.lock().await;
    let session = sessions
        .get_mut(&id)
        .with_context(|| format!("session not found: {id}"))?;
    prepare_blocked_workflow_restart(session, &id)?;
    let request = session.request_template.clone();
    let branch = session.branch.clone();
    let workdir = session.workdir.clone();
    let history = Arc::clone(&session.history);
    let usage_records = Arc::clone(&session.usage_records);
    let completed_workflows = session.completed_workflows.clone();
    let completed_goals = session.completed_goals.clone();
    let completed_multi_tasks = session.completed_multi_tasks.clone();
    let durable = session.durable.clone();
    drop(sessions);
    persist_session_snapshot(
        &id,
        &request,
        branch,
        workdir,
        SessionStatus::Queued,
        &history,
        &usage_records,
        None,
        completed_workflows,
        None,
        completed_goals,
        None,
        completed_multi_tasks,
        durable,
    );
    dispatch_next_session(state);
    Ok(SessionResponse { session_id: id })
}

async fn retry_task_planning(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionResponse>, ApiError> {
    recover_task_plan(state, id, crate::agent_core::TaskPlanningPreference::Auto)
        .await
        .map(Json)
        .map_err(|error| ApiError::new(StatusCode::CONFLICT, "task_plan_not_recoverable", error))
}

async fn run_as_one_build(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionResponse>, ApiError> {
    recover_task_plan(
        state,
        id,
        crate::agent_core::TaskPlanningPreference::OneBuild,
    )
    .await
    .map(Json)
    .map_err(|error| ApiError::new(StatusCode::CONFLICT, "task_plan_not_recoverable", error))
}

async fn recover_task_plan(
    state: AppState,
    id: String,
    preference: crate::agent_core::TaskPlanningPreference,
) -> Result<SessionResponse> {
    let mut sessions = state.sessions.lock().await;
    let session = sessions
        .get_mut(&id)
        .with_context(|| format!("session not found: {id}"))?;
    if session.status != SessionStatus::Failed
        || session.request_template.task_plan_rejected.is_none()
        || session.running
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
    session.request_template.turn_id = new_turn_id(&id);
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
    session.running = false;
    session.paused = false;
    session.status = SessionStatus::Queued;
    session.cancel_token.store(false, Ordering::SeqCst);
    session.updated_at_ms = now_millis();
    publish_event(
        &session.sender,
        &session.history,
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
    publish_session_state_changed(session);
    persist_live_session(&id, session);
    drop(sessions);
    dispatch_next_session(state);
    Ok(SessionResponse { session_id: id })
}

async fn cancel_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionResponse>, ApiError> {
    cancel_session_inner(state, id)
        .await
        .map(Json)
        .map_err(|error| ApiError::new(StatusCode::CONFLICT, "session_not_cancellable", error))
}

async fn cancel_session_inner(state: AppState, id: String) -> Result<SessionResponse> {
    let mut sessions = state.sessions.lock().await;
    let session = sessions
        .get_mut(&id)
        .with_context(|| format!("session not found: {id}"))?;
    let blocked_multi_task = session.multi_task.as_ref().is_some_and(|checkpoint| {
        checkpoint.run.stage == crate::task_queue::MultiTaskStage::Blocked
    });
    if matches!(session.status, SessionStatus::Completed)
        || (session.status == SessionStatus::Failed
            && !blocked_multi_task
            && session.workflow.as_ref().is_none_or(|checkpoint| {
                checkpoint.run.stage != crate::workflow::WorkflowStage::Blocked
            }))
    {
        bail!("session is not cancellable");
    }
    session.cancel_token.store(true, Ordering::SeqCst);
    session.pending_question.take();
    publish_event(
        &session.sender,
        &session.history,
        AgentEvent::Correction {
            kind: crate::events::CorrectionKind::Lifecycle,
            message: "Cancellation requested. Repository content and workflow evidence will be preserved."
                .to_string(),
            summary: "Cancellation requested".to_string(),
            actor: crate::events::TeamActor::workflow_steward(),
            assisting_profile: Some(session.request_template.profile),
            nesting_depth: None,
            timestamp_ms: Some(now_millis()),
        },
    );

    let terminate_environment = !session.running;
    if terminate_environment {
        if let Some(checkpoint) = session.multi_task.take() {
            let mut run = checkpoint.run;
            if !run.stage.is_terminal() {
                let now = now_millis().max(run.updated_at_ms);
                run.apply(crate::task_queue::MultiTaskEvent::Cancelled {
                    reason: "cancelled by user; completed Task commits were preserved".to_string(),
                    now_ms: now,
                })?;
            }
            archive_multi_task(session, crate::task_queue::MultiTaskCheckpoint::new(run)?);
        }
        if let Some(checkpoint) = session.workflow.take() {
            let mut run = checkpoint.run;
            if run.stage == crate::workflow::WorkflowStage::Blocked {
                run.apply(crate::workflow::WorkflowEvent::Resumed)?;
            }
            if !run.stage.is_terminal() {
                run.apply(crate::workflow::WorkflowEvent::Cancelled {
                    reason: "cancelled by user; repository content and evidence were preserved"
                        .to_string(),
                })?;
            }
            let checkpoint = crate::workflow::WorkflowCheckpoint::new(run)?;
            let summary = crate::workflow::WorkflowSummary::from(&checkpoint.run);
            if session
                .completed_workflows
                .last()
                .is_none_or(|existing| existing.id != summary.id)
            {
                session.completed_workflows.push(summary);
            }
            publish_event(
                &session.sender,
                &session.history,
                AgentEvent::WorkflowCompleted {
                    workflow_id: checkpoint.run.id,
                    outcome: crate::workflow::WorkflowOutcome::Cancelled,
                    checkpoint_sha256: checkpoint.sha256,
                    ready_evidence_sha256: None,
                    timestamp_ms: Some(now_millis()),
                },
            );
        }
        session.request_template.workflow_checkpoint = None;
        session.running = false;
        session.paused = false;
        session.status = SessionStatus::Completed;
        publish_session_state_changed(session);
    }
    session.updated_at_ms = now_millis();
    let request = session.request_template.clone();
    let branch = session.branch.clone();
    let workdir = session.workdir.clone();
    let history = Arc::clone(&session.history);
    let usage_records = Arc::clone(&session.usage_records);
    let workflow = session.workflow.clone();
    let completed_workflows = session.completed_workflows.clone();
    let goal = session.goal.clone();
    let completed_goals = session.completed_goals.clone();
    let multi_task = session.multi_task.clone();
    let completed_multi_tasks = session.completed_multi_tasks.clone();
    let durable = session.durable.clone();
    let status = session.status;
    drop(sessions);
    persist_session_snapshot(
        &id,
        &request,
        branch,
        workdir,
        status,
        &history,
        &usage_records,
        workflow,
        completed_workflows,
        goal,
        completed_goals,
        multi_task,
        completed_multi_tasks,
        durable,
    );
    if terminate_environment {
        crate::session_environment::terminate_global_session(&id)?;
    }
    Ok(SessionResponse { session_id: id })
}

async fn answer_question(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<AnswerQuestionRequest>,
) -> Result<Json<SessionResponse>, ApiError> {
    answer_question_inner(state, id, req)
        .await
        .map(Json)
        .map_err(|err| match err.downcast_ref::<AnswerQuestionError>() {
            Some(AnswerQuestionError::NotFound) => {
                ApiError::new(StatusCode::NOT_FOUND, "session_not_found", err)
            }
            Some(AnswerQuestionError::Gone) => {
                ApiError::new(StatusCode::GONE, "question_closed", err)
            }
            Some(AnswerQuestionError::Conflict) | None => {
                ApiError::new(StatusCode::CONFLICT, "question_mismatch", err)
            }
        })
}

#[derive(Debug)]
enum AnswerQuestionError {
    NotFound,
    Conflict,
    Gone,
}

impl std::fmt::Display for AnswerQuestionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => f.write_str("session not found"),
            Self::Conflict => f.write_str(
                "session does not have a pending question matching the requested question id",
            ),
            Self::Gone => f.write_str("session stopped before the answer could be delivered"),
        }
    }
}

impl std::error::Error for AnswerQuestionError {}

async fn answer_question_inner(
    state: AppState,
    id: String,
    req: AnswerQuestionRequest,
) -> Result<SessionResponse> {
    {
        let mut sessions = state.sessions.lock().await;
        let session = sessions.get_mut(&id).ok_or(AnswerQuestionError::NotFound)?;
        let Some(pending) = session.pending_question.take() else {
            anyhow::bail!(AnswerQuestionError::Conflict);
        };
        if pending.question_id != req.question_id {
            session.pending_question = Some(pending);
            anyhow::bail!(AnswerQuestionError::Conflict);
        }

        let answer = req.answer.trim().to_string();
        if answer.is_empty() || (!pending.choices.is_empty() && !pending.choices.contains(&answer))
        {
            session.pending_question = Some(pending);
            anyhow::bail!(AnswerQuestionError::Conflict);
        }

        let sender = session.sender.clone();
        let history = Arc::clone(&session.history);
        let usage_records = Arc::clone(&session.usage_records);
        let request_template = session.request_template.clone();
        let branch = session.branch.clone();
        let workdir = session.workdir.clone();
        let workflow = session.workflow.clone();
        let completed_workflows = session.completed_workflows.clone();
        let goal = session.goal.clone();
        let completed_goals = session.completed_goals.clone();
        let multi_task = session.multi_task.clone();
        let completed_multi_tasks = session.completed_multi_tasks.clone();
        let durable = session.durable.clone();
        let question_id = req.question_id.clone();
        pending
            .responder
            .send(answer.clone())
            .map_err(|_| AnswerQuestionError::Gone)?;
        session.paused = false;
        session.running = true;
        session.status = SessionStatus::Running;
        session
            .accepting_user_messages
            .store(true, Ordering::SeqCst);
        session.updated_at_ms = now_millis();
        publish_event(
            &sender,
            &history,
            AgentEvent::UserAnswer {
                question_id,
                answer,
                timestamp_ms: Some(now_millis()),
            },
        );
        persist_session_snapshot(
            &id,
            &request_template,
            branch,
            workdir,
            SessionStatus::Running,
            &history,
            &usage_records,
            workflow,
            completed_workflows,
            goal,
            completed_goals,
            multi_task,
            completed_multi_tasks,
            durable,
        );
    }
    state.update_sleep_prevention_working(true);
    Ok(SessionResponse { session_id: id })
}

fn session_history_summary(
    session: &SessionState,
) -> (Option<String>, Option<HandoffOutcome>, u64) {
    let Ok(history) = session.history.lock() else {
        return (session.title.clone(), None, 0);
    };
    (
        latest_session_title(&history).or_else(|| session.title.clone()),
        handoff_outcome_from_history(&history),
        history
            .last()
            .map(|envelope| envelope.transcript.sequence)
            .unwrap_or(0),
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
) -> Result<Json<DeleteSessionResponse>, ApiError> {
    delete_session_inner(state, &id)
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
) -> Result<DeleteSessionResponse, DeleteSessionError> {
    // Keep every authoritative phase in one owned task. Dropping a disconnected HTTP or RPC
    // response future then detaches this task instead of cancelling the transaction between awaits.
    tokio::spawn(delete_session_transaction(state, id.to_string()))
        .await
        .map_err(|error| anyhow::anyhow!("session deletion transaction failed: {error}"))?
}

async fn delete_session_transaction(
    state: AppState,
    id: String,
) -> Result<DeleteSessionResponse, DeleteSessionError> {
    {
        let mut sessions = state.sessions.lock().await;
        let Some(session) = sessions.get(&id) else {
            return Err(DeleteSessionError::NotFound(format!(
                "session not found: {id}"
            )));
        };
        if session.status == SessionStatus::Running
            || session.status == SessionStatus::Queued
            || session.pending_question.is_some()
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
            tokio::task::spawn_blocking(move || {
                session_store::delete_session(&workdir, &session_id)
            })
            .await
            .map_err(|error| anyhow::anyhow!("session persistence deletion task failed: {error}"))?
            .map_err(anyhow::Error::from)?;
        }
        sessions.remove(&id).expect("session exists");
        state.project_usage_windows.lock().await.invalidate();
        state.publish_project_session_change();
    }
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

    Ok(DeleteSessionResponse {
        session_id: id,
        deleted: true,
        cleanup_warnings,
    })
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
    mutate_project_registry(&state, move || {
        projects::set_project_notifications_by_id(&id, notify_on_finish)
    })
    .await
    .map_err(|error| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "project_notification_update_failed",
            error,
        )
    })?;
    let transition_floor = state.project_session_revision_baseline();
    Ok(Json(
        project_session_snapshot(&state, transition_floor, usage_window).await,
    ))
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
    sender: broadcast::Sender<EventEnvelope>,
    history: Arc<StdMutex<Vec<EventEnvelope>>>,
    usage_records: Arc<StdMutex<Vec<SessionMetricsSnapshot>>>,
    persisted_branch: Option<String>,
    persisted_workdir: Option<PathBuf>,
    workflow: Option<crate::workflow::WorkflowCheckpoint>,
    completed_workflows: Vec<crate::workflow::WorkflowSummary>,
    goal_deadline_ms: Option<u64>,
    task_deadline_ms: Option<u64>,
    goal: Option<crate::goal::GoalCheckpoint>,
    completed_goals: Vec<crate::goal::GoalCheckpoint>,
    multi_task: Option<crate::task_queue::MultiTaskCheckpoint>,
    completed_multi_tasks: Vec<crate::task_queue::MultiTaskCheckpoint>,
    durable: DurableSessionProjection,
    pending_user_messages: Arc<StdMutex<VecDeque<QueuedUserMessage>>>,
    accepting_user_messages: Arc<AtomicBool>,
    pause_token: Arc<AtomicBool>,
    cancel_token: Arc<AtomicBool>,
    terminal_precursor_keys: Vec<String>,
}

impl WebEventSink {
    fn emit_timestamped(&mut self, event: AgentEvent, supersedes: Vec<String>) -> String {
        let envelope = EventEnvelope::with_timestamp(event);
        let entry_key = envelope.transcript.entry_key.clone();
        if let AgentEvent::Started {
            workspace,
            focus_root,
            branch,
            ..
        } = &envelope.event
        {
            self.persisted_workdir =
                Some(PathBuf::from(focus_root.as_deref().unwrap_or(workspace)));
            self.persisted_branch = Some(branch.clone());
        }
        if let Some(metrics) = SessionMetricsSnapshot::from_event(&envelope.event) {
            tokio::runtime::Handle::current().block_on(async {
                let mut sessions = self.state.sessions.lock().await;
                let Some(session) = sessions.get_mut(&self.session_id) else {
                    return;
                };
                self.usage_records
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(metrics.clone());
                if let Some(existing) = session.metrics.as_mut() {
                    existing.add_assign(&metrics);
                } else {
                    session.metrics = Some(metrics.clone());
                }
                session.updated_at_ms = now_millis();
                let project_id = session
                    .durable
                    .project
                    .as_ref()
                    .map(|project| project.id.as_str());
                self.state
                    .project_usage_windows
                    .lock()
                    .await
                    .record_metrics(project_id, &metrics);
            });
        }
        if let AgentEvent::SessionTitle { title, .. } = &envelope.event {
            let title = title.trim().to_string();
            if !title.is_empty() {
                tokio::runtime::Handle::current().block_on(async {
                    let mut sessions = self.state.sessions.lock().await;
                    if let Some(session) = sessions.get_mut(&self.session_id) {
                        session.title = Some(title);
                        session.updated_at_ms = now_millis();
                    }
                });
            }
        }
        if let AgentEvent::DeliveryProposed {
            proposal_id,
            source_turn_id,
            task_summary,
            ..
        } = &envelope.event
        {
            let proposal = crate::workflow::DeliveryProposal {
                id: proposal_id.clone(),
                source_turn_id: source_turn_id.clone(),
                task_summary: task_summary.clone(),
            };
            self.durable.pending_delivery_proposal = Some(proposal.clone());
            tokio::runtime::Handle::current().block_on(async {
                let mut sessions = self.state.sessions.lock().await;
                if let Some(session) = sessions.get_mut(&self.session_id) {
                    session.durable.pending_delivery_proposal = Some(proposal);
                    session.updated_at_ms = now_millis();
                }
            });
        }
        if let AgentEvent::GoalProposed {
            proposal_id,
            source_turn_id,
            objective,
            criteria,
            ..
        } = &envelope.event
        {
            let proposal = crate::goal::GoalProposal {
                id: proposal_id.clone(),
                source_turn_id: source_turn_id.clone(),
                objective: objective.clone(),
                criteria: criteria.clone(),
            };
            self.durable.pending_goal_proposal = Some(proposal.clone());
            tokio::runtime::Handle::current().block_on(async {
                let mut sessions = self.state.sessions.lock().await;
                if let Some(session) = sessions.get_mut(&self.session_id) {
                    session.durable.pending_goal_proposal = Some(proposal);
                    session.updated_at_ms = now_millis();
                }
            });
        }
        publish_event_envelope_linked(&self.sender, &self.history, envelope, supersedes);
        if self.goal.is_some() {
            self.goal = tokio::runtime::Handle::current().block_on(async {
                let sessions = self.state.sessions.lock().await;
                sessions
                    .get(&self.session_id)
                    .and_then(|session| session.goal.clone())
            });
        }
        if self.multi_task.is_some() {
            self.multi_task = tokio::runtime::Handle::current().block_on(async {
                let sessions = self.state.sessions.lock().await;
                sessions
                    .get(&self.session_id)
                    .and_then(|session| session.multi_task.clone())
            });
        }
        persist_session_snapshot(
            &self.session_id,
            &self.request_template,
            self.persisted_branch.clone(),
            self.persisted_workdir.clone(),
            SessionStatus::Running,
            &self.history,
            &self.usage_records,
            self.workflow.clone(),
            self.completed_workflows.clone(),
            self.goal.clone(),
            self.completed_goals.clone(),
            self.multi_task.clone(),
            self.completed_multi_tasks.clone(),
            self.durable.clone(),
        );
        entry_key
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
        self.terminal_precursor_keys.push(entry_key);
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
        let mut pending = match self.pending_user_messages.lock() {
            Ok(pending) => pending,
            Err(_) => return Vec::new(),
        };
        let messages = pending.drain(..).collect::<Vec<_>>();
        if messages.is_empty() {
            return messages;
        }
        for message in &messages {
            publish_event(
                &self.sender,
                &self.history,
                AgentEvent::UserMessageApplied {
                    message_id: message.message_id.clone(),
                    timestamp_ms: Some(now_millis()),
                },
            );
        }
        drop(pending);
        persist_session_snapshot(
            &self.session_id,
            &self.request_template,
            self.persisted_branch.clone(),
            self.persisted_workdir.clone(),
            SessionStatus::Running,
            &self.history,
            &self.usage_records,
            self.workflow.clone(),
            self.completed_workflows.clone(),
            self.goal.clone(),
            self.completed_goals.clone(),
            self.multi_task.clone(),
            self.completed_multi_tasks.clone(),
            self.durable.clone(),
        );
        messages
    }

    fn seal_user_messages(&mut self) -> bool {
        let pending = match self.pending_user_messages.lock() {
            Ok(pending) => pending,
            Err(_) => return false,
        };
        if !pending.is_empty() {
            return false;
        }
        self.accepting_user_messages.store(false, Ordering::SeqCst);
        true
    }

    fn open_user_messages(&mut self) {
        self.accepting_user_messages.store(true, Ordering::SeqCst);
    }

    fn checkpoint_workflow(
        &mut self,
        checkpoint: &crate::workflow::WorkflowCheckpoint,
    ) -> Result<()> {
        checkpoint.validate()?;
        if self.multi_task.is_some() && self.goal.is_some() {
            let (latest_parent, latest_goal) = tokio::runtime::Handle::current().block_on(async {
                let sessions = self.state.sessions.lock().await;
                let session = sessions.get(&self.session_id);
                (
                    session.and_then(|session| session.multi_task.clone()),
                    session.and_then(|session| session.goal.clone()),
                )
            });
            let mut goal = latest_goal
                .or_else(|| self.goal.take())
                .context("Goal Task workflow lost its Goal checkpoint")?
                .run;
            goal.checkpoint_active_workflow(checkpoint.clone(), now_millis())?;
            let goal = crate::goal::GoalCheckpoint::new(goal)?;
            let mut parent = latest_parent
                .or_else(|| self.multi_task.take())
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
            let parent = crate::task_queue::MultiTaskCheckpoint::new(parent)?;
            self.goal = Some(goal.clone());
            self.multi_task = Some(parent.clone());
            self.workflow = None;
            tokio::runtime::Handle::current().block_on(async {
                let mut sessions = self.state.sessions.lock().await;
                let session = sessions
                    .get_mut(&self.session_id)
                    .with_context(|| format!("session not found: {}", self.session_id))?;
                session.goal = Some(goal);
                session.multi_task = Some(parent);
                session.workflow = None;
                session.request_template.workflow_checkpoint = None;
                session.updated_at_ms = now_millis();
                Ok::<(), anyhow::Error>(())
            })?;
            persist_session_snapshot(
                &self.session_id,
                &self.request_template,
                self.persisted_branch.clone(),
                self.persisted_workdir.clone(),
                SessionStatus::Running,
                &self.history,
                &self.usage_records,
                None,
                self.completed_workflows.clone(),
                self.goal.clone(),
                self.completed_goals.clone(),
                self.multi_task.clone(),
                self.completed_multi_tasks.clone(),
                self.durable.clone(),
            );
            return Ok(());
        }
        if let Some(parent) = self.multi_task.take() {
            let latest = tokio::runtime::Handle::current().block_on(async {
                let sessions = self.state.sessions.lock().await;
                sessions
                    .get(&self.session_id)
                    .and_then(|session| session.multi_task.clone())
            });
            let mut run = latest.unwrap_or(parent).run;
            let task_id = run
                .active_task_id
                .clone()
                .context("multi-Task workflow checkpoint has no active Task")?;
            let repository = crate::task_queue::TaskRepositoryState::capture(
                &checkpoint.run.repository.repo_root,
            )?;
            let child = crate::task_queue::TaskChildCheckpoint::Build(checkpoint.clone());
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
            let parent = crate::task_queue::MultiTaskCheckpoint::new(run)?;
            self.multi_task = Some(parent.clone());
            self.workflow = None;
            tokio::runtime::Handle::current().block_on(async {
                let mut sessions = self.state.sessions.lock().await;
                let session = sessions
                    .get_mut(&self.session_id)
                    .with_context(|| format!("session not found: {}", self.session_id))?;
                session.multi_task = Some(parent);
                session.workflow = None;
                session.request_template.workflow_checkpoint = None;
                session.updated_at_ms = now_millis();
                Ok::<(), anyhow::Error>(())
            })?;
            persist_session_snapshot(
                &self.session_id,
                &self.request_template,
                self.persisted_branch.clone(),
                self.persisted_workdir.clone(),
                SessionStatus::Running,
                &self.history,
                &self.usage_records,
                None,
                self.completed_workflows.clone(),
                self.goal.clone(),
                self.completed_goals.clone(),
                self.multi_task.clone(),
                self.completed_multi_tasks.clone(),
                self.durable.clone(),
            );
            return Ok(());
        }
        if let Some(goal) = self.goal.take() {
            let latest = tokio::runtime::Handle::current().block_on(async {
                let sessions = self.state.sessions.lock().await;
                sessions
                    .get(&self.session_id)
                    .and_then(|session| session.goal.clone())
            });
            let mut run = latest.unwrap_or(goal).run;
            run.checkpoint_active_workflow(checkpoint.clone(), now_millis())?;
            self.goal = Some(crate::goal::GoalCheckpoint::new(run)?);
            self.workflow = None;
        } else {
            let summary = crate::workflow::WorkflowSummary::from(&checkpoint.run);
            if checkpoint.run.stage.is_terminal()
                && checkpoint.run.stage != crate::workflow::WorkflowStage::Blocked
            {
                if self
                    .completed_workflows
                    .last()
                    .is_none_or(|existing| existing.id != summary.id)
                {
                    self.completed_workflows.push(summary);
                }
                self.workflow = None;
            } else {
                self.workflow = Some(checkpoint.clone());
            }
        }
        tokio::runtime::Handle::current().block_on(async {
            let mut sessions = self.state.sessions.lock().await;
            let session = sessions
                .get_mut(&self.session_id)
                .with_context(|| format!("session not found: {}", self.session_id))?;
            session.workflow = self.workflow.clone();
            session.completed_workflows = self.completed_workflows.clone();
            session.goal = self.goal.clone();
            session.completed_goals = self.completed_goals.clone();
            session.request_template.workflow_checkpoint = if session.goal.is_some() {
                None
            } else {
                session.workflow.clone()
            };
            session.updated_at_ms = now_millis();
            Ok::<(), anyhow::Error>(())
        })?;
        persist_session_snapshot(
            &self.session_id,
            &self.request_template,
            self.persisted_branch.clone(),
            self.persisted_workdir.clone(),
            SessionStatus::Running,
            &self.history,
            &self.usage_records,
            self.workflow.clone(),
            self.completed_workflows.clone(),
            self.goal.clone(),
            self.completed_goals.clone(),
            self.multi_task.clone(),
            self.completed_multi_tasks.clone(),
            self.durable.clone(),
        );
        Ok(())
    }

    fn checkpoint_multi_task(
        &mut self,
        checkpoint: &crate::task_queue::MultiTaskCheckpoint,
    ) -> Result<()> {
        checkpoint.validate()?;
        self.multi_task = Some(checkpoint.clone());
        self.workflow = None;
        tokio::runtime::Handle::current().block_on(async {
            let mut sessions = self.state.sessions.lock().await;
            let session = sessions
                .get_mut(&self.session_id)
                .with_context(|| format!("session not found: {}", self.session_id))?;
            session.multi_task = Some(checkpoint.clone());
            session.workflow = None;
            session.request_template.workflow_checkpoint = None;
            session.updated_at_ms = now_millis();
            Ok::<(), anyhow::Error>(())
        })?;
        persist_session_snapshot(
            &self.session_id,
            &self.request_template,
            self.persisted_branch.clone(),
            self.persisted_workdir.clone(),
            SessionStatus::Running,
            &self.history,
            &self.usage_records,
            None,
            self.completed_workflows.clone(),
            self.goal.clone(),
            self.completed_goals.clone(),
            self.multi_task.clone(),
            self.completed_multi_tasks.clone(),
            self.durable.clone(),
        );
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
        let goal_id = tokio::runtime::Handle::current().block_on(async {
            let mut sessions = self.state.sessions.lock().await;
            let session = sessions
                .get_mut(&self.session_id)
                .with_context(|| format!("session not found: {}", self.session_id))?;
            let mut run = session
                .goal
                .as_ref()
                .context("goal pause request lost its durable goal")?
                .run
                .clone();
            let goal_id = run.id.clone();
            let paused = run.request_pause(now_millis())?;
            session.goal = Some(crate::goal::GoalCheckpoint::new(run)?);
            session.pause_token.store(true, Ordering::SeqCst);
            publish_event(
                &session.sender,
                &session.history,
                AgentEvent::GoalPauseRequested {
                    goal_id: goal_id.clone(),
                    timestamp_ms: Some(now_millis()),
                },
            );
            if paused {
                session.status = SessionStatus::Paused;
                session.paused = true;
                publish_event(
                    &session.sender,
                    &session.history,
                    AgentEvent::GoalPaused {
                        goal_id: goal_id.clone(),
                        timestamp_ms: Some(now_millis()),
                    },
                );
            }
            persist_live_session(&self.session_id, session);
            Ok::<String, anyhow::Error>(goal_id)
        })?;
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
        let pending = tokio::runtime::Handle::current().block_on(async {
            let mut sessions = self.state.sessions.lock().await;
            let session = sessions
                .get_mut(&self.session_id)
                .with_context(|| format!("session not found: {}", self.session_id))?;
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
            publish_event(
                &session.sender,
                &session.history,
                AgentEvent::GoalChangeRequested {
                    goal_id,
                    kind: kind.clone(),
                    summary: summary.clone(),
                    timestamp_ms: Some(now_millis()),
                },
            );
            Ok::<PendingGoalChange, anyhow::Error>(pending)
        })?;
        self.durable.pending_goal_change = Some(pending);
        self.request_goal_pause(&format!("{kind} review requested: {summary}"))?;
        Ok(format!(
            "{kind} request recorded and the goal will pause for human review"
        ))
    }

    fn ask_user(&mut self, question: &str) -> Result<String> {
        self.ask_multiple_choice(question, &[])
    }

    fn ask_multiple_choice(&mut self, question: &str, choices: &[String]) -> Result<String> {
        let question = question.trim();
        if question.is_empty() {
            anyhow::bail!("ask_user question must not be empty");
        }
        let question_id = new_durable_id("question");
        let (tx, rx) = std::sync::mpsc::channel();
        let event = AgentEvent::UserQuestion {
            question_id: question_id.clone(),
            question: question.to_string(),
            choices: choices.to_vec(),
            profile: self.request_template.profile,
            timestamp_ms: Some(now_millis()),
        };

        tokio::runtime::Handle::current().block_on(async {
            {
                let mut sessions = self.state.sessions.lock().await;
                let Some(session) = sessions.get_mut(&self.session_id) else {
                    anyhow::bail!("session not found: {}", self.session_id);
                };
                session.running = false;
                session.paused = true;
                session.status = SessionStatus::Paused;
                session.pending_question = Some(PendingQuestionState {
                    question_id: question_id.clone(),
                    question: question.to_string(),
                    choices: choices.to_vec(),
                    responder: tx,
                });
                session.updated_at_ms = now_millis();
                publish_event(&session.sender, &session.history, event);
                persist_session_snapshot(
                    &self.session_id,
                    &session.request_template,
                    session.branch.clone(),
                    session.workdir.clone(),
                    SessionStatus::Paused,
                    &session.history,
                    &session.usage_records,
                    session.workflow.clone(),
                    session.completed_workflows.clone(),
                    session.goal.clone(),
                    session.completed_goals.clone(),
                    session.multi_task.clone(),
                    session.completed_multi_tasks.clone(),
                    session.durable.clone(),
                );
            }
            self.state.update_sleep_prevention_working(false);
            Ok::<(), anyhow::Error>(())
        })?;

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

fn publish_session_state_changed(session: &SessionState) {
    let status = match session.status {
        SessionStatus::Queued => crate::events::SessionLifecycleStatus::Queued,
        SessionStatus::Running => crate::events::SessionLifecycleStatus::Running,
        SessionStatus::Paused => crate::events::SessionLifecycleStatus::Paused,
        SessionStatus::Completed => crate::events::SessionLifecycleStatus::Completed,
        SessionStatus::Failed => crate::events::SessionLifecycleStatus::Failed,
    };
    publish_event(
        &session.sender,
        &session.history,
        AgentEvent::SessionStateChanged {
            status,
            running: session.running,
            paused: session.paused,
            timestamp_ms: Some(now_millis()),
        },
    );
}

fn dispatch_next_session(state: AppState) {
    tokio::spawn(async move {
        let (next, working) = {
            let mut sessions = state.sessions.lock().await;
            let has_active = sessions.values().any(|session| {
                session.status == SessionStatus::Running || session.pending_question.is_some()
            });
            if has_active {
                let working = sessions
                    .values()
                    .any(|session| session.status == SessionStatus::Running);
                (None, working)
            } else if let Some(session_id) = sessions
                .iter()
                .filter(|(_, session)| session.status == SessionStatus::Queued)
                .min_by_key(|(_, session)| session.updated_at_ms)
                .map(|(session_id, _)| session_id.clone())
            {
                let session = sessions
                    .get_mut(&session_id)
                    .expect("queued session selected from sessions map");
                session.running = true;
                session.paused = false;
                session.status = SessionStatus::Running;
                session
                    .accepting_user_messages
                    .store(true, Ordering::SeqCst);
                session.updated_at_ms = now_millis();
                publish_session_state_changed(session);
                (
                    Some((
                        session_id,
                        session.request_template.clone(),
                        session.branch.clone(),
                        session.workdir.clone(),
                        Arc::clone(&session.history),
                        Arc::clone(&session.usage_records),
                        session.workflow.clone(),
                        session.completed_workflows.clone(),
                        session.goal.clone(),
                        session.completed_goals.clone(),
                        session.multi_task.clone(),
                        session.completed_multi_tasks.clone(),
                        session.durable.clone(),
                    )),
                    true,
                )
            } else {
                (None, false)
            }
        };

        state.update_sleep_prevention_working(working);
        let Some(next) = next else {
            return;
        };

        let (
            session_id,
            request,
            branch,
            workdir,
            history,
            usage_records,
            workflow,
            completed_workflows,
            goal,
            completed_goals,
            multi_task,
            completed_multi_tasks,
            durable,
        ) = next;
        persist_session_snapshot(
            &session_id,
            &request,
            branch,
            workdir,
            SessionStatus::Running,
            &history,
            &usage_records,
            workflow,
            completed_workflows,
            goal,
            completed_goals,
            multi_task,
            completed_multi_tasks,
            durable.clone(),
        );
        spawn_agent_run(state, session_id, request);
    });
}

fn spawn_agent_run(state: AppState, session_id: String, request: AgentRequest) {
    tokio::spawn(async move {
        let (
            models_root,
            sender,
            history,
            usage_records,
            workflow,
            completed_workflows,
            goal,
            completed_goals,
            multi_task,
            completed_multi_tasks,
            durable,
            pending_user_messages,
            accepting_user_messages,
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
                session.sender.clone(),
                Arc::clone(&session.history),
                Arc::clone(&session.usage_records),
                session.workflow.clone(),
                session.completed_workflows.clone(),
                session.goal.clone(),
                session.completed_goals.clone(),
                session.multi_task.clone(),
                session.completed_multi_tasks.clone(),
                session.durable.clone(),
                Arc::clone(&session.pending_user_messages),
                Arc::clone(&session.accepting_user_messages),
                Arc::clone(&session.pause_token),
                Arc::clone(&session.cancel_token),
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
                sender,
                history,
                usage_records,
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
                completed_multi_tasks,
                durable,
                pending_user_messages,
                accepting_user_messages,
                pause_token,
                cancel_token,
                terminal_precursor_keys: Vec::new(),
            };
            run_agent_managed(request_for_run.clone(), &models_root, sink)
        })
        .await;

        let goal_projects = if matches!(
            &result,
            Ok(Ok(run_result)) if run_result.requested_goal.is_some()
        ) {
            Some(state.projects.lock().await.clone())
        } else {
            None
        };
        let mut terminate_environment = false;
        let mut sessions = state.sessions.lock().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.running = false;
            session.paused = false;
            session.pending_question = None;
            session.updated_at_ms = now_millis();
            let mut final_status = SessionStatus::Completed;
            match result {
                Ok(Ok(run_result)) => {
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
                    session.request_template.workspace_graph = run_result.workspace_graph.clone();
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
                                publish_event(
                                    &session.sender,
                                    &session.history,
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
                                publish_event(
                                    &session.sender,
                                    &session.history,
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
                            let summary = crate::workflow::WorkflowSummary::from(&checkpoint.run);
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
                        match activate_single_task_goal(session, &session_id, task_goal) {
                            Ok(()) => {
                                final_status = SessionStatus::Paused;
                                session.paused = true;
                            }
                            Err(error) => {
                                final_status = SessionStatus::Failed;
                                publish_event(
                                    &session.sender,
                                    &session.history,
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
                            &session_id,
                            proposal,
                            goal_projects.as_deref().unwrap_or(&[]),
                        ) {
                            Ok(()) => {
                                final_status = SessionStatus::Paused;
                                session.paused = true;
                            }
                            Err(error) => {
                                final_status = SessionStatus::Failed;
                                publish_event(
                                    &session.sender,
                                    &session.history,
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
                Ok(Err(err)) => {
                    final_status = SessionStatus::Failed;
                    fail_active_goal_engine(session, format!("milestone runner failed: {err:#}"));
                    publish_event(
                        &session.sender,
                        &session.history,
                        AgentEvent::Error {
                            summary: "Session failed".to_string(),
                            detail: format!("{err:#}"),
                            nesting_depth: None,
                            timestamp_ms: Some(now_millis()),
                        },
                    );
                }
                Err(err) => {
                    final_status = SessionStatus::Failed;
                    fail_active_goal_engine(
                        session,
                        format!("milestone runner task failed: {err:#}"),
                    );
                    publish_event(
                        &session.sender,
                        &session.history,
                        AgentEvent::Error {
                            summary: "Session failed".to_string(),
                            detail: format!("{err:#}"),
                            nesting_depth: None,
                            timestamp_ms: Some(now_millis()),
                        },
                    );
                }
            }
            session.cancel_token.store(false, Ordering::SeqCst);
            session.pause_token.store(false, Ordering::SeqCst);
            session.status = final_status;
            publish_session_state_changed(session);
            persist_session_snapshot(
                &session_id,
                &session.request_template,
                session.branch.clone(),
                session.workdir.clone(),
                final_status,
                &session.history,
                &session.usage_records,
                session.workflow.clone(),
                session.completed_workflows.clone(),
                session.goal.clone(),
                session.completed_goals.clone(),
                session.multi_task.clone(),
                session.completed_multi_tasks.clone(),
                session.durable.clone(),
            );
        }
        drop(sessions);
        if terminate_environment
            && let Err(error) = crate::session_environment::terminate_global_session(&session_id)
        {
            eprintln!("failed to terminate cancelled session environment {session_id}: {error:#}");
        }
        dispatch_next_session(state.clone());
    });
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
    session.pause_token.store(false, Ordering::SeqCst);
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
    session.pause_token.store(false, Ordering::SeqCst);
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
            session.paused = true;
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
    publish_event(
        &session.sender,
        &session.history,
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
    if session.cancel_token.load(Ordering::SeqCst)
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
            session.paused = true;
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
    session.pause_token.store(false, Ordering::SeqCst);
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
    let cancel_requested = session.cancel_token.load(Ordering::SeqCst)
        || run_result.termination_reason == crate::events::TerminationReason::Cancelled;
    if cancel_requested {
        run.cancel(now);
        let checkpoint = crate::goal::GoalCheckpoint::new(run)?;
        publish_event(
            &session.sender,
            &session.history,
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
        publish_event(
            &session.sender,
            &session.history,
            AgentEvent::GoalPaused {
                goal_id: checkpoint.run.id.clone(),
                timestamp_ms: Some(now),
            },
        );
        session.goal = Some(checkpoint);
        session.request_template.workflow_checkpoint = None;
        session.paused = true;
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
        session.paused = true;
        return Ok((SessionStatus::Paused, false));
    }

    let milestone_id = run
        .active_milestone_id
        .clone()
        .context("goal workflow completed without an active milestone")?;
    let workflow_id = workflow.run.id.clone();
    run.finish_active_workflow(workflow, now)?;
    publish_event(
        &session.sender,
        &session.history,
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
            session.paused = false;
            Ok((SessionStatus::Queued, false))
        }
        crate::goal::GoalStage::Paused => {
            publish_event(
                &session.sender,
                &session.history,
                AgentEvent::GoalPaused {
                    goal_id: checkpoint.run.id.clone(),
                    timestamp_ms: Some(now),
                },
            );
            session.goal = Some(checkpoint);
            session.paused = true;
            session.request_template.workflow_checkpoint = None;
            Ok((SessionStatus::Paused, false))
        }
        crate::goal::GoalStage::AwaitingUserReview => {
            publish_event(
                &session.sender,
                &session.history,
                AgentEvent::GoalReadyForReview {
                    goal_id: checkpoint.run.id.clone(),
                    checkpoint_sha256: checkpoint.sha256.clone(),
                    timestamp_ms: Some(now),
                },
            );
            session.goal = Some(checkpoint);
            session.paused = true;
            session.request_template.workflow_checkpoint = None;
            Ok((SessionStatus::Paused, false))
        }
        crate::goal::GoalStage::Completed => {
            let basis = checkpoint
                .run
                .completion_basis
                .unwrap_or(crate::goal::GoalCompletionBasis::MachineVerified);
            publish_event(
                &session.sender,
                &session.history,
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
    publish_event(
        &session.sender,
        &session.history,
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
        let receiver = session.sender.subscribe();
        let history = session.history.lock().map_err(|_| {
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
    let live_state = state.clone();
    let live_session_id = id.clone();
    let live = BroadcastStream::new(receiver)
        .take_while(|message| futures::future::ready(message.is_ok()))
        .filter_map(move |message| {
            let covered_keys = Arc::clone(&live_covered_keys);
            let state = live_state.clone();
            let session_id = live_session_id.clone();
            async move {
                let envelope = message.ok()?;
                let entry_key = envelope.transcript.entry_key.clone();
                let was_covered = covered_keys
                    .lock()
                    .map(|mut keys| keys.remove(&entry_key))
                    .unwrap_or(false);
                if was_covered {
                    return None;
                }
                let refresh = envelope.requires_session_snapshot();
                let revision = envelope.transcript.sequence;
                let mut events = Vec::new();
                if let Some(event) = session_event_sse_event(envelope) {
                    events.push(Ok(event));
                }
                if refresh
                    && let Some(session) = session_details_snapshot(&state, &session_id).await
                    && session.revision >= revision
                {
                    if let Ok(mut keys) = covered_keys.lock() {
                        keys.extend(
                            session
                                .events
                                .iter()
                                .filter(|event| event.transcript.sequence > revision)
                                .map(|event| event.transcript.entry_key.clone()),
                        );
                    }
                    if let Some(event) = session_snapshot_sse_event(&SessionStreamSnapshot {
                        session,
                        reset_history: false,
                    }) {
                        events.push(Ok(event));
                    }
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
    match request.method.as_str() {
        "pb.session.start" => {
            let params: StartSessionRequest = serde_json::from_value(request.params)?;
            let result = start_session_inner(state, defaults, params).await?;
            write_rpc_response(reader.get_mut(), request.id, result).await?;
        }
        "pb.goal.start" => {
            let params: StartGoalRequest = serde_json::from_value(request.params)?;
            match start_goal_inner(state, defaults, params).await {
                Ok(result) => write_rpc_response(reader.get_mut(), request.id, result).await?,
                Err(err) => write_rpc_error(reader.get_mut(), request.id, err.to_string()).await?,
            }
        }
        "pb.goal.get" => {
            let params: serde_json::Value = request.params;
            let goal_id = params
                .get("goal_id")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing goal_id"))?
                .to_string();
            match get_goal(Path(goal_id), State((state, defaults))).await {
                Ok(Json(result)) => {
                    write_rpc_response(reader.get_mut(), request.id, result).await?
                }
                Err(error) => {
                    write_rpc_error(
                        reader.get_mut(),
                        request.id,
                        format!(
                            "goal request failed with HTTP status {}: {}",
                            error.status.as_u16(),
                            error.message
                        ),
                    )
                    .await?
                }
            }
        }
        "pb.goal.pause" | "pb.goal.resume" | "pb.goal.cancel" | "pb.goal.accept" => {
            let params: GoalRpcMutationRequest = serde_json::from_value(request.params)?;
            let goal_id = params.goal_id;
            let digest = GoalDigestRequest {
                goal_sha256: params.goal_sha256,
                plan_sha256: params.plan_sha256,
            };
            let result = match request.method.as_str() {
                "pb.goal.pause" => {
                    pause_goal(Path(goal_id), State((state, defaults)), Json(digest)).await
                }
                "pb.goal.resume" => {
                    resume_goal(Path(goal_id), State((state, defaults)), Json(digest)).await
                }
                "pb.goal.cancel" => {
                    cancel_goal(Path(goal_id), State((state, defaults)), Json(digest)).await
                }
                "pb.goal.accept" => {
                    accept_goal(Path(goal_id), State((state, defaults)), Json(digest)).await
                }
                _ => unreachable!(),
            };
            match result {
                Ok(Json(result)) => {
                    write_rpc_response(reader.get_mut(), request.id, result).await?
                }
                Err(status) => {
                    write_rpc_error(
                        reader.get_mut(),
                        request.id,
                        format!(
                            "goal request failed with HTTP status {}: {}",
                            status.status.as_u16(),
                            status.message
                        ),
                    )
                    .await?
                }
            }
        }
        "pb.session.list" => {
            let result = session_list_snapshot(&state).await;
            write_rpc_response(reader.get_mut(), request.id, result).await?;
        }
        "pb.projects.add" => {
            let params: AddProjectRequest = serde_json::from_value(request.params)?;
            let result =
                mutate_project_registry(&state, move || projects::add_project(params)).await?;
            write_rpc_response(reader.get_mut(), request.id, result).await?;
        }
        "pb.projects.list" => {
            reload_projects(&state).await?;
            let result = project_list_snapshot(&state).await;
            write_rpc_response(reader.get_mut(), request.id, result).await?;
        }
        "pb.projects.rm" => {
            let params: RemoveProjectRequest = serde_json::from_value(request.params)?;
            let result =
                mutate_project_registry(&state, move || projects::remove_project(&params.name))
                    .await?;
            write_rpc_response(reader.get_mut(), request.id, result).await?;
        }
        "pb.projects.notifications" => {
            let params: serde_json::Value = request.params;
            let name = params
                .get("name")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing project name"))?
                .to_string();
            let notify_on_finish = params
                .get("notify_on_finish")
                .and_then(|value| value.as_bool())
                .ok_or_else(|| anyhow::anyhow!("missing notify_on_finish"))?;
            let result = mutate_project_registry(&state, move || {
                projects::set_project_notifications(&name, notify_on_finish)
            })
            .await?;
            write_rpc_response(reader.get_mut(), request.id, result).await?;
        }
        "pb.session.resume" => {
            let params: WatchSessionRequest = serde_json::from_value(request.params)?;
            match resume_session_inner(state, params.session_id).await {
                Ok(result) => write_rpc_response(reader.get_mut(), request.id, result).await?,
                Err(err) => write_rpc_error(reader.get_mut(), request.id, err.to_string()).await?,
            }
        }
        "pb.session.answer" => {
            let params: AnswerSessionQuestionRequest = serde_json::from_value(request.params)?;
            match answer_question_inner(
                state,
                params.session_id,
                AnswerQuestionRequest {
                    question_id: params.question_id,
                    answer: params.answer,
                },
            )
            .await
            {
                Ok(result) => write_rpc_response(reader.get_mut(), request.id, result).await?,
                Err(err) => write_rpc_error(reader.get_mut(), request.id, err.to_string()).await?,
            }
        }
        "pb.session.delete" => {
            let params: WatchSessionRequest = serde_json::from_value(request.params)?;
            match delete_session_inner(state, &params.session_id).await {
                Ok(result) => write_rpc_response(reader.get_mut(), request.id, result).await?,
                Err(err) => write_rpc_error(reader.get_mut(), request.id, err.to_string()).await?,
            }
        }
        "pb.session.get" => {
            let params: WatchSessionRequest = serde_json::from_value(request.params)?;
            if let Some(result) = session_details_snapshot(&state, &params.session_id).await {
                write_rpc_response(reader.get_mut(), request.id, result).await?;
            } else {
                write_rpc_error(
                    reader.get_mut(),
                    request.id,
                    format!("session not found: {}", params.session_id),
                )
                .await?;
            }
        }
        "pb.session.watch" => {
            let params: WatchSessionRequest = serde_json::from_value(request.params)?;
            if let Err(err) =
                watch_session(reader.get_mut(), request.id, state, params.session_id).await
            {
                write_rpc_error(reader.get_mut(), request.id, err.to_string()).await?;
            }
        }
        other => {
            write_rpc_error(
                reader.get_mut(),
                request.id,
                format!("unknown method: {other}"),
            )
            .await?;
        }
    }
    Ok(())
}

async fn write_rpc_response<T: Serialize>(
    stream: &mut tokio::net::UnixStream,
    id: u64,
    result: T,
) -> Result<()> {
    let response = RpcResponse {
        id,
        ok: true,
        result: Some(result),
        error: None,
    };
    write_json_line(stream, &response).await
}

async fn write_rpc_error(
    stream: &mut tokio::net::UnixStream,
    id: u64,
    error: String,
) -> Result<()> {
    let response = RpcResponse::<()> {
        id,
        ok: false,
        result: None,
        error: Some(error),
    };
    write_json_line(stream, &response).await
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

async fn watch_session(
    stream: &mut tokio::net::UnixStream,
    id: u64,
    state: AppState,
    session_id: String,
) -> Result<()> {
    let (mut receiver, history, running) = {
        let sessions = state.sessions.lock().await;
        let session = sessions
            .get(&session_id)
            .ok_or_else(|| anyhow::anyhow!("session not found: {session_id}"))?;
        let history = session
            .history
            .lock()
            .map_err(|_| anyhow::anyhow!("session history lock is poisoned: {session_id}"))?;
        // Publishers append while holding this same lock and only then broadcast. Subscribing
        // before cloning the locked history makes the snapshot/live handoff atomic.
        let receiver = session.sender.subscribe();
        (
            receiver,
            history.clone(),
            session.status == SessionStatus::Queued
                || session.status == SessionStatus::Running
                || session.pending_question.is_some(),
        )
    };

    write_rpc_response(
        stream,
        id,
        SessionFinished {
            session_id: session_id.clone(),
            running,
        },
    )
    .await?;

    let mut last_sequence = 0;
    for envelope in history {
        last_sequence = last_sequence.max(envelope.transcript.sequence);
        write_session_event(stream, envelope).await?;
    }

    loop {
        tokio::select! {
            biased;
            message = receiver.recv() => {
                match message {
                    Ok(envelope) if envelope.transcript.sequence <= last_sequence => {}
                    Ok(envelope) if envelope.transcript.sequence == last_sequence.saturating_add(1) => {
                        last_sequence = envelope.transcript.sequence;
                        write_session_event(stream, envelope).await?;
                    }
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {
                        let (_, replay) = terminal_session_snapshot(&state, &session_id, last_sequence).await?;
                        for envelope in replay {
                            last_sequence = envelope.transcript.sequence;
                            write_session_event(stream, envelope).await?;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        let (is_running, replay) = terminal_session_snapshot(&state, &session_id, last_sequence).await?;
                        for envelope in replay {
                            write_session_event(stream, envelope).await?;
                        }
                        if is_running {
                            bail!("session event stream closed while session is running: {session_id}");
                        }
                        let notification = RpcNotification {
                            method: "pb.session.finished",
                            params: SessionFinished {
                                session_id: session_id.clone(),
                                running: false,
                            },
                        };
                        write_json_line(stream, &notification).await?;
                        break;
                    }
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(250)) => {
                let (is_running, replay) = terminal_session_snapshot(&state, &session_id, last_sequence).await?;
                for envelope in replay {
                    last_sequence = envelope.transcript.sequence;
                    write_session_event(stream, envelope).await?;
                }
                if !is_running {
                    let notification = RpcNotification {
                        method: "pb.session.finished",
                        params: SessionFinished {
                            session_id: session_id.clone(),
                            running: false,
                        },
                    };
                    write_json_line(stream, &notification).await?;
                    break;
                }
            }
        }
    }

    Ok(())
}

async fn write_session_event(
    stream: &mut tokio::net::UnixStream,
    envelope: EventEnvelope,
) -> Result<()> {
    let notification = RpcNotification {
        method: "pb.session.event",
        params: envelope,
    };
    write_json_line(stream, &notification).await
}

async fn terminal_session_snapshot(
    state: &AppState,
    session_id: &str,
    last_sequence: u64,
) -> Result<(bool, Vec<EventEnvelope>)> {
    let sessions = state.sessions.lock().await;
    let session = sessions
        .get(session_id)
        .ok_or_else(|| anyhow::anyhow!("session not found: {session_id}"))?;
    let history = session
        .history
        .lock()
        .map_err(|_| anyhow::anyhow!("session history lock is poisoned: {session_id}"))?;
    let replay = terminal_replay_after(&history, last_sequence)?;
    let is_running = session.status == SessionStatus::Queued
        || session.status == SessionStatus::Running
        || session.pending_question.is_some();
    Ok((is_running, replay))
}

fn terminal_replay_after(
    history: &[EventEnvelope],
    last_sequence: u64,
) -> Result<Vec<EventEnvelope>> {
    let replay = history
        .iter()
        .filter(|envelope| envelope.transcript.sequence > last_sequence)
        .cloned()
        .collect::<Vec<_>>();
    if let Some(first) = replay.first()
        && first.transcript.sequence != last_sequence.saturating_add(1)
    {
        bail!(
            "session event history no longer contains sequence {}",
            last_sequence.saturating_add(1)
        );
    }
    if replay
        .windows(2)
        .any(|pair| pair[1].transcript.sequence != pair[0].transcript.sequence.saturating_add(1))
    {
        bail!("session event history contains a sequence gap");
    }
    Ok(replay)
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
    let (sender, _) = broadcast::channel(256);
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
    let interrupted = persisted.running || persisted.status == SessionStatus::Running;
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
            task: persisted.task,
            title,
            branch: persisted.branch,
            workdir: persisted.workdir,
            durable,
            request_template,
            running: false,
            paused: status == SessionStatus::Paused,
            status,
            pending_question: None,
            sender,
            history,
            metrics,
            usage_records: Arc::new(StdMutex::new(usage_records)),
            workflow: persisted.workflow,
            completed_workflows: persisted.completed_workflows,
            goal: persisted.goal,
            completed_goals: persisted.completed_goals,
            multi_task: persisted.multi_task,
            completed_multi_tasks: persisted.completed_multi_tasks,
            pending_user_messages,
            accepting_user_messages: Arc::new(AtomicBool::new(false)),
            pause_token: Arc::new(AtomicBool::new(false)),
            cancel_token: Arc::new(AtomicBool::new(false)),
            started_at_ms: persisted.started_at_ms,
            updated_at_ms: persisted.updated_at_ms,
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

fn persist_session_snapshot(
    session_id: &str,
    request_template: &AgentRequest,
    branch: Option<String>,
    workdir: Option<PathBuf>,
    status: SessionStatus,
    history: &StdMutex<Vec<EventEnvelope>>,
    usage_records: &StdMutex<Vec<SessionMetricsSnapshot>>,
    workflow: Option<crate::workflow::WorkflowCheckpoint>,
    completed_workflows: Vec<crate::workflow::WorkflowSummary>,
    goal: Option<crate::goal::GoalCheckpoint>,
    completed_goals: Vec<crate::goal::GoalCheckpoint>,
    multi_task: Option<crate::task_queue::MultiTaskCheckpoint>,
    completed_multi_tasks: Vec<crate::task_queue::MultiTaskCheckpoint>,
    durable: DurableSessionProjection,
) {
    let events = history
        .lock()
        .map(|history| history.clone())
        .unwrap_or_default();
    let records = usage_records
        .lock()
        .map(|records| records.clone())
        .unwrap_or_default();
    let pending_user_messages = pending_user_messages_from_events(&events);
    let mut persisted = PersistedSession::from_parts(
        session_id.to_string(),
        request_template.clone(),
        branch,
        workdir,
        status == SessionStatus::Running,
        status,
        events,
    );
    if !records.is_empty() {
        persisted.metrics = combined_metrics(&records);
        persisted.usage_records = records;
    }
    persisted.workflow = workflow;
    persisted.completed_workflows = completed_workflows;
    persisted.goal = goal;
    persisted.completed_goals = completed_goals;
    persisted.multi_task = multi_task;
    persisted.completed_multi_tasks = completed_multi_tasks;
    persisted.pending_user_messages = pending_user_messages;
    persisted.project = durable.project;
    persisted.pending_delivery_proposal = durable.pending_delivery_proposal;
    persisted.pending_goal_proposal = durable.pending_goal_proposal;
    persisted.pending_goal_change = durable.pending_goal_change;
    if let Err(err) = session_store::save_session(&persisted) {
        eprintln!("failed to persist pb session {session_id}: {err:#}");
    }
}

fn pending_user_messages_from_events(events: &[EventEnvelope]) -> Vec<QueuedUserMessage> {
    let mut pending = VecDeque::new();
    for envelope in events {
        match &envelope.event {
            AgentEvent::UserMessage {
                message_id,
                message,
                ..
            } => pending.push_back(QueuedUserMessage {
                message_id: message_id.clone(),
                message: message.clone(),
            }),
            AgentEvent::UserMessageApplied { message_id, .. } => {
                pending.retain(|message| message.message_id != *message_id);
            }
            _ => {}
        }
    }
    pending.into_iter().collect()
}

fn persist_live_session(session_id: &str, session: &SessionState) {
    persist_session_snapshot(
        session_id,
        &session.request_template,
        session.branch.clone(),
        session.workdir.clone(),
        session.status,
        &session.history,
        &session.usage_records,
        session.workflow.clone(),
        session.completed_workflows.clone(),
        session.goal.clone(),
        session.completed_goals.clone(),
        session.multi_task.clone(),
        session.completed_multi_tasks.clone(),
        session.durable.clone(),
    );
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
        .map(|(session_id, session)| {
            let workflow = latest_workflow_summary(session);
            let goal = latest_goal_checkpoint(session);
            let (title, handoff_outcome, revision) = session_history_summary(session);
            SessionListItem {
                session_id: session_id.clone(),
                task: session.task.clone(),
                title,
                running: session.running,
                paused: session.paused,
                status: session.status,
                intent: session.request_template.intent,
                branch: session.branch.clone(),
                workdir: session
                    .workdir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                project: session.durable.project.clone(),
                handoff_outcome,
                pending_question: session.pending_question.as_ref().map(pending_question_view),
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
                revision,
            }
        })
        .collect::<Vec<_>>();
    items.sort_by_key(|b| std::cmp::Reverse(b.updated_at_ms));
    items
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

    fn record_metrics(&mut self, project_id: Option<&str>, metrics: &SessionMetricsSnapshot) {
        for cached in &mut self.entries {
            cached.add_metrics(project_id, metrics);
        }
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
    let (overall_usage, project_usage) =
        usage_summaries(&projects, &sessions, &mut usage_windows, usage_window);
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
        projects: projects.clone(),
        sessions: session_list_items(&sessions),
        overall_usage,
        project_usage,
    }
}

async fn reload_projects(state: &AppState) -> Result<()> {
    let mut current_projects = state.projects.lock().await;
    let projects = tokio::task::spawn_blocking(projects::load_projects)
        .await
        .context("project registry reload task failed")??;
    let mut sessions = state.sessions.lock().await;
    for session in sessions.values_mut() {
        let Some(stored_project) = session.durable.project.as_ref() else {
            continue;
        };
        let Some(current_project) = projects
            .iter()
            .find(|project| project.id == stored_project.id)
        else {
            continue;
        };
        let previous_path = PathBuf::from(&stored_project.path);
        let current_path = PathBuf::from(&current_project.path);
        session.durable.project = Some(SessionProject {
            id: current_project.id.clone(),
            name: current_project.name.clone(),
            path: current_project.path.clone(),
        });
        if !session.running && session.workdir.as_ref() == Some(&previous_path) {
            session.workdir = Some(current_path.clone());
        }
        if !session.running && session.request_template.workdir.as_ref() == Some(&previous_path) {
            session.request_template.workdir = Some(current_path);
        }
    }
    let changed = *current_projects != projects;
    *current_projects = projects;
    if changed {
        state.project_usage_windows.lock().await.invalidate();
    }
    drop(sessions);
    drop(current_projects);
    if changed {
        state.publish_project_session_change();
    }
    Ok(())
}

async fn mutate_project_registry<T, F>(state: &AppState, mutation: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let result = {
        let _registry_guard = state.projects.lock().await;
        tokio::task::spawn_blocking(mutation)
            .await
            .context("project registry mutation task failed")??
    };
    reload_projects(state).await?;
    Ok(result)
}

async fn project_list_snapshot(state: &AppState) -> Vec<ProjectEntry> {
    state.projects.lock().await.clone()
}

async fn session_details_snapshot(state: &AppState, id: &str) -> Option<SessionDetails> {
    let sessions = state.sessions.lock().await;
    let session = sessions.get(id)?;
    let history = session.history.lock().ok()?;
    Some(session_details_from_history(id, session, &history))
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
    let revision = history
        .last()
        .map(|envelope| envelope.transcript.sequence)
        .unwrap_or(0);
    let workflow = latest_workflow_summary(session);
    let goal = latest_goal_checkpoint(session).cloned();
    SessionDetails {
        session_id: id.to_string(),
        task: session.task.clone(),
        title,
        running: session.running,
        paused: session.paused,
        status: session.status,
        intent: session.request_template.intent,
        branch: session.branch.clone(),
        workdir: session
            .workdir
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        project: session.durable.project.clone(),
        handoff_outcome,
        pending_question: session.pending_question.as_ref().map(pending_question_view),
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
        revision,
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

fn publish_event(
    sender: &broadcast::Sender<EventEnvelope>,
    history: &StdMutex<Vec<EventEnvelope>>,
    event: AgentEvent,
) {
    publish_event_linked(sender, history, event, Vec::new());
}

fn publish_event_linked(
    sender: &broadcast::Sender<EventEnvelope>,
    history: &StdMutex<Vec<EventEnvelope>>,
    event: AgentEvent,
    supersedes: Vec<String>,
) {
    let envelope = EventEnvelope::with_timestamp(event);
    publish_event_envelope_linked(sender, history, envelope, supersedes);
}

fn publish_event_envelope_linked(
    sender: &broadcast::Sender<EventEnvelope>,
    history: &StdMutex<Vec<EventEnvelope>>,
    mut envelope: EventEnvelope,
    supersedes: Vec<String>,
) {
    envelope.transcript.supersedes = supersedes;
    let Ok(mut entries) = history.lock() else {
        tracing::error!(
            entry_key = %envelope.transcript.entry_key,
            "refusing to publish an event because session history is poisoned"
        );
        return;
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
    let _ = sender.send(envelope);
}

#[cfg(test)]
mod workflow_tests {
    use super::*;

    #[tokio::test]
    async fn project_session_stream_advances_revision_for_session_events() {
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::new())),
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
        let (session_sender, _) = broadcast::channel(16);
        state.watch_session_changes("session-1".to_string(), session_sender.subscribe());
        publish_event(
            &session_sender,
            &StdMutex::new(Vec::new()),
            AgentEvent::SessionTitle {
                title: "Live title".to_string(),
                timestamp_ms: None,
            },
        );

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), changes.recv())
                .await
                .unwrap()
                .unwrap(),
            1
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
            false,
            SessionStatus::Completed,
            Vec::new(),
        );
        let (session_id, session) = session_from_persisted(persisted);
        let (project_session_sender, _) = broadcast::channel(16);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
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
        let sender = {
            let sessions = state.sessions.lock().await;
            sessions.get(&session_id).unwrap().sender.clone()
        };
        state.watch_session_changes(session_id.clone(), sender.subscribe());
        {
            let sessions = state.sessions.lock().await;
            publish_session_state_changed(sessions.get(&session_id).unwrap());
        }

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), changes.recv())
                .await
                .unwrap()
                .unwrap(),
            1
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
        assert!(snapshot.terminal_transitions[0].revision > transition_floor);
    }

    #[test]
    fn user_messages_remain_pending_until_the_agent_loop_applies_them() {
        let queued = EventEnvelope::new(AgentEvent::UserMessage {
            message_id: "message-1".to_string(),
            message: "Please keep the public API unchanged.".to_string(),
            timestamp_ms: Some(1),
        });
        let applied = EventEnvelope::new(AgentEvent::UserMessageApplied {
            message_id: "message-1".to_string(),
            timestamp_ms: Some(2),
        });

        assert_eq!(
            pending_user_messages_from_events(std::slice::from_ref(&queued)),
            vec![QueuedUserMessage {
                message_id: "message-1".to_string(),
                message: "Please keep the public API unchanged.".to_string(),
            }]
        );
        assert!(pending_user_messages_from_events(&[queued, applied]).is_empty());
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
        assert_eq!(
            terminal_replay_after(&history, 1)
                .unwrap()
                .iter()
                .map(|event| event.transcript.sequence)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );

        history.remove(1);
        assert!(terminal_replay_after(&history, 1).is_err());
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
    fn timestamped_publication_preserves_the_stream_and_history_entry_key() {
        let (sender, mut receiver) = broadcast::channel(4);
        let history = StdMutex::new(Vec::new());
        let envelope = EventEnvelope::with_timestamp(AgentEvent::UserMessage {
            message_id: "message-1".to_string(),
            message: "Keep the boundary exact.".to_string(),
            timestamp_ms: None,
        });
        let entry_key = envelope.transcript.entry_key.clone();

        publish_event_envelope_linked(&sender, &history, envelope, vec!["older-entry".to_string()]);

        let streamed = receiver.try_recv().unwrap();
        let history = history.lock().unwrap();
        assert_eq!(streamed.transcript.entry_key, entry_key);
        assert_eq!(streamed.transcript.sequence, 1);
        assert_eq!(history[0].transcript.entry_key, entry_key);
        assert_eq!(history[0].transcript.supersedes, vec!["older-entry"]);
    }

    #[test]
    fn poisoned_history_never_publishes_an_unsequenced_event() {
        let (sender, mut receiver) = broadcast::channel(4);
        let history = StdMutex::new(Vec::new());
        let _ = std::panic::catch_unwind(|| {
            let _guard = history.lock().unwrap();
            panic!("poison event history");
        });

        publish_event(
            &sender,
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
            false,
            SessionStatus::Completed,
            Vec::new(),
        );
        let (_session_id, mut session) = session_from_persisted(persisted);
        session.status = SessionStatus::Queued;
        session.running = false;
        session.paused = false;
        let mut receiver = session.sender.subscribe();

        publish_session_state_changed(&session);

        let envelope = receiver.try_recv().unwrap();
        assert!(envelope.requires_session_snapshot());
        assert!(matches!(
            envelope.event,
            AgentEvent::SessionStateChanged {
                status: crate::events::SessionLifecycleStatus::Queued,
                running: false,
                paused: false,
                ..
            }
        ));
        assert_eq!(session.history.lock().unwrap().len(), 1);
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
            false,
            SessionStatus::Completed,
            Vec::new(),
        );
        let (session_id, mut session) = session_from_persisted(persisted);
        session.running = true;
        session.status = SessionStatus::Running;
        session
            .accepting_user_messages
            .store(true, Ordering::SeqCst);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
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

        let accepting_user_messages = {
            let sessions = state.sessions.lock().await;
            let session = sessions.get(&session_id).unwrap();
            let pending = session.pending_user_messages.lock().unwrap();
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].message_id, response.message_id);
            assert_eq!(pending[0].message, "Keep the API stable.");
            assert!(session.history.lock().unwrap().iter().any(|envelope| {
                matches!(
                    &envelope.event,
                    AgentEvent::UserMessage { message, .. } if message == "Keep the API stable."
                )
            }));
            Arc::clone(&session.accepting_user_messages)
        };
        accepting_user_messages.store(false, Ordering::SeqCst);

        let rejected = send_session_message(
            Path(session_id),
            State((state, request)),
            Json(SendSessionMessageRequest {
                message: "This arrived after the final boundary.".to_string(),
            }),
        )
        .await;
        assert_eq!(rejected.unwrap_err().status, StatusCode::CONFLICT);
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
            false,
            SessionStatus::Completed,
            Vec::new(),
        );
        let (session_id, session) = session_from_persisted(persisted);
        let (project_session_sender, _) = broadcast::channel(16);
        let mut changes = project_session_sender.subscribe();
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), session)]))),
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
            false,
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
            false,
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
    fn usage_window_cache_updates_incrementally_and_stays_bounded() {
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

        let metrics = SessionMetricsSnapshot {
            prompt_tokens: 7,
            generated_tokens: 3,
            started_at_ms: 1_000,
            ended_at_ms: 2_000,
            ..Default::default()
        };
        usage_windows.record_metrics(Some("project-1"), &metrics);
        let (_, updated) = usage_summaries(&projects, &sessions, &mut usage_windows, window);
        assert_eq!(updated["project-1"].today.tokens, 10);

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
            true,
            SessionStatus::Running,
            Vec::new(),
        );
        persisted.workflow = Some(checkpoint.clone());
        let (session_id, mut restored) = session_from_persisted(persisted);
        let (question_sender, _question_receiver) = std::sync::mpsc::channel();
        restored.pending_question = Some(PendingQuestionState {
            question_id: "question-web-restore".to_string(),
            question: "Which release should I target?".to_string(),
            choices: vec!["Next".to_string(), "Current".to_string()],
            responder: question_sender,
        });

        assert_eq!(restored.status, SessionStatus::Paused);
        assert!(restored.paused);
        assert_eq!(restored.workflow.as_ref(), Some(&checkpoint));
        assert_eq!(
            restored.request_template.workflow_checkpoint.as_ref(),
            Some(&checkpoint)
        );

        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), restored)]))),
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
            false,
            SessionStatus::Failed,
            Vec::new(),
        );
        persisted.workflow = Some(checkpoint);
        let (session_id, restored) = session_from_persisted(persisted);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), restored)]))),
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
            false,
            SessionStatus::Failed,
            Vec::new(),
        );
        persisted.workflow = Some(checkpoint);
        let (session_id, restored) = session_from_persisted(persisted);
        let state = AppState {
            sessions: Arc::new(Mutex::new(HashMap::from([(session_id.clone(), restored)]))),
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
            assert!(session.history.lock().unwrap().iter().any(|envelope| {
                matches!(
                    &envelope.event,
                    AgentEvent::Correction { summary, .. }
                        if summary == "Restarting delivery from current files"
                )
            }));
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
        let response = start_goal_inner(
            state.clone(),
            request(repo.path()),
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
        .unwrap();

        let details = session_details_snapshot(&state, &response.session_id)
            .await
            .unwrap();
        assert!(details.active_goal);
        assert_eq!(details.status, SessionStatus::Paused);
        let goal = details.goal.unwrap();
        assert_eq!(goal.run.stage, crate::goal::GoalStage::AwaitingPlanApproval);
        assert_eq!(goal.run.budget, crate::goal::GoalBudget::standard());
        assert_eq!(goal.sha256, response.goal_sha256);

        let stale = mutate_active_goal(&state, &response.goal_id, "stale", |_, _| Ok(())).await;
        assert_eq!(stale.unwrap_err().status, StatusCode::CONFLICT);
        let unchanged = get_goal(Path(response.goal_id), State((state, request(repo.path()))))
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
        .unwrap();
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
        assert_ne!(revised.goal_sha256, started.goal_sha256);

        let cancelled = cancel_goal(
            Path(started.goal_id.clone()),
            State((state.clone(), defaults.clone())),
            Json(GoalDigestRequest {
                goal_sha256: revised.goal_sha256,
                plan_sha256: None,
            }),
        )
        .await
        .unwrap()
        .0;
        let details = session_details_snapshot(&state, &cancelled.session_id)
            .await
            .unwrap();
        assert!(!details.active_goal);
        assert_eq!(details.status, SessionStatus::Completed);
        let archived = get_goal(Path(started.goal_id), State((state, defaults)))
            .await
            .unwrap()
            .0;
        assert_eq!(archived.run.stage, crate::goal::GoalStage::Cancelled);
        assert_eq!(archived.run.objective, "Revised goal");
    }
}
