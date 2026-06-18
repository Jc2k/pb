use anyhow::{Context, Result};
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex, mpsc};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, broadcast};
use tokio_stream::wrappers::BroadcastStream;

use crate::agent_core::{AgentProfile, AgentRequest, EventSink, run_agent};
use crate::events::{AgentEvent, EventEnvelope};
use crate::projects::{self, AddProjectRequest, ProjectEntry, RemoveProjectRequest};
use crate::session_store::{self, PersistedSession, SessionStatus};

const MAX_HISTORY_EVENTS: usize = 1_000;
const SESSION_HISTORY_RESPONSE_LIMIT: usize = 300;

#[derive(Debug, Clone)]
pub struct ServeArgs {
    pub host: String,
    pub port: u16,
    pub socket_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartSessionRequest {
    pub task: String,
    pub model: Option<String>,
    pub model_dir: Option<String>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinueSessionRequest {
    pub task: String,
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
pub struct SessionResponse {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionListItem {
    pub session_id: String,
    pub task: String,
    pub running: bool,
    pub paused: bool,
    pub status: SessionStatus,
    pub branch: Option<String>,
    pub workdir: Option<String>,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDetails {
    pub session_id: String,
    pub task: String,
    pub running: bool,
    pub paused: bool,
    pub status: SessionStatus,
    pub branch: Option<String>,
    pub pending_question: Option<PendingQuestionView>,
    pub events: Vec<EventEnvelope>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub busy: bool,
    pub running_sessions: usize,
    pub queued_sessions: usize,
    pub paused_sessions: usize,
    pub completed_sessions: usize,
    pub total_sessions: usize,
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

#[derive(Debug)]
struct SessionState {
    task: String,
    branch: Option<String>,
    workdir: Option<PathBuf>,
    request_template: AgentRequest,
    running: bool,
    paused: bool,
    status: SessionStatus,
    pending_question: Option<PendingQuestionState>,
    sender: broadcast::Sender<EventEnvelope>,
    history: Arc<StdMutex<Vec<EventEnvelope>>>,
    updated_at_ms: u64,
}

#[derive(Debug, Clone)]
struct AppState {
    sessions: Arc<Mutex<HashMap<String, SessionState>>>,
    projects: Arc<Mutex<Vec<ProjectEntry>>>,
}

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
    let state = AppState {
        sessions: Arc::new(Mutex::new(restored_sessions)),
        projects: Arc::new(Mutex::new(project_entries)),
    };

    let app = Router::new()
        .route("/api/sessions", post(start_session).get(list_sessions))
        .route(
            "/api/sessions/{id}",
            get(get_session).delete(delete_session),
        )
        .route("/api/sessions/{id}/continue", post(continue_session))
        .route("/api/sessions/{id}/resume", post(resume_session))
        .route("/api/sessions/{id}/answer", post(answer_question))
        .route("/api/sessions/{id}/events", get(session_events))
        .route("/api/projects", get(list_projects))
        .route("/api/status", get(status))
        .route("/api/current-user.png", get(crate::user::avatar_png))
        .route("/api/current-user", get(crate::user::user_info))
        .route("/auth/github/callback", get(github_oauth_callback))
        .route("/", get(index))
        .route("/{*path}", get(static_asset))
        .with_state((state.clone(), defaults.clone()));

    if let Err(err) = spawn_unix_rpc_server(args.socket_path.clone(), state, defaults).await {
        notify_ready(&mut ready, Err(err.to_string()));
        return Err(err);
    }

    let addr: SocketAddr = match format!("{}:{}", args.host, args.port).parse() {
        Ok(addr) => addr,
        Err(err) => {
            notify_ready(&mut ready, Err(err.to_string()));
            return Err(err.into());
        }
    };
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(err) => {
            notify_ready(&mut ready, Err(err.to_string()));
            return Err(err.into());
        }
    };
    println!("pb serve listening on http://{}", addr);
    notify_ready(&mut ready, Ok(addr));
    axum::serve(listener, app).await?;
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

async fn start_session(
    State((state, defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<StartSessionRequest>,
) -> Result<Json<SessionResponse>, StatusCode> {
    start_session_inner(state, defaults, req)
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn start_session_inner(
    state: AppState,
    defaults: AgentRequest,
    req: StartSessionRequest,
) -> Result<SessionResponse> {
    let session_id = new_session_id();
    let (sender, _) = broadcast::channel(256);

    let mut request = defaults.clone();
    request.task = req.task.clone();
    if let Some(model) = req.model {
        request.model = model;
    }
    if let Some(model_dir) = req.model_dir {
        request.model_dir = Some(PathBuf::from(model_dir));
    }
    if let Some(workdir) = req.workdir {
        request.workdir = Some(PathBuf::from(workdir));
    }
    request.branch = req.branch.clone();
    request.session_id = session_id.clone();
    request.max_steps = req.max_steps.unwrap_or(request.max_steps);
    request.max_tokens = req.max_tokens.unwrap_or(request.max_tokens);
    request.ctx_size = req.ctx_size.unwrap_or(request.ctx_size);
    request.threads = req.threads.or(request.threads);
    request.threads_batch = req.threads_batch.or(request.threads_batch);
    request.gpu_layers = req.gpu_layers.unwrap_or(request.gpu_layers);
    request.temperature = req.temperature.unwrap_or(request.temperature);
    request.profile = req.profile.unwrap_or(request.profile);
    request.infer_profile = req.profile.is_none();
    request.top_k = req.top_k.unwrap_or(request.top_k);
    request.seed = req.seed.unwrap_or(request.seed);

    let now = now_millis();
    let session = SessionState {
        task: request.task.clone(),
        branch: request.branch.clone(),
        workdir: request.workdir.clone(),
        request_template: request.clone(),
        running: false,
        paused: false,
        status: SessionStatus::Queued,
        pending_question: None,
        sender: sender.clone(),
        history: Arc::new(StdMutex::new(Vec::new())),
        updated_at_ms: now,
    };

    {
        let mut sessions = state.sessions.lock().await;
        sessions.insert(session_id.clone(), session);
    }

    let empty_history = StdMutex::new(Vec::new());
    persist_session_snapshot(
        &session_id,
        &request,
        request.branch.clone(),
        request.workdir.clone(),
        SessionStatus::Queued,
        &empty_history,
    );

    dispatch_next_session(state.clone());

    Ok(SessionResponse { session_id })
}

async fn continue_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<ContinueSessionRequest>,
) -> Result<Json<SessionResponse>, StatusCode> {
    let mut sessions = state.sessions.lock().await;
    let session = sessions.get_mut(&id).ok_or(StatusCode::NOT_FOUND)?;
    if session.status != SessionStatus::Completed {
        return Err(StatusCode::CONFLICT);
    }

    let mut request = session.request_template.clone();
    request.task = req.task;
    request.infer_profile = true;
    request.branch = session.branch.clone();
    request.workdir = session.workdir.clone();
    session.task = request.task.clone();
    session.request_template = request.clone();
    session.running = false;
    session.paused = false;
    session.status = SessionStatus::Queued;
    session.pending_question = None;
    session.updated_at_ms = now_millis();

    let history = Arc::clone(&session.history);
    let branch = session.branch.clone();
    let workdir = session.workdir.clone();
    drop(sessions);
    persist_session_snapshot(
        &id,
        &request,
        branch,
        workdir,
        SessionStatus::Queued,
        &history,
    );
    dispatch_next_session(state.clone());

    Ok(Json(SessionResponse { session_id: id }))
}

async fn resume_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionResponse>, StatusCode> {
    resume_session_inner(state, id)
        .await
        .map(Json)
        .map_err(|err| match err.downcast_ref::<ResumeSessionError>() {
            Some(ResumeSessionError::NotFound) => StatusCode::NOT_FOUND,
            Some(ResumeSessionError::Conflict) | None => StatusCode::CONFLICT,
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
    if session.status != SessionStatus::Paused || session.pending_question.is_some() {
        anyhow::bail!(ResumeSessionError::Conflict);
    }
    session.running = false;
    session.paused = false;
    session.status = SessionStatus::Queued;
    session.updated_at_ms = now_millis();
    let request = session.request_template.clone();
    let branch = session.branch.clone();
    let workdir = session.workdir.clone();
    let history = Arc::clone(&session.history);
    drop(sessions);
    persist_session_snapshot(
        &id,
        &request,
        branch,
        workdir,
        SessionStatus::Queued,
        &history,
    );
    dispatch_next_session(state.clone());
    Ok(SessionResponse { session_id: id })
}

async fn answer_question(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<AnswerQuestionRequest>,
) -> Result<Json<SessionResponse>, StatusCode> {
    answer_question_inner(state, id, req)
        .await
        .map(Json)
        .map_err(|err| match err.downcast_ref::<AnswerQuestionError>() {
            Some(AnswerQuestionError::NotFound) => StatusCode::NOT_FOUND,
            Some(AnswerQuestionError::Gone) => StatusCode::GONE,
            Some(AnswerQuestionError::Conflict) | None => StatusCode::CONFLICT,
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
    if answer.is_empty() || (!pending.choices.is_empty() && !pending.choices.contains(&answer)) {
        session.pending_question = Some(pending);
        anyhow::bail!(AnswerQuestionError::Conflict);
    }

    let sender = session.sender.clone();
    let history = Arc::clone(&session.history);
    let request_template = session.request_template.clone();
    let branch = session.branch.clone();
    let workdir = session.workdir.clone();
    let question_id = req.question_id.clone();
    pending
        .responder
        .send(answer.clone())
        .map_err(|_| AnswerQuestionError::Gone)?;
    session.paused = false;
    session.running = true;
    session.status = SessionStatus::Running;
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
    );
    Ok(SessionResponse { session_id: id })
}

async fn list_sessions(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Json<Vec<SessionListItem>> {
    let sessions = state.sessions.lock().await;
    let mut items = sessions
        .iter()
        .map(|(session_id, session)| SessionListItem {
            session_id: session_id.clone(),
            task: session.task.clone(),
            running: session.running,
            paused: session.paused,
            status: session.status,
            branch: session.branch.clone(),
            workdir: session
                .workdir
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            updated_at_ms: session.updated_at_ms,
        })
        .collect::<Vec<_>>();
    items.sort_by_key(|b| std::cmp::Reverse(b.updated_at_ms));
    Json(items)
}

async fn get_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionDetails>, StatusCode> {
    let sessions = state.sessions.lock().await;
    let session = sessions.get(&id).ok_or(StatusCode::NOT_FOUND)?;
    let events = session
        .history
        .lock()
        .map(|history| {
            let start = history.len().saturating_sub(SESSION_HISTORY_RESPONSE_LIMIT);
            history[start..].to_vec()
        })
        .unwrap_or_default();
    Ok(Json(SessionDetails {
        session_id: id,
        task: session.task.clone(),
        running: session.running,
        paused: session.paused,
        status: session.status,
        branch: session.branch.clone(),
        pending_question: session.pending_question.as_ref().map(pending_question_view),
        events,
    }))
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
    Json(StatusResponse {
        busy: running_sessions > 0,
        running_sessions,
        queued_sessions,
        paused_sessions,
        completed_sessions,
        total_sessions: sessions.len(),
    })
}

async fn delete_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<DeleteSessionResponse>, StatusCode> {
    delete_session_inner(state, &id)
        .await
        .map(Json)
        .map_err(|_| StatusCode::NOT_FOUND)
}

async fn delete_session_inner(state: AppState, id: &str) -> Result<DeleteSessionResponse> {
    let session = {
        let mut sessions = state.sessions.lock().await;
        let Some(session) = sessions.get(id) else {
            anyhow::bail!("session not found: {id}");
        };
        if session.status == SessionStatus::Running
            || session.status == SessionStatus::Queued
            || session.pending_question.is_some()
        {
            anyhow::bail!("session is active: {id}");
        }
        sessions.remove(id).expect("session exists")
    };

    if let Some(workdir) = session.workdir.or(session.request_template.workdir) {
        session_store::delete_session(&workdir, id)?;
    }

    Ok(DeleteSessionResponse {
        session_id: id.to_string(),
        deleted: true,
    })
}

async fn list_projects(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Json<Vec<ProjectEntry>> {
    Json(project_list_snapshot(&state).await)
}

struct WebEventSink {
    state: AppState,
    session_id: String,
    request_template: AgentRequest,
    sender: broadcast::Sender<EventEnvelope>,
    history: Arc<StdMutex<Vec<EventEnvelope>>>,
    persisted_branch: Option<String>,
    persisted_workdir: Option<PathBuf>,
}

impl EventSink for WebEventSink {
    fn emit(&mut self, event: AgentEvent) {
        if let AgentEvent::Started {
            workspace, branch, ..
        } = &event
        {
            self.persisted_workdir = Some(PathBuf::from(workspace));
            self.persisted_branch = Some(branch.clone());
        }
        publish_event(&self.sender, &self.history, event);
        persist_session_snapshot(
            &self.session_id,
            &self.request_template,
            self.persisted_branch.clone(),
            self.persisted_workdir.clone(),
            SessionStatus::Running,
            &self.history,
        );
    }

    fn ask_user(&mut self, question: &str) -> Result<String> {
        self.ask_multiple_choice(question, &[])
    }

    fn ask_multiple_choice(&mut self, question: &str, choices: &[String]) -> Result<String> {
        let question = question.trim();
        if question.is_empty() {
            anyhow::bail!("ask_user question must not be empty");
        }
        let question_id = format!("question-{}", now_millis());
        let (tx, rx) = std::sync::mpsc::channel();
        let event = AgentEvent::UserQuestion {
            question_id: question_id.clone(),
            question: question.to_string(),
            choices: choices.to_vec(),
            timestamp_ms: Some(now_millis()),
        };

        tokio::runtime::Handle::current().block_on(async {
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
            );
            Ok::<(), anyhow::Error>(())
        })?;

        rx.recv()
            .context("session stopped before the user answered the question")
    }
}

fn dispatch_next_session(state: AppState) {
    tokio::spawn(async move {
        let next = {
            let mut sessions = state.sessions.lock().await;
            let has_active = sessions.values().any(|session| {
                session.status == SessionStatus::Running || session.pending_question.is_some()
            });
            if has_active {
                return;
            }
            let Some(session_id) = sessions
                .iter()
                .filter(|(_, session)| session.status == SessionStatus::Queued)
                .min_by_key(|(_, session)| session.updated_at_ms)
                .map(|(session_id, _)| session_id.clone())
            else {
                return;
            };
            let session = sessions
                .get_mut(&session_id)
                .expect("queued session selected from sessions map");
            session.running = true;
            session.paused = false;
            session.status = SessionStatus::Running;
            session.updated_at_ms = now_millis();
            (
                session_id,
                session.request_template.clone(),
                session.branch.clone(),
                session.workdir.clone(),
                Arc::clone(&session.history),
            )
        };

        let (session_id, request, branch, workdir, history) = next;
        persist_session_snapshot(
            &session_id,
            &request,
            branch,
            workdir,
            SessionStatus::Running,
            &history,
        );
        spawn_agent_run(state, session_id, request);
    });
}

fn spawn_agent_run(state: AppState, session_id: String, request: AgentRequest) {
    tokio::spawn(async move {
        let (models_root, sender, history) = {
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
                persisted_branch: request_for_run.branch.clone(),
                persisted_workdir: request_for_run.workdir.clone(),
            };
            run_agent(request_for_run.clone(), &models_root, sink)
        })
        .await;

        let mut sessions = state.sessions.lock().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.running = false;
            session.paused = false;
            session.status = SessionStatus::Completed;
            session.pending_question = None;
            session.updated_at_ms = now_millis();
            match result {
                Ok(Ok(run_result)) => {
                    session.branch = Some(run_result.branch);
                    session.workdir = Some(run_result.workspace_root);
                }
                Ok(Err(err)) => {
                    publish_event(
                        &session.sender,
                        &session.history,
                        AgentEvent::Error {
                            message: err.to_string(),
                            nesting_depth: None,
                            timestamp_ms: Some(now_millis()),
                        },
                    );
                }
                Err(err) => {
                    publish_event(
                        &session.sender,
                        &session.history,
                        AgentEvent::Error {
                            message: err.to_string(),
                            nesting_depth: None,
                            timestamp_ms: Some(now_millis()),
                        },
                    );
                }
            }
            persist_session_snapshot(
                &session_id,
                &session.request_template,
                session.branch.clone(),
                session.workdir.clone(),
                SessionStatus::Completed,
                &session.history,
            );
        }
        drop(sessions);
        dispatch_next_session(state.clone());
    });
}

async fn session_events(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    let receiver = {
        let sessions = state.sessions.lock().await;
        let session = sessions.get(&id).ok_or(StatusCode::NOT_FOUND)?;
        session.sender.subscribe()
    };

    let stream = BroadcastStream::new(receiver).filter_map(|message| async move {
        match message {
            Ok(envelope) => {
                let data = serde_json::to_string(&envelope).ok()?;
                Some(Ok(Event::default().data(data)))
            }
            Err(_) => None,
        }
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
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
        "pb.session.list" => {
            let result = session_list_snapshot(&state).await;
            write_rpc_response(reader.get_mut(), request.id, result).await?;
        }
        "pb.projects.add" => {
            let params: AddProjectRequest = serde_json::from_value(request.params)?;
            let result = projects::add_project(params)?;
            reload_projects(&state).await?;
            write_rpc_response(reader.get_mut(), request.id, result).await?;
        }
        "pb.projects.list" => {
            reload_projects(&state).await?;
            let result = project_list_snapshot(&state).await;
            write_rpc_response(reader.get_mut(), request.id, result).await?;
        }
        "pb.projects.rm" => {
            let params: RemoveProjectRequest = serde_json::from_value(request.params)?;
            let result = projects::remove_project(&params.name)?;
            reload_projects(&state).await?;
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
            .map(|history| history.clone())
            .unwrap_or_default();
        (
            session.sender.subscribe(),
            history,
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

    for envelope in history {
        let notification = RpcNotification {
            method: "pb.session.event",
            params: envelope,
        };
        write_json_line(stream, &notification).await?;
    }

    loop {
        let is_running = {
            let sessions = state.sessions.lock().await;
            sessions
                .get(&session_id)
                .map(|session| {
                    session.status == SessionStatus::Queued
                        || session.status == SessionStatus::Running
                        || session.pending_question.is_some()
                })
                .unwrap_or(false)
        };
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

        tokio::select! {
            message = receiver.recv() => {
                if let Ok(envelope) = message {
                    let notification = RpcNotification {
                        method: "pb.session.event",
                        params: envelope,
                    };
                    write_json_line(stream, &notification).await?;
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(250)) => {}
        }
    }

    Ok(())
}

fn restore_sessions(project_entries: &[ProjectEntry]) -> HashMap<String, SessionState> {
    session_store::restore_registered_sessions(project_entries)
        .into_iter()
        .map(session_from_persisted)
        .map(|session| (session.0, session.1))
        .collect()
}

fn session_from_persisted(persisted: PersistedSession) -> (String, SessionState) {
    let (sender, _) = broadcast::channel(256);
    let session_id = persisted.session_id.clone();
    let history = Arc::new(StdMutex::new(persisted.events));
    (
        session_id,
        SessionState {
            task: persisted.task,
            branch: persisted.branch,
            workdir: persisted.workdir,
            request_template: persisted.request_template,
            running: false,
            paused: persisted.status == Some(SessionStatus::Paused),
            status: persisted.status.unwrap_or(SessionStatus::Completed),
            pending_question: None,
            sender,
            history,
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
) {
    let events = history
        .lock()
        .map(|history| history.clone())
        .unwrap_or_default();
    let persisted = PersistedSession::from_parts(
        session_id.to_string(),
        request_template.clone(),
        branch,
        workdir,
        status == SessionStatus::Running,
        status,
        events,
    );
    if let Err(err) = session_store::save_session(&persisted) {
        eprintln!("failed to persist pb session {session_id}: {err:#}");
    }
}

async fn session_list_snapshot(state: &AppState) -> Vec<SessionListItem> {
    let sessions = state.sessions.lock().await;
    let mut items = sessions
        .iter()
        .map(|(session_id, session)| SessionListItem {
            session_id: session_id.clone(),
            task: session.task.clone(),
            running: session.running,
            paused: session.paused,
            status: session.status,
            branch: session.branch.clone(),
            workdir: session
                .workdir
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            updated_at_ms: session.updated_at_ms,
        })
        .collect::<Vec<_>>();
    items.sort_by_key(|b| std::cmp::Reverse(b.updated_at_ms));
    items
}

async fn reload_projects(state: &AppState) -> Result<()> {
    let projects = projects::load_projects()?;
    let mut current = state.projects.lock().await;
    *current = projects;
    Ok(())
}

async fn project_list_snapshot(state: &AppState) -> Vec<ProjectEntry> {
    state.projects.lock().await.clone()
}

async fn session_details_snapshot(state: &AppState, id: &str) -> Option<SessionDetails> {
    let sessions = state.sessions.lock().await;
    let session = sessions.get(id)?;
    let events = session
        .history
        .lock()
        .map(|history| {
            let start = history.len().saturating_sub(SESSION_HISTORY_RESPONSE_LIMIT);
            history[start..].to_vec()
        })
        .unwrap_or_default();
    Some(SessionDetails {
        session_id: id.to_string(),
        task: session.task.clone(),
        running: session.running,
        paused: session.paused,
        status: session.status,
        branch: session.branch.clone(),
        pending_question: session.pending_question.as_ref().map(pending_question_view),
        events,
    })
}

async fn github_oauth_callback(Query(query): Query<HashMap<String, String>>) -> Response {
    crate::github_oauth::persist_callback_from_query(&query).into_response()
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
            mime_guess::Mime::from(mime_guess::mime::TEXT_HTML)
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
    let now = now_millis();
    format!("session-{now}")
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
    let envelope = EventEnvelope::with_timestamp(event);
    let _ = sender.send(envelope.clone());
    if let Ok(mut entries) = history.lock() {
        entries.push(envelope);
        if entries.len() > MAX_HISTORY_EVENTS {
            let overflow = entries.len() - MAX_HISTORY_EVENTS;
            entries.drain(..overflow);
        }
    }
}
