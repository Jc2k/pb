use anyhow::Result;
use axum::extract::{Path, State};
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
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, broadcast};
use tokio_stream::wrappers::BroadcastStream;

use crate::agent_core::{AgentRequest, run_agent};
use crate::events::{AgentEvent, EventEnvelope};

#[derive(Debug, Clone)]
pub struct ServeArgs {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize)]
struct StartSessionRequest {
    task: String,
    model: Option<String>,
    model_dir: Option<String>,
    workdir: Option<String>,
    branch: Option<String>,
    max_steps: Option<usize>,
    max_tokens: Option<i32>,
    ctx_size: Option<u32>,
    threads: Option<i32>,
    threads_batch: Option<i32>,
    gpu_layers: Option<u32>,
    temperature: Option<f32>,
    top_k: Option<i32>,
    seed: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
struct ContinueSessionRequest {
    task: String,
}

#[derive(Debug, Serialize)]
struct SessionResponse {
    session_id: String,
}

#[derive(Debug, Serialize)]
struct SessionListItem {
    session_id: String,
    task: String,
    running: bool,
    branch: Option<String>,
    updated_at_ms: u128,
}

#[derive(Debug, Serialize)]
struct SessionDetails {
    session_id: String,
    task: String,
    running: bool,
    branch: Option<String>,
    events: Vec<EventEnvelope>,
}

#[derive(Debug, Clone)]
struct SessionState {
    task: String,
    branch: Option<String>,
    workdir: Option<PathBuf>,
    request_template: AgentRequest,
    running: bool,
    sender: broadcast::Sender<EventEnvelope>,
    history: Arc<StdMutex<Vec<EventEnvelope>>>,
    updated_at_ms: u128,
}

#[derive(Debug, Clone)]
struct AppState {
    sessions: Arc<Mutex<HashMap<String, SessionState>>>,
}

#[derive(RustEmbed)]
#[folder = "webui/dist"]
struct WebAssets;

pub async fn run_server(args: ServeArgs, defaults: AgentRequest) -> Result<()> {
    let state = AppState {
        sessions: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/api/sessions", post(start_session).get(list_sessions))
        .route("/api/sessions/{id}", get(get_session))
        .route("/api/sessions/{id}/continue", post(continue_session))
        .route("/api/sessions/{id}/events", get(session_events))
        .route("/", get(index))
        .route("/{*path}", get(static_asset))
        .with_state((state, defaults));

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    println!("pb serve listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn start_session(
    State((state, defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<StartSessionRequest>,
) -> Result<Json<SessionResponse>, StatusCode> {
    let session_id = new_session_id();
    let (sender, _) = broadcast::channel(256);

    let mut request = defaults.clone();
    request.task = req.task.clone();
    if let Some(model) = req.model {
        request.model = model;
    }
    request.model_dir = req.model_dir.map(PathBuf::from);
    request.workdir = req.workdir.as_ref().map(PathBuf::from);
    request.branch = req.branch.clone();
    request.max_steps = req.max_steps.unwrap_or(request.max_steps);
    request.max_tokens = req.max_tokens.unwrap_or(request.max_tokens);
    request.ctx_size = req.ctx_size.unwrap_or(request.ctx_size);
    request.threads = req.threads.or(request.threads);
    request.threads_batch = req.threads_batch.or(request.threads_batch);
    request.gpu_layers = req.gpu_layers.unwrap_or(request.gpu_layers);
    request.temperature = req.temperature.unwrap_or(request.temperature);
    request.top_k = req.top_k.unwrap_or(request.top_k);
    request.seed = req.seed.unwrap_or(request.seed);

    let now = now_millis();
    let session = SessionState {
        task: req.task,
        branch: request.branch.clone(),
        workdir: request.workdir.clone(),
        request_template: request.clone(),
        running: true,
        sender: sender.clone(),
        history: Arc::new(StdMutex::new(Vec::new())),
        updated_at_ms: now,
    };

    {
        let mut sessions = state.sessions.lock().await;
        sessions.insert(session_id.clone(), session);
    }

    spawn_agent_run(state.clone(), session_id.clone(), request);

    Ok(Json(SessionResponse { session_id }))
}

async fn continue_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
    Json(req): Json<ContinueSessionRequest>,
) -> Result<Json<SessionResponse>, StatusCode> {
    let mut sessions = state.sessions.lock().await;
    let session = sessions.get_mut(&id).ok_or(StatusCode::NOT_FOUND)?;
    if session.running {
        return Err(StatusCode::CONFLICT);
    }

    let mut request = session.request_template.clone();
    request.task = req.task;
    request.branch = session.branch.clone();
    request.workdir = session.workdir.clone();
    session.task = request.task.clone();
    session.running = true;
    session.updated_at_ms = now_millis();

    drop(sessions);
    spawn_agent_run(state.clone(), id.clone(), request);

    Ok(Json(SessionResponse { session_id: id }))
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
            branch: session.branch.clone(),
            updated_at_ms: session.updated_at_ms,
        })
        .collect::<Vec<_>>();
    items.sort_by(|a, b| b.updated_at_ms.cmp(&a.updated_at_ms));
    Json(items)
}

async fn get_session(
    Path(id): Path<String>,
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Result<Json<SessionDetails>, StatusCode> {
    let sessions = state.sessions.lock().await;
    let session = sessions.get(&id).ok_or(StatusCode::NOT_FOUND)?;
    let events = session.history.lock().map(|history| history.clone()).unwrap_or_default();
    Ok(Json(SessionDetails {
        session_id: id,
        task: session.task.clone(),
        running: session.running,
        branch: session.branch.clone(),
        events,
    }))
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

        let result = tokio::task::spawn_blocking(move || {
            let sender = sender.clone();
            let history = Arc::clone(&history);
            run_agent(request, &models_root, |event| {
                publish_event(&sender, &history, event);
            })
        })
        .await;

        let mut sessions = state.sessions.lock().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.running = false;
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
                        },
                    );
                }
                Err(err) => {
                    publish_event(
                        &session.sender,
                        &session.history,
                        AgentEvent::Error {
                            message: err.to_string(),
                        },
                    );
                }
            }
        }
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
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, mime.as_ref())],
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

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn publish_event(
    sender: &broadcast::Sender<EventEnvelope>,
    history: &StdMutex<Vec<EventEnvelope>>,
    event: AgentEvent,
) {
    let envelope = EventEnvelope::new(event);
    let _ = sender.send(envelope.clone());
    if let Ok(mut entries) = history.lock() {
        entries.push(envelope);
    }
}
