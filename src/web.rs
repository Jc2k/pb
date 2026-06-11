use anyhow::{Context, Result};
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
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, broadcast};
use tokio_stream::wrappers::BroadcastStream;

use crate::agent_core::{AgentRequest, run_agent};
use crate::events::{AgentEvent, EventEnvelope};

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
    pub top_k: Option<i32>,
    pub seed: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinueSessionRequest {
    pub task: String,
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
    pub branch: Option<String>,
    pub workdir: Option<String>,
    pub updated_at_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDetails {
    pub session_id: String,
    pub task: String,
    pub running: bool,
    pub branch: Option<String>,
    pub events: Vec<EventEnvelope>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub busy: bool,
    pub running_sessions: usize,
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
        .route("/api/projects", get(list_projects))
        .route("/api/status", get(status))
        .route("/", get(index))
        .route("/{*path}", get(static_asset))
        .with_state((state.clone(), defaults.clone()));

    spawn_unix_rpc_server(args.socket_path.clone(), state, defaults).await?;

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
        task: request.task.clone(),
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

    Ok(SessionResponse { session_id })
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
        branch: session.branch.clone(),
        events,
    }))
}

async fn status(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Json<StatusResponse> {
    let sessions = state.sessions.lock().await;
    let running_sessions = sessions.values().filter(|session| session.running).count();
    Json(StatusResponse {
        busy: running_sessions > 0,
        running_sessions,
        total_sessions: sessions.len(),
    })
}

async fn list_projects(
    State((state, _defaults)): State<(AppState, AgentRequest)>,
) -> Json<Vec<String>> {
    let sessions = state.sessions.lock().await;
    let mut projects: Vec<String> = sessions
        .values()
        .filter_map(|s| s.workdir.as_ref())
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    projects.sort();
    projects.dedup();
    Json(projects)
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
        (session.sender.subscribe(), history, session.running)
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
                .map(|session| session.running)
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

async fn session_list_snapshot(state: &AppState) -> Vec<SessionListItem> {
    let sessions = state.sessions.lock().await;
    let mut items = sessions
        .iter()
        .map(|(session_id, session)| SessionListItem {
            session_id: session_id.clone(),
            task: session.task.clone(),
            running: session.running,
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
        branch: session.branch.clone(),
        events,
    })
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
        if entries.len() > MAX_HISTORY_EVENTS {
            let overflow = entries.len() - MAX_HISTORY_EVENTS;
            entries.drain(..overflow);
        }
    }
}
