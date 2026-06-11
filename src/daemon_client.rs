use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use crate::cli_ui::render_event;
use crate::events::EventEnvelope;
use crate::projects::{AddProjectRequest, ProjectEntry, RemoveProjectRequest};
use crate::web::{
    SessionFinished, SessionListItem, SessionResponse, StartSessionRequest, WatchSessionRequest,
};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Serialize)]
struct RpcRequest<T: Serialize> {
    id: u64,
    method: &'static str,
    params: T,
}

#[derive(Debug, Deserialize)]
struct RpcResponse<T> {
    id: u64,
    ok: bool,
    result: Option<T>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RpcNotification {
    method: String,
    params: Value,
}

pub fn default_socket_path() -> PathBuf {
    if let Some(runtime_dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime_dir).join("pb.sock");
    }
    let user = std::env::var("USER")
        .ok()
        .filter(|user| !user.is_empty())
        .unwrap_or_else(|| "user".to_string());
    PathBuf::from(format!("/tmp/pb-{user}.sock"))
}

pub async fn start_session(
    socket_path: &PathBuf,
    params: StartSessionRequest,
) -> Result<SessionResponse> {
    request(socket_path, "pb.session.start", params).await
}

pub async fn list_sessions(socket_path: &PathBuf) -> Result<Vec<SessionListItem>> {
    request(socket_path, "pb.session.list", serde_json::json!({})).await
}

pub async fn add_project(socket_path: &PathBuf, params: AddProjectRequest) -> Result<ProjectEntry> {
    request(socket_path, "pb.projects.add", params).await
}

pub async fn list_projects(socket_path: &PathBuf) -> Result<Vec<ProjectEntry>> {
    request(socket_path, "pb.projects.list", serde_json::json!({})).await
}

pub async fn remove_project(
    socket_path: &PathBuf,
    params: RemoveProjectRequest,
) -> Result<ProjectEntry> {
    request(socket_path, "pb.projects.rm", params).await
}

pub async fn watch_session(socket_path: &PathBuf, session_id: String) -> Result<()> {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let mut stream = UnixStream::connect(socket_path).await.with_context(|| {
        format!(
            "failed to connect to pb daemon at {}",
            socket_path.display()
        )
    })?;
    let request = RpcRequest {
        id,
        method: "pb.session.watch",
        params: WatchSessionRequest { session_id },
    };
    write_json_line(&mut stream, &request).await?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let response: RpcResponse<SessionFinished> = serde_json::from_str(&line)?;
    if response.id != id {
        bail!("daemon returned response id {}, expected {id}", response.id);
    }
    if !response.ok {
        bail!(
            response
                .error
                .unwrap_or_else(|| "daemon request failed".to_string())
        );
    }

    loop {
        line.clear();
        if reader.read_line(&mut line).await? == 0 {
            break;
        }
        let notification: RpcNotification = serde_json::from_str(&line)?;
        match notification.method.as_str() {
            "pb.session.event" => {
                let envelope: EventEnvelope = serde_json::from_value(notification.params)?;
                render_event(&envelope.event);
            }
            "pb.session.finished" => break,
            _ => {}
        }
    }
    Ok(())
}

async fn request<T, R>(socket_path: &PathBuf, method: &'static str, params: T) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let mut stream = UnixStream::connect(socket_path).await.with_context(|| {
        format!(
            "failed to connect to pb daemon at {}",
            socket_path.display()
        )
    })?;
    let request = RpcRequest { id, method, params };
    write_json_line(&mut stream, &request).await?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let response: RpcResponse<R> = serde_json::from_str(&line)?;
    if response.id != id {
        bail!("daemon returned response id {}, expected {id}", response.id);
    }
    if response.ok {
        response
            .result
            .context("daemon returned an empty success response")
    } else {
        bail!(
            response
                .error
                .unwrap_or_else(|| "daemon request failed".to_string())
        )
    }
}

async fn write_json_line<T: Serialize>(stream: &mut UnixStream, value: &T) -> Result<()> {
    let mut data = serde_json::to_vec(value)?;
    data.push(b'\n');
    stream.write_all(&data).await?;
    stream.flush().await?;
    Ok(())
}
