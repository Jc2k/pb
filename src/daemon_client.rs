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
    AnswerQuestionRequest, DeleteSessionResponse, SessionFinished, SessionListItem,
    SessionResponse, StartSessionRequest, WatchSessionRequest,
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
    #[cfg(unix)]
    let user_id = unsafe { libc::geteuid() };
    #[cfg(not(unix))]
    let user_id = 0_u32;
    socket_path_for(crate::host_environment::runtime_dir(), user_id)
}

fn socket_path_for(runtime_dir: Option<PathBuf>, user_id: u32) -> PathBuf {
    if let Some(runtime_dir) = runtime_dir {
        return runtime_dir.join("pb.sock");
    }
    PathBuf::from(format!("/tmp/pb-{user_id}.sock"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn socket_path_prefers_the_runtime_directory() {
        assert_eq!(
            socket_path_for(Some(PathBuf::from("/run/user/42")), 42),
            PathBuf::from("/run/user/42/pb.sock")
        );
    }

    #[test]
    fn socket_path_fallback_uses_the_numeric_user_id() {
        assert_eq!(socket_path_for(None, 42), PathBuf::from("/tmp/pb-42.sock"));
    }
}

pub async fn start_session(
    socket_path: &PathBuf,
    params: StartSessionRequest,
) -> Result<SessionResponse> {
    request(socket_path, "pb.session.start", params).await
}

pub async fn start_goal(
    socket_path: &PathBuf,
    params: crate::web::StartGoalRequest,
) -> Result<crate::web::GoalResponse> {
    request(socket_path, "pb.goal.start", params).await
}

pub async fn get_goal(
    socket_path: &PathBuf,
    goal_id: String,
) -> Result<crate::goal::GoalCheckpoint> {
    request(
        socket_path,
        "pb.goal.get",
        serde_json::json!({ "goal_id": goal_id }),
    )
    .await
}

async fn mutate_goal(
    socket_path: &PathBuf,
    method: &'static str,
    goal_id: String,
    goal_sha256: String,
) -> Result<crate::web::GoalResponse> {
    request(
        socket_path,
        method,
        serde_json::json!({
            "goal_id": goal_id,
            "goal_sha256": goal_sha256,
        }),
    )
    .await
}

pub async fn pause_goal(
    socket_path: &PathBuf,
    goal_id: String,
    goal_sha256: String,
) -> Result<crate::web::GoalResponse> {
    mutate_goal(socket_path, "pb.goal.pause", goal_id, goal_sha256).await
}

pub async fn resume_goal(
    socket_path: &PathBuf,
    goal_id: String,
    goal_sha256: String,
) -> Result<crate::web::GoalResponse> {
    mutate_goal(socket_path, "pb.goal.resume", goal_id, goal_sha256).await
}

pub async fn cancel_goal(
    socket_path: &PathBuf,
    goal_id: String,
    goal_sha256: String,
) -> Result<crate::web::GoalResponse> {
    mutate_goal(socket_path, "pb.goal.cancel", goal_id, goal_sha256).await
}

pub async fn accept_goal(
    socket_path: &PathBuf,
    goal_id: String,
    goal_sha256: String,
) -> Result<crate::web::GoalResponse> {
    mutate_goal(socket_path, "pb.goal.accept", goal_id, goal_sha256).await
}

pub async fn list_sessions(socket_path: &PathBuf) -> Result<Vec<SessionListItem>> {
    request(socket_path, "pb.session.list", serde_json::json!({})).await
}

pub async fn answer_question(
    socket_path: &PathBuf,
    session_id: String,
    params: AnswerQuestionRequest,
) -> Result<SessionResponse> {
    request(
        socket_path,
        "pb.session.answer",
        serde_json::json!({
            "session_id": session_id,
            "question_id": params.question_id,
            "answer": params.answer,
        }),
    )
    .await
}

pub async fn resume_session(socket_path: &PathBuf, session_id: String) -> Result<SessionResponse> {
    request(
        socket_path,
        "pb.session.resume",
        WatchSessionRequest { session_id },
    )
    .await
}

pub async fn delete_session(
    socket_path: &PathBuf,
    session_id: String,
) -> Result<DeleteSessionResponse> {
    request(
        socket_path,
        "pb.session.delete",
        WatchSessionRequest { session_id },
    )
    .await
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
                render_event(&envelope);
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
