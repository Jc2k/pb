use anyhow::{Context, Result, bail};
use serde::{Serialize, de::DeserializeOwned};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use crate::cli_ui::render_event;
use crate::daemon_protocol::RpcFrame;
use crate::projects::{AddProjectRequest, ProjectEntry, RemoveProjectRequest};
use crate::web::{
    AnswerQuestionRequest, DeleteSessionMutationResponse, GoalMutationReceipt,
    ProjectMutationReceipt, SessionListItem, SessionMutationReceipt, SessionResponse,
    StartSessionRequest, WatchSessionAcknowledgement, WatchSessionRequest,
};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Serialize)]
struct RpcRequest<T: Serialize> {
    id: u64,
    method: &'static str,
    params: T,
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

    #[tokio::test]
    async fn watch_rejects_eof_before_the_finished_phase() {
        let (client, mut server) = tokio::net::UnixStream::pair().unwrap();
        let server = tokio::spawn(async move {
            write_json_line(
                &mut server,
                &RpcFrame::response(
                    7,
                    WatchSessionAcknowledgement {
                        session_id: "session-eof".to_string(),
                    },
                )
                .unwrap(),
            )
            .await
            .unwrap();
        });

        let mut reader = BufReader::new(client);
        let error = consume_watch_frames(&mut reader, 7, "session-eof")
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("before session_finished"),
            "unexpected error: {error:#}"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn watch_rejects_obsolete_acknowledgement_lifecycle_fields() {
        let (client, mut server) = tokio::net::UnixStream::pair().unwrap();
        let server = tokio::spawn(async move {
            write_json_line(
                &mut server,
                &RpcFrame::Response {
                    id: 12,
                    result: serde_json::json!({
                        "session_id": "session-obsolete-ack",
                        "active": true
                    }),
                },
            )
            .await
            .unwrap();
        });

        let mut reader = BufReader::new(client);
        let error = consume_watch_frames(&mut reader, 12, "session-obsolete-ack")
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("unknown field `active`"),
            "unexpected error: {error:#}"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn watch_rejects_an_unframed_event_sequence_gap() {
        let (client, mut server) = tokio::net::UnixStream::pair().unwrap();
        let server = tokio::spawn(async move {
            write_json_line(
                &mut server,
                &RpcFrame::response(
                    8,
                    WatchSessionAcknowledgement {
                        session_id: "session-gap".to_string(),
                    },
                )
                .unwrap(),
            )
            .await
            .unwrap();
            let mut event =
                crate::events::EventEnvelope::new(crate::events::AgentEvent::SessionTitle {
                    title: "event 4".to_string(),
                    timestamp_ms: Some(4),
                });
            event.assign_sequence(4);
            write_json_line(
                &mut server,
                &RpcFrame::SessionEvent {
                    session_id: "session-gap".to_string(),
                    event,
                },
            )
            .await
            .unwrap();
        });

        let mut reader = BufReader::new(client);
        let error = consume_watch_frames(&mut reader, 8, "session-gap")
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("jumped from 0 to 4"),
            "unexpected error: {error:#}"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn watch_advances_through_an_empty_explicit_reset() {
        let (client, mut server) = tokio::net::UnixStream::pair().unwrap();
        let server = tokio::spawn(async move {
            write_json_line(
                &mut server,
                &RpcFrame::response(
                    11,
                    WatchSessionAcknowledgement {
                        session_id: "session-empty-reset".to_string(),
                    },
                )
                .unwrap(),
            )
            .await
            .unwrap();
            write_json_line(
                &mut server,
                &RpcFrame::ReplayReset {
                    session_id: "session-empty-reset".to_string(),
                    after_sequence: 0,
                    events: Vec::new(),
                },
            )
            .await
            .unwrap();
            let mut event =
                crate::events::EventEnvelope::new(crate::events::AgentEvent::SessionTitle {
                    title: "first".to_string(),
                    timestamp_ms: Some(1),
                });
            event.assign_sequence(1);
            write_json_line(
                &mut server,
                &RpcFrame::SessionEvent {
                    session_id: "session-empty-reset".to_string(),
                    event,
                },
            )
            .await
            .unwrap();
            write_json_line(
                &mut server,
                &RpcFrame::SessionFinished {
                    session_id: "session-empty-reset".to_string(),
                },
            )
            .await
            .unwrap();
        });

        let mut reader = BufReader::new(client);
        consume_watch_frames(&mut reader, 11, "session-empty-reset")
            .await
            .unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn watch_accepts_an_explicit_retained_history_reset() {
        let (client, mut server) = tokio::net::UnixStream::pair().unwrap();
        let server = tokio::spawn(async move {
            write_json_line(
                &mut server,
                &RpcFrame::response(
                    9,
                    WatchSessionAcknowledgement {
                        session_id: "session-reset".to_string(),
                    },
                )
                .unwrap(),
            )
            .await
            .unwrap();
            let events = [2, 5]
                .into_iter()
                .map(|sequence| {
                    let mut event = crate::events::EventEnvelope::new(
                        crate::events::AgentEvent::SessionTitle {
                            title: format!("retained {sequence}"),
                            timestamp_ms: Some(sequence),
                        },
                    );
                    event.assign_sequence(sequence);
                    event
                })
                .collect();
            write_json_line(
                &mut server,
                &RpcFrame::ReplayReset {
                    session_id: "session-reset".to_string(),
                    after_sequence: 0,
                    events,
                },
            )
            .await
            .unwrap();
            write_json_line(
                &mut server,
                &RpcFrame::SessionFinished {
                    session_id: "session-reset".to_string(),
                },
            )
            .await
            .unwrap();
        });

        let mut reader = BufReader::new(client);
        consume_watch_frames(&mut reader, 9, "session-reset")
            .await
            .unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn watch_rejects_an_event_for_another_session() {
        let (client, mut server) = tokio::net::UnixStream::pair().unwrap();
        let server = tokio::spawn(async move {
            write_json_line(
                &mut server,
                &RpcFrame::response(
                    10,
                    WatchSessionAcknowledgement {
                        session_id: "session-a".to_string(),
                    },
                )
                .unwrap(),
            )
            .await
            .unwrap();
            let mut event =
                crate::events::EventEnvelope::new(crate::events::AgentEvent::SessionTitle {
                    title: "wrong session".to_string(),
                    timestamp_ms: Some(1),
                });
            event.assign_sequence(1);
            write_json_line(
                &mut server,
                &RpcFrame::SessionEvent {
                    session_id: "session-b".to_string(),
                    event,
                },
            )
            .await
            .unwrap();
        });

        let mut reader = BufReader::new(client);
        let error = consume_watch_frames(&mut reader, 10, "session-a")
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("session-b, expected session-a"),
            "unexpected error: {error:#}"
        );
        server.await.unwrap();
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
) -> Result<GoalMutationReceipt> {
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
) -> Result<GoalMutationReceipt> {
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
) -> Result<GoalMutationReceipt> {
    mutate_goal(socket_path, "pb.goal.pause", goal_id, goal_sha256).await
}

pub async fn resume_goal(
    socket_path: &PathBuf,
    goal_id: String,
    goal_sha256: String,
) -> Result<GoalMutationReceipt> {
    mutate_goal(socket_path, "pb.goal.resume", goal_id, goal_sha256).await
}

pub async fn cancel_goal(
    socket_path: &PathBuf,
    goal_id: String,
    goal_sha256: String,
) -> Result<GoalMutationReceipt> {
    mutate_goal(socket_path, "pb.goal.cancel", goal_id, goal_sha256).await
}

pub async fn accept_goal(
    socket_path: &PathBuf,
    goal_id: String,
    goal_sha256: String,
) -> Result<GoalMutationReceipt> {
    mutate_goal(socket_path, "pb.goal.accept", goal_id, goal_sha256).await
}

pub async fn list_sessions(socket_path: &PathBuf) -> Result<Vec<SessionListItem>> {
    request(socket_path, "pb.session.list", serde_json::json!({})).await
}

pub async fn answer_question(
    socket_path: &PathBuf,
    session_id: String,
    params: AnswerQuestionRequest,
) -> Result<SessionMutationReceipt> {
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

pub async fn resume_session(
    socket_path: &PathBuf,
    session_id: String,
) -> Result<SessionMutationReceipt> {
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
) -> Result<DeleteSessionMutationResponse> {
    request(
        socket_path,
        "pb.session.delete",
        WatchSessionRequest { session_id },
    )
    .await
}

pub async fn add_project(
    socket_path: &PathBuf,
    params: AddProjectRequest,
) -> Result<ProjectMutationReceipt<ProjectEntry>> {
    request(socket_path, "pb.projects.add", params).await
}

pub async fn list_projects(socket_path: &PathBuf) -> Result<Vec<ProjectEntry>> {
    request(socket_path, "pb.projects.list", serde_json::json!({})).await
}

pub async fn remove_project(
    socket_path: &PathBuf,
    params: RemoveProjectRequest,
) -> Result<ProjectMutationReceipt<ProjectEntry>> {
    request(socket_path, "pb.projects.rm", params).await
}

pub async fn watch_session(socket_path: &PathBuf, session_id: String) -> Result<()> {
    let watched_session_id = session_id.clone();
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
    consume_watch_frames(&mut reader, id, &watched_session_id).await
}

async fn consume_watch_frames(
    reader: &mut (impl AsyncBufRead + Unpin),
    id: u64,
    watched_session_id: &str,
) -> Result<()> {
    let mut line = String::new();
    let mut last_sequence = 0_u64;
    reader.read_line(&mut line).await?;
    match serde_json::from_str::<RpcFrame>(&line)? {
        RpcFrame::Response {
            id: response_id,
            result,
        } if response_id == id => {
            let acknowledgement: WatchSessionAcknowledgement = serde_json::from_value(result)?;
            if acknowledgement.session_id != watched_session_id {
                bail!(
                    "daemon acknowledged session {}, expected {}",
                    acknowledgement.session_id,
                    watched_session_id
                );
            }
        }
        RpcFrame::Response {
            id: response_id, ..
        }
        | RpcFrame::Error {
            id: response_id, ..
        } if response_id != id => {
            bail!("daemon returned response id {response_id}, expected {id}")
        }
        RpcFrame::Error { error, .. } => bail!(error),
        frame => bail!("daemon returned {frame:?} before acknowledging the watch"),
    }

    loop {
        line.clear();
        if reader.read_line(&mut line).await? == 0 {
            bail!("daemon closed the watch before session_finished");
        }
        match serde_json::from_str::<RpcFrame>(&line)? {
            RpcFrame::SessionEvent { session_id, event } if session_id == watched_session_id => {
                if event.transcript.sequence != last_sequence.saturating_add(1) {
                    bail!(
                        "daemon session event sequence jumped from {last_sequence} to {} without replay_reset",
                        event.transcript.sequence
                    );
                }
                last_sequence = event.transcript.sequence;
                render_event(&event);
            }
            RpcFrame::SessionEvent { session_id, .. } => {
                bail!("daemon streamed session {session_id}, expected {watched_session_id}")
            }
            RpcFrame::SessionFinished { session_id } if session_id == watched_session_id => {
                return Ok(());
            }
            RpcFrame::SessionFinished { session_id } => {
                bail!("daemon finished session {session_id}, expected {watched_session_id}")
            }
            RpcFrame::ReplayReset {
                session_id,
                after_sequence,
                events,
            } if session_id == watched_session_id => {
                if last_sequence != after_sequence {
                    bail!(
                        "daemon replay reset cursor {after_sequence} does not match the last event {}",
                        last_sequence
                    );
                }
                let mut previous = after_sequence;
                for envelope in &events {
                    if envelope.transcript.sequence <= previous {
                        bail!("daemon replay reset events are not strictly increasing");
                    }
                    previous = envelope.transcript.sequence;
                }
                eprintln!(
                    "warning: session event history reset after sequence {after_sequence}; unavailable events may have been omitted"
                );
                for envelope in events {
                    render_event(&envelope);
                }
                last_sequence = previous;
            }
            RpcFrame::ReplayReset { session_id, .. } => {
                bail!("daemon reset session {session_id}, expected {watched_session_id}")
            }
            RpcFrame::StreamError { session_id, error } if session_id == watched_session_id => {
                bail!(error)
            }
            RpcFrame::StreamError { session_id, .. } => {
                bail!("daemon failed session {session_id}, expected {watched_session_id}")
            }
            RpcFrame::Response { .. } | RpcFrame::Error { .. } => {
                bail!("daemon sent a request response after watch acknowledgement")
            }
        }
    }
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
    match serde_json::from_str::<RpcFrame>(&line)? {
        RpcFrame::Response {
            id: response_id,
            result,
        } if response_id == id => {
            serde_json::from_value(result).context("daemon returned an invalid success response")
        }
        RpcFrame::Response {
            id: response_id, ..
        }
        | RpcFrame::Error {
            id: response_id, ..
        } if response_id != id => {
            bail!("daemon returned response id {response_id}, expected {id}")
        }
        RpcFrame::Error { error, .. } => bail!(error),
        frame => bail!("daemon returned stream frame {frame:?} for a request"),
    }
}

async fn write_json_line<T: Serialize>(stream: &mut UnixStream, value: &T) -> Result<()> {
    let mut data = serde_json::to_vec(value)?;
    data.push(b'\n');
    stream.write_all(&data).await?;
    stream.flush().await?;
    Ok(())
}
