use serde::{Deserialize, Serialize};

use crate::events::EventEnvelope;

/// One newline-delimited frame on the local daemon socket.
///
/// The tag makes request completion and stream traffic unambiguous. In particular, a watch may
/// emit exactly one `Response` or `Error` for its request id; everything after a successful
/// response is a stream frame and can never be mistaken for a second request response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "frame", rename_all = "snake_case", deny_unknown_fields)]
pub enum RpcFrame {
    Response {
        id: u64,
        result: serde_json::Value,
    },
    Error {
        id: u64,
        error: String,
    },
    SessionEvent {
        session_id: String,
        event: EventEnvelope,
    },
    SessionFinished {
        session_id: String,
    },
    ReplayReset {
        session_id: String,
        after_sequence: u64,
        events: Vec<EventEnvelope>,
    },
    StreamError {
        session_id: String,
        error: String,
    },
}

impl RpcFrame {
    pub fn response(id: u64, result: impl Serialize) -> serde_json::Result<Self> {
        Ok(Self::Response {
            id,
            result: serde_json::to_value(result)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_daemon_frame_has_an_explicit_phase_tag() {
        let response = RpcFrame::response(7, serde_json::json!({ "ok": true })).unwrap();
        let encoded = serde_json::to_value(response).unwrap();
        assert_eq!(encoded["frame"], "response");

        let decoded: RpcFrame = serde_json::from_value(serde_json::json!({
            "frame": "stream_error",
            "session_id": "session-1",
            "error": "closed"
        }))
        .unwrap();
        assert!(matches!(decoded, RpcFrame::StreamError { .. }));

        let event = RpcFrame::SessionEvent {
            session_id: "session-1".to_string(),
            event: EventEnvelope::new(crate::events::AgentEvent::SessionTitle {
                title: "typed stream".to_string(),
                timestamp_ms: Some(1),
            }),
        };
        assert_eq!(
            serde_json::to_value(event).unwrap()["frame"],
            "session_event"
        );

        assert!(
            serde_json::from_value::<RpcFrame>(serde_json::json!({
                "frame": "response",
                "id": 7,
                "result": {},
                "notification": {}
            }))
            .is_err()
        );
    }
}
