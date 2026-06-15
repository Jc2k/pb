use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agent_core::{AgentRequest, find_git_root};
use crate::events::EventEnvelope;
use crate::projects::ProjectEntry;

const NOTES_NAMESPACE: &str = "refs/notes/pb/sessions";
const MAX_RESTORED_HISTORY_EVENTS: usize = 1_000;
const SESSION_GIT_NAME: &str = "pb";
const SESSION_GIT_EMAIL: &str = "pb@localhost";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Queued,
    Running,
    Paused,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSession {
    pub session_id: String,
    pub task: String,
    pub branch: Option<String>,
    pub workdir: Option<PathBuf>,
    pub request_template: AgentRequest,
    pub running: bool,
    #[serde(default)]
    pub status: Option<SessionStatus>,
    pub updated_at_ms: u128,
    pub events: Vec<EventEnvelope>,
}

impl PersistedSession {
    pub fn from_parts(
        session_id: String,
        request_template: AgentRequest,
        branch: Option<String>,
        workdir: Option<PathBuf>,
        running: bool,
        status: SessionStatus,
        events: Vec<EventEnvelope>,
    ) -> Self {
        Self {
            session_id,
            task: request_template.task.clone(),
            branch,
            workdir,
            request_template,
            running,
            status: Some(status),
            updated_at_ms: now_millis(),
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
    for project in projects {
        let path = PathBuf::from(&project.path);
        let Ok(root) = path.canonicalize() else {
            continue;
        };
        let root = find_git_root(&root).unwrap_or(root);
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

fn restore_project_sessions(workspace_root: &Path) -> Result<Vec<PersistedSession>> {
    ensure_git_repository(workspace_root)?;
    let refs = run_git(
        workspace_root,
        ["for-each-ref", NOTES_NAMESPACE, "--format=%(refname)"],
    )?;
    let mut sessions = Vec::new();
    for note_ref in refs.lines().map(str::trim).filter(|line| !line.is_empty()) {
        match read_note(workspace_root, note_ref).and_then(|payload| parse_session(&payload)) {
            Ok(mut session) => {
                session.status = Some(
                    match session.status.unwrap_or_else(|| {
                        if session.running {
                            SessionStatus::Running
                        } else {
                            SessionStatus::Completed
                        }
                    }) {
                        SessionStatus::Running | SessionStatus::Queued | SessionStatus::Paused => {
                            SessionStatus::Paused
                        }
                        SessionStatus::Completed => SessionStatus::Completed,
                    },
                );
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
    serde_json::from_str(payload).context("failed to parse session note")
}

fn trim_events(mut events: Vec<EventEnvelope>) -> Vec<EventEnvelope> {
    if events.len() > MAX_RESTORED_HISTORY_EVENTS {
        let overflow = events.len() - MAX_RESTORED_HISTORY_EVENTS;
        events.drain(..overflow);
    }
    events
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

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
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
            model: "model.gguf".to_string(),
            model_dir: None,
            workdir: Some(workdir.to_path_buf()),
            branch: Some("pb/test".to_string()),
            max_steps: 1,
            max_tokens: 1,
            ctx_size: 128,
            threads: None,
            threads_batch: None,
            gpu_layers: 0,
            temperature: 0.0,
            profile: crate::agent_core::AgentProfile::Build,
            infer_profile: false,
            sub_agent_depth: 0,
            top_k: 1,
            seed: 0,
            environment: None::<EnvironmentConfig>,
        }
    }

    fn init_repo() -> TempDir {
        let dir = TempDir::new().unwrap();
        run_git(dir.path(), ["init"]).unwrap();
        dir
    }

    #[test]
    fn session_note_ref_rejects_path_separators() {
        assert!(session_note_ref("session-123").is_ok());
        assert!(session_note_ref("../bad").is_err());
    }

    #[test]
    fn save_restore_and_delete_session_note() {
        let dir = init_repo();
        let request = request(dir.path());
        let session = PersistedSession::from_parts(
            "session-123".to_string(),
            request,
            Some("pb/test".to_string()),
            Some(dir.path().to_path_buf()),
            true,
            SessionStatus::Running,
            vec![EventEnvelope::new(AgentEvent::Final {
                content: "done".to_string(),
                profile: AgentProfile::Build,
            })],
        );

        save_session(&session).unwrap();
        let restored = restore_project_sessions(dir.path()).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].session_id, "session-123");
        assert!(!restored[0].running);
        assert_eq!(restored[0].events.len(), 1);

        delete_session(dir.path(), "session-123").unwrap();
        assert!(restore_project_sessions(dir.path()).unwrap().is_empty());
    }
}
