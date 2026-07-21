use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

pub const SESSION_WORKSPACE_RECORD_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceStrategy {
    WorktreeBind,
    ContainerVolume,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionWorkspaceRecord {
    pub version: u32,
    pub project_id: String,
    pub session_id: String,
    pub repository_root: PathBuf,
    pub worktree_root: PathBuf,
    pub focus_relative: PathBuf,
    pub branch: String,
    pub strategy: WorkspaceStrategy,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone)]
pub struct SessionWorkspace {
    pub record: SessionWorkspaceRecord,
    pub focus_root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct WorkspaceManager {
    state_root: PathBuf,
}

impl WorkspaceManager {
    pub fn persistent() -> Result<Self> {
        Ok(Self::new(default_state_root()?))
    }

    pub fn new(state_root: PathBuf) -> Self {
        Self { state_root }
    }

    pub fn prepare(
        &self,
        repository_root: &Path,
        focus_root: &Path,
        session_id: &str,
        branch: &str,
    ) -> Result<SessionWorkspace> {
        if session_id.trim().is_empty() {
            bail!("container session workspace requires a non-empty session id");
        }
        if branch.trim().is_empty() || branch.contains(['\n', '\r']) {
            bail!("container session workspace requires a valid branch name");
        }
        let _operation = crate::state_lock::StateFileLock::acquire(
            self.state_root
                .join("workspace-locks")
                .join(format!("{}.lock", short_sha(session_id.as_bytes()))),
            Duration::from_secs(10),
        )?;
        if let Some(existing) = self.find_record_by_session(session_id)? {
            let supplied_root = repository_root.canonicalize().with_context(|| {
                format!(
                    "failed to resolve supplied repository root {}",
                    repository_root.display()
                )
            })?;
            let valid_root = supplied_root == existing.repository_root
                || (existing.worktree_root.exists()
                    && supplied_root == existing.worktree_root.canonicalize()?);
            if !valid_root || existing.branch != branch {
                bail!(
                    "persisted session worktree does not match the requested repository or branch"
                );
            }
            if existing.worktree_root.exists() {
                let actual_branch =
                    git_capture(&existing.worktree_root, &["branch", "--show-current"])?;
                if actual_branch.trim() != branch {
                    bail!(
                        "session worktree {} is on branch '{}' rather than '{}'",
                        existing.worktree_root.display(),
                        actual_branch.trim(),
                        branch
                    );
                }
                return Ok(SessionWorkspace {
                    focus_root: existing.worktree_root.join(&existing.focus_relative),
                    record: existing,
                });
            }
            // A desired-state record written before `git worktree add` survived a crash. Remove
            // the incomplete intent, prune any stale Git administration entry, and replay
            // creation below under the same session lock.
            self.remove_record(&existing)?;
            git_run(&existing.repository_root, &["worktree", "prune"])?;
        }
        let repository_root = repository_root.canonicalize().with_context(|| {
            format!(
                "failed to resolve repository root {}",
                repository_root.display()
            )
        })?;
        let focus_root = focus_root
            .canonicalize()
            .with_context(|| format!("failed to resolve focus root {}", focus_root.display()))?;
        let focus_relative = focus_root
            .strip_prefix(&repository_root)
            .with_context(|| {
                format!(
                    "focus root {} is outside repository {}",
                    focus_root.display(),
                    repository_root.display()
                )
            })?
            .to_path_buf();
        let project_id = short_sha(repository_root.to_string_lossy().as_bytes());
        let session_key = short_sha(session_id.as_bytes());
        let project_dir = self.state_root.join("workspaces").join(&project_id);
        let worktree_root = project_dir.join(&session_key);
        let record_path = project_dir.join(format!("{session_key}.json"));

        if worktree_root.exists() {
            let existing = load_record(&record_path)?
                .context("session worktree exists without a durable ownership record")?;
            validate_adoption(&existing, &repository_root, session_id, branch)?;
            let actual_root = git_capture(&worktree_root, &["rev-parse", "--show-toplevel"])?;
            let actual_root = PathBuf::from(actual_root.trim()).canonicalize()?;
            if actual_root != worktree_root.canonicalize()? {
                bail!(
                    "session worktree {} resolves to unexpected Git root {}",
                    worktree_root.display(),
                    actual_root.display()
                );
            }
            let actual_branch = git_capture(&worktree_root, &["branch", "--show-current"])?;
            if actual_branch.trim() != branch {
                bail!(
                    "session worktree {} is on branch '{}' rather than '{}'",
                    worktree_root.display(),
                    actual_branch.trim(),
                    branch
                );
            }
            let focus_root = worktree_root.join(&existing.focus_relative);
            return Ok(SessionWorkspace {
                record: existing,
                focus_root,
            });
        }

        std::fs::create_dir_all(&project_dir)
            .with_context(|| format!("failed to create {}", project_dir.display()))?;
        let branch_ref = format!("refs/heads/{branch}");
        let branch_exists = git_status(
            &repository_root,
            &["show-ref", "--verify", "--quiet", &branch_ref],
        )?;
        let mut args = vec!["worktree", "add"];
        let worktree_text = worktree_root.to_string_lossy().into_owned();
        let record = SessionWorkspaceRecord {
            version: SESSION_WORKSPACE_RECORD_VERSION,
            project_id,
            session_id: session_id.to_string(),
            repository_root,
            worktree_root: worktree_root.clone(),
            focus_relative: focus_relative.clone(),
            branch: branch.to_string(),
            strategy: WorkspaceStrategy::WorktreeBind,
            created_at_ms: now_millis(),
        };
        // Persist ownership intent before the external Git mutation so restart reconciliation can
        // distinguish an incomplete pb worktree from an unowned directory.
        save_record_atomic(&record_path, &record)?;
        if branch_exists {
            args.push(&worktree_text);
            args.push(branch);
        } else {
            args.push("-b");
            args.push(branch);
            args.push(&worktree_text);
            args.push("HEAD");
        }
        if let Err(error) = git_run(&record.repository_root, &args).with_context(|| {
            format!(
                "failed to create task worktree for branch '{branch}'; if the branch is checked out elsewhere, finish or remove that legacy worktree first"
            )
        }) {
            let _ = std::fs::remove_file(&record_path);
            return Err(error);
        }
        Ok(SessionWorkspace {
            focus_root: worktree_root.join(focus_relative),
            record,
        })
    }

    pub fn record_for_session(
        &self,
        repository_root: &Path,
        session_id: &str,
    ) -> Result<Option<SessionWorkspaceRecord>> {
        let repository_root = repository_root.canonicalize()?;
        let project_id = short_sha(repository_root.to_string_lossy().as_bytes());
        let session_key = short_sha(session_id.as_bytes());
        load_record(
            &self
                .state_root
                .join("workspaces")
                .join(project_id)
                .join(format!("{session_key}.json")),
        )
    }

    pub fn find_record_by_session(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionWorkspaceRecord>> {
        let root = self.state_root.join("workspaces");
        if !root.exists() {
            return Ok(None);
        }
        let file_name = format!("{}.json", short_sha(session_id.as_bytes()));
        for project in std::fs::read_dir(&root)
            .with_context(|| format!("failed to list {}", root.display()))?
        {
            let project = project?;
            if !project.file_type()?.is_dir() {
                continue;
            }
            let path = project.path().join(&file_name);
            if let Some(record) = load_record(&path)? {
                if record.session_id != session_id {
                    bail!("session workspace hash collision for '{session_id}'");
                }
                return Ok(Some(record));
            }
        }
        Ok(None)
    }

    pub fn remove(&self, record: &SessionWorkspaceRecord, force: bool) -> Result<bool> {
        let _operation = crate::state_lock::StateFileLock::acquire(
            self.state_root
                .join("workspace-locks")
                .join(format!("{}.lock", short_sha(record.session_id.as_bytes()))),
            Duration::from_secs(10),
        )?;
        if !record.worktree_root.exists() {
            self.remove_record(record)?;
            return Ok(true);
        }
        let dirty = !git_capture(&record.worktree_root, &["status", "--porcelain"])?
            .trim()
            .is_empty();
        if dirty && !force {
            return Ok(false);
        }
        let worktree_text = record.worktree_root.to_string_lossy().into_owned();
        let mut args = vec!["worktree", "remove"];
        if force {
            args.push("--force");
        }
        args.push(&worktree_text);
        git_run(&record.repository_root, &args)?;
        self.remove_record(record)?;
        Ok(true)
    }

    fn remove_record(&self, record: &SessionWorkspaceRecord) -> Result<()> {
        let path = self
            .state_root
            .join("workspaces")
            .join(&record.project_id)
            .join(format!("{}.json", short_sha(record.session_id.as_bytes())));
        if path.exists() {
            std::fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        }
        Ok(())
    }
}

pub fn default_state_root() -> Result<PathBuf> {
    #[cfg(test)]
    {
        static TEST_STATE_ROOT: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
        let root = TEST_STATE_ROOT.get_or_init(|| {
            tempfile::Builder::new()
                .prefix("pb-test-state-")
                .tempdir()
                .expect("create process-local pb test state root")
        });
        return Ok(root.path().to_path_buf());
    }
    #[cfg(not(test))]
    crate::config::UserConfig::load()?.effective_state_dir()
}

fn validate_adoption(
    record: &SessionWorkspaceRecord,
    repository_root: &Path,
    session_id: &str,
    branch: &str,
) -> Result<()> {
    if record.version != SESSION_WORKSPACE_RECORD_VERSION
        || record.repository_root != repository_root
        || record.session_id != session_id
        || record.branch != branch
        || record.strategy != WorkspaceStrategy::WorktreeBind
    {
        bail!("session worktree ownership record does not match the requested session");
    }
    Ok(())
}

fn load_record(path: &Path) -> Result<Option<SessionWorkspaceRecord>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read workspace record {}", path.display()))?;
    let record: SessionWorkspaceRecord = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse workspace record {}", path.display()))?;
    if record.version != SESSION_WORKSPACE_RECORD_VERSION {
        bail!(
            "unsupported session workspace record version {} in {}",
            record.version,
            path.display()
        );
    }
    Ok(Some(record))
}

fn save_record_atomic(path: &Path, record: &SessionWorkspaceRecord) -> Result<()> {
    let parent = path.parent().context("workspace record has no parent")?;
    std::fs::create_dir_all(parent)?;
    let temp = parent.join(format!(".workspace.{}.tmp", std::process::id()));
    std::fs::write(&temp, serde_json::to_vec_pretty(record)?)?;
    std::fs::rename(&temp, path)
        .with_context(|| format!("failed to replace workspace record {}", path.display()))
}

fn git_status(workdir: &Path, args: &[&str]) -> Result<bool> {
    let status = Command::new("git")
        .args(args)
        .current_dir(workdir)
        .status()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    Ok(status.success())
}

fn git_run(workdir: &Path, args: &[&str]) -> Result<()> {
    let output = Command::new("git")
        .args(args)
        .current_dir(workdir)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

fn git_capture(workdir: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(workdir)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn short_sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))[..16].to_string()
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn git(repo: &Path, args: &[&str]) {
        git_run(repo, args).unwrap();
    }

    fn repository() -> TempDir {
        let dir = TempDir::new().unwrap();
        git(dir.path(), &["init", "-b", "main"]);
        git(dir.path(), &["config", "user.name", "pb-test"]);
        git(dir.path(), &["config", "user.email", "pb@example.invalid"]);
        std::fs::create_dir_all(dir.path().join("client")).unwrap();
        std::fs::write(dir.path().join("client/file.txt"), "one\n").unwrap();
        git(dir.path(), &["add", "."]);
        git(dir.path(), &["commit", "-m", "test: initialize"]);
        dir
    }

    #[test]
    fn worktree_is_outside_original_checkout_and_is_adopted() {
        let repo = repository();
        let state = TempDir::new().unwrap();
        let manager = WorkspaceManager::new(state.path().to_path_buf());
        let first = manager
            .prepare(
                repo.path(),
                &repo.path().join("client"),
                "session-1",
                "task/session-1",
            )
            .unwrap();
        assert!(!first.record.worktree_root.starts_with(repo.path()));
        assert_eq!(first.focus_root, first.record.worktree_root.join("client"));
        assert_eq!(
            git_capture(repo.path(), &["branch", "--show-current"])
                .unwrap()
                .trim(),
            "main"
        );

        let adopted = manager
            .prepare(
                repo.path(),
                &repo.path().join("client"),
                "session-1",
                "task/session-1",
            )
            .unwrap();
        assert_eq!(adopted.record, first.record);
        assert!(manager.remove(&first.record, false).unwrap());
    }

    #[test]
    fn dirty_worktree_is_preserved_without_force() {
        let repo = repository();
        let state = TempDir::new().unwrap();
        let manager = WorkspaceManager::new(state.path().to_path_buf());
        let workspace = manager
            .prepare(repo.path(), repo.path(), "session-2", "task/session-2")
            .unwrap();
        std::fs::write(workspace.record.worktree_root.join("new.txt"), "dirty\n").unwrap();
        assert!(!manager.remove(&workspace.record, false).unwrap());
        assert!(workspace.record.worktree_root.exists());
        assert!(manager.remove(&workspace.record, true).unwrap());
    }

    #[test]
    fn desired_record_without_worktree_is_replayed_after_crash() {
        let repo = repository();
        let state = TempDir::new().unwrap();
        let manager = WorkspaceManager::new(state.path().to_path_buf());
        let repository_root = repo.path().canonicalize().unwrap();
        let project_id = short_sha(repository_root.to_string_lossy().as_bytes());
        let session_id = "session-crash";
        let session_key = short_sha(session_id.as_bytes());
        let project_dir = state.path().join("workspaces").join(&project_id);
        let record = SessionWorkspaceRecord {
            version: SESSION_WORKSPACE_RECORD_VERSION,
            project_id,
            session_id: session_id.to_string(),
            repository_root: repository_root.clone(),
            worktree_root: project_dir.join(&session_key),
            focus_relative: PathBuf::new(),
            branch: "task/session-crash".to_string(),
            strategy: WorkspaceStrategy::WorktreeBind,
            created_at_ms: now_millis(),
        };
        save_record_atomic(&project_dir.join(format!("{session_key}.json")), &record).unwrap();

        let recovered = manager
            .prepare(
                &repository_root,
                &repository_root,
                session_id,
                "task/session-crash",
            )
            .unwrap();
        assert!(recovered.record.worktree_root.exists());
        assert_eq!(recovered.record.session_id, session_id);
        assert!(manager.remove(&recovered.record, true).unwrap());
    }
}
