use anyhow::{Context, Result, bail};
#[cfg(unix)]
use std::fs::File;
use std::fs::OpenOptions;
#[cfg(not(unix))]
use std::path::Path;
use std::path::PathBuf;
#[cfg(not(unix))]
use std::time::SystemTime;
use std::time::{Duration, Instant};

/// Cross-process state transition lock. Unix uses `flock`, which the kernel releases when a
/// process exits or crashes. The create-new fallback is retained for non-Unix builds.
pub(crate) struct StateFileLock {
    #[cfg(unix)]
    _file: File,
    #[cfg(not(unix))]
    path: PathBuf,
}

impl StateFileLock {
    pub(crate) fn acquire(path: PathBuf, timeout: Duration) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;

            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
                .with_context(|| format!("failed to open state lock {}", path.display()))?;
            let started = Instant::now();
            loop {
                let result =
                    unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
                if result == 0 {
                    return Ok(Self { _file: file });
                }
                let error = std::io::Error::last_os_error();
                let code = error.raw_os_error();
                if code != Some(libc::EWOULDBLOCK) && code != Some(libc::EAGAIN) {
                    return Err(error)
                        .with_context(|| format!("failed to lock {}", path.display()));
                }
                if started.elapsed() >= timeout {
                    bail!("timed out waiting for state lock {}", path.display());
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        }
        #[cfg(not(unix))]
        {
            let started = Instant::now();
            loop {
                match OpenOptions::new().write(true).create_new(true).open(&path) {
                    Ok(_) => return Ok(Self { path }),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        if fallback_lock_is_abandoned(&path) {
                            let _ = std::fs::remove_file(&path);
                            continue;
                        }
                        if started.elapsed() >= timeout {
                            bail!("timed out waiting for state lock {}", path.display());
                        }
                        std::thread::sleep(Duration::from_millis(25));
                    }
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("failed to create state lock {}", path.display())
                        });
                    }
                }
            }
        }
    }
}

#[cfg(unix)]
impl Drop for StateFileLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        let _ = unsafe { libc::flock(self._file.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[cfg(not(unix))]
impl Drop for StateFileLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(not(unix))]
fn fallback_lock_is_abandoned(path: &Path) -> bool {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age >= Duration::from_secs(60))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn state_lock_excludes_another_holder_and_releases_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("resource.lock");
        let first = StateFileLock::acquire(path.clone(), Duration::from_secs(1)).unwrap();
        let (sender, receiver) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let lock = StateFileLock::acquire(path, Duration::from_secs(1)).unwrap();
            sender.send(()).unwrap();
            drop(lock);
        });
        assert!(receiver.recv_timeout(Duration::from_millis(75)).is_err());
        drop(first);
        receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        thread.join().unwrap();
    }
}
