use anyhow::{Context, Result, bail};
use std::path::Path;
use std::process::Command;

/// Abstraction over container runtimes (Apple Container CLI, Docker, Podman, …).
///
/// All methods are synchronous; they block until the underlying CLI completes.
pub trait ContainerRuntime: Send + Sync {
    /// Pull an image from a registry.
    fn pull(&self, image: &str) -> Result<()>;

    /// Build an image from a Dockerfile, tagging it with `tag`.
    fn build(&self, dockerfile: &Path, tag: &str) -> Result<()>;

    /// Create a long-running container from `image`, mounting `workspace` at `/workspace`.
    /// Returns the container ID / name produced by the runtime.
    fn create(&self, image: &str, workspace: &Path) -> Result<String>;

    /// Execute a shell command inside a running container.
    fn exec(&self, container_id: &str, cmd: &str) -> Result<String>;

    /// Forcibly remove a container (equivalent to `docker rm -f`).
    fn remove(&self, container_id: &str) -> Result<()>;
}

/// A running container paired with its runtime.  Cleans itself up on drop.
pub struct ContainerHandle {
    pub runtime: Box<dyn ContainerRuntime>,
    pub container_id: String,
}

impl ContainerHandle {
    /// Execute a shell command inside this container.
    pub fn exec(&self, cmd: &str) -> Result<String> {
        self.runtime.exec(&self.container_id, cmd)
    }
}

impl Drop for ContainerHandle {
    fn drop(&mut self) {
        let _ = self.runtime.remove(&self.container_id);
    }
}

// ---------------------------------------------------------------------------
// Apple Container CLI  (https://github.com/apple/container)
// ---------------------------------------------------------------------------

/// Runtime backed by the `container` CLI shipped with apple/container on macOS.
pub struct AppleContainerRuntime;

impl ContainerRuntime for AppleContainerRuntime {
    fn pull(&self, image: &str) -> Result<()> {
        run_silent("container", &["pull", image])
    }

    fn build(&self, dockerfile: &Path, tag: &str) -> Result<()> {
        let df = dockerfile.to_string_lossy();
        let ctx = dockerfile
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());
        run_silent("container", &["build", "-t", tag, "-f", &df, &ctx])
    }

    fn create(&self, image: &str, workspace: &Path) -> Result<String> {
        let mount = format!("{}:/workspace", workspace.to_string_lossy());
        run_capture("container", &["run", "-d", "-v", &mount, image, "sleep", "infinity"])
    }

    fn exec(&self, container_id: &str, cmd: &str) -> Result<String> {
        run_capture("container", &["exec", container_id, "sh", "-c", cmd])
    }

    fn remove(&self, container_id: &str) -> Result<()> {
        run_silent("container", &["rm", "-f", container_id])
    }
}

// ---------------------------------------------------------------------------
// OCI-compatible runtime (Docker / Podman)
// ---------------------------------------------------------------------------

/// Runtime backed by `docker` or `podman`, auto-detected from `$PATH`.
pub struct OciRuntime {
    pub binary: String,
}

impl OciRuntime {
    /// Try to find `docker` or `podman` on `$PATH`.  Returns `None` when neither is found.
    pub fn detect() -> Option<Self> {
        for candidate in &["docker", "podman"] {
            if which_exists(candidate) {
                return Some(OciRuntime {
                    binary: candidate.to_string(),
                });
            }
        }
        None
    }
}

impl ContainerRuntime for OciRuntime {
    fn pull(&self, image: &str) -> Result<()> {
        run_silent(&self.binary, &["pull", image])
    }

    fn build(&self, dockerfile: &Path, tag: &str) -> Result<()> {
        let df = dockerfile.to_string_lossy();
        let ctx = dockerfile
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());
        run_silent(&self.binary, &["build", "-t", tag, "-f", &df, &ctx])
    }

    fn create(&self, image: &str, workspace: &Path) -> Result<String> {
        let mount = format!("{}:/workspace", workspace.to_string_lossy());
        run_capture(
            &self.binary,
            &["run", "-d", "-v", &mount, image, "sleep", "infinity"],
        )
    }

    fn exec(&self, container_id: &str, cmd: &str) -> Result<String> {
        run_capture(&self.binary, &["exec", container_id, "sh", "-c", cmd])
    }

    fn remove(&self, container_id: &str) -> Result<()> {
        run_silent(&self.binary, &["rm", "-f", container_id])
    }
}

// ---------------------------------------------------------------------------
// Runtime detection
// ---------------------------------------------------------------------------

/// Return the best container runtime available on this machine, or `None` when
/// no supported runtime is found.
///
/// On macOS the Apple Container CLI (`container`) is preferred when installed.
/// On all platforms `docker` and then `podman` are tried as fallbacks.
pub fn detect_runtime() -> Option<Box<dyn ContainerRuntime>> {
    // Prefer the Apple Container CLI on macOS.
    #[cfg(target_os = "macos")]
    if which_exists("container") {
        return Some(Box::new(AppleContainerRuntime));
    }

    if let Some(oci) = OciRuntime::detect() {
        return Some(Box::new(oci));
    }

    None
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` when `binary` can be located on `$PATH`.
fn which_exists(binary: &str) -> bool {
    Command::new("which")
        .arg(binary)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run a command, inheriting stdio, failing on non-zero exit.
fn run_silent(binary: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(binary)
        .args(args)
        .status()
        .with_context(|| format!("failed to spawn {binary}"))?;
    if !status.success() {
        bail!("{binary} exited with status {status}");
    }
    Ok(())
}

/// Run a command and return its trimmed stdout, failing on non-zero exit.
fn run_capture(binary: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(binary)
        .args(args)
        .output()
        .with_context(|| format!("failed to spawn {binary}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("{binary} failed: {stderr}");
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRuntime {
        commands: std::sync::Mutex<Vec<String>>,
    }

    impl MockRuntime {
        fn new() -> Self {
            MockRuntime {
                commands: std::sync::Mutex::new(Vec::new()),
            }
        }
        fn recorded(&self) -> Vec<String> {
            self.commands.lock().unwrap().clone()
        }
    }

    impl ContainerRuntime for MockRuntime {
        fn pull(&self, image: &str) -> Result<()> {
            self.commands.lock().unwrap().push(format!("pull {image}"));
            Ok(())
        }
        fn build(&self, _dockerfile: &Path, tag: &str) -> Result<()> {
            self.commands.lock().unwrap().push(format!("build {tag}"));
            Ok(())
        }
        fn create(&self, image: &str, _workspace: &Path) -> Result<String> {
            self.commands.lock().unwrap().push(format!("create {image}"));
            Ok("mock-container-id".to_string())
        }
        fn exec(&self, container_id: &str, cmd: &str) -> Result<String> {
            self.commands
                .lock()
                .unwrap()
                .push(format!("exec {container_id} {cmd}"));
            Ok("output".to_string())
        }
        fn remove(&self, container_id: &str) -> Result<()> {
            self.commands
                .lock()
                .unwrap()
                .push(format!("remove {container_id}"));
            Ok(())
        }
    }

    #[test]
    fn mock_runtime_pull() {
        let rt = MockRuntime::new();
        rt.pull("my-image:latest").unwrap();
        assert_eq!(rt.recorded(), vec!["pull my-image:latest"]);
    }

    #[test]
    fn mock_runtime_exec() {
        let rt = MockRuntime::new();
        rt.exec("abc123", "echo hi").unwrap();
        assert_eq!(rt.recorded(), vec!["exec abc123 echo hi"]);
    }

    #[test]
    fn container_handle_exec_delegates() {
        let rt = Box::new(MockRuntime::new());
        // Access the inner mock through a raw pointer so we can inspect after move.
        let mock_ptr = rt.as_ref() as *const MockRuntime;
        let handle = ContainerHandle {
            runtime: rt,
            container_id: "cid1".to_string(),
        };
        handle.exec("ls /workspace").unwrap();
        // SAFETY: the Box is still alive inside handle.
        let recorded = unsafe { &*mock_ptr }.recorded();
        assert_eq!(recorded, vec!["exec cid1 ls /workspace"]);
    }

    #[test]
    fn container_handle_drop_removes_container() {
        use std::sync::{Arc, Mutex};
        let removed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let removed_clone = Arc::clone(&removed);

        struct TrackingRuntime(Arc<Mutex<Vec<String>>>);
        impl ContainerRuntime for TrackingRuntime {
            fn pull(&self, _: &str) -> Result<()> { Ok(()) }
            fn build(&self, _: &Path, _: &str) -> Result<()> { Ok(()) }
            fn create(&self, _: &str, _: &Path) -> Result<String> { Ok("x".into()) }
            fn exec(&self, _: &str, _: &str) -> Result<String> { Ok(String::new()) }
            fn remove(&self, id: &str) -> Result<()> {
                self.0.lock().unwrap().push(id.to_string());
                Ok(())
            }
        }

        {
            let _handle = ContainerHandle {
                runtime: Box::new(TrackingRuntime(removed_clone)),
                container_id: "drop-me".to_string(),
            };
        } // handle dropped here

        assert_eq!(*removed.lock().unwrap(), vec!["drop-me"]);
    }
}
