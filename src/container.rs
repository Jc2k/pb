use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

const KEEPALIVE_SCRIPT: &str = "trap 'exit 0' TERM INT; while :; do sleep 86400; done";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeKind {
    Apple,
    Docker,
    Podman,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeInfo {
    pub kind: RuntimeKind,
    pub binary: String,
    pub version: String,
    pub capabilities: RuntimeCapabilities,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeCapabilities {
    pub internal_networks: bool,
    pub named_volumes: bool,
    pub labels: bool,
    pub resource_limits: bool,
}

impl RuntimeCapabilities {
    fn production_baseline() -> Self {
        Self {
            internal_networks: true,
            named_volumes: true,
            labels: true,
            resource_limits: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerResources {
    pub cpus: u32,
    pub memory_mb: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerMount {
    pub source: String,
    pub target: String,
    pub read_only: bool,
}

impl ContainerMount {
    pub fn bind(source: &Path, target: impl Into<String>, read_only: bool) -> Self {
        Self {
            source: source.to_string_lossy().into_owned(),
            target: target.into(),
            read_only,
        }
    }

    pub fn volume(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            read_only: false,
        }
    }

    fn cli_value(&self) -> String {
        let suffix = if self.read_only { ":ro" } else { "" };
        format!("{}:{}{suffix}", self.source, self.target)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerLaunchSpec {
    pub name: String,
    pub image: String,
    pub workdir: String,
    pub mounts: Vec<ContainerMount>,
    pub labels: BTreeMap<String, String>,
    pub env: BTreeMap<String, String>,
    pub network: Option<String>,
    pub resources: ContainerResources,
    pub tmpfs: Vec<String>,
    pub read_only_root: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkSpec {
    pub name: String,
    pub labels: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VolumeSpec {
    pub name: String,
    pub labels: BTreeMap<String, String>,
}

/// Capability-oriented abstraction over Apple Container, Docker, and Podman CLIs.
/// All methods are synchronous and block until the underlying CLI completes.
pub trait ContainerRuntime: Send + Sync {
    /// Validate the runtime version and return the capabilities pb relies upon.
    fn info(&self) -> Result<RuntimeInfo>;

    /// Pull an image from a registry.
    fn pull(&self, image: &str) -> Result<()>;

    /// Build an image from a Dockerfile, tagging it with `tag`.
    fn build(&self, dockerfile: &Path, tag: &str) -> Result<()>;

    /// Return true when the runtime already has an image tag available locally.
    fn image_exists(&self, image: &str) -> Result<bool>;

    /// Return stable local image metadata used to scope executable cache volumes.
    fn image_fingerprint(&self, image: &str) -> Result<String>;

    /// Create a long-running container from an explicit, fully-owned launch specification.
    fn create(&self, spec: &ContainerLaunchSpec) -> Result<String>;

    /// Execute a shell command inside a running container.
    fn exec(&self, container_id: &str, cmd: &str) -> Result<String>;

    /// Forcibly remove a container (equivalent to `docker rm -f`).
    fn remove(&self, container_id: &str) -> Result<()>;

    /// Create a host-only internal network.
    fn create_internal_network(&self, spec: &NetworkSpec) -> Result<()>;

    /// Remove a network owned by this session.
    fn remove_network(&self, network: &str) -> Result<()>;

    /// Create a persistent named volume if it does not already exist.
    fn ensure_volume(&self, spec: &VolumeSpec) -> Result<()>;
}

/// A running container and its ephemeral network. Persistent cache volumes are deliberately not
/// removed here. Durable daemon reconciliation will eventually complement this last-resort guard.
pub struct ContainerHandle {
    pub runtime: Box<dyn ContainerRuntime>,
    pub container_id: String,
    pub network: Option<String>,
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
        if let Some(network) = &self.network {
            let _ = self.runtime.remove_network(network);
        }
    }
}

// ---------------------------------------------------------------------------
// Apple Container CLI  (https://github.com/apple/container)
// ---------------------------------------------------------------------------

/// Runtime backed by the `container` CLI shipped with apple/container on macOS.
pub struct AppleContainerRuntime;

impl ContainerRuntime for AppleContainerRuntime {
    fn info(&self) -> Result<RuntimeInfo> {
        let version_output = run_capture("container", &["--version".to_string()])?;
        let version = parse_version(&version_output).with_context(|| {
            format!("could not parse Apple container version: {version_output}")
        })?;
        if version < (1, 0, 0) {
            bail!(
                "Apple container CLI 1.0.0 or newer is required; found {}.{}.{}",
                version.0,
                version.1,
                version.2
            );
        }
        Ok(RuntimeInfo {
            kind: RuntimeKind::Apple,
            binary: "container".to_string(),
            version: format!("{}.{}.{}", version.0, version.1, version.2),
            capabilities: RuntimeCapabilities::production_baseline(),
        })
    }

    fn pull(&self, image: &str) -> Result<()> {
        run_silent(
            "container",
            &["image".to_string(), "pull".to_string(), image.to_string()],
        )
    }

    fn build(&self, dockerfile: &Path, tag: &str) -> Result<()> {
        let df = dockerfile.to_string_lossy();
        let ctx = dockerfile
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());
        run_silent(
            "container",
            &[
                "build".to_string(),
                "-t".to_string(),
                tag.to_string(),
                "-f".to_string(),
                df.into_owned(),
                ctx,
            ],
        )
    }

    fn image_exists(&self, image: &str) -> Result<bool> {
        run_status(
            "container",
            &[
                "image".to_string(),
                "inspect".to_string(),
                image.to_string(),
            ],
        )
    }

    fn image_fingerprint(&self, image: &str) -> Result<String> {
        run_capture(
            "container",
            &[
                "image".to_string(),
                "inspect".to_string(),
                image.to_string(),
            ],
        )
        .with_context(|| format!("failed to inspect Apple container image {image}"))
    }

    fn create(&self, spec: &ContainerLaunchSpec) -> Result<String> {
        run_silent("container", &apple_run_args(spec))?;
        Ok(spec.name.clone())
    }

    fn exec(&self, container_id: &str, cmd: &str) -> Result<String> {
        run_capture(
            "container",
            &[
                "exec".to_string(),
                container_id.to_string(),
                "sh".to_string(),
                "-lc".to_string(),
                cmd.to_string(),
            ],
        )
    }

    fn remove(&self, container_id: &str) -> Result<()> {
        run_silent(
            "container",
            &["rm".to_string(), "-f".to_string(), container_id.to_string()],
        )
    }

    fn create_internal_network(&self, spec: &NetworkSpec) -> Result<()> {
        run_silent("container", &apple_network_create_args(spec))
    }

    fn remove_network(&self, network: &str) -> Result<()> {
        run_silent(
            "container",
            &[
                "network".to_string(),
                "delete".to_string(),
                network.to_string(),
            ],
        )
    }

    fn ensure_volume(&self, spec: &VolumeSpec) -> Result<()> {
        if run_status(
            "container",
            &[
                "volume".to_string(),
                "inspect".to_string(),
                spec.name.clone(),
            ],
        )? {
            return Ok(());
        }
        match run_silent("container", &apple_volume_create_args(spec)) {
            Ok(()) => Ok(()),
            Err(_create_error)
                if run_status(
                    "container",
                    &[
                        "volume".to_string(),
                        "inspect".to_string(),
                        spec.name.clone(),
                    ],
                )? =>
            {
                Ok(())
            }
            Err(create_error) => Err(create_error),
        }
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
    fn info(&self) -> Result<RuntimeInfo> {
        let output = run_capture(&self.binary, &["--version".to_string()])?;
        let kind = if self.binary == "podman" {
            RuntimeKind::Podman
        } else {
            RuntimeKind::Docker
        };
        Ok(RuntimeInfo {
            kind,
            binary: self.binary.clone(),
            version: parse_version(&output)
                .map(|version| format!("{}.{}.{}", version.0, version.1, version.2))
                .unwrap_or(output),
            capabilities: RuntimeCapabilities::production_baseline(),
        })
    }

    fn pull(&self, image: &str) -> Result<()> {
        run_silent(&self.binary, &["pull".to_string(), image.to_string()])
    }

    fn build(&self, dockerfile: &Path, tag: &str) -> Result<()> {
        let df = dockerfile.to_string_lossy();
        let ctx = dockerfile
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());
        run_silent(
            &self.binary,
            &[
                "build".to_string(),
                "-t".to_string(),
                tag.to_string(),
                "-f".to_string(),
                df.into_owned(),
                ctx,
            ],
        )
    }

    fn image_exists(&self, image: &str) -> Result<bool> {
        run_status(
            &self.binary,
            &[
                "image".to_string(),
                "inspect".to_string(),
                image.to_string(),
            ],
        )
    }

    fn image_fingerprint(&self, image: &str) -> Result<String> {
        run_capture(
            &self.binary,
            &[
                "image".to_string(),
                "inspect".to_string(),
                image.to_string(),
            ],
        )
        .with_context(|| format!("failed to inspect {} image {image}", self.binary))
    }

    fn create(&self, spec: &ContainerLaunchSpec) -> Result<String> {
        run_silent(&self.binary, &oci_run_args(spec))?;
        Ok(spec.name.clone())
    }

    fn exec(&self, container_id: &str, cmd: &str) -> Result<String> {
        run_capture(
            &self.binary,
            &[
                "exec".to_string(),
                container_id.to_string(),
                "sh".to_string(),
                "-lc".to_string(),
                cmd.to_string(),
            ],
        )
    }

    fn remove(&self, container_id: &str) -> Result<()> {
        run_silent(
            &self.binary,
            &["rm".to_string(), "-f".to_string(), container_id.to_string()],
        )
    }

    fn create_internal_network(&self, spec: &NetworkSpec) -> Result<()> {
        run_silent(&self.binary, &oci_network_create_args(spec))
    }

    fn remove_network(&self, network: &str) -> Result<()> {
        run_silent(
            &self.binary,
            &["network".to_string(), "rm".to_string(), network.to_string()],
        )
    }

    fn ensure_volume(&self, spec: &VolumeSpec) -> Result<()> {
        if run_status(
            &self.binary,
            &[
                "volume".to_string(),
                "inspect".to_string(),
                spec.name.clone(),
            ],
        )? {
            return Ok(());
        }
        match run_silent(&self.binary, &oci_volume_create_args(spec)) {
            Ok(()) => Ok(()),
            Err(_create_error)
                if run_status(
                    &self.binary,
                    &[
                        "volume".to_string(),
                        "inspect".to_string(),
                        spec.name.clone(),
                    ],
                )? =>
            {
                Ok(())
            }
            Err(create_error) => Err(create_error),
        }
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

/// Return the preferred runtime command for container-backed LSP/MCP integrations.
pub fn preferred_runtime_binary() -> Option<String> {
    #[cfg(target_os = "macos")]
    if which_exists("container") && AppleContainerRuntime.info().is_ok() {
        return Some("container".to_string());
    }
    OciRuntime::detect().and_then(|runtime| runtime.info().ok().map(|info| info.binary))
}

/// Resolve an explicitly configured integration runtime or the preferred installed runtime.
pub fn resolve_runtime_binary(configured: Option<&str>) -> Result<String> {
    resolve_runtime_binary_with(configured, preferred_runtime_binary())
}

fn resolve_runtime_binary_with(
    configured: Option<&str>,
    detected: Option<String>,
) -> Result<String> {
    if let Some(configured) = configured.filter(|value| !value.trim().is_empty()) {
        return Ok(configured.to_string());
    }
    detected.context("no container runtime found; install apple/container, Docker, or Podman")
}

fn apple_run_args(spec: &ContainerLaunchSpec) -> Vec<String> {
    run_args(spec)
}

fn oci_run_args(spec: &ContainerLaunchSpec) -> Vec<String> {
    run_args(spec)
}

fn run_args(spec: &ContainerLaunchSpec) -> Vec<String> {
    let mut args = vec![
        "run".to_string(),
        "-d".to_string(),
        "--name".to_string(),
        spec.name.clone(),
    ];
    for (key, value) in &spec.labels {
        args.push("--label".to_string());
        args.push(format!("{key}={value}"));
    }
    for (key, value) in &spec.env {
        args.push("--env".to_string());
        args.push(format!("{key}={value}"));
    }
    for mount in &spec.mounts {
        args.push("--volume".to_string());
        args.push(mount.cli_value());
    }
    args.push("--workdir".to_string());
    args.push(spec.workdir.clone());
    if let Some(network) = &spec.network {
        args.push("--network".to_string());
        args.push(network.clone());
    }
    args.push("--cpus".to_string());
    args.push(spec.resources.cpus.to_string());
    args.push("--memory".to_string());
    args.push(format!("{}M", spec.resources.memory_mb));
    for target in &spec.tmpfs {
        args.push("--tmpfs".to_string());
        args.push(target.clone());
    }
    if spec.read_only_root {
        args.push("--read-only".to_string());
    }
    args.push("--entrypoint".to_string());
    args.push("/bin/sh".to_string());
    args.push(spec.image.clone());
    args.push("-c".to_string());
    args.push(KEEPALIVE_SCRIPT.to_string());
    args
}

fn apple_network_create_args(spec: &NetworkSpec) -> Vec<String> {
    let mut args = vec![
        "network".to_string(),
        "create".to_string(),
        "--internal".to_string(),
    ];
    append_labels(&mut args, &spec.labels);
    args.push(spec.name.clone());
    args
}

fn oci_network_create_args(spec: &NetworkSpec) -> Vec<String> {
    let mut args = vec![
        "network".to_string(),
        "create".to_string(),
        "--internal".to_string(),
    ];
    append_labels(&mut args, &spec.labels);
    args.push(spec.name.clone());
    args
}

fn apple_volume_create_args(spec: &VolumeSpec) -> Vec<String> {
    let mut args = vec!["volume".to_string(), "create".to_string()];
    append_labels(&mut args, &spec.labels);
    args.push(spec.name.clone());
    args
}

fn oci_volume_create_args(spec: &VolumeSpec) -> Vec<String> {
    apple_volume_create_args(spec)
}

fn append_labels(args: &mut Vec<String>, labels: &BTreeMap<String, String>) {
    for (key, value) in labels {
        args.push("--label".to_string());
        args.push(format!("{key}={value}"));
    }
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

/// Run a command and return whether it exited successfully.
fn run_status(binary: &str, args: &[String]) -> Result<bool> {
    let output = Command::new(binary)
        .args(args)
        .output()
        .with_context(|| format!("failed to spawn {binary}"))?;
    Ok(output.status.success())
}

/// Run a command, failing on non-zero exit with bounded subprocess output.
fn run_silent(binary: &str, args: &[String]) -> Result<()> {
    let output = Command::new(binary)
        .args(args)
        .output()
        .with_context(|| format!("failed to spawn {binary}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "{binary} exited with status {}: {}{}",
            output.status,
            stderr.trim(),
            if stdout.trim().is_empty() {
                String::new()
            } else {
                format!("\nstdout: {}", stdout.trim())
            }
        );
    }
    Ok(())
}

/// Run a command and return its trimmed stdout, failing on non-zero exit.
fn run_capture(binary: &str, args: &[String]) -> Result<String> {
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

fn parse_version(output: &str) -> Option<(u64, u64, u64)> {
    for token in output.split_whitespace() {
        let candidate = token.trim_matches(|ch: char| !ch.is_ascii_digit() && ch != '.');
        let mut parts = candidate.split('.');
        let (Some(major), Some(minor), Some(patch_text)) =
            (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let (Ok(major), Ok(minor)) = (major.parse(), minor.parse()) else {
            continue;
        };
        let Ok(patch) = patch_text
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
        else {
            continue;
        };
        return Some((major, minor, patch));
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct MockRuntime {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl MockRuntime {
        fn new(events: Arc<Mutex<Vec<String>>>) -> Self {
            Self { events }
        }

        fn record(&self, event: impl Into<String>) {
            self.events.lock().unwrap().push(event.into());
        }
    }

    impl ContainerRuntime for MockRuntime {
        fn info(&self) -> Result<RuntimeInfo> {
            Ok(RuntimeInfo {
                kind: RuntimeKind::Apple,
                binary: "mock".to_string(),
                version: "1.0.0".to_string(),
                capabilities: RuntimeCapabilities::production_baseline(),
            })
        }

        fn pull(&self, image: &str) -> Result<()> {
            self.record(format!("pull {image}"));
            Ok(())
        }

        fn build(&self, _dockerfile: &Path, tag: &str) -> Result<()> {
            self.record(format!("build {tag}"));
            Ok(())
        }

        fn image_exists(&self, image: &str) -> Result<bool> {
            self.record(format!("image_exists {image}"));
            Ok(false)
        }

        fn image_fingerprint(&self, image: &str) -> Result<String> {
            self.record(format!("image fingerprint {image}"));
            Ok(format!("fingerprint:{image}"))
        }

        fn create(&self, spec: &ContainerLaunchSpec) -> Result<String> {
            self.record(format!("create {}", spec.name));
            Ok("mock-container-id".to_string())
        }

        fn exec(&self, container_id: &str, cmd: &str) -> Result<String> {
            self.record(format!("exec {container_id} {cmd}"));
            Ok("output".to_string())
        }

        fn remove(&self, container_id: &str) -> Result<()> {
            self.record(format!("remove {container_id}"));
            Ok(())
        }

        fn create_internal_network(&self, spec: &NetworkSpec) -> Result<()> {
            self.record(format!("network create {}", spec.name));
            Ok(())
        }

        fn remove_network(&self, network: &str) -> Result<()> {
            self.record(format!("network remove {network}"));
            Ok(())
        }

        fn ensure_volume(&self, spec: &VolumeSpec) -> Result<()> {
            self.record(format!("volume ensure {}", spec.name));
            Ok(())
        }
    }

    #[test]
    fn mock_runtime_pull() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let rt = MockRuntime::new(Arc::clone(&events));
        rt.pull("my-image:latest").unwrap();
        assert_eq!(*events.lock().unwrap(), vec!["pull my-image:latest"]);
    }

    #[test]
    fn mock_runtime_exec() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let rt = MockRuntime::new(Arc::clone(&events));
        rt.exec("abc123", "echo hi").unwrap();
        assert_eq!(*events.lock().unwrap(), vec!["exec abc123 echo hi"]);
    }

    #[test]
    fn container_handle_exec_delegates() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let handle = ContainerHandle {
            runtime: Box::new(MockRuntime::new(Arc::clone(&events))),
            container_id: "cid1".to_string(),
            network: None,
        };
        handle.exec("ls /workspace").unwrap();
        assert_eq!(*events.lock().unwrap(), vec!["exec cid1 ls /workspace"]);
        std::mem::forget(handle);
    }

    #[test]
    fn container_handle_drop_removes_container_then_network() {
        let events = Arc::new(Mutex::new(Vec::new()));
        {
            let _handle = ContainerHandle {
                runtime: Box::new(MockRuntime::new(Arc::clone(&events))),
                container_id: "drop-me".to_string(),
                network: Some("network-me".to_string()),
            };
        }

        assert_eq!(
            *events.lock().unwrap(),
            vec!["remove drop-me", "network remove network-me"]
        );
    }

    #[test]
    fn parses_versions_after_product_names() {
        assert_eq!(
            parse_version("container CLI version 1.0.0 (build: release)"),
            Some((1, 0, 0))
        );
        assert_eq!(
            parse_version("Docker version 28.3.2, build abc"),
            Some((28, 3, 2))
        );
        assert_eq!(parse_version("not a version"), None);
    }

    #[test]
    fn integration_runtime_prefers_explicit_then_detected_binary() {
        assert_eq!(
            resolve_runtime_binary_with(Some("podman"), Some("container".to_string())).unwrap(),
            "podman"
        );
        assert_eq!(
            resolve_runtime_binary_with(None, Some("container".to_string())).unwrap(),
            "container"
        );
        assert!(resolve_runtime_binary_with(None, None).is_err());
    }

    #[test]
    fn apple_run_arguments_encode_the_owned_launch_spec() {
        let mut labels = BTreeMap::new();
        labels.insert("pb.role".to_string(), "agent".to_string());
        labels.insert("pb.session".to_string(), "session-1".to_string());
        let mut env = BTreeMap::new();
        env.insert("CI".to_string(), "true".to_string());
        let spec = ContainerLaunchSpec {
            name: "pb-agent-session-1".to_string(),
            image: "rust:latest".to_string(),
            workdir: "/workspace".to_string(),
            mounts: vec![
                ContainerMount::bind(Path::new("/tmp/repo"), "/workspace", false),
                ContainerMount::volume("pb-cargo-cache", "/usr/local/cargo"),
            ],
            labels,
            env,
            network: Some("pb-session-1".to_string()),
            resources: ContainerResources {
                cpus: 4,
                memory_mb: 4096,
            },
            tmpfs: vec!["/tmp".to_string()],
            read_only_root: true,
        };

        assert_eq!(
            apple_run_args(&spec),
            vec![
                "run",
                "-d",
                "--name",
                "pb-agent-session-1",
                "--label",
                "pb.role=agent",
                "--label",
                "pb.session=session-1",
                "--env",
                "CI=true",
                "--volume",
                "/tmp/repo:/workspace",
                "--volume",
                "pb-cargo-cache:/usr/local/cargo",
                "--workdir",
                "/workspace",
                "--network",
                "pb-session-1",
                "--cpus",
                "4",
                "--memory",
                "4096M",
                "--tmpfs",
                "/tmp",
                "--read-only",
                "--entrypoint",
                "/bin/sh",
                "rust:latest",
                "-c",
                KEEPALIVE_SCRIPT,
            ]
        );
    }

    #[test]
    fn apple_network_and_volume_arguments_include_ownership_labels() {
        let labels = BTreeMap::from([
            ("pb.owner".to_string(), "pb".to_string()),
            ("pb.session".to_string(), "s1".to_string()),
        ]);
        assert_eq!(
            apple_network_create_args(&NetworkSpec {
                name: "pb-network".to_string(),
                labels: labels.clone(),
            }),
            vec![
                "network",
                "create",
                "--internal",
                "--label",
                "pb.owner=pb",
                "--label",
                "pb.session=s1",
                "pb-network",
            ]
        );
        assert_eq!(
            apple_volume_create_args(&VolumeSpec {
                name: "pb-cache".to_string(),
                labels,
            }),
            vec![
                "volume",
                "create",
                "--label",
                "pb.owner=pb",
                "--label",
                "pb.session=s1",
                "pb-cache",
            ]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires an installed and running Apple container runtime plus registry access"]
    fn apple_runtime_conformance() -> Result<()> {
        let runtime: Box<dyn ContainerRuntime> = Box::new(AppleContainerRuntime);
        let info = runtime.info()?;
        assert_eq!(info.kind, RuntimeKind::Apple);
        runtime.pull("docker.io/library/alpine:3.20")?;

        let workspace = tempfile::TempDir::new()?;
        let suffix = format!("{}-{}", std::process::id(), workspace.path().display());
        let name = format!("pb-conformance-{}", sanitize_test_name(&suffix));
        let network = format!("{name}-network");
        let labels = BTreeMap::from([("pb.owner".to_string(), "pb-test".to_string())]);
        runtime.create_internal_network(&NetworkSpec {
            name: network.clone(),
            labels: labels.clone(),
        })?;
        let spec = ContainerLaunchSpec {
            name: name.clone(),
            image: "docker.io/library/alpine:3.20".to_string(),
            workdir: "/workspace".to_string(),
            mounts: vec![ContainerMount::bind(workspace.path(), "/workspace", false)],
            labels,
            env: BTreeMap::new(),
            network: Some(network.clone()),
            resources: ContainerResources {
                cpus: 1,
                memory_mb: 512,
            },
            tmpfs: vec!["/tmp".to_string()],
            read_only_root: true,
        };
        let container_id = runtime.create(&spec)?;
        let handle = ContainerHandle {
            runtime,
            container_id,
            network: Some(network),
        };
        handle.exec("test -d /workspace && test ! -e /definitely-not-present")?;
        handle.exec("touch /tmp/pb-conformance && rm /tmp/pb-conformance")?;
        assert!(
            handle.exec("touch /pb-conformance-must-not-exist").is_err(),
            "Apple read-only root unexpectedly accepted a write"
        );
        assert!(
            handle
                .exec("wget -q -T 3 -O /dev/null http://example.com")
                .is_err(),
            "Apple internal network unexpectedly allowed external HTTP egress"
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn sanitize_test_name(input: &str) -> String {
        input
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() {
                    ch.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .take(28)
            .collect()
    }
}
