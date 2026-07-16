use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

const KEEPALIVE_SCRIPT: &str = "trap 'exit 0' TERM INT; while :; do sleep 86400; done";
const MAX_RUNTIME_STDOUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_RUNTIME_STDERR_BYTES: usize = 64 * 1024;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManagedResource {
    pub name: String,
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceLaunchSpec {
    pub name: String,
    pub image: String,
    pub args: Vec<String>,
    pub workdir: String,
    pub mounts: Vec<ContainerMount>,
    pub labels: BTreeMap<String, String>,
    /// Values are injected into the runtime client environment and only keys appear in argv.
    pub env: BTreeMap<String, String>,
    pub network: Option<String>,
    pub resources: ContainerResources,
    pub tmpfs: Vec<String>,
    pub read_only_root: bool,
}

/// A streaming host-side client process used for long-lived container exec protocols.
pub struct ManagedProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
    command: String,
}

impl ManagedProcess {
    pub(crate) fn spawn(binary: &str, args: &[String]) -> Result<Self> {
        Self::spawn_with_env(binary, args, &BTreeMap::new())
    }

    pub(crate) fn spawn_with_env(
        binary: &str,
        args: &[String],
        env: &BTreeMap<String, String>,
    ) -> Result<Self> {
        validate_process_env(env)?;
        let mut child = Command::new(binary)
            .args(args)
            .envs(env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("failed to spawn managed process {binary}"))?;
        Ok(Self {
            stdin: child.stdin.take(),
            stdout: child.stdout.take(),
            stderr: child.stderr.take(),
            child,
            command: format!("{binary} {}", args.join(" ")),
        })
    }

    pub fn id(&self) -> u32 {
        self.child.id()
    }

    pub fn take_stdin(&mut self) -> Result<ChildStdin> {
        self.stdin
            .take()
            .with_context(|| format!("{} stdin was already taken", self.command))
    }

    pub fn take_stdout(&mut self) -> Result<ChildStdout> {
        self.stdout
            .take()
            .with_context(|| format!("{} stdout was already taken", self.command))
    }

    pub fn take_stderr(&mut self) -> Result<ChildStderr> {
        self.stderr
            .take()
            .with_context(|| format!("{} stderr was already taken", self.command))
    }

    pub fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        self.child
            .try_wait()
            .with_context(|| format!("failed to poll {}", self.command))
    }

    pub fn wait(&mut self) -> Result<ExitStatus> {
        self.child
            .wait()
            .with_context(|| format!("failed to wait for {}", self.command))
    }

    pub fn shutdown(&mut self, timeout: Duration) -> Result<ExitStatus> {
        self.stdin.take();
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.try_wait()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                self.child
                    .kill()
                    .with_context(|| format!("failed to kill {}", self.command))?;
                return self.wait();
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

pub(crate) fn validate_process_env(env: &BTreeMap<String, String>) -> Result<()> {
    for (key, value) in env {
        let mut characters = key.chars();
        if !characters
            .next()
            .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
            || !characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            bail!("managed process environment variable name '{key}' is invalid");
        }
        if value.contains('\0') {
            bail!("managed process environment variable '{key}' contains a NUL byte");
        }
    }
    Ok(())
}

/// A named service container plus its attached stdio client. Dropping it always removes the
/// container; no service relies on `--rm` or an untracked host child.
pub struct ManagedServiceProcess {
    process: ManagedProcess,
    runtime: Box<dyn ContainerRuntime>,
    container_name: String,
}

impl ManagedServiceProcess {
    pub fn take_stdin(&mut self) -> Result<ChildStdin> {
        self.process.take_stdin()
    }

    pub fn take_stdout(&mut self) -> Result<ChildStdout> {
        self.process.take_stdout()
    }

    pub fn take_stderr(&mut self) -> Result<ChildStderr> {
        self.process.take_stderr()
    }

    pub fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        self.process.try_wait()
    }

    pub fn shutdown(&mut self, timeout: Duration) -> Result<()> {
        let process_result = self.process.shutdown(timeout).map(|_| ());
        let remove_result = self.runtime.remove(&self.container_name);
        process_result.and(remove_result)
    }

    pub fn container_name(&self) -> &str {
        &self.container_name
    }
}

impl Drop for ManagedServiceProcess {
    fn drop(&mut self) {
        let _ = self.process.shutdown(Duration::from_secs(2));
        let _ = self.runtime.remove(&self.container_name);
    }
}

pub fn spawn_managed_service(
    runtime_binary: &str,
    spec: &ServiceLaunchSpec,
) -> Result<ManagedServiceProcess> {
    let runtime = runtime_for_binary(runtime_binary)?;
    if runtime.container_exists(&spec.name)? {
        let owned = runtime
            .list_managed_containers()?
            .iter()
            .any(|resource| managed_resource_matches(resource, &spec.name, &spec.labels));
        if !owned {
            bail!(
                "refusing to replace existing container '{}' without pb ownership labels",
                spec.name
            );
        }
        runtime.remove(&spec.name)?;
    }
    let args = service_run_args(spec);
    let mut process = ManagedProcess::spawn_with_env(runtime_binary, &args, &spec.env)?;
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if runtime.container_exists(&spec.name)? {
            let owned = runtime
                .list_managed_containers()?
                .iter()
                .any(|resource| managed_resource_matches(resource, &spec.name, &spec.labels));
            if !owned {
                let _ = process.shutdown(Duration::from_secs(1));
                bail!(
                    "managed service '{}' started without the requested ownership labels",
                    spec.name
                );
            }
            break;
        }
        if let Some(status) = process.try_wait()? {
            bail!(
                "managed service '{}' exited with status {status} before its container was created",
                spec.name
            );
        }
        if Instant::now() >= deadline {
            let _ = process.shutdown(Duration::from_secs(1));
            let _ = runtime.remove(&spec.name);
            bail!(
                "timed out waiting for managed service '{}' to become observable",
                spec.name
            );
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    Ok(ManagedServiceProcess {
        process,
        runtime,
        container_name: spec.name.clone(),
    })
}

fn managed_resource_matches(
    resource: &ManagedResource,
    expected_name: &str,
    expected_labels: &BTreeMap<String, String>,
) -> bool {
    if resource.name != expected_name {
        return false;
    }
    ["dev.pb.managed", "dev.pb.project", "dev.pb.session"]
        .into_iter()
        .all(|key| {
            expected_labels
                .get(key)
                .is_none_or(|expected| resource.labels.get(key) == Some(expected))
        })
}

impl Drop for ManagedProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
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

    /// Spawn a streaming command inside a running container.
    fn spawn_exec(&self, _container_id: &str, _argv: &[String]) -> Result<ManagedProcess> {
        bail!("container runtime does not implement streaming exec")
    }

    fn spawn_exec_with_env(
        &self,
        container_id: &str,
        argv: &[String],
        env: &BTreeMap<String, String>,
    ) -> Result<ManagedProcess> {
        if !env.is_empty() {
            bail!("container runtime does not implement secure streaming exec environment");
        }
        self.spawn_exec(container_id, argv)
    }

    /// Gracefully stop a running container before deletion.
    fn stop(&self, container_id: &str, _timeout: Duration) -> Result<()> {
        self.remove(container_id)
    }

    /// Return whether a named container exists in any state.
    fn container_exists(&self, _container_id: &str) -> Result<bool> {
        Ok(false)
    }

    /// List containers owned by pb according to runtime labels.
    fn list_managed_containers(&self) -> Result<Vec<ManagedResource>> {
        Ok(Vec::new())
    }

    /// List networks owned by pb according to runtime labels.
    fn list_managed_networks(&self) -> Result<Vec<ManagedResource>> {
        Ok(Vec::new())
    }

    /// Forcibly remove a container (equivalent to `docker rm -f`).
    fn remove(&self, container_id: &str) -> Result<()>;

    /// Create a host-only internal network.
    fn create_internal_network(&self, spec: &NetworkSpec) -> Result<()>;

    /// Remove a network owned by this session.
    fn remove_network(&self, network: &str) -> Result<()>;

    /// Create a persistent named volume if it does not already exist.
    fn ensure_volume(&self, spec: &VolumeSpec) -> Result<()>;

    /// Remove an unattached persistent named volume.
    fn remove_volume(&self, _volume: &str) -> Result<()> {
        bail!("container runtime does not implement volume deletion")
    }
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

    pub fn spawn_exec(&self, argv: &[String]) -> Result<ManagedProcess> {
        self.runtime.spawn_exec(&self.container_id, argv)
    }

    pub fn spawn_exec_with_env(
        &self,
        argv: &[String],
        env: &BTreeMap<String, String>,
    ) -> Result<ManagedProcess> {
        self.runtime
            .spawn_exec_with_env(&self.container_id, argv, env)
    }

    pub fn shutdown(&self, timeout: Duration) -> Result<()> {
        self.runtime.stop(&self.container_id, timeout)
    }

    /// Force removal and report cleanup failures. Successfully removed identifiers are cleared so
    /// the Drop guard does not issue duplicate runtime operations.
    pub fn cleanup(&mut self) -> Result<()> {
        let mut first_error = None;
        if !self.container_id.is_empty() {
            match self.runtime.remove(&self.container_id) {
                Ok(()) => self.container_id.clear(),
                Err(error) => first_error = Some(error),
            }
        }
        if let Some(network) = self.network.clone() {
            match self.runtime.remove_network(&network) {
                Ok(()) => self.network = None,
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl Drop for ContainerHandle {
    fn drop(&mut self) {
        if !self.container_id.is_empty() {
            let _ = self.runtime.remove(&self.container_id);
        }
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
        run_silent_with_env("container", &apple_run_args(spec), &spec.env)?;
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

    fn spawn_exec(&self, container_id: &str, argv: &[String]) -> Result<ManagedProcess> {
        let mut args = vec![
            "exec".to_string(),
            "-i".to_string(),
            container_id.to_string(),
        ];
        args.extend(argv.iter().cloned());
        ManagedProcess::spawn("container", &args)
    }

    fn spawn_exec_with_env(
        &self,
        container_id: &str,
        argv: &[String],
        env: &BTreeMap<String, String>,
    ) -> Result<ManagedProcess> {
        let mut args = vec!["exec".to_string(), "-i".to_string()];
        for key in env.keys() {
            args.push("--env".to_string());
            args.push(key.clone());
        }
        args.push(container_id.to_string());
        args.extend(argv.iter().cloned());
        ManagedProcess::spawn_with_env("container", &args, env)
    }

    fn stop(&self, container_id: &str, timeout: Duration) -> Result<()> {
        run_silent(
            "container",
            &[
                "stop".to_string(),
                "--time".to_string(),
                timeout.as_secs().to_string(),
                container_id.to_string(),
            ],
        )
    }

    fn container_exists(&self, container_id: &str) -> Result<bool> {
        run_status(
            "container",
            &["inspect".to_string(), container_id.to_string()],
        )
    }

    fn list_managed_containers(&self) -> Result<Vec<ManagedResource>> {
        let output = run_capture(
            "container",
            &[
                "list".to_string(),
                "--all".to_string(),
                "--format".to_string(),
                "json".to_string(),
            ],
        )?;
        Ok(parse_apple_managed_resources(&output, "configuration"))
    }

    fn list_managed_networks(&self) -> Result<Vec<ManagedResource>> {
        let output = run_capture(
            "container",
            &[
                "network".to_string(),
                "list".to_string(),
                "--format".to_string(),
                "json".to_string(),
            ],
        )?;
        Ok(parse_apple_managed_resources(&output, "configuration"))
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
            validate_volume_labels(
                spec,
                &apple_volume_labels(&spec.name)?.with_context(|| {
                    format!(
                        "Apple volume {} has no readable ownership labels",
                        spec.name
                    )
                })?,
            )?;
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
                validate_volume_labels(
                    spec,
                    &apple_volume_labels(&spec.name)?.with_context(|| {
                        format!(
                            "Apple volume {} has no readable ownership labels",
                            spec.name
                        )
                    })?,
                )
            }
            Err(create_error) => Err(create_error),
        }
    }

    fn remove_volume(&self, volume: &str) -> Result<()> {
        run_silent(
            "container",
            &[
                "volume".to_string(),
                "delete".to_string(),
                volume.to_string(),
            ],
        )
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
        run_silent_with_env(&self.binary, &oci_run_args(spec), &spec.env)?;
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

    fn spawn_exec(&self, container_id: &str, argv: &[String]) -> Result<ManagedProcess> {
        let mut args = vec![
            "exec".to_string(),
            "-i".to_string(),
            container_id.to_string(),
        ];
        args.extend(argv.iter().cloned());
        ManagedProcess::spawn(&self.binary, &args)
    }

    fn spawn_exec_with_env(
        &self,
        container_id: &str,
        argv: &[String],
        env: &BTreeMap<String, String>,
    ) -> Result<ManagedProcess> {
        let mut args = vec!["exec".to_string(), "-i".to_string()];
        for key in env.keys() {
            args.push("--env".to_string());
            args.push(key.clone());
        }
        args.push(container_id.to_string());
        args.extend(argv.iter().cloned());
        ManagedProcess::spawn_with_env(&self.binary, &args, env)
    }

    fn stop(&self, container_id: &str, timeout: Duration) -> Result<()> {
        run_silent(
            &self.binary,
            &[
                "stop".to_string(),
                "--time".to_string(),
                timeout.as_secs().to_string(),
                container_id.to_string(),
            ],
        )
    }

    fn container_exists(&self, container_id: &str) -> Result<bool> {
        run_status(
            &self.binary,
            &["inspect".to_string(), container_id.to_string()],
        )
    }

    fn list_managed_containers(&self) -> Result<Vec<ManagedResource>> {
        let output = run_capture(
            &self.binary,
            &[
                "ps".to_string(),
                "--all".to_string(),
                "--filter".to_string(),
                "label=dev.pb.managed=true".to_string(),
                "--format".to_string(),
                "{{.Names}}".to_string(),
            ],
        )?;
        output
            .lines()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(|name| {
                let labels = run_capture(
                    &self.binary,
                    &[
                        "inspect".to_string(),
                        "--format".to_string(),
                        "{{json .Config.Labels}}".to_string(),
                        name.to_string(),
                    ],
                )?;
                Ok(ManagedResource {
                    name: name.to_string(),
                    labels: parse_oci_labels(&labels).with_context(|| {
                        format!("failed to inspect ownership labels for container {name}")
                    })?,
                })
            })
            .collect()
    }

    fn list_managed_networks(&self) -> Result<Vec<ManagedResource>> {
        let output = run_capture(
            &self.binary,
            &[
                "network".to_string(),
                "ls".to_string(),
                "--filter".to_string(),
                "label=dev.pb.managed=true".to_string(),
                "--format".to_string(),
                "{{.Name}}".to_string(),
            ],
        )?;
        output
            .lines()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(|name| {
                let labels = run_capture(
                    &self.binary,
                    &[
                        "network".to_string(),
                        "inspect".to_string(),
                        "--format".to_string(),
                        "{{json .Labels}}".to_string(),
                        name.to_string(),
                    ],
                )?;
                Ok(ManagedResource {
                    name: name.to_string(),
                    labels: parse_oci_labels(&labels).with_context(|| {
                        format!("failed to inspect ownership labels for network {name}")
                    })?,
                })
            })
            .collect()
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
            let labels = run_capture(
                &self.binary,
                &[
                    "volume".to_string(),
                    "inspect".to_string(),
                    "--format".to_string(),
                    "{{json .Labels}}".to_string(),
                    spec.name.clone(),
                ],
            )?;
            validate_volume_labels(spec, &parse_oci_labels(&labels)?)?;
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
                let labels = run_capture(
                    &self.binary,
                    &[
                        "volume".to_string(),
                        "inspect".to_string(),
                        "--format".to_string(),
                        "{{json .Labels}}".to_string(),
                        spec.name.clone(),
                    ],
                )?;
                validate_volume_labels(spec, &parse_oci_labels(&labels)?)
            }
            Err(create_error) => Err(create_error),
        }
    }

    fn remove_volume(&self, volume: &str) -> Result<()> {
        run_silent(
            &self.binary,
            &["volume".to_string(), "rm".to_string(), volume.to_string()],
        )
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

/// Reconstruct a runtime adapter recorded in a durable local lease.
pub fn runtime_for_binary(binary: &str) -> Result<Box<dyn ContainerRuntime>> {
    if binary.trim().is_empty() || binary.contains(['\n', '\r', '\0']) {
        bail!("invalid container runtime binary in session lease");
    }
    if binary == "container" {
        return Ok(Box::new(AppleContainerRuntime));
    }
    Ok(Box::new(OciRuntime {
        binary: binary.to_string(),
    }))
}

/// Serialize pull/build/inspect/launch for a mutable runtime image reference across daemon and CLI
/// processes. The guard must remain live until the container has been created from the inspected
/// identity, closing the tag mutation race.
pub(crate) fn acquire_image_operation_lock(
    runtime_binary: &str,
    image: &str,
) -> Result<crate::state_lock::StateFileLock> {
    if runtime_binary.trim().is_empty()
        || runtime_binary.contains(['\n', '\r', '\0'])
        || image.trim().is_empty()
        || image.contains(['\n', '\r', '\0'])
    {
        bail!("invalid runtime/image identity for image operation lock");
    }
    let key = crate::environment_lock::sha256(format!("{runtime_binary}\0{image}").as_bytes());
    let root = crate::session_workspace::default_state_root()
        .unwrap_or_else(|_| std::path::PathBuf::from(".pb/state"));
    crate::state_lock::StateFileLock::acquire(
        root.join("image-locks").join(format!("{key}.lock")),
        Duration::from_secs(15 * 60),
    )
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
    for key in spec.env.keys() {
        args.push("--env".to_string());
        args.push(key.clone());
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

fn service_run_args(spec: &ServiceLaunchSpec) -> Vec<String> {
    let mut args = vec![
        "run".to_string(),
        "-i".to_string(),
        "--name".to_string(),
        spec.name.clone(),
    ];
    append_labels(&mut args, &spec.labels);
    for key in spec.env.keys() {
        args.push("--env".to_string());
        args.push(key.clone());
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
    args.push(spec.image.clone());
    args.extend(spec.args.iter().cloned());
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
    let status = Command::new(binary)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("failed to spawn {binary}"))?;
    Ok(status.success())
}

/// Run a command, failing on non-zero exit with bounded subprocess output.
fn run_silent(binary: &str, args: &[String]) -> Result<()> {
    run_silent_with_env(binary, args, &BTreeMap::new())
}

fn run_silent_with_env(
    binary: &str,
    args: &[String],
    env: &BTreeMap<String, String>,
) -> Result<()> {
    let output = run_bounded_with_env(binary, args, env)?;
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
    let output = run_bounded(binary, args)?;
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
    if output.stdout_truncated {
        bail!(
            "{binary} output exceeded the {} byte capture limit",
            MAX_RUNTIME_STDOUT_BYTES
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

struct BoundedCommandOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_truncated: bool,
}

fn run_bounded(binary: &str, args: &[String]) -> Result<BoundedCommandOutput> {
    run_bounded_with_env(binary, args, &BTreeMap::new())
}

fn run_bounded_with_env(
    binary: &str,
    args: &[String],
    env: &BTreeMap<String, String>,
) -> Result<BoundedCommandOutput> {
    validate_process_env(env)?;
    let mut child = Command::new(binary)
        .args(args)
        .envs(env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to spawn {binary}"))?;
    let stdout = child
        .stdout
        .take()
        .context("runtime stdout was unavailable")?;
    let stderr = child
        .stderr
        .take()
        .context("runtime stderr was unavailable")?;
    let stdout_reader =
        std::thread::spawn(move || drain_bounded_stream(stdout, MAX_RUNTIME_STDOUT_BYTES));
    let stderr_reader =
        std::thread::spawn(move || drain_bounded_stream(stderr, MAX_RUNTIME_STDERR_BYTES));
    let status = child
        .wait()
        .with_context(|| format!("failed to wait for {binary}"))?;
    let (stdout, stdout_truncated) = stdout_reader
        .join()
        .map_err(|_| anyhow::anyhow!("runtime stdout reader panicked"))??;
    let (stderr, _) = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("runtime stderr reader panicked"))??;
    Ok(BoundedCommandOutput {
        status,
        stdout,
        stderr,
        stdout_truncated,
    })
}

fn drain_bounded_stream(mut reader: impl Read, limit: usize) -> Result<(Vec<u8>, bool)> {
    let mut captured = Vec::with_capacity(limit.min(8 * 1024));
    let mut truncated = false;
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let remaining = limit.saturating_sub(captured.len());
        let keep = remaining.min(read);
        captured.extend_from_slice(&buffer[..keep]);
        truncated |= keep < read;
    }
    Ok((captured, truncated))
}

fn parse_oci_labels(output: &str) -> Result<BTreeMap<String, String>> {
    serde_json::from_str(output.trim()).context("runtime returned invalid JSON labels")
}

fn apple_volume_labels(name: &str) -> Result<Option<BTreeMap<String, String>>> {
    let output = run_capture(
        "container",
        &[
            "volume".to_string(),
            "inspect".to_string(),
            name.to_string(),
        ],
    )?;
    Ok(parse_apple_managed_resources(&output, "configuration")
        .into_iter()
        .find(|resource| resource.name == name)
        .map(|resource| resource.labels))
}

fn validate_volume_labels(spec: &VolumeSpec, actual: &BTreeMap<String, String>) -> Result<()> {
    if spec
        .labels
        .iter()
        .all(|(key, expected)| actual.get(key) == Some(expected))
    {
        return Ok(());
    }
    bail!(
        "refusing to reuse volume '{}' because its ownership labels do not match the requested cache",
        spec.name
    )
}

fn parse_apple_managed_resources(output: &str, nested: &str) -> Vec<ManagedResource> {
    let values = serde_json::Deserializer::from_str(output)
        .into_iter::<serde_json::Value>()
        .filter_map(Result::ok)
        .flat_map(|value| match value {
            serde_json::Value::Array(values) => values,
            value => vec![value],
        });
    values
        .filter_map(|value| {
            let labels = find_string_map(&value, "labels").unwrap_or_default();
            if labels.get("dev.pb.managed").map(String::as_str) != Some("true") {
                return None;
            }
            let name = value
                .get(nested)
                .and_then(|value| value.get("id").or_else(|| value.get("name")))
                .or_else(|| value.get("id"))
                .or_else(|| value.get("name"))
                .and_then(serde_json::Value::as_str)?;
            Some(ManagedResource {
                name: name.to_string(),
                labels,
            })
        })
        .collect()
}

fn find_string_map(value: &serde_json::Value, key: &str) -> Option<BTreeMap<String, String>> {
    match value {
        serde_json::Value::Object(object) => {
            if let Some(serde_json::Value::Object(labels)) = object.get(key) {
                return Some(
                    labels
                        .iter()
                        .filter_map(|(key, value)| {
                            value.as_str().map(|value| (key.clone(), value.to_string()))
                        })
                        .collect(),
                );
            }
            object
                .values()
                .find_map(|value| find_string_map(value, key))
        }
        serde_json::Value::Array(values) => {
            values.iter().find_map(|value| find_string_map(value, key))
        }
        _ => None,
    }
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
    use std::io::{Read, Write};
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
                "CI",
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
        assert!(!apple_run_args(&spec).iter().any(|arg| arg == "true"));
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

    #[test]
    fn managed_service_is_named_labelled_and_never_uses_auto_remove() {
        let args = service_run_args(&ServiceLaunchSpec {
            name: "pb-svc-one".to_string(),
            image: "example/mcp:locked".to_string(),
            args: vec!["serve".to_string()],
            workdir: "/workspace".to_string(),
            mounts: vec![ContainerMount::bind(
                Path::new("/task/worktree"),
                "/workspace",
                true,
            )],
            labels: BTreeMap::from([
                ("dev.pb.managed".to_string(), "true".to_string()),
                ("dev.pb.session".to_string(), "s1".to_string()),
            ]),
            env: BTreeMap::from([("TOKEN".to_string(), "secret".to_string())]),
            network: Some("pb-svcnet-one".to_string()),
            resources: ContainerResources {
                cpus: 1,
                memory_mb: 512,
            },
            tmpfs: vec!["/tmp".to_string()],
            read_only_root: true,
        });
        assert!(args.windows(2).any(|pair| pair == ["--name", "pb-svc-one"]));
        assert!(args.iter().any(|arg| arg == "dev.pb.managed=true"));
        assert!(args.windows(2).any(|pair| pair == ["--env", "TOKEN"]));
        assert!(!args.iter().any(|arg| arg == "secret"));
        assert!(!args.iter().any(|arg| arg == "--rm"));
    }

    #[test]
    fn managed_service_replacement_requires_matching_exposed_ownership_labels() {
        let expected = BTreeMap::from([
            ("dev.pb.managed".to_string(), "true".to_string()),
            ("dev.pb.project".to_string(), "project-a".to_string()),
            ("dev.pb.session".to_string(), "session-a".to_string()),
        ]);
        let owned = ManagedResource {
            name: "pb-svc-one".to_string(),
            labels: expected.clone(),
        };
        assert!(managed_resource_matches(&owned, "pb-svc-one", &expected));
        assert!(!managed_resource_matches(
            &ManagedResource {
                name: "pb-svc-one".to_string(),
                labels: BTreeMap::new(),
            },
            "pb-svc-one",
            &expected
        ));

        let mut foreign = owned;
        foreign
            .labels
            .insert("dev.pb.session".to_string(), "session-b".to_string());
        assert!(!managed_resource_matches(&foreign, "pb-svc-one", &expected));
    }

    #[test]
    fn managed_process_streams_and_shuts_down_after_stdin_closes() {
        let mut process =
            ManagedProcess::spawn("sh", &["-c".to_string(), "cat".to_string()]).unwrap();
        let mut stdin = process.take_stdin().unwrap();
        let mut stdout = process.take_stdout().unwrap();
        stdin.write_all(b"hello\n").unwrap();
        drop(stdin);
        let mut output = String::new();
        stdout.read_to_string(&mut output).unwrap();
        assert_eq!(output, "hello\n");
        assert!(process.shutdown(Duration::from_secs(1)).unwrap().success());
    }

    #[test]
    fn managed_process_environment_rejects_invalid_keys_and_nul_values() {
        assert!(
            validate_process_env(&BTreeMap::from([("BAD-NAME".to_string(), "x".to_string())]))
                .is_err()
        );
        assert!(
            validate_process_env(&BTreeMap::from([(
                "GOOD".to_string(),
                "bad\0value".to_string()
            )]))
            .is_err()
        );
    }

    #[test]
    fn apple_inventory_parser_requires_pb_ownership_labels() {
        let output = r#"
{"status":"running","configuration":{"id":"pb-one","labels":{"dev.pb.managed":"true","dev.pb.session":"s1"}}}
{"status":"running","configuration":{"id":"foreign","labels":{"owner":"other"}}}
"#;
        assert_eq!(
            parse_apple_managed_resources(output, "configuration"),
            vec![ManagedResource {
                name: "pb-one".to_string(),
                labels: BTreeMap::from([
                    ("dev.pb.managed".to_string(), "true".to_string()),
                    ("dev.pb.session".to_string(), "s1".to_string()),
                ]),
            }]
        );
    }

    #[test]
    fn oci_inventory_labels_preserve_complete_ownership() {
        let labels = parse_oci_labels(
            r#"{"dev.pb.managed":"true","dev.pb.project":"p1","dev.pb.session":"s1"}"#,
        )
        .unwrap();
        assert_eq!(labels.get("dev.pb.project").map(String::as_str), Some("p1"));
        assert!(parse_oci_labels("managed=true").is_err());
    }

    #[test]
    fn existing_cache_volume_requires_its_complete_provenance_labels() {
        let spec = VolumeSpec {
            name: "pb-cache-one".to_string(),
            labels: BTreeMap::from([
                ("dev.pb.managed".to_string(), "true".to_string()),
                ("dev.pb.project".to_string(), "project-a".to_string()),
                ("dev.pb.fingerprint".to_string(), "lock-a".to_string()),
            ]),
        };
        assert!(validate_volume_labels(&spec, &spec.labels).is_ok());
        let mut foreign = spec.labels.clone();
        foreign.insert("dev.pb.fingerprint".to_string(), "lock-b".to_string());
        assert!(validate_volume_labels(&spec, &foreign).is_err());
        assert!(validate_volume_labels(&spec, &BTreeMap::new()).is_err());
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
        let labels = BTreeMap::from([
            ("dev.pb.managed".to_string(), "true".to_string()),
            ("dev.pb.project".to_string(), "conformance".to_string()),
            ("dev.pb.session".to_string(), name.clone()),
            ("dev.pb.role".to_string(), "conformance".to_string()),
        ]);
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
        let mut handle = ContainerHandle {
            runtime,
            container_id,
            network: Some(network),
        };
        handle
            .exec("test -d /workspace && test ! -e /definitely-not-present")
            .context("workspace bind probe failed")?;
        handle
            .exec("touch /tmp/pb-conformance && rm /tmp/pb-conformance")
            .context("tmpfs write probe failed")?;
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
        assert!(
            handle
                .exec("wget -q -T 3 -O /dev/null http://host.container.internal:9")
                .is_err(),
            "Apple internal network unexpectedly reached the standard host-service alias"
        );
        assert!(handle.runtime.container_exists(&handle.container_id)?);
        assert!(
            handle
                .runtime
                .list_managed_containers()?
                .iter()
                .any(|resource| resource.name == handle.container_id)
        );
        let mut process =
            handle.spawn_exec(&["sh".to_string(), "-lc".to_string(), "cat".to_string()])?;
        let mut stdin = process.take_stdin()?;
        let mut stdout = process.take_stdout()?;
        stdin.write_all(b"streaming-conformance\n")?;
        drop(stdin);
        let mut output = String::new();
        stdout.read_to_string(&mut output)?;
        assert_eq!(output, "streaming-conformance\n");
        assert!(process.shutdown(Duration::from_secs(2))?.success());

        let service_name = format!("{name}-service");
        let mut service = spawn_managed_service(
            "container",
            &ServiceLaunchSpec {
                name: service_name.clone(),
                image: "docker.io/library/alpine:3.20".to_string(),
                args: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "IFS= read -r line; printf 'service:%s\\n' \"$line\"".to_string(),
                ],
                workdir: "/tmp".to_string(),
                mounts: Vec::new(),
                labels: BTreeMap::from([
                    ("dev.pb.managed".to_string(), "true".to_string()),
                    ("dev.pb.project".to_string(), "conformance".to_string()),
                    ("dev.pb.session".to_string(), name.clone()),
                    ("dev.pb.role".to_string(), "service-conformance".to_string()),
                ]),
                env: BTreeMap::new(),
                network: handle.network.clone(),
                resources: ContainerResources {
                    cpus: 1,
                    memory_mb: 512,
                },
                tmpfs: vec!["/tmp".to_string()],
                read_only_root: true,
            },
        )?;
        let mut service_stdin = service.take_stdin()?;
        let mut service_stdout = service.take_stdout()?;
        service_stdin.write_all(b"protocol\n")?;
        drop(service_stdin);
        let mut service_output = String::new();
        service_stdout.read_to_string(&mut service_output)?;
        assert_eq!(service_output, "service:protocol\n");
        service.shutdown(Duration::from_secs(2))?;
        assert!(!AppleContainerRuntime.container_exists(&service_name)?);

        let stop_started = Instant::now();
        handle.shutdown(Duration::from_secs(1))?;
        assert!(
            stop_started.elapsed() < Duration::from_secs(5),
            "Apple container stop exceeded its bounded cancellation window"
        );
        let container_name = handle.container_id.clone();
        let network_name = handle.network.clone().unwrap();
        handle.cleanup()?;
        let verifier = AppleContainerRuntime;
        assert!(!verifier.container_exists(&container_name)?);
        assert!(
            !verifier
                .list_managed_networks()?
                .iter()
                .any(|resource| resource.name == network_name)
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
