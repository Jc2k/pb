use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::container::{
    self, ContainerHandle, ContainerMount, ContainerResources, ContainerRuntime, ManagedProcess,
    ManagedServiceProcess, NetworkSpec, RuntimeInfo, ServiceLaunchSpec,
};

pub const SESSION_LEASE_RECORD_VERSION: u32 = 1;
const DEFAULT_IDLE_TTL_MS: u64 = 30 * 60 * 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseState {
    New,
    Preparing,
    Ready,
    InUse,
    Idle,
    Reconciling,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseResourceKind {
    Container,
    Network,
    Workspace,
    CacheAttachment,
    LspProcess,
    McpService,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseResourceRecord {
    pub kind: LeaseResourceKind,
    pub name: String,
    pub role: String,
    pub persistent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionLeaseRecord {
    pub version: u32,
    pub lease_id: String,
    pub session_id: String,
    pub project_id: String,
    pub environment_lock_sha256: String,
    pub workspace_root: PathBuf,
    pub repository_root: PathBuf,
    pub runtime_binary: String,
    pub runtime_version: String,
    pub desired_state: LeaseState,
    pub observed_state: LeaseState,
    pub resources: Vec<LeaseResourceRecord>,
    pub created_at_ms: u64,
    pub last_used_at_ms: u64,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone)]
pub struct SessionLeaseSeed {
    pub lease_id: String,
    pub session_id: String,
    pub project_id: String,
    pub environment_lock_sha256: String,
    pub workspace_root: PathBuf,
    pub repository_root: PathBuf,
    pub container_name: String,
    pub network_name: Option<String>,
    pub runtime_info: RuntimeInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ServiceWorkspaceAccess {
    #[default]
    None,
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ServiceNetworkAccess {
    #[default]
    None,
    Session,
    Egress,
}

#[derive(Debug, Clone)]
pub struct SessionServiceSpec {
    pub service_name: String,
    pub role: String,
    pub kind: LeaseResourceKind,
    pub image: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub working_directory: Option<PathBuf>,
    pub cache_scope_sha256: String,
    pub workspace_access: ServiceWorkspaceAccess,
    pub network_access: ServiceNetworkAccess,
    pub cache_ids: Vec<String>,
}

pub struct SessionEnvironmentLease {
    record_path: PathBuf,
    record: Mutex<SessionLeaseRecord>,
    handle: Mutex<Option<ContainerHandle>>,
    cache_attachments: Mutex<Vec<crate::cache_manager::CacheAttachment>>,
    active_handles: AtomicUsize,
}

impl SessionEnvironmentLease {
    pub fn exec(&self, command: &str) -> Result<String> {
        let handle = self
            .handle
            .lock()
            .map_err(|_| anyhow::anyhow!("session container handle lock is poisoned"))?;
        handle
            .as_ref()
            .context("session container is no longer running")?
            .exec(command)
    }

    pub fn spawn_exec(&self, argv: &[String]) -> Result<ManagedProcess> {
        let handle = self
            .handle
            .lock()
            .map_err(|_| anyhow::anyhow!("session container handle lock is poisoned"))?;
        handle
            .as_ref()
            .context("session container is no longer running")?
            .spawn_exec(argv)
    }

    pub fn spawn_exec_with_env(
        &self,
        argv: &[String],
        env: &BTreeMap<String, String>,
    ) -> Result<ManagedProcess> {
        let handle = self
            .handle
            .lock()
            .map_err(|_| anyhow::anyhow!("session container handle lock is poisoned"))?;
        handle
            .as_ref()
            .context("session container is no longer running")?
            .spawn_exec_with_env(argv, env)
    }

    pub fn record(&self) -> Result<SessionLeaseRecord> {
        self.record
            .lock()
            .map(|record| record.clone())
            .map_err(|_| anyhow::anyhow!("session lease record lock is poisoned"))
    }

    fn primary_container_is_managed(&self) -> Result<bool> {
        let record = self.record()?;
        let handle = self
            .handle
            .lock()
            .map_err(|_| anyhow::anyhow!("session container handle lock is poisoned"))?;
        let Some(handle) = handle.as_ref() else {
            return Ok(false);
        };
        managed_container_matches(
            handle.runtime.as_ref(),
            &handle.container_id,
            &record.project_id,
            &record.session_id,
        )
    }

    pub fn spawn_service(&self, spec: SessionServiceSpec) -> Result<ManagedServiceProcess> {
        if !matches!(
            spec.kind,
            LeaseResourceKind::LspProcess | LeaseResourceKind::McpService
        ) {
            bail!("session service must be an LSP or MCP resource");
        }
        if spec.service_name.trim().is_empty()
            || spec.image.trim().is_empty()
            || spec.image.starts_with('-')
            || spec.image.contains(['\0', '\n', '\r'])
        {
            bail!("session service name and image are required");
        }
        if !is_sha256(&spec.cache_scope_sha256) {
            bail!("session service cache scope must be a lowercase SHA-256 digest");
        }
        container::validate_process_env(&spec.env)?;
        let snapshot = self.record()?;
        let runtime = container::runtime_for_binary(&snapshot.runtime_binary)?;
        let _image_operation =
            container::acquire_image_operation_lock(&snapshot.runtime_binary, &spec.image)?;
        if !runtime.image_exists(&spec.image)? {
            if spec.image.contains("@sha256:") {
                bail!(
                    "pinned integration image {} is not present locally; reinstall or upgrade the integration while online before starting a task",
                    spec.image
                );
            }
            runtime
                .pull(&spec.image)
                .with_context(|| format!("failed to bootstrap service image {}", spec.image))?;
        }
        let image_metadata = runtime.image_fingerprint(&spec.image)?;
        let image_lock = crate::environment_lock::sha256(image_metadata.as_bytes());
        let identity = format!(
            "{}\0{}\0{}\0{}",
            snapshot.session_id, spec.role, spec.service_name, image_lock
        );
        let suffix = &crate::environment_lock::sha256(identity.as_bytes())[..12];
        let container_name = format!("pb-svc-{suffix}");
        let network = match spec.network_access {
            ServiceNetworkAccess::Egress => None,
            ServiceNetworkAccess::Session => resource_name(&snapshot, LeaseResourceKind::Network),
            ServiceNetworkAccess::None => {
                let name = format!("pb-svcnet-{suffix}");
                let recorded = snapshot.resources.iter().any(|resource| {
                    resource.kind == LeaseResourceKind::Network && resource.name == name
                });
                let discovered = runtime
                    .list_managed_networks()?
                    .into_iter()
                    .find(|resource| resource.name == name);
                if discovered.as_ref().is_some_and(|resource| {
                    !managed_resource_labels_match(
                        &resource.labels,
                        &snapshot.project_id,
                        &snapshot.session_id,
                    )
                }) {
                    bail!("refusing to use service network {name} owned by another session");
                }
                let present = discovered.is_some();
                if !present {
                    if !recorded {
                        self.register_resource(LeaseResourceRecord {
                            kind: LeaseResourceKind::Network,
                            name: name.clone(),
                            role: format!("service:{}", spec.service_name),
                            persistent: false,
                        })?;
                    }
                    runtime.create_internal_network(&NetworkSpec {
                        name: name.clone(),
                        labels: BTreeMap::from([
                            ("dev.pb.managed".to_string(), "true".to_string()),
                            ("dev.pb.session".to_string(), snapshot.session_id.clone()),
                            ("dev.pb.project".to_string(), snapshot.project_id.clone()),
                            ("dev.pb.role".to_string(), "service-isolation".to_string()),
                        ]),
                    })?;
                }
                Some(name)
            }
        };
        if spec.network_access == ServiceNetworkAccess::Session && network.is_none() {
            bail!("service requested session network access but the task has no session network");
        }
        let mut mounts = Vec::new();
        let (workdir, workspace_mount) = match spec.workspace_access {
            ServiceWorkspaceAccess::None => {
                if spec.working_directory.is_some() {
                    bail!("service working_directory requires workspace access");
                }
                ("/tmp".to_string(), None)
            }
            ServiceWorkspaceAccess::ReadOnly => (
                resolve_service_workdir(
                    &snapshot.workspace_root,
                    spec.working_directory.as_deref(),
                )?,
                Some(ContainerMount::bind(
                    &snapshot.workspace_root,
                    snapshot.workspace_root.to_string_lossy(),
                    true,
                )),
            ),
            ServiceWorkspaceAccess::ReadWrite => (
                resolve_service_workdir(
                    &snapshot.workspace_root,
                    spec.working_directory.as_deref(),
                )?,
                Some(ContainerMount::bind(
                    &snapshot.workspace_root,
                    snapshot.workspace_root.to_string_lossy(),
                    false,
                )),
            ),
        };
        if let Some(workspace_mount) = workspace_mount {
            mounts.push(workspace_mount);
        }
        for cache_id in &spec.cache_ids {
            let mut base_cache = None;
            for resource in snapshot.resources.iter().filter(|resource| {
                resource.kind == LeaseResourceKind::CacheAttachment
                    && resource.role == "environment-cache"
            }) {
                if let Some(record) =
                    crate::cache_manager::global_cache_manager().record(&resource.name)?
                    && record.logical_id == *cache_id
                {
                    base_cache = Some(record);
                    break;
                }
            }
            let base_cache = base_cache
                .with_context(|| format!("service requested undeclared cache '{cache_id}'"))?;
            let (volume, target) = if base_cache.trust
                == crate::environment::CacheTrustClass::LspIndex
            {
                let provenance = service_cache_provenance(
                    &base_cache.provenance_sha256,
                    &image_lock,
                    &spec.cache_scope_sha256,
                    &spec.role,
                );
                let service_cache_role = format!(
                    "service-cache:{}:{}@{}",
                    spec.role,
                    cache_id,
                    &provenance[..12]
                );
                let existing = self.record()?.resources.into_iter().find(|resource| {
                    resource.kind == LeaseResourceKind::CacheAttachment
                        && resource.role == service_cache_role
                });
                if let Some(existing) = existing {
                    let record = crate::cache_manager::global_cache_manager()
                        .record(&existing.name)?
                        .with_context(|| {
                            format!("session references unknown service cache {}", existing.name)
                        })?;
                    (record.volume_name, record.target)
                } else {
                    let volume_name = format!(
                        "pb-cache-{}-{}-{}",
                        snapshot.project_id,
                        sanitize_resource_part(cache_id),
                        &provenance[..12]
                    );
                    let attachment = crate::cache_manager::global_cache_manager().acquire(
                        runtime.as_ref(),
                        crate::cache_manager::CacheSpec {
                            volume_name: volume_name.clone(),
                            logical_id: cache_id.clone(),
                            target: base_cache.target.clone(),
                            project_id: snapshot.project_id.clone(),
                            environment_lock_sha256: snapshot.environment_lock_sha256.clone(),
                            provenance_sha256: provenance,
                            trust: base_cache.trust,
                            max_bytes: base_cache.max_bytes,
                            preparing_session: snapshot.session_id.clone(),
                        },
                    )?;
                    self.register_resource(LeaseResourceRecord {
                        kind: LeaseResourceKind::CacheAttachment,
                        name: volume_name.clone(),
                        role: service_cache_role,
                        persistent: true,
                    })?;
                    self.cache_attachments
                        .lock()
                        .map_err(|_| anyhow::anyhow!("session cache attachment lock is poisoned"))?
                        .push(attachment);
                    (volume_name, base_cache.target)
                }
            } else {
                (base_cache.volume_name, base_cache.target)
            };
            mounts.push(ContainerMount::volume(volume, target));
        }
        self.register_resource(LeaseResourceRecord {
            kind: spec.kind,
            name: container_name.clone(),
            role: format!("{}@{}", spec.role, &image_lock[..12]),
            persistent: false,
        })?;
        container::spawn_managed_service(
            &snapshot.runtime_binary,
            &ServiceLaunchSpec {
                name: container_name,
                image: spec.image,
                args: spec.args,
                workdir,
                mounts,
                labels: BTreeMap::from([
                    ("dev.pb.managed".to_string(), "true".to_string()),
                    ("dev.pb.session".to_string(), snapshot.session_id),
                    ("dev.pb.project".to_string(), snapshot.project_id),
                    ("dev.pb.role".to_string(), spec.role),
                    ("dev.pb.image-lock".to_string(), image_lock),
                ]),
                env: spec.env,
                network,
                resources: ContainerResources {
                    cpus: 2,
                    memory_mb: 2_048,
                },
                tmpfs: vec!["/tmp".to_string(), "/run".to_string()],
                read_only_root: true,
            },
        )
    }

    fn register_resource(&self, resource: LeaseResourceRecord) -> Result<()> {
        let mut record = self
            .record
            .lock()
            .map_err(|_| anyhow::anyhow!("session lease record lock is poisoned"))?;
        if !record
            .resources
            .iter()
            .any(|existing| existing.kind == resource.kind && existing.name == resource.name)
        {
            record.resources.push(resource);
            record.last_used_at_ms = now_millis();
            save_record_atomic(&self.record_path, &record)?;
        }
        Ok(())
    }

    fn transition(&self, desired: LeaseState, observed: LeaseState) -> Result<()> {
        let mut record = self
            .record
            .lock()
            .map_err(|_| anyhow::anyhow!("session lease record lock is poisoned"))?;
        record.desired_state = desired;
        save_record_atomic(&self.record_path, &record)?;
        record.observed_state = observed;
        record.last_used_at_ms = now_millis();
        save_record_atomic(&self.record_path, &record)
    }

    fn mark_idle(&self, ttl_ms: u64) -> Result<()> {
        let mut record = self
            .record
            .lock()
            .map_err(|_| anyhow::anyhow!("session lease record lock is poisoned"))?;
        record.desired_state = LeaseState::Idle;
        record.observed_state = LeaseState::Idle;
        record.last_used_at_ms = now_millis();
        record.expires_at_ms = record.last_used_at_ms.saturating_add(ttl_ms);
        save_record_atomic(&self.record_path, &record)
    }

    fn shutdown(&self) -> Result<()> {
        self.transition(LeaseState::Stopping, LeaseState::Stopping)?;
        let session_id = self.record()?.session_id;
        crate::lsp::shutdown_session_services(&session_id);
        crate::mcp::shutdown_session_services(&session_id);
        let handle = self
            .handle
            .lock()
            .map_err(|_| anyhow::anyhow!("session container handle lock is poisoned"))?
            .take();
        let mut stop_error = None;
        if let Some(mut handle) = handle {
            let record = self.record()?;
            if let Err(error) = cleanup_recorded_service_resources(&record, handle.runtime.as_ref())
            {
                stop_error = Some(error);
            }
            match managed_container_matches(
                handle.runtime.as_ref(),
                &handle.container_id,
                &record.project_id,
                &record.session_id,
            ) {
                Ok(true) => {
                    if let Err(error) = handle.shutdown(Duration::from_secs(5))
                        && stop_error.is_none()
                    {
                        stop_error = Some(error);
                    }
                    if let Err(error) = handle.runtime.remove(&handle.container_id) {
                        if stop_error.is_none() {
                            stop_error = Some(error);
                        }
                    } else {
                        handle.container_id.clear();
                    }
                }
                Ok(false) => {
                    match handle.runtime.container_exists(&handle.container_id) {
                        Ok(true) if stop_error.is_none() => {
                            stop_error = Some(anyhow::anyhow!(
                                "refusing to stop recorded primary container '{}' without matching project/session labels",
                                handle.container_id
                            ));
                        }
                        Ok(_) => {}
                        Err(error) if stop_error.is_none() => stop_error = Some(error),
                        Err(_) => {}
                    }
                    // Never let the Drop backstop remove a resource whose ownership was not verified.
                    handle.container_id.clear();
                }
                Err(error) => {
                    if stop_error.is_none() {
                        stop_error = Some(error);
                    }
                    handle.container_id.clear();
                }
            }
            if let Some(network) = handle.network.clone() {
                match managed_network_matches(
                    handle.runtime.as_ref(),
                    &network,
                    &record.project_id,
                    &record.session_id,
                ) {
                    Ok(true) => {
                        if let Err(error) = handle.runtime.remove_network(&network) {
                            if stop_error.is_none() {
                                stop_error = Some(error);
                            }
                        } else {
                            handle.network = None;
                        }
                    }
                    Ok(false) => {
                        match handle.runtime.list_managed_networks() {
                            Ok(networks)
                                if networks.iter().any(|resource| resource.name == network)
                                    && stop_error.is_none() =>
                            {
                                stop_error = Some(anyhow::anyhow!(
                                    "refusing to remove recorded network '{network}' without matching project/session labels"
                                ));
                            }
                            Ok(_) => {}
                            Err(error) if stop_error.is_none() => stop_error = Some(error),
                            Err(_) => {}
                        }
                        handle.network = None;
                    }
                    Err(error) => {
                        if stop_error.is_none() {
                            stop_error = Some(error);
                        }
                        handle.network = None;
                    }
                }
            }
            // ContainerHandle::drop retries only failures for resources whose ownership was
            // verified above; foreign identifiers have already been cleared.
            drop(handle);
        } else {
            let record = self.record()?;
            let runtime = container::runtime_for_binary(&record.runtime_binary)?;
            if let Err(error) = cleanup_recorded_service_resources(&record, runtime.as_ref()) {
                stop_error = Some(error);
            }
        }
        self.cache_attachments
            .lock()
            .map_err(|_| anyhow::anyhow!("session cache attachment lock is poisoned"))?
            .clear();
        if let Some(error) = stop_error {
            self.transition(LeaseState::Stopped, LeaseState::Failed)?;
            return Err(error)
                .context("session container shutdown failed; recovery state was retained");
        }
        self.transition(LeaseState::Stopped, LeaseState::Stopped)?;
        if self.record_path.exists() {
            std::fs::remove_file(&self.record_path).with_context(|| {
                format!(
                    "failed to remove lease record {}",
                    self.record_path.display()
                )
            })?;
        }
        Ok(())
    }
}

pub struct SessionLeaseHandle {
    supervisor: &'static EnvironmentSupervisor,
    session_id: String,
    retain_after_turn: bool,
    lease: Arc<SessionEnvironmentLease>,
}

impl SessionLeaseHandle {
    pub fn exec(&self, command: &str) -> Result<String> {
        self.lease.exec(command)
    }

    pub fn spawn_exec(&self, argv: &[String]) -> Result<ManagedProcess> {
        self.lease.spawn_exec(argv)
    }

    pub fn lease(&self) -> Arc<SessionEnvironmentLease> {
        Arc::clone(&self.lease)
    }
}

impl Drop for SessionLeaseHandle {
    fn drop(&mut self) {
        self.supervisor
            .release_handle(&self.session_id, self.retain_after_turn, &self.lease);
    }
}

pub struct EnvironmentSupervisor {
    state_root: PathBuf,
    leases: Mutex<BTreeMap<String, Arc<SessionEnvironmentLease>>>,
    idle_ttl_ms: u64,
}

impl EnvironmentSupervisor {
    pub fn new(state_root: PathBuf) -> Self {
        Self {
            state_root,
            leases: Mutex::new(BTreeMap::new()),
            idle_ttl_ms: DEFAULT_IDLE_TTL_MS,
        }
    }

    pub fn with_idle_ttl(state_root: PathBuf, idle_ttl: Duration) -> Self {
        Self {
            state_root,
            leases: Mutex::new(BTreeMap::new()),
            idle_ttl_ms: idle_ttl.as_millis().min(u64::MAX as u128) as u64,
        }
    }

    pub fn acquire<F>(
        &'static self,
        seed: SessionLeaseSeed,
        runtime: Box<dyn ContainerRuntime>,
        cache_attachments: Vec<crate::cache_manager::CacheAttachment>,
        retain_after_turn: bool,
        create: F,
    ) -> Result<SessionLeaseHandle>
    where
        F: FnOnce(Box<dyn ContainerRuntime>) -> Result<ContainerHandle>,
    {
        let _operation = SessionOperationLock::acquire(self.lock_path(&seed.session_id))?;
        let cache_resource_names = cache_attachments
            .iter()
            .map(|attachment| attachment.volume_name().to_string())
            .collect::<Vec<_>>();
        if let Some(lease) = self
            .leases
            .lock()
            .map_err(|_| anyhow::anyhow!("environment supervisor lock is poisoned"))?
            .get(&seed.session_id)
            .cloned()
        {
            validate_seed(&lease.record()?, &seed)?;
            if lease.primary_container_is_managed()? {
                lease.transition(LeaseState::InUse, LeaseState::InUse)?;
                lease.active_handles.fetch_add(1, Ordering::AcqRel);
                return Ok(SessionLeaseHandle {
                    supervisor: self,
                    session_id: seed.session_id,
                    retain_after_turn,
                    lease,
                });
            }
            self.leases
                .lock()
                .map_err(|_| anyhow::anyhow!("environment supervisor lock is poisoned"))?
                .remove(&seed.session_id);
            lease.shutdown()?;
        }

        let record_path = self.record_path(&seed.session_id);
        if let Some(record) = load_record(&record_path)? {
            if validate_seed(&record, &seed).is_ok()
                && record.expires_at_ms >= now_millis()
                && managed_container_matches(
                    runtime.as_ref(),
                    &seed.container_name,
                    &seed.project_id,
                    &seed.session_id,
                )?
            {
                let lease = Arc::new(SessionEnvironmentLease {
                    record_path,
                    record: Mutex::new(record),
                    handle: Mutex::new(Some(ContainerHandle {
                        runtime,
                        container_id: seed.container_name.clone(),
                        network: seed.network_name.clone(),
                    })),
                    cache_attachments: Mutex::new(cache_attachments),
                    active_handles: AtomicUsize::new(1),
                });
                lease.transition(LeaseState::InUse, LeaseState::InUse)?;
                self.leases
                    .lock()
                    .map_err(|_| anyhow::anyhow!("environment supervisor lock is poisoned"))?
                    .insert(seed.session_id.clone(), Arc::clone(&lease));
                return Ok(SessionLeaseHandle {
                    supervisor: self,
                    session_id: seed.session_id,
                    retain_after_turn,
                    lease,
                });
            }
            cleanup_recorded_resources(&record, runtime.as_ref())?;
            let _ = std::fs::remove_file(&record_path);
        }

        let now = now_millis();
        let mut record = SessionLeaseRecord {
            version: SESSION_LEASE_RECORD_VERSION,
            lease_id: seed.lease_id.clone(),
            session_id: seed.session_id.clone(),
            project_id: seed.project_id.clone(),
            environment_lock_sha256: seed.environment_lock_sha256.clone(),
            workspace_root: seed.workspace_root.clone(),
            repository_root: seed.repository_root.clone(),
            runtime_binary: seed.runtime_info.binary.clone(),
            runtime_version: seed.runtime_info.version.clone(),
            desired_state: LeaseState::Preparing,
            observed_state: LeaseState::New,
            resources: seed_resources(&seed, &cache_resource_names),
            created_at_ms: now,
            last_used_at_ms: now,
            expires_at_ms: now.saturating_add(self.idle_ttl_ms),
        };
        save_record_atomic(&record_path, &record)?;
        let handle = match create(runtime) {
            Ok(handle) => handle,
            Err(error) => {
                record.desired_state = LeaseState::Failed;
                record.observed_state = LeaseState::Failed;
                let _ = save_record_atomic(&record_path, &record);
                return Err(error);
            }
        };
        record.desired_state = LeaseState::InUse;
        record.observed_state = LeaseState::InUse;
        record.last_used_at_ms = now_millis();
        save_record_atomic(&record_path, &record)?;
        let lease = Arc::new(SessionEnvironmentLease {
            record_path,
            record: Mutex::new(record),
            handle: Mutex::new(Some(handle)),
            cache_attachments: Mutex::new(cache_attachments),
            active_handles: AtomicUsize::new(1),
        });
        self.leases
            .lock()
            .map_err(|_| anyhow::anyhow!("environment supervisor lock is poisoned"))?
            .insert(seed.session_id.clone(), Arc::clone(&lease));
        Ok(SessionLeaseHandle {
            supervisor: self,
            session_id: seed.session_id,
            retain_after_turn,
            lease,
        })
    }

    pub fn acquire_service_only(
        &'static self,
        session_id: &str,
        workspace_root: &Path,
        repository_root: &Path,
        runtime: Box<dyn ContainerRuntime>,
        retain_after_turn: bool,
    ) -> Result<SessionLeaseHandle> {
        let _operation = SessionOperationLock::acquire(self.lock_path(session_id))?;
        let runtime_info = runtime.info()?;
        if let Some(lease) = self
            .leases
            .lock()
            .map_err(|_| anyhow::anyhow!("environment supervisor lock is poisoned"))?
            .get(session_id)
            .cloned()
        {
            let record = lease.record()?;
            let service_only = !record
                .resources
                .iter()
                .any(|resource| resource.kind == LeaseResourceKind::Container);
            if service_only
                && record.workspace_root == workspace_root
                && record.runtime_binary == runtime_info.binary
            {
                lease.transition(LeaseState::InUse, LeaseState::InUse)?;
                lease.active_handles.fetch_add(1, Ordering::AcqRel);
                return Ok(SessionLeaseHandle {
                    supervisor: self,
                    session_id: session_id.to_string(),
                    retain_after_turn,
                    lease,
                });
            }
            bail!("session already owns an incompatible environment lease");
        }

        let record_path = self.record_path(session_id);
        if let Some(record) = load_record(&record_path)? {
            cleanup_recorded_resources(&record, runtime.as_ref())?;
            std::fs::remove_file(&record_path).with_context(|| {
                format!(
                    "failed to remove stale lease record {}",
                    record_path.display()
                )
            })?;
        }
        let canonical_repository = repository_root.canonicalize().with_context(|| {
            format!(
                "failed to resolve repository root {}",
                repository_root.display()
            )
        })?;
        let project_id =
            crate::environment_lock::sha256(canonical_repository.to_string_lossy().as_bytes())
                [..12]
                .to_string();
        let authority = crate::environment_lock::sha256(
            format!(
                "service-only\0{}\0{}",
                runtime_info.binary, runtime_info.version
            )
            .as_bytes(),
        );
        let now = now_millis();
        let record = SessionLeaseRecord {
            version: SESSION_LEASE_RECORD_VERSION,
            lease_id: format!(
                "pb-svc-lease-{}",
                &crate::environment_lock::sha256(session_id.as_bytes())[..12]
            ),
            session_id: session_id.to_string(),
            project_id,
            environment_lock_sha256: authority,
            workspace_root: workspace_root.to_path_buf(),
            repository_root: canonical_repository,
            runtime_binary: runtime_info.binary,
            runtime_version: runtime_info.version,
            desired_state: LeaseState::InUse,
            observed_state: LeaseState::InUse,
            resources: vec![LeaseResourceRecord {
                kind: LeaseResourceKind::Workspace,
                name: workspace_root.to_string_lossy().into_owned(),
                role: "task-worktree".to_string(),
                persistent: false,
            }],
            created_at_ms: now,
            last_used_at_ms: now,
            expires_at_ms: now.saturating_add(self.idle_ttl_ms),
        };
        save_record_atomic(&record_path, &record)?;
        let lease = Arc::new(SessionEnvironmentLease {
            record_path,
            record: Mutex::new(record),
            handle: Mutex::new(None),
            cache_attachments: Mutex::new(Vec::new()),
            active_handles: AtomicUsize::new(1),
        });
        self.leases
            .lock()
            .map_err(|_| anyhow::anyhow!("environment supervisor lock is poisoned"))?
            .insert(session_id.to_string(), Arc::clone(&lease));
        Ok(SessionLeaseHandle {
            supervisor: self,
            session_id: session_id.to_string(),
            retain_after_turn,
            lease,
        })
    }

    pub fn mark_idle(&self, session_id: &str) -> Result<()> {
        let _operation = SessionOperationLock::acquire(self.lock_path(session_id))?;
        let lease = self
            .leases
            .lock()
            .map_err(|_| anyhow::anyhow!("environment supervisor lock is poisoned"))?
            .get(session_id)
            .cloned();
        if let Some(lease) = lease {
            lease.mark_idle(self.idle_ttl_ms)?;
        }
        Ok(())
    }

    fn release_handle(
        &self,
        session_id: &str,
        retain_after_turn: bool,
        lease: &SessionEnvironmentLease,
    ) {
        let previous = lease.active_handles.fetch_sub(1, Ordering::AcqRel);
        if previous == 0 {
            lease.active_handles.store(0, Ordering::Release);
            return;
        }
        if previous > 1 {
            return;
        }
        if retain_after_turn {
            let _ = self.mark_idle(session_id);
        } else {
            if let Err(error) = self.terminate(session_id) {
                eprintln!("failed to terminate session environment {session_id}: {error:#}");
            }
            if let Ok(manager) = crate::session_workspace::WorkspaceManager::persistent()
                && let Ok(Some(record)) = manager.find_record_by_session(session_id)
            {
                let _ = manager.remove(&record, false);
            }
        }
    }

    pub fn active_environment_lock(&self, session_id: &str) -> Result<Option<String>> {
        self.leases
            .lock()
            .map_err(|_| anyhow::anyhow!("environment supervisor lock is poisoned"))?
            .get(session_id)
            .map(|lease| lease.record().map(|record| record.environment_lock_sha256))
            .transpose()
    }

    pub fn terminate(&self, session_id: &str) -> Result<()> {
        let _operation = SessionOperationLock::acquire(self.lock_path(session_id))?;
        let lease = self
            .leases
            .lock()
            .map_err(|_| anyhow::anyhow!("environment supervisor lock is poisoned"))?
            .remove(session_id);
        if let Some(lease) = lease {
            lease.shutdown()?;
        } else {
            let path = self.record_path(session_id);
            if let Some(record) = load_record(&path)? {
                let runtime = container::runtime_for_binary(&record.runtime_binary)?;
                cleanup_recorded_resources(&record, runtime.as_ref())?;
                if path.exists() {
                    std::fs::remove_file(path)?;
                }
            }
        }
        Ok(())
    }

    pub fn reconcile(&self) -> Result<()> {
        let records_dir = self.records_dir();
        std::fs::create_dir_all(&records_dir)?;
        let mut records = Vec::new();
        for entry in std::fs::read_dir(&records_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            if let Some(record) = load_record(&entry.path())? {
                records.push((entry.path(), record));
            }
        }
        let mut active_containers = BTreeSet::new();
        let mut active_networks = BTreeSet::new();
        let now = now_millis();
        for (path, discovered_record) in records {
            let _operation =
                SessionOperationLock::acquire(self.lock_path(&discovered_record.session_id))?;
            // The discovery pass is intentionally unlocked. Reload only after acquiring the
            // session lock so reconciliation never overwrites a concurrent acquire/release.
            let Some(mut record) = load_record(&path)? else {
                continue;
            };
            if record.session_id != discovered_record.session_id {
                bail!(
                    "session lease identity changed while reconciling {}",
                    path.display()
                );
            }
            let runtime = container::runtime_for_binary(&record.runtime_binary)?;
            // Stdio protocol connections cannot survive a daemon restart. Their named resources
            // remain recoverable in the ledger, so reconciliation removes them deterministically
            // and leaves the task container available for lazy service restart.
            cleanup_recorded_service_resources(&record, runtime.as_ref())?;
            record.resources.retain(|resource| {
                !(matches!(
                    resource.kind,
                    LeaseResourceKind::LspProcess | LeaseResourceKind::McpService
                ) || (resource.kind == LeaseResourceKind::Network
                    && resource.role.starts_with("service:")))
            });
            let container = resource_name(&record, LeaseResourceKind::Container);
            let network = resource_name(&record, LeaseResourceKind::Network);
            let container_is_managed = match container.as_deref() {
                Some(name) => managed_container_matches(
                    runtime.as_ref(),
                    name,
                    &record.project_id,
                    &record.session_id,
                )?,
                None => false,
            };
            let valid = record.expires_at_ms >= now
                && !matches!(
                    record.desired_state,
                    LeaseState::Stopped | LeaseState::Failed
                )
                && container_is_managed;
            if !valid {
                cleanup_recorded_resources(&record, runtime.as_ref())?;
                let _ = std::fs::remove_file(path);
                continue;
            }
            record.desired_state = LeaseState::Idle;
            record.observed_state = LeaseState::Idle;
            save_record_atomic(&path, &record)?;
            if let Some(container) = container.clone() {
                active_containers.insert(container.clone());
                if let Some(network) = network.clone() {
                    active_networks.insert(network.clone());
                }
                let cache_attachments = record
                    .resources
                    .iter()
                    .filter(|resource| resource.kind == LeaseResourceKind::CacheAttachment)
                    .map(|resource| {
                        crate::cache_manager::global_cache_manager().attach_existing(&resource.name)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let lease = Arc::new(SessionEnvironmentLease {
                    record_path: path,
                    record: Mutex::new(record.clone()),
                    handle: Mutex::new(Some(ContainerHandle {
                        runtime,
                        container_id: container,
                        network,
                    })),
                    cache_attachments: Mutex::new(cache_attachments),
                    active_handles: AtomicUsize::new(0),
                });
                self.leases
                    .lock()
                    .map_err(|_| anyhow::anyhow!("environment supervisor lock is poisoned"))?
                    .insert(record.session_id, lease);
            }
        }

        if let Some(runtime) = container::detect_runtime() {
            for resource in runtime.list_managed_containers()? {
                if !active_containers.contains(&resource.name)
                    && has_complete_session_ownership_labels(&resource.labels)
                {
                    let _ = runtime.remove(&resource.name);
                }
            }
            for resource in runtime.list_managed_networks()? {
                if !active_networks.contains(&resource.name)
                    && has_complete_session_ownership_labels(&resource.labels)
                {
                    let _ = runtime.remove_network(&resource.name);
                }
            }
        }
        Ok(())
    }

    pub fn reap_expired(&self) -> Result<Vec<String>> {
        let now = now_millis();
        let expired = self
            .leases
            .lock()
            .map_err(|_| anyhow::anyhow!("environment supervisor lock is poisoned"))?
            .iter()
            .filter_map(|(session_id, lease)| {
                lease
                    .record()
                    .ok()
                    .filter(|record| {
                        record.observed_state == LeaseState::Idle && record.expires_at_ms < now
                    })
                    .map(|_| session_id.clone())
            })
            .collect::<Vec<_>>();
        for session_id in &expired {
            self.terminate(session_id)?;
            let manager = crate::session_workspace::WorkspaceManager::persistent()?;
            if let Some(record) = manager.find_record_by_session(session_id)? {
                let _ = manager.remove(&record, false)?;
            }
        }
        Ok(expired)
    }

    pub fn retry_failed_cleanup(&self) -> Result<Vec<String>> {
        self.retry_failed_cleanup_with(
            |binary| container::runtime_for_binary(binary),
            cleanup_persistent_session_workspace,
        )
    }

    fn retry_failed_cleanup_with(
        &self,
        mut runtime_for_binary: impl FnMut(&str) -> Result<Box<dyn ContainerRuntime>>,
        mut cleanup_workspace: impl FnMut(&str) -> Result<()>,
    ) -> Result<Vec<String>> {
        let records_dir = self.records_dir();
        std::fs::create_dir_all(&records_dir)?;
        let mut cleaned = Vec::new();
        let mut first_error = None;
        for entry in std::fs::read_dir(&records_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let Some(discovered) = load_record(&entry.path())? else {
                continue;
            };
            if discovered.observed_state != LeaseState::Failed {
                continue;
            }
            let _operation = SessionOperationLock::acquire(self.lock_path(&discovered.session_id))?;
            let Some(record) = load_record(&entry.path())? else {
                continue;
            };
            if record.observed_state != LeaseState::Failed {
                continue;
            }
            let result = runtime_for_binary(&record.runtime_binary)
                .and_then(|runtime| cleanup_recorded_resources(&record, runtime.as_ref()))
                .and_then(|()| cleanup_workspace(&record.session_id));
            match result {
                Ok(()) => {
                    if entry.path().exists() {
                        std::fs::remove_file(entry.path())?;
                    }
                    cleaned.push(record.session_id);
                }
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        first_error.map_or(Ok(cleaned), Err)
    }

    fn records_dir(&self) -> PathBuf {
        self.state_root.join("leases")
    }

    fn record_path(&self, session_id: &str) -> PathBuf {
        self.records_dir().join(format!(
            "{}.json",
            crate::environment_lock::sha256(session_id.as_bytes())
        ))
    }

    fn lock_path(&self, session_id: &str) -> PathBuf {
        self.state_root.join("lease-locks").join(format!(
            "{}.lock",
            crate::environment_lock::sha256(session_id.as_bytes())
        ))
    }
}

fn cleanup_persistent_session_workspace(session_id: &str) -> Result<()> {
    let manager = crate::session_workspace::WorkspaceManager::persistent()?;
    let Some(record) = manager.find_record_by_session(session_id)? else {
        return Ok(());
    };
    if !manager.remove(&record, false)? {
        bail!(
            "session workspace {} contains uncommitted changes and was preserved",
            record.worktree_root.display()
        );
    }
    Ok(())
}

struct SessionOperationLock {
    _lock: crate::state_lock::StateFileLock,
}

impl SessionOperationLock {
    fn acquire(path: PathBuf) -> Result<Self> {
        Ok(Self {
            _lock: crate::state_lock::StateFileLock::acquire(path, Duration::from_secs(10))?,
        })
    }
}

static GLOBAL_SUPERVISOR: OnceLock<EnvironmentSupervisor> = OnceLock::new();

pub fn global_supervisor() -> &'static EnvironmentSupervisor {
    GLOBAL_SUPERVISOR.get_or_init(|| {
        let root = crate::session_workspace::default_state_root()
            .unwrap_or_else(|_| PathBuf::from(".pb/state"));
        EnvironmentSupervisor::new(root)
    })
}

pub fn initialize_global_supervisor() -> Result<()> {
    crate::cache_manager::global_cache_manager().reconcile()?;
    global_supervisor().reconcile()
}

pub fn terminate_global_session(session_id: &str) -> Result<()> {
    global_supervisor().terminate(session_id)
}

fn validate_seed(record: &SessionLeaseRecord, seed: &SessionLeaseSeed) -> Result<()> {
    if record.version != SESSION_LEASE_RECORD_VERSION
        || record.lease_id != seed.lease_id
        || record.session_id != seed.session_id
        || record.project_id != seed.project_id
        || record.environment_lock_sha256 != seed.environment_lock_sha256
        || record.workspace_root != seed.workspace_root
        || record.repository_root != seed.repository_root
        || record.runtime_binary != seed.runtime_info.binary
        || resource_name(record, LeaseResourceKind::Container).as_deref()
            != Some(seed.container_name.as_str())
        || resource_name(record, LeaseResourceKind::Network) != seed.network_name
    {
        bail!("persisted session lease does not match the requested environment");
    }
    Ok(())
}

fn managed_container_matches(
    runtime: &dyn ContainerRuntime,
    name: &str,
    project_id: &str,
    session_id: &str,
) -> Result<bool> {
    let Some(resource) = runtime
        .list_managed_containers()?
        .into_iter()
        .find(|resource| resource.name == name)
    else {
        return Ok(false);
    };
    Ok(managed_resource_labels_match(
        &resource.labels,
        project_id,
        session_id,
    ))
}

fn managed_network_matches(
    runtime: &dyn ContainerRuntime,
    name: &str,
    project_id: &str,
    session_id: &str,
) -> Result<bool> {
    let Some(resource) = runtime
        .list_managed_networks()?
        .into_iter()
        .find(|resource| resource.name == name)
    else {
        return Ok(false);
    };
    Ok(managed_resource_labels_match(
        &resource.labels,
        project_id,
        session_id,
    ))
}

fn resolve_service_workdir(workspace_root: &Path, configured: Option<&Path>) -> Result<String> {
    let canonical_root = workspace_root.canonicalize().with_context(|| {
        format!(
            "failed to resolve service workspace {}",
            workspace_root.display()
        )
    })?;
    let candidate = match configured {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => canonical_root.join(path),
        None => canonical_root.clone(),
    };
    let candidate = candidate.canonicalize().with_context(|| {
        format!(
            "failed to resolve service working directory {}",
            candidate.display()
        )
    })?;
    if !candidate.starts_with(&canonical_root) {
        bail!(
            "service working directory {} escapes task workspace {}",
            candidate.display(),
            canonical_root.display()
        );
    }
    Ok(candidate.to_string_lossy().into_owned())
}

fn service_cache_provenance(
    base_provenance: &str,
    image_lock: &str,
    cache_scope: &str,
    role: &str,
) -> String {
    crate::environment_lock::sha256(
        format!("{base_provenance}\0{image_lock}\0{cache_scope}\0{role}").as_bytes(),
    )
}

fn sanitize_resource_part(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    let sanitized = sanitized.trim_matches('-');
    if sanitized.is_empty() {
        "cache".to_string()
    } else {
        sanitized.chars().take(24).collect()
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn managed_resource_labels_match(
    labels: &BTreeMap<String, String>,
    project_id: &str,
    session_id: &str,
) -> bool {
    if labels.get("dev.pb.managed").map(String::as_str) != Some("true") {
        return false;
    }
    if labels.get("dev.pb.project").map(String::as_str) != Some(project_id) {
        return false;
    }
    let Some(session) = labels.get("dev.pb.session") else {
        return false;
    };
    let session_hash = crate::environment_lock::sha256(session_id.as_bytes());
    if session != session_id && session != &session_hash[..12] {
        return false;
    }
    true
}

fn has_complete_session_ownership_labels(labels: &BTreeMap<String, String>) -> bool {
    labels.get("dev.pb.managed").map(String::as_str) == Some("true")
        && labels
            .get("dev.pb.project")
            .is_some_and(|value| !value.trim().is_empty())
        && labels
            .get("dev.pb.session")
            .is_some_and(|value| !value.trim().is_empty())
}

fn seed_resources(seed: &SessionLeaseSeed, cache_names: &[String]) -> Vec<LeaseResourceRecord> {
    let mut resources = vec![
        LeaseResourceRecord {
            kind: LeaseResourceKind::Workspace,
            name: seed.workspace_root.to_string_lossy().into_owned(),
            role: "task-worktree".to_string(),
            persistent: false,
        },
        LeaseResourceRecord {
            kind: LeaseResourceKind::Container,
            name: seed.container_name.clone(),
            role: "task".to_string(),
            persistent: false,
        },
    ];
    if let Some(network) = &seed.network_name {
        resources.push(LeaseResourceRecord {
            kind: LeaseResourceKind::Network,
            name: network.clone(),
            role: "session".to_string(),
            persistent: false,
        });
    }
    resources.extend(cache_names.iter().map(|name| LeaseResourceRecord {
        kind: LeaseResourceKind::CacheAttachment,
        name: name.clone(),
        role: "environment-cache".to_string(),
        persistent: true,
    }));
    resources
}

fn resource_name(record: &SessionLeaseRecord, kind: LeaseResourceKind) -> Option<String> {
    record
        .resources
        .iter()
        .find(|resource| resource.kind == kind)
        .map(|resource| resource.name.clone())
}

fn cleanup_recorded_resources(
    record: &SessionLeaseRecord,
    runtime: &dyn ContainerRuntime,
) -> Result<()> {
    let mut first_error = None;
    for resource in record.resources.iter().filter(|resource| {
        matches!(
            resource.kind,
            LeaseResourceKind::Container
                | LeaseResourceKind::LspProcess
                | LeaseResourceKind::McpService
        )
    }) {
        if managed_container_matches(
            runtime,
            &resource.name,
            &record.project_id,
            &record.session_id,
        )? {
            let _ = runtime.stop(&resource.name, Duration::from_secs(5));
            if let Err(error) = runtime.remove(&resource.name)
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        } else if runtime.container_exists(&resource.name)? && first_error.is_none() {
            first_error = Some(anyhow::anyhow!(
                "refusing to remove recorded container '{}' without matching project/session labels",
                resource.name
            ));
        }
    }
    for resource in record
        .resources
        .iter()
        .filter(|resource| resource.kind == LeaseResourceKind::Network)
    {
        let owned = managed_network_matches(
            runtime,
            &resource.name,
            &record.project_id,
            &record.session_id,
        )?;
        if owned {
            if let Err(error) = runtime.remove_network(&resource.name)
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        } else if runtime
            .list_managed_networks()?
            .iter()
            .any(|network| network.name == resource.name)
            && first_error.is_none()
        {
            first_error = Some(anyhow::anyhow!(
                "refusing to remove recorded network '{}' without matching project/session labels",
                resource.name
            ));
        }
    }
    first_error.map_or(Ok(()), Err)
}

fn cleanup_recorded_service_resources(
    record: &SessionLeaseRecord,
    runtime: &dyn ContainerRuntime,
) -> Result<()> {
    let mut first_error = None;
    for resource in record.resources.iter().filter(|resource| {
        matches!(
            resource.kind,
            LeaseResourceKind::LspProcess | LeaseResourceKind::McpService
        )
    }) {
        match managed_container_matches(
            runtime,
            &resource.name,
            &record.project_id,
            &record.session_id,
        ) {
            Ok(true) => {
                let _ = runtime.stop(&resource.name, Duration::from_secs(2));
                if let Err(error) = runtime.remove(&resource.name)
                    && first_error.is_none()
                {
                    first_error = Some(error);
                }
            }
            Ok(false) => match runtime.container_exists(&resource.name) {
                Ok(true) if first_error.is_none() => {
                    first_error = Some(anyhow::anyhow!(
                        "refusing to remove recorded service '{}' without matching project/session labels",
                        resource.name
                    ));
                }
                Ok(_) => {}
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            },
            Err(error) if first_error.is_none() => first_error = Some(error),
            Err(_) => {}
        }
    }
    for resource in record.resources.iter().filter(|resource| {
        resource.kind == LeaseResourceKind::Network && resource.role.starts_with("service:")
    }) {
        let owned = managed_network_matches(
            runtime,
            &resource.name,
            &record.project_id,
            &record.session_id,
        )?;
        if owned {
            if let Err(error) = runtime.remove_network(&resource.name)
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        } else if runtime
            .list_managed_networks()?
            .iter()
            .any(|network| network.name == resource.name)
            && first_error.is_none()
        {
            first_error = Some(anyhow::anyhow!(
                "refusing to remove recorded service network '{}' without matching project/session labels",
                resource.name
            ));
        }
    }
    first_error.map_or(Ok(()), Err)
}

fn load_record(path: &Path) -> Result<Option<SessionLeaseRecord>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read lease record {}", path.display()))?;
    let record: SessionLeaseRecord = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse lease record {}", path.display()))?;
    if record.version != SESSION_LEASE_RECORD_VERSION {
        bail!(
            "unsupported session lease record version {} in {}",
            record.version,
            path.display()
        );
    }
    Ok(Some(record))
}

fn save_record_atomic(path: &Path, record: &SessionLeaseRecord) -> Result<()> {
    let parent = path.parent().context("lease record has no parent")?;
    std::fs::create_dir_all(parent)?;
    let temp = parent.join(format!(
        ".lease.{}.{}.tmp",
        std::process::id(),
        crate::environment_lock::sha256(record.session_id.as_bytes())
    ));
    std::fs::write(&temp, serde_json::to_vec_pretty(record)?)?;
    std::fs::rename(&temp, path)
        .with_context(|| format!("failed to replace lease record {}", path.display()))
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
    use crate::container::{
        ContainerLaunchSpec, ManagedResource, NetworkSpec, RuntimeCapabilities, RuntimeKind,
        VolumeSpec,
    };
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;
    use tempfile::TempDir;

    struct RuntimeState {
        events: Mutex<Vec<String>>,
        containers: Mutex<BTreeSet<String>>,
        networks: Mutex<BTreeSet<String>>,
        fail_inventory: AtomicBool,
    }

    #[derive(Clone)]
    struct FakeRuntime {
        state: Arc<RuntimeState>,
    }

    impl ContainerRuntime for FakeRuntime {
        fn info(&self) -> Result<RuntimeInfo> {
            Ok(runtime_info())
        }
        fn pull(&self, _: &str) -> Result<()> {
            Ok(())
        }
        fn build(&self, _: &Path, _: &str) -> Result<()> {
            Ok(())
        }
        fn image_exists(&self, _: &str) -> Result<bool> {
            Ok(true)
        }
        fn image_fingerprint(&self, _: &str) -> Result<String> {
            Ok("image".to_string())
        }
        fn create(&self, spec: &ContainerLaunchSpec) -> Result<String> {
            self.state
                .containers
                .lock()
                .unwrap()
                .insert(spec.name.clone());
            Ok(spec.name.clone())
        }
        fn exec(&self, container_id: &str, cmd: &str) -> Result<String> {
            self.state
                .events
                .lock()
                .unwrap()
                .push(format!("exec {container_id} {cmd}"));
            Ok("ok".to_string())
        }
        fn remove(&self, container_id: &str) -> Result<()> {
            self.state.containers.lock().unwrap().remove(container_id);
            self.state
                .events
                .lock()
                .unwrap()
                .push(format!("remove {container_id}"));
            Ok(())
        }
        fn stop(&self, container_id: &str, _: Duration) -> Result<()> {
            self.state
                .events
                .lock()
                .unwrap()
                .push(format!("stop {container_id}"));
            Ok(())
        }
        fn container_exists(&self, container_id: &str) -> Result<bool> {
            Ok(self.state.containers.lock().unwrap().contains(container_id))
        }
        fn list_managed_containers(&self) -> Result<Vec<ManagedResource>> {
            if self.state.fail_inventory.load(Ordering::Acquire) {
                bail!("injected container inventory failure");
            }
            Ok(self
                .state
                .containers
                .lock()
                .unwrap()
                .iter()
                .map(|name| ManagedResource {
                    name: name.clone(),
                    labels: BTreeMap::from([
                        ("dev.pb.managed".to_string(), "true".to_string()),
                        ("dev.pb.project".to_string(), "p1".to_string()),
                        ("dev.pb.session".to_string(), "s1".to_string()),
                    ]),
                })
                .collect())
        }
        fn list_managed_networks(&self) -> Result<Vec<ManagedResource>> {
            if self.state.fail_inventory.load(Ordering::Acquire) {
                bail!("injected network inventory failure");
            }
            Ok(self
                .state
                .networks
                .lock()
                .unwrap()
                .iter()
                .map(|name| ManagedResource {
                    name: name.clone(),
                    labels: BTreeMap::from([
                        ("dev.pb.managed".to_string(), "true".to_string()),
                        ("dev.pb.project".to_string(), "p1".to_string()),
                        ("dev.pb.session".to_string(), "s1".to_string()),
                    ]),
                })
                .collect())
        }
        fn create_internal_network(&self, spec: &NetworkSpec) -> Result<()> {
            self.state
                .networks
                .lock()
                .unwrap()
                .insert(spec.name.clone());
            Ok(())
        }
        fn remove_network(&self, network: &str) -> Result<()> {
            self.state.networks.lock().unwrap().remove(network);
            self.state
                .events
                .lock()
                .unwrap()
                .push(format!("network remove {network}"));
            Ok(())
        }
        fn ensure_volume(&self, _: &VolumeSpec) -> Result<()> {
            Ok(())
        }
    }

    fn runtime_info() -> RuntimeInfo {
        RuntimeInfo {
            kind: RuntimeKind::Apple,
            binary: "fake".to_string(),
            version: "1.0.0".to_string(),
            capabilities: RuntimeCapabilities {
                internal_networks: true,
                named_volumes: true,
                labels: true,
                resource_limits: true,
            },
        }
    }

    fn state() -> Arc<RuntimeState> {
        Arc::new(RuntimeState {
            events: Mutex::new(Vec::new()),
            containers: Mutex::new(BTreeSet::new()),
            networks: Mutex::new(BTreeSet::new()),
            fail_inventory: AtomicBool::new(false),
        })
    }

    fn seed(root: &Path) -> SessionLeaseSeed {
        SessionLeaseSeed {
            lease_id: "lease-s1".to_string(),
            session_id: "s1".to_string(),
            project_id: "p1".to_string(),
            environment_lock_sha256: crate::environment_lock::sha256(b"lock"),
            workspace_root: root.join("worktree"),
            repository_root: root.join("repo"),
            container_name: "pb-task-s1".to_string(),
            network_name: Some("pb-net-s1".to_string()),
            runtime_info: runtime_info(),
        }
    }

    #[test]
    fn service_working_directory_must_exist_inside_the_task_workspace() {
        let dir = TempDir::new().unwrap();
        let workspace = dir.path().join("worktree");
        let nested = workspace.join("tools");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir_all(&outside).unwrap();

        assert_eq!(
            resolve_service_workdir(&workspace, Some(Path::new("tools"))).unwrap(),
            nested.canonicalize().unwrap().to_string_lossy()
        );
        let error = resolve_service_workdir(&workspace, Some(&outside))
            .unwrap_err()
            .to_string();
        assert!(error.contains("escapes task workspace"));
    }

    #[test]
    fn service_index_cache_identity_includes_image_configuration_and_role() {
        let base = crate::environment_lock::sha256(b"base");
        let image = crate::environment_lock::sha256(b"image");
        let scope = crate::environment_lock::sha256(b"scope");
        let first = service_cache_provenance(&base, &image, &scope, "lsp:rust");
        assert_ne!(
            first,
            service_cache_provenance(
                &base,
                &crate::environment_lock::sha256(b"image-two"),
                &scope,
                "lsp:rust"
            )
        );
        assert_ne!(
            first,
            service_cache_provenance(
                &base,
                &image,
                &crate::environment_lock::sha256(b"scope-two"),
                "lsp:rust"
            )
        );
        assert_ne!(
            first,
            service_cache_provenance(&base, &image, &scope, "mcp:rust")
        );
    }

    #[test]
    fn lease_is_reused_across_turns_and_terminal_cleanup_is_ordered() {
        let dir = TempDir::new().unwrap();
        let supervisor = Box::leak(Box::new(EnvironmentSupervisor::new(
            dir.path().to_path_buf(),
        )));
        let state = state();
        let create_count = Arc::new(Mutex::new(0));
        {
            let count = Arc::clone(&create_count);
            let create_state = Arc::clone(&state);
            let runtime = FakeRuntime {
                state: Arc::clone(&state),
            };
            let handle = supervisor
                .acquire(
                    seed(dir.path()),
                    Box::new(runtime),
                    Vec::new(),
                    true,
                    move |runtime| {
                        *count.lock().unwrap() += 1;
                        create_state
                            .containers
                            .lock()
                            .unwrap()
                            .insert("pb-task-s1".to_string());
                        create_state
                            .networks
                            .lock()
                            .unwrap()
                            .insert("pb-net-s1".to_string());
                        Ok(ContainerHandle {
                            runtime,
                            container_id: "pb-task-s1".to_string(),
                            network: Some("pb-net-s1".to_string()),
                        })
                    },
                )
                .unwrap();
            assert_eq!(handle.exec("true").unwrap(), "ok");
        }
        {
            let runtime = FakeRuntime {
                state: Arc::clone(&state),
            };
            let _handle = supervisor
                .acquire(
                    seed(dir.path()),
                    Box::new(runtime),
                    Vec::new(),
                    true,
                    |_| bail!("existing lease should have been reused"),
                )
                .unwrap();
        }
        assert_eq!(*create_count.lock().unwrap(), 1);
        supervisor.terminate("s1").unwrap();
        assert_eq!(
            state.events.lock().unwrap().as_slice(),
            [
                "exec pb-task-s1 true",
                "stop pb-task-s1",
                "remove pb-task-s1",
                "network remove pb-net-s1"
            ]
        );
    }

    #[test]
    fn terminal_cleanup_removes_recorded_sidecars_and_service_networks() {
        let dir = TempDir::new().unwrap();
        let supervisor = Box::leak(Box::new(EnvironmentSupervisor::new(
            dir.path().to_path_buf(),
        )));
        let state = state();
        let handle = supervisor
            .acquire(
                seed(dir.path()),
                Box::new(FakeRuntime {
                    state: Arc::clone(&state),
                }),
                Vec::new(),
                true,
                {
                    let state = Arc::clone(&state);
                    move |runtime| {
                        state
                            .containers
                            .lock()
                            .unwrap()
                            .insert("pb-task-s1".to_string());
                        state
                            .networks
                            .lock()
                            .unwrap()
                            .insert("pb-net-s1".to_string());
                        Ok(ContainerHandle {
                            runtime,
                            container_id: "pb-task-s1".to_string(),
                            network: Some("pb-net-s1".to_string()),
                        })
                    }
                },
            )
            .unwrap();
        let lease = handle.lease();
        lease
            .register_resource(LeaseResourceRecord {
                kind: LeaseResourceKind::McpService,
                name: "pb-svc-mcp".to_string(),
                role: "mcp:test".to_string(),
                persistent: false,
            })
            .unwrap();
        lease
            .register_resource(LeaseResourceRecord {
                kind: LeaseResourceKind::Network,
                name: "pb-svcnet-mcp".to_string(),
                role: "service:test".to_string(),
                persistent: false,
            })
            .unwrap();
        state
            .containers
            .lock()
            .unwrap()
            .insert("pb-svc-mcp".to_string());
        state
            .networks
            .lock()
            .unwrap()
            .insert("pb-svcnet-mcp".to_string());
        drop(handle);

        supervisor.terminate("s1").unwrap();

        assert!(state.containers.lock().unwrap().is_empty());
        assert!(state.networks.lock().unwrap().is_empty());
        assert_eq!(
            state.events.lock().unwrap().as_slice(),
            [
                "stop pb-svc-mcp",
                "remove pb-svc-mcp",
                "network remove pb-svcnet-mcp",
                "stop pb-task-s1",
                "remove pb-task-s1",
                "network remove pb-net-s1",
            ]
        );
    }

    #[test]
    fn failed_cleanup_retains_a_retryable_lease_record() {
        let dir = TempDir::new().unwrap();
        let supervisor = Box::leak(Box::new(EnvironmentSupervisor::new(
            dir.path().to_path_buf(),
        )));
        let state = state();
        let handle = supervisor
            .acquire(
                seed(dir.path()),
                Box::new(FakeRuntime {
                    state: Arc::clone(&state),
                }),
                Vec::new(),
                true,
                {
                    let state = Arc::clone(&state);
                    move |runtime| {
                        state
                            .containers
                            .lock()
                            .unwrap()
                            .insert("pb-task-s1".to_string());
                        Ok(ContainerHandle {
                            runtime,
                            container_id: "pb-task-s1".to_string(),
                            network: None,
                        })
                    }
                },
            )
            .unwrap();
        drop(handle);
        state.fail_inventory.store(true, Ordering::Release);

        assert!(supervisor.terminate("s1").is_err());
        let record_path = supervisor.record_path("s1");
        let record = load_record(&record_path).unwrap().expect("retained record");
        assert_eq!(record.desired_state, LeaseState::Stopped);
        assert_eq!(record.observed_state, LeaseState::Failed);
        assert!(record_path.exists());

        state.fail_inventory.store(false, Ordering::Release);
        let cleaned = supervisor
            .retry_failed_cleanup_with(
                |_| {
                    Ok(Box::new(FakeRuntime {
                        state: Arc::clone(&state),
                    }))
                },
                |_| Ok(()),
            )
            .unwrap();
        assert_eq!(cleaned, vec!["s1".to_string()]);
        assert!(!record_path.exists());
        assert!(state.containers.lock().unwrap().is_empty());
    }

    #[test]
    fn failed_cleanup_keeps_lease_until_workspace_cleanup_succeeds() {
        let dir = TempDir::new().unwrap();
        let supervisor = Box::leak(Box::new(EnvironmentSupervisor::new(
            dir.path().to_path_buf(),
        )));
        let state = state();
        let handle = supervisor
            .acquire(
                seed(dir.path()),
                Box::new(FakeRuntime {
                    state: Arc::clone(&state),
                }),
                Vec::new(),
                true,
                {
                    let state = Arc::clone(&state);
                    move |runtime| {
                        state
                            .containers
                            .lock()
                            .unwrap()
                            .insert("pb-task-s1".to_string());
                        Ok(ContainerHandle {
                            runtime,
                            container_id: "pb-task-s1".to_string(),
                            network: None,
                        })
                    }
                },
            )
            .unwrap();
        drop(handle);
        state.fail_inventory.store(true, Ordering::Release);
        assert!(supervisor.terminate("s1").is_err());
        state.fail_inventory.store(false, Ordering::Release);

        let record_path = supervisor.record_path("s1");
        let mut workspace_attempts = 0;
        assert!(
            supervisor
                .retry_failed_cleanup_with(
                    |_| {
                        Ok(Box::new(FakeRuntime {
                            state: Arc::clone(&state),
                        }))
                    },
                    |_| {
                        workspace_attempts += 1;
                        bail!("workspace cleanup failed")
                    },
                )
                .is_err()
        );
        assert_eq!(workspace_attempts, 1);
        assert!(record_path.exists());

        let cleaned = supervisor
            .retry_failed_cleanup_with(
                |_| {
                    Ok(Box::new(FakeRuntime {
                        state: Arc::clone(&state),
                    }))
                },
                |_| Ok(()),
            )
            .unwrap();
        assert_eq!(cleaned, vec!["s1".to_string()]);
        assert!(!record_path.exists());
    }

    #[test]
    fn teardown_inventory_failure_never_deletes_unverified_names_via_drop() {
        let dir = TempDir::new().unwrap();
        let supervisor = Box::leak(Box::new(EnvironmentSupervisor::new(
            dir.path().to_path_buf(),
        )));
        let state = state();
        let handle = supervisor
            .acquire(
                seed(dir.path()),
                Box::new(FakeRuntime {
                    state: Arc::clone(&state),
                }),
                Vec::new(),
                true,
                {
                    let state = Arc::clone(&state);
                    move |runtime| {
                        state
                            .containers
                            .lock()
                            .unwrap()
                            .insert("pb-task-s1".to_string());
                        state
                            .networks
                            .lock()
                            .unwrap()
                            .insert("pb-net-s1".to_string());
                        Ok(ContainerHandle {
                            runtime,
                            container_id: "pb-task-s1".to_string(),
                            network: Some("pb-net-s1".to_string()),
                        })
                    }
                },
            )
            .unwrap();
        drop(handle);
        state.fail_inventory.store(true, Ordering::Release);

        let error = format!("{:#}", supervisor.terminate("s1").unwrap_err());
        assert!(error.contains("inventory failure"));
        assert!(state.containers.lock().unwrap().contains("pb-task-s1"));
        assert!(state.networks.lock().unwrap().contains("pb-net-s1"));
        assert!(state.events.lock().unwrap().is_empty());
    }

    #[test]
    fn durable_adoption_rejects_foreign_ownership_labels() {
        let session_id = "session-owned";
        let session_hash = crate::environment_lock::sha256(session_id.as_bytes());
        let owned = BTreeMap::from([
            ("dev.pb.managed".to_string(), "true".to_string()),
            ("dev.pb.project".to_string(), "project-a".to_string()),
            ("dev.pb.session".to_string(), session_hash[..12].to_string()),
        ]);
        assert!(managed_resource_labels_match(
            &owned,
            "project-a",
            session_id
        ));

        let mut foreign_project = owned.clone();
        foreign_project.insert("dev.pb.project".to_string(), "project-b".to_string());
        assert!(!managed_resource_labels_match(
            &foreign_project,
            "project-a",
            session_id
        ));

        let mut foreign_session = owned;
        foreign_session.insert("dev.pb.session".to_string(), "other".to_string());
        assert!(!managed_resource_labels_match(
            &foreign_session,
            "project-a",
            session_id
        ));

        let incomplete = BTreeMap::from([("dev.pb.managed".to_string(), "true".to_string())]);
        assert!(!managed_resource_labels_match(
            &incomplete,
            "project-a",
            session_id
        ));
        assert!(!has_complete_session_ownership_labels(&incomplete));
        assert!(has_complete_session_ownership_labels(&BTreeMap::from([
            ("dev.pb.managed".to_string(), "true".to_string()),
            ("dev.pb.project".to_string(), "project-a".to_string()),
            ("dev.pb.session".to_string(), "session-a".to_string()),
        ])));
    }

    #[test]
    fn overlapping_handles_keep_the_lease_in_use_until_the_last_release() {
        let dir = TempDir::new().unwrap();
        let supervisor = Box::leak(Box::new(EnvironmentSupervisor::new(
            dir.path().to_path_buf(),
        )));
        let state = state();
        let no_network_seed = SessionLeaseSeed {
            network_name: None,
            ..seed(dir.path())
        };
        let first = supervisor
            .acquire(
                no_network_seed.clone(),
                Box::new(FakeRuntime {
                    state: Arc::clone(&state),
                }),
                Vec::new(),
                true,
                {
                    let state = Arc::clone(&state);
                    move |runtime| {
                        state
                            .containers
                            .lock()
                            .unwrap()
                            .insert("pb-task-s1".to_string());
                        Ok(ContainerHandle {
                            runtime,
                            container_id: "pb-task-s1".to_string(),
                            network: None,
                        })
                    }
                },
            )
            .unwrap();
        let second = supervisor
            .acquire(
                no_network_seed,
                Box::new(FakeRuntime {
                    state: Arc::clone(&state),
                }),
                Vec::new(),
                true,
                |_| bail!("overlapping handle should reuse the lease"),
            )
            .unwrap();

        drop(first);
        assert_eq!(
            second.lease.record().unwrap().observed_state,
            LeaseState::InUse
        );
        let lease = second.lease();
        drop(second);
        assert_eq!(lease.record().unwrap().observed_state, LeaseState::Idle);
        supervisor.terminate("s1").unwrap();
    }

    #[test]
    fn expired_idle_lease_is_reaped() {
        let dir = TempDir::new().unwrap();
        let supervisor = Box::leak(Box::new(EnvironmentSupervisor::with_idle_ttl(
            dir.path().to_path_buf(),
            Duration::ZERO,
        )));
        let state = state();
        let runtime = FakeRuntime {
            state: Arc::clone(&state),
        };
        {
            let _handle = supervisor
                .acquire(
                    seed(dir.path()),
                    Box::new(runtime),
                    Vec::new(),
                    true,
                    |runtime| {
                        state
                            .containers
                            .lock()
                            .unwrap()
                            .insert("pb-task-s1".to_string());
                        Ok(ContainerHandle {
                            runtime,
                            container_id: "pb-task-s1".to_string(),
                            network: None,
                        })
                    },
                )
                .unwrap();
        }
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(supervisor.reap_expired().unwrap(), vec!["s1"]);
    }
}
