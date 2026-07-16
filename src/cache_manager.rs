use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use crate::container::{ContainerMount, ContainerRuntime, VolumeSpec};
use crate::environment::CacheTrustClass;

pub const CACHE_RECORD_VERSION: u32 = 1;
const PREPARATION_LOCK_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheSpec {
    pub volume_name: String,
    pub logical_id: String,
    pub target: String,
    pub project_id: String,
    pub environment_lock_sha256: String,
    pub provenance_sha256: String,
    pub trust: CacheTrustClass,
    pub max_bytes: Option<u64>,
    pub preparing_session: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheRecord {
    pub version: u32,
    pub volume_name: String,
    #[serde(default)]
    pub runtime_binary: String,
    pub logical_id: String,
    pub target: String,
    pub project_id: String,
    pub environment_lock_sha256: String,
    pub provenance_sha256: String,
    pub trust: CacheTrustClass,
    pub max_bytes: Option<u64>,
    pub size_estimate_bytes: u64,
    pub active_attachments: u64,
    pub preparing_owner: Option<String>,
    pub created_at_ms: u64,
    pub last_used_at_ms: u64,
}

pub struct CacheAttachment {
    manager: &'static CacheManager,
    volume_name: String,
    target: String,
}

impl CacheAttachment {
    pub fn mount(&self) -> ContainerMount {
        ContainerMount::volume(self.volume_name.clone(), self.target.clone())
    }

    pub fn volume_name(&self) -> &str {
        &self.volume_name
    }
}

impl Drop for CacheAttachment {
    fn drop(&mut self) {
        let _ = self.manager.detach(&self.volume_name);
    }
}

pub struct CacheManager {
    state_root: PathBuf,
    operation_lock: Mutex<()>,
}

impl CacheManager {
    pub fn new(state_root: PathBuf) -> Self {
        Self {
            state_root,
            operation_lock: Mutex::new(()),
        }
    }

    pub fn acquire(
        &'static self,
        runtime: &dyn ContainerRuntime,
        spec: CacheSpec,
    ) -> Result<CacheAttachment> {
        validate_spec(&spec)?;
        let _operation = self
            .operation_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("cache manager lock is poisoned"))?;
        let _manager_lock = self.acquire_manager_file_lock()?;
        std::fs::create_dir_all(self.records_dir())?;
        std::fs::create_dir_all(self.locks_dir())?;
        let _preparation = PreparationLock::acquire(self.lock_path(&spec.volume_name))?;
        let path = self.record_path(&spec.volume_name);
        let now = now_millis();
        let runtime_binary = runtime.info()?.binary;
        let mut record = load_record(&path)?.unwrap_or_else(|| CacheRecord {
            version: CACHE_RECORD_VERSION,
            volume_name: spec.volume_name.clone(),
            runtime_binary: runtime_binary.clone(),
            logical_id: spec.logical_id.clone(),
            target: spec.target.clone(),
            project_id: spec.project_id.clone(),
            environment_lock_sha256: spec.environment_lock_sha256.clone(),
            provenance_sha256: spec.provenance_sha256.clone(),
            trust: spec.trust,
            max_bytes: spec.max_bytes,
            size_estimate_bytes: 0,
            active_attachments: 0,
            preparing_owner: None,
            created_at_ms: now,
            last_used_at_ms: now,
        });
        if record.runtime_binary.is_empty() {
            record.runtime_binary = runtime_binary.clone();
        }
        validate_record_matches(&record, &spec)?;
        if record.runtime_binary != runtime_binary {
            bail!(
                "cache volume {} belongs to runtime '{}' rather than '{}'",
                record.volume_name,
                record.runtime_binary,
                runtime_binary
            );
        }
        record.preparing_owner = Some(spec.preparing_session.clone());
        save_record_atomic(&path, &record)?;
        let ensure_result = runtime.ensure_volume(&VolumeSpec {
            name: spec.volume_name.clone(),
            labels: std::collections::BTreeMap::from([
                ("dev.pb.managed".to_string(), "true".to_string()),
                ("dev.pb.project".to_string(), spec.project_id),
                ("dev.pb.role".to_string(), "cache".to_string()),
                ("dev.pb.cache".to_string(), spec.logical_id),
                ("dev.pb.fingerprint".to_string(), spec.provenance_sha256),
                (
                    "dev.pb.trust".to_string(),
                    cache_trust_name(spec.trust).to_string(),
                ),
            ]),
        });
        if let Err(error) = ensure_result {
            record.preparing_owner = None;
            save_record_atomic(&path, &record)?;
            return Err(error)
                .with_context(|| format!("failed to prepare cache volume {}", record.volume_name));
        }
        record.preparing_owner = None;
        record.active_attachments = record.active_attachments.saturating_add(1);
        record.last_used_at_ms = now_millis();
        save_record_atomic(&path, &record)?;
        Ok(CacheAttachment {
            manager: self,
            volume_name: spec.volume_name,
            target: spec.target,
        })
    }

    pub fn reconcile(&self) -> Result<()> {
        let _operation = self
            .operation_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("cache manager lock is poisoned"))?;
        let _manager_lock = self.acquire_manager_file_lock()?;
        if !self.records_dir().exists() {
            return Ok(());
        }
        for path in record_paths(&self.records_dir())? {
            if let Some(mut record) = load_record(&path)? {
                record.active_attachments = 0;
                record.preparing_owner = None;
                save_record_atomic(&path, &record)?;
            }
        }
        Ok(())
    }

    /// Re-attaches a cache named by a durable session lease after daemon restart.
    pub fn attach_existing(&'static self, volume_name: &str) -> Result<CacheAttachment> {
        let _operation = self
            .operation_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("cache manager lock is poisoned"))?;
        let _manager_lock = self.acquire_manager_file_lock()?;
        let path = self.record_path(volume_name);
        let mut record = load_record(&path)?
            .with_context(|| format!("session references unknown cache volume {volume_name}"))?;
        record.active_attachments = record.active_attachments.saturating_add(1);
        record.last_used_at_ms = now_millis();
        save_record_atomic(&path, &record)?;
        Ok(CacheAttachment {
            manager: self,
            volume_name: volume_name.to_string(),
            target: record.target,
        })
    }

    pub fn update_size_estimate(&self, volume_name: &str, bytes: u64) -> Result<()> {
        let _operation = self
            .operation_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("cache manager lock is poisoned"))?;
        let _manager_lock = self.acquire_manager_file_lock()?;
        let path = self.record_path(volume_name);
        let mut record = load_record(&path)?.context("cache record does not exist")?;
        record.size_estimate_bytes = bytes;
        save_record_atomic(&path, &record)
    }

    pub fn gc(
        &self,
        runtime: &dyn ContainerRuntime,
        max_unused_age: Duration,
        max_total_bytes: u64,
    ) -> Result<Vec<String>> {
        let _operation = self
            .operation_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("cache manager lock is poisoned"))?;
        let _manager_lock = self.acquire_manager_file_lock()?;
        if !self.records_dir().exists() {
            return Ok(Vec::new());
        }
        let mut records = Vec::new();
        for path in record_paths(&self.records_dir())? {
            if let Some(record) = load_record(&path)? {
                records.push((path, record));
            }
        }
        records.sort_by_key(|(_, record)| record.last_used_at_ms);
        let mut total = records
            .iter()
            .map(|(_, record)| record.size_estimate_bytes)
            .sum::<u64>();
        let cutoff =
            now_millis().saturating_sub(max_unused_age.as_millis().min(u64::MAX as u128) as u64);
        let mut removed = Vec::new();
        for (path, record) in records {
            if record.active_attachments != 0 || record.preparing_owner.is_some() {
                continue;
            }
            let over_age = record.last_used_at_ms < cutoff;
            let over_budget = total > max_total_bytes;
            let over_individual_quota = record
                .max_bytes
                .is_some_and(|quota| record.size_estimate_bytes > quota);
            if !over_age && !over_budget && !over_individual_quota {
                continue;
            }
            let recorded_runtime;
            let cache_runtime: &dyn ContainerRuntime = if record.runtime_binary.is_empty()
                || runtime.info()?.binary == record.runtime_binary
            {
                runtime
            } else {
                recorded_runtime = crate::container::runtime_for_binary(&record.runtime_binary)?;
                recorded_runtime.as_ref()
            };
            cache_runtime
                .remove_volume(&record.volume_name)
                .with_context(|| format!("failed to remove cache volume {}", record.volume_name))?;
            total = total.saturating_sub(record.size_estimate_bytes);
            std::fs::remove_file(path)?;
            removed.push(record.volume_name);
        }
        Ok(removed)
    }

    pub fn record(&self, volume_name: &str) -> Result<Option<CacheRecord>> {
        load_record(&self.record_path(volume_name))
    }

    fn detach(&self, volume_name: &str) -> Result<()> {
        let _operation = self
            .operation_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("cache manager lock is poisoned"))?;
        let _manager_lock = self.acquire_manager_file_lock()?;
        let path = self.record_path(volume_name);
        let Some(mut record) = load_record(&path)? else {
            return Ok(());
        };
        record.active_attachments = record.active_attachments.saturating_sub(1);
        record.last_used_at_ms = now_millis();
        save_record_atomic(&path, &record)
    }

    fn records_dir(&self) -> PathBuf {
        self.state_root.join("caches").join("records")
    }

    fn locks_dir(&self) -> PathBuf {
        self.state_root.join("caches").join("locks")
    }

    fn record_path(&self, volume_name: &str) -> PathBuf {
        self.records_dir().join(format!(
            "{}.json",
            crate::environment_lock::sha256(volume_name.as_bytes())
        ))
    }

    fn lock_path(&self, volume_name: &str) -> PathBuf {
        self.locks_dir().join(format!(
            "{}.lock",
            crate::environment_lock::sha256(volume_name.as_bytes())
        ))
    }

    fn acquire_manager_file_lock(&self) -> Result<PreparationLock> {
        PreparationLock::acquire(self.state_root.join("caches").join("manager.lock"))
    }
}

static GLOBAL_CACHE_MANAGER: OnceLock<CacheManager> = OnceLock::new();

pub fn global_cache_manager() -> &'static CacheManager {
    GLOBAL_CACHE_MANAGER.get_or_init(|| {
        CacheManager::new(
            crate::session_workspace::default_state_root()
                .unwrap_or_else(|_| PathBuf::from(".pb/state")),
        )
    })
}

fn validate_spec(spec: &CacheSpec) -> Result<()> {
    if spec.volume_name.trim().is_empty()
        || spec.logical_id.trim().is_empty()
        || !spec.target.starts_with('/')
        || spec.preparing_session.trim().is_empty()
    {
        bail!("cache specification is incomplete or invalid");
    }
    if spec.max_bytes == Some(0) {
        bail!("cache max_bytes must be greater than zero");
    }
    Ok(())
}

fn validate_record_matches(record: &CacheRecord, spec: &CacheSpec) -> Result<()> {
    if record.version != CACHE_RECORD_VERSION
        || record.volume_name != spec.volume_name
        || record.logical_id != spec.logical_id
        || record.target != spec.target
        || record.project_id != spec.project_id
        || record.environment_lock_sha256 != spec.environment_lock_sha256
        || record.provenance_sha256 != spec.provenance_sha256
        || record.trust != spec.trust
        || record.max_bytes != spec.max_bytes
    {
        bail!("cache ownership record does not match requested cache specification");
    }
    Ok(())
}

fn cache_trust_name(trust: CacheTrustClass) -> &'static str {
    match trust {
        CacheTrustClass::Download => "download",
        CacheTrustClass::Toolchain => "toolchain",
        CacheTrustClass::ProjectExecutable => "project-executable",
        CacheTrustClass::LspIndex => "lsp-index",
    }
}

struct PreparationLock {
    _lock: crate::state_lock::StateFileLock,
}

impl PreparationLock {
    fn acquire(path: PathBuf) -> Result<Self> {
        Ok(Self {
            _lock: crate::state_lock::StateFileLock::acquire(path, PREPARATION_LOCK_TIMEOUT)?,
        })
    }
}

fn record_paths(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = std::fs::read_dir(root)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn load_record(path: &Path) -> Result<Option<CacheRecord>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path)?;
    let record: CacheRecord = serde_json::from_slice(&bytes)?;
    if record.version != CACHE_RECORD_VERSION {
        bail!("unsupported cache record version {}", record.version);
    }
    Ok(Some(record))
}

fn save_record_atomic(path: &Path, record: &CacheRecord) -> Result<()> {
    let parent = path.parent().context("cache record has no parent")?;
    std::fs::create_dir_all(parent)?;
    let temp = parent.join(format!(
        ".cache.{}.{}.tmp",
        std::process::id(),
        crate::environment_lock::sha256(record.volume_name.as_bytes())
    ));
    std::fs::write(&temp, serde_json::to_vec_pretty(record)?)?;
    std::fs::rename(&temp, path)
        .with_context(|| format!("failed to replace cache record {}", path.display()))
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        ContainerLaunchSpec, ManagedResource, NetworkSpec, RuntimeCapabilities, RuntimeInfo,
        RuntimeKind,
    };
    use std::sync::Mutex;
    use tempfile::TempDir;

    #[derive(Default)]
    struct FakeRuntime {
        events: Mutex<Vec<String>>,
        fail_ensure: bool,
    }

    impl ContainerRuntime for FakeRuntime {
        fn info(&self) -> Result<RuntimeInfo> {
            Ok(RuntimeInfo {
                kind: RuntimeKind::Apple,
                binary: "fake".to_string(),
                version: "1".to_string(),
                capabilities: RuntimeCapabilities {
                    internal_networks: true,
                    named_volumes: true,
                    labels: true,
                    resource_limits: true,
                },
            })
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
            Ok("image".into())
        }
        fn create(&self, _: &ContainerLaunchSpec) -> Result<String> {
            Ok("container".into())
        }
        fn exec(&self, _: &str, _: &str) -> Result<String> {
            Ok(String::new())
        }
        fn remove(&self, _: &str) -> Result<()> {
            Ok(())
        }
        fn create_internal_network(&self, _: &NetworkSpec) -> Result<()> {
            Ok(())
        }
        fn remove_network(&self, _: &str) -> Result<()> {
            Ok(())
        }
        fn ensure_volume(&self, spec: &VolumeSpec) -> Result<()> {
            if self.fail_ensure {
                bail!("injected volume preparation failure");
            }
            self.events
                .lock()
                .unwrap()
                .push(format!("ensure {}", spec.name));
            Ok(())
        }
        fn remove_volume(&self, volume: &str) -> Result<()> {
            self.events.lock().unwrap().push(format!("remove {volume}"));
            Ok(())
        }
        fn list_managed_containers(&self) -> Result<Vec<ManagedResource>> {
            Ok(Vec::new())
        }
    }

    fn spec() -> CacheSpec {
        CacheSpec {
            volume_name: "pb-cache-one".to_string(),
            logical_id: "cargo".to_string(),
            target: "/workspace/target".to_string(),
            project_id: "project".to_string(),
            environment_lock_sha256: crate::environment_lock::sha256(b"environment"),
            provenance_sha256: crate::environment_lock::sha256(b"provenance"),
            trust: CacheTrustClass::ProjectExecutable,
            max_bytes: Some(100),
            preparing_session: "session".to_string(),
        }
    }

    #[test]
    fn attachments_are_accounted_and_gc_never_removes_active_cache() {
        let dir = TempDir::new().unwrap();
        let manager = Box::leak(Box::new(CacheManager::new(dir.path().to_path_buf())));
        let runtime = FakeRuntime::default();
        let attachment = manager.acquire(&runtime, spec()).unwrap();
        manager
            .update_size_estimate(attachment.volume_name(), 200)
            .unwrap();
        assert_eq!(
            manager
                .record(attachment.volume_name())
                .unwrap()
                .unwrap()
                .active_attachments,
            1
        );
        assert!(manager.gc(&runtime, Duration::ZERO, 0).unwrap().is_empty());
        drop(attachment);
        assert_eq!(
            manager.gc(&runtime, Duration::ZERO, 0).unwrap(),
            vec!["pb-cache-one"]
        );
    }

    #[test]
    fn incompatible_cache_record_fails_closed() {
        let dir = TempDir::new().unwrap();
        let manager = Box::leak(Box::new(CacheManager::new(dir.path().to_path_buf())));
        let runtime = FakeRuntime::default();
        let attachment = manager.acquire(&runtime, spec()).unwrap();
        drop(attachment);
        let mut changed = spec();
        changed.project_id = "other".to_string();
        assert!(manager.acquire(&runtime, changed).is_err());
    }

    #[test]
    fn failed_preparation_clears_the_durable_owner_for_retry_and_gc() {
        let dir = TempDir::new().unwrap();
        let manager = Box::leak(Box::new(CacheManager::new(dir.path().to_path_buf())));
        let runtime = FakeRuntime {
            fail_ensure: true,
            ..Default::default()
        };
        let cache_spec = spec();

        assert!(manager.acquire(&runtime, cache_spec.clone()).is_err());
        let record = manager.record(&cache_spec.volume_name).unwrap().unwrap();
        assert_eq!(record.preparing_owner, None);
        assert_eq!(record.active_attachments, 0);
    }
}
