use std::collections::BTreeMap;
use std::fs;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant, UNIX_EPOCH};

use anyhow::Result;
use sha2::{Digest, Sha256};

use super::{FlashMoeEngine, FlashMoePlan, load};

const DEFAULT_RESIDENT_MODELS: usize = 2;
const DEFAULT_IDLE_SECONDS: u64 = 15 * 60;

#[derive(Debug, Clone)]
pub struct FlashMoeRuntimeHandle {
    engine: Arc<Mutex<FlashMoeEngine>>,
    reused: bool,
}

impl FlashMoeRuntimeHandle {
    pub fn lock(&self) -> Result<MutexGuard<'_, FlashMoeEngine>> {
        self.engine
            .lock()
            .map_err(|_| anyhow::anyhow!("shared FlashMoe runtime lock is poisoned"))
    }

    pub fn reused(&self) -> bool {
        self.reused
    }
}

#[derive(Debug)]
struct ResidentRuntime {
    engine: Arc<Mutex<FlashMoeEngine>>,
    last_used: Instant,
}

#[derive(Debug, Default)]
struct RuntimePool {
    entries: BTreeMap<String, ResidentRuntime>,
}

fn global_pool() -> &'static Mutex<RuntimePool> {
    static POOL: OnceLock<Mutex<RuntimePool>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(RuntimePool::default()))
}

pub fn load_shared(plan: &FlashMoePlan) -> Result<FlashMoeRuntimeHandle> {
    let key = runtime_key(plan);
    {
        let mut pool = global_pool()
            .lock()
            .map_err(|_| anyhow::anyhow!("FlashMoe runtime pool lock is poisoned"))?;
        prune_pool(&mut pool, false);
        if let Some(entry) = pool.entries.get_mut(&key) {
            entry.last_used = Instant::now();
            return Ok(FlashMoeRuntimeHandle {
                engine: Arc::clone(&entry.engine),
                reused: true,
            });
        }
    }

    // Model construction is intentionally outside the pool lock. Another model
    // and cache-only pool lookup must not wait behind dense/Metal initialization.
    let loaded = Arc::new(Mutex::new(load(plan)?));
    let mut pool = global_pool()
        .lock()
        .map_err(|_| anyhow::anyhow!("FlashMoe runtime pool lock is poisoned"))?;
    if let Some(entry) = pool.entries.get_mut(&key) {
        entry.last_used = Instant::now();
        return Ok(FlashMoeRuntimeHandle {
            engine: Arc::clone(&entry.engine),
            reused: true,
        });
    }
    pool.entries.insert(
        key,
        ResidentRuntime {
            engine: Arc::clone(&loaded),
            last_used: Instant::now(),
        },
    );
    prune_pool(&mut pool, true);
    Ok(FlashMoeRuntimeHandle {
        engine: loaded,
        reused: false,
    })
}

pub fn reap_idle_shared_runtimes() -> Result<usize> {
    let mut pool = global_pool()
        .lock()
        .map_err(|_| anyhow::anyhow!("FlashMoe runtime pool lock is poisoned"))?;
    let before = pool.entries.len();
    prune_pool(&mut pool, false);
    Ok(before.saturating_sub(pool.entries.len()))
}

fn prune_pool(pool: &mut RuntimePool, enforce_count: bool) {
    let idle = resident_idle_duration();
    pool.entries.retain(|_, entry| {
        Arc::strong_count(&entry.engine) > 1 || entry.last_used.elapsed() < idle
    });
    if !enforce_count {
        return;
    }
    let limit = resident_model_limit();
    while pool.entries.len() > limit {
        let candidate = pool
            .entries
            .iter()
            .filter(|(_, entry)| Arc::strong_count(&entry.engine) == 1)
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone());
        let Some(candidate) = candidate else {
            break;
        };
        pool.entries.remove(&candidate);
    }
}

fn resident_model_limit() -> usize {
    std::env::var("PB_FLASHMOE_RESIDENT_MODELS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_RESIDENT_MODELS)
}

fn resident_idle_duration() -> Duration {
    Duration::from_secs(
        std::env::var("PB_FLASHMOE_IDLE_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_IDLE_SECONDS),
    )
}

fn runtime_key(plan: &FlashMoePlan) -> String {
    let mut digest = Sha256::new();
    digest.update(plan.model.as_bytes());
    digest.update(plan.runtime_dir.to_string_lossy().as_bytes());
    digest.update(plan.quantization.as_str().as_bytes());
    digest.update(format!("{:?}", plan.routing_policy).as_bytes());
    for path in [
        &plan.model_config,
        &plan.tensor_manifest,
        &plan.non_expert_weights,
        &plan.tokenizer,
        &plan.tokenizer_config,
        &plan.chat_template,
    ] {
        digest.update(path.to_string_lossy().as_bytes());
        if let Ok(metadata) = fs::metadata(path) {
            digest.update(metadata.len().to_le_bytes());
            let modified = metadata
                .modified()
                .ok()
                .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
                .unwrap_or_else(|| Duration::from_secs(0));
            digest.update(modified.as_secs().to_le_bytes());
            digest.update(modified.subsec_nanos().to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::flashmoe::{QWEN35_MODEL, plan_unchecked};
    use tempfile::tempdir;

    #[test]
    fn runtime_key_changes_when_runtime_artifact_identity_changes() {
        let root = tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, root.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        fs::write(&plan.model_config, b"one").unwrap();
        let first = runtime_key(&plan);
        fs::write(&plan.model_config, b"a different config").unwrap();
        let second = runtime_key(&plan);
        assert_ne!(first, second);
    }

    #[test]
    fn default_pool_limits_are_finite() {
        assert!(resident_model_limit() > 0);
        assert!(resident_idle_duration() > Duration::ZERO);
    }
}
