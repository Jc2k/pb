use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, File, FileTimes, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

use super::planning::FlashMoePlan;
use super::state::{
    FlashMoeGenerationState, FlashMoeLinearAttentionLayerSnapshot,
    FlashMoeLinearAttentionSessionSnapshot, FlashMoePromptTokenRecord, FlashMoeSessionState,
    KvCache, reusable_session_prefix_len,
};
use super::types::PromptCacheSource;
use crate::inference::PromptCacheMissReason;

pub(crate) const CACHE_VERSION: &str = "flashmoe-session-v1";
const MAGIC: &[u8; 8] = b"PBFMKV01";
const MAX_TOKENS: usize = 1_000_000;
const MAX_VECTOR_FLOATS: usize = 32 * 1024 * 1024;
const MAX_SESSION_MANIFEST_BYTES: u64 = 1024 * 1024;
const CACHE_WRITE_LOCK: &str = ".write.lock";
const CACHE_WRITE_LOCK_TIMEOUT: Duration = Duration::from_secs(10);

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[derive(Debug)]
pub(super) struct FlashMoeDiskCache {
    storage_root: PathBuf,
    root: PathBuf,
    fingerprint: [u8; 32],
    layers: usize,
    max_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionManifest {
    version: String,
    checkpoints: Vec<String>,
}

impl FlashMoeDiskCache {
    pub(super) fn from_plan(
        plan: &FlashMoePlan,
        layers: usize,
        settings: &crate::config::ResolvedSessionCacheConfig,
    ) -> Option<Self> {
        if !settings.enabled {
            return None;
        }
        let storage_root = settings.root.as_ref()?.clone();
        let root = storage_root
            .join(CACHE_VERSION)
            .join(model_fingerprint_hex(plan));
        Some(Self {
            storage_root,
            root,
            fingerprint: model_fingerprint(plan),
            layers,
            max_bytes: settings.max_bytes,
        })
    }

    pub(super) fn token_key(tokens: &[u32]) -> String {
        let mut digest = Sha256::new();
        digest.update(CACHE_VERSION.as_bytes());
        for token in tokens {
            digest.update(token.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    pub(super) fn load_prefix(&self, tokens: &[u32]) -> Result<Option<FlashMoeCachedSessionState>> {
        if !self.ensure_namespace(false)? {
            return Ok(None);
        }
        self.load_checkpoint(&Self::token_key(tokens))
    }

    pub(super) fn load_session(&self, session_id: &str) -> Result<Vec<FlashMoeCachedSessionState>> {
        if !self.ensure_namespace(false)? {
            return Ok(Vec::new());
        }
        let path = self.session_manifest_path(session_id);
        let mut file = match open_cache_file(&path) {
            Ok(Some(file)) => file,
            Ok(None) => return Ok(Vec::new()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", path.display()));
            }
        };
        let bytes = read_cache_file_contents(&mut file, &path, MAX_SESSION_MANIFEST_BYTES)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let manifest: SessionManifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        if manifest.version != CACHE_VERSION {
            return Ok(Vec::new());
        }
        let mut checkpoints = Vec::new();
        for key in manifest.checkpoints {
            if let Some(checkpoint) = self.load_checkpoint(&key)? {
                checkpoints.push(checkpoint);
            }
        }
        refresh_cache_file_recency(&file, &path);
        Ok(checkpoints)
    }

    pub(super) fn persist_prefix(&self, state: &FlashMoeCachedSessionState) -> Result<()> {
        let _lock = self.acquire_write_lock()?;
        let _ = self.persist_checkpoint(state)?;
        Ok(())
    }

    pub(super) fn persist_session(
        &self,
        session_id: &str,
        states: &[FlashMoeCachedSessionState],
    ) -> Result<()> {
        let _lock = self.acquire_write_lock()?;
        let mut ordered = states.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|state| std::cmp::Reverse(state.cpu.tokens.len()));
        let mut keys = Vec::new();
        let mut remaining = self.max_bytes.saturating_sub(64 * 1024);
        for state in ordered {
            let estimated = checkpoint_size(state);
            if estimated > remaining {
                continue;
            }
            if let Some(key) = self.persist_checkpoint(state)? {
                keys.push(key);
                remaining = remaining.saturating_sub(estimated);
            }
        }
        if keys.is_empty() {
            return Ok(());
        }
        secure_directory(&self.root)?;
        let path = self.session_manifest_path(session_id);
        let mut preserved = keys
            .iter()
            .map(|key| self.checkpoint_path(key))
            .collect::<Vec<_>>();
        let manifest = SessionManifest {
            version: CACHE_VERSION.to_string(),
            checkpoints: keys,
        };
        atomic_write(&path, &serde_json::to_vec(&manifest)?)?;
        preserved.push(path);
        self.prune(&preserved)?;
        Ok(())
    }

    fn acquire_write_lock(&self) -> Result<crate::state_lock::StateFileLock> {
        self.ensure_namespace(true)?;
        crate::state_lock::StateFileLock::acquire(
            self.root.join(CACHE_WRITE_LOCK),
            CACHE_WRITE_LOCK_TIMEOUT,
        )
    }

    /// Validate each cache-owned path component before reading or creating files.
    ///
    /// `storage_root` is the user-selected trust boundary. Its parents retain normal OS path
    /// semantics, but neither it nor a version/model namespace below it may be a symlink.
    fn ensure_namespace(&self, create: bool) -> Result<bool> {
        if !validate_cache_directory(&self.storage_root, create)? {
            return Ok(false);
        }
        let relative = self
            .root
            .strip_prefix(&self.storage_root)
            .with_context(|| {
                format!(
                    "FlashMoe cache namespace {} is outside configured storage root {}",
                    self.root.display(),
                    self.storage_root.display()
                )
            })?;
        let mut current = self.storage_root.clone();
        for component in relative.components() {
            let std::path::Component::Normal(component) = component else {
                bail!(
                    "FlashMoe cache namespace contains an unsafe component: {}",
                    self.root.display()
                );
            };
            current.push(component);
            if !validate_cache_directory(&current, create)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn load_checkpoint(&self, key: &str) -> Result<Option<FlashMoeCachedSessionState>> {
        if key.len() != 64 || !key.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Ok(None);
        }
        let path = self.checkpoint_path(key);
        let file = match open_cache_file(&path) {
            Ok(Some(file)) => file,
            Ok(None) => return Ok(None),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error).with_context(|| format!("failed to open {}", path.display()));
            }
        };
        if file.metadata()?.len() > self.max_bytes {
            tracing::warn!(
                path = %path.display(),
                max_bytes = self.max_bytes,
                "ignored FlashMoe session checkpoint larger than the configured cache budget"
            );
            return Ok(None);
        }
        let mut reader = BufReader::new(file);
        let state = read_checkpoint(&mut reader, self.fingerprint, self.layers)
            .with_context(|| format!("failed to restore {}", path.display()))?;
        if Self::token_key(&state.cpu.tokens) != key {
            return Ok(None);
        }
        refresh_cache_file_recency(reader.get_ref(), &path);
        Ok(Some(state))
    }

    fn persist_checkpoint(&self, state: &FlashMoeCachedSessionState) -> Result<Option<String>> {
        let estimated = checkpoint_size(state);
        if estimated > self.max_bytes {
            tracing::warn!(
                estimated_bytes = estimated,
                max_bytes = self.max_bytes,
                tokens = state.cpu.tokens.len(),
                "FlashMoe session checkpoint exceeds the configured disk-cache budget"
            );
            return Ok(None);
        }
        let key = Self::token_key(&state.cpu.tokens);
        let path = self.checkpoint_path(&key);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                bail!("refused symlink FlashMoe checkpoint {}", path.display());
            }
            Ok(metadata) if metadata.is_file() => {
                if let Some(file) = open_cache_file(&path)? {
                    refresh_cache_file_recency(&file, &path);
                }
                return Ok(Some(key));
            }
            Ok(_) => bail!("FlashMoe checkpoint path is not a file: {}", path.display()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let parent = path.parent().context("checkpoint path has no parent")?;
        secure_directory(&self.root)?;
        secure_directory(parent)?;
        let mut temporary = NamedTempFile::new_in(parent)
            .with_context(|| format!("failed to create temporary file in {}", parent.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            temporary
                .as_file()
                .set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        {
            let mut writer = BufWriter::new(temporary.as_file_mut());
            write_checkpoint(&mut writer, self.fingerprint, state)?;
            writer.flush()?;
        }
        temporary.as_file().sync_all()?;
        match temporary.persist_noclobber(&path) {
            Ok(_) => sync_directory(parent)?,
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error.error)
                    .with_context(|| format!("failed to persist {}", path.display()));
            }
        }
        self.prune(std::slice::from_ref(&path))?;
        Ok(Some(key))
    }

    fn checkpoint_path(&self, key: &str) -> PathBuf {
        self.root.join("checkpoints").join(format!("{key}.bin"))
    }

    fn session_manifest_path(&self, session_id: &str) -> PathBuf {
        let mut digest = Sha256::new();
        digest.update(CACHE_VERSION.as_bytes());
        digest.update(session_id.as_bytes());
        self.root
            .join("sessions")
            .join(format!("{:x}.json", digest.finalize()))
    }

    fn prune(&self, preserve: &[PathBuf]) -> Result<()> {
        if !self.root.is_dir() {
            return Ok(());
        }
        let mut files = Vec::new();
        collect_cache_files(&self.root, &mut files)?;
        let mut total = files.iter().map(|(_, bytes, _)| *bytes).sum::<u64>();
        files.sort_by_key(|(_, _, modified)| *modified);
        for (path, bytes, _) in files {
            if total <= self.max_bytes {
                break;
            }
            if preserve.iter().any(|preserve| preserve == &path) {
                continue;
            }
            if let Some(key) = path
                .file_stem()
                .and_then(|value| value.to_str())
                .filter(|_| path.extension().and_then(|value| value.to_str()) == Some("bin"))
            {
                self.remove_checkpoint_from_manifests(key)?;
            }
            match fs::remove_file(&path) {
                Ok(()) => {
                    if let Some(parent) = path.parent() {
                        sync_directory(parent)?;
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("failed to prune {}", path.display()));
                }
            }
            total = total.saturating_sub(bytes);
        }
        Ok(())
    }

    fn remove_checkpoint_from_manifests(&self, key: &str) -> Result<()> {
        let sessions = self.root.join("sessions");
        let entries = match fs::read_dir(&sessions) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_symlink() || !entry.file_type()?.is_file() {
                continue;
            }
            let path = entry.path();
            let Some(bytes) = read_cache_file(&path, MAX_SESSION_MANIFEST_BYTES)? else {
                continue;
            };
            let Ok(mut manifest) = serde_json::from_slice::<SessionManifest>(&bytes) else {
                continue;
            };
            if manifest.version != CACHE_VERSION
                || !manifest
                    .checkpoints
                    .iter()
                    .any(|checkpoint| checkpoint == key)
            {
                continue;
            }
            manifest.checkpoints.retain(|checkpoint| checkpoint != key);
            if manifest.checkpoints.is_empty() {
                fs::remove_file(&path)?;
                sync_directory(&sessions)?;
            } else {
                atomic_write(&path, &serde_json::to_vec(&manifest)?)?;
            }
        }
        Ok(())
    }
}

fn write_checkpoint(
    writer: &mut impl Write,
    fingerprint: [u8; 32],
    state: &FlashMoeCachedSessionState,
) -> Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_all(&fingerprint)?;
    let token_digest = Sha256::digest(tokens_as_bytes(&state.cpu.tokens));
    writer.write_all(&token_digest)?;
    write_usize(writer, state.cpu.tokens.len())?;
    for token in &state.cpu.tokens {
        writer.write_all(&token.to_le_bytes())?;
    }
    write_usize(writer, state.cpu.kv_cache.layers)?;
    for layer in 0..state.cpu.kv_cache.layers {
        for position in 0..state.cpu.tokens.len() {
            let full = state.cpu.kv_cache.kv[layer][position].as_ref();
            let mla = state.cpu.kv_cache.mla_kv[layer][position].as_ref();
            let flags = u8::from(full.is_some()) | (u8::from(mla.is_some()) << 1);
            writer.write_all(&[flags])?;
            if let Some((key, value)) = full {
                write_f32_slice(writer, key)?;
                write_f32_slice(writer, value)?;
            }
            if let Some((latent, rotary)) = mla {
                write_f32_slice(writer, latent)?;
                write_f32_slice(writer, rotary)?;
            }
        }
    }
    write_f32_slice(writer, &state.cpu.last_hidden)?;
    write_usize(writer, state.recurrent.len())?;
    for layer in 0..state.recurrent.len() {
        let Some(snapshot) = state.recurrent.layer(layer) else {
            writer.write_all(&[0])?;
            continue;
        };
        writer.write_all(&[1])?;
        write_usize(writer, snapshot.state().conv_output_len())?;
        write_usize(writer, snapshot.state().output_len())?;
        write_f32_slice(writer, snapshot.conv_state())?;
        write_f32_slice(writer, snapshot.ssm_state())?;
    }
    Ok(())
}

fn read_checkpoint(
    mut reader: impl Read,
    expected_fingerprint: [u8; 32],
    expected_layers: usize,
) -> Result<FlashMoeCachedSessionState> {
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("incompatible FlashMoe session-cache magic");
    }
    let mut fingerprint = [0_u8; 32];
    reader.read_exact(&mut fingerprint)?;
    if fingerprint != expected_fingerprint {
        bail!("FlashMoe session-cache model fingerprint changed");
    }
    let mut expected_token_digest = [0_u8; 32];
    reader.read_exact(&mut expected_token_digest)?;
    let token_count = read_usize(&mut reader, MAX_TOKENS, "token count")?;
    let mut tokens = Vec::with_capacity(token_count);
    for _ in 0..token_count {
        let mut bytes = [0_u8; 4];
        reader.read_exact(&mut bytes)?;
        tokens.push(u32::from_le_bytes(bytes));
    }
    if Sha256::digest(tokens_as_bytes(&tokens)).as_slice() != expected_token_digest {
        bail!("FlashMoe session-cache token checksum mismatch");
    }
    let layers = read_usize(&mut reader, 4096, "layer count")?;
    if layers != expected_layers {
        bail!("FlashMoe session-cache layer count changed");
    }
    let mut kv_cache = KvCache::new(layers, token_count.max(1));
    for (position, token) in tokens.iter().copied().enumerate() {
        kv_cache.record_prompt_token(position, token)?;
    }
    for layer in 0..layers {
        for position in 0..token_count {
            let mut flags = [0_u8; 1];
            reader.read_exact(&mut flags)?;
            if flags[0] & 1 != 0 {
                let key = read_f32_vec(&mut reader)?;
                let value = read_f32_vec(&mut reader)?;
                kv_cache.record_kv(position, layer, key, value)?;
            }
            if flags[0] & 2 != 0 {
                let latent = read_f32_vec(&mut reader)?;
                let rotary = read_f32_vec(&mut reader)?;
                kv_cache.record_mla_kv(position, layer, latent, rotary)?;
            }
            if flags[0] & !3 != 0 {
                bail!("FlashMoe session-cache entry has unknown flags");
            }
        }
    }
    let last_hidden = read_f32_vec(&mut reader)?;
    let recurrent_layers = read_usize(&mut reader, 4096, "recurrent layer count")?;
    if recurrent_layers != expected_layers {
        bail!("FlashMoe session-cache recurrent layer count changed");
    }
    let mut recurrent = Vec::with_capacity(recurrent_layers);
    for layer in 0..recurrent_layers {
        let mut present = [0_u8; 1];
        reader.read_exact(&mut present)?;
        match present[0] {
            0 => recurrent.push(None),
            1 => {
                let conv_output_len =
                    read_usize(&mut reader, MAX_VECTOR_FLOATS, "conv output length")?;
                let output_len = read_usize(&mut reader, MAX_VECTOR_FLOATS, "output length")?;
                let conv_state = read_f32_vec(&mut reader)?;
                let ssm_state = read_f32_vec(&mut reader)?;
                recurrent.push(Some(FlashMoeLinearAttentionLayerSnapshot::new(
                    layer,
                    conv_state,
                    ssm_state,
                    conv_output_len,
                    output_len,
                )?));
            }
            _ => bail!("FlashMoe session-cache recurrent entry has invalid presence flag"),
        }
    }
    Ok(FlashMoeCachedSessionState {
        cpu: FlashMoeSessionState::new(tokens, kv_cache, last_hidden),
        recurrent: FlashMoeLinearAttentionSessionSnapshot::new(recurrent)?,
    })
}

fn checkpoint_size(state: &FlashMoeCachedSessionState) -> u64 {
    let mut bytes = 8 + 32 + 32 + 8 + (state.cpu.tokens.len() as u64 * 4) + 8;
    for layer in 0..state.cpu.kv_cache.layers {
        for position in 0..state.cpu.tokens.len() {
            bytes += 1;
            if let Some((key, value)) = &state.cpu.kv_cache.kv[layer][position] {
                bytes += 16 + ((key.len() + value.len()) as u64 * 4);
            }
            if let Some((latent, rotary)) = &state.cpu.kv_cache.mla_kv[layer][position] {
                bytes += 16 + ((latent.len() + rotary.len()) as u64 * 4);
            }
        }
    }
    bytes += 8 + state.cpu.last_hidden.len() as u64 * 4 + 8;
    for layer in 0..state.recurrent.len() {
        bytes += 1;
        if let Some(snapshot) = state.recurrent.layer(layer) {
            bytes += 32 + (snapshot.conv_state().len() + snapshot.ssm_state().len()) as u64 * 4;
        }
    }
    bytes
}

fn write_usize(writer: &mut impl Write, value: usize) -> Result<()> {
    writer.write_all(&u64::try_from(value)?.to_le_bytes())?;
    Ok(())
}

fn read_usize(reader: &mut impl Read, max: usize, label: &str) -> Result<usize> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    let value = usize::try_from(u64::from_le_bytes(bytes))?;
    if value > max {
        bail!("FlashMoe session-cache {label} {value} exceeds limit {max}");
    }
    Ok(value)
}

fn write_f32_slice(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    write_usize(writer, values.len())?;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_f32_vec(reader: &mut impl Read) -> Result<Vec<f32>> {
    let len = read_usize(reader, MAX_VECTOR_FLOATS, "float vector length")?;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        let mut bytes = [0_u8; 4];
        reader.read_exact(&mut bytes)?;
        values.push(f32::from_le_bytes(bytes));
    }
    Ok(values)
}

fn tokens_as_bytes(tokens: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes
}

fn model_fingerprint(plan: &FlashMoePlan) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(CACHE_VERSION.as_bytes());
    digest.update(plan.model.as_bytes());
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
    digest.finalize().into()
}

pub(crate) fn model_fingerprint_hex(plan: &FlashMoePlan) -> String {
    model_fingerprint(plan)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn secure_directory(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("failed to create {}", path.display()))?;
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!(
            "refused non-directory or symlink cache path {}",
            path.display()
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("failed to secure {}", path.display()))?;
    }
    Ok(())
}

fn validate_cache_directory(path: &Path, create: bool) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            bail!(
                "refused non-directory or symlink cache path {}",
                path.display()
            );
        }
        Ok(_) => {
            #[cfg(unix)]
            if create {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(path, fs::Permissions::from_mode(0o700))
                    .with_context(|| format!("failed to secure {}", path.display()))?;
            }
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !create => Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match fs::create_dir(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("failed to create {}", path.display()));
                }
            }
            validate_cache_directory(path, true)
        }
        Err(error) => Err(error)
            .with_context(|| format!("failed to inspect cache directory {}", path.display())),
    }
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("failed to open cache directory {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("failed to sync cache directory {}", path.display()))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

fn open_cache_file(path: &Path) -> std::io::Result<Option<File>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("refused symlink cache file {}", path.display()),
            ));
        }
        Ok(metadata) if !metadata.is_file() => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("cache path is not a regular file: {}", path.display()),
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    options.open(path).map(Some)
}

fn read_cache_file(path: &Path, max_bytes: u64) -> std::io::Result<Option<Vec<u8>>> {
    let Some(mut file) = open_cache_file(path)? else {
        return Ok(None);
    };
    read_cache_file_contents(&mut file, path, max_bytes).map(Some)
}

fn read_cache_file_contents(
    file: &mut File,
    path: &Path,
    max_bytes: u64,
) -> std::io::Result<Vec<u8>> {
    if file.metadata()?.len() > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("cache file exceeds {max_bytes} bytes: {}", path.display()),
        ));
    }
    let capacity = usize::try_from(max_bytes.min(1024 * 1024)).unwrap_or(1024 * 1024);
    let mut bytes = Vec::with_capacity(capacity);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("cache file exceeds {max_bytes} bytes: {}", path.display()),
        ));
    }
    Ok(bytes)
}

fn refresh_cache_file_recency(file: &File, path: &Path) {
    let times = FileTimes::new().set_modified(SystemTime::now());
    if let Err(error) = file.set_times(times) {
        tracing::debug!(
            path = %path.display(),
            %error,
            "could not refresh FlashMoe cache retention recency"
        );
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("cache manifest path has no parent")?;
    secure_directory(parent)?;
    let mut temporary = NamedTempFile::new_in(parent)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        temporary
            .as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to replace {}", path.display()))?;
    sync_directory(parent)?;
    Ok(())
}

fn collect_cache_files(root: &Path, files: &mut Vec<(PathBuf, u64, Duration)>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if entry.file_name() == CACHE_WRITE_LOCK {
            continue;
        }
        if file_type.is_symlink() {
            continue;
        }
        let metadata = entry.metadata()?;
        if file_type.is_dir() {
            collect_cache_files(&path, files)?;
        } else if file_type.is_file() {
            let modified = metadata
                .modified()
                .ok()
                .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
                .unwrap_or_default();
            files.push((path, metadata.len(), modified));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(super) struct FlashMoeSessionCache {
    pub(super) entries: BTreeMap<String, Vec<FlashMoeCachedSessionState>>,
    pub(super) session_order: VecDeque<String>,
    prefixes: BTreeMap<String, FlashMoeCachedSessionState>,
    prefix_sizes: BTreeMap<String, u64>,
    prefix_bytes: u64,
    prefix_order: VecDeque<String>,
    dirty_sessions: BTreeSet<String>,
    dirty_prefixes: BTreeSet<String>,
    disk: Option<FlashMoeDiskCache>,
    memory_session_limit: usize,
    memory_prefix_max_bytes: u64,
    deferred_persistence: super::types::PromptCachePersistenceStats,
}

impl Default for FlashMoeSessionCache {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
            session_order: VecDeque::new(),
            prefixes: BTreeMap::new(),
            prefix_sizes: BTreeMap::new(),
            prefix_bytes: 0,
            prefix_order: VecDeque::new(),
            dirty_sessions: BTreeSet::new(),
            dirty_prefixes: BTreeSet::new(),
            disk: None,
            memory_session_limit: crate::config::DEFAULT_FLASHMOE_MEMORY_SESSIONS,
            memory_prefix_max_bytes: crate::config::DEFAULT_FLASHMOE_MEMORY_PROMPT_ROOT_MAX_BYTES,
            deferred_persistence: super::types::PromptCachePersistenceStats::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct FlashMoeCachedSessionState {
    pub(super) cpu: FlashMoeSessionState<KvCache>,
    pub(super) recurrent: FlashMoeLinearAttentionSessionSnapshot,
}

impl FlashMoeSessionCache {
    pub(crate) fn new(
        disk: Option<FlashMoeDiskCache>,
        memory_session_limit: usize,
        memory_prefix_max_bytes: u64,
    ) -> Self {
        Self {
            disk,
            memory_session_limit,
            memory_prefix_max_bytes,
            ..Self::default()
        }
    }

    #[cfg(test)]
    pub(crate) fn begin_generation(
        &mut self,
        session_id: Option<&str>,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        layers: usize,
    ) -> FlashMoeGenerationState {
        self.begin_generation_with_base(session_id, prompt_tokens, 0, max_tokens, layers)
    }

    pub(crate) fn begin_generation_with_base(
        &mut self,
        session_id: Option<&str>,
        prompt_tokens: Vec<u32>,
        base_prefix_len: usize,
        max_tokens: usize,
        layers: usize,
    ) -> FlashMoeGenerationState {
        let preparation_started = Instant::now();
        let mut disk_read_decode_wall = Duration::ZERO;
        let mut cpu_state_validation_allocation_wall = Duration::ZERO;
        let capacity = prompt_tokens.len() + max_tokens;
        // A harness workflow keeps one logical session id while moving between
        // fresh stage prompts. Once the new prompt diverges, the cached state
        // cannot contribute to this generation and must not remain resident
        // beside the replacement KV cache for the duration of a long prefill.
        if let Some(id) = session_id {
            self.session_order.retain(|existing| existing != id);
        }
        let memory_session = session_id.and_then(|id| self.entries.remove(id));
        let mut incompatible_checkpoint_seen = false;
        let mut cache_unreadable = false;
        let mut cached = memory_session.and_then(|states| {
            let found = states
                .into_iter()
                .filter_map(|state| {
                    reusable_session_prefix_len(&state.cpu.tokens, &prompt_tokens)
                        .map(|prefix_len| (prefix_len, state))
                })
                .max_by_key(|(prefix_len, _)| *prefix_len);
            if found.is_none() {
                incompatible_checkpoint_seen = true;
            }
            found
        });
        let mut cache_source = if cached.is_some() {
            PromptCacheSource::MemorySession
        } else {
            PromptCacheSource::None
        };
        let base_prefix_len = base_prefix_len.min(prompt_tokens.len());
        let base_key = (base_prefix_len > 0)
            .then(|| FlashMoeDiskCache::token_key(&prompt_tokens[..base_prefix_len]));
        if let Some(key) = base_key.as_ref()
            && let Some(state) = self.prefixes.get(key).cloned()
            && reusable_session_prefix_len(&state.cpu.tokens, &prompt_tokens).is_some_and(
                |prefix_len| {
                    cached
                        .as_ref()
                        .is_none_or(|(cached_len, _)| prefix_len > *cached_len)
                },
            )
        {
            self.touch_prefix(key);
            cached = Some((state.cpu.tokens.len(), state));
            cache_source = PromptCacheSource::MemoryPrefix;
        }
        let restore_started = Instant::now();
        let mut used_disk = false;
        if cached.is_none()
            && let (Some(id), Some(disk)) = (session_id, self.disk.as_ref())
        {
            let disk_started = Instant::now();
            match disk.load_session(id) {
                Ok(states) => {
                    disk_read_decode_wall += disk_started.elapsed();
                    let validation_started = Instant::now();
                    let had_states = !states.is_empty();
                    if let Some(found) = states
                        .into_iter()
                        .filter_map(|state| {
                            reusable_session_prefix_len(&state.cpu.tokens, &prompt_tokens)
                                .map(|prefix_len| (prefix_len, state))
                        })
                        .max_by_key(|(prefix_len, _)| *prefix_len)
                    {
                        cached = Some(found);
                        cache_source = PromptCacheSource::DiskSession;
                        used_disk = true;
                    } else if had_states {
                        incompatible_checkpoint_seen = true;
                    }
                    cpu_state_validation_allocation_wall += validation_started.elapsed();
                }
                Err(error) => {
                    disk_read_decode_wall += disk_started.elapsed();
                    cache_unreadable = true;
                    tracing::warn!(
                        session = id,
                        error = %format!("{error:#}"),
                        "ignored unreadable FlashMoe session cache"
                    );
                }
            }
        }
        if cached.is_none()
            && let (Some(key), Some(disk)) = (base_key.as_ref(), self.disk.as_ref())
        {
            let disk_started = Instant::now();
            match disk.load_prefix(&prompt_tokens[..base_prefix_len]) {
                Ok(Some(state)) => {
                    disk_read_decode_wall += disk_started.elapsed();
                    let validation_started = Instant::now();
                    self.insert_prefix(key.clone(), state.clone());
                    self.touch_prefix(key);
                    cached = Some((base_prefix_len, state));
                    cache_source = PromptCacheSource::DiskPrefix;
                    used_disk = true;
                    cpu_state_validation_allocation_wall += validation_started.elapsed();
                }
                Ok(None) => {
                    disk_read_decode_wall += disk_started.elapsed();
                }
                Err(error) => {
                    disk_read_decode_wall += disk_started.elapsed();
                    cache_unreadable = true;
                    tracing::warn!(
                        prefix = key,
                        error = %format!("{error:#}"),
                        "ignored unreadable FlashMoe prefix cache"
                    );
                }
            }
        }
        let cache_miss_reason = cached.is_none().then(|| {
            if session_id.is_none() {
                PromptCacheMissReason::CacheDisabled
            } else if cache_unreadable {
                PromptCacheMissReason::CacheUnreadable
            } else if incompatible_checkpoint_seen {
                PromptCacheMissReason::PromptDiverged
            } else if base_prefix_len == 0 {
                PromptCacheMissReason::StablePrefixUnavailable
            } else {
                PromptCacheMissReason::ColdSession
            }
        });
        let cache_lookup_detail = if incompatible_checkpoint_seen && cached.is_some() {
            Some(crate::inference::PromptCacheLookupDetail::SessionDivergedRootHit)
        } else if incompatible_checkpoint_seen && base_prefix_len > 0 {
            Some(crate::inference::PromptCacheLookupDetail::SessionDivergedRootMissing)
        } else if incompatible_checkpoint_seen {
            Some(crate::inference::PromptCacheLookupDetail::SessionCheckpointDiverged)
        } else if cached.is_none() && base_prefix_len > 0 {
            Some(crate::inference::PromptCacheLookupDetail::ExactRootCheckpointMissing)
        } else if cached.is_none() && session_id.is_some() {
            Some(crate::inference::PromptCacheLookupDetail::SessionCheckpointMissing)
        } else {
            None
        };
        let restore_ms = used_disk
            .then(|| u64::try_from(restore_started.elapsed().as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        let allocation_started = Instant::now();
        let (kv_cache, prefill_start, cached_last_hidden, cached_recurrent) =
            if let Some((prefix_len, state)) = cached {
                let FlashMoeSessionState {
                    tokens: _,
                    mut kv_cache,
                    last_hidden,
                } = state.cpu;
                kv_cache.resize_capacity(capacity);
                let cached_last_hidden = (prefix_len == prompt_tokens.len()).then_some(last_hidden);
                (
                    kv_cache,
                    prefix_len,
                    cached_last_hidden,
                    Some(state.recurrent),
                )
            } else {
                (KvCache::new(layers, capacity), 0, None, None)
            };
        cpu_state_validation_allocation_wall += allocation_started.elapsed();
        let cache_lookup_wall = preparation_started
            .elapsed()
            .saturating_sub(disk_read_decode_wall)
            .saturating_sub(cpu_state_validation_allocation_wall);

        FlashMoeGenerationState {
            session_id: session_id.map(str::to_owned),
            prompt_tokens,
            kv_cache,
            prefill_start,
            cached_last_hidden,
            prompt_cache: None,
            cached_recurrent,
            prompt_recurrent: None,
            generated_cache: None,
            generated_recurrent: None,
            cache_source,
            cache_restore_ms: restore_ms,
            cache_lookup_ms: duration_millis(cache_lookup_wall),
            disk_read_decode_ms: duration_millis(disk_read_decode_wall),
            cpu_state_validation_allocation_ms: duration_millis(
                cpu_state_validation_allocation_wall,
            ),
            cache_miss_reason,
            cache_lookup_detail,
            base_prefix_len,
            base_cache: None,
            base_recurrent: None,
            generated: Vec::new(),
            max_tokens,
            stopped: false,
            stopped_by_terminal_tool_call: false,
            stopped_by_constraint_payload_limit: false,
        }
    }

    pub(crate) fn begin_external_prefix_generation(
        prompt_tokens: Vec<u32>,
        prefill_start: usize,
        cached_last_hidden: Option<Vec<f32>>,
        max_tokens: usize,
        layers: usize,
        cache_source: PromptCacheSource,
        cache_restore_ms: u64,
        cache_lookup_ms: u64,
        disk_read_decode_ms: u64,
        cpu_state_validation_allocation_ms: u64,
        cache_miss_reason: Option<PromptCacheMissReason>,
    ) -> Result<FlashMoeGenerationState> {
        if prefill_start > prompt_tokens.len() {
            bail!(
                "external FlashMoe prefix {prefill_start} exceeds prompt length {}",
                prompt_tokens.len()
            );
        }
        let mut kv_cache = KvCache::new(layers, prompt_tokens.len() + max_tokens);
        for (position, token) in prompt_tokens
            .iter()
            .copied()
            .enumerate()
            .take(prefill_start)
        {
            kv_cache.record_prompt_token_record(FlashMoePromptTokenRecord::new(position, token))?;
        }
        Ok(FlashMoeGenerationState {
            session_id: None,
            prompt_tokens,
            kv_cache,
            prefill_start,
            cached_last_hidden,
            prompt_cache: None,
            cached_recurrent: None,
            prompt_recurrent: None,
            generated_cache: None,
            generated_recurrent: None,
            cache_source,
            cache_restore_ms,
            cache_lookup_ms,
            disk_read_decode_ms,
            cpu_state_validation_allocation_ms,
            cache_miss_reason,
            cache_lookup_detail: None,
            base_prefix_len: 0,
            base_cache: None,
            base_recurrent: None,
            generated: Vec::new(),
            max_tokens,
            stopped: false,
            stopped_by_terminal_tool_call: false,
            stopped_by_constraint_payload_limit: false,
        })
    }

    pub(crate) fn commit_generation(
        &mut self,
        generation: &mut FlashMoeGenerationState,
    ) -> Result<super::types::PromptCachePersistenceStats> {
        let Some(session_id) = generation.session_id.as_ref() else {
            return Ok(super::types::PromptCachePersistenceStats::default());
        };
        let cpu = generation
            .prompt_cache
            .take()
            .context("session cache prompt snapshot is missing")?;
        let recurrent = generation
            .prompt_recurrent
            .take()
            .context("session cache recurrent snapshot is missing")?;
        let mut checkpoints = vec![FlashMoeCachedSessionState { cpu, recurrent }];
        if let (Some(cpu), Some(recurrent)) = (
            generation.generated_cache.take(),
            generation.generated_recurrent.take(),
        ) {
            checkpoints.push(FlashMoeCachedSessionState { cpu, recurrent });
        }
        self.entries.insert(session_id.clone(), checkpoints);
        self.touch_session(session_id);
        self.dirty_sessions.insert(session_id.clone());
        let mut persistence = self.evict_excess_sessions(self.memory_session_limit);
        if let (Some(cpu), Some(recurrent)) = (
            generation.base_cache.take(),
            generation.base_recurrent.take(),
        ) {
            let state = FlashMoeCachedSessionState { cpu, recurrent };
            let key = FlashMoeDiskCache::token_key(&state.cpu.tokens);
            self.insert_prefix(key.clone(), state);
            self.touch_prefix(&key);
            self.dirty_prefixes.insert(key);
            add_persistence_stats(&mut persistence, self.evict_excess_prefixes());
        }
        add_persistence_stats(&mut self.deferred_persistence, persistence);
        Ok(persistence)
    }

    fn touch_prefix(&mut self, key: &str) {
        self.prefix_order.retain(|existing| existing != key);
        self.prefix_order.push_back(key.to_string());
    }

    fn insert_prefix(&mut self, key: String, state: FlashMoeCachedSessionState) {
        if let Some(previous) = self.prefix_sizes.remove(&key) {
            self.prefix_bytes = self.prefix_bytes.saturating_sub(previous);
        }
        let bytes = checkpoint_size(&state);
        self.prefix_bytes = self.prefix_bytes.saturating_add(bytes);
        self.prefix_sizes.insert(key.clone(), bytes);
        self.prefixes.insert(key, state);
    }

    fn touch_session(&mut self, session_id: &str) {
        self.session_order.retain(|existing| existing != session_id);
        self.session_order.push_back(session_id.to_string());
    }

    fn evict_excess_prefixes(&mut self) -> super::types::PromptCachePersistenceStats {
        let mut persistence = super::types::PromptCachePersistenceStats::default();
        while self.prefix_bytes > self.memory_prefix_max_bytes {
            let Some(oldest) = self.prefix_order.pop_front() else {
                break;
            };
            if self.dirty_prefixes.contains(&oldest)
                && let (Some(disk), Some(state)) = (self.disk.as_ref(), self.prefixes.get(&oldest))
            {
                persistence.queued_checkpoints = persistence.queued_checkpoints.saturating_add(1);
                let started = Instant::now();
                match disk.persist_prefix(state) {
                    Ok(()) => {
                        persistence.completed_checkpoints =
                            persistence.completed_checkpoints.saturating_add(1);
                        self.dirty_prefixes.remove(&oldest);
                    }
                    Err(error) => {
                        persistence.failed_checkpoints =
                            persistence.failed_checkpoints.saturating_add(1);
                        tracing::warn!(
                            prefix = oldest,
                            error = %format!("{error:#}"),
                            "could not persist evicted FlashMoe prompt-root cache"
                        );
                    }
                }
                persistence.wall_ms = persistence
                    .wall_ms
                    .saturating_add(duration_millis(started.elapsed()));
            }
            self.prefixes.remove(&oldest);
            if let Some(bytes) = self.prefix_sizes.remove(&oldest) {
                self.prefix_bytes = self.prefix_bytes.saturating_sub(bytes);
            }
            self.dirty_prefixes.remove(&oldest);
        }
        persistence
    }

    pub(super) fn evict_excess_sessions(
        &mut self,
        limit: usize,
    ) -> super::types::PromptCachePersistenceStats {
        let mut persistence = super::types::PromptCachePersistenceStats::default();
        while self.entries.len() > limit {
            let Some(oldest) = self.session_order.pop_front() else {
                break;
            };
            if self.dirty_sessions.contains(&oldest)
                && let (Some(disk), Some(safe_prompt)) = (
                    self.disk.as_ref(),
                    self.entries.get(&oldest).and_then(|states| states.first()),
                )
            {
                persistence.queued_checkpoints = persistence.queued_checkpoints.saturating_add(1);
                let started = Instant::now();
                match disk.persist_session(&oldest, std::slice::from_ref(safe_prompt)) {
                    Ok(()) => {
                        persistence.completed_checkpoints =
                            persistence.completed_checkpoints.saturating_add(1);
                    }
                    Err(error) => {
                        persistence.failed_checkpoints =
                            persistence.failed_checkpoints.saturating_add(1);
                        tracing::warn!(
                            session = oldest,
                            error = %format!("{error:#}"),
                            "could not persist evicted FlashMoe session cache"
                        );
                    }
                }
                persistence.wall_ms = persistence
                    .wall_ms
                    .saturating_add(duration_millis(started.elapsed()));
            }
            self.entries.remove(&oldest);
            self.dirty_sessions.remove(&oldest);
        }
        persistence
    }

    pub(crate) fn persist_session(
        &mut self,
        session_id: &str,
    ) -> Result<super::types::PromptCachePersistenceStats> {
        let mut persistence = std::mem::take(&mut self.deferred_persistence);
        let Some(disk) = self.disk.as_ref() else {
            return Ok(persistence);
        };
        let prefix_keys = self.dirty_prefixes.iter().cloned().collect::<Vec<_>>();
        for key in &prefix_keys {
            if let Some(state) = self.prefixes.get(key) {
                persistence.queued_checkpoints = persistence.queued_checkpoints.saturating_add(1);
                let started = Instant::now();
                match disk.persist_prefix(state) {
                    Ok(()) => {
                        persistence.completed_checkpoints =
                            persistence.completed_checkpoints.saturating_add(1);
                        self.dirty_prefixes.remove(key);
                    }
                    Err(error) => {
                        persistence.failed_checkpoints =
                            persistence.failed_checkpoints.saturating_add(1);
                        tracing::warn!(
                            prefix = key,
                            error = %format!("{error:#}"),
                            "could not persist FlashMoe prompt-root cache"
                        );
                    }
                }
                persistence.wall_ms = persistence
                    .wall_ms
                    .saturating_add(duration_millis(started.elapsed()));
            }
        }
        if self.dirty_sessions.contains(session_id)
            && let Some(states) = self.entries.get(session_id)
            && let Some(safe_prompt) = states.first()
        {
            // The generated head is a speculative in-memory accelerator. Persist
            // only the canonical prompt boundary so restart durability does not
            // double checkpoint writes or depend on output re-tokenization.
            persistence.queued_checkpoints = persistence.queued_checkpoints.saturating_add(1);
            let started = Instant::now();
            match disk.persist_session(session_id, std::slice::from_ref(safe_prompt)) {
                Ok(()) => {
                    persistence.completed_checkpoints =
                        persistence.completed_checkpoints.saturating_add(1);
                    self.dirty_sessions.remove(session_id);
                }
                Err(error) => {
                    persistence.failed_checkpoints =
                        persistence.failed_checkpoints.saturating_add(1);
                    tracing::warn!(
                        session = session_id,
                        error = %format!("{error:#}"),
                        "could not persist FlashMoe session cache"
                    );
                }
            }
            persistence.wall_ms = persistence
                .wall_ms
                .saturating_add(duration_millis(started.elapsed()));
        }
        Ok(persistence)
    }
}

fn add_persistence_stats(
    target: &mut super::types::PromptCachePersistenceStats,
    value: super::types::PromptCachePersistenceStats,
) {
    target.queued_checkpoints = target
        .queued_checkpoints
        .saturating_add(value.queued_checkpoints);
    target.completed_checkpoints = target
        .completed_checkpoints
        .saturating_add(value.completed_checkpoints);
    target.failed_checkpoints = target
        .failed_checkpoints
        .saturating_add(value.failed_checkpoints);
    target.wall_ms = target.wall_ms.saturating_add(value.wall_ms);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::flashmoe::state::FlashMoeLinearAttentionLayerSnapshot;

    fn fixture_state() -> FlashMoeCachedSessionState {
        let mut kv = KvCache::new(2, 3);
        kv.record_kv(0, 0, vec![1.0, 2.0], vec![3.0, 4.0]).unwrap();
        kv.record_mla_kv(1, 1, vec![5.0], vec![6.0, 7.0]).unwrap();
        FlashMoeCachedSessionState {
            cpu: FlashMoeSessionState::new(vec![10, 20], kv, vec![8.0, 9.0]),
            recurrent: FlashMoeLinearAttentionSessionSnapshot::new(vec![
                Some(
                    FlashMoeLinearAttentionLayerSnapshot::new(
                        0,
                        vec![10.0],
                        vec![11.0, 12.0],
                        1,
                        2,
                    )
                    .unwrap(),
                ),
                None,
            ])
            .unwrap(),
        }
    }

    fn disk_cache(root: PathBuf) -> FlashMoeDiskCache {
        FlashMoeDiskCache {
            storage_root: root.clone(),
            root,
            fingerprint: [9_u8; 32],
            layers: 2,
            max_bytes: 1024 * 1024,
        }
    }

    #[test]
    fn cache_miss_reasons_distinguish_cold_diverged_and_disabled_sessions() {
        let mut cache = FlashMoeSessionCache::default();
        let cold = cache.begin_generation_with_base(Some("cold"), vec![1, 2], 1, 1, 2);
        assert_eq!(
            cold.cache_miss_reason(),
            Some(PromptCacheMissReason::ColdSession)
        );

        let disabled = cache.begin_generation(None, vec![1, 2], 1, 2);
        assert_eq!(
            disabled.cache_miss_reason(),
            Some(PromptCacheMissReason::CacheDisabled)
        );

        cache
            .entries
            .insert("diverged".to_string(), vec![fixture_state()]);
        let diverged = cache.begin_generation(Some("diverged"), vec![1, 2], 1, 2);
        assert_eq!(
            diverged.cache_miss_reason(),
            Some(PromptCacheMissReason::PromptDiverged)
        );
        assert_eq!(
            diverged.cache_lookup_detail(),
            Some(crate::inference::PromptCacheLookupDetail::SessionCheckpointDiverged)
        );

        cache
            .entries
            .insert("reused".to_string(), vec![fixture_state()]);
        let reused = cache.begin_generation(Some("reused"), vec![10, 20, 30], 1, 2);
        assert_eq!(reused.cache_source(), PromptCacheSource::MemorySession);
        assert_eq!(reused.cache_miss_reason(), None);
    }

    #[test]
    fn session_mismatch_can_fall_through_to_an_exact_disk_root_hit() {
        let tmp = tempfile::tempdir().unwrap();
        let disk = disk_cache(tmp.path().join("cache"));
        let root = fixture_state();
        disk.persist_prefix(&root).unwrap();
        let mut diverged = fixture_state();
        diverged.cpu.tokens = vec![90, 91];
        let mut cache = FlashMoeSessionCache::new(
            Some(disk),
            2,
            crate::config::DEFAULT_FLASHMOE_MEMORY_PROMPT_ROOT_MAX_BYTES,
        );
        cache.entries.insert("session".to_string(), vec![diverged]);

        let generation =
            cache.begin_generation_with_base(Some("session"), vec![10, 20, 30], 2, 1, 2);

        assert_eq!(generation.cache_source(), PromptCacheSource::DiskPrefix);
        assert_eq!(generation.prefill_start(), 2);
        assert_eq!(generation.cache_miss_reason(), None);
        assert_eq!(
            generation.cache_lookup_detail(),
            Some(crate::inference::PromptCacheLookupDetail::SessionDivergedRootHit)
        );
    }

    #[test]
    fn checkpoint_round_trip_preserves_full_mla_and_recurrent_state() {
        let state = fixture_state();
        let fingerprint = [7_u8; 32];
        let mut bytes = Vec::new();
        write_checkpoint(&mut bytes, fingerprint, &state).unwrap();
        let restored = read_checkpoint(bytes.as_slice(), fingerprint, 2).unwrap();
        assert_eq!(restored.cpu.tokens, vec![10, 20]);
        assert_eq!(restored.cpu.last_hidden, vec![8.0, 9.0]);
        assert_eq!(restored.cpu.kv_cache.keys_values(1, 0).unwrap().len(), 1);
        assert_eq!(restored.cpu.kv_cache.mla_records(1, 1).unwrap().len(), 1);
        assert_eq!(restored.recurrent.layer(0).unwrap().conv_state(), &[10.0]);
    }

    #[test]
    fn checkpoint_rejects_another_model_fingerprint() {
        let mut bytes = Vec::new();
        write_checkpoint(&mut bytes, [1_u8; 32], &fixture_state()).unwrap();
        let error = read_checkpoint(bytes.as_slice(), [2_u8; 32], 2).unwrap_err();
        assert!(error.to_string().contains("fingerprint"));
    }

    #[test]
    fn token_keys_are_content_addressed() {
        assert_eq!(
            FlashMoeDiskCache::token_key(&[1, 2]),
            FlashMoeDiskCache::token_key(&[1, 2])
        );
        assert_ne!(
            FlashMoeDiskCache::token_key(&[1, 2]),
            FlashMoeDiskCache::token_key(&[1, 3])
        );
    }

    #[test]
    fn disk_cache_round_trip_uses_hashed_session_manifests() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = disk_cache(tmp.path().join("cache"));
        let state = fixture_state();
        cache.persist_prefix(&state).unwrap();
        cache
            .persist_session("private session id", std::slice::from_ref(&state))
            .unwrap();

        let restored_prefix = cache.load_prefix(&[10, 20]).unwrap().unwrap();
        assert_eq!(restored_prefix.cpu.tokens, vec![10, 20]);
        let restored_session = cache.load_session("private session id").unwrap();
        assert_eq!(restored_session.len(), 1);
        assert_eq!(restored_session[0].cpu.last_hidden, vec![8.0, 9.0]);
        let manifests = fs::read_dir(cache.root.join("sessions"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(manifests.len(), 1);
        assert!(!manifests[0].contains("private"));
    }

    #[test]
    fn concurrent_cache_writers_publish_complete_manifests() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("cache");
        let first_root = root.clone();
        let second_root = root.clone();
        let first = std::thread::spawn(move || {
            let cache = disk_cache(first_root);
            cache
                .persist_session("first", std::slice::from_ref(&fixture_state()))
                .unwrap();
        });
        let second = std::thread::spawn(move || {
            let cache = disk_cache(second_root);
            cache
                .persist_session("second", std::slice::from_ref(&fixture_state()))
                .unwrap();
        });
        first.join().unwrap();
        second.join().unwrap();

        let cache = disk_cache(root);
        assert_eq!(cache.load_session("first").unwrap().len(), 1);
        assert_eq!(cache.load_session("second").unwrap().len(), 1);
    }

    #[test]
    fn prompt_root_memory_budget_evicts_lru_and_persists_dirty_state() {
        let tmp = tempfile::tempdir().unwrap();
        let disk = disk_cache(tmp.path().join("cache"));
        let first = fixture_state();
        let mut second = fixture_state();
        second.cpu.tokens = vec![30, 40];
        let first_key = FlashMoeDiskCache::token_key(&first.cpu.tokens);
        let second_key = FlashMoeDiskCache::token_key(&second.cpu.tokens);
        let budget = checkpoint_size(&first).max(checkpoint_size(&second));
        let mut cache = FlashMoeSessionCache::new(Some(disk), 2, budget);
        cache.insert_prefix(first_key.clone(), first.clone());
        cache.insert_prefix(second_key.clone(), second);
        cache.prefix_order = VecDeque::from([first_key.clone(), second_key.clone()]);
        cache.dirty_prefixes = BTreeSet::from([first_key.clone(), second_key.clone()]);

        cache.evict_excess_prefixes();

        assert!(!cache.prefixes.contains_key(&first_key));
        assert!(cache.prefixes.contains_key(&second_key));
        assert!(cache.prefix_bytes <= budget);
        assert_eq!(cache.prefix_sizes.len(), cache.prefixes.len());
        assert!(!cache.dirty_prefixes.contains(&first_key));
        assert!(cache.dirty_prefixes.contains(&second_key));
        let restored = cache
            .disk
            .as_ref()
            .unwrap()
            .load_prefix(&first.cpu.tokens)
            .unwrap()
            .unwrap();
        assert_eq!(restored.cpu.tokens, first.cpu.tokens);
    }

    #[test]
    fn checkpoint_pruning_removes_dangling_manifest_references() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cache = disk_cache(tmp.path().join("cache"));
        let state = fixture_state();
        cache
            .persist_session("session", std::slice::from_ref(&state))
            .unwrap();
        let key = FlashMoeDiskCache::token_key(&state.cpu.tokens);

        cache.max_bytes = 1;
        cache.prune(&[]).unwrap();

        assert!(!cache.session_manifest_path("session").exists());
        assert!(!cache.checkpoint_path(&key).exists());
    }

    #[test]
    fn successful_checkpoint_restore_refreshes_disk_lru_recency() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cache = disk_cache(tmp.path().join("cache"));
        let first = fixture_state();
        let mut second = fixture_state();
        second.cpu.tokens = vec![30, 40];
        cache.persist_prefix(&first).unwrap();
        cache.persist_prefix(&second).unwrap();

        let first_path = cache.checkpoint_path(&FlashMoeDiskCache::token_key(&first.cpu.tokens));
        let second_path = cache.checkpoint_path(&FlashMoeDiskCache::token_key(&second.cpu.tokens));
        let first_file = File::open(&first_path).unwrap();
        let second_file = File::open(&second_path).unwrap();
        first_file
            .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(1)))
            .unwrap();
        second_file
            .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(2)))
            .unwrap();

        cache.load_prefix(&first.cpu.tokens).unwrap().unwrap();
        cache.max_bytes = fs::metadata(&first_path).unwrap().len();
        cache.prune(&[]).unwrap();

        assert!(first_path.exists());
        assert!(!second_path.exists());
    }

    #[test]
    fn partial_checkpoint_falls_back_to_truthful_fresh_prefill() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = disk_cache(tmp.path().join("cache"));
        cache.ensure_namespace(true).unwrap();
        let tokens = [10_u32, 20];
        let key = FlashMoeDiskCache::token_key(&tokens);
        let path = cache.checkpoint_path(&key);
        secure_directory(path.parent().unwrap()).unwrap();
        fs::write(&path, b"interrupted checkpoint").unwrap();
        let mut sessions = FlashMoeSessionCache::new(
            Some(cache),
            2,
            crate::config::DEFAULT_FLASHMOE_MEMORY_PROMPT_ROOT_MAX_BYTES,
        );

        let generation =
            sessions.begin_generation_with_base(Some("session"), vec![10, 20, 30], 2, 1, 2);

        assert_eq!(generation.prefill_start(), 0);
        assert_eq!(generation.cache_source(), PromptCacheSource::None);
        assert_eq!(
            generation.cache_miss_reason(),
            Some(PromptCacheMissReason::CacheUnreadable)
        );
    }

    #[test]
    fn full_cache_budget_skips_oversized_checkpoint_without_partial_state() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cache = disk_cache(tmp.path().join("cache"));
        let state = fixture_state();
        cache.max_bytes = checkpoint_size(&state).saturating_sub(1);

        cache.persist_prefix(&state).unwrap();

        assert!(cache.load_prefix(&state.cpu.tokens).unwrap().is_none());
        assert!(
            !cache
                .checkpoint_path(&FlashMoeDiskCache::token_key(&state.cpu.tokens))
                .exists()
        );
    }

    #[test]
    fn changing_storage_root_does_not_discover_or_rewrite_old_state() {
        let tmp = tempfile::tempdir().unwrap();
        let first = disk_cache(tmp.path().join("first"));
        let second = disk_cache(tmp.path().join("second"));
        let state = fixture_state();
        first.persist_prefix(&state).unwrap();

        assert!(second.load_prefix(&state.cpu.tokens).unwrap().is_none());
        assert!(first.load_prefix(&state.cpu.tokens).unwrap().is_some());
        assert!(!second.root.exists());
    }

    #[cfg(unix)]
    #[test]
    fn read_only_namespace_fails_before_publishing_partial_state() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let blocked_parent = tmp.path().join("blocked");
        fs::create_dir(&blocked_parent).unwrap();
        let original_permissions = fs::metadata(&blocked_parent).unwrap().permissions();
        let mut read_only = original_permissions.clone();
        read_only.set_mode(0o500);
        fs::set_permissions(&blocked_parent, read_only).unwrap();
        let cache = disk_cache(blocked_parent.join("cache"));

        let state = fixture_state();
        let result = cache.persist_prefix(&state);
        fs::set_permissions(&blocked_parent, original_permissions).unwrap();

        assert!(result.is_err());
        assert!(
            !cache
                .checkpoint_path(&FlashMoeDiskCache::token_key(&state.cpu.tokens))
                .exists()
        );
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_loading_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let cache = disk_cache(tmp.path().join("cache"));
        let tokens = [10_u32, 20];
        let key = FlashMoeDiskCache::token_key(&tokens);
        let path = cache.checkpoint_path(&key);
        secure_directory(path.parent().unwrap()).unwrap();
        let outside = tmp.path().join("outside.bin");
        fs::write(&outside, b"outside").unwrap();
        symlink(&outside, &path).unwrap();

        let error = cache.load_prefix(&tokens).unwrap_err();
        assert!(format!("{error:#}").contains("symlink"));
    }

    #[cfg(unix)]
    #[test]
    fn namespace_loading_rejects_configured_root_symlink_escape() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let outside = tmp.path().join("outside");
        fs::create_dir(&outside).unwrap();
        let storage_root = tmp.path().join("cache-link");
        symlink(&outside, &storage_root).unwrap();
        let cache = FlashMoeDiskCache {
            root: storage_root.join(CACHE_VERSION).join("model"),
            storage_root,
            fingerprint: [9_u8; 32],
            layers: 2,
            max_bytes: 1024 * 1024,
        };

        let error = cache.load_prefix(&[10, 20]).unwrap_err();
        assert!(format!("{error:#}").contains("symlink"));
    }
}
