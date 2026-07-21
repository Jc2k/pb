use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

use super::planning::FlashMoePlan;
use super::state::{
    FlashMoeCachedSessionState, FlashMoeLinearAttentionLayerSnapshot,
    FlashMoeLinearAttentionSessionSnapshot, FlashMoeSessionState, KvCache,
};

const CACHE_VERSION: &str = "flashmoe-session-v1";
const MAGIC: &[u8; 8] = b"PBFMKV01";
const MAX_TOKENS: usize = 1_000_000;
const MAX_VECTOR_FLOATS: usize = 32 * 1024 * 1024;

#[derive(Debug)]
pub(super) struct FlashMoeDiskCache {
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
        let root = settings
            .root
            .as_ref()?
            .join(CACHE_VERSION)
            .join(model_fingerprint_hex(plan));
        Some(Self {
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
        self.load_checkpoint(&Self::token_key(tokens))
    }

    pub(super) fn load_session(&self, session_id: &str) -> Result<Vec<FlashMoeCachedSessionState>> {
        let path = self.session_manifest_path(session_id);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", path.display()));
            }
        };
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
        Ok(checkpoints)
    }

    pub(super) fn persist_prefix(&self, state: &FlashMoeCachedSessionState) -> Result<()> {
        let _ = self.persist_checkpoint(state)?;
        Ok(())
    }

    pub(super) fn persist_session(
        &self,
        session_id: &str,
        states: &[FlashMoeCachedSessionState],
    ) -> Result<()> {
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

    fn load_checkpoint(&self, key: &str) -> Result<Option<FlashMoeCachedSessionState>> {
        if key.len() != 64 || !key.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Ok(None);
        }
        let path = self.checkpoint_path(key);
        let file = match File::open(&path) {
            Ok(file) => file,
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
        let state = read_checkpoint(BufReader::new(file), self.fingerprint, self.layers)
            .with_context(|| format!("failed to restore {}", path.display()))?;
        if Self::token_key(&state.cpu.tokens) != key {
            return Ok(None);
        }
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
        if path.is_file() {
            return Ok(Some(key));
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
            Ok(_) => {}
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
            fs::remove_file(&path)
                .with_context(|| format!("failed to prune {}", path.display()))?;
            total = total.saturating_sub(bytes);
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

fn model_fingerprint_hex(plan: &FlashMoePlan) -> String {
    model_fingerprint(plan)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn secure_directory(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("failed to create {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("failed to secure {}", path.display()))?;
    }
    Ok(())
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
    Ok(())
}

fn collect_cache_files(root: &Path, files: &mut Vec<(PathBuf, u64, Duration)>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_cache_files(&path, files)?;
        } else if metadata.is_file() {
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
        let cache = FlashMoeDiskCache {
            root: tmp.path().join("cache"),
            fingerprint: [9_u8; 32],
            layers: 2,
            max_bytes: 1024 * 1024,
        };
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
}
