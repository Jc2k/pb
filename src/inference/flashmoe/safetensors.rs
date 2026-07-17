use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(super) struct SafetensorsIndex {
    pub(super) weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub(super) struct SafetensorShard {
    pub(super) data_start: u64,
    pub(super) tensors: BTreeMap<String, SafetensorTensorInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SafetensorsManifestSource {
    DeclaredIndex,
    ActualShardHeaders,
}

#[derive(Debug)]
pub(super) struct ResolvedSafetensorsManifest {
    pub(super) weight_map: BTreeMap<String, String>,
    pub(super) shards: BTreeMap<String, SafetensorShard>,
    pub(super) source: SafetensorsManifestSource,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct SafetensorTensorInfo {
    pub(super) dtype: String,
    pub(super) shape: Vec<usize>,
    pub(super) data_offsets: [u64; 2],
}

pub(super) fn parse_safetensors_header(path: &Path) -> Result<SafetensorShard> {
    use std::io::Read;
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open safetensors shard {}", path.display()))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("failed to stat safetensors shard {}", path.display()))?
        .len();
    if file_len < 8 {
        bail!(
            "safetensors shard {} is too small to contain a header",
            path.display()
        );
    }
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .with_context(|| format!("failed to read header length from {}", path.display()))?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    let header_start = 8usize;
    let header_end = header_start
        .checked_add(header_len)
        .context("safetensors header length overflow")?;
    if header_end as u64 > file_len {
        bail!("safetensors shard {} has truncated header", path.display());
    }
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .with_context(|| format!("failed to read safetensors header from {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&header_bytes)
        .with_context(|| format!("failed to parse safetensors header {}", path.display()))?;
    let mut tensors = BTreeMap::new();
    let object = value
        .as_object()
        .context("safetensors header must be a JSON object")?;
    for (name, entry) in object {
        if name == "__metadata__" {
            continue;
        }
        let info: SafetensorTensorInfo = serde_json::from_value(entry.clone())
            .with_context(|| format!("failed to parse safetensors tensor metadata for {name}"))?;
        if info.data_offsets[1] < info.data_offsets[0] {
            bail!("tensor {name} has invalid safetensors data_offsets");
        }
        let absolute_end = header_end as u64 + info.data_offsets[1];
        if absolute_end > file_len {
            bail!(
                "tensor {name} data range exceeds shard length in {}",
                path.display()
            );
        }
        tensors.insert(name.clone(), info);
    }
    Ok(SafetensorShard {
        data_start: header_end as u64,
        tensors,
    })
}

pub(super) fn resolve_safetensors_manifest(
    snapshot_dir: &Path,
    index_json: &Path,
) -> Result<ResolvedSafetensorsManifest> {
    let index: SafetensorsIndex = serde_json::from_slice(
        &fs::read(index_json)
            .with_context(|| format!("failed to read {}", index_json.display()))?,
    )
    .with_context(|| format!("failed to parse {}", index_json.display()))?;
    let missing_declared_shard = index
        .weight_map
        .values()
        .find(|shard| !snapshot_dir.join(shard).is_file())
        .cloned();
    if missing_declared_shard.is_none() {
        return Ok(ResolvedSafetensorsManifest {
            weight_map: index.weight_map,
            shards: BTreeMap::new(),
            source: SafetensorsManifestSource::DeclaredIndex,
        });
    }

    let mut shard_paths = Vec::new();
    for entry in fs::read_dir(snapshot_dir).with_context(|| {
        format!(
            "failed to read safetensors snapshot {}",
            snapshot_dir.display()
        )
    })? {
        let entry = entry.with_context(|| {
            format!(
                "failed to inspect an entry in safetensors snapshot {}",
                snapshot_dir.display()
            )
        })?;
        let path = entry.path();
        if entry.file_type()?.is_file()
            && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
        {
            shard_paths.push(path);
        }
    }
    shard_paths.sort();
    if shard_paths.is_empty() {
        let missing = missing_declared_shard.expect("checked above");
        bail!(
            "safetensors index references missing shard {missing}, and {} contains no actual safetensors shards",
            snapshot_dir.display()
        );
    }

    let mut weight_map = BTreeMap::new();
    let mut shards = BTreeMap::new();
    for path in shard_paths {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| format!("safetensors shard has a non-UTF-8 name: {}", path.display()))?
            .to_string();
        let shard = parse_safetensors_header(&path)?;
        for tensor in shard.tensors.keys() {
            if let Some(existing) = weight_map.insert(tensor.clone(), file_name.clone()) {
                bail!(
                    "safetensors tensor {tensor} is declared by multiple actual shards: {existing} and {file_name}"
                );
            }
        }
        shards.insert(file_name, shard);
    }
    Ok(ResolvedSafetensorsManifest {
        weight_map,
        shards,
        source: SafetensorsManifestSource::ActualShardHeaders,
    })
}

pub(super) fn resolve_unindexed_safetensors_manifest(
    snapshot_dir: &Path,
) -> Result<ResolvedSafetensorsManifest> {
    let mut shard_paths = fs::read_dir(snapshot_dir)
        .with_context(|| {
            format!(
                "failed to read safetensors snapshot {}",
                snapshot_dir.display()
            )
        })?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    shard_paths.retain(|path| {
        path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
    });
    shard_paths.sort();
    if shard_paths.is_empty() {
        bail!(
            "{} contains no safetensors index and no safetensors shards",
            snapshot_dir.display()
        );
    }
    let mut weight_map = BTreeMap::new();
    let mut shards = BTreeMap::new();
    for path in shard_paths {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| format!("safetensors shard has a non-UTF-8 name: {}", path.display()))?
            .to_string();
        let shard = parse_safetensors_header(&path)?;
        for tensor in shard.tensors.keys() {
            if let Some(existing) = weight_map.insert(tensor.clone(), file_name.clone()) {
                bail!(
                    "safetensors tensor {tensor} is declared by multiple actual shards: {existing} and {file_name}"
                );
            }
        }
        shards.insert(file_name, shard);
    }
    Ok(ResolvedSafetensorsManifest {
        weight_map,
        shards,
        source: SafetensorsManifestSource::ActualShardHeaders,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_resolves_validated_absolute_tensor_ranges() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.safetensors");
        let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let shard = parse_safetensors_header(&path).unwrap();
        assert_eq!(shard.data_start, 8 + header.len() as u64);
        assert_eq!(shard.tensors["weight"].shape, vec![1]);
        assert_eq!(shard.tensors["weight"].data_offsets, [0, 4]);
    }

    fn write_test_shard(path: &Path, tensor: &str) {
        let header =
            format!(r#"{{"{tensor}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn unindexed_manifest_scans_colibri_style_out_shards() {
        let temp = tempfile::tempdir().unwrap();
        write_test_shard(&temp.path().join("out-00000.safetensors"), "layer.0.weight");
        write_test_shard(&temp.path().join("out-00001.safetensors"), "layer.1.weight");

        let manifest = resolve_unindexed_safetensors_manifest(temp.path()).unwrap();

        assert_eq!(
            manifest.source,
            SafetensorsManifestSource::ActualShardHeaders
        );
        assert_eq!(manifest.weight_map.len(), 2);
        assert_eq!(
            manifest.weight_map["layer.0.weight"],
            "out-00000.safetensors"
        );
        assert_eq!(manifest.shards.len(), 2);
    }

    #[test]
    fn manifest_uses_declared_index_when_every_referenced_shard_exists() {
        let temp = tempfile::tempdir().unwrap();
        write_test_shard(
            &temp.path().join("model-00001-of-00001.safetensors"),
            "weight",
        );
        let index_path = temp.path().join("model.safetensors.index.json");
        fs::write(
            &index_path,
            br#"{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();

        let resolved = resolve_safetensors_manifest(temp.path(), &index_path).unwrap();

        assert_eq!(resolved.source, SafetensorsManifestSource::DeclaredIndex);
        assert_eq!(
            resolved.weight_map["weight"],
            "model-00001-of-00001.safetensors"
        );
        assert!(resolved.shards.is_empty());
    }

    #[test]
    fn manifest_uses_actual_headers_when_declared_shards_are_stale() {
        let temp = tempfile::tempdir().unwrap();
        write_test_shard(
            &temp.path().join("model-00001-of-00004.safetensors"),
            "actual.weight",
        );
        let index_path = temp.path().join("model.safetensors.index.json");
        fs::write(
            &index_path,
            br#"{"weight_map":{"stale.weight":"model-00001-of-00013.safetensors"}}"#,
        )
        .unwrap();

        let resolved = resolve_safetensors_manifest(temp.path(), &index_path).unwrap();

        assert_eq!(
            resolved.source,
            SafetensorsManifestSource::ActualShardHeaders
        );
        assert_eq!(
            resolved.weight_map["actual.weight"],
            "model-00001-of-00004.safetensors"
        );
        assert!(
            resolved
                .shards
                .contains_key("model-00001-of-00004.safetensors")
        );
        assert!(!resolved.weight_map.contains_key("stale.weight"));
    }

    #[test]
    fn actual_header_manifest_rejects_duplicate_tensor_ownership() {
        let temp = tempfile::tempdir().unwrap();
        write_test_shard(
            &temp.path().join("model-00001-of-00002.safetensors"),
            "weight",
        );
        write_test_shard(
            &temp.path().join("model-00002-of-00002.safetensors"),
            "weight",
        );
        let index_path = temp.path().join("model.safetensors.index.json");
        fs::write(
            &index_path,
            br#"{"weight_map":{"weight":"model-00001-of-00013.safetensors"}}"#,
        )
        .unwrap();

        let error = resolve_safetensors_manifest(temp.path(), &index_path).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("tensor weight is declared by multiple actual shards"),
            "{error:#}"
        );
    }
}
