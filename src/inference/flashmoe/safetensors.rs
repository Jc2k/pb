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
}
