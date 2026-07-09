use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use super::experts::EXPERT_SCALE_BIAS_DTYPE_F32;
use super::types::{ExpertQuantization, GROUP_SIZE};
use anyhow::{Context, Result, bail};

pub(crate) const TENSOR_ALIGNMENT: u64 = 4096;

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FlashMoeManifest {
    pub model: String,
    pub cache_version: String,
    pub dense_shards: Vec<String>,
    pub expert_tensors: Vec<ExpertTensorRef>,
    pub dense_tensors: Vec<DenseTensorRef>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpertTensorRef {
    pub tensor: String,
    pub shard: String,
    pub layer: Option<usize>,
    pub expert: Option<usize>,
    pub dtype: Option<String>,
    pub shape: Vec<usize>,
    pub source_offsets: Option<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q4_sources: Option<DenseQ4SourceRefs>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DenseTensorRef {
    pub tensor: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub source_offsets: [u64; 2],
    pub runtime_offset: u64,
    pub byte_len: u64,
    #[serde(default)]
    pub quantization: TensorQuantization,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q4_sources: Option<DenseQ4SourceRefs>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DenseQ4SourceRefs {
    pub scales_shard: String,
    pub scales_offsets: [u64; 2],
    pub biases_shard: String,
    pub biases_offsets: [u64; 2],
    pub scale_bias_dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TensorQuantization {
    None,
    Q4 {
        group_size: usize,
        format: String,
        #[serde(default = "default_dense_q4_scale_bias_dtype")]
        scale_bias_dtype: String,
    },
}

impl Default for TensorQuantization {
    fn default() -> Self {
        Self::None
    }
}

fn default_dense_q4_scale_bias_dtype() -> String {
    "F32".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeTensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub alignment: u64,
    pub quantization: TensorQuantization,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorRegistry {
    tensors: BTreeMap<String, RuntimeTensorEntry>,
}

impl TensorRegistry {
    pub fn load(manifest_path: &Path) -> Result<Self> {
        let manifest: FlashMoeManifest =
            serde_json::from_slice(&fs::read(manifest_path).with_context(|| {
                format!(
                    "failed to read Flash-MoE tensor manifest {}",
                    manifest_path.display()
                )
            })?)
            .with_context(|| {
                format!(
                    "failed to parse Flash-MoE tensor manifest {}",
                    manifest_path.display()
                )
            })?;
        Ok(Self::from_manifest(&manifest))
    }

    pub(crate) fn from_manifest(manifest: &FlashMoeManifest) -> Self {
        let mut tensors = BTreeMap::new();
        for tensor in &manifest.dense_tensors {
            insert_tensor_entry_with_aliases(
                &mut tensors,
                &tensor.tensor,
                RuntimeTensorEntry {
                    name: tensor.tensor.clone(),
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    byte_offset: tensor.runtime_offset,
                    byte_len: tensor.byte_len,
                    alignment: TENSOR_ALIGNMENT,
                    quantization: tensor.quantization.clone(),
                },
            );
        }
        for tensor in &manifest.expert_tensors {
            if let Some([start, end]) = tensor.source_offsets {
                insert_tensor_entry_with_aliases(
                    &mut tensors,
                    &tensor.tensor,
                    RuntimeTensorEntry {
                        name: tensor.tensor.clone(),
                        dtype: tensor
                            .dtype
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string()),
                        shape: tensor.shape.clone(),
                        byte_offset: start,
                        byte_len: end.saturating_sub(start),
                        alignment: TENSOR_ALIGNMENT,
                        quantization: TensorQuantization::Q4 {
                            group_size: GROUP_SIZE,
                            format: ExpertQuantization::FourBitProduction.as_str().to_string(),
                            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                        },
                    },
                );
            }
        }
        Self { tensors }
    }

    pub fn tensor(&self, canonical_name: &str) -> Option<&RuntimeTensorEntry> {
        self.tensors.get(canonical_name)
    }

    pub(crate) fn has_tensor_with_prefix(&self, prefix: &str) -> bool {
        self.tensors.keys().any(|name| name.starts_with(prefix))
    }

    pub fn require(&self, canonical_name: &str) -> Result<&RuntimeTensorEntry> {
        self.tensor(canonical_name)
            .with_context(|| format!("Flash-MoE tensor registry is missing {canonical_name}"))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

fn insert_tensor_entry_with_aliases(
    tensors: &mut BTreeMap<String, RuntimeTensorEntry>,
    name: &str,
    entry: RuntimeTensorEntry,
) {
    tensors
        .entry(name.to_string())
        .or_insert_with(|| entry.clone());
    let canonical_name = canonical_hf_tensor_name(name);
    if canonical_name != name {
        tensors.entry(canonical_name).or_insert(entry);
    }
}

pub(crate) fn canonical_hf_tensor_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else if let Some(rest) = name.strip_prefix("language_model.") {
        rest.to_string()
    } else if let Some(rest) = name.strip_prefix("model.visual.") {
        format!("visual.{rest}")
    } else {
        name.to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseQ4Layout {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) row_packed_bytes: usize,
    pub(crate) groups_per_row: usize,
    pub(crate) packed_bytes: usize,
    pub(crate) scales_bytes: usize,
    pub(crate) scale_bias_bytes: usize,
    pub(crate) total_bytes: usize,
}

#[cfg(test)]
pub(crate) fn dense_q4_layout(shape: &[usize], group_size: usize) -> Result<DenseQ4Layout> {
    dense_q4_layout_with_scale_bias_dtype(shape, group_size, EXPERT_SCALE_BIAS_DTYPE_F32)
}

pub(crate) fn dense_q4_layout_with_scale_bias_dtype(
    shape: &[usize],
    group_size: usize,
    scale_bias_dtype: &str,
) -> Result<DenseQ4Layout> {
    if group_size == 0 {
        bail!("dense q4 group_size must be positive");
    }
    let cols = shape.last().copied().unwrap_or(0);
    if shape.len() < 2 || cols == 0 {
        bail!(
            "dense q4 tensor shape {:?} is not a non-empty matrix",
            shape
        );
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .context("dense q4 tensor row count overflow")
        })?;
    let row_packed_bytes = cols.div_ceil(2);
    let groups_per_row = cols.div_ceil(group_size);
    let packed_bytes = rows
        .checked_mul(row_packed_bytes)
        .context("dense q4 packed byte length overflow")?;
    let groups = rows
        .checked_mul(groups_per_row)
        .context("dense q4 group count overflow")?;
    let scale_bias_bytes = dense_scale_bias_dtype_size(scale_bias_dtype)
        .with_context(|| format!("unsupported dense q4 scale/bias dtype {scale_bias_dtype}"))?;
    let scales_bytes = groups
        .checked_mul(scale_bias_bytes)
        .context("dense q4 scale byte length overflow")?;
    let total_bytes = packed_bytes
        .checked_add(scales_bytes)
        .and_then(|value| value.checked_add(scales_bytes))
        .context("dense q4 total byte length overflow")?;
    Ok(DenseQ4Layout {
        rows,
        cols,
        row_packed_bytes,
        groups_per_row,
        packed_bytes,
        scales_bytes,
        scale_bias_bytes,
        total_bytes,
    })
}

fn dense_scale_bias_dtype_size(dtype: &str) -> Result<usize> {
    match dtype.to_ascii_uppercase().as_str() {
        EXPERT_SCALE_BIAS_DTYPE_F32 | "FLOAT32" | "FP32" => Ok(4),
        "BF16" | "BFLOAT16" => Ok(2),
        other => bail!("unsupported q4 scale/bias dtype {other}"),
    }
}

pub(crate) fn validate_dense_matvec_shape(
    entry: &RuntimeTensorEntry,
    canonical_name: &str,
    expected_rows: usize,
    input_len: usize,
) -> Result<(usize, usize)> {
    let expected_shape = [expected_rows, input_len];
    match entry.shape.as_slice() {
        [rows, cols] if *rows == expected_rows && *cols == input_len => Ok((*rows, *cols)),
        _ => bail!(
            "Flash-MoE dense tensor {canonical_name} shape mismatch: expected shape {:?}, actual shape {:?}, input length {input_len}",
            expected_shape,
            entry.shape
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResidentStaticTensorRef {
    pub(crate) tensor_name: String,
    pub(crate) byte_offset: u64,
    pub(crate) dtype: String,
    pub(crate) values: usize,
    pub(crate) element_size: usize,
}

impl ResidentStaticTensorRef {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        expected_values: usize,
        allowed_dtypes: &[&str],
    ) -> Result<Option<Self>> {
        if entry.quantization != TensorQuantization::None {
            return Ok(None);
        }
        if !allowed_dtypes
            .iter()
            .any(|allowed| entry.dtype.eq_ignore_ascii_case(allowed))
        {
            return Ok(None);
        }
        let Some(element_size) = dense_dtype_size(&entry.dtype) else {
            return Ok(None);
        };
        let expected_bytes = expected_values
            .checked_mul(element_size)
            .context("resident static tensor byte length overflow")?;
        if entry.byte_len as usize != expected_bytes {
            return Ok(None);
        }
        if entry
            .byte_offset
            .checked_add(entry.byte_len)
            .map_or(true, |end| end > store_len)
        {
            return Ok(None);
        }
        if entry.byte_offset % element_size as u64 != 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            tensor_name: tensor_name.to_string(),
            byte_offset: entry.byte_offset,
            dtype: entry.dtype.clone(),
            values: expected_values,
            element_size,
        }))
    }
}

fn dense_dtype_size(dtype: &str) -> Option<usize> {
    match dtype.to_ascii_uppercase().as_str() {
        "BF16" | "BFLOAT16" | "F16" | "FLOAT16" | "FP16" => Some(2),
        "F32" | "FLOAT32" | "FP32" => Some(4),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseMmapMatvecProjection {
    pub(crate) tensor_name: String,
    pub(crate) byte_offset: u64,
    pub(crate) dtype: String,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) output_width: usize,
}

impl DenseMmapMatvecProjection {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        output_width: usize,
        input_len: usize,
        element_size: usize,
    ) -> Result<Self> {
        let (rows, cols) =
            validate_dense_matvec_shape(entry, tensor_name, output_width, input_len)?;
        let row_bytes = cols
            .checked_mul(element_size)
            .context("dense tensor resident row byte length overflow")?;
        let byte_len = rows
            .checked_mul(row_bytes)
            .context("dense tensor resident byte length overflow")?;
        if entry
            .byte_offset
            .checked_add(byte_len as u64)
            .map_or(true, |end| end > store_len)
        {
            bail!(
                "Flash-MoE dense tensor {} byte range {}..{} exceeds dense store length {}",
                tensor_name,
                entry.byte_offset,
                entry.byte_offset.saturating_add(byte_len as u64),
                store_len
            );
        }
        Ok(Self {
            tensor_name: tensor_name.to_string(),
            byte_offset: entry.byte_offset,
            dtype: entry.dtype.clone(),
            rows,
            cols,
            output_width,
        })
    }

    pub(crate) fn stride(&self) -> usize {
        self.cols
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseQ4MmapMatvecProjection {
    pub(crate) tensor_name: String,
    pub(crate) packed_byte_offset: u64,
    pub(crate) scales_byte_offset: u64,
    pub(crate) biases_byte_offset: u64,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) output_width: usize,
    pub(crate) row_packed_bytes: usize,
    pub(crate) groups_per_row: usize,
    pub(crate) group_size: usize,
    pub(crate) scale_bias_dtype: String,
}

impl DenseQ4MmapMatvecProjection {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        output_width: usize,
        input_len: usize,
    ) -> Result<Option<Self>> {
        let TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } = &entry.quantization
        else {
            return Ok(None);
        };
        let (rows, cols) =
            validate_dense_matvec_shape(entry, tensor_name, output_width, input_len)?;
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&entry.shape, *group_size, scale_bias_dtype)?;
        if entry.byte_len as usize != layout.total_bytes
            || rows != layout.rows
            || cols != layout.cols
        {
            return Ok(None);
        }
        if entry
            .byte_offset
            .checked_add(entry.byte_len)
            .map_or(true, |end| end > store_len)
        {
            return Ok(None);
        }
        let packed_byte_offset = entry.byte_offset;
        let scales_byte_offset = entry
            .byte_offset
            .checked_add(layout.packed_bytes as u64)
            .context("dense q4 projection scales offset overflow")?;
        let biases_byte_offset = scales_byte_offset
            .checked_add(layout.scales_bytes as u64)
            .context("dense q4 projection biases offset overflow")?;
        Ok(Some(Self {
            tensor_name: tensor_name.to_string(),
            packed_byte_offset,
            scales_byte_offset,
            biases_byte_offset,
            rows,
            cols,
            output_width,
            row_packed_bytes: layout.row_packed_bytes,
            groups_per_row: layout.groups_per_row,
            group_size: *group_size,
            scale_bias_dtype: scale_bias_dtype.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_quantization_defaults_to_unquantized_dense() {
        assert_eq!(TensorQuantization::default(), TensorQuantization::None);
    }

    #[test]
    fn tensor_quantization_q4_defaults_scale_bias_dtype_for_legacy_manifests() {
        let quantization: TensorQuantization =
            serde_json::from_str(r#"{"Q4":{"group_size":16,"format":"dense-q4"}}"#).unwrap();

        assert_eq!(
            quantization,
            TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "F32".to_string(),
            }
        );
    }

    #[test]
    fn dense_tensor_ref_preserves_runtime_binding_offsets() {
        let tensor = DenseTensorRef {
            tensor: "model.embed_tokens.weight".to_string(),
            shard: "model-00001.safetensors".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![8, 4],
            source_offsets: [128, 192],
            runtime_offset: 4096,
            byte_len: 64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        };

        assert_eq!(tensor.runtime_offset, 4096);
        assert_eq!(tensor.byte_len, 64);
        assert_eq!(tensor.quantization, TensorQuantization::None);
    }

    #[test]
    fn tensor_registry_builds_dense_aliases_from_manifest() {
        let manifest = FlashMoeManifest {
            model: "fixture".to_string(),
            cache_version: "test".to_string(),
            dense_shards: Vec::new(),
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: "model.language_model.layers.7.self_attn.q_proj.weight".to_string(),
                shard: "model.safetensors".to_string(),
                dtype: "BF16".to_string(),
                shape: vec![4, 8],
                source_offsets: [0, 64],
                runtime_offset: 4096,
                byte_len: 64,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };

        let registry = TensorRegistry::from_manifest(&manifest);
        let alias = registry
            .tensor("model.layers.7.self_attn.q_proj.weight")
            .unwrap();

        assert_eq!(
            alias.name,
            "model.language_model.layers.7.self_attn.q_proj.weight"
        );
        assert_eq!(alias.byte_offset, 4096);
        assert_eq!(alias.alignment, TENSOR_ALIGNMENT);
        assert!(registry.has_tensor_with_prefix("model.layers.7"));
    }

    #[test]
    fn tensor_registry_keeps_expert_manifest_refs_as_import_compatibility() {
        let manifest = FlashMoeManifest {
            model: "fixture".to_string(),
            cache_version: "test".to_string(),
            dense_shards: Vec::new(),
            dense_tensors: Vec::new(),
            expert_tensors: vec![ExpertTensorRef {
                tensor: "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
                shard: "model.safetensors".to_string(),
                layer: Some(0),
                expert: Some(1),
                dtype: Some("F32".to_string()),
                shape: vec![2, 4],
                source_offsets: Some([128, 256]),
                q4_sources: None,
            }],
        };

        let registry = TensorRegistry::from_manifest(&manifest);
        let tensor = registry
            .tensor("model.layers.0.mlp.experts.1.gate_proj.weight")
            .unwrap();

        assert_eq!(tensor.byte_offset, 128);
        assert_eq!(tensor.byte_len, 128);
        assert_eq!(
            tensor.quantization,
            TensorQuantization::Q4 {
                group_size: GROUP_SIZE,
                format: ExpertQuantization::FourBitProduction.as_str().to_string(),
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
            }
        );
    }

    #[test]
    fn dense_mmap_projection_stride_uses_runtime_cols() {
        let projection = DenseMmapMatvecProjection {
            tensor_name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            byte_offset: 4096,
            dtype: "BF16".to_string(),
            rows: 16,
            cols: 32,
            output_width: 64,
        };

        assert_eq!(projection.stride(), 32);
    }

    #[test]
    fn dense_mmap_projection_descriptor_resolves_entry_bounds() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![4, 8],
            byte_offset: 64,
            byte_len: 64,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let projection = DenseMmapMatvecProjection::from_entry(
            "model.layers.0.self_attn.q_proj.weight",
            &entry,
            256,
            4,
            8,
            2,
        )
        .unwrap();

        assert_eq!(projection.byte_offset, 64);
        assert_eq!(projection.rows, 4);
        assert_eq!(projection.cols, 8);
        assert_eq!(projection.output_width, 4);
    }

    #[test]
    fn resident_static_tensor_descriptor_resolves_offsets_and_dtype() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.self_attn.conv1d.weight".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![8],
            byte_offset: 16,
            byte_len: 16,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let resident =
            ResidentStaticTensorRef::from_entry(&entry.name, &entry, 64, 8, &["BF16", "BFLOAT16"])
                .unwrap()
                .unwrap();

        assert_eq!(resident.tensor_name, entry.name);
        assert_eq!(resident.byte_offset, 16);
        assert_eq!(resident.values, 8);
        assert_eq!(resident.element_size, 2);
    }

    #[test]
    fn resident_static_tensor_descriptor_rejects_wrong_layout_without_fallback() {
        let mut entry = RuntimeTensorEntry {
            name: "model.layers.0.self_attn.A_log".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4],
            byte_offset: 4,
            byte_len: 16,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        assert!(
            ResidentStaticTensorRef::from_entry(&entry.name, &entry, 32, 4, &["F32"])
                .unwrap()
                .is_some()
        );

        entry.byte_len = 12;
        assert!(
            ResidentStaticTensorRef::from_entry(&entry.name, &entry, 32, 4, &["F32"])
                .unwrap()
                .is_none()
        );

        entry.byte_len = 16;
        entry.quantization = TensorQuantization::Q4 {
            group_size: GROUP_SIZE,
            format: "mlx".to_string(),
            scale_bias_dtype: "F32".to_string(),
        };
        assert!(
            ResidentStaticTensorRef::from_entry(&entry.name, &entry, 32, 4, &["F32"])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn dense_q4_projection_descriptor_carries_one_binding_shape() {
        let projection = DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            packed_byte_offset: 128,
            scales_byte_offset: 256,
            biases_byte_offset: 512,
            rows: 16,
            cols: 32,
            output_width: 16,
            row_packed_bytes: 16,
            groups_per_row: 2,
            group_size: 16,
            scale_bias_dtype: "BF16".to_string(),
        };

        assert_eq!(projection.row_packed_bytes, projection.cols.div_ceil(2));
        assert_eq!(projection.groups_per_row, 2);
        assert_eq!(projection.output_width, projection.rows);
    }

    #[test]
    fn dense_q4_layout_accounts_for_scale_bias_dtype() {
        let layout = dense_q4_layout_with_scale_bias_dtype(&[2, 4], 16, "BF16").unwrap();

        assert_eq!(layout.rows, 2);
        assert_eq!(layout.cols, 4);
        assert_eq!(layout.row_packed_bytes, 2);
        assert_eq!(layout.groups_per_row, 1);
        assert_eq!(layout.packed_bytes, 4);
        assert_eq!(layout.scales_bytes, 4);
        assert_eq!(layout.scale_bias_bytes, 2);
        assert_eq!(layout.total_bytes, 12);
    }

    #[test]
    fn dense_q4_projection_descriptor_resolves_offsets_from_entry() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            byte_offset: 128,
            byte_len: 12,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "BF16".to_string(),
            },
        };

        let projection = DenseQ4MmapMatvecProjection::from_entry(
            "model.layers.0.mlp.gate_proj.weight",
            &entry,
            256,
            2,
            4,
        )
        .unwrap()
        .unwrap();

        assert_eq!(projection.packed_byte_offset, 128);
        assert_eq!(projection.scales_byte_offset, 132);
        assert_eq!(projection.biases_byte_offset, 136);
        assert_eq!(projection.rows, 2);
        assert_eq!(projection.cols, 4);
        assert_eq!(projection.scale_bias_dtype, "BF16");
    }

    #[test]
    fn dense_q4_projection_descriptor_rejects_missing_capacity() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            byte_offset: 128,
            byte_len: 12,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "BF16".to_string(),
            },
        };

        let projection = DenseQ4MmapMatvecProjection::from_entry(
            "model.layers.0.mlp.gate_proj.weight",
            &entry,
            139,
            2,
            4,
        )
        .unwrap();

        assert_eq!(projection, None);
    }
}
