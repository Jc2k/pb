use serde::{Deserialize, Serialize};

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
}
