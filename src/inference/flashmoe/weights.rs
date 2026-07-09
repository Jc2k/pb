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
