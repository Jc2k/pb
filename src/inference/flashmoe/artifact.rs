//! Shared contracts for canonical FlashMoe cache artifacts.
//!
//! This module deliberately depends on neither dense weights nor expert I/O so
//! both owners can validate the same manifest and quantization vocabulary
//! without introducing a dependency cycle.

use anyhow::{Result, bail};

pub(crate) const EXPERT_SCALE_BIAS_DTYPE_F32: &str = "F32";
pub(crate) const EXPERT_SCALE_BIAS_DTYPE_BF16: &str = "BF16";
pub(crate) const EXPERT_SCALE_DTYPE_E8M0: &str = "E8M0";
pub(crate) const EXPERT_PACK_SCALE_BIAS_DTYPE: &str = EXPERT_SCALE_BIAS_DTYPE_BF16;

pub(crate) trait AggregateExpertTensor {
    fn aggregate_tensor_name(&self) -> &str;
    fn aggregate_tensor_shape(&self) -> &[usize];
    fn aggregate_tensor_has_native_q4(&self) -> bool;
    fn aggregate_tensor_is_mxfp4(&self) -> bool {
        false
    }
}

pub(crate) trait ExpertSourceTensor: AggregateExpertTensor {
    fn expert_source_offsets(&self) -> Option<[u64; 2]>;
}

pub(crate) fn expert_scale_bias_dtype_size(dtype: &str) -> Result<usize> {
    match dtype.to_ascii_uppercase().as_str() {
        EXPERT_SCALE_BIAS_DTYPE_F32 | "FLOAT32" | "FP32" => Ok(4),
        EXPERT_SCALE_BIAS_DTYPE_BF16 | "BFLOAT16" => Ok(2),
        EXPERT_SCALE_DTYPE_E8M0 => Ok(1),
        other => bail!("unsupported q4 scale/bias dtype {other}"),
    }
}
