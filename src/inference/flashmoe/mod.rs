//! Flash-MoE inspired inference backend facade.
//!
//! The stable public surface is composed from capability, storage, scheduling,
//! execution, and adapter owners. Historical `legacy` code is test-only.

mod artifact;
mod cache;
mod capabilities;
mod constraints;
#[cfg(test)]
pub(crate) use constraints::{terminal_tool_output_is_complete, validate_native_tool_schema};
mod deepseek;
mod deepseek_metal;
mod deepseek_session;
mod experts;
mod generation_progress;
mod gguf;
mod math;
mod metal;
mod model_family;
mod planning;
mod pool;
mod runtime;
mod safetensors;
mod scheduler;
mod session_cache;
mod state;
#[cfg(test)]
mod test_fixtures;
mod text;
mod types;
mod vision;
mod weights;

/// Versioned proof that FlashMoe inference enforces the bounded Metal resource policy required by
/// harness model matrices. A zero value means real-model harness evaluation must refuse FlashMoe.
pub const HARNESS_RESOURCE_POLICY_VERSION: u32 = 1;

pub use cache::{
    build_cache_from_hf_snapshot, build_cache_from_hf_snapshot_with_quantization,
    expected_hf_files, expected_vl_hf_files,
};
pub use deepseek::{
    DEEPSEEK_V4_FLASH_CACHE_VERSION, DEEPSEEK_V4_FLASH_FILENAME, DEEPSEEK_V4_FLASH_MODEL,
    DEEPSEEK_V4_FLASH_REPOSITORY, build_deepseek_v4_flash_cache_from_gguf, is_deepseek_v4_flash,
};
pub use model_family::*;
pub use planning::*;
pub use pool::{FlashMoeRuntimeHandle, load_shared, reap_idle_shared_runtimes};
pub use runtime::{
    FlashMoeEngine, FlashMoeLoadOptions, load, load_with_options, load_with_options_and_progress,
    load_with_progress,
};
pub use types::*;
pub use vision::{ImagePreprocessor, Qwen3VLVisionConfig, VisionEncoder, VisionEncoding};
pub use weights::{
    DenseQ4SourceFormat, DenseQ4SourceRefs, DenseStore, DenseTensorRef, ExpertTensorRef,
    FlashMoeManifest, RuntimeTensorEntry, TensorQuantization, TensorRegistry,
};
