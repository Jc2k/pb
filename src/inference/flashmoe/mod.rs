//! Flash-MoE inspired inference backend facade.
//!
//! The stable public surface is composed from capability, storage, scheduling,
//! execution, and adapter owners. Historical `legacy` code is test-only.

mod cache;
mod capabilities;
mod deepseek;
mod deepseek_metal;
mod deepseek_session;
mod experts;
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
pub use capabilities::*;
pub use deepseek::*;
pub use experts::*;
pub use math::*;
pub use metal::METAL_SHADERS;
pub use model_family::*;
pub use planning::*;
pub use pool::{FlashMoeRuntimeHandle, load_shared, reap_idle_shared_runtimes};
pub use runtime::{FlashMoeEngine, load, load_with_progress};
pub use scheduler::*;
pub use types::*;
pub use vision::{ImagePreprocessor, Qwen3VLVisionConfig, VisionEncoder, VisionEncoding};
pub use weights::{
    DenseQ4SourceFormat, DenseQ4SourceRefs, DenseStore, DenseTensorRef, ExpertTensorRef,
    FlashMoeManifest, RuntimeTensorEntry, TensorQuantization, TensorRegistry,
};
