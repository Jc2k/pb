//! Flash-MoE inspired inference backend facade.
//!
//! The stable public surface is composed from capability, storage, scheduling,
//! execution, and adapter owners. Historical `legacy` code is test-only.

mod cache;
mod capabilities;
mod experts;
mod math;
mod metal;
mod model_family;
mod planning;
mod runtime;
mod safetensors;
mod scheduler;
mod state;
#[cfg(test)]
mod test_fixtures;
mod text;
mod types;
mod vision;
mod weights;

pub use cache::{
    build_cache_from_hf_snapshot, build_cache_from_hf_snapshot_with_quantization,
    expected_hf_files, expected_vl_hf_files,
};
pub use capabilities::*;
pub use experts::*;
pub use math::*;
pub use metal::METAL_SHADERS;
pub use model_family::*;
pub use planning::*;
pub use runtime::{FlashMoeEngine, load, load_with_progress};
pub use scheduler::*;
pub use types::*;
pub use vision::{ImagePreprocessor, Qwen3VLVisionConfig, VisionEncoder, VisionEncoding};
pub use weights::{
    DenseQ4SourceRefs, DenseStore, DenseTensorRef, ExpertTensorRef, FlashMoeManifest,
    RuntimeTensorEntry, TensorQuantization, TensorRegistry,
};
