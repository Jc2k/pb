//! Flash-MoE inspired inference backend facade.
//!
//! The implementation is being split out of the historical monolith in
//! `legacy`. Keep this module as the stable public surface while internals move
//! behind smaller modules.

mod cache;
mod capabilities;
mod experts;
mod legacy;
mod math;
mod metal;
mod model_family;
mod planning;
mod runtime;
mod scheduler;
mod state;
mod text;
mod types;
mod vision;
mod weights;

pub use cache::{build_cache_from_hf_snapshot, expected_hf_files, expected_vl_hf_files};
pub use capabilities::*;
pub use experts::*;
pub use legacy::*;
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
