//! Flash-MoE inspired inference backend facade.
//!
//! The implementation is being split out of the historical monolith in
//! `legacy`. Keep this module as the stable public surface while internals move
//! behind smaller modules.

mod capabilities;
mod experts;
mod legacy;
mod math;
mod metal;
mod model_family;
mod scheduler;
mod state;
mod types;
mod weights;

pub use capabilities::*;
pub use experts::*;
pub use legacy::*;
pub use math::*;
pub use metal::METAL_SHADERS;
pub use model_family::*;
pub use scheduler::*;
pub use types::*;
pub use weights::{
    DenseQ4SourceRefs, DenseTensorRef, ExpertTensorRef, FlashMoeManifest, RuntimeTensorEntry,
    TensorQuantization, TensorRegistry,
};
