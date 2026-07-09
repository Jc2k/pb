//! Flash-MoE inspired inference backend facade.
//!
//! The implementation is being split out of the historical monolith in
//! `legacy`. Keep this module as the stable public surface while internals move
//! behind smaller modules.

mod capabilities;
mod experts;
mod legacy;
mod math;
mod model_family;
mod types;

pub use capabilities::*;
pub use experts::*;
pub use legacy::*;
pub use math::*;
pub use model_family::*;
pub use types::*;
