//! Flash-MoE inspired inference backend facade.
//!
//! The implementation is being split out of the historical monolith in
//! `legacy`. Keep this module as the stable public surface while internals move
//! behind smaller modules.

mod legacy;
mod math;
mod types;

pub use legacy::*;
pub use math::*;
pub use types::*;
