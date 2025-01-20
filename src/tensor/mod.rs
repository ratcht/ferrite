// autograd/scalar/mod.rs
mod base;
mod shape;
mod utils;
mod ops;
mod creation;

// Re-export everything we want to be publicly accessible
pub use base::*;
pub use shape::*;
pub use utils::*;
pub use ops::*;
pub use creation::*;