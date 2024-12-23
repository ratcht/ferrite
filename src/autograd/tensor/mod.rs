// autograd/scalar/mod.rs
mod base;
mod shape;
mod utils;
mod ops;
mod creation;
mod backward;
mod function;

// Re-export everything we want to be publicly accessible
pub use base::*;
pub use shape::*;
pub use utils::*;
pub use ops::*;
pub use creation::*;
pub use backward::*;    // Export all public items from backward
pub use function::*;