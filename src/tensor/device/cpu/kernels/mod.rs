// tensor_storage/mod.rs
mod arithmetic;        // Internal module
mod blas;         // Internal module
mod reduction;    // Internal module
mod transform;       // Internal module

// Re-export what you want public
pub use arithmetic::*;
pub use blas::*;
pub use reduction::*;
pub use transform::*;