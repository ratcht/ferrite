//! Ferrite - A lightweight deep learning framework
//! 
//! This crate provides a dynamic computation graph with automatic differentiation,
//! designed for building and training neural networks.

mod tensor_storage;
mod autograd;
mod network;

// Import and re-export macros globally
#[macro_use]
mod macros;

// Re-export the main types
pub use tensor_storage::*;
pub use autograd::tensor::Tensor;
pub use network::*;

// Version of the crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Optional prelude module for convenient imports
pub mod prelude {
  pub use crate::tensor_storage::*;
  pub use crate::autograd::tensor::Tensor;
  pub use crate::network::*;
  pub use crate::layer; // Re-export the macro
}
