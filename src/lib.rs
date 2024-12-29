//! Ferrite - A lightweight deep learning framework
//! 
//! This crate provides a dynamic computation graph with automatic differentiation,
//! designed for building and training neural networks.

mod tensor_storage;
mod autograd;
mod network;

// Re-export the main types
pub use tensor_storage::{TensorCreation, TensorOps, TensorShape};
pub use autograd::tensor::Tensor;
pub use network::{Module, Parameter, Segment};

// Version of the crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Optional prelude module for convenient imports
pub mod prelude {
  pub use crate::tensor_storage::{TensorCreation, TensorOps, TensorShape};
  pub use crate::autograd::tensor::Tensor;
  pub use crate::network::{Module, Parameter, Segment};
}