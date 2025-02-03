//! Ferrite - A lightweight deep learning framework
//! 
//! This crate provides a dynamic computation graph with automatic differentiation,
//! designed for building and training neural networks.

#![allow(unreachable_patterns)]
#![allow(unused_variables)]
#![allow(dead_code)]

mod autograd;
mod tensor;
mod network;

// Import and re-export macros globally
#[macro_use]
mod macros;

// Re-export the main types
pub use tensor::*;
pub use network::*;
pub use autograd::*;

// Version of the crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Optional prelude module for convenient imports
pub mod prelude {
  pub use crate::tensor::*;
  pub use crate::autograd::*;
  pub use crate::network::*;
  pub use crate::layer; // Re-export the macro
  pub use Layer::Module;

}
