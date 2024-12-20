// autograd/scalar/mod.rs
mod graph;
mod operations;
mod backward;

// Re-export everything we want to be publicly accessible
pub use graph::{Graph, Scalar, Value};  // Assuming Graph is your main struct
pub use operations::*;  // Export all public items from operations
pub use backward::*;    // Export all public items from backward