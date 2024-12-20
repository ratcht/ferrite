// tensor/mod.rs
mod base;        // Internal module
mod ops;         // Internal module
mod creation;    // Internal module
mod shape;       // Internal module
mod util;     // Internal module

// Re-export what you want public
pub use base::Tensor;
pub use ops::TensorOps;
pub use creation::TensorCreation;
pub use shape::TensorShape;