// tensor_storage/mod.rs
mod base;        // Internal module
mod ops;         // Internal module
mod creation;    // Internal module
mod shape;       // Internal module
mod utils;     // Internal module

// Re-export what you want public
pub use base::TensorStorage;
pub use ops::TensorOps;
pub use creation::TensorCreation;
pub use shape::TensorShape;
pub use utils::Display;