// tensor_storage/mod.rs
mod base;        // Internal module
mod traits;
mod creation;
mod utils;

// Re-export what you want public
pub use base::*;
pub use traits::*;
pub use creation::*;
pub use utils::*;