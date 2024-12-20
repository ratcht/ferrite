use super::base::Tensor;  // Import from parent module's base.rs

pub trait TensorOps {
  fn reshape(&mut self, new_shape: Vec<usize>);
  fn transpose(&mut self);
  fn flatten(&mut self);
  fn squeeze(&mut self);
  fn unsqueeze(&mut self, dim: usize);
}


