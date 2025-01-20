use crate::tensor::*;

pub trait Loss {
  fn loss(&self, x: &Tensor, y: &Tensor) -> Tensor;
}

