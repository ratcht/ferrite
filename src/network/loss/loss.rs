use crate::autograd::tensor::*;

pub trait Loss {
  fn loss(&self, x: &Tensor, y: &Tensor) -> Tensor;
}

