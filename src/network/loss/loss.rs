use crate::tensor::*;

pub trait LossTrait {
  fn loss(&self, x: &Tensor, y: &Tensor) -> Tensor;
}

