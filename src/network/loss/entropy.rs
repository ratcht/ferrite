use super::loss::*;
use crate::tensor::*;

pub struct CrossEntropyLoss {
  
}

impl CrossEntropyLoss {
  pub fn new() -> Self {
    CrossEntropyLoss { }
  }
}

impl LossTrait for CrossEntropyLoss {
  fn loss(&self, x: &Tensor, y: &Tensor) -> Tensor {
    unimplemented!()

  }
}