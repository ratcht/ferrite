use super::loss::*;
use crate::tensor::*;

pub struct CrossEntropyLoss {
}

impl CrossEntropyLoss {
  pub fn new(reduction: &str) -> Self {
    unimplemented!()
  }
}

impl Loss for CrossEntropyLoss {
  fn loss(&self, x: &Tensor, y: &Tensor) -> Tensor {
    unimplemented!()
  }
}