use std::rc::Rc;

use crate::*;

pub trait BlasOps {
  fn matmul(&self, other: &Self, trans_a: bool, trans_b: bool) -> Self;
}


impl BlasOps for Storage {
  fn matmul(&self, other: &Self, trans_a: bool, trans_b: bool) -> Self {
    match_storage!(binary self, matmul, other, trans_a, trans_b)
  }
}

impl BlasOps for Tensor {
  fn matmul(&self, other: &Self, trans_a: bool, trans_b: bool) -> Self {
    let tensor = self.tensor().matmul(other.tensor(), trans_a, trans_b);
    
    let requires_grad = *self.requires_grad() || *other.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(MatMulGrad::new(
        self,
        other,
        &result,
        trans_a,
        trans_b
      ))));
    }
    
    result
  }
}