use crate::tensor::*;
use super::super::grad::*;


#[derive(Debug)]
pub struct MatMulGrad {
  lhs: Tensor,
  rhs: Tensor,
  output: Tensor,
  trans_a: bool,
  trans_b: bool
}

impl MatMulGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor, trans_a: bool, trans_b: bool,) -> Self {
    MatMulGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
      trans_a: trans_a,
      trans_b: trans_b,
    }
  }
}

impl GradientFunction for MatMulGrad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Case: C = t × w^T  (your test case)
    // t: (2,3), w: (2,3), C: (2,2)
    // dL/dt = dL/dC × w^T  - note: no transpose here since we already have w^T
    // dL/dw = (dL/dC × t)^T

    if let Some(lhs_grad) = &self.lhs.grad() {
      // For input t: dL/dt = dL/dC × w^T
      let grad_for_lhs = if !self.trans_b {
        out_grad.matmul(self.rhs.tensor(), false, true)
      } else {
        out_grad.matmul(self.rhs.tensor(), false, false)
      };
      lhs_grad.borrow_mut().add_tensor_assign(&grad_for_lhs);
    }

    if let Some(rhs_grad) = &self.rhs.grad() {
      // For weight w: dL/dw = (dL/dC × t)^T
      let grad_for_rhs = if !self.trans_b {
        self.lhs.tensor().matmul(&out_grad, true, false)
      } else {
        out_grad.matmul(&self.lhs.tensor(), true, false)
      };
      rhs_grad.borrow_mut().add_tensor_assign(&grad_for_rhs);
    }
  } 

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}