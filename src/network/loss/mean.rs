pub struct MSE {
  reduction: &str
}

impl Loss for MSE {

}


#[derive(Debug)]
pub struct MSEGrad {
  lhs: Tensor,
  rhs: Tensor,
  output: Tensor,
}

impl MSEGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Self {
    MSEGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for MSEGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let mut reduced_grad = out_grad.clone();
      let lhs_shape = self.lhs.tensor().shape();
      
      // Reduce along broadcasted dimensions
      for (dim, (grad_size, lhs_size)) in out_grad.shape().iter()
        .zip(lhs_shape.iter())
        .enumerate() 
      {
        if lhs_size == &1 && grad_size != &1 {
          let mut sum_dims = vec![false; out_grad.shape().len()];
          sum_dims[dim] = true;
          reduced_grad = reduced_grad.sum_dim(&sum_dims);
        }
      }
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
    
    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      let mut reduced_grad = out_grad.clone();
      let rhs_shape = self.rhs.tensor().shape();
      
      for (dim, (grad_size, rhs_size)) in out_grad.shape().iter()
        .zip(rhs_shape.iter())
        .enumerate() 
      {
        if rhs_size == &1 && grad_size != &1 {
          let mut sum_dims = vec![false; out_grad.shape().len()];
          sum_dims[dim] = true;
          reduced_grad = reduced_grad.sum_dim(&sum_dims);
        }
      }
      
      rhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}
