use crate::{reduce_grad, tensor::*};
use super::super::grad::*;


#[derive(Debug)]
pub struct AddGrad {
  lhs: Tensor,
  rhs: Tensor,
  output: Tensor,
}

impl AddGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Self {
    AddGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for AddGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let reduced_grad = reduce_grad!(out_grad, self.lhs.tensor().shape());
    
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
    
    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      let reduced_grad = reduce_grad!(out_grad, self.rhs.tensor().shape());
      
      rhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}


#[derive(Debug)]
pub struct SubGrad {
  lhs: Tensor,
  rhs: Tensor,
  output: Tensor,
}

impl SubGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Self {
    SubGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for SubGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let reduced_grad = reduce_grad!(out_grad, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
    
    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      let grad_for_rhs = &*out_grad * -1.;
      let reduced_grad = reduce_grad!(grad_for_rhs, self.rhs.tensor().shape());
      
      rhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}

#[derive(Debug)]
pub struct MulGrad {
  lhs: Tensor,
  rhs: Tensor,
  output: Tensor,
}

impl MulGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Self {
    MulGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for MulGrad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad * self.rhs.tensor();
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
    
    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      let grad_for_rhs = &*out_grad * self.lhs.tensor();
      
      let reduced_grad = reduce_grad!(grad_for_rhs, self.rhs.tensor().shape());

      rhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}


#[derive(Debug)]
pub struct DivGrad {
  lhs: Tensor,
  rhs: Tensor,
  output: Tensor,
}

impl DivGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Self {
    DivGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for DivGrad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad / self.rhs.tensor();
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
    
    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      // Form grad for rhs
      let grad_for_rhs = &(&*out_grad * self.lhs.tensor()).mul_f32(-1.) / &(self.rhs.tensor().pow_f32(2.));
      
      let reduced_grad = reduce_grad!(grad_for_rhs, self.rhs.tensor().shape());

      rhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}

#[derive(Debug)]
pub struct PowF32Grad {
  lhs: Tensor,
  rhs: f32,
  output: Tensor,
}

impl PowF32Grad {
  pub fn new(lhs: &Tensor, rhs: f32, output: &Tensor) -> Self {
    PowF32Grad {
      lhs: lhs.clone(),
      rhs: rhs,
      output: output.clone(),
    }
  }
}

impl GradientFunction for PowF32Grad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &(&*out_grad * self.rhs) * &self.lhs.tensor().pow_f32(self.rhs-1.);
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);

    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}


#[derive(Debug)]
pub struct AddF32Grad {
  lhs: Tensor,
  rhs: f32,
  output: Tensor,
}

impl AddF32Grad {
  pub fn new(lhs: &Tensor, rhs: f32, output: &Tensor) -> Self {
    AddF32Grad {
      lhs: lhs.clone(),
      rhs: rhs,
      output: output.clone(),
    }
  }
}

impl GradientFunction for AddF32Grad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let reduced_grad = reduce_grad!(out_grad, self.lhs.tensor().shape());
    
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}


#[derive(Debug)]
pub struct SubF32Grad {
  lhs: Tensor,
  rhs: f32,
  output: Tensor,
}

impl SubF32Grad {
  pub fn new(lhs: &Tensor, rhs: f32, output: &Tensor) -> Self {
    SubF32Grad {
      lhs: lhs.clone(),
      rhs: rhs,
      output: output.clone(),
    }
  }
}

impl GradientFunction for SubF32Grad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let reduced_grad = reduce_grad!(out_grad, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}

#[derive(Debug)]
pub struct MulF32Grad {
  lhs: Tensor,
  rhs: f32,
  output: Tensor,
}

impl MulF32Grad {
  pub fn new(lhs: &Tensor, rhs: f32, output: &Tensor) -> Self {
    MulF32Grad {
      lhs: lhs.clone(),
      rhs: rhs,
      output: output.clone(),
    }
  }
}

impl GradientFunction for MulF32Grad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad * self.rhs;
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}


#[derive(Debug)]
pub struct DivF32Grad {
  lhs: Tensor,
  rhs: f32,
  output: Tensor,
}

impl DivF32Grad {
  pub fn new(lhs: &Tensor, rhs: f32, output: &Tensor) -> Self {
    DivF32Grad {
      lhs: lhs.clone(),
      rhs: rhs,
      output: output.clone(),
    }
  }
}

impl GradientFunction for DivF32Grad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad / self.rhs;
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}


#[derive(Debug)]
pub struct AbsGrad {
  lhs: Tensor,
  output: Tensor,
}

impl AbsGrad {
  pub fn new(lhs: &Tensor, output: &Tensor) -> Self {
    AbsGrad {
      lhs: lhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for AbsGrad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad * &self.lhs.tensor().sign();
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}
