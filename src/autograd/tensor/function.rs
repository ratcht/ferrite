use std::rc::Rc;
use std::cell::RefCell;
use crate::Display;
use crate::tensor_storage::*;

use super::base::*;


pub trait GradientFunction {
  fn backward(&self, grad: &Tensor);
  fn prev(&self) -> Vec<Rc<RefCell<Tensor>>>;
}

impl std::fmt::Debug for dyn GradientFunction {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "GradientFunction")
  }
}

impl std::fmt::Display for dyn GradientFunction {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "GradientFunction")
  }
}

pub struct AddGrad {
  lhs: Rc<RefCell<Tensor>>,
  rhs: Rc<RefCell<Tensor>>,
}

impl AddGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor) -> Self {
    AddGrad {
      lhs: Rc::new(RefCell::new(lhs.clone())),
      rhs: Rc::new(RefCell::new(rhs.clone()))
    }
  }
}

impl GradientFunction for AddGrad {
  fn backward(&self, grad: &Tensor) {
    // Propagate gradient to both inputs
    if let Some(lhs_grad) = &self.lhs.borrow().grad() {
      lhs_grad.borrow_mut().add_tensor_assign(grad.tensor());
    }
    if let Some(rhs_grad) = &self.rhs.borrow().grad() {
      rhs_grad.borrow_mut().add_tensor_assign(grad.tensor());
    }
  }
  
  fn prev(&self) -> Vec<Rc<RefCell<Tensor>>> {
    vec![self.lhs.clone(), self.rhs.clone()]
  }
}