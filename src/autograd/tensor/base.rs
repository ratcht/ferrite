use std::rc::Rc;
use std::cell::RefCell;
use crate::tensor_storage::*;
use super::function::*;

#[derive(Clone)]
pub struct Tensor {
  tensor: TensorStorage,
  requires_grad: bool,
  grad_fn: Option<Rc<dyn GradientFunction>>,
  grad: Option<Rc<RefCell<TensorStorage>>>,
}

impl Tensor {
  pub fn new(tensor: TensorStorage, requires_grad: bool, grad_fn:Option<Rc<dyn GradientFunction>>, grad: Option<Rc<RefCell<TensorStorage>>>) -> Tensor {
    Tensor {
      tensor,
      requires_grad,
      grad_fn,
      grad
    }
  }

  pub fn tensor(&self) -> &TensorStorage {
    &self.tensor
  }

  pub fn tensor_mut(&mut self) -> &mut TensorStorage {
    &mut self.tensor
  }


  pub fn requires_grad(&self) -> &bool {
    &self.requires_grad
  }

  pub fn set_requires_grad(&mut self, requires_grad: bool) {
    self.requires_grad = requires_grad
  }



  pub fn grad_fn(&self) -> Option<Rc<dyn GradientFunction>> {
    self.grad_fn.clone()
  }

  pub fn set_grad_fn(&mut self, grad_fn: Option<Rc<dyn GradientFunction>>)  {
    self.grad_fn = grad_fn;
  }

  pub fn grad(&self) -> Option<Rc<RefCell<TensorStorage>>> {
    self.grad.clone()
  }

  pub fn set_grad(&mut self, grad: Option<Rc<RefCell<TensorStorage>>>) {
    self.grad = grad;
  }
  
}


