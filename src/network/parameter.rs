use std::{cell::RefCell, rc::Rc};
use crate::{autograd::tensor::*, TensorCreation};

pub struct Parameter {
  pub tensor: Rc<RefCell<Tensor>>,
}

impl Parameter {
  pub fn new(shape: Vec<usize>) -> Self {
    Parameter {
      tensor: Rc::new(RefCell::new(Tensor::zeros(shape, Some(true)))),
    }
  }
}