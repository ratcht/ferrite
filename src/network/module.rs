use crate::autograd::tensor::*;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

use super::parameter::*;

pub trait Segment {
  fn forward(input: Tensor) -> Tensor;
}

pub struct Module {
  parameters: HashMap<String, Rc<RefCell<Parameter>>>,
  modules: HashMap<String, Rc<RefCell<Module>>>,
  training: bool
}

impl Module {
  pub fn new() -> Self {
    Module {
      parameters: HashMap::new(),
      modules: HashMap::new(),
      training: false,
    }
  }

  pub fn add_parameter(&mut self, name: &str, parameter: Parameter) {
    self.parameters.insert(name.to_string(), Rc::new(RefCell::new(parameter)));
  }

  pub fn add_module(&mut self, name: &str, module: Module) {
    self.modules.insert(name.to_string(), Rc::new(RefCell::new(module)));
  }

  pub fn visit_parameters<F>(&self, mut f: F)
  where
    F: FnMut(&Parameter)
  {
    // Visit parameters in current module
    for (name, param) in &self.parameters {
      f(&*param.borrow());
    }
    
    // Recursively visit child modules
    for (name, module) in &self.modules {
      module.borrow().visit_parameters(&mut f);
    }
  }

  pub fn train(&self) {

  }

  pub fn eval(&self) {

  }

  pub fn zero_grad(&self) {

  }
}