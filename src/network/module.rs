use crate::autograd::tensor::*;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;


pub trait Segment {
  fn forward(&mut self, input: &Tensor) -> Tensor;
}

pub struct Module {
  parameters: HashMap<String, Rc<RefCell<Tensor>>>,
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

  pub fn add_parameter(&mut self, name: &str, parameter: Tensor) {
    self.parameters.insert(name.to_string(), Rc::new(RefCell::new(parameter)));
  }

  pub fn has_parameter(&self, name: &str) -> bool {
    self.parameters.contains_key(name)
  }

  pub fn get_parameter(&mut self, name: &str) -> Rc<RefCell<Tensor>>{
    self.parameters[name].clone()
  }

  pub fn add_module(&mut self, name: &str, module: Module) {
    self.modules.insert(name.to_string(), Rc::new(RefCell::new(module)));
  }

  pub fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Tensor)) {
    // Visit parameters in current module
    for (name, param) in &self.parameters {
      f(name, &*param.borrow());
    }
    
    // Recursively visit child modules
    for (name, module) in &self.modules {
      module.borrow().visit_parameters(f);
    }
  }

  pub fn train(&self) {

  }

  pub fn eval(&self) {

  }

  pub fn zero_grad(&self) {

  }
}