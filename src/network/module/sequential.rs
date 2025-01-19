use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::super::*;
use crate::autograd::tensor::*;


pub struct Sequential {
  layers: Vec<Box<dyn Module>>,
  training: bool,
}

impl Sequential {
  pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
    Self {
      layers,
      training: false,
    }
  }

  pub fn add(&mut self, layer: Box<dyn Module>) {
    self.layers.push(layer);
  }

  fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Tensor)) {
    for (idx, layer) in self.layers.iter().enumerate() {
      // Create a new closure that prefixes the parameter names
      let mut prefixed_f = |name: &str, tensor: &Tensor| {
        let full_name = format!("layer_{}.{}", idx, name);
        f(&full_name, tensor);
      };
      layer.visit_parameters(&mut prefixed_f);
    }
  }
}

impl Module for Sequential {
  fn forward(&mut self, input: &Tensor) -> Tensor {
    let mut current = input.clone();
    for layer in self.layers.iter_mut() {
      current = layer.forward(&current);
    }
    current
  }

  fn parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
    let mut params = HashMap::new();
    for (idx, layer) in self.layers.iter().enumerate() {
      for (name, param) in layer.parameters() {
        let full_name = format!("layer_{}.{}", idx, name);
        params.insert(full_name, param);
      }
    }
    params
  }

  fn train(&mut self) {
    self.training = true;
    for layer in &mut self.layers {
      layer.train();
    }
  }

  fn eval(&mut self) {
    self.training = false;
    for layer in &mut self.layers {
      layer.eval();
    }
  }

  fn zero_grad(&mut self) {
    for layer in &mut self.layers {
      layer.zero_grad();
    }
  }
}