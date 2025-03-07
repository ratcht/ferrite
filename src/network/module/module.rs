use crate::tensor::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};



pub trait Module {
  fn forward(&mut self, input: &Tensor) -> Tensor;
  
  // Optional methods with defaults
  fn parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
    HashMap::new()
  }
  
  fn train(&mut self) { }
  fn eval(&mut self) { }
  fn zero_grad(&mut self) { }

  /// Visit all parameters with a callback function
  fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Tensor)) {
    // Default implementation uses parameters()
    for (name, param) in self.parameters() {
      if let Ok(tensor) = param.read() {
        f(&name, &tensor);
      }
    }
  }

  /// Print all parameters and their shapes
  fn print_parameters(&self, values: bool) {
    self.visit_parameters(&mut |name, param| {
      if values {
        println!("Parameter {}: shape={:?}. values={:?}", name, param.shape(), param);
      } else {
        println!("Parameter {}: shape={:?}", name, param.shape());
      }
    });
  }
}