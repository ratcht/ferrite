use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::super::*;
use crate::autograd::tensor::*;
use crate::{TensorOps, BLASTensorOps, TensorShape, TensorCreation};

// Linear layer implementation
pub struct Linear {
  weight: Arc<RwLock<Tensor>>,
  bias: Option<Arc<RwLock<Tensor>>>,
  training: bool,
}


impl Linear {
  pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
    let bound = f32::sqrt(1./in_features as f32);
    let weight = Arc::new(RwLock::new(
      Tensor::uniform(-bound, bound, vec![out_features, in_features], Some(true))
    ));

    let bias = if bias {
      Some(Arc::new(RwLock::new(Tensor::zeros(vec![out_features], Some(true)))))
    } else {
      None
    };

    Linear{weight, bias, training: false,}
  }

  fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Tensor)) {
    // More efficient direct implementation than using parameters()
    if let Ok(weight) = self.weight.read() {
      f("weight", &weight);
    }
    if let Some(bias) = &self.bias {
      if let Ok(bias) = bias.read() {
        f("bias", &bias);
      }
    }
  }
}



impl Module for Linear {
  fn forward(&mut self, input: &Tensor) -> Tensor {
    // Get weight parameter and access its tensor
    let weight = self.weight.read()
                            .unwrap();

    // Perform matrix multiplication
    println!("Matmul: Self:{}\n Other:{}", input, weight);
    let mut output = input.matmul(&weight, false, true);

    // Add bias if present
    if let Some(bias) = &self.bias {
      let bias = bias.read()
                     .unwrap();
      output = &output + &bias;
    } 
    output
  }

  fn parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
    let mut params = HashMap::new();
    params.insert("weight".to_string(), self.weight.clone());
    if let Some(bias) = &self.bias {
      params.insert("bias".to_string(), bias.clone());
    }
    params
  }

  fn train(&mut self) {
    self.training = true;
  }

  fn eval(&mut self) {
    self.training = false;
  }

  fn zero_grad(&mut self) {
    todo!("Implement zero_grad for Linear");
  }

}


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