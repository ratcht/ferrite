use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::module::*;
use crate::tensor::*;

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
      //Tensor::uniform(-bound, bound, vec![out_features, in_features], Some(true))
      Tensor::ones(vec![out_features, in_features], Some(true))
    ));

    let bias = if bias {
      Some(Arc::new(RwLock::new(Tensor::zeros(vec![out_features], Some(false)))))
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
    let mut output = input.matmul(&weight, false, true);

    // Add bias if present
    if let Some(bias) = &self.bias {
      let bias = bias.read()
                     .unwrap();
      output = &output + &*bias;
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