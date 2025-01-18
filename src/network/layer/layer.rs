use super::super::*;
use crate::autograd::tensor::*;
use crate::{TensorOps, BLASTensorOps, TensorShape, TensorCreation};
use std::borrow::Borrow;
use std::rc::Rc;
use std::cell::{Ref, RefCell};

pub struct Linear {
  pub module: Module,
  in_features: usize,
  out_features: usize,
}


impl Linear {
  pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
    let mut module = Module::new();

    // Add weight to module
    let bound = f32::sqrt(1./in_features as f32);
    module.add_parameter("weight", Tensor::uniform(-bound, bound, vec![out_features, in_features], Some(true)));

    if bias {
      module.add_parameter("bias", Tensor::zeros(vec![out_features], Some(true)));
    }

    Linear {
      module: module,
      in_features: in_features,
      out_features: out_features
    }
  }
}



impl Segment for Linear {
  fn forward(&mut self, input: &Tensor) -> Tensor {
    // Get weight parameter and access its tensor
    let weight: Rc<RefCell<Tensor>> = self.module.get_parameter("weight");
    let weight_tensor = weight.borrow_mut();

    // Perform matrix multiplication
    println!("Matmul: Self:{}\n Other:{}", input, weight_tensor);
    let output = input.matmul(&weight_tensor, false, true);

    // Add bias if present
    if self.module.has_parameter("bias") {
      let bias: Rc<RefCell<Tensor>> = self.module.get_parameter("bias");
      let bias_tensor = bias.borrow_mut();
      
      // Add bias to each output feature
      &output + &bias_tensor
    } else {
      output
    }
  }
}