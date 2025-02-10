use crate::*;
use std::rc::Rc;

pub trait ActivationOps {
  fn binary_step(&self) -> Self;
  fn sigmoid(&self) -> Self;
  fn tanh(&self) -> Self;
  fn relu(&self) -> Self;
  fn leaky_relu(&self) -> Self;
  fn parametric_relu(&self, a: f32) -> Self;
  fn elu(&self, alpha: f32) -> Self;
  fn softmax(&self, dim: usize) -> Self;
  fn swish(&self) -> Self;
}

impl ActivationOps for Storage {
  fn binary_step(&self) -> Self {
    match_storage!(unary self, binary_step)
  }

  fn sigmoid(&self) -> Self {
    match_storage!(unary self, sigmoid)
  }

  fn tanh(&self) -> Self {
    match_storage!(unary self, tanh)
  }

  fn relu(&self) -> Self {
    match_storage!(unary self, relu)
  }

  fn leaky_relu(&self) -> Self {
    match_storage!(unary self, leaky_relu)
  }

  fn parametric_relu(&self, a: f32) -> Self {
    match_storage!(unary self, parametric_relu, a)
  }

  fn elu(&self, alpha: f32) -> Self {
    match_storage!(unary self, elu, alpha)
  }

  fn softmax(&self, dim: usize) -> Self {
    match_storage!(unary self, softmax, dim)
  }

  fn swish(&self) -> Self {
    match_storage!(unary self, swish)
  } 
}


impl ActivationOps for Tensor {
  fn binary_step(&self) -> Self {
    let tensor = self.tensor().binary_step();
    
    // Create result tensor
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    // Set up gradient function if needed
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(BinaryStepGrad::new(
        self, 
        &result
      ))));
    }
    
    result
  }
  
  fn sigmoid(&self) -> Self {
    let tensor = self.tensor().sigmoid();
    
    // Create result tensor
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    // Set up gradient function if needed
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(SigmoidGrad::new(
        self, 
        &result
      ))));
    }
    
    result
  }
  
  fn tanh(&self) -> Self {
    let tensor = self.tensor().tanh();
    
    // Create result tensor
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    // Set up gradient function if needed
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(TanhGrad::new(
        self, 
        &result
      ))));
    }
    
    result
  }
  
  fn relu(&self) -> Self {
    let tensor = self.tensor().relu();
    
    // Create result tensor
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    // Set up gradient function if needed
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(ReluGrad::new(
        self, 
        &result
      ))));
    }
    
    result
  }
  
  fn leaky_relu(&self) -> Self {
    let tensor = self.tensor().leaky_relu();
    
    // Create result tensor
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    // Set up gradient function if needed
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(LeakyReluGrad::new(
        self, 
        &result
      ))));
    }
    
    result
  }
  
  fn parametric_relu(&self, a: f32) -> Self {
    let tensor = self.tensor().parametric_relu(a);
    
    // Create result tensor
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    // Set up gradient function if needed
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(ParametricReluGrad::new(
        self, 
        a,
        &result
      ))));
    }
    
    result
  }
  
  fn elu(&self, alpha: f32) -> Self {
    let tensor = self.tensor().elu(alpha);
    
    // Create result tensor
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    // Set up gradient function if needed
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(EluGrad::new(
        self,
        alpha,
        &result
      ))));
    }
    
    result
  }
  
  fn softmax(&self, dim: usize) -> Self {
    let tensor = self.tensor().softmax(dim);
    
    // Create result tensor
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    // Set up gradient function if needed
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(SoftmaxGrad::new(
        self, 
        &result
      ))));
    }
    
    result
  }
  
  fn swish(&self) -> Self {
    let tensor = self.tensor().swish();
    
    // Create result tensor
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    // Set up gradient function if needed
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(SwishGrad::new(
        self, 
        &result
      ))));
    }
    
    result
  }
}