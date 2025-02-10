use crate::{reduce_grad, tensor::*};
use super::super::grad::*;


#[derive(Debug)]
pub struct BinaryStepGrad {
  lhs: Tensor,
  output: Tensor,
}

impl BinaryStepGrad {
  pub fn new(lhs: &Tensor, output: &Tensor) -> Self {
    BinaryStepGrad {
      lhs: lhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for BinaryStepGrad {
  fn backward(&self) {
    if let Some(lhs_grad) = &self.lhs.grad() {
      let zeros = Storage::zeros(self.lhs.tensor().shape().to_vec(), Some(self.lhs.device()), None);    
      lhs_grad.borrow_mut().add_tensor_assign(&zeros);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}



#[derive(Debug)]
pub struct SigmoidGrad {
  lhs: Tensor,
  output: Tensor,
}

impl SigmoidGrad {
  pub fn new(lhs: &Tensor, output: &Tensor) -> Self {
    SigmoidGrad {
      lhs: lhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for SigmoidGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let sigmoid_op = |x: f32| 1./(1. + f32::exp(-x));
      let grad_for_lhs = &*out_grad * &self.lhs.storage.apply(|x| sigmoid_op(x) * (1. - sigmoid_op(x)));

      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
    
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}



#[derive(Debug)]
pub struct TanhGrad {
  lhs: Tensor,
  output: Tensor,
}

impl TanhGrad {
  pub fn new(lhs: &Tensor, output: &Tensor) -> Self {
    TanhGrad {
      lhs: lhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for TanhGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let tanh_op = |x: f32| (f32::exp(x) - f32::exp(-x))/(f32::exp(x) + f32::exp(-x));
      let grad_for_lhs = &*out_grad * &self.lhs.storage.apply(|x| 1. - tanh_op(x));

      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
    
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}


#[derive(Debug)]
pub struct ReluGrad {
  lhs: Tensor,
  output: Tensor,
}

impl ReluGrad {
  pub fn new(lhs: &Tensor, output: &Tensor) -> Self {
    ReluGrad {
      lhs: lhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for ReluGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad * &self.lhs.storage.apply(|x| if x <= 0. {0.} else {1.});
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
    
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}



#[derive(Debug)]
pub struct LeakyReluGrad {
  lhs: Tensor,
  output: Tensor,
}

impl LeakyReluGrad {
  pub fn new(lhs: &Tensor, output: &Tensor) -> Self {
    LeakyReluGrad {
      lhs: lhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for LeakyReluGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad * &self.lhs.storage.apply(|x| if x <= 0. {0.1} else {1.});
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
    
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}


#[derive(Debug)]
pub struct ParametricReluGrad {
  lhs: Tensor,
  a: f32,
  output: Tensor,
}

impl ParametricReluGrad {
  pub fn new(lhs: &Tensor, a: f32, output: &Tensor) -> Self {
    ParametricReluGrad {
      lhs: lhs.clone(),
      a: a,
      output: output.clone(),
    }
  }
}

impl GradientFunction for ParametricReluGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad * &self.lhs.storage.apply(|x| if x <= 0. {self.a} else {1.});
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
    
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}




#[derive(Debug)]
pub struct EluGrad {
  lhs: Tensor,
  alpha: f32,
  output: Tensor,
}

impl EluGrad {
  pub fn new(lhs: &Tensor, alpha: f32, output: &Tensor) -> Self {
    EluGrad {
      lhs: lhs.clone(),
      alpha: alpha,
      output: output.clone(),
    }
  }
}

impl GradientFunction for EluGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad * &self.lhs.storage.apply(|x| if x <= 0. {self.alpha * f32::exp(x)} else {1.});
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
    
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}




#[derive(Debug)]
pub struct SoftmaxGrad {
  lhs: Tensor,
  output: Tensor,
}

impl SoftmaxGrad {
  pub fn new(lhs: &Tensor, output: &Tensor) -> Self {
    SoftmaxGrad {
      lhs: lhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for SoftmaxGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      // Get the softmax output: s = softmax(x)
      let s = self.output.tensor();

      // Compute the elementwise product: (dL/ds * s)
      let grad_times_s = &*out_grad * s;

      // Determine the axis over which softmax was computed.
      // For example, if softmax is computed over the last dimension:
      let axis = s.shape().len() - 1;
      
      // Sum the product along the softmax axis. This gives, for each sample,
      // the inner product \(\sum_j s_j * (dL/ds)_j\).
      let sum_along_axis = grad_times_s.sum_axis(axis);
      
      // Efficient gradient for softmax:
      // dL/dx = s * (dL/ds - sum(s * dL/ds))
      let grad_for_lhs = s * &(&*out_grad - &sum_along_axis);

      // If necessary, reduce the gradient to match the shape of the lhs Tensor.
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      // Add the computed gradient to the lhs gradient.
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}



#[derive(Debug)]
pub struct SwishGrad {
  lhs: Tensor,
  output: Tensor,
}

impl SwishGrad {
  pub fn new(lhs: &Tensor, output: &Tensor) -> Self {
    SwishGrad {
      lhs: lhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for SwishGrad {
  fn backward(&self) {
    // Get output gradient
    unimplemented!()
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}