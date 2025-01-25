use crate::tensor::*;
use crate::macros::*;

macro_rules! reduce_grad {
  ($grad:expr, $shape:expr) => {{
    let mut reduced_grad = $grad.clone();
    for (dim, (grad_size, shape_size)) in $grad.shape().iter().zip($shape.iter()).enumerate() {
      if shape_size == &1 && grad_size != &1 {
        let mut sum_dims = vec![false; $grad.shape().len()];
        sum_dims[dim] = true;
        reduced_grad = reduced_grad.sum_dim(&sum_dims);
      }
    }
    reduced_grad
  }};
}

pub trait GradientFunction: std::fmt::Debug {
  fn backward(&self);
  fn prev(&self) -> Vec<&Tensor>;
}

#[derive(Debug)]
pub struct AddGrad {
  lhs: Tensor,
  rhs: Tensor,
  output: Tensor,
}

impl AddGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Self {
    AddGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for AddGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let reduced_grad = reduce_grad!(out_grad, self.lhs.tensor().shape());
    
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
    
    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      let reduced_grad = reduce_grad!(out_grad, self.rhs.tensor().shape());
      
      rhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}


#[derive(Debug)]
pub struct SubGrad {
  lhs: Tensor,
  rhs: Tensor,
  output: Tensor,
}

impl SubGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Self {
    SubGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for SubGrad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let reduced_grad = reduce_grad!(out_grad, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
    
    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      let grad_for_rhs = &*out_grad * -1.;
      let reduced_grad = reduce_grad!(grad_for_rhs, self.rhs.tensor().shape());
      
      rhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}

#[derive(Debug)]
pub struct MulGrad {
  lhs: Tensor,
  rhs: Tensor,
  output: Tensor,
}

impl MulGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Self {
    MulGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for MulGrad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad * self.rhs.tensor();
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
    
    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      let grad_for_rhs = &*out_grad * self.lhs.tensor();
      
      let reduced_grad = reduce_grad!(grad_for_rhs, self.rhs.tensor().shape());

      rhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}


#[derive(Debug)]
pub struct DivGrad {
  lhs: Tensor,
  rhs: Tensor,
  output: Tensor,
}

impl DivGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Self {
    DivGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for DivGrad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad / self.rhs.tensor();
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
    
    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      // Form grad for rhs
      let grad_for_rhs = &(&*out_grad * self.lhs.tensor()).mul_f32(-1.) / &(self.rhs.tensor().pow_f32(2.));
      
      let reduced_grad = reduce_grad!(grad_for_rhs, self.rhs.tensor().shape());

      rhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}

#[derive(Debug)]
pub struct PowF32Grad {
  lhs: Tensor,
  rhs: f32,
  output: Tensor,
}

impl PowF32Grad {
  pub fn new(lhs: &Tensor, rhs: f32, output: &Tensor) -> Self {
    PowF32Grad {
      lhs: lhs.clone(),
      rhs: rhs,
      output: output.clone(),
    }
  }
}

impl GradientFunction for PowF32Grad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &(&*out_grad * self.rhs) * &self.lhs.tensor().pow_f32(self.rhs-1.);
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);

    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}


#[derive(Debug)]
pub struct AddF32Grad {
  lhs: Tensor,
  rhs: f32,
  output: Tensor,
}

impl AddF32Grad {
  pub fn new(lhs: &Tensor, rhs: f32, output: &Tensor) -> Self {
    AddF32Grad {
      lhs: lhs.clone(),
      rhs: rhs,
      output: output.clone(),
    }
  }
}

impl GradientFunction for AddF32Grad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let reduced_grad = reduce_grad!(out_grad, self.lhs.tensor().shape());
    
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}


#[derive(Debug)]
pub struct SubF32Grad {
  lhs: Tensor,
  rhs: f32,
  output: Tensor,
}

impl SubF32Grad {
  pub fn new(lhs: &Tensor, rhs: f32, output: &Tensor) -> Self {
    SubF32Grad {
      lhs: lhs.clone(),
      rhs: rhs,
      output: output.clone(),
    }
  }
}

impl GradientFunction for SubF32Grad {
  fn backward(&self) {
    // Get output gradient
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let reduced_grad = reduce_grad!(out_grad, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}

#[derive(Debug)]
pub struct MulF32Grad {
  lhs: Tensor,
  rhs: f32,
  output: Tensor,
}

impl MulF32Grad {
  pub fn new(lhs: &Tensor, rhs: f32, output: &Tensor) -> Self {
    MulF32Grad {
      lhs: lhs.clone(),
      rhs: rhs,
      output: output.clone(),
    }
  }
}

impl GradientFunction for MulF32Grad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad * self.rhs;
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}


#[derive(Debug)]
pub struct DivF32Grad {
  lhs: Tensor,
  rhs: f32,
  output: Tensor,
}

impl DivF32Grad {
  pub fn new(lhs: &Tensor, rhs: f32, output: &Tensor) -> Self {
    DivF32Grad {
      lhs: lhs.clone(),
      rhs: rhs,
      output: output.clone(),
    }
  }
}

impl GradientFunction for DivF32Grad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad / self.rhs;
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}


#[derive(Debug)]
pub struct AbsGrad {
  lhs: Tensor,
  output: Tensor,
}

impl AbsGrad {
  pub fn new(lhs: &Tensor, output: &Tensor) -> Self {
    AbsGrad {
      lhs: lhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for AbsGrad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Propagate to lhs
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &*out_grad * &self.lhs.tensor().sign();
      
      let reduced_grad = reduce_grad!(grad_for_lhs, self.lhs.tensor().shape());
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs]
  }
}


#[derive(Debug)]
pub struct SumGrad {
  input: Tensor,
  output: Tensor,
}

impl SumGrad {
  pub fn new(input: &Tensor, output: &Tensor) -> Self {
    SumGrad {
      input: input.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for SumGrad {
  fn backward(&self) {
    let device = self.output.device();
    if let Some(input_grad) = &self.input.grad() {
      if let Some(out_grad) = &self.output.grad() {
        // For sum, we need to expand the gradient to match input shape
        let input_shape = self.input.tensor().shape();
        let ones = Storage::ones(input_shape.clone(), Some(device), None);
        let expanded_grad = &ones * out_grad.borrow().get(&[0]);
        input_grad.borrow_mut().add_tensor_assign(&expanded_grad);
      }
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.input]
  }
}

#[derive(Debug)]
pub struct MeanGrad {
  input: Tensor,
  output: Tensor,
}

impl MeanGrad {
  pub fn new(input: &Tensor, output: &Tensor) -> Self {
    MeanGrad {
      input: input.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for MeanGrad {
  fn backward(&self) {
    let device = self.output.device();

    if let Some(input_grad) = &self.input.grad() {
      if let Some(out_grad) = &self.output.grad() {
        // For mean, expand gradient and divide by number of elements
        let input_shape = self.input.tensor().shape();
        let n_elements = input_shape.iter().product::<usize>() as f32;
        let ones = Storage::ones(input_shape.clone(), Some(device), None);
        let expanded_grad = &ones * (out_grad.borrow().get(&[0]) / n_elements);
        input_grad.borrow_mut().add_tensor_assign(&expanded_grad);
      }
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.input]
  }
}

#[derive(Debug)]
pub struct ProductGrad {
  input: Tensor,
  output: Tensor,
}

impl ProductGrad {
  pub fn new(input: &Tensor, output: &Tensor) -> Self {
    ProductGrad {
      input: input.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for ProductGrad {
  fn backward(&self) {
    let device = self.output.device();

    if let Some(input_grad) = &self.input.grad() {
      if let Some(out_grad) = &self.output.grad() {
        // For product, each element's gradient is the product of all other elements
        let input_data = self.input.tensor();
        let mut grad = Storage::zeros(input_data.shape().clone(), Some(device), None);
        let total_product = self.output.tensor().get(&[0]);
        
        // For each element, divide total product by that element to get product of others
        for i in 0..input_data.data().borrow().len() {
          let element = input_data.data().borrow()[i];
          if element != 0.0 {
            grad.data_mut()[i] = total_product / element;
          }
        }
        
        // Multiply by output gradient
        grad = &grad * out_grad.borrow().get(&[0]);
        input_grad.borrow_mut().add_tensor_assign(&grad);
      }
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.input]
  }
}


#[derive(Debug)]
pub struct PermuteGrad {
  input: Tensor,
  output: Tensor,
}


impl PermuteGrad {
  pub fn new(input: &Tensor, output: &Tensor) -> Self {
    PermuteGrad {
      input: input.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for PermuteGrad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Get input gradient if it exists (it should since we're backpropagating)
    if let Some(input_grad) = &self.input.grad() {
      // Determine the permutation that was applied
      // Compare input and output shapes/strides
      let input_shape = self.input.tensor().shape();
      let output_shape = self.output.tensor().shape();
      
      // Find the permutation by matching dimensions
      let mut permutation: Vec<usize> = Vec::new();
      for i in 0..input_shape.len() {
        for j in 0..output_shape.len() {
          if input_shape[i] == output_shape[j] {
            permutation.push(j);
            break;
          }
        }
      }
      
      // Create inverse permutation array
      let mut inverse_perm = vec![0; permutation.len()];
      for (i, &p) in permutation.iter().enumerate() {
        inverse_perm[p] = i;
      }
      
      // Apply inverse permutation to gradient
      let mut grad_tensor = out_grad.clone();
      grad_tensor.permute(&inverse_perm);
      
      // Accumulate the gradient
      input_grad.borrow_mut().add_tensor_assign(&grad_tensor);
    }
  
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.input]
  }
}

#[derive(Debug)]
pub struct MatMulGrad {
  lhs: Tensor,
  rhs: Tensor,
  output: Tensor,
  trans_a: bool,
  trans_b: bool
}

impl MatMulGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor, trans_a: bool, trans_b: bool,) -> Self {
    MatMulGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
      trans_a: trans_a,
      trans_b: trans_b,
    }
  }
}

impl GradientFunction for MatMulGrad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();

    // Case: C = t × w^T  (your test case)
    // t: (2,3), w: (2,3), C: (2,2)
    // dL/dt = dL/dC × w^T  - note: no transpose here since we already have w^T
    // dL/dw = (dL/dC × t)^T

    if let Some(lhs_grad) = &self.lhs.grad() {
      // For input t: dL/dt = dL/dC × w^T
      let grad_for_lhs = if !self.trans_b {
        out_grad.matmul(self.rhs.tensor(), false, true)
      } else {
        out_grad.matmul(self.rhs.tensor(), false, false)
      };
      lhs_grad.borrow_mut().add_tensor_assign(&grad_for_lhs);
    }

    if let Some(rhs_grad) = &self.rhs.grad() {
      // For weight w: dL/dw = (dL/dC × t)^T
      let grad_for_rhs = if !self.trans_b {
        self.lhs.tensor().matmul(&out_grad, true, false)
      } else {
        out_grad.matmul(&self.lhs.tensor(), true, false)
      };
      rhs_grad.borrow_mut().add_tensor_assign(&grad_for_rhs);
    }
  } 

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}