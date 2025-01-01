use std::rc::Rc;
use std::cell::RefCell;
use crate::tensor_storage::*;
use super::base::*;

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
      let mut reduced_grad = out_grad.clone();
      let lhs_shape = self.lhs.tensor().shape();
      
      // Reduce along broadcasted dimensions
      for (dim, (grad_size, lhs_size)) in out_grad.shape().iter()
        .zip(lhs_shape.iter())
        .enumerate() 
      {
        if lhs_size == &1 && grad_size != &1 {
          let mut sum_dims = vec![false; out_grad.shape().len()];
          sum_dims[dim] = true;
          reduced_grad = reduced_grad.sum_dim(&sum_dims);
        }
      }
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad);
    }
    
    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      let mut reduced_grad = out_grad.clone();
      let rhs_shape = self.rhs.tensor().shape();
      
      for (dim, (grad_size, rhs_size)) in out_grad.shape().iter()
        .zip(rhs_shape.iter())
        .enumerate() 
      {
        if rhs_size == &1 && grad_size != &1 {
          let mut sum_dims = vec![false; out_grad.shape().len()];
          sum_dims[dim] = true;
          reduced_grad = reduced_grad.sum_dim(&sum_dims);
        }
      }
      
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
      
      // Handle broadcasting
      let lhs_shape = self.lhs.tensor().shape();
      let mut reduced_grad_for_lhs = grad_for_lhs.clone();
      for (dim, (grad_size, lhs_size)) in grad_for_lhs.shape().iter()
        .zip(lhs_shape.iter())
        .enumerate() 
      {
        if lhs_size == &1 && grad_size != &1 {
          let mut sum_dims = vec![false; grad_for_lhs.shape().len()];
          sum_dims[dim] = true;
          reduced_grad_for_lhs = reduced_grad_for_lhs.sum_dim(&sum_dims);
        }
      }
      
      lhs_grad.borrow_mut().add_tensor_assign(&reduced_grad_for_lhs);

    }
    
    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      let grad_for_rhs = &*out_grad * self.lhs.tensor();
      
      let rhs_shape = self.rhs.tensor().shape();

      let mut reduced_grad_for_rhs = grad_for_rhs.clone();
      for (dim, (grad_size, rhs_size)) in grad_for_rhs.shape().iter()
        .zip(rhs_shape.iter())
        .enumerate() 
      {

        if rhs_size == &1 && grad_size != &1 {
          let mut sum_dims = vec![false; grad_for_rhs.shape().len()];
          sum_dims[dim] = true;
          reduced_grad_for_rhs = reduced_grad_for_rhs.sum_dim(&sum_dims);
        }

      }


      
      rhs_grad.borrow_mut().add_tensor_assign(&reduced_grad_for_rhs);
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
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
    if let Some(input_grad) = &self.input.grad() {
      if let Some(out_grad) = &self.output.grad() {
        // For sum, we need to expand the gradient to match input shape
        let input_shape = self.input.tensor().shape();
        let ones = TensorStorage::ones(input_shape.clone(), None);
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
    if let Some(input_grad) = &self.input.grad() {
      if let Some(out_grad) = &self.output.grad() {
        // For mean, expand gradient and divide by number of elements
        let input_shape = self.input.tensor().shape();
        let n_elements = input_shape.iter().product::<usize>() as f32;
        let ones = TensorStorage::ones(input_shape.clone(), None);
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
    if let Some(input_grad) = &self.input.grad() {
      if let Some(out_grad) = &self.output.grad() {
        // For product, each element's gradient is the product of all other elements
        let input_data = self.input.tensor();
        let mut grad = TensorStorage::zeros(input_data.shape().clone(), None);
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
}

impl MatMulGrad {
  pub fn new(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Self {
    MatMulGrad {
      lhs: lhs.clone(),
      rhs: rhs.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for MatMulGrad {
  fn backward(&self) {
    let out_grad = self.output.grad().unwrap();
    let out_grad = out_grad.borrow();
    // C = A @ B
    // dC/dA = gradC @ B.T
    // dC/dB = A.T @ gradC

    // A: dC/dA = gradC @ B.T
    println!("Shapes: ");
    println!("A: {:?}", self.lhs.tensor().shape());
    println!("B: {:?}", self.rhs.tensor().shape());
    println!("C: {:?}", out_grad.shape());
    if let Some(lhs_grad) = &self.lhs.grad() {
      let grad_for_lhs = &out_grad.matmul(self.rhs.tensor(), false, true);

      // Handle broadcasting
      lhs_grad.borrow_mut().add_tensor_assign(&grad_for_lhs);
      println!("done lhs:{:?}", lhs_grad);
    }

    // Propagate to rhs
    if let Some(rhs_grad) = &self.rhs.grad() {
      println!("outgrad shape: {:?}", out_grad.shape());
      println!("lhs shape: {:?}", self.lhs.tensor().shape());
      let grad_for_rhs = &self.lhs.tensor().matmul(&out_grad, true, false);
      
      // Handle broadcasting
      rhs_grad.borrow_mut().add_tensor_assign(&grad_for_rhs);
      println!("done rhs:{:?}", rhs_grad);

    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.lhs, &self.rhs]
  }
}