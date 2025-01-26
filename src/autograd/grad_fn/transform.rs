use crate::{reduce_grad, tensor::*};
use super::super::grad::*;

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