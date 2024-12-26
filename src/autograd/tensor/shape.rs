use crate::tensor_storage::*;
use crate::TensorShape;
use super::base::*;

impl TensorShape for Tensor {
  fn reshape(&mut self, new_shape: Vec<usize>) {
    self.tensor_mut().set_shape(new_shape);
  }

  fn transpose(&mut self) {
    // Transpose by swapping dimensions & strides
    self.tensor_mut().transpose();
  }

  fn permute(&mut self, dims: &[usize]) {
    self.tensor_mut().permute(dims);
  }

  fn flatten(&mut self) {
    self.tensor_mut().flatten();
  }

  fn squeeze(&mut self) {
    self.tensor_mut().squeeze();
  } 

  fn unsqueeze(&mut self, dim: usize) {
    self.tensor_mut().unsqueeze(dim);
  }

  fn broadcast(&self, new_shape: &[usize]) -> Self {
    let tensor = self.tensor().broadcast(new_shape);
    
    // When broadcasting, we need to maintain the gradient tracking
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, requires_grad);
    
    // If original tensor requires gradient, the broadcasted tensor
    // should have the same gradient function
    if requires_grad {
      result.set_grad_fn(self.grad_fn());
    }
    
    result
  }

  fn compute_broadcast_shape(&self, target_shape: &[usize]) -> Vec<usize> {
    self.tensor().compute_broadcast_shape(target_shape)
  }

  fn compute_broadcast_strides(&self, broadcast_shape: &[usize]) -> Vec<usize> {
    self.tensor().compute_broadcast_strides(broadcast_shape)
  }

  fn pad_shape(&self, target_rank: usize) -> Vec<usize> {
    self.tensor().pad_shape(target_rank)
  }

  fn broadcast_tensors(a: &Self, b: &Self) -> (Self, Self) {    
    let (ts_a, ts_b) = TensorStorage::broadcast_tensors(a.tensor(), b.tensor());

    // Create new tensors with proper gradient tracking
    let mut broadcast_a = Tensor::new(ts_a, *a.requires_grad());
    let mut broadcast_b = Tensor::new(ts_b, *b.requires_grad());

    // Maintain gradient functions if present
    if *a.requires_grad() {
      broadcast_a.set_grad_fn(a.grad_fn());
    }
    if *b.requires_grad() {
      broadcast_b.set_grad_fn(b.grad_fn());
    }

    (broadcast_a, broadcast_b)
  }
}