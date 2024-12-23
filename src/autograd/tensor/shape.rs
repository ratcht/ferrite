use std::{borrow::Borrow, rc::Rc};

use crate::{tensor_storage::*, TensorShape};
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
    Tensor::new(tensor, *self.requires_grad(), self.grad_fn(), self.grad())
  }

  /// Compute broadcast shape between two shapes
  fn compute_broadcast_shape(&self, target_shape: &[usize]) -> Vec<usize> {
    self.tensor().compute_broadcast_shape(target_shape)
  }

  /// Compute broadcast strides
  fn compute_broadcast_strides(&self, broadcast_shape: &[usize]) -> Vec<usize> {
    self.tensor().compute_broadcast_strides(broadcast_shape)
  }

  /// Pad shape with ones on the left
  fn pad_shape(&self, target_rank: usize) -> Vec<usize> {
    self.tensor().pad_shape(target_rank)
  }

  fn broadcast_tensors(a: &Self, b: &Self) -> (Self, Self) {    
    let (ts_a, ts_b) = TensorStorage::broadcast_tensors(a.tensor(), b.tensor());

    let broadcast_a = Tensor::new(ts_a, *a.requires_grad(), a.grad_fn(), a.grad());
    let broadcast_b = Tensor::new(ts_b, *b.requires_grad(), b.grad_fn(), b.grad());

    (broadcast_a, broadcast_b)
  }
 
}