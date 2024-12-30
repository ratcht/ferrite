use std::{borrow::Borrow, rc::Rc};

use super::base::TensorStorage;  // Import from parent module's base.rs
pub trait TensorShape {
  fn reshape(&mut self, new_shape: Vec<usize>);
  fn permute(&mut self, dims: &[usize]);
  fn transpose(&self) -> Self;
  fn flatten(&mut self);
  fn squeeze(&mut self);
  fn unsqueeze(&mut self, dim: usize);

  fn broadcast(&self, new_shape: &[usize]) -> Self;
  fn compute_broadcast_shape(&self, target_shape: &[usize]) -> Vec<usize>;
  fn compute_broadcast_strides(&self, broadcast_shape: &[usize]) -> Vec<usize>;
  fn pad_shape(&self, target_rank: usize) -> Vec<usize>;
  fn broadcast_tensors(a: &Self, b: &Self) -> (Self, Self) where Self: Sized;
}


impl TensorShape for TensorStorage {
  fn reshape(&mut self, new_shape: Vec<usize>) {
    self.set_shape(new_shape);
  }

  fn transpose(&self) -> Self {
    // Transpose by swapping dimensions & strides
    if self.shape().len() != 2 { panic!("Must be 2-D Tensor (Matrix)"); }

    let mut shape = self.shape().to_owned();
    shape.reverse();

    let mut stride = self.stride().to_owned();
    stride.reverse();

    TensorStorage::create(self.data(), shape, stride)
  }

  fn permute(&mut self, dims: &[usize]) {
    let self_shape = self.shape();
    let shape = dims.iter().map(|&i| self_shape[i]).collect();    

    let self_stride = self.stride();
    let stride = dims.iter().map(|&i| self_stride[i]).collect();   

    self.set_shape(shape);
    self.set_stride(stride);
  }

  fn flatten(&mut self) {
    let shape: Vec<usize> = vec![self.shape().iter().product()];
    let stride = vec![1];

    self.set_shape(shape);
    self.set_stride(stride);
  }

  fn squeeze(&mut self) {
    // Remove all 1 dimension from the shape
    let shape: Vec<usize> = self.shape().to_owned().iter().filter(|&&x| x != 1).cloned().collect();
    let stride = TensorStorage::compute_strides(&shape);

    self.set_shape(shape);
    self.set_stride(stride);
  } 

  fn unsqueeze(&mut self, dim: usize) {
    let mut shape: Vec<usize> = self.shape().to_owned();
    shape.insert(dim, 1);
    let stride = TensorStorage::compute_strides(&shape);

    self.set_shape(shape);
    self.set_stride(stride);
  }

  fn broadcast(&self, new_shape: &[usize]) -> TensorStorage {
    // Verify broadcast compatibility and get output shape
    let broadcast_shape = self.compute_broadcast_shape(new_shape);
    
    // Calculate new strides for broadcasting
    let broadcast_strides = self.compute_broadcast_strides(&broadcast_shape);

    TensorStorage::create(self.data(), broadcast_shape, broadcast_strides)
  }

  /// Compute broadcast shape between two shapes
  fn compute_broadcast_shape(&self, target_shape: &[usize]) -> Vec<usize> {
    let self_rank = self.shape().len();
    let target_rank = target_shape.len();
    let max_rank = std::cmp::max(self_rank, target_rank);
    
    // Pad shapes with 1s to match maximum rank
    let self_padded = self.pad_shape(max_rank);
    let mut result_shape = Vec::with_capacity(max_rank);

    // Compare dimensions from right to left
    for i in 0..max_rank {
      let self_dim = self_padded[i];
      let target_dim = if i >= max_rank - target_rank {
          target_shape[i - (max_rank - target_rank)]
        } else {
          1
        };

      if self_dim == target_dim {
        result_shape.push(self_dim);
      } else if self_dim == 1 {
        result_shape.push(target_dim);
      } else if target_dim == 1 {
        result_shape.push(self_dim);
      } else {
        panic!(
          "Incompatible broadcast dimensions: {} and {}",
          self_dim, target_dim
        )
      }
    }

    result_shape
  }

  /// Compute broadcast strides
  fn compute_broadcast_strides(&self, broadcast_shape: &[usize]) -> Vec<usize> {
    let self_rank = self.shape().len();
    let broadcast_rank = broadcast_shape.len();
    let rank_diff = broadcast_rank - self_rank;
    
    let mut new_strides = vec![0; broadcast_rank];
    
    // Fill in strides for dimensions that match or are broadcasted
    for i in 0..self_rank {
      let broadcast_idx = i + rank_diff;
      if broadcast_shape[broadcast_idx] == self.shape()[i] {
        new_strides[broadcast_idx] = self.stride()[i];
      } else if self.shape()[i] == 1 {
        new_strides[broadcast_idx] = 0;  // Stride of 0 for broadcasted dimensions
      } else {
        panic!("Invalid broadcast shape")
      }
    }

    new_strides
  }

  /// Pad shape with ones on the left
  fn pad_shape(&self, target_rank: usize) -> Vec<usize> {
    let mut padded = vec![1; target_rank];
    let rank_diff = target_rank - self.shape().len();
    padded[rank_diff..].copy_from_slice(self.shape());
    padded
  }

  fn broadcast_tensors(a: &Self, b: &Self) -> (Self, Self) {
    // Get the shape that both tensors should broadcast to
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // Use a's compute_broadcast_shape to get the final shape
    let broadcast_shape = a.compute_broadcast_shape(b_shape);
    
    // Broadcast both tensors to the new shape
    let broadcast_a = a.broadcast(&broadcast_shape);
    let broadcast_b = b.broadcast(&broadcast_shape);
    
    (broadcast_a, broadcast_b)
  }
  
  
}


