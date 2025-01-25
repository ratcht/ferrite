use crate::{DeviceStorage, Storage};


pub trait TransformOps {
  fn apply<F>(&self, op: F) -> Self
  where
    F: Fn(f32) -> f32;
  
  fn apply_assign<F>(&mut self, op: F)
  where
    F: Fn(f32) -> f32;

  fn elementwise_op<F>(&self, other: &Self, op: F) -> Self
  where
  F: Fn(f32, f32) -> f32;

  fn scalar_op<F>(&self, scalar: f32, op: F) -> Self
  where
  F: Fn(f32, f32) -> f32;

  fn elementwise_op_assign<F>(&mut self, other: &Self, op: F)
  where
  F: Fn(f32, f32) -> f32;

  fn scalar_op_assign<F>(&mut self, scalar: f32, op: F)
  where
  F: Fn(f32, f32) -> f32;

  

  fn sum_dim(&self, dims: &[bool]) -> Self;

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

macro_rules! match_storage {
  // Binary operation with two storage arguments
  (binary $self:expr, $method:ident, $other:expr $(, $args:expr)*) => {
    match ($self, $other) {
      (Storage::Cpu(cpu_self), Storage::Cpu(cpu_other)) => {
        Storage::Cpu(cpu_self.$method(cpu_other $(, $args)*))
      }
      _ => unimplemented!("Cross-device operations not supported"),
    }
  };

  // Unary operation with single storage argument
  (unary $self:expr, $method:ident $(, $args:expr)*) => {
    match $self {
      Storage::Cpu(cpu) => Storage::Cpu(cpu.$method($($args),*)),
      _ => unimplemented!("Device not supported"),
    }
  };
}

macro_rules! match_storage_assign {
  // Binary operation with two storage arguments
  (binary $self:expr, $method:ident, $other:expr $(, $args:expr)*) => {
    match ($self, $other) {
      (Storage::Cpu(cpu_self), Storage::Cpu(cpu_other)) => {
        cpu_self.$method(cpu_other $(, $args)*)
      }
      _ => unimplemented!("Cross-device operations not supported"),
    }
  };

  // Unary operation with single storage argument
  (unary $self:expr, $method:ident $(, $args:expr)*) => {
    match $self {
      Storage::Cpu(cpu) => cpu.$method($($args),*),
      _ => unimplemented!("Device not supported"),
    }
  };
}


impl TransformOps for Storage {
  fn apply<F>(&self, op: F) -> Self
  where
    F: Fn(f32) -> f32 {
    match_storage!(unary self, apply, op)
  }

  fn apply_assign<F>(&mut self, op: F)
  where
    F: Fn(f32) -> f32 {
    match_storage_assign!(unary self, apply_assign, op)
  }

  fn elementwise_op<F>(&self, other: &Self, op: F) -> Self
  where
  F: Fn(f32, f32) -> f32 {
    match_storage!(binary self, elementwise_op, other, op)
  }

  fn scalar_op<F>(&self, scalar: f32, op: F) -> Self
  where
  F: Fn(f32, f32) -> f32 {
    match_storage!(unary self, scalar_op, scalar, op)
  }

  fn elementwise_op_assign<F>(&mut self, other: &Self, op: F)
  where
  F: Fn(f32, f32) -> f32 {
    match_storage_assign!(binary self, elementwise_op_assign, other, op)
  }

  fn scalar_op_assign<F>(&mut self, scalar: f32, op: F)
  where
  F: Fn(f32, f32) -> f32 {
    match_storage_assign!(unary self, scalar_op_assign, scalar, op)
  }

  fn sum_dim(&self, dims: &[bool]) -> Self {
    match_storage!(unary self, sum_dim, dims)
  }

  fn reshape(&mut self, new_shape: Vec<usize>) {
    match_storage_assign!(unary self, reshape, new_shape)
  }

  fn permute(&mut self, dims: &[usize]) {
    match_storage_assign!(unary self, permute, dims)
  }

  fn transpose(&self) -> Self {
    match_storage!(unary self, transpose)
  }

  fn flatten(&mut self) {
    match_storage_assign!(unary self, flatten)
  }

  fn squeeze(&mut self) {
    match_storage_assign!(unary self, squeeze)
  }

  fn unsqueeze(&mut self, dim: usize) {
    match_storage_assign!(unary self, unsqueeze, dim)
  }

  fn broadcast(&self, new_shape: &[usize]) -> Self {
    match_storage!(unary self, broadcast, new_shape)
  }

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

  fn pad_shape(&self, target_rank: usize) -> Vec<usize> {
    let mut padded = vec![1; target_rank];
    let rank_diff = target_rank - self.shape().len();
    padded[rank_diff..].copy_from_slice(self.shape());
    padded
  }

  fn broadcast_tensors(a: &Self, b: &Self) -> (Self, Self) where Self: Sized {
    // Use a's compute_broadcast_shape to get the final shape
    let broadcast_shape = a.compute_broadcast_shape(b.shape());

    // Broadcast both tensors to the new shape
    let broadcast_a = a.broadcast(&broadcast_shape);
    let broadcast_b = b.broadcast(&broadcast_shape);
    
    (broadcast_a, broadcast_b)
  }
}