use super::{base::TensorStorage, TensorCreation, TensorShape};  // Import from parent module's base.rs
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign, Div, DivAssign};
use ndarray::array;

pub trait TensorOps {
  fn add_tensor(&self, other: &Self) -> Self;
  fn add_tensor_assign(&mut self, other: &Self);

  fn sub_tensor(&self, other: &Self) -> Self;
  fn sub_tensor_assign(&mut self, other: &Self);

  fn mul_tensor(&self, other: &Self) -> Self;
  fn mul_tensor_assign(&mut self, other: &Self);

  fn div_tensor(&self, other: &Self) -> Self;
  fn div_tensor_assign(&mut self, other: &Self);

  fn add_f32(&self, other: f32) -> Self;
  fn add_f32_assign(&mut self, other: f32);

  fn sub_f32(&self, other: f32) -> Self;
  fn sub_f32_assign(&mut self, other: f32);

  fn mul_f32(&self, other: f32) -> Self;
  fn mul_f32_assign(&mut self, other: f32);

  fn div_f32(&self, other: f32) -> Self;
  fn div_f32_assign(&mut self, other: f32);

  fn matmul(&self, other: &Self) -> Self;

  fn sum(&self) -> Self;
  fn product(&self) -> Self;
  fn mean(&self) -> Self;
}

impl TensorOps for TensorStorage {
  fn add_tensor(&self, other: &Self) -> Self {
    let (tensor_a, tensor_b) = TensorStorage::broadcast_tensors(self, other);
    tensor_a.elementwise_op(&tensor_b, |a, b| a + b)
  }

  fn sub_tensor(&self, other: &Self) -> Self {
    let (tensor_a, tensor_b) = TensorStorage::broadcast_tensors(self, other);
    tensor_a.elementwise_op(&tensor_b, |a, b| a - b)
  }

  fn mul_tensor(&self, other: &Self) -> Self {
    let (tensor_a, tensor_b) = TensorStorage::broadcast_tensors(self, other);
    tensor_a.elementwise_op(&tensor_b, |a, b| a * b)
  }

  fn div_tensor(&self, other: &Self) -> Self {
    let (tensor_a, tensor_b) = TensorStorage::broadcast_tensors(self, other);
    tensor_a.elementwise_op(&tensor_b, |a, b| a / b)
  }

  fn add_f32(&self, other: f32) -> Self {
    self.scalar_op(other, |a, b| a + b)
  }

  fn sub_f32(&self, other: f32) -> Self {
    self.scalar_op(other, |a, b| a - b)
  }

  fn mul_f32(&self, other: f32) -> Self {
    self.scalar_op(other, |a, b| a * b)
  }

  fn div_f32(&self, other: f32) -> Self {
    self.scalar_op(other, |a, b| a / b)
  }

  fn add_tensor_assign(&mut self, other: &Self) {
    // Only broadcast one side
    let broadcast_b = other.broadcast(&self.shape());
    self.elementwise_op_assign(&broadcast_b, |a, b| a + b)
  }

  fn sub_tensor_assign(&mut self, other: &Self) {
    self.elementwise_op_assign(other, |a, b| a - b)
  }

  fn mul_tensor_assign(&mut self, other: &Self) {
    self.elementwise_op_assign(other, |a, b| a * b)
  }

  fn div_tensor_assign(&mut self, other: &Self) {
    self.elementwise_op_assign(other, |a, b| a / b)
  }

  fn add_f32_assign(&mut self, other: f32) {
    self.scalar_op_assign(other, |a, b| a + b)
  }

  fn sub_f32_assign(&mut self, other: f32) {
    self.scalar_op_assign(other, |a, b| a - b)
  }

  fn mul_f32_assign(&mut self, other: f32) {
    self.scalar_op_assign(other, |a, b| a * b)
  }

  fn div_f32_assign(&mut self, other: f32) {
    self.scalar_op_assign(other, |a, b| a / b)
  }

  fn matmul(&self, other: &Self) -> Self {
    if self.shape().len() != 2 { panic!("Can't Matmul on non-matrices"); }
    if self.shape()[1] != other.shape()[0] { panic!("Array2D dimensions do not match for multiplication."); }

    let mut data = vec![1.0; self.shape()[0] * other.shape()[1]];

    for col in 0..other.shape()[1] {
      for row in 0..self.shape()[0] {
        let mut dot = 0.;
        for i in 0..self.shape()[1] {
          dot += self.data().borrow()[row * self.shape()[1] + i] * other.data().borrow()[i * other.shape()[1] + col];
        }
        data[row * other.shape()[1] + col] = dot;
      }
    }

    TensorStorage::new(data, vec![self.shape()[0], other.shape()[1]])
  }

  fn sum(&self) -> Self {
    let data: f32 = self.data().borrow().iter().sum();
    TensorStorage::from_ndarray(&array![data], None)
  }

  fn product(&self) -> Self {
    let data: f32 = self.data().borrow().iter().sum();
    TensorStorage::from_ndarray(&array![data], None)
  }

  fn mean(&self) -> Self {
    let data: f32 = f32::div(self.data().borrow().iter().sum(), self.data().borrow().len() as f32);
    TensorStorage::from_ndarray(&array![data], None)
  }
}

impl TensorStorage {
  fn elementwise_op<F>(&self, other: &Self, op: F) -> Self
  where
    F: Fn(f32, f32) -> f32,
  {
    let total_elements = self.shape().iter().product();
    let mut result = vec![0.0; total_elements];
    
    // Get data once to avoid multiple borrows
    let self_binding = self.data();
    let self_data = self_binding.borrow();
    let other_binding = other.data();
    let other_data = other_binding.borrow();
    
    // Pre-calculate dimensions for faster access
    let rank = self.shape().len();
    let shape = self.shape();
    let self_strides = self.stride();
    let other_strides = other.stride();
    
    // Use chunk size optimization for contiguous dimensions
    let mut chunk_size = 1;
    let mut contiguous_dims = 0;
    for dim in (0..rank).rev() {
      if self_strides[dim] == chunk_size && other_strides[dim] == chunk_size {
        chunk_size *= shape[dim];
        contiguous_dims += 1;
      } else {
        break;
      }
    }
    
    let outer_dims = rank - contiguous_dims;
    let mut indices = vec![0; outer_dims];
    
    // Process chunks
    let chunks = total_elements / chunk_size;
    for chunk_idx in 0..chunks {
      // Calculate base indices for the chunk
      let mut self_base_idx = 0;
      let mut other_base_idx = 0;
      
      for (dim, &idx) in indices.iter().enumerate() {
        self_base_idx += idx * self_strides[dim];
        other_base_idx += idx * other_strides[dim];
      }
      
      // Process the entire chunk
      let result_start = chunk_idx * chunk_size;
      for i in 0..chunk_size {
        let self_val = self_data[self_base_idx + i];
        let other_val = other_data[other_base_idx + i];
        result[result_start + i] = op(self_val, other_val);
      }
      
      // Update indices for outer dimensions
      for dim in (0..outer_dims).rev() {
        indices[dim] += 1;
        if indices[dim] < shape[dim] {
          break;
        }
        indices[dim] = 0;
      }
    }

    TensorStorage::new(result, self.shape().clone())
  }

  fn scalar_op<F>(&self, scalar: f32, op: F) -> Self
  where
    F: Fn(f32, f32) -> f32,
  {
    let data = self.data().borrow().iter()
      .map(|a| op(*a, scalar))
      .collect();

    TensorStorage::new(data, self.shape().clone())
  }

  fn elementwise_op_assign<F>(&mut self, other: &Self, op: F)
  where
    F: Fn(f32, f32) -> f32,
  {
    let total_elements = self.shape().iter().product();
    let mut result = vec![0.0; total_elements];
    
    // Get data once to avoid multiple borrows
    let self_binding = self.data();
    let self_data = self_binding.borrow();
    let other_binding = other.data();
    let other_data = other_binding.borrow();
    
    // Pre-calculate dimensions for faster access
    let rank = self.shape().len();
    let shape = self.shape();
    let self_strides = self.stride();
    let other_strides = other.stride();
    
    // Use chunk size optimization for contiguous dimensions
    let mut chunk_size = 1;
    let mut contiguous_dims = 0;
    for dim in (0..rank).rev() {
      if self_strides[dim] == chunk_size && other_strides[dim] == chunk_size {
        chunk_size *= shape[dim];
        contiguous_dims += 1;
      } else {
        break;
      }
    }
    
    let outer_dims = rank - contiguous_dims;
    let mut indices = vec![0; outer_dims];
    
    // Process chunks
    let chunks = total_elements / chunk_size;
    for chunk_idx in 0..chunks {
      // Calculate base indices for the chunk
      let mut self_base_idx = 0;
      let mut other_base_idx = 0;
      
      for (dim, &idx) in indices.iter().enumerate() {
        self_base_idx += idx * self_strides[dim];
        other_base_idx += idx * other_strides[dim];
      }
      
      // Process the entire chunk
      let result_start = chunk_idx * chunk_size;
      for i in 0..chunk_size {
        let self_val = self_data[self_base_idx + i];
        let other_val = other_data[other_base_idx + i];
        result[result_start + i] = op(self_val, other_val);
      }
      
      // Update indices for outer dimensions
      for dim in (0..outer_dims).rev() {
        indices[dim] += 1;
        if indices[dim] < shape[dim] {
          break;
        }
        indices[dim] = 0;
      }
    }

    self.set_data(result);
  }

  fn scalar_op_assign<F>(&mut self, scalar: f32, op: F)
  where
    F: Fn(f32, f32) -> f32,
  {
    let data = self.data().borrow().iter()
      .map(|a| op(*a, scalar))
      .collect();

    self.set_data(data);
  }

  pub fn sum_dim(&self, dims: &[bool]) -> Self {
    let mut new_shape: Vec<usize> = Vec::new();
    let mut new_data: Vec<f32> = Vec::new();
    
    // Calculate new shape
    for (i, &dim) in self.shape().iter().enumerate() {
      if !dims[i] {
        new_shape.push(dim);
      }
    }
    
    // Helper function to compute flat index
    fn compute_index(indices: &[usize], shape: &[usize], strides: &[usize]) -> usize {
      indices.iter()
        .zip(strides.iter())
        .map(|(&idx, &stride)| idx * stride)
        .sum()
    }
    
    // Perform the summation
    let mut indices = vec![0; self.shape().len()];
    let mut done = false;
    
    while !done {
      let mut sum = 0.0;
      let mut sum_indices: Vec<f32> = Vec::new();
      
      // Sum over the reduction dimensions
      let mut reduce_indices = indices.clone();
      let mut reduce_done = false;
        
      while !reduce_done {
        sum += self.get(&reduce_indices);
        
        // Increment indices for reduction dimensions
        reduce_done = true;
        for (i, reduce) in dims.iter().enumerate() {
          if *reduce {
            reduce_indices[i] += 1;
            if reduce_indices[i] < self.shape()[i] {
              reduce_done = false;
              break;
            }
            reduce_indices[i] = 0;
          }
        }
      }
        
      new_data.push(sum);
      
      // Increment indices for non-reduction dimensions
      done = true;
      for (i, reduce) in dims.iter().enumerate() {
        if !reduce {
          indices[i] += 1;
          if indices[i] < self.shape()[i] {
            done = false;
            break;
          }
          indices[i] = 0;
        }
      }
    }
    
    TensorStorage::new(new_data, new_shape)
  }
}

impl Add<&TensorStorage> for &TensorStorage {
  type Output = TensorStorage;

  fn add(self, rhs: &TensorStorage) -> Self::Output {
    self.add_tensor(rhs)
  }
}

impl AddAssign<&TensorStorage> for TensorStorage {
  fn add_assign(&mut self, rhs: &TensorStorage) {
    self.add_tensor_assign(rhs);
  }
}

impl Sub<&TensorStorage> for &TensorStorage {
  type Output = TensorStorage;

  fn sub(self, rhs: &TensorStorage) -> Self::Output {
    self.sub_tensor(rhs)
  }
}

impl SubAssign<&TensorStorage> for TensorStorage {
  fn sub_assign(&mut self, rhs: &TensorStorage) {
    self.sub_tensor_assign(rhs);
  }
}

impl Mul<&TensorStorage> for &TensorStorage {
  type Output = TensorStorage;

  fn mul(self, rhs: &TensorStorage) -> Self::Output {
    self.mul_tensor(rhs)
  }
}

impl MulAssign<&TensorStorage> for TensorStorage {
  fn mul_assign(&mut self, rhs: &TensorStorage) {
    self.mul_tensor_assign(rhs);
  }
}

impl Div<&TensorStorage> for &TensorStorage {
  type Output = TensorStorage;

  fn div(self, rhs: &TensorStorage) -> Self::Output {
    self.div_tensor(rhs)
  }
}

impl DivAssign<&TensorStorage> for TensorStorage {
  fn div_assign(&mut self, rhs: &TensorStorage) {
    self.div_tensor_assign(rhs);
  }
}

impl Add<f32> for &TensorStorage {
  type Output = TensorStorage;
  fn add(self, rhs: f32) -> Self::Output {
    self.add_f32(rhs)
  }
}

impl AddAssign<f32> for TensorStorage {
  fn add_assign(&mut self, rhs: f32) {
    self.add_f32_assign(rhs);
  }
}

impl Sub<f32> for &TensorStorage {
  type Output = TensorStorage;
  fn sub(self, rhs: f32) -> Self::Output {
    self.sub_f32(rhs)
  }
}

impl SubAssign<f32> for TensorStorage {
  fn sub_assign(&mut self, rhs: f32) {
    self.sub_f32_assign(rhs);
  }
}

impl Mul<f32> for &TensorStorage {
  type Output = TensorStorage;
  fn mul(self, rhs: f32) -> Self::Output {
    self.mul_f32(rhs)
  }
}

impl MulAssign<f32> for TensorStorage {
  fn mul_assign(&mut self, rhs: f32) {
    self.mul_f32_assign(rhs);
  }
}

impl Div<f32> for &TensorStorage {
  type Output = TensorStorage;
  fn div(self, rhs: f32) -> Self::Output {
    self.div_f32(rhs)
  }
}

impl DivAssign<f32> for TensorStorage {
  fn div_assign(&mut self, rhs: f32) {
    self.div_f32_assign(rhs);
  }
}
