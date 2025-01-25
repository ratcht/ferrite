use crate::*;

impl TransformOps for CpuStorage {
  fn apply_assign<F>(&mut self, op: F)
  where
    F: Fn(f32) -> f32,
  {
    let data = self.data().borrow().iter()
      .map(|a| op(*a))
      .collect();

    self.set_data(data);
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

  fn reshape(&mut self, new_shape: Vec<usize>) {
    self.set_shape(new_shape);
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
    let stride = Self::compute_strides(&shape);

    self.set_shape(shape);
    self.set_stride(stride);
  } 

  fn unsqueeze(&mut self, dim: usize) {
    let mut shape: Vec<usize> = self.shape().to_owned();
    shape.insert(dim, 1);
    let stride = Self::compute_strides(&shape);

    self.set_shape(shape);
    self.set_stride(stride);
  }

  fn apply<F>(&self, op: F) -> Self
  where
    F: Fn(f32) -> f32,
  {
    let data = self.data().borrow().iter()
      .map(|a| op(*a))
      .collect();

    Self::new(data, self.shape().clone())
  }

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

    Self::new(result, self.shape().clone())
  }

  fn scalar_op<F>(&self, scalar: f32, op: F) -> Self
  where
    F: Fn(f32, f32) -> f32,
  {
    let data = self.data().borrow().iter()
      .map(|a| op(*a, scalar))
      .collect();

    Self::new(data, self.shape().clone())
  }

  fn sum_dim(&self, dims: &[bool]) -> Self {
    // Handle scalar case
    if self.shape().len() == 1 && self.shape()[0] == 1 {
        return self.clone();
    }

    // Calculate new shape excluding summed dimensions
    let mut new_shape: Vec<usize> = self.shape().iter()
        .zip(dims.iter().chain(std::iter::repeat(&false)))
        .filter_map(|(&dim, &should_sum)| if !should_sum { Some(dim) } else { None })
        .collect();

    // If all dimensions are summed, return scalar
    if new_shape.is_empty() {
        let sum: f32 = self.data().borrow().iter().sum();
        return Self::new(vec![sum], vec![1]);
    }

    // Ensure at least one dimension
    if new_shape.is_empty() {
        new_shape.push(1);
    }

    let mut result = vec![0.0; new_shape.iter().product()];
    
    // Sum values maintaining non-summed dimensions
    let mut sum = 0.0;
    for i in 0..self.data().borrow().len() {
        sum += self.data().borrow()[i];
    }
    result[0] = sum;

    Self::new(result, new_shape)
  }

  fn transpose(&self) -> Self {
    // Transpose by swapping dimensions & strides
    if self.shape().len() != 2 { panic!("Must be 2-D Tensor (Matrix)"); }

    let mut shape = self.shape().to_owned();
    shape.reverse();

    let mut stride = self.stride().to_owned();
    stride.reverse();

    Self::create(self.data(), shape, stride)
  }

  fn broadcast(&self, new_shape: &[usize]) -> Self {
    // Verify broadcast compatibility and get output shape
    let broadcast_shape = self.compute_broadcast_shape(new_shape);
    
    // Calculate new strides for broadcasting
    let broadcast_strides = self.compute_broadcast_strides(&broadcast_shape);

    Self::create(self.data(), broadcast_shape, broadcast_strides)
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
    let broadcast_shape = a.compute_broadcast_shape(b.shape());
    let broadcast_a = a.broadcast(&broadcast_shape);
    let broadcast_b = b.broadcast(&broadcast_shape);
    (broadcast_a, broadcast_b)
  }
}
