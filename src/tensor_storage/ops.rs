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


#[link(name = "openblas")] // Replace "openblas" with the library you installed if different
extern "C" {
  fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
  fn cblas_dgemv(Layout: u8, trans: u8, m: i32, n: i32, alpha: f64, a: *const f64, lda: i32, x: *const f64, incx: i32, beta: f64, y: *mut f64, incy: i32);
  fn cblas_sgemm(Layout: u8, transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: f32, c: *mut f32, ldc: i32);
}

// CBLAS_LAYOUT
const CBLAS_ROW_MAJOR: u8 = 101;
const CBLAS_COL_MAJOR: u8 = 102;

// CBLAS_TRANSPOSE
const CBLAS_NO_TRANS: u8 = 111;
const CBLAS_TRANS: u8 = 112;
const CBLAS_CONJ_TRANS: u8 = 113;

pub trait BLASTensorOps {
  fn matmul(&self, other: &Self, trans_a: bool, trans_b: bool) -> Self;
}


impl BLASTensorOps for TensorStorage {
  fn matmul(&self, other: &Self, transpose_self: bool, transpose_other: bool) -> Self {
    if self.shape().len() != 2 { panic!("Can't Matmul on non-matrices"); }

    println!("Self shape: {:?}", self.shape());
    println!("other shape: {:?}", other.shape());
    // A = 3x2  B = 3x5
    if transpose_self && (self.shape()[0] != other.shape()[0]) { 
      panic!("Matrix dimensions do not match for multiplication.");
    } else if transpose_other && (self.shape()[1] != other.shape()[1]) { 
      panic!("Matrix dimensions do not match for multiplication.");
    } else if !transpose_other && !transpose_self && self.shape()[1] != other.shape()[0] { 
      panic!("Matrix dimensions do not match for multiplication.");
    }

    let layout = CBLAS_ROW_MAJOR;
    let trans_a = if transpose_self { CBLAS_TRANS } else { CBLAS_NO_TRANS }; let trans_b = if transpose_other { CBLAS_TRANS } else { CBLAS_NO_TRANS };
    let a = self.data();
    let m = if !transpose_self { self.shape()[0] as i32 } else { self.shape()[1] as i32 } ; let k = if !transpose_self { self.shape()[1] as i32 } else { self.shape()[0] as i32 };
    let n = if !transpose_other { other.shape()[1] as i32 } else { other.shape()[0] as i32 };
    let lda = self.shape()[1] as i32;   // width/columns of matrix A
    let ldb = other.shape()[1] as i32;  // width/columns of matrix B in its original form
    let ldc = n;  // width of output matrix
    let b = other.data();
    let alpha = 1.;
    let beta = 0.;
    let dim_c = vec![m as usize,n as usize];
    let mut c = vec![0.0; dim_c.iter().product()];  

    unsafe {
      let result = cblas_sgemm(layout, trans_a, trans_b, m, n, k, alpha, a.borrow().as_ptr(), lda, b.borrow().as_ptr(), ldb, beta, c.as_mut_ptr(), ldc);
      TensorStorage::new(c, dim_c)
    }

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
        return TensorStorage::new(vec![sum], vec![1]);
    }

    // Ensure at least one dimension
    if new_shape.is_empty() {
        new_shape.push(1);
    }

    let mut result = vec![0.0; new_shape.iter().product()];
    let indices = vec![0; self.shape().len()];
    
    // Sum values maintaining non-summed dimensions
    let mut sum = 0.0;
    for i in 0..self.data().borrow().len() {
        sum += self.data().borrow()[i];
    }
    result[0] = sum;

    TensorStorage::new(result, new_shape)
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

