use super::base::TensorStorage;  // Import from parent module's base.rs
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign, Div, DivAssign};

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

  fn sum(&self) -> f32;
  fn product(&self) -> f32;
  fn mean(&self) -> f32;
}

impl TensorOps for TensorStorage {
  fn add_tensor(&self, other: &Self) -> Self {
    self.elementwise_op(other, |a, b| a + b)
  }

  fn sub_tensor(&self, other: &Self) -> Self {
    self.elementwise_op(other, |a, b| a - b)
  }

  fn mul_tensor(&self, other: &Self) -> Self {
    self.elementwise_op(other, |a, b| a * b)
  }

  fn div_tensor(&self, other: &Self) -> Self {
    self.elementwise_op(other, |a, b| a / b)
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
    self.elementwise_op_assign(other, |a, b| a + b)
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

  fn sum(&self) -> f32 {
    self.data().borrow().iter().sum()
  }

  fn product(&self) -> f32 {
    self.data().borrow().iter().sum()
  }

  fn mean(&self) -> f32 {
    f32::div(self.sum(), self.data().borrow().len() as f32)
  }
}

impl TensorStorage {
  fn elementwise_op<F>(&self, other: &Self, op: F) -> Self
  where
    F: Fn(f32, f32) -> f32,
  {
    if self.shape() != other.shape() {
      panic!("Tensor shapes don't match!")
    }

    let data = self.data().borrow().iter()
      .zip(other.data().borrow().iter())
      .map(|(a, b)| op(*a, *b))
      .collect();

    TensorStorage::new(data, self.shape().clone())
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
    if self.shape() != other.shape() {
      panic!("Tensor shapes don't match!")
    }

    let data = self.data().borrow().iter()
      .zip(other.data().borrow().iter())
      .map(|(a, b)| op(*a, *b))
      .collect();

    self.set_data(data);
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
