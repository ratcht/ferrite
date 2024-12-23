use super::base::Tensor;  // Import from parent module's base.rs
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
}

impl TensorOps for Tensor {
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
        data[row*other.shape()[1]+ col ] = dot;
      }
    }

    Tensor::new(data, vec![self.shape()[0], other.shape()[1]], self.requires_grad() || other.requires_grad())

  }
}

impl Tensor {
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

    Tensor::new(data, self.shape().clone(), self.requires_grad() || other.requires_grad())
  }

  fn scalar_op<F>(&self, scalar: f32, op: F) -> Self
  where
    F: Fn(f32, f32) -> f32,
  {
    let data = self.data().borrow().iter()
      .map(|a| op(*a, scalar))
      .collect();

    Tensor::new(data, self.shape().clone(), self.requires_grad())
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
  

  


impl Add<&Tensor> for &Tensor {
  type Output = Tensor;

  fn add(self, rhs: &Tensor) -> Self::Output {
    self.add_tensor(rhs)
  }
}

impl AddAssign<&Tensor> for Tensor {
  fn add_assign(&mut self, rhs: &Tensor) {
    self.add_tensor_assign(rhs);
  }
}


impl Sub<&Tensor> for &Tensor {
  type Output = Tensor;

  fn sub(self, rhs: &Tensor) -> Self::Output {
    self.sub_tensor(rhs)
  }
}

impl SubAssign<&Tensor> for Tensor {
  fn sub_assign(&mut self, rhs: &Tensor) {
    self.sub_tensor_assign(rhs);
  }
}

impl Mul<&Tensor> for &Tensor {
  type Output = Tensor;

  fn mul(self, rhs: &Tensor) -> Self::Output {
    self.mul_tensor(rhs)
  }
}

impl MulAssign<&Tensor> for Tensor {
  fn mul_assign(&mut self, rhs: &Tensor) {
    self.mul_tensor_assign(rhs);
  }
}


impl Div<&Tensor> for &Tensor {
  type Output = Tensor;

  fn div(self, rhs: &Tensor) -> Self::Output {
    self.div_tensor(rhs)
  }
}

impl DivAssign<&Tensor> for Tensor {
  fn div_assign(&mut self, rhs: &Tensor) {
    self.div_tensor_assign(rhs);
  }
}


impl Add<f32> for &Tensor {
  type Output = Tensor;
  fn add(self, rhs: f32) -> Self::Output {
    self.add_f32(rhs)
  }
}

impl AddAssign<f32> for Tensor {
  fn add_assign(&mut self, rhs: f32) {
    self.add_f32_assign(rhs);
  }
}

impl Sub<f32> for &Tensor {
  type Output = Tensor;
  fn sub(self, rhs: f32) -> Self::Output {
    self.sub_f32(rhs)
  }
}

impl SubAssign<f32> for Tensor {
  fn sub_assign(&mut self, rhs: f32) {
    self.sub_f32_assign(rhs);
  }
}

impl Mul<f32> for &Tensor {
  type Output = Tensor;
  fn mul(self, rhs: f32) -> Self::Output {
    self.mul_f32(rhs)
  }
}

impl MulAssign<f32> for Tensor {
  fn mul_assign(&mut self, rhs: f32) {
    self.mul_f32_assign(rhs);
  }
}

impl Div<f32> for &Tensor {
  type Output = Tensor;
  fn div(self, rhs: f32) -> Self::Output {
    self.div_f32(rhs)
  }
}

impl DivAssign<f32> for Tensor {
  fn div_assign(&mut self, rhs: f32) {
    self.div_f32_assign(rhs);
  }
}


