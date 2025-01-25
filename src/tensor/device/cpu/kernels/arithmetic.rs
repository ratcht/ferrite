use crate::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign, Div, DivAssign};



impl ArithmeticOps for CpuStorage {
  fn add_tensor(&self, other: &Self) -> Self {
    let (tensor_a, tensor_b) = CpuStorage::broadcast_tensors(self, other);
    tensor_a.elementwise_op(&tensor_b, |a, b| a + b)
  }

  fn sub_tensor(&self, other: &Self) -> Self {
    let (tensor_a, tensor_b) = CpuStorage::broadcast_tensors(self, other);
    tensor_a.elementwise_op(&tensor_b, |a, b| a - b)
  }

  fn mul_tensor(&self, other: &Self) -> Self {
    let (tensor_a, tensor_b) = CpuStorage::broadcast_tensors(self, other);
    tensor_a.elementwise_op(&tensor_b, |a, b| a * b)
  }

  fn div_tensor(&self, other: &Self) -> Self {
    let (tensor_a, tensor_b) = CpuStorage::broadcast_tensors(self, other);
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

  fn pow_f32(&self, other: f32) -> Self {
    self.scalar_op(other, |a, b| f32::powf(a, b))
  }

  fn greater_than(&self, other: &Self, make_binary: bool) -> Self {
    let (tensor_a, tensor_b) = CpuStorage::broadcast_tensors(self, other);
    tensor_a.elementwise_op(&tensor_b, |a, b| if a > b { 1.0 } else if make_binary { 0.0 } else {-1.0})
  }

  fn greater_than_f32(&self, other: f32, make_binary: bool) -> Self {
    self.scalar_op(other, |a, b| if a > b { 1.0 } else if make_binary { 0.0 } else {-1.0})
  }

  fn less_than(&self, other: &Self, make_binary: bool) -> Self {
    let (tensor_a, tensor_b) = CpuStorage::broadcast_tensors(self, other);
    tensor_a.elementwise_op(&tensor_b, |a, b| if a < b { 1.0 } else if make_binary { 0.0 } else {-1.0})
  }

  fn less_than_f32(&self, other: f32, make_binary: bool) -> Self {
    self.scalar_op(other, |a, b| if a < b { 1.0 } else if make_binary { 0.0 } else {-1.0})
  }

  fn sign(&self) -> Self {
    self.apply(|a| if a > 0. { 1.0 } else if a < 0. { -1. } else { 0.0 })
  }

  fn abs(&self) -> Self {
    self.apply(|a| f32::abs(a))
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

  fn pow_f32_assign(&mut self, other: f32) {
    self.scalar_op_assign(other, |a, b| f32::powf(a, b))
  }


  fn abs_assign(&mut self) {
    self.apply_assign(|a| f32::abs(a))
  }
}
