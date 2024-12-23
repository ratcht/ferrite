use crate::tensor_storage::*;
use super::base::*;
use super::function::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign, Div, DivAssign};
use std::rc::Rc;
use std::cell::RefCell;

impl TensorOps for Tensor {
  fn add_tensor(&self, other: &Self) -> Self {
    let tensor = self.tensor().add_tensor(other.tensor());

    let grad_fn = if *self.requires_grad() {
      Some(Rc::new(AddGrad::new(self, other)) as Rc<dyn GradientFunction>)
      } else { None };

    let grad = if *self.requires_grad() {
      Some(Rc::new(AddGrad::new(self, other)) as Rc<dyn GradientFunction>)
      } else { None };

    Tensor::new(tensor, *self.requires_grad() || *other.requires_grad(), grad_fn, grad)
  }

  fn sub_tensor(&self, other: &Self) -> Self {
    let tensor = self.tensor().sub_tensor(other.tensor());

    Tensor::new(tensor, *self.requires_grad() || *other.requires_grad(), None, None)  
  }

  fn mul_tensor(&self, other: &Self) -> Self {
    let tensor = self.tensor().mul_tensor(other.tensor());

    Tensor::new(tensor, *self.requires_grad() || *other.requires_grad(), None, None)  
  }

  fn div_tensor(&self, other: &Self) -> Self {
    let tensor = self.tensor().div_tensor(other.tensor());

    Tensor::new(tensor, *self.requires_grad() || *other.requires_grad(), None, None)  
  }

  fn add_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().add_f32(other);

    Tensor::new(tensor, *self.requires_grad(), None, None)  
  }

  fn sub_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().sub_f32(other);

    Tensor::new(tensor, *self.requires_grad(), None, None) 
  }

  fn mul_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().mul_f32(other);

    Tensor::new(tensor, *self.requires_grad(), None, None) 
  }

  fn div_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().div_f32(other);

    Tensor::new(tensor, *self.requires_grad(), None, None) 
  }

  fn add_tensor_assign(&mut self, other: &Self) {
    self.tensor_mut().add_tensor_assign(other.tensor());
  }

  fn sub_tensor_assign(&mut self, other: &Self) {
    self.tensor_mut().sub_tensor_assign(other.tensor());
  }

  fn mul_tensor_assign(&mut self, other: &Self) {
    self.tensor_mut().mul_tensor_assign(other.tensor());
  }

  fn div_tensor_assign(&mut self, other: &Self) {
    self.tensor_mut().div_tensor_assign(other.tensor());
  }

  fn add_f32_assign(&mut self, other: f32) {
    self.tensor_mut().add_f32_assign(other);
  }

  fn sub_f32_assign(&mut self, other: f32) {
    self.tensor_mut().sub_f32_assign(other);
  }

  fn mul_f32_assign(&mut self, other: f32) {
    self.tensor_mut().mul_f32_assign(other);
  }

  fn div_f32_assign(&mut self, other: f32) {
    self.tensor_mut().div_f32_assign(other);
  }

  fn matmul(&self, other: &Self) -> Self {
    let tensor = self.tensor().matmul(other.tensor());

    Tensor::new(tensor, *self.requires_grad() || *other.requires_grad(), None, None)
  }

  fn sum(&self) -> f32 {
    self.tensor().sum()
  }

  fn product(&self) -> f32 {
    self.tensor().product()
  }

  fn mean(&self) -> f32 {
    self.tensor().mean()
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
