use crate::tensor_storage::*;
use super::base::*;
use super::grad::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign, Div, DivAssign};
use std::rc::Rc;
use std::cell::RefCell;

impl TensorOps for Tensor {
  fn add_tensor(&self, other: &Self) -> Self {
    // Compute the actual tensor addition
    let tensor = self.tensor().add_tensor(other.tensor());
    
    // Create result tensor
    let requires_grad = *self.requires_grad() || *other.requires_grad();
    let mut result = Tensor::new(tensor, requires_grad);
    
    // Set up gradient function if needed
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(AddGrad::new(
        self, 
        other,
        &result
      ))));
    }
    
    result
  }

  fn sub_tensor(&self, other: &Self) -> Self {
    // Compute the actual tensor addition
    let tensor = self.tensor().sub_tensor(other.tensor());
    
    // Create result tensor
    let requires_grad = *self.requires_grad() || *other.requires_grad();
    let mut result = Tensor::new(tensor, requires_grad);
    
    // Set up gradient function if needed
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(SubGrad::new(
        self, 
        other,
        &result
      ))));
    }
    
    result
  }

  fn mul_tensor(&self, other: &Self) -> Self {
    let tensor = self.tensor().mul_tensor(other.tensor());
    
    let requires_grad = *self.requires_grad() || *other.requires_grad();
    let mut result = Tensor::new(tensor, requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(MulGrad::new(
        self,
        other,
        &result
      ))));
    }
    
    result
  }

  fn div_tensor(&self, other: &Self) -> Self {
    let tensor = self.tensor().div_tensor(other.tensor());
    
    let requires_grad = *self.requires_grad() || *other.requires_grad();
    let mut result = Tensor::new(tensor, requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(DivGrad::new(
        self,
        other,
        &result
      ))));
    }
    
    result
  }
  
  fn sum(&self) -> Self {
    let tensor = self.tensor().sum();
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(SumGrad::new(self, &result))));
    }
    
    result
  }

  fn mean(&self) -> Self {
    let tensor = self.tensor().mean();
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(MeanGrad::new(self, &result))));
    }
    
    result
  }

  fn product(&self) -> Self {
    let tensor = self.tensor().product();
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(ProductGrad::new(self, &result))));
    }
    
    result
  }


  fn pow_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().pow_f32(other);
    
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(PowF32Grad::new(
        self,
        other,
        &result
      ))));
    }
    
    result
  }

  // Additional operations that don't have gradient implementations yet

  fn add_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().add_f32(other);
    Tensor::new(tensor, false)
  }

  fn sub_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().sub_f32(other);
    Tensor::new(tensor, false)
  }

  fn mul_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().mul_f32(other);
    Tensor::new(tensor, false)
  }

  fn div_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().div_f32(other);
    Tensor::new(tensor, false)
  }
  

  // Assignment operations 
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

  fn pow_f32_assign(&mut self, other: f32) {
    self.tensor_mut().pow_f32_assign(other);
  }

  fn greater_than(&self, other: &Self, make_binary: bool) -> Self {
    let tensor = self.tensor().greater_than(other.tensor(), make_binary);
    Tensor::new(tensor, false)
  }

  fn less_than(&self, other: &Self, make_binary: bool) -> Self {
    let tensor = self.tensor().less_than(other.tensor(), make_binary);
    Tensor::new(tensor, false)
  }

  fn abs(&self) -> Self {
    let tensor = self.tensor().abs();
    Tensor::new(tensor, false)
  }

  fn abs_assign(&mut self) {
    self.tensor_mut().abs_assign();
  }

  fn apply<F>(&self, op: F) -> Self
  where
    F: Fn(f32) -> f32,
  {
    let tensor = self.tensor().apply(op);
    Tensor::new(tensor, false)
  }
  
  fn apply_assign<F>(&mut self, op: F)
  where
    F: Fn(f32) -> f32,
  {
    self.tensor_mut().apply_assign(op);
  }



}

impl BLASTensorOps for Tensor {
  fn matmul(&self, other: &Self, trans_a: bool, trans_b: bool) -> Self {
    let tensor = self.tensor().matmul(other.tensor(), trans_a, trans_b);
    
    let requires_grad = *self.requires_grad() || *other.requires_grad();
    let mut result = Tensor::new(tensor, requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(MatMulGrad::new(
        self,
        other,
        &result,
        trans_a,
        trans_b
      ))));
    }
    
    result
  }

}

impl Add<&Tensor> for &Tensor {
  type Output = Tensor;
  fn add(self, rhs: &Tensor) -> Self::Output {
    self.add_tensor(rhs)
  }
}

impl Mul<&Tensor> for &Tensor {
  type Output = Tensor;
  fn mul(self, rhs: &Tensor) -> Self::Output {
    self.mul_tensor(rhs)
  }
}