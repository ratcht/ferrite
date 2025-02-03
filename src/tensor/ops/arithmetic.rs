use crate::*;
use std::{ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign}, rc::Rc};

pub trait ArithmeticOps {
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

  fn pow_f32(&self, other: f32) -> Self;
  fn pow_f32_assign(&mut self, other: f32);

  fn greater_than(&self, other: &Self, make_binary: bool) -> Self;
  fn greater_than_f32(&self, other: f32, make_binary: bool) -> Self;
  fn less_than(&self, other: &Self, make_binary: bool) -> Self;
  fn less_than_f32(&self, other: f32, make_binary: bool) -> Self;

  fn sign(&self) -> Self;
  fn abs(&self) -> Self;
  fn abs_assign(&mut self);
}

impl ArithmeticOps for Storage {
  fn add_tensor(&self, other: &Self) -> Self {
    match_storage!(binary self, add_tensor, other)
  }

  fn add_tensor_assign(&mut self, other: &Self) {
    match_storage_assign!(binary self, add_tensor_assign, other);
  }

  fn sub_tensor(&self, other: &Self) -> Self {
    match_storage!(binary self, sub_tensor, other)
  }

  fn sub_tensor_assign(&mut self, other: &Self) {
    match_storage_assign!(binary self, sub_tensor_assign, other);
  }

  fn mul_tensor(&self, other: &Self) -> Self {
    match_storage!(binary self, mul_tensor, other)
  }

  fn mul_tensor_assign(&mut self, other: &Self) {
    match_storage_assign!(binary self, mul_tensor_assign, other);
  }

  fn div_tensor(&self, other: &Self) -> Self {
    match_storage!(binary self, div_tensor, other)
  }

  fn div_tensor_assign(&mut self, other: &Self) {
    match_storage_assign!(binary self, div_tensor_assign, other);
  }

  fn add_f32(&self, other: f32) -> Self {
    match_storage!(unary self, add_f32, other)
  }

  fn add_f32_assign(&mut self, other: f32) {
    match_storage_assign!(unary self, add_f32_assign, other);
  }

  fn sub_f32(&self, other: f32) -> Self {
    match_storage!(unary self, sub_f32, other)
  }

  fn sub_f32_assign(&mut self, other: f32) {
    match_storage_assign!(unary self, sub_f32_assign, other);
  }

  fn mul_f32(&self, other: f32) -> Self {
    match_storage!(unary self, mul_f32, other)
  }

  fn mul_f32_assign(&mut self, other: f32) {
    match_storage_assign!(unary self, mul_f32_assign, other);
  }

  fn div_f32(&self, other: f32) -> Self {
    match_storage!(unary self, div_f32, other)
  }

  fn div_f32_assign(&mut self, other: f32) {
    match_storage_assign!(unary self, div_f32_assign, other);
  }

  fn pow_f32(&self, other: f32) -> Self {
    match_storage!(unary self, pow_f32, other)
  }

  fn pow_f32_assign(&mut self, other: f32) {
    match_storage_assign!(unary self, pow_f32_assign, other);
  }

  fn greater_than(&self, other: &Self, make_binary: bool) -> Self {
    match_storage!(binary self, greater_than, other, make_binary)
  }

  fn greater_than_f32(&self, other: f32, make_binary: bool) -> Self {
    match_storage!(unary self, greater_than_f32, other, make_binary)
  }

  fn less_than(&self, other: &Self, make_binary: bool) -> Self {
    match_storage!(binary self, less_than, other, make_binary)
  }

  fn less_than_f32(&self, other: f32, make_binary: bool) -> Self {
    match_storage!(unary self, less_than_f32, other, make_binary)
  }

  fn sign(&self) -> Self {
    match_storage!(unary self, sign)
  }

  fn abs(&self) -> Self {
    match_storage!(unary self, abs)
  }

  fn abs_assign(&mut self) {
    match_storage_assign!(unary self, abs_assign)
  }
}

impl ArithmeticOps for Tensor {
  fn add_tensor(&self, other: &Self) -> Self {
    // Compute the actual tensor addition
    let tensor = self.tensor().add_tensor(other.tensor());
    
    // Create result tensor
    let requires_grad = *self.requires_grad() || *other.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
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
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
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
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
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
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(DivGrad::new(
        self,
        other,
        &result
      ))));
    }
    
    result
  }


  fn pow_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().pow_f32(other);
    
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
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
    
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(AddF32Grad::new(
        self,
        other,
        &result
      ))));
    }
    
    result
  }
  fn sub_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().sub_f32(other);
    
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(SubF32Grad::new(
        self,
        other,
        &result
      ))));
    }
    
    result
  }

  fn mul_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().mul_f32(other);
    
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(MulF32Grad::new(
        self,
        other,
        &result
      ))));
    }
    
    result
  }

  fn div_f32(&self, other: f32) -> Self {
    let tensor = self.tensor().div_f32(other);
    
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(DivF32Grad::new(
        self,
        other,
        &result
      ))));
    }
    
    result
  }

  fn abs(&self) -> Self {
    let tensor = self.tensor().abs();
    
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(AbsGrad::new(
        self,
        &result
      ))));
    }
    
    result
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

  fn abs_assign(&mut self) {
    self.tensor_mut().abs_assign();
  }

  fn greater_than(&self, other: &Self, make_binary: bool) -> Self {
    let tensor = self.tensor().greater_than(other.tensor(), make_binary);
    Tensor::new(tensor, self.device(), false)
  }

  fn greater_than_f32(&self, other: f32, make_binary: bool) -> Self {
    let tensor = self.tensor().greater_than_f32(other, make_binary);
    Tensor::new(tensor, self.device(), false)
  }

  fn less_than(&self, other: &Self, make_binary: bool) -> Self {
    let tensor = self.tensor().less_than(other.tensor(), make_binary);
    Tensor::new(tensor, self.device(), false)
  }

  fn less_than_f32(&self, other: f32, make_binary: bool) -> Self {
    let tensor = self.tensor().less_than_f32(other, make_binary);
    Tensor::new(tensor, self.device(), false)
  }

  fn sign(&self) -> Self {
    let tensor = self.tensor().sign();
    Tensor::new(tensor, self.device(), false)
  }


}


macro_rules! impl_binary_ops {
  ($type:ty, $target:ty) => {
    impl Add<&$target> for &$type {
      type Output = $target;
      fn add(self, rhs: &$target) -> Self::Output {
        self.add_tensor(rhs)
      }
    }

    impl AddAssign<&$target> for $type {
      fn add_assign(&mut self, rhs: &$target) {
        self.add_tensor_assign(rhs)
      }
    }
    
    impl Sub<&$target> for &$type {
      type Output = $target;
      fn sub(self, rhs: &$target) -> Self::Output {
        self.sub_tensor(rhs)
      }
    }

    impl SubAssign<&$target> for $type {
      fn sub_assign(&mut self, rhs: &$target) {
        self.sub_tensor_assign(rhs)
      }
    }

    impl Mul<&$target> for &$type {
      type Output = $target;
      fn mul(self, rhs: &$target) -> Self::Output {
        self.mul_tensor(rhs)
      }
    }

    impl MulAssign<&$target> for $type {
      fn mul_assign(&mut self, rhs: &$target) {
        self.mul_tensor_assign(rhs)
      }
    }

    impl Div<&$target> for &$type {
      type Output = $target;
      fn div(self, rhs: &$target) -> Self::Output {
        self.div_tensor(rhs)
      }
    }

    impl DivAssign<&$target> for $type {
      fn div_assign(&mut self, rhs: &$target) {
        self.div_tensor_assign(rhs)
      }
    }
  }
}

macro_rules! impl_scalar_ops {
  ($type:ty) => {
    impl Add<f32> for &$type {
      type Output = $type;
      fn add(self, rhs: f32) -> Self::Output {
        self.add_f32(rhs)
      }
    }
    
    impl AddAssign<f32> for $type {
      fn add_assign(&mut self, rhs: f32) {
        self.add_f32_assign(rhs);
      }
    }

    impl Sub<f32> for &$type {
      type Output = $type;
      fn sub(self, rhs: f32) -> Self::Output {
        self.sub_f32(rhs)
      }
    }
    
    impl SubAssign<f32> for $type {
      fn sub_assign(&mut self, rhs: f32) {
        self.sub_f32_assign(rhs);
      }
    }

    impl Mul<f32> for &$type {
      type Output = $type;
      fn mul(self, rhs: f32) -> Self::Output {
        self.mul_f32(rhs)
      }
    }
    
    impl MulAssign<f32> for $type {
      fn mul_assign(&mut self, rhs: f32) {
        self.mul_f32_assign(rhs);
      }
    }

    impl Div<f32> for &$type {
      type Output = $type;
      fn div(self, rhs: f32) -> Self::Output {
        self.div_f32(rhs)
      }
    }
    
    impl DivAssign<f32> for $type {
      fn div_assign(&mut self, rhs: f32) {
        self.div_f32_assign(rhs);
      }
    }
      
    
  }
}

impl_binary_ops!(Tensor, Tensor);
impl_binary_ops!(Storage, Storage);

impl_scalar_ops!(Tensor);
impl_scalar_ops!(Storage);