use std::rc::Rc;
use crate::{DeviceStorage, MeanGrad, ProductGrad, Storage, SumGrad, Tensor, match_storage, match_storage_assign};

pub trait ReductionOps {
  fn sum(&self) -> Self;
  fn sum_axis(&self, axis: usize) -> Self;
  fn product(&self) -> Self;
  fn mean(&self) -> Self;
}


impl ReductionOps for Storage {
  fn sum(&self) -> Self {
    match_storage!(unary self, sum)
  }

  fn sum_axis(&self, axis: usize) -> Self {
    match_storage!(unary self, sum_axis, axis)
  }

  fn product(&self) -> Self {
    match_storage!(unary self, product)
  }

  fn mean(&self) -> Self {
    match_storage!(unary self, mean)
  }
}

impl ReductionOps for Tensor {

  fn sum(&self) -> Self {
    let tensor = self.tensor().sum();
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(SumGrad::new(self, &result))));
    }
    
    result
  }

  fn sum_axis(&self, axis: usize) -> Self {
    let tensor = self.tensor().sum_axis(axis);
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    result
  }

  fn mean(&self) -> Self {
    let tensor = self.tensor().mean();
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(MeanGrad::new(self, &result))));
    }
    
    result
  }

  fn product(&self) -> Self {
    let tensor = self.tensor().product();
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(ProductGrad::new(self, &result))));
    }
    
    result
  }

}