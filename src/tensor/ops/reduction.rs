use std::rc::Rc;

use crate::{DeviceStorage, MeanGrad, ProductGrad, Storage, SumGrad, Tensor};

pub trait ReductionOps {
  fn sum(&self) -> Self;
  fn product(&self) -> Self;
  fn mean(&self) -> Self;

  // add other ops here
}

macro_rules! match_storage {
  // Binary operation with two storage arguments
  (binary $self:expr, $method:ident, $other:expr $(, $args:expr)*) => {
    match ($self, $other) {
      (Storage::Cpu(cpu_self), Storage::Cpu(cpu_other)) => {
        Storage::Cpu(cpu_self.$method(cpu_other $(, $args)*))
      }
      _ => unimplemented!("Cross-device operations not supported"),
    }
  };

  // Unary operation with single storage argument
  (unary $self:expr, $method:ident $(, $args:expr)*) => {
    match $self {
      Storage::Cpu(cpu) => Storage::Cpu(cpu.$method($($args),*)),
      _ => unimplemented!("Device not supported"),
    }
  };
}

impl ReductionOps for Storage {
  fn sum(&self) -> Self {
    match_storage!(unary self, sum)
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