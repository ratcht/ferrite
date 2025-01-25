use std::rc::Rc;

use crate::*;

pub trait BlasOps {
  fn matmul(&self, other: &Self, trans_a: bool, trans_b: bool) -> Self;
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

macro_rules! match_storage_assign {
  // Binary operation with two storage arguments
  (binary $self:expr, $method:ident, $other:expr $(, $args:expr)*) => {
    match ($self, $other) {
      (Storage::Cpu(cpu_self), Storage::Cpu(cpu_other)) => {
        cpu_self.$method(cpu_other $(, $args)*)
      }
      _ => unimplemented!("Cross-device operations not supported"),
    }
  };

  // Unary operation with single storage argument
  (unary $self:expr, $method:ident $(, $args:expr)*) => {
    match $self {
      Storage::Cpu(cpu) => cpu.$method($($args)*),
      _ => unimplemented!("Device not supported"),
    }
  };
}

impl BlasOps for Storage {
  fn matmul(&self, other: &Self, trans_a: bool, trans_b: bool) -> Self {
    match_storage!(binary self, matmul, other, trans_a, trans_b)
  }
}

impl BlasOps for Tensor {
  fn matmul(&self, other: &Self, trans_a: bool, trans_b: bool) -> Self {
    let tensor = self.tensor().matmul(other.tensor(), trans_a, trans_b);
    
    let requires_grad = *self.requires_grad() || *other.requires_grad();
    let mut result = Tensor::new(tensor, self.device(), requires_grad);
    
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