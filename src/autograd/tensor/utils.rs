use crate::tensor_storage::*;
use super::base::*;
use std::fmt;

impl fmt::Display for Tensor {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.tensor().fmt(f)
  }
}

impl fmt::Debug for Tensor {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.tensor().fmt(f)
  }
}

impl Display for Tensor {
  fn print(&self) {
    self.tensor().print();
  }

  fn print_data_recursive<'a>(data: &'a [f32], shape: &'a [usize], stride: &'a [usize]) -> String {
    TensorStorage::print_data_recursive(data, shape, stride)
  }

  fn print_data(&self) {
    self.tensor().print_data();
  }
}