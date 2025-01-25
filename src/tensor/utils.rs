use crate::*;
use super::base::*;
use std::fmt;

impl fmt::Display for Tensor {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let storage = self.tensor();
    write!(f, "{}", storage)
  }
}

impl fmt::Debug for Tensor {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
   let storage = self.tensor();
   write!(f, "{:?}", storage)
  }
}

impl Display for Tensor {
  fn print(&self) {
    self.tensor();
  }

  fn print_data_recursive<'a>(data: &'a [f32], shape: &'a [usize], stride: &'a [usize]) -> String {
    unimplemented!()
  }

  fn print_data(&self) {
    self.tensor().print_data();
  }
}