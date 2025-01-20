use super::base::TensorStorage;  // Import from parent module's base.rs

use std::fmt;

pub trait Display {
  fn print(&self);
  fn print_data_recursive<'a>(data: &'a [f32], shape: &'a [usize], stride: &'a [usize]) -> String;
  fn print_data(&self);
}

impl fmt::Display for TensorStorage {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", Self::print_data_recursive(&self.data().borrow(), self.shape(), self.stride()))
  }
}

impl fmt::Debug for TensorStorage {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", Self::print_data_recursive(&self.data().borrow(), self.shape(), self.stride()))
  }
}

impl Display for TensorStorage {
  fn print(&self) {
    println!("Data: {:?}", self.data());
    println!("Shape: {:?}", self.shape());
    println!("Strides: {:?}", self.stride());
  }

  fn print_data_recursive<'a>(data: &'a [f32], shape: &'a [usize], stride: &'a [usize]) -> String {
    let mut res = String::new();
    res += "[";
    if shape.len() == 1 {
      for i in 0..shape[0] {
        res += &format!("{}", data[i*stride[0]]);

        if i < shape[0] - 1 {
          res += ", ";
        }
        
      }
    } else {
      for i in 0..shape[0] {
        let start = i*stride[0];
        let sub_res = Self::print_data_recursive(&data[start..], &shape[1..], &stride[1..]);
        res += &sub_res;
        if i < shape[0] - 1 {
          res += ", ";
        }
      }
    }

    res += "]";
    res
  }

  fn print_data(&self) {
    let res = Self::print_data_recursive(&self.data().borrow(), self.shape(), self.stride());
    println!("{}", res);
  }
}