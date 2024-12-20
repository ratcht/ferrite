use super::base::Tensor;  // Import from parent module's base.rs

use std::fmt;

pub trait Display {
  fn print(&self);
  fn print_data_recursive<'a>(data: &'a [f32], shape: &'a [usize], stride: &'a [usize]) -> String;
  fn print_data(&self);
}

impl fmt::Display for Tensor {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", Self::print_data_recursive(self.data(), self.shape(), self.stride()))
  }
}

impl Display for Tensor {
  fn print(&self) {
    println!("Data: {:?}", self.data());
    println!("Shape: {:?}", self.shape());
    println!("Strides: {:?}", self.stride());
    println!("Requires grad: {:?}", self.requires_grad());
  }

  fn print_data_recursive<'a>(data: &'a [f32], shape: &'a [usize], stride: &'a [usize]) -> String {
    let mut res = String::new();
    res += "[";
    if shape.len() == 1 {
      for (i, value) in data.iter().enumerate() {
        res += &format!("{}", value);

        if i < data.len() - 1 {
          res += ", ";
        }
        
      }
    } else {
      for i in 0..shape[0] {
        let sub_data = &data[i*stride[0]..(i+1)*stride[0]];
        let sub_res = Self::print_data_recursive(sub_data, &shape[1..], &stride[1..]);
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
    let res = Self::print_data_recursive(self.data(), self.shape(), self.stride());
    println!("{}", res);
  }
}