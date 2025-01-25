use crate::*;
use std::fmt;

impl fmt::Display for Storage {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Storage::Cpu(cpu) => cpu.fmt(f),
      _ => write!(f, "Tensor on unsupported device")
    }
  }
}

impl fmt::Debug for Storage {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Storage::Cpu(cpu) => cpu.fmt(f),
      _ => write!(f, "Tensor on unsupported device")
    }
  }
}

impl Display for Storage {
  fn print(&self) {
    match self {
      Storage::Cpu(cpu) => cpu.print(),
      _ => println!("Tensor on unsupported device")
    }
  }

  fn print_data_recursive<'a>(data: &'a [f32], shape: &'a [usize], stride: &'a [usize]) -> String {
    unimplemented!()
  }

  fn print_data(&self) {
    match self {
      Storage::Cpu(cpu) => cpu.print_data(),
      _ => println!("Tensor on unsupported device")
    }
  }
}