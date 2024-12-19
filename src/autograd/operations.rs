use super::{Graph, Value};  // Import Graph from parent module

pub trait Operations {
  fn add(&self, lhs: usize, rhs: usize) -> usize;
  fn mul(&self, lhs: usize, rhs: usize) -> usize;
  fn div(&self, lhs: usize, rhs: usize) -> usize;
  fn exp(&self, lhs: usize) -> usize;

}

impl Operations for Graph {
  fn add(&self, lhs: usize, rhs: usize) -> usize {
    let mut values = self.values.borrow_mut();
    let data = values[lhs].data + values[rhs].data;
    let idx = values.len();
    values.push(Value::new(data, idx, &[lhs, rhs], "+"));
    idx
  }


  fn mul(&self, lhs: usize, rhs: usize) -> usize {
    let mut values = self.values.borrow_mut();
    let data = values[lhs].data * values[rhs].data;
    let idx = values.len();
    values.push(Value::new(data, idx, &[lhs, rhs], "*"));
    idx
  }

  fn div(&self, lhs: usize, rhs: usize) -> usize {
    let mut values = self.values.borrow_mut();
    let data = values[lhs].data / values[rhs].data;
    let idx = values.len();
    values.push(Value::new(data, idx, &[lhs, rhs], "/"));
    idx
  }

  fn exp(&self, lhs: usize) -> usize {
    let mut values = self.values.borrow_mut();
    let data = values[lhs].data.exp();
    let idx = values.len();
    values.push(Value::new(data, idx, &[lhs], "exp"));
    idx
  }
}