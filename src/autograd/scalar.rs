// scalar.rs

use std::cell::{Cell, RefCell};
use std::fmt;

#[derive(Debug)]
pub struct Value {
  pub data: f32,
  pub idx: usize,
  pub prev: Vec<usize>,
  pub op: String,
  pub grad: Cell<f32>,
}

impl Value {
  pub fn new(data: f32, idx: usize, prev: &[usize], op: &str) -> Self {
    Value {
      data,
      idx,
      prev: prev.to_vec(),
      op: String::from(op),
      grad: Cell::new(0.),
    }
  }
}

impl fmt::Display for Value {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Value(data={}, grad={})", self.data, self.grad.get())
  }
}


pub struct Graph {
  pub values: RefCell<Vec<Value>>,
}

impl Graph {
  pub fn new() -> Self {
    Graph {
      values: RefCell::new(Vec::new()),
    }
  }

  pub fn scalar(&self, data: f32) -> usize {
    let mut values = self.values.borrow_mut();
    let idx = values.len();
    values.push(Value::new(data, idx, &[], ""));
    idx
  }

  

  pub fn get_value(&self, id: usize) -> f32 {
    self.values.borrow()[id].data
  }

  pub fn get_op(&self, id: usize) -> String {
    self.values.borrow()[id].op.clone()
  }

  pub fn get_grad(&self, id: usize) -> f32 {
    self.values.borrow()[id].grad.get()
  }
}