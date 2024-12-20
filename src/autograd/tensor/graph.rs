// scalar.rs

use std::cell::{Cell, RefCell};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Value<'a> {
  pub idx: usize,
  pub graph: &'a Graph,
}

impl<'a> Value<'a> {
  pub fn new(idx: usize, graph: &'a Graph) -> Self {
    Value { 
      idx,
      graph,
    }
  }

  pub fn grad(&self) -> f32 {
    self.graph.scalars.borrow()[self.idx].grad.get()
  }

  pub fn value(&self) -> f32 {
    self.graph.scalars.borrow()[self.idx].data
  }
}

#[derive(Debug)]
pub struct Scalar {
  pub data: f32,
  pub idx: usize,
  pub prev: Vec<usize>,
  pub op: String,
  pub grad: Cell<f32>,
  pub requires_grad: bool
}

impl Scalar {
  pub fn new(data: f32, idx: usize, prev: &[usize], op: &str, requires_grad: bool) -> Self {
    Scalar {
      data,
      idx,
      prev: prev.to_vec(),
      op: String::from(op),
      grad: Cell::new(0.),
      requires_grad: requires_grad,
    }
  }
}

impl fmt::Display for Scalar {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Scalar(data={}, grad={})", self.data, self.grad.get())
  }
}


#[derive(Debug)]
pub struct Graph {
  pub scalars: RefCell<Vec<Scalar>>,
}

impl Graph {
  pub fn new() -> Self {
    Graph {
      scalars: RefCell::new(Vec::new()),
    }
  }

  pub fn scalar(&self, data: f32, requires_grad: bool) -> Value {
    let mut scalars = self.scalars.borrow_mut();
    let idx = scalars.len();
    scalars.push(Scalar::new(data, idx, &[], "", requires_grad));
    
    Value {
      idx: idx,
      graph: self,
    }
  }

  pub fn get_value(&self, value: &Value) -> f32 {
    let id = value.idx;
    self.scalars.borrow()[id].data
  }

  pub fn get_op(&self, value: &Value) -> String {
    let id = value.idx;
    self.scalars.borrow()[id].op.clone()
  }

  pub fn get_grad(&self, value: &Value) -> f32 {
    let id = value.idx;
    self.scalars.borrow()[id].grad.get()
  }
}