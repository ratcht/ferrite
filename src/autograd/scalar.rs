use std::cell::Cell;
use std::rc::Rc;

use std::ops;
use std::fmt;

use crate::autograd::backward;

use super::backward::Backward;

// use super::backward::Backward;


pub struct Graph {
  pub values: Vec<Value> // Graph owns these values
}

impl Graph {
  pub fn new() -> Self {
    Graph {
      values: vec![]
    }
  }


  // create scalar
  pub fn scalar(&mut self, data: f32) -> usize {
    let id = self.values.len();
    let value = Value::new(data, id, &[], "");
    self.values.push(value);
    id
  }

  pub fn add(&mut self, lhs: usize, rhs: usize) -> usize {
    let data = self.values[lhs].data + self.values[rhs].data;
    let id = self.values.len();
    let value = Value::new(data, id, &[lhs, rhs], "+");
    self.values.push(value);
    id
  }

  pub fn mul(&mut self, lhs: usize, rhs: usize) -> usize {
    let data = self.values[lhs].data * self.values[rhs].data;
    let id = self.values.len();
    let value = Value::new(data, id, &[lhs, rhs], "*");
    self.values.push(value);
    id
  }

  pub fn backward(&self, id: usize) {
    self.values[id].grad.set(1.0);
    self.backward_from(id);
  }

  fn backward_from(&self, id: usize) {
    let value = &self.values[id];
    let grad = value.grad.get();



    
    match value.op.as_str() {
      "+" => {
        let lhs = value.prev[0];
        let rhs = value.prev[1];
    
        self.add_backward(lhs, grad);
        self.add_backward(rhs, grad);
      },
      "*" => {
        let lhs = value.prev[0];
        let rhs = value.prev[1];
            
        self.mul_backward(lhs, rhs, grad);
      },
      _ => {}
    }

    // Recursively backpropagate
    for &prev_id in &value.prev {
      self.backward_from(prev_id);
    }
  }

  pub fn get(&self, id: usize) -> &Value {
    &self.values[id]
  }
}


#[derive(Debug)]
pub struct Value {
  pub data: f32,
  idx: usize,
  prev: Vec<usize>, // stores index
  op: String,
  pub grad: Cell<f32>, // only cell needs to be interior mutable
}

impl Value {

  pub fn new(data: f32, idx: usize, prev: &[usize], op: &str) -> Self {
    Value {
      data: data,
      idx: idx,
      prev: prev.to_vec(),
      op: String::from(op),
      grad: Cell::new(0.),
    }
  }

}


impl fmt::Display for &Value {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Data: {}. Children: {:?}", self.data, self.prev)
  }
}