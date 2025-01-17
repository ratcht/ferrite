use super::{Graph, Scalar, Value};  // Import Graph from parent module
// scalar.rs
use std::cell::{RefCell};
use std::collections::HashSet;

pub trait Backward {
  fn add_backward(&self, lhs: usize, rhs: usize, grad: f32);
  fn sub_backward(&self, lhs: usize, rhs: usize, grad: f32);
  fn mul_backward(&self, lhs: usize, rhs: usize, grad: f32);
  fn div_backward(&self, lhs: usize, rhs: usize, grad: f32);
  fn exp_backward(&self, lhs: usize, grad: f32);
  fn pow_backward(&self, lhs: usize, rhs: usize, grad: f32);
  fn backward(&self, value: Value);
  fn backward_from(&self, idx: usize);

}

impl Backward for Graph {
  // d(a + b)/da = 1, d(a + b)/db = 1
  fn add_backward(&self, lhs: usize, rhs: usize, grad: f32) {
    let scalars = self.scalars.borrow();
    let lhs_scalar = &scalars[lhs];
    let rhs_scalar = &scalars[rhs];
    lhs_scalar.grad.set(lhs_scalar.grad.get() + grad);
    rhs_scalar.grad.set(rhs_scalar.grad.get() + grad);
  }

  // d(a + b)/da = 1, d(a + b)/db = 1
  fn sub_backward(&self, lhs: usize, rhs: usize, grad: f32) {
    let scalars = self.scalars.borrow();
    let lhs_scalar = &scalars[lhs];
    let rhs_scalar = &scalars[rhs];
    lhs_scalar.grad.set(lhs_scalar.grad.get() + grad);
    rhs_scalar.grad.set(rhs_scalar.grad.get() - grad);
  }

  // d(a * b)/da = b, d(a * b)/db = a
  fn mul_backward(&self, lhs: usize, rhs: usize, grad: f32) {
    let scalars = self.scalars.borrow();
    let lhs_scalar = &scalars[lhs];
    let rhs_scalar = &scalars[rhs];
    lhs_scalar.grad.set(lhs_scalar.grad.get() + rhs_scalar.data * grad);
    rhs_scalar.grad.set(rhs_scalar.grad.get() + lhs_scalar.data * grad);
  }

  // d(a/b)/da = 1/b, d(a/b)/db = (-a/b**2)
  fn div_backward(&self, lhs: usize, rhs: usize, grad: f32) {
    let scalars = self.scalars.borrow();
    let lhs_scalar = &scalars[lhs];
    let rhs_scalar = &scalars[rhs];
    lhs_scalar.grad.set(lhs_scalar.grad.get() + (1./rhs_scalar.data) * grad);
    rhs_scalar.grad.set(rhs_scalar.grad.get() + (-lhs_scalar.data/(rhs_scalar.data*rhs_scalar.data)) * grad);
  }


  // d(e^x)/dx = e^x
  fn exp_backward(&self, lhs: usize, grad: f32) {
    let scalars = self.scalars.borrow();
    let lhs_scalar = &scalars[lhs];
    let exp_x = f32::exp(lhs_scalar.data);
    lhs_scalar.grad.set(lhs_scalar.grad.get() + grad * exp_x);
  }

  // d(x^a)/dx = ax^(a-1), d(x^a)/da = x^a * lnx
  fn pow_backward(&self, lhs: usize, rhs: usize, grad: f32) {
    let scalars = self.scalars.borrow();
    let lhs_scalar = &scalars[lhs];
    let rhs_scalar = &scalars[rhs];
    lhs_scalar.grad.set(lhs_scalar.grad.get() + (rhs_scalar.data * (f32::powf(lhs_scalar.data, rhs_scalar.data - 1.))) * grad);
    rhs_scalar.grad.set(rhs_scalar.grad.get() + (f32::powf(lhs_scalar.data, rhs_scalar.data) * f32::ln(lhs_scalar.data)) * grad);
  }


  fn backward(&self, value: Value) {
    let id = value.idx;
    self.scalars.borrow()[id].grad.set(1.0);

    // Topo sort
    let mut topo = vec![];
    let mut visited = HashSet::new();

    fn build_topo(v: usize, scalars: &RefCell<Vec<Scalar>>, visited: &mut HashSet<usize>, topo: &mut Vec<usize>) {
      if !visited.contains(&v) {
        visited.insert(v);

        let children = {
          let scalars = scalars.borrow();
          scalars[v].prev.clone()
        };

        for &child in &children {
          build_topo(child, scalars, visited, topo);
        }

        topo.push(v);
      }
    }

    build_topo(id, &self.scalars, &mut visited, &mut topo);
    
    for &idx in topo.iter().rev(){
      self.backward_from(idx);
    }

   // 
  }

  fn backward_from(&self, idx: usize) {
    let scalars = self.scalars.borrow();
    let scalar = &scalars[idx];

    if !scalar.requires_grad {
      return;
    }

    let grad = scalar.grad.get();

    match scalar.op.as_str() {
      "+" => {
        let lhs = scalar.prev[0];
        let rhs = scalar.prev[1];
        self.add_backward(lhs, rhs, grad);
      },
      "-" => {
        let lhs = scalar.prev[0];
        let rhs = scalar.prev[1];
        self.sub_backward(lhs, rhs, grad);
      },
      "*" => {
        let lhs = scalar.prev[0];
        let rhs = scalar.prev[1];
        self.mul_backward(lhs, rhs, grad);
      },
      "/" => {
        let lhs = scalar.prev[0];
        let rhs = scalar.prev[1];
        self.div_backward(lhs, rhs, grad);
      },
      "exp" => {
        let lhs = scalar.prev[0];
        self.exp_backward(lhs, grad);
      },
      "pow" => {
        let lhs = scalar.prev[0];
        let rhs = scalar.prev[1];
        self.pow_backward(lhs, rhs, grad);
      },
      
      _ => {}
    }
  }

}

