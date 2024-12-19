use crate::autograd::scalar::{Graph, Value};


pub trait Backward {
  fn add_backward(&self, lhs: usize, grad: f32);
  fn mul_backward(&self, lhs: usize, rhs: usize, grad: f32);
  fn sin_backward(&self, lhs: usize, grad: f32);
  fn cos_backward(&self, lhs: usize, grad: f32);

}

impl Backward for Graph {

  fn add_backward(&self, lhs: usize, grad: f32) {
    let lhs_value = &self.values[lhs];

  
    lhs_value.grad.set(lhs_value.grad.get() + grad);

  } 
  
  fn mul_backward(&self, lhs: usize, rhs: usize, grad: f32) {
    let lhs_value = &self.values[lhs];
    let rhs_value = &self.values[rhs];

    lhs_value.grad.set(lhs_value.grad.get() + rhs_value.data * grad);
    rhs_value.grad.set(rhs_value.grad.get() + lhs_value.data * grad);
  } 

  fn sin_backward(&self, lhs: usize, grad: f32) {
    let lhs_value = &self.values[lhs];

    lhs_value.grad.set(lhs_value.grad.get() + f32::cos(lhs_value.data) * grad);
  } 

  fn cos_backward(&self, lhs: usize, grad: f32) {
    let lhs_value = &self.values[lhs];

    lhs_value.grad.set(lhs_value.grad.get() + (-f32::sin(lhs_value.data)) * grad);
  } 

}

