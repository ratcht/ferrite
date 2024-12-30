use std::cell::RefCell;

use super::base::TensorStorage;  // Import from parent module's base.rs

use ndarray::{ArrayBase, Dimension};
use num_traits::cast::AsPrimitive;

use rand::distributions::{Distribution, Uniform};


pub trait TensorCreation {
  fn zeros(shape: Vec<usize>, requires_grad: Option<bool>) -> Self;
  fn ones(shape: Vec<usize>, requires_grad: Option<bool>) -> Self;
  fn from_ndarray<S, D, T>(data: &ArrayBase<S, D>, requires_grad: Option<bool>) -> Self
  where 
    S: ndarray::Data<Elem = T>,
    T: AsPrimitive<f32>,
    D: Dimension;

  fn uniform(l_bound: f32, r_bound: f32, shape: Vec<usize>, requires_grad: Option<bool>) -> Self;
}

impl TensorCreation for TensorStorage {
  fn zeros(shape: Vec<usize>, _requires_grad: Option<bool>) -> Self {
    let size = shape.iter().product();
    let data = vec![0.0; size];
    TensorStorage::new(data, shape)
  }

  fn ones(shape: Vec<usize>, _requires_grad: Option<bool>) -> Self {
    let size = shape.iter().product();
    let data = vec![1.0; size];
    TensorStorage::new(data, shape)
  }

  fn from_ndarray<S, D, T>(data: &ArrayBase<S, D>, _requires_grad: Option<bool>) -> Self
  where 
    S: ndarray::Data<Elem = T>,
    T: AsPrimitive<f32>,
    D: Dimension,
  {
    let shape = data.shape().to_vec();
    let arr = data.mapv(|x| x.as_());
    let data = arr.iter().cloned().collect();
    TensorStorage::new(data, shape)
  }
  
  fn uniform(l_bound: f32, r_bound: f32, shape: Vec<usize>, _requires_grad: Option<bool>) -> Self {
    let uniform = Uniform::from(l_bound..r_bound); // Create a uniform distribution
    let mut rng = rand::thread_rng(); // Random number generator

    let data = (0..shape.iter().product())
      .map(|_| uniform.sample(&mut rng)) // Sample from the uniform distribution
      .collect();

    TensorStorage::new(data, shape)
  }
}