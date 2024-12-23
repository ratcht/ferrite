use super::base::TensorStorage;  // Import from parent module's base.rs

use ndarray::{ArrayBase, Dimension};
use num_traits::cast::AsPrimitive;

pub trait TensorCreation {
  fn zeros(shape: Vec<usize>, requires_grad: Option<bool>) -> Self;
  fn ones(shape: Vec<usize>, requires_grad: Option<bool>) -> Self;
  fn from_ndarray<S, D, T>(data: &ArrayBase<S, D>, requires_grad: Option<bool>) -> Self
  where 
    S: ndarray::Data<Elem = T>,
    T: AsPrimitive<f32>,
    D: Dimension;
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
}