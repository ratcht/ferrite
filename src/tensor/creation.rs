use super::base::Tensor;  // Import from parent module's base.rs

use ndarray::{ArrayBase, Dimension};
use num_traits::cast::AsPrimitive;

pub trait TensorCreation{
  fn zeros(shape: Vec<usize>, requires_grad: bool) -> Self;
  fn ones(shape: Vec<usize>, requires_grad: bool) -> Self;
  fn from_ndarray<S, D, T>(data: &ArrayBase<S, D>, requires_grad: bool) -> Self
  where 
    S: ndarray::Data<Elem = T>,
    T: AsPrimitive<f32>,
    D: Dimension;
}

impl TensorCreation for Tensor {
  fn zeros(shape: Vec<usize>, requires_grad: bool) -> Self {
    let size = shape.iter().product();
    let data = vec![0.0; size];
    Tensor::new(data, shape, requires_grad)
  }

  fn ones(shape: Vec<usize>, requires_grad: bool) -> Self {
    let size = shape.iter().product();
    let data = vec![1.0; size];
    Tensor::new(data, shape, requires_grad)
  }

  fn from_ndarray<S, D, T>(data: &ArrayBase<S, D>, requires_grad: bool) -> Self
  where 
    S: ndarray::Data<Elem = T> ,
    T: AsPrimitive<f32>,
    D: Dimension,
  {
    let shape = data.shape().to_vec();
    let arr = data.mapv(|x| x.as_());
    let data = arr.flatten().to_vec();
    Tensor::new(data, shape, requires_grad)
  }
}