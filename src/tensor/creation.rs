use crate::tensor_storage::*;
use super::base::*;
use ndarray;
use num_traits;

impl TensorCreation for Tensor {
  fn zeros(shape: Vec<usize>, requires_grad: Option<bool>) -> Self {
    let tensor = TensorStorage::zeros(shape, None);
    let requires_grad = requires_grad.unwrap_or(false);
    Tensor::new(tensor, requires_grad)
  }

  fn ones(shape: Vec<usize>, requires_grad: Option<bool>) -> Self {
    let tensor = TensorStorage::ones(shape, None);
    let requires_grad = requires_grad.unwrap_or(false);
    Tensor::new(tensor, requires_grad)
  }

  fn from_ndarray<S, D, T>(data: &ndarray::ArrayBase<S, D>, requires_grad: Option<bool>) -> Self
  where 
    S: ndarray::Data<Elem = T>,
    T: num_traits::AsPrimitive<f32>,
    D: ndarray::Dimension 
  {
    let tensor = TensorStorage::from_ndarray(data, None);
    let requires_grad = requires_grad.unwrap_or(false);
    Tensor::new(tensor, requires_grad)
  }

  fn uniform(l_bound: f32, r_bound: f32, shape: Vec<usize>, requires_grad: Option<bool>) -> Self {
    let tensor = TensorStorage::uniform(l_bound, r_bound, shape, None);
    let requires_grad = requires_grad.unwrap_or(false);
    Tensor::new(tensor, requires_grad)
  }
}