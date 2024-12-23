use crate::tensor_storage::*;
use super::base::*;

impl TensorCreation for Tensor {
  fn zeros(shape: Vec<usize>, requires_grad: Option<bool>) -> Self {
    let tensor = TensorStorage::zeros(shape, None);
    Tensor::new(tensor, requires_grad.unwrap(), None, None)
  }

  fn ones(shape: Vec<usize>, requires_grad: Option<bool>) -> Self {
    let tensor = TensorStorage::ones(shape, None);
    Tensor::new(tensor, requires_grad.unwrap(), None, None)
  }

  fn from_ndarray<S, D, T>(data: &ndarray::ArrayBase<S, D>, requires_grad: Option<bool>) -> Self
  where 
    S: ndarray::Data<Elem = T>,
    T: num_traits::AsPrimitive<f32>,
    D: ndarray::Dimension {
    let tensor = TensorStorage::from_ndarray(data, None);
    Tensor::new(tensor, requires_grad.unwrap(), None, None)
  }
}