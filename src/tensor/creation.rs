use crate::*;
use ndarray;
use num_traits;



impl Tensor {
  pub fn zeros(shape: Vec<usize>, device: Device, requires_grad: Option<bool>) -> Self {
    let tensor = Storage::zeros(shape, Some(device), None);
    let requires_grad = requires_grad.unwrap_or(false);
    Tensor::new(tensor, device, requires_grad)
  }

  pub fn ones(shape: Vec<usize>, device: Device, requires_grad: Option<bool>) -> Self {
    let tensor = Storage::ones(shape, Some(device), None);
    let requires_grad = requires_grad.unwrap_or(false);
    Tensor::new(tensor, device, requires_grad)
  }

  pub fn from_ndarray<S, D, T>(data: &ndarray::ArrayBase<S, D>, device: Device, requires_grad: Option<bool>) -> Self
  where 
    S: ndarray::Data<Elem = T>,
    T: num_traits::AsPrimitive<f32>,
    D: ndarray::Dimension 
  {
    let tensor = Storage::from_ndarray(data, Some(device), None);
    let requires_grad = requires_grad.unwrap_or(false);
    Tensor::new(tensor, device, requires_grad)
  }

  pub fn uniform(l_bound: f32, r_bound: f32, shape: Vec<usize>, device: Device, requires_grad: Option<bool>) -> Self {
    let tensor = Storage::uniform(l_bound, r_bound, shape, Some(device), None);
    let requires_grad = requires_grad.unwrap_or(false);
    Tensor::new(tensor, device, requires_grad)
  }
}