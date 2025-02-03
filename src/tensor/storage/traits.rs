use std::rc::Rc;
use std::cell::RefCell;
use ndarray::{ArrayBase, Dimension};
use num_traits::cast::AsPrimitive;

use crate::{CpuStorage};

// Device types
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum Device {
  Cpu,
  Cuda,
  Mps,
}

pub trait DeviceStorageStatic : DeviceStorage {
  fn new(data: Vec<f32>, shape: Vec<usize>) -> Self;

  fn new_with_stride(data: Vec<f32>, shape: Vec<usize>, stride: Vec<usize>) -> Self;

  fn create(data: Rc<RefCell<Vec<f32>>>, shape: Vec<usize>, stride: Vec<usize>) -> Self;

  fn compute_strides(shape: &Vec<usize>) -> Vec<usize>;
}

pub trait DeviceStorageCreation : DeviceStorage {
  fn zeros(shape: Vec<usize>, device: Option<Device>, requires_grad: Option<bool>) -> Self;
  fn ones(shape: Vec<usize>, device: Option<Device>, requires_grad: Option<bool>) -> Self;
  fn from_ndarray<S, D, T>(data: &ArrayBase<S, D>, device: Option<Device>, requires_grad: Option<bool>) -> Self
  where 
    S: ndarray::Data<Elem = T>,
    T: AsPrimitive<f32>,
    D: Dimension;

  fn uniform(l_bound: f32, r_bound: f32, shape: Vec<usize>, device: Option<Device>, requires_grad: Option<bool>) -> Self;
}


pub trait DeviceStorage  {
  fn view(&self, new_shape: Vec<usize>) -> Self where Self: Sized;

  fn data(&self) -> Rc<RefCell<Vec<f32>>>;

  fn data_mut(&self) -> std::cell::RefMut<Vec<f32>>;

  fn set_data(&mut self, data: Vec<f32>);

  fn shape(&self) -> &Vec<usize>;

  fn set_shape(&mut self, shape: Vec<usize>);

  fn stride(&self) -> &Vec<usize>;

  fn set_stride(&mut self, stride: Vec<usize>);

  fn get(&self, indices: &[usize]) -> f32;

  fn set(&mut self, indices: &[usize], value: f32);

  fn make_contiguous(&self) -> (Vec<f32>, i32);

  fn is_contiguous(&self) -> bool;
}


#[derive(Clone)]
pub enum Storage {
  Cpu(CpuStorage),
}
