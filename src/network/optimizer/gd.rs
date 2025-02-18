use super::optimizer::*;
use crate::tensor::*;
use std::{collections::HashMap, sync::{Arc, RwLock}};

pub struct SGD {
  model_params: HashMap<String, Arc<RwLock<Tensor>>>,
  lr: f32,
  momentum: f32
}

impl SGD {
  pub fn new(model_params: HashMap<String, Arc<RwLock<Tensor>>>, lr: f32, momentum: f32) -> Self {
    Self{ model_params, lr, momentum }
  }
}

impl OptimizerTrait for SGD {
  fn step(&self) {
    for (key, value) in self.model_params.iter() {
      let mut tensor = value.write().unwrap();

      let mut temp = tensor.grad().unwrap();
      let grad = temp.borrow();

      let mut storage = tensor.tensor_mut();
      
      storage.sub_tensor_assign(&(&*grad * self.lr));
    }
  }
}