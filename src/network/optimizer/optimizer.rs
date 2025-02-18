use crate::tensor::*;

pub trait OptimizerTrait {
  fn step(&self);
}

