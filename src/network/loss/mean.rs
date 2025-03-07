use super::loss::*;
use crate::tensor::*;

pub struct MSELoss {
  is_mean_reduction: bool
}

impl MSELoss {
  pub fn new(reduction: &str) -> Self {
    let is_mean_reduction = match reduction {
      "mean" => true,
      "sum" => false,
      _ => panic!("Reduction must be either 'mean' or 'sum'"),
    };

    Self{ is_mean_reduction }
  }
}

impl LossTrait for MSELoss {
  fn loss(&self, x: &Tensor, y: &Tensor) -> Tensor {
    let z_1 = x.sub_tensor(y);
    let z_2 = z_1.pow_f32(2.); 

    if self.is_mean_reduction {
      z_2.mean()
    } else {
      z_2.sum()
    }
  }
}


pub struct MAELoss {
  is_mean_reduction: bool
}

impl MAELoss {
  pub fn new(reduction: &str) -> Self {
    let is_mean_reduction = match reduction {
      "mean" => true,
      "sum" => false,
      _ => panic!("Reduction must be either 'mean' or 'sum'"),
    };

    Self{ is_mean_reduction }
  }
}

impl LossTrait for MAELoss {
  fn loss(&self, x: &Tensor, y: &Tensor) -> Tensor {
    let z_1 = x.sub_tensor(y);
    let z_2 = z_1.abs(); 

    if self.is_mean_reduction {
      z_2.mean()
    } else {
      z_2.sum()
    }
  }
}