use crate::*;
use ndarray::prelude::*;


impl ReductionOps for CpuStorage {
  fn sum(&self) -> Self {
    let data: f32 = self.data().borrow().iter().sum();
    CpuStorage::from_ndarray(&array![data], None, None)
  }

  fn product(&self) -> Self {
    let data: f32 = self.data().borrow().iter().sum();
    CpuStorage::from_ndarray(&array![data], None, None)
  }

  fn mean(&self) -> Self {
    let data: f32 = self.data().borrow().iter().sum::<f32>() / self.data().borrow().len() as f32;
    CpuStorage::from_ndarray(&array![data], None, None)
  }
}