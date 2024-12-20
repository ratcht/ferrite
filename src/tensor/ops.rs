use super::base::Tensor;  // Import from parent module's base.rs

pub trait TensorOps {
  fn add(&self, other: &Self) -> Self;
  fn mul(&self, other: &Self) -> Self;
  fn add_f32(&self, other: f32) -> Self;
  fn mul_f32(&self, other: f32) -> Self;

  fn matmul(&self, other: &Self) -> Self;
}

impl TensorOps for Tensor {
  fn add(&self, other: &Self) -> Self {
    // For now, no broadcasting
    if self.shape() != other.shape() { panic!("Tensor shapes don't match!") }

    let data = self.data().iter()
    .zip(other.data().iter())
    .map(|(a, b)| a + b)
    .collect();

    Tensor::new(data, self.shape().clone(), self.requires_grad() || other.requires_grad())
  }

  fn mul(&self, other: &Self) -> Self {
    // For now, no broadcasting
    if self.shape() != other.shape() { panic!("Tensor shapes don't match!") }

    let data = self.data().iter()
    .zip(other.data().iter())
    .map(|(a, b)| a * b)
    .collect();

    Tensor::new(data, self.shape().clone(), self.requires_grad() || other.requires_grad())
  }

  fn add_f32(&self, other: f32) -> Self {
    let data = self.data().iter()
    .map(|a| a + other)
    .collect();

    Tensor::new(data, self.shape().clone(), self.requires_grad())
  }

  fn mul_f32(&self, other: f32) -> Self {
    let data = self.data().iter()
    .map(|a| a * other)
    .collect();

    Tensor::new(data, self.shape().clone(), self.requires_grad())
  }
  
  fn matmul(&self, other: &Self) -> Self {
    if self.shape().len() != 2 { panic!("Can't Matmul on non-matrices"); }
    if self.shape()[1] != other.shape()[0] { panic!("Array2D dimensions do not match for multiplication."); }

    let mut data = vec![1.0; self.shape()[0] * other.shape()[1]];

    for col in 0..other.shape()[1] {
      for row in 0..self.shape()[0] {
        let mut dot = 0;
        for i in 0..self.shape()[1] {
          dot += self.data()[row * self.shape()[1] + i] * other.data()[i * other.shape()[1] + col];
        }
        data[row*other.shape()[1]+ col ] = dot;
      }
    }

    Tensor::new(data, vec![self.shape()[0], other.shape()[1]], self.requires_grad() || other.requires_grad())

  }
}


