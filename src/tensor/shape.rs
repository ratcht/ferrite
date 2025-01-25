use std::any::Any;
use std::rc::Rc;

use crate::*;
use crate::autograd::PermuteGrad;


impl Tensor {
  fn reshape(&mut self, new_shape: Vec<usize>) {
    self.tensor_mut().set_shape(new_shape);
  }

  fn transpose(&self) -> Self {
    // Transpose by swapping dimensions & strides

    let new_storage = self.tensor().transpose();
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(new_storage, self.device(), *self.requires_grad());
    
    if requires_grad {
      result.set_grad_fn(Some(Rc::new(PermuteGrad::new(self, &result))));
    }
    
    result
  }

  fn permute(&mut self, dims: &[usize]) {
    self.tensor_mut().permute(dims);
  }

  fn flatten(&mut self) {
    self.tensor_mut().flatten();
  }

  fn squeeze(&mut self) {
    self.tensor_mut().squeeze();
  } 

  fn unsqueeze(&mut self, dim: usize) {
    self.tensor_mut().unsqueeze(dim);
  }

  fn broadcast(&self, new_shape: &[usize]) -> Self {
    let new_storage = self.tensor().broadcast(new_shape);
    
    // When broadcasting, we need to maintain the gradient tracking
    let requires_grad = *self.requires_grad();
    let mut result = Tensor::new(new_storage, self.device(), *self.requires_grad());
    
    // If original tensor requires gradient, the broadcasted tensor
    // should have the same gradient function
    if requires_grad {
      result.set_grad_fn(self.grad_fn());
    }
    
    result
  }

  fn compute_broadcast_shape(&self, target_shape: &[usize]) -> Vec<usize> {
    self.tensor().compute_broadcast_shape(target_shape)
  }

  fn compute_broadcast_strides(&self, target_shape: &[usize]) -> Vec<usize> {
    self.tensor().compute_broadcast_strides(target_shape)
  }

  fn pad_shape(&self, target_rank: usize) -> Vec<usize> {
    self.tensor().pad_shape(target_rank)
  }


  fn broadcast_tensors(a: &Self, b: &Self) -> (Self, Self) {
    if (a.device() != b.device()) { panic!("Tensors not on same device!") }

    let (broadcast_a, broadcast_b) = Storage::broadcast_tensors(a.tensor(), b.tensor());

    let mut tensor_a = Tensor::new(broadcast_a, a.device(), *a.requires_grad());
    let mut tensor_b = Tensor::new(broadcast_b, b.device(), *b.requires_grad());

    if *a.requires_grad() {
      tensor_a.set_grad_fn(a.grad_fn());
    }
    if *b.requires_grad() {
      tensor_b.set_grad_fn(b.grad_fn());
    }

    (tensor_a, tensor_b)
  }
}