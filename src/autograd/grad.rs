use crate::tensor::*;

#[macro_export]
macro_rules! reduce_grad {
  ($grad:expr, $shape:expr) => {{
    let mut reduced_grad = $grad.clone();
    for (dim, (grad_size, shape_size)) in $grad.shape().iter().zip($shape.iter()).enumerate() {
      if shape_size == &1 && grad_size != &1 {
        let mut sum_dims = vec![false; $grad.shape().len()];
        sum_dims[dim] = true;
        reduced_grad = reduced_grad.sum_dim(&sum_dims);
      }
    }
    reduced_grad
  }};
}

pub trait GradientFunction: std::fmt::Debug {
  fn backward(&self);
  fn prev(&self) -> Vec<&Tensor>;
}