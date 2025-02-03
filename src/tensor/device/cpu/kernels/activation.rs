use crate::*;

impl ActivationOps for CpuStorage {
  fn binary_step(&self) -> Self {
    self.apply(|x| if x < 0. { 0. } else { 1. })
  }

  fn sigmoid(&self) -> Self {
    self.apply(|x| 1./(1. + f32::exp(-x)))
  }

  fn tanh(&self) -> Self {
    self.apply(|x| (f32::exp(x) - f32::exp(-x))/(f32::exp(x) + f32::exp(-x)))
  }

  fn relu(&self) -> Self {
    self.apply(|x| f32::max(0., x))
  }

  fn leaky_relu(&self) -> Self {
    self.apply(|x| f32::max(0.1*x, x))
  }

  fn parametric_relu(&self, a: f32) -> Self {
    self.apply(|x| f32::max(a*x, a))
  }

  fn elu(&self, alpha: f32) -> Self {
    self.apply(|x| if x >= 0. {x} else {alpha * (f32::exp(x) - 1.)})
  }

  fn softmax(&self) -> Self {
    // calculate sum of exp
    let exp_sum: f32 = self.data().borrow().iter().map(|&x| f32::exp(x)).sum();
    self.apply(|x| f32::exp(x) / exp_sum)
  }

  fn swish(&self) -> Self {
    self.apply(|x| x * (1./(1. + f32::exp(-x))))
  }
}