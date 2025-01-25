use ndarray::prelude::*;
use ferrite::prelude::*;
use Loss::Loss;



fn main() {
  let x = Tensor::from_ndarray(&array![[2., 2., 3.], [4., 4., 6.]], Device::Cpu, Some(true));
  println!("x: {}", x);

  let y = Tensor::from_ndarray(&array![[3., 3., 1.], [6., 5., 6.]], Device::Cpu, Some(true));
  println!("y: {}", y);

  let mut z = Tensor::from_ndarray(&array![[0.1, -0.2, 1.], [-6.5, 0.0, 6.3]], Device::Cpu, Some(true));
  println!("z: {}", z);

  let output = &x / &y;

  println!("Output: {:?}", output);

  let loss_fn = loss::MAELoss::new("mean");

  let mut f = loss_fn.loss(&output, &y);

  f.backward();

  println!("x Grad: {:?}", x.grad());
  println!("y Grad: {:?}", y.grad());
}
