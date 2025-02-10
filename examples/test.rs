use ndarray::prelude::*;
use ferrite::prelude::*;



fn main() {
  let x = Tensor::from_ndarray(&array![[2., 2., 3.], [4., 4., 6.]], Device::Cpu, Some(true));
  println!("x: {}", x);

  let x_t = x.transpose();
  println!("x_t: {}", x_t);

  let y = Tensor::from_ndarray(&array![[12., 2., 3.], [44., 4., 6.]], Device::Cpu, Some(true));
  println!("y: {}", y);

  let z = y.matmul(&x_t, false, false);
  println!("z: {:?}", z);

  let b = Tensor::from_ndarray(&array![[13.1, 22.5], [4.6, 4.5]], Device::Cpu, Some(true));

  let c = &z / &b;
  println!("c: {:?}", c);

  let d = c.softmax(1);
  println!("d: {:?}", d);

  let target = Tensor::from_ndarray(&array![0., 1.], Device::Cpu, Some(true));
  let loss_fn = Loss::MAELoss::new("mean");

  let mut loss = loss_fn.loss(&c, &target);
  println!("loss: {:?}", loss);

  loss.backward();
  println!("x grad: {:?}", x.grad());


}
