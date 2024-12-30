use ferrite::prelude::*;
use ndarray::prelude::*;





fn main() {
  let x = Tensor::from_ndarray(&array![[1,2,3],[4,5,6]], Some(true));
  let x_t = x.transpose();
  let y = Tensor::from_ndarray(&array![[2]], Some(true));

  println!("x: {:?}", x);
  println!("x_t: {:?}", x_t);
  println!("y: {:?}", y);

  let z = x_t.mul_tensor(&y);

  println!("z: {:?}", z);

  let mut f = z.sum();

  f.backward();

  println!("grad f: {:?}", f.grad());

  println!("grad z: {:?}", z.grad());

  println!("grad x: {:?}", x.grad());
  println!("grad y: {:?}", y.grad());
}
