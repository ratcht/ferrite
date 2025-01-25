use ferrite::prelude::*;
use ndarray::prelude::*;





fn main() {
  let a = Tensor::from_ndarray(&array![[1.,2.,3.],[4.,5.,6.]], Device::Cpu, Some(true)); //(2x3)
  let b = Tensor::from_ndarray(&array![[2.,2.,2.,2.,1.], [2.,2.,2.,2.,1.], [2.,2.,2.,2.,1.]], Device::Cpu, Some(true)); //(3,5)

  let y = a.matmul(&b, false, false);
  
  let mut f = y.sum();
  f.backward();

  println!("A: {}", a);
  println!("B: {}", b);
  println!("y: {}", y);
  println!("f: {}", f);

  println!("grad: A: {:?}", a.grad());
  println!("grad: B: {:?}", b.grad());
  println!("grad: y: {:?}", y.grad());
  println!("grad: f: {:?}", f.grad());

}
