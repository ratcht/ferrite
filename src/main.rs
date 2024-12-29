mod tensor_storage;  // Makes the tensor module public
use tensor_storage::*;  // Import everything public from autograd

mod autograd;  // Declare the module
use autograd::tensor::*;  // Import everything public from autograd

mod network;
use network::*;

use ndarray::prelude::*;



pub struct SimpleNetwork {
  module: Module,
}

impl SimpleNetwork {
  fn new() -> Self {
    let module = Module::new();

    SimpleNetwork {
      module: module
    }
  }
}

impl Segment for SimpleNetwork {
  fn forward(input: Tensor) -> Tensor {
    todo!()
  }
}


fn main() {
  let x = Tensor::from_ndarray(&array![[1,2,3],[4,5,6]], Some(true));
  let y = Tensor::from_ndarray(&array![[1,1,1]], Some(true));

  println!("x: {:?}", x);
  println!("y: {:?}", y);

  let z = x.mul_tensor(&y);

  println!("z: {:?}", z);

  let mut f = z.sum();

  f.backward();

  println!("grad f: {:?}", f.grad());

  println!("grad z: {:?}", z.grad());

  println!("grad x: {:?}", x.grad());
  println!("grad y: {:?}", y.grad());
}
