mod tensor_storage;  // Makes the tensor module public
use tensor_storage::*;  // Import everything public from autograd


mod autograd;  // Declare the module
use autograd::scalar::*;  // Import everything public from autograd
use autograd::tensor::*;  // Import everything public from autograd


use ndarray::prelude::*;



fn main() {
  let x = Tensor::from_ndarray(&array![[1,2,3],[4,5,6]], Some(false));
  let y = Tensor::from_ndarray(&array![[1,1,1],[1,1,1]], Some(false));

  let z = &x + &y;

  println!("{:?}", z.grad_fn());
  println!("{}", x);
  println!("{}", y);
  

  println!("{:?}", z);



}