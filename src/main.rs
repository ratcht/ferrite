mod tensor;  // Makes the tensor module public
use tensor::*;  // Import everything public from autograd


mod autograd;  // Declare the module
use autograd::scalar::*;  // Import everything public from autograd

use ndarray::prelude::*;



fn main() {
  let mut tensor1 = Tensor::from_ndarray(&array![[1,2],[3,4]], false);
  let tensor2 = Tensor::from_ndarray(&array![[5,6],[7,8]], false);
  let tensor = Tensor::from_ndarray(&array![[1,1],[1,1]], false);


  let tensor3 = &tensor * 3.;

  println!("{}", tensor2);

  println!("{}", tensor);

  println!("{}", tensor3);


}