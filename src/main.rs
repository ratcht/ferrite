mod tensor;  // Makes the tensor module public
use tensor::*;  // Import everything public from autograd


mod autograd;  // Declare the module
use autograd::scalar::*;  // Import everything public from autograd

use ndarray::prelude::*;



fn main() {
  let mut tensor = Tensor::from_ndarray(&array![[1,2],[3,4]], false);
  println!("Tensor: {}", tensor);

  println!("Shape: {:?}", tensor.shape());

  tensor.unsqueeze(0);

  println!("Tensor: {}", tensor);
  println!("Shape: {:?}", tensor.shape());

}