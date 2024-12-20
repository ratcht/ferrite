mod tensor;  // Makes the tensor module public
use tensor::*;  // Import everything public from autograd


mod autograd;  // Declare the module
use autograd::scalar::*;  // Import everything public from autograd




fn main() {
  let tensor = Tensor::ones(vec![4, 4], false);

  println!("{}", tensor);

}