mod tensor;  // Makes the tensor module public
use tensor::*;  // Import everything public from autograd


mod autograd;  // Declare the module
use autograd::scalar::*;  // Import everything public from autograd

use ndarray::prelude::*;



fn main() {
  let ndarray = array![[[[2.,1.,1.], [1.,1.,1.]], [[2.,1.,1.], [1.,1.,1.]], [[2.,1.,1.], [1.,1.,1.]], [[2.,1.,1.], [1.,1.,1.]]]];
  // let tensor = Tensor::ones(vec![4, 4], false);
  let tensor = Tensor::from_ndarray(&ndarray, true);

  // println!("{}", tensor);
  println!("Tensor: {}", tensor);
  println!("Shape: {:?}", tensor.shape());
  println!("Stride: {:?}", tensor.stride());

  let mut transpose = tensor.clone();
  transpose.permute(&[2,1,3,0]);

  println!("Tensor: {}", transpose);
  println!("Shape: {:?}", transpose.shape());
  println!("Stride: {:?}", transpose.stride());

}