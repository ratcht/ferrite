use ndarray::prelude::*;
use ferrite::prelude::*;



fn main() {
  let input = Tensor::from_ndarray(&array![[1., 2., 3.], [4., 5., 6.]], Some(true));
  println!("Input: {}", input);
  println!("Input Shape: {:?}", input.shape());

  let mut linear_layer = Layer::Linear::new(3, 2, false);

  let output = linear_layer.forward(&input);

  println!("Output: {:?}", output);
  println!("Output Shape: {:?}", output.shape());

  let mut f = output.sum();

  f.backward();

  println!("Input Grad: {:?}", input.grad());


  println!("Layer Stuff ---");
  linear_layer.module.print_parameters();





}
