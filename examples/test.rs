use ndarray::prelude::*;
use ferrite::prelude::*;



fn main() {
  let input = Tensor::from_ndarray(&array![[1., 2., 3.], [4., 5., 6.]], Some(true));
  println!("Input: {}", input);

  let mut sequential = Layer::Sequential::new(vec![
    layer!(Linear::new(3, 5, true)),
    layer!(Linear::new(5, 4, false)),
    layer!(Linear::new(4, 3, false)),
  ]);

  let output = sequential.forward(&input);

  println!("Output: {:?}", output);

  let mut f = output.sum();

  f.backward();

  println!("Input Grad: {:?}", input.grad());
}
