use ndarray::prelude::*;
use ferrite::prelude::*;



fn main() {
  // let a = Tensor::from_ndarray(&array![[1., 2., 3.], [4., 5., 6.]], Some(true)); // 3 x 2
  // let b = Tensor::from_ndarray(&array![[7., 8.], [9., 10.], [11., 12.]], Some(false)); // 2 x 4

  // println!("A: {}", a);
  // println!("B: {}", b);


  // let c = a.matmul(&b, false, false);

  // println!("C: {}", c);


  let w = Tensor::from_ndarray(&array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], Some(true));
  println!("w: {}", w);

  let w_t = w.transpose();
  println!("w_t: {}", w_t);


  let t = Tensor::from_ndarray(&array![[1., 2., 3.], [4., 5., 6.]], Some(true));
  println!("t: {}", t);


  let c_t = t.matmul(&w_t, false, false);

  println!("C: {}", c_t);

  // let input = Tensor::from_ndarray(&array![[1., 2., 2.], [3., 4., 6.], [5., 6., 7.]], Some(true)); // 3 x 2

  // let mut closure = |name: &str, tensor: &Tensor| {
  //   println!("Parameter {}: {}", name, tensor);
  // };

  // let mut layer = Layer::Linear::new(3, 5, false);
  // layer.module.visit_parameters(&mut closure);

  // let output = layer.forward(&a);
  
  // println!("output: {}", output);



}

