mod tensor;  // if your file is tensor.rs
mod autograd;


use tensor::Tensor;
use autograd::scalar;

use ndarray::prelude::*;


fn main() {
  // let arr = array![[4, 2, 5], [2, 6,7]];

  // let test = Tensor::from_ndarray(&arr, true);

  // test.print();
  // test.print_data();

  let mut graph = scalar::Graph::new();

  let x = graph.scalar(1.75);
  let a = graph.scalar(-1.2);
  let b = graph.scalar(1.9);
  let z = graph.scalar(2.2);
  let a1 = graph.mul(x, a);
  let a2 = graph.sin(a1);
  let a3 = graph.mul(b,a2);
  let y = graph.add(a3, z);

  graph.backward(y);

  println!("Grad of x: {:?}", graph.get(x).grad);
  println!("Grad of a: {:?}", graph.get(a).grad);
  println!("Grad of b: {:?}", graph.get(b).grad);
  println!("Grad of z: {:?}", graph.get(z).grad);
  println!("Grad of y: {:?}", graph.get(y).grad);

}

