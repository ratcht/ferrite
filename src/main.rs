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

  let a = graph.scalar(2.);
  let b = graph.scalar(-3.);
  let c = graph.scalar(10.);
  let e = graph.mul(a, b);
  let d = graph.add(e, c);
  let f = graph.scalar(-2.);
  let L = graph.mul(d, f);

  graph.backward(L);

  println!("Grad of a: {:?}", graph.get(a).grad);
  println!("Grad of b: {:?}", graph.get(b).grad);
  println!("Grad of c: {:?}", graph.get(c).grad);
  println!("Grad of e: {:?}", graph.get(e).grad);
  println!("Grad of d: {:?}", graph.get(d).grad);
  println!("Grad of f: {:?}", graph.get(f).grad);
  println!("Grad of L: {:?}", graph.get(L).grad);


}

