mod tensor;
mod autograd;  // Declare the module
use autograd::*;  // Import everything public from autograd




fn main() {
  let graph = Graph::new();

  let x = graph.scalar(0.8, true);

  let z = 3. / x;



  graph.backward(z);

  println!("z: value: {}, grad: {}", z.value(), z.grad());
  println!("x: value: {}, grad: {}", x.value(), x.grad());


  // let x = graph.scalar(0.8);
  // let two = graph.scalar(2.);
  // let twox = graph.mul(&two, &x);
  // let etwox = graph.exp(&twox);

  // let one = graph.scalar(1.);
  // let none = graph.scalar(-1.);

  // let num = graph.add(&etwox, &none);
  // let denum = graph.add(&etwox, &one);

  // let y = graph.div(&num, &denum);


  // println!("Forward pass values:");
  // println!("y = {}", graph.get_value(&y));
  // println!("x = {}", graph.get_value(&x));
  
  // graph.backward(&y);
  
  // println!("\nGradients:");
  // println!("dy/dy = {}", graph.get_grad(&y));
  // println!("dy/dx = {}", graph.get_grad(&x));

  // // println!("Graph: {}", y.graph);


}