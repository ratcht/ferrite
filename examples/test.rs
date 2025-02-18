use ndarray::prelude::*;
use ferrite::prelude::*;



fn main() {
  let layer1 = layer!(Linear::new(3, 4, false, Device::Cpu));
  let layer2 = layer!(Linear::new(4, 2, false, Device::Cpu));

  let mut model = Layer::Sequential::new(vec![layer1, layer2]);

  model.print_parameters(true);

  let input = Tensor::from_ndarray(&array![[1.,2.,3.], [4.,4.,4.]], Device::Cpu, Some(true));

  let predicted_y = model.forward(&input);
  
  println!("predicted_y: {:?}", predicted_y);

  let loss_fn = Loss::MSELoss::new("mean");
  let optimizer = Optimizer::SGD::new(model.parameters(), 0.01, 0.0);

  let ground_y = Tensor::from_ndarray(&array![[30.,30.], [50.,50.]], Device::Cpu, Some(false));

  let mut f = loss_fn.loss(&predicted_y, &ground_y);

  println!("f: {:?}", f);
  
  f.backward();

  optimizer.step();

  model.print_parameters(true);
}
