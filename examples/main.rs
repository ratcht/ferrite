use ferrite::prelude::*;
use ndarray::prelude::*;


fn main() {
  let mut model = Layer::Sequential::new(vec![
    layer!(Linear::new(3, 4, false, Device::Cpu)),
    layer!(Linear::new(4, 2, false, Device::Cpu))
  ]);

  let loss_fn = Loss::MSELoss::new("mean");
  let optimizer = Optimizer::SGD::new(model.parameters(), 0.01, 0.0);


  let input = Tensor::from_ndarray(&array![[1.,2.,3.], [4.,4.,4.]], Device::Cpu, Some(true));
  let output = model.forward(&input);
  
  let ground_y = Tensor::from_ndarray(&array![[30.,30.], [50.,50.]], Device::Cpu, Some(false));

  let mut f = loss_fn.loss(&output, &ground_y);

  f.backward();

  optimizer.step();

  model.print_parameters(true);
}
