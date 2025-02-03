use crate::{reduce_grad, tensor::*};
use super::super::grad::*;


#[derive(Debug)]
pub struct SumGrad {
  input: Tensor,
  output: Tensor,
}

impl SumGrad {
  pub fn new(input: &Tensor, output: &Tensor) -> Self {
    SumGrad {
      input: input.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for SumGrad {
  fn backward(&self) {
    let device = self.output.device();
    if let Some(input_grad) = &self.input.grad() {
      if let Some(out_grad) = &self.output.grad() {
        // For sum, we need to expand the gradient to match input shape
        let input_shape = self.input.tensor().shape();
        let ones = Storage::ones(input_shape.clone(), Some(device), None);
        let expanded_grad = &ones * out_grad.borrow().get(&[0]);
        input_grad.borrow_mut().add_tensor_assign(&expanded_grad);
      }
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.input]
  }
}

#[derive(Debug)]
pub struct MeanGrad {
  input: Tensor,
  output: Tensor,
}

impl MeanGrad {
  pub fn new(input: &Tensor, output: &Tensor) -> Self {
    MeanGrad {
      input: input.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for MeanGrad {
  fn backward(&self) {
    let device = self.output.device();

    if let Some(input_grad) = &self.input.grad() {
      if let Some(out_grad) = &self.output.grad() {
        // For mean, expand gradient and divide by number of elements
        let input_shape = self.input.tensor().shape();
        let n_elements = input_shape.iter().product::<usize>() as f32;
        let ones = Storage::ones(input_shape.clone(), Some(device), None);
        let expanded_grad = &ones * (out_grad.borrow().get(&[0]) / n_elements);
        input_grad.borrow_mut().add_tensor_assign(&expanded_grad);
      }
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.input]
  }
}

#[derive(Debug)]
pub struct ProductGrad {
  input: Tensor,
  output: Tensor,
}

impl ProductGrad {
  pub fn new(input: &Tensor, output: &Tensor) -> Self {
    ProductGrad {
      input: input.clone(),
      output: output.clone(),
    }
  }
}

impl GradientFunction for ProductGrad {
  fn backward(&self) {
    let device = self.output.device();

    if let Some(input_grad) = &self.input.grad() {
      if let Some(out_grad) = &self.output.grad() {
        // For product, each element's gradient is the product of all other elements
        let input_data = self.input.tensor();
        let mut grad = Storage::zeros(input_data.shape().clone(), Some(device), None);
        let total_product = self.output.tensor().get(&[0]);
        
        // For each element, divide total product by that element to get product of others
        for i in 0..input_data.data().borrow().len() {
          let element = input_data.data().borrow()[i];
          if element != 0.0 {
            grad.data_mut()[i] = total_product / element;
          }
        }
        
        // Multiply by output gradient
        grad = &grad * out_grad.borrow().get(&[0]);
        input_grad.borrow_mut().add_tensor_assign(&grad);
      }
    }
  }

  fn prev(&self) -> Vec<&Tensor> {
    vec![&self.input]
  }
}

