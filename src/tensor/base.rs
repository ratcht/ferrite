
pub struct Tensor {
  data: Vec<f32>,
  shape: Vec<usize>,
  stride: Vec<usize>,
  requires_grad: bool,
}

impl Tensor {
  pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
    // Check data
    if data.len() != shape.iter().product() { panic!("Data does not match shape!");}
    let stride = Tensor::compute_strides(&shape);
    Tensor {
      data,
      shape,
      stride,
      requires_grad,
    }
  }

  pub fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
  }

  pub fn data(&self) -> &Vec<f32> {
    &self.data
  }

  pub fn shape(&self) -> &Vec<usize> {
    &self.shape
  }

  pub fn stride(&self) -> &Vec<usize> {
    &self.stride
  }

  pub fn requires_grad(&self) -> bool {
    self.requires_grad
  }
}


