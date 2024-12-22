
#[derive(Clone)]
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
    let stride = Tensor::compute_stride(&shape);
    Tensor {
      data,
      shape,
      stride,
      requires_grad,
    }
  }

  pub fn new_with_stride(data: Vec<f32>, shape: Vec<usize>, stride: Vec<usize>, requires_grad: bool) -> Self {
    // Check data
    if data.len() != shape.iter().product() { panic!("Data does not match shape!");}
    Tensor {
      data,
      shape,
      stride,
      requires_grad,
    }
  }

  pub fn data(&self) -> &Vec<f32> {
    &self.data
  }

  pub fn set_data(&mut self, data: Vec<f32>){
    self.data = data;
  }

  pub fn shape(&self) -> &Vec<usize> {
    &self.shape
  }

  pub fn set_shape(&mut self, shape: Vec<usize>){
    self.shape = shape;
  }

  pub fn stride(&self) -> &Vec<usize> {
    &self.stride
  }

  pub fn compute_stride(shape: &Vec<usize>) -> Vec<usize> {
    let mut stride = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
      stride[i] = stride[i + 1] * shape[i + 1];
    }
    stride
  }

  pub fn set_stride(&mut self, stride: Vec<usize>){
    self.stride = stride;
  }

  pub fn requires_grad(&self) -> bool {
    self.requires_grad
  }

  pub fn get(&self, indices: &[usize]) -> f32 {
    // Ensure the number of indices matches the tensor's dimensions
    if indices.len() != self.shape().len() { panic!("Tensor index does not match shape!"); }

    // Compute the flat index
    let mut flat_index = 0;
    for (i, &idx) in indices.iter().enumerate() {
      // Validate the index for this dimension
      if idx >= self.shape()[i] {
        panic!("Tensor index out of bounds!");
      }

      // Accumulate the flat index
      flat_index += idx * self.stride()[i];
    }

    // Return the value if the flat index is valid
    self.data()[flat_index]
  }
}


