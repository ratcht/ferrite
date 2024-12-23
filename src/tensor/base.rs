use std::rc::Rc;
use std::cell::RefCell;


#[derive(Clone)]
pub struct Tensor {
  data: Rc<RefCell<Vec<f32>>>,
  shape: Vec<usize>,
  stride: Vec<usize>,
  offset: usize,
  requires_grad: bool,
}


impl Tensor {
  pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
    // Check data
    if data.len() != shape.iter().product() { panic!("Data does not match shape!");}
    let stride = Tensor::compute_strides(&shape);
    Tensor {
      data: Rc::new(RefCell::new(data)),
      shape: shape,
      stride: stride,
      offset: 0,
      requires_grad: requires_grad,
    }
  }

  pub fn new_with_stride(data: Vec<f32>, shape: Vec<usize>, stride: Vec<usize>, requires_grad: bool) -> Self {
    // Check data
    if data.len() != shape.iter().product() { panic!("Data does not match shape!");}
    Tensor {
      data: Rc::new(RefCell::new(data)),
      shape: shape,
      stride: stride,
      offset: 0,
      requires_grad: requires_grad,
    }
  }

  pub fn create(data: Rc<RefCell<Vec<f32>>>, shape: Vec<usize>, stride: Vec<usize>, requires_grad: bool) -> Self {
    // Check data
    Tensor {
      data: data,
      shape: shape,
      stride: stride,
      offset: 0,
      requires_grad: requires_grad,
    }
  }

  pub fn view(&self, new_shape: Vec<usize>) -> Self {
    // Check if the new shape is compatible
    let total_elements: usize = new_shape.iter().product();
    if total_elements != self.shape().iter().product() { panic!("New shape must have the same number of elements"); }
    let stride = Tensor::compute_strides(&new_shape);

    Tensor {
      data: Rc::clone(&self.data),
      shape: new_shape,
      stride: stride,
      offset: self.offset,
      requires_grad: self.requires_grad,
    }
  }

  pub fn data(&self) -> Rc<RefCell<Vec<f32>>> {
    Rc::clone(&self.data)
  }

  pub fn data_mut(&self) -> std::cell::RefMut<Vec<f32>> {
    self.data.borrow_mut()
  }

  pub fn set_data(&mut self, data: Vec<f32>){
    self.data = Rc::new(RefCell::new(data));
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

  pub fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
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
    self.data().borrow()[flat_index]
  }

  pub fn set(&mut self, indices: &[usize], value: f32) {
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

    self.data().borrow_mut()[flat_index] = value;
  }

}


