use super::base::Tensor;  // Import from parent module's base.rs
pub trait TensorShape {
  fn reshape(&mut self, new_shape: Vec<usize>);
  fn permute(&mut self, dims: &[usize]);
  fn transpose(&mut self);
  fn flatten(&mut self);
  fn squeeze(&mut self);
  fn unsqueeze(&mut self, dim: usize);
}


impl TensorShape for Tensor {
  fn reshape(&mut self, new_shape: Vec<usize>) {
    self.set_shape(new_shape);
  }

  fn transpose(&mut self) {
    // Transpose by swapping dimensions & strides
    if self.shape().len() != 2 { panic!("Must be 2-D Tensor (Matrix)"); }

    let mut shape = self.shape().to_owned();
    shape.reverse();

    let mut stride = self.stride().to_owned();
    stride.reverse();

    self.set_shape(shape);
    self.set_stride(stride);
  }

  fn permute(&mut self, dims: &[usize]) {
    let self_shape = self.shape();
    let shape = dims.iter().map(|&i| self_shape[i]).collect();    

    let self_stride = self.stride();
    let stride = dims.iter().map(|&i| self_stride[i]).collect();   

    self.set_shape(shape);
    self.set_stride(stride);
  }

  fn flatten(&mut self) {
    let shape: Vec<usize> = vec![self.shape().iter().product()];
    let stride = vec![1];

    self.set_shape(shape);
    self.set_stride(stride);
  }

  fn squeeze(&mut self) {
    todo!()
  }

  fn unsqueeze(&mut self, dim: usize) {
    todo!()
  }
  
  
}