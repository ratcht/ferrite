use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::cast::AsPrimitive;

pub struct Tensor {
  data: Vec<f32>,
  shape: Vec<usize>,
  stride: Vec<usize>,
  requires_grad: bool,
}

impl Tensor {
  pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
    let stride = Tensor::compute_strides(&shape);
    Tensor {
      data,
      shape,
      stride,
      requires_grad,
    }
  }

  pub fn from_ndarray<S, D, T>(data: &ArrayBase<S, D>, requires_grad: bool) -> Self
  where 
    S: ndarray::Data<Elem = T> ,
    T: AsPrimitive<f32>,
    D: Dimension,
  {
    let shape = data.shape().to_vec();
    let arr = data.mapv(|x| x.as_());
    let data = arr.flatten().to_vec();
    let stride = Tensor::compute_strides(&shape);
    Tensor {
      data,
      shape,
      stride,
      requires_grad,
    }
  }

  pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> Self {
    let size = shape.iter().product();
    let data = vec![0.0; size];
    Tensor::new(data, shape, requires_grad)
  }

  pub fn ones(shape: Vec<usize>, requires_grad: bool) -> Self {
    let size = shape.iter().product();
    let data = vec![1.0; size];
    Tensor::new(data, shape, requires_grad)
  }

  pub fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
  }

  pub fn print(&self) {
    println!("Data: {:?}", self.data);
    println!("Shape: {:?}", self.shape);
    println!("Strides: {:?}", self.stride);
    println!("Requires grad: {:?}", self.requires_grad);
  }

  fn print_data_recursive<'a>(data: &'a [f32], shape: &'a [usize], stride: &'a [usize]) -> String {
    let mut res = String::new();
    res += "[";
    if shape.len() == 1 {
      for (i, value) in data.iter().enumerate() {
        res += &format!("{}", value);

        if i < data.len() - 1 {
          res += ", ";
        }
        
      }
    } else {
      for i in 0..shape[0] {
        let sub_data = &data[i*stride[0]..(i+1)*stride[0]];
        let sub_res = Self::print_data_recursive(sub_data, &shape[1..], &stride[1..]);
        res += &sub_res;
        if i < shape[0] - 1 {
          res += ", ";
        }
      }
    }

    res += "]";
    res
  }

  pub fn print_data(&self) {
    let res = Self::print_data_recursive(&self.data, &self.shape, &self.stride);
    println!("{}", res);
  }
}