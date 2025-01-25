use crate::*;


macro_rules! match_self{
  // The pattern: we match a single expression (`$x:expr`).
  (storage $self:expr, $($x:tt)*) => {
    match $self {
      Storage::Cpu(cpu) => Storage::Cpu(cpu.$($x)*),
      // Add other devices here
      _ => unimplemented!("Device not supported"),
    }
  };

  (call $self:expr, $($x:tt)*) => {
    match $self {
      Storage::Cpu(cpu) => cpu.$($x)*,
      // Add other devices here
      _ => unimplemented!("Device not supported"),
    }
  }
}

impl DeviceStorage for Storage {
  fn view(&self, new_shape: Vec<usize>) -> Self where Self: Sized {
    match_self!(storage self, view(new_shape))
  }

  fn data(&self) -> std::rc::Rc<std::cell::RefCell<Vec<f32>>> {
    match_self!(call self, data())
  }

  fn data_mut(&self) -> std::cell::RefMut<Vec<f32>> {
    match_self!(call self, data_mut())
  }

  fn set_data(&mut self, data: Vec<f32>) {
    match_self!(call self, set_data(data));
  }

  fn shape(&self) -> &Vec<usize> {
    match_self!(call self, shape())
  }

  fn set_shape(&mut self, shape: Vec<usize>) {
    match_self!(call self, set_shape(shape));
  }

  fn stride(&self) -> &Vec<usize> {
    match_self!(call self, stride())
  }

  fn set_stride(&mut self, stride: Vec<usize>) {
    match_self!(call self, set_stride(stride));
  }

  fn get(&self, indices: &[usize]) -> f32 {
    match_self!(call self, get(indices))
  }

  fn set(&mut self, indices: &[usize], value: f32) {
    match_self!(call self, set(indices, value));
  }

  fn make_contiguous(&self) -> (Vec<f32>, i32) {
    match_self!(call self, make_contiguous())
  }

  fn is_contiguous(&self) -> bool {
    match_self!(call self, is_contiguous())
  }
}
