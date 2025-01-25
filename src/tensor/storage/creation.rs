use crate::*;

macro_rules! match_device {
  // The pattern: we match a single expression (`$x:expr`).
  (storage $device:expr, $($x:tt)*) => {
    match $device {
      Device::Cpu => Storage::Cpu(CpuStorage::$($x)*),
      // Add other devices here
      _ => unimplemented!("Device not supported"),
    }
  };

  (call $device:expr, $($x:tt)*) => {
    match $device {
      Device::Cpu => CpuStorage::$($x)*,
      // Add other devices here
      _ => unimplemented!("Device not supported"),
    }
  };
}

impl DeviceStorageCreation for Storage {
  fn zeros(shape: Vec<usize>, device: Option<Device>, _requires_grad: Option<bool>) -> Self {
    let device = device.expect("Storage: device must be non-null!");
    match_device!(storage device, zeros(shape, None, None))
  }

  fn ones(shape: Vec<usize>, device: Option<Device>, _requires_grad: Option<bool>) -> Self {
    let device = device.expect("Storage: device must be non-null!");
    match_device!(storage device, ones(shape, None, None))
  }

  fn from_ndarray<S, D, T>(data: &ndarray::ArrayBase<S, D>, device: Option<Device>, _requires_grad: Option<bool>) -> Self
  where 
    S: ndarray::Data<Elem = T>,
    T: num_traits::AsPrimitive<f32>,
    D: ndarray::Dimension 
  {
    let device = device.expect("Storage: device must be non-null!");
    match_device!(storage device, from_ndarray(data, None, None))
  }

  fn uniform(l_bound: f32, r_bound: f32, shape: Vec<usize>, device: Option<Device>, _requires_grad: Option<bool>) -> Self {
    let device = device.expect("Storage: device must be non-null!");
    match_device!(storage device, uniform(l_bound, r_bound, shape, None, None))
  }
}
