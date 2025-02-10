use crate::*;
use ndarray::prelude::*;


impl ReductionOps for CpuStorage {
  fn sum(&self) -> Self {
    let data: f32 = self.data().read().unwrap().iter().sum();
    CpuStorage::from_ndarray(&array![data], None, None)
  }

  fn sum_axis(&self, axis: usize) -> Self {
    // Compute the sizes of the "outer" dimensions, the dimension to sum over,
    // and the "inner" (trailing) dimensions.
    let outer: usize = self.shape()[..axis].iter().product();
    let axis_len: usize = self.shape()[axis];
    let trailing: usize = self.shape()[axis+1..].iter().product();
    
    // The new shape is the original shape with the summing axis removed.
    let mut new_shape = self.shape().clone();
    new_shape.remove(axis);

    // Prepare a vector for the summed data.
    let mut new_data = vec![0.0; outer * trailing];
    
    // Borrow the underlying data.
    let binding = self.data();
    let data_ref = binding.read().unwrap();
    
    // Iterate over the "outer" blocks and the "inner" trailing dimensions.
    // For each such location, sum over the elements along the `axis`.
    for i in 0..outer {
      for k in 0..trailing {
        let mut sum = 0.0;
        for j in 0..axis_len {
          // In contiguous (row-major) layout, the index is computed as:
          //   index = offset + i * (axis_len * trailing) + j * trailing + k
          let index = self.offset() + i * (axis_len * trailing) + j * trailing + k;
          sum += data_ref[index];
        }
        new_data[i * trailing + k] = sum;
      }
    }
    
    // Construct a new CpuStorage with the summed data and the new shape.
    CpuStorage::new(new_data, new_shape)
  }

  fn product(&self) -> Self {
    let data: f32 = self.data().read().unwrap().iter().sum();
    CpuStorage::from_ndarray(&array![data], None, None)
  }

  fn mean(&self) -> Self {
    let data: f32 = self.data().read().unwrap().iter().sum::<f32>() / self.data().read().unwrap().len() as f32;
    CpuStorage::from_ndarray(&array![data], None, None)
  }

  
}