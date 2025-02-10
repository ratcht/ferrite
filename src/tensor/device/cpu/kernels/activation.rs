use crate::*;
use rayon::prelude::*;


impl ActivationOps for CpuStorage {
  fn binary_step(&self) -> Self {
    self.apply(|x| if x < 0. { 0. } else { 1. })
  }

  fn sigmoid(&self) -> Self {
    self.apply(|x| 1./(1. + f32::exp(-x)))
  }

  fn tanh(&self) -> Self {
    self.apply(|x| (f32::exp(x) - f32::exp(-x))/(f32::exp(x) + f32::exp(-x)))
  }

  fn relu(&self) -> Self {
    self.apply(|x| f32::max(0., x))
  }

  fn leaky_relu(&self) -> Self {
    self.apply(|x| f32::max(0.1*x, x))
  }

  fn parametric_relu(&self, a: f32) -> Self {
    self.apply(|x| f32::max(a*x, a))
  }

  fn elu(&self, alpha: f32) -> Self {
    self.apply(|x| if x >= 0. {x} else {alpha * (f32::exp(x) - 1.)})
  }

  fn softmax(&self, dim: usize) -> Self {
    // Compute dimensions:
    // - outer: product of dimensions before `dim`
    // - axis_len: size of the softmax dimension (i.e. at `dim`)
    // - inner: product of dimensions after `dim`
    let outer: usize = self.shape()[..dim].iter().product();
    let axis_len: usize = self.shape()[dim];
    let inner: usize = self.shape()[dim + 1..].iter().product();
    let total_elements = self.shape().iter().product();

    // Acquire a read lock and clone the input data.
    let input: Vec<f32> = {
      let binding = self.data();
      let guard = binding.read().unwrap();
      guard.clone()
    };

    // Allocate an output vector of the same size.
    let mut new_data = vec![0.0; total_elements];
    let base_offset = self.offset();

    // We assume the tensor is contiguous, so the region from base_offset to
    // base_offset + outer*(axis_len*inner) contains the relevant data.
    // Split this region into `outer` mutable chunks, each of length axis_len*inner.
    new_data[base_offset..base_offset + outer * (axis_len * inner)]
      .par_chunks_mut(axis_len * inner)
      .enumerate()
      .for_each(|(i, out_slice)| {
        // Compute the corresponding slice from the input data.
        let in_start = base_offset + i * (axis_len * inner);
        let in_end = in_start + (axis_len * inner);
        let in_slice = &input[in_start..in_end];

        // For each inner index (each column within the slice)
        for k in 0..inner {
          // Find the maximum value along the softmax axis for numerical stability.
          let mut max_val = f32::NEG_INFINITY;
          for j in 0..axis_len {
            let idx = j * inner + k;
            let v = in_slice[idx];
            if v > max_val {
              max_val = v;
            }
          }

          // Compute the exponentials and their sum.
          let mut sum_exp = 0.0;
          let mut exps = vec![0.0; axis_len];
          for j in 0..axis_len {
            let idx = j * inner + k;
            let exp_val = f32::exp(in_slice[idx] - max_val);
            exps[j] = exp_val;
            sum_exp += exp_val;
          }

          // Normalize the exponentials and write the results to out_slice.
          for j in 0..axis_len {
            let idx = j * inner + k;
            out_slice[idx] = exps[j] / sum_exp;
          }
        }
      });

    // Return a new CpuStorage with the same shape as the original.
    CpuStorage::new(new_data, self.shape().clone())
  }

  fn swish(&self) -> Self {
    self.apply(|x| x * (1./(1. + f32::exp(-x))))
  }
}