use std::sync::{Arc, RwLock};
use crate::*;
use ndarray::{ArrayBase, Dimension};
use num_traits::cast::AsPrimitive;
use rand::distributions::{Distribution, Uniform};

#[derive(Clone)]
pub struct CpuStorage {
    data: Arc<RwLock<Vec<f32>>>,
    shape: Vec<usize>,
    stride: Vec<usize>,
    offset: usize,
}

impl DeviceStorageStatic for CpuStorage {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        // Check that the data length matches the product of shape dimensions.
        if data.len() != shape.iter().product::<usize>() {
            let x: usize = shape.iter().product::<usize>();
            println!("Data Len: {}. Shape iter prod {}", data.len(), x);
            println!("Data: {:?}", data);
            panic!("Data does not match shape!");
        }
        let stride = CpuStorage::compute_strides(&shape);
        CpuStorage {
            data: Arc::new(RwLock::new(data)),
            shape: shape,
            stride: stride,
            offset: 0,
        }
    }

    fn new_with_stride(data: Vec<f32>, shape: Vec<usize>, stride: Vec<usize>) -> Self {
        if data.len() != shape.iter().product::<usize>() {
            panic!("Data does not match shape!");
        }
        CpuStorage {
            data: Arc::new(RwLock::new(data)),
            shape: shape,
            stride: stride,
            offset: 0,
        }
    }

    fn create(data: Arc<RwLock<Vec<f32>>>, shape: Vec<usize>, stride: Vec<usize>) -> Self {
        CpuStorage {
            data: data,
            shape: shape,
            stride: stride,
            offset: 0,
        }
    }

    fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
        let mut stride = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        stride
    }
}

impl DeviceStorageCreation for CpuStorage {
    fn zeros(shape: Vec<usize>, _device: Option<Device>, _requires_grad: Option<bool>) -> Self {
        let size = shape.iter().product();
        let data = vec![0.0; size];
        CpuStorage::new(data, shape)
    }

    fn ones(shape: Vec<usize>, _device: Option<Device>, _requires_grad: Option<bool>) -> Self {
        let size = shape.iter().product();
        let data = vec![1.0; size];
        CpuStorage::new(data, shape)
    }

    fn from_ndarray<S, D, T>(
        data: &ArrayBase<S, D>,
        _device: Option<Device>,
        _requires_grad: Option<bool>,
    ) -> Self
    where
        S: ndarray::Data<Elem = T>,
        T: AsPrimitive<f32>,
        D: Dimension,
    {
        let shape = data.shape().to_vec();
        let arr = data.mapv(|x| x.as_());
        let data = arr.iter().cloned().collect();
        CpuStorage::new(data, shape)
    }

    fn uniform(
        l_bound: f32,
        r_bound: f32,
        shape: Vec<usize>,
        _device: Option<Device>,
        _requires_grad: Option<bool>,
    ) -> Self {
        let uniform = Uniform::from(l_bound..r_bound); // Create a uniform distribution
        let mut rng = rand::thread_rng(); // Random number generator
        let data = (0..shape.iter().product())
            .map(|_| uniform.sample(&mut rng)) // Sample from the uniform distribution
            .collect();
        CpuStorage::new(data, shape)
    }
}

impl DeviceStorage for CpuStorage {
    fn view(&self, new_shape: Vec<usize>) -> Self {
        // Check if the new shape is compatible.
        let total_elements: usize = new_shape.iter().product();
        if total_elements != self.shape.iter().product::<usize>() {
            panic!("New shape must have the same number of elements");
        }
        let stride = CpuStorage::compute_strides(&new_shape);
        CpuStorage {
            data: Arc::clone(&self.data),
            shape: new_shape,
            stride: stride,
            offset: self.offset,
        }
    }

    fn data(&self) -> Arc<RwLock<Vec<f32>>> {
        Arc::clone(&self.data)
    }

    fn data_mut(&self) -> std::sync::RwLockWriteGuard<Vec<f32>> {
        self.data.write().unwrap()
    }

    fn set_data(&mut self, data: Vec<f32>) {
        self.data = Arc::new(RwLock::new(data));
    }

    fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    fn set_shape(&mut self, shape: Vec<usize>) {
        self.shape = shape;
    }

    fn stride(&self) -> &Vec<usize> {
        &self.stride
    }

    fn set_stride(&mut self, stride: Vec<usize>) {
        self.stride = stride;
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn get(&self, indices: &[usize]) -> f32 {
        // Ensure the number of indices matches the tensor's dimensions.
        if indices.len() != self.shape.len() {
            panic!("Tensor index does not match shape!");
        }
        // Compute the flat index.
        let mut flat_index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                panic!("Tensor index out of bounds!");
            }
            flat_index += idx * self.stride[i];
        }
        // Use a read lock for safe concurrent access.
        let data = self.data.read().unwrap();
        data[flat_index]
    }

    fn set(&mut self, indices: &[usize], value: f32) {
        if indices.len() != self.shape.len() {
            panic!("Tensor index does not match shape!");
        }
        let mut flat_index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                panic!("Tensor index out of bounds!");
            }
            flat_index += idx * self.stride[i];
        }
        // Acquire a write lock for mutation.
        let mut data = self.data.write().unwrap();
        data[flat_index] = value;
    }

    fn make_contiguous(&self) -> (Vec<f32>, i32) {
        if self.is_contiguous() {
            return (self.data.read().unwrap().clone(), self.shape[1] as i32);
        }
        let mut contiguous = vec![0.0; self.shape.iter().product()];
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                contiguous[i * self.shape[1] + j] = self.get(&[i, j]);
            }
        }
        (contiguous, self.shape[1] as i32)
    }

    fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        for i in (0..self.shape.len()).rev() {
            if self.stride[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }
}
