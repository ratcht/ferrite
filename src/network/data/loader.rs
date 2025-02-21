use crate::tensor::*;
use std::sync::{Arc, RwLock};
use rand::seq::SliceRandom;
use rand::thread_rng;

// Enum to specify which dataset split to access
pub enum Split {
  Train,
  Validation,
  Test,
}

pub struct DataLoader {
  x_data: Arc<RwLock<Tensor>>,
  y_data: Arc<RwLock<Tensor>>,
  batch_size: usize,
  train_split: f32,
  test_split: f32,
  validation_split: f32,
  train_size: usize,
  test_size: usize,
  validation_size: usize,
}

impl DataLoader {
  /// Creates a new DataLoader instance, computing split sizes based on ratios.
  pub fn new(
    x_data: Arc<RwLock<Tensor>>,
    y_data: Arc<RwLock<Tensor>>,
    batch_size: usize,
    train_split: f32,
    test_split: f32,
    validation_split: f32,
  ) -> Self {
    // Ensure splits sum to 1.0 (with small tolerance for floating-point errors)
    let total_split = train_split + test_split + validation_split;
    assert!(
      (total_split - 1.0).abs() < 1e-6,
      "Split ratios must sum to 1.0"
    );

    // Get the total number of samples from x_data
    let x_data_read = x_data.read().unwrap();
    let n_samples = x_data_read.shape()[0];

    // Compute sizes for each split
    let train_size = (n_samples as f32 * train_split).floor() as usize;
    let test_size = (n_samples as f32 * test_split).floor() as usize;
    let validation_size = n_samples - train_size - test_size;

    Self {
      x_data,
      y_data,
      batch_size,
      train_split,
      test_split,
      validation_split,
      train_size,
      test_size,
      validation_size,
    }
  }

  /// Returns an iterator over batches for the specified split.
  /// If `shuffle` is true, the indices are shuffled before batching.
  pub fn batches(&self, split: Split, shuffle: bool) -> BatchIterator {
    // Determine the index range based on the split
    let (start, end) = match split {
      Split::Train => (0, self.train_size),
      Split::Validation => (self.train_size, self.train_size + self.validation_size),
      Split::Test => (
        self.train_size + self.validation_size,
        self.train_size + self.validation_size + self.test_size,
      ),
    };

    // Collect indices for the split
    let mut indices: Vec<usize> = (start..end).collect();

    // Shuffle indices if requested
    if shuffle {
      let mut rng = thread_rng();
      indices.shuffle(&mut rng);
    }

    BatchIterator {
      x_data: self.x_data.clone(),
      y_data: self.y_data.clone(),
      indices,
      batch_size: self.batch_size,
      current: 0,
    }
  }
}

/// Iterator over batches of data from the DataLoader.
pub struct BatchIterator {
    x_data: Arc<RwLock<Tensor>>,
    y_data: Arc<RwLock<Tensor>>,
    indices: Vec<usize>,
    batch_size: usize,
    current: usize,
}

impl Iterator for BatchIterator {
  type Item = (Tensor, Tensor);

  fn next(&mut self) -> Option<Self::Item> {
    if self.current >= self.indices.len() {
      None
    } else {
      // Calculate batch boundaries
      let start = self.current;
      let end = (start + self.batch_size).min(self.indices.len());
      let batch_indices = &self.indices[start..end];
      self.current = end;

      // Access data with read locks
      let x_data_read = self.x_data.read().unwrap();
      let y_data_read = self.y_data.read().unwrap();

      // Select batch data
      let x_batch = x_data_read.index_select(0, batch_indices);
      let y_batch = y_data_read.index_select(0, batch_indices);

      Some((x_batch, y_batch))
    }
  }
}