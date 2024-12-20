use std::assert_eq;
use std::ops::{Index, IndexMut, Add, Mul};

use std::fmt;

// Array1DView is a "view" into a slice of data - it doesn't own the data
struct Array1DView<'a> {
  data: &'a [i32],
}

// Array1DViewMut is a mutable view into a slice of data
pub struct Array1DViewMut<'a> {
  data: &'a mut [i32],
}

// Implementation for immutable view
impl<'a> Array1DView<'a> {
  fn new(data: &'a [i32]) -> Self {
    Array1DView { data }
  }

  fn len(&self) -> usize {
    self.data.len()
  }

  fn is_empty(&self) -> bool {
    self.data.is_empty()
  }
}

// Implementation for mutable view
impl<'a> Array1DViewMut<'a> {
  fn new(data: &'a mut [i32]) -> Self {
    Array1DViewMut { data }
  }

  fn len(&self) -> usize {
    self.data.len()
  }

  fn is_empty(&self) -> bool {
    self.data.is_empty()
  }
}

// Indexing for immutable view
impl<'a> Index<usize> for Array1DView<'a> {
    type Output = i32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

// Indexing for mutable view
impl<'a> Index<usize> for Array1DViewMut<'a> {
    type Output = i32;

    fn index(&self, index: usize) -> &Self::Output {
      &self.data[index]
    }
}

impl<'a> IndexMut<usize> for Array1DViewMut<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
      &mut self.data[index]
    }
}

struct Array2D {
  array: Vec<i32>,
  dim: (usize, usize),
}

impl Array2D {
  fn new(rows: usize, cols: usize, fill_value: i32) -> Array2D {
    let size = rows * cols;

    Array2D {
      array: vec![fill_value; size],
      dim: (rows, cols),
    }
  }

  fn identity(n: usize) -> Array2D {
    let mut array = vec![0; n*n];
    for i in 0..n {
      array[i*n + i] = 1;
    }
    Array2D {
      array: array,
      dim: (n, n),
    }
  }

  fn dim(&self) -> (usize, usize) {
    self.dim
  }

  fn get(&self, row: usize, col: usize) -> Option<&i32> {
    if row < self.dim.0 && col < self.dim.1 {
      return Some(&self.array[row * self.dim.1 + col]);
    }
    None
  }

  fn set(&mut self, row: usize, col: usize, value: i32) {
    if row >= self.dim.0 || col >= self.dim.1 {
      return;
    }

    self.array[row*self.dim.1 + col] = value;
  }

  fn print(&self) {
    let mut res = "[".to_string();
    for row in 0..self.dim.0 {
      res += "[ ";
      for col in 0..self.dim.1 {
        res += &self.array[row*self.dim.1 + col].to_string();
        res += " ";
      }
      res += "]";
      if row != self.dim.0 - 1 {
        res += "\n";
        res += " ";
      }
    
    }
    res += "]";
    
    println!("{}", res);
  }

}


// Indexing for Array2D (immutable access)
impl Index<usize> for Array2D {
  type Output = &[i32];

  fn index(&self, row: usize) -> &Self::Output {
    let start = row * self.dim.1;
    let end = start + self.dim.1;
    &self.array[start..end]
  }
}

impl Index<(usize, usize)> for Array2D {
  type Output = i32;

  fn index(&self, index: (usize, usize)) -> &Self::Output {
    &self.array[index.0 * self.dim.1 + index.1]
  }
}

// Indexing for Array2D (mutable access)
impl IndexMut<usize> for Array2D {
  fn index_mut(&mut self, row: usize) -> &mut Self::Output {
    let start = row * self.dim.1;
    let end = start + self.dim.1;
    &mut self.array[start..end]
  }
}

impl Mul<i32> for Array2D {
  type Output = Array2D;

  fn mul(self, scalar: i32) -> Self::Output {
    let new_array = self.array.iter().map(|&x| x * scalar).collect();
    Array2D {
      array: new_array,
      dim: self.dim,
    }
  }
}

impl Mul for Array2D {
  type Output = Array2D;

  fn mul(self, rhs: Array2D) -> Self::Output {
    if self.dim.1 != rhs.dim.0 {
      panic!("Array2D dimensions do not match for multiplication.");
    }

    let mut result = Array2D::new(self.dim.0, rhs.dim.1, 0);

    for col in 0..rhs.dim.1 {
      for row in 0..self.dim.0 {
        let mut dot = 0;
        for i in 0..self.dim.1 {
          dot += self.array[row * self.dim.1 + i] * rhs.array[i * rhs.dim.1 + col];
        }
        result.set(row, col, dot);
      }
    }

    return result;
  }
}


fn main() {


}
