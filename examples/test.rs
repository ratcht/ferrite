#[link(name = "openblas")] // Replace "openblas" with the library you installed if different
extern "C" {
  fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
  fn cblas_dgemv (Layout: u8, trans: u8, m: i32, n: i32, alpha: f64, a: *const f64, lda: i32, x: *const f64, incx: i32, beta: f64, y: *mut f64, incy: i32);
}

use ndarray::prelude::*;

// CBLAS_LAYOUT
const CBLAS_ROW_MAJOR: u8 = 101;
const CBLAS_COL_MAJOR: u8 = 102;

// CBLAS_TRANSPOSE
const CBLAS_NO_TRANS: u8 = 111;
const CBLAS_TRANS: u8 = 112;
const CBLAS_CONJ_TRANS: u8 = 113;

fn main() {
  let layout = CBLAS_ROW_MAJOR;
  let trans = CBLAS_NO_TRANS;
  let a: Vec<f64> = array![[1.,2.,3.],[4.,5.,6.]].flatten().to_vec();
  let m = 2;
  let n = 3;
  let lda = 3;
  let x: Vec<f64> = vec![1.0, 1.0, 1.0];
  let alpha = 1.;
  let beta = 0.;
  let mut y: Vec<f64> = vec![0.0, 0.0];

  unsafe {
    let result = cblas_dgemv(layout, trans, m, n, alpha, a.as_ptr(), lda, x.as_ptr(), 1, beta, y.as_mut_ptr(), 1);
    println!("Dot product: {:?}", y);
  }
}