#[link(name = "openblas")] // Replace "openblas" with the library you installed if different
extern "C" {
  fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
  fn cblas_dgemv (Layout: u8, trans: u8, m: i32, n: i32, alpha: f64, a: *const f64, lda: i32, x: *const f64, incx: i32, beta: f64, y: *mut f64, incy: i32);
}

use ndarray::prelude::*;

use ferrite::prelude::*;

// CBLAS_LAYOUT
const CBLAS_ROW_MAJOR: u8 = 101;
const CBLAS_COL_MAJOR: u8 = 102;

// CBLAS_TRANSPOSE
const CBLAS_NO_TRANS: u8 = 111;
const CBLAS_TRANS: u8 = 112;
const CBLAS_CONJ_TRANS: u8 = 113;

fn main() {
  let mut a = Tensor::from_ndarray(&array![[1., 2.], [3., 4.], [5., 6.]], Some(false)); // 3 x 2
  let mut b = Tensor::from_ndarray(&array![[1., 1., 1., 1.], [1., 1., 1., 1.]], Some(false)); // 2 x 4

  println!("A: {}", a);
  println!("B: {}", b);


  let c = a.matmul(&b, false, false);

  println!("C: {}", c);

}