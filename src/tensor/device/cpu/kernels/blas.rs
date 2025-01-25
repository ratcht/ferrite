use crate::*;

// CBLAS_LAYOUT
const CBLAS_ROW_MAJOR: u8 = 101;
const CBLAS_COL_MAJOR: u8 = 102;

// CBLAS_TRANSPOSE
const CBLAS_NO_TRANS: u8 = 111;
const CBLAS_TRANS: u8 = 112;
const CBLAS_CONJ_TRANS: u8 = 113;

#[link(name = "openblas")] // Replace "openblas" with the library you installed if different
extern "C" {
  fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
  fn cblas_dgemv(Layout: u8, trans: u8, m: i32, n: i32, alpha: f64, a: *const f64, lda: i32, x: *const f64, incx: i32, beta: f64, y: *mut f64, incy: i32);
  fn cblas_sgemm(Layout: u8, transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: f32, c: *mut f32, ldc: i32);
}

impl BlasOps for CpuStorage {
  fn matmul(&self, other: &Self, transpose_self: bool, transpose_other: bool) -> Self {
    if self.shape().len() != 2 { panic!("Can't Matmul on non-matrices"); }

    // Check dimensions
    if transpose_self && (self.shape()[0] != other.shape()[0]) { 
      panic!("Matrix dimensions do not match for multiplication.");
    } else if transpose_other && (self.shape()[1] != other.shape()[1]) { 
      panic!("Matrix dimensions do not match for multiplication.");
    } else if !transpose_other && !transpose_self && self.shape()[1] != other.shape()[0] { 
      panic!("Matrix dimensions do not match for multiplication.");
    }

    let layout = CBLAS_ROW_MAJOR;
    let trans_a = if transpose_self { CBLAS_TRANS } else { CBLAS_NO_TRANS };
    let trans_b = if transpose_other { CBLAS_TRANS } else { CBLAS_NO_TRANS };
    
    // Get dimensions
    let m = if !transpose_self { self.shape()[0] } else { self.shape()[1] };
    let k = if !transpose_self { self.shape()[1] } else { self.shape()[0] };
    let n = if !transpose_other { other.shape()[1] } else { other.shape()[0] };

    // Get contiguous data
    let (a_data, lda) = self.make_contiguous();
    let (b_data, ldb) = other.make_contiguous();

    let mut c = vec![0.0; (m * n) as usize];
    let ldc = n as i32;

    unsafe {
      cblas_sgemm(
        layout, trans_a, trans_b,
        m as i32, n as i32, k as i32, 1.0,
        a_data.as_ptr(), lda,
        b_data.as_ptr(), ldb, 0.0,
        c.as_mut_ptr(), ldc
      );
    }

    CpuStorage::new(c, vec![m as usize, n as usize])
  }
}