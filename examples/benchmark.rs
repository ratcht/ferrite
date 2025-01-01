use std::time::Instant;

#[link(name = "openblas")]
extern "C" {
  fn cblas_sgemm(
    layout: u8, transa: u8, transb: u8, 
    m: i32, n: i32, k: i32,
    alpha: f32,
    a: *const f32, lda: i32,
    b: *const f32, ldb: i32,
    beta: f32,
    c: *mut f32, ldc: i32
  );
}

// Constants for BLAS
const CBLAS_ROW_MAJOR: u8 = 101;
const CBLAS_NO_TRANS: u8 = 111;

// Naive matrix multiplication implementation
fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
  let mut c = vec![0.0; m * n];
  
  for i in 0..m {
    for j in 0..n {
      let mut sum = 0.0;
      for l in 0..k {
        sum += a[i * k + l] * b[l * n + j];
      }
      c[i * n + j] = sum;
    }
  }
  
  c
}

// BLAS matrix multiplication wrapper
fn blas_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
  let mut c = vec![0.0; m * n];
  
  unsafe {
    cblas_sgemm(
      CBLAS_ROW_MAJOR,
      CBLAS_NO_TRANS,
      CBLAS_NO_TRANS,
      m as i32,
      n as i32,
      k as i32,
      1.0,
      a.as_ptr(),
      k as i32,
      b.as_ptr(),
      n as i32,
      0.0,
      c.as_mut_ptr(),
      n as i32
    );
  }
  
  c
}

fn main() {
  // Test different matrix sizes
  let sizes = vec![(10, 10, 10), (100, 100, 100), (500, 500, 500), (1000, 1000, 1000)];
  
  for (m, k, n) in sizes {
    println!("\nBenchmarking {}x{} * {}x{} matrices:", m, k, k, n);
    
    // Generate random matrices
    let a: Vec<f32> = (0..m*k).map(|_| rand::random::<f32>()).collect();
    let b: Vec<f32> = (0..k*n).map(|_| rand::random::<f32>()).collect();
    
    // Benchmark naive implementation
    let start = Instant::now();
    let _c_naive = naive_matmul(&a, &b, m, k, n);
    let naive_time = start.elapsed();
    
    // Benchmark BLAS implementation
    let start = Instant::now();
    let _c_blas = blas_matmul(&a, &b, m, k, n);
    let blas_time = start.elapsed();
    
    println!("Naive implementation: {:?}", naive_time);
    println!("BLAS implementation: {:?}", blas_time);
    println!("Speedup: {:.2}x", naive_time.as_secs_f64() / blas_time.as_secs_f64());
  }
}