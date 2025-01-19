#[macro_export]
macro_rules! layer {
  // The pattern: we match a single expression (`$x:expr`).
  ($($x:tt)*) => {
      // The expansion: we generate code that prints out the expression.
      Box::new(Layer::$($x)*)
  };
}

#[macro_export]
macro_rules! reduce_grad {
  ($grad:expr, $shape:expr) => {{
      let mut reduced_grad = $grad.clone();
      for (dim, (grad_size, shape_size)) in $grad.shape().iter().zip($shape.iter()).enumerate() {
          if shape_size == &1 && grad_size != &1 {
              let mut sum_dims = vec![false; $grad.shape().len()];
              sum_dims[dim] = true;
              reduced_grad = reduced_grad.sum_dim(&sum_dims);
          }
      }
      reduced_grad
  }};
}