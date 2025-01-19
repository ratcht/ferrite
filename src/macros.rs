#[macro_export]
macro_rules! layer {
  // The pattern: we match a single expression (`$x:expr`).
  ($($x:tt)*) => {
      // The expansion: we generate code that prints out the expression.
      Box::new(Layer::$($x)*)
  };
}
