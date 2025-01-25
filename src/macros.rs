#[macro_export]
macro_rules! layer {
  // The pattern: we match a single expression (`$x:expr`).
  ($($x:tt)*) => {
      // The expansion: we generate code that prints out the expression.
      Box::new(Layer::$($x)*)
  };
}


#[macro_export]
macro_rules! grad_storage {
  // The pattern: we match a single expression (`$x:expr`).
  ($($x:tt)*) => {
      // The expansion: we generate code that prints out the expression.
      std::rc::Rc::new(std::cell::RefCell::new(Box::new($($x)*)))
  };
}


