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

#[macro_export]
macro_rules! match_storage {
  // Binary operation with two storage arguments
  (binary $self:expr, $method:ident, $other:expr $(, $args:expr)*) => {
    match ($self, $other) {
      (Storage::Cpu(cpu_self), Storage::Cpu(cpu_other)) => {
        Storage::Cpu(cpu_self.$method(cpu_other $(, $args)*))
      }
      _ => unimplemented!("Cross-device operations not supported"),
    }
  };

  // Unary operation with single storage argument
  (unary $self:expr, $method:ident $(, $args:expr)*) => {
    match $self {
      Storage::Cpu(cpu) => Storage::Cpu(cpu.$method($($args),*)),
      _ => unimplemented!("Device not supported"),
    }
  };
}

#[macro_export]
macro_rules! match_storage_assign {
  // Binary operation with two storage arguments
  (binary $self:expr, $method:ident, $other:expr $(, $args:expr)*) => {
    match ($self, $other) {
      (Storage::Cpu(cpu_self), Storage::Cpu(cpu_other)) => {
        cpu_self.$method(cpu_other $(, $args)*)
      }
      _ => unimplemented!("Cross-device operations not supported"),
    }
  };

  // Unary operation with single storage argument
  (unary $self:expr, $method:ident $(, $args:expr)*) => {
    match $self {
      Storage::Cpu(cpu) => cpu.$method($($args,)*),
      _ => unimplemented!("Device not supported"),
    }
  };
}


