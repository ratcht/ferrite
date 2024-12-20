use super::{Graph, Scalar, Value};  // Import Graph from parent module
use std::ops::{Add, Sub, Mul, Div};


pub trait Operations<'a> {
  fn add(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_>;
  fn sub(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_>;
  fn mul(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_>;
  fn div(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_>;
  fn exp(&self, lhs_val: &Value) -> Value<'_>;
  fn pow(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_>;

}

impl<'a> Operations<'a> for Graph {
  fn add(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_> {
    let lhs = lhs_val.idx;
    let rhs = rhs_val.idx;

    let mut scalars = self.scalars.borrow_mut();

    let requires_grad = scalars[lhs].requires_grad || scalars[rhs].requires_grad;
    let data = scalars[lhs].data + scalars[rhs].data;

    let idx = scalars.len();
    let scalar = Scalar::new(data, idx, &[lhs, rhs], "+", requires_grad);
    scalars.push(scalar);
    
    Value {
      idx,
      graph: self,
    }
  }

  fn sub(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_> {
    let lhs = lhs_val.idx;
    let rhs = rhs_val.idx;

    let mut scalars = self.scalars.borrow_mut();

    let requires_grad = scalars[lhs].requires_grad || scalars[rhs].requires_grad;
    let data = scalars[lhs].data - scalars[rhs].data;

    let idx = scalars.len();
    let scalar = Scalar::new(data, idx, &[lhs, rhs], "-", requires_grad);
    scalars.push(scalar);
    
    Value {
      idx,
      graph: self,
    }
  }


  fn mul(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_> {
    let lhs = lhs_val.idx;
    let rhs = rhs_val.idx;

    let mut scalars = self.scalars.borrow_mut();

    let requires_grad = scalars[lhs].requires_grad || scalars[rhs].requires_grad;
    let data = scalars[lhs].data * scalars[rhs].data;
    let idx = scalars.len();
    scalars.push(Scalar::new(data, idx, &[lhs, rhs], "*", requires_grad));
    Value {
      idx,
      graph: self,
    }
  }

  fn div(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_> {
    let lhs = lhs_val.idx;
    let rhs = rhs_val.idx;
    let mut scalars = self.scalars.borrow_mut();

    let requires_grad = scalars[lhs].requires_grad || scalars[rhs].requires_grad;
    let data = scalars[lhs].data / scalars[rhs].data;

    let idx = scalars.len();
    scalars.push(Scalar::new(data, idx, &[lhs, rhs], "/", requires_grad));
    Value {
      idx,
      graph: self,
    }
  }

  fn exp(&self, lhs_val: &Value) -> Value<'_> {
    let lhs = lhs_val.idx;

    let mut scalars = self.scalars.borrow_mut();

    let requires_grad = scalars[lhs].requires_grad;
    let data = scalars[lhs].data.exp();
    let idx = scalars.len();
    scalars.push(Scalar::new(data, idx, &[lhs], "exp", requires_grad));
    Value {
      idx,
      graph: self,
    }
  }

  fn pow(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_> {
    let lhs = lhs_val.idx;
    let rhs = rhs_val.idx;
    let mut scalars = self.scalars.borrow_mut();

    let requires_grad = scalars[lhs].requires_grad || scalars[rhs].requires_grad;
    let data = f32::powf(scalars[lhs].data, scalars[rhs].data);

    let idx = scalars.len();
    scalars.push(Scalar::new(data, idx, &[lhs, rhs], "pow", requires_grad));
    Value {
      idx,
      graph: self,
    }
  }
}


// ========================
// Add Implementation
// ========================

impl<'a> Add<Value<'a>> for Value<'a> {
  type Output = Value<'a>;

  fn add(self, rhs: Self) -> Self::Output {
    self.graph.add(&self, &rhs)
  }
}

impl<'a> Add<f32> for Value<'a> {
  type Output = Value<'a>;

  fn add(self, rhs: f32) -> Self::Output {
    let scalar = self.graph.scalar(rhs, false);
    self.graph.add(&self, &scalar)
  }
}

impl<'a> Add<Value<'a>> for f32 {
  type Output = Value<'a>;

  fn add(self, rhs: Value<'a>) -> Self::Output {
    let scalar = rhs.graph.scalar(self, false);
    rhs.graph.add(&scalar, &rhs)
  }
}

// ========================
// Sub Implementation
// ========================

impl<'a> Sub<Value<'a>> for Value<'a> {
  type Output = Value<'a>;

  fn sub(self, rhs: Self) -> Self::Output {
    self.graph.sub(&self, &rhs)
  }
}

impl<'a> Sub<f32> for Value<'a> {
  type Output = Value<'a>;

  fn sub(self, rhs: f32) -> Self::Output {
    let scalar = self.graph.scalar(rhs, false);
    self.graph.sub(&self, &scalar)
  }
}

impl<'a> Sub<Value<'a>> for f32 {
  type Output = Value<'a>;

  fn sub(self, rhs: Value<'a>) -> Self::Output {
    let scalar = rhs.graph.scalar(self, false);
    rhs.graph.sub(&scalar, &rhs)
  }
}


// ========================
// Mul Implementation
// ========================

impl<'a> Mul<Value<'a>> for Value<'a> {
  type Output = Value<'a>;

  fn mul(self, rhs: Self) -> Self::Output {
    self.graph.mul(&self, &rhs)
  }
}

impl<'a> Mul<f32> for Value<'a> {
  type Output = Value<'a>;

  fn mul(self, rhs: f32) -> Self::Output {
    let scalar = self.graph.scalar(rhs, false);
    self.graph.mul(&self, &scalar)
  }
}

impl<'a> Mul<Value<'a>> for f32 {
  type Output = Value<'a>;

  fn mul(self, rhs: Value<'a>) -> Self::Output {
    let scalar = rhs.graph.scalar(self, false);
    rhs.graph.mul(&scalar, &rhs)
  }
}



// ========================
// Div Implementation
// ========================

impl<'a> Div<Value<'a>> for Value<'a> {
  type Output = Value<'a>;

  fn div(self, rhs: Self) -> Self::Output {
    self.graph.div(&self, &rhs)
  }
}

impl<'a> Div<f32> for Value<'a> {
  type Output = Value<'a>;

  fn div(self, rhs: f32) -> Self::Output {
    let scalar = self.graph.scalar(rhs, false);
    self.graph.div(&self, &scalar)
  }
}

impl<'a> Div<Value<'a>> for f32 {
  type Output = Value<'a>;

  fn div(self, rhs: Value<'a>) -> Self::Output {
    let scalar = rhs.graph.scalar(self, false);
    rhs.graph.div(&scalar, &rhs)
  }
}