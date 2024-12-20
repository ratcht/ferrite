use super::{Graph, Scalar, Value};  // Import Graph from parent module

pub trait Operations<'a> {
  fn add(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_>;
  fn mul(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_>;
  fn div(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_>;
  fn exp(&self, lhs_val: &Value) -> Value<'_>;

}

impl<'a> Operations<'a> for Graph {
  fn add(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_> {
    let lhs = lhs_val.idx;
    let rhs = rhs_val.idx;
    let mut scalars = self.scalars.borrow_mut();
    let data = scalars[lhs].data + scalars[rhs].data;
    let idx = scalars.len();
    let scalar = Scalar::new(data, idx, &[lhs, rhs], "+");
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
    let data = scalars[lhs].data * scalars[rhs].data;
    let idx = scalars.len();
    scalars.push(Scalar::new(data, idx, &[lhs, rhs], "*"));
    Value {
      idx,
      graph: self,
    }
  }

  fn div(&self, lhs_val: &Value, rhs_val: &Value) -> Value<'_> {
    let lhs = lhs_val.idx;
    let rhs = rhs_val.idx;
    let mut scalars = self.scalars.borrow_mut();
    let data = scalars[lhs].data / scalars[rhs].data;
    let idx = scalars.len();
    scalars.push(Scalar::new(data, idx, &[lhs, rhs], "/"));
    Value {
      idx,
      graph: self,
    }
  }

  fn exp(&self, lhs_val: &Value) -> Value<'_> {
    let lhs = lhs_val.idx;
    let mut scalars = self.scalars.borrow_mut();
    let data = scalars[lhs].data.exp();
    let idx = scalars.len();
    scalars.push(Scalar::new(data, idx, &[lhs], "exp"));
    Value {
      idx,
      graph: self,
    }
  }
}