use std::rc::Rc;
use std::cell::RefCell;
use crate::tensor_storage::*;
use crate::autograd::*;
use std::collections::HashSet;

#[derive(Clone)]
pub struct Tensor {
  tensor: TensorStorage,
  requires_grad: bool,
  grad_fn: Option<Rc<dyn GradientFunction>>,
  grad: Option<Rc<RefCell<TensorStorage>>>,
}

impl Tensor {
  pub fn new(tensor: TensorStorage, requires_grad: bool) -> Self {
    let grad = if requires_grad {
      Some(Rc::new(RefCell::new(TensorStorage::zeros(tensor.shape().clone(), None))))
    } else {
      None
    };
    
    Tensor {
      tensor,
      requires_grad,
      grad_fn: None,
      grad,
    }
  }

  pub fn view(&self, tensor: TensorStorage) -> Self {
    Tensor {
      tensor: tensor,
      requires_grad: self.requires_grad,
      grad_fn: self.grad_fn.clone(),
      grad: self.grad.clone()
    }
  }

  pub fn tensor(&self) -> &TensorStorage {
    &self.tensor
  }

  pub fn tensor_mut(&mut self) -> &mut TensorStorage {
    &mut self.tensor
  }

  pub fn requires_grad(&self) -> &bool {
    &self.requires_grad
  }

  pub fn grad_fn(&self) -> Option<Rc<dyn GradientFunction>> {
    self.grad_fn.clone()
  }

  pub fn set_grad_fn(&mut self, grad_fn: Option<Rc<dyn GradientFunction>>) {
    self.grad_fn = grad_fn;
  }

  pub fn grad(&self) -> Option<Rc<RefCell<TensorStorage>>> {
    self.grad.clone()
  }

  pub fn shape(&self) -> &Vec<usize> {
    &self.tensor().shape()
  }

  pub fn backward(&mut self) {
    // Verify we're starting with a scalar
    if self.tensor().shape().len() != 1 || self.tensor().shape()[0] != 1 {
      panic!("backward() can only be called on scalar tensors");
    }

    // Initialize gradient for final output (always 1.0 for scalar outputs)
    if let Some(grad) = &self.grad {
      grad.borrow_mut().set_data(vec![1.0]);
    } else {
      panic!("Called backward on tensor that doesn't require grad");
    }

    // Build computation graph in topological order
    let mut topo = Vec::new();
    let mut visited = HashSet::new();

    fn build_topo(
      node: &Tensor, 
      topo: &mut Vec<Rc<dyn GradientFunction>>, 
      visited: &mut HashSet<*const dyn GradientFunction>
    ) {
      if let Some(grad_fn) = &node.grad_fn {
        let ptr = Rc::as_ptr(grad_fn) as *const dyn GradientFunction;
        if !visited.contains(&ptr) {
          visited.insert(ptr);
          for parent in grad_fn.prev() {
            build_topo(parent, topo, visited);
          }
          topo.push(grad_fn.clone());
        }
      }
    }

    build_topo(self, &mut topo, &mut visited);

    // Execute backward passes in reverse order
    for grad_fn in topo.iter().rev() {
      grad_fn.backward();
    }
  }
}