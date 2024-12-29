# Oxidize: A Deep Learning Library in Rust

A deep learning framework written in pure Rust, inspired by PyTorch. Used this to learn Rust and refine DL concepts.

## Features

- **Dynamic Computational Graph**: Build and modify neural networks on the fly
- **Automatic Differentiation**: Automatic computation of gradients through the `backward()` method
- **Efficient Tensor Operations**: Fast operations with broadcasting support
- **Memory Safety**: Leveraging Rust's ownership model for safe and efficient memory management
- **Rich Tensor API**: Comprehensive set of tensor operations including:
  - Element-wise operations (add, multiply, divide)
  - Matrix operations (matmul)
  - Reduction operations (sum, mean, product)
  - Shape manipulation (reshape, transpose, squeeze/unsqueeze)
  - Broadcasting support

## Quick Start

```rust
use oxidize::*;
use ndarray::array;

fn main() {
    // Create tensors with gradient tracking
    let x = Tensor::from_ndarray(&array![[1,2,3],[4,5,6]], Some(true));
    let y = Tensor::from_ndarray(&array![[1,1,1]], Some(true));

    // Perform operations
    let z = x.mul_tensor(&y);
    let mut f = z.sum();

    // Compute gradients
    f.backward();

    // Access gradients
    println!("grad x: {:?}", x.grad());
    println!("grad y: {:?}", y.grad());
}
```

## Architecture

Oxidize is built with a modular architecture:

- **TensorStorage**: Core tensor storage and operations
- **Tensor**: High-level tensor interface with autograd support
- **Module**: Base trait for neural network modules
- **Autograd**: Automatic differentiation engine
- **Parameter**: Trainable parameters for neural networks

## Implementation Details

- Uses `Rc<RefCell<>>` for shared ownership and interior mutability
- Implements efficient broadcasting with stride-based computation
- Supports n-dimensional tensors with arbitrary shape
- Provides complete automatic differentiation for supported operations
- Uses traits for clean abstraction of tensor operations

## Usage Examples

### Creating Tensors

```rust
// Create tensor filled with zeros
let zeros = Tensor::zeros(vec![2, 3], Some(true));

// Create tensor from ndarray
let data = Tensor::from_ndarray(&array![[1.0, 2.0], [3.0, 4.0]], Some(true));
```

### Neural Network Module

```rust
impl SimpleNetwork {
    fn new() -> Self {
        let module = Module::new();
        SimpleNetwork { module }
    }
}

impl Segment for SimpleNetwork {
    fn forward(input: Tensor) -> Tensor {
        // Implement your network logic here
    }
}
```

## Future Plans

- [ ] Finish building Neural Network interface
- [ ] Add more operations
- [ ] Add CUDA support
- [ ] Optimize performance for large tensors
- [ ] Add more loss functions
- [ ] Implement data loading utilities
- [ ] Add serialization support


## Acknowledgments

Inspired by:
- PyTorch
- Claude (for teaching me Rust)
