# Ferrite: A Deep Learning Library in Rust

A deep learning framework written in pure Rust, inspired by PyTorch. Used this to learn Rust and refine DL concepts.

## Installation

Add to your project using:
```bash
cargo add ferrite-dl
```

Or manually add to your `Cargo.toml`:
```toml
[dependencies]
ferrite-dl = "0.2.0"
```

## Features

- **Dynamic Computational Graph**: Build and modify neural networks on the fly
- **Automatic Differentiation**: Automatic computation of gradients through the `backward()` method
- **Device Dispatch**: Execute operations on CPU (with CUDA and MPS planned)
- **Efficient Tensor Operations**: Fast operations with broadcasting support
- **Memory Safety**: Leveraging Rust's ownership model for safe and efficient memory management
- **Rich Tensor API**: Comprehensive set of tensor operations including:
  - Element-wise operations (add, subtract, multiply, divide, power)
  - Activation functions (ReLU, Sigmoid, Tanh, ELU, LeakyReLU, Swish)
  - Matrix operations (matmul with BLAS integration)
  - Reduction operations (sum, mean, product)
  - Shape manipulation (reshape, transpose, permute, flatten, squeeze/unsqueeze)
  - Broadcasting support with optimized stride computation

## Quick Start

```rust
use ferrite::prelude::*;
use ndarray::array;

fn main() {
    // Define a Sequential model with two linear layers
    let mut model = Layer::Sequential::new(vec![
        layer!(Linear::new(3, 4, false, Device::Cpu)),
        layer!(Linear::new(4, 2, false, Device::Cpu))
    ]);

    // Define the loss function (Mean Squared Error)
    let loss_fn = Loss::MSELoss::new("mean");

    // Define the optimizer (Stochastic Gradient Descent)
    let optimizer = Optimizer::SGD::new(model.parameters(), 0.01, 0.0);

    // Create input tensor
    let input = Tensor::from_ndarray(&array![[1., 2., 3.], [4., 4., 4.]], Device::Cpu, Some(true));

    // Forward pass
    let output = model.forward(&input);

    // Define the ground truth tensor
    let ground_y = Tensor::from_ndarray(&array![[30., 30.], [50., 50.]], Device::Cpu, Some(false));

    // Compute the loss
    let mut f = loss_fn.loss(&output, &ground_y);

    // Backward pass
    f.backward();

    // Optimization step
    optimizer.step();

    // Print model parameters
    model.print_parameters(true);
}
```

## Architecture

Ferrite is built with a modular architecture:

- **Storage Layer**: Low-level tensor storage implementations with device-specific optimizations
- **Tensor Layer**: High-level tensor interface with autograd support
- **Autograd Engine**: Automatic differentiation with dynamic computational graph
- **Module System**: Composable neural network components
- **Optimization**: Various optimizers for model training
- **Loss Functions**: Multiple loss functions including MSE and MAE

### Implementation Details

- **Efficient Memory Management**: Uses `Arc<RwLock<>>` for thread-safe shared access to tensor data
- **Optimized Operations**: 
  - Stride-based computation for efficient memory access
  - BLAS integration for matrix operations
  - Vectorized operations with broadcasting support
- **Gradient Computation**:
  - Dynamic computation graph construction
  - Automatic backward pass through arbitrary computational graphs
  - Support for complex operations with proper gradient propagation

## Available Components

### Tensor Operations
- Basic arithmetic: add, subtract, multiply, divide
- Matrix operations: matmul
- Advanced operations: power, absolute value
- Broadcasting support for all operations

### Activation Functions
- Binary Step
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- Parametric ReLU
- ELU
- Softmax
- Swish

### Loss Functions
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Cross Entropy (coming soon)

### Optimizers
- Stochastic Gradient Descent (SGD)

### Modules
- Linear Layer
- Sequential Container

## Future Plans

- [ ] Add CUDA and MPS support
- [ ] Implement more optimizers (Adam, RMSprop)
- [ ] Add more loss functions
- [ ] Add convolution operations
- [ ] Implement data loading utilities
- [ ] Add model serialization
- [ ] Improve broadcasting performance
- [ ] Add more neural network layers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- PyTorch (for inspiration)
- Claude (for teaching me Rust)
