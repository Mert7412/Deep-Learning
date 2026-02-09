# Deep-Learning

A comprehensive implementation of deep learning architectures and components built from scratch using Python and NumPy. This repository provides educational implementations of various neural network architectures and building blocks, making it useful for understanding how deep learning models work at a fundamental level.

## Overview

This project implements core deep learning components including:
- **AutoGrad**: An automatic differentiation engine for computing gradients
- **Network**: A flexible neural network framework for building models
- Various neural network architectures (FNN, CNN, RNN, LSTM, Transformer)
- Loss functions and activation functions
- Optimization algorithms
- Neural network layers

## Project Structure

```
Deep-Learning/
├── AutoGrad.py              # Automatic differentiation engine
├── Network.py               # Core network class for model building
├── Activations/             # Activation functions
├── Layers/                  # Neural network layer implementations
├── LossFunctions/           # Loss function implementations
├── Optimizers/              # Optimization algorithms
├── FNN/                     # Feedforward Neural Network implementations
├── CNN/                     # Convolutional Neural Network implementations
├── RNN/                     # Recurrent Neural Network implementations
├── LSTM/                    # Long Short-Term Memory implementations
├── Transformer/             # Transformer architecture implementations
├── data/                    # Sample datasets
└── README.md                # This file
```

## Key Components

### AutoGrad.py
Implements an automatic differentiation engine with the `Array` class that supports:
- Basic arithmetic operations (+, -, *, /)
- Matrix multiplication (matmul)
- Statistical operations (mean, variance)
- Element-wise operations (sqrt, transpose)
- Automatic gradient computation through backpropagation

```python
from AutoGrad import Array
import numpy as np

# Create arrays and build computation graph
a = Array(np.array([1.0, 2.0, 3.0]))
b = Array(np.array([4.0, 5.0, 6.0]))
c = a + b
```

### Network.py
Provides a flexible `Network` class for building neural networks:
- Add layers sequentially
- Forward propagation through the network
- Backward propagation for gradient computation
- Save and load model functionality

```python
from Network import Network
from Layers import DenseLayer

network = Network()
network.add_layer(DenseLayer(input_size=784, output_size=128))
network.add_layer(DenseLayer(input_size=128, output_size=10))

# Forward pass
output = network.forward_propagation(input_data)

# Backward pass
loss = network.back_propagation(target, output, loss_function)

# Save model
network.save("my_model")
```

## Features

- **From-Scratch Implementation**: All components are implemented from scratch to aid understanding
- **Educational Focus**: Code is designed to be readable and easy to follow
- **Flexible Architecture**: Easy to compose different layers and models
- **Automatic Differentiation**: Built-in gradient computation
- **Multiple Architectures**: Support for FNN, CNN, RNN, LSTM, and Transformer models

## Getting Started

### Prerequisites
- Python 3.6+
- NumPy

### Installation

```bash
git clone https://github.com/Mert7412/Deep-Learning.git
cd Deep-Learning
```

### Basic Usage

```python
import numpy as np
from AutoGrad import Array
from Network import Network

# Create some sample data
X = np.random.randn(100, 10)
y = np.random.randn(100, 1)

# Initialize network
net = Network()
# Add layers...

# Forward pass
predictions = net.forward_propagation(X)

# Compute loss and backpropagate
loss = net.back_propagation(y, predictions, loss_function)
```

## Architecture Details

### AutoGrad Engine
The `Array` class implements:
- **Gradient accumulation**: Each operation records its children and implements a backward pass
- **Computational graph**: Operations create a directed acyclic graph (DAG) for backpropagation
- **Lazy evaluation**: Gradients are computed on-demand during backward pass

### Supported Neural Network Types
- **FNN (Feedforward Neural Networks)**: Basic multi-layer perceptrons
- **CNN (Convolutional Neural Networks)**: For image processing tasks
- **RNN (Recurrent Neural Networks)**: For sequential data
- **LSTM (Long Short-Term Memory)**: Advanced RNN variant with memory cells
- **Transformer**: Attention-based architecture for modern deep learning

## Contributing

Contributions are welcome! This is an educational project, so please:
1. Keep implementations clear and well-commented
2. Add docstrings to new functions
3. Follow the existing code style
4. Test new features thoroughly

## Learning Resources

This implementation is based on fundamental deep learning concepts:
- Automatic differentiation and backpropagation
- Neural network layer operations
- Loss functions and optimization
- Modern architectures (Transformers, etc.)

## License

This project is open source and available under the MIT License.

## Author

Created by [Mert7412](https://github.com/Mert7412)
