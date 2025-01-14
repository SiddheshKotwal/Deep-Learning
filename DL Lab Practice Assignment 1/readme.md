# Neural Network from Scratch

This project implements a simple feedforward neural network from scratch in Python. The network is trained on the XOR dataset, which is a classic example in machine learning. The aim is to demonstrate basic concepts such as forward pass, backpropagation, and training using gradient descent.

## Table of Contents
1. [Objective](#objective)
2. [Neural Network Architecture](#neural-network-architecture)
3. [Training Details](#training-details)
4. [Methodology](#methodology)
    - [Forward Pass](#forward-pass)
    - [Backpropagation](#backpropagation)
    - [Loss Function](#loss-function)
    - [Optimization](#optimization)
5. [Results](#results)
6. [Visualizations](#visualizations)
7. [Requirements](#requirements)
8. [Usage](#usage)

## Objective

The primary goal of this assignment is to implement a feedforward neural network from scratch without relying on any deep learning libraries. The focus is on understanding the internal mechanics of neural networks, such as:
- Forward pass: Propagation of input data through the network layers.
- Backpropagation: Calculation of gradients and the update of weights using gradient descent.
- Training: Iterative process to adjust weights and biases to minimize the loss function.

# Neural Network Architecture

- **Input Layer**: 2 neurons (representing the two features of the XOR dataset)
- **Hidden Layer**: 4 neurons with the Sigmoid activation function
- **Output Layer**: 1 neuron with the Sigmoid activation function (for binary classification)

## Activation Functions

- **Sigmoid**: This activation function is used in both the hidden and output layers. It maps inputs to a range between 0 and 1, making it suitable for binary classification problems.
  
  The formula for Sigmoid is:
  \[
  \sigma(x) = \frac{1}{1 + e^{-x}}
  \]

# Training Details

## Loss Function

- **Mean Squared Error (MSE)** is used to measure the difference between the predicted and actual outputs:
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_{\text{true}} - y_{\text{pred}})^2
  \]
  Where:
  - \( y_{\text{true}} \) is the actual output
  - \( y_{\text{pred}} \) is the predicted output

## Optimization

- **Gradient Descent** is used to minimize the MSE loss function. The weights and biases are updated iteratively using the gradients computed during backpropagation.

- **Learning Rate**: A learning rate of 0.1 is used to control the size of the steps taken during optimization.

- **Epochs**: The model is trained for 1000 epochs, during which the weights and biases are adjusted to minimize the loss function.

# Methodology

## Forward Pass

- In the forward pass, the input data is propagated through the network layers:
  1. **Hidden Layer**: The weighted sum of inputs is calculated, followed by applying the Sigmoid activation function.
  2. **Output Layer**: The weighted sum of hidden layer outputs is calculated, followed by the application of the Sigmoid activation function to produce the final prediction.

## Backpropagation

Backpropagation is used to adjust the weights and biases:

1. **Error Calculation**: The error is calculated as the difference between the predicted and true output values.
2. **Gradient Calculation**: The gradient of the error with respect to the weights and biases is calculated using the chain rule. This helps in determining how much each weight contributed to the error.
3. **Weight Update**: The weights and biases are updated using the computed gradients and a predefined learning rate.

# Loss Function

The **Mean Squared Error (MSE)** is chosen as the loss function:

\[
\text{MSE} = \frac{1}{n} \sum (y_{\text{true}} - y_{\text{pred}})^2
\]

Where:
- \( y_{\text{true}} \) is the actual output
- \( y_{\text{pred}} \) is the predicted output

This loss function is used to evaluate how well the network is performing and guide the optimization process.

# Optimization

**Gradient Descent** is used to optimize the neural network. The weights are updated using the gradient of the loss with respect to the weights. The update rule is:

\[
w_{\text{new}} = w_{\text{old}} - \eta \cdot \nabla L(w)
\]

Where:
- \( w_{\text{old}} \) is the old weight
- \( w_{\text{new}} \) is the updated weight
- \( \eta \) is the learning rate
- \( \nabla L(w) \) is the gradient of the loss function with respect to the weights

# Results

- **Training Performance**: The neural network successfully learns to solve the XOR problem. Initially, the loss is high, but over 1000 epochs, the network's performance improves as the loss decreases.
  
- **Final Accuracy**: The network achieves high accuracy, correctly classifying the XOR inputs.

- **Visualization of Loss Curve**: The training loss decreases over time, which shows that the network is learning to minimize the error.

At the end of training, the neural network is able to correctly classify all XOR input combinations.

## Visualizations

1. **Dataset Visualization**: A scatter plot of the XOR dataset, where each point is colored according to its corresponding output value.
   
2. **Loss Curve**: A plot that shows the Mean Squared Error (MSE) over the course of training. It demonstrates how the network's performance improves as the training progresses.

3. **Decision Boundary**: A visualization of the decision boundary learned by the network. This shows how the neural network classifies the XOR input space.

## Requirements

- **Python 3.7+**: The script is compatible with Python 3.7 or later.
- **NumPy**: A library for numerical operations.
- **Matplotlib**: A library for visualizations (plots).

## Usage

To run the script:
1. Clone or download the repository.
2. Ensure you have Python 3.7+ installed.
3. Install dependencies:
   ```bash
   pip install numpy matplotlib
4. Run the Jupyter notebook to train the neural network, visualize the results, and evaluate the model's performance.

## Acknowledgments

- Thanks to the instructors for providing this exercise that allows a deeper understanding of neural network fundamentals.
