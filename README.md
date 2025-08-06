# Neural Network from Scratch for Nonlinear Classification

This project implements a **two-layer neural network** built entirely from scratch using Python and NumPy. The network is trained to classify points from a complex, non-linearly separable synthetic dataset (e.g., the "moons" dataset).

## Project Overview

- Manually implemented **forward propagation**, **backward propagation**, and **gradient descent** to train the network.
- Uses a hidden layer with **tanh activation** and an output layer with **sigmoid activation** for binary classification.
- Includes vectorized operations for efficient computation on multiple examples simultaneously.
- Uses cross-entropy loss to quantify prediction accuracy during training.
- Visualizes the decision boundary to show how well the model separates classes.
- Achieves high accuracy (~98%) on the moons dataset, illustrating the advantage of neural networks over linear classifiers.

## Features

- Neural network implemented from scratch—no high-level deep learning libraries used.
- Clear, modular code for all key steps: initialization, forward pass, cost calculation, backward pass, and parameter updates.
- Visualizes classification boundaries to help understand model decision-making.
- Prints training loss intermittently to monitor learning progress.

## Installation

Make sure you have the following Python packages installed:

pip install numpy matplotlib scikit-learn


## How to Run

1. Clone the repository or download the code files.
2. Run the Python script or Jupyter Notebook containing the neural network code.
3. Observe the printed training cost every 1000 iterations (if enabled).
4. A plot window will show the decision boundary learned by the network.
5. Training accuracy will be printed to the console.

## Usage Example

Train the model
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

Plot decision boundary
plot_decision_boundary(lambda x: predict(parameters, x), X, Y)

Predict on training data
preds = predict(parameters, X)
accuracy = 100 * np.mean(preds == Y)
print(f"Training accuracy: {accuracy:.2f}%")

## What You Will Learn

- Fundamentals of neural networks and how they generalize logistic regression.
- Importance of nonlinear activation functions such as tanh and sigmoid.
- How to compute gradients and update parameters using backpropagation and gradient descent.
- Practical experience training a neural network from scratch on a nonlinear classification problem.
- Visualization and interpretation of decision boundaries.

## License

This project is licensed under the MIT License.

---

Feel free to contact me if you have questions or feedback!

