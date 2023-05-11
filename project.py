import numpy as np 
# pip install numpy 
import matplotlib.pyplot as plt
# pip install matplotlib

# Load the data from the file
data = np.genfromtxt('bike_sharing_data.txt', delimiter=',', skip_header=1)

# Extract the input (population) and output (profit) variables
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# Normalize the input variables
X = (X - np.mean(X)) / np.std(X)

# Add a column of ones to the input variables to represent the intercept term
X = np.hstack((np.ones_like(X), X))

# Initialize the parameters to zero
theta = np.zeros((2, 1))

# Define the learning rate and number of iterations
alpha = 0.01
num_iters = 1500

# Define the cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    J = (1 / (2 * m)) * np.sum((h - y)**2)
    return J

# Define the gradient descent function
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = X.dot(theta)
        theta = theta - (alpha / m) * X.T.dot(h - y)
        J_history[i] = compute_cost(X, y, theta)
    return theta, J_history

# Run gradient descent to learn the parameters
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)

# Print the learned parameters
print("theta: ", theta.ravel())

# Plot the data and the learned linear regression line
plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], X.dot(theta), 'r')
plt.xlabel('Population (normalized)')
plt.ylabel('Profit')
plt.show()

# Plot the cost function over the iterations
plt.plot(J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
