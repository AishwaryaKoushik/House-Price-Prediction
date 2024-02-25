#!/usr/bin/env python
# coding: utf-8

# In[145]:


import numpy as np
import pandas as pd


# Load the dataset

# In[146]:


dataframe = pd.read_csv("/content/Assignment2_q2_dataset.csv")

dataframe.head()
dataframe.shape


# Null columns

# In[147]:


dataframe.isna().sum()


# 

# In[148]:


dataframe.head()


# In[149]:


X = dataframe.loc[:, dataframe.columns != 'Price']
y = dataframe['Price']


# Applying PCA to idenitfy important Features

# In[150]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame with the provided features
# Replace 'your_target_column' with the actual target column in your dataset
# For example, if you have a column 'price', replace 'your_target_column' with 'price'

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot the explained variance ratio
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio.cumsum(), marker='o', linestyle='--')
plt.title('Explained Variance Ratio')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Determine the number of components to retain
# You can choose a threshold, e.g., 95% variance explained
cumulative_variance_threshold = 0.95
num_components = sum(explained_variance_ratio.cumsum() <= cumulative_variance_threshold)

print(f'Number of components to retain: {num_components}')

# Get the principal components and their corresponding feature names
principal_components = pd.DataFrame(pca.components_, columns=X.columns)
top_components = principal_components.iloc[:num_components, :]

# Display the most relevant features for each principal component
print('Top components and their relevant features:')
for i, component in enumerate(top_components.iterrows(), 1):
    print(f'\nPrincipal Component {i}:')
    print(component[1].sort_values(ascending=False).to_string())

# If you want to use the transformed data with reduced dimensions for further analysis, you can use X_pca



# Dropping Irrelevant Features

# In[151]:


X.drop(axis="columns",labels=["Number of schools nearby","Date","condition of the house","Postal Code","id"],inplace=True)

X.isna().sum()


# Neural Network Model with One Hidden Layer

# In[152]:


import sklearn

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(22)

    weights1 = np.random.randn(input_size, hidden_size)
    bias1 = np.zeros((1, hidden_size))

    weights2 = np.random.randn(hidden_size, output_size)
    bias2 = np.zeros((1, output_size))

    return weights1, bias1, weights2, bias2

def relu(x):
    return np.maximum(0, x)

def mean_squared_error(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.values.flatten() if isinstance(y_true, pd.Series) else y_true.flatten()
    return np.mean((y_pred - y_true) ** 2)

def forward_propagation(X, weights1, bias1, weights2, bias2):
    hidden = relu(np.dot(X, weights1) + bias1)
    y_pred = np.dot(hidden, weights2) + bias2
    return hidden, y_pred

def backward_propagation(X, hidden, y_pred, y_true, weights1, weights2):
    num_samples = X.shape[0]
    gradient_y = 2 * (y_pred - y_true) / num_samples

    gradient_hidden = np.dot(gradient_y, weights2.T) * (hidden > 0) * ~np.isnan(hidden)  

    gradient_weights1 = np.dot(X.T, gradient_hidden)
    gradient_bias1 = np.sum(gradient_hidden, axis=0, keepdims=True)

    gradient_weights2 = np.dot(hidden.T, gradient_y)
    gradient_bias2 = np.sum(gradient_y, axis=0, keepdims=True)

    return gradient_weights1, gradient_bias1, gradient_weights2, gradient_bias2

def update_parameters(weights1, bias1, weights2, bias2,
                       gradient_weights1, gradient_bias1, gradient_weights2, gradient_bias2,
                       learning_rate):
    weights1 -= learning_rate * gradient_weights1
    bias1 -= learning_rate * gradient_bias1

    weights2 -= learning_rate * gradient_weights2
    bias2 -= learning_rate * gradient_bias2

    return weights1, bias1, weights2, bias2

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
input_size = X_train.shape[1]
hidden_size = 8
output_size = 1

# Initialize parameters
weights1, bias1, weights2, bias2 = initialize_parameters(input_size, hidden_size, output_size)

# Hyperparameters
learning_rate = 0.001
epochs = 1110

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden, y_pred_train = forward_propagation(X_train, weights1, bias1, weights2, bias2)

    # Ensure y_train is a 2D NumPy array
    y_train_array = y_train.values.reshape(-1, 1) if isinstance(y_train, pd.Series) else y_train.reshape(-1, 1)

    # Calculate the mean squared error
    loss = mean_squared_error(y_pred_train, y_train_array)

    # Backward pass
    gradient_weights1, gradient_bias1, gradient_weights2, gradient_bias2 = \
        backward_propagation(X_train, hidden, y_pred_train, y_train_array, weights1, weights2)

    # Update parameters
    weights1, bias1, weights2, bias2 = \
        update_parameters(weights1, bias1, weights2, bias2,
                           gradient_weights1, gradient_bias1, gradient_weights2, gradient_bias2,
                           learning_rate)

    # Print the training loss for every few epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Testing the trained model
_, y_pred_test = forward_propagation(X_test, weights1, bias1, weights2, bias2)
test_loss = mean_squared_error(y_pred_test, y_test)

print(f"Test Loss: {test_loss}")


# Neural Network Model with two hidden layers

# In[153]:


import sklearn

def initialize_parameters(input_size, hidden1_size, hidden2_size, output_size):
    np.random.seed(42)

    weights1 = np.random.randn(input_size, hidden1_size)
    bias1 = np.zeros((1, hidden1_size))

    weights2 = np.random.randn(hidden1_size, hidden2_size)
    bias2 = np.zeros((1, hidden2_size))

    weights3 = np.random.randn(hidden2_size, output_size)
    bias3 = np.zeros((1, output_size))

    return weights1, bias1, weights2, bias2, weights3, bias3

def relu(x):
    return np.maximum(0, x)

def forward_propagation(X, weights1, bias1, weights2, bias2, weights3, bias3):
    hidden1 = relu(np.dot(X, weights1) + bias1)
    hidden2 = relu(np.dot(hidden1, weights2) + bias2)
    y_pred = np.dot(hidden2, weights3) + bias3
    return hidden1, hidden2, y_pred

def mean_squared_error(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.values.flatten() if isinstance(y_true, pd.Series) else y_true.flatten()
    return np.mean((y_pred - y_true) ** 2)

def backward_propagation(X, hidden1, hidden2, y_pred, y_true,
                         weights1, weights2, weights3):
    num_samples = X.shape[0]
    gradient_y = 2 * (y_pred - y_true) / num_samples

    gradient_weights3 = np.dot(hidden2.T, gradient_y)
    gradient_bias3 = np.sum(gradient_y, axis=0, keepdims=True)

    gradient_hidden2 = np.dot(gradient_y, weights3.T) * (hidden2 > 0)
    gradient_weights2 = np.dot(hidden1.T, gradient_hidden2)
    gradient_bias2 = np.sum(gradient_hidden2, axis=0, keepdims=True)

    gradient_hidden1 = np.dot(gradient_hidden2, weights2.T) * (hidden1 > 0)
    gradient_weights1 = np.dot(X.T, gradient_hidden1)
    gradient_bias1 = np.sum(gradient_hidden1, axis=0, keepdims=True)

    return gradient_weights1, gradient_bias1, gradient_weights2, gradient_bias2, gradient_weights3, gradient_bias3

def update_parameters(weights1, bias1, weights2, bias2, weights3, bias3,
                       gradient_weights1, gradient_bias1, gradient_weights2, gradient_bias2,
                       gradient_weights3, gradient_bias3, learning_rate):
    weights1 -= learning_rate * gradient_weights1
    bias1 -= learning_rate * gradient_bias1

    weights2 -= learning_rate * gradient_weights2
    bias2 -= learning_rate * gradient_bias2

    weights3 -= learning_rate * gradient_weights3
    bias3 -= learning_rate * gradient_bias3

    return weights1, bias1, weights2, bias2, weights3, bias3

X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_scaled, y, test_size=0.2, random_state=42)
input_size = X_train.shape[1]
hidden1_size = 16
hidden2_size = 16
output_size = 1

# Initialize parameters
weights1, bias1, weights2, bias2, weights3, bias3 = initialize_parameters(input_size, hidden1_size, hidden2_size, output_size)

# Hyperparameters
learning_rate = 0.001
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden1, hidden2, y_pred_train = forward_propagation(X_train, weights1, bias1, weights2, bias2, weights3, bias3)

    # Ensure y_train is a 2D NumPy array
    y_train_array = y_train.values.reshape(-1, 1) if isinstance(y_train, pd.Series) else y_train.reshape(-1, 1)

    # Calculate the mean squared error
    loss = mean_squared_error(y_pred_train, y_train_array)

    # Backward pass
    gradient_weights1, gradient_bias1, gradient_weights2, gradient_bias2, gradient_weights3, gradient_bias3 = \
        backward_propagation(X_train, hidden1, hidden2, y_pred_train, y_train_array, weights1, weights2, weights3)

    # Update parameters
    weights1, bias1, weights2, bias2, weights3, bias3 = \
    update_parameters(weights1, bias1, weights2, bias2, weights3, bias3,
                       gradient_weights1, gradient_bias1, gradient_weights2, gradient_bias2,
                       gradient_weights3, gradient_bias3, learning_rate)

    # Print the training loss for every few epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Testing the trained model
_, _, y_pred_test = forward_propagation(X_test, weights1, bias1, weights2, bias2, weights3, bias3)
test_loss = mean_squared_error(y_pred_test, y_test)

print(f"Test Loss: {test_loss}")


# Neural Network Model with no hidden layers

# In[154]:


np.random.seed(42)
num_samples=len(X.axes[0])
num_features=len(X.axes[1])
X_rand = np.random.rand(num_samples, num_features)
true_weights = np.random.rand(num_features, 1)
bias = 3.0
y = np.dot(X_rand, true_weights) + bias + 0.1 * np.random.randn(num_samples, 1)

# Select the specific features for your regression task
X_selected = X

# Normalize the selected features
X_normalized = (X_selected - np.mean(X_selected, axis=0)) / np.std(X_selected, axis=0)

# Split the data into training and testing sets
split_ratio = 0.8
num_train_samples = int(split_ratio * num_samples)

X_train = X_normalized[:num_train_samples]
y_train = y[:num_train_samples]

X_test = X_normalized[num_train_samples:]
y_test = y[num_train_samples:]

# Define hyperparameters
learning_rate = 0.01
epochs = 1000

# Initialize weights and bias
weights = np.random.randn(X_train.shape[1], 1)
bias = np.zeros((1, 1))

# Training the neural network
for epoch in range(epochs):
    # Forward pass
    y_pred = np.dot(X_train, weights) + bias

    # Calculate the mean squared error
    loss = np.mean((y_pred - y_train) ** 2)

    # Backward pass (gradient descent)
    gradient_weights = 2 * np.dot(X_train.T, (y_pred - y_train)) / num_train_samples
    gradient_bias = 2 * np.sum(y_pred - y_train) / num_train_samples

    # Update weights and bias
    weights -= learning_rate * gradient_weights
    bias -= learning_rate * gradient_bias

    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Testing the trained model
y_pred_test = np.dot(X_test, weights) + bias

test_loss = np.mean((y_pred_test - y_test) ** 2)
print(f"Test Loss: {test_loss}")

