import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
        self.accuracy = []
        self.predictions = []

    # sigmooid function -> maps values from 0 - 1
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features) * 0.1 
        self.bias = 0

        # performing gradient descent
        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(z)

            # calculate gradients from log-likelihood function 
            # grad w.r.t. weight
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # grad w.r.t. bias
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Binary Cross-Entropy Loss https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression
            #                                   dont want log of 0
            loss = -np.mean(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
            self.losses.append(loss)

            predictions = (y_predicted >= 0.5).astype(int)  # Convert probabilities to binary predictions
            accuracy = np.mean(predictions == y)  
            self.accuracy.append(accuracy) 

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(z)
        return (y_predicted >= 0.5).astype(int)
    
# testing with iris dataset

iris = load_iris()
X = iris.data[:, :2] 
y = iris.target

# only classes 0 and 1
mask = y != 2  
X = X[mask]
y = y[mask]

# split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize the datasets, using StandardScaler from sklearn with mean and std
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to binary classification: iris-setosa (1) vs. non-setosa (0)
y_train_binary = (y_train == 0).astype(int)  # 1 if Iris-setosa, else 0
y_test_binary = (y_test == 0).astype(int)

# create and train model
model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

# plot that stuff
plt.plot(model.losses, label='Loss')
plt.plot(model.accuracy, label='Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.title('Loss and Accuracy over Iterations')
plt.legend()
plt.show()



