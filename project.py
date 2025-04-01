import numpy as np
import pandas as pd
import matplotlib as plt

df = pd.read_csv('train.csv')

# print(df.head())

data = np.array(df)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():
    w1 = np.random.rand(10, X_train.shape[0])
    b1 = np.random.rand(10, 1)
    w2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Stability improvement
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2
    
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def ReLU_derive(Z):
    return Z > 0

def backward(z1, a1, z2, a2, w1, w2, X, Y):
    one_hot_Y = one_hot(Y)
    dz2 = a2 - one_hot_Y
    dw2 = 1 / X.shape[1] * dz2.dot(a1.T)
    db2 = 1 / X.shape[1] * np.sum(dz2, keepdims=True, axis=1)
    dz1 = w2.T.dot(dz2) * ReLU_derive(z1)
    dw1 = 1 / X.shape[1] * dz1.dot(X.T)
    db1 = 1 / X.shape[1] * np.sum(dz1, keepdims=True, axis=1)
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1    
    w2 = w2 - alpha * dw2  
    b2 = b2 - alpha * db2    
    return w1, b1, w2, b2
    
def get_predictions(a2):
    return np.argmax(a2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = backward(z1, a1, z2, a2, w1, w2, X, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(w2)
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(predictions, Y))
    return w1, b1,w2, b2

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)