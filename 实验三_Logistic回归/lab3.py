from sklearn.datasets import load_svmlight_file
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

data = load_svmlight_file("a9a.txt", n_features=123)
X_train = data[0].toarray()
Y_train = data[1].reshape((data[1].shape[0],1))
data1 = load_svmlight_file("a9a.t", n_features=123)
X_test = data1[0].toarray()
Y_test = data1[1].reshape((data1[1].shape[0], 1))

def calc_loss(X, Y, W):
    loss = 0
    for i in range(X.shape[0]):
        Xi = X[i]
        Yi = Y[i]
        exp_ywx = math.exp(-Yi*W.T.dot(Xi))
        loss = loss + math.log(1+exp_ywx)
    loss = (1/X.shape[0])*loss
    return loss

def g_func(Z):
    return 1/(1+math.exp(-Z))

def calc_validation(W):
    match_count = 0
    for i in range(X_test.shape[0]):
        Z = W.T.dot(X_test[i])
        f = g_func(Z)
        if f > 0.5:
            flag = 1
        else:
            flag = -1
        if flag == Y_test[i][0]:
            match_count = match_count + 1
    return match_count/X_test.shape[0]

def calc_gradient(X, Y, W, lamda):
    grad = 0
    for i in range(X.shape[0]):
        Xi = X[i]
        Yi = Y[i]
        exp_ywx = math.exp(-Yi*W.T.dot(Xi))
        temp = (Yi*Xi*exp_ywx)/(1+exp_ywx)
        grad = grad + temp
    grad = -(1/X.shape[0])*grad
    grad = grad.reshape(grad.shape[0], 1)
    grad = grad + (lamda*W)
    return grad

def Logistic_Regression(X, Y, learning_rate, lamda, epoch, batch):
    loss = []
    acc = []
    W = np.random.rand(X.shape[1], 1)
    for i in tqdm(range(epoch), desc="Training pass: ", leave=True):
        bat = np.random.choice(X.shape[0], batch)
        X_batch = X[bat]
        Y_batch = Y[bat]
        gradient = calc_gradient(X_batch, Y_batch, W, lamda)
        W = W - learning_rate*gradient
        loss.append(calc_loss(X, Y, W))
        acc.append(calc_validation(W))
    return W, loss, acc

def Logistic_Regression_Adam(X, Y, learning_rate, lamda, epoch, batch, eps=1e-8, beta1=0.9, beta2=0.999):
    loss = []
    acc = []
    m = 0
    v = 0
    W = np.random.rand(X.shape[1], 1)
    for i in tqdm(range(epoch), desc="Training pass: ", leave=True):
        bat = np.random.choice(X.shape[0], batch)
        X_batch = X[bat]
        Y_batch = Y[bat]
        gradient = calc_gradient(X_batch, Y_batch, W, lamda)
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        W = W - learning_rate * m / (np.sqrt(v) + eps)
        loss.append(calc_loss(X, Y, W))
        acc.append(calc_validation(W))
    return W, loss, acc

#W, loss, acc = Logistic_Regression(X_train, Y_train, 0.1, 0.01, 200, 30)
W, loss, acc = Logistic_Regression_Adam(X_train, Y_train, 0.1, 0.01, 200, 30)
plt.figure()
plt.plot(acc)
plt.title("Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Match Percent")
plt.show()
plt.figure()
plt.plot(loss)
plt.title("Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()