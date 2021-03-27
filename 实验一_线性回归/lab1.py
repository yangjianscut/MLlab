from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import  train_test_split
import numpy as np
import matplotlib.pyplot as plt
#loda data from svmlight format file
def get_data():
    data = load_svmlight_file("housing_scale.txt")
    return data[0], data[1]

#solve vector W use closed form
def solve_W_closed_form(X, Y):
    Y = np.array(Y)
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return W

#calculate loss function
def calc_loss(X, y, W):
    loss = 0.5*(np.linalg.norm(y-X.dot(W))**2)
    return loss

#solve W  vector use gradient descent
def solve_W_gradient_descent(X, Y, learning_rate, epoch):
    loss = []
    Y = Y.reshape(Y.shape[0], 1)
    W = np.random.rand(X.shape[1], 1)
    for i in range(epoch):
        gradient = -X.T.dot(Y) + X.T.dot(X).dot(W)
        # normnazie
        gradient = gradient*(1/X.shape[0])
        W = W - learning_rate*gradient
        epoch_loss = calc_loss(X, Y, W)
        loss.append(epoch_loss)
    print(loss)
    return W,loss

#solve W vector use stochastic gradient descent
def solve_W_stochastic_gradient_descent(X, Y, learning_rate, epoch, batch):
    loss = []
    Y = Y.reshape(Y.shape[0], 1)
    W = np.random.rand(X.shape[1], 1)
    for i in range(epoch):
        bat = np.random.choice(X.shape[0], batch)
        X_batch = X[bat]
        Y_batch = Y[bat]
        gradient = -X_batch.T.dot(Y_batch) + X_batch.T.dot(X_batch).dot(W)
        gradient = gradient * (1/batch)
        # normnazie
        W = W - learning_rate * gradient
        epoch_loss = calc_loss(X, Y, W)
        loss.append(epoch_loss)
    print(loss)
    return W, loss

X, y = get_data()
X = X.toarray()
X = np.hstack((np.ones((X.shape[0], 1)), X))
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
W_train = solve_W_closed_form(X_train, y_train)
print("closed form solution loss:")
print("train_loss: ")
print(calc_loss(X_train, y_train, W_train))
print("test_loss: ")
print(calc_loss(X_test, y_test, W_train))

W_gra, loss_gra = solve_W_gradient_descent(X_train, y_train, 0.1, 1000)
plt.figure()
plt.plot(loss_gra)
#plt.show()
print(W_train)
print(W_gra)
print(calc_loss(X_train, y_train, W_gra.reshape((W_gra.shape[0],))))
W_sto, loss_sto = solve_W_stochastic_gradient_descent(X_train, y_train, 0.1, 100, 5)
plt.plot(loss_sto)
plt.show()
