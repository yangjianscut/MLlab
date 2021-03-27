from sklearn.datasets import load_svmlight_file
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

data = load_svmlight_file("a9a.txt", n_features=123)
X_train = data[0].toarray()
Y_train = data[1].reshape((data[1].shape[0],1))
data1 = load_svmlight_file("a9a.t", n_features=123)
X_test = data1[0].toarray()
Y_test = data1[1].reshape((data1[1].shape[0], 1))

def calc_gradient(X, Y, W, b, c):
    grad_W = np.zeros((W.shape[0], ))
    grad_b = 0
    for i in range(X.shape[0]):
        Xi = X[i]
        Yi = Y[i]
        if Yi*(W.T.dot(Xi) + b) <= 1:
            grad_W = grad_W + (-Yi*Xi)
            grad_b = grad_b + (-Yi)
    #print(grad_W)
    #print("gW2"+str(grad_W.shape))
    #print("gb")
    grad_W = np.reshape(grad_W.shape[0], 1)
    grad_W = W + (c/X.shape[0]) * grad_W
    #print("gW3" + str(grad_W.shape))
    grad_b = (c/X.shape[0]) * grad_b
    return grad_W, grad_b

def calc_loss(X, Y, W, b, c):
    loss = 0
    for i in range(X.shape[0]):
        Xi = X[i]
        Yi = Y[i]
        loss = loss + max(0, 1 - Yi*(W.T.dot(Xi) + b))
    loss = (c/X.shape[0])*loss
    loss = loss + 0.5*(np.linalg.norm(W)**2)
    return loss

def calc_validation(W, b):
    match_count = 0
    for i in range(X_test.shape[0]):
        Xi = X_test[i]
        f = W.T.dot(Xi) + b
        if f > 0:
            flag = 1
        else:
            flag = -1
        if flag == Y_test[i][0]:
            match_count = match_count + 1
    return match_count/X_test.shape[0]

def SVM_classifier(X, Y, c, epoch, batch, learning_rate):
    loss = []
    acc = []
    W = np.random.rand(X.shape[1], 1)
    b = 0
    for i in tqdm(range(epoch), desc="Training pass: ", leave=True):
        bat = np.random.choice(X.shape[0], batch)
        X_batch = X[bat]
        Y_batch = Y[bat]
        gradient_W, gradient_b = calc_gradient(X_batch, Y_batch, W, b, c)
        W = W - learning_rate * gradient_W
        b = b - learning_rate * gradient_b
        loss.append(calc_loss(X, Y, W, b, c))
        acc.append(calc_validation(W, b))
    return W, loss, acc

def SVM_classifier_Adam(X, Y, c, epoch, batch, learning_rate, eps=1e-8, beta1=0.9, beta2=0.999):
    loss = []
    acc = []
    W = np.random.rand(X.shape[1], 1)
    b = 0
    mW = 0
    vW = 0
    mb = 0
    vb = 0
    for i in tqdm(range(epoch), desc="Training pass: ", leave=True):
        bat = np.random.choice(X.shape[0], batch)
        X_batch = X[bat]
        Y_batch = Y[bat]
        gradient_W, gradient_b = calc_gradient(X_batch, Y_batch, W, b, c)
        mW = beta1 * mW + (1 - beta1) * gradient_W
        vW = beta2 * vW + (1 - beta2) * (gradient_W ** 2)
        W = W - learning_rate * mW / (np.sqrt(vW) + eps)
        mb = beta1 * mb + (1 - beta1) * gradient_b
        vb = beta2 * vb + (1 - beta2) * (gradient_b ** 2)
        b = b - learning_rate * mb / (np.sqrt(vb) + eps)
        loss.append(calc_loss(X, Y, W, b, c))
        acc.append(calc_validation(W, b))
    return W, loss, acc

#W, loss, acc = SVM_classifier(X=X_train, Y=Y_train, c=0.001, epoch=100, batch=40, learning_rate=0.1)

W, loss, acc = SVM_classifier_Adam(X=X_train, Y=Y_train, c=0.01, epoch=100, batch=40, learning_rate=0.01)

plt.figure()
plt.plot(acc)
plt.title("acc")
plt.show()
plt.figure()
plt.plot(loss)
plt.title("loss")
plt.show()
