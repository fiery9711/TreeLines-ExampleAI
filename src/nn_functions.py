import numpy as np
from nn_constants import *

def linear(x, W, b):
    return x @ W + b

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def soft_relu(x):
    return np.log(1 + np.exp(x))

def soft_relu_derivative(x):
    expx = np.exp(x)
    return expx / (1 + expx)

def activation(x):
    return soft_relu(x)

def activation_derivative(x):
    return soft_relu_derivative(x)

def relu_derivative(x):
    return (x >= 0).astype(float)

def softmax(z):
    expz = np.exp(z)
    return expz / np.sum(expz)

def softmax_batch(z):
    out = np.exp(z)
    return out / np.sum(out, axis=1, keepdims=True)

def CCE_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

def CCE(z, y):
    return -np.sum(y * np.log(z))

def classify(yk):
    y = [0, 0, 0, 0, 0, 0, 0]
    y[yk] = 1
    return np.array(y)

def classify_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full

def random_params():
    W1 = np.random.rand(INTPUT_LAYER, HIDDEN_LAYER)
    b1 = np.random.rand(1, HIDDEN_LAYER)
    W2 = np.random.rand(HIDDEN_LAYER, OUTPUT_LAYER)
    b2 = np.random.rand(1, OUTPUT_LAYER)

    W1 = (W1 - 0.5) * 2 * np.sqrt(1/INTPUT_LAYER)
    b1 = (b1 - 0.5) * 2 * np.sqrt(1/INTPUT_LAYER)
    W2 = (W2 - 0.5) * 2 * np.sqrt(1/HIDDEN_LAYER)
    b2 = (b2 - 0.5) * 2 * np.sqrt(1/HIDDEN_LAYER)
    return W1, b1, W2, b2


def predict(x, params):
    # forward
    W1, b1, W2, b2 = params
    t = linear(x, W1, b1)
    h = relu(t)
    u = linear(h, W2, b2)
    z = softmax(u)
    return z

def accuracy(params, data):
    corrects = []
    wrongs = []
    correct = 0
    wrong = 0
    size = len(data)
    for x, y in data:
        z = predict(x, params)
        yp = np.argmax(z)
        if yp == y:
            correct += 1
            corrects.append(correct)
            wrongs.append(wrong)
        else:
            wrong += 1
            wrongs.append(wrong)
            corrects.append(correct)
    avg = correct / size
    return avg, corrects, wrongs


def mean_loss(loss, epochs=EPOCHS):
    n = len(loss)
    c = n // epochs
    if c == 0:
        c = 1
    r = []
    for i in range(epochs):
        s = 0
        for l in loss[i*c:i*c+c]:
            s += l
        r.append(s if c == 0 else s/c)
    return r
