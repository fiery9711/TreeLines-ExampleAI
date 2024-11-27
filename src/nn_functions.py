import numpy as np
from random import randint, choice, shuffle
from constants import *
from shape_generator import load

def linear(x, W, b):
    return x @ W + b

def relu(x):
    return np.maximum(0, x)

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

def simple_train(data):
    W1 = np.random.rand(INTPUT_LAYER, HIDDEN_LAYER)
    b1 = np.random.rand(1, HIDDEN_LAYER)
    W2 = np.random.rand(HIDDEN_LAYER, OUTPUT_LAYER)
    b2 = np.random.rand(1, OUTPUT_LAYER)

    W1 = (W1 - 0.5) * 2 * np.sqrt(1/INTPUT_LAYER)
    b1 = (b1 - 0.5) * 2 * np.sqrt(1/INTPUT_LAYER)
    W2 = (W2 - 0.5) * 2 * np.sqrt(1/HIDDEN_LAYER)
    b2 = (b2 - 0.5) * 2 * np.sqrt(1/HIDDEN_LAYER)
    epoch_loss = [] 

    for epoch in range(EPOCHS):
        shuffle(data)
        for x, yk in data:
            # forward
            y = classify(yk)
            t = linear(x, W1, b1)
            h = relu(t)
            u = linear(h, W2, b2)
            z = softmax(u)
            E = CCE(z, y)

            # backward
            dE_du = z - y
            dE_dW2 = h.T @ dE_du
            dE_db2 = np.sum(dE_du, axis=0, keepdims=True)
            dE_dh = dE_du @ W2.T
            dE_dt = dE_dh * relu_derivative(t)
            dE_dW1 = np.array(x, ndmin=2).T @ np.array(dE_dt, ndmin=2)
            dE_db1 = np.sum(dE_dt, axis=0, keepdims=True)
            
            # update
            W1 = W1 - LEARNING_RATE * dE_dW1
            W2 = W2 - LEARNING_RATE * dE_dW2
            b1 = b1 - LEARNING_RATE * dE_db1
            b2 = b2 - LEARNING_RATE * dE_db2

            # printing, plotting
            epoch_loss.append(E)
        # if epoch % EVERY_EPOCH == 0:
        #     print(f"simple epoch: {epoch}, loss: {E:4.3f}")
    return E, (epoch, W1, b1, W2, b2), epoch_loss

def full_train(data):
    count = len(data)
    W1 = np.random.rand(INTPUT_LAYER, HIDDEN_LAYER)
    b1 = np.random.rand(1, HIDDEN_LAYER)
    W2 = np.random.rand(HIDDEN_LAYER, OUTPUT_LAYER)
    b2 = np.random.rand(1, OUTPUT_LAYER)

    W1 = (W1 - 0.5) * 2 * np.sqrt(1/INTPUT_LAYER)
    b1 = (b1 - 0.5) * 2 * np.sqrt(1/INTPUT_LAYER)
    W2 = (W2 - 0.5) * 2 * np.sqrt(1/HIDDEN_LAYER)
    b2 = (b2 - 0.5) * 2 * np.sqrt(1/HIDDEN_LAYER)
    epoch_loss = []  
    best_loss = 10000
    best = (0, W1, b1, W2, b2)
    for epoch in range(EPOCHS):
        shuffle(data)
        dW1 = np.zeros((INTPUT_LAYER, HIDDEN_LAYER))
        dW2 = np.zeros((HIDDEN_LAYER, OUTPUT_LAYER))
        db1 = np.zeros((1, HIDDEN_LAYER))
        db2 = np.zeros((1, OUTPUT_LAYER))
        E = 0
        for x, yk in data:
            
            # forward
            y = classify(yk)
            t = linear(x, W1, b1)
            h = relu(t)
            u = linear(h, W2, b2)
            z = softmax(u)
            E += CCE(z, y)

            # backward
            dE_du = z - y
            dE_dW2 = h.T @ dE_du
            dE_db2 = np.sum(dE_du, axis=0, keepdims=True)
            dE_dh = dE_du @ W2.T
            dE_dt = dE_dh * relu_derivative(t)
            dE_dW1 = np.array(x, ndmin=2).T @ np.array(dE_dt, ndmin=2)
            dE_db1 = np.sum(dE_dt, axis=0, keepdims=True)
            
            dW1 += dE_dW1
            dW2 += dE_dW2
            db1 += dE_db1
            db2 += dE_db2

        dW1 /= count
        dW2 /= count
        db1 /= count
        db2 /= count

        # update
        W1 = W1 - LEARNING_RATE * dW1
        W2 = W2 - LEARNING_RATE * dW2
        b1 = b1 - LEARNING_RATE * db1
        b2 = b2 - LEARNING_RATE * db2
        E /= count
        minE = min(best_loss, E)
        if minE < best_loss:
            best_loss = minE
            best = (epoch, W1, b1, W2, b2)
        # printing, plotting
        epoch_loss.append(E)
        # if epoch % EVERY_EPOCH == 0:
        #     print(f"full epoch: {epoch}, loss: {E:4.3f}")
    return E, best, epoch_loss

def batch_train(data):
    count = len(data)
    batch_len = count // BATCH_SIZE
    W1 = np.random.rand(INTPUT_LAYER, HIDDEN_LAYER)
    b1 = np.random.rand(1, HIDDEN_LAYER)
    W2 = np.random.rand(HIDDEN_LAYER, OUTPUT_LAYER)
    b2 = np.random.rand(1, OUTPUT_LAYER)

    W1 = (W1 - 0.5) * 2 * np.sqrt(1/INTPUT_LAYER)
    b1 = (b1 - 0.5) * 2 * np.sqrt(1/INTPUT_LAYER)
    W2 = (W2 - 0.5) * 2 * np.sqrt(1/HIDDEN_LAYER)
    b2 = (b2 - 0.5) * 2 * np.sqrt(1/HIDDEN_LAYER)
    epoch_loss = []
    
    best = (0, W1, b1, W2, b2)
    best_loss = 1000
    for epoch in range(EPOCHS):
        shuffle(data)
        minE = 0
        E = 0
        dW1 = np.zeros((INTPUT_LAYER, HIDDEN_LAYER))
        dW2 = np.zeros((HIDDEN_LAYER, OUTPUT_LAYER))
        db1 = np.zeros((1, HIDDEN_LAYER))
        db2 = np.zeros((1, OUTPUT_LAYER))
        for i in range(batch_len):
            
            batch_x, batch_y = zip(*data[i*BATCH_SIZE : i*BATCH_SIZE+BATCH_SIZE])
            x = np.array(batch_x)
            yk = np.array(batch_y)

            # forward
            y = classify_batch(yk, OUTPUT_LAYER)
            t = linear(x, W1, b1)
            h = relu(t)
            u = linear(h, W2, b2)
            z = softmax_batch(u)
            E = np.sum(CCE_batch(z, yk))

            # backward
            dE_du = z - y
            dE_dW2 = h.T @ dE_du
            dE_db2 = np.sum(dE_du, axis=0, keepdims=True)
            dE_dh = dE_du @ W2.T
            dE_dt = dE_dh * relu_derivative(t)
            dE_dW1 = x.T @ dE_dt
            dE_db1 = np.sum(dE_dt, axis=0, keepdims=True)
            dW1 += dE_dW1
            dW2 += dE_dW2
            db1 += dE_db1
            db2 += dE_db2

        dW1 /= BATCH_SIZE
        dW2 /= BATCH_SIZE
        db1 /= BATCH_SIZE
        db2 /= BATCH_SIZE

        E /= BATCH_SIZE
        minE = min(best_loss, E)
        if minE < best_loss:
            best_loss = minE
            best = (epoch, W1, b1, W2, b2)

        # update
        W1 = W1 - LEARNING_RATE * dW1
        W2 = W2 - LEARNING_RATE * dW2
        b1 = b1 - LEARNING_RATE * db1
        b2 = b2 - LEARNING_RATE * db2
        

        # printing, plotting
        epoch_loss.append(E)
        # if epoch % EVERY_EPOCH == 0:
        #     print(f"batch epoch: {epoch}, loss: {E:4.3f}")
    return best_loss, best, epoch_loss

def predict(x, best):
    # forward
    _ , W1, b1, W2, b2 = best
    t = linear(x, W1, b1)
    h = relu(t)
    u = linear(h, W2, b2)
    z = softmax(u)
    return z

def accuracy(best, data):
    corrects = []
    wrongs = []
    correct = 0
    wrong = 0
    size = len(data)
    for x, y in data:
        z = predict(x, best)
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






