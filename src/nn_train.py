import numpy as np
from nn_constants import *
from nn_functions import *
from random import shuffle
from tqdm import tqdm

def simple_train(data):
    W1, b1, W2, b2 = random_params()    
    epoch_loss = [] 

    for epoch in tqdm(range(EPOCHS), desc="Simple train method"):
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

            W1 = W1 - LEARNING_RATE * dE_dW1
            W2 = W2 - LEARNING_RATE * dE_dW2
            b1 = b1 - LEARNING_RATE * dE_db1
            b2 = b2 - LEARNING_RATE * dE_db2

            epoch_loss.append(E)
    
    return E, (W1, b1, W2, b2), epoch_loss

def full_train(data):
    W1, b1, W2, b2 = random_params()  
    count = len(data)
    epoch_loss = []  
    for epoch in tqdm(range(EPOCHS), desc="Full epoch train method"):
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

        W1 = W1 - LEARNING_RATE * dW1
        W2 = W2 - LEARNING_RATE * dW2
        b1 = b1 - LEARNING_RATE * db1
        b2 = b2 - LEARNING_RATE * db2
        E /= count
        epoch_loss.append(E)

    return E, (W1, b1, W2, b2), epoch_loss

def batch_train(data):
    W1, b1, W2, b2 = random_params()
    count = len(data)
    batch_len = count // BATCH_SIZE
    epoch_loss = []
    
    for epoch in tqdm(range(EPOCHS), desc=f"{BATCH_SIZE}-batch train method"):
        shuffle(data)
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

        W1 = W1 - LEARNING_RATE * dW1
        W2 = W2 - LEARNING_RATE * dW2
        b1 = b1 - LEARNING_RATE * db1
        b2 = b2 - LEARNING_RATE * db2

        E /= BATCH_SIZE
        epoch_loss.append(E)

    return E, (W1, b1, W2, b2), epoch_loss