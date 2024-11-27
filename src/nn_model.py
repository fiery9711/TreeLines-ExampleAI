import pickle
from constants import MODEL_DIR, SHAPES
from os.path import join
from nn_functions import linear, relu, softmax
import numpy as np

class Model:
    def __init__(self, name = "nn-model"):
        self.name = name
    
    def set_value(self, W1, b1, W2, b2):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2
    
    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name

    def get_value(self):
        return (self.W1, self.b1, self.W2, self.b2)

    def load(self, filename):
        with open(join(MODEL_DIR, filename), "rb") as f:
            params, self.name = pickle.load(f)
        W1, b1, W2, b2 = params
        self.set_value(W1, b1, W2, b2)
    
    def save(self, filename):
        with open(join(MODEL_DIR, filename), "wb") as f:
            pickle.dump((self.get_value(), self.name), f)
    
    def predict(self, x):
        # forward
        W1, b1, W2, b2 = self.get_value()
        t = linear(x, W1, b1)
        h = relu(t)
        u = linear(h, W2, b2)
        z = softmax(u)
        yp = np.argmax(z)
        return yp, SHAPES(yp) 
