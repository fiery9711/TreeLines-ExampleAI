import pickle
from nn_constants import MODEL_DIR, SHAPES
from os.path import join
import nn_functions as nn
import numpy as np

class Model:
    def __init__(self, name = "nn-model"):
        self.name = name
        self.filename = None
    
    def set_value(self, W1, b1, W2, b2):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2
        
    
    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name

    def get_filename(self):
        return self.filename

    def get_value(self):
        return (self.W1, self.b1, self.W2, self.b2)

    def load(self, filename):
        self.filename = filename
        with open(join(MODEL_DIR, filename), "rb") as f:
            params, self.name = pickle.load(f)
        W1, b1, W2, b2 = params
        self.set_value(W1, b1, W2, b2)
    
    def save(self, filename):
        with open(join(MODEL_DIR, filename), "wb") as f:
            pickle.dump((self.get_value(), self.name), f)
    
    def predict(self, x):
        z = nn.predict(x, self.get_value())
        yp = np.argmax(z)
        z = np.around(z * 100, 0)[0].astype(int).astype(str) + "%"
        return yp, ", ".join(z)
