import tkinter as tk
from nn_model import Model
from shape_generator import shape
from constants import SHAPES

model = None
selected_shape = None

def load_model(filename):
    model.load(filename)
    print(f"{model.model()} is loaded")

def change_shape(yk):
    selected_shape = shape(yk)
    print(f"Shape changed to {yk}({SHAPES[yk]})")

class GUI:
    def __init__(self, main):
        self.main = main
        self.canvas = tk.Canvas(main, width=400, height=400)
        self.frame = tk.Frame(main)
        self.canvas.pack(expand="both", side="left")
        self.frame.pack(expand="y", side="right")

    def create_buttons(self):
        frame = tk.Frame(self.main)
        for i, v in enumerate(SHAPES):
            button = tk.Button(frame, text=v, pady=4, padx=4)
            button.index = i
    
    def create_shape(self, shape, index, name):
        x1, x2, x3 = shape
        if index == 0:
            self.canvas.create_oval(150, 150, 250, 250, outline="black", fill="white", width=2)
        if index == 1:
            self.canvas.create_line(150, 150, 250, 250, outline="black", fill="white", width=2)
        if index == 2:
            self.canvas.create_rectangle(100, 100, 300, 300, outline="black", fill="white", width=2)
        if index == 3:
            self.canvas.create_oval(100, 100, 300, 200, outline="black", fill="white", width=2)
        if index == 4:
            self.canvas.create_oval(150, 150, 200, 200, outline="black", fill="white", width=2)
            



if __name__ == "__main__":
    model = Model("Simple model")
    model.load("simple-1000.bin")
    selected_shape = shape(1)

