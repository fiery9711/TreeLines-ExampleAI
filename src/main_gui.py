import tkinter as tk
from tkinter import font
from nn_model import Model
from shape_generator import shape
from nn_constants import SHAPES, MODEL_DIR
import os

def load_model(filename):
    model = Model()
    model.load(filename)
    return model

class GUI:
    def __init__(self, main):
        self.main = main
        self.defaultFont = font.nametofont("TkDefaultFont") 
        self.frame_canvas = self.create_canvas()
        self.frame_canvas.pack(side=tk.LEFT)
        self.frame_param = self.create_params()
        self.frame_param.pack(side=tk.RIGHT)
        self.frame = self.create_buttons()
        self.frame.pack(side=tk.RIGHT)
        
        # Overriding default-font with custom settings 
        # i.e changing font-family, size and weight 
        self.defaultFont.configure(family="Times New Roman", 
                                   size=16) 
        
        self.model = None
        self.selected_shape = None
    
    def on_click_manual(self, e1, e2, e3):
        x1 = int(e1.get())
        x2 = int(e2.get())
        x3 = int(e3.get())
        self.selected_shape = ((x1, x2, x3), -1)
        self.draw_shape(-1)
        self.label.config(text = f"x1 = {x1}, x2 = {x2}, x3 = {x3}")

    def create_params(self):
        frame = tk.Frame(self.main)
        frame.pack(padx=4, pady=4)

        # X1 uchun Label va Entry
        label1 = tk.Label(frame, text="X1 =")
        label1.grid(row=0, column=0, padx=5, pady=5)
        entry1 = tk.Entry(frame, font=self.defaultFont, width=10)
        entry1.grid(row=0, column=1, padx=5, pady=5)

        # X2 uchun Label va Entry
        label2 = tk.Label(frame, text="X2 =")
        label2.grid(row=1, column=0, padx=5, pady=5)
        entry2 = tk.Entry(frame, font=self.defaultFont, width=10)
        entry2.grid(row=1, column=1, padx=5, pady=5)

        # X3 uchun Label va Entry
        label3 = tk.Label(frame, text="X3 =")
        label3.grid(row=2, column=0, padx=5, pady=5)
        entry3 = tk.Entry(frame, font=self.defaultFont, width=10)
        entry3.grid(row=2, column=1, padx=5, pady=5)

        # Chiqish tugmasi
        button = tk.Button(frame, text="Chizish", command=lambda e1=entry1, e2=entry2, e3=entry3: self.on_click_manual(e1, e2, e3))
        button.grid(row=3, column=0, columnspan=2, pady=10)
        return frame

    def on_click(self, idx):
        self.selected_shape = shape(idx)
        x1, x2, x3 = self.selected_shape[0]
        self.label.config(text = f"x1 = {x1}, x2 = {x2}, x3 = {x3}")
        self.draw_shape(idx)
    
    def predict(self):
        idx = -1
        if self.selected_shape != None and self.model != None:
            idx, ypz = self.model.predict(self.selected_shape[0])
            self.labelp.config(text = f"{ypz}")
        self.draw_shape(idx)
        


    def create_canvas(self):
        frame = tk.Frame(self.main)
        frame.pack(padx=4, pady=4)
        self.label = tk.Label(frame)
        self.label.pack()
        self.labelp = tk.Label(frame)
        self.labelp.pack()

        # Canvas yaratish
        canvas = tk.Canvas(frame, width=400, height=400, bg="white")
        self.canvas = canvas
        canvas.pack()

        # # Y qiymatini kiritish uchun Entry
        # label = tk.Label(frame, text="Y qiymatini kiriting (0-6):")
        # label.pack(pady=5)

        # entry = tk.Entry(frame)
        # entry.pack(pady=5)

        # Chizish tugmasi
        button = tk.Button(frame, text="Shaklni SI aniqlash", command=lambda: self.predict())
        button.pack(pady=10)
        return frame

    def create_buttons(self):
        frame = tk.Frame(self.main, width=300)
        frame.pack(padx=4, pady=4)

        # 7 ta tugma yaratish va ularni frame ga joylashtirish
        label = tk.Label(frame, text="Quyidagi shaklni tanglang:")
        label.pack(side=tk.TOP, pady=5)
        for i, v in enumerate(SHAPES):
            button_text = f"Shakl - {v}"
            index = i
            button = tk.Button(frame, text=button_text, command=lambda idx = index: self.on_click(idx))
            button.pack(side=tk.BOTTOM, pady=5, fill="x")
        return frame

    def draw_shape(self, y):
        canvas = self.canvas
        canvas.delete("all")  # Canvasni tozalash
        shape_name = ""  # Shakl nomini saqlash
        
        if y == 0:
            canvas.create_oval(175, 175, 225, 225, outline="black", fill="lightblue", width=2)  # Aylana
            shape_name = "Aylana"
        elif y == 1:
            canvas.create_line(50, 50, 350, 350, fill="black", width=2)  # Kesma
            shape_name = "Kesma"
        elif y == 2:
            canvas.create_rectangle(50, 50, 350, 350, outline="black", fill="lightgreen", width=2)  # Kvadrat
            shape_name = "Kvadrat"
        elif y == 3:
            canvas.create_rectangle(50, 100, 350, 300, outline="black", fill="yellow", width=2)  # To‘g‘ri to‘rtburchak
            shape_name = "To‘g‘ri to‘rtburchak"
        elif y == 4:
            canvas.create_polygon(200, 50, 50, 350, 350, 350, outline="black", fill="pink", width=2)  # Teng tomonli uchburchak
            shape_name = "Teng tomonli uchburchak"
        elif y == 5:
            canvas.create_polygon(200, 50, 100, 350, 300, 350, outline="black", fill="orange", width=2)  # Teng yonli uchburchak
            shape_name = "Teng yonli uchburchak"
        elif y == 6:
            canvas.create_polygon(50, 350, 300, 50, 350, 350, outline="black", fill="red", width=2)  # Uchburchak
            shape_name = "Uchburchak"
        else:
            shape_name = "Aniqlanmadi"
    
        # Canvasda shakl nomini ko‘rsatish
        canvas.create_text(200, 25, text=f"{shape_name}",  fill="black")
    
    def loop(self):
        self.main.mainloop()

    def set_model(self, model):
        self.model = model



def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

def on_select(event, gui, item):
    # Get the selected item
    model = load_model(item)
    gui.set_model(model)
    print(f"Model '{model.get_filename()}' loaded")

from tkinter import ttk

if __name__ == "__main__":
    root = tk.Tk("3 kesma muammosi")
    root.geometry("1000x600")
    models_list = os.listdir(MODEL_DIR)
    frame_top = tk.Frame(root)
    frame_top.pack()
    frame_content = tk.Frame(root)
    frame_content.pack(fill="x")
    gui = GUI(frame_content)
    combo = ttk.Combobox(frame_top, values=models_list, font=gui.defaultFont)
    combo.pack(pady=4, fill="x")
    combo.bind("<<ComboboxSelected>>", lambda evnt, g = gui, cb = combo: on_select(evnt, g, combo.get()))
    combo.set("Model tanglang")
    gui.draw_shape(-1)
    center_window(root)
    gui.loop()

