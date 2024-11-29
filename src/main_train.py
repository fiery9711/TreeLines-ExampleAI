from nn_constants import EPOCHS, DATASET_NAME, BATCH_SIZE, PLOT_DIR, MIN_SHAPE_VALUE, MAX_SHAPE_VALUE, SHAPE_COUNT
import matplotlib.pyplot as plt 
import nn_functions as nn
import nn_train as nnt
from shape_generator import load
import time
from nn_model import Model
from os.path import join


def work_time(func, param):
    start = time.time_ns()
    bl, w, l = func(param)
    end = time.time_ns()
    ns = end - start
    sec = ns / 1000 / 1000 / 1000
    print(f"best loss: {bl:4.3f}, best epoch: {w[0]}, avg loss: {(sum(l) / len(l)):4.3f}, time: ({sec:4.4f}) {ns} ns")
    return bl, w, l, sec 

def train(dataset_filename, show = False):
    print(f"E{EPOCHS}-S{SHAPE_COUNT} training...")
    dataset = load(dataset_filename)
    data = dataset["train"]
    print(dataset_filename, len(data))
    val = dataset["test"]
    bl1, b1, l1, s1 = work_time(nnt.simple_train, data)
    l1m = nn.mean_loss(l1, EPOCHS)
    a1, c1, w1 = nn.accuracy(b1, val)
    aa1, cc1, ww1 = nn.accuracy(b1, data)

    W1, bs1, W2, bs2 = b1
    simple = Model("Simple train method")
    simple.set_value(W1, bs1, W2, bs2)

    bl2, b2, l2, s2 = work_time(nnt.full_train, data)
    a2, c2, w2 = nn.accuracy(b2, val)
    aa2, cc2, ww2 = nn.accuracy(b2, data)
    l2m = nn.mean_loss(l2, EPOCHS)

    W1, bs1, W2, bs2 = b2
    full = Model( "Full epoch train method")
    full.set_value(W1, bs1, W2, bs2)

    bl3, b3, l3, s3= work_time(nnt.batch_train, data)
    a3, c3, w3 = nn.accuracy(b3, val)
    aa3, cc3, ww3 = nn.accuracy(b3, data)
    l3m = nn.mean_loss(l3, EPOCHS)

    W1, bs1, W2, bs2 = b3
    batch = Model("Batch data train method")
    batch.set_value(W1, bs1, W2, bs2)

    simple.save(f"demo-simple-E{EPOCHS}-S{SHAPE_COUNT}-MN{MIN_SHAPE_VALUE}-MX{MAX_SHAPE_VALUE}.bin")
    full.save(f"demo-full-E{EPOCHS}-S{SHAPE_COUNT}-MN{MIN_SHAPE_VALUE}-MX{MAX_SHAPE_VALUE}.bin")
    batch.save(f"demo-batch-E{EPOCHS}-S{SHAPE_COUNT}-MN{MIN_SHAPE_VALUE}-MX{MAX_SHAPE_VALUE}.bin")

    figure, axis = plt.subplots(3, 3)
    if not show:
        figure.set_dpi(96)
        figure.set_figheight(15)
        figure.set_figwidth(21)
    figure.suptitle(f"3 kesma muammosining turli xil usulda gradientni o'zgartirish o'rqali olingan natijalar" + 
    f"\n MIN = {MIN_SHAPE_VALUE}, MAX = {MAX_SHAPE_VALUE}" + 
    f"\n DAVRLAR = {EPOCHS} SHAKLLAR = {SHAPE_COUNT}")
    t1 = f"Xar bir shakl usuli, vaqt: {s1:4.2f} soniya"
    t2 = f"Xar bir davr usuli, vaqt: {s2:4.2f} soniya"
    t3 = f"Xar bir batch({BATCH_SIZE}) usuli, vaqt: {s3:4.2f} soniya"

    axis[0, 0].plot(l1m)
    axis[0, 0].set_xlabel(f"Davrlar")
    axis[0, 0].set_ylabel(f"Xatolik (EKX: {bl1:4.2f})")
    axis[0, 0].set_title(t1)
    axis[0, 1].plot(l2m)
    axis[0, 1].set_xlabel(f"Davrlar")
    axis[0, 1].set_ylabel(f"Xatolik (EKX: {bl2:4.2f})")
    axis[0, 1].set_title(t2)
    axis[0, 2].plot(l3m)
    axis[0, 2].set_xlabel(f"Davrlar")
    axis[0, 2].set_ylabel(f"Xatolik (EKX: {bl2:4.2f})")
    axis[0, 2].set_title(t3)

    axis[1, 0].plot(c1, color="g", label="to'g'ri topganlari")
    axis[1, 0].set_title(f"Aniqlik (Test): {a1:5.3f}%")
    axis[1, 0].set_xlabel(f"Jami test dataset shakllar soni")
    axis[1, 0].set_ylabel(f"O'sish ko'rsatkichi")
    axis[1, 0].plot(w1, color="r", label="noto'g'ri topganlari")
    axis[1, 1].plot(c2, color="g", label="to'g'ri topganlari")
    axis[1, 1].set_title(f"Aniqlik (Test): {a2:5.3f}%")
    axis[1, 1].set_xlabel(f"Jami test dataset shakllar soni")
    axis[1, 1].set_ylabel(f"O'sish ko'rsatkichi")
    axis[1, 1].plot(w2, color="r", label="noto'g'ri topganlari")
    axis[1, 2].plot(c3, color="g", label="to'g'ri topganlari")
    axis[1, 2].set_title(f"Aniqlik (Test): {a3:5.3f}%")
    axis[1, 2].set_xlabel(f"Jami test dataset shakllar soni")
    axis[1, 2].set_ylabel(f"O'sish ko'rsatkichi")
    axis[1, 2].plot(w3, color="r", label="noto'g'ri topganlari")

    axis[2, 0].plot(cc1, color="g", label="to'g'ri topganlari")
    axis[2, 0].set_title(f"Aniqlik (Train): {aa1:5.3f}%")
    axis[2, 0].set_xlabel(f"Jami train dataset shakllar soni")
    axis[2, 0].set_ylabel(f"O'sish ko'rsatkichi")
    axis[2, 0].plot(ww1, color="r", label="noto'g'ri topganlari")
    axis[2, 1].plot(cc2, color="g", label="to'g'ri topganlari")
    axis[2, 1].set_title(f"Aniqlik (Train): {aa2:5.3f}%")
    axis[2, 1].set_xlabel(f"Jami train dataset shakllar soni")
    axis[2, 1].set_ylabel(f"O'sish ko'rsatkichi")
    axis[2, 1].plot(ww2, color="r", label="noto'g'ri topganlari")
    axis[2, 2].plot(cc3, color="g", label="to'g'ri topganlari")
    axis[2, 2].set_title(f"Aniqlik (Train): {aa3:5.3f}%")
    axis[2, 2].set_xlabel(f"Jami train dataset shakllar soni")
    axis[2, 2].set_ylabel(f"O'sish ko'rsatkichi")
    axis[2, 2].plot(ww3, color="r", label="noto'g'ri topganlari")
    if show:
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.tight_layout()
        plt.show()
    figure.savefig(join(PLOT_DIR, f"demo-plots-E{EPOCHS}-SH{SHAPE_COUNT}-MN{MIN_SHAPE_VALUE}-MX{MAX_SHAPE_VALUE}.png"))
    plt.close(figure)
    print("done.")

def linear():
    global SHAPE_COUNT
    global EPOCHS
    for e in [100, 1000]:
        for s in [100, 1000]:
            SHAPE_COUNT = s
            EPOCHS = e
            datasetfilename = f"treelines-{SHAPE_COUNT}-{MIN_SHAPE_VALUE}-{MAX_SHAPE_VALUE}.bin"
            train(datasetfilename)

if __name__ == "__main__":
    linear()
    #train(DATASET_NAME, True)