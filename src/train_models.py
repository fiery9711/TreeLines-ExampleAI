from constants import EPOCHS, BATCH_SIZE, PLOT_DIR
import matplotlib.pyplot as plt 
import nn_functions as nn
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
    


def mean_loss(loss):
    n = len(loss)
    c = n // EPOCHS
    r = []
    for i in range(EPOCHS):
        s = 0
        for l in loss[i*c:i*c+c]:
            s += l
        r.append(s/c)
    return r
print("training...")
dataset = load()
data = dataset["train"]
val = dataset["test"]
bl1, b1, l1, s1 = work_time(nn.simple_train, data)
l1m = mean_loss(l1)
a1, c1, w1 = nn.accuracy(b1, val)
aa1, cc1, ww1 = nn.accuracy(b1, data)
t1 = f"simple train, epoch: {b1[0]}, loss: {bl1:4.3f}%, time: {s1:4.3} seconds"
_, W1, bs1, W2, bs2 = b1
simple = Model("Simple train method")
simple.set_value(W1, bs1, W2, bs2)

bl2, b2, l2, s2 = work_time(nn.full_train, data)
a2, c2, w2 = nn.accuracy(b2, val)
aa2, cc2, ww2 = nn.accuracy(b2, data)
l2m = mean_loss(l2)
t2 = f"full train,epoch: {b2[0]}, loss: {bl2:4.3f}%, time: {s2:4.3} seconds"
_, W1, bs1, W2, bs2 = b2
full = Model( "Full epoch train method")
full.set_value(W1, bs1, W2, bs2)

bl3, b3, l3, s3= work_time(nn.batch_train, data)
a3, c3, w3 = nn.accuracy(b3, val)
aa3, cc3, ww3 = nn.accuracy(b3, data)
l3m = mean_loss(l3)
t3 = f"batch ({BATCH_SIZE}) train, epoch {b3[0]}, loss: {bl3:4.3f}%, time: {s3:4.3} seconds"
_, W1, bs1, W2, bs2 = b3
batch = Model("Batch data train method")
batch.set_value(W1, bs1, W2, bs2)

simple.save(f"simple-{EPOCHS}.bin")
full.save(f"full-{EPOCHS}.bin")
batch.save(f"batch-{EPOCHS}.bin")

figure, axis = plt.subplots(3, 3)
axis[0, 0].plot(l1m)
axis[0, 0].set_title(t1)
axis[0, 1].plot(l2m)
axis[0, 1].set_title(t2)
axis[0, 2].plot(l3m)
axis[0, 2].set_title(t3)

axis[1, 0].plot(c1, color="g", label="corrects")
axis[1, 0].set_title(f"simple train test, accuracy: {a1:4.2f}%")
axis[1, 0].plot(w1, color="r", label="wrongs")
axis[1, 1].plot(c2, color="g", label="corrects")
axis[1, 1].set_title(f"full train test, accuracy: {a2:4.2f}%")
axis[1, 1].plot(w2, color="r", label="wrongs")
axis[1, 2].plot(c3, color="g", label="corrects")
axis[1, 2].set_title(f"batch train test, accuracy: {a3:4.2f}%")
axis[1, 2].plot(w3, color="r", label="wrongs")

axis[2, 0].plot(cc1, color="g", label="corrects")
axis[2, 0].set_title(f"simple train train, accuracy: {aa1:4.2f}%")
axis[2, 0].plot(ww1, color="r", label="wrongs")
axis[2, 1].plot(cc2, color="g", label="corrects")
axis[2, 1].set_title(f"full train train, accuracy: {aa2:4.2f}%")
axis[2, 1].plot(ww2, color="r", label="wrongs")
axis[2, 2].plot(cc3, color="g", label="corrects")
axis[2, 2].set_title(f"batch train train, accuracy: {aa3:4.2f}%")
axis[2, 2].plot(ww3, color="r", label="wrongs")
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.tight_layout()
plt.show()
figure.savefig(join(PLOT_DIR, "train-{0}.png"))
plt.close(figure)
print("done.")