import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from IPython.core.pylabtools import figsize

samples_apple = [os.path.join("./data/training/apple", np.random.choice(os.listdir("./data/training/apple"), 1)[0]) for _ in range(8)]
samples_bus = [os.path.join("./data/training/bus", np.random.choice(os.listdir("./data/training/bus"), 1)[0]) for _ in range(8)]

nrows = 4
ncols = 4

fig, ax = plt.subplots(nrows, ncols, figsize=(10, 10))
ax = ax.flatten()

for i in range(nrows * ncols):
    if i < 8:
        pic = plt.imread(samples_apple[i%8])
        ax[i].imshow(pic)
        ax[i].set_axis_off()
    else:
        pic = plt.imread(samples_bus[i%8])
        ax[i].imshow(pic)
        ax[i].set_axis_off()
plt.show()