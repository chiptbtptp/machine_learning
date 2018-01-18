import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#load the data (cat/not-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
index = 4
plt.imshow(train_set_x_orig[index])
print("y="+str(train_set_y[:, index])+ ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m*np.sum(Y * np.log(A) + (1 - Y)*np.log(1-A))

    dw = 1 / m*np.dot(X, (A - Y).T)
    db = 1/ m*np.sum(A - Y)
    cost = np.squeeze(cost)

    grads = {"dw":dw, "db":db}
    return grads, cost


