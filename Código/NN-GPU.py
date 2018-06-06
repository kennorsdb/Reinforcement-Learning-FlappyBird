# -*- coding: utf-8 -*-
from cudaUtil import NNGPU, NNGPU_class
import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
from numba import *
from numba import vectorize, float64, int32, int64, f8
import numba as numba
import gc
import math as math
from pdb import set_trace
import matplotlib.pyplot as plt
from mnist import MNIST
import struct

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def graficarXY(X, Y):
    plt.figure()
    gridpoints = 255
    x1s = np.linspace(0, 255, 255)
    x2s = np.linspace(0, 255, 255)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.show()

def graficarError(error):
    plt.figure()
    plt.plot(error)
    plt.show()



def generarDatosPrueba(features, n, func):
    X = np.random.rand(n, features)
    return (X, func(X))

def xor(X):
    Y = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        Y[i]= 1 if X[i,0] >= 0.5 and X[i,1] < 0.5 else 0
        Y[i]= 1 if X[i,0] < 0.5 and X[i,1] >= 0.5 else Y[i]
    return Y.astype(int)



mndata = MNIST('.\\NMIST')
images = read_idx('./NMIST/train-images-idx3-ubyte')
labels = read_idx('./NMIST/train-labels-idx1-ubyte')


images = images.reshape((images.shape[0],images.shape[1]*images.shape[2])  ) 
print images.shape


# Carga la función principal
if __name__ == '__main__':
    np.set_printoptions(precision=3)

    def func2(x): return ((x[:, 0]*x[:, 0]+x[:, 0])/255 < x[:, 1]).astype(int)
    
    (X, Y) = generarDatosPrueba(2, 10000, xor)
    #graficarXY(X,Y)
    config = np.array([5, 6, 8])
    (error, W) = NNGPU(X, Y, config)
    graficarError(error)
    print error
    #for i in range(10):   
    #    print X[i], NNGPU_class(X[0], W, np.append(config, 2))