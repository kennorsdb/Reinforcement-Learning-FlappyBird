# -*- coding: utf-8 -*-
import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
from numba import cuda
from numba import vectorize, float64, int32, int64, f8
import numba as numba
import gc
import math as math
from pdb import set_trace
import matplotlib.pyplot as plt

# SET NUMBA_ENABLE_CUDASIM=1

MAXCAPAS = 1000
MAXNEURONAS = 1000

def usoMemoria():
    #memoria = cuda.get_current_device().get_primary_context().get_memory_info()
    #print ('Memoria libre: ' + str(float(memoria[0]*100)/memoria[1]) + '% (' + str(memoria[0]/1048576) + 'Mb)')
    pass


@cuda.jit(device=True, inline=True)
def reLu(x):
    pass


@cuda.jit(device=True, inline=True)
def reLuPrime(x):
    return 0 if x <= 0 else 1


@cuda.jit(device=True)
def foreward(X, W, config, Z, out):
    ''' Se calcula el Foreware de una imagen
    '''
    entrada = cuda.threadIdx.x   # Número de entrada (Número de Thread)
    neurona = cuda.blockIdx.x    # Número de Neurona (Número de Bloque)
    t = cuda.shared.array(shape=1000, dtype=float64)  # Variable Temporal
    
    if entrada == 0 and neurona < config[0]: 
        out[0, neurona] = X[neurona]
    cuda.syncthreads()

    # Se calcula la clasificación para la imagen actual
    for capa in range(1, config.shape[0]):
        # Por neurona, se calcula Σ(w⋅x)
        if neurona < config[capa] and entrada < config[capa-1]:
            t[entrada] = (W[capa-1, neurona, entrada] * out[capa-1, entrada])
        cuda.syncthreads()  # Se sincronizan los threads

        # Se reduce con operación suma, de eso se encarga el Thread0 de cada Neurona
        if entrada == 0 and neurona < config[capa]:
            for j in range(config[capa]):
                Z[neurona] += t[j]
            out[capa, neurona] = max(0, Z[neurona])  # Calcula ReLu
        cuda.syncthreads()


@cuda.jit( device=True )
def softMax(Yfit, yCount, error, delta):
    entrada = cuda.threadIdx.x   # Número de entrada (Número de Thread)
    neurona = cuda.blockIdx.x    # Número de Neurona (Número de Bloque)
    error = 0
    # Se obtiene el exponente del vector Yfit
    if entrada == 0 and neurona < yCount:
        error += math.exp(Yfit[neurona])
    cuda.syncthreads()

    if entrada == 0 and neurona == 0:
        delta = math.exp(Yfit[neurona])/error

    cuda.syncthreads()



@cuda.jit ( device=True )
def crossEntropy (Y, Yfit, yCount, error, delta):
    entrada = cuda.threadIdx.x   # Número de entrada (Número de Thread)
    neurona = cuda.blockIdx.x    # Número de Neurona (Número de Bloque)
     
    softMax(Yfit, yCount, error, delta)
    if neurona == Y and entrada == 0:
        error = 15 #-math.log(delta) # Se calcula la gradiente con one hot vector
        delta -= 1
    cuda.syncthreads()
    return 15
    
@cuda.jit ( device=True )
def backProp (W, X, Z, config, out, delta):
    entrada = cuda.threadIdx.x   # Número de entrada (Número de Thread)
    neurona = cuda.blockIdx.x    # Número de Neurona (Número de Bloque)
    error = cuda.shared.array(shape=1, dtype=float64) # de tamaño constante capas 

    # Capa de Salida => No multiplica W
    if neurona <= config[-1] and entrada <= config[-2]:
        capa = config.shape[0]
        if entrada == 0:
            error[neurona] = delta [neurona] * reLuPrime(Z[capa])
            delta[neurona] = error[neurona]  * out[capa-1,neurona]
        cuda.syncthreads()

        W[capa,neurona,entrada] -= 0.001 * delta[neurona]
        cuda.syncthreads()

    for capa in range(config.shape[0]-1, 1):
        if neurona <= config[capa] and entrada <= config[capa-1]:
            a = 0
            for e in range(config[capa]):
                a +=  error[e] * W [capa+1,neurona,e]

            error[neurona] =  a * reLuPrime(Z[capa])
            delta[neurona] = error[neurona]  * out[capa-1,neurona]
            cuda.syncthreads()

            W[capa,neurona,entrada] -= 0.001 * delta[neurona]
            cuda.syncthreads()


@cuda.jit()
def NNCUDA(X, Y, W, config, error, out):
    entrada = cuda.threadIdx.x   # Número de entrada (Número de Thread)
    neurona = cuda.blockIdx.x    # Número de Neurona (Número de Bloque)
    maxEntrada = cuda.blockDim.x # Cantidad máxima de entradas 
    maxNeurona = cuda.gridDim.x  # Cantidad máxima de neuronas 
    Z = cuda.shared.array(shape=MAXCAPAS, dtype=float64) # de tamaño constante capas 
    delta = cuda.shared.array(shape=MAXCAPAS, dtype=float64) # de tamaño constante capas 
    
    for i in range(0,10000):
        foreward(X[i], W, config, Z, out)
        error[i] = crossEntropy(Y[i], out[-1], config[-1], error[i], delta[-1])
        #backProp (W, X, Z, config, out, delta)





def NNGPU(X, Y, config):
    print ("Se inicia el procesamiento de GPU")
    # Control del tiempo
    #start = timer()

    # Se prepara el vector de configuracion
    diffY = np.unique(Y).shape[0]   # La cantidad de clases de Y
    config = np.append(config[:], diffY)
    maxNeuronas = max(config)
    config = np.append(X.shape[1], config[:])

    
    #from pdb import set_trace; set_trace()

    # Se inicializa la matriz de pesos
    maxPesos = max(config)
    W = np.random.rand(config.shape[0]-1, maxNeuronas, maxPesos)+1
    # Se calculan los bloques y los threads en CUDA
    threadsPerBlock = maxPesos # en cada bloque se calcula una neurona
    blocksPerGrid = maxNeuronas
    print "     Shape de W: ", W.shape
    print "     Threads por Bloque: ", threadsPerBlock
    print "     Total de Bloques: ", blocksPerGrid
    print "     Total de Threads: ", blocksPerGrid*threadsPerBlock

    # Se mueven los datos necesarios para el GPU

    Wg = cuda.to_device(W)
    Xg = cuda.to_device(X)
    Yg = cuda.to_device(Y)
     
    errorG = cuda.device_array([X.shape[0]])
    configG = cuda.to_device(config)
    output = cuda.device_array([config.shape[0], config.max()])

    usoMemoria()
    NNCUDA[blocksPerGrid, threadsPerBlock](Xg, Yg, Wg, configG, errorG, output)

    O = output.copy_to_host()
    print (X)
    print (O)
    #print (W)
    E = errorG.copy_to_host()
    P = Wg.copy_to_host()
    return (E, P)





@cuda.jit
def NNCUDA_class(X, W, config, out):
    Z = cuda.shared.array(shape=MAXCAPAS, dtype=float64) # de tamaño constante capas 
    foreward(X, W, config, Z, out)

def NNGPU_class(X,W, config):
    maxNeuronas = max(config)
    config = np.append(X.shape[0], config[:])

    # Se inicializa la matriz de pesos
    maxPesos = max(config)
    threadsPerBlock = maxNeuronas   # en cada bloque se calcula una neurona
    blocksPerGrid = maxNeuronas
    # Se mueven los datos necesarios para el GPU
    Wg = cuda.to_device(W)
    Xg = cuda.to_device(X)

    configG = cuda.to_device(config)
    output = cuda.device_array([config.shape[0], config.max()])

    NNCUDA_class[blocksPerGrid, threadsPerBlock](Xg, Wg, configG, output)

    ret = output.copy_to_host()
    #P = Wg.copy_to_host()
    #print (P)
    return ret[-1]
