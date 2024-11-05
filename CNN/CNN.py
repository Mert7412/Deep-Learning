import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import idx2numpy

from AutoGrad import Array
from Network import Network
from Layers import Normalization,Linear,ConvolutionalLayer,Pooling,Flatten
from Activations import Relu,Softmax
from Optimizers import Adam
from LossFunctions import CrossEntropy

trainingdata = idx2numpy.convert_from_file(r"data\mnist\train-images.idx3-ubyte ")
tsh = trainingdata.shape
x = (trainingdata - np.mean(trainingdata,axis=0) )/ (np.std(trainingdata,axis=0)+1e-8)
x = Array(x.reshape(*tsh,1))
traininglables = idx2numpy.convert_from_file(r"data\mnist\train-labels.idx1-ubyte")
y1 = np.zeros((traininglables.shape[0],int(traininglables.max()-traininglables.min()+1)))
y1[np.arange(traininglables.shape[0]),np.array(traininglables,dtype=int)] = 1

y2 = Array(y1)
model = Network()
model.add_layer(Normalization(1,axis=(0,1,2 ),is_convolution=True))
model.add_layer(ConvolutionalLayer(1,16,3))
model.add_layer(Normalization(16,axis=(0,1,2 ),is_convolution=True))
model.add_layer(Relu())
model.add_layer(Pooling.MaxPooling(2))
model.add_layer(ConvolutionalLayer(16,32,4))
model.add_layer(Normalization(32,axis=(0,1,2 ),is_convolution=True))
model.add_layer(Relu())
model.add_layer(Pooling.MaxPooling(2))
model.add_layer(Flatten())
model.add_layer(Linear(800,128))
model.add_layer(Normalization(128))
model.add_layer(Relu())
model.add_layer(Linear(128,10))
model.add_layer(Normalization(10))
model.add_layer(Softmax())

c = Adam(model,CrossEntropy().forward)

c.train(x,y2,batch_size=100,learning_rate=1e-4)

