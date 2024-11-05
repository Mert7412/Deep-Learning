import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import idx2numpy

from AutoGrad import Array
from Network import Network 
from Activations import Softmax
from Optimizers import MiniBatchGradientDescent
from LossFunctions import CrossEntropy
from Layers import LSTMLayer

trainingdata = idx2numpy.convert_from_file(r"data\mnist\train-images.idx3-ubyte")
tsh = trainingdata.shape
x = (trainingdata - np.mean(trainingdata,axis=0) )/ (np.std(trainingdata,axis=0)+1e-8)

traininglables = idx2numpy.convert_from_file(r"data\mnist\train-labels.idx1-ubyte")
y1 = np.zeros((traininglables.shape[0],int(traininglables.max()-traininglables.min()+1)))
y1[np.arange(traininglables.shape[0]),np.array(traininglables,dtype=int)] = 1
y2 = Array(y1)
x = Array(x)

model = Network()
model.add_layer(LSTMLayer(28,128,output_layer=False))
model.add_layer(LSTMLayer(128,64,"many_to_one",True,Softmax().forward,10))

opt = MiniBatchGradientDescent(model,CrossEntropy().forward)
opt.train(x,y2,1e-4,1000)

