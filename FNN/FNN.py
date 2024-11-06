import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from AutoGrad import Array
from Network import Network
from Layers import Normalization,Linear
from Activations import Relu,Softmax
from Optimizers import GradientDescent,Adam
from LossFunctions import CrossEntropySoftmax,CrossEntropy

data = pd.read_csv(r"data\data1\winequality-red.csv",sep=";").values
x = data.T[:-1].T

y = data.T[-1]-3
y1 = np.zeros((y.shape[0],int(y.max()-y.min()+1)))
y1[np.arange(y.shape[0]),np.array(y,dtype=int)] = 1
y1 =Array(y1)
x = Array(x)


model = Network()
model.add_layer(Normalization(11))
model.add_layer(Linear(11,256))
model.add_layer(Relu())
model.add_layer(Normalization(256))
model.add_layer(Linear(256,512))
model.add_layer(Relu())
model.add_layer(Normalization(512))
model.add_layer(Linear(512,256))
model.add_layer(Relu())
model.add_layer(Normalization(256))        
model.add_layer(Linear(256,6))
model.add_layer(Softmax())

optimizer = Adam(model,CrossEntropySoftmax().forward)
optimizer.train(x,y1,1e-6,10000,batch_size=64)


