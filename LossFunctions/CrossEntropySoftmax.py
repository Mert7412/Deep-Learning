import numpy as np
from AutoGrad import Array

class CrossEntropySoftmax:
    def forward(self,y,ypred):
        yx = y.data
        ypredx = ypred.data
        z = -np.mean(yx*np.log(ypredx),keepdims=True)
        output = Array(z,(y,ypred),"crossentopry")

        def _backward():
            ypred.gradient += (ypredx - yx) * output.gradient
            y.gradient += -(np.log(ypredx)) * output.gradient
        output._backward = _backward
        return output