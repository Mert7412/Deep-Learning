import numpy as np
from AutoGrad import Array

class Mae:
    def forward(self,y,ypred):
        yx = y.data
        ypredx = ypred.data
        z = np.sum(np.abs(yx-ypredx),keepdims=True)
        output = Array(z,(y,ypred),"mae")

        def _backward():
            der = np.zeros(yx.shape)
            der[yx>ypredx] = -1
            der[yx<=ypredx] = 1
            ypred.gradient += der * output.gradient
            y.gradient += -der * output.gradient 
        output._backward = _backward
        return output
