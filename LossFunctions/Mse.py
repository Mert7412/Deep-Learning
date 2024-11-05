import numpy as np
from AutoGrad import Array

class Mse:
    def forward(self,y,ypred):
        yx = y.data
        ypredx = ypred.data
        z = np.mean((yx-ypredx)**2,keepdims=True)
        output = Array(z,(y,ypred),"mse")
        
        def _backward():
            der = 2*z
            ypred.gradient += -der * output.gradient
            y.gradient += der * output.gradient
        output._backward = _backward
        return output
