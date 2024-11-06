import numpy as np
from AutoGrad import Array

class Softmax:
    def forward(self,input):
        x = input.data
        e = np.exp(x-np.max(x,axis=-1,keepdims=True))
        z = e/np.sum(e,axis=-1,keepdims=True)
    
        output = Array(z,(input,),"softmax")
        def _backward():
            #der = -(z[...,np.newaxis]*z[...,np.newaxis,:]) + (z[...,np.newaxis,:]*np.diag(np.ones(len(z[0]))))
            #input.gradient += np.einsum("...w,...wh->...h",output.gradient,der)
            input.gradient += output.gradient
        output._backward = _backward 
        return output
  