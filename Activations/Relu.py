import numpy as np
from AutoGrad import Array

class Relu:
    def forward(self,input):
        x = input.data
        z = np.maximum(0,x)
        output = Array(z,(input,),"relu")
    
        def _backward():
            x[x > 0] = 1
            x[x <= 0] = 0
            input.gradient += x * output.gradient
        output._backward = _backward
        return output   