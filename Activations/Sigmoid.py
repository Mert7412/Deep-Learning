import numpy as np
from AutoGrad import Array

class Sigmoid:
    def forward(self,input):
        x = input.data
        z = 1 / (1 + np.exp(-x))
        output = Array(z,(input,),"sigmoid")

        def _backward():
            der = z * (1-z)
            input.gradient += der * output.gradient
        output._backward = _backward
        return output    
