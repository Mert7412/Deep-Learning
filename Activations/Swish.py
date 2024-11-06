import numpy as np
from AutoGrad import Array

class Swish:
    def __init__(self):
        self.beta = 1
    def forward(self,input):
        x = input.data
        sigmoid = 1/ (1 + np.exp(-self.beta*x))
        z = x * sigmoid
        output = Array(z,(input,),"Swish")

        def _backward():
            der = z + sigmoid * (1 - z)
            input.gradient += der * output.gradient
        
        output._backward = _backward
        return output
            