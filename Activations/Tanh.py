import numpy as np
from AutoGrad import Array

class Tanh:
    def forward(self,input):
        x = input.data
        z = np.tanh(x)
        output = Array(z,(input,),"tanh")

        def _backward():
            der = 1 - (z**2)
            input.gradient += der * output.gradient
        output._backward = _backward
        return output
            