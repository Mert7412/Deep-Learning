import numpy as np
from AutoGrad import Array

class LeakyRelu:
    def forward(self,input):
        x = input.data
        z = np.maximum(0.1*x,x)
        output = Array(z,(input,),"LeakyRelu")

        def _backward():
            x[x >= 0] = 1
            x[x < 0] = 0.1
            input.gradient += x * output.gradient

        output._backward = _backward
        return output
