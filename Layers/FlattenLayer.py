import numpy as np
from AutoGrad import Array

class Flatten:
    def flatten(self,input):
        x = input.data
        xs = x.shape
        z = x.reshape(xs[0],np.prod(xs[1:]))
        output = Array(z,(input,),"Flatten")

        def _backward():
            input.gradient += output.gradient.reshape(xs)
        output._backward = _backward
        return output

    def forward(self,input):
        z = self.flatten(input)
        return z



