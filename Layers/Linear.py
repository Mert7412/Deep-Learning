import numpy as np
from AutoGrad import Array


class Linear:
    def __init__(self,input_dimension,output_dimension):
        self.w = {"w" : Array(np.random.randn(input_dimension,output_dimension) * np.sqrt(2/output_dimension))}
        self.b = {"b" : Array(np.zeros((1,output_dimension)))}


    def forward(self,input):
        z = input.matmul(self.w["w"]) + self.b["b"]
        return z
