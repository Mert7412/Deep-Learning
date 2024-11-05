import numpy as np
from AutoGrad import Array

class Normalization:
    def __init__(self,input_size,axis= 0,is_convolution = False):  
        self.w = {}
        self.b = {}
        if is_convolution:
            self.w["w"] = Array(np.random.randn(1,1,1,input_size) * np.sqrt(2/input_size))  
            self.b["b"] = Array(np.zeros((1,1,1,input_size)))
        else:
            self.w["w"] = Array(np.random.randn(1,input_size) * np.sqrt(2/input_size))  
            self.b["b"] = Array(np.zeros((1,input_size)))


        self.axis = axis

        
    def forward(self,input):
        mean = input.mean(axis = self.axis)
        var = input.var(axis = self.axis)
        normalized = (input - mean) / var.sqrt()        
        bn = self.w["w"] * normalized + self.b["b"]
        return bn
    
