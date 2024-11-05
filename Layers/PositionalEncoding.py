import numpy as np
from AutoGrad import Array

class PositionalEncoding:
    def forward(self,x):
        input = x.data
        dimensions_len = len(input[0,0])
        positions = (np.ones((len(input[0]),int(dimensions_len/2))).T + np.arange(len(input[0]))).T           
        dimensions = np.arange(dimensions_len/2,dtype=int)*2
        encoding_values = np.zeros_like(input)
        encoding_values[:,:,dimensions] = np.sin(positions/(10000**(dimensions/dimensions_len)))
        encoding_values[:,:,dimensions+1] = np.cos(positions/(10000**(dimensions/dimensions_len)))
        encoding_values = Array(encoding_values)
        
        z = x + encoding_values
        return z
    
