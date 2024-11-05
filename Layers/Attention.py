import numpy as np
from AutoGrad import Array
from Activations import Softmax

class DotProductAttention:
    def __init__(self,inputd,dk,masked = False):
        self.w = {
            "w_keys": Array(np.random.randn(inputd,dk) * np.sqrt(2/dk)),
            "w_queries": Array(np.random.randn(inputd,dk) * np.sqrt(2/dk)),
            "w_values": Array(np.random.randn(inputd,dk) * np.sqrt(2/dk))
        }
        self.dk = dk
        self.masked = masked

    def forward(self,x,encoder_output = None):
        queries = x.matmul(self.w["w_queries"])
        if encoder_output:
            keys = encoder_output.matmul(self.w["w_keys"])
            values = encoder_output.matmul(self.w["w_values"])
        else:
            keys = x.matmul(self.w["w_keys"])
            values = x.matmul(self.w["w_values"])
            

        z = queries.matmul(keys.transpose()) 
        scaled = z / Array(np.sqrt(self.dk)*np.ones_like(z.data)) 
        if self.masked:
            indices = np.triu_indices(scaled.data.shape[-1],k=1)
            scaled.data[:,indices[0],indices[1]] = -np.inf
        act = Softmax().forward(scaled)
   
        attention = act.matmul(values)

        return attention
    

