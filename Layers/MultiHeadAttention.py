import numpy as np 
from Layers.Attention import DotProductAttention
from AutoGrad import Array


class MultiHeadAttention:
    def __init__(self,inputd,dk,dmodel,heads = 1,masked = False):
        self.w = {"w_o" : Array(np.random.randn(dk*heads,dmodel) * np.sqrt(2/dmodel)) }
        self.heads = [DotProductAttention(inputd,dk,masked) for i in range(heads)]
        for i in range(len(self.heads)):
            for keys,values in self.heads[i].w.items():
                self.w[f"head{i}_{keys}{i}"] = values

    def forward(self,x,encoder_output = None):
        if encoder_output: 
            concanated = self.heads[0].forward(x,encoder_output)  
        else:
            concanated = self.heads[0].forward(x)
            

   
        for i in range(len(self.heads)-1):
            if encoder_output:
                concanated = concanated.concatanate(self.heads[i+1].forward(x,encoder_output))
            else:
                concanated = concanated.concatanate(self.heads[i+1].forward(x))

        z = concanated.matmul(self.w["w_o"]) 
        return z



