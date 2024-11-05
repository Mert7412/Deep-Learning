import numpy as np
from AutoGrad import Array


class RNNLayer:
    def __init__(self,input_dimension,hidden_dimesion,input_activation,type="many_to_many"
                 ,output_layer = True,output_activation = None,output_dimesion = None):
        self.w = {"wxh" : Array(np.random.randn(input_dimension,hidden_dimesion) * np.sqrt(2/hidden_dimesion)),
                  "whh" : Array(np.random.randn(hidden_dimesion,hidden_dimesion) * np.sqrt(2/hidden_dimesion))}
         
        self.b = {"bh" : Array(np.zeros((1,hidden_dimesion)))}

        if output_layer:
            self.w["whq"] = Array(np.random.randn(hidden_dimesion,output_dimesion) * np.sqrt(2/output_dimesion))
            self.b["bq"] = Array(np.zeros((1,output_dimesion)))



        self.hidden_d = hidden_dimesion
        self.iact = input_activation
        
        self.rnntype = type
        self.output_layer = output_layer
        self.oact = output_activation

    def RNN_cells(self,x,hidden_state):
        z = self.iact(x.matmul(self.w["wxh"]) + hidden_state.matmul(self.w["whh"]) + self.b["bh"]) 
        return z
    
    def forward(self,x):
        if type(x) != list:
            x = [x[:,i,:] for i in range(len(x[0]))]
        hidden_state = Array(np.zeros((len(x[0]),self.hidden_d)))
     
        output_sequence = []
        
        for i in range(len(x)):
            
            output = self.RNN_cells(x[i],hidden_state)
            hidden_state = output
            if self.output_layer:
                
                output = self.oact(hidden_state.matmul(self.w["whq"]) + self.b["bq"])
            output_sequence.append(output)
        
        if self.rnntype == "many_to_one":
            return output_sequence[-1]
        else:
            return output_sequence
        

    


        


