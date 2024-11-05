from AutoGrad import Array
import numpy as np

class EmbeddingLayer:
    def __init__(self,vocab_size,dmodel):
        self.w = {"embeddings": Array(np.random.randn(vocab_size,dmodel) * np.sqrt(2/dmodel))}

    def forward(self,x):
        indices = x.data
        output = Array(self.w["embeddings"].data[indices],(self.w["embeddings"],),"embedding")

        def _backward():
            self.w["embeddings"].gradient = np.zeros_like(self.w["embeddings"].gradient)
            self.w["embeddings"].gradient[indices] += output.gradient
            
        output._backward = _backward
        return output

