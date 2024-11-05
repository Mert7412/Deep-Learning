import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from AutoGrad import Array
from Network import Network
from Layers import MultiHeadAttention,Normalization,Linear,EmbeddingLayer,PositionalEncoding
from Activations import Relu,Sigmoid,Softmax
from Optimizers import MiniBatchGradientDescent
from LossFunctions import CrossEntropy

def take_weights(layers,w,b,name):
    for i in range(len(layers)):
        if hasattr(layers[i],"w"):
            for k in range(len(layers[i].w.values())):
                w[f"{name}{i}_layer{i}-w{k}"] = list(layers[i].w.values())[k]
        if hasattr(layers[i],"b"):
            for j in range(len(layers[i].b.values())):
                b[f"{name}{i}_layer{i}-b{j}"] = list(layers[i].b.values())[j]

class TData:
    def __init__(self,data):
        self.data = data
    
    def __getitem__(self,index):
        return (self.data[0][index],self.data[1][index])

class Encoder:
    def __init__(self,dmodel,dk,heads):
        self.Layers = [MultiHeadAttention(dmodel,int(dk/heads),dmodel,heads),Normalization(dmodel,-1)
                       ,Linear(dmodel,4*dmodel),Relu(),Linear(4*dmodel,dmodel),Normalization(dmodel,-1)]
        self.w = {}
        self.b = {}
        take_weights(self.Layers,self.w,self.b,"encoder")
        
    def forward(self,x):
        y = x
        residual_connection = x
        for layer in self.Layers:
            if type(layer) == Normalization:
                y = layer.forward(y+residual_connection)
                residual_connection = y

            else:
                y = layer.forward(y)

        return y

class decoder:
    def __init__(self,dmodel,dk,heads):
        self.layers = [MultiHeadAttention(dmodel,int(dk/heads),dmodel,heads,True),Normalization(dmodel,-1),
                       MultiHeadAttention(dmodel,int(dk/heads),dmodel,heads),Normalization(dmodel,-1),
                       Linear(dmodel,4*dmodel),Relu(),Linear(4*dmodel,dmodel),Normalization(dmodel,-1)]
        self.w = {}
        self.b = {}
    
        take_weights(self.layers,self.w,self.b,"decoder")
    def forward(self,x,encoder_output):
        y = x
        residual_connection = x
        for i in range(len(self.layers)):
            if type(self.layers[i]) == Normalization:
                y = self.layers[i].forward(y+residual_connection)
                residual_connection = y

            elif i == 2:
                y = self.layers[i].forward(y,encoder_output)
                
            else:
                y = self.layers[i].forward(y)

        return y
    
class Transformer:
    def __init__(self,dmodel,dk,heads,encoder_vocab_size,decoder_vocab_size,n):
        self.encoder_layers = [EmbeddingLayer(encoder_vocab_size,dmodel),PositionalEncoding(),*(Encoder(dmodel,dk,heads) for i in range(n))]
        self.decoder_layers = [EmbeddingLayer(decoder_vocab_size,dmodel),PositionalEncoding(),*(decoder(dmodel,dk,heads) for i in range(n)),
                               Linear(dmodel,decoder_vocab_size),Sigmoid()]
        self.weights = {}
        self.bias = {}
        self.params = {}
    
        take_weights(self.encoder_layers,self.weights,self.bias,"encoder")
        take_weights(self.decoder_layers,self.weights,self.bias,"decoder")

        self.params.update(self.weights)
        self.params.update(self.bias)

        self.decoder_vocab_size = decoder_vocab_size
        

    def forward_propagation(self,data):
        encoder_y = data[0]
        decoder_y = data[1]
        for i in range(len(self.encoder_layers)):
            encoder_y = self.encoder_layers[i].forward(encoder_y)
            if i < 2:
                decoder_y = self.decoder_layers[i].forward(decoder_y)
            else:
                decoder_y = self.decoder_layers[i].forward(decoder_y,encoder_y)
        decoder_y = self.decoder_layers[i+1].forward(decoder_y)
        decoder_y = self.decoder_layers[i+2].forward(decoder_y)
        return decoder_y
    
    def back_propagation(self,y,ypred,loss_function):
        y = Array(np.eye(self.decoder_vocab_size)[y.data])
        loss = loss_function(y,ypred)
        loss.backward()

        return loss
        
en_data = Array(np.load(r"data\en_es_dataset\en_sentences.npy"))
en_vocab = np.load(r"data\en_es_dataset\en_vocab.npy",allow_pickle=True).item()

es_data = Array(np.load(r"data\en_es_dataset\es_sentences.npy"))
es_vocab = np.load(r"data\en_es_dataset\es_vocab.npy",allow_pickle=True).item()


model = Transformer(128,128,4,len(en_vocab),len(es_vocab),4)

opt = MiniBatchGradientDescent(model,CrossEntropy().forward)
opt.train(TData((en_data,es_data)),es_data,1e-5,1000,10)


