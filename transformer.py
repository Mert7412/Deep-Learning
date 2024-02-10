import numpy as np
from FNN import activations,derivativeofactivations,lossfunction,derlossfunction,layer

class positional_encoding:

    def forward_propagation(self,sentence):
        positions = np.arange(len(sentence[0]))
        dimensions = len(sentence[0,0])
        position_values = np.zeros((len(sentence),len(sentence[0]),dimensions))
        for j in range(int(dimensions/2)):
            position_values[:,:,2*j] = np.sin(positions/(10000**((2*j)/dimensions)))
            position_values[:,:,2*j+1] = np.cos(positions/(10000**((2*j)/dimensions)))
        return position_values + sentence


class MultiHeadAttention:
    def __init__(self,dmodel,h,masked = False):
        self.wq = np.random.randn(dmodel,dmodel)
        self.wk = np.random.randn(dmodel,dmodel)
        self.wv = np.random.randn(dmodel,dmodel)
        self.wq_i = np.random.randn(h,dmodel,int(dmodel/h))
        self.wk_i = np.random.randn(h,dmodel,int(dmodel/h))
        self.wv_i = np.random.randn(h,dmodel,int(dmodel/h))
        self.wo = np.random.randn(dmodel,dmodel)
        self.dmodel = dmodel
        self.h = h
        self.masked = masked

    def dot_product_attention(self,queries,keys,values):
        z = np.einsum("aij,ajk->aik",queries,np.transpose(keys,axes=(0,2,1)))/np.sqrt(self.dmodel)
        if self.masked:
            z = np.tril(z)
            ind = np.triu_indices(len(z[0]),k=1)
            z[:,ind[0],ind[1]] = -np.inf
        a = activations(z,"softmax")
        attention = np.einsum("aij,ajk->aik",a,values)
        return attention

    def forward(self,sentence,encoder_output = None):
        if not isinstance(encoder_output,np.ndarray):
            encoder_output = sentence
        queries = np.tensordot(sentence,self.wq,axes=((2),(0)))
        keys = np.tensordot(encoder_output,self.wk,axes=((2),(0)))
        values = np.tensordot(encoder_output,self.wv,axes=((2),(0)))

        queries_dk = np.tensordot(queries,self.wq_i,axes=((2),(1)))
        keys_dk = np.tensordot(keys,self.wk_i,axes=((2),(1)))
        values_dk = np.tensordot(values,self.wv_i,axes=((2),(1)))
        attention = []
        for i in range(self.h):
            attention.append(self.dot_product_attention(queries_dk[:,:,i],keys_dk[:,:,i],values_dk[:,:,i]))
        attention = np.dot(np.concatenate(attention,axis=-1),self.wo) + sentence
        return attention
    
class PointWiseFNN:
    def __init__(self,dmodel):
        self.w1 = np.random.randn(dmodel,dmodel*4)
        self.w2 = np.random.randn(dmodel*4,dmodel)
        self.dmodel = dmodel

    def forward(self,sentence):
        z1 = np.matmul(sentence,self.w1)
        a = activations(z1,"relu")
        z2 = np.matmul(a,self.w2) + sentence
        return z2
    
class LayerNormalization:
    def __init__(self,dmodel):
        self.alpha = np.random.randn(1,dmodel) * np.sqrt(2/dmodel)  
        self.beta = np.zeros((1,dmodel))
  
    def forward(self,x):
        self.x = x
        self.xmean = np.mean(self.x,axis=-1,keepdims=True)
        self.xvar = np.var(self.x,axis=-1,keepdims=True)
        self.normalize = (self.x - self.xmean) / (np.sqrt(self.xvar**2+1e-8))
        self.bn = self.alpha * self.normalize +self.beta
        return self.bn
    
class Encoder:
    def __init__(self,dmodel,h):
        self.network = [MultiHeadAttention(dmodel,h),LayerNormalization(dmodel),PointWiseFNN(dmodel),LayerNormalization(dmodel)]
    
    def forward_propagation(self,sentence):
        x = sentence
        for n in self.network:
            x = n.forward(x)
        return x
    
class Decoder:
    def __init__(self,dmodel,h):
        self.network = [MultiHeadAttention(dmodel,h,True),LayerNormalization(dmodel),MultiHeadAttention(dmodel,h)
                        ,LayerNormalization(dmodel),PointWiseFNN(dmodel),LayerNormalization(dmodel)]
        self.encoder_output = None
    def forward_propagation(self,sentence):
        x = sentence
        for n in self.network:
            if isinstance(n,MultiHeadAttention):
                x = n.forward(x,self.encoder_output)
            else:
                x = n.forward(x)
        return x
    
class Transformer:
    def __init__(self,N,dmodel,h):
        self.Encoders = [Encoder(dmodel,h) for i in range(N)]
        self.Decoders= [Decoder(dmodel,h) for i in range(N)]
        self.Encoders.insert(0,positional_encoding())
        self.Decoders.insert(0,positional_encoding())
        

    def forward(self,sentence1,sentence2):
        encoder_output = sentence1

        for encoder in self.Encoders:
            encoder_output = encoder.forward_propagation(encoder_output)
        
        decoder_output = sentence2
        for decoder in self.Decoders:
            decoder.encoder_output = encoder_output
            decoder_output = decoder.forward_propagation(decoder_output)
        
        return decoder_output
        


example = np.random.randn(50,10,256)

a = Transformer(6,256,8)
print(a.forward(example,example).shape)
