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
    
    def backward(self,gradient):
        pass


class MultiHeadAttention:
    def __init__(self,dmodel,h,masked = False):
        self.wq = np.random.randn(dmodel,dmodel) * np.sqrt(2/dmodel) 
        self.wk = np.random.randn(dmodel,dmodel) * np.sqrt(2/dmodel) 
        self.wv = np.random.randn(dmodel,dmodel) * np.sqrt(2/dmodel) 
        self.wo = np.random.randn(dmodel,dmodel) * np.sqrt(2/dmodel) 
        self.dmodel = dmodel
        self.h = h
        self.masked = masked

    def dot_product_attention(self,queries,keys,values):
        self.z = np.einsum("aibj,ajbk->aibk",queries,np.transpose(keys,axes=(0,3,2,1)))/np.sqrt(self.dmodel)
        if self.masked:
            self.z = np.transpose(self.z,axes=(0,2,1,3))
            self.z = np.tril(self.z)
            ind = np.triu_indices(len(self.z[0,0]),k=1)
            self.z[:,:,ind[0],ind[1]] = -np.inf
            self.z = np.transpose(self.z,axes=(0,2,1,3))
        self.a = activations(self.z,"softmax")
        attention = np.einsum("aibj,ajbk->aibk",self.a,values)
        return attention
    def der_dpattenton(self,gradient,queries,keys,values):
        gradient = gradient.reshape(values.shape)
        datt_dz = np.einsum("aibj,ajbk->aibk",gradient,np.transpose(values,axes=(0,3,2,1))) * derivativeofactivations(self.z,"softmax")
        dvalues = np.einsum("aibj,ajbk->aibk",self.a,gradient)
        dqueries = np.einsum("aibj,ajbk->aibk",datt_dz,keys) /np.sqrt(self.dmodel)
        dkeys = np.einsum("aibj,ajbk->aibk",datt_dz,queries) /np.sqrt(self.dmodel)
        sh = dkeys.shape
        return dqueries.reshape((sh[0],sh[1],self.dmodel)),dkeys.reshape((sh[0],sh[1],self.dmodel)),dvalues.reshape((sh[0],sh[1],self.dmodel))

    def forward(self,sentence,encoder_output = None):
        if not isinstance(encoder_output,np.ndarray):
            encoder_output = sentence
        self.queries = np.tensordot(sentence,self.wq,axes=((2),(0)))
        self.keys = np.tensordot(encoder_output,self.wk,axes=((2),(0)))
        self.values = np.tensordot(encoder_output,self.wv,axes=((2),(0)))
       
        self.queries_dk = self.queries.reshape((sentence.shape[0],sentence.shape[1],self.h,int(self.dmodel/self.h)))
        self.keys_dk = self.values.reshape((sentence.shape[0],sentence.shape[1],self.h,int(self.dmodel/self.h)))
        self.values_dk  = self.keys.reshape((sentence.shape[0],sentence.shape[1],self.h,int(self.dmodel/self.h)))
        self.concantenate =self.dot_product_attention(self.queries_dk,self.keys_dk,self.values_dk).reshape(self.queries.shape)
        
        
        attention = np.dot(self.concantenate,self.wo) + sentence
        return attention
    

    def backward(self,gradient,learningrate):
        dwo = np.matmul(np.transpose(self.concantenate,axes=(0,2,1)),gradient)
        self.wo -= learningrate * np.sum(dwo,axis=0)

        dattention = np.matmul(gradient,self.wo.T)
        datt_dqueriesdk ,datt_dkeysdk,datt_dvaluesdk= self.der_dpattenton(dattention,self.queries_dk,self.keys_dk,self.values_dk)
        dwq = np.tensordot(self.queries,datt_dqueriesdk,axes=((0,1),(0,1)))
        dwk = np.tensordot(self.keys,datt_dkeysdk,axes=((0,1),(0,1)))
        dwv = np.tensordot(self.values,datt_dvaluesdk,axes=((0,1),(0,1)))

        dq = np.tensordot(datt_dqueriesdk,self.wq,axes=((2),(1)))
        dk = np.tensordot(datt_dkeysdk,self.wk,axes=((2),(1)))
        dv = np.tensordot(datt_dvaluesdk,self.wv,axes=((2),(1)))

        self.wq -= learningrate * dwq
        self.wk -= learningrate * dwk
        self.wv -= learningrate * dwv
        return dq,dk,dv
    
class PointWiseFNN:
    def __init__(self,dmodel):
        self.w1 = np.random.randn(dmodel,dmodel*4) * np.sqrt(1/(2*dmodel)) 
        self.w2 = np.random.randn(dmodel*4,dmodel) * np.sqrt(2/dmodel) 
        self.dmodel = dmodel

    def forward(self,x):
        self.x = x
        self.z1 = np.matmul(x,self.w1)
        self.a = activations(self.z1,"relu")
        z2 = np.matmul(self.a,self.w2) + x
        return z2
    
    def backward(self,gradient,learningrate):
        dw2 = np.matmul(np.transpose(self.a,axes=(0,2,1)),gradient)
        self.w2 -= learningrate * np.sum(dw2,axis=0)
        dz2_dz1 = np.matmul(gradient,self.w2.T) * derivativeofactivations(self.z1,"relu")
        dw1 = np.matmul(np.transpose(self.x,axes=(0,2,1)),dz2_dz1)
        self.w1 -= learningrate * np.sum(dw1,axis=0)
        dx = np.matmul(dz2_dz1,self.w1.T) + 1
        return dx
    
class LayerNormalization:
    def __init__(self,dmodel):
        self.alpha = np.random.randn(10,dmodel) * np.sqrt(2/dmodel) 
        self.beta = np.zeros((10,dmodel))
  
    def forward(self,x):
        self.x = x
        self.xmean = np.mean(self.x,axis=-1,keepdims=True)
        self.xvar = np.var(self.x,axis=-1,keepdims=True)
        self.substrmx = self.x-self.xmean
        self.normalize = (self.substrmx) / (np.sqrt(self.xvar**2+1e-8))
        self.ln = self.alpha * self.normalize +self.beta
        return self.ln
    
    def backward(self,gradient,learningrate):
        dalpha = np.sum(self.normalize * gradient,axis=0)
        dbeta = np.sum(gradient,axis=0)
        self.alpha -= learningrate * dalpha
        self.beta -= learningrate * dbeta

        n = len(gradient[0][0])
        dln_dnormalize = self.alpha * gradient
        dmean_dx = (1/n )* np.sum(self.x,axis=-1,keepdims=True)
        dvar_dx = (1/n )* np.sum(self.substrmx,axis=-1,keepdims=True)
        dnormalize_dx = 1 + (-dmean_dx)+ (self.substrmx*(-1/2*(self.xvar**2+1e-8)-3/2)*dvar_dx)
        dx = dln_dnormalize*dnormalize_dx
        return dx
    
class Encoder:
    def __init__(self,dmodel,h):
        self.network = [MultiHeadAttention(dmodel,h),LayerNormalization(dmodel),PointWiseFNN(dmodel),LayerNormalization(dmodel)]
    
    def forward_propagation(self,sentence):
        x = sentence
        for n in self.network:
            x = n.forward(x)
        return x
    def back_propagation(self,gradient,learningrate):
        y = gradient
        for n in reversed(self.network):
            
            if isinstance(n,MultiHeadAttention):
                a = n.backward(y,learningrate)
                y = a[0]+a[1]+a[2]
            else:
                y =n.backward(y,learningrate)
        return y
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
    
    def back_propagation(self,gradient,learningrate):
        y = gradient
        dencoder = 0
        run = True
        for n in reversed(self.network):
            
            if isinstance(n,MultiHeadAttention) and run:
                a = n.backward(y,learningrate)
                y = a[0]
                dencoder += a[1]+a[2]
                run = False
            elif isinstance(n,MultiHeadAttention):
                a = n.backward(y,learningrate)
                y = a[0]+a[1]+a[2]
            else:
                y = n.backward(y,learningrate)
        return dencoder ,y
    
class Transformer:
    def __init__(self,N,dmodel,h):
        self.Encoders = [Encoder(dmodel,h) for i in range(N)]
        self.Decoders= [Decoder(dmodel,h) for i in range(N)]
        """self.Encoders.insert(0,positional_encoding())
        self.Decoders.insert(0,positional_encoding())
        """

    def forward(self,sentence1,sentence2):
        encoder_output = sentence1

        for encoder in self.Encoders:
            encoder_output = encoder.forward_propagation(encoder_output)
        
        decoder_output = sentence2
        for decoder in self.Decoders:
            decoder.encoder_output = encoder_output
            decoder_output = decoder.forward_propagation(decoder_output)
        
        return decoder_output
    
    def backward(self,sentence1,sentence2,learningrate):
        ypred = self.forward(sentence1,sentence2)
        dloss = derlossfunction(sentence2,ypred,"crossentropy")
        dencoder = 0
        for n in reversed(self.Decoders):
            back = n.back_propagation(dloss,learningrate)
            dloss = back[1]
            dencoder += back[0]
     
        for n in reversed(self.Encoders):
            dencoder = n.back_propagation(dencoder,learningrate)
        return dencoder
        


example = np.random.randn(50,10,256)

a  = Transformer(6,256,8)
a.forward(example,example)
print(a.backward(example,example,0.0001))
