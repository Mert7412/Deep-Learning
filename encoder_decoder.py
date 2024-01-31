from LSTM import *
import numpy as np
import idx2numpy

data = idx2numpy.convert_from_file(r"data\data3\train-images.idx3-ubyte")
y = idx2numpy.convert_from_file(r"data\data3\train-labels.idx1-ubyte")
y1 = np.zeros((y.shape[0],int(y.max()-y.min()+1)))


y1[np.arange(y.shape[0]),np.array(y,dtype=int)] = 1

x = (data - np.mean(data,axis=0) )/ (np.std(data,axis=0)+1e-8)

class sequential_encoder_decoder:
    def __init__(self,embedding = False):
        self.encoderlayers = [Lstm([1000,28,28],100,"none",embedding),Lstm([1000,100,28],10,"many_to_one")] 
        self.decoderlayers = [Lstm([1000,28,28],100,"none",isdecoder=True),Lstm([1000,100,28],10,"many_to_one",isdecoder=True)]

    def encoder(self,x):
        self.y = x
        self.shortterm = []
        self.longterm = []
        
        for n in self.encoderlayers:
            self.y = n.forward(self.y)
            self.shortterm.append(n.shortermemory[-2])
            self.longterm.append(n.longtermemory[-2])
        return self.shortterm,self.longterm ,self.y
    
    def decoder(self,x,shorterm,longterm):
        x = np.insert(x,0,np.zeros_like(x[:,:,0]),axis=2)
        self.y = x
        for i in range(len(self.decoderlayers)):
            self.decoderlayers[i].longtermemory[-1] = longterm[i]
            self.decoderlayers[i].shortermemory[-1] = shorterm[i]
            self.y = self.decoderlayers[i].forward(self.y)
        
        return self.y
    
    def backprogapagiton(self,x,y,learningrate,lossf="mse"):
        encoding = self.encoder(x)
        decoding = self.decoder(x,encoding[0],encoding[1])
        
        loss = derlossfunction(y,decoding,lossf).reshape(1,10,1)
     
        
        dl_dc = []
        dl_dh = []
        for n in reversed(self.decoderlayers):
            
            bacwardvalues =  n.backward(loss,learningrate)
            loss = bacwardvalues[0]
            
            
            dl_dc.append(bacwardvalues[1])
            dl_dh.append(bacwardvalues[2])
        loss = np.zeros_like(encoding[2])

        i = 0
        for n in reversed(self.encoderlayers):
            loss = n.backward(loss,learningrate,dl_dc[i],dl_dh[i])
            i+=1
            
 
    def gradientdescent(self,x,y,learningrate,iteration,lossf):
        for i in range(iteration):
            self.backprogapagiton(x,y,learningrate,lossf)

            print(lossfunction(y,self.y,lossf))

            

x = x[:1000]
y = y1[:1000]
b = sequential_encoder_decoder(embedding=True) 
b.gradientdescent(x,y,1e-4,1000,"mse")
