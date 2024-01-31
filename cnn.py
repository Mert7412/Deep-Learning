import numpy as np
import pandas as pd
import idx2numpy
from FNN import *

trainingdata = idx2numpy.convert_from_file(r"data\data3\train-images.idx3-ubyte")
tsh = trainingdata.shape
trainingdata = trainingdata.reshape(tsh[0],tsh[1],tsh[2],1)
traininglables = idx2numpy.convert_from_file(r"data\data3\train-labels.idx1-ubyte")
y1 = np.zeros((traininglables.shape[0],int(traininglables.max()-traininglables.min()+1)))


y1[np.arange(traininglables.shape[0]),np.array(traininglables,dtype=int)] = 1
class activationlayer:
    def __init__(self,activation):
        self.activation = activation
    
    def forward(self,x):
        self.x = x
        self.a = activations(x,self.activation)
        return self.a
    
    def backward(self,loss,learningrate):
        lo = loss * derivativeofactivations(self.x,self.activation)
        return lo

class convolutional_layer:
    def __init__(self,numberoffilter=32,filtersize=3,activation="sigmoid"):

        self.numberoffilter = numberoffilter
        self.activation = activation
        self.filtersize = filtersize
        
     

    def filter(self,x,p=1,s=1):
        self.s = s
        self.x = x
        self.xs = self.x.shape
        self.filt = np.random.randn(self.filtersize,self.filtersize,self.xs[3],self.numberoffilter)
        self.fs = self.filt.shape
 
        self.outputshapeh = int(((self.xs[1]-self.fs[0])/s+2*p)+1)
        self.outputshapew = int(((self.xs[2]-self.fs[1])/s+2*p)+1)

        filtersize = self.fs[0]
        self.x = np.pad(self.x,((0,0),(p,p),(p,p),(0,0)))
     
        filtered = np.zeros((self.xs[0],self.outputshapeh,self.outputshapew,self.fs[3]))
        
        for i in range(self.outputshapeh):
            for j in range(self.outputshapew):
                f = self.x[:,i*self.s:i*self.s+filtersize,j*self.s:j*self.s+filtersize].reshape(self.xs[0],self.filtersize,self.filtersize,self.xs[3],1)       
                filtered[:,i,j] = np.sum(f*self.filt ,axis=(1,2,3))
        return filtered

    def forward(self,x):
        self.z = self.filter(x)
        
        return self.z
    
    def backward(self,loss,learningrate):
        rloss = np.zeros((self.xs[0],self.xs[1],self.xs[2],self.xs[3]))
        
        dw = 0
    
        for i in range(self.outputshapeh):
            for j in range(self.outputshapew):  
                dw += self.x[:,i*self.s:i*self.s+self.filtersize,j*self.s:j*self.s+self.filtersize].reshape(self.xs[0],self.filtersize,self.filtersize,self.xs[3],1)*loss[:,i,j].reshape(self.xs[0],1,1,1,loss.shape[3])    
                pad = np.pad(loss,((0,0),(self.filtersize-1,0),(self.filtersize-1,0),(0,0)))
                sa = np.sum(pad[:,i*self.s:i*self.s+self.filtersize,j*self.s:j*self.s+self.filtersize].reshape(self.xs[0],self.filtersize,self.filtersize,1,self.fs[3])*self.filt,axis=(1,2,4))
            
                rloss[:,i,j] = sa
        
        dw = np.sum(dw,axis=0)
         
        self.filt -= learningrate*dw

        return rloss
                
    
class pooling:
    def __init__(self,s=2):
        self.s = s

    def pooling(self,filtered):
        self.filtered = filtered
        fs0,fs1,fs2,fs3 = filtered.shape
        fs22 = int(fs1/self.s)
        fs33 = int(fs2/self.s)
        self.pooled = np.zeros((fs0,fs22,fs33,fs3))
        self.imax = np.zeros((fs0,fs22,fs33,fs3),dtype=int)
        for i in range(fs22):
            for j in range(fs33):
                self.fil = filtered[:,i*self.s:i*self.s+self.s,j*self.s:j*self.s+self.s].reshape(fs0,2*self.s,fs3)
                self.pooled[:,i,j] = np.amax(self.fil,axis=1)        
                self.imax[:,i,j] = np.argmax(self.fil,axis=1)
        return self.pooled

    def forward(self,x):
        f = self.pooling(x)
        return f
    def backward(self,loss,learningrate):
        ls = loss.shape
    
        lo = np.zeros((ls[0],self.s*ls[1],self.s*ls[2],ls[3]))
        for i in range(ls[1]):
            for j in range(ls[2]):
                l = lo[:,i*self.s:i*self.s+self.s,j*self.s:j*self.s+self.s].reshape(ls[0],self.s*2,ls[3]) 
                lss = loss[:,i,j]
                im = self.imax[:,i,j]
                for e in range(ls[3]):
                     l[np.arange(ls[0]),im[:,e].flatten(),e] = lss[np.arange(ls[0]),e]
                lo[:,i:i+self.s,j:j+self.s] = l.reshape(ls[0],self.s,self.s,ls[3])
        return lo
class flat:
    def forward(self,x):
        self.xs = x.shape
        return x.reshape(self.xs[0],self.xs[1]*self.xs[2]*self.xs[3])
    def backward(self,loss,learningrate):
        return loss.reshape(self.xs)
    
class batchnormalization:  
    def forward(self,x):
        self.x = x
        xsh = self.x.shape
        self.alpha = np.random.randn(1,1,1,xsh[3])   
        self.beta = np.zeros((1,1,1,xsh[3]))
        
        self.xmean = np.mean(self.x,axis=(0,1,2),keepdims=True)
        
        self.xvar = np.var(self.x,axis=(0,1,2),keepdims=True)
        self.normalize = (self.x - self.xmean) / (np.sqrt(self.xvar+1e-8))
        
        self.bn = self.alpha * self.normalize +self.beta
        return self.bn
    def backward(self,loss,learningrate):
       
        dalpha = np.sum(loss * self.normalize,axis=(0,1,2),keepdims=True)
      
        dbeta = np.sum(loss,axis=(0,1,2),keepdims=True)
        l = len(self.x)
        dnx = self.alpha* loss
        dvar = np.sum(dnx * (-1/2*(self.x-self.xmean)*((self.xvar+1e-8)**(-3/2))),axis=(0,1,2),keepdims=True)
        dmean = np.sum(dnx * (-1/np.sqrt(self.xvar+1e-8)),axis=(0,1,2),keepdims=True ) +( dvar * np.mean(-2*(self.x-self.xmean),axis=(0,1,2),keepdims=True))
        dx = dnx*( 1/np.sqrt(self.xvar+1e-8)) + dvar *((2/l)*(self.x-self.xmean)) + dmean /l
        self.alpha -=   dalpha
        self.beta -= dbeta
        return dx
        
class cnnetwork:
    def __init__(self,sizei,ls,activation="sigmoid",outputactivation="softmax"):
        self.cnn = [batchnormalization(),convolutional_layer(),batchnormalization(),activationlayer(activation),pooling(),convolutional_layer(numberoffilter=32),batchnormalization(),activationlayer(activation),pooling(),flat()]
        self.n = FNNnetwork(sizei,ls,activation,outputactivation)
        for i in self.n.network:

            self.cnn.append(i)
    def forwardp(self,x):
        y = x 
        
        for n in self.cnn:
            
            y = n.forward(y)
    
      
        return y
    def backpro(self,x,y,learningrate,lossf = "mse"):

        yp = self.forwardp(x)
        loss = derlossfunction(y,yp,lossf)
        
        for n in reversed(self.cnn):
            
            loss = n.backward(loss,learningrate)
    def gradientdescent(self,x,y,learningrate,iteration,lossf = "mse"):
        for i in range(iteration):
            self.backpro(x,y,learningrate,lossf)
            pl = 0
            yp = self.forwardp(x)
            l = lossfunction(y,yp,lossf)
            print(f"iteration:{i+1} ,loss:{l}")
    
    def stochasticgradient(self,x,y,learningrate,iteration,lossf = "mse"):
        for m in range(iteration):
            index = np.array(range(len(x)))
            np.random.shuffle(index)
            x = x[index]
            y = y[index]
            i = np.random.randint(0,len(x),size=1)
            self.backpro(x[i],y[i],learningrate,lossf)
            yp = self.forwardp(x[i])
            l = lossfunction(y[i],yp,lossf) 
            print(f"iteration:{m+1} ,loss:{l}")
    
    def mini_batchgradient(self,x,y,learningrate,iteration,batchsize = 200,lossf = "mse"):
        for m in range(iteration):
            index = np.array(range(len(x)))
            np.random.shuffle(index)
            x = x[index]
            y = y[index]
            i = np.random.randint(0,len(x),size=batchsize)
            self.backpro(x[i],y[i],learningrate,lossf)
            yp = self.forwardp(x[i]) 
            l = lossfunction(y[i],yp,lossf)
            print(f"iteration:{m+1} ,loss:{l}")
    def predict(self,x):
        yp = self.forwardp(x)
        return np.argmax(yp,axis=1)
a = np.random.randint(0,60000,size=1000)

ls = [100,150,len(y1[0])]
sizei = 1568
cn = cnnetwork(sizei,ls)
cn.stochasticgradient(trainingdata[a],y1[a],0.001,2500,lossf="crossentropy")






