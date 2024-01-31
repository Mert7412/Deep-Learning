import numpy as np
from FNN import *
import idx2numpy


data = idx2numpy.convert_from_file(r"data\data3\train-images.idx3-ubyte")
y = idx2numpy.convert_from_file(r"data\data3\train-labels.idx1-ubyte")
y1 = np.zeros((y.shape[0],int(y.max()-y.min()+1)))

y1[np.arange(y.shape[0]),np.array(y,dtype=int)] = 1

x = (data - np.mean(data,axis=0) )/ (np.std(data,axis=0)+1e-8)
class RNNlayer:
    def __init__(self,inputs,hs,output,outputactivation,hiddenactivation,isoutputlayer,rnntype="none",):
        self.wxh = np.random.randn(inputs,hs)*0.00001
        self.whh = np.random.randn(hs,hs)*0.00001
        self.bh = np.zeros((hs))
        self.wy = np.random.randn(hs,output)*0.00001
        self.by = np.zeros((output))
        self.hss = hs
        self.ous = output
        self.outputactivation = outputactivation
        self.hiddenactivation =hiddenactivation
        self.isoutputlayer = isoutputlayer
        self.rnntype = rnntype
    def forward(self,x):
        self.x = x
        self.xs = x.shape
        self.z = np.zeros((self.xs[0],self.hss,self.xs[2]))
        self.outputs = np.zeros((self.xs[0],self.ous,self.xs[2]))
        self.hiddenstate = np.zeros((self.xs[0],self.hss,self.xs[2]+1))
        for t in range(self.xs[2]):
            
            self.z[:,:,t] = np.dot(x[:,:,t],self.wxh) + np.dot(self.hiddenstate[:,:,t-1],self.whh) + self.bh
            self.hiddenstate[:,:,t] = activations(self.z[:,:,t],self.hiddenactivation)
            if self.isoutputlayer:
                self.y = np.dot(self.hiddenstate[:,:,t],self.wy) + self.by
                self.o = activations(self.y,self.outputactivation)
                self.outputs[:,:,t] = self.o

        if self.rnntype == "many_to_one":
            return self.outputs[:,:,-1]
        else:
            return self.outputs
    def backward(self,loss,learningrate):
        dl_dwhh ,dl_dwxh, dl_dwy = np.zeros_like(self.whh),np.zeros_like(self.wxh),np.zeros_like(self.wy)
        dl_dbh ,dl_dby = np.zeros_like(self.bh),np.zeros_like(self.by)
        h_next = np.zeros((self.xs[0],self.hss))
        dl_dx = np.zeros_like(self.x)
        if self.isoutputlayer:
            if "many_to_one" == self.rnntype:
                
                delta = loss * derivativeofactivations(self.outputs[:,:,-1],self.outputactivation)
                for i in reversed(range(self.xs[2])):
                    if i == self.xs[2]-1 :
                        dl_dwy += np.dot(self.hiddenstate[:,:,i].T,delta)
                        dl_dby += np.sum(delta,axis=0)
                        delta = np.dot(delta,self.wy.T)
                    
                    dl_dh = delta *derivativeofactivations(self.z[:,:,i],self.hiddenactivation)
                    dl_dwhh += np.dot(self.hiddenstate[:,:,i-1].T,dl_dh)
                    dl_dwxh += np.dot(self.x[:,:,i].T,dl_dh)
                    dl_dbh += np.sum(dl_dh,axis=0)
                    delta = np.dot(delta,self.whh)
                    dl_dx[:,:,i] = np.dot(delta,self.wxh.T)
            elif "many_to_many":
                pass
        else:
            loss = loss[:,:,None] if len(loss.shape) == 2 else loss
            for i in reversed(range(self.xs[2])):
                dl_dh = (h_next + loss[:,:,i]) * derivativeofactivations(self.z[:,:,i],self.hiddenactivation)
                dl_dwhh += np.dot(self.hiddenstate[:,:,i-1].T,dl_dh)
                dl_dwxh += np.dot(self.x[:,:,i].T,dl_dh)
                dl_dbh += np.sum(dl_dh,axis=0)
                h_next += np.dot(dl_dh,self.whh)
                dl_dx[:,:,i] = np.dot(dl_dh,self.wxh.T)

        self.whh -= learningrate * dl_dwhh
        self.wxh -= learningrate * dl_dwxh
        self.wy -= learningrate * dl_dwy
        self.by -= learningrate * dl_dby
        self.bh -= learningrate * dl_dbh

        return dl_dx
class rnnnetwork:
    def __init__(self,input,hidden,output,ha = "tanh",oa="softmax",numberoflayer=1):
        self.rnnlayers = []
        for i in range(numberoflayer):
            if i == numberoflayer-1:
                self.rnnlayers.append(RNNlayer(input,hidden[i],output,outputactivation=oa,hiddenactivation=ha,isoutputlayer=True,rnntype="many_to_one"))
    
            else :
                self.rnnlayers.append(RNNlayer(input,hidden[i],output,outputactivation=oa,hiddenactivation=ha,isoutputlayer=False))
            input = hidden[i] 

        self.nlayer = numberoflayer

    def forwardpropagation(self,x):   
        y = x

        for n in self.rnnlayers:
            output = n.forward(y)
            y = n.hiddenstate[:,:,:-1]
        return output    
    
    def backpropagation(self,x,y,learninrate,lossf):
        ypred = self.forwardpropagation(x)
        loss = derlossfunction(y,ypred,lossf)
      
        for n in reversed(self.rnnlayers):
            loss = n.backward(loss,learninrate)

    def gradientdescent(self,x,y,learningrate,iteration,lossf="crossentropy"):
        for j in range(iteration):
            self.backpropagation(x,y,learningrate,lossf)
            ypred = self.forwardpropagation(x)
            print(lossfunction(y,ypred,lossf))
        print(np.unique(np.argmax(y,axis=1),return_counts=True))
        print(np.unique(np.argmax(ypred[:,:,-1],axis=1),return_counts=True))

x = x[:1000]
y = y1[:1000]
a = rnnnetwork(28,[100,80],10)
a.gradientdescent(x,y,0.001,1000)