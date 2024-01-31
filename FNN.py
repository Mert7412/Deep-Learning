import pandas as pd
import numpy as np
"""data = pd.read_csv(r"deep-learning,\data\data1\winequality-red.csv",sep=";").values
x = data.T[:-1].T

y = data.T[-1]-3
y1 = np.zeros((y.shape[0],int(y.max()-y.min()+1)))

y1[np.arange(y.shape[0]),np.array(y,dtype=int)] = 1"""
"""x = (x - np.mean(x,axis=0)) / np.std(x,axis=0)"""

def activations(z,activation):
    if activation == "none":
        return z
    elif activation == "relu":
        return np.maximum(0,z)
    elif activation == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif activation == "softmax":
        ex = np.exp(z.T-np.max(z,axis=1)).T
        return (ex.T/ np.sum(ex,axis=1)).T
    elif activation == "tanh":
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)+1e-8)

def derivativeofactivations(z,activation):    
    if activation == "relu":
        z[z >= 0] = 1
        z[z < 0] = 0
        return z
    elif activation == "sigmoid":
        return activations(z,"sigmoid") * (1 - activations(z,"sigmoid"))
    elif activation == "softmax":
        return np.ones(z.shape)
    elif activation == "tanh":
        return 1 - activations(z,"tanh")**2
    elif activation == "none":
        return np.ones(z.shape)
    
def lossfunction(y,ypred,lossfunction):
    if lossfunction == "mse":
        return np.mean((y-ypred)**2)
    if lossfunction == "crossentropy":
        return -np.mean(y*np.log(ypred))
    
def derlossfunction(y,ypred,lossfunction):
    if lossfunction == "mse":
        return np.mean((-2)*(y-ypred),axis=0)
    if lossfunction == "crossentropy":
        return ypred - y

"""def dersoftmax(z):
    return -(softmax(z)[:,:,None]*softmax(z)[:,None,:]) + (softmax(z)[:,None,:] * np.diag(np.ones(len(z[0]))))
"""
def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return  1 - (float(np.count_nonzero(diff)) / len(diff))

class activationlayer:
    def __init__(self,activation):
        self.activation = activation

    def forward(self,x):
        self.x = x
        self.a = activations(self.x,self.activation)
        return self.a
    
    def backward(self,loss,learningrate):
        l = loss * derivativeofactivations(self.x,self.activation)
        return l
    
class batcnormalization:
    def __init__(self,sizei):
        self.alpha = np.random.randn(1,sizei) * np.sqrt(2/sizei)  
        self.beta = np.zeros((1,sizei))
  
    def forward(self,x):
        self.x = x
        self.xmean = np.mean(self.x,axis=0,keepdims=True)
        self.xvar = np.var(self.x,axis=0,keepdims=True)
        self.normalize = (self.x - self.xmean) / (np.sqrt(self.xvar+1e-8))
        self.bn = self.alpha * self.normalize +self.beta
        return self.bn
    
    def backward(self,loss,learningrate):
        dalpha = np.sum(loss * self.normalize,axis=0,keepdims=True)      
        dbeta = np.sum(loss,axis=0,keepdims=True)
        l = len(self.x)
        dnx = self.alpha* loss
        dvar = np.sum(dnx * (-1/2*(self.x-self.xmean)*((self.xvar+1e-8)**(-3/2))),axis=0,keepdims=True)
        dmean = np.sum(dnx * (-1/np.sqrt(self.xvar+1e-8)),axis=0,keepdims=True ) +( dvar * np.mean(-2*(self.x-self.xmean),axis=0,keepdims=True))
        dx = dnx*( 1/np.sqrt(self.xvar+1e-8)) + dvar *((2/l)*(self.x-self.xmean)) + dmean * 1/l
        self.alpha -=  learningrate*dalpha
        self.beta -= learningrate*dbeta
        return dx
    
class layer:
    def __init__(self,sizei,soutput,activation):
        self.w = np.random.randn(sizei,soutput) * np.sqrt(2/soutput)
        self.b = np.zeros(soutput)
        self.activation = activation

    def forward(self,x):
        self.x = x
        self.z = np.dot(self.x,self.w) +self.b
        self.a = activations(self.z,self.activation)
        return self.a
        
    def backward(self,loss,learningrate): 
        loss = loss *derivativeofactivations(self.z,self.activation)         
        self.dw = np.dot(self.x.T ,loss)
        self.db = np.sum(loss,axis=0)
        self.w -= learningrate * self.dw
        self.b -= learningrate * self.db
        return np.dot(loss,self.w.T)
    
class FNNnetwork:
    def __init__(self,sizei,ls,activation = "sigmoid",outputactivation = "none"):
        self.network = []
        self.network.append(batcnormalization(sizei))
        self.network.append(layer(sizei,ls[0],activation))
        for i in range(len(ls)-1):
            if i == len(ls)-2:
                self.network.append(batcnormalization(ls[i]))
                self.network.append(layer(ls[i],ls[i+1],outputactivation))   
            else:
                self.network.append(batcnormalization(ls[i]))
                self.network.append(layer(ls[i],ls[i+1],activation))

    def feed_forward_pro(self,x):
        self.y = x 
        for n in self.network:
            self.y = n.forward(self.y)
        return self.y

    def backpro(self,x,y,learningrate,lossf = "mse"):
        yp = self.feed_forward_pro(x)
        loss = derlossfunction(y,yp,lossf)
    
        for n in reversed(self.network):
            
            loss = n.backward(loss,learningrate)

    def gradientdescent(self,x,y,learningrate,iteration,lossf = "mse"):
        for i in range(iteration):
            self.backpro(x,y,learningrate,lossf)
            pl = 0
            yp = self.feed_forward_pro(x)
            l = lossfunction(y,yp,lossf)
            print(l)
          
    def stochasticgradient(self,x,y,learningrate,iteration,lossf = "mse"):
        for m in range(iteration):
            index = np.array(range(len(x)))
            np.random.shuffle(index)
            x = x[index]
            y = y[index]
            i = np.random.randint(0,len(x),size=1)
            self.backpro(x[i],y[i],learningrate,lossf)
            yp = self.feed_forward_pro(x[i])
            l = lossfunction(y[i],yp,lossf) 
            print(l)

    def mini_batchgradient(self,x,y,learningrate,iteration,batchsize = 200,lossf = "mse"):
        for m in range(iteration):
            index = np.array(range(len(x)))
            np.random.shuffle(index)
            x = x[index]
            y = y[index]
            i = np.random.randint(0,len(x),size=batchsize)
            self.backpro(x[i],y[i],learningrate)
            yp = self.feed_forward_pro(x[i])
            l = lossfunction(y[i],yp,lossf)
            print(l)
  
    def predict(self,x):
        yp = self.feed_forward_pro(x)
        return np.argmax(yp,axis=1)
        

"""ls = [30,25,20,len(y1[0])]
sizei = len(x[0])
a = layer(sizei,7,"softmax")

b = network(sizei,ls,activation="relu",outputactivation="softmax")

s = a.forward(x)
b.mini_batchgradient(x,y1,0.00001,1,1000,"crossentropy")
print(np.unique(y,return_counts=True))
print(np.unique(b.predict(x),return_counts=True))
"""
"""ass = batcnormalization(sizei)
ass.forward(x)
print(ass.backward(a.backward(s,0.001),0.001))
"""

"""b.stochasticgradient(x,y1,0.001,4000)"""
"""b.gradientdescent(x,y1,0.0001,2000)"""




