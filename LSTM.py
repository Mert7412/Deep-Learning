import pandas as pd
import numpy as np 
import idx2numpy 
from FNN import *
"""data = idx2numpy.convert_from_file(r"deep-learning,\data\data3\train-images.idx3-ubyte")
y = idx2numpy.convert_from_file(r"deep-learning,\data\data3\train-labels.idx1-ubyte")
y1 = np.zeros((y.shape[0],int(y.max()-y.min()+1)))
y1[np.arange(y.shape[0]),np.array(y,dtype=int)] = 1
x = (data - np.mean(data,axis=0) )/ (np.std(data,axis=0)+1e-8)"""

def gradientclipping(gradient,threshold):
    norm = np.linalg.norm(gradient)
    if norm >= threshold:
        gradient = threshold * (gradient/norm)
    return gradient

class Lstm:
    def __init__(self,inputshape,hs,rnntype,embedding = False,isdecoder=False,embeddingshape = 100):
        if embedding == True:
            self.w_embedding  = np.random.rand(inputshape[1],embeddingshape)
            inputshape[1] = embeddingshape
        self.wx_forget_gate , self.wh_forget_gate , self.b_forget_gate = np.random.randn(inputshape[1],hs) *1e-5,np.random.randn(hs,hs)*1e-5 ,np.random.randn(hs)*1e-5 
        self.wx_input_gate , self.wh_input_gate , self.b_input_gate = np.random.randn(inputshape[1],hs)*1e-5 ,np.random.randn(hs,hs)*1e-5 ,np.random.randn(hs)*1e-5
        self.wx_input_node , self.wh_input_node , self.b_input_node = np.random.randn(inputshape[1],hs)*1e-5 ,np.random.randn(hs,hs)*1e-5 ,np.random.randn(hs)*1e-5
        self.wx_output_gate , self.wh_output_gate , self.b_output_gate = np.random.randn(inputshape[1],hs) *1e-5,np.random.randn(hs,hs)*1e-5 ,np.random.randn(hs)*1e-5
        self.longtermemory = np.zeros((inputshape[0],hs,inputshape[2]+1))
        self.shortermemory = np.zeros((inputshape[0],hs,inputshape[2]+1))
        self.hs = hs
        self.xs = inputshape
        self.rnntype = rnntype
        self.isdecoder = isdecoder
        self.embedding = embedding

    def forward(self,x):
        self.x = x 
        self.z = x if self.embedding == False else np.zeros(self.xs)
        self.output_gatez,self.input_gatez,self.input_nodez,self.forgetgatez = np.zeros((self.xs[0],self.hs,self.xs[2])),np.zeros((self.xs[0],self.hs,self.xs[2])),np.zeros((self.xs[0],self.hs,self.xs[2])),np.zeros((self.xs[0],self.hs,self.xs[2]))
        self.input_gatel = np.zeros((self.xs[0],self.hs,self.xs[2]))
        self.input_nodel = np.zeros((self.xs[0],self.hs,self.xs[2]))
        self.output_gatel = np.zeros((self.xs[0],self.hs,self.xs[2]))
        self.forgetgatel = np.zeros((self.xs[0],self.hs,self.xs[2]))  
        for i in range(self.xs[2]):
            if self.embedding == True:
                self.z[:,:,i] = np.dot(self.x[:,:,i],self.w_embedding)

            
            self.forgetgatez[:,:,i] = np.dot(self.z[:,:,i],self.wx_forget_gate) + np.dot(self.shortermemory[:,:,i-1],self.wh_forget_gate) + self.b_forget_gate
            self.forgetgatel[:,:,i] = activations(self.forgetgatez[:,:,i],"sigmoid")
            
            self.input_gatez[:,:,i] = np.dot(self.z[:,:,i],self.wx_input_gate) + np.dot(self.shortermemory[:,:,i-1],self.wh_input_gate) + self.b_input_gate
            self.input_gatel[:,:,i] = activations(self.input_gatez[:,:,i],"sigmoid")
            self.input_nodez[:,:,i] = np.dot(self.z[:,:,i],self.wx_input_node) + np.dot(self.shortermemory[:,:,i-1],self.wh_input_node) + self.b_input_node
            self.input_nodel[:,:,i] = activations(self.input_nodez[:,:,i],"tanh")
            self.longtermemory[:,:,i] = (self.longtermemory[:,:,i-1] *  self.forgetgatel[:,:,i]) + (self.input_gatel[:,:,i]*self.input_nodel[:,:,i])
            self.output_gatez[:,:,i] = np.dot(self.z[:,:,i],self.wx_output_gate) + np.dot(self.shortermemory[:,:,i-1],self.wh_output_gate) + self.b_output_gate
            self.output_gatel[:,:,i] = activations(self.output_gatez[:,:,i],"sigmoid")
            self.shortermemory[:,:,i] = self.output_gatel[:,:,i] * activations(self.longtermemory[:,:,i],"tanh")

        if self.rnntype == "many_to_one":
            return  self.shortermemory[:,:,-2] 
        else :

             return self.shortermemory[:,:,:-1]

    def backward(self,loss,learningrate,dl_dc = None,dl_dh = None):
        dl_wx_long , dl_wh_long , dl_b_long = np.zeros_like(self.wx_forget_gate),np.zeros_like(self.wh_forget_gate),np.zeros_like(self.b_forget_gate)
        dl_wx_input_gate , dl_wh_input_gate , dl_b_input_gate = np.zeros_like(self.wx_input_gate),np.zeros_like(self.wh_input_gate),np.zeros_like(self.b_input_gate)
        dl_wx_input_node , dl_wh_input_node , dl_b_input_node = np.zeros_like(self.wx_input_node),np.zeros_like(self.wh_input_node),np.zeros_like(self.b_input_node)
        dl_wx_output_gate , dl_wh_output_gate , dl_b_output_gate = np.zeros_like(self.wx_output_gate),np.zeros_like(self.wh_output_gate),np.zeros_like(self.b_output_gate)
        if self.embedding == True:
            dl_w_embedding = np.zeros_like(self.w_embedding)
        
        
        dl_dh_all = loss[:,:,None] if len(loss.shape) == 2 else loss
        

        dl_dc = dl_dc if type(dl_dc) == np.ndarray  else np.zeros((self.xs[0],self.hs)) 
        dl_dh = dl_dh if type(dl_dh) == np.ndarray  else np.zeros((self.xs[0],self.hs)) 
        dl_dx = np.zeros_like(self.z)
     
        def sumgradients(i):
            nonlocal dl_wx_long , dl_wh_long , dl_b_long ,dl_wx_input_gate , dl_wh_input_gate , dl_b_input_gate
            nonlocal dl_wx_input_node , dl_wh_input_node , dl_b_input_node,dl_wx_output_gate , dl_wh_output_gate , dl_b_output_gate 

            dl_wx_long += np.dot(self.z[:,:,i].T,dl_dforgetgate)
            dl_wh_long += np.dot(self.shortermemory[:,:,i-1].T,dl_dforgetgate)
            dl_b_long += np.sum(dl_dforgetgate,axis=0)

            dl_wx_input_gate += np.dot(self.z[:,:,i].T,dl_dinputgate)
            dl_wh_input_gate += np.dot(self.shortermemory[:,:,i-1].T,dl_dinputgate)
            dl_b_input_gate += np.sum(dl_dinputgate,axis=0)

            dl_wx_input_node += np.dot(self.z[:,:,i].T,dl_dinputnode)
            dl_wh_input_node += np.dot(self.shortermemory[:,:,i-1].T,dl_dinputnode)
            dl_b_input_node += np.sum(dl_dinputnode,axis=0)

            dl_wx_output_gate += np.dot(self.z[:,:,i].T,dl_doutputgate)
            dl_wh_output_gate += np.dot(self.shortermemory[:,:,i-1].T,dl_doutputgate)
            dl_b_output_gate += np.sum(dl_doutputgate,axis=0)
        
        for i in reversed(range(self.xs[2])):
            if i>= (self.xs[2]-dl_dh_all.shape[-1]) :
                dl_dhnext =   dl_dh_all[:,:,i-self.xs[2]]
                
                dl_dc += dl_dhnext * self.output_gatel[:,:,i] *derivativeofactivations(self.longtermemory[:,:,i],"tanh")
                dl_dh +=  dl_dhnext 

            
            dl_dforgetgate = dl_dc * self.longtermemory[:,:,i-1] * derivativeofactivations(self.forgetgatel[:,:,i],"sigmoid")
            dl_dinputgate = dl_dc *self.input_nodel[:,:,i] * derivativeofactivations(self.input_gatez[:,:,i],"sigmoid")
            dl_dinputnode = dl_dc * self.input_gatel[:,:,i] * derivativeofactivations(self.input_nodel[:,:,i],"tanh")
            dl_doutputgate = dl_dh * activations(self.longtermemory[:,:,i],"tanh") * derivativeofactivations(self.output_gatel[:,:,i],"sigmoid") 

            sumgradients(i)


            dl_dc  *= self.forgetgatel[:,:,i]
            dl_dh = np.dot(dl_dforgetgate,self.wh_forget_gate.T) +np.dot(dl_dinputgate,self.wh_input_gate.T) 
            +  np.dot(dl_dinputnode,self.wh_input_node.T)+np.dot(dl_doutputgate,self.wh_output_gate.T)

            dl_dx[:,:,i] = (np.dot(dl_dforgetgate,self.wx_output_gate.T) + np.dot(dl_dinputgate,self.wx_input_gate.T) 
            + np.dot(dl_dinputnode,self.wx_input_node.T) + np.dot(dl_dforgetgate,self.wx_forget_gate.T))
            if self.embedding == True: 
                dl_w_embedding += np.dot(self.x[:,:,i].T,dl_dx[:,:,i])
        if self.embedding ==True:
            self.w_embedding -= learningrate*dl_w_embedding
        self.wx_forget_gate-= learningrate*dl_wx_long
        self.wh_forget_gate-= learningrate*dl_wh_long
        self.b_forget_gate-= learningrate*dl_b_long 
        self.wx_input_gate -= learningrate*dl_wx_input_gate
        self.wh_input_gate -= learningrate*dl_wh_input_gate
        self.b_input_gate -= learningrate*dl_b_input_gate
        self.wx_input_node -= learningrate*dl_wx_input_node
        self.wh_input_node -= learningrate*dl_wh_input_node
        self.b_input_node -= learningrate*dl_b_input_node
        self.wx_output_gate -= learningrate*dl_wx_output_gate
        self.wh_output_gate -= learningrate*dl_wh_output_gate
        self.b_output_gate -=  learningrate*dl_b_output_gate
        if self.isdecoder == True:
            return dl_dx , dl_dc,dl_dh
        else:
            return dl_dx

class lstmnetwork:
    def __init__(self):
        self.network =[Lstm([1000,28,28],100,"many_to_one",True)]
        fn = FNNnetwork(100,[30,10],"sigmoid","softmax")
        for i in fn.network:
            self.network.append(i)
    def forwardpropagation(self,x):
        self.y = x
        for n in self.network:
            self.y = n.forward(self.y)
            
        return self.y
    
    def backpropagation(self,x,y,learningrate,lossf="crossentropy"):
        ypred = self.forwardpropagation(x)
        loss = derlossfunction(y,ypred,lossf)
        
        for n in reversed(self.network):
            loss = n.backward(loss,learningrate)
    def gradientdescent(self,x,y,learningrate,iteration):
        for i in range(iteration):
            self.backpropagation(x,y,learningrate)

            print(lossfunction(y,self.y,"crossentropy"))

    def stochasticgradient(self,x,y,learningrate,iteration,lossf = "mse"):
        for m in range(iteration):

            i = np.random.randint(0,len(x),size=1)
            self.backpropagation(x[i],y[i],learningrate,lossf)
            yp = self.forwardpropagation(x[i])
            l = lossfunction(y[i],yp,lossf) 
            print(l)

"""x = x[:1000]
y = y1[:1000]
a = lstmnetwork()
a.gradientdescent(x,y,1e-8,3000)"""




"""    def backward1(self,loss,learningrate,isdecoder=False):
        dl_wx_long , dl_wh_long , dl_b_long = np.zeros_like(self.wx_forget_gate),np.zeros_like(self.wh_forget_gate),np.zeros_like(self.b_forget_gate)
        dl_wx_input_gate , dl_wh_input_gate , dl_b_input_gate = np.zeros_like(self.wx_input_gate),np.zeros_like(self.wh_input_gate),np.zeros_like(self.b_input_gate)
        dl_wx_input_node , dl_wh_input_node , dl_b_input_node = np.zeros_like(self.wx_input_node),np.zeros_like(self.wh_input_node),np.zeros_like(self.b_input_node)
        dl_wx_output_gate , dl_wh_output_gate , dl_b_output_gate = np.zeros_like(self.wx_output_gate),np.zeros_like(self.wh_output_gate),np.zeros_like(self.b_output_gate)

        def sumgradients(i):
            nonlocal dl_wx_long , dl_wh_long , dl_b_long ,dl_wx_input_gate , dl_wh_input_gate , dl_b_input_gate
            nonlocal dl_wx_input_node , dl_wh_input_node , dl_b_input_node,dl_wx_output_gate , dl_wh_output_gate , dl_b_output_gate 

            dl_wx_long += np.dot(self.x[:,:,i].T,dforget)
            dl_wh_long += np.dot(self.shortermemory[:,:,i-1].T,dforget)
            dl_b_long += np.sum(dforget,axis=0)

            dl_wx_input_gate += np.dot(self.x[:,:,i].T,dinput_gate)
            dl_wh_input_gate += np.dot(self.shortermemory[:,:,i-1].T,dinput_gate)
            dl_b_input_gate += np.sum(dinput_gate,axis=0)

            dl_wx_input_node += np.dot(self.x[:,:,i].T,dinput_node)
            dl_wh_input_node += np.dot(self.shortermemory[:,:,i-1].T,dinput_node)
            dl_b_input_node += np.sum(dinput_node,axis=0)

            dl_wx_output_gate += np.dot(self.x[:,:,i].T,doutput_gate)
            dl_wh_output_gate += np.dot(self.shortermemory[:,:,i-1].T,doutput_gate)
            dl_b_output_gate += np.sum(doutput_gate,axis=0)
    
        
        loss = loss[:,:,None] if len(loss.shape) == 2 else loss
        losh = loss.shape[-1]
        
        if isdecoder == True:
            lossencoder = 0
        hforget = np.zeros((self.xs[0],self.hs))
        hinput_gate = np.zeros((self.xs[0],self.hs))
        hinput_node = np.zeros((self.xs[0],self.hs))
        houtput_gate = np.zeros((self.xs[0],self.hs))
        dl_dx = np.zeros_like(self.x)
     
        for i in reversed(range(self.xs[2])):
            if i >= (self.xs[2]- losh):
                dl_dc = loss[:,:,i-self.xs[2]]*self.output_gatel[:,:,i]
                dc_dtanh= dl_dc*derivativeofactivations(self.longtermemory[:,:,i],"tanh") 
                deltaoutput_gate = loss[:,:,i-self.xs[2]]*activations(self.longtermemory[:,:,i],"tanh")

            else:
                dc_dtanh,deltaoutput_gate = 0,0,0,0

            dforget = (dc_dtanh +  hforget )* self.longtermemory[:,:,i-1] * derivativeofactivations(self.forgetgatez[:,:,i],"sigmoid")
            dinput_gate = (dc_dtanh +  hinput_gate ) * self.input_nodel[:,:,i] * derivativeofactivations(self.input_gatez[:,:,i],"sigmoid") 
            dinput_node = (dc_dtanh +  hinput_node ) * self.input_gatel[:,:,i] * derivativeofactivations(self.input_nodez[:,:,i],"tanh") 
            doutput_gate = (deltaoutput_gate +  houtput_gate ) * derivativeofactivations(self.output_gatez[:,:,i],"sigmoid") 

            sumgradients(i)
            
            hforget += np.dot(dforget,self.wh_forget_gate)
            hinput_gate += np.dot(dinput_gate,self.wh_input_gate)
            hinput_node += np.dot(dinput_node,self.wh_input_node)
            houtput_gate += np.dot(doutput_gate,self.wh_output_gate)
            dl_dx[:,:,i] = (np.dot(doutput_gate,self.wx_output_gate.T) + np.dot(dinput_gate,self.wx_input_gate.T) 
                            + np.dot(dinput_node,self.wx_input_node.T) + np.dot(dforget,self.wx_forget_gate.T))
    

        self.wx_forget_gate-= learningrate*dl_wx_long
        self.wh_forget_gate-= learningrate*dl_wh_long
        self.b_forget_gate-= learningrate*dl_b_long 
        self.wx_input_gate -= learningrate*dl_wx_input_gate
        self.wh_input_gate -= learningrate*dl_wh_input_gate
        self.b_input_gate -= learningrate*dl_b_input_gate
        self.wx_input_node -= learningrate*dl_wx_input_node
        self.wh_input_node -= learningrate*dl_wh_input_node
        self.b_input_node -= learningrate*dl_b_input_node
        self.wx_output_gate -= learningrate*dl_wx_output_gate
        self.wh_output_gate -= learningrate*dl_wh_output_gate
        self.b_output_gate -=  learningrate*dl_b_output_gate
        return dl_dx"""
