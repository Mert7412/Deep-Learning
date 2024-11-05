import numpy as np

class Array:
    def __init__(self,data,childs = (),operation = None):
        self.data = data
        self.shape = data.shape
        self.childs = set(childs)
        self.gradient = np.zeros_like(self.data,dtype=float)
        self.op= operation
        self._backward = lambda:None
     
    
    def __repr__(self):
        return np.array_str(self.data)
    
    def __getitem__(self,index):
        return Array(self.data[index])
    
    def __len__(self):
        return len(self.data)
    
    def __setitem__(self,index,value):
        self.gradient[index] = value.gradient
        self.data[index] = value.data

    def __add__(self,other):
        output = Array(self.data+other.data,(self,other),"+")
        
        def _backward():
            self.update_gradient(other,output,output.gradient,output.gradient)

        output._backward = _backward    
        return output
    
    def __mul__(self,other):
        output = Array(self.data*other.data,(self,other),"*")

        def _backward():
            self_der = other.data * output.gradient
            oth_der = self.data * output.gradient
            self.update_gradient(other,output,self_der,oth_der)
        output._backward = _backward
        return output
    def __truediv__(self,other):
        output = Array(self.data/other.data,(self,other),"div")

        def _backward():
            self_der = (1/other.data) * output.gradient
            oth_der = (-self.data/(other.data**2)) * output.gradient
            self.update_gradient(other,output,self_der,oth_der)
        output._backward = _backward
        return output
    
    def __sub__(self,other):
        output = Array(self.data - other.data,(self,other),"sub")

        def _backward():
            self_der = output.gradient
            oth_der = -output.gradient
            self.update_gradient(other,output,self_der,oth_der)
        output._backward = _backward
        return output
    
    def matmul(self,other):
        output = Array(np.matmul(self.data,other.data),(self,other),"matmul")
  
        def _backward():
            self_der = np.matmul(output.gradient,np.moveaxis(other.data,-2,-1))
            oth_der = np.matmul(np.moveaxis(self.data,-2,-1),output.gradient)
            self.update_gradient(other,output,self_der,oth_der)
        output._backward = _backward
        return output
    
    def mean(self,axis = 0):
        output = Array(np.mean(self.data,axis=axis,keepdims=True),(self,),"mean")

        def _backward():
            self.gradient += (1/np.array(self.data.shape)[np.array(axis)].prod()) * output.gradient
        output._backward = _backward
        return output
    
    def var(self,axis = 0):
        output = Array(np.var(self.data,axis=axis,keepdims=True),(self,),"var")

        def _backward():
            der = (2/np.array(self.data.shape)[np.array(axis)].prod())*(self.data - np.mean(self.data,axis=axis,keepdims=True))
            self.gradient += der * output.gradient
        output._backward = _backward
        return output
    
    def sqrt(self):
        z = np.sqrt(self.data)
        output = Array(z,(self,),"sqrt")

        def _backward():
            self.gradient += (1/(2*z)) * output.gradient
        output._backward = _backward
        return output
    
    def transpose(self):
        z = np.moveaxis(self.data,-2,-1)
        output = Array(z,(self,),"transpose")

        def _backward():
            self.gradient += np.moveaxis(output.gradient,-2,-1)
        output._backward = _backward
        return output
    
    def concatanate(self,other):
        z = np.concatenate([self.data,other.data],-1)
        output = Array(z,(self,other),"concatenate")

        def _backward():
            self.gradient += output.gradient[...,:self.data.shape[-1]]
            other.gradient += output.gradient[...,-other.data.shape[-1]:]
        output._backward = _backward
        return output
    
    def __iadd__(self,other):
        self.__add__(other)

    def update_gradient(self,other,output,self_der,oth_der):
        if self.gradient.shape !=self_der.shape:
            axes = tuple(i for i in range(len(self_der.shape)) if i < len(self.gradient.shape) if self_der.shape[i] != self.gradient.shape[i])
            
            self.gradient = self.gradient +  np.sum(self_der,axis=axes,keepdims=True)
        else:
            self.gradient =  self.gradient + self_der

        
        if other.gradient.shape != oth_der.shape:
            axes = tuple(i for i in range(len(oth_der.shape)) if i < len(other.gradient.shape) if oth_der.shape[i] != other.gradient.shape[i])
            other.gradient = other.gradient + np.sum(oth_der,axis=axes,keepdims=True)  
        else:
            other.gradient = other.gradient + oth_der

    def backward(self):
        oplist = []
        visited = set()
        def op_list(k):
            if k not in visited:
                visited.add(k)
                for i in k.childs:
                    op_list(i)
               
                oplist.append(k)
                      
        op_list(self)
        self.gradient = np.ones_like(self.data,dtype=float)
        for node in reversed(oplist):
            node._backward()


            
    
