import numpy as np

class Adam:
    def __init__(self,model,loss_function):
        self.params = list(model.params.values())
        self.model = model
        self.lossf = loss_function
        
        self.mt = [np.zeros_like(i.gradient) for i in self.params]
        self.vt = [np.zeros_like(i.gradient) for i in self.params]
        
        
    def train(self,x,y,learning_rate = 0.001,iteration = 1000,batch_size=16,beta1 = 0.9,beta2 = 0.999,epsilon = 1e-8):
        for j in range(iteration):
            indices = np.random.randint(0,len(x.data),size=batch_size)
            x_training = x[indices]
            y_training = y[indices]
            

            ypred = self.model.forward_propagation(x_training)
            
            loss = self.model.back_propagation(y_training,ypred,self.lossf)

            for i in range(len(self.params)):
                self.mt[i] = beta1*self.mt[i] + (1 - beta1) * self.params[i].gradient
                mt_hat = self.mt[i] / ((1 - beta1**(j+1)))
                
                self.vt[i] = beta2 * self.vt[i] + (1 - beta2) * (self.params[i].gradient ** 2)
                vt_hat = self.vt[i] / ((1 - beta2**(j+1)))    
                

                self.params[i].data = self.params[i].data - mt_hat * (learning_rate/(np.sqrt(vt_hat) + epsilon))
        
            print(f"iteration:{j} loss:{loss}")
            
