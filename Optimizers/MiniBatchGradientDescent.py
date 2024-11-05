import numpy as np

class MiniBatchGradientDescent:
    def __init__(self,model,loss_function):
        self.model = model
        self.params = list(model.params.values())
        self.lossf = loss_function


    def train(self,x,y,learningrate,iteration,batch_size=500):
        for i in range(iteration):
            indices = np.random.randint(0,len(x.data),size=batch_size)
            x_training = x[indices]
            y_training = y[indices]
            

            ypred = self.model.forward_propagation(x_training)
            
            loss = self.model.back_propagation(y_training,ypred,self.lossf)
           

            for param in self.params:
                param.data = param.data - (learningrate * param.gradient)

            print(f"iteration:{i} loss:{loss}")
