import numpy as np

class GradientDescent:
    def __init__(self,model,loss_function):
        self.model = model
        self.params = list(model.params.values())
        self.lossf = loss_function

    def train(self,x,y,learningrate,iteration):
        for i in range(iteration):
            ypred = self.model.forward_propagation(x)
           
            loss = self.model.back_propagation(y,ypred,self.lossf)
            
            for param in self.params:
                param.data = param.data - (learningrate * param.gradient)

            print(f"iteration:{i} loss:{loss}")