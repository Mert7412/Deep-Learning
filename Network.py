from AutoGrad import Array

class Network:
    def __init__(self,sequential_data = False):
        self.params = {}
        self.layers = {}
        self.seq = sequential_data

    def add_layer(self,Layer):
        i = len(self.layers)
        self.layers[f"layer{i}"] = Layer
        if hasattr(Layer,"w"):
            for k in range(len(Layer.w.values())):
                self.params[f"layer{i}-w{k}"] = list(Layer.w.values())[k]
        if hasattr(Layer,"b"):
            for j in range(len(Layer.b.values())):
                self.params[f"layer{i}-b{j}"] = list(Layer.b.values())[j]

    def forward_propagation(self,input):
        y = input
        for i in self.layers.values():
            y = i.forward(y)
        return y
    
    def back_propagation(self,y,ypred,loss_function):
        
        if self.seq:
            loss = Array(0)
            for i in reversed(range(len(ypred))):
                loss += loss_function(y[i],ypred[i])
                loss.backward()
        else:
            loss = loss_function(y,ypred)
            loss.backward()

        return loss



 




    