from AutoGrad import Array
from os import listdir,mkdir
import pickle

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
    
    def save(self,model_name):
        if "Models" not in listdir():
            mkdir("Models")
        if model_name not in listdir("Models"):
            mkdir(f"Models/{model_name}")
        
        with open(f"Models/{model_name}/{model_name}.pickle","wb") as f:
            pickle.dump(self.layers,f)

    def load(self,model_name):
        if model_name not in listdir("Models"):
            print("Model not found")
        else:
            with open(f"Models/{model_name}/{model_name}.pickle","rb") as f:
                layers = pickle.load(f)
                for layer in layers.values():
                    self.add_layer(layer)

        





 




    