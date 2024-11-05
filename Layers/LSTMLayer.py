import numpy as np
from AutoGrad import Array
from Activations import Sigmoid,Tanh

class LSTMLayer:
    def __init__(self,input_dimension,unit_number,lstm_type = "many_to_many",
                 output_layer = True,output_activation = None,output_dimesion = None):
        self.w = {
            "forget_wx" :  Array(np.random.randn(input_dimension,unit_number) * np.sqrt(2/unit_number)),
            "forget_ws" :  Array(np.random.randn(unit_number,unit_number) * np.sqrt(2/unit_number)),
            "input_wx" :  Array(np.random.randn(input_dimension,unit_number) * np.sqrt(2/unit_number)),
            "input_ws" :  Array(np.random.randn(unit_number,unit_number) * np.sqrt(2/unit_number)),
            "candidate_wx" :  Array(np.random.randn(input_dimension,unit_number) * np.sqrt(2/unit_number)),
            "candidate_ws" :  Array(np.random.randn(unit_number,unit_number) * np.sqrt(2/unit_number)),
            "output_wx" :  Array(np.random.randn(input_dimension,unit_number) * np.sqrt(2/unit_number)),
            "output_ws" :  Array(np.random.randn(unit_number,unit_number) * np.sqrt(2/unit_number))
        }

        self.b = {
            "forget_b" : Array(np.zeros((1,unit_number))),
            "input_b" : Array(np.zeros((1,unit_number))),
            "candidate_b" : Array(np.zeros((1,unit_number))),
            "output_b" : Array(np.zeros((1,unit_number)))
        }

        if output_layer:
            self.w["whq"] = Array(np.random.randn(unit_number,output_dimesion) * np.sqrt(2/output_dimesion))
            self.b["bq"] = Array(np.zeros((1,output_dimesion)))

        self.unitn = unit_number
        self.lstm_type = lstm_type
        self.output_layer = output_layer
        self.oact = output_activation

    def LSTM_cell(self,x,short_term,long_term):
        forget_gate = x.matmul(self.w["forget_wx"]) + short_term.matmul(self.w["forget_ws"]) + self.b["forget_b"]
        forget_act = Sigmoid().forward(forget_gate)

        input_gate = x.matmul(self.w["input_wx"]) + short_term.matmul(self.w["input_ws"]) + self.b["input_b"]
        input_act = Sigmoid().forward(input_gate)

        candidate_gate = x.matmul(self.w["candidate_wx"]) + short_term.matmul(self.w["candidate_ws"]) + self.b["candidate_b"]
        candidate_act = Tanh().forward(candidate_gate)

        long_term_new = (long_term * forget_act) + (input_act * candidate_act)

        output_gate = x.matmul(self.w["output_wx"]) + short_term.matmul(self.w["output_ws"]) + self.b["output_b"]
        output_act = Sigmoid().forward(output_gate)

        short_term_new = output_act * Tanh().forward(long_term_new)

        return long_term_new ,short_term_new

    def forward(self,x):
        if type(x) != list:
            x = [x[:,i,:] for i in range(len(x[0]))]

        short_term = Array(np.zeros((len(x[0]),self.unitn)))
        long_term = Array(np.zeros((len(x[0]),self.unitn)))

        short_term_seq = []

        for input in x:
            long_term , short_term = self.LSTM_cell(input,short_term,long_term) 

            if self.output_layer:
                output = self.oact(short_term.matmul(self.w["whq"]) + self.b["bq"])
                short_term_seq.append(output)
            else:
                short_term_seq.append(short_term)

        if self.lstm_type == "many_to_one":
            return short_term_seq[-1]
        else:
            return short_term_seq
            
            




