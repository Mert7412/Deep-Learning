import numpy as np
from AutoGrad import Array

class Pooling:
    @staticmethod    
    def data_prep(input,kernel_size):
        x = input.data
        batch_size,height,width,channels = x.shape
        batch_stride, height_stride, width_stride, channel_stride = x.strides
        new_shape = (batch_size,int(height/kernel_size),int(width/kernel_size),kernel_size,kernel_size,channels)
        new_strides = (batch_stride,height_stride*kernel_size,width_stride*kernel_size,height_stride,width_stride,channel_stride)
        new_input = np.lib.stride_tricks.as_strided(x,new_shape,new_strides)
        return new_input
    
    class MaxPooling:   
        def __init__(self,kernel_size):
            self.ks = kernel_size

        def forward(self,input):
            x = Pooling.data_prep(input.data,self.ks)
            z = np.amax(x,axis=(3,4))
            output = Array(z,(input,),"MaxPooling")
            z = z.reshape(*x.shape[:3],1,1,x.shape[-1])
            def _backward():
                der = x-z
                der[der==0] = 1
                der[der<0] = 0
                der = der * output.gradient.reshape(*x.shape[:3],1,1,x.shape[-1])
                input.gradient += der.reshape(input.data.shape)
            output._backward = _backward
            return output

            
    




