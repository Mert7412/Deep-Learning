import numpy as np
from AutoGrad import Array
        

class ConvolutionalLayer:
    def __init__(self,input_channels,output_channels,kernel_size,stride =1):
        self.w = {"w" : Array(np.random.randn(output_channels,kernel_size,kernel_size,input_channels))}
        self.b = {"b" : Array(np.zeros((1,1,1,output_channels)))}
        self.stride =stride
        
    def convolve(self,input,kernel,stride = 1):
        x = input.data
        batch_size,height,width,channels = x.shape
        batch_stride, height_stride, width_stride, channel_stride = x.strides
        output_channels , kernel_height,kernel_width,input_channels = kernel.data.shape
        output_width = int((width - kernel_width)/stride) + 1
        output_height = int((height - kernel_height)/stride) + 1
        
        new_shape = (batch_size,output_height,output_width,kernel_height,kernel_width,channels) 
        new_stride = (batch_stride,height_stride*stride,width_stride*stride,height_stride,width_stride,channel_stride) 
        new_input = np.lib.stride_tricks.as_strided(x,new_shape,new_stride)
        z = np.einsum("bhwftc,kftc->bhwk",new_input,kernel.data)
        output = Array(z,(input,kernel),"convolution")
        def _backward():
            grad_shape = (batch_size,output_height+(output_height-1)*(stride-1),output_width+(output_width-1)*(stride-1),output_channels)
            extend_grad = np.zeros(grad_shape)
            extend_grad[:,::stride,::stride,:] = output.gradient
            padded = np.pad(extend_grad, ((0, 0), (kernel_height - 1, kernel_height - 1),
                                                         (kernel_width - 1, kernel_width - 1), (0, 0)))
            pad_batch_strides,pad_height_strides,pad_width_strides,pad_channel_strides = padded.strides

            new_shape_grad = (batch_size, height, width, kernel_height, kernel_width, output_channels)
            new_stride_grad = (pad_batch_strides, pad_height_strides,
                                pad_width_strides, pad_height_strides,
                                  pad_width_strides, pad_channel_strides)
            new_output_grad = np.lib.stride_tricks.as_strided(padded, new_shape_grad, new_stride_grad)
            input.gradient += np.einsum("bhwftc,cftk->bhwk",new_output_grad,np.flip(kernel.data,axis=(1,2)))
            kernel.gradient += np.einsum("bwhk,bwhftc->kftc",output.gradient,new_input)
        output._backward = _backward
        return output
    
    def forward(self,input):
        z = self.convolve(input,self.w["w"],self.stride) + self.b["b"]
        return z
    


    
