import numpy as np

example = np.random.randn(10,256)

def positional_encoding(sentence):
    positions = np.arange(len(sentence))
    dimensions = len(sentence[0])
    position_values = np.zeros((len(sentence),dimensions))
    for j in range(int(dimensions/2)):
        position_values[:,2*j] = np.sin(positions/(10000**((2*j)/dimensions)))
        position_values[:,2*j+1] = np.cos(positions/(10000**((2*j)/dimensions)))
    return position_values + sentence
