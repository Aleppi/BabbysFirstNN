import numpy as np
import mnist as mn
import neuralnetwork as nn

def labelArrCreator(x):
    arr = np.zeros((10,1))
    arr[x] = 1
    return arr


data = mn.train_images()/255
data.shape = (len(mn.train_images()),-1,1)
labels = [labelArrCreator(i) for i in mn.train_labels()]

layer_sizes = (784,10,16,10)

NN = nn.NeuralNetwork(layer_sizes)
print(NN.layerErr(data[0], labels[0], 2).shape)
