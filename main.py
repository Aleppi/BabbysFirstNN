import numpy as np
import mnist as mn
import neuralnetwork as nn

data = mn.train_images()/255
labels = [1 for i in range(len(mn.train_labels()))]

print(labels[0])

layer_sizes = (784,5,5,10)

NN = nn.NeuralNetwork(layer_sizes)
print(NN.predict(data[0]), '\n')
