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

layer_sizes = (784,5,10)

training_data = list(zip(data[:10],labels[:10]))

mini_batch = list(zip(data[:10],labels[:10]))

NN = nn.NeuralNetwork(layer_sizes)
NN.SGD(training_data, 5, 10, 3)
print("Prediction: {} \n Answer: {}".format(NN.forward(data[0]), labels[0]))
#print(training_data)
#NN.update_mini_batch(mini_batch, 3)
