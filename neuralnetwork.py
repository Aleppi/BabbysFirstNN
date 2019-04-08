import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        layer_shapes = [layer_sizes[i+1:i-len(layer_sizes)-1:-1] for i in range(len(layer_sizes)-1)]
        self.weights = [np.random.standard_normal(i) for i in layer_shapes]
        self.biases = [np.zeros((i[0], 1)) for i in layer_shapes]
        
    @staticmethod
    def rectifier(x):
        return np.fmax(0, x)

    @staticmethod
    def softmax(x):
        return np.array([np.exp(i)/sum(np.exp(x)) for i in x])
    
    def predict(self, x):
        x.shape = (-1,1)
        for w, b in zip(self.weights, self.biases):
            x = self.rectifier(np.matmul(w,x)+b)
        return self.softmax(x)

    def cost(self, x, y):
        s = 0
        for i,j in zip(self.predict(x),y):
            s += (i - j)**2
        return s

    def costAvg(self, x, y):
        s = 0
        for i,j in zip(x, y):
            s += self.cost(i,j)
        return s/len(x)
