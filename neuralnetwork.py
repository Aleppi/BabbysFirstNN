import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        layer_shapes = [layer_sizes[i+1:i-len(layer_sizes)-1:-1] for i in range(len(layer_sizes)-1)]
        self.weights = [np.random.standard_normal(i) for i in layer_shapes]
        self.biases = [np.zeros((i[0], 1)) for i in layer_shapes]
        self.layer_count = len(layer_sizes)
        
    @staticmethod
    def rectifier(x):
        return np.fmax(0, x)

    @staticmethod
    def rectifierprime(x):
        return np.piecewise(x, [x <= 0, x > 0], [0, 1])

    @staticmethod
    def softmax(x):
        return np.array([np.exp(i)/sum(np.exp(x)) for i in x])
    
    def forward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = self.rectifier(np.matmul(w,x)+b)
        return self.softmax(x)

    def cost(self, x, y):
        return 0.5*np.linalg.norm(self.forward(x) - y)**2

    def costprime(self, x, y):
        return np.linalg.norm(self.forward(x) - y)
    
    def costAvg(self, x, y):
        s = 0
        for i,j in zip(x, y):
            s += self.cost(i,j)
        return s/len(x)

    def weightedInput(self, x, l):
        for i in range(l):
            x = self.rectifier(np.matmul(self.weights[i],x)+self.biases[i])
        return x

    def layerErr(self, x, y, l):
        if l == self.layer_count - 1:
            return self.costprime(x, y)*self.rectifierprime(self.weightedInput(x, self.layer_count - 1))
        return (np.matmul(self.weights[l].T,self.layerErr(x, y, l+1)))*self.rectifierprime(self.weightedInput(x, l))

