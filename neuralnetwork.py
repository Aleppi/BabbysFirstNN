import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        layer_shapes = [layer_sizes[i+1:i-len(layer_sizes)-1:-1] for i in range(len(layer_sizes)-1)]
        self.layer_sizes = layer_sizes
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

    def partialBias(self, x, y, l, j):
        return self.layerErr(x, y, l)[j]

    def partialWeight(self, x, y, l, j, k):
        return self.rectifier(self.weightedInput(x, l-1)[k])*self.layerErr(x, y, l)[j]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for l,i in enumerate(self.layer_sizes[1:]):
            for j in range(i):
                nabla_b[l][j] = self.partialBias(x, y, l+1, j)
                for k in range(self.weights[l].shape[1]):
                    nabla_w[l][j,k] = self.partialWeight(x, y, l+1, j, k)
        return (nabla_b, nabla_w)
    
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta):
        for i in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,len(training_data),mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

