from value import Value
import random


class Neuron:
    
    def __init__(self, nin, nonlin=True):
        
        self.w = [Value(random.uniform(-1, 1))
                  for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
        
    def __call__(self, x):
        activation = self.b
        
        for wi, xi in zip(self.w, x):
            activation = activation + wi * xi
        
        if self.nonlin:
            activation = activation.relu
        
        return activation
    
    def parameters(self):
        return self.w + [self.b]