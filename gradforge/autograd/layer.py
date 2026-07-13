from gradforge.autograd.neuron import Neuron

class Layer:
    
    def __init__(self, nin, nout, nonlin=True):
        
        self.neurons = [
            Neuron(nin, nout)
            for _ in range(nout)
        ]
        
    def __call__(self, x):
        outputs = [
            neuron(x)
            for neuron in self.neurons
        ]
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def parameters(self):
        
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        
        return params