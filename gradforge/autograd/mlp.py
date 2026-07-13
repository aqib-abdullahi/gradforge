from layer import Layer

class MLP:
    
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                nonlin=(i != len(nouts) - 1)
            )
            for i in range(len(nouts))
        ]
    
    def __call__(self, x):
        
        for layer in self.layers:
            x = layer(x)
        
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        
        return params