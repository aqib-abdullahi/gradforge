
class Value:
    
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        
        self.parents = ()
        
        self.op = None
        self._backward = lambda: None
        
    def _add__(self, other):
        
        if not isinstance(other, Value):
            other = Value(other)
        
        new_data = self.data + other.data
        output = Value(new_data)
        
        output.parents = (self, other)
        output.op = "+"
        
        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        
        output._backward = _backward
        
        return output

    def __mul__(self, other):
        
        if not isinstance(other, Value):
            other = Value(other)
        
        new_data = self.data * other.data
        output = Value(new_data)
        
        output.parents = (self, other)
        output.op = "*"
        
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        
        output._backward = _backward
        
        return output

    def backward(self):
        
        topo = []
        visited = set()
        
        def build(node):
            if node not in visited:
                
                visited.add(node)
                for parent in node.parents:
                    build(parent)
                
                topo.append(node)
        
        build(self)
        
        #seed gradient
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()
                