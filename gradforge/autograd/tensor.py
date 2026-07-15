

class Tensor:
    def __init__(self, data):
        self.data = data
        self.shape = self._infer_shape(data)
        
    def _infer_shape(self, data):
        """Recursively infer the shape of a nested list

        Args:
            data (list): multidimensional array

        Returns:
            Lenght of list.
        """
        if not isinstance(data, list):
            return ()
        
        return (len(data),) + self._infer_shape(data[0])

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        
        return total

    def __getitem__(self, index):
        """Tensor indexing.

        Args:
            index (int): key of item or index
        
        Returns:
            Tensor instead of a python list
        """
        return Tensor(self.data[index])
    
    def dot(self, other):
        """Vector dot product
        """
        if len(self.shape) != 1:
            raise ValueError("First operand must be a vector.")
        if len(other.shape) != 1:
            raise ValueError("Second operand must be a vector.")
        if self.shape != other.shape:
            raise ValueError("Vectors must have the same length.")
        
        output = 0
        for a, b in zip(self.data, other.data):
            output += a * b
        
        return Tensor(output)
    
    def matmul(self, other):
        """Matrix X Vector multiplication"""
        
        if len(self.shape) != 2:
            raise ValueError("First Operand must be a matrix")
        if len(other.shape) != 1:
            raise ValueError("Second operand must be a vector")
        if self.shape[1] != other.shape[0]:
            raise ValueError("Matrix columns must equal vector length")
        
        outputs = []
        
        for row in self:
            outputs.append(row.dot(other).data)
        
        return Tensor(outputs)
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"