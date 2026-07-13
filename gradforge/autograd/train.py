from mlp import MLP
from layer import Layer

xs = [
    [2, 3, -1],
    [3, -1, 0.5],
    [0.5, 1, 1],
    [1, 1, -1],
]

ys = [
    1.0,
    -1.0,
    -1.0,
    1.0,
]

model = MLP(3, [4,4,1])
learning_rate = 0.05

for epoch in range(15):
    preds = [model(x) for x in xs]
    # print([p.data for p in preds])
    
    losses = [
        (pred - y) ** 2
        for pred, y in zip(preds, ys)
    ]
    
    loss = sum(losses) / len(losses)
    
    for p in model.parameters():
        p.grad = 0.0
    
    loss.backward()
    # for p in model.parameters()[:5]:
    #     print(p.grad)
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    print(epoch, loss.data)