import torch
from torch import nn

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss = nn.L1Loss(reduction='mean')
r = loss(input, target)
print(r)

loss_mes = nn.MSELoss()
r_mse = loss_mes(input, target)
print(r_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_ce = nn.CrossEntropyLoss()
r_ce = loss_ce(x, y)
print(r_ce)
