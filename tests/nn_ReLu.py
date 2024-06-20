import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
output = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10("../dataset",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=False)
dataloader = DataLoader(dataset, batch_size=64)

class Ilee(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        return x

ilee = Ilee()
output = ilee(input)
print(input)
print(output)

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    img, target = data
    writer.add_images("input_nonline", img, step)
    _output = ilee(img)
    writer.add_images("output_nonline", _output, step)
    step += 1

writer.close()
