import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset",
                                       train=False,
                                       download=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
_input = torch.reshape(input, (-1, 1, 5, 5))


class IleeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)
    def forward(self, x):
        x = self.maxpool1(x)
        return x


ilee = IleeModel()
output = ilee(_input)
print(output)
writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    img, target = data
    writer.add_images("input_maxpool", img, step)
    _output = ilee(img)
    writer.add_images("output_maxpool", _output, step)
    step += 1

writer.close()


