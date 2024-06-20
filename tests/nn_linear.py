import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=False)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Ilee(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, x):
        self.linear1(x)
        return x

ilee = Ilee()

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    img, target = data
    output = torch.reshape(img, (1, 1, 1, -1))
    _output = torch.flatten(img)
    __output = ilee(_output)
    print(__output)
    # writer.add_images("linear_output", __output, step)
    step += 1

writer.close()

