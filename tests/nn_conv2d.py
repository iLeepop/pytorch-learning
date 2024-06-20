import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
data_loader = DataLoader(dataset, batch_size=64)

class IleeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

ilee = IleeModel()
writer = SummaryWriter("../logs")
step = 0
for data in data_loader:
    img, target = data
    output = ilee(img)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("input", img, step)
    writer.add_images("output", output, step)
    step += 1

