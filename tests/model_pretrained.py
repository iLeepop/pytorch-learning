import torchvision
from torch import nn

vgg16_true = torchvision.models.vgg16(weights='IMAGENET1K_V1')
print(vgg16_true)
vgg16_false = torchvision.models.vgg16(weights=None)
print(vgg16_false)
train_data = torchvision.datasets.CIFAR10(root='../dataset',
                                          train=True,
                                          download=False,
                                          transform=torchvision.transforms.ToTensor())
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)

