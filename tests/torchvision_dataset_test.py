import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# train_set = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

writer = SummaryWriter("../logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        img, target = data
        writer.add_images("Epoch:{}".format(epoch), img, step)
        step += 1

writer.close()


