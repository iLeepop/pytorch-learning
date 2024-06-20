import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# 使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = torchvision.datasets.CIFAR10(root='../dataset',
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=False)
test_data = torchvision.datasets.CIFAR10(root='../dataset',
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

class Ilee(nn.Module):
    def __init__(self):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


ilee = Ilee()
ilee.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 1e-2
optimizer = torch.optim.SGD(ilee.parameters(), lr=learning_rate)

total_test_step = 0
total_train_step = 0
epoch = 10

writer = SummaryWriter("../logs_train")

ilee.train()
start_time = time.time()
for _i in range(epoch):
    print(f"-----第 {_i+1} 轮训练开始-----")
    for data in train_data_loader:
        img, target = data
        img = img.to(device)
        target = target.to(device)
        output = ilee(img)
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}，loss：{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)
            end_time = time.time()
            print(f"训练时间：{end_time - start_time}")

    # test
    ilee.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            img, target = data
            img = img.to(device)
            target = target.to(device)
            output = ilee(img)
            loss = loss_fn(output, target)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy += accuracy.item()

    print(f"整体测试集上的loss：{total_test_loss}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(ilee, f"../models/ilee_{_i}.pth")
    print("模型已保存")


writer.close()
