import torch
import torchvision.transforms

from nn_loss_network import Ilee
import cv2 as cv


img = cv.imread("../asset/airplane.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32))
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
img = transform(img).cuda()
img = torch.reshape(img, (1, 3, 32, 32))

model = torch.load("../models/ilee_39.pth")
with torch.no_grad():
    output = model(img)
print(output.argmax(1))




