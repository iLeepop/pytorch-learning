import torch
import torchvision.models

# save way 1 load way
model = torch.load("../models/vgg16_method1.pth")
print(model)

# save way 2 load way
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("../models/vgg16_method2.pth"))
print(vgg16)