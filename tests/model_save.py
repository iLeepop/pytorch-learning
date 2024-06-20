import torch
import torchvision


vgg16 = torchvision.models.vgg16(weights='IMAGENET1K_V1')

# save way 1, model structure + model parameters
torch.save(vgg16, "../models/vgg16_method1.pth")

# save way 2, model parameters (official recommended)
torch.save(vgg16.state_dict(), "../models/vgg16_method2.pth")
