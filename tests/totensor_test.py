from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2


img_path = "../dataset/train/human_image/000000002007_jpg.rf.a19a160c6197a61e7e5ccdebc9767ae3.jpg"
img = cv2.imread(img_path)

writer = SummaryWriter("../logs")

transform = transforms.ToTensor()
img_tensor = transform(img)

writer.add_image("Tensor", img_tensor)

normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_tensor = normalize(img_tensor)
writer.add_image("Normalize", img_tensor, 1)

writer.close()
