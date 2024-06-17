from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2


img_path = "../dataset/train/human_image/000000002007_jpg.rf.a19a160c6197a61e7e5ccdebc9767ae3.jpg"
img = cv2.imread(img_path)

writer = SummaryWriter("../logs")

transform = transforms.ToTensor()
img_tensor = transform(img)

writer.add_image("Tensor", img_tensor)

writer.close()
