from torch.utils.tensorboard import SummaryWriter
import cv2


writer = SummaryWriter("../logs")
img_path = "../dataset/train/human_image/000000006253_jpg.rf.af87068527f0b2d2afa22e0d9d527635.jpg"
img = cv2.imread(img_path)
writer.add_image("img", img, 2, dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()
