from torchvision import transforms
import cv2


img_path = "../dataset/train/human_image/000000002007_jpg.rf.a19a160c6197a61e7e5ccdebc9767ae3.jpg"
img = cv2.imread(img_path)
transform = transforms.ToTensor()
img_tensor = transform(img)
print(img_tensor)
