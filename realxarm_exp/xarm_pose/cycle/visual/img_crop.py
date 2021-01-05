
import os
from PIL import Image
from torchvision import transforms


img_path = '/home/xiaolonw/zqdata/workspace/cycle_nian/reallogs/explog4/train/img_batch_1806/img_0_0.945_0.994.jpg'

img = Image.open(img_path)
img = transforms.ToTensor()(img)

print(img_path)

