

import torch
from PIL import Image
from torchvision import transforms


img = '/home/xiaolonw/zqdata/workspace/cycle/logs/data_2/episode-0/img_0_1.jpg'
out_img = '/home/xiaolonw/zqdata/workspace/cycle/logs/data_2/img_0_1.jpg'

img = Image.open(img)
img = transforms.ToTensor()(img)

img = img[:,128:220,36:220]
img = transforms.ToPILImage()(img)
img.save(out_img)
