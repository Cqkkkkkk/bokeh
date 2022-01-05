import sys
from os import path

sys.path.append("..")
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.abspath(__file__)))
sys.path.append(path.abspath(__file__))


import cv2
import PIL.Image as pil
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from stacked_DMSHN import stacked_DMSHN 


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

feed_width = 1536
feed_height = 1024

import time


class DNNBlur():
    def __init__(self) -> None:
        self.device  = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.feed_width = 1536
        self.feed_height = 1024
        self.model = stacked_DMSHN().to(device)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load('./DMSHN/checkpoints/SDMSHN/sdmshn.pth', map_location=device))
    
    def forward(self, image):
        original_width, original_height = image.size
        image = image.resize((feed_width, feed_height), pil.LANCZOS)
        image = transforms.ToTensor()(image).unsqueeze(0)
        image = image.to(device)

        bok_pred = self.model(image)
        bok_pred = F.interpolate(bok_pred,(original_height, original_width), mode='bilinear', align_corners=True)
        save_image(bok_pred, 'tmp.png')
        blured = cv2.imread('tmp.png')
        return blured


if __name__ == '__main__':
    image = pil.open('/Users/chenqin/PythonProjects/GUI/data/input/1002.jpg').convert('RGB')
    blur = DNNBlur()
    blured = blur.forward(image)
    cv2.imwrite('tmp.png', blured)