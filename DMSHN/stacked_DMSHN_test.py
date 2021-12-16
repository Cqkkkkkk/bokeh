
import pdb
import os

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


bokehnet = stacked_DMSHN().to(device)
bokehnet = nn.DataParallel(bokehnet)
bokehnet.load_state_dict(torch.load('checkpoints/SDMSHN/sdmshn.pth',map_location=device))


os.makedirs('outputs/stacked_DMSHN/',exist_ok= True)

image_dir = '/Users/chenqin/PythonProjects/GUI/data/input'

with torch.no_grad():
    for image_path in os.listdir(image_dir):
        # Load image and preprocess
        input_image = pil.open(os.path.join(image_dir, image_path)).convert('RGB')
        original_width, original_height = input_image.size

        org_image = input_image
        org_image = transforms.ToTensor()(org_image).unsqueeze(0)

        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        org_image = org_image.to(device)
        input_image = input_image.to(device)

        bok_pred = bokehnet(input_image)

        bok_pred = F.interpolate(bok_pred,(original_height,original_width),mode = 'bilinear')

        pdb.set_trace()
        save_image(bok_pred, './outputs/stacked_DMSHN/'+ image_path[: image_path.find('.')] +'.png')


        del bok_pred
        del input_image
        break
    



