from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
import torchvision.transforms as transforms
import cv2
import os
import argparse
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./ckpts/res101.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnet101', help='Checkpoint path to load')
    parser.add_argument('--input_dir', default='./input')
    parser.add_argument('--output_dir', default='./output')

    args = parser.parse_args()
    return args

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


def TestDepth(input_image, output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_args()
    
    # create depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # load checkpoint
    load_ckpt(args, depth_model, None, None)
    depth_model.to(device)
    
    rgb = cv2.imread(input_image)
    rgb_c = rgb[:, :, ::-1].copy()
    
    A_resize = cv2.resize(rgb_c, (448, 448))

    img_torch = scale_torch(A_resize)[None, :, :, :]
    
    pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

    img_name = input_image.split('/')[-1]
    loc = img_name.rfind('.')
    img_name = img_name[: loc]
    out_path = os.path.join(output_dir, img_name + '-depth.png')

    # save depth
    cv2.imwrite(out_path, (pred_depth_ori / pred_depth_ori.max() * 60000).astype(np.uint16))
    


if __name__ == '__main__':
    
    TestDepth('./input/1002.jpg', './output')
