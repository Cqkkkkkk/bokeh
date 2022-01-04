import sys
from os import path

sys.path.append("..")
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.abspath(__file__)))
sys.path.append(path.abspath(__file__))


from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import os
import argparse
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./LeRes/ckpts/res101.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnet101', help='Checkpoint path to load')
    parser.add_argument('--input_dir', default='./data/input')
    parser.add_argument('--output_dir', default='./data/depth')

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


if __name__ == '__main__':  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_args()

    # create depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # load checkpoint
    load_ckpt(args, depth_model, None, None)
    depth_model.to(device)

    imgs_list = os.listdir(args.input_dir)
    imgs_list.sort()
    imgs_path = [os.path.join(args.input_dir, i) for i in imgs_list if i != 'outputs']
    image_dir_out = args.output_dir
    os.makedirs(image_dir_out, exist_ok=True)

    for i, v in enumerate(imgs_path):
        print('processing (%04d)-th image... %s' % (i, v))
        rgb = cv2.imread(v)
        rgb_c = rgb[:, :, ::-1].copy()
        gt_depth = None
        A_resize = cv2.resize(rgb_c, (448, 448))
        rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

        img_torch = scale_torch(A_resize)[None, :, :, :]
        pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        # if GT depth is available, uncomment the following part to recover the metric depth
        #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

        img_name = v.split('/')[-1]
        loc = img_name.rfind('.')

        # cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
        # save depth
        cv2.imwrite(os.path.join(image_dir_out, img_name[:loc]+'-depth.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
