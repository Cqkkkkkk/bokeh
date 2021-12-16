import pdb
import cv2
import argparse
import numpy as np
from kernel import Mean, Radial, Tanh
from modify_depth import ModifyDepthByMask

class SimpleBlur():
    def __init__(self, kernel_size=21, kernel_type='tanh') -> None:
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type
        if kernel_type == 'mean':
            self.kernel = Mean(kernel_size=kernel_size)
        elif kernel_type == 'radial':
            r = kernel_size // 2 - 1
            self.kernel = Radial(r=r, kernel_size=kernel_size)
        elif kernel_type == 'tanh':
            r = kernel_size // 2 - 1
            self.kernel = Tanh(r=r, kernel_size=kernel_size)
        elif kernel_type == 'gauss':
            self.kernel = cv2.GaussianBlur


    def forward(self, image, front_mask, back_mask, return_inter=False):
        front_image = image * front_mask
        # Original: blur only backgounds, then mask
        # back_image = image * back_mask
        # blured_image = self.kernel(back_image, (self.kernel_size, self.kernel_size), -1)
        # blured_image_back = blured_image * back_mask

        # Improved: blur using whole image, then mask
        if isinstance(self.kernel, np.ndarray):
            blured_image = cv2.filter2D(image, -1, self.kernel)
        else:
            blured_image = self.kernel(image, (self.kernel_size, self.kernel_size), -1)
        blured_image_back = blured_image * back_mask

        full_image = front_image + blured_image_back.astype(int)
        if return_inter:
            return full_image, front_image, blured_image_back
        else:
            return full_image



if __name__ == '__main__':
    # read image
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--ksize', type=int, default=15)
    argument_parser.add_argument('--threshold', type=int, default=30)
    argument_parser.add_argument('--ktype', type=str, default='tanh', choices=['mean', 'gauss', 'radius', 'tanh'])
    argument_parser.add_argument('--mdepth', action='store_true')

    args = argument_parser.parse_args()

    image = cv2.imread('./data/input/1002.jpg')

    depth = cv2.imread('./data/depth/1002-depth.png')

    if args.mdepth:
        depth_modifier = ModifyDepthByMask()
        mask = cv2.imread('./data/mask/1002-mask.png')
        modified_depth = depth_modifier.forward(depth, mask)
        cv2.imwrite('./data/depth_modify/1002-depth-modify.png', modified_depth)
        depth = modified_depth


    depth = depth / depth.max() * 255
    depth = depth.astype(int)

    front_mask = depth <= args.threshold
    back_mask = depth > args.threshold

        
    blur = SimpleBlur(kernel_size=args.ksize, kernel_type=args.ktype)
    full, front, blurd_back = blur.forward(image, front_mask, back_mask, return_inter=True)

    cv2.imwrite('./output/simple/blured.png', full)
    cv2.imwrite('./output/simple/blured_back.png', blurd_back)
    cv2.imwrite('./output/simple/front.png', front)