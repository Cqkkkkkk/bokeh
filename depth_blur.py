import pdb
import cv2
import argparse
import numpy as np
from kernel import Radial, Tanh, Mean
from modify_depth import ModifyDepthByMask

class DepthBlur():
    def __init__(self, kernel_type='radial', d_min=0, d_max=300, a=0.05, df=50, scale_factor=2) -> None:
        self.kernel_type = kernel_type
        if kernel_type == 'radial':
            self.kernel = Radial
        elif kernel_type == 'tanh':
            self.kernel = Tanh
   
        self.d_min = d_min
        self.d_max = d_max
        self.a = a
        self.df = df
        self.scale_factor = scale_factor
        

    def forward(self, image, depth):
        I_s = np.zeros_like(image)
        M_s = np.zeros_like(image)
        for d in np.arange(self.d_min, self.d_max, step=1 / self.a):
            M_d = (np.abs(depth - d) < 1 / self.a).astype(float)
            I_d = M_d * image
            r = np.abs(self.scale_factor * self.a * (d - self.df)).astype(int)
            kernel_size = (r + 1) * 2 + 1
            kernel = self.kernel(r=r, kernel_size=kernel_size)
            M_d_b = cv2.filter2D(M_d, -1, kernel=kernel)
            I_d_b = cv2.filter2D(I_d, -1, kernel=kernel)
            M_s = M_s * (1 - M_d_b) + M_d_b
            I_s = I_s * (1 - M_d_b) + I_d_b
            # M_s = M_s * (1 - M_d_b) + M_d_b * M_d
            # I_s = I_s * (1 - M_d_b) + I_d_b * M_d 
        I_b = I_s / M_s
        return I_b



if __name__ == '__main__':
    # d_min=0, d_max=300, a=0.05, df=50, scale_factor=2
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--ktype', type=str, default='tanh', choices=['radial', 'tanh'])
    argument_parser.add_argument('--dmin', type=int, default=0)
    argument_parser.add_argument('--dmax', type=int, default=300)
    argument_parser.add_argument('--a', type=float, default=0.05)
    argument_parser.add_argument('--df', type=int, default=50)
    argument_parser.add_argument('--scale', type=int, default=2)
    argument_parser.add_argument('--mdepth', action='store_true')
    args = argument_parser.parse_args()
    # read image
    image = cv2.imread('./data/input/1002.jpg')
    depth = cv2.imread('./data/depth/1002-depth.png')

    if args.mdepth:
        depth_modifier = ModifyDepthByMask()
        mask = cv2.imread('./data/mask/1002-mask.png')
        modified_depth = depth_modifier.forward(depth, mask)
        depth = modified_depth

    depth = depth / depth.max() * 255
    depth = depth.astype(int)

    depth_blur = DepthBlur(kernel_type=args.ktype, d_min=args.dmin, d_max=args.dmax, a=args.a, df=args.df, scale_factor=args.scale)
    blured = depth_blur.forward(image, depth)

    cv2.imwrite('./output/depth-spc/blured.png', blured)
