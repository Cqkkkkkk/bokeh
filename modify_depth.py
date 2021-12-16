import numpy as np
import cv2
import pdb


class ModifyDepthByMask():
    def __init__(self) -> None:
        pass

    def forward(self, depth, mask):
        mask = mask / mask.max()
        depth_modified = depth * (1 - mask)
        return depth_modified



if __name__ == '__main__':
    image = cv2.imread('./data/input/1002.jpg')
    depth = cv2.imread('./data/depth/1002-depth.png')
    mask = cv2.imread('./data/mask/1002-mask.png')

    depth_modify = ModifyDepthByMask()
    depth_modified = depth_modify.forward(depth, mask)

    cv2.imwrite('depth-modified.png', depth_modified)