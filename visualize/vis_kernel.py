import sys 
sys.path.append("..")

from kernel import Radial, Tanh
import cv2
import pdb
import numpy as np


# 2-dimention image
def magnify(image, scale):
    width, height = image.shape
    assert width == height
    result = np.zeros((width * scale, height * scale))
    for i in range(width):
        for j in range(height):
            if image[i][j] > 0:
                for l in range(scale):
                    for c in range(scale):
                        result[i * scale + l][j * scale + c] = image[i][j] 
    return result



# for r in [5, 7, 9, 11]:
#     size = r * 2 + 1
#     kernel = Radial(r=r, kernel_size=size)
#     kernel /= kernel.max()
#     kernel *= 255
#     kernel = kernel.astype(np.uint8)
#     kernel = magnify(kernel, 10)
#     kernel = np.stack([kernel, kernel, kernel], -1)


#     cv2.imwrite('radial-{}.png'.format(r), kernel)

#     kernel = Tanh(r=r, kernel_size=size)
#     kernel /= kernel.max()
#     kernel *= 255
#     kernel = kernel.astype(np.uint8)
#     kernel = magnify(kernel, 10)
#     kernel = np.stack([kernel, kernel, kernel], -1)

#     kernel = cv2.resize(kernel, (100, 100))
#     cv2.imwrite('Tanh-{}.png'.format(r), kernel)


a = np.zeros((200, 200))
a.fill(255)
a = a.astype(np.uint8)
cv2.imwrite('full.png', a)