import numpy as np
import cv2

def Radial(r=5, kernel_size=11):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            if r ** 2 - (i - center) ** 2 - (j - center) ** 2 >= 0:
                kernel[i][j] = 1
    return kernel / kernel.sum()            


def Tanh(r=5, kernel_size=11):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = 0.5 + 0.5 * np.tanh(0.25 * (r ** 2 - (i - center) ** 2 - (j - center) ** 2) + 0.5)
    return kernel / kernel.sum()      


def Mean(kernel_size=11):
    kernel = np.ones((kernel_size, kernel_size))
    return kernel / kernel.sum()  



if __name__ == '__main__':
    pass
    