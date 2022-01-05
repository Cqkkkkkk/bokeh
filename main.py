from base_gui import MainWindow
from PyQt5.QtWidgets import QApplication
import sys
import cv2
import PIL.Image as pil

from simple_blur import SimpleBlur
from depth_blur import DepthBlur
from DMSHN.DNNBlur import DNNBlur
from utils import Ndarray2QPixmap
from modify_depth import ModifyDepthByMask
from U2Net.MaskGenerator import GenerateMask

import os


class MyWindow(MainWindow):
    def __init__(self):
        super().__init__()
        self.dnnblur = None

    def func_do_convert(self):
        if self.mode != 'dnn':
            self.statusbar.showMessage(
                'Now bluring with mode [{}] and kernel [{}] with size ({}, {})'.
                format(self.mode.capitalize(), self.kernel.capitalize(),
                       self.kernel_size, self.kernel_size))
        else:
            self.statusbar.showMessage('Now bluring with mode [{}] '.format(
                self.mode.upper()))

        if self.mode == 'simple':
            threshold = self.threshold
            blur = SimpleBlur(kernel_size=self.kernel_size,
                              kernel_type=self.kernel)
            image = cv2.imread(self.image_path)

            depth = cv2.imread(self.depth_path)

            if self.depthModify:
                if not os.path.exists(self.mask_path):
                    self.statusbar.showMessage('No depth mask detected in {}. Generating'.format(self.mask_path))
                    GenerateMask(self.image_path, './data/mask')
                mask = cv2.imread(self.mask_path)
                depth = ModifyDepthByMask().forward(depth, mask)

            depth = depth / depth.max() * 255
            depth = depth.astype(int)

            front_mask = depth <= threshold
            back_mask = depth > threshold
            blured = blur.forward(image,
                                  front_mask,
                                  back_mask,
                                  return_inter=False)

            self.pic_blured.setPixmap(Ndarray2QPixmap(blured))
            self.pic_depth.setPixmap(Ndarray2QPixmap(depth))
            self.result = blured

        elif self.mode == 'depth':
            blur = DepthBlur(kernel_type=self.kernel, a=self.a, df=self.df)
            image = cv2.imread(self.image_path)
            depth = cv2.imread(self.depth_path)

            if self.depthModify:
                print(self.mask_path)
                print(os.path.exists(self.mask_path))
                if not os.path.exists(self.mask_path):
                    self.statusbar.showMessage('No depth mask detected in {}. Generating'.format(self.mask_path))
                    GenerateMask(self.image_path, './data/mask')
                mask = cv2.imread(self.mask_path)
                depth = ModifyDepthByMask().forward(depth, mask)

            depth = depth / depth.max() * 255
            depth = depth.astype(int)

            blured = blur.forward(image, depth)

            self.pic_blured.setPixmap(Ndarray2QPixmap(blured))
            self.pic_depth.setPixmap(Ndarray2QPixmap(depth))
            self.result = blured

        elif self.mode == 'dnn':
            image = pil.open(self.image_path).convert('RGB')
            if self.dnnblur is None:
                self.statusbar.showMessage('Loading dnn model')
                self.dnnblur = DNNBlur()
                self.statusbar.showMessage('Load finished')
            blur = self.dnnblur
            blured = blur.forward(image)
            
            self.pic_blured.setPixmap(Ndarray2QPixmap(blured))
            self.result = blured

        if self.mode != 'dnn':
            self.statusbar.showMessage(
                'Blur finished with mode [{}] and kernel [{}] with size ({}, {})'.
                format(self.mode.capitalize(), self.kernel.capitalize(),
                       self.kernel_size, self.kernel_size))
        else:
            self.statusbar.showMessage('Blur finfished with mode [{}] '.format(
                self.mode.upper()))




if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MyWindow()
    sys.exit(app.exec_())
