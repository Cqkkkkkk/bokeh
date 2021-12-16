import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def Ndarray2QPixmap(image):
    image = np.require(image, np.uint8, 'C')
    image = QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3,
                   QImage.Format_RGB888).rgbSwapped()
    pix = QPixmap(image)
    return QPixmap(pix)