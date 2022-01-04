import sys
from os import path

sys.path.append("..")
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.abspath(__file__)))
sys.path.append(path.abspath(__file__))


import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms 

from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET 

import argparse


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]
    
    imo.save(os.path.join(d_dir, imidx + '-mask.png'))


def main(args):

    # --------- 1. get image path and name ---------
    image_dir = args.input_dir
    prediction_dir = args.output_dir
    
    model_dir = './U2Net/ckpts/u2net.pth'

    img_name_list = glob.glob(image_dir + os.sep + '*')

    # --------- 2. dataloader ---------
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320),
                                      ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------

    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("Inferencing:", img_name_list[i_test].split(os.sep)[-1])
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--input_dir', default='./inputs')
    argument_parser.add_argument('--output_dir', default='./outputs')
    args = argument_parser.parse_args()
    main(args)
