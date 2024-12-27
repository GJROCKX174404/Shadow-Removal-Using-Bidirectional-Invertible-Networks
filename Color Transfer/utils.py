import os
import torch
import torch.nn as nn
import numpy as np
import math
import cv2

from torchvision.utils import make_grid

from collections import OrderedDict

def psnr_np(enhanced, image_dslr):
    # target = np.array(image_dslr)
    # enhanced = np.array(enhanced)
    # enhanced = np.clip(enhanced, 0, 1)
    #
    #
    # squared_error = np.square(enhanced - target)
    # mse = np.mean(squared_error)
    # psnr = 10 * np.log10(1.0 / mse)
    squares = (enhanced-image_dslr).pow(2)
    squares = squares.view([squares.shape[0],-1])
    psnr = torch.mean((-10/np.log(10))*torch.log(torch.mean(squares, dim=1)))

    return psnr

def load_model(load_path, model, strict=True):
    if isinstance(model, nn.DataParallel): 
        model = model.module
    if os.path.exists(load_path):
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        model.load_state_dict(load_net_clean, strict=False)
        print("Succcefully!!!!! pretrained model has loaded!!!!!!!!!!!!!!!")
    else:
        print("Wrong!!!!! pretrained path not exists")

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu()
        # .clamp_(*min_max)  # clamp
    # tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        # img_np = (img_np * 255.0).round()
        img_np = (img_np * 150.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)
