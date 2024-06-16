import torch
import torch.nn as nn
import torchvision
import sys
import os


import numpy as np
from PIL import Image
import PIL
import numpy as np
from utils.image_io import *
import pdb

import matplotlib.pyplot as plt
import cv2

def sig(x):
    return 2*torch.sigmoid(x) - 1

def inv_sig(x):
    return torch.log((x+1)/(1-x))


def get_gray_image_1channel(img_torch):
    assert img_torch.shape[1] == 3
    R, G, B = torch.split(img_torch, 1, dim=1)
    gray_img1 = R * 0.299 + G * 0.587 + B * 0.114
    return gray_img1

def get_gray_image_3channel(img_torch):
    gray_img1 = get_gray_image_1channel(img_torch)
    gray_img3 = gray_img1.repeat(1,3,1,1)
    return gray_img3

def get_eroded_dilated_mask(mask_torch, w_in, w_out):
    mask3_torch = torch.cat([mask_torch,mask_torch,mask_torch], dim=1)
    mask_cv = cv2.cvtColor(np.asarray(np_to_pil(torch_to_np(mask3_torch))),cv2.COLOR_RGB2BGR)
    kernel = np.ones((3,3),np.uint8)
    dilated_mask_torch = None
    eroded_mask_torch = None
    if(w_in != None): eroded_mask_torch = np_to_torch(pil_to_np(cv2.cvtColor(np.asarray(cv2.erode(mask_cv, kernel, iterations = w_in)),cv2.COLOR_BGR2RGB)))[:,0:1,:,:].to(mask_torch.device)
    if(w_out != None): dilated_mask_torch = np_to_torch(pil_to_np(cv2.cvtColor(np.asarray(cv2.dilate(mask_cv, kernel, iterations = w_out)),cv2.COLOR_BGR2RGB)))[:,0:1,:,:].to(mask_torch.device)
    
    return eroded_mask_torch, dilated_mask_torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("{} DIR maked.".format(dir_name))
    else:
        print("{} DIR exists.".format(dir_name))