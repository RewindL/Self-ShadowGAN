import glob
from re import X

from matplotlib import image

import torch
import torchvision
import torchvision.transforms
from skimage import color
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils.image_io import *
import random
import torch.nn as nn
from torch.nn import functional as F


def get_kernel_patches_with_mask(image_torch, mask_torch, kernel_size, stride, least_inside_rate, least_outside_rate, using_bias=True, using_padding=True, keep_batches=False):
    mask_torch = mask_torch.clone().detach()
    assert len(image_torch.shape) == len(mask_torch.shape) and image_torch.shape[2] == mask_torch.shape[2] and image_torch.shape[3] == mask_torch.shape[3]
    assert least_inside_rate >= 0 and least_outside_rate >= 0 and least_inside_rate + least_outside_rate <= 1
    bias = random.randrange(stride) if using_bias else 0
    if(using_padding): 
        pad = nn.ReflectionPad2d(padding=(stride, stride, stride, stride))
        image_torch= pad(image_torch)[:,:,bias:, bias:]
        mask_torch = pad(mask_torch)[:,:,bias:, bias:]
    batch_size, chn, H, W = image_torch.shape
    patch_area = kernel_size[0] * kernel_size[1]

    img_unfold_res = F.unfold(image_torch, kernel_size=kernel_size, stride=stride)
    mask_unfold_res = F.unfold(mask_torch, kernel_size=kernel_size, stride=stride)

    if(not keep_batches):
        img_patches = img_unfold_res.permute(0,2,1).reshape(-1, chn, kernel_size[0], kernel_size[1])
        mask_patches = mask_unfold_res.permute(0,2,1).reshape(-1, 1, kernel_size[0], kernel_size[1])
        mask_patches_sum = mask_patches.sum(dim=(1,2,3))
        return img_patches[(mask_patches_sum/patch_area >= least_inside_rate) & (mask_patches_sum/patch_area <= 1-least_outside_rate)]
    else:
        pass

def get_kernel_patches_in(image_torch, mask_torch, kernel_size, stride):
    return get_kernel_patches_with_mask(image_torch, mask_torch, kernel_size, stride, least_inside_rate=0.5, least_outside_rate=0)

def get_kernel_patches_out(image_torch, mask_torch, kernel_size, stride):
    return get_kernel_patches_with_mask(image_torch, mask_torch, kernel_size, stride, least_inside_rate=0, least_outside_rate=1)

def get_kernel_patches_bd(image_torch, bd_mask_torch, kernel_size, stride):
    return get_kernel_patches_with_mask(image_torch, bd_mask_torch, kernel_size, stride, least_inside_rate=0.4, least_outside_rate=0)

def get_kernel_patches_in_for_hist(image_torch, mask_torch, kernel_size, stride):
    return get_kernel_patches_with_mask(image_torch, mask_torch, kernel_size, stride, least_inside_rate=0.1, least_outside_rate=0)

def get_kernel_patches_out_for_hist(image_torch, mask_torch, kernel_size, stride):
    return get_kernel_patches_with_mask(image_torch, mask_torch, kernel_size, stride, least_inside_rate=0, least_outside_rate=1)

def image_rotate(img_torch, angle):
    img_np = np.transpose(torch_to_np(img_torch), (1,2,0))
    img_np_resized = skimage.transform.rotate(img_np, angle, resize=True)
    return np_to_torch(np.transpose(img_np_resized, (2,0,1))).to(img_torch.device)
    
