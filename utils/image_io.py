import glob

from matplotlib import image

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import skimage.io
from skimage import img_as_ubyte
import pdb
import random
import torch.nn as nn
from torch.nn import functional as F

matplotlib.use('agg')


def crop_image(img, d=32):
    """
    Make dimensions divisible by d

    :param pil img:
    :param d:

    :return:
    """

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def crop_np_image(img_np, d=32):
    return torch_to_np(crop_torch_image(np_to_torch(img_np), d))


def crop_torch_image(img, d=32):
    """
    Make dimensions divisible by d
    image is [1, 3, W, H] or [3, W, H]
    :param pil img:
    :param d:
    :return:
    """
    new_size = (img.shape[-2] - img.shape[-2] % d,
                img.shape[-1] - img.shape[-1] % d)
    pad = ((img.shape[-2] - new_size[-2]) // 2, (img.shape[-1] - new_size[-1]) // 2)

    if len(img.shape) == 4:
        return img[:, :, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]
    assert len(img.shape) == 3
    return img[:, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]


def get_params(opt_over, net, net_input, downsampler=None):
    """
    Returns parameters that we want to optimize over.
    :param opt_over: comma separated list, e.g. "net,input" or "net"
    :param net: network
    :param net_input: torch.Tensor that stores input `z`
    :param downsampler:
    :return:
    """

    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    """
    Creates a grid from a list of images by concatenating them.
    :param images_np:
    :param nrow:
    :return:
    """
    images_torch = [torch.from_numpy(x).type(torch.FloatTensor) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(name, images_np, interpolation='lanczos', output_path="output/"):
    """
    Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW or 1xHxW
        nrow: how many images will be in one row
        interpolation: interpolation used in plt.imshow
    """
    assert len(images_np) == 2 
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, 2)

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.axis('off')
    plt.savefig(output_path + "{}.png".format(name), bbox_inches='tight',pad_inches=0.0)
    plt.close()


def save_image(name, image_np, output_path="output"):
    if(torch.is_tensor(image_np)):
        image_np = torch_to_np(image_np)
        print("convert torch.Tensor to np.")
    skimage.io.imsave("{}/{}.png".format(output_path, name), img_as_ubyte(image_np.transpose(1, 2, 0)))

def save_list(name, list, step=1, output_path="output"):
    x = np.arange(step, step*(1 + len(list)), step)
    plt.plot(x, list)
    #plt.axis('off')
    plt.savefig(output_path + "/{}_list.png".format(name), bbox_inches='tight', pad_inches=0.0)
    plt.close()


def save_histogram(name, hist_input, output_path="output"):
    hist_input = hist_input.cpu()
    assert hist_input.shape[1] == 3 or hist_input.shape[1] == 1
    for i in range(hist_input.shape[0]):
        hist = hist_input[i:i+1,:,:]
        if hist.shape[1] == 1:
            plt.plot(hist[0][0], '-k')
        else:
            # plt.plot(hist[0][0], '#A72A2A', hist[0][1], '#377453', hist[0][2], '#2B5CB4', linewidth=3, alpha=0.6)
            plt.plot(hist[0][0], '-r', hist[0][1], '-g', hist[0][2], '-b')
        plt.yticks([])
        if(hist_input.shape[0] == 1):    
            plt.savefig(output_path + "/{}.png".format(name), bbox_inches='tight',pad_inches=0.2)
        else:
            plt.savefig(output_path + "/{}_{}.png".format(name, i), bbox_inches='tight',pad_inches=0.2)
        plt.close()

def save_histogram_bar(name, hist_input, output_path="output"):
    hist_input = hist_input.cpu()
    assert hist_input.shape[1] == 3 or hist_input.shape[1] == 1
    for i in range(hist_input.shape[0]):
        hist = hist_input[i:i+1,:,:]
        if hist.shape[1] == 1:
            plt.plot(hist[0][0], '-k')
        else:
            plt.plot(hist[0][0], '-r', hist[0][1], '-g', hist[0][2], '-b')
        
        plt.yticks([])
        if(hist_input.shape[0] == 1):    
            plt.savefig(output_path + "/{}.png".format(name), bbox_inches='tight',pad_inches=0.2)
        else:
            plt.savefig(output_path + "/{}_{}.png".format(name, i), bbox_inches='tight',pad_inches=0.2)
        plt.close()

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np

def prepare_image(file_name):
    """
    loads makes it divisible
    :param file_name:
    :return: the numpy representation of the image
    """

    img_pil = get_image(file_name, -1)[0]
    return pil_to_np(img_pil)



def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def median(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for c in range(shape[0]):
        for w in range(shape[1]):
            for h in range(shape[2]):
                result[c, w, h] = sorted(i[c, w, h] for i in img_np_list)[l//2]
    return result


def average(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for i in img_np_list:
        result += i
    return result / l


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_pil_yuv(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar, mode="YCbCr")

def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]
def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]

