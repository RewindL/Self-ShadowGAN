from matplotlib.pyplot import hist
import torch
from torch import nn
import numpy as np
from .layers import bn, VarianceLayer, CovarianceLayer, GrayscaleLayer
from .downsampler import * 
from torch.nn import functional
from utils.common_utils import *

class MaskAlphaLoss(nn.Module):
    def __init__(self):
        super(MaskAlphaLoss,self).__init__()
    
    def forward(self, x, y, mask, alpha):
        _,chn,_,_ = x.shape
        diff = (torch.abs(x - y))**alpha
        diff_selected = torch.masked_select(diff, mask.type(torch.BoolTensor).to(diff.device))

        return diff_selected.sum()/(chn*mask.sum())
        

class OverRangeLoss(nn.Module):
    def __init__(self):
        super(OverRangeLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x, down, up, mask=None):
        assert up >= down
        if(mask == None):
            loss = ((torch.abs(x - down) + torch.abs(x - up) - (up-down))**1).mean()
        if(mask != None):
            loss = torch.masked_select(((torch.abs(x - down) + torch.abs(x - up) - (up-down))**1), mask.type(torch.BoolTensor).to(x.device)).mean()
        return loss

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, img, target, mask=None):
        cs =torch.cosine_similarity(img, target)
        if(mask == None):
            loss = (1-cs).mean()
        else:
            loss = torch.masked_select((1-cs), mask.type(torch.BoolTensor).to(img.device)).mean()
        return loss


class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))


     
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,x):
        #x = x.sum(dim=1, keepdim=True)
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:]) 
        count_w = self._tensor_size(x[:,:,:,1:]) 
        h_tv = torch.pow(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]) , 1).sum() 
        w_tv = torch.pow(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]) , 1).sum()
        return 2 * (h_tv/count_h+w_tv/count_w) / batch_size

    def _tensor_size(self,t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class MaskTVLoss(nn.Module):
    def __init__(self):
        super(MaskTVLoss,self).__init__()
        
    def forward(self, x, mask):
        #x = x.sum(dim=1, keepdim=True)
        batch_size, chn, h_x, w_x = x.shape
        mask_del_up = mask[:,:,1:,:].type(torch.BoolTensor).to(mask.device)
        mask_del_down = mask[:,:,:-1,:].type(torch.BoolTensor).to(mask.device)
        mask_del_left = mask[:,:,:,1:].type(torch.BoolTensor).to(mask.device)
        mask_del_right = mask[:,:,:,:-1].type(torch.BoolTensor).to(mask.device)
        count_h = chn*mask_del_up.sum()
        count_w = chn*mask_del_left.sum()
        h_diff = torch.pow(torch.abs(x[:,:,1:,:]-x[:,:,:-1,:]), 1)
        w_diff = torch.pow(torch.abs(x[:,:,:,1:]-x[:,:,:,:-1]), 1)
        h_tv = torch.masked_select(h_diff, mask_del_up).sum() + torch.masked_select(h_diff, mask_del_down).sum()
        w_tv = torch.masked_select(w_diff, mask_del_left).sum() + torch.masked_select(w_diff, mask_del_right).sum()
        return (h_tv/count_h + w_tv/count_w) / batch_size

class MaskStdLoss(nn.Module):
    def __init__(self):
        super(MaskStdLoss,self).__init__()
        
    def forward(self, img, mask):
        mask = mask.type(torch.BoolTensor).to(mask.device)
        batch_size, chn, M, N = img.shape
        loss = 0
        for ch in torch.split(img, 1, dim=1):
            elements = torch.masked_select(ch, mask)
            loss += torch.std(elements)/chn
        return loss



