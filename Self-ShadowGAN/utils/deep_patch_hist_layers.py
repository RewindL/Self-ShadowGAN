import pdb
from re import X
from torch import nn, sigmoid
import torch


def phi_k(x, L, W):
    return sigmoid((x + (L / 2)) / W) - sigmoid((x - (L / 2)) / W)

def compute_pj(x, mu_k, K, L, W):
    # we assume that x has only one channel already
    # flatten spatial dims
    chn = x.shape[0] * x.shape[1]
    x = x.reshape(chn, 1, -1)
    x = x.repeat(1, K, 1)  # construct K channels
    # apply activation functions
    return phi_k(x - mu_k, L, W) # chn * K * (H*W)

def compute_pj_map(x, mu_k, K, L, W):
    batch_num, x_chn, x_H, x_W = x.shape
    pj = compute_pj(x, mu_k, K, L, W)
    pj_map = pj.reshape(batch_num, x_chn, K, x_H, x_W)
    return pj_map

class MultiImages_HistLayer(nn.Module):
    def __init__(self, K, L, W, mu_k):
        super().__init__()
        self.K = K
        self.L = L
        self.W = W
        self.mu_k = mu_k

    def forward(self, x):
        # x is a Tensor with shape Batch_num*3*H*W
        batch_num = x.shape[0]
        chn = x.shape[1]
        N = x.shape[2] * x.shape[3]
        # Kernel Histogram
        pj = compute_pj(x, self.mu_k, self.K, self.L, self.W)
        hist = pj.sum(dim=2) / N
        return hist.reshape(batch_num, chn, -1)