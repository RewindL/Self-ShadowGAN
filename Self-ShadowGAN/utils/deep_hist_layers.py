import pdb
from torch import nn, sigmoid
import torch


def phi_k(x, L, W):
    return sigmoid((x + (L / 2)) / W) - sigmoid((x - (L / 2)) / W)


def compute_pj_with_mask(x, mu_k, K, L, W, mask):
    # we assume that x has only one channel already
    # flatten spatial dims
    if mask != None:
        x = torch.masked_select(x, mask)
    x = x.reshape(1, 1, -1)
    x = x.repeat(1, K, 1)  # construct K channels
    # apply activation functions
    return phi_k(x - mu_k, L, W)

class TripleDimHistLayer(nn.Module):
    def __init__(self, K, L, W, mu_k):
        super().__init__()
        self.K = K
        self.L = L
        self.W = W
        self.mu_k = mu_k

    def forward(self, x, mask):
        hist_R = SingleDimHistLayer(self.K, self.L, self.W, self.mu_k)(x[:, 0], mask)
        hist_G = SingleDimHistLayer(self.K, self.L, self.W, self.mu_k)(x[:, 1], mask)
        hist_B = SingleDimHistLayer(self.K, self.L, self.W, self.mu_k)(x[:, 2], mask)
        hist = torch.cat([hist_R,hist_G,hist_B], dim=1)
        return hist


class SingleDimHistLayer(nn.Module):
    def __init__(self, K, L, W, mu_k):
        super().__init__()
        self.K = K
        self.L = L
        self.W = W
        self.mu_k = mu_k

    def forward(self, x, mask):
        if mask != None :
            mask = mask.type(torch.BoolTensor).to(x.device)
            N = mask.sum().item()
            pj = compute_pj_with_mask(x, self.mu_k, self.K, self.L, self.W, mask)
        else:
            N = x.numel()
            pj = compute_pj_with_mask(x, self.mu_k, self.K, self.L, self.W, mask)
        return (pj.sum(dim=2) / N).unsqueeze(0)