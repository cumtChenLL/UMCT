##DLNet: degrade light and generate low-light images
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from model.unet import UNet as UNet
def make_model(model_name):
    if model_name == "UNet":
        return UNet(3, 3)
    elif model_name == "Downblock":
        return Downblock(scale=4, n_feats=32)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class InferDw(nn.Module):
    def __init__(self, n_feats, num_block):
        super(InferDw,self).__init__()

        resblock = ResBlock(conv=default_conv, n_feats=n_feats, kernel_size=3, bias=True, 
            bn=False, act=nn.ReLU(True), res_scale=1)
        block = []
        for i in range(num_block):
            block.append(resblock)
        self.layer = nn.Sequential(*block)
        
    def forward(self, x):
        return(self.layer(x))

class Downblock(nn.Module):
    def __init__(self, scale, n_feats):
        super(Downblock,self).__init__()
        self.n_feats = n_feats
        self.scale = scale
        #kernel_size = 3
        num_block = 3

        self.dwhead = default_conv(3, self.n_feats, kernel_size=3, bias=False)
        dwinfer = InferDw(self.n_feats, num_block)
        #dwconv = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=4,stride=2,
        #        padding=1,bias=False)
        dwconv = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3,stride=1,
                padding=1,bias=False)

        self.dwbody = nn.ModuleList([dwinfer, dwconv, dwinfer, dwconv, dwinfer, dwconv])
        
        self.dwtail = default_conv(self.n_feats, 3, kernel_size=3, bias=False)

    def forward(self, x):
        feat = self.dwhead(x)
        if self.scale >1:
            res = self.dwbody[0](feat)
            feat = self.dwbody[1](feat+res)           
        if self.scale > 3:
            res = self.dwbody[2](feat)
            feat = self.dwbody[3](feat+res)
        if self.scale > 7:
            res = self.dwbody[4](feat)
            feat = self.dwbody[5](feat+res)
        return self.dwtail(feat)
