##DTNet: Degrade image resolution
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from model.unet import UNet as UNet
def make_model(model_name):
    if model_name == "UNet":
        return UNet(3, 3)
    elif model_name == "Darkblock":
        return Darkblock(n_feats=32)

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

class Darkblock(nn.Module):
    def __init__(self, n_feats):
        super(Darkblock,self).__init__()
        self.n_feats = n_feats
        #kernel_size = 3
        num_block = 3

        n_head = [default_conv(3, self.n_feats, kernel_size=3, bias=False)]
        n_body = []
        n_body.append(InferDw(self.n_feats, num_block))
        n_body.append(nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3,stride=1,
                padding=1,bias=False)
            )
        #n_body.append(nn.ReLU(True))
        n_tail = [nn.Conv2d(self.n_feats, 3, kernel_size=3, stride=1, padding=1, bias=False)]
        
        #self.dwhead = nn.Sequential(*dwhead)
        self.n_head = nn.Sequential(*n_head)
        self.n_body = nn.Sequential(*n_body)
        self.n_tail = nn.Sequential(*n_tail)
        #self.dwtail = nn.Sequential(*dwtail)

    def forward(self, x):
        x1 = self.n_head(x)
        x2 = self.n_body(x1)
        out = self.n_tail(x2+x1)
        return out