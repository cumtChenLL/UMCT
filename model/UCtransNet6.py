import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ctrans2 import Encoder as CEncoder
from model.ctrans3 import Encoder as SEncoder
'''
from ctrans2 import Encoder as CEncoder
from ctrans3 import Encoder as SEncoder
'''

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
    def __init__(self, scale, in_channels, out_channels):
        super(Downblock, self).__init__()
        self.scale = scale
        self.resblock = 3

        self.dw_infer = InferDw(in_channels, self.resblock)
        self.dw_layer = nn.Conv2d(in_channels, out_channels, kernel_size=4,stride=2,
                padding=1,bias=False)
    
    def forward(self, x):
        x_infer = self.dw_infer(x)
        x_dw = self.dw_layer(x_infer)
        return x_dw
    
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.conv_du(y)
        return x * y1


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res1 = res + x
        return res1


class Upsampler(nn.Module):
    def __init__(self, scale, in_channels, out_channels):
        super(Upsampler, self).__init__()
        self.scale = scale
        n_infer = 8
        infer_list = []
        for i in range(n_infer):
            infer_list.append(RCAB(conv=default_conv, n_feat=in_channels, kernel_size=3, reduction=16))
        self.infer = nn.Sequential(*infer_list)
        # o = (i-1)*s + k - 2p + out_padding
        self.up_layer = nn.ConvTranspose2d(in_channels = in_channels, out_channels=out_channels, kernel_size = 3, stride = 2, padding =1, output_padding=1, bias=False)

    def forward(self, x):
        x_infer = self.infer(x)
        x_up = self.up_layer(x_infer)
        return x_up

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化 -> 压缩通道 -> 恢复通道 -> sigmoid 权重
        weight = self.avg_pool(x)
        weight = self.conv1(weight)
        weight = self.relu(weight)
        weight = self.conv2(weight)
        weight = self.sigmoid(weight)
        return x * weight


class FeatureFusionModule(nn.Module):
    def __init__(self, channels):
        """
        图像超分辨中的特征融合模块
        
        参数:
            channels (int): 输入/输出的通道数（C）
        """
        super(FeatureFusionModule, self).__init__()
        
        # 注意力部分使用原始通道数（因为是 concat 后再 attention）
        self.channel_attention = ChannelAttention(in_channels=2 * channels)

        # 1x1 卷积用于将通道数从 2*C 压缩回 C
        self.compress_conv = nn.Conv2d(2 * channels, channels, kernel_size=1)

        # 后处理模块
        self.post_process = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1)
        )

        # 最后的激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, combined):
        """
        输入:
            x1: [B, C, H, W]
            x2: [B, C, H, W]

        输出:
            fused_feat: [B, C, H, W]
        """
        # combined = torch.cat([x1, x2], dim=1)  # [B, 2C, H, W]
        combined = self.post_process(combined)
        atted = self.channel_attention(combined)
        combined = combined + atted
        out = self.compress_conv(combined)  # [B, C, H, W]
        out = self.relu(out)
        return out
    
class UCTransNet(nn.Module):
    def __init__(self, scale, n_colors, n_feat, s_patch):
        super(UCTransNet, self).__init__()
        self.scale = scale 
        n_colors = n_colors
        n_feat = n_feat
        self.s_patch = s_patch

        self.head = default_conv(in_channels= n_colors, out_channels=n_feat, kernel_size=3)
        self.down0 = default_conv(in_channels= n_feat, out_channels=n_feat, kernel_size=3)
        self.down1 = Downblock(self.scale, in_channels=n_feat, out_channels = 2 * n_feat)
        self.down2 = Downblock(self.scale, in_channels=2 * n_feat, out_channels = 4 * n_feat)
        # self.down3 = Downblock(self.scale, in_channels=4 * n_feat, out_channels = 8 * n_feat)
        # self.ctrans = ChannelTransformer
        #define CCT
        self.CAT = CEncoder(vis=True, expand_ratio=4, drop_rate=0.1, n_feat=n_feat, s_patch=s_patch)
        self.SAT = SEncoder(vis=True, att_drop_rate=0.1, expand_ratio=4, n_feat=n_feat, s_patch=s_patch)
        
        # self.up1 = Upsampler(self.scale, in_channels= 16*n_feat, out_channels=4*n_feat)
        self.up1 = Upsampler(self.scale, in_channels= 8*n_feat, out_channels=2*n_feat)
        self.up2 = Upsampler(self.scale, in_channels= 4*n_feat, out_channels= n_feat)
        self.up0 = default_conv(in_channels=2*n_feat, out_channels=n_feat, kernel_size=3)
        self.up0 = FeatureFusionModule(n_feat)
        # self.fuse = FeatureFusionModule(n_feat)
        self.tail = nn.Sequential(
            default_conv(in_channels= n_feat, out_channels=n_feat, kernel_size=3),
            nn.ReLU(inplace=True),
            default_conv(in_channels= n_feat, out_channels=n_colors, kernel_size=3)
        )

    def forward(self, x):
        # x = F.interpolate(x, scale_factor= self.scale, mode='bicubic')
        x_feat = self.head(x)
        x1_feat = self.down0(x_feat)
        x2_feat = self.down1(x1_feat)
        x3_feat = self.down2(x2_feat)
        # x_dw3 = self.down3(x_dw2)

        # print(x1_feat.shape, x2_feat.shape, x3_feat.shape)
        h1, w1 = x1_feat.shape[2:]
        p1_feat = F.interpolate(x1_feat, size=[self.s_patch, self.s_patch], mode='bicubic')
        p2_feat = F.interpolate(x2_feat, size=[self.s_patch//2, self.s_patch//2], mode='bicubic')
        p3_feat = F.interpolate(x3_feat, size=[self.s_patch//4, self.s_patch//4], mode='bicubic')
        # print(p1_feat.shape, p2_feat.shape, p3_feat.shape)
        
        x_cat1, x_cat2, x_cat3, xc_weight  = self.CAT(p1_feat, p2_feat, p3_feat)
        x_sat1, x_sat2, x_sat3, xs_weight  = self.SAT(p1_feat, p2_feat, p3_feat)
        
        # print(x_cat1.shape, x_cat2.shape, x_cat3.shape)
        x_cat1 = F.interpolate(x_cat1, size=[h1, w1], mode='bicubic')
        x_cat2 = F.interpolate(x_cat2, size=[h1//2, w1//2], mode='bicubic')
        x_cat3 = F.interpolate(x_cat3, size=[h1//4, w1//4], mode='bicubic')
        
        x_sat1 = F.interpolate(x_sat1, size=[h1, w1], mode='bicubic')
        x_sat2 = F.interpolate(x_sat2, size=[h1//2, w1//2], mode='bicubic')
        x_sat3 = F.interpolate(x_sat3, size=[h1//4, w1//4], mode='bicubic')
        
        x1_attn = x1_feat + x_cat1 + x_sat1
        x2_attn = x2_feat + x_cat2 + x_sat2
        x3_attn = x3_feat + x_cat3 + x_sat3

        x2_feat = self.up1(torch.cat([x3_attn, x3_feat], dim=1))
        x1_feat = self.up2(torch.cat([x2_attn, x2_feat], dim=1))
        x0_feat = self.up0(torch.cat([x1_feat, x1_attn], dim=1))
        x_out = self.tail(x0_feat)

        return x_out

if __name__=="__main__":
    feat0 = torch.randn(16, 3, 64, 64)   # 尺度0
    scale = 4 
    n_colors = 3
    n_feat = 32
    s_patch = 64
    net = UCTransNet(scale=scale, n_colors=n_colors, n_feat=n_feat, s_patch=s_patch)
    x = net(feat0)
    
