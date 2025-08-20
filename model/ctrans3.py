import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from einops import rearrange, repeat

logger = logging.getLogger(__name__)

class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, patchsize, img_size, in_channels):
        super(Channel_Embeddings, self).__init__()
        # img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patchsize)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels,))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Embedingcat(nn.Module):
    def __init__(self, KV_size, in_channels, out_channel):
        super(Embedingcat, self).__init__()
        self.f0 = nn.Linear(in_channels[0], out_channel, bias=False)
        self.f1 = nn.Linear(in_channels[1], out_channel, bias=False)
        self.f2 = nn.Linear(in_channels[2], out_channel, bias=False)
        self.attn_norm =  LayerNorm(KV_size, eps=1e-6)
    
    def forward(self, emb1, emb2, emb3):
        emb1 = self.f0(emb1)
        emb2 = self.f1(emb2)
        emb3 = self.f2(emb3)
        emb_all = torch.cat([emb1, emb2, emb3], dim=1)
        emb_all = emb_all.transpose(-1, -2)
        emb_all = self.attn_norm(emb_all)
        return emb_all

class KV_caculation(nn.Module):
    def __init__(self, num_heads, KV_size, in_channels, out_channels):
        super(KV_caculation, self).__init__()
        self.KV_size = KV_size
        self.num_heads = num_heads
        self.emb_cat = Embedingcat(KV_size, in_channels, out_channels)
        
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()
        for _ in range(self.num_heads):
            key = nn.Linear( self.KV_size,  self.KV_size, bias=False)
            value = nn.Linear(self.KV_size,  self.KV_size, bias=False)
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        
    def forward(self, emb1, emb2, emb3):
        emb_all = self.emb_cat(emb1, emb2, emb3)
        multi_head_K_list = []
        multi_head_V_list = []
        for key in self.key:
            K = key(emb_all)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb_all)
            multi_head_V_list.append(V)
        
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)
        # print(["multi_head_K:", multi_head_K.shape])
        # print(["multi_head_V:", multi_head_V.shape])
        return multi_head_K, multi_head_V      

class Attention_org(nn.Module):
    def __init__(self, vis,channel_num, KV_size, numheads, att_drop_rate):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = KV_size
        self.num_attention_heads = numheads
        self.query1 = nn.ModuleList()

        for _ in range(numheads):
            query1 = nn.Linear(channel_num, channel_num, bias=False)
            self.query1.append(copy.deepcopy(query1))
            
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num, channel_num, bias=False)
        self.attn_dropout = Dropout(att_drop_rate)
        self.proj_dropout = Dropout(att_drop_rate)

    def forward(self, emb1, multi_head_K, multi_head_V):
        multi_head_Q1_list = []
        if emb1 is not None:
            for query1 in self.query1:
                Q1 = query1(emb1)
                multi_head_Q1_list.append(Q1)

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1)
        multi_head_Q1 = multi_head_Q1.transpose(-1, -2)
        # print(multi_head_Q1.shape)
        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K)
        # print(attention_scores1.shape) 
        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size) 
        attention_probs1 = self.softmax(self.psi(attention_scores1)) 
        # print(attention_probs4.size())

        if self.vis:
            weights = attention_probs1.mean(1) 
        else: weights=None

        attention_probs1 = self.attn_dropout(attention_probs1) 
        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V) 
        
        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() 
        context_layer1 = context_layer1.mean(dim=3) 
        # print("context_layer:",context_layer1.shape)

        O1 = self.out1(context_layer1) 
        O1 = self.proj_dropout(O1) 
        return O1, weights

class Mlp(nn.Module):
    def __init__(self, in_channel, mlp_channel, drop_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        #self.act_fn = nn.GELU()
        self.act_fn = nn.ReLU(inplace = True)
        self.dropout = Dropout(drop_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class SBlock_ViT(nn.Module):
    def __init__(self, vis, channel_num, KV_size, numheads, att_drop_rate, expand_ratio):
        super(SBlock_ViT, self).__init__()
        self.attn_norm1 = LayerNorm(channel_num,eps=1e-6)
        self.channel_attn = Attention_org(vis,channel_num, KV_size, numheads, att_drop_rate)
        self.ffn_norm1 = LayerNorm(channel_num,eps=1e-6)
        self.ffn1 = Mlp(channel_num,channel_num*expand_ratio, att_drop_rate)

    def forward(self, emb1, multi_head_K, multi_head_V):
        # print(emb1.shape)
        emb1 = emb1.transpose(-1, -2)
        org1 = emb1
        cx1 = self.attn_norm1(emb1)
        cx1, weights = self.channel_attn(cx1, multi_head_K, multi_head_V)
        cx1 = org1 + cx1
        org1 = cx1
        x1 = self.ffn_norm1(cx1)
        x1 = self.ffn1(x1)
        x1 = x1 + org1
        return x1, weights

class Encoder(nn.Module):
    def __init__(self, vis, att_drop_rate, expand_ratio, n_feat, s_patch):
        super(Encoder, self).__init__()
        # feat0:[B 256 64 64] feat1:[B 512 32 32] feat2:[B 1024 16 16]
        feat0_size = [n_feat, s_patch, s_patch]
        feat1_size = [n_feat*2, s_patch//2, s_patch//2]
        feat2_size = [n_feat*4, s_patch//4, s_patch//4]
        channelnum = [feat0_size[0], feat1_size[0], feat2_size[0]] 
        numheads = 4
        numlayers = 1
        patchsize = [4, 4, 4]
        patchnum = [feat0_size[1]//patchsize[0]*feat0_size[2]//patchsize[0], 
                    feat1_size[1]//patchsize[1]*feat1_size[2]//patchsize[1], 
                    feat2_size[1]//patchsize[2]*feat2_size[2]//patchsize[2]]
        KVS_size = sum(patchnum)
        self.feat1_embedding = Channel_Embeddings(patchsize[0], feat0_size[1:], feat0_size[0])
        self.feat2_embedding = Channel_Embeddings(patchsize[1], feat1_size[1:], feat1_size[0])
        self.feat3_embedding = Channel_Embeddings(patchsize[2], feat2_size[1:], feat2_size[0])
        
        self.kv1_cat = KV_caculation(num_heads=numheads, KV_size=KVS_size, in_channels=channelnum, out_channels=channelnum[0])
        self.kv2_cat = KV_caculation(num_heads=numheads, KV_size=KVS_size, in_channels=channelnum, out_channels=channelnum[1])
        self.kv3_cat = KV_caculation(num_heads=numheads, KV_size=KVS_size, in_channels=channelnum, out_channels=channelnum[2])
        
        self.vis = vis
        self.layer1 = nn.ModuleList()
        for _ in range(numlayers):
            layer = SBlock_ViT(vis, patchnum[0], KVS_size, numheads, att_drop_rate, expand_ratio)
            self.layer1.append(copy.deepcopy(layer))
            
        self.layer2 = nn.ModuleList()
        for _ in range(numlayers):
            layer = SBlock_ViT(vis, patchnum[1], KVS_size, numheads, att_drop_rate, expand_ratio)
            self.layer2.append(copy.deepcopy(layer))
            
        self.layer3 = nn.ModuleList()
        for _ in range(numlayers):
            layer = SBlock_ViT(vis, patchnum[2], KVS_size, numheads, att_drop_rate, expand_ratio)
            self.layer3.append(copy.deepcopy(layer))
        
        self.encoder_norm1 = LayerNorm(patchnum[0],eps=1e-6)
        self.encoder_norm2 = LayerNorm(patchnum[1],eps=1e-6)
        self.encoder_norm3 = LayerNorm(patchnum[2],eps=1e-6)
        
        self.reconstruct = Reconstruct([feat0_size, feat1_size, feat2_size], patchsize)
        

    def forward(self, feat1, feat2, feat3):
        emb1 = self.feat1_embedding(feat1)
        emb2 = self.feat2_embedding(feat2)
        emb3 = self.feat3_embedding(feat3)
        
        kv1 = self.kv1_cat(emb1, emb2, emb3)
        kv2 = self.kv2_cat(emb1, emb2, emb3)
        kv3 = self.kv3_cat(emb1, emb2, emb3)
        
        attn_weights = []
        for layer_block in self.layer1:
            emb1, weights = layer_block(emb1, kv1[0], kv1[1])
            if self.vis:
                attn_weights.append(weights)
            
        for layer_block in self.layer2:
            emb2, weights = layer_block(emb2, kv2[0], kv2[1])
            if self.vis:
                attn_weights.append(weights)
        
        for layer_block in self.layer3:
            emb3, weights = layer_block(emb3, kv3[0], kv3[1])
            if self.vis:
                attn_weights.append(weights)
        
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        
        feat1_attn, feat2_attn, feat3_attn = self.reconstruct(emb1, emb2, emb3)
        
        return feat1_attn, feat2_attn, feat3_attn, attn_weights
    
class Reconstruct(nn.Module):
    def __init__(self, feat_size, patchsize):
        super(Reconstruct, self).__init__()
        
        self.feat0_size = feat_size[0]
        self.feat1_size = feat_size[1]
        self.feat2_size = feat_size[2]
        self.patchsize = patchsize
        self.conv1 = nn.Conv2d(self.feat0_size[0], self.feat0_size[0], kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(self.feat0_size[0])
        self.conv2 = nn.Conv2d(self.feat1_size[0], self.feat1_size[0], kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(self.feat1_size[0])
        self.conv3 = nn.Conv2d(self.feat2_size[0], self.feat2_size[0], kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(self.feat2_size[0])       
        self.activation = nn.ReLU(inplace=True)
        
        
        self.conv10 = nn.Conv2d(self.feat0_size[0], self.feat0_size[0]*self.patchsize[0]*self.patchsize[0], kernel_size=3, padding=1)
        self.conv20 = nn.Conv2d(self.feat1_size[0], self.feat1_size[0]*self.patchsize[1]*self.patchsize[1], kernel_size=3, padding=1)
        self.conv30 = nn.Conv2d(self.feat2_size[0], self.feat2_size[0]*self.patchsize[2]*self.patchsize[2], kernel_size=3, padding=1)
        self.up1 = nn.PixelShuffle(self.patchsize[0]) 
        self.up2 = nn.PixelShuffle(self.patchsize[1])
        self.up3 = nn.PixelShuffle(self.patchsize[2])
        self.act = nn.GELU()
        self.conv11 = nn.Conv2d(self.feat0_size[0], self.feat0_size[0], kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(self.feat1_size[0], self.feat1_size[0], kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(self.feat2_size[0], self.feat2_size[0], kernel_size=3, padding=1)
        
    def forward(self, x1, x2, x3):
        # 恢复空间维度
        x1_feat = rearrange(x1, 'b c (h w) -> b c h w', h=self.feat0_size[1]//self.patchsize[0], w=self.feat0_size[2]//self.patchsize[0])
        x2_feat = rearrange(x2, 'b c (h w) -> b c h w', h=self.feat1_size[1]//self.patchsize[1], w=self.feat1_size[2]//self.patchsize[1])
        x3_feat = rearrange(x3, 'b c (h w) -> b c h w', h=self.feat2_size[1]//self.patchsize[2], w=self.feat2_size[2]//self.patchsize[2])
        x1_feat = self.conv1(x1_feat)
        x1_feat = self.norm1(x1_feat)
        x2_feat = self.conv2(x2_feat)
        x2_feat = self.norm2(x2_feat)
        x3_feat = self.conv3(x3_feat)
        x3_feat = self.norm3(x3_feat)
        x1_feat = self.activation(x1_feat)
        x2_feat = self.activation(x2_feat)
        x3_feat = self.activation(x3_feat)
        
        x1_feat = self.conv10(x1_feat)
        x2_feat = self.conv20(x2_feat)
        x3_feat = self.conv30(x3_feat)
        x1_feat = self.up1(x1_feat)
        x2_feat = self.up2(x2_feat)
        x3_feat = self.up3(x3_feat)
        x1_feat = self.act(x1_feat)
        x2_feat = self.act(x2_feat)
        x3_feat = self.act(x3_feat)
        x1_feat = self.conv11(x1_feat)
        x2_feat = self.conv22(x2_feat)
        x3_feat = self.conv33(x3_feat)
        
        return x1_feat, x2_feat, x3_feat


if __name__ == "__main__":

    # 输入：三个尺度的特征图
    feat0 = torch.randn(2, 256, 64, 64)   # 尺度0
    feat1 = torch.randn(2, 512, 32, 32)   # 尺度1
    feat2 = torch.randn(2, 1024, 16, 16)  # 尺度2
    '''
    patchsize = [4, 4, 4]
    feat0_patchsize = _pair(patchsize[0])
    feat1_patchsize = _pair(patchsize[1])
    feat2_patchsize = _pair(patchsize[2])
    
    print(feat0_patchsize)
    feat0_size = feat0.shape
    feat1_size = feat1.shape
    feat2_size = feat2.shape
    
    channelnum = [256, 512, 1024]
    feat_size = [64, 32, 16]
    L_size = [16*16, 8*8, 4*4]
    KVS_size = sum(L_size)
    
    feat0_embedding = Channel_Embeddings(feat0_patchsize, feat0_size[2:], feat0_size[1])
    feat1_embedding = Channel_Embeddings(feat1_patchsize, feat1_size[2:], feat1_size[1])
    feat2_embedding = Channel_Embeddings(feat2_patchsize, feat2_size[2:], feat2_size[1])
    outputs0 = feat0_embedding(feat0)
    outputs1 = feat1_embedding(feat1)
    outputs2 = feat2_embedding(feat2)
    print(outputs0.shape, outputs1.shape, outputs2.shape)
    kv0_cat = KV_caculation(num_heads=4, KV_size=KVS_size, in_channels=channelnum, out_channels=channelnum[0])
    kv1_cat = KV_caculation(num_heads=4, KV_size=KVS_size, in_channels=channelnum, out_channels=channelnum[1])
    kv2_cat = KV_caculation(num_heads=4, KV_size=KVS_size, in_channels=channelnum, out_channels=channelnum[2])
    kv0 = kv0_cat(outputs0, outputs1, outputs2)
    kv1 = kv1_cat(outputs0, outputs1, outputs2)
    kv2 = kv2_cat(outputs0, outputs1, outputs2)
    print(kv0[0].shape, kv1[0].shape, kv2[0].shape)
    
    emb1_vit = SBlock_ViT(vis=True, channel_num=L_size[1], KV_size=KVS_size, numheads=4, att_drop_rate=0.1, expand_ratio=4)
    emb1, weight1 = emb1_vit(outputs1, kv1[0], kv1[1])
    emb2_vit = SBlock_ViT(vis=True, channel_num=L_size[2], KV_size=KVS_size, numheads=4, att_drop_rate=0.1, expand_ratio=4)
    emb2, weight2 = emb2_vit(outputs2, kv2[0], kv2[1])
    print("emb2:", emb2.shape)
    '''
    feat0_size = [256, 64, 64]
    feat1_size = [512, 32, 32]
    feat2_size = [1024, 16, 16]
    patchsize = [4, 4, 4]
    encode = Encoder(vis=True, att_drop_rate=0.1, expand_ratio=4)
    emb1, emb2, emb3, attn_weights = encode(feat0, feat1, feat2)
    print(emb1.shape, emb2.shape, emb3.shape)
    
    
    
    
    
    