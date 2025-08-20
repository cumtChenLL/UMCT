import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import math
import imageio
from PIL import Image
from datasetZLSR import ZLSRDataset
from model import glnet, gtnet, dlnet, dtnet
from model.UCtransNet4 import UCTransNet

from grad import Get_gradient,Get_gradient_nopadding
def gradcomput(x):
    model = Get_gradient_nopadding().to('cuda')
    return model(x)

def prepare(*args):
        device = torch.device('cuda')
        def _prepare(tensor):
            return tensor.to(device)

        return [_prepare(a) for a in args]

torch.manual_seed(42)  # 设置随机种子
## DataLoader
patchsize = 16
scale = 4
batchsize = 16
# lr_datasetdir = '../Mydataset/our485/LOLSR_bicubic/X{}'.format(scale)
# hr_datasetdir = '../Mydataset/our485/LOLSR_train_HR'
# lr_datasetdir = '../Mydataset/Real_captured/Train/LR_bicubic/X{}'.format(scale)
# hr_datasetdir = '../Mydataset/Real_captured/Train/Normal'
lr_datasetdir = '../Mydataset/Synthetic/Train/LR_bicubic/X{}'.format(scale)
hr_datasetdir = '../Mydataset/Synthetic/Train/Normal'

dataset_train_ZSSR = ZLSRDataset(lr_datasetdir=lr_datasetdir,hr_datasetdir=hr_datasetdir, patchsize=patchsize, scale=scale, test_only=False)
train_ZSSRdataLoader = data.DataLoader(dataset=dataset_train_ZSSR, batch_size=batchsize, shuffle=True)

## test DataLoader
# lr_datasetdir = '../Mydataset/benchmark/Eval15/LR_bicubic/X{}'.format(scale)
# hr_datasetdir = '../Mydataset/benchmark/Eval15/HR'
# lr_datasetdir = '../Mydataset/Real_captured/Test/LR_bicubic/X{}'.format(scale)
# hr_datasetdir = '../Mydataset/Real_captured/Test/Normal'
lr_datasetdir = '../Mydataset/Synthetic/Test/LR_bicubic/X{}'.format(scale)
hr_datasetdir = '../Mydataset/Synthetic/Test/Normal'
dataset_test_ZSSR = ZLSRDataset(lr_datasetdir=lr_datasetdir,hr_datasetdir=hr_datasetdir, scale=scale, test_only=True)
test_ZSSRdataLoader = data.DataLoader(dataset=dataset_test_ZSSR, batch_size=1, shuffle=False)

## Model
def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable
sr_generator = gtnet.make_model('EDSR').to('cuda')
sr_generator.load_state_dict(torch.load("pre_model/EDSR_baseline_x4.pt", weights_only=True))
trainable(sr_generator, False)
sr_degrader = dtnet.make_model('Downblock').to('cuda')
sr_degrader.load_state_dict(torch.load("pre_model/sr_degradation_499_1.pt",  weights_only=True))
trainable(sr_degrader, False)

le_generator = glnet.make_model('DRBN').to('cuda')
le_generator.load_state_dict(torch.load("pre_model/model_s1.pt",  weights_only=True))
trainable(le_generator, False)
le_degrader = dlnet.make_model('Darkblock').to('cuda')
le_degrader.load_state_dict(torch.load("pre_model/le_degradation_299.pt",  weights_only=True))
trainable(le_degrader, False)

scale = 4 
n_colors = 3
n_feat = 32
s_patch = 64
zssr_generator = UCTransNet(scale=scale, n_colors=n_colors, n_feat=n_feat, s_patch=s_patch).to('cuda')
if False:
    zssr_generator.load_state_dict(torch.load('pre_model/zssr6_generator_latest.pt',  weights_only=True))

## Optimizer
zssr_optimizer = optim.Adam(zssr_generator.parameters(),lr=1e-4, betas=(0.9, 0.999), eps= 1e-8, weight_decay= 0)
## scheduler
zssr_scheduler = lrs.StepLR(zssr_optimizer, step_size=500, gamma=0.5)

## Loss
import loss_func
def cac_loss(sr, gt):
    loss = []
    l1 = loss_func.l1_loss(sr, gt)
    loss.append(l1)
    # l2 = loss_func.l2_loss(sr, gt)
    # loss.append(l2)
    ssim = 0.2 * loss_func.ssim_loss(sr, gt)
    loss.append(ssim)
    vgg2 = 0.2* loss_func.vgg2_loss(sr, gt)
    loss.append(vgg2)
    spect = 0.2*(loss_func.spect_loss(sr, gt))
    loss.append(spect)
    L = sum(loss)
    return L

## train
epoches = 1001
cur_dataloader = train_ZSSRdataLoader

penalty_list = []
aver_penalty = 0.0
zssr_generator.train()
sr_generator.eval()
sr_degrader.eval()
le_generator.eval()
le_degrader.eval()

for epoch in range(epoches):
    for batch,(imgdata, name) in enumerate(cur_dataloader):
        low, high = imgdata
        low, high = prepare(low, high)

        h, w = high.shape[2:]
        upsample = F.interpolate(low, size=[h, w], mode="bicubic")
        rec_img = zssr_generator(upsample)
        
        if False: # zero-shot
            low_le_g = le_generator(upsample)
            low_le_g = low_le_g[0]
            low_sr_g = sr_generator(low)
            
        
            if False: # online
                rec_img_sr_d = sr_degrader(rec_img)
                rec_img_le_d = le_degrader(rec_img)
                zssr_legsrd_penalty = cac_loss(rec_img_sr_d, low_le_g)
                zssr_srgled_penalty = cac_loss(rec_img_le_d, low_sr_g)
                zssr_penalty = zssr_legsrd_penalty + zssr_srgled_penalty
            
            else: # 
                zssr_sr_grad = 1.0 * cac_loss(rec_img_grad, low_sr_g_grad)
                zssr_sr_light = cac_loss(rec_img, low_le_g)
                zssr_penalty = zssr_sr_grad + zssr_sr_light
        
        else:
            zssr_penalty = cac_loss(rec_img, high)
            
        zssr_optimizer.zero_grad()
        zssr_penalty.backward()
        zssr_optimizer.step()
        penalty_list.append(zssr_penalty.item())
    zssr_scheduler.step()
    
    avg_penalty = np.mean(penalty_list)
    learning_rate = zssr_scheduler.get_last_lr()[0]
    print("[epoch:{}]: loss:{} learning rate:{} ".format(epoch, avg_penalty, learning_rate))
    
    # test
    # lr_datasetdir = 'E:/low_light/zeroshot/Mydataset/benchmark/Eval15/LR_bicubic/X{}'.format(scale)
    # hr_datasetdir = 'E:/low_light/zeroshot/Mydataset/benchmark/Eval15/HR_cut'
    with torch.no_grad():
        demo_lr = os.path.join(lr_datasetdir, 'r0603031et.png') # r0603031et.png # 55.png # 00707.png
        low = imageio.imread(demo_lr,pilmode="RGB")
        low = torch.from_numpy(low).permute(2, 0, 1).unsqueeze(0).float()/255.0
        upsample = F.interpolate(low, scale_factor=4, mode='bicubic')
        patch = zssr_generator(upsample.to('cuda'))
        demo_rec = patch.cpu()  
        demo_rec = demo_rec.squeeze().permute(1, 2, 0).numpy()
        demo_rec = (demo_rec*255).astype(np.uint8)
        imageio.imsave('demo_rec6_Synthr0603031et.png.png', demo_rec)
    
    if epoch%100==1:
        torch.save(zssr_generator.state_dict(), "pre_model/zssr6_synth_latest.pt")


