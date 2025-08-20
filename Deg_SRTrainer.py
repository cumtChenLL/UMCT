import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import math
from PIL import Image

import argparse

import datasetSR, datasetLE, datasetZSSR
from model import dlnet, dtnet, glnet, gtnet, gltnet

## Config
parser = argparse.ArgumentParser(description='ZShotSRTraining')
## Dataset
parser.add_argument('--dir_data', type=str, default= '../../datasets', #'../../Paper4/datasets',
                    help='dataset directory')
parser.add_argument('--train_dataSR', type=str, default='RealSRV3/Canon/Train',
                    help='dataset for super-resolution')  
parser.add_argument('--test_dataSR', type=str, default='RealSRV3/Canon/Test', #'LOLdataset/our485',
                    help='dataset for super-resolution')


parser.add_argument('--exist_ref_SR', type=bool, default=True,
                    help='whether existing the reference label for SR')

parser.add_argument('--patch_size', type=int, default=64,
                    help='the patch_size of cropped img')
parser.add_argument('--test_only', type=bool, default=False,
                    help='the model is training or just test')
parser.add_argument('--scale', type=int, default=4,
                    help='the scale factor of super-resolution')
parser.add_argument('--batch_size', type=int, default=8,
                    help='the batch_size of dataloader')

## model
parser.add_argument('--gtnet', type=str, default= 'EDSR',
                    help='SR_generator')
parser.add_argument('--dtnet', type=str, default= 'Downblock',
                    help='SR_degradation')


## training
parser.add_argument('--epoches', type=int, default= 500,
                    help='the number of epoches for training')
parser.add_argument('--lr', type=int, default= 0.0001,
                    help='the learning rate for training')


args =parser.parse_args()

## DataLoader
dataset_train_SR = datasetSR.SRDataset(args)
train_SRdataLoader = data.DataLoader(dataset_train_SR, args.batch_size)

args.test_only = True
dataset_test_SR = datasetSR.SRDataset(args)
test_SRdataLoader = data.DataLoader(dataset_test_SR, 1)

## Loss
sr_loss = nn.L1Loss()
sr_consis = nn.MSELoss()

## Model
sr_generator = gtnet.make_model(args.gtnet).to('cuda')
sr_degrader = dtnet.make_model(args.dtnet).to('cuda')

## Optimizer
import torch.optim as optim
import itertools
sr_parameters = itertools.chain(sr_generator.parameters(), sr_degrader.parameters())
sr_optimizer = optim.Adam(sr_parameters,lr = 1e-4, betas=(0.9, 0.999), eps=1e-8,weight_decay= 0)
deg_optimizer = optim.Adam(sr_degrader.parameters(),lr = 1e-4, betas=(0.9, 0.999), eps=1e-8,weight_decay= 0)

## scheduler
import torch.optim.lr_scheduler as lrs
sr_scheduler = lrs.StepLR(sr_optimizer, step_size=100, gamma=0.5)
deg_scheduler = lrs.StepLR(deg_optimizer, step_size=100, gamma=0.5)

## train
def prepare(*args):
        device = torch.device('cuda')
        def _prepare(tensor):
            return tensor.to(device)

        return [_prepare(a) for a in args]

def one_epoch_train(args, data_loader, generator, degrader, epoch):
    cur_dataloader = data_loader
    penalty_list = []
    aver_penalty = 0.0
    generator.train()
    degrader.train()
    
    for batch, (low, high, name) in enumerate(cur_dataloader):
        low, high = prepare(low, high)
        h, w = high.shape[2:]
        low = F.interpolate(low, size = [h, w], mode='bicubic')
        deg_img = degrader(high)
       
        sr_penalty = sr_loss(deg_img, low) + sr_consis(deg_img, low)

        deg_optimizer.zero_grad()
        sr_penalty.backward()
        deg_optimizer.step()
        penalty_list.append(sr_penalty.item())
    deg_scheduler.step()
    
    avg_penalty = np.mean(penalty_list)
    learning_rate = deg_scheduler.get_last_lr()[0]
    print("[epoch:{}]: loss:{} learning rate:{} ".format(epoch, avg_penalty, learning_rate))

#for epoch in range(args.epoches):
#    one_epoch_train(args, train_SRdataLoader, sr_generator, sr_degrader, epoch)
#    
#torch.save(sr_degrader.state_dict(), "pre_model/sr_degradation_{}_1.pt".format(epoch))

#test
def cac_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1 #255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def sr_test(args, dataloader, degrader, epoch):
    psnr_deg = []
    degrader.eval()
    with torch.no_grad():
        for batch, (low, high, name) in enumerate(dataloader):
            
            b, c, h, w = low.shape
            #print(low.shape, high.shape)
            h = h//16*16
            w = w//16*16
            low = low[:,:,0:h, 0:w]
            high = high[:, :, 0:args.scale*h, 0:args.scale*w]

            low, high = prepare(low, high)
            h, w = high.shape[2:]
            low = F.interpolate(low, size = [h, w], mode='bicubic')
            deg_img = degrader(high)
            #tensor_save_rgbimage(255*rec_img[0], "rec_"+str(name[0]), cuda=True)
            
            deg_img = deg_img.cpu().numpy()
            low = low.cpu().numpy()

            metrix_deg = cac_psnr(deg_img, low)
            
            print("{}: metrix_deg:{}".format(name, metrix_deg))
            psnr_deg.append(metrix_deg)
    
    avg_deg_psnr = np.mean(psnr_deg)
    print("[epoch:{}]  psnr-degrader:{}".format(epoch, avg_deg_psnr))

sr_degrader.load_state_dict(torch.load("pre_model/sr_degradation_499_1.pt"))
sr_test(args,test_SRdataLoader, sr_degrader,1)    
