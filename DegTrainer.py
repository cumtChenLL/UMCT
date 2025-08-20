# Zero shot super-resolution
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import math

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
parser.add_argument('--train_dataLE', type=str, default='LOL_SR/LOLdataset/our485', #'LOLdataset/our485',
                    help='dataset for enhancement')
parser.add_argument('--train_dataZSSR', type=str, default='LOLdataset',
                    help='dataset for ZSSR')
parser.add_argument('--test_dataLE', type=str, default='LOL_SR/LOLdataset/eval15', #'LOLdataset/our485',
                    help='dataset for enhancement')

parser.add_argument('--exist_ref_SR', type=bool, default=True,
                    help='whether existing the reference label for SR')
parser.add_argument('--exist_ref_LE', type=bool, default=True,
                    help='whether existing the reference label for LE')
parser.add_argument('--exist_ref_ZSSR', type=bool, default=True,
                    help='whether existing the reference label for ZSSR')

parser.add_argument('--patch_size', type=int, default=64,
                    help='the patch_size of cropped img')
parser.add_argument('--test_only', type=bool, default=False,
                    help='the model is training or just test')
parser.add_argument('--scale', type=int, default=4,
                    help='the scale factor of super-resolution')
parser.add_argument('--batch_size', type=int, default=8,
                    help='the batch_size of dataloader')

## model
parser.add_argument('--gtnet', type=str, default= 'UNet',
                    help='SR_generator')
parser.add_argument('--dtnet', type=str, default= 'UNet',
                    help='SR_degradation')
parser.add_argument('--glnet', type=str, default= 'UNet',
                    help='LE_generator')
parser.add_argument('--dlnet', type=str, default= 'UNet',
                    help='LE_degradation')

## training
parser.add_argument('--epoches', type=int, default= 100,
                    help='the number of epoches for training')
parser.add_argument('--lr', type=int, default= 100,
                    help='the number of epoches for training')


args =parser.parse_args()

## DataLoader
dataset_train_SR = datasetSR.SRDataset(args)
train_SRdataLoader = data.DataLoader(dataset_train_SR, args.batch_size)

dataset_train_LE = datasetLE.LOLDataset(args)
train_LEdataLoader = data.DataLoader(dataset_train_LE, args.batch_size)

args.test_only = False
dataset_test_LE = datasetLE.LOLDataset(args)
test_LEdataLoader = data.DataLoader(dataset_test_LE, 1)

## Loss
sr_loss = nn.L1Loss()
le_loss = nn.L1Loss()

## Model
sr_generator = gtnet.make_model(args.gtnet).to('cuda')
sr_degrader = dtnet.make_model(args.dtnet).to('cuda')

le_generator = glnet.make_model(args.glnet).to('cuda')
le_degrader = dlnet.make_model(args.dlnet).to('cuda')

## Optimizer
import torch.optim as optim
import itertools
sr_parameters = itertools.chain(sr_generator.parameters(), sr_degrader.parameters())
sr_optimizer = optim.Adam(sr_parameters,lr = 1e-4, betas=(0.9, 0.999), eps=1e-8,weight_decay= 0)
le_parameters = itertools.chain(le_generator.parameters(), le_degrader.parameters())
le_optimizer = optim.Adam(le_parameters,lr = 1e-4, betas=(0.9, 0.999), eps=1e-8,weight_decay= 0)

## scheduler
import torch.optim.lr_scheduler as lrs
sr_scheduler = lrs.StepLR(sr_optimizer, step_size=300, gamma=0.5)
le_scheduler = lrs.StepLR(le_optimizer, step_size=300, gamma=0.5)

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
        rec_img = generator(low)
        consis_low = degrader(rec_img)

        deg_img = degrader(high)
        consis_high = generator(deg_img)
       
        le_penalty = le_loss(rec_img, high) + le_loss(deg_img, low) 
        +0.1 * le_loss(consis_low, low) + 0.1 * le_loss(consis_high, high)

        le_optimizer.zero_grad()
        le_penalty.backward()
        le_optimizer.step()
        penalty_list.append(le_penalty.item())
    
    avg_penalty = np.mean(penalty_list)
    print("[epoch:{}]: loss:{} ".format(epoch, avg_penalty))
    #if(epoch%50==0):
    #    torch.save(generator.state_dict(), "experiments/le_generator_{}.pt".format(epoch))
    #    torch.save(degrader.state_dict(), "experiments/le_degrader_{}.pt".format(epoch))
    
def cac_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1 #255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def test(args, dataloader, generator, degrader, epoch):
    psnr_list = []
    psnr_rec = []
    psnr_deg = []
    generator.eval()
    degrader.eval()
    with torch.no_grad():
        for batch, (low, high, name) in enumerate(dataloader):
            '''
            b, c, h, w = low.shape
            h = h//16*16
            w = w//16*16
            low = low[:,:,0:h, 0:w]
            high = high[:,:,0:h,0:w]
            '''
            low, high = prepare(low, high)
            rec_img = generator(low)
            deg_img = degrader(high)
            
            rec_img = rec_img.cpu().numpy()
            high = high.cpu().numpy()
            deg_img = deg_img.cpu().numpy()
            low = low.cpu().numpy()

            metrix_rec = cac_psnr(rec_img, high) 
            metrix_deg = cac_psnr(deg_img, low)
            metrix = metrix_rec + metrix_deg
            psnr_list.append(metrix)
            psnr_rec.append(metrix_rec)
            psnr_deg.append(metrix_deg)
    
    avg_psnr = np.mean(psnr_list)
    avg_rec_psnr = np.mean(psnr_rec)
    avg_deg_psnr = np.mean(psnr_deg)
    print("[epoch:{}] psnr-generator:{} psnr-degrader:{} psnr:{}".format(epoch, avg_rec_psnr, avg_deg_psnr, avg_psnr))
    


for epoch in range(args.epoches):
    one_epoch_train(args, train_LEdataLoader, le_generator, le_degrader, epoch)
    test(args, test_LEdataLoader, le_generator, le_degrader, epoch)
    
## test