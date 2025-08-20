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
parser.add_argument('--train_dataLE', type=str, default='LOL_SR/LOLdataset/our485', #'LOLdataset/our485',
                    help='dataset for enhancement')
parser.add_argument('--test_dataLE', type=str, default='LOL_SR/LOLdataset/eval15', #'LOLdataset/our485',
                    help='dataset for enhancement')

parser.add_argument('--exist_ref_LE', type=bool, default=True,
                    help='whether existing the reference label for LE')

parser.add_argument('--patch_size', type=int, default=64,
                    help='the patch_size of cropped img')
parser.add_argument('--test_only', type=bool, default=False,
                    help='the model is training or just test')
parser.add_argument('--batch_size', type=int, default=8,
                    help='the batch_size of dataloader')

## model
parser.add_argument('--glnet', type=str, default= 'DRBN',
                    help='LE_generator')
parser.add_argument('--dlnet', type=str, default= 'Darkblock',
                    help='LE_degradation')

## training
parser.add_argument('--epoches', type=int, default= 500,
                    help='the number of epoches for training')
parser.add_argument('--lr', type=int, default= 0.0001,
                    help='the learning rate for training')


args =parser.parse_args()

## DataLoader
dataset_train_LE = datasetLE.LOLDataset(args)
train_LEdataLoader = data.DataLoader(dataset_train_LE, args.batch_size)

args.test_only = True
dataset_test_LE = datasetLE.LOLDataset(args)
test_LEdataLoader = data.DataLoader(dataset_test_LE, 1)


## Loss
le_loss = nn.L1Loss()

## Model
le_generator = glnet.make_model(args.glnet).to('cuda')
le_degrader = dlnet.make_model(args.dlnet).to('cuda')

## Optimizer
import torch.optim as optim
import itertools
#le_parameters = itertools.chain(le_generator.parameters(), le_degrader.parameters())
#le_optimizer = optim.Adam(le_parameters,lr = 1e-4, betas=(0.9, 0.999), eps=1e-8,weight_decay= 0)
gen_optimizer = optim.Adam(le_generator.parameters(), lr = 1e-4, betas=(0.9, 0.999), eps=1e-8,weight_decay= 0)

## scheduler
import torch.optim.lr_scheduler as lrs
#le_scheduler = lrs.StepLR(le_optimizer, step_size=300, gamma=0.5)
gen_scheduler = lrs.StepLR(gen_optimizer, step_size=100, gamma=0.5)

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
        gen_img = generator(low)
        gen_img = gen_img[0]
       
        le_penalty = le_loss(gen_img, low) 

        gen_optimizer.zero_grad()
        le_penalty.backward()
        gen_optimizer.step()
        penalty_list.append(le_penalty.item())
    gen_scheduler.step()
    
    avg_penalty = np.mean(penalty_list)
    learning_rate = gen_scheduler.get_last_lr()[0]
    print("[epoch:{}]: loss:{} learning rate:{} ".format(epoch, avg_penalty, learning_rate))

#for epoch in range(args.epoches):
#    one_epoch_train(args, train_LEdataLoader, le_generator, le_degrader, epoch)
    
#torch.save(le_generator.state_dict(), "pre_model/le_generator_{}.pt".format(epoch))


#test
def tensor_save_rgbimage(tensor, filename, cuda=False):
	if cuda:
		img = tensor.clone().cpu().clamp(0, 255).numpy()
	else:
		img = tensor.clone().clamp(0, 255).numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	img = Image.fromarray(img)
	img.save(filename)

def cac_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1 #255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def le_test(args, dataloader, generator, epoch):
    psnr_gen = []
    generator.eval()
    with torch.no_grad():
        for batch, (low, high, name) in enumerate(dataloader):
            
            b, c, h, w = low.shape
            #print(low.shape, high.shape)
            h = h//16*16
            w = w//16*16
            low = low[:,:,0:h, 0:w]
            high = high[:, :, 0:h, 0:w]

            low, high = prepare(low, high)
            gen_img = generator(low)
            gen_img = gen_img[0]
            print(gen_img.shape)
            tensor_save_rgbimage(255*gen_img[0], "rec_"+str(name[0]), cuda=True)
            
            gen_img = gen_img.cpu().numpy()
            high = high.cpu().numpy()

            metrix_gen = cac_psnr(gen_img, high)
            
            print("{}: metrix_gen:{}".format(name, metrix_gen))
            psnr_gen.append(metrix_gen)
    
    avg_gen_psnr = np.mean(psnr_gen)
    print("[epoch:{}]  psnr-generator:{}".format(epoch, avg_gen_psnr))

le_generator.load_state_dict(torch.load("pre_model/model_s1.pt"))
le_test(args,test_LEdataLoader, le_generator,1)


