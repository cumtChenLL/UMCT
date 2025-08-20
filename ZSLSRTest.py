import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import imageio
import pyiqa as iqa
from datasetZLSR import ZLSRDataset
from model.UCtransNet4 import UCTransNet

    
def prepare(*args):
        device = torch.device('cuda')
        def _prepare(tensor):
            return tensor.to(device)

        return [_prepare(a) for a in args]

torch.manual_seed(42)  # 设置随机种子

scale = 4
## test DataLoader
lr_datasetdir = '../Mydataset/benchmark/Eval15/LR_bicubic/X{}'.format(scale)
hr_datasetdir = '../Mydataset/benchmark/Eval15/HR'
dataset_test_ZSSR = ZLSRDataset(lr_datasetdir=lr_datasetdir,hr_datasetdir=hr_datasetdir, scale=scale, test_only=True)
test_ZSSRdataLoader = data.DataLoader(dataset=dataset_test_ZSSR, batch_size=1, shuffle=False)

n_colors = 3
n_feat = 32
s_patch = 64
zssr_generator = UCTransNet(scale=scale, n_colors=n_colors, n_feat=n_feat, s_patch=s_patch).to('cuda')

if True:
    zssr_generator.load_state_dict(torch.load('pre_model/zssr6_zshot_latest.pt',  weights_only=True))
    
cur_dataloader = test_ZSSRdataLoader
zssr_generator.eval()
results_path = 'results/LOL_zssr6zshot/'
with torch.no_grad():
    for batch,(imgdata, name) in enumerate(cur_dataloader):
        low, high = imgdata
        low, high = prepare(low, high)
        # print(low.shape, high.shape)
        h, w = high.shape[2:]
        upsample = F.interpolate(low, size=[h, w], mode="bicubic")
        rec_img = zssr_generator(upsample)
        rec_img = rec_img.cpu()
        demo_rec = rec_img.squeeze().permute(1, 2, 0).numpy()
        demo_rec = np.clip(demo_rec, 0, 1)
        demo_rec = (demo_rec*255).astype(np.uint8)
        imageio.imsave('{}/{}'.format(results_path, name[0]), demo_rec)
    