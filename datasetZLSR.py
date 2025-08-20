## load image list
import os
import numpy as np
import torch
import torch.utils.data as data
#from PIL import Image
# import imageio.v2 as imageio
import imageio
import random
import torchvision.transforms as transforms

class ZLSRDataset(data.Dataset):
    def __init__(self, lr_datasetdir, hr_datasetdir='', patchsize=64, scale=4, exist_ref = True, test_only=True):
        super(ZLSRDataset, self).__init__()
        self.lr_datasetdir = lr_datasetdir
        self.hr_datasetdir = hr_datasetdir
        self.exist_ref = exist_ref
        self.test_only = test_only
        self.patch_size = patchsize
        self.scale = scale
        
        if(exist_ref == True):
            self.file_list_labels = os.listdir(str(hr_datasetdir))
            self.file_list_imgs = os.listdir(str(lr_datasetdir))
            self.file_list_labels = [name for name in self.file_list_labels if any(name.endswith(ext) for ext in ['.png'])]
            self.file_list_imgs = [name for name in self.file_list_imgs if any(name.endswith(ext) for ext in ['.png'])]
            
        else:
            self.file_list_imgs = os.listdir(str(lr_datasetdir))
            self.file_list_imgs = [name for name in self.file_list_imgs if any(name.endswith(ext) for ext in ['.png'])]

        
        
    def __getitem__(self, index):
        lr_name = os.path.join(self.lr_datasetdir, self.file_list_imgs[index])
        img_data = imageio.imread(lr_name,pilmode="RGB")
        if self.exist_ref:
            hr_name = os.path.join(self.hr_datasetdir, self.file_list_labels[index])
            label_data = imageio.imread(hr_name,pilmode="RGB")
        
        if self.test_only:
            img_patch = img_data
            ih, iw = img_patch.shape[:2]
            if self.exist_ref:
                label_patch = label_data[0:ih*self.scale, 0:iw*self.scale, :]
            else:
                label_patch = None
        
        else:
            img_patch, label_patch = self.get_patch(img_data, label_data, self.patch_size, self.scale)
            
        img_patch = torch.from_numpy(img_patch).permute(2, 0, 1).float()/255.0
        label_patch = torch.from_numpy(label_patch).permute(2, 0, 1).float()/255.0
        return [img_patch, label_patch], self.file_list_imgs[index]
    
    def __len__(self):
        return len(self.file_list_imgs)

    def findlabel(self, file_list, substr):
        label_list = []
        for file_name in file_list:
            if file_name.find(substr)>0: label_list.append(file_name) 
        label_list.sort()
        return label_list

    def get_patch(self, lr, hr, patch_size=64, scale=1):
        ih, iw = lr.shape[:2]

        tp = scale * patch_size
        ip = tp // scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

        lr_patch = lr[iy:iy + ip, ix:ix + ip, :]
        if self.exist_ref:
            hr_patch = hr[ty:ty + tp, tx:tx + tp, :]
        else:
            hr_patch = None
        # print(ix, iy, tx, ty, tp)
        # print(lr.shape, hr.shape, hr_batch.shape)

        return [lr_patch, hr_patch]
    
    
if __name__=="__main__":
    lr_datasetdir = 'E:/low_light/zeroshot/Mydataset/our485/LOLSR_bicubic/X2'
    hr_datasetdir = 'E:/low_light/zeroshot/Mydataset/our485/LOLSR_train_HR'
    file_list_labels = os.listdir(str(hr_datasetdir))
    mydataset = ZLSRDataset(lr_datasetdir=lr_datasetdir, hr_datasetdir=hr_datasetdir)
    imgdata, imgname = mydataset.__getitem__(3)
    print(imgdata[0].shape, imgdata[1].shape, imgname)
    
    