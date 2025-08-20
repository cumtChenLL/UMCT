## load image list
import os
import numpy as np
import torch
import torch.utils.data as data
#from PIL import Image
import imageio
import random
import torchvision.transforms as transforms

class SRDataset(data.Dataset):
    def __init__(self,args):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        '''
        self.file_path = "../../Paper4/datasets/RealSRV3/Canon/Train/4"
        self.batch_size = 64
        self.test_only = False
        self.exist_ref = True
        '''
        if args.test_only ==False:
            self.file_path = os.path.join(args.dir_data,args.train_dataSR, str(args.scale))
        else:
            self.file_path = os.path.join(args.dir_data,args.test_dataSR, str(args.scale))
        self.patch_size = args.patch_size
        self.test_only = args.test_only
        self.exist_ref = args.exist_ref_SR

        # self.scale = 4
        self.scale = args.scale
        
        # 1 正确读入图片和标签路径
        if(self.exist_ref == True):
            file_list = os.listdir(self.file_path)
            labels = self.findlabel(file_list,"HR")
            imgs = self.findlabel(file_list,"LR")
            #print(self.labels[1],self.imgs[1])
            self.imgs = imgs + imgs
            self.labels = labels + labels

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img_name = os.path.join(self.file_path, img)
        label_name = os.path.join(self.file_path, label)
        img_data = imageio.imread(img_name,pilmode="RGB")
        label_data = imageio.imread(label_name,pilmode="RGB")
        #print(img_data.shape, label_data.shape)
        if self.test_only:
            img_patch = img_data
            ih, iw = img_patch.shape[:2]
            label_patch = label_data[0:ih*self.scale, 0:iw*self.scale, :]
        else:
            img_patch, label_patch = self.get_patch(img_data, label_data, self.patch_size, self.scale)

        img_patch, label_patch = self.img_transform(img_patch, label_patch)
        # print('处理后的图片和标签大小：',img.shape, label.shape)
        #sample = {'img': img, 'label': label}

        return img_patch, label_patch, label

    def __len__(self):
        return len(self.imgs)

    def findlabel(self, file_list, substr):
        label_list = []
        for file_name in file_list:
            if file_name.find(substr)>0: label_list.append(file_name) 
        label_list.sort()
        return label_list

    def center_crop(self, img, label, patch_size):
        """裁剪输入的图片和标签大小"""
        h, w = img.shape[:2]
        ih = random.randrange(0, h - patch_size + 1)
        iw = random.randrange(0, w - patch_size + 1)
        patch_img = img[ih:ih+patch_size, iw:iw+patch_size, :]
        patch_label = label[ih:ih+patch_size, iw:iw+patch_size, :]
        return patch_img, patch_label
    
    def get_patch(self, lr, hr, patch_size=64, scale=1):
        ih, iw = lr.shape[:2]

        tp = scale * patch_size
        ip = tp // scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

        lr_patch = lr[iy:iy + ip, ix:ix + ip, :]
        hr_patch = hr[ty:ty + tp, tx:tx + tp, :]
        # print(ix, iy, tx, ty, tp)
        # print(lr.shape, hr.shape, hr_batch.shape)

        return [lr_patch, hr_patch]

    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform(img)
        label = transform(label)

        return img, label

#train_path = "..\论文3\lowlightdata\LOLdataset\our485"
#batch_size = 64
# train_Data = LOLDataset()
# img, label_img, name = train_Data.__getitem__(0)
# print(img.shape, label_img.shape, name)
# dataloader = data.DataLoader(train_Data, 8)
