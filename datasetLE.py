## load datase for low-light image enhancement
import os
import numpy as np
import torch
import torch.utils.data as data
#from PIL import Image
import imageio
import random
import torchvision.transforms as transforms

class LOLDataset(data.Dataset):
    def __init__(self, args):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        if args.test_only==False:
            self.file_path = os.path.join(args.dir_data,args.train_dataLE)
        else:
            self.file_path = os.path.join(args.dir_data,args.test_dataLE)
        self.patch_size = args.patch_size
        self.test_only = args.test_only
        self.exist_ref = args.exist_ref_LE

        #self.exist_ref = False
        # 1 正确读入图片和标签路径
        if(self.exist_ref == True):
            self.label_path = os.path.join(self.file_path,"high")
            self.img_path = os.path.join(self.file_path,"low")
        else:
            self.label_path = os.path.join(self.file_path)
            self.img_path = os.path.join(self.file_path)
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        imgs = self.findlabel(os.listdir(self.img_path),'.png')
        labels = self.findlabel(os.listdir(self.label_path),'.png')
        self.imgs = imgs + imgs
        self.labels = labels + labels

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img_name = os.path.join(self.img_path, img)
        #print(img_name)
        label_name = os.path.join(self.label_path, label)
        img_data = imageio.imread(img_name,pilmode="RGB")
        label_data = imageio.imread(label_name,pilmode="RGB")
        if self.test_only:
            img_patch, label_patch = img_data, label_data
        else:
            img_patch, label_patch = self.center_crop(img_data, label_data, self.patch_size)

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

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        #file_path_list = [os.path.join(path, img) for img in files_list]
        files_list.sort()
        return files_list

    def center_crop(self, img, label, patch_size):
        """裁剪输入的图片和标签大小"""
        h, w = img.shape[:2]
        ih = random.randrange(0, h - patch_size + 1)
        iw = random.randrange(0, w - patch_size + 1)
        patch_img = img[ih:ih+patch_size, iw:iw+patch_size, :]
        patch_label = label[ih:ih+patch_size, iw:iw+patch_size, :]
        return patch_img, patch_label

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
'''
train_path = "..\论文3\lowlightdata\LOLdataset\our485"
batch_size = 64
train_Data = LOLDataset(train_path, batch_size)
img, labels, name = train_Data.__getitem__(0)
print(img.shape, labels.shape, name)
dataloader = data.DataLoader(train_Data, 8)
'''