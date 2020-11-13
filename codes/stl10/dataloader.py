import os
import os.path as osp

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import skimage
import numpy as np
from scipy import io
import sys
from PIL import Image 
from MiscTools import add_noise_numpy, augment,add_noise_tensor
class imgnet_png(data.Dataset):
    def __init__(self,dir_data,transform=None):

        self.dir_data = dir_data
        labels_path = "../../data/stl10/val_set/_label.npy"
        self.labels = np.load(labels_path)
        self.transform = transform
    def __len__(self):
        return 1000
    def __getitem__(self,idx):
        save_name = osp.join(self.dir_data,str(idx)+".png")
        img = Image.open(save_name)
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]
        label = torch.tensor(label)
        label = label.long()
        return img, label

def get_defense_loaders(path , isNorm=True, batch_size=64,shuffle=False):
    
    if not isNorm:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255),
            ])
 
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])


    imagedataset = imgnet_png(path,val_transform)
    val_loader = torch.utils.data.DataLoader(imagedataset,batch_size=batch_size,num_workers = 10,pin_memory=True,shuffle=shuffle)
    return val_loader


if __name__ == '__main__':
    tr_dl, _ = get_ImgNet_loaders(isTrain=False, noise=['G',15])
    for x, y in tr_dl:
        import matplotlib.pyplot as plt; import pdb; pdb.set_trace()
    pass
