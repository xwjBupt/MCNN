import torch
from torch.utils.data import Dataset,DataLoader
import os
import cv2
import pandas as pd
import numpy as np
from torchvision import transforms as T, utils
from PIL import Image
import random
from matplotlib import pyplot as plt
from scipy.io import loadmat
import math
import h5py
import scipy.misc as misc


class SHT(Dataset):
    def __init__(self, imdir, gtdir, transform=0, train=True, test=False, raw=False, num_cut=4, geo=False):
        self.imdir = imdir
        self.gtdir = gtdir
        self.train = train
        self.test = test
        self.transform = transform
        self.imname = os.listdir(self.imdir)
        self.gtname = [name.replace('jpg', 'h5') for name in self.imname]
        self.imgs = []
        self.gts = []
        self.num_it = len(self.imname)
        self.num_cut = num_cut
        self.raw = raw

        print('Loading data,wait a second')
        PIXEL_MEANS = (0.485, 0.456, 0.406)
        PIXEL_STDS = (0.229, 0.224, 0.225)
        for idx in range(self.num_it):
            if idx % 30 == 0:
                print('loaded %d imgs' % idx)
            imname = os.path.join(self.imdir, self.imname[idx])
            img = cv2.imread(imname,0)
            # img = img[:, :, ::-1]
            h = img.shape[0]
            w = img.shape[1]
            h1 = int(h/4)*4
            w1 = int(w/4)*4
            img = img.astype(np.float32, copy=False)
            # img /= 255.0
            # img -= np.array(PIXEL_MEANS)
            # img /= np.array(PIXEL_STDS)
            img = cv2.resize(img,(w1,h1),interpolation=cv2.INTER_LINEAR)
            gtname = os.path.join(self.gtdir, self.gtname[idx])
            # den = pd.read_csv((gtname), sep=',',header=None).as_matrix()

            f = h5py.File(gtname, 'r')  # 打开h5文件
            den = f['density'].value
            f.close()
            den  = den.astype(np.float32, copy=False)
            if self.raw:
                den = cv2.resize(den,(int(w1),int(h1)),interpolation=cv2.INTER_LINEAR)
                den =  den *(w*h/(w1*h1))
            else:
                w1 = w1//4
                h1 = h1//4
                den = cv2.resize(den, (w1, h1), interpolation=cv2.INTER_LINEAR)
                den = den * (w * h / (w1 * h1))

            # img = img.transpose([2,0,1])
            img = img[np.newaxis,...]
            den = den[np.newaxis, ...]
            self.imgs.append(torch.Tensor(img))
            self.gts.append(torch.Tensor(den))



    def __len__(self):

        return self.num_it

    def __getitem__(self, idx):

        img = self.imgs[idx]
        den = self.gts[idx]
        return img, den


