import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool = True,**kwargs):
        super(BasicConv, self).__init__()
        self.use_pool = use_pool
        self.conv = nn.Conv2d(in_channels,out_channels,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True)
        if self.use_pool:
            self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.use_pool :
            x = self.pool(x)

        return x

class Branch1(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(Branch1,self).__init__()
        self.inc = in_channels
        self.out = out_channels

        self.CONV = nn.Sequential(OrderedDict([
            ('Branch1Conv1',BasicConv(in_channels = self.inc,out_channels = 16,use_pool=False,kernel_size = 9,padding = 4)),
            ('Branch1Conv2', BasicConv(in_channels=16, out_channels=32, use_pool=True, kernel_size=7, padding=3)),
             ('Branch1Conv3', BasicConv(in_channels=32, out_channels=16, use_pool=True, kernel_size=7, padding=3)),
            ('Branch1Conv4', BasicConv(in_channels=16, out_channels=self.out, use_pool=False, kernel_size=7, padding=3))
        ]))

    def forward(self, x):
        x = self.CONV(x)
        return x


class Branch2(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Branch2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.CONV = nn.Sequential(OrderedDict([
            ('Branch2Conv1', BasicConv(in_channels=in_channels, out_channels=20, use_pool=False, kernel_size=7, padding=3)),
             ('Branch2Conv2', BasicConv(in_channels=20, out_channels=40, use_pool=True, kernel_size=5, padding=2)),
              ('Branch2Conv3', BasicConv(in_channels=40, out_channels=20, use_pool=True, kernel_size=5, padding=2)),
               ('Branch2Conv4', BasicConv(in_channels=20, out_channels=out_channels, use_pool=False, kernel_size=5, padding=2))
        ]))

    def forward(self, x):
        x = self.CONV(x)
        return x


class Branch3(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Branch3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.CONV = nn.Sequential(OrderedDict([
            ('Branch3Conv1', BasicConv(in_channels=in_channels, out_channels=24, use_pool=False, kernel_size=5, padding=2)),
             ('Branch3Conv2', BasicConv(in_channels=24, out_channels=48, use_pool=True, kernel_size=3, padding=1)),
              ('Branch3Conv3', BasicConv(in_channels=48, out_channels=24, use_pool=True, kernel_size=3, padding=1)),
               ('Branch3Conv4', BasicConv(in_channels=24, out_channels=out_channels, use_pool=False, kernel_size=3, padding=1))
        ]))

    def forward(self, x):
        x = self.CONV(x)
        return x

class Final(nn.Module):
    def __init__(self):
        super(Final,self).__init__()


class MCNN(nn.Module):
    def __init__(self,branch1 ,branch2 ,branch3,whole):
        super(MCNN,self).__init__()
        self.branch1 = branch1
        self.branch2 = branch2
        self.branch3 = branch3
        self.whole = whole

        if self.branch1 and not self.whole:
            self.Branch1 = Branch1(3,8)
            self.final = BasicConv(in_channels=8,out_channels=3,use_pool=False,kernel_size = 1)
        if self.branch2 and not self.whole:
                self.Branch2 = Branch2(3, 10)
                self.final = BasicConv(in_channels=10, out_channels=3, use_pool=False, kernel_size=1)
        if self.branch3 and not self.whole:
                self.Branch3 = Branch3(3, 12)
                self.final = BasicConv(in_channels=12, out_channels=3, use_pool=False, kernel_size=1)

        if self.whole:
            self.Branch1 = Branch1(1,8)
            self.Branch2 = Branch2(1, 10)
            self.Branch3 = Branch3(1, 12)
            self.final = BasicConv(in_channels=30,out_channels=1,use_pool=False,kernel_size = 1)

    def forward(self, x):
        if self.branch1 and not self.whole:
            x = self.Branch1(x)
            x = self.final(x)
            return x
        if self.branch2 and not self.whole:
            x = self.Branch2(x)
            x = self.final(x)
            return x
        if self.branch3 and not self.whole:
            x = self.Branch3(x)
            x = self.final(x)
            return x
        if  self.whole:
            x1 = self.Branch1(x)
            x2 = self.Branch2(x)
            x3 = self.Branch3(x)
            x = torch.cat([x1,x2,x3],1)
            x = self.final(x)
            return x

class M(nn.Module):
    def __init__(self, bn=False):
        super(M, self).__init__()

        self.branch1 = nn.Sequential(Conv2d(1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16, 8, 7, same_padding=True, bn=bn))

        self.branch2 = nn.Sequential(Conv2d(1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))

        self.branch3 = nn.Sequential(Conv2d(1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))

        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)

        return x