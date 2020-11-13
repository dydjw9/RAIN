from model import common

import torch.nn as nn
import torch
import numpy as np
from MiscTools import imgshift
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from options import args


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class embed_resnet(nn.Module):
    def __init__(self,edsr_model,dnn_model,RdmDU=False,gaus_noise=False,edsr_out=False,mean_filter=args.ismean_filter):
        super(embed_resnet,self).__init__()
        self.edsr_model = edsr_model
        self.dnn_model = dnn_model
        self.normVector = {'mean':torch.from_numpy(np.array([0.5, 0.5, 0.5])), 
                                'std':torch.from_numpy(np.array([0.5, 0.5, 0.5]))}
        #weather to take RdmDU
        self.isRdmDU = RdmDU

    def forward(self,x,isround=False):
        if self.isRdmDU:
            #RdmDU
            x = self.RdmDU(x,2)
        else:
            x = F.interpolate(x,scale_factor=0.5,mode = "bicubic")
        #RdmSCS
        x = imgshift(x)
        if isround:
            x = torch.round(x)
        edsr_output = self.edsr_model(x)
        x = edsr_output
        x = self.normalize(x)
        output = self.dnn_model(x)
        return output
    def normalize(self,img):
        img = torch.clamp(img,0,255)
        img = img/255
        for i in range(3):
            img[:,i,:,:] = (img[:,i,:,:]-self.normVector['mean'][i]) / self.normVector['std'][i]
        return img
    def RdmDU(self,img,kernel_size=2):
        size = img.size()
        random_mask = torch.randn(size)
        select_mask = F.max_pool2d(random_mask,kernel_size)
        select_mask = torch.nn.Upsample(scale_factor=2, mode='nearest')(select_mask)
        select_mask = (random_mask==select_mask).float().cuda()
        down_img = img*select_mask
        down_img = F.max_pool2d(down_img,kernel_size)
        return down_img

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        # self.url = url['r{}f{}x{}'.format(n_resblocks, n_feats, scale)]
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

