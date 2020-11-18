import random
import numpy as np
import sys
import skimage.color as sc
import torch
from options import args
sys.path.append("../ad_attack")
sys.path.append("../reference/RayS")
from random import randint, uniform
import torch.nn.functional as F
from advertorch.attacks import CarliniWagnerL2Attack as cwattack
from advertorch.attacks import ElasticNetL1Attack as ead
from advertorch.attacks import LinfMomentumIterativeAttack as mifgsm
from advertorch.attacks import LinfPGDAttack as pgd
from advertorch.attacks import FGSM as fgsm
from general_torch_model import GeneralTorchModel
from RayS import RayS
from simple_black import SimpleBlack,EOT,EOT_CW
def rays_attack(max_count,model,train_loader,max_epsilon,learning_rate,iters=20,isnorm=False,num_classes=10,num_samples=10):
    if isnorm:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        mean = torch.tensor(mean).float().view(3,1,1)
        std = torch.tensor(std).float().view(3,1,1)
        mmax = torch.ones(3,96,96)
        mmin = torch.zeros(3,96,96)
        mmax = ((mmax-mean)/std).cuda()
        mmin = ((mmin-mean)/std).cuda()
        learning_rate = learning_rate /(255*0.5)
        max_epsilon = max_epsilon/(255*0.5)
        preprocessing = ([0.5, 0.5, 0.5], [0.5, 0.5,0.5])
    else:
        mmax = 255
        mmin = 0
        max_epsilon = max_epsilon
        preprocessing = ([0, 0, 0], [1/255,1/255,1/255])
    torch_model = GeneralTorchModel(model, n_class=num_classes)
    adversary = RayS(torch_model, epsilon=max_epsilon)
    count = 0
    total_correct = 0
    for x,y in train_loader: 
        x = x.cuda()
        y = y.cuda()
        target = (y.clone() +3)%num_classes
        count += len(x)
        ad_ex,_,_,succ = adversary(x,y,query_limit=iters)
        if not isnorm:
            ad_ex = torch.round(ad_ex)
        diff = ad_ex -x 
        diff = diff.clamp(-1*max_epsilon,max_epsilon)
        ad_ex = x + diff
        z1 = model(ad_ex).argmax(dim=1)
        total_correct += (z1==y).sum()

        if count >= max_count:
            break
    return total_correct.cpu().numpy()/(count)



def deepfool(max_count,model,train_loader,max_epsilon,iters=20,isnorm=False,num_classes=10):
    import foolbox as fb
    import eagerpy as ep
    if isnorm:
        # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        preprocessing = dict(mean=[1, 1, 1], std=[1, 1, 1], axis=-3)
        max_epsilon = max_epsilon /(255*0.5)
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1),preprocessing=preprocessing)
 
    else:
        preprocessing = dict(mean=[1, 1, 1], std=[1, 1, 1], axis=-3)
        fmodel = fb.PyTorchModel(model, bounds=(0, 255),preprocessing=preprocessing)
        mmax = 255
        mmin = 0
    adversary = fb.attacks.deepfool.LinfDeepFoolAttack(steps=iters,candidates=5)
    count = 0
    total_correct = 0
    for x,y in train_loader: 
        x = x.cuda()
        y = y.cuda()
        count += len(x)
        x = ep.astensor(x)
        y = ep.astensor(y)
        ad_ex = adversary(fmodel,x,y,epsilons=max_epsilon)[1]
        z1 = fmodel(ad_ex).argmax(1)
        total_correct += (z1==y).sum()


        if count >= max_count:
            break
    return total_correct.numpy()/(count)
def ead_attack(max_count,model,train_loader,learning_rate,iters=20,isnorm=False,num_classes=10):
    if isnorm:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        mean = torch.tensor(mean).float().view(3,1,1)
        std = torch.tensor(std).float().view(3,1,1)
        mmax = torch.ones(3,96,96)
        mmin = torch.zeros(3,96,96)
        mmax = ((mmax-mean)/std).cuda()
        mmin = ((mmin-mean)/std).cuda()
        learning_rate = learning_rate /(255*0.5)
 
    else:
        mmax = 255
        mmin = 0
        learning_rate = learning_rate/(8)
    adversary = ead(model,num_classes = num_classes,max_iterations=iters,clip_min=mmin,clip_max=mmax,learning_rate=learning_rate,abort_early = False)
    count = 0
    total_correct = 0
    for x,y in train_loader: 
        x = x.cuda()
        y = y.cuda()
        count += len(x)
        ad_ex = adversary.perturb(x,y)
        if not isnorm:
            ad_ex = torch.round(ad_ex)
        z1 = model(ad_ex).argmax(dim=1)
        dif = x - ad_ex
        total_correct += (z1==y).sum()

        if count >= max_count:
            break
    return total_correct.cpu().numpy()/(count)

def EOT_CW_attack(max_count,model,train_loader,learning_rate,samples=20,iters=20,isnorm=False,num_classes=10,num_samples=10):
    def clip(x,z):
        upper_bound = torch.clamp(x+8,0,255)
        lower_bound = torch.clamp(x-8,0,255)
        z = torch.min(z,upper_bound)
        z = torch.max(z,lower_bound)
        return z

    if isnorm:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        mean = torch.tensor(mean).float().view(3,1,1)
        std = torch.tensor(std).float().view(3,1,1)
        mmax = torch.ones(3,96,96)
        mmin = torch.zeros(3,96,96)
        mmax = ((mmax-mean)/std).cuda()
        mmin = ((mmin-mean)/std).cuda()
        learning_rate = learning_rate /(255*0.5)
 
    else:
        mmax = 255
        mmin = 0
        learning_rate = learning_rate
    adversary = EOT_CW(model,num_classes = num_classes,max_iterations=iters,clip_min=mmin,clip_max=mmax,learning_rate=learning_rate,abort_early = True,num_samples=num_samples)
    count = 0
    total_correct = 0
    for x,y in train_loader: 
        x = x.cuda()
        y = y.cuda()
        count += len(x)
        ad_ex = adversary.perturb(x,y)
        if not isnorm:
            ad_ex = torch.round(ad_ex)
            ad_ex = clip(x,ad_ex)
        z1 = model(ad_ex).argmax(dim=1)
        dif = x - ad_ex
        total_correct += (z1==y).sum()

        if count >= max_count:
            break
    return total_correct.cpu().numpy()/(count)



def CW_attack(max_count,model,train_loader,learning_rate,iters=20,isnorm=False,num_classes=10):
    def clip(x,z):
        upper_bound = torch.clamp(x+8,0,255)
        lower_bound = torch.clamp(x-8,0,255)
        z = torch.min(z,upper_bound)
        z = torch.max(z,lower_bound)
        return z

    if isnorm:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        mean = torch.tensor(mean).float().view(3,1,1)
        std = torch.tensor(std).float().view(3,1,1)
        mmax = torch.ones(3,96,96)
        mmin = torch.zeros(3,96,96)
        mmax = ((mmax-mean)/std).cuda()
        mmin = ((mmin-mean)/std).cuda()
        learning_rate = learning_rate /(255*0.5)
 
    else:
        mmax = 255
        mmin = 0
        learning_rate = learning_rate
    adversary = cwattack(model,num_classes = num_classes,max_iterations=iters,clip_min=mmin,clip_max=mmax,learning_rate=learning_rate,abort_early = False)
    count = 0
    total_correct = 0
    for x,y in train_loader: 
        x = x.cuda()
        y = y.cuda()
        count += len(x)
        ad_ex = adversary.perturb(x,y)
        if not isnorm:
            ad_ex = torch.round(ad_ex)
            ad_ex = clip(x,ad_ex)
        z1 = model(ad_ex).argmax(dim=1)
        dif = x - ad_ex
        total_correct += (z1==y).sum()

        if count >= max_count:
            break
    return total_correct.cpu().numpy()/(count)

def fgsm_attack(max_count,model,train_loader,max_epsilon,isnorm=False,num_classes=10):
    if isnorm:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        mean = torch.tensor(mean).float().view(3,1,1)
        std = torch.tensor(std).float().view(3,1,1)
        mmax = torch.ones(3,96,96)
        mmin = torch.zeros(3,96,96)
        mmax = ((mmax-mean)/std).cuda()
        mmin = ((mmin-mean)/std).cuda()
        max_epsilon = max_epsilon/(255*0.5)
    else:
        mmax = 255
        mmin = 0
        max_epsilon = float(max_epsilon)
    adversary = fgsm(model,eps=max_epsilon,clip_min=mmin,clip_max=mmax)
    count = 0
    total_correct = 0
    # device = model.device()
    for x,y in train_loader: 
        x = x.cuda()
        y = y.cuda()
        count += len(x)
        ad_ex = adversary.perturb(x,y)
        if not isnorm:
            ad_ex = torch.round(ad_ex)
        z1 = model(ad_ex).argmax(dim=1)
        diff = ad_ex -x 
        total_correct += (z1==y).sum()

        if count >= max_count:
            break
    return total_correct.cpu().numpy()/(count)


def pgd_attack(max_count,model,train_loader,max_epsilon,learning_rate,iters=20,isnorm=False,num_classes=10):
    if isnorm:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        mean = torch.tensor(mean).float().view(3,1,1)
        std = torch.tensor(std).float().view(3,1,1)
        mmax = torch.ones(3,96,96)
        mmin = torch.zeros(3,96,96)
        mmax = ((mmax-mean)/std).cuda()
        mmin = ((mmin-mean)/std).cuda()
        learning_rate = learning_rate /(255*0.5)
        max_epsilon = max_epsilon/(255*0.5)
    else:
        mmax = 255
        mmin = 0
        learning_rate = learning_rate /1.0
        max_epsilon = max_epsilon/1.0
    adversary = pgd(model,eps=max_epsilon,nb_iter=iters,eps_iter=learning_rate,clip_min=mmin,clip_max=mmax,rand_init=False)
    count = 0
    total_correct = 0
    # device = model.device()
    for x,y in train_loader: 
        x = x.cuda()
        y = y.cuda()
        count += len(x)
        ad_ex = adversary.perturb(x,y)
        if not isnorm:
            ad_ex = torch.round(ad_ex)
        z1 = model(ad_ex).argmax(dim=1)
        diff = ad_ex -x 
        total_correct += (z1==y).sum()

        if count >= max_count:
            break
    return total_correct.cpu().numpy()/(count)


def EOT_attack(max_count,model,train_loader,max_epsilon,learning_rate,iters=20,isnorm=False,num_classes=10,num_samples=10):
    if isnorm:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        mean = torch.tensor(mean).float().view(3,1,1)
        std = torch.tensor(std).float().view(3,1,1)
        mmax = torch.ones(3,96,96)
        mmin = torch.zeros(3,96,96)
        mmax = ((mmax-mean)/std).cuda()
        mmin = ((mmin-mean)/std).cuda()
        learning_rate = learning_rate /(255*0.5)
        max_epsilon = max_epsilon/(255*0.5)
        preprocessing = ((0.5, 0.5, 0.5), (0.5, 0.5,0.5))
    else:
        mmax = 255
        mmin = 0
        preprocessing = ((0, 0, 0), (1/255,1/255,1/255))
    adversary = EOT(model,max_epsilon=max_epsilon,steps=iters,preprocessing = preprocessing,step_size=learning_rate,num_samples=num_samples)
    count = 0
    total_correct = 0
    # device = model.device()
    for x,y in train_loader: 
        x = x.cuda()
        y = y.cuda()
        count += len(x)
        ad_ex = adversary.pgd_attack(model,x,y)
        if not isnorm:
            ad_ex = torch.round(ad_ex)
        z1 = model(ad_ex).argmax(dim=1)
        diff = ad_ex -x 
        total_correct += (z1==y).sum()

        if count >= max_count:
            break
    return total_correct.cpu().numpy()/(count)




def simple_black(max_count,model,train_loader,max_epsilon,learning_rate,iters=20,isnorm=False,num_classes=10):
    if isnorm:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        mean = torch.tensor(mean).float().view(3,1,1)
        std = torch.tensor(std).float().view(3,1,1)
        mmax = torch.ones(3,96,96)
        mmin = torch.zeros(3,96,96)
        mmax = ((mmax-mean)/std).cuda()
        mmin = ((mmin-mean)/std).cuda()
        learning_rate = learning_rate /(255*0.5)
        max_epsilon = max_epsilon/(255*0.5)
        preprocessing = ((0.5, 0.5, 0.5), (0.5, 0.5,0.5))
    else:
        mmax = 255
        mmin = 0
        preprocessing = ((0, 0, 0), (1/255,1/255,1/255))
    adversary = SimpleBlack(model,max_epsilon=max_epsilon,steps=iters,preprocessing = preprocessing,step_size=learning_rate)
    count = 0
    total_correct = 0
    # device = model.device()
    for x,y in train_loader: 
        x = x.cuda()
        y = y.cuda()
        count += len(x)
        ad_ex = adversary(model,x,y)
        if not isnorm:
            ad_ex = torch.round(ad_ex)
        z1 = model(ad_ex).argmax(dim=1)
        diff = ad_ex -x 
        total_correct += (z1==y).sum()

        if count >= max_count:
            break
    return total_correct.cpu().numpy()/(count)



def mifgsm_attack(max_count,model,train_loader,max_epsilon,learning_rate,iters=20,isnorm=False,num_classes=10):
    if isnorm:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        mean = torch.tensor(mean).float().view(3,1,1)
        std = torch.tensor(std).float().view(3,1,1)
        mmax = torch.ones(3,96,96)
        mmin = torch.zeros(3,96,96)
        mmax = ((mmax-mean)/std).cuda()
        mmin = ((mmin-mean)/std).cuda()
        learning_rate = learning_rate /(255*0.5)
        max_epsilon = max_epsilon/(255*0.5)
    else:
        mmax = 255
        mmin = 0
    adversary = mifgsm(model,eps=max_epsilon,nb_iter=iters,eps_iter=learning_rate,clip_min=mmin,clip_max=mmax,decay_factor=1.0)
    count = 0
    total_correct = 0
    # device = model.device()
    for x,y in train_loader: 
        x = x.cuda()
        y = y.cuda()
        count += len(x)
        ad_ex = adversary.perturb(x,y)
        if not isnorm:
            ad_ex = torch.round(ad_ex)
        z1 = model(ad_ex).argmax(dim=1)
        diff = ad_ex -x 
        total_correct += (z1==y).sum()

        if count >= max_count:
            break
    return total_correct.cpu().numpy()/(count)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
"""
    Tools for logging
"""
import logging
import os
def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info('\n------ ******* ------ New Log ------ ******* ------')
    return logger


"""
    Tools for directories
"""
class imgshift():
    def __init__(self,proportion=args.shift_p):
        self.p = proportion
        self.rndH = 0
        self.rndW = 0
        self.alpha = 0
        self.sequence = []
    def round_clamp_bk(self,window,x):
        x += window 
        x = x % (2 * window)
        x -= window 
        return x 
    def round_clamp(self,window,x):
        return min(max(-1 * window, x),window)
    def shift(self,img):
        B,C,H,W = img.shape
        window = int(H*self.p)
        rng = randint(-1 * window, window)
        rng += self.rndH * self.alpha
        # rng = min(max(-1 * window,rng),window)
        rng = self.round_clamp(window,rng)
        rng = int(rng)
        self.sequence.append(rng)
        self.rndH = rng
        img_r = img.clone()
        if rng>0:
            img_r[:,:,rng:,:] = img[:,:,:H-rng,:]
            img_r[:,:,:rng,:] = img[:,:,H-rng:,:]
        else:
            img_r[:,:,:H+rng,:] = img[:,:,-rng:,:]
            img_r[:,:,H+rng:,:] = img[:,:,:-rng,:]
        rng = randint(-1 * window, window)
        rng += self.rndW * self.alpha
        # rng = min(max(-1 * window,rng),window)
        rng = self.round_clamp(window,rng)
        rng = int(rng)
        self.sequence.append(rng)
        self.rndW = rng
        # print(self.sequence)
        img = img_r.clone()
        if rng>0:
            img_r[:,:,:,rng:] = img[:,:,:,:W-rng]
            img_r[:,:,:,:rng] = img[:,:,:,W-rng:]
        else:
            img_r[:,:,:,:W+rng] = img[:,:,:,-rng:]
            img_r[:,:,:,W+rng:] = img[:,:,::,:-rng]
        return img_r
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)



def density_cal(x):
    sums = torch.norm(x,p=2,dim=(1,2,3,4))
    sums = torch.sqrt(sums)
    return sums

def scan_dir(dir, matching, fullPath=False):
    # Scan dir
    # Get all files matching particualr patterns: 8.png, [!._]*.png
    # if fullPath, return dir + matching; otherwise, return matching
    import glob
    file_list = glob.glob(os.path.join(dir, matching))
    if not fullPath:
        file_list = [os.path.split(x)[-1] for x in file_list]
    return file_list


"""
    Tools for data processing
"""
def augment(l):
    mode = np.random.randint(0, 8)
    # print(mode)
    def _augment(img, mode=0):
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(img)
        elif mode == 2:
            return np.rot90(img)
        elif mode == 3:
            return np.flipud(np.rot90(img))
        elif mode == 4:
            return np.rot90(img, k=2)
        elif mode == 5:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 6:
            return np.rot90(img, k=3)
        elif mode == 7:
            return np.flipud(np.rot90(img, k=3))

    return [_augment(_l, mode=mode) for _l in l]

def add_noise_numpy(x, param='.'):
    """ param should be [type, value] """
    if param is not '.':
        noise_type = param[0]
        noise_value = int(param[1])
        
        if noise_type == 'G':
            noises = np.random.normal(loc=0, scale=noise_value, size=x.shape)
        elif noise_type == 'S':
            assert False, 'Please use Guassian Noises.'
        
        x_noise = np.clip(x + (noises / 255), 0, 1)
        return x_noise
    else:
        return x

def add_noise_tensor(x, param=['G',15]):
    """ param should be [type, value] """
    if param is not '.':
        noise_type = param[0]
        noise_value = int(param[1])

        if noise_type == 'G':
            noises = np.random.normal(loc=0, scale=noise_value, size=x.shape)
        elif noise_type == 'S':
            assert False, 'Please use Guassian Noises.'
        x_noise = (x + torch.from_numpy(noises / 255).to(torch.float32))
        x_noise = torch.clamp(x_noise, 0, 1)
        return x_noise
    else:
        return x
