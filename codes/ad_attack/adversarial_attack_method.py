import os
import sys
import torch
import copy
import numpy as np
import random
import time
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pdb
import progressbar

from torch.utils.data import DataLoader
from attk.cw_black import BlackBoxL2
from data import load_data
from PIL import Image
from attk.generate_gradient import generate_gradient

from utils import Logger

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from options import args
class Mod(nn.Module):
    def __init__(self,model):
        super(Mod, self).__init__() 

        self.net = model
    def forward(self,x):
        return self.net(x)[0]


def adversarial_attack(count,model,test_loader,epsilon,learning_rate,maxiter,isnorm=False,num_classes=10,device=torch.device("cuda"),is_tiv=False,issave = False,istarget=False,save_path = None,iszoo=False,isstl10=False):

    # prepare process bar 
    bar = progressbar.ProgressBar(maxval=count, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    if isnorm:
        epsilon /= 255
        learning_rate /= 255
    # mod to store tiv model 
    if is_tiv:
        model = Mod(model)
    generate_grad = generate_gradient(
            device,
            targeted = False,
            )
    attack = BlackBoxL2(
            epsilon,
            learning_rate,
            targeted = istarget,
            max_steps = maxiter + 1,
            search_steps = 1,
            cuda = True,
            debug=False,
            isnorm=isnorm,
            num_classes= num_classes,
            iszoo = iszoo,
            isstl10 = isstl10
            )
    
    img_no = 0
    total_success = 0
    l2_total = 0.0
    avg_step = 0
    avg_time = 0
    avg_que = 0
    meta_model = None
    indice_attack = np.arange(count) 


    # add random selection to pick test samples

    for i, (img, target) in enumerate(test_loader):
        if i > count + 1:
            break
        if not i in indice_attack:
            continue 
        bar.update(i+1)
        img, target = img.to(device), target.to(device)
        pred_logit = model(img)
        
        # print('Predicted logits', pred_logit.data[0], '\nProbability', F.softmax(pred_logit, dim = 1).data[0])
        pred_label = pred_logit.argmax(dim=1)
        # print('The original label before attack', pred_label.item())
        if pred_label != target:
            continue

        if istarget :
            random_target = np.random.randint(0,999)
            if target == random_target:
                random_target = (random_target + 100) % 1000
            target = torch.tensor([random_target]).long()
        img, target = img.to(device), target.to(device)
            
        img_no += 1
        timestart = time.time()
        
        
        #######################################################################

        meta_model_copy = copy.deepcopy(meta_model)
    


        #######################################################################
        
        
            
        
        queries = 0
        adv, const, first_step = attack.run(model, meta_model_copy, img, target, i)
        timeend = time.time()

        if len(adv.shape) == 3:
            adv = adv.reshape((1,) + adv.shape)
        if issave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_img(adv,i,save_path)
        adv = torch.from_numpy(adv).permute(0, 3, 1, 2).cuda()
        #print('adv min and max', torch.min(adv).item(), torch.max(adv).item(), '', torch.min(img).item(), torch.max(img).item())
        diff = (adv-img).cpu().numpy()
        l2_distortion = np.sum(diff**2)**.5
        # print('L2 distortion', l2_distortion)
        
        adv_pred_logit = model(adv)
        adv_pred_label = adv_pred_logit.argmax(dim = 1)
        # print('The label after attack', adv_pred_label.item())
        
        success = False
        # if adv_pred_label != pred_label:
        if adv_pred_label != target:
            success = True
        if success:
            total_success += 1
            l2_total += l2_distortion
            avg_step += first_step
            avg_time += timeend - timestart
        if total_success == 0:
            pass
    print("\n")
    asr = total_success / float(img_no)
    return 1 - asr

def save_img(img,i,save_path):
    import os.path as osp
    from skimage.io import imsave
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = osp.join(save_path,str(i)+".png")
    image = conv_norm(img)
    imsave(save_name,image)
def conv_norm(img):
    img = img.reshape(224,224,3)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    img = img * std + mean
    img = img * 255
    img = img.astype(np.uint8)
    return img

if __name__ == "__main__":

    USE_DEVICE = torch.cuda.is_available()
    device = torch.device('cuda' if USE_DEVICE else 'cpu')

