"""PyTorch Carlini and Wagner L2 attack algorithm.

Based on paper by Carlini & Wagner, https://arxiv.org/abs/1608.04644 and a reference implementation at
https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
"""
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from numba import jit
from torch import optim
from torch import autograd
from .helpers import *
import pdb
class nes_grad:
    def __init__(self,variance,iters, targeted = False):
        self.variance = variance
        self.iters = iters
        self.targeted = targeted
    def _loss(self,model,img,target):
        if not self.targeted:
            loss = model(img)[0][target]
        else:
            loss = -1 * model(img)[0][target]
        return loss

    def run(self,model,img,target,num_classes,targeted = False):
        N = img.size()[-1]
        grad = torch.zeros_like(img)
        # mask = torch.zeros_like(img)
        # mask = mask.view(-1)
        # mask[indice] = 1
        # mask = mask.view(img.size())
        with torch.no_grad():
            for i in range(self.iters):
                exp_noise = torch.randn_like(img)/(N **0.5)  
                exp_noise = torch.sign(exp_noise)
                P1 = self._loss(model,img + self.variance * exp_noise,target)
                P2 = self._loss(model,img - self.variance * exp_noise,target)
                exp_deriv = (P1 - P2) /self.variance
                grad += exp_deriv * exp_noise
        return grad.detach()
def cw_loss(output, target,num_classes=10,targeted =False):
    num_classes = num_classes
    target_onehot = torch.zeros(target.size() + (num_classes,))
    if torch.cuda.is_available():
        target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = torch.autograd.Variable(target_onehot, requires_grad = False)
    
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    if not targeted:
        loss = torch.clamp(torch.log(real + 1e-30) - torch.log(other + 1e-30), min = 0.)
    else:
        loss = torch.clamp(torch.log(other + 1e-30) - torch.log(real + 1e-30), min = 0.)

    loss = torch.sum(0.5 * loss) / len(target)
    return loss
def normal_loss(output,target,targeted = False):
    if not targeted:
        loss = torch.nn.CrossEntropyLoss()
        return loss(output,target) * -1
    else:
        loss = torch.nn.CrossEntropyLoss()
        return loss(output,target) * 1
def cw_grad(model,x,target_class,num_classes=10,targeted = False):
    x.requires_grad_()
    model.zero_grad()
    output = model(x)
    loss = cw_loss(F.softmax(output,dim=1),target_class,num_classes,targeted=targeted)
    grad = torch.autograd.grad(loss, x)[0] 
    return grad
def pgd_grad(model,x,target_class,num_classes=10,targeted = False):
    x.requires_grad_()
    model.zero_grad()
    output = model(x)
    loss = normal_loss(F.softmax(output,dim=1),target_class,targeted = targeted)
    grad = torch.autograd.grad(loss, x)[0] 
    # grad = padding_layer(grad,246)
    # grad = imgshift(grad)
    return grad
class generate_gradient:

    def __init__(self, device, targeted = False, classes = 10, debug = False):
        self.debug = debug
        self.targeted = targeted # false
        self.num_classes = classes 
        self.confidence = 0  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        
        self.use_log = True
        self.batch_size = 128
        self.device = device     
        self.use_importance = True
        self.constant = 0.5
        
    def shift_target(self,target):
        label = torch.argmax(target)
        label = (label+1) % 10
        label = label.reshape(1)
        target_onehot = torch.zeros(label.size() + (self.num_classes,))
        if torch.cuda.is_available():
            target_onehot = target_onehot.to(self.device)
        target_onehot.scatter_(1, label.unsqueeze(1), 1.)
        target_var = autograd.Variable(target_onehot, requires_grad = False) 
        return target_var
        
    def _loss(self, output, target, dist, constant):
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            # target_label = self.shift_target(target)
            # other = (target_label * output).sum(1)
            if self.use_log:
                loss1 = torch.clamp(torch.log(other + 1e-30) - torch.log(real + 1e-30), min = 0.)
            else:
                loss1 = torch.clamp(other - real + self.confidence, min = 0.)  # equiv to max(..., 0.)
        else:
            if self.use_log:
                loss1 = torch.clamp(torch.log(real + 1e-30) - torch.log(other+ 1e-30), min = 0.)
            else:
                loss1 = torch.clamp(real - other + self.confidence, min = 0.)  # equiv to max(..., 0.)
        loss1 = constant * loss1

        # loss2 = dist.squeeze(1)
        # print('loss1 and loss2 is:', loss1, loss2)
        # loss = loss1 + loss2
        # pdb.set_trace()
        loss = loss1
        loss2 = dist
        return loss, loss1, loss2
    
    def run(self, model, img, target, indice):
        
        batch, c, h, w = img.size()
        var_size = c * h * w
        var_list = np.array(range(0, var_size), dtype = np.int32)
        sample_prob = np.ones(var_size, dtype = np.float32) / var_size
        # sample_prob = mask
  
        ori_img = img
            
        grad = torch.zeros(self.batch_size, dtype = torch.float32)
        modifier = torch.zeros_like(img, dtype = torch.float32)
        
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if torch.cuda.is_available():
            target_onehot = target_onehot.to(self.device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = autograd.Variable(target_onehot, requires_grad = False) 
        
        img_var = img.repeat(self.batch_size * 2 + 1, 1, 1, 1)
        if self.use_importance:
            var_indice = np.random.choice(var_list.size, self.batch_size, replace = False, p = sample_prob)
        else:
            var_indice = np.random.choice(var_list.size, self.batch_size, replace = False)
        
        indice = var_list[var_indice]
        #print('indice', indice)
        for i in range(self.batch_size):
            img_var[i*2 + 1].reshape(-1)[indice[i]] += 0.0001
            img_var[i*2 + 2].reshape(-1)[indice[i]] -= 0.0001
        
        output = F.softmax(model(img_var), dim = 1)
        # output = model(img_var)
        dist = l2_dist(img_var, ori_img, keepdim = True).squeeze(2).squeeze(2)
        # dist = 0
        loss, loss1, loss2 = self._loss(output.data, target_var, dist, self.constant)
        for i in range(self.batch_size):
            grad[i] = (loss[i * 2 + 1] - loss[i * 2 + 2]) / 0.0002
        
        modifier.reshape(-1)[indice] = grad.to(self.device)
        # pdb.set_trace()
        return modifier.cpu().numpy()[0], indice
    
