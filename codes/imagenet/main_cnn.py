import os, sys, random, time
sys.path.append("../ad_attack")
sys.path.append("../black_box_attack")
from black_attack_method import black_attack,simba_attack
from adversarial_attack_method import adversarial_attack

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from options import args
from MiscTools import *
from dataloader import get_ImgNet_loaders,get_defense_loaders

from model.edsr import EDSR,embed_resnet
from copy import deepcopy

from torchvision import datasets
import torchvision.models as models
import torchvision

if args.isRandom == False:
    random.seed(0)
    np.random.seed(seed=0)
    torch.manual_seed(0)

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def get_resnet():
    resnet50 = models.resnet50(num_classes=1000, pretrained='imagenet').cuda()
    return resnet50

def get_EDSR():
    edsr = EDSR(args)
    save_path = "../../checkpoints/edsr/edsr_baseline_x2-1bc95232.pt"
    load_stat = torch.load(save_path)
    edsr.load_state_dict(load_stat)
    return edsr

def accuracy(model, dataset_loader,max_num=10000):
    total_correct = 0
    count = 0
    total_number = 0
    for x, y in dataset_loader:
        count += 1 
        total_number += len(x)
        if count >= max_num:
            break
        x = x.cuda(args.device_ids[0])
        y = one_hot(np.array(y.numpy()), 1000)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / total_number

if __name__ == '__main__':
    if args.isTrain:
        print('===> Training a model ...')
        # Add logs
        save_dir = os.path.join(args.dir_logging, args.exp_name); makedirs(save_dir)
        logger = get_logger(logpath=os.path.join(save_dir,'_train_result.txt'))
        logger.info(os.path.abspath(__file__))
        for arg in vars(args):
            logger.info('{}: {}'.format(arg, getattr(args, arg)))

        # Build model
        model = get_resnet()
        model.eval()

        resmodel = deepcopy(model)
        resmodel.eval()

        edsr = get_EDSR()
        model = embed_resnet(edsr,model,RdmDU=args.isRdmDU,gaus_noise = args.isgaus_noise)

        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model = model.cuda(args.device_ids[0])
            criterion = criterion.cuda(args.device_ids[0])
        if len(args.device_ids) > 1:
            model = nn.DataParallel(model, args.device_ids)

        optimizer = optim.SGD(model.module.edsr_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

        if not args.resume:
            logger.info('---- Model ----')
            logger.info(model)
            logger.info('Number of parameters: {}'.format(count_parameters(model))) 

        # Construct datasets
        train_loader, val_loader = get_ImgNet_loaders(
            isTrain=True, isNorm=False,batch_size=32, test_batch_size=1)

        path = "../../data/fullimagenet/test_resnet50_101" 
        val_loader = get_defense_loaders(path,isNorm=False,batch_size=1)
        val_loader_128 = get_defense_loaders(path,isNorm=False,batch_size=32)






        # Training
        logger.info('---- Training ----')
        # best_epoch = {'epoch':0, 'acc':0}
        best_epoch = {'epoch':0, 'acc':0, 'acc_mix':0}
            # Resume model
        if args.resume:
            print("=> loading checkpoint '{}'".format(save_dir))
            checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth'))
            args.start_epoch = checkpoint['epoch']+1
            best_epoch = checkpoint['best_epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(save_dir, checkpoint['epoch']))

        iter_count = 0
        iters_per_epoch = 500
        reset_number = len(train_loader) // iters_per_epoch - 1
        for epoch in range(args.start_epoch, args.end_epoch):
            tic = time.time()
            scheduler.step(epoch)
            epoch_loss = []

            model.train()
            #control batch length
            count = 0
            iter_count += 1 
            iter_count %= reset_number 
            loss = 0
            for _, batch_tr in enumerate(train_loader):
                count += 1
                if count >= 500:
                    break

                optimizer.zero_grad()
                loss = criterion(model(batch_tr[0].cuda(args.device_ids[0])), batch_tr[1].cuda(args.device_ids[0]))
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.detach().cpu().numpy())
            # evaluation
            model.eval()
            with torch.no_grad():
                train_eval_acc = 0
                val_acc = 0
                if epoch%5==0:
                    val_acc = accuracy(model,val_loader_128)
            val_acc_n = adversarial_attack(200,model,val_loader,8,8,1,isnorm=False,num_classes=1000,istarget=True)

            # logging
            if val_acc <= 1:
                if val_acc >= best_epoch['acc']:
                    best_epoch['epoch'] = epoch
                    best_epoch['acc'] = val_acc
                    best_epoch['acc_mix'] = val_acc*10+val_acc_n
                    torch.save(model, os.path.join(save_dir, 'model_best.pth'))
            else:
                if val_acc*10+val_acc_n >= best_epoch['acc_mix']:
                    best_epoch['epoch'] = epoch
                    best_epoch['acc'] = val_acc
                    best_epoch['acc_mix'] = val_acc*10+val_acc_n
                    torch.save(model, os.path.join(save_dir, 'model_best.pth'))

            # if val_acc >= best_epoch['acc']:
            #     best_epoch['epoch'] = epoch
            #     best_epoch['acc'] = val_acc
            #     torch.save(model, os.path.join(save_dir, 'model_best.pth'))

            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_epoch': best_epoch,
                'optimizer': optimizer.state_dict()
                }
            torch.save(checkpoint,os.path.join(save_dir, 'checkpoint.pth'))
            torch.save(model, os.path.join(save_dir, 'model_{}.pth'.format(epoch)))

            logger.info(
                "Epoch {:04d}/{:04d} | Time {:.3f}s | "
                "Train Acc {:.4f} | Val Acc {:.4f}, Val_N Acc {:.4f} |"
                "Best epoch @ {:04d} with Acc {:.4f} | lr: {:.6f}".format(
                    epoch, args.end_epoch, time.time()-tic, 
                    train_eval_acc, val_acc, val_acc_n, 
                    best_epoch['epoch'], best_epoch['acc'], optimizer.state_dict()['param_groups'][0]['lr'])
                )
    else:
        assert args.isTrain == False
        print('===> Testing ...')
        # Add logs
        save_dir = os.path.join(args.dir_logging); makedirs(save_dir)
        logger = get_logger(logpath=os.path.join(save_dir,args.logging_file))
        logger.info(os.path.abspath(__file__))
        logger.info('{}: {}'.format('dir_model', getattr(args, 'dir_model')))
        logger.info('{}: {}'.format('noise_level', getattr(args, 'noise_level')))
        # Building a model
        model = get_resnet()
        model.eval()

        edsr = get_EDSR()
        model = embed_resnet(edsr,model,RdmDU=args.isRdmDU)

        if torch.cuda.is_available():
            model = model.cuda(args.device_ids[0])
        if len(args.device_ids) > 1:
            model = nn.DataParallel(model, args.device_ids)
        state_dict = torch.load(args.dir_model).state_dict()
        model.load_state_dict(state_dict)

 

        # Construct datasets
        path = "../../data/fullimagenet/test_resnet50_101" 

        val_loader = get_defense_loaders(path,isNorm=False,batch_size=1)
        val_loader_128 = get_defense_loaders(path,isNorm=False,batch_size=32)

        NUM = args.robustness_evaluation_number
        # Testing
        model.eval()
        #clean
        acc = 0 if False else accuracy(model,val_loader_128)
        #FGSM
        acc2 = 0 if False else adversarial_attack(NUM,model,val_loader,8,8,1,isnorm=False,num_classes=1000,istarget=False)
        #PGD
        acc3 = 0 if False else  adversarial_attack(NUM,model,val_loader,16,1,40,isnorm=False,num_classes=1000,istarget=False)
        #zoo
        acc4 = 0 if True else adversarial_attack(NUM,model,val_loader,8,2,160,isnorm=False,num_classes=1000,istarget=False,iszoo=True)
        #Nes
        acc5 = 0 if True else black_attack(NUM,model,val_loader,8,1,40,isnorm=False,num_classes=1000,istarget=False)
        #CW
        acc6 = 0 if False else CW_attack(NUM,model,val_loader_128,1,iters = 40,isnorm=False)
        #deep fool
        acc8 =  0 if False else deepfool(NUM,model,val_loader_128,8,40)
        #ead_attack
        acc9 = 0 if True else ead_attack(NUM,model,val_loader_128,1,iters = 40,isnorm=False)
        #EOT FGSM
        acc10 =  0 if False else EOT_attack(NUM,model,val_loader_128,8,8,iters =1,isnorm=False,num_samples=40,num_classes=1000)
        #EOT PGD
        acc11 =  0 if False else EOT_attack(NUM,model,val_loader_128,8,2,iters =40,isnorm=False,num_samples=40,num_classes=1000)
        #Rays
        acc12 =  0 if False else rays_attack(NUM,model,val_loader_128,16,8,iters =1000,isnorm=True,num_classes=1000)
        #EOT CW
        acc13 = 0 if False else EOT_CW_attack(51,model,val_loader,1,iters = 40,isnorm=False,num_samples=40)

        logger.info(
                "=== >> Test Acc: {:.5f}, FGSM Acc: {:.5f}, pgd: {:.5f} zoo {:.5f},nes {:.5f},cw {:.5f}, deepfool {:.5f}, ead {:.5f} EOT FGSM {:.5f}, EOT PGD {:.5f},rays {:.5f},EOT CW{:.5f} ".format(acc,acc2,acc3,acc4,acc5,acc6,acc8,acc9,acc10,acc11,acc12,acc13)
            )
