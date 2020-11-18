import os, sys, random, time
sys.path.append("../ad_attack")
sys.path.append("../black_box_attack")
from black_attack_method import black_attack,simba_attack
from adversarial_attack_method import adversarial_attack

import numpy as np
import time 

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

def cos_sim(model, dataset_loader1,dataset_loader2):
    total_correct = 0
    r1 = np.zeros((100,64))
    r2 = np.zeros((100,64))
    count = 0
    for x, y in dataset_loader1:
        x = x.cuda(args.device_ids[0])
        y = one_hot(np.array(y.numpy()), 10)
        r1[count] = model(x).cpu().detach().numpy()
        count +=1

    count = 0
    for x, y in dataset_loader2:
        x = x.cuda(args.device_ids[0])
        y = one_hot(np.array(y.numpy()), 10)
        r2[count] = model(x).cpu().detach().numpy()
        count +=1




    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(r1,r2)


    return sim


def accuracy(model, dataset_loader,max_num=10000):
    total_correct = 0
    count = 0
    total_number = 0
    current_time = time.time()
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
    duration = time.time() - current_time
    print("used {}".format(duration))
    return total_correct / total_number

def accuracy_withRef(model, RefDL, PertbDL):
    target_class = np.array([])
    pred_class_ref = np.array([])
    pred_class_pertb = np.array([])
    for x, y in RefDL:
        x = x.cuda(args.device_ids[0])
        pred_class_ref = np.concatenate((pred_class_ref, np.argmax(model(x).cpu().detach().numpy(), axis=1)), axis=None)

    for x, y in PertbDL:
        x = x.cuda(args.device_ids[0])
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.concatenate((target_class, np.argmax(y, axis=1)), axis=None)
        pred_class_pertb = np.concatenate((pred_class_pertb, np.argmax(model(x).cpu().detach().numpy(), axis=1)), axis=None)

    accu_ref =  np.sum(pred_class_ref == target_class) / len(target_class)
    accu_pertb_target = np.sum(pred_class_pertb == target_class)/len(target_class)
    accu_pertb_ref = np.sum((pred_class_ref == target_class) & (pred_class_pertb == target_class)) / np.sum(pred_class_ref == target_class)
    return accu_ref, accu_pertb_target, accu_pertb_ref

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
        model = embed_resnet(edsr,model,random_pooling=args.israndom_pooling,gaus_noise = args.isgaus_noise)







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
        val_loader_fgsm = get_defense_loaders("../../data/fullimagenet/fgsm8_resnet50",isNorm=False,batch_size=32)






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

        # val_acc_n = adversarial_attack(500,model,val_loader,8,8,1,issave=False,save_path = "../../data/edsr_test/pgd8")



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
                # if count <= iter_count * iters_per_epoch:
                    # continue
                # if (count - iter_count * iters_per_epoch) >= iters_per_epoch:
                    # break
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
            # val_acc_n = accuracy(model,val_loader_fgsm)


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

            # logger.info(
            #     "Epoch {:04d}/{:04d} | Time {:.3f}s | "
            #     "Train acc {:.4f} | Val Acc {:.4f}| Best epoch @ {:04d} with Acc {:.4f} | lr: {:.6f}".format(
            #         epoch, args.end_epoch, time.time()-tic, train_eval_acc, val_acc, best_epoch['epoch'], 
            #         best_epoch['acc'], optimizer.state_dict()['param_groups'][0]['lr'])
            #     )
    else:
        #change dict name function 
        def change_key_name(key):
            # key = key.replace("Downsampling.3","Downsampling.4")
            # key = key.replace("FeatureExtraction.2","FeatureExtraction.3")
            # key = key.replace("FeatureExtraction.1","FeatureExtraction.2")
            # key = key.replace("FeatureExtraction.0","FeatureExtraction.1")
            # key = key.replace("FeatureExtraction.1","FeatureExtraction.3")
            key = key.replace("module.basic_net","module")
            # print(key)
            return key
        def change_dict(dict):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in dict.items():
                new_key = change_key_name(key)
                new_state_dict[new_key] = value
            return new_state_dict
        assert args.isTrain == False
        print('===> Testing ...')
        # Add logs
        save_dir = os.path.join(args.dir_logging); makedirs(save_dir)
        logger = get_logger(logpath=os.path.join(save_dir,args.logging_file))
        logger.info(os.path.abspath(__file__))
        # for arg in vars(args):
        #     logger.info('{}: {}'.format(arg, getattr(args, arg)))
        logger.info('{}: {}'.format('dir_model', getattr(args, 'dir_model')))
        logger.info('{}: {}'.format('noise_level', getattr(args, 'noise_level')))
        # Building a model
        model = get_resnet()
        model.eval()

        resmodel = deepcopy(model)
        resmodel.eval()




        edsr = get_EDSR()
        model = embed_resnet(edsr,model,random_pooling=args.israndom_pooling,gaus_noise = args.isgaus_noise)



        if torch.cuda.is_available():
            model = model.cuda(args.device_ids[0])
        if len(args.device_ids) > 1:
            model = nn.DataParallel(model, args.device_ids)
        state_dict = torch.load(args.dir_model).state_dict()
        # modification of state_dict
        state_dict = change_dict(state_dict)
        model.load_state_dict(state_dict)

 

        # Construct datasets
        path = "../../data/fullimagenet/test_resnet50_101" 
        # path = "../../data/fullimagenet/pgd_16_40_resnet50" 

        val_loader = get_defense_loaders(path,isNorm=False,batch_size=1)
        val_loader_128 = get_defense_loaders(path,isNorm=False,batch_size=32)
        val_loader_fgsm = get_defense_loaders("../../data/fullimagenet/fgsm8_resnet50",isNorm=False,batch_size=8)


        # Testing
        model.eval()
        acc = 0 if False else accuracy(model,val_loader_128)
        # acc = 0 if False else accuracy(resmodel,val_loader)
        # acc2 = 0 if False else adversarial_attack(200,model,val_loader,8,8,1,isnorm=False,num_classes=1000,istarget=False)

        acc2 = 0 if True else fgsm_attack(200,model,val_loader_128,8,isnorm=False)
        acc3 = 0 if True else  pgd_attack(201,model,val_loader_128,8,2,40,isnorm=False)
        # acc3 = 0 if True else  adversarial_attack(101,model,val_loader,16,2,40,isnorm=False,num_classes=1000,istarget=True)
        acc4 = 0 if True else adversarial_attack(53,model,val_loader,8,2,160,isnorm=False,num_classes=1000,istarget=False,iszoo=True)
        acc5 = 0 if True else black_attack(101,model,val_loader,8,1,40,isnorm=False,num_classes=1000,istarget=False)
        # acc2 = 0 if False else simba_attack(50,model,val_loader,8,8,1,isnorm=False,num_classes=10,istarget=False)
        # acc2 = accuracy(model,val_loader_128)
        # acc2 = 0
        acc6 = 0 if True else CW_attack(101,model,val_loader_128,1,iters = 40,isnorm=False)
        acc7 = 0 if True else mifgsm_attack(101,model,val_loader_128,16,1,40)
        acc8 =  0 if True else deepfool(101,model,val_loader_128,8,20)
        acc9 = 0 if True else ead_attack(51,model,val_loader_128,1,iters = 40,isnorm=False)
        acc10 =  0 if True else simple_black(101,model,val_loader_128,32,1,iters =5000,isnorm=False,num_classes=1000)
        acc11 =  0 if True else EOT_attack(101,model,val_loader_128,8,2,iters =10,isnorm=False,num_samples=10,num_classes=1000)
        acc12 =  0 if True else rays_attack(101,model,val_loader_128,16,8,iters =1000,isnorm=True,num_classes=1000)
        acc13 = 0 if False else EOT_CW_attack(51,model,val_loader,1,iters = 40,isnorm=False,num_samples=40)
        logger.info(
                "=== >> Test Acc: {:.5f}, FGSM Acc: {:.5f}, pgd: {:.5f} zoo {:.5f},nes {:.5f},cw {:.5f},mifgsm {:.5f}, deepfool {:.5f}, ead {:.5f} simple{:.5f} EOT{:.5f} rays{:.5f} eotcw{:.5f}".format(acc,acc2,acc3,acc4,acc5,acc6,acc7,acc8,acc9,acc10,acc11,acc12,acc13)
            )
