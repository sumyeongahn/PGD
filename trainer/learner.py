from tqdm import tqdm
import numpy as np
import torch as t
import torch.nn as nn

import os
import pickle as pkl
import torch.optim as optim
from torch.utils.data import DataLoader

from module.loss import GeneralizedCELoss
from module.net import *
from module.data import *
from module.avgmeter import *
from module.ema import *
from module.gradient import *
from module.scores import *

from utils.analysis import *
        
class Learner(object):
    def __init__(self,args):
        self.device = t.device(args.device)
        self.args = args
        
        train_dataset=get_dataset(  dataset=args.dataset,
                                    data_dir=args.data_dir,
                                    split='train',
                                    bias=args.bratio)
    
        valid_dataset=get_dataset(  dataset=args.dataset,
                                    data_dir=args.data_dir,
                                    split='valid',
                                    bias=args.bratio)
    
        test_dataset=get_dataset(   dataset=args.dataset,
                                    data_dir=args.data_dir,
                                    split='test',
                                    bias=args.bratio)

        self.train_loader = DataLoader( train_dataset,
                                        batch_size = args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        drop_last=False)

        self.valid_loader = DataLoader( valid_dataset,
                                        batch_size = args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        drop_last=False)
        
        self.test_loader = DataLoader(  test_dataset,
                                        batch_size = args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        drop_last=False)
        
        self.model_b = get_model(args.model, args.num_class).to(self.device)
        self.model_d = get_model(args.model, args.num_class).to(self.device)
        

        if args.opt == 'SGD':
            self.optimizer_b = optim.SGD(
                self.model_b.parameters(),
                lr = args.lr,
                weight_decay = args.weight_decay,
                momentum = args.momentum
            )

            self.optimizer_d = optim.SGD(
                self.model_d.parameters(),
                lr = args.lr,
                weight_decay = args.weight_decay,
                momentum = args.momentum
            )

        elif args.opt == 'Adam':
            self.optimizer_b = optim.Adam(
                self.model_b.parameters(),
                lr = args.lr,
                weight_decay = args.weight_decay
            )


            self.optimizer_d = optim.Adam(
                self.model_d.parameters(),
                lr = args.lr,
                weight_decay = args.weight_decay
            )
        self.step_b = optim.lr_scheduler.StepLR(self.optimizer_b,step_size = self.args.lr_decay_step, gamma=self.args.lr_decay)
        self.step_d = optim.lr_scheduler.StepLR(self.optimizer_d,step_size = self.args.lr_decay_step, gamma=self.args.lr_decay)

        self.loss_b = nn.CrossEntropyLoss(reduction='none')
        self.loss_d = nn.CrossEntropyLoss(reduction='none')

        self.best_loss = np.inf
        self.best_acc = 0
        self.best_epoch = 0

    def reset_meter(self,epoch):
        self.epoch = epoch

        self.b_train_acc    = AvgMeter()
        self.b_train_loss   = AvgMeter()
        self.d_train_acc    = AvgMeter()
        self.d_train_loss   = AvgMeter()
        
        
        self.val_acc      = AvgMeter()
        self.val_loss     = AvgMeter()

        self.test_acc     = AvgMeter()
        self.test_loss    = AvgMeter()


    def save_models(self,option,debias=False,bias=False):
        if debias:
            t.save(self.model_d, self.args.log_dir+'model_d_'+option+'.pt')
        if bias:
            t.save(self.model_b, self.args.log_dir+'model_b_'+option+'.pt')
        

    def load_models(self,option,debias=False,bias=False):
        if debias:
            self.model_d = t.load(self.args.log_dir+'model_d_'+option+'.pt').to(self.device)
        if bias:
            self.model_b = t.load(self.args.log_dir+'model_b_'+option+'.pt').to(self.device)
        

    def evaluate(self,debias=False,bias=False):
        self.load_models('end',debias=debias,bias=bias)
        self.test_acc = AvgMeter()
        self.test_loss = AvgMeter()
        self.test(debias = debias, bias = bias)
        self.args.print('End Los %.4f' %(self.test_loss.avg))
        self.args.print('End Acc %.4f' %(self.test_acc.avg))

        self.load_models('best',debias=debias,bias=bias)
        self.test_acc = AvgMeter()
        self.test_loss = AvgMeter()
        self.test(debias = debias, bias = bias)
        self.args.print('Best Los %.4f' %(self.test_loss.avg))
        self.args.print('Best Acc %.4f' %(self.test_acc.avg))

    def print_result(self, type='both'):
        self.args.print('Epoch - %3d / %3d' %(self.epoch+1, self.args.epochs))
        self.args.print('Valid Best loss - (Epoch: %3d) %.4f // %.4f' %(self.best_epoch+1, self.best_loss, self.best_acc))
        
        if type == 'debias':
            self.args.print('Debiased Train Los %.4f' %(self.d_train_loss.avg))
            self.args.print('Debiased Train Acc %.4f' %(self.d_train_acc.avg))
        elif type == 'bias':
            self.args.print('Biased Train Los %.4f' %(self.b_train_loss.avg))
            self.args.print('Biased Train Acc %.4f' %(self.b_train_acc.avg))
        else:
            self.args.print('Debiased Train Los %.4f' %(self.d_train_loss.avg))
            self.args.print('Debiased Train Acc %.4f' %(self.d_train_acc.avg))

            self.args.print('Biased Train Los %.4f' %(self.b_train_loss.avg))
            self.args.print('Biased Train Acc %.4f' %(self.b_train_acc.avg))
        
    

        self.args.print('Valid Los %.4f' %(self.val_loss.avg))
        self.args.print('Valid Acc %.4f' %(self.val_acc.avg))

        self.args.print('Test Los %.4f' %(self.test_loss.avg))
        self.args.print('Test Acc %.4f' %(self.test_acc.avg))
        
        

    def test(self, debias = False, bias=False):
        self.model_b.eval()
        self.model_d.eval()
        
        for _, data_tuple in enumerate(self.test_loader):
            data = data_tuple[0].to(self.device)
            label = data_tuple[1].to(self.device)
            bias_label = data_tuple[2]
            idx = data_tuple[3]

            if bias and debias:
                feature_d = self.model_d.extract(data)
                feature_b = self.model_b.extract(data)

                feature = t.cat((feature_d,feature_b),dim=1)

                logit_d = self.model_d.predict(feature)
                loss_d = self.loss_d(logit_d,label)

                loss = loss_d.mean().float().detach().cpu()
                acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())
            else:
                if debias:
                    logit_d = self.model_d(data)
                    loss_d = self.loss_d(logit_d, label)
                    loss = loss_d.mean().float().detach().cpu()
                    acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())

                else:            
                    logit_b = self.model_b(data)
                    loss_b = self.loss_b(logit_b, label)
                    loss = loss_b.mean().float().detach().cpu()
                    acc = t.mean((logit_b.max(1)[1] == label).float().detach().cpu())
            
            self.test_loss.update(loss,len(label))
            self.test_acc.update(acc, len(label))
        
        if self.epoch == self.best_epoch:
            self.best_acc = self.test_acc.avg
        
    def validate(self, debias = False, bias = False):
        self.model_b.eval()
        self.model_d.eval()
        
        for _, data_tuple in enumerate(self.valid_loader):
            data = data_tuple[0].to(self.device)
            label = data_tuple[1].to(self.device)
            bias_label = data_tuple[2]
            idx = data_tuple[3]
            
            if bias and debias:
                feature_d = self.model_d.extract(data)
                feature_b = self.model_b.extract(data)

                feature = t.cat((feature_d,feature_b),dim=1)

                logit_d = self.model_d.predict(feature)
                loss_d = self.loss_d(logit_d,label)

                loss = loss_d.mean().float().detach().cpu()
                acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())
                
            else:
                if debias:
                    logit_d = self.model_d(data)
                    loss_d = self.loss_d(logit_d, label)
                    loss = loss_d.mean().float().detach().cpu()

                    acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())
                    
                    
                else:
                    logit_b = self.model_b(data)
                    loss_b = self.loss_b(logit_b, label)
                    loss = loss_b.mean().float().detach().cpu()

                    acc = t.mean((logit_b.max(1)[1] == label).float().detach().cpu())
                    
            self.val_loss.update(loss,len(label))
            self.val_acc.update(acc, len(label))

        if self.val_loss.avg <= self.best_loss:
            self.best_epoch = self.epoch
            self.best_loss = self.val_loss.avg
            self.save_models('best',debias = debias, bias = bias)

        
    def train_ours(self):
        # Initialize
        self.bias_criterion = GeneralizedCELoss(q=0.7)
        self.best_loss = np.inf
        self.best_acc = 0

        # Bias Train
        if not self.args.pretrained_bmodel:
            for epoch in range(self.args.epochs):
                self.model_b.train()
                self.model_d.train()
                self.reset_meter(epoch)
                for _, data_tuple in tqdm(enumerate(self.train_loader)):
                    data = data_tuple[0].to(self.device)
                    label = data_tuple[1].to(self.device)
                    bias_label = data_tuple[2]
                    idx = data_tuple[3]

                    logit_b = self.model_b(data)
                    loss_b = self.bias_criterion(logit_b, label)
                    loss = loss_b.mean()

                    self.optimizer_b.zero_grad()
                    loss.backward()
                    self.optimizer_b.step()

                    acc = t.mean((logit_b.max(1)[1] == label).float().detach().cpu())

                    self.b_train_loss.update(loss,len(label))
                    self.b_train_acc.update(acc, len(label))
                self.validate(bias = True)
                self.test(bias = True)
                self.print_result(type='bias')
                self.step_b.step()
            self.save_models('end',bias=True)
        self.load_models('end',bias=True)



        # Calcuate probability
        self.model_b.eval()
        target = ['fc'] if 'ResNet' in self.args.model else ['fc']
        grad = grad_feat_ext(self.model_b,target, len(self.train_loader.dataset))
        
        blabel = t.zeros(len(self.train_loader.dataset))
        idxorder = t.zeros(len(self.train_loader.dataset))
        start, end = 0,0

        for bidx, data_tuple in tqdm(enumerate(self.train_loader)):
            data = data_tuple[0].to(self.device).requires_grad_(True)
            label = data_tuple[1].to(self.device)
            bias_label = data_tuple[2]
            idx = data_tuple[3]

            logit_b = grad(data)
            loss_b = self.loss_b(logit_b, label)
            for sample_idx in range(len(label)):
                self.optimizer_b.zero_grad()
                loss_b[sample_idx].backward(retain_graph = True)

            end = start + len(label)
            blabel[start:end] = bias_label.detach().cpu()
            idxorder[start:end] = idx.detach().cpu()
            start = end
            
        for name in grad.hook_list.keys():
            if 'fc' in name:
                grad_mat = grad.hook_list[name].data
                break

        norm_scale =self.args.norm_scale
        score = t.norm(grad_mat, p=norm_scale, dim=1, keepdim=False)
        
        # Magnitude
        order = t.argsort(idxorder)
        label = label[order]
        blabel = blabel[order]
        mag = t.clamp(score[order],min=1e-8)
        inv_mag = 1./mag
        norm_inv_mag = inv_mag / t.sum(inv_mag)
        mag_prob = 1./norm_inv_mag / t.sum(1./norm_inv_mag)


        self.train_loader.dataset.update_prob(mag_prob)
        self.train_loader.dataset.prob_sample_on()
        
        if self.args.save_stats:
            gradient_analysis(label, blabel, mag, self.args.print)
            prob_analysis(label,blabel,mag_prob ,self.args.print)
        
        del(grad_mat)

        self.model_d.load_state_dict(self.model_b.state_dict())

        if not self.args.pretrained_dmodel:
            # Re-initialize
            self.best_loss = np.inf
            self.best_acc = 0
            # Debiased train
            for epoch in range(self.args.epochs):
                self.model_b.train()
                self.model_d.train()
                self.reset_meter(epoch)
                for _, data_tuple in tqdm(enumerate(self.train_loader)):
                    data = data_tuple[0].to(self.device)
                    label = data_tuple[1].to(self.device)
                    bias_label = data_tuple[2]
                    idx = data_tuple[3]

                    logit_d = self.model_d(data)
                    loss_d = self.loss_d(logit_d, label)
                    loss = loss_d.mean()

                    self.optimizer_d.zero_grad()
                    loss.backward()
                    self.optimizer_d.step()

                    acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())

                    self.d_train_loss.update(loss,len(label))
                    self.d_train_acc.update(acc, len(label))
                self.validate(debias=True)
                self.test(debias=True)
                self.print_result(type='debias')
                self.step_d.step()
            self.save_models('end',debias=True)
        self.load_models('best',debias=True)


