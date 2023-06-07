import numpy as np
import torch as t
import os
import argparse
import random

from learner import Learner
from module.logger import *

t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mitigating Dataset Bias Using Per-sample gradients')

    parser.add_argument('--gpu', help='gpu', default='0', type=str)
    parser.add_argument('--model', help='model', default='MLP', type=str)
    parser.add_argument('--batch_size', help='batch size', default=256, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--lr_decay', help='learning rate decay', default=0.9, type=float)
    parser.add_argument('--weight_decay', help='weight decay', default=0.0, type=float)
    parser.add_argument('--momentum', help='momentum number', default=0.9, type=float)
    
    

    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--num_workers', help='workers number', default=16, type=int)
    parser.add_argument('--exp', help='experiment name', default='debugging', type=str)
    parser.add_argument('--device', help='cuda or cpu', default='cuda', type=str)
    parser.add_argument('--epochs', help='# of epochs', default=100, type=int)
    parser.add_argument('--n_epochs', help='# of epochs', default=100, type=int)
    parser.add_argument('--dataset', help='dataset', default= 'colored_mnist', type=str)
    parser.add_argument('--bratio', help='minorty percentage', default= 0.01, type=float)
    parser.add_argument('--use_lr_decay', action='store_true', help='whether to use learning rate decay')
    parser.add_argument('--lr_decay_step', help='learning rate decay steps', type=int, default=10000)
    parser.add_argument('--q', help='GCE parameter q', type=float, default=0.7)
    parser.add_argument('--norm_scale', help='Norm scale', type=float, default=0.7)
    parser.add_argument('--algorithm',  help='run algorithm', default='vanilla',    type=str)

    parser.add_argument('--log_dir', help='path for saving model', default='./log/', type=str)
    parser.add_argument('--data_dir', help='path for loading data', default='./dataset/', type=str)
    
    parser.add_argument('--reproduce',  help='Reproduce',       action='store_true')
    parser.add_argument('--pretrained_nmodel',  help='Use pretrained noisy model?',       action='store_true')
    parser.add_argument('--pretrained_bmodel',  help='Use pretrained biased model?',       action='store_true')
    parser.add_argument('--pretrained_dmodel',  help='Use pretrained debiased model?',       action='store_true')
    parser.add_argument('--train',              help='Train?',       action='store_true')
    parser.add_argument('--save_stats',         help='Save stats?',       action='store_true')
    
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    
    # Path
    args.exp = args.exp+str(args.seed)
    args.log_dir = args.log_dir+args.dataset+'/'+args.algorithm+'/'+'bias_'+str(args.bratio)+'/'+args.exp+'/'
    os.makedirs(args.log_dir, exist_ok=True)
    
    
    # Logger
    _logger = logger(args)
    args.print = _logger.critical
    args.write = _logger.debug

    # Reproducibility
    if args.reproduce:
        from utils.pre_conf import *
        args = reproduce(args)
    
    # Random seed
    t.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    t.cuda.manual_seed(args.seed)
    t.cuda.manual_seed_all(args.seed)

    # Logging current configuration
    args.print(args)

    # Trainer setting
    learner = Learner(args)


    
    # train
    if args.train:
        if args.algorithm == 'vanilla':
            learner.train_vanilla()
        
        elif args.algorithm == 'lff':
            learner.train_lff()

        elif args.algorithm == 'rebias':
            learner.train_rebias()

        elif args.algorithm == 'disen':
            learner.train_disen()

        elif args.algorithm == 'jtt':
            learner.train_jtt()
            
        elif args.algorithm == 'ours':
            learner.train_ours()
            

        elif args.algorithm == 'grad_ext':
            learner.grad_ext()

        else:
            print('Wrong algorithm')
            import sys
            sys.exit(0)


    else:
        if args.algorithm == 'vanilla':
            learner.evaluate(debias=False)
        
        elif args.algorithm == 'lff':
            learner.evaluate(debias=True)

        elif args.algorithm == 'rebias':
            learner.evaluate(debias=True)

        elif args.algorithm == 'disen':
            learner.evaluate(debias=True, disen=True)

        elif args.algorithm == 'jtt':
            learner.evaluate(debias=True)
            
        elif args.algorithm == 'ours':
            learner.evaluate(debias=True)

        else:
            print('Wrong algorithm')
            import sys
            sys.exit(0)

            
