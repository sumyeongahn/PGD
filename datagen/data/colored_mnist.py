import numpy as np
import torch as t
import torchvision.transforms as tr
from torchvision.datasets import MNIST
from utils import *

color_set = t.tensor([  [0.8627451,     0.07843137,     0.23529412 ],   #0
                        [0,             0.50196078,     0.50196078 ],   #1
                        [0.99215686,    0.91372549,     0.0627451  ],   #2
                        [0,             0.58431373,     0.71372549 ],   #3
                        [0.929411765,   0.568627451,    0.129411765],   #4
                        [0.568627451,   0.117647059,    0.737254902],   #5
                        [0.274509804,   0.941176471,    0.941176471],   #6
                        [0.980392157,   0.77254902,     0.733333333],   #7
                        [0.823529412,   0.960784314,    0.235294118],   #8
                        [0.501960784,   0,              0          ]])  #9


        


def biasing(dataset, b_ratio):
    _data = dataset.data
    label = dataset.targets
    b_label = t.zeros_like(label)
    _dev = 1e-4
    
    midx = t.rand(len(label)) < b_ratio
    for idx in range(len(label)):
        if (idx+1) % int(0.1*len(label)) == 0:
            print("%3d / %3d Done..."%(idx+1, len(label)))
        if not midx[idx]:
            b_label[idx] = label[idx]
        else:
            while(True):
                rand_b_label = t.randint(0,10,(1,))
                if rand_b_label != label[idx]:
                    break
            b_label[idx] = rand_b_label
    
    dev = t.normal(0,_dev, (len(label),3))
    data = t.zeros((len(label), 3, 28, 28))
    for idx in range(len(label)):
        color = color_set[b_label[idx]] + dev[idx]
        data[idx] = t.clamp((_data[idx].unsqueeze(2).repeat(1,1,3).float()*color)/255.,0.,1.).permute((2,0,1))
    
    return data, label, b_label


def colored_mnist_gen(args):
    train_valid_split = 0.9
    
    mn = MNIST(args.data_storage,train = True, download = True)
    for b_ratio in args.bias_ratio:
        ret,train,valid = {},{},{}
        data, label, b_label = biasing(mn,b_ratio)
        train['data'] = data[:int(len(label)*train_valid_split)]
        train['label'] = label[:int(len(label)*train_valid_split)]
        train['b_label'] = b_label[:int(len(label)*train_valid_split)]
        valid['data'] = data[int(len(label)*train_valid_split):]
        valid['label'] = label[int(len(label)*train_valid_split):]
        valid['b_label'] = b_label[int(len(label)*train_valid_split):]

        ret['train'] = train
        ret['valid'] = valid
        
        data_name = args.data+'_bias_'+str(b_ratio)
        save_data(ret, args.save_dir+data_name)



    mn = MNIST(args.data_storage,train = False, download = True)
    ret = {}
    data, label, b_label = biasing(mn,0.9)
    ret['data'] = data
    ret['label'] = label
    ret['b_label'] = b_label

    data_name = args.data + '_test'
    save_data(ret, args.save_dir+data_name)


