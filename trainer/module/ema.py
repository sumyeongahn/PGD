import io
import torch as t
import numpy as np
import torch.nn as nn

class EMA:
    def __init__(self, label, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.parameter = t.zeros(label.size(0)).cuda()
        self.updated = t.zeros(label.size(0)).cuda()
        
    def update(self, data, index):
        self.parameter[index] = self.alpha * self.parameter[index] + (1-self.alpha*self.updated[index]) * data
        self.updated[index] = 1
        
    def max_loss(self, label):
        label_index = t.where(self.label == label)[0]
        return self.parameter[label_index].max()