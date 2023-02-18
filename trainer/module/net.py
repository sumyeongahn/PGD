import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.models as models

def get_model(model_tag, num_classes):
    if model_tag == "CONV":
        model =  CONV(num_classes=num_classes)
    else:
        raise NotImplementedError

    return model




class CONV(nn.Module):
    def __init__(self, num_classes = 10):
        super(CONV, self).__init__()
        self.conv1 = nn.Conv2d(3,8,4,1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.avgpool1 = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(8,32,4,1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.avgpool2 = nn.AvgPool2d(2,2)
        self.conv3 = nn.Conv2d(32,64,4,1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout()
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def extract(self, x):
        x = self.conv1(x)     
        x = self.bn1(x)     
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)   
        x = self.dropout2(x)
        x = self.avgpool2(x)
        x = self.conv3(x)   
        x = self.relu3(x)   
        x = self.bn3(x)     
        x = self.dropout3(x)
        x = self.avgpool3(x)
        
        return x

    def predict(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def forward(self, x, mode=None, return_feat=False):
        feat = x = self.extract(x)
        x = x.view(x.size(0),-1)
        final_x = self.predict(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x

