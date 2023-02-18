import torch as t
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import DataLoader


transforms = {
    "colored_mnist": {
        "train": T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(28, scale=(0.9,1.1)),
                T.ColorJitter(hue=0.05, saturation = 0.05),
                T.RandomRotation((-10,10),resample=Image.NEAREST),
                T.ToTensor()
            ]
        ),
        "valid": T.Compose([]),
        "test": T.Compose([])
        },
}



def get_dataset(dataset, data_dir, split, bias):
    if split in ['train','valid']:
        tmp= "%s/%s/%s_bias_%s.pt" %(data_dir,dataset,dataset,str(bias))
        print('%s data: '%(split),tmp)
        data = t.load(tmp)[split]
    elif split == 'test':
        tmp = "%s/%s/%s_test.pt" %(data_dir,dataset,dataset)
        print('%s data: '%(split),tmp)
        data = t.load(tmp)

    transform = transforms[dataset][split]    
    print(transform)

    return loader(data,transform)

class loader(DataLoader):
    def __init__(self,data,transform):
        self.transform = transform
        self.data = data['data'].float()
        self.label = data['label'].long()
        self.b_label = data['b_label']
        self.prob = t.zeros(len(self.label))
        self.prob_on = False

    def __getitem__(self,idx):
        idx = self.idx_sample() if self.prob_on else idx
        if type(self.b_label) == dict:
            return self.transform(self.data[idx]), self.label[idx], -1 , idx
        
        return self.transform(self.data[idx]), self.label[idx], self.b_label[idx], idx
    
    def __len__(self):
        return len(self.label)

    def idx_sample(self):
        return t.clamp(t.sum(t.rand(1)>self.prob), 0, len(self.label)-1 )


    def update_prob(self,prob):
        self.prob = t.cumsum(prob,dim=0)
        
    def prob_sample_on(self):
        self.prob_on = True
    
    def prob_sample_off(self):
        self.prob_on = False

    def cleansing(self,pos, blabel = False):
        self.data = self.data[pos]
        self.label = self.label[pos]
        self.prob = self.prob[pos]
        if type(self.b_label) != dict:
            self.b_label = self.b_label[pos]
        
        