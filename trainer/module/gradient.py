import torch as t


def module_type(name):
    return ''.join([i for i in name if not i.isdigit()])

class grad_fn:
    def __init__(self, name, data_len):
        self.name = name
        self.data = None
        self.data_len = data_len
        self.curr = 0
        self.norm = True

    def save_grad(self,grad):
        if self.data == None:
            grad = grad.reshape([1,-1]).detach().cpu()
            self.data_size = grad.shape[-1]
            self.data = t.zeros(self.data_len, self.data_size)
            self.data[self.curr] = grad
            self.curr += 1
        else:
            grad = grad.reshape([1,-1]).detach().cpu()
            self.data[self.curr] = grad
            self.curr += 1
                
class feat_fn:
    def __init__(self, data_len):
        self.data = None
        self.data_len = data_len
        self.start = 0
        self.end = 0

    def save_feat(self, feat_out):
        feat = feat_out.reshape([len(feat_out),-1]).detach().cpu()
        self.end = self.start + len(feat)
        if self.data == None:
            self.data = t.zeros((self.data_len, feat.shape[-1]))
            self.data[self.start:self.end] = feat
            self.start = self.end
        else:
            self.data[self.start:self.end] = feat
            self.start = self.end

class grad_feat_ext:
    def __init__(self, model, target_layer, data_len):
        self.model = model
        self.target_layer = target_layer
        self.hook_list = {}
        self.data_len = data_len
        self.feat_save = feat_fn(self.data_len)
        self.initialize()
        
    def initialize(self):
        for name, module in self.model._modules.items():
            if module_type(name) in self.target_layer:
                curr_name = name
                self.hook_list[curr_name] = grad_fn(curr_name,self.data_len)
                module.weight.register_hook(self.hook_list[curr_name].save_grad)
            else:
                for  name_m, module_m in module._modules.items():
                    if module_type(name_m) in self.target_layer:
                        curr_name = name + '_' + name_m
                        self.hook_list[curr_name] = grad_fn(curr_name,self.data_len)
                        module_m.weight.register_hook(self.hook_list[curr_name].save_grad)
                    else:
                        for  name_b, module_b in module_m._modules.items():
                            if module_type(name_b) in self.target_layer:
                                curr_name = name + '_' + name_m + '_' + name_b
                                self.hook_list[curr_name] = grad_fn(curr_name,self.data_len)
                                module_b.weight.register_hook(self.hook_list[curr_name].save_grad)
                            else:
                                for  name_l, module_l in module_b._modules.items():
                                    if module_type(name_l) in self.target_layer:
                                        curr_name = name + '_' + name_m + '_' + name_b + '_' + name_l
                                        self.hook_list[curr_name] = grad_fn(curr_name,self.data_len)
                                        module_l.weight.register_hook(self.hook_list[curr_name].save_grad)
                                    else:
                                        for  name_f, module_f in module_l._modules.items():
                                            if module_type(name_f) in self.target_layer:
                                                curr_name = name + '_' + name_m + '_' + name_b + '_' + name_l + '_' + name_f
                                                self.hook_list[curr_name] = grad_fn(curr_name,self.data_len)
                                                module_f.regis.weight.register_hook(self.hook_list[curr_name].save_grad)
    
    
    def __call__(self, x):
        for name, module in self.model._modules.items():
            if name  == 'fc':
                self.feat_save.save_feat(x.squeeze())
                x = module(x.squeeze())
            else:
                x = module(x)
        return x