import torch as t
import scipy as sp
from sklearn.covariance import EmpiricalCovariance

def centering(grads, sample_size = None, type = 'mean'):
    if type == 'mean':
        dominant_theta = t.mean(t.abs(grads),dim=0)
    else:
        dominant_theta = t.var(grads,dim=0)
    argsort = t.argsort(dominant_theta, descending = True)
    pick = argsort if sample_size == None else argsort[:sample_size]
    grads = grads[:,pick]
    return grads

def calc_vmf(grads):
    norm = t.norm(grads,p=None, dim=1, keepdim=True)
    norm = t.clamp(norm,min=1e-8)
    x = grads / norm
    u = t.sum(x,dim=0)
    u_norm = t.norm(u,p=None,dim=0, keepdim=False)
    d = x.size(1)
    r = u_norm / len(x)
    k = r * ( d - r ** 2 )  / ( 1 - r ** 2 )
    
    exponent = k*t.matmul(u / u_norm, x.T)
    exponent -= t.max(exponent)
    ret_tmp = t.clamp(t.exp(exponent),min=1e-8,max=1)
    return ret_tmp / t.sum(ret_tmp)

def calc_mag(grads):
    norm = t.norm(grads, p=None,dim=1, keepdim=False)
    norm = t.clamp(norm,min=1e-8)
    return norm
    mag_tmp = 1./norm
    return mag_tmp / t.sum(mag_tmp)

def calc_unit_grad(grads):
    norm = t.norm(grads, p=None, dim=1, keepdim = False).unsqueeze(dim=1).repeat(1,grads.shape[1])
    zero_norm = t.where(norm == 0)
    ret = grads / norm 
    ret[zero_norm] = 0 
    return ret


def mahalanobis(x):
    cov_fn = EmpiricalCovariance()
    cov_fn.fit(x.cpu().numpy())
    temp_precision = cov_fn.precision_
    temp_precision = t.from_numpy(temp_precision).float()
    mahal = -0.5 * t.mm(t.mm(x, temp_precision), x.t())
    return mahal.diag()
