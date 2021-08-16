import numpy as np
import torch

BETA = torch.distributions.beta.Beta(torch.tensor([2.], device=torch.device('cuda:0')), 
                                     torch.tensor([5.], device=torch.device('cuda:0')))

def _noise(x, add_noise_level=0.0, mult_noise_level=0.0, sparsity_level=0.0):
    add_noise = torch.tensor(0.0, device=torch.device('cuda:0'))
    mult_noise = torch.tensor(1.0, device=torch.device('cuda:0'))
    if add_noise_level > 0.0:
        var = add_noise_level * BETA.sample()
        add_noise = var * torch.cuda.FloatTensor(x.shape).normal_()
    if mult_noise_level > 0.0:
        mult_noise = mult_noise_level * BETA.sample() * (2*torch.cuda.FloatTensor(x.shape).uniform_()-1) + 1 
    return mult_noise * x + add_noise      

def do_noisy_mixup(x, y, alpha=0.0, add_noise_level=0.0, mult_noise_level=0.0):
    lam = BETA.sample() if alpha > 0.0 else 1.0
    index = torch.randperm(x.size()[0], device=torch.device('cuda:0'))
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return _noise(mixed_x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level), y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)