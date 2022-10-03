
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import utils as tv_utils
from torch.utils.tensorboard import SummaryWriter

import os
from pathlib import Path

writer = SummaryWriter()

def l2_norm(r):
    r /= torch.norm(r, dim=1, keepdim=True) + 1e-8
    return r

class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter
        self.image = True
        self.imagepath = args.datapath / Path('vat_images')

    def forward(self, model, x):

        with torch.no_grad():
            preds = F.softmax(model(x), dim=1)

        #random tensor sample 
        r = torch.randn(x.shape).to(x.device)
        r = l2_norm(r)

        if not os.path.exists(self.imagepath):
            os.makedirs(self.imagepath)
 
        for _ in range(self.vat_iter): 
            r.requires_grad_()                                                      
            advEx = x + self.xi * r
            if self.image: 
                grid = tv_utils.make_grid(advEx, value_range=(0,10))
                writer.add_image('Adversarial examples', grid, 0)
                self.image = False
            advPred = F.log_softmax(model(advEx), dim=1)
            adv_dist = F.kl_div(advPred, preds, reduction='batchmean')
            adv_dist.backward()
            #storing the gradient of advDistance in d 
            d = l2_norm(r.grad)
            model.zero_grad()
        writer.close()

        #calculating the adversarial pertubation 
        r_adv = d * self.eps
        advPreds = model(x + r_adv)
        advPreds = F.log_softmax(advPreds, dim=1)
        loss = F.kl_div(advPreds, preds, reduction='batchmean')

        return loss