# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2020-08-24 17:06:12
# @Last Modified by:   Ananth
# @Last Modified time: 2020-08-25 21:36:14
import torch
import torch.nn as nn 
import torch.nn.functional as F 



class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin 
        self.eps = 1e-9

    def forward(self, anchor, positive, labels):
        distances = (positive - anchor).pow(2).sum(1)
        losses = (labels.float() * distances) + \
                 (1 - labels).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2) 
        return losses.mean()


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__() 
        self.margin = margin 

    def forward(self, anchor, positive, negative, labels):
        ap = (anchor - positive).pow(2).sum(1)
        an = (anchor - negative).pow(2).sum(1)
        losses = F.relu(ap - an + self.margin)
        return losses.mean() 


def reconstruction_error(input_, output, additional_losses=[0]):
    loss_fn = nn.MSELoss()
    mse = 0 
    # print(len(input_))
    # print(len(output))
    for i in range(len(input_)):
        loss = loss_fn(input_[i], output[i])
        mse += loss
    final = mse + additional_losses
    return final 


def VAE_Loss(sample, reconstructed, mu, logvar, batch_size, total_dim):
    loss_fn = nn.MSELoss()
    MSE = loss_fn(sample, reconstructed)
    KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / (batch_size * total_dim)
    return MSE + KLD


def KLDiv(mu1, mu2, logvar1, logvar2):
    """Implements the closed form KL divergence between two gaussians, each assumed
    to have diagonal covariance matrices, proxied here with vectors. 
    
    Note that inputs are batches of size (batch_size, dim), where dim is the dimension of the gaussians.


    Args:
        mu1 ([type]): [description]
        mu2 ([type]): [description]
        logvar1 ([type]): [description]
        logvar2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    z = mu2 - mu1 
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    var2_inv = 1/var2 + 1e-5

    logdets = torch.sum(logvar2, dim=1) - torch.sum(logvar1, dim=1)
    trace_term = torch.sum(var2_inv * var1, axis=1)
    quad_term = torch.sum(torch.pow(z, 2) * var2_inv, dim=1)

    return logdets + trace_term + quad_term


class PairwiseVAELoss(nn.Module):
    def __init__(self, margin):
        super(PairwiseVAELoss, self).__init__()
        self.margin = margin 
        self.eps = 1e-9
    
    def forward(self, anchor, positive, labels):
        mu1, logvar1 = anchor[:, :, 0], anchor[:, :, 1] 
        mu2, logvar2 = positive[:, :, 0], positive[:, :, 1]
        kldist = KLDiv(mu1, mu2, logvar1, logvar2)
        losses = (labels.float() * kldist) + \
                 (1 - labels).float() * F.relu(self.margin - (kldist + self.eps).sqrt()).pow(2) 
        return losses.mean() 

class TripletVAELoss(nn.Module):
    def __init__(self, margin):
        super(TripletVAELoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9 
    
    def forward(self, anchor, positive, negative, labels):
        mu1, logvar1 = anchor[:, :, 0], anchor[:, :, 1] 
        mu2, logvar2 = positive[:, :, 0], positive[:, :, 1]
        mu3, logvar3 = negative[:, :, 0], negative[:, :, 1] 
        ap = KLDiv(mu1, mu2, logvar1, logvar2)
        an = KLDiv(mu1, mu3, logvar1, logvar3)
        losses = F.relu(ap - an + self.margin)
        return losses.mean()