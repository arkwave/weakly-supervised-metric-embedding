# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2020-08-25 11:25:26
# @Last Modified by:   Ananth
# @Last Modified time: 2020-10-15 01:26:10

from utils import jacobian
import numpy as np 
import torch 
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train_network(network, datagen, epochs, metric_loss, reconstruction_loss, 
                  optimizer, mu=1, contractive=False, device=None):
    """[summary]

    Args:
        network ([type]): [description]
        datagen ([type]): [description]
        epochs ([type]): [description]
        metric_loss ([type]): [description]
        reconstruction_loss ([type]): [description]
        optimizer ([type]): [description]
        mu (int, optional): [description]. Defaults to 1.
        contractive (bool, optional): [description]. Defaults to False.
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    
    recon_losses = []
    metric_losses = []
    total_losses = []

    print("using %s" % device)

    network = network.to(device)
    network.train() 

    for epoch in range(epochs):

        # generate the batch
        batch, labels = datagen.generate_batch()
        batch = tuple(x.to(device) for x in batch) 
        labels = labels.to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward step
        outputs = network(*batch)

        # separate into encoded and reconstructed 
        encoded, reconstructed = list(zip(*outputs))
        # encoded = [outputs[0] for output in outputs]
        # reconstructed = [i[1] for i in outputs]
        m_loss = metric_loss(*encoded, labels) * mu 

        j_norm = 0

        if contractive:
            for i in range(len(encoded)):
                J = jacobian(batch[i], encoded[i])
                j_norm += torch.norm(J)
            print("j_norm: ", j_norm)

        recon_loss = reconstruction_loss(batch, reconstructed, additional_losses=j_norm) * (1-mu)

        # add the losses together
        loss = recon_loss + m_loss 

        # append losses to history. 
        total_losses.append(loss.item())
        recon_losses.append(recon_loss.item())
        metric_losses.append(m_loss.item())

        # print("Iteration %s. Metric Loss: %s | Reconstruction Loss: %s" % (epoch, round(m_loss.item(), 3), round(recon_loss.item(), 3)))
        if epoch % (epochs/100) == 0 and epoch != 0:
            print("Iteration %s. Metric Loss: %s | Reconstruction Loss: %s" % (epoch, round(m_loss.item(), 3), round(recon_loss.item(), 3)))
            
        # backward step
        loss.backward() 
        optimizer.step() 

    return network, total_losses, recon_losses, metric_losses


def train_vae(network, train_set, test_set, optimizer, loss, epochs, batch_size=64):
    """Training/evaluation loop for the variational autoencoder.

    Args:
        network ([type]): [description]
        trainloader ([type]): [description]
        testloader ([type]): [description]
        optimizer ([type]): [description]
        epochs ([type]): [description]
    """
    recon_loss = nn.MSELoss()
    losses = []
    val_losses = []

    # instantiate dataloaders.
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_set, batch_size=1, shuffle=False)

    for epoch in range(epochs):
        for idx, (sample, label) in enumerate(trainloader):
            optimizer.zero_grad()
            total_dim = np.product(sample.shape[1:])
            sample = sample.reshape(-1, total_dim)

            # forward pass
            (mu, logvar), reconstructed = network(sample)
            # compute loss
            total_loss = loss(sample, reconstructed, mu, logvar, batch_size, total_dim)
            # backward pass. 
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

            if idx % 100 == 0 and idx > 0:
                print("epoch: %s | idx: %s | loss: %s" % (epoch, idx, total_loss.item()))

    return network, losses
