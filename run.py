# -*- coding: utf-8 -*-
# @Author: Ananth Ravi Kumar
# @Date:   2020-06-08 15:59:07
# @Last Modified by:   Ananth
# @Last Modified time: 2020-10-21 23:37:16

from networks import PairwiseEmbeddingEncoder, EmbeddingEncoder, TripletEmbeddingEncoder, VAE
from utils import PairGenerator, TripletGenerator
from losses import ContrastiveLoss, reconstruction_error, TripletLoss, VAE_Loss
from training import train_network, train_vae
import torch.optim as optim
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch 
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from plot_fns import label_plot
import numpy as np 

# # get the dataset 
# transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
# # grab and normalize training set, apply same parameters to test set.
# train_set = MNIST('./data', train=True, download=True, transform=transforms)
# param_loader = DataLoader(train_set, batch_size=len(train_set))
# data = next(iter(param_loader))
# train_data_mean, train_data_sd = data[0].mean(), data[0].std()

# # pull again with normalized params. 
# normalize = torchvision.transforms.Normalize(mean=train_data_mean, std=train_data_sd)
# transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                              normalize])

# train_set = MNIST('./data', train=True, download=True, transform=transforms)
# test_set = MNIST('./data', train=False, download=True, transform=transforms)

# latent_dim = 20
# input_dim = 784

# vae = VAE(input_dim, latent_dim)
# optimizer = optim.Adam(vae.parameters())
# loss = VAE_Loss

# network, losses = train_vae(vae, train_set, test_set, optimizer,
#                             loss, epochs=1, batch_size=64)

# plt.figure(figsize=(12, 8))
# plt.plot(range(len(losses)), losses)
# plt.grid()
# plt.xlabel("iterations")
# plt.ylabel("loss")
# plt.show()

# network.eval()
# test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
# with torch.no_grad():
#     testdata, testlabels = next(iter(test_loader))
#     testdata = testdata.reshape(-1, 784)
#     testlabels = testlabels.numpy()
#     (mus, logvars), predicted = network(testdata)

# # run tsne on mus.
# mus = mus.numpy()
# tsne_embedding = TSNE().fit_transform(mus)
# plt.figure(figsize=(8, 8))
# for cl in set(testlabels):
#     inds = np.where(testlabels == cl)[0]
#     dat = tsne_embedding[inds, :]
#     plt.scatter(dat[:, 0], dat[:, 1], label=cl, alpha=0.8, marker='.')
# plt.grid() 
# plt.legend()
# plt.title("t-SNE embedding of $\mu$")
# plt.show() 



from experiment import run_experiment 

all_results = run_experiment("pairwise", "vae", "cifar10", [0.75], [3], 
                             num_iters=10, batch_size=32, normalize=True,
                             feature_extractor='hog')