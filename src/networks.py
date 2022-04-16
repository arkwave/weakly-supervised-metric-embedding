# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2020-08-24 16:37:55
# @Last Modified by:   Ananth
# @Last Modified time: 2020-08-25 20:55:42

import torch
import torch.nn as nn 


class EmbeddingEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim, dropout_val, f=nn.LeakyReLU()):

        super(EmbeddingEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout_val)
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), f, self.dropout,
                                     nn.Linear(500, 500), f, self.dropout,
                                     nn.Linear(500, 2000), f, self.dropout,
                                     nn.Linear(2000, 10), f, self.dropout,
                                     nn.Linear(10, latent_dim)
                                     )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 10), self.dropout,
                                     nn.Linear(10, 2000), f, self.dropout,
                                     nn.Linear(2000, 500), f, self.dropout,
                                     nn.Linear(500, 500), f, self.dropout,
                                     nn.Linear(500, input_dim)) 

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded 

    def get_embedding(self, x):
        return self.encoder(x)


class TripletEmbeddingEncoder(nn.Module):

    def __init__(self, embeddingnet):
        super(TripletEmbeddingEncoder, self).__init__()
        self.embeddingnet = embeddingnet 

    def forward(self, x1, x2, x3):
        output1 = self.embeddingnet(x1)
        output2 = self.embeddingnet(x2)
        output3 = self.embeddingnet(x3)

        return output1, output2, output3 

    def get_embedding(self, x):
        return self.embeddingnet.get_embedding(x)

    def get_full_pass(self, x):
        return self.embeddingnet(x)


class PairwiseEmbeddingEncoder(nn.Module):
    def __init__(self, embeddingnet):
        super(PairwiseEmbeddingEncoder, self).__init__() 
        self.embeddingnet = embeddingnet

    def forward(self, x1, x2):
        output1 = self.embeddingnet(x1)
        output2 = self.embeddingnet(x2)
        return output1, output2 

    def get_embedding(self, x):
        return self.embeddingnet.get_embedding(x)

    def get_full_pass(self, x):
        return self.embeddingnet(x)


class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim=20, beta=1, dropout=0, f=nn.LeakyReLU(), layers=None, device=None):
        """[summary]

        Args:
            input_dim ([type]): dimensionality of input
            latent_dim (int, optional): dimensionality of latent space. Defaults to 20.
            beta (int, optional): value to be used in beta-regularization. Defaults to 1.
            dropout (int, optional): dropout value for encoder/decoder networks. Defaults to 0.
            f ([type], optional): activation function. Defaults to nn.LeakyReLU().
            layers ([type], optional): [description]. Defaults to None.
        """        

        super(VAE, self).__init__()
        self.input_dim = input_dim 
        self.latent_dim = latent_dim 
        self.beta = beta 
        self.dropout = nn.Dropout(dropout)
        self.device = device
        encoder_modules = []

        # build encoder.
        if layers is None:
            layers = [500, 500, 2000, 10]
        
        for dim in layers:
            layer = nn.Sequential(nn.Linear(input_dim, dim), 
                                  nn.BatchNorm1d(dim),
                                  self.dropout,
                                  f)
            encoder_modules.append(layer)
            input_dim = dim 
        self.encoder = nn.Sequential(*encoder_modules)

        # the mu and variance layers.
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

        # build the decoder.
        decoder_modules = []
        layers.reverse()
        input_dim = latent_dim
        for dim in layers:
            layer = nn.Sequential(nn.Linear(input_dim, dim),
                                  nn.BatchNorm1d(dim),
                                  self.dropout,
                                  f)
            decoder_modules.append(layer)
            input_dim = dim
        
        final_layer =  nn.Linear(input_dim, self.input_dim)
        decoder_modules.append(final_layer)
        
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        # encode
        mu, logvar = self.encode(x)
        # reparametrize
        z = self.reparametrize(mu, logvar)
        # decode
        recon = self.decode(z)
        combined = torch.stack([mu, logvar], dim=2)
        return combined, recon

    def reparametrize(self, mu, logvar, eval=False):
        sd = torch.exp(0.5*logvar)
        epsilon = torch.normal(0, 1, size=mu.shape)
        if (self.device is not None) and (not eval):
            sd = sd.to(self.device)
            epsilon = epsilon.to(self.device)
        z = mu + sd * epsilon 
        return z 

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar
    
    def decode(self, x):
        return self.decoder(x)
    
    def get_full_pass(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar, eval=True)
        recon = self.decode(z)
        return mu, recon 


class PairwiseVAE(nn.Module):
    def __init__(self, base_vae):
        super(PairwiseVAE, self).__init__()
        self.vae = base_vae
    
    def forward(self, x1, x2):
        output1 = self.vae(x1)
        output2 = self.vae(x2)
        return output1, output2 
    
    def get_embedding(self, x):
        return self.vae.encode(x)
    
    def get_full_pass(self, x):
        return self.vae.get_full_pass(x)


class TripletVAE(nn.Module):
    def __init__(self, base_vae):
        super(TripletVAE, self).__init__()
        self.vae = base_vae 
    
    def forward(self, x1, x2, x3):
        output1 = self.vae(x1)
        output2 = self.vae(x2)
        output3 = self.vae(x3)
        return output1, output2, output3 
    
    def get_embedding(self, x):
        return self.vae.encode(x)
    
    def get_full_pass(self, x):
        return self.vae(x)
