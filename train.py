import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from model import DAGMM
from forward_step import ComputeLoss
# from utils.utils import weights_init_normal

from utilities.utilities import *
from model import *

class TrainerAE:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device

    def train(self):
        """Training the AE model"""
        self.model = AE(self.args.latent_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        criterion = torch.nn.MSELoss()
        self.model.train()
        epoch_losses = []
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()
                
                x_hat = self.model(x)  # Get the reconstruction from the model

                loss = criterion(x_hat, x)  # Calculate the loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
            average_loss = total_loss / len(self.train_loader)
            epoch_losses.append(average_loss)
            if epoch % 5 == 0:
                print(f'Training Autoencoder... Epoch: {epoch}, Loss: {total_loss / len(self.train_loader)}')
        plot_loss(epoch_losses)

class TrainerDAGMM:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device


    def train(self):
        """Training the DAGMM model"""
        # if self.args.model == 'vae':
        #     self.model = VAE(self.args.latent_dim).to(self.device)
        # elif self.args.model == 'ae':
        #     self.model = AE(self.args.latent_dim).to(self.device)
        # elif self.args.model == 'betavae':
        #     self.model = BetaVAE(self.args.latent_dim, beta=self.args.beta).to(self.device)

        self.model = DAGMM(self.args.n_gmm, self.args.latent_dim).to(self.device)
        # self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
        #                            self.device, self.args.n_gmm)
        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   self.args.lambda_recon, self.args.lambda_kl, self.device, self.args.n_gmm)
    
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()
                
                mu, logvar, x_hat, _c, gamma, z = self.model(x)

                loss = self.compute.forward(x, x_hat, z, gamma, mu, logvar)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
            print('Training DAGMM... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
                

class TrainerBetaVAEwEstimationNetWork:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device


    def train(self):
        """Training the DAGMM model"""
        # if self.args.model == 'vae':
        #     self.model = VAE(self.args.latent_dim).to(self.device)
        # elif self.args.model == 'ae':
        #     self.model = AE(self.args.latent_dim).to(self.device)
        # elif self.args.model == 'betavae':
        self.model = BetaVAE(self.args.latent_dim, beta=self.args.beta).to(self.device)

        # self.model = DAGMM(self.args.n_gmm, self.args.latent_dim).to(self.device)
        # self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
        #                            self.device, self.args.n_gmm)
        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   self.args.lambda_recon, self.args.lambda_kl, self.device, self.args.n_gmm)
    
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()
                
                mu, logvar, x_hat, _c, gamma, z = self.model(x)

                loss = self.compute.forward(x, x_hat, z, gamma, mu, logvar)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
            print('Training DAGMM... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
                

