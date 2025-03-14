"""
 A deep generative network architecture to create weekly meal plans by employing sophisticated 
 loss functions to align the network with well-founded nutritional guidelines
 src : "AI nutrition recommendation using a deep generative model and ChatGPT", Ilias Papastratis & al., 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_units, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc_mu = nn.Linear(hidden_units, latent_dim)
        self.fc_logvar = nn.Linear(hidden_units, latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    z = mu + std * epsilon
    return z