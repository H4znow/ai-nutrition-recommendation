import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    "Encoder model for the variational auto-encoder (vae)."


    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Args:
            input_dim (int): input dimension of encoder.
            hidden_dim (int): hidden neurons dimensions.
            latent_dim (int): generated latent space dimension.
        """
        super(Encoder, self).__init__()
        # fc1: projects input features to a hidden representation.
        #   Input: [batch_size, input_dim] → Output: [batch_size, hidden_dim]
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        
        # fc2: further processes the hidden representation.
        #   Input: [batch_size, hidden_dim] → Output: [batch_size, hidden_dim]
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        # fc_mu: computes the mean of the latent distribution.
        #   Input: [batch_size, hidden_dim] → Output: [batch_size, latent_dim]
        self.fc_mu = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        
        # fc_logvar: computes the log-variance of the latent distribution.
        #   Input: [batch_size, hidden_dim] → Output: [batch_size, latent_dim]
        self.fc_logvar = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
    
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): input tensor of encoder.
        """
        # Apply fc1 with ReLU activation.
        #   x: [batch_size, input_dim] → [batch_size, hidden_dim]
        x = F.relu(self.fc1(x))
        
        # Apply fc2 with ReLU activation.
        #   x: [batch_size, hidden_dim] → h: [batch_size, hidden_dim]
        h = F.relu(self.fc2(x))
        
        # Compute the latent mean.
        #   h: [batch_size, hidden_dim] → mu: [batch_size, latent_dim]
        mu = self.fc_mu(h)
        
        # Compute the latent log-variance.
        #   h: [batch_size, hidden_dim] → logvar: [batch_size, latent_dim]
        logvar = self.fc_logvar(h)
        
        return mu, logvar