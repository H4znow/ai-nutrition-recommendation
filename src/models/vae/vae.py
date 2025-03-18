# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder: Maps user profile features to a latent distribution (mean & log-variance).
    User features include weight, height, BMI, BMR, PAL, and medical conditions.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    """
    Decoder: Generates a sequence of meals (6 meals per day) from a latent vector.
    Uses a GRUCell to generate the meal sequence. The first input is a projection
    of the latent vector into the GRU's hidden space.
    """
    def __init__(self, latent_dim, hidden_dim, num_classes, macro_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.macro_dim = macro_dim

        # Project latent vector (latent_dim) to hidden space (hidden_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)   # Raw logits for meal classes
        self.energy_head = nn.Linear(hidden_dim, 1)            # scalar
        self.macro_head = nn.Linear(hidden_dim, macro_dim)     # vector

    def forward(self, z):
        batch_size = z.size(0)
        T = 6  # Number of meals per day

        # Initialize hidden state on the same device as z
        h = torch.zeros(batch_size, self.hidden_dim, device=z.device)

        z_projected = self.latent_to_hidden(z) # Project latent vector to hidden dimension

        class_logits_seq = []
        energies_list = []
        total_macros = torch.zeros(batch_size, self.macro_dim, device=z.device)

        for t in range(T):
            if t == 0:
                inp = z_projected
            else:
                inp = h_prev

            h = self.gru_cell(inp, h)
            h_prev = h

            logits = self.classifier(h)             # Meal class logits
            energy = self.energy_head(h).squeeze(-1)  # Predicted energy for this meal, shape: [batch]
            macros = self.macro_head(h)             # Predicted macronutrients for this meal

            class_logits_seq.append(logits)
            energies_list.append(energy)
            total_macros += macros

        class_logits_seq = torch.stack(class_logits_seq, dim=1)  # Shape: [batch, T, num_classes]
        energies_tensor = torch.stack(energies_list, dim=1)        # Shape: [batch, T]
        total_energy = energies_tensor.sum(dim=1)                # Total energy over the day

        # Return also the per-meal energies for later adjustment
        return class_logits_seq, total_energy, total_macros, energies_tensor