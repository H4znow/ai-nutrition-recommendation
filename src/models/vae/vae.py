import torch
import torch.nn as nn

# Hyperparameters (values adjusted for a small-to-medium model; original paper uses larger values)
input_dim = 8         # 8 features in the input (e.g., weight, height, BMR, age, BMI, targeted energy, PAL, disease indicator)
hidden_units = 64     # Dense layer size (256 in the original paper)
latent_dim = 64       # Latent dimension (256 in the original paper)
gru_units = 128       # GRU hidden size (512 in the original paper)
num_layers = 2        # Number of GRU layers (from the original paper)
D = 500               # Nutritional expert feedback offset (in calories)
optimizer = torch.optim.Adam  # Adam optimizer is used in the original paper
batch_size = 32       # Batch size (64 in the original paper)
epochs = 500          # Number of epochs (from the original paper)
num_meals = 6         # Number of meals per day (breakfast, morning snack, lunch, afternoon snack, dinner, supper)
meal_vocab_size = 1349  # Number of unique meals in the meal database

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_units: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc_mu = nn.Linear(hidden_units, latent_dim)
        self.fc_logvar = nn.Linear(hidden_units, latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
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

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, num_meals: int, gru_units: int, num_layers: int, meal_vocab_size: int):
        super(Decoder, self).__init__()
        self.num_meals = num_meals
        # GRU will generate a hidden state for each meal (i.e., time step)
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=gru_units, num_layers=num_layers, batch_first=True)
        # Predict nutritional values for each meal time step
        self.total_energy = nn.Linear(gru_units, 1)
        self.nutrient_value = nn.Linear(gru_units, 1)
        # Classifier to predict the meal class (as probabilities)
        self.meal_classifier = nn.Linear(gru_units, meal_vocab_size)
        
    def forward(self, z: torch.Tensor):
        # z has shape: (batch_size, latent_dim)
        # Repeat the latent vector for each meal to create a sequence: (batch_size, num_meals, latent_dim)
        z_seq = z.unsqueeze(1).repeat(1, self.num_meals, 1)
        # GRU outputs:
        #   h: hidden states for each time step with shape (batch_size, num_meals, gru_units)
        #   _ : final hidden state (not used here)
        h, _ = self.gru(z_seq)
        # Predict nutritional values for each meal time step
        total_energy = self.total_energy(h)       # shape: (batch_size, num_meals, 1)
        nutrient_value = self.nutrient_value(h)     # shape: (batch_size, num_meals, 1)
        # Predict meal class probabilities for each meal time step
        meal_logits = self.meal_classifier(h)       # shape: (batch_size, num_meals, meal_vocab_size)
        meal_probs = torch.softmax(meal_logits, dim=-1)
        return total_energy, nutrient_value, meal_probs
