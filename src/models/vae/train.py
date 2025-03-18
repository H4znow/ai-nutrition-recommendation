import torch
import torch.optim as optim
from vae import Encoder, Decoder
# from dataset import generate_synthetic_data
from utils import reparameterize, compute_losses, adjust_meal_portions

def run_epoch(encoder, decoder, X_features, Y_meals, target_EIs, min_macros, max_macros, batch_size, optimizer):
    encoder.train()
    decoder.train()
    
    num_samples = X_features.size(0)
    permutation = torch.randperm(num_samples)
    total_loss = 0.0
    
    for i in range(0, num_samples, batch_size):
        optimizer.zero_grad()
        
        indices = permutation[i:i+batch_size]
        batch_X = X_features[indices]
        batch_Y = Y_meals[indices]              # [batch, 6]
        batch_target_EI = target_EIs[indices]     # [batch]
        batch_min_macros = min_macros[indices]    # [batch, 4]
        batch_max_macros = max_macros[indices]    # [batch, 4]
        
        # Encoder forward pass
        mu, logvar = encoder(batch_X)
        z = reparameterize(mu, logvar)
        
        # Decoder forward pass (now returns individual meal energies)
        class_logits, pred_energy, pred_macros, energies_tensor = decoder(z)
        
        # Optionally, adjust the meal portions using the optimizer layer:
        adjusted_energies, new_total_energy = adjust_meal_portions(energies_tensor, batch_target_EI)
        # (For training purposes, you might choose to compute EI_loss on pred_energy
        #  or on new_total_energy depending on your design. Here we continue with pred_energy.)
        
        # Compute losses using utility function
        CE_loss, KLD_loss, EI_loss, L_macro, loss = compute_losses(
            class_logits, pred_energy, pred_macros,
            batch_Y, batch_target_EI, batch_min_macros, batch_max_macros,
            mu, logvar, batch_X.size(0)
        )
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / (num_samples / batch_size)
    return avg_loss

def train_model(encoder, decoder, X_features, Y_meals, target_EIs, min_macros, max_macros,
                num_epochs=50, batch_size=32, lr=1e-3):
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    
    for epoch in range(1, num_epochs+1):
        avg_loss = run_epoch(encoder, decoder, X_features, Y_meals, target_EIs, min_macros, max_macros, batch_size, optimizer)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    # Generate synthetic data
    X_features, Y_meals, target_EIs, min_macros, max_macros = generate_synthetic_data(num_users=1000)
    
    # Define model dimensions
    input_dim = 8          # User profile features
    hidden_dim = 16
    latent_dim = 8
    num_classes = 10       # Meal classes
    macro_dim = 4          # Macronutrients (e.g., protein, carbs, fat, sfa)
    
    # Instantiate models
    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, num_classes, macro_dim)
    
    # Train the model
    train_model(encoder, decoder, X_features, Y_meals, target_EIs, min_macros, max_macros,
                num_epochs=50, batch_size=32, lr=1e-3)
