import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam


from encoder import Encoder
from decoder import Decoder
from .variatio_autencoder import EarlyStopping
from optimizer import optimize_meal_portions
from loss_functions import *

class VariationalAutoencoderTrainer:
    """
    Variational Autoencoder Trainer class that trains its two components:
    an encoder and a decoder.
    """
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 test_loader: DataLoader,
                 optimizer: Adam,
                 early_stopping : EarlyStopping = EarlyStopping(),
                 device: str = 'cpu'):
        """
        Args:
            encoder (nn.Module): Encoder network.
            decoder (nn.Module): Decoder network.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            test_loader (DataLoader): Test data loader.
            optimizer (Adam): Adam optimizer for both encoder and decoder.
            early_stopping (EarlyStopping) : instance of early stopping class.
            device (str): Device on which operations are performed.
        """
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.device = device


    def _compute_loss(self, 
                      user_features: torch.Tensor, 
                      meal_targets: torch.Tensor, 
                      target_energy_intakes: torch.Tensor, 
                      min_macronutrients: torch.Tensor, 
                      max_macronutrients: torch.Tensor):
        """
        Computes the total loss for a batch.

        Args:
            user_features (torch.Tensor): User information tensor (encoder input).
            meal_targets (torch.Tensor): Ground truth daily meals.
            target_energy_intakes (torch.Tensor): Target energy intake values.
            min_macronutrients (torch.Tensor): Minimum targeted macronutrient values.
            max_macronutrients (torch.Tensor): Maximum targeted macronutrient values.

        Returns:
            total_loss (torch.Tensor): Combined loss.
            macro_loss (torch.Tensor): Loss term for macronutrients.
            energy_loss (torch.Tensor): Loss term for energy intake.
            kl_loss (torch.Tensor): KL divergence loss.
            meal_class_loss (torch.Tensor): Loss term for meal classification.
            energies_tensor (torch.Tensor): Output tensor from the decoder regarding energies.
        """
        # Get batch size
        batch_size = user_features.size(0)

        # Forward pass through encoder
        mu, logvar = self.encoder(user_features)
        
        # Reparameterization trick
        epsilon = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        z = mu + std * epsilon
        
        # Forward pass through decoder
        meal_logits, predicted_energy, predicted_macros, energies_tensor = self.decoder(z)
        
        # Compute individual loss terms
        macro_loss     = compute_L_macro(min_macronutrients, max_macronutrients, predicted_macros)
        energy_loss    = compute_L_energy(predicted_energy, target_energy_intakes)
        kl_loss        = compute_KLD(mu, logvar, batch_size)
        meal_class_loss = compute_L_MC(meal_logits, meal_targets)
        
        total_loss = macro_loss + energy_loss + kl_loss + meal_class_loss
        return total_loss, macro_loss, energy_loss, kl_loss, meal_class_loss, energies_tensor


    def train_epoch(self):
        """
        Performs one training epoch.
        """
        self.encoder.train()
        self.decoder.train()
        total_loss = 0.0
        total_batches = len(self.train_loader)
        
        for batch_idx, (user_features, meal_targets, target_energy_intakes, min_macronutrients, max_macronutrients) in enumerate(self.train_loader):
            # Move data to device
            user_features        = user_features.to(self.device)
            meal_targets         = meal_targets.to(self.device)
            target_energy_intakes = target_energy_intakes.to(self.device)
            min_macronutrients   = min_macronutrients.to(self.device)
            max_macronutrients   = max_macronutrients.to(self.device)
            
            self.optimizer.zero_grad()

            loss, macro_loss, energy_loss, kl_loss, meal_class_loss, energies_tensor = self._compute_loss(
                user_features, meal_targets, target_energy_intakes, min_macronutrients, max_macronutrients
            )

            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Train Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f} "
                      f"(Macro: {macro_loss.item():.4f}, Energy: {energy_loss.item():.4f}, "
                      f"KLD: {kl_loss.item():.4f}, Meal Classification: {meal_class_loss.item():.4f})")
                
        avg_loss = total_loss / total_batches
        print(f"Training epoch complete. Average Loss: {avg_loss:.4f}\n")
        return avg_loss, energies_tensor


    def validate_epoch(self):
        """
        Evaluates the model on the validation set.
        """
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0.0
        total_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, (user_features, meal_targets, target_energy_intakes, min_macronutrients, max_macronutrients) in enumerate(self.val_loader):
                user_features        = user_features.to(self.device)
                meal_targets         = meal_targets.to(self.device)
                target_energy_intakes = target_energy_intakes.to(self.device)
                min_macronutrients   = min_macronutrients.to(self.device)
                max_macronutrients   = max_macronutrients.to(self.device)
                
                loss, _, _, _, _, _ = self._compute_loss(
                    user_features, meal_targets, target_energy_intakes, min_macronutrients, max_macronutrients
                )
                total_loss += loss.item()
                
        avg_loss = total_loss / total_batches
        print(f"Validation epoch complete. Average Loss: {avg_loss:.4f}\n")
        return avg_loss


    def train(self, num_epochs: int):
        """
        Trains the model for the specified number of epochs.
        
        Args:
            num_epochs (int): Number of epochs to run.
            early_stopping (int): Number of epoche to run
        
        Returns:
            train_losses (list): List of average training losses.
            val_losses (list): List of average validation losses.
        """
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_loss, _ = self.train_epoch()
            val_loss = self.validate_epoch()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.early_stopping(val_loss, self.encoder, self.decoder)
            if self.early_stopping.early_stop:
                print("Early stopping")
                self.early_stopping.load_best_model(self.encoder, self.decoder)
                break
        return train_losses, val_losses, self.encoder, self.decoder


    def predict(self, user_features: torch.Tensor):
        """
        Generates predictions given input features.
        
        Returns:
            meal_logits, predicted_energy, predicted_macros, energies_tensor
        """
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            user_features = user_features.to(self.device)
            # For prediction we use the mean (mu) as the latent representation.
            mu, _                                                            = self.encoder(user_features)
            meal_logits, predicted_energy, predicted_macros, energies_tensor = self.decoder(mu)
        return meal_logits, predicted_energy, predicted_macros, energies_tensor


    def evaluate(self):
        """
        Evaluates the model on both training and test datasets.
        
        Returns:
            dict: Dictionary containing average training and test losses.
        """
        self.encoder.eval()
        self.decoder.eval()
        
        # Evaluate on training data
        total_train_loss = 0.0
        for user_features, meal_targets, target_energy_intakes, min_macronutrients, max_macronutrients in self.train_loader:
            user_features         = user_features.to(self.device)
            meal_targets          = meal_targets.to(self.device)
            target_energy_intakes = target_energy_intakes.to(self.device)
            min_macronutrients    = min_macronutrients.to(self.device)
            max_macronutrients    = max_macronutrients.to(self.device)
            
            with torch.no_grad():
                loss, _, _, _, _, _ = self._compute_loss(
                    user_features, meal_targets, target_energy_intakes, min_macronutrients, max_macronutrients
                )
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(self.train_loader)
        
        # Evaluate on test data
        total_test_loss = 0.0
        for user_features, meal_targets, target_energy_intakes, min_macronutrients, max_macronutrients in self.test_loader:
            user_features         = user_features.to(self.device)
            meal_targets          = meal_targets.to(self.device)
            target_energy_intakes = target_energy_intakes.to(self.device)
            min_macronutrients    = min_macronutrients.to(self.device)
            max_macronutrients    = max_macronutrients.to(self.device)
            
            with torch.no_grad():
                loss, _, _, _, _, _ = self._compute_loss(
                    user_features, meal_targets, target_energy_intakes, min_macronutrients, max_macronutrients
                )
            total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(self.test_loader)
        
        print(f"Evaluation complete. Average Training Loss: {avg_train_loss:.4f}, Average Test Loss: {avg_test_loss:.4f}\n")
        return avg_train_loss, avg_test_loss
    
    
    def predict_meal_plan(self, user_features: torch.Tensor, target_energy: torch.Tensor):
        """
        Predicts the user's meal plan and scales portions so total energy = target.
        
        Args:
            user_features: [batch_size, input_dim]
            target_energy: [batch_size, 1]
        
        Returns:
            pred_meal_plan: shape [batch_size, T] (the chosen meal IDs)
            scaled_energies: shape [batch_size, T, 1]
            total_macros: shape [batch_size, macro_dim] (unchanged)
        """
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            user_features = user_features.to(self.device)
            target_energy = target_energy.to(self.device)

            # 1) Forward pass
            mu, _ = self.encoder(user_features)
            class_logits_seq, predicted_energy, predicted_macros, energies_tensor = self.decoder(mu)
            
            # 2) Pick meals (argmax)
            pred_meal_plan = torch.argmax(class_logits_seq, dim=2)  # [batch_size, T]

            # 3) Scale only the meal energies
            scaled_energies = optimize_meal_portions(
                predicted_energy=predicted_energy, 
                target_energy=target_energy,
                energies_tensor=energies_tensor
            )
            
        # Return final plan + scaled portion energies
        return pred_meal_plan, scaled_energies, predicted_macros
    

class EarlyStopping:
    """
    Implements early stopping to prevent overfitting during training.
    Saves the best model (encoder and decoder) based on validation loss.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
        delta (float): Minimum change in validation loss to qualify as an improvement.
    """
    def __init__(self, patience: int= 5, delta: float=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss: float, encoder: nn.Module, decoder: nn.Module):
        """
        Checks if the validation loss has improved. If not, increases the counter.
        Stops training if counter exceeds patience.

        Args:
            val_loss (float): Current validation loss.
            encoder (nn.Module): Encoder model.
            decoder (nn.Module): Decoder model.
        """
        score = -val_loss  # Lower validation loss is better, so we invert the score

        if self.best_score is None:
            # First epoch, initialize best score and save model states
            self.best_score = score
            self.best_model_state = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict()
            }
        elif score < self.best_score + self.delta:
            # No significant improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement found; update best score and reset counter
            self.best_score = score
            self.best_model_state = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict()
            }
            self.counter = 0

    def load_best_model(self, encoder: nn.Module, decoder: nn.Module):
        """
        Restores the encoder and decoder to their best recorded states.

        Args:
            encoder (nn.Module): Encoder model to restore.
            decoder (nn.Module): Decoder model to restore.
        """
        encoder.load_state_dict(self.best_model_state['encoder'])
        decoder.load_state_dict(self.best_model_state['decoder'])