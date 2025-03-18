import torch
import torch.nn.functional as F

def reparameterize(mu, logvar):
    """
    Reparameterization trick: sample z = mu + sigma * epsilon.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def compute_losses(class_logits, pred_energy, pred_macros,
                   batch_Y, batch_target_EI, batch_min_macros, batch_max_macros,
                   mu, logvar, batch_size):
    """
    Computes individual loss components and the total loss.
    
    Returns:
      - CE_loss: Meal classification (cross-entropy) loss.
      - KLD_loss: Kullback-Leibler divergence loss.
      - EI_loss: Energy intake (MSE) loss.
      - L_macro: Macronutrient penalty loss.
      - total_loss: Sum of all loss components.
    """
    # 1. Meal classification loss (cross-entropy)
    T = class_logits.size(1)
    CE_loss = 0.0
    for t in range(T):
        CE_loss += F.cross_entropy(class_logits[:, t, :], batch_Y[:, t])
    CE_loss = CE_loss / T
    
    # 2. Kullback-Leibler divergence loss
    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD_loss = KLD_loss / batch_size
    
    # 3. Energy intake loss (MSE)
    EI_loss = F.mse_loss(pred_energy, batch_target_EI)
    
    # 4. Macronutrient penalty loss
    diff_min = torch.abs(batch_min_macros - pred_macros)
    diff_max = torch.abs(batch_max_macros - pred_macros)
    macro_penalty = diff_min + diff_max
    L_macro = macro_penalty.mean()
    
    total_loss = CE_loss + KLD_loss + EI_loss + L_macro
    return CE_loss, KLD_loss, EI_loss, L_macro, total_loss

def adjust_meal_portions(energies, target_EI):
    """
    Adjusts the energy portion of each meal based on the difference between
    the predicted total energy and the target energy intake.
    
    energies: Tensor of shape [batch, T] with predicted energies for each meal.
    target_EI: Tensor of shape [batch] with the target energy intake for each user.
    
    Returns:
      adjusted_energies: Tensor of shape [batch, T] with adjusted meal energies.
      new_total_energy: Tensor of shape [batch] with the sum of adjusted energies.
    
    Formula: For each meal portion m_p, compute:
      d = (target_EI - pred_total_energy) / pred_total_energy
      m_p' = m_p * (1 + d)
    """
    # Compute predicted total energy per sample
    pred_total_energy = energies.sum(dim=1)  # shape: [batch]
    # Compute difference ratio d: the percentage difference between target and predicted energy
    d = (target_EI - pred_total_energy) / pred_total_energy  # shape: [batch]
    # Expand d to have the same shape as energies: [batch, 1]
    d_expanded = d.unsqueeze(1)
    # Adjust each meal portion: m_p' = m_p * (1 + d)
    adjusted_energies = energies * (1 + d_expanded)
    new_total_energy = adjusted_energies.sum(dim=1)
    return adjusted_energies, new_total_energy