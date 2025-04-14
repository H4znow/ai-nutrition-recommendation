import torch

def optimize_meal_portions(predicted_energy: torch.Tensor,
                           target_energy: torch.Tensor,
                           energies_tensor: torch.Tensor) -> torch.Tensor:
    """
    Adjusts the per-meal energies so that the total daily energy
    matches 'target_energy'.

    Args:
        predicted_energy: shape [batch_size, 1]
        target_energy:    shape [batch_size, 1]
        energies_tensor:  shape [batch_size, T, 1]
    
    Returns:
        scaled_energies:  shape [batch_size, T, 1]
                          scaled so sum(scaled_energies) = target_energy
    """
    # predicted_energy: sum of energies_tensor over T, shape [batch_size, 1]
    # d = (EI - predicted_energy) / predicted_energy
    d = (target_energy - predicted_energy) / predicted_energy  # shape: [batch_size, 1]

    # scale factor for each sample in the batch
    scale_factor = 1.0 + d  # shape: [batch_size, 1]

    # broadcast scale_factor over T time steps
    scaled_energies = energies_tensor * scale_factor.unsqueeze(dim=1)  # [batch_size, T, 1]

    return scaled_energies