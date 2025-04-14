import torch
import torch.nn.functional as F


def adjust_meal_quantity(energies: torch.Tensor, batch_target_EI: torch.Tensor):
    "Optimizer function [will be used later]"
    pred_total_energy = energies.sum(dim=1)

    d = (batch_target_EI - pred_total_energy) / pred_total_energy

    d_expanded = d.unsqueeze(1)

    adjusted_energies = energies * (1 + d_expanded)

    new_total_energy = adjusted_energies.sum(dim=1)

    return adjusted_energies, new_total_energy   


def compute_L_macro(batch_min_macros: torch.Tensor, batch_max_macros: torch.Tensor, pred_macros: torch.Tensor):
    "Maconutriment penalty loss"
    diff_min = torch.abs(batch_min_macros - pred_macros)

    diff_max = torch.abs(batch_max_macros - pred_macros)

    macro_penalty = diff_min + diff_max

    L_macro = macro_penalty.mean()

    return L_macro


def compute_L_energy(pred_energy: torch.Tensor, batch_target_EI: torch.Tensor):
    "Energy intake loss"
    L_energy = F.mse_loss(pred_energy, batch_target_EI)

    return L_energy


def compute_KLD(mu: torch.Tensor, logvar: torch.Tensor, batch_size: int):
    "Kullback-Leibler Divergence loss"
    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    KLD_loss = KLD_loss / batch_size

    return KLD_loss


def compute_L_MC(class_logits: torch.Tensor, batch_Y: torch.Tensor):
    "Cross entropy loss"
    T = class_logits.size(1)

    CE_loss = 0.0

    for t in range(T):

        CE_loss += F.cross_entropy(class_logits[:, t, :], batch_Y[:, t])

    CE_loss = CE_loss / T
    
    return CE_loss