
import torch
import math
import numpy as np

def spike_history_to_clifford(cfg, spks):
    # Create a geometric sequence of weights
    log_start = torch.tensor([1e-3], device=cfg.device).log()
    log_end = torch.tensor([1.0], device=cfg.device).log()
    log_steps = torch.linspace(0, 1, steps=cfg.topology_net.pos_xz.sliding_window, device=cfg.device)
    log_space = log_start * (1 - log_steps) + log_end * log_steps
    weights = torch.exp(log_space)
    weights /= weights.sum()  # Normalize to sum to 1

    # Compute the weighted average along the 0-th dimension (over the iterations)
    # Expand dimensions of weights to match the dimensions of window_spks
    weights_expanded = weights.unsqueeze(1)
    averaged_spks = torch.sum(spks * weights_expanded, dim=0)

    # Scale averaged_spks from [0, 1] to [-1, 1]
    scaled_spks = averaged_spks * 2 - 1
    return scaled_spks

def coords_to_rad_scaling_factor(cfg):
    return (2 * math.pi) / cfg.topology_net.pos_xz.coord_wrap

def coords_to_rad(cfg, coords):
    scaling_factor = coords_to_rad_scaling_factor(cfg)
    return coords * scaling_factor

def rad_to_coords(cfg, coords):
    scaling_factor = coords_to_rad_scaling_factor(cfg)
    return coords / scaling_factor

def coord_to_clifford(cfg, coords):
    coords_rad = coords_to_rad(cfg, coords)

    cos_theta = torch.cos(coords_rad[0])
    sin_theta = torch.sin(coords_rad[0])
    cos_phi = torch.cos(coords_rad[1])
    sin_phi = torch.sin(coords_rad[1])

    result = torch.tensor([cos_theta, sin_theta, cos_phi, sin_phi], device=cfg.device)
    return result

def coords_to_clifford(cfg, coords):
    coords_rad = coords_to_rad(cfg, coords)

    # Apply trigonometric functions to each element
    cos_theta = torch.cos(coords_rad[:, 0])  # Cosine of all theta values
    sin_theta = torch.sin(coords_rad[:, 0])  # Sine of all theta values
    cos_phi = torch.cos(coords_rad[:, 1])    # Cosine of all phi values
    sin_phi = torch.sin(coords_rad[:, 1])    # Sine of all phi values

    # Stack all results into a new tensor with shape [4, n]
    result = torch.stack((cos_theta, sin_theta, cos_phi, sin_phi), dim=0)

    # If you want to maintain the original order as [n, 4], you can transpose
    result = result.t()
    return result



def spikes_to_clifford(cfg, spikes_tensor):
    # Assume spikes_tensor is of shape (iter, 200)
    # Reshape the tensor to shape (iter, 4, 50)
    reshaped_tensor = spikes_tensor.view(-1, 4, cfg.topology_net.pos_xz.pop_code)

    # Generate neuron values tensor of shape (50)
    neuron_values = torch.linspace(-1, 1, steps=cfg.topology_net.pos_xz.pop_code, device=cfg.device)

    # Expand the dimensions of neuron_values to match the dimensions of reshaped_tensor
    # New shape of neuron_values: (1, 1, 50)
    neuron_values_expanded = neuron_values.unsqueeze(0).unsqueeze(0)

    # Compute the weighted sum along the last dimension
    # Shape of weighted_sum: (iter, 4)
    weighted_sum = torch.sum(reshaped_tensor * neuron_values_expanded, dim=2)

    # Compute the number of active neurons along the last dimension
    # Shape of num_active_neurons: (iter, 4)
    num_active_neurons = torch.sum(reshaped_tensor, dim=2) + 1e-10

    # Compute the decoded values
    decoded_values_tensor = weighted_sum / num_active_neurons

    return decoded_values_tensor