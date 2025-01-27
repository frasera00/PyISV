import numpy as np
from typing import Tuple
import torch

#################################################
############ TORCH KDE RDF FUNCTIONS ############
#################################################

@torch.jit.script
def marginal_pdf(
    values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, epsilon: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor]:

    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
    pdf = pdf / normalization

    return pdf, kernel_values

@torch.jit.script    
def computefactor(bins: torch.Tensor) -> torch.Tensor:
    return 1.0/((bins[-1]-bins[0])/(bins.size()[0]-1))

@torch.jit.script    
def normalized_histogram(x: torch.Tensor, bins: torch.Tensor,factor: torch.Tensor, bandwidth: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    
    pdf, _ = marginal_pdf(x.unsqueeze(2), bins, bandwidth, epsilon)

    return pdf*factor

@torch.jit.script
def compute_distances(AtomPos: torch.Tensor) -> torch.Tensor:
    y2 = torch.sum(AtomPos**2, 1)
    x2 = y2.reshape(-1, 1)
    xy = torch.matmul(AtomPos, AtomPos.T)
    distances = (x2 - 2*xy + y2)
    n_atoms = len(AtomPos)
    indices = torch.triu_indices((n_atoms),(n_atoms), offset = 1)
    return torch.sqrt(distances[indices[0], indices[1]])

def compute_asepbc_distances(Atoms):
    distances = Atoms.get_all_distances(mic=True)
    n_atoms = len(distances)
    indices = torch.triu_indices((n_atoms),(n_atoms), offset = 1)
    return torch.Tensor(distances[indices[0], indices[1]])

def torch_rdf_calc(pos: torch.Tensor, bins: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    return normalized_histogram(compute_distances(pos).unsqueeze(0), bins, computefactor(bins), bandwidth)

def ase_periodic_torch_rdf_calc(Atoms, bins: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    return normalized_histogram(compute_asepbc_distances(Atoms).unsqueeze(0), bins, computefactor(bins), bandwidth)

