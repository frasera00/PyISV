import numpy as np
from typing import Tuple
import torch

#################################################
############ ASE-TORCH KDE RDF FUNCTIONS ############
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

def compute_ase_all_distances(Atoms, mic=False):
    distances = Atoms.get_all_distances(mic)
    n_atoms = len(distances)
    indices = torch.triu_indices((n_atoms),(n_atoms), offset = 1)
    return torch.Tensor(distances[indices[0], indices[1]])

def compute_ase_idx_distances(conf, indices1, indices2=[], mic=False):
    dist_list = []
    if indices2 == []:
        for i in range(len(indices1)-1):
            dist_list.append(conf.get_distances(indices1[i],indices1[i+1:],mic=mic))
    else:    
        for i in range(len(indices1)):
            dist_list.append(conf.get_distances(indices1[i],indices2,mic=mic))     
    return np.hstack(dist_list) 

def torch_kde_calc(values: torch.Tensor, bins: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    return normalized_histogram(values.unsqueeze(0), bins, computefactor(bins), bandwidth)

def triple_kde_calc(specie1, specie2, conf, bins, bw, mic=False):
    there_is_s1 = False
    there_is_s2 = False
   
    indices_specie1 = [atom.index for atom in conf if atom.symbol == specie1] 
    indices_specie2 = [atom.index for atom in conf if atom.symbol == specie2]
    
    if len(indices_specie1) > 1:
        there_is_s1 = True
    if len(indices_specie2) > 1:
        there_is_s2 = True 
    
    num_bins = len(bins)
    s1_s1_rdf = np.zeros(num_bins)
    s2_s2_rdf = np.zeros(num_bins)
    s1_s2_rdf = np.zeros(num_bins)

    if there_is_s1:
        s1_s1_dist = compute_ase_idx_distances(conf, indices_specie1, indices2=[], mic=mic)
        s1_s1_rdf = torch_kde_calc(torch.Tensor(s1_s1_dist), bins, bw).numpy()

    if there_is_s2:
        s2_s2_dist = compute_ase_idx_distances(conf, indices_specie2, indices2=[], mic=mic)
        s2_s2_rdf = torch_kde_calc(torch.Tensor(s2_s2_dist), bins, bw).numpy()

    if (len(indices_specie1) > 0) and (len(indices_specie2) > 0):    
        s1_s2_dist = compute_ase_idx_distances(conf, indices_specie1, indices2=indices_specie2, mic=mic)
        s1_s2_rdf = torch_kde_calc(torch.Tensor(s1_s2_dist), bins, bw).numpy()        
    
    return s1_s1_rdf, s2_s2_rdf, s1_s2_rdf

def single_kde_calc(Atoms, bins: torch.Tensor, bandwidth: torch.Tensor, mic=False) -> torch.Tensor:
    distances = compute_ase_all_distances(Atoms, mic=mic)
    return torch_kde_calc(torch.Tensor(distances), bins, bandwidth).numpy()

