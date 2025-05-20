from typing import Tuple
from ase.io import read
from tqdm import tqdm
import numpy as np
import time
import torch
import logging
import os

@torch.jit.script
def marginal_pdf(values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, epsilon: float = 1e-10
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
def normalized_histogram(x: torch.Tensor, bins: torch.Tensor, factor: torch.Tensor, bandwidth: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Ensure x has at least 2 dimensions
    pdf, _ = marginal_pdf(x.unsqueeze(2), bins, bandwidth, epsilon)
    return pdf * factor

@torch.jit.script
def torch_kde_calc(values: torch.Tensor, bins: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    return normalized_histogram(values.unsqueeze(0), bins, computefactor(bins), bandwidth)

@torch.jit.script
def compute_pairwise_distances(positions: torch.Tensor) -> torch.Tensor:
    # positions: [N, 3] or [N, D]
    full = torch.cdist(positions, positions)  # [N, N]
    idx = torch.triu_indices(full.size(0), full.size(0), offset=1)
    return full[idx[0], idx[1]]  # [N*(N−1)/2]

def compute_ase_all_distances(Atoms, mic=False):
    distances = np.array(Atoms.get_all_distances(mic))  # Ensure distances is a NumPy array
    if distances.ndim != 2 or distances.shape[0] == 0:
        # Return an empty tensor if there are no atoms or distances is not 2D
        return torch.tensor([])
    n_atoms = distances.shape[0]
    indices = torch.triu_indices(n_atoms, n_atoms, offset=1)
    return torch.tensor(distances[indices[0], indices[1]])

def compute_ase_idx_distances(conf, indices1, indices2=None, mic=False):
    indices1 = list(indices1) if indices1 is not None else []
    indices2 = [] if indices2 is None else list(indices2)
    if len(indices1) == 0:
        return np.array([])
    dist_list = []
    if len(indices2) == 0:
        # Pairwise distances within indices1
        if len(indices1) < 2:
            return np.array([])
        for i, idx in enumerate(indices1[:-1]):
            dists = conf.get_distances(idx, indices1[i+1:], mic=mic)
            dist_list.append(dists)
    else:
        # Distances between indices1 and indices2
        for idx in indices1:
            dists = conf.get_distances(idx, indices2, mic=mic)
            dist_list.append(dists)
    if not dist_list:
        return np.array([])
    return np.hstack(dist_list)

def triple_kde_calc(specie1, specie2, atoms, bins, bandwidth, mic=False):
    """Calculate RDFs for all combinations of two species in an alloy.
    
    Returns a tuple of three tensors (s1_s1_rdf, s2_s2_rdf, s1_s2_rdf).
    Each tensor has shape [n_bins].
    """
    
    # Get atom symbols for indexing
    symbols = atoms.get_chemical_symbols()
    
    # Find indices for each species
    indices_specie1 = [i for i, s in enumerate(symbols) if s == specie1]
    indices_specie2 = [i for i, s in enumerate(symbols) if s == specie2]
    
    # Get distances for each species combination
    s1_s1_dist = compute_ase_idx_distances(atoms, indices_specie1, indices2=None, mic=mic)
    s2_s2_dist = compute_ase_idx_distances(atoms, indices_specie2, indices2=None, mic=mic)
    s1_s2_dist = compute_ase_idx_distances(atoms, indices_specie1, indices2=indices_specie2, mic=mic)
    
    # Calculate RDFs
    s1_s1_rdf = torch_kde_calc(torch.tensor(s1_s1_dist), bins, bandwidth) if len(s1_s1_dist) > 0 else torch.zeros_like(bins)
    s2_s2_rdf = torch_kde_calc(torch.tensor(s2_s2_dist), bins, bandwidth) if len(s2_s2_dist) > 0 else torch.zeros_like(bins)
    s1_s2_rdf = torch_kde_calc(torch.tensor(s1_s2_dist), bins, bandwidth) if len(s1_s2_dist) > 0 else torch.zeros_like(bins)
    
    return s1_s1_rdf, s2_s2_rdf, s1_s2_rdf

def multi_channel_rdf(species_list, atoms, bins, bandwidth, mic=False):
    """Calculate RDFs for all species pairs as separate channels."""
    n_species = len(species_list)
    n_pairs = (n_species * (n_species + 1)) // 2  # Number of unique pairs
    
    # Initialize output tensor with shape [n_pairs, n_bins]
    rdfs = torch.zeros((n_pairs, bins.size(0)), dtype=torch.float32)
    
    # Calculate RDFs for all pairs
    pair_idx = 0
    for i, species1 in enumerate(species_list):
        for j in range(i, n_species):
            species2 = species_list[j]
            if species1 == species2:
                # Same species
                symbols = atoms.get_chemical_symbols()
                indices = [k for k, s in enumerate(symbols) if s == species1]
                dist = compute_ase_idx_distances(atoms, indices, mic=mic)
                if len(dist) > 0:
                    rdfs[pair_idx] = torch_kde_calc(torch.tensor(dist), bins, bandwidth)
            else:
                # Different species
                s1_s1, s2_s2, s1_s2 = triple_kde_calc(species1, species2, atoms, bins, bandwidth, mic)
                rdfs[pair_idx] = s1_s2
            pair_idx += 1
            
    return rdfs

def concatenated_rdf(species_list, atoms, bins, bandwidth, mic=False):
    """Concatenate all RDFs into a single vector."""
    # Get multi-channel RDFs first
    multi_rdfs = multi_channel_rdf(species_list, atoms, bins, bandwidth, mic)
    
    # Flatten into a single vector
    return multi_rdfs.reshape(-1)

def weighted_attention_rdf(species_list, atoms, bins, bandwidth, mic=False):
    """Use composition-weighted attention for RDFs."""
    # Get atom symbols and count species
    symbols = atoms.get_chemical_symbols()
    total_atoms = len(symbols)
    n_species = len(species_list)  # Define n_species from species_list
    
    # Calculate species concentrations
    concentrations = {}
    for species in species_list:
        count = symbols.count(species)
        concentrations[species] = count / total_atoms if total_atoms > 0 else 0
    
    # Get multi-channel RDFs
    multi_rdfs = multi_channel_rdf(species_list, atoms, bins, bandwidth, mic)
    
    # Compute weighted sum based on pair concentrations
    pair_idx = 0
    weighted_rdf = torch.zeros_like(bins)
    
    for i, species1 in enumerate(species_list):
        for j in range(i, n_species):
            species2 = species_list[j]
            
            # Calculate pair weight (product of concentrations)
            if species1 == species2:
                weight = concentrations[species1] ** 2
            else:
                weight = 2 * concentrations[species1] * concentrations[species2]
                
            # Add weighted contribution
            weighted_rdf += weight * multi_rdfs[pair_idx]
            pair_idx += 1
            
    return weighted_rdf

def alloy_rdf(species_list, atoms, bins, bandwidth, approach="multi_channel", mic=False):
    """Calculate RDFs for alloys using the specified approach."""
    if approach == "multi_channel":
        return multi_channel_rdf(species_list, atoms, bins, bandwidth, mic)
    elif approach == "concatenated":
        return concatenated_rdf(species_list, atoms, bins, bandwidth, mic)
    elif approach == "weighted_attention":
        return weighted_attention_rdf(species_list, atoms, bins, bandwidth, mic)
    else:
        raise ValueError(f"Unknown approach: {approach}. Choose 'multi_channel', 'concatenated', or 'weighted_attention'.")

def single_kde_calc(Atoms, bins: torch.Tensor, bandwidth: torch.Tensor, mic=False) -> torch.Tensor:
    distances = compute_ase_all_distances(Atoms, mic=mic)
    if isinstance(distances, torch.Tensor) and distances.numel() == 0:
        return torch.zeros_like(bins)
    return torch_kde_calc(torch.Tensor(distances), bins, bandwidth)  # Return as a PyTorch tensor


def build_rdf(xyz_path, min_dist, max_dist, n_bins, bandwidth, output_path, device,
              mic=False, fraction=1.0, mode="single", species_list=None, approach="multi_channel"):
    """Builds RDFs (single, triple, or alloy) and saves them as images.
        For alloy mode: "multi_channel", "concatenated", or "weighted_attention"
    """
    # Load frames & labels
    frames = read(xyz_path, index=":")
    N = len(frames)
    bins = torch.linspace(min_dist, max_dist, n_bins)
    bw = torch.tensor([bandwidth], dtype=torch.float32)

    # Select subset of frames
    n_sub = int(N * fraction)
    perm = torch.randperm(N)[:n_sub] if fraction < 1.0 else torch.arange(n_sub)

    # Determine shape and RDF computation function based on mode
    if mode == "single":
        rdf_computation = lambda atoms: single_kde_calc(atoms, bins, bw, mic=mic)
        shape = [n_sub, 1, n_bins]
        
    elif mode == "triple":
        if not species_list or len(species_list) != 2:
            raise ValueError("Triple mode requires exactly 2 species in species_list")
        specie1, specie2 = species_list[:2]
        
        # For triple mode, we need to handle the tuple return value
        def triple_rdf_wrapper(atoms):
            s1_s1, s2_s2, s1_s2 = triple_kde_calc(specie1, specie2, atoms, bins, bw, mic=mic)
            return torch.stack([s1_s1, s2_s2, s1_s2])
            
        rdf_computation = triple_rdf_wrapper
        shape = [n_sub, 3, n_bins]  # 3 channels: s1-s1, s2-s2, s1-s2
        
    elif mode == "alloy":
        if not species_list:
            raise ValueError("Species list is required for alloy mode")
            
        # Shape depends on the approach
        if approach == "multi_channel":
            n_species = len(species_list)
            n_pairs = (n_species * (n_species + 1)) // 2
            shape = [n_sub, n_pairs, n_bins]
            rdf_computation = lambda atoms: multi_channel_rdf(species_list, atoms, bins, bw, mic=mic)
            
        elif approach == "concatenated":
            n_species = len(species_list)
            n_pairs = (n_species * (n_species + 1)) // 2
            shape = [n_sub, 1, n_pairs * n_bins]
            
            # Need to reshape the output to match expected shape
            def concat_rdf_wrapper(atoms):
                flat_rdf = concatenated_rdf(species_list, atoms, bins, bw, mic=mic)
                return flat_rdf.reshape(1, -1)  # Add channel dimension
                
            rdf_computation = concat_rdf_wrapper
            
        elif approach == "weighted_attention":
            shape = [n_sub, 1, n_bins]
            
            # Add channel dimension for weighted RDF
            def weighted_rdf_wrapper(atoms):
                rdf = weighted_attention_rdf(species_list, atoms, bins, bw, mic=mic)
                return rdf.reshape(1, -1)  # Add channel dimension
                
            rdf_computation = weighted_rdf_wrapper
            
        else:
            raise ValueError(f"Unknown approach: {approach}")
    else:
        raise ValueError("Invalid mode. Choose 'single', 'triple', or 'alloy'.")

    # Initialize output tensor with appropriate shape
    images = torch.empty(shape, dtype=torch.float32)

    # Process frames sequentially
    t0 = time.time()
    for out_i, frame_i in enumerate(tqdm(perm)):
        atoms = frames[frame_i]

        # compute RDF using the pre-defined logic
        rdf = rdf_computation(atoms)
        
        # Handle different output shapes
        if mode == "single":
            images[out_i, 0] = rdf.squeeze(0) if rdf.dim() > 1 else rdf
        elif mode == "triple":
            # Ensure rdf has the right shape [3, n_bins] to match images[out_i] with shape [3, n_bins]
            if rdf.dim() == 3 and rdf.size(1) == 1:  # If shape is [3, 1, n_bins]
                images[out_i] = rdf.squeeze(1)  # Squeeze to [3, n_bins]
            else:
                images[out_i] = rdf  # Already correct shape
        elif mode == "alloy":
            if approach == "multi_channel":
                images[out_i] = rdf
            elif approach == "concatenated" or approach == "weighted_attention":
                images[out_i, 0] = rdf.squeeze(0) if rdf.dim() > 1 else rdf  # Ensure correct shape

    elapsed = time.time() - t0
    logging.info(f"Computed {n_sub}/{N} frames in {elapsed:.2f}s ({elapsed/n_sub:.3f}s each)")

    # Save images with informative filename
    img_path = os.path.join(output_path, f"rdf_images_{mode}_{approach}.pt")
    torch.save(images, img_path)
    
    return images